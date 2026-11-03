//! What Lua defines above the shared built-in library, as typed AST:
//! tables, the arithmetic and comparison of dynamic values under Lua's
//! rules, calls through function values with Lua's argument adjustment,
//! coroutines over fibers, and the standard library a program reaches
//! by name. Every function here is prefixed `zl_`; the shared library's
//! are `zb_`.
//!
//! Included by the crate and by its build script, which lowers the
//! library once for the snapshot the crate carries.

mod layout;
pub use layout::{
    closing_marker, closing_tag, declare_reference_struct, declare_table_type, exit_code,
    exit_marker, file_tag, is_closing, is_exiting, light_tag, nil_error_tag,
    reference_struct_class, table_class, table_header_fields, table_tag, table_ty, thread_tag,
    userdata_tag,
};

mod expressions;
pub use expressions::{
    arg_type_name, arr_field, arr_of, at, bad_arg, bad_arg_at, box_bool, box_f64, box_i64, box_str,
    box_table, category, concat, get_bool, get_f64, get_i64, get_str, hash_field, hash_of, high_of,
    is_any_userdata, is_cat, is_file, is_func, is_int_cat, is_light, is_nil, is_nil_error,
    is_number_cat, is_shaped, is_table, is_thread, is_userdata, len, lua_error, meta_of,
    metamethod_call_check, metamethod_site, nil, node, obj_type_name, push, set_field, shape_of,
    slot_present, str_eq, tag_of, type_error, type_name, unbox_table,
};
use expressions::{nil_error, shape_hook_decls};

mod instance;
use instance::instance_hooks;

mod errors;
use errors::raising;
pub use errors::{global_var, pending, read_global, set_global, stripped_line};

mod assembly;
pub use assembly::{callee_names, library, reentrant_functions, strict_fp};

pub mod calls;
pub mod capi;
pub mod coroutines;
pub mod debug;
pub mod foreign;
pub mod gc;
pub mod io;
pub mod patterns;
pub mod stdlib;
pub mod tables;
pub mod utf8;
pub mod values;

use zyntax_builtins::build::*;
use zyntax_typed_ast::type_registry::{FieldDef, TypeMetadata};
use zyntax_typed_ast::typed_ast::{TypedAnnotation, TypedClass, TypedDeclaration, TypedField};
use zyntax_typed_ast::{Mutability, Type, TypeId, Visibility};

/// The name the table struct is registered under.
pub const TABLE_TYPE: &str = "LuaTable";

/// The instance kinds this frontend boxes: a table's address and a
/// coroutine's fiber, each under [`zyntax_builtins::instance_tag`].
pub const TABLE_KIND: usize = 0;
pub const THREAD_KIND: usize = 1;
/// `error(nil)` in flight: nil means nothing pending, so a nil error
/// value travels as this instance and is nil again when taken.
pub const NIL_ERROR_KIND: usize = 2;
/// A file: the handle of one of the host's streams.
pub const FILE_KIND: usize = 3;
/// The error in flight while a suspended coroutine is closed: its
/// `<close>` handlers see nil, and nothing catches it on the way up.
pub const CLOSING_KIND: usize = 4;
/// A weak table's key whose object the collector took: a tombstone no
/// lookup matches, under a nil value, until the table is compacted.
pub const DEAD_KEY_KIND: usize = 5;
/// A light userdata: its payload is the pointer word, which is its
/// identity for equality and as a table key. C code's light userdata
/// and `debug.upvalueid`'s identities are this kind.
pub const LIGHT_KIND: usize = 6;
/// A full userdata: the address of a block the C API lays out.
pub const USERDATA_KIND: usize = 7;

/// The types the library is written against: `List<Any>` and the
/// table struct.
#[derive(Clone)]
pub struct Types {
    pub list_type: TypeId,
    pub table_type: TypeId,
}

impl Types {
    pub fn anys(&self) -> Type {
        zyntax_builtins::list_of(self.list_type, any())
    }
    pub fn table(&self) -> Type {
        table_ty(self.table_type)
    }
    /// The hash part: the shared library's dict of dynamic keys and
    /// values.
    pub fn dict(&self) -> Type {
        zyntax_builtins::dicts::dict_type(self.list_type)
    }
}

/// A table is an array part, a hash part and a metatable. The array
/// part is a boxed list of the values at keys `1..=n`, never holding a
/// trailing nil, so its length is a border. The hash part is a boxed
/// dict (see the shared library's dicts) or null until the first key
/// outside the array. The metatable is a table pointer or null. `high`
/// is the longest the array part has been before a store of nil
/// shortened it: a key up to it may have been removed mid-traversal,
/// which `next` continues from; one past it is no key of the table.
/// `shape` is the number of the shape a table was made with, 0 for
/// none: a shaped table carries its constant-key fields in typed slots
/// after this header, laid out by the program that made it, and
/// `present` has a bit set for each slot holding a value. The program
/// defines the hooks the library reads and writes them through.
const TABLE_FIELDS: [&str; 6] = ["arr", "hash", "meta", "high", "shape", "present"];

// ─── shared helpers ─────────────────────────────────────────────────

/// The categories of the shared box, by number.
pub const BOOL: i64 = 1;
pub const INT: i64 = 2;
pub const UINT: i64 = 3;
pub const FLOAT: i64 = 4;
pub const STR: i64 = 5;
pub const CUSTOM: i64 = 255;

/// The hooks a program installs over its shapes, for the library's
/// generic paths: the slot a name has in a table's shape (-1 for
/// none), a slot's value boxed (nil when absent), a store into a slot
/// (0, or 1 when the value is not of the slot's kind), how many slots
/// a table's shape has, and a slot's name. Each is a function of the
/// program, reached through its code kept in a global: the library is
/// lowered before any program exists, and a chunk loaded later links
/// against neither. Installed by [`SHAPE_HOOKS_INSTALL`] before the
/// program runs; only a shaped table reaches them, and only a program
/// with shapes makes one.
pub const SHAPE_HOOKS: [&str; 5] = ["index", "load", "store", "count", "key"];
pub const SHAPE_HOOKS_INSTALL: &str = "zl_shape_hooks";

/// The error in flight, or nil. An error is raised by storing its value
/// here; every function that may raise leaves with a placeholder once
/// it sees the value set, until a `pcall` takes it.
pub const PENDING: &str = "zl_pending";
/// While `os.exit` closes the state, the error the last closing method
/// raised, which the next one receives, or nil.
pub const EXIT_ERROR: &str = "zl_exit_error";
/// Whether the pending error is a stack overflow: a message handler
/// cannot run on one, as the reference has no stack left for it.
pub const OVERFLOWED: &str = "zl_overflowed";
/// A message handler is running for a stack overflow: another
/// overflow while it runs is an error in error handling.
pub const HANDLING_OVERFLOW: &str = "zl_handling_overflow";
/// A library function is being entered by a method call through a
/// value (`obj:name()`): set by the call site, read and cleared by the
/// function's wrapper on entry, so its first argument is `self`.
pub const METHOD_CALL: &str = "zl_method_call";
/// How many times a message handler that raises is called again on
/// its own error before that is 'error in error handling'.
pub const HANDLER_RETRIES: i64 = 200;
/// The names of the chunks a program is made of besides the main one,
/// which `zl_chunk` names: a line stored in `zl_line` carries its
/// chunk's number in the bits above `LINE_BITS`, 0 for the main chunk
/// and `k` for the `k`th entry here.
pub const CHUNKS: &str = "zl_chunks";
/// The one empty array part every table without positional values
/// starts with, boxed; a write to a table's array part gives it one
/// of its own first (see `zl_arr_own`).
pub const ARR_EMPTY: &str = "zl_arr_empty";
/// How many program functions are on the stack; past `MAX_DEPTH` a
/// call is a stack overflow, as the reference's stack limit makes it.
/// A coroutine counts on from where it was resumed.
pub const DEPTH: &str = "zl_depth";
/// How many calls from the library into the program are in progress:
/// metamethods, iterators, comparators, what pcall protects, a
/// coroutine resumed. These nest on the reference's C stack, which
/// holds `CCALLS_LIMIT` of them.
pub const CCALLS: &str = "zl_ccalls";
pub const CCALLS_LIMIT: i64 = 200;
#[allow(dead_code)]
pub const MAX_DEPTH: i64 = 200_000;
pub const LINE_BITS: i64 = 32;
/// The line every statement of a chunk loaded without debug
/// information is on: positions show it as `-1`.
pub const STRIPPED_LINE: i64 = (1 << LINE_BITS) - 1;

/// The globals table, set before the chunk runs when the program
/// reaches its globals through one.
pub const GLOBALS: &str = "zl_G";
/// The line of the statement running, for the position a message
/// carries; zero outside any.
pub const LINE: &str = "zl_line";
/// The chunk's name, for the same.
pub const CHUNK: &str = "zl_chunk";
/// Which operand a pending type error is about, for the variable
/// description the site appends: `OPERAND_LEFT` or `OPERAND_RIGHT`,
/// with `VARINFO_INSIDE` when the description goes after the message's
/// first word rather than at its end, and `VARINFO_CALL` when the
/// error is a call's, which only a call site describes; zero for any
/// other error. Set by every raise, cleared by the check that
/// consumes it.
pub const VARINFO: &str = "zl_varinfo";
/// Set by the conversion of a `for` limit: whether the limit lies past
/// the integers on the side the step moves away from, so the loop has
/// no iteration whatever its start.
pub const FOR_SKIP: &str = "zl_for_skip";
pub const OPERAND_LEFT: i64 = 1;
pub const OPERAND_RIGHT: i64 = 2;
pub const VARINFO_INSIDE: i64 = 4;
pub const VARINFO_CALL: i64 = 8;

/// The library function giving an uncaught table error's text.
pub const ERROR_TEXT: &str = "zl_error_text";

#[cfg(test)]
mod tests {
    use super::expressions::hook_global;
    use super::*;

    /// Each library global names one storage cell: two declarations
    /// under one name would share it and overwrite each other. The
    /// shape hooks' globals are among those checked.
    #[test]
    fn library_global_names_are_distinct() {
        let (lib, _) = library(&crate::POLICY);
        let mut seen = std::collections::BTreeSet::new();
        let mut duplicates = Vec::new();
        for d in &lib.declarations {
            if let TypedDeclaration::Variable(v) = &d.node
                && let Some(name) = v.name.resolve_global()
                && !seen.insert(name.clone())
            {
                duplicates.push(name);
            }
        }
        assert!(
            duplicates.is_empty(),
            "library globals declared twice: {duplicates:?}"
        );
        for hook in SHAPE_HOOKS {
            let global = hook_global(hook);
            assert!(seen.contains(&global), "{global} is not declared");
        }
    }
}
