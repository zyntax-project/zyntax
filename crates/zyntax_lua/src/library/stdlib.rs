//! The standard library a program reaches by name: the base functions
//! and the `string`, `math`, `table`, `os`, `io` and `coroutine`
//! libraries.
//!
//! Each function has a typed implementation the lowering calls
//! directly when it sees the name, described by a [`Builtin`] so the
//! lowering knows how to pass each argument; and a wrapper of the
//! shape every function value has, so `print` can be passed around and
//! `("x"):upper()` can be looked up. The library tables themselves are
//! built on first use, from the wrappers.

use super::*;
use zyntax_builtins::functions::VARIADIC_ARITY;

/// How a Lua argument reaches a typed implementation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Param {
    /// As the dynamic value it is, nil when absent.
    Any,
    /// Any value, including nil, but one must be passed.
    Value,
    /// A value the implementation checks itself; when absent, the
    /// error says what was expected and that nothing was passed.
    Expected(&'static str),
    /// An integer: a number with an integral value, or a numeral.
    Int,
    /// A float: any number or numeral.
    Float,
    /// A string: a string or a number.
    Str,
    /// The bytes to read of a string, a number, or a foreign object that
    /// is a buffer of bytes, read in place. A string by its static type
    /// goes to the typed implementation; anything else to its `$any`
    /// form (see [`any_form`]), which reads the subject either way.
    Bytes,
    /// A table.
    Table,
    /// An integer, or this when absent or nil.
    OptInt(i64),
    /// A string, or this when absent or nil.
    OptStr(&'static str),
    /// Every remaining argument, as a list of dynamic values.
    Rest,
}

/// What a typed implementation returns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ret {
    Unit,
    Bool,
    Int,
    Float,
    Str,
    Any,
    /// A dynamic value that may hold several values.
    Multi,
    Table,
}

pub struct Builtin {
    /// The library it belongs to; empty for the base library.
    pub lib: &'static str,
    pub name: &'static str,
    /// The typed implementation.
    pub func: &'static str,
    pub params: &'static [Param],
    pub ret: Ret,
}

use Param::*;

mod catalog;
pub use catalog::BUILTINS;

/// The constants a library table holds.
pub const CONSTANTS: &[(&str, &str, Constant)] = &[
    ("math", "pi", Constant::Float(std::f64::consts::PI)),
    ("math", "huge", Constant::Float(f64::INFINITY)),
    ("math", "maxinteger", Constant::Int(i64::MAX)),
    ("math", "mininteger", Constant::Int(i64::MIN)),
    // "[\0-\x7F\xC2-\xFD][\x80-\xBF]*"
    (
        "utf8",
        "charpattern",
        Constant::Bytes("5b002d7fc22dfd5d5b802dbf5d2a"),
    ),
    // "./?.lua;./?/init.lua"
    (
        "package",
        "path",
        Constant::Bytes("2e2f3f2e6c75613b2e2f3f2f696e69742e6c7561"),
    ),
    // The directory separator, path separator, template mark, and the
    // rest, each on its own line.
    ("package", "config", Constant::Bytes("2f0a3b0a3f0a210a2d0a")),
];

#[derive(Clone, Copy, Debug)]
pub enum Constant {
    Int(i64),
    Float(f64),
    /// A string, spelled in hex since it need not be UTF-8.
    Bytes(&'static str),
}

/// The value of the global `_VERSION`.
pub const LUA_VERSION: &str = "Lua 5.4";

/// The libraries with a table of their own.
pub const LIBS: &[&str] = &[
    "string",
    "math",
    "table",
    "os",
    "io",
    "coroutine",
    "utf8",
    "debug",
    "package",
];
/// Tables of functions no global names: a file's methods and its
/// metamethods.
pub const HIDDEN_LIBS: &[&str] = &["file", "filemeta"];

/// A builtin's number, the slot its one record as a value lives in.
pub fn builtin_index(b: &Builtin) -> usize {
    BUILTINS
        .iter()
        .position(|c| c.lib == b.lib && c.name == b.name)
        .expect("a builtin")
}

/// The name of a builtin's value wrapper.
/// The implementation of a builtin with a [`Param::Bytes`] parameter
/// that reads its subject as a string or a buffer.
pub fn any_form(func: &str) -> String {
    format!("{func}$any")
}

pub fn wrapper_name(b: &Builtin) -> String {
    if b.lib.is_empty() {
        format!("zl_v_{}", b.name)
    } else {
        format!("zl_v_{}_{}", b.lib, b.name)
    }
}

/// The function building (and caching) a library's table.
pub fn lib_table_fn(lib: &str) -> String {
    format!("zl_lib_{lib}")
}

pub fn ret_type(r: Ret, t: &Types) -> Type {
    match r {
        Ret::Unit => unit(),
        Ret::Bool => boolean(),
        Ret::Int => i64(),
        Ret::Float => f64(),
        Ret::Str => string(),
        Ret::Any | Ret::Multi => any(),
        Ret::Table => t.table(),
    }
}

mod base;
mod checks;
mod loading;
mod math;
mod package;
mod packing;
mod registry;
mod strings;
mod table;
mod wrappers;

struct Context {
    anys: Type,
    table: Type,
    x: Local,
    y: Local,
    s: Local,
    sep: Local,
    i: Local,
    j: Local,
    n: Local,
    k: Local,
    f: Local,
    g: Local,
    tb: Local,
    args: Local,
    out: Local,
    acc: Local,
    what: Local,
}

pub(super) fn declarations(_policy: &zyntax_builtins::Policy, t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let table = t.table();
    let x = kept("x", any());
    let y = kept("y", any());
    let s = kept("s", string());
    let sep = kept("sep", string());
    let i = local("i", i64());
    let j = local("j", i64());
    let n = local("n", i64());
    let k = local("k", i64());
    let f = local("f", f64());
    let g = local("g", f64());
    let tb = kept("t", table.clone());
    let args = kept("args", anys.clone());
    let out = borrowed("out", anys.clone());
    let acc = kept("acc", string());
    let what = kept("what", string());
    let mut d = Vec::new();
    let context = Context {
        anys,
        table,
        x,
        y,
        s,
        sep,
        i,
        j,
        n,
        k,
        f,
        g,
        tb,
        args,
        out,
        acc,
        what,
    };
    checks::append(&context, t, &mut d);
    base::append(&context, t, &mut d);
    strings::append(&context, t, &mut d);
    math::append(&context, t, &mut d);
    table::append(&context, t, &mut d);
    packing::append(&context, t, &mut d);
    wrappers::append(&context, t, &mut d);
    loading::append(&context, t, &mut d);
    package::append(&context, t, &mut d);
    registry::append(&context, t, &mut d);
    d
}

fn u64() -> Type {
    Type::Primitive(zyntax_typed_ast::PrimitiveType::U64)
}

mod os;
use os::os_declarations;
