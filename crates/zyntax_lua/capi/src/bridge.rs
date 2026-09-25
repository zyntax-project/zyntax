//! The Lua semantics the C API reaches: a table of entry points into
//! the frontend's library.
//!
//! The API never reads a table, a shape or a dict itself. Each entry is
//! a function the frontend compiles into the running program on the
//! first successful open of a native library, over the same library
//! paths a Lua program takes, so an index, a call or a raise from C
//! behaves exactly as the same operation written in Lua. Signatures
//! use only pointers, 64-bit integers and doubles.

use crate::values::{Any, List, Str};

/// Bumped whenever an entry is added, removed, reordered or retyped.
pub const VERSION: u64 = 1;

/// A stored line carries its chunk's number above this many bits.
pub const LINE_BITS: i64 = 32;

/// Every entry, in field order, with the number of parameters it
/// takes: the frontend defines a function `lua$capi$<name>` for each.
pub const ENTRIES: [(&str, usize); 40] = [
    ("index", 2),
    ("setindex", 3),
    ("rawget", 2),
    ("rawset", 3),
    ("rawgeti", 2),
    ("rawseti", 3),
    ("next", 2),
    ("len", 1),
    ("rawlen", 1),
    ("concat", 1),
    ("arith", 3),
    ("compare", 3),
    ("rawequal", 2),
    ("getmetatable", 1),
    ("setmetatable", 2),
    ("call", 2),
    ("pcall", 3),
    ("raise", 1),
    ("take", 0),
    ("pending", 0),
    ("globals", 0),
    ("registry", 0),
    ("current", 0),
    ("main_thread", 0),
    ("thread_state", 1),
    ("set_thread_state", 2),
    ("func_new", 3),
    ("list_new", 0),
    ("list_push", 2),
    ("pack", 1),
    ("values", 1),
    ("new_table", 0),
    ("line", 0),
    ("set_line", 1),
    ("chunk_of", 1),
    ("shared", 0),
    ("number_str", 1),
    ("tonumber", 1),
    ("gc", 2),
    ("init", 0),
];

/// The frontend's name of an entry.
pub fn entry_name(entry: &str) -> String {
    format!("lua$capi${entry}")
}

/// The entry points, filled from the compiled functions' addresses in
/// [`ENTRIES`] order. Each field's comment says what the entry does;
/// "raises" means it may leave an error pending, which the caller
/// checks with `pending`.
#[repr(C)]
pub struct Bridge {
    /// `o[k]` with `__index` (raises).
    pub index: extern "C" fn(Any, Any) -> Any,
    /// `o[k] = v` with `__newindex` (raises).
    pub setindex: extern "C" fn(Any, Any, Any) -> i64,
    /// `rawget(t, k)` on a table.
    pub rawget: extern "C" fn(Any, Any) -> Any,
    /// `rawset(t, k, v)` on a table (raises on a nil or NaN key).
    pub rawset: extern "C" fn(Any, Any, Any) -> i64,
    /// `t[i]` raw, on a table.
    pub rawgeti: extern "C" fn(Any, i64) -> Any,
    /// `t[i] = v` raw, on a table.
    pub rawseti: extern "C" fn(Any, i64, Any) -> i64,
    /// `next(t, k)` on a table: `[k, v]`, or `[nil]` at the end
    /// (raises on a key the table does not hold).
    pub next: extern "C" fn(Any, Any) -> List,
    /// `#o` with `__len` (raises).
    pub len: extern "C" fn(Any) -> Any,
    /// The border of a table.
    pub rawlen: extern "C" fn(Any) -> i64,
    /// The values of a list joined with `..`, right to left (raises).
    pub concat: extern "C" fn(List) -> Any,
    /// `a op b` for the library's operator code `op`; unary minus and
    /// bitwise not read `a` alone (raises).
    pub arith: extern "C" fn(i64, Any, Any) -> Any,
    /// `a == b`, `a < b` or `a <= b` for `op` 0, 1 or 2 (raises).
    pub compare: extern "C" fn(Any, Any, i64) -> i64,
    /// Equality without metamethods.
    pub rawequal: extern "C" fn(Any, Any) -> i64,
    /// The metatable of a value that is not a full userdata, or nil.
    pub getmetatable: extern "C" fn(Any) -> Any,
    /// Set the metatable of a value that is not a full userdata.
    pub setmetatable: extern "C" fn(Any, Any) -> i64,
    /// Call `f` with a list of arguments: its results as one value
    /// (raises).
    pub call: extern "C" fn(Any, List) -> Any,
    /// Call `f` protected, `h` the message handler or nil: `pcall`'s or
    /// `xpcall`'s tuple; nil while a coroutine is being closed, which
    /// the caller passes on.
    pub pcall: extern "C" fn(Any, List, Any) -> Any,
    /// Raise `v`: the first error pending stands.
    pub raise: extern "C" fn(Any) -> i64,
    /// Take the pending error: nothing is pending afterwards.
    pub take: extern "C" fn() -> Any,
    /// The pending error, or nil.
    pub pending: extern "C" fn() -> Any,
    /// The globals table.
    pub globals: extern "C" fn() -> Any,
    /// The registry.
    pub registry: extern "C" fn() -> Any,
    /// The running coroutine, nil on the main thread.
    pub current: extern "C" fn() -> Any,
    /// The main thread as a value.
    pub main_thread: extern "C" fn() -> Any,
    /// The State a coroutine's record keeps, 0 for none.
    pub thread_state: extern "C" fn(Any) -> i64,
    /// Keep a boxed State in a coroutine's record.
    pub set_thread_state: extern "C" fn(Any, Any) -> i64,
    /// A function value whose code is `code`, calling C function
    /// `cfn`, with the upvalues in a list.
    pub func_new: extern "C" fn(i64, i64, List) -> Any,
    /// An empty list.
    pub list_new: extern "C" fn() -> List,
    /// Append to a list.
    pub list_push: extern "C" fn(List, Any) -> i64,
    /// A list's values as one value: exactly one is itself.
    pub pack: extern "C" fn(List) -> Any,
    /// One value's values as a list.
    pub values: extern "C" fn(Any) -> List,
    /// A new empty table.
    pub new_table: extern "C" fn() -> Any,
    /// The line of the statement running, chunk number above the line
    /// bits; 0 outside any.
    pub line: extern "C" fn() -> i64,
    /// Set the line the next raise is positioned at.
    pub set_line: extern "C" fn(i64) -> i64,
    /// The name of the chunk a stored line belongs to.
    pub chunk_of: extern "C" fn(i64) -> Str,
    /// Whether the program keeps its globals in the globals table.
    pub shared: extern "C" fn() -> i64,
    /// A number's text as `tostring` gives it.
    pub number_str: extern "C" fn(Any) -> Str,
    /// A number, or a numeral string's value, or nil.
    pub tonumber: extern "C" fn(Any) -> Any,
    /// `collectgarbage` for option `op` with argument `arg`.
    pub gc: extern "C" fn(i64, i64) -> i64,
    /// Seed the registry and return it.
    pub init: extern "C" fn() -> Any,
}

/// What the frontend hands over: the version it was written against,
/// and the compiled entries' addresses in [`ENTRIES`] order.
pub struct Resolved {
    pub version: u64,
    pub entries: Vec<*const u8>,
}

impl Bridge {
    /// The table over `resolved`, refused when it was written for
    /// another version or has another number of entries.
    pub fn from_resolved(resolved: &Resolved) -> Result<Bridge, String> {
        if resolved.version != VERSION {
            return Err(format!(
                "the C API bridge is version {}, the library's is {}",
                VERSION, resolved.version
            ));
        }
        if resolved.entries.len() != ENTRIES.len()
            || std::mem::size_of::<Bridge>() != ENTRIES.len() * std::mem::size_of::<usize>()
        {
            return Err(format!(
                "the C API bridge has {} entries, the library gave {}",
                ENTRIES.len(),
                resolved.entries.len()
            ));
        }
        if let Some(i) = resolved.entries.iter().position(|p| p.is_null()) {
            return Err(format!(
                "the library did not compile {}",
                entry_name(ENTRIES[i].0)
            ));
        }
        let mut words = [std::ptr::null::<u8>(); ENTRIES.len()];
        words.copy_from_slice(&resolved.entries);
        // SAFETY: `Bridge` is `ENTRIES.len()` function pointers, checked
        // above, each compiled by the frontend with the signature its
        // field gives.
        Ok(unsafe { std::mem::transmute::<[*const u8; ENTRIES.len()], Bridge>(words) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_is_a_field() {
        assert_eq!(
            std::mem::size_of::<Bridge>(),
            ENTRIES.len() * std::mem::size_of::<usize>()
        );
        let mut names: Vec<&str> = ENTRIES.iter().map(|(n, _)| *n).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), ENTRIES.len(), "entry names are distinct");
    }

    #[test]
    fn another_version_is_refused() {
        let resolved = Resolved {
            version: VERSION + 1,
            entries: vec![1 as *const u8; ENTRIES.len()],
        };
        assert!(Bridge::from_resolved(&resolved).is_err());
        let short = Resolved {
            version: VERSION,
            entries: vec![1 as *const u8; ENTRIES.len() - 1],
        };
        assert!(Bridge::from_resolved(&short).is_err());
    }
}
