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

pub const BUILTINS: &[Builtin] = &[
    // ─── base ───
    Builtin {
        lib: "",
        name: "print",
        func: "zl_print",
        params: &[Rest],
        ret: Ret::Unit,
    },
    Builtin {
        lib: "",
        name: "type",
        func: "zl_type",
        params: &[Value],
        ret: Ret::Str,
    },
    Builtin {
        lib: "",
        name: "tostring",
        func: "zl_tostring",
        params: &[Value],
        ret: Ret::Str,
    },
    Builtin {
        lib: "",
        name: "tonumber",
        func: "zl_tonumber_of",
        params: &[Value, Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "",
        name: "next",
        func: "zl_next_of",
        params: &[Table, Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "pairs",
        func: "zl_pairs",
        params: &[Value],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "ipairs",
        func: "zl_ipairs",
        params: &[Value],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "select",
        func: "zl_select",
        params: &[Expected("number"), Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "rawget",
        func: "zl_rawget",
        params: &[Table, Value],
        ret: Ret::Any,
    },
    Builtin {
        lib: "",
        name: "rawset",
        func: "zl_rawset_of",
        params: &[Table, Value, Value],
        ret: Ret::Table,
    },
    Builtin {
        lib: "",
        name: "rawequal",
        func: "zl_rawequal",
        params: &[Value, Value],
        ret: Ret::Bool,
    },
    Builtin {
        lib: "",
        name: "rawlen",
        func: "zl_rawlen",
        params: &[Expected("table or string")],
        ret: Ret::Int,
    },
    Builtin {
        lib: "",
        name: "setmetatable",
        func: "zl_setmetatable",
        params: &[Table, Any],
        ret: Ret::Table,
    },
    Builtin {
        lib: "",
        name: "getmetatable",
        func: "zl_getmetatable",
        params: &[Value],
        ret: Ret::Any,
    },
    Builtin {
        lib: "",
        name: "assert",
        func: "zl_assert",
        params: &[Value, Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "error",
        func: "zl_error",
        params: &[Any, OptInt(1)],
        ret: Ret::Unit,
    },
    Builtin {
        lib: "",
        name: "pcall",
        func: "zl_pcall",
        params: &[Value, Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "xpcall",
        func: "zl_xpcall",
        params: &[Any, Expected("function"), Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "unpack",
        func: "zl_unpack",
        params: &[Table, OptInt(1), OptInt(i64::MIN)],
        ret: Ret::Multi,
    },
    // ─── string ───
    Builtin {
        lib: "string",
        name: "len",
        func: "zl_string_len",
        params: &[Str],
        ret: Ret::Int,
    },
    Builtin {
        lib: "string",
        name: "sub",
        func: "zl_string_sub",
        params: &[Str, OptInt(1), OptInt(-1)],
        ret: Ret::Str,
    },
    Builtin {
        lib: "string",
        name: "upper",
        func: "zl_string_upper",
        params: &[Str],
        ret: Ret::Str,
    },
    Builtin {
        lib: "string",
        name: "lower",
        func: "zl_string_lower",
        params: &[Str],
        ret: Ret::Str,
    },
    Builtin {
        lib: "string",
        name: "rep",
        func: "zl_string_rep_of",
        params: &[Str, Int, OptStr("")],
        ret: Ret::Str,
    },
    Builtin {
        lib: "string",
        name: "reverse",
        func: "zl_string_reverse",
        params: &[Str],
        ret: Ret::Str,
    },
    Builtin {
        lib: "string",
        name: "byte",
        func: "zl_string_byte",
        params: &[Str, OptInt(1), OptInt(i64::MIN)],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "string",
        name: "char",
        func: "zl_string_char",
        params: &[Rest],
        ret: Ret::Str,
    },
    Builtin {
        lib: "string",
        name: "pack",
        func: "zl_string_pack",
        params: &[Str, Rest],
        ret: Ret::Str,
    },
    Builtin {
        lib: "string",
        name: "packsize",
        func: "zl_string_packsize",
        params: &[Str],
        ret: Ret::Int,
    },
    Builtin {
        lib: "string",
        name: "unpack",
        func: "zl_string_unpack",
        params: &[Str, Str, OptInt(1)],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "string",
        name: "format",
        func: "zl_string_format",
        params: &[Str, Rest],
        ret: Ret::Str,
    },
    Builtin {
        lib: "string",
        name: "find",
        func: "zl_string_find",
        params: &[Str, Str, OptInt(1), Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "string",
        name: "match",
        func: "zl_string_match",
        params: &[Str, Str, OptInt(1)],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "string",
        name: "gmatch",
        func: "zl_string_gmatch",
        params: &[Str, Str, OptInt(1)],
        ret: Ret::Any,
    },
    Builtin {
        lib: "string",
        name: "gsub",
        func: "zl_string_gsub",
        params: &[Str, Str, Value, OptInt(i64::MAX)],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "load",
        func: "zl_load",
        params: &[Expected("function"), Any, Any, Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "loadfile",
        func: "zl_loadfile",
        params: &[Str, Any, Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "dofile",
        func: "zl_dofile",
        params: &[Str],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "collectgarbage",
        func: "zl_collectgarbage",
        params: &[OptStr("collect"), Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "",
        name: "require",
        func: "zl_require",
        params: &[Str],
        ret: Ret::Any,
    },
    Builtin {
        lib: "package",
        name: "searchpath",
        func: "zl_package_searchpath",
        params: &[Str, Str, OptStr("."), OptStr("/")],
        ret: Ret::Multi,
    },
    // ─── debug: what a program can be told without a debugger ───
    Builtin {
        lib: "debug",
        name: "traceback",
        func: "zl_debug_traceback",
        params: &[Any, Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "debug",
        name: "getinfo",
        func: "zl_debug_getinfo",
        params: &[Any, Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "debug",
        name: "sethook",
        func: "zl_debug_sethook",
        params: &[Rest],
        ret: Ret::Unit,
    },
    Builtin {
        lib: "debug",
        name: "gethook",
        func: "zl_debug_gethook",
        params: &[Rest],
        ret: Ret::Any,
    },
    Builtin {
        lib: "debug",
        name: "getmetatable",
        func: "zl_debug_getmetatable",
        params: &[Value],
        ret: Ret::Any,
    },
    Builtin {
        lib: "debug",
        name: "setmetatable",
        func: "zl_debug_setmetatable",
        params: &[Value, Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "debug",
        name: "getregistry",
        func: "zl_debug_getregistry",
        params: &[],
        ret: Ret::Any,
    },
    Builtin {
        lib: "debug",
        name: "getlocal",
        func: "zl_debug_none",
        params: &[Rest],
        ret: Ret::Any,
    },
    Builtin {
        lib: "debug",
        name: "getupvalue",
        func: "zl_debug_none",
        params: &[Rest],
        ret: Ret::Any,
    },
    Builtin {
        lib: "debug",
        name: "upvalueid",
        func: "zl_debug_none",
        params: &[Rest],
        ret: Ret::Any,
    },
    // ─── utf8 ───
    Builtin {
        lib: "utf8",
        name: "char",
        func: "zl_utf8_char",
        params: &[Rest],
        ret: Ret::Str,
    },
    Builtin {
        lib: "utf8",
        name: "len",
        func: "zl_utf8_len",
        params: &[Str, OptInt(1), OptInt(-1), Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "utf8",
        name: "offset",
        func: "zl_utf8_offset",
        params: &[Str, Int, Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "utf8",
        name: "codepoint",
        func: "zl_utf8_codepoint",
        params: &[Str, OptInt(1), Any, Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "utf8",
        name: "codes",
        func: "zl_utf8_codes",
        params: &[Str, Any],
        ret: Ret::Multi,
    },
    // ─── math ───
    Builtin {
        lib: "math",
        name: "floor",
        func: "zl_math_floor",
        params: &[Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "math",
        name: "ceil",
        func: "zl_math_ceil",
        params: &[Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "math",
        name: "abs",
        func: "zl_math_abs",
        params: &[Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "math",
        name: "sqrt",
        func: "zl_math_sqrt",
        params: &[Float],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "deg",
        func: "zl_math_deg",
        params: &[Float],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "rad",
        func: "zl_math_rad",
        params: &[Float],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "sin",
        func: "zl_math_sin",
        params: &[Float],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "cos",
        func: "zl_math_cos",
        params: &[Float],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "tan",
        func: "zl_math_tan",
        params: &[Float],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "asin",
        func: "zl_math_asin",
        params: &[Float],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "acos",
        func: "zl_math_acos",
        params: &[Float],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "atan",
        func: "zl_math_atan",
        params: &[Float, Any],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "exp",
        func: "zl_math_exp",
        params: &[Float],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "log",
        func: "zl_math_log",
        params: &[Float, Any],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "pow",
        func: "zl_pow",
        params: &[Float, Float],
        ret: Ret::Float,
    },
    Builtin {
        lib: "math",
        name: "fmod",
        func: "zl_math_fmod",
        params: &[Any, Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "math",
        name: "modf",
        func: "zl_math_modf",
        params: &[Float],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "math",
        name: "max",
        func: "zl_math_max",
        params: &[Rest],
        ret: Ret::Any,
    },
    Builtin {
        lib: "math",
        name: "min",
        func: "zl_math_min",
        params: &[Rest],
        ret: Ret::Any,
    },
    Builtin {
        lib: "math",
        name: "tointeger",
        func: "zl_math_tointeger",
        params: &[Value],
        ret: Ret::Any,
    },
    Builtin {
        lib: "math",
        name: "type",
        func: "zl_math_type",
        params: &[Value],
        ret: Ret::Any,
    },
    Builtin {
        lib: "math",
        name: "random",
        func: "zl_math_random",
        params: &[Rest],
        ret: Ret::Any,
    },
    Builtin {
        lib: "math",
        name: "randomseed",
        func: "zl_math_randomseed",
        params: &[Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "math",
        name: "ult",
        func: "zl_math_ult",
        params: &[Int, Int],
        ret: Ret::Bool,
    },
    // ─── table ───
    Builtin {
        lib: "table",
        name: "insert",
        func: "zl_table_insert",
        params: &[Table, Rest],
        ret: Ret::Unit,
    },
    Builtin {
        lib: "table",
        name: "remove",
        func: "zl_table_remove",
        params: &[Table, OptInt(i64::MIN)],
        ret: Ret::Any,
    },
    Builtin {
        lib: "table",
        name: "concat",
        func: "zl_table_concat",
        params: &[Table, OptStr(""), OptInt(1), Any],
        ret: Ret::Str,
    },
    Builtin {
        lib: "table",
        name: "unpack",
        func: "zl_unpack",
        params: &[Table, OptInt(1), OptInt(i64::MIN)],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "table",
        name: "pack",
        func: "zl_table_pack",
        params: &[Rest],
        ret: Ret::Table,
    },
    Builtin {
        lib: "table",
        name: "sort",
        func: "zl_table_sort",
        params: &[Table, Any],
        ret: Ret::Unit,
    },
    // ─── os ───
    Builtin {
        lib: "os",
        name: "clock",
        func: "zl_os_clock",
        params: &[],
        ret: Ret::Float,
    },
    Builtin {
        lib: "os",
        name: "time",
        func: "zl_os_time_of",
        params: &[Any],
        ret: Ret::Int,
    },
    Builtin {
        lib: "os",
        name: "date",
        func: "zl_os_date",
        params: &[Any, Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "os",
        name: "difftime",
        func: "zl_os_difftime",
        params: &[Int, Int],
        ret: Ret::Float,
    },
    Builtin {
        lib: "os",
        name: "getenv",
        func: "zl_os_getenv",
        params: &[Str],
        ret: Ret::Any,
    },
    Builtin {
        lib: "os",
        name: "tmpname",
        func: "zl_os_tmpname",
        params: &[],
        ret: Ret::Str,
    },
    Builtin {
        lib: "os",
        name: "remove",
        func: "zl_os_remove",
        params: &[Str],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "os",
        name: "rename",
        func: "zl_os_rename",
        params: &[Str, Str],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "os",
        name: "execute",
        func: "zl_os_execute",
        params: &[Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "os",
        name: "setlocale",
        func: "zl_os_setlocale",
        params: &[Any, Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "os",
        name: "exit",
        func: "zl_os_exit",
        params: &[Any],
        ret: Ret::Unit,
    },
    // ─── io ───
    Builtin {
        lib: "io",
        name: "write",
        func: "zl_io_write_of",
        params: &[Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "io",
        name: "read",
        func: "zl_io_read",
        params: &[Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "io",
        name: "lines",
        func: "zl_io_lines",
        params: &[Any, Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "io",
        name: "open",
        func: "zl_io_open_of",
        params: &[Str, OptStr("r")],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "io",
        name: "close",
        func: "zl_io_close_of",
        params: &[Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "io",
        name: "input",
        func: "zl_io_input",
        params: &[Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "io",
        name: "output",
        func: "zl_io_output",
        params: &[Any],
        ret: Ret::Any,
    },
    Builtin {
        lib: "io",
        name: "type",
        func: "zl_io_type",
        params: &[Value],
        ret: Ret::Any,
    },
    Builtin {
        lib: "io",
        name: "popen",
        func: "zl_io_popen_of",
        params: &[Str, OptStr("r")],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "io",
        name: "tmpfile",
        func: "zl_io_tmpfile_of",
        params: &[],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "io",
        name: "flush",
        func: "zl_io_flush_of",
        params: &[],
        ret: Ret::Multi,
    },
    // ─── files: the methods, then the metamethods ───
    Builtin {
        lib: "file",
        name: "read",
        func: "zl_file_read",
        params: &[Expected("FILE*"), Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "file",
        name: "write",
        func: "zl_file_write",
        params: &[Expected("FILE*"), Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "file",
        name: "lines",
        func: "zl_file_lines_of",
        params: &[Expected("FILE*"), Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "file",
        name: "close",
        func: "zl_file_close",
        params: &[Expected("FILE*")],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "file",
        name: "seek",
        func: "zl_file_seek",
        params: &[Expected("FILE*"), Any, Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "file",
        name: "setvbuf",
        func: "zl_file_setvbuf",
        params: &[Expected("FILE*"), Any, Any],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "file",
        name: "flush",
        func: "zl_file_flush",
        params: &[Expected("FILE*")],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "filemeta",
        name: "__gc",
        func: "zl_file_release",
        params: &[Expected("FILE*")],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "filemeta",
        name: "__close",
        func: "zl_file_release",
        params: &[Expected("FILE*")],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "filemeta",
        name: "__tostring",
        func: "zl_file_tostring",
        params: &[Expected("FILE*")],
        ret: Ret::Str,
    },
    // ─── coroutine ───
    Builtin {
        lib: "coroutine",
        name: "create",
        func: "zl_co_create_of",
        params: &[Expected("function")],
        ret: Ret::Any,
    },
    Builtin {
        lib: "coroutine",
        name: "resume",
        func: "zl_co_resume",
        params: &[Expected("thread"), Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "coroutine",
        name: "yield",
        func: "zl_co_yield",
        params: &[Rest],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "coroutine",
        name: "status",
        func: "zl_co_status",
        params: &[Expected("thread")],
        ret: Ret::Str,
    },
    Builtin {
        lib: "coroutine",
        name: "wrap",
        func: "zl_co_wrap",
        params: &[Expected("function")],
        ret: Ret::Any,
    },
    Builtin {
        lib: "coroutine",
        name: "running",
        func: "zl_co_running",
        params: &[],
        ret: Ret::Multi,
    },
    Builtin {
        lib: "coroutine",
        name: "isyieldable",
        func: "zl_co_isyieldable",
        params: &[Any],
        ret: Ret::Bool,
    },
    Builtin {
        lib: "coroutine",
        name: "close",
        func: "zl_co_close",
        params: &[Expected("thread")],
        ret: Ret::Any,
    },
];

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
    ("package", "cpath", Constant::Bytes("")),
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

    // ─── argument checks ────────────────────────────────────────
    // The message's start for the `i`th of a function's values, counted
    // from zero.
    let bad_arg_at = |i: Expr, name: &str| {
        concat(vec![
            text("bad argument #"),
            call("zb_str_of_int", vec![add(i, int(1))], string()),
            text(&format!(" to '{name}'")),
        ])
    };
    // `what` starts the message: `bad argument #n to 'f'`.
    let arg_error = |what: &Local, expected: &str, got: Expr| {
        lua_error(concat(vec![
            what.e(),
            text(&format!(" ({expected} expected, got ")),
            got,
            text(")"),
        ]))
    };
    let got_of = |x: &Local| type_name(x.e());
    d.push(define(
        "zl_arg_int",
        &[&x, &what],
        i64(),
        vec![
            y.decl(call("zl_arith_operand", vec![x.e()], any())),
            when(is_nil(y.e()), vec![arg_error(&what, "number", got_of(&x))]),
            when(
                or(
                    eq(category(y.e()), int(INT)),
                    eq(category(y.e()), int(UINT)),
                ),
                vec![ret(get_i64(y.e()))],
            ),
            f.decl(get_f64(y.e())),
            when(
                not(call("zl_float_is_int", vec![f.e()], boolean())),
                vec![lua_error(add(
                    what.e(),
                    text(" (number has no integer representation)"),
                ))],
            ),
            ret(cast(f.e(), i64())),
        ],
    ));
    d.push(define(
        "zl_arg_float",
        &[&x, &what],
        f64(),
        vec![
            y.decl(call("zl_arith_operand", vec![x.e()], any())),
            when(is_nil(y.e()), vec![arg_error(&what, "number", got_of(&x))]),
            when(
                or(
                    eq(category(y.e()), int(INT)),
                    eq(category(y.e()), int(UINT)),
                ),
                vec![ret(cast(get_i64(y.e()), f64()))],
            ),
            ret(get_f64(y.e())),
        ],
    ));
    d.push(define(
        "zl_arg_str",
        &[&x, &what],
        string(),
        vec![
            when(is_nil(x.e()), vec![arg_error(&what, "string", got_of(&x))]),
            when(eq(category(x.e()), int(STR)), vec![ret(get_str(x.e()))]),
            when(
                or(
                    or(
                        eq(category(x.e()), int(INT)),
                        eq(category(x.e()), int(UINT)),
                    ),
                    eq(category(x.e()), int(FLOAT)),
                ),
                vec![ret(call("zl_number_str", vec![x.e()], string()))],
            ),
            arg_error(&what, "string", got_of(&x)),
            ret(text("")),
        ],
    ));
    // An optional integer: the default when absent.
    let default = local("default", i64());
    d.push(define(
        "zl_arg_opt_int",
        &[&x, &default, &what],
        i64(),
        vec![
            when(is_nil(x.e()), vec![ret(default.e())]),
            ret(call("zl_arg_int", vec![x.e(), what.e()], i64())),
        ],
    ));
    let default_s = kept("default", string());
    d.push(define(
        "zl_arg_opt_str",
        &[&x, &default_s, &what],
        string(),
        vec![
            when(is_nil(x.e()), vec![ret(default_s.e())]),
            ret(call("zl_arg_str", vec![x.e(), what.e()], string())),
        ],
    ));
    // An argument that was not passed at all, which the caller can
    // tell from nil: the error, then a stand-in.
    let missing = |what: &Local, expected: &str| {
        lua_error(add(
            what.e(),
            text(&format!(" ({expected} expected, got no value)")),
        ))
    };
    d.push(define_cold(
        "zl_arg_value_missing",
        &[&what],
        any(),
        vec![
            lua_error(add(what.e(), text(" (value expected)"))),
            ret(nil()),
        ],
    ));
    let kind = kept("kind", string());
    d.push(define_cold(
        "zl_arg_expected_missing",
        &[&what, &kind],
        any(),
        vec![
            lua_error(concat(vec![
                what.e(),
                text(" ("),
                kind.e(),
                text(" expected, got no value)"),
            ])),
            ret(nil()),
        ],
    ));
    d.push(define_cold(
        "zl_arg_int_missing",
        &[&what],
        i64(),
        vec![missing(&what, "number"), ret(int(0))],
    ));
    d.push(define_cold(
        "zl_arg_float_missing",
        &[&what],
        f64(),
        vec![missing(&what, "number"), ret(float(0.0))],
    ));
    d.push(define_cold(
        "zl_arg_str_missing",
        &[&what],
        string(),
        vec![missing(&what, "string"), ret(text(""))],
    ));
    d.push(define_cold(
        "zl_as_table_missing",
        &[&what],
        table.clone(),
        vec![
            missing(&what, "table"),
            ret(call("zl_table_new", vec![], table.clone())),
        ],
    ));

    // ─── base ───────────────────────────────────────────────────
    d.push(define(
        "zl_print",
        &[&args],
        unit(),
        vec![
            acc.decl(text("")),
            n.decl(len(args.e())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    when(gt(i.e(), int(0)), vec![acc.set(add(acc.e(), text("\t")))]),
                    acc.set(add(
                        acc.e(),
                        call("zl_tostring", vec![at(args.e(), i.e())], string()),
                    )),
                    i.add_assign(int(1)),
                ],
            ),
            // Written as bytes through the standard stream, as `io.write`
            // writes: a string is whatever bytes it holds.
            expr(call(
                "zl_io_write",
                vec![
                    call("zl_io_std", vec![int(1)], i64()),
                    add(acc.e(), text("\n")),
                ],
                i64(),
            )),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_type",
        &[&x],
        string(),
        vec![
            when(is_nil(x.e()), vec![ret(text("nil"))]),
            when(
                or(
                    eq(tag_of(x.e()), int(zyntax_builtins::FUNC_TAG)),
                    eq(tag_of(x.e()), int(zyntax_builtins::CODE_TAG)),
                ),
                vec![ret(text("function"))],
            ),
            when(is_file(x.e()), vec![ret(text("userdata"))]),
            ret(type_name(x.e())),
        ],
    ));
    d.push(define(
        "zl_tonumber_of",
        &[&x, &y],
        any(),
        vec![
            when(
                is_nil(y.e()),
                vec![ret(call("zl_tonumber", vec![x.e()], any()))],
            ),
            when(
                or(is_nil(x.e()), ne(category(x.e()), int(STR))),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'tonumber' (string expected, got "),
                    type_name(x.e()),
                    text(")"),
                ]))],
            ),
            ret(call(
                "zl_tonumber_base",
                vec![
                    get_str(x.e()),
                    call("zl_arg_int", vec![y.e(), bad_arg(2, "tonumber")], i64()),
                ],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_next_of",
        &[&tb, &x],
        any(),
        vec![ret(call(
            "zb_box_tuple",
            vec![call("zl_next", vec![tb.e(), x.e()], anys.clone())],
            any(),
        ))],
    ));
    d.push(define(
        "zl_rawset_of",
        &[&tb, &x, &y],
        table.clone(),
        vec![
            expr(call("zl_rawset", vec![tb.e(), x.e(), y.e()], unit())),
            ret(tb.e()),
        ],
    ));
    // `next` and `ipairs`'s iterator as function values, for `pairs`
    // and `ipairs` used as values rather than in a `for`.
    let env = borrowed("env", anys.clone());
    let a0 = local("a0", any());
    let a1 = local("a1", any());
    d.push(define(
        "zl_next_code",
        &[&env, &a0, &a1],
        any(),
        vec![ret(call(
            "zl_next_of",
            vec![
                call(
                    "zl_as_table",
                    vec![a0.e(), bad_arg(1, "next")],
                    table.clone(),
                ),
                a1.e(),
            ],
            any(),
        ))],
    ));
    d.push(define(
        "zl_ipairs_code",
        &[&env, &a0, &a1],
        any(),
        vec![
            i.decl(add(
                call(
                    "zl_arg_int",
                    vec![a1.e(), bad_arg(2, "for iterator")],
                    i64(),
                ),
                int(1),
            )),
            y.decl(call("zl_geti", vec![a0.e(), i.e()], any())),
            when(
                is_nil(y.e()),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(vec![], anys.clone())],
                    any(),
                ))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![box_i64(i.e()), y.e()], anys.clone())],
                any(),
            )),
        ],
    ));
    let code_of = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            usize(),
        )
    };
    // A builtin as a value is one record, whatever names it: the
    // reference compares functions by identity.
    d.push(global_var("zl_builtin_values", any()));
    let index = local("index", i64());
    let code = local("code", usize());
    let arity = local("arity", i64());
    d.push(define(
        "zl_builtin_value",
        &[&index, &code, &arity],
        any(),
        vec![
            when(
                is_nil(read_global("zl_builtin_values", any())),
                vec![set_global(
                    "zl_builtin_values",
                    box_table(call("zl_table_new", vec![], table.clone())),
                )],
            ),
            tb.decl(unbox_table(read_global("zl_builtin_values", any()), t)),
            y.decl(call("zl_rawgeti", vec![tb.e(), index.e()], any())),
            when(not(is_nil(y.e())), vec![ret(y.e())]),
            y.set(call("zl_func_of", vec![code.e(), arity.e()], any())),
            expr(call("zl_rawseti", vec![tb.e(), index.e(), y.e()], unit())),
            ret(y.e()),
        ],
    ));
    let builtin_value = |b: &Builtin| {
        let index = builtin_index(b);
        call(
            "zl_builtin_value",
            vec![
                int(index as i64),
                code_of(&wrapper_name(b)),
                int(VARIADIC_ARITY),
            ],
            any(),
        )
    };
    // The iterators `pairs` and `ipairs` give: `next` itself, and one
    // record for `ipairs`, past the builtins' slots.
    let next_value = || {
        let b = BUILTINS
            .iter()
            .find(|b| b.lib.is_empty() && b.name == "next")
            .expect("next");
        builtin_value(b)
    };
    let ipairs_value = || {
        call(
            "zl_builtin_value",
            vec![
                int(BUILTINS.len() as i64),
                code_of("zl_ipairs_code"),
                int(2),
            ],
            any(),
        )
    };
    d.push(define(
        "zl_pairs",
        &[&x],
        any(),
        vec![
            y.decl(call("zl_meta_of", vec![x.e(), text("__pairs")], any())),
            when(
                not(is_nil(y.e())),
                vec![ret(call("zl_call_1", vec![y.e(), x.e()], any()))],
            ),
            when(
                not(is_table(x.e())),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'for iterator' (table expected, got "),
                    type_name(x.e()),
                    text(")"),
                ]))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![next_value(), x.e(), nil()], anys.clone())],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_ipairs",
        &[&x],
        any(),
        vec![ret(call(
            "zb_box_tuple",
            vec![list(
                vec![ipairs_value(), x.e(), box_i64(int(0))],
                anys.clone(),
            )],
            any(),
        ))],
    ));
    d.push(define(
        "zl_select",
        &[&x, &args],
        any(),
        vec![
            n.decl(len(args.e())),
            when(
                and(
                    eq(category(x.e()), int(STR)),
                    str_eq(get_str(x.e()), text("#")),
                ),
                vec![ret(box_i64(n.e()))],
            ),
            i.decl(call("zl_arg_int", vec![x.e(), bad_arg(1, "select")], i64())),
            when(
                lt(i.e(), int(0)),
                vec![i.set(add(add(n.e(), i.e()), int(1)))],
            ),
            when(
                lt(i.e(), int(1)),
                vec![lua_error(text(
                    "bad argument #1 to 'select' (index out of range)",
                ))],
            ),
            ret(call("zl_values_from", vec![args.e(), i.e()], any())),
        ],
    ));
    // Equality without metamethods: the same value.
    d.push(define(
        "zl_rawequal",
        &[&x, &y],
        boolean(),
        vec![
            when(
                and(is_table(x.e()), is_table(y.e())),
                vec![ret(eq(
                    call("zb_unbox_instance_raw", vec![x.e()], i64()),
                    call("zb_unbox_instance_raw", vec![y.e()], i64()),
                ))],
            ),
            ret(call("zl_eq", vec![x.e(), y.e()], boolean())),
        ],
    ));
    d.push(define(
        "zl_rawlen",
        &[&x],
        i64(),
        vec![
            when(
                is_table(x.e()),
                vec![ret(call("zl_len", vec![unbox_table(x.e(), t)], i64()))],
            ),
            when(
                and(not(is_nil(x.e())), eq(category(x.e()), int(STR))),
                vec![ret(call("zb_str_len", vec![get_str(x.e())], i64()))],
            ),
            lua_error(concat(vec![
                text("bad argument #1 to 'rawlen' (table or string expected, got "),
                type_name(x.e()),
                text(")"),
            ])),
            ret(int(0)),
        ],
    ));
    // `assert(v, message)`: the message is the error value as it is,
    // no position added.
    d.push(define(
        "zl_assert",
        &[&x, &args],
        any(),
        vec![
            when(
                not(call("zl_truthy", vec![x.e()], boolean())),
                vec![
                    if_(
                        ge(len(args.e()), int(1)),
                        vec![expr(call(
                            "zl_raise_value",
                            vec![at(args.e(), int(0))],
                            unit(),
                        ))],
                        vec![expr(call(
                            "zl_raise_value",
                            vec![box_str(call(
                                "zl_position",
                                vec![text("assertion failed!")],
                                string(),
                            ))],
                            unit(),
                        ))],
                    ),
                    ret(nil()),
                ],
            ),
            expr(mcall(args.e(), "insert_at", vec![int(0), x.e()], unit())),
            ret(call("zl_pack", vec![args.e()], any())),
        ],
    ));
    // `pcall(f, ...)`: true and the results, or false and the error.
    // The call is from no line: what it calls directly reports no
    // position, as under the reference where the caller is C.
    let handler = kept("handler", any());
    let err = kept("err", any());
    d.push(define(
        "zl_pcall",
        &[&x, &args],
        any(),
        vec![
            set_global(LINE, int(0)),
            y.decl(call("zl_call_packed", vec![x.e(), args.e()], any())),
            when(
                not(is_nil(pending())),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![
                            box_bool(bool(false)),
                            call("zl_take_pending", vec![], any()),
                        ],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            out.decl(list(vec![box_bool(bool(true))], anys.clone())),
            expr(call("zl_append_values", vec![out.e(), y.e()], unit())),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));
    // `xpcall(f, handler, ...)`: the handler sees the error first.
    d.push(define(
        "zl_xpcall",
        &[&x, &handler, &args],
        any(),
        vec![
            when(
                not(is_func(handler.e())),
                vec![lua_error(concat(vec![
                    text("bad argument #2 to 'xpcall' (function expected, got "),
                    type_name(handler.e()),
                    text(")"),
                ]))],
            ),
            set_global(LINE, int(0)),
            y.decl(call("zl_call_packed", vec![x.e(), args.e()], any())),
            when(
                not(is_nil(pending())),
                vec![
                    err.decl(call("zl_take_pending", vec![], any())),
                    err.set(call(
                        "zl_first",
                        vec![call("zl_call_1", vec![handler.e(), err.e()], any())],
                        any(),
                    )),
                    ret(call(
                        "zb_box_tuple",
                        vec![list(vec![box_bool(bool(false)), err.e()], anys.clone())],
                        any(),
                    )),
                ],
            ),
            out.decl(list(vec![box_bool(bool(true))], anys.clone())),
            expr(call("zl_append_values", vec![out.e(), y.e()], unit())),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));
    // `table.unpack(t, i, j)`: the values `t[i]..t[j]`.
    d.push(define(
        "zl_unpack",
        &[&tb, &i, &j],
        any(),
        vec![
            when(
                eq(j.e(), int(i64::MIN)),
                vec![j.set(call("zl_len", vec![tb.e()], i64()))],
            ),
            // The reference's stack holds a million values.
            when(
                and(
                    le(i.e(), j.e()),
                    or(
                        ge(sub(cast(j.e(), u64()), cast(i.e(), u64())), int(1_000_000)),
                        lt(sub(j.e(), i.e()), int(0)),
                    ),
                ),
                vec![lua_error(text("too many results to unpack"))],
            ),
            out.decl(list(vec![], anys.clone())),
            k.decl(i.e()),
            while_(
                le(k.e(), j.e()),
                vec![
                    push(out.e(), call("zl_table_geti", vec![tb.e(), k.e()], any())),
                    k.add_assign(int(1)),
                ],
            ),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));

    // ─── string ─────────────────────────────────────────────────
    for (name, params, ret_ty, symbol) in [
        (
            "zl_string_sub",
            vec![("s", string()), ("i", i64()), ("j", i64())],
            string(),
            "$Lua$sub",
        ),
        (
            "zl_byte_at",
            vec![("s", string()), ("i", i64())],
            i64(),
            "$Lua$byte_at",
        ),
        (
            "zl_from_byte",
            vec![("b", i64())],
            string(),
            "$Lua$from_byte",
        ),
        (
            "zl_string_reverse",
            vec![("s", string())],
            string(),
            "$Lua$reverse",
        ),
        (
            "zl_string_rep",
            vec![("s", string()), ("n", i64()), ("sep", string())],
            string(),
            "$Lua$rep",
        ),
        (
            "zl_find_plain",
            vec![("s", string()), ("p", string()), ("init", i64())],
            i64(),
            "$Lua$find_plain",
        ),
        ("zl_bytes", vec![("hex", string())], string(), "$Lua$bytes"),
        (
            "zl_string_upper",
            vec![("s", string())],
            string(),
            "$Lua$upper",
        ),
        (
            "zl_string_lower",
            vec![("s", string())],
            string(),
            "$Lua$lower",
        ),
        (
            "zl_format_raw",
            vec![
                ("fmt", string()),
                ("args", anys.clone()),
                ("texts", anys.clone()),
            ],
            string(),
            "$Lua$format",
        ),
        (
            "zl_pack_raw",
            vec![("fmt", string()), ("args", anys.clone())],
            string(),
            "$Lua$pack",
        ),
        ("zl_pack_error", vec![], string(), "$Lua$pack_error"),
        (
            "zl_packsize_raw",
            vec![("fmt", string())],
            i64(),
            "$Lua$packsize",
        ),
        (
            "zl_unpack_raw",
            vec![("fmt", string()), ("s", string()), ("pos", i64())],
            i64(),
            "$Lua$unpack",
        ),
        (
            "zl_unpack_kind",
            vec![("i", i64())],
            i64(),
            "$Lua$unpack_kind",
        ),
        (
            "zl_unpack_int",
            vec![("i", i64())],
            i64(),
            "$Lua$unpack_int",
        ),
        (
            "zl_unpack_float",
            vec![("i", i64())],
            f64(),
            "$Lua$unpack_float",
        ),
        (
            "zl_unpack_str",
            vec![("i", i64())],
            string(),
            "$Lua$unpack_str",
        ),
        ("zl_unpack_next", vec![], i64(), "$Lua$unpack_next"),
        ("zl_os_clock", vec![], f64(), "$Lua$clock"),
        ("zl_os_time", vec![], i64(), "$Lua$time"),
        ("zl_os_error", vec![], string(), "$Lua$os_error"),
        (
            "zl_date_field",
            vec![("t", i64()), ("utc", boolean()), ("index", i64())],
            i64(),
            "$Lua$date_field",
        ),
        (
            "zl_time_of",
            vec![
                ("year", i64()),
                ("month", i64()),
                ("day", i64()),
                ("hour", i64()),
                ("min", i64()),
                ("sec", i64()),
                ("isdst", i64()),
            ],
            i64(),
            "$Lua$time_of",
        ),
        (
            "zl_date_check",
            vec![("fmt", string())],
            string(),
            "$Lua$date_check",
        ),
        (
            "zl_date_raw",
            vec![("fmt", string()), ("t", i64()), ("utc", boolean())],
            string(),
            "$Lua$date",
        ),
        (
            "zl_getenv",
            vec![("name", string())],
            string(),
            "$Lua$getenv",
        ),
        ("zl_tmpname", vec![], string(), "$Lua$tmpname"),
        ("zl_remove", vec![("name", string())], i64(), "$Lua$remove"),
        (
            "zl_rename",
            vec![("from", string()), ("to", string())],
            i64(),
            "$Lua$rename",
        ),
        (
            "zl_execute",
            vec![("command", string())],
            i64(),
            "$Lua$execute",
        ),
        (
            "zl_exec_result",
            vec![("status", i64()), ("want_signal", boolean())],
            i64(),
            "$Lua$exec_result",
        ),
        (
            "zl_setlocale",
            vec![("locale", string())],
            string(),
            "$Lua$setlocale",
        ),
        (
            "zl_random_seed",
            vec![("n1", i64()), ("n2", i64())],
            unit(),
            "$Lua$random_seed",
        ),
        ("zl_random_next", vec![], i64(), "$Lua$random_next"),
        ("zl_random_seed_now", vec![], i64(), "$Lua$random_seed_now"),
        ("zl_random_seed_pid", vec![], i64(), "$Lua$random_seed_pid"),
        ("zl_random_float", vec![], f64(), "$Lua$random_float"),
        (
            "zl_random_int",
            vec![("lo", i64()), ("hi", i64())],
            i64(),
            "$Lua$random_int",
        ),
    ] {
        let params: Vec<(&str, Type)> = params.into_iter().collect();
        d.push(extern_fn(name, &params, ret_ty, Some(symbol)));
    }
    d.push(define(
        "zl_string_len",
        &[&s],
        i64(),
        vec![ret(call("zb_str_len", vec![s.e()], i64()))],
    ));
    // `string.byte(s, i, j)`: the bytes at `i..=j`, `j` being `i` when
    // absent.
    d.push(define(
        "zl_string_byte",
        &[&s, &i, &j],
        any(),
        vec![
            when(eq(j.e(), int(i64::MIN)), vec![j.set(i.e())]),
            n.decl(call("zb_str_len", vec![s.e()], i64())),
            when(
                lt(i.e(), int(0)),
                vec![i.set(add(add(n.e(), i.e()), int(1)))],
            ),
            when(
                lt(j.e(), int(0)),
                vec![j.set(add(add(n.e(), j.e()), int(1)))],
            ),
            when(lt(i.e(), int(1)), vec![i.set(int(1))]),
            when(gt(j.e(), n.e()), vec![j.set(n.e())]),
            out.decl(list(vec![], anys.clone())),
            k.decl(i.e()),
            while_(
                le(k.e(), j.e()),
                vec![
                    push(
                        out.e(),
                        box_i64(call("zl_byte_at", vec![s.e(), k.e()], i64())),
                    ),
                    k.add_assign(int(1)),
                ],
            ),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));
    d.push(define(
        "zl_string_char",
        &[&args],
        string(),
        vec![
            acc.decl(text("")),
            i.decl(int(0)),
            while_(
                lt(i.e(), len(args.e())),
                vec![
                    n.decl(call(
                        "zl_arg_int",
                        vec![at(args.e(), i.e()), bad_arg_at(i.e(), "char")],
                        i64(),
                    )),
                    when(
                        or(lt(n.e(), int(0)), gt(n.e(), int(255))),
                        vec![lua_error(add(
                            bad_arg_at(i.e(), "char"),
                            text(" (value out of range)"),
                        ))],
                    ),
                    acc.set(add(acc.e(), call("zl_from_byte", vec![n.e()], string()))),
                    i.add_assign(int(1)),
                ],
            ),
            ret(acc.e()),
        ],
    ));
    // The host formats; a message it could not is prefixed by a byte
    // no text starts with.
    d.push(define(
        "zl_string_format",
        &[&s, &args],
        string(),
        vec![
            // An object's `%s` text is `tostring`'s, metamethods and
            // `__name` included, worked out here for the host.
            out.decl(list(vec![], anys.clone())),
            i.decl(int(0)),
            while_(
                lt(i.e(), len(args.e())),
                vec![
                    x.decl(at(args.e(), i.e())),
                    if_(
                        and(not(is_nil(x.e())), eq(category(x.e()), int(CUSTOM))),
                        vec![push(
                            out.e(),
                            box_str(call("zl_tostring", vec![x.e()], string())),
                        )],
                        vec![push(out.e(), nil())],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            acc.decl(call(
                "zl_format_raw",
                vec![s.e(), args.e(), out.e()],
                string(),
            )),
            when(
                eq(acc.e(), null(string())),
                vec![lua_error(call("zl_pack_error", vec![], string()))],
            ),
            ret(acc.e()),
        ],
    ));
    // ─── math ───────────────────────────────────────────────────
    let number_arg = |x: &Local, what: &str| {
        vec![
            y.decl(call("zl_arith_operand", vec![x.e()], any())),
            when(
                is_nil(y.e()),
                vec![lua_error(concat(vec![
                    text(&format!(
                        "bad argument #1 to '{what}' (number expected, got "
                    )),
                    if_expr(is_nil(x.e()), text("no value"), type_name(x.e())),
                    text(")"),
                ]))],
            ),
        ]
    };
    let is_int_box = |x: Expr| {
        let c = category(x);
        or(eq(c.clone(), int(INT)), eq(c, int(UINT)))
    };
    // A float with an integral value that fits becomes an integer.
    let int_if_fits = |f: Expr| {
        if_expr(
            call("zl_float_is_int", vec![f.clone()], boolean()),
            box_i64(cast(f.clone(), i64())),
            box_f64(f),
        )
    };
    d.push(define(
        "zl_ceil_f64",
        &[&f],
        f64(),
        vec![ret(sub(
            float(0.0),
            call("floor", vec![sub(float(0.0), f.e())], f64()),
        ))],
    ));
    for (name, what, op) in [
        ("zl_math_floor", "floor", "floor"),
        ("zl_math_ceil", "ceil", "zl_ceil_f64"),
    ] {
        let mut st = number_arg(&x, what);
        st.push(when(is_int_box(y.e()), vec![ret(y.e())]));
        st.push(ret(int_if_fits(call(op, vec![get_f64(y.e())], f64()))));
        d.push(define(name, &[&x], any(), st));
    }
    d.push(define("zl_math_abs", &[&x], any(), {
        let mut st = number_arg(&x, "abs");
        st.push(when(
            is_int_box(y.e()),
            vec![ret(if_expr(
                lt(get_i64(y.e()), int(0)),
                box_i64(sub(int(0), get_i64(y.e()))),
                y.e(),
            ))],
        ));
        st.push(ret(box_f64(call(
            "zb_math_fabs",
            vec![get_f64(y.e())],
            f64(),
        ))));
        st
    }));
    for (name, shared) in [
        ("zl_math_sqrt", "sqrt"),
        ("zl_math_exp", "zb_math_exp"),
        ("zl_math_sin", "zb_math_sin"),
        ("zl_math_cos", "zb_math_cos"),
        ("zl_math_tan", "zb_math_tan"),
        ("zl_math_asin", "zb_math_asin"),
        ("zl_math_acos", "zb_math_acos"),
    ] {
        d.push(define(
            name,
            &[&f],
            f64(),
            vec![ret(call(shared, vec![f.e()], f64()))],
        ));
    }
    d.push(define(
        "zl_math_deg",
        &[&f],
        f64(),
        vec![ret(mul(f.e(), float(180.0 / std::f64::consts::PI)))],
    ));
    d.push(define(
        "zl_math_rad",
        &[&f],
        f64(),
        vec![ret(mul(f.e(), float(std::f64::consts::PI / 180.0)))],
    ));
    d.push(define(
        "zl_math_atan",
        &[&f, &x],
        f64(),
        vec![
            g.decl(float(1.0)),
            when(
                not(is_nil(x.e())),
                vec![g.set(call("zl_arg_float", vec![x.e(), bad_arg(2, "atan")], f64()))],
            ),
            ret(call("zb_math_atan2", vec![f.e(), g.e()], f64())),
        ],
    ));
    d.push(define(
        "zl_math_log",
        &[&f, &x],
        f64(),
        vec![
            when(
                is_nil(x.e()),
                vec![ret(call("zb_math_log", vec![f.e()], f64()))],
            ),
            g.decl(call("zl_arg_float", vec![x.e(), bad_arg(2, "log")], f64())),
            when(
                eq(g.e(), float(2.0)),
                vec![ret(call("zb_math_log2", vec![f.e()], f64()))],
            ),
            when(
                eq(g.e(), float(10.0)),
                vec![ret(call("zb_math_log10", vec![f.e()], f64()))],
            ),
            ret(div(
                call("zb_math_log", vec![f.e()], f64()),
                call("zb_math_log", vec![g.e()], f64()),
            )),
        ],
    ));
    // `math.fmod`: the remainder rounded toward zero.
    d.push(define(
        "zl_math_fmod",
        &[&x, &y],
        any(),
        vec![
            when(
                and(is_int_box(x.e()), is_int_box(y.e())),
                vec![
                    when(
                        eq(get_i64(y.e()), int(0)),
                        vec![lua_error(text("bad argument #2 to 'fmod' (zero)"))],
                    ),
                    ret(box_i64(rem(get_i64(x.e()), get_i64(y.e())))),
                ],
            ),
            ret(box_f64(rem(
                call("zl_arg_float", vec![x.e(), bad_arg(1, "fmod")], f64()),
                call("zl_arg_float", vec![y.e(), bad_arg(2, "fmod")], f64()),
            ))),
        ],
    ));
    // `math.modf`: the integral part (an integer when it fits) and the
    // fraction, which is zero for an infinity.
    let whole = kept("whole", any());
    let fraction = local("fraction", f64());
    d.push(define(
        "zl_math_modf",
        &[&f],
        any(),
        vec![
            g.decl(float(0.0)),
            if_(
                ge(f.e(), float(0.0)),
                vec![g.set(call("floor", vec![f.e()], f64()))],
                vec![g.set(call("zl_ceil_f64", vec![f.e()], f64()))],
            ),
            whole.decl(nil()),
            if_(
                call("zl_float_is_int", vec![g.e()], boolean()),
                vec![whole.set(box_i64(cast(g.e(), i64())))],
                vec![whole.set(box_f64(g.e()))],
            ),
            fraction.decl(sub(f.e(), g.e())),
            when(
                or(
                    eq(f.e(), float(f64::INFINITY)),
                    eq(f.e(), float(f64::NEG_INFINITY)),
                ),
                vec![fraction.set(float(0.0))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![whole.e(), box_f64(fraction.e())], anys.clone())],
                any(),
            )),
        ],
    ));
    for (name, what, pick) in [
        ("zl_math_max", "max", "zl_lt"),
        ("zl_math_min", "min", "zl_lt"),
    ] {
        let best = kept("best", any());
        let better = if name == "zl_math_max" {
            call(pick, vec![best.e(), at(args.e(), i.e())], boolean())
        } else {
            call(pick, vec![at(args.e(), i.e()), best.e()], boolean())
        };
        d.push(define(
            name,
            &[&args],
            any(),
            vec![
                // The values are compared as `<` compares them, whatever
                // they are; one alone is the answer.
                when(
                    eq(len(args.e()), int(0)),
                    vec![lua_error(text(&format!(
                        "bad argument #1 to '{what}' (value expected)"
                    )))],
                ),
                best.decl(at(args.e(), int(0))),
                i.decl(int(1)),
                while_(
                    lt(i.e(), len(args.e())),
                    vec![
                        when(better, vec![best.set(at(args.e(), i.e()))]),
                        i.add_assign(int(1)),
                    ],
                ),
                ret(best.e()),
            ],
        ));
    }
    d.push(define(
        "zl_math_tointeger",
        &[&x],
        any(),
        vec![
            when(is_nil(x.e()), vec![ret(nil())]),
            when(is_int_box(x.e()), vec![ret(x.e())]),
            when(
                eq(category(x.e()), int(FLOAT)),
                vec![
                    f.decl(get_f64(x.e())),
                    when(
                        call("zl_float_is_int", vec![f.e()], boolean()),
                        vec![ret(box_i64(cast(f.e(), i64())))],
                    ),
                    ret(nil()),
                ],
            ),
            when(
                eq(category(x.e()), int(STR)),
                vec![ret(call(
                    "zl_math_tointeger",
                    vec![call("zl_str_to_number", vec![get_str(x.e())], any())],
                    any(),
                ))],
            ),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_math_type",
        &[&x],
        any(),
        vec![
            when(is_nil(x.e()), vec![ret(nil())]),
            when(is_int_box(x.e()), vec![ret(box_str(text("integer")))]),
            when(
                eq(category(x.e()), int(FLOAT)),
                vec![ret(box_str(text("float")))],
            ),
            ret(nil()),
        ],
    ));
    let lo = local("lo", i64());
    let hi = local("hi", i64());
    d.push(define(
        "zl_math_random",
        &[&args],
        any(),
        vec![
            n.decl(len(args.e())),
            when(
                eq(n.e(), int(0)),
                vec![ret(box_f64(call("zl_random_float", vec![], f64())))],
            ),
            when(
                gt(n.e(), int(2)),
                vec![lua_error(text("wrong number of arguments"))],
            ),
            lo.decl(int(1)),
            hi.decl(call(
                "zl_arg_int",
                vec![at(args.e(), int(0)), bad_arg(1, "random")],
                i64(),
            )),
            // `random(0)`: a whole draw.
            when(
                and(eq(n.e(), int(1)), eq(hi.e(), int(0))),
                vec![ret(box_i64(call("zl_random_next", vec![], i64())))],
            ),
            when(
                ge(n.e(), int(2)),
                vec![
                    lo.set(hi.e()),
                    hi.set(call(
                        "zl_arg_int",
                        vec![at(args.e(), int(1)), bad_arg(2, "random")],
                        i64(),
                    )),
                ],
            ),
            when(
                gt(lo.e(), hi.e()),
                vec![lua_error(text(
                    "bad argument #1 to 'random' (interval is empty)",
                ))],
            ),
            ret(box_i64(call("zl_random_int", vec![lo.e(), hi.e()], i64()))),
        ],
    ));
    // `math.randomseed([n1 [, n2]])`: the two seed numbers are
    // answered, so a run can be repeated. A float seed is taken by its
    // integer value; nothing given seeds from the clock.
    d.push(define(
        "zl_math_randomseed",
        &[&args],
        any(),
        vec![
            i.decl(int(0)),
            j.decl(int(0)),
            if_(
                eq(len(args.e()), int(0)),
                vec![
                    i.set(call("zl_random_seed_now", vec![], i64())),
                    j.set(call("zl_random_seed_pid", vec![], i64())),
                ],
                vec![
                    x.decl(at(args.e(), int(0))),
                    i.set(if_expr(
                        and(not(is_nil(x.e())), is_int_box(x.e())),
                        get_i64(x.e()),
                        cast(
                            call("zl_arg_float", vec![x.e(), bad_arg(1, "randomseed")], f64()),
                            i64(),
                        ),
                    )),
                    when(
                        gt(len(args.e()), int(1)),
                        vec![j.set(call(
                            "zl_arg_int",
                            vec![at(args.e(), int(1)), bad_arg(2, "randomseed")],
                            i64(),
                        ))],
                    ),
                    expr(call("zl_random_seed", vec![i.e(), j.e()], unit())),
                ],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![box_i64(i.e()), box_i64(j.e())], anys.clone())],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_math_ult",
        &[&i, &j],
        boolean(),
        vec![ret(lt(cast(i.e(), u64()), cast(j.e(), u64())))],
    ));

    // ─── table ──────────────────────────────────────────────────
    // `table.insert(t, v)` appends; `table.insert(t, pos, v)` shifts
    // what follows up.
    let arr = borrowed("arr", anys.clone());
    let v = kept("v", any());
    d.push(define(
        "zl_table_insert",
        &[&tb, &args],
        unit(),
        vec![
            n.decl(call("zl_len", vec![tb.e()], i64())),
            when(
                eq(len(args.e()), int(1)),
                vec![
                    expr(call(
                        "zl_table_seti",
                        vec![tb.e(), add(n.e(), int(1)), at(args.e(), int(0))],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                ne(len(args.e()), int(2)),
                vec![lua_error(text("wrong number of arguments to 'insert'"))],
            ),
            i.decl(call(
                "zl_arg_int",
                vec![at(args.e(), int(0)), bad_arg(2, "insert")],
                i64(),
            )),
            when(
                or(lt(i.e(), int(1)), gt(i.e(), add(n.e(), int(1)))),
                vec![lua_error(text(
                    "bad argument #2 to 'insert' (position out of bounds)",
                ))],
            ),
            v.decl(at(args.e(), int(1))),
            when(
                eq(i.e(), add(n.e(), int(1))),
                vec![
                    expr(call("zl_table_seti", vec![tb.e(), i.e(), v.e()], unit())),
                    ret_void(),
                ],
            ),
            // Within the array part: shift in place.
            arr.decl(super::arr_of(tb.e(), t)),
            when(
                eq(len(arr.e()), n.e()),
                vec![
                    expr(mcall(
                        arr.e(),
                        "insert_at",
                        vec![sub(i.e(), int(1)), v.e()],
                        unit(),
                    )),
                    expr(call("zl_migrate", vec![tb.e()], unit())),
                    ret_void(),
                ],
            ),
            k.decl(n.e()),
            while_(
                ge(k.e(), i.e()),
                vec![
                    expr(call(
                        "zl_table_seti",
                        vec![
                            tb.e(),
                            add(k.e(), int(1)),
                            call("zl_table_geti", vec![tb.e(), k.e()], any()),
                        ],
                        unit(),
                    )),
                    k.set(sub(k.e(), int(1))),
                ],
            ),
            expr(call("zl_table_seti", vec![tb.e(), i.e(), v.e()], unit())),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_table_remove",
        &[&tb, &i],
        any(),
        vec![
            n.decl(call("zl_len", vec![tb.e()], i64())),
            when(eq(i.e(), int(i64::MIN)), vec![i.set(n.e())]),
            when(
                and(eq(n.e(), int(0)), or(eq(i.e(), int(0)), eq(i.e(), n.e()))),
                vec![ret(call("zl_table_geti", vec![tb.e(), i.e()], any()))],
            ),
            when(
                or(lt(i.e(), int(1)), gt(i.e(), add(n.e(), int(1)))),
                vec![lua_error(text(
                    "bad argument #2 to 'remove' (position out of bounds)",
                ))],
            ),
            v.decl(call("zl_table_geti", vec![tb.e(), i.e()], any())),
            arr.decl(super::arr_of(tb.e(), t)),
            if_(
                and(eq(len(arr.e()), n.e()), le(i.e(), n.e())),
                vec![
                    expr(mcall(arr.e(), "remove_at", vec![sub(i.e(), int(1))], any())),
                    while_(
                        and(
                            gt(len(arr.e()), int(0)),
                            is_nil(at(arr.e(), sub(len(arr.e()), int(1)))),
                        ),
                        vec![expr(mcall(arr.e(), "pop_last", vec![], any()))],
                    ),
                ],
                vec![
                    k.decl(i.e()),
                    while_(
                        lt(k.e(), n.e()),
                        vec![
                            expr(call(
                                "zl_table_seti",
                                vec![
                                    tb.e(),
                                    k.e(),
                                    call("zl_table_geti", vec![tb.e(), add(k.e(), int(1))], any()),
                                ],
                                unit(),
                            )),
                            k.add_assign(int(1)),
                        ],
                    ),
                    when(
                        le(i.e(), n.e()),
                        vec![expr(call(
                            "zl_table_seti",
                            vec![tb.e(), n.e(), nil()],
                            unit(),
                        ))],
                    ),
                ],
            ),
            ret(v.e()),
        ],
    ));
    // `table.concat(t, sep, i, j)`: the last index defaults to the
    // length; the loop stops on reaching it rather than passing it, so
    // an index at either end of the integers is fine.
    let last = kept("last", any());
    let ok = local("ok", boolean());
    d.push(define(
        "zl_table_concat",
        &[&tb, &sep, &i, &last],
        string(),
        vec![
            j.decl(if_expr(
                is_nil(last.e()),
                call("zl_len", vec![tb.e()], i64()),
                call("zl_arg_int", vec![last.e(), bad_arg(4, "concat")], i64()),
            )),
            acc.decl(text("")),
            when(gt(i.e(), j.e()), vec![ret(acc.e())]),
            k.decl(i.e()),
            ok.decl(bool(true)),
            while_(
                ok.e(),
                vec![
                    v.decl(call("zl_table_geti", vec![tb.e(), k.e()], any())),
                    when(
                        or(
                            is_nil(v.e()),
                            not(or(
                                eq(category(v.e()), int(STR)),
                                or(is_int_box(v.e()), eq(category(v.e()), int(FLOAT))),
                            )),
                        ),
                        vec![lua_error(concat(vec![
                            text("invalid value ("),
                            call("zl_type", vec![v.e()], string()),
                            text(") at index "),
                            call("zb_str_of_int", vec![k.e()], string()),
                            text(" in table for 'concat'"),
                        ]))],
                    ),
                    acc.set(add(acc.e(), call("zl_concat_text", vec![v.e()], string()))),
                    if_(
                        eq(k.e(), j.e()),
                        vec![ok.set(bool(false))],
                        vec![acc.set(add(acc.e(), sep.e())), k.add_assign(int(1))],
                    ),
                ],
            ),
            ret(acc.e()),
        ],
    ));
    d.push(define(
        "zl_table_pack",
        &[&args],
        table.clone(),
        vec![
            tb.decl(call(
                "zl_table_with_arr",
                vec![call("zb_list_copy_any", vec![args.e()], anys.clone())],
                table.clone(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![tb.e(), text("n"), box_i64(len(args.e()))],
                unit(),
            )),
            ret(tb.e()),
        ],
    ));
    // `table.sort(t, comp)`: a quicksort over the array part, ordering
    // by `comp` or `<`.
    let comp = kept("comp", any());
    let lo_i = local("lo", i64());
    let hi_i = local("hi", i64());
    let pivot = kept("pivot", any());
    let tmp = kept("tmp", any());
    let less = |a: Expr, b: Expr| {
        if_expr(
            is_nil(comp.e()),
            call("zl_lt", vec![a.clone(), b.clone()], boolean()),
            call(
                "zl_truthy",
                vec![call(
                    "zl_first",
                    vec![call("zl_call_2", vec![comp.e(), a, b], any())],
                    any(),
                )],
                boolean(),
            ),
        )
    };
    d.push(define(
        "zl_sort_range",
        &[&arr, &lo_i, &hi_i, &comp],
        unit(),
        vec![
            when(ge(lo_i.e(), hi_i.e()), vec![ret_void()]),
            pivot.decl(at(arr.e(), div(add(lo_i.e(), hi_i.e()), int(2)))),
            i.decl(lo_i.e()),
            j.decl(hi_i.e()),
            while_(
                le(i.e(), j.e()),
                vec![
                    while_(
                        less(at(arr.e(), i.e()), pivot.e()),
                        vec![i.add_assign(int(1))],
                    ),
                    while_(
                        less(pivot.e(), at(arr.e(), j.e())),
                        vec![j.set(sub(j.e(), int(1)))],
                    ),
                    when(
                        le(i.e(), j.e()),
                        vec![
                            tmp.decl(at(arr.e(), i.e())),
                            set_idx(arr.e(), i.e(), at(arr.e(), j.e())),
                            set_idx(arr.e(), j.e(), tmp.e()),
                            i.add_assign(int(1)),
                            j.set(sub(j.e(), int(1))),
                        ],
                    ),
                ],
            ),
            expr(call(
                "zl_sort_range",
                vec![arr.e(), lo_i.e(), j.e(), comp.e()],
                unit(),
            )),
            expr(call(
                "zl_sort_range",
                vec![arr.e(), i.e(), hi_i.e(), comp.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_table_sort",
        &[&tb, &comp],
        unit(),
        vec![
            arr.decl(super::arr_of(tb.e(), t)),
            expr(call(
                "zl_sort_range",
                vec![arr.e(), int(0), sub(len(arr.e()), int(1)), comp.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));

    // `string.rep`, refused past the reference's size limit.
    let sep = kept("sep", string());
    let repeated = kept("repeated", string());
    d.push(define(
        "zl_string_rep_of",
        &[&s, &n, &sep],
        string(),
        vec![
            repeated.decl(call("zl_string_rep", vec![s.e(), n.e(), sep.e()], string())),
            when(
                eq(repeated.e(), null(string())),
                vec![lua_error(text("resulting string too large"))],
            ),
            ret(repeated.e()),
        ],
    ));

    // ─── string.pack ────────────────────────────────────────────
    let packed = kept("packed", string());
    d.push(define(
        "zl_string_pack",
        &[&s, &args],
        string(),
        vec![
            packed.decl(call("zl_pack_raw", vec![s.e(), args.e()], string())),
            when(
                eq(packed.e(), null(string())),
                vec![lua_error(call("zl_pack_error", vec![], string()))],
            ),
            ret(packed.e()),
        ],
    ));
    d.push(define(
        "zl_string_packsize",
        &[&s],
        i64(),
        vec![
            n.decl(call("zl_packsize_raw", vec![s.e()], i64())),
            when(
                lt(n.e(), int(0)),
                vec![lua_error(call("zl_pack_error", vec![], string()))],
            ),
            ret(n.e()),
        ],
    ));
    // `string.unpack(fmt, s, pos)`: the position is Lua's, relative
    // from the end when negative; the values come back one by one.
    let data = kept("data", string());
    let pos = local("pos", i64());
    d.push(define(
        "zl_string_unpack",
        &[&s, &data, &pos],
        any(),
        vec![
            n.decl(call("zb_str_len", vec![data.e()], i64())),
            when(
                lt(pos.e(), int(0)),
                vec![pos.set(if_expr(
                    lt(pos.e(), sub(int(0), n.e())),
                    int(1),
                    add(add(n.e(), pos.e()), int(1)),
                ))],
            ),
            when(eq(pos.e(), int(0)), vec![pos.set(int(1))]),
            k.decl(call(
                "zl_unpack_raw",
                vec![s.e(), data.e(), sub(pos.e(), int(1))],
                i64(),
            )),
            when(
                lt(k.e(), int(0)),
                vec![lua_error(call("zl_pack_error", vec![], string()))],
            ),
            out.decl(list(vec![], anys.clone())),
            i.decl(int(0)),
            while_(
                lt(i.e(), k.e()),
                vec![
                    j.decl(call("zl_unpack_kind", vec![i.e()], i64())),
                    when(
                        eq(j.e(), int(0)),
                        vec![push(
                            out.e(),
                            box_i64(call("zl_unpack_int", vec![i.e()], i64())),
                        )],
                    ),
                    when(
                        eq(j.e(), int(1)),
                        vec![push(
                            out.e(),
                            box_f64(call("zl_unpack_float", vec![i.e()], f64())),
                        )],
                    ),
                    when(
                        eq(j.e(), int(2)),
                        vec![push(
                            out.e(),
                            box_str(call("zl_unpack_str", vec![i.e()], string())),
                        )],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            push(out.e(), box_i64(call("zl_unpack_next", vec![], i64()))),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));

    // ─── os and io ──────────────────────────────────────────────
    d.extend(os_declarations(t));
    d.push(define(
        "zl_os_exit",
        &[&x],
        unit(),
        vec![
            k.decl(int(0)),
            when(
                and(not(is_nil(x.e())), eq(category(x.e()), int(BOOL))),
                vec![when(not(get_bool(x.e())), vec![k.set(int(1))])],
            ),
            when(
                and(not(is_nil(x.e())), is_int_box(x.e())),
                vec![k.set(get_i64(x.e()))],
            ),
            expr(call("zb_exit", vec![cast(k.e(), i32())], unit())),
            ret_void(),
        ],
    ));
    // A float as `%.14g`, without the `.0` that `tostring` adds, for
    // `io.write`.
    d.push(extern_fn(
        "zl_format_g",
        &[("x", f64()), ("precision", i64())],
        string(),
        Some("$Lua$format_g"),
    ));
    // ─── the value wrappers and the library tables ──────────────
    let packed = kept("packed", any());
    for b in BUILTINS {
        // Each argument as the implementation takes it, from the
        // packed list; the result boxed.
        let mut st = vec![args.decl(call("zl_values", vec![packed.e()], anys.clone()))];
        // Called as a value, the function is named as the reference
        // finds it in its library's table: `string.rep`.
        let qualified = if b.lib.is_empty() || HIDDEN_LIBS.contains(&b.lib) {
            b.name.to_string()
        } else {
            format!("{}.{}", b.lib, b.name)
        };
        // A required argument not passed at all.
        for (idx, p) in b.params.iter().enumerate() {
            let expected = match p {
                Param::Int | Param::Float => " (number expected, got no value)".to_string(),
                Param::Str => " (string expected, got no value)".to_string(),
                Param::Table => " (table expected, got no value)".to_string(),
                Param::Value => " (value expected)".to_string(),
                Param::Expected(kind) => format!(" ({kind} expected, got no value)"),
                _ => continue,
            };
            st.push(when(
                lt(len(args.e()), int(idx as i64 + 1)),
                vec![
                    lua_error(add(bad_arg(idx + 1, &qualified), text(&expected))),
                    ret(nil()),
                ],
            ));
        }
        let mut call_args = Vec::new();
        for (idx, p) in b.params.iter().enumerate() {
            let arg = call("zl_value_at", vec![args.e(), int(idx as i64 + 1)], any());
            let what = bad_arg(idx + 1, &qualified);
            call_args.push(match p {
                Param::Any | Param::Value | Param::Expected(_) => arg,
                Param::Int => call("zl_arg_int", vec![arg, what], i64()),
                Param::Float => call("zl_arg_float", vec![arg, what], f64()),
                Param::Str => call("zl_arg_str", vec![arg, what], string()),
                Param::Table => call("zl_as_table", vec![arg, what], table.clone()),
                Param::OptInt(v) => call("zl_arg_opt_int", vec![arg, int(*v), what], i64()),
                Param::OptStr(v) => call("zl_arg_opt_str", vec![arg, text(v), what], string()),
                // Everything from here on; a slice past the end is empty.
                Param::Rest => call(
                    "zl_slice",
                    vec![args.e(), int(idx as i64), len(args.e())],
                    anys.clone(),
                ),
            });
        }
        let result = call(b.func, call_args, ret_type(b.ret, t));
        st.push(match b.ret {
            Ret::Unit => expr(result),
            Ret::Bool => ret(box_bool(result)),
            Ret::Int => ret(box_i64(result)),
            Ret::Float => ret(box_f64(result)),
            Ret::Str => ret(box_str(result)),
            Ret::Any | Ret::Multi => ret(result),
            Ret::Table => ret(box_table(result)),
        });
        if b.ret == Ret::Unit {
            st.push(ret(call("zl_none", vec![], any())));
        }
        d.push(define(&wrapper_name(b), &[&env, &packed], any(), st));
    }
    // Each library's table, built once: the wrappers as function
    // values, then the constants.
    for lib in LIBS.iter().chain(HIDDEN_LIBS) {
        let cache = format!("zl_lib_{lib}_cache");
        d.push(zyntax_typed_ast::TypedNode::new(
            zyntax_typed_ast::typed_ast::TypedDeclaration::Variable(
                zyntax_typed_ast::typed_ast::TypedVariable {
                    name: intern(&cache),
                    ty: any(),
                    mutability: zyntax_typed_ast::Mutability::Mutable,
                    initializer: None,
                    visibility: zyntax_typed_ast::Visibility::Public,
                },
            ),
            Type::Unknown,
            SPAN,
        ));
        let cached = || {
            node(
                zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(&cache)),
                any(),
            )
        };
        // Cached before it is filled: `package.loaded` holds every
        // library's table, `package` included.
        let mut st = vec![
            when(not(is_nil(cached())), vec![ret(cached())]),
            tb.decl(call("zl_table_new", vec![], table.clone())),
            expr(node(
                zyntax_typed_ast::typed_ast::TypedExpression::Binary(
                    zyntax_typed_ast::typed_ast::TypedBinary {
                        op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                        left: Box::new(cached()),
                        right: Box::new(box_table(tb.e())),
                    },
                ),
                any(),
            )),
        ];
        for b in BUILTINS.iter().filter(|b| b.lib == *lib) {
            st.push(expr(call(
                "zl_rawset_str",
                vec![tb.e(), text(b.name), builtin_value(b)],
                unit(),
            )));
        }
        for (_, name, c) in CONSTANTS.iter().filter(|(l, _, _)| l == lib) {
            let value = match c {
                Constant::Int(v) => box_i64(int(*v)),
                Constant::Float(v) => box_f64(float(*v)),
                Constant::Bytes(hex) => box_str(call("zl_bytes", vec![text(hex)], string())),
            };
            st.push(expr(call(
                "zl_rawset_str",
                vec![tb.e(), text(name), value],
                unit(),
            )));
        }
        if *lib == "io" {
            for (name, getter) in [
                ("stdin", "zl_io_stdin"),
                ("stdout", "zl_io_stdout"),
                ("stderr", "zl_io_stderr"),
            ] {
                st.push(expr(call(
                    "zl_rawset_str",
                    vec![tb.e(), text(name), call(getter, vec![], any())],
                    unit(),
                )));
            }
        }
        if *lib == "package" {
            st.push(expr(call(
                "zl_rawset_str",
                vec![
                    tb.e(),
                    text("loaded"),
                    call("zl_package_loaded", vec![], any()),
                ],
                unit(),
            )));
            st.push(expr(call(
                "zl_rawset_str",
                vec![
                    tb.e(),
                    text("preload"),
                    call("zl_package_preload", vec![], any()),
                ],
                unit(),
            )));
        }
        st.push(ret(cached()));
        d.push(define(&lib_table_fn(lib), &[], any(), st));
    }
    // `arg`: the script's path at 0, its arguments from 1, built on
    // first use.
    d.push(extern_fn("zl_argc", &[], i64(), Some("$Lua$argc")));
    d.push(extern_fn(
        "zl_argv",
        &[("i", i64())],
        string(),
        Some("$Lua$argv"),
    ));
    d.push(zyntax_typed_ast::TypedNode::new(
        zyntax_typed_ast::typed_ast::TypedDeclaration::Variable(
            zyntax_typed_ast::typed_ast::TypedVariable {
                name: intern("zl_arg_cache"),
                ty: any(),
                mutability: zyntax_typed_ast::Mutability::Mutable,
                initializer: None,
                visibility: zyntax_typed_ast::Visibility::Public,
            },
        ),
        Type::Unknown,
        SPAN,
    ));
    {
        let cached = || {
            node(
                zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern("zl_arg_cache")),
                any(),
            )
        };
        d.push(define(
            "zl_arg_table",
            &[],
            any(),
            vec![
                when(not(is_nil(cached())), vec![ret(cached())]),
                tb.decl(call("zl_table_new", vec![], table.clone())),
                n.decl(call("zl_argc", vec![], i64())),
                // The interpreter sits at -1, the script at 0.
                i.decl(int(0)),
                while_(
                    lt(i.e(), n.e()),
                    vec![
                        expr(call(
                            "zl_rawseti",
                            vec![
                                tb.e(),
                                sub(i.e(), int(1)),
                                box_str(call("zl_argv", vec![i.e()], string())),
                            ],
                            unit(),
                        )),
                        i.add_assign(int(1)),
                    ],
                ),
                expr(node(
                    zyntax_typed_ast::typed_ast::TypedExpression::Binary(
                        zyntax_typed_ast::typed_ast::TypedBinary {
                            op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                            left: Box::new(cached()),
                            right: Box::new(box_table(tb.e())),
                        },
                    ),
                    any(),
                )),
                ret(cached()),
            ],
        ));
    }
    // The script's arguments after its name, as `...` at the main
    // chunk gives them.
    d.push(define(
        "zl_script_args",
        &[],
        anys.clone(),
        vec![
            out.decl(list(vec![], anys.clone())),
            n.decl(call("zl_argc", vec![], i64())),
            i.decl(int(2)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    push(out.e(), box_str(call("zl_argv", vec![i.e()], string()))),
                    i.add_assign(int(1)),
                ],
            ),
            ret(out.e()),
        ],
    ));
    // The globals table, for a program that reaches its globals through
    // `_G`: every base function, every library's table, `_G` itself,
    // `_VERSION` and `arg`. The program's own globals are set into it
    // as it runs.
    d.push(global_var(GLOBALS, table.clone()));
    {
        // Built once, and reachable through `zl_G` while it is being
        // filled: `package.loaded` holds `_G`.
        let g = || {
            node(
                zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(GLOBALS)),
                table.clone(),
            )
        };
        let mut st = vec![
            when(ne(g(), null(table.clone())), vec![ret(g())]),
            tb.decl(call("zl_table_new", vec![], table.clone())),
            expr(node(
                zyntax_typed_ast::typed_ast::TypedExpression::Binary(
                    zyntax_typed_ast::typed_ast::TypedBinary {
                        op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                        left: Box::new(g()),
                        right: Box::new(tb.e()),
                    },
                ),
                table.clone(),
            )),
        ];
        for b in BUILTINS.iter().filter(|b| b.lib.is_empty()) {
            st.push(expr(call(
                "zl_rawset_str",
                vec![tb.e(), text(b.name), builtin_value(b)],
                unit(),
            )));
        }
        for lib in LIBS {
            st.push(expr(call(
                "zl_rawset_str",
                vec![tb.e(), text(lib), call(&lib_table_fn(lib), vec![], any())],
                unit(),
            )));
        }
        st.push(expr(call(
            "zl_rawset_str",
            vec![tb.e(), text("_G"), box_table(tb.e())],
            unit(),
        )));
        st.push(expr(call(
            "zl_rawset_str",
            vec![tb.e(), text("_VERSION"), box_str(text("Lua 5.4"))],
            unit(),
        )));
        st.push(expr(call(
            "zl_rawset_str",
            vec![tb.e(), text("arg"), call("zl_arg_table", vec![], any())],
            unit(),
        )));
        st.push(ret(tb.e()));
        d.push(define("zl_globals_table", &[], table.clone(), st));
    }
    // The metatable every string shares, `{__index = string}`, which a
    // program may edit; a member of a string is what its `__index`
    // says.
    d.push(global_var("zl_string_meta", any()));
    let cached = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            any(),
        )
    };
    let assign_global = |name: &str, value: Expr| {
        expr(node(
            zyntax_typed_ast::typed_ast::TypedExpression::Binary(
                zyntax_typed_ast::typed_ast::TypedBinary {
                    op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                    left: Box::new(cached(name)),
                    right: Box::new(value),
                },
            ),
            any(),
        ))
    };
    d.push(define(
        "zl_string_metatable",
        &[],
        any(),
        vec![
            when(
                is_nil(cached("zl_string_meta")),
                vec![
                    tb.decl(call("zl_table_new", vec![], table.clone())),
                    expr(call(
                        "zl_rawset_str",
                        vec![
                            tb.e(),
                            text("__index"),
                            call(&lib_table_fn("string"), vec![], any()),
                        ],
                        unit(),
                    )),
                    assign_global("zl_string_meta", box_table(tb.e())),
                ],
            ),
            ret(cached("zl_string_meta")),
        ],
    ));
    let is_func = |x: Expr| {
        and(
            ne(x.clone(), nil()),
            eq(tag_of(x), int(zyntax_builtins::FUNC_TAG)),
        )
    };
    d.push(define(
        "zl_string_member",
        &[&x, &y],
        any(),
        vec![
            handler.decl(call(
                "zl_index",
                vec![
                    call("zl_string_metatable", vec![], any()),
                    box_str(text("__index")),
                ],
                any(),
            )),
            when(is_nil(handler.e()), vec![ret(nil())]),
            when(
                is_func(handler.e()),
                vec![ret(call(
                    "zl_first",
                    vec![call("zl_call_2", vec![handler.e(), x.e(), y.e()], any())],
                    any(),
                ))],
            ),
            ret(call("zl_index", vec![handler.e(), y.e()], any())),
        ],
    ));

    // `load(chunk, name, mode, env)`: the chunk, a string or a
    // function giving pieces of it, compiled while the program runs;
    // the function value, or nil and the message.
    d.push(extern_fn(
        "zl_load_raw",
        &[("source", string()), ("name", string()), ("env", any())],
        any(),
        Some("$Lua$load"),
    ));
    d.push(extern_fn(
        "zl_searchpath",
        &[
            ("name", string()),
            ("path", string()),
            ("sep", string()),
            ("rep", string()),
        ],
        string(),
        Some("$Lua$searchpath"),
    ));
    d.push(extern_fn(
        "zl_load_error",
        &[],
        string(),
        Some("$Lua$load_error"),
    ));
    let chunk = kept("chunk", any());
    let chunk_name = kept("chunk_name", any());
    let mode = kept("mode", any());
    let env = kept("env", any());
    let piece = local("piece", any());
    let cname = local("cname", string());
    d.push(define(
        "zl_load",
        &[&chunk, &chunk_name, &mode, &env],
        any(),
        vec![
            s.decl(text("")),
            if_(
                and(
                    not(is_nil(chunk.e())),
                    or(
                        eq(category(chunk.e()), int(STR)),
                        or(
                            eq(category(chunk.e()), int(INT)),
                            or(
                                eq(category(chunk.e()), int(UINT)),
                                eq(category(chunk.e()), int(FLOAT)),
                            ),
                        ),
                    ),
                ),
                vec![s.set(call("zl_arg_str", vec![chunk.e(), text("")], string()))],
                vec![
                    when(
                        not(is_func(chunk.e())),
                        vec![lua_error(concat(vec![
                            text("bad argument #1 to 'load' (function expected, got "),
                            type_name(chunk.e()),
                            text(")"),
                        ]))],
                    ),
                    // The reader gives pieces until nil or an empty string.
                    expr(call("zl_buf_open", vec![], unit())),
                    piece.decl(call(
                        "zl_first",
                        vec![call("zl_call_0", vec![chunk.e()], any())],
                        any(),
                    )),
                    while_(
                        and(
                            not(is_nil(piece.e())),
                            and(
                                eq(category(piece.e()), int(STR)),
                                gt(call("zb_str_len", vec![get_str(piece.e())], i64()), int(0)),
                            ),
                        ),
                        vec![
                            expr(call("zl_buf_push", vec![get_str(piece.e())], unit())),
                            piece.set(call(
                                "zl_first",
                                vec![call("zl_call_0", vec![chunk.e()], any())],
                                any(),
                            )),
                        ],
                    ),
                    s.set(call("zl_buf_close", vec![], string())),
                    when(
                        and(not(is_nil(piece.e())), ne(category(piece.e()), int(STR))),
                        vec![ret(call(
                            "zb_box_tuple",
                            vec![list(
                                vec![nil(), box_str(text("reader function must return a string"))],
                                anys.clone(),
                            )],
                            any(),
                        ))],
                    ),
                ],
            ),
            cname.decl(text("")),
            when(
                eq(category(chunk_name.e()), int(STR)),
                vec![cname.set(get_str(chunk_name.e()))],
            ),
            y.decl(call("zl_load_mode_error", vec![s.e(), mode.e()], any())),
            when(
                not(is_nil(y.e())),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(vec![nil(), y.e()], anys.clone())],
                    any(),
                ))],
            ),
            y.set(call("zl_load_raw", vec![s.e(), cname.e(), env.e()], any())),
            when(
                is_nil(y.e()),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_str(call("zl_load_error", vec![], string()))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            ret(y.e()),
        ],
    ));
    // What `mode` says about a chunk: a chunk starting with an escape
    // byte is binary, which no mode without `b` allows and which
    // cannot be loaded here at all; a text chunk needs `t`. The
    // message when the chunk is refused, or nil.
    d.push(define(
        "zl_load_mode_error",
        &[&s, &mode],
        any(),
        vec![
            k.decl(int(0)),
            when(
                gt(call("zb_str_len", vec![s.e()], i64()), int(0)),
                vec![k.set(cast(
                    call("zb_str_code_at", vec![s.e(), int(0)], i32()),
                    i64(),
                ))],
            ),
            cname.decl(text("bt")),
            when(
                and(not(is_nil(mode.e())), eq(category(mode.e()), int(STR))),
                vec![cname.set(get_str(mode.e()))],
            ),
            when(
                and(
                    eq(k.e(), int(27)),
                    not(call(
                        "zb_str_contains",
                        vec![cname.e(), text("b")],
                        boolean(),
                    )),
                ),
                vec![ret(box_str(concat(vec![
                    text("attempt to load a binary chunk (mode is '"),
                    cname.e(),
                    text("')"),
                ])))],
            ),
            when(
                and(
                    ne(k.e(), int(27)),
                    not(call(
                        "zb_str_contains",
                        vec![cname.e(), text("t")],
                        boolean(),
                    )),
                ),
                vec![ret(box_str(concat(vec![
                    text("attempt to load a text chunk (mode is '"),
                    cname.e(),
                    text("')"),
                ])))],
            ),
            when(
                eq(k.e(), int(27)),
                vec![ret(box_str(text("binary chunks cannot be loaded")))],
            ),
            ret(nil()),
        ],
    ));
    // `loadfile(name, mode, env)`: the file as a chunk named `@name`.
    d.push(extern_fn(
        "zl_read_file",
        &[("path", string())],
        string(),
        Some("$Lua$read_file"),
    ));
    let path = kept("path", string());
    d.push(define(
        "zl_loadfile",
        &[&path, &mode, &env],
        any(),
        vec![
            s.decl(call("zl_read_file", vec![path.e()], string())),
            when(
                eq(s.e(), null(string())),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_str(call("zl_load_error", vec![], string()))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            y.decl(call("zl_load_mode_error", vec![s.e(), mode.e()], any())),
            when(
                not(is_nil(y.e())),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(vec![nil(), y.e()], anys.clone())],
                    any(),
                ))],
            ),
            y.set(call(
                "zl_load_raw",
                vec![s.e(), add(text("@"), path.e()), env.e()],
                any(),
            )),
            when(
                is_nil(y.e()),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_str(call("zl_load_error", vec![], string()))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            ret(y.e()),
        ],
    ));
    // `dofile(name)`: the file run, its values returned; an error in
    // loading is raised.
    d.push(define(
        "zl_dofile",
        &[&path],
        any(),
        vec![
            s.decl(call("zl_read_file", vec![path.e()], string())),
            when(
                eq(s.e(), null(string())),
                vec![lua_error(call("zl_load_error", vec![], string()))],
            ),
            y.decl(call(
                "zl_load_raw",
                vec![s.e(), add(text("@"), path.e()), nil()],
                any(),
            )),
            when(
                is_nil(y.e()),
                vec![lua_error(call("zl_load_error", vec![], string()))],
            ),
            ret(call("zl_call_0", vec![y.e()], any())),
        ],
    ));
    // The globals a loaded chunk reads: the env it was given when that
    // is a table, else the program's.
    d.push(define(
        "zl_env_table",
        &[&env],
        table.clone(),
        vec![
            when(is_table(env.e()), vec![ret(unbox_table(env.e(), t))]),
            ret(call("zl_globals_table", vec![], table.clone())),
        ],
    ));
    // The environment a chunk was given, or the globals table when it
    // was given nothing.
    d.push(define(
        "zl_env_value",
        &[&env],
        any(),
        vec![
            when(
                is_nil(env.e()),
                vec![ret(call("zl_globals_value", vec![], any()))],
            ),
            ret(env.e()),
        ],
    ));
    // `collectgarbage(opt)`: a collection, or what the collector knows.
    d.push(extern_fn("zl_gc", &[("op", i64())], i64(), Some("$Lua$gc")));
    let opt = kept("opt", string());
    let is = |name: &str| call("zb_str_eq", vec![opt.e(), text(name)], boolean());
    d.push(define(
        "zl_collectgarbage",
        &[&opt, &x],
        any(),
        vec![
            when(
                or(is("collect"), is("step")),
                vec![
                    expr(call("zl_gc", vec![int(0)], i64())),
                    when(is("step"), vec![ret(box_bool(bool(false)))]),
                    ret(box_i64(int(0))),
                ],
            ),
            when(
                is("count"),
                vec![
                    n.decl(call("zl_gc", vec![int(1)], i64())),
                    ret(box_f64(div(cast(n.e(), f64()), float(1024.0)))),
                ],
            ),
            when(is("isrunning"), vec![ret(box_bool(bool(true)))]),
            when(
                or(is("incremental"), is("generational")),
                vec![ret(box_str(text("incremental")))],
            ),
            when(
                or(
                    or(is("stop"), is("restart")),
                    or(is("setpause"), is("setstepmul")),
                ),
                vec![ret(box_i64(int(0)))],
            ),
            lua_error(concat(vec![
                text("bad argument #1 to 'collectgarbage' (invalid option '"),
                opt.e(),
                text("')"),
            ])),
            ret(nil()),
        ],
    ));
    // `require(name)`: what is loaded, a preloaded module (the
    // program's own files, compiled with it), or a file along
    // `package.path`.
    let name = kept("name", string());
    d.push(define(
        "zl_require",
        &[&name],
        any(),
        vec![
            y.decl(call(
                "zl_index",
                vec![call("zl_package_loaded", vec![], any()), box_str(name.e())],
                any(),
            )),
            when(not(is_nil(y.e())), vec![ret(y.e())]),
            // A loader in `package.preload`: what it returns is the
            // module, or true when it returns nothing.
            handler.decl(call(
                "zl_index",
                vec![call("zl_package_preload", vec![], any()), box_str(name.e())],
                any(),
            )),
            when(
                not(is_nil(handler.e())),
                vec![
                    // The loader sees the name and the file it was found
                    // as, `./name.lua` under the program's own directory.
                    y.set(call(
                        "zl_first",
                        vec![call(
                            "zl_call_2",
                            vec![
                                handler.e(),
                                box_str(name.e()),
                                box_str(concat(vec![
                                    text("./"),
                                    call("zl_module_path", vec![name.e()], string()),
                                    text(".lua"),
                                ])),
                            ],
                            any(),
                        )],
                        any(),
                    )),
                    when(not(is_nil(pending())), vec![ret(nil())]),
                    when(is_nil(y.e()), vec![y.set(box_bool(bool(true)))]),
                    expr(call(
                        "zl_rawset_str",
                        vec![
                            unbox_table(call("zl_package_loaded", vec![], any()), t),
                            name.e(),
                            y.e(),
                        ],
                        unit(),
                    )),
                    ret(y.e()),
                ],
            ),
            // A file along `package.path`, compiled now and run as the
            // loader with the name and its file.
            handler.decl(call(
                "zl_index",
                vec![
                    call(&lib_table_fn("package"), vec![], any()),
                    box_str(text("path")),
                ],
                any(),
            )),
            when(
                or(
                    is_nil(handler.e()),
                    and(
                        ne(category(handler.e()), int(STR)),
                        and(
                            ne(category(handler.e()), int(INT)),
                            and(
                                ne(category(handler.e()), int(UINT)),
                                ne(category(handler.e()), int(FLOAT)),
                            ),
                        ),
                    ),
                ),
                vec![lua_error(text("'package.path' must be a string"))],
            ),
            path.decl(call(
                "zl_searchpath",
                vec![
                    name.e(),
                    call("zl_arg_str", vec![handler.e(), text("")], string()),
                    text("."),
                    text("/"),
                ],
                string(),
            )),
            when(
                eq(path.e(), null(string())),
                vec![lua_error(concat(vec![
                    text("module '"),
                    name.e(),
                    text("' not found:\n\tno field package.preload['"),
                    name.e(),
                    text("']\n\t"),
                    call("zl_load_error", vec![], string()),
                ]))],
            ),
            s.decl(call("zl_read_file", vec![path.e()], string())),
            when(
                eq(s.e(), null(string())),
                vec![lua_error(call("zl_load_error", vec![], string()))],
            ),
            handler.set(call(
                "zl_load_raw",
                vec![s.e(), add(text("@"), path.e()), nil()],
                any(),
            )),
            when(
                is_nil(handler.e()),
                vec![lua_error(concat(vec![
                    text("error loading module '"),
                    name.e(),
                    text("' from file '"),
                    path.e(),
                    text("':\n\t"),
                    call("zl_load_error", vec![], string()),
                ]))],
            ),
            y.set(call(
                "zl_first",
                vec![call(
                    "zl_call_2",
                    vec![handler.e(), box_str(name.e()), box_str(path.e())],
                    any(),
                )],
                any(),
            )),
            when(not(is_nil(pending())), vec![ret(nil())]),
            // The module may have set its own entry while loading.
            when(
                is_nil(y.e()),
                vec![y.set(call(
                    "zl_index",
                    vec![call("zl_package_loaded", vec![], any()), box_str(name.e())],
                    any(),
                ))],
            ),
            when(is_nil(y.e()), vec![y.set(box_bool(bool(true)))]),
            expr(call(
                "zl_rawset_str",
                vec![
                    unbox_table(call("zl_package_loaded", vec![], any()), t),
                    name.e(),
                    y.e(),
                ],
                unit(),
            )),
            ret(y.e()),
        ],
    ));
    // `package.searchpath(name, path [, sep [, rep]])`: the file found,
    // or nil and the list of files tried.
    let sep = kept("sep", string());
    let rep = kept("rep", string());
    d.push(define(
        "zl_package_searchpath",
        &[&name, &s, &sep, &rep],
        any(),
        vec![
            path.decl(call(
                "zl_searchpath",
                vec![name.e(), s.e(), sep.e(), rep.e()],
                string(),
            )),
            when(
                eq(path.e(), null(string())),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_str(call("zl_load_error", vec![], string()))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![box_str(path.e())], anys.clone())],
                any(),
            )),
        ],
    ));
    // `package.preload`: loaders by module name; the program's own
    // files are entered here before it runs.
    d.push(global_var("zl_preload", any()));
    d.push(define(
        "zl_package_preload",
        &[],
        any(),
        vec![
            when(
                is_nil(cached("zl_preload")),
                vec![assign_global(
                    "zl_preload",
                    box_table(call("zl_table_new", vec![], table.clone())),
                )],
            ),
            ret(cached("zl_preload")),
        ],
    ));
    // A module name's path: its dots as directory separators.
    d.push(define(
        "zl_module_path",
        &[&name],
        string(),
        vec![ret(call("zl_buf_replace_dots", vec![name.e()], string()))],
    ));
    d.push(define(
        "zl_preload_module",
        &[&name, &x],
        unit(),
        vec![
            expr(call(
                "zl_rawset_str",
                vec![
                    unbox_table(call("zl_package_preload", vec![], any()), t),
                    name.e(),
                    x.e(),
                ],
                unit(),
            )),
            ret_void(),
        ],
    ));
    // `package.loaded`: every library under its name, and `_G`.
    d.push(global_var("zl_loaded", any()));
    {
        let mut st = vec![when(
            not(is_nil(cached("zl_loaded"))),
            vec![ret(cached("zl_loaded"))],
        )];
        st.push(tb.decl(call("zl_table_new", vec![], table.clone())));
        for lib in LIBS {
            st.push(expr(call(
                "zl_rawset_str",
                vec![tb.e(), text(lib), call(&lib_table_fn(lib), vec![], any())],
                unit(),
            )));
        }
        st.push(expr(call(
            "zl_rawset_str",
            vec![tb.e(), text("_G"), call("zl_globals_value", vec![], any())],
            unit(),
        )));
        st.push(assign_global("zl_loaded", box_table(tb.e())));
        st.push(ret(cached("zl_loaded")));
        d.push(define("zl_package_loaded", &[], any(), st));
    }
    // `_G` as a value for a program that never reaches it as one.
    d.push(define(
        "zl_globals_value",
        &[],
        any(),
        vec![ret(box_table(call(
            "zl_globals_table",
            vec![],
            table.clone(),
        )))],
    ));

    // ─── debug ──────────────────────────────────────────────────
    // The message itself: there is no stack to print.
    d.push(define(
        "zl_debug_traceback",
        &[&x, &y],
        any(),
        vec![
            when(
                and(not(is_nil(x.e())), ne(category(x.e()), int(STR))),
                vec![ret(x.e())],
            ),
            when(is_nil(x.e()), vec![ret(box_str(text("stack traceback:")))]),
            ret(box_str(add(get_str(x.e()), text("\nstack traceback:")))),
        ],
    ));
    // What is known of the running function: its chunk and line.
    d.push(define(
        "zl_debug_getinfo",
        &[&x, &y],
        any(),
        vec![
            tb.decl(call("zl_table_new", vec![], table.clone())),
            expr(call(
                "zl_rawset_str",
                vec![
                    tb.e(),
                    text("currentline"),
                    box_i64(bitand(
                        read_global(LINE, i64()),
                        int((1i64 << LINE_BITS) - 1),
                    )),
                ],
                unit(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![
                    tb.e(),
                    text("short_src"),
                    box_str(call(
                        "zl_chunk_of",
                        vec![read_global(LINE, i64())],
                        string(),
                    )),
                ],
                unit(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![
                    tb.e(),
                    text("source"),
                    box_str(add(
                        text("@"),
                        call("zl_chunk_of", vec![read_global(LINE, i64())], string()),
                    )),
                ],
                unit(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![tb.e(), text("what"), box_str(text("Lua"))],
                unit(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![tb.e(), text("namewhat"), box_str(text(""))],
                unit(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![tb.e(), text("linedefined"), box_i64(int(0))],
                unit(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![tb.e(), text("lastlinedefined"), box_i64(int(0))],
                unit(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![tb.e(), text("nups"), box_i64(int(0))],
                unit(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![tb.e(), text("nparams"), box_i64(int(0))],
                unit(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![tb.e(), text("isvararg"), box_bool(bool(true))],
                unit(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![tb.e(), text("istailcall"), box_bool(bool(false))],
                unit(),
            )),
            when(
                is_func(x.e()),
                vec![expr(call(
                    "zl_rawset_str",
                    vec![tb.e(), text("func"), x.e()],
                    unit(),
                ))],
            ),
            ret(box_table(tb.e())),
        ],
    ));
    d.push(define(
        "zl_debug_sethook",
        &[&args],
        unit(),
        vec![ret_void()],
    ));
    d.push(define(
        "zl_debug_gethook",
        &[&args],
        any(),
        vec![ret(nil())],
    ));
    d.push(define("zl_debug_none", &[&args], any(), vec![ret(nil())]));
    d.push(define(
        "zl_debug_getmetatable",
        &[&x],
        any(),
        vec![
            when(
                eq(category(x.e()), int(STR)),
                vec![ret(call("zl_string_metatable", vec![], any()))],
            ),
            when(not(is_table(x.e())), vec![ret(nil())]),
            tb.decl(unbox_table(x.e(), t)),
            when(
                eq(meta_of(tb.e(), t), null(table.clone())),
                vec![ret(nil())],
            ),
            ret(box_table(meta_of(tb.e(), t))),
        ],
    ));
    d.push(define(
        "zl_debug_setmetatable",
        &[&x, &y],
        any(),
        vec![
            when(
                and(not(is_nil(y.e())), not(is_table(y.e()))),
                vec![lua_error(text(
                    "bad argument #2 to 'setmetatable' (nil or table expected)",
                ))],
            ),
            when(
                eq(category(x.e()), int(STR)),
                vec![assign_global("zl_string_meta", y.e()), ret(x.e())],
            ),
            when(
                is_table(x.e()),
                vec![expr(call(
                    "zl_setmetatable",
                    vec![unbox_table(x.e(), t), y.e()],
                    table.clone(),
                ))],
            ),
            ret(x.e()),
        ],
    ));
    d.push(global_var("zl_registry", any()));
    d.push(define(
        "zl_debug_getregistry",
        &[],
        any(),
        vec![
            when(
                is_nil(cached("zl_registry")),
                vec![assign_global(
                    "zl_registry",
                    box_table(call("zl_table_new", vec![], table.clone())),
                )],
            ),
            ret(cached("zl_registry")),
        ],
    ));
    d
}

fn u64() -> Type {
    Type::Primitive(zyntax_typed_ast::PrimitiveType::U64)
}

/// The `os` library above its host side: dates through the table
/// `os.date("*t")` gives and `os.time` takes, and the results a file
/// operation reports.
fn os_declarations(t: &Types) -> Vec<Decl> {
    let table = t.table();
    let anys = t.anys();
    let x = kept("x", any());
    let y = kept("y", any());
    let tb = kept("t", table.clone());
    let key = kept("key", string());
    let default = local("default", i64());
    let delta = local("delta", i64());
    let v = kept("v", any());
    let n = local("n", i64());
    let time = local("time", i64());
    let utc = local("utc", boolean());
    let s = kept("s", string());
    let bad = kept("bad", string());
    let status = local("status", i64());
    let signalled = local("signalled", boolean());
    let how = kept("how", any());
    let mut d = Vec::new();
    // A field of a date table as an integer: `default` when absent,
    // required when that is negative; `delta` is what the C library
    // subtracts, checked to fit.
    d.push(define(
        "zl_date_table_field",
        &[&tb, &key, &default, &delta],
        i64(),
        vec![
            v.decl(call("zl_rawget_str", vec![tb.e(), key.e()], any())),
            when(
                is_nil(v.e()),
                vec![
                    when(
                        lt(default.e(), int(0)),
                        vec![lua_error(concat(vec![
                            text("field '"),
                            key.e(),
                            text("' missing in date table"),
                        ]))],
                    ),
                    ret(default.e()),
                ],
            ),
            y.decl(call("zl_math_tointeger", vec![v.e()], any())),
            when(
                is_nil(y.e()),
                vec![lua_error(concat(vec![
                    text("field '"),
                    key.e(),
                    text("' is not an integer"),
                ]))],
            ),
            n.decl(sub(get_i64(y.e()), delta.e())),
            when(
                or(
                    gt(n.e(), int(i32::MAX as i64)),
                    lt(n.e(), int(i32::MIN as i64)),
                ),
                vec![lua_error(concat(vec![
                    text("field '"),
                    key.e(),
                    text("' is out-of-bound"),
                ]))],
            ),
            ret(get_i64(y.e())),
        ],
    ));
    // The date table's fields set from a time, as `*t` lists them.
    let fields = ["year", "month", "day", "hour", "min", "sec", "wday", "yday"];
    let mut st = Vec::new();
    for (i, name) in fields.iter().enumerate() {
        st.push(expr(call(
            "zl_rawset_str",
            vec![
                tb.e(),
                text(name),
                box_i64(call(
                    "zl_date_field",
                    vec![time.e(), utc.e(), int(i as i64)],
                    i64(),
                )),
            ],
            unit(),
        )));
    }
    st.push(expr(call(
        "zl_rawset_str",
        vec![
            tb.e(),
            text("isdst"),
            box_bool(gt(
                call("zl_date_field", vec![time.e(), utc.e(), int(8)], i64()),
                int(0),
            )),
        ],
        unit(),
    )));
    st.push(ret_void());
    d.push(define(
        "zl_date_table_fill",
        &[&tb, &time, &utc],
        unit(),
        st,
    ));
    // `os.time([t])`: now, or the time a date table names, its fields
    // normalized in place.
    let field = |name: &str, default: i64, delta: i64| {
        call(
            "zl_date_table_field",
            vec![tb.e(), text(name), int(default), int(delta)],
            i64(),
        )
    };
    let isdst = local("isdst", i64());
    d.push(define(
        "zl_os_time_of",
        &[&x],
        i64(),
        vec![
            when(is_nil(x.e()), vec![ret(call("zl_os_time", vec![], i64()))]),
            when(
                not(is_table(x.e())),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'time' (table expected, got "),
                    type_name(x.e()),
                    text(")"),
                ]))],
            ),
            tb.decl(unbox_table(x.e(), t)),
            isdst.decl(int(-1)),
            v.decl(call("zl_rawget_str", vec![tb.e(), text("isdst")], any())),
            when(
                not(is_nil(v.e())),
                vec![isdst.set(if_expr(
                    call("zl_truthy", vec![v.e()], boolean()),
                    int(1),
                    int(0),
                ))],
            ),
            time.decl(call(
                "zl_time_of",
                vec![
                    field("year", -1, 1900),
                    field("month", -1, 1),
                    field("day", -1, 0),
                    field("hour", 12, 0),
                    field("min", 0, 0),
                    field("sec", 0, 0),
                    isdst.e(),
                ],
                i64(),
            )),
            when(
                eq(time.e(), int(-1)),
                vec![lua_error(text(
                    "time result cannot be represented in this installation",
                ))],
            ),
            expr(call(
                "zl_date_table_fill",
                vec![tb.e(), time.e(), bool(false)],
                unit(),
            )),
            ret(time.e()),
        ],
    ));
    // `os.date([format [, time]])`: `!` for UTC, `*t` for a table,
    // otherwise `strftime` with each conversion checked first.
    d.push(define(
        "zl_os_date",
        &[&x, &y],
        any(),
        vec![
            s.decl(if_expr(
                is_nil(x.e()),
                text("%c"),
                call("zl_arg_str", vec![x.e(), bad_arg(1, "date")], string()),
            )),
            time.decl(if_expr(
                is_nil(y.e()),
                call("zl_os_time", vec![], i64()),
                call("zl_arg_int", vec![y.e(), bad_arg(2, "date")], i64()),
            )),
            utc.decl(call("zb_str_startswith", vec![s.e(), text("!")], boolean())),
            when(
                utc.e(),
                vec![s.set(call(
                    "zb_str_substring",
                    vec![s.e(), int(1), call("zb_str_len", vec![s.e()], i64())],
                    string(),
                ))],
            ),
            when(
                eq(
                    call("zl_date_field", vec![time.e(), utc.e(), int(0)], i64()),
                    int(i64::MIN),
                ),
                vec![lua_error(text(
                    "date result cannot be represented in this installation",
                ))],
            ),
            when(
                call("zb_str_startswith", vec![s.e(), text("*t")], boolean()),
                vec![
                    tb.decl(call("zl_table_new", vec![], table.clone())),
                    expr(call(
                        "zl_date_table_fill",
                        vec![tb.e(), time.e(), utc.e()],
                        unit(),
                    )),
                    ret(box_table(tb.e())),
                ],
            ),
            bad.decl(call("zl_date_check", vec![s.e()], string())),
            when(
                ne(bad.e(), null(string())),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'date' (invalid conversion specifier '%"),
                    bad.e(),
                    text("')"),
                ]))],
            ),
            ret(box_str(call(
                "zl_date_raw",
                vec![s.e(), time.e(), utc.e()],
                string(),
            ))),
        ],
    ));
    let a = local("a", i64());
    let b = local("b", i64());
    d.push(define(
        "zl_os_difftime",
        &[&a, &b],
        f64(),
        vec![ret(sub(cast(a.e(), f64()), cast(b.e(), f64())))],
    ));
    d.push(define(
        "zl_os_getenv",
        &[&s],
        any(),
        vec![
            key.decl(call("zl_getenv", vec![s.e()], string())),
            when(eq(key.e(), null(string())), vec![ret(nil())]),
            ret(box_str(key.e())),
        ],
    ));
    d.push(define(
        "zl_os_tmpname",
        &[],
        string(),
        vec![
            key.decl(call("zl_tmpname", vec![], string())),
            when(
                eq(key.e(), null(string())),
                vec![lua_error(text("unable to generate a unique filename"))],
            ),
            ret(key.e()),
        ],
    ));
    // What a file operation answers: true, or nil, the message and
    // the error's number.
    let file_result = |result: Expr| {
        vec![
            status.decl(result),
            when(
                eq(status.e(), int(0)),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(vec![box_bool(bool(true))], anys.clone())],
                    any(),
                ))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(
                    vec![
                        nil(),
                        box_str(call("zl_os_error", vec![], string())),
                        box_i64(status.e()),
                    ],
                    anys.clone(),
                )],
                any(),
            )),
        ]
    };
    d.push(define(
        "zl_os_remove",
        &[&s],
        any(),
        file_result(call("zl_remove", vec![s.e()], i64())),
    ));
    d.push(define(
        "zl_os_rename",
        &[&s, &key],
        any(),
        file_result(call("zl_rename", vec![s.e(), key.e()], i64())),
    ));
    // `os.execute([command])`: whether a shell is there, or how the
    // command ended: true or nil, "exit" or "signal", and the number.
    d.push(define(
        "zl_os_execute",
        &[&x],
        any(),
        vec![
            when(
                is_nil(x.e()),
                vec![ret(box_bool(ne(
                    call("zl_execute", vec![null(string())], i64()),
                    int(0),
                )))],
            ),
            status.decl(call(
                "zl_execute",
                vec![call(
                    "zl_arg_str",
                    vec![x.e(), bad_arg(1, "execute")],
                    string(),
                )],
                i64(),
            )),
            when(
                eq(status.e(), int(-1)),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_str(text("exit")), box_i64(int(-1))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            n.decl(call("zl_exec_result", vec![status.e(), bool(false)], i64())),
            signalled.decl(ne(
                call("zl_exec_result", vec![status.e(), bool(true)], i64()),
                int(0),
            )),
            v.decl(nil()),
            when(
                and(not(signalled.e()), eq(n.e(), int(0))),
                vec![v.set(box_bool(bool(true)))],
            ),
            how.decl(box_str(if_expr(
                signalled.e(),
                text("signal"),
                text("exit"),
            ))),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![v.e(), how.e(), box_i64(n.e())], anys.clone())],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_os_setlocale",
        &[&x, &y],
        any(),
        vec![
            key.decl(call(
                "zl_setlocale",
                vec![if_expr(
                    is_nil(x.e()),
                    null(string()),
                    call("zl_arg_str", vec![x.e(), bad_arg(1, "setlocale")], string()),
                )],
                string(),
            )),
            when(eq(key.e(), null(string())), vec![ret(nil())]),
            ret(box_str(key.e())),
        ],
    ));
    d
}
