//! The `debug` library's host side: the stack of calls a program that
//! uses the library keeps, what each of its functions is (lines,
//! parameters, upvalue and local names) and what each call site calls
//! it, and the tracebacks and `getinfo` records made from them.
//! Reached from the library as `$Lua$dbg_…` symbols.
//!
//! A program that never reaches the library registers nothing and
//! pushes no frame; every entry point here then answers as for an
//! empty stack.

mod frames;
mod info;
mod locals;
mod traceback;

pub(crate) use frames::*;
pub(crate) use info::*;
pub(crate) use locals::*;
pub(crate) use traceback::*;

use std::cell::RefCell;
use std::collections::HashMap;

use zrtl::StringPtr;

/// A frame's key: the chunk's number above the low 32 bits, the
/// function's number below. A call into the library has none.
const C_KEY: i64 = -1;
/// The bit of a call site's number marking a tail call.
pub const TAIL_SITE: i64 = crate::library::debug::TAIL_SITE;
/// The site a hook is called from.
const HOOK_SITE: i64 = crate::library::debug::HOOK_SITE;
/// The sites of metamethod calls: this less the event's place in
/// `MM_EVENTS`.
const MM_SITE: i64 = -1000;
/// The events a metamethod call site names, by place.
const MM_EVENTS: &[&str] = &[
    "index", "newindex", "call", "len", "eq", "lt", "le", "concat", "unm", "add", "sub", "mul",
    "div", "mod", "pow", "idiv", "band", "bor", "bxor", "shl", "shr", "bnot", "close",
];

/// What the lowering records of one of a chunk's functions.
#[derive(Default, Clone)]
struct FuncMeta {
    line: i64,
    last_line: i64,
    nparams: i64,
    vararg: bool,
    upvalues: Vec<String>,
    active: Vec<i64>,
    /// The locals of each call site, by the site's number: what a
    /// frame stopped at that site holds, in order.
    locals: HashMap<i64, Vec<String>>,
    /// The locals of each call site kept in cells, a bit each by slot:
    /// the spill holds their cells.
    cells: HashMap<i64, i64>,
}

/// What a call site calls the function it calls: `local`, `global`,
/// `method`, `field`, `upvalue`, `metamethod` or `for iterator`, and
/// the name; for the library's functions, the name a traceback gives.
#[derive(Default, Clone)]
struct Site {
    namewhat: String,
    name: String,
    global: String,
}

#[derive(Default)]
struct ChunkMeta {
    source: String,
    short_src: String,
    funcs: HashMap<i64, FuncMeta>,
    sites: HashMap<i64, Site>,
}

#[derive(Clone, Copy)]
struct Frame {
    key: i64,
    /// The line running where the frame was entered: its caller's.
    entry_line: i64,
    site: i64,
    /// Entered by a tail call: its caller's frame is gone.
    tail: bool,
    /// Replaced by a function it tail-called.
    replaced: bool,
    /// The last line a line hook was told of.
    hook_line: i64,
    /// The list the frame's locals were spilled to at the call it is
    /// stopped at (boxed; null for none), and that call's site. The
    /// frame's code holds the list until the call returns.
    spill: usize,
    spill_site: i64,
    /// The slots of the spill list `debug.setlocal` wrote, a bit each.
    set_mask: i64,
}

/// A stack the running thread is not on: its frames, and the line its
/// top frame stopped at.
struct Parked {
    frames: Vec<Frame>,
    line: i64,
    /// The thread died of an error: its frames are where it was raised.
    dead: bool,
}

/// The stack of the running thread when an error was last raised,
/// kept for the report should nothing catch it, for the handler
/// `xpcall` runs at the raise, and for a thread the error kills.
#[derive(Default)]
struct Raise {
    thread: i64,
    frames: Vec<Frame>,
    line: i64,
}

#[derive(Default)]
struct State {
    chunks: HashMap<i64, ChunkMeta>,
    /// The raw source names chunks `load` compiled were given.
    load_sources: HashMap<i64, String>,
    frames: Vec<Frame>,
    /// The running thread's handle: 0 for the main one.
    current: i64,
    parked: HashMap<i64, Parked>,
    active: bool,
    /// The record `getinfo` answers from.
    info: Info,
    /// The last error raised.
    raise: Raise,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

// ─── the stack as levels ────────────────────────────────────────────

/// One level of a stack as `getinfo` and a traceback see it.
#[derive(Clone)]
struct Level {
    key: i64,
    site: i64,
    tail: bool,
    line: i64,
    /// The name a library function is known by, for a level made up
    /// rather than pushed: the debug function asked, a coroutine's
    /// `yield`, the host at the bottom.
    c_name: Option<(&'static str, &'static str, &'static str)>,
}

// ─── getinfo ────────────────────────────────────────────────────────

/// A field of a `getinfo` record.
#[derive(Clone)]
enum Value {
    Nil,
    Int(i64),
    Str(String),
    Bool(bool),
    /// The function itself: the library fills it in.
    Func,
    /// The active lines, as a table: the library builds it.
    Lines(Vec<i64>),
}

#[derive(Default)]
struct Info {
    fields: Vec<(&'static str, Value)>,
    /// The key of the function the record is about, when one of the
    /// program's.
    key: Option<i64>,
}

// ─── upvalues and locals ────────────────────────────────────────────

/// The host symbols of the debug library.
pub(crate) static SYMBOLS: [zrtl::ZrtlSymbol; 36] = [
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_chunk".as_ptr(), host_dbg_chunk as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_enter".as_ptr(), host_dbg_enter as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_leave".as_ptr(), host_dbg_leave as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_switch".as_ptr(), host_dbg_switch as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_drop".as_ptr(), host_dbg_drop as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_line".as_ptr(), host_dbg_line as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_set_line".as_ptr(),
        host_dbg_set_line as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_back".as_ptr(), host_dbg_back as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_traceback".as_ptr(),
        host_dbg_traceback as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_note_raise".as_ptr(),
        host_dbg_note_raise as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_handler_enter".as_ptr(),
        host_dbg_handler_enter as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_handler_leave".as_ptr(),
        host_dbg_handler_leave as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_raise_line".as_ptr(),
        host_dbg_raise_line as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_new_thread".as_ptr(),
        host_dbg_new_thread as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_mm_site".as_ptr(), host_dbg_mm_site as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_check_options".as_ptr(),
        host_dbg_check_options as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_info_level".as_ptr(),
        host_dbg_info_level as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_info_func".as_ptr(),
        host_dbg_info_func as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_info_count".as_ptr(),
        host_dbg_info_count as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_info_name".as_ptr(),
        host_dbg_info_name as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_info_kind".as_ptr(),
        host_dbg_info_kind as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_info_int".as_ptr(),
        host_dbg_info_int as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_info_str".as_ptr(),
        host_dbg_info_str as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_info_line".as_ptr(),
        host_dbg_info_line as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_info_key".as_ptr(),
        host_dbg_info_key as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_upvalue_name".as_ptr(),
        host_dbg_upvalue_name as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_frame_depth".as_ptr(),
        host_dbg_frame_depth as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_spill".as_ptr(), host_dbg_spill as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_unspill".as_ptr(), host_dbg_unspill as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_mark_set".as_ptr(),
        host_dbg_mark_set as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_frame_spill".as_ptr(),
        host_dbg_frame_spill as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_local_name".as_ptr(),
        host_dbg_local_name as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_local_count".as_ptr(),
        host_dbg_local_count as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_local_cell".as_ptr(),
        host_dbg_local_cell as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_param_name".as_ptr(),
        host_dbg_param_name as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_frame_key".as_ptr(),
        host_dbg_frame_key as *const u8,
    ),
];
