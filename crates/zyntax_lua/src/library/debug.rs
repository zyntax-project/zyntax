//! The `debug` library over the call stack a program that uses it
//! keeps.
//!
//! Such a program is lowered to record every call: each function
//! pushes a frame on entry (its key, the caller's line, the call
//! site) and pops it on every way out, restoring the caller's line;
//! each call site stores its number first. The host keeps the frames
//! and what the lowering says of each chunk's functions and sites, and
//! answers `traceback` and `getinfo` from them. Upvalues are read and
//! written through an accessor each chunk hands over, which knows
//! where each function keeps each of them; the locals of a frame
//! through the list it spilled them to at the call it stopped at.
//! Hooks are called from the frame pushes and pops and from each
//! statement's line store.
//!
//! A program that never reaches the library pushes nothing; the one
//! cost it pays is a test of `zl_dbg_on` around a coroutine switch.

use super::*;

/// The call site the next call is made from: its chunk above the low
/// 32 bits, its number below, `TAIL_SITE` set for a tail call; 0 for
/// a call the library makes.
pub const DBG_SITE: &str = "zl_dbg_site";
/// Whether the program keeps the call stack.
pub const DBG_ON: &str = "zl_dbg_on";
/// The events a hook is set for: `HOOK_CALL`, `HOOK_RETURN`,
/// `HOOK_LINE`, `HOOK_COUNT`; 0 for no hook.
pub const HOOK_MASK: &str = "zl_hook_mask";
pub const HOOK_CALL: i64 = 1;
pub const HOOK_RETURN: i64 = 2;
pub const HOOK_LINE: i64 = 4;
pub const HOOK_COUNT: i64 = 8;
/// The bit of a site marking a tail call.
pub const TAIL_SITE: i64 = 1 << 31;
/// The site a hook is called from.
pub const HOOK_SITE: i64 = -2;
/// What an upvalue accessor is asked: the value, a store, the
/// upvalue's identity, a join to another's cell, the cell itself.
pub const UP_GET: i64 = 0;
pub const UP_SET: i64 = 1;
pub const UP_ID: i64 = 2;
pub const UP_JOIN: i64 = 3;
pub const UP_CELL: i64 = 4;
/// How many arguments an upvalue accessor takes: the key, the
/// function, the upvalue's number, a value and what is asked.
#[allow(dead_code)]
pub const UP_ARITY: i64 = 5;

const HOOK_FN: &str = "zl_hook_fn";
const HOOK_COUNT_EVERY: &str = "zl_dbg_hook_every";
const HOOK_LEFT: &str = "zl_hook_left";
const HOOK_BUSY: &str = "zl_hook_busy";
/// Set by a library function's value wrapper for the `debug` function
/// it calls: its argument errors name it as its library's field,
/// `debug.getinfo`, as the reference names a function it finds no
/// call site's name for.
pub const QUALIFY: &str = "zl_dbg_qualify";
/// Whether `debug.setmetatable` ever gave a type other than tables and
/// strings a metatable: until then no such value has one to consult.
pub const TYPE_METAS_ON: &str = "zl_type_metas_on";
/// The hooks of the threads not running, by handle: each a list of the
/// hook, its mask, its count and what is left of the count. The
/// `zl_hook_*` globals hold the running thread's.
const HOOKS: &str = "zl_hooks";
/// The running thread's handle, 0 for the main one.
const HOOK_THREAD: &str = "zl_hook_thread";
/// The last function value made for each of the program's functions,
/// by key: what `getinfo(level, "f")` answers.
const FUNCS: &str = "zl_dbg_funcs";
/// Each chunk's upvalue accessor, by the chunk's number.
const UPS: &str = "zl_dbg_ups";
/// The globals holding the metatables of nil, booleans, numbers,
/// functions and threads, in that order.
const TYPE_METAS: [(&str, &str); 5] = [
    ("nil", "zl_meta_nil"),
    ("boolean", "zl_meta_boolean"),
    ("number", "zl_meta_number"),
    ("function", "zl_meta_function"),
    ("thread", "zl_meta_thread"),
];
/// The line a line hook is told of: nil in a stripped chunk.
fn hook_line(line: Expr) -> Expr {
    if_expr(
        stripped_line(line.clone()),
        nil(),
        box_i64(bitand(line, int((1i64 << LINE_BITS) - 1))),
    )
}

mod hooks;
mod host;
mod info;
mod locals;
mod stack;
mod upvalues;

struct Context {
    anys: Type,
    table: Type,
    x: Local,
    y: Local,
    z: Local,
    f: Local,
    args: Local,
    key: Local,
    site: Local,
    line: Local,
    level: Local,
    thread: Local,
    n: Local,
    i: Local,
    j: Local,
    k: Local,
    kind: Local,
    first: Local,
    what: Local,
    name: Local,
    source: Local,
    short: Local,
    meta: Local,
    event: Local,
    tb: Local,
    lines: Local,
    acc: Local,
}

pub(super) fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let table = t.table();
    let x = kept("x", any());
    let y = kept("y", any());
    let z = kept("z", any());
    let f = kept("f", any());
    let args = kept("args", anys.clone());
    let key = local("key", i64());
    let site = local("site", i64());
    let line = local("line", i64());
    let level = local("level", i64());
    let thread = local("thread", i64());
    let n = local("n", i64());
    let i = local("i", i64());
    let j = local("j", i64());
    let k = local("k", i64());
    let kind = local("kind", i64());
    let first = local("first", i64());
    let what = kept("what", string());
    let name = kept("name", string());
    let source = kept("source", string());
    let short = kept("short", string());
    let meta = kept("meta", string());
    let event = kept("event", string());
    let tb = kept("t", table.clone());
    let lines = kept("lines", table.clone());
    let acc = kept("acc", any());
    let mut d = vec![
        global_var(DBG_SITE, i64()),
        global_var(DBG_ON, boolean()),
        global_var(HOOK_MASK, i64()),
        global_var(HOOK_FN, any()),
        global_var(HOOK_COUNT_EVERY, i64()),
        global_var(HOOK_LEFT, i64()),
        global_var(HOOK_BUSY, boolean()),
        global_var(HOOKS, any()),
        global_var(QUALIFY, boolean()),
        global_var(TYPE_METAS_ON, boolean()),
        global_var(HOOK_THREAD, i64()),
        global_var(FUNCS, any()),
        global_var(UPS, any()),
    ];
    let context = Context {
        anys,
        table,
        x,
        y,
        z,
        f,
        args,
        key,
        site,
        line,
        level,
        thread,
        n,
        i,
        j,
        k,
        kind,
        first,
        what,
        name,
        source,
        short,
        meta,
        event,
        tb,
        lines,
        acc,
    };
    host::append(&context, t, &mut d);
    stack::append(&context, t, &mut d);
    info::append(&context, t, &mut d);
    upvalues::append(&context, t, &mut d);
    locals::append(&context, t, &mut d);
    hooks::append(&context, t, &mut d);
    d
}
