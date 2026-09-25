//! The `debug` library's host side: the stack of calls a program that
//! uses the library keeps, what each of its functions is (lines,
//! parameters, upvalue and local names) and what each call site calls
//! it, and the tracebacks and `getinfo` records made from them.
//! Reached from the library as `$Lua$dbg_…` symbols.
//!
//! A program that never reaches the library registers nothing and
//! pushes no frame; every entry point here then answers as for an
//! empty stack.

use std::cell::RefCell;
use std::collections::HashMap;

use zrtl::StringPtr;

unsafe fn bytes_of(s: zrtl::StringConstPtr) -> &'static [u8] {
    if s.is_null() {
        return &[];
    }
    unsafe { zrtl::string_as_bytes(s) }
}

fn string_out(s: &str) -> StringPtr {
    zrtl::string::string_from_bytes(s.as_bytes())
}

/// A frame's key: the chunk's number above the low 32 bits, the
/// function's number below. A call into the library has none.
const C_KEY: i64 = -1;
/// The bit of a call site's number marking a tail call.
pub const TAIL_SITE: i64 = crate::library::debug::TAIL_SITE;
/// The site a hook is called from.
const HOOK_SITE: i64 = crate::library::debug::HOOK_SITE;

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
    /// The traceback of the last error raised.
    raised: String,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// The name a chunk `load` compiles was given, raw: the reference's
/// `source`.
pub(crate) fn note_load_source(index: i64, raw: String) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if s.active {
            s.load_sources.insert(index, raw);
        }
    });
}

/// A chunk's functions and call sites, as the lowering wrote them: one
/// record per line, fields separated by `\x1f`.
///
/// `F fid line last nparams vararg up1 up2 …`, `A fid line…` (active
/// lines), `S site namewhat name global`, `L fid site name…` (the
/// locals a frame of `fid` stopped at `site` holds).
pub(crate) extern "C" fn host_dbg_chunk(
    index: i64,
    source: zrtl::StringConstPtr,
    short_src: zrtl::StringConstPtr,
    meta: zrtl::StringConstPtr,
) {
    let (source, short_src, meta) =
        unsafe { (bytes_of(source), bytes_of(short_src), bytes_of(meta)) };
    let mut chunk = ChunkMeta {
        source: String::from_utf8_lossy(source).into_owned(),
        short_src: String::from_utf8_lossy(short_src).into_owned(),
        ..Default::default()
    };
    let meta = String::from_utf8_lossy(meta);
    let int = |s: Option<&str>| s.and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
    for record in meta.split('\n') {
        let mut fields = record.split('\x1f');
        match fields.next() {
            Some("F") => {
                let fid = int(fields.next());
                let f = chunk.funcs.entry(fid).or_default();
                f.line = int(fields.next());
                f.last_line = int(fields.next());
                f.nparams = int(fields.next());
                f.vararg = fields.next() == Some("1");
                f.upvalues = fields.map(str::to_string).collect();
            }
            Some("A") => {
                let fid = int(fields.next());
                let f = chunk.funcs.entry(fid).or_default();
                f.active = fields.filter_map(|l| l.parse().ok()).collect();
            }
            Some("S") => {
                let site = int(fields.next());
                let namewhat = fields.next().unwrap_or("").to_string();
                let name = fields.next().unwrap_or("").to_string();
                let global = fields.next().unwrap_or("").to_string();
                chunk.sites.insert(
                    site,
                    Site {
                        namewhat,
                        name,
                        global,
                    },
                );
            }
            Some("L") => {
                let fid = int(fields.next());
                let site = int(fields.next());
                let names = fields.map(str::to_string).collect();
                chunk
                    .funcs
                    .entry(fid)
                    .or_default()
                    .locals
                    .insert(site, names);
            }
            _ => {}
        }
    }
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.active = true;
        if let Some(raw) = s.load_sources.remove(&index) {
            chunk.source = raw;
        }
        s.chunks.insert(index, chunk);
    });
}

/// A call entered: `key` names the function (or none, for a call into
/// the library), `line` is the caller's, `site` the call's.
pub(crate) extern "C" fn host_dbg_enter(key: i64, line: i64, site: i64) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        // A tail call replaces its caller's frame only when it calls a
        // function of the program.
        let tail = key != C_KEY && site >= 0 && site & TAIL_SITE != 0;
        if tail && let Some(caller) = s.frames.last_mut() {
            caller.replaced = true;
        }
        s.frames.push(Frame {
            key,
            entry_line: line,
            site: if site < 0 { site } else { site & !TAIL_SITE },
            tail,
            replaced: false,
            hook_line: -1,
            spill: 0,
            spill_site: 0,
            set_mask: 0,
        });
    });
}

/// The call left: the line its caller was at.
pub(crate) extern "C" fn host_dbg_leave() -> i64 {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.frames.pop().map_or(0, |f| f.entry_line)
    })
}

/// The running thread becomes `to` (0 for the main one); the one left
/// stopped at `line`.
pub(crate) extern "C" fn host_dbg_switch(to: i64, line: i64) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if !s.active {
            return;
        }
        let frames = std::mem::take(&mut s.frames);
        let from = s.current;
        s.parked.insert(from, Parked { frames, line });
        s.frames = s.parked.remove(&to).map(|p| p.frames).unwrap_or_default();
        s.current = to;
    });
}

/// A thread that is done: its stack goes.
pub(crate) extern "C" fn host_dbg_drop(handle: i64) {
    STATE.with(|s| {
        s.borrow_mut().parked.remove(&handle);
    });
}

/// A statement at `line` starts in the running frame: whether a line
/// hook is told, as it is of a new line or of any line reached again
/// by a jump back (`back`).
pub(crate) extern "C" fn host_dbg_line(line: i64, back: bool) -> bool {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let Some(top) = s.frames.last_mut() else {
            return true;
        };
        let fire = back || top.hook_line != line;
        top.hook_line = line;
        fire
    })
}

/// The last line the running frame told a line hook of is `line`.
pub(crate) extern "C" fn host_dbg_set_line(line: i64) {
    STATE.with(|s| {
        if let Some(top) = s.borrow_mut().frames.last_mut() {
            top.hook_line = line;
        }
    });
}

/// The next statement of the running frame is told to a line hook
/// whatever its line: a loop went back.
pub(crate) extern "C" fn host_dbg_back() {
    STATE.with(|s| {
        if let Some(top) = s.borrow_mut().frames.last_mut() {
            top.hook_line = -1;
        }
    });
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

/// The stack of `thread` (-1 for the running one) from its top down,
/// level 0 first: the library function asking (`asking`) for the
/// running thread, the `yield` a suspended coroutine stopped in, and
/// the host below the main chunk.
fn levels(
    s: &State,
    thread: i64,
    line: i64,
    asking: (&'static str, &'static str, &'static str),
) -> Vec<Level> {
    let running = thread < 0 || thread == s.current;
    let (frames, top_line) = if running {
        (&s.frames[..], line)
    } else {
        match s.parked.get(&thread) {
            Some(p) => (&p.frames[..], p.line),
            None => (&[][..], 0),
        }
    };
    let mut out = Vec::new();
    if running {
        out.push(Level {
            key: C_KEY,
            site: 0,
            tail: false,
            line: -1,
            c_name: Some(asking),
        });
    } else if !frames.is_empty() {
        out.push(Level {
            key: C_KEY,
            site: 0,
            tail: false,
            line: -1,
            c_name: Some(("yield", "field", "coroutine.yield")),
        });
    }
    for (k, f) in frames.iter().enumerate().rev() {
        if f.replaced {
            continue;
        }
        let line = match frames.get(k + 1) {
            Some(callee) => callee.entry_line,
            None => top_line,
        };
        out.push(Level {
            key: f.key,
            site: f.site,
            tail: f.tail,
            line: if f.key == C_KEY { -1 } else { line },
            c_name: None,
        });
    }
    let main_thread = if running { s.current == 0 } else { thread == 0 };
    if main_thread {
        out.push(Level {
            key: C_KEY,
            site: 0,
            tail: false,
            line: -1,
            c_name: Some(("", "", "")),
        });
    }
    out
}

fn line_part(line: i64) -> i64 {
    line & 0xffff_ffff
}

fn chunk_part(key_or_line: i64) -> i64 {
    key_or_line >> 32
}

/// What a level's function is: its chunk and record, or none for the
/// library's.
fn func_of(s: &State, key: i64) -> Option<(&ChunkMeta, &FuncMeta)> {
    if key == C_KEY {
        return None;
    }
    let chunk = s.chunks.get(&chunk_part(key))?;
    let f = chunk.funcs.get(&(key & 0xffff_ffff))?;
    Some((chunk, f))
}

fn site_of(s: &State, site: i64) -> Option<&Site> {
    if site == 0 {
        return None;
    }
    s.chunks
        .get(&chunk_part(site))?
        .sites
        .get(&(site & 0xffff_ffff))
}

/// A level's name and what kind of name it is, as `getinfo` gives them.
fn name_of(s: &State, level: &Level) -> (String, String, String) {
    if let Some((name, namewhat, global)) = level.c_name {
        return (name.to_string(), namewhat.to_string(), global.to_string());
    }
    if level.tail {
        return Default::default();
    }
    if level.site == HOOK_SITE {
        return ("?".to_string(), "hook".to_string(), String::new());
    }
    match site_of(s, level.site) {
        Some(site) => (
            site.name.clone(),
            site.namewhat.clone(),
            site.global.clone(),
        ),
        None => Default::default(),
    }
}

fn is_main(key: i64) -> bool {
    key != C_KEY && key & 0xffff_ffff == 0
}

/// One line of a traceback: where the level is, and what it is.
fn traceback_line(s: &State, level: &Level) -> String {
    let (name, namewhat, global) = name_of(s, level);
    let func = func_of(s, level.key);
    let place = match func {
        Some((chunk, _)) if line_part(level.line) > 0 => {
            format!("{}:{}:", chunk.short_src, line_part(level.line))
        }
        Some((chunk, _)) => format!("{}:", chunk.short_src),
        None => "[C]:".to_string(),
    };
    let what = if !global.is_empty() {
        format!("function '{global}'")
    } else if !namewhat.is_empty() {
        format!("{namewhat} '{name}'")
    } else if is_main(level.key) {
        "main chunk".to_string()
    } else if let Some((chunk, f)) = func {
        format!("function <{}:{}>", chunk.short_src, f.line)
    } else {
        "?".to_string()
    };
    let mut out = format!("\n\t{place} in {what}");
    if level.tail {
        out.push_str("\n\t(...tail calls...)");
    }
    out
}

/// `debug.traceback`: `message` (when given) then the stack of
/// `thread` from `level` down, as `luaL_traceback` writes it.
pub(crate) extern "C" fn host_dbg_traceback(
    message: zrtl::StringConstPtr,
    has_message: bool,
    level: i64,
    thread: i64,
    line: i64,
) -> StringPtr {
    let message = unsafe { bytes_of(message) };
    STATE.with(|s| {
        let s = s.borrow();
        let levels = levels(&s, thread, line, ("traceback", "field", "debug.traceback"));
        let mut out = String::new();
        if has_message {
            out.push_str(&String::from_utf8_lossy(message));
            out.push('\n');
        }
        out.push_str("stack traceback:");
        let start = level.max(0) as usize;
        let shown: Vec<&Level> = levels.iter().skip(start).collect();
        // Past this many levels, the first ten and the last eleven.
        const FIRST: usize = 10;
        const LAST: usize = 11;
        if shown.len() > FIRST + LAST {
            for l in &shown[..FIRST] {
                out.push_str(&traceback_line(&s, l));
            }
            let skipped = shown.len() - FIRST - LAST;
            out.push_str(&format!("\n\t...\t(skipping {skipped} levels)"));
            for l in &shown[shown.len() - LAST..] {
                out.push_str(&traceback_line(&s, l));
            }
        } else {
            for l in shown {
                out.push_str(&traceback_line(&s, l));
            }
        }
        string_out(&out)
    })
}

/// The traceback of the running thread's stack, from its top, as the
/// standalone interpreter reports an error nothing catches.
fn traceback_from_top(s: &State, line: i64) -> String {
    let mut out = String::from("stack traceback:");
    for l in levels(s, -1, line, ("", "", "")).iter().skip(1) {
        out.push_str(&traceback_line(s, l));
    }
    out
}

/// An error was raised at `line`: where it was, kept for the report
/// should nothing catch it.
pub(crate) extern "C" fn host_dbg_note_raise(line: i64) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if s.active {
            let text = traceback_from_top(&s, line);
            s.raised = text;
        }
    });
}

/// The traceback of the last raise, for the report of an error nothing
/// caught; none when the program keeps no call stack.
pub(crate) fn uncaught_traceback() -> Option<String> {
    STATE.with(|s| {
        let s = s.borrow();
        s.active.then(|| s.raised.clone())
    })
}

/// The traceback noted at the last raise.
pub(crate) extern "C" fn host_dbg_raised() -> StringPtr {
    STATE.with(|s| string_out(&s.borrow().raised))
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

/// Whether `what` is a valid option string: 0 when it is, 1 for an
/// unknown option, 2 for `>`, which only the C API takes.
pub(crate) extern "C" fn host_dbg_check_options(what: zrtl::StringConstPtr) -> i64 {
    let what = unsafe { bytes_of(what) };
    if what.first() == Some(&b'>') {
        return 2;
    }
    if what.iter().all(|c| b"SlnrutfL".contains(c)) {
        0
    } else {
        1
    }
}

fn fill(s: &State, what: &[u8], key: i64, level: Option<&Level>) -> Info {
    let func = func_of(s, key);
    let mut fields: Vec<(&'static str, Value)> = Vec::new();
    for c in what {
        match c {
            b'S' => match func {
                Some((chunk, f)) => {
                    let main = is_main(key);
                    fields.push(("source", Value::Str(chunk.source.clone())));
                    fields.push(("short_src", Value::Str(chunk.short_src.clone())));
                    fields.push(("linedefined", Value::Int(if main { 0 } else { f.line })));
                    fields.push((
                        "lastlinedefined",
                        Value::Int(if main { 0 } else { f.last_line }),
                    ));
                    fields.push((
                        "what",
                        Value::Str(if main { "main" } else { "Lua" }.to_string()),
                    ));
                }
                None => {
                    fields.push(("source", Value::Str("=[C]".to_string())));
                    fields.push(("short_src", Value::Str("[C]".to_string())));
                    fields.push(("linedefined", Value::Int(-1)));
                    fields.push(("lastlinedefined", Value::Int(-1)));
                    fields.push(("what", Value::Str("C".to_string())));
                }
            },
            b'l' => {
                let line = match (func, level) {
                    (Some(_), Some(l)) if l.line > 0 => line_part(l.line),
                    _ => -1,
                };
                fields.push(("currentline", Value::Int(line)));
            }
            b'u' => match func {
                Some((_, f)) => {
                    fields.push(("nups", Value::Int(f.upvalues.len() as i64)));
                    fields.push(("nparams", Value::Int(f.nparams)));
                    fields.push(("isvararg", Value::Bool(f.vararg)));
                }
                None => {
                    fields.push(("nups", Value::Int(0)));
                    fields.push(("nparams", Value::Int(0)));
                    fields.push(("isvararg", Value::Bool(true)));
                }
            },
            b'n' => {
                let (name, namewhat, _) = match level {
                    Some(l) => name_of(s, l),
                    None => Default::default(),
                };
                if namewhat.is_empty() {
                    fields.push(("name", Value::Nil));
                } else {
                    fields.push(("name", Value::Str(name)));
                }
                fields.push(("namewhat", Value::Str(namewhat)));
            }
            b't' => {
                fields.push(("istailcall", Value::Bool(level.is_some_and(|l| l.tail))));
            }
            b'r' => {
                fields.push(("ftransfer", Value::Int(0)));
                fields.push(("ntransfer", Value::Int(0)));
            }
            b'L' => match func {
                Some((_, f)) => fields.push(("activelines", Value::Lines(f.active.clone()))),
                None => fields.push(("activelines", Value::Nil)),
            },
            b'f' => fields.push(("func", Value::Func)),
            _ => {}
        }
    }
    Info {
        fields,
        key: func.map(|_| key),
    }
}

/// `getinfo` of stack level `level` of `thread`: 1 and the record
/// ready, or 0 when there is no such level.
pub(crate) extern "C" fn host_dbg_info_level(
    level: i64,
    thread: i64,
    line: i64,
    what: zrtl::StringConstPtr,
) -> i64 {
    let what = unsafe { bytes_of(what) };
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let levels = levels(&s, thread, line, ("getinfo", "field", "debug.getinfo"));
        let Some(l) = usize::try_from(level).ok().and_then(|i| levels.get(i)) else {
            return 0;
        };
        let info = fill(&s, what, l.key, Some(l));
        s.info = info;
        1
    })
}

/// `getinfo` of a function: `key` names one of the program's, -1 one
/// of the library's.
pub(crate) extern "C" fn host_dbg_info_func(key: i64, what: zrtl::StringConstPtr) {
    let what = unsafe { bytes_of(what) };
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let info = fill(&s, what, key, None);
        s.info = info;
    });
}

/// How many fields the record has.
pub(crate) extern "C" fn host_dbg_info_count() -> i64 {
    STATE.with(|s| s.borrow().info.fields.len() as i64)
}

pub(crate) extern "C" fn host_dbg_info_name(i: i64) -> StringPtr {
    STATE.with(|s| {
        let s = s.borrow();
        string_out(s.info.fields.get(i as usize).map_or("", |f| f.0))
    })
}

/// A field's kind: 0 nil, 1 integer, 2 string, 3 boolean, 4 the
/// function, 5 the active lines.
pub(crate) extern "C" fn host_dbg_info_kind(i: i64) -> i64 {
    STATE.with(|s| {
        let s = s.borrow();
        match s.info.fields.get(i as usize).map(|f| &f.1) {
            Some(Value::Int(_)) => 1,
            Some(Value::Str(_)) => 2,
            Some(Value::Bool(_)) => 3,
            Some(Value::Func) => 4,
            Some(Value::Lines(_)) => 5,
            _ => 0,
        }
    })
}

pub(crate) extern "C" fn host_dbg_info_int(i: i64) -> i64 {
    STATE.with(|s| {
        let s = s.borrow();
        match s.info.fields.get(i as usize).map(|f| &f.1) {
            Some(Value::Int(v)) => *v,
            Some(Value::Bool(b)) => *b as i64,
            Some(Value::Lines(lines)) => lines.len() as i64,
            _ => 0,
        }
    })
}

pub(crate) extern "C" fn host_dbg_info_str(i: i64) -> StringPtr {
    STATE.with(|s| {
        let s = s.borrow();
        match s.info.fields.get(i as usize).map(|f| &f.1) {
            Some(Value::Str(v)) => string_out(v),
            _ => string_out(""),
        }
    })
}

/// The `j`th active line of field `i`.
pub(crate) extern "C" fn host_dbg_info_line(i: i64, j: i64) -> i64 {
    STATE.with(|s| {
        let s = s.borrow();
        match s.info.fields.get(i as usize).map(|f| &f.1) {
            Some(Value::Lines(lines)) => lines.get(j as usize).copied().unwrap_or(0),
            _ => 0,
        }
    })
}

/// The key of the function the record is about: -1 for the library's.
pub(crate) extern "C" fn host_dbg_info_key() -> i64 {
    STATE.with(|s| s.borrow().info.key.unwrap_or(C_KEY))
}

// ─── upvalues and locals ────────────────────────────────────────────

/// The name of upvalue `n` (from 1) of the function `key` names; empty
/// when it has no such upvalue.
pub(crate) extern "C" fn host_dbg_upvalue_name(key: i64, n: i64) -> StringPtr {
    STATE.with(|s| {
        let s = s.borrow();
        let name = func_of(&s, key)
            .and_then(|(_, f)| usize::try_from(n - 1).ok().and_then(|i| f.upvalues.get(i)));
        string_out(name.map_or("", String::as_str))
    })
}

/// The stack depth (from 1, bottom up) of stack level `level` of the
/// running thread, as `getlocal` numbers levels; 0 when the level is
/// none of the program's frames, -1 when there is no such level.
pub(crate) extern "C" fn host_dbg_frame_depth(level: i64) -> i64 {
    STATE.with(|s| {
        let s = s.borrow();
        if level < 1 {
            return if level == 0 { 0 } else { -1 };
        }
        let mut seen = 0;
        for (k, f) in s.frames.iter().enumerate().rev() {
            if f.replaced {
                continue;
            }
            seen += 1;
            if seen == level {
                return if f.key == C_KEY { 0 } else { k as i64 + 1 };
            }
        }
        // The host below the main chunk.
        if s.current == 0 && seen + 1 == level {
            return 0;
        }
        -1
    })
}

/// The running frame spilled its locals to `list` (boxed) before the
/// call at `site`.
pub(crate) extern "C" fn host_dbg_spill(list: *const zrtl::DynamicBox, site: i64) {
    STATE.with(|s| {
        if let Some(top) = s.borrow_mut().frames.last_mut() {
            top.spill = list as usize;
            top.spill_site = site & 0xffff_ffff;
        }
    });
}

/// The call the running frame spilled its locals for returned: the
/// list is no longer its. The slots `debug.setlocal` wrote meanwhile,
/// a bit each, for the frame to read back.
pub(crate) extern "C" fn host_dbg_unspill(_list: *const zrtl::DynamicBox) -> i64 {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let Some(top) = s.frames.last_mut() else {
            return 0;
        };
        top.spill = 0;
        std::mem::take(&mut top.set_mask)
    })
}

/// `debug.setlocal` wrote slot `i` (from 0) of the spill list of the
/// frame at stack depth `depth`.
pub(crate) extern "C" fn host_dbg_mark_set(depth: i64, i: i64) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(f) = usize::try_from(depth - 1)
            .ok()
            .and_then(|k| s.frames.get_mut(k))
            && (0..63).contains(&i)
        {
            f.set_mask |= 1 << i;
        }
    });
}

/// The boxed list the frame at stack depth `depth` spilled its locals
/// to, as a word (0 for none).
pub(crate) extern "C" fn host_dbg_frame_spill(depth: i64) -> i64 {
    STATE.with(|s| {
        let s = s.borrow();
        usize::try_from(depth - 1)
            .ok()
            .and_then(|k| s.frames.get(k))
            .map_or(0, |f| f.spill as i64)
    })
}

/// The names of the locals the frame at stack depth `depth` holds
/// where it stopped: the site it spilled them at, or its entry (its
/// parameters) when it spilled none.
fn local_names(s: &State, depth: i64) -> Option<&Vec<String>> {
    let frame = usize::try_from(depth - 1)
        .ok()
        .and_then(|k| s.frames.get(k))?;
    let (_, f) = func_of(s, frame.key)?;
    let site = if frame.spill == 0 {
        0
    } else {
        frame.spill_site
    };
    f.locals.get(&site)
}

/// The name of local `n` (from 1) of the frame at stack depth `depth`:
/// empty when it has no such local.
pub(crate) extern "C" fn host_dbg_local_name(depth: i64, n: i64) -> StringPtr {
    STATE.with(|s| {
        let s = s.borrow();
        let name = local_names(&s, depth)
            .and_then(|names| usize::try_from(n - 1).ok().and_then(|i| names.get(i)));
        string_out(name.map_or("", String::as_str))
    })
}

/// How many named locals the frame at stack depth `depth` holds.
pub(crate) extern "C" fn host_dbg_local_count(depth: i64) -> i64 {
    STATE.with(|s| local_names(&s.borrow(), depth).map_or(0, |n| n.len() as i64))
}

/// The name of parameter `n` (from 1) of the function `key` names:
/// empty when it has no such parameter.
pub(crate) extern "C" fn host_dbg_param_name(key: i64, n: i64) -> StringPtr {
    STATE.with(|s| {
        let s = s.borrow();
        let name = func_of(&s, key).and_then(|(_, f)| {
            let params = f.locals.get(&0)?;
            let i = usize::try_from(n - 1).ok()?;
            (i < f.nparams as usize).then(|| params.get(i)).flatten()
        });
        string_out(name.map_or("", String::as_str))
    })
}

/// The key of the function running at stack depth `depth`.
pub(crate) extern "C" fn host_dbg_frame_key(depth: i64) -> i64 {
    STATE.with(|s| {
        let s = s.borrow();
        usize::try_from(depth - 1)
            .ok()
            .and_then(|k| s.frames.get(k))
            .map_or(C_KEY, |f| f.key)
    })
}

/// The host symbols of the debug library.
pub(crate) static SYMBOLS: [zrtl::ZrtlSymbol; 31] = [
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
    zrtl::ZrtlSymbol::new(c"$Lua$dbg_raised".as_ptr(), host_dbg_raised as *const u8),
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
        c"$Lua$dbg_param_name".as_ptr(),
        host_dbg_param_name as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$dbg_frame_key".as_ptr(),
        host_dbg_frame_key as *const u8,
    ),
];
