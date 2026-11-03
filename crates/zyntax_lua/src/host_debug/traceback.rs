//! Traceback support.

use super::*;

/// The stack of `thread` (-1 for the running one) from its top down,
/// level 0 first: the library function asking (`asking`) for the
/// running thread, the `yield` a suspended coroutine stopped in, and
/// the host below the main chunk.
pub(super) fn levels(
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
    } else if !frames.is_empty() && !s.parked.get(&thread).is_some_and(|p| p.dead) {
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

/// A stored line's line in its chunk: -1 in a stripped chunk.
pub(super) fn line_part(line: i64) -> i64 {
    match line & crate::library::STRIPPED_LINE {
        crate::library::STRIPPED_LINE => -1,
        l => l,
    }
}

pub(super) fn chunk_part(key_or_line: i64) -> i64 {
    key_or_line >> 32
}

/// What a level's function is: its chunk and record, or none for the
/// library's.
pub(super) fn func_of(s: &State, key: i64) -> Option<(&ChunkMeta, &FuncMeta)> {
    if key == C_KEY {
        return None;
    }
    let chunk = s.chunks.get(&chunk_part(key))?;
    let f = chunk.funcs.get(&(key & 0xffff_ffff))?;
    Some((chunk, f))
}

pub(super) fn site_of(s: &State, site: i64) -> Option<&Site> {
    if site == 0 {
        return None;
    }
    s.chunks
        .get(&chunk_part(site))?
        .sites
        .get(&(site & 0xffff_ffff))
}

/// A level's name and what kind of name it is, as `getinfo` gives them.
pub(super) fn name_of(s: &State, level: &Level) -> (String, String, String) {
    if let Some((name, namewhat, global)) = level.c_name {
        return (name.to_string(), namewhat.to_string(), global.to_string());
    }
    if level.tail {
        return Default::default();
    }
    if level.site == HOOK_SITE {
        return ("?".to_string(), "hook".to_string(), String::new());
    }
    if level.site <= MM_SITE {
        let event = usize::try_from(MM_SITE - level.site)
            .ok()
            .and_then(|i| MM_EVENTS.get(i));
        if let Some(event) = event {
            return (event.to_string(), "metamethod".to_string(), String::new());
        }
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

pub(super) fn is_main(key: i64) -> bool {
    key != C_KEY && key & 0xffff_ffff == 0
}

/// One line of a traceback: where the level is, and what it is.
pub(super) fn traceback_line(s: &State, level: &Level) -> String {
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
            out.push_str(&crate::source_text(message));
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
pub(super) fn traceback_from_top(s: &State, line: i64) -> String {
    let mut out = String::from("stack traceback:");
    for l in levels(s, -1, line, ("", "", "")).iter().skip(1) {
        out.push_str(&traceback_line(s, l));
    }
    out
}

/// An error was raised at `line`: the stack it was raised on is kept.
pub(crate) extern "C" fn host_dbg_note_raise(line: i64) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if s.active {
            let State {
                raise,
                frames,
                current,
                ..
            } = &mut *s;
            raise.frames.clear();
            raise.frames.extend_from_slice(frames);
            raise.line = line;
            raise.thread = *current;
        }
    });
}

/// The traceback of the stack the last error was raised on, for the
/// report of an error nothing caught; none when the program keeps no
/// call stack.
pub(crate) fn uncaught_traceback() -> Option<String> {
    STATE.with(|s| s.borrow().active).then(raised_traceback)
}

/// The traceback of the stack the last error was raised on.
pub(super) fn raised_traceback() -> String {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let raised = unwound(&s.raise.frames);
        let line = s.raise.line;
        let running = std::mem::replace(&mut s.frames, raised);
        let text = traceback_from_top(&s, line);
        s.frames = running;
        text
    })
}

/// A message handler is about to run for the error last raised: the
/// frames the error unwound, above the running ones, are put back so
/// the handler runs where the error was raised. The depth to cut the
/// stack back to afterwards, or -1 when nothing was put back.
pub(crate) extern "C" fn host_dbg_handler_enter() -> i64 {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let depth = s.frames.len();
        let fits = s.active
            && s.raise.thread == s.current
            && s.raise.frames.len() > depth
            && s.raise.frames[..depth]
                .iter()
                .zip(&s.frames)
                .all(|(a, b)| a.key == b.key && a.entry_line == b.entry_line);
        if !fits {
            return -1;
        }
        let above = unwound(&s.raise.frames[depth..]);
        s.frames.extend(above);
        depth as i64
    })
}

/// The site of a call of the metamethod for `event`.
pub(crate) extern "C" fn host_dbg_mm_site(event: zrtl::StringConstPtr) -> i64 {
    let event = unsafe { bytes_of(event) };
    MM_EVENTS
        .iter()
        .position(|e| e.as_bytes() == event)
        .map_or(0, |i| MM_SITE - i as i64)
}

/// The line the last error was raised at.
pub(crate) extern "C" fn host_dbg_raise_line() -> i64 {
    STATE.with(|s| s.borrow().raise.line)
}

/// The handler returned: the stack is cut back to `depth`.
pub(crate) extern "C" fn host_dbg_handler_leave(depth: i64) {
    STATE.with(|s| {
        if let Ok(depth) = usize::try_from(depth) {
            s.borrow_mut().frames.truncate(depth);
        }
    });
}
