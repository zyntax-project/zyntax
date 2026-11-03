//! Locals support.

use super::*;

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
            top.spill_site = site & 0xffff_ffff & !TAIL_SITE;
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
pub(super) fn local_names(s: &State, depth: i64) -> Option<&Vec<String>> {
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

/// Whether local `n` (from 1) of the frame at stack depth `depth` was
/// spilled as its cell: 1 when it was, else 0.
pub(crate) extern "C" fn host_dbg_local_cell(depth: i64, n: i64) -> i64 {
    STATE.with(|s| {
        let s = s.borrow();
        let Some(frame) = usize::try_from(depth - 1)
            .ok()
            .and_then(|k| s.frames.get(k))
        else {
            return 0;
        };
        if frame.spill == 0 || !(1..=63).contains(&n) {
            return 0;
        }
        func_of(&s, frame.key)
            .and_then(|(_, f)| f.cells.get(&frame.spill_site))
            .map_or(0, |mask| (mask >> (n - 1)) & 1)
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
