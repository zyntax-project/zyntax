//! Frames support.

use super::*;

pub(super) unsafe fn bytes_of(s: zrtl::StringConstPtr) -> &'static [u8] {
    if s.is_null() {
        return &[];
    }
    unsafe { zrtl::string_as_bytes(s) }
}

/// A string handed back to the program: a stand-in character for a
/// byte that is not UTF-8 (see `source_text`) is that byte again.
pub(super) fn string_out(s: &str) -> StringPtr {
    zrtl::string::string_from_bytes(&crate::source_bytes(s))
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
/// locals a frame of `fid` stopped at `site` holds), `C fid site mask`
/// (which of those are spilled as their cells).
pub(crate) extern "C" fn host_dbg_chunk(
    index: i64,
    source: zrtl::StringConstPtr,
    short_src: zrtl::StringConstPtr,
    meta: zrtl::StringConstPtr,
) {
    let (source, short_src, meta) =
        unsafe { (bytes_of(source), bytes_of(short_src), bytes_of(meta)) };
    let mut chunk = ChunkMeta {
        source: crate::source_text(source).into_owned(),
        short_src: crate::source_text(short_src).into_owned(),
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
            Some("C") => {
                let fid = int(fields.next());
                let site = int(fields.next());
                let mask = int(fields.next());
                chunk.funcs.entry(fid).or_default().cells.insert(site, mask);
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
        s.parked.insert(
            from,
            Parked {
                frames,
                line,
                dead: false,
            },
        );
        s.frames = s.parked.remove(&to).map(|p| p.frames).unwrap_or_default();
        s.current = to;
    });
}

/// A thread that is done: its stack goes, unless an error killed it
/// (`failed`), when its stack is the one the error was raised on.
pub(crate) extern "C" fn host_dbg_drop(handle: i64, failed: bool) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.parked.remove(&handle);
        if failed && s.raise.thread == handle {
            let frames = unwound(&s.raise.frames);
            let line = s.raise.line;
            s.parked.insert(
                handle,
                Parked {
                    frames,
                    line,
                    dead: true,
                },
            );
        }
    });
}

/// A thread was made with handle `handle`: what a dead thread with
/// that handle left goes.
pub(crate) extern "C" fn host_dbg_new_thread(handle: i64) {
    STATE.with(|s| {
        s.borrow_mut().parked.remove(&handle);
    });
}

/// Frames that have returned, as a stack to describe: the lists their
/// locals were spilled to are not theirs any more.
pub(super) fn unwound(frames: &[Frame]) -> Vec<Frame> {
    frames
        .iter()
        .map(|f| Frame {
            spill: 0,
            spill_site: 0,
            set_mask: 0,
            ..*f
        })
        .collect()
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
