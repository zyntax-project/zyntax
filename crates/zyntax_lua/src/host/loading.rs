//! Loading support.

use super::*;

/// The chunk name as `luaO_chunkid` spells it within `LUA_IDSIZE`:
/// `=name` as given, `@name` as given or its end, anything else as
/// `[string "..."]` cut at the first line. The name's bytes are kept:
/// one that is not UTF-8 is carried as its stand-in character (see
/// `source_text`) and is that byte again in every message.
pub(super) fn chunk_name(name: &[u8]) -> String {
    const IDSIZE: usize = 60;
    let text = |b: &[u8]| crate::source_text(b).into_owned();
    if let Some(rest) = name.strip_prefix(b"=") {
        let shown = if name.len() <= IDSIZE {
            rest
        } else {
            &rest[..IDSIZE - 1]
        };
        return text(shown);
    }
    if let Some(rest) = name.strip_prefix(b"@") {
        if name.len() <= IDSIZE {
            return text(rest);
        }
        let keep = IDSIZE - "...".len() - 1;
        return format!("...{}", text(&rest[rest.len() - keep..]));
    }
    let room = IDSIZE - "[string \"".len() - "...\"]".len() - 1;
    let first_line = name.split(|&b| b == b'\n').next().unwrap_or(b"");
    let cut = first_line.len() < name.len() || name.len() >= room;
    let shown = if first_line.len() > room {
        &first_line[..room]
    } else {
        first_line
    };
    let shown = text(shown);
    if cut {
        format!("[string \"{shown}...\"]")
    } else {
        format!("[string \"{shown}\"]")
    }
}

/// `load(source, name)`: the chunk as a function value, or null with
/// the message held for `$Lua$load_error`. A binary chunk (see
/// [`crate::dump`]) is read back; when it holds a function rather than
/// a main chunk, the value is a chunk returning that function, which
/// `$Lua$load_wraps` then says.
pub(super) extern "C" fn host_load(
    source: zrtl::StringConstPtr,
    name: zrtl::StringConstPtr,
    env: *const DynamicBox,
) -> *const DynamicBox {
    let (source, name) = unsafe { (bytes_of(source), bytes_of(name)) };
    let index = LOADS.with(|n| {
        let k = n.get();
        n.set(k + 1);
        LOAD_CHUNKS_FROM + k
    });
    LOAD_WRAPS.with(|w| w.set(false));
    let loaded = if source.first() == Some(&crate::dump::SIGNATURE[0]) {
        load_binary(source, name, index, env)
    } else {
        // The reference's `source`: the name given, else the text itself.
        let raw = if name.is_empty() { source } else { name };
        let raw = crate::source_text(raw).into_owned();
        crate::host_debug::note_load_source(index, raw.clone());
        let text = crate::source_text(source);
        crate::load_chunk(&text, (&chunk_name(name), &raw), index, env, false)
    };
    match loaded {
        Ok(record) => record,
        Err(message) => {
            LOAD_ERROR.with(|e| *e.borrow_mut() = message);
            std::ptr::null()
        }
    }
}

/// A binary chunk compiled from the source it holds, under the source
/// name it holds; refused as `lundump.c` refuses it.
pub(super) fn load_binary(
    bytes: &[u8],
    name: &[u8],
    index: i64,
    env: *const DynamicBox,
) -> Result<*const DynamicBox, String> {
    let undumped = crate::dump::undump(bytes).map_err(|why| {
        format!(
            "{}: bad binary format ({why})",
            crate::dump::error_name(name)
        )
    })?;
    crate::host_debug::note_load_source(index, undumped.source.clone());
    let text = crate::dump::wrapper_text(&undumped, false);
    let short = chunk_name(undumped.source.as_bytes());
    let record = crate::load_chunk(
        &text,
        (&short, &undumped.source),
        index,
        env,
        undumped.stripped,
    )?;
    LOAD_WRAPS.with(|w| w.set(!undumped.main));
    Ok(record)
}

/// Whether the chunk the last load compiled returns the function the
/// binary chunk held, rather than being it.
pub(super) extern "C" fn host_load_wraps() -> bool {
    LOAD_WRAPS.with(|w| w.replace(false))
}

/// `string.dump(f, strip)` of the function with key `key`, or null when
/// no chunk of the program defines it.
pub(super) extern "C" fn host_dump(key: i64, strip: bool) -> StringPtr {
    match crate::dump::dump(key, strip) {
        Some(bytes) => zrtl::string::string_from_bytes(&bytes),
        None => std::ptr::null_mut(),
    }
}

/// `loadfile`/`dofile`: the file's bytes, or null with the message
/// held for `$Lua$load_error`.
pub(super) extern "C" fn host_read_file(path: zrtl::StringConstPtr) -> StringPtr {
    let path = String::from_utf8_lossy(unsafe { bytes_of(path) }).into_owned();
    match std::fs::read(&path) {
        Ok(bytes) => {
            // A byte order mark is skipped, then a leading `#` line, as
            // `lua` skips a shebang; its line break stays for the lines
            // of a text chunk, and goes before a binary one.
            let bytes = bytes
                .strip_prefix(b"\xEF\xBB\xBF")
                .map_or(&bytes[..], |rest| rest)
                .to_vec();
            let bytes = if bytes.first() == Some(&b'#') {
                match bytes.iter().position(|&b| b == b'\n') {
                    Some(nl) if bytes.get(nl + 1) == Some(&crate::dump::SIGNATURE[0]) => {
                        &bytes[nl + 1..]
                    }
                    Some(nl) => &bytes[nl..],
                    None => &[][..],
                }
            } else {
                &bytes[..]
            };
            zrtl::string::string_from_bytes(bytes)
        }
        Err(e) => {
            let (_, reason) = host_os::c_error(&e);
            LOAD_ERROR.with(|err| *err.borrow_mut() = format!("cannot open {path}: {reason}"));
            std::ptr::null_mut()
        }
    }
}

pub(super) extern "C" fn host_load_error() -> StringPtr {
    LOAD_ERROR.with(|e| zrtl::string::string_from_bytes(&crate::source_bytes(&e.borrow())))
}

/// `package.searchpath`: the first template of `path` that names a
/// readable file with `name` (its `sep` turned into `rep`) put in
/// for `?`; null when none does, with every file tried listed as
/// the reference lists them for `load_error`.
pub(super) extern "C" fn host_searchpath(
    name: zrtl::StringConstPtr,
    path: zrtl::StringConstPtr,
    sep: zrtl::StringConstPtr,
    rep: zrtl::StringConstPtr,
) -> StringPtr {
    let (name, path, sep, rep) =
        unsafe { (bytes_of(name), bytes_of(path), bytes_of(sep), bytes_of(rep)) };
    let name = if sep.is_empty() {
        name.to_vec()
    } else {
        replace_all(name, sep, rep)
    };
    // Every piece between separators is a template, the empty ones
    // included: the reference tries and reports them like any other.
    let mut tried = Vec::new();
    for template in path.split(|&b| b == b';') {
        let filename = replace_all(template, b"?", &name);
        let text = String::from_utf8_lossy(&filename).into_owned();
        if std::fs::File::open(&text).is_ok() {
            return zrtl::string::string_from_bytes(&filename);
        }
        tried.push(format!("no file '{text}'"));
    }
    LOAD_ERROR.with(|e| *e.borrow_mut() = tried.join("\n\t"));
    std::ptr::null_mut()
}

pub(super) fn replace_all(s: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if !from.is_empty() && s[i..].starts_with(from) {
            out.extend_from_slice(to);
            i += from.len();
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}
