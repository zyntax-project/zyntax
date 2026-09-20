//! What the host hands a program: the arguments it was started with,
//! reached from the library as `$Host$argc` and `$Host$argv`, the
//! clocks, `$Host$time` and `$Host$perf_counter`, byte strings and
//! whole files.
//!
//! A byte string is laid out as a string is, `[i32 length][bytes]`, so
//! the two share storage and release; what differs is that its bytes
//! are not text, so nothing here reads them as UTF-8.

use std::sync::OnceLock;
use zrtl::{StringConstPtr, StringPtr};

static ARGS: OnceLock<Vec<String>> = OnceLock::new();

/// The program's arguments, `sys.argv` in Python's terms: the script's
/// path first. Set once per process; a later call keeps the first.
pub fn set_args(args: Vec<String>) {
    let _ = ARGS.set(args);
}

extern "C" fn host_argc() -> i64 {
    ARGS.get().map_or(0, |args| args.len() as i64)
}

extern "C" fn host_argv(i: i64) -> zrtl::StringPtr {
    match usize::try_from(i)
        .ok()
        .and_then(|i| ARGS.get().and_then(|args| args.get(i)))
    {
        Some(arg) => zrtl::string_new(arg),
        None => std::ptr::null_mut(),
    }
}

/// Seconds since the Unix epoch, as `time.time()` gives them.
extern "C" fn host_time() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Seconds on a clock that only goes forward, from the first reading.
extern "C" fn host_perf_counter() -> f64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
}

/// The one address a call through a function value passes for an
/// argument it leaves out; the function's code replaces it by the
/// default it keeps. Shaped like a box so that reading it by mistake
/// finds None rather than garbage.
static MISSING_ARG: [u64; 6] = [0; 6];

extern "C" fn host_missing_arg() -> *const u8 {
    MISSING_ARG.as_ptr() as *const u8
}

/// The bytes of a blob, empty for null.
unsafe fn blob<'a>(p: StringConstPtr) -> &'a [u8] {
    if p.is_null() {
        return &[];
    }
    // SAFETY: a blob is a length followed by that many bytes.
    unsafe { std::slice::from_raw_parts(zrtl::string::string_data(p), *p as usize) }
}

/// A blob of the one byte `v`.
extern "C" fn host_bytes_of_byte(v: i64) -> StringPtr {
    zrtl::string::string_from_bytes(&[v as u8])
}

/// The box category of a byte string, the tag's low byte.
const BYTES_TAG: u32 = 6;

extern "C" fn drop_blob(p: *mut u8) {
    if !p.is_null() {
        // SAFETY: the box's payload is a blob this module copied.
        unsafe { zrtl::string::string_free(p as StringPtr) }
    }
}

/// Box a byte string: the box owns a copy, as a string's box does, and
/// frees it with itself.
extern "C" fn host_bytes_box(a: StringConstPtr) -> *mut zrtl::DynamicBox {
    // SAFETY: a blob the program holds.
    let copy = unsafe { zrtl::string::string_copy(a) };
    zrtl::DynamicBox {
        tag: zrtl::TypeTag::from_raw(BYTES_TAG),
        size: std::mem::size_of::<*const u8>() as u32,
        data: copy as *mut u8,
        dropper: Some(drop_blob),
        display_fn: None,
    }
    .into_raw()
}

extern "C" fn host_bytes_repeat(a: StringConstPtr, n: i64) -> StringPtr {
    // SAFETY: a blob the program holds.
    let a = unsafe { blob(a) };
    zrtl::string::string_from_bytes(&a.repeat(usize::try_from(n).unwrap_or(0)))
}

/// The byte at `i`, counted from the end when negative; -1 out of range.
extern "C" fn host_bytes_at(a: StringConstPtr, i: i64) -> i64 {
    // SAFETY: a blob the program holds.
    let a = unsafe { blob(a) };
    let i = if i < 0 { i + a.len() as i64 } else { i };
    usize::try_from(i)
        .ok()
        .and_then(|i| a.get(i))
        .map_or(-1, |b| *b as i64)
}

/// The bytes from `start` to `end`, both clamped into the blob.
extern "C" fn host_bytes_slice(a: StringConstPtr, start: i64, end: i64) -> StringPtr {
    // SAFETY: a blob the program holds.
    let a = unsafe { blob(a) };
    let n = a.len() as i64;
    let start = start.clamp(0, n) as usize;
    let end = end.clamp(start as i64, n) as usize;
    zrtl::string::string_from_bytes(&a[start..end])
}

extern "C" fn host_bytes_eq(a: StringConstPtr, b: StringConstPtr) -> i32 {
    // SAFETY: blobs the program holds.
    unsafe { zrtl::string::string_equals(a, b) as i32 }
}

extern "C" fn host_bytes_hash(a: StringConstPtr) -> i64 {
    // SAFETY: a blob the program holds.
    let a = unsafe { blob(a) };
    // FNV-1a, folded to a non-negative value like a string's hash.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in a {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    (h >> 1) as i64
}

/// `repr(b)`: the bytes as Python spells a bytes literal.
extern "C" fn host_bytes_repr(a: StringConstPtr) -> StringPtr {
    // SAFETY: a blob the program holds.
    let a = unsafe { blob(a) };
    let quote = if a.contains(&b'\'') && !a.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut out = Vec::with_capacity(a.len() + 3);
    out.push(b'b');
    out.push(quote);
    for &byte in a {
        match byte {
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            b if b == quote => {
                out.push(b'\\');
                out.push(b);
            }
            0x20..=0x7e => out.push(byte),
            _ => out.extend_from_slice(format!("\\x{byte:02x}").as_bytes()),
        }
    }
    out.push(quote);
    zrtl::string::string_from_bytes(&out)
}

/// The blob as text, when its bytes are UTF-8; null otherwise.
extern "C" fn host_bytes_decode(a: StringConstPtr) -> StringPtr {
    // SAFETY: a blob the program holds.
    let a = unsafe { blob(a) };
    match std::str::from_utf8(a) {
        Ok(_) => zrtl::string::string_from_bytes(a),
        Err(_) => std::ptr::null_mut(),
    }
}

/// A list as the library lays it out: storage, length, capacity.
#[repr(C)]
struct ListHeader {
    data: *const u8,
    len: i64,
    capacity: i64,
}

/// The elements of a list, `width` bytes each, as the bytes they are.
unsafe fn list_bytes<'a>(list: *const ListHeader, width: usize) -> &'a [u8] {
    if list.is_null() {
        return &[];
    }
    // SAFETY: a list header the program holds, with `len` elements.
    let header = unsafe { &*list };
    let n = usize::try_from(header.len).unwrap_or(0) * width;
    if header.data.is_null() || n == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(header.data, n) }
}

/// `bytes(ints)`: each value 0 to 255; null when one is not.
extern "C" fn host_bytes_from_ints(list: *const ListHeader) -> StringPtr {
    // SAFETY: a list of i64 the program holds.
    let values = unsafe { list_bytes(list, 8) };
    let mut out = Vec::with_capacity(values.len() / 8);
    for v in values.as_chunks::<8>().0 {
        match u8::try_from(i64::from_ne_bytes(*v)) {
            Ok(b) => out.push(b),
            Err(_) => return std::ptr::null_mut(),
        }
    }
    zrtl::string::string_from_bytes(&out)
}

/// `bytes(n)`: `n` zero bytes.
extern "C" fn host_bytes_zeros(n: i64) -> StringPtr {
    zrtl::string::string_from_bytes(&vec![0u8; usize::try_from(n).unwrap_or(0)])
}

/// A byte buffer built up, as a blob.
extern "C" fn host_bytes_from_buffer(list: *const ListHeader) -> StringPtr {
    // SAFETY: a list of u8 the program holds.
    zrtl::string::string_from_bytes(unsafe { list_bytes(list, 1) })
}

/// A blob's bytes copied to `data`, which holds room for them; the
/// count copied.
extern "C" fn host_bytes_copy_out(a: StringConstPtr, data: i64) -> i64 {
    // SAFETY: a blob the program holds; the caller sized the buffer.
    let a = unsafe { blob(a) };
    if data != 0 && !a.is_empty() {
        unsafe { std::ptr::copy_nonoverlapping(a.as_ptr(), data as *mut u8, a.len()) };
    }
    a.len() as i64
}

fn path_of(p: StringConstPtr) -> Option<String> {
    // SAFETY: a string the program holds.
    unsafe { zrtl::string::string_as_str(p) }.map(str::to_string)
}

/// Write `data` to the file at `path`, appending when `append` is not
/// zero; 0 on success, -1 on failure.
extern "C" fn host_file_write(path: StringConstPtr, data: StringConstPtr, append: i64) -> i64 {
    let Some(path) = path_of(path) else {
        return -1;
    };
    // SAFETY: a blob the program holds.
    let data = unsafe { blob(data) };
    let written = if append != 0 {
        std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .and_then(|mut f| std::io::Write::write_all(&mut f, data))
    } else {
        std::fs::write(&path, data)
    };
    if written.is_ok() { 0 } else { -1 }
}

/// The whole file at `path` as a blob; null when it cannot be read.
extern "C" fn host_file_read(path: StringConstPtr) -> StringPtr {
    let Some(path) = path_of(path) else {
        return std::ptr::null_mut();
    };
    match std::fs::read(&path) {
        Ok(bytes) => zrtl::string::string_from_bytes(&bytes),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Remove the file at `path`; 0 on success, -1 when it cannot be.
extern "C" fn host_file_remove(path: StringConstPtr) -> i64 {
    match path_of(path).map(std::fs::remove_file) {
        Some(Ok(())) => 0,
        _ => -1,
    }
}

extern "C" fn host_file_exists(path: StringConstPtr) -> i64 {
    path_of(path).is_some_and(|p| std::path::Path::new(&p).exists()) as i64
}

static INFO: zrtl::ZrtlInfo = zrtl::ZrtlInfo::new(c"python_host".as_ptr());
static SYMBOLS: [zrtl::ZrtlSymbol; 22] = [
    zrtl::ZrtlSymbol::new(c"$Host$argc".as_ptr(), host_argc as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$argv".as_ptr(), host_argv as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$time".as_ptr(), host_time as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Host$perf_counter".as_ptr(),
        host_perf_counter as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Host$missing_arg".as_ptr(), host_missing_arg as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Host$bytes_of_byte".as_ptr(),
        host_bytes_of_byte as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_box".as_ptr(), host_bytes_box as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Host$bytes_repeat".as_ptr(),
        host_bytes_repeat as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_at".as_ptr(), host_bytes_at as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_slice".as_ptr(), host_bytes_slice as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_eq".as_ptr(), host_bytes_eq as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_hash".as_ptr(), host_bytes_hash as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_repr".as_ptr(), host_bytes_repr as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Host$bytes_decode".as_ptr(),
        host_bytes_decode as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Host$bytes_from_ints".as_ptr(),
        host_bytes_from_ints as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_zeros".as_ptr(), host_bytes_zeros as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Host$bytes_from_buffer".as_ptr(),
        host_bytes_from_buffer as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Host$bytes_copy_out".as_ptr(),
        host_bytes_copy_out as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Host$file_write".as_ptr(), host_file_write as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$file_read".as_ptr(), host_file_read as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$file_remove".as_ptr(), host_file_remove as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$file_exists".as_ptr(), host_file_exists as *const u8),
];

/// The host's symbols as a plugin the runtime links like any other.
pub(crate) fn static_plugin() -> zrtl::StaticPlugin {
    zrtl::StaticPlugin {
        info: &INFO,
        symbols: &SYMBOLS,
    }
}
