//! The `os` library's host side: calendar time through the C
//! library's `mktime`, `localtime` and `strftime`, as the reference
//! does it, the environment, temporary names, and files removed or
//! renamed. Reached from the library as `$Lua$…` symbols.

use std::ffi::{CStr, CString};

use zrtl::StringPtr;

unsafe fn bytes_of(s: zrtl::StringConstPtr) -> &'static [u8] {
    unsafe { zrtl::string_as_bytes(s) }
}

fn c_string(s: zrtl::StringConstPtr) -> Option<CString> {
    CString::new(unsafe { bytes_of(s) }).ok()
}

// The last failure of a file operation, as `name: reason`, with its
// number.
thread_local! {
    static OS_ERROR: std::cell::RefCell<(String, i64)> = const { std::cell::RefCell::new((String::new(), 0)) };
}

/// The failure just seen, kept as `name: reason`, or the reason alone
/// when the operation names no file, as the reference reports them.
fn fail(name: Option<&str>) -> i64 {
    let err = std::io::Error::last_os_error();
    let code = err.raw_os_error().unwrap_or(0) as i64;
    let reason = err
        .to_string()
        .split(" (os error")
        .next()
        .unwrap_or("")
        .to_string();
    let message = match name {
        Some(name) => format!("{name}: {reason}"),
        None => reason,
    };
    OS_ERROR.with(|e| *e.borrow_mut() = (message, code));
    code
}

pub(crate) extern "C" fn host_os_error() -> StringPtr {
    OS_ERROR.with(|e| zrtl::string::string_from_bytes(e.borrow().0.as_bytes()))
}

// ─── calendar time ──────────────────────────────────────────────────

/// The broken-down time of `t`, local or UTC; none when the C library
/// cannot represent it.
fn broken_down(t: i64, utc: bool) -> Option<libc::tm> {
    // time_t is narrower than i64 on some platforms.
    #[allow(clippy::unnecessary_cast)]
    let t = t as libc::time_t;
    if t as i64 != t {
        return None;
    }
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let got = unsafe {
        if utc {
            libc::gmtime_r(&t, &mut tm)
        } else {
            libc::localtime_r(&t, &mut tm)
        }
    };
    if got.is_null() { None } else { Some(tm) }
}

/// One field of the broken-down time of `t`, numbered as `os.date`'s
/// `*t` table lists them: year, month, day, hour, min, sec, wday,
/// yday, isdst. `i64::MIN` when the time cannot be represented.
pub(crate) extern "C" fn host_date_field(t: i64, utc: bool, index: i64) -> i64 {
    let Some(tm) = broken_down(t, utc) else {
        return i64::MIN;
    };
    match index {
        0 => tm.tm_year as i64 + 1900,
        1 => tm.tm_mon as i64 + 1,
        2 => tm.tm_mday as i64,
        3 => tm.tm_hour as i64,
        4 => tm.tm_min as i64,
        5 => tm.tm_sec as i64,
        6 => tm.tm_wday as i64 + 1,
        7 => tm.tm_yday as i64 + 1,
        _ => tm.tm_isdst as i64,
    }
}

/// The time the fields name, through `mktime`, which normalizes them;
/// -1 when it cannot be represented. `isdst` is negative when unknown.
pub(crate) extern "C" fn host_time_of(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    min: i64,
    sec: i64,
    isdst: i64,
) -> i64 {
    let field = |v: i64| i32::try_from(v).ok();
    let (Some(year), Some(month), Some(day), Some(hour), Some(min), Some(sec)) = (
        field(year - 1900),
        field(month - 1),
        field(day),
        field(hour),
        field(min),
        field(sec),
    ) else {
        return -1;
    };
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    tm.tm_year = year;
    tm.tm_mon = month;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = min;
    tm.tm_sec = sec;
    tm.tm_isdst = isdst as i32;
    let t = unsafe { libc::mktime(&mut tm) };
    t as i64
}

/// The conversions `strftime` takes, as the reference lists them for
/// C99: each entry is a conversion, or a modifier followed by the
/// conversions it may precede.
const STRFTIME_OPTIONS: &[&str] = &[
    "a", "A", "b", "B", "c", "C", "d", "D", "e", "F", "g", "G", "h", "H", "I", "j", "m", "M", "n",
    "p", "r", "R", "S", "t", "T", "u", "U", "V", "w", "W", "x", "X", "y", "Y", "z", "Z", "%", "Ec",
    "EC", "Ex", "EX", "Ey", "EY", "Od", "Oe", "OH", "OI", "Om", "OM", "OS", "Ou", "OU", "OV", "Ow",
    "OW", "Oy",
];

/// The conversion at the start of `rest`, in bytes, when it is one.
fn conversion_len(rest: &[u8]) -> Option<usize> {
    STRFTIME_OPTIONS
        .iter()
        .find(|opt| rest.starts_with(opt.as_bytes()))
        .map(|opt| opt.len())
}

/// What follows the first conversion `strftime` does not take, from
/// its `%` on; empty when every conversion is one.
pub(crate) extern "C" fn host_date_check(fmt: zrtl::StringConstPtr) -> StringPtr {
    let fmt = unsafe { bytes_of(fmt) };
    let mut i = 0;
    while i < fmt.len() {
        if fmt[i] != b'%' {
            i += 1;
            continue;
        }
        match conversion_len(&fmt[i + 1..]) {
            Some(n) => i += 1 + n,
            None => return zrtl::string::string_from_bytes(&fmt[i + 1..]),
        }
    }
    zrtl::string::string_from_bytes(b"")
}

/// `t` formatted by `strftime`, local or UTC. A format is given as
/// checked; one the library cannot represent gives an empty string.
pub(crate) extern "C" fn host_date(fmt: zrtl::StringConstPtr, t: i64, utc: bool) -> StringPtr {
    let Some(tm) = broken_down(t, utc) else {
        return std::ptr::null_mut();
    };
    let Some(fmt) = c_string(fmt) else {
        return zrtl::string::string_from_bytes(b"");
    };
    if fmt.as_bytes().is_empty() {
        return zrtl::string::string_from_bytes(b"");
    }
    // strftime tells an empty result from one too long only by size:
    // the buffer grows until the text fits or is clearly empty.
    let mut size = 256;
    loop {
        let mut buf = vec![0u8; size];
        let n = unsafe {
            libc::strftime(
                buf.as_mut_ptr() as *mut libc::c_char,
                size,
                fmt.as_ptr(),
                &tm,
            )
        };
        if n > 0 {
            return zrtl::string::string_from_bytes(&buf[..n]);
        }
        if size >= fmt.as_bytes().len() * 64 + 4096 {
            return zrtl::string::string_from_bytes(b"");
        }
        size *= 4;
    }
}

// ─── the environment and files ──────────────────────────────────────

pub(crate) extern "C" fn host_getenv(name: zrtl::StringConstPtr) -> StringPtr {
    let name = String::from_utf8_lossy(unsafe { bytes_of(name) }).into_owned();
    match std::env::var_os(name) {
        Some(v) => zrtl::string::string_from_bytes(v.to_string_lossy().as_bytes()),
        None => std::ptr::null_mut(),
    }
}

/// A fresh file under the temporary directory, made and closed, as
/// the reference's `mkstemp` leaves it; its name.
pub(crate) extern "C" fn host_tmpname() -> StringPtr {
    let mut template = b"/tmp/lua_XXXXXX\0".to_vec();
    let fd = unsafe { libc::mkstemp(template.as_mut_ptr() as *mut libc::c_char) };
    if fd < 0 {
        return std::ptr::null_mut();
    }
    unsafe { libc::close(fd) };
    let name = CStr::from_bytes_until_nul(&template).map_or(&b""[..], CStr::to_bytes);
    zrtl::string::string_from_bytes(name)
}

/// Zero, or the error's number with its text kept for `os_error`.
pub(crate) extern "C" fn host_remove(name: zrtl::StringConstPtr) -> i64 {
    let text = String::from_utf8_lossy(unsafe { bytes_of(name) }).into_owned();
    let Some(path) = c_string(name) else {
        return fail(Some(&text));
    };
    if unsafe { libc::remove(path.as_ptr()) } == 0 {
        0
    } else {
        fail(Some(&text))
    }
}

pub(crate) extern "C" fn host_rename(from: zrtl::StringConstPtr, to: zrtl::StringConstPtr) -> i64 {
    let (Some(from), Some(to)) = (c_string(from), c_string(to)) else {
        return fail(None);
    };
    if unsafe { libc::rename(from.as_ptr(), to.as_ptr()) } == 0 {
        0
    } else {
        fail(None)
    }
}

/// `system(command)`: the status word, or whether a shell is there
/// when the command is null.
pub(crate) extern "C" fn host_execute(command: zrtl::StringConstPtr) -> i64 {
    if command.is_null() {
        return i64::from(unsafe { libc::system(std::ptr::null()) } != 0);
    }
    let Some(command) = c_string(command) else {
        return -1;
    };
    unsafe { libc::system(command.as_ptr()) as i64 }
}

/// How a status word from `system` ended: 0 and the exit code, or 1
/// and the signal.
pub(crate) extern "C" fn host_exec_result(status: i64, want_signal: bool) -> i64 {
    let status = status as i32;
    if libc::WIFEXITED(status) {
        if want_signal {
            0
        } else {
            libc::WEXITSTATUS(status) as i64
        }
    } else if libc::WIFSIGNALED(status) {
        if want_signal {
            1
        } else {
            libc::WTERMSIG(status) as i64
        }
    } else if want_signal {
        0
    } else {
        status as i64
    }
}
