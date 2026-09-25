//! The `os` library's host side: calendar time through the C
//! library's `mktime`, `localtime` and `strftime`, as the reference
//! does it, the environment, temporary names, and files removed or
//! renamed. Reached from the library as `$Lua$…` symbols.

use std::ffi::CString;

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
/// Only ever follows a C library call, so `errno` holds the failure.
fn fail(name: Option<&str>) -> i64 {
    let n = last_errno();
    let reason = strerror(n);
    let message = match name {
        Some(name) => format!("{name}: {reason}"),
        None => reason,
    };
    OS_ERROR.with(|e| *e.borrow_mut() = (message, i64::from(n)));
    i64::from(n)
}

#[cfg(unix)]
fn last_errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

#[cfg(windows)]
fn last_errno() -> i32 {
    unsafe extern "C" {
        fn _errno() -> *mut libc::c_int;
    }
    // SAFETY: the C runtime's per-thread `errno` cell.
    unsafe { *_errno() }
}

// ─── failures as the C library words them ───────────────────────────

/// A failure the C library would report as the `errno` it holds.
#[derive(Debug)]
struct Errno(i32);

impl std::fmt::Display for Errno {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&strerror(self.0))
    }
}

impl std::error::Error for Errno {}

/// An I/O error standing for `errno` `n`.
pub(crate) fn errno_error(n: i32) -> std::io::Error {
    std::io::Error::other(Errno(n))
}

/// A failure as the reference reports it: the `errno` the C library
/// would have set for it and `strerror`'s text for that number. An
/// error that carries no number keeps its own text, with 0.
pub(crate) fn c_error(err: &std::io::Error) -> (i64, String) {
    if let Some(Errno(n)) = err.get_ref().and_then(|e| e.downcast_ref::<Errno>()) {
        return (i64::from(*n), strerror(*n));
    }
    match err.raw_os_error() {
        Some(code) => {
            let n = errno_of(code);
            (i64::from(n), strerror(n))
        }
        None => (0, err.to_string()),
    }
}

/// On Unix an OS error code is the `errno`.
#[cfg(unix)]
fn errno_of(code: i32) -> i32 {
    code
}

/// On Windows an OS error code is a system error code, which the C
/// runtime turns into an `errno` by this table (its `_dosmaperr`).
#[cfg(windows)]
fn errno_of(code: i32) -> i32 {
    use libc::{
        E2BIG, EACCES, EAGAIN, EBADF, ECHILD, EEXIST, EINVAL, EMFILE, ENOENT, ENOEXEC, ENOMEM,
        ENOSPC, ENOTEMPTY, EPIPE, EXDEV,
    };
    match code {
        2 | 3 | 15 | 18 | 53 | 67 | 161 | 206 => ENOENT,
        4 => EMFILE,
        5 | 16 | 65 | 82 | 83 | 108 | 132 | 158 | 167 => EACCES,
        6 | 114 | 130 => EBADF,
        7 | 8 | 9 | 1816 => ENOMEM,
        10 => E2BIG,
        11 => ENOEXEC,
        17 => EXDEV,
        80 | 183 => EEXIST,
        89 | 164 | 215 => EAGAIN,
        109 => EPIPE,
        112 => ENOSPC,
        128 | 129 => ECHILD,
        145 => ENOTEMPTY,
        19..=36 => EACCES,
        188..=202 => ENOEXEC,
        _ => EINVAL,
    }
}

/// `strerror(n)`. Rust words an OS error with it on Unix.
#[cfg(unix)]
fn strerror(n: i32) -> String {
    let text = std::io::Error::from_raw_os_error(n).to_string();
    text.split(" (os error").next().unwrap_or("").to_string()
}

/// `strerror(n)`, from the C runtime: Rust words an OS error on
/// Windows with the system's text instead.
#[cfg(windows)]
fn strerror(n: i32) -> String {
    // SAFETY: the C runtime returns a NUL-terminated string it owns,
    // copied out before anything else can call it on this thread.
    unsafe { std::ffi::CStr::from_ptr(libc::strerror(n)) }
        .to_string_lossy()
        .into_owned()
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
    #[cfg(unix)]
    let ok = unsafe {
        if utc {
            !libc::gmtime_r(&t, &mut tm).is_null()
        } else {
            !libc::localtime_r(&t, &mut tm).is_null()
        }
    };
    #[cfg(windows)]
    let ok = unsafe {
        if utc {
            libc::gmtime_s(&mut tm, &t) == 0
        } else {
            libc::localtime_s(&mut tm, &t) == 0
        }
    };
    ok.then_some(tm)
}

// The C runtime's calendar functions the `libc` crate declares only
// for Unix. The UCRT exports `mktime` under its 64-bit name.
#[cfg(unix)]
use libc::{mktime, strftime};
#[cfg(windows)]
unsafe extern "C" {
    #[link_name = "_mktime64"]
    fn mktime(tm: *mut libc::tm) -> libc::time_t;
    fn strftime(
        s: *mut libc::c_char,
        max: libc::size_t,
        format: *const libc::c_char,
        tm: *const libc::tm,
    ) -> libc::size_t;
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
    let t = unsafe { mktime(&mut tm) };
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
/// its `%` on; null when every conversion is one.
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
    std::ptr::null_mut()
}

/// `t` formatted as the reference formats it: each conversion goes
/// through `strftime` on its own and every other byte is copied, so
/// a format may hold any byte. The format is given as checked. Null
/// when the time cannot be represented.
pub(crate) extern "C" fn host_date(fmt: zrtl::StringConstPtr, t: i64, utc: bool) -> StringPtr {
    let Some(tm) = broken_down(t, utc) else {
        return std::ptr::null_mut();
    };
    let fmt = unsafe { bytes_of(fmt) };
    let mut out = Vec::with_capacity(fmt.len());
    let mut i = 0;
    while i < fmt.len() {
        if fmt[i] != b'%' {
            out.push(fmt[i]);
            i += 1;
            continue;
        }
        let n = conversion_len(&fmt[i + 1..]).unwrap_or(0);
        let mut one = Vec::with_capacity(n + 2);
        one.extend_from_slice(&fmt[i..i + 1 + n]);
        one.push(0);
        let mut buf = [0u8; 256];
        let written = unsafe {
            strftime(
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len(),
                one.as_ptr() as *const libc::c_char,
                &tm,
            )
        };
        out.extend_from_slice(&buf[..written]);
        i += 1 + n;
    }
    zrtl::string::string_from_bytes(&out)
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
#[cfg(unix)]
pub(crate) extern "C" fn host_tmpname() -> StringPtr {
    let mut template = b"/tmp/lua_XXXXXX\0".to_vec();
    let fd = unsafe { libc::mkstemp(template.as_mut_ptr() as *mut libc::c_char) };
    if fd < 0 {
        return std::ptr::null_mut();
    }
    unsafe { libc::close(fd) };
    let name =
        std::ffi::CStr::from_bytes_until_nul(&template).map_or(&b""[..], std::ffi::CStr::to_bytes);
    zrtl::string::string_from_bytes(name)
}

/// A fresh file under the temporary directory, made and closed; its
/// name.
#[cfg(windows)]
pub(crate) extern "C" fn host_tmpname() -> StringPtr {
    match crate::host_io::fresh_temp_file(false) {
        Ok((path, _)) => zrtl::string::string_from_bytes(path.to_string_lossy().as_bytes()),
        Err(_) => std::ptr::null_mut(),
    }
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
#[cfg(unix)]
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

/// On Windows the status `system` gives is the exit code itself, as
/// the reference reads it there: never a signal.
#[cfg(windows)]
pub(crate) extern "C" fn host_exec_result(status: i64, want_signal: bool) -> i64 {
    if want_signal { 0 } else { status }
}
