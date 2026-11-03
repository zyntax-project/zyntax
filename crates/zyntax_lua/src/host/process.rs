//! Process support.

use super::*;

/// The program's arguments, `arg` in Lua's terms: the interpreter,
/// then the script's path, then the rest. Set once per process; a
/// later call keeps the first.
pub fn set_args(args: Vec<String>) {
    let _ = ARGS.set(args);
}

/// Whether the program ignores the environment's `LUA_*` variables, as
/// `lua -E` does.
pub fn set_ignore_env(on: bool) {
    IGNORE_ENV.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// `package.cpath` as the reference sets it: `LUA_CPATH_5_4`, else
/// `LUA_CPATH`, else the default, with a `;;` in the variable standing
/// for the default; the default alone when `ignore_env` (`-E`).
pub(super) fn cpath_value(ignore_env: bool, var: impl Fn(&str) -> Option<String>) -> String {
    let default = cpath_default();
    let path = if ignore_env {
        None
    } else {
        var("LUA_CPATH_5_4").or_else(|| var("LUA_CPATH"))
    };
    let Some(path) = path else {
        return default;
    };
    let Some(mark) = path.find(";;") else {
        return path;
    };
    let mut out = String::new();
    if mark > 0 {
        out.push_str(&path[..mark]);
        out.push(';');
    }
    out.push_str(&default);
    if mark + 2 < path.len() {
        out.push(';');
        out.push_str(&path[mark + 2..]);
    }
    out
}

/// The default, with `!` standing for the executable's directory on
/// Windows.
pub(super) fn cpath_default() -> String {
    if cfg!(windows)
        && let Some(dir) = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.display().to_string()))
    {
        return CPATH_DEFAULT.replace('!', &dir);
    }
    CPATH_DEFAULT.to_string()
}

pub(super) extern "C" fn host_cpath() -> StringPtr {
    let value = cpath_value(
        IGNORE_ENV.load(std::sync::atomic::Ordering::Relaxed),
        |name| std::env::var(name).ok(),
    );
    zrtl::string::string_from_bytes(value.as_bytes())
}

pub(super) extern "C" fn host_argc() -> i64 {
    ARGS.get().map_or(0, |args| args.len() as i64)
}

pub(super) extern "C" fn host_argv(i: i64) -> StringPtr {
    match usize::try_from(i)
        .ok()
        .and_then(|i| ARGS.get().and_then(|args| args.get(i)))
    {
        Some(arg) => zrtl::string_new(arg),
        None => std::ptr::null_mut(),
    }
}

/// Seconds of CPU time, as `os.clock` reports.
pub(super) extern "C" fn host_clock() -> f64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
}

/// Seconds since the epoch, as `os.time()` reports.
pub(super) extern "C" fn host_time() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// `os.setlocale`: only the C locale is on offer, so a query or a
/// request for it answers "C" and any other locale is refused.
pub(super) extern "C" fn host_setlocale(locale: zrtl::StringConstPtr) -> StringPtr {
    let wanted = if locale.is_null() {
        &b""[..]
    } else {
        unsafe { bytes_of(locale) }
    };
    match wanted {
        b"" | b"C" | b"POSIX" => zrtl::string::string_from_bytes(b"C"),
        _ => std::ptr::null_mut(),
    }
}

/// A host symbol the shared library declares for another frontend's
/// host: no Lua value reaches the code that calls it, so reaching it
/// is a fault in the library's dispatch.
pub(super) extern "C" fn host_unprovided() {
    eprintln!("zylua: the shared library called a host function this frontend does not provide");
    std::process::abort();
}
