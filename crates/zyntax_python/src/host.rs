//! What the host hands a program: the arguments it was started with,
//! reached from the library as `$Host$argc` and `$Host$argv`, and the
//! clocks, `$Host$time` and `$Host$perf_counter`.

use std::sync::OnceLock;

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

static INFO: zrtl::ZrtlInfo = zrtl::ZrtlInfo::new(c"python_host".as_ptr());
static SYMBOLS: [zrtl::ZrtlSymbol; 5] = [
    zrtl::ZrtlSymbol::new(c"$Host$argc".as_ptr(), host_argc as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$argv".as_ptr(), host_argv as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$time".as_ptr(), host_time as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Host$perf_counter".as_ptr(),
        host_perf_counter as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Host$missing_arg".as_ptr(), host_missing_arg as *const u8),
];

/// The host's symbols as a plugin the runtime links like any other.
pub(crate) fn static_plugin() -> zrtl::StaticPlugin {
    zrtl::StaticPlugin {
        info: &INFO,
        symbols: &SYMBOLS,
    }
}
