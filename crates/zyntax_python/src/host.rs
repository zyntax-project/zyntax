//! What the host hands a program: the arguments it was started with,
//! reached from the library as `$Host$argc` and `$Host$argv`.

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

static INFO: zrtl::ZrtlInfo = zrtl::ZrtlInfo::new(c"python_host".as_ptr());
static SYMBOLS: [zrtl::ZrtlSymbol; 2] = [
    zrtl::ZrtlSymbol::new(c"$Host$argc".as_ptr(), host_argc as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$argv".as_ptr(), host_argv as *const u8),
];

/// The host's symbols as a plugin the runtime links like any other.
pub(crate) fn static_plugin() -> zrtl::StaticPlugin {
    zrtl::StaticPlugin {
        info: &INFO,
        symbols: &SYMBOLS,
    }
}
