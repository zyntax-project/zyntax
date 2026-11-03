//! Plugin support.

use super::*;

/// The host's symbols as a plugin the runtime links like any other,
/// with a trap under every host symbol the shared library declares
/// and this host does not define.
pub(crate) fn static_plugin() -> zrtl::StaticPlugin {
    static ALL: OnceLock<&'static [zrtl::ZrtlSymbol]> = OnceLock::new();
    let symbols = ALL.get_or_init(|| {
        let provided: std::collections::HashSet<&str> = SYMBOLS
            .iter()
            .chain(crate::host_debug::SYMBOLS.iter())
            .chain(crate::host_gc::SYMBOLS.iter())
            .filter_map(|s| {
                // SAFETY: every entry's name is a C string literal.
                unsafe { std::ffi::CStr::from_ptr(s.name) }.to_str().ok()
            })
            .collect();
        let mut all: Vec<zrtl::ZrtlSymbol> = SYMBOLS
            .iter()
            .chain(crate::host_debug::SYMBOLS.iter())
            .chain(crate::host_gc::SYMBOLS.iter())
            .map(|s| zrtl::ZrtlSymbol::new(s.name, s.ptr))
            .collect();
        for name in crate::fallible::HOST_EXTERNS {
            if provided.contains(name) {
                continue;
            }
            let c: &'static std::ffi::CStr = Box::leak(
                std::ffi::CString::new(*name)
                    .expect("a symbol name")
                    .into_boxed_c_str(),
            );
            all.push(zrtl::ZrtlSymbol::new(
                c.as_ptr(),
                host_unprovided as *const u8,
            ));
        }
        Box::leak(all.into_boxed_slice())
    });
    zrtl::StaticPlugin {
        info: &INFO,
        symbols,
    }
}
