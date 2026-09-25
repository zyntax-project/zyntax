//! Native interop for Zyntax runtimes.
//!
//! Three capabilities a runtime needs to host code compiled by a C
//! compiler, none of them tied to a language:
//!
//! - [`Library`]: open a shared library and resolve its symbols.
//! - [`protect`] and the C functions behind it (`include/zrtl_native.h`):
//!   an error exit for foreign code that expects a raise never to
//!   return. The jump crosses only the foreign frames between the raise
//!   and the protect frame, so the protect frame must sit below every
//!   compiled or Rust frame the foreign code was called from.
//! - [`build::export_link_args`]: the link arguments that export a list
//!   of symbols from an executable, so a library opened later binds to
//!   them.
//!
//! Foreign functions are called from C, through their real prototypes,
//! by whoever links this crate; nothing here calls a foreign pointer.

use std::ffi::{c_int, c_void};

pub mod build;

/// A protect frame, opaque: it lives in the C stack frame of the
/// [`protect`] that made it.
#[repr(C)]
pub struct Frame {
    _opaque: [u8; 0],
}

/// The chain of protect frames a raise jumps along: the innermost
/// frame, or null. One word, kept wherever the foreign code's state is,
/// initially null.
pub type Chain = *mut Frame;

/// What `zrtl_native_raise` returns when the chain holds no frame.
pub const NO_FRAME: c_int = -1;

#[cfg(not(target_arch = "wasm32"))]
unsafe extern "C" {
    fn zrtl_native_protect(
        chain: *mut Chain,
        body: unsafe extern "C" fn(*mut c_void) -> c_int,
        ctx: *mut c_void,
        out: *mut c_int,
    ) -> c_int;
    fn zrtl_native_raise(chain: *mut Chain, status: c_int, value: c_int) -> c_int;
}

/// Run `body(ctx)` under a new protect frame on `*chain`: `Ok` with
/// what it returned, or `Err((status, value))` from a raise that
/// reached the frame.
///
/// # Safety
/// `chain` must be valid for the call and stay where it is. Every frame
/// between a raise and this call must be one a jump may discard: C
/// frames with nothing to release, never a Rust frame, a compiled
/// frame or a switch to another stack.
#[cfg(not(target_arch = "wasm32"))]
pub unsafe fn protect(
    chain: *mut Chain,
    body: unsafe extern "C" fn(*mut c_void) -> c_int,
    ctx: *mut c_void,
) -> Result<c_int, (c_int, c_int)> {
    let mut out: c_int = 0;
    // SAFETY: as the caller vouched.
    let status = unsafe { zrtl_native_protect(chain, body, ctx, &mut out) };
    if status == 0 {
        Ok(out)
    } else {
        Err((status, out))
    }
}

/// Jump to the innermost frame on `*chain` with `status` (nonzero) and
/// `value`. Returns [`NO_FRAME`] when the chain holds none.
///
/// # Safety
/// As for [`protect`]: only C frames lie between here and the target,
/// which makes this callable from Rust only where nothing between the
/// target and this call is a Rust frame, as in a test's body.
#[cfg(not(target_arch = "wasm32"))]
pub unsafe fn raise(chain: *mut Chain, status: c_int, value: c_int) -> c_int {
    // SAFETY: as the caller vouched.
    unsafe { zrtl_native_raise(chain, status, value) }
}

/// Which symbols of an opened library later libraries see.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Only lookups through this handle.
    Local,
    /// Every library opened afterwards binds to them too.
    Global,
}

/// An opened shared library. Never closed: the code in it may be
/// referenced for as long as the process runs.
#[derive(Debug)]
pub struct Library {
    handle: *mut c_void,
}

// SAFETY: the handle is an identifier the loader hands out and accepts
// from any thread.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    /// Open the library at `path`, binding every symbol now. The error
    /// is the system loader's text.
    pub fn open(path: &[u8], scope: Scope) -> Result<Library, String> {
        sys::open(path, scope).map(|handle| Library { handle })
    }

    /// The address `name` has in the library. The error is the system
    /// loader's text.
    pub fn symbol(&self, name: &[u8]) -> Result<*const c_void, String> {
        sys::symbol(self.handle, name)
    }

    /// The loader's handle.
    pub fn handle(&self) -> *mut c_void {
        self.handle
    }
}

#[cfg(unix)]
mod sys {
    use super::Scope;
    use std::ffi::{CStr, CString, c_void};

    fn last_error() -> String {
        // SAFETY: dlerror returns null or a NUL-terminated string the
        // loader owns until the next call on this thread.
        let text = unsafe { libc::dlerror() };
        if text.is_null() {
            return "unknown error".to_string();
        }
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    }

    fn c_string(bytes: &[u8]) -> Result<CString, String> {
        CString::new(bytes).map_err(|_| "name contains a zero byte".to_string())
    }

    pub(super) fn open(path: &[u8], scope: Scope) -> Result<*mut c_void, String> {
        let path = c_string(path)?;
        let flags = libc::RTLD_NOW
            | match scope {
                Scope::Local => libc::RTLD_LOCAL,
                Scope::Global => libc::RTLD_GLOBAL,
            };
        // SAFETY: a NUL-terminated path; the loader runs the library's
        // initialisers, which is what opening it means.
        let handle = unsafe { libc::dlopen(path.as_ptr(), flags) };
        if handle.is_null() {
            Err(last_error())
        } else {
            Ok(handle)
        }
    }

    pub(super) fn symbol(handle: *mut c_void, name: &[u8]) -> Result<*const c_void, String> {
        let name = c_string(name)?;
        // SAFETY: a handle dlopen gave and a NUL-terminated name.
        unsafe { libc::dlerror() };
        let p = unsafe { libc::dlsym(handle, name.as_ptr()) };
        if p.is_null() {
            Err(last_error())
        } else {
            Ok(p as *const c_void)
        }
    }
}

#[cfg(windows)]
mod sys {
    use super::Scope;
    use std::ffi::{CString, c_void};

    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;
    const FORMAT_MESSAGE_IGNORE_INSERTS: u32 = 0x0000_0200;
    const FORMAT_MESSAGE_FROM_SYSTEM: u32 = 0x0000_1000;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
        fn GetLastError() -> u32;
        fn FormatMessageW(
            flags: u32,
            source: *const c_void,
            id: u32,
            language: u32,
            buffer: *mut u16,
            size: u32,
            args: *mut c_void,
        ) -> u32;
    }

    fn last_error() -> String {
        // SAFETY: plain calls into the system with a buffer of the
        // size given.
        let code = unsafe { GetLastError() };
        let mut buffer = [0u16; 512];
        let n = unsafe {
            FormatMessageW(
                FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                std::ptr::null(),
                code,
                0,
                buffer.as_mut_ptr(),
                buffer.len() as u32,
                std::ptr::null_mut(),
            )
        };
        if n == 0 {
            return format!("system error {code}\n");
        }
        String::from_utf16_lossy(&buffer[..n as usize])
    }

    pub(super) fn open(path: &[u8], _scope: Scope) -> Result<*mut c_void, String> {
        let wide: Vec<u16> = String::from_utf8_lossy(path)
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: a NUL-terminated wide path.
        let handle = unsafe {
            LoadLibraryExW(
                wide.as_ptr(),
                std::ptr::null_mut(),
                LOAD_WITH_ALTERED_SEARCH_PATH,
            )
        };
        if handle.is_null() {
            Err(last_error())
        } else {
            Ok(handle)
        }
    }

    pub(super) fn symbol(handle: *mut c_void, name: &[u8]) -> Result<*const c_void, String> {
        let name = CString::new(name).map_err(|_| "name contains a zero byte".to_string())?;
        // SAFETY: a module LoadLibraryExW gave and a NUL-terminated name.
        let p = unsafe { GetProcAddress(handle, name.as_ptr()) };
        if p.is_null() {
            Err(last_error())
        } else {
            Ok(p as *const c_void)
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod sys {
    use super::Scope;
    use std::ffi::c_void;

    const UNSUPPORTED: &str = "dynamic libraries are not supported on this target";

    pub(super) fn open(_path: &[u8], _scope: Scope) -> Result<*mut c_void, String> {
        Err(UNSUPPORTED.to_string())
    }

    pub(super) fn symbol(_handle: *mut c_void, _name: &[u8]) -> Result<*const c_void, String> {
        Err(UNSUPPORTED.to_string())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn a_missing_library_reports_the_loaders_text() {
        let err = Library::open(b"./no-such-library-anywhere.so", Scope::Local).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn a_zero_byte_in_a_path_is_refused() {
        assert!(Library::open(b"a\0b", Scope::Local).is_err());
    }
}
