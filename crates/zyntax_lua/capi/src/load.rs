//! Native libraries: `package.loadlib` and the open functions the C
//! searchers look for.
//!
//! The library's side composes Lua's results from a status and what
//! [`c_loaded`] and [`c_error`] give. Libraries are opened once per
//! path and never closed: the first open of a path fixes whether its
//! symbols are global.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, Ordering};

use zrtl::DynamicBox;
use zrtl_native::{Library, Scope};

use crate::values::{self, Any, Str};

/// What a load returns: done, the library did not open, or it has no
/// such function.
pub const LOADED: i64 = 0;
pub const ERRLIB: i64 = 1;
pub const ERRFUNC: i64 = 2;

thread_local! {
    static LIBRARIES: RefCell<HashMap<Vec<u8>, Library>> = RefCell::new(HashMap::new());
    static ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// The value the last successful load gave, until it is taken. A root
/// of the collector once the API is ready.
pub(crate) static RESULT: AtomicPtr<DynamicBox> = AtomicPtr::new(std::ptr::null_mut());

fn fail(status: i64, message: impl Into<Vec<u8>>) -> i64 {
    ERROR.with(|e| *e.borrow_mut() = message.into());
    status
}

/// The library at `path`, opened now unless it already was.
fn library(path: &[u8], scope: Scope) -> Result<*mut std::ffi::c_void, String> {
    if let Some(h) = LIBRARIES.with(|l| l.borrow().get(path).map(Library::handle)) {
        return Ok(h);
    }
    let lib = Library::open(path, scope)?;
    let handle = lib.handle();
    LIBRARIES.with(|l| l.borrow_mut().insert(path.to_vec(), lib));
    Ok(handle)
}

/// Look `sym` up in the library at `path`: `*` only opens it with its
/// symbols global and gives true; any other name gives that C function.
fn look_for(path: &[u8], sym: &[u8]) -> i64 {
    let everything = sym == b"*";
    let scope = if everything {
        Scope::Global
    } else {
        Scope::Local
    };
    let fresh = LIBRARIES.with(|l| !l.borrow().contains_key(path));
    let handle = match library(path, scope) {
        Ok(h) => h,
        Err(e) => return fail(ERRLIB, e),
    };
    if let Err(e) = crate::ready() {
        return fail(ERRLIB, e);
    }
    if fresh {
        note_library(path, handle);
    }
    if everything {
        RESULT.store(values::box_bool(true) as *mut DynamicBox, Ordering::Release);
        return LOADED;
    }
    let found = LIBRARIES.with(|l| l.borrow().get(path).map(|lib| lib.symbol(sym)));
    match found {
        Some(Ok(p)) => {
            // SAFETY: a module's exported open function is a
            // `lua_CFunction`, which is what `lookforfunc` assumes too.
            let f: crate::state::CFunction = unsafe { std::mem::transmute(p) };
            let fv = crate::calls::light_function(f);
            RESULT.store(fv as *mut DynamicBox, Ordering::Release);
            LOADED
        }
        Some(Err(e)) => fail(ERRFUNC, e),
        None => fail(ERRLIB, "the library was not kept"),
    }
}

/// `registry._CLIBS[path]`: the handle, as the reference keeps it.
fn note_library(path: &[u8], handle: *mut std::ffi::c_void) {
    let b = crate::bridge();
    let clibs = (b.index)((b.registry)(), values::box_bytes(b"_CLIBS"));
    if unsafe { values::type_of(clibs) } != values::LUA_TTABLE {
        return;
    }
    let key = values::box_bytes(path);
    let light = values::box_pointer(handle as *mut u8, values::tags().light);
    (b.rawset)(clibs, key, light);
}

unsafe fn text<'a>(s: Str) -> &'a [u8] {
    unsafe { values::bytes_of(s) }
}

/// `$LuaC$loadlib(path, sym)`: the status of `package.loadlib`.
pub extern "C" fn c_loadlib(path: Str, sym: Str) -> i64 {
    let (path, sym) = unsafe { (text(path), text(sym)) };
    look_for(path, sym)
}

/// The open functions a module is looked for under, in order: dots
/// become `_`; a name with a hyphen is looked for as `luaopen_` and
/// the part before it, then as `luaopen_` and the part after it.
pub fn open_names(modname: &[u8]) -> Vec<Vec<u8>> {
    let name: Vec<u8> = modname
        .iter()
        .map(|&c| if c == b'.' { b'_' } else { c })
        .collect();
    let open = |part: &[u8]| {
        let mut sym = b"luaopen_".to_vec();
        sym.extend_from_slice(part);
        sym
    };
    match name.iter().position(|&c| c == b'-') {
        Some(mark) => vec![open(&name[..mark]), open(&name[mark + 1..])],
        None => vec![open(&name)],
    }
}

/// `$LuaC$loadfunc(file, modname)`: the status of looking for a
/// module's open function in `file`, under each of its [`open_names`]
/// until one is found or the library does not open.
pub extern "C" fn c_loadfunc(file: Str, modname: Str) -> i64 {
    let (file, modname) = unsafe { (text(file), text(modname)) };
    let mut status = ERRFUNC;
    for sym in open_names(modname) {
        status = look_for(file, &sym);
        if status != ERRFUNC {
            break;
        }
    }
    status
}

/// `$LuaC$loaded()`: the value the last successful load gave.
pub extern "C" fn c_loaded() -> Any {
    RESULT.swap(std::ptr::null_mut(), Ordering::AcqRel)
}

/// `$LuaC$error()`: the message of the last failed load.
pub extern "C" fn c_error() -> Str {
    ERROR.with(|e| values::new_string(&e.borrow()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_names_follow_the_reference() {
        assert_eq!(open_names(b"lib1"), vec![b"luaopen_lib1".to_vec()]);
        assert_eq!(open_names(b"lib1.sub"), vec![b"luaopen_lib1_sub".to_vec()]);
        assert_eq!(
            open_names(b"lib2-v2"),
            vec![b"luaopen_lib2".to_vec(), b"luaopen_v2".to_vec()]
        );
        assert_eq!(
            open_names(b"a.b-c.d"),
            vec![b"luaopen_a_b".to_vec(), b"luaopen_c_d".to_vec()]
        );
    }

    #[test]
    fn a_library_that_does_not_open_is_an_open_failure() {
        let s = values::new_string(b"./no-such-library.so");
        let t = values::new_string(b"luaopen_x");
        assert_eq!(c_loadlib(s, t), ERRLIB);
        let e = ERROR.with(|e| e.borrow().clone());
        assert!(!e.is_empty());
        assert_eq!(c_loadfunc(s, t), ERRLIB);
    }
}
