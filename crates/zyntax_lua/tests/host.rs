//! A host embedding Lua as a C host does: the state opened, a chunk
//! loaded with the global `load` and run with a protected call through
//! the C API's cores, its table read and its function called, and an
//! error handed back rather than ending the process. Its own process,
//! since the C API is installed once per process.

use std::ffi::{CStr, c_int};

use zyntax_embed::{TieredConfig, TieredRuntime};
use zyntax_lua_capi::api::{
    zlc_getfield, zlc_getglobal, zlc_gettop, zlc_pushinteger, zlc_pushlstring, zlc_settop,
    zlc_tointegerx, zlc_tolstring,
};
use zyntax_lua_capi::calls::{OK, zlc_pcall};
use zyntax_lua_capi::state::L;

const LUA_OK: c_int = 0;
const LUA_ERRRUN: c_int = 2;

unsafe fn push(l: L, s: &str) {
    unsafe { zlc_pushlstring(l, s.as_ptr().cast(), s.len()) };
}

/// Load `source` as a chunk and run it, leaving its first result on the
/// stack: the status of whichever step failed, else `LUA_OK`.
unsafe fn run(l: L, source: &str, name: &str) -> c_int {
    unsafe {
        let mut ty = 0;
        zlc_getglobal(l, c"load".as_ptr(), &mut ty);
        push(l, source);
        push(l, name);
        let mut status = 0;
        assert_eq!(zlc_pcall(l, 2, 2, 0, &mut status), OK);
        assert_eq!(status, LUA_OK, "load itself is not the error");
        // `load` gives the function, or nil and the message.
        zlc_settop(l, -2);
        assert_eq!(zlc_pcall(l, 0, 1, 0, &mut status), OK);
        status
    }
}

unsafe fn text(l: L, idx: c_int) -> String {
    unsafe {
        let p = zlc_tolstring(l, idx, std::ptr::null_mut());
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

#[test]
fn a_host_loads_and_calls_a_chunk() {
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    zyntax_lua::register_runtime(&mut rt).expect("registered");
    let l = zyntax_lua::open_host(&mut rt).expect("the state opens");
    unsafe {
        let module = "local M = {} function M.add(a, b) return a + b end M.answer = 42 return M";
        assert_eq!(run(l, module, "=module"), LUA_OK);
        let mut ty = 0;
        let mut isnum = 0;
        zlc_getfield(l, -1, c"answer".as_ptr(), &mut ty);
        assert_eq!(zlc_tointegerx(l, -1, &mut isnum), 42);
        zlc_settop(l, -2);
        zlc_getfield(l, -1, c"add".as_ptr(), &mut ty);
        zlc_pushinteger(l, 2);
        zlc_pushinteger(l, 3);
        let mut status = 0;
        assert_eq!(zlc_pcall(l, 2, 1, 0, &mut status), OK);
        assert_eq!(status, LUA_OK);
        assert_eq!(zlc_tointegerx(l, -1, &mut isnum), 5);
        zlc_settop(l, 0);

        // A chunk that raises hands its error back.
        assert_eq!(run(l, "error('boom')", "=broken"), LUA_ERRRUN);
        assert!(text(l, -1).contains("boom"), "{}", text(l, -1));
        zlc_settop(l, 0);

        // The state stays open: a later chunk sees what an earlier one
        // left in the globals.
        assert_eq!(run(l, "shared = 7", "=first"), LUA_OK);
        zlc_settop(l, 0);
        assert_eq!(run(l, "return shared * 6", "=second"), LUA_OK);
        assert_eq!(zlc_tointegerx(l, -1, &mut isnum), 42);
        assert_eq!(zlc_gettop(l), 1);
    }
    std::mem::forget(rt);
}
