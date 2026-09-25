//! The protect frame and the raise, driven from C frames.
#![cfg(not(target_arch = "wasm32"))]

use std::ffi::{c_int, c_void};
use zrtl_native::{Chain, NO_FRAME, protect};

#[repr(C)]
struct DepthCtx {
    chain: *mut Chain,
    depth: c_int,
    value: c_int,
    reached: c_int,
}

#[repr(C)]
struct NestedCtx {
    chain: *mut Chain,
    inner_status: c_int,
    inner_value: c_int,
    outer_value: c_int,
}

#[repr(C)]
struct PauseCtx {
    chain: *mut Chain,
    pause: extern "C" fn(),
    value: c_int,
}

#[link(name = "zrtl_native_test_frames", kind = "static")]
unsafe extern "C" {
    fn zn_test_raise_at_depth(ctx: *mut c_void) -> c_int;
    fn zn_test_return(ctx: *mut c_void) -> c_int;
    fn zn_test_nested_body(ctx: *mut c_void) -> c_int;
    fn zn_test_raise_unprotected(chain: *mut Chain) -> c_int;
    fn zn_test_pause_then_raise(ctx: *mut c_void) -> c_int;
}

#[test]
fn a_body_that_returns_gives_its_result() {
    let mut chain: Chain = std::ptr::null_mut();
    let mut ctx = DepthCtx {
        chain: &mut chain,
        depth: 0,
        value: 42,
        reached: 0,
    };
    let got = unsafe {
        protect(
            &mut chain,
            zn_test_return,
            &mut ctx as *mut _ as *mut c_void,
        )
    };
    assert_eq!(got, Ok(42));
    assert!(chain.is_null(), "the frame is popped");
}

#[test]
fn a_raise_five_c_frames_down_lands_in_the_protect_frame() {
    let mut chain: Chain = std::ptr::null_mut();
    let mut ctx = DepthCtx {
        chain: &mut chain,
        depth: 5,
        value: 1234,
        reached: 0,
    };
    let got = unsafe {
        protect(
            &mut chain,
            zn_test_raise_at_depth,
            &mut ctx as *mut _ as *mut c_void,
        )
    };
    assert_eq!(got, Err((7, 1234)));
    assert_eq!(ctx.reached, 1, "nothing after the raise ran");
    assert!(chain.is_null(), "the frame is popped");
}

#[test]
fn a_nested_chain_catches_innermost_first() {
    let mut chain: Chain = std::ptr::null_mut();
    let mut ctx = NestedCtx {
        chain: &mut chain,
        inner_status: 0,
        inner_value: 0,
        outer_value: 99,
    };
    let got = unsafe {
        protect(
            &mut chain,
            zn_test_nested_body,
            &mut ctx as *mut _ as *mut c_void,
        )
    };
    assert_eq!((ctx.inner_status, ctx.inner_value), (3, 11));
    assert_eq!(got, Err((5, 99)));
    assert!(chain.is_null());
}

#[test]
fn a_raise_with_no_frame_returns() {
    let mut chain: Chain = std::ptr::null_mut();
    assert_eq!(unsafe { zn_test_raise_unprotected(&mut chain) }, NO_FRAME);
}

extern "C" fn pause() {
    krio_fiber::yield_now();
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn a_raise_after_a_fiber_switch_lands_on_the_fibers_frame() {
    use std::cell::Cell;
    use std::rc::Rc;
    let result: Rc<Cell<Option<Result<c_int, (c_int, c_int)>>>> = Rc::new(Cell::new(None));
    let seen = result.clone();
    let mut fiber = krio_fiber::Fiber::new(move || {
        let mut chain: Chain = std::ptr::null_mut();
        let mut ctx = PauseCtx {
            chain: &mut chain,
            pause,
            value: 77,
        };
        let got = unsafe {
            protect(
                &mut chain,
                zn_test_pause_then_raise,
                &mut ctx as *mut _ as *mut c_void,
            )
        };
        assert!(chain.is_null());
        seen.set(Some(got));
    });
    assert_eq!(fiber.resume(), krio_fiber::FiberStep::Yielded);
    // Another protect frame on this stack while the fiber is suspended
    // leaves the fiber's own chain alone.
    let mut chain: Chain = std::ptr::null_mut();
    let mut ctx = DepthCtx {
        chain: &mut chain,
        depth: 2,
        value: 5,
        reached: 0,
    };
    let here = unsafe {
        protect(
            &mut chain,
            zn_test_raise_at_depth,
            &mut ctx as *mut _ as *mut c_void,
        )
    };
    assert_eq!(here, Err((7, 5)));
    assert_eq!(fiber.resume(), krio_fiber::FiberStep::Done);
    assert_eq!(result.get(), Some(Err((7, 77))));
}
