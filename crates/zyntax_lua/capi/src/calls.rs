//! Calls across the boundary: Lua calling a C function, and C calling
//! a Lua value, protected or not.
//!
//! A C function is an ordinary function value whose record is `[code,
//! arity, cfn, upvalue...]`: its code is [`capi_call`], variadic, and
//! `cfn` is the C function it calls. No compiled code calls a C
//! function's own address; only C does, through its real prototype.

use std::ffi::c_int;

use crate::bridge;
use crate::state::{CFrame, CFunction, L, RECORD_UPVALUES, State, state};
use crate::values::{self, Any, List, items};

/// What a core returns: done, or an error is pending and the veneer
/// raises it.
pub const OK: c_int = 0;
pub const RAISED: c_int = 1;

/// `LUA_MULTRET`.
const MULTRET: c_int = -1;
/// The statuses of `lua_pcall`.
const LUA_OK: c_int = 0;
const LUA_ERRRUN: c_int = 2;
const LUA_ERRERR: c_int = 5;

unsafe extern "C" {
    /// Call `f(L)` under a protect frame on L's chain: 0 with its result
    /// count in `*n`, or the status a raise carried.
    fn zlc_invoke(l: L, f: CFunction, n: *mut c_int) -> c_int;
}

/// Whether an error is pending.
pub fn pending() -> bool {
    !(bridge().pending)().is_null()
}

/// Raise `message` as an error value.
pub fn raise_message(message: &str) -> c_int {
    (bridge().raise)(values::box_bytes(message.as_bytes()));
    RAISED
}

/// The code of every C function value: calls the record's C function
/// with the arguments on a fresh frame of the running thread's API
/// stack, and returns its results as one value. When the C function
/// raises, the error is left pending and the value is nil, which the
/// caller's check for a pending error takes over.
///
/// `args` is a boxed tuple or list of the arguments, or nil for none.
pub extern "C" fn capi_call(rec: List, args: Any) -> Any {
    let b = bridge();
    let s = crate::current_state();
    // SAFETY: the State lives while its thread does, and this thread is
    // running it.
    let st = unsafe { &mut *s };
    let base = st.top;
    let arguments = if args.is_null() {
        &[][..]
    } else {
        // SAFETY: the caller passes a boxed list of values.
        unsafe { items(values::list_of(args)) }
    };
    assert!(
        st.reserve(arguments.len() + crate::state::MINSTACK),
        "C stack overflow"
    );
    for &a in arguments {
        st.push(a);
    }
    let mut frame = CFrame {
        prev: st.ci,
        base,
        rec,
        caller_line: (b.line)(),
    };
    st.ci = &mut frame;
    // Errors the API raises inside the C function carry no position,
    // as in the reference, where the running function is not Lua.
    (b.set_line)(0);
    // SAFETY: the record is a C function's, whose third item is the
    // address of a `lua_CFunction`.
    let cfn: CFunction = unsafe {
        let rec_items = items(rec);
        std::mem::transmute::<*mut u8, CFunction>(values::payload(rec_items[2]))
    };
    let mut n: c_int = 0;
    // SAFETY: the State's pointer, and a C function to call with it.
    let status = unsafe { zlc_invoke(st.l(), cfn, &mut n) };
    let result = if status == 0 {
        let n = (n.max(0) as usize).min(st.top - base);
        results(st, st.top - n, st.top)
    } else {
        std::ptr::null()
    };
    st.truncate(base);
    st.ci = frame.prev;
    (b.set_line)(frame.caller_line);
    result
}

/// Slots `from..to` as one value: none, the value itself, or a tuple.
fn results(st: &State, from: usize, to: usize) -> Any {
    let b = bridge();
    if to - from == 1 {
        return st.at(from);
    }
    let list = (b.list_new)();
    for i in from..to {
        (b.list_push)(list, st.at(i));
    }
    (b.pack)(list)
}

/// The function and `nargs` arguments on top, as the function and a
/// list; they stay on the stack.
fn callee(st: &State, nargs: usize) -> (Any, List) {
    let b = bridge();
    let at = st.top - nargs - 1;
    let list = (b.list_new)();
    for i in at + 1..st.top {
        (b.list_push)(list, st.at(i));
    }
    (st.at(at), list)
}

/// Push the values of `v`, adjusted to `nresults` unless it is
/// `LUA_MULTRET`.
fn push_results(st: &mut State, v: List, skip: usize, nresults: c_int) {
    // SAFETY: a list the bridge made.
    let vs = unsafe { items(v) };
    let vs = if vs.len() > skip { &vs[skip..] } else { &[] };
    let wanted = if nresults == MULTRET {
        vs.len()
    } else {
        nresults.max(0) as usize
    };
    assert!(st.reserve(wanted), "C stack overflow");
    for i in 0..wanted {
        st.push(vs.get(i).copied().unwrap_or(std::ptr::null()));
    }
}

/// `lua_callk`, without a continuation.
///
/// # Safety
/// `l` is a State's pointer with the function and `nargs` arguments on
/// top.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_call(l: L, nargs: c_int, nresults: c_int) -> c_int {
    let st = unsafe { state(l) };
    let b = bridge();
    let nargs = nargs.max(0) as usize;
    let (f, args) = callee(st, nargs);
    (b.set_line)(0);
    let r = (b.call)(f, args);
    (b.set_line)(0);
    let at = st.top - nargs - 1;
    if pending() {
        st.truncate(at);
        return RAISED;
    }
    let vs = (b.values)(r);
    st.truncate(at);
    push_results(st, vs, 0, nresults);
    OK
}

/// `lua_pcallk`, without a continuation: the status in `*status`.
/// Raises only the marker of a coroutine being closed, which nothing
/// catches.
///
/// # Safety
/// As [`zlc_call`], and `errfunc` 0 or a valid index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_pcall(
    l: L,
    nargs: c_int,
    nresults: c_int,
    errfunc: c_int,
    status: *mut c_int,
) -> c_int {
    let st = unsafe { state(l) };
    let b = bridge();
    let nargs = nargs.max(0) as usize;
    let handler = if errfunc == 0 {
        std::ptr::null()
    } else {
        st.value(errfunc)
    };
    let (f, args) = callee(st, nargs);
    (b.set_line)(0);
    let r = (b.pcall)(f, args, handler);
    (b.set_line)(0);
    let at = st.top - nargs - 1;
    if r.is_null() {
        st.truncate(at);
        return RAISED;
    }
    let vs = (b.values)(r);
    // SAFETY: `pcall`'s tuple: whether the call succeeded, then its
    // results or the error. A handler that failed on its own error
    // leaves the message `xpcall` gives for that.
    let (ok, err) = unsafe {
        let got = items(vs);
        (
            got.first().is_some_and(|&v| values::truth(v)),
            got.get(1).copied().unwrap_or(std::ptr::null()),
        )
    };
    let code = if ok {
        LUA_OK
    } else if !handler.is_null()
        && unsafe { values::bytes_of(values::string_of(err)) } == b"error in error handling"
    {
        LUA_ERRERR
    } else {
        LUA_ERRRUN
    };
    st.truncate(at);
    if code == 0 {
        push_results(st, vs, 1, nresults);
    } else {
        push_results(st, vs, 1, 1);
    }
    unsafe { *status = code };
    OK
}

/// `lua_error`'s core: the value on top becomes the pending error.
///
/// # Safety
/// `l` is a State's pointer with a value on top.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_error(l: L) {
    let st = unsafe { state(l) };
    let v = if st.top > st.base() {
        st.pop()
    } else {
        std::ptr::null()
    };
    (bridge().raise)(v);
}

/// The pending error as text for a panic, taken.
///
/// # Safety
/// `buf` holds `size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_panic_text(_l: L, buf: *mut u8, size: usize) {
    let v = (bridge().take)();
    let text: Vec<u8> = unsafe {
        match values::number_of(v) {
            Some(values::Number::Int(i)) => i.to_string().into_bytes(),
            Some(values::Number::Float(_)) => values::bytes_of((bridge().number_str)(v)).to_vec(),
            None => {
                let s = values::string_of(v);
                if s.is_null() {
                    b"error object is not a string".to_vec()
                } else {
                    values::bytes_of(s).to_vec()
                }
            }
        }
    };
    if size == 0 {
        return;
    }
    let n = text.len().min(size - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buf, n);
        *buf.add(n) = 0;
    }
}

/// `lua_atpanic`'s core.
///
/// # Safety
/// `l` is a State's pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_atpanic(l: L, f: Option<CFunction>) -> Option<CFunction> {
    let st = unsafe { state(l) };
    let g = unsafe { &mut *st.global };
    std::mem::replace(&mut g.panic, f)
}

/// The panic function, if any.
///
/// # Safety
/// `l` is a State's pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_panicf(l: L) -> Option<CFunction> {
    let st = unsafe { state(l) };
    unsafe { (*st.global).panic }
}

/// Whether `v` is a C function's value.
///
/// # Safety
/// `v` is null or a live box.
pub unsafe fn c_function_of(v: Any) -> Option<CFunction> {
    if !unsafe { values::has_tag(v, values::tags().function) } {
        return None;
    }
    let rec = unsafe { items(values::list_of(v)) };
    if rec.len() < RECORD_UPVALUES {
        return None;
    }
    let code = unsafe { values::payload(rec[0]) };
    if code as usize != capi_call as *const () as usize {
        return None;
    }
    Some(unsafe { std::mem::transmute::<*mut u8, CFunction>(values::payload(rec[2])) })
}

/// The value of C function `f` without upvalues: the same value
/// however often it is asked for.
pub fn light_function(f: CFunction) -> Any {
    let b = bridge();
    // SAFETY: the main State and its Global are made before C code runs.
    let light = unsafe { (*(*crate::main_state()).global).light };
    let key = f as *const () as usize as i64;
    let found = (b.rawgeti)(light, key);
    if !found.is_null() {
        return found;
    }
    let fv = (b.func_new)(capi_call as *const () as usize as i64, key, (b.list_new)());
    (b.rawseti)(light, key, fv);
    fv
}

/// `lua_pushcclosure`: a C function with the `n` values on top as its
/// upvalues, which are popped. Without upvalues the same function is
/// always the same value.
///
/// # Safety
/// `l` is a State's pointer with `n` values on top.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_pushcclosure(l: L, f: CFunction, n: c_int) {
    let st = unsafe { state(l) };
    let b = bridge();
    let n = n.max(0) as usize;
    if n == 0 {
        st.push(light_function(f));
        return;
    }
    let ups = (b.list_new)();
    for i in st.top - n..st.top {
        (b.list_push)(ups, st.at(i));
    }
    let fv = (b.func_new)(
        capi_call as *const () as usize as i64,
        f as *const () as usize as i64,
        ups,
    );
    st.push(fv);
    let top = st.top;
    st.set_at(top - 1 - n, fv);
    st.truncate(top - n);
}
