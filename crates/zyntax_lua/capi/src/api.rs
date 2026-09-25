//! The cores of the stack, access, push, table and userdata functions
//! of `lua.h`. Each takes the State pointer C code holds. One that can
//! raise returns [`OK`] or [`RAISED`], and its results go through
//! pointers; the others return their result.

use std::ffi::{CStr, c_char, c_int, c_void};

use crate::bridge;
use crate::calls::{OK, RAISED, c_function_of, pending, raise_message};
use crate::state::{L, Slot, State, state};
use crate::values::{
    self, Any, LUA_TLIGHTUSERDATA, LUA_TNONE, LUA_TSTRING, LUA_TTABLE, LUA_TUSERDATA, Number, items,
};

// ─── the stack ──────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_absindex(l: L, idx: c_int) -> c_int {
    unsafe { state(l) }.absindex(idx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_gettop(l: L) -> c_int {
    let st = unsafe { state(l) };
    (st.top - st.base()) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_settop(l: L, idx: c_int) {
    let st = unsafe { state(l) };
    let base = st.base();
    let top = if idx >= 0 {
        base + idx as usize
    } else {
        (st.top as i64 + idx as i64 + 1).max(base as i64) as usize
    };
    st.set_top(top);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_pushvalue(l: L, idx: c_int) {
    let st = unsafe { state(l) };
    let v = st.value(idx);
    st.push(v);
}

/// `lua_rotate`: the elements from `idx` to the top rotated `n`
/// positions toward the top.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_rotate(l: L, idx: c_int, n: c_int) {
    let st = unsafe { state(l) };
    let Slot::Stack(p) = st.slot(idx) else {
        return;
    };
    let t = st.top - 1;
    let m = if n >= 0 {
        t as i64 - n as i64
    } else {
        p as i64 - n as i64 - 1
    };
    if m < p as i64 - 1 || m > t as i64 {
        return;
    }
    let m = m as usize;
    // SAFETY: `p..=t` are live slots.
    let slots = unsafe { std::slice::from_raw_parts_mut(st.stack.add(p), t - p + 1) };
    let k = m + 1 - p;
    slots[..k].reverse();
    slots[k..].reverse();
    slots.reverse();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_copy(l: L, from: c_int, to: c_int) {
    let st = unsafe { state(l) };
    let v = st.value(from);
    let slot = st.slot(to);
    st.put(slot, v);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_checkstack(l: L, n: c_int) -> c_int {
    let st = unsafe { state(l) };
    (n >= 0 && st.reserve(n as usize)) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_xmove(from: L, to: L, n: c_int) {
    if from == to || n <= 0 {
        return;
    }
    let (a, b) = unsafe { (state(from), state(to)) };
    let n = (n as usize).min(a.top);
    for i in a.top - n..a.top {
        b.push(a.at(i));
    }
    let top = a.top - n;
    a.truncate(top);
}

// ─── access ─────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_type(l: L, idx: c_int) -> c_int {
    let st = unsafe { state(l) };
    match st.slot(idx) {
        Slot::None => LUA_TNONE,
        slot => unsafe { values::type_of(st.get(slot)) },
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_isnumber(l: L, idx: c_int) -> c_int {
    let mut isnum = 0;
    unsafe { zlc_tonumberx(l, idx, &mut isnum) };
    isnum
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_isstring(l: L, idx: c_int) -> c_int {
    let t = unsafe { zlc_type(l, idx) };
    (t == LUA_TSTRING || t == values::LUA_TNUMBER) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_isinteger(l: L, idx: c_int) -> c_int {
    let st = unsafe { state(l) };
    matches!(
        unsafe { values::number_of(st.value(idx)) },
        Some(Number::Int(_))
    ) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_iscfunction(l: L, idx: c_int) -> c_int {
    let st = unsafe { state(l) };
    unsafe { c_function_of(st.value(idx)) }.is_some() as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_isuserdata(l: L, idx: c_int) -> c_int {
    let t = unsafe { zlc_type(l, idx) };
    (t == LUA_TUSERDATA || t == LUA_TLIGHTUSERDATA) as c_int
}

/// The number a value is or a string converts to.
unsafe fn to_number(v: Any) -> Option<Number> {
    if let Some(n) = unsafe { values::number_of(v) } {
        return Some(n);
    }
    if unsafe { values::string_of(v) }.is_null() {
        return None;
    }
    unsafe { values::number_of((bridge().tonumber)(v)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_tonumberx(l: L, idx: c_int, isnum: *mut c_int) -> f64 {
    let st = unsafe { state(l) };
    let (n, ok) = match unsafe { to_number(st.value(idx)) } {
        Some(Number::Int(i)) => (i as f64, 1),
        Some(Number::Float(f)) => (f, 1),
        None => (0.0, 0),
    };
    if !isnum.is_null() {
        unsafe { *isnum = ok };
    }
    n
}

/// A float's integer value when it has one that fits.
fn float_to_int(f: f64) -> Option<i64> {
    if f.fract() == 0.0 && (-9223372036854775808.0..9223372036854775808.0).contains(&f) {
        Some(f as i64)
    } else {
        None
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_tointegerx(l: L, idx: c_int, isnum: *mut c_int) -> i64 {
    let st = unsafe { state(l) };
    let r = match unsafe { to_number(st.value(idx)) } {
        Some(Number::Int(i)) => Some(i),
        Some(Number::Float(f)) => float_to_int(f),
        None => None,
    };
    if !isnum.is_null() {
        unsafe { *isnum = r.is_some() as c_int };
    }
    r.unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_toboolean(l: L, idx: c_int) -> c_int {
    let st = unsafe { state(l) };
    unsafe { values::truth(st.value(idx)) as c_int }
}

/// `lua_tolstring`: a string's bytes, zero-terminated. A number becomes
/// a string where it lies. A string whose storage holds no terminator
/// is replaced where it lies by an equal one that does, which the
/// manual allows: the pointer is good while the value stays there.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_tolstring(l: L, idx: c_int, len: *mut usize) -> *const c_char {
    let st = unsafe { state(l) };
    let slot = st.slot(idx);
    let v = st.get(slot);
    let mut s = unsafe { values::string_of(v) };
    if s.is_null() {
        if unsafe { values::number_of(v) }.is_none() {
            if !len.is_null() {
                unsafe { *len = 0 };
            }
            return std::ptr::null();
        }
        let text = unsafe { values::bytes_of((bridge().number_str)(v)) };
        s = values::new_string(text);
        st.put(slot, values::box_string(s));
    } else if !unsafe { values::is_terminated(s) } {
        s = values::new_string(unsafe { values::bytes_of(s) });
        st.put(slot, values::box_string(s));
    }
    let bytes = unsafe { values::bytes_of(s) };
    if !len.is_null() {
        unsafe { *len = bytes.len() };
    }
    unsafe { zrtl::string::string_data(s) as *const c_char }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_rawlen(l: L, idx: c_int) -> u64 {
    let st = unsafe { state(l) };
    let v = st.value(idx);
    let s = unsafe { values::string_of(v) };
    if !s.is_null() {
        return unsafe { values::bytes_of(s) }.len() as u64;
    }
    match unsafe { values::type_of(v) } {
        LUA_TTABLE => (bridge().rawlen)(v) as u64,
        LUA_TUSERDATA if is_full_userdata(v) => unsafe { (*ud_header(v)).size as u64 },
        _ => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_tocfunction(l: L, idx: c_int) -> *const c_void {
    let st = unsafe { state(l) };
    match unsafe { c_function_of(st.value(idx)) } {
        Some(f) => f as *const c_void,
        None => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_touserdata(l: L, idx: c_int) -> *mut c_void {
    let st = unsafe { state(l) };
    let v = st.value(idx);
    if is_full_userdata(v) {
        return ud_payload(v) as *mut c_void;
    }
    if unsafe { values::has_tag(v, values::tags().light) } {
        return (unsafe { values::payload(v) }) as *mut c_void;
    }
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_tothread(l: L, idx: c_int) -> L {
    let st = unsafe { state(l) };
    let v = st.value(idx);
    if !unsafe { values::has_tag(v, values::tags().thread) } {
        return std::ptr::null_mut();
    }
    let main = (bridge().main_thread)();
    // SAFETY: both are live thread boxes.
    let s = if unsafe { values::payload(v) == values::payload(main) } {
        crate::main_state()
    } else {
        crate::thread_state(v)
    };
    unsafe { (*s).l() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_topointer(l: L, idx: c_int) -> *const c_void {
    let st = unsafe { state(l) };
    let v = st.value(idx);
    match unsafe { values::type_of(v) } {
        LUA_TUSERDATA if is_full_userdata(v) => ud_payload(v) as *const c_void,
        values::LUA_TNIL | values::LUA_TBOOLEAN | values::LUA_TNUMBER => std::ptr::null(),
        _ => (unsafe { values::payload(v) }) as *const c_void,
    }
}

// ─── pushes ─────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_pushnil(l: L) {
    unsafe { state(l) }.push(std::ptr::null());
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_pushnumber(l: L, n: f64) {
    unsafe { state(l) }.push(values::box_float(n));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_pushinteger(l: L, n: i64) {
    unsafe { state(l) }.push(values::box_int(n));
}

/// `lua_pushlstring`: the internal copy's bytes, zero-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_pushlstring(l: L, s: *const c_char, len: usize) -> *const c_char {
    let bytes: &[u8] = if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(s as *const u8, len) }
    };
    let string = values::new_string(bytes);
    unsafe { state(l) }.push(values::box_string(string));
    unsafe { zrtl::string::string_data(string) as *const c_char }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_pushboolean(l: L, b: c_int) {
    unsafe { state(l) }.push(values::box_bool(b != 0));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_pushlightuserdata(l: L, p: *mut c_void) {
    unsafe { state(l) }.push(values::box_pointer(p as *mut u8, values::tags().light));
}

/// `lua_pushthread`: whether the thread is the main one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_pushthread(l: L) -> c_int {
    let st = unsafe { state(l) };
    if st.thread.is_null() {
        st.push((bridge().main_thread)());
        1
    } else {
        let t = st.thread;
        st.push(t);
        0
    }
}

// ─── tables ─────────────────────────────────────────────────────────

/// A key given as a C string.
unsafe fn key(k: *const c_char) -> Any {
    values::box_bytes(unsafe { CStr::from_ptr(k) }.to_bytes())
}

/// Push `v` and report its type, or report a raise.
fn pushed(st: &mut State, v: Any, ty: *mut c_int) -> c_int {
    if pending() {
        return RAISED;
    }
    st.push(v);
    if !ty.is_null() {
        unsafe { *ty = values::type_of(v) };
    }
    OK
}

/// The table at `idx`, or a raise when it is not one.
fn table_at(st: &State, idx: c_int) -> Result<Any, c_int> {
    let t = st.value(idx);
    if unsafe { values::type_of(t) } == LUA_TTABLE {
        Ok(t)
    } else {
        Err(raise_message("table expected"))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_getglobal(l: L, name: *const c_char, ty: *mut c_int) -> c_int {
    let st = unsafe { state(l) };
    let b = bridge();
    let k = unsafe { key(name) };
    let v = (b.index)((b.globals)(), k);
    pushed(st, v, ty)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_gettable(l: L, idx: c_int, ty: *mut c_int) -> c_int {
    let st = unsafe { state(l) };
    let t = st.value(idx);
    let k = st.at(st.top - 1);
    let v = (bridge().index)(t, k);
    if pending() {
        return RAISED;
    }
    st.pop();
    pushed(st, v, ty)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_getfield(l: L, idx: c_int, k: *const c_char, ty: *mut c_int) -> c_int {
    let st = unsafe { state(l) };
    let t = st.value(idx);
    let v = (bridge().index)(t, unsafe { key(k) });
    pushed(st, v, ty)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_geti(l: L, idx: c_int, n: i64, ty: *mut c_int) -> c_int {
    let st = unsafe { state(l) };
    let t = st.value(idx);
    let v = (bridge().index)(t, values::box_int(n));
    pushed(st, v, ty)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_rawget(l: L, idx: c_int, ty: *mut c_int) -> c_int {
    let st = unsafe { state(l) };
    let t = match table_at(st, idx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let k = st.at(st.top - 1);
    let v = (bridge().rawget)(t, k);
    st.pop();
    pushed(st, v, ty)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_rawgeti(l: L, idx: c_int, n: i64, ty: *mut c_int) -> c_int {
    let st = unsafe { state(l) };
    let t = match table_at(st, idx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let v = (bridge().rawgeti)(t, n);
    pushed(st, v, ty)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_rawgetp(l: L, idx: c_int, p: *const c_void, ty: *mut c_int) -> c_int {
    let st = unsafe { state(l) };
    let t = match table_at(st, idx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let k = values::box_pointer(p as *mut u8, values::tags().light);
    let v = (bridge().rawget)(t, k);
    pushed(st, v, ty)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_createtable(l: L, _narr: c_int, _nrec: c_int) {
    let st = unsafe { state(l) };
    let t = (bridge().new_table)();
    st.push(t);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_setglobal(l: L, name: *const c_char) -> c_int {
    let st = unsafe { state(l) };
    let b = bridge();
    let k = unsafe { key(name) };
    let v = st.at(st.top - 1);
    (b.setindex)((b.globals)(), k, v);
    if pending() {
        return RAISED;
    }
    st.pop();
    OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_settable(l: L, idx: c_int) -> c_int {
    let st = unsafe { state(l) };
    let t = st.value(idx);
    let (k, v) = (st.at(st.top - 2), st.at(st.top - 1));
    (bridge().setindex)(t, k, v);
    if pending() {
        return RAISED;
    }
    let top = st.top - 2;
    st.truncate(top);
    OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_setfield(l: L, idx: c_int, k: *const c_char) -> c_int {
    let st = unsafe { state(l) };
    let t = st.value(idx);
    let k = unsafe { key(k) };
    let v = st.at(st.top - 1);
    (bridge().setindex)(t, k, v);
    if pending() {
        return RAISED;
    }
    st.pop();
    OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_seti(l: L, idx: c_int, n: i64) -> c_int {
    let st = unsafe { state(l) };
    let t = st.value(idx);
    let v = st.at(st.top - 1);
    (bridge().setindex)(t, values::box_int(n), v);
    if pending() {
        return RAISED;
    }
    st.pop();
    OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_rawset(l: L, idx: c_int) -> c_int {
    let st = unsafe { state(l) };
    let t = match table_at(st, idx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let (k, v) = (st.at(st.top - 2), st.at(st.top - 1));
    (bridge().rawset)(t, k, v);
    if pending() {
        return RAISED;
    }
    let top = st.top - 2;
    st.truncate(top);
    OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_rawseti(l: L, idx: c_int, n: i64) -> c_int {
    let st = unsafe { state(l) };
    let t = match table_at(st, idx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let v = st.at(st.top - 1);
    (bridge().rawseti)(t, n, v);
    st.pop();
    OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_rawsetp(l: L, idx: c_int, p: *const c_void) -> c_int {
    let st = unsafe { state(l) };
    let t = match table_at(st, idx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let k = values::box_pointer(p as *mut u8, values::tags().light);
    let v = st.at(st.top - 1);
    (bridge().rawset)(t, k, v);
    if pending() {
        return RAISED;
    }
    st.pop();
    OK
}

/// `lua_next`: `*more` is whether a pair was pushed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_next(l: L, idx: c_int, more: *mut c_int) -> c_int {
    let st = unsafe { state(l) };
    let t = match table_at(st, idx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let k = st.at(st.top - 1);
    let pair = (bridge().next)(t, k);
    if pending() {
        return RAISED;
    }
    st.pop();
    // SAFETY: the bridge's list.
    let kv = unsafe { items(pair) };
    if kv.len() < 2 {
        unsafe { *more = 0 };
        return OK;
    }
    let (k, v) = (kv[0], kv[1]);
    st.push(k);
    st.push(v);
    unsafe { *more = 1 };
    OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_rawequal(l: L, a: c_int, b: c_int) -> c_int {
    let st = unsafe { state(l) };
    let (sa, sb) = (st.slot(a), st.slot(b));
    if sa == Slot::None || sb == Slot::None {
        return 0;
    }
    ((bridge().rawequal)(st.get(sa), st.get(sb)) != 0) as c_int
}

/// `lua_compare`: `*result` is whether the relation holds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_compare(
    l: L,
    a: c_int,
    b: c_int,
    op: c_int,
    result: *mut c_int,
) -> c_int {
    let st = unsafe { state(l) };
    let (sa, sb) = (st.slot(a), st.slot(b));
    if sa == Slot::None || sb == Slot::None || !(0..=2).contains(&op) {
        unsafe { *result = 0 };
        return OK;
    }
    let r = (bridge().compare)(st.get(sa), st.get(sb), op as i64);
    if pending() {
        return RAISED;
    }
    unsafe { *result = (r != 0) as c_int };
    OK
}

/// The library's operator code for a `LUA_OP*` code.
fn operator(op: c_int) -> Option<i64> {
    Some(match op {
        0 => 0,   // add
        1 => 1,   // sub
        2 => 2,   // mul
        3 => 4,   // mod
        4 => 5,   // pow
        5 => 3,   // div
        6 => 6,   // idiv
        7 => 7,   // band
        8 => 8,   // bor
        9 => 9,   // bxor
        10 => 10, // shl
        11 => 11, // shr
        12 => 13, // unm
        13 => 14, // bnot
        _ => return None,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_arith(l: L, op: c_int) -> c_int {
    let st = unsafe { state(l) };
    let Some(code) = operator(op) else {
        return raise_message("invalid arithmetic operator");
    };
    let unary = op >= 12;
    let n = if unary { 1 } else { 2 };
    let a = st.at(st.top - n);
    let b = if unary { a } else { st.at(st.top - 1) };
    let r = (bridge().arith)(code, a, b);
    if pending() {
        return RAISED;
    }
    let top = st.top - n;
    st.truncate(top);
    st.push(r);
    OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_concat(l: L, n: c_int) -> c_int {
    let st = unsafe { state(l) };
    let b = bridge();
    if n <= 0 {
        st.push(values::box_bytes(b""));
        return OK;
    }
    let n = n as usize;
    if n == 1 {
        return OK;
    }
    let list = (b.list_new)();
    for i in st.top - n..st.top {
        (b.list_push)(list, st.at(i));
    }
    let r = (b.concat)(list);
    if pending() {
        return RAISED;
    }
    let top = st.top - n;
    st.truncate(top);
    st.push(r);
    OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_len(l: L, idx: c_int) -> c_int {
    let st = unsafe { state(l) };
    let r = (bridge().len)(st.value(idx));
    if pending() {
        return RAISED;
    }
    st.push(r);
    OK
}

/// `lua_stringtonumber`: the size of the string with its terminator,
/// or 0 when it is no numeral.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_stringtonumber(l: L, s: *const c_char) -> usize {
    let st = unsafe { state(l) };
    let bytes = unsafe { CStr::from_ptr(s) }.to_bytes();
    let n = (bridge().tonumber)(values::box_bytes(bytes));
    if n.is_null() {
        return 0;
    }
    st.push(n);
    bytes.len() + 1
}

// ─── metatables ─────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_getmetatable(l: L, idx: c_int) -> c_int {
    let st = unsafe { state(l) };
    let v = st.value(idx);
    let mt = if is_full_userdata(v) {
        unsafe { (*ud_header(v)).meta }
    } else {
        (bridge().getmetatable)(v)
    };
    if mt.is_null() {
        return 0;
    }
    st.push(mt);
    1
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_setmetatable(l: L, idx: c_int) -> c_int {
    let st = unsafe { state(l) };
    let v = st.value(idx);
    let mt = st.at(st.top - 1);
    if !mt.is_null() && unsafe { values::type_of(mt) } != LUA_TTABLE {
        return raise_message("table expected");
    }
    if is_full_userdata(v) {
        unsafe { (*ud_header(v)).meta = mt };
    } else {
        (bridge().setmetatable)(v, mt);
        if pending() {
            return RAISED;
        }
    }
    st.pop();
    OK
}

// ─── userdata ───────────────────────────────────────────────────────

/// The block a full userdata's box points at: this header, its user
/// values, then its payload at the next sixteen-byte boundary.
#[repr(C)]
pub struct UdHeader {
    pub meta: Any,
    pub nuv: i64,
    pub size: i64,
    _pad: i64,
}

fn payload_offset(nuv: usize) -> usize {
    (std::mem::size_of::<UdHeader>() + nuv * std::mem::size_of::<Any>()).next_multiple_of(16)
}

pub fn is_full_userdata(v: Any) -> bool {
    unsafe { values::has_tag(v, values::tags().userdata) }
}

fn ud_header(v: Any) -> *mut UdHeader {
    unsafe { values::payload(v) as *mut UdHeader }
}

fn ud_payload(v: Any) -> *mut u8 {
    let h = ud_header(v);
    unsafe { (h as *mut u8).add(payload_offset((*h).nuv as usize)) }
}

fn ud_values(v: Any) -> *mut Any {
    unsafe { (ud_header(v) as *mut u8).add(std::mem::size_of::<UdHeader>()) as *mut Any }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_newuserdatauv(l: L, size: usize, nuv: c_int) -> *mut c_void {
    let st = unsafe { state(l) };
    let nuv = nuv.max(0) as usize;
    let block = values::zeroed(payload_offset(nuv) + size.max(1));
    // SAFETY: a fresh block with room for the header.
    unsafe {
        (block as *mut UdHeader).write(UdHeader {
            meta: std::ptr::null(),
            nuv: nuv as i64,
            size: size as i64,
            _pad: 0,
        });
    }
    let v = values::box_pointer(block, values::tags().userdata);
    st.push(v);
    ud_payload(v) as *mut c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_getiuservalue(l: L, idx: c_int, n: c_int) -> c_int {
    let st = unsafe { state(l) };
    let v = st.value(idx);
    if !is_full_userdata(v) || n < 1 || n as i64 > unsafe { (*ud_header(v)).nuv } {
        st.push(std::ptr::null());
        return LUA_TNONE;
    }
    let u = unsafe { *ud_values(v).add(n as usize - 1) };
    st.push(u);
    unsafe { values::type_of(u) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_setiuservalue(l: L, idx: c_int, n: c_int) -> c_int {
    let st = unsafe { state(l) };
    let v = st.value(idx);
    let u = st.pop();
    if !is_full_userdata(v) || n < 1 || n as i64 > unsafe { (*ud_header(v)).nuv } {
        return 0;
    }
    unsafe { *ud_values(v).add(n as usize - 1) = u };
    1
}

/// `$LuaC$ud_meta`: a full userdata's metatable, or nil.
pub extern "C" fn c_ud_meta(v: Any) -> Any {
    if is_full_userdata(v) {
        unsafe { (*ud_header(v)).meta }
    } else {
        std::ptr::null()
    }
}

/// `$LuaC$ud_set_meta`: set a full userdata's metatable.
pub extern "C" fn c_ud_set_meta(v: Any, mt: Any) {
    if is_full_userdata(v) {
        unsafe { (*ud_header(v)).meta = mt };
    }
}

// ─── upvalues ───────────────────────────────────────────────────────

/// `lua_getupvalue` for a C function: upvalue `n` pushed, and whether
/// there is one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_getupvalue(l: L, funcindex: c_int, n: c_int, set: c_int) -> c_int {
    let st = unsafe { state(l) };
    let f = st.value(funcindex);
    if unsafe { c_function_of(f) }.is_none() || n < 1 {
        return 0;
    }
    let rec = unsafe { values::list_of(f) };
    let k = crate::state::RECORD_UPVALUES + n as usize - 1;
    if k >= unsafe { items(rec) }.len() {
        return 0;
    }
    if set != 0 {
        let v = st.pop();
        unsafe { *(*rec).data.add(k) = v };
    } else {
        let v = unsafe { items(rec)[k] };
        st.push(v);
    }
    1
}

// ─── the collector and the debug interface ──────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_gc(_l: L, what: c_int, arg: c_int) -> c_int {
    (bridge().gc)(what as i64, arg as i64) as c_int
}

/// Whether stack level `level` exists: 0 is the running C function, 1
/// the code that called it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_level(l: L, level: c_int) -> c_int {
    let st = unsafe { state(l) };
    (!st.ci.is_null() && (0..=1).contains(&level)) as c_int
}

/// Push the running C function at level 0, nil at any other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_push_function(l: L, level: c_int) {
    let st = unsafe { state(l) };
    let v = match unsafe { st.ci.as_ref() } {
        Some(ci) if level == 0 && !ci.rec.is_null() => {
            values::box_pointer(ci.rec as *mut u8, values::tags().function)
        }
        _ => std::ptr::null(),
    };
    st.push(v);
}

/// Where the code that called the running C function is: its line (0
/// when it has none, as for C) and its chunk's name in `buf`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_caller(l: L, buf: *mut c_char, size: usize) -> c_int {
    let st = unsafe { state(l) };
    let Some(ci) = (unsafe { st.ci.as_ref() }) else {
        return 0;
    };
    let line = ci.caller_line;
    let number = line & ((1i64 << crate::bridge::LINE_BITS) - 1);
    if number <= 0 || size == 0 {
        return 0;
    }
    let name = unsafe { values::bytes_of((bridge().chunk_of)(line)) };
    let n = name.len().min(size - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(name.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
    number as c_int
}
