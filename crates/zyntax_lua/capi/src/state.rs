//! The `lua_State` C code holds, its API stack, and the index rules.
//!
//! A State is a block of the program's heap, one per Lua thread that
//! has run C code: the main one, rooted for the life of the process,
//! and one per coroutine, kept in the coroutine's record so it lives
//! exactly as long as the coroutine. The pointer C code holds is
//! `LUA_EXTRASPACE` bytes into the block, so `lua_getextraspace` is
//! the block's first word, and the word it points at is the State's
//! chain of protect frames.
//!
//! The API stack is another block: the values C code works on, nil as
//! the null word. Every slot above the top is zero, so the collector,
//! which reads a block to its end, sees only live values in it.

use std::ffi::c_void;

use crate::values::{Any, List, items};

/// `LUA_EXTRASPACE`: the bytes before the pointer C code holds.
pub const EXTRA: usize = std::mem::size_of::<*mut c_void>();
/// `LUA_MINSTACK`: the slots free at every entry to C code.
pub const MINSTACK: usize = 20;
/// `LUAI_MAXSTACK`: the most slots `lua_checkstack` grants.
pub const MAXSTACK: usize = 1_000_000;
/// `LUA_REGISTRYINDEX`.
pub const REGISTRYINDEX: i32 = -(MAXSTACK as i32) - 1000;

/// What C code knows as `lua_State *`.
pub type L = *mut c_void;

/// A C function, `lua_CFunction`.
pub type CFunction = unsafe extern "C" fn(L) -> i32;

#[repr(C)]
pub struct State {
    /// `lua_getextraspace`.
    pub extra: *mut c_void,
    /// The innermost protect frame a raise on this State jumps to.
    pub chain: zrtl_native::Chain,
    /// The API stack: `cap` slots, `top` of them in use.
    pub stack: *mut Any,
    pub top: usize,
    pub cap: usize,
    /// The innermost C function running on this thread, or null.
    pub ci: *mut CFrame,
    /// The coroutine this State belongs to, null for the main thread.
    pub thread: Any,
    pub global: *mut Global,
}

// The C side finds the chain at the pointer it holds.
const _: () = assert!(std::mem::offset_of!(State, chain) == EXTRA);

/// What every State of the process shares.
#[repr(C)]
pub struct Global {
    pub main: *mut State,
    /// The function values of light C functions, by address, so the
    /// same function pushed twice is the same value.
    pub light: Any,
    /// `lua_atpanic`.
    pub panic: Option<CFunction>,
}

/// One running C function: where its slots start on the API stack,
/// the record of the closure it is, and the line of the Lua code that
/// called it. Lives on the Rust stack of the call.
#[repr(C)]
pub struct CFrame {
    pub prev: *mut CFrame,
    pub base: usize,
    pub rec: List,
    pub caller_line: i64,
}

/// Where an acceptable index leads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Stack(usize),
    /// Item `n` of the running closure's record.
    Upvalue(usize),
    Registry,
    /// Past the top, or an upvalue the closure does not have.
    None,
}

/// The State behind the pointer C code holds.
///
/// # Safety
/// `l` came from [`State::l`].
pub unsafe fn state<'a>(l: L) -> &'a mut State {
    unsafe { &mut *((l as *mut u8).sub(EXTRA) as *mut State) }
}

/// Items of a record before its upvalues: code, arity, C function.
pub const RECORD_UPVALUES: usize = 3;

impl State {
    /// A State with an empty stack, on the program's heap.
    pub fn create(global: *mut Global, thread: Any) -> *mut State {
        let s = crate::values::zeroed(std::mem::size_of::<State>()) as *mut State;
        let cap = 2 * MINSTACK;
        let stack = crate::values::zeroed(cap * std::mem::size_of::<Any>()) as *mut Any;
        // SAFETY: a fresh block of the State's size.
        unsafe {
            s.write(State {
                extra: std::ptr::null_mut(),
                chain: std::ptr::null_mut(),
                stack,
                top: 0,
                cap,
                ci: std::ptr::null_mut(),
                thread,
                global,
            });
        }
        s
    }

    /// The pointer C code holds for this State.
    pub fn l(&mut self) -> L {
        // SAFETY: inside the State's own block.
        unsafe { (self as *mut State as *mut u8).add(EXTRA) as L }
    }

    pub fn base(&self) -> usize {
        // SAFETY: a frame is live while it is on the chain of frames.
        unsafe { self.ci.as_ref() }.map_or(0, |ci| ci.base)
    }

    pub fn slots(&self) -> &[Any] {
        // SAFETY: `top` slots of the stack are in use.
        unsafe { std::slice::from_raw_parts(self.stack, self.top) }
    }

    /// Room for `n` more slots, growing the stack: false past
    /// `MAXSTACK`.
    pub fn reserve(&mut self, n: usize) -> bool {
        let need = self.top + n;
        if need > MAXSTACK {
            return false;
        }
        if need <= self.cap {
            return true;
        }
        let cap = need.max(self.cap * 2).min(MAXSTACK);
        let stack = crate::values::zeroed(cap * std::mem::size_of::<Any>()) as *mut Any;
        // SAFETY: both blocks hold at least `top` slots, and nothing
        // holds a pointer into the old one past this call.
        unsafe {
            std::ptr::copy_nonoverlapping(self.stack, stack, self.top);
            zrtl::heap::free(
                self.stack as *mut u8,
                self.cap * std::mem::size_of::<Any>(),
                16,
            );
        }
        self.stack = stack;
        self.cap = cap;
        true
    }

    pub fn push(&mut self, v: Any) {
        if self.top == self.cap {
            assert!(self.reserve(1), "C stack overflow");
        }
        // SAFETY: `top < cap`.
        unsafe { *self.stack.add(self.top) = v };
        self.top += 1;
    }

    /// Drop slots down to `top`, zeroing what they held.
    pub fn truncate(&mut self, top: usize) {
        while self.top > top {
            self.top -= 1;
            // SAFETY: below `cap`.
            unsafe { *self.stack.add(self.top) = std::ptr::null() };
        }
    }

    /// Set the top to `top`, the new slots nil.
    pub fn set_top(&mut self, top: usize) {
        if top < self.top {
            self.truncate(top);
        } else {
            assert!(self.reserve(top - self.top), "C stack overflow");
            self.top = top;
        }
    }

    pub fn at(&self, i: usize) -> Any {
        self.slots()[i]
    }

    pub fn set_at(&mut self, i: usize, v: Any) {
        assert!(i < self.top);
        // SAFETY: below `top`.
        unsafe { *self.stack.add(i) = v };
    }

    /// The value on top, and the slot popped.
    pub fn pop(&mut self) -> Any {
        let v = self.at(self.top - 1);
        self.truncate(self.top - 1);
        v
    }

    /// Where `idx` leads, as `index2value` resolves it.
    pub fn slot(&self, idx: i32) -> Slot {
        let base = self.base();
        if idx > 0 {
            let i = base + idx as usize - 1;
            if i < self.top {
                Slot::Stack(i)
            } else {
                Slot::None
            }
        } else if idx > REGISTRYINDEX {
            let i = self.top as i64 + idx as i64;
            if idx != 0 && i >= base as i64 {
                Slot::Stack(i as usize)
            } else {
                Slot::None
            }
        } else if idx == REGISTRYINDEX {
            Slot::Registry
        } else {
            let n = (REGISTRYINDEX - idx) as usize;
            // SAFETY: a frame is live while it is on the chain.
            match unsafe { self.ci.as_ref() } {
                Some(ci) if !ci.rec.is_null() => {
                    let rec = unsafe { items(ci.rec) };
                    if n >= 1 && RECORD_UPVALUES + n - 1 < rec.len() {
                        Slot::Upvalue(RECORD_UPVALUES + n - 1)
                    } else {
                        Slot::None
                    }
                }
                _ => Slot::None,
            }
        }
    }

    /// The value at a slot; nil for none.
    pub fn get(&self, slot: Slot) -> Any {
        match slot {
            Slot::Stack(i) => self.at(i),
            Slot::Upvalue(k) => {
                // SAFETY: the slot was resolved against the live frame.
                let rec = unsafe { (*self.ci).rec };
                unsafe { items(rec)[k] }
            }
            Slot::Registry => (crate::bridge().registry)(),
            Slot::None => std::ptr::null(),
        }
    }

    /// The value at an acceptable index.
    pub fn value(&self, idx: i32) -> Any {
        self.get(self.slot(idx))
    }

    /// Store into a slot that holds a value.
    pub fn put(&mut self, slot: Slot, v: Any) {
        match slot {
            Slot::Stack(i) => self.set_at(i, v),
            Slot::Upvalue(k) => {
                // SAFETY: the slot was resolved against the live frame,
                // and the record holds item `k`.
                unsafe {
                    let rec = (*self.ci).rec;
                    *(*rec).data.add(k) = v;
                }
            }
            Slot::Registry | Slot::None => {}
        }
    }

    /// The absolute index of an acceptable index.
    pub fn absindex(&self, idx: i32) -> i32 {
        if idx > 0 || idx <= REGISTRYINDEX {
            idx
        } else {
            (self.top - self.base()) as i32 + idx + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(n: usize) -> Any {
        n as Any
    }

    fn fresh() -> &'static mut State {
        let s = State::create(std::ptr::null_mut(), std::ptr::null());
        unsafe { &mut *s }
    }

    #[test]
    fn positive_indices_count_from_the_frame_base() {
        let s = fresh();
        for n in 1..=3 {
            s.push(value(n));
        }
        let mut frame = CFrame {
            prev: std::ptr::null_mut(),
            base: 1,
            rec: std::ptr::null_mut(),
            caller_line: 0,
        };
        s.ci = &mut frame;
        assert_eq!(s.slot(1), Slot::Stack(1));
        assert_eq!(s.slot(2), Slot::Stack(2));
        assert_eq!(s.slot(3), Slot::None, "past the top");
        assert_eq!(s.slot(-1), Slot::Stack(2));
        assert_eq!(s.slot(-2), Slot::Stack(1));
        assert_eq!(s.slot(-3), Slot::None, "below the frame");
        assert_eq!(s.slot(REGISTRYINDEX), Slot::Registry);
        assert_eq!(s.slot(REGISTRYINDEX - 1), Slot::None, "no closure");
        assert_eq!(s.absindex(-1), 2);
        assert_eq!(s.absindex(REGISTRYINDEX), REGISTRYINDEX);
        s.ci = std::ptr::null_mut();
    }

    #[test]
    fn vacated_slots_are_zero() {
        let s = fresh();
        for n in 1..=5 {
            s.push(value(n));
        }
        s.set_top(2);
        assert_eq!(s.top, 2);
        let raw = unsafe { std::slice::from_raw_parts(s.stack, s.cap) };
        assert!(raw[2..].iter().all(|v| v.is_null()));
        s.set_top(4);
        assert!(s.at(2).is_null() && s.at(3).is_null(), "new slots are nil");
    }

    #[test]
    fn the_stack_grows_and_keeps_its_values() {
        let s = fresh();
        for n in 1..=1000 {
            s.push(value(n));
        }
        assert!(s.cap >= 1000);
        assert_eq!(s.at(0), value(1));
        assert_eq!(s.at(999), value(1000));
        assert!(!s.reserve(MAXSTACK));
    }

    #[test]
    fn upvalue_indices_read_the_closure_record() {
        let s = fresh();
        let mut items: Vec<Any> = vec![value(10), value(11), value(12), value(13), value(14)];
        let mut header = crate::values::ListHeader {
            data: items.as_mut_ptr(),
            len: items.len() as i64,
            capacity: items.len() as i64,
        };
        let mut frame = CFrame {
            prev: std::ptr::null_mut(),
            base: 0,
            rec: &mut header,
            caller_line: 0,
        };
        s.ci = &mut frame;
        assert_eq!(s.slot(REGISTRYINDEX - 1), Slot::Upvalue(3));
        assert_eq!(s.value(REGISTRYINDEX - 2), value(14));
        assert_eq!(s.slot(REGISTRYINDEX - 3), Slot::None);
        s.put(Slot::Upvalue(3), value(99));
        assert_eq!(items[3], value(99));
        s.ci = std::ptr::null_mut();
    }
}
