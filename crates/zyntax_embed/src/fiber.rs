//! Safe, thread-affine host ownership for a compiled Zyntax `Fiber<T>`.
//!
//! The raw `FiberRepr` pointer never leaves this module. A fiber is resumed and
//! freed only on the thread that created the host handle, matching the native
//! backend contract.

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;
use std::thread::{self, ThreadId};

use zyntax_compiler::fiber_backend::{
    unpack_fiber_step, FIBER_STEP_DONE, FIBER_STEP_ERRORED, FIBER_STEP_YIELDED,
};
use zyntax_compiler::zrtl::{
    krio_fiber_cancel, krio_fiber_free, krio_fiber_new, krio_fiber_resume, krio_fiber_resume_with,
    FiberRepr,
};

use crate::runtime::{RuntimeError, RuntimeResult};

/// One result produced by resuming a live Zyntax fiber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZyntaxFiberStep {
    Yielded(i64),
    Done,
    Errored(i64),
}

/// Owning process-local handle for a compiled Zyntax fiber.
///
/// This type is intentionally `!Send` and `!Sync`. It is not a durable token:
/// embedders must reconstruct a new fiber from canonical application state
/// after process or thread loss.
pub struct ZyntaxFiber {
    raw: NonNull<FiberRepr>,
    creator: ThreadId,
    terminal: bool,
    _thread_affine: PhantomData<Rc<()>>,
}

impl ZyntaxFiber {
    pub(crate) fn from_entry(entry: *const u8) -> RuntimeResult<Self> {
        if entry.is_null() {
            return Err(RuntimeError::Execution(
                "compiled fiber entry point is null".into(),
            ));
        }
        // SAFETY: `entry` is a compiled no-argument `fiber def` body owned by
        // the embedding runtime. The backend creates it paused and transfers
        // unique ownership of the returned handle to this object.
        let raw = unsafe { krio_fiber_new(entry.cast_mut(), 0) };
        let raw = NonNull::new(raw).ok_or_else(|| {
            RuntimeError::Execution("compiled fiber constructor returned a null handle".into())
        })?;
        Ok(Self {
            raw,
            creator: thread::current().id(),
            terminal: false,
            _thread_affine: PhantomData,
        })
    }

    fn require_creator_thread(&self) -> RuntimeResult<()> {
        if thread::current().id() == self.creator {
            Ok(())
        } else {
            Err(RuntimeError::Execution(
                "Zyntax fiber may only be used on its creator thread".into(),
            ))
        }
    }

    fn decode_step(&mut self, packed: i64) -> ZyntaxFiberStep {
        let (tag, payload) = unpack_fiber_step(packed);
        match tag {
            FIBER_STEP_YIELDED => ZyntaxFiberStep::Yielded(payload),
            FIBER_STEP_DONE => {
                self.terminal = true;
                ZyntaxFiberStep::Done
            }
            FIBER_STEP_ERRORED => {
                self.terminal = true;
                ZyntaxFiberStep::Errored(payload)
            }
            _ => {
                self.terminal = true;
                ZyntaxFiberStep::Errored(tag)
            }
        }
    }

    /// Resumes until the next yield, completion, or fiber error.
    pub fn resume(&mut self) -> RuntimeResult<ZyntaxFiberStep> {
        self.require_creator_thread()?;
        if self.terminal {
            return Err(RuntimeError::Execution(
                "cannot resume a terminal Zyntax fiber".into(),
            ));
        }
        // SAFETY: `raw` is owned by this handle, remains on its creator
        // thread, and is freed exactly once in `Drop`.
        let packed = unsafe { krio_fiber_resume(self.raw.as_ptr()) };
        Ok(self.decode_step(packed))
    }

    /// Resumes while delivering a scalar value to the suspended yield point.
    pub fn resume_with(&mut self, value: i64) -> RuntimeResult<ZyntaxFiberStep> {
        self.require_creator_thread()?;
        if self.terminal {
            return Err(RuntimeError::Execution(
                "cannot resume a terminal Zyntax fiber".into(),
            ));
        }
        // SAFETY: same ownership and thread-affinity argument as `resume`.
        let packed = unsafe { krio_fiber_resume_with(self.raw.as_ptr(), value) };
        Ok(self.decode_step(packed))
    }

    /// Requests cooperative cancellation. The handle remains owned and is
    /// released by `Drop` whether or not it is resumed again.
    pub fn cancel(&mut self) -> RuntimeResult<()> {
        self.require_creator_thread()?;
        if !self.terminal {
            // SAFETY: `raw` is live, uniquely owned, and on its creator thread.
            unsafe { krio_fiber_cancel(self.raw.as_ptr()) };
        }
        Ok(())
    }

    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.terminal
    }
}

impl Drop for ZyntaxFiber {
    fn drop(&mut self) {
        debug_assert_eq!(thread::current().id(), self.creator);
        // SAFETY: this is the unique owning handle and `Drop` runs once.
        unsafe { krio_fiber_free(self.raw.as_ptr()) };
    }
}
