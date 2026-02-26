//! Waker implementation for async runtime
//!
//! Wakers notify the executor when a task is ready to make progress.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::task::{Context as StdContext, RawWaker, RawWakerVTable, Waker as StdWaker};

/// A waker that can wake up tasks in the executor
#[derive(Clone)]
pub struct Waker {
    /// Task ID that this waker is associated with
    task_id: usize,
    /// Reference to the executor's wake queue
    wake_queue: Arc<Mutex<VecDeque<usize>>>,
}

impl Waker {
    /// Create a new waker for a task
    pub fn new(task_id: usize, wake_queue: Arc<Mutex<VecDeque<usize>>>) -> Self {
        Waker {
            task_id,
            wake_queue,
        }
    }

    /// Wake the associated task
    pub fn wake(&self) {
        let mut queue = self.wake_queue.lock().unwrap();
        if !queue.contains(&self.task_id) {
            queue.push_back(self.task_id);
        }
    }

    /// Convert to a standard library Waker
    pub fn into_std_waker(self) -> StdWaker {
        let waker = Arc::new(self);
        let raw_waker = RawWaker::new(Arc::into_raw(waker) as *const (), &VTABLE);
        unsafe { StdWaker::from_raw(raw_waker) }
    }
}

// Vtable for the waker
const VTABLE: RawWakerVTable =
    RawWakerVTable::new(clone_waker, wake_waker, wake_by_ref_waker, drop_waker);

unsafe fn clone_waker(data: *const ()) -> RawWaker {
    let waker = Arc::from_raw(data as *const Waker);
    let cloned = waker.clone();
    std::mem::forget(waker); // Don't drop the original
    RawWaker::new(Arc::into_raw(Arc::new(cloned)) as *const (), &VTABLE)
}

unsafe fn wake_waker(data: *const ()) {
    let waker = Arc::from_raw(data as *const Waker);
    waker.wake();
}

unsafe fn wake_by_ref_waker(data: *const ()) {
    let waker = &*(data as *const Waker);
    waker.wake();
}

unsafe fn drop_waker(data: *const ()) {
    drop(Arc::from_raw(data as *const Waker));
}

/// Context for polling futures
///
/// This is a re-export of std::task::Context for convenience.
pub type Context<'a> = StdContext<'a>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waker_creation() {
        let wake_queue = Arc::new(Mutex::new(VecDeque::new()));
        let waker = Waker::new(0, wake_queue.clone());

        waker.wake();

        let queue = wake_queue.lock().unwrap();
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0], 0);
    }

    #[test]
    fn test_waker_deduplication() {
        let wake_queue = Arc::new(Mutex::new(VecDeque::new()));
        let waker = Waker::new(0, wake_queue.clone());

        // Wake multiple times
        waker.wake();
        waker.wake();
        waker.wake();

        // Should only be in queue once
        let queue = wake_queue.lock().unwrap();
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_std_waker_conversion() {
        let wake_queue = Arc::new(Mutex::new(VecDeque::new()));
        let waker = Waker::new(0, wake_queue.clone());

        let std_waker = waker.clone().into_std_waker();
        std_waker.wake();

        let queue = wake_queue.lock().unwrap();
        assert_eq!(queue.len(), 1);
    }
}
