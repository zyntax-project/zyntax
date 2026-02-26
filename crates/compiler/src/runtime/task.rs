//! Task representation for async execution
//!
//! A Task wraps a future (state machine) and tracks its execution state.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

/// A spawned async task
///
/// Tasks wrap futures and provide the interface for the executor to poll them.
pub struct Task {
    /// The future being executed (boxed for type erasure)
    future: Mutex<Pin<Box<dyn Future<Output = ()> + Send + 'static>>>,
}

impl Task {
    /// Create a new task from a future
    pub fn new<F>(future: F) -> Arc<Task>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        Arc::new(Task {
            future: Mutex::new(Box::pin(future)),
        })
    }

    /// Poll the task's future
    ///
    /// Returns Poll::Ready(()) when the task completes, or Poll::Pending if it needs more work.
    pub fn poll(&self, context: &mut Context) -> Poll<()> {
        let mut future = self.future.lock().unwrap();
        future.as_mut().poll(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::{RawWaker, RawWakerVTable, Waker};

    fn dummy_waker() -> Waker {
        fn clone(_: *const ()) -> RawWaker {
            dummy_raw_waker()
        }
        fn wake(_: *const ()) {}
        fn wake_by_ref(_: *const ()) {}
        fn drop(_: *const ()) {}

        fn dummy_raw_waker() -> RawWaker {
            RawWaker::new(std::ptr::null(), &VTABLE)
        }

        const VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);

        unsafe { Waker::from_raw(dummy_raw_waker()) }
    }

    #[test]
    fn test_task_creation() {
        let task = Task::new(async {
            // Simple async block
        });

        let waker = dummy_waker();
        let mut context = Context::from_waker(&waker);

        // Task should complete immediately
        let result = task.poll(&mut context);
        assert!(matches!(result, Poll::Ready(())));
    }

    #[test]
    fn test_task_completes() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = completed.clone();

        let task = Task::new(async move {
            completed_clone.store(true, Ordering::SeqCst);
        });

        let waker = dummy_waker();
        let mut context = Context::from_waker(&waker);

        task.poll(&mut context);

        assert!(completed.load(Ordering::SeqCst));
    }
}
