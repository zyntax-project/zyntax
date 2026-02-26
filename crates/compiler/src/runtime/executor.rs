//! Executor for running async tasks
//!
//! The executor schedules and polls tasks until they complete.

use super::task::Task;
use super::waker::{Context, Waker};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::task::Poll;

/// Simple single-threaded executor
///
/// This executor runs tasks in a queue, polling them until they complete.
pub struct Executor {
    /// Tasks indexed by ID
    tasks: HashMap<usize, Arc<Task>>,
    /// Queue of task IDs that are ready to run
    ready_queue: Arc<Mutex<VecDeque<usize>>>,
    /// Next task ID to assign
    next_task_id: usize,
}

impl Executor {
    /// Create a new executor
    pub fn new() -> Self {
        Executor {
            tasks: HashMap::new(),
            ready_queue: Arc::new(Mutex::new(VecDeque::new())),
            next_task_id: 0,
        }
    }

    /// Spawn a new task
    ///
    /// Returns the task ID.
    pub fn spawn<F>(&mut self, future: F) -> usize
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let task_id = self.next_task_id;
        self.next_task_id += 1;

        let task = Task::new(future);
        self.tasks.insert(task_id, task);

        // Add to ready queue
        self.ready_queue.lock().unwrap().push_back(task_id);

        task_id
    }

    /// Run the executor until all tasks complete
    pub fn run(&mut self) {
        loop {
            // Get next ready task
            let task_id = {
                let mut queue = self.ready_queue.lock().unwrap();
                match queue.pop_front() {
                    Some(id) => id,
                    None => break, // No more tasks
                }
            };

            // Get the task
            let task = match self.tasks.get(&task_id) {
                Some(task) => task.clone(),
                None => continue, // Task was removed
            };

            // Create waker for this task
            let waker = Waker::new(task_id, self.ready_queue.clone());
            let std_waker = waker.into_std_waker();
            let mut context = Context::from_waker(&std_waker);

            // Poll the task
            match task.poll(&mut context) {
                Poll::Ready(()) => {
                    // Task completed, remove it
                    self.tasks.remove(&task_id);
                }
                Poll::Pending => {
                    // Task is not ready, it will be woken up later
                }
            }
        }
    }

    /// Get the number of active tasks
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

/// Block on a future until it completes
///
/// This is a convenience function for running a single future to completion.
pub fn block_on<F, T>(future: F) -> T
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;

    // Wrap the future to capture its result
    let result = Arc::new(StdMutex::new(None));
    let result_clone = result.clone();

    let wrapped_future = async move {
        let output = future.await;
        *result_clone.lock().unwrap() = Some(output);
    };

    // Run the executor
    let mut executor = Executor::new();
    executor.spawn(wrapped_future);
    executor.run();

    // Extract the result
    let output = result
        .lock()
        .unwrap()
        .take()
        .expect("Future did not complete");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let executor = Executor::new();
        assert_eq!(executor.task_count(), 0);
    }

    #[test]
    fn test_spawn_task() {
        let mut executor = Executor::new();
        executor.spawn(async {});
        assert_eq!(executor.task_count(), 1);
    }

    #[test]
    fn test_run_simple_task() {
        let mut executor = Executor::new();
        executor.spawn(async {
            // Simple task that completes immediately
        });

        executor.run();

        // Task should have completed and been removed
        assert_eq!(executor.task_count(), 0);
    }

    #[test]
    fn test_multiple_tasks() {
        let mut executor = Executor::new();

        executor.spawn(async {});
        executor.spawn(async {});
        executor.spawn(async {});

        assert_eq!(executor.task_count(), 3);

        executor.run();

        assert_eq!(executor.task_count(), 0);
    }

    #[test]
    fn test_block_on() {
        let result = block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_block_on_with_await() {
        async fn inner() -> i32 {
            100
        }

        let result = block_on(async {
            let value = inner().await;
            value + 50
        });

        assert_eq!(result, 150);
    }
}
