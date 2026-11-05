use super::{CompileQueue, Job};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn saturated_workers_queue_calls_and_prioritize_active_frames() {
    let queue = CompileQueue::new(Arc::new(AtomicBool::new(false)), 1);
    queue.enter_job();
    assert_eq!(queue.idle_workers.load(Ordering::Acquire), 0);

    assert!(queue.request_hot_call(1, 20));
    assert!(queue.request_hot_call(1, 21));
    assert!(queue.begin_promotion(2, 100));
    queue.request_promote(2, 100);
    assert!(!queue.begin_promotion(2, 101));
    assert!(matches!(queue.take(), Some(Job::Promote(2, 100, false))));
    assert_eq!(queue.next_promotion(2), Some(101));
    assert_eq!(queue.next_promotion(2), None);
    assert!(matches!(queue.take(), Some(Job::Compile(1, Some(20), _))));
    assert!(queue.take().is_none(), "repeated calls share one request");
    queue.leave_job();
}

#[test]
fn failed_workers_allow_fallback_but_shutdown_does_not_compile() {
    let stop = Arc::new(AtomicBool::new(false));
    let queue = CompileQueue::new(Arc::clone(&stop), 1);
    queue.retire_worker();
    assert!(!queue.request_hot_call(1, 20));
    assert!(queue.take().is_none());

    stop.store(true, Ordering::Release);
    assert!(queue.request_hot_call(1, 20));
    assert!(queue.take().is_none(), "shutdown must not add work");
}

#[test]
fn a_call_that_goes_cold_while_workers_are_busy_can_be_requested_again() {
    let queue = CompileQueue::new(Arc::new(AtomicBool::new(false)), 1);
    queue.enter_job();
    queue.count_of(1).store(20, Ordering::Relaxed);
    assert!(queue.request_hot_call(1, 20));
    let Some(Job::Compile(id, count, at)) = queue.take() else {
        panic!("queued compile");
    };
    let old = at - std::time::Duration::from_secs(1);
    assert!(!queue.still_wanted(id, count, old));
    queue.count_of(1).store(21, Ordering::Relaxed);
    assert!(queue.request_hot_call(1, 21));
    let Some(Job::Compile(id, count, at)) = queue.take() else {
        panic!("fresh demand queues again");
    };
    queue.count_of(1).store(22, Ordering::Relaxed);
    assert!(queue.still_wanted(id, count, at - std::time::Duration::from_secs(1)));
    queue.leave_job();
}
