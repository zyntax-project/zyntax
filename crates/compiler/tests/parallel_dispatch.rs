//! Band dispatch: every part of the range computed exactly once, and
//! faster than one thread where there is enough work to matter.
//!
//! The compiler decides which loops may be spread (`parallel_safe`);
//! this is the runtime half. What must hold is that the range is
//! covered without overlap or gaps, that the caller sees the work
//! finished when the call returns, and that a range too small to be
//! worth splitting is not split.

use std::sync::atomic::{AtomicI64, Ordering};
use zyntax_compiler::zrtl::zyntax_parallel_for;

/// Split whenever there is more than a couple of iterations, so these
/// exercise the threaded path rather than the fall-back. What the grain
/// should be for a real loop is the compiler's decision, made from what
/// one iteration costs.
const GRAIN: i64 = 2;

/// Mark every index in the band, so gaps and overlaps both show up.
unsafe extern "C" fn mark(lo: i64, hi: i64, env: *mut u8) {
    let marks = &*(env as *const Vec<AtomicI64>);
    for i in lo..hi {
        marks[i as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Every index is computed once. Not zero, which would be a gap, and
/// not twice, which for a real kernel would be a wrong answer.
#[test]
fn every_index_is_computed_exactly_once() {
    for n in [1i64, 7, 1023, 100_000] {
        let marks: Vec<AtomicI64> = (0..n).map(|_| AtomicI64::new(0)).collect();
        unsafe {
            zyntax_parallel_for(0, n, GRAIN, mark, &marks as *const _ as *mut u8);
        }
        let wrong: Vec<(usize, i64)> = marks
            .iter()
            .enumerate()
            .map(|(i, m)| (i, m.load(Ordering::Relaxed)))
            .filter(|(_, c)| *c != 1)
            .take(4)
            .collect();
        assert!(
            wrong.is_empty(),
            "n={n}: indices computed the wrong number of times: {wrong:?}"
        );
    }
}

/// The work is finished when the call returns, not merely started.
#[test]
fn the_call_returns_only_once_the_work_is_done() {
    const N: i64 = 200_000;
    let marks: Vec<AtomicI64> = (0..N).map(|_| AtomicI64::new(0)).collect();
    unsafe {
        zyntax_parallel_for(0, N, GRAIN, mark, &marks as *const _ as *mut u8);
    }
    // Read immediately, with no synchronisation of our own.
    let total: i64 = marks.iter().map(|m| m.load(Ordering::Relaxed)).sum();
    assert_eq!(total, N, "the caller saw work that had not finished");
}

/// A range that starts partway through is still covered exactly.
#[test]
fn a_range_that_does_not_start_at_zero_is_covered() {
    const N: usize = 50_000;
    let marks: Vec<AtomicI64> = (0..N).map(|_| AtomicI64::new(0)).collect();
    unsafe {
        zyntax_parallel_for(10_000, 40_000, GRAIN, mark, &marks as *const _ as *mut u8);
    }
    assert_eq!(marks[9_999].load(Ordering::Relaxed), 0, "before the range");
    assert_eq!(marks[10_000].load(Ordering::Relaxed), 1, "first in range");
    assert_eq!(marks[39_999].load(Ordering::Relaxed), 1, "last in range");
    assert_eq!(marks[40_000].load(Ordering::Relaxed), 0, "past the range");
}

/// An empty or backwards range does nothing rather than misbehaving.
#[test]
fn an_empty_range_does_nothing() {
    let marks: Vec<AtomicI64> = (0..8).map(|_| AtomicI64::new(0)).collect();
    unsafe {
        zyntax_parallel_for(5, 5, GRAIN, mark, &marks as *const _ as *mut u8);
        zyntax_parallel_for(6, 2, GRAIN, mark, &marks as *const _ as *mut u8);
    }
    assert!(marks.iter().all(|m| m.load(Ordering::Relaxed) == 0));
}
