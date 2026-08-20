//! The band pool: every index computed once, and cheaply enough to be
//! worth the threads.
//!
//! `parallel_dispatch.rs` decides which loops are handed here; this is
//! the protocol underneath. Two things have to hold. Every index in the
//! range is computed exactly once, however the bands fall out, and a
//! dispatch has to cost little enough that spreading a loop is worth
//! doing at all.
//!
//! The second is why the handover is measured rather than assumed. A
//! pool whose threads sleep between dispatches pays a pair of context
//! switches for every one, and that cost is invisible in a test that
//! only checks answers. Three designs were measured here: threads
//! parked on their own queue at 58.9 microseconds, one shared slot they
//! raced for at 122.9, and the staged wait below at 21.8. Where the
//! whole point is to make handing work over cheap, a number is the only
//! thing that says whether it was.

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;
use zyntax_compiler::zrtl::{zyntax_parallel_for, BandPool};

unsafe extern "C" fn tiny(lo: i64, hi: i64, env: *mut u8) {
    let acc = &*(env as *const AtomicI64);
    let mut s = 0i64;
    for i in lo..hi {
        s += i;
    }
    acc.fetch_add(s, Ordering::Relaxed);
}

/// What one dispatch costs beyond the work it carries.
///
/// Reported rather than pinned to a number, because the number belongs
/// to the machine. The assertion is only that it is in the range where
/// spreading a loop can pay: a dispatch costing a millisecond would
/// make the whole transform pointless, and would pass every other test
/// here.
#[test]
fn handover_cost() {
    let acc = AtomicI64::new(0);
    let p = &acc as *const _ as *mut u8;
    unsafe { zyntax_parallel_for(0, 10_000, 1, tiny, p) };
    const N: usize = 500;
    let t = Instant::now();
    for _ in 0..N {
        unsafe { zyntax_parallel_for(0, 10_000, 1, tiny, p) };
    }
    let split = t.elapsed().as_secs_f64() * 1e6 / N as f64;
    let t = Instant::now();
    for _ in 0..N {
        unsafe { zyntax_parallel_for(0, 10_000, 1_000_000, tiny, p) };
    }
    let serial = t.elapsed().as_secs_f64() * 1e6 / N as f64;
    let handover = split - serial;
    println!(
        "\n  workers {}  split {split:.1} us  serial {serial:.1} us  HANDOVER {handover:.1} us",
        BandPool::shared().workers()
    );
    assert!(
        handover < 500.0,
        "a dispatch costing {handover:.1} us is too much to be worth spreading a loop for"
    );
}

unsafe extern "C" fn nested(lo: i64, hi: i64, env: *mut u8) {
    zyntax_parallel_for(lo, hi, 1, tiny, env);
}

#[test]
fn a_band_that_dispatches_again_does_not_hang() {
    let acc = AtomicI64::new(0);
    let p = &acc as *const _ as *mut u8;
    unsafe { zyntax_parallel_for(0, 100_000, 1, nested, p) };
    let n = 100_000i64;
    assert_eq!(acc.load(Ordering::Relaxed), n * (n - 1) / 2);
}

/// Hammer the protocol: a thousand dispatches of varying shapes, back
/// to back.
///
/// The wake is a cell a thread clears itself, and clearing it after
/// reporting the band done rather than before would let the next
/// dispatch fill the slot and have the clear swallow it. That shows up
/// as a hang or a short count, and only under repetition, which is what
/// this is for.
#[test]
fn a_thousand_dispatches_all_land() {
    for round in 0..1000i64 {
        let n = 1 + (round * 7) % 4096;
        let marks: Vec<AtomicI64> = (0..n).map(|_| AtomicI64::new(0)).collect();
        unsafe {
            zyntax_parallel_for(0, n, 1, mark, &marks as *const _ as *mut u8);
        }
        let total: i64 = marks.iter().map(|m| m.load(Ordering::Relaxed)).sum();
        assert_eq!(total, n, "round {round} covered {total} of {n}");
    }
}

unsafe extern "C" fn mark(lo: i64, hi: i64, env: *mut u8) {
    let marks = &*(env as *const Vec<AtomicI64>);
    for i in lo..hi {
        marks[i as usize].fetch_add(1, Ordering::Relaxed);
    }
}
