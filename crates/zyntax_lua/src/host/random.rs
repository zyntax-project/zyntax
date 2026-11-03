//! Random support.

use super::*;

pub(super) extern "C" fn host_random_seed(n1: i64, n2: i64) {
    RANDOM.with(|r| *r.borrow_mut() = Xoshiro::seeded_with(n1 as u64, n2 as u64));
}

/// Seeded from the clock and the process, as `math.randomseed()`
/// with nothing given; the first number, the second being the
/// process id.
pub(super) extern "C" fn host_random_seed_now() -> i64 {
    let n1 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(7);
    let n2 = u64::from(std::process::id());
    RANDOM.with(|r| *r.borrow_mut() = Xoshiro::seeded_with(n1, n2));
    n1 as i64
}

/// The second number `math.randomseed()` seeds with.
pub(super) extern "C" fn host_random_seed_pid() -> i64 {
    i64::from(std::process::id())
}

/// The next draw, all 64 bits.
pub(super) extern "C" fn host_random_next() -> i64 {
    RANDOM.with(|r| r.borrow_mut().next()) as i64
}

/// A float in `[0, 1)`, from the high 53 bits.
pub(super) extern "C" fn host_random_float() -> f64 {
    let bits = RANDOM.with(|r| r.borrow_mut().next());
    (bits >> 11) as f64 * (1.0 / 9007199254740992.0)
}

/// An integer in `[lo, hi]`, projected as the reference projects a
/// draw: masked to the smallest all-ones bound and drawn again while
/// out of range.
pub(super) extern "C" fn host_random_int(lo: i64, hi: i64) -> i64 {
    if lo > hi {
        return lo;
    }
    let n = (hi as u64).wrapping_sub(lo as u64);
    let mut ran = RANDOM.with(|r| r.borrow_mut().next());
    let pick = if n & n.wrapping_add(1) == 0 {
        ran & n
    } else {
        let mut lim = n;
        lim |= lim >> 1;
        lim |= lim >> 2;
        lim |= lim >> 4;
        lim |= lim >> 8;
        lim |= lim >> 16;
        lim |= lim >> 32;
        loop {
            ran &= lim;
            if ran <= n {
                break ran;
            }
            ran = RANDOM.with(|r| r.borrow_mut().next());
        }
    };
    (lo as u64).wrapping_add(pick) as i64
}
