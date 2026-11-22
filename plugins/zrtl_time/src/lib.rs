//! ZRTL Time Plugin
//!
//! Provides time and duration operations for Zyntax-based languages.
//!
//! ## Exported Symbols
//!
//! ### Current Time
//! - `$Time$now_secs` - Unix timestamp in seconds
//! - `$Time$now_millis` - Unix timestamp in milliseconds
//! - `$Time$now_micros` - Unix timestamp in microseconds
//! - `$Time$now_nanos` - Monotonic time in nanoseconds (for benchmarking)
//!
//! ### Sleep
//! - `$Time$sleep_secs` - Sleep for N seconds
//! - `$Time$sleep_millis` - Sleep for N milliseconds
//! - `$Time$sleep_micros` - Sleep for N microseconds
//!
//! ### Instant (for timing)
//! - `$Time$instant_now` - Get current instant (opaque handle)
//! - `$Time$instant_elapsed_nanos` - Nanoseconds since instant
//! - `$Time$instant_elapsed_millis` - Milliseconds since instant
//!
//! ### Formatting
//! - `$Time$format_timestamp` - Format unix timestamp as ISO 8601

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use zrtl::{zrtl_plugin, StringPtr, string_new};

// ============================================================================
// Instant Handle Management
// ============================================================================

static INSTANT_COUNTER: AtomicU64 = AtomicU64::new(1);
static INSTANTS: Mutex<Option<HashMap<u64, Instant>>> = Mutex::new(None);

fn get_instant_map() -> std::sync::MutexGuard<'static, Option<HashMap<u64, Instant>>> {
    let mut guard = INSTANTS.lock().unwrap();
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    guard
}

// ============================================================================
// Current Time Functions
// ============================================================================

/// Get current Unix timestamp in seconds
#[no_mangle]
pub extern "C" fn time_now_secs() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(_) => 0,
    }
}

/// Get current Unix timestamp in milliseconds
#[no_mangle]
pub extern "C" fn time_now_millis() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_millis() as i64,
        Err(_) => 0,
    }
}

/// Get current Unix timestamp in microseconds
#[no_mangle]
pub extern "C" fn time_now_micros() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_micros() as i64,
        Err(_) => 0,
    }
}

/// Get monotonic time in nanoseconds (for benchmarking)
///
/// This uses `Instant` internally and is only useful for measuring
/// elapsed time, not for getting wall clock time.
#[no_mangle]
pub extern "C" fn time_monotonic_nanos() -> u64 {
    // Use a static start instant for relative timing
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

// ============================================================================
// Sleep Functions
// ============================================================================

/// Sleep for N seconds
#[no_mangle]
pub extern "C" fn time_sleep_secs(secs: u64) {
    zrtl::heap::blocking(|| std::thread::sleep(Duration::from_secs(secs)));
}

/// Sleep for N milliseconds
#[no_mangle]
pub extern "C" fn time_sleep_millis(millis: u64) {
    zrtl::heap::blocking(|| std::thread::sleep(Duration::from_millis(millis)));
}

/// Sleep for N microseconds
#[no_mangle]
pub extern "C" fn time_sleep_micros(micros: u64) {
    zrtl::heap::blocking(|| std::thread::sleep(Duration::from_micros(micros)));
}

/// Sleep for N nanoseconds
#[no_mangle]
pub extern "C" fn time_sleep_nanos(nanos: u64) {
    zrtl::heap::blocking(|| std::thread::sleep(Duration::from_nanos(nanos)));
}

// ============================================================================
// Instant Functions (for precise timing)
// ============================================================================

/// Create a new instant handle for timing
///
/// Returns a handle (u64) that can be used with instant_elapsed_*.
/// Handle 0 is reserved for errors.
#[no_mangle]
pub extern "C" fn time_instant_now() -> u64 {
    let handle = INSTANT_COUNTER.fetch_add(1, Ordering::SeqCst);
    let instant = Instant::now();

    let mut map = get_instant_map();
    if let Some(ref mut m) = *map {
        m.insert(handle, instant);
    }

    handle
}

/// Get elapsed nanoseconds since the instant was created
///
/// Returns 0 if the handle is invalid.
#[no_mangle]
pub extern "C" fn time_instant_elapsed_nanos(handle: u64) -> u64 {
    let map = get_instant_map();
    if let Some(ref m) = *map {
        if let Some(instant) = m.get(&handle) {
            return instant.elapsed().as_nanos() as u64;
        }
    }
    0
}

/// Get elapsed microseconds since the instant was created
#[no_mangle]
pub extern "C" fn time_instant_elapsed_micros(handle: u64) -> u64 {
    let map = get_instant_map();
    if let Some(ref m) = *map {
        if let Some(instant) = m.get(&handle) {
            return instant.elapsed().as_micros() as u64;
        }
    }
    0
}

/// Get elapsed milliseconds since the instant was created
#[no_mangle]
pub extern "C" fn time_instant_elapsed_millis(handle: u64) -> u64 {
    let map = get_instant_map();
    if let Some(ref m) = *map {
        if let Some(instant) = m.get(&handle) {
            return instant.elapsed().as_millis() as u64;
        }
    }
    0
}

/// Get elapsed seconds (as f64) since the instant was created
#[no_mangle]
pub extern "C" fn time_instant_elapsed_secs_f64(handle: u64) -> f64 {
    let map = get_instant_map();
    if let Some(ref m) = *map {
        if let Some(instant) = m.get(&handle) {
            return instant.elapsed().as_secs_f64();
        }
    }
    0.0
}

/// Free an instant handle
#[no_mangle]
pub extern "C" fn time_instant_free(handle: u64) {
    let mut map = get_instant_map();
    if let Some(ref mut m) = *map {
        m.remove(&handle);
    }
}

// ============================================================================
// Formatting Functions
// ============================================================================

/// Format a Unix timestamp (seconds) as ISO 8601 string
///
/// Returns a ZRTL string in format "YYYY-MM-DDTHH:MM:SSZ"
/// Caller must free with `string_free`.
#[no_mangle]
pub extern "C" fn time_format_iso8601(timestamp_secs: i64) -> StringPtr {
    // Simple UTC formatting without external deps
    let secs = timestamp_secs as u64;

    // Days since Unix epoch
    let days = secs / 86400;
    let day_secs = secs % 86400;

    let hours = day_secs / 3600;
    let mins = (day_secs % 3600) / 60;
    let secs = day_secs % 60;

    // Calculate year/month/day from days since epoch (1970-01-01)
    let (year, month, day) = days_to_ymd(days);

    let formatted = format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, mins, secs
    );

    string_new(&formatted)
}

/// Format elapsed time in human readable form
///
/// Examples: "1.234s", "45.6ms", "789µs", "123ns"
#[no_mangle]
pub extern "C" fn time_format_duration_nanos(nanos: u64) -> StringPtr {
    let formatted = if nanos >= 1_000_000_000 {
        format!("{:.3}s", nanos as f64 / 1_000_000_000.0)
    } else if nanos >= 1_000_000 {
        format!("{:.3}ms", nanos as f64 / 1_000_000.0)
    } else if nanos >= 1_000 {
        format!("{:.3}µs", nanos as f64 / 1_000.0)
    } else {
        format!("{}ns", nanos)
    };

    string_new(&formatted)
}

// Helper: Convert days since epoch to (year, month, day)
fn days_to_ymd(days: u64) -> (i32, u32, u32) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    (year as i32, m, d)
}

// ============================================================================
// Plugin Export
// ============================================================================

zrtl_plugin! {
    name: "zrtl_time",
    symbols: [
        // Current time
        ("$Time$now_secs", time_now_secs),
        ("$Time$now_millis", time_now_millis),
        ("$Time$now_micros", time_now_micros),
        ("$Time$monotonic_nanos", time_monotonic_nanos),

        // Sleep
        ("$Time$sleep_secs", time_sleep_secs),
        ("$Time$sleep_millis", time_sleep_millis),
        ("$Time$sleep_micros", time_sleep_micros),
        ("$Time$sleep_nanos", time_sleep_nanos),

        // Instant (timing)
        ("$Time$instant_now", time_instant_now),
        ("$Time$instant_elapsed_nanos", time_instant_elapsed_nanos),
        ("$Time$instant_elapsed_micros", time_instant_elapsed_micros),
        ("$Time$instant_elapsed_millis", time_instant_elapsed_millis),
        ("$Time$instant_elapsed_secs_f64", time_instant_elapsed_secs_f64),
        ("$Time$instant_free", time_instant_free),

        // Formatting
        ("$Time$format_iso8601", time_format_iso8601),
        ("$Time$format_duration_nanos", time_format_duration_nanos),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use zrtl::{string_as_str, string_free};

    #[test]
    fn test_now_secs() {
        let ts = time_now_secs();
        // Should be after 2020-01-01
        assert!(ts > 1577836800);
    }

    #[test]
    fn test_instant() {
        let handle = time_instant_now();
        assert!(handle > 0);

        // Small sleep
        std::thread::sleep(Duration::from_millis(10));

        let elapsed = time_instant_elapsed_millis(handle);
        assert!(elapsed >= 10);

        time_instant_free(handle);
    }

    #[test]
    fn test_format_iso8601() {
        // 2020-01-01 00:00:00 UTC
        let result = time_format_iso8601(1577836800);
        unsafe {
            assert_eq!(string_as_str(result), Some("2020-01-01T00:00:00Z"));
            string_free(result);
        }
    }

    #[test]
    fn test_format_duration() {
        let r1 = time_format_duration_nanos(500);
        let r2 = time_format_duration_nanos(5_000);
        let r3 = time_format_duration_nanos(5_000_000);
        let r4 = time_format_duration_nanos(5_000_000_000);

        unsafe {
            assert_eq!(string_as_str(r1), Some("500ns"));
            assert!(string_as_str(r2).unwrap().contains("µs"));
            assert!(string_as_str(r3).unwrap().contains("ms"));
            assert!(string_as_str(r4).unwrap().contains("s"));

            string_free(r1);
            string_free(r2);
            string_free(r3);
            string_free(r4);
        }
    }
}
