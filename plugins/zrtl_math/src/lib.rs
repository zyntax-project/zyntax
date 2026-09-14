//! ZRTL Math Plugin
//!
//! Provides mathematical functions for Zyntax-based languages.
//!
//! ## Exported Symbols
//!
//! ### Trigonometry
//! - `$Math$sin`, `$Math$cos`, `$Math$tan` - Basic trig functions
//! - `$Math$asin`, `$Math$acos`, `$Math$atan`, `$Math$atan2` - Inverse trig
//! - `$Math$sinh`, `$Math$cosh`, `$Math$tanh` - Hyperbolic functions
//!
//! ### Exponentials & Logarithms
//! - `$Math$exp`, `$Math$exp2` - Exponential functions
//! - `$Math$log`, `$Math$log2`, `$Math$log10` - Logarithms
//! - `$Math$pow`, `$Math$sqrt`, `$Math$cbrt` - Powers and roots
//!
//! ### Rounding
//! - `$Math$floor`, `$Math$ceil`, `$Math$round`, `$Math$trunc` - Rounding
//! - `$Math$abs`, `$Math$abs_i64` - Absolute value
//!
//! ### Min/Max/Clamp
//! - `$Math$min`, `$Math$max`, `$Math$clamp` - Float min/max/clamp
//! - `$Math$min_i64`, `$Math$max_i64`, `$Math$clamp_i64` - Integer versions
//!
//! ### Random
//! - `$Math$random` - Random float [0, 1)
//! - `$Math$random_range` - Random float in range
//! - `$Math$random_int` - Random integer in range
//!
//! ### Constants
//! - `$Math$pi`, `$Math$e`, `$Math$tau` - Mathematical constants

use std::sync::atomic::{AtomicU64, Ordering};
use zrtl::zrtl_plugin;

// ============================================================================
// Random Number Generator (xorshift64* with proper seeding)
// ============================================================================

static RNG_STATE: AtomicU64 = AtomicU64::new(0);
static RNG_INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Initialize RNG state from OS entropy
fn init_rng() {
    if RNG_INITIALIZED.load(Ordering::Acquire) {
        return;
    }

    let mut seed_bytes = [0u8; 8];
    if getrandom::getrandom(&mut seed_bytes).is_ok() {
        let seed = u64::from_ne_bytes(seed_bytes);
        // Ensure non-zero state
        RNG_STATE.store(
            if seed != 0 { seed } else { 0x853c49e6748fea9b },
            Ordering::Release,
        );
    } else {
        RNG_STATE.store(0x853c49e6748fea9b, Ordering::Release);
    }
    RNG_INITIALIZED.store(true, Ordering::Release);
}

fn next_u64() -> u64 {
    init_rng();

    let mut state = RNG_STATE.load(Ordering::Relaxed);
    if state == 0 {
        state = 0x853c49e6748fea9b;
    }
    // xorshift64*
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    RNG_STATE.store(state, Ordering::Relaxed);
    state.wrapping_mul(0x2545F4914F6CDD1D)
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

/// Sine of angle in radians
#[no_mangle]
pub extern "C" fn math_sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine of angle in radians
#[no_mangle]
pub extern "C" fn math_cos(x: f64) -> f64 {
    x.cos()
}

/// Tangent of angle in radians
#[no_mangle]
pub extern "C" fn math_tan(x: f64) -> f64 {
    x.tan()
}

/// Arcsine, returns radians
#[no_mangle]
pub extern "C" fn math_asin(x: f64) -> f64 {
    x.asin()
}

/// Arccosine, returns radians
#[no_mangle]
pub extern "C" fn math_acos(x: f64) -> f64 {
    x.acos()
}

/// Arctangent, returns radians
#[no_mangle]
pub extern "C" fn math_atan(x: f64) -> f64 {
    x.atan()
}

/// Two-argument arctangent (atan2)
#[no_mangle]
pub extern "C" fn math_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Hyperbolic sine
#[no_mangle]
pub extern "C" fn math_sinh(x: f64) -> f64 {
    x.sinh()
}

/// Hyperbolic cosine
#[no_mangle]
pub extern "C" fn math_cosh(x: f64) -> f64 {
    x.cosh()
}

/// Hyperbolic tangent
#[no_mangle]
pub extern "C" fn math_tanh(x: f64) -> f64 {
    x.tanh()
}

// ============================================================================
// Exponentials & Logarithms
// ============================================================================

/// e^x
#[no_mangle]
pub extern "C" fn math_exp(x: f64) -> f64 {
    x.exp()
}

/// 2^x
#[no_mangle]
pub extern "C" fn math_exp2(x: f64) -> f64 {
    x.exp2()
}

/// Natural logarithm (base e)
#[no_mangle]
pub extern "C" fn math_log(x: f64) -> f64 {
    x.ln()
}

/// Base-2 logarithm
#[no_mangle]
pub extern "C" fn math_log2(x: f64) -> f64 {
    x.log2()
}

/// Base-10 logarithm
#[no_mangle]
pub extern "C" fn math_log10(x: f64) -> f64 {
    x.log10()
}

/// x^y (power)
#[no_mangle]
pub extern "C" fn math_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// Integer power (more efficient for integer exponents)
#[no_mangle]
pub extern "C" fn math_powi(x: f64, n: i32) -> f64 {
    x.powi(n)
}

/// Square root
#[no_mangle]
pub extern "C" fn math_sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Cube root
#[no_mangle]
pub extern "C" fn math_cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// Hypotenuse (sqrt(x^2 + y^2)) without overflow
#[no_mangle]
pub extern "C" fn math_hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

// ============================================================================
// Rounding Functions
// ============================================================================

/// Floor (round down)
#[no_mangle]
pub extern "C" fn math_floor(x: f64) -> f64 {
    x.floor()
}

/// Ceiling (round up)
#[no_mangle]
pub extern "C" fn math_ceil(x: f64) -> f64 {
    x.ceil()
}

/// Round to nearest integer (ties round away from zero)
#[no_mangle]
pub extern "C" fn math_round(x: f64) -> f64 {
    x.round()
}

/// Truncate toward zero
#[no_mangle]
pub extern "C" fn math_trunc(x: f64) -> f64 {
    x.trunc()
}

/// Fractional part
#[no_mangle]
pub extern "C" fn math_fract(x: f64) -> f64 {
    x.fract()
}

/// Absolute value (float)
#[no_mangle]
pub extern "C" fn math_abs(x: f64) -> f64 {
    x.abs()
}

/// Absolute value (integer)
#[no_mangle]
pub extern "C" fn math_abs_i64(x: i64) -> i64 {
    x.abs()
}

/// Sign of value: -1, 0, or 1
#[no_mangle]
pub extern "C" fn math_signum(x: f64) -> f64 {
    x.signum()
}

/// Copy sign from y to x
#[no_mangle]
pub extern "C" fn math_copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

// ============================================================================
// Min/Max/Clamp
// ============================================================================

/// Minimum of two floats
#[no_mangle]
pub extern "C" fn math_min(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// Maximum of two floats
#[no_mangle]
pub extern "C" fn math_max(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Clamp float to range [min, max]
#[no_mangle]
pub extern "C" fn math_clamp(x: f64, min: f64, max: f64) -> f64 {
    x.clamp(min, max)
}

/// Minimum of two integers
#[no_mangle]
pub extern "C" fn math_min_i64(a: i64, b: i64) -> i64 {
    a.min(b)
}

/// Maximum of two integers
#[no_mangle]
pub extern "C" fn math_max_i64(a: i64, b: i64) -> i64 {
    a.max(b)
}

/// Clamp integer to range [min, max]
#[no_mangle]
pub extern "C" fn math_clamp_i64(x: i64, min: i64, max: i64) -> i64 {
    x.clamp(min, max)
}

// ============================================================================
// Random Number Generation
// ============================================================================

/// Seed the random number generator
#[no_mangle]
pub extern "C" fn math_seed(seed: u64) {
    RNG_STATE.store(if seed == 0 { 1 } else { seed }, Ordering::Relaxed);
}

/// Random float in [0, 1)
#[no_mangle]
pub extern "C" fn math_random() -> f64 {
    let bits = next_u64();
    // Convert to f64 in [0, 1)
    (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Random float in [min, max)
#[no_mangle]
pub extern "C" fn math_random_range(min: f64, max: f64) -> f64 {
    min + math_random() * (max - min)
}

/// Random integer in [min, max] (inclusive)
#[no_mangle]
pub extern "C" fn math_random_int(min: i64, max: i64) -> i64 {
    if min >= max {
        return min;
    }
    let range = (max - min + 1) as u64;
    min + (next_u64() % range) as i64
}

/// Random boolean
#[no_mangle]
pub extern "C" fn math_random_bool() -> i32 {
    (next_u64() & 1) as i32
}

// ============================================================================
// Constants
// ============================================================================

/// Pi (π)
#[no_mangle]
pub extern "C" fn math_pi() -> f64 {
    std::f64::consts::PI
}

/// Euler's number (e)
#[no_mangle]
pub extern "C" fn math_e() -> f64 {
    std::f64::consts::E
}

/// Tau (2π)
#[no_mangle]
pub extern "C" fn math_tau() -> f64 {
    std::f64::consts::TAU
}

/// Square root of 2
#[no_mangle]
pub extern "C" fn math_sqrt2() -> f64 {
    std::f64::consts::SQRT_2
}

/// Natural log of 2
#[no_mangle]
pub extern "C" fn math_ln2() -> f64 {
    std::f64::consts::LN_2
}

/// Natural log of 10
#[no_mangle]
pub extern "C" fn math_ln10() -> f64 {
    std::f64::consts::LN_10
}

// ============================================================================
// Special Functions
// ============================================================================

/// Check if value is NaN
#[no_mangle]
pub extern "C" fn math_is_nan(x: f64) -> i32 {
    x.is_nan() as i32
}

/// Check if value is infinite
#[no_mangle]
pub extern "C" fn math_is_infinite(x: f64) -> i32 {
    x.is_infinite() as i32
}

/// Check if value is finite (not NaN or infinite)
#[no_mangle]
pub extern "C" fn math_is_finite(x: f64) -> i32 {
    x.is_finite() as i32
}

/// Positive infinity
#[no_mangle]
pub extern "C" fn math_infinity() -> f64 {
    f64::INFINITY
}

/// Negative infinity
#[no_mangle]
pub extern "C" fn math_neg_infinity() -> f64 {
    f64::NEG_INFINITY
}

/// NaN (Not a Number)
#[no_mangle]
pub extern "C" fn math_nan() -> f64 {
    f64::NAN
}

/// Floating-point remainder (x % y)
#[no_mangle]
pub extern "C" fn math_fmod(x: f64, y: f64) -> f64 {
    x % y
}

/// Linear interpolation: a + t*(b-a)
#[no_mangle]
pub extern "C" fn math_lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Convert degrees to radians
#[no_mangle]
pub extern "C" fn math_to_radians(degrees: f64) -> f64 {
    degrees.to_radians()
}

/// Convert radians to degrees
#[no_mangle]
pub extern "C" fn math_to_degrees(radians: f64) -> f64 {
    radians.to_degrees()
}

// ============================================================================
// Plugin Export
// ============================================================================

zrtl_plugin! {
    name: "zrtl_math",
    symbols: [
        // Trigonometry
        ("$Math$sin", math_sin),
        ("$Math$cos", math_cos),
        ("$Math$tan", math_tan),
        ("$Math$asin", math_asin),
        ("$Math$acos", math_acos),
        ("$Math$atan", math_atan),
        ("$Math$atan2", math_atan2),
        ("$Math$sinh", math_sinh),
        ("$Math$cosh", math_cosh),
        ("$Math$tanh", math_tanh),

        // Exponentials & Logarithms
        ("$Math$exp", math_exp),
        ("$Math$exp2", math_exp2),
        ("$Math$log", math_log),
        ("$Math$log2", math_log2),
        ("$Math$log10", math_log10),
        ("$Math$pow", math_pow),
        ("$Math$powi", math_powi),
        ("$Math$sqrt", math_sqrt),
        ("$Math$cbrt", math_cbrt),
        ("$Math$hypot", math_hypot),

        // Rounding
        ("$Math$floor", math_floor),
        ("$Math$ceil", math_ceil),
        ("$Math$round", math_round),
        ("$Math$trunc", math_trunc),
        ("$Math$fract", math_fract),
        ("$Math$abs", math_abs),
        ("$Math$abs_i64", math_abs_i64),
        ("$Math$signum", math_signum),
        ("$Math$copysign", math_copysign),

        // Min/Max/Clamp
        ("$Math$min", math_min),
        ("$Math$max", math_max),
        ("$Math$clamp", math_clamp),
        ("$Math$min_i64", math_min_i64),
        ("$Math$max_i64", math_max_i64),
        ("$Math$clamp_i64", math_clamp_i64),

        // Random
        ("$Math$seed", math_seed),
        ("$Math$random", math_random),
        ("$Math$random_range", math_random_range),
        ("$Math$random_int", math_random_int),
        ("$Math$random_bool", math_random_bool),

        // Constants
        ("$Math$pi", math_pi),
        ("$Math$e", math_e),
        ("$Math$tau", math_tau),
        ("$Math$sqrt2", math_sqrt2),
        ("$Math$ln2", math_ln2),
        ("$Math$ln10", math_ln10),

        // Special
        ("$Math$is_nan", math_is_nan),
        ("$Math$is_infinite", math_is_infinite),
        ("$Math$is_finite", math_is_finite),
        ("$Math$infinity", math_infinity),
        ("$Math$neg_infinity", math_neg_infinity),
        ("$Math$nan", math_nan),
        ("$Math$fmod", math_fmod),
        ("$Math$lerp", math_lerp),
        ("$Math$to_radians", math_to_radians),
        ("$Math$to_degrees", math_to_degrees),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trig() {
        assert!((math_sin(0.0) - 0.0).abs() < 1e-10);
        assert!((math_cos(0.0) - 1.0).abs() < 1e-10);
        assert!((math_sin(math_pi() / 2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_log() {
        assert!((math_exp(0.0) - 1.0).abs() < 1e-10);
        assert!((math_log(math_e()) - 1.0).abs() < 1e-10);
        assert!((math_sqrt(4.0) - 2.0).abs() < 1e-10);
        assert!((math_pow(2.0, 3.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_rounding() {
        assert_eq!(math_floor(3.7), 3.0);
        assert_eq!(math_ceil(3.2), 4.0);
        assert_eq!(math_round(3.5), 4.0);
        assert_eq!(math_trunc(-3.7), -3.0);
    }

    #[test]
    fn test_min_max() {
        assert_eq!(math_min(3.0, 5.0), 3.0);
        assert_eq!(math_max(3.0, 5.0), 5.0);
        assert_eq!(math_clamp(10.0, 0.0, 5.0), 5.0);
        assert_eq!(math_clamp(-1.0, 0.0, 5.0), 0.0);
    }

    #[test]
    fn test_random() {
        math_seed(12345);
        let r1 = math_random();
        let r2 = math_random();
        assert!(r1 >= 0.0 && r1 < 1.0);
        assert!(r2 >= 0.0 && r2 < 1.0);
        assert_ne!(r1, r2);

        let ri = math_random_int(10, 20);
        assert!(ri >= 10 && ri <= 20);
    }

    #[test]
    fn test_constants() {
        assert!((math_pi() - 3.141592653589793).abs() < 1e-10);
        assert!((math_e() - 2.718281828459045).abs() < 1e-10);
        assert!((math_tau() - 2.0 * math_pi()).abs() < 1e-10);
    }

    #[test]
    fn test_special() {
        assert_eq!(math_is_nan(f64::NAN), 1);
        assert_eq!(math_is_nan(1.0), 0);
        assert_eq!(math_is_infinite(f64::INFINITY), 1);
        assert_eq!(math_is_finite(1.0), 1);
    }

    #[test]
    fn test_lerp() {
        assert_eq!(math_lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(math_lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(math_lerp(0.0, 10.0, 1.0), 10.0);
    }

    #[test]
    fn test_degrees_radians() {
        assert!((math_to_radians(180.0) - math_pi()).abs() < 1e-10);
        assert!((math_to_degrees(math_pi()) - 180.0).abs() < 1e-10);
    }
}
