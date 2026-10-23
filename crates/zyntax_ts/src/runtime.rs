//! Native primitives; storage lifetimes are handled by the shared drop pass.

extern "C" fn print_number(value: f64) {
    if value == 0.0 {
        print!("{}", if value.is_sign_negative() { "-0" } else { "0" });
    } else {
        print!("{}", dragonbox_ecma::Buffer::new().format(value));
    }
}

extern "C" fn number_to_string(value: f64) -> zrtl::StringPtr {
    let mut buffer = dragonbox_ecma::Buffer::new();
    zrtl::string_new(if value == 0.0 {
        "0"
    } else {
        buffer.format(value)
    })
}

extern "C" fn number_pow(base: f64, exponent: f64) -> f64 {
    if base.abs() == 1.0 && exponent.is_infinite() {
        f64::NAN
    } else {
        base.powf(exponent)
    }
}

macro_rules! unary {
    ($($name:ident => $body:expr),* $(,)?) => {
        $(extern "C" fn $name(x: f64) -> f64 { let f: fn(f64) -> f64 = $body; f(x) })*
    };
}

unary! {
    math_ceil => f64::ceil,
    math_trunc => f64::trunc,
    math_sin => f64::sin,
    math_cos => f64::cos,
    math_tan => f64::tan,
    math_asin => f64::asin,
    math_acos => f64::acos,
    math_atan => f64::atan,
    math_exp => f64::exp,
    math_log => f64::ln,
    math_log2 => f64::log2,
    math_log10 => f64::log10,
    math_cbrt => f64::cbrt,
}

/// Rounds half toward positive infinity and keeps the sign of zero.
extern "C" fn math_round(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let floor = x.floor();
    let rounded = if x - floor >= 0.5 { floor + 1.0 } else { floor };
    if rounded == 0.0 && x.is_sign_negative() {
        -0.0
    } else {
        rounded
    }
}

extern "C" fn math_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// NaN if either operand is NaN; -0 is below +0.
extern "C" fn math_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() { a } else { b }
    } else {
        a.min(b)
    }
}

extern "C" fn math_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() { b } else { a }
    } else {
        a.max(b)
    }
}

/// Ends the process without running exit handlers, after flushing every output buffer.
///
/// A tier-up compile can still be running when the program ends, and the exit handlers of the
/// LLVM library wait for it, which delays a short program by the length of that compile (and
/// has crashed the process when they tear down state under a running compile). The system
/// reclaims everything at exit, so every way out of a run goes through here.
pub fn exit_process(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(unix)]
    {
        // SAFETY: a null stream flushes every open C stream; `_exit` takes no pointers and
        // never returns.
        unsafe {
            libc::fflush(std::ptr::null_mut());
            libc::_exit(code)
        }
    }
    #[cfg(not(unix))]
    std::process::exit(code)
}

/// An index TypeScript would read as `undefined` (or use to grow the array).
extern "C" fn index_trap(index: f64, length: f64) {
    let shown = if index.is_nan() {
        "NaN".to_string()
    } else if index == 0.0 {
        "0".to_string()
    } else {
        dragonbox_ecma::Buffer::new().format(index).to_string()
    };
    eprintln!(
        "RangeError: array index {shown} is outside 0..{length}; TypeScript would read undefined or grow the array"
    );
    exit_process(1);
}

/// A length `new Array(n)` rejects: negative, fractional, NaN or past 2^32 - 1.
extern "C" fn array_length_trap(_length: f64) {
    eprintln!("RangeError: Invalid array length");
    exit_process(1);
}

/// Raises the `ReferenceError` for a `let` or `const` reached before its declaration has run.
extern "C" fn binding_check(ready: bool, name: zrtl::StringConstPtr) {
    if ready {
        return;
    }
    // SAFETY: the compiler passes the binding's name as a string literal.
    let name = unsafe { zrtl::string_as_str(name) }.unwrap_or_default();
    eprintln!("ReferenceError: Cannot access '{name}' before initialization");
    exit_process(1);
}

/// A module binding read before any assignment, where TypeScript would read `undefined`.
extern "C" fn assigned_check(assigned: bool, name: zrtl::StringConstPtr) {
    if assigned {
        return;
    }
    // SAFETY: the compiler passes the binding's name as a string literal.
    let name = unsafe { zrtl::string_as_str(name) }.unwrap_or_default();
    eprintln!("TypeError: '{name}' is read before it is assigned; TypeScript would read undefined");
    exit_process(1);
}

/// A regular enum read before its declaration has run: the enum object is still `undefined`.
extern "C" fn enum_check(ready: bool, name: zrtl::StringConstPtr) {
    if ready {
        return;
    }
    // SAFETY: the compiler passes the enum's name as a string literal.
    let name = unsafe { zrtl::string_as_str(name) }.unwrap_or_default();
    eprintln!("TypeError: Cannot read properties of undefined (enum '{name}' is not declared yet)");
    exit_process(1);
}

/// ToInt32 for NaN, infinities and magnitudes the inline conversion does not cover.
extern "C" fn to_int32_slow(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    let wrapped = value.trunc().rem_euclid(4_294_967_296.0) as u64;
    i64::from(wrapped as u32 as i32)
}

/// Milliseconds since the Unix epoch, whole, as `Date.now()` reports.
extern "C" fn date_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_millis() as f64)
}

/// Milliseconds since the first read of the clock, with sub-millisecond resolution.
extern "C" fn performance_now() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
        * 1000.0
}

extern "C" fn print_undefined() {
    print!("undefined");
}

extern "C" fn print_space() {
    print!(" ");
}
extern "C" fn print_newline() {
    println!();
}

zrtl::zrtl_plugin! {
    name: "typescript_native",
    symbols: [
        ("$TS$print_number", print_number),
        ("$TS$print_undefined", print_undefined),
        ("$TS$number_to_string", number_to_string),
        ("$TS$pow", number_pow),
        ("$TS$index_trap", index_trap),
        ("$TS$array_length_trap", array_length_trap),
        ("$TS$binding_check", binding_check),
        ("$TS$assigned_check", assigned_check),
        ("$TS$enum_check", enum_check),
        ("$TS$date_now", date_now),
        ("$TS$performance_now", performance_now),
        ("$TS$to_int32_slow", to_int32_slow),
        ("$TS$math_ceil", math_ceil),
        ("$TS$math_trunc", math_trunc),
        ("$TS$math_round", math_round),
        ("$TS$math_sin", math_sin),
        ("$TS$math_cos", math_cos),
        ("$TS$math_tan", math_tan),
        ("$TS$math_asin", math_asin),
        ("$TS$math_acos", math_acos),
        ("$TS$math_atan", math_atan),
        ("$TS$math_atan2", math_atan2),
        ("$TS$math_exp", math_exp),
        ("$TS$math_log", math_log),
        ("$TS$math_log2", math_log2),
        ("$TS$math_log10", math_log10),
        ("$TS$math_cbrt", math_cbrt),
        ("$TS$math_min", math_min),
        ("$TS$math_max", math_max),
        ("$TS$space", print_space),
        ("$TS$newline", print_newline),
    ]
}

/// Install native primitives and enable proven releases without a collector.
pub fn register_runtime(
    runtime: &mut zyntax_embed::TieredRuntime,
) -> Result<(), zyntax_embed::RuntimeError> {
    runtime.set_collector(zyntax_embed::Collector::None);
    runtime.set_automatic_release(true);
    runtime.set_pattern_rewrites(false);
    runtime.register_static_plugins([
        zrtl_io::static_plugin(),
        zrtl_string::static_plugin(),
        static_plugin(),
    ])
}
