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
