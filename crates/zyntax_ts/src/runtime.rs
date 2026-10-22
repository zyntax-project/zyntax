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
