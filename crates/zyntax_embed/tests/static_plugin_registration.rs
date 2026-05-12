//! Phase C smoke test — verifies that a statically-linked ZRTL plugin
//! can be registered into `ZyntaxRuntime` without going through
//! `dlopen` / `libloading`, and that its symbols are then resolvable
//! from both the native backend's symbol table and the BC
//! interpreter's FFI table.
//!
//! This is the wasm32 plugin entry point in miniature: same
//! `register_static_plugin` call, run on native because the
//! native+wasm code paths share the same registration code.

use zrtl::zrtl_plugin;
use zyntax_embed::ZyntaxRuntime;

// Define a tiny inline plugin via the SDK macro. The macro emits the
// `static_plugin()` accessor we exercise below.
extern "C" fn smoke_plugin_add(a: i64, b: i64) -> i64 {
    a + b
}

extern "C" fn smoke_plugin_neg(a: i64) -> i64 {
    -a
}

zrtl_plugin! {
    name: "smoke_static",
    symbols: [
        ("$Smoke$add", smoke_plugin_add, (i64, i64) -> i64),
        ("$Smoke$neg", smoke_plugin_neg, (i64) -> i64),
    ]
}

#[test]
fn static_plugin_round_trip() {
    let plugin = static_plugin();

    // Plugin metadata accessible without dlopen.
    assert_eq!(plugin.name(), Some("smoke_static"));
    assert_eq!(plugin.symbols.len(), 2);

    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.register_static_plugin(plugin)
        .expect("register_static_plugin should succeed");

    // The symbol pointer should now be reachable through the
    // runtime's external-function table (same path as a dlopen'd
    // plugin — registered via `register_function` internally).
    let add_ptr = rt
        .external_function_ptr("$Smoke$add")
        .expect("add should be registered");
    assert_eq!(add_ptr as usize, smoke_plugin_add as *const () as usize);

    let neg_ptr = rt
        .external_function_ptr("$Smoke$neg")
        .expect("neg should be registered");
    assert_eq!(neg_ptr as usize, smoke_plugin_neg as *const () as usize);

    // Plugin signatures should also be populated, so
    // `Grammar2::parse_with_signatures` sees them for type checks.
    let sigs = rt.plugin_signatures();
    assert!(sigs.contains_key("$Smoke$add"));
    assert!(sigs.contains_key("$Smoke$neg"));
    assert_eq!(sigs["$Smoke$add"].param_count, 2);
    assert_eq!(sigs["$Smoke$neg"].param_count, 1);
}
