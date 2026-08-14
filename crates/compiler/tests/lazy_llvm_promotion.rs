//! Does on-demand LLVM promotion work for a single function?
//!
//! The shipping LLVM path compiles the whole module in a background thread
//! at install and only defers the pointer swap. Lazy LLVM instead wants one
//! function compiled when it gets hot, which is the `Tier2Backend::LLVM`
//! route through `compile_at_tier`. This drives that route directly.

#![cfg(all(feature = "llvm-backend", feature = "cranelift-backend"))]

mod common;

use zyntax_compiler::hir::HirModule;
use zyntax_compiler::tiered_backend::{
    OptimizationTier, Tier2Backend, TieredBackend, TieredConfig,
};
use zyntax_typed_ast::InternedString;

#[test]
fn a_single_function_promotes_to_llvm_on_demand() {
    let (function, _header) = common::counted_loop();
    let func_id = function.id;

    let mut module = HirModule::new(InternedString::new_global("lazy_llvm"));
    module.functions.insert(func_id, function);

    let mut config = TieredConfig::default();
    config.tier2_backend = Tier2Backend::LLVM;
    config.verbosity = 2;

    let mut backend = TieredBackend::new(config).expect("tiered backend with LLVM tier 2");
    backend.compile_module(module).expect("tier-0 compile");

    let before = backend.get_function_pointer(func_id);
    assert!(before.is_some(), "tier-0 should install a pointer");

    backend
        .optimize_function(func_id, OptimizationTier::Optimized)
        .expect("force promote to tier 2 (LLVM)");

    let mut swapped = false;
    for _ in 0..300 {
        if backend.get_function_pointer(func_id) != before {
            swapped = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        swapped,
        "promoting to tier 2 should install a different (LLVM) code pointer"
    );
}
