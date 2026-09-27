//! A function the baseline cannot compile stays interpreted when a hot
//! call asks for its code.
//!
//! A call counted past the threshold with no compile worker to hand it
//! to compiles on the calling thread. A compile that fails leaves the
//! function to the interpreter; the process goes on.

#![cfg(feature = "cranelift-backend")]

use std::collections::HashSet;

use zyntax_compiler::hir::{
    HirConstant, HirFunction, HirFunctionSignature, HirInstruction, HirModule, HirTerminator,
    HirType, HirValueKind,
};
use zyntax_compiler::tiered_backend::{TieredBackend, TieredConfig};
use zyntax_typed_ast::InternedString;

/// `fn wide() -> i64 { splat(1.0) : <8 x f32>; 0 }`: a vector wider
/// than Cranelift holds, so the baseline declines it.
fn wide_vector_function() -> HirFunction {
    let signature = HirFunctionSignature {
        params: vec![],
        returns: vec![HirType::I64],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };
    let mut f = HirFunction::new(InternedString::new_global("wide"), signature);
    let f32x8 = HirType::Vector(Box::new(HirType::F32), 8);
    let one = f.create_value(HirType::F32, HirValueKind::Constant(HirConstant::F32(1.0)));
    let zero = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
    let v = f.create_value(f32x8.clone(), HirValueKind::Instruction);
    let entry = f.entry_block;
    let blk = f.blocks.get_mut(&entry).unwrap();
    blk.instructions.push(HirInstruction::VectorSplat {
        result: v,
        ty: f32x8,
        scalar: one,
    });
    blk.terminator = HirTerminator::Return { values: vec![zero] };
    f.attributes.optimized = true;
    f.attributes.deferred = true;
    f
}

/// One test: the lazy compiler a backend installs is the process's.
#[test]
fn a_hot_call_to_a_function_that_does_not_compile_runs_interpreted() {
    // No worker: the hot call compiles on its own thread.
    // SAFETY: the only test in this binary, set before any thread reads
    // the environment.
    unsafe { std::env::set_var("ZYNTAX_DISABLE_WARM_UP", "1") };
    let function = wide_vector_function();
    let id = function.id;
    let mut module = HirModule::new(InternedString::new_global("declined"));
    module.functions.insert(id, function);
    let config = TieredConfig {
        baseline_threshold: 1,
        ..TieredConfig::default()
    };
    let mut backend = TieredBackend::new(config).expect("tiered backend");
    backend
        .compile_module_lazily(module, None, HashSet::from([id]), HashSet::new(), false)
        .expect("module compiles");
    let mut tick = backend.interpreter_tick_callback(id).expect("a tick");
    for _ in 0..3 {
        assert_eq!(tick(), None, "no code for a function the baseline declines");
    }
}
