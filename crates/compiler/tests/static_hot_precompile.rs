//! Static hotness can schedule background compilation, but an entry
//! without native code starts on the lowered interpreter body.

#![cfg(feature = "cranelift-backend")]

mod common;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use zyntax_compiler::hir_interp::HirInterpreter;
use zyntax_compiler::opt_audit;
use zyntax_compiler::value::ZyntaxValue;

use zyntax_compiler::hir::{
    HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirType, HirValue, HirValueKind,
};
use zyntax_compiler::tiered_backend::{TieredBackend, TieredConfig};
use zyntax_typed_ast::InternedString;

const TRIPS: i32 = 5000;

/// `count_to` with its bound the constant [`TRIPS`] rather than its
/// parameter, left for its first call as the runtime leaves a program's
/// functions.
fn constant_bound_loop() -> (HirFunction, HirId) {
    let (mut function, header) = common::counted_loop();
    let bound = HirId::new();
    function.values.insert(
        bound,
        HirValue {
            id: bound,
            ty: HirType::I32,
            kind: HirValueKind::Constant(HirConstant::I32(TRIPS)),
            span: None,
        },
    );
    for inst in &mut function.blocks.get_mut(&header).unwrap().instructions {
        if let HirInstruction::Binary { right, .. } = inst {
            *right = bound;
        }
    }
    function.attributes.optimized = false;
    function.attributes.deferred = true;
    (function, header)
}

fn backend_for(function: HirFunction, config: TieredConfig) -> (TieredBackend, HirId) {
    let id = function.id;
    let mut module = HirModule::new(InternedString::new_global("static_hot"));
    module.functions.insert(id, function);
    let mut backend = TieredBackend::new(config).expect("tiered backend");
    backend.set_emit_osr_probes(true);
    backend
        .compile_module_lazily(module, None, HashSet::from([id]), HashSet::new(), false)
        .expect("module compiles");
    (backend, id)
}

fn run(entry: *const u8) -> i32 {
    // SAFETY: the entry is `count_to`'s code, `fn(i32) -> i32`.
    let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(entry) };
    f(0)
}

/// One test: the lazy compiler and warm-up configuration are process-wide.
#[test]
fn a_hot_static_loop_starts_on_lowered_ir_without_waiting_for_compilation() {
    // No worker can win the race and publish native code before the first call.
    // SAFETY: the only test in this binary; set before starting any workers.
    unsafe { std::env::set_var("ZYNTAX_DISABLE_WARM_UP", "1") };
    opt_audit::enable();
    let (function, _) = constant_bound_loop();
    assert!(zyntax_compiler::loop_facts::hot_static_loop(&function));
    let config = TieredConfig {
        enable_osr: false,
        ..TieredConfig::default()
    };
    let (backend, id) = backend_for(function, config);
    let mut tick = backend.interpreter_tick_callback(id).expect("a tick");
    assert_eq!(
        tick(),
        None,
        "static hotness must not compile on first entry"
    );
    assert_eq!(opt_audit::pipeline_runs(id), 0);

    let body = backend.interpreter_body_source()(id).expect("a lowered body");
    assert!(!body.attributes.optimized);
    assert_eq!(
        opt_audit::pipeline_runs(id),
        0,
        "body preparation must not optimize"
    );
    backend.set_ticked_functions([id], []);
    let binding = backend.interpreter_binding(Arc::new(HashMap::new()));
    let mut interpreter = HirInterpreter::new();
    binding.bind(&mut interpreter);
    let module = backend.interpreter_module(id).expect("module");
    let result = interpreter
        .call(&module, "count_to", vec![ZyntaxValue::Int(0)])
        .unwrap();
    assert!(matches!(result, ZyntaxValue::Int(n) if n == i64::from((0..TRIPS).sum::<i32>())));
    assert_eq!(
        opt_audit::pipeline_runs(id),
        0,
        "the short call finishes interpreted"
    );

    // Actual call hotness still compiles a baseline when no worker is available.
    let entry = (0..20)
        .find_map(|_| tick())
        .expect("baseline after hot calls");
    assert_eq!(run(entry), (0..TRIPS).sum::<i32>());
    assert_eq!(tick(), Some(entry));
}
