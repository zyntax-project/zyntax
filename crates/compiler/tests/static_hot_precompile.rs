//! A function whose loop runs a constant number of times, enough to be
//! hot, is compiled before its first call: the interpreter's first call
//! gets native code from the tick, waiting for a compile still under
//! way rather than running the body itself.

#![cfg(feature = "cranelift-backend")]

mod common;

use std::collections::HashSet;

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
            uses: Default::default(),
            span: None,
        },
    );
    for inst in &mut function.blocks.get_mut(&header).unwrap().instructions {
        if let HirInstruction::Binary { right, .. } = inst {
            *right = bound;
        }
    }
    function.attributes.optimized = true;
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

/// One test: the lazy compiler a backend installs is the process's, so
/// two backends made side by side would answer each other's calls.
#[test]
fn a_hot_static_loop_is_compiled_before_its_first_call() {
    let sum: i32 = (0..TRIPS).sum();

    // Asked at once, so the compile the worker was given at load is
    // still under way, or not yet begun: the tick waits for it or makes
    // it, and the call never starts interpreted.
    let (function, _) = constant_bound_loop();
    let (backend, id) = backend_for(function, TieredConfig::default());
    let mut tick = backend.interpreter_tick_callback(id).expect("a tick");
    let entry = tick().expect("native code at the first call");
    assert_eq!(run(entry), sum);
    // The second call finds the code installed.
    assert_eq!(tick(), Some(entry));
    drop(tick);
    drop(backend);

    // A bound the loop reads from its parameter leaves the first call
    // to the interpreter.
    let (mut function, _) = common::counted_loop();
    function.attributes.optimized = true;
    function.attributes.deferred = true;
    let (backend, id) = backend_for(function, TieredConfig::default());
    let mut tick = backend.interpreter_tick_callback(id).expect("a tick");
    assert_eq!(tick(), None);
    drop(tick);
    drop(backend);

    // Once the baseline is in, the optimizing tier is asked for at once
    // and installs without the loop being run.
    #[cfg(feature = "llvm-backend")]
    {
        use zyntax_compiler::tiered_backend::Tier2Backend;

        let (function, _) = constant_bound_loop();
        let config = TieredConfig {
            tier2_backend: Tier2Backend::LLVM,
            ..TieredConfig::default()
        };
        let (backend, id) = backend_for(function, config);
        let mut tick = backend.interpreter_tick_callback(id).expect("a tick");
        let baseline = tick().expect("native code at the first call");
        let mut promoted = None;
        for _ in 0..600 {
            match backend.promoted_function_pointer(id) {
                Some(p) if p != baseline => {
                    promoted = Some(p);
                    break;
                }
                _ => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        }
        let promoted = promoted.expect("the optimizing tier installs");
        assert_eq!(run(promoted), sum);
    }
}
