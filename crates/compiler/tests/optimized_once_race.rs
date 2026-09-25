//! Callers asking for a lazy function's optimised body at once, while
//! the compile worker makes it too, all get one body, made by one run of
//! the pipeline.

#![cfg(feature = "cranelift-backend")]

mod common;

use std::collections::HashSet;
use std::sync::{Arc, Barrier};

use zyntax_compiler::hir::{
    HirConstant, HirInstruction, HirModule, HirType, HirValue, HirValueKind,
};
use zyntax_compiler::opt_audit;
use zyntax_compiler::osr;
use zyntax_compiler::tiered_backend::{TieredBackend, TieredConfig};
use zyntax_typed_ast::InternedString;

const ASKERS: usize = 8;

/// One test: the lazy optimizer a backend installs is the process's.
#[test]
fn racing_callers_share_one_run_of_the_pipeline() {
    opt_audit::enable();
    for _ in 0..200 {
        // A loop hot before its first call, so the worker is asked to
        // compile it as the module loads.
        let (mut function, header) = common::counted_loop();
        let bound = zyntax_compiler::hir::HirId::new();
        function.values.insert(
            bound,
            HirValue {
                id: bound,
                ty: HirType::I32,
                kind: HirValueKind::Constant(HirConstant::I32(5000)),
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
        let id = function.id;
        let mut module = HirModule::new(InternedString::new_global("optimized_once_race"));
        module.functions.insert(id, function);
        let config = TieredConfig {
            enable_osr: true,
            ..TieredConfig::default()
        };
        let mut backend = TieredBackend::new(config).expect("tiered backend");
        backend
            .compile_module_lazily(module, None, HashSet::from([id]), HashSet::new(), false)
            .expect("module compiles");
        let (_, _, bead) = backend.interpreter_bridge();
        let bead = bead(id).expect("a bead");

        let barrier = Arc::new(Barrier::new(ASKERS));
        let askers: Vec<_> = (0..ASKERS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    osr::lazy_optimized_body(bead).expect("an optimised body")
                })
            })
            .collect();
        let bodies: Vec<_> = askers
            .into_iter()
            .map(|t| t.join().expect("asker"))
            .collect();
        assert!(
            bodies.iter().all(|b| Arc::ptr_eq(b, &bodies[0])),
            "callers were handed different bodies"
        );
        drop(backend);
        assert_eq!(
            opt_audit::pipeline_runs(id),
            1,
            "the pipeline ran more than once over one function"
        );
    }
}
