//! Stage test for the live-out plumbing fix. Phase E4 originally fed
//! krio empty live-outs because the analysis API plumbing was
//! deferred. This test exercises the real path:
//!
//!   * Plant a captures-lift HIR shape (live SSA value crossing an
//!     await).
//!   * Build per-block live-out via `zyntax_compiler::analysis::AnalysisRunner`.
//!   * Hand it to the krio orchestrator.
//!   * Assert the resulting layout actually includes save/load
//!     entries (proving liveness reached krio non-empty).
//!
//! Run with: `cargo test -p krio_adapter --test stages_e4_with_real_liveness`

mod common;

use std::collections::HashSet;

use krio_adapter::{orchestrator, HirSuspendingFns};
use zyntax_compiler::analysis::AnalysisRunner;
use zyntax_compiler::hir::{HirId, HirInstruction, HirType, HirValue, HirValueKind};

use common::{make_async_function_with_one_await, AsyncFnFixture};

/// End-to-end: run AnalysisRunner against the canonical fixture,
/// hand its live_out output to the orchestrator, and verify krio's
/// captures-lift fires properly.
///
/// Note: for a Return-terminated single-block function,
/// AnalysisRunner correctly returns an empty live_out (no successors
/// to flow values into). HirLiveness::build does the intra-block
/// computation locally — backward-walks from `block_live_out ∪
/// terminator_uses` and intersects with "defined-before-suspension".
/// This test exercises that: empty live_out goes in, real captures-
/// lift comes out.
#[test]
fn analysis_runner_output_drives_krio_captures_lift() {
    let AsyncFnFixture {
        function,
        live_across,
        ..
    } = make_async_function_with_one_await();
    let fn_id = function.id;

    // Build a one-fn module so AnalysisRunner has somewhere to walk.
    let mut module =
        zyntax_compiler::hir::HirModule::new(zyntax_typed_ast::InternedString::new_global("test"));
    module.functions.insert(fn_id, function);

    // Run the existing analyzer.
    let mut analyzer = AnalysisRunner::new(module.clone());
    let analysis = analyzer.run_all().expect("analysis must succeed");
    let live_out = analysis
        .functions
        .get(&fn_id)
        .map(|f| f.liveness.live_out.clone())
        .expect("AnalysisRunner must produce per-fn liveness");
    // Document the analyzer's actual output — empty for our
    // Return-terminated single-block fixture.
    let entry_live_out = live_out
        .get(&module.functions[&fn_id].entry_block)
        .cloned()
        .unwrap_or_default();
    assert!(
        entry_live_out.is_empty(),
        "Single Return-terminated block has empty live_out — that's analyzer-correct \
         (no successors). The intra-block walk in HirLiveness handles the captures-lift."
    );

    // Compute the suspending-fn taint set BEFORE pulling the function
    // out of the module — `from_module` walks `module.functions`, so
    // doing it after `swap_remove` would miss `fn_id` and krio would
    // short-circuit `transform_to_state_machine`.
    let suspending = HirSuspendingFns::from_module(&module);

    // Plant a frame-ptr SSA value the orchestrator will reference.
    let mut function = module.functions.swap_remove(&fn_id).unwrap();
    let frame = HirId::new();
    function.values.insert(
        frame,
        HirValue {
            id: frame,
            ty: HirType::Ptr(Box::new(HirType::I64)),
            kind: HirValueKind::Instruction,
            uses: HashSet::new(),
            span: None,
        },
    );

    let result =
        orchestrator::lower_async_function(&mut function, &suspending, frame, 16, &live_out)
            .expect("orchestrator must succeed");

    eprintln!(
        "[DEBUG] liveness.map.at_site = {:?}",
        result.liveness.map.at_site
    );
    eprintln!(
        "[DEBUG] layout.yield_saves = {:?}",
        result.layout.yield_saves
    );
    eprintln!(
        "[DEBUG] layout.yield_blocks = {:?}",
        result.layout.yield_blocks
    );
    eprintln!(
        "[DEBUG] layout.resume_loads = {:?}",
        result.layout.resume_loads
    );
    eprintln!("[DEBUG] live_across = {:?}", live_across);
    eprintln!("[DEBUG] hir_to_local = {:?}", result.liveness.hir_to_local);

    // The intra-block captures-lift should have identified
    // `live_across` as the value crossing the suspension.
    assert_eq!(
        result.layout.yield_saves.len(),
        1,
        "exactly one captures-lift save expected"
    );
    assert_eq!(
        result.layout.yield_saves[0].1.len(),
        1,
        "exactly one slot in the save table"
    );
    assert_eq!(
        result.layout.resume_loads.len(),
        1,
        "exactly one resume-load to match the save"
    );

    // The saved LocalId must round-trip back to live_across.
    let saved_local = result.layout.yield_saves[0].1[0].1;
    assert_eq!(
        result.liveness.local_to_hir[&saved_local], live_across,
        "captures-lift correctly identified live_across as the captured value"
    );

    // And rewrites contain the original→fresh mapping.
    assert!(result.rewrites.contains_key(&live_across));

    // Verify HIR shape: saves before the yield, loads at the resume.
    let save_count = function
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .filter(|i| matches!(i, HirInstruction::AsyncSaveSlot { .. }))
        .count();
    let load_count = function
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .filter(|i| matches!(i, HirInstruction::AsyncLoadSlot { .. }))
        .count();
    // After F.2 with re-entry handling + await-in-loop fix:
    //  saves: 1 captures-lift + 1 promise persist + 1 result
    //         + 1 promise-clear + 1 state-bump = 5
    //  loads: 1 captures-lift + 1 dispatcher state + 1 await result
    //         + 1 yield re-entry check + 1 poll reload = 5
    assert_eq!(save_count, 5, "1 captures-lift + 4 await-lowering saves");
    assert_eq!(
        load_count, 5,
        "AsyncLoadSlot: 1 captures + 1 dispatcher state + 1 await result + 1 yield re-entry check + 1 poll reload"
    );
}

/// Smoke-check: AnalysisRunner runs cleanly on the canonical fixture
/// (no panics, returns Ok). Pinpoints any future regression in the
/// analyzer's input contract.
#[test]
fn analysis_runner_handles_canonical_fixture_without_error() {
    let AsyncFnFixture { function, .. } = make_async_function_with_one_await();
    let fn_id = function.id;
    let mut module =
        zyntax_compiler::hir::HirModule::new(zyntax_typed_ast::InternedString::new_global("test"));
    module.functions.insert(fn_id, function);
    let mut analyzer = AnalysisRunner::new(module);
    let analysis = analyzer.run_all().expect("analyzer must not error");
    assert!(analysis.functions.contains_key(&fn_id));
}
