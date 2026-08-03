//! Does a tier-1 promotion through the real ladder install OSR entries?
//!
//! `test_osr_helper_emission_for_counted_loop` proves a tier-1
//! `compile_function` *produces* helpers. This goes one step further and
//! drives `TieredBackend`, which is what a running program uses, then asks
//! the bead whether an entry actually landed under the layout's site key —
//! the same question `osr_probe` asks from JIT'd code.

use indexmap::IndexMap;

use zyntax_compiler::hir::{
    BinaryOp, HirBlock, HirConstant, HirFunction, HirFunctionSignature, HirId, HirInstruction,
    HirModule, HirParam, HirPhi, HirTerminator, HirType, HirValue, HirValueKind,
};
use zyntax_compiler::osr;
use zyntax_compiler::tiered_backend::{OptimizationTier, TieredBackend, TieredConfig};
use zyntax_typed_ast::InternedString;

/// `fn count_to(n: i32) -> i32 { let mut sum = 0; for i in 0..n { sum += i } sum }`
/// as hand-built HIR, plus the loop header's `HirId`.
fn counted_loop() -> (HirFunction, HirId) {
    let i32_ty = HirType::I32;

    let entry_id = HirId::new();
    let header_id = HirId::new();
    let body_id = HirId::new();
    let exit_id = HirId::new();

    let n_id = HirId::new();
    let zero_i_id = HirId::new();
    let zero_sum_id = HirId::new();
    let one_id = HirId::new();
    let phi_i = HirId::new();
    let phi_sum = HirId::new();
    let next_sum_id = HirId::new();
    let next_i_id = HirId::new();
    let cmp_id = HirId::new();

    let mut values: IndexMap<HirId, HirValue> = IndexMap::new();
    values.insert(
        n_id,
        HirValue {
            id: n_id,
            ty: i32_ty.clone(),
            kind: HirValueKind::Parameter(0),
            uses: Default::default(),
            span: None,
        },
    );
    for (id, v) in [(zero_i_id, 0), (zero_sum_id, 0), (one_id, 1)] {
        values.insert(
            id,
            HirValue {
                id,
                ty: i32_ty.clone(),
                kind: HirValueKind::Constant(HirConstant::I32(v)),
                uses: Default::default(),
                span: None,
            },
        );
    }
    for id in [phi_i, phi_sum, next_sum_id, next_i_id] {
        values.insert(
            id,
            HirValue {
                id,
                ty: i32_ty.clone(),
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
    }
    values.insert(
        cmp_id,
        HirValue {
            id: cmp_id,
            ty: HirType::Bool,
            kind: HirValueKind::Instruction,
            uses: Default::default(),
            span: None,
        },
    );

    let entry_block = HirBlock {
        id: entry_id,
        label: Some(InternedString::new_global("entry")),
        phis: vec![],
        instructions: vec![],
        terminator: HirTerminator::Branch { target: header_id },
        dominance_frontier: Default::default(),
        predecessors: vec![],
        successors: vec![header_id],
    };

    let header_block = HirBlock {
        id: header_id,
        label: Some(InternedString::new_global("header")),
        phis: vec![
            HirPhi {
                result: phi_i,
                ty: i32_ty.clone(),
                incoming: vec![(zero_i_id, entry_id), (next_i_id, body_id)],
            },
            HirPhi {
                result: phi_sum,
                ty: i32_ty.clone(),
                incoming: vec![(zero_sum_id, entry_id), (next_sum_id, body_id)],
            },
        ],
        instructions: vec![HirInstruction::Binary {
            op: BinaryOp::Lt,
            result: cmp_id,
            ty: HirType::Bool,
            left: phi_i,
            right: n_id,
        }],
        terminator: HirTerminator::CondBranch {
            condition: cmp_id,
            true_target: body_id,
            false_target: exit_id,
        },
        dominance_frontier: Default::default(),
        predecessors: vec![entry_id, body_id],
        successors: vec![body_id, exit_id],
    };

    let body_block = HirBlock {
        id: body_id,
        label: Some(InternedString::new_global("body")),
        phis: vec![],
        instructions: vec![
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: next_sum_id,
                ty: i32_ty.clone(),
                left: phi_sum,
                right: phi_i,
            },
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: next_i_id,
                ty: i32_ty.clone(),
                left: phi_i,
                right: one_id,
            },
        ],
        terminator: HirTerminator::Branch { target: header_id },
        dominance_frontier: Default::default(),
        predecessors: vec![header_id],
        successors: vec![header_id],
    };

    let exit_block = HirBlock {
        id: exit_id,
        label: Some(InternedString::new_global("exit")),
        phis: vec![],
        instructions: vec![],
        terminator: HirTerminator::Return {
            values: vec![phi_sum],
        },
        dominance_frontier: Default::default(),
        predecessors: vec![header_id],
        successors: vec![],
    };

    let mut blocks: IndexMap<HirId, HirBlock> = IndexMap::new();
    blocks.insert(entry_id, entry_block);
    blocks.insert(header_id, header_block);
    blocks.insert(body_id, body_block);
    blocks.insert(exit_id, exit_block);

    let signature = HirFunctionSignature {
        params: vec![HirParam {
            id: n_id,
            name: InternedString::new_global("n"),
            ty: i32_ty.clone(),
            attributes: Default::default(),
        }],
        returns: vec![i32_ty],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: true,
    };

    let mut function = HirFunction::new(InternedString::new_global("count_to"), signature);
    function.values = values;
    function.blocks = blocks;
    function.entry_block = entry_id;
    function.is_external = false;

    (function, header_id)
}

/// Whether any registered bead holds an OSR entry under `site`.
fn any_bead_has_entry(site: u64) -> bool {
    osr::bead_registry()
        .read()
        .unwrap()
        .values()
        .any(|bead| bead.osr_entry(site).is_some_and(|p| !p.is_null()))
}

/// Promoting to tier 1 through `TieredBackend` should leave an OSR entry
/// the runtime probe can find. If this fails, the install side is the gap;
/// if it passes, the gap is only that tier-0 never emits the probe.
#[test]
fn a_tier1_promotion_installs_an_osr_entry() {
    let (function, header_id) = counted_loop();
    let func_id = function.id;

    let layout =
        osr::osr_layout(&function, header_id).expect("counted loop should have an OSR layout");
    let site = layout.site_key();

    assert!(
        !any_bead_has_entry(site),
        "no OSR entry should exist before promotion"
    );

    let mut module = HirModule::new(InternedString::new_global("osr_test"));
    module.functions.insert(func_id, function);

    let mut config = TieredConfig::default();
    config.verbosity = 2;
    let mut backend = TieredBackend::new(config).expect("tiered backend");
    backend.compile_module(module).expect("tier-0 compile");
    backend
        .optimize_function(func_id, OptimizationTier::Standard)
        .expect("force promote to tier 1");

    // Promotion may be queued on a beadie broker thread; poll rather than
    // assuming the compile already ran.
    let mut landed = false;
    for _ in 0..200 {
        if any_bead_has_entry(site) {
            landed = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        landed,
        "tier-1 promotion should install an OSR entry under site 0x{site:x}"
    );
}
