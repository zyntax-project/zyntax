//! A loop-free function's first call from native code runs a quick
//! baseline, the body as lowered, until the worker's optimised compile
//! replaces it. The quick code is reached through the stub alone: the
//! call cell keeps the stub, so nothing that reads the cell once (a
//! closure, a function value) keeps the quick code after the
//! replacement lands.

#![cfg(feature = "cranelift-backend")]

use std::collections::HashSet;

use indexmap::IndexMap;
use zyntax_compiler::hir::{
    BinaryOp, HirBlock, HirConstant, HirFunction, HirFunctionSignature, HirId, HirInstruction,
    HirModule, HirParam, HirTerminator, HirType, HirValue, HirValueKind,
};
use zyntax_compiler::tiered_backend::{TieredBackend, TieredConfig};
use zyntax_typed_ast::InternedString;

/// `add_one(x: i32) -> i32 { x + 1 }`, left for its first call as the
/// runtime leaves a program's functions.
fn add_one() -> HirFunction {
    let entry_id = HirId::new();
    let x = HirId::new();
    let one = HirId::new();
    let sum = HirId::new();
    let value = |id, kind| HirValue {
        id,
        ty: HirType::I32,
        kind,
        uses: Default::default(),
        span: None,
    };
    let mut values: IndexMap<HirId, HirValue> = IndexMap::new();
    values.insert(x, value(x, HirValueKind::Parameter(0)));
    values.insert(one, value(one, HirValueKind::Constant(HirConstant::I32(1))));
    values.insert(sum, value(sum, HirValueKind::Instruction));
    let block = HirBlock {
        id: entry_id,
        label: Some(InternedString::new_global("entry")),
        phis: vec![],
        instructions: vec![HirInstruction::Binary {
            op: BinaryOp::Add,
            result: sum,
            ty: HirType::I32,
            left: x,
            right: one,
        }],
        terminator: HirTerminator::Return { values: vec![sum] },
        dominance_frontier: Default::default(),
        predecessors: vec![],
        successors: vec![],
    };
    let mut blocks = IndexMap::new();
    blocks.insert(entry_id, block);
    let mut function = HirFunction::new(
        InternedString::new_global("add_one"),
        HirFunctionSignature {
            params: vec![HirParam {
                id: x,
                name: InternedString::new_global("x"),
                ty: HirType::I32,
                attributes: Default::default(),
                ownership: Default::default(),
            }],
            returns: vec![HirType::I32],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: true,
        },
    );
    function.entry_block = entry_id;
    function.blocks = blocks;
    function.values = values;
    function.attributes.optimized = true;
    function.attributes.deferred = true;
    function
}

/// One test: the lazy compiler a backend installs is the process's.
#[test]
fn a_quick_baseline_never_reaches_the_call_cell() {
    let function = add_one();
    let id = function.id;
    let mut module = HirModule::new(InternedString::new_global("quick_baseline"));
    module.functions.insert(id, function);
    let mut backend = TieredBackend::new(TieredConfig::default()).expect("tiered backend");
    backend.set_emit_osr_probes(true);
    backend
        .compile_module_lazily(module, None, HashSet::from([id]), HashSet::new(), false)
        .expect("module compiles");
    let (_, entry, bead) = backend.interpreter_bridge();
    let stub = entry(id).expect("the stub is in the cell");
    let Some(bead) = bead(id) else {
        // OSR off in this environment: no bead to read the entry of.
        return;
    };

    // SAFETY: the stub stands for `add_one`, `fn(i32) -> i32`.
    let call: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(stub) };
    assert_eq!(call(41), 42);
    let cell = entry(id).expect("an entry") as usize;
    let first = zyntax_compiler::osr::published_entry(bead);

    // A first entry that changes was a quick baseline: it never sat in
    // the cell. One that stays was the optimised compile, and the cell
    // may hold it.
    let mut replaced = false;
    for _ in 0..500 {
        if zyntax_compiler::osr::published_entry(bead) != first {
            replaced = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    if replaced {
        assert_ne!(cell, first, "the quick baseline was published to the cell");
        assert_eq!(
            cell, stub as usize,
            "the cell lost its stub to the quick baseline"
        );
        let last = entry(id).expect("an entry") as usize;
        assert_eq!(
            last,
            zyntax_compiler::osr::published_entry(bead),
            "the replacement took the cell"
        );
    }
    assert_eq!(call(1), 2);
}
