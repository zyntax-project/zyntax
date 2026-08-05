//! Shared HIR fixtures for the tier / OSR integration tests.
//!
//! Every test binary that includes this module compiles all of it, so a
//! fixture only some of them need still has to be allowed to go unused.
#![allow(dead_code)]

use indexmap::IndexMap;

use zyntax_compiler::hir::{
    BinaryOp, HirBlock, HirConstant, HirFunction, HirFunctionSignature, HirId, HirInstruction,
    HirParam, HirPhi, HirTerminator, HirType, HirValue, HirValueKind,
};
use zyntax_typed_ast::InternedString;

/// [`counted_loop`] with an extra loop-carried aggregate, plus the header's
/// `HirId`.
///
/// An aggregate is where the two backends stop agreeing about
/// representation — one holds it as a pointer, the other as a value — so a
/// live-in of this shape is what a helper's frame has to reconcile.
pub fn aggregate_carrying_loop() -> (HirFunction, HirId) {
    let (mut function, header_id) = counted_loop();

    // The phi is declared as the aggregate, but every value feeding it is
    // already a pointer to one, so the emitted phi takes the producers'
    // pointer shape rather than the declared aggregate's.
    let aggregate_ty = HirType::Array(Box::new(HirType::F64), 3);
    let producer_ty = HirType::Ptr(Box::new(HirType::F64));

    // Carried around the loop untouched: the header phi's only other input
    // is itself, which is precisely the self-referential shape that has to
    // agree with whatever the frame hands back.
    let body_param = HirId::new();
    let phi_body = HirId::new();
    function.values.insert(
        body_param,
        HirValue {
            id: body_param,
            ty: producer_ty.clone(),
            kind: HirValueKind::Parameter(1),
            uses: Default::default(),
            span: None,
        },
    );
    function.values.insert(
        phi_body,
        HirValue {
            id: phi_body,
            ty: producer_ty.clone(),
            kind: HirValueKind::Instruction,
            uses: Default::default(),
            span: None,
        },
    );

    let entry_id = function.entry_block;
    let body_id = function
        .blocks
        .get(&header_id)
        .and_then(|b| b.successors.first().copied())
        .expect("header should branch into a body");
    function
        .blocks
        .get_mut(&header_id)
        .expect("header exists")
        .phis
        .push(HirPhi {
            result: phi_body,
            ty: aggregate_ty.clone(),
            incoming: vec![(body_param, entry_id), (phi_body, body_id)],
        });

    function.signature.params.push(HirParam {
        id: body_param,
        name: InternedString::new_global("body"),
        ty: producer_ty,
        attributes: Default::default(),
    });

    (function, header_id)
}

/// `fn count_to(n: i32) -> i32 { let mut sum = 0; for i in 0..n { sum += i } sum }`
/// as hand-built HIR, plus the loop header's `HirId`.
pub fn counted_loop() -> (HirFunction, HirId) {
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
