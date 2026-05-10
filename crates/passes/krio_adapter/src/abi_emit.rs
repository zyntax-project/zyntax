//! ABI scaffolding emitter — wraps a krio-transformed HIR function in
//! the runtime-expected calling convention.
//!
//! After `orchestrator::lower_async_function` runs, the function has
//! its captures-lift transform but still has its original signature
//! `(p1, p2, ...) -> T`. The ZynML runtime calls async functions
//! through a Promise-returning entry function whose poll fn has
//! signature `fn(*mut u8) -> i64` (0 = pending, positive = ready
//! value encoded as i64). This module bridges between the two:
//!
//! * [`reshape_to_poll_abi`] mutates the krio-transformed function
//!   in place into the poll ABI: drops original params, adds a
//!   single `state_machine: *mut u8` param, prepends a load-prologue
//!   that reads original-param values from their assigned slots,
//!   rewrites yield-block returns to `0_i64` (pending), wraps
//!   non-yield returns with a cast to i64 (ready value).
//!
//! * [`generate_promise_entry`] returns a NEW function with the
//!   original async name and signature `(p1, p2, ...) -> *Promise<T>`.
//!   It mallocs the state machine, stores params at their slots,
//!   mallocs the 16-byte Promise struct, and returns it.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use krio_async::StateMachineLayout;
use zyntax_compiler::hir::{
    BinaryOp, CastOp, HirBlock, HirCallable, HirConstant, HirFunction, HirFunctionSignature,
    HirFunctionType, HirId, HirInstruction, HirParam, HirStructType, HirTerminator, HirType,
    HirValue, HirValueKind, Intrinsic, ParamAttributes,
};
use zyntax_typed_ast::InternedString;

use crate::{HirBlockId, HirFnId, HirLocalId};

/// Reshape a krio-transformed function into the runtime poll ABI.
///
/// Pre: `function` was mutated by `lower_async_function` — it has the
/// captures-lift transform (AsyncSaveSlot/AsyncLoadSlot ops + Switch
/// dispatcher at entry), and its signature is the original
/// `(p1, p2, ...) -> T` shape.
///
/// Post: signature is `(state_machine: *mut u8) -> i64`. A new entry
/// block is prepended that loads each original param from its slot
/// and branches to the krio dispatcher. All `frame: X` references in
/// AsyncSaveSlot/AsyncLoadSlot ops point at the new state_machine
/// param. Yield-block returns produce `0_i64` (pending sentinel).
/// Non-yield returns wrap their value in a cast to i64.
///
/// The function is renamed `{name}$poll` and `is_async` is cleared.
/// Returns the new poll fn name.
pub fn reshape_to_poll_abi(
    function: &mut HirFunction,
    layout: &StateMachineLayout<HirBlockId, HirLocalId, HirFnId>,
    param_slots: &[(HirId, u32)],
) -> InternedString {
    let original_return_ty = function
        .signature
        .returns
        .first()
        .cloned()
        .unwrap_or(HirType::Void);

    // Step 0: neutralize the original Parameter kinds. The Cranelift
    // backend (cranelift_backend.rs:1432) iterates `function.values`
    // for `HirValueKind::Parameter(idx)` to map params to the entry
    // block's incoming Cranelift values. Leaving the original params
    // as Parameter(N) would collide with the new state_machine param
    // we're about to mint as Parameter(0). The originals are now
    // orphans — body uses got rewritten to load results in Step 3 —
    // so flip them to `Instruction` so they're harmless side data.
    for v in function.values.values_mut() {
        if matches!(v.kind, HirValueKind::Parameter(_)) {
            v.kind = HirValueKind::Instruction;
        }
    }

    // Step 1: mint a fresh `state_machine` SSA value — this becomes
    // the function's sole param, AND the new `frame` for every
    // AsyncSaveSlot/AsyncLoadSlot op currently in the function.
    let sm_ptr_id = HirId::new();
    function.values.insert(
        sm_ptr_id,
        HirValue {
            id: sm_ptr_id,
            ty: HirType::Ptr(Box::new(HirType::U8)),
            kind: HirValueKind::Parameter(0),
            uses: HashSet::new(),
            span: None,
        },
    );

    // Step 2: rewrite all `frame: X` references in async ops to the
    // new state_machine param. Krio originally pointed `frame` at the
    // old first param (or a synthetic HirId from the caller closure);
    // either way, we now own the frame.
    for block in function.blocks.values_mut() {
        for inst in &mut block.instructions {
            match inst {
                HirInstruction::AsyncSaveSlot { frame, .. }
                | HirInstruction::AsyncLoadSlot { frame, .. } => {
                    *frame = sm_ptr_id;
                }
                _ => {}
            }
        }
    }

    // Step 3: build the param-load prologue. For each original param,
    // emit AsyncLoadSlot { result: <fresh>, ty, frame: sm_ptr, slot }
    // and rewrite all uses of the original param HirId to the load
    // result.
    let mut prologue_insts: Vec<HirInstruction> = Vec::with_capacity(param_slots.len());
    let mut rewrites: IndexMap<HirId, HirId> = IndexMap::new();
    for (orig_param_id, slot) in param_slots {
        // Pull the original param's HIR type from values[].
        let orig_ty = function
            .values
            .get(orig_param_id)
            .map(|v| v.ty.clone())
            .unwrap_or(HirType::I64);
        let load_id = HirId::new();
        function.values.insert(
            load_id,
            HirValue {
                id: load_id,
                ty: orig_ty.clone(),
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );
        prologue_insts.push(HirInstruction::AsyncLoadSlot {
            result: load_id,
            ty: orig_ty,
            frame: sm_ptr_id,
            slot: *slot,
        });
        rewrites.insert(*orig_param_id, load_id);
    }

    // Apply rewrites across all instructions and terminators. This
    // re-points the body from the original param HirIds to the new
    // load results.
    for block in function.blocks.values_mut() {
        for inst in &mut block.instructions {
            inst.replace_uses(&rewrites);
        }
        block.terminator.replace_uses(&rewrites);
    }

    // Step 4: insert the prologue block BEFORE the existing entry
    // (which is krio's dispatcher). The prologue branches to the
    // dispatcher unconditionally.
    let old_entry_hir = function.entry_block;
    let prologue_id = HirId::new();
    let prologue_block = HirBlock {
        id: prologue_id,
        label: Some(InternedString::new_global("krio_param_prologue")),
        phis: vec![],
        instructions: prologue_insts,
        terminator: HirTerminator::Branch {
            target: old_entry_hir,
        },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![old_entry_hir],
    };

    // IndexMap insertion order matters for backend iteration. Build a
    // new IndexMap with the prologue first, then all existing blocks.
    let mut new_blocks: IndexMap<HirId, HirBlock> = IndexMap::new();
    new_blocks.insert(prologue_id, prologue_block);
    for (id, block) in function.blocks.drain(..) {
        new_blocks.insert(id, block);
    }
    function.blocks = new_blocks;
    function.entry_block = prologue_id;

    // Step 5: rewrite yield-block returns to const 0_i64 (pending).
    let yield_block_seq_ids: HashSet<HirBlockId> =
        layout.yield_saves.iter().map(|(bb, _)| *bb).collect();
    // Map seq IDs back to HirIds. Krio's HirCoroCfg uses 0-indexed
    // seq numbers over function.blocks insertion order — but we just
    // inserted the prologue at position 0, shifting everything by 1.
    // Resolve using the BLOCK INDEX at the time krio ran (i.e., the
    // current block list MINUS the prologue we just added).
    let yield_hir_ids: HashSet<HirId> = function
        .blocks
        .keys()
        .skip(1) // skip the prologue we just prepended
        .enumerate()
        .filter_map(|(i, bb_id)| {
            if yield_block_seq_ids.contains(&HirBlockId(i as u32)) {
                Some(*bb_id)
            } else {
                None
            }
        })
        .collect();

    // Pending sentinel constant.
    let zero_i64_id = HirId::new();
    function.values.insert(
        zero_i64_id,
        HirValue {
            id: zero_i64_id,
            ty: HirType::I64,
            kind: HirValueKind::Constant(HirConstant::I64(0)),
            uses: HashSet::new(),
            span: None,
        },
    );

    // Step 6: walk every Return-terminated block.
    //   - if it's a yield block: replace return values with [zero_i64]
    //   - otherwise: insert a cast to i64 of the original return value
    //     and replace `values` with the cast result
    let block_ids: Vec<HirId> = function.blocks.keys().copied().collect();
    for bb_id in block_ids {
        // Need to inspect terminator BEFORE potentially mutating
        // instructions.
        let is_yield = yield_hir_ids.contains(&bb_id);
        let block = function
            .blocks
            .get_mut(&bb_id)
            .expect("block id from snapshot");
        let new_terminator = match &block.terminator {
            HirTerminator::Return { values } => {
                if is_yield {
                    HirTerminator::Return {
                        values: vec![zero_i64_id],
                    }
                } else if values.is_empty() {
                    // Void return → encode as constant 1 (Ready, no value).
                    let one_id = HirId::new();
                    function.values.insert(
                        one_id,
                        HirValue {
                            id: one_id,
                            ty: HirType::I64,
                            kind: HirValueKind::Constant(HirConstant::I64(1)),
                            uses: HashSet::new(),
                            span: None,
                        },
                    );
                    HirTerminator::Return {
                        values: vec![one_id],
                    }
                } else {
                    let original_val = values[0];
                    let original_ty = function
                        .values
                        .get(&original_val)
                        .map(|v| v.ty.clone())
                        .unwrap_or(original_return_ty.clone());
                    let cast_id = HirId::new();
                    function.values.insert(
                        cast_id,
                        HirValue {
                            id: cast_id,
                            ty: HirType::I64,
                            kind: HirValueKind::Instruction,
                            uses: HashSet::new(),
                            span: None,
                        },
                    );
                    let cast_op = pick_to_i64_cast(&original_ty);
                    // re-fetch the block to insert cast (we lost &mut to it)
                    let block_mut = function
                        .blocks
                        .get_mut(&bb_id)
                        .expect("block id from snapshot");
                    block_mut.instructions.push(HirInstruction::Cast {
                        op: cast_op,
                        result: cast_id,
                        ty: HirType::I64,
                        operand: original_val,
                    });
                    HirTerminator::Return {
                        values: vec![cast_id],
                    }
                }
            }
            other => other.clone(),
        };
        function
            .blocks
            .get_mut(&bb_id)
            .expect("block id from snapshot")
            .terminator = new_terminator;
    }

    // Step 7: replace the function's signature with the poll ABI.
    let sm_param = HirParam {
        id: sm_ptr_id,
        name: InternedString::new_global("state_machine"),
        ty: HirType::Ptr(Box::new(HirType::U8)),
        attributes: ParamAttributes::default(),
    };
    function.signature.params = vec![sm_param];
    function.signature.returns = vec![HirType::I64];
    function.signature.is_async = false;

    // Step 8: rename to `{name}$poll`.
    let old_name = function
        .name
        .resolve_global()
        .unwrap_or_else(|| String::from("anon"));
    let new_name = InternedString::new_global(&format!("{}$poll", old_name));
    function.name = new_name;
    new_name
}

/// Pick the right cast op to convert `from_ty` into i64 for the
/// runtime's i64-encoded ready-value channel.
fn pick_to_i64_cast(from_ty: &HirType) -> CastOp {
    match from_ty {
        HirType::Bool | HirType::U8 | HirType::U16 | HirType::U32 | HirType::U64 => CastOp::ZExt,
        HirType::I8 | HirType::I16 | HirType::I32 => CastOp::SExt,
        HirType::I64 => CastOp::Bitcast, // same width — bitcast is a no-op cast
        HirType::F32 | HirType::F64 => CastOp::Bitcast,
        HirType::Ptr(_) | HirType::Opaque(_) | HirType::Function(_) => CastOp::PtrToInt,
        _ => CastOp::Bitcast,
    }
}

/// Pick the right cast op to convert `to_ty` (a parameter type) FROM
/// its source into i64 for storing into the state-machine's
/// 8-byte slot. The runtime stores params in 8-byte slots so we
/// always extend up to i64.
fn pick_param_to_i64_cast(from_ty: &HirType) -> CastOp {
    pick_to_i64_cast(from_ty)
}

/// Generate a Promise-returning entry function that wraps a krio'd
/// poll function. This is the function the runtime sees as `{name}`.
///
/// Body sketch:
/// ```text
///   sm = malloc(num_slots * 8)
///   *(sm + state_slot * 8) = 0_u32         // initial state
///   for (param, slot) in param_slots:
///       *(sm + slot * 8) = cast<i64>(param) // store args
///   poll_ptr = &poll_fn                    // CreateClosure no captures
///   promise = malloc(16)
///   *(promise + 0) = sm
///   *(promise + 8) = poll_ptr
///   return promise
/// ```
pub fn generate_promise_entry(
    original_name: InternedString,
    original_signature: &HirFunctionSignature,
    poll_fn_id: HirId,
    num_slots: u32,
    param_slots: &[(HirId, u32)],
    state_slot: u32,
) -> HirFunction {
    let promise_struct_ty = HirType::Struct(HirStructType {
        name: Some(InternedString::new_global("Promise")),
        fields: vec![
            HirType::Ptr(Box::new(HirType::U8)), // state_machine
            HirType::Function(Box::new(HirFunctionType {
                params: vec![HirType::Ptr(Box::new(HirType::U8))],
                returns: vec![HirType::I64],
                lifetime_params: vec![],
                is_variadic: false,
            })), // poll_fn
        ],
        packed: false,
    });
    let promise_ptr_ty = HirType::Ptr(Box::new(promise_struct_ty.clone()));

    // Entry function signature: keep original params, return *Promise.
    let entry_sig = HirFunctionSignature {
        params: original_signature.params.clone(),
        returns: vec![promise_ptr_ty.clone()],
        type_params: original_signature.type_params.clone(),
        const_params: original_signature.const_params.clone(),
        lifetime_params: original_signature.lifetime_params.clone(),
        is_variadic: false,
        is_async: false, // the entry IS the public face; no further async transform
        effects: original_signature.effects.clone(),
        is_pure: false,
    };

    let mut entry = HirFunction::new(original_name, entry_sig);

    // Re-mint param SSA values so they're owned by THIS function.
    // The new param HirIds match the entry's signature.params.id.
    let mut entry_param_ids: Vec<HirId> = Vec::with_capacity(original_signature.params.len());
    for (i, p) in original_signature.params.iter().enumerate() {
        entry.values.insert(
            p.id,
            HirValue {
                id: p.id,
                ty: p.ty.clone(),
                kind: HirValueKind::Parameter(i as u32),
                uses: HashSet::new(),
                span: None,
            },
        );
        entry_param_ids.push(p.id);
    }

    // Build a single-block body that does the malloc/store dance.
    let entry_block_id = entry.entry_block;
    let mut instructions: Vec<HirInstruction> = Vec::new();

    // sm_size = num_slots * 8
    let sm_size_id = mint_const_i64(&mut entry.values, num_slots as i64 * 8);

    // sm = malloc(sm_size)
    let sm_ptr_id = mint_value(
        &mut entry.values,
        HirType::Ptr(Box::new(HirType::U8)),
        HirValueKind::Instruction,
    );
    instructions.push(HirInstruction::Call {
        result: Some(sm_ptr_id),
        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
        args: vec![sm_size_id],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });

    // Initialize state_id slot to 0 (as i64, matching uniform 8-byte
    // slot layout — the dispatcher reads i64 from the slot).
    let state_init_id = mint_const_i64(&mut entry.values, 0);
    let state_slot_offset_id = mint_const_i64(&mut entry.values, state_slot as i64 * 8);
    let state_field_ptr_id = mint_value(
        &mut entry.values,
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Instruction,
    );
    instructions.push(HirInstruction::Binary {
        result: state_field_ptr_id,
        op: BinaryOp::Add,
        ty: HirType::I64,
        left: sm_ptr_id,
        right: state_slot_offset_id,
    });
    instructions.push(HirInstruction::Store {
        ptr: state_field_ptr_id,
        value: state_init_id,
        align: 8,
        volatile: false,
    });

    // Store each param into its slot, casting up to i64. The
    // param HirIds used here are the ENTRY function's signature
    // params (re-minted as Parameter(i) values above), NOT the
    // body-facing SSA HirIds from `param_slots[i].0` which only
    // existed in the krio'd poll function. We use param_slots only
    // for the slot indices, taking them positionally.
    debug_assert_eq!(
        param_slots.len(),
        original_signature.params.len(),
        "param_slots count must match signature.params count"
    );
    for (i, p) in original_signature.params.iter().enumerate() {
        let slot = param_slots[i].1;
        let param_ty = p.ty.clone();
        let cast_id = mint_value(&mut entry.values, HirType::I64, HirValueKind::Instruction);
        let cast_op = pick_param_to_i64_cast(&param_ty);
        instructions.push(HirInstruction::Cast {
            op: cast_op,
            result: cast_id,
            ty: HirType::I64,
            operand: p.id,
        });
        let slot_offset_id = mint_const_i64(&mut entry.values, slot as i64 * 8);
        let field_ptr_id = mint_value(
            &mut entry.values,
            HirType::Ptr(Box::new(HirType::I64)),
            HirValueKind::Instruction,
        );
        instructions.push(HirInstruction::Binary {
            result: field_ptr_id,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: sm_ptr_id,
            right: slot_offset_id,
        });
        instructions.push(HirInstruction::Store {
            ptr: field_ptr_id,
            value: cast_id,
            align: 8,
            volatile: false,
        });
    }

    // Get the poll fn pointer via CreateClosure (matches the legacy
    // pattern at async_support.rs:1898 — the backend lowers this to a
    // function-address load).
    let poll_fn_ty = HirType::Function(Box::new(HirFunctionType {
        params: vec![HirType::Ptr(Box::new(HirType::U8))],
        returns: vec![HirType::I64],
        lifetime_params: vec![],
        is_variadic: false,
    }));
    let poll_fn_ptr_id = mint_value(&mut entry.values, poll_fn_ty.clone(), HirValueKind::Instruction);
    instructions.push(HirInstruction::CreateClosure {
        result: poll_fn_ptr_id,
        closure_ty: poll_fn_ty,
        function: poll_fn_id,
        captures: vec![],
    });

    // Allocate the 16-byte Promise struct.
    let promise_size_id = mint_const_i64(&mut entry.values, 16);
    let promise_ptr_id = mint_value(&mut entry.values, promise_ptr_ty.clone(), HirValueKind::Instruction);
    instructions.push(HirInstruction::Call {
        result: Some(promise_ptr_id),
        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
        args: vec![promise_size_id],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });

    // Store sm_ptr at Promise+0.
    instructions.push(HirInstruction::Store {
        ptr: promise_ptr_id,
        value: sm_ptr_id,
        align: 8,
        volatile: false,
    });

    // Store poll_fn_ptr at Promise+8.
    let promise_offset_8_id = mint_const_i64(&mut entry.values, 8);
    let poll_field_ptr_id = mint_value(
        &mut entry.values,
        HirType::Ptr(Box::new(HirType::U8)),
        HirValueKind::Instruction,
    );
    instructions.push(HirInstruction::Binary {
        result: poll_field_ptr_id,
        op: BinaryOp::Add,
        ty: HirType::I64,
        left: promise_ptr_id,
        right: promise_offset_8_id,
    });
    instructions.push(HirInstruction::Store {
        ptr: poll_field_ptr_id,
        value: poll_fn_ptr_id,
        align: 8,
        volatile: false,
    });

    // Populate the entry block.
    let entry_block = entry
        .blocks
        .get_mut(&entry_block_id)
        .expect("HirFunction::new always creates an entry block");
    entry_block.instructions = instructions;
    entry_block.terminator = HirTerminator::Return {
        values: vec![promise_ptr_id],
    };

    let _ = entry_param_ids; // prevent unused warning if we extend later

    entry
}

fn mint_value(
    values: &mut IndexMap<HirId, HirValue>,
    ty: HirType,
    kind: HirValueKind,
) -> HirId {
    let id = HirId::new();
    values.insert(
        id,
        HirValue {
            id,
            ty,
            kind,
            uses: HashSet::new(),
            span: None,
        },
    );
    id
}

fn mint_const_i64(values: &mut IndexMap<HirId, HirValue>, val: i64) -> HirId {
    mint_value(
        values,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(val)),
    )
}
