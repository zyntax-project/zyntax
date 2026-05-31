//! HIR-level loop-invariant code motion.
//!
//! For each natural loop produced by `analysis::loops`, hoists pure
//! instructions whose operands are all defined outside the loop into
//! the loop's preheader block. The preheader is the unique
//! outside-the-loop predecessor of the header; we use it as-is
//! rather than synthesising one. Loops that don't have a unique
//! outside-the-loop predecessor are skipped — a separate "ensure
//! preheader" pass would be cheap to add later and is the only thing
//! gating us from broader coverage.
//!
//! ## What we hoist
//!
//! Pure, non-trapping instructions whose operands are *loop-
//! invariant* — defined outside the loop or in a block that
//! dominates the loop header:
//!
//!   * `Binary`           (add / sub / mul / bitwise / shifts /
//!                         comparisons, excluding `Div` and `Rem`
//!                         to avoid changing trap semantics)
//!   * `Unary`            (Neg / Not / FNeg)
//!   * `Cast`
//!   * `GetElementPtr`    — pure address arithmetic
//!   * `ExtractValue`     — pure aggregate field read
//!   * `Select`           — pure ternary
//!
//! ## What we don't touch
//!
//!   * `Load`   — moving across a hidden Store changes semantics
//!   * `Store` / atomics / fences
//!   * `Call` / `IndirectCall` — arbitrary side effects, even
//!                                marked-pure callees can observe
//!                                globals we don't model
//!   * `Alloca` — each iteration's alloca is distinct
//!   * Integer `Div` / `Rem` — moving them changes whether a
//!                              div-by-zero trap fires at all
//!   * `CreateClosure` / `AsyncSaveSlot` / `AsyncLoadSlot` —
//!                              identity / frame state
//!   * `Phi` — control-flow-dependent by definition
//!
//! ## Algorithm sketch
//!
//! 1. Build dominator tree + loop forest.
//! 2. For each loop (innermost first — propagates inner hoists to
//!    outer-loop visibility), determine the preheader: the unique
//!    predecessor of the header outside the loop body. Skip if
//!    absent or non-unique.
//! 3. Seed an *invariant set* with everything defined outside the
//!    loop body (any value whose defining block isn't in the loop).
//! 4. Iterate over instructions in the loop body. For each one
//!    whose result is not yet in the set and whose operands are all
//!    in the set and which is in our "safe to hoist" list, mark as
//!    invariant and queue for hoisting.
//! 5. Loop until no new invariants found. Each pass may unlock new
//!    candidates (transitive invariance).
//! 6. Move hoisted instructions to the END of the preheader's
//!    instruction list, immediately before its terminator. Order
//!    among hoisted instructions preserves their original block-
//!    visit order so operand definitions still come before uses.
//!
//! The hoist is a literal block-to-block move; it preserves SSA
//! because the result HirId is unchanged — every use that was
//! inside the loop now reads from a value defined in the preheader
//! (which dominates the loop body).

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{BinaryOp, HirFunction, HirId, HirInstruction};
use std::collections::HashSet;

/// Stats surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LicmStats {
    /// Number of instructions hoisted to a preheader.
    pub hoisted: usize,
    /// Number of loops considered (whether or not anything hoisted).
    pub loops_visited: usize,
    /// Number of loops skipped because no unique preheader exists.
    pub loops_skipped_no_preheader: usize,
}

/// Run LICM over `func`.
pub fn run(func: &mut HirFunction) -> LicmStats {
    let dt = DominatorTree::new(func);
    let lf = LoopForest::detect(func, &dt);
    if lf.loops().is_empty() {
        return LicmStats::default();
    }

    let mut stats = LicmStats::default();

    // Innermost-first ordering — `LoopForest::loops()` already
    // returns smaller-body-first. Processing inner loops before
    // outer ones means an inner-loop hoist makes its result visible
    // to the outer-loop invariance check on the same run.
    for lp in lf.loops() {
        stats.loops_visited += 1;
        let preheader = match unique_outside_predecessor(func, lp) {
            Some(p) => p,
            None => {
                stats.loops_skipped_no_preheader += 1;
                continue;
            }
        };

        stats.hoisted += hoist_loop(func, lp, preheader);
    }

    stats
}

/// Module-level entry — runs LICM on every function in `module`.
pub fn run_module(module: &mut crate::hir::HirModule) -> LicmStats {
    let mut total = LicmStats::default();
    for func in module.functions.values_mut() {
        let s = run(func);
        total.hoisted += s.hoisted;
        total.loops_visited += s.loops_visited;
        total.loops_skipped_no_preheader += s.loops_skipped_no_preheader;
    }
    total
}

/// Find the header's unique outside-the-loop predecessor. Returns
/// `None` when there is zero or more than one — we don't synthesise
/// a preheader in this pass.
fn unique_outside_predecessor(func: &HirFunction, lp: &NaturalLoop) -> Option<HirId> {
    let header = func.blocks.get(&lp.header)?;
    let outside: Vec<HirId> = header
        .predecessors
        .iter()
        .copied()
        .filter(|p| !lp.body.contains(p))
        .collect();
    if outside.len() == 1 {
        Some(outside[0])
    } else {
        None
    }
}

/// Hoist invariant instructions from `lp.body` into `preheader`.
/// Returns the number of instructions hoisted.
fn hoist_loop(func: &mut HirFunction, lp: &NaturalLoop, preheader: HirId) -> usize {
    // Seed invariant set with every value defined outside the
    // loop body. By "defined outside" we mean: produced by an
    // instruction whose containing block isn't in `lp.body`, or
    // produced as a Parameter / Global / Constant (none of those
    // live in any block).
    let mut invariant: HashSet<HirId> = HashSet::new();

    // Walk every value in the function; if its kind isn't an
    // Instruction it lives outside any block by construction
    // (Param / Constant / Global / Undef). Mark those invariant
    // upfront.
    for (id, val) in &func.values {
        match val.kind {
            crate::hir::HirValueKind::Constant(_)
            | crate::hir::HirValueKind::Parameter(_)
            | crate::hir::HirValueKind::Global(_)
            | crate::hir::HirValueKind::Undef => {
                invariant.insert(*id);
            }
            crate::hir::HirValueKind::Instruction => {
                // Defer — see if its defining block is outside the
                // loop. This is O(blocks * insts); the body sizes we
                // see don't make it interesting.
            }
        }
    }

    // For Instruction-kind values, also mark them invariant if the
    // block that produces them isn't part of the loop body.
    let mut value_block: indexmap::IndexMap<HirId, HirId> = indexmap::IndexMap::new();
    for (block_id, block) in &func.blocks {
        for inst in &block.instructions {
            if let Some(res) = instruction_result(inst) {
                value_block.insert(res, *block_id);
            }
        }
        // Phi results are defined at the head of their block;
        // record so the operand-check can see "header phi" as
        // loop-internal even though it's a Phi not an Instruction.
        for phi in &block.phis {
            value_block.insert(phi.result, *block_id);
        }
    }
    for (val_id, block_id) in &value_block {
        if !lp.body.contains(block_id) {
            invariant.insert(*val_id);
        }
    }

    // Iterate-to-fixed-point: each pass may unlock new candidates
    // because a hoisted instruction's result becomes invariant for
    // the next pass.
    let mut total_hoisted = 0;
    for _ in 0..32 {
        let mut to_hoist: Vec<(HirId, HirInstruction)> = Vec::new();

        // Collect candidates from every block in the loop body.
        // Preserve original (block-iteration, position) order so
        // when we re-insert into the preheader, operand
        // definitions still precede their uses.
        for &block_id in &lp.body {
            let Some(block) = func.blocks.get(&block_id) else {
                continue;
            };
            for inst in &block.instructions {
                let Some(res) = instruction_result(inst) else {
                    continue;
                };
                if invariant.contains(&res) {
                    continue; // already invariant — was hoisted on
                              // a prior iteration or wasn't a body
                              // instruction.
                }
                if !is_safe_to_hoist(inst) {
                    continue;
                }
                if !operands_all_invariant(inst, &invariant) {
                    continue;
                }
                invariant.insert(res);
                to_hoist.push((block_id, inst.clone()));
            }
        }

        if to_hoist.is_empty() {
            break;
        }

        // Remove hoisted instructions from their original blocks.
        let hoisted_ids: HashSet<HirId> = to_hoist
            .iter()
            .filter_map(|(_, inst)| instruction_result(inst))
            .collect();
        for (block_id, _) in &to_hoist {
            if let Some(block) = func.blocks.get_mut(block_id) {
                block
                    .instructions
                    .retain(|inst| match instruction_result(inst) {
                        Some(r) => !hoisted_ids.contains(&r),
                        None => true,
                    });
            }
        }

        // Insert into the preheader, before its terminator (which
        // for HIR is stored separately from `instructions`, so we
        // just append).
        if let Some(ph) = func.blocks.get_mut(&preheader) {
            for (_, inst) in &to_hoist {
                ph.instructions.push(inst.clone());
            }
        }

        total_hoisted += to_hoist.len();
    }

    total_hoisted
}

/// Is `inst` shape we consider safe to relocate? Excludes ops with
/// side effects or non-deterministic trap semantics.
fn is_safe_to_hoist(inst: &HirInstruction) -> bool {
    match inst {
        HirInstruction::Binary { op, .. } => match op {
            // Integer Div / Rem can trap on zero — hoisting them
            // out of a loop changes whether the trap fires per
            // iteration, so we leave them in place.
            BinaryOp::Div | BinaryOp::Rem => false,
            _ => true,
        },
        HirInstruction::Unary { .. } | HirInstruction::Cast { .. } => true,
        HirInstruction::GetElementPtr { .. } | HirInstruction::ExtractValue { .. } => true,
        HirInstruction::Select { .. } => true,
        _ => false,
    }
}

/// Are every operand referenced by `inst` already in `invariant`?
fn operands_all_invariant(inst: &HirInstruction, invariant: &HashSet<HirId>) -> bool {
    let mut all_in = true;
    let check = |id: HirId| invariant.contains(&id);
    match inst {
        HirInstruction::Binary { left, right, .. } => {
            all_in = check(*left) && check(*right);
        }
        HirInstruction::Unary { operand, .. } => all_in = check(*operand),
        HirInstruction::Cast { operand, .. } => all_in = check(*operand),
        HirInstruction::GetElementPtr { ptr, indices, .. } => {
            all_in = check(*ptr) && indices.iter().all(|i| check(*i));
        }
        HirInstruction::ExtractValue { aggregate, .. } => all_in = check(*aggregate),
        HirInstruction::Select {
            condition,
            true_val,
            false_val,
            ..
        } => {
            all_in = check(*condition) && check(*true_val) && check(*false_val);
        }
        _ => all_in = false,
    }
    all_in
}

fn instruction_result(inst: &HirInstruction) -> Option<HirId> {
    match inst {
        HirInstruction::Binary { result, .. } => Some(*result),
        HirInstruction::Unary { result, .. } => Some(*result),
        HirInstruction::Cast { result, .. } => Some(*result),
        HirInstruction::GetElementPtr { result, .. } => Some(*result),
        HirInstruction::ExtractValue { result, .. } => Some(*result),
        HirInstruction::InsertValue { result, .. } => Some(*result),
        HirInstruction::Load { result, .. } => Some(*result),
        HirInstruction::Alloca { result, .. } => Some(*result),
        HirInstruction::Call { result, .. } => *result,
        HirInstruction::IndirectCall { result, .. } => *result,
        HirInstruction::Select { result, .. } => Some(*result),
        HirInstruction::Atomic { result, .. } => Some(*result),
        _ => None,
    }
}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        BinaryOp, HirBlock, HirFunctionSignature, HirTerminator, HirType, HirValue, HirValueKind,
    };
    use zyntax_typed_ast::InternedString;

    fn empty_sig(ret: HirType) -> HirFunctionSignature {
        HirFunctionSignature {
            params: vec![],
            returns: vec![ret],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        }
    }

    fn mk_func() -> (HirFunction, HirId, HirId, HirId, HirId) {
        // Build the canonical while-loop shape:
        //   entry → header → body → header (back edge)
        //              ↓
        //             exit
        let mut f = HirFunction::new(InternedString::new_global("t"), empty_sig(HirType::I32));
        let entry = HirId::new();
        let header = HirId::new();
        let body = HirId::new();
        let exit = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        for id in [entry, header, body, exit] {
            f.blocks.insert(id, HirBlock::new(id));
        }
        f.blocks.get_mut(&entry).unwrap().successors = vec![header];
        f.blocks.get_mut(&header).unwrap().predecessors = vec![entry, body];
        f.blocks.get_mut(&header).unwrap().successors = vec![body, exit];
        f.blocks.get_mut(&body).unwrap().predecessors = vec![header];
        f.blocks.get_mut(&body).unwrap().successors = vec![header];
        f.blocks.get_mut(&exit).unwrap().predecessors = vec![header];

        let cond_id = HirId::new();
        f.values.insert(
            cond_id,
            HirValue {
                id: cond_id,
                ty: HirType::Bool,
                kind: HirValueKind::Constant(crate::hir::HirConstant::Bool(true)),
                uses: Default::default(),
                span: None,
            },
        );
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: header };
        f.blocks.get_mut(&header).unwrap().terminator = HirTerminator::CondBranch {
            condition: cond_id,
            true_target: body,
            false_target: exit,
        };
        f.blocks.get_mut(&body).unwrap().terminator = HirTerminator::Branch { target: header };
        f.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Return { values: vec![] };

        (f, entry, header, body, exit)
    }

    fn add_param(f: &mut HirFunction, ty: HirType, idx: u32) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Parameter(idx),
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn add_inst(f: &mut HirFunction, ty: HirType) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    #[test]
    fn hoists_invariant_add_into_preheader() {
        // body: r = a + b   ← a and b are params, both invariant
        let (mut f, entry, _header, body, _exit) = mk_func();
        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let r = add_inst(&mut f, HirType::I32);
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r,
                ty: HirType::I32,
                left: a,
                right: b,
            });

        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 1);
        assert!(f.blocks[&body].instructions.is_empty());
        assert_eq!(f.blocks[&entry].instructions.len(), 1);
    }

    #[test]
    fn does_not_hoist_when_operand_is_loop_internal() {
        // body: phi i = ...
        //       r   = i + 1    ← i is body-local (phi), not invariant
        let (mut f, _entry, header, body, _exit) = mk_func();
        let one = HirId::new();
        f.values.insert(
            one,
            HirValue {
                id: one,
                ty: HirType::I32,
                kind: HirValueKind::Constant(crate::hir::HirConstant::I32(1)),
                uses: Default::default(),
                span: None,
            },
        );
        // Put a Phi result in the header so it's defined in the
        // loop body — emulating the loop induction variable.
        let i = add_inst(&mut f, HirType::I32);
        f.blocks
            .get_mut(&header)
            .unwrap()
            .phis
            .push(crate::hir::HirPhi {
                result: i,
                ty: HirType::I32,
                incoming: vec![],
            });
        let r = add_inst(&mut f, HirType::I32);
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r,
                ty: HirType::I32,
                left: i,
                right: one,
            });

        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 0, "i depends on phi — must stay in loop");
    }

    #[test]
    fn does_not_hoist_div_even_if_invariant() {
        // body: r = a / b   ← a and b invariant, but Div may trap
        // and hoisting it changes whether trap fires per iteration.
        let (mut f, _entry, _header, body, _exit) = mk_func();
        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let r = add_inst(&mut f, HirType::I32);
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Div,
                result: r,
                ty: HirType::I32,
                left: a,
                right: b,
            });

        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 0);
        assert_eq!(f.blocks[&body].instructions.len(), 1);
    }

    #[test]
    fn chained_invariants_hoist_in_order() {
        // body: r1 = a + b
        //       r2 = r1 * c   ← becomes invariant after r1 hoist
        // Both should end up in the preheader, with r1 before r2.
        let (mut f, entry, _header, body, _exit) = mk_func();
        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let c = add_param(&mut f, HirType::I32, 2);
        let r1 = add_inst(&mut f, HirType::I32);
        let r2 = add_inst(&mut f, HirType::I32);
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r1,
                ty: HirType::I32,
                left: a,
                right: b,
            });
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Mul,
                result: r2,
                ty: HirType::I32,
                left: r1,
                right: c,
            });

        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 2);
        assert!(f.blocks[&body].instructions.is_empty());
        let entry_insts = &f.blocks[&entry].instructions;
        assert_eq!(entry_insts.len(), 2);
        // r1's add comes before r2's mul.
        let p_r1 = entry_insts
            .iter()
            .position(|i| matches!(i, HirInstruction::Binary { result, .. } if *result == r1))
            .unwrap();
        let p_r2 = entry_insts
            .iter()
            .position(|i| matches!(i, HirInstruction::Binary { result, .. } if *result == r2))
            .unwrap();
        assert!(p_r1 < p_r2, "operand def must precede use after hoisting");
    }

    #[test]
    fn function_with_no_loops_is_a_noop() {
        let mut f = HirFunction::new(InternedString::new_global("nop"), empty_sig(HirType::I32));
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![] };
        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 0);
        assert_eq!(stats.loops_visited, 0);
    }
}
