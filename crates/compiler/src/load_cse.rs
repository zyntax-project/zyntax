//! Intra-block redundant-Load elimination with simple alias barriers.
//!
//! Plain `cse::eliminate` skips `Load` because Loads can alias with
//! `Store`s through a memory layer we don't model — two Loads with
//! the same pointer expression aren't equivalent if a Store happened
//! between them. This module fills the gap with the simplest correct
//! alias check we can ship:
//!
//!   * Walk each block top-to-bottom.
//!   * Track `Load { ptr, result }` entries in a per-block "available
//!     loads" map keyed by the pointer's canonical HirId (chasing
//!     substitutions through any prior CSE).
//!   * On a *memory-killing* instruction (`Store`, `Call`,
//!     `IndirectCall`, `Atomic`, `Fence`), clear the map — we
//!     conservatively assume anything in memory could have changed.
//!   * On a redundant `Load` (its pointer is already in the map),
//!     record a substitution from the new load result to the
//!     previously-seen load result.
//!
//! That's enough for the most common win we lose without it:
//! repeated struct field access (`x.field`, `x.field`, …) where the
//! intervening code doesn't write to anything aliasing. Cross-block
//! Load CSE would need a dominator-walk-with-barrier-summary scheme;
//! that's the natural follow-up but adds real complexity. Per-block
//! coverage is the high-value cheap version.
//!
//! After collecting substitutions we sweep the function with the
//! same machinery `cse::apply_substitutions` uses (rewrite operand
//! references, drop the now-orphaned defining Loads).

use crate::cse;
use crate::hir::{HirFunction, HirId, HirInstruction, HirModule};
use std::collections::HashMap;

/// Public stats — same shape as `CseStats` so callers can compose.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LoadCseStats {
    /// Number of redundant Loads removed from blocks.
    pub eliminated: usize,
}

/// Run intra-block Load CSE on `func`.
pub fn run(func: &mut HirFunction) -> LoadCseStats {
    let mut substitutions: HashMap<HirId, HirId> = HashMap::new();

    for block in func.blocks.values() {
        let mut available: HashMap<HirId, HirId> = HashMap::new();
        for inst in &block.instructions {
            match inst {
                HirInstruction::Load { result, ptr, .. } => {
                    let canonical_ptr = chase(*ptr, &substitutions);
                    match available.get(&canonical_ptr) {
                        Some(&prev_result) => {
                            substitutions.insert(*result, prev_result);
                        }
                        None => {
                            available.insert(canonical_ptr, *result);
                        }
                    }
                }
                // Memory-killing ops invalidate everything we
                // believe about memory contents. Conservative
                // clearing matches what most early alias-aware CSE
                // implementations do before they grow up.
                //
                // `AsyncSaveSlot` writes to the SM frame so a Load
                // before it and a Load after it of an
                // overlapping-frame pointer can legitimately read
                // different values. Treat it as a memory barrier
                // (otherwise the krio-emitted state-machine poll-fn
                // re-uses a stale state Load between yields and
                // the SM keeps re-parking instead of advancing).
                //
                // `CreateClosure` may capture by reference to a
                // mutable environment; conservatively treat as a
                // barrier.
                HirInstruction::Store { .. }
                | HirInstruction::Call { .. }
                | HirInstruction::IndirectCall { .. }
                | HirInstruction::Atomic { .. }
                | HirInstruction::Fence { .. }
                | HirInstruction::AsyncSaveSlot { .. }
                | HirInstruction::CreateClosure { .. } => {
                    available.clear();
                }
                _ => {}
            }
        }
    }

    if substitutions.is_empty() {
        return LoadCseStats::default();
    }

    let _rewrites = cse::apply_substitutions_public(func, &substitutions);
    let eliminated = cse::remove_redundant_instructions_public(func, &substitutions);
    LoadCseStats { eliminated }
}

/// Module-level entry.
pub fn run_module(module: &mut HirModule) -> LoadCseStats {
    let mut total = LoadCseStats::default();
    for func in module.functions.values_mut() {
        let s = run(func);
        total.eliminated += s.eliminated;
    }
    total
}

fn chase(mut id: HirId, subs: &HashMap<HirId, HirId>) -> HirId {
    let mut seen = 0;
    while let Some(&next) = subs.get(&id) {
        if next == id || seen > 64 {
            break;
        }
        id = next;
        seen += 1;
    }
    id
}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        BinaryOp, HirBlock, HirFunctionSignature, HirTerminator, HirType, HirValue, HirValueKind,
    };
    use std::collections::HashSet;
    use zyntax_typed_ast::InternedString;

    fn sig() -> HirFunctionSignature {
        HirFunctionSignature {
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
        }
    }

    fn mk_func() -> (HirFunction, HirId) {
        let mut f = HirFunction::new(InternedString::new_global("t"), sig());
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        (f, entry)
    }

    fn add_param(f: &mut HirFunction, ty: HirType, idx: u32) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Parameter(idx),
                uses: HashSet::new(),
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
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }

    fn push(f: &mut HirFunction, entry: HirId, inst: HirInstruction) {
        f.blocks.get_mut(&entry).unwrap().instructions.push(inst);
    }

    #[test]
    fn two_loads_same_ptr_collapse_to_one() {
        let (mut f, entry) = mk_func();
        let p = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 0);
        let l1 = add_inst(&mut f, HirType::I64);
        let l2 = add_inst(&mut f, HirType::I64);
        let sum = add_inst(&mut f, HirType::I64);
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l1,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l2,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: sum,
                ty: HirType::I64,
                left: l1,
                right: l2,
            },
        );
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![sum] };

        let stats = run(&mut f);
        assert_eq!(stats.eliminated, 1);
        // Both operands of the Add should now point at l1.
        let blk = &f.blocks[&entry];
        let add = blk
            .instructions
            .iter()
            .find_map(|i| {
                if let HirInstruction::Binary {
                    op: BinaryOp::Add,
                    left,
                    right,
                    ..
                } = i
                {
                    Some((*left, *right))
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(add, (l1, l1));
        // Only one Load left.
        assert_eq!(
            blk.instructions
                .iter()
                .filter(|i| matches!(i, HirInstruction::Load { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn intervening_store_blocks_collapse() {
        let (mut f, entry) = mk_func();
        let p = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 0);
        let v = add_param(&mut f, HirType::I64, 1);
        let l1 = add_inst(&mut f, HirType::I64);
        let l2 = add_inst(&mut f, HirType::I64);
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l1,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Store {
                value: v,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l2,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return {
            values: vec![l1, l2],
        };

        let stats = run(&mut f);
        assert_eq!(stats.eliminated, 0, "Store between Loads must block CSE");
        assert_eq!(f.blocks[&entry].instructions.len(), 3);
    }

    #[test]
    fn intervening_call_blocks_collapse() {
        let (mut f, entry) = mk_func();
        let p = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 0);
        let l1 = add_inst(&mut f, HirType::I64);
        let l2 = add_inst(&mut f, HirType::I64);
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l1,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Call {
                result: None,
                callee: crate::hir::HirCallable::Symbol("opaque".to_string()),
                args: vec![],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l2,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        let stats = run(&mut f);
        assert_eq!(stats.eliminated, 0, "Call between Loads must block CSE");
    }

    #[test]
    fn different_pointers_do_not_alias_each_other() {
        let (mut f, entry) = mk_func();
        let p1 = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 0);
        let p2 = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 1);
        let l1 = add_inst(&mut f, HirType::I64);
        let l2 = add_inst(&mut f, HirType::I64);
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l1,
                ty: HirType::I64,
                ptr: p1,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l2,
                ty: HirType::I64,
                ptr: p2,
                align: 8,
                volatile: false,
            },
        );
        let stats = run(&mut f);
        assert_eq!(stats.eliminated, 0, "different ptrs are different loads");
    }
}
