//! Remove phis whose result nothing reads.
//!
//! SSA construction places a phi for every variable written anywhere in a
//! loop, without asking whether the previous iteration's value can ever be
//! observed. A variable that is *declared inside* the loop body — `let av
//! = load(...)` — is written before it is read on every path, so the phi's
//! result has no uses at all. It is pure bookkeeping.
//!
//! Those phis are not free. Each one becomes a loop-carried block
//! parameter in the backend, so the value stays live across the back edge
//! and occupies a register for the whole loop. In a vector kernel that is
//! a 128-bit register held hostage per spurious phi, and the register
//! allocator pays for it by spilling something real — typically the
//! accumulator, which then round-trips through the stack every iteration.
//!
//! Dropping them is safe and cheap: HIR keeps phis in `block.phis` and
//! branch terminators carry no edge arguments, so removing a phi needs no
//! terminator rewriting. The pass iterates to a fixed point because one
//! phi's operands may be the only reader of another.
//!
//! Turned off with `ZYNTAX_DISABLE_PHI_PRUNE=1`.

use crate::hir::{IdHashMap as HashMap, IdHashSet as HashSet};

use crate::hir::{HirFunction, HirId, HirInstruction, HirModule, HirTerminator};

/// Counters surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PhiPruneStats {
    /// Phis removed.
    pub removed: usize,
    /// Fixed-point rounds taken.
    pub rounds: usize,
}

/// Run the pass over every function in `module`.
pub fn run_module(module: &mut HirModule) -> PhiPruneStats {
    if std::env::var("ZYNTAX_DISABLE_PHI_PRUNE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return PhiPruneStats::default();
    }
    let mut stats = PhiPruneStats::default();
    for func in module.functions_to_optimize() {
        let s = run_function(func);
        stats.removed += s.removed;
        stats.rounds = stats.rounds.max(s.rounds);
    }
    stats
}

/// Run the pass over a single function.
pub fn run_function(func: &mut HirFunction) -> PhiPruneStats {
    let mut stats = PhiPruneStats::default();
    func.rebuild_cfg_edges();
    prune_stale_incomings(func);
    // Removing a phi can make a trivial phi collapsible and vice versa,
    // so iterate. The bound is generous; convergence is typically one or
    // two rounds.
    for _ in 0..16 {
        stats.rounds += 1;
        let live = live_phis(func);
        let mut removed_this_round = 0;
        for block in func.blocks.values_mut() {
            let before = block.phis.len();
            block.phis.retain(|p| live.contains(&p.result));
            removed_this_round += before - block.phis.len();
        }
        removed_this_round += collapse_trivial_phis(func);
        stats.removed += removed_this_round;
        if removed_this_round == 0 {
            break;
        }
    }
    // The results and the undefined incomings of the phis removed.
    func.sweep_unreferenced_values();
    stats
}

/// CFG simplification may delete or merge a predecessor without removing
/// its old phi arm. Edges are rebuilt from terminators and those stale
/// arms discarded before deciding whether a phi is live or trivial.
fn prune_stale_incomings(func: &mut HirFunction) {
    let mut reachable = HashSet::default();
    let mut work = vec![func.entry_block];
    while let Some(block_id) = work.pop() {
        if !reachable.insert(block_id) {
            continue;
        }
        if let Some(block) = func.blocks.get(&block_id) {
            work.extend(block.successors.iter().copied());
        }
    }
    for block in func.blocks.values_mut() {
        if block.predecessors.is_empty() || block.phis.is_empty() {
            continue;
        }
        let predecessors = &block.predecessors;
        for phi in &mut block.phis {
            phi.incoming.retain(|(_, predecessor)| {
                predecessors.contains(predecessor) && reachable.contains(predecessor)
            });
        }
    }
}

/// Replace a phi that can only ever be one value with that value.
///
/// A variable that is live across a loop but never reassigned in it still
/// gets a phi, and that phi's only incomings are its own result and the
/// value it started as. It is not dead, so the removal above leaves it,
/// and it reads as a definition in the loop header. Anything asking where
/// a value comes from is told the header, which is how a buffer that a
/// function allocated itself stops looking loop-invariant and a loop over
/// it stops being recognised as one shape.
fn collapse_trivial_phis(func: &mut HirFunction) -> usize {
    let mut replacements: HashMap<HirId, HirId> = HashMap::default();
    for block in func.blocks.values() {
        for phi in &block.phis {
            let mut distinct = phi
                .incoming
                .iter()
                .map(|(v, _)| *v)
                .filter(|v| *v != phi.result);
            let Some(first) = distinct.next() else {
                continue;
            };
            if distinct.all(|v| v == first) {
                replacements.insert(phi.result, first);
            }
        }
    }
    if replacements.is_empty() {
        return 0;
    }
    // A phi may name another phi being collapsed in the same round, so
    // follow each chain to what it finally resolves to.
    let resolve = |mut v: HirId| -> HirId {
        for _ in 0..16 {
            match replacements.get(&v) {
                Some(&next) if next != v => v = next,
                _ => break,
            }
        }
        v
    };
    let final_map: crate::hir::IdMap<HirId, HirId> = replacements
        .keys()
        .map(|&k| (k, resolve(k)))
        .filter(|(k, v)| k != v)
        .collect();
    if final_map.is_empty() {
        return 0;
    }
    let removed = final_map.len();
    for block in func.blocks.values_mut() {
        block.phis.retain(|p| !final_map.contains_key(&p.result));
        for phi in &mut block.phis {
            for (value, _) in &mut phi.incoming {
                if let Some(&to) = final_map.get(value) {
                    *value = to;
                }
            }
        }
        for inst in &mut block.instructions {
            inst.replace_uses(&final_map);
        }
        block.terminator.replace_uses(&final_map);
    }
    removed
}

/// Which phi results a real instruction can observe.
///
/// Asking "is this phi's result mentioned anywhere" is the wrong
/// question, because phis can hold each other up. Nested loops leave
/// exactly that: SSA construction gives the outer header a phi for every
/// variable the inner loop writes, the inner header takes it as its
/// incoming from outside, and the inner one reinitialises before reading,
/// so the two name each other and nothing else names either. Both look
/// used, so neither is ever dropped, and a three-deep loop nest keeps a
/// register per level for values no instruction reads.
///
/// So it is answered the other way round. Nothing is live until an
/// instruction or a terminator reads it; a phi reached only through other
/// phis becomes live only once one of those is. A cycle no real reader
/// enters is never marked, and the whole cycle is dropped together. That
/// is also why the set is computed in full before anything is removed:
/// taking one member out first would leave the rest naming a value that
/// no longer exists.
fn live_phis(func: &HirFunction) -> HashSet<HirId> {
    // Where each phi result is defined, and what it names as incomings.
    let mut incomings: HashMap<HirId, Vec<HirId>> = HashMap::default();
    for block in func.blocks.values() {
        for phi in &block.phis {
            incomings.insert(phi.result, phi.incoming.iter().map(|(v, _)| *v).collect());
        }
    }

    // Seed with the phis an instruction or terminator reads directly.
    let mut live: HashSet<HirId> = HashSet::default();
    let mut work: Vec<HirId> = Vec::new();
    let mut seed = |v: HirId| {
        if incomings.contains_key(&v) && live.insert(v) {
            work.push(v);
        }
    };
    for block in func.blocks.values() {
        for inst in &block.instructions {
            inst.for_each_operand(&mut seed);
        }
        block.terminator.for_each_operand(&mut seed);
    }

    // A live phi's incomings are observable through it, so any phi among
    // them is live too.
    while let Some(v) = work.pop() {
        let Some(sources) = incomings.get(&v) else {
            continue;
        };
        for &src in sources {
            if incomings.contains_key(&src) && live.insert(src) {
                work.push(src);
            }
        }
    }
    live
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        BinaryOp, HirBlock, HirFunctionSignature, HirPhi, HirType, HirValue, HirValueKind,
    };
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

    fn mk() -> (HirFunction, HirId) {
        let mut f = HirFunction::new(InternedString::new_global("t"), sig());
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        (f, entry)
    }

    fn val(f: &mut HirFunction) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty: HirType::I64,
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    /// A phi nothing reads is removed.
    #[test]
    fn unused_phi_is_removed() {
        let (mut f, entry) = mk();
        let incoming = val(&mut f);
        let dead = val(&mut f);
        f.blocks.get_mut(&entry).unwrap().phis.push(HirPhi {
            result: dead,
            ty: HirType::I64,
            incoming: vec![(incoming, entry)],
        });
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![] };

        let stats = run_function(&mut f);
        assert_eq!(stats.removed, 1);
        assert!(f.blocks[&entry].phis.is_empty());
    }

    /// A single-incoming phi is replaced by its incoming value.
    #[test]
    fn used_single_incoming_phi_is_collapsed() {
        let (mut f, entry) = mk();
        let incoming = val(&mut f);
        let live = val(&mut f);
        f.blocks.get_mut(&entry).unwrap().phis.push(HirPhi {
            result: live,
            ty: HirType::I64,
            incoming: vec![(incoming, entry)],
        });
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![live] };

        let stats = run_function(&mut f);
        assert_eq!(stats.removed, 1);
        assert!(f.blocks[&entry].phis.is_empty());
        assert!(matches!(
            &f.blocks[&entry].terminator,
            HirTerminator::Return { values } if values == &vec![incoming]
        ));
    }

    /// A phi read only by another dead phi goes too, and the whole
    /// chain goes at once. Liveness is decided before anything is
    /// removed, so the length of the chain costs no extra rounds.
    #[test]
    fn a_chain_of_unread_phis_goes_in_one_sweep() {
        let (mut f, entry) = mk();
        let seed = val(&mut f);
        let first = val(&mut f);
        let second = val(&mut f);
        let blk = f.blocks.get_mut(&entry).unwrap();
        blk.phis.push(HirPhi {
            result: first,
            ty: HirType::I64,
            incoming: vec![(seed, entry)],
        });
        // `second` is only read by `first`, which is itself unused.
        blk.phis.push(HirPhi {
            result: second,
            ty: HirType::I64,
            incoming: vec![(first, entry)],
        });
        blk.terminator = HirTerminator::Return { values: vec![] };

        let stats = run_function(&mut f);
        assert_eq!(stats.removed, 2, "both phis should go");
        assert!(f.blocks[&entry].phis.is_empty());
    }

    /// A single-incoming phi feeding an instruction is substituted.
    #[test]
    fn single_incoming_phi_used_by_instruction_is_collapsed() {
        let (mut f, entry) = mk();
        let incoming = val(&mut f);
        let live = val(&mut f);
        let sum = val(&mut f);
        let blk = f.blocks.get_mut(&entry).unwrap();
        blk.phis.push(HirPhi {
            result: live,
            ty: HirType::I64,
            incoming: vec![(incoming, entry)],
        });
        blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: sum,
            ty: HirType::I64,
            left: live,
            right: incoming,
        });
        blk.terminator = HirTerminator::Return { values: vec![sum] };

        assert_eq!(run_function(&mut f).removed, 1);
        assert!(f.blocks[&entry].phis.is_empty());
        assert!(matches!(
            &f.blocks[&entry].instructions[0],
            HirInstruction::Binary { left, .. } if *left == incoming
        ));
    }

    #[test]
    fn drops_phi_arm_from_deleted_predecessor() {
        let (mut f, entry) = mk();
        let join = HirId::new();
        let stale = HirId::new();
        f.blocks.insert(join, HirBlock::new(join));
        f.blocks.insert(stale, HirBlock::new(stale));
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: join };
        f.blocks.get_mut(&stale).unwrap().terminator = HirTerminator::Branch { target: join };
        let live_value = val(&mut f);
        let stale_value = val(&mut f);
        let phi_result = val(&mut f);
        let join_block = f.blocks.get_mut(&join).unwrap();
        join_block.phis.push(HirPhi {
            result: phi_result,
            ty: HirType::I64,
            incoming: vec![(live_value, entry), (stale_value, stale)],
        });
        join_block.terminator = HirTerminator::Return {
            values: vec![phi_result],
        };

        let stats = run_function(&mut f);
        assert_eq!(stats.removed, 1);
        assert!(f.blocks[&join].phis.is_empty());
        assert!(matches!(
            &f.blocks[&join].terminator,
            HirTerminator::Return { values } if values == &vec![live_value]
        ));
    }
}
