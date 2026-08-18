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

use std::collections::{HashMap, HashSet};

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
    for func in module.functions.values_mut() {
        let s = run_function(func);
        stats.removed += s.removed;
        stats.rounds = stats.rounds.max(s.rounds);
    }
    stats
}

/// Run the pass over a single function.
pub fn run_function(func: &mut HirFunction) -> PhiPruneStats {
    let mut stats = PhiPruneStats::default();
    // One removal can orphan another phi (a dead phi may have been the
    // only reader of a second one), so iterate. The bound is generous;
    // convergence is typically one or two rounds.
    for _ in 0..16 {
        stats.rounds += 1;
        let used = collect_used_values(func);
        let mut removed_this_round = 0;
        for block in func.blocks.values_mut() {
            let before = block.phis.len();
            block.phis.retain(|p| used.contains(&p.result));
            removed_this_round += before - block.phis.len();
        }
        removed_this_round += collapse_trivial_phis(func);
        stats.removed += removed_this_round;
        if removed_this_round == 0 {
            break;
        }
    }
    stats
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
    let mut replacements: HashMap<HirId, HirId> = HashMap::new();
    for block in func.blocks.values() {
        for phi in &block.phis {
            // A phi with one incoming is left alone. The shape this is
            // for is the loop-carried one, which has an edge from
            // outside the loop and one from within, and a single-entry
            // phi is a different construct that the removal above
            // already decides on.
            if phi.incoming.len() < 2 {
                continue;
            }
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
    let final_map: indexmap::IndexMap<HirId, HirId> = replacements
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

/// Every value id read anywhere in the function: instruction operands,
/// terminator operands, and phi incoming values. A phi result absent from
/// this set is written but never observed.
fn collect_used_values(func: &HirFunction) -> HashSet<HirId> {
    let mut used = HashSet::new();
    for block in func.blocks.values() {
        for phi in &block.phis {
            for (value, _) in &phi.incoming {
                used.insert(*value);
            }
        }
        for inst in &block.instructions {
            collect_inst_operands(inst, &mut used);
        }
        collect_terminator_operands(&block.terminator, &mut used);
    }
    used
}

fn collect_inst_operands(inst: &HirInstruction, used: &mut HashSet<HirId>) {
    macro_rules! u {
        ($($v:expr),*) => {{ $( used.insert(*$v); )* }};
    }
    match inst {
        HirInstruction::Binary { left, right, .. } => u!(left, right),
        HirInstruction::Unary { operand, .. } => u!(operand),
        HirInstruction::Cast { operand, .. } => u!(operand),
        HirInstruction::Load { ptr, .. } => u!(ptr),
        HirInstruction::Store { value, ptr, .. } => u!(value, ptr),
        HirInstruction::GetElementPtr { ptr, indices, .. } => {
            u!(ptr);
            used.extend(indices.iter().copied());
        }
        HirInstruction::Call { args, .. } => used.extend(args.iter().copied()),
        HirInstruction::IndirectCall { func_ptr, args, .. } => {
            u!(func_ptr);
            used.extend(args.iter().copied());
        }
        HirInstruction::Select {
            condition,
            true_val,
            false_val,
            ..
        } => u!(condition, true_val, false_val),
        HirInstruction::ExtractValue { aggregate, .. } => u!(aggregate),
        HirInstruction::InsertValue {
            aggregate, value, ..
        } => u!(aggregate, value),
        HirInstruction::Atomic { ptr, value, .. } => {
            u!(ptr);
            if let Some(v) = value {
                used.insert(*v);
            }
        }
        HirInstruction::Alloca { count, .. } => {
            if let Some(c) = count {
                used.insert(*c);
            }
        }
        HirInstruction::VectorSplat { scalar, .. } => u!(scalar),
        HirInstruction::VectorExtractLane { vector, .. } => u!(vector),
        HirInstruction::VectorInsertLane { vector, scalar, .. } => u!(vector, scalar),
        HirInstruction::VectorHorizontalReduce { vector, .. } => u!(vector),
        HirInstruction::VectorUnaryOp { operand, .. } => u!(operand),
        HirInstruction::VectorMinMax { left, right, .. } => u!(left, right),
        HirInstruction::VectorDot { acc, a, b, .. } => u!(acc, a, b),
        HirInstruction::VectorLoad { ptr, .. } => u!(ptr),
        HirInstruction::VectorStore { value, ptr, .. } => u!(value, ptr),
        // Anything not enumerated is treated as reading nothing here. That
        // is only safe because a missed operand can at worst let a phi
        // survive that could have been dropped — never the reverse. The
        // conservative direction is deliberate: this pass must not remove
        // a phi something still reads, so unknown shapes fall back to
        // marking every value they mention.
        other => collect_unknown_operands(other, used),
    }
}

/// Catch-all for instruction shapes not enumerated above: mark every value
/// the instruction mentions as used, so a phi feeding it is never dropped.
fn collect_unknown_operands(inst: &HirInstruction, used: &mut HashSet<HirId>) {
    // The debug rendering names every `HirId` the instruction holds. Using
    // it keeps this fallback correct as new instruction variants land,
    // rather than silently under-approximating uses when one is added.
    let text = format!("{inst:?}");
    let mut rest = text.as_str();
    while let Some(pos) = rest.find("HirId(") {
        rest = &rest[pos + "HirId(".len()..];
        let end = match rest.find(')') {
            Some(e) => e,
            None => break,
        };
        if let Ok(raw) = rest[..end].trim().parse::<u32>() {
            used.insert(HirId::from_raw(raw));
        }
        rest = &rest[end..];
    }
}

fn collect_terminator_operands(term: &HirTerminator, used: &mut HashSet<HirId>) {
    match term {
        HirTerminator::Return { values } => used.extend(values.iter().copied()),
        HirTerminator::CondBranch { condition, .. } => {
            used.insert(*condition);
        }
        HirTerminator::Switch { value, .. } => {
            used.insert(*value);
        }
        HirTerminator::Invoke { args, .. } => used.extend(args.iter().copied()),
        HirTerminator::PatternMatch { value, .. } => {
            used.insert(*value);
        }
        HirTerminator::Branch { .. } | HirTerminator::Unreachable => {}
    }
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

    /// A phi whose result is read stays.
    #[test]
    fn used_phi_is_kept() {
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
        assert_eq!(stats.removed, 0);
        assert_eq!(f.blocks[&entry].phis.len(), 1);
    }

    /// A phi read only by another (dead) phi goes too — this is why the
    /// pass iterates rather than making a single sweep.
    #[test]
    fn phi_chain_collapses_to_fixed_point() {
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
        assert!(stats.rounds >= 2, "removal should need more than one round");
    }

    /// A phi feeding a real instruction is never dropped.
    #[test]
    fn phi_used_by_instruction_is_kept() {
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

        assert_eq!(run_function(&mut f).removed, 0);
    }
}
