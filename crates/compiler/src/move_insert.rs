//! Turn an owning parameter into a move the borrow check can see.
//!
//! A parameter marked owned ends the caller's claim on whatever it is
//! passed, but that fact lives on the signature, where nothing walking
//! instructions can find it. This pass reads it once and writes it into
//! the instruction stream: the argument is moved into a fresh value and
//! the call is given that value instead.
//!
//! Passing the same value to two owning parameters then reads as what it
//! is. The second `Move` is a use of something already moved, which is
//! the error the borrow check already knows how to report, so releasing
//! a buffer twice is caught without teaching the checker anything about
//! releasing buffers.
//!
//! Rewriting the argument rather than leaving it in place matters: a
//! `Move` beside an unchanged call would make the call's own use of the
//! argument the first use-after-move, and every consuming call would
//! report against itself.

use std::collections::HashMap;

use crate::hir::{
    HirCallable, HirFunction, HirId, HirInstruction, HirModule, HirType, HirValue, HirValueKind,
    ParamOwnership,
};
use std::collections::HashSet;

#[derive(Debug, Default, Clone, Copy)]
pub struct MoveInsertStats {
    /// Arguments rewritten to pass through a move.
    pub moves_inserted: usize,
}

/// Which parameters of a callee consume their argument, by function id
/// and by link name, since a call names its callee either way.
struct ConsumeIndex {
    by_id: HashMap<HirId, Vec<bool>>,
    by_name: HashMap<String, Vec<bool>>,
}

impl ConsumeIndex {
    fn build(module: &HirModule) -> Self {
        let mut by_id = HashMap::new();
        let mut by_name = HashMap::new();
        for func in module.functions.values() {
            let consumes: Vec<bool> = func
                .signature
                .params
                .iter()
                .map(|p| p.ownership.consumes())
                .collect();
            if !consumes.iter().any(|c| *c) {
                continue;
            }
            by_id.insert(func.id, consumes.clone());
            // A call by symbol names either the link name or the
            // declared one, so both reach the same signature.
            if let Some(link) = func.link_name.as_ref() {
                by_name.insert(link.clone(), consumes.clone());
            }
            if let Some(name) = func.name.resolve_global() {
                by_name.insert(name, consumes);
            }
        }
        Self { by_id, by_name }
    }

    fn for_callee(&self, callee: &HirCallable) -> Option<Vec<bool>> {
        match callee {
            HirCallable::Function(id) => self.by_id.get(id).cloned(),
            HirCallable::Symbol(name) => self.by_name.get(name).cloned(),
            // Some intrinsics end the caller's claim without any
            // signature to read it from, because the frontend never
            // lowers them as a call to a declared function. Releasing a
            // buffer is the whole point of this pass and takes this
            // path, so an ownership modifier written in a declaration
            // would never have been consulted for it.
            HirCallable::Intrinsic(i) => intrinsic_consumes(i),
            // An indirect call has no signature to read here.
            _ => None,
        }
    }
}

/// Which arguments an intrinsic takes ownership of.
///
/// Each of these leaves its first argument unusable: the storage is
/// handed back, moved, or destroyed. `Malloc` is deliberately absent,
/// since it produces ownership rather than consuming it.
fn intrinsic_consumes(intrinsic: &crate::hir::Intrinsic) -> Option<Vec<bool>> {
    use crate::hir::Intrinsic;
    match intrinsic {
        Intrinsic::Free | Intrinsic::Drop => Some(vec![true]),
        // The old pointer is invalid whether or not the block moved.
        Intrinsic::Realloc => Some(vec![true, false]),
        _ => None,
    }
}

fn create_value(func: &mut HirFunction, ty: HirType) -> HirId {
    let id = HirId::new();
    func.values.insert(
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

pub fn run_module(module: &mut HirModule) -> MoveInsertStats {
    let mut stats = MoveInsertStats::default();
    let index = ConsumeIndex::build(module);

    let func_ids: Vec<HirId> = module.functions.keys().copied().collect();
    for fid in func_ids {
        let Some(func) = module.functions.get(&fid) else {
            continue;
        };
        if func.is_external {
            continue;
        }
        let block_ids: Vec<HirId> = func.blocks.keys().copied().collect();

        for bid in block_ids {
            // What to move, decided before touching the function so the
            // signature index and the function are not borrowed at once.
            let mut plan: Vec<(usize, Vec<usize>)> = Vec::new();
            {
                let func = module.functions.get(&fid).unwrap();
                let Some(block) = func.blocks.get(&bid) else {
                    continue;
                };
                for (i, inst) in block.instructions.iter().enumerate() {
                    if let HirInstruction::Call { callee, args, .. } = inst {
                        let Some(consumes) = index.for_callee(callee) else {
                            continue;
                        };
                        let positions: Vec<usize> = args
                            .iter()
                            .enumerate()
                            .filter(|(a, _)| consumes.get(*a).copied().unwrap_or(false))
                            .map(|(a, _)| a)
                            .collect();
                        if !positions.is_empty() {
                            plan.push((i, positions));
                        }
                    }
                }
            }
            if plan.is_empty() {
                continue;
            }

            // Apply back to front so earlier instruction indices stay put.
            for (inst_idx, positions) in plan.into_iter().rev() {
                let mut moves: Vec<HirInstruction> = Vec::new();
                let mut rewrites: Vec<(usize, HirId)> = Vec::new();
                for pos in positions {
                    let (arg, ty) = {
                        let func = module.functions.get(&fid).unwrap();
                        let block = func.blocks.get(&bid).unwrap();
                        let HirInstruction::Call { args, .. } = &block.instructions[inst_idx]
                        else {
                            continue;
                        };
                        let arg = args[pos];
                        let ty = func
                            .values
                            .get(&arg)
                            .map(|v| v.ty.clone())
                            .unwrap_or(HirType::I64);
                        (arg, ty)
                    };
                    let func = module.functions.get_mut(&fid).unwrap();
                    let moved = create_value(func, ty.clone());
                    moves.push(HirInstruction::Move {
                        result: moved,
                        ty,
                        source: arg,
                    });
                    rewrites.push((pos, moved));
                    stats.moves_inserted += 1;
                }

                let func = module.functions.get_mut(&fid).unwrap();
                let block = func.blocks.get_mut(&bid).unwrap();
                if let HirInstruction::Call { args, .. } = &mut block.instructions[inst_idx] {
                    for (pos, moved) in rewrites {
                        args[pos] = moved;
                    }
                }
                for (n, mv) in moves.into_iter().enumerate() {
                    block.instructions.insert(inst_idx + n, mv);
                }
            }
        }
    }
    stats
}
