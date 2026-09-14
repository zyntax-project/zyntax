//! Dead stores into storage allocated in the same block.
//!
//! A constructor fills every field with its default and the body then
//! stores the real value over it, so after inlining a block holds two
//! stores to one field of one fresh allocation with nothing reading it
//! in between. The first is dead.
//!
//! The pass is deliberately narrow, so that it needs no alias analysis:
//! it looks only at pointers derived from a `malloc` in the same block
//! by constant-offset `gep`s and casts. Nothing else can name that
//! memory until its address leaves the block or is stored somewhere,
//! and the pass stops tracking an allocation at the first such escape.
//! A load through a tracked pointer keeps every earlier store whose
//! bytes it may read.

use crate::hir::{
    HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirType, HirValueKind,
    Intrinsic,
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DeadStoreStats {
    /// Stores removed.
    pub removed: usize,
}

pub fn run_module(module: &mut HirModule) -> DeadStoreStats {
    let mut total = DeadStoreStats::default();
    for func in module.functions_to_optimize() {
        if func.is_external {
            continue;
        }
        total.removed += run_function(func);
    }
    total
}

fn run_function(func: &mut HirFunction) -> usize {
    let consts: HashMap<HirId, i64> = func
        .values
        .iter()
        .filter_map(|(id, v)| match &v.kind {
            HirValueKind::Constant(c) => const_as_i64(c).map(|n| (*id, n)),
            _ => None,
        })
        .collect();
    let types: HashMap<HirId, HirType> = func
        .values
        .iter()
        .map(|(id, v)| (*id, v.ty.clone()))
        .collect();
    let mut removed = 0;
    let block_ids: Vec<HirId> = func.blocks.keys().copied().collect();
    for bid in block_ids {
        let dead = {
            let Some(block) = func.blocks.get(&bid) else {
                continue;
            };
            dead_stores_in(&block.instructions, &consts, &types)
        };
        if dead.is_empty() {
            continue;
        }
        if let Some(block) = func.blocks.get_mut(&bid) {
            let mut idx = 0;
            block.instructions.retain(|_| {
                let keep = !dead.contains(&idx);
                idx += 1;
                keep
            });
            removed += dead.len();
        }
    }
    removed
}

/// A byte range of one allocation.
#[derive(Clone, Copy)]
struct Place {
    alloc: HirId,
    offset: i64,
}

/// Indices of the stores in `insts` whose bytes are written again
/// before anything reads them.
fn dead_stores_in(
    insts: &[HirInstruction],
    consts: &HashMap<HirId, i64>,
    types: &HashMap<HirId, HirType>,
) -> HashSet<usize> {
    // Pointers into an allocation made here, by allocation and offset.
    let mut places: HashMap<HirId, Place> = HashMap::new();
    // Allocations whose address has left the block's view.
    let mut escaped: HashSet<HirId> = HashSet::new();
    // Per allocation, the stores not yet read or overwritten: index,
    // offset, width.
    let mut pending: HashMap<HirId, Vec<(usize, i64, i64)>> = HashMap::new();
    let mut dead = HashSet::new();

    let width_of = |v: &HirId| types.get(v).map(size_of).unwrap_or(8) as i64;
    let overlaps = |a: (i64, i64), b: (i64, i64)| a.0 < b.0 + b.1 && b.0 < a.0 + a.1;

    for (idx, inst) in insts.iter().enumerate() {
        match inst {
            HirInstruction::Call {
                result: Some(result),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                ..
            } => {
                places.insert(
                    *result,
                    Place {
                        alloc: *result,
                        offset: 0,
                    },
                );
            }
            HirInstruction::GetElementPtr {
                result,
                ty,
                ptr,
                indices,
            } => {
                if let Some(base) = places.get(ptr).copied() {
                    match const_offset(ty, indices, consts) {
                        Some(off) => {
                            places.insert(
                                *result,
                                Place {
                                    alloc: base.alloc,
                                    offset: base.offset + off,
                                },
                            );
                        }
                        None => {
                            escaped.insert(base.alloc);
                        }
                    }
                }
            }
            HirInstruction::Cast {
                result, operand, ..
            } => {
                if let Some(base) = places.get(operand).copied() {
                    if matches!(types.get(result), Some(HirType::Ptr(_))) {
                        places.insert(*result, base);
                    } else {
                        // An address as an integer may come back as a
                        // pointer nothing here recognises.
                        escaped.insert(base.alloc);
                    }
                }
            }
            HirInstruction::Store { value, ptr, .. } => {
                if let Some(place) = places.get(value) {
                    escaped.insert(place.alloc);
                }
                let Some(place) = places.get(ptr).copied() else {
                    continue;
                };
                // Once the address is out, a store may be read through
                // it at any time; nothing from here on is decided.
                if escaped.contains(&place.alloc) {
                    pending.remove(&place.alloc);
                    continue;
                }
                let width = width_of(value);
                let list = pending.entry(place.alloc).or_default();
                // An earlier store of exactly these bytes, unread since,
                // is dead.
                list.retain(|(earlier, off, w)| {
                    if *off == place.offset && *w == width {
                        dead.insert(*earlier);
                        false
                    } else {
                        true
                    }
                });
                list.push((idx, place.offset, width));
            }
            HirInstruction::Load { result, ptr, .. } => {
                let Some(place) = places.get(ptr).copied() else {
                    continue;
                };
                // What the load reads stays.
                let width = width_of(result);
                if let Some(list) = pending.get_mut(&place.alloc) {
                    list.retain(|(_, off, w)| !overlaps((*off, *w), (place.offset, width)));
                }
            }
            HirInstruction::Call { args, .. } => {
                // A callee handed the address may read or keep it.
                for a in args {
                    if let Some(place) = places.get(a) {
                        escaped.insert(place.alloc);
                    }
                }
            }
            other => {
                // Any other reader of a tracked pointer: keep everything.
                for used in other.operands() {
                    if let Some(place) = places.get(&used) {
                        escaped.insert(place.alloc);
                    }
                }
            }
        }
    }
    dead
}

/// The byte offset a GEP with constant indices adds, by the stride
/// rule the backends use: `ty` is what one step covers.
fn const_offset(ty: &HirType, indices: &[HirId], consts: &HashMap<HirId, i64>) -> Option<i64> {
    let mut offset = 0i64;
    let mut cur = ty.clone();
    for idx in indices {
        let v = *consts.get(idx)?;
        match &cur {
            HirType::U8 | HirType::I8 => offset += v,
            HirType::Ptr(inner) => {
                offset += v.checked_mul(size_of(inner) as i64)?;
                cur = (**inner).clone();
            }
            HirType::Array(elem, _) => {
                offset += v.checked_mul(size_of(elem) as i64)?;
                cur = (**elem).clone();
            }
            _ => return None,
        }
    }
    Some(offset)
}

fn const_as_i64(c: &HirConstant) -> Option<i64> {
    Some(match c {
        HirConstant::I8(v) => *v as i64,
        HirConstant::I16(v) => *v as i64,
        HirConstant::I32(v) => *v as i64,
        HirConstant::I64(v) => *v,
        HirConstant::U8(v) => *v as i64,
        HirConstant::U16(v) => *v as i64,
        HirConstant::U32(v) => *v as i64,
        HirConstant::U64(v) => *v as i64,
        HirConstant::USize(v) => *v as i64,
        HirConstant::ISize(v) => *v,
        _ => return None,
    })
}

fn size_of(ty: &HirType) -> usize {
    match ty {
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 | HirType::Ptr(_) => 8,
        HirType::I128 | HirType::U128 => 16,
        HirType::Struct(s) => s.fields.iter().map(size_of).sum::<usize>().max(1),
        HirType::Array(elem, n) => size_of(elem).saturating_mul(*n as usize),
        _ => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirFunctionSignature, HirTerminator, HirValue};
    use indexmap::IndexMap;
    use zyntax_typed_ast::InternedString;

    fn value(values: &mut IndexMap<HirId, HirValue>, ty: HirType, kind: HirValueKind) -> HirId {
        let id = HirId::new();
        values.insert(
            id,
            HirValue {
                id,
                ty,
                kind,
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    /// `p = malloc 16; store 0 -> p+8; store x -> p+8; load p+8`: the
    /// first store is dead, the second is not.
    #[test]
    fn a_store_written_over_before_any_read_is_removed() {
        let mut values = IndexMap::new();
        let c16 = value(
            &mut values,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(16)),
        );
        let c8 = value(
            &mut values,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(8)),
        );
        let zero = value(
            &mut values,
            HirType::F64,
            HirValueKind::Constant(HirConstant::F64(0.0)),
        );
        let x = value(&mut values, HirType::F64, HirValueKind::Parameter(0));
        let p = value(
            &mut values,
            HirType::Ptr(Box::new(HirType::U8)),
            HirValueKind::Instruction,
        );
        let g = value(
            &mut values,
            HirType::Ptr(Box::new(HirType::U8)),
            HirValueKind::Instruction,
        );
        let f = value(
            &mut values,
            HirType::Ptr(Box::new(HirType::F64)),
            HirValueKind::Instruction,
        );
        let r = value(&mut values, HirType::F64, HirValueKind::Instruction);

        let block_id = HirId::new();
        let mut block = HirBlock::new(block_id);
        block.instructions = vec![
            HirInstruction::Call {
                result: Some(p),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                args: vec![c16],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
            HirInstruction::GetElementPtr {
                result: g,
                ty: HirType::U8,
                ptr: p,
                indices: vec![c8],
            },
            HirInstruction::Cast {
                op: crate::hir::CastOp::Bitcast,
                result: f,
                ty: HirType::Ptr(Box::new(HirType::F64)),
                operand: g,
            },
            HirInstruction::Store {
                value: zero,
                ptr: f,
                align: 8,
                volatile: false,
            },
            HirInstruction::Store {
                value: x,
                ptr: f,
                align: 8,
                volatile: false,
            },
            HirInstruction::Load {
                result: r,
                ty: HirType::F64,
                ptr: f,
                align: 8,
                volatile: false,
            },
        ];
        block.terminator = HirTerminator::Return { values: vec![r] };

        let mut func = HirFunction::new(
            InternedString::new_global("f"),
            HirFunctionSignature {
                params: vec![],
                returns: vec![HirType::F64],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
                is_fiber: false,
                effects: vec![],
                is_pure: false,
            },
        );
        func.values = values;
        func.blocks.insert(block_id, block);
        func.entry_block = block_id;

        assert_eq!(run_function(&mut func), 1);
        let stores: Vec<HirId> = func.blocks[&block_id]
            .instructions
            .iter()
            .filter_map(|i| match i {
                HirInstruction::Store { value, .. } => Some(*value),
                _ => None,
            })
            .collect();
        assert_eq!(
            stores,
            vec![x],
            "the store of the parameter is the one kept"
        );
    }
}
