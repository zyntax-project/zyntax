//! Remove overwritten scalar stores after forwarding has removed their reads.
//! Availability stops at block boundaries and potentially observing operations.

use super::*;
use smallvec::SmallVec;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Global(HirId),
    Value(HirId),
}

fn key(f: &HirFunction, ptr: HirId) -> Key {
    match f.values.get(&ptr).map(|v| &v.kind) {
        Some(HirValueKind::Global(id)) => Key::Global(*id),
        _ => Key::Value(ptr),
    }
}

pub(super) fn eliminate(f: &mut HirFunction) {
    let mut removals = Vec::new();
    let mut addresses = None;
    for (id, block) in &f.blocks {
        let mut pending: SmallVec<[(Key, usize, HirId, &HirType); 4]> = SmallVec::new();
        let mut dead = IdHashSet::default();
        for (index, inst) in block.instructions.iter().enumerate() {
            match inst {
                HirInstruction::Store {
                    ptr,
                    value,
                    volatile: false,
                    ..
                } => {
                    let Some(ty) = scalar_store_type(f, *value) else {
                        pending.clear();
                        continue;
                    };
                    let k = key(f, *ptr);
                    if let Some(slot) = pending.iter().position(|(key, ..)| *key == k) {
                        let (_, earlier, _, previous_ty) = pending.swap_remove(slot);
                        if previous_ty == ty {
                            dead.insert(earlier);
                        }
                    }
                    pending.retain(|(_, _, other, other_ty)| {
                        let addresses = addresses.get_or_insert_with(|| Addresses::new(f));
                        addresses.fresh_against_global(*ptr, *other)
                            || separate_bytes(
                                addresses.get(*ptr, 32),
                                hir_ty_byte_size(ty),
                                addresses.get(*other, 32),
                                hir_ty_byte_size(other_ty),
                            )
                    });
                    // Bound tracking cost for large initialization blocks.
                    if pending.len() == 64 {
                        pending.clear();
                    }
                    pending.push((k, index, *ptr, ty));
                }
                HirInstruction::Load {
                    ptr,
                    ty,
                    volatile: false,
                    ..
                } => {
                    pending.retain(|(_, _, other, other_ty)| {
                        let addresses = addresses.get_or_insert_with(|| Addresses::new(f));
                        addresses.fresh_against_global(*ptr, *other)
                            || separate_bytes(
                                addresses.get(*ptr, 32),
                                hir_ty_byte_size(ty),
                                addresses.get(*other, 32),
                                hir_ty_byte_size(other_ty),
                            )
                    });
                }
                HirInstruction::Binary { op, .. }
                    if !matches!(op, crate::hir::BinaryOp::Div | crate::hir::BinaryOp::Rem) => {}
                HirInstruction::Unary { .. } | HirInstruction::Select { .. } => {}
                _ => pending.clear(),
            }
        }
        if !dead.is_empty() {
            removals.push((*id, dead));
        }
    }
    for (id, dead) in removals {
        let mut index = 0;
        f.blocks.get_mut(&id).unwrap().instructions.retain(|_| {
            let keep = !dead.contains(&index);
            index += 1;
            keep
        });
    }
}
