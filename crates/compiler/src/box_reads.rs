//! Reads out of a dynamic box as the loads they are.
//!
//! The runtime's box is `{ tag: u32, size: u32, data: *u8, .. }`
//! (`zrtl::DynamicBoxRepr`), with the value behind `data`. Its readers
//! `zyntax_box_header_tag`, `zyntax_box_data` and
//! `zyntax_box_payload_{i64,f64,bool}` take a box the caller has checked
//! already, not null and holding a payload of the width read, and do one
//! or two loads. Left as calls they are opaque: nothing hoists one out of
//! a loop, merges two of the same box, or drops one nobody reads. This
//! pass replaces each call with the loads it stands for.
//!
//! Runs after the release sites are placed. A released box takes its
//! payload with it, and the release analysis follows the reader's call to
//! learn that the payload is the box under another name; a load tells it
//! nothing of the kind.
//!
//! `ZYNTAX_DISABLE_BOX_READS=1` leaves the calls in place; safe.

use std::collections::{HashMap, HashSet};

use crate::hir::{
    CastOp, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirType,
    HirValue, HirValueKind,
};

/// Byte offset of `tag` in the box header.
const TAG_OFFSET: i64 = 0;
/// Byte offset of `data` in the box header.
const DATA_OFFSET: i64 = 8;

/// What a reader loads.
#[derive(Clone, Copy)]
enum Read {
    /// The header's tag.
    Tag,
    /// The `data` pointer, as the type the call result has.
    Data,
    /// The value behind `data`, as the type the call result has.
    Payload,
    /// One byte behind `data`, widened to the type the call result has.
    PayloadByte,
}

fn read_of(symbol: &str) -> Option<Read> {
    match symbol {
        "zyntax_box_header_tag" => Some(Read::Tag),
        "zyntax_box_data" => Some(Read::Data),
        "zyntax_box_payload_i64" | "zyntax_box_payload_f64" => Some(Read::Payload),
        "zyntax_box_payload_bool" => Some(Read::PayloadByte),
        _ => None,
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BoxReadStats {
    /// Reader calls replaced by loads.
    pub expanded: usize,
}

pub fn run_module(module: &mut HirModule) -> BoxReadStats {
    let mut stats = BoxReadStats::default();
    if std::env::var_os("ZYNTAX_DISABLE_BOX_READS").is_some() {
        return stats;
    }
    // A reader reached through an extern declaration of the module is
    // the same reader.
    let externs: HashMap<HirId, Read> = module
        .functions
        .iter()
        .filter(|(_, f)| f.is_external)
        .filter_map(|(id, f)| f.link_name.as_deref().and_then(read_of).map(|r| (*id, r)))
        .collect();
    for func in module.functions.values_mut() {
        if func.is_external {
            continue;
        }
        stats.expanded += run_function(func, &externs);
    }
    stats
}

fn run_function(func: &mut HirFunction, externs: &HashMap<HirId, Read>) -> usize {
    let block_ids: Vec<HirId> = func.blocks.keys().copied().collect();
    let mut expanded = 0;
    for block_id in block_ids {
        let instructions =
            std::mem::take(&mut func.blocks.get_mut(&block_id).unwrap().instructions);
        let mut out = Vec::with_capacity(instructions.len());
        for inst in instructions {
            let read = match &inst {
                HirInstruction::Call {
                    result: Some(_),
                    callee,
                    args,
                    ..
                } if args.len() == 1 => match callee {
                    HirCallable::Symbol(name) => read_of(name),
                    HirCallable::Function(id) => externs.get(id).copied(),
                    _ => None,
                },
                _ => None,
            };
            match (read, &inst) {
                (
                    Some(read),
                    HirInstruction::Call {
                        result: Some(result),
                        args,
                        ..
                    },
                ) => {
                    expand(func, *result, args[0], read, &mut out);
                    expanded += 1;
                }
                _ => out.push(inst),
            }
        }
        func.blocks.get_mut(&block_id).unwrap().instructions = out;
    }
    expanded
}

/// The loads for one reader call, ending in the value `result` the call
/// defined, so nothing downstream changes.
fn expand(
    func: &mut HirFunction,
    result: HirId,
    boxed: HirId,
    read: Read,
    out: &mut Vec<HirInstruction>,
) {
    let result_ty = func.values[&result].ty.clone();
    match read {
        Read::Tag => {
            let ptr = field_ptr(func, boxed, TAG_OFFSET, result_ty.clone(), out);
            out.push(load(result, result_ty, ptr));
        }
        // A value of an aggregate type is the address of the aggregate,
        // and a load of the type itself would copy it: load the address.
        Read::Data if matches!(result_ty, HirType::Struct(_) | HirType::Array(_, _)) => {
            let addr_ty = HirType::Ptr(Box::new(result_ty.clone()));
            let ptr = field_ptr(func, boxed, DATA_OFFSET, addr_ty.clone(), out);
            let addr = value(func, addr_ty.clone());
            out.push(load(addr, addr_ty, ptr));
            out.push(HirInstruction::Cast {
                result,
                ty: result_ty,
                op: CastOp::Bitcast,
                operand: addr,
            });
        }
        Read::Data => {
            let ptr = field_ptr(func, boxed, DATA_OFFSET, result_ty.clone(), out);
            out.push(load(result, result_ty, ptr));
        }
        Read::Payload => {
            let data = data_ptr(func, boxed, result_ty.clone(), out);
            out.push(load(result, result_ty, data));
        }
        Read::PayloadByte => {
            let data = data_ptr(func, boxed, HirType::U8, out);
            let byte = value(func, HirType::U8);
            out.push(load(byte, HirType::U8, data));
            out.push(HirInstruction::Cast {
                result,
                ty: result_ty,
                op: CastOp::ZExt,
                operand: byte,
            });
        }
    }
}

/// The `data` pointer of `boxed`, typed as a pointer to `pointee`.
fn data_ptr(
    func: &mut HirFunction,
    boxed: HirId,
    pointee: HirType,
    out: &mut Vec<HirInstruction>,
) -> HirId {
    let data_ty = HirType::Ptr(Box::new(pointee));
    let ptr = field_ptr(func, boxed, DATA_OFFSET, data_ty.clone(), out);
    let data = value(func, data_ty.clone());
    out.push(load(data, data_ty, ptr));
    data
}

/// The address `offset` bytes into `boxed`, as a pointer to `field_ty`.
fn field_ptr(
    func: &mut HirFunction,
    boxed: HirId,
    offset: i64,
    field_ty: HirType,
    out: &mut Vec<HirInstruction>,
) -> HirId {
    let offset_id = HirId::new();
    func.values.insert(
        offset_id,
        HirValue {
            id: offset_id,
            ty: HirType::I64,
            kind: HirValueKind::Constant(HirConstant::I64(offset)),
            uses: HashSet::new(),
            span: None,
        },
    );
    let gep = value(func, HirType::Ptr(Box::new(HirType::U8)));
    out.push(HirInstruction::GetElementPtr {
        result: gep,
        ty: HirType::U8,
        ptr: boxed,
        indices: vec![offset_id],
    });
    let ptr_ty = HirType::Ptr(Box::new(field_ty));
    let ptr = value(func, ptr_ty.clone());
    out.push(HirInstruction::Cast {
        result: ptr,
        ty: ptr_ty,
        op: CastOp::Bitcast,
        operand: gep,
    });
    ptr
}

fn load(result: HirId, ty: HirType, ptr: HirId) -> HirInstruction {
    HirInstruction::Load {
        result,
        ty,
        ptr,
        align: 8,
        volatile: false,
    }
}

fn value(func: &mut HirFunction, ty: HirType) -> HirId {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zrtl::DynamicBoxRepr;

    #[test]
    fn offsets_match_the_runtime_box() {
        assert_eq!(std::mem::offset_of!(DynamicBoxRepr, tag) as i64, TAG_OFFSET);
        assert_eq!(
            std::mem::offset_of!(DynamicBoxRepr, data) as i64,
            DATA_OFFSET
        );
    }
}
