//! Dynamic boxes made, read and released in HIR.
//!
//! The runtime's box is `{ tag: u32, size: u32, data: *u8, dropper,
//! display }` (`zrtl::DynamicBoxRepr`), with the value behind `data`.
//! Every operation on one is a runtime call: `zyntax_box_i64` and its
//! kin allocate a box and a payload, the readers `zyntax_box_header_tag`,
//! `zyntax_box_data`, `zyntax_box_pointer` and
//! `zyntax_box_payload_{i64,f64,bool}` load a word or two out of a box
//! the caller has checked already, and
//! `zyntax_box_free` releases one. As calls they are opaque: nothing
//! hoists one out of a loop, merges two of the same box, drops one
//! nobody reads, or sees an allocation whose every use is in view.
//!
//! This pass replaces each with what it stands for. A scalar box is one
//! allocation through the allocation intrinsic with the payload in the
//! same block, right after the header, and no dropper; the readers are
//! the loads they name; a release of a box made here is the release
//! intrinsic, since there is no dropper to run. A release of a box made
//! anywhere else stays a call, because the runtime's release runs the
//! dropper and returns the block to whichever allocator made it.
//!
//! Runs after the release sites are placed. A released box takes its
//! payload with it, and the release analysis follows the reader's call
//! to learn that the payload is the box under another name; a load
//! tells it nothing of the kind.
//!
//! `ZYNTAX_DISABLE_BOX_READS=1` leaves the calls in place; safe.

use std::collections::{HashMap, HashSet};

use crate::hir::{
    CastOp, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirType,
    HirValue, HirValueKind,
};

/// Byte offset of `tag` in the box header.
const TAG_OFFSET: i64 = 0;
/// Byte offset of `size` in the box header.
const SIZE_OFFSET: i64 = 4;
/// Byte offset of `data` in the box header.
const DATA_OFFSET: i64 = 8;
/// Byte offset of `dropper` in the box header.
const DROPPER_OFFSET: i64 = 16;
/// Byte offset of `display_fn` in the box header.
const DISPLAY_OFFSET: i64 = 24;
/// Size of the header; a payload made here follows it.
const HEADER_SIZE: i64 = 32;

/// A box the pass makes: the tag it carries and the payload's type.
#[derive(Clone)]
enum Make {
    /// A scalar payload right after the header, `data` pointing at it.
    Scalar { tag: u32, payload: HirType },
    /// A header alone, `data` given by the caller along with the tag.
    Pointer,
}

fn make_of(symbol: &str) -> Option<Make> {
    use crate::zrtl::TypeTag;
    Some(match symbol {
        "zyntax_box_i64" => Make::Scalar {
            tag: TypeTag::I64.0,
            payload: HirType::I64,
        },
        "zyntax_box_f64" => Make::Scalar {
            tag: TypeTag::F64.0,
            payload: HirType::F64,
        },
        "zyntax_box_i32" => Make::Scalar {
            tag: TypeTag::I32.0,
            payload: HirType::I32,
        },
        "zyntax_box_f32" => Make::Scalar {
            tag: TypeTag::F32.0,
            payload: HirType::F32,
        },
        "zyntax_box_bool" => Make::Scalar {
            tag: TypeTag::BOOL.0,
            payload: HirType::U8,
        },
        "zyntax_box_ptr" => Make::Pointer,
        _ => return None,
    })
}

/// The release of a box.
const FREE: &str = "zyntax_box_free";

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
        "zyntax_box_data" | "zyntax_box_pointer" => Some(Read::Data),
        "zyntax_box_payload_i64" | "zyntax_box_payload_f64" => Some(Read::Payload),
        "zyntax_box_payload_bool" => Some(Read::PayloadByte),
        _ => None,
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BoxStats {
    /// Reader calls replaced by loads.
    pub expanded: usize,
    /// Boxes made through the allocation intrinsic.
    pub made: usize,
    /// Releases of such boxes through the release intrinsic.
    pub released: usize,
}

pub fn run_module(module: &mut HirModule) -> BoxStats {
    let mut stats = BoxStats::default();
    if std::env::var_os("ZYNTAX_DISABLE_BOX_READS").is_some() {
        return stats;
    }
    // A reader reached through an extern declaration of the module is
    // the same reader.
    let externs: HashMap<HirId, String> = module
        .functions
        .iter()
        .filter(|(_, f)| f.is_external)
        .filter_map(|(id, f)| f.link_name.clone().map(|n| (*id, n)))
        .collect();
    for func in module.functions.values_mut() {
        if func.is_external {
            continue;
        }
        let s = run_function(func, &externs);
        // The loads and stores are new to the passes that follow.
        if s.expanded + s.made + s.released > 0 {
            func.attributes.optimized = false;
        }
        stats.expanded += s.expanded;
        stats.made += s.made;
        stats.released += s.released;
    }
    stats
}

fn run_function(func: &mut HirFunction, externs: &HashMap<HirId, String>) -> BoxStats {
    let mut stats = BoxStats::default();
    // The symbol a call reaches, by name or through an extern declaration.
    let symbol_of = |callee: &HirCallable| -> Option<String> {
        match callee {
            HirCallable::Symbol(name) => Some(name.clone()),
            HirCallable::Function(id) => externs.get(id).cloned(),
            _ => None,
        }
    };
    // Boxes made in this function have no dropper, so their release is
    // the release intrinsic. A release site follows its box in block
    // order, which is the order walked here.
    let mut made_here: HashSet<HirId> = HashSet::new();
    let block_ids: Vec<HirId> = func.blocks.keys().copied().collect();
    for block_id in block_ids {
        let instructions =
            std::mem::take(&mut func.blocks.get_mut(&block_id).unwrap().instructions);
        let mut out = Vec::with_capacity(instructions.len());
        for inst in instructions {
            let HirInstruction::Call {
                result,
                callee,
                args,
                ..
            } = &inst
            else {
                out.push(inst);
                continue;
            };
            let Some(symbol) = symbol_of(callee) else {
                out.push(inst);
                continue;
            };
            match (result, read_of(&symbol), make_of(&symbol)) {
                (Some(result), Some(read), _) if args.len() == 1 => {
                    expand(func, *result, args[0], read, &mut out);
                    stats.expanded += 1;
                }
                (Some(result), _, Some(make)) => {
                    make_box(func, *result, args, make, &mut out);
                    made_here.insert(*result);
                    stats.made += 1;
                }
                (None, _, _)
                    if symbol == FREE && args.len() == 1 && made_here.contains(&args[0]) =>
                {
                    out.push(HirInstruction::Call {
                        result: None,
                        callee: HirCallable::Intrinsic(crate::hir::Intrinsic::Free),
                        args: vec![args[0]],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    });
                    stats.released += 1;
                }
                _ => out.push(inst),
            }
        }
        func.blocks.get_mut(&block_id).unwrap().instructions = out;
    }
    stats
}

/// The allocation and stores for one box, ending in the value `result`
/// the call defined.
fn make_box(
    func: &mut HirFunction,
    result: HirId,
    args: &[HirId],
    make: Make,
    out: &mut Vec<HirInstruction>,
) {
    let (total, tag, size, payload) = match make {
        Make::Scalar { tag, payload } => {
            let width = hir_ty_width(&payload);
            (HEADER_SIZE + 8, Some(tag), width, Some(payload))
        }
        Make::Pointer => (HEADER_SIZE, None, 8, None),
    };
    let size_const = constant(func, HirType::I64, HirConstant::I64(total));
    out.push(HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Intrinsic(crate::hir::Intrinsic::Malloc),
        args: vec![size_const],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    let tag_value = match tag {
        Some(tag) => constant(func, HirType::U32, HirConstant::U32(tag)),
        // A pointer box's tag arrives as the call's second argument.
        None => args[1],
    };
    store_field(func, result, TAG_OFFSET, tag_value, HirType::U32, out);
    let size_value = constant(func, HirType::U32, HirConstant::U32(size as u32));
    store_field(func, result, SIZE_OFFSET, size_value, HirType::U32, out);
    let null = constant(func, HirType::I64, HirConstant::I64(0));
    store_field(func, result, DROPPER_OFFSET, null, HirType::I64, out);
    store_field(func, result, DISPLAY_OFFSET, null, HirType::I64, out);
    match payload {
        Some(payload) => {
            // `data` points at the payload, right after the header.
            let payload_ptr = field_ptr(func, result, HEADER_SIZE, payload.clone(), out);
            let as_int = value(func, HirType::I64);
            out.push(HirInstruction::Cast {
                result: as_int,
                ty: HirType::I64,
                op: CastOp::PtrToInt,
                operand: payload_ptr,
            });
            store_field(func, result, DATA_OFFSET, as_int, HirType::I64, out);
            out.push(HirInstruction::Store {
                value: args[0],
                ptr: payload_ptr,
                align: 8,
                volatile: false,
            });
        }
        None => {
            // `data` is the pointer given. A value of an aggregate type is
            // the address of the aggregate, and a store of the type itself
            // would copy it: store the address.
            let data_ty = func.values[&args[0]].ty.clone();
            if matches!(
                data_ty,
                HirType::Struct(_) | HirType::Array(_, _) | HirType::Union(_)
            ) {
                let address = value(func, HirType::I64);
                out.push(HirInstruction::Cast {
                    result: address,
                    ty: HirType::I64,
                    op: CastOp::PtrToInt,
                    operand: args[0],
                });
                store_field(func, result, DATA_OFFSET, address, HirType::I64, out);
            } else {
                store_field(func, result, DATA_OFFSET, args[0], data_ty, out);
            }
        }
    }
}

/// `store value` at `offset` bytes into `boxed`, as a `field_ty`.
fn store_field(
    func: &mut HirFunction,
    boxed: HirId,
    offset: i64,
    value: HirId,
    field_ty: HirType,
    out: &mut Vec<HirInstruction>,
) {
    let ptr = field_ptr(func, boxed, offset, field_ty, out);
    out.push(HirInstruction::Store {
        value,
        ptr,
        align: 8,
        volatile: false,
    });
}

fn hir_ty_width(ty: &HirType) -> i64 {
    match ty {
        HirType::U8 | HirType::I8 | HirType::Bool => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        _ => 8,
    }
}

fn constant(func: &mut HirFunction, ty: HirType, c: HirConstant) -> HirId {
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty,
            kind: HirValueKind::Constant(c),
            uses: HashSet::new(),
            span: None,
        },
    );
    id
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
    let offset_id = constant(func, HirType::I64, HirConstant::I64(offset));
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
            std::mem::offset_of!(DynamicBoxRepr, size) as i64,
            SIZE_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(DynamicBoxRepr, data) as i64,
            DATA_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(DynamicBoxRepr, dropper) as i64,
            DROPPER_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(DynamicBoxRepr, display_fn) as i64,
            DISPLAY_OFFSET
        );
        assert_eq!(std::mem::size_of::<DynamicBoxRepr>() as i64, HEADER_SIZE);
    }
}
