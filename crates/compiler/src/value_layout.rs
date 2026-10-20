//! In-memory value layouts shared by the interpreter and native backends.

use crate::hir::HirType;

/// Byte size of a value of `ty` in memory, laid out as the native
/// backends lay it out: fields at their natural alignment, the total
/// rounded up to the struct's alignment.
pub fn size_of_hir_ty(ty: &HirType) -> usize {
    match ty {
        HirType::Void => 0,
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 => 8,
        HirType::Ptr(_) | HirType::Ref { .. } | HirType::USize | HirType::ISize => {
            crate::target_pointer_size()
        }
        HirType::I128 | HirType::U128 => 16,
        HirType::Struct(s) => struct_size(s),
        HirType::Union(u) => union_layout(u).size,
        HirType::Array(elem, n) => size_of_hir_ty(elem).saturating_mul((*n) as usize),
        HirType::Vector(elem, n) => size_of_hir_ty(elem).saturating_mul((*n) as usize),
        _ => crate::target_pointer_size(),
    }
}

/// Alignment of a value of `ty` in memory.
pub fn align_of_hir_ty(ty: &HirType) -> usize {
    match ty {
        HirType::Void => 1,
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 => 8,
        HirType::Ptr(_) | HirType::Ref { .. } | HirType::USize | HirType::ISize => {
            crate::target_pointer_size()
        }
        HirType::I128 | HirType::U128 => 16,
        HirType::Struct(s) => {
            if s.packed {
                1
            } else {
                s.fields.iter().map(align_of_hir_ty).max().unwrap_or(1)
            }
        }
        HirType::Union(u) => union_layout(u).align,
        HirType::Array(elem, _) => align_of_hir_ty(elem),
        HirType::Vector(elem, _) => align_of_hir_ty(elem),
        _ => crate::target_pointer_size(),
    }
}

/// Where a struct's fields sit.
pub struct StructLayout {
    pub offsets: Vec<usize>,
    pub size: usize,
    pub align: usize,
}

fn struct_size(s: &crate::hir::HirStructType) -> usize {
    let mut size = 0usize;
    let mut align = 1usize;
    for field in &s.fields {
        let a = if s.packed { 1 } else { align_of_hir_ty(field) };
        align = align.max(a);
        size = size.div_ceil(a) * a + size_of_hir_ty(field);
    }
    size.div_ceil(align) * align
}

/// The layout of `s`: each field at its natural alignment unless the
/// struct is packed, the size rounded up to the struct's alignment.
pub fn struct_layout(s: &crate::hir::HirStructType) -> StructLayout {
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(s.fields.len());
    let mut align = 1usize;
    for field in &s.fields {
        let a = if s.packed { 1 } else { align_of_hir_ty(field) };
        align = align.max(a);
        offset = offset.div_ceil(a) * a;
        offsets.push(offset);
        offset += size_of_hir_ty(field);
    }
    let size = offset.div_ceil(align) * align;
    StructLayout {
        offsets,
        size,
        align,
    }
}

/// Where a union stores its tag and the largest payload, including padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnionLayout {
    pub payload_offset: usize,
    pub size: usize,
    pub align: usize,
}

pub fn union_layout(union: &crate::hir::HirUnionType) -> UnionLayout {
    let payload_size = union
        .variants
        .iter()
        .map(|v| size_of_hir_ty(&v.ty))
        .max()
        .unwrap_or(0);
    let payload_align = union
        .variants
        .iter()
        .map(|v| align_of_hir_ty(&v.ty))
        .max()
        .unwrap_or(1);
    let (tag_size, tag_align) = if union.is_c_union {
        (0, 1)
    } else {
        (
            size_of_hir_ty(&union.discriminant_type),
            align_of_hir_ty(&union.discriminant_type),
        )
    };
    let align = payload_align.max(tag_align);
    let payload_offset = tag_size.div_ceil(payload_align) * payload_align;
    let size = (payload_offset + payload_size).max(1).div_ceil(align) * align;
    UnionLayout {
        payload_offset,
        size,
        align,
    }
}
