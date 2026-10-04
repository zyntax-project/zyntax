//! The word a reference object carries ahead of its fields.
//!
//! An object of a reference type (one whose instances live on the heap
//! and are passed by pointer) starts with one word its type owns: the
//! descriptor a host heap gives the type, or 0. The word is part of the
//! layout whether or not a host is installed, so lowered code, snapshots
//! included, has one layout in every configuration. A frontend whose
//! objects stay foreign to the host marks the type header-free.
//!
//! The descriptor is read from the global [`descriptor_name`] names, which
//! the runtime fills when the module is installed, and stored at offset 0
//! by every allocation of the type.

use zyntax_typed_ast::InternedString;
use zyntax_typed_ast::type_registry::TypeDefinition;

use crate::hir::HirType;

/// The metadata key a frontend sets on a reference type whose objects
/// carry no header word.
pub const HEADER_FREE_KEY: &str = "header_free";

/// The prefix of a type's descriptor global.
pub const DESCRIPTOR_PREFIX: &str = "$typedesc$";

/// Whether objects of `def` carry the header word.
pub fn has_header(def: &TypeDefinition) -> bool {
    def.metadata.is_reference
        && !def
            .metadata
            .custom
            .contains_key(&InternedString::new_global(HEADER_FREE_KEY))
}

/// The type's name with its module, as a host sees it: `module.Name`.
pub fn qualified_name(def: &TypeDefinition) -> String {
    let name = def.name.resolve_global().unwrap_or_default();
    match def.module.and_then(|m| m.resolve_global()) {
        Some(module) if !module.is_empty() => format!("{module}.{name}"),
        _ => name,
    }
}

/// The global holding `def`'s descriptor.
pub fn descriptor_name(def: &TypeDefinition) -> String {
    format!("{DESCRIPTOR_PREFIX}{}", qualified_name(def))
}

/// The header word's type in a struct's field list.
pub fn header_field() -> HirType {
    HirType::I64
}

/// Byte offsets of `fields` laid out on the heap, and the object's size:
/// each field aligned to its own size, the total padded to the largest.
/// Every reader and writer of a reference object's fields uses this.
pub fn field_layout(fields: &[HirType]) -> (Vec<u64>, u64) {
    let mut offsets = Vec::with_capacity(fields.len());
    let mut running: u64 = 0;
    let mut max_align: u64 = 1;
    for ty in fields {
        let size = crate::ssa::hir_ty_size(ty) as u64;
        let align = size.max(1);
        max_align = max_align.max(align);
        if align > 1 {
            running = (running + align - 1) & !(align - 1);
        }
        offsets.push(running);
        running += size.max(1);
    }
    if max_align > 1 {
        running = (running + max_align - 1) & !(max_align - 1);
    }
    (offsets, running.max(1))
}

/// What a field holds, as a host reads it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FieldKind {
    /// A signed integer of the field's size.
    Int,
    /// An unsigned integer of the field's size.
    UInt,
    /// A float of the field's size.
    Float,
    Bool,
    /// A string the runtime made.
    Str,
    /// A pointer to an object of the named reference type.
    Object(String),
    /// A dynamic value: a box.
    Any,
    /// Any other pointer (a list, a closure, a host object's box).
    Pointer,
    /// Bytes the host has no reading for.
    Raw,
}

/// One field of a [`TypeDescriptor`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DescribedField {
    pub name: String,
    /// Byte offset from the object's start, past the header word.
    pub offset: u64,
    pub size: u64,
    pub kind: FieldKind,
}

/// A reference type as a host's heap is told about it: the global its
/// descriptor goes in, its name, size and fields.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeDescriptor {
    pub global: String,
    pub name: String,
    pub size: u64,
    pub fields: Vec<DescribedField>,
}
