//! The calling convention of a compiled function, from its HIR signature.
//!
//! Every backend that emits native code, and every call site that
//! reaches such code from another tier, derives the shape of a call
//! from here, so that code one tier compiled can be entered by code
//! another tier compiled. The rules are the ones the Cranelift backend
//! has always used, since it represents an aggregate value as the
//! address of its bytes:
//!
//! * a scalar, a pointer, a vector and a function pointer travel as
//!   themselves;
//! * a `Void` parameter travels as a byte nothing reads, so the
//!   parameters after it keep their registers;
//! * a struct of one scalar field travels as that scalar;
//! * any other struct, and an array, travels as the address of its
//!   storage;
//! * a struct returned by value is written through a destination the
//!   caller provides as a leading parameter, and that address is also
//!   the return value; a growable list header is returned as the
//!   address it already lives at; a function whose address is taken, or
//!   an extern, returns a struct directly, since its callers cannot be
//!   made to pass a destination.
//!
//! Every compiled function takes the target's C convention, whatever
//! convention its HIR names. Its address can reach an indirect call, a
//! call cell, the interpreter's thunk or a host calling it as
//! `extern "C"`, and none of those knows which function it calls, so a
//! convention chosen per function is one some caller gets wrong.

use crate::hir::{HirFunction, HirStructType, HirType};

/// How one parameter or return value travels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// As the value itself, in a register.
    Direct,
    /// As the address of its storage.
    Pointer,
}

/// A function's convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionAbi {
    /// A leading parameter holding the caller's storage for the returned
    /// struct, which the function fills and returns; `None` when the
    /// function returns nothing that way.
    pub destination: Option<HirType>,
    /// How each declared parameter travels, in declaration order.
    pub params: Vec<Pass>,
    /// How each declared return value travels; a `Void` return is left
    /// out.
    pub returns: Vec<Pass>,
}

impl FunctionAbi {
    /// Whether every parameter and return value is a register value:
    /// a signature any tier can enter from any other without agreeing
    /// on storage.
    pub fn is_scalar(&self) -> bool {
        self.destination.is_none()
            && self.params.iter().all(|p| *p == Pass::Direct)
            && self.returns.iter().all(|p| *p == Pass::Direct)
    }
}

/// The convention of `function`. `address_taken` says whether the
/// function's address is observable, which rules out a destination
/// return.
pub fn function_abi(function: &HirFunction, address_taken: bool) -> FunctionAbi {
    let destination = if address_taken {
        None
    } else {
        destination_return_type(function).cloned()
    };
    let params = function
        .signature
        .params
        .iter()
        .map(|p| pass_of(&p.ty))
        .collect();
    let returns = function
        .signature
        .returns
        .iter()
        .filter(|t| **t != HirType::Void)
        .map(pass_of)
        .collect();
    FunctionAbi {
        destination,
        params,
        returns,
    }
}

/// Whether LLVM can compile `function` to the convention above, so its
/// entry can stand in a call cell Cranelift callers read.
///
/// A struct or array parameter arrives as the address of the caller's
/// storage, which the body reads and writes in place; a struct of one
/// scalar field arrives as that scalar; a struct returned by value is
/// written through the destination. A union, several return values, an
/// array returned by value, and a struct returned by value from a
/// function whose address is taken (which returns it directly) are
/// refused, as is any aggregate whose fields the two backends might lay
/// out differently.
pub fn llvm_entry_abi_supported(function: &HirFunction, address_taken: bool) -> bool {
    let abi = function_abi(function, address_taken);
    let returns: Vec<&HirType> = function
        .signature
        .returns
        .iter()
        .filter(|t| **t != HirType::Void)
        .collect();
    let param = |ty: &HirType, pass: &Pass| match (ty, pass) {
        (HirType::Struct(s), Pass::Direct) => struct_carried_as_its_field(s).is_some(),
        (HirType::Struct(_) | HirType::Array(_, _), Pass::Pointer) => laid_out_alike(ty),
        (HirType::Union(_) | HirType::Array(_, _), _) => false,
        _ => true,
    };
    let ret = |ty: &HirType, pass: &Pass| match (ty, pass) {
        (HirType::Struct(s), Pass::Direct) => struct_carried_as_its_field(s).is_some(),
        (HirType::Struct(s), Pass::Pointer) if is_growable_list_header(s) => true,
        (HirType::Struct(_), Pass::Pointer) => {
            abi.destination.as_ref() == Some(ty) && laid_out_alike(ty)
        }
        (HirType::Union(_) | HirType::Array(_, _), _) => false,
        _ => true,
    };
    returns.len() <= 1
        && returns.iter().zip(&abi.returns).all(|(t, p)| ret(t, p))
        && function
            .signature
            .params
            .iter()
            .zip(&abi.params)
            .all(|(p, pass)| param(&p.ty, pass))
}

/// Whether both backends put the bytes of a value of `ty` in the same
/// places: fields at their natural alignment, nested structs and arrays
/// inline, every leaf a scalar or an address.
fn laid_out_alike(ty: &HirType) -> bool {
    match ty {
        HirType::Struct(s) => s.fields.iter().all(laid_out_alike),
        HirType::Array(elem, _) => laid_out_alike(elem),
        HirType::Bool
        | HirType::I8
        | HirType::I16
        | HirType::I32
        | HirType::I64
        | HirType::I128
        | HirType::U8
        | HirType::U16
        | HirType::U32
        | HirType::U64
        | HirType::U128
        | HirType::F32
        | HirType::F64
        | HirType::USize
        | HirType::ISize
        | HirType::Ptr(_)
        | HirType::Ref { .. }
        | HirType::Function(_)
        | HirType::Opaque(_) => true,
        _ => false,
    }
}

/// How a value of `ty` travels.
pub fn pass_of(ty: &HirType) -> Pass {
    match ty {
        HirType::Struct(s) if struct_carried_as_its_field(s).is_some() => Pass::Direct,
        HirType::Struct(_) | HirType::Array(_, _) => Pass::Pointer,
        _ => Pass::Direct,
    }
}

/// The field a single-field struct is carried as, when it is carried as
/// its field rather than by address.
///
/// A struct wrapping one scalar has the same shape as that scalar, so it
/// travels in a register. Anything else needs memory.
pub fn struct_carried_as_its_field(struct_ty: &HirStructType) -> Option<&HirType> {
    if struct_ty.fields.len() != 1 {
        return None;
    }
    let field = struct_ty.fields.first()?;
    matches!(
        field,
        HirType::I8
            | HirType::I16
            | HirType::I32
            | HirType::I64
            | HirType::I128
            | HirType::U8
            | HirType::U16
            | HirType::U32
            | HirType::U64
            | HirType::U128
            | HirType::F32
            | HirType::F64
            | HirType::Bool
    )
    .then_some(field)
}

/// The struct a function hands back through a destination its caller
/// provides, if it hands one back that way.
///
/// The only memory a function can put a struct's bytes in on its own is
/// its frame, which its caller outlives, so the caller supplies the
/// memory and the function writes through it. Anything with a shape
/// that fits in a register is returned in one and is not covered here.
///
/// Only structs. `Array` and `Union` travel as pointers too, but an
/// array-typed value is not reliably the frame memory a struct's is:
/// copying one that already points at a buffer would hand back a copy
/// where the buffer itself was meant. A growable list is excluded for
/// the same reason: its header lives on the heap and is the list's
/// identity, so its address is what a function hands back.
pub fn destination_return_type(function: &HirFunction) -> Option<&HirType> {
    if function.is_external {
        return None;
    }
    if function.signature.returns.len() != 1 {
        return None;
    }
    match function.signature.returns.first()? {
        ret @ HirType::Struct(s)
            if struct_carried_as_its_field(s).is_none() && !is_growable_list_header(s) =>
        {
            Some(ret)
        }
        _ => None,
    }
}

/// The `{data, len, capacity}` header of a growable list, by its shape.
pub fn is_growable_list_header(struct_ty: &HirStructType) -> bool {
    struct_ty
        .name
        .and_then(|n| n.resolve_global())
        .is_some_and(|n| n == "List" || n == "Array")
        && struct_ty.fields.len() == 3
        && struct_ty
            .fields
            .iter()
            .all(|f| matches!(f, HirType::I64 | HirType::U64 | HirType::Ptr(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirFunctionSignature, HirParam, ParamOwnership};
    use zyntax_typed_ast::InternedString;

    fn function(params: Vec<HirType>, returns: Vec<HirType>) -> HirFunction {
        let signature = HirFunctionSignature {
            params: params
                .into_iter()
                .map(|ty| HirParam {
                    id: crate::hir::HirId::new(),
                    name: InternedString::new_global("p"),
                    ty,
                    attributes: Default::default(),
                    ownership: ParamOwnership::default(),
                })
                .collect(),
            returns,
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        };
        HirFunction::new(InternedString::new_global("f"), signature)
    }

    fn vec3() -> HirType {
        HirType::Struct(HirStructType {
            name: Some(InternedString::new_global("Vec3")),
            fields: vec![HirType::F64, HirType::F64, HirType::F64],
            packed: false,
        })
    }

    fn list() -> HirType {
        HirType::Struct(HirStructType {
            name: Some(InternedString::new_global("List")),
            fields: vec![HirType::I64, HirType::I64, HirType::I64],
            packed: false,
        })
    }

    #[test]
    fn a_struct_travels_by_pointer_and_returns_through_a_destination() {
        let f = function(vec![vec3(), HirType::I64], vec![vec3()]);
        let abi = function_abi(&f, false);
        assert_eq!(abi.params, vec![Pass::Pointer, Pass::Direct]);
        assert_eq!(abi.returns, vec![Pass::Pointer]);
        assert_eq!(abi.destination, Some(vec3()));
        assert!(!abi.is_scalar());
        // Observable address: the callers cannot pass a destination.
        assert_eq!(function_abi(&f, true).destination, None);
    }

    #[test]
    fn a_list_header_returns_as_its_own_address() {
        let f = function(vec![list()], vec![list()]);
        let abi = function_abi(&f, false);
        assert_eq!(abi.params, vec![Pass::Pointer]);
        assert_eq!(abi.returns, vec![Pass::Pointer]);
        assert_eq!(abi.destination, None);
        // An address in, an address out: the LLVM entry can take it.
        assert!(llvm_entry_abi_supported(&f, false));
        // A struct returned as a value goes through the destination.
        let f = function(vec![list()], vec![vec3()]);
        assert!(llvm_entry_abi_supported(&f, false));
        // Unless the address is taken, when it would come back directly.
        assert!(!llvm_entry_abi_supported(&f, true));
    }

    #[test]
    fn llvm_entry_takes_aggregates_by_address_and_refuses_unions() {
        let f = function(vec![list(), HirType::I64], vec![HirType::I64]);
        assert!(llvm_entry_abi_supported(&f, false));
        let f = function(vec![vec3(), HirType::I64], vec![vec3()]);
        assert!(llvm_entry_abi_supported(&f, false));
        let f = function(vec![HirType::Array(Box::new(HirType::F64), 4)], vec![]);
        assert!(llvm_entry_abi_supported(&f, false));
        let wrapped = HirType::Struct(HirStructType {
            name: None,
            fields: vec![HirType::I64],
            packed: false,
        });
        let f = function(vec![wrapped.clone()], vec![wrapped]);
        assert!(llvm_entry_abi_supported(&f, false));
        let union = HirType::Union(Box::new(crate::hir::HirUnionType {
            name: None,
            variants: vec![],
            discriminant_type: Box::new(HirType::U8),
            is_c_union: false,
        }));
        let f = function(vec![union.clone()], vec![HirType::I64]);
        assert!(!llvm_entry_abi_supported(&f, false));
        let holding_union = HirType::Struct(HirStructType {
            name: None,
            fields: vec![HirType::I64, union],
            packed: false,
        });
        let f = function(vec![holding_union], vec![HirType::I64]);
        assert!(!llvm_entry_abi_supported(&f, false));
        let f = function(vec![HirType::I64], vec![HirType::I64, HirType::I64]);
        assert!(!llvm_entry_abi_supported(&f, false));
    }

    #[test]
    fn scalars_and_a_wrapped_scalar_are_direct() {
        let wrapped = HirType::Struct(HirStructType {
            name: None,
            fields: vec![HirType::I64],
            packed: false,
        });
        let f = function(
            vec![
                HirType::I64,
                HirType::F64,
                wrapped,
                HirType::Ptr(Box::new(HirType::U8)),
            ],
            vec![HirType::I64],
        );
        let abi = function_abi(&f, false);
        assert!(abi.is_scalar());
        assert_eq!(abi.params.len(), 4);
    }
}
