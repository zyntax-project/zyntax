//! Calling a compiled function through a raw pointer.
//!
//! A native call is made through one arity-specific trampoline per
//! argument count, with `i64` as the universal container that each
//! return type is reinterpreted out of.

use super::types::{NativeSignature, NativeType, RuntimeError, RuntimeResult};
use crate::value::ZyntaxValue;
use zyntax_compiler::zrtl::DynamicValue;

fn value_to_native(value: &ZyntaxValue, ty: NativeType) -> RuntimeResult<i64> {
    match (value, ty) {
        (ZyntaxValue::Int(n), NativeType::I32) => Ok(*n as i64),
        (ZyntaxValue::Int(n), NativeType::I64) => Ok(*n),
        (ZyntaxValue::Float(f), NativeType::F32) => Ok((*f as f32).to_bits() as i64),
        (ZyntaxValue::Float(f), NativeType::F64) => Ok(f.to_bits() as i64),
        (ZyntaxValue::Bool(b), NativeType::Bool) => Ok(if *b { 1 } else { 0 }),
        _ => Err(RuntimeError::Execution(format!(
            "Cannot convert {:?} to {:?}",
            value, ty
        ))),
    }
}

// `ZyntaxValue` ↔ `InterpValue` conversion is now expressed via
// `From` impls living in [`zyntax_compiler::hir_interp`] (where
// `InterpValue` is defined). The lossy table that used to live here
// is gone — both directions are lossless for primitives and use
// `Tuple` for aggregates; richer host marshalling (strings/maps/
// named structs) is the natural next extension.

fn native_to_value(raw: i64, ty: NativeType) -> RuntimeResult<ZyntaxValue> {
    Ok(match ty {
        NativeType::I32 => ZyntaxValue::Int(raw as i32 as i64),
        NativeType::I64 => ZyntaxValue::Int(raw),
        NativeType::F32 => ZyntaxValue::Float(f32::from_bits(raw as u32) as f64),
        NativeType::F64 => ZyntaxValue::Float(f64::from_bits(raw as u64)),
        NativeType::Bool => ZyntaxValue::Bool(raw != 0),
        NativeType::Void => ZyntaxValue::Null,
        NativeType::Ptr => ZyntaxValue::Int(raw),
    })
}

// Native call dispatch functions
// These use i64 as a universal container and reinterpret based on return type

unsafe fn call_0(ptr: *const u8, ret: NativeType) -> i64 {
    match ret {
        NativeType::I32 => {
            let f: extern "C" fn() -> i32 = std::mem::transmute(ptr);
            f() as i64
        }
        NativeType::I64 => {
            let f: extern "C" fn() -> i64 = std::mem::transmute(ptr);
            f()
        }
        NativeType::F32 => {
            let f: extern "C" fn() -> f32 = std::mem::transmute(ptr);
            f().to_bits() as i64
        }
        NativeType::F64 => {
            let f: extern "C" fn() -> f64 = std::mem::transmute(ptr);
            f().to_bits() as i64
        }
        NativeType::Bool => {
            let f: extern "C" fn() -> i8 = std::mem::transmute(ptr);
            f() as i64
        }
        NativeType::Void | NativeType::Ptr => {
            let f: extern "C" fn() = std::mem::transmute(ptr);
            f();
            0
        }
    }
}

unsafe fn call_1(ptr: *const u8, a0: i64, ret: NativeType) -> i64 {
    match ret {
        NativeType::I32 => {
            let f: extern "C" fn(i64) -> i32 = std::mem::transmute(ptr);
            f(a0) as i64
        }
        NativeType::I64 => {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(ptr);
            f(a0)
        }
        NativeType::F32 => {
            let f: extern "C" fn(i64) -> f32 = std::mem::transmute(ptr);
            f(a0).to_bits() as i64
        }
        NativeType::F64 => {
            let f: extern "C" fn(i64) -> f64 = std::mem::transmute(ptr);
            f(a0).to_bits() as i64
        }
        NativeType::Bool => {
            let f: extern "C" fn(i64) -> i8 = std::mem::transmute(ptr);
            f(a0) as i64
        }
        NativeType::Void | NativeType::Ptr => {
            let f: extern "C" fn(i64) = std::mem::transmute(ptr);
            f(a0);
            0
        }
    }
}

unsafe fn call_2(ptr: *const u8, a0: i64, a1: i64, ret: NativeType) -> i64 {
    match ret {
        NativeType::I32 => {
            let f: extern "C" fn(i64, i64) -> i32 = std::mem::transmute(ptr);
            f(a0, a1) as i64
        }
        NativeType::I64 => {
            let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(ptr);
            f(a0, a1)
        }
        NativeType::F32 => {
            let f: extern "C" fn(i64, i64) -> f32 = std::mem::transmute(ptr);
            f(a0, a1).to_bits() as i64
        }
        NativeType::F64 => {
            let f: extern "C" fn(i64, i64) -> f64 = std::mem::transmute(ptr);
            f(a0, a1).to_bits() as i64
        }
        NativeType::Bool => {
            let f: extern "C" fn(i64, i64) -> i8 = std::mem::transmute(ptr);
            f(a0, a1) as i64
        }
        NativeType::Void | NativeType::Ptr => {
            let f: extern "C" fn(i64, i64) = std::mem::transmute(ptr);
            f(a0, a1);
            0
        }
    }
}

unsafe fn call_3(ptr: *const u8, a0: i64, a1: i64, a2: i64, ret: NativeType) -> i64 {
    match ret {
        NativeType::I32 => {
            let f: extern "C" fn(i64, i64, i64) -> i32 = std::mem::transmute(ptr);
            f(a0, a1, a2) as i64
        }
        NativeType::I64 => {
            let f: extern "C" fn(i64, i64, i64) -> i64 = std::mem::transmute(ptr);
            f(a0, a1, a2)
        }
        NativeType::F32 => {
            let f: extern "C" fn(i64, i64, i64) -> f32 = std::mem::transmute(ptr);
            f(a0, a1, a2).to_bits() as i64
        }
        NativeType::F64 => {
            let f: extern "C" fn(i64, i64, i64) -> f64 = std::mem::transmute(ptr);
            f(a0, a1, a2).to_bits() as i64
        }
        NativeType::Bool => {
            let f: extern "C" fn(i64, i64, i64) -> i8 = std::mem::transmute(ptr);
            f(a0, a1, a2) as i64
        }
        NativeType::Void | NativeType::Ptr => {
            let f: extern "C" fn(i64, i64, i64) = std::mem::transmute(ptr);
            f(a0, a1, a2);
            0
        }
    }
}

unsafe fn call_4(ptr: *const u8, a0: i64, a1: i64, a2: i64, a3: i64, ret: NativeType) -> i64 {
    match ret {
        NativeType::I32 => {
            let f: extern "C" fn(i64, i64, i64, i64) -> i32 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3) as i64
        }
        NativeType::I64 => {
            let f: extern "C" fn(i64, i64, i64, i64) -> i64 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3)
        }
        NativeType::F32 => {
            let f: extern "C" fn(i64, i64, i64, i64) -> f32 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3).to_bits() as i64
        }
        NativeType::F64 => {
            let f: extern "C" fn(i64, i64, i64, i64) -> f64 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3).to_bits() as i64
        }
        NativeType::Bool => {
            let f: extern "C" fn(i64, i64, i64, i64) -> i8 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3) as i64
        }
        NativeType::Void | NativeType::Ptr => {
            let f: extern "C" fn(i64, i64, i64, i64) = std::mem::transmute(ptr);
            f(a0, a1, a2, a3);
            0
        }
    }
}

unsafe fn call_5(
    ptr: *const u8,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    ret: NativeType,
) -> i64 {
    match ret {
        NativeType::I32 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64) -> i32 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4) as i64
        }
        NativeType::I64 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64) -> i64 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4)
        }
        NativeType::F32 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64) -> f32 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4).to_bits() as i64
        }
        NativeType::F64 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64) -> f64 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4).to_bits() as i64
        }
        NativeType::Bool => {
            let f: extern "C" fn(i64, i64, i64, i64, i64) -> i8 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4) as i64
        }
        NativeType::Void | NativeType::Ptr => {
            let f: extern "C" fn(i64, i64, i64, i64, i64) = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4);
            0
        }
    }
}

unsafe fn call_6(
    ptr: *const u8,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    ret: NativeType,
) -> i64 {
    match ret {
        NativeType::I32 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i32 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5) as i64
        }
        NativeType::I64 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5)
        }
        NativeType::F32 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> f32 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5).to_bits() as i64
        }
        NativeType::F64 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> f64 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5).to_bits() as i64
        }
        NativeType::Bool => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i8 = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5) as i64
        }
        NativeType::Void | NativeType::Ptr => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64) = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5);
            0
        }
    }
}

unsafe fn call_7(
    ptr: *const u8,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
    ret: NativeType,
) -> i64 {
    match ret {
        NativeType::I32 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i32 =
                std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6) as i64
        }
        NativeType::I64 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64 =
                std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6)
        }
        NativeType::F32 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> f32 =
                std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6).to_bits() as i64
        }
        NativeType::F64 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> f64 =
                std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6).to_bits() as i64
        }
        NativeType::Bool => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i8 =
                std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6) as i64
        }
        NativeType::Void | NativeType::Ptr => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6);
            0
        }
    }
}

unsafe fn call_8(
    ptr: *const u8,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
    a7: i64,
    ret: NativeType,
) -> i64 {
    match ret {
        NativeType::I32 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i32 =
                std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6, a7) as i64
        }
        NativeType::I64 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
                std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6, a7)
        }
        NativeType::F32 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> f32 =
                std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6, a7).to_bits() as i64
        }
        NativeType::F64 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> f64 =
                std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6, a7).to_bits() as i64
        }
        NativeType::Bool => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i8 =
                std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6, a7) as i64
        }
        NativeType::Void | NativeType::Ptr => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) = std::mem::transmute(ptr);
            f(a0, a1, a2, a3, a4, a5, a6, a7);
            0
        }
    }
}

/// Call a function with dynamic values using a signature
///
/// This is the signature-based dispatch for async function calls.
/// Returns the raw pointer result (for Promise-returning async functions).
///
/// # Safety
/// The caller must ensure the function pointer has the correct signature.
pub(super) unsafe fn call_with_signature(
    ptr: *const u8,
    args: &[DynamicValue],
    signature: &NativeSignature,
) -> *const u8 {
    // Convert DynamicValue to i64 for the native call
    let native_args: Vec<i64> = args
        .iter()
        .zip(&signature.params)
        .map(|(arg, _ty)| dynamic_to_i64(arg))
        .collect();

    // For async functions, the return type is always a pointer (*Promise<T>)
    // We dispatch based on argument count
    match native_args.len() {
        0 => {
            let f: extern "C" fn() -> *const u8 = std::mem::transmute(ptr);
            f()
        }
        1 => {
            let f: extern "C" fn(i64) -> *const u8 = std::mem::transmute(ptr);
            f(native_args[0])
        }
        2 => {
            let f: extern "C" fn(i64, i64) -> *const u8 = std::mem::transmute(ptr);
            f(native_args[0], native_args[1])
        }
        3 => {
            let f: extern "C" fn(i64, i64, i64) -> *const u8 = std::mem::transmute(ptr);
            f(native_args[0], native_args[1], native_args[2])
        }
        4 => {
            let f: extern "C" fn(i64, i64, i64, i64) -> *const u8 = std::mem::transmute(ptr);
            f(
                native_args[0],
                native_args[1],
                native_args[2],
                native_args[3],
            )
        }
        5 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64) -> *const u8 = std::mem::transmute(ptr);
            f(
                native_args[0],
                native_args[1],
                native_args[2],
                native_args[3],
                native_args[4],
            )
        }
        6 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> *const u8 =
                std::mem::transmute(ptr);
            f(
                native_args[0],
                native_args[1],
                native_args[2],
                native_args[3],
                native_args[4],
                native_args[5],
            )
        }
        7 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> *const u8 =
                std::mem::transmute(ptr);
            f(
                native_args[0],
                native_args[1],
                native_args[2],
                native_args[3],
                native_args[4],
                native_args[5],
                native_args[6],
            )
        }
        8 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> *const u8 =
                std::mem::transmute(ptr);
            f(
                native_args[0],
                native_args[1],
                native_args[2],
                native_args[3],
                native_args[4],
                native_args[5],
                native_args[6],
                native_args[7],
            )
        }
        _ => {
            log::error!(
                "Unsupported argument count: {}. Maximum is 8.",
                native_args.len()
            );
            std::ptr::null()
        }
    }
}

/// Convert a DynamicValue to i64 for native calls
pub(super) fn dynamic_to_i64(value: &DynamicValue) -> i64 {
    // Try each primitive type accessor
    if let Some(i) = value.get_i32() {
        return i as i64;
    }
    if let Some(i) = value.get_i64() {
        return i;
    }
    if let Some(f) = value.get_f32() {
        return f.to_bits() as i64;
    }
    if let Some(f) = value.get_f64() {
        return f.to_bits() as i64;
    }
    if let Some(b) = value.get_bool() {
        return if b { 1 } else { 0 };
    }
    // For pointer types, just use the raw pointer value
    if !value.value_ptr.is_null() {
        return value.value_ptr as i64;
    }
    0
}

/// Call a native function with the given signature
///
/// # Safety
/// The caller must ensure the function pointer has the correct signature.
pub(super) unsafe fn call_native_with_signature(
    ptr: *const u8,
    args: &[ZyntaxValue],
    signature: &NativeSignature,
) -> RuntimeResult<ZyntaxValue> {
    // Convert arguments to native values on the stack
    // We use a union-like approach with i64 as the largest type
    let native_args: Vec<i64> = args
        .iter()
        .zip(&signature.params)
        .map(|(arg, ty)| value_to_native(arg, *ty))
        .collect::<Result<Vec<_>, _>>()?;

    // Dispatch based on argument count and return type
    // This generates the actual function call with proper ABI
    // Supports up to 8 arguments
    let result_i64 = match native_args.len() {
        0 => call_0(ptr, signature.ret),
        1 => call_1(ptr, native_args[0], signature.ret),
        2 => call_2(ptr, native_args[0], native_args[1], signature.ret),
        3 => call_3(
            ptr,
            native_args[0],
            native_args[1],
            native_args[2],
            signature.ret,
        ),
        4 => call_4(
            ptr,
            native_args[0],
            native_args[1],
            native_args[2],
            native_args[3],
            signature.ret,
        ),
        5 => call_5(
            ptr,
            native_args[0],
            native_args[1],
            native_args[2],
            native_args[3],
            native_args[4],
            signature.ret,
        ),
        6 => call_6(
            ptr,
            native_args[0],
            native_args[1],
            native_args[2],
            native_args[3],
            native_args[4],
            native_args[5],
            signature.ret,
        ),
        7 => call_7(
            ptr,
            native_args[0],
            native_args[1],
            native_args[2],
            native_args[3],
            native_args[4],
            native_args[5],
            native_args[6],
            signature.ret,
        ),
        8 => call_8(
            ptr,
            native_args[0],
            native_args[1],
            native_args[2],
            native_args[3],
            native_args[4],
            native_args[5],
            native_args[6],
            native_args[7],
            signature.ret,
        ),
        n => {
            return Err(RuntimeError::Execution(format!(
                "Unsupported argument count: {}. Maximum is 8.",
                n
            )))
        }
    };

    // Convert result back to ZyntaxValue
    native_to_value(result_i64, signature.ret)
}

/// Call a function pointer with dynamic arguments
///
/// Supports up to 8 arguments. For more arguments, consider using
/// a struct-based calling convention or libffi.
///
/// # Safety
///
/// The caller must ensure:
/// - `ptr` is a valid function pointer with the correct signature
/// - `args` contains the correct number and types of arguments
pub(super) unsafe fn call_dynamic_function(
    ptr: *const u8,
    args: &[DynamicValue],
) -> RuntimeResult<DynamicValue> {
    let result = match args.len() {
        0 => {
            let f: extern "C" fn() -> DynamicValue = std::mem::transmute(ptr);
            f()
        }
        1 => {
            let f: extern "C" fn(DynamicValue) -> DynamicValue = std::mem::transmute(ptr);
            f(args[0].clone())
        }
        2 => {
            let f: extern "C" fn(DynamicValue, DynamicValue) -> DynamicValue =
                std::mem::transmute(ptr);
            f(args[0].clone(), args[1].clone())
        }
        3 => {
            let f: extern "C" fn(DynamicValue, DynamicValue, DynamicValue) -> DynamicValue =
                std::mem::transmute(ptr);
            f(args[0].clone(), args[1].clone(), args[2].clone())
        }
        4 => {
            let f: extern "C" fn(
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
            ) -> DynamicValue = std::mem::transmute(ptr);
            f(
                args[0].clone(),
                args[1].clone(),
                args[2].clone(),
                args[3].clone(),
            )
        }
        5 => {
            let f: extern "C" fn(
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
            ) -> DynamicValue = std::mem::transmute(ptr);
            f(
                args[0].clone(),
                args[1].clone(),
                args[2].clone(),
                args[3].clone(),
                args[4].clone(),
            )
        }
        6 => {
            let f: extern "C" fn(
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
            ) -> DynamicValue = std::mem::transmute(ptr);
            f(
                args[0].clone(),
                args[1].clone(),
                args[2].clone(),
                args[3].clone(),
                args[4].clone(),
                args[5].clone(),
            )
        }
        7 => {
            let f: extern "C" fn(
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
            ) -> DynamicValue = std::mem::transmute(ptr);
            f(
                args[0].clone(),
                args[1].clone(),
                args[2].clone(),
                args[3].clone(),
                args[4].clone(),
                args[5].clone(),
                args[6].clone(),
            )
        }
        8 => {
            let f: extern "C" fn(
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
                DynamicValue,
            ) -> DynamicValue = std::mem::transmute(ptr);
            f(
                args[0].clone(),
                args[1].clone(),
                args[2].clone(),
                args[3].clone(),
                args[4].clone(),
                args[5].clone(),
                args[6].clone(),
                args[7].clone(),
            )
        }
        n => {
            return Err(RuntimeError::Execution(
                format!("Functions with {} arguments not supported (max 8). Consider using a struct-based calling convention.", n)
            ));
        }
    };
    Ok(result)
}
