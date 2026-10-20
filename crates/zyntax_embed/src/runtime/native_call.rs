//! Calling a compiled function through a raw pointer.
//!
//! Floating-point arguments use the shared typed ABI bridge. Integer-only
//! arguments use arity-specific trampolines with a typed return register.

use super::types::{NativeSignature, NativeType, RuntimeError, RuntimeResult};
use crate::value::ZyntaxValue;
use zyntax_compiler::zrtl::DynamicValue;

fn value_to_native(value: &ZyntaxValue, ty: NativeType) -> RuntimeResult<i64> {
    match (value, ty) {
        (ZyntaxValue::Int(n), NativeType::I32) => Ok(*n as i32 as i64),
        (ZyntaxValue::Int(n), NativeType::I64 | NativeType::Ptr) => Ok(*n),
        (ZyntaxValue::Pointer(p), NativeType::Ptr) => Ok(*p as i64),
        (ZyntaxValue::Null, NativeType::Ptr) => Ok(0),
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
        NativeType::I64 | NativeType::Ptr => {
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
        NativeType::Void => {
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
        NativeType::I64 | NativeType::Ptr => {
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
        NativeType::Void => {
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
        NativeType::I64 | NativeType::Ptr => {
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
        NativeType::Void => {
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
        NativeType::I64 | NativeType::Ptr => {
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
        NativeType::Void => {
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
        NativeType::I64 | NativeType::Ptr => {
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
        NativeType::Void => {
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
        NativeType::I64 | NativeType::Ptr => {
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
        NativeType::Void => {
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
        NativeType::I64 | NativeType::Ptr => {
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
        NativeType::Void => {
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
        NativeType::I64 | NativeType::Ptr => {
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
        NativeType::Void => {
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
        NativeType::I64 | NativeType::Ptr => {
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
        NativeType::Void => {
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
    if args.len() != signature.params.len() || args.len() > 8 {
        return Err(RuntimeError::Execution(format!(
            "Native call expects {} arguments (maximum 8), got {}",
            signature.params.len(),
            args.len()
        )));
    }
    if ptr.is_null() {
        return Err(RuntimeError::Execution(
            "Null native function pointer".into(),
        ));
    }
    if signature.params.contains(&NativeType::F32) {
        return Err(RuntimeError::Execution(
            "Native host ABI with f32 parameters is unsupported".into(),
        ));
    }
    // Validate every value before selecting a trampoline.
    let native_args: Vec<i64> = args
        .iter()
        .zip(&signature.params)
        .map(|(arg, ty)| value_to_native(arg, *ty))
        .collect::<Result<Vec<_>, _>>()?;

    if signature.params.contains(&NativeType::F64) {
        use zyntax_compiler::zrtl::{MAX_PARAMS, TypeTag, ZrtlSigFlags, ZrtlSymbolSig};
        let mut sig = ZrtlSymbolSig {
            param_count: args.len() as u8,
            flags: ZrtlSigFlags::NONE,
            return_type: match signature.ret {
                NativeType::F32 => TypeTag::F32,
                NativeType::F64 => TypeTag::F64,
                NativeType::Bool => TypeTag::BOOL,
                NativeType::Void => TypeTag::VOID,
                _ => TypeTag::I64,
            },
            params: [TypeTag::VOID; MAX_PARAMS],
        };
        let values: Vec<_> = native_args
            .iter()
            .zip(&signature.params)
            .enumerate()
            .map(|(idx, (raw, ty))| {
                if *ty == NativeType::F64 {
                    sig.params[idx] = TypeTag::F64;
                    ZyntaxValue::Float(f64::from_bits(*raw as u64))
                } else {
                    sig.params[idx] = TypeTag::I64;
                    ZyntaxValue::Int(*raw)
                }
            })
            .collect();
        // SAFETY: the caller supplies the signature; the bridge checks its shape
        // and sends each validated scalar through its native register class.
        let result = unsafe { zyntax_compiler::hir_interp::call_scalar_native(ptr, &values, &sig) }
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;
        return Ok(match result {
            ZyntaxValue::Int(raw) => native_to_value(raw, signature.ret)?,
            ZyntaxValue::F32(value) => ZyntaxValue::Float(value as f64),
            ZyntaxValue::Void => ZyntaxValue::Null,
            value => value,
        });
    }

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
            )));
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
            return Err(RuntimeError::Execution(format!(
                "Functions with {} arguments not supported (max 8). Consider using a struct-based calling convention.",
                n
            )));
        }
    };
    Ok(result)
}

#[cfg(test)]
mod scalar_abi_tests {
    use super::*;

    fn invoke(
        ptr: *const u8,
        params: &[NativeType],
        ret: NativeType,
        args: &[ZyntaxValue],
    ) -> ZyntaxValue {
        // SAFETY: each test pairs its C function with its declared scalar signature.
        unsafe { call_native_with_signature(ptr, args, &NativeSignature::new(params, ret)) }
            .unwrap()
    }

    #[test]
    fn floating_arguments_and_returns_use_float_registers() {
        extern "C" fn scale(x: f64) -> f64 {
            x * 10.0 + 2.0
        }
        extern "C" fn narrow(x: f64) -> f32 {
            x as f32 + 0.5
        }
        assert_eq!(
            invoke(
                scale as *const u8,
                &[NativeType::F64],
                NativeType::F64,
                &[ZyntaxValue::Float(5.0)]
            ),
            ZyntaxValue::Float(52.0)
        );
        assert_eq!(
            invoke(
                narrow as *const u8,
                &[NativeType::F64],
                NativeType::F32,
                &[ZyntaxValue::Float(3.0)]
            ),
            ZyntaxValue::Float(3.5)
        );
    }

    #[test]
    fn mixed_arguments_preserve_order_within_each_register_class() {
        extern "C" fn interleaved(a: f64, b: i64, c: f64, d: i64) -> f64 {
            a * 1000.0 + b as f64 * 100.0 + c * 10.0 + d as f64
        }
        extern "C" fn reversed(a: i64, b: f64) -> i32 {
            a as i32 - b as i32
        }
        assert_eq!(
            invoke(
                interleaved as *const u8,
                &[
                    NativeType::F64,
                    NativeType::I64,
                    NativeType::F64,
                    NativeType::I64
                ],
                NativeType::F64,
                &[
                    ZyntaxValue::Float(1.0),
                    ZyntaxValue::Int(2),
                    ZyntaxValue::Float(3.0),
                    ZyntaxValue::Int(4)
                ]
            ),
            ZyntaxValue::Float(1234.0)
        );
        assert_eq!(
            invoke(
                reversed as *const u8,
                &[NativeType::I64, NativeType::F64],
                NativeType::I32,
                &[ZyntaxValue::Int(4), ZyntaxValue::Float(9.0)]
            ),
            ZyntaxValue::Int(-5)
        );
    }

    #[test]
    fn pointer_results_and_arguments_survive_both_paths() {
        extern "C" fn identity(p: *mut u8) -> *mut u8 {
            p
        }
        extern "C" fn offset(x: f64, p: *mut u8) -> *mut u8 {
            p.wrapping_add(x as usize)
        }
        let mut bytes = [0u8; 8];
        let p = bytes.as_mut_ptr();
        assert_eq!(
            invoke(
                identity as *const u8,
                &[NativeType::Ptr],
                NativeType::Ptr,
                &[ZyntaxValue::Pointer(p)]
            ),
            ZyntaxValue::Int(p as i64)
        );
        assert_eq!(
            invoke(
                offset as *const u8,
                &[NativeType::F64, NativeType::Ptr],
                NativeType::Ptr,
                &[ZyntaxValue::Float(3.0), ZyntaxValue::Int(p as i64)]
            ),
            ZyntaxValue::Int(p.wrapping_add(3) as i64)
        );
        assert_eq!(
            invoke(
                identity as *const u8,
                &[NativeType::Ptr],
                NativeType::Ptr,
                &[ZyntaxValue::Null]
            ),
            ZyntaxValue::Int(0)
        );
    }

    #[test]
    fn floating_arguments_support_boolean_and_void_results() {
        extern "C" fn positive(x: f64) -> bool {
            x > 0.0
        }
        extern "C" fn store(x: f64, p: *mut f64) {
            // SAFETY: the test passes a live, aligned, exclusively borrowed f64.
            unsafe {
                *p = x;
            }
        }
        for (input, expected) in [(2.0, true), (-2.0, false)] {
            assert_eq!(
                invoke(
                    positive as *const u8,
                    &[NativeType::F64],
                    NativeType::Bool,
                    &[ZyntaxValue::Float(input)]
                ),
                ZyntaxValue::Bool(expected)
            );
        }
        let mut output = 0.0;
        assert_eq!(
            invoke(
                store as *const u8,
                &[NativeType::F64, NativeType::Ptr],
                NativeType::Void,
                &[
                    ZyntaxValue::Float(12.5),
                    ZyntaxValue::Pointer((&raw mut output).cast())
                ]
            ),
            ZyntaxValue::Null
        );
        assert_eq!(output, 12.5);
    }

    #[test]
    fn invalid_shapes_and_values_are_rejected_before_calling() {
        extern "C" fn unreachable() -> i64 {
            panic!("must reject before native dispatch")
        }
        let ptr = unreachable as *const u8;
        for (params, args) in [
            (vec![NativeType::I64], vec![]),
            (vec![], vec![ZyntaxValue::Int(1)]),
            (vec![NativeType::F64], vec![ZyntaxValue::Bool(true)]),
            (vec![NativeType::F32], vec![ZyntaxValue::Float(1.0)]),
            (vec![NativeType::Void], vec![ZyntaxValue::Null]),
            (vec![NativeType::I64; 9], vec![ZyntaxValue::Int(0); 9]),
        ] {
            assert!(
                unsafe {
                    call_native_with_signature(
                        ptr,
                        &args,
                        &NativeSignature::new(&params, NativeType::I64),
                    )
                }
                .is_err()
            );
        }
        assert!(
            unsafe {
                call_native_with_signature(
                    std::ptr::null(),
                    &[],
                    &NativeSignature::new(&[], NativeType::I64),
                )
            }
            .is_err()
        );
        #[cfg(not(target_os = "windows"))]
        assert!(
            unsafe {
                call_native_with_signature(
                    ptr,
                    &vec![ZyntaxValue::Float(0.0); 5],
                    &NativeSignature::new(&[NativeType::F64; 5], NativeType::I64),
                )
            }
            .is_err()
        );
    }
}
