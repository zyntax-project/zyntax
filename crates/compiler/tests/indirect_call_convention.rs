//! A call through a function pointer reaches a native `extern "C"`
//! function on every tier.
//!
//! A frontend hands compiled code the address of a host function as a
//! value, and compiled code calls it through a pointer whose function
//! type has the default calling convention. The callee is Rust or C
//! compiled for the platform's convention, so the default must be that
//! convention at an indirect call site on the interpreter, Cranelift
//! and LLVM alike: a pointer, an integer and a float must each arrive
//! in the register the platform puts them in.

#![cfg(feature = "cranelift-backend")]

use std::collections::HashSet;
use zyntax_compiler::hir::{
    CastOp, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirFunctionType, HirId,
    HirInstruction, HirModule, HirParam, HirTerminator, HirType, HirValue, HirValueKind,
    ParamAttributes,
};
use zyntax_typed_ast::InternedString;

/// The native target: each argument lands in its own decimal places.
extern "C" fn target(p: *const u8, n: i64, x: f64) -> i64 {
    p as i64 * 1_000_000 + n * 1000 + (x * 4.0) as i64
}

const POINTER: i64 = 16;
const INTEGER: i64 = 7;
const FLOAT: f64 = 2.5;
const EXPECTED: i64 = 16_007_010;

fn sig(params: Vec<HirType>, returns: Vec<HirType>) -> HirFunctionSignature {
    HirFunctionSignature {
        params: params
            .into_iter()
            .enumerate()
            .map(|(i, ty)| HirParam {
                id: HirId::new(),
                name: InternedString::new_global(&format!("p{i}")),
                ty,
                attributes: ParamAttributes::default(),
                ownership: Default::default(),
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
    }
}

fn add_value(func: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty,
            kind,
            uses: HashSet::new(),
            span: None,
        },
    );
    id
}

fn byte_ptr() -> HirType {
    HirType::Ptr(Box::new(HirType::U8))
}

/// The pointer's type: a function of the default convention.
fn target_ty() -> HirType {
    HirType::Function(Box::new(HirFunctionType {
        params: vec![byte_ptr(), HirType::I64, HirType::F64],
        returns: vec![HirType::I64],
        lifetime_params: vec![],
        is_variadic: false,
    }))
}

/// `def main(fp): i64 { return fp(16 as *u8, 7, 2.5) }`
fn build_module() -> (HirModule, HirId) {
    let mut f = HirFunction::new(
        InternedString::new_global("main"),
        sig(vec![target_ty()], vec![HirType::I64]),
    );
    let fp = add_value(&mut f, target_ty(), HirValueKind::Parameter(0));
    let raw = add_value(
        &mut f,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(POINTER)),
    );
    let p = add_value(&mut f, byte_ptr(), HirValueKind::Instruction);
    let n = add_value(
        &mut f,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(INTEGER)),
    );
    let x = add_value(
        &mut f,
        HirType::F64,
        HirValueKind::Constant(HirConstant::F64(FLOAT)),
    );
    let r = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let entry = f.entry_block;
    let blk = f.blocks.get_mut(&entry).unwrap();
    blk.instructions.push(HirInstruction::Cast {
        op: CastOp::IntToPtr,
        result: p,
        ty: byte_ptr(),
        operand: raw,
    });
    blk.instructions.push(HirInstruction::Call {
        result: Some(r),
        callee: HirCallable::Indirect(fp),
        args: vec![p, n, x],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    blk.terminator = HirTerminator::Return { values: vec![r] };
    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("indirect_convention"));
    module.functions.insert(id, f);
    (module, id)
}

fn target_address() -> *const u8 {
    target as *const u8
}

#[test]
fn the_interpreter_calls_a_native_pointer_with_the_platform_convention() {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    use zyntax_compiler::value::ZyntaxValue;

    let (module, _) = build_module();
    // A call through a pointer that is none of the module's functions
    // goes through a caller the baseline compiles for its shape.
    let backend = zyntax_compiler::cranelift_backend::CraneliftBackend::new().expect("backend");
    let shared =
        std::sync::Arc::new(zyntax_compiler::beadie_adapter::ZyntaxCraneliftBackend::new(backend));
    let mut interp = HirInterpreter::new();
    interp.set_native_bridge(
        Box::new(move |sig| shared.with_lock(|be| be.interp_thunk(sig).ok())),
        Box::new(|_| None),
        Box::new(|_| None),
    );
    let got = interp
        .call(
            &module,
            "main",
            vec![ZyntaxValue::Pointer(target_address() as *mut u8)],
        )
        .expect("run");
    assert_eq!(value_to_i64(&got), Some(EXPECTED));
}

#[test]
fn cranelift_calls_a_native_pointer_with_the_platform_convention() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    let (module, main_id) = build_module();
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend.get_function_ptr(main_id).expect("main compiled");
    let f: unsafe extern "C" fn(*const u8) -> i64 = unsafe { std::mem::transmute(ptr) };
    assert_eq!(unsafe { f(target_address()) }, EXPECTED);
}

#[cfg(feature = "llvm-backend")]
#[test]
fn llvm_calls_a_native_pointer_with_the_platform_convention() {
    use inkwell::context::Context;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;

    if zyntax_compiler::llvm_link::find_linker().is_err() {
        eprintln!("no system linker; skipping the LLVM leg");
        return;
    }
    let (module, main_id) = build_module();
    let context = Context::create();
    let mut backend = LLVMJitBackend::new(&context).expect("backend");
    backend.compile_module(&module).expect("compile");
    let ptr = backend
        .get_function_pointer(main_id)
        .expect("main compiled");
    let f: unsafe extern "C" fn(*const u8) -> i64 = unsafe { std::mem::transmute(ptr) };
    assert_eq!(unsafe { f(target_address()) }, EXPECTED);
}
