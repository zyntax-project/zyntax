//! A `Void` parameter keeps the parameters after it where the other
//! tier looks for them.
//!
//! Cranelift passes a `Void` as a byte in its own register. An LLVM
//! entry standing in a call cell that Cranelift callers read, and an
//! LLVM call through a cell holding Cranelift's code, must pass it the
//! same way, or every later argument arrives in the register of the
//! one before it.

#![cfg(all(feature = "cranelift-backend", feature = "llvm-backend"))]

use std::collections::HashSet;
use std::sync::Arc;
use zyntax_compiler::hir::{
    BinaryOp, HirBlock, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirId,
    HirInstruction, HirModule, HirParam, HirTerminator, HirType, HirValue, HirValueKind,
    ParamAttributes,
};
use zyntax_typed_ast::InternedString;

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

fn konst(func: &mut HirFunction, v: i64) -> HirId {
    add_value(
        func,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(v)),
    )
}

fn body(func: &mut HirFunction) -> &mut HirBlock {
    let entry = func.entry_block;
    func.blocks.get_mut(&entry).unwrap()
}

/// `def pick(a: i64, u: Void, b: i64, c: i64): i64 { return a*100 + b*10 + c }`
fn build_pick() -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global("pick"),
        sig(
            vec![HirType::I64, HirType::Void, HirType::I64, HirType::I64],
            vec![HirType::I64],
        ),
    );
    let a = add_value(&mut f, HirType::I64, HirValueKind::Parameter(0));
    let b = add_value(&mut f, HirType::I64, HirValueKind::Parameter(2));
    let c = add_value(&mut f, HirType::I64, HirValueKind::Parameter(3));
    let hundred = konst(&mut f, 100);
    let ten = konst(&mut f, 10);
    let v: Vec<HirId> = (0..4)
        .map(|_| add_value(&mut f, HirType::I64, HirValueKind::Instruction))
        .collect();
    let arith = |op, result, left, right| HirInstruction::Binary {
        op,
        result,
        ty: HirType::I64,
        left,
        right,
    };
    let blk = body(&mut f);
    blk.instructions.extend([
        arith(BinaryOp::Mul, v[0], a, hundred),
        arith(BinaryOp::Mul, v[1], b, ten),
        arith(BinaryOp::Add, v[2], v[0], v[1]),
        arith(BinaryOp::Add, v[3], v[2], c),
    ]);
    blk.terminator = HirTerminator::Return { values: vec![v[3]] };
    f
}

/// `def pick_main(): i64 { return pick(1, (), 2, 3) }`
fn build_pick_main(pick_id: HirId) -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global("pick_main"),
        sig(vec![], vec![HirType::I64]),
    );
    let args: Vec<HirId> = [1, 2, 3].into_iter().map(|v| konst(&mut f, v)).collect();
    let unit = add_value(&mut f, HirType::Void, HirValueKind::Undef);
    let r = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let blk = body(&mut f);
    blk.instructions.push(HirInstruction::Call {
        result: Some(r),
        callee: HirCallable::Function(pick_id),
        args: vec![args[0], unit, args[1], args[2]],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    blk.terminator = HirTerminator::Return { values: vec![r] };
    f
}

const PICK_EXPECTED: i64 = 123;

fn pick_module() -> (HirModule, HirFunction, HirFunction) {
    let pick = build_pick();
    let main = build_pick_main(pick.id);
    let mut module = HirModule::new(InternedString::new_global("void_param"));
    module.functions.insert(pick.id, pick.clone());
    module.functions.insert(main.id, main.clone());
    (module, pick, main)
}

/// One call cell, entered first by Cranelift's code for `pick` and then
/// by LLVM's, from the same Cranelift caller.
#[test]
fn llvm_entry_takes_the_parameters_after_a_void_where_cranelift_passes_them() {
    use inkwell::context::Context;
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;

    let (module, pick, main) = pick_module();
    let mut cranelift = CraneliftBackend::new().expect("backend");
    cranelift.set_reloadable_calls(true);
    cranelift.compile_module(&module).expect("compile");
    cranelift.finalize_definitions().expect("finalize");
    let ptr = cranelift.get_function_ptr(main.id).expect("main compiled");
    let run: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    assert_eq!(unsafe { run() }, PICK_EXPECTED);

    let context = Context::create();
    let mut llvm = LLVMJitBackend::new(&context).expect("backend");
    llvm.set_use_mcjit(true);
    llvm.set_module_context(Arc::new(module.clone()));
    llvm.set_cross_tier_links(cranelift.reload_key(), Arc::new(|_| None));
    llvm.compile_function(pick.id, &pick)
        .expect("LLVM compiles the entry");
    let entry = llvm.get_function_pointer(pick.id).expect("pick compiled");
    cranelift.publish_call_target(pick.id, entry as usize);
    assert_eq!(unsafe { run() }, PICK_EXPECTED);
}

/// An LLVM entry calling Cranelift's `pick` through its cell.
#[test]
fn llvm_passes_the_parameters_after_a_void_where_cranelift_takes_them() {
    use inkwell::context::Context;
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;

    let (module, _pick, main) = pick_module();
    let mut cranelift = CraneliftBackend::new().expect("backend");
    cranelift.set_reloadable_calls(true);
    cranelift.compile_module(&module).expect("compile");
    cranelift.finalize_definitions().expect("finalize");

    let context = Context::create();
    let mut llvm = LLVMJitBackend::new(&context).expect("backend");
    llvm.set_use_mcjit(true);
    llvm.set_module_context(Arc::new(module.clone()));
    llvm.set_cross_tier_links(cranelift.reload_key(), Arc::new(|_| None));
    llvm.compile_function(main.id, &main)
        .expect("LLVM compiles the caller");
    let entry = llvm.get_function_pointer(main.id).expect("main compiled");
    let run: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(entry) };
    assert_eq!(unsafe { run() }, PICK_EXPECTED);
}
