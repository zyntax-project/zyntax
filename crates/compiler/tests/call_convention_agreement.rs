#![cfg(feature = "cranelift-backend")]

//! A call reaches a compiled function with the convention the function
//! was compiled with, however the call finds it: a function value made
//! from it, its address taken by reference, or its call cell. The HIR
//! may ask for `Fast`; what the callee and every caller use is one
//! convention either way.
//!
//! A debug build of the backend refuses a call whose known callee is
//! declared with another convention, so compiling these modules is most
//! of the check; running them is the rest.

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::*;
use zyntax_typed_ast::{InternedString, arena::AstArena};

fn name(s: &str) -> InternedString {
    AstArena::new().intern_string(s)
}

fn signature(params: &[(&str, HirType)], ret: HirType) -> HirFunctionSignature {
    HirFunctionSignature {
        params: params
            .iter()
            .map(|(n, ty)| HirParam {
                id: HirId::new(),
                name: name(n),
                ty: ty.clone(),
                attributes: ParamAttributes::default(),
                ownership: Default::default(),
            })
            .collect(),
        returns: vec![ret],
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

/// `step(acc, i) = acc + i`, asking for the `Fast` convention as a
/// lambda's HIR may.
fn step() -> HirFunction {
    let mut f = HirFunction::new(
        name("step"),
        signature(&[("acc", HirType::I64), ("i", HirType::I64)], HirType::I64),
    );
    f.calling_convention = CallingConvention::Fast;
    let acc = f.create_value(HirType::I64, HirValueKind::Parameter(0));
    let i = f.create_value(HirType::I64, HirValueKind::Parameter(1));
    let sum = f.create_value(HirType::I64, HirValueKind::Instruction);
    let entry = f.blocks.get_mut(&f.entry_block).unwrap();
    entry.add_instruction(HirInstruction::Binary {
        op: BinaryOp::Add,
        result: sum,
        ty: HirType::I64,
        left: acc,
        right: i,
    });
    entry.set_terminator(HirTerminator::Return { values: vec![sum] });
    f
}

/// How `main` comes by the function it calls.
#[derive(Clone, Copy)]
enum Made {
    /// A closure made from it, as a lambda is.
    Closure,
    /// Its address, taken by reference.
    Reference,
}

/// `main() = f(40, 2)` where `f` is `step`, made as `made` says and
/// called through the pointer.
fn main_calling(step: HirId, made: Made) -> HirFunction {
    let mut f = HirFunction::new(name("main"), signature(&[], HirType::I64));
    f.calling_convention = CallingConvention::C;
    let fn_ty = HirType::Function(Box::new(HirFunctionType {
        params: vec![HirType::I64, HirType::I64],
        returns: vec![HirType::I64],
        lifetime_params: vec![],
        is_variadic: false,
    }));
    let forty = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(40)));
    let two = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(2)));
    let target = f.create_value(fn_ty.clone(), HirValueKind::Instruction);
    let result = f.create_value(HirType::I64, HirValueKind::Instruction);
    let make = match made {
        Made::Closure => HirInstruction::CreateClosure {
            result: target,
            closure_ty: fn_ty,
            function: step,
            captures: vec![],
        },
        Made::Reference => HirInstruction::Call {
            result: Some(target),
            callee: HirCallable::FuncRef(step),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
    };
    let entry = f.blocks.get_mut(&f.entry_block).unwrap();
    entry.add_instruction(make);
    entry.add_instruction(HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Indirect(target),
        args: vec![forty, two],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    entry.set_terminator(HirTerminator::Return {
        values: vec![result],
    });
    f
}

fn module_calling(made: Made) -> (HirModule, HirId) {
    let mut module = HirModule::new(name("call_convention_agreement"));
    let step = step();
    let step_id = step.id;
    let main = main_calling(step_id, made);
    let main_id = main.id;
    module.functions.insert(step_id, step);
    module.functions.insert(main_id, main);
    (module, main_id)
}

/// Compile `main` and `step` with `backend` and run `main`.
fn run(mut backend: CraneliftBackend, made: Made) -> i64 {
    let (module, main) = module_calling(made);
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let entry = backend.get_function_ptr(main).expect("main was compiled");
    // SAFETY: `main` is compiled for this host with the platform's C
    // convention, takes nothing and returns an i64.
    let main: extern "C" fn() -> i64 = unsafe { std::mem::transmute(entry) };
    main()
}

#[test]
fn a_closure_is_called_with_its_functions_convention() {
    assert_eq!(run(CraneliftBackend::new().unwrap(), Made::Closure), 42);
}

#[test]
fn a_function_reference_is_called_with_its_functions_convention() {
    assert_eq!(run(CraneliftBackend::new().unwrap(), Made::Reference), 42);
}

#[test]
fn a_closure_read_from_a_call_cell_is_called_with_its_functions_convention() {
    let mut backend = CraneliftBackend::new().unwrap();
    backend.set_reloadable_calls(true);
    assert_eq!(run(backend, Made::Closure), 42);
}

#[test]
fn a_reference_read_from_a_call_cell_is_called_with_its_functions_convention() {
    let mut backend = CraneliftBackend::new().unwrap();
    backend.set_reloadable_calls(true);
    assert_eq!(run(backend, Made::Reference), 42);
}
