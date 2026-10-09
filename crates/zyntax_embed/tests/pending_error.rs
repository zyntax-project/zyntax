//! A host takes the error a call left in the runtime's error-flag global,
//! whether it called by name or jumped to the function's code.

use zyntax_embed::{TieredRuntime, ZyntaxValue};
use zyntax_typed_ast::typed_ast::BinaryOp;
use zyntax_typed_ast::{
    Mutability, PrimitiveType, Span, Type, TypedASTBuilder, TypedProgram, Visibility,
};

/// `failure: i64`, the error flag; `fail(code)` stores `code` in it and
/// `succeed()` leaves it alone.
fn program() -> TypedProgram {
    let mut b = TypedASTBuilder::new();
    let span = Span::new(0, 0);
    let i64_ty = Type::Primitive(PrimitiveType::I64);
    let flag = b.variable_declaration(
        "failure",
        i64_ty.clone(),
        Mutability::Mutable,
        None,
        Visibility::Public,
        span,
    );

    let code = b.parameter("code", i64_ty.clone(), Mutability::Immutable, span);
    let target = b.variable("failure", i64_ty.clone(), span);
    let value = b.variable("code", i64_ty.clone(), span);
    let store = b.binary(BinaryOp::Assign, target, value, i64_ty.clone(), span);
    let store = b.expression_statement(store, span);
    let zero = b.int_literal(0, span);
    let ret = b.return_stmt(zero, span);
    let body = b.block(vec![store, ret], span);
    let fail = b.function(
        "fail",
        vec![code],
        i64_ty.clone(),
        body,
        Visibility::Public,
        false,
        span,
    );

    let one = b.int_literal(1, span);
    let ret = b.return_stmt(one, span);
    let body = b.block(vec![ret], span);
    let succeed = b.function(
        "succeed",
        Vec::new(),
        i64_ty,
        body,
        Visibility::Public,
        false,
        span,
    );
    b.program(vec![flag, fail, succeed], span)
}

#[test]
fn a_host_takes_the_error_a_call_left_pending() {
    let mut runtime = TieredRuntime::development().expect("runtime");
    runtime.set_error_flag_global("failure");
    runtime
        .compile_typed_program(program())
        .expect("program compiles");

    assert_eq!(runtime.take_pending_error(), None);
    runtime.call_raw("succeed", &[]).expect("succeed runs");
    assert_eq!(runtime.take_pending_error(), None);

    runtime
        .call_raw("fail", &[ZyntaxValue::Int(7)])
        .expect("fail runs");
    assert_eq!(runtime.take_pending_error(), Some(7));
    // Taking it cleared the global.
    assert_eq!(runtime.take_pending_error(), None);

    // A host that jumps to the code sees the same global.
    let fail = runtime.get_function_ptr("fail").expect("fail is compiled");
    let fail: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(fail) };
    assert_eq!(fail(9), 0);
    assert_eq!(runtime.take_pending_error(), Some(9));
    assert_eq!(runtime.take_pending_error(), None);
}

#[test]
fn a_runtime_without_an_error_flag_has_nothing_pending() {
    let mut runtime = TieredRuntime::development().expect("runtime");
    runtime
        .compile_typed_program(program())
        .expect("program compiles");
    runtime
        .call_raw("fail", &[ZyntaxValue::Int(7)])
        .expect("fail runs");
    assert_eq!(runtime.take_pending_error(), None);
}

#[test]
fn a_host_configures_the_flag_of_a_prelowered_module() {
    use std::sync::{Arc, Mutex};
    use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
    use zyntax_typed_ast::{AstArena, TypeRegistry};

    let mut context = LoweringContext::new(
        zyntax_typed_ast::InternedString::new_global("library"),
        Arc::new(TypeRegistry::new()),
        Arc::new(Mutex::new(AstArena::new())),
        LoweringConfig::default(),
    );
    let module = context.lower_program(&mut program()).expect("lowers");
    let flag = module.globals.values().next().expect("flag");
    assert!(!flag.error_flag);

    let mut runtime = TieredRuntime::development().expect("runtime");
    runtime.set_error_flag_global("failure");
    runtime.compile_module(module).expect("module compiles");
    runtime
        .call_raw("fail", &[ZyntaxValue::Int(23)])
        .expect("fail runs");
    assert_eq!(runtime.take_pending_error(), Some(23));
    assert_eq!(runtime.take_pending_error(), None);
    runtime.call_raw("succeed", &[]).expect("succeed runs");
    assert_eq!(runtime.take_pending_error(), None);
    runtime
        .call_raw("fail", &[ZyntaxValue::Int(7)])
        .expect("fail runs");
    assert_eq!(runtime.take_pending_error(), Some(7));
    assert_eq!(runtime.take_pending_error(), None);
}
