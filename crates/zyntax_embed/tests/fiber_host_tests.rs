#![cfg(feature = "native")]

use zyntax_embed::{ZyntaxFiberStep, ZyntaxRuntime, ZyntaxValue};
use zyntax_typed_ast::{
    typed_ast::typed_node, PrimitiveType, Span, Type, TypedASTBuilder, TypedDeclaration,
    TypedStatement, Visibility,
};

fn yielding_fiber_program() -> zyntax_typed_ast::TypedProgram {
    let mut builder = TypedASTBuilder::new();
    let span = Span::new(0, 0);
    let yielded = builder.int_literal(7, span);
    let yield_statement = typed_node(
        TypedStatement::Yield(Box::new(yielded)),
        Type::Primitive(PrimitiveType::Unit),
        span,
    );
    let loop_body = builder.block(vec![yield_statement], span);
    let condition = builder.bool_literal(true, span);
    let resident_loop = builder.while_loop(condition, loop_body, span);
    let body = builder.block(vec![resident_loop], span);
    let mut declaration = builder.function(
        "host_fiber",
        Vec::new(),
        Type::Primitive(PrimitiveType::I64),
        body,
        Visibility::Public,
        false,
        span,
    );
    let TypedDeclaration::Function(function) = &mut declaration.node else {
        unreachable!("TypedASTBuilder::function must return a function declaration")
    };
    function.is_fiber = true;
    builder.program(vec![declaration], span)
}

#[test]
fn host_handle_owns_and_resumes_a_compiled_fiber() {
    let mut runtime = ZyntaxRuntime::new().expect("native runtime");
    runtime
        .compile_typed_program(yielding_fiber_program())
        .expect("fiber program compiles");

    let mut fiber = runtime
        .call_fiber("host_fiber", &[])
        .expect("host fiber is created paused");
    assert_eq!(fiber.resume().unwrap(), ZyntaxFiberStep::Yielded(7));
    assert_eq!(fiber.resume().unwrap(), ZyntaxFiberStep::Yielded(7));
    assert!(!fiber.is_terminal());

    let error = match runtime.call_fiber("host_fiber", &[ZyntaxValue::Int(1)]) {
        Ok(_) => panic!("the initial host API is deliberately zero-argument"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("argument"));
}
