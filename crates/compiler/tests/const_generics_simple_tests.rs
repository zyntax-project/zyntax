//! Simplified tests for const generics and monomorphization

use std::collections::HashMap;
use zyntax_compiler::{
    hir::*, monomorphize_module, ConstEvalContext, ConstEvaluator, MonomorphizationContext,
};
use zyntax_typed_ast::arena::AstArena;

fn create_test_arena() -> AstArena {
    AstArena::new()
}

fn intern_str(arena: &mut AstArena, s: &str) -> zyntax_typed_ast::InternedString {
    arena.intern_string(s)
}

#[test]
fn test_const_generic_function_signature() {
    let mut arena = create_test_arena();

    // Create a function with both type and const parameters
    let t_param = intern_str(&mut arena, "T");
    let n_param = intern_str(&mut arena, "N");

    let signature = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: intern_str(&mut arena, "arr"),
            ty: HirType::Array(Box::new(HirType::Opaque(t_param)), 10),
            attributes: ParamAttributes::default(),
        }],
        returns: vec![HirType::Void],
        type_params: vec![HirTypeParam {
            name: t_param,
            constraints: vec![],
        }],
        const_params: vec![HirConstParam {
            name: n_param,
            ty: HirType::U64,
            default: None,
        }],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let func = HirFunction::new(intern_str(&mut arena, "test_array"), signature);

    // Verify const params are present
    assert_eq!(func.signature.const_params.len(), 1);
    assert_eq!(func.signature.const_params[0].name, n_param);
    assert_eq!(func.signature.type_params.len(), 1);
}

#[test]
fn test_const_evaluation_arithmetic() {
    let mut eval_ctx = ConstEvalContext::new();

    // Test arithmetic operations
    let result = eval_ctx
        .eval_binary_op(BinaryOp::Add, &HirConstant::I64(5), &HirConstant::I64(3))
        .unwrap();
    assert_eq!(result, HirConstant::I64(8));

    let result = eval_ctx
        .eval_binary_op(BinaryOp::Mul, &HirConstant::I64(4), &HirConstant::I64(7))
        .unwrap();
    assert_eq!(result, HirConstant::I64(28));

    // Test unsigned arithmetic
    let result = eval_ctx
        .eval_binary_op(BinaryOp::Add, &HirConstant::U64(100), &HirConstant::U64(50))
        .unwrap();
    assert_eq!(result, HirConstant::U64(150));
}

#[test]
fn test_const_evaluation_bitwise() {
    let ctx = ConstEvalContext::new();

    // Test bitwise operations
    let result = ctx
        .eval_binary_op(
            BinaryOp::And,
            &HirConstant::I64(0b1111),
            &HirConstant::I64(0b1010),
        )
        .unwrap();
    assert_eq!(result, HirConstant::I64(0b1010));

    let result = ctx
        .eval_binary_op(
            BinaryOp::Or,
            &HirConstant::I64(0b1100),
            &HirConstant::I64(0b0011),
        )
        .unwrap();
    assert_eq!(result, HirConstant::I64(0b1111));

    let result = ctx
        .eval_binary_op(
            BinaryOp::Xor,
            &HirConstant::I64(0b1111),
            &HirConstant::I64(0b1010),
        )
        .unwrap();
    assert_eq!(result, HirConstant::I64(0b0101));
}

#[test]
fn test_const_evaluation_comparison() {
    let ctx = ConstEvalContext::new();

    // Test comparison operations
    let result = ctx
        .eval_binary_op(BinaryOp::Eq, &HirConstant::I64(5), &HirConstant::I64(5))
        .unwrap();
    assert_eq!(result, HirConstant::Bool(true));

    let result = ctx
        .eval_binary_op(BinaryOp::Lt, &HirConstant::I64(3), &HirConstant::I64(5))
        .unwrap();
    assert_eq!(result, HirConstant::Bool(true));

    let result = ctx
        .eval_binary_op(BinaryOp::Gt, &HirConstant::I64(10), &HirConstant::I64(5))
        .unwrap();
    assert_eq!(result, HirConstant::Bool(true));
}

#[test]
fn test_const_evaluation_unary() {
    let ctx = ConstEvalContext::new();

    // Test unary operations
    let result = ctx
        .eval_unary_op(UnaryOp::Neg, &HirConstant::I64(42))
        .unwrap();
    assert_eq!(result, HirConstant::I64(-42));

    let result = ctx
        .eval_unary_op(UnaryOp::Not, &HirConstant::Bool(true))
        .unwrap();
    assert_eq!(result, HirConstant::Bool(false));

    let result = ctx
        .eval_unary_op(UnaryOp::Not, &HirConstant::I64(0))
        .unwrap();
    assert_eq!(result, HirConstant::I64(-1)); // All bits set
}

#[test]
fn test_monomorphization_context() {
    let mut arena = create_test_arena();
    let mut mono_ctx = MonomorphizationContext::new();

    // Create a generic function
    let t_param = intern_str(&mut arena, "T");

    let signature = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: intern_str(&mut arena, "value"),
            ty: HirType::Opaque(t_param),
            attributes: ParamAttributes::default(),
        }],
        returns: vec![HirType::Opaque(t_param)],
        type_params: vec![HirTypeParam {
            name: t_param,
            constraints: vec![],
        }],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let generic_func = HirFunction::new(intern_str(&mut arena, "identity"), signature);
    let generic_id = generic_func.id;

    // Register the generic function
    mono_ctx.register_generic(generic_func);

    // Instantiate with T=I32
    let type_args = vec![HirType::I32];
    let const_args = vec![];

    let instance_id = mono_ctx
        .get_or_create_instance(generic_id, type_args.clone(), const_args.clone())
        .unwrap();

    // Verify we get a different ID for the monomorphized instance
    assert_ne!(instance_id, generic_id);

    // Try to get the same instance again - should return the cached one
    let cached_id = mono_ctx
        .get_or_create_instance(generic_id, type_args, const_args)
        .unwrap();

    assert_eq!(instance_id, cached_id);
}

#[test]
fn test_module_monomorphization() {
    let mut arena = create_test_arena();
    let mut module = HirModule::new(intern_str(&mut arena, "test_module"));

    // Add a generic function to the module
    let t_param = intern_str(&mut arena, "T");
    let signature = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: intern_str(&mut arena, "value"),
            ty: HirType::Opaque(t_param),
            attributes: ParamAttributes::default(),
        }],
        returns: vec![HirType::Opaque(t_param)],
        type_params: vec![HirTypeParam {
            name: t_param,
            constraints: vec![],
        }],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let generic_func = HirFunction::new(intern_str(&mut arena, "identity"), signature);
    module.add_function(generic_func);

    // Run monomorphization pass
    let result = monomorphize_module(&mut module);
    assert!(result.is_ok());
}

#[test]
fn test_const_evaluation_with_overflow() {
    let ctx = ConstEvalContext::new();

    // Test wrapping addition with overflow
    let result = ctx
        .eval_binary_op(
            BinaryOp::Add,
            &HirConstant::U64(u64::MAX),
            &HirConstant::U64(1),
        )
        .unwrap();
    assert_eq!(result, HirConstant::U64(0)); // Should wrap around

    // Test division by zero error
    let result = ctx.eval_binary_op(BinaryOp::Div, &HirConstant::I64(10), &HirConstant::I64(0));
    assert!(result.is_err());
}

#[test]
fn test_const_evaluation_intrinsics() {
    let mut eval_ctx = ConstEvalContext::new();

    // Test sizeof intrinsic
    let sizeof_inst = HirInstruction::Call {
        result: Some(HirId::new()),
        callee: HirCallable::Intrinsic(Intrinsic::SizeOf),
        args: vec![],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let result = eval_ctx.eval_const_expr(&sizeof_inst).unwrap();
    assert_eq!(result, HirConstant::U64(8));

    // Test alignof intrinsic
    let alignof_inst = HirInstruction::Call {
        result: Some(HirId::new()),
        callee: HirCallable::Intrinsic(Intrinsic::AlignOf),
        args: vec![],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let result = eval_ctx.eval_const_expr(&alignof_inst).unwrap();
    assert_eq!(result, HirConstant::U64(8));
}
