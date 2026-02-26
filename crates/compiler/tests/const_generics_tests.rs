//! Comprehensive tests for const generics and monomorphization

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
fn test_const_generic_array_type() {
    let mut arena = create_test_arena();

    // Create array type with const generic parameter: Array<T, N>
    let n_param = intern_str(&mut arena, "N");
    let t_param = intern_str(&mut arena, "T");

    // Create function signature with const generic
    let signature = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: intern_str(&mut arena, "arr"),
            ty: HirType::Array(Box::new(HirType::Opaque(t_param)), 10), // Fixed size for now
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
}

#[test]
fn test_const_evaluation_in_array_size() {
    let mut eval_ctx = ConstEvalContext::new();

    // Test evaluating const expression: 4 * 8 = 32
    let mul_inst = HirInstruction::Binary {
        op: BinaryOp::Mul,
        result: HirId::new(),
        ty: HirType::I64,
        left: HirId::new(),
        right: HirId::new(),
    };

    // Add const values to context
    let left_id = match &mul_inst {
        HirInstruction::Binary { left, .. } => *left,
        _ => unreachable!(),
    };
    let right_id = match &mul_inst {
        HirInstruction::Binary { right, .. } => *right,
        _ => unreachable!(),
    };

    eval_ctx.const_values.insert(left_id, HirConstant::I64(4));
    eval_ctx.const_values.insert(right_id, HirConstant::I64(8));

    let result = eval_ctx.eval_const_expr(&mul_inst).unwrap();
    assert_eq!(result, HirConstant::I64(32));
}

#[test]
fn test_monomorphization_with_const_generics() {
    let mut arena = create_test_arena();
    let mut mono_ctx = MonomorphizationContext::new();

    // Create a generic function: fn buffer<T, const N: usize>() -> [T; N]
    let t_param = intern_str(&mut arena, "T");
    let n_param = intern_str(&mut arena, "N");

    let signature = HirFunctionSignature {
        params: vec![],
        returns: vec![
            HirType::Array(Box::new(HirType::Opaque(t_param)), 0), // Placeholder size
        ],
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

    let generic_func = HirFunction::new(intern_str(&mut arena, "buffer"), signature);
    let generic_id = generic_func.id;

    // Register the generic function
    mono_ctx.register_generic(generic_func);

    // Instantiate with T=u8, N=256
    let type_args = vec![HirType::U8];
    let const_args = vec![HirConstant::U64(256)];

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
fn test_const_generic_in_struct() {
    let mut arena = create_test_arena();

    // Create a struct with const generic: FixedBuffer<T, const SIZE: usize>
    let size_param = intern_str(&mut arena, "SIZE");

    let buffer_struct = HirStructType {
        name: Some(intern_str(&mut arena, "FixedBuffer")),
        fields: vec![
            HirType::Array(Box::new(HirType::U8), 0), // Will be substituted with SIZE
            HirType::U64,                             // Current position
        ],
        packed: false,
    };

    // Create instance with SIZE=1024
    let concrete_buffer = HirType::Generic {
        base: Box::new(HirType::Struct(buffer_struct)),
        type_args: vec![],
        const_args: vec![HirConstant::U64(1024)],
    };

    // Verify the structure
    match &concrete_buffer {
        HirType::Generic { const_args, .. } => {
            assert_eq!(const_args.len(), 1);
            assert_eq!(const_args[0], HirConstant::U64(1024));
        }
        _ => panic!("Expected generic type"),
    }
}

#[test]
fn test_const_evaluation_with_intrinsics() {
    let mut eval_ctx = ConstEvalContext::new();

    // Test sizeof intrinsic (placeholder implementation returns 8)
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

#[test]
fn test_const_generic_substitution_in_types() {
    let mut arena = create_test_arena();
    let evaluator = ConstEvaluator::new();

    let n_param = intern_str(&mut arena, "N");

    // Create type with const generic: Array<T, N>
    let const_generic_type = HirType::ConstGeneric(n_param);

    // Create substitution map
    let mut const_args = HashMap::new();
    const_args.insert(n_param, HirConstant::U64(32));

    // Substitute const generics
    let substituted = evaluator.substitute_const_generics(&const_generic_type, &const_args);

    // The simple implementation converts const N to [u8; N]
    match substituted {
        HirType::Array(elem, size) => {
            assert!(matches!(*elem, HirType::U8));
            assert_eq!(size, 32);
        }
        _ => panic!("Expected array type after substitution"),
    }
}

#[test]
fn test_const_binary_operations() {
    let ctx = ConstEvalContext::new();

    // Test all binary operations
    let test_cases = vec![
        // Arithmetic
        (
            BinaryOp::Add,
            HirConstant::I64(5),
            HirConstant::I64(3),
            HirConstant::I64(8),
        ),
        (
            BinaryOp::Sub,
            HirConstant::I64(10),
            HirConstant::I64(3),
            HirConstant::I64(7),
        ),
        (
            BinaryOp::Mul,
            HirConstant::I64(4),
            HirConstant::I64(7),
            HirConstant::I64(28),
        ),
        (
            BinaryOp::Div,
            HirConstant::I64(20),
            HirConstant::I64(4),
            HirConstant::I64(5),
        ),
        (
            BinaryOp::Rem,
            HirConstant::I64(17),
            HirConstant::I64(5),
            HirConstant::I64(2),
        ),
        // Unsigned arithmetic
        (
            BinaryOp::Add,
            HirConstant::U64(100),
            HirConstant::U64(50),
            HirConstant::U64(150),
        ),
        (
            BinaryOp::Mul,
            HirConstant::U64(12),
            HirConstant::U64(12),
            HirConstant::U64(144),
        ),
        // Bitwise
        (
            BinaryOp::And,
            HirConstant::I64(0b1111),
            HirConstant::I64(0b1010),
            HirConstant::I64(0b1010),
        ),
        (
            BinaryOp::Or,
            HirConstant::I64(0b1100),
            HirConstant::I64(0b0011),
            HirConstant::I64(0b1111),
        ),
        (
            BinaryOp::Xor,
            HirConstant::I64(0b1111),
            HirConstant::I64(0b1010),
            HirConstant::I64(0b0101),
        ),
        (
            BinaryOp::Shl,
            HirConstant::I64(1),
            HirConstant::I64(3),
            HirConstant::I64(8),
        ),
        (
            BinaryOp::Shr,
            HirConstant::I64(16),
            HirConstant::I64(2),
            HirConstant::I64(4),
        ),
        // Comparisons
        (
            BinaryOp::Eq,
            HirConstant::I64(5),
            HirConstant::I64(5),
            HirConstant::Bool(true),
        ),
        (
            BinaryOp::Ne,
            HirConstant::I64(5),
            HirConstant::I64(3),
            HirConstant::Bool(true),
        ),
        (
            BinaryOp::Lt,
            HirConstant::I64(3),
            HirConstant::I64(5),
            HirConstant::Bool(true),
        ),
        (
            BinaryOp::Gt,
            HirConstant::I64(10),
            HirConstant::I64(5),
            HirConstant::Bool(true),
        ),
    ];

    for (op, left, right, expected) in test_cases {
        let result = ctx.eval_binary_op(op, &left, &right).unwrap();
        assert_eq!(
            result, expected,
            "Failed for {:?} with {:?} and {:?}",
            op, left, right
        );
    }
}

#[test]
fn test_const_unary_operations() {
    let ctx = ConstEvalContext::new();

    // Test negation
    let result = ctx
        .eval_unary_op(UnaryOp::Neg, &HirConstant::I64(42))
        .unwrap();
    assert_eq!(result, HirConstant::I64(-42));

    // Test logical not
    let result = ctx
        .eval_unary_op(UnaryOp::Not, &HirConstant::Bool(true))
        .unwrap();
    assert_eq!(result, HirConstant::Bool(false));

    // Test bitwise not
    let result = ctx
        .eval_unary_op(UnaryOp::Not, &HirConstant::I64(0))
        .unwrap();
    assert_eq!(result, HirConstant::I64(-1)); // All bits set
}

#[test]
fn test_module_monomorphization() {
    let mut arena = create_test_arena();
    let mut module = HirModule::new(intern_str(&mut arena, "test_module"));

    // Add a generic function
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
