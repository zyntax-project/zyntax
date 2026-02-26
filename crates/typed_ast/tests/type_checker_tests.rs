//! # Type Checker Tests
//!
//! Comprehensive tests for the enhanced type checker, covering:
//! - Enhanced parameter system (out, ref, inout, rest, optional)
//! - Named arguments in function calls
//! - Field access and array indexing
//! - Pattern matching type checking
//! - Error cases and edge conditions

use zyntax_typed_ast::type_registry::{TypeId, TypeRegistry};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::*;
use zyntax_typed_ast::{AsyncKind, CallingConvention, NullabilityKind};

#[cfg(test)]
mod type_checker_tests {
    use super::*;

    fn create_test_environment() -> (TypeRegistry, crate::arena::AstArena) {
        let env = TypeRegistry::new();
        let arena = crate::arena::AstArena::new();

        (env, arena)
    }

    fn create_test_span() -> Span {
        Span::new(0, 10)
    }

    #[test]
    fn test_enhanced_parameters_basic() {
        let mut builder = TypedASTBuilder::new();
        let (env, _arena) = create_test_environment();
        let mut checker = TypeChecker::new(Box::new(env));
        let span = create_test_span();

        // Create a function with various parameter types
        let regular_param = builder.parameter(
            "x",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );
        let default_val = builder.int_literal(42, span);
        let optional_param = builder.optional_parameter(
            "y",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            default_val,
            span,
        );
        let out_param = builder.out_parameter("result", Type::Primitive(PrimitiveType::I32), span);

        let params = vec![regular_param, optional_param, out_param];
        let body = builder.block(vec![], span);

        let function = builder.function(
            "test_func",
            params,
            Type::Primitive(PrimitiveType::Unit),
            body,
            Visibility::Public,
            false,
            span,
        );

        let program = builder.program(vec![function], span);

        // Type check should succeed - new API doesn't return Result
        checker.check_program(&program);
        assert!(
            !checker.has_errors(),
            "Enhanced parameters should type check successfully. Errors: {}",
            if checker.has_errors() {
                "has errors" // In real usage, we'd display the actual diagnostics
            } else {
                "none"
            }
        );
    }

    #[test]
    fn test_named_arguments_valid() {
        let mut builder = TypedASTBuilder::new();
        let (env, _arena) = create_test_environment();
        let mut checker = TypeChecker::new(Box::new(env));
        let span = create_test_span();

        // Create function with named parameters support
        let param1 = builder.parameter(
            "width",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );
        let param2 = builder.parameter(
            "height",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );

        let params = vec![param1, param2];
        let body = builder.block(vec![], span);

        let function = builder.function(
            "calculate",
            params,
            Type::Primitive(PrimitiveType::Unit),
            body,
            Visibility::Public,
            false,
            span,
        );

        let width = builder.intern("width");
        let height = builder.intern("height");
        // Create a call with named arguments
        let callee = builder.variable(
            "calculate",
            Type::Function {
                params: vec![
                    ParamInfo {
                        name: Some(width),
                        ty: Type::Primitive(PrimitiveType::I32),
                        is_optional: false,
                        is_varargs: false,
                        is_keyword_only: false,
                        is_positional_only: false,
                        is_out: false,
                        is_ref: false,
                        is_inout: false,
                    },
                    ParamInfo {
                        name: Some(height),
                        ty: Type::Primitive(PrimitiveType::I32),
                        is_optional: false,
                        is_varargs: false,
                        is_keyword_only: false,
                        is_positional_only: false,
                        is_out: false,
                        is_ref: false,
                        is_inout: false,
                    },
                ],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: true,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            span,
        );

        let arg1 = builder.int_literal(10, span);
        let arg2 = builder.int_literal(20, span);

        let named_call = builder.call_named(
            callee,
            vec![("width", arg1), ("height", arg2)],
            Type::Primitive(PrimitiveType::Unit),
            span,
        );

        let main = builder.intern("main");

        let call_stmt = builder.expression_statement(named_call, span);
        let body = builder.block(vec![call_stmt], span);
        let program = builder.program(
            vec![
                function,
                TypedNode {
                    node: TypedDeclaration::Function(TypedFunction {
                        name: main,
                        params: vec![],
                        type_params: vec![],
                        body: Some(body),
                        return_type: Type::Primitive(PrimitiveType::Unit),
                        visibility: Visibility::Public,
                        is_async: false,
                        is_external: false,
                        calling_convention: CallingConvention::Default,
                        link_name: None,
                        ..Default::default()
                    }),
                    ty: Type::Function {
                        params: vec![],
                        return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                        is_varargs: false,
                        has_named_params: false,
                        has_default_params: false,
                        async_kind: AsyncKind::Sync,
                        calling_convention: CallingConvention::Default,
                        nullability: NullabilityKind::NonNull,
                    },
                    span,
                },
            ],
            span,
        );

        checker.check_program(&program);
        if checker.has_errors() {
            println!("Type checker reported errors:");
            for diagnostic in checker.diagnostics().diagnostics() {
                println!("  {:?}", diagnostic);
            }
        }
        assert!(
            !checker.has_errors(),
            "Named arguments should type check successfully"
        );
    }

    #[test]
    fn test_out_parameters_require_assignable() {
        let mut builder = TypedASTBuilder::new();
        let (env, _arena) = create_test_environment();
        let mut checker = TypeChecker::new(Box::new(env));
        let span = create_test_span();

        // Create function with out parameter
        let out_param = builder.out_parameter("result", Type::Primitive(PrimitiveType::I32), span);
        let params = vec![out_param];
        let body = builder.block(vec![], span);

        let function = builder.function(
            "get_result",
            params,
            Type::Primitive(PrimitiveType::Unit),
            body,
            Visibility::Public,
            false,
            span,
        );

        let result = builder.intern("result");
        // Create call with non-assignable expression (literal)
        let callee = builder.variable(
            "get_result",
            Type::Function {
                params: vec![ParamInfo {
                    name: Some(result),
                    ty: Type::Primitive(PrimitiveType::I32),
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: true, // This is an out parameter!
                    is_ref: false,
                    is_inout: false,
                }],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            span,
        );

        let literal_arg = builder.int_literal(42, span); // Not assignable!

        let invalid_call = builder.call_positional(
            callee,
            vec![literal_arg],
            Type::Primitive(PrimitiveType::Unit),
            span,
        );

        let call_stmt = builder.expression_statement(invalid_call, span);
        let main_body = builder.block(vec![call_stmt], span);

        let main_func = TypedNode {
            node: TypedDeclaration::Function(TypedFunction {
                name: builder.intern("main"),
                params: vec![],
                type_params: vec![],
                body: Some(main_body),
                return_type: Type::Primitive(PrimitiveType::Unit),
                visibility: Visibility::Public,
                is_async: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
                ..Default::default()
            }),
            ty: Type::Function {
                params: vec![],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            span,
        };

        let program = builder.program(vec![function, main_func], span);

        checker.check_program(&program);

        assert!(
            checker.has_errors(),
            "Out parameters should reject non-assignable expressions"
        );

        // Note: With the new diagnostic system, we'd check specific error codes
        // For now, we just verify that errors were reported
    }

    #[test]
    fn test_field_access_type_checking() {
        let mut builder = TypedASTBuilder::new();
        let (env, _arena) = create_test_environment();
        let mut checker = TypeChecker::new(Box::new(env));
        let span = create_test_span();

        // Create a struct type (simplified for testing)
        let point_type = Type::Named {
            id: TypeId::next(), // Point type
            type_args: vec![],
            const_args: Vec::new(),
            variance: Vec::new(),
            nullability: NullabilityKind::NonNull,
        };

        // Create a variable declaration of struct type
        let point_decl = builder.let_statement(
            "point",
            point_type.clone(),
            Mutability::Immutable,
            None,
            span,
        );

        // Create a variable reference
        let point_var = builder.variable("point", point_type.clone(), span);

        // Create field access expressions
        let x_access = builder.field_access(
            point_var.clone(),
            "x",
            Type::Primitive(PrimitiveType::I32),
            span,
        );
        let y_access =
            builder.field_access(point_var, "y", Type::Primitive(PrimitiveType::I32), span);

        let x_stmt = builder.expression_statement(x_access, span);
        let y_stmt = builder.expression_statement(y_access, span);

        let main_body = builder.block(vec![point_decl, x_stmt, y_stmt], span);

        let main_func = TypedNode {
            node: TypedDeclaration::Function(TypedFunction {
                name: builder.intern("main"),
                params: vec![],
                type_params: vec![],
                body: Some(main_body),
                return_type: Type::Primitive(PrimitiveType::Unit),
                visibility: Visibility::Public,
                is_async: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
                ..Default::default()
            }),
            ty: Type::Function {
                params: vec![],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            span,
        };

        let program = builder.program(vec![main_func], span);

        checker.check_program(&program);
        assert!(
            !checker.has_errors(),
            "Field access should type check successfully"
        );
    }

    #[test]
    fn test_array_indexing_type_checking() {
        let mut builder = TypedASTBuilder::new();
        let (env, _arena) = create_test_environment();
        let mut checker = TypeChecker::new(Box::new(env));
        let span = create_test_span();

        // Create array variable declaration
        let array_type = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: Some(ConstValue::Int(10)),
            nullability: NullabilityKind::NonNull,
        };
        let array_decl = builder.let_statement(
            "numbers",
            array_type.clone(),
            Mutability::Immutable,
            None,
            span,
        );

        // Create array variable reference
        let array_var = builder.variable("numbers", array_type, span);

        // Create valid index (integer)
        let index = builder.int_literal(0, span);

        // Create array access
        let array_access =
            builder.index(array_var, index, Type::Primitive(PrimitiveType::I32), span);
        let access_stmt = builder.expression_statement(array_access, span);

        let main_body = builder.block(vec![array_decl, access_stmt], span);

        let main_func = TypedNode {
            node: TypedDeclaration::Function(TypedFunction {
                name: builder.intern("main"),
                params: vec![],
                type_params: vec![],
                body: Some(main_body),
                return_type: Type::Primitive(PrimitiveType::Unit),
                visibility: Visibility::Public,
                is_async: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
                ..Default::default()
            }),
            ty: Type::Function {
                params: vec![],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            span,
        };

        let program = builder.program(vec![main_func], span);

        checker.check_program(&program);
        if checker.has_errors() {
            println!("Type checker reported errors:");
            for diagnostic in checker.diagnostics().diagnostics() {
                println!("  {:?}", diagnostic);
            }
        }
        assert!(
            !checker.has_errors(),
            "Array indexing should type check successfully"
        );
    }

    #[test]
    fn test_invalid_array_index_type() {
        let mut builder = TypedASTBuilder::new();
        let (env, _arena) = create_test_environment();
        let mut checker = TypeChecker::new(Box::new(env));
        let span = create_test_span();

        // Create array variable
        let array_var = builder.variable(
            "numbers",
            Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
                size: Some(ConstValue::Int(10)),
                nullability: NullabilityKind::NonNull,
            },
            span,
        );

        // Create invalid index (string instead of integer)
        let invalid_index = builder.string_literal("not_a_number", span);

        // Create array access with invalid index
        let array_access = builder.index(
            array_var,
            invalid_index,
            Type::Primitive(PrimitiveType::I32),
            span,
        );
        let access_stmt = builder.expression_statement(array_access, span);

        let main_body = builder.block(vec![access_stmt], span);

        let main_func = TypedNode {
            node: TypedDeclaration::Function(TypedFunction {
                name: builder.intern("main"),
                params: vec![],
                type_params: vec![],
                body: Some(main_body),
                return_type: Type::Primitive(PrimitiveType::Unit),
                visibility: Visibility::Public,
                is_async: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
                ..Default::default()
            }),
            ty: Type::Function {
                params: vec![],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            span,
        };

        let program = builder.program(vec![main_func], span);

        checker.check_program(&program);
        assert!(
            checker.has_errors(),
            "Invalid array index type should be rejected"
        );

        // Note: With the new diagnostic system, we'd check specific error codes
        // For now, we just verify that errors were reported
    }

    #[test]
    fn test_too_many_arguments_error() {
        let mut builder = TypedASTBuilder::new();
        let (env, _arena) = create_test_environment();
        let mut checker = TypeChecker::new(Box::new(env));
        let span = create_test_span();

        // Create function that takes 2 parameters
        let param1 = builder.parameter(
            "x",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );
        let param2 = builder.parameter(
            "y",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );

        let function_type = Type::Function {
            params: vec![
                ParamInfo {
                    name: Some(builder.intern("x")),
                    ty: Type::Primitive(PrimitiveType::I32),
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
                ParamInfo {
                    name: Some(builder.intern("y")),
                    ty: Type::Primitive(PrimitiveType::I32),
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
            ],
            return_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            is_varargs: false,
            has_named_params: false,
            has_default_params: false,
            async_kind: AsyncKind::Sync,
            calling_convention: CallingConvention::Default,
            nullability: NullabilityKind::NonNull,
        };

        // Create call with 3 arguments (too many!)
        let callee = builder.variable("add", function_type, span);
        let arg1 = builder.int_literal(1, span);
        let arg2 = builder.int_literal(2, span);
        let arg3 = builder.int_literal(3, span); // Extra argument

        let invalid_call = builder.call_positional(
            callee,
            vec![arg1, arg2, arg3],
            Type::Primitive(PrimitiveType::I32),
            span,
        );

        let call_stmt = builder.expression_statement(invalid_call, span);
        let main_body = builder.block(vec![call_stmt], span);

        let main_func = TypedNode {
            node: TypedDeclaration::Function(TypedFunction {
                name: builder.intern("main"),
                params: vec![],
                type_params: vec![],
                body: Some(main_body),
                return_type: Type::Primitive(PrimitiveType::Unit),
                visibility: Visibility::Public,
                is_async: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
                ..Default::default()
            }),
            ty: Type::Function {
                params: vec![],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            span,
        };

        let program = builder.program(vec![main_func], span);

        checker.check_program(&program);
        assert!(
            checker.has_errors(),
            "Too many arguments should be rejected"
        );

        // Note: With the new diagnostic system, we'd check specific error codes
        // For now, we just verify that errors were reported
    }

    #[test]
    fn test_rest_parameters() {
        let mut builder = TypedASTBuilder::new();
        let (env, _arena) = create_test_environment();
        let mut checker = TypeChecker::new(Box::new(env));
        let span = create_test_span();

        // Create function with rest parameter
        let regular_param = builder.parameter(
            "base",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );
        // Rest parameters should have array type
        let rest_param_type = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: None,
            nullability: NullabilityKind::NonNull,
        };
        let rest_param =
            builder.rest_parameter("numbers", rest_param_type, Mutability::Immutable, span);

        // Create the sum function with empty body (returns Unit to match I32 expected)
        let sum_body = builder.block(vec![], span);
        let sum_func = builder.function(
            "sum",
            vec![regular_param, rest_param],
            Type::Primitive(PrimitiveType::Unit), // Changed to Unit to avoid mismatch
            sum_body,
            Visibility::Public,
            false,
            span,
        );

        let function_type = Type::Function {
            params: vec![
                ParamInfo {
                    name: Some(builder.intern("base")),
                    ty: Type::Primitive(PrimitiveType::I32),
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
                ParamInfo {
                    name: Some(builder.intern("numbers")),
                    ty: Type::Array {
                        element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
                        size: None,
                        nullability: NullabilityKind::NonNull,
                    },
                    is_optional: false,
                    is_varargs: true,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
            ],
            return_type: Box::new(Type::Primitive(PrimitiveType::Unit)), // Changed to Unit
            is_varargs: true,
            has_named_params: false,
            has_default_params: false,
            async_kind: AsyncKind::Sync,
            calling_convention: CallingConvention::Default,
            nullability: NullabilityKind::NonNull,
        };

        // Create call with variable number of arguments
        let callee = builder.variable("sum", function_type, span);
        let arg1 = builder.int_literal(10, span); // base
        let arg2 = builder.int_literal(1, span); // rest args
        let arg3 = builder.int_literal(2, span);
        let arg4 = builder.int_literal(3, span);

        let rest_call = builder.call_positional(
            callee,
            vec![arg1, arg2, arg3, arg4],
            Type::Primitive(PrimitiveType::Unit), // Changed to match function return type
            span,
        );

        let call_stmt = builder.expression_statement(rest_call, span);
        let main_body = builder.block(vec![call_stmt], span);

        let main_func = TypedNode {
            node: TypedDeclaration::Function(TypedFunction {
                name: builder.intern("main"),
                params: vec![],
                type_params: vec![],
                body: Some(main_body),
                return_type: Type::Primitive(PrimitiveType::Unit),
                visibility: Visibility::Public,
                is_async: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
                ..Default::default()
            }),
            ty: Type::Function {
                params: vec![],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            span,
        };

        let program = builder.program(vec![sum_func, main_func], span);

        checker.check_program(&program);
        if checker.has_errors() {
            println!("Type checker reported errors:");
            for diagnostic in checker.diagnostics().diagnostics() {
                println!("  {:?}", diagnostic);
            }
        }
        assert!(
            !checker.has_errors(),
            "Rest parameters should type check successfully"
        );
    }

    #[test]
    fn test_constraint_solver_integration() {
        let mut builder = TypedASTBuilder::new();
        let (env, _arena) = create_test_environment();
        let mut checker = TypeChecker::new(Box::new(env));
        let span = create_test_span();

        // Test that the constraint solver works alongside regular type checking
        // Create a simple function call that should exercise the constraint solver
        let param = builder.parameter(
            "x",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );
        let body = builder.block(vec![], span);

        let function = builder.function(
            "constraint_test",
            vec![param],
            Type::Primitive(PrimitiveType::Unit),
            body,
            Visibility::Public,
            false,
            span,
        );

        let x_name = builder.intern("x");
        let main_name = builder.intern("main");

        let main_call = {
            let callee = builder.variable(
                "constraint_test",
                Type::Function {
                    params: vec![ParamInfo {
                        name: Some(x_name),
                        ty: Type::Primitive(PrimitiveType::I32),
                        is_optional: false,
                        is_varargs: false,
                        is_keyword_only: false,
                        is_positional_only: false,
                        is_out: false,
                        is_ref: false,
                        is_inout: false,
                    }],
                    return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                    is_varargs: false,
                    has_named_params: false,
                    has_default_params: false,
                    async_kind: AsyncKind::Sync,
                    calling_convention: CallingConvention::Default,
                    nullability: NullabilityKind::NonNull,
                },
                span,
            );

            let arg = builder.int_literal(42, span);
            builder.call_positional(
                callee,
                vec![arg],
                Type::Primitive(PrimitiveType::Unit),
                span,
            )
        };

        let call_stmt = builder.expression_statement(main_call, span);
        let main_body = builder.block(vec![call_stmt], span);

        let main_func = TypedNode {
            node: TypedDeclaration::Function(TypedFunction {
                name: main_name,
                params: vec![],
                type_params: vec![],
                body: Some(main_body),
                return_type: Type::Primitive(PrimitiveType::Unit),
                visibility: Visibility::Public,
                is_async: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
                ..Default::default()
            }),
            ty: Type::Function {
                params: vec![],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            span,
        };

        let program = builder.program(vec![function, main_func], span);

        checker.check_program(&program);

        // The program should type check successfully with the constraint solver integration
        assert!(
            !checker.has_errors(),
            "Constraint solver integration should work correctly"
        );
    }

    #[test]
    fn test_mixed_positional_and_named_args() {
        let (env, _arena) = create_test_environment();
        let mut checker = TypeChecker::new(Box::new(env));
        let span = create_test_span();

        let mut builder = TypedASTBuilder::new();

        // Create the process function with mixed parameter styles
        let data_param = builder.parameter(
            "data",
            Type::Primitive(PrimitiveType::String),
            Mutability::Immutable,
            span,
        );
        let debug_default = builder.bool_literal(false, span);
        let debug_param = builder.optional_parameter(
            "debug",
            Type::Primitive(PrimitiveType::Bool),
            Mutability::Immutable,
            debug_default,
            span,
        );
        let timeout_default = builder.int_literal(3000, span);
        let timeout_param = builder.optional_parameter(
            "timeout",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            timeout_default,
            span,
        );

        let process_body = builder.block(vec![], span);
        let process_func = builder.function(
            "process",
            vec![data_param, debug_param, timeout_param],
            Type::Primitive(PrimitiveType::Unit),
            process_body,
            Visibility::Public,
            false,
            span,
        );

        let function_type = Type::Function {
            params: vec![
                ParamInfo {
                    name: Some(builder.intern("data")),
                    ty: Type::Primitive(PrimitiveType::String),
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
                ParamInfo {
                    name: Some(builder.intern("debug")),
                    ty: Type::Primitive(PrimitiveType::Bool),
                    is_optional: true,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
                ParamInfo {
                    name: Some(builder.intern("timeout")),
                    ty: Type::Primitive(PrimitiveType::I32),
                    is_optional: true,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
            ],
            return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
            is_varargs: false,
            has_named_params: true,
            has_default_params: true,
            async_kind: AsyncKind::Sync,
            calling_convention: CallingConvention::Default,
            nullability: NullabilityKind::NonNull,
        };

        // Call with positional + named arguments
        let callee = builder.variable("process", function_type, span);
        let pos_arg = builder.string_literal("input_data", span);
        let named_arg1 = builder.bool_literal(true, span);
        let named_arg2 = builder.int_literal(5000, span);

        let mixed_call = builder.call_mixed(
            callee,
            vec![pos_arg],                                        // Positional
            vec![("debug", named_arg1), ("timeout", named_arg2)], // Named
            vec![],                                               // Generic args
            Type::Primitive(PrimitiveType::Unit),
            span,
        );

        let call_stmt = builder.expression_statement(mixed_call, span);
        let main_body = builder.block(vec![call_stmt], span);

        let main_func = TypedNode {
            node: TypedDeclaration::Function(TypedFunction {
                name: builder.intern("main"),
                params: vec![],
                type_params: vec![],
                body: Some(main_body),
                return_type: Type::Primitive(PrimitiveType::Unit),
                visibility: Visibility::Public,
                is_async: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
                ..Default::default()
            }),
            ty: Type::Function {
                params: vec![],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            span,
        };

        let program = builder.program(vec![process_func, main_func], span);

        checker.check_program(&program);
        if checker.has_errors() {
            println!("Type checker reported errors:");
            for diagnostic in checker.diagnostics().diagnostics() {
                println!("  {:?}", diagnostic);
            }
        }
        assert!(
            !checker.has_errors(),
            "Mixed positional and named arguments should type check successfully"
        );
    }
}
