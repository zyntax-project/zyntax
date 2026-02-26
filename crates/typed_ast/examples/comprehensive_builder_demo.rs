//! # Comprehensive TypedAST Builder Demonstration
//!
//! This example showcases the full capabilities of the enhanced TypedAST builder,
//! demonstrating all the patterns and constructs that are now supported.

use zyntax_typed_ast::{
    typed_ast::*, AsyncKind, CallingConvention, Mutability, NullabilityKind, PrimitiveType, Span,
    Type, TypeId, TypedASTBuilder, Variance, Visibility,
};

fn main() {
    println!("🚀 Comprehensive TypedAST Builder Demonstration\n");

    let mut builder = TypedASTBuilder::new();
    let span = Span::new(0, 10);

    // ====== Enhanced Parameter System Demo ======
    println!("📋 Enhanced Parameter System:");

    // Regular parameter
    let regular_param = builder.parameter(
        "x",
        Type::Primitive(PrimitiveType::I32),
        Mutability::Immutable,
        span,
    );
    println!(
        "✅ Regular parameter: {}",
        builder.arena().resolve_string(regular_param.name).unwrap()
    );

    // Optional parameter with default value
    let default_value = builder.int_literal(42, span);
    let optional_param = builder.optional_parameter(
        "y",
        Type::Primitive(PrimitiveType::I32),
        Mutability::Immutable,
        default_value,
        span,
    );
    println!(
        "✅ Optional parameter with default: {}",
        builder.arena().resolve_string(optional_param.name).unwrap()
    );

    // Rest/variadic parameter
    let rest_param = builder.rest_parameter(
        "args",
        Type::Primitive(PrimitiveType::I32),
        Mutability::Immutable,
        span,
    );
    println!(
        "✅ Rest parameter: {}",
        builder.arena().resolve_string(rest_param.name).unwrap()
    );

    // Out parameter (C#-style)
    let out_param = builder.out_parameter("result", Type::Primitive(PrimitiveType::I32), span);
    println!(
        "✅ Out parameter: {}",
        builder.arena().resolve_string(out_param.name).unwrap()
    );

    // Ref parameter (C#-style)
    let ref_param = builder.ref_parameter(
        "value",
        Type::Primitive(PrimitiveType::I32),
        Mutability::Mutable,
        span,
    );
    println!(
        "✅ Ref parameter: {}",
        builder.arena().resolve_string(ref_param.name).unwrap()
    );

    // InOut parameter (Swift-style)
    let inout_param = builder.inout_parameter("data", Type::Primitive(PrimitiveType::I32), span);
    println!(
        "✅ InOut parameter: {}",
        builder.arena().resolve_string(inout_param.name).unwrap()
    );

    // ====== Named Arguments in Function Calls Demo ======
    println!("\n🔧 Named Arguments & Function Calls:");

    let func_var = builder.variable("calculate", Type::Primitive(PrimitiveType::I32), span);
    let arg1 = builder.int_literal(10, span);
    let arg2 = builder.int_literal(20, span);

    // Function call with named arguments
    let named_call = builder.call_named(
        func_var,
        vec![("width", arg1), ("height", arg2)],
        Type::Primitive(PrimitiveType::I32),
        span,
    );
    println!("✅ Named function call: calculate(width: 10, height: 20)");

    // Mixed positional and named arguments
    let func_var2 = builder.variable("process", Type::Primitive(PrimitiveType::String), span);
    let pos_arg = builder.string_literal("data", span);
    let named_arg1 = builder.bool_literal(true, span);
    let named_arg2 = builder.int_literal(100, span);

    let mixed_call = builder.call_mixed(
        func_var2,
        vec![pos_arg],
        vec![("debug", named_arg1), ("timeout", named_arg2)],
        vec![],
        Type::Primitive(PrimitiveType::String),
        span,
    );
    println!("✅ Mixed call: process(\"data\", debug: true, timeout: 100)");

    // ====== Pattern Matching Builders Demo ======
    println!("\n🎯 Advanced Pattern Matching:");

    // Struct pattern
    let px_var = builder.intern("px");
    let py_var = builder.intern("py");
    let struct_pattern = builder.struct_pattern(
        "Point",
        vec![
            (
                "x",
                typed_node(TypedPattern::immutable_var(px_var), Type::Any, span),
            ),
            (
                "y",
                typed_node(TypedPattern::immutable_var(py_var), Type::Any, span),
            ),
        ],
        span,
    );
    println!("✅ Struct pattern: Point {{ x: px, y: py }}");

    // Enum pattern
    let value_var = builder.intern("value");
    let enum_pattern = builder.enum_pattern(
        "Option",
        "Some",
        vec![typed_node(
            TypedPattern::immutable_var(value_var),
            Type::Any,
            span,
        )],
        span,
    );
    println!("✅ Enum pattern: Some(value)");

    // Array pattern
    let first_var = builder.intern("first");
    let array_pattern = builder.array_pattern(
        vec![
            typed_node(TypedPattern::immutable_var(first_var), Type::Any, span),
            typed_node(TypedPattern::wildcard(), Type::Any, span),
        ],
        span,
    );
    println!("✅ Array pattern: [first, _]");

    // Slice pattern with rest
    let middle_var = builder.intern("middle");
    let last_var = builder.intern("last");
    let slice_pattern = builder.slice_pattern(
        vec![typed_node(
            TypedPattern::immutable_var(first_var),
            Type::Any,
            span,
        )],
        Some(typed_node(
            TypedPattern::Rest {
                name: Some(middle_var),
                mutability: Mutability::Immutable,
            },
            Type::Any,
            span,
        )),
        vec![typed_node(
            TypedPattern::immutable_var(last_var),
            Type::Any,
            span,
        )],
        span,
    );
    println!("✅ Slice pattern: [first, ..middle, last]");

    // Map pattern
    let rest_var = builder.intern("rest");
    let n_var = builder.intern("n");
    let a_var = builder.intern("a");
    let map_pattern = builder.map_pattern(
        vec![
            (
                "name",
                typed_node(TypedPattern::immutable_var(n_var), Type::Any, span),
            ),
            (
                "age",
                typed_node(TypedPattern::immutable_var(a_var), Type::Any, span),
            ),
        ],
        Some(("rest", Mutability::Immutable)),
        false,
        span,
    );
    println!("✅ Map pattern: {{ name: n, age: a, ..rest }}");

    // ====== Expression Builders Demo ======
    println!("\n🔧 Expression Building:");

    // Various literals
    let int_expr = builder.int_literal(42, span);
    let str_expr = builder.string_literal("hello world", span);
    let bool_expr = builder.bool_literal(true, span);
    let char_expr = builder.char_literal('🦀', span);
    let unit_expr = builder.unit_literal(span);

    println!("✅ Literals: integer(42), string(\"hello world\"), bool(true), char('🦀'), unit");

    // Binary operations
    let x_var = builder.variable("x", Type::Primitive(PrimitiveType::I32), span);
    let y_var = builder.variable("y", Type::Primitive(PrimitiveType::I32), span);
    let add_expr = builder.binary(
        BinaryOp::Add,
        x_var,
        y_var,
        Type::Primitive(PrimitiveType::I32),
        span,
    );
    println!("✅ Binary expression: x + y");

    // Method call
    let obj_var = builder.variable("obj", Type::Primitive(PrimitiveType::String), span);
    let method_call = builder.method_call(
        obj_var,
        "len",
        vec![],
        Type::Primitive(PrimitiveType::I32),
        span,
    );
    println!("✅ Method call: obj.len()");

    // Struct literal
    let x_val = builder.int_literal(10, span);
    let y_val = builder.int_literal(20, span);
    let point_name = builder.intern("Point");
    let struct_lit = builder.struct_literal(
        "Point",
        vec![("x", x_val), ("y", y_val)],
        Type::Named {
            id: TypeId::next(),
            type_args: vec![],
            const_args: Vec::new(),
            variance: Vec::new(),
            nullability: NullabilityKind::NonNull,
        }, // Point type
        span,
    );
    println!("✅ Struct literal: Point {{ x: 10, y: 20 }}");

    // ====== Statement Builders Demo ======
    println!("\n📝 Statement Building:");

    // Let statement
    let init_val = builder.int_literal(0, span);
    let let_stmt = builder.let_statement(
        "counter",
        Type::Primitive(PrimitiveType::I32),
        Mutability::Mutable,
        Some(init_val),
        span,
    );
    println!("✅ Let statement: let mut counter: i32 = 0");

    // If statement
    let condition = builder.bool_literal(true, span);
    let int_lit_1 = builder.int_literal(1, span);
    let then_stmt = builder.expression_statement(int_lit_1, span);
    let then_block = builder.block(vec![then_stmt], span);
    let int_lit_2 = builder.int_literal(2, span);
    let else_stmt = builder.expression_statement(int_lit_2, span);
    let else_block = builder.block(vec![else_stmt], span);
    let if_stmt = builder.if_statement(condition, then_block, Some(else_block), span);
    println!("✅ If statement: if true {{ 1 }} else {{ 2 }}");

    // For loop
    let iterable = builder.variable(
        "items",
        Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: None,
            nullability: NullabilityKind::NonNull,
        },
        span,
    );
    let item_var = builder.variable("item", Type::Primitive(PrimitiveType::I32), span);
    let loop_stmt = builder.expression_statement(item_var, span);
    let loop_body = builder.block(vec![loop_stmt], span);
    let for_stmt = builder.for_loop("item", iterable, loop_body, span);
    println!("✅ For loop: for item in items {{ item }}");

    // ====== Coroutine & Async Builders Demo ======
    println!("\n⚡ Coroutine & Async Features:");

    // Async coroutine
    let async_body = builder.int_literal(42, span);
    let async_stmt = builder.coroutine(CoroutineKind::Async, async_body, vec![], span);
    println!("✅ Async coroutine: async {{ 42 }}");

    // Defer statement
    let cleanup_expr = builder.variable("cleanup", Type::Primitive(PrimitiveType::Unit), span);
    let defer_stmt = builder.defer(cleanup_expr, span);
    println!("✅ Defer statement: defer cleanup");

    // ====== Function Declaration Demo ======
    println!("\n🏗️ Function Declaration:");

    // Build a comprehensive function with various parameter types
    let params = vec![regular_param, optional_param, rest_param];

    let func_stmt = builder.expression_statement(add_expr, span);
    let func_body = builder.block(vec![func_stmt], span);

    let function = builder.function(
        "calculate_sum",
        params,
        Type::Primitive(PrimitiveType::I32),
        func_body,
        Visibility::Public,
        false,
        span,
    );

    println!("✅ Function: pub fn calculate_sum(x: i32, y: i32 = 42, ..args: i32) -> i32");

    // ====== Complete Program Demo ======
    println!("\n🎯 Complete Program:");

    // Create a program with multiple declarations
    let program = builder.program(vec![function], span);

    println!(
        "✅ Program with {} declarations",
        program.declarations.len()
    );

    // ====== Summary ======
    println!("\n🎉 TypedAST Builder Features Summary:");
    println!("📋 Enhanced Parameter System:");
    println!("   • Regular, Optional, Rest, Out, Ref, InOut parameters");
    println!("   • Default values for optional parameters");
    println!("   • Parameter attributes for validation");

    println!("📋 Advanced Function Calls:");
    println!("   • Named arguments support");
    println!("   • Mixed positional and named arguments");
    println!("   • Generic type arguments");

    println!("📋 Comprehensive Pattern Matching:");
    println!("   • Struct, Enum, Array, Slice patterns");
    println!("   • Map patterns with rest syntax");
    println!("   • Wildcard and variable binding patterns");

    println!("📋 Rich Expression System:");
    println!("   • All literal types with proper typing");
    println!("   • Binary and unary operations");
    println!("   • Method calls and field access");
    println!("   • Struct literals and lambdas");

    println!("📋 Complete Statement Coverage:");
    println!("   • Let bindings with mutability");
    println!("   • Control flow (if, while, for, match)");
    println!("   • Expression statements");

    println!("📋 Coroutine & Async Support:");
    println!("   • Async, Generator, Goroutine patterns");
    println!("   • Defer statements for cleanup");
    println!("   • Select statements for channels");

    println!("📋 Type-Safe Construction:");
    println!("   • All nodes carry type and span information");
    println!("   • String interning for efficiency");
    println!("   • Comprehensive test coverage");

    println!("\n✨ Ready for comprehensive language support!");
}
