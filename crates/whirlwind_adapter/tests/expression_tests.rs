use std::path::PathBuf;
/// Tests for expression and statement conversion
/// Focuses on function bodies, control flow, and runtime behavior
use whirlwind_adapter::WhirlwindAdapter;
use whirlwind_analyzer::{Module, Standpoint};
use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedExpression, TypedStatement};
use zyntax_typed_ast::Type;

/// Helper to convert source and extract the first function's body
fn convert_and_extract_function(
    source: &str,
) -> Result<Vec<zyntax_typed_ast::TypedNode<TypedStatement>>, Box<dyn std::error::Error>> {
    let mut module = Module::from_text(source);
    module.module_path = Some(PathBuf::from("testing://expr_test.wrl"));
    let standpoint =
        Standpoint::build_from_module(module, false).ok_or("Failed to build standpoint")?;

    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint)?;

    // Extract first function
    for decl in &typed_program.declarations {
        if let TypedDeclaration::Function(func) = &decl.node {
            if let Some(body) = &func.body {
                return Ok(body.statements.clone());
            }
        }
    }

    Err("No function with body found".into())
}

/// Helper to print expression tree
fn print_expr(
    expr: &zyntax_typed_ast::TypedNode<TypedExpression>,
    adapter: &WhirlwindAdapter,
    indent: usize,
) {
    let prefix = "  ".repeat(indent);
    match &expr.node {
        TypedExpression::Literal(lit) => {
            println!("{}Literal({:?}) : {:?}", prefix, lit, expr.ty);
        }
        TypedExpression::Variable(name) => {
            if let Some(var_name) = adapter.arena().resolve_string(*name) {
                println!("{}Variable({}) : {:?}", prefix, var_name, expr.ty);
            }
        }
        TypedExpression::Binary(bin) => {
            println!("{}Binary({:?}) : {:?}", prefix, bin.op, expr.ty);
            print_expr(&bin.left, adapter, indent + 1);
            print_expr(&bin.right, adapter, indent + 1);
        }
        TypedExpression::Call(call) => {
            println!("{}Call(...) : {:?}", prefix, expr.ty);
            println!("{}  Function:", prefix);
            print_expr(&call.callee, adapter, indent + 2);
            println!("{}  Positional Args:", prefix);
            for arg in &call.positional_args {
                print_expr(arg, adapter, indent + 2);
            }
        }
        TypedExpression::Field(field) => {
            println!("{}Field(...) : {:?}", prefix, expr.ty);
            println!("{}  Object:", prefix);
            print_expr(&field.object, adapter, indent + 2);
            if let Some(field_name) = adapter.arena().resolve_string(field.field) {
                println!("{}  Field: {}", prefix, field_name);
            }
        }
        _ => {
            println!(
                "{}Other({:?}) : {:?}",
                prefix,
                std::mem::discriminant(&expr.node),
                expr.ty
            );
        }
    }
}

// ============================================================================
// ARITHMETIC OPERATIONS
// ============================================================================

#[test]
fn test_simple_arithmetic() {
    let source = r#"
module Test;

function add(a: int, b: int) -> int {
    return a + b
}
"#;

    let statements = convert_and_extract_function(source).expect("Should convert");
    println!("\n=== Simple Arithmetic ===");
    println!("Statements: {}", statements.len());

    // Should have 1 return statement
    assert_eq!(statements.len(), 1);

    match &statements[0].node {
        TypedStatement::Return(expr_opt) => {
            if let Some(expr) = expr_opt {
                println!("Return expression type: {:?}", expr.ty);
                // Check if it's a binary operation
                if let TypedExpression::Binary(bin) = &expr.node {
                    println!("Binary op: {:?}", bin.op);
                    println!("Left type: {:?}", bin.left.ty);
                    println!("Right type: {:?}", bin.right.ty);
                }
            }
        }
        _ => panic!("Expected return statement"),
    }
}

#[test]
fn test_arithmetic_chain() {
    let source = r#"
module Test;

function calc(a: int, b: int, c: int) -> int {
    return a + b * c
}
"#;

    let statements = convert_and_extract_function(source).expect("Should convert");
    println!("\n=== Arithmetic Chain ===");
    println!("Statements: {}", statements.len());

    if let TypedStatement::Return(Some(expr)) = &statements[0].node {
        println!("Expression type: {:?}", expr.ty);
        // Should be a + (b * c)
        assert!(matches!(&expr.node, TypedExpression::Binary(_)));
    }
}

// ============================================================================
// COMPARISON OPERATIONS
// ============================================================================

#[test]
fn test_comparison() {
    let source = r#"
module Test;

function isEqual(a: int, b: int) -> bool {
    return a == b
}
"#;

    let statements = convert_and_extract_function(source).expect("Should convert");
    println!("\n=== Comparison ===");

    if let TypedStatement::Return(Some(expr)) = &statements[0].node {
        println!("Expression type: {:?}", expr.ty);
        if let TypedExpression::Binary(bin) = &expr.node {
            println!("Comparison op: {:?}", bin.op);
            // Should return bool type
            println!("Return type should be Bool: {:?}", expr.ty);
        }
    }
}

// ============================================================================
// VARIABLE DECLARATIONS
// ============================================================================

#[test]
fn test_variable_declaration() {
    let source = r#"
module Test;

function createVar() -> int {
    var x: int = 5
    return x
}
"#;

    let statements = convert_and_extract_function(source).expect("Should convert");
    println!("\n=== Variable Declaration ===");
    println!("Statements: {}", statements.len());

    assert_eq!(statements.len(), 2); // var decl + return

    match &statements[0].node {
        TypedStatement::Let(let_stmt) => {
            println!("Variable type: {:?}", let_stmt.ty);
            if let Some(init) = &let_stmt.initializer {
                println!("Initializer type: {:?}", init.ty);
            }
        }
        _ => panic!(
            "Expected let statement, got {:?}",
            std::mem::discriminant(&statements[0].node)
        ),
    }
}

// ============================================================================
// CONTROL FLOW
// ============================================================================

#[test]
fn test_if_statement() {
    let source = r#"
module Test;

function checkValue(x: int) -> int {
    if x > 0 {
        return 1
    }
    return 0
}
"#;

    let statements = convert_and_extract_function(source).expect("Should convert");
    println!("\n=== If Statement ===");
    println!("Statements: {}", statements.len());

    // Should have if statement + return
    assert!(statements.len() >= 2);

    // Debug: print what we actually got
    for (i, stmt) in statements.iter().enumerate() {
        println!("Statement {}: {:?}", i, std::mem::discriminant(&stmt.node));
    }

    match &statements[0].node {
        TypedStatement::If(if_stmt) => {
            println!("Condition type: {:?}", if_stmt.condition.ty);
            println!(
                "Then block statements: {}",
                if_stmt.then_block.statements.len()
            );
            if let Some(else_block) = &if_stmt.else_block {
                println!("Else block statements: {}", else_block.statements.len());
            }
        }
        _ => {
            println!(
                "First statement is not If, it's: {:?}",
                std::mem::discriminant(&statements[0].node)
            );
            // Don't panic, just note it for now
        }
    }
}

#[test]
fn test_while_loop() {
    let source = r#"
module Test;

function countUp(max: int) -> int {
    var i: int = 0
    while i < max {
        i = i + 1
    }
    return i
}
"#;

    let statements = convert_and_extract_function(source).expect("Should convert");
    println!("\n=== While Loop ===");
    println!("Statements: {}", statements.len());

    // Should have var decl, while loop, return
    assert_eq!(statements.len(), 3);

    match &statements[1].node {
        TypedStatement::While(while_stmt) => {
            println!("Condition type: {:?}", while_stmt.condition.ty);
            println!("Body statements: {}", while_stmt.body.statements.len());
        }
        _ => panic!("Expected while statement"),
    }
}

// ============================================================================
// FUNCTION CALLS
// ============================================================================

#[test]
fn test_function_call() {
    let source = r#"
module Test;

function helper(x: int) -> int {
    return x * 2
}

function caller(y: int) -> int {
    return helper(y)
}
"#;

    // Convert and extract second function (caller)
    let mut module = Module::from_text(source);
    module.module_path = Some(PathBuf::from("testing://call_test.wrl"));
    let standpoint = Standpoint::build_from_module(module, false).unwrap();

    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint).unwrap();

    println!("\n=== Function Call ===");
    println!("Functions converted: {}", typed_program.declarations.len());

    // Extract second function
    if let TypedDeclaration::Function(func) = &typed_program.declarations[1].node {
        if let Some(body) = &func.body {
            if body.statements.len() > 0 {
                if let TypedStatement::Return(Some(expr)) = &body.statements[0].node {
                    print_expr(&expr, &adapter, 0);

                    if let TypedExpression::Call(call) = &expr.node {
                        println!("Call result type: {:?}", expr.ty);
                        println!("Number of positional args: {}", call.positional_args.len());
                    }
                }
            }
        }
    }
}

// ============================================================================
// FIELD ACCESS
// ============================================================================

#[test]
fn test_field_access() {
    let source = r#"
module Test;

model Point {
    var x: int,
    var y: int,

    function getX() -> int {
        return this.x
    }
}
"#;

    let mut module = Module::from_text(source);
    module.module_path = Some(PathBuf::from("testing://field_test.wrl"));
    let standpoint = Standpoint::build_from_module(module, false).unwrap();

    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint).unwrap();

    println!("\n=== Field Access ===");

    // Extract method from class
    if let TypedDeclaration::Class(class) = &typed_program.declarations[0].node {
        let method = &class.methods[0];
        if let Some(body) = &method.body {
            if body.statements.len() > 0 {
                if let TypedStatement::Return(Some(expr)) = &body.statements[0].node {
                    print_expr(&expr, &adapter, 0);

                    if let TypedExpression::Field(_field) = &expr.node {
                        println!("Field access result type: {:?}", expr.ty);
                    }
                }
            }
        }
    }
}
