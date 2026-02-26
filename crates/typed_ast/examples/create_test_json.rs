// Create a proper TypedAST JSON file using the TypedASTBuilder API

use std::fs;
use zyntax_typed_ast::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = TypedASTBuilder::new();
    let span = Span::new(0, 10);

    // fn main() -> i32 { return 42; }
    let forty_two = builder.int_literal(42, span);
    let return_stmt = builder.return_stmt(forty_two, span);

    let main_func = builder.function(
        "main",
        vec![],
        Type::Primitive(PrimitiveType::I32),
        TypedBlock {
            statements: vec![return_stmt],
            span,
        },
        Visibility::Public,
        false,
        span,
    );

    let program = builder.program(vec![main_func], span);

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&program)?;

    // Create test_data directory if it doesn't exist
    std::fs::create_dir_all("../../test_data")?;
    fs::write("../../test_data/simple_main.json", &json)?;

    println!("✅ Created test_data/simple_main.json");
    println!("\nJSON content:");
    println!("{}", json);
    println!("\nTest with:");
    println!("./target/release/zyntax compile test_data/simple_main.json --run -v");

    Ok(())
}
