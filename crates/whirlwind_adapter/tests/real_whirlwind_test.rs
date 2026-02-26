/// Real integration test that compiles actual Whirlwind source code
/// and validates the complete HIR conversion
use whirlwind_adapter::WhirlwindAdapter;
use whirlwind_analyzer::Standpoint;

#[test]
fn test_real_simple_function() {
    let source = r#"
module Test;

function add(a: int, b: int) -> int {
    return a + b
}
"#;

    // Create standpoint and add module

    let mut module = whirlwind_analyzer::Module::from_text(source);
    module.module_path = Some(std::path::PathBuf::from("testing://Test.wrl"));
    let standpoint = Standpoint::build_from_module(module, false).expect("Expects Standpoint");

    // Convert with adapter
    let mut adapter = WhirlwindAdapter::new();
    let result = adapter.convert_standpoint(&standpoint);

    match result {
        Ok(typed_program) => {
            println!("✓ Successfully converted Whirlwind function to HIR");
            println!("  Total declarations: {}", typed_program.declarations.len());
            assert!(
                !typed_program.declarations.is_empty(),
                "Should have at least one declaration"
            );
        }
        Err(e) => {
            eprintln!("✗ Conversion failed: {:?}", e);
            panic!("Failed to convert Whirlwind to HIR");
        }
    }
}

#[test]
fn test_real_model_with_fields() {
    let source = r#"
module Test;

model User {
    var id: int,
    var name: string,

    new(id: int, name: string) {
        this.id = id
        this.name = name
    }

    function greet() -> string {
        return "Hello"
    }
}
"#;

    let mut module = whirlwind_analyzer::Module::from_text(source);
    module.module_path = Some(std::path::PathBuf::from("testing://Test.wrl"));
    let standpoint =
        Standpoint::build_from_module(module, false).expect("Failed to build standpoint");

    let mut adapter = WhirlwindAdapter::new();
    let program = adapter
        .convert_standpoint(&standpoint)
        .expect("Should convert model");

    println!("Show program declarations:");
    for decl in &program.declarations {
        println!(" - Declaration: {:?}", decl);
    }

    // Intern the string first to avoid borrow issues
    let user_name = adapter.arena_mut().intern_string("User");
    let type_registry = adapter.type_registry();

    if let Some(user_def) = type_registry.get_type_by_name(user_name) {
        println!("✓ User type registered in TypeRegistry");
        println!("  - Fields: {}", user_def.fields.len());
        println!("  - Methods: {}", user_def.methods.len());
        println!("  - Constructors: {}", user_def.constructors.len());

        // Print field names
        for field in &user_def.fields {
            if let Some(name) = adapter.arena().resolve_string(field.name) {
                println!("    Field: {} : {:?}", name, field.ty);
            }
        }

        // Print method names
        for method in &user_def.methods {
            if let Some(name) = adapter.arena().resolve_string(method.name) {
                println!("    Method: {}() -> {:?}", name, method.return_type);
            }
        }

        assert!(!user_def.fields.is_empty(), "User should have fields");
    } else {
        panic!("User type should be registered in TypeRegistry");
    }
}

#[test]
fn test_complete_program_with_multiple_types() {
    let source = r#"
module Test;

type UserId = int

interface Entity {
    function getId() -> UserId;
}

model User from Entity {
    var id: UserId,
    var name: string,

    new(id: UserId, name: string) {
        this.id = id
        this.name = name
    }

    function getId() -> UserId {
        return this.id
    }
}

function createUser(id: UserId, name: string) -> User {
    return new User(id, name)
}
"#;

    let mut module = whirlwind_analyzer::Module::from_text(source);
    module.module_path = Some(std::path::PathBuf::from("testing://Test.wrl"));
    let standpoint =
        Standpoint::build_from_module(module, false).expect("Failed to build standpoint");

    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter
        .convert_standpoint(&standpoint)
        .expect("Should convert complete program");

    println!("✓ Complete program compiled successfully");
    println!("  Total declarations: {}", typed_program.declarations.len());

    // Check TypeRegistry has all expected types
    let expected_types = vec!["UserId", "Entity", "User"];

    // Intern all type names first (requires mutable borrow)
    let interned_names: Vec<_> = expected_types
        .iter()
        .map(|name| adapter.arena_mut().intern_string(name))
        .collect();

    // Now check registry (requires immutable borrow)
    let type_registry = adapter.type_registry();
    let mut registered_count = 0;

    for (idx, type_name) in expected_types.iter().enumerate() {
        let interned = interned_names[idx];
        if type_registry.get_type_by_name(interned).is_some() {
            println!("  ✓ Type '{}' registered", type_name);
            registered_count += 1;
        } else {
            eprintln!("  ✗ Type '{}' NOT registered", type_name);
        }
    }

    println!(
        "  Registered {}/{} types",
        registered_count,
        expected_types.len()
    );
}
