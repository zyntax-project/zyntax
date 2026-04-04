#![allow(unused)]
//! # Type Registry API Demonstration
//!
//! This example shows how language implementers can use the TypeRegistry
//! to register built-in complex types while keeping primitive types simple.

use zyntax_typed_ast::{
    type_registry::{
        FieldDef, MethodSig, Mutability, TypeBound, TypeDefinition, TypeId, TypeKind, TypeMetadata,
        TypeParam, TypeRegistry, Variance, Visibility,
    },
    AstArena, PrimitiveType, Span, Type,
};
use zyntax_typed_ast::{AsyncKind, CallingConvention, ConstValue, NullabilityKind};

fn main() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let span = Span::new(0, 0);

    println!("🚀 TypeRegistry API Demo: Language-Agnostic Complex Type Registration\n");

    // 1. Primitives stay as Type::Primitive - no registration needed
    println!("📋 Primitive types (no registration needed):");
    let int_type = Type::Primitive(PrimitiveType::I32);
    let string_type = Type::Primitive(PrimitiveType::String);
    let bool_type = Type::Primitive(PrimitiveType::Bool);
    println!("   ✅ int: {:?}", int_type);
    println!("   ✅ string: {:?}", string_type);
    println!("   ✅ bool: {:?}", bool_type);

    // 2. Register built-in complex types like List<T>
    println!("\n📋 Registering built-in complex types:");

    // Register List<T> type
    let list_type_id = registry.register_struct_type(
        arena.intern_string("List"),
        vec![TypeParam {
            name: arena.intern_string("T"),
            bounds: vec![],
            variance: Variance::Covariant,
            default: None,
            span,
        }],
        vec![], // No fields exposed (implementation detail)
        vec![
            MethodSig {
                name: arena.intern_string("push"),
                type_params: vec![],
                params: vec![],
                return_type: Type::Named {
                    id: TypeId::next(), // Unit type
                    type_args: vec![],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                },
                is_static: false,
                is_async: false,
                visibility: Visibility::Public,
                span,
                where_clause: vec![],
                is_extension: false,
            },
            MethodSig {
                name: arena.intern_string("len"),
                type_params: vec![],
                params: vec![],
                return_type: Type::Named {
                    id: TypeId::next(), // usize type
                    type_args: vec![],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                },
                is_static: false,
                is_async: false,
                visibility: Visibility::Public,
                span,
                where_clause: vec![],
                is_extension: false,
            },
        ],
        vec![], // No constructors
        TypeMetadata::default(),
        span,
    );
    println!("   ✅ Registered List<T> with ID: {:?}", list_type_id);

    // Register HashMap<K, V> type
    let hashmap_type_id = registry.register_struct_type(
        arena.intern_string("HashMap"),
        vec![
            TypeParam {
                name: arena.intern_string("K"),
                bounds: vec![TypeBound::Custom {
                    name: arena.intern_string("Hash"),
                    args: vec![],
                }],
                variance: Variance::Invariant,
                default: None,
                span,
            },
            TypeParam {
                name: arena.intern_string("V"),
                bounds: vec![],
                variance: Variance::Covariant,
                default: None,
                span,
            },
        ],
        vec![],
        vec![
            MethodSig {
                name: arena.intern_string("insert"),
                type_params: vec![],
                params: vec![],
                return_type: Type::Optional(Box::new(Type::TypeVar(
                    zyntax_typed_ast::type_registry::TypeVar::unbound(arena.intern_string("V")),
                ))),
                is_static: false,
                is_async: false,
                visibility: Visibility::Public,
                span,
                where_clause: vec![],
                is_extension: false,
            },
            MethodSig {
                name: arena.intern_string("get"),
                type_params: vec![],
                params: vec![],
                return_type: Type::Optional(Box::new(Type::Reference {
                    ty: Box::new(Type::TypeVar(zyntax_typed_ast::TypeVar::unbound(
                        arena.intern_string("V"),
                    ))),
                    mutability: Mutability::Immutable,
                    lifetime: None,
                    nullability: NullabilityKind::NonNull,
                })),
                is_static: false,
                is_async: false,
                visibility: Visibility::Public,
                span,
                where_clause: vec![],
                is_extension: false,
            },
        ],
        vec![],
        TypeMetadata::default(),
        span,
    );
    println!(
        "   ✅ Registered HashMap<K, V> with ID: {:?}",
        hashmap_type_id
    );

    // 3. Language implementers can create types using the registry
    println!("\n📋 Creating complex types:");

    // List<String>
    let list_of_strings = registry.make_type(
        list_type_id,
        vec![Type::Primitive(PrimitiveType::String).into()], // Convert to Type
    );
    println!("   ✅ List<String>: {:?}", list_of_strings);

    // HashMap<String, i32>
    let string_to_int_map = registry.make_type(
        hashmap_type_id,
        vec![
            Type::Primitive(PrimitiveType::String).into(),
            Type::Primitive(PrimitiveType::I32).into(),
        ],
    );
    println!("   ✅ HashMap<String, i32>: {:?}", string_to_int_map);

    // 4. Type lookup by name
    println!("\n📋 Type lookup by name:");

    if let Some(list_def) = registry.get_type_by_name(arena.intern_string("List")) {
        println!("   ✅ Found List type definition:");
        println!("      - Name: {:?}", list_def.name);
        println!("      - Kind: {:?}", list_def.kind);
        println!(
            "      - Type params: {} ({})",
            list_def.type_params.len(),
            list_def
                .type_params
                .iter()
                .map(|p| format!("{:?}", p.name))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "      - Methods: {} ({})",
            list_def.methods.len(),
            list_def
                .methods
                .iter()
                .map(|m| format!("{:?}", m.name))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // 5. Language-specific usage examples
    println!("\n📋 Language-specific usage examples:");

    println!("\n   🦀 Rust-style:");
    println!("      let my_list: List<String> = List::new();");
    println!("      let my_map: HashMap<String, i32> = HashMap::new();");

    println!("\n   ☕ Java-style:");
    println!("      List<String> myList = new ArrayList<>();");
    println!("      HashMap<String, Integer> myMap = new HashMap<>();");

    println!("\n   🔷 TypeScript-style:");
    println!("      let myList: List<string> = new List<string>();");
    println!("      let myMap: Map<string, number> = new Map<string, number>();");

    println!("\n   💎 C#-style:");
    println!("      List<string> myList = new List<string>();");
    println!("      Dictionary<string, int> myDict = new Dictionary<string, int>();");

    println!("\n✨ Key Benefits:");
    println!("   🎯 Language implementers define their own complex types");
    println!("   🔧 No hardcoded assumptions about syntax or naming");
    println!("   📝 Primitive types remain simple and universal");
    println!("   🚀 Complex generic types are fully supported");
    println!("   🌍 Same TypedAST works for multiple target languages");
}
