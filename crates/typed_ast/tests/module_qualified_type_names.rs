//! A type name belongs to the module that declares it.
//!
//! Two modules may spell a type name the same way. The registry keys each
//! definition by `(module, name)`, so merging one module's types into
//! another's keeps both, and a bare name resolves through the scope: the
//! current module first, then its imports, then the unqualified names.

use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{
    FieldDef, Mutability, PrimitiveType, Type, TypeMetadata, TypeRegistry, Visibility,
};
use zyntax_typed_ast::{AstArena, InternedString};

fn field(arena: &mut AstArena, name: &str) -> FieldDef {
    FieldDef {
        name: arena.intern_string(name),
        ty: Type::Primitive(PrimitiveType::F64),
        visibility: Visibility::Public,
        mutability: Mutability::Mutable,
        is_static: false,
        span: Span::new(0, 0),
        getter: None,
        setter: None,
        is_synthetic: false,
    }
}

/// A registry for `module` holding one struct `name` with `field_names`.
fn module_with_struct(module: &str, name: &str, field_names: &[&str]) -> TypeRegistry {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    registry.set_current_module(Some(InternedString::new_global(module)));
    let fields: Vec<FieldDef> = field_names.iter().map(|f| field(&mut arena, f)).collect();
    registry.register_struct_type(
        arena.intern_string(name),
        vec![],
        fields,
        vec![],
        vec![],
        TypeMetadata::default(),
        Span::new(0, 0),
    );
    registry
}

#[test]
fn a_declaration_and_an_imported_declaration_of_the_same_name_both_survive() {
    let mut importer = module_with_struct("main", "Point", &["x", "y"]);
    let imported = module_with_struct("geometry", "Point", &["a", "b", "c"]);
    importer.merge_from(&imported);

    let main_module = InternedString::new_global("main");
    let geometry = InternedString::new_global("geometry");
    let point = InternedString::new_global("Point");

    let mine = importer
        .get_type_in_module(Some(main_module), point)
        .expect("the importing module's `Point` should still be registered");
    let theirs = importer
        .get_type_in_module(Some(geometry), point)
        .expect("the imported module's `Point` should be registered too");

    assert_eq!(
        mine.fields.len(),
        2,
        "importing module's Point has 2 fields"
    );
    assert_eq!(theirs.fields.len(), 3, "imported module's Point has 3");
    assert_ne!(mine.id, theirs.id, "the two `Point`s are distinct types");
}

#[test]
fn a_bare_name_resolves_to_the_current_module_before_an_import() {
    let mut importer = module_with_struct("main", "Point", &["x", "y"]);
    let imported = module_with_struct("geometry", "Point", &["a", "b", "c"]);
    importer.merge_from(&imported);

    let resolved = importer
        .get_type_by_name(InternedString::new_global("Point"))
        .expect("`Point` should resolve");
    assert_eq!(
        resolved.module,
        Some(InternedString::new_global("main")),
        "the current module wins over an import"
    );
    assert_eq!(resolved.fields.len(), 2);
}

#[test]
fn a_bare_name_the_current_module_does_not_declare_falls_to_its_imports() {
    let mut importer = module_with_struct("main", "Vector", &["x"]);
    let imported = module_with_struct("geometry", "Point", &["a", "b", "c"]);
    importer.merge_from(&imported);

    // Merging puts the imported module into the import scope.
    assert_eq!(
        importer.imported_modules(),
        [InternedString::new_global("geometry")]
    );

    let resolved = importer
        .get_type_by_name(InternedString::new_global("Point"))
        .expect("`Point` should resolve through the import scope");
    assert_eq!(
        resolved.module,
        Some(InternedString::new_global("geometry"))
    );
    assert_eq!(resolved.fields.len(), 3);
}

#[test]
fn a_referenced_name_never_outranks_a_declared_one() {
    // A module that only references a name — a generic parameter, say —
    // registers it as an atomic placeholder belonging to no module.
    let mut importer = module_with_struct("main", "P", &["x", "y"]);
    let mut imported = TypeRegistry::new();
    imported.set_current_module(Some(InternedString::new_global("prelude")));
    imported.register_atomic_type(
        InternedString::new_global("P"),
        TypeMetadata::default(),
        Span::new(0, 0),
    );
    importer.merge_from(&imported);

    let resolved = importer
        .get_type_by_name(InternedString::new_global("P"))
        .expect("`P` should resolve");
    assert_eq!(
        resolved.fields.len(),
        2,
        "the declaration outranks the placeholder"
    );
    assert_eq!(resolved.module, Some(InternedString::new_global("main")));
}
