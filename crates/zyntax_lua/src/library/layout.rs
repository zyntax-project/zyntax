//! Layout support for the Lua library.

use super::*;

pub fn table_tag() -> i64 {
    zyntax_builtins::instance_tag(TABLE_KIND)
}

pub fn thread_tag() -> i64 {
    zyntax_builtins::instance_tag(THREAD_KIND)
}

pub fn nil_error_tag() -> i64 {
    zyntax_builtins::instance_tag(NIL_ERROR_KIND)
}

pub fn file_tag() -> i64 {
    zyntax_builtins::instance_tag(FILE_KIND)
}

pub fn closing_tag() -> i64 {
    zyntax_builtins::instance_tag(CLOSING_KIND)
}

pub fn userdata_tag() -> i64 {
    zyntax_builtins::instance_tag(USERDATA_KIND)
}

pub fn light_tag() -> i64 {
    zyntax_builtins::instance_tag(LIGHT_KIND)
}

pub fn is_closing(x: Expr) -> Expr {
    and(ne(x.clone(), nil()), eq(tag_of(x), int(closing_tag())))
}

pub fn closing_marker() -> Expr {
    call(
        "zb_box_instance_raw",
        vec![int(1), int32(closing_tag() as i32)],
        any(),
    )
}

/// `os.exit(code, true)` in flight: the closing kind, so nothing catches
/// it, with `code` above the low two bits (10, where a coroutine's close
/// has 01). The host exits with `code` once it reaches the entry.
pub fn exit_marker(code: Expr) -> Expr {
    call(
        "zb_box_instance_raw",
        vec![
            bitor(shl(code, int(2)), int(2)),
            int32(closing_tag() as i32),
        ],
        any(),
    )
}

pub fn is_exiting(x: Expr) -> Expr {
    and(
        is_closing(x.clone()),
        eq(bitand(exit_payload(x), int(3)), int(2)),
    )
}

/// The status an exit marker carries.
pub fn exit_code(x: Expr) -> Expr {
    shr(exit_payload(x), int(2))
}

pub(super) fn exit_payload(x: Expr) -> Expr {
    call("zb_unbox_instance_raw", vec![x], i64())
}

/// The table struct's type, a pointer to a heap object.
pub fn table_ty(id: TypeId) -> Type {
    Type::Named {
        id,
        type_args: Vec::new(),
        const_args: Vec::new(),
        variance: Vec::new(),
        nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
    }
}

pub(super) fn table_field_type(name: &str, id: TypeId) -> Type {
    match name {
        "meta" => table_ty(id),
        "high" | "shape" | "present" => i64(),
        _ => any(),
    }
}

/// The table header's fields, with their types.
pub fn table_header_fields(id: TypeId) -> Vec<(String, Type)> {
    TABLE_FIELDS
        .iter()
        .map(|name| (name.to_string(), table_field_type(name, id)))
        .collect()
}

/// Register the table struct in `registry` and return its id.
pub fn declare_table_type(registry: &mut zyntax_typed_ast::TypeRegistry) -> TypeId {
    let id = TypeId::next();
    declare_reference_struct(registry, id, TABLE_TYPE, &table_header_fields(id));
    id
}

/// Register a heap struct named `name` with `fields` under `id`.
pub fn declare_reference_struct(
    registry: &mut zyntax_typed_ast::TypeRegistry,
    id: TypeId,
    name: &str,
    fields: &[(String, Type)],
) {
    let fields: Vec<FieldDef> = fields
        .iter()
        .map(|(name, ty)| FieldDef {
            name: intern(name),
            ty: ty.clone(),
            visibility: Visibility::Public,
            mutability: Mutability::Mutable,
            is_static: false,
            span: SPAN,
            getter: None,
            setter: None,
            is_synthetic: false,
        })
        .collect();
    registry.register_type(zyntax_typed_ast::type_registry::TypeDefinition {
        id,
        module: None,
        name: intern(name),
        kind: zyntax_typed_ast::type_registry::TypeKind::Struct {
            fields: fields.clone(),
            is_tuple: false,
        },
        type_params: Vec::new(),
        constraints: Vec::new(),
        fields,
        methods: Vec::new(),
        constructors: Vec::new(),
        metadata: TypeMetadata {
            is_reference: true,
            // A table stays a foreign object to a host heap, and its
            // header fields are read in place (`host_gc::TableHeader`).
            custom: [(
                intern(zyntax_compiler::object_header::HEADER_FREE_KEY),
                String::new(),
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        },
        span: SPAN,
    });
}

/// The struct's declaration, so the lowering lays it out.
pub fn table_class(id: TypeId) -> Decl {
    reference_struct_class(TABLE_TYPE, &table_header_fields(id))
}

/// A heap struct's declaration: every holder shares the one object.
pub fn reference_struct_class(name: &str, fields: &[(String, Type)]) -> Decl {
    typed_node_decl(TypedDeclaration::Class(TypedClass {
        name: intern(name),
        type_params: Vec::new(),
        extends: None,
        implements: Vec::new(),
        fields: fields
            .iter()
            .map(|(name, ty)| TypedField {
                name: intern(name),
                ty: ty.clone(),
                initializer: None,
                visibility: Visibility::Public,
                mutability: Mutability::Mutable,
                is_static: false,
                span: SPAN,
            })
            .collect(),
        methods: Vec::new(),
        constructors: Vec::new(),
        visibility: Visibility::Public,
        is_abstract: false,
        is_final: false,
        // Every holder shares the one heap object.
        annotations: vec![TypedAnnotation {
            name: intern("reference"),
            args: Vec::new(),
            span: SPAN,
        }],
        span: SPAN,
    }))
}

pub(super) fn typed_node_decl(d: TypedDeclaration) -> Decl {
    zyntax_typed_ast::TypedNode::new(d, Type::Unknown, SPAN)
}
