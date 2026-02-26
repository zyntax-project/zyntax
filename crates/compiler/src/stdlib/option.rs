use crate::hir::{BinaryOp, HirId, HirType, HirUnionVariant};
/// Option<T> type implementation using HIR Builder
///
/// Provides a generic optional value type:
/// ```
/// union Option<T> {
///     None,      // No value
///     Some(T),   // Value of type T
/// }
/// ```
use crate::hir_builder::HirBuilder;

/// Builds the Option<T> type and its methods
pub fn build_option_type(builder: &mut HirBuilder) {
    // Build generic Option<T> methods
    build_unwrap(builder);
    build_is_some(builder);
    build_is_none(builder);
    build_unwrap_or(builder);
    build_map(builder);
    build_and_then(builder);
}

/// Builds: fn option_unwrap<T>(opt: Option<T>) -> T
/// Panics if opt is None
fn build_unwrap(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let void_ty = builder.void_type();

    let variants = vec![
        HirUnionVariant {
            name: builder.intern("None"),
            ty: void_ty,
            discriminant: 0,
        },
        HirUnionVariant {
            name: builder.intern("Some"),
            ty: t_param.clone(),
            discriminant: 1,
        },
    ];
    let option_ty = builder.union_type(Some("Option"), variants);

    let func_id = builder
        .begin_generic_function("option_unwrap", vec!["T"])
        .param("opt", option_ty.clone())
        .returns(t_param.clone())
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let some_block = builder.create_block("some_case");
    let none_block = builder.create_block("none_case");

    builder.set_insert_point(entry);

    // Get parameter and extract discriminant
    let opt = builder.get_param(0);
    let discriminant = builder.extract_discriminant(opt);
    let one = builder.const_i32(1);

    // Check if discriminant == 1 (Some)
    let bool_ty = builder.bool_type();
    let is_some = builder.icmp(BinaryOp::Eq, discriminant, one, bool_ty);
    builder.cond_br(is_some, some_block, none_block);

    // Some case: extract and return the value
    builder.set_insert_point(some_block);
    let value = builder.extract_union_value(opt, 1, t_param);
    builder.ret(value);

    // None case: panic (abort)
    builder.set_insert_point(none_block);
    builder.panic();
    builder.unreachable();
}

/// Builds: fn option_is_some<T>(opt: Option<T>) -> bool
fn build_is_some(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let void_ty = builder.void_type();
    let bool_ty = builder.bool_type();

    let variants = vec![
        HirUnionVariant {
            name: builder.intern("None"),
            ty: void_ty,
            discriminant: 0,
        },
        HirUnionVariant {
            name: builder.intern("Some"),
            ty: t_param,
            discriminant: 1,
        },
    ];
    let option_ty = builder.union_type(Some("Option"), variants);

    let func_id = builder
        .begin_generic_function("option_is_some", vec!["T"])
        .param("opt", option_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Get discriminant and check if == 1
    let opt = builder.get_param(0);
    let discriminant = builder.extract_discriminant(opt);
    let one = builder.const_i32(1);
    let is_some = builder.icmp(BinaryOp::Eq, discriminant, one, bool_ty);

    builder.ret(is_some);
}

/// Builds: fn option_is_none<T>(opt: Option<T>) -> bool
fn build_is_none(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let void_ty = builder.void_type();
    let bool_ty = builder.bool_type();

    let variants = vec![
        HirUnionVariant {
            name: builder.intern("None"),
            ty: void_ty,
            discriminant: 0,
        },
        HirUnionVariant {
            name: builder.intern("Some"),
            ty: t_param,
            discriminant: 1,
        },
    ];
    let option_ty = builder.union_type(Some("Option"), variants);

    let func_id = builder
        .begin_generic_function("option_is_none", vec!["T"])
        .param("opt", option_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Get discriminant and check if == 0
    let opt = builder.get_param(0);
    let discriminant = builder.extract_discriminant(opt);
    let zero = builder.const_i32(0);
    let is_none = builder.icmp(BinaryOp::Eq, discriminant, zero, bool_ty);

    builder.ret(is_none);
}

/// Builds: fn option_unwrap_or<T>(opt: Option<T>, default: T) -> T
fn build_unwrap_or(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let void_ty = builder.void_type();

    let variants = vec![
        HirUnionVariant {
            name: builder.intern("None"),
            ty: void_ty,
            discriminant: 0,
        },
        HirUnionVariant {
            name: builder.intern("Some"),
            ty: t_param.clone(),
            discriminant: 1,
        },
    ];
    let option_ty = builder.union_type(Some("Option"), variants);

    let func_id = builder
        .begin_generic_function("option_unwrap_or", vec!["T"])
        .param("opt", option_ty.clone())
        .param("default", t_param.clone())
        .returns(t_param.clone())
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let some_block = builder.create_block("some_case");
    let none_block = builder.create_block("none_case");

    builder.set_insert_point(entry);

    // Get parameter and extract discriminant
    let opt = builder.get_param(0);
    let default_val = builder.get_param(1);
    let discriminant = builder.extract_discriminant(opt);
    let one = builder.const_i32(1);

    // Check if discriminant == 1 (Some)
    let bool_ty = builder.bool_type();
    let is_some = builder.icmp(BinaryOp::Eq, discriminant, one, bool_ty);
    builder.cond_br(is_some, some_block, none_block);

    // Some case: extract and return the value
    builder.set_insert_point(some_block);
    let value = builder.extract_union_value(opt, 1, t_param);
    builder.ret(value);

    // None case: return default
    builder.set_insert_point(none_block);
    builder.ret(default_val);
}

/// Builds: fn option_map<T, U>(opt: Option<T>, f: fn(T) -> U) -> Option<U>
/// Note: Simplified - requires function pointers
fn build_map(builder: &mut HirBuilder) {
    // TODO: Requires function pointer support
    // For now, we'll skip this as it requires more advanced features
    let _ = builder;
}

/// Builds: fn option_and_then<T, U>(opt: Option<T>, f: fn(T) -> Option<U>) -> Option<U>
/// Note: Simplified - requires function pointers
fn build_and_then(builder: &mut HirBuilder) {
    // TODO: Requires function pointer support
    // For now, we'll skip this as it requires more advanced features
    let _ = builder;
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::AstArena;

    #[test]
    fn test_build_option_i32() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_option", &mut arena);

        build_option_type(&mut builder);

        let module = builder.finish();

        // Should have created generic functions
        assert!(module.functions.len() >= 4);

        // Check function names
        let func_names: Vec<String> = module
            .functions
            .values()
            .map(|f| arena.resolve_string(f.name).unwrap().to_string())
            .collect();

        assert!(func_names.contains(&"option_unwrap".to_string()));
        assert!(func_names.contains(&"option_is_some".to_string()));
        assert!(func_names.contains(&"option_is_none".to_string()));
        assert!(func_names.contains(&"option_unwrap_or".to_string()));

        // Verify they're generic (have type params)
        for func in module.functions.values() {
            let name = arena.resolve_string(func.name).unwrap();
            if name.starts_with("option_") && !name.contains("map") && !name.contains("and_then") {
                assert!(
                    !func.signature.type_params.is_empty(),
                    "Function {} should be generic",
                    name
                );
            }
        }
    }

    #[test]
    fn test_option_i32_unwrap_structure() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_option", &mut arena);

        build_unwrap(&mut builder);

        let module = builder.finish();

        // Find the unwrap function
        let unwrap_func = module
            .functions
            .values()
            .find(|f| arena.resolve_string(f.name) == Some("option_unwrap"))
            .expect("unwrap function should exist");

        // Should have 4 blocks: entry, some_case, none_case, and panic block
        assert_eq!(unwrap_func.blocks.len(), 4);

        // Should have 1 parameter
        assert_eq!(unwrap_func.signature.params.len(), 1);

        // Should have 1 return type
        assert_eq!(unwrap_func.signature.returns.len(), 1);

        // Should be generic (have type params)
        assert_eq!(unwrap_func.signature.type_params.len(), 1);
    }
}
