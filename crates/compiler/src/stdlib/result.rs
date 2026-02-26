use crate::hir::{BinaryOp, HirId, HirType, HirUnionVariant};
/// Result<T,E> type implementation using HIR Builder
///
/// Provides a generic result type for error handling:
/// ```
/// union Result<T, E> {
///     Ok(T),    // Success value
///     Err(E),   // Error value
/// }
/// ```
use crate::hir_builder::HirBuilder;

/// Builds the Result<T,E> type and its methods
pub fn build_result_type(builder: &mut HirBuilder) {
    // Build generic Result<T,E> methods
    build_unwrap(builder);
    build_unwrap_err(builder);
    build_is_ok(builder);
    build_is_err(builder);
    build_unwrap_or(builder);
}

/// Builds: fn result_unwrap<T,E>(res: Result<T,E>) -> T
/// Returns Ok value, panics if Err
fn build_unwrap(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let e_param = builder.type_param("E");

    let variants = vec![
        HirUnionVariant {
            name: builder.intern("Ok"),
            ty: t_param.clone(),
            discriminant: 0,
        },
        HirUnionVariant {
            name: builder.intern("Err"),
            ty: e_param,
            discriminant: 1,
        },
    ];
    let result_ty = builder.union_type(Some("Result"), variants);

    let func_id = builder
        .begin_generic_function("result_unwrap", vec!["T", "E"])
        .param("res", result_ty.clone())
        .returns(t_param.clone())
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let ok_block = builder.create_block("ok_case");
    let err_block = builder.create_block("err_case");

    builder.set_insert_point(entry);

    // Get parameter and extract discriminant
    let res = builder.get_param(0);
    let discriminant = builder.extract_discriminant(res);
    let zero = builder.const_i32(0);

    // Check if discriminant == 0 (Ok)
    let bool_ty = builder.bool_type();
    let is_ok = builder.icmp(BinaryOp::Eq, discriminant, zero, bool_ty);
    builder.cond_br(is_ok, ok_block, err_block);

    // Ok case: extract and return the value
    builder.set_insert_point(ok_block);
    let value = builder.extract_union_value(res, 0, t_param);
    builder.ret(value);

    // Err case: panic
    builder.set_insert_point(err_block);
    builder.panic();
    builder.unreachable();
}

/// Builds: fn result_unwrap_err<T,E>(res: Result<T,E>) -> E
/// Returns Err value, panics if Ok
fn build_unwrap_err(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let e_param = builder.type_param("E");

    let variants = vec![
        HirUnionVariant {
            name: builder.intern("Ok"),
            ty: t_param,
            discriminant: 0,
        },
        HirUnionVariant {
            name: builder.intern("Err"),
            ty: e_param.clone(),
            discriminant: 1,
        },
    ];
    let result_ty = builder.union_type(Some("Result"), variants);

    let func_id = builder
        .begin_generic_function("result_unwrap_err", vec!["T", "E"])
        .param("res", result_ty.clone())
        .returns(e_param.clone())
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let ok_block = builder.create_block("ok_case");
    let err_block = builder.create_block("err_case");

    builder.set_insert_point(entry);

    // Get parameter and extract discriminant
    let res = builder.get_param(0);
    let discriminant = builder.extract_discriminant(res);
    let one = builder.const_i32(1);

    // Check if discriminant == 1 (Err)
    let bool_ty = builder.bool_type();
    let is_err = builder.icmp(BinaryOp::Eq, discriminant, one, bool_ty);
    builder.cond_br(is_err, err_block, ok_block);

    // Err case: extract and return the error
    builder.set_insert_point(err_block);
    let error = builder.extract_union_value(res, 1, e_param);
    builder.ret(error);

    // Ok case: panic
    builder.set_insert_point(ok_block);
    builder.panic();
    builder.unreachable();
}

/// Builds: fn result_is_ok<T,E>(res: Result<T,E>) -> bool
fn build_is_ok(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let e_param = builder.type_param("E");
    let bool_ty = builder.bool_type();

    let variants = vec![
        HirUnionVariant {
            name: builder.intern("Ok"),
            ty: t_param,
            discriminant: 0,
        },
        HirUnionVariant {
            name: builder.intern("Err"),
            ty: e_param,
            discriminant: 1,
        },
    ];
    let result_ty = builder.union_type(Some("Result"), variants);

    let func_id = builder
        .begin_generic_function("result_is_ok", vec!["T", "E"])
        .param("res", result_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Get discriminant and check if == 0
    let res = builder.get_param(0);
    let discriminant = builder.extract_discriminant(res);
    let zero = builder.const_i32(0);
    let is_ok = builder.icmp(BinaryOp::Eq, discriminant, zero, bool_ty);

    builder.ret(is_ok);
}

/// Builds: fn result_is_err<T,E>(res: Result<T,E>) -> bool
fn build_is_err(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let e_param = builder.type_param("E");
    let bool_ty = builder.bool_type();

    let variants = vec![
        HirUnionVariant {
            name: builder.intern("Ok"),
            ty: t_param,
            discriminant: 0,
        },
        HirUnionVariant {
            name: builder.intern("Err"),
            ty: e_param,
            discriminant: 1,
        },
    ];
    let result_ty = builder.union_type(Some("Result"), variants);

    let func_id = builder
        .begin_generic_function("result_is_err", vec!["T", "E"])
        .param("res", result_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Get discriminant and check if == 1
    let res = builder.get_param(0);
    let discriminant = builder.extract_discriminant(res);
    let one = builder.const_i32(1);
    let is_err = builder.icmp(BinaryOp::Eq, discriminant, one, bool_ty);

    builder.ret(is_err);
}

/// Builds: fn result_unwrap_or<T,E>(res: Result<T,E>, default: T) -> T
fn build_unwrap_or(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let e_param = builder.type_param("E");

    let variants = vec![
        HirUnionVariant {
            name: builder.intern("Ok"),
            ty: t_param.clone(),
            discriminant: 0,
        },
        HirUnionVariant {
            name: builder.intern("Err"),
            ty: e_param,
            discriminant: 1,
        },
    ];
    let result_ty = builder.union_type(Some("Result"), variants);

    let func_id = builder
        .begin_generic_function("result_unwrap_or", vec!["T", "E"])
        .param("res", result_ty.clone())
        .param("default", t_param.clone())
        .returns(t_param.clone())
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let ok_block = builder.create_block("ok_case");
    let err_block = builder.create_block("err_case");

    builder.set_insert_point(entry);

    // Get parameter and extract discriminant
    let res = builder.get_param(0);
    let default_val = builder.get_param(1);
    let discriminant = builder.extract_discriminant(res);
    let zero = builder.const_i32(0);

    // Check if discriminant == 0 (Ok)
    let bool_ty = builder.bool_type();
    let is_ok = builder.icmp(BinaryOp::Eq, discriminant, zero, bool_ty);
    builder.cond_br(is_ok, ok_block, err_block);

    // Ok case: extract and return the value
    builder.set_insert_point(ok_block);
    let value = builder.extract_union_value(res, 0, t_param);
    builder.ret(value);

    // Err case: return default
    builder.set_insert_point(err_block);
    builder.ret(default_val);
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::AstArena;

    #[test]
    fn test_build_result_i32_i32() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_result", &mut arena);

        build_result_type(&mut builder);

        let module = builder.finish();

        // Should have created generic functions
        assert!(module.functions.len() >= 5);

        // Check function names
        let func_names: Vec<String> = module
            .functions
            .values()
            .map(|f| arena.resolve_string(f.name).unwrap().to_string())
            .collect();

        assert!(func_names.contains(&"result_unwrap".to_string()));
        assert!(func_names.contains(&"result_unwrap_err".to_string()));
        assert!(func_names.contains(&"result_is_ok".to_string()));
        assert!(func_names.contains(&"result_is_err".to_string()));
        assert!(func_names.contains(&"result_unwrap_or".to_string()));

        // Verify they're generic (have type params)
        for func in module.functions.values() {
            let name = arena.resolve_string(func.name).unwrap();
            if name.starts_with("result_") {
                assert!(
                    !func.signature.type_params.is_empty(),
                    "Function {} should be generic",
                    name
                );
                assert_eq!(
                    func.signature.type_params.len(),
                    2,
                    "Function {} should have 2 type params (T and E)",
                    name
                );
            }
        }
    }

    #[test]
    fn test_result_i32_i32_unwrap_structure() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_result", &mut arena);

        build_unwrap(&mut builder);

        let module = builder.finish();

        // Find the unwrap function
        let unwrap_func = module
            .functions
            .values()
            .find(|f| arena.resolve_string(f.name) == Some("result_unwrap"))
            .expect("unwrap function should exist");

        // Should have 4 blocks: entry, ok_case, err_case, and panic block
        assert_eq!(unwrap_func.blocks.len(), 4);

        // Should have 1 parameter
        assert_eq!(unwrap_func.signature.params.len(), 1);

        // Should have 1 return type
        assert_eq!(unwrap_func.signature.returns.len(), 1);

        // Should be generic with 2 type params (T and E)
        assert_eq!(unwrap_func.signature.type_params.len(), 2);
    }
}
