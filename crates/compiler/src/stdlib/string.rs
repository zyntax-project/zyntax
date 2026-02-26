use crate::hir::{BinaryOp, HirId, HirType};
/// String type implementation using HIR Builder
///
/// Provides a UTF-8 string type backed by vec_u8:
/// ```
/// struct String {
///     bytes: Vec_u8,  // UTF-8 byte storage
/// }
/// ```
use crate::hir_builder::HirBuilder;

/// Helper to get common types used across String functions
fn get_string_types(builder: &mut HirBuilder) -> (HirType, HirType, HirType, HirType) {
    let usize_ty = builder.u64_type();
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
    let vec_u8_ty = builder.struct_type(
        Some("Vec_u8"),
        vec![ptr_u8_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let string_ty = builder.struct_type(Some("String"), vec![vec_u8_ty.clone()]);

    (usize_ty, vec_u8_ty, string_ty, ptr_u8_ty)
}

/// Builds the String type and its methods
pub fn build_string_type(builder: &mut HirBuilder) {
    // Build String methods
    build_new(builder);
    build_with_capacity(builder);
    build_len(builder);
    build_capacity(builder);
    build_as_ptr(builder);
    build_push(builder);
    build_clear(builder);
    build_free(builder);
}

/// Builds: fn string_new() -> String
/// Creates an empty string with default capacity
fn build_new(builder: &mut HirBuilder) {
    let (_, _, string_ty, _) = get_string_types(builder);

    let func_id = builder
        .begin_function("string_new")
        .returns(string_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Call vec_u8_new to create empty vector
    let vec_u8_new_name = builder.intern("vec_u8_new");
    let vec_u8_new_ref = builder.get_function_by_name(vec_u8_new_name);
    let vec_u8_new_func = builder.function_ref(vec_u8_new_ref);
    let vec_bytes = builder.call(vec_u8_new_func, vec![]).unwrap();

    // Create String struct with the vec
    let string_val = builder.create_struct(string_ty, vec![vec_bytes]);
    builder.ret(string_val);
}

/// Builds: fn string_with_capacity(cap: u64) -> String
/// Creates an empty string with specified capacity
/// NOTE: Currently just creates an empty string, ignoring capacity since vec_u8_with_capacity doesn't exist yet
fn build_with_capacity(builder: &mut HirBuilder) {
    let (usize_ty, _, string_ty, _) = get_string_types(builder);

    let func_id = builder
        .begin_function("string_with_capacity")
        .param("cap", usize_ty.clone())
        .returns(string_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let _cap = builder.get_param(0); // Unused for now

    // Call vec_u8_new (ignoring capacity for now)
    let vec_u8_new_name = builder.intern("vec_u8_new");
    let vec_u8_new_ref = builder.get_function_by_name(vec_u8_new_name);
    let vec_u8_new_func = builder.function_ref(vec_u8_new_ref);
    let vec_bytes = builder.call(vec_u8_new_func, vec![]).unwrap();

    // Create String struct
    let string_val = builder.create_struct(string_ty, vec![vec_bytes]);
    builder.ret(string_val);
}

/// Builds: fn string_len(s: *String) -> u64
/// Returns the length in bytes
fn build_len(builder: &mut HirBuilder) {
    let (usize_ty, vec_u8_ty, string_ty, _) = get_string_types(builder);
    let ptr_string_ty = builder.ptr_type(string_ty.clone());

    let func_id = builder
        .begin_function("string_len")
        .param("s", ptr_string_ty.clone())
        .returns(usize_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s_ptr = builder.get_param(0);

    // Load String
    let s = builder.load(s_ptr, string_ty.clone());

    // Extract bytes field
    let vec_bytes = builder.extract_struct_field(s, 0, vec_u8_ty.clone());

    // Extract len field from Vec_u8
    let len = builder.extract_struct_field(vec_bytes, 1, usize_ty);

    builder.ret(len);
}

/// Builds: fn string_capacity(s: *String) -> u64
/// Returns the capacity in bytes
fn build_capacity(builder: &mut HirBuilder) {
    let (usize_ty, vec_u8_ty, string_ty, _) = get_string_types(builder);
    let ptr_string_ty = builder.ptr_type(string_ty.clone());

    let func_id = builder
        .begin_function("string_capacity")
        .param("s", ptr_string_ty.clone())
        .returns(usize_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s_ptr = builder.get_param(0);

    // Load String
    let s = builder.load(s_ptr, string_ty.clone());

    // Extract bytes field
    let vec_bytes = builder.extract_struct_field(s, 0, vec_u8_ty.clone());

    // Extract cap field from Vec_u8
    let cap = builder.extract_struct_field(vec_bytes, 2, usize_ty);

    builder.ret(cap);
}

/// Builds: fn string_as_ptr(s: *String) -> *u8
/// Returns pointer to the byte data (for FFI)
fn build_as_ptr(builder: &mut HirBuilder) {
    let (_, vec_u8_ty, string_ty, ptr_u8_ty) = get_string_types(builder);
    let ptr_string_ty = builder.ptr_type(string_ty.clone());

    let func_id = builder
        .begin_function("string_as_ptr")
        .param("s", ptr_string_ty.clone())
        .returns(ptr_u8_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s_ptr = builder.get_param(0);

    // Load String
    let s = builder.load(s_ptr, string_ty.clone());

    // Extract bytes field
    let vec_bytes = builder.extract_struct_field(s, 0, vec_u8_ty.clone());

    // Extract ptr field from Vec_u8
    let ptr = builder.extract_struct_field(vec_bytes, 0, ptr_u8_ty);

    builder.ret(ptr);
}

/// Builds: fn string_push(s: *String, byte: u8)
/// Appends a single byte to the string
fn build_push(builder: &mut HirBuilder) {
    let (_, vec_u8_ty, string_ty, _) = get_string_types(builder);
    let u8_ty = builder.u8_type();
    let ptr_string_ty = builder.ptr_type(string_ty.clone());
    let ptr_vec_u8_ty = builder.ptr_type(vec_u8_ty.clone());
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("string_push")
        .param("s", ptr_string_ty.clone())
        .param("byte", u8_ty.clone())
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s_ptr = builder.get_param(0);
    let byte = builder.get_param(1);

    // Get pointer to bytes field (first field of String)
    let zero = builder.const_u64(0);
    let vec_ptr = builder.ptr_add(s_ptr, zero, ptr_string_ty.clone());
    let vec_ptr_typed = builder.bitcast(vec_ptr, ptr_vec_u8_ty.clone());

    // Call vec_u8_push
    let vec_u8_push_name = builder.intern("vec_u8_push");
    let vec_u8_push_ref = builder.get_function_by_name(vec_u8_push_name);
    let vec_u8_push_func = builder.function_ref(vec_u8_push_ref);
    builder.call(vec_u8_push_func, vec![vec_ptr_typed, byte]);

    builder.ret_void();
}

/// Builds: fn string_clear(s: *String)
/// Clears the string (sets len to 0)
fn build_clear(builder: &mut HirBuilder) {
    let (_, vec_u8_ty, string_ty, _) = get_string_types(builder);
    let ptr_string_ty = builder.ptr_type(string_ty.clone());
    let ptr_vec_u8_ty = builder.ptr_type(vec_u8_ty.clone());
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("string_clear")
        .param("s", ptr_string_ty.clone())
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s_ptr = builder.get_param(0);

    // Get pointer to bytes field
    let zero = builder.const_u64(0);
    let vec_ptr = builder.ptr_add(s_ptr, zero, ptr_string_ty.clone());
    let vec_ptr_typed = builder.bitcast(vec_ptr, ptr_vec_u8_ty.clone());

    // Call vec_u8_clear
    let vec_u8_clear_name = builder.intern("vec_u8_clear");
    let vec_u8_clear_ref = builder.get_function_by_name(vec_u8_clear_name);
    let vec_u8_clear_func = builder.function_ref(vec_u8_clear_ref);
    builder.call(vec_u8_clear_func, vec![vec_ptr_typed]);

    builder.ret_void();
}

/// Builds: fn string_free(s: String)
/// Deallocates the string (consumes by value)
fn build_free(builder: &mut HirBuilder) {
    let (_, vec_u8_ty, string_ty, _) = get_string_types(builder);
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("string_free")
        .param("s", string_ty.clone())
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s = builder.get_param(0);

    // Extract bytes field
    let vec_bytes = builder.extract_struct_field(s, 0, vec_u8_ty.clone());

    // Call vec_u8_free
    let vec_u8_free_name = builder.intern("vec_u8_free");
    let vec_u8_free_ref = builder.get_function_by_name(vec_u8_free_name);
    let vec_u8_free_func = builder.function_ref(vec_u8_free_ref);
    builder.call(vec_u8_free_func, vec![vec_bytes]);

    builder.ret_void();
}

#[cfg(test)]
mod tests {
    // NOTE: String tests are in stdlib/mod.rs because String depends on vec_u8
    // and needs the full stdlib context to work properly.
}
