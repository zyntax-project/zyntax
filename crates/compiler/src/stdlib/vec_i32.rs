use crate::hir::{BinaryOp, CallingConvention, HirType};
/// vec_i32: Performance-optimized concrete vector for i32
///
/// This is a specialized implementation for i32 that bypasses generic monomorphization
/// for maximum performance. Use this for hot paths and performance-critical code.
///
/// For generic types, use Vec<T> which will be monomorphized.
///
/// Structure:
/// ```
/// struct vec_i32 {
///     ptr: *i32,      // Pointer to heap-allocated array
///     len: usize,     // Number of elements currently stored
///     cap: usize,     // Capacity of allocated array
/// }
/// ```
///
/// Growth strategy: Double capacity when full (4 → 8 → 16 → 32)
/// Initial capacity: 4 elements
use crate::hir_builder::HirBuilder;

/// Build vec_i32 type and all methods (performance-optimized concrete version)
pub fn build_vec_i32_type(builder: &mut HirBuilder) {
    // Declare C realloc for resizing
    declare_c_realloc(builder);

    // Build all vec_i32 methods
    build_vec_i32_new(builder);
    build_vec_i32_with_capacity(builder);
    build_vec_i32_push(builder);
    build_vec_i32_pop(builder);
    build_vec_i32_get(builder);
    build_vec_i32_set(builder);
    build_vec_i32_len(builder);
    build_vec_i32_capacity(builder);
    build_vec_i32_clear(builder);
    build_vec_i32_free(builder);
}

/// Declare C realloc function
fn declare_c_realloc(builder: &mut HirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let usize_ty = builder.u64_type();

    let _realloc = builder
        .begin_extern_function("realloc", CallingConvention::C)
        .param("ptr", ptr_u8_ty.clone())
        .param("new_size", usize_ty)
        .returns(ptr_u8_ty)
        .build();
}

/// Build: fn vec_i32_new() -> vec_i32
fn build_vec_i32_new(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty);
    let usize_ty = builder.u64_type();

    // struct vec_i32 { ptr: *i32, len: usize, cap: usize }
    let vec_i32_ty = builder.struct_type(
        Some("vec_i32"),
        vec![ptr_i32_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );

    let func_id = builder
        .begin_function("vec_i32_new")
        .returns(vec_i32_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Allocate initial capacity of 4 elements
    let initial_cap = builder.const_u64(4);
    let i32_size = builder.const_u64(4); // sizeof(i32) = 4 bytes
    let usize_ty_copy = usize_ty.clone();
    let alloc_size = builder.mul(initial_cap, i32_size, usize_ty_copy);

    // Call malloc
    let malloc_name = builder.intern("malloc");
    let malloc_id = builder.get_function_by_name(malloc_name);
    let malloc_ref = builder.function_ref(malloc_id);
    let ptr_u8 = builder.call(malloc_ref, vec![alloc_size]).unwrap();

    // Cast *u8 to *i32
    let ptr_i32 = builder.bitcast(ptr_u8, ptr_i32_ty);

    // Build struct: { ptr: ptr_i32, len: 0, cap: 4 }
    let zero = builder.const_u64(0);
    let vec_value = builder.create_struct(vec_i32_ty, vec![ptr_i32, zero, initial_cap]);

    builder.ret(vec_value);
}

/// Build: fn vec_i32_with_capacity(capacity: usize) -> vec_i32
fn build_vec_i32_with_capacity(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty);
    let usize_ty = builder.u64_type();

    let vec_i32_ty = builder.struct_type(
        Some("vec_i32"),
        vec![ptr_i32_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );

    let func_id = builder
        .begin_function("vec_i32_with_capacity")
        .param("capacity", usize_ty.clone())
        .returns(vec_i32_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let capacity = builder.get_param(0);

    // Allocate: capacity * 4 bytes
    let i32_size = builder.const_u64(4);
    let usize_ty_copy = usize_ty.clone();
    let alloc_size = builder.mul(capacity, i32_size, usize_ty_copy);

    // Call malloc
    let malloc_name = builder.intern("malloc");
    let malloc_id = builder.get_function_by_name(malloc_name);
    let malloc_ref = builder.function_ref(malloc_id);
    let ptr_u8 = builder.call(malloc_ref, vec![alloc_size]).unwrap();

    // Cast *u8 to *i32
    let ptr_i32 = builder.bitcast(ptr_u8, ptr_i32_ty);

    // Build struct: { ptr: ptr_i32, len: 0, cap: capacity }
    let zero = builder.const_u64(0);
    let vec_value = builder.create_struct(vec_i32_ty, vec![ptr_i32, zero, capacity]);

    builder.ret(vec_value);
}

/// Build: fn vec_i32_push(vec: *vec_i32, value: i32) -> void
fn build_vec_i32_push(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty.clone());
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let vec_i32_ty = builder.struct_type(
        Some("vec_i32"),
        vec![ptr_i32_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_i32_ty.clone());

    let func_id = builder
        .begin_function("vec_i32_push")
        .param("vec", ptr_vec_ty)
        .param("value", i32_ty.clone())
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    let check_capacity = builder.create_block("check_capacity");
    let need_grow = builder.create_block("need_grow");
    let no_grow = builder.create_block("no_grow");
    let insert_element = builder.create_block("insert_element");

    builder.set_insert_point(entry);
    let vec_ptr = builder.get_param(0);
    let value = builder.get_param(1);

    let vec_val = builder.load(vec_ptr, vec_i32_ty.clone());
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_i32_ty.clone());
    let len_field = builder.extract_struct_field(vec_val, 1, usize_ty.clone());
    let cap_field = builder.extract_struct_field(vec_val, 2, usize_ty.clone());

    builder.br(check_capacity);

    // Check if len == cap (need to grow)
    builder.set_insert_point(check_capacity);
    let bool_ty = builder.bool_type();
    let is_full = builder.icmp(BinaryOp::Eq, len_field, cap_field, bool_ty);
    builder.cond_br(is_full, need_grow, no_grow);

    // Grow: double capacity and realloc
    builder.set_insert_point(need_grow);
    let two = builder.const_u64(2);
    let new_cap = builder.mul(cap_field, two, usize_ty.clone());
    let i32_size = builder.const_u64(4);
    let new_size = builder.mul(new_cap, i32_size, usize_ty.clone());

    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let ptr_u8 = builder.bitcast(ptr_field, ptr_u8_ty.clone());

    let realloc_name = builder.intern("realloc");
    let realloc_id = builder.get_function_by_name(realloc_name);
    let realloc_ref = builder.function_ref(realloc_id);
    let new_ptr_u8 = builder.call(realloc_ref, vec![ptr_u8, new_size]).unwrap();

    let new_ptr = builder.bitcast(new_ptr_u8, ptr_i32_ty.clone());

    let updated_vec = builder.create_struct(vec_i32_ty.clone(), vec![new_ptr, len_field, new_cap]);
    builder.store(updated_vec, vec_ptr);

    builder.br(insert_element);

    // No grow: just continue
    builder.set_insert_point(no_grow);
    builder.br(insert_element);

    // Insert element at vec[len]
    builder.set_insert_point(insert_element);

    let vec_val2 = builder.load(vec_ptr, vec_i32_ty.clone());
    let ptr_field2 = builder.extract_struct_field(vec_val2, 0, ptr_i32_ty.clone());
    let len_field2 = builder.extract_struct_field(vec_val2, 1, usize_ty.clone());
    let cap_field2 = builder.extract_struct_field(vec_val2, 2, usize_ty.clone());

    let elem_ptr = builder.ptr_add(ptr_field2, len_field2, ptr_i32_ty.clone());
    builder.store(value, elem_ptr);

    let one = builder.const_u64(1);
    let new_len = builder.add(len_field2, one, usize_ty.clone());

    let final_vec = builder.create_struct(vec_i32_ty, vec![ptr_field2, new_len, cap_field2]);
    builder.store(final_vec, vec_ptr);

    let unit = builder.unit_value();
    builder.ret(unit);
}

/// Build: fn vec_i32_pop(vec: *vec_i32) -> Option<i32>
/// Returns Some(last element), or None if empty
fn build_vec_i32_pop(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty.clone());
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let vec_i32_ty = builder.struct_type(
        Some("vec_i32"),
        vec![ptr_i32_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_i32_ty.clone());

    // Create Option<i32> type
    let option_variants = vec![
        crate::hir::HirUnionVariant {
            name: builder.intern("None"),
            ty: void_ty.clone(),
            discriminant: 0,
        },
        crate::hir::HirUnionVariant {
            name: builder.intern("Some"),
            ty: i32_ty.clone(),
            discriminant: 1,
        },
    ];
    let option_ty = builder.union_type(Some("Option"), option_variants);

    let func_id = builder
        .begin_function("vec_i32_pop")
        .param("vec", ptr_vec_ty)
        .returns(option_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    let is_empty_check = builder.create_block("is_empty_check");
    let empty_case = builder.create_block("empty_case");
    let non_empty_case = builder.create_block("non_empty_case");

    builder.set_insert_point(entry);
    let vec_ptr = builder.get_param(0);

    let vec_val = builder.load(vec_ptr, vec_i32_ty.clone());
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_i32_ty.clone());
    let len_field = builder.extract_struct_field(vec_val, 1, usize_ty.clone());
    let cap_field = builder.extract_struct_field(vec_val, 2, usize_ty.clone());

    builder.br(is_empty_check);

    // Check if len == 0
    builder.set_insert_point(is_empty_check);
    let zero = builder.const_u64(0);
    let bool_ty = builder.bool_type();
    let is_empty = builder.icmp(BinaryOp::Eq, len_field, zero, bool_ty);
    builder.cond_br(is_empty, empty_case, non_empty_case);

    // Empty: return None
    builder.set_insert_point(empty_case);
    let undef_val = builder.undef(void_ty);
    let none_val = builder.create_union(0, undef_val, option_ty.clone());
    builder.ret(none_val);

    // Non-empty: return Some(last element)
    builder.set_insert_point(non_empty_case);

    let one = builder.const_u64(1);
    let new_len = builder.sub(len_field, one, usize_ty.clone());

    let elem_ptr = builder.ptr_add(ptr_field, new_len, ptr_i32_ty.clone());
    let elem_val = builder.load(elem_ptr, i32_ty);

    let updated_vec = builder.create_struct(vec_i32_ty, vec![ptr_field, new_len, cap_field]);
    builder.store(updated_vec, vec_ptr);

    let some_val = builder.create_union(1, elem_val, option_ty);
    builder.ret(some_val);
}

/// Build: fn vec_i32_get(vec: *vec_i32, index: usize) -> Option<i32>
/// Returns Some(element) at index, or None if out of bounds
fn build_vec_i32_get(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty.clone());
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let vec_i32_ty = builder.struct_type(
        Some("vec_i32"),
        vec![ptr_i32_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_i32_ty.clone());

    // Create Option<i32> type
    let option_variants = vec![
        crate::hir::HirUnionVariant {
            name: builder.intern("None"),
            ty: void_ty.clone(),
            discriminant: 0,
        },
        crate::hir::HirUnionVariant {
            name: builder.intern("Some"),
            ty: i32_ty.clone(),
            discriminant: 1,
        },
    ];
    let option_ty = builder.union_type(Some("Option"), option_variants);

    let func_id = builder
        .begin_function("vec_i32_get")
        .param("vec", ptr_vec_ty)
        .param("index", usize_ty.clone())
        .returns(option_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    let bounds_check = builder.create_block("bounds_check");
    let out_of_bounds = builder.create_block("out_of_bounds");
    let in_bounds = builder.create_block("in_bounds");

    builder.set_insert_point(entry);
    let vec_ptr = builder.get_param(0);
    let index = builder.get_param(1);

    let vec_val = builder.load(vec_ptr, vec_i32_ty);
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_i32_ty.clone());
    let len_field = builder.extract_struct_field(vec_val, 1, usize_ty);

    builder.br(bounds_check);

    // Check if index >= len
    builder.set_insert_point(bounds_check);
    let bool_ty = builder.bool_type();
    let is_out_of_bounds = builder.icmp(BinaryOp::Ge, index, len_field, bool_ty);
    builder.cond_br(is_out_of_bounds, out_of_bounds, in_bounds);

    // Out of bounds: return None
    builder.set_insert_point(out_of_bounds);
    let undef_val = builder.undef(void_ty);
    let none_val = builder.create_union(0, undef_val, option_ty.clone());
    builder.ret(none_val);

    // In bounds: return Some(vec[index])
    builder.set_insert_point(in_bounds);
    let elem_ptr = builder.ptr_add(ptr_field, index, ptr_i32_ty);
    let elem_val = builder.load(elem_ptr, i32_ty);
    let some_val = builder.create_union(1, elem_val, option_ty);
    builder.ret(some_val);
}

/// Build: fn vec_i32_set(vec: *vec_i32, index: usize, value: i32) -> bool
/// Returns false if out of bounds
fn build_vec_i32_set(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty.clone());
    let usize_ty = builder.u64_type();
    let bool_ty = builder.bool_type();

    let vec_i32_ty = builder.struct_type(
        Some("vec_i32"),
        vec![ptr_i32_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_i32_ty.clone());

    let func_id = builder
        .begin_function("vec_i32_set")
        .param("vec", ptr_vec_ty)
        .param("index", usize_ty.clone())
        .param("value", i32_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    let bounds_check = builder.create_block("bounds_check");
    let out_of_bounds = builder.create_block("out_of_bounds");
    let in_bounds = builder.create_block("in_bounds");

    builder.set_insert_point(entry);
    let vec_ptr = builder.get_param(0);
    let index = builder.get_param(1);
    let value = builder.get_param(2);

    let vec_val = builder.load(vec_ptr, vec_i32_ty);
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_i32_ty.clone());
    let len_field = builder.extract_struct_field(vec_val, 1, usize_ty);

    builder.br(bounds_check);

    // Check if index >= len
    builder.set_insert_point(bounds_check);
    let is_out_of_bounds = builder.icmp(BinaryOp::Ge, index, len_field, bool_ty.clone());
    builder.cond_br(is_out_of_bounds, out_of_bounds, in_bounds);

    // Out of bounds: return false
    builder.set_insert_point(out_of_bounds);
    let false_val = builder.const_bool(false);
    builder.ret(false_val);

    // In bounds: set vec[index] = value, return true
    builder.set_insert_point(in_bounds);
    let elem_ptr = builder.ptr_add(ptr_field, index, ptr_i32_ty);
    builder.store(value, elem_ptr);
    let true_val = builder.const_bool(true);
    builder.ret(true_val);
}

/// Build: fn vec_i32_len(vec: *vec_i32) -> usize
fn build_vec_i32_len(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty);
    let usize_ty = builder.u64_type();

    let vec_i32_ty = builder.struct_type(
        Some("vec_i32"),
        vec![ptr_i32_ty, usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_i32_ty.clone());

    let func_id = builder
        .begin_function("vec_i32_len")
        .param("vec", ptr_vec_ty)
        .returns(usize_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_ptr = builder.get_param(0);
    let vec_val = builder.load(vec_ptr, vec_i32_ty);
    let len_field = builder.extract_struct_field(vec_val, 1, usize_ty);

    builder.ret(len_field);
}

/// Build: fn vec_i32_capacity(vec: *vec_i32) -> usize
fn build_vec_i32_capacity(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty);
    let usize_ty = builder.u64_type();

    let vec_i32_ty = builder.struct_type(
        Some("vec_i32"),
        vec![ptr_i32_ty, usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_i32_ty.clone());

    let func_id = builder
        .begin_function("vec_i32_capacity")
        .param("vec", ptr_vec_ty)
        .returns(usize_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_ptr = builder.get_param(0);
    let vec_val = builder.load(vec_ptr, vec_i32_ty);
    let cap_field = builder.extract_struct_field(vec_val, 2, usize_ty);

    builder.ret(cap_field);
}

/// Build: fn vec_i32_clear(vec: *vec_i32) -> void
fn build_vec_i32_clear(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty);
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let vec_i32_ty = builder.struct_type(
        Some("vec_i32"),
        vec![ptr_i32_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_i32_ty.clone());

    let func_id = builder
        .begin_function("vec_i32_clear")
        .param("vec", ptr_vec_ty)
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_ptr = builder.get_param(0);
    let vec_val = builder.load(vec_ptr, vec_i32_ty.clone());
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_i32_ty);
    let cap_field = builder.extract_struct_field(vec_val, 2, usize_ty);

    // Set len to 0
    let zero = builder.const_u64(0);
    let cleared_vec = builder.create_struct(vec_i32_ty, vec![ptr_field, zero, cap_field]);
    builder.store(cleared_vec, vec_ptr);

    let unit = builder.unit_value();
    builder.ret(unit);
}

/// Build: fn vec_i32_free(vec: vec_i32) -> void
fn build_vec_i32_free(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty);
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let vec_i32_ty = builder.struct_type(
        Some("vec_i32"),
        vec![ptr_i32_ty.clone(), usize_ty.clone(), usize_ty],
    );

    let func_id = builder
        .begin_function("vec_i32_free")
        .param("vec", vec_i32_ty.clone())
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_val = builder.get_param(0);
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_i32_ty);

    // Cast *i32 to *u8 for free
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let ptr_u8 = builder.bitcast(ptr_field, ptr_u8_ty);

    // Call free
    let free_name = builder.intern("free");
    let free_id = builder.get_function_by_name(free_name);
    let free_ref = builder.function_ref(free_id);
    let _result = builder.call(free_ref, vec![ptr_u8]);

    let unit = builder.unit_value();
    builder.ret(unit);
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::AstArena;

    #[test]
    fn test_build_vec_i32() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_vec_i32", &mut arena);

        // Declare dependencies
        crate::stdlib::memory::build_memory_functions(&mut builder);

        // Build vec_i32
        build_vec_i32_type(&mut builder);

        let module = builder.finish();

        // Verify all functions exist
        let expected_functions = vec![
            "vec_i32_new",
            "vec_i32_with_capacity",
            "vec_i32_push",
            "vec_i32_pop",
            "vec_i32_get",
            "vec_i32_set",
            "vec_i32_len",
            "vec_i32_capacity",
            "vec_i32_clear",
            "vec_i32_free",
        ];

        for func_name in expected_functions {
            let found = module
                .functions
                .values()
                .any(|f| arena.resolve_string(f.name) == Some(func_name));
            assert!(found, "Function {} not found", func_name);

            // Verify it's NOT generic (no type params)
            let func = module
                .functions
                .values()
                .find(|f| arena.resolve_string(f.name) == Some(func_name))
                .unwrap();
            assert!(
                func.signature.type_params.is_empty(),
                "Concrete vec_i32 function {} should not be generic",
                func_name
            );
        }
    }
}
