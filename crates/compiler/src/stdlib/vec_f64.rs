use crate::hir::{BinaryOp, CallingConvention};
/// vec_f64: Performance-optimized concrete vector for f64
/// Use this for numerical computing, physics simulations, graphics, etc.
use crate::hir_builder::HirBuilder;

pub fn build_vec_f64_type(builder: &mut HirBuilder) {
    // Declare C realloc for dynamic growth
    declare_c_realloc(builder);

    // Build all vec_f64 functions
    build_vec_f64_new(builder);
    build_vec_f64_push(builder);
    build_vec_f64_pop(builder);
    build_vec_f64_get(builder);
    build_vec_f64_set(builder);
    build_vec_f64_len(builder);
    build_vec_f64_capacity(builder);
    build_vec_f64_clear(builder);
    build_vec_f64_free(builder);
}

/// Declare C realloc function for resizing
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

/// Build: fn vec_f64_new() -> vec_f64
fn build_vec_f64_new(builder: &mut HirBuilder) {
    let f64_ty = builder.f64_type();
    let ptr_f64_ty = builder.ptr_type(f64_ty);
    let usize_ty = builder.u64_type();

    let vec_f64_ty = builder.struct_type(
        Some("vec_f64"),
        vec![ptr_f64_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );

    let func_id = builder
        .begin_function("vec_f64_new")
        .returns(vec_f64_ty.clone())
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let initial_cap = builder.const_u64(4);
    let f64_size = builder.const_u64(8);
    let alloc_size = builder.mul(initial_cap, f64_size, usize_ty.clone());
    let malloc_name = builder.intern("malloc");
    let malloc_id = builder.get_function_by_name(malloc_name);
    let malloc_ref = builder.function_ref(malloc_id);
    let ptr_u8 = builder.call(malloc_ref, vec![alloc_size]).unwrap();
    let ptr_f64 = builder.bitcast(ptr_u8, ptr_f64_ty.clone());
    let zero = builder.const_u64(0);
    let vec_value = builder.create_struct(vec_f64_ty.clone(), vec![ptr_f64, zero, initial_cap]);
    builder.ret(vec_value);
}

/// Build: fn vec_f64_push(vec: *vec_f64, value: f64)
fn build_vec_f64_push(builder: &mut HirBuilder) {
    let f64_ty = builder.f64_type();
    let ptr_f64_ty = builder.ptr_type(f64_ty.clone());
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let vec_f64_ty = builder.struct_type(
        Some("vec_f64"),
        vec![ptr_f64_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_f64_ty.clone());

    let func_id = builder
        .begin_function("vec_f64_push")
        .param("vec", ptr_vec_ty)
        .param("value", f64_ty.clone())
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

    let vec_val = builder.load(vec_ptr, vec_f64_ty.clone());
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_f64_ty.clone());
    let len_field = builder.extract_struct_field(vec_val, 1, usize_ty.clone());
    let cap_field = builder.extract_struct_field(vec_val, 2, usize_ty.clone());

    builder.br(check_capacity);

    // Check if len == cap
    builder.set_insert_point(check_capacity);
    let bool_ty = builder.bool_type();
    let is_full = builder.icmp(BinaryOp::Eq, len_field, cap_field, bool_ty);
    builder.cond_br(is_full, need_grow, no_grow);

    // Grow: new_cap = cap * 2, realloc
    builder.set_insert_point(need_grow);
    let two = builder.const_u64(2);
    let new_cap = builder.mul(cap_field, two, usize_ty.clone());
    let f64_size = builder.const_u64(8);
    let new_size = builder.mul(new_cap, f64_size, usize_ty.clone());

    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let old_ptr_u8 = builder.bitcast(ptr_field, ptr_u8_ty.clone());

    let realloc_name = builder.intern("realloc");
    let realloc_id = builder.get_function_by_name(realloc_name);
    let realloc_ref = builder.function_ref(realloc_id);
    let new_ptr_u8 = builder
        .call(realloc_ref, vec![old_ptr_u8, new_size])
        .unwrap();
    let new_ptr_f64 = builder.bitcast(new_ptr_u8, ptr_f64_ty.clone());

    let grown_vec =
        builder.create_struct(vec_f64_ty.clone(), vec![new_ptr_f64, len_field, new_cap]);
    builder.store(grown_vec, vec_ptr);
    builder.br(insert_element);

    // No grow needed
    builder.set_insert_point(no_grow);
    builder.br(insert_element);

    // Insert element at vec[len]
    builder.set_insert_point(insert_element);

    let vec_val2 = builder.load(vec_ptr, vec_f64_ty.clone());
    let ptr_field2 = builder.extract_struct_field(vec_val2, 0, ptr_f64_ty.clone());
    let len_field2 = builder.extract_struct_field(vec_val2, 1, usize_ty.clone());
    let cap_field2 = builder.extract_struct_field(vec_val2, 2, usize_ty.clone());

    let elem_ptr = builder.ptr_add(ptr_field2, len_field2, ptr_f64_ty.clone());
    builder.store(value, elem_ptr);

    let one = builder.const_u64(1);
    let new_len = builder.add(len_field2, one, usize_ty.clone());

    let final_vec = builder.create_struct(vec_f64_ty, vec![ptr_field2, new_len, cap_field2]);
    builder.store(final_vec, vec_ptr);

    let unit = builder.unit_value();
    builder.ret(unit);
}

/// Build: fn vec_f64_pop(vec: *vec_f64) -> Option<f64>
/// Returns Some(last element), or None if empty
fn build_vec_f64_pop(builder: &mut HirBuilder) {
    let f64_ty = builder.f64_type();
    let ptr_f64_ty = builder.ptr_type(f64_ty.clone());
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let vec_f64_ty = builder.struct_type(
        Some("vec_f64"),
        vec![ptr_f64_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_f64_ty.clone());

    // Create Option<f64> type
    let option_variants = vec![
        crate::hir::HirUnionVariant {
            name: builder.intern("None"),
            ty: void_ty.clone(),
            discriminant: 0,
        },
        crate::hir::HirUnionVariant {
            name: builder.intern("Some"),
            ty: f64_ty.clone(),
            discriminant: 1,
        },
    ];
    let option_ty = builder.union_type(Some("Option"), option_variants);

    let func_id = builder
        .begin_function("vec_f64_pop")
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

    let vec_val = builder.load(vec_ptr, vec_f64_ty.clone());
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_f64_ty.clone());
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

    let elem_ptr = builder.ptr_add(ptr_field, new_len, ptr_f64_ty.clone());
    let elem_val = builder.load(elem_ptr, f64_ty);

    let updated_vec = builder.create_struct(vec_f64_ty, vec![ptr_field, new_len, cap_field]);
    builder.store(updated_vec, vec_ptr);

    let some_val = builder.create_union(1, elem_val, option_ty);
    builder.ret(some_val);
}

/// Build: fn vec_f64_get(vec: *vec_f64, index: usize) -> Option<f64>
/// Returns Some(element) at index, or None if out of bounds
fn build_vec_f64_get(builder: &mut HirBuilder) {
    let f64_ty = builder.f64_type();
    let ptr_f64_ty = builder.ptr_type(f64_ty.clone());
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let vec_f64_ty = builder.struct_type(
        Some("vec_f64"),
        vec![ptr_f64_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_f64_ty.clone());

    // Create Option<f64> type
    let option_variants = vec![
        crate::hir::HirUnionVariant {
            name: builder.intern("None"),
            ty: void_ty.clone(),
            discriminant: 0,
        },
        crate::hir::HirUnionVariant {
            name: builder.intern("Some"),
            ty: f64_ty.clone(),
            discriminant: 1,
        },
    ];
    let option_ty = builder.union_type(Some("Option"), option_variants);

    let func_id = builder
        .begin_function("vec_f64_get")
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

    let vec_val = builder.load(vec_ptr, vec_f64_ty);
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_f64_ty.clone());
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
    let elem_ptr = builder.ptr_add(ptr_field, index, ptr_f64_ty);
    let elem_val = builder.load(elem_ptr, f64_ty);
    let some_val = builder.create_union(1, elem_val, option_ty);
    builder.ret(some_val);
}

/// Build: fn vec_f64_set(vec: *vec_f64, index: usize, value: f64) -> bool
/// Returns false if out of bounds
fn build_vec_f64_set(builder: &mut HirBuilder) {
    let f64_ty = builder.f64_type();
    let ptr_f64_ty = builder.ptr_type(f64_ty.clone());
    let usize_ty = builder.u64_type();
    let bool_ty = builder.bool_type();

    let vec_f64_ty = builder.struct_type(
        Some("vec_f64"),
        vec![ptr_f64_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_f64_ty.clone());

    let func_id = builder
        .begin_function("vec_f64_set")
        .param("vec", ptr_vec_ty)
        .param("index", usize_ty.clone())
        .param("value", f64_ty)
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

    let vec_val = builder.load(vec_ptr, vec_f64_ty);
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_f64_ty.clone());
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
    let elem_ptr = builder.ptr_add(ptr_field, index, ptr_f64_ty);
    builder.store(value, elem_ptr);
    let true_val = builder.const_bool(true);
    builder.ret(true_val);
}

/// Build: fn vec_f64_len(vec: *vec_f64) -> usize
fn build_vec_f64_len(builder: &mut HirBuilder) {
    let f64_ty = builder.f64_type();
    let ptr_f64_ty = builder.ptr_type(f64_ty);
    let usize_ty = builder.u64_type();

    let vec_f64_ty = builder.struct_type(
        Some("vec_f64"),
        vec![ptr_f64_ty, usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_f64_ty.clone());

    let func_id = builder
        .begin_function("vec_f64_len")
        .param("vec", ptr_vec_ty)
        .returns(usize_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_ptr = builder.get_param(0);
    let vec_val = builder.load(vec_ptr, vec_f64_ty);
    let len_field = builder.extract_struct_field(vec_val, 1, usize_ty);
    builder.ret(len_field);
}

/// Build: fn vec_f64_capacity(vec: *vec_f64) -> usize
fn build_vec_f64_capacity(builder: &mut HirBuilder) {
    let f64_ty = builder.f64_type();
    let ptr_f64_ty = builder.ptr_type(f64_ty);
    let usize_ty = builder.u64_type();

    let vec_f64_ty = builder.struct_type(
        Some("vec_f64"),
        vec![ptr_f64_ty, usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_f64_ty.clone());

    let func_id = builder
        .begin_function("vec_f64_capacity")
        .param("vec", ptr_vec_ty)
        .returns(usize_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_ptr = builder.get_param(0);
    let vec_val = builder.load(vec_ptr, vec_f64_ty);
    let cap_field = builder.extract_struct_field(vec_val, 2, usize_ty);
    builder.ret(cap_field);
}

/// Build: fn vec_f64_clear(vec: *vec_f64)
fn build_vec_f64_clear(builder: &mut HirBuilder) {
    let f64_ty = builder.f64_type();
    let ptr_f64_ty = builder.ptr_type(f64_ty);
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let vec_f64_ty = builder.struct_type(
        Some("vec_f64"),
        vec![ptr_f64_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_f64_ty.clone());

    let func_id = builder
        .begin_function("vec_f64_clear")
        .param("vec", ptr_vec_ty)
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_ptr = builder.get_param(0);
    let vec_val = builder.load(vec_ptr, vec_f64_ty.clone());
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_f64_ty.clone());
    let cap_field = builder.extract_struct_field(vec_val, 2, usize_ty.clone());

    let zero = builder.const_u64(0);
    let cleared_vec = builder.create_struct(vec_f64_ty, vec![ptr_field, zero, cap_field]);
    builder.store(cleared_vec, vec_ptr);

    let unit = builder.unit_value();
    builder.ret(unit);
}

/// Build: fn vec_f64_free(vec: vec_f64)
fn build_vec_f64_free(builder: &mut HirBuilder) {
    let f64_ty = builder.f64_type();
    let ptr_f64_ty = builder.ptr_type(f64_ty);
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let vec_f64_ty = builder.struct_type(
        Some("vec_f64"),
        vec![ptr_f64_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );

    let func_id = builder
        .begin_function("vec_f64_free")
        .param("vec", vec_f64_ty.clone())
        .returns(void_ty)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let vec_val = builder.get_param(0);
    let ptr_field = builder.extract_struct_field(vec_val, 0, ptr_f64_ty);
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let ptr_u8 = builder.bitcast(ptr_field, ptr_u8_ty);
    let free_name = builder.intern("free");
    let free_id = builder.get_function_by_name(free_name);
    let free_ref = builder.function_ref(free_id);
    let _result = builder.call(free_ref, vec![ptr_u8]);
    let unit = builder.unit_value();
    builder.ret(unit);
}
