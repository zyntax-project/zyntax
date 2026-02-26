use crate::hir::{BinaryOp, HirId, HirType};
/// Iterator-style functions for Vec<T>
///
/// Provides functional programming patterns for working with vectors.
///
/// For now, we implement simple functional-style iterator functions rather than
/// a full Iterator trait, since full trait dispatch would require more complex
/// type registry integration.
use crate::hir_builder::HirBuilder;

/// Build: fn vec_for_each<T>(vec: *Vec<T>, f: fn(*T) -> void)
/// Applies a function to each element in the vector
///
/// The callback receives a pointer to each element (avoiding copies for large types)
pub fn build_vec_for_each(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let ptr_t_ty = builder.ptr_type(t_param.clone());
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    // struct Vec<T> { ptr: *T, len: usize, cap: usize }
    let vec_t_ty = builder.struct_type(
        Some("Vec"),
        vec![ptr_t_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_ty = builder.ptr_type(vec_t_ty.clone());

    // fn(*T) -> void
    let callback_ty = builder.function_type(vec![ptr_t_ty.clone()], void_ty.clone());

    let func_id = builder
        .begin_generic_function("vec_for_each", vec!["T"])
        .param("vec", ptr_vec_ty.clone())
        .param("f", callback_ty)
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    let loop_header = builder.create_block("loop_header");
    let loop_body = builder.create_block("loop_body");
    let loop_exit = builder.create_block("loop_exit");

    builder.set_insert_point(entry);

    let vec_param = builder.get_param(0);
    let callback_param = builder.get_param(1);

    // Load vec.len (field index 1)
    let len_ptr = builder.get_element_ptr(vec_param, 1, usize_ty.clone());
    let len = builder.load(len_ptr, usize_ty.clone());

    // Initialize loop counter
    let zero = builder.const_u64(0);
    let counter_alloc = builder.alloca(usize_ty.clone());
    builder.store(zero, counter_alloc);

    builder.br(loop_header);

    // loop_header: check if counter < len
    builder.set_insert_point(loop_header);
    let counter = builder.load(counter_alloc, usize_ty.clone());
    let continue_loop = builder.icmp(BinaryOp::Lt, counter, len, builder.bool_type());
    builder.cond_br(continue_loop, loop_body, loop_exit);

    // loop_body: get element and call callback
    builder.set_insert_point(loop_body);

    // Load vec.ptr (field index 0)
    let ptr_field = builder.get_element_ptr(vec_param, 0, ptr_t_ty.clone());
    let data_ptr = builder.load(ptr_field, ptr_t_ty.clone());

    // Get element at index: data_ptr + counter
    let counter_val = builder.load(counter_alloc, usize_ty.clone());
    let elem_ptr = builder.ptr_add(data_ptr, counter_val, ptr_t_ty.clone());

    // Call callback with pointer to element
    builder.call(callback_param, vec![elem_ptr]);

    // Increment counter
    let one = builder.const_u64(1);
    let next_counter = builder.add(counter_val, one, usize_ty.clone());
    builder.store(counter_alloc, next_counter);

    builder.br(loop_header);

    // loop_exit: return
    builder.set_insert_point(loop_exit);
    let unit = builder.unit_value();
    builder.ret(unit);
}

/// Build: fn vec_map<T,U>(vec: *Vec<T>, f: fn(*T) -> U) -> Vec<U>
/// Maps a function over each element, creating a new vector
///
/// The callback receives a pointer to each element and returns a new value
pub fn build_vec_map(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let u_param = builder.type_param("U");
    let ptr_t_ty = builder.ptr_type(t_param.clone());
    let ptr_u_ty = builder.ptr_type(u_param.clone());
    let usize_ty = builder.u64_type();

    // struct Vec<T> { ptr: *T, len: usize, cap: usize }
    let vec_t_ty = builder.struct_type(
        Some("Vec"),
        vec![ptr_t_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let vec_u_ty = builder.struct_type(
        Some("Vec"),
        vec![ptr_u_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_t_ty = builder.ptr_type(vec_t_ty.clone());

    // fn(*T) -> U
    let callback_ty = builder.function_type(vec![ptr_t_ty.clone()], u_param.clone());

    let func_id = builder
        .begin_generic_function("vec_map", vec!["T", "U"])
        .param("vec", ptr_vec_t_ty.clone())
        .param("f", callback_ty)
        .returns(vec_u_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    let loop_header = builder.create_block("loop_header");
    let loop_body = builder.create_block("loop_body");
    let loop_exit = builder.create_block("loop_exit");

    builder.set_insert_point(entry);

    let vec_param = builder.get_param(0);
    let callback_param = builder.get_param(1);

    // Load input vec.len (field index 1)
    let len_ptr = builder.get_element_ptr(vec_param, 1, usize_ty.clone());
    let len = builder.load(len_ptr, usize_ty.clone());

    // Allocate result vector with same capacity as input
    // Call vec_new<U>() to get empty vector
    // For now, we'll create the struct directly
    // TODO: Call vec_with_capacity<U>(len) once available

    // Calculate size for U array: len * sizeof(U)
    let u_size = builder.size_of_type(u_param.clone());
    let total_size = builder.mul(len, u_size, usize_ty.clone());

    // Call malloc
    let malloc_name = builder.intern("malloc");
    let malloc_id = builder.get_function_by_name(malloc_name);
    let malloc_ref = builder.function_ref(malloc_id);
    let ptr_u8 = builder.call(malloc_ref, vec![total_size]).unwrap();

    // Cast *u8 to *U
    let result_ptr = builder.bitcast(ptr_u8, ptr_u_ty.clone());

    // Initialize loop counter
    let zero = builder.const_u64(0);
    let counter_alloc = builder.alloca(usize_ty.clone());
    builder.store(zero, counter_alloc);

    builder.br(loop_header);

    // loop_header: check if counter < len
    builder.set_insert_point(loop_header);
    let counter = builder.load(counter_alloc, usize_ty.clone());
    let continue_loop = builder.icmp(BinaryOp::Lt, counter, len, builder.bool_type());
    builder.cond_br(continue_loop, loop_body, loop_exit);

    // loop_body: map element
    builder.set_insert_point(loop_body);

    // Load input vec.ptr (field index 0)
    let ptr_field = builder.get_element_ptr(vec_param, 0, ptr_t_ty.clone());
    let input_data_ptr = builder.load(ptr_field, ptr_t_ty.clone());

    // Get input element at index
    let counter_val = builder.load(counter_alloc, usize_ty.clone());
    let input_elem_ptr = builder.ptr_add(input_data_ptr, counter_val, ptr_t_ty.clone());

    // Call callback to get mapped value
    let mapped_value = builder.call(callback_param, vec![input_elem_ptr]).unwrap();

    // Store mapped value in result array
    let result_elem_ptr = builder.ptr_add(result_ptr, counter_val, ptr_u_ty.clone());
    builder.store(mapped_value, result_elem_ptr);

    // Increment counter
    let one = builder.const_u64(1);
    let next_counter = builder.add(counter_val, one, usize_ty.clone());
    builder.store(counter_alloc, next_counter);

    builder.br(loop_header);

    // loop_exit: build result Vec<U>
    builder.set_insert_point(loop_exit);

    // Build Vec<U> { ptr: result_ptr, len: len, cap: len }
    let result_vec = builder.create_struct(vec_u_ty, vec![result_ptr, len, len]);

    builder.ret(result_vec);
}

/// Build: fn vec_filter<T>(vec: *Vec<T>, pred: fn(*T) -> bool) -> Vec<T>
/// Filters elements that satisfy the predicate
///
/// The predicate receives a pointer to each element and returns true to keep it
pub fn build_vec_filter(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let ptr_t_ty = builder.ptr_type(t_param.clone());
    let usize_ty = builder.u64_type();
    let bool_ty = builder.bool_type();

    // struct Vec<T> { ptr: *T, len: usize, cap: usize }
    let vec_t_ty = builder.struct_type(
        Some("Vec"),
        vec![ptr_t_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_t_ty = builder.ptr_type(vec_t_ty.clone());

    // fn(*T) -> bool
    let predicate_ty = builder.function_type(vec![ptr_t_ty.clone()], bool_ty.clone());

    let func_id = builder
        .begin_generic_function("vec_filter", vec!["T"])
        .param("vec", ptr_vec_t_ty.clone())
        .param("pred", predicate_ty)
        .returns(vec_t_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    let loop_header = builder.create_block("loop_header");
    let loop_body = builder.create_block("loop_body");
    let check_pred = builder.create_block("check_pred");
    let copy_elem = builder.create_block("copy_elem");
    let skip_elem = builder.create_block("skip_elem");
    let loop_exit = builder.create_block("loop_exit");

    builder.set_insert_point(entry);

    let vec_param = builder.get_param(0);
    let pred_param = builder.get_param(1);

    // Load input vec.len
    let len_ptr = builder.get_element_ptr(vec_param, 1, usize_ty.clone());
    let len = builder.load(len_ptr, usize_ty.clone());

    // Allocate result array (max size = input size)
    let t_size = builder.size_of_type(t_param.clone());
    let total_size = builder.mul(len, t_size, usize_ty.clone());

    let malloc_name = builder.intern("malloc");
    let malloc_id = builder.get_function_by_name(malloc_name);
    let malloc_ref = builder.function_ref(malloc_id);
    let ptr_u8 = builder.call(malloc_ref, vec![total_size]).unwrap();

    let result_ptr = builder.bitcast(ptr_u8, ptr_t_ty.clone());

    // Initialize counters
    let zero = builder.const_u64(0);
    let input_counter = builder.alloca(usize_ty.clone());
    let output_counter = builder.alloca(usize_ty.clone());
    builder.store(zero, input_counter);
    builder.store(zero, output_counter);

    builder.br(loop_header);

    // loop_header: check if input_counter < len
    builder.set_insert_point(loop_header);
    let in_count = builder.load(input_counter, usize_ty.clone());
    let continue_loop = builder.icmp(BinaryOp::Lt, in_count, len, bool_ty.clone());
    builder.cond_br(continue_loop, loop_body, loop_exit);

    // loop_body: load element
    builder.set_insert_point(loop_body);

    let ptr_field = builder.get_element_ptr(vec_param, 0, ptr_t_ty.clone());
    let input_data_ptr = builder.load(ptr_field, ptr_t_ty.clone());

    let in_count_val = builder.load(input_counter, usize_ty.clone());
    let input_elem_ptr = builder.ptr_add(input_data_ptr, in_count_val, ptr_t_ty.clone());

    builder.br(check_pred);

    // check_pred: call predicate
    builder.set_insert_point(check_pred);
    let pred_result = builder.call(pred_param, vec![input_elem_ptr]).unwrap();
    builder.cond_br(pred_result, copy_elem, skip_elem);

    // copy_elem: copy element to result
    builder.set_insert_point(copy_elem);

    let elem_value = builder.load(input_elem_ptr, t_param.clone());
    let out_count = builder.load(output_counter, usize_ty.clone());
    let result_elem_ptr = builder.ptr_add(result_ptr, out_count, ptr_t_ty.clone());
    builder.store(elem_value, result_elem_ptr);

    // Increment output counter
    let one = builder.const_u64(1);
    let next_out = builder.add(out_count, one, usize_ty.clone());
    builder.store(output_counter, next_out);

    builder.br(skip_elem);

    // skip_elem: increment input counter
    builder.set_insert_point(skip_elem);
    let in_val = builder.load(input_counter, usize_ty.clone());
    let one_copy = builder.const_u64(1);
    let next_in = builder.add(in_val, one_copy, usize_ty.clone());
    builder.store(input_counter, next_in);

    builder.br(loop_header);

    // loop_exit: build result Vec<T>
    builder.set_insert_point(loop_exit);

    let final_len = builder.load(output_counter, usize_ty.clone());
    let result_vec = builder.create_struct(
        vec_t_ty,
        vec![result_ptr, final_len, len], // cap = original len, len = filtered count
    );

    builder.ret(result_vec);
}

/// Build: fn vec_fold<T,Acc>(vec: *Vec<T>, init: Acc, f: fn(Acc, *T) -> Acc) -> Acc
/// Folds (reduces) a vector to a single value
///
/// The callback receives the accumulator and a pointer to each element
pub fn build_vec_fold(builder: &mut HirBuilder) {
    let t_param = builder.type_param("T");
    let acc_param = builder.type_param("Acc");
    let ptr_t_ty = builder.ptr_type(t_param.clone());
    let usize_ty = builder.u64_type();

    // struct Vec<T> { ptr: *T, len: usize, cap: usize }
    let vec_t_ty = builder.struct_type(
        Some("Vec"),
        vec![ptr_t_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_vec_t_ty = builder.ptr_type(vec_t_ty.clone());

    // fn(Acc, *T) -> Acc
    let callback_ty =
        builder.function_type(vec![acc_param.clone(), ptr_t_ty.clone()], acc_param.clone());

    let func_id = builder
        .begin_generic_function("vec_fold", vec!["T", "Acc"])
        .param("vec", ptr_vec_t_ty.clone())
        .param("init", acc_param.clone())
        .param("f", callback_ty)
        .returns(acc_param.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    let loop_header = builder.create_block("loop_header");
    let loop_body = builder.create_block("loop_body");
    let loop_exit = builder.create_block("loop_exit");

    builder.set_insert_point(entry);

    let vec_param = builder.get_param(0);
    let init_param = builder.get_param(1);
    let callback_param = builder.get_param(2);

    // Load vec.len
    let len_ptr = builder.get_element_ptr(vec_param, 1, usize_ty.clone());
    let len = builder.load(len_ptr, usize_ty.clone());

    // Initialize accumulator and counter
    let zero = builder.const_u64(0);
    let counter_alloc = builder.alloca(usize_ty.clone());
    let acc_alloc = builder.alloca(acc_param.clone());
    builder.store(zero, counter_alloc);
    builder.store(init_param, acc_alloc);

    builder.br(loop_header);

    // loop_header: check if counter < len
    builder.set_insert_point(loop_header);
    let counter = builder.load(counter_alloc, usize_ty.clone());
    let continue_loop = builder.icmp(BinaryOp::Lt, counter, len, builder.bool_type());
    builder.cond_br(continue_loop, loop_body, loop_exit);

    // loop_body: fold element
    builder.set_insert_point(loop_body);

    // Load vec.ptr
    let ptr_field = builder.get_element_ptr(vec_param, 0, ptr_t_ty.clone());
    let data_ptr = builder.load(ptr_field, ptr_t_ty.clone());

    // Get element at index
    let counter_val = builder.load(counter_alloc, usize_ty.clone());
    let elem_ptr = builder.ptr_add(data_ptr, counter_val, ptr_t_ty.clone());

    // Load accumulator
    let acc_val = builder.load(acc_alloc, acc_param.clone());

    // Call callback with (accumulator, element_ptr)
    let new_acc = builder
        .call(callback_param, vec![acc_val, elem_ptr])
        .unwrap();

    // Store new accumulator
    builder.store(new_acc, acc_alloc);

    // Increment counter
    let one = builder.const_u64(1);
    let next_counter = builder.add(counter_val, one, usize_ty.clone());
    builder.store(counter_alloc, next_counter);

    builder.br(loop_header);

    // loop_exit: return final accumulator
    builder.set_insert_point(loop_exit);
    let final_acc = builder.load(acc_alloc, acc_param);
    builder.ret(final_acc);
}

/// Builds all iterator-style functions
pub fn build_iterator(builder: &mut HirBuilder) {
    build_vec_for_each(builder);
    build_vec_map(builder);
    build_vec_filter(builder);
    build_vec_fold(builder);
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::AstArena;

    #[test]
    fn test_build_vec_for_each() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_iterator", &mut arena);

        // Declare malloc dependency
        crate::stdlib::memory::build_memory_functions(&mut builder);

        build_iterator(&mut builder);

        let module = builder.finish();

        // Should have created vec_for_each
        assert!(!module.functions.is_empty());

        // Check function names
        let func_names: Vec<String> = module
            .functions
            .values()
            .map(|f| arena.resolve_string(f.name).unwrap().to_string())
            .collect();

        assert!(func_names.contains(&"vec_for_each".to_string()));

        // Verify it's generic
        let for_each_func = module
            .functions
            .values()
            .find(|f| arena.resolve_string(f.name) == Some("vec_for_each"))
            .expect("vec_for_each should exist");

        assert!(
            !for_each_func.signature.type_params.is_empty(),
            "vec_for_each should be generic"
        );
    }

    #[test]
    fn test_vec_for_each_structure() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_iterator", &mut arena);

        build_vec_for_each(&mut builder);

        let module = builder.finish();

        // Find the vec_for_each function
        let func = module
            .functions
            .values()
            .find(|f| arena.resolve_string(f.name) == Some("vec_for_each"))
            .expect("vec_for_each function should exist");

        // Should have 4 blocks: entry, loop_header, loop_body, loop_exit
        // Note: May have 5 blocks due to PHI node creation, but should have at least 4
        assert!(
            func.blocks.len() >= 4,
            "Expected at least 4 blocks, got {}",
            func.blocks.len()
        );

        // Should have 2 parameters (vec and callback)
        assert_eq!(func.signature.params.len(), 2);

        // Should return void
        assert_eq!(func.signature.returns.len(), 1);

        // Should be generic (have 1 type param: T)
        assert_eq!(func.signature.type_params.len(), 1);
    }

    #[test]
    fn test_build_vec_map() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_iterator", &mut arena);

        // Declare malloc dependency
        crate::stdlib::memory::build_memory_functions(&mut builder);

        build_vec_map(&mut builder);

        let module = builder.finish();

        // Find the vec_map function
        let func = module
            .functions
            .values()
            .find(|f| arena.resolve_string(f.name) == Some("vec_map"))
            .expect("vec_map function should exist");

        // Should have 4 blocks: entry, loop_header, loop_body, loop_exit
        assert!(func.blocks.len() >= 4, "Expected at least 4 blocks");

        // Should have 2 parameters (vec and callback)
        assert_eq!(func.signature.params.len(), 2);

        // Should return Vec<U>
        assert_eq!(func.signature.returns.len(), 1);

        // Should be generic (have 2 type params: T, U)
        assert_eq!(func.signature.type_params.len(), 2);
    }

    #[test]
    fn test_build_vec_filter() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_iterator", &mut arena);

        // Declare malloc dependency
        crate::stdlib::memory::build_memory_functions(&mut builder);

        build_vec_filter(&mut builder);

        let module = builder.finish();

        // Find the vec_filter function
        let func = module
            .functions
            .values()
            .find(|f| arena.resolve_string(f.name) == Some("vec_filter"))
            .expect("vec_filter function should exist");

        // Should have 7 blocks: entry, loop_header, loop_body, check_pred, copy_elem, skip_elem, loop_exit
        assert!(func.blocks.len() >= 6, "Expected at least 6 blocks");

        // Should have 2 parameters (vec and predicate)
        assert_eq!(func.signature.params.len(), 2);

        // Should return Vec<T>
        assert_eq!(func.signature.returns.len(), 1);

        // Should be generic (have 1 type param: T)
        assert_eq!(func.signature.type_params.len(), 1);
    }

    #[test]
    fn test_build_vec_fold() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_iterator", &mut arena);

        build_vec_fold(&mut builder);

        let module = builder.finish();

        // Find the vec_fold function
        let func = module
            .functions
            .values()
            .find(|f| arena.resolve_string(f.name) == Some("vec_fold"))
            .expect("vec_fold function should exist");

        // Should have 4 blocks: entry, loop_header, loop_body, loop_exit
        assert!(func.blocks.len() >= 4, "Expected at least 4 blocks");

        // Should have 3 parameters (vec, init, callback)
        assert_eq!(func.signature.params.len(), 3);

        // Should return Acc
        assert_eq!(func.signature.returns.len(), 1);

        // Should be generic (have 2 type params: T, Acc)
        assert_eq!(func.signature.type_params.len(), 2);
    }

    #[test]
    fn test_all_iterator_functions() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_iterator", &mut arena);

        // Declare malloc dependency
        crate::stdlib::memory::build_memory_functions(&mut builder);

        build_iterator(&mut builder);

        let module = builder.finish();

        // Should have created all 4 iterator functions
        let func_names: Vec<String> = module
            .functions
            .values()
            .map(|f| arena.resolve_string(f.name).unwrap().to_string())
            .collect();

        assert!(func_names.contains(&"vec_for_each".to_string()));
        assert!(func_names.contains(&"vec_map".to_string()));
        assert!(func_names.contains(&"vec_filter".to_string()));
        assert!(func_names.contains(&"vec_fold".to_string()));

        // Count iterator functions (filter out memory functions)
        let iter_count = func_names
            .iter()
            .filter(|name| name.starts_with("vec_"))
            .count();
        assert_eq!(iter_count, 4, "Should have exactly 4 iterator functions");
    }
}
