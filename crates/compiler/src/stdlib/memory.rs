use crate::hir::{BinaryOp, CallingConvention, HirId, HirType, HirUnionVariant};
/// Memory management functions using HIR Builder
///
/// Provides safe wrappers around C memory allocation functions:
/// - malloc/free wrappers
/// - Box<T> heap allocation (conceptually)
/// - Memory utilities
use crate::hir_builder::HirBuilder;

/// Builds memory management functions
pub fn build_memory_functions(builder: &mut HirBuilder) {
    // Declare extern C functions first
    declare_c_memory_functions(builder);

    // Build safe wrappers
    build_safe_allocate(builder);
    build_safe_deallocate(builder);
    build_alloc_i32(builder);
    build_dealloc_i32(builder);
}

/// Declares extern C memory functions (malloc, free)
fn declare_c_memory_functions(builder: &mut HirBuilder) {
    let usize_ty = builder.u64_type();
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
    let void_ty = builder.void_type();

    // extern "C" fn malloc(size: u64) -> *u8
    let _malloc = builder
        .begin_extern_function("malloc", CallingConvention::C)
        .param("size", usize_ty.clone())
        .returns(ptr_u8_ty.clone())
        .build();

    // extern "C" fn free(ptr: *u8) -> void
    let _free = builder
        .begin_extern_function("free", CallingConvention::C)
        .param("ptr", ptr_u8_ty)
        .returns(void_ty)
        .build();
}

/// Builds: fn allocate(size: u64) -> Option<*u8>
/// Safe wrapper around malloc that returns None on failure
fn build_safe_allocate(builder: &mut HirBuilder) {
    let usize_ty = builder.u64_type();
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
    let void_ty = builder.void_type();

    // Build Option<*u8> type
    let option_ptr_variants = vec![
        HirUnionVariant {
            name: builder.intern("None"),
            ty: void_ty.clone(),
            discriminant: 0,
        },
        HirUnionVariant {
            name: builder.intern("Some"),
            ty: ptr_u8_ty.clone(),
            discriminant: 1,
        },
    ];
    let option_ptr_ty = builder.union_type(Some("Option_ptr_u8"), option_ptr_variants);

    let func_id = builder
        .begin_function("allocate")
        .param("size", usize_ty)
        .returns(option_ptr_ty.clone())
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let call_malloc = builder.create_block("call_malloc");
    let check_null = builder.create_block("check_null");
    let success_block = builder.create_block("success");
    let failure_block = builder.create_block("failure");

    builder.set_insert_point(entry);

    // Get size parameter
    let size = builder.get_param(0);
    builder.br(call_malloc);

    // Call malloc
    builder.set_insert_point(call_malloc);

    // Get malloc function reference
    let malloc_name = builder.intern("malloc");
    let malloc_id = builder.get_function_by_name(malloc_name);
    let malloc_ref = builder.function_ref(malloc_id);

    // Call malloc and get the returned pointer
    let ptr = builder
        .call(malloc_ref, vec![size])
        .expect("malloc should return a value");

    builder.br(check_null);

    // Check if null
    builder.set_insert_point(check_null);
    let null = builder.null_ptr(u8_ty);
    let bool_ty = builder.bool_type();
    let is_null = builder.icmp(BinaryOp::Eq, ptr, null, bool_ty);
    builder.cond_br(is_null, failure_block, success_block);

    // Success: return Some(ptr)
    builder.set_insert_point(success_block);
    let some_value = builder.create_union(1, ptr, option_ptr_ty.clone());
    builder.ret(some_value);

    // Failure: return None
    builder.set_insert_point(failure_block);
    let unit = builder.unit_value();
    let none_value = builder.create_union(0, unit, option_ptr_ty);
    builder.ret(none_value);
}

/// Builds: fn deallocate(ptr: *u8) -> void
/// Safe wrapper around free
fn build_safe_deallocate(builder: &mut HirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("deallocate")
        .param("ptr", ptr_u8_ty)
        .returns(void_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Get ptr parameter
    let ptr = builder.get_param(0);

    // Call free
    let free_name = builder.intern("free");
    let free_id = builder.get_function_by_name(free_name);
    let free_ref = builder.function_ref(free_id);
    builder.call(free_ref, vec![ptr]);

    // Return void
    let unit = builder.unit_value();
    builder.ret(unit);
}

/// Builds: fn alloc_i32(value: i32) -> *i32
/// Allocates an i32 on the heap and initializes it
/// Returns the pointer to the allocated i32
fn build_alloc_i32(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let u64_ty = builder.u64_type();
    let u8_ty = builder.u8_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty.clone());

    let func_id = builder
        .begin_function("alloc_i32")
        .param("value", i32_ty.clone())
        .returns(ptr_i32_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Get value parameter
    let value = builder.get_param(0);

    // Size of i32 is 4 bytes - create as u64 constant
    let size_u64 = builder.const_u64(4);

    // Get malloc function reference
    let malloc_name = builder.intern("malloc");
    let malloc_id = builder.get_function_by_name(malloc_name);
    let malloc_ref = builder.function_ref(malloc_id);

    // Call malloc(4) to allocate 4 bytes
    let ptr_u8 = builder
        .call(malloc_ref, vec![size_u64])
        .expect("malloc should return a value");

    // Cast *u8 to *i32 using bitcast
    let ptr_i32 = builder.bitcast(ptr_u8, ptr_i32_ty.clone());

    // Store the value into the allocated memory
    builder.store(value, ptr_i32);

    // Return the pointer
    builder.ret(ptr_i32);
}

/// Builds: fn dealloc_i32(ptr: *i32) -> void
/// Deallocates an i32 from the heap
fn build_dealloc_i32(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let u8_ty = builder.u8_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty);
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("dealloc_i32")
        .param("ptr", ptr_i32_ty.clone())
        .returns(void_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Get ptr parameter
    let ptr = builder.get_param(0);

    // Cast *i32 to *u8 for free
    let ptr_u8 = builder.bitcast(ptr, ptr_u8_ty);

    // Call free
    let free_name = builder.intern("free");
    let free_id = builder.get_function_by_name(free_name);
    let free_ref = builder.function_ref(free_id);
    builder.call(free_ref, vec![ptr_u8]);

    // Return void
    let unit = builder.unit_value();
    builder.ret(unit);
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::AstArena;

    #[test]
    fn test_build_memory_functions() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_memory", &mut arena);

        build_memory_functions(&mut builder);

        let module = builder.finish();

        // Should have created:
        // - 2 extern functions (malloc, free)
        // - 4 wrapper functions (allocate, deallocate, alloc_i32, dealloc_i32)
        assert!(module.functions.len() >= 6);

        // Check function names
        let func_names: Vec<String> = module
            .functions
            .values()
            .map(|f| arena.resolve_string(f.name).unwrap().to_string())
            .collect();

        assert!(func_names.contains(&"malloc".to_string()));
        assert!(func_names.contains(&"free".to_string()));
        assert!(func_names.contains(&"allocate".to_string()));
        assert!(func_names.contains(&"deallocate".to_string()));
        assert!(func_names.contains(&"alloc_i32".to_string()));
        assert!(func_names.contains(&"dealloc_i32".to_string()));
    }

    #[test]
    fn test_malloc_is_extern() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_memory", &mut arena);

        declare_c_memory_functions(&mut builder);

        let module = builder.finish();

        // Find malloc function
        let malloc_func = module
            .functions
            .values()
            .find(|f| arena.resolve_string(f.name) == Some("malloc"))
            .expect("malloc function should exist");

        // Should be external
        assert!(malloc_func.is_external);

        // Should use C calling convention
        assert_eq!(malloc_func.calling_convention, CallingConvention::C);

        // Should have no blocks (external)
        assert!(malloc_func.blocks.is_empty());
    }

    #[test]
    fn test_allocate_structure() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_memory", &mut arena);

        declare_c_memory_functions(&mut builder);
        build_safe_allocate(&mut builder);

        let module = builder.finish();

        // Find the allocate function
        let allocate_func = module
            .functions
            .values()
            .find(|f| arena.resolve_string(f.name) == Some("allocate"))
            .expect("allocate function should exist");

        // Should have 6 blocks: entry, call_malloc, check_null, success, failure, plus extras from malloc call
        assert_eq!(allocate_func.blocks.len(), 6);

        // Should have 1 parameter (size)
        assert_eq!(allocate_func.signature.params.len(), 1);

        // Should return Option<*u8> (1 return type)
        assert_eq!(allocate_func.signature.returns.len(), 1);
    }
}
