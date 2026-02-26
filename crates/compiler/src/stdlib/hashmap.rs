use crate::hir::{BinaryOp, CallingConvention, HirId, HirType};
/// HashMap<K,V>: Generic hash table with open addressing
///
/// Structure:
/// ```
/// struct HashMap<K,V> {
///     buckets: *Bucket<K,V>,  // Array of buckets
///     len: usize,              // Number of key-value pairs
///     cap: usize,              // Capacity (number of buckets)
/// }
///
/// struct Bucket<K,V> {
///     state: u8,    // 0=empty, 1=occupied, 2=tombstone
///     key: K,       // Key (undefined if empty)
///     value: V,     // Value (undefined if empty)
/// }
/// ```
///
/// Hash function signature: fn(key: *K) -> u64
///
/// Collision resolution: Linear probing
/// Load factor: Resize when len > cap * 0.75
/// Growth strategy: Double capacity (16 → 32 → 64)
/// Initial capacity: 16 buckets
use crate::hir_builder::HirBuilder;

/// Declare memset extern function
fn declare_c_memset(builder: &mut HirBuilder) {
    let memset_name = builder.intern("memset");

    // Check if already declared
    if builder.has_function(memset_name) {
        return;
    }

    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let i32_ty = builder.i32_type();
    let usize_ty = builder.u64_type();

    // extern "C" fn memset(ptr: *u8, value: i32, size: usize) -> *u8
    let _memset = builder
        .begin_extern_function("memset", CallingConvention::C)
        .param("ptr", ptr_u8_ty.clone())
        .param("value", i32_ty)
        .param("size", usize_ty)
        .returns(ptr_u8_ty)
        .build();
}

/// Declare realloc extern function for resizing
fn declare_c_realloc(builder: &mut HirBuilder) {
    let realloc_name = builder.intern("realloc");

    // Check if already declared
    if builder.has_function(realloc_name) {
        return;
    }

    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let usize_ty = builder.u64_type();

    // extern "C" fn realloc(ptr: *u8, new_size: usize) -> *u8
    let _realloc = builder
        .begin_extern_function("realloc", CallingConvention::C)
        .param("ptr", ptr_u8_ty.clone())
        .param("new_size", usize_ty)
        .returns(ptr_u8_ty)
        .build();
}

/// Build HashMap<K,V> type and all methods
pub fn build_hashmap_type(builder: &mut HirBuilder) {
    // Declare C functions first
    declare_c_memset(builder);
    declare_c_realloc(builder);

    // Build HashMap<K,V> methods
    build_hashmap_new(builder);
    build_hashmap_with_capacity(builder);
    build_hashmap_resize(builder); // Resize helper
    build_hashmap_insert(builder);
    build_hashmap_get(builder);
    build_hashmap_remove(builder);
    build_hashmap_contains_key(builder);
    build_hashmap_len(builder);
    build_hashmap_capacity(builder);
    build_hashmap_clear(builder);
    build_hashmap_free(builder);
}

/// Build hash functions for primitive types
pub fn build_hash_functions(builder: &mut HirBuilder) {
    build_hash_i32(builder);
    build_hash_i64(builder);
    build_hash_u32(builder);
    build_hash_u64(builder);
    build_hash_bool(builder);
    build_hash_u8(builder);
}

/// Build equality functions for primitive types
pub fn build_equality_functions(builder: &mut HirBuilder) {
    build_eq_i32(builder);
    build_eq_i64(builder);
    build_eq_u32(builder);
    build_eq_u64(builder);
    build_eq_bool(builder);
    build_eq_u8(builder);
}

/// Entry point for building all HashMap functionality
pub fn build_hashmap(builder: &mut HirBuilder) {
    build_hash_functions(builder);
    build_equality_functions(builder);
    build_hashmap_type(builder);
}

// ============================================================================
// Hash Functions for Primitive Types
// ============================================================================

/// Build: fn hash_i32(key: *i32) -> u64
/// Simple hash function for i32 using FNV-1a variant
fn build_hash_i32(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty.clone());
    let u64_ty = builder.u64_type();

    let func_id = builder
        .begin_function("hash_i32")
        .param("key", ptr_i32_ty.clone())
        .returns(u64_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Load the key value
    let key_val = builder.load(builder.get_param(0), i32_ty.clone());

    // Cast i32 to u64 for hashing
    let key_u64 = builder.zext(key_val, u64_ty.clone());

    // FNV-1a hash for 32-bit integer
    // hash = FNV_OFFSET_BASIS
    // hash = hash ^ key
    // hash = hash * FNV_PRIME
    let fnv_offset = builder.const_u64(0xcbf29ce484222325u64);
    let fnv_prime = builder.const_u64(0x100000001b3u64);

    let hash1 = builder.xor(fnv_offset, key_u64, u64_ty.clone());
    let hash2 = builder.mul(hash1, fnv_prime, u64_ty.clone());

    builder.ret(hash2);
}

/// Build: fn hash_i64(key: *i64) -> u64
fn build_hash_i64(builder: &mut HirBuilder) {
    let i64_ty = builder.i64_type();
    let ptr_i64_ty = builder.ptr_type(i64_ty.clone());
    let u64_ty = builder.u64_type();

    let func_id = builder
        .begin_function("hash_i64")
        .param("key", ptr_i64_ty.clone())
        .returns(u64_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Load the key value
    let key_val = builder.load(builder.get_param(0), i64_ty.clone());

    // Bitcast i64 to u64 (reinterpret bits)
    // Note: This should work since i64 and u64 have the same size
    let key_u64 = key_val; // For now, treat as same value (both are 64-bit)

    // FNV-1a hash
    let fnv_offset = builder.const_u64(0xcbf29ce484222325u64);
    let fnv_prime = builder.const_u64(0x100000001b3u64);

    let hash1 = builder.xor(fnv_offset, key_u64, u64_ty.clone());
    let hash2 = builder.mul(hash1, fnv_prime, u64_ty.clone());

    builder.ret(hash2);
}

/// Build: fn hash_u32(key: *u32) -> u64
fn build_hash_u32(builder: &mut HirBuilder) {
    let u32_ty = builder.u32_type();
    let ptr_u32_ty = builder.ptr_type(u32_ty.clone());
    let u64_ty = builder.u64_type();

    let func_id = builder
        .begin_function("hash_u32")
        .param("key", ptr_u32_ty.clone())
        .returns(u64_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Load the key value
    let key_val = builder.load(builder.get_param(0), u32_ty.clone());

    // Zero-extend u32 to u64
    let key_u64 = builder.zext(key_val, u64_ty.clone());

    // FNV-1a hash
    let fnv_offset = builder.const_u64(0xcbf29ce484222325u64);
    let fnv_prime = builder.const_u64(0x100000001b3u64);

    let hash1 = builder.xor(fnv_offset, key_u64, u64_ty.clone());
    let hash2 = builder.mul(hash1, fnv_prime, u64_ty.clone());

    builder.ret(hash2);
}

/// Build: fn hash_u64(key: *u64) -> u64
fn build_hash_u64(builder: &mut HirBuilder) {
    let u64_ty = builder.u64_type();
    let ptr_u64_ty = builder.ptr_type(u64_ty.clone());

    let func_id = builder
        .begin_function("hash_u64")
        .param("key", ptr_u64_ty.clone())
        .returns(u64_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Load the key value
    let key_val = builder.load(builder.get_param(0), u64_ty.clone());

    // FNV-1a hash
    let fnv_offset = builder.const_u64(0xcbf29ce484222325u64);
    let fnv_prime = builder.const_u64(0x100000001b3u64);

    let hash1 = builder.xor(fnv_offset, key_val, u64_ty.clone());
    let hash2 = builder.mul(hash1, fnv_prime, u64_ty.clone());

    builder.ret(hash2);
}

/// Build: fn hash_bool(key: *bool) -> u64
fn build_hash_bool(builder: &mut HirBuilder) {
    let bool_ty = builder.bool_type();
    let ptr_bool_ty = builder.ptr_type(bool_ty.clone());
    let u64_ty = builder.u64_type();

    let func_id = builder
        .begin_function("hash_bool")
        .param("key", ptr_bool_ty.clone())
        .returns(u64_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Load the key value
    let key_val = builder.load(builder.get_param(0), bool_ty.clone());

    // Convert bool to u64 (false=0, true=1)
    let key_u64 = builder.zext(key_val, u64_ty.clone());

    // Simple hash: just XOR with a constant
    let hash_const = builder.const_u64(0x517cc1b727220a95u64);
    let hash = builder.xor(hash_const, key_u64, u64_ty.clone());

    builder.ret(hash);
}

/// Build: fn hash_u8(key: *u8) -> u64
fn build_hash_u8(builder: &mut HirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
    let u64_ty = builder.u64_type();

    let func_id = builder
        .begin_function("hash_u8")
        .param("key", ptr_u8_ty.clone())
        .returns(u64_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Load the key value
    let key_val = builder.load(builder.get_param(0), u8_ty.clone());

    // Zero-extend u8 to u64
    let key_u64 = builder.zext(key_val, u64_ty.clone());

    // FNV-1a hash
    let fnv_offset = builder.const_u64(0xcbf29ce484222325u64);
    let fnv_prime = builder.const_u64(0x100000001b3u64);

    let hash1 = builder.xor(fnv_offset, key_u64, u64_ty.clone());
    let hash2 = builder.mul(hash1, fnv_prime, u64_ty.clone());

    builder.ret(hash2);
}

// ============================================================================
// Equality Functions for Primitive Types
// ============================================================================

/// Build: fn eq_i32(a: *i32, b: *i32) -> bool
/// Compare two i32 values for equality
fn build_eq_i32(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_i32_ty = builder.ptr_type(i32_ty.clone());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("eq_i32")
        .param("a", ptr_i32_ty.clone())
        .param("b", ptr_i32_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Load both values
    let a_val = builder.load(builder.get_param(0), i32_ty.clone());
    let b_val = builder.load(builder.get_param(1), i32_ty.clone());

    // Compare for equality
    let result = builder.icmp(BinaryOp::Eq, a_val, b_val, i32_ty);

    builder.ret(result);
}

/// Build: fn eq_i64(a: *i64, b: *i64) -> bool
fn build_eq_i64(builder: &mut HirBuilder) {
    let i64_ty = builder.i64_type();
    let ptr_i64_ty = builder.ptr_type(i64_ty.clone());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("eq_i64")
        .param("a", ptr_i64_ty.clone())
        .param("b", ptr_i64_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let a_val = builder.load(builder.get_param(0), i64_ty.clone());
    let b_val = builder.load(builder.get_param(1), i64_ty.clone());

    let result = builder.icmp(BinaryOp::Eq, a_val, b_val, i64_ty);

    builder.ret(result);
}

/// Build: fn eq_u32(a: *u32, b: *u32) -> bool
fn build_eq_u32(builder: &mut HirBuilder) {
    let u32_ty = builder.u32_type();
    let ptr_u32_ty = builder.ptr_type(u32_ty.clone());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("eq_u32")
        .param("a", ptr_u32_ty.clone())
        .param("b", ptr_u32_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let a_val = builder.load(builder.get_param(0), u32_ty.clone());
    let b_val = builder.load(builder.get_param(1), u32_ty.clone());

    let result = builder.icmp(BinaryOp::Eq, a_val, b_val, u32_ty);

    builder.ret(result);
}

/// Build: fn eq_u64(a: *u64, b: *u64) -> bool
fn build_eq_u64(builder: &mut HirBuilder) {
    let u64_ty = builder.u64_type();
    let ptr_u64_ty = builder.ptr_type(u64_ty.clone());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("eq_u64")
        .param("a", ptr_u64_ty.clone())
        .param("b", ptr_u64_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let a_val = builder.load(builder.get_param(0), u64_ty.clone());
    let b_val = builder.load(builder.get_param(1), u64_ty.clone());

    let result = builder.icmp(BinaryOp::Eq, a_val, b_val, u64_ty);

    builder.ret(result);
}

/// Build: fn eq_bool(a: *bool, b: *bool) -> bool
fn build_eq_bool(builder: &mut HirBuilder) {
    let bool_ty = builder.bool_type();
    let ptr_bool_ty = builder.ptr_type(bool_ty.clone());

    let func_id = builder
        .begin_function("eq_bool")
        .param("a", ptr_bool_ty.clone())
        .param("b", ptr_bool_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let a_val = builder.load(builder.get_param(0), bool_ty.clone());
    let b_val = builder.load(builder.get_param(1), bool_ty.clone());

    let result = builder.icmp(BinaryOp::Eq, a_val, b_val, bool_ty);

    builder.ret(result);
}

/// Build: fn eq_u8(a: *u8, b: *u8) -> bool
fn build_eq_u8(builder: &mut HirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("eq_u8")
        .param("a", ptr_u8_ty.clone())
        .param("b", ptr_u8_ty.clone())
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let a_val = builder.load(builder.get_param(0), u8_ty.clone());
    let b_val = builder.load(builder.get_param(1), u8_ty.clone());

    let result = builder.icmp(BinaryOp::Eq, a_val, b_val, u8_ty);

    builder.ret(result);
}

// ============================================================================
// HashMap Core Functions
// ============================================================================

/// Build: fn hashmap_new<K,V>() -> HashMap<K,V>
/// Creates empty hashmap with initial capacity of 16
fn build_hashmap_new(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let usize_ty = builder.u64_type();

    // struct Bucket<K,V> { state: u8, key: K, value: V }
    let bucket_kv_ty = builder.struct_type(
        Some("Bucket"),
        vec![u8_ty.clone(), k_param.clone(), v_param.clone()],
    );
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty.clone());

    // struct HashMap<K,V> { buckets: *Bucket<K,V>, len: usize, cap: usize }
    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );

    let func_id = builder
        .begin_generic_function("hashmap_new", vec!["K", "V"])
        .returns(hashmap_kv_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Allocate initial capacity of 16 buckets
    let initial_cap = builder.const_u64(16);

    // Get sizeof(Bucket<K,V>)
    let bucket_size = builder.size_of_type(bucket_kv_ty.clone());
    let alloc_size = builder.mul(initial_cap, bucket_size, usize_ty.clone());

    // Call malloc
    let malloc_name = builder.intern("malloc");
    let malloc_id = builder.get_function_by_name(malloc_name);
    let malloc_ref = builder.function_ref(malloc_id);
    let ptr_u8 = builder.call(malloc_ref, vec![alloc_size]).unwrap();

    // Cast *u8 to *Bucket<K,V>
    let buckets_ptr = builder.bitcast(ptr_u8, ptr_bucket_ty.clone());

    // Initialize all buckets to empty (state = 0)
    // We'll use memset for this
    let memset_name = builder.intern("memset");
    let memset_id = builder.get_function_by_name(memset_name);
    let memset_ref = builder.function_ref(memset_id);

    // Cast buckets_ptr back to *u8 for memset
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let buckets_u8 = builder.bitcast(buckets_ptr, ptr_u8_ty);

    // memset(buckets, 0, alloc_size)
    let zero_i32 = builder.const_i32(0);
    let _ = builder.call(memset_ref, vec![buckets_u8, zero_i32, alloc_size]);

    // Build struct: { buckets: buckets_ptr, len: 0, cap: 16 }
    let zero_len = builder.const_u64(0);
    let hashmap_value =
        builder.create_struct(hashmap_kv_ty, vec![buckets_ptr, zero_len, initial_cap]);

    builder.ret(hashmap_value);
}

/// Build: fn hashmap_with_capacity<K,V>(capacity: usize) -> HashMap<K,V>
/// Creates empty hashmap with specified capacity
fn build_hashmap_with_capacity(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let usize_ty = builder.u64_type();

    // struct Bucket<K,V> { state: u8, key: K, value: V }
    let bucket_kv_ty = builder.struct_type(
        Some("Bucket"),
        vec![u8_ty.clone(), k_param.clone(), v_param.clone()],
    );
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty.clone());

    // struct HashMap<K,V> { buckets: *Bucket<K,V>, len: usize, cap: usize }
    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );

    let func_id = builder
        .begin_generic_function("hashmap_with_capacity", vec!["K", "V"])
        .param("capacity", usize_ty.clone())
        .returns(hashmap_kv_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let capacity = builder.get_param(0);

    // Get sizeof(Bucket<K,V>)
    let bucket_size = builder.size_of_type(bucket_kv_ty.clone());
    let alloc_size = builder.mul(capacity, bucket_size, usize_ty.clone());

    // Call malloc
    let malloc_name = builder.intern("malloc");
    let malloc_id = builder.get_function_by_name(malloc_name);
    let malloc_ref = builder.function_ref(malloc_id);
    let ptr_u8 = builder.call(malloc_ref, vec![alloc_size]).unwrap();

    // Cast *u8 to *Bucket<K,V>
    let buckets_ptr = builder.bitcast(ptr_u8, ptr_bucket_ty.clone());

    // Initialize all buckets to empty using memset
    let memset_name = builder.intern("memset");
    let memset_id = builder.get_function_by_name(memset_name);
    let memset_ref = builder.function_ref(memset_id);

    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let buckets_u8 = builder.bitcast(buckets_ptr, ptr_u8_ty);
    let zero_i32 = builder.const_i32(0);
    let _ = builder.call(memset_ref, vec![buckets_u8, zero_i32, alloc_size]);

    // Build struct: { buckets: buckets_ptr, len: 0, cap: capacity }
    let zero_len = builder.const_u64(0);
    let hashmap_value = builder.create_struct(hashmap_kv_ty, vec![buckets_ptr, zero_len, capacity]);

    builder.ret(hashmap_value);
}

/// Build: fn hashmap_resize<K,V>(map: *HashMap<K,V>, hash_fn: fn(*K) -> u64)
/// Doubles the capacity and rehashes all entries
fn build_hashmap_resize(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();
    let ptr_k_ty = builder.ptr_type(k_param.clone());
    let u64_ty = builder.u64_type();

    // struct Bucket<K,V> { state: u8, key: K, value: V }
    let bucket_kv_ty = builder.struct_type(
        Some("Bucket"),
        vec![u8_ty.clone(), k_param.clone(), v_param.clone()],
    );
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty.clone());

    // struct HashMap<K,V> { buckets: *Bucket<K,V>, len: usize, cap: usize }
    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_hashmap_ty = builder.ptr_type(hashmap_kv_ty.clone());

    // Hash function type
    let hash_fn_ty = builder.function_type(vec![ptr_k_ty.clone()], u64_ty.clone());

    let func_id = builder
        .begin_generic_function("hashmap_resize", vec!["K", "V"])
        .param("map", ptr_hashmap_ty.clone())
        .param("hash_fn", hash_fn_ty)
        .returns(void_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    let allocate_new = builder.create_block("allocate_new");
    let rehash_loop = builder.create_block("rehash_loop");
    let loop_check = builder.create_block("loop_check");
    let check_occupied = builder.create_block("check_occupied");
    let rehash_entry = builder.create_block("rehash_entry");
    let find_new_slot = builder.create_block("find_new_slot");
    let new_slot_check = builder.create_block("new_slot_check");
    let found_new_slot = builder.create_block("found_new_slot");
    let next_new_probe = builder.create_block("next_new_probe");
    let next_old_bucket = builder.create_block("next_old_bucket");
    let finish_resize = builder.create_block("finish_resize");

    builder.set_insert_point(entry);

    let map_param = builder.get_param(0);
    let hash_fn_param = builder.get_param(1);

    // Get old cap and buckets
    let old_cap_ptr = builder.get_element_ptr(map_param, 2, usize_ty.clone());
    let old_cap = builder.load(old_cap_ptr, usize_ty.clone());

    let old_buckets_ptr_ptr = builder.get_element_ptr(map_param, 0, ptr_bucket_ty.clone());
    let old_buckets = builder.load(old_buckets_ptr_ptr, ptr_bucket_ty.clone());

    builder.br(allocate_new);

    // allocate_new: Allocate new bucket array with 2x capacity
    builder.set_insert_point(allocate_new);

    // new_cap = old_cap * 2
    let two = builder.const_u64(2);
    let new_cap = builder.mul(old_cap, two, usize_ty.clone());

    // Calculate new size
    let bucket_size = builder.size_of_type(bucket_kv_ty.clone());
    let new_size = builder.mul(new_cap, bucket_size, usize_ty.clone());

    // Allocate new array
    let malloc_name = builder.intern("malloc");
    let malloc_id = builder.get_function_by_name(malloc_name);
    let malloc_ref = builder.function_ref(malloc_id);
    let new_ptr_u8 = builder.call(malloc_ref, vec![new_size]).unwrap();
    let new_buckets = builder.bitcast(new_ptr_u8, ptr_bucket_ty.clone());

    // Initialize new buckets to zero
    let memset_name = builder.intern("memset");
    let memset_id = builder.get_function_by_name(memset_name);
    let memset_ref = builder.function_ref(memset_id);
    let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
    let new_buckets_u8 = builder.bitcast(new_buckets, ptr_u8_ty.clone());
    let zero_i32 = builder.const_i32(0);
    let _ = builder.call(memset_ref, vec![new_buckets_u8, zero_i32, new_size]);

    // Allocate loop counter
    let counter_alloc = builder.alloca(usize_ty.clone());
    let zero = builder.const_u64(0);
    builder.store(zero, counter_alloc);

    builder.br(rehash_loop);

    // rehash_loop: Iterate through old buckets
    builder.set_insert_point(rehash_loop);
    builder.br(loop_check);

    // loop_check: Check if done
    builder.set_insert_point(loop_check);

    let counter = builder.load(counter_alloc, usize_ty.clone());
    let done = builder.icmp(BinaryOp::Ge, counter, old_cap, usize_ty.clone());
    builder.cond_br(done, finish_resize, check_occupied);

    // check_occupied: Get current bucket and check if occupied
    builder.set_insert_point(check_occupied);

    let old_bucket_ptr = builder.ptr_add(old_buckets, counter, bucket_kv_ty.clone());
    let state_ptr = builder.get_element_ptr(old_bucket_ptr, 0, u8_ty.clone());
    let state = builder.load(state_ptr, u8_ty.clone());

    // If state == 1 (occupied), rehash it
    let one_u8 = builder.const_u8(1);
    let is_occupied = builder.icmp(BinaryOp::Eq, state, one_u8, u8_ty.clone());
    builder.cond_br(is_occupied, rehash_entry, next_old_bucket);

    // rehash_entry: Rehash this entry into new table
    builder.set_insert_point(rehash_entry);

    // Load key and value
    let key_ptr = builder.get_element_ptr(old_bucket_ptr, 1, k_param.clone());
    let key = builder.load(key_ptr, k_param.clone());
    let value_ptr = builder.get_element_ptr(old_bucket_ptr, 2, v_param.clone());
    let value = builder.load(value_ptr, v_param.clone());

    // Hash the key
    let key_alloc = builder.alloca(k_param.clone());
    builder.store(key, key_alloc);
    let hash = builder.call(hash_fn_param, vec![key_alloc]).unwrap();

    // new_index = hash % new_cap
    let new_index = builder.urem(hash, new_cap, usize_ty.clone());

    // Allocate probe counter for new table
    let new_probe_alloc = builder.alloca(usize_ty.clone());
    builder.store(zero, new_probe_alloc);

    builder.br(find_new_slot);

    // find_new_slot: Find empty slot in new table
    builder.set_insert_point(find_new_slot);
    builder.br(new_slot_check);

    // new_slot_check: Check if found empty slot
    builder.set_insert_point(new_slot_check);

    let new_probe = builder.load(new_probe_alloc, usize_ty.clone());
    let new_idx_offset = builder.add(new_index, new_probe, usize_ty.clone());
    let new_idx = builder.urem(new_idx_offset, new_cap, usize_ty.clone());

    let new_bucket_ptr = builder.ptr_add(new_buckets, new_idx, bucket_kv_ty.clone());
    let new_state_ptr = builder.get_element_ptr(new_bucket_ptr, 0, u8_ty.clone());
    let new_state = builder.load(new_state_ptr, u8_ty.clone());

    // If empty, insert here
    let zero_u8 = builder.const_u8(0);
    let is_empty = builder.icmp(BinaryOp::Eq, new_state, zero_u8, u8_ty.clone());
    builder.cond_br(is_empty, found_new_slot, next_new_probe);

    // found_new_slot: Insert into new table
    builder.set_insert_point(found_new_slot);

    builder.store(one_u8, new_state_ptr);
    let new_key_ptr = builder.get_element_ptr(new_bucket_ptr, 1, k_param.clone());
    builder.store(key, new_key_ptr);
    let new_value_ptr = builder.get_element_ptr(new_bucket_ptr, 2, v_param.clone());
    builder.store(value, new_value_ptr);

    builder.br(next_old_bucket);

    // next_new_probe: Continue probing in new table
    builder.set_insert_point(next_new_probe);

    let old_new_probe = builder.load(new_probe_alloc, usize_ty.clone());
    let one = builder.const_u64(1);
    let inc_new_probe = builder.add(old_new_probe, one, usize_ty.clone());
    builder.store(inc_new_probe, new_probe_alloc);

    builder.br(new_slot_check);

    // next_old_bucket: Move to next old bucket
    builder.set_insert_point(next_old_bucket);

    let old_counter = builder.load(counter_alloc, usize_ty.clone());
    let inc_counter = builder.add(old_counter, one, usize_ty.clone());
    builder.store(inc_counter, counter_alloc);

    builder.br(loop_check);

    // finish_resize: Update map and free old buckets
    builder.set_insert_point(finish_resize);

    // Update map->buckets
    builder.store(new_buckets, old_buckets_ptr_ptr);

    // Update map->cap
    builder.store(new_cap, old_cap_ptr);

    // Free old buckets
    let ptr_u8_old = builder.bitcast(old_buckets, ptr_u8_ty);
    let free_name = builder.intern("free");
    let free_id = builder.get_function_by_name(free_name);
    let free_ref = builder.function_ref(free_id);
    let _ = builder.call(free_ref, vec![ptr_u8_old]);

    let unit = builder.unit_value();
    builder.ret(unit);
}

/// Build: fn hashmap_insert<K,V>(map: *HashMap<K,V>, key: K, value: V, hash_fn: fn(*K) -> u64, eq_fn: fn(*K, *K) -> bool)
/// Inserts or updates a key-value pair
fn build_hashmap_insert(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();
    let bool_ty = builder.bool_type();
    let ptr_k_ty = builder.ptr_type(k_param.clone());
    let u64_ty = builder.u64_type();

    // struct Bucket<K,V> { state: u8, key: K, value: V }
    let bucket_kv_ty = builder.struct_type(
        Some("Bucket"),
        vec![u8_ty.clone(), k_param.clone(), v_param.clone()],
    );
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty.clone());

    // struct HashMap<K,V> { buckets: *Bucket<K,V>, len: usize, cap: usize }
    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_hashmap_ty = builder.ptr_type(hashmap_kv_ty.clone());

    // Hash function type: fn(*K) -> u64
    let hash_fn_ty = builder.function_type(vec![ptr_k_ty.clone()], u64_ty.clone());

    // Equality function type: fn(*K, *K) -> bool
    let eq_fn_ty = builder.function_type(vec![ptr_k_ty.clone(), ptr_k_ty.clone()], bool_ty.clone());

    let func_id = builder
        .begin_generic_function("hashmap_insert", vec!["K", "V"])
        .param("map", ptr_hashmap_ty.clone())
        .param("key", k_param.clone())
        .param("value", v_param.clone())
        .param("hash_fn", hash_fn_ty)
        .param("eq_fn", eq_fn_ty)
        .returns(void_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    let check_resize = builder.create_block("check_resize");
    let do_resize = builder.create_block("do_resize");
    let hash_key = builder.create_block("hash_key");
    let find_slot = builder.create_block("find_slot");
    let loop_check = builder.create_block("loop_check");
    let check_empty = builder.create_block("check_empty");
    let check_tombstone = builder.create_block("check_tombstone");
    let check_key_match = builder.create_block("check_key_match");
    let update_value = builder.create_block("update_value");
    let found_slot = builder.create_block("found_slot");
    let next_probe = builder.create_block("next_probe");
    let insert_done = builder.create_block("insert_done");

    builder.set_insert_point(entry);

    let map_param = builder.get_param(0);
    let key_param = builder.get_param(1);
    let value_param = builder.get_param(2);
    let hash_fn_param = builder.get_param(3);
    let eq_fn_param = builder.get_param(4);

    builder.br(check_resize);

    // check_resize: Check if we need to resize (load factor > 0.75)
    builder.set_insert_point(check_resize);

    // Get map->len and map->cap
    let len_ptr = builder.get_element_ptr(map_param, 1, usize_ty.clone());
    let len = builder.load(len_ptr, usize_ty.clone());

    let cap_ptr = builder.get_element_ptr(map_param, 2, usize_ty.clone());
    let cap = builder.load(cap_ptr, usize_ty.clone());

    // Check if len * 4 >= cap * 3 (equivalent to len/cap >= 0.75)
    let four = builder.const_u64(4);
    let three = builder.const_u64(3);
    let len_times_4 = builder.mul(len, four, usize_ty.clone());
    let cap_times_3 = builder.mul(cap, three, usize_ty.clone());

    let should_resize = builder.icmp(BinaryOp::Ge, len_times_4, cap_times_3, usize_ty.clone());
    builder.cond_br(should_resize, do_resize, hash_key);

    // do_resize: Call resize function
    builder.set_insert_point(do_resize);

    let resize_name = builder.intern("hashmap_resize");
    let resize_id = builder.get_function_by_name(resize_name);
    let resize_ref = builder.function_ref(resize_id);
    let _ = builder.call(resize_ref, vec![map_param, hash_fn_param]);

    builder.br(hash_key);

    // hash_key: Compute hash and initial index
    builder.set_insert_point(hash_key);

    // Reload cap and buckets (they may have changed after resize)
    let cap = builder.load(cap_ptr, usize_ty.clone());
    let buckets_ptr_ptr = builder.get_element_ptr(map_param, 0, ptr_bucket_ty.clone());
    let buckets_ptr = builder.load(buckets_ptr_ptr, ptr_bucket_ty.clone());

    // Allocate key on stack to pass pointer to hash function
    let key_alloc = builder.alloca(k_param.clone());
    builder.store(key_param, key_alloc);

    // Call hash_fn(&key)
    let hash = builder.call(hash_fn_param, vec![key_alloc]).unwrap();

    // index = hash % cap
    let index = builder.urem(hash, cap, usize_ty.clone());

    // Allocate probe counter
    let probe_alloc = builder.alloca(usize_ty.clone());
    let zero = builder.const_u64(0);
    builder.store(zero, probe_alloc);

    builder.br(find_slot);

    // find_slot: Linear probing to find empty/tombstone/matching slot
    builder.set_insert_point(find_slot);
    builder.br(loop_check);

    // loop_check: Check if we've probed all buckets
    builder.set_insert_point(loop_check);

    let probe_count = builder.load(probe_alloc, usize_ty.clone());
    let probe_complete = builder.icmp(BinaryOp::Ge, probe_count, cap, usize_ty.clone());

    // If we've probed all buckets, just return (map is full - should resize first)
    builder.cond_br(probe_complete, insert_done, check_empty);

    // check_empty: Get current bucket
    builder.set_insert_point(check_empty);

    // current_index = (index + probe_count) % cap
    let offset = builder.load(probe_alloc, usize_ty.clone());
    let current_idx_raw = builder.add(index, offset, usize_ty.clone());
    let current_idx = builder.urem(current_idx_raw, cap, usize_ty.clone());

    // bucket_ptr = &buckets[current_idx]
    let bucket_ptr = builder.ptr_add(buckets_ptr, current_idx, bucket_kv_ty.clone());

    // Load bucket.state
    let state_ptr = builder.get_element_ptr(bucket_ptr, 0, u8_ty.clone());
    let state = builder.load(state_ptr, u8_ty.clone());

    // Check if state == 0 (empty)
    let zero_u8 = builder.const_u8(0);
    let is_empty = builder.icmp(BinaryOp::Eq, state, zero_u8, u8_ty.clone());
    builder.cond_br(is_empty, found_slot, check_tombstone);

    // check_tombstone: Check if state == 2 (tombstone)
    builder.set_insert_point(check_tombstone);

    let two_u8 = builder.const_u8(2);
    let is_tombstone = builder.icmp(BinaryOp::Eq, state, two_u8, u8_ty.clone());
    builder.cond_br(is_tombstone, found_slot, check_key_match);

    // check_key_match: Check if this bucket has the same key (update case)
    builder.set_insert_point(check_key_match);

    // Get pointer to bucket's key (field index 1)
    let bucket_key_ptr = builder.get_element_ptr(bucket_ptr, 1, k_param.clone());

    // Call eq_fn(&input_key, &bucket_key)
    let keys_equal = builder
        .call(eq_fn_param, vec![key_alloc, bucket_key_ptr])
        .unwrap();

    // If keys match, update the value; otherwise continue probing
    builder.cond_br(keys_equal, update_value, next_probe);

    // update_value: Keys match, update the value (don't increment len)
    builder.set_insert_point(update_value);

    // Just update the value field (state and key stay the same)
    let update_value_ptr = builder.get_element_ptr(bucket_ptr, 2, v_param.clone());
    builder.store(value_param, update_value_ptr);

    builder.br(insert_done);

    // found_slot: Insert the key-value pair
    builder.set_insert_point(found_slot);

    // Set state = 1 (occupied)
    let one_u8 = builder.const_u8(1);
    builder.store(one_u8, state_ptr);

    // Set key
    let key_ptr = builder.get_element_ptr(bucket_ptr, 1, k_param.clone());
    builder.store(key_param, key_ptr);

    // Set value
    let value_ptr = builder.get_element_ptr(bucket_ptr, 2, v_param.clone());
    builder.store(value_param, value_ptr);

    // Increment map->len
    let len_ptr = builder.get_element_ptr(map_param, 1, usize_ty.clone());
    let old_len = builder.load(len_ptr, usize_ty.clone());
    let one = builder.const_u64(1);
    let new_len = builder.add(old_len, one, usize_ty.clone());
    builder.store(new_len, len_ptr);

    builder.br(insert_done);

    // next_probe: Increment probe counter and continue
    builder.set_insert_point(next_probe);

    let old_probe = builder.load(probe_alloc, usize_ty.clone());
    let one_probe = builder.const_u64(1);
    let new_probe = builder.add(old_probe, one_probe, usize_ty.clone());
    builder.store(new_probe, probe_alloc);

    builder.br(loop_check);

    // insert_done: Return
    builder.set_insert_point(insert_done);

    let unit = builder.unit_value();
    builder.ret(unit);
}

/// Build: fn hashmap_get<K,V>(map: *HashMap<K,V>, key: *K, hash_fn: fn(*K) -> u64, eq_fn: fn(*K, *K) -> bool) -> Option<V>
/// Looks up a key and returns Option<V>
fn build_hashmap_get(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let bool_ty = builder.bool_type();
    let usize_ty = builder.u64_type();
    let ptr_k_ty = builder.ptr_type(k_param.clone());
    let u64_ty = builder.u64_type();

    // struct Bucket<K,V> { state: u8, key: K, value: V }
    let bucket_kv_ty = builder.struct_type(
        Some("Bucket"),
        vec![u8_ty.clone(), k_param.clone(), v_param.clone()],
    );
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty.clone());

    // struct HashMap<K,V> { buckets: *Bucket<K,V>, len: usize, cap: usize }
    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_hashmap_ty = builder.ptr_type(hashmap_kv_ty.clone());

    // Option<V> = enum { None, Some(V) } = struct { tag: u8, value: V }
    let option_v_ty = builder.struct_type(Some("Option"), vec![u8_ty.clone(), v_param.clone()]);

    // Hash function type
    let hash_fn_ty = builder.function_type(vec![ptr_k_ty.clone()], u64_ty.clone());

    // Equality function type
    let eq_fn_ty = builder.function_type(vec![ptr_k_ty.clone(), ptr_k_ty.clone()], bool_ty.clone());

    let func_id = builder
        .begin_generic_function("hashmap_get", vec!["K", "V"])
        .param("map", ptr_hashmap_ty.clone())
        .param("key", ptr_k_ty.clone())
        .param("hash_fn", hash_fn_ty)
        .param("eq_fn", eq_fn_ty)
        .returns(option_v_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    let hash_key = builder.create_block("hash_key");
    let find_slot = builder.create_block("find_slot");
    let loop_check = builder.create_block("loop_check");
    let check_state = builder.create_block("check_state");
    let check_key_match = builder.create_block("check_key_match");
    let compare_keys = builder.create_block("compare_keys");
    let found_value = builder.create_block("found_value");
    let next_probe = builder.create_block("next_probe");
    let not_found = builder.create_block("not_found");

    builder.set_insert_point(entry);

    let map_param = builder.get_param(0);
    let key_param = builder.get_param(1);
    let hash_fn_param = builder.get_param(2);
    let eq_fn_param = builder.get_param(3);

    // Get map->cap and map->buckets
    let cap_ptr = builder.get_element_ptr(map_param, 2, usize_ty.clone());
    let cap = builder.load(cap_ptr, usize_ty.clone());

    let buckets_ptr_ptr = builder.get_element_ptr(map_param, 0, ptr_bucket_ty.clone());
    let buckets_ptr = builder.load(buckets_ptr_ptr, ptr_bucket_ty.clone());

    builder.br(hash_key);

    // hash_key: Compute hash and initial index
    builder.set_insert_point(hash_key);

    let hash = builder.call(hash_fn_param, vec![key_param]).unwrap();
    let index = builder.urem(hash, cap, usize_ty.clone());

    let probe_alloc = builder.alloca(usize_ty.clone());
    let zero = builder.const_u64(0);
    builder.store(zero, probe_alloc);

    builder.br(find_slot);

    // find_slot: Linear probing
    builder.set_insert_point(find_slot);
    builder.br(loop_check);

    // loop_check
    builder.set_insert_point(loop_check);

    let probe_count = builder.load(probe_alloc, usize_ty.clone());
    let probe_complete = builder.icmp(BinaryOp::Ge, probe_count, cap, usize_ty.clone());
    builder.cond_br(probe_complete, not_found, check_state);

    // check_state: Get current bucket
    builder.set_insert_point(check_state);

    let offset = builder.load(probe_alloc, usize_ty.clone());
    let current_idx_raw = builder.add(index, offset, usize_ty.clone());
    let current_idx = builder.urem(current_idx_raw, cap, usize_ty.clone());

    let bucket_ptr = builder.ptr_add(buckets_ptr, current_idx, bucket_kv_ty.clone());

    let state_ptr = builder.get_element_ptr(bucket_ptr, 0, u8_ty.clone());
    let state = builder.load(state_ptr, u8_ty.clone());

    // If state == 0 (empty), key not found
    let zero_u8 = builder.const_u8(0);
    let is_empty = builder.icmp(BinaryOp::Eq, state, zero_u8, u8_ty.clone());
    builder.cond_br(is_empty, not_found, check_key_match);

    // check_key_match: If state == 1, check if key matches
    builder.set_insert_point(check_key_match);

    // Check if state == 1 (occupied)
    let one_u8 = builder.const_u8(1);
    let is_occupied = builder.icmp(BinaryOp::Eq, state, one_u8, u8_ty.clone());

    // If not occupied (tombstone), continue probing
    builder.cond_br(is_occupied, compare_keys, next_probe);

    // compare_keys: If occupied, compare keys
    builder.set_insert_point(compare_keys);

    // Get pointer to bucket's key
    let bucket_key_ptr = builder.get_element_ptr(bucket_ptr, 1, k_param.clone());

    // Call eq_fn(search_key, bucket_key)
    let keys_equal = builder
        .call(eq_fn_param, vec![key_param, bucket_key_ptr])
        .unwrap();

    // If keys match, return the value; otherwise continue probing
    builder.cond_br(keys_equal, found_value, next_probe);

    // found_value: Return Some(value)
    builder.set_insert_point(found_value);

    let value_ptr = builder.get_element_ptr(bucket_ptr, 2, v_param.clone());
    let value = builder.load(value_ptr, v_param.clone());

    // Build Option::Some(value) = { tag: 1, value: value }
    let some_option = builder.create_struct(option_v_ty.clone(), vec![one_u8, value]);

    builder.ret(some_option);

    // next_probe
    builder.set_insert_point(next_probe);

    let old_probe = builder.load(probe_alloc, usize_ty.clone());
    let one_probe = builder.const_u64(1);
    let new_probe = builder.add(old_probe, one_probe, usize_ty.clone());
    builder.store(new_probe, probe_alloc);

    builder.br(loop_check);

    // not_found: Return None
    builder.set_insert_point(not_found);

    // Build Option::None = { tag: 0, value: undefined }
    // We need a default value for V
    let zero_v = builder.unit_value(); // Placeholder - should use default for V
    let none_option = builder.create_struct(option_v_ty, vec![zero_u8, zero_v]);

    builder.ret(none_option);
}

/// Build: fn hashmap_len<K,V>(map: *HashMap<K,V>) -> usize
fn build_hashmap_len(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let usize_ty = builder.u64_type();

    let bucket_kv_ty = builder.struct_type(
        Some("Bucket"),
        vec![u8_ty.clone(), k_param.clone(), v_param.clone()],
    );
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty);

    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty, usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_hashmap_ty = builder.ptr_type(hashmap_kv_ty);

    let func_id = builder
        .begin_generic_function("hashmap_len", vec!["K", "V"])
        .param("map", ptr_hashmap_ty)
        .returns(usize_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let map_param = builder.get_param(0);

    // Get map->len (field index 1)
    let len_ptr = builder.get_element_ptr(map_param, 1, usize_ty.clone());
    let len = builder.load(len_ptr, usize_ty.clone());

    builder.ret(len);
}

/// Build: fn hashmap_capacity<K,V>(map: *HashMap<K,V>) -> usize
fn build_hashmap_capacity(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let usize_ty = builder.u64_type();

    let bucket_kv_ty = builder.struct_type(
        Some("Bucket"),
        vec![u8_ty.clone(), k_param.clone(), v_param.clone()],
    );
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty);

    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty, usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_hashmap_ty = builder.ptr_type(hashmap_kv_ty);

    let func_id = builder
        .begin_generic_function("hashmap_capacity", vec!["K", "V"])
        .param("map", ptr_hashmap_ty)
        .returns(usize_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let map_param = builder.get_param(0);

    // Get map->cap (field index 2)
    let cap_ptr = builder.get_element_ptr(map_param, 2, usize_ty.clone());
    let cap = builder.load(cap_ptr, usize_ty.clone());

    builder.ret(cap);
}

/// Build: fn hashmap_contains_key<K,V>(map: *HashMap<K,V>, key: *K, hash_fn: fn(*K) -> u64, eq_fn: fn(*K, *K) -> bool) -> bool
fn build_hashmap_contains_key(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let usize_ty = builder.u64_type();
    let bool_ty = builder.bool_type();
    let ptr_k_ty = builder.ptr_type(k_param.clone());
    let u64_ty = builder.u64_type();

    let bucket_kv_ty = builder.struct_type(
        Some("Bucket"),
        vec![u8_ty.clone(), k_param.clone(), v_param.clone()],
    );
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty.clone());

    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_hashmap_ty = builder.ptr_type(hashmap_kv_ty);

    let hash_fn_ty = builder.function_type(vec![ptr_k_ty.clone()], u64_ty.clone());
    let eq_fn_ty = builder.function_type(vec![ptr_k_ty.clone(), ptr_k_ty.clone()], bool_ty.clone());

    let func_id = builder
        .begin_generic_function("hashmap_contains_key", vec!["K", "V"])
        .param("map", ptr_hashmap_ty)
        .param("key", ptr_k_ty.clone())
        .param("hash_fn", hash_fn_ty)
        .param("eq_fn", eq_fn_ty)
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    let hash_key = builder.create_block("hash_key");
    let find_slot = builder.create_block("find_slot");
    let loop_check = builder.create_block("loop_check");
    let check_state = builder.create_block("check_state");
    let compare_keys = builder.create_block("compare_keys");
    let found_key = builder.create_block("found_key");
    let next_probe = builder.create_block("next_probe");
    let return_true = builder.create_block("return_true");
    let not_found = builder.create_block("not_found");

    builder.set_insert_point(entry);

    let map_param = builder.get_param(0);
    let key_param = builder.get_param(1);
    let hash_fn_param = builder.get_param(2);
    let eq_fn_param = builder.get_param(3);

    // Get map->buckets, map->cap
    let buckets_ptr = builder.get_element_ptr(map_param, 0, ptr_bucket_ty.clone());
    let buckets = builder.load(buckets_ptr, ptr_bucket_ty.clone());

    let cap_ptr = builder.get_element_ptr(map_param, 2, usize_ty.clone());
    let cap = builder.load(cap_ptr, usize_ty.clone());

    builder.br(hash_key);

    // hash_key: Call hash_fn(key)
    builder.set_insert_point(hash_key);
    let hash_value = builder.call(hash_fn_param, vec![key_param]).unwrap();

    // index = hash % cap
    let index = builder.urem(hash_value, cap, usize_ty.clone());

    builder.br(find_slot);

    // find_slot: Initialize probe counter
    builder.set_insert_point(find_slot);
    let zero = builder.const_u64(0);
    let probe_alloc = builder.alloca(usize_ty.clone());
    builder.store(zero, probe_alloc);

    builder.br(loop_check);

    // loop_check: Check if probe < cap
    builder.set_insert_point(loop_check);
    let probe = builder.load(probe_alloc, usize_ty.clone());
    let continue_loop = builder.icmp(BinaryOp::Lt, probe, cap, usize_ty.clone());
    builder.cond_br(continue_loop, check_state, not_found);

    // check_state: Calculate current bucket index and check state
    builder.set_insert_point(check_state);
    let current_index = builder.add(index, probe, usize_ty.clone());
    let wrapped_index = builder.urem(current_index, cap, usize_ty.clone());

    let bucket_ptr = builder.ptr_add(buckets, wrapped_index, bucket_kv_ty.clone());
    let state_ptr = builder.get_element_ptr(bucket_ptr, 0, u8_ty.clone());
    let state = builder.load(state_ptr, u8_ty.clone());

    // Check if state == 0 (empty) - key not found
    let zero_u8 = builder.const_u8(0);
    let is_empty = builder.icmp(BinaryOp::Eq, state, zero_u8, u8_ty.clone());
    builder.cond_br(is_empty, not_found, compare_keys);

    // compare_keys: Check if state == 1 (occupied) and compare keys
    builder.set_insert_point(compare_keys);
    let one_u8 = builder.const_u8(1);
    let is_occupied = builder.icmp(BinaryOp::Eq, state, one_u8, u8_ty.clone());

    // If not occupied (tombstone), skip to next probe
    builder.cond_br(is_occupied, found_key, next_probe);

    // found_key: Compare keys using eq_fn
    builder.set_insert_point(found_key);
    let bucket_key_ptr = builder.get_element_ptr(bucket_ptr, 1, k_param);
    let keys_equal = builder
        .call(eq_fn_param, vec![key_param, bucket_key_ptr])
        .unwrap();

    // If keys match, return true; otherwise continue probing
    builder.cond_br(keys_equal, return_true, next_probe);

    // return_true: Key found, return true
    builder.set_insert_point(return_true);
    let true_val = builder.const_bool(true);
    builder.ret(true_val);

    // next_probe: Increment probe and continue
    builder.set_insert_point(next_probe);
    let current_probe = builder.load(probe_alloc, usize_ty.clone());
    let one = builder.const_u64(1);
    let next_probe_val = builder.add(current_probe, one, usize_ty.clone());
    builder.store(next_probe_val, probe_alloc);
    builder.br(loop_check);

    // not_found: Return false
    builder.set_insert_point(not_found);
    let false_val = builder.const_bool(false);
    builder.ret(false_val);
}

/// Build: fn hashmap_remove<K,V>(map: *HashMap<K,V>, key: *K, hash_fn: fn(*K) -> u64, eq_fn: fn(*K, *K) -> bool) -> bool
fn build_hashmap_remove(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let usize_ty = builder.u64_type();
    let bool_ty = builder.bool_type();
    let ptr_k_ty = builder.ptr_type(k_param.clone());
    let u64_ty = builder.u64_type();

    let bucket_kv_ty = builder.struct_type(
        Some("Bucket"),
        vec![u8_ty.clone(), k_param.clone(), v_param.clone()],
    );
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty.clone());

    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty.clone(), usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_hashmap_ty = builder.ptr_type(hashmap_kv_ty);

    let hash_fn_ty = builder.function_type(vec![ptr_k_ty.clone()], u64_ty.clone());
    let eq_fn_ty = builder.function_type(vec![ptr_k_ty.clone(), ptr_k_ty.clone()], bool_ty.clone());

    let func_id = builder
        .begin_generic_function("hashmap_remove", vec!["K", "V"])
        .param("map", ptr_hashmap_ty.clone())
        .param("key", ptr_k_ty.clone())
        .param("hash_fn", hash_fn_ty)
        .param("eq_fn", eq_fn_ty)
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    let hash_key = builder.create_block("hash_key");
    let find_slot = builder.create_block("find_slot");
    let loop_check = builder.create_block("loop_check");
    let check_state = builder.create_block("check_state");
    let check_key_match = builder.create_block("check_key_match");
    let compare_keys = builder.create_block("compare_keys");
    let found_key = builder.create_block("found_key");
    let next_probe = builder.create_block("next_probe");
    let not_found = builder.create_block("not_found");

    builder.set_insert_point(entry);

    let map_param = builder.get_param(0);
    let key_param = builder.get_param(1);
    let hash_fn_param = builder.get_param(2);
    let eq_fn_param = builder.get_param(3);

    // Get map->cap and map->buckets
    let cap_ptr = builder.get_element_ptr(map_param, 2, usize_ty.clone());
    let cap = builder.load(cap_ptr, usize_ty.clone());

    let buckets_ptr_ptr = builder.get_element_ptr(map_param, 0, ptr_bucket_ty.clone());
    let buckets_ptr = builder.load(buckets_ptr_ptr, ptr_bucket_ty.clone());

    builder.br(hash_key);

    // hash_key: Compute hash and initial index
    builder.set_insert_point(hash_key);

    let hash = builder.call(hash_fn_param, vec![key_param]).unwrap();
    let index = builder.urem(hash, cap, usize_ty.clone());

    let probe_alloc = builder.alloca(usize_ty.clone());
    let zero = builder.const_u64(0);
    builder.store(zero, probe_alloc);

    builder.br(find_slot);

    // find_slot: Linear probing
    builder.set_insert_point(find_slot);
    builder.br(loop_check);

    // loop_check: Check if we've probed all buckets
    builder.set_insert_point(loop_check);

    let probe_count = builder.load(probe_alloc, usize_ty.clone());
    let probe_complete = builder.icmp(BinaryOp::Ge, probe_count, cap, usize_ty.clone());
    builder.cond_br(probe_complete, not_found, check_state);

    // check_state: Get current bucket
    builder.set_insert_point(check_state);

    let offset = builder.load(probe_alloc, usize_ty.clone());
    let current_idx_raw = builder.add(index, offset, usize_ty.clone());
    let current_idx = builder.urem(current_idx_raw, cap, usize_ty.clone());

    let bucket_ptr = builder.ptr_add(buckets_ptr, current_idx, bucket_kv_ty.clone());

    let state_ptr = builder.get_element_ptr(bucket_ptr, 0, u8_ty.clone());
    let state = builder.load(state_ptr, u8_ty.clone());

    // If state == 0 (empty), key not found
    let zero_u8 = builder.const_u8(0);
    let is_empty = builder.icmp(BinaryOp::Eq, state, zero_u8, u8_ty.clone());
    builder.cond_br(is_empty, not_found, check_key_match);

    // check_key_match: If state == 1, check if key matches
    builder.set_insert_point(check_key_match);

    let one_u8 = builder.const_u8(1);
    let is_occupied = builder.icmp(BinaryOp::Eq, state, one_u8, u8_ty.clone());
    builder.cond_br(is_occupied, compare_keys, next_probe);

    // compare_keys: Compare keys
    builder.set_insert_point(compare_keys);

    let bucket_key_ptr = builder.get_element_ptr(bucket_ptr, 1, k_param.clone());
    let keys_equal = builder
        .call(eq_fn_param, vec![key_param, bucket_key_ptr])
        .unwrap();
    builder.cond_br(keys_equal, found_key, next_probe);

    // found_key: Mark as tombstone and decrement len
    builder.set_insert_point(found_key);

    // Set state = 2 (tombstone)
    let two_u8 = builder.const_u8(2);
    builder.store(two_u8, state_ptr);

    // Decrement map->len
    let len_ptr = builder.get_element_ptr(map_param, 1, usize_ty.clone());
    let old_len = builder.load(len_ptr, usize_ty.clone());
    let one = builder.const_u64(1);
    let new_len = builder.sub(old_len, one, usize_ty.clone());
    builder.store(new_len, len_ptr);

    // Return true
    let true_val = builder.const_bool(true);
    builder.ret(true_val);

    // next_probe: Continue probing
    builder.set_insert_point(next_probe);

    let old_probe = builder.load(probe_alloc, usize_ty.clone());
    let one_probe = builder.const_u64(1);
    let new_probe = builder.add(old_probe, one_probe, usize_ty.clone());
    builder.store(new_probe, probe_alloc);

    builder.br(loop_check);

    // not_found: Return false
    builder.set_insert_point(not_found);

    let false_val = builder.const_bool(false);
    builder.ret(false_val);
}

/// Build: fn hashmap_clear<K,V>(map: *HashMap<K,V>)
fn build_hashmap_clear(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let bucket_kv_ty = builder.struct_type(Some("Bucket"), vec![u8_ty.clone(), k_param, v_param]);
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty);

    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty, usize_ty.clone(), usize_ty.clone()],
    );
    let ptr_hashmap_ty = builder.ptr_type(hashmap_kv_ty);

    let func_id = builder
        .begin_generic_function("hashmap_clear", vec!["K", "V"])
        .param("map", ptr_hashmap_ty)
        .returns(void_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let map_param = builder.get_param(0);

    // Set map->len = 0
    let len_ptr = builder.get_element_ptr(map_param, 1, usize_ty.clone());
    let zero = builder.const_u64(0);
    builder.store(zero, len_ptr);

    // TODO: Should also memset buckets to 0

    let unit = builder.unit_value();
    builder.ret(unit);
}

/// Build: fn hashmap_free<K,V>(map: *HashMap<K,V>)
fn build_hashmap_free(builder: &mut HirBuilder) {
    let k_param = builder.type_param("K");
    let v_param = builder.type_param("V");
    let u8_ty = builder.u8_type();
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();

    let bucket_kv_ty = builder.struct_type(Some("Bucket"), vec![u8_ty.clone(), k_param, v_param]);
    let ptr_bucket_ty = builder.ptr_type(bucket_kv_ty);

    let hashmap_kv_ty = builder.struct_type(
        Some("HashMap"),
        vec![ptr_bucket_ty.clone(), usize_ty.clone(), usize_ty],
    );
    let ptr_hashmap_ty = builder.ptr_type(hashmap_kv_ty);

    let func_id = builder
        .begin_generic_function("hashmap_free", vec!["K", "V"])
        .param("map", ptr_hashmap_ty)
        .returns(void_ty.clone())
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let map_param = builder.get_param(0);

    // Get map->buckets
    let buckets_ptr_ptr = builder.get_element_ptr(map_param, 0, ptr_bucket_ty.clone());
    let buckets_ptr = builder.load(buckets_ptr_ptr, ptr_bucket_ty);

    // Cast to *u8 for free
    let u8_ptr_ty = builder.ptr_type(u8_ty);
    let ptr_u8 = builder.bitcast(buckets_ptr, u8_ptr_ty);

    // Call free
    let free_name = builder.intern("free");
    let free_id = builder.get_function_by_name(free_name);
    let free_ref = builder.function_ref(free_id);
    let _ = builder.call(free_ref, vec![ptr_u8]);

    let unit = builder.unit_value();
    builder.ret(unit);
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::AstArena;

    #[test]
    fn test_build_hash_functions() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_hash", &mut arena);

        build_hash_functions(&mut builder);

        let module = builder.finish();

        // Should have 6 hash functions
        assert_eq!(module.functions.len(), 6);

        // Verify function names
        let func_names: Vec<String> = module
            .functions
            .iter()
            .map(|(_, f)| arena.resolve_string(f.name).unwrap().to_string())
            .collect();

        assert!(func_names.contains(&"hash_i32".to_string()));
        assert!(func_names.contains(&"hash_i64".to_string()));
        assert!(func_names.contains(&"hash_u32".to_string()));
        assert!(func_names.contains(&"hash_u64".to_string()));
        assert!(func_names.contains(&"hash_bool".to_string()));
        assert!(func_names.contains(&"hash_u8".to_string()));
    }

    #[test]
    fn test_build_hashmap_new() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_hashmap", &mut arena);

        // Need memory functions for malloc and memset
        crate::stdlib::memory::build_memory_functions(&mut builder);
        declare_c_memset(&mut builder);

        build_hashmap_new(&mut builder);

        let module = builder.finish();

        // Should have hashmap_new + memory functions
        assert!(module.functions.len() >= 1);

        // Find hashmap_new
        let (_, hashmap_new) = module
            .functions
            .iter()
            .find(|(_, f)| arena.resolve_string(f.name) == Some("hashmap_new"))
            .expect("hashmap_new not found");

        // Should be generic with 2 type params (K, V)
        assert_eq!(hashmap_new.signature.type_params.len(), 2);

        // Should have no parameters (no args)
        assert_eq!(hashmap_new.signature.params.len(), 0);

        // Should have at least entry block
        assert!(!hashmap_new.blocks.is_empty());
    }

    #[test]
    fn test_build_hashmap_insert() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_hashmap_insert", &mut arena);

        crate::stdlib::memory::build_memory_functions(&mut builder);
        declare_c_memset(&mut builder);
        declare_c_realloc(&mut builder);

        // Need resize function since insert calls it
        build_hashmap_resize(&mut builder);
        build_hashmap_insert(&mut builder);

        let module = builder.finish();

        let (_, hashmap_insert) = module
            .functions
            .iter()
            .find(|(_, f)| arena.resolve_string(f.name) == Some("hashmap_insert"))
            .expect("hashmap_insert not found");

        // Should be generic with 2 type params
        assert_eq!(hashmap_insert.signature.type_params.len(), 2);

        // Should have 5 parameters: map, key, value, hash_fn, eq_fn
        assert_eq!(hashmap_insert.signature.params.len(), 5);

        // Should have multiple blocks for control flow
        assert!(hashmap_insert.blocks.len() >= 5);
    }

    #[test]
    fn test_build_hashmap_get() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_hashmap_get", &mut arena);

        crate::stdlib::memory::build_memory_functions(&mut builder);

        build_hashmap_get(&mut builder);

        let module = builder.finish();

        let (_, hashmap_get) = module
            .functions
            .iter()
            .find(|(_, f)| arena.resolve_string(f.name) == Some("hashmap_get"))
            .expect("hashmap_get not found");

        // Should be generic with 2 type params
        assert_eq!(hashmap_get.signature.type_params.len(), 2);

        // Should have 4 parameters: map, key, hash_fn, eq_fn
        assert_eq!(hashmap_get.signature.params.len(), 4);

        // Should have multiple blocks
        assert!(hashmap_get.blocks.len() >= 5);
    }

    #[test]
    fn test_build_all_hashmap_functions() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_all_hashmap", &mut arena);

        crate::stdlib::memory::build_memory_functions(&mut builder);
        build_hashmap(&mut builder);

        let module = builder.finish();

        let func_names: Vec<String> = module
            .functions
            .iter()
            .map(|(_, f)| arena.resolve_string(f.name).unwrap().to_string())
            .collect();

        // Count hash functions
        let hash_count = func_names
            .iter()
            .filter(|name| name.starts_with("hash_"))
            .count();
        assert_eq!(hash_count, 6, "Should have exactly 6 hash functions");

        // Count hashmap functions
        let hashmap_count = func_names
            .iter()
            .filter(|name| name.starts_with("hashmap_"))
            .count();
        assert_eq!(
            hashmap_count, 11,
            "Should have exactly 11 hashmap functions"
        );

        // Verify specific hashmap functions exist
        assert!(func_names.contains(&"hashmap_new".to_string()));
        assert!(func_names.contains(&"hashmap_with_capacity".to_string()));
        assert!(func_names.contains(&"hashmap_resize".to_string()));
        assert!(func_names.contains(&"hashmap_insert".to_string()));
        assert!(func_names.contains(&"hashmap_get".to_string()));
        assert!(func_names.contains(&"hashmap_remove".to_string()));
        assert!(func_names.contains(&"hashmap_contains_key".to_string()));
        assert!(func_names.contains(&"hashmap_len".to_string()));
        assert!(func_names.contains(&"hashmap_capacity".to_string()));
        assert!(func_names.contains(&"hashmap_clear".to_string()));
        assert!(func_names.contains(&"hashmap_free".to_string()));
    }
}
