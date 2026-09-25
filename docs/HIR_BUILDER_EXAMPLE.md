# HIR Builder API - Usage Examples

## Overview

The HIR Builder provides a fluent API for constructing HIR modules directly in Rust, similar to LLVM's IRBuilder or Cranelift's FunctionBuilder. This enables language engineers to translate their AST to HIR without requiring source-based compilation.

## Basic Example: Simple Function

```rust
use zyntax_compiler::HirBuilder;
use zyntax_typed_ast::AstArena;

let mut arena = AstArena::new();
let mut builder = HirBuilder::new("my_module", &mut arena);

// Build types first
let i32_ty = builder.i32_type();

// Define function: fn add(a: i32, b: i32) -> i32
let func_id = builder.begin_function("add")
    .param("a", i32_ty.clone())
    .param("b", i32_ty.clone())
    .returns(i32_ty.clone())
    .build();

// Set function context
builder.set_current_function(func_id);

// Create entry block
let entry = builder.create_block("entry");
builder.set_insert_point(entry);

// Build function body: return a + b
let a = builder.get_param(0);
let b = builder.get_param(1);
let result = builder.add(a, b, i32_ty);
builder.ret(result);

// Finish and get the HIR module
let module = builder.finish();
```

## Standard Library Example: Option<T>

Here's how you would build a standard library `Option<T>` type and its methods:

```rust
use zyntax_compiler::HirBuilder;
use zyntax_typed_ast::AstArena;

let mut arena = AstArena::new();
let mut builder = HirBuilder::new("std_option", &mut arena);

// Define Option<T> as a union type
// union Option<T> {
//     None,      // variant 0: no value
//     Some(T),   // variant 1: value of type T
// }

// For demonstration, let's build Option<i32>
let i32_ty = builder.i32_type();
let void_ty = builder.void_type();

let option_i32_variants = vec![
    HirUnionVariant {
        name: builder.intern("None"),
        ty: void_ty,  // None carries no data
        discriminant: 0,
    },
    HirUnionVariant {
        name: builder.intern("Some"),
        ty: i32_ty.clone(),  // Some carries an i32
        discriminant: 1,
    },
];

let option_i32_ty = builder.union_type(Some("Option"), option_i32_variants);

// Build: fn unwrap(opt: Option<i32>) -> i32
let unwrap_func = builder.begin_function("unwrap")
    .param("opt", option_i32_ty.clone())
    .returns(i32_ty.clone())
    .build();

builder.set_current_function(unwrap_func);
let entry = builder.create_block("entry");
let some_block = builder.create_block("some_case");
let none_block = builder.create_block("none_case");
builder.set_insert_point(entry);

// Get parameter and extract discriminant
let opt = builder.get_param(0);
let discriminant = builder.extract_discriminant(opt);
let one = builder.const_i32(1);

// Check if discriminant == 1 (Some)
let is_some = builder.icmp(IntPredicate::Eq, discriminant, one, builder.bool_type());
builder.cond_br(is_some, some_block, none_block);

// Some case: extract and return the value
builder.set_insert_point(some_block);
let value = builder.extract_union_value(opt, 1, i32_ty.clone());
builder.ret(value);

// None case: panic (abort)
builder.set_insert_point(none_block);
builder.panic();
builder.unreachable();

let module = builder.finish();
```

## FFI Example: Wrapping C malloc/free

Using Gap 11's extern function support:

```rust
use zyntax_compiler::{HirBuilder, CallingConvention};
use zyntax_typed_ast::AstArena;

let mut arena = AstArena::new();
let mut builder = HirBuilder::new("std_alloc", &mut arena);

// Build types
let usize_ty = builder.u64_type();
let u8_ty = builder.u8_type();
let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
let void_ty = builder.void_type();

// Declare extern functions
let malloc = builder.begin_extern_function("malloc", CallingConvention::C)
    .param("size", usize_ty.clone())
    .returns(ptr_u8_ty.clone())
    .build();

let free = builder.begin_extern_function("free", CallingConvention::C)
    .param("ptr", ptr_u8_ty.clone())
    .returns(void_ty.clone())
    .build();

// Build safe wrapper: fn allocate(size: usize) -> Option<*u8>
// Returns None if allocation fails (null pointer)

let option_ptr_variants = vec![
    HirUnionVariant {
        name: builder.intern("None"),
        ty: void_ty,
        discriminant: 0,
    },
    HirUnionVariant {
        name: builder.intern("Some"),
        ty: ptr_u8_ty.clone(),
        discriminant: 1,
    },
];

let option_ptr_ty = builder.union_type(Some("Option"), option_ptr_variants);

let allocate_func = builder.begin_function("allocate")
    .param("size", usize_ty.clone())
    .returns(option_ptr_ty.clone())
    .build();

builder.set_current_function(allocate_func);
let entry = builder.create_block("entry");
let null_check = builder.create_block("null_check");
let success_block = builder.create_block("success");
let failure_block = builder.create_block("failure");

builder.set_insert_point(entry);

// Call malloc
let size = builder.get_param(0);
let malloc_ref = builder.function_ref(malloc);
let ptr = builder.call(malloc_ref, vec![size]).unwrap();

// Check if null
builder.br(null_check);

builder.set_insert_point(null_check);
let null = builder.null_ptr(u8_ty);
let is_null = builder.icmp(IntPredicate::Eq, ptr, null, builder.bool_type());
builder.cond_br(is_null, failure_block, success_block);

// Success: return Some(ptr)
builder.set_insert_point(success_block);
let some_value = builder.create_union(option_ptr_ty.clone(), 1, ptr);
builder.ret(some_value);

// Failure: return None
builder.set_insert_point(failure_block);
let none_value = builder.create_union(option_ptr_ty, 0, builder.unit_value());
builder.ret(none_value);

let module = builder.finish();
```

## Architecture Benefits

### 1. No Parser Required
Language frontends can directly construct HIR from their AST without needing to generate and parse Zyntax source code.

### 2. Type Safety
The builder API provides compile-time type checking for HIR construction. Invalid HIR structures are caught at Rust compile time.

### 3. Familiar Pattern
Developers familiar with LLVM IRBuilder or Cranelift FunctionBuilder will find this API intuitive.

### 4. Standard Library in HIR
The compiler can ship with a pre-built HIR standard library, avoiding bootstrap issues.

### 5. Performance
Direct HIR construction is faster than parsing source code.

## API Structure

### Module Building
- `HirBuilder::new(name, arena)` - Create new module builder
- `builder.finish()` - Complete module and return `HirModule`

### Type Construction
- `builder.i32_type()`, `builder.bool_type()`, etc. - Primitive types
- `builder.ptr_type(pointee)` - Pointer types
- `builder.struct_type(name, fields)` - Struct types
- `builder.union_type(name, variants)` - Union/enum types
- `builder.array_type(element, count)` - Array types

### Function Building
- `builder.begin_function(name)` - Start regular function
- `builder.begin_extern_function(name, cc)` - Start extern function
- `builder.begin_generic_function(name, type_params)` - Start generic function
- `.param(name, type)` - Add parameter
- `.returns(type)` - Set return type
- `.build()` - Finish function signature

### Block Building
- `builder.create_block(name)` - Create new block
- `builder.set_insert_point(block)` - Set where to insert instructions

### Instruction Building
- `builder.add(lhs, rhs, type)` - Arithmetic
- `builder.load(ptr, type)` - Memory load
- `builder.store(value, ptr)` - Memory store
- `builder.call(callee, args)` - Function call
- `builder.icmp(pred, lhs, rhs, type)` - Integer comparison

### Control Flow
- `builder.ret(value)` - Return value
- `builder.br(target)` - Unconditional branch
- `builder.cond_br(cond, then_block, else_block)` - Conditional branch
- `builder.switch(value, default, cases)` - Switch statement
- `builder.unreachable()` - Unreachable marker

### Constants
- `builder.const_i32(value)` - Integer constant
- `builder.const_bool(value)` - Boolean constant
- `builder.null_ptr(type)` - Null pointer
- `builder.unit_value()` - Unit/void value

### Union/Enum Operations
- `builder.create_union(type, variant, value)` - Create union value
- `builder.extract_union_value(union, variant, type)` - Extract variant
- `builder.extract_discriminant(union)` - Get discriminant

## Future Extensions

### Planned Features
1. **Compiler Intrinsics** - size_of, align_of, transmute
2. **Link Name Attributes** - Custom symbol names for FFI
3. **Variadic Functions** - printf-style functions
4. **Inline Assembly** - Platform-specific operations
5. **Metadata Attributes** - Documentation, optimization hints

### Example: Compiler Intrinsics (Future)

```rust
// Future API:
let size = builder.call_intrinsic(Intrinsic::SizeOf, vec![], vec![struct_ty]);
let align = builder.call_intrinsic(Intrinsic::AlignOf, vec![], vec![struct_ty]);
```

## Integration with Gap 10 (Standard Library)

The HIR Builder enables Gap 10 by allowing the standard library to be written as HIR construction code:

```rust
// std/lib.rs - Standard library as HIR builder code
pub fn build_stdlib(arena: &mut AstArena) -> HirModule {
    let mut builder = HirBuilder::new("std", arena);

    // Build Option<T>
    build_option_type(&mut builder);

    // Build Result<T, E>
    build_result_type(&mut builder);

    // Build Vec<T>
    build_vec_type(&mut builder);

    // Build String
    build_string_type(&mut builder);

    builder.finish()
}
```

This approach:
- ✅ No parser needed
- ✅ Type-safe construction
- ✅ Version controlled as Rust code
- ✅ Can be unit tested
- ✅ Fast compilation
- ✅ Easy to maintain

## Status

**Completed**: ✅ HIR Builder implementation (708 lines)
**Next Step**: Use HIR Builder to implement Gap 10 (Standard Library)

---

**Created**: November 13, 2025
**Gap**: Gap 10 - Standard Library (Phase 1)
**Related**: Gap 11 - Extern Functions & FFI
