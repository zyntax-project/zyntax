# Chapter 11: HIR Builder

The `HirBuilder` provides a fluent API for constructing HIR (High-level Intermediate Representation) directly, bypassing TypedAST. This is useful for:

- Building standard library types and functions
- Language frontend engineers who want to emit HIR directly
- Testing and prototyping compiler features
- Compiler intrinsics and built-ins

## Design Philosophy

The HIR Builder follows the same pattern as LLVM's IRBuilder and Cranelift's FunctionBuilder:

- **Stateful builder** - Tracks current insertion point (function, block)
- **Type-safe API** - Prevents invalid HIR construction
- **Automatic SSA management** - Values are automatically in SSA form
- **Arena-based strings** - Uses `AstArena` for zero-cost string interning

## Getting Started

```rust
use zyntax_compiler::hir_builder::HirBuilder;
use zyntax_typed_ast::arena::AstArena;

fn main() {
    // Create arena and builder
    let mut arena = AstArena::new();
    let mut builder = HirBuilder::new("my_module", &mut arena);

    // Define a simple add function: fn add(a: i32, b: i32) -> i32
    let i32_ty = builder.i32_type();
    let func_id = builder.begin_function("add")
        .param("a", i32_ty.clone())
        .param("b", i32_ty.clone())
        .returns(i32_ty.clone())
        .build();

    // Build function body
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let a = builder.get_param(0);
    let b = builder.get_param(1);
    let result = builder.add(a, b, i32_ty);
    builder.ret(result);

    // Finish and get the HIR module
    let hir_module = builder.finish();
}
```

## Type Construction

### Primitive Types

```rust
let void_ty = builder.void_type();   // void
let bool_ty = builder.bool_type();   // bool

// Signed integers
let i8_ty = builder.i8_type();       // i8
let i16_ty = builder.i16_type();     // i16
let i32_ty = builder.i32_type();     // i32
let i64_ty = builder.i64_type();     // i64

// Unsigned integers
let u8_ty = builder.u8_type();       // u8
let u16_ty = builder.u16_type();     // u16
let u32_ty = builder.u32_type();     // u32
let u64_ty = builder.u64_type();     // u64

// Floating point
let f32_ty = builder.f32_type();     // f32
let f64_ty = builder.f64_type();     // f64
```

### Compound Types

```rust
// Pointer type: *T
let ptr_i32 = builder.ptr_type(builder.i32_type());

// Array type: [T; N]
let arr_ty = builder.array_type(builder.i32_type(), 10);

// Struct type: { i32, f64, bool }
let struct_ty = builder.struct_type(
    Some("Point"),
    vec![builder.i32_type(), builder.i32_type()],
);

// Function type: fn(i32, i32) -> bool
let func_ty = builder.function_type(
    vec![builder.i32_type(), builder.i32_type()],
    builder.bool_type(),
);
```

### Generic Types

```rust
// Type parameter (opaque)
let t_param = builder.type_param("T");

// Generic type instantiation: Vec<i32>
let vec_i32 = builder.generic_type(
    vec_base_type,
    vec![builder.i32_type()],  // type args
    vec![],                     // const args
);
```

### Union/Enum Types

```rust
use zyntax_compiler::hir::HirUnionVariant;

// Create an Option-like enum
let option_ty = builder.union_type(
    Some("Option"),
    vec![
        HirUnionVariant {
            name: builder.intern("None"),
            fields: vec![],
        },
        HirUnionVariant {
            name: builder.intern("Some"),
            fields: vec![builder.i32_type()],
        },
    ],
);
```

## Function Construction

### Basic Functions

```rust
let i32_ty = builder.i32_type();

// fn add(a: i32, b: i32) -> i32
let func_id = builder.begin_function("add")
    .param("a", i32_ty.clone())
    .param("b", i32_ty.clone())
    .returns(i32_ty.clone())
    .build();
```

### Generic Functions

```rust
// fn identity<T>(x: T) -> T
let func_id = builder.begin_generic_function("identity", vec!["T"])
    .param("x", builder.type_param("T"))
    .returns(builder.type_param("T"))
    .build();
```

### External Functions

```rust
use zyntax_compiler::hir::CallingConvention;

// extern "C" fn malloc(size: u64) -> *u8
let malloc = builder.begin_extern_function("malloc", CallingConvention::C)
    .param("size", builder.u64_type())
    .returns(builder.ptr_type(builder.u8_type()))
    .build();

// extern "C" fn free(ptr: *u8)
let free = builder.begin_extern_function("free", CallingConvention::C)
    .param("ptr", builder.ptr_type(builder.u8_type()))
    .returns(builder.void_type())
    .build();
```

### Setting Up Function Bodies

After building a function, set up the body:

```rust
// Set the current function
builder.set_current_function(func_id);

// Create the entry block
let entry = builder.create_block("entry");
builder.set_insert_point(entry);

// Access parameters by index
let param0 = builder.get_param(0);
let param1 = builder.get_param(1);
```

## Block Construction

### Creating Blocks

```rust
builder.set_current_function(func_id);

// Create named blocks
let entry = builder.create_block("entry");
let then_block = builder.create_block("then");
let else_block = builder.create_block("else");
let merge_block = builder.create_block("merge");

// Set insertion point
builder.set_insert_point(entry);
```

### PHI Nodes

PHI nodes merge values from different control flow paths:

```rust
// In merge_block, after converging from then_block and else_block
builder.set_insert_point(merge_block);

let result = builder.add_phi(
    builder.i32_type(),
    vec![
        (then_value, then_block),   // value from then_block
        (else_value, else_block),   // value from else_block
    ],
);
```

## Constants

### Integer Constants

```rust
let i32_val = builder.const_i32(42);
let u64_val = builder.const_u64(1000);
let u8_val = builder.const_u8(255);
let bool_val = builder.const_bool(true);
```

### Floating Point Constants

```rust
let f64_val = builder.const_f64(3.14159);
```

### String Constants

```rust
// Creates a global string and returns pointer to it
let str_ptr = builder.string_constant("Hello, World!");
```

### Special Values

```rust
// Null pointer
let null = builder.null_ptr(builder.i32_type());

// Unit/void value
let unit = builder.unit_value();

// Undefined value (for uninitialized data)
let undef = builder.undef(builder.i32_type());
```

## Arithmetic Instructions

### Binary Operations

```rust
let i32_ty = builder.i32_type();
let a = builder.get_param(0);
let b = builder.get_param(1);

// Arithmetic
let sum = builder.add(a, b, i32_ty.clone());
let diff = builder.sub(a, b, i32_ty.clone());
let product = builder.mul(a, b, i32_ty.clone());
let quotient = builder.div(a, b, i32_ty.clone());
let remainder = builder.urem(a, b, i32_ty.clone());

// Bitwise
let xor_result = builder.xor(a, b, i32_ty.clone());
```

### Comparisons

```rust
use zyntax_compiler::hir::BinaryOp;

let i32_ty = builder.i32_type();

// Equality
let eq = builder.icmp_eq(a, b, i32_ty.clone());

// Other comparisons using icmp
let lt = builder.icmp(BinaryOp::Lt, a, b, i32_ty.clone());
let le = builder.icmp(BinaryOp::Le, a, b, i32_ty.clone());
let gt = builder.icmp(BinaryOp::Gt, a, b, i32_ty.clone());
let ge = builder.icmp(BinaryOp::Ge, a, b, i32_ty.clone());
let ne = builder.icmp(BinaryOp::Ne, a, b, i32_ty.clone());
```

### Type Conversions

```rust
// Zero extend (u8 -> u64)
let extended = builder.zext(u8_value, builder.u64_type());

// Bitcast (reinterpret bits)
let reinterpreted = builder.bitcast(ptr_value, builder.ptr_type(builder.i32_type()));
```

## Memory Operations

### Stack Allocation

```rust
// Allocate space on the stack
let ptr = builder.alloca(builder.i32_type());  // Returns *i32
```

### Load and Store

```rust
// Store value to pointer
builder.store(value, ptr);

// Load value from pointer
let loaded = builder.load(ptr, builder.i32_type());
```

### Struct Field Access

```rust
// Given a pointer to a struct, get pointer to a field
let field_ptr = builder.get_element_ptr(struct_ptr, 0, builder.i32_type());
let field_value = builder.load(field_ptr, builder.i32_type());

// Or extract from a struct value directly
let field = builder.extract_struct_field(struct_value, 0, builder.i32_type());
```

### Creating Structs

```rust
let struct_ty = builder.struct_type(Some("Point"), vec![
    builder.i32_type(),
    builder.i32_type(),
]);

// Create struct from field values
let x_val = builder.const_i32(10);
let y_val = builder.const_i32(20);
let point = builder.create_struct(struct_ty, vec![x_val, y_val]);
```

### Pointer Arithmetic

```rust
// ptr + offset (in elements, not bytes)
let next_element = builder.ptr_add(ptr, offset, ptr_ty);
```

## Control Flow

### Terminators

Every block must end with a terminator:

```rust
// Return with value
builder.ret(result);

// Return void
builder.ret_void();

// Unconditional branch
builder.br(target_block);

// Conditional branch
builder.cond_br(condition, then_block, else_block);

// Switch on value
builder.switch(
    discriminant,
    vec![
        (HirConstant::I32(0), case0_block),
        (HirConstant::I32(1), case1_block),
    ],
    default_block,
);

// Unreachable (for dead code)
builder.unreachable();
```

### Control Flow Example

```rust
// if (x > 0) { return x; } else { return -x; }
let func_id = builder.begin_function("abs")
    .param("x", builder.i32_type())
    .returns(builder.i32_type())
    .build();

builder.set_current_function(func_id);

let entry = builder.create_block("entry");
let then_block = builder.create_block("then");
let else_block = builder.create_block("else");

// Entry block
builder.set_insert_point(entry);
let x = builder.get_param(0);
let zero = builder.const_i32(0);
let is_positive = builder.icmp(BinaryOp::Gt, x, zero, builder.i32_type());
builder.cond_br(is_positive, then_block, else_block);

// Then block: return x
builder.set_insert_point(then_block);
builder.ret(x);

// Else block: return -x
builder.set_insert_point(else_block);
let neg_x = builder.sub(zero, x, builder.i32_type());
builder.ret(neg_x);
```

## Function Calls

### Direct Calls

```rust
// Call a function by ID
let result = builder.call(callee_func_id, vec![arg1, arg2]);
```

### Calling External Symbols

```rust
// Call a runtime symbol by name
let result = builder.call_symbol("$IO$println_i64", vec![value]);
```

### Intrinsics

```rust
use zyntax_compiler::hir::Intrinsic;

// Get size of type at runtime
let size = builder.size_of_type(builder.i32_type());

// Get alignment of type
let align = builder.align_of_type(builder.i32_type());

// Call intrinsic directly
builder.call_intrinsic(Intrinsic::MemCpy, vec![dest, src, len]);

// Panic/abort
builder.panic();
```

### Function References

```rust
// Get a reference to a function for indirect calls
let func_ref = builder.function_ref(target_func_id);
let result = builder.call(func_ref, vec![arg1, arg2]);
```

## Union/Enum Operations

### Creating Union Values

```rust
// Create Option::Some(42)
let value = builder.const_i32(42);
let some_42 = builder.create_union(
    1,  // variant index for Some
    value,
    option_type.clone(),
);
```

### Inspecting Unions

```rust
// Get the discriminant (which variant is active)
let discriminant = builder.extract_discriminant(union_value);

// Extract value from a variant
let inner_value = builder.extract_union_value(
    union_value,
    1,  // variant index
    builder.i32_type(),  // expected type
);
```

## Complete Example: Fibonacci

```rust
use zyntax_compiler::hir_builder::HirBuilder;
use zyntax_compiler::hir::{BinaryOp, HirConstant};
use zyntax_typed_ast::arena::AstArena;

fn build_fibonacci() -> zyntax_compiler::hir::HirModule {
    let mut arena = AstArena::new();
    let mut builder = HirBuilder::new("fibonacci", &mut arena);

    let i32_ty = builder.i32_type();

    // fn fib(n: i32) -> i32
    let func_id = builder.begin_function("fib")
        .param("n", i32_ty.clone())
        .returns(i32_ty.clone())
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let base_case = builder.create_block("base_case");
    let recursive = builder.create_block("recursive");

    // Entry: check if n <= 1
    builder.set_insert_point(entry);
    let n = builder.get_param(0);
    let one = builder.const_i32(1);
    let is_base = builder.icmp(BinaryOp::Le, n, one, i32_ty.clone());
    builder.cond_br(is_base, base_case, recursive);

    // Base case: return n
    builder.set_insert_point(base_case);
    builder.ret(n);

    // Recursive case: return fib(n-1) + fib(n-2)
    builder.set_insert_point(recursive);

    // fib(n - 1)
    let n_minus_1 = builder.sub(n, one, i32_ty.clone());
    let fib_ref = builder.function_ref(func_id);
    let fib_n1 = builder.call(fib_ref, vec![n_minus_1]).unwrap();

    // fib(n - 2)
    let two = builder.const_i32(2);
    let n_minus_2 = builder.sub(n, two, i32_ty.clone());
    let fib_ref2 = builder.function_ref(func_id);
    let fib_n2 = builder.call(fib_ref2, vec![n_minus_2]).unwrap();

    // return fib(n-1) + fib(n-2)
    let result = builder.add(fib_n1, fib_n2, i32_ty);
    builder.ret(result);

    builder.finish()
}
```

## Complete Example: Linked List

```rust
fn build_linked_list_sum() -> HirModule {
    let mut arena = AstArena::new();
    let mut builder = HirBuilder::new("list_ops", &mut arena);

    // Define Node type: { value: i32, next: *Node }
    let i32_ty = builder.i32_type();
    let node_ty = builder.struct_type(Some("Node"), vec![
        i32_ty.clone(),                               // value
        builder.ptr_type(builder.void_type()),        // next (opaque ptr)
    ]);
    let ptr_node = builder.ptr_type(node_ty.clone());

    // fn sum_list(head: *Node) -> i32
    let func_id = builder.begin_function("sum_list")
        .param("head", ptr_node.clone())
        .returns(i32_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    let loop_block = builder.create_block("loop");
    let done_block = builder.create_block("done");

    // Entry: initialize sum = 0, current = head
    builder.set_insert_point(entry);
    let initial_sum = builder.const_i32(0);
    let head = builder.get_param(0);
    builder.br(loop_block);

    // Loop block with PHI nodes
    builder.set_insert_point(loop_block);
    let sum_phi = builder.add_phi(i32_ty.clone(), vec![
        (initial_sum, entry),
        // Will add loop edge later
    ]);
    let current_phi = builder.add_phi(ptr_node.clone(), vec![
        (head, entry),
        // Will add loop edge later
    ]);

    // Check if current is null
    let null_ptr = builder.null_ptr(node_ty.clone());
    let is_null = builder.icmp_eq(current_phi, null_ptr, ptr_node.clone());
    builder.cond_br(is_null, done_block, loop_block);  // Continue in loop

    // Note: In real code, you'd have a loop body block
    // This is simplified for the example

    // Done: return sum
    builder.set_insert_point(done_block);
    builder.ret(sum_phi);

    builder.finish()
}
```

## Utility Methods

### String Interning

```rust
// Intern a string for use as identifier
let name = builder.intern("my_identifier");

// Check if function exists
if builder.has_function(name) {
    let func_id = builder.get_function_by_name(name);
}
```

### Getting Current Context

```rust
// Get current block ID
let block_id = builder.current_block_id();

// Get entry block of current function
let entry = builder.entry_block();
```

## Integration with Backends

The HIR module produced by `HirBuilder` can be directly compiled:

```rust
use zyntax_compiler::cranelift_backend::CraneliftBackend;

let hir_module = builder.finish();

// JIT compilation with Cranelift
let mut backend = CraneliftBackend::new()?;
backend.compile_module(&hir_module)?;

// Execute
let fn_ptr = backend.get_function_pointer(func_id)?;
let result = unsafe {
    let f: extern "C" fn(i32) -> i32 = std::mem::transmute(fn_ptr);
    f(10)
};
```

Or with LLVM for AOT:

```rust
use inkwell::context::Context;
use zyntax_compiler::llvm_backend::LLVMBackend;

let context = Context::create();
let mut backend = LLVMBackend::new(&context, "my_module");
let ir = backend.compile_module(&hir_module)?;

// Write LLVM IR to file
std::fs::write("output.ll", &ir)?;
```

## Next Steps

- [Chapter 10](./10-packaging-distribution.md): Package and distribute compiled modules
- [Chapter 9](./09-reference.md): Complete API reference
