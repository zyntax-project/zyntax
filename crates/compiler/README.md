# Compiler - HIR and Code Generation

**Version**: 1.0.0
**Status**: Production-Ready

## Overview

The compiler crate handles the transformation of TypedAST into executable native code through a multi-tier compilation pipeline. It provides High-Level Intermediate Representation (HIR), optimizations, and multiple backend targets.

## Architecture

```
TypedAST → HIR Lowering → Optimization → Backend → Native Code
```

### Components

1. **HIR (High-Level IR)**
   - SSA-based intermediate representation
   - Platform-agnostic
   - Type-erased for efficient compilation
   - Control flow graph with dominance analysis

2. **HIR Builder API**
   - Fluent interface for programmatic IR construction
   - Type-safe SSA value management
   - Block and function builders

3. **Tiered JIT Compilation**
   - Tier 1: Cranelift baseline (fast compilation, <1ms)
   - Tier 2: Cranelift optimized (moderate opts, ~10ms)
   - Tier 3: LLVM JIT (aggressive opts, ~50ms)
   - Runtime profiling and hot-path detection

4. **Backends**
   - **Cranelift JIT**: Fast baseline compilation
   - **LLVM JIT**: Optimized hot-path compilation
   - **LLVM AOT**: Ahead-of-time native binaries (in progress)

## Key Features

- ✅ Complete HIR instruction set
- ✅ SSA construction with phi nodes
- ✅ CFG analysis (dominators, post-dominators)
- ✅ Dead code elimination
- ✅ Tiered compilation with profiling
- ✅ Function calls and recursion
- ✅ Local variables and stack allocation
- ✅ Arithmetic and comparison operations

## Testing

```bash
# Run all compiler tests
cargo test --package zyntax_compiler

# Run end-to-end tests
cargo test --test end_to_end_comprehensive
cargo test --test end_to_end_simple
```

## Usage Example

```rust
use zyntax_compiler::{HirBuilder, CraneliftBackend};
use zyntax_typed_ast::arena::AstArena;

let mut arena = AstArena::new();
let mut builder = HirBuilder::new("example", &mut arena);

// Build a simple function: fn add(a: i32, b: i32) -> i32
let i32_ty = builder.i32_type();
let func = builder.begin_function("add")
    .param("a", i32_ty.clone())
    .param("b", i32_ty.clone())
    .returns(i32_ty.clone())
    .build();

builder.set_current_function(func);
let entry = builder.entry_block();
builder.set_insert_point(entry);

let a = builder.get_param(0);
let b = builder.get_param(1);
let result = builder.add(a, b, i32_ty);
builder.ret(result);

let module = builder.finish();

// Compile and execute
let mut backend = CraneliftBackend::new().unwrap();
backend.compile_module(&module).unwrap();

let fn_ptr = backend.get_function_ptr(func).unwrap();
let f: fn(i32, i32) -> i32 = unsafe { std::mem::transmute(fn_ptr) };
assert_eq!(f(2, 3), 5);
```

## Documentation

- [../../docs/HIR_BUILDER_EXAMPLE.md](../../docs/HIR_BUILDER_EXAMPLE.md) - HIR Builder guide
- [../../docs/tiered-compilation.md](../../docs/tiered-compilation.md) - Tiered JIT design
- [../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) - Overall architecture

## License

Part of the Zyntax compiler infrastructure project.
