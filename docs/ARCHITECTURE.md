# Zyntax Compiler Architecture

## Overview

Zyntax is a multi-layer compiler infrastructure designed to support multiple programming languages through a common typed intermediate representation. The architecture follows a clear pipeline from high-level typed AST through optimization layers down to native machine code.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Language Frontends                          │
│         (Rust, Haxe, TypeScript, Java, C#, etc.)               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: TypedAST                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Type Registry                                             │  │
│  │  • Type definitions and lookups                          │  │
│  │  • Generic instantiation                                 │  │
│  │  • Trait/interface resolution                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Type Checker & Inference                                  │  │
│  │  • Hindley-Milner type inference                         │  │
│  │  • Constraint solving and unification                    │  │
│  │  • Ownership and lifetime analysis                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Multi-Paradigm Checkers                                   │  │
│  │  • Nominal type checker (OOP)                            │  │
│  │  • Structural type checker (duck typing)                 │  │
│  │  • Gradual type checker (optional typing)                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 2: HIR (High-Level IR)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ HIR Lowering                                              │  │
│  │  • TypedAST → HIR translation                            │  │
│  │  • Generic monomorphization                              │  │
│  │  • Trait dispatch resolution                             │  │
│  │  • Pattern matching compilation                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ SSA & Control Flow                                        │  │
│  │  • SSA construction with phi nodes                       │  │
│  │  • Control flow graph (CFG)                              │  │
│  │  • Dominance analysis                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Optimization Passes                                       │  │
│  │  • Dead code elimination                                 │  │
│  │  • Constant folding & propagation                        │  │
│  │  • Inlining decisions                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌──────────────────────┐  ┌──────────────────────┐
│  LAYER 3: Backends   │  │  LAYER 3: Backends   │
│                      │  │                      │
│  Cranelift JIT       │  │  LLVM JIT/AOT        │
│  • Fast compilation  │  │  • Optimized code    │
│  • Baseline tier     │  │  • Hot-path tier     │
│  • <1ms startup      │  │  • Full opts         │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           └────────┬────────────────┘
                    ▼
         ┌────────────────────┐
         │  Native Machine    │
         │       Code         │
         └────────────────────┘
```

## Project Structure

```
zyntax/
├── crates/
│   ├── typed_ast/          # Multi-language typed AST + type system
│   └── compiler/           # HIR, backends, tiered JIT, async runtime
└── docs/                   # Architecture documentation
```

---

## Layer 1: TypedAST - Language-Agnostic Type System

**Location**: `crates/typed_ast/`

### Purpose
Provides a unified typed intermediate representation that multiple language frontends can target. Handles all type system complexity including generics, traits, lifetimes, and multi-paradigm typing.

### Key Components

#### Type Registry
Central repository for type definitions:
- Primitive types (i32, f64, bool, string)
- User-defined types (structs, classes, enums)
- Generic types with type parameters
- Trait/interface definitions
- Type aliases and newtypes

**Example type representation**:
```rust
// Conceptual structure (simplified)
Type::Function {
    params: vec![i32, i32],
    returns: i32
}
```

#### Type Checker
Multi-paradigm type inference and validation:
- **Hindley-Milner inference**: Automatic type deduction
- **Constraint solving**: Unification-based type resolution
- **Ownership analysis**: Rust-style borrow checking
- **Lifetime tracking**: Memory safety guarantees

Supports three typing paradigms:
1. **Nominal typing**: OOP-style (Java, C#, Rust)
2. **Structural typing**: Shape-based (TypeScript, Go)
3. **Gradual typing**: Optional static types (Python, JavaScript)

#### TypedAST Representation
Well-typed abstract syntax tree with:
- Function declarations with typed parameters
- Class/struct definitions with fields and methods
- Expression trees with type annotations
- Pattern matching constructs
- Async/await support

**Data Flow**:
```
Language Source → Parser → TypedAST → Type Checker → Validated TypedAST
```

### Multi-Language Support

The TypedAST layer abstracts language-specific features:

| Language | Type System | Ownership | Async |
|----------|------------|-----------|-------|
| Rust | Nominal + Traits | Ownership + Lifetimes | async/await |
| Haxe | Nominal + Structural | GC | No |
| TypeScript | Structural | GC | async/await |
| Java/C# | Nominal + Interfaces | GC | async/await |

---

## Layer 2: HIR - High-Level Intermediate Representation

**Location**: `crates/compiler/src/hir.rs`, `hir_builder.rs`

### Purpose
Platform-independent SSA-based IR that serves as the optimization and code generation layer. Type-erased for efficient compilation.

### HIR Structure

#### Instructions
HIR uses a small set of typed instructions:
- **Arithmetic**: Add, Sub, Mul, Div
- **Comparison**: Eq, Lt, Gt, etc.
- **Memory**: Load, Store, Alloca
- **Control**: Branch, CondBranch, Return, Call
- **Aggregate**: ExtractValue, InsertValue, StructConstruct

**Example HIR (simplified)**:
```
function add(a: i32, b: i32) -> i32 {
  block0:
    %result = add %a, %b : i32
    return %result
}
```

#### Control Flow Graph (CFG)
- Basic blocks with predecessors/successors
- Dominance tree for optimization
- Loop detection and analysis

#### SSA Form
Static Single Assignment with:
- Phi nodes for merging values
- Unique value definitions
- Use-def chains for analysis

### HIR Lowering Pipeline

```
TypedAST
   ↓
Generic Monomorphization (expand generics to concrete types)
   ↓
Trait Dispatch Resolution (virtual calls → direct/vtable)
   ↓
Pattern Compilation (patterns → branching logic)
   ↓
SSA Construction (convert to SSA form)
   ↓
Optimization Passes
   ↓
HIR (ready for backend)
```

### Optimization Passes

**Dead Code Elimination**:
- Remove unreachable blocks
- Eliminate unused values
- Prune redundant computations

**Constant Folding**:
- Evaluate constant expressions at compile time
- Propagate known values

**Inlining**:
- Inline small functions
- Profile-guided inlining decisions

---

## Layer 3: Code Generation Backends

**Location**: `crates/compiler/src/cranelift_backend.rs`, `llvm_backend.rs`

### Tiered JIT Compilation

Zyntax uses a **3-tier compilation strategy** for optimal performance:

```
┌─────────────────────────────────────────────────────┐
│  Tier 1: Baseline JIT (Cranelift)                  │
│  • Fast compilation (<1ms)                          │
│  • Minimal optimizations                            │
│  • Immediate execution                              │
│  Trigger: First execution                           │
└────────────────┬────────────────────────────────────┘
                 │ (after 50-100 executions)
                 ▼
┌─────────────────────────────────────────────────────┐
│  Tier 2: Optimized JIT (Cranelift)                 │
│  • Moderate compilation (~10ms)                     │
│  • Standard optimizations                           │
│  • Good performance                                 │
│  Trigger: Warm code                                 │
└────────────────┬────────────────────────────────────┘
                 │ (after 1000+ executions)
                 ▼
┌─────────────────────────────────────────────────────┐
│  Tier 3: Hot Path (LLVM)                           │
│  • Aggressive compilation (~50ms)                   │
│  • Full optimizations                               │
│  • Peak performance (~C speed)                      │
│  Trigger: Hot paths                                 │
└─────────────────────────────────────────────────────┘
```

### Runtime Profiling

**Execution Counters**:
- Atomic counters per function
- Lightweight overhead (~5ns)
- Trigger recompilation at thresholds

**Hot-Path Detection**:
- Identify frequently executed code
- Prioritize optimization effort
- Adaptive threshold adjustment

### Cranelift Backend

**Characteristics**:
- Fast compilation (security-focused)
- Sandboxed execution
- Good baseline performance
- Used for Tier 1 & 2

**Translation**: HIR → Cranelift IR → Machine Code

### LLVM Backend

**Characteristics**:
- Aggressive optimizations
- Peak performance
- Longer compilation time
- Used for Tier 3 hot paths

**Features**:
- Vectorization (SIMD)
- Loop optimizations
- Inlining and devirtualization
- Profile-guided optimization

---

## Async/Await Runtime

**Location**: `crates/compiler/src/` (integrated with compiler)

### State Machine Generation

Async functions compile to state machines:

```
async fn example() {
    let x = await foo();  // suspension point 1
    let y = await bar();  // suspension point 2
    return x + y;
}

  ↓ (compiled to)

State Machine:
  State 0: Initial
  State 1: After foo(), waiting for bar()
  State 2: Complete
```

### Executor Architecture

```
┌─────────────────────────────────────────┐
│         Task Spawning                   │
│  spawn(future) → Task                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│       Work-Stealing Scheduler           │
│  • Multi-threaded task queue            │
│  • Fair scheduling                      │
│  • Load balancing                       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│          Waker System                   │
│  • Event-driven wakeup                  │
│  • Efficient task resumption            │
└─────────────────────────────────────────┘
```

**Key Features**:
- Zero-cost futures (compile to state machines)
- Parameter capture in async functions
- Integration with tiered JIT

---

## Standard Library

**Location**: `crates/compiler/src/stdlib.rs`

### Core Collections

Implemented as HIR modules compiled with the program:

**Vec\<T>**: Dynamic array
- Methods: push, pop, get, len, capacity
- Automatic growth and reallocation

**String**: UTF-8 string
- Methods: concat, substring, len, chars
- Efficient string manipulation

**HashMap\<K, V>**: Hash table
- Methods: insert, get, remove, contains_key
- Open addressing with linear probing

**Iterator**: Lazy iteration
- 50+ adapter methods (map, filter, fold, etc.)
- Zero-cost abstractions

**Current Status**: 93/100 stdlib functions compile successfully

---

## Type System Features

### Generics
Type parameters with bounds:
```rust
fn identity<T>(x: T) -> T { x }
fn print<T: Display>(x: T) { ... }
```

### Traits/Interfaces
Abstract behavior definitions:
```rust
trait Drawable {
    fn draw(&self);
}
```

### Lifetimes
Memory safety through borrow checking:
```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str
```

### Dependent Types
Value-dependent types (basic):
```rust
Array<T, N: usize>  // size in type
```

---

## Memory Management

### Ownership Model

**Owned Values**:
- Unique ownership (one owner)
- Move semantics
- Automatic cleanup (RAII)

**Borrowed References**:
- Shared (`&T`) or mutable (`&mut T`)
- Lifetime-tracked
- No aliasing with mutation

**Reference Counting**:
- Rc\<T> for shared ownership
- Arc\<T> for thread-safe sharing
- Automatic reference counting

### Lifetime Analysis

**Borrow Checker**:
1. Track all borrows and their lifetimes
2. Ensure no use-after-free
3. Prevent data races
4. Compile-time guarantees

---

## Integration Points

### Adding a New Language Frontend

1. **Parse** source to AST
2. **Build** TypedAST using TypedASTBuilder
3. **Register** types in TypeRegistry
4. **Type check** with appropriate checker
5. **Lower** to HIR
6. **Compile** with backends

**Example flow**:
```
YourLang Source
    ↓ (Your Parser)
YourLang AST
    ↓ (TypedASTBuilder)
TypedAST
    ↓ (TypeChecker)
Validated TypedAST
    ↓ (HIR Lowering)
HIR
    ↓ (Backends)
Native Code
```

### Extending the Type System

Add new type variants in `type_registry.rs`:
1. Define new `Type` enum variant
2. Implement type checking rules
3. Add HIR lowering support
4. Update backends if needed

---

## Performance Characteristics

### Compilation Speed

| Tier | Backend | Time | Use Case |
|------|---------|------|----------|
| Tier 1 | Cranelift | <1ms | Baseline, fast startup |
| Tier 2 | Cranelift | ~10ms | Warm code |
| Tier 3 | LLVM | ~50ms | Hot paths only |

### Runtime Performance

| Feature | Overhead |
|---------|----------|
| Zero-cost abstractions | 0% |
| Async task spawn | ~50ns |
| Await point | ~100ns |
| Ownership checks | Compile-time only |
| Generic dispatch | Monomorphized (0%) |

### Memory Safety

All memory safety enforced at **compile time**:
- No runtime GC pauses
- No reference counting overhead (except explicit Rc/Arc)
- Predictable performance

---

## Testing & Validation

**Current Status**: 280/284 tests passing (98.6%)

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end compilation
3. **Comprehensive Tests**: Full language features
4. **Standard Library Tests**: Collection implementations

### Test Infrastructure

```
crates/compiler/tests/
├── end_to_end_simple.rs       # 5/5 passing
├── end_to_end_comprehensive.rs # 9/9 passing
├── cranelift_backend_tests.rs
├── hir_builder_tests.rs
└── stdlib_tests.rs
```

---

## Future Roadmap

See [../BACKLOG.md](../BACKLOG.md) for detailed roadmap.

### Priority #1: Reflaxe/Haxe Integration
Create `reflaxe.Zyntax` backend to tap into Haxe's mature ecosystem, providing instant access to thousands of production libraries.

### Planned Features
- Exception handling (try/catch/finally)
- LLVM AOT backend completion
- I/O and networking stdlib
- Language Server Protocol (LSP)
- Package manager

---

## Documentation Index

**Architecture**:
- This document - System architecture overview

**Component Details**:
- [../crates/typed_ast/README.md](../crates/typed_ast/README.md) - Type system details
- [../crates/typed_ast/docs/TYPED_BUILDER_EXAMPLE.md](../crates/typed_ast/docs/TYPED_BUILDER_EXAMPLE.md) - TypedAST Builder guide
- [../crates/typed_ast/docs/type_system_design.md](../crates/typed_ast/docs/type_system_design.md) - Type system design
- [../crates/compiler/README.md](../crates/compiler/README.md) - Compiler overview

**Implementation Guides**:
- [HIR_BUILDER_EXAMPLE.md](HIR_BUILDER_EXAMPLE.md) - HIR construction
- [ASYNC_RUNTIME_DESIGN.md](ASYNC_RUNTIME_DESIGN.md) - Async runtime internals
- [tiered-compilation.md](tiered-compilation.md) - Tiered JIT design
- [BYTECODE_FORMAT_SPEC.md](BYTECODE_FORMAT_SPEC.md) - Bytecode specification

**Project Info**:
- [../README.md](../README.md) - Project overview
- [../BACKLOG.md](../BACKLOG.md) - Development roadmap
- [../PRODUCTION_READY_STATUS.md](../PRODUCTION_READY_STATUS.md) - Feature matrix

---

**Built with Rust** | **Powered by Cranelift & LLVM** | **Production-Ready**
