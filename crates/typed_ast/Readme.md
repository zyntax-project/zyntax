# TypedAST - Multi-Language Type System

**Version**: 1.0.0
**Status**: Production-Ready

## Overview

The TypedAST crate provides a language-agnostic typed abstract syntax tree and comprehensive type system infrastructure for building programming language compilers. It serves as the common intermediate representation layer that multiple language frontends can target.

## Key Features

### Multi-Paradigm Type System
- **Nominal Types**: Classes, interfaces, structs with explicit names
- **Structural Types**: Duck typing, shape-based compatibility
- **Gradual Types**: Optional static typing with dynamic fallback
- **Linear Types**: Resource management and uniqueness tracking
- **Effect Types**: Computational effect tracking

### Advanced Type Features
- **Generics**: Type parameters with variance and bounds
- **Const Generics**: Fixed-size arrays and const type parameters, evaluated at compile time
- **Traits/Interfaces**: Abstract behavior definitions with associated types
- **Lifetimes**: Borrow checking and ownership analysis
- **Type Inference**: Hindley-Milner with extensions
- **Constraint Solving**: Unification and subtyping
- **Pattern Matching**: Exhaustiveness checking and refinement

### Language Interoperability
Supports type systems from multiple languages:
- **Rust**: Ownership, lifetimes, traits, generics
- **Java/C#**: Classes, interfaces, generics, nullable types
- **TypeScript**: Structural typing, union types, intersection types
- **Swift**: Protocols, associated types, value types

## Architecture

```
┌──────────────────────────────────────────────────┐
│         Language Frontends                       │
│    (ZynML, Python, Lua, etc.)                    │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│              TypedAST Layer                      │
│  ┌────────────────────────────────────────────┐ │
│  │ Type Registry                              │ │
│  │  • Type definitions and lookups            │ │
│  │  • Generic instantiation                   │ │
│  │  • Trait/interface resolution              │ │
│  └────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────┐ │
│  │ Type Checker                               │ │
│  │  • Hindley-Milner type inference           │ │
│  │  • Constraint generation and solving       │ │
│  │  • Ownership and lifetime analysis         │ │
│  └────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────┐ │
│  │ TypedAST Representation                    │ │
│  │  • Typed expressions and statements        │ │
│  │  • Function and class declarations         │ │
│  │  • Pattern matching constructs             │ │
│  └────────────────────────────────────────────┘ │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│              HIR Lowering                        │
│         (Compiler Backend)                       │
└──────────────────────────────────────────────────┘
```

## Core Components

### Type Registry (`type_registry.rs`)
Central repository for all type definitions. Manages:
- Type creation and lookup
- Generic type instantiation
- Trait and interface definitions
- Type equivalence and subtyping

### Type Checker (`type_checker.rs`)
Performs type inference and validation:
- Expression type inference
- Pattern exhaustiveness checking
- Trait implementation verification
- Generic constraint satisfaction

### Constraint Solver (`constraint_solver.rs`)
Unification-based type inference:
- Constraint generation from expressions
- Unification algorithm for type equations
- Subtyping constraint solving
- Error reporting with type mismatch details

### Diagnostics (`diagnostics.rs`)
Rich error reporting:
- Source span tracking
- Multi-level diagnostics (error/warning/info)
- Suggestions for fixes
- Integration with language servers

## Type System Overview

### Type Hierarchy

```
Type
├── Primitive (i32, f64, bool, string)
├── Struct (nominal or structural)
├── Enum (tagged unions)
├── Function (with parameter and return types)
├── Generic (type parameters)
├── Trait (abstract interfaces)
├── Tuple (product types)
├── Array (fixed or dynamic size)
├── Reference (borrowed references with lifetimes)
├── Pointer (raw or smart pointers)
├── Union (sum types)
├── Intersection (combined types)
├── Const (const generic parameters and const-constrained types)
└── Effect (computational effects)
```

### Ownership and Lifetimes

The type system tracks ownership through:
- **Owned**: Unique ownership (move semantics)
- **Borrowed**: Shared (`&T`) or mutable (`&mut T`) references
- **Reference Counted**: `Rc<T>`, `Arc<T>` for shared ownership
- **Lifetime Parameters**: Explicit lifetime annotations

Lifetime checking ensures:
- No use-after-free
- No data races
- Proper resource cleanup

## Usage Example

```rust
use zyntax_typed_ast::{TypeRegistry, TypeChecker, Type};

// Create type registry
let mut registry = TypeRegistry::new();

// Define types
let i32_type = registry.register_primitive(PrimitiveType::I32);
let string_type = registry.register_primitive(PrimitiveType::String);

// Define a function type: fn(i32, i32) -> i32
let add_fn_type = registry.register_function_type(
    vec![i32_type.clone(), i32_type.clone()],
    i32_type.clone()
);

// Type check expressions
let mut checker = TypeChecker::new(&registry);
let expr_type = checker.check_expression(&some_expression)?;
```

## Integration with Compiler

TypedAST serves as the input to the compiler's HIR lowering phase:

1. **Frontend** parses source code
2. **TypedAST** performs type checking and inference
3. **HIR Lowering** converts TypedAST to high-level IR
4. **Backend** generates machine code

## Testing

Run tests:
```bash
cargo test --package zyntax_typed_ast
```

## Documentation

**TypedAST Construction:**
- [docs/TYPED_BUILDER_EXAMPLE.md](docs/TYPED_BUILDER_EXAMPLE.md) - TypedAST Builder guide and examples

**Type System Design:**
- [docs/type_system_design.md](docs/type_system_design.md) - In-depth type system architecture

**Platform Architecture:**
- [../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) - Overall Zyntax architecture

## License

Part of the Zyntax compiler infrastructure project.
