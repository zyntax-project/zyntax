# ZynParser - Phase 1 POC

> **Parser generator for multi-language Zyntax frontends**

## Overview

ZynParser is a proof-of-concept demonstrating how PEG grammars (via pest) can be used to generate TypedAST for the Zyntax compiler infrastructure. This Phase 1 POC validates the approach before implementing the full .zyn grammar format in later phases.

## Current Status: Phase 2 (Zig) - 95% Complete ✅

### Phase 1 (Calculator POC) - Complete ✅
- **PEG Grammar**: Calculator grammar (calculator.pest) for arithmetic expressions
- **TypedAST Generation**: Parse trees automatically convert to Zyntax TypedAST
- **Type Inference**: Simple type inference for binary operations

### Phase 2 (Zig Subset) - 95% Complete ✅
- **Full Zig Grammar**: Functions, structs, control flow, types
- **Array Support**: Array literals `[_]T{...}`, sized arrays `[N]T`, indexing
- **String Literals**: `"Hello, World!"` lowered to global `*i8` constants
- **Optional Types**: `?T` syntax parses and builds to TypedAST
- **Error Unions**: `!T` syntax parses and builds to TypedAST Result type
- **Pattern Matching**: `if let` syntax parses and builds to Match statements
- **Control Flow**: if/else, while loops, for loops, break, continue
- **Logical Operators**: and/or with short-circuit evaluation
- **E2E Pipeline**: Zig → TypedAST → HIR → Cranelift JIT → Execution

**Known Limitations**:
- Pattern matching execution requires pattern matching backend (tracked in BACKLOG.md Issue #0 Phase 3)
- `if let` syntax validates and builds to TypedAST Match statements, but execution needs discriminant lowering
- Optional/Result type operations need stdlib integration via plugin system

### Features Demonstrated

1. **Parsing**: Source → pest Parse Tree (using stock pest)
2. **AST Building**: Parse Tree → TypedAST (manual conversion)
3. **Type System**: Type inference and checking
4. **Error Handling**: ParseError → BuildError propagation

## Architecture

```
Source Code ("2 + 3")
    ↓
[pest Parser]
    ↓
Parse Tree (Pairs<Rule>)
    ↓
[TypedAstBuilder]
    ↓
TypedAST (TypedNode<TypedExpression>)
    ↓
[Future: HIR Lowering]
    ↓
HIR → Cranelift JIT → Execution
```

## Usage

```rust
use zyn_parser::{CalculatorParser, TypedAstBuilder};
use pest::Parser;

// Parse source code
let source = "2 + 3";
let pairs = CalculatorParser::parse(Rule::program, source)?;

// Build TypedAST
let mut builder = TypedAstBuilder::new();
let typed_expr = builder.build_program(pairs)?;

// typed_expr is now a TypedNode<TypedExpression>
// Ready for HIR lowering and compilation
```

## Grammar Features

**Supported**:
- Integer and float literals
- Binary operators: `+`, `-`, `*`, `/`, `%`
- Unary operators: `-`, `!`
- Parenthesized expressions
- Operator precedence (multiplication before addition)

**Example Programs**:
```
42                  // Integer literal
3.14                // Float literal
2 + 3               // Addition
(2 + 3) * 4         // Parentheses and precedence
-42                 // Unary negation
1 + 2 * 3           // Precedence: 1 + (2 * 3)
```

## Testing

```bash
# Run all tests
cargo test --package zyn_parser

# Run unit tests
cargo test --package zyn_parser --lib

# Run integration tests
cargo test --package zyn_parser --test end_to_end
```

## Next Steps

### Phase 2: Full Language Grammar (Zig Subset)

Target: Statically-typed systems language (Zig recommended)

**Why Zig?**
- Explicit, static type system (aligns with Zyntax TypedAST)
- No hidden memory allocations (clear ownership semantics)
- Comptime evaluation (maps to type-level computation)
- Error unions (explicit error handling)
- Simple, consistent syntax

**Grammar Scope**:
- Functions with explicit types
- Struct definitions
- Control flow (if, while, for, switch)
- Primitive types (i32, f64, bool)
- Pointers and slices
- Error unions (Result types)
- Basic generics

**Timeline**: 4-8 weeks

### Phase 3: Fork pest → ZynPEG

Goal: Native `.zyn` grammar format with TypedAST action blocks

**New Syntax**:
```zyn
@imports {
    use zyntax_typed_ast::*;
}

@context {
    arena: &mut AstArena,
    type_registry: &mut TypeRegistry,
}

binary_op = { expr ~ "+" ~ expr }
  -> TypedExpression {
      expr: BinaryOp(Add, $1, $3),
      ty: infer_type($1.ty, Add, $3.ty),
      span: span($1, $3)
  }
```

**Timeline**: 8-12 weeks

### Phase 4: Multi-Language Ecosystem

Target languages (statically-typed):
- Zig (Phase 2)
- Swift
- Kotlin
- Go
- Rust subset

**Timeline**: Ongoing (6+ months)

## Implementation Pattern

This POC demonstrates the pattern that will be automated in Phase 3:

### Current (Manual)
1. Write `.pest` grammar file
2. Manually implement `TypedAstBuilder`
3. Map each Rule to TypedAST construction
4. Handle type inference manually

### Future (Automated .zyn)
1. Write `.zyn` grammar with action blocks
2. ZynPEG generates Rust parser + TypedAST builder
3. Type inference helpers built into grammar
4. Rich error diagnostics automatically generated

## Project Structure

```
crates/zyn_parser/
├── Cargo.toml
├── README.md (this file)
├── src/
│   ├── lib.rs                  # Parser definition
│   ├── calculator.pest         # PEG grammar
│   └── typed_ast_builder.rs    # TypedAST construction
└── tests/
    └── end_to_end.rs           # Integration tests
```

## Contributing

See [ZYN_PARSER_IMPLEMENTATION.md](../../docs/ZYN_PARSER_IMPLEMENTATION.md) for the full implementation plan.

**Good First Issues**:
- Add more arithmetic operators (**, //, bit shifts)
- Add comparison operators (<, >, ==, !=)
- Add boolean operators (&&, ||)
- Improve error messages with spans
- Add float literal support (currently converts to int)

## License

Apache 2.0 - See [LICENSE](../../LICENSE)

---

**Status**: Phase 1 POC Complete ✅
**Next Milestone**: Phase 2 - Zig Grammar (Q1 2025)
