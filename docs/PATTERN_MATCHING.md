# Pattern Matching Implementation

## Overview

This document describes the complete pattern matching implementation in the Zyntax compiler, including discriminant-based conditional branching and value extraction for union types (Optional and Result).

## Table of Contents

- [Architecture](#architecture)
- [Implementation Layers](#implementation-layers)
- [HIR Instructions](#hir-instructions)
- [Data Flow](#data-flow)
- [Examples](#examples)
- [Testing](#testing)
- [Future Work](#future-work)

## Architecture

Pattern matching in Zyntax is implemented through a three-layer architecture:

```
Source Code (TypedAST)
    ↓
CFG Layer (Pattern Metadata)
    ↓
SSA Layer (Code Generation)
    ↓
HIR Instructions
    ↓
Cranelift/LLVM Backend
```

### Design Principles

1. **Separation of Concerns**: CFG handles structure, SSA handles code generation
2. **Metadata Flow**: Pattern information flows from CFG to SSA through block metadata
3. **Type Safety**: All discriminant indices computed from type system
4. **Extensibility**: Easy to add new pattern types and union variants

## Implementation Layers

### 1. CFG Layer (`typed_cfg.rs`)

**Purpose**: Create control flow structure and store pattern metadata

#### Key Structures

```rust
/// Pattern check information for a basic block
pub struct PatternCheckInfo {
    /// The scrutinee expression being matched
    pub scrutinee: TypedNode<TypedExpression>,
    /// The pattern being checked in this block
    pub pattern: TypedNode<TypedPattern>,
    /// Variant index if this is a union variant check
    pub variant_index: Option<u32>,
}

/// Basic block with pattern metadata
pub struct TypedBasicBlock {
    pub id: HirId,
    pub label: Option<InternedString>,
    pub statements: Vec<TypedNode<TypedStatement>>,
    pub terminator: TypedTerminator,
    pub pattern_check: Option<PatternCheckInfo>,  // ← Pattern metadata
}
```

#### CFG Block Structure for Match

For a match statement like:
```rust
match opt_val {
    Some(x) => x + 1,
    None => 0,
}
```

CFG creates:
1. **Entry block**: Evaluates scrutinee, jumps to first pattern check
2. **Pattern check blocks**: One per arm, stores pattern metadata
   - `pattern_check: Some(PatternCheckInfo { variant_index: 1, ... })` for Some
   - `pattern_check: Some(PatternCheckInfo { variant_index: 0, ... })` for None
3. **Body blocks**: Execute arm expressions, also have pattern metadata for extraction
4. **Merge block**: Where all arms converge

#### Key Methods

```rust
/// Extract pattern check information for SSA to use
fn extract_pattern_check_info(
    &self,
    scrutinee: &TypedNode<TypedExpression>,
    pattern: &TypedNode<TypedPattern>,
) -> Option<PatternCheckInfo>
```

Maps patterns to variant indices:
- Identifies enum patterns (Some, None, Ok, Err)
- Calls `get_variant_index()` to resolve discriminant values
- Returns metadata for SSA layer

```rust
/// Get the discriminant index for a variant in a union/enum type
fn get_variant_index(&self, ty: &Type, variant_name: &InternedString) -> Option<u32>
```

Hardcoded mappings for built-in types:
- `Optional<T>`: None=0, Some=1
- `Result<T,E>`: Ok=0, Err=1

**Future**: Will query type registry for custom enums

### 2. SSA Layer (`ssa.rs`)

**Purpose**: Generate HIR instructions for pattern matching

#### Match Context

Tracks state across pattern matching blocks:

```rust
struct MatchContext {
    /// The scrutinee value being matched
    scrutinee_value: HirId,
    /// The extracted discriminant (for union types)
    discriminant_value: Option<HirId>,
    /// The union type being matched
    union_type: Option<Box<HirUnionType>>,
}
```

Stored in `SsaBuilder` and created when processing Match statements.

#### Key Methods

##### Discriminant Extraction

```rust
// In process_statement() for TypedStatement::Match
let scrutinee_val = self.translate_expression(block_id, &match_stmt.scrutinee)?;
let scrutinee_hir_type = self.convert_type(&match_stmt.scrutinee.ty);

if let HirType::Union(union_type) = &scrutinee_hir_type {
    let discriminant_id = HirId::new();

    self.add_instruction(
        block_id,
        HirInstruction::GetUnionDiscriminant {
            result: discriminant_id,
            union_val: scrutinee_val,
        },
    );

    self.match_context = Some(MatchContext {
        scrutinee_value: scrutinee_val,
        discriminant_value: Some(discriminant_id),
        union_type: Some(union_type.clone()),
    });
}
```

##### Discriminant Comparison

```rust
fn generate_pattern_discriminant_check(
    &mut self,
    block_id: HirId,
    true_target: HirId,
    variant_index: u32,
) -> CompilerResult<HirTerminator>
```

Generates:
1. Constant for variant index: `HirConstant::U32(variant_index)`
2. Comparison instruction: `Binary(Eq, discriminant, variant_const)`
3. Conditional branch: `CondBranch { condition, true_target, false_target }`

Called when processing Jump terminators with pattern metadata.

##### Value Extraction

```rust
fn extract_pattern_bindings(
    &mut self,
    block_id: HirId,
    pattern: &TypedNode<TypedPattern>,
    variant_index: u32,
) -> CompilerResult<()>
```

For patterns like `Some(x)`:
1. Finds the variant in union type by discriminant
2. Generates `ExtractUnionValue` instruction
3. Binds extracted value to pattern variable: `write_variable(name, block_id, extracted_id)`

Called at the start of match arm body blocks.

## HIR Instructions

### GetUnionDiscriminant

Extracts the discriminant tag from a union value.

```rust
HirInstruction::GetUnionDiscriminant {
    result: HirId,      // Where to store discriminant (u32)
    union_val: HirId,   // The union value to inspect
}
```

**Example**:
```rust
let opt: Option<i32> = Some(42);
let discriminant = GetUnionDiscriminant(opt);  // Returns 1 (Some)
```

### ExtractUnionValue

Extracts the inner value from a union variant.

```rust
HirInstruction::ExtractUnionValue {
    result: HirId,        // Where to store extracted value
    union_val: HirId,     // The union value
    variant_index: u32,   // Which variant to extract from
    ty: HirType,          // Type of the inner value
}
```

**Example**:
```rust
let opt: Option<i32> = Some(42);
let x = ExtractUnionValue(opt, variant_index=1, ty=i32);  // Returns 42
```

### Binary (Eq) for Comparison

Standard equality comparison for discriminant checks.

```rust
HirInstruction::Binary {
    result: HirId,
    op: BinaryOp::Eq,
    left: discriminant_id,   // Discriminant from GetUnionDiscriminant
    right: const_id,         // Constant for variant index
    ty: HirType::Bool,
}
```

### CondBranch

Conditional branch based on pattern match result.

```rust
HirTerminator::CondBranch {
    condition: comparison_result,  // Result of discriminant == variant_index
    true_target: body_block,       // Execute this arm
    false_target: next_pattern,    // Try next pattern
}
```

## Data Flow

### Complete Example: `if let Some(x) = opt`

#### 1. TypedAST Representation

```rust
TypedStatement::Match(TypedMatch {
    scrutinee: opt_val,  // Type: Optional<i32>
    arms: [
        TypedMatchArm {
            pattern: TypedPattern::Enum {
                variant: "Some",
                fields: [TypedPattern::Identifier { name: "x" }]
            },
            body: /* use x */
        },
        TypedMatchArm {
            pattern: TypedPattern::Enum {
                variant: "None",
                fields: []
            },
            body: /* alternative */
        }
    ]
})
```

#### 2. CFG Structure

```
Entry Block (bb0):
  statements: [evaluate opt_val]
  terminator: Jump(bb1)
  pattern_check: None

Pattern Check Block (bb1) - Some:
  statements: []
  terminator: Jump(bb2)  ← Will be upgraded to CondBranch
  pattern_check: Some(PatternCheckInfo {
      scrutinee: opt_val,
      pattern: Some(x),
      variant_index: Some(1)
  })

Body Block (bb2) - Some arm:
  statements: [/* use x */]
  terminator: Jump(bb_merge)
  pattern_check: Some(PatternCheckInfo {
      pattern: Some(x),
      variant_index: Some(1)
  })

Pattern Check Block (bb3) - None:
  statements: []
  terminator: Jump(bb4)
  pattern_check: Some(PatternCheckInfo {
      variant_index: Some(0)
  })

Body Block (bb4) - None arm:
  statements: [/* alternative */]
  terminator: Jump(bb_merge)
  pattern_check: Some(...)

Merge Block (bb_merge):
  ...
```

#### 3. HIR Generation (SSA Layer)

**Entry Block (bb0)**:
```rust
// Translate scrutinee
%1 = Variable("opt_val")

// Extract discriminant
%2 = GetUnionDiscriminant(%1)

// Store match context
match_context = MatchContext {
    scrutinee_value: %1,
    discriminant_value: Some(%2),
    ...
}
```

**Pattern Check Block (bb1)**:
```rust
// Create constant for Some variant (index 1)
%3 = Const(U32(1))

// Compare discriminant
%4 = Binary(Eq, %2, %3)

// Conditional branch (replaces Jump)
CondBranch(%4, bb2, bb3)
```

**Body Block (bb2)**:
```rust
// Extract value for pattern variable x
%5 = ExtractUnionValue(%1, variant_index=1, ty=i32)

// Bind to variable
write_variable("x", bb2, %5)

// Execute body with x available
...
```

**Pattern Check Block (bb3)**:
```rust
// None has no inner value, just check discriminant
%6 = Const(U32(0))
%7 = Binary(Eq, %2, %6)
CondBranch(%7, bb4, bb_unreachable)
```

#### 4. Cranelift Backend

The HIR instructions map directly to Cranelift IR:

```text
block0:
    v1 = load.i64 [opt_val]
    v2 = load.i32 v1             ; Load discriminant field
    br_table v2, block2, [block1]

block1:  ; Some case
    v3 = iconst.i32 1
    v4 = icmp eq v2, v3
    brz v4, block3
    v5 = load.i32 v1+4           ; Load value field (offset by discriminant)
    ; Use v5 as x
    ...

block2:  ; None case
    ...
```

## Examples

### Example 1: Simple Optional Unwrap

**Source Code**:
```rust
fn get_value(opt: ?i32) -> i32 {
    match opt {
        Some(x) => x,
        None => 0,
    }
}
```

**Generated HIR** (simplified):
```rust
Entry:
    %opt = Parameter(0)
    %disc = GetUnionDiscriminant(%opt)

Check_Some:
    %one = Const(U32(1))
    %is_some = Binary(Eq, %disc, %one)
    CondBranch(%is_some, Body_Some, Check_None)

Body_Some:
    %x = ExtractUnionValue(%opt, 1, i32)
    Return(%x)

Check_None:
    %zero_const = Const(U32(0))
    %is_none = Binary(Eq, %disc, %zero_const)
    CondBranch(%is_none, Body_None, Unreachable)

Body_None:
    %zero = Const(I32(0))
    Return(%zero)
```

### Example 2: Result Error Handling

**Source Code**:
```rust
fn handle_result(res: !i32) -> i32 {
    match res {
        Ok(val) => val * 2,
        Err(e) => -1,
    }
}
```

**Generated HIR**:
```rust
Entry:
    %res = Parameter(0)
    %disc = GetUnionDiscriminant(%res)

Check_Ok:
    %zero = Const(U32(0))  // Ok = 0
    %is_ok = Binary(Eq, %disc, %zero)
    CondBranch(%is_ok, Body_Ok, Check_Err)

Body_Ok:
    %val = ExtractUnionValue(%res, 0, i32)
    %two = Const(I32(2))
    %result = Binary(Mul, %val, %two)
    Return(%result)

Check_Err:
    %one = Const(U32(1))  // Err = 1
    %is_err = Binary(Eq, %disc, %one)
    CondBranch(%is_err, Body_Err, Unreachable)

Body_Err:
    %err_val = ExtractUnionValue(%res, 1, ErrorType)
    // Error value available but unused in this example
    %neg_one = Const(I32(-1))
    Return(%neg_one)
```

### Example 3: if let Expression

**Source Code**:
```rust
if (let value = maybe) {
    return value + 1;
} else {
    return 0;
}
```

**Desugared to Match**:
```rust
match maybe {
    Some(value) => value + 1,
    _ => 0,
}
```

Then compiled using the standard pattern matching pipeline.

## Testing

### Unit Tests

Located in `crates/compiler/tests/pattern_matching_tests.rs`:

1. **Pattern Compilation Tests** (15 tests)
   - `test_simple_constant_pattern_match`
   - `test_union_variant_pattern_match`
   - `test_wildcard_pattern_match`
   - `test_binding_pattern`
   - `test_exhaustiveness_checking_*`
   - `test_struct_pattern_*`
   - `test_discriminant_extraction_for_optional_type`

2. **Integration Tests**
   - `test_zig_if_let_syntax` in `crates/zyn_parser/tests/zig_e2e_jit.rs`

### Test Coverage

```
✅ Pattern metadata storage in CFG
✅ Discriminant extraction
✅ Conditional branching on discriminants
✅ Value extraction with ExtractUnionValue
✅ Variable binding in match arms
✅ Exhaustiveness checking
✅ if let syntax desugaring
```

### Running Tests

```bash
# All pattern matching tests
cargo test --package zyntax_compiler --test pattern_matching_tests

# if let syntax test
cargo test --package zyn_parser --test zig_e2e_jit test_zig_if_let_syntax

# Compiler library tests
cargo test --package zyntax_compiler --lib
```

## Type Support

### Currently Supported

| Type | Discriminant | Variants | Status |
|------|--------------|----------|--------|
| `Optional<T>` | u32 | None=0, Some=1 | ✅ Complete |
| `Result<T,E>` | u32 | Ok=0, Err=1 | ✅ Complete |

### Union Type Representation

Optional and Result are represented as tagged unions in HIR:

```rust
HirType::Union(HirUnionType {
    name: None,
    variants: vec![
        HirUnionVariant {
            name: "None",
            ty: HirType::Void,
            discriminant: 0,
        },
        HirUnionVariant {
            name: "Some",
            ty: T,  // Inner type
            discriminant: 1,
        },
    ],
    discriminant_type: Box::new(HirType::U32),
    is_c_union: false,
})
```

## Performance Considerations

### Optimization Opportunities

1. **Discriminant Caching**: Discriminant is extracted once and reused across pattern checks
2. **Dead Branch Elimination**: Unreachable patterns can be removed by backend
3. **Switch Table Generation**: Multiple variant checks can be compiled to switch/br_table
4. **Inline Small Arms**: Simple match arms can be inlined

### Current Implementation

- **Linear Pattern Checking**: Patterns checked sequentially (Some, then None)
- **Branch Prediction**: Modern CPUs handle the conditional branches efficiently
- **No Redundant Extraction**: Values only extracted when pattern matches

### Future Optimizations

1. **Decision Tree Compilation**: Use decision trees for complex patterns
2. **Pattern Reordering**: Check most common patterns first
3. **Jump Tables**: For enums with many variants
4. **Pattern Specialization**: Compile-time pattern evaluation where possible

## Limitations and Future Work

### Current Limitations

1. **Pattern Types**: Enum variants support recursively nested enum patterns
   through dominance-safe decision blocks. An inner payload is extracted only
   on the matching outer-variant edge.
   - No tuple destructuring: `Some((x, y))` ❌
   - Nested enum patterns: `Some(Ok(x))` ✅
   - No struct patterns in variants: `Some(Point { x, y })` ❌

2. **Enum Resolution**: Named enums resolve through the `TypeRegistry`;
   `Optional` and `Result` also support their direct built-in type forms.
   - Need type registry integration for user-defined enums

3. **Guards**: Pattern guards partially implemented
   - Only simple boolean expressions
   - No complex guard conditions

4. **Exhaustiveness**: Basic exhaustiveness checking
   - Doesn't detect unreachable patterns
   - No non-exhaustive match warnings

### Future Enhancements

#### Phase 1: Enhanced Patterns
- [ ] Tuple destructuring: `Some((x, y))`
- [ ] Struct patterns: `Some(Point { x, y })`
- [ ] Array/slice patterns: `[first, rest @ ..]`
- [ ] Multiple bindings per variant

#### Phase 2: Custom Types
- [ ] Type registry integration for custom enums
- [ ] Variant index computation from type definitions
- [ ] Generic enum support

#### Phase 3: Advanced Features
- [ ] Nested pattern matching
- [ ] Or-patterns: `Some(1 | 2 | 3)`
- [ ] Range patterns: `1..=10`
- [ ] Complex guard expressions

#### Phase 4: Optimizations
- [ ] Decision tree compilation (already has infrastructure)
- [ ] Pattern reordering based on frequency
- [ ] Jump table generation for large enums
- [ ] Pattern specialization

## References

### Related Files

- `crates/compiler/src/typed_cfg.rs` - CFG construction with pattern metadata
- `crates/compiler/src/ssa.rs` - SSA construction and code generation
- `crates/compiler/src/pattern_matching.rs` - Pattern match compiler (decision trees)
- `crates/compiler/src/hir.rs` - HIR instruction definitions
- `crates/compiler/src/stdlib/option.rs` - Example manual HIR construction
- `crates/typed_ast/src/typed_ast.rs` - TypedPattern definitions

### Related Documentation

- `docs/ARCHITECTURE.md` - Overall compiler architecture
- `docs/HIR_BUILDER_EXAMPLE.md` - HIR construction examples
- `docs/ZYN_PARSER_IMPLEMENTATION.md` - Parser and AST building

### Commit History

Key commits for pattern matching:
- `feat: Implement Optional and Result type conversion to HIR unions`
- `feat: Add pattern checking infrastructure to TypedCFG`
- `feat: Implement discriminant-based conditional branching for pattern matching`
- `feat: Implement ExtractUnionValue for pattern variable bindings`

## Conclusion

The Zyntax pattern matching implementation provides a solid, production-ready foundation for pattern matching on union types. The architecture is clean, extensible, and well-tested. The three-layer design (CFG → SSA → HIR) ensures proper separation of concerns and makes future enhancements straightforward.

**Status**: ✅ Production Ready for Optional and Result types
