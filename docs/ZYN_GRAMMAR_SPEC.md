# ZynPEG Grammar Specification

**Version**: 3.0 (TypedAST Actions)
**Last Updated**: February 2026

## Overview

ZynPEG is a PEG parser generator with semantic actions that construct TypedAST nodes directly. Version 3.0 replaces the previous JSON command-block system with a typed action language that mirrors Rust struct/enum syntax, enabling grammars to build type-safe ASTs at parse time without an intermediate representation.

The runtime is provided by `zyn_peg::runtime2` — a Packrat-memoized interpreter that achieves O(n × grammar_size) parsing time.

## Architecture

```text
┌──────────────────────────────────────────────────┐
│  .zyn grammar  →  parse_grammar()  →  GrammarIR  │
└──────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│  source code  →  GrammarInterpreter  →  TypedAST │
│                                                  │
│  • Packrat memoization (O(n) per rule)           │
│  • No JSON serialization                         │
│  • No pest VM or code generation                 │
└──────────────────────────────────────────────────┘
```

The `Grammar2` struct in `zyntax_embed` wraps this pipeline:

```rust
let grammar = Grammar2::from_source(include_str!("my_lang.zyn"))?;
let program: TypedProgram = grammar.parse(source_code)?;
```

## Grammar File Structure

A `.zyn` grammar file consists of directives followed by rule definitions:

```zyn
// Directives
@language { ... }
@types { ... }       // optional
@builtin { ... }     // optional

// Rule definitions
rule_name = modifier? { pattern }
  -> action
```

## Directives

### @language

Defines metadata about the language.

```zyn
@language {
    name: "ZynML",
    version: "1.0",
    file_extensions: [".ml", ".zynml"],
    entry_point: "main",
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Language name |
| `version` | string | Yes | Language version |
| `file_extensions` | string[] | No | File extensions to associate |
| `entry_point` | string | No | Function to call with `--run` flag |

### @types

Declares opaque extern types and their function return mappings.

```zyn
@types {
    opaque: [$Tensor, $Audio, $Model],
    returns: {
        tensor_zeros: $Tensor,
        audio_load: $Audio,
    }
}
```

- **`opaque`**: Types that are opaque at the language level (backed by ZRTL plugins)
- **`returns`**: Maps builtin alias names to their return type, overriding ZRTL signature inference

### @builtin

Maps grammar-level function/method/operator names to ZRTL symbol names. `Grammar2::parse_with_signatures()` uses this to inject extern declarations.

```zyn
@builtin {
    // Function aliases: grammar_name -> symbol_name
    tensor_zeros: "$Tensor$zeros",
    tensor_arange: "$Tensor$arange",

    // Method aliases (prefixed with @)
    @sum: "tensor_sum_f32",
    @mean: "tensor_mean_f32",

    // Operator aliases (prefixed with $@)
    $@add: "tensor_add_f32",
}
```

## Rule Definitions

### Basic Syntax

```zyn
rule_name = modifier? { pattern }
  -> action
```

### Rule Modifiers

| Modifier | Name | Description |
|----------|------|-------------|
| `@` | Atomic | No implicit whitespace skipping inside the rule |
| `_` | Silent | Rule is consumed but produces no value |
| `$` | Compound | Atomic but preserves inner token structure |
| `!` | Non-atomic | Forces whitespace skipping even inside atomic context |

### PEG Pattern Syntax

ZynPEG uses standard PEG operators extended with named bindings:

```zyn
// Sequence (whitespace is skipped between elements by default)
a ~ b ~ c

// Named binding: binds result of rule to a local variable
name:rule_ref

// Choice (ordered, first match wins)
a | b | c

// Repetition
a*        // Zero or more → Vec<T>
a+        // One or more  → Vec<T>
a?        // Optional     → Option<T>
a{n}      // Exactly n
a{n,}     // At least n
a{n,m}    // Between n and m

// Predicates (consume no input)
&a        // Positive lookahead
!a        // Negative lookahead

// Grouping
(a ~ b) | c

// Literals
"keyword"
'a'..'z'

// Built-in terminals
SOI              // Start of input
EOI              // End of input
ANY              // Any single character
ASCII_DIGIT      // 0-9
ASCII_ALPHA      // a-z, A-Z
ASCII_ALPHANUMERIC
ASCII_HEX_DIGIT
NEWLINE
WHITESPACE       // Auto-skipped between sequence elements
COMMENT          // Auto-skipped between sequence elements
```

### Named Bindings

Bindings capture rule results into local variables for use in actions:

```zyn
fn_def = { "def" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ body:block }
//                   ^^^^^                   ^^^^^^                   ^^^^
//               bound to 'name'        bound to 'params'      bound to 'body'
```

Binding types follow from the pattern:
- `name:rule` → `T` (the rule's return type)
- `items:rule*` → `Vec<T>`
- `opt:rule?` → `Option<T>`

## Actions

Actions follow the `->` arrow and describe how to construct TypedAST nodes from the parsed bindings. There are five action kinds.

### Construct — Direct AST Node Construction

The primary action type. Syntax mirrors Rust struct/enum construction:

```zyn
rule_name = { "def" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ ":" ~ ret:type_expr ~ body:block }
  -> TypedDeclaration::Function {
      name: intern(name),
      params: params.unwrap_or([]),
      return_type: ret,
      body: Some(body),
      is_async: false,
  }
```

The type path (`TypedDeclaration::Function`) identifies the enum variant or struct to construct. Field values are **ExprIR** expressions (see [Action Expressions](#action-expressions) below).

### PassThrough — Forward a Binding

For wrapper rules that just select between alternatives, `-> binding` returns the binding directly:

```zyn
// Choice rule: returns whichever alternative matched
statement = { let_stmt | assign_stmt | expr_stmt | return_stmt }
  -> stmt   // if the binding is named 'stmt'

// Or with implicit binding from a single-rule pattern:
factor = { inner:paren_expr | inner:number }
  -> inner
```

When a rule has no action, the last successfully bound value is returned.

### HelperCall — Built-in Helper Functions

Calls a built-in helper that operates on bindings:

```zyn
// prepend_list: combines first element with rest Vec into Vec
fn_params = { first:fn_param ~ rest:fn_param_comma* }
  -> prepend_list(first, rest)

// fold_left_ops: builds left-associative binary expression tree
additive_expr = { first:multiplicative_expr ~ rest:additive_rest* }
  -> fold_left_ops(first, rest)

// intern: interns a string into the arena
type_param = { name:identifier ~ (":" ~ type_bounds)? }
  -> intern(name)
```

Available helpers:

| Helper | Signature | Description |
|--------|-----------|-------------|
| `intern(s)` | `(text) → InternedString` | Intern a string into the arena |
| `prepend_list(first, rest)` | `(T, Vec<T>) → Vec<T>` | Prepend first element to rest |
| `fold_left_ops(first, rest)` | `(Expr, Vec<(op, Expr)>) → Expr` | Build left-associative binary tree |
| `make_pair(op, operand)` | `(op, Expr) → (op, Expr)` | Package an operator and operand for `fold_left_ops` |

### Match — Branch on a Binding Value

Dispatches to different construct actions based on a string binding value:

```zyn
stmt = { kind:("let" | "const") ~ name:identifier ~ "=" ~ value:expr }
  -> match kind {
      "let"   => TypedStatement::Let   { name: intern(name), value: value },
      "const" => TypedStatement::Const { name: intern(name), value: value },
  }
```

### Conditional — If/Else on an Expression

Branches on a boolean expression:

```zyn
fn_decl = { "def" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ ret:(":" ~ type_expr)? ~ body:block }
  -> if ret.is_some() {
      TypedDeclaration::Function { name: intern(name), params: params.unwrap_or([]), return_type: ret, body: Some(body) }
  } else {
      TypedDeclaration::Procedure { name: intern(name), params: params.unwrap_or([]), body: body }
  }
```

## Action Expressions

Action field values are **ExprIR** expressions. The following forms are available:

### Binding Reference

```zyn
name        // value of the binding 'name'
```

### Function Calls

```zyn
intern(name)                  // intern string → InternedString
prepend_list(first, rest)     // Vec construction helper
Some(value)                   // wrap in Option::Some
Box::new(expr)                // heap-box a value
```

### Method Calls

```zyn
params.unwrap_or([])          // Option<T> → T, using [] as default
opt.is_some()                 // bool
binding.text                  // get matched text as String
binding.span                  // get Span for the match
```

### Struct / Enum Construction

Inline struct or enum variant expressions within a field value:

```zyn
-> TypedExpression::Binary {
    left: Box::new(TypedExpression::Variable { name: intern(obj) }),
    op: op,
    right: right,
}
```

### List Literals

```zyn
path: [intern(name)]          // single-element Vec
declarations: []              // empty Vec
```

### Primitives

```zyn
"string"
42
true / false
```

### Binary Operations

```zyn
a == b
a && b
a || b
```

## Complete Examples

### Simple Import Rule

```zyn
import_simple = { "import" ~ name:identifier }
  -> TypedDeclaration::Import {
      path: [intern(name)],
  }
```

### Function Definition

```zyn
fn_def = {
    "def" ~ name:identifier
    ~ "(" ~ params:fn_params? ~ ")"
    ~ ":" ~ ret:type_expr
    ~ body:block
}
  -> TypedDeclaration::Function {
      name: intern(name),
      params: params.unwrap_or([]),
      return_type: ret,
      body: Some(body),
      is_async: false,
  }

// Parameters accumulate via prepend_list
fn_params = { first:fn_param ~ rest:fn_param_comma* }
  -> prepend_list(first, rest)

fn_param_comma = { "," ~ param:fn_param }
  -> param

fn_param = { name:identifier ~ ":" ~ ty:type_expr }
  -> TypedParameter {
      name: intern(name),
      ty: ty,
  }
```

### Left-Associative Binary Expressions

```zyn
// additive_rest packages (op, operand) pairs for fold_left_ops
additive_expr = { first:multiplicative_expr ~ rest:additive_rest* }
  -> fold_left_ops(first, rest)

additive_rest = { op:additive_op ~ operand:multiplicative_expr }
  -> make_pair(op, operand)

additive_op = @{ "+" | "-" }
```

### Wrapper / Pass-Through Rule

```zyn
// Choice rule: delegates to whichever alternative matched
type_expr = { optional_type | fn_type | generic_type | primitive_type | simple_type }

// Each alternative handles its own action; type_expr has no explicit action
// (implicitly passes through the result of the matched alternative)
```

### Struct Literal

```zyn
struct_field = { name:identifier ~ ":" ~ ty:type_expr }
  -> TypedField {
      name: intern(name),
      ty: ty,
  }

struct_fields = { first:struct_field ~ rest:struct_field_comma* ~ ","? }
  -> prepend_list(first, rest)
```

### Boxed Sub-Expressions

```zyn
comparison_with_op = { left:range_expr ~ op:comparison_op ~ right:range_expr }
  -> TypedExpression::Binary {
      op: op,
      left: Box::new(left),
      right: Box::new(right),
  }
```

### Program Entry Rule

```zyn
program = { SOI ~ items:top_level_items ~ EOI }
  -> TypedProgram {
      declarations: items,
  }

top_level_items = { decl:top_level_item* }
  -> decl   // Vec<TypedDeclaration> collected by the repeat

top_level_item = { fn_def | struct_def | import_stmt | ... }
  // passthrough — no action needed
```

## Grammar2 API

`Grammar2` (in `zyntax_embed`) is the primary interface for using a `.zyn` grammar at runtime:

```rust
use zyntax_embed::Grammar2;

// Load from embedded grammar source
let grammar = Grammar2::from_source(include_str!("my_lang.zyn"))?;

// Parse source → TypedProgram (direct, no signatures)
let program = grammar.parse(source_code)?;

// Parse with ZRTL plugin signatures for proper extern type resolution
let program = grammar.parse_with_signatures(source, filename, &plugin_sigs)?;

// Metadata access
grammar.name()             // → &str
grammar.version()          // → &str
grammar.file_extensions()  // → &[String]
grammar.entry_point()      // → Option<&str>
grammar.grammar_ir()       // → &GrammarIR (for inspection)
```

`parse_with_signatures` additionally injects `extern` function declarations for all entries in the `@builtin` directive, with types resolved from ZRTL plugin signatures or `@types.returns` overrides.

## Legacy JSON Actions (Backwards Compatibility)

The old JSON command-block syntax is still parsed and executed, but is deprecated. New grammars should use TypedAST actions exclusively.

**Old (JSON):**
```zyn
number = @{ ASCII_DIGIT+ }
  -> TypedExpression {
      "get_text": true,
      "parse_int": true,
      "define": "int_literal",
      "args": { "value": "$result" }
  }
```

**New (TypedAST):**
```zyn
number = @{ ASCII_DIGIT+ }
  -> TypedExpression::IntLiteral {
      value: number,
  }
```

Key differences:

| JSON (v2) | TypedAST (v3) |
|-----------|---------------|
| `"$1"`, `"$2"` positional references | Named bindings: `name:rule` |
| `"define": "node_type"` with args dict | `TypedAST::Variant { field: value }` |
| `"commands": [...]` sequential blocks | Inline expressions in field values |
| `"get_text": true` / `"parse_int": true` | Atomic rule (`@`) captures text automatically |
| `"fold_binary": { ... }` | `fold_left_ops(first, rest)` helper |
| `"store"` / `"load"` temporaries | Direct binding references |

## Packrat Memoization

The `runtime2` interpreter uses Packrat memoization keyed on `(position, rule_id)`. Each `execute_rule` call:

1. Checks `state.check_memo(rule_id)` — returns cached `Success`/`Failure` immediately on hit
2. Marks the entry as `InProgress` to detect left-recursive cycles
3. Executes the rule pattern and action
4. Stores the result via `state.store_memo_at(start_pos, rule_id, entry)`

This ensures each `(position, rule)` pair is evaluated at most once, converting exponential PEG backtracking to O(n × grammar_size).

## Best Practices

1. **Use atomic rules for tokens** — mark lexical rules with `@` so the matched text is captured automatically and whitespace is not skipped
2. **Use `_` (silent) for delimiters** — punctuation rules like commas, semicolons, and brackets rarely need to appear in the AST
3. **Use `prepend_list` for lists** — pair a `first:rule` binding with `rest:rule_comma*` (where `rule_comma = { "," ~ item:rule } -> item`) and combine with `prepend_list(first, rest)`
4. **Use `fold_left_ops` for binary operators** — pair with `make_pair(op, operand)` in the rest rule for correct left-associativity
5. **Use `intern()` for all identifier strings** — interns into the global arena for cheap equality and deduplication
6. **Prefer passthrough for choice rules** — if a rule just selects between alternatives, each alternative can have its own action; the choice rule needs no action

## Error Handling

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `Parse error at L:C: expected [...]` | PEG match failure | Check pattern syntax and token spelling |
| `unknown rule: foo` | Rule referenced but not defined | Define the missing rule |
| `binding 'name' not found` | Action references a binding not in the pattern | Add `name:rule` to the pattern |
| `left recursion detected` | Rule calls itself without consuming input | Refactor to use `rest*` style (no direct left recursion) |
| `UnexpectedResult` | Entry rule did not return `TypedProgram` | Ensure the `program` rule action returns `TypedProgram { ... }` |

## See Also

- [ZYN_PARSER_IMPLEMENTATION.md](ZYN_PARSER_IMPLEMENTATION.md) — Implementation details of the grammar parser and interpreter
- [BYTECODE_FORMAT_SPEC.md](BYTECODE_FORMAT_SPEC.md) — HIR/SSA bytecode format produced after parsing
- `crates/zyn_peg/src/grammar/ir.rs` — `GrammarIR`, `RuleIR`, `ActionIR`, `ExprIR`, `PatternIR` definitions
- `crates/zyn_peg/src/runtime2/interpreter.rs` — `GrammarInterpreter` and Packrat memoization
- `crates/zynml/ml.zyn` — Full reference grammar for ZynML showing all action patterns in use
