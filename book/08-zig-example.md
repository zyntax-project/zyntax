# Chapter 8: Complete Example - Zig Grammar

This chapter walks through the complete Zig grammar implementation (`zig.zyn`), explaining each section and the design decisions behind it.

## Overview

The Zig grammar supports:
- Functions with typed parameters
- Structs and enums
- Control flow (if, while, for)
- Expressions with proper operator precedence
- Type expressions (pointers, optionals, arrays)

## File Structure

```zyn
// 1. Language metadata
@language { ... }

// 2. Program structure
program = { ... }
declarations = { ... }
declaration = { ... }

// 3. Type declarations
struct_decl = { ... }
enum_decl = { ... }

// 4. Function declarations
fn_decl = { ... }

// 5. Statements
statement = { ... }
if_stmt = { ... }
while_stmt = { ... }
// ...

// 6. Expressions (by precedence)
expr = { ... }
logical_or = { ... }
// ... down to atoms

// 7. Literals and identifiers
integer_literal = { ... }
identifier = { ... }

// 8. Operators
add_op = { ... }
// ...

// 9. Whitespace/comments
WHITESPACE = { ... }
COMMENT = { ... }
```

## Language Metadata

```zyn
@language {
    name: "Zig",
    version: "0.11",
    file_extensions: [".zig"],
    entry_point: "main",
}
```

This metadata tells the compiler:
- The language name for error messages
- Which file extensions to recognize
- Which function to execute for `--run`

## Program Structure

### The Entry Point

```zyn
program = { SOI ~ decls:declaration* ~ EOI }
  -> TypedProgram { declarations: decls }
```

This matches the entire file (`SOI` to `EOI`) and collects declarations into a `Vec`.

### Declaration Dispatch

```zyn
declaration = { struct_decl | enum_decl | fn_decl | const_decl | var_decl }
```

Passthrough — each alternative handles its own action. Order matters! More specific rules should come first if there's ambiguity.

## Struct Declarations

```zyn
struct_decl = { "const" ~ name:identifier ~ "=" ~ "struct" ~ "{" ~ fields:struct_fields? ~ "}" ~ ";" }
  -> TypedDeclaration::Struct {
      name: intern(name),
      fields: fields.unwrap_or([]),
  }
```

Named bindings make the mapping explicit:
- `name:identifier` → the struct name
- `fields:struct_fields?` → `Option<Vec<TypedField>>`; `.unwrap_or([])` gives an empty list when absent

### Struct Fields

```zyn
struct_fields = { first:struct_field ~ rest:struct_field_comma* ~ ","? }
  -> prepend_list(first, rest)

struct_field_comma = { "," ~ f:struct_field }
  -> f

struct_field = { name:identifier ~ ":" ~ ty:type_expr }
  -> TypedField { name: intern(name), ty: ty }
```

Pattern: `prepend_list(first, rest)` accumulates a comma-separated list. Optional trailing comma (`","?`) is common in modern languages.

## Enum Declarations

```zyn
enum_decl = { "const" ~ name:identifier ~ "=" ~ "enum" ~ "{" ~ variants:enum_variants? ~ "}" ~ ";" }
  -> TypedDeclaration::Enum {
      name: intern(name),
      variants: variants.unwrap_or([]),
  }

enum_variants = { first:enum_variant ~ rest:enum_variant_comma* ~ ","? }
  -> prepend_list(first, rest)

enum_variant_comma = { "," ~ v:enum_variant }
  -> v

enum_variant = { name:identifier }
  -> TypedVariant { name: intern(name) }
```

Example:
```zig
const Color = enum {
    Red,
    Green,
    Blue,
};
```

The runtime assigns discriminant values (0, 1, 2) automatically.

## Function Declarations

### No Split Needed

With named bindings, a single rule handles both with and without parameters:

```zyn
fn_decl = { "fn" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ ret:type_expr ~ body:block }
  -> TypedDeclaration::Function {
      name: intern(name),
      params: params.unwrap_or([]),
      return_type: ret,
      body: Some(body),
      is_async: false,
  }
```

**Why no split?** The old JSON `$N` positional references shifted when optional children were absent — `$3` could be `type_expr` or `block` depending on whether params existed. Named bindings (`params:fn_params?`) give you an `Option<Vec<TypedParameter>>` regardless of position. `.unwrap_or([])` provides the empty list default inline.

### Parameters

```zyn
fn_params = { first:fn_param ~ rest:fn_param_comma* }
  -> prepend_list(first, rest)

fn_param_comma = { "," ~ p:fn_param }
  -> p

fn_param = { name:identifier ~ ":" ~ ty:type_expr }
  -> TypedParameter { name: intern(name), ty: ty }
```

## Statements

### Statement Dispatch

```zyn
statement = { if_stmt | while_stmt | for_stmt | return_stmt | break_stmt |
              continue_stmt | local_const | local_var | assign_stmt | expr_stmt }
```

Passthrough — each alternative handles its own action. Order consideration: `if_stmt` before `expr_stmt` because an identifier `if_something` could otherwise match.

### If Statement

```zyn
if_stmt = { if_else | if_only }

if_only = { "if" ~ "(" ~ cond:expr ~ ")" ~ body:block }
  -> TypedStatement::If {
      condition: cond,
      then_branch: body,
      else_branch: None,
  }

if_else = { "if" ~ "(" ~ cond:expr ~ ")" ~ then_b:block ~ "else" ~ else_b:block }
  -> TypedStatement::If {
      condition: cond,
      then_branch: then_b,
      else_branch: Some(else_b),
  }
```

**Important**: `if_else` must come before `if_only` in the choice, otherwise `if_only` would always match first!

### While Loop

```zyn
while_stmt = { "while" ~ "(" ~ cond:expr ~ ")" ~ body:block }
  -> TypedStatement::While { condition: cond, body: body }
```

### For Loop (Zig Style)

```zyn
for_stmt = { "for" ~ "(" ~ iter:expr ~ ")" ~ "|" ~ binding:identifier ~ "|" ~ body:block }
  -> TypedStatement::For { iterable: iter, binding: intern(binding), body: body }
```

Zig's for loop: `for (slice) |item| { ... }`

### Return Statement

```zyn
return_stmt = { "return" ~ val:expr? ~ ";" }
  -> TypedStatement::Return { value: val }
```

`val` is `Option<TypedExpression>` — `None` when `expr?` doesn't match.

### Block

```zyn
block = { "{" ~ stmts:statement* ~ "}" }
  -> TypedBlock { stmts: stmts }
```

## Switch Expressions

Zig supports switch expressions for pattern matching against values. The grammar handles multiple pattern types including literals, ranges, struct patterns, tagged union patterns, and error patterns.

### Basic Structure

```zyn
switch_expr = { "switch" ~ "(" ~ scrutinee:expr ~ ")" ~ "{" ~ cases:switch_cases? ~ "}" }
  -> TypedExpression::Switch {
      scrutinee: Box::new(scrutinee),
      cases: cases.unwrap_or([]),
  }

switch_cases = { first:switch_case ~ rest:switch_case_comma* ~ ","? }
  -> prepend_list(first, rest)

switch_case_comma = { "," ~ c:switch_case }
  -> c
```

`TypedExpression::Switch` takes:
- `scrutinee`: The expression being matched against
- `cases`: Vec of case arms

### Switch Cases

Each case has a pattern and a body:

```zyn
// Value case: pattern => expr
switch_case_value = { pat:switch_pattern ~ "=>" ~ body:expr }
  -> TypedExpression::SwitchCase {
      pattern: pat,
      body: Box::new(body),
  }

// Else case: else => expr
switch_case_else = { "else" ~ "=>" ~ body:expr }
  -> TypedExpression::SwitchCase {
      pattern: TypedPattern::Wildcard,
      body: Box::new(body),
  }
```

The `else` case inlines `TypedPattern::Wildcard` directly as a field value.

### Pattern Types

#### Literal Patterns

Match exact values:

```zyn
switch_literal_pattern = { val:(integer_literal | string_literal) }
  -> TypedPattern::Literal { value: val }
```

Example:
```zig
const result = switch (x) {
    1 => 10,
    2 => 20,
    else => 0,
};
```

#### Wildcard Pattern

Match anything (used for `_` or `else`):

```zyn
switch_wildcard_pattern = { "_" }
  -> TypedPattern::Wildcard
```

#### Range Patterns

Match values within a range:

```zyn
switch_range_pattern = { start:integer_literal ~ ".." ~ end:integer_literal }
  -> TypedPattern::Range { start: start, end: end, inclusive: false }
```

Example:

```zig
const result = switch (x) {
    0..9 => "single digit",
    10..99 => "double digit",
    else => "other",
};
```

#### Tagged Union Patterns

Match enum or tagged union variants (Zig uses `.variant` syntax):

```zyn
switch_tagged_union_pattern = { "." ~ name:identifier }
  -> TypedPattern::EnumVariant { name: intern(name) }
```

Example:

```zig
const result = switch (optional_value) {
    .some => 100,
    .none => 0,
};
```

**Note**: Tagged union patterns against non-enum types (like integers) gracefully return `false` in the backend, allowing the else case to match.

#### Struct Patterns

Match struct values by field:

```zyn
switch_struct_pattern = { type_name:identifier ~ "{" ~ fields:struct_field_patterns? ~ "}" }
  -> TypedPattern::Struct { name: intern(type_name), fields: fields.unwrap_or([]) }

switch_struct_field_pattern = { "." ~ name:identifier ~ pat:("=" ~ switch_pattern)? }
  -> TypedPattern::FieldPattern { name: intern(name), pattern: pat }
```

Example:

```zig
const result = switch (point) {
    Point{ .x = 0, .y = 0 } => "origin",
    Point{ .x = 0 } => "on y-axis",
    else => "elsewhere",
};
```

#### Error Patterns

Match error values from error unions:

```zyn
switch_error_pattern = { "error" ~ "." ~ name:identifier }
  -> TypedPattern::Error { name: intern(name) }
```

Example:

```zig
const result = switch (error_union) {
    error.OutOfMemory => "memory error",
    error.NotFound => "not found",
    else => "success or other",
};
```

#### Pointer Patterns

Match through pointer dereference:

```zyn
switch_pointer_pattern = { "*" ~ inner:switch_pattern }
  -> TypedPattern::Pointer { inner: Box::new(inner), mutable: false }
```

### Testing Switch Expressions

```zig
// Literal pattern match
fn main() i32 {
    const x = 2;
    const result = switch (x) {
        1 => 10,
        2 => 20,
        else => 0,
    };
    return result;
}
```

```bash
# Returns: 20
```

```zig
// Else case (no match)
fn main() i32 {
    const x = 99;
    const result = switch (x) {
        1 => 10,
        2 => 20,
        else => 0,
    };
    return result;
}
```

```bash
# Returns: 0
```

## Expressions

### The Precedence Chain

Operators are handled by a chain from lowest to highest precedence. Each level uses `fold_left_ops` with a companion `rest` rule that packages `(op, operand)` pairs using `make_pair`:

```zyn
expr = { e:logical_or }
  -> e

// Lowest: OR
logical_or = { first:logical_and ~ rest:logical_or_rest* }
  -> fold_left_ops(first, rest)

logical_or_rest = { op:or_op ~ operand:logical_and }
  -> make_pair(op, operand)

// AND
logical_and = { first:comparison ~ rest:logical_and_rest* }
  -> fold_left_ops(first, rest)

logical_and_rest = { op:and_op ~ operand:comparison }
  -> make_pair(op, operand)

// Comparison
comparison = { first:addition ~ rest:comparison_rest* }
  -> fold_left_ops(first, rest)

comparison_rest = { op:comparison_op ~ operand:addition }
  -> make_pair(op, operand)

comparison_op = @{ "==" | "!=" | "<=" | ">=" | "<" | ">" }

// Addition/Subtraction
addition = { first:multiplication ~ rest:addition_rest* }
  -> fold_left_ops(first, rest)

addition_rest = { op:add_sub_op ~ operand:multiplication }
  -> make_pair(op, operand)

add_sub_op = @{ "+" | "-" }

// Multiplication/Division
multiplication = { first:unary ~ rest:multiplication_rest* }
  -> fold_left_ops(first, rest)

multiplication_rest = { op:mul_div_op ~ operand:unary }
  -> make_pair(op, operand)

mul_div_op = @{ "*" | "/" }

// Unary (highest before atoms)
unary = { unary_with_op | primary }

unary_with_op = { op:unary_op ~ operand:primary }
  -> TypedExpression::Unary { op: op, operand: Box::new(operand) }

unary_op = @{ "-" | "!" }
```

### The `fold_left_ops` Pattern

For `1 + 2 + 3`:

1. Parse: `first = 1`, `rest = [("+", 2), ("+", 3)]`
2. Fold: `Binary(+, 1, 2)` → `Binary(+, Binary(+, 1, 2), 3)`

This creates left-associative trees automatically. The atomic operator rules (`add_sub_op = @{ "+" | "-" }`) capture operator text without whitespace interference.

### Postfix Expressions

```zyn
postfix_expr = { call_expr | field_expr | index_expr | atom }

// Function call
call_expr = { callee:atom ~ "(" ~ args:call_args? ~ ")" }
  -> TypedExpression::Call {
      callee: Box::new(callee),
      args: args.unwrap_or([]),
  }

// Field access
field_expr = { obj:atom ~ "." ~ name:identifier }
  -> TypedExpression::FieldAccess {
      object: Box::new(obj),
      field: intern(name),
  }

// Index access
index_expr = { obj:atom ~ "[" ~ idx:expr ~ "]" }
  -> TypedExpression::Index {
      object: Box::new(obj),
      index: Box::new(idx),
  }
```

### Atoms (Highest Precedence)

```zyn
atom = { try_expr | struct_init | array_literal | bool_literal |
         string_literal | integer_literal | identifier_expr | paren_expr }
```

Passthrough — each alternative has its own action. Order matters: `struct_init` (starts with identifier) before `identifier_expr`.

### Struct Initialization

```zyn
struct_init = { type_name:identifier ~ "{" ~ fields:struct_init_fields? ~ "}" }
  -> TypedExpression::StructLiteral {
      type_name: intern(type_name),
      fields: fields.unwrap_or([]),
  }

struct_init_fields = { first:struct_init_field ~ rest:struct_init_field_comma* ~ ","? }
  -> prepend_list(first, rest)

struct_init_field_comma = { "," ~ f:struct_init_field }
  -> f

struct_init_field = { "." ~ name:identifier ~ "=" ~ val:expr }
  -> TypedExpression::FieldInit { name: intern(name), value: Box::new(val) }
```

Example: `Point{ .x = 10, .y = 20 }`

### Parenthesized Expressions

```zyn
paren_expr = _{ "(" ~ expr ~ ")" }
```

Silent rule (`_{ }`) - matches but doesn't create a node. The inner `expr` passes through directly.

## Type Expressions

```zyn
type_expr = { pointer_type | optional_type | error_union_type | array_type |
              primitive_type | name_type }

pointer_type = { "*" ~ "const"? ~ pointee:type_expr }
  -> Type::Pointer { pointee: Box::new(pointee) }

optional_type = { "?" ~ inner:type_expr }
  -> Type::Optional { inner: Box::new(inner) }

array_type = { "[" ~ size:integer_literal? ~ "]" ~ element:type_expr }
  -> Type::Array { size: size, element: Box::new(element) }

primitive_type = @{ "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" |
                    "f32" | "f64" | "bool" | "void" }
  -> Type::Named { name: intern(primitive_type) }

// Fall-through for user-defined type names
name_type = { name:identifier }
  -> Type::Named { name: intern(name) }
```

## Identifiers and Keywords

### Keyword Protection

```zyn
keyword = @{
    ("struct" | "enum" | "fn" | "const" | "var" | "if" | "else" | "while" | "for" |
     "return" | "break" | "continue" | "try" | "and" | "or" | "true" | "false" |
     "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "f32" | "f64" |
     "bool" | "void")
    ~ !(ASCII_ALPHANUMERIC | "_")
}

// Atomic rule captures text; action interns it into the arena
identifier = @{ !keyword ~ (ASCII_ALPHA | "_") ~ (ASCII_ALPHANUMERIC | "_")* }
  -> intern(identifier)
```

**Key patterns**:
1. `~ !(ASCII_ALPHANUMERIC | "_")` ensures "iffy" doesn't match as "if" + "fy"
2. `!keyword` prevents identifiers from being keywords
3. Both are atomic (`@{ }`) for proper token handling

## Operators

Operators are atomic rules (`@`) so their text is captured automatically. Longer operators must come first in alternatives to avoid partial matches:

```zyn
// Must check longer operators first
comparison_op = @{ "==" | "!=" | "<=" | ">=" | "<" | ">" }

add_sub_op = @{ "+" | "-" }
mul_div_op = @{ "*" | "/" }

and_op = @{ "and" }
or_op  = @{ "or" }

unary_op = @{ "-" | "!" }
```

These are used as the `op` binding in rest rules (e.g. `addition_rest = { op:add_sub_op ~ operand:multiplication } -> make_pair(op, operand)`).

## Whitespace and Comments

```zyn
WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
COMMENT = _{ "//" ~ (!"\n" ~ ANY)* ~ "\n"? }
```

Both are silent (`_{ }`) - they match but don't appear in the parse tree.

## Testing the Grammar

### Simple Function

```zig
fn main() i32 {
    return 42;
}
```

```bash
zyntax compile --grammar zig.zyn --source test.zig --run
# Output: result: main() returned: 42
```

### Struct with Field Access

```zig
const Point = struct {
    x: i32,
    y: i32,
};

fn main() i32 {
    const p = Point{ .x = 10, .y = 20 };
    return p.x;
}
```

```bash
# Returns: 10
```

### Enum Variants

```zig
const Color = enum {
    Red,
    Green,
    Blue,
};

fn main() i32 {
    return Color.Green;
}
```

```bash
# Returns: 1 (Green's discriminant)
```

### Arithmetic Expression

```zig
fn main() i32 {
    return 2 + 3 * 4;
}
```

```bash
# Returns: 14 (multiplication before addition)
```

## Common Patterns Summary

| Pattern | Use Case |
|---------|----------|
| Named bindings (`name:rule`) | Self-documenting patterns; optional `?` → `Option<T>`, star `*` → `Vec<T>` |
| `fold_left_ops` + `make_pair` | Left-associative binary operators |
| `prepend_list(first, rest)` | Collect comma-separated lists |
| `.unwrap_or([])` | Default for optional list bindings |
| Keyword protection | Prevent identifiers matching keywords |
| Silent rules (`_{}`) | Grouping/delimiters without AST nodes |
| Atomic rules (`@{}`) | Token-level matching; text auto-captured |

## Next Steps

- [Chapter 9](./09-reference.md): Complete command and API reference
- Try modifying the grammar to add new features!
