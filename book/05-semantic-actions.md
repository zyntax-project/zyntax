# Chapter 5: Semantic Actions

Semantic actions define how grammar rules build TypedAST nodes from parsed input. Each rule can have an action that follows the `->` arrow. Actions use a typed syntax that mirrors Rust struct/enum construction rather than JSON commands.

## Named Bindings

Before actions can reference parsed values, those values must be **bound** to names in the pattern:

```zyn
// Bind results to local variables for use in the action
fn_def = { "def" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ ":" ~ ret:type_expr ~ body:block }
//                   ^^^^                   ^^^^^^                   ^^^        ^^^
//             bound as 'name'        bound as 'params'         as 'ret'   as 'body'
```

The binding syntax is `variable_name:rule_name`. The type of the binding follows the pattern:

| Pattern | Binding type |
|---------|-------------|
| `name:rule` | `T` (rule's return value) |
| `items:rule*` | `Vec<T>` |
| `opt:rule?` | `Option<T>` |

This replaces the old `$1`, `$2` positional references. Named bindings make actions self-documenting.

## Action Kinds

There are five ways to write an action. Each follows the `->` arrow.

### 1. Construct — Build an AST Node

The primary action. Syntax mirrors Rust struct/enum construction:

```zyn
fn_def = { "def" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ ":" ~ ret:type_expr ~ body:block }
  -> TypedDeclaration::Function {
      name: intern(name),
      params: params.unwrap_or([]),
      return_type: ret,
      body: Some(body),
      is_async: false,
  }
```

The type path identifies the TypedAST variant to construct. Field values are **action expressions** (see [below](#action-expressions)).

### 2. PassThrough — Forward a Binding

For wrapper rules that just select between alternatives:

```zyn
// No explicit action: implicitly passes through the last matched value
statement = { let_stmt | assign_stmt | return_stmt | expr_stmt }

// Or explicitly:
factor = { inner:paren_expr | inner:number }
  -> inner
```

### 3. HelperCall — Built-in Helpers

Calls a helper function with bindings as arguments:

```zyn
// Combine first + rest into a Vec
fn_params = { first:fn_param ~ rest:fn_param_comma* }
  -> prepend_list(first, rest)

// Build a left-associative binary expression tree
additive_expr = { first:multiplicative_expr ~ rest:additive_rest* }
  -> fold_left_ops(first, rest)

// Intern a string identifier
type_param = { name:identifier }
  -> intern(name)
```

### 4. Match — Branch on a Value

Dispatch to different constructs based on a string binding:

```zyn
visibility_kw = { "pub" | "priv" }

decl = { vis:visibility_kw ~ "def" ~ name:identifier ~ ... }
  -> match vis {
      "pub"  => TypedDeclaration::Function { visibility: Visibility::Public,  name: intern(name), ... },
      "priv" => TypedDeclaration::Function { visibility: Visibility::Private, name: intern(name), ... },
  }
```

### 5. Conditional — If/Else

Branch on a boolean expression:

```zyn
fn_or_proc = { "def" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ ret:(":" ~ type_expr)? ~ body:block }
  -> if ret.is_some() {
      TypedDeclaration::Function {
          name: intern(name),
          params: params.unwrap_or([]),
          return_type: ret,
          body: Some(body),
      }
  } else {
      TypedDeclaration::Procedure {
          name: intern(name),
          params: params.unwrap_or([]),
          body: body,
      }
  }
```

## Action Expressions

Field values in actions are **expressions** that can reference bindings, call helpers, and construct nested nodes.

### Binding References

```zyn
name        // value of the binding 'name'
params      // Vec<TypedParameter> collected from 'params:fn_param*'
ret         // Option<Type> from 'ret:type_expr?'
```

### Helper Functions

```zyn
intern(name)              // String → InternedString (always use for identifiers)
Some(value)               // wrap in Option::Some
Box::new(expr)            // heap-box a value
prepend_list(first, rest) // T + Vec<T> → Vec<T>
```

### Method Calls

```zyn
params.unwrap_or([])   // Option<Vec<T>> → Vec<T>, with [] as default
opt.is_some()          // Option<T> → bool
binding.text           // get matched text as String
binding.span           // get the source Span
```

### Nested Node Construction

Fields can contain inline TypedAST node construction:

```zyn
comparison_with_op = { left:range_expr ~ op:comparison_op ~ right:range_expr }
  -> TypedExpression::Binary {
      op: op,
      left: Box::new(left),
      right: Box::new(right),
  }

pipe_call = { callee:identifier ~ "(" ~ args:call_args? ~ ")" }
  -> TypedExpression::Call {
      callee: Box::new(TypedExpression::Variable { name: intern(callee) }),
      args: args.unwrap_or([]),
  }
```

### List Literals

```zyn
path: [intern(name)]     // single-element Vec
declarations: []         // empty Vec
```

### String, Int, Bool Literals

```zyn
is_async: false
visibility: Visibility::Public
```

## Common Patterns

### Accumulating Lists

The standard pattern for comma-separated lists uses `prepend_list` and a helper rule for the comma:

```zyn
// The params list rule
fn_params = { first:fn_param ~ rest:fn_param_comma* }
  -> prepend_list(first, rest)

// Strip the comma, return the param
fn_param_comma = { "," ~ param:fn_param }
  -> param

// Build a single param
fn_param = { name:identifier ~ ":" ~ ty:type_expr }
  -> TypedParameter {
      name: intern(name),
      ty: ty,
  }
```

Usage in the parent rule:

```zyn
fn_def = { "def" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ ... }
  -> TypedDeclaration::Function {
      name: intern(name),
      params: params.unwrap_or([]),   // Option<Vec<TypedParameter>> → Vec<TypedParameter>
      ...
  }
```

### Left-Associative Binary Operators

Use `fold_left_ops` with a paired `make_pair` rest rule:

```zyn
additive_expr = { first:multiplicative_expr ~ rest:additive_rest* }
  -> fold_left_ops(first, rest)

// Package each (op, operand) pair
additive_rest = { op:additive_op ~ operand:multiplicative_expr }
  -> make_pair(op, operand)

// Atomic rule captures op text automatically
additive_op = @{ "+" | "-" }
```

For input `a + b - c` this builds:
```
Binary(-, Binary(+, a, b), c)
```

Full precedence chain:

```zyn
expr = { e:pipe_expr }
  -> e

pipe_expr = { first:or_expr ~ ... }       // lowest precedence
or_expr  = { inner:and_expr ~ ("||" ~ and_expr)* }  -> inner
and_expr = { inner:comparison_expr ~ ("&&" ~ comparison_expr)* } -> inner
comparison_expr = { comparison_with_op | comparison_no_op }
comparison_with_op = { left:additive_expr ~ op:comparison_op ~ right:additive_expr }
  -> TypedExpression::Binary { op: op, left: Box::new(left), right: Box::new(right) }
comparison_no_op = { inner:additive_expr }
  -> inner
additive_expr = { first:multiplicative_expr ~ rest:additive_rest* }
  -> fold_left_ops(first, rest)
multiplicative_expr = { first:unary_expr ~ rest:multiplicative_rest* }
  -> fold_left_ops(first, rest)
```

### Postfix Chains (Call, Index, Field Access)

The postfix expression pattern builds a chain of operations:

```zyn
postfix_expr = { base:primary_expr ~ suffix:postfix_suffix* }
  -> fold_left_ops(base, suffix)

postfix_suffix = { suffix_call | suffix_method | suffix_field | suffix_index }

suffix_call = { "(" ~ args:call_args? ~ ")" }
  -> TypedExpression::Suffix::Call { args: args.unwrap_or([]) }

suffix_field = { "." ~ name:identifier ~ !"(" }
  -> TypedExpression::Suffix::Field { name: intern(name) }

suffix_method = { "." ~ name:identifier ~ "(" ~ args:call_args? ~ ")" }
  -> TypedExpression::Suffix::Method { name: intern(name), args: args.unwrap_or([]) }
```

### Literals (Atomic Rules)

Atomic rules (`@`) capture their matched text automatically. Access it via the binding name:

```zyn
// The @-modifier makes 'integer' atomic; the binding captures its text
integer_literal = @{ "-"? ~ ASCII_DIGIT+ }
  -> TypedExpression::IntLiteral { value: integer_literal }
//                                        ^^^^^^^^^^^^^^ text of the match

string_literal = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" }
  -> TypedExpression::StringLiteral { value: string_literal }

bool_literal = @{ "true" | "false" }
  -> TypedExpression::BoolLiteral { value: bool_literal }
```

### Import and Path Rules

```zyn
import_simple = { "import" ~ name:identifier }
  -> TypedDeclaration::Import {
      path: [intern(name)],
  }

import_aliased = { "import" ~ path:module_path ~ "as" ~ alias:identifier }
  -> TypedDeclaration::Import {
      path: [intern(path)],
      alias: Some(intern(alias)),
  }

// @-rule captures the full dotted path as a single string
module_path = @{ identifier ~ ("." ~ identifier)* }
```

### Struct Fields and Constructors

```zyn
struct_field = { name:identifier ~ ":" ~ ty:type_expr }
  -> TypedField {
      name: intern(name),
      ty: ty,
  }

struct_fields = { first:struct_field ~ rest:struct_field_comma* ~ ","? }
  -> prepend_list(first, rest)

struct_field_comma = { "," ~ field:struct_field }
  -> field

struct_def = { "struct" ~ name:identifier ~ "{" ~ fields:struct_fields? ~ "}" }
  -> TypedDeclaration::Struct {
      name: intern(name),
      fields: fields.unwrap_or([]),
  }
```

### Optional Return Types

```zyn
// Optional return type: ": type_expr"
fn_def = {
    "def" ~ name:identifier
    ~ "(" ~ params:fn_params? ~ ")"
    ~ ret:(":" ~ ret_ty:type_expr)?
    ~ body:block
}
  -> if ret.is_some() {
      TypedDeclaration::Function {
          name: intern(name),
          params: params.unwrap_or([]),
          return_type: ret,
          body: Some(body),
      }
  } else {
      TypedDeclaration::Function {
          name: intern(name),
          params: params.unwrap_or([]),
          return_type: Type::Unit,
          body: Some(body),
      }
  }
```

## Complete Expression Grammar Example

```zyn
@language { name: "MyLang", version: "1.0" }

// Entry
program = { SOI ~ items:top_level_item* ~ EOI }
  -> TypedProgram { declarations: items }

top_level_item = { fn_def }

// Function definition
fn_def = { "fn" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ ":" ~ ret:type_expr ~ body:block }
  -> TypedDeclaration::Function {
      name: intern(name),
      params: params.unwrap_or([]),
      return_type: ret,
      body: Some(body),
  }

fn_params = { first:fn_param ~ rest:fn_param_comma* }
  -> prepend_list(first, rest)

fn_param_comma = { "," ~ param:fn_param }
  -> param

fn_param = { name:identifier ~ ":" ~ ty:type_expr }
  -> TypedParameter { name: intern(name), ty: ty }

// Block
block = { "{" ~ stmts:statement* ~ "}" }
  -> TypedBlock { stmts: stmts }

// Statements
statement = { return_stmt | let_stmt | expr_stmt }

return_stmt = { "return" ~ value:expr? ~ ";" }
  -> TypedStatement::Return { value: value }

let_stmt = { "let" ~ name:identifier ~ "=" ~ init:expr ~ ";" }
  -> TypedStatement::Let { name: intern(name), init: init }

expr_stmt = { e:expr ~ ";" }
  -> TypedStatement::Expr { expr: e }

// Expressions (left-associative operators)
expr = { e:additive_expr }
  -> e

additive_expr = { first:multiplicative_expr ~ rest:additive_rest* }
  -> fold_left_ops(first, rest)

additive_rest = { op:additive_op ~ operand:multiplicative_expr }
  -> make_pair(op, operand)

additive_op = @{ "+" | "-" }

multiplicative_expr = { first:unary_expr ~ rest:multiplicative_rest* }
  -> fold_left_ops(first, rest)

multiplicative_rest = { op:multiplicative_op ~ operand:unary_expr }
  -> make_pair(op, operand)

multiplicative_op = @{ "*" | "/" | "%" }

unary_expr = { unary_with_op | postfix_expr }

unary_with_op = { op:unary_op ~ operand:postfix_expr }
  -> TypedExpression::Unary {
      op: op,
      operand: Box::new(operand),
  }

unary_op = @{ "-" | "!" }

postfix_expr = { base:primary_expr ~ suffix:postfix_suffix* }
  -> fold_left_ops(base, suffix)

postfix_suffix = { suffix_call | suffix_field }

suffix_call = { "(" ~ args:call_arg_list? ~ ")" }
  -> TypedExpression::Suffix::Call { args: args.unwrap_or([]) }

suffix_field = { "." ~ name:identifier ~ !"(" }
  -> TypedExpression::Suffix::Field { name: intern(name) }

// Primary expressions
primary_expr = { int_literal | bool_literal | string_literal | paren_expr | var_expr }

int_literal = @{ ASCII_DIGIT+ }
  -> TypedExpression::IntLiteral { value: int_literal }

bool_literal = @{ "true" | "false" }
  -> TypedExpression::BoolLiteral { value: bool_literal }

string_literal = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" }
  -> TypedExpression::StringLiteral { value: string_literal }

paren_expr = _{ "(" ~ expr ~ ")" }

var_expr = { name:identifier }
  -> TypedExpression::Variable { name: intern(name) }

// Types
type_expr = { ty:identifier }
  -> Type::Named { name: intern(ty) }

// Terminals
identifier = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }

WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
COMMENT = _{ "//" ~ (!"\n" ~ ANY)* ~ "\n"? }
```

## Migrating from Legacy JSON Actions

If you have grammars using the old JSON command syntax, here is the migration mapping:

| Old (JSON) | New (TypedAST) |
|-----------|----------------|
| `"$1"`, `"$2"` | `first_binding`, `second_binding` (named) |
| `-> String { "get_text": true }` | `@` rule modifier — text is captured automatically |
| `"define": "int_literal", "args": {"value": "$result"}` | `TypedExpression::IntLiteral { value: int_literal }` |
| `"define": "function", "args": {"name":"$1","params":"$2",...}` | `TypedDeclaration::Function { name: intern(name), params: params, ... }` |
| `"fold_binary": {"operand":"term","operator":"add_op\|sub_op"}` | `fold_left_ops(first, rest)` with `additive_rest = { op:op ~ operand:term } -> make_pair(op, operand)` |
| `"get_child": {"index": 0}` | `-> inner` (passthrough) or no action |
| `"get_all_children": true` | Repetition binding: `items:rule*` → `items` is `Vec<T>` |
| `"commands": [...]` | Direct field expressions (no sequencing needed) |

**Before (JSON):**

```zyn
fn_decl = { "fn" ~ identifier ~ "(" ~ fn_params ~ ")" ~ type_expr ~ block }
  -> TypedDeclaration {
      "commands": [
          { "define": "function", "args": {
              "name": "$1",
              "params": "$2",
              "return_type": "$3",
              "body": "$4"
          }}
      ]
  }
```

**After (TypedAST):**

```zyn
fn_decl = { "fn" ~ name:identifier ~ "(" ~ params:fn_params ~ ")" ~ ret:type_expr ~ body:block }
  -> TypedDeclaration::Function {
      name: intern(name),
      params: params,
      return_type: ret,
      body: Some(body),
  }
```

## Next Steps

- [Chapter 6](./06-typed-ast.md): Understand the TypedAST node types your actions produce
- [Chapter 7](./07-typed-ast-builder.md): Use the builder API directly in Rust code
- [Chapter 15](./15-building-dsls.md): See these patterns applied in a complete DSL
