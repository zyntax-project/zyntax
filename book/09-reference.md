# Chapter 9: Reference

Complete reference for Zyn grammar commands and the TypedAST builder API.

## Action Reference

Semantic actions follow the `->` arrow and build TypedAST nodes from named bindings. There are five action kinds.

### Construct — Build a TypedAST Node

The primary action. Field values are expressions over bindings:

```zyn
fn_param = { name:identifier ~ ":" ~ ty:type_expr }
  -> TypedParameter { name: intern(name), ty: ty }

integer_literal = @{ ASCII_DIGIT+ }
  -> TypedExpression::IntLiteral { value: integer_literal }
```

Atomic rules (`@`) capture matched text automatically — the binding name holds the text.

### PassThrough — Return a Binding

For wrapper/choice rules:

```zyn
statement = { if_stmt | while_stmt | return_stmt | expr_stmt }
// no action needed — implicitly passes through the matched alternative

factor = { inner:paren_expr | inner:number }
  -> inner
```

### HelperCall — Built-in Helpers

```zyn
fn_params = { first:fn_param ~ rest:fn_param_comma* }
  -> prepend_list(first, rest)

additive_expr = { first:term ~ rest:additive_rest* }
  -> fold_left_ops(first, rest)

additive_rest = { op:add_op ~ operand:term }
  -> make_pair(op, operand)

type_name = { name:identifier }
  -> intern(name)
```

### Match — Branch on a String Binding

```zyn
decl = { kind:("let" | "const") ~ name:identifier ~ "=" ~ val:expr }
  -> match kind {
      "let"   => TypedStatement::Let   { name: intern(name), init: val },
      "const" => TypedStatement::Const { name: intern(name), init: val },
  }
```

### Conditional — If/Else

```zyn
fn_decl = { "fn" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ ret:type_expr? ~ body:block }
  -> if ret.is_some() {
      TypedDeclaration::Function { name: intern(name), params: params.unwrap_or([]), return_type: ret, body: Some(body) }
  } else {
      TypedDeclaration::Procedure { name: intern(name), params: params.unwrap_or([]), body: body }
  }
```

## Action Expression Reference

### Binding References

```zyn
name        // value of the binding 'name'
params      // Vec<T> from 'params:rule*'
ret         // Option<T> from 'ret:rule?'
```

### Helper Functions

| Call | Signature | Description |
|------|-----------|-------------|
| `intern(s)` | `text → InternedString` | Intern a string into the arena |
| `prepend_list(first, rest)` | `(T, Vec<T>) → Vec<T>` | Prepend first to rest Vec |
| `fold_left_ops(first, rest)` | `(Expr, Vec<(op,Expr)>) → Expr` | Build left-assoc binary tree |
| `make_pair(op, operand)` | `(op, Expr) → (op, Expr)` | Package for `fold_left_ops` |
| `Box::new(expr)` | `T → Box<T>` | Heap-box a value |
| `Some(value)` | `T → Option<T>` | Wrap in Some |

### Method Calls

```zyn
params.unwrap_or([])   // Option<Vec<T>> → Vec<T>
opt.is_some()          // Option<T> → bool
binding.text           // get matched text as String
binding.span           // get source Span
```

### Nested Node Construction

```zyn
-> TypedExpression::Call {
    callee: Box::new(TypedExpression::Variable { name: intern(name) }),
    args: args.unwrap_or([]),
}
```

### List Literals

```zyn
path: [intern(name)]   // single-element Vec
params: []             // empty Vec
```

### Primitives

```zyn
"string"   42   true   false
```

## TypedAST Variant Quick Reference

### Expressions (`TypedExpression::`)

| Variant | Key Fields | Description |
|---------|-----------|-------------|
| `IntLiteral` | `value` | Integer literal |
| `BoolLiteral` | `value` | Boolean literal |
| `StringLiteral` | `value` | String literal |
| `Variable` | `name` | Variable reference |
| `Binary` | `op, left, right` | Binary operation |
| `Unary` | `op, operand` | Unary operation |
| `Call` | `callee, args` | Function call |
| `FieldAccess` | `object, field` | Field access |
| `Index` | `object, index` | Index access |
| `StructLiteral` | `type_name, fields` | Struct literal |
| `ArrayLiteral` | `elements` | Array literal |

### Statements (`TypedStatement::`)

| Variant | Key Fields | Description |
|---------|-----------|-------------|
| `Let` | `name, init` | Variable declaration |
| `Const` | `name, init` | Constant declaration |
| `Return` | `value?` | Return statement |
| `Expr` | `expr` | Expression statement |
| `Assign` | `target, value` | Assignment |
| `If` | `condition, then_branch, else_branch?` | If statement |
| `While` | `condition, body` | While loop |
| `For` | `iterable, binding, body` | For loop |

### Declarations (`TypedDeclaration::`)

| Variant | Key Fields | Description |
|---------|-----------|-------------|
| `Function` | `name, params, return_type, body` | Function declaration |
| `Struct` | `name, fields` | Struct declaration |
| `Enum` | `name, variants` | Enum declaration |
| `Import` | `path, alias?` | Import declaration |

### Types (`Type::`)

| Variant | Key Fields | Description |
|---------|-----------|-------------|
| `Named` | `name` | Named/primitive type |
| `Pointer` | `pointee` | Pointer type |
| `Optional` | `inner` | Optional type |
| `Array` | `size?, element` | Array type |
| `Extern` | `name` | Extern/opaque type |

## Grammar Syntax Reference

### Rule Types

| Syntax | Description | Creates Node |
|--------|-------------|--------------|
| `rule = { ... }` | Normal rule | Yes |
| `rule = @{ ... }` | Atomic rule | Yes |
| `rule = _{ ... }` | Silent rule | No |

### Operators

| Operator | Name | Description |
|----------|------|-------------|
| `~` | Sequence | Match in order |
| `\|` | Choice | First match wins |
| `*` | Zero or more | Repeat 0+ times |
| `+` | One or more | Repeat 1+ times |
| `?` | Optional | Match 0 or 1 time |
| `!` | Not | Succeed if doesn't match |
| `&` | And | Succeed if matches (no consume) |

### Built-in Rules

| Rule | Matches |
|------|---------|
| `SOI` | Start of input |
| `EOI` | End of input |
| `ANY` | Any character |
| `ASCII` | ASCII character (0x00-0x7F) |
| `ASCII_DIGIT` | 0-9 |
| `ASCII_ALPHA` | a-z, A-Z |
| `ASCII_ALPHANUMERIC` | a-z, A-Z, 0-9 |
| `ASCII_HEX_DIGIT` | 0-9, a-f, A-F |
| `NEWLINE` | \n or \r\n |

### Special Rules

| Rule | Purpose |
|------|---------|
| `WHITESPACE` | Define whitespace handling |
| `COMMENT` | Define comment syntax |

## TypedAST Builder API

### Construction

```rust
use zyntax_typed_ast::TypedASTBuilder;

let mut builder = TypedASTBuilder::new();
```

### Type Helpers

```rust
builder.i32_type()     // Type::Primitive(PrimitiveType::I32)
builder.i64_type()     // Type::Primitive(PrimitiveType::I64)
builder.bool_type()    // Type::Primitive(PrimitiveType::Bool)
builder.string_type()  // Type::Primitive(PrimitiveType::String)
builder.unit_type()    // Type::Primitive(PrimitiveType::Unit)
builder.char_type()    // Type::Primitive(PrimitiveType::Char)
builder.f32_type()     // Type::Primitive(PrimitiveType::F32)
builder.f64_type()     // Type::Primitive(PrimitiveType::F64)
```

### Span Helpers

```rust
builder.span(start, end)  // Create span from byte offsets
builder.dummy_span()      // Create (0, 0) span for testing
```

### Expression Builders

```rust
// Literals
builder.int_literal(42, span)
builder.string_literal("hello", span)
builder.bool_literal(true, span)
builder.char_literal('x', span)
builder.unit_literal(span)

// References
builder.variable("name", ty, span)

// Operations
builder.binary(BinaryOp::Add, left, right, result_ty, span)
builder.unary(UnaryOp::Minus, operand, result_ty, span)

// Access
builder.field_access(object, "field", field_ty, span)
builder.index(object, index_expr, element_ty, span)

// Calls
builder.call_positional(callee, args_vec, return_ty, span)
builder.call_named(callee, named_args_vec, return_ty, span)

// Composite
builder.struct_literal("Name", fields_vec, struct_ty, span)
builder.array_literal(elements_vec, array_ty, span)
builder.tuple(elements_vec, tuple_ty, span)
builder.lambda(params_vec, body, lambda_ty, span)

// Special
builder.cast(expr, target_ty, span)
builder.try_expr(expr, result_ty, span)
builder.await_expr(expr, result_ty, span)
builder.reference(expr, mutability, ptr_ty, span)
builder.dereference(expr, deref_ty, span)
```

### Statement Builders

```rust
// Declarations
builder.let_statement("name", ty, mutability, init_opt, span)

// Control flow
builder.if_statement(condition, then_block, else_opt, span)
builder.while_loop(condition, body, span)
builder.for_loop("binding", iterable, body, span)
builder.loop_stmt(body, span)

// Jumps
builder.return_stmt(value, span)
builder.return_void(span)
builder.break_stmt(span)
builder.break_with_value(value, span)
builder.continue_stmt(span)

// Other
builder.expression_statement(expr, span)
builder.throw_stmt(exception, span)
builder.block(statements_vec, span)
```

### Pattern Builders

```rust
builder.struct_pattern("Name", fields_vec, span)
builder.enum_pattern("Enum", "Variant", fields_vec, span)
builder.array_pattern(patterns_vec, span)
builder.slice_pattern(prefix, middle_opt, suffix, span)
```

## CLI Reference

### Compile Command

```bash
zyntax compile [OPTIONS] [INPUT]...

Options:
  -s, --source <SOURCE>    Source file (with --grammar)
  -g, --grammar <GRAMMAR>  ZynPEG grammar file (.zyn)
  -o, --output <OUTPUT>    Output file path
  -b, --backend <BACKEND>  Backend (jit, llvm) [default: jit]
  -v, --verbose            Verbose output
  -O, --opt-level <LEVEL>  Optimization (0-3) [default: 2]
  -f, --format <FORMAT>    Input format (auto, typed-ast, hir-bytecode, zyn)
      --run                Run immediately (JIT only)
```

### Examples

```bash
# Compile and run Zig file
zyntax compile --grammar zig.zyn --source hello.zig --run

# Compile to object file
zyntax compile --grammar zig.zyn --source main.zig -o main.o

# Verbose compilation
zyntax compile --grammar zig.zyn --source test.zig -v --run

# Use LLVM backend
zyntax compile --grammar zig.zyn --source test.zig -b llvm -o test.o
```

## Error Messages

### Grammar Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Rule not found" | Reference to undefined rule | Define the rule or check spelling |
| "Left recursion detected" | `a = { a ~ ... }` | Rewrite using repetition |
| `binding 'x' not found` | Action references a binding not in the pattern | Add `x:rule` to the pattern |
| `left recursion detected` | Rule calls itself without consuming input | Refactor to use `rest*` style |

### Runtime Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Cannot access fields on non-struct type" | Field access on wrong type | Check object type |
| "Unknown variant" | Enum variant not found | Check variant name |
| "Type mismatch" | Incompatible types | Check expression types |

## Best Practices

### Grammar Organization

1. **Group related rules** - Keep declarations, statements, expressions separate
2. **Order choices correctly** - Longer/more specific first
3. **Use meaningful names** - `if_stmt` not `rule1`
4. **Comment complex rules** - Explain non-obvious patterns

### Performance

1. **Avoid excessive backtracking** - Use negative lookahead
2. **Make atomic rules atomic** - Use `@{ }` for tokens
3. **Keep grammar focused** - Don't over-generalize

### Debugging

1. **Test incrementally** - Add rules one at a time
2. **Use verbose mode** - `--verbose` shows parse tree
3. **Use named bindings** - `name:rule` makes patterns self-documenting
4. **Start simple** - Get basic cases working first
