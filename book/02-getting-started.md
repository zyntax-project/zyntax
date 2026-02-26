# Chapter 2: Getting Started

## Prerequisites

Before using Zyn, ensure you have:

- Rust toolchain (1.70+)
- The `zyntax` CLI tool

```bash
# Build zyntax from source
cargo build --release

# Verify installation
./target/release/zyntax --help
```

## Your First Grammar

Let's create a minimal calculator language that supports:
- Integer literals
- Addition and subtraction
- Parentheses for grouping

### Step 1: Create the Grammar File

Create `calc.zyn`:

```zyn
// Calculator Language Grammar
@language {
    name: "Calc",
    version: "1.0",
    file_extensions: [".calc"],
    entry_point: "main",
}

// Program structure: wrap the expression in a main function
program = { SOI ~ e:expr ~ EOI }
  -> TypedProgram {
      declarations: [
          TypedDeclaration::Function {
              name: intern("main"),
              params: [],
              return_type: Type::Named { name: intern("i64") },
              body: Some(TypedBlock {
                  stmts: [TypedStatement::Return { value: Some(e) }],
              }),
              is_async: false,
          }
      ],
  }

// Expression with addition/subtraction (left-associative)
expr = { first:term ~ rest:expr_rest* }
  -> fold_left_ops(first, rest)

expr_rest = { op:add_sub_op ~ operand:term }
  -> make_pair(op, operand)

add_sub_op = @{ "+" | "-" }

// Terms are atoms or parenthesized expressions — passthrough
term = { integer | paren_expr }

// Parenthesized expression (silent rule - doesn't create node)
paren_expr = _{ "(" ~ expr ~ ")" }

// Integer literal — atomic rule captures text automatically
integer = @{ ASCII_DIGIT+ }
  -> TypedExpression::IntLiteral { value: integer }

// Whitespace handling
WHITESPACE = _{ " " | "\t" | "\n" }
```

### Step 2: Create a Test File

Create `test.calc`:

```
1 + 2 + 3
```

### Step 3: Compile and Run

```bash
zyntax compile --grammar calc.zyn --source test.calc --run
```

## Understanding the Grammar

### Language Metadata

```zyn
@language {
    name: "Calc",
    version: "1.0",
    file_extensions: [".calc"],
    entry_point: "main",
}
```

This block defines metadata about your language:
- `name`: Language identifier
- `version`: Grammar version
- `file_extensions`: Associated file types
- `entry_point`: The function to execute (for JIT compilation)

### Grammar Rules

Rules follow PEG syntax with optional semantic actions:

```zyn
rule_name = { pattern }
  -> TypedAST::Variant { field: value, ... }
```

| Syntax | Meaning |
|--------|---------|
| `{ }` | Normal rule (creates parse node) |
| `@{ }` | Atomic rule (no whitespace handling) |
| `_{ }` | Silent rule (matches but creates no node) |

### Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `~` | Sequence | `a ~ b` matches a then b |
| `|` | Choice | `a | b` matches a or b |
| `*` | Zero or more | `a*` matches "", "a", "aa", ... |
| `+` | One or more | `a+` matches "a", "aa", ... |
| `?` | Optional | `a?` matches "" or "a" |
| `!` | Not predicate | `!a` succeeds if a fails |
| `&` | And predicate | `&a` succeeds if a matches (no consume) |

### Built-in Rules

| Rule | Matches |
|------|---------|
| `SOI` | Start of input |
| `EOI` | End of input |
| `ANY` | Any single character |
| `ASCII_DIGIT` | 0-9 |
| `ASCII_ALPHA` | a-z, A-Z |
| `ASCII_ALPHANUMERIC` | a-z, A-Z, 0-9 |
| `WHITESPACE` | Define whitespace handling |
| `COMMENT` | Define comment syntax |

## Semantic Actions

Each rule can have a semantic action that builds a TypedAST node directly from parsed bindings:

```zyn
// Atomic rule (@) — matched text is available via the binding name 'integer'
integer = @{ ASCII_DIGIT+ }
  -> TypedExpression::IntLiteral { value: integer }
```

The `->` arrow connects the grammar rule to its action:
- `TypedExpression::IntLiteral` is the TypedAST variant to construct
- `value: integer` sets the field using the captured text from the binding

Named bindings in the pattern make values available to the action:

```zyn
fn_param = { name:identifier ~ ":" ~ ty:type_expr }
  -> TypedParameter { name: intern(name), ty: ty }
//                          ^^^^                ^^
//                    binding 'name'      binding 'ty'
```

### Common Action Patterns

| Pattern | Description |
|---------|-------------|
| `TypedX::Y { field: binding }` | Construct a TypedAST node from bindings |
| `-> binding` | Passthrough — return a binding directly |
| `-> fold_left_ops(first, rest)` | Build a left-associative binary expression tree |
| `-> prepend_list(first, rest)` | Combine first element + rest Vec into a Vec |
| `-> intern(name)` | Intern a string binding into the arena |
| `-> if cond { ... } else { ... }` | Branch on a boolean binding |

## Project Structure

A typical Zyn project looks like:

```
my-language/
├── grammar/
│   └── mylang.zyn      # Grammar definition
├── examples/
│   ├── hello.mylang    # Example source files
│   └── test.mylang
└── tests/
    └── parser_tests.rs # Test cases
```

## Next Steps

Now that you have a working grammar:

1. [Chapter 4](./04-grammar-syntax.md): Learn advanced grammar patterns
2. [Chapter 5](./05-semantic-actions.md): Master semantic actions
3. [Chapter 8](./08-zig-example.md): Study a complete real-world example
