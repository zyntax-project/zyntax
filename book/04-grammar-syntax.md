# Chapter 4: Grammar Syntax

Zyn uses a PEG (Parser Expression Grammar) syntax compatible with the Pest parser generator. This chapter covers all grammar constructs in detail.

## Rule Definitions

### Basic Rules

```zyn
// Normal rule - creates a parse node, handles whitespace between elements
rule_name = { pattern }

// Atomic rule - no whitespace handling, treats content as single token
identifier = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }

// Silent rule - matches but doesn't create a node
paren_expr = _{ "(" ~ expr ~ ")" }
```

### When to Use Each Type

| Rule Type | Use Case | Example |
|-----------|----------|---------|
| Normal `{ }` | Compound syntax structures | `if_stmt = { "if" ~ expr ~ block }` |
| Atomic `@{ }` | Tokens, literals, identifiers | `integer = @{ ASCII_DIGIT+ }` |
| Silent `_{ }` | Grouping without AST nodes | `paren_expr = _{ "(" ~ expr ~ ")" }` |

## Sequence and Choice

### Sequence (`~`)

Matches patterns in order:

```zyn
// Matches: "if" followed by expression followed by block
if_stmt = { "if" ~ expr ~ block }

// With optional parts
if_else = { "if" ~ expr ~ block ~ ("else" ~ block)? }
```

### Ordered Choice (`|`)

Tries alternatives in order, takes first match:

```zyn
// IMPORTANT: Order matters! Longer matches should come first
statement = { if_stmt | while_stmt | return_stmt | expr_stmt }

// Wrong order - "if" would match before "ifeq"
// keyword = { "if" | "ifeq" }  // BAD

// Correct order
keyword = { "ifeq" | "if" }     // GOOD
```

## Repetition

### Zero or More (`*`)

```zyn
// Matches: "", "a", "aa", "aaa", ...
statements = { statement* }

// With separator
args = { expr ~ ("," ~ expr)* }
```

### One or More (`+`)

```zyn
// Matches: "1", "12", "123", ...
digits = @{ ASCII_DIGIT+ }

// At least one statement required
block = { "{" ~ statement+ ~ "}" }
```

### Optional (`?`)

```zyn
// Optional else branch
if_stmt = { "if" ~ expr ~ block ~ else_branch? }

// Optional trailing comma
list = { "[" ~ (expr ~ ("," ~ expr)* ~ ","?)? ~ "]" }
```

## Predicates

### Negative Lookahead (`!`)

Succeeds if pattern does NOT match (doesn't consume input):

```zyn
// Identifier that's not a keyword
identifier = @{ !keyword ~ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }

// Keyword must not be followed by alphanumeric
keyword = @{ ("if" | "else" | "while") ~ !ASCII_ALPHANUMERIC }
```

### Positive Lookahead (`&`)

Succeeds if pattern matches (doesn't consume input):

```zyn
// Match only if followed by "("
function_name = { identifier ~ &"(" }
```

## Character Classes

### Built-in Classes

```zyn
ANY                 // Any single character
ASCII              // Any ASCII character (0x00-0x7F)
ASCII_DIGIT        // 0-9
ASCII_NONZERO_DIGIT // 1-9
ASCII_ALPHA        // a-z, A-Z
ASCII_ALPHANUMERIC // a-z, A-Z, 0-9
ASCII_ALPHA_LOWER  // a-z
ASCII_ALPHA_UPPER  // A-Z
ASCII_HEX_DIGIT    // 0-9, a-f, A-F
ASCII_OCT_DIGIT    // 0-7
ASCII_BIN_DIGIT    // 0-1
NEWLINE            // \n or \r\n
```

### Custom Ranges

```zyn
// Character range
lowercase = { 'a'..'z' }

// Multiple ranges
hex_digit = { '0'..'9' | 'a'..'f' | 'A'..'F' }
```

## String Matching

### Exact Match

```zyn
// Case-sensitive literal
if_keyword = { "if" }

// Multi-character operators
arrow = { "->" }
fat_arrow = { "=>" }
```

### Case Insensitive

```zyn
// Matches "if", "IF", "If", "iF"
if_keyword = { ^"if" }
```

## Special Rules

### WHITESPACE

Defines what counts as whitespace. Normal rules automatically skip this between elements:

```zyn
WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
```

### COMMENT

Defines comment syntax. Comments are skipped like whitespace:

```zyn
// Single-line comments
COMMENT = _{ "//" ~ (!"\n" ~ ANY)* ~ "\n"? }

// Block comments (non-nested)
COMMENT = _{ "/*" ~ (!"*/" ~ ANY)* ~ "*/" }

// Both styles
COMMENT = _{
    "//" ~ (!"\n" ~ ANY)* ~ "\n"?
  | "/*" ~ (!"*/" ~ ANY)* ~ "*/"
}
```

### SOI and EOI

Start and End of Input markers:

```zyn
program = { SOI ~ declarations ~ EOI }
```

## Handling Precedence

PEG handles operator precedence through grammar structure, not precedence tables.

### Left-Associative Operators

Build a chain of rules from lowest to highest precedence. Each level uses `fold_left_ops` with a companion `rest` rule that packages `(op, operand)` pairs using `make_pair`:

```zyn
// Lowest precedence: logical OR
expr = { e:logical_or }
  -> e

logical_or = { first:logical_and ~ rest:logical_or_rest* }
  -> fold_left_ops(first, rest)

logical_or_rest = { op:or_op ~ operand:logical_and }
  -> make_pair(op, operand)

or_op = @{ "or" }

logical_and = { first:comparison ~ rest:logical_and_rest* }
  -> fold_left_ops(first, rest)

logical_and_rest = { op:and_op ~ operand:comparison }
  -> make_pair(op, operand)

and_op = @{ "and" }

comparison = { first:addition ~ rest:comparison_rest* }
  -> fold_left_ops(first, rest)

comparison_rest = { op:comparison_op ~ operand:addition }
  -> make_pair(op, operand)

comparison_op = @{ "==" | "!=" | "<=" | ">=" | "<" | ">" }

addition = { first:multiplication ~ rest:addition_rest* }
  -> fold_left_ops(first, rest)

addition_rest = { op:add_op ~ operand:multiplication }
  -> make_pair(op, operand)

add_op = @{ "+" | "-" }

multiplication = { first:unary ~ rest:multiplication_rest* }
  -> fold_left_ops(first, rest)

multiplication_rest = { op:mul_op ~ operand:unary }
  -> make_pair(op, operand)

mul_op = @{ "*" | "/" }

// Highest precedence: unary then atoms
unary = { unary_with_op | atom }

unary_with_op = { op:unary_op ~ operand:atom }
  -> TypedExpression::Unary { op: op, operand: Box::new(operand) }

unary_op = @{ "-" | "!" }

atom = { number | identifier | paren_expr }

paren_expr = _{ "(" ~ expr ~ ")" }
```

`fold_left_ops` takes the `first` value and the `rest` Vec of `(op, operand)` pairs and builds a left-associative binary tree. For input `a + b - c` it produces `Binary(-, Binary(+, a, b), c)`.

### Right-Associative Operators

Use recursion for right-associativity:

```zyn
// Right-associative: a = b = c parses as a = (b = c)
assignment = { identifier ~ "=" ~ assignment | expr }
```

## Common Patterns

### Lists with Separators

```zyn
// Comma-separated, no trailing
args = { expr ~ ("," ~ expr)* }

// Comma-separated with optional trailing
items = { item ~ ("," ~ item)* ~ ","? }

// Empty allowed
opt_args = { (expr ~ ("," ~ expr)*)? }
```

### Keyword Protection

Prevent identifiers from matching keywords:

```zyn
keyword = @{
    ("if" | "else" | "while" | "for" | "return" | "fn" | "const" | "var")
    ~ !(ASCII_ALPHANUMERIC | "_")
}

identifier = @{ !keyword ~ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }
```

### String Literals with Escapes

```zyn
string_literal = @{ "\"" ~ string_inner* ~ "\"" }

string_inner = {
    !("\"" | "\\") ~ ANY  // Any char except quote or backslash
  | escape_seq
}

escape_seq = { "\\" ~ ("n" | "r" | "t" | "\\" | "\"" | "0") }
```

### Numbers

```zyn
// Integer with optional sign
integer = @{ "-"? ~ ASCII_DIGIT+ }

// Float
float = @{ "-"? ~ ASCII_DIGIT+ ~ "." ~ ASCII_DIGIT+ ~ exponent? }
exponent = @{ ("e" | "E") ~ ("+" | "-")? ~ ASCII_DIGIT+ }

// Hex literal
hex = @{ "0x" ~ ASCII_HEX_DIGIT+ }
```

## Debugging Tips

### Ambiguity

If parsing is slow or incorrect, check for:

1. **Left recursion** - PEG doesn't support it
   ```zyn
   // BAD: Left recursion
   expr = { expr ~ "+" ~ term | term }

   // GOOD: Use repetition
   expr = { term ~ ("+" ~ term)* }
   ```

2. **Ambiguous choices** - First match wins
   ```zyn
   // Matches "ifx" as "if" then "x"
   stmt = { "if" | identifier }

   // Use negative lookahead
   if_kw = @{ "if" ~ !ASCII_ALPHANUMERIC }
   ```

### Testing Rules

Test individual rules by making them the entry point:

```bash
# Test just the expression rule
echo "1 + 2 * 3" | zyntax parse --grammar calc.zyn --rule expr
```

## Next Steps

Now that you understand grammar syntax:

- [Chapter 5](./05-semantic-actions.md): Learn how to attach AST-building actions to rules
- [Chapter 8](./08-zig-example.md): See these patterns applied in a real language
