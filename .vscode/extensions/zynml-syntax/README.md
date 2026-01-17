# ZynML Language Support

Syntax highlighting for ZynML - the Machine Learning DSL for Zyntax.

## Features

- Syntax highlighting for `.zynml` and `.zyn` files
- Bracket matching and auto-closing
- Comment toggling (`//` and `/* */`)
- Code folding

## Highlighted Constructs

### Keywords
- Control flow: `if`, `else`, `elif`, `match`, `case`, `for`, `while`, `return`
- Declarations: `def`, `fn`, `struct`, `enum`, `trait`, `impl`, `type`, `let`, `mut`
- Async: `async`, `await`
- Effects: `effect`, `handler`, `where`

### Types
- Primitives: `i64`, `f64`, `bool`, `str`, `String`
- Built-ins: `Option`, `Result`, `List`, `HashMap`, `Tensor`, `Iterator`
- Type variables (capitalized identifiers)

### Literals
- Numbers with optional suffixes: `1000ms`, `5s`, `3.14`
- Strings and f-strings: `"hello"`, `f"Hello {name}"`
- Booleans: `true`, `false`
- Option/Result: `Some`, `None`, `Ok`, `Err`

### Operators
- Pipe: `|>`
- Arrows: `=>`, `->`, `:`
- Null coalesce: `??`
- Optional: `?`
- Path: `::`

### Annotations
- `@effect(IO)`
- `@device(GPU(0))`
- `@jit`

## Installation

### Local Development

1. Copy or symlink this folder to your VS Code extensions directory:
   - macOS/Linux: `~/.vscode/extensions/`
   - Windows: `%USERPROFILE%\.vscode\extensions\`

2. Reload VS Code

### From Workspace

Add to your `.vscode/settings.json`:

```json
{
  "files.associations": {
    "*.zynml": "zynml",
    "*.zyn": "zynml"
  }
}
```

## Example

```zynml
// Define a struct with generics
struct Point<T> {
    x: T,
    y: T
}

// Implement Display trait
impl<T> Display for Point<T>
    where T: Display
{
    def to_string(self): String {
        f"({self.x}, {self.y})"
    }
}

// Function with pattern matching
def process(data: List<i64>): ?i64 {
    match data.first() {
        case Some(v) { Some(v * 2) }
        case None() { None }
    }
}

// Async function with effect
@effect(IO)
async def fetch_and_process(url: String): Tensor {
    data = await http.get(url)
    data |> parse() |> normalize() |> transform()
}
```

## License

Apache-2.0
