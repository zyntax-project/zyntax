# Zyntax CLI

Command-line interface for the Zyntax compiler. Supports multiple input formats and compilation backends.

## Installation

```bash
cargo build --package zyntax_cli --release
```

The binary will be available at `target/release/zyntax`.

## Usage

```bash
zyntax compile [OPTIONS] <INPUT>...
```

### Options

- `-o, --output <OUTPUT>` - Output file path
- `-b, --backend <BACKEND>` - Backend to use: `jit` (default), `llvm`
- `-O, --opt-level <LEVEL>` - Optimization level: 0-3 (default: 2)
- `-f, --format <FORMAT>` - Input format: `auto` (default), `typed-ast`, `hir-bytecode`, `zyn`
- `-s, --source <SOURCE>` - Source code file (for ZynPEG grammar-based compilation)
- `-g, --grammar <GRAMMAR>` - ZynPEG grammar file (.zyn)
- `--run` - Run the compiled program immediately (JIT only)
- `-v, --verbose` - Verbose output

## Input Formats

The Zyntax CLI supports three input formats:

### 1. ZynPEG Grammar-Based (`.zyn` grammar + source file)

Compile any language using a ZynPEG grammar definition. This is the most flexible format for custom language compilation.

**When to use:**
- Custom language frontends with ZynPEG grammar
- Zig-syntax compilation
- Prototyping new language syntax

**Example:**

```bash
# Compile source using a grammar file
zyntax compile --source my_code.zig --grammar zig.zyn --format zyn -o output --run

# Or let the CLI auto-detect when both --source and --grammar are provided
zyntax compile --source fibonacci.zig --grammar crates/zyn_parser/src/zig.pest --run
```

**Currently Supported Grammars:**
- `zig` - Zig language subset

**Adding New Grammars:**
1. Create a `.zyn` grammar file in `crates/zyn_peg/grammars/`
2. Run the ZynPEG generator to produce pest + AST builder
3. Add the parser to `zyn_parser` crate
4. Register the parser in `crates/zyntax_cli/src/formats/zyn_grammar.rs`

### 2. HIR Bytecode (`.zbc` files)

Precompiled HIR modules in binary format. This is the most efficient format for production use.

**When to use:**
- Direct HIR construction via HirBuilder API
- Caching compiled modules
- Fast compilation pipeline
- Binary distribution

**Example:**

```bash
# Compile HIR bytecode to native
zyntax compile module.zbc -o myprogram --run

# Auto-detect format
zyntax compile module.zbc --backend jit
```

**Generating HIR bytecode:**

```rust
use zyntax_compiler::hir_builder::HirBuilder;
use zyntax_compiler::bytecode::{Format, serialize_module_to_file};
use zyntax_typed_ast::AstArena;

let mut arena = AstArena::new();
let mut builder = HirBuilder::new("mymodule", &mut arena);

// Build your HIR...
let i32_ty = builder.i32_type();
let main_fn = builder.begin_function("main")
    .returns(i32_ty.clone())
    .build();
// ... construct function body ...

let module = builder.finish();

// Serialize to .zbc file
serialize_module_to_file(&module, Format::Postcard, std::path::Path::new("module.zbc"))?;
```

### 3. TypedAST JSON (`.json` files)

Language-agnostic typed AST in JSON format, for frontends that build a typed AST outside Zyntax.

**When to use:**
- Custom language frontends
- Debugging and inspection
- Cross-language interop

**Example:**

```bash
# Compile TypedAST JSON
zyntax compile output/*.json -o myprogram

# Explicit format specification
zyntax compile output/ --format typed-ast --backend jit --run
```

**JSON Structure:**

```json
{
  "type": "class_declaration",
  "name": "Main",
  "methods": [
    {
      "type": "method",
      "name": "main",
      "params": [],
      "return_type": {"type": "primitive", "kind": "Unit"},
      "body": {
        "type": "block",
        "statements": [...]
      },
      "is_static": true
    }
  ]
}
```

## Examples

### Compile Zig Source with Grammar

```bash
# Using the built-in Zig grammar
zyntax compile --source fibonacci.zig --grammar zig.zyn --format zyn --run

# Verbose output to see compilation steps
zyntax compile -v --source add.zig --grammar zig.zyn --format zyn --run
```

Output:
```
info: Input format: ZynGrammar
info: Grammar file: "zig.zyn"
info: Source file: "fibonacci.zig"
info: Read 150 bytes of source code
info: Read 8500 bytes of grammar
info: Using built-in Zig parser
info: Parsed to TypedProgram with 2 declarations
info: Lowering complete
info: Monomorphization complete
info: Lowered to HIR with 2 functions
info: Compiling with Jit backend (opt level 2)...
success: Compilation successful
```

### Compile HIR Bytecode

```bash
# Compile with JIT backend
zyntax compile module.zbc --backend jit --run

# Compile with LLVM AOT (when implemented)
zyntax compile module.zbc --backend llvm -o myprogram -O3
```

### Verbose Compilation

```bash
zyntax compile -v output/*.json -o myprogram
```

Output:
```
info: Input format: TypedAst
info: Found 3 JSON file(s)
info: Parsing output/Main.json
info: Parsing output/Utils.json
info: Parsing output/Config.json
info: Building HIR...
info: Compiling with jit backend (opt level 2)...
info: Compiling functions...
success: Compilation successful
```

### Auto-detect Format

```bash
# Detects .zbc extension -> HIR bytecode
zyntax compile module.zbc

# Detects .json extension -> TypedAST
zyntax compile output/*.json

# Auto-detect grammar mode when --source and --grammar provided
zyntax compile --source code.zig --grammar zig.zyn

# Directory with mixed files -> error (specify --format)
zyntax compile . --format typed-ast
```

## Backends

### Cranelift JIT (default)

Fast JIT compilation for development and rapid iteration.

**Characteristics:**
- Very fast compilation
- Good runtime performance
- No external dependencies
- Supports tiered compilation

**Usage:**

```bash
zyntax compile input.zbc --backend jit --run
```

### LLVM AOT

Ahead-of-time compilation with aggressive optimizations for production.

**Characteristics:**
- Slower compilation
- Excellent runtime performance
- Requires LLVM installation
- Full optimization pipeline

**Status:** In development

**Usage:**

```bash
zyntax compile input.zbc --backend llvm -o myprogram -O3
```

## Exit Codes

- `0` - Success
- `1` - Compilation error
- `2` - Invalid arguments

## Environment Variables

- `RUST_LOG` - Set logging level (e.g., `RUST_LOG=debug zyntax compile ...`)

## See Also

- [ZynPEG Grammar Conventions](../zyn_peg/GRAMMAR_CONVENTIONS.md)
- [HIR Builder API](../compiler/README.md)
- [TypedAST Documentation](../typed_ast/README.md)
- [Bytecode Specification](../../docs/BYTECODE_FORMAT_SPEC.md)
