# Zyntax: Multi-Paradigm Compiler Infrastructure

> A high-performance, multi-paradigm compiler infrastructure with advanced type system features, tiered JIT compilation, and async/await runtime support.

[![CI](https://github.com/darmie/zyntax/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/darmie/zyntax/actions/workflows/ci.yml)
[![ZynML Tests](https://img.shields.io/github/actions/workflow/status/darmie/zyntax/ci.yml?branch=main&label=zynml%20tests)](https://github.com/darmie/zyntax/actions/workflows/ci.yml)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](./crates/zyn_parser/tests)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

---

## Quick Start

New to Zyntax? **Read [The Zyn Book](https://github.com/darmie/zyntax/wiki)** - a comprehensive guide covering everything from basic grammar syntax to building complete DSLs with runtime plugins.

```bash
# Build zyntax
cargo build --release

# Compile and run a Zig file using the zig.zyn grammar
./target/release/zyntax compile \
    --grammar crates/zyn_peg/grammars/zig.zyn \
    --source examples/hello.zig \
    --run

# Start an interactive REPL
./target/release/zyntax repl --grammar crates/zyn_peg/grammars/zig.zyn
```

---

## 🎯 What is Zyntax?

**Zyntax is a complete compiler infrastructure and runtime framework** designed for building high-performance, memory-safe programming languages. It provides:

- **Complete Compilation Pipeline**: TypedAST → HIR → Native Code

- **Tiered JIT Compilation**: 3-tier optimization (Baseline → Standard → Optimized)

- **Advanced Type System**: Generics, traits, lifetimes, dependent types

- **Async/Await Runtime**: Zero-cost futures with complete executor infrastructure

- **Production-Ready stdlib**: Vec, String, HashMap, Iterator (93/100 functions compile)

- **Multi-Backend**: Cranelift JIT (fast) + LLVM AOT/JIT (optimized, fully working)

- **HIR Builder API**: Type-safe, fluent interface for IR construction

Think of Zyntax as **LLVM + Rust's type system + V8's tiered compilation** - a complete foundation for building modern, high-performance programming languages.

---

## 🛠️ Zyntax CLI

The Zyntax command-line interface provides a unified compilation toolchain with multiple input format support:

```bash
# Build the CLI
cargo build --release

# Compile and run a program with JIT
zyntax compile input.json --jit

# Multiple input formats supported
zyntax compile program.zbc --format hir-bytecode -o output
zyntax compile --source code.calc --grammar calc.zyn --format zyn --jit
```

### CLI Features

- **Dual-Format Support**: Compile from JSON TypedAST or ZBC bytecode
- **JIT Execution**: Run programs directly with `--jit` flag
- **Multiple Backends**: Choose Cranelift (fast) or LLVM (optimized)
- **Format Auto-Detection**: Automatically detects input format from file extension
- **Rich Diagnostics**: Clear error messages with source location tracking

### Usage Examples

```bash
# Compile JSON TypedAST to executable (AOT)
zyntax compile program.json -o myapp

# JIT compile and run immediately
zyntax compile program.json --jit

# Compile ZBC bytecode format
zyntax compile program.zbc -o myapp

# Use LLVM backend for maximum optimization
zyntax compile program.json --backend llvm -o myapp

# JIT with LLVM backend
zyntax compile program.json --backend llvm --jit
```

---

## 📦 ZBC Bytecode Format

**ZBC (Zyntax ByteCode)** is a portable, architecture-independent bytecode format designed for efficient serialization and distribution of compiled programs.

### Key Features

- **Portable**: Architecture-independent binary format
- **Compact**: Efficient binary encoding with compression
- **Type-Preserving**: Maintains full type information for verification
- **Module-Based**: Supports separate compilation and linking
- **Version-Safe**: Built-in format versioning for compatibility

### Format Overview

```text
┌─────────────────────────────────────┐
│      ZBC File Header                │
│  - Magic number: 0x5A42_4300       │
│  - Version: 1.0                     │
│  - Metadata section offset          │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│      Type Definitions               │
│  - Structs, enums, traits          │
│  - Generic type parameters          │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│      Function Definitions           │
│  - Signature with parameter types   │
│  - HIR instruction stream           │
│  - SSA value numbering              │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│      Constant Pool                  │
│  - String literals                  │
│  - Numeric constants                │
└─────────────────────────────────────┘
```

### Use Cases

1. **Distribution**: Ship pre-compiled modules to users
2. **Caching**: Cache compiled TypedAST for faster rebuilds
3. **Cross-Platform**: Compile once, run on any Zyntax-supported platform
4. **Integration**: Load modules from multiple source languages

See [Bytecode Format Specification](./docs/BYTECODE_FORMAT_SPEC.md) for complete details.

---

## 📝 Zyn Grammar Format

**Zyn** is Zyntax's domain-specific language for defining custom programming language frontends. It extends PEG (Parsing Expression Grammar) syntax with JSON-based semantic actions that construct TypedAST nodes directly from parsed syntax.

### How Zyn Works with Zyntax

```text
┌─────────────────┐      ┌───────────────┐      ┌──────────────┐      ┌──────────────┐
│  Source Code    │  →   │     Zyn       │  →   │  TypedAST    │  →   │   Native     │
│  (your_lang.x)  │      │   Grammar     │      │   (JSON)     │      │   Binary     │
└─────────────────┘      └───────────────┘      └──────────────┘      └──────────────┘
                              ↓
                         ┌───────────────┐
                         │  Cranelift/   │
                         │  LLVM Backend │
                         └───────────────┘
```

Zyn grammars define both syntax (what patterns to match) and semantics (what AST nodes to create). This enables:

- **Custom Language Frontends**: Define your own language syntax and compile to native code
- **Runtime Grammar Loading**: No Rust recompilation needed—load grammars dynamically
- **Seamless Integration**: Output TypedAST that flows through Zyntax's HIR and backend pipeline
- **Interactive Development**: Test grammar changes instantly with the built-in REPL

### Quick Example

```zyn
@language {
    name: "Calculator",
    version: "1.0",
    file_extensions: [".calc"],
    entry_point: "main",
}

// Build a program: expression → return → function → program
program = { SOI ~ expr ~ EOI }
  -> TypedProgram {
      "commands": [
          { "define": "return_stmt", "args": { "value": "$1" }, "store": "ret" },
          { "define": "function", "args": { "name": "main", "params": [], "body": "$ret" } },
          { "define": "program", "args": { "declarations": ["$result"] } }
      ]
  }

// Binary expression with left-associative folding
expr = { term ~ ((add_op | sub_op) ~ term)* }
  -> TypedExpression {
      "fold_binary": { "operand": "term", "operator": "add_op|sub_op" }
  }

// Integer literal: get text, parse, create AST node
number = @{ ASCII_DIGIT+ }
  -> TypedExpression {
      "get_text": true,
      "parse_int": true,
      "define": "int_literal",
      "args": { "value": "$result" }
  }

add_op = { "+" } -> String { "get_text": true }
sub_op = { "-" } -> String { "get_text": true }
WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
```

### CLI Usage

```bash
# JIT compile and run with grammar
zyntax compile --source input.calc --grammar calc.zyn --format zyn --jit

# AOT compile to executable
zyntax compile --source code.mylang --grammar mylang.zyn --format zyn -o output

# Use LLVM backend for maximum optimization
zyntax compile --source code.mylang --grammar mylang.zyn --backend llvm -o output

# Interactive REPL mode
zyntax repl --grammar calc.zyn
```

### Interactive REPL

Start an interactive session to evaluate expressions on the fly:

```
$ zyntax repl --grammar examples/zpeg_test/calc.zyn
Zyntax REPL
Grammar: examples/zpeg_test/calc.zyn
✓ Calculator grammar loaded (11 rules)

Calculator> 2 + 3 * 4
[1] = 14
Calculator> (10 + 5) * 2
[2] = 30
Calculator> :help
REPL Commands:
  :help, :h, :?    Show this help message
  :quit, :q, :exit Exit the REPL
  :verbose, :v     Toggle verbose mode
  :clear, :c       Clear the screen
  :{              Start multi-line input (end with :})

Multi-line Input:
  - End a line with \ to continue on the next line
  - Lines with unclosed { automatically continue
  - Use :{ to start explicit multi-line mode, :} to execute
  - Press Ctrl+C to cancel multi-line input
Calculator> :quit
Goodbye!
```

### Key Features

| Feature | Description |
|---------|-------------|
| **`define`** | Create AST nodes with named arguments: `"define": "int_literal", "args": { "value": 42 }` |
| **`commands`** | Sequential command execution with `$result` chaining |
| **`store`** | Save intermediate results: `"store": "myvar"` → access as `"$myvar"` |
| **`fold_binary`** | Left-associative binary operator folding |
| **`get_text`** | Extract matched text content |
| **`get_child`** | Access child nodes by index or name |

### Available Node Types

- **Literals**: `int_literal`, `float_literal`, `string_literal`, `bool_literal`, `char_literal`
- **Expressions**: `variable`, `binary_op`, `unary_op`, `call_expr`, `method_call`, `field_access`, `index`, `array`, `struct_literal`, `cast`, `lambda`, `switch_expr`
- **Statements**: `let_stmt`, `assignment`, `return_stmt`, `if_stmt`, `while_stmt`, `for_stmt`, `break_stmt`, `continue_stmt`, `expression_stmt`, `block`
- **Declarations**: `function`, `param`, `program`
- **Types**: `primitive_type`, `pointer_type`, `array_type`, `named_type`, `function_type`
- **Patterns**: `literal_pattern`, `wildcard_pattern`, `range_pattern`, `identifier_pattern`, `struct_pattern`, `field_pattern`, `enum_pattern`, `array_pattern`, `pointer_pattern`, `error_pattern`, `switch_case`

See [Zyn Grammar Specification](./docs/ZYN_GRAMMAR_SPEC.md) for complete documentation.

---

## 🔌 Frontend Integrations

Zyntax supports multiple language frontends through its TypedAST intermediate representation. Create your own programming language or integrate existing ones:

### ✅ Zyn - Create Your Own Language

Define a custom language with Zyn grammar and compile to native code:

```bash
# Write your language grammar (mylang.zyn)
# Then compile source files written in your language
zyntax compile --source program.mylang --grammar mylang.zyn --format zyn --run

# Or use interactive REPL to test your language
zyntax repl --grammar mylang.zyn
```

**Output Targets:**

| Target | Description |
|--------|-------------|
| **TypedAST** | JSON intermediate representation for tooling integration |
| **Bytecode** | Portable `.zbc` format for distribution and caching |
| **JIT** | Cranelift-powered just-in-time compilation |
| **AOT** | LLVM-based ahead-of-time native executables |

**Status:** ✅ **Production-ready** - Full compilation pipeline with REPL support

### ✅ Haxe Integration (via Reflaxe)

Compile Haxe code to native executables using the [reflaxe.zyntax](./reflaxe.zyntax/README.md) backend:

```bash
# Install dependencies
haxelib install reflaxe 4.0.0-beta
haxelib dev reflaxe.zyntax ./reflaxe.zyntax

# Compile Haxe to native
haxe -lib reflaxe.zyntax -main Main -D zyntax-output=out
zyntax compile out/*.json -o myprogram --run
```

**Status:** 🚧 In development - JSON AST generation complete, HIR conversion in progress

See [Haxe Integration Guide](./docs/HAXE_INTEGRATION.md) for details.

### ✅ HIR Builder API - Programmatic Code Generation

Build HIR modules directly from Rust code. Perfect for:
- **Code generators** that emit Zyntax IR from other tools
- **DSL implementations** that construct code at runtime
- **Compiler backends** for languages with existing parsers
- **Testing and prototyping** new language features

```rust
use zyntax_compiler::hir_builder::HirBuilder;
use zyntax_typed_ast::arena::AstArena;

let mut arena = AstArena::new();
let mut builder = HirBuilder::new("hello", &mut arena);

// fn main() -> i32 { return 42; }
let i32_ty = builder.i32_type();
let main_fn = builder.begin_function("main")
    .returns(i32_ty.clone())
    .build();

builder.set_current_function(main_fn);
let entry = builder.entry_block();
builder.set_insert_point(entry);

let value = builder.const_i32(42);
builder.ret(value);

let module = builder.finish();

// Compile to native code with JIT
let mut backend = CraneliftBackend::new().unwrap();
backend.compile_module(&module).unwrap();

// Execute!
let fn_ptr = backend.get_function_ptr(main_fn).unwrap();
let result = unsafe {
    let f: fn() -> i32 = std::mem::transmute(fn_ptr);
    f()
};
assert_eq!(result, 42);
```

**Status:** ✅ **Production-ready** - Full SSA-based IR construction with type system integration

### ✅ Embedding SDK - Embed Zyntax in Your Application

Use `zyntax_embed` to embed the Zyntax JIT runtime in Rust applications:

```rust
use zyntax_embed::{ZyntaxRuntime, LanguageGrammar};

// Load a grammar and create runtime
let grammar = LanguageGrammar::compile_zyn_file("grammars/zig.zyn")?;
let mut runtime = ZyntaxRuntime::new()?;

// Compile and run code
runtime.compile_with_grammar(&grammar, r#"
    pub fn add(a: i32, b: i32) i32 {
        return a + b;
    }
"#)?;

let result: i32 = runtime.call("add", &[10.into(), 32.into()])?;
println!("Result: {}", result); // 42
```

**Register native functions:**

```rust
extern "C" fn native_print(x: i32) { println!("{}", x); }

let symbols = &[("native_print", native_print as *const u8)];
let mut runtime = ZyntaxRuntime::with_symbols(symbols)?;
```

**Features:**
- Multi-tier JIT compilation (Cranelift baseline → LLVM optimized)
- Bidirectional Rust ↔ Zyntax value conversion
- External function registration for native interop
- ZRTL plugin loading for runtime libraries
- Async/await with Promise API

**Status:** ✅ **Production-ready** - Full embedding API with grammar support

See [Embedding SDK Documentation](./book/12-embedding-sdk.md) for complete guide.

### 🔜 Other Integrations

- **Whirlwind** - Direct AST adapter (in progress)
- **Custom JSON** - Generate TypedAST JSON directly from any toolchain

---

## 🚀 Development Setup

```bash
# Clone the repository
git clone https://github.com/darmie/zyntax.git
cd zyntax

# Build the compiler
cargo build --release

# Run tests
cargo test

# Run comprehensive end-to-end tests
cargo test --test end_to_end_comprehensive
cargo test --test end_to_end_simple
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Language Frontends                        │
│         (Your language's parser → TypedAST)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Zyntax TypedAST Layer                          │
│  • Multi-paradigm type checking (structural/nominal/gradual)│
│  • Generics, traits, lifetimes, dependent types             │
│  • Advanced analysis (ownership, escape, lifetimes)         │
│  • Rich diagnostics with span tracking                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           High-Level IR (HIR) + Lowering                    │
│  • SSA-based intermediate representation                    │
│  • Control flow graph (CFG) with dominance analysis         │
│  • Type-erased, platform-agnostic                          │
│  • HIR Builder API for programmatic construction            │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    ▼         ▼
        ┌─────────────────┐ ┌──────────────────┐
        │ Cranelift JIT   │ │   LLVM AOT       │
        │ (Baseline/Fast) │ │ (Optimized)      │
        └────────┬────────┘ └────────┬─────────┘
                 │                   │
                 └───────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │   Native Machine     │
              │        Code          │
              └──────────────────────┘
```

### Key Components

**📦 Crates:**
- `typed_ast/` - Multi-language typed AST with rich type system
- `compiler/` - HIR generation, lowering, backend code generation, and async runtime
- `whirlwind_adapter/` - Whirlwind language integration

**🎨 Key Features:**
- **Tiered Compilation**: Hot code recompiles with better optimizations
- **Zero-Cost Abstractions**: Traits, generics, closures compile to native code
- **Async/Await**: Complete runtime with task scheduling and waker infrastructure
- **Memory Safety**: Ownership, borrowing, lifetimes enforced at compile time

---

## 🔥 Current Status

### Test Results (98.6% Pass Rate)
```
✅ 280/284 tests passing
✅ All end-to-end comprehensive tests passing (9/9)
✅ All end-to-end simple tests passing (5/5)
✅ Standard library: 93/100 functions compile successfully
```

### What's Working

#### ✅ Core Compiler Pipeline
- Complete TypedAST → HIR → Native code pipeline
- Full SSA construction with phi nodes
- Control flow graph analysis (dominators, post-dominators)
- Dead code elimination

#### ✅ Type System
- **Generics**: Type parameters with bounds `fn foo<T: Clone>(x: T)`
- **Traits**: Interface definitions with associated types
- **Lifetimes**: Borrow checker with lifetime inference
- **Dependent Types**: Basic refinement types and indexed families
- **Multi-paradigm**: Structural, nominal, and gradual typing

#### ✅ Language Features
- Functions with parameters and returns
- Basic arithmetic (`+`, `-`, `*`, `/`)
- Function calls (including recursive)
- Local variables with stack allocation
- Switch expressions with pattern matching (literals, wildcards, ranges, structs, enums, errors)
- Async/await syntax and runtime

#### ✅ Tiered JIT Compilation
- **Tier 1**: Cranelift baseline JIT (fast compilation)
- **Tier 2**: Cranelift optimized (moderate optimizations)
- **Tier 3**: LLVM JIT (aggressive optimizations for hot paths) - **fully working**
- Runtime profiling with atomic execution counters
- Hot-path detection and automatic recompilation
- LLVM AOT backend for native executables (functions, structs, generics)

#### ✅ Standard Library
- `Vec<T>` - Dynamic array with push/pop/indexing
- `String` - UTF-8 string with manipulation methods
- `HashMap<K,V>` - Hash table with insert/get/remove
- `Iterator` - Lazy iterator trait with 50+ adapters

#### ✅ Async Runtime
- Complete executor with task scheduling
- Waker infrastructure for efficient event-driven code
- Parameter capture in async state machines
- Integration with tiered JIT compilation

---

## 📚 Documentation

### The Zyn Book

**[The Zyn Book](https://github.com/darmie/zyntax/wiki)** is the comprehensive guide to Zyntax:

1. [Introduction](https://github.com/darmie/zyntax/wiki/01-Introduction) - What is Zyn and why use it?
2. [Getting Started](https://github.com/darmie/zyntax/wiki/02-Getting-Started) - Your first Zyn grammar
3. [Using the CLI](https://github.com/darmie/zyntax/wiki/03-Using-the-CLI) - Compilation, execution, and REPL
4. [Grammar Syntax](https://github.com/darmie/zyntax/wiki/04-Grammar-Syntax) - PEG-based grammar rules
5. [Semantic Actions](https://github.com/darmie/zyntax/wiki/05-Semantic-Actions) - JSON command blocks
6. [The TypedAST](https://github.com/darmie/zyntax/wiki/06-The-TypedAST) - Understanding the target representation
7. [TypedAST Builder](https://github.com/darmie/zyntax/wiki/07-TypedAST-Builder) - Building AST nodes programmatically
8. [Complete Example: Zig](https://github.com/darmie/zyntax/wiki/08-Zig-Example) - A real-world grammar walkthrough
9. [Reference](https://github.com/darmie/zyntax/wiki/09-Reference) - Command reference and API
10. [Packaging & Distribution](https://github.com/darmie/zyntax/wiki/10-Packaging-Distribution) - ZPack format, AOT linking
11. [HIR Builder](https://github.com/darmie/zyntax/wiki/11-HIR-Builder) - Building HIR directly for custom backends
12. [Embedding SDK](https://github.com/darmie/zyntax/wiki/12-Embedding-SDK) - Embedding Zyntax in Rust applications
13. [Async Runtime](https://github.com/darmie/zyntax/wiki/13-Async-Runtime) - Promise-based async native runtime
14. [Runtime Plugins](https://github.com/darmie/zyntax/wiki/14-Runtime-Plugins) - ZRTL standard library plugins
15. [Building DSLs](https://github.com/darmie/zyntax/wiki/15-Building-DSLs) - Creating domain-specific languages
16. [Tutorial: Image Pipeline DSL](https://github.com/darmie/zyntax/wiki/16-Image-Pipeline-DSL) - Step-by-step DSL tutorial

### Technical Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - Complete system architecture
- **[HIR Builder Example](docs/HIR_BUILDER_EXAMPLE.md)** - How to construct HIR programmatically
- **[Async Runtime Design](docs/ASYNC_RUNTIME_DESIGN.md)** - Async/await internals
- **[Bytecode Spec](docs/BYTECODE_FORMAT_SPEC.md)** - Bytecode serialization format
- **[Backlog](BACKLOG.md)** - Development roadmap and tasks
- **[Production Status](PRODUCTION_READY_STATUS.md)** - Detailed feature matrix

---

## 🎯 Use Cases

### 1. Language Implementation
Build a new programming language by targeting Zyntax:
```rust
// Your language parser
YourLanguage → TypedAST → HIR → Native Code
```

**Example**: Implement a Python-like language with JIT compilation and type inference.

### 2. Domain-Specific Languages (DSLs)
Create high-performance DSLs for specific domains:
- Game scripting languages
- Data processing pipelines
- Configuration languages with validation

### 3. Cross-Language Interop
Use Zyntax as a common compilation target:
```
Multiple Languages → TypedAST → Shared Runtime
```

### 4. Research Platform
Experiment with advanced type system features:
- Effect systems
- Dependent types
- Linear types
- Algebraic effects

---

## 🚧 Roadmap

See [BACKLOG.md](BACKLOG.md) for detailed tasks.

### Q4 2025 (Current): Core Stabilization ✅ COMPLETE

- ✅ Zig parser with full control flow support (continue, break, while loops)
- ✅ Fix SSA variable reads for unsealed blocks (continue statement bug)
- ✅ Logical operators with short-circuit evaluation
- ✅ Array types, indexing, and array index assignment
- ✅ String literals (lowered to global `*i8`)
- ✅ 71/71 Zig E2E tests passing (100%)
- ✅ Zig-style error handling (try/catch/orelse on error unions)
- ✅ Pattern matching (if let, switch, Some/None/Ok/Err)
- ✅ Generic functions with monomorphization
- ✅ Switch expressions with multi-case patterns and else clause
- ✅ Pattern matching grammar (literals, wildcards, ranges, structs, enums, errors, pointers)

### Q1 2026: Production Features

- ✅ LLVM AOT/JIT backend core complete (functions, structs, generics, control flow)
- ✅ Switch expression pattern matching in LLVM backend
- 🔄 Haxe-style exception handling (throw/catch/finally with stack unwinding)
- 🔄 Complete I/O and networking standard library
- 🔄 String operations (needs stdlib integration via plugin system)

### Q2 2026: Ecosystem & Integration

- 🔄 Complete Reflaxe/Haxe integration
- 🔄 Run Haxe standard library through Zyntax
- 🔄 Performance benchmarking vs existing targets
- 🔄 100% test pass rate

### Q3 2026: Developer Experience

- 🔄 Language Server Protocol (LSP) implementation
- 🔄 Package manager
- 🔄 Comprehensive documentation and tutorials
- 🔄 VSCode/IntelliJ integration

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Pick a task** from [BACKLOG.md](BACKLOG.md)
2. **Check documentation** in [docs/](docs/)
3. **Run tests** to understand the system: `cargo test`
4. **Implement incrementally** with test coverage
5. **Submit a PR** with clear description

### Development Setup

```bash
# Install Rust (1.70+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/yourusername/zyntax.git
cd zyntax
cargo build

# Run tests
cargo test --workspace

# Run specific test suites
cargo test --package zyntax_compiler
cargo test --test end_to_end_comprehensive
```


---

## 📊 Performance

### Compilation Speed
- **Baseline JIT (Cranelift)**: <1ms for small functions
- **Optimized JIT (LLVM)**: 10-100ms for hot paths
- **Tiered compilation**: Amortizes optimization cost over runtime

### Runtime Performance
- **Zero-cost abstractions**: Generics and traits compile to direct calls
- **Async overhead**: ~100ns per await point
- **Memory safety**: No runtime overhead for ownership checks

### Comparison (Estimated)
```
Language/VM          | Startup | Hot Code | Memory Safety
---------------------|---------|----------|---------------
Zyntax (Tier 1)     | ~1ms    | 2-3x C   | Compile-time
Zyntax (Tier 3)     | ~50ms   | ~C speed | Compile-time
V8 (JavaScript)     | ~50ms   | ~C speed | Runtime GC
HotSpot (Java)      | ~100ms  | ~C speed | Runtime GC
PyPy (Python)       | ~200ms  | 5-10x C  | Runtime GC
CPython (Python)    | ~50ms   | 50-100x C| Runtime GC
```

*Note: Benchmarks are preliminary. Real-world performance depends on workload.*

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Cranelift** - Fast, secure code generator
- **LLVM** - Optimizing compiler infrastructure
- **Rust** - Type system inspiration and implementation language
- **Haxe** - Multi-target compilation model
- **V8/HotSpot** - Tiered compilation strategies

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/zyntax/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/zyntax/discussions)

---

**Built with 🦀 Rust** | **Powered by Cranelift & LLVM** | **Production-Ready Core**
