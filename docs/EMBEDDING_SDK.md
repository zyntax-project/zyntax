# Zyntax Embedding SDK

This document provides architectural documentation for the `zyntax_embed` crate, which enables embedding the Zyntax JIT compiler in Rust applications.

## Overview

The Zyntax Embedding SDK provides a complete solution for:

1. **Language Grammar Interface** - Parse any language with a ZynPEG grammar
2. **JIT Compilation** - Compile to native code at runtime
3. **Multi-Tier Optimization** - Automatic optimization of hot code paths
4. **Value Interop** - Bidirectional conversion between Rust and Zyntax types
5. **Async Support** - Promise-based async operations
6. **Plugin System** - Load ZRTL native runtime libraries

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Application                             │
├─────────────────────────────────────────────────────────────────────┤
│                        zyntax_embed                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ LanguageGrammar │  │  ZyntaxRuntime  │  │   TieredRuntime     │ │
│  │                 │  │                 │  │                     │ │
│  │ • load()        │  │ • compile_with_ │  │ • development()     │ │
│  │ • compile_zyn() │  │   grammar()     │  │ • production()      │ │
│  │ • parse()       │  │ • call()        │  │ • optimize_function │ │
│  │ • parse_to_json │  │ • call_async()  │  │ • statistics()      │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘ │
│           │                    │                      │            │
│  ┌────────▼────────────────────▼──────────────────────▼──────────┐ │
│  │                    Value Conversion Layer                      │ │
│  │  ZyntaxValue  |  ZyntaxString  |  ZyntaxArray  |  FromZyntax   │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       zyn_peg (Grammar Runtime)                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │   ZpegModule    │  │   pest_vm       │  │  CommandInterpreter │ │
│  │                 │  │                 │  │                     │ │
│  │ • pest_grammar  │──│ • parse()       │──│ • execute_rule()    │ │
│  │ • rules         │  │                 │  │ • host functions    │ │
│  └─────────────────┘  └─────────────────┘  └──────────┬──────────┘ │
└──────────────────────────────────────────────────────│────────────┘
                                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     zyntax_typed_ast                                │
│                                                                     │
│    TypedProgram → TypedDeclaration → TypedExpression/Statement      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     zyntax_compiler                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │   Lowering      │  │  HIR Module     │  │  Cranelift/LLVM     │ │
│  │                 │  │                 │  │                     │ │
│  │ TypedAST → HIR  │──│ • functions     │──│ • compile()         │ │
│  │                 │  │ • types         │  │ • get_function_ptr  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
crates/zyntax_embed/src/
├── lib.rs          # Public API exports
├── grammar.rs      # LanguageGrammar - grammar loading and parsing
├── runtime.rs      # ZyntaxRuntime, TieredRuntime - JIT compilation
├── value.rs        # ZyntaxValue - runtime value representation
├── convert.rs      # FromZyntax, IntoZyntax traits
├── string.rs       # ZyntaxString - zero-copy string wrapper
├── array.rs        # ZyntaxArray - zero-copy array wrapper
└── error.rs        # Error types
```

## Key Components

### LanguageGrammar

The `LanguageGrammar` struct provides the interface for parsing source code:

```rust
pub struct LanguageGrammar {
    module: Arc<ZpegModule>,      // Compiled grammar
    vm: Arc<Mutex<Option<PestVmCache>>>,  // Cached pest VM
}
```

**Responsibilities:**
- Load/compile ZynPEG grammars
- Parse source code to TypedProgram
- Cache compiled pest VM rules for performance
- Provide grammar metadata (name, version, extensions)

**Flow:**
```
.zyn source → ZynGrammarParser → ZynGrammar → ZpegCompiler → ZpegModule
                                                                 ↓
source code → pest_vm → parse tree → CommandInterpreter → TypedProgram
```

### ZyntaxRuntime

Single-tier JIT runtime for basic use cases:

```rust
pub struct ZyntaxRuntime {
    backend: CraneliftBackend,
    function_ids: HashMap<String, HirId>,
    config: CompilationConfig,
    external_functions: HashMap<String, ExternalFunction>,
    import_resolvers: Vec<ImportResolverCallback>,
}
```

**Key Methods:**
- `compile_with_grammar()` - Full pipeline from source to native code
- `compile_module()` - Compile pre-built HIR module
- `call()` / `call_raw()` - Execute compiled functions
- `call_async()` - Execute async functions, returns Promise
- `hot_reload()` - Replace function implementation

### TieredRuntime

Multi-tier JIT with automatic optimization:

```rust
pub struct TieredRuntime {
    backend: TieredBackend,
    function_ids: HashMap<String, HirId>,
    config: TieredConfig,
}
```

**Tiers:**
| Tier | Backend | Trigger |
|------|---------|---------|
| 0 (Baseline) | Cranelift fast | Initial compilation |
| 1 (Standard) | Cranelift optimized | warm_threshold calls |
| 2 (Optimized) | Cranelift/LLVM | hot_threshold calls |

**Optimization Flow:**
```
Function called → Increment counter → Check threshold
                                           ↓
                          (threshold reached)
                                           ↓
                    Queue for background optimization
                                           ↓
                    Background worker recompiles
                                           ↓
                    Atomically swap function pointer
```

### ZyntaxValue

Universal runtime value type:

```rust
pub enum ZyntaxValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<ZyntaxValue>),
    Struct { name: String, fields: HashMap<String, ZyntaxValue> },
    Enum { name: String, variant: String, data: Box<ZyntaxValue> },
    Function { name: String, ptr: *const u8 },
    Opaque { type_meta: TypeMeta, data: Vec<u8> },
}
```

### ZyntaxPromise

Async operation handle:

```rust
pub struct ZyntaxPromise {
    state: Arc<Mutex<PromiseInner>>,
}

pub enum PromiseState {
    Pending,
    Ready(ZyntaxValue),
    Failed(String),
}
```

**Implements:**
- `poll()` - Check completion state
- `then()` / `catch()` - Chain operations
- `await_result()` - Block until complete
- `Future` trait - Use with async/await

## Compilation Pipeline

### Full Pipeline (compile_with_grammar)

```
Source Code
     │
     ▼ (LanguageGrammar.parse)
TypedProgram
     │
     ▼ (LoweringContext.lower_program)
HirModule
     │
     ▼ (monomorphize_module)
HirModule (monomorphized)
     │
     ▼ (CraneliftBackend.compile_module)
Native Code
     │
     ▼ (finalize_definitions)
Function Pointers
```

### Manual Pipeline

For advanced use cases requiring custom processing:

```rust
// 1. Parse
let typed_program = grammar.parse(source)?;

// 2. Custom processing on TypedProgram
// ... modify typed_program ...

// 3. Lower to HIR
let hir = compile_to_hir(&typed_program, registry, config)?;

// 4. Custom HIR transformations
// ... modify hir ...

// 5. Compile
runtime.compile_module(&hir)?;
```

## Import Resolution

Import resolvers are called in order until one succeeds:

```rust
pub type ImportResolverCallback = Box<dyn Fn(&str) -> Result<Option<String>, String> + Send + Sync>;
```

**Resolution Flow:**
```
import "foo.bar"
       │
       ▼
Resolver 1 → None (not found)
       │
       ▼
Resolver 2 → Some(source) ✓
       │
       ▼
Parse and compile imported module
```

**Built-in Resolvers:**
- Callback-based (custom logic)
- Filesystem-based (directory + extension)

### Host modules

A host exposes its own classes and functions to a program through `TieredRuntime::add_host_modules`. The resolver answers a module path with a `zyntax_embed::host::HostModule`: its classes (constructor, methods, fields, statics) and functions, typed with `HostType`. Calls run through the `zyntax_embed::foreign::Foreign` the host installed with `foreign::install`, the same protocol the Python and Lua frontends use.

An import names its module by its whole path, longest first. `import game.Prompt` is module `game.Prompt`, or else member `Prompt` of module `game`; `from game import Prompt` is module `game.Prompt`, or else member `Prompt` of `game`. An import no host module answers goes to the other resolvers.

Each class `C` becomes a type, its value the foreign object's box:

| Host member | Call |
|---|---|
| constructor | `C::new(args)` |
| static method | `C::m(args)` |
| method | `c.m(args)` |
| field `x` | `c.x()`, `c.set_x(v)` |
| static field `x` | `C::x()`, `C::set_x(v)` |
| module function `f` | `f(args)` |

`Bool`, `Int`, `Float` and `Str` cross as the language's own values; a class of the same module crosses as its type and any other host object, or `Dynamic`, as `Any`. `Bytes` and function types are refused. Each call is checked against the declared signature when the program compiles. A call the host fails returns its type's zero and leaves the error for `foreign::take_error`.

## ZRTL Plugin System

Native runtime libraries follow the ZRTL format:

```rust
// Plugin exports
#[no_mangle]
pub static _zrtl_info: ZrtlInfo;      // Plugin metadata
#[no_mangle]
pub static _zrtl_symbols: [ZrtlSymbol]; // Symbol table
```

**Symbol naming convention:** `$TypeName$method_name`

**Loading:**
```rust
runtime.load_plugin("path/to/plugin.zrtl")?;
// Symbols now available for FFI calls
```

## Memory Layout

### ZyntaxString
```
┌───────────┬────────────────────────┐
│ i32 len   │     UTF-8 bytes        │
└───────────┴────────────────────────┘
```

### ZyntaxArray
```
┌───────────┬───────────┬────────────────────────┐
│ i32 cap   │ i32 len   │      elements          │
└───────────┴───────────┴────────────────────────┘
```

## Thread Safety

| Component | Thread Safety |
|-----------|---------------|
| LanguageGrammar | Shareable (Arc), VM cache uses Mutex |
| ZyntaxRuntime | Not shareable, use per-thread |
| TieredRuntime | Function pointers atomically swapped |
| ZyntaxPromise | Thread-safe via internal Mutex |
| ZyntaxValue | Owned, not thread-safe |

## Error Types

```rust
pub enum GrammarError {
    LoadError(String),
    ParseError(String),
    CompileError(String),
    SourceParseError(String),
    AstBuildError(String),
    IoError(std::io::Error),
    JsonError(serde_json::Error),
}

pub enum RuntimeError {
    Compilation(CompilerError),
    FunctionNotFound(String),
    Conversion(ConversionError),
    Execution(String),
    Promise(String),
    ArgumentCount { expected: usize, got: usize },
}
```

## Performance Considerations

1. **Grammar Caching**: `LanguageGrammar` caches the compiled pest VM
2. **Pre-compile Grammars**: Use `.zpeg` files for faster startup
3. **Tiered Compilation**: Hot functions automatically optimized
4. **Zero-Copy Types**: `ZyntaxString`/`ZyntaxArray` avoid copies at FFI boundary
5. **Function Pointer Caching**: Runtime maintains function_id → pointer map

## Usage Scenarios

### Scenario 1: Scripting Engine
```
Application loads grammar → User provides scripts → JIT compile and execute
```

### Scenario 2: Language Playground
```
Grammar bundled in app → User types code → Parse and execute interactively
```

### Scenario 3: DSL Execution
```
Custom DSL grammar → Load business rules → Execute with runtime data
```

### Scenario 4: Hot-Reloadable Game Logic
```
Initial game scripts loaded → Developer edits → hot_reload() without restart
```

## See Also

- [The Zyn Book: Embedding SDK](../book/12-embedding-sdk.md) - Tutorial and examples
- [Tiered Compilation](./tiered-compilation.md) - Multi-tier JIT architecture
- [Plugin Architecture](./PLUGIN_ARCHITECTURE.md) - ZRTL plugin system
- [TypedAST](../book/06-typed-ast.md) - Intermediate representation
