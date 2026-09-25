# Embedding Zyntax in Rust Applications

This chapter covers how to embed the Zyntax JIT runtime in your Rust applications using the `zyntax_embed` crate. This enables you to compile and execute code from any language with a ZynPEG grammar directly from Rust.

## Overview

The Zyntax Embedding SDK provides:

- **Language Grammar Interface**: Parse source code using any `.zyn` grammar
- **Multi-Language Runtime**: Register multiple grammars and compile from different languages
- **JIT Compilation**: Compile TypedAST to native code at runtime
- **Multi-Tier Optimization**: Automatic optimization of hot code paths
- **Bidirectional Interop**: Seamless conversion between Rust and Zyntax types
- **Async/Await Support**: Promise-based async operations
- **Hot Reloading**: Update functions without restarting
- **ZRTL Plugin Loading**: Load native runtime libraries dynamically

## Installation

Add `zyntax_embed` to your `Cargo.toml`:

```toml
[dependencies]
zyntax_embed = { path = "path/to/zyntax/crates/zyntax_embed" }
```

## Quick Start

### Basic Usage with Grammar

```rust
use zyntax_embed::{ZyntaxRuntime, LanguageGrammar, ZyntaxValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile grammar from .zyn file
    let grammar = LanguageGrammar::compile_zyn_file("grammars/zig.zyn")?;

    // Create runtime
    let mut runtime = ZyntaxRuntime::new()?;

    // Compile source code
    runtime.compile_with_grammar(&grammar, r#"
        pub fn add(a: i32, b: i32) i32 {
            return a + b;
        }

        pub fn main() i32 {
            return add(10, 32);
        }
    "#)?;

    // Call compiled function
    let result: i32 = runtime.call("main", &[])?;
    println!("Result: {}", result); // Output: 42

    Ok(())
}
```

### Compiling Grammar from Source

```rust
use zyntax_embed::LanguageGrammar;

// Compile grammar from .zyn source
let grammar = LanguageGrammar::compile_zyn(include_str!("my_lang.zyn"))?;

// Or from a file
let grammar = LanguageGrammar::compile_zyn_file("grammars/my_lang.zyn")?;

// Save compiled grammar for faster loading later
grammar.save("my_lang.zpeg")?;
```

## The LanguageGrammar API

`LanguageGrammar` wraps a compiled ZynPEG module and provides parsing capabilities.

### Creating Grammars

| Method | Description |
|--------|-------------|
| `load(path)` | Load pre-compiled `.zpeg` file |
| `compile_zyn(source)` | Compile from `.zyn` grammar source string |
| `compile_zyn_file(path)` | Compile from `.zyn` grammar file |
| `from_json(json)` | Load from serialized JSON string |
| `from_module(module)` | Create from existing `ZpegModule` |

### Grammar Metadata

```rust
let grammar = LanguageGrammar::load("zig.zpeg")?;

println!("Language: {}", grammar.name());           // "Zig"
println!("Version: {}", grammar.version());         // "0.11"
println!("Extensions: {:?}", grammar.file_extensions()); // [".zig"]

if let Some(entry) = grammar.entry_point() {
    println!("Entry point: {}", entry);  // "main"
}

// Get builtin function mappings
for (name, symbol) in grammar.builtins() {
    println!("Builtin '{}' maps to '{}'", name, symbol);
}
```

### Parsing Source Code

```rust
// Parse to TypedProgram (for further processing)
let program: TypedProgram = grammar.parse(source_code)?;

// Parse to JSON (for debugging or serialization)
let json: String = grammar.parse_to_json(source_code)?;
println!("{}", json);
```

### Custom AST Builders

For advanced use cases, implement `AstHostFunctions` for custom AST construction:

```rust
use zyntax_embed::{AstHostFunctions, NodeHandle, TypedAstBuilder};

// Use the default builder
let json = grammar.parse_with_builder(source, TypedAstBuilder::new())?;

// Or implement your own AstHostFunctions for custom behavior
struct MyCustomBuilder { /* ... */ }
impl AstHostFunctions for MyCustomBuilder {
    // Implement required methods...
}
```

## Multi-Language Runtime

The runtime supports registering multiple language grammars and loading modules by language name. This enables polyglot applications where different parts of your codebase can use different languages.

### Registering Grammars

```rust
use zyntax_embed::{ZyntaxRuntime, LanguageGrammar};

let mut runtime = ZyntaxRuntime::new()?;

// Register grammars by language name
runtime.register_grammar("zig", LanguageGrammar::compile_zyn_file("grammars/zig.zyn")?);
runtime.register_grammar("python", LanguageGrammar::compile_zyn_file("grammars/python.zyn")?);
runtime.register_grammar("calc", LanguageGrammar::compile_zyn_file("grammars/calc.zyn")?);

// Or use convenience methods
runtime.register_grammar_file("mylang", "grammars/mylang.zyn")?;
runtime.register_grammar_zpeg("lua", "grammars/lua.zpeg")?;

// Query registered languages
println!("Languages: {:?}", runtime.languages());       // ["zig", "python", "calc", "mylang", "lua"]
println!("Has Python: {}", runtime.has_language("python")); // true
```

### Loading Modules by Language

```rust
// Load modules specifying the language
let functions = runtime.load_module("zig", r#"
    pub fn add(a: i32, b: i32) i32 {
        return a + b;
    }

    pub fn factorial(n: i32) i32 {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
"#)?;

println!("Loaded functions: {:?}", functions); // ["add", "factorial"]

// Load from another language
runtime.load_module("calc", "def multiply(a, b) = a * b")?;

// Call any function regardless of source language
let sum: i32 = runtime.call("add", &[10.into(), 32.into()])?;
let fact: i32 = runtime.call("factorial", &[5.into()])?;
```

### Auto-Detection from File Extension

When grammars declare file extensions in their `@language` metadata, the runtime automatically maps extensions to languages:

```rust
// The zig.zyn grammar declares: file_extensions: [".zig"]
runtime.register_grammar("zig", LanguageGrammar::compile_zyn_file("zig.zyn")?);

// Load files - language auto-detected from extension
runtime.load_module_file("./src/math.zig")?;   // Uses "zig" grammar
runtime.load_module_file("./lib/utils.py")?;   // Uses "python" grammar

// Query extension mappings
println!("Language for .zig: {:?}", runtime.language_for_extension(".zig")); // Some("zig")
println!("Language for .py: {:?}", runtime.language_for_extension("py"));    // Some("python")
```

### Cross-Language Function Calls

All modules loaded into the same runtime share a common execution environment. Functions can call each other across language boundaries using **explicit exports**:

```rust
// Load core utilities in Zig and EXPORT them for cross-module linking
runtime.load_module_with_exports("zig", r#"
    pub fn square(x: i32) i32 { return x * x; }
    pub fn cube(x: i32) i32 { return x * x * x; }
"#, &["square", "cube"])?;

// Load a DSL that uses the Zig functions via extern declarations
runtime.load_module("calc", r#"
    extern fn square(x: i32) i32;
    extern fn cube(x: i32) i32;

    def sum_of_powers(a, b) = square(a) + cube(b)
"#)?;

// The calc function calls into the Zig implementation
let result: i32 = runtime.call("sum_of_powers", &[3.into(), 2.into()])?;
assert_eq!(result, 17); // square(3) + cube(2) = 9 + 8 = 17
```

#### Export Management

Functions must be explicitly exported to be available for cross-module linking:

```rust
// Method 1: Export during load
runtime.load_module_with_exports("zig", source, &["fn1", "fn2"])?;

// Method 2: Export after load
runtime.load_module("zig", source)?;
runtime.export_function("my_func")?;

// Check for symbol conflicts
if let Some(existing_ptr) = runtime.check_export_conflict("my_func") {
    println!("Warning: my_func already exported at {:?}", existing_ptr);
}

// List all exported symbols
for (name, ptr) in runtime.exported_symbols() {
    println!("Exported: {} at {:?}", name, ptr);
}
```

**Note:** Attempting to export a function with the same name as an existing export will log a warning and overwrite the existing symbol.

### TieredRuntime Multi-Language Support

The `TieredRuntime` also supports multi-language modules with the same API:

```rust
use zyntax_embed::TieredRuntime;

let mut runtime = TieredRuntime::production()?;

runtime.register_grammar("zig", LanguageGrammar::compile_zyn_file("zig.zyn")?);
runtime.load_module("zig", "pub fn compute(x: i32) i32 { return x * 2; }")?;

// Hot functions are automatically optimized regardless of source language
let result: i32 = runtime.call("compute", &[21.into()])?;
```

## Runtime Options

### ZyntaxRuntime (Single-Tier JIT)

Best for simple use cases with predictable performance:

```rust
use zyntax_embed::ZyntaxRuntime;

// Basic runtime
let mut runtime = ZyntaxRuntime::new()?;

// With custom configuration
let config = CompilationConfig::default();
let mut runtime = ZyntaxRuntime::with_config(config)?;

// With external FFI symbols
let symbols = &[
    ("my_c_func", my_c_func as *const u8),
    ("log_value", log_value as *const u8),
];
let mut runtime = ZyntaxRuntime::with_symbols(symbols)?;
```

### TieredRuntime (Multi-Tier JIT)

Adaptive compilation with automatic optimization for production workloads:

```rust
use zyntax_embed::{TieredRuntime, TieredConfig, OptimizationTier};

// Development mode: Fast startup, no background optimization
let mut runtime = TieredRuntime::development()?;

// Production mode: Full tiered optimization
let mut runtime = TieredRuntime::production()?;

// Custom configuration
let config = TieredConfig {
    warm_threshold: 100,      // Calls before Tier 1
    hot_threshold: 1000,      // Calls before Tier 2
    enable_background_optimization: true,
    ..Default::default()
};
let mut runtime = TieredRuntime::with_config(config)?;
```

#### Tiered Compilation Architecture

| Tier | Name | Backend | When Used |
|------|------|---------|-----------|
| 0 | Baseline | Cranelift (fast) | Cold code, startup |
| 1 | Standard | Cranelift (optimized) | Warm code |
| 2 | Optimized | Cranelift/LLVM | Hot code |

Functions automatically promote through tiers based on execution frequency:

```rust
// Check current optimization tier
let tier = runtime.get_function_tier("hot_function")?;

// Force optimization for latency-sensitive code
runtime.optimize_function("critical_path", OptimizationTier::Optimized)?;

// Get statistics
let stats = runtime.statistics();
println!("{}", stats.format());
// Output:
// TieredStatistics:
//   Baseline (Tier 0): 45 functions
//   Standard (Tier 1): 12 functions
//   Optimized (Tier 2): 3 functions
```

## Compiling and Executing Code

### Full Pipeline with Grammar

```rust
use zyntax_embed::{ZyntaxRuntime, LanguageGrammar};

let grammar = LanguageGrammar::compile_zyn(ZIG_GRAMMAR)?;
let mut runtime = ZyntaxRuntime::new()?;

// Single method compiles: parse → lower → compile
runtime.compile_with_grammar(&grammar, source_code)?;

// Now call functions
let result: i32 = runtime.call("main", &[])?;
```

### Manual Pipeline (Advanced)

For more control, use the manual pipeline:

```rust
use zyntax_embed::{
    ZyntaxRuntime, LanguageGrammar, TypedProgram,
    compile_to_hir, HirModule,
};

// Step 1: Parse source to TypedAST
let grammar = LanguageGrammar::load("lang.zpeg")?;
let typed_program: TypedProgram = grammar.parse(source)?;

// Step 2: Lower to HIR (custom lowering possible here)
let hir_module: HirModule = compile_to_hir(&typed_program, type_registry, config)?;

// Step 3: Compile HIR to native code
let mut runtime = ZyntaxRuntime::new()?;
runtime.compile_module(&hir_module)?;

// Step 4: Execute
let result: i32 = runtime.call("main", &[])?;
```

## Calling Functions

### Type-Safe Calls

```rust
// Automatic type conversion
let sum: i32 = runtime.call("add", &[5.into(), 3.into()])?;
let greeting: String = runtime.call("greet", &["World".into()])?;
let point: (f64, f64) = runtime.call("make_point", &[1.0.into(), 2.0.into()])?;
```

### Raw Value Calls

```rust
let result: ZyntaxValue = runtime.call_raw("compute", &[data.into()])?;

match result {
    ZyntaxValue::Int(n) => println!("Integer: {}", n),
    ZyntaxValue::Float(f) => println!("Float: {}", f),
    ZyntaxValue::String(s) => println!("String: {}", s),
    ZyntaxValue::Bool(b) => println!("Bool: {}", b),
    ZyntaxValue::Array(items) => {
        println!("Array with {} items", items.len());
    }
    ZyntaxValue::Struct { name, fields } => {
        println!("Struct {}: {:?}", name, fields);
    }
    ZyntaxValue::Null => println!("Null"),
    _ => println!("Other: {:?}", result),
}
```

### Native Calling with Signatures

For JIT-compiled functions, use `call_function` with an explicit signature for optimal performance. This bypasses the `DynamicValue` ABI and calls functions with native types directly.

```rust
use zyntax_embed::{ZyntaxRuntime, NativeType, NativeSignature};

// Define the function signature: (i32, i32) -> i32
let sig = NativeSignature::new(&[NativeType::I32, NativeType::I32], NativeType::I32);

// Call with native types
let result = runtime.call_function("add", &[10.into(), 32.into()], &sig)?;
assert_eq!(result.as_i32().unwrap(), 42);

// Or parse signature from a string
let sig = NativeSignature::parse("(i32, i32) -> i32").unwrap();
let result = runtime.call_function("multiply", &[6.into(), 7.into()], &sig)?;
```

#### Supported Native Types

| NativeType | Rust Equivalent | Description |
|------------|-----------------|-------------|
| `I32` | `i32` | 32-bit signed integer |
| `I64` | `i64` | 64-bit signed integer |
| `F32` | `f32` | 32-bit float |
| `F64` | `f64` | 64-bit float |
| `Bool` | `bool` | Boolean (passed as i8) |
| `Void` | `()` | No return value |
| `Ptr` | `*mut u8` | Pointer type |

#### Signature String Format

The `NativeSignature::parse` method accepts strings in the format:

- `"() -> void"` - No parameters, no return
- `"(i32) -> i32"` - Single parameter
- `"(i32, i32) -> i64"` - Multiple parameters
- `"(f64, f64) -> f64"` - Floating point types

#### When to Use Native Calling

Use `call_function` when:

- You know the exact function signature at compile time
- You need maximum performance for hot paths
- You're calling JIT-compiled functions with primitive types

Use `call` or `call_raw` when:

- You need automatic type conversion
- The function signature is dynamic
- You're working with complex types (structs, arrays)

### Async Functions and Promises

```rust
use zyntax_embed::{ZyntaxPromise, PromiseState};

// Call async function
let promise: ZyntaxPromise = runtime.call_async("fetch_data", &[url.into()])?;

// Option 1: Block and wait
let data: String = promise.await_result()?;

// Option 2: Poll manually
loop {
    match promise.poll() {
        PromiseState::Pending => {
            // Do other work...
            std::thread::yield_now();
        }
        PromiseState::Ready(value) => {
            println!("Got: {:?}", value);
            break;
        }
        PromiseState::Failed(err) => {
            eprintln!("Error: {}", err);
            break;
        }
    }
}

// Option 3: Chain with .then() and .catch()
let processed = promise
    .then(|data| ZyntaxValue::String(format!("Processed: {:?}", data)))
    .catch(|err| ZyntaxValue::String(format!("Error: {}", err)));

// Option 4: Use with async/await (implements Future)
async fn process_data(runtime: &ZyntaxRuntime) -> Result<String, RuntimeError> {
    let promise = runtime.call_async("fetch", &["data".into()])?;
    let result: String = promise.await?;
    Ok(result)
}
```

## Value Conversion

### ZyntaxValue

The universal runtime value type:

```rust
use zyntax_embed::ZyntaxValue;

// From Rust types
let int_val: ZyntaxValue = 42i32.into();
let float_val: ZyntaxValue = 3.14f64.into();
let str_val: ZyntaxValue = "hello".into();
let bool_val: ZyntaxValue = true.into();
let vec_val: ZyntaxValue = vec![1, 2, 3].into();

// Create structs
let point = ZyntaxValue::new_struct("Point")
    .field("x", 10.0f64)
    .field("y", 20.0f64)
    .build();

// Access struct fields
if let ZyntaxValue::Struct { fields, .. } = &point {
    if let Some(ZyntaxValue::Float(x)) = fields.get("x") {
        println!("x = {}", x);
    }
}
```

### FromZyntax / IntoZyntax Traits

```rust
use zyntax_embed::{FromZyntax, IntoZyntax, ZyntaxValue};

// Rust → ZyntaxValue
let value: ZyntaxValue = 42i32.into_zyntax();

// ZyntaxValue → Rust
let num: i32 = FromZyntax::from_zyntax(value)?;
let opt: Option<i32> = FromZyntax::from_zyntax(nullable_value)?;
```

### Zero-Copy String and Array Types

```rust
use zyntax_embed::{ZyntaxString, ZyntaxArray};

// ZyntaxString: Length-prefixed UTF-8
let s = ZyntaxString::from_str("Hello!");
let ptr = s.as_ptr();  // Memory: [i32 length][utf8_bytes...]
println!("{}", s.as_str().unwrap());

// ZyntaxArray: Capacity + length header
let mut arr: ZyntaxArray<i32> = [1, 2, 3].into();
arr.push(4);
let ptr = arr.as_ptr();  // Memory: [i32 capacity][i32 length][elements...]
```

## Import Resolution

Register resolvers for handling import statements:

```rust
// Callback-based resolver
runtime.add_import_resolver(Box::new(|module_path| {
    match module_path {
        "std.math" => Ok(Some(include_str!("stdlib/math.zig").to_string())),
        "std.io" => Ok(Some(include_str!("stdlib/io.zig").to_string())),
        _ => Ok(None), // Not found, try next resolver
    }
}));

// File-system resolver
runtime.add_filesystem_resolver("./src", "zig");  // Looks for .zig files
runtime.add_filesystem_resolver("./lib", "hx");   // Looks for .hx files

// Check resolver count
println!("Resolvers: {}", runtime.import_resolver_count());
```

## External Function Registration

Register Rust functions directly with the runtime without creating a ZRTL plugin. This is the simplest way to expose native functionality to Zyntax code.

### Registering Functions at Construction

Use `with_symbols` to register FFI functions when creating the runtime:

```rust
use zyntax_embed::ZyntaxRuntime;

// Define extern "C" functions
extern "C" fn native_add(a: i32, b: i32) -> i32 {
    a + b
}

extern "C" fn native_print(value: i32) {
    println!("[Native] Value: {}", value);
}

extern "C" fn native_sqrt(x: f64) -> f64 {
    x.sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Register symbols at construction
    let symbols: &[(&str, *const u8)] = &[
        ("native_add", native_add as *const u8),
        ("native_print", native_print as *const u8),
        ("native_sqrt", native_sqrt as *const u8),
    ];

    let mut runtime = ZyntaxRuntime::with_symbols(symbols)?;

    // Now Zyntax code can call these functions
    runtime.compile_with_grammar(&grammar, r#"
        // Declare external functions
        extern fn native_add(a: i32, b: i32) i32;
        extern fn native_print(value: i32) void;

        pub fn main() i32 {
            const result = native_add(10, 32);
            native_print(result);
            return result;
        }
    "#)?;

    let result: i32 = runtime.call("main", &[])?;
    Ok(())
}
```

### Registering Functions Dynamically

Add functions after runtime creation using `register_function`:

```rust
use zyntax_embed::{ZyntaxRuntime, ExternalFunction};

let mut runtime = ZyntaxRuntime::new()?;

// Register individual functions
runtime.register_function(ExternalFunction {
    name: "log_message".to_string(),
    ptr: log_message as *const u8,
    signature: "(ptr, i32) -> void".to_string(),  // Optional signature hint
})?;

// Register multiple functions
runtime.register_functions(&[
    ExternalFunction::new("math_sin", math_sin as *const u8),
    ExternalFunction::new("math_cos", math_cos as *const u8),
    ExternalFunction::new("math_tan", math_tan as *const u8),
])?;
```

### Function Signatures

External functions must use the C ABI (`extern "C"`). Supported parameter and return types:

| Rust Type | Zyntax Type | Notes |
|-----------|-------------|-------|
| `i32` | `i32` | Signed 32-bit integer |
| `i64` | `i64` | Signed 64-bit integer |
| `f32` | `f32` | 32-bit float |
| `f64` | `f64` | 64-bit float |
| `bool` | `bool` | Boolean (as i8) |
| `*const u8` | `[]const u8` | Pointer to string/slice data |
| `*mut T` | `*T` | Mutable pointer |
| `()` | `void` | No return value |

### Working with Strings

Strings are passed as length-prefixed pointers:

```rust
use std::slice;

/// Receive a Zyntax string (length-prefixed)
extern "C" fn print_string(ptr: *const u8) {
    unsafe {
        // First 4 bytes are the length (i32)
        let len = *(ptr as *const i32) as usize;
        let data = ptr.add(4);
        let bytes = slice::from_raw_parts(data, len);
        if let Ok(s) = std::str::from_utf8(bytes) {
            println!("{}", s);
        }
    }
}

/// Return a Zyntax string (must allocate with Zyntax allocator)
extern "C" fn create_greeting(name_ptr: *const u8) -> *const u8 {
    unsafe {
        // Read input string
        let len = *(name_ptr as *const i32) as usize;
        let data = name_ptr.add(4);
        let name = std::str::from_utf8(slice::from_raw_parts(data, len))
            .unwrap_or("World");

        // Create output (using Zyntax allocator in real code)
        let greeting = format!("Hello, {}!", name);
        // ... allocate and return
    }
}
```

### Working with Arrays

Arrays use capacity + length header:

```rust
/// Sum an array of integers
extern "C" fn sum_array(ptr: *const u8) -> i64 {
    unsafe {
        let header = ptr as *const i32;
        let _capacity = *header;
        let length = *header.add(1) as usize;
        let data = (ptr as *const i32).add(2);

        let mut sum: i64 = 0;
        for i in 0..length {
            sum += *data.add(i) as i64;
        }
        sum
    }
}
```

### Callback Functions

Pass Zyntax functions to Rust:

```rust
type ZyntaxCallback = extern "C" fn(i32) -> i32;

extern "C" fn apply_twice(f: ZyntaxCallback, x: i32) -> i32 {
    f(f(x))
}

// Register it
let symbols = &[("apply_twice", apply_twice as *const u8)];
let mut runtime = ZyntaxRuntime::with_symbols(symbols)?;

// Use from Zyntax
runtime.compile_with_grammar(&grammar, r#"
    extern fn apply_twice(f: fn(i32) i32, x: i32) i32;

    fn double(n: i32) i32 {
        return n * 2;
    }

    pub fn main() i32 {
        return apply_twice(double, 5);  // Returns 20
    }
"#)?;
```

### Complete Example: Math Library

```rust
use zyntax_embed::{ZyntaxRuntime, LanguageGrammar};

// Math functions
extern "C" fn math_abs(x: f64) -> f64 { x.abs() }
extern "C" fn math_floor(x: f64) -> f64 { x.floor() }
extern "C" fn math_ceil(x: f64) -> f64 { x.ceil() }
extern "C" fn math_sqrt(x: f64) -> f64 { x.sqrt() }
extern "C" fn math_pow(base: f64, exp: f64) -> f64 { base.powf(exp) }
extern "C" fn math_sin(x: f64) -> f64 { x.sin() }
extern "C" fn math_cos(x: f64) -> f64 { x.cos() }
extern "C" fn math_log(x: f64) -> f64 { x.ln() }

// I/O functions
extern "C" fn io_print_i32(x: i32) { println!("{}", x); }
extern "C" fn io_print_f64(x: f64) { println!("{}", x); }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let symbols: &[(&str, *const u8)] = &[
        // Math
        ("math_abs", math_abs as *const u8),
        ("math_floor", math_floor as *const u8),
        ("math_ceil", math_ceil as *const u8),
        ("math_sqrt", math_sqrt as *const u8),
        ("math_pow", math_pow as *const u8),
        ("math_sin", math_sin as *const u8),
        ("math_cos", math_cos as *const u8),
        ("math_log", math_log as *const u8),
        // I/O
        ("io_print_i32", io_print_i32 as *const u8),
        ("io_print_f64", io_print_f64 as *const u8),
    ];

    let grammar = LanguageGrammar::compile_zyn_file("grammars/zig.zyn")?;
    let mut runtime = ZyntaxRuntime::with_symbols(symbols)?;

    runtime.compile_with_grammar(&grammar, r#"
        // External declarations
        extern fn math_sqrt(x: f64) f64;
        extern fn math_pow(base: f64, exp: f64) f64;
        extern fn io_print_f64(x: f64) void;

        pub fn hypotenuse(a: f64, b: f64) f64 {
            return math_sqrt(math_pow(a, 2.0) + math_pow(b, 2.0));
        }

        pub fn main() void {
            const h = hypotenuse(3.0, 4.0);
            io_print_f64(h);  // Prints: 5
        }
    "#)?;

    runtime.call::<()>("main", &[])?;
    Ok(())
}
```

### External Functions vs ZRTL Plugins

| Feature | External Functions | ZRTL Plugins |
|---------|-------------------|--------------|
| Setup complexity | Simple, inline code | Separate crate + build |
| Distribution | Compiled into host app | Separate `.zrtl` files |
| Hot loading | No | Yes |
| Symbol namespacing | Manual | Automatic (`$Type$method`) |
| Best for | App-specific functions | Reusable libraries |

Use External Functions when you need to expose host application functionality. Use ZRTL Plugins when creating reusable runtime libraries for distribution.

## ZRTL Plugin Loading

Load native runtime libraries (written in Rust or C):

```rust
// Load single plugin
runtime.load_plugin("./my_runtime.zrtl")?;

// Load all plugins from directory
let count = runtime.load_plugins_from_directory("./plugins")?;
println!("Loaded {} plugins", count);
```

Creating a ZRTL plugin in Rust:

```rust
// In your plugin crate (lib.rs)
use zrtl::{zrtl_plugin, zrtl_export};

zrtl_plugin!("my_runtime");

#[zrtl_export("$MyRuntime$add")]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[zrtl_export("$MyRuntime$print")]
pub extern "C" fn print_value(value: i32) {
    println!("Value: {}", value);
}
```

## Hot Reloading

Update functions at runtime without restarting:

```rust
// Original function compiled
runtime.compile_with_grammar(&grammar, "pub fn compute(x: i32) i32 { return x * 2; }")?;

let v1: i32 = runtime.call("compute", &[10.into()])?;
println!("v1: {}", v1);  // 20

// Hot reload with new implementation
let new_function = /* compile new version */;
runtime.hot_reload("compute", &new_function)?;

let v2: i32 = runtime.call("compute", &[10.into()])?;
println!("v2: {}", v2);  // New result
```

## Error Handling

```rust
use zyntax_embed::{RuntimeError, GrammarError, ConversionError};

// Grammar errors
match LanguageGrammar::compile_zyn(bad_grammar) {
    Ok(g) => { /* use grammar */ }
    Err(GrammarError::ParseError(msg)) => eprintln!("Parse error: {}", msg),
    Err(GrammarError::CompileError(msg)) => eprintln!("Compile error: {}", msg),
    Err(e) => eprintln!("Error: {}", e),
}

// Runtime errors
match runtime.call::<i32>("compute", &args) {
    Ok(result) => println!("Result: {}", result),
    Err(RuntimeError::FunctionNotFound(name)) => {
        eprintln!("Function '{}' not found", name);
    }
    Err(RuntimeError::Compilation(e)) => {
        eprintln!("Compilation failed: {}", e);
    }
    Err(RuntimeError::Conversion(ConversionError::TypeMismatch { expected, actual })) => {
        eprintln!("Type error: expected {:?}, got {:?}", expected, actual);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Complete Example: Scripting Engine

A full example showing how to build a scripting engine:

```rust
use zyntax_embed::{
    ZyntaxRuntime, LanguageGrammar, ZyntaxValue,
    RuntimeError, GrammarError,
};
use std::collections::HashMap;

struct ScriptEngine {
    runtime: ZyntaxRuntime,
    grammar: LanguageGrammar,
    scripts: HashMap<String, String>,
}

impl ScriptEngine {
    pub fn new(grammar_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let grammar = LanguageGrammar::load(grammar_path)?;
        let mut runtime = ZyntaxRuntime::new()?;

        // Register built-in functions
        runtime.add_import_resolver(Box::new(|path| {
            match path {
                "builtins" => Ok(Some(BUILTINS_SOURCE.to_string())),
                _ => Ok(None),
            }
        }));

        Ok(Self {
            runtime,
            grammar,
            scripts: HashMap::new(),
        })
    }

    pub fn load_script(&mut self, name: &str, source: &str) -> Result<(), RuntimeError> {
        self.runtime.compile_with_grammar(&self.grammar, source)?;
        self.scripts.insert(name.to_string(), source.to_string());
        Ok(())
    }

    pub fn call_function<T: zyntax_embed::FromZyntax>(
        &self,
        func: &str,
        args: &[ZyntaxValue],
    ) -> Result<T, RuntimeError> {
        self.runtime.call(func, args)
    }

    pub fn eval(&mut self, expr: &str) -> Result<ZyntaxValue, RuntimeError> {
        // Wrap expression in a function and call it
        let wrapper = format!("pub fn __eval__() {{ return {}; }}", expr);
        self.runtime.compile_with_grammar(&self.grammar, &wrapper)?;
        self.runtime.call_raw("__eval__", &[])
    }
}

const BUILTINS_SOURCE: &str = r#"
    pub fn print(msg: []const u8) void {
        // Mapped to native print
    }
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = ScriptEngine::new("zig.zpeg")?;

    // Load a script
    engine.load_script("math", r#"
        pub fn factorial(n: i32) i32 {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
    "#)?;

    // Call functions
    let result: i32 = engine.call_function("factorial", &[5.into()])?;
    println!("5! = {}", result);  // 120

    // Evaluate expressions
    let value = engine.eval("10 + 20 * 2")?;
    println!("Result: {:?}", value);  // Int(50)

    Ok(())
}
```

## Performance Tips

1. **Pre-compile grammars**: Use `.zpeg` files instead of compiling `.zyn` at startup
2. **Use TieredRuntime for production**: Automatically optimizes hot paths
3. **Pre-warm critical paths**: Call `optimize_function()` for latency-sensitive code
4. **Reuse runtime instances**: Creating new runtimes has overhead
5. **Use zero-copy types**: `ZyntaxString` and `ZyntaxArray` avoid copying
6. **Batch imports**: Load all scripts before execution begins

## Memory Safety

- `ZyntaxValue` follows Rust ownership semantics
- `ZyntaxString` and `ZyntaxArray` track ownership for proper cleanup
- Promise state is thread-safe via internal mutex
- Function pointers are atomically swapped during tier promotion
- ZRTL plugins must follow the C ABI for safety

## Next Steps

- See [Async Runtime](./13-async-runtime.md) for detailed async/Promise documentation
- See [Grammar Syntax](./04-grammar-syntax.md) for writing `.zyn` grammars
- See [TypedAST](./06-typed-ast.md) for understanding the intermediate representation
- See [ZRTL Plugins](./10-packaging-distribution.md) for creating native runtime libraries
