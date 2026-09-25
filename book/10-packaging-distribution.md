# Packaging and Distribution

This chapter covers how to package and distribute compiled Zyntax programs, including the ZPack format for JIT execution and static library linking for AOT compilation.

## Compilation Modes Overview

Zyntax supports two primary compilation modes:

| Mode | Backend | Use Case | Runtime Symbols |
|------|---------|----------|-----------------|
| **JIT** | Cranelift, LLVM | Development, scripting | Loaded from ZPack (.zpack) |
| **AOT** | LLVM | Production, distribution | Linked from static libraries (.a/.lib) |

### JIT Mode

JIT (Just-In-Time) compilation compiles and executes code immediately. It's fast to start and ideal for development workflows.

```bash
# JIT execution with runtime from ZPack
zyntax compile --jit --pack runtime.zpack \
  --grammar lang.zyn --source main.lang
```

### AOT Mode

AOT (Ahead-Of-Time) compilation produces native executables. It generates optimized machine code suitable for production deployment.

```bash
# AOT compilation with static library linking
zyntax compile --backend llvm \
  --grammar lang.zyn --source main.lang \
  -o myapp --lib runtime
```

## ZPack Format

ZPack is a compressed archive format designed for **JIT execution**. It bundles bytecode modules and platform-specific dynamic libraries.

### Archive Structure

```text
my_runtime.zpack (ZIP archive)
├── manifest.json           # Package metadata
├── modules/                 # Compiled bytecode modules
│   ├── std/
│   │   ├── Array.zbc
│   │   ├── String.zbc
│   │   └── Math.zbc
│   └── main.zbc
└── lib/                    # Platform-specific dynamic libraries
    ├── x86_64-apple-darwin/
    │   └── runtime.zrtl
    ├── x86_64-unknown-linux-gnu/
    │   └── runtime.zrtl
    ├── aarch64-apple-darwin/
    │   └── runtime.zrtl
    └── x86_64-pc-windows-msvc/
        └── runtime.zrtl
```

### File Extensions

| Extension | Description |
|-----------|-------------|
| `.zpack` | ZPack archive (ZIP format) |
| `.zbc` | Zyntax Bytecode (compiled HIR module) |
| `.zrtl` | Zyntax Runtime Library (dynamic library) |

### Manifest File

The `manifest.json` describes the package:

```json
{
  "version": 1,
  "name": "mylang-runtime",
  "package_version": "1.0.0",
  "description": "MyLang standard library runtime for Zyntax",
  "source_language": "mylang",
  "targets": [
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-unknown-linux-gnu"
  ],
  "modules": [
    "std/Array",
    "std/String",
    "std/Math"
  ],
  "exports": [
    {
      "name": "$mylang$trace$int",
      "signature": "fn(i32) -> void",
      "doc": "Print an integer to stdout"
    }
  ]
}
```

### Creating a ZPack

Use the `zyntax pack create` command:

```bash
# Create a ZPack with modules and runtime
zyntax pack create \
  --output mylang-runtime.zpack \
  --name mylang-runtime \
  --version 1.0.0 \
  --language mylang \
  --module modules/std/ \
  --runtime x86_64-apple-darwin:lib/darwin/runtime.zrtl \
  --runtime x86_64-unknown-linux-gnu:lib/linux/runtime.zrtl \
  --runtime aarch64-apple-darwin:lib/darwin-arm/runtime.zrtl
```

### Inspecting a ZPack

```bash
# List contents
zyntax pack list runtime.zpack

# Verbose output with module details
zyntax pack list runtime.zpack -v

# Extract to directory
zyntax pack extract runtime.zpack --output ./extracted/

# Show current platform target
zyntax pack target
```

### Using ZPacks at Runtime

Load a ZPack for JIT execution:

```bash
# Single pack
zyntax compile --jit --pack mylang-runtime.zpack \
  --grammar mylang.zyn --source main.mylang

# Multiple packs (combined)
zyntax compile --jit \
  --pack mylang-runtime.zpack \
  --pack my-extensions.zpack \
  --grammar mylang.zyn --source main.mylang
```

## Building Runtime Libraries

Runtime libraries provide the implementation of external functions called by compiled code.

### Dynamic Libraries (.zrtl)

For JIT mode, build a shared library with exported symbols:

```bash
# macOS
clang -shared -fPIC -o runtime.zrtl runtime.c

# Linux
gcc -shared -fPIC -o runtime.zrtl runtime.c

# Windows
cl /LD runtime.c /Fe:runtime.zrtl
```

### Static Libraries (.a/.lib)

For AOT mode, build a static library:

```bash
# Unix (macOS/Linux)
clang -c runtime.c -o runtime.o
ar rcs libruntime.a runtime.o

# Windows
cl /c runtime.c
lib runtime.obj /OUT:runtime.lib
```

### Runtime Symbol Naming

Runtime symbols follow a naming convention for cross-language interop:

```text
$<language>$<type>$<operation>
```

Examples:

- `$mylang$trace$int` - MyLang trace function for integers
- `$mylang$Array$push` - Array push method
- `$zig$print` - Zig print function

### Example Runtime (C)

```c
// runtime.c
#include <stdio.h>
#include <stdint.h>

// Trace function for integers
void mylang_trace_int(int32_t value) {
    printf("%d\n", value);
}

// Trace function for strings
void mylang_trace_string(const char* value) {
    printf("%s\n", value);
}

// Array operations
typedef struct {
    void** data;
    int32_t length;
    int32_t capacity;
} MyLangArray;

void mylang_array_push(MyLangArray* arr, void* value) {
    // Implementation...
}
```

When compiled, use `export_name` attributes or linker scripts to set the exact symbol names:

```c
// Using GCC/Clang attributes
__attribute__((visibility("default")))
void __attribute__((alias("mylang_trace_int")))
    $mylang$trace$int(int32_t);
```

Or with Rust using the `zrtl_macros` crate:

```rust
use zrtl_macros::{zrtl_plugin, zrtl_export};

// Define the plugin (required once per library)
zrtl_plugin!("mylang_runtime");

#[zrtl_export("$mylang$trace$int")]
pub extern "C" fn trace_int(value: i32) {
    println!("{}", value);
}

#[zrtl_export("$mylang$trace$string")]
pub extern "C" fn trace_string(s: *const std::ffi::c_char) {
    let c_str = unsafe { std::ffi::CStr::from_ptr(s) };
    println!("{}", c_str.to_string_lossy());
}
```

Build as a dynamic library (`cdylib`) for JIT mode:

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib"]

[dependencies]
zrtl_macros = { path = "path/to/sdk/zrtl_macros" }
inventory = "0.3"
```

Or as a static library (`staticlib`) for AOT mode:

```toml
# Cargo.toml
[lib]
crate-type = ["staticlib"]
```

## AOT Compilation and Linking

### Basic AOT Workflow

```bash
# 1. Compile source to executable with linked runtime
zyntax compile --backend llvm \
  --grammar mylang.zyn --source main.mylang \
  -o myapp --lib runtime

# 2. Or manually link (two-step process)
zyntax compile --backend llvm \
  --grammar mylang.zyn --source main.mylang \
  -o myapp.o --emit-obj
cc myapp.o -L/usr/local/lib -lruntime -o myapp
```

### Library Search Paths

When you specify `--lib runtime`, Zyntax searches for the library in standard locations:

**macOS:**

- `/usr/local/lib`
- `/opt/homebrew/lib`
- `/usr/lib`
- `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib`

**Linux:**

- `/usr/local/lib`
- `/usr/lib`
- `/usr/lib/x86_64-linux-gnu`
- `/usr/lib/aarch64-linux-gnu`

**Windows:**

- `C:\Windows\System32`
- `C:\Program Files\Common Files`

### Library Name Resolution

The `--lib` flag accepts multiple formats:

```bash
# Full path
--lib /path/to/libruntime.a

# Library name (searches for libruntime.a)
--lib runtime

# File name (searches in standard paths)
--lib libruntime.a
```

### Linking Multiple Libraries

```bash
zyntax compile --backend llvm \
  --grammar mylang.zyn --source main.mylang \
  -o myapp \
  --lib mylang-runtime \
  --lib myextensions \
  --lib pthread
```

## Cross-Platform Distribution

### Strategy 1: Fat ZPack (JIT)

Include runtime libraries for all target platforms in a single ZPack:

```bash
zyntax pack create \
  --output universal-runtime.zpack \
  --name my-runtime \
  --runtime x86_64-apple-darwin:lib/darwin-x64/runtime.zrtl \
  --runtime aarch64-apple-darwin:lib/darwin-arm64/runtime.zrtl \
  --runtime x86_64-unknown-linux-gnu:lib/linux-x64/runtime.zrtl \
  --runtime x86_64-pc-windows-msvc:lib/windows-x64/runtime.zrtl
```

Users can run on any supported platform:

```bash
# Works on any platform with matching runtime in the pack
zyntax compile --jit --pack universal-runtime.zpack \
  --grammar mylang.zyn --source main.mylang
```

### Strategy 2: Platform-Specific Builds (AOT)

Build separate executables for each target:

```bash
# macOS x86_64
zyntax compile --backend llvm \
  --grammar mylang.zyn --source main.mylang \
  -o myapp-darwin-x64 \
  --lib darwin-x64/libruntime.a

# macOS ARM64
zyntax compile --backend llvm \
  --grammar mylang.zyn --source main.mylang \
  -o myapp-darwin-arm64 \
  --lib darwin-arm64/libruntime.a

# Linux x86_64
zyntax compile --backend llvm \
  --grammar mylang.zyn --source main.mylang \
  -o myapp-linux-x64 \
  --lib linux-x64/libruntime.a
```

### Supported Target Triples

| Target Triple | Platform |
|---------------|----------|
| `x86_64-apple-darwin` | macOS Intel |
| `aarch64-apple-darwin` | macOS Apple Silicon |
| `x86_64-unknown-linux-gnu` | Linux x64 (glibc) |
| `x86_64-unknown-linux-musl` | Linux x64 (musl) |
| `aarch64-unknown-linux-gnu` | Linux ARM64 |
| `x86_64-pc-windows-msvc` | Windows x64 (MSVC) |
| `x86_64-pc-windows-gnu` | Windows x64 (MinGW) |

Check your current target:

```bash
zyntax pack target
```

## Bytecode Caching

Zyntax supports incremental compilation through bytecode caching.

### Cache Location

By default, cached bytecode is stored in:

- **Unix:** `~/.zyntax/cache/`
- **Windows:** `%USERPROFILE%\.zyntax\cache\`

### Cache Commands

```bash
# View cache statistics
zyntax cache stats

# List cached modules
zyntax cache list
zyntax cache list -v  # Verbose

# Clear the cache
zyntax cache clear

# Use a custom cache directory
zyntax compile --cache-dir ./my-cache \
  --grammar mylang.zyn --source main.mylang

# Disable caching
zyntax compile --no-cache \
  --grammar mylang.zyn --source main.mylang
```

### How Caching Works

1. Source files are hashed based on content
2. If a matching `.zbc` file exists, it's loaded instead of recompiling
3. Dependencies trigger recompilation of dependent modules

## Example: Complete Distribution Workflow

Here's a complete example of building and distributing a MyLang application:

### 1. Build the Runtime

```bash
# Compile runtime for all targets
for target in darwin-x64 darwin-arm64 linux-x64; do
  cd runtime && make TARGET=$target
done
```

### 2. Create the ZPack (for JIT users)

```bash
zyntax pack create \
  --output dist/mylang-runtime.zpack \
  --name mylang-runtime \
  --version 1.0.0 \
  --language mylang \
  --description "MyLang runtime for Zyntax" \
  --runtime x86_64-apple-darwin:runtime/darwin-x64/runtime.zrtl \
  --runtime aarch64-apple-darwin:runtime/darwin-arm64/runtime.zrtl \
  --runtime x86_64-unknown-linux-gnu:runtime/linux-x64/runtime.zrtl
```

### 3. Build AOT Executables

```bash
# For each platform
zyntax compile --backend llvm \
  --grammar grammars/mylang.zyn --source src/main.mylang \
  -o dist/myapp-darwin-x64 \
  --lib runtime/darwin-x64/libruntime.a \
  -O3

zyntax compile --backend llvm \
  --grammar grammars/mylang.zyn --source src/main.mylang \
  -o dist/myapp-linux-x64 \
  --lib runtime/linux-x64/libruntime.a \
  -O3
```

### 4. Distribution Structure

```text
dist/
├── mylang-runtime.zpack      # For JIT users
├── myapp-darwin-x64        # Native binary (macOS Intel)
├── myapp-darwin-arm64      # Native binary (macOS ARM)
├── myapp-linux-x64         # Native binary (Linux)
└── README.md
```

### 5. User Installation

**JIT users:**

```bash
# Download the zpack
curl -O https://example.com/dist/mylang-runtime.zpack

# Run directly with grammar and source
zyntax compile --jit --pack mylang-runtime.zpack \
  --grammar mylang.zyn --source script.mylang
```

**AOT users:**

```bash
# Download the native binary for their platform
curl -O https://example.com/dist/myapp-linux-x64
chmod +x myapp-linux-x64
./myapp-linux-x64
```

## Custom Packaging (Advanced)

For users who want to bypass the CLI and use the compiler and runtime plugin framework directly, Zyntax provides a comprehensive Rust API.

### Direct Compiler API

Use `zyntax_compiler` to compile TypedAST directly to HIR and native code:

```rust
use std::sync::Arc;
use zyntax_compiler::{
    compile_to_hir, CompilationConfig,
    cranelift_backend::CraneliftBackend,
    hir::HirModule,
};
use zyntax_typed_ast::{TypedProgram, TypeRegistry};

fn compile_program(program: &TypedProgram) -> Result<(), Box<dyn std::error::Error>> {
    // Create type registry
    let type_registry = Arc::new(TypeRegistry::new());

    // Configure compilation
    let config = CompilationConfig {
        opt_level: 2,
        debug_info: true,
        target_triple: "x86_64-unknown-linux-gnu".to_string(),
        hot_reload: false,
        enable_monomorphization: true,
        memory_strategy: Some(zyntax_compiler::memory_management::MemoryStrategy::ARC),
        async_runtime: None,
    };

    // Compile TypedAST → HIR
    let hir_module = compile_to_hir(program, type_registry, config)?;

    // Generate native code with Cranelift
    let mut backend = CraneliftBackend::new()?;
    backend.compile_module(&hir_module)?;

    // Execute a function
    let main_id = find_main_function(&hir_module)?;
    let result = backend.execute_function::<i64>(main_id)?;
    println!("Result: {}", result);

    Ok(())
}
```

### RuntimePlugin Trait

Implement the `RuntimePlugin` trait to provide runtime symbols programmatically:

```rust
use zyntax_compiler::plugin::{RuntimePlugin, PluginRegistry};

pub struct MyRuntimePlugin;

impl RuntimePlugin for MyRuntimePlugin {
    fn name(&self) -> &str {
        "my_runtime"
    }

    fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)> {
        vec![
            ("$MyRuntime$add", add as *const u8),
            ("$MyRuntime$print", print_value as *const u8),
            ("$MyRuntime$allocate", allocate as *const u8),
        ]
    }

    fn on_load(&self) -> Result<(), String> {
        // Initialize runtime state
        println!("MyRuntime loaded");
        Ok(())
    }

    fn on_unload(&self) -> Result<(), String> {
        // Cleanup
        Ok(())
    }
}

extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}

extern "C" fn print_value(value: i32) {
    println!("{}", value);
}

extern "C" fn allocate(size: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    unsafe { std::alloc::alloc(layout) }
}

// Register with the compiler
fn setup_runtime() -> PluginRegistry {
    let mut registry = PluginRegistry::new();
    registry.register(Box::new(MyRuntimePlugin)).unwrap();
    registry
}
```

### ZRTL Plugin Loading API

Load ZRTL plugins directly without using ZPack archives:

```rust
use zyntax_compiler::zrtl::{ZrtlPlugin, ZrtlRegistry};

fn load_runtime_plugins() -> Result<Vec<(&'static str, *const u8)>, Box<dyn std::error::Error>> {
    let mut registry = ZrtlRegistry::new();

    // Load individual plugins
    registry.load_plugin("./runtime/mylang_runtime.zrtl")?;
    registry.load_plugin("./runtime/math_extensions.zrtl")?;

    // Or load all plugins from a directory
    registry.load_directory("./plugins/")?;

    // Collect all symbols for the backend
    let symbols = registry.collect_symbols();
    println!("Loaded {} runtime symbols", symbols.len());

    Ok(symbols)
}
```

### Direct ZPack API

Create and manipulate ZPack archives programmatically:

```rust
use zyntax_compiler::zpack::{ZPack, ZPackBuilder, ZPackManifest};
use std::path::Path;

fn create_custom_zpack() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ZPackBuilder::new("my-runtime", "1.0.0");

    // Set metadata
    builder.set_description("Custom runtime for my language");
    builder.set_language("mylang");
    builder.set_entry_point("Main.main");

    // Add bytecode modules
    builder.add_module(Path::new("modules/std/Array.zbc"))?;
    builder.add_module(Path::new("modules/std/String.zbc"))?;
    builder.add_module_dir(Path::new("modules/core/"))?;

    // Add runtime libraries for different targets
    builder.add_runtime("x86_64-apple-darwin", Path::new("lib/darwin/runtime.zrtl"))?;
    builder.add_runtime("x86_64-unknown-linux-gnu", Path::new("lib/linux/runtime.zrtl"))?;
    builder.add_runtime("aarch64-apple-darwin", Path::new("lib/darwin-arm/runtime.zrtl"))?;

    // Add export declarations
    builder.add_export("$MyLang$print", Some("fn(i32) -> void"), Some("Print an integer"));
    builder.add_export("$MyLang$readLine", Some("fn() -> String"), Some("Read a line from stdin"));

    // Build the archive
    builder.build(Path::new("dist/my-runtime.zpack"))?;

    Ok(())
}

fn load_and_use_zpack() -> Result<(), Box<dyn std::error::Error>> {
    // Load a ZPack
    let zpack = ZPack::load(Path::new("runtime.zpack"))?;

    // Inspect contents
    println!("Package: {} v{}", zpack.manifest().name, zpack.manifest().package_version);
    println!("Modules: {:?}", zpack.manifest().modules);
    println!("Targets: {:?}", zpack.manifest().targets);

    // Get runtime symbols for current platform
    let symbols = zpack.runtime_symbols();
    println!("Loaded {} symbols", symbols.len());

    // Extract bytecode modules
    for module_path in &zpack.manifest().modules {
        let bytecode = zpack.get_module(module_path)?;
        println!("Module {}: {} bytes", module_path, bytecode.len());
    }

    Ok(())
}
```

### Custom JIT Compilation Pipeline

Build a complete custom compilation pipeline:

```rust
use std::sync::Arc;
use zyntax_compiler::{
    compile_to_hir, CompilationConfig,
    cranelift_backend::CraneliftBackend,
    plugin::PluginRegistry,
    zrtl::ZrtlRegistry,
};
use zyntax_typed_ast::{TypedProgram, TypeRegistry};

pub struct CustomCompiler {
    plugin_registry: PluginRegistry,
    zrtl_registry: ZrtlRegistry,
    type_registry: Arc<TypeRegistry>,
    config: CompilationConfig,
}

impl CustomCompiler {
    pub fn new() -> Self {
        Self {
            plugin_registry: PluginRegistry::new(),
            zrtl_registry: ZrtlRegistry::new(),
            type_registry: Arc::new(TypeRegistry::new()),
            config: CompilationConfig::default(),
        }
    }

    pub fn with_opt_level(mut self, level: u8) -> Self {
        self.config.opt_level = level;
        self
    }

    pub fn load_zrtl(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.zrtl_registry.load_plugin(path)?;
        Ok(())
    }

    pub fn register_plugin(&mut self, plugin: Box<dyn zyntax_compiler::plugin::RuntimePlugin>) -> Result<(), String> {
        self.plugin_registry.register(plugin)
    }

    pub fn compile_and_run(&self, program: &TypedProgram) -> Result<i64, Box<dyn std::error::Error>> {
        // Compile to HIR
        let hir_module = compile_to_hir(program, self.type_registry.clone(), self.config.clone())?;

        // Collect all runtime symbols
        let mut symbols = self.plugin_registry.collect_symbols();
        symbols.extend(self.zrtl_registry.collect_symbols());

        // Create backend with runtime symbols
        let mut backend = CraneliftBackend::with_runtime_symbols(&symbols)?;
        backend.compile_module(&hir_module)?;

        // Find and execute main
        let main_id = hir_module.functions.iter()
            .find(|(_, f)| f.name.resolve_global().map(|n| n == "main").unwrap_or(false))
            .map(|(id, _)| *id)
            .ok_or("No main function found")?;

        let fn_ptr = backend.get_function_pointer(main_id)
            .ok_or("Failed to get main function pointer")?;

        let result = unsafe {
            let f: extern "C" fn() -> i64 = std::mem::transmute(fn_ptr);
            f()
        };

        Ok(result)
    }
}

// Usage
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut compiler = CustomCompiler::new()
        .with_opt_level(2);

    // Load runtime plugins
    compiler.load_zrtl("./runtime/mylang.zrtl")?;
    compiler.register_plugin(Box::new(MyRuntimePlugin))?;

    // Parse and compile your program
    let program: TypedProgram = /* parse your source */;
    let result = compiler.compile_and_run(&program)?;

    println!("Program returned: {}", result);
    Ok(())
}
```

### Custom AOT Compilation Pipeline

Build a complete AOT compilation pipeline using the LLVM backend:

```rust
use std::sync::Arc;
use std::path::Path;
use zyntax_compiler::{
    compile_to_hir, CompilationConfig,
    hir::HirModule,
};
use zyntax_typed_ast::{TypedProgram, TypeRegistry};

#[cfg(feature = "llvm-backend")]
use inkwell::{
    context::Context,
    targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
    },
    OptimizationLevel,
};

pub struct AotCompiler {
    type_registry: Arc<TypeRegistry>,
    config: CompilationConfig,
    target_triple: String,
}

impl AotCompiler {
    pub fn new() -> Self {
        Self {
            type_registry: Arc::new(TypeRegistry::new()),
            config: CompilationConfig::default(),
            target_triple: "x86_64-unknown-linux-gnu".to_string(),
        }
    }

    pub fn with_opt_level(mut self, level: u8) -> Self {
        self.config.opt_level = level;
        self
    }

    pub fn with_target(mut self, triple: &str) -> Self {
        self.target_triple = triple.to_string();
        self.config.target_triple = triple.to_string();
        self
    }

    /// Compile TypedAST to HIR
    pub fn compile_to_hir(&self, program: &TypedProgram) -> Result<HirModule, Box<dyn std::error::Error>> {
        let hir = compile_to_hir(program, self.type_registry.clone(), self.config.clone())?;
        Ok(hir)
    }

    /// Compile HIR to object file using LLVM
    #[cfg(feature = "llvm-backend")]
    pub fn compile_to_object(
        &self,
        hir_module: &HirModule,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use zyntax_compiler::llvm_backend::LLVMBackend;

        // Initialize LLVM targets
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| format!("Failed to initialize LLVM target: {}", e))?;

        // Create LLVM context and backend
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "zyntax_aot");

        // Compile HIR to LLVM IR
        let _llvm_ir = backend.compile_module(hir_module)?;

        // Get target machine
        let triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple)?;

        let opt_level = match self.config.opt_level {
            0 => OptimizationLevel::None,
            1 => OptimizationLevel::Less,
            2 => OptimizationLevel::Default,
            _ => OptimizationLevel::Aggressive,
        };

        let target_machine = target
            .create_target_machine(
                &triple,
                "generic",
                "",
                opt_level,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or("Failed to create target machine")?;

        // Write object file
        target_machine
            .write_to_file(backend.module(), FileType::Object, output_path)
            .map_err(|e| format!("Failed to write object file: {}", e))?;

        Ok(())
    }

    /// Compile HIR to LLVM IR string (for debugging)
    #[cfg(feature = "llvm-backend")]
    pub fn compile_to_llvm_ir(&self, hir_module: &HirModule) -> Result<String, Box<dyn std::error::Error>> {
        use zyntax_compiler::llvm_backend::LLVMBackend;

        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "zyntax_aot");
        let ir = backend.compile_module(hir_module)?;
        Ok(ir)
    }

    /// Full pipeline: TypedAST → Object File → Executable
    #[cfg(feature = "llvm-backend")]
    pub fn compile_to_executable(
        &self,
        program: &TypedProgram,
        output_path: &Path,
        static_libs: &[&Path],
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::process::Command;

        // Compile to HIR
        let hir_module = self.compile_to_hir(program)?;

        // Compile to object file
        let obj_path = output_path.with_extension("o");
        self.compile_to_object(&hir_module, &obj_path)?;

        // Link with system linker
        let mut linker = Command::new("cc");
        linker.arg(&obj_path);
        linker.arg("-o");
        linker.arg(output_path);

        // Add static libraries
        for lib in static_libs {
            linker.arg(lib);
        }

        let status = linker.status()?;
        if !status.success() {
            return Err(format!("Linker failed with status: {}", status).into());
        }

        // Clean up object file
        std::fs::remove_file(&obj_path)?;

        Ok(())
    }
}

// Usage
#[cfg(feature = "llvm-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = AotCompiler::new()
        .with_opt_level(3)
        .with_target("x86_64-apple-darwin");

    // Parse your source to TypedAST
    let program: TypedProgram = /* parse source */;

    // Option 1: Just get LLVM IR for inspection
    let hir = compiler.compile_to_hir(&program)?;
    let ir = compiler.compile_to_llvm_ir(&hir)?;
    println!("LLVM IR:\n{}", ir);

    // Option 2: Compile to object file only
    compiler.compile_to_object(&hir, Path::new("output.o"))?;

    // Option 3: Full compilation to executable with runtime linking
    compiler.compile_to_executable(
        &program,
        Path::new("myapp"),
        &[Path::new("/usr/local/lib/libruntime.a")],
    )?;

    Ok(())
}
```

### Cross-Compilation with Custom Targets

```rust
#[cfg(feature = "llvm-backend")]
use inkwell::targets::{Target, TargetTriple, InitializationConfig};

fn initialize_cross_compilation() {
    // Initialize all targets for cross-compilation
    Target::initialize_all(&InitializationConfig::default());
}

fn compile_for_target(
    hir_module: &HirModule,
    target_triple: &str,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use inkwell::context::Context;
    use inkwell::targets::{CodeModel, FileType, RelocMode, Target, TargetMachine, TargetTriple};
    use zyntax_compiler::llvm_backend::LLVMBackend;

    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "cross_compile");
    backend.compile_module(hir_module)?;

    let triple = TargetTriple::create(target_triple);
    let target = Target::from_triple(&triple)?;

    let target_machine = target
        .create_target_machine(
            &triple,
            "generic",  // CPU
            "",         // Features
            inkwell::OptimizationLevel::Aggressive,
            RelocMode::PIC,  // Position-independent code for shared libs
            CodeModel::Default,
        )
        .ok_or("Failed to create target machine")?;

    target_machine.write_to_file(backend.module(), FileType::Object, output_path)?;

    Ok(())
}

// Cross-compile for multiple targets
fn build_all_platforms(hir: &HirModule) -> Result<(), Box<dyn std::error::Error>> {
    initialize_cross_compilation();

    let targets = [
        ("x86_64-apple-darwin", "myapp-darwin-x64.o"),
        ("aarch64-apple-darwin", "myapp-darwin-arm64.o"),
        ("x86_64-unknown-linux-gnu", "myapp-linux-x64.o"),
        ("aarch64-unknown-linux-gnu", "myapp-linux-arm64.o"),
    ];

    for (triple, output) in targets {
        println!("Compiling for {}...", triple);
        compile_for_target(hir, triple, Path::new(output))?;
    }

    Ok(())
}
```

### DynamicValue API for Type-Safe FFI

Use the ZRTL `DynamicValue` type for runtime type-safe values:

```rust
use zyntax_compiler::zrtl::{DynamicValue, TypeId, TypeMeta, TypeRegistry};

// Create dynamic values
let int_val = DynamicValue::from_i32(42);
let float_val = DynamicValue::from_f64(3.14);
let string_val = DynamicValue::from_string("hello".to_string());

// Type checking
assert!(int_val.is_type(TypeId::I32));
assert!(float_val.is_type(TypeId::F64));

// Safe accessors
if let Some(n) = int_val.get_i32() {
    println!("Integer: {}", n);
}

// Register custom types
let mut type_registry = TypeRegistry::new();
let my_type_id = type_registry.register(zyntax_compiler::zrtl::TypeInfo {
    name: "MyStruct".to_string(),
    size: std::mem::size_of::<MyStruct>() as u32,
    alignment: std::mem::align_of::<MyStruct>() as u32,
    dropper: Some(drop_my_struct),
    category: zyntax_compiler::zrtl::TypeCategory::Struct,
});

extern "C" fn drop_my_struct(ptr: *mut u8) {
    unsafe { let _ = Box::from_raw(ptr as *mut MyStruct); }
}
```

### Cargo.toml Dependencies

```toml
[dependencies]
zyntax_compiler = { version = "0.1", features = ["cranelift-backend"] }
zyntax_typed_ast = "0.1"

# Optional: For LLVM backend
# zyntax_compiler = { version = "0.1", features = ["llvm-backend"] }

# For building ZRTL plugins
zrtl_macros = "0.1"
inventory = "0.3"
```

## Summary

| Use Case | Format | API/Command |
|----------|--------|-------------|
| Development/Scripting | ZPack + JIT | `zyntax compile --jit --pack runtime.zpack -g lang.zyn -s main.lang` |
| Production Deployment | Static Library + AOT | `zyntax compile --backend llvm -g lang.zyn -s main.lang -o app --lib runtime` |
| Cross-Platform JIT | Fat ZPack | Include all platform `.zrtl` files in one ZPack |
| Cross-Platform AOT | Per-Platform Builds | Build separate binaries with platform-specific `.a` files |
| Custom JIT Embedding | Rust API | `compile_to_hir()` + `CraneliftBackend` |
| Custom AOT Embedding | Rust API | `compile_to_hir()` + `LLVMBackend` + linker |
| Cross-Compilation | LLVM API | `Target::initialize_all()` + custom `TargetTriple` |
| Plugin Development | RuntimePlugin | Implement `RuntimePlugin` trait or use `zrtl_macros` |

### Key Points

1. **ZPack is for JIT** - Contains dynamic libraries loaded at runtime
2. **Static libraries are for AOT** - Linked at compile time by the system linker
3. **CLI is frontend-agnostic** - Runtime symbols come from packs/libraries, not built-in
4. **Library search** - `--lib foo` searches standard paths for `libfoo.a`
5. **Fat ZPacks** - Include multiple platform runtimes for universal distribution
6. **RuntimePlugin trait** - Embed Zyntax in your Rust application with custom runtime symbols
7. **DynamicValue** - Type-safe FFI for dynamically typed values and generic containers
