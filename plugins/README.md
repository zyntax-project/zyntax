# ZRTL Plugins

ZRTL (Zyntax Runtime Library) plugins provide native functionality to Zyntax programs. Each plugin exports symbols that can be called from compiled code.

## Available Plugins

| Plugin | Description | Key Symbols |
|--------|-------------|-------------|
| **zrtl_io** | Console I/O | `$IO$print`, `$IO$println`, `$IO$read_line` |
| **zrtl_fs** | File system operations | `$FS$read_file`, `$FS$write_file`, `$FS$exists` |
| **zrtl_time** | Time and date | `$Time$now`, `$Time$sleep`, `$Time$format` |
| **zrtl_string** | String manipulation | `$String$concat`, `$String$split`, `$String$trim` |
| **zrtl_math** | Math functions | `$Math$sin`, `$Math$cos`, `$Math$sqrt`, `$Math$pow` |
| **zrtl_json** | JSON parsing/generation | `$JSON$parse`, `$JSON$stringify` |
| **zrtl_regex** | Regular expressions | `$Regex$match`, `$Regex$replace`, `$Regex$split` |
| **zrtl_net** | TCP/UDP networking | `$Net$tcp_connect`, `$Net$tcp_listen`, `$Net$udp_bind` |
| **zrtl_http** | HTTP client | `$HTTP$get`, `$HTTP$post`, `$HTTP$request` |
| **zrtl_websocket** | WebSocket client/server | `$WS$connect`, `$WS$send`, `$WS$on_message` |
| **zrtl_thread** | Threading and sync | `$Thread$spawn`, `$Thread$join`, `$Mutex$new` |
| **zrtl_process** | Process management | `$Process$spawn`, `$Process$exec`, `$Process$exit` |
| **zrtl_env** | Environment variables | `$Env$get`, `$Env$set`, `$Env$vars` |
| **zrtl_crypto** | Cryptography | `$Crypto$hash_sha256`, `$Crypto$aes_encrypt` |
| **zrtl_compress** | Compression | `$Compress$gzip`, `$Compress$gunzip`, `$Compress$zstd` |
| **zrtl_sql** | SQLite database | `$SQL$open`, `$SQL$execute`, `$SQL$query` |
| **zrtl_paint** | 2D graphics (tiny-skia) | `$Paint$canvas_create`, `$Paint$fill_circle`, `$Paint$fill_rect` |
| **zrtl_window** | Windowing (SDL2) | `$Window$create`, `$Window$poll_events`, `$Window$present` |
| **zrtl_sdl** | Low-level SDL2 bindings | `$SDL$init`, `$SDL$create_texture`, `$SDL$render_copy` |
| **zrtl_image** | Image loading | `$Image$load`, `$Image$decode`, `$Image$to_canvas` |
| **zrtl_svg** | SVG rendering | `$SVG$parse`, `$SVG$render`, `$SVG$to_canvas` |
| **zrtl_xml** | XML parsing | `$XML$parse`, `$XML$query`, `$XML$serialize` |
| **zrtl_simd** | SIMD-accelerated compute | `$SIMD$dot_product_f32`, `$SIMD$mat4_mul`, `$SIMD$blend_rgba` |

## Building Plugins

### Build All Plugins

```bash
cd plugins
./build_zrtl.sh
```

This creates `.zrtl` plugin files in `target/zrtl/`.

### Build Individual Plugin

```bash
cargo build --release -p zrtl_paint
```

## Using Plugins

### From CLI

```bash
# Load plugins via --plugins flag
zyntax compile --grammar my.zyn --source code.txt --plugins zrtl_io,zrtl_paint --run
```

### From Rust (Embedding SDK)

```rust
use zyntax_embed::ZyntaxRuntime;

let mut runtime = ZyntaxRuntime::new()?;

// Load plugins
runtime.load_plugin("plugins/target/zrtl/zrtl_io.zrtl")?;
runtime.load_plugin("plugins/target/zrtl/zrtl_paint.zrtl")?;

// Now compile and run code that uses $IO$ and $Paint$ symbols
```

### From Grammar (DSL)

Map plugin symbols to DSL builtins:

```zyn
@builtin {
    print: "$IO$println",
    draw_circle: "$Paint$fill_circle",
}
```

## Creating Custom Plugins

Use the `zrtl_plugin!` macro to define a plugin:

```rust
use zyntax_embed::zrtl_plugin;

// Define exported functions
#[no_mangle]
pub extern "C" fn my_function(x: i32) -> i32 {
    x * 2
}

// Export symbols
zrtl_plugin! {
    name: "my_plugin",
    version: "1.0.0",
    symbols: [
        ("$MyPlugin$my_function", my_function),
    ]
}
```

Build as a cdylib:

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib"]

[dependencies]
zyntax_embed = { path = "../crates/zyntax_embed" }
```

See [Chapter 14: Runtime Plugins](https://github.com/zyntax-project/zyntax/wiki/14-Runtime-Plugins) in The Zyn Book for complete documentation.

## Plugin Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    Zyntax Runtime                        │
├─────────────────────────────────────────────────────────┤
│  Symbol Table                                            │
│  ┌─────────────────┬─────────────────────────────────┐  │
│  │ $IO$println     │ → fn(*const u8)                 │  │
│  │ $Paint$canvas   │ → fn(u32, u32) -> u64           │  │
│  │ $Math$sqrt      │ → fn(f64) -> f64                │  │
│  └─────────────────┴─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
          ▲                    ▲                    ▲
          │                    │                    │
    ┌─────┴─────┐        ┌─────┴─────┐        ┌─────┴─────┐
    │  zrtl_io  │        │zrtl_paint │        │ zrtl_math │
    │  .zrtl    │        │  .zrtl    │        │  .zrtl    │
    └───────────┘        └───────────┘        └───────────┘
```

Plugins are loaded dynamically at runtime. Each plugin registers its symbols in the runtime's symbol table, making them available for compiled code to call.

## Symbol Naming Convention

All ZRTL symbols follow the pattern: `$PluginName$function_name`

- `$IO$println` - Print line from IO plugin
- `$Paint$fill_circle` - Fill circle from Paint plugin
- `$Math$sin` - Sine function from Math plugin

This namespacing prevents collisions between plugins.
