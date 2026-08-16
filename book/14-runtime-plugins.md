# Runtime Plugins (ZRTL)

The Zyntax Runtime Library (ZRTL) provides native functionality to languages built with Zyntax. Rather than implementing I/O, networking, or threading in each language, ZRTL plugins provide a consistent, high-performance native layer that any Zyntax-based language can use.

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    Your Language                        │
│              (Haxe, Zig, Custom DSL)                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Zyntax Compiler                       │
│            (Grammar → TypedAST → HIR → Code)            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    ZRTL Plugins                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │  I/O    │ │   FS    │ │  Time   │ │  Net    │  ...  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Operating System                      │
└─────────────────────────────────────────────────────────┘
```

## Plugin Types

ZRTL plugins are built as:

- **Dynamic libraries** (`.zrtl` files) - For JIT execution and runtime loading
- **Static libraries** (`.a` files) - For AOT compilation and standalone binaries

## Standard Plugins

### zrtl_io - Input/Output

Basic I/O operations for console interaction.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$IO$print` | `(StringPtr) -> void` | Print string without newline |
| `$IO$println` | `(StringPtr) -> void` | Print string with newline |
| `$IO$print_int` | `(i64) -> void` | Print integer |
| `$IO$print_float` | `(f64) -> void` | Print float |
| `$IO$print_bool` | `(i32) -> void` | Print boolean |
| `$IO$read_line` | `() -> StringPtr` | Read line from stdin |
| `$IO$flush` | `() -> void` | Flush stdout |
| `$IO$eprint` | `(StringPtr) -> void` | Print to stderr |
| `$IO$eprintln` | `(StringPtr) -> void` | Print to stderr with newline |

### zrtl_fs - File System

File and directory operations.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$FS$read_file` | `(StringPtr) -> StringPtr` | Read entire file to string |
| `$FS$write_file` | `(StringPtr, StringPtr) -> i32` | Write string to file |
| `$FS$append_file` | `(StringPtr, StringPtr) -> i32` | Append to file |
| `$FS$exists` | `(StringPtr) -> i32` | Check if path exists |
| `$FS$is_file` | `(StringPtr) -> i32` | Check if path is file |
| `$FS$is_dir` | `(StringPtr) -> i32` | Check if path is directory |
| `$FS$mkdir` | `(StringPtr) -> i32` | Create directory |
| `$FS$mkdir_all` | `(StringPtr) -> i32` | Create directory recursively |
| `$FS$remove` | `(StringPtr) -> i32` | Remove file |
| `$FS$remove_dir` | `(StringPtr) -> i32` | Remove empty directory |
| `$FS$remove_dir_all` | `(StringPtr) -> i32` | Remove directory recursively |
| `$FS$rename` | `(StringPtr, StringPtr) -> i32` | Rename/move file |
| `$FS$copy` | `(StringPtr, StringPtr) -> i32` | Copy file |
| `$FS$list_dir` | `(StringPtr) -> ArrayPtr` | List directory contents |
| `$FS$file_size` | `(StringPtr) -> i64` | Get file size in bytes |

### zrtl_time - Time & Duration

Time operations and sleeping.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Time$now_secs` | `() -> i64` | Unix timestamp (seconds) |
| `$Time$now_millis` | `() -> i64` | Unix timestamp (milliseconds) |
| `$Time$now_micros` | `() -> i64` | Unix timestamp (microseconds) |
| `$Time$monotonic_nanos` | `() -> u64` | Monotonic time (nanoseconds) |
| `$Time$sleep_secs` | `(u64) -> void` | Sleep for seconds |
| `$Time$sleep_millis` | `(u64) -> void` | Sleep for milliseconds |
| `$Time$sleep_micros` | `(u64) -> void` | Sleep for microseconds |
| `$Time$instant_now` | `() -> u64` | Create timing instant |
| `$Time$instant_elapsed_nanos` | `(u64) -> u64` | Nanoseconds since instant |
| `$Time$instant_elapsed_millis` | `(u64) -> u64` | Milliseconds since instant |
| `$Time$instant_free` | `(u64) -> void` | Free instant handle |
| `$Time$format_iso8601` | `(i64) -> StringPtr` | Format as ISO 8601 |

### zrtl_thread - Threading & Atomics

Concurrency primitives.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Thread$spawn` | `(fn(i64)->i64, i64) -> u64` | Spawn thread with function |
| `$Thread$spawn_closure` | `(*ZrtlClosure, i64) -> u64` | Spawn with closure |
| `$Thread$join` | `(u64) -> i64` | Wait for thread, get result |
| `$Thread$current_id` | `() -> u64` | Get current thread ID |
| `$Thread$yield_now` | `() -> void` | Yield to scheduler |
| `$Thread$park` | `() -> void` | Park current thread |
| `$Thread$unpark` | `(u64) -> i32` | Unpark thread by handle |
| `$Thread$available_parallelism` | `() -> u32` | Get CPU core count |
| `$Atomic$new` | `(i64) -> u64` | Create atomic i64 |
| `$Atomic$load` | `(u64) -> i64` | Load atomically |
| `$Atomic$store` | `(u64, i64) -> void` | Store atomically |
| `$Atomic$add` | `(u64, i64) -> i64` | Fetch-add |
| `$Atomic$compare_exchange` | `(u64, i64, i64) -> i32` | CAS operation |
| `$Mutex$new` | `() -> u64` | Create mutex |
| `$Mutex$lock` | `(u64) -> i32` | Lock mutex |
| `$Mutex$try_lock` | `(u64) -> i32` | Try lock (non-blocking) |
| `$Mutex$unlock` | `(u64) -> i32` | Unlock mutex |

### zrtl_net - Networking

TCP and UDP socket operations.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Net$tcp_connect` | `(StringPtr) -> u64` | Connect to TCP server |
| `$Net$tcp_connect_timeout` | `(StringPtr, u64) -> u64` | Connect with timeout (ms) |
| `$Net$tcp_read` | `(u64, u32) -> ArrayPtr` | Read bytes from connection |
| `$Net$tcp_write` | `(u64, ArrayPtr) -> i64` | Write bytes |
| `$Net$tcp_write_string` | `(u64, StringPtr) -> i64` | Write string |
| `$Net$tcp_close` | `(u64) -> void` | Close connection |
| `$Net$tcp_listen` | `(StringPtr) -> u64` | Create TCP listener |
| `$Net$tcp_accept` | `(u64) -> u64` | Accept connection |
| `$Net$tcp_listener_close` | `(u64) -> void` | Close listener |
| `$Net$udp_bind` | `(StringPtr) -> u64` | Create UDP socket |
| `$Net$udp_send_to` | `(u64, StringPtr, ArrayPtr) -> i64` | Send UDP packet |
| `$Net$udp_recv` | `(u64, u32) -> ArrayPtr` | Receive UDP packet |
| `$Net$udp_close` | `(u64) -> void` | Close UDP socket |

### zrtl_env - Environment

Environment variables and process information.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Env$get` | `(StringPtr) -> StringPtr` | Get env variable |
| `$Env$set` | `(StringPtr, StringPtr) -> i32` | Set env variable |
| `$Env$remove` | `(StringPtr) -> void` | Remove env variable |
| `$Env$has` | `(StringPtr) -> i32` | Check if env var exists |
| `$Env$args_count` | `() -> i32` | Get argument count |
| `$Env$arg` | `(i32) -> StringPtr` | Get argument at index |
| `$Env$exe_path` | `() -> StringPtr` | Get executable path |
| `$Env$exit` | `(i32) -> !` | Exit process |
| `$Env$pid` | `() -> u32` | Get process ID |
| `$Env$home_dir` | `() -> StringPtr` | Get home directory |
| `$Env$temp_dir` | `() -> StringPtr` | Get temp directory |
| `$Env$current_dir` | `() -> StringPtr` | Get working directory |
| `$Env$os` | `() -> StringPtr` | Get OS name |
| `$Env$arch` | `() -> StringPtr` | Get CPU architecture |

### zrtl_math - Mathematics

Mathematical functions for numerical computation.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Math$sin` | `(f64) -> f64` | Sine (radians) |
| `$Math$cos` | `(f64) -> f64` | Cosine (radians) |
| `$Math$tan` | `(f64) -> f64` | Tangent (radians) |
| `$Math$asin` | `(f64) -> f64` | Arcsine |
| `$Math$acos` | `(f64) -> f64` | Arccosine |
| `$Math$atan` | `(f64) -> f64` | Arctangent |
| `$Math$atan2` | `(f64, f64) -> f64` | Two-argument arctangent |
| `$Math$sinh` | `(f64) -> f64` | Hyperbolic sine |
| `$Math$cosh` | `(f64) -> f64` | Hyperbolic cosine |
| `$Math$tanh` | `(f64) -> f64` | Hyperbolic tangent |
| `$Math$exp` | `(f64) -> f64` | e^x |
| `$Math$exp2` | `(f64) -> f64` | 2^x |
| `$Math$log` | `(f64) -> f64` | Natural logarithm |
| `$Math$log2` | `(f64) -> f64` | Base-2 logarithm |
| `$Math$log10` | `(f64) -> f64` | Base-10 logarithm |
| `$Math$pow` | `(f64, f64) -> f64` | Power (x^y) |
| `$Math$sqrt` | `(f64) -> f64` | Square root |
| `$Math$cbrt` | `(f64) -> f64` | Cube root |
| `$Math$hypot` | `(f64, f64) -> f64` | Hypotenuse |
| `$Math$floor` | `(f64) -> f64` | Round down |
| `$Math$ceil` | `(f64) -> f64` | Round up |
| `$Math$round` | `(f64) -> f64` | Round to nearest |
| `$Math$trunc` | `(f64) -> f64` | Truncate toward zero |
| `$Math$abs` | `(f64) -> f64` | Absolute value (float) |
| `$Math$abs_i64` | `(i64) -> i64` | Absolute value (int) |
| `$Math$min` | `(f64, f64) -> f64` | Minimum |
| `$Math$max` | `(f64, f64) -> f64` | Maximum |
| `$Math$clamp` | `(f64, f64, f64) -> f64` | Clamp to range |
| `$Math$random` | `() -> f64` | Random [0, 1) |
| `$Math$random_range` | `(f64, f64) -> f64` | Random in range |
| `$Math$random_int` | `(i64, i64) -> i64` | Random integer |
| `$Math$seed` | `(u64) -> void` | Seed RNG |
| `$Math$pi` | `() -> f64` | Pi constant |
| `$Math$e` | `() -> f64` | Euler's number |
| `$Math$tau` | `() -> f64` | Tau (2π) |
| `$Math$lerp` | `(f64, f64, f64) -> f64` | Linear interpolation |
| `$Math$to_radians` | `(f64) -> f64` | Degrees to radians |
| `$Math$to_degrees` | `(f64) -> f64` | Radians to degrees |

### zrtl_string - String Manipulation

String manipulation and formatting functions.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$length` | `(StringPtr) -> i64` | Get byte length |
| `$String$char_count` | `(StringPtr) -> i64` | Get character count (Unicode) |
| `$String$is_empty` | `(StringPtr) -> i32` | Check if empty |
| `$String$concat` | `(StringPtr, StringPtr) -> StringPtr` | Concatenate strings |
| `$String$repeat` | `(StringPtr, i64) -> StringPtr` | Repeat n times |
| `$String$to_upper` | `(StringPtr) -> StringPtr` | Convert to uppercase |
| `$String$to_lower` | `(StringPtr) -> StringPtr` | Convert to lowercase |
| `$String$trim` | `(StringPtr) -> StringPtr` | Trim whitespace |
| `$String$contains` | `(StringPtr, StringPtr) -> i32` | Check if contains |
| `$String$starts_with` | `(StringPtr, StringPtr) -> i32` | Check prefix |
| `$String$ends_with` | `(StringPtr, StringPtr) -> i32` | Check suffix |
| `$String$index_of` | `(StringPtr, StringPtr) -> i64` | Find first index |
| `$String$replace_all` | `(StringPtr, StringPtr, StringPtr) -> StringPtr` | Replace all |
| `$String$substring` | `(StringPtr, i64, i64) -> StringPtr` | Extract substring |
| `$String$split` | `(StringPtr, StringPtr) -> ArrayPtr` | Split by delimiter |
| `$String$parse_int` | `(StringPtr) -> i64` | Parse as integer |
| `$String$parse_float` | `(StringPtr) -> f64` | Parse as float |
| `$String$from_int` | `(i64) -> StringPtr` | Integer to string |
| `$String$from_float` | `(f64) -> StringPtr` | Float to string |

### zrtl_json - JSON Parsing

JSON parsing and manipulation using opaque handles.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Json$parse` | `(StringPtr) -> u64` | Parse JSON string |
| `$Json$stringify` | `(u64) -> StringPtr` | Convert to JSON string |
| `$Json$free` | `(u64) -> void` | Free JSON handle |
| `$Json$get_type` | `(u64) -> i32` | Get type (0=null,1=bool,2=num,3=str,4=arr,5=obj) |
| `$Json$get_bool` | `(u64) -> i32` | Get boolean value |
| `$Json$get_int` | `(u64) -> i64` | Get integer value |
| `$Json$get_float` | `(u64) -> f64` | Get float value |
| `$Json$get_string` | `(u64) -> StringPtr` | Get string value |
| `$Json$get` | `(u64, StringPtr) -> u64` | Get object property |
| `$Json$set` | `(u64, StringPtr, u64) -> void` | Set object property |
| `$Json$array_length` | `(u64) -> i64` | Get array length |
| `$Json$array_get` | `(u64, i64) -> u64` | Get array element |
| `$Json$object` | `() -> u64` | Create empty object |
| `$Json$array` | `() -> u64` | Create empty array |
| `$Json$path_get` | `(u64, StringPtr) -> u64` | Get by dot-path |

### zrtl_regex - Regular Expressions

Regex matching and replacement using compiled patterns.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Regex$compile` | `(StringPtr) -> u64` | Compile pattern |
| `$Regex$free` | `(u64) -> void` | Free pattern handle |
| `$Regex$matches` | `(u64, StringPtr) -> i32` | Check if matches |
| `$Regex$find_match` | `(u64, StringPtr) -> StringPtr` | Find first match |
| `$Regex$find_all` | `(u64, StringPtr) -> ArrayPtr` | Find all matches |
| `$Regex$replace_first` | `(u64, StringPtr, StringPtr) -> StringPtr` | Replace first |
| `$Regex$replace_all_compiled` | `(u64, StringPtr, StringPtr) -> StringPtr` | Replace all |
| `$Regex$capture` | `(u64, StringPtr, i32) -> StringPtr` | Get capture group |
| `$Regex$split` | `(u64, StringPtr) -> ArrayPtr` | Split by pattern |
| `$Regex$is_match` | `(StringPtr, StringPtr) -> i32` | Quick match check |
| `$Regex$replace_all` | `(StringPtr, StringPtr, StringPtr) -> StringPtr` | Quick replace |

### zrtl_process - Subprocess Execution

Process spawning and management.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Process$run` | `(StringPtr, ArrayPtr) -> i32` | Run command, return exit code |
| `$Process$run_capture` | `(StringPtr, ArrayPtr) -> StringPtr` | Run and capture stdout |
| `$Process$shell` | `(StringPtr) -> i32` | Run shell command |
| `$Process$shell_capture` | `(StringPtr) -> StringPtr` | Shell with capture |
| `$Process$spawn` | `(StringPtr, ArrayPtr) -> u64` | Spawn process |
| `$Process$wait` | `(u64) -> i32` | Wait for completion |
| `$Process$kill` | `(u64) -> i32` | Kill process |
| `$Process$is_running` | `(u64) -> i32` | Check if running |
| `$Process$read_stdout` | `(u64) -> StringPtr` | Read stdout |
| `$Process$write_stdin` | `(u64, StringPtr) -> i32` | Write to stdin |
| `$Process$current_pid` | `() -> u32` | Get current PID |
| `$Process$command_exists` | `(StringPtr) -> i32` | Check if command exists |

## Graphics & Media Plugins

### zrtl_window - Cross-Platform Windowing

High-level windowing API built on SDL2. Auto-initializes when creating windows.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Window$create` | `(StringPtr, i32, i32) -> u64` | Create centered window |
| `$Window$create_at` | `(StringPtr, i32, i32, i32, i32) -> u64` | Create at position |
| `$Window$destroy` | `(u64) -> void` | Destroy window |
| `$Window$is_open` | `(u64) -> i32` | Check if window open |
| `$Window$close` | `(u64) -> void` | Mark window for closing |
| `$Window$set_title` | `(u64, StringPtr) -> i32` | Set window title |
| `$Window$get_size` | `(u64) -> WindowSize` | Get window dimensions |
| `$Window$clear` | `(u64, u32) -> void` | Clear with RGBA color |
| `$Window$present` | `(u64) -> void` | Present frame |
| `$Window$draw_pixel` | `(u64, i32, i32) -> i32` | Draw single pixel |
| `$Window$draw_line` | `(u64, i32, i32, i32, i32) -> i32` | Draw line |
| `$Window$draw_rect` | `(u64, i32, i32, i32, i32) -> i32` | Draw rectangle outline |
| `$Window$fill_rect` | `(u64, i32, i32, i32, i32) -> i32` | Fill rectangle |
| `$Window$fill_rect_color` | `(u64, i32, i32, i32, i32, u32) -> i32` | Fill with color |
| `$Window$blit` | `(u64, *u8, i32, i32, i32, i32, i32) -> i32` | Blit pixel buffer |
| `$Window$blit_scaled` | `(...) -> i32` | Blit with scaling |
| `$Window$poll_event` | `() -> WindowEvent` | Poll events (non-blocking) |
| `$Window$wait_event` | `() -> WindowEvent` | Wait for event |
| `$Window$delay` | `(u32) -> void` | Sleep milliseconds |

**Color format**: RGBA as `0xRRGGBBAA` (e.g., `0xFF0000FF` = red)

**Event types**: `EVENT_QUIT`, `EVENT_KEY_DOWN`, `EVENT_KEY_UP`, `EVENT_MOUSE_MOVE`, `EVENT_MOUSE_DOWN`, `EVENT_MOUSE_UP`, `EVENT_RESIZE`

### zrtl_sdl - Low-Level SDL2 Bindings

Direct SDL2 access for advanced use cases. Requires explicit initialization.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Sdl$init` | `() -> i32` | Initialize SDL |
| `$Sdl$quit` | `() -> void` | Quit SDL |
| `$Sdl$window_create` | `(StringPtr, i32, i32, u32, u32) -> u64` | Create window |
| `$Sdl$window_create_centered` | `(StringPtr, u32, u32) -> u64` | Create centered |
| `$Sdl$set_draw_color` | `(u64, u8, u8, u8, u8) -> void` | Set RGBA color |
| `$Sdl$clear` | `(u64) -> void` | Clear canvas |
| `$Sdl$present` | `(u64) -> void` | Present frame |
| `$Sdl$draw_point` | `(u64, i32, i32) -> i32` | Draw point |
| `$Sdl$draw_line` | `(u64, i32, i32, i32, i32) -> i32` | Draw line |
| `$Sdl$draw_rect` | `(u64, i32, i32, u32, u32) -> i32` | Draw rect outline |
| `$Sdl$fill_rect` | `(u64, i32, i32, u32, u32) -> i32` | Fill rect |
| `$Sdl$blit_rgba` | `(u64, *u8, u32, u32, u32, i32, i32) -> i32` | Blit RGBA buffer |
| `$Sdl$poll_event` | `() -> SdlEvent` | Poll events |
| `$Sdl$wait_event` | `() -> SdlEvent` | Wait for event |
| `$Sdl$get_ticks` | `() -> u32` | Milliseconds since init |
| `$Sdl$delay` | `(u32) -> void` | Sleep milliseconds |

### zrtl_paint - 2D Software Rasterization

High-quality 2D rendering with anti-aliased shapes, paths, and transforms.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Paint$canvas_create` | `(u32, u32) -> u64` | Create canvas |
| `$Paint$canvas_free` | `(u64) -> void` | Free canvas |
| `$Paint$canvas_clear` | `(u64, Color) -> void` | Clear with color |
| `$Paint$canvas_get_buffer` | `(u64) -> CanvasBuffer` | Get pixel buffer info |
| `$Paint$canvas_data_ptr` | `(u64) -> *u8` | Get raw pixel pointer |
| `$Paint$canvas_save_png` | `(u64, StringPtr) -> i32` | Save to PNG file |
| `$Paint$rgb` | `(u8, u8, u8) -> Color` | Create RGB color |
| `$Paint$rgba` | `(u8, u8, u8, u8) -> Color` | Create RGBA color |
| `$Paint$hex` | `(u32) -> Color` | Color from hex |
| `$Paint$fill_rect` | `(u64, f32, f32, f32, f32, Color)` | Fill rectangle |
| `$Paint$stroke_rect` | `(...)` | Stroke rectangle |
| `$Paint$fill_circle` | `(u64, f32, f32, f32, Color)` | Fill circle |
| `$Paint$stroke_circle` | `(...)` | Stroke circle |
| `$Paint$fill_ellipse` | `(u64, f32, f32, f32, f32, Color)` | Fill ellipse |
| `$Paint$fill_rounded_rect` | `(..., f32, Color)` | Fill rounded rect |
| `$Paint$draw_line` | `(u64, f32, f32, f32, f32, Color)` | Draw line |
| `$Paint$path_create` | `() -> u64` | Create path builder |
| `$Paint$path_move_to` | `(u64, f32, f32)` | Move to point |
| `$Paint$path_line_to` | `(u64, f32, f32)` | Line to point |
| `$Paint$path_quad_to` | `(u64, f32, f32, f32, f32)` | Quadratic bezier |
| `$Paint$path_cubic_to` | `(...)` | Cubic bezier |
| `$Paint$path_fill` | `(u64, u64, Color)` | Fill path |
| `$Paint$path_stroke` | `(u64, u64, Color)` | Stroke path |
| `$Paint$transform_translate` | `(u64, f32, f32)` | Apply translation |
| `$Paint$transform_rotate` | `(u64, f32)` | Apply rotation (radians) |
| `$Paint$transform_scale` | `(u64, f32, f32)` | Apply scale |
| `$Paint$set_stroke_width` | `(u64, f32)` | Set stroke width |

**Rendering to Window**: Use `canvas_get_buffer()` to get pixel data, then `$Window$blit()`:

```
let canvas = $Paint$canvas_create(800, 600);
$Paint$fill_circle(canvas, 400, 300, 100, $Paint$rgb(255, 0, 0));

let buffer = $Paint$canvas_get_buffer(canvas);
$Window$blit(win, buffer.data, buffer.width, buffer.height, buffer.pitch, 0, 0);
$Window$present(win);
```

### zrtl_svg - SVG Parsing & Rendering

Parse and render SVG files using resvg/usvg.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Svg$parse` | `(StringPtr) -> u64` | Parse SVG from string |
| `$Svg$parse_file` | `(StringPtr) -> u64` | Parse SVG from file |
| `$Svg$free` | `(u64) -> void` | Free SVG handle |
| `$Svg$get_width` | `(u64) -> f32` | Get SVG width |
| `$Svg$get_height` | `(u64) -> f32` | Get SVG height |
| `$Svg$render` | `(u64) -> RenderResult` | Render at natural size |
| `$Svg$render_scaled` | `(u64, f32) -> RenderResult` | Render with scale |
| `$Svg$pixmap_data` | `(u64) -> *u8` | Get rendered pixel data |
| `$Svg$pixmap_save_png` | `(u64, StringPtr) -> i32` | Save pixmap to PNG |
| `$Svg$pixmap_free` | `(u64) -> void` | Free rendered pixmap |

### zrtl_image - Image Encoding/Decoding

Load and save images in common formats (PNG, JPEG, GIF, WebP, BMP, ICO, TIFF).

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Image$load` | `(StringPtr) -> u64` | Load image from file |
| `$Image$load_from_memory` | `(ArrayPtr) -> u64` | Load from bytes |
| `$Image$free` | `(u64) -> void` | Free image |
| `$Image$width` | `(u64) -> u32` | Get width |
| `$Image$height` | `(u64) -> u32` | Get height |
| `$Image$get_pixel` | `(u64, u32, u32) -> u32` | Get RGBA pixel |
| `$Image$set_pixel` | `(u64, u32, u32, u32) -> void` | Set RGBA pixel |
| `$Image$save_png` | `(u64, StringPtr) -> i32` | Save as PNG |
| `$Image$save_jpeg` | `(u64, StringPtr, u8) -> i32` | Save as JPEG (quality) |
| `$Image$resize` | `(u64, u32, u32) -> u64` | Resize image |
| `$Image$crop` | `(u64, u32, u32, u32, u32) -> u64` | Crop region |
| `$Image$flip_horizontal` | `(u64) -> void` | Flip horizontally |
| `$Image$flip_vertical` | `(u64) -> void` | Flip vertically |
| `$Image$rotate90` | `(u64) -> u64` | Rotate 90 degrees |
| `$Image$to_grayscale` | `(u64) -> u64` | Convert to grayscale |
| `$Image$get_data` | `(u64) -> *u8` | Get raw RGBA data |

### zrtl_xml - XML Parsing & Generation

XML document manipulation with quick-xml.

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$Xml$parse` | `(StringPtr) -> u64` | Parse XML string |
| `$Xml$parse_file` | `(StringPtr) -> u64` | Parse XML file |
| `$Xml$free` | `(u64) -> void` | Free document |
| `$Xml$stringify` | `(u64) -> StringPtr` | Convert to string |
| `$Xml$root` | `(u64) -> u64` | Get root element |
| `$Xml$tag_name` | `(u64) -> StringPtr` | Get element tag name |
| `$Xml$text_content` | `(u64) -> StringPtr` | Get text content |
| `$Xml$get_attribute` | `(u64, StringPtr) -> StringPtr` | Get attribute |
| `$Xml$set_attribute` | `(u64, StringPtr, StringPtr) -> void` | Set attribute |
| `$Xml$children` | `(u64) -> ArrayPtr` | Get child elements |
| `$Xml$find_by_tag` | `(u64, StringPtr) -> ArrayPtr` | Find by tag name |
| `$Xml$create_element` | `(StringPtr) -> u64` | Create element |
| `$Xml$append_child` | `(u64, u64) -> void` | Append child |

## Building Plugins

### Dynamic Libraries (.zrtl)

```bash
cd plugins
./build_zrtl.sh --release
```

Output in `plugins/target/zrtl/`:

```text
zrtl_io.zrtl
zrtl_fs.zrtl
zrtl_time.zrtl
zrtl_thread.zrtl
zrtl_net.zrtl
zrtl_env.zrtl
zrtl_math.zrtl
zrtl_string.zrtl
zrtl_json.zrtl
zrtl_regex.zrtl
zrtl_process.zrtl
plugins.json
```

### Static Libraries (.a)

```bash
cd plugins
./build_static.sh --release

# Cross-compile for specific target
./build_static.sh --target aarch64-apple-darwin

# Build for all supported targets
./build_static.sh --all
```

Output in `plugins/target/static/<target>/`:

```text
libzrtl_io.a
libzrtl_fs.a
...
libs.json
```

## Using Plugins in Your Language

### Grammar Integration

Map your language's standard library to ZRTL symbols:

```zyn
@builtin {
    // I/O
    print: "$IO$print",
    println: "$IO$println",
    readLine: "$IO$read_line",

    // File System
    readFile: "$FS$read_file",
    writeFile: "$FS$write_file",

    // Time
    now: "$Time$now_millis",
    sleep: "$Time$sleep_millis",
}
```

### In Your Language Source

```zig
// Your custom language
fn main() {
    println("Hello from ZRTL!");

    let contents = readFile("data.txt");
    println(contents);

    sleep(1000);  // Sleep 1 second
}
```

### Loading at Runtime (JIT)

ZRTL plugins are loaded at runtime using the `ZyntaxRuntime` API:

```rust
use zyntax_embed::{ZyntaxRuntime, LanguageGrammar};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Create a new runtime
    let mut runtime = ZyntaxRuntime::new()?;

    // Step 2: Load ZRTL plugins (dynamic libraries)
    // Option A: Load individual plugins
    runtime.load_plugin("plugins/target/zrtl/zrtl_io.zrtl")?;
    runtime.load_plugin("plugins/target/zrtl/zrtl_fs.zrtl")?;

    // Option B: Load all plugins from a directory
    let count = runtime.load_plugins_from_directory("plugins/target/zrtl")?;
    println!("Loaded {} plugins", count);

    // Step 3: Load your language grammar
    let grammar = LanguageGrammar::compile_zyn(include_str!("my_lang.zyn"))?;
    runtime.register_grammar("mylang", grammar.clone())?;
    runtime.map_extension("mylang", "ml")?;  // .ml files use this grammar

    // Step 4: Compile source code that uses plugin functions
    let source = r#"
        fn main() {
            // These calls resolve to $IO$println and $FS$read_file
            println("Hello from ZRTL!");
            let contents = readFile("data.txt");
            println(contents);
        }
    "#;

    runtime.compile_source(&grammar, source)?;

    // Step 5: Call the compiled function
    runtime.call::<()>("main", &[])?;

    Ok(())
}
```

### Loading Plugins with Custom Symbols

For advanced use cases, you can also register external symbols directly:

```rust
use zyntax_embed::ZyntaxRuntime;

// Define a native function
extern "C" fn my_custom_print(msg: i64) {
    println!("Custom: {}", msg);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime with custom symbols
    let symbols: &[(&str, *const u8)] = &[
        ("$Custom$print", my_custom_print as *const u8),
    ];
    let mut runtime = ZyntaxRuntime::with_symbols(symbols)?;

    // Now code can call $Custom$print
    // ...

    Ok(())
}
```

### Linking Plugins Into the Host

A host that knows which plugins it needs can link them in and register
them directly, instead of opening `.zrtl` files at startup. Depend on
the plugin crates and hand the runtime their `static_plugin()`
accessors:

```rust
runtime.register_static_plugins([
    zrtl_io::static_plugin(),
    zrtl_fs::static_plugin(),
])?;
```

This is the only way to load plugins on wasm32, where there is no
`dlopen`. On native it removes one `dlopen` per plugin from startup,
which is worth doing for a CLI that runs one program and exits: opening
eight plugins costs several milliseconds warm, and considerably more
when the files are not in the page cache.

Register before installing a grammar. Builtin mappings such as `println`
to `$IO$println_dynamic` are resolved as the language is installed, and
the signatures a plugin carries are read while lowering, so they have to
be present before a program is compiled rather than when it is run.

`register_static_plugin` registers one plugin and rebuilds the JIT
module. `register_static_plugins` takes a set and rebuilds once for all
of them, so prefer it when the whole set is known up front.

### Static Linking (AOT)

For standalone executables, link against static libraries:

```bash
# Compile your language to object file
zyntax compile --source main.mylang --output main.o --aot

# Link with ZRTL static libs
cc main.o \
    -L plugins/target/static/aarch64-apple-darwin \
    -lzrtl_io -lzrtl_fs -lzrtl_time \
    -o main
```

## Memory Management

ZRTL uses specific memory conventions:

### Strings

ZRTL strings have inline length: `[i32 length][utf8 bytes...]`

```rust
// Creating strings
let s: StringPtr = string_new("hello");

// Reading strings
let len = string_length(s);
let data = string_data(s);

// Freeing strings (caller responsibility for returned strings)
string_free(s);
```

### Arrays

ZRTL arrays: `[i32 capacity][i32 length][elements...]`

```rust
let arr: ArrayPtr = array_new::<i32>(10);  // capacity 10
array_push(arr, 42);
let val = array_get::<i32>(arr, 0);
array_free(arr);
```

### Handle-based Resources

File handles, sockets, threads, etc. return `u64` handles:

```rust
let handle = tcp_connect(addr);  // Returns handle
// ... use handle ...
tcp_close(handle);  // Must close when done
```

## Creating Custom Plugins

### Plugin Structure

```rust
// my_plugin/src/lib.rs
use zrtl::{zrtl_plugin, StringPtr, string_new};

#[no_mangle]
pub extern "C" fn my_greet(name: StringPtr) -> StringPtr {
    let name_str = unsafe { zrtl::string_as_str(name) }.unwrap_or("World");
    string_new(&format!("Hello, {}!", name_str))
}

zrtl_plugin! {
    name: "my_plugin",
    symbols: [
        ("$My$greet", my_greet),
    ]
}
```

### Cargo.toml

```toml
[package]
name = "my_plugin"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "staticlib", "rlib"]

[dependencies]
zrtl = { path = "../../sdk/zrtl" }
```

### Adding to Workspace

Add your plugin directory to `plugins/Cargo.toml`:

```toml
[workspace]
members = [
    "zrtl_io",
    "zrtl_fs",
    # ... existing plugins
    "my_plugin",  # Your new plugin
]
```

The build scripts will automatically discover and build it.

## Symbol Naming Convention

ZRTL uses a hierarchical naming scheme:

```text
$Module$function_name
```

Examples:
- `$IO$println` - I/O module, println function
- `$FS$read_file` - File system module, read_file function
- `$Net$tcp_connect` - Network module, tcp_connect function

This allows:
1. Clear organization by domain
2. No conflicts between plugins
3. Easy mapping in grammar `@builtin` blocks
