# Building Domain-Specific Languages

Zyntax's combination of a flexible grammar system (ZynPEG), rich runtime plugins (ZRTL), and native compilation makes it an ideal platform for creating domain-specific languages (DSLs). This chapter explores how to leverage the full stack to build powerful, specialized languages.

## Why Build DSLs with Zyntax?

Traditional DSL approaches have trade-offs:

| Approach | Pros | Cons |
|----------|------|------|
| Embedded DSLs (macros) | Easy to implement | Limited syntax, host language constraints |
| Interpreter-based | Full syntax control | Slow execution, no native integration |
| Custom compilers | Full control | Massive implementation effort |

**Zyntax offers the best of all worlds:**

- **Full syntax control** via ZynPEG grammars
- **Native performance** via Cranelift/LLVM compilation
- **Rich runtime** via ZRTL plugins (I/O, graphics, networking, etc.)
- **Minimal boilerplate** - just write a `.zyn` grammar file

## DSL Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Your DSL Source                            │
│                    (custom syntax)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ZynPEG Grammar                               │
│              (.zyn file with semantic actions)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TypedAST                                  │
│            (universal typed representation)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Zyntax Compiler                              │
│                (HIR → SSA → Native Code)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ZRTL Plugins                                 │
│    ┌─────┐ ┌─────┐ ┌──────┐ ┌───────┐ ┌─────┐ ┌─────┐        │
│    │ I/O │ │ FS  │ │Window│ │ Paint │ │ SVG │ │ Net │  ...    │
│    └─────┘ └─────┘ └──────┘ └───────┘ └─────┘ └─────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Example DSLs

### 1. Graphics/Visualization DSL

Leverage `zrtl_paint` and `zrtl_window` to create a Processing/p5.js-style creative coding language:

```text
// sketch.art - Creative coding DSL

canvas 800 600

background #1a1a2e

fill #e94560
circle 400 300 100

fill #16213e
for i in 0..10 {
    rect 50 + i * 70, 500, 60, 20
}

stroke #0f3460 width 3
line 0 550 800 550
```

**Grammar highlights:**

```zyn
@language {
    name: "ArtLang",
    version: "1.0",
    file_extensions: [".art"],
    entry_point: "sketch_main"
}

// Map DSL builtins to ZRTL symbols
@builtin {
    canvas_create: "$Paint$canvas_create",
    fill_circle: "$Paint$fill_circle",
    fill_rect: "$Paint$fill_rect",
    set_color: "$Paint$rgb",
}

canvas_stmt = { "canvas" ~ w:integer ~ h:integer }
  -> TypedStatement::Let {
      name: intern("canvas"),
      init: TypedExpression::Call {
          callee: Box::new(TypedExpression::Variable { name: intern("canvas_create") }),
          args: [w, h],
      },
  }

fill_stmt = { "fill" ~ c:color }
  -> TypedStatement::Let {
      name: intern("fill_color"),
      init: c,
  }

circle_stmt = { "circle" ~ x:expr ~ y:expr ~ r:expr }
  -> TypedStatement::Expr {
      expr: TypedExpression::Call {
          callee: Box::new(TypedExpression::Variable { name: intern("fill_circle") }),
          args: [
              TypedExpression::Variable { name: intern("canvas") },
              x, y, r,
              TypedExpression::Variable { name: intern("fill_color") },
          ],
      },
  }
```

The grammar builds TypedAST nodes (`var_decl`, `call`) that reference `@builtin` symbols. During compilation, these resolve to ZRTL plugin functions like `$Paint$fill_circle`.

**Running the DSL:**

```rust
use zyntax_embed::{ZyntaxRuntime, LanguageGrammar};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut runtime = ZyntaxRuntime::new()?;

    // Load required ZRTL plugins for graphics
    runtime.load_plugin("plugins/target/zrtl/zrtl_paint.zrtl")?;
    runtime.load_plugin("plugins/target/zrtl/zrtl_window.zrtl")?;

    // Load the ArtLang grammar
    let grammar = LanguageGrammar::compile_zyn(include_str!("art.zyn"))?;
    runtime.register_grammar("artlang", grammar.clone())?;

    // Compile and run
    let source = std::fs::read_to_string("sketch.art")?;
    runtime.compile_source(&grammar, &source)?;
    runtime.call::<()>("sketch_main", &[])?;

    Ok(())
}
```

### 2. Data Pipeline DSL

Create a language for ETL and data transformation:

```text
// pipeline.flow - Data transformation DSL

source "data/sales.csv" as sales
source "data/products.json" as products

transform sales {
    filter revenue > 1000
    map {
        product_id,
        total: revenue * quantity,
        date: parse_date(sale_date)
    }
}

join sales with products on product_id

aggregate by category {
    total_revenue: sum(total),
    count: count()
}

output "reports/summary.json"
```

**Leveraging ZRTL plugins:**

- `zrtl_fs` for file I/O
- `zrtl_json` for JSON parsing/generation
- `zrtl_string` for text manipulation
- `zrtl_sql` for embedded database operations

### 3. Hardware Description DSL

A simplified HDL for education or prototyping:

```text
// counter.hdl - Hardware description

module counter(clk: clock, reset: bit, out: bits[8]) {
    reg count: bits[8] = 0

    on rising(clk) {
        if reset {
            count <- 0
        } else {
            count <- count + 1
        }
    }

    out <- count
}
```

### 4. Game Scripting DSL

A language for game logic with built-in entity/component concepts:

```text
// player.game - Game entity script

entity Player {
    component Position { x: 0, y: 0 }
    component Velocity { dx: 0, dy: 0 }
    component Sprite { image: "player.png" }

    on update(dt) {
        if key_pressed(KEY_RIGHT) {
            Velocity.dx = 200
        }
        Position.x += Velocity.dx * dt
    }

    on collision(other: Enemy) {
        emit DamageEvent { amount: 10 }
    }
}
```

**Runtime using ZRTL:**

- `zrtl_window` for windowing/input
- `zrtl_paint` for 2D rendering
- `zrtl_image` for sprite loading
- `zrtl_thread` for game loop timing

### 5. Configuration DSL

A type-safe configuration language:

```text
// app.config - Typed configuration

database {
    host: "localhost"
    port: 5432
    pool_size: 10
    timeout: 30s

    ssl {
        enabled: true
        cert_path: "/etc/ssl/cert.pem"
    }
}

server {
    listen: 8080
    workers: cpu_count() * 2

    routes {
        "/api/*" -> api_handler
        "/static/*" -> static_files("./public")
    }
}
```

## Building a Complete DSL: Step by Step

Let's build a simple **charting DSL** that generates visualizations:

### Step 1: Define the Grammar

```zyn
// chart.zyn - Charting DSL grammar

@language {
    name: "ChartLang",
    version: "1.0",
    file_extensions: [".chart"],
    entry_point: "render_chart"
}

// Map builtins to chart plugin symbols
@builtin {
    chart_set_type: "$Chart$set_type",
    chart_set_title: "$Chart$set_title",
    chart_add_data: "$Chart$add_data",
    chart_set_style: "$Chart$set_style",
    chart_render: "$Chart$render",
}

// Entry point: produces TypedProgram containing the render_chart function
program = { SOI ~ chart:chart_definition ~ EOI }
  -> TypedProgram { declarations: [chart] }

chart_definition = { ct:chart_type ~ t:title? ~ d:data_section ~ s:style_section? }
  -> TypedDeclaration::Function {
      name: intern("render_chart"),
      params: [],
      return_type: Type::Named { name: intern("void") },
      body: Some(TypedBlock { stmts: d }),
      is_async: false,
  }

chart_type = @{ ("bar" | "line" | "pie") ~ " chart" }
  -> TypedStatement::Expr {
      expr: TypedExpression::Call {
          callee: Box::new(TypedExpression::Variable { name: intern("chart_set_type") }),
          args: [TypedExpression::StringLiteral { value: chart_type }],
      },
  }

title = { "title" ~ s:string_literal }
  -> TypedStatement::Expr {
      expr: TypedExpression::Call {
          callee: Box::new(TypedExpression::Variable { name: intern("chart_set_title") }),
          args: [s],
      },
  }

data_section = { "data" ~ "{" ~ points:data_point* ~ "}" }
  -> points

data_point = { label:string_literal ~ ":" ~ val:number }
  -> TypedStatement::Expr {
      expr: TypedExpression::Call {
          callee: Box::new(TypedExpression::Variable { name: intern("chart_add_data") }),
          args: [label, val],
      },
  }

style_section = { "style" ~ "{" ~ props:style_prop* ~ "}" }
  -> props

style_prop = { key:identifier ~ ":" ~ val:(color | number | string_literal) }
  -> TypedStatement::Expr {
      expr: TypedExpression::Call {
          callee: Box::new(TypedExpression::Variable { name: intern("chart_set_style") }),
          args: [TypedExpression::StringLiteral { value: key }, val],
      },
  }

// Terminals — atomic rules capture their text automatically via the binding name
string_literal = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" }
  -> TypedExpression::StringLiteral { value: string_literal }

number = @{ "-"? ~ ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT+)? }
  -> TypedExpression::FloatLiteral { value: number }

color = @{ "#" ~ ASCII_HEX_DIGIT{6} }
  -> TypedExpression::StringLiteral { value: color }

identifier = @{ ASCII_ALPHA ~ ASCII_ALPHANUMERIC* }
  -> intern(identifier)

WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
COMMENT = _{ "//" ~ (!"\n" ~ ANY)* }
```

### Step 2: Create the Runtime Plugin

```rust
// plugins/zrtl_chart/src/lib.rs

use std::cell::RefCell;
use zrtl::{zrtl_plugin, StringPtr, string_as_str};

thread_local! {
    static CHART: RefCell<ChartState> = RefCell::new(ChartState::default());
}

#[derive(Default)]
struct ChartState {
    chart_type: String,
    title: String,
    data: Vec<(String, f64)>,
    styles: std::collections::HashMap<String, String>,
}

#[no_mangle]
pub extern "C" fn chart_set_type(type_ptr: StringPtr) {
    let type_str = unsafe { string_as_str(type_ptr) }.unwrap_or("bar");
    CHART.with(|c| c.borrow_mut().chart_type = type_str.to_string());
}

#[no_mangle]
pub extern "C" fn chart_set_title(title_ptr: StringPtr) {
    let title = unsafe { string_as_str(title_ptr) }.unwrap_or("");
    CHART.with(|c| c.borrow_mut().title = title.to_string());
}

#[no_mangle]
pub extern "C" fn chart_add_data(label_ptr: StringPtr, value: f64) {
    let label = unsafe { string_as_str(label_ptr) }.unwrap_or("");
    CHART.with(|c| c.borrow_mut().data.push((label.to_string(), value)));
}

#[no_mangle]
pub extern "C" fn chart_render(width: u32, height: u32) -> u64 {
    CHART.with(|c| {
        let state = c.borrow();

        // Create a paint canvas
        let canvas = zrtl_paint::canvas_create(width, height);

        // Clear with white background
        let white = zrtl_paint::paint_rgb(255, 255, 255);
        zrtl_paint::canvas_clear(canvas, white);

        // Render based on chart type
        match state.chart_type.as_str() {
            "bar" => render_bar_chart(canvas, &state),
            "line" => render_line_chart(canvas, &state),
            "pie" => render_pie_chart(canvas, &state),
            _ => {}
        }

        canvas
    })
}

fn render_bar_chart(canvas: u64, state: &ChartState) {
    let bar_width = 50.0;
    let gap = 20.0;
    let max_value = state.data.iter().map(|(_, v)| *v).fold(0.0_f64, f64::max);
    let scale = 400.0 / max_value;

    for (i, (label, value)) in state.data.iter().enumerate() {
        let x = 50.0 + (i as f32) * (bar_width + gap);
        let height = (*value as f32) * scale as f32;
        let y = 450.0 - height;

        // Draw bar
        let color = zrtl_paint::paint_hex(0x4A90D9);
        zrtl_paint::fill_rect(canvas, x, y, bar_width, height, color);
    }
}

zrtl_plugin! {
    name: "zrtl_chart",
    symbols: [
        ("$Chart$set_type", chart_set_type),
        ("$Chart$set_title", chart_set_title),
        ("$Chart$add_data", chart_add_data),
        ("$Chart$set_style", chart_set_style),
        ("$Chart$render", chart_render),
    ]
}
```

### Step 3: Write DSL Programs

```text
// sales_report.chart

bar chart
title "Q4 Sales by Region"

data {
    "North": 45000
    "South": 32000
    "East": 51000
    "West": 28000
}

style {
    bar_color: #4A90D9
    background: #F5F5F5
    font_size: 14
}
```

### Step 4: Run the DSL

#### Option A: Via CLI

```bash
# Run a chart program - only need the chart plugin
# (it uses zrtl_paint internally via Rust imports)
zyntax compile --grammar chart.zyn \
               --source sales_report.chart \
               --plugins zrtl_chart \
               --run
```

#### Option B: Embedded in Rust Application

```rust
use zyntax_embed::{ZyntaxRuntime, LanguageGrammar};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut runtime = ZyntaxRuntime::new()?;

    // Load only the chart plugin - it handles paint/window internally
    runtime.load_plugin("plugins/target/zrtl/zrtl_chart.zrtl")?;

    // Load ChartLang grammar
    let grammar = LanguageGrammar::compile_zyn(include_str!("chart.zyn"))?;
    runtime.register_grammar("chartlang", grammar.clone())?;

    // Compile and run the chart
    let source = std::fs::read_to_string("sales_report.chart")?;
    runtime.compile_source(&grammar, &source)?;
    runtime.call::<()>("render_chart", &[])?;

    Ok(())
}
```

Note: The `zrtl_chart` plugin uses `zrtl_paint` internally via Rust crate dependencies (not runtime symbol lookup). This is the recommended pattern for DSL plugins - they encapsulate their dependencies and expose only domain-specific symbols like `$Chart$set_type`.

The key insight: `zrtl_plugin!` **defines** what symbols a plugin exports. The runtime **loads** plugins at startup using `load_plugin()` or via the CLI `--plugins` flag. Your DSL grammar then calls those symbols (e.g., `$Chart$set_type`).

## Advanced Patterns

### Domain-Specific Types

Your DSL can define custom type names using atomic rules that capture their text automatically:

```zyn
// Standard primitive types — atomic rule captures text as a type name
primitive_type = @{ "i32" | "i64" | "f32" | "f64" | "bool" | "void" }
  -> Type::Named { name: intern(primitive_type) }

// DSL-specific type aliases — treated the same way
// The compiler treats these as their underlying types
dsl_type = @{ "Currency" | "Percentage" | "Date" | "Duration" }
  -> Type::Named { name: intern(dsl_type) }

// For named types from an identifier (e.g. user-defined structs)
name_type = { n:identifier }
  -> Type::Named { name: intern(n) }

// Combined type expression — passthrough, each alternative handles its own action
type_expr = { primitive_type | dsl_type | name_type }
```

Note: The grammar only parses type names as strings. Type semantics (e.g., treating `Currency` as `f64`) must be handled in your compiler or runtime, not in the grammar.

### Compile-Time Validation

Use grammar predicates for domain validation:

```zyn
// Only allow valid CSS colors
css_color = @{
    "#" ~ ASCII_HEX_DIGIT{3,8} |
    color_name
}

color_name = {
    "red" | "blue" | "green" | "white" | "black" |
    "transparent" | "inherit"
}
```

### Interop with Host Language

DSLs can call back into a host language:

```rust
// Register host functions that DSL code can call
zrtl_plugin! {
    name: "my_app_runtime",
    symbols: [
        ("$App$get_user", get_current_user),
        ("$App$send_email", send_email),
        ("$App$log", app_logger),
    ]
}
```

```text
// In your DSL
user = App.get_user()
if user.is_admin {
    App.send_email(user.email, "Admin Report", report_data)
}
```

## Best Practices

### 1. Start Simple

Begin with a minimal grammar and expand:

```zyn
// Start with just the core concepts
program = { statement* }
statement = { assignment | expression }
```

### 2. Leverage Existing ZRTL Plugins

Don't reinvent the wheel. ZRTL provides:

| Domain | Plugins |
|--------|---------|
| Graphics | `zrtl_window`, `zrtl_paint`, `zrtl_svg`, `zrtl_image` |
| I/O | `zrtl_io`, `zrtl_fs` |
| Data | `zrtl_json`, `zrtl_xml`, `zrtl_sql` |
| Network | `zrtl_net`, `zrtl_http`, `zrtl_websocket` |
| System | `zrtl_process`, `zrtl_thread`, `zrtl_time` |
| Text | `zrtl_string`, `zrtl_regex` |
| Security | `zrtl_crypto`, `zrtl_compress` |

### 3. Design for Readability

Your DSL syntax should be intuitive for domain experts:

```text
// Good: Reads like natural language
send email to "user@example.com" with subject "Hello"

// Avoid: Too programmer-focused
Email.send({to: "user@example.com", subject: "Hello"})
```

### 4. Provide Good Error Messages

Use grammar rules to catch common mistakes:

```zyn
// Catch missing semicolons with helpful error
statement = {
    valid_statement ~ ";" |
    valid_statement ~ &(statement | EOI) ~ PUSH_ERROR("missing semicolon")
}
```

### 5. Document Runtime Symbols

Create clear documentation for available runtime functions:

```markdown
## Available Functions

### Drawing
- `circle(x, y, radius)` - Draw a filled circle
- `rect(x, y, width, height)` - Draw a filled rectangle
- `line(x1, y1, x2, y2)` - Draw a line

### Colors
- `fill(color)` - Set fill color for shapes
- `stroke(color)` - Set stroke color for outlines
```

## Summary

Building DSLs with Zyntax gives you:

1. **Full syntax control** - ZynPEG lets you design the perfect syntax for your domain
2. **Native performance** - Compiles to optimized machine code
3. **Rich ecosystem** - 20+ ZRTL plugins for common functionality
4. **Easy integration** - Custom runtime plugins in Rust
5. **Cross-platform** - Works anywhere Zyntax runs

Whether you're building a configuration language, a data transformation tool, a game scripting system, or a visualization DSL, Zyntax provides the foundation to create powerful, domain-specific languages with minimal effort.
