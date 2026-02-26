//! ImagePipe DSL Runner
//!
//! A command-line tool for running ImagePipe programs - a domain-specific
//! language for image processing pipelines.
//!
//! ## Usage
//!
//! ```bash
//! # Run a pipeline script
//! imagepipe run examples/enhance.imgpipe
//!
//! # Parse and show the AST (for debugging)
//! imagepipe parse examples/enhance.imgpipe
//!
//! # Interactive REPL mode
//! imagepipe repl
//! ```
//!
//! ## Plugins Used
//!
//! - `zrtl_image` - Image loading, saving, and manipulation
//! - `zrtl_simd` - SIMD-accelerated pixel operations
//! - `zrtl_paint` - 2D graphics primitives
//! - `zrtl_fs` - File system operations
//! - `zrtl_io` - Console I/O

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// ImagePipe - Image Processing Pipeline DSL
#[derive(Parser)]
#[command(name = "imagepipe")]
#[command(about = "Run image processing pipelines written in ImagePipe DSL")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run an ImagePipe program
    Run {
        /// Path to the .imgpipe file
        file: PathBuf,

        /// Directory containing ZRTL plugins
        #[arg(long, default_value = "plugins/target/zrtl")]
        plugins: PathBuf,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Parse an ImagePipe program and show the AST
    Parse {
        /// Path to the .imgpipe file
        file: PathBuf,

        /// Output format: json or tree
        #[arg(long, default_value = "tree")]
        format: String,
    },

    /// Start an interactive REPL
    Repl {
        /// Directory containing ZRTL plugins
        #[arg(long, default_value = "plugins/target/zrtl")]
        plugins: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            file,
            plugins,
            verbose,
        } => run_pipeline(&file, &plugins, verbose),
        Commands::Parse { file, format } => parse_and_display(&file, &format),
        Commands::Repl { plugins } => run_repl(&plugins),
    }
}

/// Run an ImagePipe program using the new Grammar2 runtime
fn run_pipeline(file: &PathBuf, plugins_dir: &PathBuf, verbose: bool) -> Result<()> {
    use zyntax_embed::{Grammar2, ZyntaxRuntime};

    if verbose {
        println!("Loading ImagePipe grammar (v2.0)...");
    }

    // Load the grammar using Grammar2 (new GrammarInterpreter-based parser)
    let grammar_source = include_str!("../imagepipe.zyn");
    let grammar =
        Grammar2::from_source(grammar_source).context("Failed to compile ImagePipe grammar")?;

    if verbose {
        println!("Language: {} v{}", grammar.name(), grammar.version());
        println!("Creating runtime...");
    }

    // Create runtime
    let mut runtime = ZyntaxRuntime::new().context("Failed to create Zyntax runtime")?;

    // Load required ZRTL plugins
    let plugins = [
        "zrtl_image",
        "zrtl_simd",
        "zrtl_paint",
        "zrtl_fs",
        "zrtl_io",
    ];

    for plugin_name in &plugins {
        let plugin_path = plugins_dir.join(format!("{}.zrtl", plugin_name));
        if plugin_path.exists() {
            if verbose {
                println!("Loading plugin: {}", plugin_name);
            }
            runtime
                .load_plugin(&plugin_path)
                .with_context(|| format!("Failed to load plugin: {}", plugin_name))?;
        } else if verbose {
            println!("Plugin not found (skipping): {}", plugin_path.display());
        }
    }

    // Read source file
    let source = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read file: {}", file.display()))?;

    if verbose {
        println!("Parsing pipeline with Grammar2...");
    }

    // Parse directly to TypedProgram using Grammar2
    let filename = file.to_string_lossy();
    let typed_program = grammar
        .parse_with_signatures(&source, &filename, &runtime.plugin_signatures())
        .context("Failed to parse ImagePipe program")?;

    if verbose {
        println!(
            "Parsed program with {} declarations",
            typed_program.declarations.len()
        );
        println!("Compiling pipeline...");
    }

    // Lower to HIR and compile
    let function_names = runtime
        .compile_typed_program(typed_program)
        .context("Failed to compile ImagePipe program")?;

    if verbose {
        println!("Compiled functions: {:?}", function_names);
        println!("Running pipeline...\n");
    }

    // Execute the main entry point function if it exists
    // Check for the entry point declared in the grammar first
    let entry_point = grammar.entry_point().unwrap_or("main");

    if function_names.contains(&entry_point.to_string()) {
        runtime
            .call::<()>(entry_point, &[])
            .context("Pipeline execution failed")?;
    } else if function_names.contains(&"main".to_string()) {
        runtime
            .call::<()>("main", &[])
            .context("Pipeline execution failed")?;
    } else if !function_names.is_empty() {
        // Try to call the first function as the entry point
        let entry = &function_names[0];
        if verbose {
            println!(
                "No 'main' or '{}' function, calling '{}'",
                entry_point, entry
            );
        }
        runtime
            .call::<()>(entry, &[])
            .context("Pipeline execution failed")?;
    } else {
        println!("No functions found in the pipeline script");
    }

    if verbose {
        println!("\nPipeline completed successfully!");
    }

    Ok(())
}

/// Parse an ImagePipe program and display the AST
fn parse_and_display(file: &PathBuf, format: &str) -> Result<()> {
    use zyntax_embed::Grammar2;

    // Load the grammar using Grammar2
    let grammar_source = include_str!("../imagepipe.zyn");
    let grammar =
        Grammar2::from_source(grammar_source).context("Failed to compile ImagePipe grammar")?;

    // Read source file
    let source = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read file: {}", file.display()))?;

    // Parse to TypedProgram
    let filename = file.to_string_lossy();
    let program = grammar
        .parse_with_filename(&source, &filename)
        .context("Failed to parse ImagePipe program")?;

    match format {
        "json" => {
            // Serialize to JSON
            let json = serde_json::to_string_pretty(&program)
                .context("Failed to serialize AST to JSON")?;
            println!("{}", json);
        }
        "tree" | _ => {
            // Pretty print a simplified tree view
            println!("Parsed AST for: {}\n", file.display());
            print_typed_program_tree(&program);
        }
    }

    Ok(())
}

/// Print a simplified tree view of the TypedProgram
fn print_typed_program_tree(program: &zyntax_typed_ast::TypedProgram) {
    use zyntax_typed_ast::{TypedDeclaration, TypedExpression, TypedStatement};

    println!("TypedProgram:");
    println!("  declarations: {} total", program.declarations.len());

    for (i, decl) in program.declarations.iter().enumerate() {
        println!("  [{}]:", i);
        match &decl.node {
            TypedDeclaration::Function(func) => {
                let name = func
                    .name
                    .resolve_global()
                    .unwrap_or_else(|| "<unknown>".to_string());
                println!("    Function: {}", name);
                println!("      return_type: {:?}", func.return_type);

                if let Some(ref body) = func.body {
                    println!("      body: {} statements", body.statements.len());

                    for (j, stmt) in body.statements.iter().enumerate() {
                        print!("        [{j}] ");
                        match &stmt.node {
                            TypedStatement::Let(let_stmt) => {
                                let var_name = let_stmt
                                    .name
                                    .resolve_global()
                                    .unwrap_or_else(|| "<unknown>".to_string());
                                println!("Let {} : {:?}", var_name, let_stmt.ty);
                            }
                            TypedStatement::Expression(expr) => match &expr.node {
                                TypedExpression::Call(call) => {
                                    let callee_name = match &call.callee.node {
                                        TypedExpression::Variable(name) => name
                                            .resolve_global()
                                            .unwrap_or_else(|| "<expr>".to_string()),
                                        _ => "<expr>".to_string(),
                                    };
                                    println!(
                                        "Call {}(...) [{} args]",
                                        callee_name,
                                        call.positional_args.len()
                                    );
                                }
                                TypedExpression::Binary(bin) => {
                                    println!("Binary {:?}", bin.op);
                                }
                                TypedExpression::Literal(lit) => {
                                    println!("Literal {:?}", lit);
                                }
                                other => {
                                    println!("{:?}", std::mem::discriminant(other));
                                }
                            },
                            TypedStatement::Return(ret) => {
                                if let Some(ref expr) = ret {
                                    println!("Return {:?}", std::mem::discriminant(&expr.node));
                                } else {
                                    println!("Return (void)");
                                }
                            }
                            other => {
                                println!("{:?}", std::mem::discriminant(other));
                            }
                        }
                    }
                }
            }
            TypedDeclaration::Variable(var_decl) => {
                let name = var_decl
                    .name
                    .resolve_global()
                    .unwrap_or_else(|| "<unknown>".to_string());
                println!("    Variable: {}", name);
            }
            other => {
                println!("    {:?}", std::mem::discriminant(other));
            }
        }
    }
}

/// Run an interactive REPL
fn run_repl(plugins_dir: &PathBuf) -> Result<()> {
    use std::io::{self, BufRead, Write};
    use zyntax_embed::{Grammar2, ZyntaxRuntime};

    println!("ImagePipe REPL v2.0 (Grammar2 backend)");
    println!("Type 'help' for available commands, 'exit' to quit\n");

    // Load grammar using Grammar2
    let grammar_source = include_str!("../imagepipe.zyn");
    let grammar =
        Grammar2::from_source(grammar_source).context("Failed to compile ImagePipe grammar")?;

    // Create runtime with plugins
    let mut runtime = ZyntaxRuntime::new()?;

    let plugins = [
        "zrtl_image",
        "zrtl_simd",
        "zrtl_paint",
        "zrtl_fs",
        "zrtl_io",
    ];
    for plugin_name in &plugins {
        let plugin_path = plugins_dir.join(format!("{}.zrtl", plugin_name));
        if plugin_path.exists() {
            runtime.load_plugin(&plugin_path).ok();
        }
    }

    // Track loaded images
    let mut current_image: Option<String> = None;

    let stdin = io::stdin();
    loop {
        // Show prompt with current image
        let prompt = match &current_image {
            Some(img) => format!("[{}] imgpipe> ", img),
            None => "imgpipe> ".to_string(),
        };
        print!("{}", prompt);
        io::stdout().flush()?;

        // Read input
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }
        let line = line.trim();

        match line {
            "" => continue,
            "exit" | "quit" => {
                println!("Goodbye!");
                break;
            }
            "help" => {
                print_repl_help();
            }
            "images" => {
                println!("Current image: {:?}", current_image);
            }
            _ if line.starts_with("load ") => {
                // Parse: load "path" as name
                let rest = line.strip_prefix("load ").unwrap();
                let program = format!("{}\n", line);
                match compile_and_run_v2(&mut runtime, &grammar, &program) {
                    Ok(()) => {
                        // Extract variable name
                        if let Some(pos) = rest.find(" as ") {
                            let name = rest[pos + 4..].trim();
                            current_image = Some(name.to_string());
                            println!("Loaded image as '{}'", name);
                        }
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            _ if line.starts_with("save ") => {
                let program = format!("{}\n", line);
                match compile_and_run_v2(&mut runtime, &grammar, &program) {
                    Ok(()) => println!("Image saved"),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            _ if line.starts_with("pipeline ") => {
                // Multi-line pipeline input
                println!("Enter pipeline operations (end with '}}'):");
                let mut program = format!("{}\n", line);
                loop {
                    print!("... ");
                    io::stdout().flush()?;
                    let mut next_line = String::new();
                    stdin.lock().read_line(&mut next_line)?;
                    program.push_str(&next_line);
                    if next_line.contains('}') {
                        break;
                    }
                }
                match compile_and_run_v2(&mut runtime, &grammar, &program) {
                    Ok(()) => println!("Pipeline applied"),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            _ => {
                // Try to execute as a single operation on current image
                if let Some(img_name) = &current_image {
                    let program = format!("pipeline {} {{\n    {}\n}}\n", img_name, line);
                    match compile_and_run_v2(&mut runtime, &grammar, &program) {
                        Ok(()) => println!("Operation applied"),
                        Err(e) => eprintln!("Error: {}", e),
                    }
                } else {
                    eprintln!("No image loaded. Use 'load \"path\" as name' first.");
                }
            }
        }
    }

    Ok(())
}

fn compile_and_run_v2(
    runtime: &mut zyntax_embed::ZyntaxRuntime,
    grammar: &zyntax_embed::Grammar2,
    source: &str,
) -> Result<()> {
    // Parse to TypedProgram using Grammar2
    let typed_program = grammar
        .parse_with_signatures(source, "repl.imgpipe", &runtime.plugin_signatures())
        .context("Parse failed")?;

    // Compile and run
    let functions = runtime
        .compile_typed_program(typed_program)
        .context("Compilation failed")?;

    // Try to find an entry point to run
    if functions.contains(&"main".to_string()) {
        runtime
            .call::<()>("main", &[])
            .context("Execution failed")?;
    } else if !functions.is_empty() {
        // For REPL, the grammar should generate inline execution
        // The module loading itself executes the pipeline operations
    }

    Ok(())
}

fn print_repl_help() {
    println!("ImagePipe REPL Commands:");
    println!();
    println!("  load \"path\" as name  - Load an image file");
    println!("  save name as \"path\"  - Save an image to file");
    println!("  pipeline name {{ ... }} - Apply operations to an image");
    println!("  images               - Show loaded images");
    println!("  help                 - Show this help");
    println!("  exit                 - Exit the REPL");
    println!();
    println!("Pipeline Operations:");
    println!();
    println!("  resize WxH           - Resize to exact dimensions");
    println!("  resize fit WxH       - Resize to fit within bounds");
    println!("  crop X,Y to X,Y      - Crop region");
    println!("  rotate 90|180|270    - Rotate image");
    println!("  flip horizontal      - Flip horizontally");
    println!("  flip vertical        - Flip vertically");
    println!("  blur SIGMA           - Gaussian blur");
    println!("  brightness +/-N      - Adjust brightness");
    println!("  contrast N           - Adjust contrast");
    println!("  grayscale            - Convert to grayscale");
    println!("  invert               - Invert colors");
    println!();
    println!("Example:");
    println!("  load \"photo.jpg\" as img");
    println!("  pipeline img {{");
    println!("      resize fit 800x600");
    println!("      brightness +10");
    println!("  }}");
    println!("  save img as \"output.png\"");
}
