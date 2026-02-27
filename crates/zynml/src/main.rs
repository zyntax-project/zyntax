//! ZynML CLI - Command-line interface for the ZynML DSL
//!
//! ## Usage
//!
//! ```bash
//! # Run a ZynML program
//! zynml run script.ml
//!
//! # Parse and show the AST
//! zynml parse script.ml
//!
//! # Start interactive REPL
//! zynml repl
//!
//! # Show version and info
//! zynml info
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use zynml::{ZynML, ZynMLConfig, OPTIONAL_PLUGINS, REQUIRED_PLUGINS, ZYNML_GRAMMAR};

/// ZynML - Machine Learning DSL for Zyntax
#[derive(Parser)]
#[command(name = "zynml")]
#[command(about = "ZynML - A unified DSL for Machine Learning")]
#[command(version)]
#[command(author = "Zyntax Project")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a ZynML program
    Run {
        /// Path to the .ml or .zynml file
        file: PathBuf,

        /// Directory containing ZRTL plugins
        #[arg(long, default_value = "plugins/target/zrtl")]
        plugins: PathBuf,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Load optional plugins (image, json, http)
        #[arg(long)]
        all_plugins: bool,
    },

    /// Parse a ZynML program and show the AST
    Parse {
        /// Path to the .ml or .zynml file
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

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show information about ZynML
    Info,

    /// Validate a ZynML program without running it
    Check {
        /// Path to the .ml or .zynml file
        file: PathBuf,
    },
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            file,
            plugins,
            verbose,
            all_plugins,
        } => run_program(&file, &plugins, verbose, all_plugins),
        Commands::Parse { file, format } => parse_and_display(&file, &format),
        Commands::Repl { plugins, verbose } => run_repl(&plugins, verbose),
        Commands::Info => show_info(),
        Commands::Check { file } => check_program(&file),
    }
}

/// Run a ZynML program
fn run_program(
    file: &PathBuf,
    plugins_dir: &PathBuf,
    verbose: bool,
    all_plugins: bool,
) -> Result<()> {
    if verbose {
        println!("ZynML v{}", env!("CARGO_PKG_VERSION"));
        println!("Loading program: {}", file.display());
    }

    let config = ZynMLConfig {
        plugins_dir: plugins_dir.to_string_lossy().to_string(),
        load_optional: all_plugins,
        verbose,
        runtime_profile: zynml::ZynMLRuntimeProfile::Classic,
    };

    let mut zynml = ZynML::with_config(config)?;

    if verbose {
        println!("Running...\n");
    }

    zynml.run_file(file)?;

    if verbose {
        println!("\nProgram completed successfully.");
    }

    Ok(())
}

/// Parse a ZynML program and display the AST
fn parse_and_display(file: &PathBuf, format: &str) -> Result<()> {
    use zynml::Grammar2;

    // Compile grammar using Grammar2 (direct TypedAST generation)
    let grammar =
        Grammar2::from_source(ZYNML_GRAMMAR).context("Failed to compile ZynML grammar")?;

    // Read source
    let source = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read file: {}", file.display()))?;

    // Parse to TypedProgram using Grammar2
    let program = grammar
        .parse_with_filename(&source, &file.to_string_lossy())
        .context("Failed to parse ZynML program")?;

    match format {
        "json" => {
            // Serialize TypedProgram to JSON
            let json = serde_json::to_string_pretty(&program)
                .context("Failed to serialize AST to JSON")?;
            println!("{}", json);
        }
        "tree" | _ => {
            // Print tree view
            println!("AST for: {}\n", file.display());
            println!("Parsed {} declarations", program.declarations.len());
            for decl in &program.declarations {
                println!("  - {:?}", std::mem::discriminant(&decl.node));
            }
        }
    }

    Ok(())
}

/// Print a simplified tree view of the AST
fn print_ast_tree(json: &str) -> Result<()> {
    let parsed: serde_json::Value = serde_json::from_str(json)?;

    fn print_value(value: &serde_json::Value, indent: usize) {
        let prefix = "  ".repeat(indent);
        match value {
            serde_json::Value::Object(map) => {
                // Print important keys first
                if let Some(serde_json::Value::String(kind)) = map.get("kind") {
                    println!("{}kind: {}", prefix, kind);
                }
                if let Some(serde_json::Value::String(name)) = map.get("name") {
                    println!("{}name: {}", prefix, name);
                }
                if let Some(serde_json::Value::String(op)) = map.get("op") {
                    println!("{}op: {}", prefix, op);
                }

                // Print nested structures
                for (key, val) in map {
                    if key == "kind" || key == "name" || key == "op" {
                        continue;
                    }
                    if matches!(
                        val,
                        serde_json::Value::Object(_) | serde_json::Value::Array(_)
                    ) {
                        if !matches!(val, serde_json::Value::Array(arr) if arr.is_empty()) {
                            println!("{}{}:", prefix, key);
                            print_value(val, indent + 1);
                        }
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for (i, item) in arr.iter().enumerate() {
                    println!("{}[{}]:", prefix, i);
                    print_value(item, indent + 1);
                }
            }
            _ => {}
        }
    }

    print_value(&parsed, 0);
    Ok(())
}

/// Check a ZynML program for errors without running it
fn check_program(file: &PathBuf) -> Result<()> {
    use zynml::Grammar2;

    // Compile grammar using Grammar2 (direct TypedAST generation)
    let grammar =
        Grammar2::from_source(ZYNML_GRAMMAR).context("Failed to compile ZynML grammar")?;

    // Read source
    let source = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read file: {}", file.display()))?;

    // Try to parse with Grammar2
    match grammar.parse_with_filename(&source, &file.to_string_lossy()) {
        Ok(program) => {
            println!(
                "OK: {} is valid ZynML ({} declarations)",
                file.display(),
                program.declarations.len()
            );
            Ok(())
        }
        Err(e) => {
            eprintln!("ERROR in {}:", file.display());
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

/// Run an interactive REPL
fn run_repl(plugins_dir: &PathBuf, verbose: bool) -> Result<()> {
    println!("ZynML REPL v{}", env!("CARGO_PKG_VERSION"));
    println!("Type 'help' for commands, 'exit' to quit\n");

    let config = ZynMLConfig {
        plugins_dir: plugins_dir.to_string_lossy().to_string(),
        verbose,
        load_optional: true,
        runtime_profile: zynml::ZynMLRuntimeProfile::Classic,
    };

    let mut zynml = ZynML::with_config(config)?;
    let mut history: Vec<String> = Vec::new();

    let stdin = io::stdin();

    loop {
        print!("zynml> ");
        io::stdout().flush()?;

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
            "history" => {
                for (i, cmd) in history.iter().enumerate() {
                    println!("{}: {}", i + 1, cmd);
                }
            }
            "clear" => {
                // Clear screen (ANSI escape)
                print!("\x1B[2J\x1B[1;1H");
                io::stdout().flush()?;
            }
            _ if line.starts_with("load ") => {
                let path = line.strip_prefix("load ").unwrap().trim();
                match zynml.load_file(std::path::Path::new(path)) {
                    Ok(funcs) => println!("Loaded functions: {:?}", funcs),
                    Err(e) => eprintln!("Error: {}", e),
                }
                history.push(line.to_string());
            }
            _ if line.starts_with("parse ") => {
                let expr = line.strip_prefix("parse ").unwrap().trim();
                match zynml.parse_to_json(expr) {
                    Ok(json) => {
                        let parsed: serde_json::Value = serde_json::from_str(&json)?;
                        println!("{}", serde_json::to_string_pretty(&parsed)?);
                    }
                    Err(e) => eprintln!("Parse error: {}", e),
                }
            }
            _ if line.starts_with("run ") => {
                let code = line.strip_prefix("run ").unwrap().trim();
                if let Err(e) = zynml.run(code) {
                    eprintln!("Error: {}", e);
                }
                history.push(line.to_string());
            }
            _ => {
                // Try to evaluate as an expression or statement
                let code =
                    if line.contains('=') || line.starts_with("let ") || line.starts_with("fn ") {
                        line.to_string()
                    } else {
                        // Wrap in a print statement for expressions
                        format!("println({})", line)
                    };

                match zynml.run(&code) {
                    Ok(_) => {}
                    Err(e) => eprintln!("Error: {}", e),
                }
                history.push(line.to_string());
            }
        }
    }

    Ok(())
}

/// Print REPL help
fn print_repl_help() {
    println!("ZynML REPL Commands:");
    println!("  help        - Show this help");
    println!("  exit/quit   - Exit the REPL");
    println!("  clear       - Clear the screen");
    println!("  history     - Show command history");
    println!("  load <file> - Load a ZynML file");
    println!("  parse <code>- Parse and show AST");
    println!("  run <code>  - Run ZynML code");
    println!();
    println!("Enter expressions directly to evaluate them.");
    println!();
    println!("Examples:");
    println!("  let x = 42");
    println!("  let t = tensor([1.0, 2.0, 3.0])");
    println!("  x |> f() |> g(10)");
}

/// Show information about ZynML
fn show_info() -> Result<()> {
    println!("ZynML - Machine Learning DSL for Zyntax");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Features:");
    println!("  - Tensor operations with SIMD acceleration");
    println!("  - Audio processing (STFT, mel spectrograms)");
    println!("  - Text tokenization (BPE)");
    println!("  - Vector search (Flat, HNSW indexes)");
    println!("  - Model loading (SafeTensors format)");
    println!();
    println!("Required Plugins:");
    for plugin in REQUIRED_PLUGINS {
        println!("  - {}", plugin);
    }
    println!();
    println!("Optional Plugins:");
    for plugin in OPTIONAL_PLUGINS {
        println!("  - {}", plugin);
    }
    println!();
    println!("File Extensions: .ml, .zynml");
    println!();
    println!("Usage:");
    println!("  zynml run <file.ml>      Run a ZynML program");
    println!("  zynml parse <file.ml>    Show the AST");
    println!("  zynml check <file.ml>    Validate syntax");
    println!("  zynml repl               Interactive mode");
    Ok(())
}
