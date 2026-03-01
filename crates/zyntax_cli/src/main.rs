#![allow(unused, dead_code, deprecated)]

//! Zyntax CLI - Command-line interface for the Zyntax compiler
//!
//! Supports multiple input formats:
//! - HIR bytecode (.zbc files) - Direct HIR deserialization
//! - TypedAST JSON (.json files) - Language-agnostic IR from frontends
//! - ZynPEG grammar (.zyn files) - Grammar-based parsing for custom languages
//!
//! Multiple backends:
//! - Cranelift JIT - Fast compilation for development
//! - LLVM AOT - Optimized compilation for production
//!
//! Import Resolution:
//! - Multiple module architectures (Haxe, Java, Rust, Python, TypeScript, Go, Deno)
//! - Incremental compilation with ZBC caching

mod backends;
mod cli;
mod commands;
mod formats;

use clap::Parser;
use colored::Colorize;
use std::process;

use cli::{Cli, Commands};

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    let result = match &cli.command {
        Commands::Compile { .. } => {
            if let Some(args) = cli.command.compile_args() {
                commands::compile(
                    args.input,
                    args.source,
                    args.grammar,
                    args.output,
                    args.backend,
                    args.opt_level,
                    args.format,
                    args.jit,
                    args.entry_point,
                    args.resolver,
                    args.source_roots,
                    args.lib_paths,
                    args.import_map,
                    args.cache_dir,
                    args.no_cache,
                    args.grammar1,
                    args.packs,
                    args.static_libs,
                    cli.verbose,
                )
            } else {
                unreachable!("Compile command should have compile args")
            }
        }

        Commands::Repl { .. } => {
            if let Some(args) = cli.command.repl_args() {
                commands::repl(
                    args.grammar,
                    args.backend,
                    args.opt_level,
                    args.resolver,
                    args.source_roots,
                    args.lib_paths,
                    cli.verbose,
                )
            } else {
                unreachable!("Repl command should have repl args")
            }
        }

        Commands::Cache { .. } => {
            if let Some(action) = cli.command.cache_args() {
                commands::cache(action, cli.verbose)
            } else {
                unreachable!("Cache command should have cache args")
            }
        }

        Commands::Pack { .. } => {
            if let Some(action) = cli.command.pack_args() {
                commands::pack(action, cli.verbose)
            } else {
                unreachable!("Pack command should have pack args")
            }
        }

        Commands::Version => commands::version(),
    };

    if let Err(e) = result {
        eprintln!("{} {}", "error:".red().bold(), e);
        process::exit(1);
    }
}
