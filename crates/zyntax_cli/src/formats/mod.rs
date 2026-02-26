//! Input format handling (HIR bytecode, TypedAST JSON, and ZynPEG grammar-based)

pub mod hir_bytecode;
pub mod typed_ast_json;
pub mod zyn_grammar;

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// HIR bytecode (.zbc files)
    HirBytecode,
    /// TypedAST JSON (.json files)
    TypedAst,
    /// ZynPEG grammar-based parsing (requires --grammar and --source)
    ZynGrammar,
}

/// Detect input format based on format string and file extensions
pub fn detect_format(
    format_arg: &str,
    inputs: &[PathBuf],
    grammar: Option<&PathBuf>,
    source: Option<&PathBuf>,
) -> Result<InputFormat, Box<dyn std::error::Error>> {
    match format_arg {
        "auto" => {
            // If grammar and source are provided, use ZynGrammar
            if grammar.is_some() && source.is_some() {
                Ok(InputFormat::ZynGrammar)
            } else if grammar.is_some() || source.is_some() {
                Err(
                    "Both --grammar and --source must be provided for grammar-based compilation"
                        .into(),
                )
            } else {
                auto_detect_format(inputs)
            }
        }
        "hir-bytecode" | "bytecode" | "zbc" => Ok(InputFormat::HirBytecode),
        "typed-ast" | "json" => Ok(InputFormat::TypedAst),
        "zyn" | "zyn-grammar" => {
            if grammar.is_none() || source.is_none() {
                Err("Format 'zyn' requires both --grammar and --source options".into())
            } else {
                Ok(InputFormat::ZynGrammar)
            }
        }
        _ => Err(format!("Unknown format: {}", format_arg).into()),
    }
}

fn auto_detect_format(inputs: &[PathBuf]) -> Result<InputFormat, Box<dyn std::error::Error>> {
    if let Some(first_input) = inputs.first() {
        let ext = if first_input.is_file() {
            first_input.extension().and_then(|s| s.to_str())
        } else if first_input.is_dir() {
            scan_directory_for_format(first_input)?
        } else {
            None
        };

        match ext {
            Some("zbc") => Ok(InputFormat::HirBytecode),
            Some("json") => Ok(InputFormat::TypedAst),
            _ => Err("Could not auto-detect format. Please specify --format.".into()),
        }
    } else {
        Err("No input files provided".into())
    }
}

fn scan_directory_for_format(
    dir: &PathBuf,
) -> Result<Option<&'static str>, Box<dyn std::error::Error>> {
    let mut has_zbc = false;
    let mut has_json = false;

    for entry in walkdir::WalkDir::new(dir)
        .max_depth(3)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) {
            match ext {
                "zbc" => has_zbc = true,
                "json" => has_json = true,
                _ => {}
            }
        }
    }

    if has_zbc && has_json {
        return Err(
            "Directory contains both .zbc and .json files. Please specify format explicitly."
                .into(),
        );
    } else if has_zbc {
        Ok(Some("zbc"))
    } else if has_json {
        Ok(Some("json"))
    } else {
        Ok(None)
    }
}
