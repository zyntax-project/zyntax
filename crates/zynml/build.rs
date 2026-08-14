use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use zyntax_embed::{CompiledImport, LanguageGrammar};

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out = PathBuf::from(env::var("OUT_DIR")?);
    let grammar_path = manifest.join("ml.zyn");

    println!("cargo:rerun-if-changed={}", grammar_path.display());
    for relative in [
        "stdlib/prelude.zynml",
        "stdlib/tensor.zynml",
        "stdlib/simd.zynml",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            manifest.join(relative).display()
        );
    }

    let grammar_source = fs::read_to_string(&grammar_path)?;
    let grammar = LanguageGrammar::compile_zyn(&grammar_source)?;
    fs::write(out.join("zynml.grammar"), grammar.to_compiled_bytes()?)?;

    let parser = grammar
        .direct_parser()
        .ok_or("compiled ZynML grammar did not contain GrammarIR")?;
    compile_import(
        &parser,
        &manifest.join("stdlib/prelude.zynml"),
        "prelude",
        &out.join("prelude.zast"),
    )?;
    compile_import(
        &parser,
        &manifest.join("stdlib/tensor.zynml"),
        "tensor",
        &out.join("tensor.zast"),
    )?;
    compile_import(
        &parser,
        &manifest.join("stdlib/simd.zynml"),
        "simd",
        &out.join("simd.zast"),
    )?;
    Ok(())
}

fn compile_import(
    parser: &zyntax_embed::Grammar2,
    source_path: &Path,
    module_name: &str,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let source = fs::read_to_string(source_path)?;
    let program = parser.parse_with_filename(&source, module_name)?;
    let artifact = CompiledImport::new("zynml", module_name, program);
    fs::write(output_path, artifact.encode()?)?;
    Ok(())
}
