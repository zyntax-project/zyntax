use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use zyntax_embed::{LanguageGrammar, SnapshotBuilder};

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out = PathBuf::from(env::var("OUT_DIR")?);
    let grammar_path = manifest.join("ml.zyn");

    println!("cargo:rerun-if-changed={}", grammar_path.display());
    let modules = ["prelude", "tensor", "simd"];
    for module in modules {
        println!(
            "cargo:rerun-if-changed={}",
            manifest.join(format!("stdlib/{module}.zynml")).display()
        );
    }

    let grammar_source = fs::read_to_string(&grammar_path)?;
    let grammar = LanguageGrammar::compile_zyn(&grammar_source)?;
    let parser = grammar
        .direct_parser()
        .ok_or("compiled ZynML grammar did not contain GrammarIR")?;

    // The order these are added is the order their type ids are
    // reserved when the snapshot installs.
    let mut snapshot = SnapshotBuilder::new("zynml").grammar(grammar.to_compiled_bytes()?);
    for module in modules {
        let source = fs::read_to_string(manifest.join(format!("stdlib/{module}.zynml")))?;
        let program = parser.parse_with_filename(&source, module)?;
        snapshot = snapshot.module_with_source(module, program, source)?;
    }
    snapshot.build_in(&out)?;
    Ok(())
}
