use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use zyntax_embed::{LanguageGrammar, SnapshotBuilder};

/// Link flags for the binary: on macOS the executable exports no
/// symbol but `main` and loads no dylib nothing binds to. The JIT
/// resolves the runtime's symbols from the tables it is handed, never
/// through the executable's exports, and every symbol dyld need not
/// coalesce or bind is time before `main`. Linux exports nothing from
/// an executable by default and drops unused dylibs on its own.
fn emit_link_flags(out: &std::path::Path) -> Result<(), Box<dyn Error>> {
    if env::var("CARGO_CFG_TARGET_OS")? != "macos" {
        return Ok(());
    }
    let exports = out.join("exported_symbols.txt");
    fs::write(&exports, "_main\n")?;
    println!(
        "cargo:rustc-link-arg-bins=-Wl,-exported_symbols_list,{}",
        exports.display()
    );
    println!("cargo:rustc-link-arg-bins=-Wl,-dead_strip_dylibs");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out = PathBuf::from(env::var("OUT_DIR")?);
    emit_link_flags(&out)?;
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
