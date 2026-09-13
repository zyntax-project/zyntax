//! Lower the built-in library once, for the snapshot the crate carries.
//!
//! A Python program links against the library rather than compiling it:
//! the snapshot holds the library's declarations for typing and its HIR
//! for running, and the runtime lowers only the program.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use zyntax_embed::{lower_for_snapshot_releasing, SnapshotBuilder};
use zyntax_typed_ast::{InternedString, Span, TypedProgram};

#[path = "src/policy.rs"]
mod policy;

fn main() -> Result<(), Box<dyn Error>> {
    let out = PathBuf::from(env::var("OUT_DIR")?);
    println!("cargo:rerun-if-changed=src/policy.rs");
    println!("cargo:rerun-if-changed=../zyntax_builtins/src");

    // Lowered for the target, not for the machine building it.
    let width: usize = env::var("CARGO_CFG_TARGET_POINTER_WIDTH")?.parse()?;
    zyntax_compiler::set_target_pointer_size(width / 8);

    let library = zyntax_builtins::library(&policy::POLICY);
    let program = TypedProgram {
        declarations: library.declarations,
        language: Some(InternedString::new_global("python")),
        span: Span::new(0, 0),
        source_files: Vec::new(),
        type_registry: library.type_registry,
    };

    // Released as the runtime will release: no Python program frees
    // anything by hand.
    let hir = lower_for_snapshot_releasing(
        policy::LIBRARY_MODULE,
        program.clone(),
        indexmap::IndexMap::new(),
        Vec::new(),
        true,
    )?;
    SnapshotBuilder::new("python")
        .module_lowered(policy::LIBRARY_MODULE, program, &hir)?
        .build_in(&out)?;

    // Which library functions can raise, for the frontend's guards.
    let mut fallible = String::from("pub const FALLIBLE: &[&str] = &[\n");
    for name in &library.fallible {
        fallible.push_str(&format!("    {name:?},\n"));
    }
    fallible.push_str("];\n");
    fs::write(out.join("fallible.rs"), fallible)?;
    Ok(())
}
