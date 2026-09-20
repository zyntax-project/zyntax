//! Lower the built-in library once, for the snapshot the crate carries.
//!
//! A Lua program links against the library rather than compiling it:
//! the snapshot holds the library's declarations for typing and its HIR
//! for running, and the runtime lowers only the program.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use zyntax_embed::{SnapshotBuilder, lower_for_snapshot_releasing};
use zyntax_typed_ast::{InternedString, Span, TypedProgram};

#[path = "src/library/mod.rs"]
mod library;
#[path = "src/policy.rs"]
mod policy;

fn main() -> Result<(), Box<dyn Error>> {
    let out = PathBuf::from(env::var("OUT_DIR")?);
    println!("cargo:rerun-if-changed=src/policy.rs");
    println!("cargo:rerun-if-changed=src/library");
    println!("cargo:rerun-if-changed=../zyntax_builtins/src");

    // Lowered for the target, not for the machine building it.
    let width: usize = env::var("CARGO_CFG_TARGET_POINTER_WIDTH")?.parse()?;
    zyntax_compiler::set_target_pointer_size(width / 8);

    let (lib, _types) = library::library(&policy::POLICY);
    let lib_declarations = lib.declarations.clone();
    let program = TypedProgram {
        declarations: lib.declarations,
        language: Some(InternedString::new_global("lua")),
        span: Span::new(0, 0),
        source_files: Vec::new(),
        type_registry: lib.type_registry,
    };

    // Released as the runtime will release: no Lua program frees
    // anything by hand.
    let reentrant = library::reentrant_functions(&program.declarations);
    let hir = lower_for_snapshot_releasing(
        policy::LIBRARY_MODULE,
        program.clone(),
        indexmap::IndexMap::new(),
        Vec::new(),
        true,
    )?;
    SnapshotBuilder::new("lua")
        .module_lowered(policy::LIBRARY_MODULE, program, &hir)?
        .build_in(&out)?;

    // Which library functions can raise, for the frontend's checks,
    // and which may run the program's code before returning.
    let mut fallible = String::from("pub const FALLIBLE: &[&str] = &[\n");
    for name in &lib.fallible {
        fallible.push_str(&format!("    {name:?},\n"));
    }
    fallible.push_str("];\n");
    fallible.push_str("pub const REENTRANT: &[&str] = &[\n");
    for name in &reentrant {
        fallible.push_str(&format!("    {name:?},\n"));
    }
    fallible.push_str("];\n");
    // The host symbols the shared library declares, whichever
    // frontend's host provides them: the ones this host does not are
    // bound to a trap, so the code referring to them still links.
    let mut host_externs: Vec<String> = lib_declarations
        .iter()
        .filter_map(|d| match &d.node {
            zyntax_typed_ast::typed_ast::TypedDeclaration::Function(f) if f.is_external => f
                .link_name
                .as_ref()
                .and_then(|n| n.resolve_global())
                .filter(|n| n.starts_with("$Host$")),
            _ => None,
        })
        .collect();
    host_externs.sort();
    host_externs.dedup();
    fallible.push_str("pub const HOST_EXTERNS: &[&str] = &[\n");
    for name in &host_externs {
        fallible.push_str(&format!("    {name:?},\n"));
    }
    fallible.push_str("];\n");
    fs::write(out.join("fallible.rs"), fallible)?;
    Ok(())
}
