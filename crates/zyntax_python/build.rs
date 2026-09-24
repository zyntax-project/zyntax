//! Lower the built-in library once, for the snapshot the crate carries.
//!
//! A Python program links against the library rather than compiling it:
//! the snapshot holds the library's declarations for typing and its HIR
//! for running, and the runtime lowers only the program.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use zyntax_embed::{SnapshotBuilder, lower_for_snapshot_releasing};
use zyntax_typed_ast::{InternedString, Span, TypedProgram};

#[path = "src/policy.rs"]
mod policy;

/// Link flags for the binary: on macOS the executable exports no
/// symbol but `main` and loads no dylib nothing binds to. The JIT
/// resolves the runtime's symbols from the tables it is handed, never
/// through the executable's exports, and every symbol dyld need not
/// coalesce or bind is time before `main`. Linux exports nothing from
/// an executable by default and drops unused dylibs on its own.
fn emit_link_flags(out: &std::path::Path) -> Result<(), BuildError> {
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

/// The stack the build runs on. Lowering the library recurses as deep
/// as it nests, past the 1 MB a Windows main thread has.
const STACK_BYTES: usize = 256 << 20;

type BuildError = Box<dyn Error + Send + Sync>;

fn main() -> Result<(), BuildError> {
    std::thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(build)?
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

fn build() -> Result<(), BuildError> {
    let out = PathBuf::from(env::var("OUT_DIR")?);
    emit_link_flags(&out)?;
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
