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

// What the runtime alone uses of the library (the C API's bridge) is
// dead here.
#[allow(dead_code)]
#[path = "src/library/mod.rs"]
mod library;
#[path = "src/policy.rs"]
mod policy;

/// Link flags for the binary: it exports the Lua C API, which C
/// libraries it opens bind to, and on macOS nothing else but `main`,
/// with no dylib loaded that nothing binds to. The JIT resolves the
/// runtime's symbols from the tables it is handed, never through the
/// executable's exports, and every symbol dyld need not coalesce or
/// bind is time before `main`. Windows exports nothing yet: a C module
/// there cannot bind to the executable.
fn emit_link_flags(out: &std::path::Path) -> Result<(), BuildError> {
    let os = env::var("CARGO_CFG_TARGET_OS")?;
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let list = env::var("DEP_ZL_CAPI_EXPORTS")?;
    println!("cargo:rerun-if-changed={list}");
    let names: Vec<String> = fs::read_to_string(&list)?
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    if os == "windows" {
        return Ok(());
    }
    for arg in zrtl_native::build::export_link_args(&os, &target_env, &names, out)? {
        println!("cargo:rustc-link-arg-bins={arg}");
    }
    if os == "macos" {
        println!("cargo:rustc-link-arg-bins=-Wl,-dead_strip_dylibs");
    }
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

    // Every hook the shared library declares without a symbol is one
    // this frontend must define, and the runtime only finds out when
    // code naming it is compiled: checked here instead. (The other
    // symbols without one are the platform's, `sqrt` and `free`.)
    let defined: std::collections::HashSet<String> = lib_declarations
        .iter()
        .filter_map(|d| match &d.node {
            zyntax_typed_ast::typed_ast::TypedDeclaration::Function(f) if !f.is_external => {
                f.name.resolve_global()
            }
            _ => None,
        })
        .collect();
    let undefined: Vec<String> = lib_declarations
        .iter()
        .filter_map(|d| match &d.node {
            zyntax_typed_ast::typed_ast::TypedDeclaration::Function(f)
                if f.is_external && f.link_name.is_none() =>
            {
                f.name.resolve_global()
            }
            _ => None,
        })
        .filter(|name| name.starts_with("zb_hook_") && !defined.contains(name))
        .collect();
    if !undefined.is_empty() {
        return Err(format!(
            "the shared library declares hooks the Lua library does not define: {}",
            undefined.join(", ")
        )
        .into());
    }

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
