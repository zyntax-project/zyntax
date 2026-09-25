//! The C API under conditions the conformance run does not set: a
//! collector that runs at every allocation, each tier running the code
//! that calls into C, and the executable's exports.
//!
//! Each case is a program of `conformance/capi/`, run as the
//! conformance harness runs it and held to the output pinned there.

#[path = "common/c_modules.rs"]
mod c_modules;

use std::path::{Path, PathBuf};
use std::process::Command;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("conformance")
        .join("capi")
}

/// Build the modules, or say why the test does not run.
fn modules() -> bool {
    let sources = c_modules::sources_in(&dir());
    let pairs: Vec<(&str, &str)> = sources
        .iter()
        .map(|(l, s)| (l.as_str(), s.as_str()))
        .collect();
    match c_modules::build(&dir(), &pairs) {
        Ok(()) => true,
        Err(reason) => {
            eprintln!("skipped: {reason}");
            false
        }
    }
}

/// Run a case with `env` set; its stdout, and whether it exited 0.
fn run(case: &str, env: &[(&str, &str)]) -> (String, bool) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_zylua"));
    cmd.current_dir(dir()).arg("run").arg(case);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("zylua runs");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.success(),
    )
}

fn pinned(case: &str) -> String {
    let path = dir().join(case).with_extension("expected");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn check(case: &str, env: &[(&str, &str)]) {
    let (out, ok) = run(case, env);
    assert!(ok, "{case} under {env:?} failed:\n{out}");
    assert_eq!(out, pinned(case), "{case} under {env:?}");
}

/// Values C code holds only on its API stack, in a closure's upvalues
/// or in the registry survive a collection at every allocation.
#[test]
fn values_c_holds_survive_constant_collection() {
    if !modules() {
        return;
    }
    for case in ["gc.lua", "basics.lua", "errors.lua"] {
        check(case, &[("ZYNTAX_GC_FLOOR_KB", "1")]);
    }
}

/// A C function called from interpreted code, from code compiled at
/// the baseline, and from LLVM code where this build has it.
#[test]
fn every_tier_calls_c_functions() {
    if !modules() {
        return;
    }
    check("tiers.lua", &[("ZYNTAX_BASELINE_THRESHOLD", "1000000000")]);
    check("tiers.lua", &[("ZYNTAX_BASELINE_THRESHOLD", "1")]);
    check("tiers.lua", &[]);
    let backend = Command::new(env!("CARGO_BIN_EXE_zylua"))
        .arg("backend")
        .env("ZYLUA_LLVM", "1")
        .output()
        .expect("zylua runs");
    if String::from_utf8_lossy(&backend.stdout).trim() == "llvm" {
        check("tiers.lua", &[("ZYLUA_LLVM", "1")]);
        check("errors.lua", &[("ZYLUA_LLVM", "1")]);
    }
}

/// The executable exports every name of the API, which is what a C
/// module binds to when it is opened.
#[test]
fn the_executable_exports_the_api() {
    if cfg!(windows) {
        return;
    }
    let exe = env!("CARGO_BIN_EXE_zylua");
    let listing = if cfg!(target_os = "macos") {
        Command::new("nm").args(["-gU", exe]).output()
    } else {
        Command::new("nm")
            .args(["-D", "--defined-only", exe])
            .output()
    };
    let Ok(listing) = listing else {
        eprintln!("skipped: no nm");
        return;
    };
    let text = String::from_utf8_lossy(&listing.stdout);
    let exported: std::collections::HashSet<&str> = text
        .lines()
        .filter_map(|l| l.split_whitespace().last())
        .map(|s| {
            s.strip_prefix('_')
                .filter(|_| cfg!(target_os = "macos"))
                .unwrap_or(s)
        })
        .collect();
    let missing: Vec<&&str> = zyntax_lua_capi::EXPORT_NAMES
        .iter()
        .filter(|n| !exported.contains(**n))
        .collect();
    assert!(missing.is_empty(), "not exported: {missing:?}");
}
