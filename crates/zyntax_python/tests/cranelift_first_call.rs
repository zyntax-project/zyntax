//! Programs compiled by Cranelift from their first call.
//!
//! `ZYNTAX_BASELINE_THRESHOLD=1` compiles every function the first time
//! it is called, so a body the interpreter would otherwise run once runs
//! as the optimised baseline instead. Each case under
//! `tests/fixtures/cranelift_first_call/` prints what CPython prints,
//! pinned beside it as `.expected`. The scimark cases run beside the
//! speed kernel and its shims, as `benchmarks/speed/run.py` stages them.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cranelift_first_call")
}

/// A directory holding `case` and, when it imports one, the scimark
/// kernel with the shims it imports.
fn stage(case: &str, warm_up: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "zypy-cranelift-first-call-{}-{case}-{warm_up}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    let file = format!("{case}.py");
    fs::copy(fixtures().join(&file), dir.join(&file)).unwrap();
    let speed = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/speed");
    fs::copy(speed.join("kernels/scimark.py"), dir.join("scimark.py")).unwrap();
    for shim in ["util.py", "optparse.py"] {
        fs::copy(speed.join("shims").join(shim), dir.join(shim)).unwrap();
    }
    dir
}

fn run(case: &str, warm_up: bool) {
    let expected = fs::read_to_string(fixtures().join(format!("{case}.expected"))).unwrap();
    let dir = stage(case, warm_up);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_zypy"));
    cmd.arg("run")
        .arg(dir.join(format!("{case}.py")))
        .current_dir(&dir)
        .env("ZYNTAX_BASELINE_THRESHOLD", "1");
    if warm_up {
        cmd.env_remove("ZYNTAX_DISABLE_WARM_UP");
    } else {
        cmd.env("ZYNTAX_DISABLE_WARM_UP", "1");
    }
    let output = cmd.output().expect("zypy starts");
    let _ = fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{case} exited {:?}: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(stdout, expected, "{case} (warm-up {warm_up})");
}

/// A pivot search over rows read through a dynamic field: the running
/// maximum is a loop-carried box and each candidate a fresh one.
#[test]
fn a_pivot_search_over_boxed_rows() {
    run("pivot_search", false);
    run("pivot_search", true);
}

/// scimark's LU factorisation of a 7x7 matrix.
#[test]
fn scimark_lu_factor() {
    run("lu_scimark", false);
    run("lu_scimark", true);
}

/// The same factorisation with the matrix's rows held in a field
/// another instance gives an int, so every element read is boxed.
#[test]
fn scimark_lu_factor_on_boxed_rows() {
    run("lu_scimark_dynamic", false);
    run("lu_scimark_dynamic", true);
}
