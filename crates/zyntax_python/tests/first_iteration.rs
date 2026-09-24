//! A kernel's first iteration pays for the compiles its calls make. A
//! loop-free function first called from native code runs its body as
//! lowered until the worker's optimised compile replaces it, so no call
//! waits on the per-function pipeline for one.
#![cfg(all(feature = "llvm-backend", not(debug_assertions)))]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A directory holding the kernel and the speed harness's shims, as
/// `benchmarks/speed/run.py` stages it.
fn stage(kernel: &str) -> PathBuf {
    let speed = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/speed");
    let dir = std::env::temp_dir().join(format!(
        "zypy-first-iteration-{}-{kernel}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::copy(speed.join("kernels").join(kernel), dir.join(kernel)).unwrap();
    for shim in ["util.py", "optparse.py"] {
        fs::copy(speed.join("shims").join(shim), dir.join(shim)).unwrap();
    }
    dir
}

/// What the lazy compiler traced over a run of `iterations`.
fn trace(kernel: &str, iterations: usize) -> String {
    let dir = stage(kernel);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_zypy"));
    cmd.arg("run")
        .arg(dir.join(kernel))
        .arg("-n")
        .arg(iterations.to_string())
        .current_dir(&dir)
        .env("ZYPY_LLVM", "1")
        .env("ZYNTAX_TRACE_LAZY", "1")
        .env_remove("ZYNTAX_DISABLE_WARM_UP");
    let output = cmd.output().expect("zypy starts");
    let _ = fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "{stderr}");
    let times = stdout
        .lines()
        .filter(|l| l.trim().parse::<f64>().is_ok())
        .count();
    assert_eq!(times, iterations, "{stdout}");
    stderr
}

#[test]
fn no_loop_free_function_waits_on_the_pipeline_at_its_first_call() {
    let trace = trace("deltablue.py", 5);
    assert!(
        trace.contains("[lazy] quick baseline "),
        "no quick baseline was compiled:\n{trace}"
    );
    let waited: Vec<&str> = trace
        .lines()
        .filter(|l| {
            l.starts_with("[lazy] compiled ")
                && l.contains(", loop-free)")
                && l.ends_with(" on zyntax-first-call-compile")
        })
        .collect();
    assert!(
        waited.is_empty(),
        "loop-free functions compiled through the pipeline on a caller's thread:\n{}",
        waited.join("\n")
    );
}
