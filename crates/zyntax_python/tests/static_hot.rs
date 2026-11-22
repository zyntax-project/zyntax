use std::path::Path;
use std::process::{Command, Output};

/// `zypy` on the fixture with the compile workers off, so the order of
/// what happens does not depend on how fast code is made.
fn run(disable: bool) -> Output {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/static_hot_loop.py");
    let mut command = Command::new(env!("CARGO_BIN_EXE_zypy"));
    command
        .arg("run")
        .arg(script)
        .env_remove("ZYPY_LLVM")
        .env("ZYNTAX_DISABLE_WARM_UP", "1")
        .env("ZYNTAX_OSR_TRACE", "1")
        .env("ZYNTAX_TRACE_INTERP", "1");
    if disable {
        command.env("ZYNTAX_DISABLE_STATIC_HOT", "1");
    }
    command.output().expect("zypy starts")
}

/// A function whose loop runs a constant number of times starts in the
/// interpreter, as every function does: its first call does not wait
/// for code. The running loop asks for native code and the frame
/// transfers to it.
#[test]
fn a_constant_bound_loop_starts_interpreted_and_transfers_its_running_frame() {
    let output = run(false);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "1500000", "{stderr}");
    let entered = stderr
        .find("[interp] call main(")
        .expect("main is interpreted");
    let asked = stderr
        .find("[interp] asks at site=")
        .expect("the running loop asks for code");
    let transferred = stderr
        .find("[interp] transfer site=")
        .expect("the frame transfers to native code");
    assert!(entered < asked && asked < transferred, "{stderr}");
}

/// `ZYNTAX_DISABLE_STATIC_HOT=1` leaves the function to the interpreter
/// until its loop warms, and the frame leaves through a resume point.
#[test]
fn the_switch_leaves_the_loop_to_the_interpreter() {
    let output = run(true);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "1500000", "{stderr}");
    assert!(stderr.contains("Interpreted {"), "{stderr}");
}
