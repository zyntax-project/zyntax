use std::path::Path;
use std::process::{Command, Output};

fn run(disable: bool) -> Output {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/static_hot_loop.py");
    let mut command = Command::new(env!("CARGO_BIN_EXE_zypy"));
    command
        .arg("run")
        .arg(script)
        .env("ZYPY_LLVM", "1")
        .env("ZYNTAX_OSR_TRACE", "1")
        .env("ZYNTAX_TRACE_INTERP", "1");
    if disable {
        command.env("ZYNTAX_DISABLE_STATIC_HOT", "1");
    }
    command.output().expect("zypy starts")
}

/// A function whose loop runs a constant number of times is compiled
/// before its first call: the interpreter hands that call to native
/// code, and no frame of it leaves the interpreter through a resume
/// point.
#[test]
fn a_constant_bound_loop_is_never_interpreted() {
    let output = run(false);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "1500000", "{stderr}");
    assert!(!stderr.contains("outlined resume point"), "{stderr}");
    assert!(!stderr.contains("[interp] transfer site="), "{stderr}");
    assert!(!stderr.contains("Interpreted {"), "{stderr}");
    assert_eq!(stderr.matches("(main) at tier 0").count(), 1, "{stderr}");
    // The optimizing tier compiles the function once, not a region of it.
    assert!(stderr.matches("(main) at tier 1").count() <= 1, "{stderr}");
    assert!(!stderr.contains("main$resume"), "{stderr}");
    let call = stderr.find("[interp] call main(").expect("main is called");
    // The next thing the interpreter does is call the native code.
    let after = stderr[call..]
        .lines()
        .skip(1)
        .find(|l| l.starts_with("[interp] "))
        .unwrap_or_default();
    assert!(after.starts_with("[interp] native"), "{stderr}");
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
