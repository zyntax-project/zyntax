use std::path::Path;
use std::process::Command;

/// An interpreted frame leaves through a resume point compiled before
/// the function's own body; the resumed code calls the function, whose
/// definition does not exist yet.
#[test]
fn a_resumed_frame_calls_its_function_before_the_body_is_compiled() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recursive_loop.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYNTAX_OSR_TRACE", "1")
        .env("ZYNTAX_TRACE_INTERP", "1")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "2399976", "{stderr}");
    assert!(
        stderr.contains("[interp] transfer site="),
        "the frame never left through a resume point:\n{stderr}"
    );
}
