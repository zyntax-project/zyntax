use std::path::Path;
use std::process::Command;

/// The function returns the table its loop fills. The frame leaves
/// through a resume point that hands the table back, and the call after
/// runs the compiled body.
#[test]
fn a_frame_in_a_function_returning_a_table_leaves_through_a_resume_point() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/table_return.lua");
    let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
        .arg("run")
        .arg(script)
        .env("ZYNTAX_OSR_TRACE", "1")
        .env("ZYNTAX_TRACE_INTERP", "1")
        .output()
        .expect("zylua starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(
        stdout.trim(),
        "30\t0\t1483337328\n30\t1483337328",
        "{stderr}"
    );
    assert!(
        stderr.contains("lua$main$1 site=") && stderr.contains("[interp] transfer site="),
        "the frame never left through a resume point:\n{stderr}"
    );
}
