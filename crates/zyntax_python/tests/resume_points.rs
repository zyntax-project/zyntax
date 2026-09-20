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
        stderr.contains("outlined resume point"),
        "the resume point was not outlined:\n{stderr}"
    );
    assert!(
        stderr.contains("[interp] transfer site="),
        "the frame never left through a resume point:\n{stderr}"
    );
}

/// The frame leaves at an inner loop's header: the outer loop's counter
/// and the bodies loop's state come in from the frame and are repaired
/// at the header, and the region runs the rest of the call.
#[test]
fn a_frame_leaves_an_inner_loop_into_the_outlined_region() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nested_loops.py");
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
    assert_eq!(stdout.trim(), "1297500.0\n129.75", "{stderr}");
    assert!(
        stderr.contains("outlined resume point") && stderr.contains("[interp] transfer site="),
        "{stderr}"
    );
}

/// The frame resumed into the outlined region moves on to the
/// optimizing tier mid-loop: the region is compiled there too, and its
/// resume points answer the probes of the region's baseline code.
#[cfg(feature = "llvm-backend")]
#[test]
fn a_frame_in_the_outlined_region_moves_to_the_optimizing_tier() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/long_single_call.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYPY_LLVM", "1")
        .env("ZYNTAX_OSR_TRACE", "1")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "25574991424", "{stderr}");
    assert!(
        stderr.contains("outlined resume point"),
        "the resume point was not outlined:\n{stderr}"
    );
    let region = stderr
        .find("(run$resume0) at tier 1")
        .expect("the region is compiled at the optimizing tier");
    assert!(
        stderr[region..].contains("(llvm)"),
        "the frame never moved to the optimizing tier:\n{stderr}"
    );
}
