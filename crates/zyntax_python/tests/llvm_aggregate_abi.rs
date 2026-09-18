#![cfg(feature = "llvm-backend")]

use std::path::Path;
use std::process::Command;

#[test]
fn promoted_list_parameter_keeps_caller_header_and_aliases() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llvm_list_header.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYPY_LLVM", "1")
        .env("ZYNTAX_DISABLE_INTERP_OPTS", "1")
        .env("ZYNTAX_OSR_TRACE", "1")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "5000 10000 10000 10000 4999");
    let grow = stderr
        .find("(grow) at tier 2")
        .expect("grow promotes to LLVM");
    assert!(
        stderr[grow..].contains("[osr] llvm install tier=2"),
        "{stderr}"
    );
    assert!(!stderr.contains("LLVM compile failed"), "{stderr}");
}
