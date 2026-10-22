//! A boolean passed to a native function after the caller was recompiled by a tier-up
//! keeps one signature for the symbol.

use std::io::Write;
use std::process::Command;

#[test]
fn boolean_arguments_survive_tier_up() {
    let mut file = tempfile::Builder::new().suffix(".ts").tempfile().unwrap();
    file.write_all(
        b"let s = 0;\nfor (let i = 0; i < 3000000; i++) { s += i % 7; }\nconsole.log(s, s > 1);\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_zynode"))
        .arg("run")
        .arg(file.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "8999994 true\n");
}
