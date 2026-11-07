//! A number of unknown kind combined with a float is a float: the
//! results of `+ - * // %` match the dynamic arithmetic's, kind
//! included, for integers and floats alike.

use std::path::Path;
use std::process::Command;

#[test]
fn unknown_kind_with_float_answers_as_dynamic_arithmetic_does() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
        .arg("run")
        .arg(dir.join("number_with_float.lua"))
        .output()
        .expect("zylua starts");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected = std::fs::read_to_string(dir.join("number_with_float.expected")).unwrap();
    assert_eq!(String::from_utf8_lossy(&output.stdout), expected);
}
