//! A number of unknown kind combined with a float is a float, and an
//! integer's float word is its value: the results of `+ - * // %`,
//! negation, `math.abs` and the rounding functions match the dynamic
//! arithmetic's, kind included, for integers and floats alike.

use std::path::Path;
use std::process::Command;

fn pinned(name: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
        .arg("run")
        .arg(dir.join(format!("{name}.lua")))
        .output()
        .expect("zylua starts");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected = std::fs::read_to_string(dir.join(format!("{name}.expected"))).unwrap();
    assert_eq!(String::from_utf8_lossy(&output.stdout), expected);
}

#[test]
fn unknown_kind_with_float_answers_as_dynamic_arithmetic_does() {
    pinned("number_with_float");
}

#[test]
fn integers_of_unknown_kind_convert_to_float_exactly() {
    pinned("number_float_word");
}
