//! An array part holding only integers, or only floats, is stored flat,
//! without a box per element. Every way of reading or changing such a
//! table gives the answer a boxed array part does: `ints.lua` checks
//! itself and prints `done`; the output of `traversal.lua` and
//! `floats.lua` is pinned.

use std::path::Path;
use std::process::Command;

fn run(name: &str) -> String {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("flat_arrays")
        .join(name);
    let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
        .arg("run")
        .arg(&script)
        .output()
        .expect("zylua starts");
    assert!(
        output.status.success(),
        "{name} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn integer_arrays_answer_as_boxed_ones_do() {
    assert_eq!(run("ints.lua").trim(), "done");
}

fn pinned(name: &str) {
    let expected = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("flat_arrays")
            .join(format!("{name}.expected")),
    )
    .expect("pinned output");
    assert_eq!(run(&format!("{name}.lua")), expected);
}

#[test]
fn traversal_and_mutation_of_integer_arrays() {
    pinned("traversal");
}

#[test]
fn float_arrays_answer_as_boxed_ones_do() {
    pinned("floats");
}
