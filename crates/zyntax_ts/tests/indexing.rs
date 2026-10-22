//! An index that cannot be proven in range is guarded by default and refused on request.

use std::io::Write;
use std::process::Command;
use zyntax_ts::{ErrorKind, Indexing, Options};

const SOURCE: &str = "function at(xs: number[], i: number): number { return xs[i]; }\n\
                      console.log(at([1, 2, 3], 1));\n\
                      console.log(at([1, 2, 3], 3));\n";

fn run(extra: &[&str]) -> std::process::Output {
    let mut file = tempfile::Builder::new().suffix(".ts").tempfile().unwrap();
    file.write_all(SOURCE.as_bytes()).unwrap();
    Command::new(env!("CARGO_BIN_EXE_zynode"))
        .arg("run")
        .args(extra)
        .arg(file.path())
        .output()
        .unwrap()
}

#[test]
fn checked_indexing_stops_at_an_index_typescript_would_read_as_undefined() {
    let output = run(&[]);
    assert_eq!(String::from_utf8_lossy(&output.stdout), "2\n");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("array index 3 is outside 0..3"), "{stderr}");
    assert!(!output.status.success());
}

#[test]
fn proven_indexing_refuses_the_unproven_index_at_compile_time() {
    let error = zyntax_ts::parse_program(
        SOURCE,
        "indexing.ts",
        Options {
            indexing: Indexing::Proven,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Unsupported);
    assert!(zyntax_ts::parse_program(SOURCE, "indexing.ts", Default::default()).is_ok());
    let output = run(&["--indexing", "proven"]);
    assert!(String::from_utf8_lossy(&output.stderr).contains("proven integer bounds"));
}
