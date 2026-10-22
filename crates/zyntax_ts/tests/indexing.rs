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

/// Runs a program with a deadline: an index conversion that traps in compiled code hangs.
fn run_source(source: &str) -> std::process::Output {
    let mut file = tempfile::Builder::new().suffix(".ts").tempfile().unwrap();
    file.write_all(source.as_bytes()).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_zynode"))
        .arg("run")
        .arg(file.path())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while child.try_wait().unwrap().is_none() {
        if std::time::Instant::now() > deadline {
            child.kill().unwrap();
            panic!("program did not finish: {source}");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    child.wait_with_output().unwrap()
}

#[test]
fn unrepresentable_indexes_stop_cleanly_in_compiled_code() {
    for index in ["NaN", "1e19", "-1e300", "Infinity", "2.5"] {
        let output = run_source(&format!(
            "function at(xs: number[], i: number): number {{ return xs[i]; }}\n\
             const xs: number[] = [1, 2, 3];\n\
             let sum = 0;\n\
             for (let i = 0; i < 300000; i++) {{ sum += at(xs, i % 3); }}\n\
             console.log(sum);\n\
             console.log(at(xs, {index}));\n"
        ));
        assert!(!output.status.success(), "{index}");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "600000\n",
            "{index}"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("is outside 0..3"),
            "{index}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
