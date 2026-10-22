//! A program that finishes while an LLVM tier-up compile is still running does not wait for it.
//!
//! Needs the LLVM backend, and a release build so the program is short enough to end first:
//! `cargo test --release -p zyntax_ts --features llvm-backend --test exit`.
#![cfg(feature = "llvm-backend")]

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A hot top-level loop over many functions, so one large module is compiling at the end.
fn source() -> String {
    let functions = 120;
    let mut source = String::new();
    for i in 0..functions {
        source.push_str(&format!(
            "function f{i}(x: number): number {{ let s = 0; for (let k = 0; k < 40; k++) {{ s += (x + k * {}) % 7; }} return s; }}\n",
            i + 1
        ));
    }
    let calls: Vec<_> = (0..functions).map(|i| format!("f{i}(r)")).collect();
    source.push_str(&format!(
        "let total = 0;\nfor (let r = 0; r < 1000; r++) {{ total += {}; }}\nconsole.log(total);\n",
        calls.join(" + ")
    ));
    source
}

#[test]
fn exit_does_not_wait_for_a_compile_in_flight() {
    let mut file = tempfile::Builder::new().suffix(".ts").tempfile().unwrap();
    file.write_all(source().as_bytes()).unwrap();
    // The delay depends on timing, so the worst of several runs decides.
    let mut worst = Duration::ZERO;
    for _ in 0..5 {
        let mut child = Command::new(env!("CARGO_BIN_EXE_zynode"))
            .args(["run", "--llvm"])
            .arg(file.path())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let printed = Instant::now();
        assert!(!line.is_empty(), "the program printed nothing");
        let status = child.wait().unwrap();
        assert!(status.success());
        worst = worst.max(printed.elapsed());
    }
    assert!(
        worst < Duration::from_millis(400),
        "the process stayed alive {worst:?} after its output"
    );
}
