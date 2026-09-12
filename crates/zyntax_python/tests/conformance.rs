//! Conformance against CPython, by category.
//!
//! A conformance case is a Python program under `conformance/<category>/`.
//! It conforms when this frontend's compiled output matches what CPython
//! prints for the same program, byte for byte. Nothing here states an
//! expectation by hand: `python3` produces it, and it is pinned beside
//! the program as `.expected` so the suite runs without Python and so
//! every expectation is a reviewed file rather than a belief.
//!
//! The categories follow CPython's own `Lib/test` modules, and the
//! cases in each are taken from the assertions those modules make,
//! rewritten as programs that print what they assert. CPython is the
//! standard; this measures distance from it.
//!
//! ## Known failures
//!
//! `conformance/KNOWN_FAILURES` lists cases that are expected to fail,
//! each with the git-bug issue tracking it. A listed case that fails is
//! reported and does not fail the suite. A listed case that PASSES fails
//! the suite, so the list cannot go stale. An unlisted case that fails
//! is a regression and fails the suite.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance")
}

/// `category/file.py` -> git-bug id, from KNOWN_FAILURES.
fn known_failures() -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Ok(text) = fs::read_to_string(root().join("KNOWN_FAILURES")) else {
        return out;
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        if let (Some(case), Some(issue)) = (parts.next(), parts.next()) {
            out.insert(case.to_string(), issue.to_string());
        }
    }
    out
}

/// What one run produced.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    stdout: String,
    status: i32,
}

/// Run a command with a deadline. A conformance case that hangs is a
/// failure that must be reported, not a suite that never finishes.
fn run_bounded(mut cmd: Command, limit: Duration) -> Outcome {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return Outcome {
                stdout: format!("<could not start: {e}>"),
                status: -1,
            }
        }
    };
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                if let Some(mut s) = child.stdout.take() {
                    let _ = s.read_to_string(&mut out);
                }
                return Outcome {
                    stdout: out,
                    status: status.code().unwrap_or(-2),
                };
            }
            Ok(None) if start.elapsed() > limit => {
                let _ = child.kill();
                let _ = child.wait();
                return Outcome {
                    stdout: format!("<timed out after {:?}>", limit),
                    status: -3,
                };
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => {
                return Outcome {
                    stdout: format!("<wait failed: {e}>"),
                    status: -1,
                }
            }
        }
    }
}

/// CPython's answer, from the pinned file or from `python3` when the
/// file does not exist yet (and then written, so the next run needs
/// neither Python nor a decision).
fn expected_for(case: &Path) -> Option<Outcome> {
    let pin = case.with_extension("expected");
    if let Ok(text) = fs::read_to_string(&pin) {
        return Some(Outcome {
            stdout: text,
            status: 0,
        });
    }
    let mut cmd = Command::new("python3");
    cmd.arg(case);
    let got = run_bounded(cmd, Duration::from_secs(30));
    if got.status != 0 {
        eprintln!(
            "  cannot pin {}: python3 exited {} (is python3 installed?)",
            case.display(),
            got.status
        );
        return None;
    }
    let _ = fs::write(&pin, &got.stdout);
    Some(got)
}

fn ours_for(case: &Path) -> Outcome {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_zypy"));
    cmd.arg("run").arg(case);
    run_bounded(cmd, Duration::from_secs(60))
}

/// Run every case in one category and report.
fn category(name: &str) {
    let dir = root().join(name);
    let known = known_failures();
    let mut cases: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("no conformance category `{name}` at {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "py"))
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "category `{name}` has no cases");

    let mut passed = 0;
    let mut known_failed: Vec<(String, String)> = Vec::new();
    let mut regressions: Vec<String> = Vec::new();
    let mut fixed: Vec<(String, String)> = Vec::new();
    let mut unpinned = 0;

    for case in &cases {
        let key = format!("{name}/{}", case.file_name().unwrap().to_string_lossy());
        let Some(expected) = expected_for(case) else {
            unpinned += 1;
            continue;
        };
        let got = ours_for(case);
        let ok = got.status == 0 && got.stdout == expected.stdout;
        match (ok, known.get(&key)) {
            (true, None) => passed += 1,
            (true, Some(issue)) => fixed.push((key, issue.clone())),
            (false, Some(issue)) => known_failed.push((key, issue.clone())),
            (false, None) => {
                regressions.push(format!(
                    "{key}\n    expected (CPython):\n{}\n    got (zypy, exit {}):\n{}",
                    indent(&expected.stdout),
                    got.status,
                    indent(&got.stdout)
                ));
            }
        }
    }

    println!(
        "conformance/{name}: {passed} pass, {} known failing, {} regression(s), {} newly passing, {unpinned} unpinned",
        known_failed.len(),
        regressions.len(),
        fixed.len()
    );
    for (k, issue) in &known_failed {
        println!("  known failing: {k}  ({issue})");
    }

    let mut problems = Vec::new();
    if !regressions.is_empty() {
        problems.push(format!(
            "{} case(s) fail and are not in KNOWN_FAILURES:\n\n{}",
            regressions.len(),
            regressions.join("\n\n")
        ));
    }
    if !fixed.is_empty() {
        let list: Vec<String> = fixed.iter().map(|(k, i)| format!("  {k}  ({i})")).collect();
        problems.push(format!(
            "{} case(s) now pass but are still listed in KNOWN_FAILURES; remove them and \
             close their issues:\n{}",
            fixed.len(),
            list.join("\n")
        ));
    }
    assert!(problems.is_empty(), "\n{}", problems.join("\n\n"));
}

fn indent(s: &str) -> String {
    s.lines()
        .map(|l| format!("        {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

macro_rules! categories {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                category(stringify!($name));
            }
        )*
    };
}

categories! {
    grammar,
    int,
    float,
    bool,
    compare,
    unary,
    binop,
    augassign,
    scope,
    functions,
    class,
    list,
    tuple,
    dict,
    set,
    string,
    generators,
    exceptions,
    builtin,
}
