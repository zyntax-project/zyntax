//! Conformance against the reference Lua 5.4, by category.
//!
//! A conformance case is a Lua program under `conformance/<category>/`.
//! It conforms when this frontend's compiled output matches what the
//! reference interpreter prints for the same program, byte for byte,
//! with the same exit status. Nothing here states an expectation by
//! hand: `lua5.4` produces it, and it is pinned beside the program as
//! `.expected` (and `.status` when the program exits non-zero) so the
//! suite runs without Lua and every expectation is a reviewed file.
//!
//! `conformance/official/` is the test suite that ships with Lua 5.4
//! (lua.org/tests), run as `all.lua` runs each file with `_U=true`, the
//! portable subset. The hand-written categories are programs that
//! print what they assert.
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

/// `category/file.lua` -> git-bug id, from KNOWN_FAILURES.
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
    // Only stdout is compared. A piped stderr nobody reads would stall
    // the child once it filled the pipe.
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return Outcome {
                stdout: format!("<could not start: {e}>"),
                status: -1,
            };
        }
    };
    // Drained as it is written, so a program that prints more than a
    // pipe holds does not wait on a reader that waits on its exit.
    let reader = child.stdout.take().map(|mut s| {
        std::thread::spawn(move || {
            let mut out = String::new();
            let _ = s.read_to_string(&mut out);
            out
        })
    });
    let collect = |reader: Option<std::thread::JoinHandle<String>>| {
        reader.and_then(|r| r.join().ok()).unwrap_or_default()
    };
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return Outcome {
                    stdout: collect(reader),
                    status: status.code().unwrap_or(-2),
                };
            }
            Ok(None) if start.elapsed() > limit => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = collect(reader);
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
                };
            }
        }
    }
}

/// The reference interpreter, wherever it is: `LUA` names it, else
/// `lua5.4` then `lua` on the path.
fn reference() -> Command {
    if let Ok(lua) = std::env::var("LUA") {
        return Command::new(lua);
    }
    for candidate in ["lua5.4", "/opt/homebrew/opt/lua@5.4/bin/lua5.4", "lua"] {
        if Command::new(candidate).arg("-v").output().is_ok() {
            return Command::new(candidate);
        }
    }
    Command::new("lua5.4")
}

/// The official suite runs each file with `_U=true`, the portable
/// subset. Every case runs from its own directory under its bare name,
/// so a chunk name in its output is the same on every checkout.
fn is_official(case: &Path) -> bool {
    case.parent()
        .and_then(|d| d.file_name())
        .is_some_and(|d| d == "official")
}

/// The reference's answer, from the pinned file or from the reference
/// interpreter when the file does not exist yet (and then written, so
/// the next run needs neither Lua nor a decision).
fn expected_for(case: &Path) -> Option<Outcome> {
    let pin = case.with_extension("expected");
    // A program that exits with a status pins it beside its output; one
    // that exits 0 pins nothing extra.
    let status_pin = case.with_extension("status");
    if let Ok(text) = fs::read_to_string(&pin) {
        let status = fs::read_to_string(&status_pin)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        return Some(Outcome {
            stdout: text,
            status,
        });
    }
    let mut cmd = reference();
    if is_official(case) {
        cmd.arg("-e").arg("_U=true");
    }
    cmd.current_dir(case.parent().unwrap());
    cmd.arg("--").arg(case.file_name().unwrap());
    let got = run_bounded(cmd, Duration::from_secs(120));
    if got.status < 0 || got.status > 100 {
        eprintln!(
            "  cannot pin {}: the reference interpreter exited {} (is lua5.4 installed?)",
            case.display(),
            got.status
        );
        return None;
    }
    let _ = fs::write(&pin, &got.stdout);
    if got.status != 0 {
        let _ = fs::write(&status_pin, got.status.to_string());
    }
    Some(got)
}

fn ours_for(case: &Path) -> Outcome {
    // A binary run the moment it was linked can die before its first
    // instruction (the loader refusing it, a spawn failing under load);
    // that says nothing about the program, so it is tried again.
    for attempt in 0..3 {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_zylua"));
        // The warm-up thread races first-call compiles; see git-bug
        // 5f65d75241969a35a13e2bbe5a008e10bdcea080024cd9c5fae56badabe3b24a.
        cmd.env("ZYNTAX_DISABLE_WARM_UP", "1");
        cmd.arg("run");
        if is_official(case) {
            cmd.arg("-e").arg("_U=true");
        }
        cmd.current_dir(case.parent().unwrap());
        cmd.arg(case.file_name().unwrap());
        let got = run_bounded(cmd, Duration::from_secs(120));
        let died_before_running =
            got.status < 0 && got.status != -3 && got.stdout.is_empty() || got.status == -1;
        if !died_before_running || attempt == 2 {
            return got;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    unreachable!("the last attempt returns")
}

/// Run every case in one category and report.
fn category(name: &str) {
    let dir = root().join(name);
    let known = known_failures();
    let mut cases: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("no conformance category `{name}` at {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "lua"))
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
        let ok = got.status == expected.status && got.stdout == expected.stdout;
        match (ok, known.get(&key)) {
            (true, None) => passed += 1,
            (true, Some(issue)) => fixed.push((key, issue.clone())),
            (false, Some(issue)) => known_failed.push((key, issue.clone())),
            (false, None) => {
                regressions.push(format!(
                    "{key}\n    expected (lua5.4, exit {}):\n{}\n    got (zylua, exit {}):\n{}",
                    expected.status,
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
    basics,
    numbers,
    strings,
    control,
    functions,
    closures,
    tables,
    metatables,
    coroutines,
    stdlib,
    errors,
    patterns,
    official,
}
