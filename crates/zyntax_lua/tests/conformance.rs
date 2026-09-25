//! Conformance against the reference Lua 5.4, by category.
//!
//! A conformance case is a Lua program under `conformance/<category>/`.
//! It conforms when this frontend's compiled output matches what the
//! reference interpreter prints for the same program, byte for byte,
//! with the same exit status. Nothing here states an expectation by
//! hand: `lua5.4` produces it, and it is pinned beside the program as
//! `.expected` (and `.status` when the program exits non-zero) so the
//! suite runs without Lua and every expectation is a reviewed file.
//! A case pinned with `.message` also conforms in the first line its
//! uncaught error writes to stderr, after the program's name: a case
//! that exits non-zero pins one when it is first pinned.
//!
//! `conformance/official/` is the test suite that ships with Lua 5.4
//! (lua.org/tests), run as `all.lua` runs each file with `_U=true`, the
//! portable subset, and judged as `all.lua` judges it: a file conforms
//! when it runs to its end, exiting as the reference does with the
//! reference's last line. The hand-written categories are programs that
//! print what they assert, compared byte for byte.
//!
//! ## Known failures
//!
//! `conformance/KNOWN_FAILURES` lists cases that are expected to fail,
//! each with the git-bug issue tracking it. A listed case that fails is
//! reported and does not fail the suite. A listed case that PASSES fails
//! the suite, so the list cannot go stale. An unlisted case that fails
//! is a regression and fails the suite.
//!
//! An entry may end with a platform family, `unix` or `windows`: the
//! case's pinned output holds only there, because the reference's own
//! answer differs elsewhere. On that family the case runs as any other
//! and must pass; on any other it is not run and is reported as skipped.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance")
}

/// One KNOWN_FAILURES entry: the issue, and the platform family the
/// case's pinned output is limited to when the entry names one.
struct Listed {
    issue: String,
    only_on: Option<String>,
}

/// `category/file.lua` -> its entry, from KNOWN_FAILURES.
fn known_failures() -> HashMap<String, Listed> {
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
            let only_on = parts.next().map(str::to_string);
            if let Some(family) = &only_on {
                assert!(
                    matches!(family.as_str(), "unix" | "windows"),
                    "KNOWN_FAILURES: `{case}` names platform `{family}`; the families are `unix` and `windows`"
                );
            }
            out.insert(
                case.to_string(),
                Listed {
                    issue: issue.to_string(),
                    only_on,
                },
            );
        }
    }
    out
}

/// Whether this build targets the platform family `family`.
fn on_family(family: &str) -> bool {
    match family {
        "unix" => cfg!(unix),
        "windows" => cfg!(windows),
        _ => false,
    }
}

/// What one run produced.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    /// Its bytes, whatever they are: a Lua string need not be UTF-8.
    stdout: Vec<u8>,
    status: i32,
    /// The first line of stderr, less the program's name that `lua`
    /// puts before an uncaught error's message.
    message: String,
}

/// The message line of what a run wrote to stderr.
fn message_of(stderr: &str) -> String {
    let line = stderr.lines().next().unwrap_or("");
    match line.split_once(": ") {
        Some((_, message)) => message.to_string(),
        None => line.to_string(),
    }
}

/// Run a command with a deadline. A conformance case that hangs is a
/// failure that must be reported, not a suite that never finishes.
fn run_bounded(mut cmd: Command, limit: Duration) -> Outcome {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return Outcome {
                stdout: format!("<could not start: {e}>").into_bytes(),
                status: -1,
                message: String::new(),
            };
        }
    };
    // Both drained as they are written, so a program that prints more
    // than a pipe holds does not wait on a reader that waits on its
    // exit.
    fn drain<R: Read + Send + 'static>(s: Option<R>) -> Option<std::thread::JoinHandle<Vec<u8>>> {
        s.map(|mut s| {
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = s.read_to_end(&mut bytes);
                bytes
            })
        })
    }
    let reader = drain(child.stdout.take());
    let errors = drain(child.stderr.take());
    let collect = |reader: Option<std::thread::JoinHandle<Vec<u8>>>| {
        reader.and_then(|r| r.join().ok()).unwrap_or_default()
    };
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return Outcome {
                    stdout: collect(reader),
                    status: status.code().unwrap_or(-2),
                    message: message_of(&String::from_utf8_lossy(&collect(errors))),
                };
            }
            Ok(None) if start.elapsed() > limit => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = collect(reader);
                let _ = collect(errors);
                return Outcome {
                    stdout: format!("<timed out after {:?}>", limit).into_bytes(),
                    status: -3,
                    message: String::new(),
                };
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => {
                return Outcome {
                    stdout: format!("<wait failed: {e}>").into_bytes(),
                    status: -1,
                    message: String::new(),
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
/// the next run needs neither Lua nor a decision), with the message
/// pinned for it if any.
fn expected_for(case: &Path) -> Option<(Outcome, Option<String>)> {
    let pin = case.with_extension("expected");
    // A program that exits with a status pins it and its message
    // beside its output; one that exits 0 pins nothing extra.
    let status_pin = case.with_extension("status");
    let message_pin = case.with_extension("message");
    if let Ok(bytes) = fs::read(&pin) {
        let status = fs::read_to_string(&status_pin)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let message = fs::read_to_string(&message_pin)
            .ok()
            .map(|m| m.trim_end_matches('\n').to_string());
        return Some((
            Outcome {
                stdout: bytes,
                status,
                message: message.clone().unwrap_or_default(),
            },
            message,
        ));
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
    let mut message = None;
    if got.status != 0 {
        let _ = fs::write(&status_pin, got.status.to_string());
        let _ = fs::write(&message_pin, format!("{}\n", got.message));
        message = Some(got.message.clone());
    }
    Some((got, message))
}

/// Whether the compile worker runs beside the program: off, every
/// compile is made on the thread that asks for it; on, the worker takes
/// requests and replaces the quick first compiles of loop-free
/// functions with optimised ones.
#[derive(Clone, Copy)]
enum WarmUp {
    Off,
    On,
}

fn ours_for(case: &Path, warm_up: WarmUp) -> Outcome {
    // A binary run the moment it was linked can die before its first
    // instruction (the loader refusing it, a spawn failing under load);
    // that says nothing about the program, so it is tried again.
    for attempt in 0..3 {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_zylua"));
        match warm_up {
            WarmUp::Off => cmd.env("ZYNTAX_DISABLE_WARM_UP", "1"),
            WarmUp::On => cmd.env_remove("ZYNTAX_DISABLE_WARM_UP"),
        };
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
fn category(name: &str, warm_up: WarmUp) {
    let dir = root().join(name);
    // The official suite ships `libs/P1` as an empty directory, which
    // a checkout cannot hold; `attrib.lua` writes into it.
    if name == "official" {
        let _ = fs::create_dir_all(dir.join("libs").join("P1"));
    }
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
    let mut skipped: Vec<(String, String, String)> = Vec::new();
    let mut unpinned = 0;

    for case in &cases {
        let key = format!("{name}/{}", case.file_name().unwrap().to_string_lossy());
        // A case pinned for one platform family is not run on another;
        // on its own family the entry is not a known failure.
        let listed = match known.get(&key) {
            Some(Listed {
                issue,
                only_on: Some(family),
            }) => {
                if !on_family(family) {
                    skipped.push((key, family.clone(), issue.clone()));
                    continue;
                }
                None
            }
            Some(Listed {
                issue,
                only_on: None,
            }) => Some(issue),
            None => None,
        };
        let Some((expected, message)) = expected_for(case) else {
            unpinned += 1;
            continue;
        };
        let got = ours_for(case, warm_up);
        let ok = conforms(case, &got, &expected, message.as_deref());
        match (ok, listed) {
            (true, None) => passed += 1,
            (true, Some(issue)) => fixed.push((key, issue.clone())),
            (false, Some(issue)) => known_failed.push((key, issue.clone())),
            (false, None) => {
                regressions.push(format!(
                    "{key}\n    expected (lua5.4, exit {}):\n{}\n    got (zylua, exit {}):\n{}{}",
                    expected.status,
                    indent(&expected.stdout),
                    got.status,
                    indent(&got.stdout),
                    if message.is_some() && got.message != expected.message {
                        format!(
                            "\n    message expected: {}\n    message got:      {}",
                            expected.message, got.message
                        )
                    } else {
                        String::new()
                    }
                ));
            }
        }
    }

    let mode = match warm_up {
        WarmUp::Off => "",
        WarmUp::On => " (warm-up on)",
    };
    println!(
        "conformance/{name}{mode}: {passed} pass, {} known failing, {} regression(s), {} newly passing, {} skipped on this platform, {unpinned} unpinned",
        known_failed.len(),
        regressions.len(),
        fixed.len(),
        skipped.len()
    );
    for (k, issue) in &known_failed {
        println!("  known failing: {k}  ({issue})");
    }
    for (k, family, issue) in &skipped {
        println!("  skipped: {k}, pinned for {family} only  ({issue})");
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

/// Whether `got` matches the reference. An official file asserts what
/// it tests and ends by printing its final marker (`OK` for most), so it
/// conforms when it exits as the reference did and its last line is the
/// reference's; what it prints before that (seeds, timings, counts) is
/// not compared. A file the reference itself fails, and every
/// hand-written case, is compared byte for byte, and in its pinned
/// message when it has one.
fn conforms(case: &Path, got: &Outcome, expected: &Outcome, message: Option<&str>) -> bool {
    if got.status != expected.status {
        return false;
    }
    if is_official(case) && expected.status == 0 {
        return last_line(&got.stdout) == last_line(&expected.stdout);
    }
    got.stdout == expected.stdout && message.is_none_or(|m| got.message == m)
}

/// The last line that is not blank.
fn last_line(bytes: &[u8]) -> &[u8] {
    bytes
        .split(|&b| b == b'\n')
        .rev()
        .find(|l| !l.trim_ascii().is_empty())
        .unwrap_or(&[])
}

/// Output for a failure report; bytes that are not UTF-8 show as U+FFFD.
fn indent(bytes: &[u8]) -> String {
    let s = String::from_utf8_lossy(bytes);
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
                category(stringify!($name), WarmUp::Off);
            }
        )*
        /// Every category again with the compile worker on.
        mod warm_up {
            $(
                #[test]
                fn $name() {
                    super::category(stringify!($name), super::WarmUp::On);
                }
            )*
        }
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
    globals,
    utf8,
    modules,
    load,
    official,
}
