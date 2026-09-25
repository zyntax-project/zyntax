//! The official Lua 5.4.9 suite run whole, as lua.org runs it: `all.lua`
//! runs every file in one state and in its own order.
//!
//! - `basic_all`: `zylua -e _U=true all.lua`, the portable configuration,
//!   on the default tier and again with `ZYLUA_LLVM=1`.
//! - `complete_all`: `zylua all.lua`, the complete configuration.
//! - `main_lua`: `zylua main.lua` in the complete configuration, where
//!   it runs this binary with the stand-alone interpreter's options.
//!
//! The complete configuration runs on unix with the C test libraries in
//! `official/libs` built.
//!
//! A run conforms when it exits as the reference does, prints
//! `final OK !!!` exactly when the reference does, and ends on the
//! reference's last non-blank line; when the reference exits non-zero,
//! its error message must match too. The reference's answer is pinned:
//! `official/all.expected` for the portable run (the per-file run of
//! `all.lua` has the same derived configuration), and
//! `official_complete/<file>.<family>.expected` for the complete one.
//!
//! These cases are judged in release builds only; a debug build prints
//! why each is skipped. KNOWN_FAILURES lists them as
//! `official_suite/<case>`, under the same rules as the conformance
//! categories.

mod common;

use common::*;
use std::process::Command;
use std::time::Duration;

/// How a whole-suite run is made and what it is compared with.
struct Case {
    /// Its KNOWN_FAILURES key, after `official_suite/`.
    name: &'static str,
    args: &'static [&'static str],
    llvm: bool,
    pin: Pin,
    /// The reference command that pins the answer when none is pinned.
    reference_args: &'static [&'static str],
    limit: Duration,
}

const FINAL_OK: &[u8] = b"final OK !!!";

fn complete_pin(file: &str) -> Pin {
    Pin {
        dir: root().join("official_complete"),
        stem: format!("{file}.{}", std::env::consts::OS),
    }
}

/// Why `case` cannot run here, if it cannot.
fn skip_reason(case: &Case, complete: bool) -> Option<String> {
    if cfg!(debug_assertions) {
        return Some("a debug build; the whole suite is judged in release".into());
    }
    if complete && !cfg!(unix) {
        return Some("the complete configuration needs a POSIX shell".into());
    }
    if complete && let Err(why) = official_libs() {
        return Some(why);
    }
    if case.llvm {
        let backend = Command::new(env!("CARGO_BIN_EXE_zylua"))
            .arg("backend")
            .env("ZYLUA_LLVM", "1")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        if backend != "llvm" {
            return Some(format!(
                "zylua was built without the LLVM tier (backend: {backend:?})"
            ));
        }
    }
    None
}

fn run(case: Case, complete: bool) {
    let key = format!("official_suite/{}", case.name);
    if let Some(why) = skip_reason(&case, complete) {
        println!("{key}: skipped, {why}");
        return;
    }
    let listed = known_failures().remove(&key);
    if let Some(Listed {
        only_on: Some(family),
        issue,
    }) = &listed
        && !on_family(family)
    {
        println!("{key}: skipped, pinned for {family} only  ({issue})");
        return;
    }
    let dir = official_dir();
    let _ = std::fs::create_dir_all(dir.join("libs").join("P1"));

    let mut reference = reference();
    reference.current_dir(&dir).args(case.reference_args);
    let Some((expected, message)) = case.pin.read_or_pin(reference, case.limit) else {
        println!("{key}: skipped, no pinned answer and no reference interpreter");
        return;
    };

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_zylua"));
    cmd.current_dir(&dir).args(case.args);
    cmd.env_remove("ZYNTAX_DISABLE_WARM_UP");
    if case.llvm {
        cmd.env("ZYLUA_LLVM", "1");
    } else {
        cmd.env_remove("ZYLUA_LLVM");
    }
    let started = std::time::Instant::now();
    let got = run_bounded(cmd, case.limit);
    let elapsed = started.elapsed();

    let ok = got.status == expected.status
        && has_line(&got.stdout, FINAL_OK) == has_line(&expected.stdout, FINAL_OK)
        && last_line(&got.stdout) == last_line(&expected.stdout)
        && message.as_deref().is_none_or(|m| got.message == m);
    let issue = listed.map(|l| l.issue);
    println!(
        "{key}: {} in {elapsed:.1?} (exit {}, reference exit {})",
        match (ok, &issue) {
            (true, None) => "conforms".to_string(),
            (true, Some(i)) => format!("conforms but is listed in KNOWN_FAILURES ({i})"),
            (false, Some(i)) => format!("known failing ({i})"),
            (false, None) => "FAILS".to_string(),
        },
        got.status,
        expected.status,
    );
    if !ok {
        println!(
            "  zylua stdout:\n{}\n  zylua stderr:\n{}\n  reference stdout (pinned, exit {}{}):\n{}",
            head_and_tail(&got.stdout, 40),
            head_and_tail(&got.stderr, 40),
            expected.status,
            message
                .map(|m| format!(", message {m:?}"))
                .unwrap_or_default(),
            head_and_tail(&expected.stdout, 40),
        );
    }
    match (ok, issue) {
        (false, None) => panic!("{key} does not conform and is not in KNOWN_FAILURES"),
        (true, Some(issue)) => panic!(
            "{key} now conforms but is still listed in KNOWN_FAILURES; remove it and close {issue}"
        ),
        _ => {}
    }
}

fn basic(name: &'static str, llvm: bool) -> Case {
    Case {
        name,
        args: &["-e", "_U=true", "all.lua"],
        llvm,
        pin: Pin {
            dir: official_dir(),
            stem: "all".into(),
        },
        reference_args: &["-e", OFFICIAL_PRELUDE, "--", "all.lua"],
        limit: Duration::from_secs(600),
    }
}

#[test]
fn basic_all() {
    run(basic("basic_all", false), false);
}

#[test]
fn basic_all_llvm() {
    run(basic("basic_all_llvm", true), false);
}

#[test]
fn complete_all() {
    let case = Case {
        name: "complete_all",
        args: &["all.lua"],
        llvm: false,
        pin: complete_pin("all"),
        reference_args: &["--", "all.lua"],
        limit: Duration::from_secs(1800),
    };
    run(case, true);
}

#[test]
fn main_lua() {
    let case = Case {
        name: "main_lua",
        args: &["main.lua"],
        llvm: false,
        pin: complete_pin("main"),
        reference_args: &["--", "main.lua"],
        limit: Duration::from_secs(600),
    };
    run(case, true);
}
