//! Conformance against the reference Lua 5.4, by category.
//!
//! A conformance case is a Lua program under `conformance/<category>/`.
//! It conforms when this frontend's compiled output matches what the
//! reference interpreter prints for the same program, byte for byte,
//! with the same exit status. Nothing here states an expectation by
//! hand: `lua5.4` produces it, and it is pinned beside the program as
//! `.expected` (and `.status` when the program exits non-zero) so the
//! suite runs without Lua and every expectation is a reviewed file.
//! A case pinned with `.message` also conforms in the line its uncaught
//! error writes to stderr, after the program's name: a case that exits
//! non-zero pins one when it is first pinned.
//!
//! `conformance/official/` is the test suite that ships with Lua 5.4.9
//! (lua.org/tests). Each file runs on its own in the configuration
//! `all.lua` derives from `_U=true` (`_soft`, `_port` and `_nomsg` set)
//! and is judged as `all.lua` judges it: a file conforms when it runs to
//! its end, exiting as the reference does with the reference's last
//! line. `official_complete` runs the same files in lua.org's complete
//! configuration (no prelude, the C test libraries built), in release
//! builds on unix, against pins kept per platform family in
//! `conformance/official_complete/<file>.<family>.expected`. The
//! hand-written categories are programs that print what they assert,
//! compared byte for byte. Every run's stdin is a pipe closed at once.
//!
//! ## Known failures
//!
//! `conformance/KNOWN_FAILURES` lists cases that are expected to fail,
//! each with the git-bug issue tracking it. A listed case that fails is
//! reported and does not fail the suite. A listed case that PASSES fails
//! the suite, so the list cannot go stale. An unlisted case that fails
//! is a regression and fails the suite.
//!
//! An entry may end with a platform family, `unix`, `windows`, `macos`
//! or `linux`: the case's pinned output holds only there, because the
//! reference's own answer differs elsewhere. On that family the case
//! runs as any other and must pass; on any other it is not run and is
//! reported as skipped. An entry that ends with `failing-on <family>`
//! is known to fail on that family only, and must pass on the others.
//!
//! ## C modules
//!
//! `conformance/capi/` and the complete configuration's `attrib.lua`
//! load C modules, built from the sources beside them before the
//! category runs (see `common/c_modules.rs`); the reference is pinned
//! with the same libraries built. Where they cannot be built, those
//! cases are reported as skipped with the reason, never as passing.

mod common;

use common::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// How a category's cases run and are judged.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Suite {
    /// Hand-written programs, compared byte for byte.
    Written,
    /// The official files, each run on its own in the configuration
    /// `all.lua` derives from `_U=true`.
    Official,
    /// The official files in lua.org's complete configuration: no
    /// prelude, the C test libraries built, pinned per platform family.
    Complete,
}

impl Suite {
    fn of(category: &str) -> Suite {
        match category {
            "official" => Suite::Official,
            "official_complete" => Suite::Complete,
            _ => Suite::Written,
        }
    }

    /// Where the category's programs are.
    fn sources(self, category: &str) -> PathBuf {
        match self {
            Suite::Complete => official_dir(),
            _ => root().join(category),
        }
    }

    fn prelude(self) -> Option<&'static str> {
        match self {
            Suite::Official => Some(OFFICIAL_PRELUDE),
            _ => None,
        }
    }

    /// Where the reference's answer for `case` is pinned.
    fn pin(self, category: &str, case: &Path) -> Pin {
        let stem = case.file_stem().unwrap().to_string_lossy();
        match self {
            Suite::Complete => Pin {
                dir: root().join(category),
                stem: format!("{stem}.{}", std::env::consts::OS),
            },
            _ => Pin {
                dir: case.parent().unwrap().to_path_buf(),
                stem: stem.into_owned(),
            },
        }
    }
}

/// The reference's answer for `case`, pinned or else produced by the
/// reference interpreter and pinned, with its message if any. Every
/// case runs from its own directory under its bare name, so a chunk
/// name in its output is the same on every checkout.
fn expected_for(suite: Suite, category: &str, case: &Path) -> Option<(Outcome, Option<String>)> {
    let mut cmd = reference();
    if let Some(prelude) = suite.prelude() {
        cmd.arg("-e").arg(prelude);
    }
    cmd.current_dir(case.parent().unwrap());
    cmd.arg("--").arg(case.file_name().unwrap());
    suite
        .pin(category, case)
        .read_or_pin(cmd, Duration::from_secs(120))
}

/// Whether the compile worker runs beside the program: off, every
/// compile is made on the thread that asks for it; on, the worker takes
/// requests and replaces the quick first compiles of loop-free
/// functions with optimised ones.
#[derive(Clone, Copy)]
enum WarmUp {
    Off,
    On,
    /// Off, and the collector runs at nearly every allocation, so weak
    /// tables are swept in the middle of whatever the program is doing.
    Stressed,
}

fn ours_for(suite: Suite, case: &Path, warm_up: WarmUp) -> Outcome {
    // A binary run the moment it was linked can die before its first
    // instruction (the loader refusing it, a spawn failing under load);
    // that says nothing about the program, so it is tried again.
    for attempt in 0..3 {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_zylua"));
        match warm_up {
            WarmUp::Off => cmd.env("ZYNTAX_DISABLE_WARM_UP", "1"),
            WarmUp::On => cmd.env_remove("ZYNTAX_DISABLE_WARM_UP"),
            WarmUp::Stressed => cmd
                .env("ZYNTAX_DISABLE_WARM_UP", "1")
                .env("ZYNTAX_GC_FLOOR_KB", "1"),
        };
        cmd.arg("run");
        if let Some(prelude) = suite.prelude() {
            cmd.arg("-e").arg(prelude);
        }
        cmd.current_dir(case.parent().unwrap());
        cmd.arg(case.file_name().unwrap());
        let got = run_bounded(cmd, FILE_DEADLINE);
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
    let suite = Suite::of(name);
    let dir = suite.sources(name);
    // The official suite ships `libs/P1` as an empty directory, which
    // a checkout cannot hold; `attrib.lua` writes into it.
    if suite != Suite::Written {
        let _ = fs::create_dir_all(dir.join("libs").join("P1"));
    }
    // The C modules the category's cases load, and why they are not
    // there when they cannot be built.
    let c_modules: Option<Result<(), String>> = match name {
        "official_complete" => Some(official_libs()),
        "capi" => {
            let sources = c_modules::sources_in(&dir);
            let pairs: Vec<(&str, &str)> = sources
                .iter()
                .map(|(l, s)| (l.as_str(), s.as_str()))
                .collect();
            Some(c_modules::build(&dir, &pairs))
        }
        _ => None,
    };
    let needs_c = |key: &str| name == "capi" || key == "official_complete/attrib.lua";
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
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut unbuilt: Vec<(String, String)> = Vec::new();
    let mut unpinned = 0;

    for case in &cases {
        let file = case.file_name().unwrap().to_string_lossy();
        let key = format!("{name}/{file}");
        if suite != Suite::Written
            && let Some((_, why)) = NOT_RUN.iter().find(|(f, _)| *f == file)
        {
            skipped.push((key, why.to_string()));
            continue;
        }
        // A case pinned for one platform family is not run on another;
        // on its own family the entry is not a known failure.
        let listed = match known.get(&key) {
            Some(Listed {
                issue,
                only_on: Some(family),
                ..
            }) => {
                if !on_family(family) {
                    skipped.push((key, format!("pinned for {family} only  ({issue})")));
                    continue;
                }
                None
            }
            Some(Listed {
                issue,
                fails_on: Some(family),
                ..
            }) => on_family(family).then_some(issue),
            Some(Listed { issue, .. }) => Some(issue),
            None => None,
        };
        if let Some(Err(reason)) = &c_modules
            && needs_c(&key)
        {
            unbuilt.push((key, reason.clone()));
            continue;
        }
        let Some((expected, message)) = expected_for(suite, name, case) else {
            unpinned += 1;
            continue;
        };
        let got = ours_for(suite, case, warm_up);
        let ok = conforms(suite, &got, &expected, message.as_deref());
        match (ok, listed) {
            (true, None) => passed += 1,
            (true, Some(issue)) => fixed.push((key, issue.clone())),
            (false, Some(issue)) => known_failed.push((key, issue.clone())),
            (false, None) => {
                regressions.push(format!(
                    "{key}\n    expected (lua5.4, exit {}):\n{}\n    got (zylua, exit {}):\n{}{}",
                    expected.status,
                    head_and_tail(&expected.stdout, 40),
                    got.status,
                    head_and_tail(&got.stdout, 40),
                    if got.message != expected.message
                        && (message.is_some() || suite != Suite::Written)
                    {
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
        WarmUp::Stressed => " (collector stressed)",
    };
    println!(
        "conformance/{name}{mode}: {passed} pass, {} known failing, {} regression(s), {} newly passing, {} skipped, {unpinned} unpinned",
        known_failed.len(),
        regressions.len(),
        fixed.len(),
        skipped.len() + unbuilt.len()
    );
    for (k, issue) in &known_failed {
        println!("  known failing: {k}  ({issue})");
    }
    for (k, why) in &skipped {
        println!("  skipped: {k}, {why}");
    }
    for (k, reason) in &unbuilt {
        println!("  skipped: {k}, its C modules are not built: {reason}");
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
/// reference's; what it prints before that (seeds, timings, counts, the
/// interpreter's path) is not compared. When the reference itself
/// exits non-zero, its error message must match too. A hand-written
/// case is compared byte for byte, and in its pinned message when it
/// has one.
fn conforms(suite: Suite, got: &Outcome, expected: &Outcome, message: Option<&str>) -> bool {
    if got.status != expected.status {
        return false;
    }
    if suite != Suite::Written {
        return last_line(&got.stdout) == last_line(&expected.stdout)
            && message.is_none_or(|m| got.message == m);
    }
    got.stdout == expected.stdout && message.is_none_or(|m| got.message == m)
}

/// The complete configuration runs only where lua.org defines it
/// (unix), in a release build, with the C test libraries built.
#[test]
fn official_complete() {
    let skip = if cfg!(debug_assertions) {
        Some("a debug build; the complete configuration is judged in release".to_string())
    } else if !cfg!(unix) {
        Some("the complete configuration needs a POSIX shell".to_string())
    } else {
        official_libs().err()
    };
    match skip {
        Some(why) => println!("conformance/official_complete: skipped, {why}"),
        None => category("official_complete", WarmUp::On),
    }
}

#[test]
fn families_parse() {
    let known = parse_known_failures(
        "# comment\n\
         official/a.lua 1111111\n\
         official/b.lua 2222222   macos\n\
         official/c.lua 3333333 linux\n\
         stdlib/d.lua 4444444 unix\n\
         stdlib/e.lua 5555555 windows\n\
         official/f.lua 6666666 failing-on linux\n",
    );
    let family = |case: &str| known[case].only_on.as_deref();
    assert_eq!(known.len(), 6);
    assert_eq!(known["official/a.lua"].issue, "1111111");
    assert_eq!(family("official/a.lua"), None);
    assert_eq!(family("official/b.lua"), Some("macos"));
    assert_eq!(family("official/c.lua"), Some("linux"));
    assert_eq!(family("stdlib/d.lua"), Some("unix"));
    assert_eq!(family("stdlib/e.lua"), Some("windows"));
    assert_eq!(family("official/f.lua"), None);
    assert_eq!(known["official/f.lua"].fails_on.as_deref(), Some("linux"));
    assert_eq!(on_family("macos"), cfg!(target_os = "macos"));
    assert_eq!(on_family("linux"), cfg!(target_os = "linux"));
    assert_eq!(on_family("unix"), cfg!(unix));
    assert!(!on_family("plan9"));
}

#[test]
#[should_panic(expected = "names platform `darwin`")]
fn families_refuse_an_unknown_one() {
    parse_known_failures("official/a.lua 1111111 darwin\n");
}

#[test]
fn message_is_the_uncaught_errors() {
    let stderr = "sh: nothing: command not found\n\
                  lua5.4: files.lua:762: assertion failed!\n\
                  stack traceback:\n\
                  \t[C]: in function 'assert'\n";
    assert_eq!(message_of(stderr), "files.lua:762: assertion failed!");
    assert_eq!(message_of("lua: boom\n"), "boom");
    assert_eq!(message_of(""), "");
}

#[test]
fn last_line_skips_the_blank_tail() {
    // gc.lua's finalizer prints this with an extra newline at close.
    let out = b"final OK !!!\n>>> closing state <<<\n\n";
    assert_eq!(last_line(out), b">>> closing state <<<");
    assert!(has_line(out, b"final OK !!!"));
    assert_eq!(last_line(b"OK\n  \n\t\n"), b"OK");
    assert_eq!(last_line(b""), b"");
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

/// The gc category once more with the collector stressed.
#[test]
fn gc_stressed() {
    category("gc", WarmUp::Stressed);
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
    debug,
    gc,
    capi,
    official,
}
