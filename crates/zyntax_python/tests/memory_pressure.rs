//! Memory under pressure: a program's peak memory must not grow with
//! how long it runs.
//!
//! Each program under `pressure/` allocates something on every step of
//! a loop and keeps only a scalar, so everything it makes is garbage by
//! the next step. It takes the step count on the command line and is
//! run twice, at a small count and at one a hundred times larger; the
//! difference in peak resident memory between the two is what the run
//! failed to release. A bounded program grows by a slab or two; a leak
//! grows by the count.
//!
//! `pressure/KNOWN_LEAKS` lists the programs known to grow, each with
//! the git-bug issue that tracks why. A known leak that stops growing
//! is reported too, so the list is kept honest.
//!
//! Peak memory is read from the child's `rusage`, which is Unix only;
//! elsewhere the test has nothing to measure and passes.

#![cfg(unix)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Steps for the short and the long run.
const SMALL: u64 = 2_000;
const LARGE: u64 = 200_000;
/// Growth a bounded program is allowed between the two, for the slabs
/// its allocator takes and the noise of the process.
const ALLOWED_GROWTH: u64 = 3 << 20;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("pressure")
}

fn known_leaks() -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Ok(text) = fs::read_to_string(root().join("KNOWN_LEAKS")) else {
        return out;
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        if let (Some(file), Some(issue)) = (parts.next(), parts.next()) {
            out.insert(file.to_string(), issue.to_string());
        }
    }
    out
}

/// Run `program` with `steps` and report its exit status and peak
/// resident memory in bytes.
///
/// The child is waited for through `wait4`, which is where the peak
/// memory comes from; `Child::wait` does not report it.
#[allow(clippy::zombie_processes)]
fn run(program: &Path, steps: u64) -> (i32, u64) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(program)
        .arg(steps.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("zypy starts");
    // Drained so a chatty program cannot block on a full pipe.
    let drain = child.stdout.take().map(|mut s| {
        std::thread::spawn(move || {
            let mut out = Vec::new();
            let _ = std::io::Read::read_to_end(&mut s, &mut out);
        })
    });
    let pid = child.id() as libc::pid_t;
    let mut status: libc::c_int = 0;
    // SAFETY: `usage` is a plain C struct the call fills in; `pid` is a
    // child of this process that has not been waited for.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let waited = unsafe { libc::wait4(pid, &mut status, 0, &mut usage) };
    assert_eq!(waited, pid, "wait4 on the child");
    if let Some(d) = drain {
        let _ = d.join();
    }
    let code = if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else {
        -1
    };
    // Bytes on macOS, kilobytes elsewhere.
    let unit: u64 = if cfg!(target_os = "macos") { 1 } else { 1024 };
    (code, usage.ru_maxrss as u64 * unit)
}

#[test]
fn peak_memory_does_not_grow_with_the_step_count() {
    let known = known_leaks();
    let mut programs: Vec<PathBuf> = fs::read_dir(root())
        .expect("pressure directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "py"))
        .collect();
    programs.sort();

    let mut failures = Vec::new();
    let mut known_count = 0;
    for program in &programs {
        let name = program
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let (small_status, small) = run(program, SMALL);
        let (large_status, large) = run(program, LARGE);
        if small_status != 0 || large_status != 0 {
            failures.push(format!(
                "{name}: exited {small_status} at {SMALL} steps and {large_status} at {LARGE}"
            ));
            continue;
        }
        let growth = large.saturating_sub(small);
        let leaks = growth > ALLOWED_GROWTH;
        let per_step = growth / (LARGE - SMALL);
        match (leaks, known.get(&name)) {
            (true, Some(issue)) => {
                known_count += 1;
                eprintln!(
                    "  {name}: known leak, {} MB to {} MB, about {per_step} bytes a step ({issue})",
                    small >> 20,
                    large >> 20
                );
            }
            (true, None) => failures.push(format!(
                "{name}: peak memory grew from {} MB to {} MB between {SMALL} and {LARGE} steps, about {per_step} bytes a step",
                small >> 20,
                large >> 20
            )),
            (false, Some(issue)) => failures.push(format!(
                "{name}: listed in KNOWN_LEAKS ({issue}) but grew only {} bytes; take it off the list",
                growth
            )),
            (false, None) => eprintln!(
                "  {name}: {} MB to {} MB",
                small >> 20,
                large >> 20
            ),
        }
    }
    eprintln!(
        "pressure: {} programs, {} known leaks, {} failures",
        programs.len(),
        known_count,
        failures.len()
    );
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
