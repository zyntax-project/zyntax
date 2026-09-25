//! Memory under pressure: a program's peak memory must not grow with
//! how long it runs.
//!
//! Each program under `pressure/` allocates something on every step of
//! a loop and keeps only a scalar, so everything it makes is garbage by
//! the next step. It takes the step count on the command line and is
//! run twice, at a small count and at one many times larger; the
//! difference in peak resident memory between the two is what the run
//! failed to release. A bounded program grows by a slab or two; a leak
//! grows by the count.
//!
//! `pressure/KNOWN_LEAKS` lists the programs known to grow, each with
//! the git-bug issue that tracks why. A known leak that stops growing
//! is reported too, so the list is kept honest.
//!
//! The collector's heap floor is lowered for the runs, so what it
//! keeps is bounded by the live set rather than by the floor and the
//! difference between the two runs measures leaks alone; it also makes
//! the collector run often, which is what finds a root it misses.
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
const LARGE: u64 = 400_000;
/// Steps for the two runs with the LLVM tier on. Its first install
/// brings in LLVM's code and state, a fixed footprint larger than the
/// allowance, on a background thread some time after a function turns
/// hot; the short run is long enough that every promotion either run
/// makes has landed in both.
const SMALL_LLVM: u64 = 100_000;
const LARGE_LLVM: u64 = 1_000_000;
/// Growth a bounded program is allowed between the two: the slabs its
/// allocator takes, the code the long run compiles that the short one
/// interprets, and the bodies kept for later tiers, all of which stop
/// growing once the program is warm. A leak of one small block a step
/// is twice this over the long run.
const ALLOWED_GROWTH: u64 = 5 << 20;
/// The collector's heap floor for the runs, in KB.
const HEAP_FLOOR_KB: u64 = 512;

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
        .env("ZYNTAX_GC_FLOOR_KB", HEAP_FLOOR_KB.to_string())
        // The long run promotes what the short one leaves at the
        // baseline, and each tier's code is memory the program's heap
        // is not; the tiers stay where they start.
        .env("ZYNTAX_DISABLE_OSR", "1")
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

    // zypy tiers up to LLVM only when built with it and asked to.
    let llvm = cfg!(feature = "llvm-backend") && std::env::var_os("ZYPY_LLVM").is_some();
    let (small_steps, large_steps) = if llvm {
        (SMALL_LLVM, LARGE_LLVM)
    } else {
        (SMALL, LARGE)
    };

    let mut failures = Vec::new();
    let mut known_count = 0;
    for program in &programs {
        let name = program
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let (small_status, small) = run(program, small_steps);
        let (large_status, large) = run(program, large_steps);
        if small_status != 0 || large_status != 0 {
            failures.push(format!(
                "{name}: exited {small_status} at {small_steps} steps and {large_status} at {large_steps}"
            ));
            continue;
        }
        let growth = large.saturating_sub(small);
        let leaks = growth > ALLOWED_GROWTH;
        let per_step = growth / (large_steps - small_steps);
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
                "{name}: peak memory grew from {} MB to {} MB between {small_steps} and {large_steps} steps, about {per_step} bytes a step",
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
