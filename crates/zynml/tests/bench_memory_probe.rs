//! Which benchmark kernel accounts for the memory the suite uses.
//!
//! Peak RSS for a full run is measured in gigabytes, which is far more
//! than any kernel's buffers. This loads each kernel the way the suite
//! does, one at a time, and reports resident size after each, so the
//! growth can be attributed to a kernel rather than to the harness.

use std::path::{Path, PathBuf};
use zynml::ZynML;

/// Resident size of this process, in MB.
fn rss_mb() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .unwrap_or(0)
        / 1024
}

fn kernels() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "zynml"))
        .collect();
    v.sort();
    v
}

/// A second pass over the same kernels must not cost what the first
/// did. Whatever the first pass paid for compilation and allocator
/// arenas is already paid, so growth on the second is storage a program
/// actually kept.
///
/// Before the drop-site work, one pass grew 219 MB, nearly all of it
/// programs that allocated per iteration and released nothing. The
/// bound is not measuring the residue, it is there to fail loudly if a
/// per-iteration leak comes back.
///
/// Stated as a fraction of what the first pass cost rather than as a
/// number of megabytes. A leak makes the second pass cost what the
/// first did, so the quantity being watched is a ratio; an absolute
/// bound measures instead how much this particular set of kernels
/// allocates, and moves whenever one is added.
///
/// The ratio also has to survive the difference between machines: the
/// same tree grows 2% here and 34% on the Linux runner, so the bound is
/// half rather than something tighter.
#[test]
fn a_second_pass_does_not_cost_what_the_first_did() {
    let files = kernels();
    let before_first = rss_mb();
    for f in &files {
        if let Ok(src) = std::fs::read_to_string(f) {
            if let Ok(mut rt) = ZynML::new() {
                if rt.load_source(&src).is_ok() {
                    let _ = rt.call_with_result::<i64>("main");
                }
            }
        }
    }
    let after_first = rss_mb();
    for f in &files {
        if let Ok(src) = std::fs::read_to_string(f) {
            if let Ok(mut rt) = ZynML::new() {
                if rt.load_source(&src).is_ok() {
                    let _ = rt.call_with_result::<i64>("main");
                }
            }
        }
    }
    let after_second = rss_mb();
    println!("\n  after pass 1: {after_first} MB");
    println!("  after pass 2: {after_second} MB");
    let growth = after_second as i64 - after_first as i64;
    println!("  second-pass growth: {growth} MB");

    // Judged against what the first pass cost, not against a fixed
    // number of megabytes. A leak that returns makes the second pass
    // cost what the first did, which is the thing being watched for and
    // is a ratio; an absolute bound instead tracks how much the suite
    // happens to allocate, so adding a kernel moves it. `binary_trees`
    // allocates thirty million nodes on purpose and moved it enough to
    // fail on CI at 60 MB against a 60 MB bound while passing locally
    // at -7 MB.
    // Half, not a smaller fraction: a returning leak makes the second
    // pass cost essentially what the first did, so anything well under
    // that catches it, and the headroom is what stops an environment
    // difference from reading as one. This machine sees 2% and the
    // Linux runner 34%, for the same tree.
    let first_growth = (after_first as i64 - before_first as i64).max(1);
    println!("  first-pass growth: {first_growth} MB");
    assert!(
        growth * 2 < first_growth,
        "a second pass grew {growth} MB against the first pass's \
         {first_growth} MB. A pass that costs anything like the one \
         before it is the shape of a program allocating per iteration \
         and releasing nothing."
    );

    // Attribute what is left, now that first-time costs are paid.
    println!("\n  third pass, per kernel:");
    let mut prev = rss_mb();
    for f in &files {
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        if let Ok(src) = std::fs::read_to_string(f) {
            if let Ok(mut rt) = ZynML::new() {
                if rt.load_source(&src).is_ok() {
                    let _ = rt.call_with_result::<i64>("main");
                }
            }
        }
        let now = rss_mb();
        let d = now as i64 - prev as i64;
        if d != 0 {
            println!("    {name:<34}{d:>6} MB");
        }
        prev = now;
    }
}

#[test]
fn report_memory_per_kernel() {
    let base = rss_mb();
    println!("\n  {:<34}{:>10}{:>10}", "kernel", "after MB", "delta MB");
    println!("  {}", "-".repeat(54));
    let mut prev = base;
    let mut rows: Vec<(String, u64)> = Vec::new();
    for f in kernels() {
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        let src = match std::fs::read_to_string(&f) {
            Ok(s) => s,
            Err(_) => continue,
        };
        // Load and run once, then drop, exactly as one harness
        // iteration does.
        {
            let mut rt = match ZynML::new() {
                Ok(r) => r,
                Err(_) => continue,
            };
            if rt.load_source(&src).is_err() {
                continue;
            }
            let _ = rt.call_with_result::<i64>("main");
        }
        let now = rss_mb();
        println!("  {:<34}{:>10}{:>10}", name, now, now as i64 - prev as i64);
        rows.push((name, now.saturating_sub(prev)));
        prev = now;
    }
    println!("  {}", "-".repeat(54));
    println!(
        "  start {base} MB, end {prev} MB, growth {} MB",
        prev - base
    );
    rows.sort_by_key(|(_, d)| std::cmp::Reverse(*d));
    println!("\n  largest contributors:");
    for (name, d) in rows.iter().take(5) {
        println!("    {name:<32}{d:>8} MB");
    }
}
