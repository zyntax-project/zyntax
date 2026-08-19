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
