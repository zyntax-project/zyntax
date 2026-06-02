//! Benchmark suite runner for ZynML.
//!
//! Measures each `bench_*.zynml` source under
//! `crates/zynml/examples/` across the targets we control today:
//!
//!   * `zyntax-interp`     — BC interpreter, NO HIR optimization
//!                            pipeline. Floor for what the
//!                            interpreter can do on its own.
//!   * `zyntax-interp-opt` — BC interpreter after
//!                            `run_interp_safe_opts` has finished
//!                            its fixed-point sweep (const_fold,
//!                            cse, load_cse, inline, licm,
//!                            loop_vectorize, reduction_vectorize,
//!                            cfg_simplify, alloca_promote,
//!                            drop_insert).
//!
//! Each measurement separates compile-time from execute-time, runs
//! `WARMUP` warmup iterations followed by `RUNS` timed iterations,
//! and keeps the median wall-clock time. Output is a JSON file at
//! `website/benchmark/results.json` consumed by the static page at
//! `website/benchmark/index.html`.
//!
//! Usage:
//!     cargo run --release --package zynml --example bench_runner
//!     cargo run --release --package zynml --example bench_runner -- --out /tmp/bench.json
//!     cargo run --release --package zynml --example bench_runner -- --runs 5
//!
//! Sibling-project precedent: the layout (per-kernel JSON, page
//! reads JSON and renders bars) mirrors `wren_lift/site/benchmark/`
//! and `rayzor/compiler/benchmarks/`. Kept deliberately small —
//! adds external-language targets (python3, node, …) as a
//! follow-up once the comparison story is worth the CI complexity.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::profiling::ProfileConfig;
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_compiler::{run_interp_safe_opts, HirModule};
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};

const WARMUP: usize = 3;
const RUNS: usize = 9;

/// Extra in-loop warmup calls reserved for JIT targets — drives
/// beadie's `TieredAdapter` past the warm-threshold so the
/// background Cranelift compile finishes before measurement
/// starts. Uses a low warm-threshold ([`jit_tier_config`]), so 16
/// calls is comfortably above the trigger point.
const JIT_TIER_WARMUP_CALLS: usize = 16;

/// One (kernel, target) measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TargetResult {
    /// Median wall-clock time of `RUNS` measurement iterations,
    /// expressed in seconds — the unit the static page renders.
    /// Zero when [`Self::error`] is set.
    seconds: f64,
    /// Median compile-only time (parse + lower + opt). Useful for
    /// pulling apart compile cost from execute cost in charts.
    compile_ms: f64,
    /// Median execute-only time (call `main`, no setup). The page
    /// can sum `compile_ms + exec_ms` if it wants total cost.
    exec_ms: f64,
    /// The value `main` returned, formatted via `Debug`. Tracking
    /// it pins correctness across runs — a future opt pass that
    /// silently changes the bench result fails the workflow loudly.
    /// `"—"` when [`Self::error`] is set.
    result: String,
    /// When the kernel fails to compile or execute on this target,
    /// the error message goes here and the timings are zeroed. The
    /// page renders these as a red "FAILED" badge instead of a
    /// chart bar — the suite is explicit about which tier each
    /// kernel breaks on, rather than silently dropping the row.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Per-kernel collection of target results.
type KernelResults = BTreeMap<String, TargetResult>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Suite {
    /// Per-kernel measurements, keyed by kernel name. Each kernel
    /// maps to a target-name → result table so the static page can
    /// iterate without knowing target names ahead of time.
    #[serde(flatten)]
    kernels: BTreeMap<String, KernelResults>,
    /// Run metadata: when, where, with what commit. The page reads
    /// these into the "updated · commit · arch" line.
    meta: Meta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Meta {
    /// ISO-8601 UTC timestamp set at runner start. Hand-formatted
    /// because the runner doesn't depend on `chrono` and stdlib
    /// only knows about SystemTime.
    date: String,
    /// Short commit SHA — pulled via `git rev-parse --short HEAD`,
    /// or "—" if git isn't available.
    commit: String,
    /// `std::env::consts::ARCH` (e.g. "aarch64", "x86_64").
    arch: String,
    /// `std::env::consts::OS` (e.g. "macos", "linux").
    os: String,
    /// Always equal to `RUNS` — kept in JSON so the page can
    /// render "median of N" accurately if we ever change the
    /// constant.
    runs: usize,
}

/// Each benchmark source lives at
/// `crates/zynml/examples/<name>.zynml` and gets run across every
/// target listed in [`TARGETS`].
const KERNELS: &[&str] = &["bench_mandelbrot", "bench_nbody", "bench_fib"];

/// Each target produces one [`TargetResult`] per kernel.
const TARGETS: &[Target] = &[
    Target {
        key: "zyntax-interp",
        label: "Zyntax · BC interp",
        run_with_opts: false,
        install_jit: false,
    },
    Target {
        key: "zyntax-interp-opt",
        label: "Zyntax · BC interp + opt",
        run_with_opts: true,
        install_jit: false,
    },
    Target {
        key: "zyntax-tiered",
        label: "Zyntax · BC interp → Cranelift tier-up",
        run_with_opts: true,
        install_jit: true,
    },
];

#[derive(Debug, Clone, Copy)]
struct Target {
    /// Stable key used in the JSON output (snake-case ASCII).
    key: &'static str,
    /// Human-readable label for charts. Currently unused by the
    /// page (it renders the JSON keys directly) but kept here so a
    /// future page can pick it up without re-running the bench.
    #[allow(dead_code)]
    label: &'static str,
    /// Whether to apply [`run_interp_safe_opts`] before installing
    /// the module. The two-target layout (`run_with_opts: false`
    /// vs `true`) gives the page a direct opt-pipeline payoff
    /// comparison per kernel.
    run_with_opts: bool,
    /// Install the BC interp → Cranelift JIT tier ladder for this
    /// target. Setup goes into `compile_ms`; warmup drives the
    /// async compile to completion before timed iterations start
    /// dispatching through the JIT'd code.
    install_jit: bool,
}

/// Custom `TieredConfig` for the JIT target — warm-threshold of 1
/// so the very first interpreter tick schedules the Cranelift
/// compile. Without this, the default warm-threshold (100) would
/// require hundreds of warmup iterations to trigger tier-up; we
/// can't reasonably afford that per timed measurement.
fn jit_tier_config() -> TieredConfig {
    let mut cfg = TieredConfig::default();
    cfg.profile_config = ProfileConfig {
        warm_threshold: 1,
        hot_threshold: 5,
        ..ProfileConfig::default()
    };
    cfg
}

fn main() {
    // Parse command-line flags.
    let mut out_path: Option<PathBuf> = None;
    let mut runs_override: Option<usize> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                out_path = args.next().map(PathBuf::from);
            }
            "--runs" => {
                runs_override = args.next().and_then(|s| s.parse().ok());
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: bench_runner [--out <path>] [--runs <n>]\n\
                     Defaults: out = website/benchmark/results.json, runs = {RUNS}"
                );
                return;
            }
            other => {
                eprintln!("warning: ignoring unknown arg {other:?}");
            }
        }
    }
    let runs = runs_override.unwrap_or(RUNS);
    let default_out =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../website/benchmark/results.json");
    let out_path = out_path.unwrap_or(default_out);

    let mut suite = Suite {
        kernels: BTreeMap::new(),
        meta: Meta {
            date: rfc3339_now(),
            commit: git_short_sha(),
            arch: env::consts::ARCH.to_string(),
            os: env::consts::OS.to_string(),
            runs,
        },
    };

    for kernel in KERNELS {
        let pretty = kernel.strip_prefix("bench_").unwrap_or(kernel);
        let source_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("examples/{kernel}.zynml"));
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|e| panic!("read {source_path:?}: {e}"));

        eprintln!("==> kernel {pretty}");
        let mut per_kernel: KernelResults = BTreeMap::new();
        for target in TARGETS {
            let r = measure(&source, target, runs);
            if let Some(err) = r.error.as_ref() {
                eprintln!("    {:<22} FAILED — {err}", target.key);
            } else {
                eprintln!(
                    "    {:<22} compile={:>7.2}ms exec={:>9.2}ms total={:>9.2}ms  -> {}",
                    target.key,
                    r.compile_ms,
                    r.exec_ms,
                    r.seconds * 1000.0,
                    r.result,
                );
            }
            per_kernel.insert(target.key.to_string(), r);
        }
        suite.kernels.insert(pretty.to_string(), per_kernel);
    }

    // Make sure the output directory exists. `mkdir -p` shape so
    // a fresh checkout can write into `website/benchmark/` without
    // anyone hand-creating it.
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|e| panic!("mkdir {parent:?}: {e}"));
    }
    let json = serde_json::to_string_pretty(&suite).expect("serialize results");
    fs::write(&out_path, json).unwrap_or_else(|e| panic!("write {out_path:?}: {e}"));
    eprintln!("\nwrote {}", out_path.display());
}

/// Parse + lower + (optionally) opt + run, repeated until we have
/// `runs` timed iterations. Returns medians. If any iteration
/// fails (parse error, compile error, runtime panic, etc.) the
/// returned result has its `error` field populated and all
/// timings zeroed — the suite is honest about which tier broke,
/// rather than silently dropping the row.
fn measure(source: &str, target: &Target, runs: usize) -> TargetResult {
    let mut compile_samples = Vec::with_capacity(runs);
    let mut exec_samples = Vec::with_capacity(runs);
    let mut last_result_str = String::new();

    // Warmup — toss the times, but the runs still mutate caches /
    // allocators / OS-page state so timed iters run on a steady
    // baseline. A failure during warmup short-circuits straight
    // to the `error` shape; no point spending the runs-loop's
    // budget on something that's reliably broken.
    for _ in 0..WARMUP {
        if let Err(e) = one_iteration(source, target) {
            return failed_result(&e);
        }
    }
    for _ in 0..runs {
        match one_iteration(source, target) {
            Ok((compile_ms, exec_ms, r)) => {
                compile_samples.push(compile_ms);
                exec_samples.push(exec_ms);
                last_result_str = format!("{r:?}");
            }
            Err(e) => return failed_result(&e),
        }
    }

    let median_compile = median(&mut compile_samples);
    let median_exec = median(&mut exec_samples);
    TargetResult {
        seconds: (median_compile + median_exec) / 1000.0,
        compile_ms: median_compile,
        exec_ms: median_exec,
        result: last_result_str,
        error: None,
    }
}

fn failed_result(error: &str) -> TargetResult {
    TargetResult {
        seconds: 0.0,
        compile_ms: 0.0,
        exec_ms: 0.0,
        result: "—".to_string(),
        error: Some(error.to_string()),
    }
}

/// One full kernel run: lower from source, optionally apply the
/// HIR opt pipeline, install into a fresh runtime, drive the JIT
/// tier-up if requested, then call `main`. Returns
/// `(compile_ms, exec_ms, result)`.
///
/// `compile_ms` is always the **cold-path setup**: parse + lower
/// + optional HIR opts + `compile_module` + (for the tiered
/// target) `install_interp_jit_with`. Tiered mode cold-starts
/// with the BC interpreter, so its compile cost should look
/// almost identical to the interp targets — the only delta is
/// the tier-ladder install itself, which is cheap. The expensive
/// part (the asynchronous Cranelift compile that beadie's
/// `TieredAdapter` schedules) happens during the *warmup* calls
/// below, NOT during `compile_ms`.
///
/// `exec_ms` is the single timed `main()` call. For the tiered
/// target, warmup runs untimed before the measurement so that
/// call dispatches through the JIT'd code rather than the cold
/// BC interp loop. For non-JIT targets there's no warmup — the
/// timed call IS the cold first call.
fn one_iteration(source: &str, target: &Target) -> Result<(f64, f64, ZyntaxValue), String> {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).map_err(|e| format!("grammar: {e:?}"))?;

    let compile_start = Instant::now();
    let program = grammar
        .parse_with_filename(source, "<bench>")
        .map_err(|e| format!("parse: {e:?}"))?;
    let rt = ZyntaxRuntime::new().map_err(|e| format!("rt: {e:?}"))?;
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut module: HirModule = rt
        .lower_typed_program(program, builtins)
        .map_err(|e| format!("lower: {e:?}"))?;
    if target.run_with_opts {
        let _ = run_interp_safe_opts(&mut module);
    }
    let mut rt = ZyntaxRuntime::new().map_err(|e| format!("rt: {e:?}"))?;
    rt.compile_module(&module)
        .map_err(|e| format!("compile_module: {e:?}"))?;
    if target.install_jit {
        rt.install_interp_jit_with(jit_tier_config())
            .map_err(|e| format!("install_interp_jit: {e:?}"))?;
    }
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;

    if target.install_jit {
        // Warm beadie's `TieredAdapter` past the threshold so the
        // background Cranelift compile finishes and subsequent
        // interp ticks dispatch to the JIT'd code. Run *untimed*
        // — this isn't compile cost (the compile already
        // happened above, only the install was synchronous), and
        // it isn't part of the steady-state exec we want to
        // measure either. The cost of getting to steady state is
        // a separate axis the page doesn't currently report.
        for _ in 0..JIT_TIER_WARMUP_CALLS {
            rt.call_function_raw("main", vec![])
                .map_err(|e| format!("jit warmup: {e:?}"))?;
        }
    }

    // Wrap the timed call in `catch_unwind` so a panic inside the
    // Cranelift backend (a known failure mode for some HIR shapes)
    // surfaces as an `error` row on the bench page rather than
    // aborting the whole suite.
    let exec_start = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rt.call_function_raw("main", vec![])
    }))
    .map_err(|p| {
        if let Some(s) = p.downcast_ref::<&str>() {
            format!("panic: {s}")
        } else if let Some(s) = p.downcast_ref::<String>() {
            format!("panic: {s}")
        } else {
            "panic: <opaque>".to_string()
        }
    })?
    .map_err(|e| format!("call: {e:?}"))?;
    let exec_ms = exec_start.elapsed().as_secs_f64() * 1000.0;

    Ok((compile_ms, exec_ms, result))
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    }
}

/// RFC-3339 / ISO-8601 UTC timestamp at runner start. Hand-rolled
/// from `SystemTime` to avoid pulling in `chrono` for one
/// timestamp.
fn rfc3339_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days since 1970-01-01.
    let mut days = secs / 86_400;
    let mut secs_in_day = secs % 86_400;
    let hour = secs_in_day / 3600;
    secs_in_day %= 3600;
    let minute = secs_in_day / 60;
    let second = secs_in_day % 60;
    // Year/month/day from days-since-epoch via the standard
    // civil-from-days algorithm (Howard Hinnant). Compact + exact
    // through year 32767.
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    let _ = &mut days;
    format!(
        "{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z",
        y = y as i32,
        m = m,
        d = d,
        hour = hour,
        minute = minute,
        second = second
    )
}

fn git_short_sha() -> String {
    use std::process::Command;
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "—".to_string())
}
