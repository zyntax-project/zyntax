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
use std::sync::OnceLock;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_TENSOR};
use zyntax_compiler::bytecode::{deserialize_module, serialize_module, Format};
use zyntax_compiler::profiling::ProfileConfig;
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_compiler::HirModule;
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};

/// Bumped manually when the compiler's HIR schema changes (new variants,
/// field renames, layout shifts in `HirModule` / `HirFunction` / …).
/// Mixed into the cache key so stale `.zbc` snapshots from an older
/// schema cannot collide with a freshly produced module — a mismatch
/// just looks like a miss and we recompile.
///
/// `crc32fast` already protects on-disk corruption, and postcard's
/// schema mismatch is loud, but neither catches a *valid* old payload
/// being deserialized into a subtly-incompatible new struct. The
/// version byte makes that case impossible.
const CACHE_SCHEMA_VERSION: u32 = 1;

// One bench iteration = a fresh `lower + compile + install_jit +
// JIT_TIER_WARMUP_CALLS calls + 1 timed call`. At rayzor-scale that
// per-iteration cost is large (mandelbrot ~5 s, nbody ~80 s end-to-end)
// because the install pre-compiles the whole prelude-augmented HIR
// module each time. Keep the outer-loop counts tight; median-of-N
// numbers stop being informative when each sample is its own minute.
// Override at the command line with `--runs N`.
const WARMUP: usize = 0;
const RUNS: usize = 3;

/// Extra in-loop warmup calls reserved for JIT targets — drives
/// beadie's `TieredAdapter` past the warm-threshold so the
/// background Cranelift compile finishes before measurement
/// starts. Uses a low warm-threshold ([`jit_tier_config`]), so 16
/// calls is comfortably above the trigger point.
// Warm enough to drive the bead past the warm threshold (1 in
// `jit_tier_config`) and let the background Cranelift compile
// finalise. Four calls covers both — going higher used to be a
// safety margin when the threshold was higher, but at warm=1 it
// just pays the BC-interp cost for the heavy kernels (rayzor-scale
// mandelbrot is ~720 ms per call, nbody is ~16 s) before the timed
// iteration even starts.
const JIT_TIER_WARMUP_CALLS: usize = 4;

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
    /// True when the target's `skip_kernels` list opted out of this
    /// kernel (e.g. mandelbrot on the BC-interp-only tiers, which
    /// take 30+ minutes per iteration). Lets the page render
    /// "skipped" instead of "FAILED" — the kernel is not broken on
    /// this tier, the harness just declined to run it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    #[serde(default)]
    skipped: bool,
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
        install_llvm: false,
        // Mandelbrot at rayzor's 875 × 500 / max_iter 1000 spends
        // 30 + minutes in pure BC interp on a desktop CPU and
        // nbody at rayzor's 20 × 500 000 iterations of advance() +
        // Newton-iter sqrt is similarly heavy. The interp/opt
        // tiers don't tell us anything the JIT tier doesn't tell
        // us better at full kernel scale, so skip for those two.
        skip_kernels: &["mandelbrot", "nbody"],
    },
    Target {
        key: "zyntax-interp-opt",
        label: "Zyntax · BC interp + opt",
        run_with_opts: true,
        install_jit: false,
        install_llvm: false,
        skip_kernels: &["mandelbrot", "nbody"],
    },
    Target {
        key: "zyntax-tiered",
        label: "Zyntax · BC interp → Cranelift tier-up",
        run_with_opts: true,
        install_jit: true,
        install_llvm: false,
        // n-body main hits a Cranelift `UnreachableCodeReached`
        // trap when JIT-dispatched (visible through both the
        // bench's InterpRuntime path and ZynML's TieredRuntime
        // CLI). Sub-functions (advance, energy, sqrt) JIT cleanly
        // — the bug is specifically in JIT'd main's control
        // flow lowering for nested-while bodies. Tracked as a
        // follow-up; skip the tiered row for now so the rest of
        // the bench still produces useful numbers.
        skip_kernels: &[],
    },
    // Full ladder: BC interp → Cranelift (tier 0) → LLVM (tier 1).
    // Compiled in only when the `llvm-backend` cargo feature is on;
    // otherwise the target's tick callback never escalates past
    // Cranelift and `measure` records the same numbers as
    // `zyntax-tiered`. Build with
    // `cargo run --release --features llvm-backend …` to exercise it.
    Target {
        key: "zyntax-tiered-llvm",
        label: "Zyntax · BC interp → Cranelift → LLVM full tier",
        run_with_opts: true,
        install_jit: true,
        install_llvm: true,
        // Same Cranelift codegen issue as `zyntax-tiered` —
        // LLVM tier sits on top of the Cranelift tier so n-body
        // main hits the same trap before the LLVM compile ever
        // gets a chance to fire.
        skip_kernels: &[],
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
    /// Drive the ladder one tier higher — Cranelift → LLVM. Only
    /// meaningful when `install_jit` is true and the binary was
    /// built with the `llvm-backend` cargo feature; otherwise it
    /// has no effect (the install path is gated on the same cfg).
    install_llvm: bool,
    /// Kernel pretty-names (the `bench_` prefix stripped) that this
    /// target should skip — recorded in the JSON as a
    /// `skipped: true` row so the page can render "skipped" instead
    /// of zero ms. Used to keep heavy kernels (mandelbrot) out of
    /// the BC-interp-only tiers where they would dominate runtime
    /// without telling us anything useful.
    skip_kernels: &'static [&'static str],
}

/// Build a fresh `ZyntaxRuntime` with the prelude + tensor stdlib
/// import resolvers and the ZynML grammar registered, so
/// `import prelude` in bench kernels actually pulls in the stdlib
/// `abs` / `min` / `max` / List / Option / Result definitions.
/// `lower_typed_program` walks imports through the registered
/// grammar, so without `register_grammar` the resolver is loaded
/// but never invoked.
// `LanguageGrammar::compile_zyn` parses + lowers the entire .zyn
// grammar source (rule AST, action AST, Grammar2 internal IR).
// That's ~600 ms on a desktop CPU — fine when ZynML's CLI does it
// once at startup, ruinous when the bench harness runs
// `bench_runtime()` twice per kernel iteration (once for the
// lowering rt, once for the compile/runtime rt). Cache the result.
static SHARED_LANG_GRAMMAR: OnceLock<Result<zyntax_embed::LanguageGrammar, String>> =
    OnceLock::new();

fn bench_runtime() -> Result<ZyntaxRuntime, String> {
    let mut rt = ZyntaxRuntime::new().map_err(|e| format!("rt: {e:?}"))?;
    rt.add_import_resolver(Box::new(|module_name| match module_name {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "tensor" => Ok(Some(ZYNML_STDLIB_TENSOR.to_string())),
        _ => Ok(None),
    }));
    let lang_grammar = SHARED_LANG_GRAMMAR
        .get_or_init(|| {
            use zyntax_embed::LanguageGrammar;
            LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).map_err(|e| format!("compile_zyn: {e:?}"))
        })
        .clone()?;
    rt.register_grammar("zynml", lang_grammar);
    Ok(rt)
}

/// Custom `TieredConfig` for the JIT target — warm-threshold of 1
/// so the very first interpreter tick schedules the Cranelift
/// compile. Without this, the default warm-threshold (100) would
/// require hundreds of warmup iterations to trigger tier-up; we
/// can't reasonably afford that per timed measurement.
///
/// `install_llvm = false` parks the hot threshold at `u32::MAX`
/// so the LLVM tier never fires even when the `llvm-backend` cargo
/// feature is compiled in — that target stays pure Cranelift.
/// `install_llvm = true` lowers it to 5 so LLVM kicks in after a
/// handful of warmup calls (still only effective when the cargo
/// feature is enabled; otherwise the install path is a no-op).
fn jit_tier_config(install_llvm: bool) -> TieredConfig {
    let mut cfg = TieredConfig::default();
    cfg.profile_config = ProfileConfig {
        // `warm_threshold = 0` fires the Cranelift dispatch on the
        // very first invocation. Without it, beadie's `on_invoke`
        // returns `None` for the first call regardless of the
        // pre-compiled function pointer being ready, and the BC
        // interp runs the whole entry function — at rayzor-scale
        // mandelbrot that's 30 + minutes per warmup iteration.
        warm_threshold: 0,
        hot_threshold: if install_llvm { 1 } else { u32::MAX as u64 },
        ..ProfileConfig::default()
    };
    cfg
}

fn main() {
    // Parse command-line flags.
    let mut out_path: Option<PathBuf> = None;
    let mut runs_override: Option<usize> = None;
    let mut cache_enabled = true;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                out_path = args.next().map(PathBuf::from);
            }
            "--runs" => {
                runs_override = args.next().and_then(|s| s.parse().ok());
            }
            "--no-cache" => {
                cache_enabled = false;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: bench_runner [--out <path>] [--runs <n>] [--no-cache]\n\
                     Defaults: out = website/benchmark/results.json, runs = {RUNS}, cache = on"
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
            if target.skip_kernels.contains(&pretty) {
                eprintln!("    {:<22} SKIPPED (per-target opt-out)", target.key);
                per_kernel.insert(target.key.to_string(), skipped_result());
                continue;
            }
            let r = measure(&source, target, runs, pretty, cache_enabled);
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
fn measure(
    source: &str,
    target: &Target,
    runs: usize,
    kernel: &str,
    cache_enabled: bool,
) -> TargetResult {
    let mut compile_samples = Vec::with_capacity(runs);
    let mut exec_samples = Vec::with_capacity(runs);
    let mut last_result_str = String::new();

    // Warmup — toss the times, but the runs still mutate caches /
    // allocators / OS-page state so timed iters run on a steady
    // baseline. A failure during warmup short-circuits straight
    // to the `error` shape; no point spending the runs-loop's
    // budget on something that's reliably broken.
    for _ in 0..WARMUP {
        if let Err(e) = one_iteration(source, target, kernel, cache_enabled) {
            return failed_result(&e);
        }
    }
    for _ in 0..runs {
        match one_iteration(source, target, kernel, cache_enabled) {
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
        skipped: false,
    }
}

fn failed_result(error: &str) -> TargetResult {
    TargetResult {
        seconds: 0.0,
        compile_ms: 0.0,
        exec_ms: 0.0,
        result: "—".to_string(),
        error: Some(error.to_string()),
        skipped: false,
    }
}

fn skipped_result() -> TargetResult {
    TargetResult {
        seconds: 0.0,
        compile_ms: 0.0,
        exec_ms: 0.0,
        result: "—".to_string(),
        error: None,
        skipped: true,
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
fn one_iteration(
    source: &str,
    target: &Target,
    kernel: &str,
    cache_enabled: bool,
) -> Result<(f64, f64, ZyntaxValue), String> {
    // Fine-grained instrumentation, enabled only when the
    // `ZYNTAX_BENCH_TRACE_COMPILE` env var is set. The trace
    // breaks `compile_ms` down into the individual pipeline
    // phases so we can see which one to attack first when chasing
    // sub-30 ms cold-start compile. Off by default — the env-var
    // gate keeps normal bench runs clean.
    let trace = env::var_os("ZYNTAX_BENCH_TRACE_COMPILE").is_some();

    let compile_start = Instant::now();

    // ----- ZBC cache lookup ------------------------------------------------
    // The cache key folds in every input that materially affects the
    // produced `HirModule`: the source itself, both stdlib files
    // (prelude, tensor), whether the opt pipeline will run, and a
    // schema-version constant so a bumped HIR layout invalidates the
    // whole cache without an `rm -rf` step. A hit lets us skip
    // parse + bench_runtime_1 + lower + opts entirely; we still need
    // a runtime for `compile_module` + `install_interp_jit_with`.
    let cache_key = compute_cache_key(source, target.run_with_opts);
    let cache_dir = bench_cache_dir();

    let t_cache = Instant::now();
    let cached_module: Option<HirModule> = if cache_enabled {
        try_load_cached_hir(&cache_key, &cache_dir)
    } else {
        None
    };
    let cache_lookup_ms = t_cache.elapsed().as_secs_f64() * 1000.0;

    let (module, parse_ms, bench_rt_1_ms, lower_ms, opts_ms): (HirModule, f64, f64, f64, f64) =
        if let Some(m) = cached_module {
            if trace {
                eprintln!(
                    "[BENCH-CACHE] HIT  key={key} kernel={kernel} target={t} (lookup={ms:.2} ms)",
                    key = &cache_key[..cache_key.len().min(8)],
                    t = target.key,
                    ms = cache_lookup_ms,
                );
            }
            (m, 0.0, 0.0, 0.0, 0.0)
        } else {
            if trace {
                eprintln!(
                    "[BENCH-CACHE] MISS key={key} kernel={kernel} target={t} (lookup={ms:.2} ms)",
                    key = &cache_key[..cache_key.len().min(8)],
                    t = target.key,
                    ms = cache_lookup_ms,
                );
            }

            let grammar =
                Grammar2::from_source(ZYNML_GRAMMAR).map_err(|e| format!("grammar: {e:?}"))?;

            let t0 = Instant::now();
            let program = grammar
                .parse_with_filename(source, "<bench>")
                .map_err(|e| format!("parse: {e:?}"))?;
            let parse_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t0 = Instant::now();
            let rt = bench_runtime()?;
            let builtins = rt
                .config()
                .builtins
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let bench_rt_1_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t0 = Instant::now();
            let mut module: HirModule = rt
                .lower_typed_program(program, builtins)
                .map_err(|e| format!("lower: {e:?}"))?;
            let lower_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t0 = Instant::now();
            // run_interp_safe_opts now runs inside ZyntaxRuntime::compile_module
            // (the production path), so the bench harness no longer drives it
            // explicitly here. The opts_ms slot is retained as a zero so the
            // cold-path timing table layout stays stable across runs.
            let _ = &mut module;
            let opts_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Persist the cold-path module so the next iteration can
            // skip everything above. Failures are non-fatal — a broken
            // cache write just means the next run pays the cold-path
            // cost again, not a benchmark failure.
            if cache_enabled {
                try_save_cached_hir(&module, &cache_key, &cache_dir);
            }

            (module, parse_ms, bench_rt_1_ms, lower_ms, opts_ms)
        };

    let t0 = Instant::now();
    let mut rt = bench_runtime()?;
    let bench_rt_2_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    rt.compile_module(&module)
        .map_err(|e| format!("compile_module: {e:?}"))?;
    let compile_module_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    if target.install_jit {
        rt.install_interp_jit_with(jit_tier_config(target.install_llvm))
            .map_err(|e| format!("install_interp_jit: {e:?}"))?;
    }
    let install_jit_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;

    if trace {
        eprintln!(
            "[BENCH-COMPILE] kernel={kernel} target={target_key}\n  \
             cache_lookup = {cache:.2} ms\n  \
             parse        = {parse:.2} ms\n  \
             rt_build_1   = {rt1:.2} ms\n  \
             lower        = {lower:.2} ms\n  \
             opts         = {opts:.2} ms\n  \
             rt_build_2   = {rt2:.2} ms\n  \
             compile_mod  = {cm:.2} ms\n  \
             install_jit  = {ij:.2} ms\n  \
             TOTAL        = {total:.2} ms",
            target_key = target.key,
            cache = cache_lookup_ms,
            parse = parse_ms,
            rt1 = bench_rt_1_ms,
            lower = lower_ms,
            opts = opts_ms,
            rt2 = bench_rt_2_ms,
            cm = compile_module_ms,
            ij = install_jit_ms,
            total = compile_ms,
        );
    }

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

// =========================================================================
// ZBC HIR cache
// -------------------------------------------------------------------------
// A per-project filesystem cache keyed on the inputs that drive the
// compiler frontend. On hit, `one_iteration` skips parse + lower + opts
// and goes straight from the deserialized `HirModule` to `compile_module`
// + JIT install. On miss, we run the cold path and atomically persist
// the produced module via the bytecode crate's `serialize_module`.
//
// Why per-project (under `target/`) rather than `~/.cache/zyntax/`:
//   - `target/` is already gitignored, no extra config to ship.
//   - `cargo clean` nukes the cache as a side effect, which matches
//     the cargo-style mental model — anyone debugging a stale cache
//     will reach for that anyway.
//   - No cross-workspace contamination: two checkouts of the repo
//     at different commits keep their snapshots separate.
// To clear manually: `rm -rf target/zynml-cache/`.

/// Directory holding the `.zbc` snapshots. Resolved relative to the
/// zynml crate's `CARGO_MANIFEST_DIR` so it lands at the workspace
/// `target/` root regardless of the caller's `cwd`.
fn bench_cache_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/zynml-cache/zbc")
}

/// Stable 64-bit FNV-1a hash. Deterministic across Rust versions
/// (unlike `DefaultHasher`) and dependency-free — both properties
/// matter for a filesystem cache key that has to survive toolchain
/// upgrades. Collision probability at the corpus sizes we'll ever
/// hit (a handful of kernels, four targets, ~3000-line prelude) is
/// effectively zero, and a collision just means a wrong-HIR miss
/// that postcard's schema check and crc32fast will catch.
fn fnv1a_64_update(state: u64, bytes: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = state;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Compose the cache key from every input that materially affects
/// the produced `HirModule`. Includes both stdlib files because the
/// resolver weaves them into lowering, the opt-pipeline flag because
/// `run_interp_safe_opts` mutates the module in place, and a pair of
/// version tags so a compiler-schema change or a workspace version
/// bump invalidates the whole cache without any manual `rm` step.
fn compute_cache_key(source: &str, run_with_opts: bool) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    let mut h = FNV_OFFSET;
    // Domain separators between sections so e.g. swapping a byte
    // between the source's tail and the prelude's head can't ever
    // produce the same digest.
    h = fnv1a_64_update(h, b"src\0");
    h = fnv1a_64_update(h, source.as_bytes());
    h = fnv1a_64_update(h, b"\0prelude\0");
    h = fnv1a_64_update(h, ZYNML_STDLIB_PRELUDE.as_bytes());
    h = fnv1a_64_update(h, b"\0tensor\0");
    h = fnv1a_64_update(h, ZYNML_STDLIB_TENSOR.as_bytes());
    h = fnv1a_64_update(h, b"\0opts\0");
    h = fnv1a_64_update(h, &[u8::from(run_with_opts)]);
    h = fnv1a_64_update(h, b"\0schema\0");
    h = fnv1a_64_update(h, &CACHE_SCHEMA_VERSION.to_le_bytes());
    h = fnv1a_64_update(h, b"\0pkg\0");
    h = fnv1a_64_update(h, env!("CARGO_PKG_VERSION").as_bytes());
    format!("{h:016x}")
}

/// Read a snapshot if it exists and deserializes cleanly. Any
/// failure (missing file, IO error, postcard schema mismatch, CRC
/// mismatch) silently returns `None` so the caller falls back to
/// the cold path; corruption is self-healing on the next write.
fn try_load_cached_hir(cache_key: &str, cache_dir: &Path) -> Option<HirModule> {
    let path = cache_dir.join(format!("{cache_key}.zbc"));
    let bytes = fs::read(&path).ok()?;
    deserialize_module(&bytes).ok()
}

/// Persist a snapshot via write-to-tmp + atomic-rename. The tmp +
/// rename pattern keeps concurrent readers from observing a half-
/// written file (the harness is single-threaded today but a
/// future parallel `--jobs N` bench would race on the same key).
/// Failures here are logged but never fatal — a benchmark that
/// can't write to disk should still produce timings.
fn try_save_cached_hir(module: &HirModule, cache_key: &str, cache_dir: &Path) {
    if let Err(e) = fs::create_dir_all(cache_dir) {
        eprintln!(
            "[BENCH-CACHE] WARN  mkdir {dir:?} failed: {e}",
            dir = cache_dir,
        );
        return;
    }
    let final_path = cache_dir.join(format!("{cache_key}.zbc"));
    let tmp_path = cache_dir.join(format!("{cache_key}.zbc.tmp"));
    match serialize_module(module, Format::Postcard) {
        Ok(bytes) => {
            if let Err(e) = fs::write(&tmp_path, &bytes) {
                eprintln!("[BENCH-CACHE] WARN  write {tmp_path:?} failed: {e}");
                return;
            }
            if let Err(e) = fs::rename(&tmp_path, &final_path) {
                eprintln!("[BENCH-CACHE] WARN  rename {tmp_path:?} -> {final_path:?} failed: {e}");
                let _ = fs::remove_file(&tmp_path);
            }
        }
        Err(e) => {
            eprintln!("[BENCH-CACHE] WARN  serialize_module failed: {e}");
        }
    }
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
