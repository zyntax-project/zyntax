//! Benchmark suite runner for ZynML.
//!
//! Measures each `bench_*.zynml` source under
//! `crates/zynml/benchmarks/` across the targets we control today:
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
//! The layout — per-kernel JSON, page reads the JSON and renders bars
//! — is kept deliberately small. External-language targets (python3,
//! node, …) are a follow-up, once the comparison is worth the CI
//! complexity.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use zynml::{ZynML, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD, ZYNML_STDLIB_TENSOR};
use zyntax_compiler::bytecode::{deserialize_module, serialize_module, Format};
use zyntax_compiler::profiling::ProfileConfig;
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_compiler::HirModule;
use zyntax_embed::ZyntaxValue;

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
const CACHE_SCHEMA_VERSION: u32 = 2;

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
    /// Median time to construct the runtime: decoding the language
    /// snapshot, installing the grammar and registering plugins.
    ///
    /// A deployed program pays this once before it can compile
    /// anything, so leaving it out of the numbers understates what a
    /// run costs. It is reported separately rather than folded into
    /// `compile_ms` so the two can be attacked independently.
    #[serde(default)]
    setup_ms: f64,
    /// Median compile-only time (parse + lower + opt). Useful for
    /// pulling apart compile cost from execute cost in charts.
    compile_ms: f64,
    /// Median execute-only time (call `main`, no setup). The page
    /// can sum `compile_ms + exec_ms` if it wants total cost.
    exec_ms: f64,
    /// Setup, compile and execute for the first iteration, before any
    /// warmup: what the very first run of this kernel actually cost.
    ///
    /// Every other number here is a median of warmed iterations, which
    /// is the right shape for comparing steady-state throughput and the
    /// wrong shape for asking what a user waits for. The gap between
    /// this and `setup_ms + compile_ms + exec_ms` is what warmup hides.
    #[serde(default)]
    cold_ms: f64,
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
    /// How the page should section the rows, in order.
    ///
    /// Carried in the results rather than in the page, so a kernel
    /// added to the table above appears under the right heading
    /// without the page being touched. A results file without it
    /// renders as one flat list, which is what older ones do.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    groups: Vec<GroupSection>,
}

/// One published section: a heading, what it measures, and the rows
/// under it in the order they should appear.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GroupSection {
    title: String,
    blurb: String,
    /// Shown apart from the blurb, because what a reader must know
    /// before comparing a number is not the same kind of statement as
    /// what the number is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    caveat: Option<String>,
    kernels: Vec<String>,
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
    /// CPU brand string (e.g. "Apple M1", "Intel(R) Xeon(R) Platinum
    /// 8370C CPU @ 2.80GHz"). Empty string if probing failed.
    /// Read from `sysctl -n machdep.cpu.brand_string` on macOS and
    /// `/proc/cpuinfo` on Linux.
    #[serde(default)]
    cpu: String,
    /// Logical CPU count (`thread::available_parallelism`). 0 if the
    /// probe failed.
    #[serde(default)]
    cpu_cores: usize,
    /// Total system RAM in GiB, integer-truncated. 0 if the probe
    /// failed. Read from `sysctl -n hw.memsize` on macOS and
    /// `/proc/meminfo` (MemTotal) on Linux.
    #[serde(default)]
    ram_gb: u64,
    /// `hostname` output. Public CI runners (GitHub Actions) give
    /// out unique-per-job names — useful for cross-referencing a
    /// noisy bench result with the specific runner instance it
    /// landed on.
    #[serde(default)]
    host: String,
}

/// Each benchmark source lives at
/// `crates/zynml/benchmarks/<name>.zynml` and gets run across every
/// target listed in [`TARGETS`]. The second tuple element is the
/// expected `Debug`-formatted result; the bench harness asserts
/// every successful iteration matches it and fails the run with a
/// non-zero exit code on mismatch. Without this assertion, a
/// miscompile that returns the wrong value (e.g. fib LLVM tier
/// returning `Int(38)` instead of `Int(102334155)` after the
/// recursive-inline pass landed broken) still reports a "green"
/// CI workflow because the bench harness records the value but
/// never validates it.
const KERNELS: &[Kernel] = &[
    Kernel::new("bench_mandelbrot", "Int(112789639)").expecting_without_opts("Int(112790102)"),
    // Four rows of the same fractal, chosen because they straddle the
    // escape boundary where fusing `2*zx*zy + cy` changes the answer:
    // strict evaluation gives 908586 and the fused one 908666. Small
    // enough for the bytecode interpreter to finish, so every tier can
    // be asked the same question. A tier can be swapped under a running
    // loop, so they have to agree.
    Kernel::new("bench_mandelbrot_strip", "Int(908666)").expecting_without_opts("Int(908586)"),
    Kernel::new("bench_nbody", "Int(-169077)"),
    Kernel::new("bench_nbody_ref", "Int(-169077)"),
    Kernel::new("bench_fib", "Int(102334155)"),
    // Same source as `bench_fib`, compiled with pure-call PRE off.
    //
    // With the pass on, recursive self-inlining plus the cross-branch
    // hoist collapses most of the call tree, so the kernel stops
    // measuring the call dispatch it exists to measure. Publishing both
    // rows keeps the dispatch number visible and makes what the pass is
    // worth a number rather than a claim.
    //
    // The row moves the Cranelift tier and not the LLVM one, because
    // LLVM's own optimizer performs the same collapse whether or not
    // ours ran. Measured on x86_64 with the cache off: Cranelift 7.7 ms
    // with the pass and 295 ms without, LLVM 7.1 ms either way. So the
    // pair reads as what the pass is worth to the tier that lacks it.
    Kernel::new("bench_fib", "Int(102334155)")
        .published_as("fib_no_pure_call_pre")
        .without_pure_call_pre(),
    Kernel::new("bench_inlined_call", "Int(350000000)"),
    Kernel::new("bench_free_function_call", "Int(350000000)"),
    // diagnostic-only — kept out of CI publish surface but used for
    // tracing operator-overload lowering. Expected: a + b * 10M with
    // a=(1,2,3), b=(4,5,6) → acc = (50000000, 70000000, 90000000)
    // → sum 210000000.
    Kernel::new("bench_op_overload", "Int(210000000)"),
    Kernel::new("bench_op_overload_ref", "Int(21000000)"),
    Kernel::new("bench_any_field", "Int(1500000)"),
    Kernel::new("bench_any_cast", "Int(2500000)"),
    // Branch-heavy, data-dependent kernels. The others are numeric loops
    // that the HIR passes reshape before Cranelift sees them, which hides
    // what Cranelift's own optimizer contributes; these two do not give
    // LICM, auto-vectorization or SROA anything to work with.
    // Tensor kernels at real shapes. Each isolates a different
    // property: the element-wise add is the shape the vectorizer
    // matches, axpy is a multiply feeding an add with a loop-invariant
    // scalar and is the shape it does not, dot carries an accumulator
    // rather than storing, and matmul is strided and shaped.
    Kernel::new("bench_tensor_add", "Int(12)").ml(),
    Kernel::new("bench_tensor_axpy", "Int(2)").ml(),
    Kernel::new("bench_tensor_dot", "Int(8388608)").ml(),
    Kernel::new("bench_tensor_matmul", "Int(2048)").ml(),
    // One transformer layer's prefill. The tensor kernels above each
    // isolate one shape; this is what a real workload looks like once
    // they are composed, and the phase of serving a prompt where the
    // work is dense enough for spreading it across cores to pay.
    Kernel::new("bench_llm_prefill", "Int(806)").ml(),
    // The other half of serving: one token at a time, where every
    // matrix multiply is a matrix-vector product and the cost is in
    // reading the weights rather than multiplying by them. It is also
    // where a dispatch is paid a thousand times rather than a dozen.
    Kernel::new("bench_llm_decode", "Int(55842)").ml(),
    Kernel::new("bench_collatz", "Int(35669673)"),
    Kernel::new("bench_branchy", "Int(140)"),
];

/// One published measurement: a source file plus the pipeline it is
/// compiled with.
///
/// Two entries may name the same source and differ only in pipeline,
/// which is how an optimisation's contribution becomes a published
/// number instead of an assertion. They must share the source for the
/// comparison to mean anything, so the source is named once and the
/// row is renamed instead.
#[derive(Debug, Clone, Copy)]
struct Kernel {
    /// Source file stem under `benchmarks/`.
    source: &'static str,
    /// Row name in the JSON, defaulting to `source` without its
    /// `bench_` prefix.
    published_as: Option<&'static str>,
    /// The value `main` must return on every tier.
    expected: &'static str,
    /// The value to expect from a row that runs without the HIR
    /// passes, where those passes change it.
    ///
    /// Contracting a multiply and an add rounds once where the pair
    /// rounds twice, so a kernel near a decision boundary answers
    /// differently depending on whether contraction ran. That is a
    /// different arithmetic, not a wrong one, and a single expected
    /// value would report it as a miscompile. `None` means the passes
    /// do not change this kernel's result, which is true of every
    /// kernel that computes in integers.
    expected_without_opts: Option<&'static str>,
    /// Whether cross-branch pure-call PRE runs for this row.
    pure_call_pre: bool,
    /// Which section of the published page this row belongs under.
    ///
    /// The page reads the grouping from the results rather than
    /// carrying its own list, so a kernel added here appears in the
    /// right place without the page being edited.
    group: Group,
}

/// The sections the published benchmark page is divided into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Group {
    /// Language and compiler shapes: control flow, calls, arithmetic.
    Core,
    /// Tensor and model kernels, where the question is throughput on
    /// real numeric work rather than what the compiler does with a
    /// language construct.
    Ml,
}

impl Group {
    /// The heading this section is published under.
    fn title(self) -> &'static str {
        match self {
            Group::Core => "Language and compiler kernels",
            Group::Ml => "ML kernels",
        }
    }

    /// What the section is measuring, in one sentence, for the page.
    fn blurb(self) -> &'static str {
        match self {
            Group::Core => {
                "Classic shapes, each chosen for what it makes the compiler do: \
                 recursion and call dispatch, branch-heavy control flow, \
                 floating-point loops. Two rows may share a source and differ \
                 only in which pass ran, which is what turns that pass's \
                 contribution into a number."
            }
            Group::Ml => {
                "Numeric kernels at shapes a real workload uses. The tensor rows \
                 isolate one shape each: the elementwise add is what the \
                 vectorizer matches, axpy is a multiply feeding an add that it \
                 does not, dot carries an accumulator, and matmul is strided. \
                 The two model rows are the halves of serving one: prefill \
                 pushes every token through at once and is bound by arithmetic, \
                 decode moves one token at a time and is bound by how fast the \
                 weights arrive."
            }
        }
    }

    /// What a reader has to know before comparing these against
    /// anything else.
    fn caveat(self) -> Option<&'static str> {
        match self {
            Group::Core => None,
            Group::Ml => Some(
                "No hardware acceleration. These do not call Accelerate, a \
                 vendor BLAS, cuBLAS, or any neural engine, and the two model \
                 kernels reach no library at all: the matrix multiplies, the \
                 softmax and its exponential, and the normalisation and its \
                 reciprocal square root are written in the language and \
                 compiled here. Read against a hardware-accelerated stack these \
                 would lose, and the comparison would be measuring something \
                 else. What they measure is what the compiler does with the \
                 arithmetic it is given.",
            ),
        }
    }
}

impl Kernel {
    /// The value this kernel returns when the HIR passes are off, for a
    /// kernel whose arithmetic they change.
    const fn expecting_without_opts(mut self, expected: &'static str) -> Self {
        self.expected_without_opts = Some(expected);
        self
    }

    const fn new(source: &'static str, expected: &'static str) -> Self {
        Self {
            source,
            published_as: None,
            expected,
            expected_without_opts: None,
            pure_call_pre: true,
            group: Group::Core,
        }
    }

    /// Publish this row under the ML section.
    const fn ml(mut self) -> Self {
        self.group = Group::Ml;
        self
    }

    /// Publish under a different name, for a second pipeline over the
    /// same source.
    const fn published_as(mut self, name: &'static str) -> Self {
        self.published_as = Some(name);
        self
    }

    /// Compile this row with cross-branch pure-call PRE off.
    const fn without_pure_call_pre(mut self) -> Self {
        self.pure_call_pre = false;
        self
    }

    /// The source's own name, with the `bench_` prefix stripped. Shared
    /// by every row over that source.
    fn source_name(&self) -> &'static str {
        self.source.strip_prefix("bench_").unwrap_or(self.source)
    }

    /// The name this row is published and filtered under.
    fn name(&self) -> &'static str {
        self.published_as
            .unwrap_or_else(|| self.source.strip_prefix("bench_").unwrap_or(self.source))
    }
}

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
        skip_kernels: &[
            "mandelbrot",
            "nbody",
            "nbody_ref",
            "fib",
            "inlined_call",
            "free_function_call",
            "collatz",
            "branchy",
            "tensor_matmul",
            "llm_prefill",
            "llm_decode",
        ],
    },
    Target {
        key: "zyntax-interp-opt",
        label: "Zyntax · BC interp + opt",
        run_with_opts: true,
        install_jit: false,
        install_llvm: false,
        skip_kernels: &[
            "mandelbrot",
            "nbody",
            "nbody_ref",
            "fib",
            "inlined_call",
            "free_function_call",
            "collatz",
            "branchy",
            "tensor_matmul",
            "llm_prefill",
            "llm_decode",
        ],
    },
    Target {
        key: "zyntax-tiered",
        label: "Zyntax · BC interp → Cranelift tier-up",
        run_with_opts: true,
        install_jit: true,
        install_llvm: false,
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
/// Set by `--osr`; read when each target builds its tier config.
static OSR_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn jit_tier_config(install_llvm: bool, llvm_cache_key: Option<String>) -> TieredConfig {
    let mut cfg = TieredConfig::default();
    cfg.llvm_cache_key = llvm_cache_key;
    // On-stack replacement is off by default: it emits a probe on every
    // back edge, which is a cost the other numbers should not carry.
    cfg.enable_osr = OSR_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
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
    // Disable beadie's auto multi-tier promotion. The Cranelift compile
    // closure at interp_runtime.rs:769 deliberately ignores the tier
    // argument and always returns the Cranelift pointer (the LLVM tier
    // is wired through a separate hand-rolled side-channel that calls
    // `LLVMJitBackend::compile_function` directly). Leaving beadie's
    // tier 2 promotion broker enabled means it races the side-channel:
    // it submits a "tier 2" compile that re-fetches the Cranelift
    // pointer and bumps generation 1 → 2 before LLVM gets a chance,
    // and the side-channel's `generation() == 1` gate then misses.
    // Disable here so only the side-channel can flip past tier 1.
    cfg.enable_background_optimization = false;
    cfg
}

fn main() {
    // Parse command-line flags.
    let mut out_path: Option<PathBuf> = None;
    let mut runs_override: Option<usize> = None;
    let mut cache_enabled = true;
    let mut kernel_filter: Option<String> = None;
    let mut target_filter: Option<String> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                out_path = args.next().map(PathBuf::from);
            }
            "--runs" => {
                runs_override = args.next().and_then(|s| s.parse().ok());
            }
            "--osr" => {
                OSR_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            "--no-cache" => {
                cache_enabled = false;
            }
            "--filter" => {
                kernel_filter = args.next();
            }
            "--target" => {
                target_filter = args.next();
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: bench_runner [--out <path>] [--runs <n>] [--no-cache] [--osr]\n\
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
        groups: Vec::new(),
        meta: Meta {
            date: rfc3339_now(),
            commit: git_short_sha(),
            arch: env::consts::ARCH.to_string(),
            os: env::consts::OS.to_string(),
            runs,
            cpu: probe_cpu_brand(),
            cpu_cores: probe_cpu_cores(),
            ram_gb: probe_ram_gb(),
            host: probe_hostname(),
        },
    };

    let mut value_mismatches: Vec<String> = Vec::new();

    for kernel_spec in KERNELS {
        let pretty = kernel_spec.name();
        let kernel = kernel_spec.source;
        let expected = &kernel_spec.expected;
        // Each row states the pipeline it wants, so a row that opts out
        // of a pass cannot leak that choice into the next one.
        zyntax_compiler::pure_call_pre::set_enabled(Some(kernel_spec.pure_call_pre));
        if let Some(f) = &kernel_filter {
            // Exact-match against the stripped kernel name. Substring
            // was the old default; it silently matched `nbody` against
            // both `nbody` and `nbody_ref`, so the per-kernel GHA matrix
            // ran nbody_ref twice — once alone, once chained behind
            // nbody on the same runner — and the published number was
            // the thermally-throttled second pass (~100 ms worse than
            // the dedicated job).
            // Comma-separated so two rows over one source can be
            // compared in a single process.
            if !f.split(',').any(|want| want.trim() == pretty) {
                continue;
            }
        }
        let source_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("benchmarks/{kernel}.zynml"));
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|e| panic!("read {source_path:?}: {e}"));

        eprintln!("==> kernel {pretty}");
        let mut per_kernel: KernelResults = BTreeMap::new();
        for target in TARGETS {
            if let Some(tf) = &target_filter {
                if !target.key.contains(tf.as_str()) {
                    continue;
                }
            }
            // Opt-outs are keyed by the source, since what makes a
            // kernel too slow for a tier is what it computes, not which
            // pipeline row is measuring it.
            // A build without `llvm-backend` still has this target,
            // but nothing escalates past Cranelift in it, so it would
            // report Cranelift's numbers under LLVM's name. Say so
            // instead of publishing a figure for a tier that was never
            // in the binary.
            if target.install_llvm && !cfg!(feature = "llvm-backend") {
                eprintln!(
                    "    {:<22} SKIPPED (built without the llvm-backend feature)",
                    target.key
                );
                per_kernel.insert(target.key.to_string(), skipped_result());
                continue;
            }
            if target.skip_kernels.contains(&kernel_spec.source_name()) {
                eprintln!("    {:<22} SKIPPED (per-target opt-out)", target.key);
                per_kernel.insert(target.key.to_string(), skipped_result());
                continue;
            }
            let r = measure(
                &source,
                target,
                runs,
                pretty,
                cache_enabled,
                kernel_spec.pure_call_pre,
            );
            if let Some(err) = r.error.as_ref() {
                eprintln!("    {:<22} FAILED — {err}", target.key);
            } else {
                eprintln!(
                    "    {:<22} setup={:>6.2}ms compile={:>7.2}ms exec={:>9.2}ms \
                     total={:>9.2}ms cold={:>9.2}ms  -> {}",
                    target.key,
                    r.setup_ms,
                    r.compile_ms,
                    r.exec_ms,
                    r.seconds * 1000.0,
                    r.cold_ms,
                    r.result,
                );
                // Correctness gate: every successful tier on this
                // kernel must return the canonical reference value.
                // Without this, a miscompile that silently returns
                // garbage (e.g. fib LLVM returning Int(38) after the
                // recursive-inline pass landed broken) still shows
                // as a "green" CI workflow because the bench harness
                // only records the value, never validates it.
                // Accumulate every mismatch across the run so one
                // tier failing doesn't suppress visibility into
                // other tiers; we surface them all + exit non-zero
                // at the end.
                let expected = match kernel_spec.expected_without_opts {
                    Some(without) if !target.run_with_opts => without,
                    _ => kernel_spec.expected,
                };
                if r.result != expected {
                    eprintln!(
                        "    {:<22} VALUE MISMATCH — got {}, expected {}",
                        target.key, r.result, expected
                    );
                    value_mismatches.push(format!(
                        "{}/{}: got {}, expected {}",
                        pretty, target.key, r.result, expected
                    ));
                }
            }
            per_kernel.insert(target.key.to_string(), r);
        }
        suite.kernels.insert(pretty.to_string(), per_kernel);
    }
    zyntax_compiler::pure_call_pre::set_enabled(None);

    // Make sure the output directory exists. `mkdir -p` shape so
    // a fresh checkout can write into `website/benchmark/` without
    // anyone hand-creating it.
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|e| panic!("mkdir {parent:?}: {e}"));
    }
    // Section the rows the page will show. Only sections with a row
    // in this run are carried, so a filtered run does not publish an
    // empty heading.
    for group in [Group::Core, Group::Ml] {
        let kernels: Vec<String> = KERNELS
            .iter()
            .filter(|k| k.group == group)
            .map(|k| k.name().to_string())
            .filter(|name| suite.kernels.contains_key(name))
            .collect();
        if !kernels.is_empty() {
            suite.groups.push(GroupSection {
                title: group.title().to_string(),
                blurb: group.blurb().to_string(),
                caveat: group.caveat().map(|c| c.to_string()),
                kernels,
            });
        }
    }

    let json = serde_json::to_string_pretty(&suite).expect("serialize results");
    fs::write(&out_path, json).unwrap_or_else(|e| panic!("write {out_path:?}: {e}"));
    eprintln!("\nwrote {}", out_path.display());

    // Fail the process AFTER writing results.json so the partial
    // (still-real) data lands on disk for forensics, but the CI
    // workflow still goes red on a miscompile. Filter-/target-
    // restricted runs are exempt: a `--filter fib` invocation that
    // only exercises one kernel isn't claiming to validate the
    // whole suite, just to time that one. Anything that lists at
    // least one mismatch fails.
    if !value_mismatches.is_empty() {
        eprintln!("\nVALUE MISMATCHES — bench will exit non-zero:");
        for m in &value_mismatches {
            eprintln!("  {m}");
        }
        std::process::exit(2);
    }
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
    pure_call_pre: bool,
) -> TargetResult {
    let mut setup_samples = Vec::with_capacity(runs);
    let mut compile_samples = Vec::with_capacity(runs);
    let mut exec_samples = Vec::with_capacity(runs);
    let mut last_result_str = String::new();

    // Warmup — toss the times, but the runs still mutate caches /
    // allocators / OS-page state so timed iters run on a steady
    // baseline. A failure during warmup short-circuits straight
    // to the `error` shape; no point spending the runs-loop's
    // budget on something that's reliably broken.
    // The first iteration is the only cold one: it pays for page-cache
    // misses, allocator growth and every lazily-decoded artifact the
    // process shares. Warmup exists to take that out of the timed runs,
    // so it has to be measured here or it is never measured at all.
    let cold = match one_iteration(source, target, kernel, cache_enabled, pure_call_pre) {
        Ok((setup_ms, compile_ms, exec_ms, _)) => setup_ms + compile_ms + exec_ms,
        Err(e) => return failed_result(&e),
    };
    for _ in 0..WARMUP {
        if let Err(e) = one_iteration(source, target, kernel, cache_enabled, pure_call_pre) {
            return failed_result(&e);
        }
    }
    for _ in 0..runs {
        match one_iteration(source, target, kernel, cache_enabled, pure_call_pre) {
            Ok((setup_ms, compile_ms, exec_ms, r)) => {
                setup_samples.push(setup_ms);
                compile_samples.push(compile_ms);
                exec_samples.push(exec_ms);
                last_result_str = format!("{r:?}");
            }
            Err(e) => return failed_result(&e),
        }
    }

    let median_setup = median(&mut setup_samples);
    let median_compile = median(&mut compile_samples);
    let median_exec = median(&mut exec_samples);
    TargetResult {
        seconds: (median_compile + median_exec) / 1000.0,
        setup_ms: median_setup,
        compile_ms: median_compile,
        exec_ms: median_exec,
        cold_ms: cold,
        result: last_result_str,
        error: None,
        skipped: false,
    }
}

fn failed_result(error: &str) -> TargetResult {
    TargetResult {
        seconds: 0.0,
        setup_ms: 0.0,
        compile_ms: 0.0,
        exec_ms: 0.0,
        cold_ms: 0.0,
        result: "—".to_string(),
        error: Some(error.to_string()),
        skipped: false,
    }
}

fn skipped_result() -> TargetResult {
    TargetResult {
        seconds: 0.0,
        setup_ms: 0.0,
        compile_ms: 0.0,
        exec_ms: 0.0,
        cold_ms: 0.0,
        result: "—".to_string(),
        error: None,
        skipped: true,
    }
}

/// One full kernel run: lower from source, optionally apply the
/// HIR opt pipeline, install into a fresh runtime, drive the JIT
/// tier-up if requested, then call `main`. Returns
/// `(setup_ms, compile_ms, exec_ms, result)`.
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
    pure_call_pre: bool,
) -> Result<(f64, f64, f64, ZyntaxValue), String> {
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
    // whole cache without an `rm -rf` step. A hit lets us skip parse,
    // lowering and opts; we still need an artifact-backed runtime for
    // `compile_module` + `install_interp_jit_with`.
    let cache_key = compute_cache_key(source, target.run_with_opts, pure_call_pre);
    let cache_dir = bench_cache_dir();

    let t_cache = Instant::now();
    let cached_module: Option<HirModule> = if cache_enabled {
        try_load_cached_hir(&cache_key, &cache_dir)
    } else {
        None
    };
    let cache_lookup_ms = t_cache.elapsed().as_secs_f64() * 1000.0;

    // Construct exactly the runtime production ZynML uses. This decodes the
    // build-time grammar and stdlib artifacts and installs their compiled
    // import resolver. Keeping this runtime for lowering *and* compilation
    // mirrors `ZynML::load_source`; constructing a second runtime here would
    // charge setup twice and would no longer be a deployable cold-start metric.
    let t0 = Instant::now();
    let mut zynml = ZynML::new().map_err(|e| format!("runtime setup: {e:?}"))?;
    let runtime_setup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (module, parse_ms, lower_ms, opts_ms): (HirModule, f64, f64, f64) =
        if let Some(m) = cached_module {
            if trace {
                eprintln!(
                    "[BENCH-CACHE] HIT  key={key} kernel={kernel} target={t} (lookup={ms:.2} ms)",
                    key = &cache_key[..cache_key.len().min(8)],
                    t = target.key,
                    ms = cache_lookup_ms,
                );
            }
            (m, 0.0, 0.0, 0.0)
        } else {
            if trace {
                eprintln!(
                    "[BENCH-CACHE] MISS key={key} kernel={kernel} target={t} (lookup={ms:.2} ms)",
                    key = &cache_key[..cache_key.len().min(8)],
                    t = target.key,
                    ms = cache_lookup_ms,
                );
            }

            let t0 = Instant::now();
            let program = zynml
                .grammar2()
                .ok_or_else(|| "compiled Grammar2 parser unavailable".to_string())?
                .parse_with_filename(source, "<bench>")
                .map_err(|e| format!("parse: {e:?}"))?;
            let parse_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let builtins = zynml
                .runtime()
                .config()
                .builtins
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            let t0 = Instant::now();
            let mut module: HirModule = zynml
                .runtime()
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

            (module, parse_ms, lower_ms, opts_ms)
        };

    let t0 = Instant::now();
    // The row that exists to show what the HIR passes are worth has to
    // actually run without them. They moved inside `compile_module`, so
    // asking the runtime is the only way to say so.
    zynml
        .runtime_mut()
        .set_run_interp_opts(target.run_with_opts);
    zynml
        .runtime_mut()
        .compile_module(&module)
        .map_err(|e| format!("compile_module: {e:?}"))?;
    let compile_module_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    if target.install_jit {
        // Hand the LLVM dylib cache the same content-hash we used
        // for the HIR cache above — they share `cache_enabled` and
        // invalidate together via the schema-version constant in
        // `compute_cache_key`. On a cache hit the LLVM backend skips
        // the entire install pipeline (lower → opt → link → dlopen)
        // and reuses the already-mapped dylib's function pointers.
        let llvm_cache_key = if cache_enabled && target.install_llvm {
            Some(cache_key.clone())
        } else {
            None
        };
        zynml
            .runtime()
            .install_interp_jit_with(jit_tier_config(target.install_llvm, llvm_cache_key))
            .map_err(|e| format!("install_interp_jit: {e:?}"))?;
    }
    let install_jit_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Setting up the runtime is not compiling a program. A host builds
    // one runtime and compiles against it for the rest of the process,
    // so charging every kernel for decoding the grammar and stdlib
    // artifacts measures the host's startup rather than the compiler.
    // The cache lookup goes with it: that is the harness looking for a
    // snapshot, not work a compile does. What is left is parse through
    // lowering, optimisation, codegen and install.
    let compile_ms =
        (compile_start.elapsed().as_secs_f64() * 1000.0) - runtime_setup_ms - cache_lookup_ms;

    if trace {
        eprintln!(
            "[BENCH-COMPILE] kernel={kernel} target={target_key}\n  \
             cache_lookup = {cache:.2} ms (not counted)\n  \
             runtime_setup= {runtime_setup:.2} ms (reported as setup_ms)\n  \
             parse        = {parse:.2} ms\n  \
             lower        = {lower:.2} ms\n  \
             opts         = {opts:.2} ms\n  \
             compile_mod  = {cm:.2} ms\n  \
             install_jit  = {ij:.2} ms\n  \
             TOTAL        = {total:.2} ms",
            target_key = target.key,
            cache = cache_lookup_ms,
            parse = parse_ms,
            runtime_setup = runtime_setup_ms,
            lower = lower_ms,
            opts = opts_ms,
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
            zynml
                .runtime()
                .call_function_raw("main", vec![])
                .map_err(|e| format!("jit warmup: {e:?}"))?;
        }
    }

    // Wrap the timed call in `catch_unwind` so a panic inside the
    // Cranelift backend (a known failure mode for some HIR shapes)
    // surfaces as an `error` row on the bench page rather than
    // aborting the whole suite.
    let exec_start = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        zynml.runtime().call_function_raw("main", vec![])
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

    Ok((runtime_setup_ms, compile_ms, exec_ms, result))
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
fn compute_cache_key(source: &str, run_with_opts: bool, pure_call_pre: bool) -> String {
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
    h = fnv1a_64_update(h, ZYNML_STDLIB_SIMD.as_bytes());
    h = fnv1a_64_update(h, b"\0opts\0");
    h = fnv1a_64_update(h, &[u8::from(run_with_opts)]);
    // The pass configuration is part of what produced the artifact.
    // Two rows can share a source and differ only here, and without
    // this they collide: the second row is handed the first row's
    // compiled dylib and reports the pipeline it opted out of.
    h = fnv1a_64_update(h, b"\0pure_call_pre\0");
    h = fnv1a_64_update(h, &[u8::from(pure_call_pre)]);
    h = fnv1a_64_update(h, b"\0schema\0");
    h = fnv1a_64_update(h, &CACHE_SCHEMA_VERSION.to_le_bytes());
    h = fnv1a_64_update(h, b"\0pkg\0");
    h = fnv1a_64_update(h, env!("CARGO_PKG_VERSION").as_bytes());
    // What produced the snapshot is part of what the snapshot is. A
    // change to lowering alters the module without altering the source,
    // the stdlib or the schema, so a key built from those alone lets a
    // snapshot from an older compiler answer for a newer one. The
    // schema constant does not cover it either: it is bumped by hand,
    // so it protects only the changes someone remembered to bump for.
    h = fnv1a_64_update(h, b"\0build\0");
    h = fnv1a_64_update(h, &build_fingerprint().to_le_bytes());
    format!("{h:016x}")
}

/// Identifies the binary asking for the cache.
///
/// The compiler is linked into this executable, so the executable
/// changing is exactly the condition under which a snapshot it wrote
/// earlier may no longer describe what it would produce now. Size and
/// modification time are enough to say "a different build": a rebuild
/// that changes nothing costs one recompile, while reusing a stale
/// snapshot costs a wrong published number.
///
/// Falls back to a constant when the executable cannot be inspected,
/// which reverts to the previous behaviour rather than disabling the
/// cache outright.
fn build_fingerprint() -> u64 {
    let Ok(exe) = env::current_exe() else {
        return 0;
    };
    let Ok(meta) = fs::metadata(&exe) else {
        return 0;
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    mtime ^ meta.len().rotate_left(32)
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

/// Best-effort CPU brand string. macOS uses `sysctl`, Linux reads
/// `/proc/cpuinfo`'s `model name` line. Falls back to "" so the
/// page can render a single em-dash rather than a stack of probe
/// errors.
fn probe_cpu_brand() -> String {
    use std::process::Command;
    if cfg!(target_os = "macos") {
        if let Ok(o) = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if o.status.success() {
                return String::from_utf8_lossy(&o.stdout).trim().to_string();
            }
        }
    } else if cfg!(target_os = "linux") {
        if let Ok(s) = fs::read_to_string("/proc/cpuinfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("model name") {
                    if let Some(v) = rest.split(':').nth(1) {
                        return v.trim().to_string();
                    }
                }
            }
        }
    }
    String::new()
}

/// Logical CPU count via `std::thread::available_parallelism`. This
/// honours cgroup CPU quotas (so a 4-CPU GHA runner reports 4, not
/// the bare-metal hypervisor's higher count), which is the number
/// the bench actually sees scheduling-wise.
fn probe_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}

/// Total system RAM in GiB, integer-truncated. macOS via `sysctl
/// -n hw.memsize` (bytes), Linux via `/proc/meminfo`'s `MemTotal:
/// <KiB> kB` line. Truncates rather than rounds so 15.6 GiB CI
/// machines report 15, matching how rayzor's page renders.
fn probe_ram_gb() -> u64 {
    use std::process::Command;
    if cfg!(target_os = "macos") {
        if let Ok(o) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if o.status.success() {
                if let Ok(bytes) = String::from_utf8_lossy(&o.stdout).trim().parse::<u64>() {
                    return bytes / (1024 * 1024 * 1024);
                }
            }
        }
    } else if cfg!(target_os = "linux") {
        if let Ok(s) = fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let kb: u64 = rest
                        .trim()
                        .split_whitespace()
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    return kb / (1024 * 1024);
                }
            }
        }
    }
    0
}

/// Hostname via the `hostname` binary. Works on every CI runner we
/// target and avoids pulling a libc-bindings crate into the bench
/// example just for this.
fn probe_hostname() -> String {
    use std::process::Command;
    Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_default()
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
