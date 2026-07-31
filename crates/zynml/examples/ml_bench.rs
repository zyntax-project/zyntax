//! ML micro-benchmark runner for ZynML (`ml_bench`).
//!
//! Mirrors `crates/zynml/examples/bench_runner.rs` in spirit — a
//! phase-tagged registry of ML kernels, each measured on the tiers we
//! control, separating compile time from execute time, median-of-N,
//! JSON out — but oriented at the ML roadmap (`docs/ML_ROADMAP.md`)
//! instead of the general-purpose language kernels.
//!
//! ## What runs today
//!
//! Only **Phase 0**'s baseline is real. The ZynML roadmap's Phases 1–4
//! (`Tensor<T, const RANK>`, explicit SIMD intrinsics, typed aligned
//! buffers, GEMM, attention, …) need language enablers that are not yet
//! built, so those kernels are registered `pending: true` and skipped
//! by the runner with a printed `PENDING (Phase N)` line. They are
//! never compiled.
//!
//! The one real kernel — `qdot_u8i8_baseline` — is the quantized
//! `u8×i8` dot that auto-vectorizes to `vpdpbusd` (x86 VNNI) / `sdot`
//! (aarch64) / `i32x4.dot` (wasm) / a widening fallback. It is built as
//! a hand-assembled `HirFunction` (the same shape as the compiler's
//! `test_vector_dot_i8x16_vpdpbusd`) rather than compiled from a ZynML
//! source string: expressing the exact widening-accumulate loop
//! template the auto-vectorizer recognizes needs `i8`/`u8` element
//! types and widening casts in ZynML source that the frontend does not
//! surface cleanly yet. The hand-built HIR exercises the identical
//! backend `VectorDot` lowering.
//!
//! It is measured across the full tier ladder, mirroring `bench_runner.rs`.
//! Cranelift is not the final tier — LLVM sits above it:
//!   * `interp`      — pure BC interpreter (`Op::VDot`), no JIT.
//!   * `tiered`      — BC interp → Cranelift tier-up (via
//!                     `install_interp_jit_with`).
//!   * `tiered-llvm` — BC interp → Cranelift → LLVM, the full production
//!                     ladder. Only escalates past Cranelift when built
//!                     with `--features llvm-backend` (else it records the
//!                     same numbers as `tiered`, like `bench_runner`'s
//!                     `zyntax-tiered-llvm`). Confirm the LLVM tier really
//!                     fires with `ZYNTAX_TRACE_TIER_UP=1` (look for
//!                     `[TIER-UP-INSTALL-LLVM]`). For this memory-bound
//!                     streaming int8 dot LLVM lands but shows ~no leverage
//!                     over Cranelift — both lower the inner loop to the
//!                     same VNNI-class sequence and exec is dominated by the
//!                     buffer sweep. The LLVM tier's leverage is a
//!                     compute-bound story (Phase 2 GEMM), not this kernel.
//!   * `cranelift`   — a direct `CraneliftBackend` compile+finalize, a
//!                     clean pure-codegen number with no interp/dispatch.
//! Every tier carries the same correctness pin (result must equal the
//! Rust reference, else the row is a reported failure). Each row also
//! reports its **leverage** — the throughput speedup versus the `interp`
//! baseline — so the payoff of each promotion is legible.
//!
//! Exercise the LLVM tier with (LLVM 21 toolchain required):
//!     LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm \
//!         cargo run --release --package zynml --example ml_bench \
//!         --features llvm-backend
//!
//! Usage:
//!     cargo run --release --package zynml --example ml_bench
//!     cargo run --release --package zynml --example ml_bench -- --out /tmp/ml.json
//!     cargo run --release --package zynml --example ml_bench -- --runs 7

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::{
    BinaryOp, CallingConvention, HirConstant, HirFunction, HirFunctionSignature, HirId,
    HirInstruction, HirParam, HirPhi, HirTerminator, HirType, HirValueKind, ParamAttributes,
};
use zyntax_compiler::hir_interp::value_to_i64;
use zyntax_compiler::profiling::ProfileConfig;
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_compiler::HirModule;
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};
use zyntax_typed_ast::InternedString;

const WARMUP: usize = 3;
const RUNS: usize = 7;

/// Untimed warmup calls for the tiered target — drives the interp past
/// the warm threshold so the background Cranelift compile lands and
/// subsequent ticks dispatch to the JIT'd code before the timed call.
/// Same count/intent as `bench_runner.rs`.
const JIT_TIER_WARMUP_CALLS: usize = 4;

/// Elements per benchmarked buffer for the real Phase 0 kernel. Must be
/// a multiple of 16 (the i8x16 chunk width). 1 MiB of each operand keeps
/// the measured region comfortably above per-call FFI overhead.
const QDOT_N: usize = 1 << 20;

/// What unit a kernel's throughput is reported in.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum Metric {
    /// Compute-bound floating point (GEMM, attention).
    Gflops,
    /// Memory-bound bandwidth (elementwise, norms).
    Gbps,
    /// Integer op throughput (int8 quantized dot / GEMM).
    Gops,
}

impl Metric {
    fn unit(self) -> &'static str {
        match self {
            Metric::Gflops => "GFLOP/s",
            Metric::Gbps => "GB/s",
            Metric::Gops => "GOP/s",
        }
    }
}

/// One registered ML kernel. `pending` kernels are documented scaffolds
/// from the roadmap — never compiled or run. The single non-`pending`
/// entry carries a `builder` that hand-assembles the HIR to measure.
struct Kernel {
    /// Stable kernel name (JSON key).
    name: &'static str,
    /// Roadmap phase this kernel belongs to (0–4).
    phase: u8,
    /// True for future-phase scaffolds: printed as `PENDING`, not run.
    pending: bool,
    /// Throughput unit.
    metric: Metric,
    /// Baseline this kernel is measured against (roadmap "Baselines &
    /// targets" table), for the record / the eventual page.
    baseline: &'static str,
    /// One-line note: what the kernel does / what it is waiting on.
    note: &'static str,
    /// Hand-built HIR + op-count + call harness. `None` for pending.
    builder: Option<fn() -> RealKernel>,
}

/// A ready-to-run hand-built kernel: the HIR to compile, the working-set
/// buffers (kept alive for the duration of every timed call), and the
/// metadata needed to run it on any tier and check the result.
struct RealKernel {
    /// The module to compile. It contains the kernel function (resolved
    /// name [`Self::fn_name`]) plus anything that function needs — a
    /// hand-built kernel is a single function, a kernel compiled from
    /// ZynML source carries whatever its imports pulled in.
    module: HirModule,
    /// Resolved name `call_function_raw` looks the function up by.
    fn_name: &'static str,
    /// First operand buffer (`a`, signed i8 view).
    a_buf: Vec<i8>,
    /// Second operand buffer (`b`, u8 values in an i8 bit pattern).
    b_buf: Vec<i8>,
    /// Element count (`n_bytes` argument), a multiple of 16.
    n: usize,
    /// Ops executed in one pass — for deriving the throughput metric.
    ops: u64,
    /// Expected result value (Rust reference) every tier's result must
    /// equal — a miscompile that returns garbage is a reported failure.
    expected: i64,
}

impl RealKernel {
    /// The interp/tiered call arguments: two host buffer pointers and the
    /// element count. `Op::VLoad`/`Op::Gep` deref these real pointers.
    fn call_args(&self) -> Vec<ZyntaxValue> {
        vec![
            ZyntaxValue::Pointer(self.a_buf.as_ptr() as *mut u8),
            ZyntaxValue::Pointer(self.b_buf.as_ptr() as *mut u8),
            ZyntaxValue::Int(self.n as i64),
        ]
    }

    /// The module the runtime compiles and resolves [`Self::fn_name`] in.
    fn module(&self) -> HirModule {
        self.module.clone()
    }

    /// The kernel function itself. A source-compiled module also carries
    /// whatever its imports pulled in, so the kernel is found by name
    /// rather than assumed to be the only function present.
    fn func(&self) -> &HirFunction {
        self.module
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some(self.fn_name))
            .unwrap_or_else(|| panic!("kernel `{}` missing from its module", self.fn_name))
    }
}

/// The tiers `qdot_u8i8_baseline` is measured on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    /// Pure BC interpreter, no JIT.
    Interp,
    /// BC interp → Cranelift tier-up (production path).
    Tiered,
    /// BC interp → Cranelift → LLVM, the full ladder. Escalates past
    /// Cranelift only under `--features llvm-backend`.
    TieredLlvm,
    /// Direct `CraneliftBackend` compile+finalize (pure codegen).
    Cranelift,
}

impl Tier {
    fn key(self) -> &'static str {
        match self {
            Tier::Interp => "interp",
            Tier::Tiered => "tiered",
            Tier::TieredLlvm => "tiered-llvm",
            Tier::Cranelift => "cranelift",
        }
    }

    /// Whether this tier drives the interp→Cranelift[→LLVM] JIT ladder.
    fn installs_jit(self) -> bool {
        matches!(self, Tier::Tiered | Tier::TieredLlvm)
    }
}

const TIERS: [Tier; 4] = [
    Tier::Interp,
    Tier::Tiered,
    Tier::TieredLlvm,
    Tier::Cranelift,
];

/// `TieredConfig` for the JIT tiers — a copy of
/// `bench_runner::jit_tier_config`. `warm_threshold = 0` fires the
/// Cranelift dispatch on the first interp tick. `hot_threshold` gates the
/// LLVM side-channel: `1` (when `install_llvm`) escalates to LLVM after one
/// call; `u32::MAX` parks it so only Cranelift ever fires. Beadie's tier-2
/// auto promotion is disabled so the LLVM side-channel isn't raced. LLVM
/// only actually compiles when the `llvm-backend` cargo feature is built
/// in; otherwise `hot_threshold = 1` is inert and the tier stays Cranelift.
fn jit_tier_config(install_llvm: bool) -> TieredConfig {
    let mut cfg = TieredConfig::default();
    cfg.verbosity = std::env::var("ML_BENCH_VERBOSE").is_ok() as u8;
    cfg.profile_config = ProfileConfig {
        warm_threshold: 0,
        hot_threshold: if install_llvm { 1 } else { u32::MAX as u64 },
        ..ProfileConfig::default()
    };
    cfg.enable_background_optimization = false;
    cfg
}

/// The phase-tagged kernel registry. Phase 0 is real; Phases 1–4 are
/// `pending: true` scaffolds citing `docs/ML_ROADMAP.md`.
fn registry() -> Vec<Kernel> {
    vec![
        // ---- Phase 0 — language enablers (the only real kernel) ----
        Kernel {
            name: "qdot_u8i8_baseline",
            phase: 0,
            pending: false,
            metric: Metric::Gops,
            baseline: "auto-vectorized u8xi8 dot (today)",
            note: "widening-accumulate u8xi8 dot -> VectorDot (vpdpbusd/sdot/i32x4.dot/fallback)",
            builder: Some(build_qdot_u8i8_baseline),
        },
        // qdot via the explicit `dot_u8i8` intrinsic — Phase 0 deliverable
        // (@intrinsic tag -> emitter table). Target: >= baseline GOP/s.
        Kernel {
            name: "qdot_u8i8_intrinsic",
            phase: 0,
            pending: false,
            metric: Metric::Gops,
            baseline: "qdot_u8i8_baseline",
            note: "roadmap Phase 0: the same dot written in pure ZynML",
            builder: Some(build_qdot_u8i8_intrinsic),
        },
        Kernel {
            name: "fma_intrinsic",
            phase: 0,
            pending: true,
            metric: Metric::Gflops,
            baseline: "a*b+c auto-fusion",
            note: "roadmap Phase 0: explicit fma intrinsic vs auto-fused a*b+c",
            builder: None,
        },
        // ---- Phase 1 — dense Tensor<T> (memory-bound -> GB/s) ----
        Kernel {
            name: "elementwise_add_f32",
            phase: 1,
            pending: true,
            metric: Metric::Gbps,
            baseline: "ZRTL $Tensor$add",
            note: "roadmap Phase 1: elementwise add, saturate bandwidth",
            builder: None,
        },
        Kernel {
            name: "softmax_f32",
            phase: 1,
            pending: true,
            metric: Metric::Gbps,
            baseline: "% of bandwidth",
            note: "roadmap Phase 1: numerically stable softmax",
            builder: None,
        },
        Kernel {
            name: "rmsnorm_f32",
            phase: 1,
            pending: true,
            metric: Metric::Gbps,
            baseline: "% of bandwidth",
            note: "roadmap Phase 1: RMSNorm",
            builder: None,
        },
        Kernel {
            name: "gelu_f32",
            phase: 1,
            pending: true,
            metric: Metric::Gbps,
            baseline: "% of bandwidth",
            note: "roadmap Phase 1: GELU activation",
            builder: None,
        },
        // ---- Phase 2 — dense matmul (compute-bound -> GFLOP/s) ----
        Kernel {
            name: "gemm_f32",
            phase: 2,
            pending: true,
            metric: Metric::Gflops,
            baseline: "zrtl_simd $SIMD$gemm_f32",
            note: "roadmap Phase 2: tiled/register-blocked f32 GEMM over vload/fma/vstore",
            builder: None,
        },
        Kernel {
            name: "gemm_f32_fused",
            phase: 2,
            pending: true,
            metric: Metric::Gflops,
            baseline: "gemm_f32 (unfused + separate epilogue)",
            note: "roadmap Phase 2: GEMM + fused bias+GELU epilogue, fusion is free",
            builder: None,
        },
        // ---- Phase 3 — quantized (int8 -> GOP/s) ----
        Kernel {
            name: "qgemm_int8",
            phase: 3,
            pending: true,
            metric: Metric::Gops,
            baseline: "gemm_f32",
            note: "roadmap Phase 3: int8 GEMM via dot_u8i8 -> vpdpbusd, per-channel scales",
            builder: None,
        },
        Kernel {
            name: "quantize_dequantize_f32",
            phase: 3,
            pending: true,
            metric: Metric::Gbps,
            baseline: "% of bandwidth",
            note: "roadmap Phase 3: quantize/dequantize throughput (Q8_0, Q4_K)",
            builder: None,
        },
        // ---- Phase 4 — attention + runnable model ----
        Kernel {
            name: "attention",
            phase: 4,
            pending: true,
            metric: Metric::Gflops,
            baseline: "tokens/s target per model",
            note: "roadmap Phase 4: QK^T.scale.softmax.V, KV-cache, causal mask, flash tiling",
            builder: None,
        },
        Kernel {
            name: "transformer_block_e2e",
            phase: 4,
            pending: true,
            metric: Metric::Gflops,
            baseline: "captured known-good tiny model",
            note: "roadmap Phase 4: small quantized model end-to-end (prefill + decode)",
            builder: None,
        },
    ]
}

// =========================================================================
// Phase 0 real kernel: hand-built u8xi8 quantized dot
// -------------------------------------------------------------------------

/// Build `fn qdot(a: *i8, b: *i8, n_bytes: i64) -> i32` — a loop over
/// the two byte buffers in i8x16 chunks that accumulates into an i32x4
/// via `VectorDot { rhs_unsigned: true }` (the u8×i8 form) and returns
/// the horizontal sum. This is the exact backend `VectorDot` lowering
/// the auto-vectorizer produces for the widening-accumulate loop
/// `for i { acc += widen(a[i]) * widen(b[i]) }`, so it folds to
/// `vpdpbusd`/`sdot`/`i32x4.dot`/the widening fallback per host.
fn build_qdot_u8i8_baseline() -> RealKernel {
    let ptr_i8 = || HirType::Ptr(Box::new(HirType::I8));
    let i8x16 = HirType::Vector(Box::new(HirType::I8), 16);
    let i32x4 = HirType::Vector(Box::new(HirType::I32), 4);

    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: InternedString::new_global("a"),
                ty: ptr_i8(),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: InternedString::new_global("b"),
                ty: ptr_i8(),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: InternedString::new_global("n_bytes"),
                ty: HirType::I64,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(InternedString::new_global("qdot_u8i8"), sig);
    func.calling_convention = CallingConvention::C;

    let a_ptr = func.create_value(ptr_i8(), HirValueKind::Parameter(0));
    let b_ptr = func.create_value(ptr_i8(), HirValueKind::Parameter(1));
    let n_bytes = func.create_value(HirType::I64, HirValueKind::Parameter(2));

    // Loop-carried SSA values.
    let zero_i64 = func.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
    let sixteen = func.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(16)));
    let zero_i32 = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(0)));
    let acc0 = func.create_value(i32x4.clone(), HirValueKind::Instruction);

    let off_phi = func.create_value(HirType::I64, HirValueKind::Instruction);
    let acc_phi = func.create_value(i32x4.clone(), HirValueKind::Instruction);
    let cond = func.create_value(HirType::Bool, HirValueKind::Instruction);

    let a_chunk = func.create_value(ptr_i8(), HirValueKind::Instruction);
    let b_chunk = func.create_value(ptr_i8(), HirValueKind::Instruction);
    let av = func.create_value(i8x16.clone(), HirValueKind::Instruction);
    let bv = func.create_value(i8x16.clone(), HirValueKind::Instruction);
    let acc_next = func.create_value(i32x4.clone(), HirValueKind::Instruction);
    let off_next = func.create_value(HirType::I64, HirValueKind::Instruction);
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);

    let entry = func.entry_block;
    let header = func.create_block();
    let body = func.create_block();
    let exit = func.create_block();

    // entry: acc0 = splat_i32x4(0); goto header
    {
        let b = func.blocks.get_mut(&entry).unwrap();
        b.add_instruction(HirInstruction::VectorSplat {
            result: acc0,
            ty: i32x4.clone(),
            scalar: zero_i32,
        });
        b.set_terminator(HirTerminator::Branch { target: header });
    }

    // header: phi off, phi acc; cond = off < n_bytes; condbranch body/exit
    {
        let b = func.blocks.get_mut(&header).unwrap();
        b.phis.push(HirPhi {
            result: off_phi,
            ty: HirType::I64,
            incoming: vec![(zero_i64, entry), (off_next, body)],
        });
        b.phis.push(HirPhi {
            result: acc_phi,
            ty: i32x4.clone(),
            incoming: vec![(acc0, entry), (acc_next, body)],
        });
        b.add_instruction(HirInstruction::Binary {
            op: BinaryOp::Lt,
            result: cond,
            ty: HirType::Bool,
            left: off_phi,
            right: n_bytes,
        });
        b.set_terminator(HirTerminator::CondBranch {
            condition: cond,
            true_target: body,
            false_target: exit,
        });
    }

    // body: load a/b chunk at off; acc_next = dot(acc, a, b, rhs_unsigned);
    //       off_next = off + 16; goto header
    {
        let b = func.blocks.get_mut(&body).unwrap();
        b.add_instruction(HirInstruction::GetElementPtr {
            result: a_chunk,
            ty: HirType::I8,
            ptr: a_ptr,
            indices: vec![off_phi],
        });
        b.add_instruction(HirInstruction::GetElementPtr {
            result: b_chunk,
            ty: HirType::I8,
            ptr: b_ptr,
            indices: vec![off_phi],
        });
        b.add_instruction(HirInstruction::VectorLoad {
            result: av,
            ty: i8x16.clone(),
            ptr: a_chunk,
            align: 1,
        });
        b.add_instruction(HirInstruction::VectorLoad {
            result: bv,
            ty: i8x16.clone(),
            ptr: b_chunk,
            align: 1,
        });
        b.add_instruction(HirInstruction::VectorDot {
            result: acc_next,
            acc: acc_phi,
            a: av,
            b: bv,
            rhs_i7: false,
            rhs_unsigned: true,
        });
        b.add_instruction(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: off_next,
            ty: HirType::I64,
            left: off_phi,
            right: sixteen,
        });
        b.set_terminator(HirTerminator::Branch { target: header });
    }

    // exit: return horizontal_reduce_add(acc)
    {
        let b = func.blocks.get_mut(&exit).unwrap();
        b.add_instruction(HirInstruction::VectorHorizontalReduce {
            result,
            ty: HirType::I32,
            vector: acc_phi,
            op: BinaryOp::Add,
        });
        b.set_terminator(HirTerminator::Return {
            values: vec![result],
        });
    }

    // Working set: two byte buffers of QDOT_N. Fill with 1s so the i32
    // lanes never overflow (QDOT_N < i32::MAX) and the reference result
    // is simply QDOT_N — the bench measures throughput, not correctness
    // (the ml_harness_tests own the u8×i8 numerical check, including a
    // b>127 unsigned case).
    let n = QDOT_N;
    let mut module = HirModule::new(InternedString::new_global("ml_bench_mod"));
    module.add_function(func);
    RealKernel {
        module,
        fn_name: "qdot_u8i8",
        a_buf: vec![1i8; n],
        b_buf: vec![1i8; n], // bit pattern 0x01 == u8 1
        n,
        ops: 2 * n as u64,  // one multiply + one add per element
        expected: n as i64, // Σ 1*1 over n elements
    }
}

/// The same quantized dot, written in **pure ZynML** instead of
/// hand-built HIR — the Phase 0 exit-gate kernel. It uses only the
/// language surface: a typed pointer, whole-vector loads, the quantized
/// dot method, and a horizontal reduce. Measuring it against
/// `qdot_u8i8_baseline` answers the phase's question: does the explicit
/// intrinsic path match the hand-built one?
fn build_qdot_u8i8_intrinsic() -> RealKernel {
    const SRC: &str = r#"
import prelude

def qdot_u8i8_src(a: Ptr<i8>, b: Ptr<i8>, n_bytes: i64): i32 {
    let mut acc: i32x4 = i32x4::splat(0)
    let mut off: i64 = 0
    while off < n_bytes {
        let av: i8x16 = vload_i8x16(a + off)
        let bv: i8x16 = vload_i8x16(b + off)
        acc = acc.dot_u8i8(av, bv)
        off = off + 16
    }
    return acc.sum()
}
"#;

    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(SRC, "<qdot_u8i8_intrinsic>")
        .expect("kernel should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
        _ => Ok(None),
    }));
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt
        .lower_typed_program(program, builtins)
        .expect("kernel should lower to HIR");

    let n = QDOT_N;
    RealKernel {
        module,
        fn_name: "qdot_u8i8_src",
        a_buf: vec![1i8; n],
        b_buf: vec![1i8; n], // bit pattern 0x01 == u8 1
        n,
        ops: 2 * n as u64,
        expected: n as i64,
    }
}

// =========================================================================
// Runner
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KernelResult {
    phase: u8,
    metric: String,
    baseline: String,
    note: String,
    /// True when this kernel was skipped as a pending scaffold.
    pending: bool,
    /// Populated only for real kernels that ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    tiers: Option<BTreeMap<String, TierResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TierResult {
    compile_ms: f64,
    exec_ms: f64,
    /// Throughput in the kernel's metric unit (GOP/s, GFLOP/s, GB/s).
    throughput: f64,
    /// Leverage: throughput speedup versus the `interp` tier (1.0 for
    /// interp itself). Quantifies what promoting to this tier buys.
    #[serde(skip_serializing_if = "Option::is_none")]
    speedup_vs_interp: Option<f64>,
    /// Result value the finalized function returned (correctness pin).
    result: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MlSuite {
    #[serde(flatten)]
    kernels: BTreeMap<String, KernelResult>,
    meta: Meta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Meta {
    arch: String,
    os: String,
    runs: usize,
    commit: String,
}

fn main() {
    let mut out_path: Option<PathBuf> = None;
    let mut runs = RUNS;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out_path = args.next().map(PathBuf::from),
            "--runs" => {
                if let Some(v) = args.next().and_then(|s| s.parse().ok()) {
                    runs = v;
                }
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: ml_bench [--out <path>] [--runs <n>]\n\
                     Defaults: out = website/benchmark/ml_results.json, runs = {RUNS}"
                );
                return;
            }
            other => eprintln!("warning: ignoring unknown arg {other:?}"),
        }
    }

    let default_out =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../website/benchmark/ml_results.json");
    let out_path = out_path.unwrap_or(default_out);

    let mut suite = MlSuite {
        kernels: BTreeMap::new(),
        meta: Meta {
            arch: env::consts::ARCH.to_string(),
            os: env::consts::OS.to_string(),
            runs,
            commit: git_short_sha(),
        },
    };

    let mut value_mismatches: Vec<String> = Vec::new();

    for kernel in registry() {
        if kernel.pending {
            eprintln!(
                "  {:<26} PENDING (Phase {})  — {}",
                kernel.name, kernel.phase, kernel.note
            );
            suite.kernels.insert(
                kernel.name.to_string(),
                KernelResult {
                    phase: kernel.phase,
                    metric: kernel.metric.unit().to_string(),
                    baseline: kernel.baseline.to_string(),
                    note: kernel.note.to_string(),
                    pending: true,
                    tiers: None,
                },
            );
            continue;
        }

        eprintln!("==> kernel {} (Phase {})", kernel.name, kernel.phase);
        let builder = kernel
            .builder
            .expect("non-pending kernel must have a builder");
        let expected = builder().expected;

        // Measure every tier first, then fill in leverage relative to the
        // interp baseline before printing/serializing — so each row can
        // report the speedup its promotion buys.
        let mut measured: Vec<(Tier, TierResult)> = Vec::with_capacity(TIERS.len());
        for tier in TIERS {
            let res = match measure_tier(builder, tier, runs) {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("    {:<11} FAILED — {e}", tier.key());
                    TierResult {
                        compile_ms: 0.0,
                        exec_ms: 0.0,
                        throughput: 0.0,
                        speedup_vs_interp: None,
                        result: 0,
                        error: Some(e),
                    }
                }
            };
            measured.push((tier, res));
        }

        // The interp tier is the leverage baseline (1.0x).
        let interp_tp = measured
            .iter()
            .find(|(t, _)| *t == Tier::Interp)
            .map(|(_, r)| r.throughput)
            .filter(|tp| *tp > 0.0);

        let mut tiers: BTreeMap<String, TierResult> = BTreeMap::new();
        for (tier, mut res) in measured {
            let key = tier.key();
            if res.error.is_none() {
                res.speedup_vs_interp = interp_tp.map(|base| res.throughput / base);
                let leverage = res
                    .speedup_vs_interp
                    .map(|s| format!("{s:>8.1}x vs interp"))
                    .unwrap_or_else(|| "        —".to_string());
                eprintln!(
                    "    {:<11} compile={:>8.2}ms exec={:>9.3}ms  {:>9.2} {}  {}  -> {}",
                    key,
                    res.compile_ms,
                    res.exec_ms,
                    res.throughput,
                    kernel.metric.unit(),
                    leverage,
                    res.result,
                );
                // Correctness pin: every tier must return the Rust
                // reference. A JIT-dispatch marshalling bug that returns
                // garbage is surfaced here, never papered over.
                if res.result != expected {
                    eprintln!(
                        "    {:<11} VALUE MISMATCH — got {}, expected {}",
                        key, res.result, expected
                    );
                    value_mismatches.push(format!(
                        "{}/{}: got {}, expected {}",
                        kernel.name, key, res.result, expected
                    ));
                }
            }
            tiers.insert(key.to_string(), res);
        }

        suite.kernels.insert(
            kernel.name.to_string(),
            KernelResult {
                phase: kernel.phase,
                metric: kernel.metric.unit().to_string(),
                baseline: kernel.baseline.to_string(),
                note: kernel.note.to_string(),
                pending: false,
                tiers: Some(tiers),
            },
        );
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|e| panic!("mkdir {parent:?}: {e}"));
    }
    let json = serde_json::to_string_pretty(&suite).expect("serialize");
    fs::write(&out_path, json).unwrap_or_else(|e| panic!("write {out_path:?}: {e}"));
    eprintln!("\nwrote {}", out_path.display());

    if !value_mismatches.is_empty() {
        eprintln!("\nVALUE MISMATCHES — ml_bench will exit non-zero:");
        for m in &value_mismatches {
            eprintln!("  {m}");
        }
        std::process::exit(2);
    }
}

/// Drive `WARMUP` + `runs` timed passes of one kernel on one tier,
/// separating compile from execute time, returning medians + the derived
/// throughput + the result (correctness pin). The three tiers share this
/// machinery; only [`run_once`] differs per tier.
fn measure_tier(
    builder: fn() -> RealKernel,
    tier: Tier,
    runs: usize,
) -> Result<TierResult, String> {
    let mut compile_samples = Vec::with_capacity(runs);
    let mut exec_samples = Vec::with_capacity(runs);
    let mut throughputs = Vec::with_capacity(runs);
    let mut last_result = 0i64;

    for i in 0..(WARMUP + runs) {
        let kernel = builder();
        let ops = kernel.ops;
        let (compile_ms, exec_s, result) = run_once(&kernel, tier)?;

        if i >= WARMUP {
            compile_samples.push(compile_ms);
            exec_samples.push(exec_s * 1000.0);
            // GOP/s == ops / seconds / 1e9 (same denominator for GFLOP/s and,
            // for GB/s, where callers pass byte counts as "ops").
            let g = if exec_s > 0.0 {
                ops as f64 / exec_s / 1e9
            } else {
                0.0
            };
            throughputs.push(g);
            last_result = result;
        }
    }

    Ok(TierResult {
        compile_ms: median(&mut compile_samples),
        exec_ms: median(&mut exec_samples),
        throughput: median(&mut throughputs),
        speedup_vs_interp: None, // filled once the interp baseline is known
        result: last_result,
        error: None,
    })
}

/// One full pass of `kernel` on `tier`: returns `(compile_ms, exec_s,
/// result_i64)`. `compile_ms` is the cold-path setup (backend build +
/// compile [+ JIT install]); `exec_s` is the single timed call (for the
/// tiered tier, preceded by untimed JIT warmup so the call dispatches
/// through the Cranelift-compiled code, not the cold interp loop).
fn run_once(kernel: &RealKernel, tier: Tier) -> Result<(f64, f64, i64), String> {
    match tier {
        // Direct CraneliftBackend: compile + finalize, then call the raw
        // function pointer through the C ABI.
        Tier::Cranelift => {
            // Run the same HIR optimization pipeline the runtime applies
            // inside `compile_module`. Without it this tier would compile
            // unoptimized HIR while every other tier compiles optimized
            // HIR, so the row would report the cost of skipping the
            // optimizer rather than the codegen number it is meant to
            // isolate.
            let mut module = kernel.module();
            zyntax_compiler::run_interp_safe_opts(&mut module);
            let func = module
                .functions
                .values()
                .find(|f| f.name.resolve_global().as_deref() == Some(kernel.fn_name))
                .ok_or_else(|| format!("kernel `{}` missing after opts", kernel.fn_name))?
                .clone();
            let id = func.id;
            let t0 = Instant::now();
            let mut backend = CraneliftBackend::new().map_err(|e| format!("backend: {e:?}"))?;
            backend
                .compile_function(id, &func)
                .map_err(|e| format!("compile: {e:?}"))?;
            backend
                .finalize_definitions()
                .map_err(|e| format!("finalize: {e:?}"))?;
            let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let ptr = backend
                .get_function_ptr(id)
                .ok_or_else(|| "get_function_ptr returned None".to_string())?;
            let f = unsafe {
                std::mem::transmute::<
                    *const u8,
                    unsafe extern "C" fn(*const i8, *const i8, i64) -> i32,
                >(ptr)
            };
            // Untimed warmup, matching what the tiered tiers get. Without
            // it this row times the first touch of a multi-megabyte
            // working set straight from memory, while the tiered rows —
            // which warm up before measuring — report steady state. That
            // gap is cache residency, not codegen, and it made direct
            // codegen look several times worse than it is.
            for _ in 0..JIT_TIER_WARMUP_CALLS {
                unsafe {
                    f(
                        kernel.a_buf.as_ptr(),
                        kernel.b_buf.as_ptr(),
                        kernel.n as i64,
                    )
                };
            }
            let t1 = Instant::now();
            let r = unsafe {
                f(
                    kernel.a_buf.as_ptr(),
                    kernel.b_buf.as_ptr(),
                    kernel.n as i64,
                )
            };
            let exec_s = t1.elapsed().as_secs_f64();
            Ok((compile_ms, exec_s, r as i64))
        }

        // BC interpreter, optionally with the Cranelift[→LLVM] ladder.
        Tier::Interp | Tier::Tiered | Tier::TieredLlvm => {
            let module = kernel.module();
            let t0 = Instant::now();
            let mut rt = ZyntaxRuntime::new().map_err(|e| format!("rt: {e:?}"))?;
            rt.compile_module(&module)
                .map_err(|e| format!("compile_module: {e:?}"))?;
            if tier.installs_jit() {
                rt.install_interp_jit_with(jit_tier_config(tier == Tier::TieredLlvm))
                    .map_err(|e| format!("install_interp_jit: {e:?}"))?;
            }
            let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Untimed warmup drives the interp past the warm/hot threshold
            // so the background Cranelift (then, for tiered-llvm, LLVM)
            // compile lands and the timed call dispatches to the JIT'd code.
            if tier.installs_jit() {
                for _ in 0..JIT_TIER_WARMUP_CALLS {
                    rt.call_function_raw(kernel.fn_name, kernel.call_args())
                        .map_err(|e| format!("jit warmup: {e:?}"))?;
                }
            }

            let t1 = Instant::now();
            let ret = rt
                .call_function_raw(kernel.fn_name, kernel.call_args())
                .map_err(|e| format!("call: {e:?}"))?;
            let exec_s = t1.elapsed().as_secs_f64();
            let result =
                value_to_i64(&ret).ok_or_else(|| format!("non-integer return: {ret:?}"))?;
            Ok((compile_ms, exec_s, result))
        }
    }
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
        .unwrap_or_else(|| "-".to_string())
}
