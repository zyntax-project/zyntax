# ZynML ML Roadmap — Pure-ZynML Tensor, Quantized Tensors, Fused Matmul

Status: planning. Owner: ZynML core. Anchor workload: **quantized transformer
inference**.

## North Star

ZynML owns its ML domain as a first-class part of the language. `Tensor`,
`QuantizedTensor`, matmul, attention, and the norm/activation/fusion machinery
are written in **pure ZynML**, lowered through the inline SIMD path so every
kernel is **optimizer-visible** — FMA fusion, elementwise fusion, and the VNNI
`vpdpbusd` / SDOT / `i32x4.dot` lowering reach *into* the kernel instead of
stopping at an FFI wall. `zrtl_tensor` stays as an alternate backend / escape
hatch for embedders; the pure-ZynML `Tensor` is the language's own and the
default going forward.

The bar is **not** matching today's (thin, F32-only, matmul-less) ZRTL Tensor.
The bar is **what real inference needs**: dense + quantized GEMM, attention with
a KV-cache, stable softmax + norms, fused epilogues, and int8/int4 weights.

## Principles

- **Inference-first.** Training layers on later — the forward ops become the
  primitives autograd differentiates. No gradient tape in the first passes.
- **Static-rank primary, dynamic fallback.** `Tensor<T, const RANK>` for the
  shapes that dominate inference (best codegen); a rank-erased `DynTensor`
  escape for genuinely dynamic cases.
- **No FFI wall.** Every hot kernel is pure ZynML so the HIR optimizers see
  through it. Fast instructions are reached via **explicit intrinsics**
  (`@intrinsic` tag → emitter table), not by hoping the auto-vectorizer's loop
  template matches.
- **All four backends.** cranelift, LLVM, wasm, interpreter — parity per phase.
- **Every phase gated.** No phase is "done" without (1) a micro-bench hitting
  its perf target and (2) a test harness proving numerical correctness **and**
  codegen (the fast instruction is actually emitted).

## Foundation already in place (do not rebuild)

- First-class `f32x4` / `i8x16` / `i32x4` vector types with full
  elementwise / reduce / min-max / sqrt / lane ops on all four backends.
- Auto-vectorization of plain scalar loops → SIMD, **including** the widening
  `u8×i8 → vpdpbusd` / SDOT / `i32x4.dot` quantized dot and `a*b+c → FMA`.
- Generic structs + trait-based operator overloading; bi-modal memory menu
  (drop-site / GC / ownership).
- Const-generics **backend** support (`MonomorphizationKey.const_args`,
  `ConstEvaluator`, `Type::Named.const_args`) — only the parser surface is
  missing.

## Harness architecture (shared across phases)

Two harnesses, both run **per backend** (cranelift / LLVM / wasm / interp):

- **`ml_bench`** — micro-benchmark runner (mirrors `crates/zynml/examples/bench_runner.rs`).
  Registers ML kernels, measures **GFLOP/s** (compute-bound: GEMM, attention),
  **GB/s** (memory-bound: elementwise, norms), or **GOP/s** (int8), separating
  compile time from execute time (median-of-N). Emits JSON. Each kernel carries
  a **baseline** (ZRTL-parity where it exists, else % of roofline peak) and a
  **target**.
- **ML test harness** — every kernel gets two checks:
  1. **Numerical correctness** — compile + run, compare against a reference
     (Rust or a captured known-good) within a stated tolerance.
  2. **Codegen verification** — compile, capture IR / disasm per backend, assert
     the fast instruction is emitted (`vpdpbusd` on VNNI x86, `sdot` on aarch64,
     `i32x4.dot` on wasm, `fma`, or the documented fallback). Reuses the
     established patterns: `set_capture_ir` + disasm assert (cranelift),
     IR-text assert (LLVM), `validate_full` (wasm), and NUC/Intel-SDE execution
     for VNNI hardware.

Layout: `docs/ML_ROADMAP.md` (this file) is the spine; benches live under the
`ml_bench` harness; per-phase tests live in a dedicated ML test module. Each
phase adds its kernels to both; a kernel is not merged until both are green.

---

## Phase 0 — Language enablers (the keystone)

Nothing else can be pure ZynML until these land. Contained, high-leverage,
mostly parser + intrinsic-table work (backend is ready).

**Deliverables**
- **Const-generic surface syntax** — wire the parser to populate `const_args`
  (currently hardcoded `[]` at every production); the backend already consumes
  it. Unblocks `Tensor<T, const RANK: usize>` and compile-time shapes.
- **Explicit SIMD intrinsics** via the `@intrinsic("tag")` + tag→emitter table
  pattern (built-in traits intercepted at HIR, **not** extern FFI):
  `dot_u8i8` / `dot_i8i8` (+ wire the latent `rhs_i7` producer), `fma`,
  `vload` / `vstore`, `hreduce`.
- **Typed aligned buffer** — `Ptr<T>` + `alloc<T>(n, align)` / `free` in source,
  plus cross-block drop-site (or `Drop`/RAII) so a Tensor owns a real
  contiguous typed buffer, not ZRTL's untyped `List<T>`.

**Micro-benches**
- `qdot_u8i8_baseline` — the auto-vectorized `u8×i8` dot that **works today**
  (establishes the number the explicit intrinsic must match).
- `qdot_u8i8_intrinsic` — the same via `dot_u8i8`; target: **≥ baseline
  GOP/s** and deterministic VNNI emission.
- `fma_intrinsic` vs `a*b+c` auto-fusion — parity.

**Test harness**
- Codegen: `dot_u8i8` → `vpdpbusd` (x86 VNNI), `sdot` (aarch64),
  `i32x4.dot_i8x16` (wasm), widening fallback (interp / non-VNNI); `fma` fused;
  `vload`/`vstore` → aligned SIMD load/store; `Tensor<T, const N>`
  monomorphizes to distinct instantiations.
- Correctness: `dot_u8i8` == reference (incl. a `b > 127` case that
  distinguishes signed×unsigned); `alloc`/`free` round-trip with no leak.

**Exit gate:** one **int8 quantized-matmul microkernel in pure ZynML** compiles,
runs, and lowers to `vpdpbusd`/SDOT/wasm-dot across all four backends, at ≥ the
auto-vectorized baseline. If clean, the architecture is validated.

---

## Phase 1 — Dense `Tensor<T>`

**Deliverables**
- `Tensor<T, const RANK>` + `DynTensor`: shape/strides, views, reshape,
  transpose, contiguous, indexing.
- Broadcasting.
- Elementwise (+ bias / residual add), reduce (sum/mean/max), **stable
  softmax**, **RMSNorm / LayerNorm**, **GELU / SiLU / ReLU**.

**Micro-benches** (memory-bound → GB/s)
- `elementwise_add_f32` vs ZRTL `$Tensor$add` (target: match/beat, saturate
  bandwidth); `softmax_f32`; `rmsnorm_f32`; `gelu_f32`.

**Test harness**
- Correctness: each op vs a reference (incl. broadcasting cases, numerical
  stability for softmax/norm).
- Codegen: elementwise loop → `VectorLoad`/`VectorStore`/binop; reduce →
  horizontal reduce; no `$Tensor$*` FFI symbol in the pure-ZynML path.

---

## Phase 2 — Dense matmul (GEMM)

**Deliverables**
- Tiled / cache-blocked f32 GEMM with a register-blocked microkernel over
  `vload`/`fma`/`vstore`; bf16 storage path.
- **Fused epilogue** (bias + activation) so FMA + elementwise fusion fire
  inside the kernel.

**Micro-benches** (compute-bound → GFLOP/s)
- `gemm_f32` sweep over M/N/K vs `zrtl_simd $SIMD$gemm_f32` (the existing
  baseline) and a **% of peak** target; roofline plot.
- `gemm_f32_fused` (+ bias + GELU) — verify no perf loss vs unfused + separate
  epilogue, i.e. fusion is free.

**Test harness**
- Correctness: GEMM vs reference within tolerance; fused == matmul then bias
  then activation.
- Codegen: inner loop → FMA; zero FFI calls; fusion confirmed in HIR.

---

## Phase 3 — Quantized

**Deliverables**
- `QuantizedTensor`: block-quant layout (Q8_0 first, then Q4_K-style grouped
  int4) — grouped quants + per-group `scale` / `zero_point`.
- quantize / dequantize; **int8 (then int4) GEMM** via `dot_u8i8` → `vpdpbusd`;
  per-channel scales; dynamic activation quantization.

**Micro-benches** (int8 → GOP/s)
- `qgemm_int8` vs `gemm_f32` (speedup) and vs a reference int8 impl; `quantize`
  / `dequantize` throughput.

**Test harness**
- Correctness: quantized matmul vs f32 within quantization error; quant/dequant
  round-trip; per-group scale application.
- Codegen: int8 GEMM inner → `vpdpbusd` (x86 VNNI) / `sdot` (aarch64) /
  `i32x4.dot` (wasm) / widening fallback; **execution on VNNI hardware** (NUC
  Alder Lake) and under Intel SDE.

---

## Phase 4 — Attention + a runnable model

**Deliverables**
- Attention: QKᵀ · scale · softmax · V, KV-cache, causal mask; flash-style
  tiling (no full score matrix materialized).
- Assemble a transformer block; run a **small quantized model end-to-end** in
  pure ZynML.

**Micro-benches**
- `attention` over a sequence-length sweep (tokens/s); end-to-end model
  tokens/s (prefill + decode).

**Test harness**
- Correctness: attention vs reference; end-to-end logits vs a captured
  known-good for a tiny model.
- Codegen: attention uses the fused / quantized paths; no FFI in the hot loop.

---

## Baselines & targets

| kernel | baseline | target |
| --- | --- | --- |
| `qdot_u8i8` | auto-vec dot (today) | ≥ baseline, VNNI emitted |
| `elementwise_add_f32` | ZRTL `$Tensor$add` | ≥ ZRTL, ≈ bandwidth |
| `gemm_f32` | `zrtl_simd $SIMD$gemm_f32` | ≥ ZRTL, % of peak (rising) |
| `qgemm_int8` | `gemm_f32` | measurable speedup |
| `attention` | — | tokens/s target per model |

(Absolute % -of-peak targets are ratcheted up phase over phase, not fixed here.)

## Cross-cutting

- **All-backends parity** is a merge gate every phase (cranelift / LLVM / wasm /
  interp), per the project rule.
- **Perf regression gate:** the `ml_bench` numbers are tracked; a kernel that
  regresses fails the gate.
- **ZRTL coexistence:** same-surface API; embedders can select the ZRTL backend,
  but the pure-ZynML Tensor is optimizer-visible and default.
- This file is the durable spine — update it as phases land; each phase's PRs
  reference the phase section.
