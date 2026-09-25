# ZynML GPU Compute System

**Status**: CPU SIMD half partly built; GPU half not built (checked 2026-09-25)
**Priority**: High (after core ZynML stabilization)
**Complexity**: Very High

## Status and direction (September 2026)

Checked against the code on 2026-09-25 (HEAD `a900deba`). This document is the kernel surface, the Apple (Metal) design and the benchmark plan. The NVIDIA design is [GPU_AOT_ARCHITECTURE.md](../GPU_AOT_ARCHITECTURE.md), which carries the same decisions. Sections below that no longer hold carry a **Superseded** or **Status** note in place.

### Built

- **Grammar.** `compute(args) @modifiers { block }` (`compute_expr` in `crates/zynml/ml.zyn`), with each modifier a general annotation, and the in-body directive `@kernel <identifier>` plus `yield`. The arguments are the kernel's inputs.
- **`@kernel elementwise`, one shape.** A body of exactly `for i in r { arr[i] = arr[i] OP scalar }` over a compute argument lowers (`emit_elementwise_simd_loop`, `crates/compiler/src/ssa.rs`) to a 4-lane vector load, splat, op and store loop with a scalar remainder.
- **`@kernel reduce`, partially.** With a `yield` directly in the body the result is the last yielded value; nothing accumulates, and there is no operator slot (`reduce(+)` cannot be written).
- **First-class SIMD.** `HirType::Vector` and the HIR vector instructions lower natively on Cranelift, LLVM, wasm and the interpreter; `target_vector.rs` gives the host width, and the vectorization passes cover ordinary loops.
- **Parallel loops**, behind `ZYNTAX_PARALLEL_LOOPS=1`, not yet tied to `compute()`.
- **Accelerate GEMM.** `zrtl_tensor`'s `tensor_matmul_2d` calls Accelerate `cblas_sgemm` on Apple, a portable loop elsewhere.

### Not built

- GPU code generation and GPU runtimes of every kind: no MSL, PTX, SPIR-V or WGSL, no Metal, CUDA, Vulkan or WebGPU dependency.
- Kernel type checking (the type checker gives `compute` a fresh type variable), the `@device`, `@workgroup` and `@kernel(x)` modifiers (parsed and never read), device management, async compute and the memory API.
- Every other body and kernel kind (`matmul`, `conv2d`, `fused`, `attention`, `out[i] = f(x[i])`, multi-input, broadcast, a `yield` inside a loop) is not lowered as a kernel. A body not marked `@kernel elementwise` with a `yield` directly in it returns the last yielded value; every other one lowers to a call of `$Zyntax$compute`, which is not defined. Both are bugs, tracked in git-bug, and become compile errors.
- The `zyntax` module in Python and Lua.

### Decided direction

1. **Kernels are a Zyntax capability.** See "The `zyntax` kernel surface" below. ZynML reaches kernels natively; every other frontend through a `zyntax` module.
2. **Kernels are a typed subset.** A kernel body types to fixed-width scalars, vectors and buffers. Dynamic code inside a kernel is a compile error that names what is unsupported; it never falls back to a runtime dispatch or to boxed values.
3. **HIR is the kernel IR.** `compute()` lowers straight to HIR, which already carries first-class vectors on all four CPU backends. There is no separate Compute IR (the CIR section below is superseded).
4. **Metal is the first GPU backend,** because the Mac (M1 Pro) is the only GPU available. Design in "Metal Backend" below: MSL generated from HIR, compiled at run time by the Metal framework, which needs no Xcode, Metal toolchain or C compiler on the user's machine.
5. **GEMM-shaped kernels on Apple dispatch to system accelerators,** following the `zrtl_tensor` decision that FFI exists only for hardware generated code cannot reach: Accelerate (AMX) for small and medium matrices, MPS for large ones. Elementwise, reduction and fused kernels stay generated code.
6. **NVIDIA** goes HIR, LLVM IR, in-process NVPTX, PTX, loaded through `cudarc`, with a CUDA Tile IR track for tile kernels later (GPU_AOT_ARCHITECTURE.md). No NVIDIA GPU is available to test on, so it follows Metal.
7. **The portable path (Vulkan, WebGPU) is deferred.** When it returns, HIR lowers to naga IR, one emitter for SPIR-V, WGSL and MSL, and Linux runs Vulkan through `ash`.
8. **Benchmarks** replace the old target table: see "Performance Targets".

### The `zyntax` kernel surface

Fibers, effects, handlers and kernels are Zyntax capabilities, and each frontend exposes them through a `zyntax` module in its own idiom:

- **ZynML**: natively, `compute(args) @kernel(...) @device(...) { ... }` and the in-body `@kernel` directive.
- **Python (zypy)**: `from zyntax import kernel`, then `@kernel` on a function whose body is the typed subset. Surfaces only zypy provides need no CPython support.
- **Lua**: a `zyntax` table (`zyntax.kernel(fn)`).

The module is opt-in. All three hand the compiler the same thing: a typed kernel body lowered to HIR, then to a CPU SIMD loop or a GPU backend by `@device`.

## Overview

The `compute()` construct in ZynML provides a unified way to express parallel computations that can be dispatched to:
- **GPU** (CUDA, Metal, Vulkan Compute, WebGPU)
- **SIMD CPU** (AVX2, AVX-512, NEON)
- **Accelerators** (TPU, NPU, future hardware)

The goal is that the same kernel code works across all backends. **Status:** only CPU SIMD exists, for the shapes listed in the status section. TPU and NPU backends are not planned; on Apple the neural hardware is reached through system frameworks, per the `zrtl_tensor` decision.

## Design Goals

1. **Single Source** - Write once, run on any device
2. **Explicit Parallelism** - Clear parallel structure, no magic
3. **Composable** - Kernels can be fused automatically
4. **Type Safe** - Catch dimension errors at compile time
5. **Debuggable** - Run a kernel on the CPU for debugging by choosing `@device("cpu")`; never a silent fallback
6. **Performant** - Match hand-written CUDA/Metal performance

## Syntax Design

**Status:** this section is the target syntax. Today the grammar takes `compute(args) @annotation... { block }` and, inside the block, `@kernel <identifier>` with no parameters, so `reduce(+)`, `reduce(max, axis=1)`, `@shared`, `@broadcast` and `@tile` have no slot yet, and there is no implicit `out`. Only the in-place elementwise shape lowers to a kernel. A body not marked `@kernel elementwise` with a `yield` directly in it returns the last yielded value; any other body calls an undefined runtime function. Both are tracked bugs; until those forms are built they will be compile errors.

### Basic Compute Expression

```zynml
// Compute expression syntax
let output = compute(input1, input2, ...) [@device(device)] [@async] {
    @kernel kernel_type [kernel_params]
    [@workgroup(x, y, z)]
    [@shared(size)]

    // Kernel body
    for index_vars in ranges:
        // Computation
        out[indices] = expression
}
```

### Kernel Types

```zynml
// === Element-wise Operations ===
// Automatically parallelized per-element

let y = compute(x) {
    @kernel elementwise
    for i in 0..len:
        out[i] = relu(x[i])
}

// With multiple inputs
let z = compute(x, y) {
    @kernel elementwise
    for i in 0..len:
        out[i] = x[i] * y[i] + 1.0
}

// Broadcasting supported
let scaled = compute(matrix, vector) {
    @kernel elementwise
    @broadcast(vector, axis=1)
    for i in 0..rows, j in 0..cols:
        out[i, j] = matrix[i, j] * vector[j]
}


// === Reduction Operations ===
// Parallel reduction with specified operator

let sum = compute(data) {
    @kernel reduce(+)
    for i in 0..len:
        yield data[i]
}

let max_val = compute(data) {
    @kernel reduce(max)
    for i in 0..len:
        yield data[i]
}

// Reduction with transformation
let norm = compute(data) {
    @kernel reduce(+)
    for i in 0..len:
        yield data[i] * data[i]
} |> sqrt()

// Reduction along axis
let row_sums = compute(matrix) {
    @kernel reduce(+, axis=1)
    for i in 0..rows, j in 0..cols:
        yield matrix[i, j]
}

// Argmax (reduction returning index)
let max_idx = compute(data) {
    @kernel reduce(argmax)
    for i in 0..len:
        yield (data[i], i)  // (value, index) pair
}


// === Matrix Operations ===

// Matrix multiply
let C = compute(A, B) {
    @kernel matmul
    @workgroup(16, 16)
    @shared(16 * 16 * 4 * 2)  // Tile A and B in shared memory
    for i in 0..M, j in 0..N:
        var sum = 0.0
        for k in 0..K:
            sum += A[i, k] * B[k, j]
        out[i, j] = sum
}

// Batched matmul
let C = compute(A, B) {
    @kernel batched_matmul
    @workgroup(16, 16, 1)
    for b in 0..batch, i in 0..M, j in 0..N:
        var sum = 0.0
        for k in 0..K:
            sum += A[b, i, k] * B[b, k, j]
        out[b, i, j] = sum
}


// === Convolution ===

let output = compute(input, kernel, bias) {
    @kernel conv2d
    @workgroup(8, 8, 4)
    for b in 0..batch, oc in 0..out_channels:
        for oh in 0..out_h, ow in 0..out_w:
            var sum = bias[oc]
            for ic in 0..in_channels:
                for kh in 0..kernel_h, kw in 0..kernel_w:
                    let ih = oh * stride + kh - padding
                    let iw = ow * stride + kw - padding
                    if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                        sum += input[b, ic, ih, iw] * kernel[oc, ic, kh, kw]
            out[b, oc, oh, ow] = sum
}

// Depthwise convolution
let output = compute(input, kernel) {
    @kernel conv2d_depthwise
    @workgroup(8, 8)
    for b in 0..batch, c in 0..channels:
        for oh in 0..out_h, ow in 0..out_w:
            var sum = 0.0
            for kh in 0..kernel_h, kw in 0..kernel_w:
                sum += input[b, c, oh + kh, ow + kw] * kernel[c, kh, kw]
            out[b, c, oh, ow] = sum
}


// === Attention Mechanisms ===

// Scaled dot-product attention
let attn = compute(Q, K, V, mask) {
    @kernel attention
    @workgroup(32, 4)  // seq_tile, head_tile
    @shared(32 * 64 * 4)  // K, V tiles

    for b in 0..batch, h in 0..heads, i in 0..seq_len:
        // Compute max for numerical stability
        var max_score = -inf
        for j in 0..seq_len:
            if mask[i, j]:
                let score = dot(Q[b, h, i], K[b, h, j]) / sqrt(head_dim)
                max_score = max(max_score, score)

        // Compute softmax denominator
        var sum_exp = 0.0
        for j in 0..seq_len:
            if mask[i, j]:
                let score = dot(Q[b, h, i], K[b, h, j]) / sqrt(head_dim)
                sum_exp += exp(score - max_score)

        // Weighted sum of values
        for d in 0..head_dim:
            var acc = 0.0
            for j in 0..seq_len:
                if mask[i, j]:
                    let score = dot(Q[b, h, i], K[b, h, j]) / sqrt(head_dim)
                    let weight = exp(score - max_score) / sum_exp
                    acc += weight * V[b, h, j, d]
            out[b, h, i, d] = acc
}

// Flash attention (memory-efficient)
let attn = compute(Q, K, V) {
    @kernel flash_attention
    @workgroup(64)
    @tile(block_q=64, block_kv=64)

    for b in 0..batch, h in 0..heads:
        for q_block in 0..seq_len step 64:
            // Online softmax computation
            var m = -inf  // Running max
            var l = 0.0   // Running sum of exp
            var o = zeros(64, head_dim)  // Running output

            for kv_block in 0..seq_len step 64:
                // Load K, V blocks to shared memory
                let K_block = K[b, h, kv_block:kv_block+64]
                let V_block = V[b, h, kv_block:kv_block+64]

                // Compute attention scores for this block
                let S = Q[b, h, q_block:q_block+64] @ transpose(K_block) / sqrt(head_dim)

                // Update running statistics
                let m_new = max(m, row_max(S))
                let l_new = exp(m - m_new) * l + row_sum(exp(S - m_new))

                // Update output
                o = exp(m - m_new) * o + exp(S - m_new) @ V_block
                m = m_new
                l = l_new

            // Normalize
            out[b, h, q_block:q_block+64] = o / l


// === Custom Kernels ===

// Softmax (fused)
let probs = compute(logits) {
    @kernel fused
    // First pass: find max
    let max_val = reduce(max, logits)
    // Second pass: compute exp and sum
    let shifted = logits - max_val
    let exp_vals = exp(shifted)
    let sum_exp = reduce(+, exp_vals)
    // Third pass: normalize
    out = exp_vals / sum_exp
}

// Layer norm (fused)
let normalized = compute(x, gamma, beta) {
    @kernel fused
    @workgroup(256)

    for b in 0..batch:
        // Compute mean
        var mean = 0.0
        for i in 0..hidden:
            mean += x[b, i]
        mean /= hidden

        // Compute variance
        var var = 0.0
        for i in 0..hidden:
            let diff = x[b, i] - mean
            var += diff * diff
        var /= hidden

        // Normalize and scale
        let inv_std = rsqrt(var + eps)
        for i in 0..hidden:
            out[b, i] = (x[b, i] - mean) * inv_std * gamma[i] + beta[i]
}

// GELU activation
let activated = compute(x) {
    @kernel elementwise
    for i in 0..len:
        // GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let x3 = x[i] * x[i] * x[i]
        out[i] = x[i] * 0.5 * (1.0 + tanh(0.7978845608 * (x[i] + 0.044715 * x3)))
}

// RoPE (Rotary Position Embedding)
let rotated = compute(x, cos_cache, sin_cache) {
    @kernel elementwise
    for b in 0..batch, h in 0..heads, s in 0..seq, d in 0..head_dim/2:
        let x0 = x[b, h, s, d * 2]
        let x1 = x[b, h, s, d * 2 + 1]
        let c = cos_cache[s, d]
        let s = sin_cache[s, d]
        out[b, h, s, d * 2] = x0 * c - x1 * s
        out[b, h, s, d * 2 + 1] = x0 * s + x1 * c
}
```

### Device Management

**Status:** not built. `@device` is parsed and ignored, so `@device("metal")` changes nothing: the in-place elementwise shape still becomes CPU SIMD code, and any other body takes the paths described in the status section. `@device("auto")` will mean a size-based choice at compile time (small kernels to the CPU SIMD path, large ones to the GPU), which is a stated rule, not a fallback.

```zynml
// Query available devices
let devices = compute_devices()
// Returns: [
//   {id: "cuda:0", type: "gpu", name: "NVIDIA RTX 4090", memory: 24GB},
//   {id: "cuda:1", type: "gpu", name: "NVIDIA RTX 4090", memory: 24GB},
//   {id: "metal:0", type: "gpu", name: "Apple M2 Max", memory: 32GB},
//   {id: "cpu", type: "cpu", name: "AMD Ryzen 9", cores: 16},
// ]

// Set default device
config {
    compute_device: "cuda:0"  // or "metal:0", "cpu", "auto"
}

// Per-computation device selection
let gpu_result = compute(data) @device("cuda:0") {
    @kernel matmul
    ...
}

let cpu_result = compute(data) @device("cpu") {
    @kernel matmul
    ...
}

// Automatic device selection (GPU if available, else CPU)
let result = compute(data) @device("auto") {
    @kernel matmul
    ...
}

// Multi-GPU (data parallel)
let result = compute(large_data) @device(["cuda:0", "cuda:1"]) @parallel(data) {
    @kernel matmul
    ...
}

// Multi-GPU (model parallel)
let result = compute(data) @device(["cuda:0", "cuda:1"]) @parallel(model) {
    @kernel large_model_forward
    ...
}
```

### Async Execution

**Status:** not built. `@async` will reuse Zyntax's fibers and async (the same `await`) rather than introduce a separate future type.

```zynml
// Synchronous (default)
let result = compute(data) {
    @kernel expensive
    ...
}
// Blocks until complete

// Asynchronous
let future = compute(data) @async {
    @kernel expensive
    ...
}
// Returns immediately

// Do other work while GPU computes
let cpu_work = process_metadata()

// Wait for GPU result
let result = await future

// Multiple async computations
let futures = [
    compute(batch1) @async { ... },
    compute(batch2) @async { ... },
    compute(batch3) @async { ... },
]

// Wait for all
let results = await all(futures)

// Wait for any (returns first completed)
let (result, index) = await any(futures)

// Async with timeout
let result = await future timeout 1000ms else default_value
```

### Memory Management

**Status:** not built. On Apple unified memory, tensor storage is allocated page-aligned so Metal can wrap it without a copy (`storageModeShared`); explicit device copies matter only on discrete GPUs.

```zynml
// Explicit device memory allocation
let gpu_tensor = allocate(shape=[1024, 1024], dtype=float32, device="cuda:0")

// Copy to device
let gpu_data = data.to_device("cuda:0")

// Copy back to CPU
let cpu_data = gpu_data.to_device("cpu")

// Zero-copy (if supported)
let gpu_view = data.to_device("cuda:0", copy=false)  // Unified memory

// Pinned memory (for faster transfers)
let pinned = allocate(shape=[1024, 1024], dtype=float32, device="cpu", pinned=true)

// Memory pool (reduce allocation overhead)
let pool = memory_pool(device="cuda:0", size=1GB)
let tensor1 = pool.allocate(shape=[512, 512])
let tensor2 = pool.allocate(shape=[256, 256])
pool.free(tensor1)
pool.reset()  // Free all allocations
```

## Implementation Architecture

### Compilation Pipeline

```
ZynML compute() block
        │
        ▼
┌───────────────────┐
│   Parse & Type    │
│     Check         │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   HIR             │  ← first-class vectors; the one kernel IR
│   (kernel subset) │
└────────┬──────────┘
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  CUDA        │  │  Metal       │  │  Vulkan      │  │  CPU SIMD    │
│  Backend     │  │  Backend     │  │  (deferred)  │  │  (HIR Vector │
│  (NVPTX)     │  │  (MSL)       │  │  (naga IR)   │  │  4 backends) │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
   NVIDIA GPU       Apple GPU         Any GPU            CPU
```

**Status:** only the CPU SIMD column exists: HIR `Vector` instructions lowered by Cranelift, LLVM, wasm and the interpreter. Metal is next, NVIDIA follows, Vulkan is deferred.

### Compute IR (CIR)

**Superseded (2026-09-25):** there is no CIR, and none is planned. `compute()` lowers directly to HIR (`crates/compiler/src/ssa.rs`), and HIR is the kernel IR every backend lowers from. The sketch is kept because its operation list is a useful checklist for what the HIR kernel subset must cover.

```rust
// Compute Intermediate Representation

pub enum CirType {
    Scalar(ScalarType),
    Tensor { shape: Vec<Dim>, dtype: ScalarType },
    Pointer(Box<CirType>),
}

pub enum ScalarType {
    F16, F32, F64,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
}

pub enum Dim {
    Fixed(usize),
    Dynamic(String),  // Named dimension
}

pub struct ComputeKernel {
    name: String,
    kernel_type: KernelType,
    inputs: Vec<KernelInput>,
    output: KernelOutput,
    workgroup_size: Option<[u32; 3]>,
    shared_memory: Option<usize>,
    body: Vec<CirStatement>,
}

pub enum KernelType {
    Elementwise,
    Reduce(ReduceOp),
    MatMul,
    Conv2d { stride: u32, padding: u32 },
    Attention,
    Custom,
    Fused(Vec<KernelType>),
}

pub enum ReduceOp {
    Sum, Product, Min, Max, And, Or, ArgMin, ArgMax,
}

pub struct KernelInput {
    name: String,
    ty: CirType,
    access: AccessMode,
}

pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

pub enum CirStatement {
    // Control flow
    For { var: String, range: Range, body: Vec<CirStatement> },
    If { cond: CirExpr, then_body: Vec<CirStatement>, else_body: Vec<CirStatement> },

    // Assignment
    Let { var: String, value: CirExpr },
    Assign { target: CirExpr, value: CirExpr },

    // Parallel constructs
    ParallelFor { vars: Vec<String>, ranges: Vec<Range>, body: Vec<CirStatement> },
    Barrier,  // Synchronization
    AtomicOp { op: AtomicOp, target: CirExpr, value: CirExpr },

    // Reduction
    Yield(CirExpr),

    // Memory
    SharedAlloc { name: String, size: usize },
    Load { dst: String, src: CirExpr },
    Store { dst: CirExpr, src: CirExpr },
}

pub enum CirExpr {
    // Literals
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),

    // Variables
    Var(String),
    ThreadIdx(Axis),
    BlockIdx(Axis),
    BlockDim(Axis),

    // Indexing
    Index { base: Box<CirExpr>, indices: Vec<CirExpr> },
    Slice { base: Box<CirExpr>, ranges: Vec<Range> },

    // Arithmetic
    BinOp { op: BinOp, left: Box<CirExpr>, right: Box<CirExpr> },
    UnaryOp { op: UnaryOp, operand: Box<CirExpr> },

    // Math functions
    MathFn { fn_name: MathFn, args: Vec<CirExpr> },

    // Special
    Dot { a: Box<CirExpr>, b: Box<CirExpr> },
    Reduce { op: ReduceOp, input: Box<CirExpr>, axis: Option<usize> },
}

pub enum MathFn {
    Sin, Cos, Tan, Exp, Log, Sqrt, Rsqrt, Abs, Floor, Ceil,
    Tanh, Sinh, Cosh, Pow, Min, Max, Clamp,
}
```

### Backend Implementations

#### CUDA Backend

**Superseded (2026-09-25):** PTX is not written by hand. It comes from HIR through LLVM IR and the in-process NVPTX target of the LLVM 21 the compiler already links, and `cudarc` loads it; the driver JIT-compiles PTX on load, so the "Compile PTX to cubin" step below does not exist on that path. Design in [GPU_AOT_ARCHITECTURE.md](../GPU_AOT_ARCHITECTURE.md).

```rust
pub struct CudaBackend {
    context: CudaContext,
    module_cache: HashMap<KernelHash, CudaModule>,
}

impl ComputeBackend for CudaBackend {
    fn compile(&mut self, kernel: &ComputeKernel) -> Result<CompiledKernel> {
        // Generate PTX
        let ptx = self.generate_ptx(kernel)?;

        // Compile PTX to cubin
        let module = self.context.load_ptx(&ptx)?;

        Ok(CompiledKernel::Cuda(module))
    }

    fn execute(&self, kernel: &CompiledKernel, args: &[TensorArg]) -> Result<()> {
        let CompiledKernel::Cuda(module) = kernel else { bail!("Wrong backend") };

        // Set up kernel arguments
        let mut kernel_args = Vec::new();
        for arg in args {
            kernel_args.push(arg.device_ptr());
        }

        // Launch kernel
        let grid = self.compute_grid_size(kernel, args);
        let block = kernel.workgroup_size.unwrap_or([256, 1, 1]);

        module.launch(grid, block, &kernel_args)?;

        Ok(())
    }

    fn generate_ptx(&self, kernel: &ComputeKernel) -> Result<String> {
        let mut ptx = String::new();

        // Header
        writeln!(ptx, ".version 7.0")?;
        writeln!(ptx, ".target sm_80")?;
        writeln!(ptx, ".address_size 64")?;

        // Kernel function
        writeln!(ptx, ".visible .entry {}(", kernel.name)?;
        for (i, input) in kernel.inputs.iter().enumerate() {
            let param_type = self.cir_type_to_ptx(&input.ty);
            writeln!(ptx, "  .param .{} param_{},", param_type, i)?;
        }
        writeln!(ptx, ") {{")?;

        // Generate body
        self.generate_ptx_body(&mut ptx, kernel)?;

        writeln!(ptx, "}}")?;

        Ok(ptx)
    }
}
```

#### Metal Backend

**Status (2026-09-25): not built; the design below is decided, with these corrections.**

- **Emitter.** MSL source generated from the HIR kernel subset, in `crates/compiler` beside the other backends, language-neutral. LLVM has no Apple GPU (AIR) target, so MSL is the route; candle, PyTorch Inductor's MPS backend and CubeCL compile the same way, which keeps the benchmark comparison like for like.
- **Compile.** `MTLDevice newLibraryWithSource:options:error:` then `newComputePipelineStateWithFunction:`. This uses the OS's own Metal compiler service: verified on the M1 Pro under macOS 26.6.2 on 2026-09-25 with no Metal toolchain installed (`xcrun metal` was missing). The offline `metal` compiler and AIR/metallib emission are not used: the toolchain cannot be redistributed, and direct AIR emission depends on reverse-engineered formats that broke on a macOS release. Revisit only if cold compile time dominates time to first result; Metal's on-disk cache already makes repeat compiles fast, and `MTLBinaryArchive` is the documented next step.
- **Binding.** The host side (device, queue, buffers, launch) lives in a runtime crate on `objc2-metal` and `objc2-metal-performance-shaders` 0.3.x, linked with a `cfg(target_vendor = "apple")` `build.rs` line the way `zrtl_tensor` links Accelerate. The `metal` crate (metal-rs) used in the sketch is deprecated in favour of objc2-metal; wgpu and candle have moved.
- **Pipeline cache.** Pipeline states are cached by kernel hash; the sketch below builds one per `execute`.
- **Batching.** Dispatches encode into one open command buffer, committed and waited on only when the host reads a result. A synchronous commit-and-wait per dispatch, as in the sketch, costs about two orders of magnitude more than a batched dispatch on the M1 Pro (measured 2026-09-25).
- **GEMM.** Chosen by size: Accelerate CBLAS (AMX) for small and medium matrices, `MPSMatrixMultiplication` encoded into the same command buffer for large ones (on the M1 Pro, fp32, measured 2026-09-25, Accelerate led at n=512 and MPS from n=1024). A generated `simdgroup_matrix` MSL GEMM, as candle and MLX use, covers fused epilogues MPS cannot express. `zrtl_tensor` moves from the deprecated `cblas_sgemm` symbol to the `$NEWLAPACK` CBLAS symbols. Classic BNNS is deprecated since macOS 15 and not used; MPSGraph only for convolution parity with PyTorch MPS, which is built on it.
- **Metal 4 tensor ops** (`MetalPerformancePrimitives` `matmul2d`) compile on the M1 Pro under macOS 26 but reach Neural Accelerators only on M5 and A19 GPUs. They are a later tier gated on device family, not a promise of tensor-core speed on the M1 Pro.

```rust
pub struct MetalBackend {
    device: metal::Device,
    command_queue: metal::CommandQueue,
    library_cache: HashMap<KernelHash, metal::Library>,
}

impl ComputeBackend for MetalBackend {
    fn compile(&mut self, kernel: &ComputeKernel) -> Result<CompiledKernel> {
        // Generate MSL (Metal Shading Language)
        let msl = self.generate_msl(kernel)?;

        // Compile MSL
        let library = self.device.new_library_with_source(&msl, &Default::default())?;

        Ok(CompiledKernel::Metal(library))
    }

    fn execute(&self, kernel: &CompiledKernel, args: &[TensorArg]) -> Result<()> {
        let CompiledKernel::Metal(library) = kernel else { bail!("Wrong backend") };

        let function = library.get_function(&kernel.name)?;
        let pipeline = self.device.new_compute_pipeline_state_with_function(&function)?;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);

        for (i, arg) in args.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(arg.metal_buffer()), 0);
        }

        let grid_size = self.compute_grid_size(kernel, args);
        let threadgroup_size = kernel.workgroup_size.unwrap_or([256, 1, 1]);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    fn generate_msl(&self, kernel: &ComputeKernel) -> Result<String> {
        let mut msl = String::new();

        writeln!(msl, "#include <metal_stdlib>")?;
        writeln!(msl, "using namespace metal;")?;

        // Kernel function
        writeln!(msl, "kernel void {}(", kernel.name)?;
        for (i, input) in kernel.inputs.iter().enumerate() {
            let metal_type = self.cir_type_to_metal(&input.ty);
            writeln!(msl, "  device {} *arg{} [[buffer({})]],", metal_type, i, i)?;
        }
        writeln!(msl, "  uint3 gid [[thread_position_in_grid]]")?;
        writeln!(msl, ") {{")?;

        // Generate body
        self.generate_msl_body(&mut msl, kernel)?;

        writeln!(msl, "}}")?;

        Ok(msl)
    }
}
```

#### CPU SIMD Backend

**Superseded (2026-09-25):** CPU SIMD was built differently. There is no `CpuSimdBackend` or `SimdLevel`: kernel bodies lower to HIR `Vector` instructions in `ssa.rs`, and Cranelift, LLVM, wasm and the interpreter each lower those natively. Width comes from `target_vector.rs`, except that the elementwise kernel loop still hard-codes 4 lanes (tracked in git-bug). Matmul is not generated on the CPU; on Apple it goes to Accelerate through `zrtl_tensor`.

```rust
pub struct CpuSimdBackend {
    // Uses existing Cranelift infrastructure
    backend: CraneliftBackend,
    simd_level: SimdLevel,
}

pub enum SimdLevel {
    Scalar,
    Sse41,
    Avx2,
    Avx512,
    Neon,
}

impl ComputeBackend for CpuSimdBackend {
    fn compile(&mut self, kernel: &ComputeKernel) -> Result<CompiledKernel> {
        // Lower to Cranelift IR with SIMD instructions
        let mut builder = self.backend.create_function();

        match kernel.kernel_type {
            KernelType::Elementwise => {
                self.compile_elementwise(&mut builder, kernel)?;
            }
            KernelType::Reduce(op) => {
                self.compile_reduction(&mut builder, kernel, op)?;
            }
            KernelType::MatMul => {
                self.compile_matmul(&mut builder, kernel)?;
            }
            _ => {
                self.compile_generic(&mut builder, kernel)?;
            }
        }

        let func = builder.finalize()?;
        Ok(CompiledKernel::Cpu(func))
    }

    fn compile_elementwise(&self, builder: &mut FunctionBuilder, kernel: &ComputeKernel) -> Result<()> {
        // Vectorize loop by SIMD width
        let simd_width = match self.simd_level {
            SimdLevel::Avx512 => 16,  // 512 bits / 32 bits
            SimdLevel::Avx2 => 8,     // 256 bits / 32 bits
            SimdLevel::Sse41 | SimdLevel::Neon => 4,  // 128 bits / 32 bits
            SimdLevel::Scalar => 1,
        };

        // Generate SIMD loop + scalar remainder
        // ...

        Ok(())
    }
}
```

## Dependencies

**Superseded (2026-09-25):** the 2025-12 list (`cuda-runtime`/`cuda-driver` 0.3, `metal` 0.27, `vulkano` 0.34, `wgpu` 0.19, and `cpu`/`cuda`/`metal`/`vulkan`/`webgpu` features) was never added to any `Cargo.toml` and is years out of date. The planned dependencies, versions as of 2026-09-25:

```toml
# Metal (Apple only), host side of the Metal backend
[target.'cfg(target_vendor = "apple")'.dependencies]
objc2-metal = "0.3"
objc2-metal-performance-shaders = "0.3"

[dependencies]
# NVIDIA, behind the compiler's opt-in `cuda` feature (see GPU_AOT_ARCHITECTURE.md)
cudarc = { version = "0.19", optional = true, default-features = false, features = ["std", "driver", "dynamic-loading", "cuda-12090"] }

# Tile IR emission for the NVIDIA tile track (later)
cutile-ir = { version = "0.3", optional = true }

# Portable path (deferred): naga for IR and its SPIR-V/WGSL/MSL writers, ash for Vulkan.
# Not vulkano-shaders, which compiles GLSL through shaderc (a C++ toolchain).
```

PTX emission needs no dependency: inkwell's default `target-all` already enables NVPTX in the `llvm-backend` build.

## Implementation Phases

**Status (2026-09-25):** the 2025-12 week schedule is superseded; phases keep their order without dates. Checked against the code.

### Phase 1: CPU SIMD Backend
- [x] Existing `zrtl_simd` operations
- ~~Compute IR design~~ (superseded: HIR is the kernel IR)
- [x] First-class HIR vectors on Cranelift, LLVM, wasm and the interpreter
- [ ] Elementwise kernels: one in-place shape done; `out[i] = f(x[i])`, multi-input and broadcast not done; lane count hard-coded to 4
- [ ] Reduction kernels: partial; returns the last direct `yield`, no accumulation, no operator
- [ ] Value-checked `compute()` tests on all four backends (today only Cranelift checks values; the ZynML tests only compile)

### Phase 2: Compute Syntax
- [x] Grammar extension for `compute()`
- [ ] Type checking for kernels
- [ ] Lowering reads `@kernel(x)`, `@device`, `@workgroup`
- [ ] Unsupported kernel shapes are compile errors, not calls to `$Zyntax$compute`
- [ ] `zyntax` kernel module in Python and Lua

### Phase 3: Metal Backend (next GPU step)
- [ ] MSL code generation from HIR
- [ ] Metal runtime on objc2-metal: pipeline cache, batched command buffer
- [ ] GEMM dispatch: Accelerate and MPS by size; generated simdgroup_matrix GEMM for fused epilogues
- [ ] Shared memory (threadgroup) support
- [ ] Benchmark harness (see Performance Targets)

### Phase 4: CUDA Backend (needs an sm_80+ Linux machine)
- [ ] PTX via LLVM NVPTX
- [ ] `cudarc` runtime integration
- [ ] Memory management
- [ ] Async execution on Zyntax fibers
- [ ] Tile IR track through `cutile-ir`

### Phase 5: Advanced Features
- [ ] Kernel fusion
- [ ] Auto-tuning
- [ ] Multi-GPU support
- [ ] Flash attention kernel

### Phase 6: Portable Backend (deferred)
- [ ] naga IR emission from HIR (SPIR-V, WGSL, MSL writers)
- [ ] Vulkan through `ash` on Linux
- [ ] Browser integration and ZynBook GPU support (WGSL)

Deferred because there is no discrete GPU to benchmark on, wgpu adds per-dispatch overhead that native Metal does not, and WGSL has no stable simdgroup-matrix support, so this path cannot meet the PyTorch MPS and candle Metal targets.

## Performance Targets

**Superseded (2026-09-25):** the 2025-12 table of ratios against PyTorch on CUDA (1.1x matmul, 1.5x attention, 2 to 3x fused) was never measured, and no CUDA hardware is available to measure it.

Kernels are benchmarked against:

1. **PyTorch**, eager and `torch.compile`.
2. **The Python kernel ecosystem**: numba, JAX/XLA, Taichi, numpy, and Triton where a CUDA GPU exists.
3. **Rust's candle.**

| Where | Zyntax runs | Compared against |
|-------|-------------|------------------|
| CPU, Mac (M1 Pro) | CPU SIMD kernels, Accelerate for GEMM | PyTorch CPU, numba, JAX, Taichi, numpy, candle CPU |
| CPU, NUC (x86_64 Linux) | CPU SIMD kernels | PyTorch CPU, numba, JAX, Taichi, numpy, candle CPU |
| GPU, Mac | Metal kernels, MPS for large GEMM | PyTorch MPS, candle Metal |
| GPU, NVIDIA (when available) | PTX kernels | PyTorch CUDA, Triton, candle CUDA |

Workloads: elementwise (saxpy, GELU), reductions (sum, softmax, layernorm), GEMM, convolution, attention, and LLM prefill and decode. Every row reports steady-state kernel time and time to first result (compile and JIT included), from interleaved runs with their spread. Numbers live with the benchmark results, dated, not in this document.

### Memory Efficiency

Design targets, none built:

| Operation | ZynML | PyTorch |
|-----------|-------|---------|
| Attention (seq=4096) | O(n) | O(n²) |
| Fused LayerNorm | 1 pass | 2 passes |
| Activation checkpointing | Automatic | Manual |

## Example: End-to-End Transformer Block

```zynml
pipeline transformer_block(x: tensor[batch, seq, hidden], layer: int) -> tensor[batch, seq, hidden]:
    let weights = model.layers[layer]

    // Self-attention with fused softmax
    let qkv = compute(x, weights.qkv_proj) {
        @kernel matmul
        @device("auto")
        for b in 0..batch, s in 0..seq, h in 0..3*hidden:
            var sum = 0.0
            for i in 0..hidden:
                sum += x[b, s, i] * weights.qkv_proj[i, h]
            out[b, s, h] = sum
    }

    let (q, k, v) = split(qkv, 3, axis=-1)

    // Reshape for multi-head
    let q = reshape(q, [batch, seq, num_heads, head_dim])
    let k = reshape(k, [batch, seq, num_heads, head_dim])
    let v = reshape(v, [batch, seq, num_heads, head_dim])

    // Flash attention
    let attn_out = compute(q, k, v) @device("auto") {
        @kernel flash_attention
        @workgroup(64)
        // ... flash attention implementation
    }

    // Output projection
    let attn_out = reshape(attn_out, [batch, seq, hidden])
    let projected = compute(attn_out, weights.out_proj) {
        @kernel matmul
        for b in 0..batch, s in 0..seq, h in 0..hidden:
            var sum = 0.0
            for i in 0..hidden:
                sum += attn_out[b, s, i] * weights.out_proj[i, h]
            out[b, s, h] = sum
    }

    // Residual + LayerNorm (fused)
    let normed = compute(x, projected, weights.ln1_weight, weights.ln1_bias) {
        @kernel fused
        for b in 0..batch, s in 0..seq:
            // Add residual
            var sum = 0.0
            var sq_sum = 0.0
            for h in 0..hidden:
                let val = x[b, s, h] + projected[b, s, h]
                sum += val
                sq_sum += val * val

            let mean = sum / hidden
            let var = sq_sum / hidden - mean * mean
            let inv_std = rsqrt(var + 1e-6)

            for h in 0..hidden:
                let val = x[b, s, h] + projected[b, s, h]
                out[b, s, h] = (val - mean) * inv_std * weights.ln1_weight[h] + weights.ln1_bias[h]
    }

    // FFN (fused GELU)
    let ffn_out = compute(normed, weights.ffn_up, weights.ffn_down) {
        @kernel fused
        @shared(batch * seq * ffn_hidden * 4)

        for b in 0..batch, s in 0..seq:
            // Up projection + GELU
            shared var up[ffn_hidden]
            for f in 0..ffn_hidden:
                var sum = 0.0
                for h in 0..hidden:
                    sum += normed[b, s, h] * weights.ffn_up[h, f]
                // GELU
                let x = sum
                let x3 = x * x * x
                up[f] = x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x3)))

            barrier()

            // Down projection
            for h in 0..hidden:
                var sum = 0.0
                for f in 0..ffn_hidden:
                    sum += up[f] * weights.ffn_down[f, h]
                out[b, s, h] = sum
    }

    // Final residual + LayerNorm
    let output = compute(normed, ffn_out, weights.ln2_weight, weights.ln2_bias) {
        @kernel fused
        // ... same as above
    }

    return output
```

This is the target: a complete, GPU-accelerated transformer block written entirely in ZynML. **Status:** no part of it compiles to a kernel today; `matmul` and `fused` kernels take the unsupported-body paths described in the status section (the last direct `yield`, or the undefined runtime dispatch).
