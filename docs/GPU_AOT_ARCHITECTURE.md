# GPU AOT Architecture: NVPTX via LLVM IR

**Status**: Not built; direction revised 2026-09-25
**Dependencies**: LLVM Backend (`llvm-backend` feature), HIR first-class vectors, kernel typing
**Feature Flag**: none today. PTX emission rides `llvm-backend`; the CUDA driver binding will be its own opt-in feature (see Feature Flag below)

---

## Status and direction (September 2026)

Checked against the code on 2026-09-25 (HEAD `a900deba`). This document is the NVIDIA design. The Apple (Metal) path, the kernel surface and the benchmark plan are in [09-gpu-compute-system.md](ml-dsl-plans/09-gpu-compute-system.md); the two share this section's decisions. Where a section below no longer holds it carries a **Superseded** or **Status** note in place.

### Built

- ZynML `compute(args) @modifiers { block }` grammar (`crates/zynml/ml.zyn`, `compute_expr`), lowered in `crates/compiler/src/ssa.rs`.
- `@kernel elementwise` for one shape: `for i in r { arr[i] = arr[i] OP scalar }` over a compute argument, lowered by `emit_elementwise_simd_loop` to a 4-lane vector loop with a scalar remainder.
- `@kernel reduce`, partially: with a `yield` directly in the body, the result is the last yielded value. Nothing accumulates yet.
- First-class `HirType::Vector` and the HIR vector instructions, lowered natively on Cranelift, LLVM, wasm and the interpreter. Host vector width comes from `target_vector.rs`; the `auto_vectorize`, `loop_vectorize` and `reduction_vectorize` passes vectorize ordinary loops.
- Parallel loops (`parallel_safe.rs`, `parallel_dispatch.rs`), off unless `ZYNTAX_PARALLEL_LOOPS=1`, and not tied to `compute()`.

### Not built

- GPU code generation of any kind: no NVPTX, MSL, SPIR-V or WGSL emission, no GPU runtime, no driver binding. The LLVM tier initializes only the host target. The one GPU name in the tree is the `LoweringTarget::Nvptx` variant in `pattern_engine`, which one pattern-engine test selects and nothing lowers to.
- No `compute` Cargo feature, and none of the typed-AST or HIR kernel types in Parts 1 and 2.
- `@device`, `@workgroup` and `@kernel(x)` modifiers are parsed into `TypedComputeExpr` and never read. Kernel bodies are not type checked.
- A `compute()` body outside the recognised shapes either returns its last directly yielded value, when it is not marked `@kernel elementwise` and has a `yield` directly in the body, or lowers to a call of `$Zyntax$compute`, which is not defined anywhere. Both are bugs, not design (see Tracking).
- No `zyntax` module in the Python or Lua frontends.

### Decided direction

1. **Kernels are a Zyntax capability, not a ZynML feature.** Fibers, effects, handlers and kernels reach every language through a `zyntax` module in that language's idiom: Python `from zyntax import kernel`, a `zyntax` table in Lua, ZynML natively through `compute()` and `@kernel`. The module is opt-in, and surfaces that only zypy provides need no CPython support. The compiler side (kernel lowering from HIR, backends, runtime) is language-neutral, so each frontend only maps its typed subset onto HIR.
2. **Kernels are a typed subset.** A kernel body types to fixed-width scalars, vectors and buffers. Dynamic code inside a kernel is a compile error that names what is unsupported; it never falls back to a runtime dispatch or to boxed values.
3. **HIR is the kernel IR.** Kernel bodies lower to HIR with first-class vectors, and every GPU backend lowers from HIR. There is no separate compute IR.
4. **Apple first, through Metal.** The Mac (M1 Pro) is the only GPU available to test on. Metal Shading Language is generated from HIR and compiled at run time by the Metal framework, which needs no Xcode, Metal toolchain or C compiler on the user's machine. GEMM-shaped kernels dispatch to Accelerate (AMX) for small and medium sizes and to MPS for large ones, following the `zrtl_tensor` decision that FFI exists only for system accelerators generated code cannot reach. Design in 09-gpu-compute-system.md.
5. **NVIDIA through LLVM NVPTX.** HIR goes to LLVM IR on the existing LLVM 21 backend, then through an in-process NVPTX target machine (`nvptx64-nvidia-cuda`, `sm_80` floor) to PTX, which the CUDA driver JIT-compiles on load. The driver binding is `cudarc` with dynamic loading, behind an opt-in feature. Tile-level kernels (GEMM, attention, fused norms) get a second lowering to CUDA Tile IR bytecode through `cutile-ir` once the SIMT path works. Why: the NVPTX target is already linked (inkwell's default `target-all`), emitting PTX needs no CUDA toolkit, and NVIDIA's own `cuda-oxide` ends in the same LLVM NVPTX backend. What CUDA Rust contributes is in Part 5, "NVIDIA CUDA Rust (2026-09-08)".
6. **The portable path is deferred.** Vulkan and WebGPU wait: there is no discrete GPU to benchmark on, wgpu adds per-dispatch API overhead native Metal does not have, and WGSL has no stable simdgroup-matrix support. When the path is taken up, HIR lowers to naga IR (one emitter for SPIR-V, WGSL and MSL) and Linux runs Vulkan through `ash`. LLVM's `spirv64` target is not the route: it emits OpenCL-flavour SPIR-V, which Vulkan rejects.
7. **Benchmarks.** Kernels are measured against PyTorch (eager and `torch.compile`), then the Python kernel ecosystem (numba, JAX/XLA, Taichi, numpy, and Triton where a CUDA GPU exists), then Rust's candle. CPU on the Mac (M1 Pro) and the NUC (x86_64 Linux); GPU on the Mac through Metal against PyTorch MPS and candle Metal. Steady-state kernel time and time to first result (compile included) are both reported. Workloads: elementwise (saxpy, GELU), reductions (sum, softmax, layernorm), GEMM, convolution, attention, and LLM prefill and decode.

Decided (2026-09-25): GEMM-shaped kernels on NVIDIA dispatch to cuBLAS/cuBLASLt, loaded dynamically through `cudarc` like the driver, where that is fastest, the way they dispatch to Accelerate and MPS on Apple. Elementwise, reduction and fused kernels stay generated code.

### Test hardware

Neither the Mac nor the NUC (Intel Iris Xe graphics) has an NVIDIA GPU (checked 2026-09-25). Emitted PTX and Tile IR are golden-tested on both machines, since LLVM emits PTX without a GPU. Executing NVIDIA kernels and running the CUDA benchmarks needs an `sm_80` or newer Linux machine.

### Tracking

git-bug issues (show with `git-bug bug show <id>`):

- `13983a22693b250bbf4b91f5dca193b3ba0df4bbc094722791e0d3c7cf170811`: unmatched `compute()` bodies call the undefined `$Zyntax$compute` or return their last direct `yield`.
- `ab79beb59e7f172037869210d65eb668770a80c9099ddfc7dc373e3d6e796988`: `@kernel reduce` returns the last yield.
- `a127ddefe45d10932af8f3b2d4816181ce0d2d40ba3ddcc6465976286d00b2bf`: compute modifiers are parsed and never read.
- `77df2243b3427cedf9c0e9c5f79f88efd67a61f842ca54ba845f44065c95e3cb`: the elementwise loop hard-codes 4 lanes.
- `4eb2f34e13c8a37737b22e0b241a03d66d5deaee863217dd5c5efe85c25c02f5`: GPU backends and the `zyntax` kernel module.

---

## Executive Summary

This document describes the architecture for adding GPU compute support to Zyntax via NVPTX (NVIDIA PTX) code generation through LLVM IR. The design extends the existing LLVM backend to emit GPU kernels alongside CPU code, for kernels written in any Zyntax frontend: ZynML natively, and Python and Lua through the `zyntax` module.

**Key Goals:**
- Compile GPU kernels from HIR to NVPTX via LLVM IR
- ~~Zero-allocation critical CPU paths for ultra-low-latency execution~~ (superseded: out of GPU scope, see Part 4)
- GPU primitives in HIR, fed by kernel typing of the frontend's typed subset
- Seamless integration with existing tiered JIT compilation
- Support for heterogeneous CPU+GPU workloads

---

## Feature Flag

**Superseded (2026-09-25).** The 2025-12 plan was a `compute` feature pulling in `cuda-sys` and an `llvm-sys/nvptx` feature. Neither exists: `llvm-sys` has no `nvptx` feature, `cuda-sys` never went past 0.2.0, and no `compute` feature was ever added. The plan below replaces it.

PTX emission needs no new feature. The compiler already depends on inkwell 0.7.1 with `llvm21-1` and default features on (`crates/compiler/Cargo.toml`, under `llvm-backend`). inkwell's default `target-all` includes `target-nvptx`, so `Target::initialize_nvptx` is available in every `llvm-backend` build. PTX is text and can be emitted, verified and golden-tested on any machine.

Loading and launching PTX needs a CUDA driver binding, which is the only new opt-in:

```toml
# crates/compiler/Cargo.toml (package zyntax_compiler), planned
[features]
cuda = ["llvm-backend", "dep:cudarc"]

[dependencies]
cudarc = { version = "0.19", optional = true, default-features = false, features = ["std", "driver", "dynamic-loading", "cuda-12090"] }
```

`cudarc` with `dynamic-loading` needs no CUDA library at build time; it ships pregenerated bindings and `dlopen`s `libcuda` at run time. The feature therefore builds on macOS and on machines without CUDA, and a build with it runs everywhere, reporting "no CUDA driver" where there is none. The exact `cuda-NNNNN` binding set is chosen when the backend is built (versions as of 2026-09-25).

### Conditional Compilation

The NVPTX emitter compiles under `llvm-backend`; only the runtime is gated on `cuda`:

```rust
// In crates/compiler/src/lib.rs (planned)
#[cfg(feature = "llvm-backend")]
pub mod llvm_nvptx_backend;   // HIR -> PTX text

#[cfg(feature = "cuda")]
pub mod cuda_runtime;         // PTX load, buffers, launch
```

**Status:** none of these modules exist, and HIR has no GPU instructions. The 2025-12 text here said "GPU instructions are always defined in HIR" and sketched `HirInstruction::is_gpu_instruction`; neither exists.

### Runtime Detection

With the `cuda` feature, the runtime asks the driver for a device and reports its absence as a developer-facing error, never by silently running the kernel elsewhere:

```rust
#[cfg(feature = "cuda")]
pub fn cuda_available() -> bool {
    cudarc::driver::CudaContext::new(0).is_ok()
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_available() -> bool {
    false
}
```

### CLI Usage

**Status:** the CLI accepts only `cranelift` and `llvm` backends (`crates/zyntax_cli/src/backends/mod.rs`). The planned GPU flags are:

```bash
# Emit PTX for inspection (llvm-backend build)
zyntax compile --backend nvptx source.zyn

# Compile and run kernels on an NVIDIA GPU (cuda feature)
zyntax compile --backend cuda source.zyn
```

### Why Opt-In?

1. **Runtime dependency**: launching kernels needs the NVIDIA driver (`libcuda`) on the target machine. PTX emission does not, and the build machine needs neither the CUDA toolkit nor extra LLVM targets.
2. **Platform**: CUDA runs on Linux (and Windows) with `sm_80` or newer for the paths this design uses; the Mac and the NUC cannot execute it.
3. **Portability**: the default build carries no GPU runtime code.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Language Frontends                                 │
│        (ZynML natively; Python, Lua via the `zyntax` kernel module)          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: TypedAST + Kernel Metadata                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Kernel Annotations                                                      │ │
│  │  • @kernel(type), @device, @workgroup, @shared                         │ │
│  │  • @critical_path, @no_allocation, @inline_always                      │ │
│  │  • @vectorizable, @gpu_compute                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ GPU Type Extensions                                                     │ │
│  │  • Tensor types with device placement                                   │ │
│  │  • Shared memory types                                                  │ │
│  │  • Address space annotations                                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: HIR + GPU Primitives                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ GPU Instructions                                                        │ │
│  │  • ThreadIdx, BlockIdx, BlockDim, GridDim                              │ │
│  │  • SyncThreads, SharedMemAlloc, AtomicOp                               │ │
│  │  • WarpShuffle, MemoryFence                                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Address Space Model                                                     │ │
│  │  • Generic (0), Global (1), Shared (3), Constant (4), Local (5)        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Kernel Function Metadata                                                │ │
│  │  • Entry point markers                                                  │ │
│  │  • Launch configuration (grid/block dims)                              │ │
│  │  • Resource requirements (registers, shared memory)                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│  LAYER 3A: CPU Backend           │  │  LAYER 3B: GPU Backend           │
│                                  │  │                                  │
│  ┌────────────────────────────┐  │  │  ┌────────────────────────────┐  │
│  │ Cranelift JIT              │  │  │  │ LLVM NVPTX Backend         │  │
│  │  • Baseline tier           │  │  │  │  • Target: nvptx64-nvidia  │  │
│  │  • Native SIMD (HIR ops)   │  │  │  │  • PTX emission            │  │
│  │  • Fast compilation        │  │  │  │  • Kernel metadata         │  │
│  └────────────────────────────┘  │  │  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │  │  ┌────────────────────────────┐  │
│  │ LLVM x86/ARM Backend       │  │  │  │ CUDA Driver Runtime        │  │
│  │  • Optimizing tier         │  │  │  │  • Kernel loading          │  │
│  │  • Native SIMD (HIR ops)   │  │  │  │  • Memory management       │  │
│  │  • LLVM O3 and its passes  │  │  │  │  • Stream synchronization  │  │
│  └────────────────────────────┘  │  │  └────────────────────────────┘  │
└──────────────────────────────────┘  └──────────────────────────────────┘
              │                                     │
              └─────────────┬───────────────────────┘
                            ▼
              ┌──────────────────────────────────┐
              │      Heterogeneous Runtime       │
              │  • Unified memory management     │
              │  • CPU/GPU task scheduling       │
              │  • Async execution streams       │
              └──────────────────────────────────┘
```

**Status:** layers 1 and 2 describe planned additions; none of their types exist. The tier ladder today is interpreter, then Cranelift (`OptimizationTier::Baseline`), then LLVM (`Optimized`), in `crates/compiler/src/tiered_backend.rs`. CPU SIMD is HIR `Vector` instructions, which Cranelift, LLVM, wasm and the interpreter each lower natively; the vectorising passes (`auto_vectorize`, `loop_vectorize`, `reduction_vectorize`) run on HIR before either CPU tier compiles it, so both tiers emit SIMD.

---

## Part 1: TypedAST Kernel Metadata Extensions

**Status (2026-09-25): not built.** `KernelMetadata`, `ExecutionConstraints`, `CompileTarget`, `DeviceTarget`, `TensorShape`, `AddressSpace` and the GPU type variants below do not exist. `TypedFunction` carries only generic `annotations`, and nothing reads a function-level `@kernel`. The kernel metadata that does exist sits on the expression: `TypedComputeExpr { args, modifiers, kernel_attrs, body }` in `crates/typed_ast/src/typed_ast.rs`, built by the zyn_peg interpreter, and no pass reads `modifiers` or `kernel_attrs` yet. Kernel metadata will come from typing the kernel subset each frontend hands over (ZynML `compute()`, Python and Lua `@kernel` through the `zyntax` module), so the shapes below are a sketch of what that typing must record, not a fixed API.

### 1.1 Kernel Annotation System

Extend `TypedAST` with annotations that mark functions for GPU compilation:

```rust
// In crates/typed_ast/src/typed_ast.rs

/// Kernel metadata attached to function declarations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetadata {
    /// Kernel type (affects code generation strategy)
    pub kernel_type: KernelType,

    /// Target device for execution
    pub device: DeviceTarget,

    /// Workgroup/block dimensions (None = runtime-specified)
    pub workgroup_size: Option<(u32, u32, u32)>,

    /// Required shared memory in bytes
    pub shared_memory_bytes: u32,

    /// Maximum registers per thread (0 = no limit)
    pub max_registers: u32,

    /// Whether this is a device-callable function (not entry point)
    pub is_device_function: bool,

    /// Optimization hints
    pub hints: KernelHints,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelType {
    /// Element-wise operations (map)
    Elementwise,
    /// Reduction operations (sum, max, etc.)
    Reduce,
    /// Matrix multiplication
    MatMul,
    /// 2D convolution
    Conv2D,
    /// Attention mechanism (transformer)
    Attention,
    /// Flash attention (memory-efficient)
    FlashAttention,
    /// Scan/prefix sum
    Scan,
    /// Sort operations
    Sort,
    /// Custom user-defined kernel
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceTarget {
    /// CPU execution (SIMD vectors, parallel loops)
    Cpu,
    /// NVIDIA GPU via CUDA/PTX
    Cuda { compute_capability: (u32, u32) },
    /// AMD GPU via ROCm/HIP
    Rocm,
    /// Apple GPU via Metal (MSL source, not LLVM; see 09-gpu-compute-system.md)
    Metal,
    /// Vulkan compute (deferred; Vulkan SPIR-V, not LLVM's OpenCL-flavour spirv64)
    Vulkan,
    /// Automatic device selection
    Auto,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KernelHints {
    /// Use tensor cores (NVIDIA Ampere+)
    pub use_tensor_cores: bool,
    /// Vectorization width hint
    pub vector_width: Option<u32>,
    /// Loop unroll factor
    pub unroll_factor: Option<u32>,
    /// Memory coalescing hint
    pub coalesced_access: bool,
    /// Async copy hint (for shared memory staging)
    pub async_copy: bool,
}
```

### 1.2 Critical Path Annotations

**Superseded (2026-09-25):** low-latency CPU constraints are out of GPU scope and not planned here. Kept for reference.

For ultra-low-latency CPU code:

```rust
/// Execution path constraints for low-latency code
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionConstraints {
    /// No heap allocation allowed (stack-only)
    pub no_allocation: bool,

    /// No system calls allowed
    pub no_syscalls: bool,

    /// Always inline this function
    pub inline_always: bool,

    /// Disable bounds checking
    pub unsafe_unchecked: bool,

    /// Pin to specific CPU core (runtime hint)
    pub cpu_affinity: Option<u32>,

    /// Target latency in nanoseconds (for profiling)
    pub target_latency_ns: Option<u64>,

    /// Disable all runtime checks
    pub critical_path: bool,
}

/// Extended function declaration with kernel and execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedFunctionDeclaration {
    pub name: InternedString,
    pub parameters: Vec<TypedParameter>,
    pub return_type: Type,
    pub body: Option<TypedBlock>,
    pub is_async: bool,
    pub visibility: Visibility,

    // New fields for GPU/low-latency support
    pub kernel_metadata: Option<KernelMetadata>,
    pub execution_constraints: Option<ExecutionConstraints>,
    pub compile_target: CompileTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CompileTarget {
    #[default]
    Cpu,
    Gpu,
    /// Generate both CPU and GPU versions
    Heterogeneous,
}
```

### 1.3 GPU Type Extensions

Extend the type system with GPU-specific types:

```rust
// In crates/typed_ast/src/type_registry.rs

/// Extended type variants for GPU computing
pub enum Type {
    // ... existing variants ...

    /// Tensor type with shape, dtype, and device placement
    Tensor {
        element_type: Box<Type>,
        shape: TensorShape,
        device: DeviceTarget,
    },

    /// Pointer with explicit address space
    DevicePointer {
        pointee: Box<Type>,
        address_space: AddressSpace,
    },

    /// Shared memory array (block-local)
    SharedArray {
        element_type: Box<Type>,
        size: u32,
    },

    /// Warp-level type (32-wide SIMD on NVIDIA)
    WarpValue {
        element_type: Box<Type>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorShape {
    /// Static shape known at compile time
    Static(Vec<u64>),
    /// Dynamic shape with optional bounds
    Dynamic {
        rank: usize,
        max_dims: Option<Vec<u64>>,
    },
    /// Symbolic shape (for shape polymorphism)
    Symbolic(Vec<ShapeExpr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AddressSpace {
    Generic = 0,   // Default address space
    Global = 1,    // Device main memory
    Shared = 3,    // Block-local shared memory
    Constant = 4,  // Read-only constant memory
    Local = 5,     // Thread-private local memory
}
```

---

## Part 2: HIR GPU Primitives

**Status (2026-09-25): not built.** `HirInstruction` has no GPU variants, and `HirFunction` has no `kernel_info` and `HirModule` no `gpu_metadata` (see the struct definitions in `crates/compiler/src/hir.rs`). What HIR does have is first-class vectors: `HirType::Vector(elem, lanes)` and the vector instructions (splat, lane extract and insert, horizontal reduce, load, store, unary, min/max, dot), lowered natively on Cranelift, LLVM, wasm and the interpreter. The GPU primitives below extend that HIR; they are the design, and HIR remains the one kernel IR every GPU backend (NVPTX here, MSL in 09-gpu-compute-system.md) lowers from.

### 2.1 GPU Instruction Set

Extend `HirInstruction` with GPU-specific operations:

```rust
// In crates/compiler/src/hir.rs

pub enum HirInstruction {
    // ... existing instructions ...

    // ═══════════════════════════════════════════════════════════
    // GPU Thread/Block Indexing
    // ═══════════════════════════════════════════════════════════

    /// Get thread index within block: threadIdx.{x,y,z}
    ThreadIdx {
        dim: GpuDimension,
        result: HirId,
    },

    /// Get block index within grid: blockIdx.{x,y,z}
    BlockIdx {
        dim: GpuDimension,
        result: HirId,
    },

    /// Get block dimensions: blockDim.{x,y,z}
    BlockDim {
        dim: GpuDimension,
        result: HirId,
    },

    /// Get grid dimensions: gridDim.{x,y,z}
    GridDim {
        dim: GpuDimension,
        result: HirId,
    },

    // ═══════════════════════════════════════════════════════════
    // Synchronization
    // ═══════════════════════════════════════════════════════════

    /// Barrier synchronization within thread block
    /// All threads in block must reach this point before any proceed
    SyncThreads,

    /// Memory fence at specified scope
    GpuMemoryFence {
        scope: GpuFenceScope,
    },

    /// Warp-level barrier (no memory ordering)
    WarpSync {
        mask: Option<HirId>,  // Active thread mask (None = all)
    },

    // ═══════════════════════════════════════════════════════════
    // Shared Memory
    // ═══════════════════════════════════════════════════════════

    /// Allocate block-local shared memory
    SharedMemAlloc {
        result: HirId,
        element_type: HirType,
        num_elements: u32,
        align: u32,
    },

    /// Dynamic shared memory access (size determined at launch)
    DynamicSharedMemPtr {
        result: HirId,
        element_type: HirType,
        offset: HirId,  // Offset in bytes
    },

    // ═══════════════════════════════════════════════════════════
    // Atomic Operations (GPU-specific)
    // ═══════════════════════════════════════════════════════════

    /// GPU atomic operation with scope
    GpuAtomicOp {
        op: GpuAtomicKind,
        result: HirId,
        ptr: HirId,
        value: HirId,
        scope: GpuAtomicScope,
    },

    // ═══════════════════════════════════════════════════════════
    // Warp-Level Primitives
    // ═══════════════════════════════════════════════════════════

    /// Warp shuffle: exchange values between lanes
    WarpShuffle {
        mode: WarpShuffleMode,
        result: HirId,
        value: HirId,
        lane_or_delta: HirId,
        width: u32,  // Shuffle width (usually 32)
    },

    /// Warp vote: ballot/any/all
    WarpVote {
        op: WarpVoteOp,
        result: HirId,
        predicate: HirId,
    },

    /// Warp match: find threads with matching values
    WarpMatch {
        result: HirId,
        value: HirId,
    },

    // ═══════════════════════════════════════════════════════════
    // Tensor Core Operations (NVIDIA Ampere+)
    // ═══════════════════════════════════════════════════════════

    /// Matrix multiply-accumulate using tensor cores
    TensorCoreMMA {
        result: HirId,
        a: HirId,           // Matrix A fragment
        b: HirId,           // Matrix B fragment
        c: HirId,           // Accumulator
        layout: MmaLayout,  // Row/column major
    },

    /// Load matrix fragment from memory to registers
    TensorCoreLoad {
        result: HirId,
        ptr: HirId,
        layout: MmaLayout,
        fragment_type: MmaFragmentType,
    },

    /// Store matrix fragment from registers to memory
    TensorCoreStore {
        ptr: HirId,
        fragment: HirId,
        layout: MmaLayout,
    },

    // ═══════════════════════════════════════════════════════════
    // Async Copy (Ampere+)
    // ═══════════════════════════════════════════════════════════

    /// Async copy from global to shared memory
    AsyncCopy {
        dst: HirId,         // Shared memory destination
        src: HirId,         // Global memory source
        size_bytes: u32,
        cache_hint: CacheHint,
    },

    /// Commit async copy group
    AsyncCopyCommit {
        group: u32,
    },

    /// Wait for async copies to complete
    AsyncCopyWait {
        count: u32,  // Number of groups to wait for
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuDimension {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuFenceScope {
    /// Thread scope (compiler barrier only)
    Thread,
    /// Block scope (visible to all threads in block)
    Block,
    /// Device scope (visible to all threads on device)
    Device,
    /// System scope (visible to CPU and all GPUs)
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuAtomicKind {
    Add, Sub, Min, Max,
    And, Or, Xor,
    Exchange,
    CompareAndSwap,
    Inc, Dec,  // Wrap-around increment/decrement
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuAtomicScope {
    Block,
    Device,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarpShuffleMode {
    /// Direct exchange: read from lane idx
    Idx,
    /// Up: read from (lane - delta)
    Up,
    /// Down: read from (lane + delta)
    Down,
    /// Xor: read from (lane ^ mask)
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarpVoteOp {
    /// Return ballot mask of predicate across warp
    Ballot,
    /// True if any thread's predicate is true
    Any,
    /// True if all threads' predicates are true
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MmaLayout {
    RowMajor,
    ColMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MmaFragmentType {
    /// 16x16 matrix A (f16)
    MatrixA_16x16_F16,
    /// 16x16 matrix B (f16)
    MatrixB_16x16_F16,
    /// 16x16 accumulator (f32)
    Accumulator_16x16_F32,
    /// 8x8 matrix for smaller tiles
    MatrixA_8x8_F16,
    MatrixB_8x8_F16,
    Accumulator_8x8_F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheHint {
    Default,
    /// Cache at L1
    CacheL1,
    /// Cache at L2 only
    CacheL2,
    /// Streaming (don't cache)
    Streaming,
}
```

### 2.2 Kernel Function Metadata in HIR

```rust
// In crates/compiler/src/hir.rs

/// Extended HirFunction with kernel metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirFunction {
    pub id: HirId,
    pub name: InternedString,
    pub signature: HirFunctionSignature,
    pub entry_block: HirId,
    pub blocks: IndexMap<HirId, HirBlock>,
    pub locals: IndexMap<HirId, HirLocal>,
    pub values: IndexMap<HirId, HirValue>,
    pub previous_version: Option<HirId>,
    pub is_external: bool,
    pub calling_convention: CallingConvention,
    pub attributes: FunctionAttributes,
    pub link_name: Option<String>,

    // New: GPU kernel metadata
    pub kernel_info: Option<HirKernelInfo>,

    // New: Critical path constraints
    pub execution_constraints: Option<HirExecutionConstraints>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirKernelInfo {
    /// This is a GPU kernel entry point
    pub is_kernel_entry: bool,

    /// Kernel type for optimization hints
    pub kernel_type: KernelType,

    /// Target GPU architecture
    pub target_arch: GpuArch,

    /// Launch bounds
    pub launch_bounds: Option<LaunchBounds>,

    /// Required shared memory
    pub shared_memory_bytes: u32,

    /// Maximum register usage
    pub max_registers: Option<u32>,

    /// Kernel uses these address spaces
    pub address_spaces_used: HashSet<AddressSpace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuArch {
    /// NVIDIA sm_XX
    Sm(u32),  // e.g., Sm(80) for Ampere
    /// AMD gfx
    Gfx(u32),
    /// Generic PTX
    Ptx(u32),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LaunchBounds {
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Minimum blocks per SM (optional)
    pub min_blocks_per_sm: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HirExecutionConstraints {
    /// No dynamic memory allocation
    pub no_alloc: bool,
    /// Always inline
    pub inline_always: bool,
    /// No bounds checking
    pub no_bounds_check: bool,
    /// No null checks
    pub no_null_check: bool,
    /// This is on the critical execution path
    pub critical_path: bool,
    /// Preferred optimization level
    pub opt_level: OptimizationLevel,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization (debug)
    O0,
    /// Basic optimization
    #[default]
    O1,
    /// Standard optimization
    O2,
    /// Aggressive optimization
    O3,
    /// Size optimization
    Os,
    /// Ultra-low-latency (may sacrifice portability)
    Olatency,
}
```

### 2.3 HIR Module Extensions

```rust
// Extended HirModule with GPU support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirModule {
    pub id: HirId,
    pub name: InternedString,
    pub functions: IndexMap<HirId, HirFunction>,
    pub globals: IndexMap<HirId, HirGlobal>,
    pub types: IndexMap<TypeId, HirType>,
    pub imports: Vec<HirImport>,
    pub exports: Vec<HirExport>,
    pub version: u64,
    pub dependencies: HashSet<HirId>,

    // New: GPU-specific module metadata
    pub gpu_metadata: Option<GpuModuleMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuModuleMetadata {
    /// Target GPU architectures
    pub target_archs: Vec<GpuArch>,

    /// Minimum PTX version, if a feature needs more than the SM's floor.
    /// Normally None: LLVM picks the minimum PTX ISA for the chosen SM.
    pub ptx_version: Option<(u32, u32)>,

    /// CUDA compute capability
    pub compute_capability: (u32, u32),  // e.g., (8, 0) for sm_80

    /// Required CUDA features
    pub required_features: HashSet<String>,

    /// Kernel entry points in this module
    pub kernel_entries: Vec<HirId>,

    /// Device functions (callable from kernels)
    pub device_functions: Vec<HirId>,

    /// Constant memory globals
    pub constant_memory: Vec<HirId>,
}
```

---

## Part 3: LLVM NVPTX Backend

**Status (2026-09-25): not built; the approach holds.** Checked with Homebrew LLVM 21 (21.1.8 `llc`/`opt`; the compiler links 21.1.2 through the `llvm-config` on `PATH`) and no CUDA toolkit installed: an `nvptx64-nvidia-cuda` module compiles to PTX (`.target sm_80`). The code below is updated for LLVM 21:

- Kernels are marked with the `ptx_kernel` calling convention alone. The `!nvvm.annotations` `"kernel"` entry is a legacy form LLVM only auto-upgrades.
- Launch bounds are function attributes: `"nvvm.maxntid"`, `"nvvm.reqntid"`, `"nvvm.minctasm"` and `"nvvm.maxnreg"`. The old `"maxntidx"` annotation is upgraded to `"nvvm.maxntid"`.
- The PTX version is not hand-mapped. LLVM chooses the minimum PTX ISA for the SM (`sm_89` gets PTX 7.8 even when `+ptx75` is requested). The 2025-12 table here mapped `sm_89` to PTX 7.5, sent Blackwell to PTX 6.0 and included Volta. Pass `+ptxNN` only when a feature needs a newer ISA than the SM's floor; PTX 9.0 targets need LLVM 22.
- `sm_80` is the floor, matching NVIDIA's CUDA Rust tracks.

### 3.1 Target Initialization

```rust
// In crates/compiler/src/llvm_nvptx_backend.rs

use inkwell::{
    targets::{Target, TargetMachine, InitializationConfig, RelocMode, CodeModel},
    OptimizationLevel,
};

pub struct NvptxBackend<'ctx> {
    /// Base LLVM backend (reuse existing infrastructure)
    base: LLVMBackend<'ctx>,

    /// NVPTX target machine
    target_machine: TargetMachine,

    /// Compute capability (e.g., 80 for sm_80; 80 is the floor)
    compute_capability: u32,
}

impl<'ctx> NvptxBackend<'ctx> {
    pub fn new(
        context: &'ctx Context,
        module_name: &str,
        compute_capability: u32,
    ) -> CompilerResult<Self> {
        // Initialize NVPTX target
        Target::initialize_nvptx(&InitializationConfig::default());

        // Create target triple
        let triple = TargetTriple::create("nvptx64-nvidia-cuda");
        let target = Target::from_triple(&triple)
            .map_err(|e| CompilerError::CodeGen(format!("NVPTX target error: {}", e)))?;

        if compute_capability < 80 {
            return Err(CompilerError::CodeGen(format!(
                "NVPTX backend needs compute capability 8.0 or newer, got sm_{}",
                compute_capability
            )));
        }

        // Create target machine. No +ptxNN feature: LLVM selects the
        // minimum PTX ISA that supports the chosen SM.
        let cpu = format!("sm_{}", compute_capability);
        let features = "";

        let target_machine = target
            .create_target_machine(
                &triple,
                &cpu,
                features,
                OptimizationLevel::Aggressive,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or_else(|| CompilerError::CodeGen("Failed to create NVPTX target machine".into()))?;

        // Create base LLVM backend
        let base = LLVMBackend::new(context, module_name);

        Ok(Self {
            base,
            target_machine,
            compute_capability,
        })
    }

    /// Compile HIR module to PTX string
    pub fn compile_to_ptx(&mut self, hir_module: &HirModule) -> CompilerResult<String> {
        // Compile HIR to LLVM IR using base backend
        self.compile_module(hir_module)?;

        // Set correct data layout and triple
        self.set_nvptx_metadata()?;

        // Emit PTX
        let ptx = self.emit_ptx()?;

        Ok(ptx)
    }

    fn compile_module(&mut self, hir_module: &HirModule) -> CompilerResult<()> {
        // Use base LLVM backend compilation
        // But intercept kernel functions for special handling

        for (id, func) in &hir_module.functions {
            if let Some(kernel_info) = &func.kernel_info {
                if kernel_info.is_kernel_entry {
                    // Compile as kernel entry point
                    self.compile_kernel_function(*id, func)?;
                } else {
                    // Compile as device function
                    self.compile_device_function(*id, func)?;
                }
            } else {
                // Regular function - use base backend
                // (These typically won't be emitted to PTX)
            }
        }

        Ok(())
    }

    fn compile_kernel_function(&mut self, id: HirId, func: &HirFunction) -> CompilerResult<()> {
        // Create function with correct calling convention for kernels
        let fn_value = self.base.declare_function(id, func)?;

        // The ptx_kernel calling convention is what marks a kernel entry
        fn_value.set_call_conventions(inkwell::llvm_sys::LLVMCallConv::LLVMPTXKernelCallConv as u32);

        // Launch bounds are string function attributes
        if let Some(bounds) = func.kernel_info.as_ref().and_then(|k| k.launch_bounds) {
            self.add_launch_bound(fn_value, "nvvm.maxntid", bounds.max_threads_per_block);
            if let Some(min_blocks) = bounds.min_blocks_per_sm {
                self.add_launch_bound(fn_value, "nvvm.minctasm", min_blocks);
            }
        }

        // Compile function body with GPU instruction support
        self.compile_function_body(id, func)?;

        Ok(())
    }

    fn add_launch_bound(&self, fn_value: FunctionValue<'ctx>, key: &str, value: u32) {
        let context = self.base.module().get_context();
        let attr = context.create_string_attribute(key, &value.to_string());
        fn_value.add_attribute(AttributeLoc::Function, attr);
    }

    fn set_nvptx_metadata(&mut self) -> CompilerResult<()> {
        let module = self.base.module();

        // Set data layout for NVPTX64
        module.set_data_layout(&self.target_machine.get_target_data().get_data_layout());

        // Set target triple
        module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

        Ok(())
    }

    fn emit_ptx(&self) -> CompilerResult<String> {
        let module = self.base.module();

        // Emit assembly (PTX) to buffer
        let buffer = self.target_machine
            .write_to_memory_buffer(module, inkwell::targets::FileType::Assembly)
            .map_err(|e| CompilerError::CodeGen(format!("PTX emission error: {}", e)))?;

        // Convert to string
        let ptx = String::from_utf8_lossy(buffer.as_slice()).to_string();

        Ok(ptx)
    }
}
```

### 3.2 GPU Instruction Lowering

```rust
// In crates/compiler/src/llvm_nvptx_backend.rs

impl<'ctx> NvptxBackend<'ctx> {
    /// Compile a single HIR instruction with GPU support
    fn compile_gpu_instruction(&mut self, inst: &HirInstruction) -> CompilerResult<()> {
        match inst {
            HirInstruction::ThreadIdx { dim, result } => {
                self.compile_thread_idx(*dim, *result)
            }
            HirInstruction::BlockIdx { dim, result } => {
                self.compile_block_idx(*dim, *result)
            }
            HirInstruction::BlockDim { dim, result } => {
                self.compile_block_dim(*dim, *result)
            }
            HirInstruction::GridDim { dim, result } => {
                self.compile_grid_dim(*dim, *result)
            }
            HirInstruction::SyncThreads => {
                self.compile_sync_threads()
            }
            HirInstruction::SharedMemAlloc { result, element_type, num_elements, align } => {
                self.compile_shared_mem_alloc(*result, element_type, *num_elements, *align)
            }
            HirInstruction::GpuAtomicOp { op, result, ptr, value, scope } => {
                self.compile_gpu_atomic(*op, *result, *ptr, *value, *scope)
            }
            HirInstruction::WarpShuffle { mode, result, value, lane_or_delta, width } => {
                self.compile_warp_shuffle(*mode, *result, *value, *lane_or_delta, *width)
            }
            // ... other GPU instructions
            _ => {
                // Delegate to base backend for non-GPU instructions
                self.base.compile_instruction(inst)
            }
        }
    }

    fn compile_thread_idx(&mut self, dim: GpuDimension, result: HirId) -> CompilerResult<()> {
        // Call NVVM intrinsic: llvm.nvvm.read.ptx.sreg.tid.{x,y,z}
        let intrinsic_name = match dim {
            GpuDimension::X => "llvm.nvvm.read.ptx.sreg.tid.x",
            GpuDimension::Y => "llvm.nvvm.read.ptx.sreg.tid.y",
            GpuDimension::Z => "llvm.nvvm.read.ptx.sreg.tid.z",
        };

        let context = self.base.context;
        let i32_ty = context.i32_type();

        // Declare intrinsic
        let intrinsic_ty = i32_ty.fn_type(&[], false);
        let intrinsic = self.base.module().add_function(intrinsic_name, intrinsic_ty, None);

        // Call intrinsic
        let value = self.base.builder.build_call(intrinsic, &[], "tid")?
            .try_as_basic_value()
            .left()
            .unwrap();

        self.base.value_map.insert(result, value);
        Ok(())
    }

    fn compile_block_idx(&mut self, dim: GpuDimension, result: HirId) -> CompilerResult<()> {
        let intrinsic_name = match dim {
            GpuDimension::X => "llvm.nvvm.read.ptx.sreg.ctaid.x",
            GpuDimension::Y => "llvm.nvvm.read.ptx.sreg.ctaid.y",
            GpuDimension::Z => "llvm.nvvm.read.ptx.sreg.ctaid.z",
        };

        // Similar to thread_idx...
        self.compile_nvvm_sreg_intrinsic(intrinsic_name, result)
    }

    fn compile_block_dim(&mut self, dim: GpuDimension, result: HirId) -> CompilerResult<()> {
        let intrinsic_name = match dim {
            GpuDimension::X => "llvm.nvvm.read.ptx.sreg.ntid.x",
            GpuDimension::Y => "llvm.nvvm.read.ptx.sreg.ntid.y",
            GpuDimension::Z => "llvm.nvvm.read.ptx.sreg.ntid.z",
        };

        self.compile_nvvm_sreg_intrinsic(intrinsic_name, result)
    }

    fn compile_grid_dim(&mut self, dim: GpuDimension, result: HirId) -> CompilerResult<()> {
        let intrinsic_name = match dim {
            GpuDimension::X => "llvm.nvvm.read.ptx.sreg.nctaid.x",
            GpuDimension::Y => "llvm.nvvm.read.ptx.sreg.nctaid.y",
            GpuDimension::Z => "llvm.nvvm.read.ptx.sreg.nctaid.z",
        };

        self.compile_nvvm_sreg_intrinsic(intrinsic_name, result)
    }

    fn compile_nvvm_sreg_intrinsic(&mut self, name: &str, result: HirId) -> CompilerResult<()> {
        let context = self.base.context;
        let i32_ty = context.i32_type();

        let intrinsic_ty = i32_ty.fn_type(&[], false);
        let intrinsic = self.base.module().add_function(name, intrinsic_ty, None);

        let value = self.base.builder.build_call(intrinsic, &[], "sreg")?
            .try_as_basic_value()
            .left()
            .unwrap();

        self.base.value_map.insert(result, value);
        Ok(())
    }

    fn compile_sync_threads(&mut self) -> CompilerResult<()> {
        // __syncthreads: LLVM 21 names it llvm.nvvm.barrier.cta.sync.aligned.all
        // (barrier 0); llvm.nvvm.barrier0 is only auto-upgraded to it.
        let context = self.base.context;
        let void_ty = context.void_type();
        let i32_ty = context.i32_type();

        let intrinsic_ty = void_ty.fn_type(&[i32_ty.into()], false);
        let intrinsic = self.base.module().add_function(
            "llvm.nvvm.barrier.cta.sync.aligned.all",
            intrinsic_ty,
            None,
        );

        self.base.builder.build_call(intrinsic, &[i32_ty.const_zero().into()], "")?;
        Ok(())
    }

    fn compile_shared_mem_alloc(
        &mut self,
        result: HirId,
        element_type: &HirType,
        num_elements: u32,
        align: u32,
    ) -> CompilerResult<()> {
        let context = self.base.context;
        let elem_ty = self.base.translate_type(element_type)?;

        // Create array type
        let array_ty = elem_ty.array_type(num_elements);

        // Allocate in address space 3 (shared memory)
        let global = self.base.module().add_global(
            array_ty,
            Some(AddressSpace::from(3)),  // Shared memory
            &format!("shared_mem_{:?}", result),
        );

        // Mark as internal linkage (not exported)
        global.set_linkage(inkwell::module::Linkage::Internal);
        global.set_alignment(align);

        // Initialize to undef
        global.set_initializer(&array_ty.get_undef());

        // Store pointer
        let ptr = global.as_pointer_value();
        self.base.value_map.insert(result, ptr.into());

        Ok(())
    }

    fn compile_gpu_atomic(
        &mut self,
        op: GpuAtomicKind,
        result: HirId,
        ptr: HirId,
        value: HirId,
        scope: GpuAtomicScope,
    ) -> CompilerResult<()> {
        let ptr_val = self.base.get_value(ptr)?
            .into_pointer_value();
        let val = self.base.get_value(value)?;

        // Map to LLVM atomicrmw operation
        let rmw_op = match op {
            GpuAtomicKind::Add => AtomicRMWBinOp::Add,
            GpuAtomicKind::Sub => AtomicRMWBinOp::Sub,
            GpuAtomicKind::Min => AtomicRMWBinOp::Min,
            GpuAtomicKind::Max => AtomicRMWBinOp::Max,
            GpuAtomicKind::And => AtomicRMWBinOp::And,
            GpuAtomicKind::Or => AtomicRMWBinOp::Or,
            GpuAtomicKind::Xor => AtomicRMWBinOp::Xor,
            GpuAtomicKind::Exchange => AtomicRMWBinOp::Xchg,
            _ => return Err(CompilerError::CodeGen("Unsupported GPU atomic op".into())),
        };

        // Scope and ordering are independent in LLVM NVPTX: scope is a
        // syncscope ("block", "device", or the default system scope),
        // ordering is the atomic's own memory order.
        let syncscope = match scope {
            GpuAtomicScope::Block => "block",
            GpuAtomicScope::Device => "device",
            GpuAtomicScope::System => "",
        };
        let ordering = LLVMAtomicOrdering::LLVMMonotonic; // relaxed unless the HIR op asks for more

        let result_val = self.base.builder.build_atomicrmw(
            rmw_op,
            ptr_val,
            val.into_int_value(),
            ordering,
        )?;
        // inkwell has no syncscope argument; set it through llvm-sys
        // (LLVMSetAtomicSyncScopeID with LLVMGetSyncScopeID(syncscope)).
        self.set_sync_scope(result_val, syncscope);

        self.base.value_map.insert(result, result_val.into());
        Ok(())
    }

    fn compile_warp_shuffle(
        &mut self,
        mode: WarpShuffleMode,
        result: HirId,
        value: HirId,
        lane_or_delta: HirId,
        width: u32,
    ) -> CompilerResult<()> {
        let context = self.base.context;
        let i32_ty = context.i32_type();

        let val = self.base.get_value(value)?;
        let lane = self.base.get_value(lane_or_delta)?;

        // Determine intrinsic based on mode
        let intrinsic_name = match mode {
            WarpShuffleMode::Idx => "llvm.nvvm.shfl.sync.idx.i32",
            WarpShuffleMode::Up => "llvm.nvvm.shfl.sync.up.i32",
            WarpShuffleMode::Down => "llvm.nvvm.shfl.sync.down.i32",
            WarpShuffleMode::Xor => "llvm.nvvm.shfl.sync.bfly.i32",
        };

        // Create intrinsic type: (mask, value, lane/delta, width) -> value
        let intrinsic_ty = i32_ty.fn_type(
            &[i32_ty.into(), i32_ty.into(), i32_ty.into(), i32_ty.into()],
            false,
        );
        let intrinsic = self.base.module().add_function(intrinsic_name, intrinsic_ty, None);

        // Full warp mask
        let mask = i32_ty.const_int(0xFFFFFFFF, false);
        let width_val = i32_ty.const_int(width as u64, false);

        let result_val = self.base.builder.build_call(
            intrinsic,
            &[mask.into(), val.into(), lane.into(), width_val.into()],
            "shfl",
        )?
        .try_as_basic_value()
        .left()
        .unwrap();

        self.base.value_map.insert(result, result_val);
        Ok(())
    }
}
```

---

## Part 4: Low-Latency CPU Backend Optimizations

**Superseded (2026-09-25).** None of this part was built, and it is out of GPU scope. `CriticalPathOptimizer`, `compile_critical_path` and `SimdCodegen` do not exist, and the legacy per-pass `PassManager::add_*_pass` calls in 4.2 are gone in LLVM 21, which has only the new pass manager. CPU SIMD for kernels is covered by what is built instead: HIR `Vector` instructions on every backend, host width from `TargetVector::host` (`crates/compiler/src/target_vector.rs`), and the `auto_vectorize`, `loop_vectorize`, `reduction_vectorize` and `fma_contract` passes. The text is kept for reference only.

### 4.1 Critical Path Compiler Mode

For ultra-low-latency execution:

```rust
// In crates/compiler/src/critical_path.rs

use crate::hir::{HirFunction, HirExecutionConstraints, OptimizationLevel};

/// Critical path optimization pass
pub struct CriticalPathOptimizer {
    /// Target latency in nanoseconds
    target_latency_ns: u64,
    /// Allow unsafe optimizations
    allow_unsafe: bool,
}

impl CriticalPathOptimizer {
    pub fn new(target_latency_ns: u64) -> Self {
        Self {
            target_latency_ns,
            allow_unsafe: false,
        }
    }

    /// Optimize a function for minimal latency
    pub fn optimize(&mut self, func: &mut HirFunction) -> CompilerResult<()> {
        let constraints = func.execution_constraints.as_ref()
            .cloned()
            .unwrap_or_default();

        if constraints.critical_path {
            self.apply_critical_path_optimizations(func, &constraints)?;
        }

        Ok(())
    }

    fn apply_critical_path_optimizations(
        &mut self,
        func: &mut HirFunction,
        constraints: &HirExecutionConstraints,
    ) -> CompilerResult<()> {
        // 1. Remove all allocation
        if constraints.no_alloc {
            self.remove_allocations(func)?;
        }

        // 2. Inline all function calls
        if constraints.inline_always {
            self.force_inline_all(func)?;
        }

        // 3. Remove bounds checking
        if constraints.no_bounds_check {
            self.remove_bounds_checks(func)?;
        }

        // 4. Vectorize where possible
        self.vectorize_loops(func)?;

        // 5. Prefetch data
        self.insert_prefetches(func)?;

        // 6. Align hot loops
        self.align_hot_loops(func)?;

        Ok(())
    }

    fn remove_allocations(&self, func: &mut HirFunction) -> CompilerResult<()> {
        for (_, block) in func.blocks.iter_mut() {
            block.instructions.retain(|inst| {
                match inst {
                    HirInstruction::Alloca { .. } => {
                        log::warn!("Removing heap allocation in critical path");
                        false
                    }
                    HirInstruction::Call { callee: HirCallable::Intrinsic(Intrinsic::Malloc), .. } |
                    HirInstruction::Call { callee: HirCallable::Intrinsic(Intrinsic::Realloc), .. } => {
                        log::warn!("Removing heap allocation call in critical path");
                        false
                    }
                    _ => true,
                }
            });
        }
        Ok(())
    }

    fn force_inline_all(&self, func: &mut HirFunction) -> CompilerResult<()> {
        // Mark all calls for inlining
        func.attributes.inline = InlineHint::Always;
        func.attributes.always_inline = true;
        Ok(())
    }

    fn remove_bounds_checks(&self, func: &mut HirFunction) -> CompilerResult<()> {
        // Transform Load/Store with bounds checks to direct access
        // This is unsafe but explicitly requested
        for (_, block) in func.blocks.iter_mut() {
            for inst in &mut block.instructions {
                match inst {
                    // Remove conditional bounds checks before array access
                    // (This requires dataflow analysis in practice)
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn vectorize_loops(&self, func: &mut HirFunction) -> CompilerResult<()> {
        // Auto-vectorization hints for LLVM
        // Add loop metadata for vectorization
        Ok(())
    }

    fn insert_prefetches(&self, func: &mut HirFunction) -> CompilerResult<()> {
        // Analyze memory access patterns and insert prefetch instructions
        Ok(())
    }

    fn align_hot_loops(&self, func: &mut HirFunction) -> CompilerResult<()> {
        // Ensure loop headers are aligned to cache line boundaries
        Ok(())
    }
}
```

### 4.2 LLVM Backend Critical Path Extensions

```rust
// In crates/compiler/src/llvm_backend.rs - extend existing implementation

impl<'ctx> LLVMBackend<'ctx> {
    /// Compile with critical path optimizations
    pub fn compile_critical_path(
        &mut self,
        func: &HirFunction,
    ) -> CompilerResult<()> {
        let constraints = func.execution_constraints.as_ref();

        // Apply function attributes for aggressive optimization
        if let Some(fn_value) = self.functions.get(&func.id) {
            if constraints.map(|c| c.inline_always).unwrap_or(false) {
                fn_value.add_attribute(
                    inkwell::attributes::AttributeLoc::Function,
                    self.context.create_string_attribute("alwaysinline", ""),
                );
            }

            if constraints.map(|c| c.no_alloc).unwrap_or(false) {
                // Mark as not allocating
                fn_value.add_attribute(
                    inkwell::attributes::AttributeLoc::Function,
                    self.context.create_string_attribute("noalloc", ""),
                );
            }

            if constraints.map(|c| c.critical_path).unwrap_or(false) {
                // Mark as hot
                fn_value.add_attribute(
                    inkwell::attributes::AttributeLoc::Function,
                    self.context.create_string_attribute("hot", ""),
                );

                // Disable stack protection
                fn_value.add_attribute(
                    inkwell::attributes::AttributeLoc::Function,
                    self.context.create_string_attribute("nossp", ""),
                );
            }
        }

        Ok(())
    }

    /// Add LLVM optimization passes for low-latency code
    pub fn add_low_latency_passes(&self, pass_manager: &PassManager<Module<'ctx>>) {
        // Aggressive inlining
        pass_manager.add_always_inliner_pass();

        // SROA: Scalar Replacement of Aggregates (eliminate structs)
        pass_manager.add_scalar_repl_aggregates_pass();

        // GVN: Global Value Numbering (eliminate redundant loads)
        pass_manager.add_gvn_pass();

        // Loop vectorization
        pass_manager.add_loop_vectorize_pass();

        // SLP vectorization (straight-line code)
        pass_manager.add_slp_vectorize_pass();

        // Instruction combining
        pass_manager.add_instruction_combining_pass();

        // Tail call optimization
        pass_manager.add_tail_call_elimination_pass();

        // Dead code elimination
        pass_manager.add_aggressive_dce_pass();
    }
}
```

### 4.3 SIMD Vectorization Support

```rust
// In crates/compiler/src/simd.rs

/// SIMD width detection and code generation
pub struct SimdCodegen {
    /// Target vector width in bits
    vector_width: u32,
    /// Available SIMD features
    features: SimdFeatures,
}

#[derive(Debug, Clone, Default)]
pub struct SimdFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512vl: bool,
    pub neon: bool,  // ARM
}

impl SimdCodegen {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;

            let features = SimdFeatures {
                sse: is_x86_feature_detected!("sse"),
                sse2: is_x86_feature_detected!("sse2"),
                sse3: is_x86_feature_detected!("sse3"),
                ssse3: is_x86_feature_detected!("ssse3"),
                sse4_1: is_x86_feature_detected!("sse4.1"),
                sse4_2: is_x86_feature_detected!("sse4.2"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512vl: is_x86_feature_detected!("avx512vl"),
                ..Default::default()
            };

            let vector_width = if features.avx512f { 512 }
                else if features.avx2 { 256 }
                else if features.sse2 { 128 }
                else { 64 };

            Self { vector_width, features }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                vector_width: 128,  // NEON is 128-bit
                features: SimdFeatures { neon: true, ..Default::default() },
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                vector_width: 64,
                features: SimdFeatures::default(),
            }
        }
    }

    /// Get recommended vectorization width for a scalar type
    pub fn recommend_width(&self, scalar_bits: u32) -> u32 {
        self.vector_width / scalar_bits
    }

    /// Generate target features string for LLVM
    pub fn llvm_features(&self) -> String {
        let mut features = Vec::new();

        if self.features.avx512f { features.push("+avx512f"); }
        if self.features.avx512vl { features.push("+avx512vl"); }
        if self.features.avx2 { features.push("+avx2"); }
        if self.features.avx { features.push("+avx"); }
        if self.features.sse4_2 { features.push("+sse4.2"); }
        if self.features.sse4_1 { features.push("+sse4.1"); }

        features.join(",")
    }
}
```

---

## Part 5: CUDA Driver Runtime Integration

**Status (2026-09-25): not built. The driver binding is decided (5.0); the hand-written wrapper in 5.1 and 5.2 is superseded by it** and kept only as a map of the responsibilities the runtime carries (module cache, streams, memory pool).

### 5.0 Driver binding: cudarc

The runtime uses `cudarc` (MIT OR Apache-2.0) with `dynamic-loading`, behind the `cuda` feature:

- **Context**: `CudaContext::new(ordinal)`, which retains the device's primary context. Do not call `cuCtxCreate` directly; it exists in several versioned driver entry points.
- **Module load**: PTX text from the NVPTX backend (or a cubin, or a Tile IR bytecode image) through `CudaContext::load_module`, backed by `cuModuleLoadData`. The driver JIT-compiles PTX on load, so no toolkit is needed at run time.
- **Function lookup**: `CudaModule::load_function(name)`.
- **Launch**: the raw `result::launch_kernel(f, grid, block, smem, stream, &mut [*mut c_void])`, with the argument array built from the kernel's HIR signature. The typed `launch_builder` is for hand-written host code.
- **Libraries**: `cudarc` wraps cuBLAS and cuBLASLt under the same loading model; GEMM-shaped kernels dispatch to them (see the status section).

Why `cudarc`: it builds on macOS and on machines without CUDA (pregenerated bindings for CUDA 11.4 through 13.x), needs only `libcuda` at run time, and its load and launch API has the same shape as NVIDIA's `cuda_core`, so a later switch is cheap.

### 5.0.1 NVIDIA CUDA Rust (2026-09-08)

NVIDIA announced CUDA Rust on 2026-09-08 as two tracks for writing GPU kernels in Rust. Facts below were read on 2026-09-25; both projects are early (cuda-oxide calls itself alpha, cutile-rs "early stage") and their requirements moved within weeks of the announcement.

- **cuda-oxide** (NVlabs/cuda-oxide, Apache-2.0): a custom rustc codegen backend for the SIMT model. `#[kernel]` functions go from Rust MIR through Pliron IR dialects (`dialect-mir`, `dialect-nvvm`, `dialect-ptx`) to LLVM IR, then through an external `llc -march=nvptx64` to PTX. Needs a pinned nightly (the announcement said nightly-2026-04-03; the repository has since moved to nightly-2026-08-28), CUDA Toolkit 13.0+ and driver R580+ (the announcement said CUDA 12.x+), clang-21/libclang, Linux, and `sm_80` or newer. Host crates: `cuda_host` (`#[cuda_module]` typed loading) and `cuda_device` stay in cuda-oxide and are not on crates.io.
- **cutile-rs** (NVlabs/cutile-rs, Apache-2.0, crates.io): the tile model on stable Rust 1.89+. Kernels compile through CUDA Tile IR, recommended with CUDA 13.3; Linux and `sm_80` or newer. It now also hosts `cuda_core` (`CudaContext`, `DeviceBuffer`, `LaunchConfig1D`, `load_module_from_ptx_src`, `launch_kernel`), shared by both tracks, and `cutile-ir`, a pure Rust Tile IR builder and bytecode writer with no LLVM or C++ dependency.
- NVIDIA plans interop with CUDA C++ and CUDA Python.

What Zyntax reuses and what it does not:

| Piece | Decision | Why |
|-------|----------|-----|
| cuda-oxide's rustc front end | Not used | Zyntax kernels are not Rust source; HIR is the kernel IR. |
| `cuda-oxide-codegen` (its rustc-independent PTX backend) | Not used | Its input is a Rust-MIR-shaped Pliron dialect pinned to one git revision and documented as not a stable interchange format; it is unpublished and shells out to `llc` and `opt`. Zyntax already has the same LLVM NVPTX backend in process. |
| `cuda_core` host crate | Not now; `cudarc` instead | Its `cuda-bindings` dependency runs bindgen against a CUDA 13 toolkit and libclang at build time, which the light build cannot require. It is a fork of cudarc's driver layer with the same load and launch shape, so switching later is cheap if NVIDIA's interop settles on it. |
| `cutile-ir` | Used, for the tile track | CUDA Tile IR is a documented, versioned bytecode that conforming drivers load. A second lowering from HIR to Tile IR serves GEMM, attention and fused norms after the SIMT path works. Driver-JIT of bytecode must be verified on real hardware first, since cutile-rs itself only compiles through the toolkit's `tileiras`. |
| cuda-oxide's NVVM intrinsic catalog and PTX-floor table | Reference | Which NVVM intrinsics exist per SM and which minimum PTX each target needs, for when Zyntax grows tensor-core kernels (TMA, wgmma, mbarrier). |
| cutile-rs's disjoint-partition safety model | Reference | The design reference for how `@kernel` outputs are split across threads without data races. |

Decided (2026-09-25): transcendental math in NVIDIA kernels (`exp`, `log`, `sin` and the rest) is generated by Zyntax from NVVM approximate intrinsics plus refinement, so a kernel needs only the driver. `libdevice` ships with the CUDA toolkit under NVIDIA's EULA and is not used.

### 5.1 Runtime Module Management

```rust
// In crates/compiler/src/cuda_runtime.rs

use std::ffi::CString;
use std::ptr;

/// CUDA driver API wrapper for kernel execution
pub struct CudaRuntime {
    /// CUDA context
    context: CUcontext,
    /// Loaded modules (PTX -> cuModule)
    modules: HashMap<String, CudaModule>,
    /// Active streams
    streams: Vec<CudaStream>,
    /// Device properties
    device_props: CudaDeviceProps,
}

pub struct CudaModule {
    module: CUmodule,
    kernels: HashMap<String, CUfunction>,
}

pub struct CudaStream {
    stream: CUstream,
    /// Events for timing/sync
    events: Vec<CUevent>,
}

#[derive(Debug, Clone)]
pub struct CudaDeviceProps {
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory: usize,
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_multiprocessor: u32,
    pub warp_size: u32,
    pub shared_memory_per_block: usize,
    pub registers_per_block: u32,
}

impl CudaRuntime {
    /// Initialize CUDA runtime
    pub fn new() -> Result<Self, CudaError> {
        // Initialize CUDA driver
        unsafe {
            let result = cuInit(0);
            if result != 0 {
                return Err(CudaError::InitFailed(result));
            }
        }

        // Get device
        let mut device: CUdevice = 0;
        unsafe {
            cuDeviceGet(&mut device, 0)?;
        }

        // Create context
        let mut context: CUcontext = ptr::null_mut();
        unsafe {
            cuCtxCreate(&mut context, 0, device)?;
        }

        // Query device properties
        let device_props = Self::query_device_props(device)?;

        Ok(Self {
            context,
            modules: HashMap::new(),
            streams: Vec::new(),
            device_props,
        })
    }

    /// Load PTX code as a module
    pub fn load_ptx(&mut self, name: &str, ptx: &str) -> Result<(), CudaError> {
        let ptx_cstring = CString::new(ptx)?;

        let mut module: CUmodule = ptr::null_mut();
        unsafe {
            cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const _)?;
        }

        self.modules.insert(name.to_string(), CudaModule {
            module,
            kernels: HashMap::new(),
        });

        Ok(())
    }

    /// Get kernel function from module
    pub fn get_kernel(&mut self, module_name: &str, kernel_name: &str) -> Result<CUfunction, CudaError> {
        let module = self.modules.get_mut(module_name)
            .ok_or(CudaError::ModuleNotFound)?;

        if let Some(&func) = module.kernels.get(kernel_name) {
            return Ok(func);
        }

        // Load kernel
        let kernel_cstring = CString::new(kernel_name)?;
        let mut func: CUfunction = ptr::null_mut();

        unsafe {
            cuModuleGetFunction(&mut func, module.module, kernel_cstring.as_ptr())?;
        }

        module.kernels.insert(kernel_name.to_string(), func);
        Ok(func)
    }

    /// Launch a kernel
    pub fn launch_kernel(
        &self,
        kernel: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: Option<&CudaStream>,
        args: &[*mut std::ffi::c_void],
    ) -> Result<(), CudaError> {
        let stream_handle = stream.map(|s| s.stream).unwrap_or(ptr::null_mut());

        unsafe {
            cuLaunchKernel(
                kernel,
                grid_dim.0, grid_dim.1, grid_dim.2,
                block_dim.0, block_dim.1, block_dim.2,
                shared_mem_bytes,
                stream_handle,
                args.as_ptr() as *mut _,
                ptr::null_mut(),
            )?;
        }

        Ok(())
    }

    /// Create a new execution stream
    pub fn create_stream(&mut self) -> Result<usize, CudaError> {
        let mut stream: CUstream = ptr::null_mut();
        unsafe {
            cuStreamCreate(&mut stream, 0)?;
        }

        let idx = self.streams.len();
        self.streams.push(CudaStream {
            stream,
            events: Vec::new(),
        });

        Ok(idx)
    }

    /// Synchronize all pending operations
    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe {
            cuCtxSynchronize()?;
        }
        Ok(())
    }

    /// Allocate device memory
    pub fn malloc(&self, size: usize) -> Result<CUdeviceptr, CudaError> {
        let mut ptr: CUdeviceptr = 0;
        unsafe {
            cuMemAlloc(&mut ptr, size)?;
        }
        Ok(ptr)
    }

    /// Free device memory
    pub fn free(&self, ptr: CUdeviceptr) -> Result<(), CudaError> {
        unsafe {
            cuMemFree(ptr)?;
        }
        Ok(())
    }

    /// Copy data to device
    pub fn memcpy_to_device<T>(&self, dst: CUdeviceptr, src: &[T]) -> Result<(), CudaError> {
        let size = std::mem::size_of_val(src);
        unsafe {
            cuMemcpyHtoD(dst, src.as_ptr() as *const _, size)?;
        }
        Ok(())
    }

    /// Copy data from device
    pub fn memcpy_from_device<T>(&self, dst: &mut [T], src: CUdeviceptr) -> Result<(), CudaError> {
        let size = std::mem::size_of_val(dst);
        unsafe {
            cuMemcpyDtoH(dst.as_mut_ptr() as *mut _, src, size)?;
        }
        Ok(())
    }

    fn query_device_props(device: CUdevice) -> Result<CudaDeviceProps, CudaError> {
        // Query various device properties via cuDeviceGetAttribute
        // ... implementation details ...
        todo!()
    }
}

impl Drop for CudaRuntime {
    fn drop(&mut self) {
        // Cleanup streams
        for stream in &self.streams {
            unsafe { cuStreamDestroy(stream.stream); }
        }

        // Cleanup modules
        for (_, module) in &self.modules {
            unsafe { cuModuleUnload(module.module); }
        }

        // Destroy context
        unsafe { cuCtxDestroy(self.context); }
    }
}

#[derive(Debug)]
pub enum CudaError {
    InitFailed(i32),
    ModuleNotFound,
    KernelNotFound,
    LaunchFailed(i32),
    MemoryError(i32),
    Other(String),
}
```

### 5.2 Memory Management

```rust
// In crates/compiler/src/cuda_memory.rs

/// GPU memory allocator with pooling
pub struct GpuMemoryPool {
    /// Pool of pre-allocated buffers by size class
    pools: HashMap<usize, Vec<CUdeviceptr>>,
    /// Active allocations
    active: HashMap<CUdeviceptr, usize>,
    /// Runtime reference
    runtime: Arc<CudaRuntime>,
    /// Total allocated bytes
    total_allocated: usize,
    /// Memory limit
    memory_limit: usize,
}

impl GpuMemoryPool {
    pub fn new(runtime: Arc<CudaRuntime>, memory_limit: usize) -> Self {
        Self {
            pools: HashMap::new(),
            active: HashMap::new(),
            runtime,
            total_allocated: 0,
            memory_limit,
        }
    }

    /// Allocate from pool or create new allocation
    pub fn alloc(&mut self, size: usize) -> Result<CUdeviceptr, CudaError> {
        // Round up to next power of 2 (size class)
        let size_class = size.next_power_of_two();

        // Try to get from pool
        if let Some(pool) = self.pools.get_mut(&size_class) {
            if let Some(ptr) = pool.pop() {
                self.active.insert(ptr, size_class);
                return Ok(ptr);
            }
        }

        // Allocate new
        if self.total_allocated + size_class > self.memory_limit {
            return Err(CudaError::MemoryError(-1));
        }

        let ptr = self.runtime.malloc(size_class)?;
        self.active.insert(ptr, size_class);
        self.total_allocated += size_class;

        Ok(ptr)
    }

    /// Return allocation to pool
    pub fn free(&mut self, ptr: CUdeviceptr) {
        if let Some(size_class) = self.active.remove(&ptr) {
            self.pools.entry(size_class).or_default().push(ptr);
        }
    }

    /// Actually free all pooled memory
    pub fn flush(&mut self) {
        for (_, ptrs) in self.pools.drain() {
            for ptr in ptrs {
                let _ = self.runtime.free(ptr);
            }
        }
        self.total_allocated = 0;
    }
}

/// Unified memory manager for heterogeneous CPU+GPU workloads
pub struct UnifiedMemory {
    /// GPU memory pool
    gpu_pool: GpuMemoryPool,
    /// Pinned host memory for fast transfers
    pinned_allocations: Vec<(*mut u8, usize)>,
    /// Registered host memory (page-locked)
    registered_memory: Vec<(*mut u8, usize)>,
}

impl UnifiedMemory {
    /// Allocate pinned host memory (for fast CPU<->GPU transfers)
    pub fn alloc_pinned(&mut self, size: usize) -> Result<*mut u8, CudaError> {
        let mut ptr: *mut u8 = ptr::null_mut();
        unsafe {
            cuMemAllocHost(&mut ptr as *mut _ as *mut _, size)?;
        }
        self.pinned_allocations.push((ptr, size));
        Ok(ptr)
    }

    /// Register existing host memory for fast transfers
    pub fn register_host_memory(&mut self, ptr: *mut u8, size: usize) -> Result<(), CudaError> {
        unsafe {
            cuMemHostRegister(ptr as *mut _, size, 0)?;
        }
        self.registered_memory.push((ptr, size));
        Ok(())
    }
}
```

---

## Part 6: Integration with ZynML and DSLs

### 6.1 DSL Kernel Generation Pipeline

**Status (2026-09-25):** the source form below is written in the syntax the grammar accepts today: modifiers after the argument list, a braced single-variable `for`, and plain assignment (ZynML has no `+=`). It parses, but lowering ignores `@kernel(matmul)`, `@device` and `@workgroup`. The block is classified Generic, and because it has no direct `yield` it becomes a call of the undefined `$Zyntax$compute`, so running it fails on the Cranelift tier with an unresolved-symbol panic; that is a bug tracked in git-bug. The in-body `@workgroup(16, 16)` directive and the two-variable `for i in 0..M, j in 0..N:` loop of the 2025-12 draft do not parse: the only in-body directive is `@kernel <identifier>`, and `for` takes one variable and a braced block. The pipeline below is the target. Python and Lua kernels enter the same pipeline at the typed-AST step through the `zyntax` module. On the CPU side the path is HIR vector instructions plus `parallel_dispatch` (behind `ZYNTAX_PARALLEL_LOOPS=1` today), not OpenMP, and a kernel on a device the build cannot reach is a compile or launch error, not a silent CPU run.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│             Kernel source (ZynML compute, or @kernel via `zyntax`)          │
│                                                                             │
│  compute(A, B) @kernel(matmul) @device("cuda:0") @workgroup(16, 16) {       │
│      for i in 0..M {                                                        │
│          for j in 0..N {                                                    │
│              mut sum = 0.0                                                  │
│              for k in 0..K {                                                │
│                  sum = sum + A[i * K + k] * B[k * N + j]                    │
│              }                                                              │
│              out[i * N + j] = sum                                           │
│          }                                                                  │
│      }                                                                      │
│  }                                                                          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ZynPEG Grammar Actions                                │
│                                                                              │
│  Parse @kernel, @device, @workgroup annotations                             │
│  Type the kernel subset; record kernel kind, device, workgroup               │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HIR Lowering with GPU Primitives                      │
│                                                                              │
│  - Map loop indices to ThreadIdx/BlockIdx                                    │
│  - Insert SyncThreads for shared memory access                               │
│  - Generate SharedMemAlloc for tiled algorithms                              │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│  CPU Backend (@device("cpu"))    │  │  NVPTX Backend                   │
│                                  │  │                                  │
│  Generate vectorized loops       │  │  Generate PTX via LLVM IR        │
│  HIR Vector instructions         │  │  ptx_kernel CC, nvvm.maxntid     │
│  parallel_dispatch bands         │  │  Emit to PTX string              │
└──────────────────────────────────┘  └────────────────┬─────────────────┘
                                                       │
                                                       ▼
                                      ┌──────────────────────────────────┐
                                      │  CUDA Runtime                    │
                                      │                                  │
                                      │  Load PTX module                 │
                                      │  Get kernel function             │
                                      │  Launch with grid/block dims     │
                                      └──────────────────────────────────┘
```

### 6.2 Kernel Fusion Optimization

**Status (2026-09-25): not built.** No `kernel_fusion.rs` exists.

```rust
// In crates/compiler/src/kernel_fusion.rs

/// Fuse multiple kernels into a single launch
pub struct KernelFusionPass {
    /// Maximum shared memory to allow fusion
    max_shared_memory: usize,
    /// Maximum register pressure to allow fusion
    max_registers: u32,
}

impl KernelFusionPass {
    /// Attempt to fuse consecutive kernel launches
    pub fn fuse_kernels(&self, kernels: &[HirFunction]) -> Vec<HirFunction> {
        let mut fused = Vec::new();
        let mut current_group: Vec<&HirFunction> = Vec::new();

        for kernel in kernels {
            if self.can_fuse_with_group(&current_group, kernel) {
                current_group.push(kernel);
            } else {
                // Emit fused kernel for current group
                if !current_group.is_empty() {
                    fused.push(self.create_fused_kernel(&current_group));
                }
                current_group = vec![kernel];
            }
        }

        // Don't forget last group
        if !current_group.is_empty() {
            fused.push(self.create_fused_kernel(&current_group));
        }

        fused
    }

    fn can_fuse_with_group(&self, group: &[&HirFunction], kernel: &HirFunction) -> bool {
        // Check data dependencies
        // Check resource usage (shared memory, registers)
        // Check launch configuration compatibility
        true // Placeholder
    }

    fn create_fused_kernel(&self, kernels: &[&HirFunction]) -> HirFunction {
        // Merge kernel bodies
        // Handle shared memory layout
        // Insert synchronization points
        todo!()
    }
}
```

---

## Part 7: Implementation Roadmap

**Superseded (2026-09-25):** the 2025-12 roadmap scheduled twelve weeks across Q1 and Q2 2026; none of it was built. The phases keep their order, without dates, and follow the Metal backend (09-gpu-compute-system.md) because NVIDIA execution needs hardware the project does not have yet. Everything below is not started unless marked.

### Phase 0: Kernel front end (shared with Metal)

| Task | Description | Status |
|------|-------------|--------|
| Kernel typing | Type `compute()` bodies and `@kernel` functions as the typed subset; dynamic code is a compile error | Not started (the type checker returns a fresh type variable for `compute`) |
| Read the modifiers | Lowering consumes `@kernel(x)`, `@device`, `@workgroup` | Not started (parsed and dropped) |
| No runtime dispatch | Unrecognised kernel shapes become compile errors instead of calls to `$Zyntax$compute` | Bug, tracked in git-bug |
| `zyntax` module | `from zyntax import kernel` in Python, a `zyntax` table in Lua | Not started |

### Phase 1: Foundation

| Task | Description |
|------|-------------|
| HIR GPU Primitives | Add thread/block indexing, barriers, shared memory to HIR |
| NVPTX Target Setup | `Target::initialize_nvptx`, `nvptx64-nvidia-cuda`, `sm_80` floor, under `llvm-backend` |
| Basic PTX Emission | Compile a simple kernel to PTX; golden-test the PTX on the Mac and the NUC |

**Milestone:** PTX for an elementwise kernel, verified by LLVM, with no GPU present.

### Phase 2: Core Backend

| Task | Description |
|------|-------------|
| Thread Indexing | ThreadIdx/BlockIdx/BlockDim/GridDim through `llvm.nvvm.read.ptx.sreg.*` |
| Synchronization | `llvm.nvvm.barrier.cta.sync.aligned.all` |
| Shared Memory | Address space 3 allocations |
| Atomic Operations | `atomicrmw` with syncscope |
| Warp Primitives | shuffle, vote, match |

**Milestone:** parallel reduction kernel correct on an `sm_80`+ Linux machine.

### Phase 3: Runtime Integration

| Task | Description |
|------|-------------|
| CUDA Runtime | `cudarc` context, module load from PTX, function lookup, raw launch |
| Memory | Device buffers, pinned host memory, a pool |
| Streams | Async execution tied to Zyntax fibers and async, not a separate future type |
| DSL Integration | ZynML `compute()` and `zyntax` `@kernel` reach the backend |

**Milestone:** end-to-end `@kernel` on an NVIDIA GPU, benchmarked per the status section.

### Phase 4: Advanced Features

| Task | Description |
|------|-------------|
| Tile IR track | HIR to CUDA Tile IR bytecode through `cutile-ir` for GEMM, attention, fused norms |
| Tensor Cores | MMA operations on the SIMT path where Tile IR does not serve |
| Async Copy | Global to shared staging |
| ~~Critical Path Optimizer~~ | Superseded: out of GPU scope (Part 4) |
| ~~SIMD Codegen~~ | Superseded: CPU SIMD is built as HIR vectors and the vectorization passes |

---

## Appendix A: LLVM NVPTX Intrinsics Reference

### Thread Indexing
```llvm
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
```

### Synchronization

In LLVM 21, `@llvm.nvvm.barrier0()` is auto-upgraded to `@llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)`; emit the new name.

```llvm
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32)
declare void @llvm.nvvm.barrier0()  ; legacy, auto-upgraded
declare void @llvm.nvvm.barrier.sync(i32)
declare void @llvm.nvvm.membar.cta()
declare void @llvm.nvvm.membar.gl()
declare void @llvm.nvvm.membar.sys()
```

### Warp Operations
```llvm
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32)
declare i32 @llvm.nvvm.shfl.sync.up.i32(i32, i32, i32, i32)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)
declare i32 @llvm.nvvm.vote.ballot.sync(i32, i1)
declare i1 @llvm.nvvm.vote.any.sync(i32, i1)
declare i1 @llvm.nvvm.vote.all.sync(i32, i1)
```

### Tensor Cores (Ampere+)
```llvm
declare {<2 x half>, <2 x half>, <2 x half>, <2 x half>}
    @llvm.nvvm.wmma.load.a.sync.row.m16n16k16.f16(ptr addrspace(1), i32)
declare void @llvm.nvvm.wmma.store.d.sync.row.m16n16k16.f32(ptr addrspace(1),
    float, float, float, float, float, float, float, float, i32)
declare {float, float, float, float, float, float, float, float}
    @llvm.nvvm.wmma.mma.sync.row.row.m16n16k16.f32.f16(
        <2 x half>, <2 x half>, <2 x half>, <2 x half>,
        <2 x half>, <2 x half>, <2 x half>, <2 x half>,
        float, float, float, float, float, float, float, float)
```

---

## Appendix B: Performance Targets

**Superseded (2026-09-25):** the 2025-12 appendix listed undated A100 and PCIe figures and CPU latency targets as fact. They were not measured and do not apply to the machines available: the Mac is unified memory with no host-to-device copy for shared buffers, and neither the Mac nor the NUC has an NVIDIA GPU.

Kernels are judged by the benchmark matrix in the status section: PyTorch eager and `torch.compile`, then numba, JAX/XLA, Taichi, numpy and Triton, then candle; CPU on the Mac and the NUC, GPU on the Mac through Metal; steady state and time to first result both reported. NVIDIA rows (PyTorch CUDA, Triton, candle CUDA) are added when an `sm_80`+ Linux machine is available. Numbers go in benchmark results with their date and machine, not in this document.

---

## References

- [LLVM NVPTX Backend](https://llvm.org/docs/NVPTXUsage.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA NVVM IR Specification](https://docs.nvidia.com/cuda/nvvm-ir-spec/)
- [Tensor Core Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Triton Language](https://github.com/openai/triton) - Reference for DSL design
- [Halide](https://halide-lang.org/) - Reference for scheduling DSL
- [Introducing CUDA Rust: two tracks for writing GPU kernels](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/) (NVIDIA, 2026-09-08)
- [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) - SIMT kernels in Rust through Pliron and LLVM NVPTX
- [NVlabs/cutile-rs](https://github.com/NVlabs/cutile-rs) - tile kernels through CUDA Tile IR; hosts `cuda_core` and `cutile-ir`
- [CUDA Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/) and [NVIDIA/cuda-tile](https://github.com/NVIDIA/cuda-tile)
- [cudarc](https://github.com/chelsea0x3b/cudarc) - CUDA driver binding with dynamic loading

---

*Last Updated: 2026-09-25*
*Version: 1.1*
*Status: Not built; direction revised 2026-09-25 (see Status and direction)*
