# Beadie Integration Plan — Hot-Tier with OSR

## Objective

Replace the home-grown promotion/profiling logic in [`tiered_backend.rs`](src/tiered_backend.rs)
with `beadie`'s [`TieredAdapter`](https://docs.rs/beadie-backend/0.3.0/beadie_backend/struct.TieredAdapter.html),
gaining real on-stack replacement (OSR) so long-running single invocations
(scripts, training loops, event loops) can be promoted mid-execution.

## What Beadie Provides

`beadie` (v0.3.0) is split into four crates:

| Crate | Role |
|---|---|
| `beadie-core` | `Bead`, `Chain`, `Broker`, `OsrEntry`, hotness/deopt policies |
| `beadie-backend` | `JitBackend` trait, `BackendAdapter`, `TieredAdapter` |
| `beadie-cranelift` | Concrete `JitBackend` impl wrapping Cranelift's `JITModule` |
| `beadie` | Top-level re-export crate |

The pieces relevant to us:

### `JitBackend` trait

```rust
pub trait JitBackend: Send + Sync + 'static {
    type FunctionDef: Send + 'static;       // vendor IR container
    type Error: std::error::Error + Send + Sync + 'static;
    fn compile(&self, bead: &Arc<Bead>, def: Self::FunctionDef) -> Result<*mut (), Self::Error>;
}
```

Implemented once per backend. Returns a native function pointer.

### `TieredAdapter`

```rust
TieredAdapter::new(policies: Vec<Box<dyn HotnessPolicy>>) -> Self
adapter.register(core: CoreHandle, on_invalidate: ...) -> TieredBound
adapter.on_invoke(&bound, |tier_idx, &Arc<Bead>| -> *mut ()) -> Option<*mut ()>
adapter.force_promote(&bound, target_tier, compile) -> bool
adapter.on_bailout(&bound, BailoutInfo) -> DeoptDecision
```

Each tier has its own promotion policy and a dedicated worker thread.
Tier 0 uses the primary `Broker` (state transition Interpreted → Compiled);
tier 1+ use `PromotionBroker`s (state stays Compiled, code pointer is
atomically swapped via `Bead::swap_compiled`).

### `BackendAdapter::on_invoke_osr`

```rust
adapter.on_invoke_osr(&bound, |bead| -> OsrBuild<FunctionDef>) -> Option<*mut ()>

pub struct OsrBuild<D> {
    pub def: D,                // backend IR for the main entry
    pub osr: Vec<OsrEntry>,    // one entry per hot loop header
}
pub struct OsrEntry {
    pub site: u64,             // opaque key (bytecode offset / block id)
    pub code: *mut (),         // native pointer to the loop-header label
}
```

The runtime probes back-edges with `bead.osr_entry(site) -> Option<*mut ()>`
to transfer a live interpreter frame into compiled code without unwinding.

### Hotness & Deopt Policies

- `ThresholdPolicy::new(N)` — compile after N invocations
- `ThresholdPolicy::new(N).queue_ahead(K)` — submit at N-K to absorb compile latency
- `TieredPolicy` — escalating thresholds per generation
- `TieredDeoptPolicy::new(blacklist_after, revert_after)` — cap promotion on bailout

## What `tiered_backend.rs` Has Now

[`crates/compiler/src/tiered_backend.rs`](src/tiered_backend.rs) (693 lines) implements:

- 3 fixed tiers: `Baseline`, `Standard`, `Optimized`
- Hand-rolled call counters in `ProfileData` (with `is_warm`/`is_hot`)
- `optimization_queue: VecDeque<(HirId, OptimizationTier)>` + `Mutex`
- `optimizing: HashSet<HirId>` to avoid re-enqueueing
- One background `worker_handle` thread that polls the queue every 10ms
- `function_pointers: HashMap<HirId, usize>` — written under `RwLock`,
  read on every call without atomics
- LLVM tier 2 wired via `LLVMJitBackend` (feature-gated)

Used by `TieredRuntime` in [`runtime.rs:3250`](../zyntax_embed/src/runtime.rs).

### Gaps vs Beadie

1. **No OSR** — only call-site promotion; long single invocations never tier up.
   `tiered_backend.rs:30` lists OSR as a future extension.
2. **No deoptimization** — once promoted, no way to bail back to baseline on
   speculation failure.
3. **Function pointer reads aren't atomic** — `RwLock<HashMap>` per call.
   Beadie's `Bead::compiled()` is one acquire-load.
4. **One worker thread for all tiers** — beadie spawns one per tier, so a
   slow LLVM compile on a hot function doesn't block a baseline Cranelift
   compile on a different function.
5. **Manual epoch management** — old function pointers are leaked.
   Beadie uses `crossbeam-epoch` for safe reclamation.
6. **No `queue_ahead`** — we always wait for the full threshold before
   submitting, paying the full compile latency on the first hot call.

## Integration Strategy

### Phase 1 — Replace tier coordination (no ZynML behavior change)

Drop the hand-rolled coordinator. Keep `CraneliftBackend` (and the LLVM
backend behind the feature flag) but route tier promotion through
`TieredAdapter`.

**Steps:**

1. Add `beadie = "0.3"` and `beadie-backend = "0.3"` to
   [`crates/compiler/Cargo.toml`](Cargo.toml). Skip `beadie-cranelift` —
   we already have a Cranelift backend with extensive ZRTL/DynamicBox
   plumbing that the upstream adapter doesn't know about.

2. Implement `JitBackend` for a thin wrapper around our existing
   `CraneliftBackend`:

   ```rust
   // crates/compiler/src/beadie_adapter.rs
   pub struct ZyntaxCraneliftBackend { inner: Arc<Mutex<CraneliftBackend>> }

   impl JitBackend for ZyntaxCraneliftBackend {
       type FunctionDef = ZyntaxFunctionDef; // (HirId, HirFunction, OptimizationTier)
       type Error = CompilerError;
       fn compile(&self, _bead: &Arc<Bead>, def: Self::FunctionDef) -> Result<*mut (), CompilerError> {
           // call inner.compile_function(def.id, &def.function) at the requested tier
           // return inner.get_function_ptr(def.id) as *mut ()
       }
   }
   ```

   Implement the same for `LLVMJitBackend` behind `cfg(feature = "llvm-backend")`.

3. Replace `TieredBackend` internals with:

   ```rust
   pub struct TieredBackend {
       adapter: TieredAdapter,
       beads: HashMap<HirId, TieredBound>,
       cranelift: Arc<ZyntaxCraneliftBackend>,
       #[cfg(feature = "llvm-backend")]
       llvm: Option<Arc<ZyntaxLlvmBackend>>,
       module: Arc<RwLock<Option<HirModule>>>,
   }
   ```

   Each compiled function owns one `TieredBound`. The `compile` closure
   passed to `on_invoke` dispatches by tier index:
   `0 => cranelift baseline, 1 => cranelift speed, 2 => llvm`.

4. `record_call(func_id)` becomes:

   ```rust
   if let Some(bound) = self.beads.get(&func_id) {
       self.adapter.on_invoke(bound, |tier, bead| { /* compile dispatch */ })
   }
   ```

5. `get_function_pointer(func_id)` returns `bound.bead().compiled()` —
   one atomic load, no `RwLock`.

6. Map our `OptimizationTier` enum to tier indices (0/1/2). Map our
   `TieredConfig::development/production/production_llvm` to
   `Vec<Box<dyn HotnessPolicy>>` constructors:

   ```rust
   fn dev_policies() -> Vec<Box<dyn HotnessPolicy>> {
       vec![
           Box::new(ThresholdPolicy::new(10)),     // baseline (low — fast iteration)
           Box::new(ThresholdPolicy::new(100).queue_ahead(20)),
           Box::new(ThresholdPolicy::new(1000).queue_ahead(100)),
       ]
   }
   ```

**Outcome:** ZynML programs run identically. The `record_call` hot path
becomes a single `tick()` + atomic load when the function is already at
the highest tier. Background compilation gets dedicated workers per tier.

### Phase 2 — Wire OSR for long-running calls

OSR matters most for ZynML's actual workloads: training/inference loops
where one call to `train()` runs for minutes and never returns. With only
call-site promotion, those calls never tier up.

**Required from the codegen side:**

1. **Annotate hot loop headers in HIR.** During SSA construction, mark
   blocks that are loop headers (have a back-edge predecessor). Add a
   stable `loop_id: u32` per loop. Emit a back-edge probe at each back-edge:

   ```rust
   // pseudo-HIR
   if let Some(osr) = beadie_osr_probe(self_bead_ptr, loop_id) {
       // tail-call through osr with current locals as the continuation frame
       transfer_to_native(osr, &locals);
   }
   ```

2. **Implement `on_invoke_osr` factory** that, given a Bead, builds:
   - The full Cranelift function (main entry compiled normally)
   - One `OsrEntry` per loop header — emit the function with a labelled
     entry block per loop, and use `module.declare_function` + a stub
     that branches into that block; the resolved address goes in `OsrEntry.code`

   Cranelift doesn't expose mid-function entry points directly, so the
   simplest implementation is per-loop helper functions:

   ```rust
   fn fn_main(args) -> result { ... }                // normal entry
   fn fn_osr_loop_3(locals_record) -> result { ... } // resumes at loop header
   ```

   The main function and OSR helpers share lowered IR but enter at
   different blocks. Cranelift can emit them as siblings in the same
   `JITModule`. Each gets a finalized pointer; pack them into
   `Vec<OsrEntry>` keyed by `loop_id`.

3. **Runtime back-edge probe** in the SSA builder:

   ```rust
   // at every loop back-edge, before jumping to the header:
   let osr = bead.osr_entry(loop_id);
   if !osr.is_null() {
       // marshal locals into the layout the OSR helper expects
       // (this is the hard part — must match the JIT's calling convention)
       transfer(osr, locals);
   }
   ```

   For ZynML's typed locals, the layout is just the function's local var
   slots in declaration order. The OSR helper takes a pointer to that
   layout and reads the locals back into Cranelift Vars at entry.

4. **Use `on_invoke_osr` in the dispatch:**

   ```rust
   self.adapter.on_invoke(bound, |tier, bead| {
       let module = self.module.read().unwrap();
       let function = module.as_ref()?.functions.get(&id)?;
       match tier {
           0 => self.cranelift.compile_baseline(bead, function),
           1 => {
               // tier 1 — also build OSR table
               let build = self.cranelift.compile_with_osr(bead, function)?;
               // we'd return ptr but TieredAdapter only takes a flat compile fn,
               // so we install OSR via bead.swap_compiled_with_osr after the
               // primary swap completes — see Phase 3
               build.entry
           }
           2 => self.llvm.compile_optimized(bead, function),
           _ => null_mut(),
       }
   });
   ```

**Limitation of `TieredAdapter`:** the multi-tier closure returns a
single `*mut ()`, not `OsrCompileResult`. To install OSR alongside a
tier promotion, we'd use `Bead::swap_compiled_with_osr` directly via a
custom broker, or use `BackendAdapter::on_invoke_osr` for the tiers
that need OSR (skipping `TieredAdapter` for those). See Phase 3.

### Phase 3 — Two-adapter setup for OSR-aware tiers

`TieredAdapter`'s tier-promotion broker uses `swap_compiled` (no OSR).
For tier-up that should ALSO refresh the OSR table, we need
`swap_compiled_with_osr`. Two options:

**Option A:** use `BackendAdapter::on_invoke_osr` only (single tier with
OSR). Lose multi-tier coordination. Simplest.

**Option B:** patch beadie upstream to accept an `OsrCompileResult`
in the tiered closure. Best long-term — file a PR.

**Option C:** do tier 0 via `TieredAdapter` (no OSR — baseline is fast
enough that mid-call OSR isn't needed), and run a parallel
`BackendAdapter::on_invoke_osr` for tier 1+. The bead is shared (both
adapters write to the same `Bead` underneath via `Arc<Bead>`), so OSR
installation from the OSR-aware adapter doesn't conflict with
`TieredAdapter`'s code-pointer swaps. This needs careful sequencing
but works without upstream changes.

**Recommended:** start with **Option A** for ZynML — one
`BackendAdapter` with OSR, `TieredPolicy` for the hotness curve.
Multi-tier (Cranelift→LLVM) is a Phase 4 concern; LLVM mostly matters
for AOT, not JIT.

### Phase 4 — Deopt for speculation

Once tiered Cranelift→LLVM is wired, hook bailouts:

```rust
// In compiled code: emitted as a guard check
extern "C" fn __zyn_bailout(bead_ptr: *const Bead, guard_id: u32, pc_offset: u32) {
    let info = BailoutInfo { guard_id: guard_id as u64, pc_offset, generation: ... };
    let decision = adapter.on_bailout(&bound, info);
    match decision {
        DeoptDecision::Recompile => bead.reload(),
        DeoptDecision::Blacklist => { /* never JIT this again */ }
        DeoptDecision::RevertToTier1 => { /* TieredAdapter caps max_tier */ }
        DeoptDecision::PauseRecompile => { /* delay re-promotion */ }
    }
}
```

This is pure value-add for the type-specialised fast paths in our SSA
builder (`println(double(5))`, etc.) — when a Dynamic argument's
runtime tag breaks our specialization, we deopt instead of crashing.

## Quantified Impact

| Metric | Today | With beadie |
|---|---|---|
| Hot-path lookup | `RwLock::read().get()` | `Bead.compiled()` — one acq-load |
| Promotion threshold | Fixed N | `queue_ahead(K)` — compile starts at N-K |
| Compile parallelism | 1 worker for all tiers | 1 worker per tier |
| Long single calls | Never promoted | Promoted via OSR back-edge probe |
| Speculation failure | Crash / corrupt result | `on_bailout` → revert/blacklist |
| Old code reclamation | Never (leak) | Epoch-based via crossbeam-epoch |
| Tiered DSL config | Bespoke `OptimizationTier` enum | `Vec<Box<dyn HotnessPolicy>>` |

## Risk

- **OSR locals layout must match the JIT's calling convention exactly.**
  This is the largest implementation cost. Get it wrong and you transfer
  garbage into compiled code — uncontrolled UB. Test with a simple loop
  (`for i in 0..1_000_000 { sum += i }`) and verify the OSR helper
  sees the right `i` and `sum`.
- **Bead `core` pointer must outlive the bead.** We currently use
  `HirId` as identity; treating the `HirFunction` arena as the core works
  if the arena outlives the runtime, which it does in our embedding.
- **The existing `TieredBackend` API is unused.** No external callers in
  `crates/zynml/`, so refactoring is contained to `compiler/` + `zyntax_embed/`.

## Files to touch

```
crates/compiler/Cargo.toml                  # add beadie, beadie-backend deps
crates/compiler/src/beadie_adapter.rs       # new — JitBackend impls
crates/compiler/src/tiered_backend.rs       # rewritten — wraps TieredAdapter
crates/compiler/src/lib.rs                  # re-export new types
crates/zyntax_embed/src/runtime.rs:3250+    # TieredRuntime keeps its API,
                                            # internals updated
```

The public `TieredRuntime` API in `runtime.rs` (its `development()`,
`production()`, `production_llvm()` constructors and `load_module*`
methods) stays unchanged. ZynML's `RuntimeProfile::Tiered*` enum keeps
working. All churn is internal.

## Phasing

1. **Phase 1** (mechanical): swap implementation, keep behavior. Verify
   `cargo test -p zyntax_compiler` and the existing ZynML examples still
   pass.
2. **Phase 2 + 3** (real benefit): OSR for tier-1 promotion of
   long-running functions. Pick Option A unless you want to upstream a
   change to beadie.
3. **Phase 4** (quality): deopt on speculation failure — eliminates
   silent miscompiles for type-specialised fast paths.

---

## Sibling-project survey: how Zura and wren_lift use beadie

Both live next to zyntax in `/Users/amaterasu/Vibranium/`.

### Zura (`zura/zura/crates/zura_jit/src/lib.rs`, 171 lines)

Zura's JIT does **not** use OSR. It registers one bead per function
([line 73-76](file:///Users/amaterasu/Vibranium/zura/zura/crates/zura_jit/src/lib.rs#L73))
and uses `Beadie::on_invoke` (the simplest single-tier path):

```rust
// zura_jit/src/lib.rs:104
self.beadie.on_invoke(bead, move |b| {
    compile_function_jit(&llvm, &bodies, &fn_map, &interner, fn_idx, b)
})
```

Compile path lowers MIR → LLVM IR via `zura_codegen::func::compile_function`,
then calls `JitBackend::compile(llvm, bead, def)`. Threshold is
`ThresholdPolicy::new(100)`. Header comment says **"Cranelift (mid-tier)
will be inserted between VM and LLVM later"** — but at the time of
reading, only one tier is wired and OSR isn't on the roadmap they show.

**Takeaway for us:** Zura is the wrong sibling to copy from for OSR. It's
the call-site-only baseline.

### wren_lift (`wren_lift/src/`) — **this is the OSR reference design**

wren_lift implements full OSR over beadie. The wiring spans three crates:

#### 1. MIR records OSR points during bytecode lowering

`src/mir/bytecode.rs:249`:

```rust
/// Metadata for a bytecode back-edge that may later transfer into native OSR.
pub struct OsrPoint {
    /// Bytecode offset of the Branch opcode that forms the back-edge.
    pub branch_offset: u32,
    /// Bytecode offset of the target loop-header block.
    pub target_offset: u32,
    /// MIR block id for the target loop header.
    pub target_block: BlockId,
    /// Registers passed to native OSR: external live-ins first, then target
    /// block params after branch binding.
    pub param_regs: Vec<u16>,
}
```

`BytecodeFunction.osr_points: Vec<OsrPoint>` — every back-edge that
*could* OSR is annotated at lowering time with the register layout
the JIT will expect to receive.

#### 2. Cranelift backend emits one helper function per OSR target

`src/codegen/cranelift_backend.rs:1309 fn compile_osr_entries`:

For each target block reachable via a back-edge (`collect_osr_targets`),
the backend:

1. Computes an `OsrEntryLayout` (`fn osr_entry_layout`,
   [line 1431](file:///Users/amaterasu/Vibranium/wren_lift/src/codegen/cranelift_backend.rs#L1431))
   — analyzes which values are live-in at the target block. **Skips
   loops whose live-in layout is unsupported** (more than 4 params,
   non-Value types, defs reachable only outside the OSR sub-graph,
   self-recursive call sites). This conservative analysis is the key
   correctness guard: only emit OSR for shapes the runtime can transfer.

2. Declares a separate Cranelift function per target:
   `{safe_name}_osr_bb{N}` with signature
   `(i64, i64, i64, i64) -> i64` (up to 4 params, packed `Value` bits).

3. Lowers MIR with `Some(layout)` passed in — the lowering uses this to
   teach the entry block to bind the function params to the right MIR
   `ValueId`s instead of running the function from MIR block 0.

4. Finalizes both the main entry and all OSR helpers in one
   `module.finalize_definitions()` call (
   [line 1186](file:///Users/amaterasu/Vibranium/wren_lift/src/codegen/cranelift_backend.rs#L1186))
   so they share the same JITModule.

5. Returns
   ```rust
   pub struct NativeOsrEntry {
       target_block: BlockId,
       param_count: u16,
       ptr: *const u8,
   }
   ```
   one per surviving target.

#### 3. Engine encodes the site key and installs alongside the main code

`src/runtime/tier.rs:60`:

```rust
#[inline]
pub fn encode_osr_site(block_id: u32, param_count: u16) -> u64 {
    ((block_id as u64) << 16) | (param_count as u64)
}
```

`src/runtime/engine.rs:324`:

```rust
fn encode_osr_entries(entries: &[NativeOsrEntry]) -> Vec<beadie::OsrEntry> {
    entries.iter()
        .filter(|e| !e.ptr.is_null())
        .map(|e| beadie::OsrEntry {
            site: super::tier::encode_osr_site(e.target_block.0, e.param_count),
            code: e.ptr as *mut (),
        })
        .collect()
}
```

Install path: `tier.install_or_swap_osr(id, code, entries)` (their
[`tier.rs:252`](file:///Users/amaterasu/Vibranium/wren_lift/src/runtime/tier.rs#L252))
does:
- `BeadState::Interpreted` → `eager_install(code)` then
  `swap_compiled_with_osr(code, osr)`. Two-step because beadie's first-
  install API doesn't take an OSR vec.
- `BeadState::Compiled` → `swap_compiled_with_osr` directly. Used for
  tier-up; both pointer and OSR table replaced atomically under a new
  generation.

#### 4. Back-edge probe in the bytecode interpreter

`src/runtime/vm_interp.rs:3680` (the `Op::Branch` handler when target
< branch_offset, i.e., backward jump):

```rust
backedge_counter = backedge_counter.wrapping_add(1);
let should_tier_up = vm.engine.record_call(func_id);
if should_tier_up { vm.engine.request_tier_up(func_id, &vm.interner); }

if should_tier_up
    || vm.engine.has_pending_compilations()
    || (backedge_counter & 63) == 0  // sample 1/64 back-edges
{
    if vm.engine.has_pending_compilations() {
        // Save frame state before draining (compile may finish + install).
        unsafe { (*fiber).mir_frames.last_mut().map(|f| {
            f.values = std::mem::take(&mut values);
            f.pc = pc;
        }); }
        vm.engine.poll_compilations();
        vm.engine.drain_compile_queue(&vm.interner);
        unsafe { (*fiber).mir_frames.last_mut().map(|f| {
            values = std::mem::take(&mut f.values);
        }); }
    }
    match try_enter_loop_osr(vm, fiber, func_id, ..., bc, branch_offset, target,
                              &mut values, stop_depth)? {
        OsrTransfer::NotEntered => {}
        OsrTransfer::ContinueFiberLoop => continue 'fiber_loop,
        OsrTransfer::Return(value) => return Ok(value),
    }
}
```

Two cost-control choices to copy:
- **1-in-64 sampling.** Polling for an OSR table on every back-edge is
  expensive. The hot path increments a counter; the OSR check runs every
  64 iterations OR when the tier-up policy fires OR when the broker has
  finished a compile.
- **Frame snapshot around `drain_compile_queue`.** Compile completion
  happens on a worker thread; the install runs on the dispatch thread
  but mutates `engine.jit_code`. Save `values` and `pc` into the frame
  so an installer can read them, then steal them back.

#### 5. The transfer itself (`fn try_enter_loop_osr` line 401)

```rust
fn try_enter_loop_osr(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    func_id: FuncId,
    ...
    bc: &BytecodeFunction,
    branch_offset: u32,
    target_offset: u32,
    values: &mut Vec<Value>,
    ...
) -> Result<OsrTransfer, RuntimeError> {
    // 1. Find the OsrPoint matching (branch_offset, target_offset)
    let Some(point) = bc.osr_points.iter().find(|p|
        p.branch_offset == branch_offset && p.target_offset == target_offset
    ) else { return Ok(OsrTransfer::NotEntered); };
    if point.param_regs.len() > 4 { return Ok(OsrTransfer::NotEntered); }

    // 2. Marshal live-in registers into args (in declaration order).
    let mut osr_args = SmallVec::<[Value; 4]>::new();
    for &reg in &point.param_regs {
        let value = values.get(reg as usize).copied()
            .filter(|v| !v.is_undefined())
            .ok_or_else(|| return Ok(OsrTransfer::NotEntered))?;
        osr_args.push(value);
    }

    // 3. Look up the entry pointer for (target_block, param_count).
    let entry = vm.engine.active_osr_entry(func_id, point.target_block, osr_args.len())
        .ok_or_else(|| return Ok(OsrTransfer::NotEntered))?;

    // 4. Save the current frame so a deopt back to the interpreter resumes
    //    at the OSR target, not at the back-edge.
    unsafe { if let Some(frame) = (*fiber).mir_frames.last_mut() {
        frame.pc = target_offset;
        frame.values = std::mem::take(values);
        frame.bc_ptr = bc as *const BytecodeFunction;
    } }

    // 5. Snapshot + install JIT context (closure, defining class, fiber roots).
    //    See lines 474-523. This is the part that's most engine-specific.

    // 6. Call into native code. The compiled function reads its params,
    //    runs the loop, returns a Value (as i64 bits).
    let result_bits = unsafe { call_jit_fn(entry.ptr, &osr_args) };

    // 7. Restore JIT context. Handle pending fiber actions (yield/transfer
    //    inside the OSR'd loop is currently rejected — see line 580).
    // 8. Decode the return value, pop the frame, signal the interpreter loop.
}
```

Key design decisions:
- **Param count cap of 4** matches the Cranelift signature
  `(i64, i64, i64, i64) -> i64` chosen by the backend. Both sides
  agree on this constant; loops that need more live-ins just don't
  OSR.
- **Live-ins are passed as ABI args, not via a pointer to the
  interpreter's value array.** The compiled code never reads the
  interpreter's values vec — once it's running it owns its own SSA
  state. This isolates the JIT from interpreter layout changes.
- **`encode_osr_site(block, param_count)` is the bead lookup key.** The
  param count goes in the lower 16 bits so different leaf/tail variants
  of the same block (different live-in counts) get different slots.
- **`call_jit_fn` is unsafe** but the calling convention is
  rigorously specified in `runtime_fns`.

### Concrete prescription for zyntax

Mirror wren_lift's structure layer-by-layer; skip Zura's pattern.

**HIR layer** (analogue of wren_lift's MIR `osr_points`):

```rust
// crates/compiler/src/hir.rs
pub struct OsrPoint {
    /// HIR block id of the back-edge source.
    pub branch_block: HirId,
    /// HIR block id of the loop-header target.
    pub target_block: HirId,
    /// Live-in HirIds at the target — order matches the JIT signature.
    pub live_ins: Vec<HirId>,
}
pub struct HirFunction {
    // ...
    pub osr_points: Vec<OsrPoint>,
}
```

Populate during SSA construction's loop analysis: the dominator-tree
pass already exists (added earlier this session — Cooper/Harvey/Kennedy
in `advanced_analysis.rs`). A back-edge is a CFG edge `b → h` where `h
dominates b`. For each such pair, record the live-in set at `h`.

**Cranelift backend** (analogue of wren_lift's `compile_osr_entries`):

```rust
// crates/compiler/src/cranelift_backend.rs  (new module: osr.rs)
fn compile_osr_helpers(
    func: &HirFunction,
    module: &mut JITModule,
    main_name: &str,
) -> Vec<NativeOsrEntry> {
    let mut entries = Vec::new();
    for point in &func.osr_points {
        let layout = match osr_entry_layout(func, point) {
            Some(l) => l,
            None => continue,         // skip unsupportable shapes
        };
        if layout.param_count > 4 { continue; }   // ABI cap
        let helper_name = format!("{}_osr_bb{}", main_name, point.target_block.0);
        let func_id = declare_helper(module, &helper_name, layout.param_count);
        lower_function_body_with_osr_entry(func, module, func_id, &layout)?;
        entries.push(NativeOsrEntry {
            target_block: point.target_block,
            param_count: layout.param_count as u16,
            ptr: /* filled after finalize */,
        });
    }
    // Finalize once, then resolve ptrs.
    entries
}
```

Note: zyntax's existing Cranelift backend already declares helper
functions for the SIMD `compute @kernel` lowering — it has the
plumbing to emit multiple sibling functions in one `JITModule`. OSR
helpers reuse that mechanism.

**Backend integration with beadie:**

```rust
// crates/compiler/src/beadie_adapter.rs (or extend tiered_backend.rs)
fn install_baseline(&self, id: HirId, ptr: *mut (), osr: Vec<OsrEntry>) {
    let bound = &self.beads[&id];
    // Replicate wren_lift's install_or_swap_osr (tier.rs:252)
    match bound.bead().state() {
        BeadState::Compiled => bound.bead().swap_compiled_with_osr(ptr, osr),
        _ => {
            bound.bead().eager_install(ptr);
            bound.bead().swap_compiled_with_osr(ptr, osr)
        }
    };
}

pub fn osr_entry(&self, id: HirId, target_block: HirId, param_count: u16) -> Option<*mut ()> {
    let site = encode_osr_site(target_block, param_count);
    self.beads.get(&id)?.bead().osr_entry(site)
}

fn encode_osr_site(target_block: HirId, param_count: u16) -> u64 {
    // HirId is a UUID — hash to u32, pack with param_count.
    let block_hash = stable_hash_u32(target_block);
    ((block_hash as u64) << 16) | (param_count as u64)
}
```

**Back-edge probe in the SSA builder** (analogue of vm_interp.rs:3680):

zyntax doesn't have a bytecode interpreter — programs run JIT-compiled
from the start. So the OSR usage shape is different: instead of
"interpreter polls JIT", it's "Cranelift-compiled tier-0 baseline
polls for an LLVM-tier-1 entry". The probe lives in the baseline
emission, at every back-edge:

```rust
// In ssa.rs / cranelift_backend.rs back-edge emission:
//   if (osr_check_counter & 63) == 0 {
//       let osr_ptr = beadie_osr_entry(self_func_id, target_block, n_live_ins);
//       if !osr_ptr.is_null() {
//           // Marshal live-ins into args, tail-call osr_ptr, return its result.
//       }
//   }
//   counter += 1;
//   br loop_header
```

The `beadie_osr_entry` function pointer is registered as a runtime
symbol in `CraneliftConfig`, same way as `dispatch_trampoline` in the
beadie-cranelift `fib.rs` example. The compiled baseline calls into
zyntax's runtime, which calls `bead.osr_entry(site)`.

**ABI cap of 4 args** (matching wren_lift) is fine for ZynML — most
loops have a couple of induction variables and an accumulator. Loops
needing more live-ins simply don't OSR; they tier up at the next call
site.

### Differences from wren_lift to keep in mind

| Concern | wren_lift | zyntax (ours) |
|---|---|---|
| Tier 0 | Bytecode interpreter | Cranelift baseline JIT |
| Tier 1 | Cranelift JIT | LLVM JIT (already wired) |
| Frame layout | `mir_frames: Vec<Frame>` with `values: Vec<Value>` | Cranelift function frames; locals already in machine regs |
| OSR direction | Interpreted → Native | Tier-0 native → Tier-1 native |
| Live-in marshalling | Read from `values: Vec<Value>` indexed by `param_regs` | Pass through Cranelift's calling convention; baseline emits the call to the helper directly |

The critical simplification for zyntax: **we never have to materialize
locals from a interpreter value-vec into machine registers**. Both
sides are JIT-compiled; live-ins are already in registers/spill slots
at the back-edge. The OSR helper just receives them as ordinary
function arguments. No "reconstruct interpreter state into JIT shape"
glue — that whole class of bug (the `param_regs` filtering and bounds
checks in wren_lift's `try_enter_loop_osr`) doesn't exist for us.
