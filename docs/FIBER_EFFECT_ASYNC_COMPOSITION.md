# Fiber × Effect × Async composition — full implementation plan

Goal: take Zyntax from "the three concurrency primitives exist per-rail but don't
compose" to "users can freely mix `fiber def`, algebraic effects, and `async def`
with coherent runtime semantics."

Derived from a 20-program design audit. Each phase leaves the tree green, is
independently shippable, and unblocks a named set of user programs. Phases 0–3
are the "coherent composition" gate; 4–7 are "compose well"; 8 is the unification.

## Substrate facts (the constraints every phase works within)

- `async def` + resumable-effect handlers → krio_adapter AOT state-machine
  (`lower_async_function`, poll-fn ABI `fn(*mut u8) -> i64`, uniform 8-byte slots).
- `fiber def` → krio-fiber stackful stack-switching (`krio_fiber_*` symbols,
  mmap'd 64 KiB stack, per-arch context switch). Disjoint substrate.
- Three thread-locals: `ACTIVE_FIBER`/`ACTIVE_TRAMPOLINE` (save/restored, composes),
  `ABORT_PAYLOAD`/`ERROR_MAP` (per-thread, fiber-unaware), `HANDLER_STACK`
  (per-thread, NOT save/restored across fiber switch, currently dead in compiled
  output because `@with` never lowers).
- `Type::Fiber(T)` carries yield type only; error type is hardwired
  `FiberError<String>`. Lifting to per-fiber E is out of scope for this plan.
- Cooperative single-threaded is the committed scheduling model. Fibers are
  `!Send`; resume must stay on creating thread.

## Cross-cutting decisions (commit before phase work)

1. Cooperative, single-threaded scheduling. All existing thread-locals assume it.
2. One unified Suspension envelope is the end state (Phase 8). Until then, each
   suspension kind (fiber yield, await, effect Pending) gets a bridge, and we
   accept the tag-space pressure on `FiberStep` (2 bits) as a known debt.
3. Handler stack becomes per-Task (Phase 8); Phase 4 is the snapshot/restore stopgap.
4. Drop is compiler-emitted (`FiberDrop` at every scope exit incl. exceptional),
   not GC-swept.
5. `Type::Fiber` stays `(T)`; `FiberError` stays `<String>`. Separate follow-up.

## Shared infrastructure built in Phase 0, reused everywhere

`ScopeExitEmitter` — a single SSA helper that, given a scope and a list of cleanup
actions, emits those actions on EVERY exit edge (fallthrough, return, break,
continue, abort, exceptional unwind). Both `with`-block `pop_handler` (Phase 1) and
`FiberDrop` (Phase 2) route through it so exit-edge coverage can't drift between
the two. This is the single most important piece of internal design — get it right
once.

---

## Phase 0 — Stop crashing; guardrails; shared scaffolding

**Size:** small. **Unblocks:** the 7+ streaming/http/game/ecs programs stop
panicking on the default build. Turns silent miscompiles into diagnostics.

### 0.1 Make krio the default async backend
- `crates/zyntax_embed/Cargo.toml`: fold `krio-async-backend` into `native`
  (or make it default-on). Today it's off by default (line 85), so plain
  `async def` runs the legacy `AsyncCompiler` which panics on fiber-handle-in-loop
  captures at `cranelift_backend.rs:1862`.
- `crates/zynml/Cargo.toml`: propagate the feature.
- Verify: `cargo test --workspace` count unchanged; the composition-C shape
  (`async def` iterating a fiber in a while-let) compiles + runs.

### 0.2 Root-cause the `CreateClosure: Lambda function HirId(...) not found` warning
- Surfaces on the await-in-loop-with-fiber variant even on the krio path. Likely a
  Lambda-hoist vs captures-lift slot-allocation ordering bug in
  `crates/passes/krio_adapter/src/abi_emit.rs`.
- Verify: warning gone; a test that awaits inside a while-let that also drives a
  fiber produces correct output with no warning.

### 0.3 Reject incoherent compositions with clean diagnostics
- `await` inside a `fiber def` body: add an `in_fiber` context flag in
  `crates/typed_ast/src/type_checker.rs`; reject `TypedExpression::Await` under it.
  Diagnostic: "await is not allowed in a fiber body; drive the fiber from an async
  function instead."
- `is_fiber && !effects.is_empty()` where any handler is resumable: reject at
  parse-to-typed in `crates/compiler/src/lowering.rs` validation. (Non-resumable
  effects in a fiber are fine — they're direct calls. Only resumable ones collide
  with the SM transform.) Diagnostic points at the unguarded composition.
  NOTE: this rejection is LIFTED in Phase 5 once cooperative-fiber lands. It's a
  guardrail, not a permanent ban.
- Harden the silent intrinsic skip: `cranelift_backend.rs:2415-2416`'s
  `_ => { continue; }` → `Err(CompilerError::Backend(...))`. A silent skip + a
  downstream value_map panic is worse than a clean error.
- Verify: three new `#[should_panic]`/`Err`-expecting tests.

### 0.4 Build `ScopeExitEmitter`
- New helper in `crates/compiler/src/ssa.rs` (or a small sibling module). Given a
  block and a `Vec<CleanupAction>`, walk the CFG's exit edges for that scope and
  emit each action before each terminator. Actions are an enum; Phase 1 adds
  `PopHandler(frame_id)`, Phase 2 adds `DropFiber(hir_id)`.
- Verify: unit test with a synthetic scope having return + break + fallthrough
  edges; assert the action appears on all three.

---

## Phase 1 — `with` block syntax + handler scoping

**Size:** medium. **Unblocks:** ~15 programs that use `with H { ... }`.
**Depends on:** 0.4 (ScopeExitEmitter). **Prerequisite for:** 3, 4, 7.

### 1.1 Grammar
- `crates/zynml/ml.zyn`:
  ```
  with_stmt          = "with" ~ handlers:with_handler_list ~ body:block
  with_handler_list  = with_handler ~ ("," ~ with_handler)*
  with_handler       = name:ident ~ args:call_args?
  ```
- Add `with_stmt` to the statement alternatives. `with` becomes a reserved word —
  run a naming-collision pass over stdlib + examples first.
- Push order = declaration order; pop in reverse on exit.

### 1.2 Parser construction
- `crates/zyn_peg/src/runtime2/interpreter.rs`: `construct_with_stmt` building a
  new `TypedStatement::With { handlers: Vec<TypedWithHandler>, body: TypedBlock }`.
  Each handler carries name + optional args (args needed for parameterised
  handlers like `BoundedCredit(capacity=4096)` — though arg-bearing handler
  construction fully lands with Phase 3 handler-state).

### 1.3 Typed AST
- `crates/typed_ast/src/typed_ast.rs`: `TypedStatement::With` variant +
  `TypedWithHandler { name, args }`. Update the ~exhaustive match sites
  (visitor, type checker, printer).

### 1.4 Handler-push lowering
- `crates/compiler/src/ssa.rs`: `TypedStatement::With` handler. For each handler in
  order, emit `Call::Symbol("__zyntax_effect_push_handler", [effect_id,
  handler_state_ptr, op_table])` at block entry. Register a `PopHandler(frame_id)`
  cleanup action with the ScopeExitEmitter for the block. Translate body.
- `handler_state_ptr` is null/0 until Phase 3 adds handler state.
- `op_table` maps op-name → handler-fn-id; built from the module's handler decls.

### 1.5 Activate runtime handler resolution
- `crates/zyntax_embed/src/effect_runtime.rs`: `push_handler`/`pop_handler` already
  exist as unit-tested primitives but nothing calls them from compiled code. Wire
  the runtime symbols so the SSA calls resolve.
- Change `PerformEffect` lowering in `cranelift_backend.rs:4309` from static
  `module.handlers.first-match-by-effect_id` to a runtime
  `__zyntax_effect_lookup_handler(effect_id)` that consults `HANDLER_STACK`. This
  is what makes `with H` semantically meaningful (regional handler selection).
- Keep the static path as the fallback when `HANDLER_STACK` is empty (module-scope
  single-handler case — the common case still works without a `with`).

### 1.6 Remove `@with` annotation
- It's parsed-but-never-lowered today. Delete the annotation path in
  `crates/passes/algebraic_effects/src/annotations.rs`. Any code using `@with(H)`
  now gets a clean "unknown annotation; use a `with H { }` block" diagnostic.

### Verify Phase 1
- `with StderrLog { perform Log.warn("x") }` routes to StderrLog and prints.
- Two handlers for the same effect; `with H2 { }` in a nested scope picks H2 while
  the outer `with H1 { }` picks H1 — regional dispatch works.
- Multi-handler `with A, B, C { }` pushes 3, pops 3 in reverse on every exit
  (return / break / fallthrough tested).
- Handler frame count returns to baseline after the block (no leak).

---

## Phase 2 — Deterministic fiber lifecycle (drop) — ✅ SHIPPED (8fdf32f)

**Size:** medium. **Unblocks:** 6+ programs (socket-holding fiber, async-owns-fiber
early return, all cancellation stories). **Depends on:** 0.4. **Prereq for:** 6.

**Shipped scope:** 2.1/2.2/2.4 as specified. 2.3 landed as a conservative
`emit_fiber_drops` pass (lowering.rs) rather than a ScopeExitEmitter integration:
drops only entry-block `FiberNew` results (they dominate all returns, so the
handle is always initialised) and skips any fiber that escapes via a `Return`
value or a phi. Guarantees no double-free; leaves inner/conditional and
call-received fibers leaking until a later liveness-based pass. `break`/abort/
exceptional edges and the move-tracking ownership rule are the follow-up.

### 2.1 FiberDrop HIR op
- `crates/compiler/src/hir.rs`: `HirInstruction::FiberDrop { fiber: HirId }`.
  Update the two match sites (`hir.rs:1132`-style display + validation).

### 2.2 Runtime free
- `crates/compiler/src/zrtl.rs`: `krio_fiber_free(fiber: *mut FiberRepr)` runtime
  symbol. Registered in `fiber_runtime_symbols()`.
- `crates/compiler/src/fiber_backend.rs`: `FiberCfg::fiber_free(&self, fiber)`.
- `crates/passes/krio_adapter/src/fiber.rs`: `KrioFiberBackend::fiber_free` does
  `Box::from_raw(fiber as *mut Fiber)` (drops the box → unmaps the 64 KiB stack +
  guard page) AND clears `ERROR_MAP[fiber_addr]` so an address-reused fiber can't
  inherit a stale error.

### 2.3 Drop-site emission
- `crates/compiler/src/ssa.rs`: when a `Type::Fiber(_)` binding goes out of scope,
  register a `DropFiber(hir_id)` cleanup with the ScopeExitEmitter. Covers normal
  fallthrough, `return`, `break`, early return inside async, and abort/exceptional
  edges.
- Ownership rule: a fiber is dropped by whoever holds the binding. Moving a fiber
  (returning it, storing it) transfers the drop obligation — needs the existing
  move/ownership tracking (borrow_check.rs seed) or a conservative "drop at end of
  defining scope unless moved" analysis.

### 2.4 Lowering pass
- `crates/compiler/src/fiber_lowering.rs`: `apply_krio_fiber_lowering` learns to
  rewrite `FiberDrop` → `Call::Symbol("krio_fiber_free", [fiber])`.

### Verify Phase 2
- A fiber created and fully consumed in a scope: `krio_fiber_free` called once at
  scope exit; no leak (valgrind/leak-count in a test harness, or an alloc counter).
- Async body that early-returns while holding a fiber: fiber freed on the return
  edge.
- `ERROR_MAP` entry gone after free.
- Existing `fiber_execution` tests still green (drop must not double-free a fiber
  the caller already exhausted).

---

## Phase 3 — Handler state across performs — ✅ SHIPPED (8c63488)

**Size:** medium. **Unblocks:** 4+ programs (db-pool handler, credit handler,
handler-scoped-fiber). **Depends on:** 1 (with-block + push_handler).

**Shipped scope:** 3.1/3.2/3.3 for the non-resumable path. State is a
synthesized `@reference` struct `H$state`; `synthesize_handler_state`
(runtime.rs, pre-registry-snapshot) registers it, builds an `H$new()`
constructor that allocates+initialises it, and prepends `self: H$state` to
every non-resumable op. `lower_with_scopes` calls `H$new()` at scope entry,
passes the pointer to `push_handler`, and frees it on each exit edge;
`__zyntax_effect_lookup_state` feeds it to the perform site as the implicit
first arg. A stateful handler must be entered via `with` (static path passes
null self). Follow-ups: resumable-handler state (krio path), typed drop of
state fields (e.g. an `Option<Fiber<T>>` field — currently raw `free`, no
field destructors), and a regional override matching the default's
statefulness.

### 3.1 Handler-state syntax
- `crates/zynml/ml.zyn`: allow `var name: T = init` / `let name: T = init`
  declarations inside `handler H for E { ... }` body, before the op `def`s.
- Parser: attach state fields to the handler declaration.

### 3.2 State region allocation
- `crates/zyntax_embed/src/effect_runtime.rs`: `push_handler` allocates a
  handler-state region sized to the handler's state fields, runs the initialisers,
  stores the pointer in the `HANDLER_STACK` frame. `pop_handler` runs drop on the
  state region.
- Handler op bodies get an implicit `self` pointing at the state region;
  `self.field` reads/writes go through it.

### 3.3 self in op bodies
- `crates/compiler/src/ssa.rs`: handler op lowering threads the state pointer as an
  implicit first param; `self.field` becomes a GEP+load/store on it.

### Verify Phase 3
- db-pool handler: `conn` checked out on first perform, reused on subsequent
  performs within the same `with` scope; released on `pop_handler`.
- Credit handler: counter persists across `request`/`release` performs.
- A handler-scoped `Option<Fiber<T>>` persists across performs (this is the
  "fiber recreated per perform" fix from the composition-B audit).

---

## Phase 4 — Handler stack survives fiber stack switch — ✅ SHIPPED (feaa281)

**Size:** small. **Unblocks:** 3+ programs (perform inside fiber body sees outer
handler; sensor pipeline; parser-combinator). **Depends on:** 1.

**Shipped scope:** The primary goal (a perform inside a fiber resolves against
the handlers active at resume) already held via the same-thread thread-local
stack. But the plan's literal "snapshot len, truncate after resume" is
UNSOUND — it strands a fiber's own open handlers before it pops them on the
next resume, regressing multi-yield fibers. Shipped the correct version: a
per-fiber handler-stack SEGMENT. `FiberResume`/`FiberResumeWith` lower to a
triple bracketed by `__zyntax_effect_fiber_enter` (re-push the fiber's saved
frames, return the caller's baseline depth) and `__zyntax_effect_fiber_leave`
(lift frames above baseline into the fiber's segment, truncate to baseline);
`FiberDrop` emits `__zyntax_effect_fiber_forget`. Each fiber thus owns a
segment layered on the caller's baseline: no leak into the caller, no
stranding across yields, and interleaved fibers keep their own handlers. The
non-resumable-in-fiber case was never rejected (the guardrail is
resumable-only), so nothing to relax; resumable-in-fiber stays deferred.

### 4.1 Snapshot/restore around resume
- `crates/passes/krio_adapter/src/fiber.rs`: in `KrioFiberBackend::fiber_resume`
  and `fiber_resume_with`, snapshot `HANDLER_STACK.len()` before the context switch
  and truncate back to it after. Mirrors the `prev_active`/`prev_fiber` save/restore
  at `krio-fiber/src/fiber.rs:477-497`.
- This makes a `perform` inside a fiber body resolve against the handler stack that
  was active when the fiber was resumed — the lexically-expected behaviour.
- This ALSO relaxes the Phase 0.3 rejection for the non-resumable-in-fiber case:
  a fiber performing a non-resumable effect is now well-defined. (Resumable-effect-
  in-fiber still waits for Phase 5/7.)

### Verify Phase 4
- `with H { let f = fiber_that_performs_H(); f.next() }` — the perform inside the
  fiber body resolves to H.
- Nested: `with H1 { let f = g(); with H2 { f.next() } }` — a perform during that
  `f.next()` sees H2 on top.
- Handler stack length identical before and after a full fiber drain.

---

## Phase 5 — Cooperative fiber-in-async

**Size:** medium. **Unblocks:** 5+ programs (backpressure, http server, ecs, game
loop). **Depends on:** 0.1 (krio default), 2 (drop). **Lifts:** part of 0.3.

### 5.1 @cooperative fiber def
- `@cooperative` annotation on a `fiber def` makes the async transform treat
  `krio_fiber_resume` as a `DirectYield` suspension site.
- `crates/passes/krio_adapter/src/lib.rs`: `HirAsyncHooks::classify` (711-730)
  recognises `Call::Symbol("krio_fiber_resume")` as `DirectYield` when the callee
  fiber is cooperative.
- Effect: `f.next()` inside an async body releases the executor between fiber steps
  instead of blocking it for the whole fiber run.

### 5.2 Fiber-step-ready bridge
- `krio_fiber_resume` on a cooperative fiber that yields registers the fiber with
  `__zyntax_register_future(fiber_resume_trampoline, fiber_ptr, ...)` keyed on
  "next fiber step ready", returns Pending to the async SM. The SM parks; the
  executor runs other tasks; the trampoline re-drives the fiber on next poll.
- `crates/zyntax_embed/src/host_futures.rs`: wire the trampoline.

### Verify Phase 5
- Two async tasks each driving a `@cooperative` fiber interleave (task A's fiber
  step, then task B's, then A's) rather than A running to completion first.
- Backpressure program: producer fiber suspends inside `request(n)` while the sink
  drains; observed as parked, not spun.

---

## Phase 6 — Cancellation propagation across rails

**Size:** large. **Unblocks:** 4+ programs (ml-inference cancel, http server
shutdown, socket cleanup). **Depends on:** 2 (FiberDrop), 1 (pop_handler).

### 6.1 Cancel token from async to fiber
- `Promise::cancel()` marks the async task cancelled. On next poll the SM runs its
  cleanup path: `FiberDrop` on every owned fiber, `pop_handler` on every open
  handler frame, in reverse acquisition order.
- Drop ordering invariant: a fiber that owns an effect `Resume<T>` must drop before
  the handler frame that would resume it, so a cancelled resume doesn't fire into a
  freed continuation.

### 6.2 In-flight await cleanup
- A cancelled task with an outstanding `FUTURE_TABLE` entry must deregister it so
  the resolver doesn't drive a dead SM.

### Verify Phase 6
- Long-running async driving a fiber; `cancel()` mid-iteration frees the fiber's
  mmap stack and closes its open fds (observed via fd count / leak counter).
- Handler frames opened by the cancelled task are popped; `HANDLER_STACK` returns
  to baseline.
- Cancel during an outstanding await removes the FUTURE_TABLE entry.

---

## Phase 7 — Pending-returning (async/parking) handlers

**Size:** large. **Unblocks:** 3 programs (credit-parks-continuation,
handler-that-awaits, effect_handler_returns_pending). **Depends on:** 5.

### 7.1 Replace spin-poll with Pending propagation
- `__zyntax_effect_resume` (`effect_runtime.rs:252-372`) currently spin-polls the
  caller's SM under `LOOP_BUDGET=100_000`. A handler that returns Pending (parked
  continuation, or one that awaits) must instead propagate Pending up to the
  executor, not burn the budget.
- Requires the resume path to be a real suspension point, coordinated with the
  async SM the caller lives in.

### 7.2 FiberStep envelope widening (if fiber-await lands here)
- If a cooperative fiber body needs to await: widen `FiberStep` beyond 2 tag bits
  (or repurpose payload encoding) for a `PendingOnFuture { handle }` variant.
  `crates/passes/krio_adapter/src/fiber.rs:139-144` is the binding constraint.
- `krio_fiber_await(promise)` runtime symbol; `encode_step` surfaces the new tag;
  `emit_fiber_next` decoder + executor handle it.
- (This is the only place the "await in fiber" story could ever be revisited — and
  only under `@cooperative`. The Phase 0.3 blanket rejection of await-in-fiber
  stays for non-cooperative fibers.)

### Verify Phase 7
- Credit handler parks the producer's continuation; the sink's `release` wakes it;
  no busy-spin (CPU flat while parked).
- A resumable handler that `await`s an I/O op returns Pending; executor runs other
  tasks; handler resumes when the I/O completes.

---

## Phase 8 — Unified scheduler (redesign)

**Size:** redesign. **Unblocks:** multi-fiber multiplexing (spawn_two_fibers,
game_tick, ecs_tick) + retires the per-rail thread-local debt.
**Depends on:** everything above.

### 8.1 Route all three rails through krio-runtime
- `Fiber<T>`, `Promise<T>`, and effect resumable SMs become variants of the
  krio-runtime `Task`, driven by one `Scheduler`, suspending via one `Suspension`
  enum (`/Users/amaterasu/Vibranium/krio/crates/krio-runtime/src/lib.rs:31-53`
  already sketches this).

### 8.2 Per-Task storage
- `HANDLER_STACK`, `FUTURE_TABLE`, `ABORT_PAYLOAD`, `ERROR_MAP` move from
  thread-local to per-Task, indexed by the scheduler. Retires the Phase 4
  snapshot/restore stopgap and the address-keyed ERROR_MAP.

### 8.3 `spawn` surface
- A `spawn(task)` primitive returning a handle; the scheduler multiplexes.
- Defines `select`/`race` across fibers and tasks (the critique flagged these as
  missing patterns).

### Verify Phase 8
- N fibers spawned, driven fairly by the scheduler; no starvation.
- `select` across two fibers returns whichever yields first.
- Thread-local pressure gone (grep: no `thread_local!` in the hot rails).

---

## Dependency graph

```
0 ──┬─> 1 ──┬─> 3
    │       ├─> 4
    │       └─> 7 ──┐
    ├─> 2 ──> 6     │
    └─> 5 ──────────┴─> 8
```

## Coherence gate

After **Phases 0–4**, ~15 of the 20 audited programs compile and run correctly with
sound semantics. That's the point to call the composition story "coherent" and
document it publicly. Phases 5–7 turn "runs correctly but blocks the executor" into
"runs cooperatively." Phase 8 turns "one task at a time" into "many."

## Invariants to enforce throughout (and record in memory)

- Single-thread fiber affinity: resume must stay on the creating thread. Add a
  `debug_assert` on `thread::current().id()` vs a TLS-stamped id at `fiber_resume`.
  `Fiber: !Send` is Rust-only; the C ABI accepts any caller.
- Drop ordering under panic/cancel: a fiber owning a `Resume<T>` drops before the
  handler frame that would resume it. Enforced by ScopeExitEmitter acquisition-order
  tracking.
- ScopeExitEmitter is the single source of exit-edge truth for BOTH pop_handler and
  FiberDrop — they must never diverge on which edges they cover.

## Test strategy per phase

Every phase adds e2e tests under `crates/zynml/tests/` driving the actual JIT path
(the `zynml run` path), mirroring the existing `fiber_execution.rs` shape. Unit
tests for the SSA-level pieces (ScopeExitEmitter, drop-site analysis) live in the
compiler crate. Each phase's "Verify" section is the acceptance checklist.
