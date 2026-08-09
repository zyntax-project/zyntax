# Hot Reload

Hot reload replaces a function's code while the program runs, without
losing runtime state. Live fibers keep their state across a reload and
pick up edited code at their next yield boundary; effect handlers rebind
in place, so a `with H` scope entered before the edit dispatches to the
edited implementation after it. A downstream project drives FSMs with
persistent fibers and uses effects as the observable boundary of a UI
DSL; both must survive an edit-reload cycle with their state intact.

## The reload contract

**Code is replaced at well-defined boundaries; state is never copied.**
The boundaries, from coarse to fine:

1. **Call boundary** — the next call to an edited function runs the new
   code. This is the default and always available.
2. **Yield boundary** — a suspended fiber resumes into the edited step
   code against its existing state, when the state layout is compatible.
3. **Loop boundary** — a running loop transfers into the edited body via
   OSR, when a resume point for its header exists in the new code.

State lives where it always lived — fiber state structures, handler
state (`H$state`), globals, the heap — and reload never moves it. What
changes is which code pointers the existing indirection cells hold. Old
code pages are retained for the life of the process (generations already
guarantee this), so a frame or fiber stack still executing pre-edit code
runs to its next boundary safely.

**Layout is the compatibility contract.** A reload that changes the
shape of live state — a fiber's state struct, a handler's state struct,
a type used by a suspended frame — cannot migrate that state. Each
reloadable unit carries a layout hash; on mismatch the unit falls back
one level: a fiber completes on its old code (or restarts, by policy), a
handler keeps its old implementation until its scope exits. The reload
report says which units migrated and which fell back, and why.

## What exists today

- **Per-function pointer table with versioning and rollback**
  (`cranelift_backend::hot_reload_function` / `rollback_function`).
  Recompiles one function and updates the shared pointer table;
  generation-suffixed symbols keep every version's code mapped.
- **Bead dispatch** — tier promotion already swaps a function's entry
  pointer atomically under running code (`swap_compiled`), and callers
  that enter through the bead observe the swap.
- **OSR** — a running loop transfers into higher-tier code with its live
  state carried in a frame, on both Cranelift and LLVM tiers, verified
  on aarch64 and x86_64. This is the loop-boundary mechanism; reload
  reuses it by publishing the edited function's helpers against the
  running code's probe sites.
- **Effect dispatch is already indirect.** Regional dispatch reads the
  handler stack and calls through per-handler op tables (module globals
  holding function pointers). Rebinding a handler is a table write, not
  a recompile of its callers.
- **Fibers hold a closure pointer taken at creation**
  (`krio_fiber_new(closure, stack_size)`), and every resume drives the
  fiber through its step encoding. The resume path is the natural
  indirection point.

## What is missing

- **Cross-function call rebinding.** Compiled functions call each other
  directly (`declare_func_in_func`); an unchanged caller keeps calling
  the old callee after a reload. Reloadable calls must go through a
  per-function cell the reload writes — one load per call, the same
  shape as the OSR helper slot.
- **Stable site identity across edits.** OSR site keys derive from block
  indices, which shift under edits. Loop-boundary reload needs keys
  stable under unrelated edits (loop ordinal within the function, not
  block id), so the running code's probe sites match the edited code's
  helpers.
- **A module diff.** Reload takes edited source, reparses, and must
  decide which functions actually changed — content hash per function
  over a normalized HIR, so formatting edits reload nothing.
- **The driver.** Watch the source (or take an explicit
  `reload(source)` call from an embedder), diff, recompile the changed
  set, swap, publish OSR helpers, patch op tables, and report.
- **Fiber resume indirection and the fiber registry.** Resume must read
  the current step code through a cell, and the runtime must be able to
  enumerate live fibers with their layout hashes to decide migration
  per fiber.

## Phases

Build in order; each phase leaves the tree green and is useful alone.

**Phase 0 — reload driver, call boundary.** Function-level HIR content
hash; `reload(source) -> ReloadReport` on the embedder runtime; CLI
`zynml watch <file>`. Changed functions recompile and swap through the
existing pointer table and beads. Calls to reloadable functions route
through per-function cells so unchanged callers observe the swap.
Report: `{reloaded, unchanged, failed}` per function.

**Phase 1 — loop boundary.** Stable OSR site keys (loop ordinal).
On reload of a function with live frames, compile its OSR helpers
against the *old* layout's sites and publish them; the running loop's
next probe transfers into the edited body. Falls back to call boundary
when live-in layouts are incompatible — reported, not silent.

**Phase 2 — persistent fibers.** Resume goes through the fiber's cell;
a reload that changes the fiber's function writes the cell, and the
next resume runs edited code against the existing state when the state
layout hash matches. Registry of live fibers (id, function, layout
hash, status). Policy on mismatch: `complete-on-old` (default) or
`restart`, chosen per reload call. This is the FSM story: each yield is
a state boundary, and an edit lands at the next transition.

**Phase 3 — effects.** Reload of a handler implementation patches the
handler's op-table entries in place; handler state is untouched. Scopes
already entered dispatch to the edited ops on their next perform.
A built-in `Reload` effect surfaces reload events to the program —
the UI DSL's observable boundary: a framework installs a handler and
receives `{function, kind: swapped|migrated|fell_back}` performs.

**Phase 4 — hardening.**

- *All-or-nothing apply.* The reload compiles the whole edit set with
  cell publication deferred; a compile failure anywhere aborts the
  reload — nothing swapped, published, or patched, `aborted` set on
  the report. Per-function declines (unsupported shapes) stay skips.
- *Rollback.* `rollback_last_reload()` restores the generation the
  last applied reload replaced: beads, reload cells, resume points and
  dispatch-table slots all swing back; state untouched. One-shot, and
  observable as the same event a reload emits.
- *Handler-state layout guard, and the migration escape hatch.* A
  stateful handler's state struct is shared between its ctor and its
  ops; an edit that changes that layout declines the whole handler
  group together, so every generation keeps a consistent view.
  Same-shape edits (an initializer value) reload freely. Set
  `StateMigration::ByFieldName` and the group reloads instead: the
  reload plans, per handler, where each field present in both layouts
  moves, and the runtime walks the live frames that name those regions
  — the current stack, fibers' saved segments, parked tasks' — moving
  each one into a region the edited constructor allocated. Fields the
  edit adds start from its initializers; fields it drops are dropped;
  the report names both.
- *Effect-performing functions reload.* An effect's identity crosses
  generations as a number, and a dispatch table is named by an id too,
  so a freshly parsed edit numbers everything differently. A reloaded
  body is rewritten onto the running program's ids — the effect it
  performs, the table it reads, the constant a `with` scope pushes
  under — and matched globals are reused rather than recompiled, since
  their addresses are live in handler frames. Whichever side of a
  perform/scope pair reloads, the other still finds it. Reloaded
  bodies compile against the running module, so a perform keeps its
  operation index. A table the edit introduces is emitted with empty
  slots and filled once its functions are compiled.
- *Async tasks.* A frame that hands out its own address — an async
  poll fn re-parking itself — pins that address to its own generation
  instead of reading the reload cell: a suspended task completes on
  the code it started with, and a task spawned after the edit runs the
  edited code from its first poll.
- *Host-driven fibers.* `get_fiber` / `resume_fiber[_within]` /
  `bind_fiber_handler` / `drop_fiber` / `fiber_info` on the tiered
  runtime: a framework gets a machine instance from a compiled
  `fiber def`, holds a `FiberToken` across reloads and OSR, and steps
  it with handler scopes installed around each step. Handler-state
  persistence is explicit: `resume_fiber_within` opens fresh scopes
  per step; `bind_fiber_handler` allocates state once and carries it
  in the fiber's handler segment for the machine's lifetime. The edit
  edges surface as values: a deleted function answers `MachineGone`
  (drop + remount, not a trap) and leaves the shape registry, and a
  changed yield shape marks the handle stale via a shape generation
  while payloads keep decoding with the creation shape. Rollback
  restores the handles' metadata along with the code. A failed
  multi-handler install unwinds its pushed frames; runtime shutdown
  frees any machines still registered. Names are FQN-aware —
  unqualified names resolve when unambiguous — and `get_effect_handler`
  resolves a handler once into a pinned `EffectHandlerToken` for drives and
  binds, so a host never re-resolves a bare name that a later edit
  could ambiguate. `register_builtin_class` exists on the tiered
  runtime, same seam as the classic one.

Migrating a live frame's locals across an incompatible edit remains a
non-goal: the fallback ladder exists so it is never needed.

## Costs, measured before shipped

The per-call cell load is the only new steady-state cost, and it is the
same mechanism the OSR probe measurement covered (ptrslot: +0.5% on the
bench suite). It is still measured again on the kernels, and the cell
routing is per-function opt-out-able: a build that never reloads pays
nothing when the driver is absent.

## Non-goals

- Migrating a live frame's locals across an incompatible edit. The
  fallback ladder exists so this is never needed.
- Reloading type layouts under live state. Layout changes fall back;
  migration hooks (Phase 4) are the escape hatch, not silent coercion.
- Cross-process or persisted-image reload. This is in-process only.
