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

**Phase 4 — hardening.** Layout-migration hooks (a user-supplied
migration function per type, for the cases a hash mismatch is fixable);
rollback integration (a failed reload restores the previous
generation); interaction with async tasks (await frames are state
machines like fibers and take the same cell treatment).

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
