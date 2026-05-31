# wren_lift GC survey — what's actually there, and how to lift it into Zyntax

A reference reading of `wren_lift/src/runtime/gc*.rs` (and the
interpreter / VM glue that drives it), focused on the parts Zyntax
should adopt as its first opt-in GC menu entry. **Not a plan** —
descriptive only. The plan happens after we agree on which pieces
to lift.

Source paths in this document are relative to
`/Users/amaterasu/Vibranium/wren_lift/`. Zyntax-side analogues are
relative to the Zyntax workspace root.

---

## 0. Correction up front

The session's earlier note ("wren_lift uses generational immix") is
wrong. There is **no immix anywhere in wren_lift** — no code,
docs, reference dir, or git history mentions it. The default is a
**generational collector with a bump-pointer copying nursery and a
mark-sweep old gen** (`src/runtime/gc.rs:1`). The body of this
survey reflects the actual algorithm.

That said, the *menu architecture* the user described — multiple
GC algorithms behind a trait, runtime-selectable — is exactly what
wren_lift ships. That pattern is the highest-leverage thing to
lift, independent of which algorithms Zyntax eventually offers.

---

## 1. Architecture: trait + enum dispatch (the menu pattern)

`src/runtime/gc_trait.rs` defines:

```rust
pub trait GcAllocator {
    fn alloc_string(...) -> *mut ObjString;
    fn alloc_list(...) -> *mut ObjList;
    // ... one per object kind ...
    fn intern_string(...) -> *mut ObjString;
    fn write_barrier(&mut self, source: *mut ObjHeader, value: Value);
    fn collect(&mut self, roots: &mut [Value]);
    fn should_collect(&self) -> bool;
    fn stats(&self) -> &GcStats;
}

pub enum GcImpl {
    Generational(super::gc::Gc),       // default
    Arena(ArenaGc),                    // alloc-only, free on drop
    MarkSweep(MarkSweepGc),            // simple stop-the-world
}

pub enum GcStrategy {
    #[default] Generational,
    Arena,
    MarkSweep,
}
```

Dispatch is by enum-match (a `gc_dispatch!` macro), **not** trait
objects or generics. The comment at line 65-71 spells out why:
trait objects add vtable indirection on every allocation; generics
infect 30+ files with parameter pollution. Enum match becomes a
single predicted branch.

The CLI wires `--gc` to a `GcStrategy` enum at `src/main.rs:223-238`:

```rust
let gc_strategy = match cli.gc {
    GcMode::Generational => GcStrategy::Generational,
    GcMode::Arena => GcStrategy::Arena,
    GcMode::MarkSweep => GcStrategy::MarkSweep,
};
```

**Lift directly:** this trait+enum-dispatch shape is the right
seam for Zyntax. `CompilationConfig.memory_strategy` already exists;
extending its variants for opt-in GC algorithms is the natural
place to wire this.

---

## 2. Object header — 24 bytes, repr(C)

Every heap object starts with `ObjHeader` (`src/runtime/object.rs:127-141`):

```
offset 0:  obj_type   u8       // type tag
offset 1:  gc_mark    u8       // 0=white, 1=gray, 2=black, 3=forwarded
offset 2:  generation u8       // 0=young (nursery), 1=old
offset 3-7: padding
offset 8:  next       *mut     // intrusive linked list (sweep walks this)
offset 16: class      *mut     // method dispatch
                              // total 24 bytes
```

Two GC-relevant decisions baked into this header:

- **Tri-color marking inline** in `gc_mark`. `FORWARDED` (value 3)
  is overloaded — used during minor GC, where the surviving
  nursery slot stores a forwarding address in place of its data
  and the mark byte signals "look in the forwarding field".
- **Intrusive linked list** via `next`. The old gen is just a head
  pointer + linked chain; sweep walks the chain and unlinks dead
  objects. Cheap, but precludes parallel sweep without extra
  bookkeeping.

**Zyntax mapping:** today `HirType::Ptr(Opaque(name))` carries no
GC-aware metadata. To make objects GC-managed, Zyntax would need
either a `HirType::Gc(...)` variant or a per-type attribute that
the lowering uses to allocate via the GC trait (vs. malloc/Box).
The header layout itself is opaque to the language — the
allocator owns it; user code never reads `gc_mark`.

---

## 3. Default GC — generational bump-nursery + mark-sweep old gen

### 3.1 Nursery (`gc.rs:121-219`)

Single contiguous `Vec<u8>` with a bump pointer:

```rust
struct Nursery {
    buffer: Vec<u8>,
    alloc_ptr: usize,
}
```

- `try_alloc<T>(val)` aligns, bumps, returns `*mut T`. O(1).
- `reset()` slams `alloc_ptr` back to 0 after a collection.
- `contains(ptr)` is a simple address-range check — used by the
  write barrier to identify young pointers.

Allocation fast path (`alloc<T>`, `gc.rs:711-724`): if the nursery
has space, bump-allocate there and push the new header into
`nursery_objects: Vec<*mut ObjHeader>` (so the collector can
iterate them later). On nursery overflow, fall through to direct
old-gen allocation.

Default nursery size: **64 MB**, overridable via `WLIFT_GC_NURSERY_MB`
env var. The comment at `gc.rs:238-251` notes the tension: bigger
nursery → fewer collections but larger floor (matters on small
VMs like Fly's shared-cpu-1x); smaller nursery → more frequent
collections but tighter resident set.

### 3.2 Old gen (`gc.rs:35-119` for `OldArena`, plus a parallel
intrusive linked list)

Chunk-based bump allocator:

```rust
pub struct OldArena {
    chunks: Vec<Vec<u8>>,     // each chunk is a separately heap-allocated block
    alloc_ptr: usize,
    chunk_size: usize,         // 256 KB initial
}
```

Plus the parallel state in `Gc`:

```rust
old_objects: *mut ObjHeader,  // head of intrusive linked list
old_count: usize,
old_arena: OldArena,
remembered_set: Vec<*mut ObjHeader>,  // old objs with young pointers
intern_table: HashMap<u64, Vec<*mut ObjString>>,
```

Each new chunk is 256 KB. Pointers into earlier chunks stay valid
because the `Vec<Vec<u8>>` never reallocates the inner chunks
(only adds new ones). Sweep walks `old_objects` and reclaims
unmarked entries; chunks that end up entirely empty get released
back to the OS (`release_empty_old_chunks`, `gc.rs:940`).

### 3.3 Allocation policy quirks

A few decisions worth flagging:

- **`alloc_class` always goes to old gen** (`gc.rs:653-661`):
  ```rust
  // ObjClass instances live for the program's lifetime and
  // get baked into JIT class-check constants. Allocate them
  // straight to the old arena so the pointer is stable for
  // the lifetime of any JIT'd code that references it — a
  // nursery-then-promoted class would invalidate every
  // class-check that captured the old address.
  ```
  This is a **JIT-stability constraint** baked into the GC.
  Zyntax has the same problem: any address baked into wasm-JIT'd
  code must not move. Zyntax's `register_static_plugins` is the
  natural analogue — the symbol pointers are similarly stable
  for-life and must skip nursery promotion.

- **`alloc_instance` allocates both header + fields in nursery
  in one shot** (`gc.rs:665-699`). Falls back to old gen if the
  combined size doesn't fit. This avoids fragmenting a struct
  across nursery + old gen on the first cycle.

- **String interning** is hash-keyed (`intern_table:
  HashMap<u64, Vec<*mut ObjString>>`). Collisions resolved by
  linear scan of the bucket's value list, comparing actual
  bytes. Interned strings still participate in normal GC —
  the table itself gets updated to track forwarded addresses
  during minor GC.

### 3.4 should_collect heuristic (`gc.rs:404-450`)

Multi-trigger. Fires when ANY of:

1. Nursery used > **75% of capacity** — primary trigger for
   short-lived churn.
2. Nursery used > **64 MB absolute ceiling** — caps idle
   floor on long-running event loops where the nursery
   would otherwise sit at hundreds of MB of dead objects.
3. **External bytes** charged via `track_external` exceed
   **32 MB**. Used by `Fiber.new` to charge krio mmap stacks
   that don't show up in nursery accounting. Without this,
   a web server's RSS climbs unbounded per request.
4. **Old gen count** ≥ `gc_threshold` (starts at 256,
   grows by `heap_grow_factor: f64 = 2.0` per cycle).
5. **Wall-clock heartbeat**: 30 s since last major + any
   allocation since. Catches drip-feed workloads (favicon
   polls, prefetch hits) where neither byte threshold trips
   for minutes.

The 5-way OR is the kind of "every axis I've seen go wrong in
production" heuristic that comes from running this in real
deployments. Worth importing wholesale into Zyntax with the
same env-var override for the nursery size.

### 3.5 Write barrier (`gc.rs:771-793`)

Precise post-write barrier. Generational invariant: anytime an
old object stores a pointer to a young object, that old object
joins the `remembered_set`.

```rust
pub fn write_barrier(&mut self, source: *mut ObjHeader, value: Value) {
    if !value.is_object() || source.is_null() {
        return;
    }
    unsafe {
        if (*source).generation != GEN_OLD {
            return;
        }
        if let Some(obj_ptr) = value.as_object() {
            let target = obj_ptr as *mut ObjHeader;
            if !target.is_null() && (*target).generation == GEN_YOUNG {
                self.remembered_set.push(source);
            }
        }
    }
}
```

The order of the checks is tuned for the common case: non-object
values bail before touching `*source`. The comment at lines 774-781
notes this was a deliberate reorder — the previous version
dereferenced `source.generation` first, which paid a memory
load on every store even for primitive writes.

**Call sites in the interpreter** (`vm_interp.rs`):
- `SetUpvalue` → `vm.gc.write_barrier(uv_ptr, v)` (line 1506)
- `SetStaticField` → `vm.gc.write_barrier(class_ptr, v)` (line 1558)
- Instance-field store → `vm.gc.write_barrier(ptr, val)` (line 1744)

There's also a debug invariant validator at `gc.rs:454-589`:
`validate_write_barriers()` walks every old object, finds every
young reference reachable from it (by checking field-by-field
per object type), and panics if any are missing from the
remembered set. Exhaustive, expensive, debug-only — but exactly
the kind of belt-and-suspenders that catches the "I forgot to
barrier this store" class of bug at test time instead of in
production. Zyntax should adopt the same pattern.

### 3.6 Minor GC algorithm (`gc.rs:825-870`)

```
1. Mark from roots (gray stack DFS).
2. Mark from remembered set (treat each entry as a root —
   it's an old object that might reference young).
3. Process gray stack to completion.
4. process_nursery: walk nursery_objects.
     For each live object, Box-promote to old gen.
     Store forwarding address INLINE in the nursery slot
     (gc_mark = FORWARDED, next 8 bytes = new address).
     Dead objects: drop in place.
5. Update all pointers that pointed into nursery to their
   forwarded addresses:
     - update_roots_inline(roots, &nursery)
     - update_old_gen_pointers_inline()
     - update_intern_table_inline()
6. Reset nursery arena (bump pointer → 0).
7. Reset gc_mark to WHITE on all old objects (they were
   marked during tracing).
```

The **inline forwarding pointer** is the clever bit: no separate
forwarding table. Step 4 writes the new address into the dead
nursery slot's header, and step 5 reads it back. The slot is
recycled when step 6 resets the arena. This works because nursery
slots are stable until reset (the underlying `Vec<u8>` isn't
touched until `nursery.reset()`).

### 3.7 Major GC algorithm (`gc.rs:873-934`)

Minor GC + sweep old gen + release empty arena chunks +
optional `madvise(MADV_FREE)` via wlift_alloc.

```
1. Mark from roots + remembered set (same as minor).
2. Promote nursery survivors (same as minor).
3. Sweep dead old-gen objects: walk old_objects, free
   anything still WHITE, splice out of the linked list,
   reset BLACK survivors to WHITE.
4. release_empty_old_chunks: free any OldArena chunk
   that contains no surviving object. Critical for
   long-running servers — without it, a chunk that
   ever held a now-dead object stays allocated forever.
5. wlift_alloc::pressure_release() — calls madvise to
   hand free pages back to the OS so process RSS
   actually drops.
```

The combination of (3) (linked-list traversal sweep), (4)
(chunk reclamation), and (5) (OS page release) is what
makes the steady-state RSS track the working set rather than
the high-water mark. Steps 4-5 are an operational detail that
matters disproportionately for hosted environments (containers,
serverless) but is invisible from the algorithm description.

### 3.8 The minor↔major decision (`gc.rs:798-821`)

```rust
let external_pressure = self.external_bytes_since_gc > 4 * 1024 * 1024;
if external_pressure || self.minor_since_major >= self.config.major_gc_interval {
    self.collect_major(roots);
} else {
    self.collect_minor(roots);
}
```

- Force major when **off-heap pressure** exceeds 4 MB (about 16
  krio fibers' worth of mmap stacks). Without this, dead fibers
  promoted by an earlier minor sweep accumulate in old gen.
- Otherwise, every 8th collection (`major_gc_interval: u32 = 8`)
  is a major.

---

## 4. Safepoint placement

Two safepoint mechanisms run in parallel:

### 4.1 Interpreter safepoint check (`vm_interp.rs:1312-1373`)

Inside the bytecode dispatch loop, every `check_interval` steps
(0xF / 0xFF / 0xFFF depending on `step_limit` config):

```rust
if steps & check_interval == 0 {
    if vm.gc.should_collect() || vm.gc_requested {
        // 1. Stash live registers back into frame.values
        //    so the GC can scan them as roots.
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            frame.values = std::mem::take(&mut values);
            frame.pc = pc;
        }
        // 2. Trigger collection.
        vm.collect_garbage();
        vm.method_cache.invalidate();
        vm.engine.invalidate_inline_caches();
        // 3. Restore registers from (possibly forwarded) frame.
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            values = std::mem::take(&mut frame.values);
        }
    }
}
```

The **stash-around-collect** dance is the crucial pattern: live
interpreter registers go back into the frame structure before GC
runs (so they're discoverable as roots) and come back out after.
Without this, registers that were "live in a CPU register but not
yet written back to memory" would be invisible to the collector.

The check is gated on `steps & check_interval == 0` — a single AND
and a branch, hottest path stays fast.

### 4.2 JIT safepoint stack maps (`vm.rs:2915-2984`)

JIT'd functions don't go through the interpreter dispatch loop,
so the safepoint there isn't enough. The JIT lowering emits
**stack-map metadata** at every potential GC point (call sites,
loop back-edges, allocation sites):

```rust
struct LiveRootMetadata {
    location: RootLocation,   // e.g. Spill(offset)
    // …other metadata
}

struct SafepointMetadata {
    code_offset: u32,         // PC offset within the function
    live_roots: Vec<LiveRootMetadata>,
}
```

At GC time, `scan_native_stack_roots()` does:

1. **Pass 1** — walk explicitly-pushed JIT frame list (each
   `wren_call_N` boundary pushed `(fp, func_id, ret_addr)`).
   For each frame, find the active safepoint by matching
   `ret_addr - code_range.start` against `safepoint.code_offset`,
   then scan only the spill slots listed in
   `safepoint.live_roots`.
2. **Pass 2** — aarch64 frame-pointer chain walking, for AOT
   bodies that don't push JIT frames (entered straight from the
   linker's main).

**Precise scanning**: the safepoint stack map tells the GC
*exactly* which slots hold object pointers at that PC. The fallback
is to skip scanning if no match found — wren_lift deliberately
doesn't fall back to conservative (full-stack scan), because that
would scan non-object spill slots and corrupt them.

**Zyntax mapping:** the BC interpreter pattern (4.1) ports
directly — Zyntax's `HirInterpreter` already has a dispatch loop
that could host a safepoint check at a similar interval. The
JIT stack-map machinery (4.2) is more involved; the wasm backend
would need to emit safepoint metadata alongside the wasm bytes,
and the host glue would need to know how to discover live spill
slots in wasm linear memory. Probably a later phase — start with
interpreter-only safepoints.

---

## 5. Root-set discovery

Three sources of roots, all consolidated when the safepoint fires:

1. **Fiber stacks** — every `ObjFiber` carries `stack: Vec<Value>`
   and a chain of `mir_frames`, each with its own `values: Vec<Value>`
   register file and `closure` / `defining_class` pointers.
2. **JIT spill slots** — discovered via the stack-map walk in 4.2.
3. **Global module variables** — `ObjModule` carries
   `variables: Vec<Value>`; all loaded modules are roots.

The fiber set is discovered via `GcImpl::for_each_fiber` (`gc_trait.rs:229`),
which goes through the GC's *own object list* rather than a side
table — that way every fiber the closure sees is, by definition,
currently allocated. (The comment at `gc_trait.rs:223-228` calls
this out as anti-dangling-pointer hygiene.)

**Zyntax mapping:** the natural analogue is the `HirInterpreter`'s
register VM state — `CompiledFunction` register files, the value
stack, the symbol table. Plus krio-fiber stacks for async (which
Zyntax already shares with wren_lift via the krio adapter).

---

## 6. The other two menu entries — NOT being lifted

wren_lift ships two additional variants. **Neither is being lifted
into Zyntax.** Recorded here only for reference.

### 6.1 ArenaGc (`gc_arena.rs`, 274 LOC) — explicitly excluded

"Allocates, never collects." All objects are `Box::into_raw`-ed
and tracked in a `Vec<*mut ObjHeader>`; the whole arena is freed
when the `ArenaGc` drops. O(1) alloc, zero pause time,
**unbounded memory growth**.

**Excluded from Zyntax's menu by user decision.** Even with docs
warning "only for short-lived workloads," exposing it lets a user
pick it for a long-running embedder and OOM. The menu shouldn't
ship foot-guns.

### 6.2 MarkSweepGc (`gc_marksweep.rs`, 532 LOC) — not in first lift

Simple non-generational stop-the-world. All objects in a single
`Vec`, threshold-triggered mark phase + sweep phase. No nursery,
no promotion, no write barriers needed. Pause time proportional
to total heap; no compaction.

**Not part of the first-lift scope.** The user has scoped initial
GC work to *just* the generational variant (§3 and §7). MarkSweep
might join the menu later — its main appeal as a baseline /
no-barrier fallback doesn't outweigh the cost of building it now
when the generational collector is the only one being shipped
first.

---

## 7. Mapping onto Zyntax — concrete seams

This section identifies the specific files / line ranges where
the integration would land. Not a plan; a map.

### 7.1 GC strategy selection

| Concept | wren_lift | Zyntax analogue |
|---|---|---|
| Strategy enum | `runtime/gc_trait.rs::GcStrategy` | `compiler/src/lib.rs::MemoryStrategy` (already exists, opt: `Option<MemoryStrategy>`) |
| Enum dispatch | `GcImpl` + `gc_dispatch!` macro | New: `compiler/src/gc/mod.rs::GcImpl` (parallel structure) |
| Trait | `GcAllocator` | New: `compiler/src/gc/trait.rs::GcAllocator` |
| CLI surface | `main.rs::cli.gc` | `zyntax_cli` `--memory-strategy` flag (extend existing) |

### 7.2 Object header

`HirType::Gc(...)` or `HirType::HeapPtr(Opaque(...))` would be the
likely shape. The lowering's `Alloc`/`Box` HIR ops (today they
emit malloc + free at drop sites under the default
speculative-drop strategy) would route to `gc.alloc_*` instead
when the strategy is GC-enabled.

Header layout itself sits in `compiler/src/gc/object.rs` — modeled
after `wren_lift/src/runtime/object.rs:127-141`, adapted for
Zyntax's actual object kinds (no ObjFn / ObjClass per se; we
have HirFunction id references, struct/enum instances,
List / Map / String stdlib types).

### 7.3 Safepoint check

| wren_lift | Zyntax |
|---|---|
| `vm_interp.rs:1330-1373` (steps & check_interval == 0) | `crates/compiler/src/hir_interp.rs` dispatch loop — same shape, same `gc.should_collect()` predicate |
| Stash registers back into frame before GC | `HirInterpreter` register file needs to be discoverable from outside the inner loop — minor refactor |
| Stack map for JIT spills | Phase out — the wasm backend would need a `SafepointMetadata` emission pass, deferred |

### 7.4 Write barrier

In Zyntax, every `HirInstruction::Store` with a heap-pointer
target needs to be barrier-aware **when the GC strategy is
generational**. Two ways:

1. **Lowering pass**: a transformation that walks the HIR after
   the SSA build, inserts `Call(Symbol("gc_write_barrier"), [src, val])`
   before each Store whose target is heap-typed. Cleanly
   separated from the rest of lowering, and naturally disabled
   under non-GC strategies (default speculative-drop, future
   non-generational variants) by skipping the pass entirely.
2. **Cranelift/Wasm backend hook**: emit the barrier inline at
   Store sites. Faster (no extra HIR instruction, no extra
   regalloc pressure) but couples the backends to GC details.

Option 1 is the cleaner first pass. The barrier-as-Symbol-call
also makes it cheap to A/B different barrier implementations
(remembered set, card marking, etc.) without backend changes.

### 7.5 should_collect heuristic

Lift the 5-trigger predicate from `gc.rs:404-450` essentially
verbatim. Constants (75%, 64 MB, 32 MB, gc_threshold growth
factor, 30 s heartbeat) all become `GcConfig` fields with the
same env-var overrides. Don't try to invent something better
on first pass — these values were tuned against real wren_lift
deployments.

### 7.6 Root-set discovery

Zyntax-specific. Sources:

- **HirInterpreter register files** (`CompiledFunction.regs`,
  the value stack).
- **HirInterpreter symbol table** (`HirInterpreter::symbol_table_snapshot`,
  the FFI registered symbols — these are weak roots: the
  underlying function pointer doesn't need GC tracing because
  it's a code address, but any boxed receiver state stored
  alongside might).
- **Global / module statics** if/when Zyntax adds module-level
  state.
- **Krio fiber stacks** when async is in use. The krio adapter
  already exposes fiber discovery; same pattern as
  `GcImpl::for_each_fiber`.

### 7.7 Debug invariant check

Port `gc.rs::validate_write_barriers` (`#[cfg(debug_assertions)]`,
exhaustive young-pointer walk per old object). High debugging
value, zero release cost.

---

## 8. What's out of scope here

- **Concurrent / parallel GC.** wren_lift is single-threaded
  stop-the-world. The interpreter loop stops, the collector
  runs, the loop resumes. For Zyntax's first opt-in GC entry,
  same constraint is fine — concurrent collection is a future
  menu entry of its own.
- **Generational *immix* (mark-region).** Not in wren_lift, not
  on this survey. A separate study if/when that algorithm
  joins the menu.
- **The wasm-JIT path.** Zyntax's wasm tier-1 emitter would need
  safepoint stack maps to fully integrate; the BC interpreter
  is the proper first integration target.
- **Off-heap accounting** (`track_external`). Important once
  Zyntax has krio fibers in scope, but not on the critical
  path for a minimum-viable GC menu entry.

---

## 9. References

- `wren_lift/src/runtime/gc.rs` — default GC (2289 LOC)
- `wren_lift/src/runtime/gc_arena.rs` — Arena variant (274 LOC)
- `wren_lift/src/runtime/gc_marksweep.rs` — MarkSweep variant (532 LOC)
- `wren_lift/src/runtime/gc_trait.rs` — trait + enum dispatch (317 LOC)
- `wren_lift/src/runtime/object.rs:127-141` — ObjHeader layout
- `wren_lift/src/runtime/vm_interp.rs:1312-1373` — interpreter safepoint
- `wren_lift/src/runtime/vm.rs:2915-2984` — JIT precise stack scanning
- `wren_lift/src/main.rs:223-238` — CLI strategy plumbing
