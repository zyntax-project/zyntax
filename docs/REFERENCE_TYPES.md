# Reference Types (`@reference`)

Zyntax classes are **value types by default**: an instance lives inline in
whatever container holds it, and `let a = bodies[i]` copies the whole
struct. The compiler can opt a class into **reference semantics** with a
single attribute:

```zynml
@reference
struct Body {
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    mass: f64
}
```

Once a class is annotated, its instances live on the heap, its array
slots hold pointers, and `let a = bodies[i]` copies a pointer instead of
the whole struct. Field reads and writes go through the pointer.

This is part of a broader **bi-modal memory model**: every Zyntax program
chooses value-type or reference-type semantics per class, orthogonal to
whichever memory-management strategy is in effect.

## Why two modes

The choice is structurally load-bearing on numerical kernels. The
classic n-body benchmark with five `Body` instances:

```
                  median exec_ms        Array<T> slot size
value-type Body        1590 ms                56 bytes
@reference Body         653 ms                 8 bytes
```

That ~2.4× speedup is not from any clever optimization pass — both
versions compile through identical pipelines. It's pure access-pattern
delta:

* `Array<Body>` (value) becomes `Array<Ptr<Body>>` (reference): the
  array slot shrinks from 56 B to 8 B, and five bodies fit comfortably
  inside one cache line as pointers.
* The hot loop's `let mut a = bodies[i]` switches from a 56-byte struct
  memcpy to an 8-byte pointer load.
* Field mutations switch from `copy → mutate → store back into the
  array slot` to in-place `GetElementPtr + Store` through the pointer.

The choice is exposed because there is no universally correct answer:
small POD types (`Point`, `Color`, `Vec3`) usually want value semantics
for cache density; large mutable graphs (`Tree`, `Graph`, `Node`) want
reference semantics for aliasing and shared mutation. The decision
belongs to the program, not the compiler.

## Semantic differences

For a class `C` with fields `f1, f2, ...`:

| Operation             | Value-type (default)                       | `@reference`                                    |
|-----------------------|--------------------------------------------|-------------------------------------------------|
| `C { f1: …, f2: … }`  | inline aggregate, no allocation            | `Malloc(sizeof(C))` then per-field `GEP+Store`  |
| `let a = c`           | copies the whole struct                    | copies the pointer (aliases the same instance)  |
| `a.f1`                | aggregate-extract (no memory access)       | `GEP(a, field_offset)` then `Load`              |
| `a.f1 = x`            | rebuild aggregate, store back to `a`       | `GEP(a, field_offset)` then `Store`             |
| `arr[i] = c`          | copies struct into the slot                | stores the pointer into the slot                |
| `arr[i].f1 = x`       | mutate temporary, store struct back        | `GEP+Load` slot pointer, then `GEP+Store` field |

The most subtle difference is aliasing. With reference semantics, two
bindings to the same instance share storage:

```zynml
let a = bodies[0]
let b = bodies[0]
a.vx = 1.0
// Value-type:    b.vx is still its original value.
// @reference:    b.vx is now 1.0 (a and b are the same pointer).
```

This matches mainstream object semantics in Java, Python, C# (reference
types) and is what most large mutable data structures want. For
numerical work it also enables the in-place update pattern the n-body
benchmark exploits.

## Lowering

The compiler routes `@reference` classes through a different lowering
path entirely:

```
TypedClass(annotations: [@reference], fields)
        │
        ▼
TypeMetadata { is_reference: true, … }   (typed_ast::type_registry)
        │
        ▼
ssa.rs::convert_type returns:
        HirType::Ptr(Box::new(HirType::Struct{Foo fields}))
                    instead of HirType::Struct{…}
        │
        ▼
ssa.rs Struct-literal arm                  ssa.rs Field-access arm
        │                                          │
        ▼                                          ▼
emit Call(Intrinsic::Malloc, sizeof)         emit GetElementPtr(ptr, offset)
emit per-field GetElementPtr + Store         emit Load (read) or Store (write)
return the ptr as the value                  return the loaded/written value
```

Three points carry the design:

1. **`TypeMetadata.is_reference` is the single source of truth.** It is
   set during type registration (`import_chain::register_struct_declarations`
   and the two mirror paths in `runtime.rs`) and read by `convert_type`.
   Every other site reads through `convert_type`, so adding a new SSA
   instruction never needs to special-case `@reference`.

2. **`HirType::Ptr(Struct{...})` is the carrier.** This is the same
   variant used for `*T` and existing FFI types, so every backend
   (Cranelift, LLVM, BC interpreter, bytecode serialization) already
   knows how to handle it. There is no new HIR type to land.

3. **Lowering is a branch, not a fork.** Both the struct-literal arm
   and the field-access arm check `convert_type` for `Ptr(Struct)` and
   take the heap path; otherwise they fall through to the existing
   value-type code. Classes without `@reference` are bit-identical to
   what they compiled to before this feature shipped.

## Interaction with `scalar_replace_alloc`

The `scalar_replace_alloc` pass deletes opaque `Call(Intrinsic::Malloc)`
when the result demonstrably does not escape its single basic block.
Reference-class instantiation produces exactly the pattern this pass
recognises (constant-size malloc, constant-offset GEPs, no escaping
operand). For short-lived locally-scoped instances, the heap allocation
gets folded back into SSA registers and the runtime cost collapses to
zero.

```
@reference struct Point { x: i64, y: i64 }

def main() -> i64:
    let p = Point { x: 10, y: 20 }
    return p.x + p.y
```

```
ZYNTAX_SRA_DUMP=1 zynml run …/ref_class_sra.zynml
scalar_replace_alloc: examined=1 eliminated=1 frees=0 escapes=0
```

For instances that escape — stored into a container, passed to an
opaque function, returned — the malloc stays and the speculative
drop-site analysis (`drop_insert.rs`) pairs it with a `Free`. On
n-body the five `Body` mallocs escape into the `bodies` array, so the
malloc-elimination path does not fire, but the access-pattern win
alone delivers the 2.4× speedup.

In short: SRA recovers value-type performance for ephemeral
`@reference` use; the structural access-pattern win does the heavy
lifting when the instance must live somewhere.

## Memory management orthogonality

Reference semantics decide **how field access is lowered**. They do
not decide **when memory is freed**. Zyntax exposes that as a separate
opt-in axis with three planned entries:

1. **Speculative drop-site analysis (default).** Compile-time `Free`
   insertion via `drop_insert.rs`. V1 handles same-block lifetimes; a
   future cross-block dataflow extension is needed for instances that
   survive a loop iteration.
2. **Opt-in GC menu.** A `CompilationConfig.memory_strategy: Option<MemoryStrategy>`
   selects a collector (planned: generational copy-nursery + mark-sweep
   old gen first; mark-sweep and RC variants later).
3. **Opt-in Rust-style borrow/lifetime/move.** `borrow_check.rs` is the
   seed of this third entry; extending it from a single pass into a
   per-program opt-in mode is the path.

Any of these can pair with either value-type or `@reference` access
semantics. A program can run with `@reference` classes under
drop-site analysis (n-body today), under a generational GC (future
graph workloads), or under explicit borrow checking (future
concurrent code).

## Known gaps in V1

* **Cross-block escape analysis** in `scalar_replace_alloc` and
  `drop_insert.rs` is still pending. Loop-carried `@reference`
  instances that escape across basic-block boundaries are
  conservatively kept alive for the function's lifetime. This is
  correct but leaves perf on the table for programs that allocate
  fresh `@reference` instances inside hot loops.
* **Custom destructors** (`def drop(self)`) are not yet recognised by
  the runtime. `memory_management::get_destructor` currently returns
  `None` for all classes. The hook exists; wiring it is a follow-up.
* **Method receiver `self`** for `@reference` classes routes through
  the same `convert_type` branch as field access, so it should work
  end-to-end, but the integration tests do not yet exercise methods
  on a reference class.
* **`Shared<T>` / Arc opt-in** for explicitly-shared mutable state is
  planned in `memory_management.rs` (the `ARCManager` already knows
  Ref types need refcount semantics) but is not yet exposed at the
  language surface.

## Case study: n-body

The n-body benchmark integrates Newtonian gravity for 5 bodies over
10 million `advance(0.01)` steps. Both kernel variants are in
`crates/zynml/examples/`:

* [`bench_nbody.zynml`](../crates/zynml/examples/bench_nbody.zynml) —
  value-type `Body`. Baseline.
* [`bench_nbody_ref.zynml`](../crates/zynml/examples/bench_nbody_ref.zynml) —
  byte-identical except for the leading `@reference` on `Body`.

The two files diverge by exactly one line. The compiler does the rest:

* `bodies = [sun, jupiter, saturn, uranus, neptune]` builds an
  `Array<Body>` (5 × 56 = 280 B) or `Array<Ptr<Body>>` (5 × 8 = 40 B)
  depending on the annotation.
* `let mut a = bodies[i]` issues a 56-byte struct-load or an 8-byte
  pointer-load.
* `a.vx = a.vx - dx * b.mass * mag` rebuilds-and-stores the Body
  or `GEP+Store`s a single `f64` through the pointer.
* `bodies[i] = a` writes 56 B back into the array or rewrites the
  same pointer (effectively a no-op).

Five-trial medians on `zyntax-tiered`:

```
bench_nbody      ~1590 ms   →   Int(-169077)
bench_nbody_ref   ~653 ms   →   Int(-169077)
```

Identical numerical result. Different runtime cost. One-line change.
