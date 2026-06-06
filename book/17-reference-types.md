# Chapter 17: Reference Types

By default a Zyntax `class` (or `struct`) is a **value type**: every
`let a = bodies[i]` copies the whole struct, every `arr[i] = c` writes
the whole struct into the slot, and field access reads from an inline
aggregate with no pointer indirection. This is fast for small data
(`Point`, `Color`, `Vec3`) because there is no heap traffic and the
data sits in cache.

For larger structures, structures that are passed around, or any data
you want to mutate in place from multiple bindings, value semantics
become a tax. A single attribute switches the class into **reference
type** semantics:

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

Once `Body` is `@reference`, instances live on the heap, `Array<Body>`
holds pointers, `let a = bodies[i]` copies a pointer, and field reads
and writes go through that pointer.

## When to reach for it

| Use value-type (default)                    | Use `@reference`                                     |
|---------------------------------------------|------------------------------------------------------|
| Small POD: `Point2D`, `Color`, `Vec3`       | Anything with more than ~4 fields                    |
| Math kernels with tight cache packing       | Mutable graphs, trees, scene nodes                   |
| Pure data that flows through pipelines      | Long-lived objects whose identity matters            |
| Anything you copy more often than mutate    | Anything you mutate from multiple places             |

A useful rule of thumb: if you find yourself writing `arr[i] = a` to
manually store back a temporary you just mutated, the type wants to
be `@reference`.

## What changes

For a class `C` with fields `f1, f2, …`:

| Code                  | Value-type semantics                       | `@reference` semantics                          |
|-----------------------|--------------------------------------------|-------------------------------------------------|
| `C { f1: …, f2: … }`  | inline aggregate, no allocation            | one heap allocation, fields written via pointer |
| `let a = c`           | copies the whole struct                    | copies the pointer (aliases `c`)                |
| `a.f1`                | reads from the inline aggregate            | reads through the pointer                       |
| `a.f1 = x`            | mutates `a`'s storage                      | mutates the shared instance through the pointer |
| `arr[i] = c`          | copies struct into the slot                | stores the pointer into the slot                |
| `arr[i].f1 = x`       | mutates a temporary, store struct back     | mutates the shared instance directly            |

The biggest semantic difference is **aliasing**:

```zynml
let a = bodies[0]
let b = bodies[0]
a.vx = 1.0
// Value-type:    b.vx is still its original value.
// @reference:    b.vx is now 1.0 — a and b are the same instance.
```

This matches mainstream object semantics in Java, Python, and C# for
reference types, and is what most large mutable structures want. For
numerical work it also enables in-place updates that the value-type
path can't express efficiently.

## A concrete example: n-body

The classic n-body benchmark integrates Newtonian gravity for five
bodies. Here's the value-type inner loop:

```zynml
struct Body {
    x: f64, y: f64, z: f64,
    vx: f64, vy: f64, vz: f64,
    mass: f64
}

def advance(bodies: Array<Body>, n: i64, dt: f64): i64 {
    let mut i: i64 = 0
    while i < n {
        let mut a = bodies[i]            // copies 56 bytes
        let mut j: i64 = i + 1
        while j < n {
            let mut b = bodies[j]        // copies 56 bytes
            // ... compute dx, dy, dz, distance, mag ...
            a.vx = a.vx - dx * b.mass * mag
            // ... more updates to a and b ...
            bodies[j] = b                // copies 56 bytes back
            j = j + 1
        }
        bodies[i] = a                    // copies 56 bytes back
        i = i + 1
    }
    return n
}
```

Each inner-loop iteration moves ~224 bytes of `Body` data
(2 loads + 2 stores). The `Array<Body>` slot is 56 bytes wide, so
five bodies are 280 bytes of array — multiple cache lines.

Now change exactly one thing:

```zynml
@reference
struct Body {
    x: f64, y: f64, z: f64,
    vx: f64, vy: f64, vz: f64,
    mass: f64
}
```

Everything else stays the same. The compiler now lowers `Body` as a
pointer:

* `Array<Body>` becomes `Array<Ptr<Body>>` — 5 × 8 = 40 bytes, one
  cache line.
* `let mut a = bodies[i]` becomes an 8-byte pointer load.
* `a.vx = a.vx - …` becomes a direct `GetElementPtr + Load + Store`
  through the pointer; the body data is mutated in place.
* `bodies[i] = a` writes the same pointer back into the slot — it
  carries no work since the pointee was modified in place.

Five-trial medians on `zyntax-tiered`:

```
bench_nbody      ~1590 ms   →   Int(-169077)
bench_nbody_ref   ~653 ms   →   Int(-169077)
```

Identical numerical result. One annotation. ~2.4× faster.

Both kernels ship in the repo:

* [`bench_nbody.zynml`](../crates/zynml/examples/bench_nbody.zynml) — value-type baseline
* [`bench_nbody_ref.zynml`](../crates/zynml/examples/bench_nbody_ref.zynml) — `@reference` version

You can run them yourself:

```bash
cargo build --release --example bench_runner -p zynml
./target/release/examples/bench_runner --runs 3 --filter nbody
```

## What `@reference` does *not* do

`@reference` decides how field access is lowered. It does **not**:

* **Add automatic refcounting.** Reference-type instances are not
  `Arc`-counted. If you want shared ownership with refcounts, that's
  a separate opt-in (the `Shared<T>` story below).
* **Make the class garbage-collected.** Memory is reclaimed by
  Zyntax's speculative drop-site analysis (the default
  memory-management mode), which inserts `Free` calls at compile
  time. A GC opt-in is planned as a separate axis.
* **Add nullability.** A `@reference` instance is still a non-null
  reference. Use `Option<T>` if you need nullability.
* **Change method dispatch.** Methods on a `@reference` class are
  monomorphised the same way as on a value-type class.

## How memory is reclaimed

Zyntax frees `@reference` instances using **speculative drop-site
analysis**. The compiler inserts `Free` calls at the point where it
proves an instance is no longer reachable:

```zynml
def compute(): i64 {
    let p = Point { x: 10, y: 20 }   // Malloc here
    let r = p.x + p.y
    return r
                                     // Free inserted here by drop_insert
}
```

For short-lived locally-scoped instances, the compiler can go one
step further and **eliminate the heap allocation entirely** via the
`scalar_replace_alloc` pass. The `Point` above never escapes
`compute`, so the compiler reverts to value semantics under the hood
— heap-allocation syntax, zero runtime heap cost:

```bash
ZYNTAX_SRA_DUMP=1 zynml run examples/ref_class_sra.zynml
scalar_replace_alloc: examined=1 eliminated=1 frees=0 escapes=0
```

The pass currently handles same-basic-block lifetimes; instances
that survive across loop iterations or function boundaries keep
their malloc/free pair. Cross-block escape analysis is on the
roadmap.

## How this fits the bigger picture

Zyntax exposes memory management as a separate **opt-in axis** with
three planned modes:

1. **Speculative drop-site analysis (default, shipping today).**
   Compile-time `Free` insertion. No runtime cost. Same-block
   lifetimes handled in V1.
2. **Opt-in GC menu (planned).** A selectable collector — first
   entry is a generational copy-nursery + mark-sweep old gen, with
   mark-sweep and RC variants to follow. For programs where the
   speculative analysis is too conservative.
3. **Opt-in borrow/lifetime/move (planned).** Rust-style explicit
   ownership for safety-critical or FFI-heavy code where compile-time
   aliasing guarantees matter.

`@reference` is orthogonal to that axis. You can use `@reference`
classes under any of the three modes. The annotation decides "how do
field accesses lower"; the memory-management mode decides "when does
the storage go away".

## Tips and gotchas

* **Don't `@reference` your `Vec3` or `Point2D`.** Small POD math
  types benefit hugely from value semantics — the SIMD passes can
  pack them, the compiler can keep them in registers, and the cache
  footprint stays tight.
* **Do `@reference` your DOM-like trees, scene graphs, ECS
  components.** Anything with identity, shared mutation, or
  recursive structure.
* **Aliasing surprises are real.** If you write a loop like:
  ```zynml
  let a = bodies[0]
  let b = bodies[0]
  a.vx = 1.0
  return b.vx   // 1.0 under @reference, original value otherwise
  ```
  that's a feature — but make sure it's the feature you wanted.
* **Method calls work normally.** `body.normalize()` works for
  both value-type and `@reference` `Body`. The compiler routes the
  `self` parameter through the same lowering that field access uses,
  so the receiver is a struct value in one case and a pointer in
  the other.
* **`Array<Body>` and `Array<@reference Body>` are different
  layouts.** You can't pass one to a function expecting the other.
  Decide which one your domain wants once and stick with it.
