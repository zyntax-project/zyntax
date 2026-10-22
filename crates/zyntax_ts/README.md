# Native TypeScript

`zyntax_ts` parses TypeScript with Oxc and lowers it to Zyntax's typed AST. The
`zynode` executable runs the program through the existing interpreter-first
tiered runtime. Native execution does not emit JavaScript or require Node.js.

```sh
cargo build --release -p zyntax_ts
target/release/zynode check example.ts
target/release/zynode run example.ts
target/release/zynode run --soundness checked example.ts
```

The initial frontend implements functions with typed parameters, inferred local
variables, primitive aliases, numeric and boolean operators, string concatenation and
equality, branches, loops, and fixed-layout objects with scalar fields.
Objects have reference identity: aliases observe the same mutations, borrowed
parameters do not acquire ownership, and fresh returned objects transfer their
lifetime to the caller. Property initializers run in source order even when
physical field order differs.

Scalar-record width views retain the concrete allocation and expose only the
declared properties. Local views can select any subset of fields without copying
storage. Calls and annotated returns accept a view when every required field has
the same native offset; these conversions use ordinary pointer casts and share
the existing drop analysis. Offset checks are cached by layout pair. Other layout
adaptations and assignments that change a binding's concrete layout still need
implementation. Object literals retain TypeScript's excess-property checks.

Return types are inferred for primitives, strings, fixed-layout objects and void
functions. Callees can appear later in the source: their signatures are resolved
from an iterative dependency worklist, and each body is checked and lowered once.
An annotated return can anchor a recursive call group. Direct self-calls used as whole return expressions
do not contribute an inference candidate: concrete base returns determine the
result. Without another candidate, fallthrough gives void and a function with
only recursive returns has type `never`. Native `never` calls carry no result,
and their returned paths end in `unreachable`. Arithmetic, conditional and mutual return cycles produce strict
type diagnostics with an annotation hint. Recursive function-value types,
inference through local aliases, unannotated recursive generics, general heterogeneous
return unions and differing readonly result views remain gaps.

Scalar `number | undefined` and `boolean | undefined` values use the compiler's
native tagged unions. Explicit and inferred optional returns, direct calls,
conditional expressions, nullish coalescing, equality, `typeof`, truthiness and
console output preserve `undefined` without heap boxing. Branches narrow lexical
bindings while retaining their declared storage type; assignments and loop writes
invalidate stale facts. `void` remains distinct from `undefined`. Optional callback
signatures and assertions that remove an unproven `undefined` alternative still
need implementation.

Unconstrained generic functions support inferred and explicit type arguments,
including forward calls, nested generic calls and recursion. Each body is checked
once, then Zyntax's shared HIR specializer caches concrete native instances.
Object arguments retain reference identity and their readonly views. Generic
constraints, defaults, const type parameters and operators whose operands remain
generic still need implementation.

Capture-free arrows and named function values use typed native function
references. Parameters can be annotated or receive contextual types from a
variable, argument, assignment or return. Expression and block bodies, aliases,
returned functions and higher-order calls support numbers, booleans, void
results and other scalar-signature functions. These references allocate no
environment and retain their signatures through interpreter, Cranelift and LLVM
calls. Lexical captures, function object identity/properties, parameter-count
adaptation and aggregate callback signatures need further implementation.

Dense `number[]`, `boolean[]`, `string[]` and `Array<T>` values use shared native
list storage. Reads and writes emit typed GEPs and loads/stores; length and push
use the compiler's inline list operations. Generic array helpers specialize once
per concrete element type, including native integer, float and string layouts in
the shared compiler. TypeScript `number` keeps its `f64` representation. Canonical
array loops use integer induction and convert to `number` where the source uses
the counter as a value.

Bounds proofs follow literals, aliases, returned array summaries, length guards
and canonical loops. Unproven reads that could yield `undefined`, sparse extension,
length mutation and other array methods need further implementation. String array
storage currently requires a proven static string lifetime; owned reference
containers and generic element mutation require additional lifetime analysis.
These checks preserve direct element access without introducing a collector.

Primitive `number | boolean | string` alternatives use shared native tagged unions.
Declared and inferred returns, forwarding, widening, `typeof` narrowing, mutation,
strict equality, truthiness and formatting preserve their payload types across
tiers without heap boxes. String construction requires a proven static lifetime;
owned/reference payloads and combinations with nullable values need further work.

General reference-bearing unions, capturing closures, classes and other
unimplemented constructs produce `Unsupported` diagnostics. These are implementation gaps, measured by
the conformance suite; they are not permanent language restrictions.

Native symbols are qualified by an interned module identity. `parse_program`
uses the supplied filename; resolvers can use `parse_module` with a canonical
`ModuleId` and a separate diagnostic filename. `zynode` canonicalizes its input
path. Hosts obtain callable names with `ModuleId::function_name`, rather than
constructing native names. Functions, lifted arrows, bindings and record layouts
retain separate identities when libraries are linked through shared imports or
snapshots.

Hosts check named imports with `parse_module_with_dependencies`, supplying
`ModuleDependency` entries resolved through shared compiled imports or
`Snapshot::module_interface`. Exported functions, interfaces, aliases and local
export renaming retain their checker facts in `TypedProgram::checker_interface`.
The interface contains interned signatures and referenced property views without
executable bodies. Readonly policy, generic specialization, borrowed parameters
and array result bounds/provenance survive source and snapshot imports. Named
imports within a source module combine their native signature requests. Explicit type-only and
unused named imports do not evaluate dependencies; bare imports retain side effects.
`load_project` and `zynode` resolve relative `.ts` files, extensionless imports,
directory indexes, and emitted `.js` specifiers backed by TypeScript source.
Each canonical file is checked once, including type-only dependencies. Runtime
bodies move into the shared compiled-import linker; type-only bodies are dropped.
Dependency initializers run once before their importers. Diagnostics identify
the file that failed, including errors in transitive dependencies.
Package resolution, default/namespace imports, `export ... from`, `export *` and
cyclic/live-binding semantics still need integration. Cycles report `Unsupported`.
Snapshot and compiled import schema versions have changed; rebuild older artifacts.

Diagnostics use Zyntax's shared Ariadne formatter, including source labels and
related declaration locations. `NO_COLOR` disables colors; `CLICOLOR_FORCE`
enables them. Use `--diagnostics=json` for structured diagnostic categories,
byte ranges and source filenames.

## Type and soundness policies

Strict static checking is enabled. Ordinary supported TypeScript assignment
rules are the default, including assignment of a readonly property view to a
writable view. `--soundness checked` adds a check against weakening readonly
views at initialization, assignment, arguments, returns and assertions. It is
an additional policy, not a claim that the frontend is a complete sound type
system. The setting does not authorize unsafe pointer reinterpretation.

For example, TypeScript accepts this program, and the default policy prints
`2`. Checked mode refuses the writable alias:

```ts
type Read = { readonly x: number };
type Write = { x: number };
const r: Read = { x: 1 };
const w: Write = r;
w.x = 2;
console.log(r.x);
```

Numbers use `f64`, including NaN, infinities and signed zero. Functions carry
the compiler's `strict_fp` annotation to preserve separate source arithmetic
rounding. Only operations whose semantics are implemented are accepted; a
type that has no native representation is not silently lowered to an integer.

## Memory

The frontend enables automatic release and selects `Collector::None`.
Zyntax's shared `drop_insert` analysis tracks derived aliases, returned
parameters, fresh owned returns, retention and liveness before inserting frees.
Recursive constructors and borrowed forwarding calls use the same ownership proof.
This follows the separation used by Rayzor's `insert_free`: a borrow that
retains nothing is distinct from a function that returns an alias, and fresh
storage is distinct from both. Static typing alone is not a lifetime proof.

The ownership tests inspect actual HIR release sites: the callee does not free
an object it returns, the caller releases it once through borrowed aliases,
early returns release locals on each path, and repeated analysis does not
insert duplicate frees. Escaping graphs, reference-valued fields and cycles
need further ownership support before this frontend accepts them. No collector
is enabled as a fallback.

## Conformance

Run the pinned acceptance/output suite without Node.js or TypeScript installed:

```sh
cargo test -p zyntax_ts
python3 crates/zyntax_ts/conformance/run.py --zynode target/debug/zynode
```

The reference pins are produced by **TypeScript 6.0.2** with strict checking
and Node.js execution. The reference alone transpiles test programs; that
code is never used by the native frontend. To verify or refresh pins:

```sh
npm ci --prefix crates/zyntax_ts/conformance --ignore-scripts
python3 crates/zyntax_ts/conformance/run.py --live
python3 crates/zyntax_ts/conformance/run.py --update
python3 -m unittest discover -s crates/zyntax_ts/conformance -p test_harness.py
```

`--typescript PATH` selects the pinned compiler package and `--node PATH`
selects its runtime. A different compiler version is rejected. Pins include
the source hash and compiler options, so editing a fixture requires refreshing
its oracle. Multi-file fixtures list the entry first in `files`; pins hash every
file's relative path and contents. `--only TEXT` filters cases; `--out PATH` saves
the machine-readable report. The runner never invokes Cargo.

The default profiles disable promotion for the interpreter control and request
early Cranelift promotion for the baseline control. They both use
`TieredRuntime`; selecting a profile is not proof that every short fixture
reaches native code. A hot-call fixture exercises promotion. Build with
`--features llvm-backend` and pass `--profile llvm` to include LLVM selection.

Each case compares acceptance and diagnostic category. Executable cases also
compare stdout and exit status. Exact TypeScript diagnostic codes/messages
remain in the oracle for future diagnostic-compatibility work, but are not yet
required to match. Additional checked-policy rejections are tested separately
from TypeScript compatibility.

An unsupported construct fails compatibility unless the case is listed in
`conformance/KNOWN_FAILURES` with a full git-bug issue ID. A listed case that
starts passing is an `XPASS` and fails the suite. Crashes, timeouts and unrelated
diagnostics do not count as expected unsupported failures. The JSON report
preserves all these statuses.

### Upstream TypeScript tests

Use a checkout of Microsoft's TypeScript conformance sources:

```sh
python3 crates/zyntax_ts/conformance/run.py \
  --upstream /path/to/TypeScript/tests/cases/conformance \
  --only expressions --limit 100 --profile baseline --out /tmp/ts-upstream.json
```

The driver reads `@filename` sections and expands compiler-option matrices.
It compares native acceptance with the pinned reference compiler rather than
guessing acceptance from an upstream filename. Upstream mode currently checks
types only. Upstream multi-root projects, relaxed-checking options and unsupported
test directives are explicitly reported as `SKIP`, with reasons; they are not
compatibility passes. Unimplemented syntax and representations in runnable
single-file tests report `UNSUPPORTED`. Selecting no cases or only skipped
cases does not produce a successful suite result. This is a starting harness
for growing compatibility, not a claim to pass the upstream suite.

## Frontend allocation audit

The frontend uses interned identity, borrowed source spellings, shared immutable
signatures and cached layouts. Use the compiler's Rust allocation counters to
profile parsing and typing independently of JIT compilation:

```sh
cargo test -p zyntax_ts --features allocation-audit --test allocations -- --ignored --nocapture
```

This reports cold and warm Rust heap requests for a fixed synthetic input.
It excludes direct native-library, slab and libc allocations, and is not an
execution benchmark.
