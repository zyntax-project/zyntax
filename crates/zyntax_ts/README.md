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

The initial frontend implements annotated functions, inferred local variables,
primitive aliases, numeric and boolean operators, string concatenation and
equality, branches, loops, and fixed-layout objects with scalar fields.
Objects have reference identity: aliases observe the same mutations, borrowed
parameters do not acquire ownership, and fresh returned objects transfer their
lifetime to the caller. Property initializers run in source order even when
physical field order differs.

Functions currently require return annotations. Arrays, generic functions,
unions, closures, classes, multi-file linking and other unimplemented constructs
produce `Unsupported` diagnostics. These are implementation gaps, measured by
the conformance suite; they are not permanent language restrictions.

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
its oracle. `--only TEXT` filters cases; `--out PATH` saves the machine-readable
report. The runner never invokes Cargo.

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
types only. Native multi-file projects, relaxed-checking options and unsupported
test directives are explicitly reported as `SKIP`, with reasons; they are not
compatibility passes. Unimplemented syntax and representations in runnable
single-file tests report `UNSUPPORTED`. Selecting no cases or only skipped
cases does not produce a successful suite result. This is a starting harness
for growing compatibility, not a claim to pass the upstream suite.
