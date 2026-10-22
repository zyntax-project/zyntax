# Testing TypeScript

[Quick start](../../../crates/zyntax_ts/README.md) · [Compatibility](compatibility.md) · [Architecture](architecture.md)

Run commands from the repository root. The harness never invokes Cargo; build the
binary before running comparisons.

## Native tests and pinned conformance

```sh
cargo test -p zyntax_ts
cargo build -p zyntax_ts --bin zynode
python3 crates/zyntax_ts/conformance/run.py --zynode target/debug/zynode
```

Pins use **TypeScript 7.1.0-beta** with strict checking and Node.js execution. Running
against saved pins requires neither tool. Node executes only reference programs;
the native frontend does not consume their emitted JavaScript.

The default profiles are interpreter and early Cranelift promotion, both through
`TieredRuntime`. Profile selection alone does not prove native execution for a
short fixture; hot-call and Rust tier tests exercise actual promotion. To include
LLVM selection:

```sh
cargo build -p zyntax_ts --bin zynode --features llvm-backend
python3 crates/zyntax_ts/conformance/run.py \
  --profile interpreter --profile baseline --profile llvm
```

## Verify or refresh reference pins

```sh
npm ci --prefix crates/zyntax_ts/conformance --ignore-scripts
python3 crates/zyntax_ts/conformance/run.py --live
python3 crates/zyntax_ts/conformance/run.py --update
python3 -m unittest discover -s crates/zyntax_ts/conformance -p test_harness.py
```

`--typescript PATH` and `--node PATH` select reference tools; a compiler version
other than the exact `package.json` pin is rejected. The oracle uses TypeScript's
native synchronous API and in-memory emitter, disposing programs, resolvers and
the compiler client after each request. POSIX timeouts stop the whole child
process group. Pins include compiler options and source hashes.
Multi-file fixtures list the entry first in `files` and hash every file's relative
path and contents. Refresh pins after changing fixtures. Use `--only TEXT` to
filter cases and `--out PATH` to save a JSON report.

## Reading results

- Cases compare acceptance and diagnostic category. Executable cases also compare
  stdout and exit status. Exact TypeScript messages/codes are retained but are
  not required to match. Checked-policy rejections are tested separately.
- Unsupported cases fail compatibility unless `conformance/KNOWN_FAILURES` links
  them to a full git-bug ID. An unexpected pass (`XPASS`) fails the suite too.
  Crashes, timeouts and unrelated diagnostics are never accepted as known failures.
- Empty selections and selections containing only skipped cases do not pass.

## Upstream TypeScript cases

```sh
python3 crates/zyntax_ts/conformance/run.py \
  --upstream /path/to/TypeScript/tests/cases/conformance \
  --only expressions --limit 100 --profile baseline --out /tmp/ts-upstream.json
```

Upstream mode reads `@filename` sections and compiler-option matrices, then
compares type acceptance with the pinned checker. It does not compare execution.
Multi-root projects, relaxed-checking options and unsupported directives are
reported as `SKIP` with reasons. Missing native syntax/representations in runnable
single-file cases report `UNSUPPORTED`. Skips are not compatibility passes.

## Allocation audit

```sh
cargo test -p zyntax_ts --features allocation-audit --test allocations -- --ignored --nocapture --test-threads=1
```

These isolated diagnostic fixtures record frontend and project-cache Rust heap
requests. They exclude direct native-library, slab and libc allocations; requested
bytes are not RSS. Keep allocation diagnostics separate from execution benchmarks
and record measurements in git-bug.

For whole-process diagnostics, build `zynode` with `allocation-audit`; it reports
Rust requests to stderr by phase. `ZYNTAX_TRACE_LOWER_PHASES=1` also reports runtime
setup, project loading, primitive registration, compile-and-run, and shutdown
times. Cached project loading includes registration and cache preparation on a
miss. Compiler phases appear nested within these totals; do not add them twice.
Use an ordinary release build for wall-time comparisons, with no Cargo jobs
running during samples.
