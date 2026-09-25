# Zyntax

> Compiler frontend infrastructure with tiered JIT compilation and native code generation

[![CI](https://github.com/darmie/zyntax/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/darmie/zyntax/actions/workflows/ci.yml)
[![ZynML Tests](https://img.shields.io/github/actions/workflow/status/darmie/zyntax/ci.yml?branch=main&label=zynml%20tests)](https://github.com/darmie/zyntax/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)

---

## Overview

Zyntax is a compiler frontend infrastructure written in Rust. A language frontend hands it
a typed AST; Zyntax lowers that to an SSA IR, optimises it once, starts running
it in an interpreter, and compiles each function to native code as it proves
hot.

- **Tiered execution**: HIR interpreter, Cranelift and LLVM, promoted per
  function, with on-stack replacement into running loops
- **One optimisation pipeline**: each body is optimised once in HIR and every
  tier compiles that same body
- **SIMD in the IR**: vector operations are HIR instructions both CPU tiers lower
  natively
- **Effects, fibers and async** as first-class HIR operations
- **Runtime plugins (ZRTL)**: native libraries linked into the executables
- **Grammar-defined languages**: a Zyn grammar maps syntax to typed AST, so a new
  language needs no Rust

---

## Languages

| Language | Binary | Crate |
|---|---|---|
| Python 3 | `zypy` | [crates/zyntax_python](crates/zyntax_python) |
| Lua 5.4 | `zylua` | [crates/zyntax_lua](crates/zyntax_lua) |
| ZynML, a language for ML pipelines | `zynml` | [crates/zynml](crates/zynml) |
| Your own, from a Zyn grammar | `zyntax` | [crates/zyn_peg](crates/zyn_peg) |

---

## Quick start

```bash
cargo build --release --features llvm-backend --bin zypy --bin zylua --bin zynml --bin zyntax

ZYPY_LLVM=1 ./target/release/zypy run program.py
ZYLUA_LLVM=1 ./target/release/zylua program.lua
./target/release/zynml run program.zynml
./target/release/zyntax repl --grammar examples/zpeg_test/calc.zyn
```

`ZYPY_LLVM` and `ZYLUA_LLVM` let hot code tier up to LLVM; without them the
interpreter and Cranelift run it. The LLVM build needs LLVM 21 (see
[Contributing](#contributing)); leave out `--features llvm-backend` for a build
without it.

New to Zyn grammars: **[The Zyn Book](https://github.com/darmie/zyntax/wiki)**.

---

## Architecture

A frontend produces a typed AST, either from a Zyn grammar's semantic actions or
from a hand-written frontend. The typed AST is lowered to HIR, an SSA form over
basic blocks, and HIR is optimised once. The tiered runtime interprets it first
and promotes functions to Cranelift, then to LLVM, as their counters cross the
thresholds.

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the layers and
**[docs/tiered-compilation.md](docs/tiered-compilation.md)** for the tier
ladder.

---

## Artifacts

| Format | What it is |
|---|---|
| `.zbc` | one HIR module in bytecode. [spec](docs/BYTECODE_FORMAT_SPEC.md) |
| `.zpack` | a package: HIR modules plus per-target ZRTL libraries. [guide](book/10-packaging-distribution.md) |
| `.zrtl` | a runtime plugin as a dynamic library. [guide](book/14-runtime-plugins.md) |
| HIR snapshot | a language's standard library, lowered and optimised at build time. [API](docs/SNAPSHOT_API.md) |

---

## Documentation

- **[The Zyn Book](https://github.com/darmie/zyntax/wiki)**: grammars, the CLI, the typed AST, packaging, embedding, plugins
- **[Architecture](docs/ARCHITECTURE.md)**: layers, IR, backends
- **[Zyn grammar spec](docs/ZYN_GRAMMAR_SPEC.md)**: syntax and semantic actions
- **[Embedding SDK](book/12-embedding-sdk.md)**: running Zyntax inside a Rust program
- **[Fibers, effects and async](docs/FIBER_EFFECT_ASYNC_COMPOSITION.md)**
- **[Bytecode format](docs/BYTECODE_FORMAT_SPEC.md)**

---

## Contributing

### LLVM

The LLVM tier needs **LLVM 21**, which `llvm-sys` finds through
`LLVM_SYS_211_PREFIX`:

```bash
# macOS
brew install llvm@21
export LLVM_SYS_211_PREFIX=$(brew --prefix llvm@21)

# Debian / Ubuntu
wget -qO llvm.sh https://apt.llvm.org/llvm.sh && sudo bash llvm.sh 21 all
export LLVM_SYS_211_PREFIX=/usr/lib/llvm-21
```

### Build and test

```bash
cargo build
cargo test -p zyntax_compiler
cargo test -p zyntax_lua --test conformance
cargo test -p zyntax_python --test conformance
```

Test the crates you change; the whole workspace takes a long time.

### Issues

Issues are tracked with [git-bug](https://github.com/git-bug/git-bug) and live
in the repository under `refs/bugs/*`. Run `git-bug pull` to fetch them, then
`git-bug bug --status open` to list open work; filter with `--label bug`,
`--label perf` or an area such as `--label area:lua`, and read one with
`git-bug bug show <id>`.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgments

The Cranelift project, LLVM, the Lua and Python projects, and the Rust community.

## Contact

[Issues](https://github.com/darmie/zyntax/issues) · [Discussions](https://github.com/darmie/zyntax/discussions)
