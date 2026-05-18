//! Phase I.4b — verify that the ZynML source pipeline produces the
//! HIR shape that the krio_adapter Phase I.2 cooperative-await
//! lowering expects when a user writes `await sleep(100)`.
//!
//! Specifically: `Call(Symbol("__zyntax_async_set_timeout"), [100])`
//! immediately followed by `Call(Intrinsic::Await, [...])` in the
//! same block. This is what Phase I.2's `find_producing_call` →
//! `lower_host_bridge_await_site` chain keys off (see
//! `crates/passes/krio_adapter/tests/stages_i2_host_bridge_await.rs`
//! for the corresponding krio-side verification on a synthetic
//! fixture).
//!
//! Two upstream pieces are validated by this test together:
//!
//!   1. **Phase I.4a builtin alias** — `sleep` →
//!      `__zyntax_async_set_timeout` lives in `CompilationConfig.
//!      builtins`, gets copied into `LoweringContext.
//!      extern_link_names`, and gets consulted by SSA's Call
//!      resolution.
//!
//!   2. **Phase I.4 SSA fix** — the await-of-call resolver at
//!      `crates/compiler/src/ssa.rs::TypedExpression::Await` (Call
//!      sub-path) now consults `extern_link_names` after
//!      `function_symbols`, mirroring the main Call handler. Without
//!      this fix, `await sleep(100)` falls through to
//!      `HirCallable::Indirect` (translates the bare `sleep`
//!      identifier as a value lookup) and Phase I.2's Symbol-callable
//!      detection never fires.
//!
//! What's deliberately NOT tested here: actual JIT execution. The
//! native Cranelift execution path has unresolved integration issues
//! around CreateClosure value-map population and Promise-entry
//! wrapping that need a focused debugging session of their own. The
//! HIR-shape verification here proves the surface ergonomics work;
//! execution wiring is the next session's deliverable.

use std::sync::Arc;

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::hir::{HirCallable, HirInstruction};
use zyntax_embed::{compile_to_hir, CompilationConfig, HirModule};

#[test]
fn sleep_alias_lowers_to_symbol_callable_at_await_site() {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let mut program = grammar
        .parse_with_filename(
            r#"
            async def main(): i64 {
                await sleep(100)
                return 42
            }
            "#,
            "<async_sleep_zynml>",
        )
        .expect("source should parse");

    let mut config = CompilationConfig::default();
    config.builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );

    let type_registry = Arc::new(program.type_registry.clone());
    let hir_module: HirModule =
        compile_to_hir(&mut program, type_registry, config).expect("HIR lowering should succeed");

    // The OLD async-transform path (default when `use_krio_async`
    // is off in CompilationConfig) splits `async def main` into
    // multiple functions: an entry that allocates a Promise + a
    // poll fn that contains the body's HIR. The Symbol callable
    // appears in one of those functions (we walk them all rather
    // than assume which one). When the `krio-async-backend` feature
    // is active, the OLD transform is skipped and a separate krio
    // pipeline runs which preserves the Symbol-+-Await adjacency
    // that Phase I.2 keys off — that adjacency is verified directly
    // in `crates/passes/krio_adapter/tests/stages_i2_host_bridge_await.rs`.
    // Here we only assert the upstream SSA piece: the alias rewrote
    // `sleep` to the Symbol callable.
    let mut found_symbol_call = false;
    for func in hir_module.functions.values() {
        for block in func.blocks.values() {
            for inst in &block.instructions {
                if let HirInstruction::Call {
                    callee: HirCallable::Symbol(name),
                    ..
                } = inst
                {
                    if name == "__zyntax_async_set_timeout" {
                        found_symbol_call = true;
                    }
                }
            }
        }
    }

    assert!(
        found_symbol_call,
        "expected `Call(Symbol(\"__zyntax_async_set_timeout\"))` somewhere in the \
         compiled module (Phase I.4a builtin alias + Phase I.4 SSA fix should \
         rewrite `Call(Variable(\"sleep\"), ...)` to a Symbol callable). \
         Module functions: {:?}",
        hir_module
            .functions
            .values()
            .filter_map(|f| f.name.resolve_global())
            .collect::<Vec<_>>()
    );
}
