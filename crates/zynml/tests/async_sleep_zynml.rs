//! Phase I.4b/I.4c — end-to-end native execution of cooperative-async
//! ZynML source.
//!
//! Parses + compiles + drives:
//!
//! ```zynml
//! async def main(): i64 {
//!     await sleep(100)
//!     return 42
//! }
//! ```
//!
//! Asserts the Promise resolves with 42 and wall-clock measures
//! ≥ 90 ms (proves the cooperative parking machinery + native
//! synchronous bridge fired — if `sleep(100)` collapsed to no-op
//! the test would finish in ~0 ms).
//!
//! Execution test requires `krio-async-backend` feature so
//! `apply_krio_async_lowering` runs and Phase I.2's cooperative-await
//! lowering kicks in. The shape test below has no such requirement
//! and runs in every config.

use std::sync::Arc;
#[cfg(feature = "krio-async-backend")]
use std::time::Instant;

use zynml::{Grammar2, ZYNML_GRAMMAR};
#[cfg(feature = "krio-async-backend")]
use zyntax_embed::ZyntaxRuntime;
use zyntax_embed::{compile_to_hir, CompilationConfig};

#[cfg(feature = "krio-async-backend")]
#[test]
fn await_sleep_runs_to_42_with_wall_clock_yield() {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
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

    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.config_mut().builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    rt.compile_typed_program(program)
        .expect("typed-program → HIR → krio → install should succeed");

    let promise = rt
        .call_async("main", &[])
        .expect("call_async should return a Promise");
    let start = Instant::now();
    let result = promise.await_raw().expect("Promise should resolve cleanly");
    let elapsed = start.elapsed();

    assert_eq!(result.as_i64(), Some(42), "got {:?}", result);
    assert!(
        elapsed.as_millis() >= 90,
        "expected ≥ 90ms (cooperative parking + native sleep bridge); got {:?}",
        elapsed
    );
}

// Compile-only verification (no feature gate) — proves the SSA
// builtin-alias rewrite produces a Symbol callable even without
// the krio-async-backend feature. The execution counterpart above
// only runs with the feature.
#[test]
fn sleep_alias_lowers_to_symbol_callable() {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let mut program = grammar
        .parse_with_filename(
            r#"
            async def main(): i64 {
                await sleep(100)
                return 42
            }
            "#,
            "<async_sleep_zynml_shape>",
        )
        .expect("source should parse");

    let mut config = CompilationConfig::default();
    config.builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    let type_registry = Arc::new(program.type_registry.clone());
    let hir_module =
        compile_to_hir(&mut program, type_registry, config).expect("HIR lowering should succeed");

    let mut found = false;
    for func in hir_module.functions.values() {
        for block in func.blocks.values() {
            for inst in &block.instructions {
                if let zyntax_compiler::hir::HirInstruction::Call {
                    callee: zyntax_compiler::hir::HirCallable::Symbol(name),
                    ..
                } = inst
                {
                    if name == "__zyntax_async_set_timeout" {
                        found = true;
                    }
                }
            }
        }
    }
    assert!(
        found,
        "expected `Call(Symbol(\"__zyntax_async_set_timeout\"))` somewhere in the module"
    );
}
