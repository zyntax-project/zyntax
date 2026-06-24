//! Verifies that `yield expr` inside a `fiber def` function body
//! lowers to `HirInstruction::FiberYield`. Regular (non-fiber)
//! functions still route `yield` through the compute-reduction path
//! and reject it outside of `compute { ... }`.

use std::sync::Arc;

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::hir::HirInstruction;
use zyntax_embed::{compile_to_hir, CompilationConfig};

fn lower(src: &str) -> zyntax_compiler::hir::HirModule {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let mut program = grammar
        .parse_with_filename(src, "<fiber_yield_lowering>")
        .expect("source should parse");
    let type_registry = Arc::new(program.type_registry.clone());
    compile_to_hir(&mut program, type_registry, CompilationConfig::default())
        .expect("HIR lowering should succeed")
}

fn function_by_name<'a>(
    module: &'a zyntax_compiler::hir::HirModule,
    name: &str,
) -> &'a zyntax_compiler::hir::HirFunction {
    for func in module.functions.values() {
        if let Some(s) = func.name.resolve_global() {
            if s == name {
                return func;
            }
        }
    }
    panic!("function `{name}` not found in lowered HIR module");
}

fn count_fiber_yields(func: &zyntax_compiler::hir::HirFunction) -> usize {
    func.blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .filter(|inst| matches!(inst, HirInstruction::FiberYield { .. }))
        .count()
}

#[test]
fn yield_in_fiber_def_emits_fiber_yield_op() {
    let module = lower(
        r#"
        fiber def gen() {
            yield 1
            yield 2
            yield 3
        }
        "#,
    );
    let gen = function_by_name(&module, "gen");
    assert!(
        gen.signature.is_fiber,
        "is_fiber flag must propagate to HIR signature"
    );
    assert_eq!(
        count_fiber_yields(gen),
        3,
        "expected three FiberYield ops, one per `yield` in the body"
    );
}

#[test]
fn yield_outside_fiber_def_does_not_emit_fiber_yield() {
    // A `yield` inside a regular `def` (not a `fiber def`, not
    // inside `compute { ... }`) is an error at lowering. The driver
    // skips the function rather than aborting the whole module, so
    // the assertion is "no FiberYield op survives in the module" —
    // which guards against the worse failure mode (regular yields
    // accidentally producing fiber ops).
    let module = lower(
        r#"
        def stray() {
            yield 42
        }
        def main(): i64 {
            return 0
        }
        "#,
    );
    let total_fiber_yields: usize = module.functions.values().map(count_fiber_yields).sum();
    assert_eq!(
        total_fiber_yields, 0,
        "regular `def` must not produce FiberYield ops"
    );
}
