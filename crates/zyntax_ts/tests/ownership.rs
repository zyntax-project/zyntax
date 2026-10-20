//! Drop insertion follows aliases and ownership transfer with the collector off.

use std::sync::{Arc, Mutex};
use zyntax_compiler::hir::{HirCallable, HirInstruction, HirModule, Intrinsic};
use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
use zyntax_typed_ast::{InternedString, arena::AstArena};

fn lower(source: &str) -> HirModule {
    let mut program = zyntax_ts::parse_program(source, "ownership.ts", Default::default()).unwrap();
    let registry = Arc::new(program.type_registry.clone());
    let arena = Arc::new(Mutex::new(AstArena::new()));
    let mut context = LoweringContext::new(
        InternedString::new_global("ownership"),
        registry,
        arena,
        LoweringConfig::default(),
    );
    let mut module = context.lower_program(&mut program).unwrap();
    module.automatic_release = true;
    module
}

fn frees(module: &HirModule, name: &str) -> usize {
    module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(name))
        .unwrap()
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .filter(|i| {
            matches!(
                i,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Free),
                    ..
                }
            )
        })
        .count()
}

#[test]
fn returned_storage_is_freed_by_the_caller_and_borrowed_aliases_are_not_double_freed() {
    let mut module = lower(
        r#"
        type Point = { x: number };
        function make(x: number) { return { x: x }; }
        function alias(p: Point) { return p; }
        const p: Point = make(3);
        const q: Point = alias(p);
        q.x = 9;
        console.log(p.x, q.x);
    "#,
    );
    let first = zyntax_compiler::drop_insert::run_module(&mut module);
    assert!(first.frees_inserted > 0);
    assert_eq!(
        frees(&module, "ts$fn$make"),
        0,
        "return transfers the allocation"
    );
    assert_eq!(frees(&module, "ts$fn$alias"), 0, "the argument is borrowed");
    assert_eq!(
        frees(&module, zyntax_ts::ENTRY),
        1,
        "aliases share one owned allocation"
    );
    zyntax_compiler::drop_insert::run_module(&mut module);
    assert_eq!(
        frees(&module, zyntax_ts::ENTRY),
        1,
        "reanalysis is idempotent"
    );
}

#[test]
fn local_allocation_is_released_on_both_return_paths() {
    let mut module = lower(
        r#"
        type Point = { x: number };
        function value(flag: boolean): number {
            const p: Point = { x: 4 };
            if (flag) return p.x;
            return p.x + 1;
        }
        console.log(value(true), value(false));
    "#,
    );
    zyntax_compiler::drop_insert::run_module(&mut module);
    assert_eq!(frees(&module, "ts$fn$value"), 2);
}

#[test]
fn native_registration_does_not_enable_the_collector() {
    let mut runtime = zyntax_embed::TieredRuntime::new(Default::default()).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    assert!(!zyntax_compiler::collector::is_enabled());
    runtime.stop();
}
