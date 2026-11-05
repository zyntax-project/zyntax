use zyntax_compiler::hir::HirModule;
use zyntax_embed::{Collector, TieredRuntime};
use zyntax_typed_ast::InternedString;

#[test]
fn collector_policy_requires_a_collector_and_precedes_code_loading() {
    let mut runtime = TieredRuntime::development().unwrap();
    assert!(
        runtime
            .set_gc_managed_drops(true)
            .unwrap_err()
            .to_string()
            .contains("collector")
    );
    runtime.set_collector(Collector::MarkSweep);
    runtime.set_gc_managed_drops(true).unwrap();
    runtime.set_gc_managed_drops(false).unwrap();
    runtime.set_gc_managed_drops(true).unwrap();
    runtime
        .compile_module(HirModule::new(InternedString::new_global("gc_policy")))
        .unwrap();
    assert!(
        runtime
            .set_gc_managed_drops(false)
            .unwrap_err()
            .to_string()
            .contains("before loading")
    );
}
