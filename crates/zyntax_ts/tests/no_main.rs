//! A script supplies top-level statements; function-only files need no entry.

use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::TieredRuntime;
use zyntax_typed_ast::TypedDeclaration;
static TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn empty_and_declaration_only_files_have_no_initializer_or_main() {
    let _test = TEST.lock().unwrap();
    let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    runtime.enter_only_through_entry_points();
    for source in [
        "",
        "type Point = { x:number };",
        "function main():number{return 42;}",
    ] {
        let program = zyntax_ts::parse_program(source, "library.ts", Default::default()).unwrap();
        assert!(!program.declarations.iter().any(|d| match &d.node {
            TypedDeclaration::Initializer(_) => true,
            TypedDeclaration::Function(f) => f.name.resolve_global().as_deref() == Some("ts$main"),
            _ => false,
        }));
        let hir = runtime.lower_to_hir(program.clone()).unwrap();
        assert!(hir.initializers.is_empty());
        assert!(hir.functions.values().all(|f| f.is_external));
        runtime.compile_typed_program(program).unwrap();
    }
    assert_eq!(runtime.statistics().baseline_functions, 0);
    runtime.stop();
}

#[test]
fn a_script_emits_a_block_and_preserves_strict_floating_point() {
    let _test = TEST.lock().unwrap();
    let program = zyntax_ts::parse_program(
        "const n=1; console.log(n+2);",
        "script.ts",
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        program
            .declarations
            .iter()
            .filter(|d| matches!(d.node, TypedDeclaration::Initializer(_)))
            .count(),
        1
    );
    let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    runtime.enter_only_through_entry_points();
    let hir = runtime.lower_to_hir(program).unwrap();
    assert_eq!(hir.initializers.len(), 1);
    let initializer = &hir.functions[&hir.initializers[0]];
    assert!(initializer.attributes.strict_fp);
    assert!(
        initializer
            .name
            .resolve_global()
            .unwrap()
            .starts_with("__zyntax_init$")
    );
    assert!(
        !hir.functions
            .values()
            .any(|f| f.name.resolve_global().as_deref() == Some("ts$main"))
    );
    runtime.stop();
}
