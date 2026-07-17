//! The `@cooperative` function annotation (short alias `@coop`) is recognized
//! at lowering and recorded on `HirFunction::attributes.cooperative`. Both
//! spellings set the flag; an unannotated function leaves it false. No codegen
//! consumes the flag yet — this only pins down that the annotation is
//! registered so both spellings mean the same thing.

use std::sync::Arc;

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{compile_to_hir, CompilationConfig};

fn lower(src: &str) -> zyntax_compiler::hir::HirModule {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let mut program = grammar
        .parse_with_filename(src, "<coop_annotation_lowering>")
        .expect("source should parse");
    let type_registry = Arc::new(program.type_registry.clone());
    compile_to_hir(&mut program, type_registry, CompilationConfig::default())
        .expect("HIR lowering should succeed")
}

fn cooperative_flag(module: &zyntax_compiler::hir::HirModule, name: &str) -> bool {
    for func in module.functions.values() {
        if func.name.resolve_global().as_deref() == Some(name) {
            return func.attributes.cooperative;
        }
    }
    panic!("function `{name}` not found in lowered HIR module");
}

#[test]
fn cooperative_and_coop_are_equivalent() {
    let module = lower(
        r#"
        @cooperative
        async def long_spelling(): i64 {
            return 1
        }
        @coop
        async def short_spelling(): i64 {
            return 2
        }
        async def plain(): i64 {
            return 3
        }
        "#,
    );

    assert!(
        cooperative_flag(&module, "long_spelling"),
        "@cooperative must set attributes.cooperative"
    );
    assert!(
        cooperative_flag(&module, "short_spelling"),
        "@coop must set attributes.cooperative (alias of @cooperative)"
    );
    assert!(
        !cooperative_flag(&module, "plain"),
        "an unannotated function must leave attributes.cooperative false"
    );
}
