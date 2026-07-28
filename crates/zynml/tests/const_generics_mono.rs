//! Monomorphization test: const-generic *struct* field-layout specialization.
//!
//! A generic `Buffer<T, const N: usize> { data: [T; N] }` used at two
//! distinct instantiations must lower its `data` field to differently
//! sized arrays: `Buffer<f32, 4>` → `Array(F32, 4)` and
//! `Buffer<f32, 8>` → `Array(F32, 8)`. This exercises use-site
//! substitution of both the type argument (`T` → `f32`) and the const
//! argument (`N` → 4 / 8) into the struct's field types at `convert_type`.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::hir::HirType;
use zyntax_compiler::HirModule;
use zyntax_embed::ZyntaxRuntime;

fn lower(source: &str) -> HirModule {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<const_generics_mono>")
        .expect("source should parse");
    let rt = ZyntaxRuntime::new().expect("runtime");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    rt.lower_typed_program(program, builtins)
        .expect("typed program should lower to HIR")
}

/// Assert `name`'s return type is a struct whose single field is a
/// `[f32; expected_size]` array.
fn assert_returns_buffer_of(module: &HirModule, name: &str, expected_size: u64) {
    let f = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(name))
        .unwrap_or_else(|| panic!("function `{name}` should be lowered"));

    let struct_ty = match &f.signature.returns[0] {
        HirType::Struct(s) => s,
        // Tolerate a `@reference` heap layout, though `Buffer` is a plain
        // value struct here.
        HirType::Ptr(inner) => match inner.as_ref() {
            HirType::Struct(s) => s,
            other => panic!("expected Ptr(Struct) for `{name}`, got Ptr({other:?})"),
        },
        other => panic!("expected HirType::Struct for `{name}`, got {other:?}"),
    };

    assert_eq!(
        struct_ty.fields.len(),
        1,
        "Buffer must have exactly one field (`data`)"
    );

    match &struct_ty.fields[0] {
        HirType::Array(elem, n) => {
            assert_eq!(
                *n, expected_size,
                "`data` field size must specialize to {expected_size} for `{name}`"
            );
            assert!(
                matches!(**elem, HirType::F32),
                "`data` element type must be f32, got {elem:?}"
            );
        }
        other => panic!("expected `data: HirType::Array` for `{name}`, got {other:?}"),
    }
}

#[test]
fn const_generic_struct_field_specializes_per_instantiation() {
    let module = lower(
        r#"
        struct Buffer<T, const N: usize> { data: [T; N] }
        def make4(): Buffer<f32, 4> { }
        def make8(): Buffer<f32, 8> { }
        "#,
    );

    // Distinct const args must yield distinct field-array sizes through the
    // same generic struct.
    assert_returns_buffer_of(&module, "make4", 4);
    assert_returns_buffer_of(&module, "make8", 8);
}
