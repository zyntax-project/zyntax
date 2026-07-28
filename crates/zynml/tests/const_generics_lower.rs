//! Lowering test: const generic parameters reach the HIR signature.
//!
//! Parses a generic function declaring both an ordinary type parameter
//! and a const generic parameter (`def make<T, const N: usize>()`),
//! lowers it to HIR, and asserts the signature splits them correctly:
//! `T` lands in `type_params`, `N` lands in `const_params` carrying an
//! integer type. This closes the declaration half of const generics —
//! the monomorphizer already binds `const_params` against a use site's
//! const args (Milestone A wired the use-site `const_args`).

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::hir::HirType;
use zyntax_compiler::HirModule;
use zyntax_embed::ZyntaxRuntime;

fn lower(source: &str) -> HirModule {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<const_generics_lower>")
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

#[test]
fn const_param_reaches_hir_signature() {
    let module = lower(
        r#"
        def make<T, const N: usize>(): i64 {
            return 0
        }
        "#,
    );
    let f = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some("make"))
        .expect("function `make` should be lowered");

    assert_eq!(
        f.signature.const_params.len(),
        1,
        "N must be lowered as the sole const param"
    );
    let cp = &f.signature.const_params[0];
    assert_eq!(cp.name.resolve_global().as_deref(), Some("N"));
    assert!(
        matches!(cp.ty, HirType::U64),
        "const N: usize must lower to a 64-bit int, got {:?}",
        cp.ty
    );

    assert_eq!(
        f.signature.type_params.len(),
        1,
        "T must remain the sole ordinary type param, kept out of const_params"
    );
    assert_eq!(
        f.signature.type_params[0].name.resolve_global().as_deref(),
        Some("T")
    );
}

/// Assert the sole return type of `name` is a 4-element f32 array.
fn assert_returns_f32x4_array(module: &HirModule, name: &str) {
    let f = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(name))
        .unwrap_or_else(|| panic!("function `{name}` should be lowered"));
    match &f.signature.returns[0] {
        HirType::Array(elem, n) => {
            assert_eq!(*n, 4, "fixed size must lower to 4");
            assert!(
                matches!(**elem, HirType::F32),
                "element type must be f32, got {elem:?}"
            );
        }
        other => panic!("expected a sized HirType::Array, got {other:?}"),
    }
}

#[test]
fn slice_type_lowers_to_sized_array() {
    // `[f32; 4]` slice syntax → sized HIR array.
    let module = lower(
        r#"
        def f(): [f32; 4] {
        }
        "#,
    );
    assert_returns_f32x4_array(&module, "f");
}

#[test]
fn array_generic_with_const_lowers_to_sized_array() {
    // `Array<f32, 4>` generic spelling → same sized HIR array.
    let module = lower(
        r#"
        def g(): Array<f32, 4> {
        }
        "#,
    );
    assert_returns_f32x4_array(&module, "g");
}
