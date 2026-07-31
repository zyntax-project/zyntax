//! `impl f64 { ... }` — the receiver type of an impl block on a primitive.
//!
//! The parser resolves no type names, so an impl block records its target
//! as a bare name and every method's `self` arrives as
//! `Type::Unresolved("f64")`. Resolution used to look only at the type
//! registry, where primitives are not registered, so `self` fell through
//! to the `i64` catch-all — a float receiver typed as an integer, with the
//! return type beside it still correctly `f64`.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::hir::HirType;
use zyntax_embed::ZyntaxRuntime;

/// Lower `src` and return the HIR parameter and return types of the first
/// function whose name contains `needle`.
fn signature_of(src: &str, needle: &str) -> (Vec<HirType>, Vec<HirType>) {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    let program = grammar
        .parse_with_filename(src, "<primitive_impl>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
        _ => Ok(None),
    }));
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt
        .lower_typed_program(program, builtins)
        .expect("program should lower");

    let func = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().is_some_and(|n| n.contains(needle)))
        .unwrap_or_else(|| panic!("no lowered function matching `{needle}`"));

    (
        func.signature.params.iter().map(|p| p.ty.clone()).collect(),
        func.signature.returns.clone(),
    )
}

#[test]
fn a_float_impl_gives_self_a_float_type() {
    let (params, returns) = signature_of(
        r#"
        impl f64 {
            def doubled(self): f64 {
                return self * 2.0
            }
        }
        def main(): i64 {
            return 0
        }
        "#,
        "doubled",
    );
    assert_eq!(
        params,
        vec![HirType::F64],
        "`self` should be the f64 the impl targets"
    );
    assert_eq!(returns, vec![HirType::F64]);
}

#[test]
fn an_integer_impl_gives_self_an_integer_type() {
    let (params, returns) = signature_of(
        r#"
        impl i32 {
            def doubled(self): i32 {
                return self * 2
            }
        }
        def main(): i64 {
            return 0
        }
        "#,
        "doubled",
    );
    assert_eq!(params, vec![HirType::I32]);
    assert_eq!(returns, vec![HirType::I32]);
}
