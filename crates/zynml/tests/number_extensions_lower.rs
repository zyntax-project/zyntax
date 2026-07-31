//! Static extensions on the number types.
//!
//! `x.sqrt()` / `x.abs()` are methods every number carries, declared in
//! the prelude as static extensions (`impl f64 { … }`). Each must lower
//! to a direct intrinsic — one hardware instruction at the call site —
//! not a call into a stdlib function.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE};
use zyntax_compiler::hir::{HirCallable, HirFunction, HirInstruction, Intrinsic};
use zyntax_compiler::HirModule;
use zyntax_embed::ZyntaxRuntime;

fn lower(source: &str) -> HirModule {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<number_extensions_lower>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.add_import_resolver(Box::new(|m| {
        if m == "prelude" {
            Ok(Some(ZYNML_STDLIB_PRELUDE.to_string()))
        } else {
            Ok(None)
        }
    }));
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    rt.lower_typed_program(program, builtins)
        .expect("typed program should lower to HIR")
}

fn func<'m>(module: &'m HirModule, name: &str) -> &'m HirFunction {
    module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(name))
        .unwrap_or_else(|| panic!("function `{name}` should be lowered"))
}

fn insts(f: &HirFunction) -> impl Iterator<Item = &HirInstruction> {
    f.blocks.values().flat_map(|b| b.instructions.iter())
}

fn calls_intrinsic(f: &HirFunction, want: Intrinsic) -> bool {
    insts(f).any(|i| {
        matches!(
            i,
            HirInstruction::Call { callee: HirCallable::Intrinsic(got), .. } if *got == want
        )
    })
}

fn has_plain_call(f: &HirFunction) -> bool {
    insts(f).any(|i| {
        matches!(
            i,
            HirInstruction::Call {
                callee: HirCallable::Function(_) | HirCallable::Symbol(_),
                ..
            }
        )
    })
}

#[test]
fn number_math_methods_lower_to_intrinsics() {
    let module = lower(
        r#"
        import prelude
        def k_sqrt(x: f64): f64 { return x.sqrt() }
        def k_abs(x: f64): f64 { return x.abs() }
        "#,
    );

    let s = func(&module, "k_sqrt");
    assert!(
        calls_intrinsic(s, Intrinsic::Sqrt),
        "x.sqrt() should lower to the sqrt intrinsic"
    );
    assert!(
        !has_plain_call(s),
        "x.sqrt() must not lower to a stdlib function call"
    );

    let a = func(&module, "k_abs");
    assert!(
        calls_intrinsic(a, Intrinsic::Fabs),
        "x.abs() should lower to the abs intrinsic"
    );
    assert!(
        !has_plain_call(a),
        "x.abs() must not lower to a stdlib function call"
    );
}

/// The free-function spelling stays available (existing kernels call
/// `sqrt(d2)`), and lowers to the same intrinsic.
#[test]
fn free_function_form_still_lowers_to_intrinsic() {
    let module = lower(
        r#"
        import prelude
        def k(x: f64): f64 { return sqrt(x) }
        "#,
    );
    let f = func(&module, "k");
    assert!(
        calls_intrinsic(f, Intrinsic::Sqrt),
        "sqrt(x) should lower to the sqrt intrinsic"
    );
}
