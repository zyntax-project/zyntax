//! Math functions with hardware equivalents reach HIR as compiler intrinsics, so every
//! tier lowers them to its own native instruction instead of a library call.

use std::sync::{Arc, Mutex};
use zyntax_compiler::hir::{HirCallable, HirInstruction, HirModule, Intrinsic};
use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
use zyntax_typed_ast::{InternedString, arena::AstArena};

fn lower(source: &str) -> HirModule {
    let mut program = zyntax_ts::parse_program(source, "math.ts", Default::default()).unwrap();
    let registry = Arc::new(program.type_registry.clone());
    let arena = Arc::new(Mutex::new(AstArena::new()));
    let mut context = LoweringContext::new(
        InternedString::new_global("math"),
        registry,
        arena,
        LoweringConfig::default(),
    );
    context.lower_program(&mut program).unwrap()
}

fn callees(module: &HirModule) -> Vec<HirCallable> {
    module
        .functions
        .values()
        .filter(|f| f.name.resolve_global().is_some_and(|n| n.contains("math")))
        .flat_map(|f| f.blocks.values())
        .flat_map(|b| &b.instructions)
        .filter_map(|i| match i {
            HirInstruction::Call { callee, .. } => Some(callee.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn sqrt_abs_and_floor_are_intrinsics_not_calls() {
    let module = lower(
        "function f(x: number): number { return Math.sqrt(x) + Math.abs(x) + Math.floor(x); }\n\
         console.log(f(2));",
    );
    let callees = callees(&module);
    for intrinsic in [Intrinsic::Sqrt, Intrinsic::Fabs, Intrinsic::Floor] {
        assert!(
            callees
                .iter()
                .any(|c| matches!(c, HirCallable::Intrinsic(i) if *i == intrinsic)),
            "{intrinsic:?} missing from {callees:?}"
        );
    }
    for callee in &callees {
        if let HirCallable::Symbol(name) = callee {
            assert!(
                !matches!(name.as_str(), "sqrt" | "abs" | "floor" | "fabs"),
                "library call {name}"
            );
        }
    }
}
