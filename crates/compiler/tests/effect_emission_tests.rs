//! M2: end-to-end Tier 1 verification — TypedAST → HIR pipeline must
//! emit `HirInstruction::PerformEffect` for calls to effect operations
//! inside `@effect(E)`-annotated functions.
//!
//! Distinct from `effect_compilation_tests.rs` which constructs HIR by
//! hand. These tests start from TypedAST (the form the parser
//! produces) and run the full lowering pipeline, asserting on the
//! resulting HIR.

use std::sync::Arc;

use zyntax_compiler::{
    compile_to_hir, effect_analysis::analyze_effects,
    effect_handler_resolution::resolve_handlers, hir::HirInstruction, CompilationConfig,
};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{PrimitiveType, Type, Visibility};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::{InternedString, TypeRegistry};

fn span() -> Span {
    Span::new(0, 0)
}

fn ident_arg(name: &str) -> TypedAnnotationArg {
    TypedAnnotationArg::Positional(TypedAnnotationValue::Identifier(
        InternedString::new_global(name),
    ))
}

/// Build a `@effect(Log) def run() { info(42) }` program with an
/// `effect Log { def info(msg: i64): unit }` declaration. After
/// lowering, the call to `info` should be a `PerformEffect`.
fn build_effect_program() -> TypedProgram {
    // effect Log { def info(msg: i64): unit }
    let log_effect = TypedEffect {
        name: InternedString::new_global("Log"),
        type_params: vec![],
        operations: vec![TypedEffectOp {
            name: InternedString::new_global("info"),
            type_params: vec![],
            params: vec![TypedParameter {
                name: InternedString::new_global("msg"),
                ty: Type::Primitive(PrimitiveType::I64),
                ..Default::default()
            }],
            return_type: Type::Primitive(PrimitiveType::Unit),
            span: span(),
        }],
        span: span(),
    };

    // handler Console for Log { def info(self, msg: i64): unit { } }
    let handler = TypedEffectHandler {
        name: InternedString::new_global("Console"),
        effect_name: InternedString::new_global("Log"),
        type_params: vec![],
        fields: vec![],
        handlers: vec![TypedEffectHandlerImpl {
            op_name: InternedString::new_global("info"),
            return_type: Type::Primitive(PrimitiveType::Unit),
            params: vec![TypedParameter {
                name: InternedString::new_global("msg"),
                ty: Type::Primitive(PrimitiveType::I64),
                ..Default::default()
            }],
            body: Some(TypedBlock {
                statements: vec![],
                span: span(),
            }),
            ..Default::default()
        }],
        span: span(),
    };

    // The call: info(42)
    let info_call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("info")),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            )),
            positional_args: vec![TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(42)),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )],
            named_args: vec![],
            type_args: vec![],
        }),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    // @effect(Log) def run() { info(42) }
    let run_fn = TypedFunction {
        name: InternedString::new_global("run"),
        return_type: Type::Primitive(PrimitiveType::Unit),
        body: Some(TypedBlock {
            statements: vec![TypedNode::new(
                TypedStatement::Expression(Box::new(info_call)),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            )],
            span: span(),
        }),
        annotations: vec![TypedAnnotation {
            name: InternedString::new_global("effect"),
            args: vec![ident_arg("Log")],
            span: span(),
        }],
        visibility: Visibility::Public,
        ..Default::default()
    };

    TypedProgram {
        declarations: vec![
            TypedNode::new(
                TypedDeclaration::Effect(log_effect),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
            TypedNode::new(
                TypedDeclaration::EffectHandler(handler),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
            TypedNode::new(
                TypedDeclaration::Function(run_fn),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
        ],
        ..Default::default()
    }
}

#[test]
fn effect_op_call_lowers_to_perform_effect() {
    let mut program = build_effect_program();
    let registry = Arc::new(TypeRegistry::new());
    let module = compile_to_hir(&mut program, registry, CompilationConfig::default())
        .expect("compile_to_hir must succeed");

    // Find the `run` function in the lowered module.
    let run_fn = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some("run"))
        .expect("`run` should be in the lowered module");

    // run_fn's HIR signature should carry the effect annotation through.
    let effect_names: Vec<String> = run_fn
        .signature
        .effects
        .iter()
        .map(|s| s.resolve_global().unwrap_or_default())
        .collect();
    assert_eq!(
        effect_names,
        vec!["Log"],
        "func.effects from @effect(Log) should reach HirFunctionSignature.effects"
    );

    // Scan all blocks for HirInstruction::PerformEffect emission.
    let mut perform_count = 0;
    let mut plain_calls_to_info = 0;
    for block in run_fn.blocks.values() {
        for inst in &block.instructions {
            match inst {
                HirInstruction::PerformEffect { op_name, .. } => {
                    if op_name.resolve_global().as_deref() == Some("info") {
                        perform_count += 1;
                    }
                }
                HirInstruction::Call { callee, .. } => {
                    if let zyntax_compiler::hir::HirCallable::Function(_) = callee {
                        // We can't easily check the callee name without
                        // walking the symbols. The negative assertion
                        // below is enough: count plain Calls and ensure
                        // none of them shadow our PerformEffect emission.
                        plain_calls_to_info += 1;
                    }
                }
                _ => {}
            }
        }
    }

    assert_eq!(
        perform_count, 1,
        "expected exactly one PerformEffect for the `info(42)` call site"
    );
    // The op call should NOT have been lowered as a regular Call. (If
    // it had been, the algebraic_effects rewrite would have renamed
    // `info` → `info_fn`, which would resolve to a missing function.)
    assert_eq!(
        plain_calls_to_info, 0,
        "the info() call should have been lowered as PerformEffect, not Call"
    );
}

#[test]
fn emitted_perform_effect_passes_analysis_and_resolution() {
    // M2.2: end-to-end pipeline check. Build TypedAST → lower to HIR
    // → run analyze_effects and resolve_handlers against the resulting
    // module. The analysis must:
    //   * find the Log effect declaration
    //   * find the Console handler
    //   * register exactly 1 perform-site for `info`
    //   * mark Console as inlinable (simple, non-resumable)
    let mut program = build_effect_program();
    let registry = Arc::new(TypeRegistry::new());
    let module = compile_to_hir(&mut program, registry, CompilationConfig::default())
        .expect("compile_to_hir must succeed");

    // Effect analysis. The module has 1 effect declaration, 1 handler,
    // and 1 PerformEffect emitted by M1.3.
    let effect_analysis =
        analyze_effects(&module).expect("analyze_effects must succeed on emitted HIR");
    assert_eq!(
        effect_analysis.defined_effects.len(),
        1,
        "Log effect should appear in the analysis"
    );
    assert_eq!(
        effect_analysis.defined_handlers.len(),
        1,
        "Console handler should appear in the analysis"
    );

    // Handler resolution. Should find exactly 1 perform-site (the
    // `info(42)` call) and mark the Console handler as inlinable.
    let handler_resolution =
        resolve_handlers(&module).expect("resolve_handlers must succeed");
    assert_eq!(
        handler_resolution.stats.total_perform_sites, 1,
        "M1.3's PerformEffect should register as exactly 1 perform-site"
    );
    assert_eq!(
        handler_resolution.inlinable_handlers.len(),
        1,
        "Console (simple, non-resumable) should be inlinable"
    );
}

#[test]
fn fn_without_effect_annotation_does_not_emit_perform_effect() {
    // Same Effect declaration, but the function is NOT @effect-annotated.
    // The call to `info(42)` should NOT lower to PerformEffect — without
    // `@effect(Log)`, the SSA builder's effect_op_map is empty for this
    // function, so it falls through to normal Call lowering.
    let mut program = build_effect_program();
    // Strip the @effect annotation from `run`.
    for decl in &mut program.declarations {
        if let TypedDeclaration::Function(f) = &mut decl.node {
            if f.name.resolve_global().as_deref() == Some("run") {
                f.annotations.clear();
                f.effects.clear();
            }
        }
    }
    let registry = Arc::new(TypeRegistry::new());
    // We don't care about a clean lowering here — the absent handler
    // resolution may fail. Just ensure the SSA path didn't emit
    // PerformEffect.
    if let Ok(module) = compile_to_hir(&mut program, registry, CompilationConfig::default()) {
        if let Some(run_fn) = module
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some("run"))
        {
            let perform_count = run_fn
                .blocks
                .values()
                .flat_map(|b| b.instructions.iter())
                .filter(|i| matches!(i, HirInstruction::PerformEffect { .. }))
                .count();
            assert_eq!(
                perform_count, 0,
                "fn without @effect annotation should NOT have PerformEffect"
            );
        }
    }
}
