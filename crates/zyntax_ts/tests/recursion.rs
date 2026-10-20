//! Recursive inference follows TypeScript while retaining concrete native signatures.

use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{TieredRuntime, ZyntaxValue};

const SOURCE: &str = r#"
    function down(n: number) { if (n <= 0) return 1.5; return down(n - 1); }
    function reverse(n: number) { if (n > 0) return (reverse(n - 1)); return 2.5; }
    function flag(n: number) { if (n <= 0) return true; return flag(n - 1); }
    function text(n: number) { if (n <= 0) return "done"; return text(n - 1); }
    function recurseVoid(n: number) { if (n > 0) recurseVoid(n - 1); }
    function forever(n: number) { return forever(n); }
    function delegated(n: number) { return forever(n); }
    function finite(n: number) { if (n < 0) return forever(n); return 6.5; }
    function partial(n: number) { if (n > 0) return partial(n - 1); }
    function pair(n: number) { if (n <= 0) return { x: 4 }; return pair(n - 1); }
    function score(n: number) { const p = pair(n); return down(n) + reverse(n) + p.x; }
"#;

#[test]
fn recursive_bodies_and_calls_have_their_settled_types() {
    use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedExpression, TypedStatement};
    use zyntax_typed_ast::{PrimitiveType, Type};
    let program = zyntax_ts::parse_program(SOURCE, "recursion.ts", Default::default()).unwrap();
    fn returns(statements: &[zyntax_typed_ast::TypedNode<TypedStatement>], expected: &Type) {
        for statement in statements {
            match &statement.node {
                TypedStatement::Return(Some(e)) if matches!(e.node, TypedExpression::Call(_)) => {
                    assert_eq!(&e.ty, expected);
                }
                TypedStatement::If(i) => {
                    returns(&i.then_block.statements, expected);
                    if let Some(b) = &i.else_block {
                        returns(&b.statements, expected);
                    }
                }
                _ => {}
            }
        }
    }
    for declaration in &program.declarations {
        if let TypedDeclaration::Function(f) = &declaration.node {
            let name = f.name.resolve_global().unwrap();
            let expected = match name.as_str() {
                "ts$m$recursion_2ets$fn$down"
                | "ts$m$recursion_2ets$fn$reverse"
                | "ts$m$recursion_2ets$fn$score" => Type::Primitive(PrimitiveType::F64),
                "ts$m$recursion_2ets$fn$flag" => Type::Primitive(PrimitiveType::Bool),
                "ts$m$recursion_2ets$fn$text" => Type::Primitive(PrimitiveType::String),
                "ts$m$recursion_2ets$fn$recurseVoid" | "ts$m$recursion_2ets$fn$partial" => {
                    Type::Primitive(PrimitiveType::Unit)
                }
                "ts$m$recursion_2ets$fn$forever" | "ts$m$recursion_2ets$fn$delegated" => {
                    Type::Never
                }
                _ => continue,
            };
            assert_eq!(f.return_type, expected, "{name}");
            returns(&f.body.as_ref().unwrap().statements, &expected);
        }
    }
}

fn check(runtime: &mut TieredRuntime) {
    assert_eq!(
        runtime
            .call_raw("ts$m$recursion_2ets$fn$finite", &[ZyntaxValue::Float(1.0)])
            .unwrap(),
        ZyntaxValue::Float(6.5)
    );
    assert_eq!(
        runtime
            .call_raw("ts$m$recursion_2ets$fn$score", &[ZyntaxValue::Float(5.0)])
            .unwrap(),
        ZyntaxValue::Float(8.0)
    );
    assert_eq!(
        runtime
            .call_raw("ts$m$recursion_2ets$fn$flag", &[ZyntaxValue::Float(5.0)])
            .unwrap(),
        ZyntaxValue::Bool(true)
    );
    runtime
        .call_raw(
            "ts$m$recursion_2ets$fn$recurseVoid",
            &[ZyntaxValue::Float(5.0)],
        )
        .unwrap();
    runtime
        .call_raw("ts$m$recursion_2ets$fn$partial", &[ZyntaxValue::Float(0.0)])
        .unwrap();
}

#[test]
fn never_returns_have_no_fabricated_native_value() {
    use std::sync::{Arc, Mutex};
    use zyntax_compiler::hir::{HirCallable, HirInstruction, HirTerminator};
    use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
    use zyntax_typed_ast::{InternedString, arena::AstArena};
    let mut program = zyntax_ts::parse_program(SOURCE, "recursion.ts", Default::default()).unwrap();
    let mut context = LoweringContext::new(
        InternedString::new_global("recursion"),
        Arc::new(program.type_registry.clone()),
        Arc::new(Mutex::new(AstArena::new())),
        LoweringConfig::default(),
    );
    let module = context.lower_program(&mut program).unwrap();
    for name in [
        "ts$m$recursion_2ets$fn$forever",
        "ts$m$recursion_2ets$fn$delegated",
    ] {
        let f = module
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some(name))
            .unwrap();
        assert!(f.signature.returns.is_empty());
        assert!(f.blocks.values().all(
            |b| !matches!(&b.terminator, HirTerminator::Return { values } if !values.is_empty())
        ));
        assert!(
            f.blocks
                .values()
                .any(|b| matches!(b.terminator, HirTerminator::Unreachable))
        );
        assert!(
            f.blocks
                .values()
                .flat_map(|b| &b.instructions)
                .any(|i| matches!(
                    i,
                    HirInstruction::Call {
                        result: None,
                        callee: HirCallable::Function(_),
                        ..
                    }
                ))
        );
    }
}

#[test]
fn recursive_function_types_and_value_cycles_remain_explicit_gaps() {
    for source in [
        "function f(){return f;}",
        "function f(){return ()=>f();}",
        "function f(n:number){const g=f;return 7;}",
        "function f<T>(n:T){return f<T>(n);}",
        "function f(n:number){return f(f(n));}",
        "function f(n:number){return void f(n);}",
    ] {
        let error = zyntax_ts::parse_program(source, "pending.ts", Default::default()).unwrap_err();
        assert_eq!(
            error.kind,
            zyntax_ts::ErrorKind::Unsupported,
            "{source}: {error}"
        );
    }
}

#[test]
fn recursive_functions_and_never_signatures_compile_on_each_tier() {
    let program = zyntax_ts::parse_program(SOURCE, "recursion.ts", Default::default()).unwrap();
    let config = TieredConfig {
        baseline_threshold: u32::MAX,
        enable_osr: false,
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    let names: Vec<_> = runtime
        .compile_typed_program(program)
        .unwrap()
        .into_iter()
        .filter(|name| name.starts_with("ts$m$recursion_2ets$fn$"))
        .collect();
    check(&mut runtime);
    assert_eq!(runtime.statistics().baseline_functions, 0);
    for name in &names {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        for name in &names {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while runtime.statistics().optimized_functions < names.len() {
            assert!(
                std::time::Instant::now() < deadline,
                "LLVM recursion did not install"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        check(&mut runtime);
    }
    assert!(!zyntax_compiler::collector::is_enabled());
    runtime.stop();
}

#[test]
fn strict_cycles_report_a_type_error_at_a_function_name() {
    for source in [
        "function f(n:number){if(n<1)return 1;return n*f(n-1);}",
        "function f(n:number){return n<1?1:f(n-1);}",
        "function f(n:number){return g(n);} function g(n:number){return f(n);}",
        "function f(n:number){return (f)(n);}",
        "function identity(n:number):number{return n;} function f(n:number){return identity(f(n));}",
    ] {
        let error = zyntax_ts::parse_program(source, "cycle.ts", Default::default()).unwrap_err();
        assert_eq!(error.kind, zyntax_ts::ErrorKind::Type, "{source}: {error}");
        assert!(["f", "g"].contains(&&source[error.start..error.end]));
        assert!(
            error
                .render("cycle.ts", source, false)
                .contains("add a return type annotation")
        );
    }
}
