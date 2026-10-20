//! Generic bodies lower once; concrete instances share the native compiler path.

use std::sync::{Arc, Mutex};
use zyntax_compiler::hir::{HirCallable, HirInstruction, HirModule, HirType};
use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
use zyntax_typed_ast::{InternedString, Type, arena::AstArena, typed_ast::TypedDeclaration};

fn lower(source: &str) -> HirModule {
    let mut program = zyntax_ts::parse_program(source, "generics.ts", Default::default()).unwrap();
    for declaration in &program.declarations {
        if let TypedDeclaration::Function(f) = &declaration.node {
            if !f.type_params.is_empty() {
                assert!(matches!(f.params[0].ty, Type::TypeVar(_)));
            }
        }
    }
    let registry = Arc::new(program.type_registry.clone());
    let mut context = LoweringContext::new(
        InternedString::new_global("generics"),
        registry,
        Arc::new(Mutex::new(AstArena::new())),
        LoweringConfig::default(),
    );
    context.lower_program(&mut program).unwrap()
}

#[test]
fn generic_signatures_remain_symbolic_until_specialization() {
    let mut module =
        lower("function id<T>(x:T):T{return x;} const p={x:1,y:2}; const q=id(p); q.x=9;");
    let generic = module
        .functions
        .values()
        .find(|f| !f.signature.type_params.is_empty())
        .unwrap();
    assert!(
        matches!(generic.signature.params[0].ty, HirType::Opaque(_)),
        "{:?}",
        generic.signature
    );
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    for (id, f) in &module.functions {
        assert_eq!(*id, f.id);
        assert!(f.signature.type_params.is_empty());
    }
    let instance = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().unwrap().contains("__mono_"))
        .unwrap();
    assert!(
        matches!(instance.signature.params[0].ty, HirType::Ptr(_)),
        "{:?}",
        instance.signature
    );
    assert_eq!(
        instance.signature.returns,
        vec![instance.signature.params[0].ty.clone()]
    );
    assert!(instance.attributes.strict_fp);
    assert!(
        instance
            .blocks
            .values()
            .flat_map(|b| &b.instructions)
            .all(|i| !matches!(
                i,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(_),
                    ..
                }
            ))
    );
}

#[test]
fn nested_calls_reuse_each_concrete_body() {
    let mut module = lower(
        r#"
        function identity<T>(x: T): T { return x; }
        function through<U>(x: U): U { return identity<U>(x); }
        const p = { x: 3 }; const q = identity(p);
        identity(1); identity(2); through(3); through(4);
        through("hello"); identity(true);
    "#,
    );
    assert_eq!(
        module
            .functions
            .values()
            .filter(|f| !f.signature.type_params.is_empty())
            .count(),
        2
    );
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    let count = |prefix: &str| {
        module
            .functions
            .values()
            .filter(|f| f.name.resolve_global().unwrap().starts_with(prefix))
            .count()
    };
    assert_eq!(count("ts$fn$identity__mono_"), 4);
    assert_eq!(count("ts$fn$through__mono_"), 2);
    for (id, f) in &module.functions {
        assert_eq!(*id, f.id);
        assert!(f.signature.type_params.is_empty());
        for instruction in f.blocks.values().flat_map(|b| &b.instructions) {
            if let HirInstruction::Call {
                result,
                callee: HirCallable::Function(callee),
                type_args,
                ..
            } = instruction
            {
                assert!(type_args.is_empty());
                let target = &module.functions[callee];
                if let Some(result) = result {
                    assert_eq!(f.values[result].ty, target.signature.returns[0]);
                }
            }
        }
    }
    let functions = module.functions.len();
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    assert_eq!(module.functions.len(), functions);
}

#[test]
fn generic_operators_do_not_fall_back_to_pointer_arithmetic_or_comparison() {
    let arithmetic = "function add<T>(x: T, y: T): T { return x + y; }";
    let error =
        zyntax_ts::parse_program(arithmetic, "arithmetic.ts", Default::default()).unwrap_err();
    assert_eq!(error.kind, zyntax_ts::ErrorKind::Type);
    let equality = "function equal<T>(x: T, y: T): boolean { return x === y; } console.log(equal(\"a\", \"a\"));";
    let error = zyntax_ts::parse_program(equality, "equality.ts", Default::default()).unwrap_err();
    assert_eq!(error.kind, zyntax_ts::ErrorKind::Unsupported);
}
