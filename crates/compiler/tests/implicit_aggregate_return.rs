#![cfg(feature = "cranelift-backend")]

//! The return the SSA builder adds where a body can fall off its end.
//!
//! A function whose every path returns still ends in a block its
//! statements never reach, and the builder closes that block with a
//! return of a placeholder value. The placeholder must have the return
//! type's shape: a struct is held by address on Cranelift, and once the
//! function is inlined its caller reads the placeholder's fields like
//! any other struct's. The program lowers from typed AST, is inlined,
//! and runs on Cranelift, LLVM and the HIR interpreter.

use std::sync::Arc;

use zyntax_compiler::hir::{HirConstant, HirFunction, HirId, HirModule, HirType, HirValueKind};
use zyntax_compiler::{CompilationConfig, compile_to_hir};
use zyntax_typed_ast::type_registry::Mutability;
use zyntax_typed_ast::typed_ast::{
    TypedBinary, TypedBlock, TypedCall, TypedIf, TypedIndex, TypedLet, TypedParameter,
};
use zyntax_typed_ast::{
    BinaryOp, CallingConvention, InternedString, PrimitiveType, Span, Type, TypeRegistry,
    TypedDeclaration, TypedExpression, TypedFunction, TypedLiteral, TypedNode, TypedProgram,
    TypedStatement, Visibility,
};

type Expr = TypedNode<TypedExpression>;
type Stmt = TypedNode<TypedStatement>;

fn span() -> Span {
    Span::new(0, 10)
}

fn node<T>(inner: T, ty: Type) -> TypedNode<T> {
    TypedNode {
        node: inner,
        ty,
        span: span(),
    }
}

fn name(s: &str) -> InternedString {
    InternedString::new_global(s)
}

fn i64_ty() -> Type {
    Type::Primitive(PrimitiveType::I64)
}

fn f64_ty() -> Type {
    Type::Primitive(PrimitiveType::F64)
}

fn bool_ty() -> Type {
    Type::Primitive(PrimitiveType::Bool)
}

fn unit_ty() -> Type {
    Type::Primitive(PrimitiveType::Unit)
}

fn pair_ty() -> Type {
    Type::Tuple(vec![i64_ty(), f64_ty()])
}

fn int(v: i128) -> Expr {
    node(TypedExpression::Literal(TypedLiteral::Integer(v)), i64_ty())
}

fn float(v: f64) -> Expr {
    node(TypedExpression::Literal(TypedLiteral::Float(v)), f64_ty())
}

fn var(s: &str, ty: Type) -> Expr {
    node(TypedExpression::Variable(name(s)), ty)
}

fn bin(op: BinaryOp, l: Expr, r: Expr, ty: Type) -> Expr {
    node(
        TypedExpression::Binary(TypedBinary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }),
        ty,
    )
}

fn pair(a: Expr, b: Expr) -> Expr {
    node(TypedExpression::Tuple(vec![a, b]), pair_ty())
}

fn field(object: Expr, i: i128, ty: Type) -> Expr {
    node(
        TypedExpression::Index(TypedIndex {
            object: Box::new(object),
            index: Box::new(int(i)),
        }),
        ty,
    )
}

fn call(callee: &str, args: Vec<Expr>, ty: Type) -> Expr {
    node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(node(TypedExpression::Variable(name(callee)), unit_ty())),
            positional_args: args,
            named_args: vec![],
            type_args: vec![],
        }),
        ty,
    )
}

fn let_(s: &str, ty: Type, init: Expr) -> Stmt {
    node(
        TypedStatement::Let(TypedLet {
            name: name(s),
            ty,
            mutability: Mutability::Mutable,
            initializer: Some(Box::new(init)),
            span: span(),
        }),
        unit_ty(),
    )
}

fn ret(e: Expr) -> Stmt {
    node(TypedStatement::Return(Some(Box::new(e))), unit_ty())
}

fn block(statements: Vec<Stmt>) -> TypedBlock {
    TypedBlock {
        statements,
        span: span(),
    }
}

fn if_(cond: Expr, then: Vec<Stmt>) -> Stmt {
    node(
        TypedStatement::If(TypedIf {
            condition: Box::new(cond),
            then_block: block(then),
            else_block: None,
            span: span(),
        }),
        unit_ty(),
    )
}

fn function(fn_name: &str, return_type: Type, body: Vec<Stmt>) -> TypedNode<TypedDeclaration> {
    node(
        TypedDeclaration::Function(TypedFunction {
            name: name(fn_name),
            params: vec![TypedParameter {
                name: name("a"),
                ty: i64_ty(),
                mutability: Mutability::Mutable,
                ..Default::default()
            }],
            return_type,
            body: Some(block(body)),
            visibility: Visibility::Public,
            calling_convention: CallingConvention::Rust,
            ..Default::default()
        }),
        unit_ty(),
    )
}

/// `pick(a)` returns `(a, 2.5)` for a positive `a` and `(7, 0.5)`
/// otherwise, each from an `if` of its own: the block after the second
/// `if` is the tail no call reaches. `run(a)` is
/// `pick(a)[0] * 10 + pick(a)[1] * 2`.
fn module() -> HirModule {
    let declarations = vec![
        function(
            "pick",
            pair_ty(),
            vec![
                if_(
                    bin(BinaryOp::Gt, var("a", i64_ty()), int(0), bool_ty()),
                    vec![ret(pair(var("a", i64_ty()), float(2.5)))],
                ),
                if_(
                    bin(BinaryOp::Le, var("a", i64_ty()), int(0), bool_ty()),
                    vec![ret(pair(int(7), float(0.5)))],
                ),
            ],
        ),
        function(
            "run",
            i64_ty(),
            vec![
                let_(
                    "t",
                    pair_ty(),
                    call("pick", vec![var("a", i64_ty())], pair_ty()),
                ),
                let_(
                    "scaled",
                    f64_ty(),
                    bin(
                        BinaryOp::Mul,
                        field(var("t", pair_ty()), 1, f64_ty()),
                        float(2.0),
                        f64_ty(),
                    ),
                ),
                ret(bin(
                    BinaryOp::Add,
                    bin(
                        BinaryOp::Mul,
                        field(var("t", pair_ty()), 0, i64_ty()),
                        int(10),
                        i64_ty(),
                    ),
                    node(
                        TypedExpression::Cast(zyntax_typed_ast::typed_ast::TypedCast {
                            expr: Box::new(var("scaled", f64_ty())),
                            target_type: i64_ty(),
                        }),
                        i64_ty(),
                    ),
                    i64_ty(),
                )),
            ],
        ),
    ];
    let mut program = TypedProgram {
        language: None,
        declarations,
        span: span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    };
    let mut module = compile_to_hir(
        &mut program,
        Arc::new(TypeRegistry::new()),
        CompilationConfig {
            opt_level: 0,
            debug_info: false,
            enable_monomorphization: true,
            memory_strategy: None,
            ..Default::default()
        },
    )
    .expect("the program lowers");
    // Inlined, then split into fields, as the tiers do: `run` reads the
    // fields of each value `pick` returns, the placeholder's included.
    zyntax_compiler::inline::run_module(&mut module);
    zyntax_compiler::aggregate_scalarize::run_module(&mut module);
    module
}

fn find(module: &HirModule, fn_name: &str) -> HirId {
    module
        .functions
        .values()
        .find(|f: &&HirFunction| f.name.resolve_global().as_deref() == Some(fn_name))
        .unwrap_or_else(|| panic!("{fn_name} is in the module"))
        .id
}

/// `run(a)` for a positive and a non-positive `a`.
const PROBES: [(i64, i64); 2] = [(3, 35), (-1, 71)];

type Entry = extern "C" fn(i64) -> i64;

#[test]
fn the_placeholder_has_the_return_types_shape() {
    let module = module();
    let pick = &module.functions[&find(&module, "pick")];
    let mistyped: Vec<String> = module
        .functions
        .values()
        .flat_map(|f| f.values.values())
        .filter(|v| matches!(v.ty, HirType::Struct(_)))
        .filter_map(|v| match &v.kind {
            HirValueKind::Constant(c) if !matches!(c, HirConstant::Struct(_)) => {
                Some(format!("{:?}: {:?} = {c:?}", v.id, v.ty))
            }
            _ => None,
        })
        .collect();
    assert!(
        mistyped.is_empty(),
        "a struct-typed value holds a scalar constant:\n{}",
        mistyped.join("\n")
    );
    // The tail is there to be checked: `pick` returns from three blocks.
    let returns = pick
        .blocks
        .values()
        .filter(|b| {
            matches!(
                b.terminator,
                zyntax_compiler::hir::HirTerminator::Return { .. }
            )
        })
        .count();
    assert!(returns >= 3, "pick has {returns} returns");
    assert!(
        module.functions[&find(&module, "run")]
            .blocks
            .values()
            .flat_map(|b| b.instructions.iter())
            .all(|i| !matches!(
                i,
                zyntax_compiler::hir::HirInstruction::Call {
                    callee: zyntax_compiler::hir::HirCallable::Function(_),
                    ..
                }
            )),
        "pick is inlined into run"
    );
}

#[test]
fn inlined_on_cranelift() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let module = module();
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend
        .get_function_ptr(find(&module, "run"))
        .expect("run is compiled");
    let f: Entry = unsafe { std::mem::transmute(ptr) };
    for (a, want) in PROBES {
        assert_eq!(f(a), want, "run({a})");
    }
}

#[test]
fn inlined_on_the_interpreter() {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    use zyntax_compiler::value::ZyntaxValue;
    let module = module();
    for (a, want) in PROBES {
        let mut interp = HirInterpreter::new();
        let v = interp
            .call(&module, "run", vec![ZyntaxValue::Int(a)])
            .unwrap_or_else(|e| panic!("run({a}): {e:?}"));
        assert_eq!(value_to_i64(&v), Some(want), "run({a}): {v:?}");
    }
}

#[cfg(feature = "llvm-backend")]
#[test]
fn inlined_on_llvm() {
    use inkwell::context::Context;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;
    if zyntax_compiler::llvm_link::find_linker().is_err() {
        eprintln!("no system linker; skipping the LLVM leg");
        return;
    }
    let module = module();
    let context: &'static Context = Box::leak(Box::new(Context::create()));
    let mut backend = LLVMJitBackend::new(context).expect("backend");
    backend.compile_module(&module).expect("compile");
    let ptr = backend
        .get_function_pointer(find(&module, "run"))
        .expect("run is compiled");
    let f: Entry = unsafe { std::mem::transmute(ptr) };
    for (a, want) in PROBES {
        assert_eq!(f(a), want, "run({a})");
    }
}
