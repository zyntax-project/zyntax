#![cfg(feature = "cranelift-backend")]

//! Ordered compares of unsigned integers use the unsigned order on every
//! tier: `u64::MAX` is greater than 1, where a signed compare of the same
//! bits would say -1 < 1. The operands are parameters, so no fold decides
//! the compares before a backend sees them.

use std::sync::Arc;

use zyntax_compiler::hir::{HirFunction, HirId, HirModule};
use zyntax_compiler::{CompilationConfig, compile_to_hir};
use zyntax_typed_ast::type_registry::Mutability;
use zyntax_typed_ast::typed_ast::{TypedBinary, TypedBlock, TypedCast, TypedParameter};
use zyntax_typed_ast::{
    BinaryOp, CallingConvention, InternedString, PrimitiveType, Span, Type, TypeRegistry,
    TypedDeclaration, TypedExpression, TypedFunction, TypedLiteral, TypedNode, TypedProgram,
    TypedStatement, Visibility,
};

type Expr = TypedNode<TypedExpression>;

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

fn prim(p: PrimitiveType) -> Type {
    Type::Primitive(p)
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

fn var(s: &str) -> Expr {
    node(
        TypedExpression::Variable(InternedString::new_global(s)),
        prim(PrimitiveType::U64),
    )
}

fn int(v: i128) -> Expr {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v)),
        prim(PrimitiveType::I64),
    )
}

/// `weight` if `x op y` holds, else 0.
fn bit(op: BinaryOp, weight: i128) -> Expr {
    let holds = bin(op, var("x"), var("y"), prim(PrimitiveType::Bool));
    let as_int = node(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(holds),
            target_type: prim(PrimitiveType::I64),
        }),
        prim(PrimitiveType::I64),
    );
    bin(BinaryOp::Mul, as_int, int(weight), prim(PrimitiveType::I64))
}

/// `run(x, y)`: bit 1 for `x > y`, 2 for `x >= y`, 4 for `x < y`, 8 for
/// `x <= y`.
fn module() -> HirModule {
    let i64_ty = prim(PrimitiveType::I64);
    let sum = [
        bit(BinaryOp::Gt, 1),
        bit(BinaryOp::Ge, 2),
        bit(BinaryOp::Lt, 4),
        bit(BinaryOp::Le, 8),
    ]
    .into_iter()
    .reduce(|a, b| bin(BinaryOp::Add, a, b, prim(PrimitiveType::I64)))
    .expect("four terms");
    let body = vec![TypedNode::new(
        TypedStatement::Return(Some(Box::new(sum))),
        Type::Never,
        span(),
    )];
    let run = node(
        TypedDeclaration::Function(TypedFunction {
            name: InternedString::new_global("run"),
            params: ["x", "y"]
                .iter()
                .map(|p| TypedParameter {
                    name: InternedString::new_global(p),
                    ty: prim(PrimitiveType::U64),
                    mutability: Mutability::Immutable,
                    ..Default::default()
                })
                .collect(),
            return_type: i64_ty,
            body: Some(TypedBlock {
                statements: body,
                span: span(),
            }),
            visibility: Visibility::Public,
            calling_convention: CallingConvention::Rust,
            ..Default::default()
        }),
        prim(PrimitiveType::Unit),
    );
    let mut program = TypedProgram {
        language: None,
        declarations: vec![run],
        span: span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    };
    compile_to_hir(
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
    .expect("the program lowers")
}

fn find(module: &HirModule) -> HirId {
    module
        .functions
        .values()
        .find(|f: &&HirFunction| f.name.resolve_global().as_deref() == Some("run"))
        .expect("run is in the module")
        .id
}

/// `u64::MAX` against 1: greater and greater-or-equal hold.
const BIG_VS_ONE: i64 = 1 + 2;
/// 1 against `u64::MAX`: less and less-or-equal hold.
const ONE_VS_BIG: i64 = 4 + 8;

type Entry = extern "C" fn(u64, u64) -> i64;

#[test]
fn unsigned_compares_on_cranelift() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let module = module();
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend.get_function_ptr(find(&module)).expect("compiled");
    let f: Entry = unsafe { std::mem::transmute(ptr) };
    assert_eq!(f(u64::MAX, 1), BIG_VS_ONE);
    assert_eq!(f(1, u64::MAX), ONE_VS_BIG);
}

#[test]
fn unsigned_compares_on_the_interpreter() {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    use zyntax_compiler::value::ZyntaxValue;
    let module = module();
    let mut interp = HirInterpreter::new();
    for (x, y, expected) in [(u64::MAX, 1, BIG_VS_ONE), (1, u64::MAX, ONE_VS_BIG)] {
        let v = interp
            .call(
                &module,
                "run",
                vec![ZyntaxValue::UInt(x), ZyntaxValue::UInt(y)],
            )
            .unwrap_or_else(|e| panic!("run: {e:?}"));
        assert_eq!(value_to_i64(&v), Some(expected), "{x} vs {y}: {v:?}");
    }
}

#[cfg(feature = "llvm-backend")]
#[test]
fn unsigned_compares_on_llvm() {
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
        .get_function_pointer(find(&module))
        .expect("compiled");
    let f: Entry = unsafe { std::mem::transmute(ptr) };
    assert_eq!(f(u64::MAX, 1), BIG_VS_ONE);
    assert_eq!(f(1, u64::MAX), ONE_VS_BIG);
}
