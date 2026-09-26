#![cfg(feature = "cranelift-backend")]

//! A growable list's header as an integer and back.
//!
//! A frontend that keeps a list inside another value (a struct field, an
//! element of a list of addresses) stores the header's address as a
//! `usize` and casts it back where it reads. The cast must name the very
//! header the list variable holds: a push through the value cast back
//! grows the one list every holder sees. The cases lower from typed AST
//! and run on Cranelift, LLVM and the HIR interpreter.

use std::sync::Arc;

use zyntax_compiler::hir::{HirFunction, HirId, HirModule};
use zyntax_compiler::{CompilationConfig, compile_to_hir};
use zyntax_typed_ast::type_registry::{Mutability, NullabilityKind};
use zyntax_typed_ast::typed_ast::{
    TypedBinary, TypedBlock, TypedCast, TypedIndex, TypedLet, TypedMethodCall, TypedParameter,
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

fn usize_ty() -> Type {
    Type::Primitive(PrimitiveType::USize)
}

fn unit_ty() -> Type {
    Type::Primitive(PrimitiveType::Unit)
}

fn list_ty(elem: Type) -> Type {
    Type::Array {
        element_type: Box::new(elem),
        size: None,
        nullability: NullabilityKind::NonNull,
    }
}

fn pair_ty() -> Type {
    Type::Tuple(vec![usize_ty(), i64_ty()])
}

fn int(v: i128) -> Expr {
    node(TypedExpression::Literal(TypedLiteral::Integer(v)), i64_ty())
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

fn cast(e: Expr, ty: Type) -> Expr {
    node(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(e),
            target_type: ty.clone(),
        }),
        ty,
    )
}

fn index(object: Expr, i: Expr, ty: Type) -> Expr {
    node(
        TypedExpression::Index(TypedIndex {
            object: Box::new(object),
            index: Box::new(i),
        }),
        ty,
    )
}

fn method(receiver: Expr, m: &str, args: Vec<Expr>, ty: Type) -> Expr {
    node(
        TypedExpression::MethodCall(TypedMethodCall {
            receiver: Box::new(receiver),
            method: name(m),
            type_args: vec![],
            positional_args: args,
            named_args: vec![],
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

fn expr_stmt(e: Expr) -> Stmt {
    node(TypedStatement::Expression(Box::new(e)), unit_ty())
}

fn ret(e: Expr) -> Stmt {
    node(TypedStatement::Return(Some(Box::new(e))), unit_ty())
}

fn function(
    fn_name: &str,
    params: &[(&str, Type)],
    return_type: Type,
    body: Vec<Stmt>,
) -> TypedNode<TypedDeclaration> {
    node(
        TypedDeclaration::Function(TypedFunction {
            name: name(fn_name),
            params: params
                .iter()
                .map(|(p, ty)| TypedParameter {
                    name: name(p),
                    ty: ty.clone(),
                    mutability: Mutability::Mutable,
                    ..Default::default()
                })
                .collect(),
            return_type,
            body: Some(TypedBlock {
                statements: body,
                span: span(),
            }),
            visibility: Visibility::Public,
            calling_convention: CallingConvention::Rust,
            ..Default::default()
        }),
        unit_ty(),
    )
}

/// `xs` is [1, 2]. Its address goes into field 0 of a tuple and into
/// element 0 of a list of addresses, each is cast back, and 3 and 4 are
/// pushed through the two. Every holder then sees [1, 2, 3, 4]:
/// 1000 when both casts give back `xs`'s own address, plus len(xs) * 100,
/// the length through the field * 10, and xs[3].
const EXPECTED: i64 = 1444;

fn module() -> HirModule {
    let ints = || list_ty(i64_ty());
    let addrs = || list_ty(usize_ty());
    let declarations = vec![function(
        "run",
        &[],
        i64_ty(),
        vec![
            let_(
                "xs",
                ints(),
                node(TypedExpression::Array(vec![int(1), int(2)]), ints()),
            ),
            let_("a", usize_ty(), cast(var("xs", ints()), usize_ty())),
            let_(
                "t",
                pair_ty(),
                node(
                    TypedExpression::Tuple(vec![var("a", usize_ty()), int(7)]),
                    pair_ty(),
                ),
            ),
            let_(
                "ys",
                addrs(),
                node(TypedExpression::Array(vec![var("a", usize_ty())]), addrs()),
            ),
            let_(
                "via_field",
                ints(),
                cast(index(var("t", pair_ty()), int(0), usize_ty()), ints()),
            ),
            expr_stmt(method(
                var("via_field", ints()),
                "push",
                vec![int(3)],
                unit_ty(),
            )),
            let_(
                "via_list",
                ints(),
                cast(index(var("ys", addrs()), int(0), usize_ty()), ints()),
            ),
            expr_stmt(method(
                var("via_list", ints()),
                "push",
                vec![int(4)],
                unit_ty(),
            )),
            let_(
                "same",
                i64_ty(),
                bin(
                    BinaryOp::Mul,
                    cast(
                        bin(
                            BinaryOp::Eq,
                            cast(var("via_field", ints()), usize_ty()),
                            cast(var("via_list", ints()), usize_ty()),
                            Type::Primitive(PrimitiveType::Bool),
                        ),
                        i64_ty(),
                    ),
                    cast(
                        bin(
                            BinaryOp::Eq,
                            cast(var("via_list", ints()), usize_ty()),
                            var("a", usize_ty()),
                            Type::Primitive(PrimitiveType::Bool),
                        ),
                        i64_ty(),
                    ),
                    i64_ty(),
                ),
            ),
            ret(bin(
                BinaryOp::Add,
                bin(
                    BinaryOp::Add,
                    bin(BinaryOp::Mul, var("same", i64_ty()), int(1000), i64_ty()),
                    bin(
                        BinaryOp::Mul,
                        method(var("xs", ints()), "len", vec![], i64_ty()),
                        int(100),
                        i64_ty(),
                    ),
                    i64_ty(),
                ),
                bin(
                    BinaryOp::Add,
                    bin(
                        BinaryOp::Mul,
                        method(var("via_field", ints()), "len", vec![], i64_ty()),
                        int(10),
                        i64_ty(),
                    ),
                    index(var("xs", ints()), int(3), i64_ty()),
                    i64_ty(),
                ),
                i64_ty(),
            )),
        ],
    )];
    let mut program = TypedProgram {
        language: None,
        declarations,
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
            // The lists are never released; nothing here is about ownership.
            memory_strategy: None,
            ..Default::default()
        },
    )
    .expect("the program lowers")
}

fn find(module: &HirModule, fn_name: &str) -> HirId {
    module
        .functions
        .values()
        .find(|f: &&HirFunction| f.name.resolve_global().as_deref() == Some(fn_name))
        .unwrap_or_else(|| panic!("{fn_name} is in the module"))
        .id
}

type Entry = extern "C" fn() -> i64;

#[test]
fn list_header_casts_on_cranelift() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let module = module();
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend
        .get_function_ptr(find(&module, "run"))
        .expect("run is compiled");
    let f: Entry = unsafe { std::mem::transmute(ptr) };
    assert_eq!(f(), EXPECTED);
}

#[test]
fn list_header_casts_on_the_interpreter() {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    let module = module();
    let mut interp = HirInterpreter::new();
    let v = interp
        .call(&module, "run", vec![])
        .unwrap_or_else(|e| panic!("run: {e:?}"));
    assert_eq!(value_to_i64(&v), Some(EXPECTED), "{v:?}");
}

#[cfg(feature = "llvm-backend")]
#[test]
fn list_header_casts_on_llvm() {
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
    assert_eq!(f(), EXPECTED);
}
