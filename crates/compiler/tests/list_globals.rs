#![cfg(feature = "cranelift-backend")]

//! A module global of growable list type.
//!
//! The global holds the address of the list's header, so the function
//! that assigns the list, the one that pushes onto it and the one that
//! reads it back all name one list. The cases lower from typed AST and
//! run on Cranelift, LLVM and the HIR interpreter.

use std::sync::Arc;

use zyntax_compiler::hir::{HirFunction, HirId, HirModule};
use zyntax_compiler::{CompilationConfig, compile_to_hir};
use zyntax_typed_ast::type_registry::{Mutability, NullabilityKind};
use zyntax_typed_ast::typed_ast::{
    TypedBinary, TypedBlock, TypedCall, TypedIndex, TypedMethodCall, TypedParameter, TypedVariable,
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

fn assign(s: &str, value: Expr) -> Stmt {
    let ty = value.ty.clone();
    expr_stmt(bin(BinaryOp::Assign, var(s, ty.clone()), value, ty))
}

/// `data` is assigned [], `run` pushes 1 and `add` pushes 2, so `run`
/// reads back len 2, then 1 and 2: 2 * 100 + 1 * 10 + 2.
const EXPECTED: i64 = 212;

fn module() -> HirModule {
    let ints = || list_ty(i64_ty());
    let declarations = vec![
        node(
            TypedDeclaration::Variable(TypedVariable {
                name: name("data"),
                ty: ints(),
                mutability: Mutability::Mutable,
                initializer: None,
                visibility: Visibility::Public,
            }),
            unit_ty(),
        ),
        function(
            "add",
            &[("x", i64_ty())],
            unit_ty(),
            vec![expr_stmt(method(
                var("data", ints()),
                "push",
                vec![var("x", i64_ty())],
                unit_ty(),
            ))],
        ),
        function(
            "run",
            &[],
            i64_ty(),
            vec![
                assign("data", node(TypedExpression::Array(vec![]), ints())),
                expr_stmt(method(var("data", ints()), "push", vec![int(1)], unit_ty())),
                expr_stmt(node(
                    TypedExpression::Call(TypedCall {
                        callee: Box::new(var(
                            "add",
                            Type::Function {
                                params: vec![],
                                return_type: Box::new(unit_ty()),
                                is_varargs: false,
                                has_named_params: false,
                                has_default_params: false,
                                async_kind: Default::default(),
                                calling_convention: CallingConvention::Rust,
                                nullability: NullabilityKind::NonNull,
                            },
                        )),
                        positional_args: vec![int(2)],
                        named_args: vec![],
                        type_args: vec![],
                    }),
                    unit_ty(),
                )),
                ret(bin(
                    BinaryOp::Add,
                    bin(
                        BinaryOp::Mul,
                        method(var("data", ints()), "len", vec![], i64_ty()),
                        int(100),
                        i64_ty(),
                    ),
                    bin(
                        BinaryOp::Add,
                        bin(
                            BinaryOp::Mul,
                            index(var("data", ints()), int(0), i64_ty()),
                            int(10),
                            i64_ty(),
                        ),
                        index(var("data", ints()), int(1), i64_ty()),
                        i64_ty(),
                    ),
                    i64_ty(),
                )),
            ],
        ),
    ];
    let mut program = TypedProgram {
        checker_interface: None,
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
            // The list is never released; nothing here is about ownership.
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
fn list_global_on_cranelift() {
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
fn list_global_on_the_interpreter() {
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
fn list_global_on_llvm() {
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
