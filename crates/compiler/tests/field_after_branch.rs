#![cfg(feature = "cranelift-backend")]

//! A field read whose object branched before it settled.
//!
//! The object is a block that assigns its result in an `if` before
//! yielding it, so its value is the merge's phi. The field read belongs
//! in the block where that phi is defined, after the branch. The cases
//! lower from typed AST and run on Cranelift, LLVM and the HIR
//! interpreter.

use std::sync::Arc;

use zyntax_compiler::hir::{HirFunction, HirId, HirModule};
use zyntax_compiler::{CompilationConfig, compile_to_hir};
use zyntax_typed_ast::type_registry::{FieldDef, Mutability, NullabilityKind, TypeMetadata};
use zyntax_typed_ast::typed_ast::{
    TypedBinary, TypedBlock, TypedCall, TypedFieldAccess, TypedFieldInit, TypedIf, TypedLet,
    TypedParameter, TypedStructLiteral,
};
use zyntax_typed_ast::{
    BinaryOp, CallingConvention, InternedString, PrimitiveType, Span, Type, TypeId, TypeRegistry,
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

fn bool_ty() -> Type {
    Type::Primitive(PrimitiveType::Bool)
}

fn point_ty(id: TypeId) -> Type {
    Type::Named {
        id,
        type_args: vec![],
        const_args: vec![],
        variance: vec![],
        nullability: NullabilityKind::NonNull,
    }
}

fn point(id: TypeId, x: i128, y: i128) -> Expr {
    node(
        TypedExpression::Struct(TypedStructLiteral {
            name: name("Point"),
            fields: vec![
                TypedFieldInit {
                    name: name("x"),
                    value: Box::new(int(x)),
                },
                TypedFieldInit {
                    name: name("y"),
                    value: Box::new(int(y)),
                },
            ],
        }),
        point_ty(id),
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

fn call(f: &str, args: Vec<Expr>, ty: Type) -> Expr {
    node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(var(
                f,
                Type::Function {
                    params: vec![],
                    return_type: Box::new(ty.clone()),
                    is_varargs: false,
                    has_named_params: false,
                    has_default_params: false,
                    async_kind: Default::default(),
                    calling_convention: CallingConvention::Rust,
                    nullability: NullabilityKind::NonNull,
                },
            )),
            positional_args: args,
            named_args: vec![],
            type_args: vec![],
        }),
        ty,
    )
}

fn field_def(s: &str) -> FieldDef {
    FieldDef {
        name: name(s),
        ty: i64_ty(),
        visibility: Visibility::Public,
        mutability: Mutability::Mutable,
        is_static: false,
        span: span(),
        getter: None,
        setter: None,
        is_synthetic: false,
    }
}

/// `pick(c)` reads `.x` of `{ let p = a; if c > 0 { p = b }; p }`, with
/// `a.x` 10 and `b.x` 20; `run` returns `pick(1) * 100 + pick(0)`.
const EXPECTED: i64 = 2010;

fn module() -> HirModule {
    let mut registry = TypeRegistry::new();
    let id = registry.register_struct_type(
        name("Point"),
        vec![],
        vec![field_def("x"), field_def("y")],
        vec![],
        vec![],
        TypeMetadata::default(),
        span(),
    );
    let pt = || point_ty(id);
    let chosen = node(
        TypedExpression::Block(TypedBlock {
            statements: vec![
                let_("p", pt(), var("a", pt())),
                node(
                    TypedStatement::If(TypedIf {
                        condition: Box::new(bin(
                            BinaryOp::Gt,
                            var("c", i64_ty()),
                            int(0),
                            bool_ty(),
                        )),
                        then_block: TypedBlock {
                            statements: vec![expr_stmt(bin(
                                BinaryOp::Assign,
                                var("p", pt()),
                                var("b", pt()),
                                pt(),
                            ))],
                            span: span(),
                        },
                        else_block: None,
                        span: span(),
                    }),
                    unit_ty(),
                ),
                expr_stmt(var("p", pt())),
            ],
            span: span(),
        }),
        pt(),
    );
    let read = node(
        TypedExpression::Field(TypedFieldAccess {
            object: Box::new(chosen),
            field: name("x"),
        }),
        i64_ty(),
    );
    let declarations = vec![
        function(
            "pick",
            &[("c", i64_ty())],
            i64_ty(),
            vec![
                let_("a", pt(), point(id, 10, 1)),
                let_("b", pt(), point(id, 20, 2)),
                ret(read),
            ],
        ),
        function(
            "run",
            &[],
            i64_ty(),
            vec![ret(bin(
                BinaryOp::Add,
                bin(
                    BinaryOp::Mul,
                    call("pick", vec![int(1)], i64_ty()),
                    int(100),
                    i64_ty(),
                ),
                call("pick", vec![int(0)], i64_ty()),
                i64_ty(),
            ))],
        ),
    ];
    let mut program = TypedProgram {
        checker_interface: None,
        language: None,
        declarations,
        span: span(),
        source_files: vec![],
        type_registry: registry.clone(),
    };
    compile_to_hir(
        &mut program,
        Arc::new(registry),
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
fn field_after_branch_on_cranelift() {
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
fn field_after_branch_on_the_interpreter() {
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
fn field_after_branch_on_llvm() {
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
