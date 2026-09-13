//! A module lowered elsewhere joins a program that declares its
//! functions: the declarations take the module's ids, no body is lowered
//! for them, and a call reaches the body the module brought.

use std::sync::{Arc, Mutex};
use zyntax_compiler::bytecode::{deserialize_module, serialize_module, Format};
use zyntax_compiler::hir::{HirCallable, HirInstruction, HirModule};
use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
use zyntax_typed_ast::{
    typed_node, AstArena, BinaryOp, CallingConvention, InternedString, Mutability, PrimitiveType,
    Span, Type, TypeRegistry, TypedBinary, TypedBlock, TypedCall, TypedDeclaration,
    TypedExpression, TypedFunction, TypedLiteral, TypedParameter, TypedProgram, TypedStatement,
    Visibility,
};

const SPAN: Span = Span { start: 0, end: 0 };

fn i64_ty() -> Type {
    Type::Primitive(PrimitiveType::I64)
}

fn function(
    name: InternedString,
    params: Vec<TypedParameter>,
    body: Option<TypedBlock>,
    module: Option<InternedString>,
) -> TypedDeclaration {
    TypedDeclaration::Function(TypedFunction {
        name,
        annotations: vec![],
        effects: vec![],
        with_handlers: vec![],
        type_params: vec![],
        params,
        return_type: i64_ty(),
        body,
        visibility: Visibility::Public,
        is_async: false,
        is_fiber: false,
        is_pure: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
        module,
    })
}

fn program(declarations: Vec<TypedDeclaration>) -> TypedProgram {
    TypedProgram {
        language: None,
        declarations: declarations
            .into_iter()
            .map(|d| typed_node(d, Type::Primitive(PrimitiveType::Unit), SPAN))
            .collect(),
        span: SPAN,
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    }
}

fn lower(name: &str, program: &mut TypedProgram, config: LoweringConfig) -> HirModule {
    let mut arena = AstArena::new();
    let module_name = arena.intern_string(name);
    let mut ctx = LoweringContext::new(
        module_name,
        Arc::new(TypeRegistry::new()),
        Arc::new(Mutex::new(arena)),
        config,
    );
    ctx.lower_program(program).expect("lowers")
}

/// `fn twice(x: i64) -> i64 { return x * 2 }`, lowered on its own and
/// carried through bytes as a snapshot would carry it.
fn library() -> HirModule {
    let twice = InternedString::new_global("twice");
    let x = InternedString::new_global("x");
    let body = TypedBlock {
        statements: vec![typed_node(
            TypedStatement::Return(Some(Box::new(typed_node(
                TypedExpression::Binary(TypedBinary {
                    op: BinaryOp::Mul,
                    left: Box::new(typed_node(TypedExpression::Variable(x), i64_ty(), SPAN)),
                    right: Box::new(typed_node(
                        TypedExpression::Literal(TypedLiteral::Integer(2)),
                        i64_ty(),
                        SPAN,
                    )),
                }),
                i64_ty(),
                SPAN,
            )))),
            Type::Primitive(PrimitiveType::Unit),
            SPAN,
        )],
        span: SPAN,
    };
    let params = vec![TypedParameter::regular(
        x,
        i64_ty(),
        Mutability::Immutable,
        SPAN,
    )];
    let mut lib = program(vec![function(twice, params, Some(body), None)]);
    let module = lower("lib", &mut lib, LoweringConfig::default());
    let bytes = serialize_module(&module, Format::Postcard).expect("serializes");
    deserialize_module(&bytes).expect("deserializes")
}

/// A program declaring `twice` without a body and calling it from `main`.
fn client() -> TypedProgram {
    let twice = InternedString::new_global("twice");
    let x = InternedString::new_global("x");
    let main = InternedString::new_global("main");
    let call = typed_node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(typed_node(TypedExpression::Variable(twice), i64_ty(), SPAN)),
            positional_args: vec![typed_node(
                TypedExpression::Literal(TypedLiteral::Integer(21)),
                i64_ty(),
                SPAN,
            )],
            named_args: vec![],
            type_args: vec![],
        }),
        i64_ty(),
        SPAN,
    );
    let body = TypedBlock {
        statements: vec![typed_node(
            TypedStatement::Return(Some(Box::new(call))),
            Type::Primitive(PrimitiveType::Unit),
            SPAN,
        )],
        span: SPAN,
    };
    let params = vec![TypedParameter::regular(
        x,
        i64_ty(),
        Mutability::Immutable,
        SPAN,
    )];
    program(vec![
        function(twice, params, None, Some(InternedString::new_global("lib"))),
        function(main, vec![], Some(body), None),
    ])
}

#[test]
fn a_declaration_links_to_the_prelowered_body() {
    let lib = Arc::new(library());
    let twice_id = lib.functions.values().next().expect("twice").id;

    let mut program = client();
    let config = LoweringConfig {
        prelowered: vec![Arc::clone(&lib)],
        ..LoweringConfig::default()
    };
    let module = lower("app", &mut program, config);

    let names: Vec<String> = module
        .functions
        .values()
        .filter_map(|f| f.name.resolve_global())
        .collect();
    assert_eq!(names.len(), 2, "twice arrives once, main once: {names:?}");

    let twice = module
        .functions
        .get(&twice_id)
        .expect("the library's twice, under its id");
    assert!(!twice.blocks.is_empty(), "the body came with it");

    let main = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some("main"))
        .expect("main");
    let target = main
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .find_map(|inst| match inst {
            HirInstruction::Call {
                callee: HirCallable::Function(id),
                ..
            } => Some(*id),
            _ => None,
        })
        .expect("main calls twice directly");
    assert_eq!(target, twice_id, "the call lands on the library's id");
}

#[cfg(feature = "cranelift-backend")]
#[test]
fn the_linked_program_runs() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    let lib = Arc::new(library());
    let mut program = client();
    let module = lower(
        "app",
        &mut program,
        LoweringConfig {
            prelowered: vec![lib],
            ..LoweringConfig::default()
        },
    );
    let main_id = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some("main"))
        .expect("main")
        .id;

    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compiles");
    backend.finalize_definitions().expect("finalizes");
    let ptr = backend.get_function_ptr(main_id).expect("main is compiled");
    let main: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    assert_eq!(unsafe { main() }, 42);
}
