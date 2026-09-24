//! A closed program, one the host enters only through its entry points,
//! builds only the functions of its own that something built reaches: a
//! call, or a taken address. An open program builds every one, since a
//! host may call any of them by name.

use std::sync::{Arc, Mutex};
use zyntax_compiler::hir::HirModule;
use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
use zyntax_typed_ast::{
    AstArena, CallingConvention, InternedString, PrimitiveType, Span, Type, TypeRegistry,
    TypedBlock, TypedCall, TypedDeclaration, TypedExpression, TypedFunction, TypedLiteral,
    TypedProgram, TypedStatement, Visibility, typed_node,
};

const SPAN: Span = Span::new(0, 0);

fn i64_ty() -> Type {
    Type::Primitive(PrimitiveType::I64)
}

fn unit() -> Type {
    Type::Primitive(PrimitiveType::Unit)
}

fn name(s: &str) -> InternedString {
    InternedString::new_global(s)
}

fn call(callee: &str) -> zyntax_typed_ast::TypedNode<TypedExpression> {
    typed_node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(typed_node(
                TypedExpression::Variable(name(callee)),
                i64_ty(),
                SPAN,
            )),
            positional_args: vec![],
            named_args: vec![],
            type_args: vec![],
        }),
        i64_ty(),
        SPAN,
    )
}

fn stmt(
    expr: zyntax_typed_ast::TypedNode<TypedExpression>,
) -> zyntax_typed_ast::TypedNode<TypedStatement> {
    typed_node(TypedStatement::Expression(Box::new(expr)), unit(), SPAN)
}

fn ret(
    expr: zyntax_typed_ast::TypedNode<TypedExpression>,
) -> zyntax_typed_ast::TypedNode<TypedStatement> {
    typed_node(TypedStatement::Return(Some(Box::new(expr))), unit(), SPAN)
}

fn int(v: i128) -> zyntax_typed_ast::TypedNode<TypedExpression> {
    typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(v)),
        i64_ty(),
        SPAN,
    )
}

fn function(
    fn_name: &str,
    statements: Vec<zyntax_typed_ast::TypedNode<TypedStatement>>,
) -> TypedDeclaration {
    TypedDeclaration::Function(TypedFunction {
        name: name(fn_name),
        annotations: vec![],
        effects: vec![],
        with_handlers: vec![],
        type_params: vec![],
        params: vec![],
        return_type: i64_ty(),
        body: Some(TypedBlock {
            statements,
            span: SPAN,
        }),
        visibility: Visibility::Public,
        is_async: false,
        is_fiber: false,
        is_pure: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
        module: None,
    })
}

/// `main` calls `used`, which calls `deeper`, and takes the address of
/// `recorded` as a function value; `unreached` is named by nothing.
fn program() -> TypedProgram {
    let declarations = vec![
        function("unreached", vec![ret(int(1))]),
        function("deeper", vec![ret(int(2))]),
        function("used", vec![ret(call("deeper"))]),
        function("recorded", vec![ret(int(3))]),
        function(
            "main",
            vec![
                stmt(typed_node(
                    TypedExpression::Variable(name("recorded")),
                    i64_ty(),
                    SPAN,
                )),
                ret(call("used")),
            ],
        ),
    ];
    TypedProgram {
        language: None,
        declarations: declarations
            .into_iter()
            .map(|d| typed_node(d, unit(), SPAN))
            .collect(),
        span: SPAN,
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    }
}

fn lower(closed: bool, entry: &str) -> HirModule {
    let mut arena = AstArena::new();
    let module_name = arena.intern_string("app");
    let mut ctx = LoweringContext::new(
        module_name,
        Arc::new(TypeRegistry::new()),
        Arc::new(Mutex::new(arena)),
        LoweringConfig {
            entry_names: vec![entry.to_string()],
            closed,
            ..LoweringConfig::default()
        },
    );
    ctx.lower_program(&mut program()).expect("lowers")
}

fn built(module: &HirModule) -> Vec<String> {
    let mut names: Vec<String> = module
        .functions
        .values()
        .filter(|f| !f.blocks.is_empty())
        .filter_map(|f| f.name.resolve_global())
        .collect();
    names.sort();
    names
}

#[test]
fn unreached_program_functions_are_not_lowered() {
    assert_eq!(
        built(&lower(true, "main")),
        ["deeper", "main", "recorded", "used"],
        "a closed program builds what its entry calls, through any number \
         of calls, and what it takes the address of"
    );
}

#[test]
fn an_open_program_lowers_every_function() {
    assert_eq!(
        built(&lower(false, "main")),
        ["deeper", "main", "recorded", "unreached", "used"],
    );
}

#[test]
fn a_closed_program_without_its_entry_lowers_every_function() {
    assert_eq!(
        built(&lower(true, "start")),
        ["deeper", "main", "recorded", "unreached", "used"],
    );
}
