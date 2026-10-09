//! A module lowered elsewhere joins a program that declares its
//! functions: the declarations take the module's ids, no body is lowered
//! for them, and a call reaches the body the module brought.

use std::sync::{Arc, Mutex};
use zyntax_compiler::bytecode::{
    Format, deserialize_module, deserialize_module_lazy, serialize_module,
};
use zyntax_compiler::hir::{HirCallable, HirInstruction, HirModule};
use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
use zyntax_typed_ast::{
    AstArena, BinaryOp, CallingConvention, InternedString, Mutability, PrimitiveType, Span, Type,
    TypeRegistry, TypedBinary, TypedBlock, TypedCall, TypedDeclaration, TypedExpression,
    TypedFunction, TypedLiteral, TypedParameter, TypedProgram, TypedStatement, Visibility,
    typed_node,
};

const SPAN: Span = Span::new(0, 0);

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

fn external_function(name: InternedString, module: Option<InternedString>) -> TypedDeclaration {
    let mut declaration = function(name, vec![], None, module);
    let TypedDeclaration::Function(function) = &mut declaration else {
        unreachable!();
    };
    function.is_external = true;
    declaration
}

fn returning_call(name: InternedString) -> TypedBlock {
    let call = typed_node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(typed_node(TypedExpression::Variable(name), i64_ty(), SPAN)),
            positional_args: vec![],
            named_args: vec![],
            type_args: vec![],
        }),
        i64_ty(),
        SPAN,
    );
    TypedBlock {
        statements: vec![typed_node(
            TypedStatement::Return(Some(Box::new(call))),
            Type::Primitive(PrimitiveType::Unit),
            SPAN,
        )],
        span: SPAN,
    }
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
    let lib = library();
    let twice_id = lib.functions.values().next().expect("twice").id;
    let lib = Arc::new(zyntax_compiler::bytecode::LazyModule::eager(lib));

    let mut program = client();
    let config = LoweringConfig {
        defer_prelowered_bodies: false,
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

#[test]
fn a_split_prelowered_body_can_stay_encoded_after_lowering() {
    let bytes = serialize_module(&library(), Format::Split).expect("serializes split");
    let lib = Arc::new(deserialize_module_lazy(bytes).expect("reads split directory"));
    let twice_id = lib.by_name("twice").expect("twice shell").id;
    let mut program = client();
    let unused = InternedString::new_global("unused");
    let unused_body = TypedBlock {
        statements: vec![typed_node(
            TypedStatement::Return(Some(Box::new(typed_node(
                TypedExpression::Literal(TypedLiteral::Integer(0)),
                i64_ty(),
                SPAN,
            )))),
            Type::Primitive(PrimitiveType::Unit),
            SPAN,
        )],
        span: SPAN,
    };
    program.declarations.push(typed_node(
        function(unused, vec![], Some(unused_body), None),
        Type::Primitive(PrimitiveType::Unit),
        SPAN,
    ));
    let mut arena = AstArena::new();
    let module_name = arena.intern_string("app");
    let mut ctx = LoweringContext::new(
        module_name,
        Arc::new(TypeRegistry::new()),
        Arc::new(Mutex::new(arena)),
        LoweringConfig {
            prelowered: vec![Arc::clone(&lib)],
            defer_prelowered_bodies: true,
            entry_names: vec!["main".into()],
            closed: true,
            ..LoweringConfig::default()
        },
    );

    let module = ctx.lower_program(&mut program).expect("lowers");
    assert_eq!(ctx.entered_functions(), Some(vec!["main".into()]));
    let twice = module
        .functions
        .get(&twice_id)
        .expect("twice shell is linked");
    assert!(twice.blocks.is_empty(), "the body remains encoded");
    assert!(
        module
            .functions
            .values()
            .all(|function| function.name != unused),
        "an unrelated program body remains unlowered"
    );
    let sources = ctx.deferred_prelowered();
    let source = sources.get(&twice_id).expect("body source is retained");
    assert!(
        source
            .function(twice_id)
            .is_some_and(|body| !body.blocks.is_empty()),
        "the retained source decodes the body on demand"
    );
}

#[test]
fn a_body_replacing_a_linked_extern_has_its_dependencies_adopted() {
    let hook = InternedString::new_global("hook");
    let dependency = InternedString::new_global("dependency");
    let library_entry = InternedString::new_global("library_entry");
    let main = InternedString::new_global("main");
    let library_name = InternedString::new_global("lib");

    let dependency_body = TypedBlock {
        statements: vec![typed_node(
            TypedStatement::Return(Some(Box::new(typed_node(
                TypedExpression::Literal(TypedLiteral::Integer(42)),
                i64_ty(),
                SPAN,
            )))),
            Type::Primitive(PrimitiveType::Unit),
            SPAN,
        )],
        span: SPAN,
    };
    let mut library_program = program(vec![
        external_function(hook, None),
        function(dependency, vec![], Some(dependency_body), None),
        function(library_entry, vec![], Some(returning_call(hook)), None),
    ]);
    let library = lower("lib", &mut library_program, LoweringConfig::default());
    let bytes = serialize_module(&library, Format::Split).expect("serializes split");
    let library = Arc::new(deserialize_module_lazy(bytes).expect("reads split directory"));
    let hook_id = library.by_name("hook").expect("hook shell").id;
    let dependency_id = library.by_name("dependency").expect("dependency shell").id;

    for defer_prelowered_bodies in [false, true] {
        // `main` first reaches the linked library entry. That entry reaches
        // the library's external hook shell; lowering then supplies the
        // program's hook body, whose call to `dependency` must trigger
        // another link scan. A deferred library body exposes that call only
        // through its split-module directory.
        let mut client = program(vec![
            function(library_entry, vec![], None, Some(library_name)),
            function(dependency, vec![], None, Some(library_name)),
            function(hook, vec![], Some(returning_call(dependency)), None),
            function(main, vec![], Some(returning_call(library_entry)), None),
        ]);
        let module = lower(
            "app",
            &mut client,
            LoweringConfig {
                prelowered: vec![Arc::clone(&library)],
                defer_prelowered_bodies,
                entry_names: vec!["main".into()],
                closed: true,
                ..LoweringConfig::default()
            },
        );

        assert!(
            module
                .functions
                .get(&hook_id)
                .is_some_and(|function| !function.blocks.is_empty()),
            "the program's hook body is built when a deferred library body calls it"
        );
        let dependency = module
            .functions
            .get(&dependency_id)
            .expect("the hook's linked dependency is adopted");
        assert!(
            defer_prelowered_bodies || !dependency.blocks.is_empty(),
            "an eager dependency carries its body"
        );
    }
}

#[cfg(feature = "cranelift-backend")]
#[test]
fn the_linked_program_runs() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    let lib = Arc::new(zyntax_compiler::bytecode::LazyModule::eager(library()));
    let mut program = client();
    let module = lower(
        "app",
        &mut program,
        LoweringConfig {
            defer_prelowered_bodies: false,
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

#[test]
fn private_library_literals_follow_reached_bodies() {
    use zyntax_compiler::hir::{
        HirConstant, HirGlobal, HirId, HirType, HirValueKind, Linkage, Visibility,
    };
    let mut source = library();
    let mut globals = Vec::new();
    for (text, linkage, initializer) in [
        (
            "reached_literal",
            Linkage::Private,
            HirConstant::String(InternedString::new_global("used")),
        ),
        (
            "unreached_literal",
            Linkage::Private,
            HirConstant::String(InternedString::new_global("unused")),
        ),
        (
            "host_literal",
            Linkage::External,
            HirConstant::String(InternedString::new_global("host")),
        ),
        ("private_state", Linkage::Private, HirConstant::I64(7)),
    ] {
        let id = HirId::new();
        source.globals.insert(
            id,
            HirGlobal {
                id,
                name: InternedString::new_global(text),
                ty: if matches!(initializer, HirConstant::I64(_)) {
                    HirType::I64
                } else {
                    HirType::Ptr(Box::new(HirType::I8))
                },
                initializer: Some(initializer),
                is_const: true,
                is_thread_local: false,
                linkage,
                visibility: Visibility::Default,
                error_flag: false,
            },
        );
        globals.push(id);
    }
    let twice = source.functions.values_mut().next().unwrap();
    let ptr = twice.create_value(
        HirType::Ptr(Box::new(HirType::I8)),
        HirValueKind::Global(globals[0]),
    );
    let byte = twice.create_value(HirType::I8, HirValueKind::Instruction);
    twice
        .blocks
        .get_mut(&twice.entry_block)
        .unwrap()
        .instructions
        .push(HirInstruction::Load {
            result: byte,
            ty: HirType::I8,
            ptr,
            align: 1,
            volatile: true,
        });
    let bytes = serialize_module(&source, Format::Split).unwrap();
    let lib = Arc::new(deserialize_module_lazy(bytes).unwrap());
    let ids: Vec<_> = [
        "reached_literal",
        "unreached_literal",
        "host_literal",
        "private_state",
    ]
    .iter()
    .map(|name| {
        lib.stripped()
            .globals
            .values()
            .find(|g| g.name.resolve_global().as_deref() == Some(name))
            .unwrap()
            .id
    })
    .collect();
    for (has_entry, deferred) in [(true, false), (false, false), (true, true)] {
        let module = lower(
            "literal_client",
            &mut client(),
            LoweringConfig {
                prelowered: vec![Arc::clone(&lib)],
                defer_prelowered_bodies: deferred,
                entry_names: if has_entry {
                    vec!["main".into()]
                } else {
                    vec![]
                },
                ..LoweringConfig::default()
            },
        );
        assert!(
            module.globals.contains_key(&ids[0]),
            "a reached library load needs its literal"
        );
        assert_eq!(
            module.globals.contains_key(&ids[1]),
            !has_entry || deferred,
            "unreached literals stay available when bodies have not been inspected"
        );
        assert!(
            module.globals.contains_key(&ids[2]),
            "host-visible literals remain exposed"
        );
        assert!(
            module.globals.contains_key(&ids[3]),
            "private runtime state remains available"
        );
        for function in module.functions.values() {
            for value in function.values.values() {
                if let HirValueKind::Global(id) = value.kind {
                    assert!(module.globals.contains_key(&id), "dangling global {id:?}");
                }
            }
        }
    }
}

#[test]
fn configured_error_flag_marks_imported_state() {
    use zyntax_compiler::hir::{HirConstant, HirGlobal, HirId, HirType, Linkage};
    use zyntax_typed_ast::TypedVariable;
    let pending = InternedString::new_global("pending");
    for recorded in [false, true] {
        let mut source = library();
        let id = HirId::new();
        source.globals.insert(
            id,
            HirGlobal {
                id,
                name: pending,
                ty: HirType::I64,
                initializer: Some(HirConstant::I64(0)),
                is_const: false,
                is_thread_local: false,
                linkage: Linkage::External,
                visibility: zyntax_compiler::hir::Visibility::Default,
                error_flag: recorded,
            },
        );
        let bytes = serialize_module(&source, Format::Split).unwrap();
        let library = Arc::new(deserialize_module_lazy(bytes).unwrap());
        for deferred in [false, true] {
            for exposed in [false, true] {
                for configured in [
                    None,
                    Some(pending),
                    Some(InternedString::new_global("other")),
                ] {
                    let mut program = client();
                    if exposed {
                        program.declarations.push(typed_node(
                            TypedDeclaration::Variable(TypedVariable {
                                name: pending,
                                ty: i64_ty(),
                                mutability: Mutability::Mutable,
                                initializer: None,
                                visibility: Visibility::Public,
                            }),
                            Type::Primitive(PrimitiveType::Unit),
                            SPAN,
                        ));
                    }
                    let module = lower(
                        "flag_client",
                        &mut program,
                        LoweringConfig {
                            prelowered: vec![Arc::clone(&library)],
                            defer_prelowered_bodies: deferred,
                            error_flag_global: configured,
                            ..LoweringConfig::default()
                        },
                    );
                    let global = module.globals.values().find(|g| g.name == pending).unwrap();
                    assert_eq!(
                        global.error_flag,
                        recorded || configured == Some(pending),
                        "recorded={recorded} deferred={deferred} exposed={exposed} configured={configured:?}"
                    );
                    assert_eq!(global.initializer, Some(HirConstant::I64(0)));
                }
            }
        }
    }
}
