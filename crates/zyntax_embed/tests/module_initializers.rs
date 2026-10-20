//! Script blocks and libraries do not require a source-level entry function.

use std::sync::{Arc, Mutex};
use zyntax_compiler::{
    hir::*,
    tiered_backend::{OptimizationTier, TieredConfig},
    zrtl::*,
};
use zyntax_embed::{CompiledImport, Snapshot, SnapshotBuilder, TieredRuntime, ZyntaxRuntime};
use zyntax_typed_ast::typed_ast::TypedInitializer;
use zyntax_typed_ast::*;

static TEST: Mutex<()> = Mutex::new(());
static EVENTS: Mutex<Vec<i64>> = Mutex::new(Vec::new());
const SPAN: Span = Span::new(0, 0);
const RECORD: &str = "record_initialization";

extern "C" fn record(value: i64) {
    EVENTS.lock().unwrap().push(value);
}

fn signature() -> ZrtlSymbolSig {
    let mut params = [TypeTag::VOID; MAX_PARAMS];
    params[0] = TypeTag::new(
        TypeCategory::Int,
        PrimitiveSize::Bits64 as u16,
        TypeFlags::NONE,
    );
    ZrtlSymbolSig {
        param_count: 1,
        flags: ZrtlSigFlags::EFFECTFUL,
        return_type: TypeTag::VOID,
        params,
    }
}

fn block(values: &[i64]) -> TypedBlock {
    TypedBlock {
        statements: values
            .iter()
            .map(|value| {
                typed_node(
                    TypedStatement::Expression(Box::new(typed_node(
                        TypedExpression::Call(TypedCall {
                            callee: Box::new(typed_node(
                                TypedExpression::Variable(InternedString::new_global(RECORD)),
                                Type::Primitive(PrimitiveType::Unit),
                                SPAN,
                            )),
                            positional_args: vec![typed_node(
                                TypedExpression::Literal(TypedLiteral::Integer((*value).into())),
                                Type::Primitive(PrimitiveType::I64),
                                SPAN,
                            )],
                            named_args: vec![],
                            type_args: vec![],
                        }),
                        Type::Primitive(PrimitiveType::Unit),
                        SPAN,
                    ))),
                    Type::Primitive(PrimitiveType::Unit),
                    SPAN,
                )
            })
            .collect(),
        span: SPAN,
    }
}

fn import(name: &str) -> TypedNode<TypedDeclaration> {
    typed_node(
        TypedDeclaration::Import(TypedImport {
            module_path: vec![InternedString::new_global(name)],
            language: None,
            items: vec![],
            span: SPAN,
        }),
        Type::Primitive(PrimitiveType::Unit),
        SPAN,
    )
}

fn program(values: &[i64]) -> TypedProgram {
    let mut p = TypedProgram {
        language: Some(InternedString::new_global("script")),
        ..Default::default()
    };
    p.declarations.push(typed_node(
        TypedDeclaration::Function(TypedFunction {
            name: InternedString::new_global(RECORD),
            is_external: true,
            params: vec![TypedParameter::regular(
                InternedString::new_global("n"),
                Type::Primitive(PrimitiveType::I64),
                Mutability::Immutable,
                SPAN,
            )],
            return_type: Type::Primitive(PrimitiveType::Unit),
            ..Default::default()
        }),
        Type::Primitive(PrimitiveType::Unit),
        SPAN,
    ));
    if !values.is_empty() {
        p.declarations.push(typed_node(
            TypedDeclaration::Initializer(TypedInitializer {
                body: block(values),
                annotations: vec![],
                module: None,
            }),
            Type::Primitive(PrimitiveType::Unit),
            SPAN,
        ));
    }
    p
}

fn runtime() -> TieredRuntime {
    let mut r = TieredRuntime::new(TieredConfig {
        baseline_threshold: u32::MAX,
        enable_osr: false,
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    })
    .unwrap();
    r.register_function_typed(RECORD, record as *const u8, signature());
    r.finalize_runtime_symbols().unwrap();
    r
}

#[test]
fn top_level_code_runs_before_an_explicit_entry_and_is_not_repeated_by_promotion() {
    let _test = TEST.lock().unwrap();
    EVENTS.lock().unwrap().clear();
    let mut p = program(&[1, 2]);
    p.declarations.push(typed_node(
        TypedDeclaration::Function(TypedFunction {
            name: InternedString::new_global("main"),
            body: Some(block(&[9])),
            return_type: Type::Primitive(PrimitiveType::Unit),
            ..Default::default()
        }),
        Type::Primitive(PrimitiveType::Unit),
        SPAN,
    ));
    let mut r = runtime();
    r.declare_entry_points(["main"]);
    r.enter_only_through_entry_points();
    let hir = r.lower_to_hir(p.clone()).unwrap();
    let reachable = zyntax_compiler::dce::reachable_function_ids(&hir, &["main"]);
    assert!(hir.initializers.iter().all(|id| reachable.contains(id)));
    r.compile_typed_program(p).unwrap();
    assert_eq!(*EVENTS.lock().unwrap(), [1, 2]);
    assert_eq!(r.statistics().baseline_functions, 0);
    r.call::<()>("main", &[]).unwrap();
    r.optimize_function("main", OptimizationTier::Baseline)
        .unwrap();
    r.call::<()>("main", &[]).unwrap();
    #[cfg(feature = "llvm-backend")]
    {
        r.optimize_function("main", OptimizationTier::Optimized)
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while r.statistics().optimized_functions < 1 {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        r.call::<()>("main", &[]).unwrap();
    }
    let events = EVENTS.lock().unwrap();
    assert_eq!(&events[..2], &[1, 2]);
    assert!(events[2..].iter().all(|n| *n == 9));
    r.stop();
}

#[test]
fn snapshot_initializers_survive_relocation_and_run_once_before_each_client() {
    let _test = TEST.lock().unwrap();
    EVENTS.lock().unwrap().clear();
    let mut library = program(&[1]);
    library.declarations.push(typed_node(
        TypedDeclaration::Function(TypedFunction {
            name: InternedString::new_global("unused_export"),
            body: Some(block(&[])),
            return_type: Type::Primitive(PrimitiveType::Unit),
            visibility: zyntax_typed_ast::Visibility::Public,
            ..Default::default()
        }),
        Type::Primitive(PrimitiveType::Unit),
        SPAN,
    ));
    let hir =
        zyntax_embed::lower_for_snapshot("library", library.clone(), Default::default(), vec![])
            .unwrap();
    assert_eq!(hir.initializers.len(), 1);
    let bytes = SnapshotBuilder::new("script")
        .module_lowered("library", library, &hir)
        .unwrap()
        .encode()
        .unwrap();
    let snapshot = Arc::new(Snapshot::load_owned(bytes).unwrap());
    let mut r = runtime();
    r.install_snapshot(snapshot).unwrap();
    r.enter_only_through_entry_points();
    for value in [2, 3] {
        let mut p = program(&[value]);
        let mut selected = import("library");
        if let TypedDeclaration::Import(i) = &mut selected.node {
            i.items.push(TypedImportItem::Named {
                name: InternedString::new_global("unused_export"),
                alias: None,
            });
        }
        p.declarations.insert(0, selected);
        r.compile_typed_program(p).unwrap();
    }
    let mut joined = program(&[4]);
    joined.declarations.insert(0, import("library"));
    r.join_typed_program(joined, &[]).unwrap();
    assert_eq!(*EVENTS.lock().unwrap(), [1, 2, 3, 4]);
    r.stop();
}

#[test]
fn source_imports_initialize_shared_dependencies_before_the_importers() {
    let _test = TEST.lock().unwrap();
    EVENTS.lock().unwrap().clear();
    let mut left = program(&[2]);
    left.declarations.insert(0, import("shared"));
    let mut right = program(&[3]);
    right.declarations.insert(0, import("shared"));
    let imports = [("shared", program(&[1])), ("left", left), ("right", right)];
    let mut r = runtime();
    r.enter_only_through_entry_points();
    r.add_compiled_import_resolver(Box::new(move |name| {
        Ok(imports
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(name, p)| CompiledImport::new("script", *name, p.clone())))
    }));
    let mut p = program(&[4]);
    p.declarations.insert(0, import("right"));
    p.declarations.insert(0, import("left"));
    r.compile_typed_program(p).unwrap();
    assert_eq!(*EVENTS.lock().unwrap(), [1, 2, 3, 4]);
    r.stop();
}

#[test]
fn source_and_snapshot_imports_keep_their_declaration_order() {
    let _test = TEST.lock().unwrap();
    EVENTS.lock().unwrap().clear();
    let library = program(&[3]);
    let hir =
        zyntax_embed::lower_for_snapshot("right", library.clone(), Default::default(), vec![])
            .unwrap();
    let snapshot = Arc::new(
        Snapshot::load_owned(
            SnapshotBuilder::new("script")
                .module_lowered("right", library, &hir)
                .unwrap()
                .encode()
                .unwrap(),
        )
        .unwrap(),
    );
    let mut r = runtime();
    r.install_snapshot(snapshot).unwrap();
    r.enter_only_through_entry_points();
    r.add_compiled_import_resolver(Box::new(|name| {
        Ok((name == "left").then(|| CompiledImport::new("script", "left", program(&[2]))))
    }));
    let mut p = program(&[4]);
    p.declarations.insert(0, import("right"));
    p.declarations.insert(0, import("left"));
    r.compile_typed_program(p).unwrap();
    assert_eq!(*EVENTS.lock().unwrap(), [2, 3, 4]);
    r.stop();
}

#[test]
fn failed_initialization_stops_loading_and_caches_the_failure() {
    let _test = TEST.lock().unwrap();
    EVENTS.lock().unwrap().clear();
    let mut module =
        zyntax_embed::lower_for_snapshot("failure", program(&[1]), Default::default(), vec![])
            .unwrap();
    let init = module.functions.get_mut(&module.initializers[0]).unwrap();
    init.blocks.get_mut(&init.entry_block).unwrap().terminator = HirTerminator::Unreachable;
    let mut r = runtime();
    assert!(r.compile_module(module.clone()).is_err());
    assert!(r.compile_module(module).is_err());
    assert_eq!(*EVENTS.lock().unwrap(), [1]);
    r.stop();
}

#[test]
fn classic_runtime_also_executes_top_level_blocks_without_main() {
    let _test = TEST.lock().unwrap();
    EVENTS.lock().unwrap().clear();
    let mut r = ZyntaxRuntime::new().unwrap();
    r.register_function_typed(RECORD, record as *const u8, signature());
    r.finalize_runtime_symbols().unwrap();
    let mut p = program(&[5, 6]);
    p.declarations.push(typed_node(
        TypedDeclaration::Function(TypedFunction {
            name: InternedString::new_global("after"),
            body: Some(block(&[7])),
            return_type: Type::Primitive(PrimitiveType::Unit),
            ..Default::default()
        }),
        Type::Primitive(PrimitiveType::Unit),
        SPAN,
    ));
    r.compile_typed_program(p).unwrap();
    assert_eq!(*EVENTS.lock().unwrap(), [5, 6]);
    r.call::<()>("after", &[]).unwrap();
    assert_eq!(*EVENTS.lock().unwrap(), [5, 6, 7]);
}
