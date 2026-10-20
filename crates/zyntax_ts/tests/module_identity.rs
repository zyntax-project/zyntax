//! Native module identities survive shared source and snapshot linking.

use std::sync::{Arc, Mutex};
use zyntax_compiler::hir::{HirCallable, HirInstruction, HirType, Intrinsic};
use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{CompiledImport, Snapshot, SnapshotBuilder, TieredRuntime, ZyntaxValue};
use zyntax_ts::{ModuleId, parse_module, parse_program};
use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedImport, TypedImportItem};
use zyntax_typed_ast::{PrimitiveType, Span, Type, TypedNode, TypedProgram};

static TEST: Mutex<()> = Mutex::new(());

const LEFT: &str = r#"
    type Point={x:number;y:boolean};
    function make(n:number):Point{return{x:n,y:true};}
    function second<T>(xs:T[],fallback:T):T{if(xs.length>1)return xs[1];return fallback;}
    function select():(n:number)=>number{return n=>n+1;}
    function score(n:number):number{
        const p=make(n);const xs=[p.x,p.x+4];const f=select();
        return f(second(xs,0))+(p.y?10:0);
    }
    score(0);
"#;

const RIGHT: &str = r#"
    type Point={a:boolean;x:number;y:number};
    function make(n:number):Point{return{a:false,x:n+10,y:3};}
    function second<T>(xs:T[],fallback:T):T{if(xs.length>1)return xs[1];return fallback;}
    function select():(n:number)=>number{return n=>n*2;}
    function score(n:number):number{
        const p=make(n);const xs=[p.x,p.x+p.y];const f=select();
        return f(second(xs,0))+(p.a?100:0);
    }
    score(0);
"#;

fn definitions(program: &TypedProgram) -> Vec<zyntax_typed_ast::InternedString> {
    program
        .declarations
        .iter()
        .filter_map(|d| match &d.node {
            TypedDeclaration::Function(f) if !f.is_external => Some(f.name),
            _ => None,
        })
        .collect()
}

#[test]
fn identities_are_stable_and_do_not_alias_source_names_or_linker_escapes() {
    let left = parse_program(LEFT, "left/index.ts", Default::default()).unwrap();
    let right = parse_program(RIGHT, "right/index.ts", Default::default()).unwrap();
    let left_names = definitions(&left);
    assert!(
        definitions(&right)
            .iter()
            .all(|name| !left_names.contains(name))
    );
    for program in [&left, &right] {
        let origin = program.type_registry.current_module().unwrap();
        let records: Vec<_> = program.type_registry.get_all_types().collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].module, Some(origin));
        assert!(program.declarations.iter().any(|d| {
            matches!(&d.node, TypedDeclaration::Initializer(i) if i.module == Some(origin))
        }));
    }
    let ids = [
        "pkg/a.ts",
        "pkg_2fa.ts",
        "pkg.a",
        "pkg_P_a",
        "β.ts",
        "_ce_b2.ts",
        "",
    ];
    let mut names = rustc_hash::FxHashSet::default();
    for id in ids {
        let module = ModuleId::new(id);
        for source_name in ["score", "café", "caf_Xe9_", "caf_c3_a9", "$fn"] {
            let name = module.function_name(source_name).resolve_global().unwrap();
            assert!(
                name.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'$')
            );
            assert!(names.insert(name));
        }
        let source = "function café(n:number):number{return n;} function caf_c3_a9(n:number):number{return café(n)+1;}";
        let a = parse_module(source, "diagnostics.ts", module, Default::default()).unwrap();
        let b = parse_module(source, "another/path.ts", module, Default::default()).unwrap();
        assert_eq!(definitions(&a), definitions(&b));
        assert!(definitions(&a).contains(&module.function_name("café")));
        assert!(definitions(&a).contains(&module.function_name("caf_c3_a9")));
        assert_ne!(
            module.function_name("café"),
            module.function_name("caf_c3_a9")
        );
    }
}

fn linked_program(modules: &[ModuleId; 2]) -> TypedProgram {
    let mut root = parse_program("", "root.ts", Default::default()).unwrap();
    for module in modules {
        root.declarations.push(TypedNode::new(
            TypedDeclaration::Import(TypedImport {
                language: None,
                module_path: vec![module.name()],
                items: vec![TypedImportItem::Named {
                    name: module.function_name("score"),
                    alias: None,
                }],
                span: Span::default(),
            }),
            Type::Primitive(PrimitiveType::Unit),
            Span::default(),
        ));
    }
    root
}

fn link_and_check(snapshot: bool) {
    let modules = [ModuleId::new("pkg/a.ts"), ModuleId::new("pkg_2fa.ts")];
    let programs = [LEFT, RIGHT]
        .into_iter()
        .zip(modules)
        .map(|(source, id)| parse_module(source, "index.ts", id, Default::default()).unwrap())
        .collect::<Vec<_>>();
    let mut runtime = TieredRuntime::new(TieredConfig {
        baseline_threshold: u32::MAX,
        enable_osr: false,
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    })
    .unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    if snapshot {
        let mut builder = SnapshotBuilder::new("typescript");
        for (program, id) in programs.into_iter().zip(modules) {
            let name = id.name().resolve_global().unwrap();
            let hir = zyntax_embed::lower_for_snapshot_releasing(
                &name,
                program.clone(),
                Default::default(),
                vec![],
                true,
                None,
            )
            .unwrap();
            builder = builder.module_lowered(name, program, &hir).unwrap();
        }
        runtime
            .install_snapshot(Arc::new(
                Snapshot::load_owned(builder.encode().unwrap()).unwrap(),
            ))
            .unwrap();
    } else {
        runtime.add_compiled_import_resolver(Box::new(move |name| {
            Ok(programs.iter().zip(modules).find_map(|(p, id)| {
                (id.name().resolve_global().as_deref() == Some(name))
                    .then(|| CompiledImport::new("typescript", name, p.clone()))
            }))
        }));
    }
    let scores = modules.map(|m| m.function_name("score").resolve_global().unwrap());
    runtime.declare_entry_points(scores.iter().map(String::as_str));
    runtime.enter_only_through_entry_points();
    let root = linked_program(&modules);
    let mut hir = runtime.lower_to_hir(root.clone()).unwrap();
    assert_eq!(hir.initializers.len(), 2);
    // Snapshot concrete bodies stay encoded until reached by execution or JIT.
    if !snapshot {
        zyntax_compiler::drop_insert::run_module(&mut hir);
        let mut geps = 0;
        let mut frees = 0;
        for f in hir.functions.values() {
            for instruction in f.blocks.values().flat_map(|b| &b.instructions) {
                match instruction {
                    HirInstruction::GetElementPtr {
                        ty: HirType::Ptr(ty),
                        ..
                    } if **ty == HirType::F64 => geps += 1,
                    HirInstruction::Call {
                        callee: HirCallable::Intrinsic(Intrinsic::Free),
                        ..
                    } => frees += 1,
                    HirInstruction::Call {
                        callee: HirCallable::Symbol(name),
                        ..
                    } => assert!(!name.contains("Dynamic") && !name.contains("Box")),
                    _ => {}
                }
            }
        }
        assert!(geps >= 2, "both libraries retain native float array access");
        assert!(
            frees >= 6,
            "records, headers and buffers have static releases"
        );
    }
    let names = runtime.compile_typed_program(root).unwrap();
    let natives: Vec<_> = names
        .into_iter()
        .filter(|name| name.starts_with("ts$m$"))
        .collect();
    assert_eq!(runtime.statistics().baseline_functions, 0);
    assert_eq!(runtime.statistics().optimized_functions, 0);
    let check = |runtime: &mut TieredRuntime| {
        for n in [-3.0, 0.0, 5.5] {
            assert_eq!(
                runtime
                    .call_raw(&scores[0], &[ZyntaxValue::Float(n)])
                    .unwrap(),
                ZyntaxValue::Float(n + 15.0)
            );
            assert_eq!(
                runtime
                    .call_raw(&scores[1], &[ZyntaxValue::Float(n)])
                    .unwrap(),
                ZyntaxValue::Float(2.0 * n + 26.0)
            );
        }
    };
    check(&mut runtime);
    runtime
        .optimize_function(&scores[0], OptimizationTier::Baseline)
        .unwrap();
    let baseline = runtime.statistics().baseline_functions;
    assert!(baseline > 0 && baseline < natives.len());
    check(&mut runtime);
    for name in &natives {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        runtime
            .optimize_function(&scores[0], OptimizationTier::Optimized)
            .unwrap();
        wait(&runtime, 1);
        assert!(runtime.statistics().optimized_functions < natives.len());
        check(&mut runtime);
        for name in &natives {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        wait(&runtime, natives.len());
        check(&mut runtime);
    }
    assert!(!zyntax_compiler::collector::is_enabled());
    runtime.stop();
}

#[cfg(feature = "llvm-backend")]
fn wait(runtime: &TieredRuntime, count: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while runtime.statistics().optimized_functions < count {
        assert!(std::time::Instant::now() < deadline, "LLVM did not install");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn separately_parsed_libraries_keep_their_bodies_layouts_and_callbacks() {
    let _test = TEST.lock().unwrap();
    link_and_check(false);
}

#[test]
fn snapshotted_libraries_keep_their_bodies_layouts_and_callbacks() {
    let _test = TEST.lock().unwrap();
    link_and_check(true);
}
