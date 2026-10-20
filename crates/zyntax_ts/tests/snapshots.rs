//! Native generic libraries retain templates across the shared snapshot boundary.

use std::sync::Arc;
use zyntax_compiler::hir::*;
use zyntax_embed::{Snapshot, SnapshotBuilder};
use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedImport, TypedImportItem};
use zyntax_typed_ast::{InternedString, Span, Type, TypedNode, TypedProgram};

// Tiered backends install process-wide lazy-compiler callbacks.
static TIERED_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "allocation-audit")]
#[global_allocator]
static ALLOCATOR: zyntax_compiler::allocation_audit::CountingAllocator =
    zyntax_compiler::allocation_audit::CountingAllocator;

const LIBRARY: &str = r#"
    function identity<T>(x:T):T{return x;}
    function second<T>(xs:T[],fallback:T):T{if(xs.length>1)return xs[1];return fallback;}
    function through<U>(xs:U[],fallback:U):U{return second<U>(xs,fallback);}
    function alias<T>(xs:T[]):T[]{return xs;}
    function recur<T>(x:T,n:number):T{if(n<=0)return identity(x);return recur<T>(x,n-1);}
    function bump(n:number):number{return n+1;}
    function step<T>(x:T,other:T,n:number):T{if(bump(n)>0)return recur<T>(x,2);return other;}
    function literal<T>(x:T):string{return "from snapshot";}
    function unused<T>(x:T):T{return x;}
"#;

fn library_program() -> TypedProgram {
    zyntax_ts::parse_program(LIBRARY, "library.ts", Default::default()).unwrap()
}

fn library() -> Arc<Snapshot> {
    let program = library_program();
    let hir = zyntax_embed::lower_for_snapshot_releasing(
        "library",
        program.clone(),
        Default::default(),
        vec![],
        true,
        None,
    )
    .unwrap();
    Arc::new(
        Snapshot::load_owned(
            SnapshotBuilder::new("typescript")
                .module_lowered("library", program, &hir)
                .unwrap()
                .encode()
                .unwrap(),
        )
        .unwrap(),
    )
}

// Parse against the library's signatures until the native frontend has import syntax.
// Runtime linking must use the snapshot; none of these source bodies reach it.
fn client(source: &str) -> TypedProgram {
    let mut program = zyntax_ts::parse_program(
        &format!("{LIBRARY}\n{source}"),
        "client.ts",
        Default::default(),
    )
    .unwrap();
    let mut items = Vec::new();
    program.declarations.retain(|d| {
        if matches!(&d.node, TypedDeclaration::Function(f) if f.name.resolve_global().as_deref()==Some("ts$fn$bump")) {
            return false;
        }
        if let TypedDeclaration::Function(f) = &d.node
            && !f.type_params.is_empty()
        {
            items.push(TypedImportItem::Named {
                name: f.name,
                alias: None,
            });
            return false;
        }
        true
    });
    program.declarations.insert(
        0,
        TypedNode::new(
            TypedDeclaration::Import(TypedImport {
                language: None,
                module_path: vec![InternedString::new_global("library")],
                items,
                span: Span::new(0, 0),
            }),
            Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
            Span::new(0, 0),
        ),
    );
    program
}

#[test]
fn snapshot_preserves_uninstantiated_generic_bodies() {
    let program = zyntax_ts::parse_program(
        "function identity<T>(x:T):T{return x;} function second<T>(xs:T[],fallback:T):T{if(xs.length>1)return xs[1];return fallback;}",
        "library.ts",
        Default::default(),
    )
    .unwrap();
    let hir =
        zyntax_embed::lower_for_snapshot("library", program.clone(), Default::default(), vec![])
            .unwrap();
    for name in ["ts$fn$identity", "ts$fn$second"] {
        let function = hir
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some(name))
            .expect("a library export needs its unspecialized template");
        assert_eq!(function.signature.type_params.len(), 1);
        assert!(!function.blocks.is_empty());
        assert!(!function.attributes.optimized);
    }
    let bytes = SnapshotBuilder::new("typescript")
        .module_lowered("library", program, &hir)
        .unwrap()
        .encode()
        .unwrap();
    let snapshot = Snapshot::load_owned(bytes).unwrap();
    let import = snapshot.module("library").unwrap().unwrap();
    let encoded = import.hir().unwrap();
    for name in ["ts$fn$identity", "ts$fn$second"] {
        let shell = encoded.by_name(name).unwrap();
        assert_eq!(shell.signature.type_params.len(), 1);
        assert!(shell.blocks.is_empty());
        let body = encoded.function(shell.id).unwrap();
        assert!(!body.blocks.is_empty());
    }
}

#[test]
fn imported_generics_run_across_interpreter_native_and_mixed_tiers() {
    let _runtime = TIERED_TEST.lock().unwrap();
    use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
    use zyntax_embed::{TieredRuntime, ZyntaxValue};
    let snapshot = library();
    for closed in [false, true] {
        let config = TieredConfig {
            baseline_threshold: u32::MAX,
            enable_osr: false,
            #[cfg(feature = "llvm-backend")]
            tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
            ..Default::default()
        };
        let mut runtime = TieredRuntime::new(config).unwrap();
        runtime.install_snapshot(Arc::clone(&snapshot)).unwrap();
        zyntax_ts::register_runtime(&mut runtime).unwrap();
        if closed {
            runtime.enter_only_through_entry_points();
        }
        let program = client(
            r#"
            function score(n:number):number {
                const a=[n,n+2,n+4]; const b=alias(a); b.push(n+8);
                return through(b,0)+through(a,0)+identity(n)+step(n,-100,1)+b.length;
            }
            function words():number {return through(["first","second","third"],"fallback")==="second" && literal(2)==="from snapshot" ? 1 : -1;}
            function flags():number {return through([false,true,false],false) && !through([true,false,true],true) ? 1 : -1;}
            score(1);words();flags();
        "#,
        );
        let names = runtime.compile_typed_program(program).unwrap();
        assert!(!names.iter().any(|n| n.starts_with("ts$fn$unused")));
        assert!(!names.iter().any(|n| n == "ts$fn$second"));
        assert_eq!(
            names
                .iter()
                .filter(|n| n.starts_with("ts$fn$second__mono_"))
                .count(),
            3
        );
        let check = |runtime: &mut TieredRuntime| {
            for n in [-2.0, 0.0, 3.0, 7.0] {
                assert_eq!(
                    runtime
                        .call_raw("ts$fn$score", &[ZyntaxValue::Float(n)])
                        .unwrap(),
                    ZyntaxValue::Float(4.0 * n + 8.0)
                );
            }
            for name in ["ts$fn$words", "ts$fn$flags"] {
                assert_eq!(
                    runtime.call_raw(name, &[]).unwrap(),
                    ZyntaxValue::Float(1.0)
                );
            }
        };
        check(&mut runtime);
        assert_eq!(runtime.statistics().baseline_functions, 0);
        assert_eq!(runtime.statistics().optimized_functions, 0);
        let natives: Vec<_> = names.iter().filter(|n| n.starts_with("ts$fn$")).collect();
        let leaves: Vec<_> = natives
            .iter()
            .copied()
            .filter(|n| n.starts_with("ts$fn$second__mono_"))
            .collect();
        for name in &leaves {
            runtime
                .optimize_function(name, OptimizationTier::Baseline)
                .unwrap();
        }
        check(&mut runtime);
        for name in &natives {
            runtime
                .optimize_function(name, OptimizationTier::Baseline)
                .unwrap();
        }
        check(&mut runtime);
        #[cfg(feature = "llvm-backend")]
        {
            let wait = |runtime: &TieredRuntime, count| {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
                while runtime.statistics().optimized_functions < count {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "LLVM promotion did not install"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            };
            for name in &leaves {
                runtime
                    .optimize_function(name, OptimizationTier::Optimized)
                    .unwrap();
            }
            wait(&runtime, leaves.len());
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
}

fn caller(template: &HirFunction, name: &str, ty: HirType) -> HirFunction {
    let mut signature = template.signature.clone();
    signature.type_params.clear();
    signature.returns = vec![ty.clone()];
    signature.params.last_mut().unwrap().ty = ty.clone();
    let mut f = HirFunction::new(InternedString::new_global(name), signature);
    let types: Vec<_> = f.signature.params.iter().map(|p| p.ty.clone()).collect();
    let args = types
        .into_iter()
        .enumerate()
        .map(|(i, ty)| f.create_value(ty, HirValueKind::Parameter(i as u32)))
        .collect();
    let result = f.create_value(ty.clone(), HirValueKind::Instruction);
    let block = f.blocks.get_mut(&f.entry_block).unwrap();
    block.instructions.push(HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Function(template.id),
        args,
        type_args: vec![ty],
        const_args: vec![],
        is_tail: false,
    });
    block.terminator = HirTerminator::Return {
        values: vec![result],
    };
    f
}

#[test]
fn snapshotted_arrays_specialize_with_native_strides_and_reuse() {
    let _runtime = TIERED_TEST.lock().unwrap();
    let snapshot = library();
    let import = snapshot.module("library").unwrap().unwrap();
    let encoded = import.hir().unwrap();
    let template = encoded
        .function(encoded.by_name("ts$fn$second").unwrap().id)
        .unwrap();
    let mut module = HirModule::new(InternedString::new_global("snapshot_strides"));
    let mut callers = Vec::new();
    for (name, ty) in [
        ("integer", HirType::I64),
        ("float", HirType::F64),
        ("string", HirType::Ptr(Arc::new(HirType::I8))),
        ("boolean", HirType::Bool),
        ("integer_again", HirType::I64),
    ] {
        let f = caller(&template, name, ty);
        callers.push(f.id);
        module.functions.insert(f.id, f);
    }
    module.functions.insert(template.id, template);
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    assert_eq!(module.functions.len(), 9);
    for f in module.functions.values().filter(|f| {
        f.name
            .resolve_global()
            .unwrap()
            .starts_with("ts$fn$second__mono_")
    }) {
        let element = &f.signature.returns[0];
        assert!(f.blocks.values().flat_map(|b| &b.instructions).any(|i| matches!(i,HirInstruction::GetElementPtr{ty:HirType::Ptr(t),..} if t.as_ref()==element)));
        if element == &HirType::Bool {
            assert!(
                f.blocks
                    .values()
                    .flat_map(|b| &b.instructions)
                    .any(|i| matches!(
                        i,
                        HirInstruction::Load {
                            ty: HirType::Bool,
                            align: 1,
                            ..
                        }
                    ))
            );
        }
    }
    let mut backend = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
    backend.compile_module(&module).unwrap();
    // SAFETY: the generated callers take a List header and the native element type.
    unsafe {
        let integer: extern "C" fn(*const usize, i64) -> i64 =
            std::mem::transmute(backend.get_function_ptr(callers[0]).unwrap());
        let float: extern "C" fn(*const usize, f64) -> f64 =
            std::mem::transmute(backend.get_function_ptr(callers[1]).unwrap());
        let string: extern "C" fn(*const usize, *const u8) -> *const u8 =
            std::mem::transmute(backend.get_function_ptr(callers[2]).unwrap());
        let boolean: extern "C" fn(*const usize, bool) -> bool =
            std::mem::transmute(backend.get_function_ptr(callers[3]).unwrap());
        let ints = [-7i64, 42];
        let floats = [1.5f64, 4.25];
        let bools = [false, true, false];
        let strings = [
            c"first".as_ptr().cast::<u8>(),
            c"second".as_ptr().cast::<u8>(),
        ];
        assert_eq!(integer([ints.as_ptr() as usize, 2, 2].as_ptr(), 9), 42);
        assert_eq!(float([floats.as_ptr() as usize, 2, 2].as_ptr(), 9.0), 4.25);
        assert_eq!(
            string([strings.as_ptr() as usize, 2, 2].as_ptr(), std::ptr::null()),
            strings[1]
        );
        assert!(boolean([bools.as_ptr() as usize, 3, 3].as_ptr(), false));
        assert_eq!(integer([0, 0, 0].as_ptr(), 9), 9);
    }

    use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
    use zyntax_embed::{TieredRuntime, ZyntaxValue};
    let leaf = module
        .functions
        .values()
        .find(|f| {
            f.name
                .resolve_global()
                .unwrap()
                .starts_with("ts$fn$second__mono_")
                && f.signature.returns == [HirType::I64]
        })
        .unwrap()
        .name
        .resolve_global()
        .unwrap();
    let config = TieredConfig {
        baseline_threshold: u32::MAX,
        enable_osr: false,
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    runtime.compile_module(module).unwrap();
    let ints = [-7i64, 42];
    let header = [ints.as_ptr() as usize, 2, 2];
    let check = |runtime: &TieredRuntime| {
        assert_eq!(
            runtime
                .call_raw(
                    "integer",
                    &[
                        ZyntaxValue::Int(header.as_ptr() as i64),
                        ZyntaxValue::Int(9),
                    ]
                )
                .unwrap(),
            ZyntaxValue::Int(42)
        );
    };
    check(&runtime);
    assert_eq!(runtime.statistics().baseline_functions, 0);
    for name in [&leaf, "integer"] {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
        check(&runtime);
    }
    #[cfg(feature = "llvm-backend")]
    for (index, name) in [&leaf, "integer"].into_iter().enumerate() {
        runtime
            .optimize_function(name, OptimizationTier::Optimized)
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while runtime.statistics().optimized_functions < index + 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "integer LLVM promotion did not install"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        check(&runtime);
    }
    runtime.stop();
}

#[test]
fn a_generic_signature_without_a_body_cannot_be_specialized() {
    let snapshot = library();
    let import = snapshot.module("library").unwrap().unwrap();
    let template = import.hir().unwrap().by_name("ts$fn$identity").unwrap();
    let f = caller(template, "missing_body", HirType::I64);
    let mut module = HirModule::new(InternedString::new_global("missing_template"));
    module.functions.insert(f.id, f);
    module.functions.insert(template.id, template.clone());
    let error = zyntax_compiler::monomorphize_module(&mut module).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("generic template body unavailable"),
        "{error}"
    );
}

#[test]
fn library_concrete_instances_do_not_consume_templates_or_reuse_names() {
    let program = zyntax_ts::parse_program(
        "function identity<T>(x:T):T{return x;} function local():number{return identity(1);}",
        "concrete.ts",
        Default::default(),
    )
    .unwrap();
    let mut module =
        zyntax_embed::lower_for_snapshot("concrete", program, Default::default(), vec![]).unwrap();
    let template = module
        .functions
        .values()
        .find(|f| !f.signature.type_params.is_empty())
        .unwrap();
    let f = caller(template, "later", HirType::Bool);
    module.functions.insert(f.id, f);
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    let names: Vec<_> = module.functions.values().map(|f| f.name).collect();
    assert_eq!(
        names.len(),
        names.iter().collect::<std::collections::HashSet<_>>().len()
    );
    assert_eq!(
        names
            .iter()
            .filter(|n| n
                .resolve_global()
                .unwrap()
                .starts_with("ts$fn$identity__mono_"))
            .count(),
        2
    );
    assert!(
        module
            .functions
            .values()
            .all(|f| f.signature.type_params.is_empty())
    );
}

#[cfg(feature = "allocation-audit")]
#[test]
#[ignore = "allocation profile; run alone with --ignored --test-threads=1"]
fn unused_snapshot_template_stays_encoded() {
    let _runtime = TIERED_TEST.lock().unwrap();
    use zyntax_compiler::allocation_audit::{self, Phase};
    use zyntax_compiler::tiered_backend::TieredConfig;
    use zyntax_embed::TieredRuntime;
    let program = library_program();
    let mut module =
        zyntax_embed::lower_for_snapshot("library", program.clone(), Default::default(), vec![])
            .unwrap();
    let unused = module
        .functions
        .values_mut()
        .find(|f| f.name.resolve_global().as_deref() == Some("ts$fn$unused"))
        .unwrap();
    for i in 0..65536 {
        unused.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(i)));
    }
    let bytes = SnapshotBuilder::new("typescript")
        .module_lowered("library", program, &module)
        .unwrap()
        .encode()
        .unwrap();
    let image_bytes = bytes.len();
    let snapshot = Arc::new(Snapshot::load_owned(bytes).unwrap());
    let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
    runtime.install_snapshot(snapshot).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    runtime.enter_only_through_entry_points();
    let program = client("console.log(through([1,2],0));");
    let before = allocation_audit::for_phase(Phase::Decode);
    let hir = runtime.lower_to_hir(program).unwrap();
    let after = allocation_audit::for_phase(Phase::Decode);
    assert!(
        !hir.functions.values().any(|f| f
            .name
            .resolve_global()
            .unwrap()
            .starts_with("ts$fn$unused"))
    );
    assert_eq!(
        after.entries - before.entries,
        2,
        "only through and second templates decode"
    );
    assert!(
        after.requested_bytes - before.requested_bytes < (image_bytes / 8) as u64,
        "unused template body was decoded"
    );
    runtime.stop();
}
