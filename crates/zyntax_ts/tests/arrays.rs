//! Typed arrays retain element GEPs, specialization and shared ownership analysis.

use std::sync::{Arc, Mutex};
use zyntax_compiler::hir::{HirCallable, HirInstruction, HirModule, HirType, Intrinsic};
use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
use zyntax_typed_ast::{InternedString, arena::AstArena};

const SOURCE: &str = r#"
    function first<T>(xs:T[],fallback:T):T { if(xs.length>0)return xs[0];return fallback; }
    function through<U>(xs:U[],fallback:U):U { return first<U>(xs,fallback); }
    function alias<T>(xs:T[]):T[] { return xs; }
    function make(n:number):number[] { return [n,n+1,n+2]; }
    function scale(xs:number[],k:number):number {
        let sum=0;for(let i=0;i<xs.length;i++){xs[i]*=k;sum+=xs[i];}return sum;
    }
    function boolCount(xs:boolean[]) {
        let n=0;for(let i=0;i<xs.length;i+=1){if(xs[i])n+=1;}return n;
    }
    function score(n:number) {
        const xs=make(n);const ys=alias(xs);ys.push(n+3,n+4);
        return scale(ys,2)+through(xs,0);
    }
    function boolScore(n:number) { const xs=[true,false,true,false,true];return boolCount(xs)+(through(xs,false)?n:0); }
    function wordScore(n:number) { const xs=["a","b","c"];let last="";for(let i=0;i<xs.length;i++)last=xs[i];return last==="c" && through(xs,"x")==="a"?n+1:-10; }
    function pair(xs:number[]) { if(xs.length>1)return xs[0]*100+xs[1];return -1; }
    function probe() { return pair([3,2]); }
"#;

fn lower(source: &str) -> HirModule {
    let mut program = zyntax_ts::parse_program(source, "arrays.ts", Default::default()).unwrap();
    let mut context = LoweringContext::new(
        InternedString::new_global("arrays"),
        Arc::new(program.type_registry.clone()),
        Arc::new(Mutex::new(AstArena::new())),
        LoweringConfig::default(),
    );
    let mut module = context.lower_program(&mut program).unwrap();
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    module
}

#[test]
fn generic_array_elements_specialize_once_without_dynamic_helpers() {
    let mut module = lower(SOURCE);
    for prefix in ["ts$fn$first__mono_", "ts$fn$through__mono_"] {
        let functions: Vec<_> = module
            .functions
            .values()
            .filter(|f| f.name.resolve_global().unwrap().starts_with(prefix))
            .collect();
        assert_eq!(functions.len(), 3, "{prefix}");
        let returns: Vec<_> = functions
            .iter()
            .map(|f| f.signature.returns[0].clone())
            .collect();
        assert!(returns.contains(&HirType::F64));
        assert!(returns.contains(&HirType::Bool));
        assert!(returns.contains(&HirType::Ptr(Arc::new(HirType::I8))));
        for function in functions {
            assert!(function.signature.type_params.is_empty());
            for instruction in function.blocks.values().flat_map(|b| &b.instructions) {
                if let HirInstruction::Load {
                    ty: HirType::Bool,
                    align,
                    ..
                } = instruction
                {
                    assert_eq!(*align, 1);
                }
            }

            if prefix.contains("first") {
                assert!(function.blocks.values().flat_map(|b|&b.instructions).any(|i|matches!(i,HirInstruction::GetElementPtr{ty:HirType::Ptr(element),..} if element.as_ref()==&function.signature.returns[0])));
            }
        }
    }
    for function in module.functions.values() {
        for instruction in function.blocks.values().flat_map(|b| &b.instructions) {
            if let HirInstruction::Call {
                callee: HirCallable::Symbol(name),
                ..
            } = instruction
            {
                assert!(
                    !name.contains("box_")
                        && !name.contains("getitem")
                        && !name.contains("list_get")
                        && !name.contains("list_set"),
                    "{name}"
                );
            }
        }
    }
    let count = module.functions.len();
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    assert_eq!(module.functions.len(), count);
}

#[test]
fn optimized_hir_and_clif_keep_typed_geps_and_integer_induction() {
    let module = lower(SOURCE);
    let scale = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some("ts$fn$scale"))
        .unwrap()
        .clone();
    let mut module = HirModule::new(InternedString::new_global("scale"));
    module.functions.insert(scale.id, scale);
    zyntax_compiler::run_interp_safe_opts(&mut module);
    let f = module.functions.values().next().unwrap();
    assert!(f.blocks.values().flat_map(|b|&b.instructions).any(|i|matches!(i,HirInstruction::GetElementPtr{ty:HirType::Ptr(element),..} if **element==HirType::F64)));
    assert!(
        !f.blocks
            .values()
            .flat_map(|b| &b.instructions)
            .any(|i| matches!(
                i,
                HirInstruction::Cast {
                    op: zyntax_compiler::hir::CastOp::FpToSi,
                    ..
                }
            )),
        "no float-to-int conversion per index"
    );
    let mut backend = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
    backend.set_capture_ir(true);
    backend.compile_module(&module).unwrap();
    assert!(backend.get_function_ptr(f.id).is_some());
    let (clif, _) = backend.take_captured_ir().unwrap();
    assert!(clif.contains("load.f64") && clif.contains("fmul"), "{clif}");
    assert!(!clif.contains("fcvt_to_sint"), "{clif}");
}

#[test]
fn array_headers_and_buffers_share_one_owner_through_returns_and_aliases() {
    let mut module = lower(
        r#"
        function make(n:number):number[]{return [n,n+1];}
        function alias(xs:number[]):number[]{return xs;}
        const a=make(3);const b=alias(a);b[1]=7;console.log(a[1]);
    "#,
    );
    module.automatic_release = true;
    zyntax_compiler::drop_insert::run_module(&mut module);
    for f in module.functions.values().filter(|f| !f.is_external) {
        let frees = f
            .blocks
            .values()
            .flat_map(|b| &b.instructions)
            .filter(|i| {
                matches!(
                    i,
                    HirInstruction::Call {
                        callee: HirCallable::Intrinsic(Intrinsic::Free),
                        ..
                    }
                )
            })
            .count();
        let name = f.name.resolve_global().unwrap();
        assert_eq!(
            frees,
            if module.initializers.contains(&f.id) {
                2
            } else {
                0
            },
            "{name}: one buffer and one header"
        );
    }
    let before = format!("{module:?}");
    zyntax_compiler::drop_insert::run_module(&mut module);
    assert_eq!(before, format!("{module:?}"));
}

#[test]
fn unproven_indexing_and_reference_lifetimes_are_explicit_gaps() {
    for source in [
        "const a=[1];console.log(a[2]);",
        "const a=[1];console.log(a[-1]);",
        "const a=[1];console.log(a[0.5]);",
        "function f(a:number[],i:number){return a[i];}",
        "const a=[1];a[3]=9;",
        "const a=[1,,3];",
        "const s=\"a\"+\"b\";const a=[s];",
        "function f(s:string){return [s];}",
        "function f<T>(x:T){return [x];}",
        "function f<T>(xs:T[],x:T){if(xs.length>0)xs[0]=x;}",
        "function f(xs:number[]){let step=1;for(let i=0;i<xs.length;i+=step){step=-1;console.log(xs[i]);}}",
        "let i=0;console.log([i=7][i]);",
        "function alias(a:number[],n:number){return a;}const a=[1];let i=0;console.log(alias(a,i=3)[i]);",
        "function pick(n:number,a:number[]){return a;}let a=[1];const b=pick((a=[]).length,a);console.log(b[0]);",
        "function pick<T>(x:T,y:T):T{return x;}let s=\"a\"+\"b\";const a=[pick(s,s=\"literal\")];",
        "let a=[1];if(a.length>0 && (a=[]).length===0)console.log(a[0]);",
        "let i=0;const a=[1];if(i<a.length && (i=3)===3)console.log(a[i]);",
    ] {
        let error = zyntax_ts::parse_program(source, "gap.ts", Default::default()).unwrap_err();
        assert_eq!(
            error.kind,
            zyntax_ts::ErrorKind::Unsupported,
            "{source}: {error}"
        );
    }
}

#[test]
fn arrays_agree_across_interpreter_native_and_mixed_tiers() {
    use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
    use zyntax_embed::{TieredRuntime, ZyntaxValue};
    let program = zyntax_ts::parse_program(SOURCE, "tiers.ts", Default::default()).unwrap();
    let config = TieredConfig {
        baseline_threshold: u32::MAX,
        enable_osr: false,
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    let names: Vec<_> = runtime
        .compile_typed_program(program)
        .unwrap()
        .into_iter()
        .filter(|n| n.starts_with("ts$fn$"))
        .collect();
    let check = |runtime: &mut TieredRuntime| {
        assert_eq!(
            runtime.call_raw("ts$fn$probe", &[]).unwrap(),
            ZyntaxValue::Float(302.0)
        );
        for n in [-2.0, 0.0, 3.0, 7.0] {
            for (function, expected) in [
                ("score", 12.0 * n + 20.0),
                ("boolScore", 3.0 + n),
                ("wordScore", n + 1.0),
            ] {
                assert_eq!(
                    runtime
                        .call_raw(&format!("ts$fn${function}"), &[ZyntaxValue::Float(n)])
                        .unwrap(),
                    ZyntaxValue::Float(expected),
                    "{function}({n})"
                );
            }
        }
    };
    check(&mut runtime);
    assert_eq!(runtime.statistics().baseline_functions, 0);
    for name in names
        .iter()
        .filter(|n| n.starts_with("ts$fn$first__mono_") || n.as_str() == "ts$fn$make")
    {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    for name in &names {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        let wait = |runtime: &TieredRuntime, installed| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            while runtime.statistics().optimized_functions < installed {
                assert!(
                    std::time::Instant::now() < deadline,
                    "array LLVM promotion did not install"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        };
        let leaves: Vec<_> = names
            .iter()
            .filter(|n| n.starts_with("ts$fn$first__mono_") || n.as_str() == "ts$fn$make")
            .collect();
        for name in &leaves {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        wait(&runtime, leaves.len());
        check(&mut runtime);
        runtime
            .optimize_function("ts$fn$score", OptimizationTier::Optimized)
            .unwrap();
        wait(&runtime, leaves.len() + 1);
        check(&mut runtime);
        for name in names
            .iter()
            .filter(|n| !leaves.contains(n) && n.as_str() != "ts$fn$score")
        {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        wait(&runtime, names.len());
        check(&mut runtime);
    }
    assert!(!zyntax_compiler::collector::is_enabled());
    runtime.stop();
}

#[test]
fn shared_array_monomorphization_has_native_integer_float_and_string_strides() {
    use zyntax_compiler::hir::*;
    let mut program = zyntax_ts::parse_program(
        "function second<T>(xs:T[],fallback:T):T{if(xs.length>1)return xs[1];return fallback;}",
        "second.ts",
        Default::default(),
    )
    .unwrap();
    let mut context = LoweringContext::new(
        InternedString::new_global("strides"),
        Arc::new(program.type_registry.clone()),
        Arc::new(Mutex::new(AstArena::new())),
        LoweringConfig::default(),
    );
    let mut module = context.lower_program(&mut program).unwrap();
    let generic = module
        .functions
        .values()
        .find(|f| !f.signature.type_params.is_empty())
        .unwrap();
    let generic_id = generic.id;
    let header = generic.signature.params[0].ty.clone();
    let mut callers = Vec::new();
    for (name, ty) in [
        ("integer", HirType::I64),
        ("float", HirType::F64),
        ("string", HirType::Ptr(Arc::new(HirType::I8))),
        ("integer_again", HirType::I64),
    ] {
        let mut f = HirFunction::new(
            InternedString::new_global(name),
            HirFunctionSignature {
                params: [header.clone(), ty.clone()]
                    .into_iter()
                    .map(|ty| HirParam {
                        id: HirId::new(),
                        name: InternedString::new_global("p"),
                        ty,
                        attributes: Default::default(),
                        ownership: Default::default(),
                    })
                    .collect(),
                returns: vec![ty.clone()],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
                is_fiber: false,
                effects: vec![],
                is_pure: false,
            },
        );
        let args: Vec<_> = f
            .signature
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let id = HirId::new();
                f.values.insert(
                    id,
                    HirValue {
                        id,
                        ty: p.ty.clone(),
                        kind: HirValueKind::Parameter(i as u32),
                        span: None,
                    },
                );
                id
            })
            .collect();
        let result = HirId::new();
        f.values.insert(
            result,
            HirValue {
                id: result,
                ty: ty.clone(),
                kind: HirValueKind::Instruction,
                span: None,
            },
        );
        let block = f.blocks.get_mut(&f.entry_block).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Function(generic_id),
            args,
            type_args: vec![ty],
            const_args: vec![],
            is_tail: false,
        });
        block.terminator = HirTerminator::Return {
            values: vec![result],
        };
        callers.push(f.id);
        module.functions.insert(f.id, f);
    }
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    assert_eq!(
        module
            .functions
            .values()
            .filter(|f| f
                .name
                .resolve_global()
                .unwrap()
                .starts_with("ts$fn$second__mono_"))
            .count(),
        3
    );
    let mut backend = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
    backend.compile_module(&module).unwrap();
    let pointers: Vec<_> = callers
        .iter()
        .map(|id| backend.get_function_ptr(*id).expect("specialized caller"))
        .collect();
    let integer: unsafe extern "C" fn(*const usize, i64) -> i64 =
        unsafe { std::mem::transmute(pointers[0]) };
    let float: unsafe extern "C" fn(*const usize, f64) -> f64 =
        unsafe { std::mem::transmute(pointers[1]) };
    let text: unsafe extern "C" fn(*const usize, *const u8) -> *const u8 =
        unsafe { std::mem::transmute(pointers[2]) };
    let ints = [-7i64, 42];
    let floats = [1.5f64, 4.25];
    let a = zrtl::string_new("a");
    let b = zrtl::string_new("b");
    let strings = [a, b];
    let header = |data: usize| [data, 2, 2];
    unsafe {
        assert_eq!(integer(header(ints.as_ptr() as usize).as_ptr(), 9), 42);
        assert_eq!(float(header(floats.as_ptr() as usize).as_ptr(), 9.0), 4.25);
        assert_eq!(
            text(header(strings.as_ptr() as usize).as_ptr(), std::ptr::null()),
            b.cast()
        );
        assert_eq!(integer([0, 0, 0].as_ptr(), 9), 9);
        zrtl::string_free(a);
        zrtl::string_free(b);
    }
}
