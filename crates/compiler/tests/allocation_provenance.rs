#![cfg(feature = "cranelift-backend")]

use zyntax_compiler::{hir::*, load_cse};
use zyntax_typed_ast::InternedString;

fn ptr() -> HirType {
    HirType::Ptr(std::sync::Arc::new(HirType::I64))
}
fn val(f: &mut HirFunction, ty: HirType) -> HirId {
    f.create_value(ty, HirValueKind::Instruction)
}
fn int(f: &mut HirFunction, n: i64) -> HirId {
    f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(n)))
}
fn store(ptr: HirId, value: HirId) -> HirInstruction {
    HirInstruction::Store {
        ptr,
        value,
        align: 8,
        volatile: false,
    }
}
fn load(result: HirId, ptr: HirId, ty: HirType) -> HirInstruction {
    HirInstruction::Load {
        result,
        ptr,
        ty,
        align: 8,
        volatile: false,
    }
}

struct Fixture {
    module: HirModule,
    function: HirId,
    global: HirId,
}
fn fixture(allocated: bool, loaded_alias: bool) -> Fixture {
    let mut f = HirFunction::new(
        InternedString::new_global("f"),
        HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I64],
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
    f.attributes.osr_region = true;
    let args: Vec<_> = (0..2)
        .map(|i| {
            let id = f.create_value(ptr(), HirValueKind::Parameter(i));
            f.signature.params.push(HirParam {
                id,
                name: InternedString::new_global("p"),
                ty: ptr(),
                attributes: ParamAttributes {
                    allocation_backed: allocated,
                    ..Default::default()
                },
                ownership: Default::default(),
            });
            id
        })
        .collect();
    let global = HirId::new();
    let global_ty = if loaded_alias { ptr() } else { HirType::I64 };
    let address = f.create_value(
        HirType::Ptr(std::sync::Arc::new(global_ty.clone())),
        HirValueKind::Global(global),
    );
    let one = int(&mut f, 1);
    let two = int(&mut f, 2);
    let seven = int(&mut f, 7);
    let first = val(&mut f, HirType::I64);
    let second = val(&mut f, HirType::I64);
    let sum = val(&mut f, HirType::I64);
    let alias = val(&mut f, ptr());
    let block = f.blocks.get_mut(&f.entry_block).unwrap();
    if loaded_alias {
        block.instructions.extend([
            load(alias, address, ptr()),
            store(args[0], one),
            store(alias, seven),
            load(sum, args[0], HirType::I64),
        ]);
    } else {
        block.instructions.extend([
            store(args[0], one),
            store(address, two),
            load(first, args[0], HirType::I64),
            store(args[1], seven),
            load(second, args[0], HirType::I64),
            HirInstruction::Binary {
                result: sum,
                op: BinaryOp::Add,
                ty: HirType::I64,
                left: first,
                right: second,
            },
        ]);
    }
    block.terminator = HirTerminator::Return { values: vec![sum] };
    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("allocated"));
    module.add_function(f);
    module.globals.insert(
        global,
        HirGlobal {
            id: global,
            name: InternedString::new_global("g"),
            ty: global_ty,
            initializer: Some(if loaded_alias {
                HirConstant::Null(ptr())
            } else {
                HirConstant::I64(0)
            }),
            is_const: false,
            is_thread_local: false,
            linkage: Linkage::Internal,
            visibility: Visibility::Default,
            error_flag: false,
        },
    );
    Fixture {
        module,
        function: id,
        global,
    }
}

fn execute(a: &Fixture, alias_args: bool, global_arg: bool, loaded_alias: bool) -> i64 {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let mut backend = CraneliftBackend::new().unwrap();
    backend.compile_module(&a.module).unwrap();
    backend.finalize_definitions().unwrap();
    let (global, _) = backend.global_data_addr(a.global).unwrap();
    let mut x = 0i64;
    let mut y = 0i64;
    let p = if global_arg {
        global as *mut i64
    } else {
        &mut x as *mut i64
    };
    let q = if alias_args { p } else { &mut y as *mut i64 };
    // SAFETY: the fixture declares a pointer-sized global when loaded_alias is set.
    if loaded_alias {
        unsafe {
            (global as *mut *mut i64).write(p);
        }
    }
    // SAFETY: this fixture has exactly the fn(*mut i64, *mut i64) -> i64 ABI.
    let call: extern "C" fn(*mut i64, *mut i64) -> i64 =
        unsafe { std::mem::transmute(backend.get_function_ptr(a.function).unwrap()) };
    call(p, q)
}

#[cfg(feature = "llvm-backend")]
fn execute_llvm(a: &Fixture, alias_args: bool) -> i64 {
    use inkwell::{OptimizationLevel, context::Context};
    use zyntax_compiler::llvm_backend::LLVMBackend;
    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "allocated");
    backend.compile_module(&a.module).unwrap();
    backend.module().verify().unwrap();
    let engine = backend
        .module()
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();
    let mut x = 0i64;
    let mut y = 0i64;
    let p = &mut x as *mut i64;
    let q = if alias_args { p } else { &mut y };
    // SAFETY: fixture signature and scalar storage match the requested native ABI.
    let call = unsafe {
        engine
            .get_function::<unsafe extern "C" fn(*mut i64, *mut i64) -> i64>(&format!(
                "func_{:?}",
                a.function
            ))
            .unwrap()
    };
    unsafe { call.call(p, q) }
}

#[test]
fn allocated_parameters_forward_across_globals_but_may_alias_each_other() {
    let mut a = fixture(true, false);
    assert_eq!(execute(&a, true, false, false), 8);
    assert_eq!(execute(&a, false, false, false), 2);
    let stats = load_cse::run_regions(a.module.functions.get_mut(&a.function).unwrap());
    assert_eq!(
        stats.eliminated, 1,
        "only the load across static global storage forwards"
    );
    assert_eq!(execute(&a, true, false, false), 8);
    assert_eq!(execute(&a, false, false, false), 2);
    #[cfg(feature = "llvm-backend")]
    {
        assert_eq!(execute_llvm(&a, true), 8);
        assert_eq!(execute_llvm(&a, false), 2);
    }
}

#[test]
fn unknown_parameters_can_point_at_global_storage() {
    let mut a = fixture(false, false);
    assert_eq!(execute(&a, false, true, false), 4);
    assert_eq!(
        load_cse::run_regions(a.module.functions.get_mut(&a.function).unwrap()).eliminated,
        0
    );
    assert_eq!(execute(&a, false, true, false), 4);
}

#[test]
fn a_pointer_loaded_from_a_global_can_alias_allocated_storage() {
    let mut a = fixture(true, true);
    assert_eq!(execute(&a, false, false, true), 7);
    load_cse::run_regions(a.module.functions.get_mut(&a.function).unwrap());
    assert_eq!(execute(&a, false, false, true), 7);
}

#[test]
fn allocation_provenance_round_trips_in_all_bytecode_formats() {
    use zyntax_compiler::bytecode::{self, Format};
    let a = fixture(true, false);
    for format in [
        Format::Postcard,
        Format::Bincode,
        Format::Json,
        Format::Split,
    ] {
        let bytes = bytecode::serialize_module(&a.module, format).unwrap();
        let decoded = bytecode::deserialize_module(&bytes).unwrap();
        let function = decoded
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some("f"))
            .unwrap();
        assert!(
            function
                .signature
                .params
                .iter()
                .all(|p| p.attributes.allocation_backed && !p.attributes.noalias)
        );
    }
}
