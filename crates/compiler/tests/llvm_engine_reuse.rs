//! Repeated installs keep old code and data alive and resolve fresh symbols.
#![cfg(feature = "llvm-backend")]
mod common;

use inkwell::context::Context;
use std::sync::Arc;
use zyntax_compiler::{hir::*, llvm_jit_backend::LLVMJitBackend, osr, reload};
use zyntax_typed_ast::InternedString;

extern "C" fn plus_ten(n: i64) -> i64 {
    n + 10
}
extern "C" fn plus_twenty(n: i64) -> i64 {
    n + 20
}

fn global_reader() -> (HirModule, HirId, HirId, HirId) {
    let mut signature = common::counted_loop().0.signature;
    signature.params.clear();
    signature.returns = vec![HirType::I64];
    signature.is_pure = false;
    let mut host = HirFunction::new(
        InternedString::new_global("zyntax_engine_test_host"),
        signature.clone(),
    );
    let arg = host.create_value(HirType::I64, HirValueKind::Parameter(0));
    host.signature.params.push(HirParam {
        id: arg,
        name: InternedString::new_global("value"),
        ty: HirType::I64,
        attributes: Default::default(),
        ownership: Default::default(),
    });
    host.is_external = true;
    let mut f = HirFunction::new(InternedString::new_global("main"), signature);
    let id = f.id;
    let global = HirId::new();
    let ptr = f.create_value(
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Global(global),
    );
    let loaded = f.create_value(HirType::I64, HirValueKind::Instruction);
    let result = f.create_value(HirType::I64, HirValueKind::Instruction);
    let entry = f.blocks.get_mut(&f.entry_block).unwrap();
    entry.instructions = vec![
        HirInstruction::Load {
            result: loaded,
            ty: HirType::I64,
            ptr,
            align: 8,
            volatile: true,
        },
        HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Function(host.id),
            args: vec![loaded],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
    ];
    entry.terminator = HirTerminator::Return {
        values: vec![result],
    };
    let mut twin = f.clone();
    twin.id = HirId::new();
    twin.name = InternedString::new_global("twin");
    let twin_id = twin.id;
    let mut m = HirModule::new(InternedString::new_global("engine_reuse"));
    m.functions.insert(id, f);
    m.functions.insert(twin.id, twin);
    m.functions.insert(host.id, host);
    m.globals.insert(
        global,
        HirGlobal {
            id: global,
            name: InternedString::new_global("state"),
            ty: HirType::I64,
            initializer: Some(HirConstant::I64(1)),
            is_const: false,
            is_thread_local: false,
            linkage: Linkage::External,
            visibility: Visibility::Default,
            error_flag: false,
        },
    );
    (m, id, twin_id, global)
}

#[test]
fn repeated_modules_keep_private_data_and_host_bindings() {
    let (mut module, id, twin, global) = global_reader();
    let context = Context::create();
    let mut backend = LLVMJitBackend::new(&context).unwrap();
    backend.set_use_mcjit(true);
    let mut installed = Vec::new();
    for generation in 0..12 {
        let offset = if generation % 2 == 0 { 10 } else { 20 };
        let host = if offset == 10 {
            plus_ten as *const u8
        } else {
            plus_twenty as *const u8
        };
        backend.register_symbol("zyntax_engine_test_host", host);
        module.globals.get_mut(&global).unwrap().initializer = Some(HirConstant::I64(generation));
        backend.compile_module(&module).unwrap();
        for function in [id, twin] {
            // SAFETY: both fixture entries have the C () -> i64 signature.
            let call: extern "C" fn() -> i64 =
                unsafe { std::mem::transmute(backend.get_function_pointer(function).unwrap()) };
            installed.push((call, generation + offset));
        }
        for (call, expected) in &installed {
            assert_eq!(call(), *expected);
        }
    }
    std::thread::spawn(move || {
        for (call, expected) in installed {
            assert_eq!(call(), expected);
        }
    })
    .join()
    .unwrap();
}

#[test]
fn shared_globals_can_rebind_without_changing_old_code() {
    let (module, id, _, global) = global_reader();
    let context = Context::create();
    let mut backend = LLVMJitBackend::new(&context).unwrap();
    backend.set_use_mcjit(true);
    backend.register_symbol("zyntax_engine_test_host", plus_ten as *const u8);
    backend.set_module_context(Arc::new(module.clone()));
    let mut storage = Vec::new();
    let mut installed = Vec::new();
    for generation in 0i64..8 {
        let value = Box::new(generation);
        let address = value.as_ref() as *const i64 as usize;
        storage.push(value);
        backend.set_cross_tier_links(
            reload::next_backend_key(),
            Arc::new(move |g| (g == global).then_some(address)),
        );
        backend
            .compile_function(id, &module.functions[&id])
            .unwrap();
        // SAFETY: the fixture entry has the C () -> i64 signature.
        let call: extern "C" fn() -> i64 =
            unsafe { std::mem::transmute(backend.get_function_pointer(id).unwrap()) };
        installed.push(call);
        for (index, call) in installed.iter().enumerate() {
            assert_eq!(call(), index as i64 + 10);
        }
    }
    *storage[0] = 100;
    assert_eq!(installed[0](), 110);
}

#[test]
fn unresolved_install_does_not_poison_the_engine() {
    let (mut module, id, _, _) = global_reader();
    let context = Context::create();
    let mut backend = LLVMJitBackend::new(&context).unwrap();
    backend.set_use_mcjit(true);
    backend.register_symbol("zyntax_engine_test_host", plus_ten as *const u8);
    backend.compile_module(&module).unwrap();
    // SAFETY: the fixture entry has the C () -> i64 signature.
    let old: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(backend.get_function_pointer(id).unwrap()) };
    let missing = "zyntax_engine_test_missing_host";
    module
        .functions
        .values_mut()
        .find(|f| f.is_external)
        .unwrap()
        .name = InternedString::new_global(missing);
    assert!(
        backend
            .compile_module(&module)
            .unwrap_err()
            .to_string()
            .contains(missing)
    );
    assert_eq!(old(), 11);
    backend.register_symbol(missing, plus_twenty as *const u8);
    backend.compile_module(&module).unwrap();
    // SAFETY: the fixture entry has the C () -> i64 signature.
    let new: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(backend.get_function_pointer(id).unwrap()) };
    assert_eq!(new(), 21);
    assert_eq!(old(), 11);
}

#[test]
fn repeated_entries_and_osr_helpers_resolve_the_current_body() {
    let (mut function, header) = common::counted_loop();
    let layout = osr::osr_layout(&function, header).unwrap();
    let id = function.id;
    let sum = function.blocks[&header].phis[1].result;
    let next_sum = function.blocks[&header].phis[1].incoming[1].0;
    let counter = function.blocks[&header].phis[0].result;
    let n = function.signature.params[0].id;
    let mut frame = vec![0u64; (layout.frame.size as usize).div_ceil(8)];
    for (v, offset) in layout.live_ins.iter().zip(&layout.frame.offsets) {
        let value: i32 = if *v == n {
            5
        } else if *v == counter {
            2
        } else if *v == sum {
            1
        } else {
            panic!("unexpected live-in")
        };
        // SAFETY: the frame has the layout's size and alignment.
        unsafe {
            frame
                .as_mut_ptr()
                .cast::<u8>()
                .add(*offset as usize)
                .cast::<i32>()
                .write(value);
        }
    }
    let context = Context::create();
    let mut backend = LLVMJitBackend::new(&context).unwrap();
    backend.set_use_mcjit(true);
    backend.set_compile_tier(1);
    backend.set_osr_helper_sites(Some([layout.site_key()].into_iter().collect()));
    let mut installed = Vec::new();
    for subtract in [false, true, false, true] {
        for block in function.blocks.values_mut() {
            for inst in &mut block.instructions {
                if let HirInstruction::Binary { result, op, .. } = inst {
                    if *result == next_sum {
                        *op = if subtract {
                            BinaryOp::Sub
                        } else {
                            BinaryOp::Add
                        };
                    }
                }
            }
        }
        let mut module = HirModule::new(InternedString::new_global("repeated_osr"));
        module.functions.insert(id, function.clone());
        backend.compile_module(&module).unwrap();
        let helpers = backend.take_pending_osr_helpers();
        assert_eq!(helpers.len(), 1);
        // SAFETY: entry is i32 -> i32; the helper takes the corresponding frame.
        let entry: extern "C" fn(i32) -> i32 =
            unsafe { std::mem::transmute(backend.get_function_pointer(id).unwrap()) };
        let helper: extern "C" fn(*const u8) -> i32 = unsafe { std::mem::transmute(helpers[0].2) };
        installed.push((entry, helper, subtract));
        for (entry, helper, subtract) in &installed {
            assert_eq!(entry(5), if *subtract { -10 } else { 10 });
            assert_eq!(
                helper(frame.as_ptr().cast()),
                if *subtract { -8 } else { 10 }
            );
        }
    }
}

#[test]
fn installing_modules_while_old_llvm_code_runs_keeps_its_pages_executable() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    struct Stop<'a>(&'a AtomicBool);
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }
    let (mut module, id, _, global) = global_reader();
    let context = Context::create();
    let mut backend = LLVMJitBackend::new(&context).unwrap();
    backend.set_use_mcjit(true);
    backend.register_symbol("zyntax_engine_test_host", plus_ten as *const u8);
    backend.compile_module(&module).unwrap();
    // SAFETY: the fixture entry has the C () -> i64 signature.
    let old: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(backend.get_function_pointer(id).unwrap()) };
    let running = AtomicBool::new(true);
    let calls = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        // Stop the worker before the scope joins it, including during unwinding.
        let _stop = Stop(&running);
        scope.spawn(|| {
            while running.load(Ordering::Acquire) {
                assert_eq!(old(), 11);
                calls.fetch_add(1, Ordering::Relaxed);
            }
        });
        while calls.load(Ordering::Relaxed) == 0 {
            std::thread::yield_now();
        }
        for generation in 2..10 {
            module.globals.get_mut(&global).unwrap().initializer =
                Some(HirConstant::I64(generation));
            backend.compile_module(&module).unwrap();
            // SAFETY: the fixture entry has the C () -> i64 signature.
            let new: extern "C" fn() -> i64 =
                unsafe { std::mem::transmute(backend.get_function_pointer(id).unwrap()) };
            assert_eq!(new(), generation + 10);
        }
    });
    assert!(calls.load(Ordering::Relaxed) > 0);
}
