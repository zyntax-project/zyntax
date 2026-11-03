//! Symbol resolution follows registration changes without probing unused sites.
mod common;

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use zyntax_compiler::hir::{
    HirCallable, HirFunction, HirFunctionSignature, HirInstruction, HirModule, HirParam,
    HirTerminator, HirType, HirValueKind,
};
use zyntax_compiler::hir_interp::{HirInterpreter, SharedSymbols, SymbolEntry};
use zyntax_compiler::value::ZyntaxValue;
use zyntax_typed_ast::InternedString;

extern "C" fn seven() -> i64 {
    7
}
extern "C" fn eleven() -> i64 {
    11
}

fn fixture(sites: usize) -> HirModule {
    let mut f = HirFunction::new(
        InternedString::new_global("symbols"),
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
    let take_cold = f.create_value(HirType::Bool, HirValueKind::Parameter(0));
    f.signature.params.push(HirParam {
        id: take_cold,
        name: InternedString::new_global("take_cold"),
        ty: HirType::Bool,
        attributes: Default::default(),
        ownership: Default::default(),
    });
    let cold = f.create_block();
    let hot = f.create_block();
    f.blocks.get_mut(&f.entry_block).unwrap().terminator = HirTerminator::CondBranch {
        condition: take_cold,
        true_target: cold,
        false_target: hot,
    };
    for (block, count) in [(cold, sites), (hot, 1)] {
        for i in 0..count {
            let result = f.create_value(HirType::I64, HirValueKind::Instruction);
            let b = f.blocks.get_mut(&block).unwrap();
            b.instructions.push(HirInstruction::Call {
                result: Some(result),
                callee: HirCallable::Symbol(format!("probe_symbol_{i}")),
                args: vec![],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            });
            b.terminator = HirTerminator::Return {
                values: vec![result],
            };
        }
    }
    let mut module = HirModule::new(InternedString::new_global("symbol_test"));
    module.functions.insert(f.id, f);
    module
}

fn invoke(interp: &mut HirInterpreter, module: &HirModule, cold: bool) -> ZyntaxValue {
    interp
        .call(module, "symbols", vec![ZyntaxValue::Bool(cold)])
        .unwrap()
}

fn shared(ptr: *const u8) -> Arc<SharedSymbols> {
    let entry = SymbolEntry {
        ptr,
        param_count: 0,
        sig: None,
    };
    Arc::new(LazyLock::new(Box::new(move || {
        HashMap::from([("probe_symbol_0".into(), entry)])
    })))
}

#[test]
fn symbol_replacement_is_visible_to_cached_bytecode() {
    let module = fixture(1);
    let mut interp = HirInterpreter::new();
    interp.register_symbol("probe_symbol_0", seven as *const u8, 0);
    assert_eq!(invoke(&mut interp, &module, false), ZyntaxValue::Int(7));
    interp.register_symbol("probe_symbol_0", eleven as *const u8, 0);
    assert_eq!(invoke(&mut interp, &module, false), ZyntaxValue::Int(11));
}

#[test]
fn shared_replacement_preserves_shared_precedence() {
    let module = fixture(1);
    let mut interp = HirInterpreter::new();
    interp.register_symbol("probe_symbol_0", eleven as *const u8, 0);
    let first = shared(seven as *const u8);
    interp.share_symbols(Arc::clone(&first));
    assert_eq!(invoke(&mut interp, &module, false), ZyntaxValue::Int(7));
    interp.share_symbols(first);
    assert_eq!(invoke(&mut interp, &module, false), ZyntaxValue::Int(7));
    interp.share_symbols(shared(eleven as *const u8));
    assert_eq!(invoke(&mut interp, &module, false), ZyntaxValue::Int(11));
}

#[test]
fn a_missing_symbol_can_be_registered_after_bytecode_was_cached() {
    let module = fixture(2);
    let mut interp = HirInterpreter::new();
    interp.register_symbol("probe_symbol_0", seven as *const u8, 0);
    assert_eq!(invoke(&mut interp, &module, false), ZyntaxValue::Int(7));
    assert!(
        interp
            .call(&module, "symbols", vec![ZyntaxValue::Bool(true)])
            .is_err()
    );
    interp.register_symbol("probe_symbol_1", eleven as *const u8, 0);
    assert_eq!(invoke(&mut interp, &module, true), ZyntaxValue::Int(11));
}

#[test]
#[ignore = "release-only interpreter symbol dispatch measurement"]
fn measure_symbol_frames() {
    use std::hint::black_box;
    let repeats: usize = std::env::var("INTERP_BENCH_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    println!(
        "value_bytes={} op_bytes={}",
        std::mem::size_of::<ZyntaxValue>(),
        std::mem::size_of::<zyntax_compiler::hir_interp::Op>()
    );
    for (roots, sites) in [(false, 1), (false, 24), (true, 1), (true, 24)] {
        if roots {
            zyntax_compiler::collector::enable();
        }
        let module = fixture(sites);
        let mut interp = HirInterpreter::new();
        for i in 0..sites {
            interp.register_symbol(format!("probe_symbol_{i}"), seven as *const u8, 0);
        }
        assert_eq!(invoke(&mut interp, &module, false), ZyntaxValue::Int(7));
        let start = std::time::Instant::now();
        for _ in 0..repeats {
            let result = invoke(&mut interp, &module, black_box(false));
            assert_eq!(black_box(result), ZyntaxValue::Int(7));
        }
        println!(
            "roots={roots} symbols={sites} repeats={repeats} ns_per_call={:.2}",
            start.elapsed().as_nanos() as f64 / repeats as f64
        );
        if roots {
            zyntax_compiler::collector::disable();
        }
    }
}

#[test]
fn untaken_symbol_sites_do_not_initialize_the_shared_table() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut module = fixture(24);
    let f = module.functions.values_mut().next().unwrap();
    let HirTerminator::CondBranch {
        false_target: hot, ..
    } = f.blocks[&f.entry_block].terminator
    else {
        panic!("branch")
    };
    let result = f.create_value(
        HirType::I64,
        HirValueKind::Constant(zyntax_compiler::hir::HirConstant::I64(7)),
    );
    let block = f.blocks.get_mut(&hot).unwrap();
    block.instructions.clear();
    block.terminator = HirTerminator::Return {
        values: vec![result],
    };
    let initialized = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&initialized);
    let symbols: Arc<SharedSymbols> = Arc::new(LazyLock::new(Box::new(move || {
        observed.fetch_add(1, Ordering::Relaxed);
        HashMap::new()
    })));
    let mut interp = HirInterpreter::new();
    interp.share_symbols(symbols);
    assert_eq!(invoke(&mut interp, &module, false), ZyntaxValue::Int(7));
    assert_eq!(initialized.load(Ordering::Relaxed), 0);
}

#[test]
fn typed_replacement_invalidates_a_cached_untyped_entry() {
    use zyntax_compiler::zrtl::{MAX_PARAMS, TypeTag, ZrtlSigFlags, ZrtlSymbolSig};
    let module = fixture(1);
    let mut interp = HirInterpreter::new();
    interp.register_symbol("probe_symbol_0", seven as *const u8, 0);
    assert_eq!(invoke(&mut interp, &module, false), ZyntaxValue::Int(7));
    let sig = ZrtlSymbolSig {
        param_count: 0,
        flags: ZrtlSigFlags::NONE,
        return_type: TypeTag::I64,
        params: [TypeTag::VOID; MAX_PARAMS],
    };
    interp.register_symbol_typed("probe_symbol_0", eleven as *const u8, sig);
    assert_eq!(invoke(&mut interp, &module, false), ZyntaxValue::Int(11));
}

#[test]
#[ignore = "release-only scalar interpreter control measurement"]
fn measure_scalar_loop() {
    let (f, _) = common::counted_loop();
    let mut module = HirModule::new(InternedString::new_global("scalar_control"));
    module.functions.insert(f.id, f);
    let mut interp = HirInterpreter::new();
    let mut run = || {
        let result = interp
            .call(
                &module,
                "count_to",
                vec![ZyntaxValue::Int(std::hint::black_box(10_000))],
            )
            .unwrap();
        assert_eq!(result.as_int(), Some(49_995_000));
    };
    run();
    let start = std::time::Instant::now();
    for _ in 0..100 {
        run();
    }
    println!(
        "scalar_iterations=1000000 ns_per_iteration={:.2}",
        start.elapsed().as_nanos() as f64 / 1_000_000.0
    );
}

#[test]
#[ignore = "release-only cold bytecode control measurement"]
fn measure_cold_symbol_frames() {
    let module = fixture(24);
    let mut elapsed = std::time::Duration::ZERO;
    for _ in 0..1_000 {
        let mut interp = HirInterpreter::new();
        for i in 0..24 {
            interp.register_symbol(format!("probe_symbol_{i}"), seven as *const u8, 0);
        }
        let start = std::time::Instant::now();
        let value = invoke(&mut interp, &module, false);
        elapsed += start.elapsed();
        assert_eq!(value, ZyntaxValue::Int(7));
    }
    println!(
        "cold_symbols=24 ns_per_call={:.2}",
        elapsed.as_nanos() as f64 / 1_000.0
    );
}
