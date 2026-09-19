//! A loop in a function returning a struct through a destination can be
//! transferred like any other.
//!
//! The destination pointer travels in the OSR frame; the helper writes
//! its result through it and returns it, which is what the function's
//! own return does. Covered for the tier-0 probe (a stub helper proves
//! the frame carried the destination), for the real tier-1 helper (the
//! loop it inherits completes with the right pair), and for an
//! interpreted frame.

#![cfg(feature = "cranelift-backend")]

use std::collections::HashSet;
use std::sync::Arc;
use zyntax_compiler::beadie_adapter::ZyntaxCraneliftBackend;
use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::{
    BinaryOp, HirConstant, HirFunction, HirFunctionSignature, HirId, HirInstruction, HirModule,
    HirParam, HirPhi, HirStructType, HirTerminator, HirType, HirValue, HirValueKind,
    ParamAttributes,
};
use zyntax_compiler::osr;
use zyntax_typed_ast::InternedString;

fn pair_ty() -> HirType {
    HirType::Struct(HirStructType {
        name: Some(InternedString::new_global("Pair")),
        fields: vec![HirType::I64, HirType::I64],
        packed: false,
    })
}

fn sig(params: Vec<HirType>, returns: Vec<HirType>) -> HirFunctionSignature {
    HirFunctionSignature {
        params: params
            .into_iter()
            .enumerate()
            .map(|(i, ty)| HirParam {
                id: HirId::new(),
                name: InternedString::new_global(&format!("p{}", i)),
                ty,
                attributes: ParamAttributes::default(),
                ownership: Default::default(),
            })
            .collect(),
        returns,
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    }
}

fn add_value(func: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty,
            kind,
            uses: HashSet::new(),
            span: None,
        },
    );
    id
}

fn konst(func: &mut HirFunction, v: i64) -> HirId {
    add_value(
        func,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(v)),
    )
}

fn arith(op: BinaryOp, result: HirId, left: HirId, right: HirId) -> HirInstruction {
    HirInstruction::Binary {
        op,
        result,
        ty: HirType::I64,
        left,
        right,
    }
}

/// `def sums(n: i64): Pair { a = 0; b = 0; for i in 0..n { a += i; b += 2*i }
/// return Pair { a, b } }`, with the header returned.
fn build_sums() -> (HirFunction, HirId) {
    let mut f = HirFunction::new(
        InternedString::new_global("sums"),
        sig(vec![HirType::I64], vec![pair_ty()]),
    );
    let n = add_value(&mut f, HirType::I64, HirValueKind::Parameter(0));
    let zero = konst(&mut f, 0);
    let one = konst(&mut f, 1);
    let two = konst(&mut f, 2);
    let i = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let a = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let b = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let cond = add_value(&mut f, HirType::Bool, HirValueKind::Instruction);
    let a2 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let twice = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let b2 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let i2 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let undef = add_value(&mut f, pair_ty(), HirValueKind::Undef);
    let with_a = add_value(&mut f, pair_ty(), HirValueKind::Instruction);
    let pair = add_value(&mut f, pair_ty(), HirValueKind::Instruction);

    let entry = f.entry_block;
    let header = f.create_block();
    let body = f.create_block();
    let exit = f.create_block();
    f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: header };
    {
        let blk = f.blocks.get_mut(&header).unwrap();
        blk.phis.push(HirPhi {
            result: i,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (i2, body)],
        });
        blk.phis.push(HirPhi {
            result: a,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (a2, body)],
        });
        blk.phis.push(HirPhi {
            result: b,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (b2, body)],
        });
        blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Lt,
            result: cond,
            ty: HirType::Bool,
            left: i,
            right: n,
        });
        blk.terminator = HirTerminator::CondBranch {
            condition: cond,
            true_target: body,
            false_target: exit,
        };
    }
    {
        let blk = f.blocks.get_mut(&body).unwrap();
        blk.instructions.push(arith(BinaryOp::Add, a2, a, i));
        blk.instructions.push(arith(BinaryOp::Mul, twice, i, two));
        blk.instructions.push(arith(BinaryOp::Add, b2, b, twice));
        blk.instructions.push(arith(BinaryOp::Add, i2, i, one));
        blk.terminator = HirTerminator::Branch { target: header };
    }
    {
        let blk = f.blocks.get_mut(&exit).unwrap();
        blk.instructions.push(HirInstruction::InsertValue {
            result: with_a,
            ty: pair_ty(),
            aggregate: undef,
            value: a,
            indices: vec![0],
        });
        blk.instructions.push(HirInstruction::InsertValue {
            result: pair,
            ty: pair_ty(),
            aggregate: with_a,
            value: b,
            indices: vec![1],
        });
        blk.terminator = HirTerminator::Return { values: vec![pair] };
    }
    (f, header)
}

/// The pair `sums(n)` produces: (sum of i, twice that).
fn expected(n: i64) -> [i64; 2] {
    let s = n * (n - 1) / 2;
    [s, 2 * s]
}

/// The destination a Cranelift tier-0 function of this shape takes first.
type Sums = extern "C" fn(*mut u8, i64) -> *mut u8;

static STUB_DESTINATION_OFFSET: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// A helper that ignores the loop and writes a marker pair through the
/// destination it finds in the frame.
extern "C" fn stub_helper(frame: *mut u8) -> *mut u8 {
    let at = STUB_DESTINATION_OFFSET.load(std::sync::atomic::Ordering::Acquire) as usize;
    // SAFETY: the probe stored the destination at this offset.
    unsafe {
        let dest = std::ptr::read_unaligned(frame.add(at) as *const *mut u8);
        std::ptr::write_unaligned(dest as *mut i64, 111);
        std::ptr::write_unaligned(dest.add(8) as *mut i64, 222);
        dest
    }
}

#[test]
fn the_layout_carries_the_destination_and_returns_the_pointer() {
    let (function, header) = build_sums();
    let layout = osr::osr_layout(&function, header).expect("a layout");
    assert!(layout.destination.is_some());
    assert!(matches!(layout.return_type, HirType::Ptr(_)));
    let at = layout.destination.unwrap();
    assert!(at as usize + 8 <= layout.frame.size as usize);
    assert!(layout.frame.offsets.iter().all(|&o| o + 8 <= at));
}

#[test]
fn a_tier0_probe_hands_the_destination_to_the_helper() {
    const BEAD: u64 = 0xDE57;
    let (function, header) = build_sums();
    let func_id = function.id;
    let layout = osr::osr_layout(&function, header).expect("a layout");
    STUB_DESTINATION_OFFSET.store(
        layout.destination.expect("a destination"),
        std::sync::atomic::Ordering::Release,
    );
    let site = layout.site_key();

    let osr_syms = osr::osr_runtime_symbols();
    let mut backend = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");
    backend.set_compile_tier(0);
    backend.set_compile_bead_id(BEAD);
    backend
        .compile_function(func_id, &function)
        .expect("tier-0 compile");
    backend.finalize_definitions().expect("finalize");
    let f: Sums = unsafe { std::mem::transmute(backend.get_function_ptr(func_id).unwrap()) };

    let mut out = [0i64; 2];
    let got = f(out.as_mut_ptr() as *mut u8, 10);
    assert_eq!(
        got,
        out.as_mut_ptr() as *mut u8,
        "the function returns its destination"
    );
    assert_eq!(out, expected(10), "tier 0 alone");

    osr::publish_helper(BEAD, site, stub_helper as *mut ());
    let mut out = [0i64; 2];
    let got = f(out.as_mut_ptr() as *mut u8, 10);
    assert_eq!(
        got,
        out.as_mut_ptr() as *mut u8,
        "the helper's result is the function's"
    );
    assert_eq!(out, [111, 222], "the helper wrote through the destination");
}

#[test]
fn the_tier1_helper_completes_the_loop_and_returns_the_pair() {
    const BEAD: u64 = 0xDE58;
    let (function, header) = build_sums();
    let func_id = function.id;
    let site = osr::osr_layout(&function, header)
        .expect("a layout")
        .site_key();

    let osr_syms = osr::osr_runtime_symbols();
    let mut backend = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");
    backend.set_compile_tier(0);
    backend.set_compile_bead_id(BEAD);
    backend
        .compile_function(func_id, &function)
        .expect("tier-0 compile");
    backend.finalize_definitions().expect("finalize tier 0");
    let f: Sums = unsafe { std::mem::transmute(backend.get_function_ptr(func_id).unwrap()) };

    backend.set_compile_tier(1);
    backend
        .compile_function(func_id, &function)
        .expect("tier-1 compile");
    backend.finalize_definitions().expect("finalize tier 1");
    let (helper_site, helper_code) = backend
        .take_pending_osr_helpers()
        .into_iter()
        .find(|(s, _)| *s == site)
        .expect("tier 1 emits a helper for the header");
    osr::publish_helper(BEAD, helper_site, helper_code);

    for n in [10, 100, 1000] {
        let mut out = [0i64; 2];
        let got = f(out.as_mut_ptr() as *mut u8, n);
        assert_eq!(got, out.as_mut_ptr() as *mut u8);
        assert_eq!(out, expected(n), "sums({n}) through the tier-1 helper");
    }
}

#[test]
fn an_interpreted_frame_transfers_with_its_destination() {
    use zyntax_compiler::hir_interp::HirInterpreter;

    const BEAD: u64 = 0xDE59;
    let (function, header) = build_sums();
    let func_id = function.id;
    let site = osr::osr_layout(&function, header)
        .expect("a layout")
        .site_key();
    let mut module = HirModule::new(InternedString::new_global("sums"));
    module.functions.insert(func_id, function.clone());

    // A tier-1 helper to transfer into.
    let osr_syms = osr::osr_runtime_symbols();
    let mut backend = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");
    backend.set_compile_tier(1);
    backend.set_compile_bead_id(BEAD);
    backend
        .compile_function(func_id, &function)
        .expect("tier-1 compile");
    backend.finalize_definitions().expect("finalize");
    let (helper_site, helper_code) = backend
        .take_pending_osr_helpers()
        .into_iter()
        .find(|(s, _)| *s == site)
        .expect("a helper for the header");
    osr::publish_helper(BEAD, helper_site, helper_code);

    // The interpreter's bridge: thunks from the same backend, no native
    // entries, and the bead every frame of the function ticks.
    let shared = Arc::new(ZyntaxCraneliftBackend::new(backend));
    let for_thunks = Arc::clone(&shared);
    let mut interp = HirInterpreter::new();
    interp.set_native_bridge(
        Box::new(move |sig| for_thunks.with_lock(|be| be.interp_thunk(sig).ok())),
        Box::new(|_| None),
        Box::new(move |id| (id == func_id).then_some(BEAD)),
    );

    // Long enough to pass the visits a frame waits before asking.
    let n = 100_000i64;
    let result = interp
        .call(
            &module,
            "sums",
            vec![zyntax_compiler::value::ZyntaxValue::Int(n)],
        )
        .expect("run");
    let zyntax_compiler::value::ZyntaxValue::Pointer(p) = result else {
        panic!("a struct comes back as the pointer to its storage, got {result:?}");
    };
    // SAFETY: the interpreter's destination holds the pair.
    let got = unsafe { [*(p as *const i64), *(p as *const i64).add(1)] };
    assert_eq!(got, expected(n));
}
