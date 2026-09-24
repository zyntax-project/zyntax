//! A struct returned by value must outlive the call that produced it.
//!
//! Two calls to the same aggregate-returning function must yield two
//! independent values. The HIR says so plainly: `insertvalue` builds a
//! value and `return` hands it back, with no memory in the picture. Each
//! backend then picks its own representation for that value, and a
//! representation that borrows the callee's frame cannot survive the
//! return.
//!
//! The same module is run on every backend that can execute it, because
//! "aggregates are values" is a property of the IR, not of one target.
//! Testing one backend would only say which one was looked at.

#![cfg(feature = "cranelift-backend")]

use std::collections::HashSet;
use zyntax_compiler::hir::{
    BinaryOp, HirBlock, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirId,
    HirInstruction, HirModule, HirParam, HirStructType, HirTerminator, HirType, HirValue,
    HirValueKind, ParamAttributes,
};
use zyntax_typed_ast::InternedString;

/// `mk` is called with 1 and with 2, so the pair fields are 1/10 and
/// 2/20. Weighing each field by a different power of ten makes any
/// aliasing legible in the single number a test can assert on: the
/// right answer is 2040, and two names reaching one value gives 4040.
const EXPECTED: i64 = 2040;

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

fn body(func: &mut HirFunction) -> &mut HirBlock {
    let entry = func.entry_block;
    func.blocks.get_mut(&entry).unwrap()
}

/// `def mk(n: i64): Pair { return Pair { a: n, b: n * 10 } }`
fn build_mk() -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global("mk"),
        sig(vec![HirType::I64], vec![pair_ty()]),
    );

    let n = add_value(&mut f, HirType::I64, HirValueKind::Parameter(0));
    let ten = konst(&mut f, 10);
    let scaled = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let undef = add_value(&mut f, pair_ty(), HirValueKind::Undef);
    let with_a = add_value(&mut f, pair_ty(), HirValueKind::Instruction);
    let with_b = add_value(&mut f, pair_ty(), HirValueKind::Instruction);

    let blk = body(&mut f);
    blk.instructions.push(HirInstruction::Binary {
        op: BinaryOp::Mul,
        result: scaled,
        ty: HirType::I64,
        left: n,
        right: ten,
    });
    blk.instructions.push(HirInstruction::InsertValue {
        result: with_a,
        ty: pair_ty(),
        aggregate: undef,
        value: n,
        indices: vec![0],
    });
    blk.instructions.push(HirInstruction::InsertValue {
        result: with_b,
        ty: pair_ty(),
        aggregate: with_a,
        value: scaled,
        indices: vec![1],
    });
    blk.terminator = HirTerminator::Return {
        values: vec![with_b],
    };
    f
}

/// `def main(): i64 { let p = mk(1); let q = mk(2); return
/// p.a*1000 + p.b*100 + q.a*10 + q.b }`
fn build_main(mk_id: HirId) -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global("main"),
        sig(vec![], vec![HirType::I64]),
    );

    let one = konst(&mut f, 1);
    let two = konst(&mut f, 2);
    let k1000 = konst(&mut f, 1000);
    let k100 = konst(&mut f, 100);
    let k10 = konst(&mut f, 10);

    let p = add_value(&mut f, pair_ty(), HirValueKind::Instruction);
    let q = add_value(&mut f, pair_ty(), HirValueKind::Instruction);
    let pa = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let pb = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let qa = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let qb = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let t0 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let t1 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let t2 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let s0 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let s1 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let s2 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);

    let call = |result: HirId, arg: HirId| HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Function(mk_id),
        args: vec![arg],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };
    let extract = |result: HirId, agg: HirId, idx: u32| HirInstruction::ExtractValue {
        result,
        ty: HirType::I64,
        aggregate: agg,
        indices: vec![idx],
    };
    let arith = |op: BinaryOp, result: HirId, left: HirId, right: HirId| HirInstruction::Binary {
        op,
        result,
        ty: HirType::I64,
        left,
        right,
    };

    let blk = body(&mut f);
    // Both calls happen before either result is read, which is what
    // makes a shared destination observable: with one buffer between
    // them the second call overwrites the first before anyone looks.
    blk.instructions.push(call(p, one));
    blk.instructions.push(call(q, two));
    blk.instructions.push(extract(pa, p, 0));
    blk.instructions.push(extract(pb, p, 1));
    blk.instructions.push(extract(qa, q, 0));
    blk.instructions.push(extract(qb, q, 1));
    blk.instructions.push(arith(BinaryOp::Mul, t0, pa, k1000));
    blk.instructions.push(arith(BinaryOp::Mul, t1, pb, k100));
    blk.instructions.push(arith(BinaryOp::Mul, t2, qa, k10));
    blk.instructions.push(arith(BinaryOp::Add, s0, t0, t1));
    blk.instructions.push(arith(BinaryOp::Add, s1, s0, t2));
    blk.instructions.push(arith(BinaryOp::Add, s2, s1, qb));
    blk.terminator = HirTerminator::Return { values: vec![s2] };
    f
}

fn build_module() -> (HirModule, HirId) {
    let mk = build_mk();
    let mk_id = mk.id;
    let main = build_main(mk_id);
    let main_id = main.id;

    let mut module = HirModule::new(InternedString::new_global("struct_return"));
    module.functions.insert(mk_id, mk);
    module.functions.insert(main_id, main);
    (module, main_id)
}

/// `def echo(p: Pair): Pair { return p }` — a struct arrives by address,
/// so handing it straight back is the one case where the result could
/// come to share storage with an argument rather than with another
/// result.
fn build_echo() -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global("echo"),
        sig(vec![pair_ty()], vec![pair_ty()]),
    );
    let p = add_value(&mut f, pair_ty(), HirValueKind::Parameter(0));
    body(&mut f).terminator = HirTerminator::Return { values: vec![p] };
    f
}

/// `def main(): i64 { let a = mk(1); let b = echo(a); a.a*1000 + ... }`
/// with `a` overwritten in between, so a `b` that aliases `a` reports
/// the new contents rather than the ones it was handed.
fn build_echo_main(mk_id: HirId, echo_id: HirId) -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global("echo_main"),
        sig(vec![], vec![HirType::I64]),
    );

    let one = konst(&mut f, 1);
    let k1000 = konst(&mut f, 1000);
    let k100 = konst(&mut f, 100);
    let k10 = konst(&mut f, 10);
    let nine = konst(&mut f, 9);

    let a = add_value(&mut f, pair_ty(), HirValueKind::Instruction);
    let b = add_value(&mut f, pair_ty(), HirValueKind::Instruction);
    // Rebuilding `a` with different contents after `echo` has copied it.
    let a2 = add_value(&mut f, pair_ty(), HirValueKind::Instruction);
    let aa = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let ab = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let ba = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let bb = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let t0 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let t1 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let t2 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let s0 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let s1 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let s2 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);

    let extract = |result: HirId, agg: HirId, idx: u32| HirInstruction::ExtractValue {
        result,
        ty: HirType::I64,
        aggregate: agg,
        indices: vec![idx],
    };
    let arith = |op: BinaryOp, result: HirId, left: HirId, right: HirId| HirInstruction::Binary {
        op,
        result,
        ty: HirType::I64,
        left,
        right,
    };

    let blk = body(&mut f);
    blk.instructions.push(HirInstruction::Call {
        result: Some(a),
        callee: HirCallable::Function(mk_id),
        args: vec![one],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    blk.instructions.push(HirInstruction::Call {
        result: Some(b),
        callee: HirCallable::Function(echo_id),
        args: vec![a],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    // mk(9) = {9, 90}; a shared destination would drag `b` along with it.
    blk.instructions.push(HirInstruction::Call {
        result: Some(a2),
        callee: HirCallable::Function(mk_id),
        args: vec![nine],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    blk.instructions.push(extract(aa, a2, 0));
    blk.instructions.push(extract(ab, a2, 1));
    blk.instructions.push(extract(ba, b, 0));
    blk.instructions.push(extract(bb, b, 1));
    blk.instructions.push(arith(BinaryOp::Mul, t0, aa, k1000));
    blk.instructions.push(arith(BinaryOp::Mul, t1, ab, k100));
    blk.instructions.push(arith(BinaryOp::Mul, t2, ba, k10));
    blk.instructions.push(arith(BinaryOp::Add, s0, t0, t1));
    blk.instructions.push(arith(BinaryOp::Add, s1, s0, t2));
    blk.instructions.push(arith(BinaryOp::Add, s2, s1, bb));
    blk.terminator = HirTerminator::Return { values: vec![s2] };
    f
}

/// `a2` is {9, 90} and `b` is still {1, 10}: 9*1000 + 90*100 + 1*10 + 10.
const ECHO_EXPECTED: i64 = 18020;

fn build_echo_module() -> (HirModule, HirId) {
    let mk = build_mk();
    let mk_id = mk.id;
    let echo = build_echo();
    let echo_id = echo.id;
    let main = build_echo_main(mk_id, echo_id);
    let main_id = main.id;

    let mut module = HirModule::new(InternedString::new_global("struct_echo"));
    module.functions.insert(mk_id, mk);
    module.functions.insert(echo_id, echo);
    module.functions.insert(main_id, main);
    (module, main_id)
}

#[test]
fn cranelift_gives_two_calls_two_values() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    let (module, main_id) = build_module();
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend.get_function_ptr(main_id).expect("main compiled");
    let f: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    let got = unsafe { f() };

    assert_eq!(
        got, EXPECTED,
        "two calls returning a struct must give two independent values; \
         {} means both names reached one",
        got
    );
}

#[test]
fn cranelift_keeps_a_returned_argument_separate_from_the_argument() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    let (module, main_id) = build_echo_module();
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend.get_function_ptr(main_id).expect("main compiled");
    let f: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    let got = unsafe { f() };

    assert_eq!(
        got, ECHO_EXPECTED,
        "handing an argument back must copy it, not alias it; got {}",
        got
    );
}

#[test]
fn the_interpreter_keeps_a_returned_argument_separate_from_the_argument() {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};

    let (module, _) = build_echo_module();
    let mut interp = HirInterpreter::new();
    let result = interp.call(&module, "echo_main", vec![]).expect("run");
    assert_eq!(value_to_i64(&result), Some(ECHO_EXPECTED));
}

#[test]
fn the_interpreter_gives_two_calls_two_values() {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};

    let (module, _) = build_module();
    let mut interp = HirInterpreter::new();
    let result = interp.call(&module, "main", vec![]).expect("run");
    assert_eq!(value_to_i64(&result), Some(EXPECTED));
}

#[cfg(feature = "llvm-backend")]
#[test]
fn llvm_gives_two_calls_two_values() {
    use inkwell::context::Context;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;

    if zyntax_compiler::llvm_link::find_linker().is_err() {
        eprintln!("no system linker; skipping the LLVM leg");
        return;
    }

    let (module, main_id) = build_module();
    let context = Context::create();
    let mut backend = LLVMJitBackend::new(&context).expect("backend");
    backend.compile_module(&module).expect("compile");
    let ptr = backend
        .get_function_pointer(main_id)
        .expect("main compiled");
    let f: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    let got = unsafe { f() };

    assert_eq!(got, EXPECTED);
}

/// A struct entering an LLVM entry by address and leaving through the
/// destination, called through the cell Cranelift callers use.
#[cfg(feature = "llvm-backend")]
mod aggregate_entry {
    use super::*;

    fn vec3_ty() -> HirType {
        HirType::Struct(HirStructType {
            name: Some(InternedString::new_global("Vec3")),
            fields: vec![HirType::I64, HirType::I64, HirType::I64],
            packed: false,
        })
    }

    /// `def bump(a: Vec3, k: i64): Vec3 { a.x += k; return Vec3 { x: a.x * 2,
    /// y: a.y + k, z: a.z * k } }`: writes into the argument's storage, then
    /// returns a fresh struct through the destination.
    fn build_bump() -> HirFunction {
        let mut f = HirFunction::new(
            InternedString::new_global("bump"),
            sig(vec![vec3_ty(), HirType::I64], vec![vec3_ty()]),
        );
        let a = add_value(&mut f, vec3_ty(), HirValueKind::Parameter(0));
        let k = add_value(&mut f, HirType::I64, HirValueKind::Parameter(1));
        let two = konst(&mut f, 2);
        let i64v = |f: &mut HirFunction| add_value(f, HirType::I64, HirValueKind::Instruction);
        let vec3v = |f: &mut HirFunction| add_value(f, vec3_ty(), HirValueKind::Instruction);
        let (a0, s, b0, b1, b2, t0, t1, t2) = (
            i64v(&mut f),
            i64v(&mut f),
            i64v(&mut f),
            i64v(&mut f),
            i64v(&mut f),
            i64v(&mut f),
            i64v(&mut f),
            i64v(&mut f),
        );
        let a1 = vec3v(&mut f);
        let undef = add_value(&mut f, vec3_ty(), HirValueKind::Undef);
        let (r0, r1, r2) = (vec3v(&mut f), vec3v(&mut f), vec3v(&mut f));
        let extract = |result: HirId, agg: HirId, idx: u32| HirInstruction::ExtractValue {
            result,
            ty: HirType::I64,
            aggregate: agg,
            indices: vec![idx],
        };
        let insert =
            |result: HirId, agg: HirId, value: HirId, idx: u32| HirInstruction::InsertValue {
                result,
                ty: vec3_ty(),
                aggregate: agg,
                value,
                indices: vec![idx],
            };
        let arith =
            |op: BinaryOp, result: HirId, left: HirId, right: HirId| HirInstruction::Binary {
                op,
                result,
                ty: HirType::I64,
                left,
                right,
            };
        let blk = body(&mut f);
        blk.instructions.extend([
            extract(a0, a, 0),
            arith(BinaryOp::Add, s, a0, k),
            insert(a1, a, s, 0),
            extract(b0, a1, 0),
            extract(b1, a1, 1),
            extract(b2, a1, 2),
            arith(BinaryOp::Mul, t0, b0, two),
            arith(BinaryOp::Add, t1, b1, k),
            arith(BinaryOp::Mul, t2, b2, k),
            insert(r0, undef, t0, 0),
            insert(r1, r0, t1, 1),
            insert(r2, r1, t2, 2),
        ]);
        blk.terminator = HirTerminator::Return { values: vec![r2] };
        f
    }

    /// `def bump_main(): i64 { let a = Vec3 {1, 2, 3}; let r = bump(a, 10);
    /// ... }`, folding the bytes of `a` after the call and of `r` into one
    /// number, two decimal digits per field.
    fn build_bump_main(bump_id: HirId) -> HirFunction {
        let mut f = HirFunction::new(
            InternedString::new_global("bump_main"),
            sig(vec![], vec![HirType::I64]),
        );
        let consts: Vec<HirId> = [1, 2, 3, 10]
            .into_iter()
            .map(|v| konst(&mut f, v))
            .collect();
        let hundred = konst(&mut f, 100);
        let undef = add_value(&mut f, vec3_ty(), HirValueKind::Undef);
        let a: Vec<HirId> = (0..3)
            .map(|_| add_value(&mut f, vec3_ty(), HirValueKind::Instruction))
            .collect();
        let r = add_value(&mut f, vec3_ty(), HirValueKind::Instruction);
        let fields: Vec<HirId> = (0..6)
            .map(|_| add_value(&mut f, HirType::I64, HirValueKind::Instruction))
            .collect();
        let acc: Vec<HirId> = (0..10)
            .map(|_| add_value(&mut f, HirType::I64, HirValueKind::Instruction))
            .collect();
        let blk = body(&mut f);
        let mut prev = undef;
        for (i, slot) in a.iter().enumerate() {
            blk.instructions.push(HirInstruction::InsertValue {
                result: *slot,
                ty: vec3_ty(),
                aggregate: prev,
                value: consts[i],
                indices: vec![i as u32],
            });
            prev = *slot;
        }
        blk.instructions.push(HirInstruction::Call {
            result: Some(r),
            callee: HirCallable::Function(bump_id),
            args: vec![a[2], consts[3]],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        for (i, field) in fields.iter().enumerate() {
            blk.instructions.push(HirInstruction::ExtractValue {
                result: *field,
                ty: HirType::I64,
                aggregate: if i < 3 { a[2] } else { r },
                indices: vec![(i % 3) as u32],
            });
        }
        // ((((f0 * 100 + f1) * 100 + f2) * 100 + f3) * 100 + f4) * 100 + f5
        let mut total = fields[0];
        for (step, field) in fields[1..].iter().enumerate() {
            let (scaled, summed) = (acc[2 * step], acc[2 * step + 1]);
            blk.instructions.push(HirInstruction::Binary {
                op: BinaryOp::Mul,
                result: scaled,
                ty: HirType::I64,
                left: total,
                right: hundred,
            });
            blk.instructions.push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: summed,
                ty: HirType::I64,
                left: scaled,
                right: *field,
            });
            total = summed;
        }
        blk.terminator = HirTerminator::Return {
            values: vec![total],
        };
        f
    }

    /// `a` is {11, 2, 3} after the call and `r` is {22, 12, 30}.
    const BUMP_EXPECTED: i64 = 11_02_03_22_12_30;

    /// One call cell, entered first by Cranelift's code for `bump` and then
    /// by LLVM's, from the same Cranelift caller: the argument's storage and
    /// the returned struct come out byte for byte the same.
    #[test]
    fn llvm_entry_takes_a_struct_by_address_and_returns_through_the_destination() {
        use inkwell::context::Context;
        use std::sync::Arc;
        use zyntax_compiler::cranelift_backend::CraneliftBackend;
        use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;

        let bump = build_bump();
        let bump_id = bump.id;
        let main = build_bump_main(bump_id);
        let main_id = main.id;
        let mut module = HirModule::new(InternedString::new_global("struct_entry"));
        module.functions.insert(bump_id, bump.clone());
        module.functions.insert(main_id, main);

        let mut cranelift = CraneliftBackend::new().expect("backend");
        cranelift.set_reloadable_calls(true);
        cranelift.compile_module(&module).expect("compile");
        cranelift.finalize_definitions().expect("finalize");
        let ptr = cranelift.get_function_ptr(main_id).expect("main compiled");
        let run: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
        let on_cranelift = unsafe { run() };
        assert_eq!(on_cranelift, BUMP_EXPECTED);

        let context = Context::create();
        let mut llvm = LLVMJitBackend::new(&context).expect("backend");
        llvm.set_use_mcjit(true);
        llvm.set_module_context(Arc::new(module.clone()));
        llvm.set_cross_tier_links(cranelift.reload_key(), Arc::new(|_| None));
        llvm.compile_function(bump_id, &bump)
            .expect("LLVM takes the aggregate entry");
        let entry = llvm.get_function_pointer(bump_id).expect("bump compiled");
        cranelift.publish_call_target(bump_id, entry as usize);
        let on_llvm = unsafe { run() };
        assert_eq!(on_llvm, on_cranelift);
    }
}
