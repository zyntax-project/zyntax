//! A struct field that is itself a struct is a value like any other.
//!
//! `insertvalue` of an aggregate into a field copies the whole
//! aggregate, and `extractvalue` of such a field yields the aggregate
//! whose own fields can then be read. The HIR names the field's type
//! and says nothing about its width, so a backend that moves one word
//! for every field reads or writes the wrong bytes the moment a field
//! is wider than a register.
//!
//! The same module runs on every backend that can execute it.

#![cfg(feature = "cranelift-backend")]

use std::collections::HashSet;
use zyntax_compiler::hir::{
    BinaryOp, HirBlock, HirConstant, HirFunction, HirFunctionSignature, HirId, HirInstruction,
    HirModule, HirParam, HirStructType, HirTerminator, HirType, HirValue, HirValueKind,
    ParamAttributes,
};
use zyntax_typed_ast::InternedString;

/// `p` is {3, 40} when it goes into `o` = {7, p}; afterwards `p` is
/// rebuilt as {9, 40}. Reading `o` back must give the copy taken at
/// insertion (7, 3, 40) and `p` its new first field (9), each weighed
/// by a different power of ten: 9*10000 + 7*1000 + 3*100 + 40.
const EXPECTED: i64 = 97340;

fn pair_ty() -> HirType {
    HirType::Struct(HirStructType {
        name: Some(InternedString::new_global("Pair")),
        fields: vec![HirType::I64, HirType::I64],
        packed: false,
    })
}

fn outer_ty() -> HirType {
    HirType::Struct(HirStructType {
        name: Some(InternedString::new_global("Outer")),
        fields: vec![HirType::I64, pair_ty()],
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

fn insert(
    result: HirId,
    ty: HirType,
    agg: HirId,
    value: HirId,
    indices: Vec<u32>,
) -> HirInstruction {
    HirInstruction::InsertValue {
        result,
        ty,
        aggregate: agg,
        value,
        indices,
    }
}

fn extract(result: HirId, ty: HirType, agg: HirId, indices: Vec<u32>) -> HirInstruction {
    HirInstruction::ExtractValue {
        result,
        ty,
        aggregate: agg,
        indices,
    }
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

/// ```text
/// def main(): i64 {
///     let p = Pair { a: 3, b: 40 }
///     let o = Outer { tag: 7, inner: p }
///     let p2 = p with a = 9
///     let inner = o.inner
///     return p2.a*10000 + o.tag*1000 + inner.a*100 + o.inner.b
/// }
/// ```
fn build_main() -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global("main"),
        sig(vec![], vec![HirType::I64]),
    );

    let three = konst(&mut f, 3);
    let forty = konst(&mut f, 40);
    let seven = konst(&mut f, 7);
    let nine = konst(&mut f, 9);
    let k10000 = konst(&mut f, 10000);
    let k1000 = konst(&mut f, 1000);
    let k100 = konst(&mut f, 100);

    let p_undef = add_value(&mut f, pair_ty(), HirValueKind::Undef);
    let p_a = add_value(&mut f, pair_ty(), HirValueKind::Instruction);
    let p = add_value(&mut f, pair_ty(), HirValueKind::Instruction);
    let o_undef = add_value(&mut f, outer_ty(), HirValueKind::Undef);
    let o_tag = add_value(&mut f, outer_ty(), HirValueKind::Instruction);
    let o = add_value(&mut f, outer_ty(), HirValueKind::Instruction);
    let p2 = add_value(&mut f, pair_ty(), HirValueKind::Instruction);
    let inner = add_value(&mut f, pair_ty(), HirValueKind::Instruction);

    let p2a = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let tag = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let ia = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let ib = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let t0 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let t1 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let t2 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let s0 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let s1 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);
    let s2 = add_value(&mut f, HirType::I64, HirValueKind::Instruction);

    let blk = body(&mut f);
    blk.instructions
        .push(insert(p_a, pair_ty(), p_undef, three, vec![0]));
    blk.instructions
        .push(insert(p, pair_ty(), p_a, forty, vec![1]));
    blk.instructions
        .push(insert(o_tag, outer_ty(), o_undef, seven, vec![0]));
    blk.instructions
        .push(insert(o, outer_ty(), o_tag, p, vec![1]));
    // Rebuilding `p` after it went into `o`: a field that only kept
    // `p`'s address would follow it.
    blk.instructions
        .push(insert(p2, pair_ty(), p, nine, vec![0]));
    blk.instructions.push(extract(inner, pair_ty(), o, vec![1]));
    blk.instructions
        .push(extract(p2a, HirType::I64, p2, vec![0]));
    blk.instructions
        .push(extract(tag, HirType::I64, o, vec![0]));
    blk.instructions
        .push(extract(ia, HirType::I64, inner, vec![0]));
    blk.instructions
        .push(extract(ib, HirType::I64, o, vec![1, 1]));
    blk.instructions.push(arith(BinaryOp::Mul, t0, p2a, k10000));
    blk.instructions.push(arith(BinaryOp::Mul, t1, tag, k1000));
    blk.instructions.push(arith(BinaryOp::Mul, t2, ia, k100));
    blk.instructions.push(arith(BinaryOp::Add, s0, t0, t1));
    blk.instructions.push(arith(BinaryOp::Add, s1, s0, t2));
    blk.instructions.push(arith(BinaryOp::Add, s2, s1, ib));
    blk.terminator = HirTerminator::Return { values: vec![s2] };
    f
}

fn build_module() -> (HirModule, HirId) {
    let main = build_main();
    let main_id = main.id;
    let mut module = HirModule::new(InternedString::new_global("nested_fields"));
    module.functions.insert(main_id, main);
    (module, main_id)
}

#[test]
fn cranelift_copies_a_struct_into_a_field_and_reads_it_back() {
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
        "o = {{7, {{3, 40}}}} and p2 = {{9, 40}}; got {}",
        got
    );
}

#[test]
fn the_interpreter_copies_a_struct_into_a_field_and_reads_it_back() {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};

    let (module, _) = build_module();
    let mut interp = HirInterpreter::new();
    let result = interp.call(&module, "main", vec![]).expect("run");
    assert_eq!(value_to_i64(&result), Some(EXPECTED));
}

#[cfg(feature = "llvm-backend")]
#[test]
fn llvm_copies_a_struct_into_a_field_and_reads_it_back() {
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
