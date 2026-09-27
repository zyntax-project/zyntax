//! Storage that reaches a phi which does not own it stays alive.
//!
//! A loop keeps the largest of the objects it makes: each iteration
//! allocates one on either arm of a branch, so the object is itself a
//! join phi, and hands it to the loop's running value when it is the
//! larger. The running value is a pair of phis naming each other, the
//! header's and the merge's. The join phi may own its object only while
//! the running value owns what it is handed; when the running value
//! owns nothing, releasing the object after its last read in the body
//! frees the storage the loop goes on to read.
//!
//! The module goes through the release pass and then runs on every
//! backend that can execute it.

#![cfg(feature = "cranelift-backend")]

use std::collections::HashSet;
use zyntax_compiler::hir::{
    BinaryOp, HirBlock, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirId,
    HirInstruction, HirModule, HirPhi, HirTerminator, HirType, HirValue, HirValueKind, Intrinsic,
};
use zyntax_typed_ast::InternedString;

/// The objects hold 100, 101, 102 and 103; the last is the largest.
const EXPECTED: i64 = 103;

fn sig() -> HirFunctionSignature {
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

fn inst(func: &mut HirFunction, ty: HirType) -> HirId {
    add_value(func, ty, HirValueKind::Instruction)
}

fn block(func: &mut HirFunction, id: HirId) -> &mut HirBlock {
    func.blocks.get_mut(&id).unwrap()
}

fn malloc(result: HirId, size: HirId) -> HirInstruction {
    HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
        args: vec![size],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    }
}

fn binary(op: BinaryOp, result: HirId, ty: HirType, left: HirId, right: HirId) -> HirInstruction {
    HirInstruction::Binary {
        op,
        result,
        ty,
        left,
        right,
    }
}

fn load(result: HirId, ptr: HirId) -> HirInstruction {
    HirInstruction::Load {
        result,
        ty: HirType::I64,
        ptr,
        align: 8,
        volatile: false,
    }
}

fn store(value: HirId, ptr: HirId) -> HirInstruction {
    HirInstruction::Store {
        value,
        ptr,
        align: 8,
        volatile: false,
    }
}

/// ```text
/// entry:  t0 = malloc 8; *t0 = 0; -> header
/// header: t = phi [t0, entry] [tn, merge]; i = phi [0, entry] [i1, merge]
///         i < 4 ? pick : exit
/// pick:   i & 1 == 1 ? left : right
/// left:   a1 = malloc 8; -> join
/// right:  a2 = malloc 8; -> join
/// join:   a = phi [a1, left] [a2, right]; *a = i + 100
///         *a > *t ? take : skip
/// take:   -> merge
/// skip:   -> merge
/// merge:  tn = phi [a, take] [t, skip]; i1 = i + 1; -> header
/// exit:   return *t
/// ```
fn build_module() -> (HirModule, HirId) {
    let mut f = HirFunction::new(InternedString::new_global("keep_largest"), sig());
    let entry = f.entry_block;
    let header = f.create_block();
    let pick = f.create_block();
    let left = f.create_block();
    let right = f.create_block();
    let join = f.create_block();
    let take = f.create_block();
    let skip = f.create_block();
    let merge = f.create_block();
    let exit = f.create_block();

    let ptr = HirType::Ptr(Box::new(HirType::I64));
    let c0 = konst(&mut f, 0);
    let c1 = konst(&mut f, 1);
    let c4 = konst(&mut f, 4);
    let c8 = konst(&mut f, 8);
    let c100 = konst(&mut f, 100);

    let t0 = inst(&mut f, ptr.clone());
    let t = inst(&mut f, ptr.clone());
    let tn = inst(&mut f, ptr.clone());
    let i = inst(&mut f, HirType::I64);
    let i1 = inst(&mut f, HirType::I64);
    let more = inst(&mut f, HirType::Bool);
    let bit = inst(&mut f, HirType::I64);
    let odd = inst(&mut f, HirType::Bool);
    let a1 = inst(&mut f, ptr.clone());
    let a2 = inst(&mut f, ptr.clone());
    let a = inst(&mut f, ptr.clone());
    let v = inst(&mut f, HirType::I64);
    let av = inst(&mut f, HirType::I64);
    let tv = inst(&mut f, HirType::I64);
    let larger = inst(&mut f, HirType::Bool);
    let result = inst(&mut f, HirType::I64);

    let edges = [
        (entry, header),
        (header, pick),
        (header, exit),
        (pick, left),
        (pick, right),
        (left, join),
        (right, join),
        (join, take),
        (join, skip),
        (take, merge),
        (skip, merge),
        (merge, header),
    ];
    // Loop detection reads the edge lists, so they are filled in.
    for (from, to) in edges {
        block(&mut f, from).successors.push(to);
        block(&mut f, to).predecessors.push(from);
    }

    let b = block(&mut f, entry);
    b.instructions.push(malloc(t0, c8));
    b.instructions.push(store(c0, t0));
    b.terminator = HirTerminator::Branch { target: header };

    let b = block(&mut f, header);
    b.phis.push(HirPhi {
        result: t,
        ty: ptr.clone(),
        incoming: vec![(t0, entry), (tn, merge)],
    });
    b.phis.push(HirPhi {
        result: i,
        ty: HirType::I64,
        incoming: vec![(c0, entry), (i1, merge)],
    });
    b.instructions
        .push(binary(BinaryOp::Lt, more, HirType::Bool, i, c4));
    b.terminator = HirTerminator::CondBranch {
        condition: more,
        true_target: pick,
        false_target: exit,
    };

    let b = block(&mut f, pick);
    b.instructions
        .push(binary(BinaryOp::And, bit, HirType::I64, i, c1));
    b.instructions
        .push(binary(BinaryOp::Eq, odd, HirType::Bool, bit, c1));
    b.terminator = HirTerminator::CondBranch {
        condition: odd,
        true_target: left,
        false_target: right,
    };

    let b = block(&mut f, left);
    b.instructions.push(malloc(a1, c8));
    b.terminator = HirTerminator::Branch { target: join };

    let b = block(&mut f, right);
    b.instructions.push(malloc(a2, c8));
    b.terminator = HirTerminator::Branch { target: join };

    let b = block(&mut f, join);
    b.phis.push(HirPhi {
        result: a,
        ty: ptr.clone(),
        incoming: vec![(a1, left), (a2, right)],
    });
    b.instructions
        .push(binary(BinaryOp::Add, v, HirType::I64, i, c100));
    b.instructions.push(store(v, a));
    b.instructions.push(load(av, a));
    b.instructions.push(load(tv, t));
    b.instructions
        .push(binary(BinaryOp::Gt, larger, HirType::Bool, av, tv));
    b.terminator = HirTerminator::CondBranch {
        condition: larger,
        true_target: take,
        false_target: skip,
    };

    block(&mut f, take).terminator = HirTerminator::Branch { target: merge };
    block(&mut f, skip).terminator = HirTerminator::Branch { target: merge };

    let b = block(&mut f, merge);
    b.phis.push(HirPhi {
        result: tn,
        ty: ptr.clone(),
        incoming: vec![(a, take), (t, skip)],
    });
    b.instructions
        .push(binary(BinaryOp::Add, i1, HirType::I64, i, c1));
    b.terminator = HirTerminator::Branch { target: header };

    let b = block(&mut f, exit);
    b.instructions.push(load(result, t));
    b.terminator = HirTerminator::Return {
        values: vec![result],
    };

    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("owned_phi_release"));
    module.automatic_release = true;
    module.functions.insert(id, f);
    zyntax_compiler::drop_insert::run_module(&mut module);
    (module, id)
}

#[test]
fn the_interpreter_reads_the_kept_object() {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};

    let (module, _) = build_module();
    let mut interp = HirInterpreter::new();
    // The allocator the compiled tiers share, which takes a freed block
    // back; the interpreter's own arena never does.
    interp.use_native_allocator();
    let result = interp.call(&module, "keep_largest", vec![]).expect("run");
    assert_eq!(value_to_i64(&result), Some(EXPECTED));
}

#[test]
fn cranelift_reads_the_kept_object() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    let (module, id) = build_module();
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend.get_function_ptr(id).expect("compiled");
    let f: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    assert_eq!(unsafe { f() }, EXPECTED);
}

#[cfg(feature = "llvm-backend")]
#[test]
fn llvm_reads_the_kept_object() {
    use inkwell::context::Context;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;

    if zyntax_compiler::llvm_link::find_linker().is_err() {
        eprintln!("no system linker; skipping the LLVM leg");
        return;
    }

    let (module, id) = build_module();
    let context = Context::create();
    let mut backend = LLVMJitBackend::new(&context).expect("backend");
    backend.compile_module(&module).expect("compile");
    let ptr = backend.get_function_pointer(id).expect("compiled");
    let f: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    assert_eq!(unsafe { f() }, EXPECTED);
}
