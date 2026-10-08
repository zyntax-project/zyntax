#![cfg(feature = "cranelift-backend")]
mod common;

use zyntax_compiler::{aggregate_split, hir::*, hir_interp::HirInterpreter, value::ZyntaxValue};
use zyntax_typed_ast::InternedString;

fn fixture(call: bool) -> (HirModule, HirId) {
    let mut signature = common::counted_loop().0.signature;
    signature.params.clear();
    signature.returns = vec![HirType::I64];
    signature.is_pure = false;
    let mut f = HirFunction::new(InternedString::new_global("snapshot"), signature.clone());
    let scalar_ptr = HirType::Ptr(Box::new(HirType::I64));
    let pair = HirType::Struct(HirStructType {
        name: None,
        fields: vec![HirType::I64, HirType::I64],
        packed: false,
    });
    for (i, ty) in [HirType::Ptr(Box::new(pair.clone())), scalar_ptr.clone()]
        .into_iter()
        .enumerate()
    {
        let id = f.create_value(ty.clone(), HirValueKind::Parameter(i as u32));
        f.signature.params.push(HirParam {
            id,
            name: InternedString::new_global("ptr"),
            ty,
            attributes: Default::default(),
            ownership: Default::default(),
        });
    }
    let p = f.signature.params[0].id;
    let q = f.signature.params[1].id;
    let loaded = f.create_value(pair.clone(), HirValueKind::Instruction);
    let first = f.create_value(HirType::I64, HirValueKind::Instruction);
    let second = f.create_value(HirType::I64, HirValueKind::Instruction);
    let sum = f.create_value(HirType::I64, HirValueKind::Instruction);
    let update = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(99)));
    let mut m = HirModule::new(InternedString::new_global("snapshot"));
    let effect = if call {
        let mut callee = HirFunction::new(InternedString::new_global("mutate"), signature);
        let ptr = callee.create_value(scalar_ptr.clone(), HirValueKind::Parameter(0));
        callee.signature.params.push(HirParam {
            id: ptr,
            name: InternedString::new_global("ptr"),
            ty: scalar_ptr,
            attributes: Default::default(),
            ownership: Default::default(),
        });
        let value = callee.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(99)));
        let b = callee.blocks.get_mut(&callee.entry_block).unwrap();
        b.instructions.push(HirInstruction::Store {
            ptr,
            value,
            align: 8,
            volatile: false,
        });
        b.terminator = HirTerminator::Return {
            values: vec![value],
        };
        let id = callee.id;
        m.functions.insert(id, callee);
        HirInstruction::Call {
            result: None,
            callee: HirCallable::Function(id),
            args: vec![q],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        }
    } else {
        HirInstruction::Store {
            ptr: q,
            value: update,
            align: 8,
            volatile: false,
        }
    };
    let b = f.blocks.get_mut(&f.entry_block).unwrap();
    b.instructions = vec![
        HirInstruction::Load {
            result: loaded,
            ty: pair,
            ptr: p,
            align: 8,
            volatile: false,
        },
        effect,
        HirInstruction::ExtractValue {
            result: first,
            ty: HirType::I64,
            aggregate: loaded,
            indices: vec![0],
        },
        HirInstruction::ExtractValue {
            result: second,
            ty: HirType::I64,
            aggregate: loaded,
            indices: vec![1],
        },
        HirInstruction::Binary {
            result: sum,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: first,
            right: second,
        },
    ];
    b.terminator = HirTerminator::Return { values: vec![sum] };
    let id = f.id;
    m.functions.insert(id, f);
    (m, id)
}

fn check(mut call: impl FnMut(*mut i64, *mut i64) -> i64) {
    for alias in [false, true] {
        let mut pair = [11i64, 24];
        let mut other = 7;
        let p = pair.as_mut_ptr();
        let q = if alias { p } else { &mut other };
        assert_eq!(call(p, q), 35);
        assert_eq!(pair, [if alias { 99 } else { 11 }, 24]);
        assert_eq!(other, if alias { 7 } else { 99 });
    }
}

#[test]
fn snapshot_survives_an_aliasing_write_or_call_in_every_tier() {
    for call in [false, true] {
        let (mut m, id) = fixture(call);
        assert_eq!(aggregate_split::run_module(&mut m).field_reads_only, 1);
        let mut interp = HirInterpreter::new();
        check(|p, q| {
            match interp
                .call(
                    &m,
                    "snapshot",
                    vec![
                        ZyntaxValue::Pointer(p.cast()),
                        ZyntaxValue::Pointer(q.cast()),
                    ],
                )
                .unwrap()
            {
                ZyntaxValue::Int(n) => n,
                other => panic!("{other:?}"),
            }
        });
        let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
        clif.compile_module(&m).unwrap();
        clif.finalize_definitions().unwrap();
        // SAFETY: fixture signature is C (pointer, pointer) -> i64; backend stays alive.
        let compiled: unsafe extern "C" fn(*mut i64, *mut i64) -> i64 =
            unsafe { std::mem::transmute(clif.get_function_ptr(id).unwrap()) };
        check(|p, q| unsafe { compiled(p, q) });
        #[cfg(feature = "llvm-backend")]
        {
            let context = inkwell::context::Context::create();
            let mut llvm =
                zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context).unwrap();
            llvm.compile_module(&m).unwrap();
            // SAFETY: same fixture signature, with the backend kept alive.
            let compiled: unsafe extern "C" fn(*mut i64, *mut i64) -> i64 =
                unsafe { std::mem::transmute(llvm.get_function_pointer(id).unwrap()) };
            check(|p, q| unsafe { compiled(p, q) });
        }
    }
}
