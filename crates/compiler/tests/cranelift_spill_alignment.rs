//! Register spills keep their alignment after addressable local storage.
#![cfg(feature = "cranelift-backend")]
mod common;

use zyntax_compiler::{cranelift_backend::CraneliftBackend, hir::*};
use zyntax_typed_ast::InternedString;

unsafe extern "C" fn update_slot(slot: *mut f64) {
    // SAFETY: the compiled caller passes its live f64 local.
    unsafe { *slot = *slot * 2.0 + 1.0 };
}

fn fixture() -> (HirModule, HirId) {
    let mut signature = common::counted_loop().0.signature;
    signature.params.clear();
    signature.returns = vec![HirType::F64];
    let mut f = HirFunction::new(
        InternedString::new_global("spill_floats"),
        signature.clone(),
    );
    f.calling_convention = CallingConvention::C;
    let mut arguments = Vec::new();
    for i in 0..8 {
        let id = f.create_value(HirType::F64, HirValueKind::Parameter(i));
        f.signature.params.push(HirParam {
            id,
            name: InternedString::new_global(&format!("a{i}")),
            ty: HirType::F64,
            attributes: Default::default(),
            ownership: Default::default(),
        });
        arguments.push(id);
    }
    signature.returns.clear();
    signature.params.push(HirParam {
        id: HirId::new(),
        name: InternedString::new_global("slot"),
        ty: HirType::Ptr(Box::new(HirType::F64)),
        attributes: Default::default(),
        ownership: Default::default(),
    });
    let mut host = HirFunction::new(InternedString::new_global("spill_update_slot"), signature);
    host.is_external = true;
    host.calling_convention = CallingConvention::C;
    let slot = f.create_value(
        HirType::Ptr(Box::new(HirType::F64)),
        HirValueKind::Instruction,
    );
    let loaded = f.create_value(HirType::F64, HirValueKind::Instruction);
    let mut instructions = vec![
        HirInstruction::Alloca {
            result: slot,
            ty: HirType::F64,
            count: None,
            align: 8,
        },
        HirInstruction::Store {
            value: arguments[0],
            ptr: slot,
            align: 8,
            volatile: false,
        },
        HirInstruction::Call {
            result: None,
            callee: HirCallable::Function(host.id),
            args: vec![slot],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
        HirInstruction::Load {
            result: loaded,
            ty: HirType::F64,
            ptr: slot,
            align: 8,
            volatile: false,
        },
    ];
    let mut result = loaded;
    for arg in arguments {
        let product = f.create_value(HirType::F64, HirValueKind::Instruction);
        let sum = f.create_value(HirType::F64, HirValueKind::Instruction);
        instructions.push(HirInstruction::Binary {
            result: product,
            op: BinaryOp::FMul,
            ty: HirType::F64,
            left: arg,
            right: loaded,
        });
        instructions.push(HirInstruction::Binary {
            result: sum,
            op: BinaryOp::FAdd,
            ty: HirType::F64,
            left: result,
            right: product,
        });
        result = sum;
    }
    let entry = f.blocks.get_mut(&f.entry_block).unwrap();
    entry.instructions = instructions;
    entry.terminator = HirTerminator::Return {
        values: vec![result],
    };
    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("spill_alignment"));
    module.functions.insert(host.id, host);
    module.functions.insert(id, f);
    (module, id)
}

#[test]
fn floats_and_addressable_local_survive_a_host_call() {
    let (module, id) = fixture();
    let mut backend =
        CraneliftBackend::with_runtime_symbols(&[("spill_update_slot", update_slot as *const u8)])
            .unwrap();
    backend.set_capture_ir(true);
    backend.compile_module(&module).unwrap();
    backend.finalize_definitions().unwrap();
    // SAFETY: this is the exact C signature in the fixture.
    let call: unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, f64, f64) -> f64 =
        unsafe { std::mem::transmute(backend.get_function_ptr(id).unwrap()) };
    for a in [-9.25, 0.0, 0.5, 17.0] {
        let expected = (a * 2.0 + 1.0) * (a + 36.0);
        assert_eq!(
            unsafe { call(a, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) },
            expected
        );
    }
    #[cfg(target_arch = "x86_64")]
    {
        let (_, disassembly) = backend.take_captured_ir().unwrap();
        let disassembly = disassembly.unwrap();
        let mut spills = 0;
        for line in disassembly
            .lines()
            .filter(|line| line.contains("movdqu") && line.contains("(%rsp)"))
        {
            let address = line.split("(%rsp)").next().unwrap();
            let offset = address.rsplit(['+', ',', ' ']).next().unwrap();
            let offset = if let Some(hex) = offset.strip_prefix("0x") {
                u32::from_str_radix(hex, 16)
            } else {
                offset.parse()
            }
            .unwrap_or_else(|_| panic!("unrecognized spill stack offset: {line}"));
            assert_eq!(offset % 16, 0, "unaligned XMM spill: {line}");
            spills += 1;
        }
        assert!(
            spills >= 8,
            "fixture must force XMM spills across the host call:\n{disassembly}"
        );
    }
}
