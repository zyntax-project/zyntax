//! Entry constants keep branch targets and call-site liveness aligned.
use zyntax_compiler::hir::*;
use zyntax_compiler::hir_interp::{HirInterpreter, Memory, Op, compile_function};
use zyntax_compiler::value::ZyntaxValue;
use zyntax_typed_ast::InternedString;

fn function(name: &str) -> HirFunction {
    let param = HirParam {
        id: HirId::new(),
        name: InternedString::new_global("n"),
        ty: HirType::I64,
        attributes: Default::default(),
        ownership: Default::default(),
    };
    HirFunction::new(
        InternedString::new_global(name),
        HirFunctionSignature {
            params: vec![param],
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
    )
}

#[test]
fn constants_precede_switch_targets_and_call_read_sets() {
    let mut module = HirModule::new(InternedString::new_global("constants"));
    let mut identity = function("identity");
    identity.blocks[&identity.entry_block].terminator = HirTerminator::Return {
        values: vec![identity.signature.params[0].id],
    };
    let callee = identity.id;
    module.add_function(identity);
    let mut f = function("choose");
    let selector = f.signature.params[0].id;
    let mut blocks = Vec::new();
    for number in [42, 7, -1] {
        let block = f.create_block();
        let value = f.create_value(
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(number)),
        );
        let result = f.create_value(HirType::I64, HirValueKind::Instruction);
        f.blocks[&block].instructions.push(HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Function(callee),
            args: vec![value],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        f.blocks[&block].terminator = HirTerminator::Return {
            values: vec![result],
        };
        blocks.push(block);
    }
    f.blocks[&f.entry_block].terminator = HirTerminator::Switch {
        value: selector,
        cases: vec![
            (HirConstant::I64(1), blocks[0]),
            (HirConstant::I64(2), blocks[1]),
        ],
        default: blocks[2],
    };
    let code = compile_function(&module, &mut Memory::new(), &f).unwrap();
    let prefix = code
        .code
        .iter()
        .take_while(|op| matches!(op, Op::LoadConst { .. }))
        .count();
    assert_eq!(prefix, 3);
    assert!(code.read_set_at[..prefix].iter().all(|i| *i == u32::MAX));
    let mut calls = 0;
    for (pc, op) in code.code.iter().enumerate() {
        if let Op::CallFn { .. } = op {
            calls += 1;
            let read = &code.read_sets[code.read_set_at[pc] as usize];
            for constant in &code.code[..prefix] {
                let Op::LoadConst { dst, .. } = constant else {
                    unreachable!()
                };
                assert_ne!(read[*dst as usize / 64] & (1 << (*dst as usize % 64)), 0);
            }
        }
        if let Op::Switch { table, default, .. } = op {
            for target in std::iter::once(default)
                .chain(code.switch_pool[*table as usize].iter().map(|(_, pc)| pc))
            {
                assert!((*target as usize) >= prefix && (*target as usize) < code.code.len());
                assert!(matches!(code.code[*target as usize], Op::CallFn { .. }));
            }
        }
    }
    assert_eq!(calls, 3);
    module.add_function(f);
    let mut interpreter = HirInterpreter::new();
    for (argument, expected) in [(1, 42), (2, 7), (0, -1), (1, 42)] {
        assert_eq!(
            interpreter
                .call(&module, "choose", vec![ZyntaxValue::Int(argument)])
                .unwrap(),
            ZyntaxValue::Int(expected)
        );
    }
}
