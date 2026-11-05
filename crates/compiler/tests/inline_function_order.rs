use zyntax_compiler::hir::*;
use zyntax_typed_ast::InternedString;

fn function(index: usize) -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global(&format!("ordered_{index}")),
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
    let value = f.create_value(
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(index as i64)),
    );
    f.blocks[&f.entry_block].terminator = HirTerminator::Return {
        values: vec![value],
    };
    f
}

fn call(f: &mut HirFunction, callee: HirId) {
    f.blocks[&f.entry_block]
        .instructions
        .push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Function(callee),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
}

#[test]
fn rejected_calls_preserve_module_order_and_bodies() {
    // Every selection of stable and changing functions, including adjacent
    // removals and gaps between them. A no-op pass must preserve the module.
    for stable in 0..64 {
        let mut module = HirModule::new(InternedString::new_global("ordered_no_inline"));
        let mut functions: Vec<_> = (0..6).map(function).collect();
        for index in 0..functions.len() {
            let callee = functions[(index + 1) % functions.len()].id;
            let f = &mut functions[index];
            f.attributes.optimized = stable & (1 << index) != 0;
            f.attributes.no_inline = true;
            call(f, callee);
        }
        for f in functions {
            module.functions.insert(f.id, f);
        }
        let order: Vec<_> = module.functions.keys().copied().collect();
        let original = bincode::serialize(&module).unwrap();
        for _ in 0..3 {
            let stats = zyntax_compiler::inline::run_module(&mut module);
            assert_eq!(stats.inlined, 0);
            assert_eq!(module.functions.keys().copied().collect::<Vec<_>>(), order);
            assert_eq!(
                bincode::serialize(&module).unwrap(),
                original,
                "{stable:06b}"
            );
        }
    }
}

#[test]
fn accepted_inline_keeps_function_positions() {
    let mut module = HirModule::new(InternedString::new_global("ordered_inline"));
    let mut functions: Vec<_> = (0..7).map(function).collect();
    let callee = functions[4].id;
    let caller = functions[2].id;
    call(&mut functions[2], callee);
    for (index, mut f) in functions.into_iter().enumerate() {
        f.attributes.optimized = index != 2 && index != 4;
        module.functions.insert(f.id, f);
    }
    let order: Vec<_> = module.functions.keys().copied().collect();
    let unchanged: Vec<_> = module
        .functions
        .iter()
        .filter(|(id, _)| **id != caller)
        .map(|(id, f)| (*id, bincode::serialize(f).unwrap()))
        .collect();
    let stats = zyntax_compiler::inline::run_module(&mut module);
    assert_eq!(stats.inlined, 1);
    assert_eq!(module.functions.keys().copied().collect::<Vec<_>>(), order);
    for (id, original) in unchanged {
        assert_eq!(
            bincode::serialize(&module.functions[&id]).unwrap(),
            original
        );
    }
    assert!(
        module.functions[&caller].blocks[&module.functions[&caller].entry_block]
            .instructions
            .is_empty()
    );
}
