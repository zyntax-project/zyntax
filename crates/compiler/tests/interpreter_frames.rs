//! Reused frames preserve call boundaries, including owned host values and errors.
use std::sync::{Arc, Mutex};
use zyntax_compiler::hir::{
    HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirInstruction, HirModule,
    HirParam, HirTerminator, HirType, HirValueKind,
};
use zyntax_compiler::hir_interp::{HirInterpreter, InterpError};
use zyntax_compiler::value::ZyntaxValue;
use zyntax_typed_ast::InternedString;

fn function(name: &str, callee: Option<HirCallable>, calls: usize) -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global(name),
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
    let arg = f.create_value(HirType::I64, HirValueKind::Parameter(0));
    f.signature.params.push(HirParam {
        id: arg,
        name: InternedString::new_global("value"),
        ty: HirType::I64,
        attributes: Default::default(),
        ownership: Default::default(),
    });
    let mut result = arg;
    if let Some(callee) = callee {
        for _ in 0..calls {
            result = f.create_value(HirType::I64, HirValueKind::Instruction);
            f.blocks[&f.entry_block]
                .instructions
                .push(HirInstruction::Call {
                    result: Some(result),
                    callee: callee.clone(),
                    args: vec![arg],
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                });
        }
    }
    f.blocks[&f.entry_block].terminator = HirTerminator::Return {
        values: vec![result],
    };
    f
}

#[test]
fn errors_and_different_frame_sizes_do_not_retain_arguments() {
    let leaf = function("leaf", Some(HirCallable::Symbol("echo".into())), 16);
    let caller = function("caller", Some(HirCallable::Function(leaf.id)), 2);
    let small = function("small", None, 0);
    let mut module = HirModule::new(InternedString::new_global("frames"));
    for f in [leaf, caller, small] {
        module.functions.insert(f.id, f);
    }
    let mut interp = HirInterpreter::new();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let capture = Arc::clone(&observed);
    interp.set_symbol_call_dispatcher(Box::new(move |_, args| {
        assert_eq!(args.len(), 1);
        capture.lock().unwrap().push(args[0].clone());
        if args[0] == ZyntaxValue::String("fail".into()) {
            return Err(InterpError::Host("requested failure".into()));
        }
        Ok(Some(args[0].clone()))
    }));
    for _ in 0..3 {
        assert!(
            interp
                .call(&module, "caller", vec![ZyntaxValue::String("fail".into())])
                .is_err()
        );
        // The host value API permits owned payloads; these must be dropped
        // even when a nested interpreter call fails.
        let value = ZyntaxValue::String("next call".into());
        assert_eq!(
            interp.call(&module, "caller", vec![value.clone()]).unwrap(),
            value
        );
        assert_eq!(
            interp.call(&module, "small", vec![]).unwrap(),
            ZyntaxValue::Undef
        );
    }
    assert_eq!(observed.lock().unwrap().len(), 3 * 33);
}

#[test]
fn indirect_dispatch_can_take_argument_ownership_on_success_and_error() {
    let mut f = function(
        "indirect",
        Some(HirCallable::Symbol("placeholder".into())),
        2,
    );
    let handle = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(23)));
    for inst in &mut f.blocks[&f.entry_block].instructions {
        if let HirInstruction::Call { callee, .. } = inst {
            *callee = HirCallable::Indirect(handle);
        }
    }
    let mut module = HirModule::new(InternedString::new_global("indirect_frames"));
    module.functions.insert(f.id, f);
    let mut interp = HirInterpreter::new();
    interp.set_indirect_call_dispatcher(Box::new(|handle, mut args| {
        assert_eq!(handle, 23);
        assert_eq!(args.len(), 1);
        let result = args.pop().unwrap();
        if result == ZyntaxValue::Int(-1) {
            return Err(InterpError::Host("requested failure".into()));
        }
        Ok(result)
    }));
    for input in [7, -1, 11, 0] {
        let got = interp.call(&module, "indirect", vec![ZyntaxValue::Int(input)]);
        if input == -1 {
            assert!(got.is_err());
        } else {
            assert_eq!(got.unwrap(), ZyntaxValue::Int(input));
        }
    }
}
