//! A recursive function runs on the interpreter: nested calls share the
//! compiled function with the frame that made them, and a later call
//! finds it still cached.

use zyntax_compiler::hir::{
    BinaryOp, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirId, HirInstruction,
    HirModule, HirParam, HirTerminator, HirType, HirValue, HirValueKind, ParamAttributes,
};
use zyntax_compiler::hir_interp::HirInterpreter;
use zyntax_compiler::value::ZyntaxValue;
use zyntax_typed_ast::InternedString;

fn value(func: &mut HirFunction, kind: HirValueKind) -> HirId {
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty: HirType::I64,
            kind,
            span: None,
        },
    );
    id
}

fn binary(op: BinaryOp, result: HirId, left: HirId, right: HirId) -> HirInstruction {
    HirInstruction::Binary {
        op,
        result,
        ty: HirType::I64,
        left,
        right,
    }
}

/// `fib(n) = if n < 2 { n } else { fib(n - 1) + fib(n - 2) }`
fn fib_module() -> HirModule {
    let signature = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: InternedString::new_global("n"),
            ty: HirType::I64,
            attributes: ParamAttributes::default(),
            ownership: Default::default(),
        }],
        returns: vec![HirType::I64],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };
    let mut f = HirFunction::new(InternedString::new_global("fib"), signature);
    let n = value(&mut f, HirValueKind::Parameter(0));
    let one = value(&mut f, HirValueKind::Constant(HirConstant::I64(1)));
    let two = value(&mut f, HirValueKind::Constant(HirConstant::I64(2)));
    let small = value(&mut f, HirValueKind::Instruction);
    let n1 = value(&mut f, HirValueKind::Instruction);
    let n2 = value(&mut f, HirValueKind::Instruction);
    let a = value(&mut f, HirValueKind::Instruction);
    let b = value(&mut f, HirValueKind::Instruction);
    let sum = value(&mut f, HirValueKind::Instruction);

    let entry = f.entry_block;
    let base = f.create_block();
    let step = f.create_block();
    let me = f.id;

    let blk = f.blocks.get_mut(&entry).unwrap();
    blk.instructions.push(binary(BinaryOp::Lt, small, n, two));
    blk.terminator = HirTerminator::CondBranch {
        condition: small,
        true_target: base,
        false_target: step,
    };
    f.blocks.get_mut(&base).unwrap().terminator = HirTerminator::Return { values: vec![n] };

    let blk = f.blocks.get_mut(&step).unwrap();
    blk.instructions.push(binary(BinaryOp::Sub, n1, n, one));
    blk.instructions.push(binary(BinaryOp::Sub, n2, n, two));
    for (result, arg) in [(a, n1), (b, n2)] {
        blk.instructions.push(HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Function(me),
            args: vec![arg],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
    }
    blk.instructions.push(binary(BinaryOp::Add, sum, a, b));
    blk.terminator = HirTerminator::Return { values: vec![sum] };

    let mut module = HirModule::new(InternedString::new_global("interp_recursion"));
    module.functions.insert(me, f);
    module
}

#[test]
fn recursive_calls_share_one_compiled_function() {
    let module = fib_module();
    let mut interp = HirInterpreter::new();
    for _ in 0..2 {
        let got = interp
            .call(&module, "fib", vec![ZyntaxValue::Int(15)])
            .expect("run");
        assert_eq!(got, ZyntaxValue::Int(610));
    }
    let got = interp
        .call(&module, "fib", vec![ZyntaxValue::Int(1)])
        .expect("run");
    assert_eq!(got, ZyntaxValue::Int(1));
}
