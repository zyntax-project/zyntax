//! A struct a caller names by its address, inlined into a callee that
//! takes it by value.
//!
//! A list header is passed this way: the caller holds a pointer and the
//! callee's parameter is the struct. Once inlined, the callee's field
//! reads must still read through that pointer, in the interpreter as in
//! compiled code.

use zyntax_compiler::hir::*;
use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
use zyntax_typed_ast::InternedString;

fn signature(params: Vec<HirParam>, returns: Vec<HirType>) -> HirFunctionSignature {
    HirFunctionSignature {
        params,
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

fn constant(f: &mut HirFunction, n: i64) -> HirId {
    f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(n)))
}

/// `second(s) = s.1`, called by `run` with the address of `{7, 42}`.
fn module() -> HirModule {
    let pair = HirType::Struct(HirStructType {
        name: None,
        fields: vec![HirType::I64, HirType::I64].into(),
        packed: false,
    });
    let s = HirId::new();
    let mut callee = HirFunction::new(
        InternedString::new_global("second"),
        signature(
            vec![HirParam {
                id: s,
                name: InternedString::new_global("s"),
                ty: pair.clone(),
                attributes: Default::default(),
                ownership: Default::default(),
            }],
            vec![HirType::I64],
        ),
    );
    callee.values.insert(
        s,
        HirValue {
            id: s,
            ty: pair.clone(),
            kind: HirValueKind::Parameter(0),
            span: None,
        },
    );
    let field = callee.create_value(HirType::I64, HirValueKind::Instruction);
    let entry = callee.entry_block;
    let b = callee.blocks.get_mut(&entry).unwrap();
    b.instructions = vec![HirInstruction::ExtractValue {
        result: field,
        ty: HirType::I64,
        aggregate: s,
        indices: vec![1],
    }];
    b.terminator = HirTerminator::Return {
        values: vec![field],
    };

    let mut caller = HirFunction::new(
        InternedString::new_global("run"),
        signature(vec![], vec![HirType::I64]),
    );
    let size = constant(&mut caller, 16);
    let eight = constant(&mut caller, 8);
    let seven = constant(&mut caller, 7);
    let answer = constant(&mut caller, 42);
    let byte_ptr = HirType::Ptr(std::sync::Arc::new(HirType::U8));
    let object = caller.create_value(byte_ptr.clone(), HirValueKind::Instruction);
    let second_field = caller.create_value(byte_ptr, HirValueKind::Instruction);
    let result = caller.create_value(HirType::I64, HirValueKind::Instruction);
    let entry = caller.entry_block;
    let b = caller.blocks.get_mut(&entry).unwrap();
    b.instructions = vec![
        HirInstruction::Call {
            result: Some(object),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
        HirInstruction::Store {
            value: seven,
            ptr: object,
            align: 8,
            volatile: false,
        },
        HirInstruction::GetElementPtr {
            result: second_field,
            ty: HirType::U8,
            ptr: object,
            indices: vec![eight],
        },
        HirInstruction::Store {
            value: answer,
            ptr: second_field,
            align: 8,
            volatile: false,
        },
        HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Function(callee.id),
            args: vec![object],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
    ];
    b.terminator = HirTerminator::Return {
        values: vec![result],
    };
    let mut m = HirModule::new(InternedString::new_global("by_address"));
    m.functions.insert(callee.id, callee);
    m.functions.insert(caller.id, caller);
    m
}

#[test]
fn an_inlined_struct_parameter_reads_through_the_callers_address() {
    let mut m = module();
    assert_eq!(zyntax_compiler::inline::run_module(&mut m).inlined, 1);
    let mut interp = HirInterpreter::new();
    let v = interp
        .call(&m, "run", vec![])
        .unwrap_or_else(|e| panic!("run: {e:?}"));
    assert_eq!(value_to_i64(&v), Some(42), "{v:?}");
}
