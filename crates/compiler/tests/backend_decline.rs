//! A backend that has no encoding for a construct declines the function
//! rather than failing the module or dropping the instruction.
//!
//! Cranelift holds 128-bit vectors and nothing wider. A module is
//! entitled to contain wider ones, because a different backend can
//! compile them, so meeting one is not an error in the module. It used
//! to warn and continue, which left the instruction unemitted and
//! whatever read its result undefined.
//!
//! What must hold: the function is named as declined, the rest of the
//! module still compiles, and the decline is told apart from a defect
//! so that one is not buried among the other.
//!
//! A cast between widths is the other shape that has to decline rather
//! than be built. Cranelift's width-changing float casts resolve over
//! scalars only, so one over a vector is not something it can encode,
//! and building it anyway ends the process at verification rather than
//! returning an error a caller could act on.

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::{
    HirConstant, HirFunction, HirFunctionSignature, HirInstruction, HirModule, HirTerminator,
    HirType, HirValueKind,
};
use zyntax_typed_ast::InternedString;

/// The decline and skip counts are process-global, so a test that
/// resets one while another is reading it sees the wrong number. These
/// run one at a time.
static COUNTING: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ret_i64_sig() -> HirFunctionSignature {
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

/// `fn wide() -> i64 { splat(1.0) : <8 x f32>; 0 }` — a width this
/// backend has no register for.
fn wide_vector_function(name: &str) -> HirFunction {
    let mut f = HirFunction::new(InternedString::new_global(name), ret_i64_sig());
    let f32x8 = HirType::Vector(Box::new(HirType::F32), 8);
    let one = f.create_value(HirType::F32, HirValueKind::Constant(HirConstant::F32(1.0)));
    let zero = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
    let v = f.create_value(f32x8.clone(), HirValueKind::Instruction);
    let entry = f.entry_block;
    let blk = f.blocks.get_mut(&entry).unwrap();
    blk.instructions.push(HirInstruction::VectorSplat {
        result: v,
        ty: f32x8,
        scalar: one,
    });
    blk.terminator = HirTerminator::Return { values: vec![zero] };
    f
}

/// `fn narrow() -> i64 { 7 }` — nothing unusual, must be unaffected.
fn plain_function(name: &str) -> HirFunction {
    let mut f = HirFunction::new(InternedString::new_global(name), ret_i64_sig());
    let seven = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(7)));
    let entry = f.entry_block;
    let blk = f.blocks.get_mut(&entry).unwrap();
    blk.terminator = HirTerminator::Return {
        values: vec![seven],
    };
    f
}

/// `fn narrowing_cast() -> i64 { demote(splat(1.0) : <2 x f64>) : <2 x f32>; 0 }`
///
/// The broadcast is 128 bits, which this backend holds, so it is only
/// the cast that has no encoding. A wider broadcast would be declined
/// before the cast was ever reached and would say nothing about it.
fn wide_cast_function(name: &str) -> HirFunction {
    let mut f = HirFunction::new(InternedString::new_global(name), ret_i64_sig());
    let f64x4 = HirType::Vector(Box::new(HirType::F64), 2);
    let f32x4 = HirType::Vector(Box::new(HirType::F32), 2);
    let one = f.create_value(HirType::F64, HirValueKind::Constant(HirConstant::F64(1.0)));
    let zero = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
    let wide = f.create_value(f64x4.clone(), HirValueKind::Instruction);
    let narrow = f.create_value(f32x4.clone(), HirValueKind::Instruction);
    let entry = f.entry_block;
    let blk = f.blocks.get_mut(&entry).unwrap();
    blk.instructions.push(HirInstruction::VectorSplat {
        result: wide,
        ty: f64x4,
        scalar: one,
    });
    blk.instructions.push(HirInstruction::Cast {
        op: zyntax_compiler::hir::CastOp::FpTrunc,
        result: narrow,
        ty: f32x4,
        operand: wide,
    });
    blk.terminator = HirTerminator::Return { values: vec![zero] };
    f
}

/// A cast Cranelift cannot encode is declined, not built.
///
/// Built, it reaches a width constraint that resolves over scalars
/// only and ends the process at verification. The point of declining
/// is that the caller is left with something it can act on: another
/// backend, or the interpreter.
#[test]
fn a_cast_between_vector_widths_is_declined_not_built() {
    let wide = wide_cast_function("wide_cast");
    let plain = plain_function("beside_it");
    let (wide_id, plain_id) = (wide.id, plain.id);

    let mut module = HirModule::new(InternedString::new_global("m"));
    module.functions.insert(wide_id, wide);
    module.functions.insert(plain_id, plain);

    let _serialised = COUNTING.lock().unwrap_or_else(|e| e.into_inner());
    let mut backend = CraneliftBackend::new().expect("backend");
    backend
        .compile_module(&module)
        .expect("a cast this backend cannot encode must not fail the module");

    assert!(
        backend.declined_functions().contains(&wide_id),
        "the function holding the cast should be named as declined"
    );
    assert!(
        !backend.declined_functions().contains(&plain_id),
        "its neighbour should be untouched"
    );
}

/// The declined function is named, and its neighbour still compiles.
#[test]
fn a_width_without_an_encoding_is_declined_not_fatal() {
    let wide = wide_vector_function("wide");
    let plain = plain_function("narrow");
    let (wide_id, plain_id) = (wide.id, plain.id);

    let mut module = HirModule::new(InternedString::new_global("m"));
    module.functions.insert(wide_id, wide);
    module.functions.insert(plain_id, plain);

    let _serialised = COUNTING.lock().unwrap_or_else(|e| e.into_inner());
    zyntax_compiler::reset_cranelift_declined_function_count();
    let mut backend = CraneliftBackend::new().expect("backend");
    backend
        .compile_module(&module)
        .expect("a width this backend lacks must not fail the module");

    assert!(
        backend.declined_functions().contains(&wide_id),
        "the wide function should be named as declined so a caller can route it"
    );
    assert!(
        !backend.declined_functions().contains(&plain_id),
        "the ordinary function should be untouched"
    );
    assert_eq!(
        zyntax_compiler::cranelift_declined_function_count(),
        1,
        "exactly one function should have been declined"
    );
}

/// A decline is not a defect. The skip count records functions that
/// failed for some other reason, and mixing the two would hide real
/// ones among capability differences.
#[test]
fn a_decline_is_not_counted_as_a_defect() {
    let wide = wide_vector_function("wide_only");
    let wide_id = wide.id;
    let mut module = HirModule::new(InternedString::new_global("m2"));
    module.functions.insert(wide_id, wide);

    let _serialised = COUNTING.lock().unwrap_or_else(|e| e.into_inner());
    zyntax_compiler::reset_cranelift_declined_function_count();
    zyntax_compiler::reset_cranelift_skipped_function_count();
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compile");

    assert_eq!(zyntax_compiler::cranelift_declined_function_count(), 1);
    assert_eq!(
        zyntax_compiler::cranelift_skipped_function_count(),
        0,
        "a width this backend lacks is not a codegen defect"
    );
}
