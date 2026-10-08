//! Exact floating division by powers of two, after FMA contraction.
use crate::hir::HirConstant;
use cranelift_codegen::ir::InstBuilder;

fn reciprocal(value: &HirConstant) -> Option<HirConstant> {
    match value {
        HirConstant::F64(d) if d.is_normal() && d.to_bits() & ((1u64 << 52) - 1) == 0 => {
            let r = 1.0 / d;
            r.is_normal().then_some(HirConstant::F64(r))
        }
        HirConstant::F32(d) if d.is_normal() && d.to_bits() & ((1u32 << 23) - 1) == 0 => {
            let r = 1.0 / d;
            r.is_normal().then_some(HirConstant::F32(r))
        }
        _ => None,
    }
}

/// Lower here so HIR FMA contraction never treats division as an original multiply.
pub(crate) fn emit(
    builder: &mut cranelift_frontend::FunctionBuilder<'_>,
    lhs: cranelift_codegen::ir::Value,
    rhs: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    use cranelift_codegen::ir::{InstructionData, Opcode, ValueDef};
    // ZYNTAX_DISABLE_POWER2_DIV retains floating division; safe.
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*DISABLED.get_or_init(|| std::env::var_os("ZYNTAX_DISABLE_POWER2_DIV").is_some()) {
        let rhs = builder.func.dfg.resolve_aliases(rhs);
        if let ValueDef::Result(inst, _) = builder.func.dfg.value_def(rhs) {
            let value = match &builder.func.dfg.insts[inst] {
                InstructionData::UnaryIeee64 {
                    opcode: Opcode::F64const,
                    imm,
                } => Some(HirConstant::F64(f64::from_bits(imm.bits()))),
                InstructionData::UnaryIeee32 {
                    opcode: Opcode::F32const,
                    imm,
                } => Some(HirConstant::F32(f32::from_bits(imm.bits()))),
                _ => None,
            };
            if let Some(value) = value {
                match reciprocal(&value) {
                    Some(HirConstant::F64(r)) => {
                        let r = builder.ins().f64const(r);
                        return builder.ins().fmul(lhs, r);
                    }
                    Some(HirConstant::F32(r)) => {
                        let r = builder.ins().f32const(r);
                        return builder.ins().fmul(lhs, r);
                    }
                    _ => {}
                }
            }
        }
    }
    builder.ins().fdiv(lhs, rhs)
}
