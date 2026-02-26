//! # Const Evaluation Engine
//!
//! Provides compile-time evaluation of const expressions for const generics,
//! const functions, and compile-time optimizations.

use crate::hir::{BinaryOp, HirConstant, HirFunction, HirId, HirInstruction, HirType, UnaryOp};
use crate::{CompilerError, CompilerResult};
use std::collections::HashMap;
use zyntax_typed_ast::InternedString;

/// Const evaluation context
#[derive(Debug)]
pub struct ConstEvalContext {
    /// Values of const parameters
    pub const_params: HashMap<InternedString, HirConstant>,
    /// Evaluated const values
    pub const_values: HashMap<HirId, HirConstant>,
    /// Current evaluation depth (for recursion limit)
    depth: usize,
    /// Maximum evaluation depth
    max_depth: usize,
}

impl ConstEvalContext {
    /// Create a new const evaluation context
    pub fn new() -> Self {
        Self {
            const_params: HashMap::new(),
            const_values: HashMap::new(),
            depth: 0,
            max_depth: 1000, // Prevent infinite recursion
        }
    }

    /// Add a const parameter value
    pub fn add_const_param(&mut self, name: InternedString, value: HirConstant) {
        self.const_params.insert(name, value);
    }

    /// Evaluate a const expression
    pub fn eval_const_expr(&mut self, expr: &HirInstruction) -> CompilerResult<HirConstant> {
        if self.depth > self.max_depth {
            return Err(CompilerError::Analysis(
                "Const evaluation recursion limit exceeded".into(),
            ));
        }

        self.depth += 1;
        let result = self.eval_instruction(expr);
        self.depth -= 1;

        result
    }

    /// Evaluate a HIR instruction as a const expression
    fn eval_instruction(&mut self, inst: &HirInstruction) -> CompilerResult<HirConstant> {
        match inst {
            HirInstruction::Binary {
                op, left, right, ..
            } => {
                let left_val = self.get_const_value(*left)?;
                let right_val = self.get_const_value(*right)?;
                self.eval_binary_op(*op, &left_val, &right_val)
            }

            HirInstruction::Unary { op, operand, .. } => {
                let operand_val = self.get_const_value(*operand)?;
                self.eval_unary_op(*op, &operand_val)
            }

            // Const intrinsics
            HirInstruction::Call {
                callee: crate::hir::HirCallable::Intrinsic(intrinsic),
                args: _,
                ..
            } => {
                match intrinsic {
                    crate::hir::Intrinsic::SizeOf => {
                        // Size of type - requires type information
                        // For now, return a placeholder
                        Ok(HirConstant::U64(8)) // Default size
                    }
                    crate::hir::Intrinsic::AlignOf => {
                        // Alignment of type
                        Ok(HirConstant::U64(8)) // Default alignment
                    }
                    _ => Err(CompilerError::Analysis(format!(
                        "Intrinsic {:?} not available in const context",
                        intrinsic
                    ))),
                }
            }

            _ => Err(CompilerError::Analysis(
                "Instruction not allowed in const context".into(),
            )),
        }
    }

    /// Get a const value by ID
    fn get_const_value(&self, id: HirId) -> CompilerResult<HirConstant> {
        self.const_values
            .get(&id)
            .cloned()
            .ok_or_else(|| CompilerError::Analysis("Value not available in const context".into()))
    }

    /// Evaluate a binary operation on constants
    pub fn eval_binary_op(
        &self,
        op: BinaryOp,
        left: &HirConstant,
        right: &HirConstant,
    ) -> CompilerResult<HirConstant> {
        use BinaryOp::*;
        use HirConstant::*;

        match (op, left, right) {
            // Integer arithmetic
            (Add, I64(a), I64(b)) => Ok(I64(a.wrapping_add(*b))),
            (Sub, I64(a), I64(b)) => Ok(I64(a.wrapping_sub(*b))),
            (Mul, I64(a), I64(b)) => Ok(I64(a.wrapping_mul(*b))),
            (Div, I64(a), I64(b)) => {
                if *b == 0 {
                    Err(CompilerError::Analysis(
                        "Division by zero in const expression".into(),
                    ))
                } else {
                    Ok(I64(a / b))
                }
            }
            (Rem, I64(a), I64(b)) => {
                if *b == 0 {
                    Err(CompilerError::Analysis(
                        "Remainder by zero in const expression".into(),
                    ))
                } else {
                    Ok(I64(a % b))
                }
            }

            // Unsigned arithmetic
            (Add, U64(a), U64(b)) => Ok(U64(a.wrapping_add(*b))),
            (Sub, U64(a), U64(b)) => Ok(U64(a.wrapping_sub(*b))),
            (Mul, U64(a), U64(b)) => Ok(U64(a.wrapping_mul(*b))),
            (Div, U64(a), U64(b)) => {
                if *b == 0 {
                    Err(CompilerError::Analysis(
                        "Division by zero in const expression".into(),
                    ))
                } else {
                    Ok(U64(a / b))
                }
            }

            // Bitwise operations
            (And, I64(a), I64(b)) => Ok(I64(a & b)),
            (Or, I64(a), I64(b)) => Ok(I64(a | b)),
            (Xor, I64(a), I64(b)) => Ok(I64(a ^ b)),
            (Shl, I64(a), I64(b)) => Ok(I64(a << (*b as u32))),
            (Shr, I64(a), I64(b)) => Ok(I64(a >> (*b as u32))),

            // Comparisons
            (Eq, I64(a), I64(b)) => Ok(Bool(a == b)),
            (Ne, I64(a), I64(b)) => Ok(Bool(a != b)),
            (Lt, I64(a), I64(b)) => Ok(Bool(a < b)),
            (Le, I64(a), I64(b)) => Ok(Bool(a <= b)),
            (Gt, I64(a), I64(b)) => Ok(Bool(a > b)),
            (Ge, I64(a), I64(b)) => Ok(Bool(a >= b)),

            _ => Err(CompilerError::Analysis(format!(
                "Const evaluation not implemented for {:?} with operands {:?} and {:?}",
                op, left, right
            ))),
        }
    }

    /// Evaluate a unary operation on a constant
    pub fn eval_unary_op(&self, op: UnaryOp, operand: &HirConstant) -> CompilerResult<HirConstant> {
        use HirConstant::*;
        use UnaryOp::*;

        match (op, operand) {
            (Neg, I64(a)) => Ok(I64(-a)),
            (Not, Bool(a)) => Ok(Bool(!a)),
            (Not, I64(a)) => Ok(I64(!a)),
            _ => Err(CompilerError::Analysis(format!(
                "Const evaluation not implemented for {:?} with operand {:?}",
                op, operand
            ))),
        }
    }
}

/// Const evaluator for the compiler
pub struct ConstEvaluator {
    context: ConstEvalContext,
}

impl ConstEvaluator {
    pub fn new() -> Self {
        Self {
            context: ConstEvalContext::new(),
        }
    }

    /// Evaluate a const function
    pub fn eval_const_function(
        &mut self,
        _func: &HirFunction,
        _args: Vec<HirConstant>,
    ) -> CompilerResult<HirConstant> {
        // For now, only support simple const functions
        // Full implementation would need to interpret HIR
        Err(CompilerError::Analysis(
            "Const function evaluation not yet implemented".into(),
        ))
    }

    /// Check if an expression is const-evaluable
    pub fn is_const_expr(&self, expr: &HirInstruction) -> bool {
        match expr {
            HirInstruction::Binary { .. } | HirInstruction::Unary { .. } => true,
            HirInstruction::Call {
                callee: crate::hir::HirCallable::Intrinsic(intrinsic),
                ..
            } => {
                matches!(
                    intrinsic,
                    crate::hir::Intrinsic::SizeOf | crate::hir::Intrinsic::AlignOf
                )
            }
            _ => false,
        }
    }

    /// Substitute const generic parameters in a type
    pub fn substitute_const_generics(
        &self,
        ty: &HirType,
        const_args: &HashMap<InternedString, HirConstant>,
    ) -> HirType {
        match ty {
            HirType::Array(elem_ty, size) => {
                let new_elem = self.substitute_const_generics(elem_ty, const_args);
                HirType::Array(Box::new(new_elem), *size)
            }
            HirType::ConstGeneric(name) => {
                // If we have a value for this const generic, we need to convert it to a type
                // This is a simplified version - real implementation would be more complex
                if let Some(value) = const_args.get(name) {
                    match value {
                        HirConstant::U64(n) => HirType::Array(Box::new(HirType::U8), *n), // Example: const N -> [u8; N]
                        _ => ty.clone(),
                    }
                } else {
                    ty.clone()
                }
            }
            HirType::Generic {
                base,
                type_args,
                const_args: type_const_args,
            } => {
                let new_base = self.substitute_const_generics(base, const_args);
                let new_type_args = type_args
                    .iter()
                    .map(|t| self.substitute_const_generics(t, const_args))
                    .collect();

                HirType::Generic {
                    base: Box::new(new_base),
                    type_args: new_type_args,
                    const_args: type_const_args.clone(),
                }
            }
            _ => ty.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_eval_arithmetic() {
        let ctx = ConstEvalContext::new();

        // Test simple arithmetic
        let result = ctx.eval_binary_op(BinaryOp::Add, &HirConstant::I64(5), &HirConstant::I64(3));
        assert_eq!(result.unwrap(), HirConstant::I64(8));

        let result = ctx.eval_binary_op(BinaryOp::Mul, &HirConstant::I64(4), &HirConstant::I64(7));
        assert_eq!(result.unwrap(), HirConstant::I64(28));

        // Test division by zero
        let result = ctx.eval_binary_op(BinaryOp::Div, &HirConstant::I64(10), &HirConstant::I64(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_const_eval_comparisons() {
        let ctx = ConstEvalContext::new();

        let result = ctx.eval_binary_op(BinaryOp::Lt, &HirConstant::I64(5), &HirConstant::I64(10));
        assert_eq!(result.unwrap(), HirConstant::Bool(true));

        let result = ctx.eval_binary_op(BinaryOp::Eq, &HirConstant::I64(7), &HirConstant::I64(7));
        assert_eq!(result.unwrap(), HirConstant::Bool(true));
    }

    #[test]
    fn test_const_eval_unary() {
        let ctx = ConstEvalContext::new();

        let result = ctx.eval_unary_op(UnaryOp::Neg, &HirConstant::I64(42));
        assert_eq!(result.unwrap(), HirConstant::I64(-42));

        let result = ctx.eval_unary_op(UnaryOp::Not, &HirConstant::Bool(true));
        assert_eq!(result.unwrap(), HirConstant::Bool(false));
    }
}
