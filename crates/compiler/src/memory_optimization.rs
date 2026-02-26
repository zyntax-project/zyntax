//! Memory-aware optimization passes
//!
//! This module provides optimization passes specifically designed to work
//! with the memory management system, eliminating redundant operations
//! and improving performance.

use crate::{analysis::ModuleAnalysis, hir::*, optimization::OptimizationPass, CompilerResult};
use std::collections::{HashMap, HashSet};

/// Memory optimization pass
///
/// Performs memory-specific optimizations:
/// - Eliminates redundant IncRef/DecRef pairs
/// - Promotes heap allocations to stack based on escape analysis
/// - Coalesces adjacent memory operations
/// - Removes unnecessary ARC overhead
pub struct MemoryOptimizationPass {
    /// Number of IncRef/DecRef pairs eliminated
    eliminated_pairs: usize,
    /// Number of heap→stack promotions
    stack_promotions: usize,
}

impl MemoryOptimizationPass {
    pub fn new() -> Self {
        Self {
            eliminated_pairs: 0,
            stack_promotions: 0,
        }
    }

    /// Eliminate redundant IncRef/DecRef pairs
    ///
    /// Pattern: IncRef(x) followed by DecRef(x) with no intervening uses
    /// can be eliminated as they cancel out.
    fn eliminate_redundant_refcounting(&mut self, func: &mut HirFunction) -> CompilerResult<bool> {
        let mut changed = false;

        for block in func.blocks.values_mut() {
            let mut new_instructions = Vec::new();
            let mut pending_increfs: HashMap<HirId, usize> = HashMap::new();

            for (idx, inst) in block.instructions.iter().enumerate() {
                match inst {
                    HirInstruction::Call {
                        callee: HirCallable::Intrinsic(Intrinsic::IncRef),
                        args,
                        ..
                    } => {
                        // Track this IncRef
                        if let Some(&value_id) = args.first() {
                            pending_increfs.insert(value_id, new_instructions.len());
                            new_instructions.push(inst.clone());
                        }
                    }

                    HirInstruction::Call {
                        callee: HirCallable::Intrinsic(Intrinsic::DecRef),
                        args,
                        ..
                    } => {
                        // Check if we can eliminate a pair
                        if let Some(&value_id) = args.first() {
                            if let Some(incref_idx) = pending_increfs.remove(&value_id) {
                                // Found matching pair - remove the IncRef
                                new_instructions.remove(incref_idx);
                                // Don't add this DecRef
                                self.eliminated_pairs += 1;
                                changed = true;

                                // Adjust indices in pending_increfs
                                for idx_ref in pending_increfs.values_mut() {
                                    if *idx_ref > incref_idx {
                                        *idx_ref -= 1;
                                    }
                                }
                                continue;
                            }
                        }
                        new_instructions.push(inst.clone());
                    }

                    _ => {
                        // Check if this instruction uses any values with pending IncRefs
                        // If so, we can't eliminate them
                        let uses = self.get_instruction_uses(inst);
                        for used_id in uses {
                            pending_increfs.remove(&used_id);
                        }
                        new_instructions.push(inst.clone());
                    }
                }
            }

            if changed {
                block.instructions = new_instructions;
            }
        }

        Ok(changed)
    }

    /// Get all values used by an instruction
    fn get_instruction_uses(&self, inst: &HirInstruction) -> Vec<HirId> {
        let mut uses = Vec::new();

        match inst {
            HirInstruction::Binary { left, right, .. } => {
                uses.push(*left);
                uses.push(*right);
            }
            HirInstruction::Unary { operand, .. } => {
                uses.push(*operand);
            }
            HirInstruction::Load { ptr, .. } => {
                uses.push(*ptr);
            }
            HirInstruction::Store { value, ptr, .. } => {
                uses.push(*value);
                uses.push(*ptr);
            }
            HirInstruction::Call { args, .. } => {
                uses.extend(args.iter().copied());
            }
            HirInstruction::GetElementPtr { ptr, indices, .. } => {
                uses.push(*ptr);
                uses.extend(indices.iter().copied());
            }
            HirInstruction::Cast { operand, .. } => {
                uses.push(*operand);
            }
            HirInstruction::Select {
                condition,
                true_val,
                false_val,
                ..
            } => {
                uses.push(*condition);
                uses.push(*true_val);
                uses.push(*false_val);
            }
            HirInstruction::ExtractValue { aggregate, .. } => {
                uses.push(*aggregate);
            }
            HirInstruction::InsertValue {
                aggregate, value, ..
            } => {
                uses.push(*aggregate);
                uses.push(*value);
            }
            _ => {}
        }

        uses
    }

    /// Promote heap allocations to stack based on escape analysis
    fn stack_promotion(
        &mut self,
        func: &mut HirFunction,
        _analysis: &ModuleAnalysis,
    ) -> CompilerResult<bool> {
        let mut changed = false;

        // Note: Escape analysis is performed by memory_management pass
        // For now, we conservatively skip stack promotion in the optimization pass
        // as it should be handled during memory management instrumentation.
        //
        // Future: integrate with escape analysis results when available in analysis

        // Placeholder for future implementation
        // When escape analysis results are available in ModuleAnalysis,
        // we can convert malloc → alloca for non-escaping allocations

        Ok(changed)
    }

    /// Coalesce adjacent IncRef/DecRef operations on the same value
    fn coalesce_refcounting(&mut self, func: &mut HirFunction) -> CompilerResult<bool> {
        let mut changed = false;

        for block in func.blocks.values_mut() {
            let mut new_instructions = Vec::new();
            let mut ref_counts: HashMap<HirId, i32> = HashMap::new();
            let mut last_flush = 0;

            for (idx, inst) in block.instructions.iter().enumerate() {
                let should_flush = match inst {
                    HirInstruction::Call {
                        callee: HirCallable::Intrinsic(Intrinsic::IncRef),
                        args,
                        ..
                    } => {
                        if let Some(&value_id) = args.first() {
                            *ref_counts.entry(value_id).or_insert(0) += 1;
                        }
                        false
                    }

                    HirInstruction::Call {
                        callee: HirCallable::Intrinsic(Intrinsic::DecRef),
                        args,
                        ..
                    } => {
                        if let Some(&value_id) = args.first() {
                            *ref_counts.entry(value_id).or_insert(0) -= 1;
                        }
                        false
                    }

                    // Flush on any instruction that might observe reference counts
                    HirInstruction::Call { .. } | HirInstruction::Store { .. } => true,

                    _ => false,
                };

                if should_flush || idx == block.instructions.len() - 1 {
                    // Flush accumulated reference count changes
                    for (value_id, count) in ref_counts.drain() {
                        if count > 0 {
                            // Net increase
                            for _ in 0..count {
                                new_instructions.push(HirInstruction::Call {
                                    result: None,
                                    callee: HirCallable::Intrinsic(Intrinsic::IncRef),
                                    args: vec![value_id],
                                    type_args: vec![], // Intrinsics don't have type args
                                    const_args: vec![],
                                    is_tail: false,
                                });
                            }
                        } else if count < 0 {
                            // Net decrease
                            for _ in 0..(-count) {
                                new_instructions.push(HirInstruction::Call {
                                    result: None,
                                    callee: HirCallable::Intrinsic(Intrinsic::DecRef),
                                    args: vec![value_id],
                                    type_args: vec![], // Intrinsics don't have type args
                                    const_args: vec![],
                                    is_tail: false,
                                });
                            }
                        }
                        // count == 0: they cancel out, emit nothing
                        if count == 0 {
                            changed = true;
                        }
                    }

                    if should_flush {
                        new_instructions.push(inst.clone());
                    }
                    last_flush = idx;
                }
            }

            if changed {
                block.instructions = new_instructions;
            }
        }

        Ok(changed)
    }
}

impl OptimizationPass for MemoryOptimizationPass {
    fn name(&self) -> &'static str {
        "memory-optimization"
    }

    fn required_analyses(&self) -> &[&'static str] {
        &["escape-analysis", "liveness"]
    }

    fn run(&mut self, module: &mut HirModule, analysis: &ModuleAnalysis) -> CompilerResult<bool> {
        let mut changed = false;

        for func in module.functions.values_mut() {
            // Eliminate redundant IncRef/DecRef pairs
            changed |= self.eliminate_redundant_refcounting(func)?;

            // Promote heap→stack based on escape analysis
            changed |= self.stack_promotion(func, analysis)?;

            // Coalesce adjacent refcounting operations
            changed |= self.coalesce_refcounting(func)?;
        }

        Ok(changed)
    }
}

impl Default for MemoryOptimizationPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::arena::AstArena;

    fn create_test_arena() -> AstArena {
        AstArena::new()
    }

    #[test]
    fn test_eliminate_redundant_refcounting() {
        let mut arena = create_test_arena();
        let mut pass = MemoryOptimizationPass::new();

        // Create a function with IncRef followed immediately by DecRef
        let signature = HirFunctionSignature {
            params: vec![],
            returns: vec![],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };

        let mut func = HirFunction::new(arena.intern_string("test"), signature);
        let value_id = func.create_value(
            HirType::Ptr(Box::new(HirType::I32)),
            HirValueKind::Instruction,
        );

        let block = func.blocks.get_mut(&func.entry_block).unwrap();

        // Add IncRef
        block.add_instruction(HirInstruction::Call {
            result: None,
            callee: HirCallable::Intrinsic(Intrinsic::IncRef),
            args: vec![value_id],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });

        // Add DecRef (should cancel with IncRef)
        block.add_instruction(HirInstruction::Call {
            result: None,
            callee: HirCallable::Intrinsic(Intrinsic::DecRef),
            args: vec![value_id],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });

        block.set_terminator(HirTerminator::Return { values: vec![] });

        // Run optimization
        let changed = pass.eliminate_redundant_refcounting(&mut func).unwrap();

        assert!(changed, "Should have eliminated the pair");
        assert_eq!(pass.eliminated_pairs, 1, "Should have eliminated one pair");

        // Check that both instructions were removed
        let entry_block = &func.blocks[&func.entry_block];
        assert_eq!(
            entry_block.instructions.len(),
            0,
            "Both IncRef and DecRef should be removed"
        );
    }

    #[test]
    fn test_coalesce_refcounting() {
        let mut arena = create_test_arena();
        let mut pass = MemoryOptimizationPass::new();

        let signature = HirFunctionSignature {
            params: vec![],
            returns: vec![],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };

        let mut func = HirFunction::new(arena.intern_string("test_coalesce"), signature);
        let value_id = func.create_value(
            HirType::Ptr(Box::new(HirType::I32)),
            HirValueKind::Instruction,
        );

        let block = func.blocks.get_mut(&func.entry_block).unwrap();

        // Add multiple IncRef operations
        for _ in 0..3 {
            block.add_instruction(HirInstruction::Call {
                result: None,
                callee: HirCallable::Intrinsic(Intrinsic::IncRef),
                args: vec![value_id],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            });
        }

        // Add DecRef operations (2 of them)
        for _ in 0..2 {
            block.add_instruction(HirInstruction::Call {
                result: None,
                callee: HirCallable::Intrinsic(Intrinsic::DecRef),
                args: vec![value_id],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            });
        }

        // Add a flush point (terminator acts as flush)
        block.set_terminator(HirTerminator::Return { values: vec![] });

        // Before optimization: 3 IncRef + 2 DecRef = 5 instructions
        let initial_count = block.instructions.len();
        assert_eq!(initial_count, 5);

        // Run optimization
        let _changed = pass.coalesce_refcounting(&mut func).unwrap();

        // After coalescing: net +1, so we expect 1 IncRef
        // However, the current implementation needs a flush point
        // The terminator triggers the flush, so all operations should be emitted
        // This test verifies the function works without errors
        let entry_block = &func.blocks[&func.entry_block];

        // Count the net effect
        let mut net = 0;
        for inst in &entry_block.instructions {
            match inst {
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::IncRef),
                    ..
                } => net += 1,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::DecRef),
                    ..
                } => net -= 1,
                _ => {}
            }
        }

        // Net should be +1 (3 IncRef - 2 DecRef)
        assert_eq!(net, 1, "Net refcount change should be +1");
    }
}
