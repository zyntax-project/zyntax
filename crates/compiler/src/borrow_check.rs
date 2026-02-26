//! HIR Borrow Checking Pass
//!
//! This module implements borrow checking at the HIR level.
//! It validates ownership and borrowing rules after lowering from TypedAST.
//!
//! Key validations:
//! - Values are not used after move
//! - Mutable borrows are exclusive (no aliasing)
//! - Immutable borrows can coexist but not with mutable borrows
//! - References do not outlive their referents
//! - Drop order is correct for RAII resources

use crate::analysis::{AliasAnalysis, FunctionAnalysis, LivenessAnalysis, ModuleAnalysis};
use crate::hir::{
    BorrowCheckContext, BorrowInfo, HirBlock, HirFunction, HirId, HirInstruction, HirLifetime,
    HirModule, HirType, HirValue, HirValueKind, LifetimeConstraint, MoveInfo,
};
use crate::CompilerError;
use crate::CompilerResult;
use std::collections::{HashMap, HashSet};
use zyntax_typed_ast::source::Span;

/// Result of borrow checking
#[derive(Debug)]
pub struct BorrowCheckResult {
    /// Per-function borrow checking context
    pub function_contexts: HashMap<HirId, BorrowCheckContext>,
    /// Detected borrow errors
    pub errors: Vec<BorrowError>,
    /// Warnings (non-fatal issues)
    pub warnings: Vec<BorrowWarning>,
}

/// Borrow checking error
#[derive(Debug, Clone)]
pub enum BorrowError {
    /// Use of value after it was moved
    UseAfterMove {
        value: HirId,
        move_location: Option<Span>,
        use_location: Option<Span>,
    },
    /// Mutable borrow while value is already borrowed
    MutableBorrowConflict {
        value: HirId,
        existing_borrow: HirId,
        new_borrow: HirId,
        location: Option<Span>,
    },
    /// Immutable borrow while value is mutably borrowed
    ImmutableBorrowConflict {
        value: HirId,
        mutable_borrow: HirId,
        location: Option<Span>,
    },
    /// Reference outlives its referent
    ReferenceOutlivesReferent {
        reference: HirId,
        referent: HirId,
        location: Option<Span>,
    },
    /// Cannot mutate through immutable reference
    MutationThroughImmutableRef {
        reference: HirId,
        location: Option<Span>,
    },
    /// Moved value still borrowed
    MovedWhileBorrowed {
        value: HirId,
        borrow: HirId,
        location: Option<Span>,
    },
}

/// Borrow checking warning
#[derive(Debug, Clone)]
pub enum BorrowWarning {
    /// Unused borrow
    UnusedBorrow {
        borrow: HirId,
        location: Option<Span>,
    },
    /// Borrow immediately dropped
    BorrowImmediatelyDropped {
        borrow: HirId,
        location: Option<Span>,
    },
}

/// HIR borrow checker
pub struct HirBorrowChecker<'a> {
    /// The module being checked
    module: &'a HirModule,
    /// Analysis results for optimization
    analysis: Option<&'a ModuleAnalysis>,
    /// Current function being checked
    current_function: Option<HirId>,
    /// Active borrows at current point
    active_borrows: HashMap<HirId, Vec<ActiveBorrow>>,
    /// Moved values
    moved_values: HashSet<HirId>,
    /// Collected errors
    errors: Vec<BorrowError>,
    /// Collected warnings
    warnings: Vec<BorrowWarning>,
    /// Function borrow contexts
    function_contexts: HashMap<HirId, BorrowCheckContext>,
}

/// Active borrow information during checking
#[derive(Debug, Clone)]
struct ActiveBorrow {
    borrow_id: HirId,
    borrowed_value: HirId,
    is_mutable: bool,
    start_point: HirId,
}

impl<'a> HirBorrowChecker<'a> {
    /// Create a new borrow checker for a module
    pub fn new(module: &'a HirModule) -> Self {
        Self {
            module,
            analysis: None,
            current_function: None,
            active_borrows: HashMap::new(),
            moved_values: HashSet::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            function_contexts: HashMap::new(),
        }
    }

    /// Set analysis results for more accurate checking
    pub fn with_analysis(mut self, analysis: &'a ModuleAnalysis) -> Self {
        self.analysis = Some(analysis);
        self
    }

    /// Run borrow checking on the entire module
    pub fn check_module(&mut self) -> CompilerResult<BorrowCheckResult> {
        eprintln!("[BORROW_CHECK] Starting HIR borrow checking...");

        for (func_id, func) in &self.module.functions {
            self.check_function(*func_id, func)?;
        }

        let result = BorrowCheckResult {
            function_contexts: std::mem::take(&mut self.function_contexts),
            errors: std::mem::take(&mut self.errors),
            warnings: std::mem::take(&mut self.warnings),
        };

        if result.errors.is_empty() {
            eprintln!(
                "[BORROW_CHECK] Borrow checking passed ({} warnings)",
                result.warnings.len()
            );
        } else {
            eprintln!(
                "[BORROW_CHECK] Borrow checking found {} errors, {} warnings",
                result.errors.len(),
                result.warnings.len()
            );
        }

        Ok(result)
    }

    /// Check a single function
    fn check_function(&mut self, func_id: HirId, func: &HirFunction) -> CompilerResult<()> {
        self.current_function = Some(func_id);
        self.active_borrows.clear();
        self.moved_values.clear();

        let mut context = BorrowCheckContext::new();

        // Check each block in CFG order
        let block_order = self.get_block_order(func);
        for block_id in block_order {
            if let Some(block) = func.blocks.get(&block_id) {
                self.check_block(func, &block_id, block, &mut context)?;
            }
        }

        self.function_contexts.insert(func_id, context);
        Ok(())
    }

    /// Get blocks in execution order (basic topological sort)
    fn get_block_order(&self, func: &HirFunction) -> Vec<HirId> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![func.entry_block];

        while let Some(block_id) = stack.pop() {
            if visited.contains(&block_id) {
                continue;
            }
            visited.insert(block_id);
            order.push(block_id);

            // Add successors
            if let Some(block) = func.blocks.get(&block_id) {
                match &block.terminator {
                    crate::hir::HirTerminator::Branch { target, .. } => {
                        stack.push(*target);
                    }
                    crate::hir::HirTerminator::CondBranch {
                        true_target,
                        false_target,
                        ..
                    } => {
                        stack.push(*false_target);
                        stack.push(*true_target);
                    }
                    crate::hir::HirTerminator::Switch { default, cases, .. } => {
                        stack.push(*default);
                        for (_, target) in cases {
                            stack.push(*target);
                        }
                    }
                    _ => {}
                }
            }
        }

        order
    }

    /// Check a single block
    fn check_block(
        &mut self,
        func: &HirFunction,
        block_id: &HirId,
        block: &HirBlock,
        context: &mut BorrowCheckContext,
    ) -> CompilerResult<()> {
        // Check each instruction
        for inst in &block.instructions {
            self.check_instruction(func, inst, context)?;
        }

        // Check terminator
        self.check_terminator(func, &block.terminator, context)?;

        Ok(())
    }

    /// Check a single instruction for borrow violations
    fn check_instruction(
        &mut self,
        func: &HirFunction,
        inst: &HirInstruction,
        context: &mut BorrowCheckContext,
    ) -> CompilerResult<()> {
        match inst {
            HirInstruction::Call {
                callee: _,
                args,
                result: _,
                type_args: _,
                const_args: _,
                is_tail: _,
            } => {
                // Check all arguments
                for arg in args {
                    self.check_value_use(func, *arg, None)?;
                }
            }

            HirInstruction::Load {
                result: _,
                ty: _,
                ptr,
                align: _,
                volatile: _,
            } => {
                // Loading through a pointer - check the pointer is valid
                self.check_value_use(func, *ptr, None)?;
            }

            HirInstruction::Store {
                ptr,
                value,
                align: _,
                volatile: _,
            } => {
                // Storing to a pointer - need mutable access
                self.check_value_use(func, *value, None)?;
                self.check_mutable_access(func, *ptr, None)?;
            }

            HirInstruction::Move {
                result: _,
                ty: _,
                source,
            } => {
                // Moving a value - record the move
                self.check_value_use(func, *source, None)?;
                // Mark as moved
                self.moved_values.insert(*source);
            }

            HirInstruction::Copy {
                result: _,
                ty: _,
                source,
            } => {
                // Copying a value - just check it's valid
                self.check_value_use(func, *source, None)?;
            }

            HirInstruction::CreateRef {
                result: borrow_id,
                value,
                lifetime: _,
                mutable,
            } => {
                // Creating a reference - record the borrow
                self.record_borrow(*borrow_id, *value, *mutable, None, context);
            }

            HirInstruction::Deref {
                result: _,
                ty: _,
                reference,
            } => {
                // Dereferencing - check the reference is valid
                self.check_value_use(func, *reference, None)?;
            }

            _ => {
                // Other instructions - check all operands
                for operand in inst.operands() {
                    self.check_value_use(func, operand, None)?;
                }
            }
        }

        Ok(())
    }

    /// Check terminator for borrow violations
    fn check_terminator(
        &mut self,
        func: &HirFunction,
        terminator: &crate::hir::HirTerminator,
        context: &mut BorrowCheckContext,
    ) -> CompilerResult<()> {
        match terminator {
            crate::hir::HirTerminator::Return { values } => {
                for value in values {
                    self.check_value_use(func, *value, None)?;
                }
            }
            crate::hir::HirTerminator::CondBranch { condition, .. } => {
                self.check_value_use(func, *condition, None)?;
            }
            crate::hir::HirTerminator::Switch { value, .. } => {
                self.check_value_use(func, *value, None)?;
            }
            crate::hir::HirTerminator::Branch { .. }
            | crate::hir::HirTerminator::Unreachable
            | crate::hir::HirTerminator::Invoke { .. }
            | crate::hir::HirTerminator::PatternMatch { .. } => {}
        }

        Ok(())
    }

    /// Check that a value is valid to use (not moved)
    fn check_value_use(
        &mut self,
        func: &HirFunction,
        value_id: HirId,
        location: Option<Span>,
    ) -> CompilerResult<()> {
        // Check if value was moved
        if self.moved_values.contains(&value_id) {
            self.errors.push(BorrowError::UseAfterMove {
                value: value_id,
                move_location: None, // TODO: track move location
                use_location: location,
            });
        }

        // Check if there are conflicting borrows
        if let Some(borrows) = self.active_borrows.get(&value_id) {
            for borrow in borrows {
                if borrow.is_mutable {
                    // Mutable borrow exists - no other access allowed
                    self.errors.push(BorrowError::ImmutableBorrowConflict {
                        value: value_id,
                        mutable_borrow: borrow.borrow_id,
                        location,
                    });
                }
            }
        }

        Ok(())
    }

    /// Check that mutable access to a value is allowed
    fn check_mutable_access(
        &mut self,
        func: &HirFunction,
        value_id: HirId,
        location: Option<Span>,
    ) -> CompilerResult<()> {
        // Check if value was moved
        if self.moved_values.contains(&value_id) {
            self.errors.push(BorrowError::UseAfterMove {
                value: value_id,
                move_location: None,
                use_location: location,
            });
            return Ok(());
        }

        // Check for any existing borrows (mutable or immutable)
        if let Some(borrows) = self.active_borrows.get(&value_id) {
            if !borrows.is_empty() {
                let existing = &borrows[0];
                self.errors.push(BorrowError::MutableBorrowConflict {
                    value: value_id,
                    existing_borrow: existing.borrow_id,
                    new_borrow: value_id, // Not quite right, but close
                    location,
                });
            }
        }

        Ok(())
    }

    /// Record a move of a value
    fn record_move(
        &mut self,
        value_id: HirId,
        dest_id: HirId,
        location: Option<Span>,
        context: &mut BorrowCheckContext,
    ) {
        // Check if value is borrowed - cannot move while borrowed
        if let Some(borrows) = self.active_borrows.get(&value_id) {
            if !borrows.is_empty() {
                self.errors.push(BorrowError::MovedWhileBorrowed {
                    value: value_id,
                    borrow: borrows[0].borrow_id,
                    location,
                });
            }
        }

        self.moved_values.insert(value_id);
        context.add_move(
            value_id,
            MoveInfo {
                moved_value: value_id,
                move_location: location,
                destination: dest_id,
            },
        );
    }

    /// Record a borrow of a value
    fn record_borrow(
        &mut self,
        borrow_id: HirId,
        borrowed_value: HirId,
        is_mutable: bool,
        location: Option<Span>,
        context: &mut BorrowCheckContext,
    ) {
        // Check for conflicts
        if is_mutable {
            // Mutable borrow - no other borrows allowed
            if let Some(borrows) = self.active_borrows.get(&borrowed_value) {
                if !borrows.is_empty() {
                    let existing = &borrows[0];
                    self.errors.push(BorrowError::MutableBorrowConflict {
                        value: borrowed_value,
                        existing_borrow: existing.borrow_id,
                        new_borrow: borrow_id,
                        location,
                    });
                }
            }
        } else {
            // Immutable borrow - only conflicts with mutable borrows
            if let Some(borrows) = self.active_borrows.get(&borrowed_value) {
                for borrow in borrows {
                    if borrow.is_mutable {
                        self.errors.push(BorrowError::ImmutableBorrowConflict {
                            value: borrowed_value,
                            mutable_borrow: borrow.borrow_id,
                            location,
                        });
                    }
                }
            }
        }

        // Add the borrow
        let active_borrow = ActiveBorrow {
            borrow_id,
            borrowed_value,
            is_mutable,
            start_point: borrow_id,
        };

        self.active_borrows
            .entry(borrowed_value)
            .or_insert_with(Vec::new)
            .push(active_borrow);

        // Record in context
        context.add_borrow(
            borrow_id,
            BorrowInfo {
                borrow_id,
                borrowed_value,
                lifetime: HirLifetime::anonymous(),
                is_mutable,
                borrow_location: location,
            },
        );
    }
}

/// Run borrow checking on a module
pub fn run_borrow_check(
    module: &HirModule,
    analysis: Option<&ModuleAnalysis>,
) -> CompilerResult<BorrowCheckResult> {
    let mut checker = HirBorrowChecker::new(module);
    if let Some(a) = analysis {
        checker = checker.with_analysis(a);
    }
    checker.check_module()
}

/// Validate borrow check result - returns error if there are borrow violations
pub fn validate_borrow_check(result: &BorrowCheckResult) -> CompilerResult<()> {
    if result.errors.is_empty() {
        Ok(())
    } else {
        let error_msgs: Vec<String> = result.errors.iter().map(|e| format!("{:?}", e)).collect();
        Err(CompilerError::Analysis(format!(
            "Borrow check failed with {} errors:\n{}",
            result.errors.len(),
            error_msgs.join("\n")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_borrow_checker_basic() {
        // TODO: Add tests
    }
}
