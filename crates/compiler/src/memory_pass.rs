//! Memory management optimization pass
//!
//! This pass applies memory management rules during compilation:
//! - Inserts reference counting operations (ARC)
//! - Adds drop calls for values going out of scope
//! - Performs escape analysis for stack allocation
//! - Optimizes memory operations

use crate::{
    analysis::ModuleAnalysis, hir::*, memory_management::*, optimization::OptimizationPass,
    CompilerResult,
};
use std::collections::HashMap;

/// Memory management optimization pass
pub struct MemoryManagementPass {
    /// Memory strategy to use
    strategy: MemoryStrategy,
    /// ARC manager for reference counting
    arc_manager: ARCManager,
    /// Drop manager for destructors
    drop_manager: DropManager,
    /// Escape analysis results
    escape_analysis: EscapeAnalysis,
    /// Memory contexts per function
    contexts: HashMap<HirId, MemoryContext>,
    /// Stack promotion pass
    stack_promotion: StackPromotionPass,
}

impl MemoryManagementPass {
    pub fn new(strategy: MemoryStrategy) -> Self {
        Self {
            strategy,
            arc_manager: ARCManager::new(),
            drop_manager: DropManager::new(),
            escape_analysis: EscapeAnalysis::new(),
            contexts: HashMap::new(),
            stack_promotion: StackPromotionPass::new(),
        }
    }

    /// Apply memory management rules to a module
    pub fn apply(&mut self, module: &mut HirModule) -> CompilerResult<()> {
        // First pass: Analyze all functions
        for (func_id, func) in &module.functions {
            let mut ctx = MemoryContext::new(self.strategy);

            // Run escape analysis
            let mut func_escape = EscapeAnalysis::new();
            func_escape.analyze(func)?;
            ctx.escape_info = func_escape.results;

            self.contexts.insert(*func_id, ctx);
        }

        // Second pass: Transform functions based on strategy
        for (func_id, func) in &mut module.functions {
            // Run stack promotion first using escape analysis results
            if let Some(ctx) = self.contexts.get(func_id) {
                self.stack_promotion
                    .promote_function(func, &ctx.escape_info)?;
            }

            match self.strategy {
                MemoryStrategy::ARC => {
                    self.apply_arc(func)?;
                }
                MemoryStrategy::Ownership => {
                    self.apply_ownership(func)?;
                }
                MemoryStrategy::Manual => {
                    self.apply_manual(func)?;
                }
                MemoryStrategy::GC => {
                    self.apply_gc(func)?;
                }
                MemoryStrategy::Hybrid => {
                    self.apply_hybrid(func)?;
                }
            }

            // Always insert drops for cleanup
            self.drop_manager.insert_drops(func)?;
        }

        // Log stack promotion summary
        let summary = self.stack_promotion.get_summary();
        if !summary.is_empty() {
            eprintln!("[MEMORY_PASS] {}", summary);
        }

        Ok(())
    }

    /// Apply ARC rules
    fn apply_arc(&mut self, func: &mut HirFunction) -> CompilerResult<()> {
        // Instrument function with retain/release
        self.arc_manager.instrument_function(func)?;

        // Optimize redundant retain/release pairs
        self.optimize_arc_operations(func)?;

        Ok(())
    }

    /// Apply ownership rules (like Rust)
    fn apply_ownership(&mut self, func: &mut HirFunction) -> CompilerResult<()> {
        let ctx = self.contexts.get(&func.id).unwrap();

        // Transform heap allocations to stack where possible
        for block in func.blocks.values_mut() {
            let mut new_instructions = Vec::new();

            for inst in &block.instructions {
                match inst {
                    HirInstruction::Call {
                        result: Some(result),
                        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                        args,
                        ..
                    } => {
                        // Check if we can stack allocate
                        if ctx.can_stack_allocate(*result) {
                            // Convert to alloca
                            new_instructions.push(HirInstruction::Alloca {
                                result: *result,
                                ty: HirType::I8, // Will be cast later
                                count: args.get(0).cloned(),
                                align: 8,
                            });
                        } else {
                            new_instructions.push(inst.clone());
                        }
                    }
                    _ => new_instructions.push(inst.clone()),
                }
            }

            block.instructions = new_instructions;
        }

        Ok(())
    }

    /// Apply manual memory management rules
    fn apply_manual(&mut self, func: &mut HirFunction) -> CompilerResult<()> {
        // For manual mode, just ensure all mallocs have corresponding frees
        // This is handled by the drop manager
        Ok(())
    }

    /// Apply garbage collection rules
    fn apply_gc(&mut self, func: &mut HirFunction) -> CompilerResult<()> {
        // Insert GC safepoints at function calls and loops
        for block in func.blocks.values_mut() {
            let mut new_instructions = Vec::new();

            for inst in &block.instructions {
                new_instructions.push(inst.clone());

                // Add GC safepoint after calls
                if matches!(inst, HirInstruction::Call { .. }) {
                    new_instructions.push(HirInstruction::Call {
                        result: None,
                        callee: HirCallable::Intrinsic(Intrinsic::GCSafepoint),
                        args: vec![],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    });
                }
            }

            block.instructions = new_instructions;
        }

        Ok(())
    }

    /// Apply hybrid strategy
    fn apply_hybrid(&mut self, func: &mut HirFunction) -> CompilerResult<()> {
        // Use ARC for heap objects, ownership for stack
        self.apply_ownership(func)?;
        self.apply_arc(func)?;
        Ok(())
    }

    /// Optimize redundant ARC operations
    fn optimize_arc_operations(&mut self, func: &mut HirFunction) -> CompilerResult<()> {
        // Simple optimization: remove retain/release pairs with no uses between
        for block in func.blocks.values_mut() {
            let mut optimized = Vec::new();
            let mut i = 0;

            while i < block.instructions.len() {
                let inst = &block.instructions[i];

                // Look for retain followed by release of same value
                if let HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::IncRef),
                    args: retain_args,
                    ..
                } = inst
                {
                    // Check next instruction
                    if i + 1 < block.instructions.len() {
                        if let HirInstruction::Call {
                            callee: HirCallable::Intrinsic(Intrinsic::DecRef),
                            args: release_args,
                            ..
                        } = &block.instructions[i + 1]
                        {
                            if retain_args == release_args {
                                // Skip both instructions
                                i += 2;
                                continue;
                            }
                        }
                    }
                }

                optimized.push(inst.clone());
                i += 1;
            }

            block.instructions = optimized;
        }

        Ok(())
    }
}

impl OptimizationPass for MemoryManagementPass {
    fn name(&self) -> &'static str {
        "memory-management"
    }

    fn required_analyses(&self) -> &[&'static str] {
        &["escape-analysis", "liveness"]
    }

    fn run(&mut self, module: &mut HirModule, _analysis: &ModuleAnalysis) -> CompilerResult<bool> {
        self.apply(module)?;
        Ok(true) // Always reports changes for now
    }
}

/// Intrinsic for GC safepoint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GCSafepoint;

// Extend the Intrinsic enum in HIR to include GCSafepoint
impl std::fmt::Display for GCSafepoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gc.safepoint")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::arena::AstArena;

    fn create_test_arena() -> AstArena {
        AstArena::new()
    }

    fn intern_str(arena: &mut AstArena, s: &str) -> zyntax_typed_ast::InternedString {
        arena.intern_string(s)
    }

    #[test]
    fn test_memory_pass_creation() {
        let pass = MemoryManagementPass::new(MemoryStrategy::ARC);
        assert_eq!(pass.strategy, MemoryStrategy::ARC);
    }

    #[test]
    fn test_memory_pass_application() {
        let mut arena = create_test_arena();
        let mut module = HirModule::new(intern_str(&mut arena, "test"));

        // Create a simple function with allocation
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::Ptr(Box::new(HirType::I32))],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };

        let mut func = HirFunction::new(intern_str(&mut arena, "alloc_test"), sig);
        let size = func.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(4)));
        let ptr = func.create_value(
            HirType::Ptr(Box::new(HirType::I32)),
            HirValueKind::Instruction,
        );

        let malloc = HirInstruction::Call {
            result: Some(ptr),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        };

        let block = func.blocks.get_mut(&func.entry_block).unwrap();
        block.add_instruction(malloc);
        block.set_terminator(HirTerminator::Return { values: vec![ptr] });

        module.add_function(func);

        // Apply memory management pass
        let mut pass = MemoryManagementPass::new(MemoryStrategy::ARC);
        let result = pass.apply(&mut module);
        assert!(result.is_ok());
    }
}
