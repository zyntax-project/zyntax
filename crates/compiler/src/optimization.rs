//! # Optimization Infrastructure
//!
//! Provides optimization passes that work on the HIR representation.
//! These optimizations are shared between Cranelift and LLVM backends.

use crate::analysis::ModuleAnalysis;
use crate::hir::{HirBlock, HirFunction, HirId, HirInstruction, HirModule};
use crate::CompilerResult;
use indexmap::IndexMap;
use std::collections::HashSet;

/// Optimization pass trait
pub trait OptimizationPass: Send + Sync {
    /// Name of this optimization
    fn name(&self) -> &'static str;

    /// Required analyses
    fn required_analyses(&self) -> &[&'static str];

    /// Run the optimization
    fn run(&mut self, module: &mut HirModule, analysis: &ModuleAnalysis) -> CompilerResult<bool>;
}

/// Optimization pipeline
pub struct OptimizationPipeline {
    passes: Vec<Box<dyn OptimizationPass>>,
    level: OptLevel,
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    None,    // -O0
    Less,    // -O1
    Default, // -O2
    More,    // -O3
}

impl OptimizationPipeline {
    pub fn new(level: OptLevel) -> Self {
        let mut pipeline = Self {
            passes: Vec::new(),
            level,
        };

        // Add passes based on optimization level
        pipeline.add_default_passes();

        pipeline
    }

    /// Add default optimization passes
    fn add_default_passes(&mut self) {
        use self::passes::*;

        match self.level {
            OptLevel::None => {
                // No optimizations
            }
            OptLevel::Less => {
                // Basic optimizations
                self.passes.push(Box::new(DeadCodeElimination::new()));
                self.passes.push(Box::new(ConstantFolding::new()));
            }
            OptLevel::Default => {
                // Standard optimizations
                self.passes.push(Box::new(DeadCodeElimination::new()));
                self.passes.push(Box::new(ConstantFolding::new()));
                self.passes
                    .push(Box::new(CommonSubexpressionElimination::new()));
                self.passes.push(Box::new(SimplifyCfg::new()));
            }
            OptLevel::More => {
                // Aggressive optimizations
                self.passes.push(Box::new(DeadCodeElimination::new()));
                self.passes.push(Box::new(ConstantFolding::new()));
                self.passes
                    .push(Box::new(CommonSubexpressionElimination::new()));
                self.passes.push(Box::new(SimplifyCfg::new()));
                self.passes.push(Box::new(LoopInvariantCodeMotion::new()));
                self.passes.push(Box::new(Inlining::new()));
            }
        }
    }

    /// Run all optimization passes
    pub fn run(&mut self, module: &mut HirModule, analysis: &ModuleAnalysis) -> CompilerResult<()> {
        let mut changed = true;
        let mut iteration = 0;

        // Fixed-point iteration
        while changed && iteration < 10 {
            changed = false;

            for pass in &mut self.passes {
                let pass_changed = pass.run(module, analysis)?;
                changed |= pass_changed;
            }

            iteration += 1;
        }

        Ok(())
    }
}

/// Standard optimization passes
pub mod passes {
    use super::*;

    /// Dead code elimination
    pub struct DeadCodeElimination {
        removed_instructions: usize,
    }

    impl DeadCodeElimination {
        pub fn new() -> Self {
            Self {
                removed_instructions: 0,
            }
        }
    }

    impl OptimizationPass for DeadCodeElimination {
        fn name(&self) -> &'static str {
            "dead-code-elimination"
        }

        fn required_analyses(&self) -> &[&'static str] {
            &["liveness"]
        }

        fn run(
            &mut self,
            module: &mut HirModule,
            analysis: &ModuleAnalysis,
        ) -> CompilerResult<bool> {
            let mut changed = false;

            for (func_id, func) in &mut module.functions {
                if let Some(func_analysis) = analysis.functions.get(func_id) {
                    changed |= self.eliminate_dead_code(func, &func_analysis.liveness)?;
                }
            }

            Ok(changed)
        }
    }

    impl DeadCodeElimination {
        fn eliminate_dead_code(
            &mut self,
            func: &mut HirFunction,
            liveness: &crate::analysis::LivenessAnalysis,
        ) -> CompilerResult<bool> {
            let mut changed = false;

            for (_, block) in &mut func.blocks {
                let mut new_instructions = Vec::new();

                for inst in &block.instructions {
                    // Keep instruction if it has side effects or its result is used
                    if self.has_side_effects(inst) || self.is_used(inst, &func.values) {
                        new_instructions.push(inst.clone());
                    } else {
                        self.removed_instructions += 1;
                        changed = true;
                    }
                }

                block.instructions = new_instructions;
            }

            Ok(changed)
        }

        fn has_side_effects(&self, inst: &HirInstruction) -> bool {
            matches!(
                inst,
                HirInstruction::Store { .. }
                    | HirInstruction::Call { .. }
                    | HirInstruction::Atomic { .. }
                    | HirInstruction::Fence { .. }
            )
        }

        fn is_used(
            &self,
            inst: &HirInstruction,
            values: &IndexMap<HirId, crate::hir::HirValue>,
        ) -> bool {
            if let Some(result) = self.get_result(inst) {
                if let Some(value) = values.get(&result) {
                    return !value.uses.is_empty();
                }
            }
            true // Conservative
        }

        fn get_result(&self, inst: &HirInstruction) -> Option<HirId> {
            match inst {
                HirInstruction::Binary { result, .. }
                | HirInstruction::Unary { result, .. }
                | HirInstruction::Alloca { result, .. }
                | HirInstruction::Load { result, .. }
                | HirInstruction::GetElementPtr { result, .. }
                | HirInstruction::Cast { result, .. }
                | HirInstruction::Select { result, .. }
                | HirInstruction::ExtractValue { result, .. }
                | HirInstruction::InsertValue { result, .. }
                | HirInstruction::Atomic { result, .. } => Some(*result),

                HirInstruction::Call { result, .. } => *result,

                _ => None,
            }
        }
    }

    /// Constant folding
    pub struct ConstantFolding {
        folded_count: usize,
    }

    impl ConstantFolding {
        pub fn new() -> Self {
            Self { folded_count: 0 }
        }
    }

    impl OptimizationPass for ConstantFolding {
        fn name(&self) -> &'static str {
            "constant-folding"
        }

        fn required_analyses(&self) -> &[&'static str] {
            &[]
        }

        fn run(
            &mut self,
            module: &mut HirModule,
            _analysis: &ModuleAnalysis,
        ) -> CompilerResult<bool> {
            let mut changed = false;

            for (_, func) in &mut module.functions {
                changed |= self.fold_constants(func)?;
            }

            Ok(changed)
        }
    }

    impl ConstantFolding {
        fn fold_constants(&mut self, func: &mut HirFunction) -> CompilerResult<bool> {
            // NOTE: Constant folding deferred to backend.
            // Requires: (1) Constant propagation, (2) Expression evaluation, (3) Instruction replacement
            // FUTURE (v2.0): Implement HIR-level constant folding pass
            // Estimated effort: 8-12 hours
            Ok(false)
        }
    }

    /// Common subexpression elimination
    pub struct CommonSubexpressionElimination {
        eliminated_count: usize,
    }

    impl CommonSubexpressionElimination {
        pub fn new() -> Self {
            Self {
                eliminated_count: 0,
            }
        }
    }

    impl OptimizationPass for CommonSubexpressionElimination {
        fn name(&self) -> &'static str {
            "common-subexpression-elimination"
        }

        fn required_analyses(&self) -> &[&'static str] {
            &["aliases"]
        }

        fn run(
            &mut self,
            module: &mut HirModule,
            _analysis: &ModuleAnalysis,
        ) -> CompilerResult<bool> {
            // NOTE: CSE deferred to backend.
            // Requires: (1) Value numbering, (2) Dominator tree, (3) Alias analysis integration
            // FUTURE (v2.0): Implement HIR-level CSE pass
            // Estimated effort: 10-15 hours
            Ok(false)
        }
    }

    /// Simplify control flow graph
    pub struct SimplifyCfg {
        merged_blocks: usize,
    }

    impl SimplifyCfg {
        pub fn new() -> Self {
            Self { merged_blocks: 0 }
        }
    }

    impl OptimizationPass for SimplifyCfg {
        fn name(&self) -> &'static str {
            "simplify-cfg"
        }

        fn required_analyses(&self) -> &[&'static str] {
            &[]
        }

        fn run(
            &mut self,
            module: &mut HirModule,
            _analysis: &ModuleAnalysis,
        ) -> CompilerResult<bool> {
            let mut changed = false;

            for (_, func) in &mut module.functions {
                changed |= self.simplify_function(func)?;
            }

            Ok(changed)
        }
    }

    impl SimplifyCfg {
        fn simplify_function(&mut self, func: &mut HirFunction) -> CompilerResult<bool> {
            let mut changed = false;

            // Merge blocks with single predecessor and successor
            let blocks_to_merge = self.find_mergeable_blocks(func);

            for (pred, succ) in blocks_to_merge {
                if self.merge_blocks(func, pred, succ)? {
                    self.merged_blocks += 1;
                    changed = true;
                }
            }

            Ok(changed)
        }

        fn find_mergeable_blocks(&self, func: &HirFunction) -> Vec<(HirId, HirId)> {
            let mut mergeable = Vec::new();

            for (block_id, block) in &func.blocks {
                if block.successors.len() == 1 {
                    let succ_id = block.successors[0];
                    if let Some(succ) = func.blocks.get(&succ_id) {
                        if succ.predecessors.len() == 1 && succ.phis.is_empty() {
                            mergeable.push((*block_id, succ_id));
                        }
                    }
                }
            }

            mergeable
        }

        fn merge_blocks(
            &self,
            func: &mut HirFunction,
            pred: HirId,
            succ: HirId,
        ) -> CompilerResult<bool> {
            // NOTE: Block merging partially implemented (detection done above, merging deferred).
            // Requires: (1) Instruction transfer, (2) Phi node updates, (3) CFG edge rewiring
            // FUTURE (v2.0): Complete block merging implementation
            // Estimated effort: 4-6 hours
            Ok(false)
        }
    }

    /// Loop invariant code motion
    pub struct LoopInvariantCodeMotion {
        hoisted_count: usize,
    }

    impl LoopInvariantCodeMotion {
        pub fn new() -> Self {
            Self { hoisted_count: 0 }
        }
    }

    impl OptimizationPass for LoopInvariantCodeMotion {
        fn name(&self) -> &'static str {
            "loop-invariant-code-motion"
        }

        fn required_analyses(&self) -> &[&'static str] {
            &["loops", "aliases"]
        }

        fn run(
            &mut self,
            module: &mut HirModule,
            analysis: &ModuleAnalysis,
        ) -> CompilerResult<bool> {
            // NOTE: LICM deferred to backend.
            // Requires: (1) Loop detection, (2) Invariant analysis, (3) Hoisting logic
            // FUTURE (v2.0): Implement HIR-level LICM pass
            // Estimated effort: 12-16 hours
            Ok(false)
        }
    }

    /// Function inlining
    pub struct Inlining {
        inlined_count: usize,
        max_size: usize,
    }

    impl Inlining {
        pub fn new() -> Self {
            Self {
                inlined_count: 0,
                max_size: 100, // Max instructions to inline
            }
        }
    }

    impl OptimizationPass for Inlining {
        fn name(&self) -> &'static str {
            "inlining"
        }

        fn required_analyses(&self) -> &[&'static str] {
            &["call-graph"]
        }

        fn run(
            &mut self,
            module: &mut HirModule,
            analysis: &ModuleAnalysis,
        ) -> CompilerResult<bool> {
            // NOTE: Inlining deferred to backend.
            // Requires: (1) Cost heuristics, (2) Function cloning, (3) Call site replacement, (4) CFG merging
            // FUTURE (v2.0): Implement HIR-level inlining pass
            // Estimated effort: 15-20 hours
            Ok(false)
        }
    }
}
