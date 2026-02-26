//! # Effect Handler Resolution
//!
//! This module provides handler resolution for algebraic effects.
//!
//! Handler resolution determines which handler handles each effect operation
//! based on the dynamic handler stack established by HandleEffect instructions.
//!
//! ## Resolution Strategy
//!
//! 1. Build handler scope tree from HandleEffect instructions
//! 2. For each PerformEffect, find the innermost handler that handles the effect
//! 3. Record the resolution for use by codegen
//!
//! ## Optimization Opportunities
//!
//! When a handler is statically known:
//! - Inline simple handlers (single return, no continuation capture)
//! - Convert to direct function calls when handler is known at compile time
//! - Eliminate handler overhead for pure handlers

use crate::effect_analysis::{HandlerScope, ModuleEffectAnalysis};
use crate::hir::{
    HirBlock, HirEffect, HirEffectHandler, HirEffectHandlerImpl, HirFunction, HirId,
    HirInstruction, HirModule, HirTerminator, HirType,
};
use crate::CompilerError;
use crate::CompilerResult;
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet, VecDeque};
use zyntax_typed_ast::InternedString;

/// Resolution of which handler handles an effect operation
#[derive(Debug, Clone)]
pub struct HandlerResolution {
    /// Effect operation site
    pub perform_site: PerformSite,
    /// Resolved handler (if statically known)
    pub resolved_handler: Option<ResolvedHandler>,
    /// Whether this requires dynamic dispatch
    pub is_dynamic: bool,
}

/// Location of a PerformEffect instruction
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PerformSite {
    pub function_id: HirId,
    pub block_id: HirId,
    pub instruction_index: usize,
}

/// A resolved handler for an effect operation
#[derive(Debug, Clone)]
pub struct ResolvedHandler {
    /// Handler ID
    pub handler_id: HirId,
    /// Handler name (for debugging)
    pub handler_name: InternedString,
    /// The specific operation implementation
    pub impl_index: usize,
    /// Whether this handler can be inlined
    pub can_inline: bool,
    /// Optimization hint
    pub optimization: HandlerOptimization,
}

/// Optimization opportunities for a handler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandlerOptimization {
    /// No optimization possible (dynamic dispatch required)
    Dynamic,
    /// Handler is statically known, can generate direct call
    StaticDispatch,
    /// Handler can be fully inlined
    Inline,
    /// Handler is a simple return (no continuation capture)
    SimpleReturn,
    /// Handler is pure (no state mutation)
    Pure,
}

/// Results of handler resolution for a module
#[derive(Debug, Clone)]
pub struct ModuleHandlerResolution {
    /// Resolutions for each function
    pub functions: HashMap<HirId, FunctionHandlerResolution>,
    /// Handler inlining candidates
    pub inlinable_handlers: HashSet<HirId>,
    /// Statistics
    pub stats: ResolutionStats,
}

/// Handler resolution for a single function
#[derive(Debug, Clone)]
pub struct FunctionHandlerResolution {
    /// All handler resolutions in this function
    pub resolutions: Vec<HandlerResolution>,
    /// Handler scope tree for this function
    pub scope_tree: HandlerScopeTree,
}

/// Tree structure representing nested handler scopes
#[derive(Debug, Clone)]
pub struct HandlerScopeTree {
    /// Root scopes (handlers at function level)
    pub roots: Vec<HandlerScopeNode>,
    /// Map from block to containing scope
    pub block_to_scope: HashMap<HirId, HirId>,
}

/// Node in the handler scope tree
#[derive(Debug, Clone)]
pub struct HandlerScopeNode {
    /// Handler ID
    pub handler_id: HirId,
    /// Effect being handled
    pub effect_id: HirId,
    /// Effect name
    pub effect_name: InternedString,
    /// Blocks within this scope
    pub scope_blocks: HashSet<HirId>,
    /// Child handler scopes (nested handlers)
    pub children: Vec<HandlerScopeNode>,
}

/// Statistics about handler resolution
#[derive(Debug, Clone, Default)]
pub struct ResolutionStats {
    pub total_perform_sites: usize,
    pub statically_resolved: usize,
    pub dynamically_resolved: usize,
    pub inlinable: usize,
    pub simple_returns: usize,
}

/// Main handler resolution pass
pub struct HandlerResolver<'a> {
    module: &'a HirModule,
    effect_analysis: Option<&'a ModuleEffectAnalysis>,
}

impl<'a> HandlerResolver<'a> {
    pub fn new(module: &'a HirModule) -> Self {
        Self {
            module,
            effect_analysis: None,
        }
    }

    pub fn with_effect_analysis(mut self, analysis: &'a ModuleEffectAnalysis) -> Self {
        self.effect_analysis = Some(analysis);
        self
    }

    /// Run handler resolution
    pub fn resolve(&self) -> CompilerResult<ModuleHandlerResolution> {
        let mut result = ModuleHandlerResolution {
            functions: HashMap::new(),
            inlinable_handlers: HashSet::new(),
            stats: ResolutionStats::default(),
        };

        // First, identify inlinable handlers
        self.identify_inlinable_handlers(&mut result);

        // Resolve handlers for each function
        for (func_id, func) in &self.module.functions {
            let func_resolution = self.resolve_function(func, &result.inlinable_handlers)?;

            // Update stats
            for resolution in &func_resolution.resolutions {
                result.stats.total_perform_sites += 1;
                if resolution.is_dynamic {
                    result.stats.dynamically_resolved += 1;
                } else {
                    result.stats.statically_resolved += 1;
                    if let Some(handler) = &resolution.resolved_handler {
                        if handler.can_inline {
                            result.stats.inlinable += 1;
                        }
                        if handler.optimization == HandlerOptimization::SimpleReturn {
                            result.stats.simple_returns += 1;
                        }
                    }
                }
            }

            result.functions.insert(*func_id, func_resolution);
        }

        Ok(result)
    }

    /// Identify handlers that can be inlined
    fn identify_inlinable_handlers(&self, result: &mut ModuleHandlerResolution) {
        for (handler_id, handler) in &self.module.handlers {
            if self.is_handler_inlinable(handler) {
                result.inlinable_handlers.insert(*handler_id);
            }
        }
    }

    /// Check if a handler can be inlined
    fn is_handler_inlinable(&self, handler: &HirEffectHandler) -> bool {
        // Handler can be inlined if:
        // 1. All implementations are small (single block, few instructions)
        // 2. No continuation capture (is_resumable = false)
        // 3. No recursive handler calls

        for impl_ in &handler.implementations {
            // Skip if implementation uses continuation
            if impl_.is_resumable {
                return false;
            }

            // Skip if implementation is too large
            if impl_.blocks.len() > 2 {
                return false;
            }

            // Check instruction count in each block
            for (_, block) in &impl_.blocks {
                if block.instructions.len() > 10 {
                    return false;
                }

                // Check for recursive effect calls
                for inst in &block.instructions {
                    if let HirInstruction::PerformEffect { effect_id, .. } = inst {
                        if *effect_id == handler.effect_id {
                            return false; // Recursive
                        }
                    }
                }
            }
        }

        true
    }

    /// Resolve handlers for a single function
    fn resolve_function(
        &self,
        func: &HirFunction,
        inlinable_handlers: &HashSet<HirId>,
    ) -> CompilerResult<FunctionHandlerResolution> {
        // Build handler scope tree
        let scope_tree = self.build_scope_tree(func);

        // Resolve each PerformEffect instruction
        let mut resolutions = Vec::new();

        for (block_id, block) in &func.blocks {
            for (inst_index, inst) in block.instructions.iter().enumerate() {
                if let HirInstruction::PerformEffect {
                    effect_id, op_name, ..
                } = inst
                {
                    let site = PerformSite {
                        function_id: func.id,
                        block_id: *block_id,
                        instruction_index: inst_index,
                    };

                    let resolution = self.resolve_perform_site(
                        &site,
                        *effect_id,
                        *op_name,
                        &scope_tree,
                        inlinable_handlers,
                    );

                    resolutions.push(resolution);
                }
            }
        }

        Ok(FunctionHandlerResolution {
            resolutions,
            scope_tree,
        })
    }

    /// Build the handler scope tree for a function
    fn build_scope_tree(&self, func: &HirFunction) -> HandlerScopeTree {
        let mut roots = Vec::new();
        let mut block_to_scope = HashMap::new();

        // Find all HandleEffect instructions
        for (block_id, block) in &func.blocks {
            for inst in &block.instructions {
                if let HirInstruction::HandleEffect {
                    handler_id,
                    body_block,
                    continuation_block,
                    ..
                } = inst
                {
                    if let Some(handler) = self.module.handlers.get(handler_id) {
                        let effect_name = self
                            .module
                            .effects
                            .get(&handler.effect_id)
                            .map(|e| e.name)
                            .unwrap_or_else(|| InternedString::new_global("unknown"));

                        // Compute scope blocks
                        let scope_blocks =
                            self.compute_scope_blocks(func, *body_block, *continuation_block);

                        // Update block_to_scope mapping
                        for scope_block in &scope_blocks {
                            block_to_scope.insert(*scope_block, *handler_id);
                        }

                        let node = HandlerScopeNode {
                            handler_id: *handler_id,
                            effect_id: handler.effect_id,
                            effect_name,
                            scope_blocks,
                            children: Vec::new(), // TODO: Build nested structure
                        };

                        roots.push(node);
                    }
                }
            }
        }

        HandlerScopeTree {
            roots,
            block_to_scope,
        }
    }

    /// Compute all blocks within a handler scope
    fn compute_scope_blocks(
        &self,
        func: &HirFunction,
        body_block: HirId,
        continuation_block: HirId,
    ) -> HashSet<HirId> {
        let mut scope_blocks = HashSet::new();
        let mut worklist = VecDeque::new();

        worklist.push_back(body_block);

        while let Some(block_id) = worklist.pop_front() {
            if block_id == continuation_block {
                continue;
            }
            if !scope_blocks.insert(block_id) {
                continue;
            }

            if let Some(block) = func.blocks.get(&block_id) {
                for succ in &block.successors {
                    worklist.push_back(*succ);
                }
            }
        }

        scope_blocks
    }

    /// Resolve a single PerformEffect site
    fn resolve_perform_site(
        &self,
        site: &PerformSite,
        effect_id: HirId,
        op_name: InternedString,
        scope_tree: &HandlerScopeTree,
        inlinable_handlers: &HashSet<HirId>,
    ) -> HandlerResolution {
        // Find the handler for this effect in the scope tree
        let handler_scope = self.find_handler_for_effect(site.block_id, effect_id, scope_tree);

        match handler_scope {
            Some(scope_node) => {
                // Found a handler - resolve the specific operation
                let handler = self.module.handlers.get(&scope_node.handler_id);

                let resolved_handler = handler.and_then(|h| {
                    // Find the operation implementation
                    let impl_index = h
                        .implementations
                        .iter()
                        .position(|impl_| impl_.op_name == op_name)?;

                    let can_inline = inlinable_handlers.contains(&h.id);
                    let is_resumable = h.implementations[impl_index].is_resumable;

                    let optimization = if can_inline {
                        if !is_resumable && h.implementations[impl_index].blocks.len() == 1 {
                            HandlerOptimization::SimpleReturn
                        } else {
                            HandlerOptimization::Inline
                        }
                    } else {
                        HandlerOptimization::StaticDispatch
                    };

                    Some(ResolvedHandler {
                        handler_id: h.id,
                        handler_name: h.name,
                        impl_index,
                        can_inline,
                        optimization,
                    })
                });

                HandlerResolution {
                    perform_site: site.clone(),
                    resolved_handler,
                    is_dynamic: false,
                }
            }
            None => {
                // No handler found - requires dynamic dispatch
                HandlerResolution {
                    perform_site: site.clone(),
                    resolved_handler: None,
                    is_dynamic: true,
                }
            }
        }
    }

    /// Find the handler for an effect at a given block
    fn find_handler_for_effect<'t>(
        &self,
        block_id: HirId,
        effect_id: HirId,
        scope_tree: &'t HandlerScopeTree,
    ) -> Option<&'t HandlerScopeNode> {
        // Search from innermost to outermost scope
        for scope in &scope_tree.roots {
            if scope.effect_id == effect_id && scope.scope_blocks.contains(&block_id) {
                return Some(scope);
            }

            // Check nested scopes
            if let Some(nested) = self.find_in_children(block_id, effect_id, &scope.children) {
                return Some(nested);
            }
        }

        None
    }

    fn find_in_children<'b>(
        &self,
        block_id: HirId,
        effect_id: HirId,
        children: &'b [HandlerScopeNode],
    ) -> Option<&'b HandlerScopeNode> {
        for child in children {
            if child.effect_id == effect_id && child.scope_blocks.contains(&block_id) {
                return Some(child);
            }

            if let Some(nested) = self.find_in_children(block_id, effect_id, &child.children) {
                return Some(nested);
            }
        }
        None
    }
}

/// Convenience function to run handler resolution
pub fn resolve_handlers(module: &HirModule) -> CompilerResult<ModuleHandlerResolution> {
    let resolver = HandlerResolver::new(module);
    resolver.resolve()
}

/// Run handler resolution with effect analysis results
pub fn resolve_handlers_with_analysis(
    module: &HirModule,
    effect_analysis: &ModuleEffectAnalysis,
) -> CompilerResult<ModuleHandlerResolution> {
    let resolver = HandlerResolver::new(module).with_effect_analysis(effect_analysis);
    resolver.resolve()
}

/// Check if a handler resolution allows inlining
pub fn can_inline_handler(resolution: &HandlerResolution) -> bool {
    resolution
        .resolved_handler
        .as_ref()
        .map(|h| h.can_inline)
        .unwrap_or(false)
}

/// Get optimization level for a resolution
pub fn get_optimization_level(resolution: &HandlerResolution) -> HandlerOptimization {
    resolution
        .resolved_handler
        .as_ref()
        .map(|h| h.optimization)
        .unwrap_or(HandlerOptimization::Dynamic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::*;
    use std::collections::HashSet;

    fn create_test_module() -> HirModule {
        HirModule {
            id: HirId::new(),
            name: InternedString::new_global("test"),
            functions: IndexMap::new(),
            globals: IndexMap::new(),
            types: IndexMap::new(),
            imports: vec![],
            exports: vec![],
            version: 0,
            dependencies: HashSet::new(),
            effects: IndexMap::new(),
            handlers: IndexMap::new(),
        }
    }

    #[test]
    fn test_empty_module_resolution() {
        let module = create_test_module();
        let result = resolve_handlers(&module).unwrap();

        assert!(result.functions.is_empty());
        assert_eq!(result.stats.total_perform_sites, 0);
    }

    #[test]
    fn test_handler_inlining_detection() {
        let mut module = create_test_module();

        // Add a simple effect
        let effect_id = HirId::new();
        module.effects.insert(
            effect_id,
            HirEffect {
                id: effect_id,
                name: InternedString::new_global("Simple"),
                type_params: vec![],
                operations: vec![HirEffectOp {
                    id: HirId::new(),
                    name: InternedString::new_global("op"),
                    type_params: vec![],
                    params: vec![],
                    return_type: HirType::Void,
                }],
            },
        );

        // Add a simple handler (should be inlinable)
        let handler_id = HirId::new();
        let impl_block_id = HirId::new();
        let mut impl_blocks = IndexMap::new();
        impl_blocks.insert(
            impl_block_id,
            HirBlock {
                id: impl_block_id,
                label: None,
                phis: vec![],
                instructions: vec![],
                terminator: HirTerminator::Return { values: vec![] },
                dominance_frontier: HashSet::new(),
                predecessors: vec![],
                successors: vec![],
            },
        );

        module.handlers.insert(
            handler_id,
            HirEffectHandler {
                id: handler_id,
                name: InternedString::new_global("SimpleHandler"),
                effect_id,
                type_params: vec![],
                state_fields: vec![],
                implementations: vec![HirEffectHandlerImpl {
                    op_name: InternedString::new_global("op"),
                    type_params: vec![],
                    params: vec![],
                    return_type: HirType::Void,
                    entry_block: impl_block_id,
                    blocks: impl_blocks,
                    is_resumable: false,
                }],
            },
        );

        let result = resolve_handlers(&module).unwrap();

        // Handler should be identified as inlinable
        assert!(result.inlinable_handlers.contains(&handler_id));
    }
}
