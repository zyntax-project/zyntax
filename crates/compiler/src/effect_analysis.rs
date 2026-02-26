//! # Effect Analysis
//!
//! Provides effect inference and checking for algebraic effects at the HIR level.
//!
//! This module implements:
//! - Effect inference: Determines which effects a function may perform
//! - Effect propagation: Propagates effect requirements through the call graph
//! - Effect checking: Validates effect annotations and purity constraints
//! - Handler scope analysis: Ensures effect operations are within valid handler scopes

use crate::analysis::{CallGraph, ModuleAnalysis};
use crate::hir::{
    HirBlock, HirEffect, HirEffectHandler, HirFunction, HirId, HirInstruction, HirModule,
    HirTerminator,
};
use crate::CompilerError;
use crate::CompilerResult;
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet, VecDeque};
use zyntax_typed_ast::InternedString;

/// Effect analysis results for a module
#[derive(Debug, Clone)]
pub struct ModuleEffectAnalysis {
    /// Effect analysis for each function
    pub functions: HashMap<HirId, FunctionEffectAnalysis>,
    /// Effect call graph - shows effect dependencies between functions
    pub effect_call_graph: EffectCallGraph,
    /// All effects defined in the module
    pub defined_effects: HashMap<HirId, EffectInfo>,
    /// All handlers defined in the module
    pub defined_handlers: HashMap<HirId, HandlerInfo>,
    /// Validation errors found during analysis
    pub errors: Vec<EffectError>,
    /// Warnings found during analysis
    pub warnings: Vec<EffectWarning>,
}

/// Effect analysis for a single function
#[derive(Debug, Clone)]
pub struct FunctionEffectAnalysis {
    /// Effects directly performed by this function (via PerformEffect instructions)
    pub direct_effects: HashSet<EffectOccurrence>,
    /// Effects required by called functions (transitive)
    pub transitive_effects: HashSet<InternedString>,
    /// Total effect set (direct + transitive)
    pub total_effects: HashSet<InternedString>,
    /// Effect handlers active in this function
    pub handler_scopes: Vec<HandlerScope>,
    /// Whether this function is marked as pure
    pub is_pure: bool,
    /// Declared effects from function signature
    pub declared_effects: Vec<InternedString>,
    /// Effect operations with their locations
    pub effect_sites: Vec<EffectSite>,
}

/// A single occurrence of an effect operation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EffectOccurrence {
    /// The effect being performed
    pub effect_name: InternedString,
    /// The operation being invoked
    pub operation_name: InternedString,
    /// The block where this occurs
    pub block_id: HirId,
    /// Index within the block
    pub instruction_index: usize,
}

/// Information about an effect site in code
#[derive(Debug, Clone)]
pub struct EffectSite {
    /// Effect name
    pub effect: InternedString,
    /// Operation name
    pub operation: InternedString,
    /// Block ID
    pub block: HirId,
    /// Instruction index
    pub index: usize,
    /// Whether this is within a handler scope
    pub in_handler_scope: bool,
    /// The handler ID if within scope
    pub handler_id: Option<HirId>,
}

/// Handler scope tracking
#[derive(Debug, Clone)]
pub struct HandlerScope {
    /// Handler ID
    pub handler_id: HirId,
    /// Effect being handled
    pub effect_id: HirId,
    /// Effect name for lookup
    pub effect_name: InternedString,
    /// Entry block of the handler scope
    pub entry_block: HirId,
    /// All blocks within this handler scope
    pub scope_blocks: HashSet<HirId>,
}

/// Effect call graph
#[derive(Debug, Clone)]
pub struct EffectCallGraph {
    /// For each function, which effects it requires (directly or transitively)
    pub required_effects: HashMap<HirId, HashSet<InternedString>>,
    /// For each effect, which functions perform it
    pub effect_performers: HashMap<InternedString, HashSet<HirId>>,
    /// Functions that are effect-polymorphic (have effect type parameters)
    pub effect_polymorphic: HashSet<HirId>,
}

/// Information about a defined effect
#[derive(Debug, Clone)]
pub struct EffectInfo {
    pub id: HirId,
    pub name: InternedString,
    pub operations: Vec<InternedString>,
}

/// Information about a defined handler
#[derive(Debug, Clone)]
pub struct HandlerInfo {
    pub id: HirId,
    pub name: InternedString,
    pub handled_effect: HirId,
    pub handled_operations: Vec<InternedString>,
}

/// Effect analysis error
#[derive(Debug, Clone)]
pub struct EffectError {
    pub kind: EffectErrorKind,
    pub function_id: HirId,
    pub message: String,
    pub location: Option<EffectLocation>,
}

/// Effect analysis warning
#[derive(Debug, Clone)]
pub struct EffectWarning {
    pub kind: EffectWarningKind,
    pub function_id: HirId,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum EffectErrorKind {
    /// Pure function performs an effect
    PureViolation,
    /// Effect performed without handler in scope
    UnhandledEffect,
    /// Declared effect not actually performed
    UnusedEffectDeclaration,
    /// Effect operation not found in effect definition
    UnknownEffectOperation,
    /// Resume called outside handler
    InvalidResume,
    /// Handler doesn't handle all effect operations
    IncompleteHandler,
}

#[derive(Debug, Clone)]
pub enum EffectWarningKind {
    /// Effect declared but all instances are handled internally
    FullyHandledEffect,
    /// Unreachable effect operation
    UnreachableEffect,
}

#[derive(Debug, Clone)]
pub struct EffectLocation {
    pub block_id: HirId,
    pub instruction_index: usize,
}

/// Main effect analysis runner
pub struct EffectAnalyzer<'a> {
    module: &'a HirModule,
    call_graph: Option<&'a CallGraph>,
}

impl<'a> EffectAnalyzer<'a> {
    pub fn new(module: &'a HirModule) -> Self {
        Self {
            module,
            call_graph: None,
        }
    }

    pub fn with_call_graph(mut self, call_graph: &'a CallGraph) -> Self {
        self.call_graph = Some(call_graph);
        self
    }

    /// Run complete effect analysis
    pub fn analyze(&self) -> CompilerResult<ModuleEffectAnalysis> {
        let mut result = ModuleEffectAnalysis {
            functions: HashMap::new(),
            effect_call_graph: EffectCallGraph {
                required_effects: HashMap::new(),
                effect_performers: HashMap::new(),
                effect_polymorphic: HashSet::new(),
            },
            defined_effects: HashMap::new(),
            defined_handlers: HashMap::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        // Step 1: Collect effect and handler definitions
        self.collect_definitions(&mut result);

        // Step 2: Analyze each function for direct effects
        for (func_id, func) in &self.module.functions {
            let func_analysis = self.analyze_function(func, &result.defined_effects)?;
            result.functions.insert(*func_id, func_analysis);
        }

        // Step 3: Build effect call graph and propagate effects
        self.build_effect_call_graph(&mut result)?;

        // Step 4: Validate effect annotations
        self.validate_effects(&mut result)?;

        Ok(result)
    }

    /// Collect all effect and handler definitions from the module
    fn collect_definitions(&self, result: &mut ModuleEffectAnalysis) {
        // Collect effects
        for (effect_id, effect) in &self.module.effects {
            let operations: Vec<InternedString> =
                effect.operations.iter().map(|op| op.name).collect();

            result.defined_effects.insert(
                *effect_id,
                EffectInfo {
                    id: *effect_id,
                    name: effect.name,
                    operations,
                },
            );
        }

        // Collect handlers
        for (handler_id, handler) in &self.module.handlers {
            let handled_operations: Vec<InternedString> = handler
                .implementations
                .iter()
                .map(|impl_| impl_.op_name)
                .collect();

            result.defined_handlers.insert(
                *handler_id,
                HandlerInfo {
                    id: *handler_id,
                    name: handler.name,
                    handled_effect: handler.effect_id,
                    handled_operations,
                },
            );
        }
    }

    /// Analyze a single function for effect operations
    fn analyze_function(
        &self,
        func: &HirFunction,
        defined_effects: &HashMap<HirId, EffectInfo>,
    ) -> CompilerResult<FunctionEffectAnalysis> {
        let mut direct_effects = HashSet::new();
        let mut effect_sites = Vec::new();
        let mut handler_scopes = Vec::new();

        // Build handler scope map from HandleEffect instructions
        let scope_map = self.build_handler_scope_map(func);

        // Scan all blocks for effect operations
        for (block_id, block) in &func.blocks {
            for (inst_index, inst) in block.instructions.iter().enumerate() {
                match inst {
                    HirInstruction::PerformEffect {
                        effect_id, op_name, ..
                    } => {
                        // Look up effect name from effect_id
                        let effect_name = defined_effects
                            .get(effect_id)
                            .map(|e| e.name)
                            .unwrap_or_else(|| InternedString::new_global("unknown"));

                        let occurrence = EffectOccurrence {
                            effect_name,
                            operation_name: *op_name,
                            block_id: *block_id,
                            instruction_index: inst_index,
                        };
                        direct_effects.insert(occurrence);

                        // Check if within handler scope
                        let (in_scope, handler_id) =
                            self.check_handler_scope(*block_id, &scope_map, effect_name);

                        effect_sites.push(EffectSite {
                            effect: effect_name,
                            operation: *op_name,
                            block: *block_id,
                            index: inst_index,
                            in_handler_scope: in_scope,
                            handler_id,
                        });
                    }

                    HirInstruction::HandleEffect {
                        handler_id,
                        body_block,
                        continuation_block,
                        ..
                    } => {
                        // Get handler info
                        if let Some(handler) = self.module.handlers.get(handler_id) {
                            let effect_name = defined_effects
                                .get(&handler.effect_id)
                                .map(|e| e.name)
                                .unwrap_or_else(|| InternedString::new_global("unknown"));

                            // Compute blocks within handler scope
                            let scope_blocks = self.compute_handler_scope_blocks(
                                func,
                                *body_block,
                                *continuation_block,
                            );

                            handler_scopes.push(HandlerScope {
                                handler_id: *handler_id,
                                effect_id: handler.effect_id,
                                effect_name,
                                entry_block: *body_block,
                                scope_blocks,
                            });
                        }
                    }

                    _ => {}
                }
            }
        }

        // Extract declared effects from function signature
        let declared_effects = func.signature.effects.clone();

        Ok(FunctionEffectAnalysis {
            direct_effects,
            transitive_effects: HashSet::new(), // Filled in during propagation
            total_effects: HashSet::new(),      // Filled in during propagation
            handler_scopes,
            is_pure: func.signature.is_pure,
            declared_effects,
            effect_sites,
        })
    }

    /// Build a map from blocks to their enclosing handler scopes
    fn build_handler_scope_map(
        &self,
        func: &HirFunction,
    ) -> HashMap<HirId, Vec<(HirId, InternedString)>> {
        let mut scope_map: HashMap<HirId, Vec<(HirId, InternedString)>> = HashMap::new();

        // Find all HandleEffect instructions and their scope blocks
        for (_, block) in &func.blocks {
            for inst in &block.instructions {
                if let HirInstruction::HandleEffect {
                    handler_id,
                    body_block,
                    continuation_block,
                    ..
                } = inst
                {
                    if let Some(handler) = self.module.handlers.get(handler_id) {
                        if let Some(effect) = self.module.effects.get(&handler.effect_id) {
                            let scope_blocks = self.compute_handler_scope_blocks(
                                func,
                                *body_block,
                                *continuation_block,
                            );

                            for scope_block in scope_blocks {
                                scope_map
                                    .entry(scope_block)
                                    .or_default()
                                    .push((*handler_id, effect.name));
                            }
                        }
                    }
                }
            }
        }

        scope_map
    }

    /// Check if a block is within a handler scope for a given effect
    fn check_handler_scope(
        &self,
        block_id: HirId,
        scope_map: &HashMap<HirId, Vec<(HirId, InternedString)>>,
        effect_name: InternedString,
    ) -> (bool, Option<HirId>) {
        if let Some(handlers) = scope_map.get(&block_id) {
            for (handler_id, handled_effect) in handlers {
                if *handled_effect == effect_name {
                    return (true, Some(*handler_id));
                }
            }
        }
        (false, None)
    }

    /// Compute all blocks within a handler scope (from body_block to continuation_block)
    fn compute_handler_scope_blocks(
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
                continue; // Don't include continuation block
            }
            if !scope_blocks.insert(block_id) {
                continue; // Already visited
            }

            if let Some(block) = func.blocks.get(&block_id) {
                for succ in &block.successors {
                    worklist.push_back(*succ);
                }
            }
        }

        scope_blocks
    }

    /// Build effect call graph and propagate effects through call graph
    fn build_effect_call_graph(&self, result: &mut ModuleEffectAnalysis) -> CompilerResult<()> {
        // Initialize required effects with direct effects
        for (func_id, func_analysis) in &result.functions {
            let direct_effect_names: HashSet<InternedString> = func_analysis
                .direct_effects
                .iter()
                .map(|e| e.effect_name)
                .collect();

            result
                .effect_call_graph
                .required_effects
                .insert(*func_id, direct_effect_names.clone());

            // Track which functions perform each effect
            for effect_name in direct_effect_names {
                result
                    .effect_call_graph
                    .effect_performers
                    .entry(effect_name)
                    .or_default()
                    .insert(*func_id);
            }
        }

        // Propagate effects through call graph using fixed-point iteration
        if let Some(call_graph) = self.call_graph {
            let mut changed = true;
            while changed {
                changed = false;

                for (caller_id, callees) in &call_graph.direct_calls {
                    let mut caller_effects: HashSet<InternedString> = result
                        .effect_call_graph
                        .required_effects
                        .get(caller_id)
                        .cloned()
                        .unwrap_or_default();

                    let original_count = caller_effects.len();

                    // Add effects from all callees
                    for callee_id in callees {
                        if let Some(callee_effects) =
                            result.effect_call_graph.required_effects.get(callee_id)
                        {
                            // Only propagate effects that aren't handled by caller
                            let caller_analysis = result.functions.get(caller_id);
                            for effect in callee_effects {
                                let is_handled = caller_analysis
                                    .map(|a| {
                                        a.handler_scopes.iter().any(|h| h.effect_name == *effect)
                                    })
                                    .unwrap_or(false);

                                if !is_handled {
                                    caller_effects.insert(*effect);
                                }
                            }
                        }
                    }

                    if caller_effects.len() > original_count {
                        result
                            .effect_call_graph
                            .required_effects
                            .insert(*caller_id, caller_effects);
                        changed = true;
                    }
                }
            }
        }

        // Update function analyses with transitive effects
        for (func_id, func_analysis) in result.functions.iter_mut() {
            let direct_effects: HashSet<InternedString> = func_analysis
                .direct_effects
                .iter()
                .map(|e| e.effect_name)
                .collect();

            let total_effects = result
                .effect_call_graph
                .required_effects
                .get(func_id)
                .cloned()
                .unwrap_or_default();

            func_analysis.transitive_effects =
                total_effects.difference(&direct_effects).cloned().collect();

            func_analysis.total_effects = total_effects;
        }

        Ok(())
    }

    /// Validate effect annotations and constraints
    fn validate_effects(&self, result: &mut ModuleEffectAnalysis) -> CompilerResult<()> {
        for (func_id, func_analysis) in &result.functions {
            // Check 1: Pure functions must not perform any effects
            if func_analysis.is_pure && !func_analysis.total_effects.is_empty() {
                let effect_names: Vec<String> = func_analysis
                    .total_effects
                    .iter()
                    .filter_map(|e| e.resolve_global())
                    .collect();

                result.errors.push(EffectError {
                    kind: EffectErrorKind::PureViolation,
                    function_id: *func_id,
                    message: format!(
                        "Pure function performs effects: {}",
                        effect_names.join(", ")
                    ),
                    location: func_analysis.effect_sites.first().map(|s| EffectLocation {
                        block_id: s.block,
                        instruction_index: s.index,
                    }),
                });
            }

            // Check 2: All effect sites should be within handler scope OR declared in signature
            for site in &func_analysis.effect_sites {
                if !site.in_handler_scope {
                    // Check if effect is declared in function signature
                    let is_declared = func_analysis.declared_effects.contains(&site.effect);

                    if !is_declared {
                        result.errors.push(EffectError {
                            kind: EffectErrorKind::UnhandledEffect,
                            function_id: *func_id,
                            message: format!(
                                "Effect '{}' operation '{}' performed without handler or declaration",
                                site.effect.resolve_global().unwrap_or_default(),
                                site.operation.resolve_global().unwrap_or_default()
                            ),
                            location: Some(EffectLocation {
                                block_id: site.block,
                                instruction_index: site.index,
                            }),
                        });
                    }
                }
            }

            // Check 3: Declared effects should actually be performed or propagated
            let total_effect_names: HashSet<InternedString> = func_analysis.total_effects.clone();

            for declared in &func_analysis.declared_effects {
                if !total_effect_names.contains(declared) {
                    result.warnings.push(EffectWarning {
                        kind: EffectWarningKind::FullyHandledEffect,
                        function_id: *func_id,
                        message: format!(
                            "Declared effect '{}' is not performed by this function or its callees",
                            declared.resolve_global().unwrap_or_default()
                        ),
                    });
                }
            }
        }

        // Check 4: Validate handlers handle all required operations
        for (handler_id, handler_info) in &result.defined_handlers {
            if let Some(effect_info) = result.defined_effects.get(&handler_info.handled_effect) {
                let handled: HashSet<_> = handler_info.handled_operations.iter().collect();
                let required: HashSet<_> = effect_info.operations.iter().collect();

                let missing: Vec<_> = required.difference(&handled).collect();
                if !missing.is_empty() {
                    let missing_names: Vec<String> =
                        missing.iter().filter_map(|n| n.resolve_global()).collect();

                    result.errors.push(EffectError {
                        kind: EffectErrorKind::IncompleteHandler,
                        function_id: *handler_id, // Using handler_id as location
                        message: format!(
                            "Handler '{}' doesn't implement all operations for effect '{}': missing {}",
                            handler_info.name.resolve_global().unwrap_or_default(),
                            effect_info.name.resolve_global().unwrap_or_default(),
                            missing_names.join(", ")
                        ),
                        location: None,
                    });
                }
            }
        }

        Ok(())
    }
}

/// Run effect analysis on a module
pub fn analyze_effects(module: &HirModule) -> CompilerResult<ModuleEffectAnalysis> {
    let analyzer = EffectAnalyzer::new(module);
    analyzer.analyze()
}

/// Run effect analysis with call graph information
pub fn analyze_effects_with_call_graph(
    module: &HirModule,
    call_graph: &CallGraph,
) -> CompilerResult<ModuleEffectAnalysis> {
    let analyzer = EffectAnalyzer::new(module).with_call_graph(call_graph);
    analyzer.analyze()
}

/// Check if effect analysis found any errors
pub fn has_effect_errors(analysis: &ModuleEffectAnalysis) -> bool {
    !analysis.errors.is_empty()
}

/// Get a summary of effects for a function
pub fn get_function_effect_summary(
    analysis: &ModuleEffectAnalysis,
    func_id: HirId,
) -> Option<EffectSummary> {
    analysis
        .functions
        .get(&func_id)
        .map(|func_analysis| EffectSummary {
            direct_effects: func_analysis
                .direct_effects
                .iter()
                .map(|e| (e.effect_name, e.operation_name))
                .collect(),
            total_effects: func_analysis.total_effects.clone(),
            is_pure: func_analysis.is_pure,
            has_handlers: !func_analysis.handler_scopes.is_empty(),
        })
}

/// Summary of effects for display
#[derive(Debug, Clone)]
pub struct EffectSummary {
    pub direct_effects: Vec<(InternedString, InternedString)>,
    pub total_effects: HashSet<InternedString>,
    pub is_pure: bool,
    pub has_handlers: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::*;
    use indexmap::IndexMap;
    use std::collections::HashSet;
    use zyntax_typed_ast::TypeId;

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

    fn create_test_block(id: HirId) -> HirBlock {
        HirBlock {
            id,
            label: None,
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Return { values: vec![] },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        }
    }

    fn create_test_signature() -> HirFunctionSignature {
        HirFunctionSignature {
            params: vec![],
            returns: vec![],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_async: false,
            is_variadic: false,
            effects: vec![],
            is_pure: false,
        }
    }

    fn create_test_function(id: HirId, name: &str, is_pure: bool) -> HirFunction {
        let block_id = HirId::new();
        let mut blocks = IndexMap::new();
        blocks.insert(block_id, create_test_block(block_id));

        let mut sig = create_test_signature();
        sig.is_pure = is_pure;

        HirFunction {
            id,
            name: InternedString::new_global(name),
            signature: sig,
            entry_block: block_id,
            blocks,
            locals: IndexMap::new(),
            values: IndexMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
            link_name: None,
        }
    }

    #[test]
    fn test_empty_module_analysis() {
        let module = create_test_module();
        let result = analyze_effects(&module).unwrap();

        assert!(result.functions.is_empty());
        assert!(result.errors.is_empty());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_pure_function_no_effects() {
        let mut module = create_test_module();

        // Create a pure function with no effects
        let func_id = HirId::new();
        let func = create_test_function(func_id, "pure_fn", true);
        module.functions.insert(func_id, func);

        let result = analyze_effects(&module).unwrap();

        assert!(result.errors.is_empty());
        let func_analysis = result.functions.get(&func_id).unwrap();
        assert!(func_analysis.is_pure);
        assert!(func_analysis.total_effects.is_empty());
    }

    #[test]
    fn test_function_with_effects_declaration() {
        let mut module = create_test_module();

        // Create a function that declares effects
        let func_id = HirId::new();
        let mut func = create_test_function(func_id, "effectful_fn", false);

        // Add effect declaration to signature
        func.signature.effects = vec![InternedString::new_global("IO")];

        module.functions.insert(func_id, func);

        let result = analyze_effects(&module).unwrap();

        // Function declares effects but doesn't perform them - should warn
        let func_analysis = result.functions.get(&func_id).unwrap();
        assert!(!func_analysis.is_pure);
        assert_eq!(func_analysis.declared_effects.len(), 1);
    }
}
