//! # SSA Form Construction
//! 
//! Converts CFG to Static Single Assignment form for both Cranelift and LLVM.
//! Uses the efficient algorithm from "Simple and Efficient Construction of SSA Form"
//! by Braun et al.

use std::collections::{HashSet, VecDeque};
use indexmap::IndexMap;
use std::sync::Arc;
use zyntax_typed_ast::{InternedString, Type, ConstValue, typed_ast::{TypedNode, TypedExpression}};
use petgraph::visit::EdgeRef; // For .source() method on edges
use crate::hir::{
    HirId, HirFunction, HirBlock, HirInstruction, HirValueKind, HirConstant,
    HirType, HirPhi, HirTerminator, CastOp, HirParam, HirFunctionSignature
};
use crate::cfg::{ControlFlowGraph, BasicBlock};
use crate::CompilerResult;

/// SSA builder state
pub struct SsaBuilder {
    /// Current function being built
    function: HirFunction,
    /// Variable definitions per block
    definitions: IndexMap<HirId, IndexMap<InternedString, HirId>>,
    /// Incomplete phi nodes to be filled
    incomplete_phis: IndexMap<(HirId, InternedString), HirId>,
    /// Variable counter for versioning
    var_counter: IndexMap<InternedString, u32>,
    /// Type information for variables
    var_types: IndexMap<InternedString, HirType>,
    /// TypedAST type information for variables (preserves nominal types for method dispatch)
    var_typed_ast_types: IndexMap<InternedString, Type>,
    /// Sealed blocks (all predecessors known)
    sealed_blocks: HashSet<HirId>,
    /// Filled blocks (all definitions complete)
    filled_blocks: HashSet<HirId>,
    /// Type registry for field lookups and type information
    type_registry: Arc<zyntax_typed_ast::TypeRegistry>,
    /// Arena for interning strings
    arena: Arc<std::sync::Mutex<zyntax_typed_ast::AstArena>>,
    /// Generated closure functions (collected during translation)
    closure_functions: Vec<HirFunction>,
    /// Function symbol table for resolving function references
    function_symbols: IndexMap<InternedString, HirId>,
    /// Generated string globals (collected during translation)
    string_globals: Vec<crate::hir::HirGlobal>,
    /// Track which variables are written in each block (for loop phi placement)
    variable_writes: IndexMap<HirId, HashSet<InternedString>>,
    /// Flag: after IDF placement, don't create new phis
    idf_placement_done: bool,
    /// Current match context (scrutinee and discriminant for pattern matching)
    match_context: Option<MatchContext>,
    /// Continuation block for control flow expressions (if/match)
    /// When set, indicates that control flow has branched and this is the merge/end block
    continuation_block: Option<HirId>,
    /// Original return type from TypedAST (for auto-wrapping Result returns)
    original_return_type: Option<Type>,
    /// Variables that have their address taken (need stack allocation)
    address_taken_vars: HashSet<InternedString>,
    /// Stack slots for address-taken variables (var name -> alloca result)
    stack_slots: IndexMap<InternedString, HirId>,
    /// External function link names (alias -> ZRTL symbol)
    /// e.g., "tensor_add" -> "$Tensor$add"
    extern_link_names: IndexMap<InternedString, String>,
}

/// Context for pattern matching
#[derive(Debug, Clone)]
struct MatchContext {
    /// The scrutinee value being matched
    scrutinee_value: HirId,
    /// The extracted discriminant (for union types)
    discriminant_value: Option<HirId>,
    /// The union type being matched (if applicable)
    union_type: Option<Box<crate::hir::HirUnionType>>,
}

/// SSA form representation
#[derive(Debug)]
pub struct SsaForm {
    pub function: HirFunction,
    /// Def-use chains for optimization
    pub def_use_chains: IndexMap<HirId, HashSet<HirId>>,
    /// Use-def chains for analysis
    pub use_def_chains: IndexMap<HirId, HirId>,
    /// Closure functions generated during translation
    pub closure_functions: Vec<HirFunction>,
    /// String globals generated during translation
    pub string_globals: Vec<crate::hir::HirGlobal>,
}

/// Phi node during construction
#[derive(Debug, Clone)]
pub struct PhiNode {
    pub result: HirId,
    pub variable: InternedString,
    pub block: HirId,
    pub operands: Vec<(HirId, HirId)>, // (value, predecessor_block)
}

/// Dominance information for SSA construction
/// Computed using Cooper-Harvey-Kennedy algorithm
#[derive(Debug, Clone)]
struct DominanceInfo {
    /// Immediate dominator for each block
    idom: IndexMap<HirId, HirId>,
    /// Dominance frontiers for each block
    dom_frontier: IndexMap<HirId, HashSet<HirId>>,
    /// Reverse postorder traversal
    rpo: Vec<HirId>,
}

impl DominanceInfo {
    /// Compute dominance information from TypedCFG
    fn compute(cfg: &crate::typed_cfg::TypedControlFlowGraph) -> Self {
        use petgraph::visit::Dfs;
        use petgraph::Direction;

        // Step 1: Compute reverse postorder (RPO)
        let mut rpo = Vec::new();
        let mut visited = HashSet::new();
        Self::postorder_dfs(cfg, cfg.entry, &mut visited, &mut rpo);
        rpo.reverse();

        // Step 2: Compute immediate dominators using Cooper-Harvey-Kennedy
        let mut idom: IndexMap<HirId, HirId> = IndexMap::new();
        let entry_id = cfg.node_map[&cfg.entry];
        idom.insert(entry_id, entry_id); // Entry dominates itself

        let mut changed = true;
        while changed {
            changed = false;
            for &block_idx in rpo.iter().skip(1) {
                let block = &cfg.graph[block_idx];
                let block_id = block.id;

                // Find new idom from processed predecessors
                let mut new_idom = None;
                for edge in cfg.graph.edges_directed(block_idx, Direction::Incoming) {
                    let pred_idx = edge.source();
                    let pred_id = cfg.graph[pred_idx].id;

                    if idom.contains_key(&pred_id) {
                        new_idom = match new_idom {
                            None => Some(pred_id),
                            Some(current) => Some(Self::intersect(&idom, current, pred_id, &rpo, cfg)),
                        };
                    }
                }

                if let Some(new_idom_val) = new_idom {
                    if idom.get(&block_id) != Some(&new_idom_val) {
                        idom.insert(block_id, new_idom_val);
                        changed = true;
                    }
                }
            }
        }

        // Step 3: Compute dominance frontiers
        let dom_frontier = Self::compute_frontiers(cfg, &idom);

        DominanceInfo {
            idom,
            dom_frontier,
            rpo: rpo.iter().map(|&idx| cfg.graph[idx].id).collect(),
        }
    }

    /// Postorder DFS traversal
    fn postorder_dfs(
        cfg: &crate::typed_cfg::TypedControlFlowGraph,
        node: petgraph::graph::NodeIndex,
        visited: &mut HashSet<petgraph::graph::NodeIndex>,
        postorder: &mut Vec<petgraph::graph::NodeIndex>,
    ) {
        if visited.contains(&node) {
            return;
        }
        visited.insert(node);

        for edge in cfg.graph.edges_directed(node, petgraph::Direction::Outgoing) {
            Self::postorder_dfs(cfg, edge.target(), visited, postorder);
        }

        postorder.push(node);
    }

    /// Find intersection of two dominators
    fn intersect(
        idom: &IndexMap<HirId, HirId>,
        mut b1: HirId,
        mut b2: HirId,
        rpo: &[petgraph::graph::NodeIndex],
        cfg: &crate::typed_cfg::TypedControlFlowGraph,
    ) -> HirId {
        // Get RPO indices for blocks
        let rpo_map: IndexMap<HirId, usize> = rpo.iter()
            .enumerate()
            .map(|(i, &idx)| (cfg.graph[idx].id, i))
            .collect();

        while b1 != b2 {
            while rpo_map.get(&b1).unwrap_or(&usize::MAX) > rpo_map.get(&b2).unwrap_or(&usize::MAX) {
                b1 = idom[&b1];
            }
            while rpo_map.get(&b2).unwrap_or(&usize::MAX) > rpo_map.get(&b1).unwrap_or(&usize::MAX) {
                b2 = idom[&b2];
            }
        }
        b1
    }

    /// Compute dominance frontiers
    fn compute_frontiers(
        cfg: &crate::typed_cfg::TypedControlFlowGraph,
        idom: &IndexMap<HirId, HirId>,
    ) -> IndexMap<HirId, HashSet<HirId>> {
        use petgraph::Direction;

        let mut frontiers: IndexMap<HirId, HashSet<HirId>> = IndexMap::new();

        for node_idx in cfg.graph.node_indices() {
            let block = &cfg.graph[node_idx];
            let block_id = block.id;

            // Get predecessors
            let preds: Vec<_> = cfg.graph.edges_directed(node_idx, Direction::Incoming)
                .map(|e| cfg.graph[e.source()].id)
                .collect();

            if preds.len() >= 2 {
                // Block with multiple predecessors - compute its dominance frontier
                for &pred in &preds {
                    let mut runner = pred;
                    // Walk up dominator tree from pred until we reach block's idom
                    while runner != idom.get(&block_id).copied().unwrap_or(block_id) {
                        frontiers.entry(runner).or_insert_with(HashSet::new).insert(block_id);
                        let next_runner = idom.get(&runner).copied();
                        if next_runner.is_none() || next_runner == Some(runner) {
                            break; // Reached entry or self-dom
                        }
                        runner = next_runner.unwrap();
                    }
                }
            }
        }

        frontiers
    }

    /// Check if block `a` dominates block `b`
    fn dominates(&self, a: HirId, b: HirId) -> bool {
        let mut current = b;
        while let Some(&dom) = self.idom.get(&current) {
            if dom == a {
                return true;
            }
            if dom == current {
                break; // Reached entry
            }
            current = dom;
        }
        false
    }
}

impl SsaBuilder {
    pub fn new(
        function: HirFunction,
        type_registry: Arc<zyntax_typed_ast::TypeRegistry>,
        arena: Arc<std::sync::Mutex<zyntax_typed_ast::AstArena>>,
        function_symbols: IndexMap<InternedString, HirId>,
    ) -> Self {
        Self {
            function,
            definitions: IndexMap::new(),
            incomplete_phis: IndexMap::new(),
            var_counter: IndexMap::new(),
            var_types: IndexMap::new(),
            var_typed_ast_types: IndexMap::new(),
            sealed_blocks: HashSet::new(),
            filled_blocks: HashSet::new(),
            type_registry,
            arena,
            closure_functions: Vec::new(),
            function_symbols,
            string_globals: Vec::new(),
            variable_writes: IndexMap::new(),
            idf_placement_done: false,
            match_context: None,
            continuation_block: None,
            original_return_type: None,
            address_taken_vars: HashSet::new(),
            stack_slots: IndexMap::new(),
            extern_link_names: IndexMap::new(),
        }
    }

    /// Create SsaBuilder with original return type information
    /// This is needed for auto-wrapping return values in Result::Ok for error union types
    pub fn with_return_type(mut self, return_type: Type) -> Self {
        self.original_return_type = Some(return_type);
        self
    }

    /// Set external function link names for alias resolution
    /// Maps alias names (e.g., "tensor_add") to ZRTL symbols (e.g., "$Tensor$add")
    pub fn with_extern_link_names(mut self, link_names: IndexMap<InternedString, String>) -> Self {
        self.extern_link_names = link_names;
        self
    }
    
    /// Build SSA form from CFG
    pub fn build_from_cfg(mut self, cfg: &ControlFlowGraph) -> CompilerResult<SsaForm> {
        // Initialize blocks
        for (block_id, _) in &self.function.blocks {
            self.definitions.insert(*block_id, IndexMap::new());
        }

        // Process blocks in dominance order
        let dom_order = self.compute_dominance_order(cfg);

        for block_id in dom_order {
            self.process_block(block_id, cfg)?;
            self.seal_block(block_id);
        }

        // Fill remaining incomplete phis
        self.fill_incomplete_phis();

        // Build def-use chains
        let (def_use_chains, use_def_chains) = self.build_def_use_chains();

        Ok(SsaForm {
            function: self.function,
            def_use_chains,
            use_def_chains,
            closure_functions: self.closure_functions,
            string_globals: self.string_globals,
        })
    }

    /// Build SSA form from TypedControlFlowGraph
    /// This is the new approach that processes TypedAST directly
    pub fn build_from_typed_cfg(mut self, cfg: &crate::typed_cfg::TypedControlFlowGraph) -> CompilerResult<SsaForm> {
        // Create HirBlocks for all blocks in the CFG (except entry which already exists)
        for node_idx in cfg.graph.node_indices() {
            let typed_block = &cfg.graph[node_idx];
            let block_id = typed_block.id;

            // Skip if block already exists (entry block)
            if !self.function.blocks.contains_key(&block_id) {
                let hir_block = HirBlock::new(block_id);
                self.function.blocks.insert(block_id, hir_block);
            }
        }

        // Set up predecessors based on CFG edges
        for node_idx in cfg.graph.node_indices() {
            let typed_block = &cfg.graph[node_idx];
            let block_id = typed_block.id;

            // Find predecessors from incoming edges
            let mut predecessors = Vec::new();
            for edge in cfg.graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                let pred_node = edge.source();
                let pred_id = cfg.graph[pred_node].id;
                predecessors.push(pred_id);
            }

            if let Some(hir_block) = self.function.blocks.get_mut(&block_id) {
                hir_block.predecessors = predecessors;
            }
        }

        // Initialize definitions for all blocks
        for (block_id, _) in &self.function.blocks {
            self.definitions.insert(*block_id, IndexMap::new());
        }

        // CRITICAL FIX: Initialize function parameters as HIR values in entry block
        // Without this, parameters are treated as undefined variables and phi nodes are created!
        // Clone params first to avoid borrow checker issues
        let params = self.function.signature.params.clone();
        let entry_block = self.function.entry_block;

        for (param_index, param) in params.iter().enumerate() {
            eprintln!("[PARAM DEBUG] Param '{}' has HIR type: {:?}",
                param.name.resolve_global().unwrap_or_default(), param.ty);

            let param_value_id = self.create_value(param.ty.clone(), HirValueKind::Parameter(param_index as u32));

            // Store parameter type for SSA variable tracking
            self.var_types.insert(param.name, param.ty.clone());

            eprintln!("[PARAM DEBUG] Inserted into var_types: var_types['{}'] = {:?}",
                param.name.resolve_global().unwrap_or_default(), self.var_types.get(&param.name));

            // Define parameter in entry block so it's available to all code
            eprintln!("[PARAM DEBUG] write_variable({}, entry_block={:?})",
                param.name.resolve_global().unwrap_or_default(), entry_block);
            self.write_variable(param.name, entry_block, param_value_id);
        }

        // CRITICAL: Seal entry block immediately after defining parameters
        // Entry block has no predecessors, so it can be sealed right away
        // This allows other blocks to read parameters from the entry block
        self.seal_block(entry_block);
        self.filled_blocks.insert(entry_block);

        // CRITICAL: Scan for address-taken variables BEFORE processing blocks
        // These variables need stack allocation instead of SSA registers
        self.scan_cfg_for_address_taken_vars(cfg);

        // IDF-BASED SSA: Place phis using Iterated Dominance Frontier
        // CRITICAL: This must run BEFORE blocks are processed
        // It scans the CFG to find variable writes without translating to HIR
        self.place_phis_using_idf(cfg);

        // Mark IDF placement as done - no new phis should be created after this
        self.idf_placement_done = true;

        // CRITICAL: Propagate parameters to all blocks
        // Parameters don't need phis (they're never reassigned), but they need to be
        // available in all blocks. Copy them from entry block to all other blocks.
        let param_defs: Vec<_> = self.definitions.get(&entry_block)
            .map(|defs| defs.iter().map(|(&var, &val)| (var, val)).collect())
            .unwrap_or_default();

        for (block_id, _) in &self.function.blocks {
            if *block_id != entry_block {
                for (var, val) in &param_defs {
                    // Only copy if the block doesn't already have this variable
                    // (it might have a phi for variables that ARE reassigned)
                    if let Some(defs) = self.definitions.get_mut(block_id) {
                        defs.entry(*var).or_insert(*val);
                    }
                }
            }
        }

        // Process blocks in dominance-friendly order
        // For now, we use a simple worklist algorithm that processes blocks when their
        // non-back-edge predecessors are ready
        let mut worklist: Vec<_> = cfg.graph.node_indices().collect();
        let mut processed_blocks = std::collections::HashSet::new();
        processed_blocks.insert(entry_block);

        // Process entry block statements and terminator
        let mut found_entry = false;
        for node_idx in cfg.graph.node_indices() {
            let typed_block = &cfg.graph[node_idx];
            eprintln!("[SSA DEBUG] CFG block: typed_block.id={:?}, entry_block={:?}, match={}",
                typed_block.id, entry_block, typed_block.id == entry_block);
            if typed_block.id == entry_block {
                found_entry = true;
                // Track current block - try expressions may create continuation blocks
                let mut current_block = entry_block;
                eprintln!("[SSA DEBUG] Processing entry block {:?} with {} statements", entry_block, typed_block.statements.len());
                for stmt in &typed_block.statements {
                    current_block = self.process_statement(current_block, stmt)?;
                }
                self.process_typed_terminator(current_block, &typed_block.terminator, &typed_block.pattern_check)?;
                break;
            }
        }
        if !found_entry {
            eprintln!("[SSA DEBUG] WARNING: entry_block {:?} NOT FOUND in CFG!", entry_block);
        }

        // Keep processing until worklist is empty
        // Use a shared error cell to capture errors from the closure
        use std::cell::RefCell;
        let first_error: RefCell<Option<crate::CompilerError>> = RefCell::new(None);

        while !worklist.is_empty() {
            let mut made_progress = false;

            worklist.retain(|&node_idx| {
                // If we already have an error, just remove remaining items
                if first_error.borrow().is_some() {
                    return false;
                }

                let typed_block = &cfg.graph[node_idx];
                let block_id = typed_block.id;

                // Skip entry block - already processed
                if block_id == entry_block {
                    return false; // Remove from worklist
                }

                // Skip if already processed
                if processed_blocks.contains(&block_id) {
                    return false; // Remove from worklist
                }

                // Check if we can process this block
                // Process when at least one predecessor is filled for forward progress
                let block_info = self.function.blocks.get(&block_id).unwrap();
                let has_filled_pred = block_info.predecessors.iter().any(|pred| self.filled_blocks.contains(pred));

                if !has_filled_pred {
                    return true; // Keep in worklist
                }

                // Seal block only if all predecessors are filled
                let all_preds_filled = block_info.predecessors.iter().all(|pred| self.filled_blocks.contains(pred));
                if all_preds_filled && !self.sealed_blocks.contains(&block_id) {
                    self.seal_block(block_id);
                }

                // Extract pattern bindings if this is a match arm body
                if let Some(pattern_info) = &typed_block.pattern_check {
                    if let Some(variant_index) = pattern_info.variant_index {
                        if let Err(e) = self.extract_pattern_bindings(block_id, &pattern_info.pattern, variant_index) {
                            *first_error.borrow_mut() = Some(e);
                            return false;
                        }
                    }
                }

                // Process each TypedStatement in this block
                // Track current block - try expressions may create continuation blocks
                let mut current_block = block_id;
                for stmt in &typed_block.statements {
                    match self.process_statement(current_block, stmt) {
                        Ok(next_block) => current_block = next_block,
                        Err(e) => {
                            *first_error.borrow_mut() = Some(e);
                            return false;
                        }
                    }
                }

                // Process the terminator
                if let Err(e) = self.process_typed_terminator(current_block, &typed_block.terminator, &typed_block.pattern_check) {
                    *first_error.borrow_mut() = Some(e);
                    return false;
                }

                // Mark block as filled
                self.filled_blocks.insert(block_id);

                // Seal if not sealed yet
                if !self.sealed_blocks.contains(&block_id) {
                    self.seal_block(block_id);
                }

                processed_blocks.insert(block_id);
                made_progress = true;
                false // Remove from worklist
            });

            // Check if there was an error during the retain
            if let Some(err) = first_error.borrow_mut().take() {
                return Err(err);
            }

            if !made_progress && !worklist.is_empty() {
                // No progress made but worklist not empty - force process remaining blocks
                for &node_idx in &worklist {
                    let typed_block = &cfg.graph[node_idx];
                    let block_id = typed_block.id;

                    if processed_blocks.contains(&block_id) {
                        continue;
                    }

                    // Extract pattern bindings if this is a match arm body
                    if let Some(pattern_info) = &typed_block.pattern_check {
                        if let Some(variant_index) = pattern_info.variant_index {
                            self.extract_pattern_bindings(block_id, &pattern_info.pattern, variant_index)?;
                        }
                    }

                    // Track current block - try expressions may create continuation blocks
                    let mut current_block = block_id;
                    for stmt in &typed_block.statements {
                        current_block = self.process_statement(current_block, stmt)?;
                    }
                    self.process_typed_terminator(current_block, &typed_block.terminator, &typed_block.pattern_check)?;
                    self.filled_blocks.insert(block_id);
                    if !self.sealed_blocks.contains(&block_id) {
                        self.seal_block(block_id);
                    }
                    processed_blocks.insert(block_id);
                }
                break;
            }
        }

        // Fill remaining incomplete phis (from IDF placement)
        self.fill_incomplete_phis();

        // NOTE: verify_and_fix_phi_incoming() is DISABLED because it can create new phis
        // via read_variable(), which breaks Cranelift block parameter mapping.
        // If IDF is correct, all phis should already have all incoming values.
        // self.verify_and_fix_phi_incoming();

        // Build def-use chains
        let (def_use_chains, use_def_chains) = self.build_def_use_chains();

        Ok(SsaForm {
            function: self.function,
            def_use_chains,
            use_def_chains,
            closure_functions: self.closure_functions,
            string_globals: self.string_globals,
        })
    }

    /// Process a typed terminator
    fn process_typed_terminator(
        &mut self,
        block_id: HirId,
        terminator: &crate::typed_cfg::TypedTerminator,
        pattern_check: &Option<crate::typed_cfg::PatternCheckInfo>,
    ) -> CompilerResult<()> {
        use crate::typed_cfg::TypedTerminator;

        // First, translate expressions (requires mutable self)
        let hir_terminator = match terminator {
            TypedTerminator::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    let mut value_id = self.translate_expression(block_id, expr)?;

                    // Auto-wrap return value in Result::Ok if function returns error union (!T)
                    // and the expression is not already a Result type
                    if let Some(Type::Result { ok_type, err_type }) = &self.original_return_type {
                        // Check if the expression type is NOT already a Result
                        // If it's a plain value, wrap it in Ok(value)
                        if !matches!(&expr.ty, Type::Result { .. }) {
                            // Create Result union type
                            use crate::hir::{HirUnionType, HirUnionVariant};

                            let ok_hir_ty = self.convert_type(ok_type);
                            let err_hir_ty = self.convert_type(err_type);

                            let mut arena = self.arena.lock().unwrap();
                            let ok_name = arena.intern_string("Ok");
                            let err_name = arena.intern_string("Err");
                            drop(arena);

                            let union_ty = HirUnionType {
                                name: None,
                                variants: vec![
                                    HirUnionVariant {
                                        name: ok_name,
                                        ty: ok_hir_ty.clone(),
                                        discriminant: 0,
                                    },
                                    HirUnionVariant {
                                        name: err_name,
                                        ty: err_hir_ty,
                                        discriminant: 1,
                                    },
                                ],
                                discriminant_type: Box::new(HirType::U32),
                                is_c_union: false,
                            };

                            let union_hir_ty = HirType::Union(Box::new(union_ty));
                            let result_id = self.create_value(
                                union_hir_ty.clone(),
                                HirValueKind::Instruction
                            );

                            self.add_instruction(
                                block_id,
                                HirInstruction::CreateUnion {
                                    result: result_id,
                                    union_ty: union_hir_ty,
                                    variant_index: 0, // Ok variant
                                    value: value_id,
                                },
                            );

                            value_id = result_id;
                        }
                    }

                    // Check if the function returns void - if so, ignore the expression value
                    let is_void_return = matches!(
                        &self.original_return_type,
                        Some(Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit)) | None
                    );

                    // Check if the expression set a continuation block (for control flow expressions)
                    if let Some(continuation) = self.continuation_block.take() {
                        // Control flow expression - set Return on continuation block, not entry block
                        let cont_block = self.function.blocks.get_mut(&continuation)
                            .ok_or_else(|| crate::CompilerError::Analysis("Continuation block not found".into()))?;
                        cont_block.terminator = if is_void_return {
                            HirTerminator::Return { values: vec![] }
                        } else {
                            HirTerminator::Return { values: vec![value_id] }
                        };

                        // Entry block already has correct terminator (Branch/CondBranch), so return None
                        // to signal that we shouldn't overwrite it
                        return Ok(());
                    } else {
                        // Regular expression - return normally from entry block
                        if is_void_return {
                            HirTerminator::Return { values: vec![] }
                        } else {
                            HirTerminator::Return { values: vec![value_id] }
                        }
                    }
                } else {
                    HirTerminator::Return { values: vec![] }
                }
            }

            TypedTerminator::Jump(target) => {
                // Check if this is a pattern check block (has pattern_check with false_target)
                if let Some(pattern_info) = pattern_check {
                    if let (Some(variant_index), Some(false_target)) =
                        (pattern_info.variant_index, pattern_info.false_target) {
                        // This is an enum pattern check - generate discriminant comparison
                        self.generate_pattern_discriminant_check(
                            block_id,
                            *target,
                            false_target,
                            variant_index,
                        )?
                    } else {
                        // Body block or non-enum pattern, just jump
                        HirTerminator::Branch { target: *target }
                    }
                } else {
                    // Check if a try expression created a continuation block
                    // If so, the Jump should be placed on the continuation block
                    if let Some(continuation) = self.continuation_block.take() {
                        // Update predecessor relationship: continuation -> target
                        {
                            let target_block = self.function.blocks.get_mut(target).unwrap();
                            // Replace block_id with continuation in predecessors
                            if let Some(pos) = target_block.predecessors.iter().position(|&p| p == block_id) {
                                target_block.predecessors[pos] = continuation;
                            } else {
                                target_block.predecessors.push(continuation);
                            }
                        }
                        // Update successor relationship
                        {
                            let cont_block = self.function.blocks.get_mut(&continuation).unwrap();
                            cont_block.successors.push(*target);
                            cont_block.terminator = HirTerminator::Branch { target: *target };
                        }
                        // Return early - don't set terminator on original block
                        return Ok(());
                    }
                    // Regular jump, no pattern check
                    HirTerminator::Branch { target: *target }
                }
            }

            TypedTerminator::CondBranch { condition, true_target, false_target } => {
                let cond_val = self.translate_expression(block_id, condition)?;
                HirTerminator::CondBranch {
                    condition: cond_val,
                    true_target: *true_target,
                    false_target: *false_target,
                }
            }

            TypedTerminator::Unreachable => {
                // Handle implicit returns for functions without explicit return statements
                match &self.original_return_type {
                    Some(Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit)) | None => {
                        // Void/Unit return or no return type specified - implicit void return
                        HirTerminator::Return { values: vec![] }
                    }
                    Some(_return_ty) => {
                        // Non-void function: check if last statement in the block is an expression to implicitly return
                        // We need to look at the TypedBasicBlock to see if there's an expression statement at the end
                        // that we should implicitly return
                        // For now, keep as Unreachable - the type checker should catch missing returns
                        HirTerminator::Unreachable
                    }
                }
            }
        };

        // Then set it (requires mutable block access)
        let hir_block = self.function.blocks.get_mut(&block_id)
            .ok_or_else(|| crate::CompilerError::Analysis("Block not found".into()))?;
        hir_block.terminator = hir_terminator;

        Ok(())
    }

    /// Generate discriminant check for pattern matching
    /// Returns a CondBranch terminator that checks if discriminant == variant_index
    fn generate_pattern_discriminant_check(
        &mut self,
        block_id: HirId,
        true_target: HirId,
        false_target: HirId,
        variant_index: u32,
    ) -> CompilerResult<HirTerminator> {
        // Get the discriminant from match context
        let discriminant_val = self.match_context
            .as_ref()
            .and_then(|ctx| ctx.discriminant_value)
            .ok_or_else(|| crate::CompilerError::Analysis(
                "Pattern check block has no match context with discriminant".into()
            ))?;

        // Create constant for the variant index we're checking
        let variant_const = crate::hir::HirConstant::U32(variant_index);
        let const_id = HirId::new();
        self.function.values.insert(const_id, crate::hir::HirValue {
            id: const_id,
            ty: HirType::U32,
            kind: crate::hir::HirValueKind::Constant(variant_const),
            uses: HashSet::new(),
            span: None,
        });

        // Generate comparison: discriminant == variant_index
        let cmp_result = HirId::new();
        self.add_instruction(
            block_id,
            HirInstruction::Binary {
                result: cmp_result,
                op: crate::hir::BinaryOp::Eq,
                left: discriminant_val,
                right: const_id,
                ty: HirType::Bool,
            },
        );

        log::debug!("[SSA] Generated pattern check: discriminant({:?}) == {} -> {:?}",
                 discriminant_val, variant_index, cmp_result);

        // Use the false_target provided by CFG (next pattern check or unreachable block)
        Ok(HirTerminator::CondBranch {
            condition: cmp_result,
            true_target,
            false_target,
        })
    }

    /// Extract pattern bindings for a match arm body
    /// Generates ExtractUnionValue instructions for pattern variables
    fn extract_pattern_bindings(
        &mut self,
        block_id: HirId,
        pattern: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedPattern>,
        variant_index: u32,
    ) -> CompilerResult<()> {
        use zyntax_typed_ast::typed_ast::TypedPattern;

        // Get the scrutinee value from match context
        let (scrutinee_val, union_type) = if let Some(ctx) = &self.match_context {
            (ctx.scrutinee_value, ctx.union_type.clone())
        } else {
            return Ok(()); // No match context, nothing to extract
        };

        let union_type = match union_type {
            Some(ty) => ty,
            None => return Ok(()), // Not a union type, no extraction needed
        };

        // Extract bindings from the pattern
        match &pattern.node {
            TypedPattern::Enum { fields, .. } => {
                // For patterns like Some(x), extract the inner value
                if fields.len() == 1 {
                    if let TypedPattern::Identifier { name, .. } = &fields[0].node {
                        // Get the type of the variant's inner value
                        let variant = union_type.variants.iter()
                            .find(|v| v.discriminant == variant_index as u64)
                            .ok_or_else(|| crate::CompilerError::Analysis(
                                format!("Variant with index {} not found", variant_index)
                            ))?;

                        // Generate ExtractUnionValue instruction
                        let extracted_id = HirId::new();
                        self.add_instruction(
                            block_id,
                            HirInstruction::ExtractUnionValue {
                                result: extracted_id,
                                union_val: scrutinee_val,
                                variant_index,
                                ty: variant.ty.clone(),
                            },
                        );

                        // Bind the extracted value to the variable
                        self.write_variable(*name, block_id, extracted_id);

                        log::debug!("[SSA] Extracted union value for pattern variable {:?}: val={:?}",
                                 name, extracted_id);
                    }
                }
            }
            _ => {
                // Other patterns don't need extraction (for now)
            }
        }

        Ok(())
    }

    /// Translate enum constructors (Some, None, Ok, Err) to CreateUnion instructions
    pub fn translate_enum_constructor(
        &mut self,
        block_id: HirId,
        constructor_name: &str,
        args: &[TypedNode<TypedExpression>],
        result_ty: &Type,
    ) -> CompilerResult<HirId> {
        use crate::hir::{HirInstruction, HirType, HirUnionType, HirUnionVariant};

        // Determine variant index and extract inner type
        let (variant_index, inner_ty, union_type) = match constructor_name {
            "None" => {
                // None: Optional<T> with discriminant 0, void value
                // Extract T from Optional<T> or infer from context
                let inner_hir_ty = if let Type::Optional(inner) = result_ty {
                    self.convert_type(inner)
                } else {
                    // Fallback: use void if type is unclear
                    HirType::Void
                };

                let mut arena = self.arena.lock().unwrap();
                let none_name = arena.intern_string("None");
                let some_name = arena.intern_string("Some");
                drop(arena);

                let union_ty = HirUnionType {
                    name: None,
                    variants: vec![
                        HirUnionVariant {
                            name: none_name,
                            ty: HirType::Void,
                            discriminant: 0,
                        },
                        HirUnionVariant {
                            name: some_name,
                            ty: inner_hir_ty.clone(),
                            discriminant: 1,
                        },
                    ],
                    discriminant_type: Box::new(HirType::U32),
                    is_c_union: false,
                };

                (0, HirType::Void, union_ty)
            }
            "Some" => {
                // Some(value): Optional<T> with discriminant 1
                if args.len() != 1 {
                    return Err(crate::CompilerError::Analysis(
                        format!("Some() requires exactly 1 argument, got {}", args.len())
                    ));
                }

                let value_ty = self.convert_type(&args[0].ty);

                let mut arena = self.arena.lock().unwrap();
                let none_name = arena.intern_string("None");
                let some_name = arena.intern_string("Some");
                drop(arena);

                let union_ty = HirUnionType {
                    name: None,
                    variants: vec![
                        HirUnionVariant {
                            name: none_name,
                            ty: HirType::Void,
                            discriminant: 0,
                        },
                        HirUnionVariant {
                            name: some_name,
                            ty: value_ty.clone(),
                            discriminant: 1,
                        },
                    ],
                    discriminant_type: Box::new(HirType::U32),
                    is_c_union: false,
                };

                (1, value_ty, union_ty)
            }
            "Ok" => {
                // Ok(value): Result<T,E> with discriminant 0
                if args.len() != 1 {
                    log::debug!("[SSA] ERROR: Ok() called with {} args, result_ty={:?}", args.len(), result_ty);
                    return Err(crate::CompilerError::Analysis(
                        format!("Ok() requires exactly 1 argument, got {}", args.len())
                    ));
                }

                let value_ty = self.convert_type(&args[0].ty);

                // Extract error type from Result<T,E> or default to i32
                let error_ty = if let Type::Result { err_type, .. } = result_ty {
                    self.convert_type(err_type)
                } else {
                    HirType::I32
                };

                let mut arena = self.arena.lock().unwrap();
                let ok_name = arena.intern_string("Ok");
                let err_name = arena.intern_string("Err");
                drop(arena);

                let union_ty = HirUnionType {
                    name: None,
                    variants: vec![
                        HirUnionVariant {
                            name: ok_name,
                            ty: value_ty.clone(),
                            discriminant: 0,
                        },
                        HirUnionVariant {
                            name: err_name,
                            ty: error_ty,
                            discriminant: 1,
                        },
                    ],
                    discriminant_type: Box::new(HirType::U32),
                    is_c_union: false,
                };

                (0, value_ty, union_ty)
            }
            "Err" => {
                // Err(error): Result<T,E> with discriminant 1
                if args.len() != 1 {
                    return Err(crate::CompilerError::Analysis(
                        format!("Err() requires exactly 1 argument, got {}", args.len())
                    ));
                }

                let error_ty = self.convert_type(&args[0].ty);

                // Extract value type from Result<T,E> or default to i32
                let value_ty = if let Type::Result { ok_type, .. } = result_ty {
                    self.convert_type(ok_type)
                } else {
                    HirType::I32
                };

                let mut arena = self.arena.lock().unwrap();
                let ok_name = arena.intern_string("Ok");
                let err_name = arena.intern_string("Err");
                drop(arena);

                let union_ty = HirUnionType {
                    name: None,
                    variants: vec![
                        HirUnionVariant {
                            name: ok_name,
                            ty: value_ty,
                            discriminant: 0,
                        },
                        HirUnionVariant {
                            name: err_name,
                            ty: error_ty.clone(),
                            discriminant: 1,
                        },
                    ],
                    discriminant_type: Box::new(HirType::U32),
                    is_c_union: false,
                };

                (1, error_ty, union_ty)
            }
            _ => {
                return Err(crate::CompilerError::Analysis(
                    format!("Unknown enum constructor: {}", constructor_name)
                ));
            }
        };

        // Translate the value argument if present
        let value_id = if args.is_empty() {
            // For None: create a void/unit value
            self.create_undef(HirType::Void)
        } else {
            // For Some/Ok/Err: translate the argument
            self.translate_expression(block_id, &args[0])?
        };

        // Create the union value - must register in values map with Ptr type
        // Union values are represented as pointers to stack-allocated memory
        let union_hir_ty = HirType::Union(Box::new(union_type.clone()));
        let result_id = self.create_value(
            union_hir_ty.clone(),
            HirValueKind::Instruction
        );

        self.add_instruction(
            block_id,
            HirInstruction::CreateUnion {
                result: result_id,
                union_ty: union_hir_ty,
                variant_index,
                value: value_id,
            },
        );

        // Track use
        if !args.is_empty() {
            self.add_use(value_id, result_id);
        }

        log::debug!("[SSA] Created union for {}: result={:?}, variant_index={}, value={:?}",
                 constructor_name, result_id, variant_index, value_id);

        Ok(result_id)
    }

    /// Process a basic block
    fn process_block(&mut self, block_id: HirId, cfg: &ControlFlowGraph) -> CompilerResult<()> {
        // Find the CFG block
        let cfg_node = cfg.block_map.get(&block_id)
            .and_then(|&node| cfg.graph.node_weight(node))
            .ok_or_else(|| crate::CompilerError::Analysis("Block not found in CFG".into()))?;
        
        // Process each statement
        // Track current block - may change if try expressions split the block
        let mut current_block = block_id;
        for (i, stmt) in cfg_node.statements.iter().enumerate() {
            log::debug!("[SSA] process_block: stmt {} with current_block={:?}", i, current_block);
            current_block = self.process_statement(current_block, stmt)?;
            log::debug!("[SSA] process_block: after stmt {}, current_block={:?}", i, current_block);
        }

        // Process terminator on the final current block
        if let Some(term) = &cfg_node.terminator {
            self.process_terminator(current_block, term)?;
        }
        
        self.filled_blocks.insert(block_id);
        Ok(())
    }
    
    /// Process a statement
    /// Returns the current block ID (may be different if a try expression created a continuation block)
    fn process_statement(
        &mut self,
        block_id: HirId,
        stmt: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedStatement>
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::typed_ast::TypedStatement;

        log::debug!("[SSA] process_statement: block={:?}, stmt_variant={:?}",
                 block_id, std::mem::discriminant(&stmt.node));

        // NOTE: Don't clear continuation_block here - it should persist across statements
        // and be consumed by the terminator. Only try expressions set it.

        match &stmt.node {
            TypedStatement::Let(let_stmt) => {
                // Evaluate the initializer if present
                if let Some(value) = &let_stmt.initializer {
                    let value_id = self.translate_expression(block_id, value)?;

                    // Check if a try expression set a continuation block
                    // If so, subsequent writes should go to that block
                    let write_block = self.continuation_block.unwrap_or(block_id);

                    // Record variable type (both HIR and TypedAST versions)
                    let hir_type = self.convert_type(&let_stmt.ty);
                    self.var_types.insert(let_stmt.name, hir_type.clone());

                    // For TypedAST type, use initializer's type if variable type is Any
                    // This works around the issue where type inference doesn't update the AST
                    let typed_ast_type = if matches!(let_stmt.ty, Type::Any) {
                        value.ty.clone()
                    } else {
                        let_stmt.ty.clone()
                    };
                    self.var_typed_ast_types.insert(let_stmt.name, typed_ast_type);

                    // Check if this variable has its address taken
                    if self.address_taken_vars.contains(&let_stmt.name) {
                        // Allocate stack slot and store value
                        let stack_slot = self.create_value(
                            HirType::Ptr(Box::new(hir_type.clone())),
                            HirValueKind::Instruction
                        );

                        self.add_instruction(write_block, HirInstruction::Alloca {
                            result: stack_slot,
                            ty: hir_type.clone(),
                            count: None,
                            align: 8,
                        });

                        // Store initial value
                        self.add_instruction(write_block, HirInstruction::Store {
                            value: value_id,
                            ptr: stack_slot,
                            align: 8,
                            volatile: false,
                        });

                        // Track stack slot for this variable
                        self.stack_slots.insert(let_stmt.name, stack_slot);
                        log::debug!("[SSA] Allocated stack slot {:?} for address-taken var {:?}", stack_slot, let_stmt.name);
                    } else {
                        // Normal SSA: Create assignment in the continuation block if set
                        self.write_variable(let_stmt.name, write_block, value_id);
                    }
                }
            }
            
            TypedStatement::Expression(expr) => {
                // Evaluate expression for side effects
                self.translate_expression(block_id, expr)?;
            }

            TypedStatement::Match(match_stmt) => {
                // Handle match statement: evaluate scrutinee
                log::debug!("[SSA] Match scrutinee expression: {:?}", match_stmt.scrutinee.node);
                let scrutinee_val = self.translate_expression(block_id, &match_stmt.scrutinee)?;

                // Check if scrutinee is a union type (Optional, Result, or custom union)
                let scrutinee_hir_type = self.convert_type(&match_stmt.scrutinee.ty);

                if let HirType::Union(union_type) = &scrutinee_hir_type {
                    // Extract discriminant for pattern matching
                    let discriminant_id = HirId::new();

                    self.add_instruction(
                        block_id,
                        HirInstruction::GetUnionDiscriminant {
                            result: discriminant_id,
                            union_val: scrutinee_val,
                        },
                    );

                    // Store match context for use in pattern check blocks
                    self.match_context = Some(MatchContext {
                        scrutinee_value: scrutinee_val,
                        discriminant_value: Some(discriminant_id),
                        union_type: Some(union_type.clone()),
                    });

                    log::debug!("[SSA] Match on union type: scrutinee={:?}, discriminant={:?}",
                             scrutinee_val, discriminant_id);
                } else {
                    // Non-union match (e.g., literals, structs)
                    self.match_context = Some(MatchContext {
                        scrutinee_value: scrutinee_val,
                        discriminant_value: None,
                        union_type: None,
                    });
                    log::debug!("[SSA] Match on non-union type: {:?}", scrutinee_hir_type);
                }

                // Pattern binding variables will be handled when the pattern arm body is processed
                // The CFG creates separate blocks for each pattern check and arm body
            }

            // Note: Control flow statements (While, If, etc.) are now handled at the
            // TypedCFG level by TypedCfgBuilder.split_at_control_flow()
            // This is the solution to Gap 2 - multi-block CFG construction

            // Note: TypedStatement doesn't have Assign variant - assignments are expressions

            _ => {
                // Other statements handled by terminator processing
            }
        }

        // Return the continuation block if set (try expression), otherwise the original block
        let result_block = self.continuation_block.unwrap_or(block_id);
        if self.continuation_block.is_some() {
            log::debug!("[SSA] process_statement: returning continuation_block {:?} instead of {:?}",
                     result_block, block_id);
        }
        Ok(result_block)
    }

    /// Process a terminator
    fn process_terminator(
        &mut self,
        block_id: HirId,
        term: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedStatement>
    ) -> CompilerResult<()> {
        use zyntax_typed_ast::typed_ast::TypedStatement;
        
        match &term.node {
            TypedStatement::Return(expr) => {
                let values = if let Some(expr) = expr {
                    vec![self.translate_expression(block_id, expr)?]
                } else {
                    vec![]
                };

                // If a control flow expression (if/match) set a continuation block,
                // place the Return terminator there instead of on the entry block
                let target_block = self.continuation_block.take().unwrap_or(block_id);

                let block = self.function.blocks.get_mut(&target_block).unwrap();
                block.terminator = HirTerminator::Return { values };
            }
            
            TypedStatement::If(if_stmt) => {
                let condition = &if_stmt.condition;
                let cond_value = self.translate_expression(block_id, condition)?;

                // The CFG builder already created the branches
                let block = self.function.blocks.get_mut(&block_id).unwrap();
                if block.successors.len() == 2 {
                    block.terminator = HirTerminator::CondBranch {
                        condition: cond_value,
                        true_target: block.successors[0],
                        false_target: block.successors[1],
                    };
                }
            }
            
            _ => {
                // Default to branch to first successor
                let block = self.function.blocks.get_mut(&block_id).unwrap();
                if let Some(&target) = block.successors.first() {
                    block.terminator = HirTerminator::Branch { target };
                }
            }
        }

        Ok(())
    }

    /// Translate expression to SSA value
    fn translate_expression(
        &mut self,
        block_id: HirId,
        expr: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::typed_ast::TypedExpression;
        use crate::hir::{BinaryOp, UnaryOp};

        match &expr.node {
            TypedExpression::Variable(name) => {
                // Check if this is an address-taken variable - need to load from stack
                if let Some(&stack_slot) = self.stack_slots.get(name) {
                    let var_ty = self.var_types.get(name).cloned().unwrap_or(HirType::I64);
                    let result = self.create_value(var_ty.clone(), HirValueKind::Instruction);
                    self.add_instruction(block_id, HirInstruction::Load {
                        result,
                        ty: var_ty,
                        ptr: stack_slot,
                        align: 8,
                        volatile: false,
                    });
                    self.add_use(stack_slot, result);
                    log::debug!("[SSA] Load address-taken var {:?} from stack slot {:?} -> {:?}", name, stack_slot, result);
                    return Ok(result);
                }

                // First try to read the variable - if it exists in scope, use it
                // Only treat as enum constructor if it's not a defined variable
                if let Some(value_id) = self.try_read_variable(*name, block_id) {
                    log::trace!("[SSA] Variable {:?} found in scope -> {:?}", name, value_id);
                    return Ok(value_id);
                }

                log::trace!("[SSA] Variable {:?} NOT found by try_read_variable!", name);

                // Variable not in scope - check for enum constructors
                // Use resolve_global() since the string may have been interned by a different arena (e.g., ZigBuilder)
                let name_str = name.resolve_global().unwrap_or_else(|| {
                    // Fallback to local arena if global resolution fails
                    let arena = self.arena.lock().unwrap();
                    arena.resolve_string(*name).map(|s| s.to_string()).unwrap_or_default()
                });

                log::debug!("[SSA] Variable not found: {:?} resolves to '{}', type: {:?}", name, name_str, expr.ty);

                // Check for known enum constructors
                match name_str.as_str() {
                    "None" => {
                        log::debug!("[SSA] Recognized as enum constructor: {}", name_str);
                        return self.translate_enum_constructor(block_id, &name_str, &[], &expr.ty);
                    }
                    "Some" | "Ok" | "Err" => {
                        // These constructors require args - shouldn't appear as bare variables
                        // They should be Call expressions, not Variable expressions
                        // This likely means the Zig builder is producing the wrong AST
                        log::debug!("[SSA] ERROR: {} found as bare variable without args, type={:?}", name_str, expr.ty);
                        // Read as undefined variable (will likely fail later)
                        Ok(self.read_variable(*name, block_id))
                    }
                    _ => {
                        // If not an enum constructor, reading as undefined variable
                        log::debug!("[SSA] Not an enum constructor, reading as undefined variable");
                        Ok(self.read_variable(*name, block_id))
                    }
                }
            }
            
            TypedExpression::Literal(lit) => {
                use zyntax_typed_ast::typed_ast::TypedLiteral;

                // String literals need to be stored as globals
                if let TypedLiteral::String(s) = lit {
                    Ok(self.create_string_global(*s))
                } else {
                    let ty = self.convert_type(&expr.ty);
                    let constant = self.translate_literal_with_type(lit, &ty);
                    Ok(self.create_value(ty, HirValueKind::Constant(constant)))
                }
            }
            
            TypedExpression::Binary(binary) => {
                use zyntax_typed_ast::typed_ast::BinaryOp as FrontendOp;

                let op = &binary.op;
                let left = &binary.left;
                let right = &binary.right;

                // Special handling for assignment
                if matches!(op, FrontendOp::Assign) {
                    return self.translate_assignment(block_id, left, right);
                }

                // Logical AND/OR operators - use bitwise operations for now
                // TODO: Implement proper short-circuit evaluation
                // For now, both sides are evaluated (no short-circuiting)
                if matches!(op, FrontendOp::And | FrontendOp::Or) {
                    let left_val = self.translate_expression(block_id, left)?;
                    let right_val = self.translate_expression(block_id, right)?;
                    let result_type = self.convert_type(&expr.ty);

                    let hir_op = match op {
                        FrontendOp::And => crate::hir::BinaryOp::And,  // Bitwise AND for now
                        FrontendOp::Or => crate::hir::BinaryOp::Or,    // Bitwise OR for now
                        _ => unreachable!(),
                    };

                    let result = self.create_value(result_type.clone(), HirValueKind::Instruction);
                    let inst = HirInstruction::Binary {
                        op: hir_op,
                        result,
                        ty: result_type,
                        left: left_val,
                        right: right_val,
                    };

                    self.add_instruction(block_id, inst);
                    self.add_use(left_val, result);
                    self.add_use(right_val, result);

                    return Ok(result);
                }

                // Check if this is a non-primitive type that might use operator overloading
                // For opaque/named types, try to dispatch to trait method
                // First, resolve the actual type from the variable if the expression is a Variable
                let left_actual_ty = self.resolve_actual_type(&left.node, &left.ty);
                let right_actual_ty = self.resolve_actual_type(&right.node, &right.ty);
                eprintln!("[DEBUG SSA] Binary op {:?}, left.ty={:?} (actual: {:?}), right.ty={:?} (actual: {:?})",
                    op, left.ty, left_actual_ty, right.ty, right_actual_ty);

                // Create a modified left/right with resolved types for trait dispatch
                let mut left_with_type = left.clone();
                let mut right_with_type = right.clone();
                left_with_type.ty = left_actual_ty;
                right_with_type.ty = right_actual_ty;

                if let Some(trait_call) = self.try_operator_trait_dispatch(block_id, op, &left_with_type, &right_with_type, &expr.ty)? {
                    eprintln!("[DEBUG SSA] Using trait dispatch for binary op");
                    return Ok(trait_call);
                }
                eprintln!("[DEBUG SSA] No trait dispatch, using native binary op");

                // Regular binary operations for primitive types
                let left_val = self.translate_expression(block_id, left)?;
                let right_val = self.translate_expression(block_id, right)?;
                let result_type = self.convert_type(&expr.ty);

                let hir_op = self.convert_binary_op(op);

                // For comparisons, use the operand type (not Bool result type) for the instruction
                let inst_type = match hir_op {
                    crate::hir::BinaryOp::Lt | crate::hir::BinaryOp::Le |
                    crate::hir::BinaryOp::Gt | crate::hir::BinaryOp::Ge |
                    crate::hir::BinaryOp::Eq | crate::hir::BinaryOp::Ne => {
                        // Use the left operand type so cranelift can determine signed/unsigned
                        self.convert_type(&left.ty)
                    }
                    _ => result_type.clone()
                };

                let result = self.create_value(result_type.clone(), HirValueKind::Instruction);

                let inst = HirInstruction::Binary {
                    op: hir_op,
                    result,
                    ty: inst_type,
                    left: left_val,
                    right: right_val,
                };

                self.add_instruction(block_id, inst);
                self.add_use(left_val, result);
                self.add_use(right_val, result);

                Ok(result)
            }
            
            TypedExpression::Unary(unary) => {
                let op = &unary.op;
                let operand = &unary.operand;
                let operand_val = self.translate_expression(block_id, operand)?;
                let result_type = self.convert_type(&expr.ty);
                
                let hir_op = self.convert_unary_op(op);
                let result = self.create_value(result_type.clone(), HirValueKind::Instruction);
                
                let inst = HirInstruction::Unary {
                    op: hir_op,
                    result,
                    ty: result_type,
                    operand: operand_val,
                };
                
                self.add_instruction(block_id, inst);
                self.add_use(operand_val, result);
                
                Ok(result)
            }
            
            TypedExpression::Call(call) => {
                let callee = &call.callee;
                let args = &call.positional_args;

                // Check if callee is a function name (direct call) vs expression (indirect call)
                // Path expressions should be resolved to Variable during lowering/type resolution
                let (hir_callable, indirect_callee_val) = if let TypedExpression::Variable(func_name) = &callee.node {
                    // Resolve the function name string using global interner
                    // (InternedString.resolve_global() returns the actual string value)
                    let name_str = func_name.resolve_global().unwrap_or_else(|| {
                        // Fallback to local arena if global resolution fails
                        let arena = self.arena.lock().unwrap();
                        arena.resolve_string(*func_name).map(|s| s.to_string()).unwrap_or_default()
                    });

                    // Check for enum constructors (Some, Ok, Err)
                    if name_str == "Some" || name_str == "Ok" || name_str == "Err" {
                        return self.translate_enum_constructor(block_id, &name_str, args, &expr.ty);
                    }

                    // Check if this is an external runtime symbol (e.g., "$haxe$trace$int")
                    // External symbols start with '$' and are resolved at link time
                    if name_str.starts_with('$') {
                        (crate::hir::HirCallable::Symbol(name_str), None)
                    } else if let Some(&func_id) = self.function_symbols.get(func_name) {
                        // Direct function call to known function
                        (crate::hir::HirCallable::Function(func_id), None)
                    } else if let Some(link_name) = self.extern_link_names.get(func_name) {
                        // External function with link_name (e.g., tensor_add -> $Tensor$add)
                        log::debug!("[SSA] Resolved extern call '{}' -> '{}'", name_str, link_name);
                        (crate::hir::HirCallable::Symbol(link_name.clone()), None)
                    } else {
                        // Debug: Log what we're looking for and what's available
                        if !self.extern_link_names.is_empty() {
                            let available: Vec<_> = self.extern_link_names.keys()
                                .filter_map(|k| k.resolve_global())
                                .collect();
                            log::debug!("[SSA] Function '{}' not in extern_link_names ({} entries). Sample: {:?}",
                                name_str, self.extern_link_names.len(), &available[..available.len().min(10)]);
                        }
                        // Variable lookup (function pointer)
                        let callee_val = self.translate_expression(block_id, callee)?;
                        (crate::hir::HirCallable::Indirect(callee_val), Some(callee_val))
                    }
                } else {
                    // General expression (e.g., field access, method call result)
                    let callee_val = self.translate_expression(block_id, callee)?;
                    (crate::hir::HirCallable::Indirect(callee_val), Some(callee_val))
                };

                // Translate arguments
                let mut arg_vals = Vec::new();
                for arg in args {
                    let arg_val = self.translate_expression(block_id, arg)?;
                    arg_vals.push(arg_val);
                }

                // Translate type arguments for generic calls
                let type_args: Vec<HirType> = call.type_args.iter()
                    .map(|ty| self.convert_type(ty))
                    .collect();

                // Create call instruction
                let result_type = self.convert_type(&expr.ty);
                let result = if result_type != HirType::Void {
                    Some(self.create_value(result_type.clone(), HirValueKind::Instruction))
                } else {
                    None
                };

                let inst = HirInstruction::Call {
                    result,
                    callee: hir_callable,
                    args: arg_vals.clone(),
                    type_args,  // Preserve type arguments from TypedCall
                    const_args: vec![], // TODO: Preserve const args when TypedCall supports them
                    // NOTE: Tail call detection requires control flow analysis.
                    // Need to verify: (1) call is last instruction, (2) no cleanup needed, (3) compatible calling convention
                    // WORKAROUND: Always false (safe, but misses optimization opportunity)
                    // FUTURE (v2.0): Add tail call analysis pass
                    // Estimated effort: 8-10 hours (requires CFG analysis)
                    is_tail: false,
                };

                self.add_instruction(block_id, inst);

                // Add uses (for indirect calls, track callee value)
                if let Some(callee_val) = indirect_callee_val {
                    self.add_use(callee_val, result.unwrap_or(callee_val));
                }

                // Track argument uses
                let result_or_void = result.unwrap_or_else(|| self.create_undef(HirType::Void));
                for arg in &arg_vals {
                    self.add_use(*arg, result_or_void);
                }

                Ok(result_or_void)
            }
            
            TypedExpression::Field(field_access) => {
                let object = &field_access.object;
                let field = &field_access.field;
                let object_val = self.translate_expression(block_id, object)?;

                // Resolve actual type - if object is a variable, look up its type from var_types
                let object_type = if let TypedExpression::Variable(var_name) = &object.node {
                    // Variable - get actual type from var_types (which was updated during type resolution)
                    if let Some(hir_type) = self.var_types.get(var_name) {
                        eprintln!("[FIELD ACCESS DEBUG] Variable '{}' has hir_type: {:?}",
                            var_name.resolve_global().unwrap_or_default(), hir_type);
                        // Convert HIR type back to TypedAST Type for get_field_index
                        let converted = self.hir_type_to_typed_ast_type(hir_type);
                        eprintln!("[FIELD ACCESS DEBUG] Converted to typed_ast_type: {:?}", converted);
                        converted
                    } else {
                        eprintln!("[FIELD ACCESS DEBUG] Variable '{}' not in var_types, using object.ty: {:?}",
                            var_name.resolve_global().unwrap_or_default(), object.ty);
                        object.ty.clone()
                    }
                } else {
                    eprintln!("[FIELD ACCESS DEBUG] Not a variable, using object.ty: {:?}", object.ty);
                    object.ty.clone()
                };

                // Special case: accessing 'value' field on an abstract type
                // Abstract types are zero-cost - they ARE their underlying value
                // So accessing the 'value' field just returns the value itself
                if let Type::Named { id, .. } = &object_type {
                    if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                        if matches!(&type_def.kind, zyntax_typed_ast::type_registry::TypeKind::Abstract { .. }) {
                            let field_name_str = {
                                let arena = self.arena.lock().unwrap();
                                arena.resolve_string(*field).map(|s| s.to_string()).unwrap_or_default()
                            };

                            if field_name_str == "value" {
                                // For abstract types, accessing 'value' just returns the object itself
                                // since abstract types ARE their underlying value (zero-cost abstraction)
                                eprintln!("[FIELD ACCESS] Abstract type field access optimization: returning object directly");
                                return Ok(object_val);
                            }
                        }
                    }
                }

                // Calculate field index
                let field_index = self.get_field_index(&object_type, field)?;
                let result_type = self.convert_type(&expr.ty);
                let result = self.create_value(result_type.clone(), HirValueKind::Instruction);

                let inst = HirInstruction::ExtractValue {
                    result,
                    ty: result_type,
                    aggregate: object_val,
                    indices: vec![field_index],
                };

                self.add_instruction(block_id, inst);
                self.add_use(object_val, result);

                Ok(result)
            }
            
            TypedExpression::Index(index_expr) => {
                let object = &index_expr.object;
                let index = &index_expr.index;
                let object_val = self.translate_expression(block_id, object)?;
                let index_val = self.translate_expression(block_id, index)?;
                
                // Create GEP instruction
                let result_type = HirType::Ptr(Box::new(self.convert_type(&expr.ty)));
                let gep_result = self.create_value(result_type.clone(), HirValueKind::Instruction);
                
                let gep_inst = HirInstruction::GetElementPtr {
                    result: gep_result,
                    ty: result_type,
                    ptr: object_val,
                    indices: vec![index_val],
                };
                
                self.add_instruction(block_id, gep_inst);
                self.add_use(object_val, gep_result);
                self.add_use(index_val, gep_result);
                
                // Load the value
                let load_result = self.create_value(
                    self.convert_type(&expr.ty), 
                    HirValueKind::Instruction
                );
                
                let load_inst = HirInstruction::Load {
                    result: load_result,
                    ty: self.convert_type(&expr.ty),
                    ptr: gep_result,
                    // NOTE: Proper alignment calculation requires type layout information.
                    // Need TypeRegistry or TargetData to compute alignment from type.
                    // WORKAROUND: Fixed 4-byte alignment (works for most common types: i32, f32, pointers)
                    // FUTURE (v2.0): Calculate from type using target data layout
                    // Estimated effort: 3-4 hours (add alignment calculation utility)
                    align: 4,
                    volatile: false,
                };
                
                self.add_instruction(block_id, load_inst);
                self.add_use(gep_result, load_result);
                
                Ok(load_result)
            }
            
            TypedExpression::If(if_expr) => {
                // Evaluate if expression with branches
                let condition = &if_expr.condition;
                let then_branch = &if_expr.then_branch;
                let else_branch = &if_expr.else_branch;

                let cond_val = self.translate_expression(block_id, condition)?;

                // Create blocks for then/else/merge
                let then_block_id = HirId::new();
                let else_block_id = HirId::new();
                let merge_block_id = HirId::new();

                // Create blocks in function
                self.function.blocks.insert(then_block_id, HirBlock::new(then_block_id));
                self.function.blocks.insert(else_block_id, HirBlock::new(else_block_id));
                self.function.blocks.insert(merge_block_id, HirBlock::new(merge_block_id));

                // Initialize definitions for new blocks
                self.definitions.insert(then_block_id, IndexMap::new());
                self.definitions.insert(else_block_id, IndexMap::new());
                self.definitions.insert(merge_block_id, IndexMap::new());

                // Set conditional branch terminator for current block
                self.function.blocks.get_mut(&block_id).unwrap().terminator = HirTerminator::CondBranch {
                    condition: cond_val,
                    true_target: then_block_id,
                    false_target: else_block_id,
                };

                // Update predecessors/successors
                self.function.blocks.get_mut(&block_id).unwrap().successors = vec![then_block_id, else_block_id];
                self.function.blocks.get_mut(&then_block_id).unwrap().predecessors.push(block_id);
                self.function.blocks.get_mut(&else_block_id).unwrap().predecessors.push(block_id);

                // Translate then branch
                let then_val = self.translate_expression(then_block_id, then_branch)?;
                self.function.blocks.get_mut(&then_block_id).unwrap().terminator = HirTerminator::Branch { target: merge_block_id };
                self.function.blocks.get_mut(&then_block_id).unwrap().successors = vec![merge_block_id];
                self.function.blocks.get_mut(&merge_block_id).unwrap().predecessors.push(then_block_id);

                // Translate else branch
                let else_val = self.translate_expression(else_block_id, else_branch)?;
                self.function.blocks.get_mut(&else_block_id).unwrap().terminator = HirTerminator::Branch { target: merge_block_id };
                self.function.blocks.get_mut(&else_block_id).unwrap().successors = vec![merge_block_id];
                self.function.blocks.get_mut(&merge_block_id).unwrap().predecessors.push(else_block_id);

                // Create phi in merge block
                let result_type = self.convert_type(&expr.ty);
                let result = self.create_value(result_type.clone(), HirValueKind::Instruction);

                self.function.blocks.get_mut(&merge_block_id).unwrap().phis.push(HirPhi {
                    result,
                    ty: result_type,
                    incoming: vec![(then_val, then_block_id), (else_val, else_block_id)],
                });

                // Set continuation block so subsequent code executes in merge block
                self.continuation_block = Some(merge_block_id);

                Ok(result)
            }

            TypedExpression::Struct(struct_lit) => {
                // Build struct value directly using InsertValue operations
                let struct_ty = self.convert_type(&expr.ty);

                eprintln!("[SSA STRUCT LIT] Building struct literal with type {:?}, {} fields",
                    struct_ty, struct_lit.fields.len());

                // Start with an undefined struct value
                let mut current_struct = self.create_value(
                    struct_ty.clone(),
                    HirValueKind::Undef
                );

                // Insert each field value into the struct
                for (i, field) in struct_lit.fields.iter().enumerate() {
                    let field_val = self.translate_expression(block_id, &field.value)?;

                    eprintln!("[SSA STRUCT LIT] Inserting field {}", i);

                    // Create new struct value with the field inserted
                    let next_struct = self.create_value(
                        struct_ty.clone(),
                        HirValueKind::Instruction
                    );

                    self.add_instruction(block_id, HirInstruction::InsertValue {
                        result: next_struct,
                        ty: struct_ty.clone(),
                        aggregate: current_struct,
                        value: field_val,
                        indices: vec![i as u32],
                    });

                    self.add_use(current_struct, next_struct);
                    self.add_use(field_val, next_struct);

                    current_struct = next_struct;
                }

                eprintln!("[SSA STRUCT LIT] Returning struct value");

                // Return the final struct value (not a pointer!)
                Ok(current_struct)
            }

            TypedExpression::Array(elements) => {
                // Create ZRTL array format: [i32 capacity][i32 length][elements...]
                // This format allows plugins to safely read array length
                let elem_ty = if let Type::Array { element_type, .. } = &expr.ty {
                    self.convert_type(element_type)
                } else {
                    HirType::I64 // Fallback
                };

                // Calculate element size based on type
                let elem_size = match &elem_ty {
                    HirType::I8 | HirType::U8 | HirType::Bool => 1,
                    HirType::I16 | HirType::U16 => 2,
                    HirType::I32 | HirType::U32 | HirType::F32 => 4,
                    HirType::I64 | HirType::U64 | HirType::F64 | HirType::Ptr(_) => 8,
                    HirType::I128 | HirType::U128 => 16,
                    _ => 8, // Default for complex types
                };
                let num_elements = elements.len();

                // ZRTL array header: 8 bytes (capacity i32 + length i32)
                const ZRTL_HEADER_BYTES: usize = 8;
                let total_size = ZRTL_HEADER_BYTES + num_elements * elem_size;

                // Allocate raw bytes for the entire ZRTL array
                let alloc_ty = HirType::Array(Box::new(HirType::U8), total_size as u64);
                let alloc_result = self.create_value(HirType::Ptr(Box::new(HirType::I32)), HirValueKind::Instruction);

                self.add_instruction(block_id, HirInstruction::Alloca {
                    result: alloc_result,
                    ty: alloc_ty,
                    count: None,
                    align: 8,
                });

                // Store capacity at offset 0 (i32)
                let cap_const = self.create_value(HirType::I32, HirValueKind::Constant(crate::hir::HirConstant::I32(num_elements as i32)));
                self.add_instruction(block_id, HirInstruction::Store {
                    value: cap_const,
                    ptr: alloc_result,
                    align: 4,
                    volatile: false,
                });

                // Store length at offset 4 (i32) - need to calculate ptr + 4
                let len_const = self.create_value(HirType::I32, HirValueKind::Constant(crate::hir::HirConstant::I32(num_elements as i32)));
                let offset_4 = self.create_value(HirType::I64, HirValueKind::Constant(crate::hir::HirConstant::I64(4)));
                let len_ptr = self.create_value(HirType::Ptr(Box::new(HirType::I32)), HirValueKind::Instruction);
                self.add_instruction(block_id, HirInstruction::GetElementPtr {
                    result: len_ptr,
                    ty: HirType::U8, // Base type for byte offset calculation
                    ptr: alloc_result,
                    indices: vec![offset_4],
                });
                self.add_instruction(block_id, HirInstruction::Store {
                    value: len_const,
                    ptr: len_ptr,
                    align: 4,
                    volatile: false,
                });

                // Store each element starting at offset 8
                for (i, elem_expr) in elements.iter().enumerate() {
                    let elem_val = self.translate_expression(block_id, elem_expr)?;

                    let offset = ZRTL_HEADER_BYTES + i * elem_size;
                    let offset_const = self.create_value(HirType::I64, HirValueKind::Constant(crate::hir::HirConstant::I64(offset as i64)));
                    let elem_ptr = self.create_value(HirType::Ptr(Box::new(elem_ty.clone())), HirValueKind::Instruction);
                    self.add_instruction(block_id, HirInstruction::GetElementPtr {
                        result: elem_ptr,
                        ty: HirType::U8, // Base type for byte offset calculation
                        ptr: alloc_result,
                        indices: vec![offset_const],
                    });

                    self.add_instruction(block_id, HirInstruction::Store {
                        value: elem_val,
                        ptr: elem_ptr,
                        align: elem_size.min(8) as u32,
                        volatile: false,
                    });
                }

                // Return pointer to the ZRTL array header
                Ok(alloc_result)
            }

            TypedExpression::Tuple(elements) => {
                // Convert tuple to struct
                let field_types: Vec<_> = elements.iter()
                    .map(|e| self.convert_type(&e.ty))
                    .collect();

                let tuple_ty = HirType::Struct(crate::hir::HirStructType {
                    name: None,
                    fields: field_types,
                    packed: false,
                });

                let alloc_result = self.create_value(tuple_ty.clone(), HirValueKind::Instruction);

                self.add_instruction(block_id, HirInstruction::Alloca {
                    result: alloc_result,
                    ty: tuple_ty.clone(),
                    count: None,
                    align: 8,
                });

                // Initialize each element
                for (i, elem_expr) in elements.iter().enumerate() {
                    let elem_val = self.translate_expression(block_id, elem_expr)?;

                    let insert_result = self.create_value(tuple_ty.clone(), HirValueKind::Instruction);
                    self.add_instruction(block_id, HirInstruction::InsertValue {
                        result: insert_result,
                        ty: tuple_ty.clone(),
                        aggregate: alloc_result,
                        value: elem_val,
                        indices: vec![i as u32],
                    });
                }

                Ok(alloc_result)
            }

            TypedExpression::Cast(cast) => {
                let operand_val = self.translate_expression(block_id, &cast.expr)?;
                let target_ty = self.convert_type(&cast.target_type);
                let result = self.create_value(target_ty.clone(), HirValueKind::Instruction);

                // NOTE: Proper cast operation selection requires type information.
                // Should determine: IntToInt, IntToFloat, FloatToInt, FloatToFloat, Bitcast, etc.
                // Needs source type width, target type width, signedness information.
                //
                // WORKAROUND: Always uses Bitcast (works for pointer casts and same-size conversions)
                // FUTURE (v2.0): Add type-aware cast selection logic
                // Estimated effort: 4-5 hours (needs type width/signedness utilities)
                let cast_op = CastOp::Bitcast;

                self.add_instruction(block_id, HirInstruction::Cast {
                    op: cast_op,
                    result,
                    ty: target_ty,
                    operand: operand_val,
                });

                self.add_use(operand_val, result);
                Ok(result)
            }

            TypedExpression::Reference(reference) => {
                // Take reference of expression - return stack slot address for address-taken variables
                use zyntax_typed_ast::typed_ast::TypedExpression as TE;

                // Check if the inner expression is a variable with a stack slot
                if let TE::Variable(var_name) = &reference.expr.node {
                    if let Some(&stack_slot) = self.stack_slots.get(var_name) {
                        log::debug!("[SSA] Reference to address-taken var {:?} -> stack slot {:?}", var_name, stack_slot);
                        return Ok(stack_slot);
                    }
                }

                // For non-stack-allocated expressions, this is an error in the current model
                // as SSA values don't have addresses
                log::debug!("[SSA] Warning: Taking reference of non-stack-allocated expression");
                let operand_val = self.translate_expression(block_id, &reference.expr)?;
                Ok(operand_val)
            }

            TypedExpression::Dereference(expr) => {
                // Dereference pointer
                let ptr_val = self.translate_expression(block_id, expr)?;
                let result_ty = self.convert_type(&expr.ty);
                let result = self.create_value(result_ty.clone(), HirValueKind::Instruction);

                self.add_instruction(block_id, HirInstruction::Load {
                    result,
                    ty: result_ty,
                    ptr: ptr_val,
                    align: 8,
                    volatile: false,
                });

                self.add_use(ptr_val, result);
                Ok(result)
            }

            TypedExpression::MethodCall(method_call) => {
                // Resolve receiver type - if receiver is a variable, look up its actual type
                // This works around the issue where type inference doesn't update variable references in the AST
                let receiver_type = if let TypedExpression::Variable(var_name) = &method_call.receiver.node {
                    eprintln!("[METHOD_CALL] Receiver is variable '{}'", var_name.resolve_global().unwrap_or_default());
                    // Look up the variable's actual type from when it was declared
                    if let Some(var_ty) = self.var_typed_ast_types.get(var_name) {
                        eprintln!("[METHOD_CALL] Found variable type: {:?}", var_ty);
                        var_ty.clone()
                    } else {
                        eprintln!("[METHOD_CALL] Variable type not found, using receiver type: {:?}", method_call.receiver.ty);
                        method_call.receiver.ty.clone()
                    }
                } else {
                    // Non-variable expression - use the type directly
                    eprintln!("[METHOD_CALL] Receiver is expression with type: {:?}", method_call.receiver.ty);
                    method_call.receiver.ty.clone()
                };

                // Resolve method to mangled function name
                let mangled_name = self.resolve_method_to_function(
                    &receiver_type,
                    method_call.method,
                )?;

                // Translate receiver and arguments
                let receiver_val = self.translate_expression(block_id, &method_call.receiver)?;

                let mut arg_vals = vec![receiver_val];
                for arg in &method_call.positional_args {
                    let arg_val = self.translate_expression(block_id, arg)?;
                    arg_vals.push(arg_val);
                }

                // Look up the function by mangled name - first in function_symbols, then extern_link_names
                let hir_callable = if let Some(&func_id) = self.function_symbols.get(&mangled_name) {
                    // Direct function call to known function
                    crate::hir::HirCallable::Function(func_id)
                } else if let Some(link_name) = self.extern_link_names.get(&mangled_name) {
                    // External function with link_name (e.g., Tensor$sum -> $Tensor$sum)
                    crate::hir::HirCallable::Symbol(link_name.clone())
                } else {
                    let name_str = mangled_name.resolve_global().unwrap_or_default();
                    return Err(crate::CompilerError::Analysis(
                        format!("Method function '{}' not found in symbol table", name_str)
                    ));
                };

                // Determine result type based on the mangled function name
                // For extern methods like $Tensor$sum_f32, parse the return type from the suffix
                let result_type = if matches!(expr.ty, Type::Any | Type::Unknown) {
                    let mangled_str = mangled_name.resolve_global().unwrap_or_default();

                    // Check common return type suffixes for Tensor methods
                    // Methods like sum_f32, mean_f32 return f32
                    // Methods like zeros, ones, arange return Tensor (opaque ptr)
                    let hir_type = if mangled_str.ends_with("_f32") || mangled_str.contains("$sum") || mangled_str.contains("$mean")
                        || mangled_str.contains("$max") || mangled_str.contains("$min") || mangled_str.contains("$std") || mangled_str.contains("$var") {
                        // Reduction methods that return f32
                        log::debug!("[METHOD_CALL] Inferred F32 return type for '{}'", mangled_str);
                        HirType::F32
                    } else if mangled_str.contains("$ndim") || mangled_str.contains("$numel") {
                        // Methods that return i64
                        log::debug!("[METHOD_CALL] Inferred I64 return type for '{}'", mangled_str);
                        HirType::I64
                    } else if let Type::Named { id, .. } = &receiver_type {
                        // Fall back to looking up in trait implementations for named types
                        let receiver_type_id = *id;
                        let mut method_return_type = None;
                        for (_trait_id, impls) in self.type_registry.iter_implementations() {
                            for impl_def in impls {
                                if let Type::Named { id: impl_type_id, .. } = &impl_def.for_type {
                                    if *impl_type_id == receiver_type_id {
                                        for method in &impl_def.methods {
                                            if method.signature.name == method_call.method {
                                                method_return_type = Some(method.signature.return_type.clone());
                                                break;
                                            }
                                        }
                                    }
                                }
                                if method_return_type.is_some() {
                                    break;
                                }
                            }
                            if method_return_type.is_some() {
                                break;
                            }
                        }
                        let typed_return_type = method_return_type.unwrap_or(Type::Primitive(zyntax_typed_ast::PrimitiveType::I64));
                        self.convert_type(&typed_return_type)
                    } else {
                        // For extern types, assume opaque return (returns same type as receiver)
                        log::debug!("[METHOD_CALL] Assuming opaque return type for extern method '{}'", mangled_str);
                        self.convert_type(&receiver_type)
                    };
                    hir_type
                } else {
                    // Use the annotated type from the expression
                    self.convert_type(&expr.ty)
                };

                let result = if result_type != HirType::Void {
                    Some(self.create_value(result_type.clone(), HirValueKind::Instruction))
                } else {
                    None
                };

                self.add_instruction(block_id, HirInstruction::Call {
                    result,
                    callee: hir_callable,
                    args: arg_vals.clone(),
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                });

                // Track argument uses
                let result_or_void = result.unwrap_or_else(|| self.create_undef(HirType::Void));
                for arg in &arg_vals {
                    self.add_use(*arg, result_or_void);
                }

                Ok(result_or_void)
            }

            TypedExpression::Match(match_expr) => {
                self.translate_match(block_id, match_expr, &expr.ty)
            }

            TypedExpression::Await(async_expr) => {
                // Emit an Intrinsic::Await call for the async state machine transformation.
                // The AsyncCompiler will later detect these await points and split the function
                // into a state machine with states before/after each await.
                //
                // Translation: `await future_expr` becomes:
                //   %future = <translate future_expr>  -- this is a *Promise<T> pointer
                //   %result = call @intrinsic::await(%future)  -- this returns T
                //
                // The async transformation phase will then convert this into proper
                // state machine code that yields and resumes.

                eprintln!("[DEBUG] SSA: Matched TypedExpression::Await!");

                // The result type is what the await expression evaluates to (T, not *Promise<T>)
                let result_ty = self.convert_type(&expr.ty);

                // The future type is *Promise<T> - the async call returns a Promise pointer
                let promise_ty = HirType::Ptr(Box::new(HirType::Opaque(InternedString::new_global("Promise"))));

                // Translate the expression being awaited (the future/async call)
                // We need to ensure the Call result has Promise pointer type, not the final value type
                // First check if it's a Call expression - if so, override its result type
                if let TypedExpression::Call(call) = &async_expr.node {
                    // Translate the call with Promise result type instead of the declared return type
                    let callee = &call.callee;
                    let args = &call.positional_args;

                    // Resolve the callable
                    let (hir_callable, _) = if let TypedExpression::Variable(func_name) = &callee.node {
                        let name_str = func_name.resolve_global().unwrap_or_else(|| {
                            let arena = self.arena.lock().unwrap();
                            arena.resolve_string(*func_name).map(|s| s.to_string()).unwrap_or_default()
                        });

                        if let Some(&func_id) = self.function_symbols.get(func_name) {
                            (crate::hir::HirCallable::Function(func_id), None)
                        } else {
                            let callee_val = self.translate_expression(block_id, callee)?;
                            (crate::hir::HirCallable::Indirect(callee_val), Some(callee_val))
                        }
                    } else {
                        let callee_val = self.translate_expression(block_id, callee)?;
                        (crate::hir::HirCallable::Indirect(callee_val), Some(callee_val))
                    };

                    // Translate arguments
                    let mut arg_vals = Vec::new();
                    for arg in args {
                        let arg_val = self.translate_expression(block_id, arg)?;
                        arg_vals.push(arg_val);
                    }

                    // Create the Call result with Promise pointer type (not the function's return type)
                    let future_val = self.create_value(promise_ty.clone(), HirValueKind::Instruction);

                    let call_inst = HirInstruction::Call {
                        result: Some(future_val),
                        callee: hir_callable,
                        args: arg_vals,
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    };

                    self.add_instruction(block_id, call_inst);
                    eprintln!("[DEBUG] SSA: Created async Call with Promise result type, future_val={:?}", future_val);

                    // Create result value for the await (with actual T type)
                    let result = self.create_value(result_ty.clone(), HirValueKind::Instruction);

                    // Emit the intrinsic await call
                    let await_inst = HirInstruction::Call {
                        callee: crate::hir::HirCallable::Intrinsic(crate::hir::Intrinsic::Await),
                        args: vec![future_val],
                        result: Some(result),
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    };

                    self.add_instruction(block_id, await_inst);
                    Ok(result)
                } else {
                    // For non-call await expressions (e.g., await some_promise_variable)
                    let future_val = self.translate_expression(block_id, async_expr)?;
                    log::trace!("[SSA] Await non-call: future_val = {:?}", future_val);

                    // Create result value for the await
                    let result = self.create_value(result_ty.clone(), HirValueKind::Instruction);

                    // Emit the intrinsic await call
                    let await_inst = HirInstruction::Call {
                        callee: crate::hir::HirCallable::Intrinsic(crate::hir::Intrinsic::Await),
                        args: vec![future_val],
                        result: Some(result),
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    };

                    if let Some(block) = self.function.blocks.get_mut(&block_id) {
                        block.add_instruction(await_inst);
                    }

                    Ok(result)
                }
            }

            TypedExpression::Try(try_expr) => {
                // Gap 8 Phase 2: Rust-style ? operator for Result<T, E>
                //
                // Desugars: let value = operation()?;
                // Into:     let tmp = operation();
                //           let value = match tmp {
                //               Ok(v) => v,
                //               Err(e) => return Err(e),
                //           };
                //
                // This is different from exception-based try-catch.
                //
                // LIMITATION: The ? operator creates multiple basic blocks (ok_block, err_block, continue_block).
                // This works correctly, but subsequent instructions should ideally go into continue_block.
                // For now, the CFG is correct but the block continuation might need refinement for complex expressions.
                // This is fine for common cases like: let x = foo()?;
                self.translate_try_operator(block_id, try_expr, &expr.ty)
            }

            TypedExpression::Range(range) => {
                // Range expressions create range objects
                // This would need runtime support
                // For now, create a struct with start/end
                let start_val = if let Some(start) = &range.start {
                    self.translate_expression(block_id, start)?
                } else {
                    self.create_undef(HirType::I64)
                };
                let end_val = if let Some(end) = &range.end {
                    self.translate_expression(block_id, end)?
                } else {
                    self.create_undef(HirType::I64)
                };

                // Create a simple struct with two fields
                let range_ty = HirType::Struct(crate::hir::HirStructType {
                    name: None,
                    fields: vec![HirType::I64, HirType::I64],
                    packed: false,
                });

                let alloc_result = self.create_value(range_ty.clone(), HirValueKind::Instruction);
                self.add_instruction(block_id, HirInstruction::Alloca {
                    result: alloc_result,
                    ty: range_ty.clone(),
                    count: None,
                    align: 8,
                });

                let insert1 = self.create_value(range_ty.clone(), HirValueKind::Instruction);
                self.add_instruction(block_id, HirInstruction::InsertValue {
                    result: insert1,
                    ty: range_ty.clone(),
                    aggregate: alloc_result,
                    value: start_val,
                    indices: vec![0],
                });

                let insert2 = self.create_value(range_ty.clone(), HirValueKind::Instruction);
                self.add_instruction(block_id, HirInstruction::InsertValue {
                    result: insert2,
                    ty: range_ty,
                    aggregate: insert1,
                    value: end_val,
                    indices: vec![1],
                });

                Ok(insert2)
            }

            TypedExpression::Lambda(lambda) => {
                self.translate_closure(block_id, lambda, &expr.ty)
            }

            _ => {
                // Fallback for any remaining unhandled expressions
                Ok(self.create_undef(self.convert_type(&expr.ty)))
            }
        }
    }
    
    /// Write a variable in SSA form
    fn write_variable(&mut self, var: InternedString, block: HirId, value: HirId) {
        log::debug!("[SSA] write_variable({:?}, {:?}, {:?})", var, block, value);
        self.definitions
            .get_mut(&block)
            .unwrap()
            .insert(var, value);

        // Track variable writes for loop phi placement
        self.variable_writes
            .entry(block)
            .or_insert_with(HashSet::new)
            .insert(var);
    }
    
    /// Read a variable in SSA form
    /// Try to read a variable without creating phis - returns None if not found
    fn try_read_variable(&self, var: InternedString, block: HirId) -> Option<HirId> {
        self.definitions.get(&block).and_then(|defs| defs.get(&var)).copied()
    }

    fn read_variable(&mut self, var: InternedString, block: HirId) -> HirId {
        if let Some(&value) = self.definitions.get(&block).and_then(|defs| defs.get(&var)) {
            log::debug!("[SSA] read_variable({:?}, {:?}) = {:?} (found in definitions)", var, block, value);
            return value;
        }

        log::debug!("[SSA] read_variable({:?}, {:?}) - not in definitions, recursing", var, block);
        // Not defined in this block - need phi or recursive lookup
        self.read_variable_recursive(var, block)
    }
    
    /// Recursively read variable, inserting phis as needed (before IDF)
    /// After IDF placement, this won't create new phis - it will only traverse existing ones
    fn read_variable_recursive(&mut self, var: InternedString, block: HirId) -> HirId {
        let (predecessors, is_sealed) = {
            let block_info = self.function.blocks.get(&block).unwrap();
            (block_info.predecessors.clone(), self.sealed_blocks.contains(&block))
        };
        log::trace!("[SSA] read_variable_recursive({:?}, {:?}): sealed={}, predecessors={:?}", var, block, is_sealed, predecessors);

        if !is_sealed {
            // Block not sealed - create incomplete phi (ONLY if IDF not done yet)
            let phi_key = (block, var);

            if let Some(&phi_val) = self.incomplete_phis.get(&phi_key) {
                return phi_val;
            }

            // After IDF placement, don't create new phis
            // Instead, try to read from predecessors to find the value
            if self.idf_placement_done {
                // CRITICAL: Create a temporary placeholder phi to break cycles in loops
                // Without this, while loop CFG (header -> body -> header) causes infinite recursion
                let ty = self.var_types.get(&var).cloned().unwrap_or(HirType::I64);
                let placeholder = self.create_value(ty.clone(), HirValueKind::Instruction);

                // Write the placeholder to break cycles
                self.write_variable(var, block, placeholder);

                // Now try reading from each predecessor
                let mut found_value = None;
                for pred in &predecessors {
                    let val = self.read_variable(var, *pred);
                    // Check if it's not undef and not our placeholder
                    if val != placeholder {
                        if let Some(v) = self.function.values.get(&val) {
                            if !matches!(v.kind, HirValueKind::Undef) {
                                found_value = Some(val);
                                break;
                            }
                        }
                    }
                }

                if let Some(val) = found_value {
                    // Found a real value - update definition to use it
                    self.write_variable(var, block, val);
                    return val;
                }

                // All predecessors returned undef or placeholder - variable is truly undefined
                let undef = self.create_undef(ty);
                self.write_variable(var, block, undef);
                return undef;
            }

            let ty = self.var_types.get(&var).cloned()
                .unwrap_or(HirType::I64); // Default type
            let phi_val = self.create_value(ty.clone(), HirValueKind::Instruction);

            self.incomplete_phis.insert(phi_key, phi_val);
            self.function.blocks.get_mut(&block).unwrap().phis.push(HirPhi {
                result: phi_val,
                ty,
                incoming: vec![],
            });

            phi_val
        } else if predecessors.len() == 1 {
            // Single predecessor - no phi needed
            self.read_variable(var, predecessors[0])
        } else {
            // Multiple predecessors - may need phi
            // After IDF, phis should already exist, so don't create new ones
            if self.idf_placement_done {
                // Check if a phi already exists for this variable in this block
                let phi_key = (block, var);
                if let Some(&phi_val) = self.incomplete_phis.get(&phi_key) {
                    return phi_val;
                }
                // Check if phi is in definitions (already filled)
                if let Some(defs) = self.definitions.get(&block) {
                    if let Some(&val) = defs.get(&var) {
                        return val;
                    }
                }
                // No phi found - variable doesn't need a phi (like read-only parameters)
                // IMPORTANT: To avoid infinite recursion with loops, we need to create a
                // temporary placeholder phi before recursing to predecessors.
                let ty = self.var_types.get(&var).cloned().unwrap_or(HirType::I64);
                let phi_val = self.create_value(ty.clone(), HirValueKind::Instruction);

                // Temporarily write to break cycles
                self.write_variable(var, block, phi_val);

                // Try reading from each predecessor until we find the value
                let mut found_values = Vec::new();
                for pred in &predecessors {
                    let val = self.read_variable(var, *pred);
                    // Check if it's not undef
                    if let Some(v) = self.function.values.get(&val) {
                        if !matches!(v.kind, HirValueKind::Undef) {
                            found_values.push((val, *pred));
                        }
                    }
                }

                // If all values are the same (and not the phi itself), just return that value
                let unique_vals: HashSet<_> = found_values.iter()
                    .map(|(v, _)| v)
                    .filter(|&&v| v != phi_val)
                    .collect();

                if unique_vals.len() == 1 {
                    // All predecessors have same value - no phi needed
                    let single_val = **unique_vals.iter().next().unwrap();
                    self.write_variable(var, block, single_val);
                    return single_val;
                } else if unique_vals.is_empty() {
                    // All predecessors returned undef or just this phi
                    let undef = self.create_undef(ty);
                    self.write_variable(var, block, undef);
                    return undef;
                } else {
                    // Need a real phi node
                    self.function.blocks.get_mut(&block).unwrap().phis.push(HirPhi {
                        result: phi_val,
                        ty,
                        incoming: found_values,
                    });
                    return phi_val;
                }
            }

            let ty = self.var_types.get(&var).cloned()
                .unwrap_or(HirType::I64);
            let phi_val = self.create_value(ty.clone(), HirValueKind::Instruction);

            // Temporarily write to avoid infinite recursion
            self.write_variable(var, block, phi_val);

            // Collect values from predecessors
            let mut incoming = Vec::new();

            log::debug!("[SSA] Filling phi for var {:?} in block {:?} with {} predecessors", var, block, predecessors.len());
            for pred in predecessors {
                let pred_val = self.read_variable(var, pred);
                log::debug!("[SSA]   Phi incoming: ({:?}, {:?})", pred_val, pred);
                incoming.push((pred_val, pred));
            }
            log::debug!("[SSA] Phi filled with {} incoming values", incoming.len());

            // Check if phi is trivial
            let non_phi_vals: HashSet<_> = incoming.iter()
                .map(|(val, _)| val)
                .filter(|&&v| v != phi_val)
                .collect();

            if non_phi_vals.len() == 1 {
                // Trivial phi - replace with the single value
                let single_val = **non_phi_vals.iter().next().unwrap();
                self.write_variable(var, block, single_val);
                single_val
            } else if non_phi_vals.is_empty() {
                // All inputs are this phi - undefined
                self.create_undef(ty)
            } else {
                // Non-trivial phi
                self.function.blocks.get_mut(&block).unwrap().phis.push(HirPhi {
                    result: phi_val,
                    ty,
                    incoming,
                });
                phi_val
            }
        }
    }
    
    /// Seal a block (all predecessors known)
    fn seal_block(&mut self, block: HirId) {
        self.sealed_blocks.insert(block);

        // CRITICAL: When using IDF-based SSA, don't fill phis during sealing!
        // IDF places all phis upfront, and we fill them AFTER all blocks are processed.
        // Filling during sealing causes phis to be filled before their predecessor
        // blocks are processed, leading to undef values.
        if self.idf_placement_done {
            // Skip phi filling - will be done in batch after all blocks processed
            return;
        }

        // Process incomplete phis for this block (demand-driven SSA only)
        let incomplete: Vec<_> = self.incomplete_phis.iter()
            .filter(|((b, _), _)| *b == block)
            .map(|((b, v), _)| (*b, *v))
            .collect();

        for (_, var) in incomplete {
            self.fill_incomplete_phi(block, var);
        }
    }
    
    /// Fill an incomplete phi using pure IDF approach
    /// Uses read_variable() which may traverse phis, but won't create new ones after IDF
    fn fill_incomplete_phi(&mut self, block: HirId, var: InternedString) {
        let phi_key = (block, var);
        if let Some(phi_val) = self.incomplete_phis.remove(&phi_key) {
            // Get predecessors
            let preds = self.function.blocks[&block].predecessors.clone();
            let mut incoming = Vec::new();

            log::debug!("[SSA] fill_incomplete_phi: var={:?}, block={:?}, phi_val={:?}, predecessors={}",
                     var, block, phi_val, preds.len());
            for pred in preds {
                // Use read_variable to get the value - this will traverse phis if needed
                let pred_val = self.read_variable(var, pred);
                log::debug!("[SSA]   from pred {:?} → value {:?}", pred, pred_val);
                incoming.push((pred_val, pred));
            }

            log::debug!("[SSA] Phi complete with {} incoming values", incoming.len());

            // Update the phi
            if let Some(phi) = self.function.blocks.get_mut(&block)
                .and_then(|b| b.phis.iter_mut().find(|p| p.result == phi_val))
            {
                phi.incoming = incoming;
            }
        }
    }
    
    /// Fill all IDF-placed phis
    fn fill_incomplete_phis(&mut self) {
        let incomplete: Vec<_> = self.incomplete_phis.keys().cloned().collect();
        for (block, var) in incomplete {
            self.fill_incomplete_phi(block, var);
        }
    }

    /// Verify and fix phi incoming values
    /// Ensures all phis have incoming values from all predecessors
    fn verify_and_fix_phi_incoming(&mut self) {
        // Collect all blocks that have phis
        let blocks_with_phis: Vec<HirId> = self.function.blocks.iter()
            .filter(|(_, block)| !block.phis.is_empty())
            .map(|(id, _)| *id)
            .collect();

        for block_id in blocks_with_phis {
            // Get predecessors
            let preds = self.function.blocks[&block_id].predecessors.clone();

            // For each phi in this block
            let mut phi_fixes = Vec::new();

            {
                let block = &self.function.blocks[&block_id];
                for phi in &block.phis {
                    // Check if phi has incoming from all predecessors
                    let incoming_blocks: HashSet<_> = phi.incoming.iter()
                        .map(|(_, block)| *block)
                        .collect();

                    // Find missing predecessors
                    let missing: Vec<_> = preds.iter()
                        .filter(|pred| !incoming_blocks.contains(pred))
                        .copied()
                        .collect();

                    if !missing.is_empty() {
                        // We need to fix this phi
                        // Try to find which variable this phi is for by checking definitions
                        if let Some(var) = self.find_variable_for_phi(block_id, phi.result) {
                            phi_fixes.push((phi.result, var, missing));
                        }
                    }
                }
            }

            // Apply fixes
            for (phi_val, var, missing_preds) in phi_fixes {
                for pred in missing_preds {
                    // Read the variable value from the predecessor
                    let pred_val = self.read_variable(var, pred);

                    // Add to phi's incoming list
                    if let Some(block) = self.function.blocks.get_mut(&block_id) {
                        if let Some(phi) = block.phis.iter_mut().find(|p| p.result == phi_val) {
                            phi.incoming.push((pred_val, pred));
                        }
                    }
                }
            }
        }
    }

    /// Find which variable a phi corresponds to by checking definitions
    fn find_variable_for_phi(&self, block_id: HirId, phi_val: HirId) -> Option<InternedString> {
        if let Some(defs) = self.definitions.get(&block_id) {
            for (var, &val) in defs {
                if val == phi_val {
                    return Some(*var);
                }
            }
        }
        None
    }

    /// Scan CFG to collect variable types from Let statements
    /// This must be done before phi placement since phis need correct types
    fn scan_cfg_for_variable_types(&mut self, cfg: &crate::typed_cfg::TypedControlFlowGraph) {
        use zyntax_typed_ast::typed_ast::{TypedStatement};

        for node_idx in cfg.graph.node_indices() {
            let typed_block = &cfg.graph[node_idx];

            // Scan all statements in the block
            for stmt in &typed_block.statements {
                match &stmt.node {
                    // Let statements have type annotations
                    TypedStatement::Let(let_stmt) => {
                        let hir_type = self.convert_type(&let_stmt.ty);
                        self.var_types.insert(let_stmt.name, hir_type);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Scan CFG to find which blocks write which variables
    /// This is done by analyzing TypedCFG WITHOUT translating to HIR
    fn scan_cfg_for_variable_writes(&self, cfg: &crate::typed_cfg::TypedControlFlowGraph) -> IndexMap<HirId, HashSet<InternedString>> {
        use zyntax_typed_ast::typed_ast::{TypedExpression, TypedStatement, BinaryOp};

        let mut writes: IndexMap<HirId, HashSet<InternedString>> = IndexMap::new();

        for node_idx in cfg.graph.node_indices() {
            let typed_block = &cfg.graph[node_idx];
            let block_id = typed_block.id;
            let mut block_writes = HashSet::new();

            // Scan all statements in the block
            for stmt in &typed_block.statements {
                match &stmt.node {
                    // Let statements define variables
                    TypedStatement::Let(let_stmt) => {
                        if let_stmt.initializer.is_some() {
                            block_writes.insert(let_stmt.name);
                        }
                    }
                    // Expression statements might contain assignments
                    TypedStatement::Expression(expr) => {
                        self.collect_assigned_vars(expr, &mut block_writes);
                    }
                    _ => {}
                }
            }

            if !block_writes.is_empty() {
                writes.insert(block_id, block_writes);
            }
        }

        writes
    }

    /// Scan CFG to find which variables have their address taken
    /// Variables with their address taken need stack allocation instead of SSA registers
    fn scan_cfg_for_address_taken_vars(&mut self, cfg: &crate::typed_cfg::TypedControlFlowGraph) {
        use zyntax_typed_ast::typed_ast::{TypedExpression, TypedStatement};

        for node_idx in cfg.graph.node_indices() {
            let typed_block = &cfg.graph[node_idx];

            // Scan all statements in the block
            for stmt in &typed_block.statements {
                match &stmt.node {
                    TypedStatement::Let(let_stmt) => {
                        // Check initializer for address-of expressions
                        if let Some(init) = &let_stmt.initializer {
                            self.collect_address_taken_vars_from_expr(init);
                        }
                    }
                    TypedStatement::Expression(expr) => {
                        self.collect_address_taken_vars_from_expr(expr);
                    }
                    _ => {}
                }
            }
        }

        log::debug!("[SSA] Address-taken variables: {:?}", self.address_taken_vars);
    }

    /// Recursively collect variables that have their address taken
    fn collect_address_taken_vars_from_expr(&mut self, expr: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>) {
        use zyntax_typed_ast::typed_ast::TypedExpression;

        match &expr.node {
            TypedExpression::Reference(reference) => {
                // If the reference is to a variable, mark it as address-taken
                if let TypedExpression::Variable(var_name) = &reference.expr.node {
                    self.address_taken_vars.insert(*var_name);
                    log::debug!("[SSA] Found address-taken variable: {:?}", var_name);
                }
                // Also recurse into the inner expression
                self.collect_address_taken_vars_from_expr(&reference.expr);
            }
            TypedExpression::Binary(binary) => {
                self.collect_address_taken_vars_from_expr(&binary.left);
                self.collect_address_taken_vars_from_expr(&binary.right);
            }
            TypedExpression::Unary(unary) => {
                self.collect_address_taken_vars_from_expr(&unary.operand);
            }
            TypedExpression::Call(call) => {
                self.collect_address_taken_vars_from_expr(&call.callee);
                for arg in &call.positional_args {
                    self.collect_address_taken_vars_from_expr(arg);
                }
            }
            TypedExpression::Field(field) => {
                self.collect_address_taken_vars_from_expr(&field.object);
            }
            TypedExpression::Index(index) => {
                self.collect_address_taken_vars_from_expr(&index.object);
                self.collect_address_taken_vars_from_expr(&index.index);
            }
            TypedExpression::Dereference(inner) => {
                self.collect_address_taken_vars_from_expr(inner);
            }
            TypedExpression::If(if_expr) => {
                self.collect_address_taken_vars_from_expr(&if_expr.condition);
                self.collect_address_taken_vars_from_expr(&if_expr.then_branch);
                self.collect_address_taken_vars_from_expr(&if_expr.else_branch);
            }
            TypedExpression::Block(block) => {
                for stmt in &block.statements {
                    if let zyntax_typed_ast::typed_ast::TypedStatement::Expression(e) = &stmt.node {
                        self.collect_address_taken_vars_from_expr(e);
                    }
                }
            }
            // Leaf nodes - no recursion needed
            TypedExpression::Literal(_) |
            TypedExpression::Variable(_) => {}
            // Handle other expression types as needed
            _ => {}
        }
    }

    /// Place phis using Iterated Dominance Frontier (IDF) algorithm
    /// This is the proper SSA construction approach that places phis only where needed
    fn place_phis_using_idf(&mut self, cfg: &crate::typed_cfg::TypedControlFlowGraph) {
        // CRITICAL: Scan CFG to collect variable types FIRST
        // Phis need correct types when they're created
        self.scan_cfg_for_variable_types(cfg);

        // Compute dominance information
        let dom_info = DominanceInfo::compute(cfg);

        // Scan CFG to find variable writes BEFORE translating blocks
        let scanned_writes = self.scan_cfg_for_variable_writes(cfg);

        // Group variable writes by variable name
        // This creates a map: variable -> set of blocks that define it
        let mut defs_per_var: IndexMap<InternedString, HashSet<HirId>> = IndexMap::new();

        // Use scanned writes instead of self.variable_writes
        for (block_id, vars) in &scanned_writes {
            for &var in vars {
                defs_per_var.entry(var).or_insert_with(HashSet::new).insert(*block_id);
            }
        }

        // IMPORTANT: Only parameters are defined in entry block
        // Other variables (let statements) will be found by scan_cfg_for_variable_writes
        // So we DON'T need to add entry block for all variables

        // For each variable, compute IDF and place phis
        for (var, def_blocks) in defs_per_var {
            // Compute iterated dominance frontier
            let idf = self.compute_idf(&def_blocks, &dom_info);

            // Place phi at each block in IDF
            for &block_id in &idf {
                let phi_key = (block_id, var);

                // Skip if phi already exists (from demand-driven creation)
                if self.incomplete_phis.contains_key(&phi_key) {
                    continue;
                }

                // Get variable type
                let ty = self.var_types.get(&var).cloned().unwrap_or(HirType::I64);

                // Create incomplete phi
                let phi_val = self.create_value(ty.clone(), HirValueKind::Instruction);

                self.incomplete_phis.insert(phi_key, phi_val);
                self.function.blocks.get_mut(&block_id).unwrap().phis.push(HirPhi {
                    result: phi_val,
                    ty,
                    incoming: vec![],
                });

                // Define it in the block so reads find it
                self.definitions.get_mut(&block_id).unwrap().insert(var, phi_val);
            }
        }
    }

    /// Compute Iterated Dominance Frontier for a set of blocks
    fn compute_idf(
        &self,
        def_blocks: &HashSet<HirId>,
        dom_info: &DominanceInfo
    ) -> HashSet<HirId> {
        let mut worklist: Vec<HirId> = def_blocks.iter().copied().collect();
        let mut idf = HashSet::new();
        let mut processed = HashSet::new();

        while let Some(block) = worklist.pop() {
            if !processed.insert(block) {
                continue; // Already processed
            }

            // Get dominance frontier of this block
            if let Some(frontier) = dom_info.dom_frontier.get(&block) {
                for &df_block in frontier {
                    if idf.insert(df_block) {
                        // New block added to IDF - process its frontier too
                        worklist.push(df_block);
                    }
                }
            }
        }

        idf
    }

    /// Collect variables written in loop blocks by analyzing TypedCFG
    fn collect_loop_vars_from_cfg(
        &self,
        cfg: &crate::typed_cfg::TypedControlFlowGraph,
        block_id: HirId,
        header_id: HirId,
        vars: &mut HashSet<InternedString>
    ) {
        use zyntax_typed_ast::typed_ast::{TypedStatement, TypedExpression};

        // Avoid infinite recursion
        if block_id == header_id {
            return;
        }

        // Find the TypedBasicBlock for this block_id
        for node_idx in cfg.graph.node_indices() {
            let typed_block = &cfg.graph[node_idx];
            if typed_block.id == block_id {
                // Scan statements for variable assignments
                for stmt in &typed_block.statements {
                    match &stmt.node {
                        TypedStatement::Let(let_stmt) => {
                            // Variable declaration
                            vars.insert(let_stmt.name);
                        }
                        TypedStatement::Expression(expr) => {
                            // Check for assignment expressions
                            self.collect_assigned_vars(expr, vars);
                        }
                        _ => {}
                    }
                }

                // Recursively collect from predecessors
                if let Some(hir_block) = self.function.blocks.get(&block_id) {
                    for &pred in &hir_block.predecessors {
                        if pred != header_id && pred != self.function.entry_block {
                            self.collect_loop_vars_from_cfg(cfg, pred, header_id, vars);
                        }
                    }
                }
                break;
            }
        }
    }

    /// Helper to collect variables from assignment expressions
    fn collect_assigned_vars(
        &self,
        expr: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>,
        vars: &mut HashSet<InternedString>
    ) {
        use zyntax_typed_ast::typed_ast::{TypedExpression, BinaryOp};

        match &expr.node {
            TypedExpression::Binary(bin) if matches!(bin.op, BinaryOp::Assign) => {
                // Assignment: target = value
                if let TypedExpression::Variable(name) = &bin.left.node {
                    vars.insert(*name);
                }
            }
            _ => {}
        }
    }

    /// Create a new SSA value
    fn create_value(&mut self, ty: HirType, kind: HirValueKind) -> HirId {
        self.function.create_value(ty, kind)
    }

    /// Create an undefined value
    fn create_undef(&mut self, ty: HirType) -> HirId {
        self.create_value(ty, HirValueKind::Undef)
    }

    /// Create a global string constant and return a value referencing it
    fn create_string_global(&mut self, string_name: InternedString) -> HirId {
        use crate::hir::{HirGlobal, HirConstant, Linkage, Visibility};

        // Create a unique global ID and name
        let global_id = HirId::new();

        // Create the global with the string data
        let global = HirGlobal {
            id: global_id,
            name: string_name, // Use the interned string name as the global name
            ty: HirType::Ptr(Box::new(HirType::I8)), // String is a pointer to i8
            initializer: Some(HirConstant::String(string_name)),
            is_const: true,
            is_thread_local: false,
            linkage: Linkage::Private,
            visibility: Visibility::Default,
        };

        // Store the global for later addition to the module
        self.string_globals.push(global);

        // Create a value that references this global
        let value_id = self.create_value(
            HirType::Ptr(Box::new(HirType::I8)),
            HirValueKind::Global(global_id)
        );

        value_id
    }

    /// Add an instruction to a block
    fn add_instruction(&mut self, block: HirId, inst: HirInstruction) {
        log::debug!("[SSA] add_instruction: block={:?}, inst={:?}", block, std::mem::discriminant(&inst));
        if let HirInstruction::Binary { result, op, .. } = &inst {
            log::debug!("[SSA]   -> Binary op={:?}, result={:?}", op, result);
        }
        self.function.blocks.get_mut(&block).unwrap().add_instruction(inst);
    }
    
    /// Add a use of a value
    fn add_use(&mut self, value: HirId, user: HirId) {
        if let Some(val) = self.function.values.get_mut(&value) {
            val.uses.insert(user);
        }
    }
    
    /// Compute dominance order for processing
    fn compute_dominance_order(&self, _cfg: &ControlFlowGraph) -> Vec<HirId> {
        // Simple DFS order for now
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![self.function.entry_block];
        
        while let Some(block) = stack.pop() {
            if visited.insert(block) {
                order.push(block);
                
                if let Some(block_info) = self.function.blocks.get(&block) {
                    for &succ in &block_info.successors {
                        if !visited.contains(&succ) {
                            stack.push(succ);
                        }
                    }
                }
            }
        }
        
        order
    }
    
    /// Build def-use and use-def chains
    fn build_def_use_chains(&self) -> (IndexMap<HirId, HashSet<HirId>>, IndexMap<HirId, HirId>) {
        let mut def_use = IndexMap::new();
        let mut use_def = IndexMap::new();
        
        // Process all values
        for (value_id, value) in &self.function.values {
            def_use.insert(*value_id, value.uses.clone());
            
            // For each use, record the definition
            for &use_id in &value.uses {
                use_def.insert(use_id, *value_id);
            }
        }
        
        (def_use, use_def)
    }

    /// Resolve a method call to a mangled function name
    /// Looks up which trait impl provides the method for the receiver type
    fn resolve_method_to_function(
        &self,
        receiver_type: &Type,
        method_name: InternedString,
    ) -> CompilerResult<InternedString> {
        use zyntax_typed_ast::TypeId;

        // Get the type ID from Named type
        let type_id = match receiver_type {
            Type::Named { id, .. } => *id,
            Type::Extern { name, .. } => {
                // For extern types, method calls go through the impl block
                // The mangled name format is: {TypeName}${method_name} (no leading $)
                // E.g., Tensor$add for Tensor.add()
                let arena = self.arena.lock().unwrap();
                let type_name_str = arena.resolve_string(*name)
                    .ok_or_else(|| crate::CompilerError::Analysis("Could not resolve extern type name".into()))?;
                let method_name_str = arena.resolve_string(method_name)
                    .ok_or_else(|| crate::CompilerError::Analysis("Could not resolve method name".into()))?;

                // Strip $ prefix if present (extern type names often have runtime_prefix like "$Tensor")
                let base_type_name = type_name_str.strip_prefix('$').unwrap_or(&type_name_str);
                // Format: {TypeName}${method_name} (matches lowering.rs format)
                let mangled = format!("{}${}", base_type_name, method_name_str);
                drop(arena);
                return Ok(InternedString::new_global(&mangled));
            }
            Type::Unknown | Type::Any => {
                let method_name_str = self.arena.lock().unwrap().resolve_string(method_name)
                    .unwrap_or_default().to_string();
                return Err(crate::CompilerError::Analysis(
                    format!("Cannot resolve method '{}' on unknown type - add a type annotation to the variable", method_name_str)
                ))
            }
            _ => {
                return Err(crate::CompilerError::Analysis(
                    format!("Cannot call methods on non-nominal type: {:?}", receiver_type)
                ))
            }
        };

        // First, check for inherent methods (impl Type { ... } without trait)
        // Inherent methods take priority over trait methods
        eprintln!("[METHOD DISPATCH] Looking up type_id {:?} in type_registry", type_id);
        if let Some(type_def) = self.type_registry.get_type_by_id(type_id) {
            eprintln!("[METHOD DISPATCH] Checking type {} ({:?}), {} inherent methods",
                type_def.name.resolve_global().unwrap_or_default(), type_id, type_def.methods.len());
            eprintln!("[METHOD DISPATCH] Looking for method: {} ({:?})",
                method_name.resolve_global().unwrap_or_default(), method_name);
            for method in &type_def.methods {
                eprintln!("[METHOD DISPATCH]   Available method: {} ({:?}), match={}",
                    method.name.resolve_global().unwrap_or_default(), method.name, method.name == method_name);
                if method.name == method_name {
                    // Found inherent method!
                    // Format for inherent methods: {TypeName}${method_name}
                    let mangled = {
                        let arena = self.arena.lock().unwrap();
                        let type_name_str = arena.resolve_string(type_def.name)
                            .unwrap_or_default();
                        let method_name_str = arena.resolve_string(method_name)
                            .unwrap_or_default();
                        format!("{}${}", type_name_str, method_name_str)
                    };

                    eprintln!("[METHOD DISPATCH] Found inherent method '{}' for type '{}'",
                        method_name, type_def.name);
                    return Ok(InternedString::new_global(&mangled));
                }
            }
        } else {
            eprintln!("[METHOD DISPATCH] Type {:?} not found in type_registry!", type_id);
        }

        // Look up which traits this type implements
        // Check all implementations in the type registry
        for (_trait_id, impls) in self.type_registry.iter_implementations() {
            for impl_def in impls {
                // Check if this impl is for our type
                if let Type::Named { id: impl_type_id, .. } = &impl_def.for_type {
                    if *impl_type_id == type_id {
                        // Check if this impl has the method we're looking for
                        for method in &impl_def.methods {
                            if method.signature.name == method_name {
                                // Found it! Return mangled name
                                // Format: {TypeName}${TraitName}${method_name}
                                let type_def = self.type_registry.get_type_by_id(type_id)
                                    .ok_or_else(|| crate::CompilerError::Analysis(
                                        format!("Type {:?} not found in registry", type_id)
                                    ))?;

                                let trait_def = self.type_registry.get_trait_by_id(impl_def.trait_id)
                                    .ok_or_else(|| crate::CompilerError::Analysis(
                                        format!("Trait {:?} not found in registry", impl_def.trait_id)
                                    ))?;

                                let mangled = {
                                    let arena = self.arena.lock().unwrap();
                                    let type_name_str = arena.resolve_string(type_def.name)
                                        .unwrap_or_default();
                                    let trait_name_str = arena.resolve_string(trait_def.name)
                                        .unwrap_or_default();
                                    let method_name_str = arena.resolve_string(method_name)
                                        .unwrap_or_default();
                                    format!("{}${}${}", type_name_str, trait_name_str, method_name_str)
                                };

                                eprintln!("[METHOD DISPATCH] Found trait method (trait: {:?})",
                                    impl_def.trait_id);
                                return Ok(InternedString::new_global(&mangled));
                            }
                        }
                    }
                }
            }
        }

        // Method not found
        let arena = self.arena.lock().unwrap();
        let method_name_str = arena.resolve_string(method_name)
            .unwrap_or("unknown")
            .to_string();
        drop(arena);

        Err(crate::CompilerError::Analysis(
            format!("Method '{}' not found for type '{:?}'", method_name_str, receiver_type)
        ))
    }

    /// Convert frontend type to HIR type
    fn convert_type(&self, ty: &Type) -> HirType {
        use zyntax_typed_ast::PrimitiveType;
        
        match ty {
            Type::Primitive(prim) => match prim {
                PrimitiveType::Bool => HirType::Bool,
                PrimitiveType::I8 => HirType::I8,
                PrimitiveType::I16 => HirType::I16,
                PrimitiveType::I32 => HirType::I32,
                PrimitiveType::I64 => HirType::I64,
                PrimitiveType::I128 => HirType::I128,
                PrimitiveType::U8 => HirType::U8,
                PrimitiveType::U16 => HirType::U16,
                PrimitiveType::U32 => HirType::U32,
                PrimitiveType::U64 => HirType::U64,
                PrimitiveType::U128 => HirType::U128,
                PrimitiveType::F32 => HirType::F32,
                PrimitiveType::F64 => HirType::F64,
                PrimitiveType::Unit => HirType::Void,
                _ => HirType::I64, // Default
            },
            Type::Tuple(types) if types.is_empty() => HirType::Void,
            Type::Reference { ty, .. } => HirType::Ptr(Box::new(self.convert_type(ty))),
            Type::Array { element_type, size: Some(size_val), .. } => {
                // Extract size from ConstValue
                let size = match size_val {
                    ConstValue::Int(i) => *i as u64,
                    ConstValue::UInt(u) => *u,
                    _ => 0, // Default size
                };
                HirType::Array(
                    Box::new(self.convert_type(element_type)),
                    size
                )
            },
            Type::Optional(inner_ty) => {
                // Convert Optional<T> to a tagged union: enum { None, Some(T) }
                use crate::hir::{HirUnionType, HirUnionVariant};

                // Convert inner type FIRST before locking arena
                let inner_hir_type = self.convert_type(inner_ty);

                let mut arena = self.arena.lock().unwrap();
                let none_name = arena.intern_string("None");
                let some_name = arena.intern_string("Some");
                drop(arena);

                HirType::Union(Box::new(HirUnionType {
                    name: None,
                    variants: vec![
                        HirUnionVariant {
                            name: none_name,
                            ty: HirType::Void,
                            discriminant: 0,
                        },
                        HirUnionVariant {
                            name: some_name,
                            ty: inner_hir_type,
                            discriminant: 1,
                        },
                    ],
                    discriminant_type: Box::new(HirType::U32),
                    is_c_union: false,
                }))
            },
            Type::Result { ok_type, err_type } => {
                // Convert Result<T, E> to a tagged union: enum { Ok(T), Err(E) }
                use crate::hir::{HirUnionType, HirUnionVariant};

                // Convert inner types FIRST before locking arena
                let ok_hir_type = self.convert_type(ok_type);
                let err_hir_type = self.convert_type(err_type);

                let mut arena = self.arena.lock().unwrap();
                let ok_name = arena.intern_string("Ok");
                let err_name = arena.intern_string("Err");
                drop(arena);

                HirType::Union(Box::new(HirUnionType {
                    name: None,
                    variants: vec![
                        HirUnionVariant {
                            name: ok_name,
                            ty: ok_hir_type,
                            discriminant: 0,
                        },
                        HirUnionVariant {
                            name: err_name,
                            ty: err_hir_type,
                            discriminant: 1,
                        },
                    ],
                    discriminant_type: Box::new(HirType::U32),
                    is_c_union: false,
                }))
            },
            Type::Struct { fields, .. } => {
                // Convert inline struct type to HirType::Struct
                use crate::hir::HirStructType;

                let hir_fields: Vec<HirType> = fields.iter()
                    .map(|field| self.convert_type(&field.ty))
                    .collect();

                HirType::Struct(HirStructType {
                    name: None,
                    fields: hir_fields,
                    packed: false,
                })
            },
            Type::Extern { name, .. } => {
                // Extern/Opaque types are pointers to opaque structs at the HIR level
                // The name is used for trait dispatch (e.g., $Tensor -> $Tensor$add)
                HirType::Ptr(Box::new(HirType::Opaque(*name)))
            },
            Type::Named { id, .. } => {
                // Look up the type definition in the registry
                if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                    use crate::hir::HirStructType;
                    use zyntax_typed_ast::type_registry::TypeKind;

                    // Abstract types are zero-cost wrappers with struct layout
                    // They must be treated as structs for field access to work
                    // The backend will optimize away the wrapper at codegen time
                    if let TypeKind::Abstract { .. } = &type_def.kind {
                        eprintln!("[CONVERT TYPE SSA] Abstract type '{}' → struct with {} fields",
                            type_def.name.resolve_global().unwrap_or_default(),
                            type_def.fields.len());

                        let hir_fields: Vec<HirType> = type_def.fields.iter()
                            .map(|field| self.convert_type(&field.ty))
                            .collect();

                        return HirType::Struct(HirStructType {
                            name: Some(type_def.name),
                            fields: hir_fields,
                            packed: false,
                        });
                    }

                    // Regular Named types (structs, classes, enums) convert to struct types
                    // Convert the fields to HIR types
                    let hir_fields: Vec<HirType> = type_def.fields.iter()
                        .map(|field| self.convert_type(&field.ty))
                        .collect();

                    HirType::Struct(HirStructType {
                        name: Some(type_def.name),
                        fields: hir_fields,
                        packed: false,
                    })
                } else {
                    eprintln!("[WARN] Named type {:?} not found in registry, defaulting to I64", id);
                    HirType::I64
                }
            },
            Type::Unresolved(name) => {
                // Look up the type in TypeRegistry by name and convert as Named type
                if let Some(type_def) = self.type_registry.get_type_by_name(*name) {
                    use crate::hir::HirStructType;
                    use zyntax_typed_ast::type_registry::TypeKind;

                    eprintln!("[CONVERT TYPE SSA] Unresolved type '{}' → resolved to {:?}",
                        name.resolve_global().unwrap_or_default(),
                        type_def.kind);

                    // Abstract types are zero-cost wrappers with struct layout
                    if let TypeKind::Abstract { .. } = &type_def.kind {
                        let hir_fields: Vec<HirType> = type_def.fields.iter()
                            .map(|field| self.convert_type(&field.ty))
                            .collect();

                        return HirType::Struct(HirStructType {
                            name: Some(type_def.name),
                            fields: hir_fields,
                            packed: false,
                        });
                    }

                    // Regular types (structs, classes, enums) convert to struct types
                    let hir_fields: Vec<HirType> = type_def.fields.iter()
                        .map(|field| self.convert_type(&field.ty))
                        .collect();

                    HirType::Struct(HirStructType {
                        name: Some(type_def.name),
                        fields: hir_fields,
                        packed: false,
                    })
                } else {
                    eprintln!("[WARN] Unresolved type '{}' not found in registry, defaulting to I64",
                        name.resolve_global().unwrap_or_default());
                    HirType::I64
                }
            },
            Type::Any => {
                // Type::Any means "infer from context" - we need to check the initializer's type
                // For now, default to I64 but this should be improved
                eprintln!("[WARN] Type::Any encountered in convert_type, defaulting to I64");
                HirType::I64
            },
            _ => HirType::I64, // Default for complex types
        }
    }

    /// Convert HIR type back to TypedAST Type (reverse of convert_type)
    /// This is used when we need to look up type information that was stored in HIR format
    fn hir_type_to_typed_ast_type(&self, hir_ty: &HirType) -> Type {
        use zyntax_typed_ast::PrimitiveType;
        use zyntax_typed_ast::type_registry::NullabilityKind;

        match hir_ty {
            HirType::Bool => Type::Primitive(PrimitiveType::Bool),
            HirType::I8 => Type::Primitive(PrimitiveType::I8),
            HirType::I16 => Type::Primitive(PrimitiveType::I16),
            HirType::I32 => Type::Primitive(PrimitiveType::I32),
            HirType::I64 => Type::Primitive(PrimitiveType::I64),
            HirType::I128 => Type::Primitive(PrimitiveType::I128),
            HirType::U8 => Type::Primitive(PrimitiveType::U8),
            HirType::U16 => Type::Primitive(PrimitiveType::U16),
            HirType::U32 => Type::Primitive(PrimitiveType::U32),
            HirType::U64 => Type::Primitive(PrimitiveType::U64),
            HirType::U128 => Type::Primitive(PrimitiveType::U128),
            HirType::F32 => Type::Primitive(PrimitiveType::F32),
            HirType::F64 => Type::Primitive(PrimitiveType::F64),
            HirType::Void => Type::Primitive(PrimitiveType::Unit),
            HirType::Ptr(inner) => {
                Type::Reference {
                    ty: Box::new(self.hir_type_to_typed_ast_type(inner)),
                    mutability: zyntax_typed_ast::type_registry::Mutability::Mutable,
                    lifetime: None,
                    nullability: NullabilityKind::NonNull,
                }
            },
            HirType::Array(elem_ty, size) => {
                Type::Array {
                    element_type: Box::new(self.hir_type_to_typed_ast_type(elem_ty)),
                    size: Some(zyntax_typed_ast::type_registry::ConstValue::UInt(*size)),
                    nullability: NullabilityKind::NonNull,
                }
            },
            HirType::Struct(struct_ty) => {
                // If struct has a name, look it up in type registry to get TypeId
                if let Some(name) = struct_ty.name {
                    if let Some(type_def) = self.type_registry.get_type_by_name(name) {
                        return Type::Named {
                            id: type_def.id,
                            type_args: vec![],
                            const_args: vec![],
                            variance: vec![],
                            nullability: NullabilityKind::NonNull,
                        };
                    }
                }
                // Fallback: inline struct type
                Type::Struct {
                    fields: vec![], // We don't reconstruct fields here
                    is_anonymous: false,
                    nullability: NullabilityKind::NonNull,
                }
            },
            HirType::Union(_) => {
                // Optional/Result types - return as Any for now
                Type::Any
            },
            HirType::Opaque(name) => {
                // Check if this opaque type is actually a registered type (struct/abstract) in the registry
                if let Some(type_def) = self.type_registry.get_type_by_name(*name) {
                    // It's a known type, return as Named so field access works
                    return Type::Named {
                        id: type_def.id,
                        type_args: vec![],
                        const_args: vec![],
                        variance: vec![],
                        nullability: NullabilityKind::NonNull,
                    };
                }
                // Otherwise, it's a true extern/opaque type
                Type::Extern {
                    name: *name,
                    layout: None,
                }
            },
            _ => Type::Any, // Default fallback
        }
    }

    /// Convert binary operator
    fn convert_binary_op(&self, op: &zyntax_typed_ast::typed_ast::BinaryOp) -> crate::hir::BinaryOp {
        use zyntax_typed_ast::typed_ast::BinaryOp as FrontendOp;
        use crate::hir::BinaryOp as HirOp;
        
        match op {
            FrontendOp::Add => HirOp::Add,
            FrontendOp::Sub => HirOp::Sub,
            FrontendOp::Mul => HirOp::Mul,
            FrontendOp::Div => HirOp::Div,
            FrontendOp::Rem => HirOp::Rem,
            FrontendOp::BitAnd => HirOp::And,
            FrontendOp::BitOr => HirOp::Or,
            FrontendOp::BitXor => HirOp::Xor,
            FrontendOp::Shl => HirOp::Shl,
            FrontendOp::Shr => HirOp::Shr,
            FrontendOp::Eq => HirOp::Eq,
            FrontendOp::Ne => HirOp::Ne,
            FrontendOp::Lt => HirOp::Lt,
            FrontendOp::Le => HirOp::Le,
            FrontendOp::Gt => HirOp::Gt,
            FrontendOp::Ge => HirOp::Ge,
            _ => HirOp::Add, // Default
        }
    }
    
    /// Convert unary operator
    fn convert_unary_op(&self, op: &zyntax_typed_ast::typed_ast::UnaryOp) -> crate::hir::UnaryOp {
        use zyntax_typed_ast::typed_ast::UnaryOp as FrontendOp;
        use crate::hir::UnaryOp as HirOp;

        match op {
            FrontendOp::Minus => HirOp::Neg,
            FrontendOp::Not => HirOp::Not,
            _ => HirOp::Neg, // Default
        }
    }

    /// Try to dispatch a binary operator to a trait method call.
    /// Returns Some(result_value) if the operator should be dispatched via trait,
    /// or None if the regular binary instruction should be used.
    fn try_operator_trait_dispatch(
        &mut self,
        block_id: HirId,
        op: &zyntax_typed_ast::typed_ast::BinaryOp,
        left: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>,
        right: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>,
        result_type: &Type,
    ) -> CompilerResult<Option<HirId>> {
        use zyntax_typed_ast::typed_ast::BinaryOp as FrontendOp;

        // Only consider trait dispatch for non-primitive types
        let left_type = &left.ty;
        if !self.is_trait_dispatchable_type(left_type) {
            return Ok(None);
        }

        // Get the method name for this operator
        let method_name = match op {
            FrontendOp::Add => "add",
            FrontendOp::Sub => "sub",
            FrontendOp::Mul => "mul",
            FrontendOp::Div => "div",
            FrontendOp::Rem => "mod",
            FrontendOp::Eq => "eq",
            FrontendOp::Ne => "ne",
            FrontendOp::Lt => "lt",
            FrontendOp::Le => "le",
            FrontendOp::Gt => "gt",
            FrontendOp::Ge => "ge",
            FrontendOp::BitAnd => "bitand",
            FrontendOp::BitOr => "bitor",
            FrontendOp::BitXor => "bitxor",
            _ => return Ok(None), // No trait method for this operator
        };

        // Get the type name for constructing the method symbol
        let type_name = self.get_type_symbol_prefix(left_type);
        if type_name.is_none() {
            return Ok(None);
        }
        let type_name = type_name.unwrap();

        // Construct the method function name: TypeName$method
        // For extern types, the type_name already starts with '$' (e.g., "$Tensor")
        // so we just append $method to get "$Tensor$add"
        let method_symbol = if type_name.starts_with('$') {
            // Already has $ prefix (extern type)
            format!("{}${}", type_name, method_name)
        } else {
            // Regular type, add $ prefix for ZRTL compatibility
            format!("${}${}", type_name, method_name)
        };
        let method_name_interned = InternedString::new_global(&method_symbol);
        log::debug!("[SSA] Operator trait dispatch: {} for type {}", method_symbol, type_name);

        // Try to look up the function ID for this method
        // If it's a compiled ZynML function (like Duration$add), we need HirCallable::Function
        // If it's an external plugin function (like $Tensor$add), we need HirCallable::Symbol
        let callee = if let Some(&func_id) = self.function_symbols.get(&method_name_interned) {
            log::debug!("[SSA] Found function ID for {}", method_symbol);
            crate::hir::HirCallable::Function(func_id)
        } else {
            log::debug!("[SSA] No function ID found, using external symbol for {}", method_symbol);
            crate::hir::HirCallable::Symbol(method_symbol)
        };

        // Translate arguments
        let left_val = self.translate_expression(block_id, left)?;
        let right_val = self.translate_expression(block_id, right)?;

        // For binary operations, the result type should be the same as the operand type
        // (e.g., Tensor + Tensor = Tensor). Use left operand's type instead of expression type.
        let hir_result_type = self.convert_type(left_type);
        log::debug!("[SSA] Operator trait dispatch result type: {:?} (from left type: {:?})", hir_result_type, left_type);

        // Create call instruction to the trait method
        let result = if hir_result_type != HirType::Void {
            Some(self.create_value(hir_result_type.clone(), HirValueKind::Instruction))
        } else {
            None
        };

        let inst = HirInstruction::Call {
            result,
            callee,
            args: vec![left_val, right_val],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        };

        self.add_instruction(block_id, inst);
        self.add_use(left_val, result.unwrap_or(left_val));
        self.add_use(right_val, result.unwrap_or(right_val));

        Ok(Some(result.unwrap_or_else(|| self.create_undef(HirType::Void))))
    }

    /// Resolve the actual type for an expression, looking up variable types if needed.
    /// This is needed because expression nodes may have type `Any` even when the
    /// variable has a more specific type recorded in var_typed_ast_types.
    fn resolve_actual_type(&self, expr: &zyntax_typed_ast::typed_ast::TypedExpression, fallback: &Type) -> Type {
        use zyntax_typed_ast::typed_ast::TypedExpression;

        match expr {
            TypedExpression::Variable(name) => {
                // Look up the variable's actual type from our tracking
                if let Some(var_ty) = self.var_typed_ast_types.get(name) {
                    if !matches!(var_ty, Type::Any | Type::Unknown) {
                        log::debug!("[resolve_actual_type] Variable '{}' has type {:?}",
                            name.resolve_global().unwrap_or_default(), var_ty);
                        return var_ty.clone();
                    }
                }
                // Fall back to the expression's type
                fallback.clone()
            }
            _ => fallback.clone(),
        }
    }

    /// Check if a type should use trait dispatch for operators
    fn is_trait_dispatchable_type(&self, ty: &Type) -> bool {
        match ty {
            // Primitive types - use built-in operations, no trait dispatch
            Type::Primitive(_) => false,

            // All other types (Extern, Named, Unknown, etc.) should try trait dispatch
            // This allows any type with an impl block to use operator overloading
            Type::Extern { name, .. } => {
                let name_str = name.resolve_global()
                    .unwrap_or_else(|| {
                        let arena = self.arena.lock().unwrap();
                        arena.resolve_string(*name)
                            .map(|s| s.to_string())
                            .unwrap_or_default()
                    });
                log::debug!("[trait_dispatch] Type::Extern name='{}' -> dispatchable", name_str);
                true
            }
            Type::Named { id, .. } => {
                // Any named type can have trait impls
                if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                    let name = type_def.name.resolve_global()
                        .unwrap_or_else(|| {
                            let arena = self.arena.lock().unwrap();
                            arena.resolve_string(type_def.name)
                                .map(|s| s.to_string())
                                .unwrap_or_default()
                        });
                    log::debug!("[trait_dispatch] Type::Named name='{}' -> dispatchable", name);
                    return true;
                }
                // Also check HIR conversion
                let hir_ty = self.convert_type(ty);
                matches!(hir_ty, HirType::Opaque(_))
            }
            // Unknown types - try trait dispatch (will use the type name)
            Type::Unknown => {
                log::debug!("[trait_dispatch] Type::Unknown -> try trait dispatch");
                true
            }
            // Other types - try trait dispatch
            _ => {
                log::debug!("[trait_dispatch] Other type -> try trait dispatch");
                true
            }
        }
    }

    /// Get the symbol prefix for a type (e.g., "$Tensor" for $Tensor$add)
    fn get_type_symbol_prefix(&self, ty: &Type) -> Option<String> {
        match ty {
            // Extern types have the name directly
            Type::Extern { name, .. } => {
                let name_str = name.resolve_global()
                    .unwrap_or_else(|| {
                        let arena = self.arena.lock().unwrap();
                        arena.resolve_string(*name)
                            .map(|s| s.to_string())
                            .unwrap_or_default()
                    });
                log::debug!("[trait_dispatch] Type::Extern prefix: '{}'", name_str);
                Some(name_str)
            }
            Type::Named { id, .. } => {
                if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                    // Use the type name
                    let name = type_def.name.resolve_global()
                        .unwrap_or_else(|| {
                            let arena = self.arena.lock().unwrap();
                            arena.resolve_string(type_def.name)
                                .map(|s| s.to_string())
                                .unwrap_or_default()
                        });
                    log::debug!("[trait_dispatch] Type::Named prefix: '{}'", name);
                    Some(name)
                } else {
                    // If not in registry, try to get from HIR type
                    let hir_ty = self.convert_type(ty);
                    if let HirType::Opaque(name) = hir_ty {
                        let name_str = name.resolve_global()
                            .unwrap_or_else(|| {
                                let arena = self.arena.lock().unwrap();
                                arena.resolve_string(name)
                                    .map(|s| s.to_string())
                                    .unwrap_or_default()
                            });
                        log::debug!("[trait_dispatch] Type::Named (opaque) prefix: '{}'", name_str);
                        Some(name_str)
                    } else {
                        log::debug!("[trait_dispatch] Type::Named (no registry) -> None");
                        None
                    }
                }
            }
            // Unknown types - use the HIR representation
            Type::Unknown => {
                // Convert to HIR and try to extract name
                let hir_ty = self.convert_type(ty);
                if let HirType::Opaque(name) = hir_ty {
                    let name_str = name.resolve_global()
                        .unwrap_or_else(|| {
                            let arena = self.arena.lock().unwrap();
                            arena.resolve_string(name)
                                .map(|s| s.to_string())
                                .unwrap_or_default()
                        });
                    log::debug!("[trait_dispatch] Type::Unknown converted to opaque prefix: '{}'", name_str);
                    Some(name_str)
                } else {
                    log::debug!("[trait_dispatch] Type::Unknown (not opaque) -> None");
                    None
                }
            }
            _ => {
                log::debug!("[trait_dispatch] Other type -> None");
                None
            }
        }
    }

    /// Translate literal to constant (legacy - always uses default types)
    fn translate_literal(&self, lit: &zyntax_typed_ast::typed_ast::TypedLiteral) -> crate::hir::HirConstant {
        self.translate_literal_with_type(lit, &HirType::I32)
    }

    /// Translate literal to constant with target type information
    fn translate_literal_with_type(&self, lit: &zyntax_typed_ast::typed_ast::TypedLiteral, target_ty: &HirType) -> crate::hir::HirConstant {
        use zyntax_typed_ast::typed_ast::TypedLiteral;
        use crate::hir::HirConstant;

        match lit {
            TypedLiteral::Bool(b) => HirConstant::Bool(*b),
            // Integer literals respect the target type
            TypedLiteral::Integer(i) => {
                match target_ty {
                    HirType::I8 => HirConstant::I8(*i as i8),
                    HirType::I16 => HirConstant::I16(*i as i16),
                    HirType::I32 => HirConstant::I32(*i as i32),
                    HirType::I64 => HirConstant::I64(*i as i64),
                    HirType::U8 => HirConstant::U8(*i as u8),
                    HirType::U16 => HirConstant::U16(*i as u16),
                    HirType::U32 => HirConstant::U32(*i as u32),
                    HirType::U64 => HirConstant::U64(*i as u64),
                    _ => HirConstant::I32(*i as i32), // Default to I32
                }
            }
            // Float literals use the target type (F32 or F64)
            TypedLiteral::Float(f) => {
                match target_ty {
                    HirType::F32 => HirConstant::F32(*f as f32),
                    _ => HirConstant::F64(*f),
                }
            }
            TypedLiteral::String(s) => HirConstant::String(*s),
            TypedLiteral::Char(c) => HirConstant::I32(*c as i32),
            TypedLiteral::Unit => HirConstant::Struct(vec![]),
            // Null is represented as a None variant in Optional type (discriminant 1)
            TypedLiteral::Null => HirConstant::I32(0), // null pointer / None discriminant
            // Undefined is used for uninitialized memory - use 0 as placeholder
            TypedLiteral::Undefined => HirConstant::I32(0), // Undefined/uninitialized
        }
    }
    
    /// Get field index in struct using TypeRegistry
    fn get_field_index(&self, struct_type: &Type, field_name: &InternedString) -> CompilerResult<u32> {
        // Resolve the type if it's unresolved
        let resolved_type = match struct_type {
            Type::Unresolved(name) => {
                // Look up the type in the registry by name
                self.type_registry.get_type_by_name(*name)
                    .map(|type_def| Type::Named {
                        id: type_def.id,
                        type_args: vec![],
                        const_args: vec![],
                        variance: vec![],
                        nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
                    })
                    .ok_or_else(|| crate::CompilerError::Analysis(
                        format!("Unresolved type {:?} not found in registry", name)
                    ))?
            }
            _ => struct_type.clone()
        };

        // Extract the type ID from the resolved struct type
        let type_id = match resolved_type {
            Type::Named { id, .. } => id,
            Type::Struct { fields, .. } => {
                // For inline struct types, search the fields directly
                for (idx, field) in fields.iter().enumerate() {
                    if &field.name == field_name {
                        return Ok(idx as u32);
                    }
                }
                return Err(crate::CompilerError::Analysis(
                    format!("Field {:?} not found in inline struct type", field_name)
                ));
            }
            _ => {
                return Err(crate::CompilerError::Analysis(
                    format!("Cannot access fields on non-struct type: {:?}", resolved_type)
                ));
            }
        };

        // Look up the type definition in the registry
        let type_def = self.type_registry.get_type_by_id(type_id)
            .ok_or_else(|| crate::CompilerError::Analysis(
                format!("Type with ID {:?} not found in registry", type_id)
            ))?;

        // Find the field index in the type definition
        eprintln!("[DEBUG] Looking for field {:?} in type {:?} which has {} fields",
            field_name, type_def.name, type_def.fields.len());
        for (idx, field) in type_def.fields.iter().enumerate() {
            eprintln!("[DEBUG]   Field {}: name={:?}", idx, field.name);
            if &field.name == field_name {
                return Ok(idx as u32);
            }
        }

        // Field not found - also try string comparison as fallback
        eprintln!("[DEBUG] Direct comparison failed, trying string resolution");
        for (idx, field) in type_def.fields.iter().enumerate() {
            if let (Some(field_name_str), Some(lookup_name_str)) =
                (field.name.resolve_global(), field_name.resolve_global()) {
                eprintln!("[DEBUG]   Comparing '{}' == '{}'", field_name_str, lookup_name_str);
                if field_name_str == lookup_name_str {
                    eprintln!("[DEBUG] Found match by string comparison!");
                    return Ok(idx as u32);
                }
            }
        }

        // Field not found
        Err(crate::CompilerError::Analysis(
            format!("Field {:?} not found in type {:?}", field_name, type_def.name)
        ))
    }

    /// Translate assignment expression
    ///
    /// Handles: variable assignment, field assignment, array element assignment
    fn translate_assignment(
        &mut self,
        block_id: HirId,
        target: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>,
        value_expr: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::typed_ast::TypedExpression;

        // First evaluate the right-hand side (the value to assign)
        let value = self.translate_expression(block_id, value_expr)?;

        // Now handle different assignment targets
        match &target.node {
            // Simple variable assignment: x = value
            TypedExpression::Variable(name) => {
                // Check if this is an address-taken variable - store to stack slot
                if let Some(&stack_slot) = self.stack_slots.get(name) {
                    self.add_instruction(block_id, HirInstruction::Store {
                        value,
                        ptr: stack_slot,
                        align: 8,
                        volatile: false,
                    });
                    log::debug!("[SSA] Store to address-taken var {:?} stack slot {:?}", name, stack_slot);
                } else {
                    // Use write_variable to update the SSA variable tracking
                    self.write_variable(*name, block_id, value);
                }
                // Assignment expression returns the assigned value
                Ok(value)
            }

            // Field assignment: obj.field = value
            TypedExpression::Field(_) => {
                self.translate_lvalue_assign(block_id, target, value)?;
                Ok(value)
            }

            // Array element assignment: arr[index] = value
            TypedExpression::Index(_) => {
                self.translate_lvalue_assign(block_id, target, value)?;
                Ok(value)
            }

            // Dereference assignment: *ptr = value
            TypedExpression::Dereference(inner) => {
                // Evaluate the pointer
                let ptr = self.translate_expression(block_id, inner)?;

                // Create a Store instruction to write through the pointer
                let inst = HirInstruction::Store {
                    value,
                    ptr,
                    align: 8, // Default alignment (can be refined based on type)
                    volatile: false,
                };

                self.add_instruction(block_id, inst);
                self.add_use(ptr, value);
                self.add_use(value, value); // Value uses itself

                Ok(value)
            }

            // Invalid assignment target
            _ => Err(crate::CompilerError::Analysis(
                format!("Invalid assignment target: {:?}", target.node)
            ))
        }
    }

    /// Handle complex lvalue assignment
    fn translate_lvalue_assign(
        &mut self,
        block_id: HirId,
        target: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>,
        value: HirId
    ) -> CompilerResult<()> {
        use zyntax_typed_ast::typed_ast::TypedExpression;

        match &target.node {
            TypedExpression::Field(field_access) => {
                // Field assignment: obj.field = value
                // Strategy: Load object, use InsertValue to update field, store back

                let object = &field_access.object;
                let field_name = &field_access.field;

                // Get the field index from the object's type
                let field_index = self.get_field_index(&object.ty, field_name)?;

                // Evaluate the object expression to get the struct value
                let object_val = self.translate_expression(block_id, object)?;

                // Create an InsertValue instruction to update the field
                let result_type = self.convert_type(&object.ty);
                let result = self.create_value(result_type.clone(), HirValueKind::Instruction);

                let inst = HirInstruction::InsertValue {
                    result,
                    aggregate: object_val,
                    value,
                    indices: vec![field_index],
                    ty: result_type,
                };

                self.add_instruction(block_id, inst);
                self.add_use(object_val, result);
                self.add_use(value, result);

                // If the object is a variable, write the updated value back
                if let TypedExpression::Variable(var_name) = &object.node {
                    self.write_variable(*var_name, block_id, result);
                }

                Ok(())
            }
            TypedExpression::Index(index_expr) => {
                // Array assignment: arr[index] = value
                // Strategy: Bounds check, load array, compute GEP, store value

                let array = &index_expr.object;
                let index = &index_expr.index;

                // Evaluate the array and index expressions
                let array_val = self.translate_expression(block_id, array)?;
                let index_val = self.translate_expression(block_id, index)?;

                // Get element type and size from array type
                let (element_type, array_size) = match &array.ty {
                    Type::Array { element_type, size, .. } => {
                        (self.convert_type(element_type), size.clone())
                    }
                    _ => return Err(crate::CompilerError::Analysis(
                        format!("Cannot index into non-array type: {:?}", array.ty)
                    ))
                };

                // BOUNDS CHECKING: Emit runtime bounds check if array has known size
                if let Some(size_const) = array_size {
                    // Extract numeric size from ConstValue
                    let size_value = match size_const {
                        ConstValue::Int(i) => i as u64,
                        ConstValue::UInt(u) => u,
                        _ => {
                            // For complex const values, skip bounds check for now
                            // TODO: Use const evaluator to compute size
                            0
                        }
                    };

                    if size_value > 0 {
                        // Create constant for array size
                        let size_hir = self.create_value(
                            HirType::I64,
                            HirValueKind::Constant(crate::hir::HirConstant::I64(size_value as i64))
                        );

                        // Compare: index < size
                        let cond_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
                        let cmp_inst = HirInstruction::Binary {
                            op: crate::hir::BinaryOp::Lt,
                            result: cond_result,
                            ty: HirType::Bool,
                            left: index_val,
                            right: size_hir,
                        };

                        self.add_instruction(block_id, cmp_inst);
                        self.add_use(index_val, cond_result);
                        self.add_use(size_hir, cond_result);

                        // BACKEND INTEGRATION: The comparison result can be used by backends
                        // for bounds checking optimization:
                        //
                        // - LLVM: Can use llvm.assume(cond_result) or emit conditional trap
                        // - Cranelift: Can use the comparison for trap insertion
                        //
                        // FUTURE ENHANCEMENT: Emit explicit conditional branch here
                        // This requires:
                        // 1. Create error block with trap instruction
                        // 2. Create continuation block for valid access
                        // 3. Split current block and emit CondBranch terminator
                        // 4. Continue lowering in continuation block
                        //
                        // Challenge: Expression translation doesn't currently support
                        // block splitting. Would need to refactor to return new block_id
                        // or use a different architecture (e.g., builder pattern with
                        // explicit control flow methods).
                        //
                        // For now: Comparison is emitted, backends can use it.
                        // Estimated effort to add explicit trap: 3-4 hours
                    }
                }

                // Create GetElementPtr instruction to get pointer to element
                // Note: ty should be the ARRAY type so Cranelift can calculate element size
                let array_hir_type = self.convert_type(&array.ty);
                let ptr_type = HirType::Ptr(Box::new(element_type.clone()));
                let ptr = self.create_value(ptr_type, HirValueKind::Instruction);

                let gep_inst = HirInstruction::GetElementPtr {
                    result: ptr,
                    ptr: array_val,
                    indices: vec![index_val],
                    ty: array_hir_type, // Pass array type, not element type
                };

                self.add_instruction(block_id, gep_inst);
                self.add_use(array_val, ptr);
                self.add_use(index_val, ptr);

                // Store the value at the computed address
                let store_inst = HirInstruction::Store {
                    value,
                    ptr,
                    align: 8, // Default alignment
                    volatile: false,
                };

                self.add_instruction(block_id, store_inst);
                self.add_use(ptr, value);
                self.add_use(value, value);

                Ok(())
            }
            _ => Ok(())
        }
    }

    // ========== PATTERN MATCHING LOWERING ==========

    /// Translate a match expression to HIR
    ///
    /// Strategy: Generate a decision tree with sequential arm testing:
    /// 1. Evaluate scrutinee once and store
    /// 2. For each arm:
    ///    - Create test block for pattern matching
    ///    - Create body block for arm execution
    ///    - If pattern matches and guard passes, jump to body
    ///    - Otherwise, continue to next arm
    /// 3. After body, jump to end block with result
    fn translate_match(
        &mut self,
        mut block_id: HirId,
        match_expr: &zyntax_typed_ast::typed_ast::TypedMatchExpr,
        result_ty: &Type,
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::typed_ast::TypedPattern;

        // Evaluate scrutinee
        let scrutinee_val = self.translate_expression(block_id, &match_expr.scrutinee)?;
        let scrutinee_ty = self.convert_type(&match_expr.scrutinee.ty);

        // Create end block for all arms to converge
        let end_block_id = HirId::new();
        self.function.blocks.insert(end_block_id, HirBlock::new(end_block_id));
        self.definitions.insert(end_block_id, IndexMap::new());

        // Result phi node will collect values from each arm
        let result_hir_ty = self.convert_type(result_ty);
        let result = self.create_value(result_hir_ty.clone(), HirValueKind::Instruction);

        // Track predecessor blocks and their result values for phi node
        let mut phi_operands: Vec<(HirId, HirId)> = Vec::new();

        // Process each match arm
        for (_arm_idx, arm) in match_expr.arms.iter().enumerate() {
            // Create block for testing this arm's pattern
            let test_block_id = HirId::new();
            self.function.blocks.insert(test_block_id, HirBlock::new(test_block_id));
            self.definitions.insert(test_block_id, IndexMap::new());

            // Create block for executing this arm's body
            let body_block_id = HirId::new();
            self.function.blocks.insert(body_block_id, HirBlock::new(body_block_id));
            self.definitions.insert(body_block_id, IndexMap::new());

            // Create block for next arm (or unreachable if last)
            let next_block_id = HirId::new();
            self.function.blocks.insert(next_block_id, HirBlock::new(next_block_id));
            self.definitions.insert(next_block_id, IndexMap::new());

            // Current block jumps to test block
            self.function.blocks.get_mut(&block_id).unwrap().terminator =
                HirTerminator::Branch { target: test_block_id };

            // In test block: check if pattern matches
            let pattern_matches = self.translate_pattern_test(
                test_block_id,
                &arm.pattern,
                scrutinee_val,
                &match_expr.scrutinee.ty,
            )?;

            // If guard exists, evaluate it and AND with pattern match result
            let final_condition = if let Some(guard_expr) = &arm.guard {
                let guard_val = self.translate_expression(test_block_id, guard_expr)?;

                // Create AND: pattern_matches && guard
                let and_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
                let and_inst = HirInstruction::Binary {
                    op: crate::hir::BinaryOp::And,
                    result: and_result,
                    ty: HirType::Bool,
                    left: pattern_matches,
                    right: guard_val,
                };

                self.add_instruction(test_block_id, and_inst);
                self.add_use(pattern_matches, and_result);
                self.add_use(guard_val, and_result);

                and_result
            } else {
                pattern_matches
            };

            // Conditional branch: if pattern matches (and guard passes), go to body, else next arm
            self.function.blocks.get_mut(&test_block_id).unwrap().terminator =
                HirTerminator::CondBranch {
                    condition: final_condition,
                    true_target: body_block_id,
                    false_target: next_block_id,
                };

            // In body block: execute arm body and jump to end
            let arm_result = self.translate_expression(body_block_id, &arm.body)?;

            self.function.blocks.get_mut(&body_block_id).unwrap().terminator =
                HirTerminator::Branch { target: end_block_id };

            // Record this arm's result for phi node
            phi_operands.push((arm_result, body_block_id));

            // Continue with next block
            block_id = next_block_id;
        }

        // After all arms, if we reach here, no pattern matched
        // This should be unreachable if exhaustiveness checking is done by type checker
        self.function.blocks.get_mut(&block_id).unwrap().terminator =
            HirTerminator::Unreachable;

        // Create phi node in end block to collect results
        if !phi_operands.is_empty() {
            let phi = HirPhi {
                result,
                ty: result_hir_ty,
                incoming: phi_operands,
            };

            self.function.blocks.get_mut(&end_block_id).unwrap().phis.push(phi);
        }

        // Set continuation block so that Return statements know to use end_block instead of entry block
        self.continuation_block = Some(end_block_id);

        Ok(result)
    }

    /// Translate the ? operator for Result<T, E> error propagation
    ///
    /// Desugars: `operation()?`
    /// Into:     ```
    ///           let tmp = operation();
    ///           match tmp {
    ///               Ok(value) => value,
    ///               Err(error) => return Err(error),
    ///           }
    ///           ```
    ///
    /// Gap 8 Phase 2 implementation.
    ///
    /// SIMPLIFIED VERSION: Instead of creating multiple blocks, this version:
    /// 1. Checks if the result is an error
    /// 2. If error, early returns
    /// 3. If ok, extracts and returns the value
    ///
    /// All done in the same block to avoid SSA complications with loop variables.
    fn translate_try_operator(
        &mut self,
        block_id: HirId,
        try_expr: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>,
        result_value_ty: &Type,
    ) -> CompilerResult<HirId> {
        // Evaluate the inner expression (e.g., risky_operation())
        let inner_result_val = self.translate_expression(block_id, try_expr)?;
        let inner_result_ty = &try_expr.ty;

        // Check if the type is Result<T, E>
        let (ok_ty, err_ty) = self.extract_result_type_args(inner_result_ty)?;

        // Convert types to HIR
        let hir_ok_ty = self.convert_type(&ok_ty);
        let hir_err_ty = self.convert_type(&err_ty);
        let hir_result_ty = self.convert_type(inner_result_ty);

        // Extract discriminant: Result is a union with discriminant 0 = Ok, 1 = Err
        let discriminant_val = self.create_value(HirType::U32, HirValueKind::Instruction);
        self.add_instruction(block_id, HirInstruction::ExtractValue {
            result: discriminant_val,
            ty: HirType::U32,
            aggregate: inner_result_val,
            indices: vec![0],
        });

        // Check if discriminant == 0 (Ok)
        let zero_const = self.create_value(
            HirType::U32,
            HirValueKind::Constant(crate::hir::HirConstant::U32(0))
        );

        let is_ok = self.create_value(HirType::Bool, HirValueKind::Instruction);
        self.add_instruction(block_id, HirInstruction::Binary {
            op: crate::hir::BinaryOp::Eq,
            result: is_ok,
            ty: HirType::U32,
            left: discriminant_val,
            right: zero_const,
        });

        // Extract the Ok value unconditionally
        // (This is safe because if it's an error, we'll return early)
        let ok_value = self.create_value(hir_ok_ty.clone(), HirValueKind::Instruction);
        self.add_instruction(block_id, HirInstruction::ExtractValue {
            result: ok_value,
            ty: hir_ok_ty.clone(),
            aggregate: inner_result_val,
            indices: vec![1],
        });

        // Create error block for early return
        let err_block_id = HirId::new();
        self.function.blocks.insert(err_block_id, HirBlock::new(err_block_id));
        self.definitions.insert(err_block_id, IndexMap::new());

        // Create continue block for success path
        let continue_block_id = HirId::new();
        self.function.blocks.insert(continue_block_id, HirBlock::new(continue_block_id));
        self.definitions.insert(continue_block_id, IndexMap::new());

        // Set up predecessors/successors
        {
            let block = self.function.blocks.get_mut(&block_id).unwrap();
            block.successors.push(continue_block_id);
            block.successors.push(err_block_id);
        }
        {
            let continue_block = self.function.blocks.get_mut(&continue_block_id).unwrap();
            continue_block.predecessors.push(block_id);
        }
        {
            let err_block = self.function.blocks.get_mut(&err_block_id).unwrap();
            err_block.predecessors.push(block_id);
        }

        // Branch: if is_ok, continue; else, go to error block
        self.function.blocks.get_mut(&block_id).unwrap().terminator =
            HirTerminator::CondBranch {
                condition: is_ok,
                true_target: continue_block_id,
                false_target: err_block_id,
            };

        // ERR BLOCK: Extract error and return early
        let err_value = self.create_value(hir_err_ty.clone(), HirValueKind::Instruction);
        self.add_instruction(err_block_id, HirInstruction::ExtractValue {
            result: err_value,
            ty: hir_err_ty.clone(), // Err value type (E)
            aggregate: inner_result_val,
            indices: vec![1], // Data field (contains the actual Ok/Err value)
        });

        // Construct a new Err(error) to return
        // Create Result<T, E> union with variant_index = 1 (Err)
        let return_err = self.create_value(hir_result_ty.clone(), HirValueKind::Instruction);
        self.add_instruction(err_block_id, HirInstruction::CreateUnion {
            result: return_err,
            union_ty: hir_result_ty.clone(),
            variant_index: 1, // Err variant
            value: err_value,
        });

        // Early return with Err(error)
        self.function.blocks.get_mut(&err_block_id).unwrap().terminator =
            HirTerminator::Return {
                values: vec![return_err],
            };

        // CONTINUE BLOCK: This block continues normal execution
        // Set continuation_block so that subsequent statements in the same
        // logical block are added to continue_block instead of the original block.
        // This enables chained try expressions like:
        //   const a = try get_a();
        //   const b = try get_b();
        // where the second try needs to be in continue_block from the first.
        self.continuation_block = Some(continue_block_id);

        // Since ok_value was extracted in block_id (before the branch),
        // we return ok_value directly. This works because:
        // 1. If control flows to continue_block (is_ok == true), ok_value is valid
        // 2. If control flows to err_block, we early return, so ok_value isn't used

        // Return the ok_value which was extracted in block_id
        Ok(ok_value)
    }

    /// Extract T and E from Result<T, E> type
    /// Returns (T, E) if successful, error otherwise
    fn extract_result_type_args(&self, ty: &Type) -> CompilerResult<(Type, Type)> {
        match ty {
            // Direct Result type (from Zig's !T error union)
            Type::Result { ok_type, err_type } => {
                Ok((ok_type.as_ref().clone(), err_type.as_ref().clone()))
            }
            Type::Named { type_args, .. } => {
                // For now, we assume any Named type with 2 type arguments could be Result
                // Type checker should ensure only Result<T, E> can use ? operator
                // TODO: Add explicit check for Result TypeId when TypeRegistry API supports it
                if type_args.len() == 2 {
                    return Ok((type_args[0].clone(), type_args[1].clone()));
                }

                Err(crate::CompilerError::Lowering(format!(
                    "? operator requires Result<T, E> type with 2 type arguments, found {} type arguments",
                    type_args.len()
                )))
            }
            _ => Err(crate::CompilerError::Lowering(
                "? operator requires Result<T, E> type, found non-generic type".to_string()
            )),
        }
    }

    /// Test if a pattern matches a scrutinee value
    /// Returns a boolean HIR value indicating match success
    fn translate_pattern_test(
        &mut self,
        block_id: HirId,
        pattern: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedPattern>,
        scrutinee_val: HirId,
        scrutinee_ty: &Type,
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::typed_ast::{TypedPattern, TypedLiteralPattern};

        log::debug!("[SSA] translate_pattern_test: pattern={:?}", pattern.node);

        match &pattern.node {
            // Wildcard always matches
            TypedPattern::Wildcard => {
                let true_val = self.create_value(
                    HirType::Bool,
                    HirValueKind::Constant(crate::hir::HirConstant::Bool(true))
                );
                Ok(true_val)
            }

            // Variable binding always matches (and binds the value)
            TypedPattern::Identifier { name, mutability: _ } => {
                // Bind the scrutinee to this variable name
                self.write_variable(*name, block_id, scrutinee_val);

                // Pattern always matches
                let true_val = self.create_value(
                    HirType::Bool,
                    HirValueKind::Constant(crate::hir::HirConstant::Bool(true))
                );
                Ok(true_val)
            }

            // Literal pattern: compare scrutinee with literal value
            TypedPattern::Literal(lit_pattern) => {
                self.translate_literal_pattern_test(block_id, lit_pattern, scrutinee_val)
            }

            // Tuple pattern: check each element
            TypedPattern::Tuple(element_patterns) => {
                self.translate_tuple_pattern_test(block_id, element_patterns, scrutinee_val, scrutinee_ty)
            }

            // Struct pattern: check fields
            TypedPattern::Struct { name, fields } => {
                self.translate_struct_pattern_test(block_id, name, fields, scrutinee_val, scrutinee_ty)
            }

            // Enum variant pattern: check tag and fields
            TypedPattern::Enum { name, variant, fields } => {
                self.translate_enum_pattern_test(block_id, name, variant, fields, scrutinee_val, scrutinee_ty)
            }

            // Or pattern: test each sub-pattern and OR results
            TypedPattern::Or(patterns) => {
                let mut result = None;

                for sub_pattern in patterns {
                    let sub_result = self.translate_pattern_test(block_id, sub_pattern, scrutinee_val, scrutinee_ty)?;

                    result = Some(if let Some(prev_result) = result {
                        // OR: prev_result || sub_result
                        let or_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
                        let or_inst = HirInstruction::Binary {
                            op: crate::hir::BinaryOp::Or,
                            result: or_result,
                            ty: HirType::Bool,
                            left: prev_result,
                            right: sub_result,
                        };

                        self.add_instruction(block_id, or_inst);
                        self.add_use(prev_result, or_result);
                        self.add_use(sub_result, or_result);

                        or_result
                    } else {
                        sub_result
                    });
                }

                result.ok_or_else(|| crate::CompilerError::Analysis(
                    "Or pattern must have at least one sub-pattern".to_string()
                ))
            }

            // Array pattern: check each element (similar to tuple)
            TypedPattern::Array(element_patterns) => {
                use zyntax_typed_ast::Type as FrontendType;

                // Extract array element type
                let element_type = match scrutinee_ty {
                    FrontendType::Array { element_type, .. } => element_type,
                    _ => return Err(crate::CompilerError::Analysis(
                        format!("Expected array type for array pattern, got {:?}", scrutinee_ty)
                    )),
                };

                // Test each element
                let mut result = None;

                for (idx, pattern) in element_patterns.iter().enumerate() {
                    // Extract element from array using GetElementPtr
                    let elem_ptr = self.create_value(
                        HirType::Ptr(Box::new(self.convert_type(element_type))),
                        HirValueKind::Instruction
                    );

                    let gep_inst = HirInstruction::GetElementPtr {
                        result: elem_ptr,
                        ptr: scrutinee_val,
                        indices: vec![self.create_value(
                            HirType::I64,
                            HirValueKind::Constant(crate::hir::HirConstant::I64(idx as i64))
                        )],
                        ty: self.convert_type(element_type),
                    };

                    self.add_instruction(block_id, gep_inst);
                    self.add_use(scrutinee_val, elem_ptr);

                    // Load element value
                    let elem_val = self.create_value(self.convert_type(element_type), HirValueKind::Instruction);
                    let load_inst = HirInstruction::Load {
                        result: elem_val,
                        ptr: elem_ptr,
                        ty: self.convert_type(element_type),
                        align: 8,
                        volatile: false,
                    };

                    self.add_instruction(block_id, load_inst);
                    self.add_use(elem_ptr, elem_val);

                    // Test pattern against element
                    let elem_test = self.translate_pattern_test(block_id, pattern, elem_val, element_type)?;

                    // AND with previous results
                    result = Some(if let Some(prev_result) = result {
                        let and_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
                        let and_inst = HirInstruction::Binary {
                            op: crate::hir::BinaryOp::And,
                            result: and_result,
                            ty: HirType::Bool,
                            left: prev_result,
                            right: elem_test,
                        };

                        self.add_instruction(block_id, and_inst);
                        self.add_use(prev_result, and_result);
                        self.add_use(elem_test, and_result);

                        and_result
                    } else {
                        elem_test
                    });
                }

                result.ok_or_else(|| crate::CompilerError::Analysis(
                    "Array pattern must have at least one element".to_string()
                ))
            }

            // At pattern: binding @ pattern (e.g., x @ Some(y))
            TypedPattern::At { name, mutability: _, pattern: inner_pattern } => {
                // Bind the scrutinee to the name
                self.write_variable(*name, block_id, scrutinee_val);

                // Then test the inner pattern
                self.translate_pattern_test(block_id, inner_pattern, scrutinee_val, scrutinee_ty)
            }

            // Range pattern: test if value is in range
            TypedPattern::Range { start, end, inclusive } => {
                // Test: start <= scrutinee && scrutinee <= end (if inclusive)
                //       start <= scrutinee && scrutinee < end (if exclusive)

                // Get start literal value
                let start_val = self.translate_literal_pattern_to_value(start)?;

                // Test: start <= scrutinee
                let ge_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
                let ge_inst = HirInstruction::Binary {
                    op: crate::hir::BinaryOp::Ge,
                    result: ge_result,
                    ty: HirType::Bool,
                    left: scrutinee_val,
                    right: start_val,
                };

                self.add_instruction(block_id, ge_inst);
                self.add_use(scrutinee_val, ge_result);
                self.add_use(start_val, ge_result);

                // Get end literal value
                let end_val = self.translate_literal_pattern_to_value(end)?;

                // Test: scrutinee <= end (or < end if not inclusive)
                let le_or_lt_op = if *inclusive {
                    crate::hir::BinaryOp::Le
                } else {
                    crate::hir::BinaryOp::Lt
                };

                let le_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
                let le_inst = HirInstruction::Binary {
                    op: le_or_lt_op,
                    result: le_result,
                    ty: HirType::Bool,
                    left: scrutinee_val,
                    right: end_val,
                };

                self.add_instruction(block_id, le_inst);
                self.add_use(scrutinee_val, le_result);
                self.add_use(end_val, le_result);

                // AND both conditions
                let final_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
                let and_inst = HirInstruction::Binary {
                    op: crate::hir::BinaryOp::And,
                    result: final_result,
                    ty: HirType::Bool,
                    left: ge_result,
                    right: le_result,
                };

                self.add_instruction(block_id, and_inst);
                self.add_use(ge_result, final_result);
                self.add_use(le_result, final_result);

                Ok(final_result)
            }

            // Reference pattern: dereference and test inner pattern
            TypedPattern::Reference { pattern: inner_pattern, mutability: _ } => {
                // Dereference the scrutinee
                let deref_ty = self.convert_type(&inner_pattern.ty);
                let deref_val = self.create_value(deref_ty.clone(), HirValueKind::Instruction);

                let load_inst = HirInstruction::Load {
                    result: deref_val,
                    ptr: scrutinee_val,
                    ty: deref_ty,
                    align: 8,
                    volatile: false,
                };

                self.add_instruction(block_id, load_inst);
                self.add_use(scrutinee_val, deref_val);

                // Test inner pattern on dereferenced value
                self.translate_pattern_test(block_id, inner_pattern, deref_val, &inner_pattern.ty)
            }

            // TODO: Implement remaining pattern types (Slice, Box, Guard, Rest, etc.)
            _ => {
                // For unimplemented patterns, return true (always match)
                // TODO: Add proper error reporting
                let true_val = self.create_value(
                    HirType::Bool,
                    HirValueKind::Constant(crate::hir::HirConstant::Bool(true))
                );
                Ok(true_val)
            }
        }
    }

    /// Convert a literal pattern to a HIR value (for range patterns)
    fn translate_literal_pattern_to_value(
        &mut self,
        lit_pattern: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedLiteralPattern>,
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::typed_ast::TypedLiteralPattern;

        let val = match &lit_pattern.node {
            TypedLiteralPattern::Integer(i) => {
                self.create_value(
                    HirType::I64,
                    HirValueKind::Constant(crate::hir::HirConstant::I64(*i as i64))
                )
            }
            TypedLiteralPattern::Float(f) => {
                self.create_value(
                    HirType::F64,
                    HirValueKind::Constant(crate::hir::HirConstant::F64(*f))
                )
            }
            TypedLiteralPattern::Char(c) => {
                self.create_value(
                    HirType::I32,
                    HirValueKind::Constant(crate::hir::HirConstant::I32(*c as i32))
                )
            }
            _ => {
                return Err(crate::CompilerError::Analysis(
                    "Only integer, float, and char literals supported in range patterns".to_string()
                ));
            }
        };

        Ok(val)
    }

    /// Test literal pattern match
    fn translate_literal_pattern_test(
        &mut self,
        block_id: HirId,
        lit_pattern: &zyntax_typed_ast::typed_ast::TypedLiteralPattern,
        scrutinee_val: HirId,
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::typed_ast::TypedLiteralPattern;

        // Create HIR constant for the literal
        let lit_val = match lit_pattern {
            TypedLiteralPattern::Integer(i) => {
                self.create_value(
                    HirType::I64,
                    HirValueKind::Constant(crate::hir::HirConstant::I64(*i as i64))
                )
            }
            TypedLiteralPattern::Float(f) => {
                self.create_value(
                    HirType::F64,
                    HirValueKind::Constant(crate::hir::HirConstant::F64(*f))
                )
            }
            TypedLiteralPattern::Bool(b) => {
                self.create_value(
                    HirType::Bool,
                    HirValueKind::Constant(crate::hir::HirConstant::Bool(*b))
                )
            }
            TypedLiteralPattern::String(_s) => {
                // TODO: String comparison requires runtime support
                // For now, always return true
                return Ok(self.create_value(
                    HirType::Bool,
                    HirValueKind::Constant(crate::hir::HirConstant::Bool(true))
                ));
            }
            TypedLiteralPattern::Char(c) => {
                self.create_value(
                    HirType::I32, // Chars are i32
                    HirValueKind::Constant(crate::hir::HirConstant::I32(*c as i32))
                )
            }
            TypedLiteralPattern::Byte(b) => {
                self.create_value(
                    HirType::I8,
                    HirValueKind::Constant(crate::hir::HirConstant::I8(*b as i8))
                )
            }
            TypedLiteralPattern::ByteString(_bs) => {
                // TODO: ByteString comparison requires runtime support
                // For now, always return true
                return Ok(self.create_value(
                    HirType::Bool,
                    HirValueKind::Constant(crate::hir::HirConstant::Bool(true))
                ));
            }
            TypedLiteralPattern::Unit => {
                // Unit always matches
                return Ok(self.create_value(
                    HirType::Bool,
                    HirValueKind::Constant(crate::hir::HirConstant::Bool(true))
                ));
            }
            TypedLiteralPattern::Null => {
                // Null always matches null values
                return Ok(self.create_value(
                    HirType::Bool,
                    HirValueKind::Constant(crate::hir::HirConstant::Bool(true))
                ));
            }
        };

        // Generate equality comparison
        let cmp_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
        let cmp_inst = HirInstruction::Binary {
            op: crate::hir::BinaryOp::Eq,
            result: cmp_result,
            ty: HirType::Bool,
            left: scrutinee_val,
            right: lit_val,
        };

        self.add_instruction(block_id, cmp_inst);
        self.add_use(scrutinee_val, cmp_result);
        self.add_use(lit_val, cmp_result);

        Ok(cmp_result)
    }

    /// Test tuple pattern match
    fn translate_tuple_pattern_test(
        &mut self,
        block_id: HirId,
        element_patterns: &[zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedPattern>],
        scrutinee_val: HirId,
        scrutinee_ty: &Type,
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::Type as FrontendType;

        // Extract tuple element types
        let element_types = match scrutinee_ty {
            FrontendType::Tuple(types) => types,
            _ => return Err(crate::CompilerError::Analysis(
                format!("Expected tuple type for tuple pattern, got {:?}", scrutinee_ty)
            )),
        };

        if element_patterns.len() != element_types.len() {
            return Err(crate::CompilerError::Analysis(
                format!("Tuple pattern has {} elements but type has {}",
                    element_patterns.len(), element_types.len())
            ));
        }

        // Test each element
        let mut result = None;

        for (idx, (pattern, elem_ty)) in element_patterns.iter().zip(element_types.iter()).enumerate() {
            // Extract element from tuple
            let elem_val = self.create_value(self.convert_type(elem_ty), HirValueKind::Instruction);
            let extract_inst = HirInstruction::ExtractValue {
                result: elem_val,
                aggregate: scrutinee_val,
                indices: vec![idx as u32],
                ty: self.convert_type(elem_ty),
            };

            self.add_instruction(block_id, extract_inst);
            self.add_use(scrutinee_val, elem_val);

            // Test pattern against element
            let elem_test = self.translate_pattern_test(block_id, pattern, elem_val, elem_ty)?;

            // AND with previous results
            result = Some(if let Some(prev_result) = result {
                let and_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
                let and_inst = HirInstruction::Binary {
                    op: crate::hir::BinaryOp::And,
                    result: and_result,
                    ty: HirType::Bool,
                    left: prev_result,
                    right: elem_test,
                };

                self.add_instruction(block_id, and_inst);
                self.add_use(prev_result, and_result);
                self.add_use(elem_test, and_result);

                and_result
            } else {
                elem_test
            });
        }

        result.ok_or_else(|| crate::CompilerError::Analysis(
            "Tuple pattern must have at least one element".to_string()
        ))
    }

    /// Test struct pattern match
    ///
    /// Extracts each field from the struct and recursively tests the field pattern.
    /// All field tests must pass (AND logic).
    fn translate_struct_pattern_test(
        &mut self,
        block_id: HirId,
        struct_name: &InternedString,
        field_patterns: &[zyntax_typed_ast::typed_ast::TypedFieldPattern],
        scrutinee_val: HirId,
        scrutinee_ty: &Type,
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::Type as FrontendType;

        // If no fields to match, always succeeds
        if field_patterns.is_empty() {
            let true_val = self.create_value(
                HirType::Bool,
                HirValueKind::Constant(crate::hir::HirConstant::Bool(true))
            );
            return Ok(true_val);
        }

        // Validate scrutinee is a struct type
        match scrutinee_ty {
            FrontendType::Struct { .. } | FrontendType::Named { .. } => {
                // Valid struct types
            }
            _ => return Err(crate::CompilerError::Analysis(
                format!("Expected struct type for struct pattern, got {:?}", scrutinee_ty)
            )),
        };

        // Test each field pattern
        let mut result = None;

        for field_pattern in field_patterns {
            // Get field index from TypeRegistry
            let field_index = self.get_field_index(scrutinee_ty, &field_pattern.name)?;

            // Extract field value from struct
            let field_ty = self.convert_type(&field_pattern.pattern.ty);
            let field_val = self.create_value(field_ty.clone(), HirValueKind::Instruction);

            let extract_inst = HirInstruction::ExtractValue {
                result: field_val,
                aggregate: scrutinee_val,
                indices: vec![field_index],
                ty: field_ty,
            };

            self.add_instruction(block_id, extract_inst);
            self.add_use(scrutinee_val, field_val);

            // Recursively test the field pattern
            let field_test = self.translate_pattern_test(
                block_id,
                &field_pattern.pattern,
                field_val,
                &field_pattern.pattern.ty
            )?;

            // AND with previous results
            result = Some(if let Some(prev_result) = result {
                let and_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
                let and_inst = HirInstruction::Binary {
                    op: crate::hir::BinaryOp::And,
                    result: and_result,
                    ty: HirType::Bool,
                    left: prev_result,
                    right: field_test,
                };

                self.add_instruction(block_id, and_inst);
                self.add_use(prev_result, and_result);
                self.add_use(field_test, and_result);

                and_result
            } else {
                field_test
            });
        }

        result.ok_or_else(|| crate::CompilerError::Analysis(
            "Struct pattern must have at least one field".to_string()
        ))
    }

    /// Test enum pattern match
    ///
    /// Checks if the enum discriminant matches the variant tag, then extracts
    /// and tests the payload fields if the tag matches.
    fn translate_enum_pattern_test(
        &mut self,
        block_id: HirId,
        enum_name: &InternedString,
        variant_name: &InternedString,
        field_patterns: &[zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedPattern>],
        scrutinee_val: HirId,
        scrutinee_ty: &Type,
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::Type as FrontendType;

        // Check if the scrutinee is actually an enum/named type that supports enum patterns
        // If the scrutinee is a primitive type, enum pattern matching always fails
        let is_enum_type = matches!(scrutinee_ty,
            FrontendType::Named { .. } |
            FrontendType::Union(_) |
            FrontendType::Optional(_) |
            FrontendType::Result { .. }
        );

        if !is_enum_type {
            // For non-enum types (like integers), enum patterns always fail to match
            // Return a constant false value
            let false_val = self.create_value(
                HirType::Bool,
                HirValueKind::Constant(crate::hir::HirConstant::Bool(false))
            );
            return Ok(false_val);
        }

        // Step 1: Extract discriminant (tag) from enum
        // Enums are typically represented as tagged unions: { tag: u32, payload: union { ... } }
        // The tag is usually at index 0

        let tag_val = self.create_value(HirType::U32, HirValueKind::Instruction);
        let extract_tag_inst = HirInstruction::ExtractValue {
            result: tag_val,
            aggregate: scrutinee_val,
            indices: vec![0], // Tag is at index 0
            ty: HirType::U32,
        };

        self.add_instruction(block_id, extract_tag_inst);
        self.add_use(scrutinee_val, tag_val);

        // Step 2: Get expected discriminant value for this variant
        // TODO: Look up variant discriminant from TypeRegistry
        // For now, use a placeholder value based on variant name hash
        let expected_discriminant = variant_name.to_string().len() as u32; // Placeholder

        let expected_val = self.create_value(
            HirType::U32,
            HirValueKind::Constant(crate::hir::HirConstant::U32(expected_discriminant))
        );

        // Step 3: Compare tag with expected discriminant
        let tag_matches = self.create_value(HirType::Bool, HirValueKind::Instruction);
        let tag_cmp_inst = HirInstruction::Binary {
            op: crate::hir::BinaryOp::Eq,
            result: tag_matches,
            ty: HirType::Bool,
            left: tag_val,
            right: expected_val,
        };

        self.add_instruction(block_id, tag_cmp_inst);
        self.add_use(tag_val, tag_matches);
        self.add_use(expected_val, tag_matches);

        // Step 4: If no payload fields, return tag match result
        if field_patterns.is_empty() {
            return Ok(tag_matches);
        }

        // Step 5: Extract payload from enum (at index 1)
        // The payload is a union, we need to extract the specific variant's data
        let payload_ty = HirType::Void; // TODO: Get actual payload type
        let payload_val = self.create_value(payload_ty.clone(), HirValueKind::Instruction);

        let extract_payload_inst = HirInstruction::ExtractValue {
            result: payload_val,
            aggregate: scrutinee_val,
            indices: vec![1], // Payload is at index 1
            ty: payload_ty,
        };

        self.add_instruction(block_id, extract_payload_inst);
        self.add_use(scrutinee_val, payload_val);

        // Step 6: Test each field pattern against payload fields
        let mut field_result = None;

        for (idx, field_pattern) in field_patterns.iter().enumerate() {
            // Extract field from payload
            let field_ty = self.convert_type(&field_pattern.ty);
            let field_val = self.create_value(field_ty.clone(), HirValueKind::Instruction);

            let extract_field_inst = HirInstruction::ExtractValue {
                result: field_val,
                aggregate: payload_val,
                indices: vec![idx as u32],
                ty: field_ty,
            };

            self.add_instruction(block_id, extract_field_inst);
            self.add_use(payload_val, field_val);

            // Test pattern against field
            let field_test = self.translate_pattern_test(
                block_id,
                field_pattern,
                field_val,
                &field_pattern.ty
            )?;

            // AND with previous field results
            field_result = Some(if let Some(prev_result) = field_result {
                let and_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
                let and_inst = HirInstruction::Binary {
                    op: crate::hir::BinaryOp::And,
                    result: and_result,
                    ty: HirType::Bool,
                    left: prev_result,
                    right: field_test,
                };

                self.add_instruction(block_id, and_inst);
                self.add_use(prev_result, and_result);
                self.add_use(field_test, and_result);

                and_result
            } else {
                field_test
            });
        }

        // Step 7: AND tag match with field tests
        if let Some(fields_match) = field_result {
            let final_result = self.create_value(HirType::Bool, HirValueKind::Instruction);
            let final_and_inst = HirInstruction::Binary {
                op: crate::hir::BinaryOp::And,
                result: final_result,
                ty: HirType::Bool,
                left: tag_matches,
                right: fields_match,
            };

            self.add_instruction(block_id, final_and_inst);
            self.add_use(tag_matches, final_result);
            self.add_use(fields_match, final_result);

            Ok(final_result)
        } else {
            // No fields, just return tag match
            Ok(tag_matches)
        }
    }

    // ========== CLOSURE LOWERING ==========

    /// Translate a lambda/closure expression to HIR
    ///
    /// Current implementation: Basic infrastructure for closures
    /// - Detects captures vs. no captures
    /// - Creates environment struct for capturing closures
    /// - Allocates and initializes environment with captured values
    ///
    /// TODO (Gap 6 Phase 2 - requires architectural changes):
    /// - Generate separate HIR function for lambda body
    /// - Create thunk function with environment parameter
    /// - Return proper closure object {function_ptr, environment}
    /// - Implement indirect call for closure invocation
    ///
    /// Estimated remaining effort: 15-20 hours
    /// Requires: Separate SSA builder context for nested functions
    fn translate_closure(
        &mut self,
        block_id: HirId,
        lambda: &zyntax_typed_ast::typed_ast::TypedLambda,
        closure_ty: &Type,
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::typed_ast::TypedLambdaBody;

        // Build function signature from lambda type
        let (param_types, return_type) = match closure_ty {
            Type::Function { params, return_type, .. } => {
                let hir_params: Vec<HirType> = params.iter()
                    .map(|p| self.convert_type(&p.ty))
                    .collect();
                let hir_return = self.convert_type(return_type);
                (hir_params, hir_return)
            }
            _ => {
                // Fallback: no params, i32 return
                (vec![], HirType::I32)
            }
        };

        // Generate a unique function ID for the lambda
        let lambda_func_id = HirId::new();
        let lambda_name = {
            let mut arena = self.arena.lock().unwrap();
            // Use Debug format since HirId inner field is private
            arena.intern_string(&format!("__lambda_{:?}", lambda_func_id).replace("-", "").chars().take(20).collect::<String>())
        };

        // Create HIR params for signature
        let hir_params: Vec<crate::hir::HirParam> = param_types.iter().enumerate().map(|(idx, ty)| {
            let param_name = {
                let mut arena = self.arena.lock().unwrap();
                arena.intern_string(&format!("arg{}", idx))
            };
            crate::hir::HirParam {
                id: HirId::new(),
                name: param_name,
                ty: ty.clone(),
                attributes: crate::hir::ParamAttributes::default(),
            }
        }).collect();

        // Create the lambda function signature
        let signature = crate::hir::HirFunctionSignature {
            params: hir_params.clone(),
            returns: vec![return_type.clone()],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };

        // Create entry block for lambda
        let entry_block_id = HirId::new();
        let mut entry_block = crate::hir::HirBlock {
            id: entry_block_id,
            label: None,
            phis: vec![],
            instructions: Vec::new(),
            terminator: crate::hir::HirTerminator::Unreachable,
            dominance_frontier: HashSet::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        };

        // Create parameter values
        let param_values: Vec<(HirId, HirType)> = hir_params.iter()
            .map(|p| (p.id, p.ty.clone()))
            .collect();

        // Create the lambda HirFunction
        let mut lambda_func = HirFunction {
            id: lambda_func_id,
            name: lambda_name,
            signature,
            entry_block: entry_block_id,
            blocks: IndexMap::new(),
            locals: IndexMap::new(),
            values: IndexMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: crate::hir::CallingConvention::Fast,
            attributes: crate::hir::FunctionAttributes::default(),
            link_name: None,
        };

        // Add parameter values to function
        for (idx, (param_id, param_ty)) in param_values.iter().enumerate() {
            lambda_func.values.insert(*param_id, crate::hir::HirValue {
                id: *param_id,
                ty: param_ty.clone(),
                kind: crate::hir::HirValueKind::Parameter(idx as u32),
                uses: HashSet::new(),
                span: None,
            });
        }

        // Collect outer scope variables for capture support
        // Get all variables defined in the current block that can be captured
        let outer_captures: IndexMap<InternedString, HirId> = self.definitions
            .get(&block_id)
            .cloned()
            .unwrap_or_default();

        // Translate lambda body
        let result_val = match &lambda.body {
            TypedLambdaBody::Expression(expr) => {
                self.translate_lambda_expr(&mut lambda_func, &mut entry_block, &param_values, &lambda.params, &outer_captures, expr, &return_type)?
            }
            TypedLambdaBody::Block(_block) => {
                // Block body - for now, just return 0
                let val_id = HirId::new();
                lambda_func.values.insert(val_id, crate::hir::HirValue {
                    id: val_id,
                    ty: return_type.clone(),
                    kind: crate::hir::HirValueKind::Constant(crate::hir::HirConstant::I32(0)),
                    uses: HashSet::new(),
                    span: None,
                });
                val_id
            }
        };

        // Set return terminator
        entry_block.terminator = crate::hir::HirTerminator::Return {
            values: vec![result_val],
        };

        // Add entry block to function
        lambda_func.blocks.insert(entry_block_id, entry_block);

        // Store the lambda function for later compilation
        self.closure_functions.push(lambda_func);

        // Create closure type using proper structure
        let func_type = crate::hir::HirFunctionType {
            params: param_types.clone(),
            returns: vec![return_type.clone()],
            lifetime_params: vec![],
            is_variadic: false,
        };

        let closure_ty_hir = HirType::Closure(Box::new(crate::hir::HirClosureType {
            function_type: func_type,
            captures: vec![],
            call_mode: crate::hir::HirClosureCallMode::Fn,
        }));

        let closure_result = self.create_value(closure_ty_hir.clone(), HirValueKind::Instruction);

        // Generate CreateClosure instruction
        let create_closure_inst = HirInstruction::CreateClosure {
            result: closure_result,
            closure_ty: closure_ty_hir,
            function: lambda_func_id, // Use HirId, not name
            captures: vec![],
        };

        self.add_instruction(block_id, create_closure_inst);

        Ok(closure_result)
    }

    /// Translate a lambda expression body
    fn translate_lambda_expr(
        &self,
        func: &mut HirFunction,
        block: &mut crate::hir::HirBlock,
        param_values: &[(HirId, HirType)],
        lambda_params: &[zyntax_typed_ast::typed_ast::TypedLambdaParam],
        outer_captures: &IndexMap<InternedString, HirId>,
        expr: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedExpression>,
        result_ty: &HirType,
    ) -> CompilerResult<HirId> {
        use zyntax_typed_ast::typed_ast::TypedExpression;

        match &expr.node {
            TypedExpression::Literal(lit) => {
                let constant = self.translate_literal(lit);
                let val_id = HirId::new();
                func.values.insert(val_id, crate::hir::HirValue {
                    id: val_id,
                    ty: result_ty.clone(),
                    kind: crate::hir::HirValueKind::Constant(constant),
                    uses: HashSet::new(),
                    span: None,
                });
                Ok(val_id)
            }
            TypedExpression::Variable(name) => {
                // Look up in lambda params first
                for (idx, param) in lambda_params.iter().enumerate() {
                    if param.name == *name {
                        if let Some((id, _)) = param_values.get(idx) {
                            return Ok(*id);
                        }
                    }
                }

                // Look up in captured variables from outer scope
                if let Some(&outer_val_id) = outer_captures.get(name) {
                    // Found in captures - copy the value's constant into the lambda
                    if let Some(outer_value) = self.function.values.get(&outer_val_id) {
                        // Clone the captured value into the lambda function
                        let val_id = HirId::new();
                        func.values.insert(val_id, crate::hir::HirValue {
                            id: val_id,
                            ty: outer_value.ty.clone(),
                            kind: outer_value.kind.clone(),
                            uses: HashSet::new(),
                            span: None,
                        });
                        return Ok(val_id);
                    }
                }

                // Not found - return undef
                let val_id = HirId::new();
                func.values.insert(val_id, crate::hir::HirValue {
                    id: val_id,
                    ty: result_ty.clone(),
                    kind: crate::hir::HirValueKind::Undef,
                    uses: HashSet::new(),
                    span: None,
                });
                Ok(val_id)
            }
            TypedExpression::Binary(bin) => {
                // Translate binary expression
                let left_id = self.translate_lambda_expr(func, block, param_values, lambda_params, outer_captures, &bin.left, result_ty)?;
                let right_id = self.translate_lambda_expr(func, block, param_values, lambda_params, outer_captures, &bin.right, result_ty)?;

                let hir_op = self.convert_binary_op(&bin.op);
                let result_id = HirId::new();
                func.values.insert(result_id, crate::hir::HirValue {
                    id: result_id,
                    ty: result_ty.clone(),
                    kind: crate::hir::HirValueKind::Instruction,
                    uses: HashSet::new(),
                    span: None,
                });

                let binary_inst = HirInstruction::Binary {
                    op: hir_op,
                    result: result_id,
                    ty: result_ty.clone(),
                    left: left_id,
                    right: right_id,
                };
                block.instructions.push(binary_inst);

                Ok(result_id)
            }
            _ => {
                // Fallback - return constant 0
                let val_id = HirId::new();
                func.values.insert(val_id, crate::hir::HirValue {
                    id: val_id,
                    ty: result_ty.clone(),
                    kind: crate::hir::HirValueKind::Constant(crate::hir::HirConstant::I32(0)),
                    uses: HashSet::new(),
                    span: None,
                });
                Ok(val_id)
            }
        }
    }

    /// Create and initialize environment struct for a capturing closure
    ///
    /// This generates HIR instructions to:
    /// 1. Define environment struct type with fields for each capture
    /// 2. Allocate environment struct on stack
    /// 3. Initialize each field with captured variable values
    /// 4. Return pointer to initialized environment
    fn translate_closure_environment(
        &mut self,
        block_id: HirId,
        lambda: &zyntax_typed_ast::typed_ast::TypedLambda,
    ) -> CompilerResult<HirId> {
        // Step 1: Create environment struct type
        let mut env_field_types = Vec::new();
        for capture in &lambda.captures {
            let capture_ty = self.convert_type(&capture.ty);
            let field_ty = if capture.by_ref {
                // Capture by reference: use pointer
                HirType::Ptr(Box::new(capture_ty))
            } else {
                // Capture by value
                capture_ty
            };
            env_field_types.push(field_ty);
        }

        let env_struct_ty = HirType::Struct(crate::hir::HirStructType {
            name: None, // Anonymous environment struct
            fields: env_field_types.clone(),
            packed: false,
        });

        // Step 2: Allocate environment struct
        let env_ptr_result = self.create_value(
            HirType::Ptr(Box::new(env_struct_ty.clone())),
            HirValueKind::Instruction
        );

        let alloca_inst = HirInstruction::Alloca {
            result: env_ptr_result,
            ty: env_struct_ty.clone(),
            count: None,
            align: 8,
        };

        self.add_instruction(block_id, alloca_inst);

        // Step 3: Initialize environment struct with captured values
        for (idx, capture) in lambda.captures.iter().enumerate() {
            // Read captured variable value
            let captured_val = self.read_variable(capture.name, block_id);

            // If capturing by reference, take address
            let value_to_store = if capture.by_ref {
                // TODO: Take address of captured variable
                captured_val
            } else {
                captured_val
            };

            // Store into environment struct field
            // Use InsertValue to update the struct
            let updated_env = self.create_value(env_struct_ty.clone(), HirValueKind::Instruction);
            let insert_inst = HirInstruction::InsertValue {
                result: updated_env,
                aggregate: env_ptr_result,
                value: value_to_store,
                indices: vec![idx as u32],
                ty: env_struct_ty.clone(),
            };

            self.add_instruction(block_id, insert_inst);
            self.add_use(env_ptr_result, updated_env);
            self.add_use(value_to_store, updated_env);
        }

        // Return environment pointer
        Ok(env_ptr_result)
    }
}

impl SsaForm {
    /// Verify SSA properties
    pub fn verify(&self) -> CompilerResult<()> {
        // Each value should be defined exactly once
        let mut defined_values = HashSet::new();
        
        for (_, block) in &self.function.blocks {
            // Check phis
            for phi in &block.phis {
                if !defined_values.insert(phi.result) {
                    return Err(crate::CompilerError::Analysis(
                        format!("Value {:?} defined multiple times", phi.result)
                    ));
                }
            }
            
            // Check instructions
            for inst in &block.instructions {
                if let Some(result) = inst.get_result() {
                    if !defined_values.insert(result) {
                        return Err(crate::CompilerError::Analysis(
                            format!("Value {:?} defined multiple times", result)
                        ));
                    }
                }
            }
        }
        
        // Each use should have a definition (or be a constant/parameter)
        for (use_id, def_id) in &self.use_def_chains {
            if !defined_values.contains(def_id) {
                // Check if it's a constant or parameter (self-defining values)
                if let Some(value) = self.function.values.get(def_id) {
                    match value.kind {
                        HirValueKind::Constant(_) | HirValueKind::Parameter(_) | HirValueKind::Undef | HirValueKind::Global(_) => {
                            // These are self-defining, so it's OK
                            continue;
                        }
                        _ => {}
                    }
                }

                return Err(crate::CompilerError::Analysis(
                    format!("Use {:?} has undefined definition {:?}", use_id, def_id)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Optimize trivial phis
    ///
    /// A trivial phi has only one unique non-self incoming value, meaning all
    /// paths merge with the same value. We can safely replace uses of the phi
    /// result with this single value.
    pub fn optimize_trivial_phis(&mut self) {
        // First pass: identify trivial phis and build replacement map
        let mut replacements: IndexMap<HirId, HirId> = IndexMap::new();

        // Collect all phi results to identify self-referential chains
        let all_phi_results: HashSet<HirId> = self.function.blocks.values()
            .flat_map(|b| b.phis.iter().map(|p| p.result))
            .collect();

        for (_, block) in &self.function.blocks {
            for phi in &block.phis {
                // Check if phi is trivial: all non-self values must be the same
                // A phi referencing itself or other phis doesn't make it trivial
                let non_self_values: HashSet<_> = phi.incoming.iter()
                    .map(|(val, _)| *val)
                    .filter(|&v| v != phi.result)
                    .collect();

                // Check if this phi has a self-reference (indicates a loop header phi)
                let has_self_reference = phi.incoming.iter().any(|(val, _)| *val == phi.result);

                // Only trivial if:
                // 1. There's exactly one non-self value
                // 2. That value is NOT another phi (otherwise it might be updated in a loop)
                // 3. The phi does NOT have a self-reference (loop header phis are not trivial!)
                if non_self_values.len() == 1 && !has_self_reference {
                    let replacement = *non_self_values.iter().next().unwrap();
                    // Don't remove phis that reference other phis - they might form loop-carried dependencies
                    if !all_phi_results.contains(&replacement) {
                        replacements.insert(phi.result, replacement);
                    }
                }
            }
        }

        if replacements.is_empty() {
            return;
        }

        // Second pass: replace uses in all instructions
        for (_, block) in &mut self.function.blocks {
            for inst in &mut block.instructions {
                inst.replace_uses(&replacements);
            }

            // Replace uses in terminators
            block.terminator.replace_uses(&replacements);

            // Replace uses in other phi nodes' incoming values
            for phi in &mut block.phis {
                for (val, _) in &mut phi.incoming {
                    if let Some(&replacement) = replacements.get(val) {
                        *val = replacement;
                    }
                }
            }
        }

        // Third pass: remove trivial phis
        for (_, block) in &mut self.function.blocks {
            block.phis.retain(|phi| !replacements.contains_key(&phi.result));
        }
    }
}

impl HirInstruction {
    /// Get the result value of this instruction if any
    fn get_result(&self) -> Option<HirId> {
        match self {
            HirInstruction::Binary { result, .. } |
            HirInstruction::Unary { result, .. } |
            HirInstruction::Alloca { result, .. } |
            HirInstruction::Load { result, .. } |
            HirInstruction::GetElementPtr { result, .. } |
            HirInstruction::Cast { result, .. } |
            HirInstruction::Select { result, .. } |
            HirInstruction::ExtractValue { result, .. } |
            HirInstruction::InsertValue { result, .. } |
            HirInstruction::Atomic { result, .. } |
            HirInstruction::CreateUnion { result, .. } |
            HirInstruction::GetUnionDiscriminant { result, .. } |
            HirInstruction::ExtractUnionValue { result, .. } |
            HirInstruction::CreateClosure { result, .. } |
            HirInstruction::CreateTraitObject { result, .. } |
            HirInstruction::UpcastTraitObject { result, .. } => Some(*result),

            HirInstruction::Call { result, .. } |
            HirInstruction::IndirectCall { result, .. } |
            HirInstruction::CallClosure { result, .. } |
            HirInstruction::TraitMethodCall { result, .. } => *result,
            
            HirInstruction::CreateRef { result, .. } |
            HirInstruction::Deref { result, .. } |
            HirInstruction::Move { result, .. } |
            HirInstruction::Copy { result, .. } => Some(*result),
            
            HirInstruction::Store { .. } |
            HirInstruction::Fence { .. } |
            HirInstruction::BeginLifetime { .. } |
            HirInstruction::EndLifetime { .. } |
            HirInstruction::LifetimeConstraint { .. } |
            HirInstruction::Resume { .. } |
            HirInstruction::AbortEffect { .. } => None,

            // Effect instructions with optional result
            HirInstruction::PerformEffect { result, .. } |
            HirInstruction::HandleEffect { result, .. } => *result,

            // CaptureContinuation always has a result
            HirInstruction::CaptureContinuation { result, .. } => Some(*result),
        }
    }
}