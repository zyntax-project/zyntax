//! # Typed Control Flow Graph Builder
//!
//! This module creates CFG structures from TypedAST without converting to HIR first.
//! This breaks the circular dependency between CFG and SSA construction:
//! - TypedCfgBuilder creates CFG structure from TypedAST (control flow only)
//! - SsaBuilder then processes TypedStatements to emit HIR instructions
//!
//! This is the solution to Gap #4 (CFG Construction) described in INTEGRATION_GAPS_ANALYSIS.md

use crate::CompilerResult;
use crate::hir::HirId;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use zyntax_typed_ast::{
    InternedString, Span, Type,
    typed_ast::{
        TypedBlock, TypedExpression, TypedMatchArm, TypedNode, TypedPattern, TypedStatement,
        typed_node,
    },
};

/// A `with H { body }` scope discovered during CFG construction.
/// Consumed by `LoweringContext` after SSA to emit
/// `__zyntax_effect_push_handler` at `entry` and
/// `__zyntax_effect_pop_handler` on every edge that leaves the scope.
#[derive(Debug, Clone)]
pub struct WithScopeInfo {
    /// The handler name (`with H`) — resolved to a `HirEffectHandler`
    /// at the post-SSA lowering pass.
    pub handler_name: InternedString,
    /// Block where the scope begins (push the handler here). This is
    /// also the entry of the body's block subgraph.
    pub entry: HirId,
    /// Block the scope falls through to on normal exit.
    pub after: HirId,
    /// Every block belonging to the scope body (entry included). Used
    /// to classify which terminator edges leave the scope.
    pub body_blocks: Vec<HirId>,
}

/// Builder for creating CFG from TypedAST
pub struct TypedCfgBuilder {
    /// Next block ID to allocate
    next_block_id: u32,
    /// Stack of loop contexts for Break/Continue handling
    /// Each entry is (header_id, after_id) for the loop
    loop_stack: Vec<(HirId, HirId)>,
    /// The block each label of the function starts, made when the
    /// label or a `goto` to it is first met.
    labels: HashMap<InternedString, HirId>,
    /// `with` scopes discovered during construction, in source order.
    pub with_scopes: Vec<WithScopeInfo>,
    /// Distinguishes this builder's hidden locals from those of every
    /// other builder run over the same function.
    name_scope: String,
    /// The innermost loop around each block made, as (continue target,
    /// exit). A block outside every loop has no entry.
    pub loop_of: HashMap<HirId, (HirId, HirId)>,
}

/// Control flow graph with TypedAST statements (not yet converted to HIR)
pub struct TypedControlFlowGraph {
    /// Graph structure: nodes are basic blocks, edges are control flow
    pub graph: DiGraph<TypedBasicBlock, ()>,
    /// Entry block node
    pub entry: NodeIndex,
    /// Exit block node
    pub exit: NodeIndex,
    /// Map from HirId to graph node
    pub block_map: HashMap<HirId, NodeIndex>,
    /// Map from graph node to HirId
    pub node_map: HashMap<NodeIndex, HirId>,
    /// The innermost loop around each block, as (continue target, exit).
    pub loop_of: HashMap<HirId, (HirId, HirId)>,
}

/// A basic block containing TypedAST statements
#[derive(Debug, Clone)]
pub struct TypedBasicBlock {
    /// Unique ID for this block
    pub id: HirId,
    /// Label for this block (optional)
    pub label: Option<InternedString>,
    /// Statements in this block
    pub statements: Vec<TypedNode<TypedStatement>>,
    /// How control flow exits this block
    pub terminator: TypedTerminator,
    /// Pattern check metadata (for pattern matching blocks)
    pub pattern_check: Option<PatternCheckInfo>,
}

/// Pattern check information for a basic block
#[derive(Debug, Clone)]
pub struct PatternCheckInfo {
    /// The scrutinee expression being matched
    pub scrutinee: TypedNode<TypedExpression>,
    /// The pattern being checked in this block
    pub pattern: TypedNode<TypedPattern>,
    /// Variant index if this is a union variant check
    pub variant_index: Option<u32>,
    /// Target block if pattern check fails (for match arms)
    pub false_target: Option<HirId>,
}

/// Control flow terminator for TypedBasicBlock
#[derive(Debug, Clone)]
pub enum TypedTerminator {
    /// Return from function
    Return(Option<Box<TypedNode<TypedExpression>>>),
    /// Unconditional jump to block
    Jump(HirId),
    /// Conditional branch
    CondBranch {
        condition: Box<TypedNode<TypedExpression>>,
        true_target: HirId,
        false_target: HirId,
    },
    /// Unreachable code
    Unreachable,
}

impl TypedCfgBuilder {
    pub fn new() -> Self {
        Self {
            next_block_id: 0,
            loop_stack: Vec::new(),
            labels: HashMap::new(),
            with_scopes: Vec::new(),
            name_scope: String::new(),
            loop_of: HashMap::new(),
        }
    }

    /// A builder for statements nested in an expression of a function
    /// another builder already split. `scope` must differ between the
    /// builders run over one function, so their hidden locals do not
    /// share names. `outer` is the loop around the expression, as
    /// (continue target, exit): a `break` or `continue` among the
    /// statements that no loop of their own encloses acts on it.
    pub fn nested(scope: &str, outer: Option<(HirId, HirId)>) -> Self {
        Self {
            name_scope: format!("{scope}_"),
            loop_stack: outer.into_iter().collect(),
            ..Self::new()
        }
    }

    /// Generate a new unique block ID
    fn new_block_id(&mut self) -> HirId {
        let id = HirId::new();
        self.next_block_id += 1;
        id
    }

    /// The block a label starts, the same for every mention of it.
    fn label_block(&mut self, name: InternedString) -> HirId {
        if let Some(&id) = self.labels.get(&name) {
            return id;
        }
        let id = self.new_block_id();
        self.labels.insert(name, id);
        id
    }

    /// A local the builder introduces for itself, named so no source
    /// program can spell it and distinct per loop variable and block.
    fn hidden_name(&mut self, role: &str, loop_var: InternedString) -> InternedString {
        let n = self.next_block_id;
        self.next_block_id += 1;
        InternedString::new_global(&format!(
            "__{role}_{}_{}{n}",
            loop_var.resolve_global().unwrap_or_default(),
            self.name_scope
        ))
    }

    /// Build CFG from a typed block
    /// entry_block_id should be the ID of the entry block from the HirFunction
    pub fn build_from_block(
        &mut self,
        block: &TypedBlock,
        entry_block_id: HirId,
    ) -> CompilerResult<TypedControlFlowGraph> {
        let mut graph = DiGraph::new();
        let mut block_map = HashMap::new();
        let mut node_map = HashMap::new();

        // Use the provided entry block ID (from HirFunction)
        let entry_id = entry_block_id;

        // Process block with control flow splitting (this is a function body)
        let (blocks, entry_id_final, exit_id) =
            self.split_at_control_flow(block, entry_id, true)?;

        // Add all blocks to graph and create mapping
        for typed_block in blocks {
            let block_id = typed_block.id;
            let node = graph.add_node(typed_block);
            block_map.insert(block_id, node);
            node_map.insert(node, block_id);
        }

        // Add edges based on terminators
        // Collect edges first to avoid borrow checker issues
        let edges_to_add: Vec<(NodeIndex, NodeIndex)> = graph
            .node_indices()
            .filter_map(|node| {
                let block = &graph[node];
                match &block.terminator {
                    TypedTerminator::Jump(target) => block_map
                        .get(target)
                        .map(|&target_node| vec![(node, target_node)]),
                    TypedTerminator::CondBranch {
                        true_target,
                        false_target,
                        ..
                    } => {
                        let mut edges = Vec::new();
                        if let Some(&true_node) = block_map.get(true_target) {
                            edges.push((node, true_node));
                        }
                        if let Some(&false_node) = block_map.get(false_target) {
                            edges.push((node, false_node));
                        }
                        if edges.is_empty() { None } else { Some(edges) }
                    }
                    _ => None,
                }
            })
            .flatten()
            .collect();

        // A `break` or `continue` in a block value leaves from inside the
        // block that holds the expression. Its edge is made when the
        // expression is lowered, but phi placement and sealing need it
        // now.
        let mut edges_to_add = edges_to_add;
        for node in graph.node_indices() {
            let block = &graph[node];
            let Some(&(continue_target, exit)) = self.loop_of.get(&block.id) else {
                continue;
            };
            let (breaks, continues) = escaping_jumps(block);
            for (jumps, target) in [(breaks, exit), (continues, continue_target)] {
                if jumps
                    && let Some(&target_node) = block_map.get(&target)
                    && !edges_to_add.contains(&(node, target_node))
                {
                    edges_to_add.push((node, target_node));
                }
            }
        }

        // Add collected edges
        for (source, target) in edges_to_add {
            graph.add_edge(source, target, ());
        }

        let entry_node = block_map[&entry_id_final];
        let exit_node = block_map[&exit_id];

        Ok(TypedControlFlowGraph {
            graph,
            entry: entry_node,
            exit: exit_node,
            block_map,
            node_map,
            loop_of: self.loop_of.clone(),
        })
    }

    /// Process a TypedBlock into a TypedBasicBlock
    fn process_block(
        &mut self,
        block: &TypedBlock,
        block_id: HirId,
    ) -> CompilerResult<(TypedBasicBlock, HirId)> {
        let mut statements = Vec::new();
        let mut terminator = TypedTerminator::Unreachable;

        // Process each statement
        for stmt in &block.statements {
            match &stmt.node {
                TypedStatement::Return(expr) => {
                    // Explicit return terminates the block
                    terminator = TypedTerminator::Return(expr.clone());
                    break; // No more statements after return
                }

                // For all other statements, treat as non-terminating
                _ => {
                    statements.push(stmt.clone());
                }
            }
        }

        Ok((
            TypedBasicBlock {
                id: block_id,
                label: None,
                statements,
                terminator,
                pattern_check: None,
            },
            block_id, // exit_id (same as entry for simple blocks)
        ))
    }

    /// Split a block at control flow boundaries
    /// Returns (all_blocks, entry_block_id, exit_block_id)
    ///
    /// `is_function_body`: if true, treat a single trailing expression as an implicit return
    fn split_at_control_flow(
        &mut self,
        block: &TypedBlock,
        entry_id: HirId,
        is_function_body: bool,
    ) -> CompilerResult<(Vec<TypedBasicBlock>, HirId, HirId)> {
        self.split_statements(&block.statements, entry_id, is_function_body)
    }

    /// [`Self::split_at_control_flow`] over a statement list.
    pub(crate) fn split_statements(
        &mut self,
        statements: &[TypedNode<TypedStatement>],
        entry_id: HirId,
        is_function_body: bool,
    ) -> CompilerResult<(Vec<TypedBasicBlock>, HirId, HirId)> {
        let split = self.split_statements_in_loop(statements, entry_id, is_function_body)?;
        // The loop bodies split inside recorded their own blocks first.
        if let Some(&innermost) = self.loop_stack.last() {
            for block in &split.0 {
                self.loop_of.entry(block.id).or_insert(innermost);
            }
        }
        Ok(split)
    }

    fn split_statements_in_loop(
        &mut self,
        statements: &[TypedNode<TypedStatement>],
        entry_id: HirId,
        is_function_body: bool,
    ) -> CompilerResult<(Vec<TypedBasicBlock>, HirId, HirId)> {
        log::debug!(
            "[CFG] split_at_control_flow: entry_id={:?}, statements={}",
            entry_id,
            statements.len()
        );
        let mut all_blocks = Vec::new();
        let mut current_statements = Vec::new();
        let mut current_block_id = entry_id;
        let mut exit_id = entry_id;

        for (stmt_idx, stmt) in statements.iter().enumerate() {
            log::debug!(
                "[CFG]   stmt[{}]: {:?}, current_block={:?}",
                stmt_idx,
                std::mem::discriminant(&stmt.node),
                current_block_id
            );
            match &stmt.node {
                TypedStatement::If(if_stmt) => {
                    // Create block for statements before If
                    let then_id = self.new_block_id();
                    let else_id = if if_stmt.else_block.is_some() {
                        self.new_block_id()
                    } else {
                        self.new_block_id() // Merge block
                    };
                    let merge_id = self.new_block_id();

                    // Current block ends with conditional branch
                    all_blocks.push(TypedBasicBlock {
                        id: current_block_id,
                        label: None,
                        statements: current_statements.clone(),
                        terminator: TypedTerminator::CondBranch {
                            condition: if_stmt.condition.clone(),
                            true_target: then_id,
                            false_target: else_id,
                        },
                        pattern_check: None,
                    });

                    // Process then block (not a function body)
                    let (then_blocks, _, then_exit) =
                        self.split_at_control_flow(&if_stmt.then_block, then_id, false)?;
                    all_blocks.extend(then_blocks);

                    // Check if then block has a definite terminator (return) BEFORE modifying
                    let then_returns = all_blocks
                        .iter()
                        .rev()
                        .find(|b| b.id == then_exit)
                        .map(|b| matches!(b.terminator, TypedTerminator::Return(_)))
                        .unwrap_or(false);

                    // Make then block jump to merge if it doesn't already have a definite terminator
                    if !then_returns {
                        if let Some(last_block) =
                            all_blocks.iter_mut().rev().find(|b| b.id == then_exit)
                        {
                            if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                                last_block.terminator = TypedTerminator::Jump(merge_id);
                            }
                        }
                    }

                    // Process else block or create empty else
                    let else_returns = if let Some(ref else_block) = if_stmt.else_block {
                        let (else_blocks, _, else_exit) =
                            self.split_at_control_flow(else_block, else_id, false)?;
                        all_blocks.extend(else_blocks);

                        // Check if else block has a definite terminator (return) BEFORE modifying
                        let has_definite_terminator = all_blocks
                            .iter()
                            .rev()
                            .find(|b| b.id == else_exit)
                            .map(|b| matches!(b.terminator, TypedTerminator::Return(_)))
                            .unwrap_or(false);

                        // Make else block jump to merge if it doesn't have a definite terminator
                        if !has_definite_terminator {
                            if let Some(last_block) =
                                all_blocks.iter_mut().rev().find(|b| b.id == else_exit)
                            {
                                if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                                    last_block.terminator = TypedTerminator::Jump(merge_id);
                                }
                            }
                        }

                        has_definite_terminator
                    } else {
                        // Empty else block jumps directly to merge
                        all_blocks.push(TypedBasicBlock {
                            id: else_id,
                            label: None,
                            statements: vec![],
                            terminator: TypedTerminator::Jump(merge_id),
                            pattern_check: None,
                        });
                        false
                    };

                    // Only create merge block if at least one branch can reach it
                    // If both branches return/break/continue, there's no merge point
                    if then_returns && else_returns {
                        // Both branches have definite terminators - no merge block needed
                        // The if statement itself terminates the function/loop
                        exit_id = current_block_id; // Exit at the if block
                        // Don't update current_block_id - we're done
                        // Early return to avoid creating unreachable merge block
                        return Ok((all_blocks, entry_id, exit_id));
                    } else {
                        // Start new block after If (merge point)
                        current_statements = Vec::new();
                        current_block_id = merge_id;
                        exit_id = merge_id;
                    }
                }

                TypedStatement::While(while_stmt) => {
                    log::debug!(
                        "[CFG] While: closing current_block={:?} with {} stmts",
                        current_block_id,
                        current_statements.len()
                    );
                    // Create blocks for while loop
                    let header_id = self.new_block_id();
                    let body_id = self.new_block_id();
                    let after_id = self.new_block_id();
                    log::debug!(
                        "[CFG] While: created header={:?}, body={:?}, after={:?}",
                        header_id,
                        body_id,
                        after_id
                    );

                    // Current block ends with jump to header
                    all_blocks.push(TypedBasicBlock {
                        id: current_block_id,
                        label: None,
                        statements: current_statements.clone(),
                        terminator: TypedTerminator::Jump(header_id),
                        pattern_check: None,
                    });

                    // Header block evaluates condition
                    all_blocks.push(TypedBasicBlock {
                        id: header_id,
                        label: None,
                        statements: vec![],
                        terminator: TypedTerminator::CondBranch {
                            condition: while_stmt.condition.clone(),
                            true_target: body_id,
                            false_target: after_id,
                        },
                        pattern_check: None,
                    });

                    // Push loop context for Break/Continue
                    self.loop_stack.push((header_id, after_id));

                    // Process body block
                    log::debug!("[CFG] While: processing body with entry={:?}", body_id);
                    let (body_blocks, _, body_exit) =
                        self.split_at_control_flow(&while_stmt.body, body_id, false)?;
                    log::debug!(
                        "[CFG] While: body returned {} blocks, body_exit={:?}",
                        body_blocks.len(),
                        body_exit
                    );
                    all_blocks.extend(body_blocks);

                    // Pop loop context
                    self.loop_stack.pop();

                    // Make body block jump back to header
                    if let Some(last_block) =
                        all_blocks.iter_mut().rev().find(|b| b.id == body_exit)
                    {
                        log::debug!(
                            "[CFG] While: body_exit block has terminator: {:?}",
                            last_block.terminator
                        );
                        if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                            log::debug!("[CFG] While: setting body_exit to Jump(header)");
                            last_block.terminator = TypedTerminator::Jump(header_id);
                        }
                    }

                    // Continue with after block for subsequent statements
                    current_statements = Vec::new();
                    current_block_id = after_id;
                    exit_id = after_id;
                    log::debug!(
                        "[CFG] While: continuing with current_block={:?}",
                        current_block_id
                    );
                }

                TypedStatement::Loop(loop_stmt) => {
                    // Infinite loop: loop { body }
                    // Creates: entry → header → body → header
                    //                     ↓
                    //                   exit (for break)

                    use zyntax_typed_ast::typed_ast::TypedLoop;

                    match loop_stmt {
                        TypedLoop::Infinite { body, .. } => {
                            let header_id = self.new_block_id();
                            let body_id = self.new_block_id();
                            let after_id = self.new_block_id();

                            // Current block jumps to header
                            all_blocks.push(TypedBasicBlock {
                                id: current_block_id,
                                label: None,
                                statements: current_statements.clone(),
                                terminator: TypedTerminator::Jump(header_id),
                                pattern_check: None,
                            });

                            // Header block (no condition, always enters body)
                            all_blocks.push(TypedBasicBlock {
                                id: header_id,
                                label: None,
                                statements: vec![],
                                terminator: TypedTerminator::Jump(body_id),
                                pattern_check: None,
                            });

                            // Push loop context for Break/Continue
                            self.loop_stack.push((header_id, after_id));

                            // Process body block
                            let (body_blocks, _, body_exit) =
                                self.split_at_control_flow(body, body_id, false)?;
                            all_blocks.extend(body_blocks);

                            // Pop loop context
                            self.loop_stack.pop();

                            // Make body block jump back to header (unless it has break/return)
                            if let Some(last_block) =
                                all_blocks.iter_mut().rev().find(|b| b.id == body_exit)
                            {
                                if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                                    last_block.terminator = TypedTerminator::Jump(header_id);
                                }
                            }

                            // Continue with after block
                            current_statements = Vec::new();
                            current_block_id = after_id;
                            exit_id = after_id;
                        }
                        TypedLoop::ForEach {
                            pattern,
                            iterator,
                            body,
                        } => {
                            // For-each loop: for item in collection
                            // Similar to While: entry → header → body → header → exit
                            // Header evaluates iterator.next(), body processes item

                            let header_id = self.new_block_id();
                            let body_id = self.new_block_id();
                            let after_id = self.new_block_id();

                            // Current block jumps to header
                            all_blocks.push(TypedBasicBlock {
                                id: current_block_id,
                                label: None,
                                statements: current_statements.clone(),
                                terminator: TypedTerminator::Jump(header_id),
                                pattern_check: None,
                            });

                            // Header block (iterator logic will be handled by SSA builder)
                            // For now, we model it as: if iterator.has_next() then body else exit
                            // The actual iterator protocol will be implemented in SSA/HIR lowering
                            all_blocks.push(TypedBasicBlock {
                                id: header_id,
                                label: None,
                                statements: vec![],
                                // TODO: Create proper iterator condition expression
                                // For now, treat as unconditional to body (will be fixed in SSA)
                                terminator: TypedTerminator::Jump(body_id),
                                pattern_check: None,
                            });

                            // Push loop context
                            self.loop_stack.push((header_id, after_id));

                            // Process body block
                            let (body_blocks, _, body_exit) =
                                self.split_at_control_flow(body, body_id, false)?;
                            all_blocks.extend(body_blocks);

                            // Pop loop context
                            self.loop_stack.pop();

                            // Body loops back to header
                            if let Some(last_block) =
                                all_blocks.iter_mut().rev().find(|b| b.id == body_exit)
                            {
                                if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                                    last_block.terminator = TypedTerminator::Jump(header_id);
                                }
                            }

                            // Continue with after block
                            current_statements = Vec::new();
                            current_block_id = after_id;
                            exit_id = after_id;
                        }

                        TypedLoop::ForCStyle {
                            init,
                            condition,
                            update,
                            body,
                        } => {
                            // C-style for: for (init; condition; update) body
                            // Structure: entry → init → header → body → update → header → exit
                            //                              ↓
                            //                            exit

                            // Process init statement first (if present)
                            if let Some(init_stmt) = init {
                                current_statements.push(*init_stmt.clone());
                            }

                            let header_id = self.new_block_id();
                            let body_id = self.new_block_id();
                            let update_id = self.new_block_id();
                            let after_id = self.new_block_id();

                            // Current block (with init) jumps to header
                            all_blocks.push(TypedBasicBlock {
                                id: current_block_id,
                                label: None,
                                statements: current_statements.clone(),
                                terminator: TypedTerminator::Jump(header_id),
                                pattern_check: None,
                            });

                            // Header evaluates condition
                            if let Some(cond) = condition {
                                all_blocks.push(TypedBasicBlock {
                                    id: header_id,
                                    label: None,
                                    statements: vec![],
                                    terminator: TypedTerminator::CondBranch {
                                        condition: cond.clone(),
                                        true_target: body_id,
                                        false_target: after_id,
                                    },
                                    pattern_check: None,
                                });
                            } else {
                                // No condition = infinite loop (like while(true))
                                all_blocks.push(TypedBasicBlock {
                                    id: header_id,
                                    label: None,
                                    statements: vec![],
                                    terminator: TypedTerminator::Jump(body_id),
                                    pattern_check: None,
                                });
                            }

                            // Push loop context (continue goes to update, not header)
                            self.loop_stack.push((update_id, after_id));

                            // Process body
                            let (body_blocks, _, body_exit) =
                                self.split_at_control_flow(body, body_id, false)?;
                            all_blocks.extend(body_blocks);

                            // Pop loop context
                            self.loop_stack.pop();

                            // Body goes to update block
                            if let Some(last_block) =
                                all_blocks.iter_mut().rev().find(|b| b.id == body_exit)
                            {
                                if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                                    last_block.terminator = TypedTerminator::Jump(update_id);
                                }
                            }

                            // Update block executes update expression and loops back to header
                            let mut update_statements = vec![];
                            if let Some(upd) = update {
                                // Update expression becomes a statement in the block
                                update_statements.push(typed_node(
                                    zyntax_typed_ast::typed_ast::TypedStatement::Expression(
                                        upd.clone(),
                                    ),
                                    upd.ty.clone(),
                                    upd.span,
                                ));
                            }

                            all_blocks.push(TypedBasicBlock {
                                id: update_id,
                                label: None,
                                statements: update_statements,
                                terminator: TypedTerminator::Jump(header_id),
                                pattern_check: None,
                            });

                            // Continue with after block
                            current_statements = Vec::new();
                            current_block_id = after_id;
                            exit_id = after_id;
                        }

                        TypedLoop::While { condition, body } => {
                            // While loop inside Loop enum
                            // Same structure as TypedStatement::While
                            let header_id = self.new_block_id();
                            let body_id = self.new_block_id();
                            let after_id = self.new_block_id();

                            all_blocks.push(TypedBasicBlock {
                                id: current_block_id,
                                label: None,
                                statements: current_statements.clone(),
                                terminator: TypedTerminator::Jump(header_id),
                                pattern_check: None,
                            });

                            all_blocks.push(TypedBasicBlock {
                                id: header_id,
                                label: None,
                                statements: vec![],
                                terminator: TypedTerminator::CondBranch {
                                    condition: condition.clone(),
                                    true_target: body_id,
                                    false_target: after_id,
                                },
                                pattern_check: None,
                            });

                            self.loop_stack.push((header_id, after_id));
                            let (body_blocks, _, body_exit) =
                                self.split_at_control_flow(body, body_id, false)?;
                            all_blocks.extend(body_blocks);
                            self.loop_stack.pop();

                            if let Some(last_block) =
                                all_blocks.iter_mut().rev().find(|b| b.id == body_exit)
                            {
                                if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                                    last_block.terminator = TypedTerminator::Jump(header_id);
                                }
                            }

                            current_statements = Vec::new();
                            current_block_id = after_id;
                            exit_id = after_id;
                        }

                        TypedLoop::DoWhile { body, condition } => {
                            // Do-while: body executes at least once, then checks condition
                            // Structure: entry → body → header → body (if true) → exit (if false)

                            let body_id = self.new_block_id();
                            let header_id = self.new_block_id();
                            let after_id = self.new_block_id();

                            // Entry jumps directly to body (executes at least once)
                            all_blocks.push(TypedBasicBlock {
                                id: current_block_id,
                                label: None,
                                statements: current_statements.clone(),
                                terminator: TypedTerminator::Jump(body_id),
                                pattern_check: None,
                            });

                            self.loop_stack.push((header_id, after_id));
                            let (body_blocks, _, body_exit) =
                                self.split_at_control_flow(body, body_id, false)?;
                            all_blocks.extend(body_blocks);
                            self.loop_stack.pop();

                            // Body goes to header for condition check
                            if let Some(last_block) =
                                all_blocks.iter_mut().rev().find(|b| b.id == body_exit)
                            {
                                if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                                    last_block.terminator = TypedTerminator::Jump(header_id);
                                }
                            }

                            // Header checks condition: true = loop back to body, false = exit
                            all_blocks.push(TypedBasicBlock {
                                id: header_id,
                                label: None,
                                statements: vec![],
                                terminator: TypedTerminator::CondBranch {
                                    condition: condition.clone(),
                                    true_target: body_id,
                                    false_target: after_id,
                                },
                                pattern_check: None,
                            });

                            current_statements = Vec::new();
                            current_block_id = after_id;
                            exit_id = after_id;
                        }
                    }
                }

                TypedStatement::For(for_stmt) => {
                    // Try to desugar `for i in range(start, end)` into a C-style for loop.
                    // Extract loop variable name from pattern.
                    let loop_var = match &for_stmt.pattern.node {
                        TypedPattern::Identifier { name, .. } => Some(*name),
                        _ => None,
                    };

                    // A counted loop, spelled either way a frontend spells
                    // one: `range(end)`, `range(start, end)`,
                    // `range(start, end, step)`, or the language's own
                    // `start..end` / `start..=end`. Anything else is not
                    // one and is refused below rather than guessed at.
                    let counted = counted_loop_bounds(&for_stmt.iterator);
                    if let (Some(var_name), Some(bounds)) = (loop_var, counted) {
                        let span = for_stmt.iterator.span;
                        let CountedLoop {
                            start: start_expr,
                            end: end_expr,
                            step: step_expr,
                            inclusive,
                        } = bounds;

                        let ty = start_expr.ty.clone();
                        let var = |name: InternedString| {
                            typed_node(TypedExpression::Variable(name), ty.clone(), span)
                        };
                        let int_lit = |v: i128| {
                            typed_node(
                                TypedExpression::Literal(
                                    zyntax_typed_ast::typed_ast::TypedLiteral::Integer(v),
                                ),
                                ty.clone(),
                                span,
                            )
                        };
                        let bind = |name: InternedString, value: TypedNode<TypedExpression>| {
                            typed_node(
                                TypedStatement::Let(zyntax_typed_ast::typed_ast::TypedLet {
                                    name,
                                    ty: value.ty.clone(),
                                    mutability: zyntax_typed_ast::Mutability::Mutable,
                                    initializer: Some(Box::new(value)),
                                    span,
                                }),
                                Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
                                span,
                            )
                        };
                        let binary = |op: zyntax_typed_ast::typed_ast::BinaryOp,
                                      l: TypedNode<TypedExpression>,
                                      r: TypedNode<TypedExpression>,
                                      out_ty: Type| {
                            typed_node(
                                TypedExpression::Binary(zyntax_typed_ast::typed_ast::TypedBinary {
                                    op,
                                    left: Box::new(l),
                                    right: Box::new(r),
                                }),
                                out_ty,
                                span,
                            )
                        };
                        use zyntax_typed_ast::typed_ast::BinaryOp as Op;
                        let bool_ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::Bool);

                        // The bounds are evaluated once, before the loop,
                        // into locals of the loop's own. Python and every
                        // range-loop language read them once.
                        let end_name = self.hidden_name("end", var_name);
                        current_statements.push(bind(var_name, start_expr.clone()));
                        current_statements.push(bind(end_name, end_expr));

                        // A literal step decides the comparison here; any
                        // other step is bound and tested for sign at run time,
                        // since counting down to `end` means `i > end`.
                        let literal_step = step_expr.as_ref().and_then(|s| match &s.node {
                            TypedExpression::Literal(
                                zyntax_typed_ast::typed_ast::TypedLiteral::Integer(v),
                            ) => Some(*v),
                            TypedExpression::Unary(zyntax_typed_ast::typed_ast::TypedUnary {
                                op: zyntax_typed_ast::typed_ast::UnaryOp::Minus,
                                operand,
                            }) => match &operand.node {
                                TypedExpression::Literal(
                                    zyntax_typed_ast::typed_ast::TypedLiteral::Integer(v),
                                ) => Some(-*v),
                                _ => None,
                            },
                            _ => None,
                        });
                        let up = if inclusive { Op::Le } else { Op::Lt };
                        let down = if inclusive { Op::Ge } else { Op::Gt };
                        let (cond_expr, step) = match (step_expr, literal_step) {
                            (None, _) => (
                                binary(up, var(var_name), var(end_name), bool_ty.clone()),
                                int_lit(1),
                            ),
                            (Some(step), Some(v)) => (
                                binary(
                                    if v < 0 { down } else { up },
                                    var(var_name),
                                    var(end_name),
                                    bool_ty.clone(),
                                ),
                                step,
                            ),
                            (Some(step), None) => {
                                let step_name = self.hidden_name("step", var_name);
                                current_statements.push(bind(step_name, step));
                                let zero = int_lit(0);
                                let ascending = binary(
                                    Op::And,
                                    binary(Op::Gt, var(step_name), zero.clone(), bool_ty.clone()),
                                    binary(up, var(var_name), var(end_name), bool_ty.clone()),
                                    bool_ty.clone(),
                                );
                                let descending = binary(
                                    Op::And,
                                    binary(Op::Lt, var(step_name), zero, bool_ty.clone()),
                                    binary(down, var(var_name), var(end_name), bool_ty.clone()),
                                    bool_ty.clone(),
                                );
                                (
                                    binary(Op::Or, ascending, descending, bool_ty.clone()),
                                    var(step_name),
                                )
                            }
                        };

                        // Update: i = i + step
                        let update_expr = binary(
                            Op::Assign,
                            var(var_name),
                            binary(Op::Add, var(var_name), step, ty.clone()),
                            ty.clone(),
                        );

                        // Build the C-style for loop block structure
                        let header_id = self.new_block_id();
                        let body_id = self.new_block_id();
                        let update_id = self.new_block_id();
                        let after_id = self.new_block_id();

                        // Entry block → header
                        all_blocks.push(TypedBasicBlock {
                            id: current_block_id,
                            label: None,
                            statements: current_statements.clone(),
                            terminator: TypedTerminator::Jump(header_id),
                            pattern_check: None,
                        });

                        // Header: conditional branch (i < end)
                        all_blocks.push(TypedBasicBlock {
                            id: header_id,
                            label: None,
                            statements: vec![],
                            terminator: TypedTerminator::CondBranch {
                                condition: Box::new(cond_expr),
                                true_target: body_id,
                                false_target: after_id,
                            },
                            pattern_check: None,
                        });

                        // Body
                        self.loop_stack.push((update_id, after_id));
                        let (body_blocks, _, body_exit) =
                            self.split_at_control_flow(&for_stmt.body, body_id, false)?;
                        all_blocks.extend(body_blocks);
                        self.loop_stack.pop();

                        // Body exit → update
                        if let Some(last_block) =
                            all_blocks.iter_mut().rev().find(|b| b.id == body_exit)
                        {
                            if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                                last_block.terminator = TypedTerminator::Jump(update_id);
                            }
                        }

                        // Update block: i = i + 1; jump header
                        all_blocks.push(TypedBasicBlock {
                            id: update_id,
                            label: None,
                            statements: vec![typed_node(
                                TypedStatement::Expression(Box::new(update_expr)),
                                Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
                                span,
                            )],
                            terminator: TypedTerminator::Jump(header_id),
                            pattern_check: None,
                        });

                        current_statements = Vec::new();
                        current_block_id = after_id;
                        exit_id = after_id;
                    } else {
                        // Not a counted loop. There is no iterator protocol
                        // to fall back on, and the fallback that stood here
                        // built a loop with no exit: a `for` over anything
                        // unrecognised compiled to code that never returned,
                        // with nothing said. Refusing it is the difference
                        // between a message naming the iterator and a
                        // program that hangs.
                        return Err(crate::CompilerError::Lowering(format!(
                            "`for` over this iterator is not supported: only a counted range \
                             (`range(end)`, `range(start, end)`, `start..end`) can be iterated, \
                             and the loop variable must be a plain name (at {:?})",
                            for_stmt.iterator.span
                        )));
                    }
                }

                TypedStatement::ForCStyle(for_c_stmt) => {
                    // C-style for loop (separate statement type)
                    // Same structure as TypedLoop::ForCStyle

                    // Process init
                    if let Some(init_stmt) = &for_c_stmt.init {
                        current_statements.push(*init_stmt.clone());
                    }

                    let header_id = self.new_block_id();
                    let body_id = self.new_block_id();
                    let update_id = self.new_block_id();
                    let after_id = self.new_block_id();

                    all_blocks.push(TypedBasicBlock {
                        id: current_block_id,
                        label: None,
                        statements: current_statements.clone(),
                        terminator: TypedTerminator::Jump(header_id),
                        pattern_check: None,
                    });

                    // Header with condition
                    if let Some(cond) = &for_c_stmt.condition {
                        all_blocks.push(TypedBasicBlock {
                            id: header_id,
                            label: None,
                            statements: vec![],
                            terminator: TypedTerminator::CondBranch {
                                condition: cond.clone(),
                                true_target: body_id,
                                false_target: after_id,
                            },
                            pattern_check: None,
                        });
                    } else {
                        all_blocks.push(TypedBasicBlock {
                            id: header_id,
                            label: None,
                            statements: vec![],
                            terminator: TypedTerminator::Jump(body_id),
                            pattern_check: None,
                        });
                    }

                    self.loop_stack.push((update_id, after_id));
                    let (body_blocks, _, body_exit) =
                        self.split_at_control_flow(&for_c_stmt.body, body_id, false)?;
                    all_blocks.extend(body_blocks);
                    self.loop_stack.pop();

                    if let Some(last_block) =
                        all_blocks.iter_mut().rev().find(|b| b.id == body_exit)
                    {
                        if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                            last_block.terminator = TypedTerminator::Jump(update_id);
                        }
                    }

                    // Update block
                    let mut update_statements = vec![];
                    if let Some(upd) = &for_c_stmt.update {
                        update_statements.push(typed_node(
                            zyntax_typed_ast::typed_ast::TypedStatement::Expression(upd.clone()),
                            upd.ty.clone(),
                            upd.span,
                        ));
                    }

                    all_blocks.push(TypedBasicBlock {
                        id: update_id,
                        label: None,
                        statements: update_statements,
                        terminator: TypedTerminator::Jump(header_id),
                        pattern_check: None,
                    });

                    current_statements = Vec::new();
                    current_block_id = after_id;
                    exit_id = after_id;
                }

                TypedStatement::Block(block) => {
                    // Close current block before processing nested block
                    let block_entry_id = self.new_block_id();
                    let after_block_id = self.new_block_id();

                    all_blocks.push(TypedBasicBlock {
                        id: current_block_id,
                        label: None,
                        statements: current_statements.clone(),
                        terminator: TypedTerminator::Jump(block_entry_id),
                        pattern_check: None,
                    });

                    // Nested block - recursively process
                    let (block_blocks, _, block_exit) =
                        self.split_at_control_flow(block, block_entry_id, false)?;

                    // Add all blocks from the nested block
                    all_blocks.extend(block_blocks);

                    // Make the block exit jump to the continuation block
                    if let Some(last_block) =
                        all_blocks.iter_mut().rev().find(|b| b.id == block_exit)
                    {
                        if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                            last_block.terminator = TypedTerminator::Jump(after_block_id);
                        }
                    }

                    // Continue after the nested block with a fresh block
                    current_statements = Vec::new();
                    current_block_id = after_block_id;
                    exit_id = after_block_id;
                }

                TypedStatement::With(with_stmt) => {
                    // `with H { body }` — for this slice, lower the body
                    // like a nested block so it executes. Single-handler
                    // performs already resolve correctly via static
                    // dispatch, so this is behaviourally correct today.
                    //
                    // NEXT SLICE: emit `__zyntax_effect_push_handler` at
                    // `with_entry_id` and `__zyntax_effect_pop_handler`
                    // on every exit edge out of `with_exit` (via the
                    // shared ScopeExitEmitter) so handler scoping becomes
                    // regional. The entry/exit boundary is kept distinct
                    // here precisely so that insertion has a home.
                    let with_entry_id = self.new_block_id();
                    let after_with_id = self.new_block_id();

                    all_blocks.push(TypedBasicBlock {
                        id: current_block_id,
                        label: None,
                        statements: current_statements.clone(),
                        terminator: TypedTerminator::Jump(with_entry_id),
                        pattern_check: None,
                    });

                    let (with_blocks, _, with_exit) =
                        self.split_at_control_flow(&with_stmt.body, with_entry_id, false)?;
                    // Record the body's block ids (entry included) so the
                    // post-SSA pass can classify scope-exiting edges.
                    let mut body_block_ids: Vec<HirId> = with_blocks.iter().map(|b| b.id).collect();
                    if !body_block_ids.contains(&with_entry_id) {
                        body_block_ids.push(with_entry_id);
                    }
                    all_blocks.extend(with_blocks);

                    if let Some(last_block) =
                        all_blocks.iter_mut().rev().find(|b| b.id == with_exit)
                    {
                        if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                            last_block.terminator = TypedTerminator::Jump(after_with_id);
                        }
                    }

                    // Single handler today (`Vec` in the typed AST is for
                    // forward-compat); record it for the push/pop pass.
                    if let Some(handler) = with_stmt.handlers.first() {
                        self.with_scopes.push(WithScopeInfo {
                            handler_name: handler.name,
                            entry: with_entry_id,
                            after: after_with_id,
                            body_blocks: body_block_ids,
                        });
                    }

                    current_statements = Vec::new();
                    current_block_id = after_with_id;
                    exit_id = after_with_id;
                }

                TypedStatement::Expression(expr) => {
                    // A block used as a statement is split like a block
                    // statement: the statements so far close into a block
                    // that jumps into it, and what follows continues in a
                    // fresh block after it. The recursion must not reuse
                    // the current id, which is already claimed by the
                    // statements before the block.
                    if let TypedExpression::Block(block) = &expr.node {
                        let block_entry_id = self.new_block_id();
                        let after_block_id = self.new_block_id();
                        all_blocks.push(TypedBasicBlock {
                            id: current_block_id,
                            label: None,
                            statements: current_statements.clone(),
                            terminator: TypedTerminator::Jump(block_entry_id),
                            pattern_check: None,
                        });
                        let (block_blocks, _, block_exit) =
                            self.split_at_control_flow(block, block_entry_id, false)?;
                        all_blocks.extend(block_blocks);
                        if let Some(last_block) =
                            all_blocks.iter_mut().rev().find(|b| b.id == block_exit)
                        {
                            if matches!(last_block.terminator, TypedTerminator::Unreachable) {
                                last_block.terminator = TypedTerminator::Jump(after_block_id);
                            }
                        }
                        current_statements = Vec::new();
                        current_block_id = after_block_id;
                        exit_id = after_block_id;
                    } else {
                        current_statements.push(stmt.clone());
                    }
                }

                // A label starts its own block, which the statements
                // before it fall into; a goto ends its block with a
                // jump there and what follows is unreached.
                TypedStatement::Label(name) => {
                    let label_id = self.label_block(*name);
                    all_blocks.push(TypedBasicBlock {
                        id: current_block_id,
                        label: None,
                        statements: current_statements.clone(),
                        terminator: TypedTerminator::Jump(label_id),
                        pattern_check: None,
                    });
                    current_statements = Vec::new();
                    current_block_id = label_id;
                    exit_id = label_id;
                }

                TypedStatement::Goto(name) => {
                    let label_id = self.label_block(*name);
                    all_blocks.push(TypedBasicBlock {
                        id: current_block_id,
                        label: None,
                        statements: current_statements.clone(),
                        terminator: TypedTerminator::Jump(label_id),
                        pattern_check: None,
                    });
                    let unreachable_id = self.new_block_id();
                    current_statements = Vec::new();
                    current_block_id = unreachable_id;
                }

                TypedStatement::Break(value_opt) => {
                    // Break jumps to loop exit
                    if let Some(&(_header_id, exit_id)) = self.loop_stack.last() {
                        all_blocks.push(TypedBasicBlock {
                            id: current_block_id,
                            label: None,
                            statements: current_statements.clone(),
                            terminator: TypedTerminator::Jump(exit_id),
                            pattern_check: None,
                        });

                        // Create a new unreachable block for any statements after break
                        let unreachable_id = self.new_block_id();
                        current_statements = Vec::new();
                        current_block_id = unreachable_id;
                        // Don't update exit_id - break already jumped
                    } else {
                        // Break outside loop - treat as error or unreachable
                        // For now, just add to current block and let type checker catch it
                        current_statements.push(stmt.clone());
                    }
                }

                TypedStatement::Continue => {
                    log::debug!(
                        "[CFG] Continue: current_block={:?} with {} stmts",
                        current_block_id,
                        current_statements.len()
                    );
                    // Continue jumps to loop header
                    if let Some(&(header_id, _exit_id)) = self.loop_stack.last() {
                        log::debug!("[CFG] Continue: jumping to header={:?}", header_id);
                        all_blocks.push(TypedBasicBlock {
                            id: current_block_id,
                            label: None,
                            statements: current_statements.clone(),
                            terminator: TypedTerminator::Jump(header_id),
                            pattern_check: None,
                        });

                        // Create a new unreachable block for any statements after continue
                        let unreachable_id = self.new_block_id();
                        log::debug!(
                            "[CFG] Continue: created unreachable block={:?}",
                            unreachable_id
                        );
                        current_statements = Vec::new();
                        current_block_id = unreachable_id;
                    } else {
                        // Continue outside loop
                        current_statements.push(stmt.clone());
                    }
                }

                TypedStatement::Match(match_stmt) => {
                    // Match statement: evaluate scrutinee, then check each arm's pattern
                    // Structure: entry(scrutinee) → arm1_check → arm1_body → merge
                    //                                    ↓           ↓
                    //                               arm2_check → arm2_body → merge
                    //                                    ↓           ↓
                    //                               arm3_check → arm3_body → merge
                    //                                    ↓
                    //                               unreachable (if exhaustive)

                    if match_stmt.arms.is_empty() {
                        // Empty match - just treat as regular statement
                        current_statements.push(stmt.clone());
                        return Ok((all_blocks, entry_id, exit_id));
                    }

                    let merge_id = self.new_block_id();

                    // Entry block evaluates scrutinee and jumps to first pattern check
                    let first_pattern_id = self.new_block_id();

                    // Add the Match statement to the entry block so SSA can process it
                    let mut entry_statements = current_statements.clone();
                    entry_statements.push(stmt.clone());

                    all_blocks.push(TypedBasicBlock {
                        id: current_block_id,
                        label: None,
                        statements: entry_statements,
                        terminator: TypedTerminator::Jump(first_pattern_id),
                        pattern_check: None,
                    });

                    let mut prev_pattern_id = first_pattern_id;

                    // Create blocks for each arm
                    for (i, arm) in match_stmt.arms.iter().enumerate() {
                        let body_id = self.new_block_id();
                        let next_pattern_id = if i + 1 < match_stmt.arms.len() {
                            self.new_block_id() // Next arm's pattern check
                        } else {
                            self.new_block_id() // Unreachable (match should be exhaustive)
                        };

                        // Pattern check block
                        if let Some(guard) = &arm.guard {
                            // With guard: conditional on guard expression
                            all_blocks.push(TypedBasicBlock {
                                id: prev_pattern_id,
                                label: None,
                                statements: vec![],
                                terminator: TypedTerminator::CondBranch {
                                    condition: guard.clone(),
                                    true_target: body_id,
                                    false_target: next_pattern_id,
                                },
                                pattern_check: None,
                            });
                        } else {
                            // No guard: try to generate pattern check
                            let pattern_check = self.generate_pattern_check(
                                &match_stmt.scrutinee,
                                &arm.pattern,
                                arm.pattern.span,
                            );

                            if let Some(check_expr) = pattern_check {
                                // Pattern requires runtime check (e.g., literal comparison)
                                all_blocks.push(TypedBasicBlock {
                                    id: prev_pattern_id,
                                    label: None,
                                    statements: vec![],
                                    terminator: TypedTerminator::CondBranch {
                                        condition: Box::new(check_expr),
                                        true_target: body_id,
                                        false_target: next_pattern_id,
                                    },
                                    pattern_check: None,
                                });
                            } else {
                                // Pattern always matches (wildcard, binding, or check handled by SSA)
                                // For enum patterns, store pattern info for SSA to generate discriminant check
                                let pattern_check_info = self.extract_pattern_check_info(
                                    &match_stmt.scrutinee,
                                    &arm.pattern,
                                    Some(next_pattern_id),
                                );

                                // Jump directly to body (SSA will upgrade to CondBranch if needed)
                                all_blocks.push(TypedBasicBlock {
                                    id: prev_pattern_id,
                                    label: None,
                                    statements: vec![],
                                    terminator: TypedTerminator::Jump(body_id),
                                    pattern_check: pattern_check_info,
                                });
                            }
                        }

                        // Body block - extract statements from the arm body
                        // If arm body is a Block expression, extract its statements
                        // Otherwise, wrap it in an Expression statement.
                        //
                        // The last statement of the arm body can be a flow
                        // terminator (Return / Break / Continue) — promote it
                        // to a real CFG terminator so the merge edge is
                        // omitted. This is what makes `while let Some(x) = ...
                        // { ... }` (which desugars to `while true { match ... {
                        // case _ { break } ... } }`) actually exit the loop:
                        // the Break arm's terminator becomes
                        // `Jump(loop_exit)`, not `Jump(merge)`.
                        let (body_stmts, body_terminator) = match &arm.body.node {
                            TypedExpression::Block(block) => {
                                if let Some(last_stmt) = block.statements.last() {
                                    let prefix =
                                        block.statements[..block.statements.len() - 1].to_vec();
                                    match &last_stmt.node {
                                        TypedStatement::Return(ret_expr) => {
                                            (prefix, TypedTerminator::Return(ret_expr.clone()))
                                        }
                                        TypedStatement::Break(_) => {
                                            let target = self
                                                .loop_stack
                                                .last()
                                                .map(|&(_h, e)| e)
                                                .unwrap_or(merge_id);
                                            (prefix, TypedTerminator::Jump(target))
                                        }
                                        TypedStatement::Continue => {
                                            let target = self
                                                .loop_stack
                                                .last()
                                                .map(|&(h, _e)| h)
                                                .unwrap_or(merge_id);
                                            (prefix, TypedTerminator::Jump(target))
                                        }
                                        _ => (
                                            block.statements.clone(),
                                            TypedTerminator::Jump(merge_id),
                                        ),
                                    }
                                } else {
                                    (vec![], TypedTerminator::Jump(merge_id))
                                }
                            }
                            _ => {
                                // Non-block expression, wrap in Expression statement
                                let body_stmt = typed_node(
                                    zyntax_typed_ast::typed_ast::TypedStatement::Expression(
                                        arm.body.clone(),
                                    ),
                                    arm.body.ty.clone(),
                                    arm.body.span,
                                );
                                (vec![body_stmt], TypedTerminator::Jump(merge_id))
                            }
                        };

                        // Store pattern info on body block for variable extraction
                        let body_pattern_info = self.extract_pattern_check_info(
                            &match_stmt.scrutinee,
                            &arm.pattern,
                            None,
                        );

                        // An arm can contain nested loops and branches. Split
                        // them into CFG blocks before attaching the arm's
                        // pattern binding to its entry block.
                        let body_block = TypedBlock {
                            statements: body_stmts,
                            span: arm.body.span,
                        };
                        let (mut body_blocks, _, body_exit) =
                            self.split_at_control_flow(&body_block, body_id, false)?;
                        if let Some(entry) = body_blocks.iter_mut().find(|b| b.id == body_id) {
                            entry.pattern_check = body_pattern_info;
                        }
                        if let Some(exit) = body_blocks.iter_mut().find(|b| b.id == body_exit) {
                            if matches!(exit.terminator, TypedTerminator::Unreachable) {
                                exit.terminator = body_terminator;
                            }
                        }
                        all_blocks.extend(body_blocks);

                        prev_pattern_id = next_pattern_id;
                    }

                    // Last pattern check block is unreachable (if match is exhaustive)
                    all_blocks.push(TypedBasicBlock {
                        id: prev_pattern_id,
                        label: None,
                        statements: vec![],
                        terminator: TypedTerminator::Unreachable,
                        pattern_check: None,
                    });

                    // Continue with merge block
                    current_statements = Vec::new();
                    current_block_id = merge_id;
                    exit_id = merge_id;
                }

                TypedStatement::Return(expr) => {
                    // Current block ends with return
                    all_blocks.push(TypedBasicBlock {
                        id: current_block_id,
                        label: None,
                        statements: current_statements.clone(),
                        terminator: TypedTerminator::Return(expr.clone()),
                        pattern_check: None,
                    });

                    // No more processing after return
                    exit_id = current_block_id;
                    return Ok((all_blocks, entry_id, exit_id));
                }

                _ => {
                    // Regular statement - add to current block
                    current_statements.push(stmt.clone());
                }
            }
        }

        // Always create a final block if we have a current_block_id that hasn't been added yet
        // This handles the case where control flow statements leave us with an empty continuation block
        log::debug!(
            "[CFG] End: current_block={:?}, current_statements={}, exit={:?}",
            current_block_id,
            current_statements.len(),
            exit_id
        );
        if !all_blocks.iter().any(|b| b.id == current_block_id) {
            log::debug!(
                "[CFG] End: creating final block with {} statements",
                current_statements.len()
            );

            // Special case: If this is a function body and has exactly one statement that's an expression,
            // treat it as an implicit return. This handles cases like:
            //   fn add(self, rhs: Tensor) -> Tensor { extern tensor_add(self, rhs) }
            // where the single expression should be returned.
            // Do NOT apply this to blocks inside control flow (if, match, etc.) - those should not implicitly return.
            // A statement of no value (a call for its effect) is not a
            // return: the block ends as any other, and the function's
            // own return follows.
            let (final_statements, terminator) = if is_function_body
                && current_statements.len() == 1
            {
                match &current_statements[0].node {
                    TypedStatement::Expression(expr)
                        if !matches!(
                            expr.ty,
                            Type::Primitive(zyntax_typed_ast::type_registry::PrimitiveType::Unit)
                        ) =>
                    {
                        // Single expression in function body - implicitly return it
                        (
                            vec![],
                            TypedTerminator::Return(Some(Box::new((**expr).clone()))),
                        )
                    }
                    _ => (current_statements, TypedTerminator::Unreachable),
                }
            } else {
                // Multiple statements, no statements, or not a function body - keep as unreachable
                (current_statements, TypedTerminator::Unreachable)
            };

            all_blocks.push(TypedBasicBlock {
                id: current_block_id,
                label: None,
                statements: final_statements,
                terminator,
                pattern_check: None,
            });
            exit_id = current_block_id;
        } else {
            log::debug!("[CFG] End: current_block already exists, not creating");
        }

        log::debug!(
            "[CFG] Returning: {} blocks, entry={:?}, exit={:?}",
            all_blocks.len(),
            entry_id,
            exit_id
        );
        Ok((all_blocks, entry_id, exit_id))
    }

    /// Generate a pattern check condition for simple patterns
    /// Returns None if the pattern always matches (wildcard, simple binding)
    fn generate_pattern_check(
        &self,
        scrutinee: &TypedNode<TypedExpression>,
        pattern: &TypedNode<TypedPattern>,
        span: Span,
    ) -> Option<TypedNode<TypedExpression>> {
        use zyntax_typed_ast::typed_ast::{BinaryOp, TypedBinary, TypedLiteral};

        log::debug!("[CFG] generate_pattern_check: pattern={:?}", pattern.node);

        match &pattern.node {
            // Wildcard always matches - no check needed
            TypedPattern::Wildcard => None,

            // Simple identifier binding on Optional type means "is Some" check
            // For ?T types, we assume this is checking if the value exists
            TypedPattern::Identifier { .. } => {
                // Check if scrutinee is Optional type
                match &scrutinee.ty {
                    Type::Optional(_inner_ty) => {
                        // For now, we can't easily check discriminant here at TypedAST level
                        // The SSA builder will need to handle this
                        // Return None to indicate unconditional match (value will be bound in SSA)
                        None
                    }
                    _ => None, // Non-optional types: binding always succeeds
                }
            }

            // Enum variant pattern: check discriminant
            TypedPattern::Enum { variant, .. } => {
                // For enum patterns like Some(x), we need to check the discriminant
                // This requires extracting discriminant at runtime
                // For now, return None and let SSA handle it
                // TODO: Generate discriminant check expression
                log::debug!(
                    "[CFG] TODO: Generate discriminant check for variant {:?}",
                    variant
                );
                None
            }

            // Literal pattern: check equality
            TypedPattern::Literal(lit_pattern) => {
                use zyntax_typed_ast::typed_ast::TypedLiteralPattern;
                let lit_expr = match lit_pattern {
                    TypedLiteralPattern::Integer(i) => {
                        TypedExpression::Literal(TypedLiteral::Integer(*i))
                    }
                    TypedLiteralPattern::Bool(b) => {
                        TypedExpression::Literal(TypedLiteral::Bool(*b))
                    }
                    _ => return None, // Other literals not yet supported
                };

                // Use the scrutinee's type for the comparison — ensures both sides
                // have matching types in the Cranelift verifier.
                // Generate: scrutinee == literal
                Some(typed_node(
                    TypedExpression::Binary(TypedBinary {
                        op: BinaryOp::Eq,
                        left: Box::new(scrutinee.clone()),
                        right: Box::new(typed_node(lit_expr, scrutinee.ty.clone(), span)),
                    }),
                    Type::Primitive(zyntax_typed_ast::PrimitiveType::Bool),
                    span,
                ))
            }

            // Struct pattern: AND together each field's pattern check
            // `Point { x: 0, y: 0 }` becomes `scrutinee.x == 0 && scrutinee.y == 0`
            // Field bindings (`x` shorthand) don't generate checks — they always match.
            TypedPattern::Struct {
                fields: field_patterns,
                ..
            } => {
                use zyntax_typed_ast::typed_ast::{TypedFieldAccess, TypedPattern};

                let mut checks: Vec<TypedNode<TypedExpression>> = Vec::new();

                for field_pat in field_patterns {
                    // If the field pattern is a binding (Identifier) or wildcard,
                    // it always matches — skip the check.
                    match &field_pat.pattern.node {
                        TypedPattern::Identifier { .. } | TypedPattern::Wildcard => continue,
                        _ => {}
                    }

                    // Build scrutinee.field_name expression
                    let field_access = typed_node(
                        TypedExpression::Field(TypedFieldAccess {
                            object: Box::new(scrutinee.clone()),
                            field: field_pat.name,
                        }),
                        Type::Unknown,
                        field_pat.pattern.span,
                    );

                    // Recursively generate the check for this field's pattern
                    if let Some(check) = self.generate_pattern_check(
                        &field_access,
                        &field_pat.pattern,
                        field_pat.pattern.span,
                    ) {
                        checks.push(check);
                    }
                }

                if checks.is_empty() {
                    // No checks needed (all bindings) — pattern always matches
                    return None;
                }

                // AND all checks together via BitAnd (no short-circuit needed —
                // each check is a simple field comparison and they all get evaluated)
                let mut combined = checks[0].clone();
                for check in &checks[1..] {
                    combined = typed_node(
                        TypedExpression::Binary(TypedBinary {
                            op: BinaryOp::BitAnd,
                            left: Box::new(combined),
                            right: Box::new(check.clone()),
                        }),
                        Type::Primitive(zyntax_typed_ast::PrimitiveType::Bool),
                        span,
                    );
                }
                Some(combined)
            }

            // Tuple pattern: AND together each element's pattern check.
            // Element bindings (Identifier/Wildcard) don't generate checks.
            // Note: tuple element access requires positional field access which
            // we represent as Field with synthetic field names "_0", "_1", etc.
            // For now we only support binding/wildcard tuples (no literal checks).
            TypedPattern::Tuple(element_patterns) => {
                use zyntax_typed_ast::typed_ast::TypedPattern;
                // If any element pattern needs a check, we'd need positional
                // field access which isn't expressible in TypedAST. For now,
                // require all elements to be bindings or wildcards.
                for ep in element_patterns {
                    match &ep.node {
                        TypedPattern::Identifier { .. } | TypedPattern::Wildcard => continue,
                        _ => return None, // Can't generate check for nested patterns yet
                    }
                }
                None // All bindings — no check needed
            }

            // Other patterns not yet implemented
            _ => None,
        }
    }

    /// Extract pattern check information for SSA to use
    /// Returns PatternCheckInfo if this pattern requires runtime checking
    fn extract_pattern_check_info(
        &self,
        scrutinee: &TypedNode<TypedExpression>,
        pattern: &TypedNode<TypedPattern>,
        false_target: Option<HirId>,
    ) -> Option<PatternCheckInfo> {
        use zyntax_typed_ast::typed_ast::TypedPattern;

        match &pattern.node {
            // Enum variant patterns need discriminant checks
            TypedPattern::Enum { variant, .. } => {
                // Get variant index from the type system
                let variant_index = self.get_variant_index(&scrutinee.ty, variant)?;

                Some(PatternCheckInfo {
                    scrutinee: scrutinee.clone(),
                    pattern: pattern.clone(),
                    variant_index: Some(variant_index),
                    false_target,
                })
            }

            // Constructor patterns like `Some(x)` / `None()` / `Ok(v)`
            // / `Err(e)` arrive here when the parser tagged them
            // as `Constructor { constructor: Type::Unresolved(name), ... }`
            // rather than as full `Enum` patterns (it can't resolve
            // the enum context generically). Treat them the same
            // way at this layer: derive the variant index by name.
            TypedPattern::Constructor { constructor, .. } => {
                // Only handle `Type::Unresolved(name)` here — the parser
                // emits the constructor as an unresolved type because it
                // doesn't know which enum the variant belongs to. For
                // `Type::Named { id, .. }` cases (where the parser DID
                // resolve the parent enum) we'd need a TypeRegistry
                // lookup to get the name — not threaded here yet.
                let name = match constructor {
                    Type::Unresolved(name) => *name,
                    _ => return None,
                };
                let variant_index = self.get_variant_index(&scrutinee.ty, &name)?;

                Some(PatternCheckInfo {
                    scrutinee: scrutinee.clone(),
                    pattern: pattern.clone(),
                    variant_index: Some(variant_index),
                    false_target,
                })
            }

            // Struct/tuple/array patterns: SSA needs to extract bindings recursively.
            // variant_index is None — they aren't union variants.
            TypedPattern::Struct { .. } | TypedPattern::Tuple(_) | TypedPattern::Array(_) => {
                Some(PatternCheckInfo {
                    scrutinee: scrutinee.clone(),
                    pattern: pattern.clone(),
                    variant_index: None,
                    false_target,
                })
            }

            // Wildcards and simple bindings don't need checks
            TypedPattern::Wildcard => None,
            TypedPattern::Identifier { .. } => None,

            // Other patterns might need checks but we'll handle them later
            _ => None,
        }
    }

    /// Get the discriminant index for a variant in a union/enum type
    fn get_variant_index(&self, ty: &Type, variant_name: &InternedString) -> Option<u32> {
        match ty {
            Type::Optional(_) => {
                // Optional has two variants: None (0) and Some (1)
                let mut arena = zyntax_typed_ast::arena::AstArena::new();
                let none = arena.intern_string("None");
                let some = arena.intern_string("Some");

                if variant_name == &none {
                    Some(0)
                } else if variant_name == &some {
                    Some(1)
                } else {
                    None
                }
            }

            Type::Result { .. } => {
                // Result has two variants: Ok (0) and Err (1)
                let mut arena = zyntax_typed_ast::arena::AstArena::new();
                let ok = arena.intern_string("Ok");
                let err = arena.intern_string("Err");

                if variant_name == &ok {
                    Some(0)
                } else if variant_name == &err {
                    Some(1)
                } else {
                    None
                }
            }

            // Fallback for cases where the scrutinee's typed-AST `.ty`
            // is `Any` / `Unresolved` (common when the parser can't
            // infer a method-call's return type — e.g. built-in
            // dispatch like `Fiber<T>::next() -> Option<T>`). The
            // variant *name* uniquely identifies the union shape for
            // language-defined unions (Option, Result), so it's
            // safe to recognise them by name. A user-defined enum
            // that re-uses one of these names would clash with the
            // built-in anyway.
            _ => {
                let mut arena = zyntax_typed_ast::arena::AstArena::new();
                let none = arena.intern_string("None");
                let some = arena.intern_string("Some");
                let ok = arena.intern_string("Ok");
                let err = arena.intern_string("Err");

                if variant_name == &none {
                    Some(0)
                } else if variant_name == &some {
                    Some(1)
                } else if variant_name == &ok {
                    Some(0)
                } else if variant_name == &err {
                    Some(1)
                } else {
                    None
                }
            }
        }
    }
}

/// The bounds of a counted loop, however the frontend wrote it.
struct CountedLoop {
    start: TypedNode<TypedExpression>,
    end: TypedNode<TypedExpression>,
    step: Option<TypedNode<TypedExpression>>,
    inclusive: bool,
}

/// `range(...)` with one to three arguments, or a range expression.
/// `None` for anything else, which the caller refuses.
fn counted_loop_bounds(iter: &TypedNode<TypedExpression>) -> Option<CountedLoop> {
    let zero = |span: Span| {
        typed_node(
            TypedExpression::Literal(zyntax_typed_ast::typed_ast::TypedLiteral::Integer(0)),
            Type::Primitive(zyntax_typed_ast::PrimitiveType::I64),
            span,
        )
    };
    match &iter.node {
        TypedExpression::Range(r) => Some(CountedLoop {
            start: r
                .start
                .as_deref()
                .cloned()
                .unwrap_or_else(|| zero(iter.span)),
            end: r.end.as_deref()?.clone(),
            step: None,
            inclusive: r.inclusive,
        }),
        TypedExpression::Call(call) => {
            let TypedExpression::Variable(name) = &call.callee.node else {
                return None;
            };
            if name.resolve_global().as_deref() != Some("range") {
                return None;
            }
            let (start, end, step) = match call.positional_args.as_slice() {
                [end] => (zero(iter.span), end.clone(), None),
                [start, end] => (start.clone(), end.clone(), None),
                [start, end, step] => (start.clone(), end.clone(), Some(step.clone())),
                _ => return None,
            };
            Some(CountedLoop {
                start,
                end,
                step,
                inclusive: false,
            })
        }
        _ => None,
    }
}

/// What [`visit_nested_statements`] and [`visit_nested_expression`] meet.
pub(crate) enum Nested<'a> {
    Expr(&'a TypedNode<TypedExpression>),
    Stmt(&'a TypedNode<TypedStatement>),
    /// A loop body begins; what follows up to the matching end is in it.
    LoopBodyStart,
    LoopBodyEnd,
}

/// Visit `stmts` and everything nested in them, statements and
/// expressions both, in source order. Closure bodies are functions of
/// their own and are not entered. A loop's condition, iterator and
/// update are visited outside its body, as they are split.
pub(crate) fn visit_nested_statements<'a>(
    stmts: &'a [TypedNode<TypedStatement>],
    f: &mut dyn FnMut(Nested<'a>),
) {
    use zyntax_typed_ast::typed_ast::TypedLoop;
    let body = |b: &'a TypedBlock, f: &mut dyn FnMut(Nested<'a>)| {
        f(Nested::LoopBodyStart);
        visit_nested_statements(&b.statements, f);
        f(Nested::LoopBodyEnd);
    };
    for stmt in stmts {
        f(Nested::Stmt(stmt));
        match &stmt.node {
            TypedStatement::Let(l) => {
                if let Some(init) = &l.initializer {
                    visit_nested_expression(init, f);
                }
            }
            TypedStatement::LetPattern(l) => visit_nested_expression(&l.initializer, f),
            TypedStatement::Expression(e) | TypedStatement::Yield(e) | TypedStatement::Throw(e) => {
                visit_nested_expression(e, f)
            }
            TypedStatement::Return(Some(e)) | TypedStatement::Break(Some(e)) => {
                visit_nested_expression(e, f)
            }
            TypedStatement::If(i) => {
                visit_nested_expression(&i.condition, f);
                visit_nested_statements(&i.then_block.statements, f);
                if let Some(e) = &i.else_block {
                    visit_nested_statements(&e.statements, f);
                }
            }
            TypedStatement::While(w) => {
                visit_nested_expression(&w.condition, f);
                body(&w.body, f);
            }
            TypedStatement::For(l) => {
                visit_nested_expression(&l.iterator, f);
                body(&l.body, f);
            }
            TypedStatement::ForCStyle(l) => {
                if let Some(init) = &l.init {
                    visit_nested_statements(std::slice::from_ref(&**init), f);
                }
                for e in [&l.condition, &l.update].into_iter().flatten() {
                    visit_nested_expression(e, f);
                }
                body(&l.body, f);
            }
            TypedStatement::Loop(l) => match l {
                TypedLoop::ForEach {
                    iterator, body: b, ..
                } => {
                    visit_nested_expression(iterator, f);
                    body(b, f);
                }
                TypedLoop::ForCStyle {
                    init,
                    condition,
                    update,
                    body: b,
                } => {
                    if let Some(init) = init {
                        visit_nested_statements(std::slice::from_ref(&**init), f);
                    }
                    for e in [condition, update].into_iter().flatten() {
                        visit_nested_expression(e, f);
                    }
                    body(b, f);
                }
                TypedLoop::While { condition, body: b }
                | TypedLoop::DoWhile { body: b, condition } => {
                    visit_nested_expression(condition, f);
                    body(b, f);
                }
                TypedLoop::Infinite { body: b } => body(b, f),
            },
            TypedStatement::Match(m) => {
                visit_nested_expression(&m.scrutinee, f);
                for arm in &m.arms {
                    if let Some(g) = &arm.guard {
                        visit_nested_expression(g, f);
                    }
                    visit_nested_expression(&arm.body, f);
                }
            }
            TypedStatement::Block(b) => visit_nested_statements(&b.statements, f),
            TypedStatement::With(w) => visit_nested_statements(&w.body.statements, f),
            TypedStatement::Return(None)
            | TypedStatement::Break(None)
            | TypedStatement::Continue
            | TypedStatement::Label(_)
            | TypedStatement::Goto(_)
            | TypedStatement::Try(_)
            | TypedStatement::Coroutine(_)
            | TypedStatement::Defer(_)
            | TypedStatement::Select(_) => {}
        }
    }
}

/// [`visit_nested_statements`] from an expression.
pub(crate) fn visit_nested_expression<'a>(
    expr: &'a TypedNode<TypedExpression>,
    f: &mut dyn FnMut(Nested<'a>),
) {
    f(Nested::Expr(expr));
    let mut each = |e: &'a TypedNode<TypedExpression>| visit_nested_expression(e, f);
    match &expr.node {
        TypedExpression::Binary(b) => {
            each(&b.left);
            each(&b.right);
        }
        TypedExpression::Unary(u) => each(&u.operand),
        TypedExpression::Call(c) => {
            each(&c.callee);
            for a in &c.positional_args {
                each(a);
            }
            for a in &c.named_args {
                each(&a.value);
            }
        }
        TypedExpression::MethodCall(m) => {
            each(&m.receiver);
            for a in &m.positional_args {
                each(a);
            }
            for a in &m.named_args {
                each(&a.value);
            }
        }
        TypedExpression::Field(field) => each(&field.object),
        TypedExpression::Index(i) => {
            each(&i.object);
            each(&i.index);
        }
        TypedExpression::Array(items) | TypedExpression::Tuple(items) => {
            for e in items {
                each(e);
            }
        }
        TypedExpression::Struct(s) => {
            for field in &s.fields {
                each(&field.value);
            }
        }
        TypedExpression::Match(m) => {
            each(&m.scrutinee);
            for arm in &m.arms {
                if let Some(g) = &arm.guard {
                    each(g);
                }
                each(&arm.body);
            }
        }
        TypedExpression::If(i) => {
            each(&i.condition);
            each(&i.then_branch);
            each(&i.else_branch);
        }
        TypedExpression::Cast(c) => each(&c.expr),
        TypedExpression::Await(e) | TypedExpression::Try(e) | TypedExpression::Dereference(e) => {
            each(e)
        }
        TypedExpression::Reference(r) => each(&r.expr),
        TypedExpression::Range(r) => {
            for e in [&r.start, &r.end].into_iter().flatten() {
                each(e);
            }
        }
        TypedExpression::Slice(s) => {
            each(&s.object);
            for e in [&s.start, &s.end, &s.step].into_iter().flatten() {
                each(e);
            }
        }
        TypedExpression::Block(b) => visit_nested_statements(&b.statements, f),
        TypedExpression::Compute(c) => {
            for a in &c.args {
                each(a);
            }
            visit_nested_statements(&c.body.statements, f);
        }
        TypedExpression::Lambda(_)
        | TypedExpression::Literal(_)
        | TypedExpression::Variable(_)
        | TypedExpression::ListComprehension(_)
        | TypedExpression::ImportModifier(_)
        | TypedExpression::Path(_) => {}
    }
}

/// Whether a `break`, and whether a `continue`, nested in the
/// expressions of `block` acts on the loop around the block: one no
/// loop inside the expression encloses. A `match` statement's arms are
/// blocks of their own, so only its scrutinee is looked in.
pub(crate) fn escaping_jumps(block: &TypedBasicBlock) -> (bool, bool) {
    let (mut breaks, mut continues, mut depth) = (false, false, 0usize);
    let mut f = |n: Nested<'_>| match n {
        Nested::LoopBodyStart => depth += 1,
        Nested::LoopBodyEnd => depth -= 1,
        Nested::Stmt(s) if depth == 0 => match s.node {
            TypedStatement::Break(_) => breaks = true,
            TypedStatement::Continue => continues = true,
            _ => {}
        },
        _ => {}
    };
    // Only what is nested in an expression counts: the block's own
    // statements hold no loop control, which splitting made jumps.
    for stmt in &block.statements {
        match &stmt.node {
            TypedStatement::Match(m) => visit_nested_expression(&m.scrutinee, &mut f),
            TypedStatement::Let(l) => {
                if let Some(init) = &l.initializer {
                    visit_nested_expression(init, &mut f);
                }
            }
            TypedStatement::Expression(e) | TypedStatement::Yield(e) => {
                visit_nested_expression(e, &mut f)
            }
            _ => {}
        }
    }
    match &block.terminator {
        TypedTerminator::CondBranch { condition, .. } => visit_nested_expression(condition, &mut f),
        TypedTerminator::Return(Some(value)) => visit_nested_expression(value, &mut f),
        _ => {}
    }
    (breaks, continues)
}

impl Default for TypedCfgBuilder {
    fn default() -> Self {
        Self::new()
    }
}
