//! # Typed Control Flow Graph Builder
//!
//! This module creates CFG structures from TypedAST without converting to HIR first.
//! This breaks the circular dependency between CFG and SSA construction:
//! - TypedCfgBuilder creates CFG structure from TypedAST (control flow only)
//! - SsaBuilder then processes TypedStatements to emit HIR instructions
//!
//! This is the solution to Gap #4 (CFG Construction) described in INTEGRATION_GAPS_ANALYSIS.md

use crate::hir::HirId;
use crate::CompilerResult;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use zyntax_typed_ast::{
    typed_ast::{
        typed_node, TypedBlock, TypedExpression, TypedMatchArm, TypedNode, TypedPattern,
        TypedStatement,
    },
    InternedString, Span, Type,
};

/// Builder for creating CFG from TypedAST
pub struct TypedCfgBuilder {
    /// Next block ID to allocate
    next_block_id: u32,
    /// Stack of loop contexts for Break/Continue handling
    /// Each entry is (header_id, after_id) for the loop
    loop_stack: Vec<(HirId, HirId)>,
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
        }
    }

    /// Generate a new unique block ID
    fn new_block_id(&mut self) -> HirId {
        let id = HirId::new();
        self.next_block_id += 1;
        id
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
                        if edges.is_empty() {
                            None
                        } else {
                            Some(edges)
                        }
                    }
                    _ => None,
                }
            })
            .flatten()
            .collect();

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
        log::debug!(
            "[CFG] split_at_control_flow: entry_id={:?}, statements={}",
            entry_id,
            block.statements.len()
        );
        let mut all_blocks = Vec::new();
        let mut current_statements = Vec::new();
        let mut current_block_id = entry_id;
        let mut exit_id = entry_id;

        for (stmt_idx, stmt) in block.statements.iter().enumerate() {
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

                    // Detect `range(start, end)` or `range(start, end, step)` call patterns.
                    let range_args = match &for_stmt.iterator.node {
                        TypedExpression::Call(call) => {
                            let is_range = match &call.callee.node {
                                TypedExpression::Variable(name) => {
                                    let mut arena = zyntax_typed_ast::arena::AstArena::new();
                                    let range_sym = arena.intern_string("range");
                                    *name == range_sym
                                }
                                _ => false,
                            };
                            if is_range
                                && (call.positional_args.len() == 2
                                    || call.positional_args.len() == 3)
                            {
                                Some(call.positional_args.clone())
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    if let (Some(var_name), Some(args)) = (loop_var, range_args) {
                        // Desugar: for i in range(start, end) => C-style for loop
                        let span = for_stmt.iterator.span;
                        let start_expr = args[0].clone();
                        let end_expr = args[1].clone();
                        let step_expr = if args.len() == 3 {
                            Some(args[2].clone())
                        } else {
                            None
                        };

                        // Init: let mut i = start
                        let init_stmt = typed_node(
                            TypedStatement::Let(zyntax_typed_ast::typed_ast::TypedLet {
                                name: var_name,
                                ty: start_expr.ty.clone(),
                                mutability: zyntax_typed_ast::Mutability::Mutable,
                                initializer: Some(Box::new(start_expr.clone())),
                                span,
                            }),
                            Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
                            span,
                        );
                        current_statements.push(init_stmt);

                        // Condition: i < end
                        let cond_expr = typed_node(
                            TypedExpression::Binary(zyntax_typed_ast::typed_ast::TypedBinary {
                                op: zyntax_typed_ast::typed_ast::BinaryOp::Lt,
                                left: Box::new(typed_node(
                                    TypedExpression::Variable(var_name),
                                    start_expr.ty.clone(),
                                    span,
                                )),
                                right: Box::new(end_expr),
                            }),
                            Type::Primitive(zyntax_typed_ast::PrimitiveType::Bool),
                            span,
                        );

                        // Update: i = i + step (default step = 1)
                        let step = step_expr.unwrap_or_else(|| {
                            typed_node(
                                TypedExpression::Literal(
                                    zyntax_typed_ast::typed_ast::TypedLiteral::Integer(1),
                                ),
                                start_expr.ty.clone(),
                                span,
                            )
                        });
                        let update_expr = typed_node(
                            TypedExpression::Binary(zyntax_typed_ast::typed_ast::TypedBinary {
                                op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                                left: Box::new(typed_node(
                                    TypedExpression::Variable(var_name),
                                    start_expr.ty.clone(),
                                    span,
                                )),
                                right: Box::new(typed_node(
                                    TypedExpression::Binary(
                                        zyntax_typed_ast::typed_ast::TypedBinary {
                                            op: zyntax_typed_ast::typed_ast::BinaryOp::Add,
                                            left: Box::new(typed_node(
                                                TypedExpression::Variable(var_name),
                                                start_expr.ty.clone(),
                                                span,
                                            )),
                                            right: Box::new(step),
                                        },
                                    ),
                                    start_expr.ty.clone(),
                                    span,
                                )),
                            }),
                            start_expr.ty.clone(),
                            span,
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
                        // General for-each loop (iterator protocol)
                        // TODO: Implement general iterator desugaring
                        // For now, emit unconditional loop (matches previous behavior)
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
                            terminator: TypedTerminator::Jump(body_id),
                            pattern_check: None,
                        });

                        self.loop_stack.push((header_id, after_id));
                        let (body_blocks, _, body_exit) =
                            self.split_at_control_flow(&for_stmt.body, body_id, false)?;
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

                TypedStatement::Expression(expr) => {
                    // Check if expression is a Block - if so, flatten it
                    if let TypedExpression::Block(block) = &expr.node {
                        log::debug!(
                            "[CFG] Expression(Block): flattening block with {} statements",
                            block.statements.len()
                        );
                        // Recursively process the block's statements
                        let (block_blocks, _block_entry, block_exit) =
                            self.split_at_control_flow(block, current_block_id, false)?;
                        all_blocks.extend(block_blocks);
                        current_statements = Vec::new();
                        current_block_id = block_exit;
                        exit_id = block_exit;
                    } else {
                        // Regular expression - add as statement
                        current_statements.push(stmt.clone());
                    }
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
                        // Otherwise, wrap it in an Expression statement
                        let (body_stmts, body_terminator) = match &arm.body.node {
                            TypedExpression::Block(block) => {
                                // Extract statements from the block
                                // Last statement might be a return - if so, use it as terminator
                                if let Some(last_stmt) = block.statements.last() {
                                    if matches!(last_stmt.node, TypedStatement::Return(_)) {
                                        let stmts =
                                            block.statements[..block.statements.len() - 1].to_vec();
                                        let ret_stmt =
                                            &block.statements[block.statements.len() - 1];
                                        let term = if let TypedStatement::Return(ret_expr) =
                                            &ret_stmt.node
                                        {
                                            TypedTerminator::Return(ret_expr.clone())
                                        } else {
                                            TypedTerminator::Jump(merge_id)
                                        };
                                        (stmts, term)
                                    } else {
                                        (block.statements.clone(), TypedTerminator::Jump(merge_id))
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

                        all_blocks.push(TypedBasicBlock {
                            id: body_id,
                            label: None,
                            statements: body_stmts,
                            terminator: body_terminator,
                            pattern_check: body_pattern_info,
                        });

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
            let (final_statements, terminator) =
                if is_function_body && current_statements.len() == 1 {
                    if let TypedStatement::Expression(expr) = &current_statements[0].node {
                        // Single expression in function body - implicitly return it
                        (
                            vec![],
                            TypedTerminator::Return(Some(Box::new((**expr).clone()))),
                        )
                    } else {
                        (current_statements, TypedTerminator::Unreachable)
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

            // TODO: Handle custom enums/unions from type registry
            _ => None,
        }
    }
}

impl Default for TypedCfgBuilder {
    fn default() -> Self {
        Self::new()
    }
}
