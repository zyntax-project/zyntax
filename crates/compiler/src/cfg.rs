//! # Control Flow Graph Construction
//!
//! Builds CFG from TypedAST, preparing for SSA conversion.
//! The CFG is designed to be compatible with both Cranelift and LLVM backends.

use crate::hir::{HirBlock, HirFunction, HirId, HirInstruction, HirTerminator, HirType, HirValue};
use crate::CompilerResult;
use petgraph::algo::{all_simple_paths, dominators};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::{DfsPostOrder, Walker};
use std::collections::{HashMap, HashSet, VecDeque};
use zyntax_typed_ast::typed_ast::{TypedBlock, TypedExpression, TypedMatchArm, TypedStatement};
use zyntax_typed_ast::{InternedString, Span, TypedNode};

/// Control Flow Graph representation
#[derive(Debug)]
pub struct ControlFlowGraph {
    /// The underlying graph structure
    pub graph: DiGraph<BasicBlock, CfgEdge>,
    /// Entry node of the CFG
    pub entry: NodeIndex,
    /// Exit node of the CFG
    pub exit: NodeIndex,
    /// Mapping from block ID to graph node
    pub block_map: HashMap<HirId, NodeIndex>,
    /// Reverse mapping from graph node to block ID
    pub node_map: HashMap<NodeIndex, HirId>,
    /// Dominance tree
    pub dominance: Option<DominanceInfo>,
    /// Loop information
    pub loops: Vec<LoopInfo>,
}

/// Basic block in the CFG
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: HirId,
    pub label: Option<InternedString>,
    /// Statements in this block (before lowering to HIR)
    pub statements: Vec<TypedNode<TypedStatement>>,
    /// Terminator expression (if, return, etc.)
    pub terminator: Option<TypedNode<TypedStatement>>,
    /// Live variables at block entry
    pub live_in: HashSet<InternedString>,
    /// Live variables at block exit
    pub live_out: HashSet<InternedString>,
    /// Variables defined in this block
    pub defs: HashSet<InternedString>,
    /// Variables used in this block
    pub uses: HashSet<InternedString>,
    pub span: Span,
}

/// Edge between basic blocks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfgEdge {
    /// Unconditional branch
    Unconditional,
    /// True branch of conditional
    True,
    /// False branch of conditional
    False,
    /// Switch case
    Case(i64),
    /// Default case of switch
    Default,
    /// Exception/panic edge
    Exception,
}

/// Dominance information
#[derive(Debug)]
pub struct DominanceInfo {
    /// Immediate dominator of each node
    pub idom: HashMap<NodeIndex, NodeIndex>,
    /// Dominance tree (children of each node)
    pub dom_tree: HashMap<NodeIndex, Vec<NodeIndex>>,
    /// Dominance frontier for each node
    pub dom_frontier: HashMap<NodeIndex, HashSet<NodeIndex>>,
}

/// Loop information
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// Loop header block
    pub header: NodeIndex,
    /// Blocks in the loop
    pub blocks: HashSet<NodeIndex>,
    /// Back edges to the header
    pub back_edges: Vec<(NodeIndex, NodeIndex)>,
    /// Loop exit blocks
    pub exits: HashSet<NodeIndex>,
    /// Nested depth
    pub depth: u32,
    /// Parent loop (if nested)
    pub parent: Option<usize>,
}

/// CFG builder from TypedAST
pub struct CfgBuilder {
    graph: DiGraph<BasicBlock, CfgEdge>,
    current_block: Option<NodeIndex>,
    block_map: HashMap<HirId, NodeIndex>,
    node_map: HashMap<NodeIndex, HirId>,
    /// Stack of loop headers for break/continue
    loop_stack: Vec<(NodeIndex, NodeIndex)>, // (header, exit)
    /// Deferred edges (for forward jumps)
    deferred_edges: Vec<(NodeIndex, HirId, CfgEdge)>,
}

impl CfgBuilder {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            current_block: None,
            block_map: HashMap::new(),
            node_map: HashMap::new(),
            loop_stack: Vec::new(),
            deferred_edges: Vec::new(),
        }
    }

    /// Build CFG from a typed block
    pub fn build_from_block(&mut self, block: &TypedBlock) -> CompilerResult<ControlFlowGraph> {
        // Create entry block
        let entry_id = HirId::new();
        let entry_block = BasicBlock {
            id: entry_id,
            label: None, // Would need arena to intern strings
            statements: Vec::new(),
            terminator: None,
            live_in: HashSet::new(),
            live_out: HashSet::new(),
            defs: HashSet::new(),
            uses: HashSet::new(),
            span: block.span.clone(),
        };

        let entry = self.graph.add_node(entry_block);
        self.block_map.insert(entry_id, entry);
        self.node_map.insert(entry, entry_id);
        self.current_block = Some(entry);

        // Process statements
        for stmt in &block.statements {
            self.process_statement(stmt)?;
        }

        // Create exit block
        let exit_id = HirId::new();
        let exit_block = BasicBlock {
            id: exit_id,
            label: None, // Would need arena to intern strings
            statements: Vec::new(),
            terminator: None,
            live_in: HashSet::new(),
            live_out: HashSet::new(),
            defs: HashSet::new(),
            uses: HashSet::new(),
            span: block.span.clone(),
        };

        let exit = self.graph.add_node(exit_block);
        self.block_map.insert(exit_id, exit);
        self.node_map.insert(exit, exit_id);

        // Connect current block to exit if not already terminated
        if let Some(current) = self.current_block {
            let current_block = &self.graph[current];
            if current_block.terminator.is_none() {
                self.graph.add_edge(current, exit, CfgEdge::Unconditional);
            }
        }

        // Resolve deferred edges
        self.resolve_deferred_edges();

        // Build CFG structure
        let mut cfg = ControlFlowGraph {
            graph: std::mem::take(&mut self.graph),
            entry,
            exit,
            block_map: std::mem::take(&mut self.block_map),
            node_map: std::mem::take(&mut self.node_map),
            dominance: None,
            loops: Vec::new(),
        };

        // Compute dominance and loop information
        cfg.compute_dominance();
        cfg.detect_loops();
        cfg.compute_liveness();

        Ok(cfg)
    }

    /// Process a statement
    fn process_statement(&mut self, stmt: &TypedNode<TypedStatement>) -> CompilerResult<()> {
        match &stmt.node {
            TypedStatement::Expression(expr) => {
                self.add_statement_to_current_block(stmt.clone());
            }

            TypedStatement::Return(expr) => {
                self.add_statement_to_current_block(stmt.clone());
                self.terminate_current_block(stmt.clone());
            }

            TypedStatement::If(if_stmt) => {
                let condition = &if_stmt.condition;
                let then_block = &if_stmt.then_block;
                let else_block = if_stmt.else_block.as_ref();
                // Add condition evaluation to current block
                let cond_block = self.current_block.unwrap();
                self.terminate_current_block(stmt.clone());

                // Create then block
                let then_id = HirId::new();
                let then_node = self.create_block(then_id, None, then_block.span.clone());
                self.graph.add_edge(cond_block, then_node, CfgEdge::True);

                // Process then branch
                self.current_block = Some(then_node);
                self.process_block(then_block)?;
                let then_exit = self.current_block.unwrap();

                // Create merge block
                let merge_id = HirId::new();
                let merge_node = self.create_block(merge_id, None, stmt.span.clone());

                if let Some(else_block) = else_block {
                    // Create else block
                    let else_id = HirId::new();
                    let else_node = self.create_block(else_id, None, else_block.span.clone());
                    self.graph.add_edge(cond_block, else_node, CfgEdge::False);

                    // Process else branch
                    self.current_block = Some(else_node);
                    self.process_block(else_block)?;
                    let else_exit = self.current_block.unwrap();

                    // Connect both branches to merge
                    self.graph
                        .add_edge(then_exit, merge_node, CfgEdge::Unconditional);
                    self.graph
                        .add_edge(else_exit, merge_node, CfgEdge::Unconditional);
                } else {
                    // No else branch - connect condition directly to merge
                    self.graph.add_edge(cond_block, merge_node, CfgEdge::False);
                    self.graph
                        .add_edge(then_exit, merge_node, CfgEdge::Unconditional);
                }

                self.current_block = Some(merge_node);
            }

            TypedStatement::While(while_stmt) => {
                let condition = &while_stmt.condition;
                let body = &while_stmt.body;
                // Create loop header
                let header_id = HirId::new();
                let header_node = self.create_block(header_id, None, stmt.span.clone());

                // Connect current to header
                if let Some(current) = self.current_block {
                    self.graph
                        .add_edge(current, header_node, CfgEdge::Unconditional);
                }

                // Create loop body
                let body_id = HirId::new();
                let body_node = self.create_block(body_id, None, body.span.clone());
                self.graph.add_edge(header_node, body_node, CfgEdge::True);

                // Create loop exit
                let exit_id = HirId::new();
                let exit_node = self.create_block(exit_id, None, stmt.span.clone());
                self.graph.add_edge(header_node, exit_node, CfgEdge::False);

                // Process loop body
                self.loop_stack.push((header_node, exit_node));
                self.current_block = Some(body_node);
                self.process_block(body)?;
                let body_exit = self.current_block.unwrap();
                self.loop_stack.pop();

                // Back edge from body to header
                self.graph
                    .add_edge(body_exit, header_node, CfgEdge::Unconditional);

                self.current_block = Some(exit_node);
            }

            TypedStatement::For(for_stmt) => {
                let pattern = &for_stmt.pattern;
                let iterator = &for_stmt.iterator;
                let body = &for_stmt.body;
                // Similar to while loop but with iterator initialization
                self.process_for_loop(pattern, iterator, body, stmt.span.clone())?;
            }

            TypedStatement::Match(match_stmt) => {
                let expr = &match_stmt.scrutinee;
                let arms = &match_stmt.arms;
                self.process_match(expr, arms, stmt.span.clone())?;
            }

            TypedStatement::Block(block) => {
                self.process_block(block)?;
            }

            TypedStatement::Break(_) => {
                if let Some((_, exit)) = self.loop_stack.last() {
                    self.graph
                        .add_edge(self.current_block.unwrap(), *exit, CfgEdge::Unconditional);
                    // Create unreachable block for any code after break
                    let unreachable_id = HirId::new();
                    let unreachable_node =
                        self.create_block(unreachable_id, None, stmt.span.clone());
                    self.current_block = Some(unreachable_node);
                }
            }

            TypedStatement::Continue => {
                if let Some((header, _)) = self.loop_stack.last() {
                    self.graph.add_edge(
                        self.current_block.unwrap(),
                        *header,
                        CfgEdge::Unconditional,
                    );
                    // Create unreachable block for any code after continue
                    let unreachable_id = HirId::new();
                    let unreachable_node = self.create_block(
                        unreachable_id,
                        None, // InternedString doesn't have From<&str>
                        stmt.span.clone(),
                    );
                    self.current_block = Some(unreachable_node);
                }
            }

            _ => {
                // Other statements just get added to current block
                self.add_statement_to_current_block(stmt.clone());
            }
        }

        Ok(())
    }

    /// Process a block of statements
    fn process_block(&mut self, block: &TypedBlock) -> CompilerResult<()> {
        for stmt in &block.statements {
            self.process_statement(stmt)?;
        }
        Ok(())
    }

    /// Process a for loop
    fn process_for_loop(
        &mut self,
        pattern: &TypedNode<zyntax_typed_ast::typed_ast::TypedPattern>,
        iterator: &TypedNode<TypedExpression>,
        body: &TypedBlock,
        span: Span,
    ) -> CompilerResult<()> {
        // Create initialization block for iterator
        let init_id = HirId::new();
        let init_node = self.create_block(init_id, None, span.clone());
        if let Some(current) = self.current_block {
            self.graph
                .add_edge(current, init_node, CfgEdge::Unconditional);
        }

        // Create loop header
        let header_id = HirId::new();
        let header_node = self.create_block(header_id, None, span.clone());
        self.graph
            .add_edge(init_node, header_node, CfgEdge::Unconditional);

        // Create loop body
        let body_id = HirId::new();
        let body_node = self.create_block(body_id, None, body.span.clone());
        self.graph.add_edge(header_node, body_node, CfgEdge::True);

        // Create loop exit
        let exit_id = HirId::new();
        let exit_node = self.create_block(exit_id, None, span);
        self.graph.add_edge(header_node, exit_node, CfgEdge::False);

        // Process loop body
        self.loop_stack.push((header_node, exit_node));
        self.current_block = Some(body_node);
        self.process_block(body)?;
        let body_exit = self.current_block.unwrap();
        self.loop_stack.pop();

        // Back edge from body to header
        self.graph
            .add_edge(body_exit, header_node, CfgEdge::Unconditional);

        self.current_block = Some(exit_node);
        Ok(())
    }

    /// Process a match expression
    fn process_match(
        &mut self,
        expr: &TypedNode<TypedExpression>,
        arms: &[TypedMatchArm],
        span: Span,
    ) -> CompilerResult<()> {
        let match_block = self.current_block.unwrap();

        // Create merge block
        let merge_id = HirId::new();
        let merge_node = self.create_block(merge_id, None, span);

        // Process each arm
        for (i, arm) in arms.iter().enumerate() {
            let arm_id = HirId::new();
            let _span = arm.body.span.clone();
            let arm_node = self.create_block(arm_id, None, _span.clone());

            // Add edge from match block to arm
            self.graph
                .add_edge(match_block, arm_node, CfgEdge::Case(i as i64));

            // Process arm body
            self.current_block = Some(arm_node);
            // Create a statement from the expression body
            let body_stmt = TypedNode::new(
                TypedStatement::Expression(Box::new(arm.body.as_ref().clone())),
                arm.body.ty.clone(),
                arm.body.span.clone(),
            );
            self.process_statement(&body_stmt)?;

            // Connect to merge block
            if let Some(current) = self.current_block {
                self.graph
                    .add_edge(current, merge_node, CfgEdge::Unconditional);
            }
        }

        self.current_block = Some(merge_node);
        Ok(())
    }

    /// Create a new basic block
    fn create_block(&mut self, id: HirId, label: Option<InternedString>, span: Span) -> NodeIndex {
        let block = BasicBlock {
            id,
            label,
            statements: Vec::new(),
            terminator: None,
            live_in: HashSet::new(),
            live_out: HashSet::new(),
            defs: HashSet::new(),
            uses: HashSet::new(),
            span,
        };

        let node = self.graph.add_node(block);
        self.block_map.insert(id, node);
        self.node_map.insert(node, id);
        node
    }

    /// Add statement to current block
    fn add_statement_to_current_block(&mut self, stmt: TypedNode<TypedStatement>) {
        if let Some(current) = self.current_block {
            self.graph[current].statements.push(stmt);
        }
    }

    /// Terminate current block
    fn terminate_current_block(&mut self, terminator: TypedNode<TypedStatement>) {
        if let Some(current) = self.current_block {
            self.graph[current].terminator = Some(terminator);
        }
    }

    /// Resolve deferred edges
    fn resolve_deferred_edges(&mut self) {
        for (from, to_id, edge_type) in &self.deferred_edges {
            if let Some(&to) = self.block_map.get(to_id) {
                self.graph.add_edge(*from, to, *edge_type);
            }
        }
    }
}

impl ControlFlowGraph {
    /// Compute dominance information
    pub fn compute_dominance(&mut self) {
        let doms = dominators::simple_fast(&self.graph, self.entry);
        let mut idom = HashMap::new();
        let mut dom_tree: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();

        // Build immediate dominator map and dominance tree
        for node in self.graph.node_indices() {
            if node != self.entry {
                if let Some(dom) = doms.immediate_dominator(node) {
                    idom.insert(node, dom);
                    dom_tree.entry(dom).or_insert_with(Vec::new).push(node);
                }
            }
        }

        // Compute dominance frontier
        let dom_frontier = self.compute_dominance_frontier(&idom);

        self.dominance = Some(DominanceInfo {
            idom,
            dom_tree,
            dom_frontier,
        });
    }

    /// Compute dominance frontier for SSA construction
    fn compute_dominance_frontier(
        &self,
        idom: &HashMap<NodeIndex, NodeIndex>,
    ) -> HashMap<NodeIndex, HashSet<NodeIndex>> {
        let mut df: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();

        for node in self.graph.node_indices() {
            df.insert(node, HashSet::new());
        }

        // For each edge X -> Y
        for edge in self.graph.edge_indices() {
            let (x, y) = self.graph.edge_endpoints(edge).unwrap();

            // Walk up dominator tree from X
            let mut runner = x;
            while runner != self.entry {
                if let Some(&dom_y) = idom.get(&y) {
                    if runner == dom_y {
                        break;
                    }
                }

                // Y is in dominance frontier of runner
                df.get_mut(&runner).unwrap().insert(y);

                // Move up dominator tree
                if let Some(&next) = idom.get(&runner) {
                    runner = next;
                } else {
                    break;
                }
            }
        }

        df
    }

    /// Detect natural loops
    pub fn detect_loops(&mut self) {
        let mut loops = Vec::new();
        let mut back_edges = Vec::new();

        // Find back edges using DFS
        let mut dfs = DfsPostOrder::new(&self.graph, self.entry);
        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();

        self.find_back_edges(self.entry, &mut visited, &mut on_stack, &mut back_edges);

        // For each back edge, find the natural loop
        for (tail, head) in back_edges {
            let mut loop_blocks = HashSet::new();
            loop_blocks.insert(head);
            loop_blocks.insert(tail);

            // Find all blocks in the loop
            let mut work_list = vec![tail];
            while let Some(block) = work_list.pop() {
                for pred in self
                    .graph
                    .neighbors_directed(block, petgraph::Direction::Incoming)
                {
                    if !loop_blocks.contains(&pred) {
                        loop_blocks.insert(pred);
                        work_list.push(pred);
                    }
                }
            }

            // Find loop exits
            let mut exits = HashSet::new();
            for &block in &loop_blocks {
                for succ in self.graph.neighbors(block) {
                    if !loop_blocks.contains(&succ) {
                        exits.insert(succ);
                    }
                }
            }

            loops.push(LoopInfo {
                header: head,
                blocks: loop_blocks,
                back_edges: vec![(tail, head)],
                exits,
                depth: 0, // Will be computed later
                parent: None,
            });
        }

        // Compute loop nesting
        self.compute_loop_nesting(&mut loops);

        self.loops = loops;
    }

    /// Find back edges using DFS
    fn find_back_edges(
        &self,
        node: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        on_stack: &mut HashSet<NodeIndex>,
        back_edges: &mut Vec<(NodeIndex, NodeIndex)>,
    ) {
        visited.insert(node);
        on_stack.insert(node);

        for succ in self.graph.neighbors(node) {
            if !visited.contains(&succ) {
                self.find_back_edges(succ, visited, on_stack, back_edges);
            } else if on_stack.contains(&succ) {
                // Back edge found
                back_edges.push((node, succ));
            }
        }

        on_stack.remove(&node);
    }

    /// Compute loop nesting information
    fn compute_loop_nesting(&self, loops: &mut Vec<LoopInfo>) {
        // Check containment for each pair of loops
        for i in 0..loops.len() {
            for j in 0..loops.len() {
                if i != j {
                    let loop_i = &loops[i];
                    let loop_j = &loops[j];

                    // Check if loop j is nested in loop i
                    if loop_i.blocks.contains(&loop_j.header) {
                        loops[j].parent = Some(i);
                        loops[j].depth = loops[i].depth + 1;
                    }
                }
            }
        }
    }

    /// Compute liveness analysis
    pub fn compute_liveness(&mut self) {
        // Initialize live sets
        for node in self.graph.node_indices() {
            let block = &mut self.graph[node];

            // Compute uses and defs for this block
            Self::compute_uses_defs(block);
        }

        // Fixed-point iteration for liveness
        let mut changed = true;
        while changed {
            changed = false;

            // Process blocks in reverse postorder
            let mut rpo: Vec<_> = DfsPostOrder::new(&self.graph, self.entry)
                .iter(&self.graph)
                .collect();
            rpo.reverse();

            for node in rpo {
                let mut new_live_out = HashSet::new();

                // Union of live_in sets of successors
                for succ in self.graph.neighbors(node) {
                    let succ_live_in = &self.graph[succ].live_in;
                    new_live_out.extend(succ_live_in);
                }

                // Compute new live_in: (live_out - defs) ∪ uses
                let block = &self.graph[node];
                let mut new_live_in: HashSet<_> =
                    new_live_out.difference(&block.defs).cloned().collect();
                new_live_in.extend(&block.uses);

                // Update if changed
                let block = &mut self.graph[node];
                if new_live_in != block.live_in || new_live_out != block.live_out {
                    block.live_in = new_live_in;
                    block.live_out = new_live_out;
                    changed = true;
                }
            }
        }
    }

    /// Compute uses and defs for a basic block
    fn compute_uses_defs(block: &mut BasicBlock) {
        // This is a simplified version - real implementation would analyze expressions
        for stmt in &block.statements {
            match &stmt.node {
                TypedStatement::Let(let_stmt) => {
                    // Let defines a variable
                    block.defs.insert(let_stmt.name);
                    // Initializer uses variables
                    if let Some(init) = &let_stmt.initializer {
                        Self::collect_expr_uses(init, &mut block.uses);
                    }
                }
                TypedStatement::Expression(expr) => {
                    Self::collect_expr_uses(expr, &mut block.uses);
                }
                _ => {}
            }
        }

        // Remove defs from uses (can't use what you define)
        block.uses = block.uses.difference(&block.defs).cloned().collect();
    }

    /// Collect variable uses from an expression
    fn collect_expr_uses(expr: &TypedNode<TypedExpression>, uses: &mut HashSet<InternedString>) {
        match &expr.node {
            TypedExpression::Variable(name) => {
                uses.insert(*name);
            }
            TypedExpression::Binary(binary) => {
                let left = &binary.left;
                let right = &binary.right;
                Self::collect_expr_uses(left, uses);
                Self::collect_expr_uses(right, uses);
            }
            TypedExpression::Unary(unary) => {
                let operand = &unary.operand;
                Self::collect_expr_uses(operand, uses);
            }
            TypedExpression::Call(call) => {
                let callee = &call.callee;
                let args = &call.positional_args;
                Self::collect_expr_uses(callee, uses);
                for arg in args {
                    Self::collect_expr_uses(arg, uses);
                }
            }
            TypedExpression::Field(field_access) => {
                let object = &field_access.object;
                Self::collect_expr_uses(object, uses);
            }
            TypedExpression::Index(index_expr) => {
                let object = &index_expr.object;
                let index = &index_expr.index;
                Self::collect_expr_uses(object, uses);
                Self::collect_expr_uses(index, uses);
            }
            _ => {} // Other expressions don't directly use variables
        }
    }

    /// Convert to HIR function
    pub fn to_hir_function(
        &self,
        name: InternedString,
        sig: crate::hir::HirFunctionSignature,
    ) -> HirFunction {
        let mut func = HirFunction::new(name, sig);

        // Convert each basic block to HIR block
        for node in self.graph.node_indices() {
            let cfg_block = &self.graph[node];
            let hir_block_id = cfg_block.id;

            // Get or create HIR block
            let hir_block = func.blocks.get_mut(&hir_block_id).unwrap();
            hir_block.label = cfg_block.label;

            // Set predecessors and successors
            for pred in self
                .graph
                .neighbors_directed(node, petgraph::Direction::Incoming)
            {
                hir_block.predecessors.push(self.graph[pred].id);
            }
            for succ in self.graph.neighbors(node) {
                hir_block.successors.push(self.graph[succ].id);
            }

            // Set dominance frontier
            if let Some(ref dom_info) = self.dominance {
                if let Some(df) = dom_info.dom_frontier.get(&node) {
                    for &df_node in df {
                        hir_block.dominance_frontier.insert(self.graph[df_node].id);
                    }
                }
            }
        }

        func
    }
}
