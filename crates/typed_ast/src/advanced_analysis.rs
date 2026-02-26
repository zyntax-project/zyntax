//! # Advanced Analysis Framework
//!
//! Comprehensive analysis system for TypedAST including:
//! - Data Flow Graph (DFG) construction and analysis
//! - Control Flow Graph (CFG) mapping and optimization
//! - Ownership analysis for memory safety
//! - Lifetime analysis for borrow checking
//! - TypedAST lowering to analysis-friendly IR

use crate::arena::InternedString;
use crate::source::Span;
use crate::type_registry::{Lifetime, Mutability, Type};
use crate::typed_ast::*;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;

// ====== CORE ANALYSIS INFRASTRUCTURE ======

/// Unique identifier for analysis nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AnalysisNodeId(u32);

impl AnalysisNodeId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for AnalysisNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Analysis context that coordinates all analysis phases
pub struct AnalysisContext {
    /// DFG analysis
    pub dfg: DataFlowGraph,
    /// CFG analysis  
    pub cfg: ControlFlowGraph,
    /// Ownership analysis
    pub ownership: OwnershipAnalysis,
    /// Lifetime analysis
    pub lifetime: LifetimeAnalysis,
    /// Next available node ID
    next_id: u32,
    /// Mapping from TypedAST nodes to analysis nodes
    node_mapping: HashMap<Span, AnalysisNodeId>,
    /// Current analysis phase
    phase: AnalysisPhase,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalysisPhase {
    /// Initial phase: convert TypedAST to analysis IR
    Lowering,
    /// Build control flow graph
    ControlFlow,
    /// Build data flow graph
    DataFlow,
    /// Perform ownership analysis
    Ownership,
    /// Perform lifetime analysis
    Lifetime,
    /// Final optimizations and verification
    Optimization,
}

impl AnalysisContext {
    pub fn new() -> Self {
        Self {
            dfg: DataFlowGraph::new(),
            cfg: ControlFlowGraph::new(),
            ownership: OwnershipAnalysis::new(),
            lifetime: LifetimeAnalysis::new(),
            next_id: 0,
            node_mapping: HashMap::new(),
            phase: AnalysisPhase::Lowering,
        }
    }

    fn next_node_id(&mut self) -> AnalysisNodeId {
        let id = AnalysisNodeId::new(self.next_id);
        self.next_id += 1;
        id
    }

    /// Complete analysis pipeline for a TypedAST program
    pub fn analyze_program(
        &mut self,
        program: &TypedProgram,
    ) -> Result<AnalysisResult, AnalysisError> {
        // Phase 1: Lower TypedAST to analysis IR
        self.phase = AnalysisPhase::Lowering;
        self.lower_program(program)?;

        // Phase 2: Build control flow graph
        self.phase = AnalysisPhase::ControlFlow;
        self.build_control_flow()?;

        // Phase 3: Build data flow graph
        self.phase = AnalysisPhase::DataFlow;
        self.build_data_flow()?;

        // Phase 4: Ownership analysis
        self.phase = AnalysisPhase::Ownership;
        self.analyze_ownership()?;

        // Phase 5: Lifetime analysis
        self.phase = AnalysisPhase::Lifetime;
        self.analyze_lifetimes()?;

        // Phase 6: Optimizations
        self.phase = AnalysisPhase::Optimization;
        self.optimize()?;

        Ok(AnalysisResult {
            cfg: self.cfg.clone(),
            dfg: self.dfg.clone(),
            ownership: self.ownership.clone(),
            lifetime: self.lifetime.clone(),
        })
    }
}

// ====== DATA FLOW GRAPH (DFG) ======

/// Data Flow Graph for tracking value flow and dependencies
#[derive(Debug, Clone)]
pub struct DataFlowGraph {
    /// All data flow nodes
    pub nodes: HashMap<AnalysisNodeId, DFGNode>,
    /// Edges representing data dependencies
    pub edges: HashMap<AnalysisNodeId, Vec<DFGEdge>>,
    /// Entry points (function parameters, global variables)
    pub entry_points: Vec<AnalysisNodeId>,
    /// Exit points (return statements, function end)
    pub exit_points: Vec<AnalysisNodeId>,
    /// Dominance frontiers for SSA form
    pub dominance_frontiers: HashMap<AnalysisNodeId, HashSet<AnalysisNodeId>>,
}

#[derive(Debug, Clone)]
pub struct DFGNode {
    pub id: AnalysisNodeId,
    pub kind: DFGNodeKind,
    pub ty: Type,
    pub span: Span,
    /// Variables defined by this node
    pub defines: HashSet<InternedString>,
    /// Variables used by this node
    pub uses: HashSet<InternedString>,
    /// Live variables at this point
    pub live_vars: HashSet<InternedString>,
}

#[derive(Debug, Clone)]
pub enum DFGNodeKind {
    /// Variable assignment
    Assignment {
        target: InternedString,
        source: AnalysisNodeId,
    },
    /// Function call
    Call {
        callee: AnalysisNodeId,
        args: Vec<AnalysisNodeId>,
        returns: Option<InternedString>,
    },
    /// Literal value
    Literal { value: DFGLiteral },
    /// Variable reference
    Variable { name: InternedString },
    /// Binary operation
    BinaryOp {
        op: BinaryOp,
        left: AnalysisNodeId,
        right: AnalysisNodeId,
    },
    /// Unary operation
    UnaryOp {
        op: UnaryOp,
        operand: AnalysisNodeId,
    },
    /// Field access
    FieldAccess {
        object: AnalysisNodeId,
        field: InternedString,
    },
    /// Array/pointer dereference
    Dereference { target: AnalysisNodeId },
    /// Address-of operation
    AddressOf {
        target: AnalysisNodeId,
        mutability: Mutability,
    },
    /// Phi node for SSA form
    Phi {
        inputs: Vec<(AnalysisNodeId, AnalysisNodeId)>,
    }, // (value, block)
}

#[derive(Debug, Clone)]
pub enum DFGLiteral {
    Integer(i128),
    Float(f64),
    Bool(bool),
    String(InternedString),
    Char(char),
    Unit,
}

#[derive(Debug, Clone)]
pub struct DFGEdge {
    pub target: AnalysisNodeId,
    pub kind: DFGEdgeKind,
}

#[derive(Debug, Clone)]
pub enum DFGEdgeKind {
    /// Data dependency (value flows from source to target)
    DataDependency,
    /// Control dependency (execution depends on condition)
    ControlDependency,
    /// Memory dependency (memory access ordering)
    MemoryDependency,
    /// Anti-dependency (write after read)
    AntiDependency,
    /// Output dependency (write after write)
    OutputDependency,
}

impl DataFlowGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            entry_points: Vec::new(),
            exit_points: Vec::new(),
            dominance_frontiers: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: DFGNode) {
        self.nodes.insert(node.id, node);
    }

    pub fn add_edge(&mut self, from: AnalysisNodeId, edge: DFGEdge) {
        self.edges.entry(from).or_default().push(edge);
    }

    /// Compute dominance frontiers for SSA construction
    pub fn compute_dominance_frontiers(&mut self) {
        // Implementation of dominance frontier algorithm
        // This is a simplified version - full implementation would use
        // Lengauer-Tarjan algorithm for efficiency

        for &node_id in self.nodes.keys() {
            let mut frontier = HashSet::new();

            // Find all nodes that this node dominates
            for &other_id in self.nodes.keys() {
                if node_id != other_id && self.dominates(node_id, other_id) {
                    // Check if there's a successor not dominated by node_id
                    if let Some(edges) = self.edges.get(&other_id) {
                        for edge in edges {
                            if !self.dominates(node_id, edge.target) {
                                frontier.insert(edge.target);
                            }
                        }
                    }
                }
            }

            self.dominance_frontiers.insert(node_id, frontier);
        }
    }

    /// Check if node 'a' dominates node 'b'
    fn dominates(&self, a: AnalysisNodeId, b: AnalysisNodeId) -> bool {
        // Simplified dominance check
        // In practice, this would use pre-computed dominance trees
        if a == b {
            return true;
        }

        // For now, just check if 'a' is reachable from all entry points to 'b'
        self.entry_points
            .iter()
            .all(|&entry| self.path_exists_through(entry, b, a))
    }

    fn path_exists_through(
        &self,
        start: AnalysisNodeId,
        end: AnalysisNodeId,
        through: AnalysisNodeId,
    ) -> bool {
        if start == end {
            return start == through;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current) {
                continue;
            }

            if current == end {
                return true;
            }

            if current != through && current != start {
                continue; // Must go through 'through' node
            }

            if let Some(edges) = self.edges.get(&current) {
                for edge in edges {
                    queue.push_back(edge.target);
                }
            }
        }

        false
    }

    /// Convert to SSA form
    pub fn convert_to_ssa(&mut self) {
        self.compute_dominance_frontiers();

        // Insert phi nodes at dominance frontiers
        let mut phi_nodes = HashMap::new();

        for (&node_id, frontier) in &self.dominance_frontiers {
            if let Some(node) = self.nodes.get(&node_id) {
                for &var in &node.defines {
                    for &frontier_node in frontier {
                        // Insert phi node for this variable
                        let phi_id =
                            AnalysisNodeId::new(self.nodes.len() as u32 + phi_nodes.len() as u32);
                        let phi = DFGNode {
                            id: phi_id,
                            kind: DFGNodeKind::Phi { inputs: Vec::new() },
                            ty: node.ty.clone(),
                            span: node.span,
                            defines: [var].into_iter().collect(),
                            uses: HashSet::new(),
                            live_vars: HashSet::new(),
                        };
                        phi_nodes.insert(phi_id, phi);
                    }
                }
            }
        }

        // Add phi nodes to the graph
        for (id, phi) in phi_nodes {
            self.nodes.insert(id, phi);
        }
    }

    /// Perform live variable analysis
    pub fn compute_liveness(&mut self) {
        let mut changed = true;

        while changed {
            changed = false;

            // Process nodes in reverse postorder
            let node_ids: Vec<_> = self.nodes.keys().cloned().collect();
            for node_id in node_ids {
                let mut new_live_vars = HashSet::new();

                // Live_out = union of Live_in of all successors
                if let Some(edges) = self.edges.get(&node_id) {
                    for edge in edges {
                        if let Some(successor) = self.nodes.get(&edge.target) {
                            new_live_vars.extend(&successor.live_vars);
                        }
                    }
                }

                // Live_in = (Live_out - Def) union Use
                if let Some(node) = self.nodes.get(&node_id) {
                    for var in &new_live_vars {
                        if !node.defines.contains(var) {
                            // Variable is live in if it's live out and not defined here
                        }
                    }
                    new_live_vars.extend(&node.uses);
                }

                // Update if changed
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    if node.live_vars != new_live_vars {
                        node.live_vars = new_live_vars;
                        changed = true;
                    }
                }
            }
        }
    }
}

// ====== CONTROL FLOW GRAPH (CFG) ======

/// Control Flow Graph for analyzing program control flow
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// Basic blocks
    pub blocks: HashMap<AnalysisNodeId, CFGBlock>,
    /// Control flow edges
    pub edges: HashMap<AnalysisNodeId, Vec<CFGEdge>>,
    /// Entry block
    pub entry: Option<AnalysisNodeId>,
    /// Exit blocks
    pub exits: Vec<AnalysisNodeId>,
    /// Loop information
    pub loops: Vec<LoopInfo>,
}

#[derive(Debug, Clone)]
pub struct CFGBlock {
    pub id: AnalysisNodeId,
    /// Statements in this block
    pub statements: Vec<AnalysisNodeId>,
    /// Terminator (branch, return, etc.)
    pub terminator: Option<CFGTerminator>,
    /// Dominance information
    pub dominators: HashSet<AnalysisNodeId>,
    /// Post-dominance information
    pub post_dominators: HashSet<AnalysisNodeId>,
}

#[derive(Debug, Clone)]
pub enum CFGTerminator {
    /// Unconditional branch
    Branch { target: AnalysisNodeId },
    /// Conditional branch
    ConditionalBranch {
        condition: AnalysisNodeId,
        then_block: AnalysisNodeId,
        else_block: AnalysisNodeId,
    },
    /// Function return
    Return { value: Option<AnalysisNodeId> },
    /// Switch/match statement
    Switch {
        discriminant: AnalysisNodeId,
        cases: Vec<(AnalysisNodeId, AnalysisNodeId)>, // (value, target)
        default: Option<AnalysisNodeId>,
    },
    /// Unreachable code
    Unreachable,
}

#[derive(Debug, Clone)]
pub struct CFGEdge {
    pub target: AnalysisNodeId,
    pub kind: CFGEdgeKind,
}

#[derive(Debug, Clone)]
pub enum CFGEdgeKind {
    /// Normal control flow
    Normal,
    /// True branch of conditional
    True,
    /// False branch of conditional
    False,
    /// Exception handling
    Exception,
    /// Back edge (loop)
    BackEdge,
}

#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// Loop header block
    pub header: AnalysisNodeId,
    /// All blocks in the loop
    pub blocks: HashSet<AnalysisNodeId>,
    /// Loop exit blocks
    pub exits: Vec<AnalysisNodeId>,
    /// Nested loops
    pub nested_loops: Vec<LoopInfo>,
    /// Loop kind
    pub kind: LoopKind,
}

#[derive(Debug, Clone)]
pub enum LoopKind {
    /// While loop
    While,
    /// For loop
    For,
    /// Do-while loop
    DoWhile,
    /// Infinite loop
    Infinite,
}

impl ControlFlowGraph {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            edges: HashMap::new(),
            entry: None,
            exits: Vec::new(),
            loops: Vec::new(),
        }
    }

    pub fn add_block(&mut self, block: CFGBlock) {
        self.blocks.insert(block.id, block);
    }

    pub fn add_edge(&mut self, from: AnalysisNodeId, edge: CFGEdge) {
        self.edges.entry(from).or_default().push(edge);
    }

    /// Detect natural loops in the CFG
    pub fn detect_loops(&mut self) {
        let mut loops = Vec::new();
        let mut visited = HashSet::new();

        if let Some(entry) = self.entry {
            self.detect_loops_dfs(entry, &mut visited, &mut Vec::new(), &mut loops);
        }

        self.loops = loops;
    }

    fn detect_loops_dfs(
        &self,
        current: AnalysisNodeId,
        visited: &mut HashSet<AnalysisNodeId>,
        path: &mut Vec<AnalysisNodeId>,
        loops: &mut Vec<LoopInfo>,
    ) {
        if path.contains(&current) {
            // Found a back edge - this indicates a loop
            let loop_start_idx = path.iter().position(|&x| x == current).unwrap();
            let loop_blocks: HashSet<_> = path[loop_start_idx..].iter().copied().collect();

            let loop_info = LoopInfo {
                header: current,
                blocks: loop_blocks,
                exits: Vec::new(), // Would need to compute properly
                nested_loops: Vec::new(),
                kind: LoopKind::While, // Would need to infer from structure
            };

            loops.push(loop_info);
            return;
        }

        if !visited.insert(current) {
            return;
        }

        path.push(current);

        if let Some(edges) = self.edges.get(&current) {
            for edge in edges {
                self.detect_loops_dfs(edge.target, visited, path, loops);
            }
        }

        path.pop();
    }

    /// Compute dominance relationships
    pub fn compute_dominance(&mut self) {
        // Simplified dominance computation
        // Full implementation would use Lengauer-Tarjan algorithm

        let block_ids: Vec<_> = self.blocks.keys().cloned().collect();
        for block_id in block_ids {
            let mut dominators = HashSet::new();

            if Some(block_id) == self.entry {
                dominators.insert(block_id);
            } else {
                // A block is dominated by the intersection of dominators of all predecessors
                let predecessors: Vec<_> = self
                    .edges
                    .iter()
                    .filter_map(|(from, edges)| {
                        edges.iter().find(|e| e.target == block_id).map(|_| *from)
                    })
                    .collect();

                if !predecessors.is_empty() {
                    // Start with dominators of first predecessor
                    if let Some(first_block) = self.blocks.get(&predecessors[0]) {
                        dominators = first_block.dominators.clone();
                    }

                    // Intersect with dominators of other predecessors
                    for &pred in &predecessors[1..] {
                        if let Some(pred_block) = self.blocks.get(&pred) {
                            dominators = dominators
                                .intersection(&pred_block.dominators)
                                .copied()
                                .collect();
                        }
                    }
                }

                dominators.insert(block_id);
            }

            if let Some(block) = self.blocks.get_mut(&block_id) {
                block.dominators = dominators;
            }
        }
    }
}

// ====== OWNERSHIP ANALYSIS ======

/// Ownership analysis for memory safety
#[derive(Debug, Clone)]
pub struct OwnershipAnalysis {
    /// Ownership information for each variable
    pub ownership_map: HashMap<InternedString, OwnershipInfo>,
    /// Borrow relationships
    pub borrows: HashMap<InternedString, Vec<BorrowInfo>>,
    /// Move operations
    pub moves: Vec<MoveInfo>,
    /// Drop points
    pub drops: Vec<DropInfo>,
}

#[derive(Debug, Clone)]
pub struct OwnershipInfo {
    /// Variable name
    pub variable: InternedString,
    /// Ownership kind
    pub kind: OwnershipKind,
    /// Type being owned
    pub ty: Type,
    /// Span where ownership is established
    pub span: Span,
    /// Whether this is a mutable owner
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub enum OwnershipKind {
    /// Full ownership
    Owned,
    /// Shared reference
    SharedRef,
    /// Mutable reference
    MutableRef,
    /// Weak reference (doesn't affect ownership)
    Weak,
    /// Moved (no longer valid)
    Moved,
}

#[derive(Debug, Clone)]
pub struct BorrowInfo {
    /// The borrowed variable
    pub borrowed: InternedString,
    /// The borrower
    pub borrower: InternedString,
    /// Borrow kind
    pub kind: BorrowKind,
    /// Lifetime of the borrow
    pub lifetime: Option<Lifetime>,
    /// Span of the borrow
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum BorrowKind {
    /// Shared borrow (&T)
    Shared,
    /// Mutable borrow (&mut T)
    Mutable,
}

#[derive(Debug, Clone)]
pub struct MoveInfo {
    /// Variable being moved
    pub variable: InternedString,
    /// Destination of the move
    pub destination: Option<InternedString>,
    /// Span of the move
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct DropInfo {
    /// Variable being dropped
    pub variable: InternedString,
    /// Span where drop occurs
    pub span: Span,
    /// Whether this is an explicit drop
    pub explicit: bool,
}

impl OwnershipAnalysis {
    pub fn new() -> Self {
        Self {
            ownership_map: HashMap::new(),
            borrows: HashMap::new(),
            moves: Vec::new(),
            drops: Vec::new(),
        }
    }

    /// Analyze ownership for a variable assignment
    pub fn analyze_assignment(
        &mut self,
        target: InternedString,
        source: &DFGNode,
    ) -> Result<(), OwnershipError> {
        match &source.kind {
            DFGNodeKind::Variable { name } => {
                // Check if source is being moved
                if let Some(ownership) = self.ownership_map.get(name) {
                    match ownership.kind {
                        OwnershipKind::Owned => {
                            // Move the ownership
                            self.moves.push(MoveInfo {
                                variable: *name,
                                destination: Some(target),
                                span: source.span,
                            });

                            // Update ownership
                            let mut new_ownership = ownership.clone();
                            new_ownership.variable = target;
                            self.ownership_map.insert(target, new_ownership);

                            // Mark source as moved
                            if let Some(source_ownership) = self.ownership_map.get_mut(name) {
                                source_ownership.kind = OwnershipKind::Moved;
                            }
                        }
                        OwnershipKind::SharedRef | OwnershipKind::MutableRef => {
                            // Can copy references
                            let mut new_ownership = ownership.clone();
                            new_ownership.variable = target;
                            self.ownership_map.insert(target, new_ownership);
                        }
                        OwnershipKind::Moved => {
                            return Err(OwnershipError::UseAfterMove {
                                variable: *name,
                                span: source.span,
                            });
                        }
                        _ => {}
                    }
                }
            }
            DFGNodeKind::AddressOf {
                target: addr_target,
                mutability,
            } => {
                // Create a borrow
                let borrow_kind = match mutability {
                    Mutability::Mutable => BorrowKind::Mutable,
                    Mutability::Immutable => BorrowKind::Shared,
                };

                // Check if we can borrow
                if let Some(target_node) = self.find_variable_node(*addr_target) {
                    if let DFGNodeKind::Variable { name: borrowed_var } = &target_node.kind {
                        self.check_borrow_validity(*borrowed_var, &borrow_kind, source.span)?;

                        // Record the borrow
                        let borrow = BorrowInfo {
                            borrowed: *borrowed_var,
                            borrower: target,
                            kind: borrow_kind.clone(),
                            lifetime: None, // Would be computed by lifetime analysis
                            span: source.span,
                        };

                        self.borrows.entry(*borrowed_var).or_default().push(borrow);

                        // Create ownership info for the reference
                        let ownership_kind = match borrow_kind {
                            BorrowKind::Shared => OwnershipKind::SharedRef,
                            BorrowKind::Mutable => OwnershipKind::MutableRef,
                        };

                        self.ownership_map.insert(
                            target,
                            OwnershipInfo {
                                variable: target,
                                kind: ownership_kind,
                                ty: source.ty.clone(),
                                span: source.span,
                                mutable: matches!(mutability, Mutability::Mutable),
                            },
                        );
                    }
                }
            }
            _ => {
                // For other kinds, create owned value
                self.ownership_map.insert(
                    target,
                    OwnershipInfo {
                        variable: target,
                        kind: OwnershipKind::Owned,
                        ty: source.ty.clone(),
                        span: source.span,
                        mutable: false,
                    },
                );
            }
        }

        Ok(())
    }

    fn find_variable_node(&self, node_id: AnalysisNodeId) -> Option<&DFGNode> {
        // This would need access to the DFG - simplified for now
        None
    }

    fn check_borrow_validity(
        &self,
        variable: InternedString,
        kind: &BorrowKind,
        span: Span,
    ) -> Result<(), OwnershipError> {
        if let Some(existing_borrows) = self.borrows.get(&variable) {
            for borrow in existing_borrows {
                match (kind, &borrow.kind) {
                    (BorrowKind::Mutable, _) | (_, BorrowKind::Mutable) => {
                        // Cannot have mutable borrow with any other borrow
                        return Err(OwnershipError::ConflictingBorrow {
                            variable,
                            span,
                            existing_span: borrow.span,
                        });
                    }
                    (BorrowKind::Shared, BorrowKind::Shared) => {
                        // Multiple shared borrows are OK
                    }
                }
            }
        }

        Ok(())
    }

    /// Check for use-after-move errors
    pub fn check_use_after_move(
        &self,
        variable: InternedString,
        span: Span,
    ) -> Result<(), OwnershipError> {
        if let Some(ownership) = self.ownership_map.get(&variable) {
            if matches!(ownership.kind, OwnershipKind::Moved) {
                return Err(OwnershipError::UseAfterMove { variable, span });
            }
        }

        Ok(())
    }
}

// ====== LIFETIME ANALYSIS ======

/// Lifetime analysis for borrow checking
#[derive(Debug, Clone)]
pub struct LifetimeAnalysis {
    /// Lifetime constraints
    pub constraints: Vec<LifetimeConstraint>,
    /// Lifetime variables
    pub lifetime_vars: HashMap<InternedString, LifetimeInfo>,
    /// Scope information
    pub scopes: HashMap<AnalysisNodeId, ScopeInfo>,
}

#[derive(Debug, Clone)]
pub struct LifetimeConstraint {
    /// The constraint kind
    pub kind: ConstraintKind,
    /// Source location for error reporting
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ConstraintKind {
    /// Lifetime 'a must outlive 'b
    Outlives { a: Lifetime, b: Lifetime },
    /// Variable has lifetime 'a
    VarLifetime {
        var: InternedString,
        lifetime: Lifetime,
    },
    /// Function parameter lifetime constraint
    ParamConstraint {
        param: InternedString,
        lifetime: Lifetime,
    },
    /// Return value lifetime constraint
    ReturnConstraint { lifetime: Lifetime },
}

#[derive(Debug, Clone)]
pub struct LifetimeInfo {
    /// The lifetime variable
    pub lifetime: Lifetime,
    /// Variables associated with this lifetime
    pub variables: HashSet<InternedString>,
    /// Scope where this lifetime is valid
    pub scope: AnalysisNodeId,
}

#[derive(Debug, Clone)]
pub struct ScopeInfo {
    /// Scope identifier
    pub id: AnalysisNodeId,
    /// Parent scope
    pub parent: Option<AnalysisNodeId>,
    /// Variables declared in this scope
    pub variables: HashSet<InternedString>,
    /// Lifetime parameters for this scope
    pub lifetime_params: HashSet<Lifetime>,
}

impl LifetimeAnalysis {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            lifetime_vars: HashMap::new(),
            scopes: HashMap::new(),
        }
    }

    /// Add a lifetime constraint
    pub fn add_constraint(&mut self, constraint: LifetimeConstraint) {
        self.constraints.push(constraint);
    }

    /// Solve lifetime constraints
    pub fn solve_constraints(&mut self) -> Result<(), LifetimeError> {
        // Simplified constraint solving
        // Full implementation would use graph-based algorithms

        let constraints = self.constraints.clone();
        for constraint in &constraints {
            match &constraint.kind {
                ConstraintKind::Outlives { a, b } => {
                    self.check_outlives_constraint(a, b, constraint.span)?;
                }
                ConstraintKind::VarLifetime { var, lifetime } => {
                    self.assign_var_lifetime(*var, lifetime.clone());
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn check_outlives_constraint(
        &self,
        a: &Lifetime,
        b: &Lifetime,
        span: Span,
    ) -> Result<(), LifetimeError> {
        // Check if lifetime 'a outlives lifetime 'b
        // This would involve checking scope relationships
        Ok(())
    }

    fn assign_var_lifetime(&mut self, var: InternedString, lifetime: Lifetime) {
        let info = LifetimeInfo {
            lifetime: lifetime.clone(),
            variables: [var].into_iter().collect(),
            scope: AnalysisNodeId::new(0), // Would compute proper scope
        };

        self.lifetime_vars.insert(var, info);
    }
}

// ====== ERROR TYPES ======

#[derive(Debug, Clone)]
pub enum AnalysisError {
    /// Ownership analysis error
    Ownership(OwnershipError),
    /// Lifetime analysis error
    Lifetime(LifetimeError),
    /// Control flow analysis error
    ControlFlow(ControlFlowError),
    /// Data flow analysis error
    DataFlow(DataFlowError),
    /// Generic analysis error
    Generic(String),
}

#[derive(Debug, Clone)]
pub enum OwnershipError {
    /// Use after move
    UseAfterMove {
        variable: InternedString,
        span: Span,
    },
    /// Conflicting borrow
    ConflictingBorrow {
        variable: InternedString,
        span: Span,
        existing_span: Span,
    },
    /// Double free
    DoubleFree {
        variable: InternedString,
        span: Span,
    },
    /// Invalid move
    InvalidMove {
        variable: InternedString,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub enum LifetimeError {
    /// Lifetime constraint violation
    ConstraintViolation { constraint: String, span: Span },
    /// Dangling reference
    DanglingReference {
        variable: InternedString,
        span: Span,
    },
    /// Lifetime parameter mismatch
    ParameterMismatch {
        expected: Lifetime,
        found: Lifetime,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub enum ControlFlowError {
    /// Unreachable code
    UnreachableCode { span: Span },
    /// Missing return
    MissingReturn { span: Span },
    /// Invalid break/continue
    InvalidBreakContinue { span: Span },
}

#[derive(Debug, Clone)]
pub enum DataFlowError {
    /// Uninitialized variable use
    UninitializedUse {
        variable: InternedString,
        span: Span,
    },
    /// Dead code
    DeadCode { span: Span },
}

// ====== ANALYSIS RESULT ======

/// Combined result of all analysis phases
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Control flow graph
    pub cfg: ControlFlowGraph,
    /// Data flow graph
    pub dfg: DataFlowGraph,
    /// Ownership analysis results
    pub ownership: OwnershipAnalysis,
    /// Lifetime analysis results
    pub lifetime: LifetimeAnalysis,
}

// ====== IMPLEMENTATION STUBS ======
// These would be implemented in full for a complete system

impl AnalysisContext {
    fn lower_program(&mut self, _program: &TypedProgram) -> Result<(), AnalysisError> {
        // Convert TypedAST to analysis IR
        Ok(())
    }

    fn build_control_flow(&mut self) -> Result<(), AnalysisError> {
        // Build CFG from lowered IR
        Ok(())
    }

    fn build_data_flow(&mut self) -> Result<(), AnalysisError> {
        // Build DFG from lowered IR
        Ok(())
    }

    fn analyze_ownership(&mut self) -> Result<(), AnalysisError> {
        // Perform ownership analysis
        Ok(())
    }

    fn analyze_lifetimes(&mut self) -> Result<(), AnalysisError> {
        // Perform lifetime analysis
        Ok(())
    }

    fn optimize(&mut self) -> Result<(), AnalysisError> {
        // Perform optimizations
        Ok(())
    }
}

impl Default for AnalysisContext {
    fn default() -> Self {
        Self::new()
    }
}
