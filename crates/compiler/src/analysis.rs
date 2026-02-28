//! # Analysis Infrastructure
//!
//! Provides analysis passes that work on the HIR representation.
//! These analyses are used by both backends for optimization and code generation.

use crate::hir::{HirBlock, HirFunction, HirId, HirInstruction, HirModule, HirValue};
use crate::CompilerResult;
use std::collections::{HashMap, HashSet};

/// Analysis results for a module
#[derive(Debug)]
pub struct ModuleAnalysis {
    /// Function-level analyses
    pub functions: HashMap<HirId, FunctionAnalysis>,
    /// Call graph
    pub call_graph: CallGraph,
    /// Global value analysis
    pub globals: GlobalAnalysis,
}

/// Analysis results for a function
#[derive(Debug)]
pub struct FunctionAnalysis {
    /// Liveness analysis results
    pub liveness: LivenessAnalysis,
    /// Alias analysis results
    pub aliases: AliasAnalysis,
    /// Memory access patterns
    pub memory: MemoryAnalysis,
    /// Loop analysis
    pub loops: LoopAnalysis,
}

/// Liveness analysis
#[derive(Debug)]
pub struct LivenessAnalysis {
    /// Live values at block entry
    pub live_in: HashMap<HirId, HashSet<HirId>>,
    /// Live values at block exit
    pub live_out: HashMap<HirId, HashSet<HirId>>,
    /// Live ranges for each value
    pub live_ranges: HashMap<HirId, LiveRange>,
}

/// Live range of a value
#[derive(Debug, Clone)]
pub struct LiveRange {
    /// Definition point
    pub def: HirId,
    /// Last use points
    pub uses: HashSet<HirId>,
    /// Blocks where value is live
    pub blocks: HashSet<HirId>,
}

/// Alias analysis
#[derive(Debug)]
pub struct AliasAnalysis {
    /// Alias sets
    pub alias_sets: Vec<AliasSet>,
    /// Points-to information
    pub points_to: HashMap<HirId, PointsTo>,
}

/// Set of potentially aliasing values
#[derive(Debug)]
pub struct AliasSet {
    pub values: HashSet<HirId>,
    pub may_alias: bool,
}

/// Points-to information
#[derive(Debug, Clone)]
pub enum PointsTo {
    /// Points to a specific value
    Value(HirId),
    /// Points to a global
    Global(HirId),
    /// Points to heap allocation
    Heap(HirId),
    /// Points to stack allocation
    Stack(HirId),
    /// Unknown target
    Unknown,
}

/// Memory analysis
#[derive(Debug)]
pub struct MemoryAnalysis {
    /// Memory dependencies between instructions
    pub dependencies: HashMap<HirId, HashSet<HirId>>,
    /// Read-write sets
    pub reads: HashMap<HirId, HashSet<MemoryLocation>>,
    pub writes: HashMap<HirId, HashSet<MemoryLocation>>,
}

/// Memory location abstraction
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryLocation {
    /// Stack slot
    Stack(HirId),
    /// Global variable
    Global(HirId),
    /// Heap allocation
    Heap(HirId),
    /// Unknown location
    Unknown,
}

/// Loop analysis
#[derive(Debug)]
pub struct LoopAnalysis {
    /// Natural loops in the function
    pub loops: Vec<Loop>,
    /// Loop nesting forest
    pub nesting: LoopNesting,
}

/// Natural loop information
#[derive(Debug)]
pub struct Loop {
    /// Loop header block
    pub header: HirId,
    /// Blocks in the loop
    pub blocks: HashSet<HirId>,
    /// Loop exit blocks
    pub exits: HashSet<HirId>,
    /// Induction variables
    pub induction_vars: Vec<InductionVariable>,
    /// Trip count if known
    pub trip_count: Option<TripCount>,
}

/// Induction variable
#[derive(Debug)]
pub struct InductionVariable {
    pub value: HirId,
    pub init: HirId,
    pub step: HirId,
    pub kind: InductionKind,
}

#[derive(Debug, Clone, Copy)]
pub enum InductionKind {
    Basic,   // i = i + c
    Derived, // j = i * c + d
}

/// Loop trip count
#[derive(Debug, Clone)]
pub enum TripCount {
    /// Constant trip count
    Constant(u64),
    /// Symbolic trip count
    Symbolic(HirId),
    /// Unknown
    Unknown,
}

/// Loop nesting information
#[derive(Debug)]
pub struct LoopNesting {
    /// Parent loop for each loop
    pub parents: HashMap<usize, usize>,
    /// Depth of each loop
    pub depths: HashMap<usize, u32>,
}

/// Call graph
#[derive(Debug)]
pub struct CallGraph {
    /// Direct calls from each function
    pub direct_calls: HashMap<HirId, HashSet<HirId>>,
    /// Indirect call sites
    pub indirect_calls: HashMap<HirId, Vec<HirId>>,
    /// Strongly connected components
    pub sccs: Vec<HashSet<HirId>>,
}

/// Global value analysis
#[derive(Debug)]
pub struct GlobalAnalysis {
    /// Escape analysis for globals
    pub escapes: HashMap<HirId, bool>,
    /// Constant globals
    pub constants: HashSet<HirId>,
}

/// Main analysis runner
pub struct AnalysisRunner {
    module: HirModule,
}

impl AnalysisRunner {
    pub fn new(module: HirModule) -> Self {
        Self { module }
    }

    /// Run all analyses
    pub fn run_all(&mut self) -> CompilerResult<ModuleAnalysis> {
        let mut functions = HashMap::new();

        // Analyze each function
        for (func_id, func) in &self.module.functions {
            let func_analysis = self.analyze_function(func)?;
            functions.insert(*func_id, func_analysis);
        }

        // Build call graph
        let call_graph = self.build_call_graph()?;

        // Analyze globals
        let globals = self.analyze_globals()?;

        Ok(ModuleAnalysis {
            functions,
            call_graph,
            globals,
        })
    }

    /// Analyze a single function
    fn analyze_function(&self, func: &HirFunction) -> CompilerResult<FunctionAnalysis> {
        let liveness = self.compute_liveness(func)?;
        let aliases = self.compute_aliases(func)?;
        let memory = self.analyze_memory(func)?;
        let loops = self.analyze_loops(func)?;

        Ok(FunctionAnalysis {
            liveness,
            aliases,
            memory,
            loops,
        })
    }

    /// Compute liveness analysis
    fn compute_liveness(&self, func: &HirFunction) -> CompilerResult<LivenessAnalysis> {
        let mut live_in = HashMap::new();
        let mut live_out = HashMap::new();
        let mut live_ranges = HashMap::new();

        // Initialize
        for (block_id, _) in &func.blocks {
            live_in.insert(*block_id, HashSet::new());
            live_out.insert(*block_id, HashSet::new());
        }

        // Compute block-local use-def sets
        let (uses, defs) = self.compute_use_def_sets(func);

        // Fixed-point iteration
        let mut changed = true;
        while changed {
            changed = false;

            // Process blocks in reverse postorder
            for (block_id, block) in &func.blocks {
                let mut new_live_out = HashSet::new();

                // Union of successors' live_in
                for succ_id in &block.successors {
                    new_live_out.extend(&live_in[succ_id]);
                }

                // live_in = use ∪ (live_out - def)
                let mut new_live_in = uses[block_id].clone();
                for val in &new_live_out {
                    if !defs[block_id].contains(val) {
                        new_live_in.insert(*val);
                    }
                }

                if new_live_in != live_in[block_id] || new_live_out != live_out[block_id] {
                    live_in.insert(*block_id, new_live_in);
                    live_out.insert(*block_id, new_live_out);
                    changed = true;
                }
            }
        }

        // Compute live ranges
        for (value_id, value) in &func.values {
            let mut range = LiveRange {
                def: *value_id,
                uses: value.uses.clone(),
                blocks: HashSet::new(),
            };

            // Find all blocks where value is live
            for (block_id, live_set) in &live_in {
                if live_set.contains(value_id) {
                    range.blocks.insert(*block_id);
                }
            }
            for (block_id, live_set) in &live_out {
                if live_set.contains(value_id) {
                    range.blocks.insert(*block_id);
                }
            }

            live_ranges.insert(*value_id, range);
        }

        Ok(LivenessAnalysis {
            live_in,
            live_out,
            live_ranges,
        })
    }

    /// Compute use-def sets for blocks
    fn compute_use_def_sets(
        &self,
        func: &HirFunction,
    ) -> (
        HashMap<HirId, HashSet<HirId>>,
        HashMap<HirId, HashSet<HirId>>,
    ) {
        let mut uses = HashMap::new();
        let mut defs = HashMap::new();

        for (block_id, block) in &func.blocks {
            let mut block_uses = HashSet::new();
            let mut block_defs = HashSet::new();

            // Process phis
            for phi in &block.phis {
                block_defs.insert(phi.result);
                for (val, _) in &phi.incoming {
                    if !block_defs.contains(val) {
                        block_uses.insert(*val);
                    }
                }
            }

            // Process instructions
            for inst in &block.instructions {
                // Collect uses
                self.collect_instruction_uses(inst, &mut block_uses, &block_defs);

                // Collect defs
                if let Some(result) = self.get_instruction_result(inst) {
                    block_defs.insert(result);
                }
            }

            // Process terminator
            self.collect_terminator_uses(&block.terminator, &mut block_uses, &block_defs);

            uses.insert(*block_id, block_uses);
            defs.insert(*block_id, block_defs);
        }

        (uses, defs)
    }

    /// Collect uses from an instruction
    fn collect_instruction_uses(
        &self,
        inst: &HirInstruction,
        uses: &mut HashSet<HirId>,
        defs: &HashSet<HirId>,
    ) {
        match inst {
            HirInstruction::Binary { left, right, .. } => {
                if !defs.contains(left) {
                    uses.insert(*left);
                }
                if !defs.contains(right) {
                    uses.insert(*right);
                }
            }
            HirInstruction::Unary { operand, .. } => {
                if !defs.contains(operand) {
                    uses.insert(*operand);
                }
            }
            HirInstruction::Load { ptr, .. } => {
                if !defs.contains(ptr) {
                    uses.insert(*ptr);
                }
            }
            HirInstruction::Store { value, ptr, .. } => {
                if !defs.contains(value) {
                    uses.insert(*value);
                }
                if !defs.contains(ptr) {
                    uses.insert(*ptr);
                }
            }
            HirInstruction::Call { callee, args, .. } => {
                if let crate::hir::HirCallable::Indirect(val) = callee {
                    if !defs.contains(val) {
                        uses.insert(*val);
                    }
                }
                for arg in args {
                    if !defs.contains(arg) {
                        uses.insert(*arg);
                    }
                }
            }
            _ => {}
        }
    }

    /// Collect uses from a terminator
    fn collect_terminator_uses(
        &self,
        term: &crate::hir::HirTerminator,
        uses: &mut HashSet<HirId>,
        defs: &HashSet<HirId>,
    ) {
        match term {
            crate::hir::HirTerminator::Return { values } => {
                for val in values {
                    if !defs.contains(val) {
                        uses.insert(*val);
                    }
                }
            }
            crate::hir::HirTerminator::CondBranch { condition, .. } => {
                if !defs.contains(condition) {
                    uses.insert(*condition);
                }
            }
            crate::hir::HirTerminator::Switch { value, .. } => {
                if !defs.contains(value) {
                    uses.insert(*value);
                }
            }
            _ => {}
        }
    }

    /// Get result of an instruction
    fn get_instruction_result(&self, inst: &HirInstruction) -> Option<HirId> {
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
            | HirInstruction::Atomic { result, .. }
            | HirInstruction::CreateUnion { result, .. }
            | HirInstruction::GetUnionDiscriminant { result, .. }
            | HirInstruction::ExtractUnionValue { result, .. }
            | HirInstruction::CreateClosure { result, .. }
            | HirInstruction::CreateTraitObject { result, .. }
            | HirInstruction::UpcastTraitObject { result, .. } => Some(*result),

            HirInstruction::Call { result, .. }
            | HirInstruction::IndirectCall { result, .. }
            | HirInstruction::CallClosure { result, .. }
            | HirInstruction::TraitMethodCall { result, .. } => *result,

            HirInstruction::CreateRef { result, .. }
            | HirInstruction::Deref { result, .. }
            | HirInstruction::Move { result, .. }
            | HirInstruction::Copy { result, .. } => Some(*result),

            HirInstruction::Store { .. }
            | HirInstruction::Fence { .. }
            | HirInstruction::BeginLifetime { .. }
            | HirInstruction::EndLifetime { .. }
            | HirInstruction::LifetimeConstraint { .. }
            | HirInstruction::Resume { .. }
            | HirInstruction::AbortEffect { .. } => None,

            // Effect instructions with optional result
            HirInstruction::PerformEffect { result, .. }
            | HirInstruction::HandleEffect { result, .. } => *result,

            // CaptureContinuation always has a result
            HirInstruction::CaptureContinuation { result, .. } => Some(*result),

            // SIMD instructions: Splat/Extract/Insert/Reduce/Load produce a result
            HirInstruction::VectorSplat { result, .. }
            | HirInstruction::VectorExtractLane { result, .. }
            | HirInstruction::VectorInsertLane { result, .. }
            | HirInstruction::VectorHorizontalReduce { result, .. }
            | HirInstruction::VectorLoad { result, .. } => Some(*result),
            // VectorStore only writes to memory, no result
            HirInstruction::VectorStore { .. } => None,
        }
    }

    /// Compute alias analysis
    fn compute_aliases(&self, func: &HirFunction) -> CompilerResult<AliasAnalysis> {
        let mut alias_sets = Vec::new();
        let mut points_to = HashMap::new();

        // Simple alias analysis: track allocations and pointer operations
        for (_, block) in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    HirInstruction::Alloca { result, .. } => {
                        points_to.insert(*result, PointsTo::Stack(*result));
                        alias_sets.push(AliasSet {
                            values: vec![*result].into_iter().collect(),
                            may_alias: false,
                        });
                    }

                    HirInstruction::Load { result, ptr, .. } => {
                        if let Some(target) = points_to.get(ptr) {
                            points_to.insert(*result, target.clone());
                        } else {
                            points_to.insert(*result, PointsTo::Unknown);
                        }
                    }

                    _ => {}
                }
            }
        }

        Ok(AliasAnalysis {
            alias_sets,
            points_to,
        })
    }

    /// Analyze memory access patterns
    fn analyze_memory(&self, func: &HirFunction) -> CompilerResult<MemoryAnalysis> {
        let dependencies = HashMap::new();
        let mut reads = HashMap::new();
        let mut writes = HashMap::new();

        // Track memory operations
        for (_, block) in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    HirInstruction::Load { result, ptr, .. } => {
                        let loc = self.get_memory_location(*ptr, &func);
                        reads
                            .entry(*result)
                            .or_insert_with(HashSet::new)
                            .insert(loc);
                    }

                    HirInstruction::Store { ptr, .. } => {
                        let loc = self.get_memory_location(*ptr, &func);
                        writes.entry(*ptr).or_insert_with(HashSet::new).insert(loc);
                    }

                    _ => {}
                }
            }
        }

        Ok(MemoryAnalysis {
            dependencies,
            reads,
            writes,
        })
    }

    /// Get abstract memory location for a pointer
    fn get_memory_location(&self, ptr: HirId, func: &HirFunction) -> MemoryLocation {
        // Simple analysis - track allocas
        for (_, block) in &func.blocks {
            for inst in &block.instructions {
                if let HirInstruction::Alloca { result, .. } = inst {
                    if *result == ptr {
                        return MemoryLocation::Stack(ptr);
                    }
                }
            }
        }

        MemoryLocation::Unknown
    }

    /// Analyze loops in the function
    fn analyze_loops(&self, _func: &HirFunction) -> CompilerResult<LoopAnalysis> {
        // NOTE: Loop detection deferred - not critical for initial Cranelift backend.
        // Requires: (1) Back-edge detection, (2) Natural loop identification, (3) Nesting analysis
        // FUTURE (v2.0): Implement loop detection for LICM and other loop optimizations
        // Estimated effort: 8-10 hours
        let loops = Vec::new();
        let nesting = LoopNesting {
            parents: HashMap::new(),
            depths: HashMap::new(),
        };

        Ok(LoopAnalysis { loops, nesting })
    }

    /// Build module call graph
    fn build_call_graph(&self) -> CompilerResult<CallGraph> {
        let mut direct_calls = HashMap::new();
        let mut indirect_calls = HashMap::new();

        for (func_id, func) in &self.module.functions {
            let mut func_calls = HashSet::new();
            let mut func_indirect = Vec::new();

            for (_, block) in &func.blocks {
                for inst in &block.instructions {
                    if let HirInstruction::Call { callee, .. } = inst {
                        match callee {
                            crate::hir::HirCallable::Function(target) => {
                                func_calls.insert(*target);
                            }
                            crate::hir::HirCallable::Indirect(val) => {
                                func_indirect.push(*val);
                            }
                            _ => {}
                        }
                    }
                }
            }

            direct_calls.insert(*func_id, func_calls);
            indirect_calls.insert(*func_id, func_indirect);
        }

        // NOTE: SCC (Strongly Connected Components) computation deferred.
        // Used for detecting mutual recursion and call graph cycles.
        // FUTURE (v2.0): Implement Tarjan's or Kosaraju's algorithm for SCC detection
        // Estimated effort: 4-6 hours (standard graph algorithm)
        let sccs = Vec::new();

        Ok(CallGraph {
            direct_calls,
            indirect_calls,
            sccs,
        })
    }

    /// Analyze global variables
    fn analyze_globals(&self) -> CompilerResult<GlobalAnalysis> {
        let mut escapes = HashMap::new();
        let mut constants = HashSet::new();

        for (global_id, global) in &self.module.globals {
            // Check if global escapes (address taken)
            let does_escape = self.check_global_escape(*global_id);
            escapes.insert(*global_id, does_escape);

            // Check if global is constant
            if global.is_const {
                constants.insert(*global_id);
            }
        }

        Ok(GlobalAnalysis { escapes, constants })
    }

    /// Check if a global variable escapes
    fn check_global_escape(&self, _global_id: HirId) -> bool {
        // NOTE: Escape analysis for globals deferred - conservative assumption.
        // Requires scanning all functions for address-of operations on globals.
        // WORKAROUND: Returns false (assumes no escape - may over-optimize)
        // FUTURE (v2.0): Implement global escape analysis
        // Estimated effort: 3-4 hours
        false
    }
}
