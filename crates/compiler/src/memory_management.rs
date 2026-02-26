//! # Memory Management Infrastructure
//!
//! Provides reference counting, automatic memory management, and ownership tracking
//! for the HIR. This module implements both manual and automatic reference counting
//! strategies that can be selected based on language requirements.

use crate::hir::*;
use crate::{CompilerError, CompilerResult};
use std::collections::{HashMap, HashSet};

/// Memory management strategy for the compiler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStrategy {
    /// Manual memory management (malloc/free)
    Manual,
    /// Automatic Reference Counting (like Swift/Objective-C)
    ARC,
    /// Garbage Collection (tracing GC)
    GC,
    /// Ownership-based (like Rust)
    Ownership,
    /// Hybrid (combination of strategies)
    Hybrid,
}

/// Reference count information for a value
#[derive(Debug, Clone)]
pub struct RefCountInfo {
    /// Current reference count
    pub count: u64,
    /// Whether this is a weak reference
    pub is_weak: bool,
    /// Values that this value references
    pub references: HashSet<HirId>,
    /// Values that reference this value
    pub referenced_by: HashSet<HirId>,
}

/// Memory management context for a function
#[derive(Debug, Clone)]
pub struct MemoryContext {
    /// Memory strategy being used
    pub strategy: MemoryStrategy,
    /// Reference count information for values
    pub ref_counts: HashMap<HirId, RefCountInfo>,
    /// Values that need to be dropped
    pub pending_drops: Vec<HirId>,
    /// Escape analysis results
    pub escape_info: HashMap<HirId, EscapeInfo>,
    /// Allocation sites
    pub allocations: HashMap<HirId, AllocationInfo>,
}

/// Information about value escape behavior
#[derive(Debug, Clone)]
pub struct EscapeInfo {
    /// Whether the value escapes the current function
    pub escapes: bool,
    /// Functions this value escapes to
    pub escape_targets: HashSet<HirId>,
    /// Whether the value is returned
    pub is_returned: bool,
    /// Whether the value is stored in heap
    pub stored_in_heap: bool,
}

/// Information about an allocation
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Type of the allocated value
    pub ty: HirType,
    /// Size of allocation in bytes
    pub size: Option<u64>,
    /// Alignment requirement
    pub align: u32,
    /// Whether this is a stack allocation
    pub is_stack: bool,
    /// Allocation site location
    pub location: Option<zyntax_typed_ast::Span>,
}

/// Automatic Reference Counting (ARC) manager
pub struct ARCManager {
    /// Functions that need ARC instrumentation
    pub arc_functions: HashSet<HirId>,
    /// Types that support ARC
    pub arc_types: HashSet<HirType>,
    /// Weak reference mappings
    pub weak_refs: HashMap<HirId, HirId>,
}

impl ARCManager {
    pub fn new() -> Self {
        Self {
            arc_functions: HashSet::new(),
            arc_types: HashSet::new(),
            weak_refs: HashMap::new(),
        }
    }

    /// Insert reference counting operations for a function
    pub fn instrument_function(&mut self, func: &mut HirFunction) -> CompilerResult<()> {
        // Track all values that need reference counting
        let mut arc_values = Vec::new();

        for (value_id, value) in &func.values {
            if self.needs_arc(&value.ty) {
                arc_values.push(*value_id);
            }
        }

        // Insert retain/release calls
        for block in func.blocks.values_mut() {
            self.instrument_block(block, &arc_values)?;
        }

        // Mark function as ARC-instrumented
        self.arc_functions.insert(func.id);

        Ok(())
    }

    /// Check if a type needs ARC
    pub fn needs_arc(&self, ty: &HirType) -> bool {
        match ty {
            HirType::Ptr(_) | HirType::Ref { .. } => true,
            HirType::Array(elem_ty, _) => self.needs_arc(elem_ty),
            HirType::Struct(struct_ty) => struct_ty.fields.iter().any(|f| self.needs_arc(f)),
            HirType::Union(union_ty) => union_ty.variants.iter().any(|v| self.needs_arc(&v.ty)),
            HirType::Closure(_) => true,
            _ => false,
        }
    }

    /// Instrument a block with ARC operations
    fn instrument_block(
        &mut self,
        block: &mut HirBlock,
        arc_values: &[HirId],
    ) -> CompilerResult<()> {
        let mut new_instructions = Vec::new();

        for inst in &block.instructions {
            // Insert retain before use
            if let Some(values) = self.get_used_values(inst) {
                for value in values {
                    if arc_values.contains(&value) {
                        new_instructions.push(self.create_retain(value));
                    }
                }
            }

            new_instructions.push(inst.clone());

            // Insert release after last use
            if let Some(result) = self.get_result(inst) {
                if arc_values.contains(&result) {
                    // This is simplified - real implementation would track last use
                    new_instructions.push(self.create_release(result));
                }
            }
        }

        block.instructions = new_instructions;
        Ok(())
    }

    /// Create a retain (increment reference count) instruction
    fn create_retain(&self, value: HirId) -> HirInstruction {
        HirInstruction::Call {
            result: None,
            callee: HirCallable::Intrinsic(Intrinsic::IncRef),
            args: vec![value],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        }
    }

    /// Create a release (decrement reference count) instruction
    fn create_release(&self, value: HirId) -> HirInstruction {
        HirInstruction::Call {
            result: None,
            callee: HirCallable::Intrinsic(Intrinsic::DecRef),
            args: vec![value],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        }
    }

    /// Get values used by an instruction
    fn get_used_values(&self, inst: &HirInstruction) -> Option<Vec<HirId>> {
        match inst {
            HirInstruction::Binary { left, right, .. } => Some(vec![*left, *right]),
            HirInstruction::Unary { operand, .. } => Some(vec![*operand]),
            HirInstruction::Load { ptr, .. } => Some(vec![*ptr]),
            HirInstruction::Store { value, ptr, .. } => Some(vec![*value, *ptr]),
            HirInstruction::Call { args, .. } => Some(args.clone()),
            HirInstruction::CreateRef { value, .. } => Some(vec![*value]),
            HirInstruction::Deref { reference, .. } => Some(vec![*reference]),
            HirInstruction::Move { source, .. } => Some(vec![*source]),
            HirInstruction::Copy { source, .. } => Some(vec![*source]),
            _ => None,
        }
    }

    /// Get result of an instruction
    fn get_result(&self, inst: &HirInstruction) -> Option<HirId> {
        match inst {
            HirInstruction::Binary { result, .. }
            | HirInstruction::Unary { result, .. }
            | HirInstruction::Alloca { result, .. }
            | HirInstruction::Load { result, .. }
            | HirInstruction::CreateRef { result, .. }
            | HirInstruction::Deref { result, .. }
            | HirInstruction::Move { result, .. }
            | HirInstruction::Copy { result, .. } => Some(*result),
            HirInstruction::Call { result, .. } => *result,
            _ => None,
        }
    }
}

/// Drop/destructor management
pub struct DropManager {
    /// Types with custom destructors
    pub drop_types: HashMap<HirType, HirId>, // Type -> Destructor function
    /// Values that need dropping
    pub needs_drop: HashSet<HirId>,
    /// Drop order dependencies
    pub drop_order: Vec<HirId>,
}

impl DropManager {
    pub fn new() -> Self {
        Self {
            drop_types: HashMap::new(),
            needs_drop: HashSet::new(),
            drop_order: Vec::new(),
        }
    }

    /// Register a type with a custom destructor
    pub fn register_destructor(&mut self, ty: HirType, destructor: HirId) {
        self.drop_types.insert(ty, destructor);
    }

    /// Insert drop calls at appropriate points
    pub fn insert_drops(&mut self, func: &mut HirFunction) -> CompilerResult<()> {
        // Analyze which values need dropping
        self.analyze_drops(func)?;

        // Create drop sequences before mutating blocks
        let mut drop_sequences = HashMap::new();
        for (block_id, block) in &func.blocks {
            match &block.terminator {
                HirTerminator::Return { .. } => {
                    // Create drops for this block
                    let drops = self.create_drop_sequence(func);
                    drop_sequences.insert(*block_id, drops);
                }
                _ => {}
            }
        }

        // Now insert the drop sequences
        for (block_id, drops) in drop_sequences {
            if let Some(block) = func.blocks.get_mut(&block_id) {
                block.instructions.extend(drops);
            }
        }

        Ok(())
    }

    /// Analyze which values need dropping
    fn analyze_drops(&mut self, func: &HirFunction) -> CompilerResult<()> {
        for (value_id, value) in &func.values {
            if self.needs_drop_type(&value.ty) {
                self.needs_drop.insert(*value_id);
            }
        }

        // Compute drop order based on dependencies
        self.compute_drop_order(func)?;

        Ok(())
    }

    /// Check if a type needs dropping
    pub fn needs_drop_type(&self, ty: &HirType) -> bool {
        // Check if type has custom destructor
        if self.drop_types.contains_key(ty) {
            return true;
        }

        // Check if type contains values that need dropping
        match ty {
            HirType::Ptr(_) => true, // Heap allocations need freeing
            HirType::Array(elem_ty, _) => self.needs_drop_type(elem_ty),
            HirType::Struct(struct_ty) => struct_ty.fields.iter().any(|f| self.needs_drop_type(f)),
            HirType::Union(union_ty) => union_ty
                .variants
                .iter()
                .any(|v| self.needs_drop_type(&v.ty)),
            HirType::Closure(_) => true, // Closures have captured environment
            _ => false,
        }
    }

    /// Compute the order in which values should be dropped
    fn compute_drop_order(&mut self, _func: &HirFunction) -> CompilerResult<()> {
        // Simple LIFO order for now
        // Real implementation would consider dependencies
        self.drop_order = self.needs_drop.iter().copied().collect();
        self.drop_order.reverse();
        Ok(())
    }

    /// Create drop instruction sequence
    fn create_drop_sequence(&self, func: &HirFunction) -> Vec<HirInstruction> {
        let mut drops = Vec::new();

        for value_id in &self.drop_order {
            if let Some(value) = func.values.get(value_id) {
                if let Some(destructor) = self.drop_types.get(&value.ty) {
                    // Call custom destructor
                    drops.push(HirInstruction::Call {
                        result: None,
                        callee: HirCallable::Function(*destructor),
                        args: vec![*value_id],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    });
                } else {
                    // Call generic drop intrinsic
                    drops.push(HirInstruction::Call {
                        result: None,
                        callee: HirCallable::Intrinsic(Intrinsic::Drop),
                        args: vec![*value_id],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    });
                }
            }
        }

        drops
    }
}

/// Escape analysis for optimizing allocations
pub struct EscapeAnalysis {
    /// Analysis results
    pub results: HashMap<HirId, EscapeInfo>,
}

impl EscapeAnalysis {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Analyze escape behavior of values in a function
    pub fn analyze(&mut self, func: &HirFunction) -> CompilerResult<()> {
        // Initialize all values as non-escaping
        for value_id in func.values.keys() {
            self.results.insert(
                *value_id,
                EscapeInfo {
                    escapes: false,
                    escape_targets: HashSet::new(),
                    is_returned: false,
                    stored_in_heap: false,
                },
            );
        }

        // Analyze each block
        for block in func.blocks.values() {
            self.analyze_block(block)?;
        }

        // Propagate escape information
        self.propagate_escapes()?;

        Ok(())
    }

    /// Analyze escape behavior in a block
    fn analyze_block(&mut self, block: &HirBlock) -> CompilerResult<()> {
        for inst in &block.instructions {
            match inst {
                HirInstruction::Store { value, ptr, .. } => {
                    // Value escapes if stored through pointer
                    if let Some(info) = self.results.get_mut(value) {
                        info.stored_in_heap = true;
                        info.escapes = true;
                    }
                }
                HirInstruction::Call { args, .. } => {
                    // Arguments escape through function calls
                    for arg in args {
                        if let Some(info) = self.results.get_mut(arg) {
                            info.escapes = true;
                        }
                    }
                }
                _ => {}
            }
        }

        // Check terminator
        match &block.terminator {
            HirTerminator::Return { values } => {
                for value in values {
                    if let Some(info) = self.results.get_mut(value) {
                        info.is_returned = true;
                        info.escapes = true;
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Propagate escape information transitively
    fn propagate_escapes(&mut self) -> CompilerResult<()> {
        // Simple fixpoint iteration
        let mut changed = true;
        while changed {
            changed = false;

            // If a value contains an escaping value, it also escapes
            let current_results = self.results.clone();
            for (value_id, info) in &mut self.results {
                if !info.escapes {
                    // Check if any referenced values escape
                    // This is simplified - real implementation would track references
                    if current_results
                        .values()
                        .any(|other| other.escapes && other.escape_targets.contains(value_id))
                    {
                        info.escapes = true;
                        changed = true;
                    }
                }
            }
        }

        Ok(())
    }
}

impl MemoryContext {
    pub fn new(strategy: MemoryStrategy) -> Self {
        Self {
            strategy,
            ref_counts: HashMap::new(),
            pending_drops: Vec::new(),
            escape_info: HashMap::new(),
            allocations: HashMap::new(),
        }
    }

    /// Track a new allocation
    pub fn track_allocation(&mut self, value: HirId, info: AllocationInfo) {
        self.allocations.insert(value, info);

        // Initialize reference count if using ARC
        if self.strategy == MemoryStrategy::ARC {
            self.ref_counts.insert(
                value,
                RefCountInfo {
                    count: 1,
                    is_weak: false,
                    references: HashSet::new(),
                    referenced_by: HashSet::new(),
                },
            );
        }
    }

    /// Increment reference count
    pub fn retain(&mut self, value: HirId) {
        if let Some(ref_info) = self.ref_counts.get_mut(&value) {
            ref_info.count += 1;
        }
    }

    /// Decrement reference count
    pub fn release(&mut self, value: HirId) {
        if let Some(ref_info) = self.ref_counts.get_mut(&value) {
            ref_info.count = ref_info.count.saturating_sub(1);
            if ref_info.count == 0 {
                self.pending_drops.push(value);
            }
        }
    }

    /// Check if a value can be stack allocated
    pub fn can_stack_allocate(&self, value: HirId) -> bool {
        if let Some(escape_info) = self.escape_info.get(&value) {
            !escape_info.escapes
        } else {
            false
        }
    }

    /// Get all non-escaping allocations that can be promoted to stack
    pub fn get_stack_promotable_allocations(&self) -> Vec<HirId> {
        self.allocations
            .iter()
            .filter(|(id, info)| {
                // Can promote if:
                // 1. Not already on stack
                // 2. Doesn't escape
                // 3. Has known size
                !info.is_stack && self.can_stack_allocate(**id) && info.size.is_some()
            })
            .map(|(id, _)| *id)
            .collect()
    }
}

/// Stack promotion pass - converts heap allocations to stack where safe
pub struct StackPromotionPass {
    /// Allocations that were promoted
    pub promoted: Vec<HirId>,
    /// Allocations that couldn't be promoted (with reasons)
    pub not_promoted: Vec<(HirId, String)>,
}

impl StackPromotionPass {
    pub fn new() -> Self {
        Self {
            promoted: Vec::new(),
            not_promoted: Vec::new(),
        }
    }

    /// Run stack promotion on a function
    pub fn promote_function(
        &mut self,
        func: &mut HirFunction,
        escape_info: &HashMap<HirId, EscapeInfo>,
    ) -> CompilerResult<()> {
        eprintln!(
            "[STACK_PROMOTION] Analyzing function '{}'",
            func.name
                .resolve_global()
                .unwrap_or_else(|| "?".to_string())
        );

        // Find all heap allocations (malloc calls)
        let mut allocations_to_promote = Vec::new();

        for block in func.blocks.values() {
            for inst in &block.instructions {
                if let HirInstruction::Call {
                    result: Some(result),
                    callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                    args,
                    ..
                } = inst
                {
                    // Check if this allocation escapes
                    if let Some(info) = escape_info.get(result) {
                        if info.escapes {
                            self.not_promoted.push((
                                *result,
                                format!(
                                    "escapes: returned={}, stored_in_heap={}",
                                    info.is_returned, info.stored_in_heap
                                ),
                            ));
                        } else {
                            // Safe to promote to stack
                            allocations_to_promote.push((*result, args.clone()));
                        }
                    } else {
                        // No escape info - be conservative
                        self.not_promoted
                            .push((*result, "no escape info".to_string()));
                    }
                }
            }
        }

        eprintln!(
            "[STACK_PROMOTION] Found {} promotable allocations, {} not promotable",
            allocations_to_promote.len(),
            self.not_promoted.len()
        );

        // Actually promote the allocations
        for (result_id, args) in allocations_to_promote {
            self.promote_allocation(func, result_id, args)?;
            self.promoted.push(result_id);
        }

        Ok(())
    }

    /// Convert a malloc call to a stack allocation
    fn promote_allocation(
        &mut self,
        func: &mut HirFunction,
        result_id: HirId,
        malloc_args: Vec<HirId>,
    ) -> CompilerResult<()> {
        // Find and replace the malloc instruction
        for block in func.blocks.values_mut() {
            let mut new_instructions = Vec::new();

            for inst in &block.instructions {
                if let HirInstruction::Call {
                    result: Some(result),
                    callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                    ..
                } = inst
                {
                    if *result == result_id {
                        // Replace with alloca
                        eprintln!(
                            "[STACK_PROMOTION] Promoting allocation {:?} to stack",
                            result_id
                        );
                        new_instructions.push(HirInstruction::Alloca {
                            result: result_id,
                            ty: HirType::I8, // Byte array, will be cast
                            count: malloc_args.get(0).cloned(),
                            align: 8,
                        });
                        continue;
                    }
                }
                new_instructions.push(inst.clone());
            }

            block.instructions = new_instructions;
        }

        // Also need to remove the corresponding free if there is one
        for block in func.blocks.values_mut() {
            block.instructions.retain(|inst| {
                if let HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Free),
                    args,
                    ..
                } = inst
                {
                    // Remove free of the promoted allocation
                    !args.contains(&result_id)
                } else {
                    true
                }
            });
        }

        Ok(())
    }

    /// Get summary of promotion results
    pub fn get_summary(&self) -> String {
        format!(
            "Stack promotion: {} promoted, {} not promoted",
            self.promoted.len(),
            self.not_promoted.len()
        )
    }
}

/// Unified cleanup behavior that bridges LinearTypeChecker's cleanup rules with DropManager
/// This allows consistent resource management across the TypedAST and HIR levels.
#[derive(Debug, Clone, PartialEq)]
pub enum UnifiedCleanupBehavior {
    /// Automatic cleanup (RAII/drop) - matches LinearTypeChecker's CleanupBehavior::Automatic
    Automatic {
        destructor: Option<HirId>,
        is_fallible: bool,
    },
    /// Manual cleanup required - user must call cleanup explicitly
    Manual { cleanup_intrinsic: Intrinsic },
    /// No cleanup needed (e.g., Copy types, primitives)
    None,
    /// Reference counted cleanup (decrement refcount, free if zero)
    RefCounted,
    /// Deferred cleanup (cleanup at end of scope or transaction)
    Deferred { scope_id: HirId },
}

/// Linearity kind for HIR-level tracking
/// Mirrors LinearityKind from TypedAST level for consistency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HirLinearityKind {
    /// Must be used exactly once (linear) - e.g., file handles
    Linear,
    /// Must be used at most once (affine) - e.g., unique pointers
    Affine,
    /// Can be used multiple times (relevant)
    Relevant,
    /// Can be used zero or more times (unrestricted) - e.g., Copy types
    Unrestricted,
    /// Cannot be copied but can be borrowed (unique)
    Unique,
    /// Read-only, can be shared but not mutated
    Shared,
}

/// Information about a type's cleanup requirements
#[derive(Debug, Clone)]
pub struct CleanupInfo {
    /// The type being cleaned up
    pub ty: HirType,
    /// How to clean up values of this type
    pub cleanup_behavior: UnifiedCleanupBehavior,
    /// Linearity kind determining usage patterns
    pub linearity: HirLinearityKind,
    /// Whether this type needs cleanup at all
    pub needs_cleanup: bool,
    /// Priority for cleanup ordering (higher = cleanup first)
    pub cleanup_priority: i32,
}

/// Unified Drop/Cleanup Manager that integrates with LinearTypeChecker
pub struct UnifiedCleanupManager {
    /// Type cleanup information
    type_cleanup: HashMap<HirType, CleanupInfo>,
    /// Values pending cleanup in current scope
    pending_cleanup: Vec<HirId>,
    /// Scope stack for deferred cleanup
    scope_stack: Vec<ScopeCleanupInfo>,
    /// Custom destructors registered from TypedAST
    custom_destructors: HashMap<HirType, HirId>,
    /// Memory strategy being used
    strategy: MemoryStrategy,
}

/// Cleanup information for a scope
#[derive(Debug, Clone)]
pub struct ScopeCleanupInfo {
    /// Scope identifier
    pub scope_id: HirId,
    /// Values that need cleanup when scope ends
    pub values_to_cleanup: Vec<HirId>,
    /// Deferred cleanup actions
    pub deferred_actions: Vec<CleanupAction>,
}

/// A cleanup action to perform
#[derive(Debug, Clone)]
pub enum CleanupAction {
    /// Call destructor on a value
    CallDestructor { value: HirId, destructor: HirId },
    /// Call an intrinsic (e.g., Free, DecRef)
    CallIntrinsic {
        intrinsic: Intrinsic,
        args: Vec<HirId>,
    },
    /// Drop a value using generic drop
    Drop { value: HirId },
}

impl UnifiedCleanupManager {
    pub fn new(strategy: MemoryStrategy) -> Self {
        Self {
            type_cleanup: HashMap::new(),
            pending_cleanup: Vec::new(),
            scope_stack: Vec::new(),
            custom_destructors: HashMap::new(),
            strategy,
        }
    }

    /// Register cleanup information for a type
    pub fn register_type_cleanup(&mut self, ty: HirType, info: CleanupInfo) {
        self.type_cleanup.insert(ty, info);
    }

    /// Register a custom destructor for a type
    pub fn register_destructor(&mut self, ty: HirType, destructor: HirId) {
        self.custom_destructors.insert(ty.clone(), destructor);

        // Also update cleanup info
        if let Some(info) = self.type_cleanup.get_mut(&ty) {
            info.cleanup_behavior = UnifiedCleanupBehavior::Automatic {
                destructor: Some(destructor),
                is_fallible: false,
            };
        } else {
            self.type_cleanup.insert(
                ty.clone(),
                CleanupInfo {
                    ty,
                    cleanup_behavior: UnifiedCleanupBehavior::Automatic {
                        destructor: Some(destructor),
                        is_fallible: false,
                    },
                    linearity: HirLinearityKind::Affine,
                    needs_cleanup: true,
                    cleanup_priority: 0,
                },
            );
        }
    }

    /// Enter a new cleanup scope
    pub fn enter_scope(&mut self) -> HirId {
        let scope_id = HirId::new();
        self.scope_stack.push(ScopeCleanupInfo {
            scope_id,
            values_to_cleanup: Vec::new(),
            deferred_actions: Vec::new(),
        });
        scope_id
    }

    /// Exit a scope and get cleanup actions
    pub fn exit_scope(&mut self) -> Option<Vec<CleanupAction>> {
        self.scope_stack.pop().map(|scope| {
            let mut actions = scope.deferred_actions;

            // Add cleanup for values in reverse order (LIFO)
            for value in scope.values_to_cleanup.into_iter().rev() {
                if let Some(destructor) = self.get_destructor_for_value(value) {
                    actions.push(CleanupAction::CallDestructor { value, destructor });
                } else {
                    // Use strategy-appropriate cleanup
                    match self.strategy {
                        MemoryStrategy::ARC => {
                            actions.push(CleanupAction::CallIntrinsic {
                                intrinsic: Intrinsic::DecRef,
                                args: vec![value],
                            });
                        }
                        MemoryStrategy::Manual => {
                            actions.push(CleanupAction::CallIntrinsic {
                                intrinsic: Intrinsic::Free,
                                args: vec![value],
                            });
                        }
                        _ => {
                            actions.push(CleanupAction::Drop { value });
                        }
                    }
                }
            }

            actions
        })
    }

    /// Track a value that needs cleanup in the current scope
    pub fn track_value(&mut self, value: HirId, ty: &HirType) {
        if self.needs_cleanup(ty) {
            if let Some(scope) = self.scope_stack.last_mut() {
                scope.values_to_cleanup.push(value);
            } else {
                self.pending_cleanup.push(value);
            }
        }
    }

    /// Check if a type needs cleanup
    pub fn needs_cleanup(&self, ty: &HirType) -> bool {
        if let Some(info) = self.type_cleanup.get(ty) {
            info.needs_cleanup
        } else {
            // Default: pointers and closures need cleanup
            match ty {
                HirType::Ptr(_) => true,
                HirType::Closure(_) => true,
                HirType::Array(elem, _) => self.needs_cleanup(elem),
                HirType::Struct(s) => s.fields.iter().any(|f| self.needs_cleanup(f)),
                _ => false,
            }
        }
    }

    /// Get the destructor for a value (if any)
    fn get_destructor_for_value(&self, _value: HirId) -> Option<HirId> {
        // Would need to look up value's type and find destructor
        // For now, returns None (uses generic cleanup)
        None
    }

    /// Get linearity for a type
    pub fn get_linearity(&self, ty: &HirType) -> HirLinearityKind {
        if let Some(info) = self.type_cleanup.get(ty) {
            info.linearity
        } else {
            // Defaults based on type kind
            match ty {
                // Primitives are unrestricted (Copy)
                HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::F32
                | HirType::F64
                | HirType::Bool => HirLinearityKind::Unrestricted,

                // Pointers are affine by default
                HirType::Ptr(_) => HirLinearityKind::Affine,

                // References are shared
                HirType::Ref { .. } => HirLinearityKind::Shared,

                // Closures are affine
                HirType::Closure(_) => HirLinearityKind::Affine,

                // Other types default to unrestricted
                _ => HirLinearityKind::Unrestricted,
            }
        }
    }

    /// Convert cleanup actions to HIR instructions
    pub fn actions_to_instructions(&self, actions: &[CleanupAction]) -> Vec<HirInstruction> {
        actions
            .iter()
            .map(|action| match action {
                CleanupAction::CallDestructor { value, destructor } => HirInstruction::Call {
                    result: None,
                    callee: HirCallable::Function(*destructor),
                    args: vec![*value],
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                },
                CleanupAction::CallIntrinsic { intrinsic, args } => HirInstruction::Call {
                    result: None,
                    callee: HirCallable::Intrinsic(*intrinsic),
                    args: args.clone(),
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                },
                CleanupAction::Drop { value } => HirInstruction::Call {
                    result: None,
                    callee: HirCallable::Intrinsic(Intrinsic::Drop),
                    args: vec![*value],
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                },
            })
            .collect()
    }

    /// Insert cleanup instructions into a function
    pub fn insert_cleanup(&mut self, func: &mut HirFunction) -> CompilerResult<()> {
        eprintln!(
            "[CLEANUP] Inserting cleanup for function '{}'",
            func.name
                .resolve_global()
                .unwrap_or_else(|| "?".to_string())
        );

        // Track all values that need cleanup
        for (value_id, value) in &func.values {
            self.track_value(*value_id, &value.ty);
        }

        // Create cleanup sequences for return blocks
        let mut cleanup_sequences = HashMap::new();

        for (block_id, block) in &func.blocks {
            if matches!(block.terminator, HirTerminator::Return { .. }) {
                // Get cleanup for pending values
                let cleanup_actions: Vec<CleanupAction> = self
                    .pending_cleanup
                    .iter()
                    .rev()
                    .filter_map(|&value| {
                        if let Some(val) = func.values.get(&value) {
                            if self.needs_cleanup(&val.ty) {
                                if let Some(destructor) = self.custom_destructors.get(&val.ty) {
                                    Some(CleanupAction::CallDestructor {
                                        value,
                                        destructor: *destructor,
                                    })
                                } else {
                                    Some(CleanupAction::Drop { value })
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect();

                if !cleanup_actions.is_empty() {
                    cleanup_sequences
                        .insert(*block_id, self.actions_to_instructions(&cleanup_actions));
                }
            }
        }

        // Insert cleanup sequences
        for (block_id, instructions) in cleanup_sequences {
            if let Some(block) = func.blocks.get_mut(&block_id) {
                // Insert before terminator
                block.instructions.extend(instructions);
            }
        }

        self.pending_cleanup.clear();
        Ok(())
    }
}

/// Bridge between TypedAST LinearTypeChecker and HIR cleanup
pub fn convert_linearity_kind(typed_ast_linearity: &str) -> HirLinearityKind {
    match typed_ast_linearity {
        "Linear" => HirLinearityKind::Linear,
        "Affine" => HirLinearityKind::Affine,
        "Relevant" => HirLinearityKind::Relevant,
        "Unrestricted" => HirLinearityKind::Unrestricted,
        "Unique" => HirLinearityKind::Unique,
        "Shared" => HirLinearityKind::Shared,
        _ => HirLinearityKind::Unrestricted,
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
    fn test_arc_manager_creation() {
        let arc_manager = ARCManager::new();
        assert!(arc_manager.arc_functions.is_empty());
        assert!(arc_manager.arc_types.is_empty());
    }

    #[test]
    fn test_drop_manager_creation() {
        let drop_manager = DropManager::new();
        assert!(drop_manager.drop_types.is_empty());
        assert!(drop_manager.needs_drop.is_empty());
    }

    #[test]
    fn test_escape_analysis_creation() {
        let escape_analysis = EscapeAnalysis::new();
        assert!(escape_analysis.results.is_empty());
    }

    #[test]
    fn test_memory_context_allocation_tracking() {
        let mut ctx = MemoryContext::new(MemoryStrategy::ARC);
        let value_id = HirId::new();

        let alloc_info = AllocationInfo {
            ty: HirType::I32,
            size: Some(4),
            align: 4,
            is_stack: false,
            location: None,
        };

        ctx.track_allocation(value_id, alloc_info);

        assert!(ctx.allocations.contains_key(&value_id));
        assert!(ctx.ref_counts.contains_key(&value_id));
        assert_eq!(ctx.ref_counts[&value_id].count, 1);
    }

    #[test]
    fn test_reference_counting() {
        let mut ctx = MemoryContext::new(MemoryStrategy::ARC);
        let value_id = HirId::new();

        ctx.track_allocation(
            value_id,
            AllocationInfo {
                ty: HirType::I32,
                size: Some(4),
                align: 4,
                is_stack: false,
                location: None,
            },
        );

        // Test retain
        ctx.retain(value_id);
        assert_eq!(ctx.ref_counts[&value_id].count, 2);

        // Test release
        ctx.release(value_id);
        assert_eq!(ctx.ref_counts[&value_id].count, 1);

        // Test release to zero
        ctx.release(value_id);
        assert_eq!(ctx.ref_counts[&value_id].count, 0);
        assert!(ctx.pending_drops.contains(&value_id));
    }
}
