//! # Async/Coroutine Support for HIR
//!
//! Implements async function compilation, coroutine state machines, and
//! async runtime integration for the HIR. This module provides the foundation
//! for async/await syntax and coroutine-based programming models.

use crate::hir::*;
use crate::CompilerResult;
use indexmap::IndexMap;
use std::collections::{HashSet, VecDeque};
use zyntax_typed_ast::InternedString;

/// Async function state machine representation
#[derive(Debug, Clone)]
pub struct AsyncStateMachine {
    /// Unique identifier for this state machine
    pub id: HirId,
    /// Original async function ID
    pub original_function: HirId,
    /// Original async function name (used for generating _new and _poll functions)
    pub original_name: InternedString,
    /// States in the state machine
    pub states: IndexMap<AsyncStateId, AsyncState>,
    /// Initial state
    pub initial_state: AsyncStateId,
    /// Final/completion state
    pub final_state: AsyncStateId,
    /// Captured variables (closure environment)
    pub captures: Vec<AsyncCapture>,
    /// Type of the final result
    pub result_type: HirType,
    /// Values from the original function (constants, etc.) needed by instructions
    pub values: IndexMap<HirId, HirValue>,
}

/// State identifier in an async state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncStateId(pub u32);

/// A state in an async state machine
#[derive(Debug, Clone)]
pub struct AsyncState {
    /// State identifier
    pub id: AsyncStateId,
    /// Local variables in this state
    pub locals: Vec<AsyncLocal>,
    /// Instructions to execute in this state
    pub instructions: Vec<HirInstruction>,
    /// Terminator for this state
    pub terminator: AsyncTerminator,
}

/// Local variable in an async state
#[derive(Debug, Clone)]
pub struct AsyncLocal {
    /// Variable identifier
    pub id: HirId,
    /// Variable name
    pub name: InternedString,
    /// Variable type
    pub ty: HirType,
    /// Whether this variable persists across await points
    pub persists: bool,
}

/// Captured variable for async closure
#[derive(Debug, Clone)]
pub struct AsyncCapture {
    /// Capture identifier
    pub id: HirId,
    /// Captured variable name
    pub name: InternedString,
    /// Type of captured variable
    pub ty: HirType,
    /// Capture mode
    pub mode: AsyncCaptureMode,
}

/// How a variable is captured in async closure
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsyncCaptureMode {
    /// Capture by value (move)
    ByValue,
    /// Capture by immutable reference
    ByRef(HirLifetime),
    /// Capture by mutable reference
    ByMutRef(HirLifetime),
}

/// Terminator instructions for async states
#[derive(Debug, Clone)]
pub enum AsyncTerminator {
    /// Continue to next state unconditionally
    Continue { next_state: AsyncStateId },
    /// Conditional continue - branch to different states based on condition
    CondContinue {
        /// Condition value (bool)
        condition: HirId,
        /// State to continue to if condition is true
        true_state: AsyncStateId,
        /// State to continue to if condition is false
        false_state: AsyncStateId,
    },
    /// Await an async operation
    Await {
        /// Future to await (HirId of the Promise pointer)
        future: HirId,
        /// Where to store the result value when Ready
        result: Option<HirId>,
        /// State to resume after await completes
        resume_state: AsyncStateId,
    },
    /// Yield a value (for generators)
    Yield {
        /// Value to yield
        value: HirId,
        /// State to resume when next() is called
        resume_state: AsyncStateId,
    },
    /// Return from async function
    Return { value: Option<HirId> },
    /// Panic/throw an error
    Panic { error: HirId },
}

/// Async runtime integration
#[derive(Debug, Clone)]
pub struct AsyncRuntime {
    /// Runtime type (e.g., Tokio, async-std, custom)
    pub runtime_type: AsyncRuntimeType,
    /// Executor function for spawning tasks
    pub executor: HirId,
    /// Future trait implementation
    pub future_trait: HirId,
    /// Wake mechanism
    pub waker_type: HirId,
}

/// Type of async runtime
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsyncRuntimeType {
    /// Tokio runtime
    Tokio,
    /// async-std runtime
    AsyncStd,
    /// Custom runtime
    Custom(InternedString),
    /// No runtime (bare futures)
    None,
}

/// Async function compiler
pub struct AsyncCompiler {
    /// State machine counter
    next_state_id: u32,
    /// Generated state machines
    state_machines: IndexMap<HirId, AsyncStateMachine>,
    /// Async runtime configuration
    runtime: Option<AsyncRuntime>,
}

impl AsyncCompiler {
    pub fn new() -> Self {
        Self {
            next_state_id: 0,
            state_machines: IndexMap::new(),
            runtime: None,
        }
    }

    /// Set async runtime configuration
    pub fn set_runtime(&mut self, runtime: AsyncRuntime) {
        self.runtime = Some(runtime);
    }

    /// Compile an async function into a state machine
    pub fn compile_async_function(
        &mut self,
        func: &HirFunction,
    ) -> CompilerResult<AsyncStateMachine> {
        // Analyze the function for await points and control flow
        let await_points = self.find_await_points(func)?;

        // Create state machine structure
        let machine_id = HirId::new();
        let initial_state = self.next_state_id();
        let final_state = self.next_state_id();

        let mut state_machine = AsyncStateMachine {
            id: machine_id,
            original_function: func.id,
            original_name: func.name,
            states: IndexMap::new(),
            initial_state,
            final_state,
            captures: self.analyze_captures(func)?,
            result_type: func
                .signature
                .returns
                .first()
                .cloned()
                .unwrap_or(HirType::Void),
            // Copy values from the original function (constants, etc.) needed by instructions
            values: func.values.clone(),
        };

        // Build states based on await points
        self.build_states(func, &await_points, &mut state_machine)?;

        // Store the state machine
        self.state_machines.insert(func.id, state_machine.clone());

        Ok(state_machine)
    }

    /// Find all await points in a function
    fn find_await_points(&self, func: &HirFunction) -> CompilerResult<Vec<AwaitPoint>> {
        let mut await_points = Vec::new();

        eprintln!(
            "[DEBUG] find_await_points for '{}': {} blocks",
            func.name,
            func.blocks.len()
        );
        for (block_id, block) in &func.blocks {
            eprintln!(
                "[DEBUG]   Block {:?}: {} instructions",
                block_id,
                block.instructions.len()
            );
            for (inst_idx, inst) in block.instructions.iter().enumerate() {
                eprintln!("[DEBUG]     Inst {}: {:?}", inst_idx, inst);
                if let HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Await),
                    args,
                    result,
                    ..
                } = inst
                {
                    await_points.push(AwaitPoint {
                        block_id: *block_id,
                        instruction_index: inst_idx,
                        future: args.get(0).copied().unwrap_or_else(HirId::new),
                        result: *result,
                    });
                }
            }
        }

        Ok(await_points)
    }

    /// Analyze captures for async closure
    ///
    /// This performs escape analysis to determine which variables need to be
    /// captured in the state machine struct. A variable needs to be captured if:
    /// 1. It's a parameter or local defined before an await point
    /// 2. It's used after that await point
    ///
    /// Variables that are only used within a single segment don't need capturing.
    fn analyze_captures(&self, func: &HirFunction) -> CompilerResult<Vec<AsyncCapture>> {
        let mut captures = Vec::new();
        let mut captured_ids = HashSet::new();

        // Find all await point instruction indices
        let await_indices: HashSet<usize> = func
            .blocks
            .values()
            .flat_map(|block| block.instructions.iter().enumerate())
            .filter_map(|(idx, inst)| {
                if let HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Await),
                    ..
                } = inst
                {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        // Build a map from parameter index to SSA value ID
        // The SSA builder creates new HirIds for parameters that are different from param.id
        let param_ssa_ids: Vec<HirId> = (0..func.signature.params.len())
            .map(|idx| {
                // Find the value with HirValueKind::Parameter(idx)
                func.values.iter()
                    .find(|(_, v)| matches!(&v.kind, HirValueKind::Parameter(i) if *i as usize == idx))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| func.signature.params[idx].id) // Fallback to param.id
            })
            .collect();

        log::trace!(
            "[ASYNC] analyze_captures: param_ssa_ids = {:?}",
            param_ssa_ids
        );

        // If no await points AND single block, we still need to capture all parameters for the state machine
        // But for multi-block functions (loops), we need full capture analysis even without await
        if await_indices.is_empty() && func.blocks.len() == 1 {
            for (idx, param) in func.signature.params.iter().enumerate() {
                let ssa_id = param_ssa_ids.get(idx).copied().unwrap_or(param.id);
                captures.push(AsyncCapture {
                    id: ssa_id, // Use SSA value ID, not signature param.id
                    name: param.name,
                    ty: param.ty.clone(),
                    mode: AsyncCaptureMode::ByValue,
                });
            }
            return Ok(captures);
        }

        // For functions with loops containing await, we need a more conservative capture strategy.
        // The linear "before/after await" analysis doesn't work because loops cause values
        // to be used in iterations after the await completes.
        //
        // Strategy: Capture all values that are:
        // 1. Defined in any block
        // 2. Used in any other block (cross-block flow)
        //
        // This ensures values persist across state transitions in loops.

        // Collect all value definitions and uses per block
        let mut defined_in_block: IndexMap<HirId, HashSet<HirId>> = IndexMap::new();
        let mut used_in_block: IndexMap<HirId, HashSet<HirId>> = IndexMap::new();

        for (block_id, block) in &func.blocks {
            let mut defs = HashSet::new();
            let mut uses = HashSet::new();

            for inst in &block.instructions {
                // Collect uses first (before definitions affect them)
                let inst_uses = self.collect_instruction_uses(inst);
                uses.extend(inst_uses);

                // Collect definitions
                if let Some(result) = self.get_instruction_result(inst) {
                    defs.insert(result);
                }
            }

            // Also collect uses from terminators
            match &block.terminator {
                HirTerminator::Return { values } => uses.extend(values.iter().copied()),
                HirTerminator::CondBranch { condition, .. } => {
                    uses.insert(*condition);
                }
                HirTerminator::Switch { value, .. } => {
                    uses.insert(*value);
                }
                _ => {}
            }

            defined_in_block.insert(*block_id, defs);
            used_in_block.insert(*block_id, uses);
        }

        // Find values that need capturing: used in any block but not locally defined
        // We capture all non-constant values that cross block boundaries
        let mut needs_capture: HashSet<HirId> = HashSet::new();

        // First, collect all definitions across all blocks
        let all_defs: HashSet<HirId> = defined_in_block
            .values()
            .flat_map(|defs| defs.iter().copied())
            .collect();

        for (block_id, uses) in &used_in_block {
            let local_defs = defined_in_block.get(block_id).cloned().unwrap_or_default();
            for used_id in uses {
                // Skip if defined locally in this block
                if local_defs.contains(used_id) {
                    continue;
                }

                // Check if this is a constant value (constants don't need capturing)
                if let Some(value) = func.values.get(used_id) {
                    if matches!(value.kind, HirValueKind::Constant(_)) {
                        continue;
                    }
                }

                // This value is used but not locally defined - it needs capturing
                // unless it's already defined somewhere else in the function
                // (in which case it flows between blocks)
                if all_defs.contains(used_id) || func.values.get(used_id).is_some() {
                    needs_capture.insert(*used_id);
                    eprintln!("[DEBUG] analyze_captures: need to capture {:?} (used in block {:?}, not local)",
                        used_id, block_id);
                }
            }
        }

        // IMPORTANT: Capture ALL parameters, not just those used after await.
        // The poll function receives only a pointer to the state machine struct,
        // not the original function parameters. State 0's instructions still
        // reference the original parameter SSA IDs, so we must load them from
        // the state machine at the start of each state.
        for (idx, param) in func.signature.params.iter().enumerate() {
            let ssa_id = param_ssa_ids.get(idx).copied().unwrap_or(param.id);
            if !captured_ids.contains(&ssa_id) {
                captured_ids.insert(ssa_id);
                captures.push(AsyncCapture {
                    id: ssa_id, // Use SSA value ID
                    name: param.name,
                    ty: param.ty.clone(),
                    mode: AsyncCaptureMode::ByValue,
                });
                eprintln!(
                    "[DEBUG] analyze_captures: captured parameter[{}] id={:?} name={}",
                    idx, ssa_id, param.name
                );
            }
        }

        // Capture values that flow across blocks (cross-block live values)
        // These are values defined in one block and used in another
        for value_id in &needs_capture {
            if !captured_ids.contains(value_id) {
                // Get the type from the function's values map
                if let Some(value) = func.values.get(value_id) {
                    captured_ids.insert(*value_id);
                    eprintln!(
                        "[DEBUG] analyze_captures: capturing cross-block value {:?} ty={:?}",
                        value_id, value.ty
                    );
                    captures.push(AsyncCapture {
                        id: *value_id,
                        name: InternedString::new_global("__cross_block_val"),
                        ty: value.ty.clone(),
                        mode: AsyncCaptureMode::ByValue,
                    });
                }
            }
        }

        // Capture phi node results - these represent loop variables that need to persist
        // across state transitions. In a state machine, phi nodes are replaced by loads from
        // the state machine struct, so we need to capture the phi results.
        for block in func.blocks.values() {
            for phi in &block.phis {
                if !captured_ids.contains(&phi.result) {
                    // Get the type from the phi's incoming values
                    let phi_ty = phi
                        .incoming
                        .first()
                        .and_then(|(val_id, _)| func.values.get(val_id))
                        .map(|v| v.ty.clone())
                        .unwrap_or(HirType::I32);

                    captured_ids.insert(phi.result);
                    eprintln!(
                        "[DEBUG] analyze_captures: capturing phi result {:?} ty={:?}",
                        phi.result, phi_ty
                    );
                    captures.push(AsyncCapture {
                        id: phi.result,
                        name: InternedString::new_global("__phi_val"),
                        ty: phi_ty,
                        mode: AsyncCaptureMode::ByValue,
                    });
                }
            }
        }

        // Also add locals that need capturing (if func.locals is populated)
        // IMPORTANT: For async functions with await in loops OR multi-block functions,
        // mutable locals (var declarations) need to be captured even if they're not strictly
        // "used after await" in linear analysis. This is because loop iterations re-enter states.
        let is_multi_block = func.blocks.len() > 1;
        for (local_id, local) in &func.locals {
            let should_capture = needs_capture.contains(local_id)
                || (local.is_mutable && (!await_indices.is_empty() || is_multi_block));

            if should_capture && !captured_ids.contains(local_id) {
                captured_ids.insert(*local_id);
                eprintln!(
                    "[DEBUG] analyze_captures: capturing local {:?} name={} is_mutable={}",
                    local_id, local.name, local.is_mutable
                );
                captures.push(AsyncCapture {
                    id: *local_id,
                    name: local.name,
                    ty: local.ty.clone(),
                    mode: if local.is_mutable {
                        AsyncCaptureMode::ByMutRef(HirLifetime::anonymous())
                    } else {
                        AsyncCaptureMode::ByValue
                    },
                });
            }
        }

        // Capture Promise pointers from await points - these need to persist across polls
        // Each await point creates a nested Promise that must be stored in the state machine
        eprintln!(
            "[DEBUG] analyze_captures: looking for await instructions in {} blocks",
            func.blocks.len()
        );
        for block in func.blocks.values() {
            eprintln!(
                "[DEBUG] analyze_captures: block has {} instructions",
                block.instructions.len()
            );
            for inst in &block.instructions {
                eprintln!("[DEBUG] analyze_captures: checking instruction: {:?}", inst);
                if let HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Await),
                    args,
                    result,
                    ..
                } = inst
                {
                    eprintln!(
                        "[DEBUG] analyze_captures: FOUND AWAIT! args={:?}, result={:?}",
                        args, result
                    );
                    // Capture the future (Promise pointer) - first arg of await
                    if let Some(future_id) = args.first() {
                        eprintln!(
                            "[DEBUG] analyze_captures: future_id={:?}, already_captured={}",
                            future_id,
                            captured_ids.contains(future_id)
                        );
                        if !captured_ids.contains(future_id) {
                            captured_ids.insert(*future_id);
                            // Get the type of the future from the values map
                            let future_ty =
                                func.values.get(future_id).map(|v| v.ty.clone()).unwrap_or(
                                    HirType::Ptr(Box::new(HirType::Opaque(
                                        InternedString::new_global("Promise"),
                                    ))),
                                );
                            eprintln!(
                                "[DEBUG] Capturing await future: {:?} with type {:?}",
                                future_id, future_ty
                            );
                            captures.push(AsyncCapture {
                                id: *future_id,
                                name: InternedString::new_global("__awaited_promise"),
                                ty: future_ty,
                                mode: AsyncCaptureMode::ByValue,
                            });
                        }
                    }
                    // Also capture the result slot if present - the resume state needs it
                    if let Some(result_id) = result {
                        if !captured_ids.contains(result_id) {
                            captured_ids.insert(*result_id);
                            let result_ty = func
                                .values
                                .get(result_id)
                                .map(|v| v.ty.clone())
                                .unwrap_or(HirType::I64);
                            eprintln!(
                                "[DEBUG] Capturing await result slot: {:?} with type {:?}",
                                result_id, result_ty
                            );
                            captures.push(AsyncCapture {
                                id: *result_id,
                                name: InternedString::new_global("__await_result"),
                                ty: result_ty,
                                mode: AsyncCaptureMode::ByValue,
                            });
                        }
                    }
                }
            }
        }
        for (i, c) in captures.iter().enumerate() {
            eprintln!(
                "[DEBUG] analyze_captures: final capture[{}]: id={:?} name={} ty={:?}",
                i, c.id, c.name, c.ty
            );
        }
        eprintln!(
            "[DEBUG] analyze_captures: total captures = {}",
            captures.len()
        );

        // If no captures identified but we have parameters, capture all parameters
        // (conservative fallback for safety)
        // IMPORTANT: Use the parameter VALUE id from func.values, not the param declaration id.
        // Instructions reference the value id, not the declaration id.
        if captures.is_empty() && !func.signature.params.is_empty() {
            log::trace!("[ASYNC] Capturing parameters for {:?}", func.name);
            for (param_idx, param) in func.signature.params.iter().enumerate() {
                // Find the HirId for this parameter's VALUE (not declaration)
                // The SSA builder creates values with HirValueKind::Parameter(idx) for each parameter
                let value_id = func.values.iter()
                    .find(|(_, v)| matches!(v.kind, HirValueKind::Parameter(idx) if idx == param_idx as u32))
                    .map(|(id, _)| *id)
                    .unwrap_or(param.id); // Fallback to declaration id if not found

                log::trace!(
                    "[ASYNC] param[{}] {:?}: declaration id={:?}, value id={:?}",
                    param_idx,
                    param.name,
                    param.id,
                    value_id
                );

                captures.push(AsyncCapture {
                    id: value_id,
                    name: param.name,
                    ty: param.ty.clone(),
                    mode: AsyncCaptureMode::ByValue,
                });
            }
        }

        Ok(captures)
    }

    /// Collect all HirIds used by an instruction
    fn collect_instruction_uses(&self, inst: &HirInstruction) -> Vec<HirId> {
        match inst {
            HirInstruction::Binary { left, right, .. } => vec![*left, *right],
            HirInstruction::Unary { operand, .. } => vec![*operand],
            HirInstruction::Load { ptr, .. } => vec![*ptr],
            HirInstruction::Store { ptr, value, .. } => vec![*ptr, *value],
            HirInstruction::Call { args, .. } => args.clone(),
            HirInstruction::IndirectCall { func_ptr, args, .. } => {
                let mut uses = vec![*func_ptr];
                uses.extend(args.iter().copied());
                uses
            }
            HirInstruction::GetElementPtr { ptr, indices, .. } => {
                let mut uses = vec![*ptr];
                uses.extend(indices.iter().copied());
                uses
            }
            HirInstruction::ExtractValue { aggregate, .. } => vec![*aggregate],
            HirInstruction::InsertValue {
                aggregate, value, ..
            } => vec![*aggregate, *value],
            HirInstruction::Cast { operand, .. } => vec![*operand],
            HirInstruction::Select {
                condition,
                true_val,
                false_val,
                ..
            } => {
                vec![*condition, *true_val, *false_val]
            }
            HirInstruction::Atomic { ptr, value, .. } => {
                let mut uses = vec![*ptr];
                if let Some(v) = value {
                    uses.push(*v);
                }
                uses
            }
            HirInstruction::CreateUnion { value, .. } => vec![*value],
            HirInstruction::GetUnionDiscriminant { union_val, .. } => vec![*union_val],
            HirInstruction::ExtractUnionValue { union_val, .. } => vec![*union_val],
            HirInstruction::CreateTraitObject {
                data_ptr,
                vtable_id,
                ..
            } => {
                vec![*data_ptr, *vtable_id]
            }
            _ => vec![],
        }
    }

    /// Get the result HirId of an instruction if it produces one
    fn get_instruction_result(&self, inst: &HirInstruction) -> Option<HirId> {
        match inst {
            HirInstruction::Binary { result, .. }
            | HirInstruction::Unary { result, .. }
            | HirInstruction::Alloca { result, .. }
            | HirInstruction::Load { result, .. }
            | HirInstruction::GetElementPtr { result, .. }
            | HirInstruction::ExtractValue { result, .. }
            | HirInstruction::InsertValue { result, .. }
            | HirInstruction::Cast { result, .. }
            | HirInstruction::Select { result, .. }
            | HirInstruction::Atomic { result, .. }
            | HirInstruction::CreateUnion { result, .. }
            | HirInstruction::GetUnionDiscriminant { result, .. }
            | HirInstruction::ExtractUnionValue { result, .. }
            | HirInstruction::CreateTraitObject { result, .. } => Some(*result),
            HirInstruction::Call { result, .. } | HirInstruction::IndirectCall { result, .. } => {
                *result
            }
            HirInstruction::Store { .. } | HirInstruction::Fence { .. } => None,
            _ => None,
        }
    }

    /// Build state machine states
    fn build_states(
        &mut self,
        func: &HirFunction,
        await_points: &[AwaitPoint],
        state_machine: &mut AsyncStateMachine,
    ) -> CompilerResult<()> {
        // Split function at await points
        let segments = self.split_at_await_points(func, await_points)?;

        // Pre-allocate state IDs for all segments to ensure consistency
        // State 0 is the initial state (already allocated)
        // States 1..n are for subsequent segments
        let mut state_ids: Vec<AsyncStateId> = Vec::with_capacity(segments.len());
        for i in 0..segments.len() {
            if i == 0 {
                state_ids.push(state_machine.initial_state);
            } else {
                state_ids.push(self.next_state_id());
            }
        }

        log::trace!(
            "[ASYNC] build_states: allocated {} state IDs: {:?}",
            state_ids.len(),
            state_ids
        );

        // Find which await point corresponds to which segment
        // For multi-block functions, await points might not align 1:1 with segments
        let mut segment_await_map: IndexMap<usize, &AwaitPoint> = IndexMap::new();
        for await_point in await_points {
            // Find the segment index that corresponds to this await point's block
            // The segment just before the await contains the Call instruction
            for (seg_idx, segment) in segments.iter().enumerate() {
                // Check if this segment ends with a Branch (indicating it precedes an await)
                // and contains the future-producing instruction
                if matches!(segment.terminator, HirTerminator::Branch { .. }) {
                    // Check if the await's future is produced by an instruction in this segment
                    let contains_future_call = segment.instructions.iter().any(|inst| {
                        if let HirInstruction::Call {
                            result: Some(r), ..
                        } = inst
                        {
                            *r == await_point.future
                        } else {
                            false
                        }
                    });
                    if contains_future_call {
                        segment_await_map.insert(seg_idx, await_point);
                        break;
                    }
                }
            }
        }

        // Create states for each segment
        for (i, segment) in segments.iter().enumerate() {
            let state_id = state_ids[i];

            // Determine terminator based on segment's original terminator
            // and whether this segment has an await
            let terminator = if let Some(await_point) = segment_await_map.get(&i) {
                // This segment ends with an await
                let resume_state = if i + 1 < segments.len() {
                    state_ids[i + 1]
                } else {
                    // Should not happen - await should always have a resume segment
                    state_machine.initial_state
                };
                log::trace!(
                    "[ASYNC] build_states: segment {} has await, resume_state = {:?}",
                    i,
                    resume_state
                );
                AsyncTerminator::Await {
                    future: await_point.future,
                    result: await_point.result,
                    resume_state,
                }
            } else {
                // No await in this segment - use original terminator
                // Use resolved_targets if available (set by split_multi_block_preserving_cfg)
                match &segment.terminator {
                    HirTerminator::Return { values } => {
                        log::trace!("[ASYNC] build_states: segment {} returns", i);
                        AsyncTerminator::Return {
                            value: values.first().copied(),
                        }
                    }
                    HirTerminator::Branch { .. } => {
                        // Unconditional branch - use resolved_targets if available
                        let next_state = if let Some((target_seg, _)) = segment.resolved_targets {
                            if target_seg < state_ids.len() {
                                state_ids[target_seg]
                            } else {
                                if i + 1 < state_ids.len() {
                                    state_ids[i + 1]
                                } else {
                                    state_machine.initial_state
                                }
                            }
                        } else {
                            // No resolved target - use next sequential state
                            if i + 1 < state_ids.len() {
                                state_ids[i + 1]
                            } else {
                                state_machine.initial_state
                            }
                        };
                        log::trace!("[ASYNC] build_states: segment {} has Branch, resolved_targets={:?}, next_state = {:?}",
                            i, segment.resolved_targets, next_state);
                        AsyncTerminator::Continue { next_state }
                    }
                    HirTerminator::CondBranch { condition, .. } => {
                        // Conditional branch - use resolved_targets if available
                        let (true_state, false_state) =
                            if let Some((true_seg, Some(false_seg))) = segment.resolved_targets {
                                let ts = if true_seg < state_ids.len() {
                                    state_ids[true_seg]
                                } else {
                                    state_machine.initial_state
                                };
                                let fs = if false_seg < state_ids.len() {
                                    state_ids[false_seg]
                                } else {
                                    state_machine.initial_state
                                };
                                (ts, fs)
                            } else {
                                // No resolved targets - fallback to sequential (not ideal but safe)
                                let next = if i + 1 < state_ids.len() {
                                    state_ids[i + 1]
                                } else {
                                    state_machine.initial_state
                                };
                                (next, next)
                            };

                        log::trace!("[ASYNC] build_states: segment {} has CondBranch, resolved_targets={:?}, condition={:?}, true_state={:?}, false_state={:?}",
                            i, segment.resolved_targets, condition, true_state, false_state);
                        AsyncTerminator::CondContinue {
                            condition: *condition,
                            true_state,
                            false_state,
                        }
                    }
                    HirTerminator::Switch { .. } => {
                        // TODO: Handle switch with proper case mapping
                        let next_state = if let Some((target_seg, _)) = segment.resolved_targets {
                            if target_seg < state_ids.len() {
                                state_ids[target_seg]
                            } else {
                                state_machine.initial_state
                            }
                        } else {
                            if i + 1 < segments.len() {
                                state_ids[i + 1]
                            } else {
                                state_machine.initial_state
                            }
                        };
                        log::trace!(
                            "[ASYNC] build_states: segment {} has Switch, next_state = {:?}",
                            i,
                            next_state
                        );
                        AsyncTerminator::Continue { next_state }
                    }
                    _ => {
                        log::trace!(
                            "[ASYNC] build_states: segment {} has other terminator: {:?}",
                            i,
                            segment.terminator
                        );
                        AsyncTerminator::Return { value: None }
                    }
                }
            };

            let state = AsyncState {
                id: state_id,
                locals: self.extract_locals_from_segment(segment)?,
                instructions: segment.instructions.clone(),
                terminator,
            };

            log::trace!(
                "[ASYNC] build_states: created state {:?} with {} instructions",
                state_id,
                state.instructions.len()
            );
            eprintln!(
                "[DEBUG] build_states: State {:?} has {} instructions:",
                state_id,
                state.instructions.len()
            );
            for (idx, inst) in state.instructions.iter().enumerate() {
                eprintln!("[DEBUG]   State {:?} inst[{}]: {:?}", state_id, idx, inst);
            }
            state_machine.states.insert(state_id, state);
        }

        Ok(())
    }

    /// Split function into segments at await points
    ///
    /// This traverses all blocks in the function's CFG, collecting instructions
    /// and splitting into segments whenever an await point is encountered.
    /// Each segment represents a continuous sequence of operations that can
    /// execute without yielding.
    fn split_at_await_points(
        &self,
        func: &HirFunction,
        await_points: &[AwaitPoint],
    ) -> CompilerResult<Vec<CodeSegment>> {
        let mut segments: Vec<CodeSegment> = Vec::new();

        // Build a map of await points by (block_id, instruction_index)
        let await_point_map: HashSet<(HirId, usize)> = await_points
            .iter()
            .map(|ap| (ap.block_id, ap.instruction_index))
            .collect();

        // For simple functions with a single block (common case), use optimized path
        if func.blocks.len() == 1 {
            return self.split_single_block(func, await_points);
        }

        // For multi-block functions (loops, conditionals), preserve CFG structure
        // Each block becomes a segment, and blocks containing await are split
        // into pre-await and post-await segments
        self.split_multi_block_preserving_cfg(func, await_points)
    }

    /// Optimized path for single-block functions (most common case)
    ///
    /// For single-block async functions with await points:
    /// - Segments BEFORE an await get a Branch terminator (to signal await point)
    /// - The FINAL segment gets the original terminator (usually Return)
    ///
    /// This ensures that `segment_await_map` in `build_states` correctly identifies
    /// which segments end with await, allowing proper AsyncTerminator::Await creation.
    fn split_single_block(
        &self,
        func: &HirFunction,
        await_points: &[AwaitPoint],
    ) -> CompilerResult<Vec<CodeSegment>> {
        let mut segments = Vec::new();

        if let Some(entry_block) = func.blocks.get(&func.entry_block) {
            log::trace!(
                "[ASYNC] split_single_block: entry block terminator = {:?}",
                entry_block.terminator
            );
            log::trace!(
                "[ASYNC] split_single_block: entry block has {} instructions",
                entry_block.instructions.len()
            );
            eprintln!(
                "[DEBUG] split_single_block: entry block has {} instructions, {} await_points",
                entry_block.instructions.len(),
                await_points.len()
            );
            for ap in await_points {
                eprintln!(
                    "[DEBUG]   await_point at instruction {}",
                    ap.instruction_index
                );
            }

            let mut current_instructions = Vec::new();

            for (i, inst) in entry_block.instructions.iter().enumerate() {
                // Check if this is an await point
                let is_await_point = await_points.iter().any(|ap| ap.instruction_index == i);

                if is_await_point {
                    // Push segment before await with a Branch terminator
                    // The Branch terminator signals to build_states that this segment has an await
                    eprintln!("[DEBUG] split_single_block: instruction {} is await point, creating segment with {} instructions",
                        i, current_instructions.len());
                    segments.push(CodeSegment {
                        instructions: current_instructions,
                        terminator: HirTerminator::Branch {
                            target: HirId::new(),
                        }, // Dummy target
                        resolved_targets: None,
                    });
                    current_instructions = Vec::new();
                    // Skip the await instruction itself (it's handled by the state machine)
                } else {
                    current_instructions.push(inst.clone());
                }
            }

            // Add final segment with the original terminator (usually Return)
            eprintln!("[DEBUG] split_single_block: final segment has {} instructions with original terminator",
                current_instructions.len());
            segments.push(CodeSegment {
                instructions: current_instructions,
                terminator: entry_block.terminator.clone(),
                resolved_targets: None,
            });
        }

        eprintln!(
            "[DEBUG] split_single_block: created {} segments total",
            segments.len()
        );
        for (i, seg) in segments.iter().enumerate() {
            eprintln!(
                "[DEBUG]   segment[{}]: {} instructions, terminator={:?}",
                i,
                seg.instructions.len(),
                seg.terminator
            );
        }

        Ok(segments)
    }

    /// Split multi-block function preserving CFG structure
    ///
    /// For functions with loops or conditionals, we preserve the block structure
    /// and only split blocks that contain await points. Each block becomes a segment,
    /// allowing the state machine to preserve loop back-edges.
    ///
    /// The resolved_targets field in each CodeSegment is populated with the target
    /// segment indices, which build_states uses to create proper state transitions.
    fn split_multi_block_preserving_cfg(
        &self,
        func: &HirFunction,
        await_points: &[AwaitPoint],
    ) -> CompilerResult<Vec<CodeSegment>> {
        let mut segments = Vec::new();
        // Track which segment each original block maps to
        let mut block_to_segment: IndexMap<HirId, usize> = IndexMap::new();

        // Build a map of await points by block_id
        let await_points_by_block: IndexMap<HirId, Vec<&AwaitPoint>> =
            await_points.iter().fold(IndexMap::new(), |mut acc, ap| {
                acc.entry(ap.block_id).or_default().push(ap);
                acc
            });

        // Process blocks in BFS order from entry
        let mut block_order = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(func.entry_block);

        while let Some(block_id) = queue.pop_front() {
            if visited.contains(&block_id) {
                continue;
            }
            visited.insert(block_id);
            block_order.push(block_id);

            if let Some(block) = func.blocks.get(&block_id) {
                // Add successors to queue
                match &block.terminator {
                    HirTerminator::Branch { target } => queue.push_back(*target),
                    HirTerminator::CondBranch {
                        true_target,
                        false_target,
                        ..
                    } => {
                        queue.push_back(*true_target);
                        queue.push_back(*false_target);
                    }
                    HirTerminator::Switch { default, cases, .. } => {
                        queue.push_back(*default);
                        for (_, target) in cases {
                            queue.push_back(*target);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Process each block
        for block_id in &block_order {
            let block = match func.blocks.get(block_id) {
                Some(b) => b,
                None => continue,
            };

            // Record this block's first segment index
            block_to_segment.insert(*block_id, segments.len());

            // Check if this block has await points
            if let Some(block_awaits) = await_points_by_block.get(block_id) {
                // Split this block at await points
                let mut current_instructions = Vec::new();

                for (inst_idx, inst) in block.instructions.iter().enumerate() {
                    let is_await = block_awaits
                        .iter()
                        .any(|ap| ap.instruction_index == inst_idx);

                    if is_await {
                        // Push segment before await (if any)
                        if !current_instructions.is_empty() {
                            segments.push(CodeSegment {
                                instructions: current_instructions,
                                terminator: HirTerminator::Branch {
                                    target: HirId::new(),
                                },
                                resolved_targets: None, // Will be resolved later
                            });
                            current_instructions = Vec::new();
                        }
                    } else {
                        current_instructions.push(inst.clone());
                    }
                }

                // Push remaining instructions with original terminator
                segments.push(CodeSegment {
                    instructions: current_instructions,
                    terminator: block.terminator.clone(),
                    resolved_targets: None, // Will be resolved later
                });
            } else {
                // No await in this block - treat as a single segment
                segments.push(CodeSegment {
                    instructions: block.instructions.clone(),
                    terminator: block.terminator.clone(),
                    resolved_targets: None, // Will be resolved later
                });
            }
        }

        // Log the block_to_segment mapping
        eprintln!("[DEBUG] split_multi_block_preserving_cfg: block_to_segment mapping:");
        for (block_id, seg_idx) in &block_to_segment {
            eprintln!("[DEBUG]   {:?} -> segment {}", block_id, seg_idx);
        }

        // Resolve branch targets from block IDs to segment indices
        for seg in &mut segments {
            match &seg.terminator {
                HirTerminator::Branch { target } => {
                    if let Some(&seg_idx) = block_to_segment.get(target) {
                        seg.resolved_targets = Some((seg_idx, None));
                    }
                }
                HirTerminator::CondBranch {
                    true_target,
                    false_target,
                    ..
                } => {
                    let true_seg = block_to_segment.get(true_target).copied();
                    let false_seg = block_to_segment.get(false_target).copied();
                    if let (Some(t), Some(f)) = (true_seg, false_seg) {
                        seg.resolved_targets = Some((t, Some(f)));
                    }
                }
                HirTerminator::Switch { default, .. } => {
                    // For now just handle the default case for switch
                    if let Some(&seg_idx) = block_to_segment.get(default) {
                        seg.resolved_targets = Some((seg_idx, None));
                    }
                }
                _ => {}
            }
        }

        // Ensure at least one segment
        if segments.is_empty() {
            segments.push(CodeSegment {
                instructions: Vec::new(),
                terminator: HirTerminator::Return { values: vec![] },
                resolved_targets: None,
            });
        }

        eprintln!(
            "[DEBUG] split_multi_block_preserving_cfg: created {} segments from {} blocks",
            segments.len(),
            func.blocks.len()
        );
        for (i, seg) in segments.iter().enumerate() {
            eprintln!(
                "[DEBUG]   Segment {}: {} instructions, terminator={:?}, resolved_targets={:?}",
                i,
                seg.instructions.len(),
                seg.terminator,
                seg.resolved_targets
            );
        }

        Ok(segments)
    }

    /// Extract local variables from a code segment
    ///
    /// This analyzes the instructions in a segment to find:
    /// 1. Alloca instructions (stack allocations)
    /// 2. Values defined by instructions that need to be stored in the state machine
    ///
    /// Local variables are marked with `persists: true` if they need to be stored
    /// in the state machine struct to survive across await points.
    fn extract_locals_from_segment(
        &self,
        segment: &CodeSegment,
    ) -> CompilerResult<Vec<AsyncLocal>> {
        let mut locals = Vec::new();
        let mut defined_values: HashSet<HirId> = HashSet::new();

        for inst in &segment.instructions {
            // Handle Alloca instructions specially - these are explicit local variables
            if let HirInstruction::Alloca { result, ty, .. } = inst {
                locals.push(AsyncLocal {
                    id: *result,
                    name: InternedString::default(), // Will be resolved from debug info if available
                    ty: ty.clone(),
                    persists: true, // Stack allocations typically need to persist
                });
                defined_values.insert(*result);
                continue;
            }

            // Get the result of this instruction if it produces one
            if let Some(result_id) = self.get_instruction_result(inst) {
                // Don't duplicate alloca results
                if defined_values.contains(&result_id) {
                    continue;
                }

                // Determine if this local needs to persist based on whether it's used
                // in subsequent instructions within this segment
                let ty = self.get_instruction_result_type(inst);
                let uses = self.collect_instruction_uses(inst);

                // A local persists if its result is used by other instructions
                // (We'll refine this later based on await point analysis)
                let persists = !uses.is_empty();

                locals.push(AsyncLocal {
                    id: result_id,
                    name: InternedString::default(), // Anonymous local
                    ty,
                    persists,
                });
                defined_values.insert(result_id);
            }
        }

        Ok(locals)
    }

    /// Get the result type of an instruction
    fn get_instruction_result_type(&self, inst: &HirInstruction) -> HirType {
        match inst {
            HirInstruction::Binary { op, .. } => {
                // Binary operations typically preserve the operand type or return bool for comparisons
                match op {
                    BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge => HirType::Bool,
                    _ => HirType::I64, // Default to i64 for arithmetic
                }
            }
            HirInstruction::Unary { op, .. } => match op {
                UnaryOp::Not => HirType::Bool,
                _ => HirType::I64,
            },
            HirInstruction::Alloca { ty, .. } => HirType::Ptr(Box::new(ty.clone())),
            HirInstruction::Load { ty, .. } => ty.clone(),
            HirInstruction::GetElementPtr { ty, .. } => ty.clone(),
            HirInstruction::ExtractValue { ty, .. } => ty.clone(),
            HirInstruction::InsertValue { ty, .. } => ty.clone(),
            HirInstruction::Cast { ty, .. } => ty.clone(),
            HirInstruction::Select { ty, .. } => ty.clone(),
            HirInstruction::Call { .. } => HirType::Void, // Actual return type is in callee signature
            HirInstruction::IndirectCall { return_ty, .. } => return_ty.clone(),
            HirInstruction::Atomic { ty, .. } => ty.clone(),
            HirInstruction::CreateUnion { union_ty, .. } => union_ty.clone(),
            HirInstruction::GetUnionDiscriminant { .. } => HirType::U32, // Discriminant is typically u32
            HirInstruction::ExtractUnionValue { ty, .. } => ty.clone(),
            HirInstruction::CreateTraitObject { .. } => {
                // Fat pointer: {data_ptr, vtable_ptr}
                HirType::Opaque(InternedString::default()) // Placeholder
            }
            _ => HirType::Void,
        }
    }

    /// Generate next state ID
    fn next_state_id(&mut self) -> AsyncStateId {
        let id = AsyncStateId(self.next_state_id);
        self.next_state_id += 1;
        id
    }

    /// Get size of a type in bytes
    fn type_size(&self, ty: &HirType) -> i64 {
        match ty {
            HirType::Bool => 1,
            HirType::I8 | HirType::U8 => 1,
            HirType::I16 | HirType::U16 => 2,
            HirType::I32 | HirType::U32 | HirType::F32 => 4,
            HirType::I64 | HirType::U64 | HirType::F64 => 8,
            HirType::I128 | HirType::U128 => 16,
            HirType::Ptr(_) | HirType::Opaque(_) | HirType::Function(_) => 8, // 64-bit pointers
            HirType::Void => 0,
            HirType::Array(elem, len) => self.type_size(elem) * (*len as i64),
            HirType::Struct(struct_ty) => struct_ty.fields.iter().map(|f| self.type_size(f)).sum(),
            _ => 8, // Default to pointer size for unknown types
        }
    }

    /// Store updated capture values back to the state machine before transitioning states.
    ///
    /// For loop variables, the state's instructions may compute new values (e.g., `i + 1`).
    /// In SSA form, this creates a new HirId for the result. We need to track which new
    /// values should update which captures, then store them back to the state machine struct.
    ///
    /// Algorithm:
    /// 1. For each instruction in the state that takes a capture as an operand and produces a result,
    ///    track the result as an "update" to that capture.
    /// 2. Find the LAST update for each capture (since SSA may have multiple updates).
    /// 3. Store the final values back to their capture slots.
    fn store_updated_captures_back(
        &self,
        state: &AsyncState,
        state_machine: &AsyncStateMachine,
        wrapper: &mut HirFunction,
        state_block: &mut HirBlock,
        self_param: HirId,
    ) {
        // Build a map: capture_id -> last update value
        // We scan instructions to find which results update which captures
        let mut capture_updates: IndexMap<HirId, HirId> = IndexMap::new();
        let capture_ids: HashSet<HirId> = state_machine.captures.iter().map(|c| c.id).collect();

        eprintln!("[DEBUG] store_updated_captures_back: captures are:");
        for (idx, cap) in state_machine.captures.iter().enumerate() {
            eprintln!(
                "[DEBUG]   capture[{}]: id={:?} name={}",
                idx, cap.id, cap.name
            );
        }

        eprintln!(
            "[DEBUG] store_updated_captures_back: state has {} instructions",
            state.instructions.len()
        );
        for inst in &state.instructions {
            eprintln!("[DEBUG] store_updated_captures_back: inst = {:?}", inst);
            match inst {
                HirInstruction::Binary {
                    left,
                    right,
                    result,
                    op,
                    ..
                } => {
                    eprintln!("[DEBUG] store_updated_captures_back: checking Binary left={:?} right={:?} result={:?}",
                        left, right, result);
                    // Pattern: `x = x + something` or `x = x - something` etc.
                    // Only the LEFT operand of an Add/Sub/etc. is considered the "updated" variable,
                    // assuming the pattern follows `var = var + expr`.
                    //
                    // We DON'T consider right operands as being updated, because patterns like
                    // `y = x + z` don't update `x` or `z`, they just use them.
                    match op {
                        BinaryOp::Add
                        | BinaryOp::Sub
                        | BinaryOp::Mul
                        | BinaryOp::Div
                        | BinaryOp::Rem
                        | BinaryOp::And
                        | BinaryOp::Or
                        | BinaryOp::Xor
                        | BinaryOp::Shl
                        | BinaryOp::Shr => {
                            // Only track left operand as potentially updated
                            eprintln!(
                                "[DEBUG]   Is left {:?} in capture_ids? {}",
                                left,
                                capture_ids.contains(left)
                            );
                            if capture_ids.contains(left) {
                                if let Some(capture) =
                                    state_machine.captures.iter().find(|c| c.id == *left)
                                {
                                    eprintln!("[DEBUG]   Found capture for left: {:?}", capture.id);
                                    capture_updates.insert(capture.id, *result);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Now store the updated values back to their capture slots
        let mut current_offset: i64 = 4; // Start after state field
        for capture in &state_machine.captures {
            let capture_size = self.type_size(&capture.ty);
            let align = capture_size;
            current_offset = (current_offset + align - 1) & !(align - 1);

            // Check if this capture was updated
            if let Some(updated_value) = capture_updates.get(&capture.id) {
                eprintln!("[DEBUG] store_updated_captures_back: storing capture {:?} -> updated value {:?} at offset {}",
                    capture.id, updated_value, current_offset);

                // Calculate capture slot pointer
                let offset_const = wrapper.create_value(
                    HirType::I64,
                    HirValueKind::Constant(HirConstant::I64(current_offset)),
                );
                let slot_ptr = wrapper.create_value(
                    HirType::Ptr(Box::new(capture.ty.clone())),
                    HirValueKind::Instruction,
                );
                state_block.add_instruction(HirInstruction::Binary {
                    result: slot_ptr,
                    op: BinaryOp::Add,
                    ty: HirType::I64,
                    left: self_param,
                    right: offset_const,
                });

                // Store the updated value
                state_block.add_instruction(HirInstruction::Store {
                    ptr: slot_ptr,
                    value: *updated_value,
                    align: capture_size as u32,
                    volatile: false,
                });
            }

            current_offset += capture_size;
        }
    }

    /// Generate poll function for the state machine (internal implementation)
    ///
    /// This creates the poll function for the state machine. The function is named
    /// `__{original_name}_poll` (with __ prefix to indicate internal).
    ///
    /// For `async fn compute(x: i32) i32`, this generates:
    /// - `__compute_poll(self: *StateMachine) -> i64` (poll result)
    ///
    /// Poll convention: returns i64 where 0 = Pending, non-zero = Ready(value)
    pub fn generate_poll_function(
        &self,
        state_machine: &AsyncStateMachine,
        arena: &mut zyntax_typed_ast::arena::AstArena,
        original_func: &HirFunction,
    ) -> CompilerResult<HirFunction> {
        // Generate internal poll function name: __{original}_poll
        let poll_fn_name = if let Some(name_str) = state_machine.original_name.resolve_global() {
            arena.intern_string(&format!("__{}_poll", name_str))
        } else {
            arena.intern_string("__anonymous_async_poll")
        };

        // Poll function signature: fn(*mut StateMachine) -> i64
        // Return convention: 0 = Pending, non-zero = Ready(value)
        let wrapper_sig = HirFunctionSignature {
            params: vec![
                // state_machine parameter (pointer to state machine struct)
                HirParam {
                    id: HirId::new(),
                    name: arena.intern_string("state_machine"),
                    ty: HirType::Ptr(Box::new(HirType::U8)), // Generic byte pointer
                    attributes: ParamAttributes::default(),
                },
            ],
            // Use i64 for poll result: 0 = Pending, non-zero = Ready(value)
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };

        let mut wrapper = HirFunction::new(poll_fn_name, wrapper_sig);

        // Check if this is a simple async function (no await points)
        // For simple functions, we can preserve the original block structure
        eprintln!(
            "[DEBUG] generate_poll_function for '{}': states.len() = {}",
            state_machine.original_name,
            state_machine.states.len()
        );
        for (id, state) in &state_machine.states {
            eprintln!(
                "[DEBUG]   State {:?}: {} instructions, terminator = {:?}",
                id,
                state.instructions.len(),
                state.terminator
            );
        }
        if state_machine.states.len() == 1 {
            eprintln!("[DEBUG] Using build_simple_async_wrapper (single state)");
            self.build_simple_async_wrapper(&mut wrapper, state_machine, original_func)?;
        } else {
            eprintln!("[DEBUG] Using build_state_dispatch (multiple states)");
            // Build state machine dispatch logic for functions with await points
            self.build_state_dispatch(&mut wrapper, state_machine, original_func)?;
        }

        Ok(wrapper)
    }

    /// Generate the async entry function that returns *Promise<T>
    ///
    /// This creates a function with the ORIGINAL async function name that:
    /// 1. Allocates the state machine struct (via malloc)
    /// 2. Initializes state to 0
    /// 3. Stores parameters as captures
    /// 4. Allocates Promise struct (via malloc)
    /// 5. Stores state_machine_ptr and poll_fn_ptr in Promise
    /// 6. Returns pointer to Promise
    ///
    /// Promise layout (16 bytes on 64-bit):
    /// - offset 0: state_machine: *mut u8 (8 bytes)
    /// - offset 8: poll_fn: fn(*mut u8) -> i64 (8 bytes)
    pub fn generate_async_entry_function(
        &self,
        state_machine: &AsyncStateMachine,
        poll_func: &HirFunction,
        original_func: &HirFunction,
        arena: &mut zyntax_typed_ast::arena::AstArena,
    ) -> CompilerResult<HirFunction> {
        let entry_block_id = HirId::new();
        let mut blocks = IndexMap::new();
        let mut values = IndexMap::new();
        let mut instructions = Vec::new();

        // Calculate state machine size: 4 bytes for state + size per capture
        // We need to calculate proper alignment: 8-byte aligned for pointers
        let mut state_machine_size: i64 = 4; // state: u32
        for capture in &state_machine.captures {
            // Align to capture size (4 for i32, 8 for pointers)
            let capture_size = self.type_size(&capture.ty);
            let align = capture_size;
            state_machine_size = (state_machine_size + align - 1) & !(align - 1);
            state_machine_size += capture_size;
        }
        eprintln!(
            "[DEBUG] generate_async_entry for {}: state_machine_size={} for {} captures",
            original_func.name,
            state_machine_size,
            state_machine.captures.len()
        );

        // Create constant for state machine size
        let size_const_id = HirId::new();
        values.insert(
            size_const_id,
            HirValue {
                id: size_const_id,
                ty: HirType::I64,
                kind: HirValueKind::Constant(HirConstant::I64(state_machine_size)),
                uses: HashSet::new(),
                span: None,
            },
        );

        // Call malloc to allocate state machine
        let sm_ptr_id = HirId::new();
        values.insert(
            sm_ptr_id,
            HirValue {
                id: sm_ptr_id,
                ty: HirType::Ptr(Box::new(HirType::U8)),
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );

        instructions.push(HirInstruction::Call {
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size_const_id],
            result: Some(sm_ptr_id),
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });

        // Initialize state = 0
        let state_const_id = HirId::new();
        values.insert(
            state_const_id,
            HirValue {
                id: state_const_id,
                ty: HirType::U32,
                kind: HirValueKind::Constant(HirConstant::U32(0)),
                uses: HashSet::new(),
                span: None,
            },
        );

        instructions.push(HirInstruction::Store {
            ptr: sm_ptr_id,
            value: state_const_id,
            align: 4,
            volatile: false,
        });

        // Build a map from parameter SSA IDs to parameter index and type
        // This allows us to identify which captures are parameters
        let param_info: Vec<(HirId, usize, HirType)> = original_func.signature.params.iter()
            .enumerate()
            .filter_map(|(idx, param)| {
                // Find the SSA value ID for this parameter
                original_func.values.iter()
                    .find(|(_, v)| matches!(&v.kind, HirValueKind::Parameter(i) if *i as usize == idx))
                    .map(|(id, _)| (*id, idx, param.ty.clone()))
            })
            .collect();

        // Store captured parameters AND initial values for local variables
        // Parameters need to be stored from function arguments
        // Local variables with constant initial values need those constants stored
        let mut current_offset: i64 = 4; // Start after state field
        for capture in &state_machine.captures {
            // Align to capture size
            let capture_size = self.type_size(&capture.ty);
            let align = capture_size;
            current_offset = (current_offset + align - 1) & !(align - 1);

            // Check if this capture corresponds to a parameter
            if let Some((_, param_idx, param_ty)) =
                param_info.iter().find(|(id, _, _)| *id == capture.id)
            {
                eprintln!(
                    "[DEBUG] generate_async_entry: storing param[{}] at offset {}",
                    param_idx, current_offset
                );

                // Create parameter value reference
                let param_value_id = HirId::new();
                values.insert(
                    param_value_id,
                    HirValue {
                        id: param_value_id,
                        ty: param_ty.clone(),
                        kind: HirValueKind::Parameter(*param_idx as u32),
                        uses: HashSet::new(),
                        span: None,
                    },
                );

                // Create offset constant
                let offset_const_id = HirId::new();
                values.insert(
                    offset_const_id,
                    HirValue {
                        id: offset_const_id,
                        ty: HirType::I64,
                        kind: HirValueKind::Constant(HirConstant::I64(current_offset)),
                        uses: HashSet::new(),
                        span: None,
                    },
                );

                // Calculate pointer: sm_ptr + offset
                let field_ptr = HirId::new();
                values.insert(
                    field_ptr,
                    HirValue {
                        id: field_ptr,
                        ty: HirType::Ptr(Box::new(param_ty.clone())),
                        kind: HirValueKind::Instruction,
                        uses: HashSet::new(),
                        span: None,
                    },
                );

                instructions.push(HirInstruction::Binary {
                    result: field_ptr,
                    op: BinaryOp::Add,
                    ty: HirType::I64,
                    left: sm_ptr_id,
                    right: offset_const_id,
                });

                // Store parameter value with proper alignment
                instructions.push(HirInstruction::Store {
                    ptr: field_ptr,
                    value: param_value_id,
                    align: capture_size as u32,
                    volatile: false,
                });
            } else {
                // For non-parameter captures, try to find their initial constant value
                // Strategy: Look through ALL phi nodes to find ones that reference this capture ID
                // and check their incoming values for constants from the entry block

                let mut found_constant: Option<HirConstant> = None;

                // First check if the capture itself is a constant in the values map
                if let Some(orig_value) = original_func.values.get(&capture.id) {
                    eprintln!(
                        "[DEBUG] generate_async_entry: capture {:?} has value kind {:?}",
                        capture.id, orig_value.kind
                    );
                    if let HirValueKind::Constant(constant) = &orig_value.kind {
                        found_constant = Some(constant.clone());
                    }
                }

                // If not found directly, look for phi nodes that produce this capture
                // The phi's incoming values from the entry block should be the initial values
                // We need to identify the entry block - it's the one with no predecessors
                if found_constant.is_none() {
                    // First, find the entry block (no predecessors)
                    let entry_block_id = original_func
                        .blocks
                        .iter()
                        .find(|(_, block)| block.predecessors.is_empty())
                        .map(|(id, _)| *id);

                    eprintln!("[DEBUG] Entry block id: {:?}", entry_block_id);

                    'phi_search: for (_, block) in &original_func.blocks {
                        for phi in &block.phis {
                            if phi.result == capture.id {
                                eprintln!(
                                    "[DEBUG] Found phi producing capture {:?}: incoming={:?}",
                                    capture.id, phi.incoming
                                );

                                // First try to find the constant incoming from the entry block
                                for (value_id, block_id) in &phi.incoming {
                                    // Check if this incoming is from the entry block
                                    let is_from_entry =
                                        entry_block_id.map(|e| *block_id == e).unwrap_or(false);
                                    if let Some(val) = original_func.values.get(value_id) {
                                        if let HirValueKind::Constant(c) = &val.kind {
                                            eprintln!("[DEBUG] Phi incoming {:?} from block {:?} is constant {:?} (from_entry={})",
                                                value_id, block_id, c, is_from_entry);
                                            if is_from_entry {
                                                // This is definitely the initial value
                                                found_constant = Some(c.clone());
                                                break 'phi_search;
                                            } else if found_constant.is_none() {
                                                // Keep this as a fallback in case we don't find one from entry
                                                found_constant = Some(c.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // If still not found, scan all values for constants that might be initial values
                // This is a heuristic: find constants that are used by the capture ID somehow
                if found_constant.is_none() {
                    // Look for constants in the values map with matching type
                    // For i32 captures with name containing common patterns, try to find their init values
                    // This is fragile but helps for common cases like "i = 1" and "total = 0"
                    let name_str = format!("{}", capture.name);
                    eprintln!(
                        "[DEBUG] Scanning for initial value for capture {:?} (name={}, ty={:?})",
                        capture.id, name_str, capture.ty
                    );

                    // Check if there's a phi somewhere that uses this capture
                    // and has a constant incoming from the entry path
                    for (_, block) in &original_func.blocks {
                        for phi in &block.phis {
                            // Check incoming values for constants
                            for (value_id, _block_id) in &phi.incoming {
                                if let Some(val) = original_func.values.get(value_id) {
                                    if let HirValueKind::Constant(c) = &val.kind {
                                        // Check if this constant's type matches our capture
                                        let const_ty = match c {
                                            HirConstant::I32(_) => HirType::I32,
                                            HirConstant::I64(_) => HirType::I64,
                                            HirConstant::U32(_) => HirType::U32,
                                            HirConstant::U64(_) => HirType::U64,
                                            HirConstant::F32(_) => HirType::F32,
                                            HirConstant::F64(_) => HirType::F64,
                                            HirConstant::Bool(_) => HirType::Bool,
                                            _ => continue,
                                        };
                                        if const_ty == capture.ty && found_constant.is_none() {
                                            eprintln!("[DEBUG] Found matching constant {:?} for capture {:?}", c, capture.id);
                                            found_constant = Some(c.clone());
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(constant) = found_constant {
                    eprintln!("[DEBUG] generate_async_entry: storing initial constant {:?} at offset {} for capture {:?} (name={})",
                        constant, current_offset, capture.id, capture.name);

                    // Create the constant value
                    let const_value_id = HirId::new();
                    values.insert(
                        const_value_id,
                        HirValue {
                            id: const_value_id,
                            ty: capture.ty.clone(),
                            kind: HirValueKind::Constant(constant.clone()),
                            uses: HashSet::new(),
                            span: None,
                        },
                    );

                    // Create offset constant
                    let offset_const_id = HirId::new();
                    values.insert(
                        offset_const_id,
                        HirValue {
                            id: offset_const_id,
                            ty: HirType::I64,
                            kind: HirValueKind::Constant(HirConstant::I64(current_offset)),
                            uses: HashSet::new(),
                            span: None,
                        },
                    );

                    // Calculate pointer: sm_ptr + offset
                    let field_ptr = HirId::new();
                    values.insert(
                        field_ptr,
                        HirValue {
                            id: field_ptr,
                            ty: HirType::Ptr(Box::new(capture.ty.clone())),
                            kind: HirValueKind::Instruction,
                            uses: HashSet::new(),
                            span: None,
                        },
                    );

                    instructions.push(HirInstruction::Binary {
                        result: field_ptr,
                        op: BinaryOp::Add,
                        ty: HirType::I64,
                        left: sm_ptr_id,
                        right: offset_const_id,
                    });

                    // Store the constant value
                    eprintln!("[DEBUG] generate_async_entry: HIR Store ptr={:?} value={:?} (constant={:?}, offset={})",
                        field_ptr, const_value_id, constant, current_offset);
                    instructions.push(HirInstruction::Store {
                        ptr: field_ptr,
                        value: const_value_id,
                        align: capture_size as u32,
                        volatile: false,
                    });
                } else {
                    eprintln!(
                        "[DEBUG] generate_async_entry: NO initial constant found for capture {:?}",
                        capture.id
                    );
                }
            }

            // Move to next field regardless of whether we stored
            current_offset += capture_size;
        }

        // Get the poll function pointer via CreateClosure (with no captures)
        let poll_fn_ptr_id = HirId::new();
        let poll_fn_ptr_ty = HirType::Function(Box::new(HirFunctionType {
            params: vec![HirType::Ptr(Box::new(HirType::U8))],
            returns: vec![HirType::I64],
            lifetime_params: vec![],
            is_variadic: false,
        }));
        values.insert(
            poll_fn_ptr_id,
            HirValue {
                id: poll_fn_ptr_id,
                ty: poll_fn_ptr_ty.clone(),
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );

        // CreateClosure with no captures to get function pointer
        instructions.push(HirInstruction::CreateClosure {
            result: poll_fn_ptr_id,
            closure_ty: poll_fn_ptr_ty,
            function: poll_func.id,
            captures: vec![],
        });

        // Create Promise struct type
        let promise_type = HirType::Struct(HirStructType {
            name: Some(arena.intern_string("Promise")),
            fields: vec![
                HirType::Ptr(Box::new(HirType::U8)), // state_machine
                HirType::Function(Box::new(HirFunctionType {
                    params: vec![HirType::Ptr(Box::new(HirType::U8))],
                    returns: vec![HirType::I64],
                    lifetime_params: vec![],
                    is_variadic: false,
                })), // poll_fn
            ],
            packed: false,
        });

        // Allocate Promise struct on heap (16 bytes for two pointers)
        let promise_size_id = HirId::new();
        values.insert(
            promise_size_id,
            HirValue {
                id: promise_size_id,
                ty: HirType::I64,
                kind: HirValueKind::Constant(HirConstant::I64(16)),
                uses: HashSet::new(),
                span: None,
            },
        );

        let promise_ptr_id = HirId::new();
        values.insert(
            promise_ptr_id,
            HirValue {
                id: promise_ptr_id,
                ty: HirType::Ptr(Box::new(promise_type.clone())),
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );
        instructions.push(HirInstruction::Call {
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![promise_size_id],
            result: Some(promise_ptr_id),
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });

        // Store state_machine_ptr at offset 0
        instructions.push(HirInstruction::Store {
            ptr: promise_ptr_id,
            value: sm_ptr_id,
            align: 8,
            volatile: false,
        });

        // Store poll_fn_ptr at offset 8
        let offset_8_id = HirId::new();
        values.insert(
            offset_8_id,
            HirValue {
                id: offset_8_id,
                ty: HirType::I64,
                kind: HirValueKind::Constant(HirConstant::I64(8)),
                uses: HashSet::new(),
                span: None,
            },
        );

        let poll_fn_ptr_offset_id = HirId::new();
        values.insert(
            poll_fn_ptr_offset_id,
            HirValue {
                id: poll_fn_ptr_offset_id,
                ty: HirType::Ptr(Box::new(HirType::U8)),
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );
        instructions.push(HirInstruction::Binary {
            result: poll_fn_ptr_offset_id,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: promise_ptr_id,
            right: offset_8_id,
        });
        instructions.push(HirInstruction::Store {
            ptr: poll_fn_ptr_offset_id,
            value: poll_fn_ptr_id,
            align: 8,
            volatile: false,
        });

        // Create entry block - return the Promise pointer
        blocks.insert(
            entry_block_id,
            HirBlock {
                id: entry_block_id,
                label: None,
                phis: Vec::new(),
                instructions,
                terminator: HirTerminator::Return {
                    values: vec![promise_ptr_id],
                },
                dominance_frontier: HashSet::new(),
                predecessors: Vec::new(),
                successors: Vec::new(),
            },
        );

        // Create function with original name but *Promise return type
        // IMPORTANT: Use the ORIGINAL function's ID so that other functions can
        // call it using the original ID (e.g., step2 calling step1)
        let func_name = original_func.name;
        let params = original_func.signature.params.clone();

        Ok(HirFunction {
            id: original_func.id, // Use original ID so callers can find us
            name: func_name,
            signature: HirFunctionSignature {
                params,
                returns: vec![HirType::Ptr(Box::new(promise_type))],
                type_params: Vec::new(),
                const_params: Vec::new(),
                lifetime_params: Vec::new(),
                is_variadic: false,
                is_async: false,
                effects: Vec::new(),
                is_pure: false,
            },
            entry_block: entry_block_id,
            blocks,
            locals: IndexMap::new(),
            values,
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::C,
            attributes: FunctionAttributes::default(),
            link_name: None,
        })
    }

    /// Build a simple async wrapper that preserves the original function's block structure
    ///
    /// This is used for async functions WITHOUT await points. Instead of creating
    /// a state machine dispatch, we:
    /// 1. Copy all blocks from the original function
    /// 2. Add parameter loading from state machine at the entry
    /// 3. Transform Return terminators to return Poll::Ready(value)
    fn build_simple_async_wrapper(
        &self,
        wrapper: &mut HirFunction,
        state_machine: &AsyncStateMachine,
        original_func: &HirFunction,
    ) -> CompilerResult<()> {
        // Remove the default entry block created by HirFunction::new()
        // We'll add our own param_load block as the entry
        let old_entry = wrapper.entry_block;
        wrapper.blocks.remove(&old_entry);

        // Copy all values from the original function EXCEPT Parameters
        // Parameters are loaded from the state machine, so we don't copy them.
        // If we copy them, Cranelift will create iconst instructions for them
        // instead of using the loaded values.
        log::trace!("[ASYNC] Copying values from original func to wrapper (excluding Parameters)");
        for (hir_id, hir_value) in &original_func.values {
            // Skip Parameter values - they will be loaded from state machine
            if matches!(hir_value.kind, HirValueKind::Parameter(_)) {
                log::trace!(
                    "[ASYNC] SKIPPING {:?} (will be loaded from state machine)",
                    hir_id
                );
                continue;
            }
            if !wrapper.values.contains_key(hir_id) {
                wrapper.values.insert(*hir_id, hir_value.clone());
            }
        }

        // Get self parameter (state machine pointer)
        let self_param = wrapper.create_value(
            HirType::Ptr(Box::new(HirType::Opaque(InternedString::new_global(
                "StateMachine",
            )))),
            HirValueKind::Parameter(0),
        );

        // Create a new entry block that loads parameters from state machine
        let param_load_block_id = HirId::new();
        let mut param_load_block = HirBlock::new(param_load_block_id);
        param_load_block.label = Some(InternedString::new_global("param_load"));

        // Load captured parameters from the state machine struct
        // Layout must match generate_async_entry_function's allocation
        log::trace!(
            "[ASYNC] build_simple_async_wrapper: {} captures",
            state_machine.captures.len()
        );
        let mut current_offset: i64 = 4; // Start after state field
        for capture in &state_machine.captures {
            // Align to capture size (must match generate_async_entry_function)
            let capture_size = self.type_size(&capture.ty);
            let align = capture_size;
            current_offset = (current_offset + align - 1) & !(align - 1);

            let offset_const = wrapper.create_value(
                HirType::I64,
                HirValueKind::Constant(HirConstant::I64(current_offset)),
            );
            let field_ptr = wrapper.create_value(
                HirType::Ptr(Box::new(capture.ty.clone())),
                HirValueKind::Instruction,
            );

            param_load_block.add_instruction(HirInstruction::Binary {
                result: field_ptr,
                op: BinaryOp::Add,
                ty: HirType::I64,
                left: self_param,
                right: offset_const,
            });

            param_load_block.add_instruction(HirInstruction::Load {
                result: capture.id,
                ty: capture.ty.clone(),
                ptr: field_ptr,
                align: capture_size as u32,
                volatile: false,
            });

            wrapper.values.insert(
                capture.id,
                HirValue {
                    id: capture.id,
                    ty: capture.ty.clone(),
                    kind: HirValueKind::Instruction,
                    uses: HashSet::new(),
                    span: None,
                },
            );

            current_offset += capture_size;
        }

        // Branch to original entry block
        param_load_block.set_terminator(HirTerminator::Branch {
            target: original_func.entry_block,
        });

        // Copy all blocks from original function, transforming Return terminators
        for (block_id, block) in &original_func.blocks {
            let mut new_block = HirBlock::new(*block_id);
            new_block.label = block.label;
            new_block.phis = block.phis.clone();
            new_block.instructions = block.instructions.clone();

            // Transform terminator
            let new_terminator = match &block.terminator {
                HirTerminator::Return { values } => {
                    eprintln!("[DEBUG] Transforming return with values: {:?}", values);
                    // Transform return to Poll::Ready(value)
                    if let Some(v) = values.first() {
                        eprintln!(
                            "[DEBUG] Return value: {:?}, in wrapper.values? {}",
                            v,
                            wrapper.values.contains_key(v)
                        );

                        // If the return value isn't in the wrapper's values, we need to add it
                        // This can happen for phi node results that are used in returns
                        if !wrapper.values.contains_key(v) {
                            if let Some(orig_value) = original_func.values.get(v) {
                                eprintln!(
                                    "[DEBUG] Adding missing return value {:?} from original_func",
                                    v
                                );
                                wrapper.values.insert(*v, orig_value.clone());
                            } else {
                                eprintln!("[DEBUG] WARNING: Return value {:?} not found in original_func.values either!", v);
                                // Check if it's a phi result
                                for (blk_id, blk) in &original_func.blocks {
                                    for phi in &blk.phis {
                                        if phi.result == *v {
                                            eprintln!("[DEBUG] Found phi producing return value in block {:?}", blk_id);
                                            // Add the phi result as a value
                                            let phi_ty = original_func
                                                .values
                                                .get(&phi.incoming[0].0)
                                                .map(|val| val.ty.clone())
                                                .unwrap_or(HirType::I32);
                                            wrapper.values.insert(
                                                *v,
                                                HirValue {
                                                    id: *v,
                                                    ty: phi_ty,
                                                    kind: HirValueKind::Instruction,
                                                    uses: HashSet::new(),
                                                    span: None,
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // Cast to i64
                        let cast_result =
                            wrapper.create_value(HirType::I64, HirValueKind::Instruction);
                        new_block.add_instruction(HirInstruction::Cast {
                            op: CastOp::SExt,
                            result: cast_result,
                            ty: HirType::I64,
                            operand: *v,
                        });
                        HirTerminator::Return {
                            values: vec![cast_result],
                        }
                    } else {
                        // Void return -> return 1 (Ready with no value)
                        let ready_const = wrapper.create_value(
                            HirType::I64,
                            HirValueKind::Constant(HirConstant::I64(1)),
                        );
                        HirTerminator::Return {
                            values: vec![ready_const],
                        }
                    }
                }
                other => other.clone(),
            };
            new_block.set_terminator(new_terminator);

            wrapper.blocks.insert(*block_id, new_block);
        }

        // Add the param load block and set it as entry
        wrapper.blocks.insert(param_load_block_id, param_load_block);
        wrapper.entry_block = param_load_block_id;

        log::trace!(
            "[ASYNC] build_simple_async_wrapper complete: entry={:?}, blocks={}",
            wrapper.entry_block,
            wrapper.blocks.len()
        );

        Ok(())
    }

    /// Build state machine dispatch logic
    ///
    /// This creates the switch dispatch that routes to the correct state handler
    /// based on the current state field in the state machine struct.
    fn build_state_dispatch(
        &self,
        wrapper: &mut HirFunction,
        state_machine: &AsyncStateMachine,
        original_func: &HirFunction,
    ) -> CompilerResult<()> {
        // Copy all values from the original function to the wrapper EXCEPT Parameters.
        // Parameters from the original function are now stored in the state machine struct,
        // not as function parameters to the poll function. The poll function only has
        // one parameter: the state machine pointer itself.
        //
        // If we copy Parameter values, we'll have conflicting Parameter(0) definitions
        // (one from original func, one for state machine pointer) which breaks Cranelift.
        for (hir_id, hir_value) in &original_func.values {
            // Skip Parameter values - they will be loaded from state machine
            if matches!(hir_value.kind, HirValueKind::Parameter(_)) {
                log::trace!(
                    "[ASYNC] build_state_dispatch: SKIPPING value {:?} (Parameter)",
                    hir_id
                );
                continue;
            }
            if !wrapper.values.contains_key(hir_id) {
                wrapper.values.insert(*hir_id, hir_value.clone());
            }
        }

        // Create switch on current state
        let state_field = wrapper.create_value(HirType::U32, HirValueKind::Instruction);

        // Get self parameter (state machine pointer) - first parameter
        let self_param = wrapper.create_value(
            HirType::Ptr(Box::new(HirType::Opaque(InternedString::new_global(
                "StateMachine",
            )))),
            HirValueKind::Parameter(0),
        );

        // Load current state from state machine's first field (state: u32)
        let load_state = HirInstruction::Load {
            result: state_field,
            ty: HirType::U32,
            ptr: self_param,
            align: 4,
            volatile: false,
        };

        // Create blocks for each state and build switch cases
        let mut cases = Vec::new();
        let mut state_blocks = Vec::new();

        for (state_id, state) in &state_machine.states {
            // Create a block for this state's handler
            let block_id = HirId::new();
            let mut state_block = HirBlock::new(block_id);
            state_block.label = Some(InternedString::new_global(&format!("state_{}", state_id.0)));

            // Load captured parameters from the state machine struct
            // Layout must match generate_async_entry_function's allocation:
            // { state: u32 (offset 0), captures... with proper alignment }
            eprintln!(
                "[DEBUG] build_state_dispatch for {}: state {} has {} captures",
                original_func.name,
                state_id.0,
                state_machine.captures.len()
            );
            for c in &state_machine.captures {
                eprintln!(
                    "[DEBUG]   Capture: {:?} name={} ty={:?}",
                    c.id, c.name, c.ty
                );
            }
            let mut current_offset: i64 = 4; // Start after state field
            for capture in &state_machine.captures {
                // Align to capture size (must match generate_async_entry_function)
                let capture_size = self.type_size(&capture.ty);
                let align = capture_size;
                current_offset = (current_offset + align - 1) & !(align - 1);

                // Create offset constant
                let offset_const = wrapper.create_value(
                    HirType::I64,
                    HirValueKind::Constant(HirConstant::I64(current_offset)),
                );
                eprintln!(
                    "[DEBUG] build_state_dispatch: loading capture {:?} at offset {} (size={})",
                    capture.id, current_offset, capture_size
                );

                // Calculate pointer: self_param + offset
                let field_ptr = wrapper.create_value(
                    HirType::Ptr(Box::new(capture.ty.clone())),
                    HirValueKind::Instruction,
                );
                state_block.add_instruction(HirInstruction::Binary {
                    result: field_ptr,
                    op: BinaryOp::Add,
                    ty: HirType::I64,
                    left: self_param,
                    right: offset_const,
                });
                eprintln!("[DEBUG] build_state_dispatch: field_ptr={:?}", field_ptr);

                // Load the captured value - use the ORIGINAL capture HirId so instructions can reference it
                // This is the key: we're defining a value with the capture's original HirId
                state_block.add_instruction(HirInstruction::Load {
                    result: capture.id,
                    ty: capture.ty.clone(),
                    ptr: field_ptr,
                    align: capture_size as u32,
                    volatile: false,
                });
                eprintln!(
                    "[DEBUG] build_state_dispatch: created Load with result={:?}",
                    capture.id
                );

                // Register this value in the wrapper's values map
                wrapper.values.insert(
                    capture.id,
                    HirValue {
                        id: capture.id,
                        ty: capture.ty.clone(),
                        kind: HirValueKind::Instruction,
                        uses: HashSet::new(),
                        span: None,
                    },
                );

                // Move to next field
                current_offset += capture_size;
            }

            // Add instructions from this state (for non-Await terminators)
            // For Await terminators, we handle instructions specially below
            if !matches!(&state.terminator, AsyncTerminator::Await { .. }) {
                for inst in &state.instructions {
                    state_block.add_instruction(inst.clone());
                }
            }

            // Debug: print all value IDs used in this state's instructions
            eprintln!("[DEBUG] State {} instruction value references:", state_id.0);
            for (i, inst) in state.instructions.iter().enumerate() {
                eprintln!("[DEBUG]   inst[{}]: {:?}", i, inst);
            }

            // Set terminator based on async terminator
            // Poll convention: 0 = Pending, non-zero = Ready(value)
            let terminator = match &state.terminator {
                AsyncTerminator::Return { value } => {
                    // Return Poll::Ready(value)
                    // For Ready, we return the value itself (or 1 if void)
                    if let Some(v) = value {
                        // Cast the return value to i64 (poll function returns i64)
                        // We need to sign-extend i32 to i64
                        let cast_result =
                            wrapper.create_value(HirType::I64, HirValueKind::Instruction);
                        state_block.add_instruction(HirInstruction::Cast {
                            op: CastOp::SExt,
                            result: cast_result,
                            ty: HirType::I64,
                            operand: *v,
                        });
                        HirTerminator::Return {
                            values: vec![cast_result],
                        }
                    } else {
                        // Void return - create a "Ready with no value" constant (use 1)
                        let ready_const = wrapper.create_value(
                            HirType::I64,
                            HirValueKind::Constant(HirConstant::I64(1)),
                        );
                        HirTerminator::Return {
                            values: vec![ready_const],
                        }
                    }
                }
                AsyncTerminator::Await {
                    future,
                    result,
                    resume_state,
                } => {
                    // Poll the nested future and handle the result
                    //
                    // On FIRST entry to this state: Promise slot is null (0)
                    //   - Execute state instructions (Call to async function)
                    //   - Store the Promise in state machine
                    //   - Poll the Promise
                    //
                    // On SUBSEQUENT entries: Promise slot has valid pointer
                    //   - Skip the Call (Promise already exists)
                    //   - Poll the existing Promise
                    //
                    // Promise layout (16 bytes):
                    // - offset 0: state_machine: *mut u8 (8 bytes)
                    // - offset 8: poll_fn: fn(*mut u8) -> i64 (8 bytes)

                    // Find the offset of the future capture (Promise slot) in state machine
                    let mut future_offset: i64 = 4; // Start after state field
                    for capture in &state_machine.captures {
                        let capture_size = self.type_size(&capture.ty);
                        let align = capture_size;
                        future_offset = (future_offset + align - 1) & !(align - 1);
                        if capture.id == *future {
                            break;
                        }
                        future_offset += capture_size;
                    }

                    // Check if Promise (loaded from capture) is null
                    let null_const = wrapper
                        .create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
                    let promise_is_null =
                        wrapper.create_value(HirType::Bool, HirValueKind::Instruction);
                    state_block.add_instruction(HirInstruction::Binary {
                        result: promise_is_null,
                        op: BinaryOp::Eq,
                        ty: HirType::I64,
                        left: *future, // The loaded capture (Promise pointer or 0)
                        right: null_const,
                    });

                    // Create blocks for the two paths
                    let call_block_id = HirId::new();
                    let poll_block_id = HirId::new();

                    state_block.set_terminator(HirTerminator::CondBranch {
                        condition: promise_is_null,
                        true_target: call_block_id, // Null -> need to call
                        false_target: poll_block_id, // Non-null -> skip to poll
                    });

                    // === CALL BLOCK: Execute Call, store Promise, jump to poll ===
                    let mut call_block = HirBlock::new(call_block_id);
                    call_block.label = Some(InternedString::new_global(&format!(
                        "await_call_{}",
                        state_id.0
                    )));

                    // Execute the state's instructions (includes the Call)
                    for inst in &state.instructions {
                        call_block.add_instruction(inst.clone());
                    }

                    // Now *future HirId has the Call result (Promise pointer)
                    // Store it in the state machine at the Promise slot
                    let future_offset_const = wrapper.create_value(
                        HirType::I64,
                        HirValueKind::Constant(HirConstant::I64(future_offset)),
                    );
                    let future_slot_ptr = wrapper.create_value(
                        HirType::Ptr(Box::new(HirType::Ptr(Box::new(HirType::U8)))),
                        HirValueKind::Instruction,
                    );
                    call_block.add_instruction(HirInstruction::Binary {
                        result: future_slot_ptr,
                        op: BinaryOp::Add,
                        ty: HirType::I64,
                        left: self_param,
                        right: future_offset_const,
                    });
                    call_block.add_instruction(HirInstruction::Store {
                        ptr: future_slot_ptr,
                        value: *future,
                        align: 8,
                        volatile: false,
                    });

                    call_block.set_terminator(HirTerminator::Branch {
                        target: poll_block_id,
                    });
                    wrapper.blocks.insert(call_block_id, call_block);

                    // === POLL BLOCK: Poll the Promise (whether newly created or loaded) ===
                    let mut poll_block = HirBlock::new(poll_block_id);
                    poll_block.label = Some(InternedString::new_global(&format!(
                        "await_poll_{}",
                        state_id.0
                    )));

                    // Since we come here from two paths, we need to reload the Promise from state machine
                    // to ensure we have a valid value regardless of which path was taken
                    let future_offset_const2 = wrapper.create_value(
                        HirType::I64,
                        HirValueKind::Constant(HirConstant::I64(future_offset)),
                    );
                    let future_slot_ptr2 = wrapper.create_value(
                        HirType::Ptr(Box::new(HirType::Ptr(Box::new(HirType::U8)))),
                        HirValueKind::Instruction,
                    );
                    poll_block.add_instruction(HirInstruction::Binary {
                        result: future_slot_ptr2,
                        op: BinaryOp::Add,
                        ty: HirType::I64,
                        left: self_param,
                        right: future_offset_const2,
                    });

                    let promise_ptr = wrapper.create_value(
                        HirType::Ptr(Box::new(HirType::U8)),
                        HirValueKind::Instruction,
                    );
                    poll_block.add_instruction(HirInstruction::Load {
                        result: promise_ptr,
                        ty: HirType::Ptr(Box::new(HirType::U8)),
                        ptr: future_slot_ptr2,
                        align: 8,
                        volatile: false,
                    });

                    // Load the nested state machine pointer from Promise[0]
                    let nested_sm = wrapper.create_value(
                        HirType::Ptr(Box::new(HirType::U8)),
                        HirValueKind::Instruction,
                    );
                    poll_block.add_instruction(HirInstruction::Load {
                        result: nested_sm,
                        ty: HirType::Ptr(Box::new(HirType::U8)),
                        ptr: promise_ptr,
                        align: 8,
                        volatile: false,
                    });

                    // Calculate Promise[8] = promise_ptr + 8
                    let offset_8 = wrapper
                        .create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(8)));
                    let poll_fn_ptr_addr = wrapper.create_value(
                        HirType::Ptr(Box::new(HirType::U8)),
                        HirValueKind::Instruction,
                    );
                    poll_block.add_instruction(HirInstruction::Binary {
                        result: poll_fn_ptr_addr,
                        op: BinaryOp::Add,
                        ty: HirType::I64,
                        left: promise_ptr,
                        right: offset_8,
                    });

                    // Load the poll function pointer from Promise[8]
                    let poll_fn_ty = HirType::Function(Box::new(HirFunctionType {
                        params: vec![HirType::Ptr(Box::new(HirType::U8))],
                        returns: vec![HirType::I64],
                        lifetime_params: vec![],
                        is_variadic: false,
                    }));
                    let poll_fn_ptr =
                        wrapper.create_value(poll_fn_ty.clone(), HirValueKind::Instruction);
                    poll_block.add_instruction(HirInstruction::Load {
                        result: poll_fn_ptr,
                        ty: poll_fn_ty,
                        ptr: poll_fn_ptr_addr,
                        align: 8,
                        volatile: false,
                    });

                    // Call the poll function: poll_result = poll_fn(nested_sm)
                    let poll_result = wrapper.create_value(HirType::I64, HirValueKind::Instruction);
                    poll_block.add_instruction(HirInstruction::IndirectCall {
                        func_ptr: poll_fn_ptr,
                        args: vec![nested_sm],
                        result: Some(poll_result),
                        return_ty: HirType::I64,
                    });

                    // Check if poll_result == 0 (Pending)
                    let zero_const = wrapper
                        .create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
                    let is_pending = wrapper.create_value(HirType::Bool, HirValueKind::Instruction);
                    poll_block.add_instruction(HirInstruction::Binary {
                        result: is_pending,
                        op: BinaryOp::Eq,
                        ty: HirType::I64,
                        left: poll_result,
                        right: zero_const,
                    });

                    // Create blocks for pending and ready branches
                    let pending_block_id = HirId::new();
                    let ready_block_id = HirId::new();

                    // Set terminator to branch based on is_pending
                    poll_block.set_terminator(HirTerminator::CondBranch {
                        condition: is_pending,
                        true_target: pending_block_id, // Pending -> return 0
                        false_target: ready_block_id,  // Ready -> extract value, update state
                    });

                    // Insert poll_block into wrapper
                    wrapper.blocks.insert(poll_block_id, poll_block);

                    // Create pending block - just return 0
                    let mut pending_block = HirBlock::new(pending_block_id);
                    pending_block.label = Some(InternedString::new_global(&format!(
                        "await_pending_{}",
                        state_id.0
                    )));
                    let pending_ret = wrapper
                        .create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
                    pending_block.set_terminator(HirTerminator::Return {
                        values: vec![pending_ret],
                    });
                    wrapper.blocks.insert(pending_block_id, pending_block);

                    // Create ready block - store result and update state
                    let mut ready_block = HirBlock::new(ready_block_id);
                    ready_block.label = Some(InternedString::new_global(&format!(
                        "await_ready_{}",
                        state_id.0
                    )));

                    // Store the poll result in the result slot if present
                    // The result slot should be a capture that we can store into
                    if let Some(result_id) = result {
                        // Find the offset of the result capture in state machine
                        let mut result_offset: i64 = 4; // Start after state field
                        let mut found_offset = false;
                        let mut result_capture_size: i64 = 4; // Default size
                        for capture in &state_machine.captures {
                            // Align to capture size
                            let capture_size = self.type_size(&capture.ty);
                            let align = capture_size;
                            result_offset = (result_offset + align - 1) & !(align - 1);

                            if capture.id == *result_id {
                                found_offset = true;
                                result_capture_size = capture_size;
                                break;
                            }
                            result_offset += capture_size;
                        }

                        if found_offset {
                            // Calculate result slot pointer in state machine
                            let result_offset_const = wrapper.create_value(
                                HirType::I64,
                                HirValueKind::Constant(HirConstant::I64(result_offset)),
                            );
                            let result_slot_ptr = wrapper.create_value(
                                HirType::Ptr(Box::new(HirType::I32)),
                                HirValueKind::Instruction,
                            );
                            ready_block.add_instruction(HirInstruction::Binary {
                                result: result_slot_ptr,
                                op: BinaryOp::Add,
                                ty: HirType::I64,
                                left: self_param,
                                right: result_offset_const,
                            });

                            // Truncate poll_result (i64) to i32 for storage
                            let result_i32 =
                                wrapper.create_value(HirType::I32, HirValueKind::Instruction);
                            ready_block.add_instruction(HirInstruction::Cast {
                                op: CastOp::Trunc,
                                result: result_i32,
                                ty: HirType::I32,
                                operand: poll_result,
                            });

                            // Store the result
                            ready_block.add_instruction(HirInstruction::Store {
                                ptr: result_slot_ptr,
                                value: result_i32,
                                align: 4,
                                volatile: false,
                            });
                        }
                    }

                    // IMPORTANT: Clear the promise pointer so that the next iteration
                    // of the loop will create a new promise instead of reusing the old one
                    let null_ptr = wrapper
                        .create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
                    let clear_future_offset_const = wrapper.create_value(
                        HirType::I64,
                        HirValueKind::Constant(HirConstant::I64(future_offset)),
                    );
                    let clear_future_slot_ptr = wrapper.create_value(
                        HirType::Ptr(Box::new(HirType::Ptr(Box::new(HirType::U8)))),
                        HirValueKind::Instruction,
                    );
                    ready_block.add_instruction(HirInstruction::Binary {
                        result: clear_future_slot_ptr,
                        op: BinaryOp::Add,
                        ty: HirType::I64,
                        left: self_param,
                        right: clear_future_offset_const,
                    });
                    ready_block.add_instruction(HirInstruction::Store {
                        ptr: clear_future_slot_ptr,
                        value: null_ptr,
                        align: 8,
                        volatile: false,
                    });
                    eprintln!(
                        "[DEBUG] Added promise pointer clear at offset {} in ready block",
                        future_offset
                    );

                    // Update state to resume_state
                    let resume_state_const = wrapper.create_value(
                        HirType::U32,
                        HirValueKind::Constant(HirConstant::U32(resume_state.0)),
                    );
                    ready_block.add_instruction(HirInstruction::Store {
                        ptr: self_param,
                        value: resume_state_const,
                        align: 4,
                        volatile: false,
                    });

                    // Return Pending(0) to signal caller to poll again
                    // (On next poll, we'll dispatch to the resume state)
                    let ready_ret = wrapper
                        .create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
                    ready_block.set_terminator(HirTerminator::Return {
                        values: vec![ready_ret],
                    });
                    wrapper.blocks.insert(ready_block_id, ready_block);

                    // Add this state to cases and blocks, then continue to next iteration
                    cases.push((HirConstant::U32(state_id.0), block_id));
                    state_blocks.push((block_id, state_block));
                    continue;
                }
                AsyncTerminator::Continue { next_state } => {
                    // Continue to next state - update state field then return Pending (0)
                    // This is crucial for loops to work: we must transition to next_state

                    // IMPORTANT: Store modified captures back to the state machine.
                    // For SSA-form instructions like `new_i = old_i + 1`, we need to find
                    // the FINAL value of each mutable variable and store it back.
                    // We do this by analyzing the state's instructions to find which
                    // capture values were "updated" (i.e., used as operand in a result-producing instruction).
                    eprintln!(
                        "[DEBUG] State {} Continue: about to store updated captures",
                        state_id.0
                    );
                    self.store_updated_captures_back(
                        state,
                        state_machine,
                        wrapper,
                        &mut state_block,
                        self_param,
                    );
                    eprintln!(
                        "[DEBUG] State {} Continue: done storing updated captures",
                        state_id.0
                    );

                    let next_state_const = wrapper.create_value(
                        HirType::U32,
                        HirValueKind::Constant(HirConstant::U32(next_state.0)),
                    );
                    state_block.add_instruction(HirInstruction::Store {
                        ptr: self_param,
                        value: next_state_const,
                        align: 4,
                        volatile: false,
                    });
                    let pending_const = wrapper
                        .create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
                    HirTerminator::Return {
                        values: vec![pending_const],
                    }
                }
                AsyncTerminator::CondContinue {
                    condition,
                    true_state,
                    false_state,
                } => {
                    // Conditional state transition - create two blocks for true/false paths
                    // Each path updates state and returns Pending

                    let true_block_id = HirId::new();
                    let false_block_id = HirId::new();

                    // Set state_block terminator to branch based on condition
                    state_block.set_terminator(HirTerminator::CondBranch {
                        condition: *condition,
                        true_target: true_block_id,
                        false_target: false_block_id,
                    });

                    // True branch: update state to true_state, return Pending
                    let mut true_block = HirBlock::new(true_block_id);
                    true_block.label = Some(InternedString::new_global(&format!(
                        "cond_true_{}",
                        state_id.0
                    )));
                    let true_state_const = wrapper.create_value(
                        HirType::U32,
                        HirValueKind::Constant(HirConstant::U32(true_state.0)),
                    );
                    true_block.add_instruction(HirInstruction::Store {
                        ptr: self_param,
                        value: true_state_const,
                        align: 4,
                        volatile: false,
                    });
                    let true_pending = wrapper
                        .create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
                    true_block.set_terminator(HirTerminator::Return {
                        values: vec![true_pending],
                    });
                    wrapper.blocks.insert(true_block_id, true_block);

                    // False branch: update state to false_state, return Pending
                    let mut false_block = HirBlock::new(false_block_id);
                    false_block.label = Some(InternedString::new_global(&format!(
                        "cond_false_{}",
                        state_id.0
                    )));
                    let false_state_const = wrapper.create_value(
                        HirType::U32,
                        HirValueKind::Constant(HirConstant::U32(false_state.0)),
                    );
                    false_block.add_instruction(HirInstruction::Store {
                        ptr: self_param,
                        value: false_state_const,
                        align: 4,
                        volatile: false,
                    });
                    let false_pending = wrapper
                        .create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
                    false_block.set_terminator(HirTerminator::Return {
                        values: vec![false_pending],
                    });
                    wrapper.blocks.insert(false_block_id, false_block);

                    // Add to cases and continue - terminator already set above
                    cases.push((HirConstant::U32(state_id.0), block_id));
                    state_blocks.push((block_id, state_block));
                    continue;
                }
                AsyncTerminator::Yield {
                    value,
                    resume_state: _,
                } => {
                    // Yield value and return Poll::Ready(Some(value))
                    HirTerminator::Return {
                        values: vec![*value],
                    }
                }
                AsyncTerminator::Panic { .. } => HirTerminator::Unreachable,
            };
            state_block.set_terminator(terminator);

            cases.push((HirConstant::U32(state_id.0), block_id));
            state_blocks.push((block_id, state_block));
        }

        // Create unreachable default block (for invalid state values)
        let default_block_id = HirId::new();
        let mut default_block = HirBlock::new(default_block_id);
        default_block.label = Some(InternedString::new_global("state_invalid"));
        default_block.set_terminator(HirTerminator::Unreachable);

        // Set up entry block with switch
        let entry_block = wrapper.blocks.get_mut(&wrapper.entry_block).unwrap();
        entry_block.add_instruction(load_state);
        entry_block.set_terminator(HirTerminator::Switch {
            value: state_field,
            default: default_block_id,
            cases,
        });

        // Add all state blocks to the function
        wrapper.blocks.insert(default_block_id, default_block);
        for (block_id, block) in state_blocks {
            wrapper.blocks.insert(block_id, block);
        }

        Ok(())
    }

    /// Generate the state machine struct type
    ///
    /// Creates a struct with:
    /// - state: u32 (current state)
    /// - One field per captured variable
    pub fn generate_state_machine_struct(
        &self,
        state_machine: &AsyncStateMachine,
        arena: &mut zyntax_typed_ast::arena::AstArena,
    ) -> HirType {
        let mut fields = Vec::new();

        // Field 1: state: u32
        fields.push(HirType::U32);

        // Add captured variable fields
        for capture in &state_machine.captures {
            fields.push(capture.ty.clone());
        }

        HirType::Struct(HirStructType {
            name: Some(arena.intern_string("AsyncStateMachine")),
            fields,
            packed: false,
        })
    }

    /// Generate constructor function for the state machine
    ///
    /// Creates a function that:
    /// - Takes a pointer to output buffer as first parameter (sret convention)
    /// - Takes original function parameters as remaining parameters
    /// - Initializes state = 0
    /// - Stores parameters as captures
    /// - Returns void (struct is written to the output pointer)
    ///
    /// This uses explicit sret convention for reliable FFI with Rust.
    /// The struct layout is:
    /// - Offset 0: state (u32, 4 bytes)
    /// - Offset 4: first captured param (4 bytes for i32)
    /// - etc.
    pub fn generate_state_machine_constructor(
        &self,
        state_machine: &AsyncStateMachine,
        _struct_type: HirType,
        original_func: &HirFunction,
        arena: &mut zyntax_typed_ast::arena::AstArena,
    ) -> HirFunction {
        let entry_block_id = HirId::new();
        let mut blocks = IndexMap::new();
        let mut values = IndexMap::new();
        let mut instructions = Vec::new();

        // Parameter 0: output pointer (sret) - generic pointer type
        let out_ptr_id = HirId::new();
        let out_ptr_ty = HirType::Ptr(Box::new(HirType::U8)); // Generic byte pointer
        values.insert(
            out_ptr_id,
            HirValue {
                id: out_ptr_id,
                ty: out_ptr_ty.clone(),
                kind: HirValueKind::Parameter(0),
                uses: HashSet::new(),
                span: None,
            },
        );

        // Create constant for state = 0
        let state_const_id = HirId::new();
        values.insert(
            state_const_id,
            HirValue {
                id: state_const_id,
                ty: HirType::U32,
                kind: HirValueKind::Constant(HirConstant::U32(0)),
                uses: HashSet::new(),
                span: None,
            },
        );

        // Store state = 0 directly at the base pointer (offset 0)
        instructions.push(HirInstruction::Store {
            ptr: out_ptr_id,
            value: state_const_id,
            align: 4,
            volatile: false,
        });

        // Store captured parameters at successive offsets
        // Each field is 4 bytes apart (assuming i32 captures)
        // Offset: 0 = state (u32), 4 = first param, 8 = second param, etc.
        let mut current_offset: i64 = 4; // Start after state field

        for (idx, _capture) in state_machine.captures.iter().enumerate() {
            // Find the corresponding parameter
            if let Some(param) = original_func.signature.params.get(idx) {
                // Create parameter value reference (shifted by 1 for sret)
                let param_value_id = HirId::new();
                values.insert(
                    param_value_id,
                    HirValue {
                        id: param_value_id,
                        ty: param.ty.clone(),
                        kind: HirValueKind::Parameter((idx + 1) as u32), // +1 for sret
                        uses: HashSet::new(),
                        span: None,
                    },
                );

                // Create offset constant
                let offset_const_id = HirId::new();
                values.insert(
                    offset_const_id,
                    HirValue {
                        id: offset_const_id,
                        ty: HirType::I64,
                        kind: HirValueKind::Constant(HirConstant::I64(current_offset)),
                        uses: HashSet::new(),
                        span: None,
                    },
                );

                // Calculate pointer: base + offset using Binary Add (as pointer arithmetic)
                let field_ptr = HirId::new();
                values.insert(
                    field_ptr,
                    HirValue {
                        id: field_ptr,
                        ty: HirType::Ptr(Box::new(param.ty.clone())),
                        kind: HirValueKind::Instruction,
                        uses: HashSet::new(),
                        span: None,
                    },
                );

                instructions.push(HirInstruction::Binary {
                    result: field_ptr,
                    op: BinaryOp::Add,
                    ty: HirType::I64, // Pointer is 64-bit on modern systems
                    left: out_ptr_id,
                    right: offset_const_id,
                });

                // Store parameter value at the calculated address
                instructions.push(HirInstruction::Store {
                    ptr: field_ptr,
                    value: param_value_id,
                    align: 4,
                    volatile: false,
                });

                // Move to next field (4 bytes for i32)
                current_offset += 4;
            }
        }

        // Create entry block with void return
        blocks.insert(
            entry_block_id,
            HirBlock {
                id: entry_block_id,
                label: None,
                phis: Vec::new(),
                instructions,
                terminator: HirTerminator::Return {
                    values: vec![], // Void return
                },
                dominance_frontier: HashSet::new(),
                predecessors: Vec::new(),
                successors: Vec::new(),
            },
        );

        // Create constructor function signature
        // Resolve the original function name first, then create _new suffix
        let base_name = original_func
            .name
            .resolve_global()
            .unwrap_or_else(|| format!("async_fn_{:?}", original_func.id));
        let constructor_name = arena.intern_string(&format!("{}_new", base_name));

        // Build parameter list: first is output pointer (sret), then original params
        let mut params = vec![HirParam {
            id: HirId::new(),
            name: arena.intern_string("__sret"),
            ty: out_ptr_ty,
            attributes: ParamAttributes::default(),
        }];
        params.extend(original_func.signature.params.clone());

        HirFunction {
            id: HirId::new(),
            name: constructor_name,
            signature: HirFunctionSignature {
                params,
                returns: vec![], // Void return - struct is written via sret pointer
                type_params: Vec::new(),
                const_params: Vec::new(),
                lifetime_params: Vec::new(),
                is_variadic: false,
                is_async: false,
                effects: Vec::new(),
                is_pure: false,
            },
            entry_block: entry_block_id,
            blocks,
            locals: IndexMap::new(),
            values,
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::C,
            attributes: FunctionAttributes::default(),
            link_name: None,
        }
    }
}

/// Await point in async function
#[derive(Debug, Clone)]
struct AwaitPoint {
    /// Block containing the await
    block_id: HirId,
    /// Index of await instruction in block
    instruction_index: usize,
    /// Future being awaited
    future: HirId,
    /// Result of the await
    result: Option<HirId>,
}

/// Code segment between await points
#[derive(Debug, Clone)]
struct CodeSegment {
    /// Instructions in this segment
    instructions: Vec<HirInstruction>,
    /// Terminator of this segment
    terminator: HirTerminator,
    /// Resolved target segment indices (for Branch/CondBranch/Switch)
    /// This is populated by split_multi_block_preserving_cfg to preserve loop back-edges.
    /// For Branch: (target_segment_idx, None)
    /// For CondBranch: (true_target_segment_idx, Some(false_target_segment_idx))
    resolved_targets: Option<(usize, Option<usize>)>,
}

// Note: Intrinsics Await and Yield are defined in hir.rs

/// Extend HIR types for async support
impl HirType {
    /// Create a future type with arena support
    pub fn future_with_arena(
        result_type: HirType,
        arena: &mut zyntax_typed_ast::arena::AstArena,
    ) -> Self {
        HirType::Generic {
            base: Box::new(HirType::Opaque(arena.intern_string("Future"))),
            type_args: vec![result_type],
            const_args: vec![],
        }
    }

    /// Create an async function type with arena support
    pub fn async_function_with_arena(
        params: Vec<HirType>,
        result_type: HirType,
        arena: &mut zyntax_typed_ast::arena::AstArena,
    ) -> Self {
        HirType::Function(Box::new(HirFunctionType {
            params,
            returns: vec![Self::future_with_arena(result_type, arena)],
            lifetime_params: vec![],
            is_variadic: false,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::arena::AstArena;

    fn create_test_arena() -> AstArena {
        AstArena::new()
    }

    fn intern_str(arena: &mut AstArena, s: &str) -> InternedString {
        arena.intern_string(s)
    }

    #[test]
    fn test_async_compiler_creation() {
        let compiler = AsyncCompiler::new();
        assert_eq!(compiler.next_state_id, 0);
        assert!(compiler.state_machines.is_empty());
    }

    #[test]
    fn test_async_state_machine_creation() {
        let mut arena = create_test_arena();
        let mut compiler = AsyncCompiler::new();

        // Create a simple async function
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::future_with_arena(HirType::I32, &mut arena)],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: true, // Test is for async function
            effects: vec![],
            is_pure: false,
        };

        let func = HirFunction::new(intern_str(&mut arena, "async_test"), sig);

        let result = compiler.compile_async_function(&func);
        assert!(result.is_ok());

        let state_machine = result.unwrap();
        assert_eq!(state_machine.original_function, func.id);
        assert!(state_machine
            .states
            .contains_key(&state_machine.initial_state));
    }

    #[test]
    fn test_async_types() {
        let mut arena = create_test_arena();
        let future_i32 = HirType::future_with_arena(HirType::I32, &mut arena);
        match future_i32 {
            HirType::Generic {
                base, type_args, ..
            } => match base.as_ref() {
                HirType::Opaque(_) => {
                    assert_eq!(type_args.len(), 1);
                    assert_eq!(type_args[0], HirType::I32);
                }
                _ => panic!("Expected opaque base type"),
            },
            _ => panic!("Expected generic future type"),
        }

        let async_fn = HirType::async_function_with_arena(
            vec![HirType::I32],
            HirType::Ptr(Box::new(HirType::I8)),
            &mut arena,
        );
        match async_fn {
            HirType::Function(func_ty) => {
                assert_eq!(func_ty.params.len(), 1);
                assert_eq!(func_ty.returns.len(), 1);
                assert!(matches!(func_ty.returns[0], HirType::Generic { .. }));
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_async_capture_modes() {
        let by_value = AsyncCaptureMode::ByValue;
        let by_ref = AsyncCaptureMode::ByRef(HirLifetime::anonymous());
        let by_mut_ref = AsyncCaptureMode::ByMutRef(HirLifetime::static_lifetime());

        assert_eq!(by_value, AsyncCaptureMode::ByValue);
        assert!(matches!(by_ref, AsyncCaptureMode::ByRef(_)));
        assert!(matches!(by_mut_ref, AsyncCaptureMode::ByMutRef(_)));
    }

    #[test]
    fn test_async_runtime_types() {
        let mut arena = create_test_arena();
        let tokio = AsyncRuntimeType::Tokio;
        let custom = AsyncRuntimeType::Custom(intern_str(&mut arena, "MyRuntime"));
        let none = AsyncRuntimeType::None;

        assert_eq!(tokio, AsyncRuntimeType::Tokio);
        assert!(matches!(custom, AsyncRuntimeType::Custom(_)));
        assert_eq!(none, AsyncRuntimeType::None);
    }

    #[test]
    fn test_state_machine_struct_generation() {
        let mut arena = create_test_arena();
        let compiler = AsyncCompiler::new();

        // Create a state machine with captures
        let capture = AsyncCapture {
            id: HirId::new(),
            name: intern_str(&mut arena, "x"),
            ty: HirType::I32,
            mode: AsyncCaptureMode::ByValue,
        };

        let state_machine = AsyncStateMachine {
            id: HirId::new(),
            original_function: HirId::new(),
            original_name: intern_str(&mut arena, "test_async"),
            states: IndexMap::new(),
            initial_state: AsyncStateId(0),
            final_state: AsyncStateId(1),
            captures: vec![capture],
            result_type: HirType::I32,
            values: IndexMap::new(),
        };

        let struct_type = compiler.generate_state_machine_struct(&state_machine, &mut arena);

        // Verify it's a struct type
        match struct_type {
            HirType::Struct(struct_ty) => {
                // Should have state + captured variables
                assert_eq!(struct_ty.fields.len(), 2); // state + x
                assert_eq!(struct_ty.fields[0], HirType::U32); // state field
                assert_eq!(struct_ty.fields[1], HirType::I32); // x field
                assert!(!struct_ty.packed);
            }
            _ => panic!("Expected struct type"),
        }
    }

    #[test]
    fn test_state_machine_constructor_generation() {
        let mut arena = create_test_arena();
        let compiler = AsyncCompiler::new();

        // Create a simple async function
        let func_id = HirId::new();
        let param = HirParam {
            id: HirId::new(),
            name: intern_str(&mut arena, "x"),
            ty: HirType::I32,
            attributes: ParamAttributes::default(),
        };

        let sig = HirFunctionSignature {
            params: vec![param],
            returns: vec![HirType::I32],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: true,
            effects: vec![],
            is_pure: false,
        };

        let func = HirFunction {
            id: func_id,
            name: intern_str(&mut arena, "test_async"),
            signature: sig,
            entry_block: HirId::new(),
            blocks: IndexMap::new(),
            locals: IndexMap::new(),
            values: IndexMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::C,
            attributes: FunctionAttributes::default(),
            link_name: None,
        };

        // Create state machine
        let capture = AsyncCapture {
            id: HirId::new(),
            name: intern_str(&mut arena, "x"),
            ty: HirType::I32,
            mode: AsyncCaptureMode::ByValue,
        };

        let state_machine = AsyncStateMachine {
            id: HirId::new(),
            original_function: func_id,
            original_name: intern_str(&mut arena, "test_async"),
            states: IndexMap::new(),
            initial_state: AsyncStateId(0),
            final_state: AsyncStateId(1),
            captures: vec![capture],
            result_type: HirType::I32,
            values: IndexMap::new(),
        };

        let struct_type = compiler.generate_state_machine_struct(&state_machine, &mut arena);
        let constructor = compiler.generate_state_machine_constructor(
            &state_machine,
            struct_type.clone(),
            &func,
            &mut arena,
        );

        // Verify constructor signature
        // Constructor has: sret pointer (output) + original params
        assert_eq!(constructor.signature.params.len(), 2); // sret + original x param
        assert_eq!(constructor.signature.returns.len(), 0); // Uses sret, so void return
        assert!(!constructor.signature.is_async); // Constructor is not async

        // Verify it has an entry block
        assert!(constructor.blocks.contains_key(&constructor.entry_block));

        // Verify entry block has return terminator
        let entry_block = &constructor.blocks[&constructor.entry_block];
        assert!(matches!(
            entry_block.terminator,
            HirTerminator::Return { .. }
        ));

        // Verify return values (sret convention - void return)
        match &entry_block.terminator {
            HirTerminator::Return { values } => {
                assert_eq!(values.len(), 0); // Void return - struct written via sret pointer
            }
            _ => panic!("Expected return terminator"),
        }
    }

    #[test]
    fn test_state_machine_struct_without_captures() {
        let mut arena = create_test_arena();
        let compiler = AsyncCompiler::new();

        // Create a state machine without captures
        let state_machine = AsyncStateMachine {
            id: HirId::new(),
            original_function: HirId::new(),
            original_name: intern_str(&mut arena, "empty_async"),
            states: IndexMap::new(),
            initial_state: AsyncStateId(0),
            final_state: AsyncStateId(1),
            captures: vec![], // No captures
            result_type: HirType::Void,
            values: IndexMap::new(),
        };

        let struct_type = compiler.generate_state_machine_struct(&state_machine, &mut arena);

        // Should only have state field
        match struct_type {
            HirType::Struct(struct_ty) => {
                assert_eq!(struct_ty.fields.len(), 1); // Only state field
                assert_eq!(struct_ty.fields[0], HirType::U32);
            }
            _ => panic!("Expected struct type"),
        }
    }

    #[test]
    fn test_constructor_with_multiple_parameters() {
        let mut arena = create_test_arena();
        let compiler = AsyncCompiler::new();

        // Create async function with 3 parameters
        let func_id = HirId::new();
        let params = vec![
            HirParam {
                id: HirId::new(),
                name: intern_str(&mut arena, "x"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: intern_str(&mut arena, "y"),
                ty: HirType::I64,
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: intern_str(&mut arena, "msg"),
                ty: HirType::Ptr(Box::new(HirType::I8)),
                attributes: ParamAttributes::default(),
            },
        ];

        let sig = HirFunctionSignature {
            params: params.clone(),
            returns: vec![HirType::Void],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: true,
            effects: vec![],
            is_pure: false,
        };

        let func = HirFunction {
            id: func_id,
            name: intern_str(&mut arena, "multi_param_async"),
            signature: sig,
            entry_block: HirId::new(),
            blocks: IndexMap::new(),
            locals: IndexMap::new(),
            values: IndexMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::C,
            attributes: FunctionAttributes::default(),
            link_name: None,
        };

        // Create captures for all 3 parameters
        let captures = vec![
            AsyncCapture {
                id: HirId::new(),
                name: intern_str(&mut arena, "x"),
                ty: HirType::I32,
                mode: AsyncCaptureMode::ByValue,
            },
            AsyncCapture {
                id: HirId::new(),
                name: intern_str(&mut arena, "y"),
                ty: HirType::I64,
                mode: AsyncCaptureMode::ByValue,
            },
            AsyncCapture {
                id: HirId::new(),
                name: intern_str(&mut arena, "msg"),
                ty: HirType::Ptr(Box::new(HirType::I8)),
                mode: AsyncCaptureMode::ByValue,
            },
        ];

        let state_machine = AsyncStateMachine {
            id: HirId::new(),
            original_function: func_id,
            original_name: intern_str(&mut arena, "multi_param_async"),
            states: IndexMap::new(),
            initial_state: AsyncStateId(0),
            final_state: AsyncStateId(1),
            captures,
            result_type: HirType::Void,
            values: IndexMap::new(),
        };

        let struct_type = compiler.generate_state_machine_struct(&state_machine, &mut arena);
        let constructor = compiler.generate_state_machine_constructor(
            &state_machine,
            struct_type.clone(),
            &func,
            &mut arena,
        );

        // Verify struct has 4 fields (state + 3 params)
        match &struct_type {
            HirType::Struct(struct_ty) => {
                assert_eq!(struct_ty.fields.len(), 4); // state + x + y + msg
                assert_eq!(struct_ty.fields[0], HirType::U32); // state
                assert_eq!(struct_ty.fields[1], HirType::I32); // x
                assert_eq!(struct_ty.fields[2], HirType::I64); // y
                assert_eq!(struct_ty.fields[3], HirType::Ptr(Box::new(HirType::I8)));
                // msg
            }
            _ => panic!("Expected struct type"),
        }

        // Verify constructor uses sret convention with Store instructions
        let entry_block = &constructor.blocks[&constructor.entry_block];

        // Should have Store instructions for state + 3 params = 4 stores
        // Plus Binary/GEP instructions for computing field pointers
        let store_count = entry_block
            .instructions
            .iter()
            .filter(|inst| matches!(inst, HirInstruction::Store { .. }))
            .count();
        assert_eq!(store_count, 4); // state + x + y + msg

        // Verify parameters are properly referenced
        // sret pointer + 3 original params = 4 parameter values
        let param_values: Vec<_> = constructor
            .values
            .values()
            .filter(|v| matches!(v.kind, HirValueKind::Parameter(_)))
            .collect();
        assert_eq!(param_values.len(), 4); // sret + 3 original params
    }
}
