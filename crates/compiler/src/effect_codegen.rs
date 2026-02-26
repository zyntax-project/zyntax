//! # Effect Code Generation
//!
//! This module provides code generation support for algebraic effects.
//!
//! ## Overview
//!
//! Algebraic effects require sophisticated code generation to support:
//! - Effect operation dispatch to the appropriate handler
//! - Continuation capture and resumption
//! - Handler scope management
//! - Efficient compilation of common patterns
//!
//! ## Implementation Tiers
//!
//! ### Tier 1: Simple Effects (Current Implementation)
//!
//! For effects without continuation capture (is_resumable = false):
//! - PerformEffect becomes a direct call to the handler implementation
//! - HandleEffect sets up handler state and compiles the body inline
//! - Resume and AbortEffect are not needed (handler returns directly)
//!
//! This covers most ML use cases:
//! ```text
//! effect Logging {
//!     def log(msg: str)      // Simple side effect, no continuation
//! }
//!
//! @handler(Logging)
//! struct ConsoleLogger {
//!     @handles(log)
//!     def log_impl(self, msg: str) {
//!         print(msg)          // No resume, returns directly
//!     }
//! }
//! ```
//!
//! ### Tier 2: State Effects (Future)
//!
//! For stateful handlers with linear state access:
//! - Handler state allocated on stack or heap
//! - State passed to operation implementations
//! - State updated atomically between operations
//!
//! ```text
//! effect State<S> {
//!     def get(): S
//!     def put(s: S)
//! }
//!
//! @handler(State<int>)
//! struct IntState {
//!     value: int,
//!
//!     @handles(get)
//!     def get_impl(self): int {
//!         self.value
//!     }
//!
//!     @handles(put)
//!     def put_impl(mut self, v: int) {
//!         self.value = v
//!     }
//! }
//! ```
//!
//! ### Tier 3: Full Delimited Continuations (Future)
//!
//! For effects that need to capture and resume continuations:
//! - Continuation represented as closure capturing stack state
//! - Trampoline-based execution model
//! - CPS (Continuation-Passing Style) transformation
//!
//! ```text
//! effect Async {
//!     def await<T>(promise: Promise<T>): T
//! }
//!
//! @handler(Async)
//! struct AsyncHandler {
//!     @handles(await)
//!     def await_impl<T>(self, promise: Promise<T>, resume: Resume<T>): T {
//!         // Can store resume and call it later
//!         promise.then(|result| resume(result))
//!     }
//! }
//! ```
//!
//! ## Code Generation Strategies
//!
//! ### Direct Call (Most Efficient)
//!
//! When handler is statically known and simple:
//! ```text
//! // Source:
//! let x = perform(State.get())
//!
//! // Generated (direct call):
//! let x = IntState$effect$get(handler_state)
//! ```
//!
//! ### Inlined Handler
//!
//! When handler body is small and non-recursive:
//! ```text
//! // Source:
//! let x = perform(State.get())
//!
//! // Generated (inlined):
//! let x = handler_state.value  // Directly inlined
//! ```
//!
//! ### Runtime Dispatch
//!
//! When handler is not statically known:
//! ```text
//! // Source:
//! let x = perform(State.get())
//!
//! // Generated (runtime dispatch):
//! let x = __zyntax_effect_perform(
//!     "State",           // effect name
//!     "get",             // operation name
//!     [],                // arguments
//!     __handler_stack    // runtime handler stack
//! )
//! ```
//!
//! ## Continuation Implementation (Tier 3 Design)
//!
//! ### CPS Transformation
//!
//! For resumable handlers, we transform to CPS:
//! ```text
//! // Original:
//! def compute() {
//!     let x = perform(State.get())
//!     let y = x + 1
//!     perform(State.put(y))
//!     y * 2
//! }
//!
//! // CPS-transformed:
//! def compute(k: Continuation<int>) {
//!     State$get(handler, |x| {
//!         let y = x + 1
//!         State$put(handler, y, |_| {
//!             k(y * 2)
//!         })
//!     })
//! }
//! ```
//!
//! ### Trampoline Execution
//!
//! To avoid stack overflow with deep continuations:
//! ```text
//! enum Bounce<T> {
//!     Done(T),
//!     Continue(Box<dyn FnOnce() -> Bounce<T>>)
//! }
//!
//! fn run<T>(mut bounce: Bounce<T>) -> T {
//!     loop {
//!         match bounce {
//!             Bounce::Done(v) => return v,
//!             Bounce::Continue(f) => bounce = f(),
//!         }
//!     }
//! }
//! ```
//!
//! ### Continuation Capture
//!
//! For CaptureContinuation instruction:
//! 1. Save current stack frame to heap-allocated closure
//! 2. Create continuation object with resume function
//! 3. Pass continuation to handler
//!
//! ```text
//! // Source:
//! @handles(await)
//! def await_impl<T>(self, promise: Promise<T>, resume: Resume<T>): T {
//!     if promise.is_ready() {
//!         resume(promise.get())  // Resume immediately
//!     } else {
//!         promise.on_ready(|v| resume(v))  // Resume later
//!         // Abort current computation
//!     }
//! }
//!
//! // Generated:
//! fn await_impl(handler: *Handler, promise: *Promise, k: *Continuation) {
//!     if promise_is_ready(promise) {
//!         continuation_resume(k, promise_get(promise))
//!     } else {
//!         // Store continuation for later invocation
//!         promise_on_ready(promise, k)
//!         // Return to trampoline
//!         return Bounce::Suspend
//!     }
//! }
//! ```
//!
//! ## Memory Layout
//!
//! ### Handler State
//!
//! ```text
//! struct HandlerState {
//!     vtable: *HandlerVTable,     // Dispatch table for operations
//!     effect_id: u64,             // Effect identifier
//!     state_size: u32,            // Size of state data
//!     state_data: [u8; N],        // Captured state fields
//! }
//! ```
//!
//! ### Continuation
//!
//! ```text
//! struct Continuation {
//!     resume_fn: fn(*void, *void) -> Bounce,  // Resume function
//!     frame_data: *void,                       // Captured stack frame
//!     frame_size: u32,                         // Size of frame
//!     handler_chain: *HandlerStack,            // Handler stack at capture
//! }
//! ```
//!
//! ## Cranelift Integration
//!
//! This module provides helper functions that the Cranelift backend calls:
//!
//! 1. `analyze_perform_effect()` - Determine codegen strategy for PerformEffect
//! 2. `analyze_handle_effect()` - Prepare HandleEffect codegen info
//! 3. `mangle_handler_op_name()` - Generate function names for handler ops
//! 4. `get_handler_ops_info()` - Get operation signatures for handler
//!
//! The Cranelift backend then uses this information to:
//! - Generate direct calls for simple effects
//! - Inline handler code when profitable
//! - Generate runtime dispatch for dynamic handlers
//! - (Future) Generate CPS-transformed code for resumable handlers

use crate::effect_handler_resolution::{
    FunctionHandlerResolution, HandlerOptimization, HandlerResolution, ModuleHandlerResolution,
    ResolvedHandler,
};
use crate::hir::{
    HirBlock, HirEffect, HirEffectHandler, HirEffectHandlerImpl, HirFunction, HirId,
    HirInstruction, HirModule, HirType,
};
use crate::CompilerError;
use crate::CompilerResult;
use indexmap::IndexMap;
use std::collections::HashMap;
use zyntax_typed_ast::InternedString;

/// Effect codegen context
///
/// Holds information needed during code generation for effects.
#[derive(Debug)]
pub struct EffectCodegenContext {
    /// Handler resolution results
    pub handler_resolution: Option<ModuleHandlerResolution>,
    /// Current handler stack (for nested handlers)
    pub handler_stack: Vec<HandlerStackEntry>,
    /// Handler state variables (for stateful handlers)
    pub handler_states: HashMap<HirId, HandlerStateInfo>,
}

/// Entry in the handler stack
#[derive(Debug, Clone)]
pub struct HandlerStackEntry {
    /// Handler ID
    pub handler_id: HirId,
    /// Effect being handled
    pub effect_id: HirId,
    /// Effect name
    pub effect_name: InternedString,
    /// Variable holding handler state (if any)
    pub state_var: Option<StateVarInfo>,
    /// Continuation block (where to resume after handler)
    pub continuation_block: HirId,
}

/// Information about a state variable
#[derive(Debug, Clone)]
pub struct StateVarInfo {
    /// Variable ID for the state
    pub var_id: HirId,
    /// Type of the state
    pub state_type: HirType,
}

/// Handler state information
#[derive(Debug, Clone)]
pub struct HandlerStateInfo {
    /// Fields in the handler state
    pub fields: Vec<(InternedString, HirType, HirId)>,
    /// Allocated stack slot for state (in Cranelift)
    pub stack_slot: Option<u32>,
}

impl EffectCodegenContext {
    pub fn new() -> Self {
        Self {
            handler_resolution: None,
            handler_stack: Vec::new(),
            handler_states: HashMap::new(),
        }
    }

    pub fn with_resolution(mut self, resolution: ModuleHandlerResolution) -> Self {
        self.handler_resolution = Some(resolution);
        self
    }

    /// Push a handler onto the stack when entering a HandleEffect scope
    pub fn push_handler(&mut self, entry: HandlerStackEntry) {
        self.handler_stack.push(entry);
    }

    /// Pop a handler when leaving a HandleEffect scope
    pub fn pop_handler(&mut self) -> Option<HandlerStackEntry> {
        self.handler_stack.pop()
    }

    /// Find the handler for an effect
    pub fn find_handler(&self, effect_id: HirId) -> Option<&HandlerStackEntry> {
        // Search from innermost (last) to outermost (first)
        self.handler_stack
            .iter()
            .rev()
            .find(|h| h.effect_id == effect_id)
    }
}

impl Default for EffectCodegenContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Code generation info for a PerformEffect instruction
#[derive(Debug, Clone)]
pub struct PerformEffectCodegen {
    /// How to generate the code
    pub strategy: PerformStrategy,
    /// Handler to call (if statically known)
    pub handler: Option<ResolvedHandler>,
    /// Effect operation index
    pub op_index: Option<usize>,
}

/// Strategy for generating PerformEffect code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformStrategy {
    /// Direct call to handler implementation (most efficient)
    DirectCall,
    /// Inline the handler code
    Inline,
    /// Use runtime dispatch (dynamic handler lookup)
    RuntimeDispatch,
    /// Not handled - will trap
    Unhandled,
}

/// Analyze a PerformEffect instruction and determine codegen strategy
pub fn analyze_perform_effect(
    effect_id: HirId,
    op_name: InternedString,
    ctx: &EffectCodegenContext,
    function_id: HirId,
    block_id: HirId,
    inst_index: usize,
) -> PerformEffectCodegen {
    // First, check if we have handler resolution results
    if let Some(resolution) = &ctx.handler_resolution {
        if let Some(func_resolution) = resolution.functions.get(&function_id) {
            // Find the resolution for this specific perform site
            for res in &func_resolution.resolutions {
                if res.perform_site.function_id == function_id
                    && res.perform_site.block_id == block_id
                    && res.perform_site.instruction_index == inst_index
                {
                    if let Some(handler) = &res.resolved_handler {
                        let strategy = match handler.optimization {
                            HandlerOptimization::Inline | HandlerOptimization::SimpleReturn => {
                                PerformStrategy::Inline
                            }
                            HandlerOptimization::StaticDispatch => PerformStrategy::DirectCall,
                            HandlerOptimization::Pure => PerformStrategy::DirectCall,
                            HandlerOptimization::Dynamic => PerformStrategy::RuntimeDispatch,
                        };

                        return PerformEffectCodegen {
                            strategy,
                            handler: Some(handler.clone()),
                            op_index: Some(handler.impl_index),
                        };
                    }
                }
            }
        }
    }

    // Fall back to runtime handler stack
    if let Some(handler_entry) = ctx.find_handler(effect_id) {
        return PerformEffectCodegen {
            strategy: PerformStrategy::RuntimeDispatch,
            handler: None,
            op_index: None,
        };
    }

    // No handler found - effect is unhandled
    PerformEffectCodegen {
        strategy: PerformStrategy::Unhandled,
        handler: None,
        op_index: None,
    }
}

/// Code generation info for HandleEffect instruction
#[derive(Debug, Clone)]
pub struct HandleEffectCodegen {
    /// Handler being installed
    pub handler_id: HirId,
    /// Effect being handled
    pub effect_id: HirId,
    /// Whether handler needs state allocation
    pub needs_state: bool,
    /// Size of handler state (in bytes)
    pub state_size: usize,
    /// Body block to compile
    pub body_block: HirId,
    /// Continuation block after handler
    pub continuation_block: HirId,
}

/// Analyze a HandleEffect instruction
pub fn analyze_handle_effect(
    handler_id: HirId,
    handler_state: &[HirId],
    body_block: HirId,
    continuation_block: HirId,
    module: &HirModule,
) -> Option<HandleEffectCodegen> {
    let handler = module.handlers.get(&handler_id)?;

    let needs_state = !handler.state_fields.is_empty();
    let state_size = if needs_state {
        // Estimate state size (8 bytes per field for simplicity)
        handler.state_fields.len() * 8
    } else {
        0
    };

    Some(HandleEffectCodegen {
        handler_id,
        effect_id: handler.effect_id,
        needs_state,
        state_size,
        body_block,
        continuation_block,
    })
}

/// Generate mangled name for a handler operation
pub fn mangle_handler_op_name(handler_name: InternedString, op_name: InternedString) -> String {
    let handler_str = handler_name.resolve_global().unwrap_or_default();
    let op_str = op_name.resolve_global().unwrap_or_default();
    format!("{}$effect${}", handler_str, op_str)
}

/// Generate mangled name for handler state type
pub fn mangle_handler_state_name(handler_name: InternedString) -> String {
    let handler_str = handler_name.resolve_global().unwrap_or_default();
    format!("{}$state", handler_str)
}

/// Information about a handler operation for codegen
#[derive(Debug, Clone)]
pub struct HandlerOpInfo {
    /// Mangled function name
    pub function_name: String,
    /// Parameter types
    pub param_types: Vec<HirType>,
    /// Return type
    pub return_type: HirType,
    /// Whether this operation uses continuation
    pub uses_continuation: bool,
}

/// Get codegen info for all operations of a handler
pub fn get_handler_ops_info(handler: &HirEffectHandler) -> Vec<HandlerOpInfo> {
    handler
        .implementations
        .iter()
        .map(|impl_| HandlerOpInfo {
            function_name: mangle_handler_op_name(handler.name, impl_.op_name),
            param_types: impl_.params.iter().map(|p| p.ty.clone()).collect(),
            return_type: impl_.return_type.clone(),
            uses_continuation: impl_.is_resumable,
        })
        .collect()
}

/// Runtime effect operations (for dynamic dispatch)
///
/// These would be implemented as runtime functions that the generated code calls.
pub mod runtime {
    /// Push a handler onto the runtime handler stack
    pub const EFFECT_PUSH_HANDLER: &str = "__zyntax_effect_push_handler";
    /// Pop a handler from the runtime handler stack
    pub const EFFECT_POP_HANDLER: &str = "__zyntax_effect_pop_handler";
    /// Perform an effect operation (runtime dispatch)
    pub const EFFECT_PERFORM: &str = "__zyntax_effect_perform";
    /// Resume a continuation
    pub const EFFECT_RESUME: &str = "__zyntax_effect_resume";
    /// Abort with a value
    pub const EFFECT_ABORT: &str = "__zyntax_effect_abort";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mangle_handler_op_name() {
        let handler = InternedString::new_global("StateHandler");
        let op = InternedString::new_global("get");
        let mangled = mangle_handler_op_name(handler, op);
        assert_eq!(mangled, "StateHandler$effect$get");
    }

    #[test]
    fn test_empty_context() {
        let ctx = EffectCodegenContext::new();
        assert!(ctx.handler_stack.is_empty());
        assert!(ctx.handler_resolution.is_none());
    }

    #[test]
    fn test_handler_stack() {
        let mut ctx = EffectCodegenContext::new();

        let entry = HandlerStackEntry {
            handler_id: HirId::new(),
            effect_id: HirId::new(),
            effect_name: InternedString::new_global("State"),
            state_var: None,
            continuation_block: HirId::new(),
        };

        ctx.push_handler(entry.clone());
        assert_eq!(ctx.handler_stack.len(), 1);

        let popped = ctx.pop_handler();
        assert!(popped.is_some());
        assert!(ctx.handler_stack.is_empty());
    }
}
