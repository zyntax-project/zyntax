//! # High-level Intermediate Representation (HIR)
//!
//! Platform-agnostic IR that can be lowered to both Cranelift IR and LLVM IR.
//! This representation maintains high-level type information while being close
//! enough to machine semantics for efficient code generation.
//!
//! ## Design Goals
//!
//! - Compatible with both Cranelift and LLVM type systems
//! - Supports SSA form with explicit phi nodes
//! - Preserves type information for optimization
//! - Enables hot-reloading via function versioning
//! - Memory safe with explicit lifetime tracking

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use uuid::Uuid;
use zyntax_typed_ast::{InternedString, Span, Type, TypeId};

// ============================================================================
// Algebraic Effects - HIR Types
// ============================================================================

/// Effect declaration in HIR
///
/// Represents an algebraic effect with its operations.
/// Effects are compiled to handler dispatch mechanisms.
///
/// Example:
/// ```ignore
/// effect State<S> {
///     def get(): S
///     def put(s: S)
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirEffect {
    pub id: HirId,
    pub name: InternedString,
    pub type_params: Vec<HirTypeParam>,
    pub operations: Vec<HirEffectOp>,
}

/// Effect operation signature
///
/// Each operation can be performed within the effect's scope
/// and will be handled by the active handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirEffectOp {
    pub id: HirId,
    pub name: InternedString,
    pub type_params: Vec<HirTypeParam>,
    pub params: Vec<HirParam>,
    pub return_type: HirType,
}

/// Effect handler definition in HIR
///
/// A handler provides implementations for all operations of an effect.
/// Handlers can be stateful (capturing variables from the enclosing scope).
///
/// Example:
/// ```ignore
/// handler StateHandler<S> for State<S> {
///     def get(): S { ... }
///     def put(s: S) { ... }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirEffectHandler {
    pub id: HirId,
    pub name: InternedString,
    pub effect_id: HirId,
    pub type_params: Vec<HirTypeParam>,
    /// Handler state fields (captured from enclosing scope)
    pub state_fields: Vec<HirHandlerField>,
    /// Operation implementations
    pub implementations: Vec<HirEffectHandlerImpl>,
}

/// Field in a handler's state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirHandlerField {
    pub name: InternedString,
    pub ty: HirType,
}

/// Implementation of a single effect operation in a handler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirEffectHandlerImpl {
    pub op_name: InternedString,
    pub type_params: Vec<HirTypeParam>,
    pub params: Vec<HirParam>,
    pub return_type: HirType,
    /// Function body (basic blocks)
    pub entry_block: HirId,
    pub blocks: IndexMap<HirId, HirBlock>,
    /// Whether this handler uses the continuation (resumable)
    pub is_resumable: bool,
}

/// Lifetime identifier for memory safety tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LifetimeId(Uuid);

impl LifetimeId {
    pub fn new() -> Self {
        LifetimeId(Uuid::new_v4())
    }

    /// Static lifetime (lives for the entire program)
    pub fn static_lifetime() -> Self {
        LifetimeId(Uuid::from_bytes([0; 16]))
    }

    /// Anonymous lifetime (inferred)
    pub fn anonymous() -> Self {
        LifetimeId(Uuid::from_bytes([1; 16]))
    }
}

/// Lifetime parameter in HIR
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirLifetime {
    pub id: LifetimeId,
    pub name: Option<InternedString>,
    pub bounds: Vec<LifetimeBound>,
}

/// Lifetime bounds and constraints
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LifetimeBound {
    /// 'a: 'b (lifetime 'a outlives lifetime 'b)
    Outlives(LifetimeId),
    /// Static lifetime bound
    Static,
}

impl HirLifetime {
    pub fn new(name: Option<InternedString>) -> Self {
        Self {
            id: LifetimeId::new(),
            name,
            bounds: Vec::new(),
        }
    }

    pub fn static_lifetime() -> Self {
        // For now, create a placeholder static lifetime name
        // TODO: Use proper string interning from arena
        Self {
            id: LifetimeId::static_lifetime(),
            name: None, // Use None for static lifetime until we have proper arena access
            bounds: vec![LifetimeBound::Static],
        }
    }

    pub fn anonymous() -> Self {
        Self {
            id: LifetimeId::anonymous(),
            name: None,
            bounds: Vec::new(),
        }
    }
}

/// Unique identifier for HIR entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirId(Uuid);

impl HirId {
    pub fn new() -> Self {
        HirId(Uuid::new_v4())
    }
}

/// HIR module representing a compilation unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirModule {
    pub id: HirId,
    pub name: InternedString,
    pub functions: IndexMap<HirId, HirFunction>,
    pub globals: IndexMap<HirId, HirGlobal>,
    pub types: IndexMap<TypeId, HirType>,
    pub imports: Vec<HirImport>,
    pub exports: Vec<HirExport>,
    /// Metadata for hot-reloading support
    pub version: u64,
    pub dependencies: HashSet<HirId>,
    /// Algebraic effect declarations
    pub effects: IndexMap<HirId, HirEffect>,
    /// Effect handler definitions
    pub handlers: IndexMap<HirId, HirEffectHandler>,
}

/// HIR function with CFG and SSA form
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirFunction {
    pub id: HirId,
    pub name: InternedString,
    pub signature: HirFunctionSignature,
    pub entry_block: HirId,
    pub blocks: IndexMap<HirId, HirBlock>,
    pub locals: IndexMap<HirId, HirLocal>,
    /// SSA values defined in this function
    pub values: IndexMap<HirId, HirValue>,
    /// For hot-reloading: previous version of this function
    pub previous_version: Option<HirId>,
    pub is_external: bool,
    pub calling_convention: CallingConvention,
    pub attributes: FunctionAttributes,
    /// Override symbol name for linking (e.g., "$haxe$trace$int" for extern trace)
    pub link_name: Option<String>,
}

/// Function signature compatible with both backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirFunctionSignature {
    pub params: Vec<HirParam>,
    pub returns: Vec<HirType>,
    pub type_params: Vec<HirTypeParam>,
    pub const_params: Vec<HirConstParam>,
    pub lifetime_params: Vec<HirLifetime>,
    pub is_variadic: bool,
    pub is_async: bool,
    /// Effects this function may perform (algebraic effects)
    pub effects: Vec<InternedString>,
    /// Whether this function is pure (no effects, no side effects)
    pub is_pure: bool,
}

/// Method signature for trait/interface methods
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirMethodSignature {
    pub name: InternedString,
    pub params: Vec<HirType>, // Parameter types only (no names/IDs)
    pub return_type: HirType,
    pub is_static: bool,
    pub is_async: bool,
}

/// Function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirParam {
    pub id: HirId,
    pub name: InternedString,
    pub ty: HirType,
    pub attributes: ParamAttributes,
}

/// Parameter attributes for ABI compatibility
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParamAttributes {
    pub by_ref: bool,
    pub sret: bool, // Structure return
    pub zext: bool, // Zero extend
    pub sext: bool, // Sign extend
    pub noalias: bool,
    pub nonnull: bool,
    pub readonly: bool,
}

/// Basic block in SSA form
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirBlock {
    pub id: HirId,
    pub label: Option<InternedString>,
    /// Phi nodes must come first
    pub phis: Vec<HirPhi>,
    /// Regular instructions
    pub instructions: Vec<HirInstruction>,
    /// Terminator instruction (must be last)
    pub terminator: HirTerminator,
    /// Dominance frontier for SSA construction
    pub dominance_frontier: HashSet<HirId>,
    pub predecessors: Vec<HirId>,
    pub successors: Vec<HirId>,
}

/// SSA phi node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirPhi {
    pub result: HirId,
    pub ty: HirType,
    pub incoming: Vec<(HirId, HirId)>, // (value, block)
}

/// HIR instruction - compatible with both backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HirInstruction {
    /// Arithmetic operations
    Binary {
        op: BinaryOp,
        result: HirId,
        ty: HirType,
        left: HirId,
        right: HirId,
    },

    /// Unary operations
    Unary {
        op: UnaryOp,
        result: HirId,
        ty: HirType,
        operand: HirId,
    },

    /// Memory allocation
    Alloca {
        result: HirId,
        ty: HirType,
        count: Option<HirId>,
        align: u32,
    },

    /// Memory load
    Load {
        result: HirId,
        ty: HirType,
        ptr: HirId,
        align: u32,
        volatile: bool,
    },

    /// Memory store
    Store {
        value: HirId,
        ptr: HirId,
        align: u32,
        volatile: bool,
    },

    /// Get element pointer (GEP)
    GetElementPtr {
        result: HirId,
        ty: HirType,
        ptr: HirId,
        indices: Vec<HirId>,
    },

    /// Function call
    Call {
        result: Option<HirId>,
        callee: HirCallable,
        args: Vec<HirId>,
        /// Type arguments for generic function calls (e.g., identity<i32>(42))
        type_args: Vec<HirType>,
        /// Const arguments for const generic calls (e.g., array<i32, 5>())
        const_args: Vec<HirConstant>,
        is_tail: bool,
    },

    /// Indirect call through function pointer (for trait dispatch)
    IndirectCall {
        result: Option<HirId>,
        func_ptr: HirId,
        args: Vec<HirId>,
        return_ty: HirType,
    },

    /// Type cast
    Cast {
        op: CastOp,
        result: HirId,
        ty: HirType,
        operand: HirId,
    },

    /// Select (ternary conditional)
    Select {
        result: HirId,
        ty: HirType,
        condition: HirId,
        true_val: HirId,
        false_val: HirId,
    },

    /// Extract value from aggregate
    ExtractValue {
        result: HirId,
        ty: HirType,
        aggregate: HirId,
        indices: Vec<u32>,
    },

    /// Insert value into aggregate
    InsertValue {
        result: HirId,
        ty: HirType,
        aggregate: HirId,
        value: HirId,
        indices: Vec<u32>,
    },

    /// Atomic operations
    Atomic {
        op: AtomicOp,
        result: HirId,
        ty: HirType,
        ptr: HirId,
        value: Option<HirId>,
        ordering: AtomicOrdering,
    },

    /// Memory fence
    Fence { ordering: AtomicOrdering },

    /// Create a union value with specified variant
    CreateUnion {
        result: HirId,
        union_ty: HirType,
        variant_index: u32,
        value: HirId,
    },

    /// Get discriminant value from a union
    GetUnionDiscriminant { result: HirId, union_val: HirId },

    /// Extract value from union variant (unsafe - assumes correct variant)
    ExtractUnionValue {
        result: HirId,
        ty: HirType,
        union_val: HirId,
        variant_index: u32,
    },

    /// Create a trait object (fat pointer) with data and vtable pointers
    CreateTraitObject {
        result: HirId,
        trait_id: zyntax_typed_ast::TypeId,
        data_ptr: HirId,
        vtable_id: HirId, // Global vtable ID
    },

    /// Upcast a trait object to a super-trait
    ///
    /// Converts a trait object for SubTrait to a trait object for SuperTrait.
    /// Both trait objects share the same data pointer but have different vtables.
    ///
    /// Example: Shape: Drawable
    /// ```ignore
    /// let shape: dyn Shape = ...;
    /// let drawable: dyn Drawable = shape;  // Upcast
    /// ```
    ///
    /// Implementation:
    /// 1. Extract data pointer from sub-trait object (fat pointer field 0)
    /// 2. Load super-trait vtable for the same concrete type
    /// 3. Create new fat pointer { data_ptr, super_vtable_ptr }
    UpcastTraitObject {
        result: HirId,
        sub_trait_object: HirId, // Source fat pointer (dyn SubTrait)
        sub_trait_id: zyntax_typed_ast::TypeId,
        super_trait_id: zyntax_typed_ast::TypeId,
        super_vtable_id: HirId, // Global ID of super-trait vtable
    },

    /// Call a method on a trait object (dynamic dispatch)
    ///
    /// This performs the following operations:
    /// 1. Extract vtable pointer from trait object (fat pointer field 1)
    /// 2. Load function pointer from vtable[method_index]
    /// 3. Extract data pointer from trait object (fat pointer field 0)
    /// 4. Call function pointer with (self=data_ptr, ...args)
    TraitMethodCall {
        result: Option<HirId>,
        trait_object: HirId,            // Fat pointer { *data, *vtable }
        method_index: usize,            // Index into vtable
        method_sig: HirMethodSignature, // Method signature for type-safe call
        args: Vec<HirId>,               // Arguments (not including self)
        return_ty: HirType, // Redundant with method_sig.return_type but kept for backward compat
    },

    /// Create a closure with captured values
    CreateClosure {
        result: HirId,
        closure_ty: HirType,
        function: HirId,      // Function that implements the closure
        captures: Vec<HirId>, // Values to capture
    },

    /// Call a closure
    CallClosure {
        result: Option<HirId>,
        closure: HirId,
        args: Vec<HirId>,
    },

    /// Create a reference with lifetime tracking
    CreateRef {
        result: HirId,
        value: HirId,
        lifetime: HirLifetime,
        mutable: bool,
    },

    /// Dereference a reference
    Deref {
        result: HirId,
        ty: HirType,
        reference: HirId,
    },

    /// Move a value (transfer ownership)
    Move {
        result: HirId,
        ty: HirType,
        source: HirId,
    },

    /// Copy a value (for Copy types)
    Copy {
        result: HirId,
        ty: HirType,
        source: HirId,
    },

    /// Begin a lifetime scope
    BeginLifetime { lifetime: HirLifetime },

    /// End a lifetime scope
    EndLifetime { lifetime: HirLifetime },

    /// Assert that a lifetime outlives another
    LifetimeConstraint {
        longer: HirLifetime,
        shorter: HirLifetime,
    },

    // ========================================================================
    // Algebraic Effects Instructions
    // ========================================================================
    /// Perform an effect operation
    ///
    /// Invokes an effect operation, which will be handled by the nearest
    /// enclosing handler for this effect. The handler may resume the
    /// computation or abort it.
    ///
    /// Example:
    /// ```ignore
    /// let value = perform State.get()  // Perform 'get' operation on State effect
    /// ```
    PerformEffect {
        result: Option<HirId>,
        /// Effect being performed
        effect_id: HirId,
        /// Operation name within the effect
        op_name: InternedString,
        /// Arguments to the operation
        args: Vec<HirId>,
        /// Return type of the operation
        return_ty: HirType,
    },

    /// Install an effect handler for a computation
    ///
    /// Wraps the computation in `body_block` with the given handler.
    /// When effect operations are performed in the body, they will be
    /// dispatched to this handler.
    ///
    /// For non-resumable handlers, this can be compiled to simple inlining.
    /// For resumable handlers, this requires CPS transformation or
    /// continuation capture.
    HandleEffect {
        result: Option<HirId>,
        /// The handler to install
        handler_id: HirId,
        /// Handler state (captured variables)
        handler_state: Vec<HirId>,
        /// The computation to wrap
        body_block: HirId,
        /// Block to continue to after the handled computation completes
        continuation_block: HirId,
        /// Return type of the handled computation
        return_ty: HirType,
    },

    /// Resume a suspended computation (continuation)
    ///
    /// Used in effect handler implementations to resume the computation
    /// that performed the effect operation. The value is passed to the
    /// effect operation's result.
    ///
    /// Example:
    /// ```ignore
    /// handler StateHandler for State<S> {
    ///     def get(): S {
    ///         resume(self.state)  // Resume with current state
    ///     }
    /// }
    /// ```
    Resume {
        /// Value to pass to the effect operation's result
        value: HirId,
        /// The continuation to resume
        continuation: HirId,
    },

    /// Abort the current effect handler scope
    ///
    /// Used in effect handlers to abort the computation without resuming.
    /// The value becomes the result of the HandleEffect instruction.
    AbortEffect {
        /// Value to return from the handler
        value: HirId,
        /// The handler scope being aborted
        handler_scope: HirId,
    },

    /// Capture the current continuation
    ///
    /// Creates a reified continuation that can be stored and resumed later.
    /// Used for multi-shot effects (where a handler may resume multiple times).
    CaptureContinuation {
        result: HirId,
        /// Type of value the continuation expects when resumed
        resume_ty: HirType,
    },

    // ========================================================================
    // SIMD / Vector Instructions
    // ========================================================================

    /// Broadcast a scalar value to all lanes of a SIMD vector.
    ///
    /// `ty` must be `HirType::Vector(elem_ty, lanes)`.
    /// `scalar` must be a value of type `elem_ty`.
    VectorSplat {
        result: HirId,
        ty: HirType,
        scalar: HirId,
    },

    /// Extract a single lane from a SIMD vector register to a scalar.
    ///
    /// `ty` is the scalar element type (output).
    /// `lane` must be less than the vector's lane count.
    VectorExtractLane {
        result: HirId,
        ty: HirType,
        vector: HirId,
        lane: u8,
    },

    /// Insert a scalar into a specific lane of a SIMD vector.
    ///
    /// `ty` must be `HirType::Vector(elem_ty, lanes)` (output type).
    /// `scalar` must match the element type of `vector`.
    VectorInsertLane {
        result: HirId,
        ty: HirType,
        vector: HirId,
        scalar: HirId,
        lane: u8,
    },

    /// Reduce all lanes of a SIMD vector to a single scalar using a
    /// commutative binary operation.
    ///
    /// `ty` is the scalar output type (element type of `vector`).
    /// Supported ops: Add, Sub, FAdd, FSub — only Add and FAdd make sense
    /// semantically for reductions; Sub/FSub reduce left-to-right.
    VectorHorizontalReduce {
        result: HirId,
        ty: HirType,
        vector: HirId,
        op: BinaryOp,
    },

    /// Load a SIMD vector from a memory pointer.
    ///
    /// `ty` must be `HirType::Vector(elem_ty, lanes)`.
    /// `ptr` must point to the first element (element type, not vector type).
    /// Loads `lanes` contiguous elements starting at `ptr`.
    VectorLoad {
        result: HirId,
        ty: HirType,
        ptr: HirId,
        align: u32,
    },

    /// Store a SIMD vector to a memory pointer.
    ///
    /// `value` must have type `HirType::Vector(elem_ty, lanes)`.
    /// `ptr` must point to the first element (element type, not vector type).
    /// Stores `lanes` contiguous elements starting at `ptr`.
    VectorStore {
        value: HirId,
        ptr: HirId,
        align: u32,
    },
}

/// Block terminator instructions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HirTerminator {
    /// Return from function
    Return { values: Vec<HirId> },

    /// Unconditional branch
    Branch { target: HirId },

    /// Conditional branch
    CondBranch {
        condition: HirId,
        true_target: HirId,
        false_target: HirId,
    },

    /// Multi-way branch (switch)
    Switch {
        value: HirId,
        default: HirId,
        cases: Vec<(HirConstant, HirId)>,
    },

    /// Unreachable code
    Unreachable,

    /// Exception handling
    Invoke {
        callee: HirCallable,
        args: Vec<HirId>,
        normal: HirId,
        unwind: HirId,
    },

    /// Pattern match on a value
    PatternMatch {
        value: HirId,
        patterns: Vec<HirPattern>,
        default: Option<HirId>,
    },
}

impl HirInstruction {
    /// Replace uses of old values with new values according to the replacement map
    pub fn replace_uses(&mut self, replacements: &IndexMap<HirId, HirId>) {
        fn replace(id: &mut HirId, map: &IndexMap<HirId, HirId>) {
            if let Some(&new_id) = map.get(id) {
                *id = new_id;
            }
        }

        match self {
            HirInstruction::Binary { left, right, .. } => {
                replace(left, replacements);
                replace(right, replacements);
            }
            HirInstruction::Unary { operand, .. } => {
                replace(operand, replacements);
            }
            HirInstruction::Alloca { count, .. } => {
                if let Some(c) = count {
                    replace(c, replacements);
                }
            }
            HirInstruction::Load { ptr, .. } => {
                replace(ptr, replacements);
            }
            HirInstruction::Store { value, ptr, .. } => {
                replace(value, replacements);
                replace(ptr, replacements);
            }
            HirInstruction::GetElementPtr { ptr, indices, .. } => {
                replace(ptr, replacements);
                for idx in indices {
                    replace(idx, replacements);
                }
            }
            HirInstruction::Call { args, .. } => {
                for arg in args {
                    replace(arg, replacements);
                }
            }
            HirInstruction::IndirectCall { func_ptr, args, .. } => {
                replace(func_ptr, replacements);
                for arg in args {
                    replace(arg, replacements);
                }
            }
            HirInstruction::Cast { operand, .. } => {
                replace(operand, replacements);
            }
            HirInstruction::Select {
                condition,
                true_val,
                false_val,
                ..
            } => {
                replace(condition, replacements);
                replace(true_val, replacements);
                replace(false_val, replacements);
            }
            HirInstruction::ExtractValue { aggregate, .. } => {
                replace(aggregate, replacements);
            }
            HirInstruction::InsertValue {
                aggregate, value, ..
            } => {
                replace(aggregate, replacements);
                replace(value, replacements);
            }
            HirInstruction::Atomic { ptr, value, .. } => {
                replace(ptr, replacements);
                if let Some(v) = value {
                    replace(v, replacements);
                }
            }
            HirInstruction::Fence { .. } => {}
            HirInstruction::CreateUnion { value, .. } => {
                replace(value, replacements);
            }
            HirInstruction::GetUnionDiscriminant { union_val, .. } => {
                replace(union_val, replacements);
            }
            HirInstruction::ExtractUnionValue { union_val, .. } => {
                replace(union_val, replacements);
            }
            HirInstruction::CreateTraitObject {
                data_ptr,
                vtable_id,
                ..
            } => {
                replace(data_ptr, replacements);
                replace(vtable_id, replacements);
            }
            HirInstruction::UpcastTraitObject {
                sub_trait_object,
                super_vtable_id,
                ..
            } => {
                replace(sub_trait_object, replacements);
                replace(super_vtable_id, replacements);
            }
            HirInstruction::TraitMethodCall {
                trait_object, args, ..
            } => {
                replace(trait_object, replacements);
                for arg in args {
                    replace(arg, replacements);
                }
            }
            HirInstruction::CreateClosure {
                function, captures, ..
            } => {
                replace(function, replacements);
                for cap in captures {
                    replace(cap, replacements);
                }
            }
            HirInstruction::CallClosure { closure, args, .. } => {
                replace(closure, replacements);
                for arg in args {
                    replace(arg, replacements);
                }
            }
            HirInstruction::CreateRef { value, .. } => {
                replace(value, replacements);
            }
            HirInstruction::Deref { reference, .. } => {
                replace(reference, replacements);
            }
            HirInstruction::Move { source, .. } => {
                replace(source, replacements);
            }
            HirInstruction::Copy { source, .. } => {
                replace(source, replacements);
            }
            HirInstruction::BeginLifetime { .. }
            | HirInstruction::EndLifetime { .. }
            | HirInstruction::LifetimeConstraint { .. } => {}
            // Algebraic effects
            HirInstruction::PerformEffect {
                effect_id, args, ..
            } => {
                replace(effect_id, replacements);
                for arg in args {
                    replace(arg, replacements);
                }
            }
            HirInstruction::HandleEffect {
                handler_id,
                handler_state,
                body_block,
                continuation_block,
                ..
            } => {
                replace(handler_id, replacements);
                for s in handler_state {
                    replace(s, replacements);
                }
                replace(body_block, replacements);
                replace(continuation_block, replacements);
            }
            HirInstruction::Resume {
                value,
                continuation,
            } => {
                replace(value, replacements);
                replace(continuation, replacements);
            }
            HirInstruction::AbortEffect {
                value,
                handler_scope,
            } => {
                replace(value, replacements);
                replace(handler_scope, replacements);
            }
            HirInstruction::CaptureContinuation { .. } => {}
            // SIMD instructions
            HirInstruction::VectorSplat { scalar, .. } => {
                replace(scalar, replacements);
            }
            HirInstruction::VectorExtractLane { vector, .. } => {
                replace(vector, replacements);
            }
            HirInstruction::VectorInsertLane { vector, scalar, .. } => {
                replace(vector, replacements);
                replace(scalar, replacements);
            }
            HirInstruction::VectorHorizontalReduce { vector, .. } => {
                replace(vector, replacements);
            }
            HirInstruction::VectorLoad { ptr, .. } => {
                replace(ptr, replacements);
            }
            HirInstruction::VectorStore { value, ptr, .. } => {
                replace(value, replacements);
                replace(ptr, replacements);
            }
        }
    }

    /// Get all operand HirIds used by this instruction
    pub fn operands(&self) -> Vec<HirId> {
        let mut ops = Vec::new();
        match self {
            HirInstruction::Binary { left, right, .. } => {
                ops.push(*left);
                ops.push(*right);
            }
            HirInstruction::Unary { operand, .. } => {
                ops.push(*operand);
            }
            HirInstruction::Alloca { count, .. } => {
                if let Some(c) = count {
                    ops.push(*c);
                }
            }
            HirInstruction::Load { ptr, .. } => {
                ops.push(*ptr);
            }
            HirInstruction::Store { value, ptr, .. } => {
                ops.push(*value);
                ops.push(*ptr);
            }
            HirInstruction::GetElementPtr { ptr, indices, .. } => {
                ops.push(*ptr);
                ops.extend(indices.iter().copied());
            }
            HirInstruction::Call { args, .. } => {
                ops.extend(args.iter().copied());
            }
            HirInstruction::IndirectCall { func_ptr, args, .. } => {
                ops.push(*func_ptr);
                ops.extend(args.iter().copied());
            }
            HirInstruction::Cast { operand, .. } => {
                ops.push(*operand);
            }
            HirInstruction::Select {
                condition,
                true_val,
                false_val,
                ..
            } => {
                ops.push(*condition);
                ops.push(*true_val);
                ops.push(*false_val);
            }
            HirInstruction::ExtractValue { aggregate, .. } => {
                ops.push(*aggregate);
            }
            HirInstruction::InsertValue {
                aggregate, value, ..
            } => {
                ops.push(*aggregate);
                ops.push(*value);
            }
            HirInstruction::Atomic { ptr, value, .. } => {
                ops.push(*ptr);
                if let Some(v) = value {
                    ops.push(*v);
                }
            }
            HirInstruction::Fence { .. } => {}
            HirInstruction::CreateUnion { value, .. } => {
                ops.push(*value);
            }
            HirInstruction::GetUnionDiscriminant { union_val, .. } => {
                ops.push(*union_val);
            }
            HirInstruction::ExtractUnionValue { union_val, .. } => {
                ops.push(*union_val);
            }
            HirInstruction::CreateTraitObject {
                data_ptr,
                vtable_id,
                ..
            } => {
                ops.push(*data_ptr);
                ops.push(*vtable_id);
            }
            HirInstruction::UpcastTraitObject {
                sub_trait_object,
                super_vtable_id,
                ..
            } => {
                ops.push(*sub_trait_object);
                ops.push(*super_vtable_id);
            }
            HirInstruction::TraitMethodCall {
                trait_object, args, ..
            } => {
                ops.push(*trait_object);
                ops.extend(args.iter().copied());
            }
            HirInstruction::CreateClosure {
                function, captures, ..
            } => {
                ops.push(*function);
                ops.extend(captures.iter().copied());
            }
            HirInstruction::CallClosure { closure, args, .. } => {
                ops.push(*closure);
                ops.extend(args.iter().copied());
            }
            HirInstruction::CreateRef { value, .. } => {
                ops.push(*value);
            }
            HirInstruction::Deref { reference, .. } => {
                ops.push(*reference);
            }
            HirInstruction::Move { source, .. } => {
                ops.push(*source);
            }
            HirInstruction::Copy { source, .. } => {
                ops.push(*source);
            }
            HirInstruction::BeginLifetime { .. }
            | HirInstruction::EndLifetime { .. }
            | HirInstruction::LifetimeConstraint { .. } => {}
            // Algebraic effects
            HirInstruction::PerformEffect {
                effect_id, args, ..
            } => {
                ops.push(*effect_id);
                ops.extend(args.iter().copied());
            }
            HirInstruction::HandleEffect {
                handler_id,
                handler_state,
                body_block,
                continuation_block,
                ..
            } => {
                ops.push(*handler_id);
                ops.extend(handler_state.iter().copied());
                ops.push(*body_block);
                ops.push(*continuation_block);
            }
            HirInstruction::Resume {
                value,
                continuation,
            } => {
                ops.push(*value);
                ops.push(*continuation);
            }
            HirInstruction::AbortEffect {
                value,
                handler_scope,
            } => {
                ops.push(*value);
                ops.push(*handler_scope);
            }
            HirInstruction::CaptureContinuation { .. } => {}
            // SIMD instructions
            HirInstruction::VectorSplat { scalar, .. } => {
                ops.push(*scalar);
            }
            HirInstruction::VectorExtractLane { vector, .. } => {
                ops.push(*vector);
            }
            HirInstruction::VectorInsertLane { vector, scalar, .. } => {
                ops.push(*vector);
                ops.push(*scalar);
            }
            HirInstruction::VectorHorizontalReduce { vector, .. } => {
                ops.push(*vector);
            }
            HirInstruction::VectorLoad { ptr, .. } => {
                ops.push(*ptr);
            }
            HirInstruction::VectorStore { value, ptr, .. } => {
                ops.push(*value);
                ops.push(*ptr);
            }
        }
        ops
    }
}

impl HirTerminator {
    /// Replace uses of old values with new values according to the replacement map
    pub fn replace_uses(&mut self, replacements: &IndexMap<HirId, HirId>) {
        fn replace(id: &mut HirId, map: &IndexMap<HirId, HirId>) {
            if let Some(&new_id) = map.get(id) {
                *id = new_id;
            }
        }

        match self {
            HirTerminator::Return { values } => {
                for v in values {
                    replace(v, replacements);
                }
            }
            HirTerminator::Branch { .. } => {}
            HirTerminator::CondBranch { condition, .. } => {
                replace(condition, replacements);
            }
            HirTerminator::Switch { value, .. } => {
                replace(value, replacements);
            }
            HirTerminator::Unreachable => {}
            HirTerminator::Invoke { args, .. } => {
                for arg in args {
                    replace(arg, replacements);
                }
            }
            HirTerminator::PatternMatch { value, .. } => {
                replace(value, replacements);
            }
        }
    }
}

/// HIR value in SSA form
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirValue {
    pub id: HirId,
    pub ty: HirType,
    pub kind: HirValueKind,
    /// Uses of this value for def-use chains
    pub uses: HashSet<HirId>,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HirValueKind {
    /// Function parameter
    Parameter(u32),
    /// Instruction result
    Instruction,
    /// Constant value
    Constant(HirConstant),
    /// Global reference
    Global(HirId),
    /// Undefined value (for optimizations)
    Undef,
}

/// HIR types - subset compatible with both backends
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HirType {
    /// Void type
    Void,

    /// Boolean (i1)
    Bool,

    /// Integer types
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,

    /// Floating point types
    F32,
    F64,

    /// Pointer type
    Ptr(Box<HirType>),

    /// Reference type with lifetime
    Ref {
        lifetime: HirLifetime,
        pointee: Box<HirType>,
        mutable: bool,
    },

    /// Array type
    Array(Box<HirType>, u64),

    /// Vector type (SIMD)
    Vector(Box<HirType>, u32),

    /// Structure type
    Struct(HirStructType),

    /// Union type (tagged union with discriminant)
    Union(Box<HirUnionType>),

    /// Function type
    Function(Box<HirFunctionType>),

    /// Closure type (function with captured environment)
    Closure(Box<HirClosureType>),

    /// Opaque type (for forward declarations)
    Opaque(InternedString),

    /// Const generic parameter reference
    ConstGeneric(InternedString),

    /// Type with const generic arguments
    Generic {
        base: Box<HirType>,
        type_args: Vec<HirType>,
        const_args: Vec<HirConstant>,
    },

    /// Trait object type (dyn Trait)
    /// Represents a trait object with dynamic dispatch
    TraitObject {
        trait_id: zyntax_typed_ast::TypeId,
        vtable: Option<HirId>, // vtable global ID (resolved during lowering)
    },

    /// Interface type (structural or nominal)
    /// Used for both Go-style structural interfaces and Java-style nominal interfaces
    Interface {
        methods: Vec<HirMethodSignature>,
        is_structural: bool, // true for duck-typing, false for nominal
    },

    /// Promise type for async functions
    ///
    /// An async function `async fn foo() -> T` returns `Promise<T>`.
    /// The Promise contains:
    /// - A pointer to the state machine struct
    /// - A function pointer to the poll function
    ///
    /// At runtime, Promise is represented as a struct:
    /// ```ignore
    /// struct Promise<T> {
    ///     state_machine: *mut StateMachine,
    ///     poll_fn: fn(*mut StateMachine, *Context) -> PollResult<T>,
    /// }
    /// ```
    Promise(Box<HirType>),

    /// Associated type projection
    /// Represents an associated type in a trait, e.g., `<T as Iterator>::Item`
    ///
    /// During monomorphization, this gets resolved to a concrete type based on
    /// the trait implementation's associated type binding.
    ///
    /// Example:
    /// ```ignore
    /// trait Iterator {
    ///     type Item;  // Associated type declaration
    ///     fn next(&mut self) -> Option<Self::Item>;
    /// }
    ///
    /// impl Iterator for Vec<i32> {
    ///     type Item = i32;  // Associated type binding
    ///     fn next(&mut self) -> Option<i32> { ... }
    /// }
    /// ```
    ///
    /// In HIR, `<Vec<i32> as Iterator>::Item` becomes:
    /// ```ignore
    /// HirType::AssociatedType {
    ///     trait_id: Iterator,
    ///     self_ty: Vec<i32>,
    ///     name: "Item"
    /// }
    /// ```
    ///
    /// Which resolves to `HirType::I32` during monomorphization.
    AssociatedType {
        trait_id: zyntax_typed_ast::TypeId,
        self_ty: Box<HirType>, // The implementing type
        name: InternedString,  // Associated type name (e.g., "Item")
    },

    /// Continuation type for algebraic effects
    ///
    /// Represents a captured continuation that can be resumed.
    /// The `resume_ty` is the type of value that can be passed when resuming.
    /// The `result_ty` is the type of value returned when the continuation completes.
    ///
    /// Example:
    /// In a handler for `effect State<S> { def get(): S }`,
    /// the continuation type would be `Continuation<S, T>` where S is
    /// passed to `get()`'s result and T is the return type of the handled computation.
    Continuation {
        /// Type of value passed when resuming (becomes the effect operation result)
        resume_ty: Box<HirType>,
        /// Type of value returned when the continuation completes
        result_ty: Box<HirType>,
    },

    /// Effect row type for effect polymorphism
    ///
    /// Represents a set of effects that a function may perform.
    /// Used for effect inference and checking.
    EffectRow {
        /// Known effects in the row
        effects: Vec<InternedString>,
        /// Optional tail variable for open effect rows
        tail: Option<InternedString>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirStructType {
    pub name: Option<InternedString>,
    pub fields: Vec<HirType>,
    pub packed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirUnionType {
    pub name: Option<InternedString>,
    /// Variants with their names and types
    pub variants: Vec<HirUnionVariant>,
    /// Discriminant type (usually u8 or u32)
    pub discriminant_type: Box<HirType>,
    /// Whether this is a C-style union (no discriminant)
    pub is_c_union: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirUnionVariant {
    pub name: InternedString,
    pub ty: HirType,
    /// Discriminant value for this variant
    pub discriminant: u64,
}

/// Pattern for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirPattern {
    pub kind: HirPatternKind,
    pub target: HirId,                    // Block to jump to if pattern matches
    pub bindings: Vec<HirPatternBinding>, // Variables to bind if pattern matches
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HirPatternKind {
    /// Literal constant pattern
    Constant(HirConstant),
    /// Union variant pattern
    UnionVariant {
        union_ty: HirType,
        variant_index: u32,
        /// Sub-pattern for the variant's value
        inner_pattern: Option<Box<HirPattern>>,
    },
    /// Struct destructuring pattern
    Struct {
        struct_ty: HirType,
        field_patterns: Vec<(u32, HirPattern)>, // (field_index, pattern)
    },
    /// Wildcard pattern (matches anything)
    Wildcard,
    /// Binding pattern (binds to a variable)
    Binding(InternedString),
    /// Guard pattern (pattern with additional condition)
    Guard {
        pattern: Box<HirPattern>,
        condition: HirId, // Boolean expression that must be true
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirPatternBinding {
    pub name: InternedString,
    pub value_id: HirId,
    pub ty: HirType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirFunctionType {
    pub params: Vec<HirType>,
    pub returns: Vec<HirType>,
    pub lifetime_params: Vec<HirLifetime>,
    pub is_variadic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirClosureType {
    pub function_type: HirFunctionType,
    pub captures: Vec<HirCapture>,
    /// Whether this closure can be called once (FnOnce), multiple times (Fn), or mutably (FnMut)
    pub call_mode: HirClosureCallMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirCapture {
    pub name: InternedString,
    pub ty: HirType,
    pub mode: HirCaptureMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HirCaptureMode {
    /// Capture by value (move)
    ByValue,
    /// Capture by immutable reference
    ByRef,
    /// Capture by mutable reference
    ByMutRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HirClosureCallMode {
    /// Can only be called once (FnOnce)
    Once,
    /// Can be called multiple times immutably (Fn)
    Fn,
    /// Can be called multiple times mutably (FnMut)
    FnMut,
}

/// Constant values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HirConstant {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    F32(f32),
    F64(f64),
    Null(HirType),
    Array(Vec<HirConstant>),
    Struct(Vec<HirConstant>),
    String(InternedString),
    /// Virtual method table for trait dispatch
    VTable(HirVTable),
}

impl Eq for HirConstant {}

impl std::hash::Hash for HirConstant {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use HirConstant::*;
        std::mem::discriminant(self).hash(state);
        match self {
            Bool(v) => v.hash(state),
            I8(v) => v.hash(state),
            I16(v) => v.hash(state),
            I32(v) => v.hash(state),
            I64(v) => v.hash(state),
            I128(v) => v.hash(state),
            U8(v) => v.hash(state),
            U16(v) => v.hash(state),
            U32(v) => v.hash(state),
            U64(v) => v.hash(state),
            U128(v) => v.hash(state),
            // For floating point, hash the bits representation
            F32(v) => v.to_bits().hash(state),
            F64(v) => v.to_bits().hash(state),
            Null(ty) => ty.hash(state),
            Array(vals) => vals.hash(state),
            Struct(vals) => vals.hash(state),
            String(s) => s.hash(state),
            VTable(vtable) => vtable.id.hash(state), // Hash by ID
        }
    }
}

/// Binary operations supported by both backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    // Bitwise
    And,
    Or,
    Xor,
    Shl,
    Shr,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Floating point
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    // Floating point comparison
    FEq,
    FNe,
    FLt,
    FLe,
    FGt,
    FGe,
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
    FNeg,
}

/// Cast operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CastOp {
    // Integer casts
    Trunc,
    ZExt,
    SExt,
    // Float casts
    FpTrunc,
    FpExt,
    // Float/Int conversions
    FpToUi,
    FpToSi,
    UiToFp,
    SiToFp,
    // Pointer casts
    PtrToInt,
    IntToPtr,
    // Bitcast (reinterpret)
    Bitcast,
}

/// Atomic operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomicOp {
    Load,
    Store,
    Exchange,
    Add,
    Sub,
    And,
    Or,
    Xor,
    CompareExchange,
}

/// Memory ordering for atomics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomicOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Calling conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallingConvention {
    Fast,   // Internal functions
    C,      // C calling convention
    System, // Platform default
    WebKit, // For JS interop
}

/// Function attributes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionAttributes {
    pub inline: InlineHint,
    pub no_return: bool,
    pub no_unwind: bool,
    pub no_inline: bool,
    pub always_inline: bool,
    pub cold: bool,
    pub hot: bool,
    pub pure: bool,
    pub const_fn: bool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InlineHint {
    #[default]
    None,
    Hint,
    Always,
    Never,
}

/// Global variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirGlobal {
    pub id: HirId,
    pub name: InternedString,
    pub ty: HirType,
    pub initializer: Option<HirConstant>,
    pub is_const: bool,
    pub is_thread_local: bool,
    pub linkage: Linkage,
    pub visibility: Visibility,
}

/// Virtual method table (vtable) for trait objects
///
/// A vtable contains function pointers for all methods of a trait implementation.
/// Layout: [method_0_ptr, method_1_ptr, ..., method_N_ptr]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HirVTable {
    pub id: HirId,
    pub trait_id: zyntax_typed_ast::TypeId,
    pub for_type: HirType,
    pub methods: Vec<HirVTableEntry>,
}

/// Entry in a vtable mapping method name to implementation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HirVTableEntry {
    pub method_name: InternedString,
    pub function_id: HirId, // ID of the implementing function
    pub signature: HirMethodSignature,
}

/// Import declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirImport {
    pub name: InternedString,
    pub kind: ImportKind,
    pub attributes: ImportAttributes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportKind {
    /// External function import
    Function(HirFunctionSignature),
    /// External global/constant import
    Global(HirType),
    /// External type import (opaque or defined)
    Type {
        /// The HIR representation of the type
        ty: HirType,
        /// Type ID for symbol table registration
        type_id: zyntax_typed_ast::TypeId,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportAttributes {
    pub dll_import: Option<String>,
    pub weak: bool,
}

/// Export declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirExport {
    pub name: InternedString,
    pub internal_name: InternedString,
    pub kind: ExportKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportKind {
    Function(HirId),
    Global(HirId),
}

/// Linkage types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Linkage {
    External,
    Internal,
    Private,
    Weak,
    LinkOnce,
}

/// Visibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Default,
    Hidden,
    Protected,
}

/// Callable target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HirCallable {
    /// Direct function call
    Function(HirId),
    /// Indirect call through function pointer
    Indirect(HirId),
    /// Intrinsic function
    Intrinsic(Intrinsic),
    /// External symbol call (e.g., "$haxe$trace$int" for runtime functions)
    /// The symbol name is looked up in the runtime symbol registry at link time
    Symbol(String),
}

/// Compiler intrinsics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Intrinsic {
    // Memory
    Memcpy,
    Memset,
    Memmove,
    // Math
    Sqrt,
    Sin,
    Cos,
    Pow,
    Log,
    Exp,
    // Bit manipulation
    Ctpop,
    Ctlz,
    Cttz,
    Bswap,
    // Type queries
    SizeOf,
    AlignOf,
    // Overflow checking
    AddWithOverflow,
    SubWithOverflow,
    MulWithOverflow,
    // Memory management
    Malloc,  // Allocate heap memory
    Free,    // Free heap memory
    Realloc, // Resize allocation
    Drop,    // Call destructor
    // Reference counting (optional)
    IncRef, // Increment reference count
    DecRef, // Decrement reference count
    // Stack allocation (already exists as instruction)
    Alloca, // Stack allocation
    // Garbage collection
    GCSafepoint, // GC safepoint for collection
    // Async/coroutine support
    Await, // Await a future
    Yield, // Yield a value (generators)
    // Error handling (Gap 8)
    Panic, // Panic with message (unrecoverable error)
    Abort, // Abort execution immediately (no cleanup)

    // ZRTL Value Conversion (for extern/plugin calls)
    /// Convert closure to ZrtlClosure: (fn_ptr, env_ptr, env_size) -> *ZrtlClosure
    /// Args: fn_ptr (raw function pointer), env_ptr (captured environment), env_size (i64)
    ClosureToZrtl,
    /// Convert DynamicBox to ZRTL DynamicBox: (tag, value_ptr) -> *DynamicBox
    /// Args: type_tag (i32), value_ptr (pointer to value)
    BoxToZrtl,
    /// Convert primitive to DynamicBox: (value, type_tag) -> *DynamicBox
    /// Args: value (any primitive), type_tag (i32)
    PrimitiveToBox,
    /// Get type tag for a type: () -> i32
    /// Computes ZRTL TypeTag from HIR type
    TypeTagOf,
}

/// Local variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirLocal {
    pub id: HirId,
    pub name: InternedString,
    pub ty: HirType,
    pub is_mutable: bool,
    pub lifetime: Option<HirLifetime>,
}

/// Type parameter for generics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirTypeParam {
    pub name: InternedString,
    pub constraints: Vec<TypeConstraint>,
}

/// Const parameter for const generics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirConstParam {
    pub name: InternedString,
    pub ty: HirType,
    pub default: Option<HirConstant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeConstraint {
    /// Must implement trait/interface
    Trait(InternedString),
    /// Must be a subtype
    Subtype(HirType),
    /// Size constraint
    Sized,
}

/// Borrow checking context for a function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorrowCheckContext {
    /// Active borrows at each program point
    pub active_borrows: IndexMap<HirId, Vec<BorrowInfo>>,
    /// Move information
    pub moves: IndexMap<HirId, MoveInfo>,
    /// Lifetime constraints
    pub lifetime_constraints: Vec<LifetimeConstraint>,
    /// Local variable lifetimes
    pub local_lifetimes: IndexMap<HirId, HirLifetime>,
}

/// Information about an active borrow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorrowInfo {
    pub borrow_id: HirId,
    pub borrowed_value: HirId,
    pub lifetime: HirLifetime,
    pub is_mutable: bool,
    pub borrow_location: Option<Span>,
}

/// Information about a move operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveInfo {
    pub moved_value: HirId,
    pub move_location: Option<Span>,
    pub destination: HirId,
}

/// Lifetime constraint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifetimeConstraint {
    pub longer_lifetime: HirLifetime,
    pub shorter_lifetime: HirLifetime,
    pub constraint_location: Option<Span>,
}

/// Ownership mode for values
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OwnershipMode {
    /// Value is owned (moved)
    Owned,
    /// Value is borrowed immutably
    Borrowed(HirLifetime),
    /// Value is borrowed mutably
    BorrowedMut(HirLifetime),
    /// Value is copied (for Copy types)
    Copied,
}

impl BorrowCheckContext {
    pub fn new() -> Self {
        Self {
            active_borrows: IndexMap::new(),
            moves: IndexMap::new(),
            lifetime_constraints: Vec::new(),
            local_lifetimes: IndexMap::new(),
        }
    }

    /// Add a new borrow
    pub fn add_borrow(&mut self, point: HirId, borrow: BorrowInfo) {
        self.active_borrows
            .entry(point)
            .or_insert_with(Vec::new)
            .push(borrow);
    }

    /// Record a move operation
    pub fn add_move(&mut self, moved_value: HirId, move_info: MoveInfo) {
        self.moves.insert(moved_value, move_info);
    }

    /// Add a lifetime constraint
    pub fn add_lifetime_constraint(&mut self, constraint: LifetimeConstraint) {
        self.lifetime_constraints.push(constraint);
    }

    /// Check if a value has been moved
    pub fn is_moved(&self, value: HirId) -> bool {
        self.moves.contains_key(&value)
    }

    /// Get active borrows at a program point
    pub fn get_active_borrows(&self, point: HirId) -> &[BorrowInfo] {
        self.active_borrows
            .get(&point)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

impl HirModule {
    pub fn new(name: InternedString) -> Self {
        Self {
            id: HirId::new(),
            name,
            functions: IndexMap::new(),
            globals: IndexMap::new(),
            types: IndexMap::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            version: 0,
            dependencies: HashSet::new(),
            effects: IndexMap::new(),
            handlers: IndexMap::new(),
        }
    }

    /// Add a function to the module
    pub fn add_function(&mut self, func: HirFunction) {
        self.functions.insert(func.id, func);
    }

    /// Add an effect declaration to the module
    pub fn add_effect(&mut self, effect: HirEffect) {
        self.effects.insert(effect.id, effect);
    }

    /// Add an effect handler to the module
    pub fn add_handler(&mut self, handler: HirEffectHandler) {
        self.handlers.insert(handler.id, handler);
    }

    /// Add a global to the module
    pub fn add_global(&mut self, global: HirGlobal) {
        self.globals.insert(global.id, global);
    }

    /// Increment version for hot-reloading
    pub fn increment_version(&mut self) {
        self.version += 1;
    }
}

impl HirFunction {
    pub fn new(name: InternedString, signature: HirFunctionSignature) -> Self {
        let entry_block_id = HirId::new();
        let mut blocks = IndexMap::new();
        blocks.insert(entry_block_id, HirBlock::new(entry_block_id));

        Self {
            id: HirId::new(),
            name,
            signature,
            entry_block: entry_block_id,
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

    /// Create a new basic block
    pub fn create_block(&mut self) -> HirId {
        let block_id = HirId::new();
        self.blocks.insert(block_id, HirBlock::new(block_id));
        block_id
    }

    /// Create a new SSA value
    pub fn create_value(&mut self, ty: HirType, kind: HirValueKind) -> HirId {
        let value_id = HirId::new();
        let value = HirValue {
            id: value_id,
            ty,
            kind,
            uses: HashSet::new(),
            span: None,
        };
        self.values.insert(value_id, value);
        value_id
    }
}

impl HirBlock {
    pub fn new(id: HirId) -> Self {
        Self {
            id,
            label: None,
            phis: Vec::new(),
            instructions: Vec::new(),
            terminator: HirTerminator::Unreachable,
            dominance_frontier: HashSet::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }

    /// Add a phi node
    pub fn add_phi(&mut self, phi: HirPhi) {
        self.phis.push(phi);
    }

    /// Add an instruction
    pub fn add_instruction(&mut self, inst: HirInstruction) {
        self.instructions.push(inst);
    }

    /// Set the terminator
    pub fn set_terminator(&mut self, term: HirTerminator) {
        self.terminator = term;
    }
}
