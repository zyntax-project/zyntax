//! Effect System for Tracking Side Effects
//!
//! Implements a comprehensive effect system for tracking and managing side effects
//! in programs. This enables pure functional programming paradigms while allowing
//! controlled side effects through explicit effect types and handlers.
//!
//! Key features:
//! - Effect types: I/O, State, Exceptions, Async, etc.
//! - Effect inference and checking
//! - Effect handlers and structured control flow
//! - Purity analysis and optimization
//! - Effect composition and algebraic operations

use crate::arena::InternedString;
use crate::source::Span;
use crate::type_registry::Type;
use crate::typed_ast::{
    TypedBlock, TypedDeclaration, TypedExpression, TypedFunction, TypedProgram, TypedStatement,
    TypedVariable,
};
use std::collections::{BTreeSet, HashMap};

/// Function signature for effect operations
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    pub params: Vec<Type>,
    pub return_type: Type,
    pub effects: EffectSet,
}

/// Effect system for tracking computational side effects
pub struct EffectSystem {
    /// Effect type definitions
    pub effect_types: HashMap<EffectId, EffectTypeInfo>,

    /// Function effect signatures
    pub function_effects: HashMap<InternedString, EffectSignature>,

    /// Effect inference context
    inference_context: EffectInferenceContext,

    /// Effect handlers in scope
    pub effect_handlers: Vec<EffectHandler>,

    /// Current effect context
    pub current_effects: EffectSet,

    /// Scope stack for effect tracking
    pub scope_stack: Vec<EffectScope>,

    /// Error accumulator
    pub errors: Vec<EffectError>,
}

/// Information about an effect type
#[derive(Debug, Clone, PartialEq)]
pub struct EffectTypeInfo {
    pub id: EffectId,
    pub name: InternedString,
    pub effect_kind: EffectKind,
    pub parameters: Vec<EffectParam>,
    pub constraints: Vec<EffectConstraint>,
    pub composition_rules: CompositionRules,
    pub handler_requirements: HandlerRequirements,
}

/// Different kinds of effects
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectKind {
    /// I/O operations (files, network, console)
    IO,

    /// State mutation (variables, data structures)
    State,

    /// Exception throwing and handling
    Exception,

    /// Asynchronous operations
    Async,

    /// Memory allocation and deallocation
    Memory,

    /// Non-deterministic operations (random, time)
    Nondeterministic,

    /// Divergence (infinite loops, non-termination)
    Divergence,

    /// Control flow effects (continuations, coroutines)
    Control,

    /// Resource management (locks, handles)
    Resource,

    /// Pure computation (no side effects)
    Pure,

    /// Custom user-defined effect
    Custom,
}

/// Effect parameters for parameterized effects
#[derive(Debug, Clone, PartialEq)]
pub struct EffectParam {
    pub name: InternedString,
    pub param_type: Type,
    pub variance: EffectVariance,
    pub constraints: Vec<EffectConstraint>,
}

/// Variance for effect parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectVariance {
    /// Covariant (effect can be substituted with subeffect)
    Covariant,

    /// Contravariant (effect can be substituted with supereffect)
    Contravariant,

    /// Invariant (exact effect match required)
    Invariant,

    /// Bivariant (any effect substitution allowed)
    Bivariant,
}

/// Effect signature for functions
#[derive(Debug, Clone, PartialEq)]
pub struct EffectSignature {
    /// Input effects (effects that must be available)
    pub input_effects: EffectSet,

    /// Output effects (effects that this function produces)
    pub output_effects: EffectSet,

    /// Effect transformations (how effects are modified)
    pub transformations: Vec<EffectTransformation>,

    /// Effect requirements
    pub requirements: Vec<EffectRequirement>,

    /// Whether this function is pure
    pub is_pure: bool,

    /// Effect bounds
    pub bounds: Vec<EffectBound>,
}

/// Set of effects
#[derive(Debug, Clone, PartialEq)]
pub struct EffectSet {
    /// Individual effects
    pub effects: BTreeSet<Effect>,

    /// Effect variables (for inference)
    pub variables: BTreeSet<EffectVar>,

    /// Effect unions
    pub unions: Vec<EffectUnion>,

    /// Effect intersections
    pub intersections: Vec<EffectIntersection>,
}

/// Individual effect instance
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Effect {
    pub id: EffectId,
    pub region: Option<EffectRegionId>,
    pub intensity: EffectIntensity,
}

/// Effect variable for inference
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EffectVar {
    pub id: EffectVarId,
    pub kind: EffectVarKind,
}

/// Effect union (A | B)
#[derive(Debug, Clone, PartialEq)]
pub struct EffectUnion {
    pub effects: Vec<EffectSet>,
    pub span: Span,
}

/// Effect intersection (A & B)
#[derive(Debug, Clone, PartialEq)]
pub struct EffectIntersection {
    pub effects: Vec<EffectSet>,
    pub span: Span,
}

/// Effect region for scoping effects
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EffectRegion {
    pub id: EffectRegionId,
    pub name: Option<InternedString>,
    pub parent: Option<EffectRegionId>,
    pub scope_kind: EffectScopeKind,
}

/// Effect intensity (how much of an effect)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EffectIntensity {
    /// No effect
    None,

    /// Potential effect (might happen)
    Potential,

    /// Definite effect (will happen)
    Definite,

    /// Repeated effect (happens multiple times)
    Repeated,

    /// Unbounded effect (unknown frequency)
    Unbounded,
}

/// Effect transformation
#[derive(Debug, Clone, PartialEq)]
pub enum EffectTransformation {
    /// Transform one effect to another
    Transform {
        from: Effect,
        to: Effect,
        condition: Option<EffectCondition>,
    },

    /// Filter out specific effects
    Filter {
        effects: EffectSet,
        predicate: EffectPredicate,
    },

    /// Mask effects (hide from caller)
    Mask {
        effects: EffectSet,
        replacement: Option<EffectSet>,
    },

    /// Amplify effect intensity
    Amplify {
        effect: Effect,
        factor: EffectIntensity,
    },

    /// Compose effects
    Compose {
        effects: Vec<EffectSet>,
        operator: CompositionOperator,
    },
}

/// Effect requirement
#[derive(Debug, Clone, PartialEq)]
pub enum EffectRequirement {
    /// Requires specific effect to be available
    RequiresEffect(Effect),

    /// Requires effect handler
    RequiresHandler(EffectId),

    /// Requires pure context
    RequiresPure,

    /// Requires specific effect capability
    RequiresCapability(EffectCapability),

    /// Custom requirement
    Custom(InternedString, Vec<Type>),
}

/// Effect capability
#[derive(Debug, Clone, PartialEq)]
pub struct EffectCapability {
    pub effect_id: EffectId,
    pub operations: Vec<EffectOperation>,
    pub permissions: EffectPermissions,
}

/// Effect operation
#[derive(Debug, Clone, PartialEq)]
pub struct EffectOperation {
    pub name: InternedString,
    pub signature: FunctionSignature,
    pub effect_type: Effect,
}

/// Effect permissions
#[derive(Debug, Clone, PartialEq)]
pub struct EffectPermissions {
    pub can_perform: bool,
    pub can_handle: bool,
    pub can_transform: bool,
    pub can_observe: bool,
}

/// Effect bound
#[derive(Debug, Clone, PartialEq)]
pub enum EffectBound {
    /// Effect must be subeffect of another
    SubEffect(Effect, Effect),

    /// Effect must equal another
    Equal(Effect, Effect),

    /// Effect must be disjoint from another
    Disjoint(Effect, Effect),

    /// Effect must be composable with another
    Composable(Effect, Effect),

    /// Custom constraint
    Custom(EffectPredicate),
}

/// Effect handler for structured control flow
#[derive(Debug, Clone, PartialEq)]
pub struct EffectHandler {
    pub id: EffectHandlerId,
    pub handled_effects: EffectSet,
    pub handler_type: HandlerType,
    pub operations: Vec<EffectHandlerOperation>,
    pub return_effect: Option<EffectSet>,
    pub scope: Span,
}

/// Types of effect handlers
#[derive(Debug, Clone, PartialEq)]
pub enum HandlerType {
    /// Try-catch style exception handler
    Exception,

    /// Algebraic effect handler
    Algebraic,

    /// Resource management handler (RAII)
    Resource,

    /// Async/await handler
    Async,

    /// State handler
    State,

    /// Custom handler
    Custom(InternedString),
}

/// Effect handler operation
#[derive(Debug, Clone, PartialEq)]
pub struct EffectHandlerOperation {
    pub operation_name: InternedString,
    pub parameters: Vec<EffectParam>,
    pub continuation_type: Option<Type>,
    pub handler_body: Option<InternedString>, // Reference to handler implementation
}

/// Effect inference context
#[derive(Debug, Clone)]
pub struct EffectInferenceContext {
    /// Effect variables and their bounds
    pub effect_vars: HashMap<EffectVarId, EffectVarInfo>,

    /// Constraints to solve
    pub constraints: Vec<EffectConstraint>,

    /// Current effect substitution
    pub substitution: EffectSubstitution,

    /// Inference options
    pub options: EffectInferenceOptions,
}

/// Effect variable information
#[derive(Debug, Clone, PartialEq)]
pub struct EffectVarInfo {
    pub id: EffectVarId,
    pub kind: EffectVarKind,
    pub bounds: Vec<EffectBound>,
    pub solution: Option<EffectSet>,
    pub scope_level: usize,
}

/// Effect variable kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EffectVarKind {
    /// Regular effect variable
    Effect,

    /// Effect row variable (for extensible effects)
    Row,

    /// Effect level variable (for effect levels/ranks)
    Level,

    /// Effect region variable
    Region,
}

/// Effect constraints for inference
#[derive(Debug, Clone, PartialEq)]
pub enum EffectConstraint {
    /// Effect equality
    Equal(EffectSet, EffectSet, Span),

    /// Effect subtyping
    SubEffect(EffectSet, EffectSet, Span),

    /// Effect disjointness
    Disjoint(EffectSet, EffectSet, Span),

    /// Effect composition
    Compose(EffectSet, EffectSet, EffectSet, CompositionOperator, Span),

    /// Effect variable bounds
    VarBounds(EffectVarId, Vec<EffectBound>, Span),

    /// Effect handler constraint
    HandlerAvailable(EffectSet, EffectHandlerId, Span),

    /// Purity constraint
    Pure(EffectSet, Span),

    /// Custom constraint
    Custom(EffectPredicate, Span),
}

/// Effect substitution
#[derive(Debug, Clone, Default)]
pub struct EffectSubstitution {
    pub effect_vars: HashMap<EffectVarId, EffectSet>,
    pub region_vars: HashMap<EffectRegionId, EffectRegion>,
}

/// Effect inference options
#[derive(Debug, Clone)]
pub struct EffectInferenceOptions {
    pub allow_effect_polymorphism: bool,
    pub infer_purity: bool,
    pub check_effect_safety: bool,
    pub enable_effect_optimization: bool,
    pub max_effect_depth: usize,
}

/// Effect scope for tracking
#[derive(Debug, Clone, PartialEq)]
pub struct EffectScope {
    pub scope_kind: EffectScopeKind,
    pub available_effects: EffectSet,
    pub handled_effects: EffectSet,
    pub required_effects: EffectSet,
    pub handlers: Vec<EffectHandlerId>,
    pub span: Span,
}

/// Effect scope kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EffectScopeKind {
    /// Function scope
    Function,

    /// Block scope
    Block,

    /// Handler scope
    Handler,

    /// Try scope
    Try,

    /// Async scope
    Async,

    /// Loop scope
    Loop,

    /// Conditional scope
    Conditional,
}

/// Effect condition
#[derive(Debug, Clone, PartialEq)]
pub enum EffectCondition {
    /// Always true
    Always,

    /// Never true
    Never,

    /// Depends on runtime value
    Runtime(InternedString),

    /// Depends on type information
    TypeDependent(Type),

    /// Custom predicate
    Custom(EffectPredicate),
}

/// Effect predicate
#[derive(Debug, Clone, PartialEq)]
pub struct EffectPredicate {
    pub name: InternedString,
    pub parameters: Vec<Type>,
    pub body: Option<InternedString>,
}

/// Composition operator for effects
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompositionOperator {
    /// Sequential composition (A; B)
    Sequential,

    /// Parallel composition (A || B)
    Parallel,

    /// Alternative composition (A | B)
    Alternative,

    /// Intersection composition (A & B)
    Intersection,

    /// Difference composition (A \ B)
    Difference,

    /// Custom composition
    Custom,
}

/// Composition rules for effect types
#[derive(Debug, Clone, PartialEq)]
pub struct CompositionRules {
    pub commutative: bool,
    pub associative: bool,
    pub idempotent: bool,
    pub identity: Option<Effect>,
    pub absorbing: Option<Effect>,
    pub custom_rules: Vec<CompositionRule>,
}

/// Custom composition rule
#[derive(Debug, Clone, PartialEq)]
pub struct CompositionRule {
    pub pattern: CompositionPattern,
    pub result: EffectSet,
    pub condition: Option<EffectCondition>,
}

/// Composition pattern
#[derive(Debug, Clone, PartialEq)]
pub enum CompositionPattern {
    /// Binary pattern (A op B)
    Binary(EffectSet, CompositionOperator, EffectSet),

    /// Unary pattern (op A)
    Unary(UnaryEffectOperator, EffectSet),

    /// N-ary pattern (op(A1, A2, ..., An))
    NAry(CompositionOperator, Vec<EffectSet>),

    /// Wildcard pattern
    Wildcard,
}

/// Unary effect operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryEffectOperator {
    /// Effect negation
    Not,

    /// Effect optionality
    Optional,

    /// Effect repetition
    Repeat,

    /// Effect amplification
    Amplify,
}

/// Handler requirements
#[derive(Debug, Clone, PartialEq)]
pub struct HandlerRequirements {
    pub required_operations: Vec<InternedString>,
    pub optional_operations: Vec<InternedString>,
    pub handler_constraints: Vec<HandlerConstraint>,
}

/// Handler constraint
#[derive(Debug, Clone, PartialEq)]
pub enum HandlerConstraint {
    /// Handler must be stateless
    Stateless,

    /// Handler must be reentrant
    Reentrant,

    /// Handler must preserve effects
    EffectPreserving,

    /// Custom constraint
    Custom(InternedString),
}

/// Unique identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EffectId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EffectVarId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EffectRegionId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EffectHandlerId(u32);

impl EffectId {
    pub fn next() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_ID: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl EffectVarId {
    pub fn next() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_ID: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl EffectRegionId {
    pub fn next() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_ID: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl EffectHandlerId {
    pub fn next() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_ID: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Effect system errors
#[derive(Debug, Clone, PartialEq)]
pub enum EffectError {
    /// Unhandled effect
    UnhandledEffect {
        effect: Effect,
        span: Span,
        available_handlers: Vec<EffectHandlerId>,
    },

    /// Effect not available in context
    EffectNotAvailable {
        effect: Effect,
        context: EffectSet,
        span: Span,
    },

    /// Conflicting effects
    ConflictingEffects {
        effect1: Effect,
        effect2: Effect,
        conflict_reason: String,
        span: Span,
    },

    /// Invalid effect composition
    InvalidComposition {
        effects: Vec<EffectSet>,
        operator: CompositionOperator,
        reason: String,
        span: Span,
    },

    /// Effect constraint violation
    ConstraintViolation {
        constraint: EffectConstraint,
        violation_span: Span,
        reason: String,
    },

    /// Missing effect handler
    MissingHandler {
        effect: Effect,
        required_operations: Vec<InternedString>,
        span: Span,
    },

    /// Invalid effect handler
    InvalidHandler {
        handler_id: EffectHandlerId,
        effect: Effect,
        reason: String,
        span: Span,
    },

    /// Effect inference failed
    InferenceFailed {
        effect_var: EffectVarId,
        constraints: Vec<EffectConstraint>,
        span: Span,
    },

    /// Purity violation
    PurityViolation {
        expected_pure: bool,
        actual_effects: EffectSet,
        span: Span,
    },

    /// Effect scope error
    ScopeError {
        scope_kind: EffectScopeKind,
        error_type: String,
        span: Span,
    },
}

/// Result type for effect system operations
pub type EffectResult<T> = Result<T, EffectError>;

impl EffectSystem {
    pub fn new() -> Self {
        Self {
            effect_types: HashMap::new(),
            function_effects: HashMap::new(),
            inference_context: EffectInferenceContext::new(),
            effect_handlers: Vec::new(),
            current_effects: EffectSet::empty(),
            scope_stack: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Check effect constraints for a program
    pub fn check_program(&mut self, program: &TypedProgram) -> EffectResult<()> {
        self.enter_scope(EffectScopeKind::Function, Span::new(0, 0));

        for declaration in &program.declarations {
            self.check_declaration(&declaration.node)?;
        }

        self.exit_scope()?;

        // Solve remaining effect constraints
        self.solve_constraints()?;

        // Check for unhandled effects
        self.check_unhandled_effects()?;

        Ok(())
    }

    /// Check a declaration for effect constraints
    fn check_declaration(&mut self, decl: &TypedDeclaration) -> EffectResult<()> {
        match decl {
            TypedDeclaration::Function(func) => self.check_function(func),
            TypedDeclaration::Variable(var) => self.check_variable_declaration(var),
            _ => Ok(()), // Other declarations don't have effects
        }
    }

    /// Check a function for effect constraints
    fn check_function(&mut self, func: &TypedFunction) -> EffectResult<()> {
        let span = Span::new(0, 0); // TODO: get actual span from context
        self.enter_scope(EffectScopeKind::Function, span);

        // Get or infer function effect signature
        let effect_sig = self.get_or_infer_function_effects(func)?;

        // Check that function body satisfies effect signature
        let body_effects = if let Some(ref body) = func.body {
            self.infer_block_effects(body)?
        } else {
            EffectSet::empty() // Extern functions have no effects
        };
        self.check_effect_compatibility(&body_effects, &effect_sig.output_effects, span)?;

        self.exit_scope()?;
        Ok(())
    }

    /// Check a variable declaration
    fn check_variable_declaration(&mut self, var: &TypedVariable) -> EffectResult<()> {
        // If there's an initializer, check its effects
        if let Some(init) = &var.initializer {
            let init_effects = self.infer_expression_effects(&init.node)?;
            self.add_effects_to_context(init_effects);
        }

        Ok(())
    }

    /// Infer effects for a block
    fn infer_block_effects(&mut self, block: &TypedBlock) -> EffectResult<EffectSet> {
        let mut combined_effects = EffectSet::empty();
        for stmt_node in &block.statements {
            let stmt_effects = self.infer_statement_effects(&stmt_node.node)?;
            combined_effects = self.compose_effects(
                combined_effects,
                stmt_effects,
                CompositionOperator::Sequential,
            )?;
        }
        Ok(combined_effects)
    }

    /// Infer effects for a statement
    fn infer_statement_effects(&mut self, stmt: &TypedStatement) -> EffectResult<EffectSet> {
        match stmt {
            TypedStatement::Expression(expr) => self.infer_expression_effects(&expr.node),
            TypedStatement::Block(block) => self.infer_block_effects(block),
            TypedStatement::If(if_stmt) => {
                let cond_effects = self.infer_expression_effects(&if_stmt.condition.node)?;
                let then_effects = self.infer_block_effects(&if_stmt.then_block)?;

                let mut combined = self.compose_effects(
                    cond_effects,
                    then_effects,
                    CompositionOperator::Sequential,
                )?;

                if let Some(else_block) = &if_stmt.else_block {
                    let else_effects = self.infer_block_effects(else_block)?;
                    combined = self.compose_effects(
                        combined,
                        else_effects,
                        CompositionOperator::Alternative,
                    )?;
                }

                Ok(combined)
            }
            TypedStatement::While(while_stmt) => {
                let cond_effects = self.infer_expression_effects(&while_stmt.condition.node)?;
                let body_effects = self.infer_block_effects(&while_stmt.body)?;

                // Loops can execute zero or more times
                let loop_effects = self.amplify_effects(body_effects, EffectIntensity::Repeated)?;
                self.compose_effects(cond_effects, loop_effects, CompositionOperator::Sequential)
            }
            TypedStatement::Let(let_stmt) => {
                if let Some(init) = &let_stmt.initializer {
                    self.infer_expression_effects(&init.node)
                } else {
                    Ok(EffectSet::empty())
                }
            }
            TypedStatement::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    self.infer_expression_effects(&expr.node)
                } else {
                    Ok(EffectSet::empty())
                }
            }
            _ => {
                // Other statements not yet implemented
                Ok(EffectSet::empty())
            }
        }
    }

    /// Infer effects for an expression
    fn infer_expression_effects(&mut self, expr: &TypedExpression) -> EffectResult<EffectSet> {
        match expr {
            TypedExpression::Variable(_) => {
                // Variable access is generally pure, but could have effects in some languages
                Ok(EffectSet::empty())
            }
            TypedExpression::Call(call) => {
                // Function calls inherit effects from the function and arguments
                let callee_effects = self.infer_expression_effects(&call.callee.node)?;

                let mut arg_effects = EffectSet::empty();
                for arg in &call.positional_args {
                    let arg_effect = self.infer_expression_effects(&arg.node)?;
                    arg_effects = self.compose_effects(
                        arg_effects,
                        arg_effect,
                        CompositionOperator::Sequential,
                    )?;
                }

                // Get function effect signature if available
                let call_effects = self.get_call_effects()?;

                let combined = self.compose_effects(
                    callee_effects,
                    arg_effects,
                    CompositionOperator::Sequential,
                )?;
                self.compose_effects(combined, call_effects, CompositionOperator::Sequential)
            }
            TypedExpression::Field(field_access) => {
                // Field access inherits effects from object
                self.infer_expression_effects(&field_access.object.node)
            }
            TypedExpression::Binary(binary) => {
                let left_effects = self.infer_expression_effects(&binary.left.node)?;
                let right_effects = self.infer_expression_effects(&binary.right.node)?;
                self.compose_effects(left_effects, right_effects, CompositionOperator::Sequential)
            }
            TypedExpression::Unary(unary) => self.infer_expression_effects(&unary.operand.node),
            TypedExpression::If(if_expr) => {
                let cond_effects = self.infer_expression_effects(&if_expr.condition.node)?;
                let then_effects = self.infer_expression_effects(&if_expr.then_branch.node)?;

                let mut combined = self.compose_effects(
                    cond_effects,
                    then_effects,
                    CompositionOperator::Sequential,
                )?;

                let else_effects = self.infer_expression_effects(&if_expr.else_branch.node)?;
                combined =
                    self.compose_effects(combined, else_effects, CompositionOperator::Alternative)?;

                Ok(combined)
            }
            TypedExpression::Literal(_) => {
                // Literals are pure
                Ok(EffectSet::empty())
            }
            TypedExpression::Array(elements) => {
                let mut combined_effects = EffectSet::empty();
                for elem in elements {
                    let elem_effects = self.infer_expression_effects(&elem.node)?;
                    combined_effects = self.compose_effects(
                        combined_effects,
                        elem_effects,
                        CompositionOperator::Sequential,
                    )?;
                }

                // Array creation might have memory allocation effect
                let memory_effect = self.create_memory_effect();
                combined_effects.effects.insert(memory_effect);

                Ok(combined_effects)
            }
            TypedExpression::Index(index_expr) => {
                let object_effects = self.infer_expression_effects(&index_expr.object.node)?;
                let index_effects = self.infer_expression_effects(&index_expr.index.node)?;
                self.compose_effects(
                    object_effects,
                    index_effects,
                    CompositionOperator::Sequential,
                )
            }
            _ => {
                // Other expressions not yet implemented
                Ok(EffectSet::empty())
            }
        }
    }

    /// Enter a new effect scope
    pub fn enter_scope(&mut self, kind: EffectScopeKind, span: Span) {
        let scope = EffectScope {
            scope_kind: kind,
            available_effects: self.current_effects.clone(),
            handled_effects: EffectSet::empty(),
            required_effects: EffectSet::empty(),
            handlers: Vec::new(),
            span,
        };
        self.scope_stack.push(scope);
    }

    /// Exit the current effect scope
    pub fn exit_scope(&mut self) -> EffectResult<()> {
        if let Some(scope) = self.scope_stack.pop() {
            // Check that all required effects are handled or available
            for effect in &scope.required_effects.effects {
                if !scope.handled_effects.contains_effect(effect)
                    && !scope.available_effects.contains_effect(effect)
                {
                    return Err(EffectError::UnhandledEffect {
                        effect: effect.clone(),
                        span: scope.span,
                        available_handlers: scope.handlers.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Add effects to current context
    pub fn add_effects_to_context(&mut self, effects: EffectSet) {
        self.current_effects = self
            .compose_effects(
                self.current_effects.clone(),
                effects,
                CompositionOperator::Sequential,
            )
            .unwrap_or(self.current_effects.clone());
    }

    /// Get or infer function effect signature
    fn get_or_infer_function_effects(
        &mut self,
        func: &TypedFunction,
    ) -> EffectResult<EffectSignature> {
        if let Some(sig) = self.function_effects.get(&func.name) {
            Ok(sig.clone())
        } else {
            // Infer effects from function body
            let body_effects = if let Some(ref body) = func.body {
                self.infer_block_effects(body)?
            } else {
                EffectSet::empty() // Extern functions have no effects
            };
            let sig = EffectSignature {
                input_effects: EffectSet::empty(),
                output_effects: body_effects,
                transformations: Vec::new(),
                requirements: Vec::new(),
                is_pure: false, // Will be determined by effects
                bounds: Vec::new(),
            };
            self.function_effects.insert(func.name, sig.clone());
            Ok(sig)
        }
    }

    /// Get effects from a function call
    fn get_call_effects(&self) -> EffectResult<EffectSet> {
        // Would look up function signature and return its effects
        // For now, return empty effect set
        Ok(EffectSet::empty())
    }

    /// Check effect compatibility
    pub fn check_effect_compatibility(
        &self,
        actual: &EffectSet,
        expected: &EffectSet,
        span: Span,
    ) -> EffectResult<()> {
        if !self.is_effect_subtype(actual, expected) {
            return Err(EffectError::ConstraintViolation {
                constraint: EffectConstraint::SubEffect(actual.clone(), expected.clone(), span),
                violation_span: span,
                reason: "Effect signature mismatch".to_string(),
            });
        }
        Ok(())
    }

    /// Check if one effect set is a subtype of another
    pub fn is_effect_subtype(&self, sub: &EffectSet, sup: &EffectSet) -> bool {
        // A simplified subtyping check - all effects in sub must be in sup or handled
        for effect in &sub.effects {
            if !sup.contains_effect(effect) && !self.is_effect_handled(effect) {
                return false;
            }
        }
        true
    }

    /// Check if an effect is handled in the current context
    pub fn is_effect_handled(&self, effect: &Effect) -> bool {
        for handler in &self.effect_handlers {
            if handler.handled_effects.contains_effect(effect) {
                return true;
            }
        }
        false
    }

    /// Compose two effect sets
    pub fn compose_effects(
        &self,
        left: EffectSet,
        right: EffectSet,
        op: CompositionOperator,
    ) -> EffectResult<EffectSet> {
        match op {
            CompositionOperator::Sequential => {
                let mut result = left;
                for effect in right.effects {
                    result.effects.insert(effect);
                }
                for var in right.variables {
                    result.variables.insert(var);
                }
                result.unions.extend(right.unions);
                result.intersections.extend(right.intersections);
                Ok(result)
            }
            CompositionOperator::Parallel => {
                // Parallel composition might require special handling
                Ok(self.compose_effects(left, right, CompositionOperator::Sequential)?)
            }
            CompositionOperator::Alternative => {
                // Alternative composition creates a union
                Ok(EffectSet {
                    effects: BTreeSet::new(),
                    variables: BTreeSet::new(),
                    unions: vec![EffectUnion {
                        effects: vec![left, right],
                        span: Span::new(0, 0),
                    }],
                    intersections: Vec::new(),
                })
            }
            _ => {
                // Other operators not yet implemented
                Ok(left)
            }
        }
    }

    /// Amplify effect intensity
    pub fn amplify_effects(
        &self,
        effects: EffectSet,
        intensity: EffectIntensity,
    ) -> EffectResult<EffectSet> {
        let mut result = effects;
        let amplified_effects: BTreeSet<Effect> = result
            .effects
            .into_iter()
            .map(|mut effect| {
                effect.intensity = intensity;
                effect
            })
            .collect();
        result.effects = amplified_effects;
        Ok(result)
    }

    /// Create a state effect
    pub fn create_state_effect(&self) -> Effect {
        Effect {
            id: self.get_builtin_effect_id(EffectKind::State),
            region: None,
            intensity: EffectIntensity::Definite,
        }
    }

    /// Create a memory effect
    pub fn create_memory_effect(&self) -> Effect {
        Effect {
            id: self.get_builtin_effect_id(EffectKind::Memory),
            region: None,
            intensity: EffectIntensity::Definite,
        }
    }

    /// Get builtin effect ID for a kind
    fn get_builtin_effect_id(&self, kind: EffectKind) -> EffectId {
        // Would maintain a registry of builtin effects
        // For now, create a deterministic ID based on kind
        EffectId(kind as u32)
    }

    /// Solve effect constraints
    pub fn solve_constraints(&mut self) -> EffectResult<()> {
        // Implement constraint solving algorithm
        // This is a simplified version - real implementation would be more complex

        let mut changed = true;
        while changed {
            changed = false;

            for constraint in &self.inference_context.constraints.clone() {
                if self.try_solve_constraint(constraint)? {
                    changed = true;
                }
            }
        }

        Ok(())
    }

    /// Try to solve a single constraint
    fn try_solve_constraint(&mut self, constraint: &EffectConstraint) -> EffectResult<bool> {
        match constraint {
            EffectConstraint::Equal(left, right, _) => {
                // Try to unify effect sets
                self.unify_effect_sets(left, right)
            }
            EffectConstraint::SubEffect(sub, sup, _) => {
                // Check subtyping constraint
                Ok(self.is_effect_subtype(sub, sup))
            }
            _ => {
                // Other constraints not yet implemented
                Ok(false)
            }
        }
    }

    /// Unify two effect sets
    fn unify_effect_sets(&mut self, left: &EffectSet, right: &EffectSet) -> EffectResult<bool> {
        // Simplified unification - real implementation would be more sophisticated
        Ok(left == right)
    }

    /// Check for unhandled effects
    fn check_unhandled_effects(&self) -> EffectResult<()> {
        for effect in &self.current_effects.effects {
            if !self.is_effect_handled(effect) {
                return Err(EffectError::UnhandledEffect {
                    effect: effect.clone(),
                    span: Span::new(0, 0),
                    available_handlers: Vec::new(),
                });
            }
        }
        Ok(())
    }

    /// Add an effect type definition
    pub fn add_effect_type(&mut self, info: EffectTypeInfo) {
        self.effect_types.insert(info.id, info);
    }

    /// Add an effect handler
    pub fn add_effect_handler(&mut self, handler: EffectHandler) {
        self.effect_handlers.push(handler);
    }

    /// Get effect signature for a function
    pub fn get_function_signature(&self, name: &InternedString) -> Option<&EffectSignature> {
        self.function_effects.get(name)
    }

    /// Set effect signature for a function
    pub fn set_function_signature(&mut self, name: InternedString, signature: EffectSignature) {
        self.function_effects.insert(name, signature);
    }
}

impl EffectSet {
    /// Create an empty effect set
    pub fn empty() -> Self {
        Self {
            effects: BTreeSet::new(),
            variables: BTreeSet::new(),
            unions: Vec::new(),
            intersections: Vec::new(),
        }
    }

    /// Create a pure effect set (no effects)
    pub fn pure() -> Self {
        Self::empty()
    }

    /// Check if effect set contains a specific effect
    pub fn contains_effect(&self, effect: &Effect) -> bool {
        self.effects.contains(effect)
    }

    /// Check if effect set is pure (no effects)
    pub fn is_pure(&self) -> bool {
        self.effects.is_empty()
            && self.variables.is_empty()
            && self.unions.is_empty()
            && self.intersections.is_empty()
    }

    /// Add an effect to the set
    pub fn add_effect(&mut self, effect: Effect) {
        self.effects.insert(effect);
    }

    /// Add multiple effects
    pub fn add_effects(&mut self, effects: impl IntoIterator<Item = Effect>) {
        self.effects.extend(effects);
    }
}

impl EffectSignature {
    /// Create a pure function signature (no effects)
    pub fn pure() -> Self {
        Self {
            input_effects: EffectSet::pure(),
            output_effects: EffectSet::pure(),
            transformations: Vec::new(),
            requirements: Vec::new(),
            is_pure: true,
            bounds: Vec::new(),
        }
    }

    /// Create an I/O function signature
    pub fn io() -> Self {
        let mut sig = Self::pure();
        sig.output_effects.add_effect(Effect {
            id: EffectId(EffectKind::IO as u32),
            region: None,
            intensity: EffectIntensity::Definite,
        });
        sig.is_pure = false;
        sig
    }

    /// Create a state function signature
    pub fn state() -> Self {
        let mut sig = Self::pure();
        sig.output_effects.add_effect(Effect {
            id: EffectId(EffectKind::State as u32),
            region: None,
            intensity: EffectIntensity::Definite,
        });
        sig.is_pure = false;
        sig
    }
}

impl EffectInferenceContext {
    pub fn new() -> Self {
        Self {
            effect_vars: HashMap::new(),
            constraints: Vec::new(),
            substitution: EffectSubstitution::default(),
            options: EffectInferenceOptions::default(),
        }
    }
}

impl Default for EffectInferenceOptions {
    fn default() -> Self {
        Self {
            allow_effect_polymorphism: true,
            infer_purity: true,
            check_effect_safety: true,
            enable_effect_optimization: true,
            max_effect_depth: 10,
        }
    }
}

/// Helper functions for creating common effect types
impl EffectTypeInfo {
    /// Create I/O effect type
    pub fn io_effect() -> Self {
        Self {
            id: EffectId(EffectKind::IO as u32),
            name: InternedString::from_symbol(string_interner::Symbol::try_from_usize(1).unwrap()),
            effect_kind: EffectKind::IO,
            parameters: Vec::new(),
            constraints: Vec::new(),
            composition_rules: CompositionRules::default(),
            handler_requirements: HandlerRequirements::default(),
        }
    }

    /// Create state effect type
    pub fn state_effect() -> Self {
        Self {
            id: EffectId(EffectKind::State as u32),
            name: InternedString::from_symbol(string_interner::Symbol::try_from_usize(2).unwrap()),
            effect_kind: EffectKind::State,
            parameters: Vec::new(),
            constraints: Vec::new(),
            composition_rules: CompositionRules::default(),
            handler_requirements: HandlerRequirements::default(),
        }
    }

    /// Create exception effect type
    pub fn exception_effect() -> Self {
        Self {
            id: EffectId(EffectKind::Exception as u32),
            name: InternedString::from_symbol(string_interner::Symbol::try_from_usize(3).unwrap()),
            effect_kind: EffectKind::Exception,
            parameters: Vec::new(),
            constraints: Vec::new(),
            composition_rules: CompositionRules::default(),
            handler_requirements: HandlerRequirements::default(),
        }
    }
}

impl Default for CompositionRules {
    fn default() -> Self {
        Self {
            commutative: false,
            associative: true,
            idempotent: false,
            identity: None,
            absorbing: None,
            custom_rules: Vec::new(),
        }
    }
}

impl Default for HandlerRequirements {
    fn default() -> Self {
        Self {
            required_operations: Vec::new(),
            optional_operations: Vec::new(),
            handler_constraints: Vec::new(),
        }
    }
}
