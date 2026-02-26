//! Linear and Affine Type Systems
//!
//! Implements linear and affine type systems for resource management and memory safety.
//! Linear types must be used exactly once, while affine types must be used at most once.
//!
//! Key features:
//! - Linear types: must be consumed exactly once (e.g., file handles)
//! - Affine types: must be consumed at most once (e.g., unique pointers)
//! - Substructural typing: types that violate structural rules
//! - Resource tracking: automatic resource lifecycle management
//! - Borrowing system: temporary access without ownership transfer

use crate::arena::InternedString;
use crate::source::Span;
use crate::type_registry::{Type, TypeId};
use crate::typed_ast::{
    TypedBlock, TypedDeclaration, TypedExpression, TypedFunction, TypedProgram, TypedStatement,
    TypedVariable,
};
use std::collections::{HashMap, HashSet, VecDeque};

/// Lifetime representation for linear type system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Lifetime(u32);

impl Lifetime {
    pub fn new(id: u32) -> Self {
        Lifetime(id)
    }

    pub fn static_lifetime() -> Self {
        Lifetime(0)
    }
}

/// Linear type system for resource management and memory safety
pub struct LinearTypeChecker {
    /// Variable usage tracking
    variable_usage: HashMap<InternedString, UsageInfo>,

    /// Resource tracking for linear resources
    resource_tracker: ResourceTracker,

    /// Borrowing system for temporary access
    borrow_checker: BorrowChecker,

    /// Linear type definitions
    linear_types: HashMap<TypeId, LinearTypeInfo>,

    /// Constraint context for linear constraints
    #[allow(dead_code)]
    linear_constraints: Vec<LinearConstraint>,

    /// Current scope for variable tracking
    scope_stack: Vec<ScopeInfo>,

    /// Error accumulator
    errors: Vec<LinearTypeError>,
}

/// Information about variable usage in linear context
#[derive(Debug, Clone, PartialEq)]
pub struct UsageInfo {
    /// How many times this variable has been used
    pub use_count: usize,

    /// Where the variable was last used
    pub last_use_span: Option<Span>,

    /// Whether the variable has been moved
    pub is_moved: bool,

    /// Whether the variable is currently borrowed
    pub is_borrowed: bool,

    /// The linearity kind of this variable
    pub linearity: LinearityKind,

    /// Active borrows of this variable
    pub active_borrows: Vec<BorrowId>,
}

/// Different kinds of linearity constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinearityKind {
    /// Must be used exactly once (linear)
    Linear,

    /// Must be used at most once (affine)
    Affine,

    /// Can be used multiple times (relevant)
    Relevant,

    /// Can be used zero or more times (unrestricted)
    Unrestricted,

    /// Cannot be copied but can be borrowed (unique)
    Unique,

    /// Read-only, can be shared but not mutated
    Shared,
}

/// Resource tracking for managing linear resources
#[derive(Debug, Clone)]
pub struct ResourceTracker {
    /// Active resources that need cleanup
    active_resources: HashMap<ResourceId, ResourceInfo>,

    /// Resource dependencies (which resources depend on others)
    #[allow(dead_code)]
    dependencies: HashMap<ResourceId, HashSet<ResourceId>>,

    /// Resource cleanup order
    #[allow(dead_code)]
    cleanup_order: VecDeque<ResourceId>,

    /// Resource allocation sites
    allocation_sites: HashMap<ResourceId, Span>,
}

/// Information about a tracked resource
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceInfo {
    pub id: ResourceId,
    pub resource_type: Type,
    pub linearity: LinearityKind,
    pub is_consumed: bool,
    pub cleanup_fn: Option<InternedString>,
    pub allocation_span: Span,
}

/// Borrow checking for temporary access to linear resources
#[derive(Debug, Clone)]
pub struct BorrowChecker {
    /// Active borrows
    active_borrows: HashMap<BorrowId, BorrowInfo>,

    /// Borrow relationships (what borrows what)
    #[allow(dead_code)]
    borrow_graph: HashMap<InternedString, Vec<BorrowId>>,

    /// Lifetime constraints
    #[allow(dead_code)]
    lifetime_constraints: Vec<LifetimeConstraint>,

    /// Current lifetime scope
    #[allow(dead_code)]
    lifetime_scope: Vec<LifetimeScope>,
}

/// Information about an active borrow
#[derive(Debug, Clone, PartialEq)]
pub struct BorrowInfo {
    pub id: BorrowId,
    pub borrowed_var: InternedString,
    pub borrow_kind: BorrowKind,
    pub lifetime: Option<Lifetime>,
    pub borrow_span: Span,
    pub is_mutable: bool,
}

/// Different kinds of borrows
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BorrowKind {
    /// Immutable borrow (shared reference)
    Shared,

    /// Mutable borrow (exclusive reference)
    Mutable,

    /// Move borrow (transfers ownership temporarily)
    Move,

    /// Weak borrow (doesn't extend lifetime)
    Weak,
}

/// Linear type information
#[derive(Debug, Clone, PartialEq)]
pub struct LinearTypeInfo {
    pub type_id: TypeId,
    pub linearity: LinearityKind,
    pub resource_kind: Option<ResourceKind>,
    pub cleanup_behavior: CleanupBehavior,
    pub borrowing_rules: BorrowingRules,
}

/// Different kinds of resources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// File handle or stream
    FileHandle,

    /// Network connection
    NetworkConnection,

    /// Memory allocation
    Memory,

    /// Database connection
    Database,

    /// Mutex or lock
    Lock,

    /// Custom resource type
    Custom,
}

/// How resources should be cleaned up
#[derive(Debug, Clone, PartialEq)]
pub enum CleanupBehavior {
    /// Automatic cleanup (RAII/drop)
    Automatic(Option<InternedString>),

    /// Manual cleanup required
    Manual(InternedString),

    /// No cleanup needed
    None,

    /// Custom cleanup function
    Custom(CleanupFunction),
}

/// Custom cleanup function specification
#[derive(Debug, Clone, PartialEq)]
pub struct CleanupFunction {
    pub function_name: InternedString,
    pub parameters: Vec<Type>,
    pub is_fallible: bool,
}

/// Borrowing rules for a linear type
#[derive(Debug, Clone, PartialEq)]
pub struct BorrowingRules {
    /// Can this type be borrowed immutably?
    pub allows_shared_borrow: bool,

    /// Can this type be borrowed mutably?
    pub allows_mutable_borrow: bool,

    /// Can multiple shared borrows exist simultaneously?
    pub allows_multiple_shared: bool,

    /// Maximum borrow depth
    pub max_borrow_depth: Option<usize>,

    /// Custom borrowing constraints
    pub custom_constraints: Vec<BorrowConstraint>,
}

/// Custom borrowing constraints
#[derive(Debug, Clone, PartialEq)]
pub enum BorrowConstraint {
    /// Borrow must outlive a specific lifetime
    OutlivesLifetime(Lifetime),

    /// Borrow cannot coexist with another borrow
    ExclusiveWith(BorrowKind),

    /// Borrow requires a specific condition
    RequiresCondition(InternedString),

    /// Custom predicate
    CustomPredicate(InternedString, Vec<Type>),
}

/// Linear constraints for type checking
#[derive(Debug, Clone, PartialEq)]
pub enum LinearConstraint {
    /// Variable must be used exactly once
    MustUseOnce(InternedString, Span),

    /// Variable must be used at most once
    MustUseAtMostOnce(InternedString, Span),

    /// Variable cannot be used after this point
    NoUseAfter(InternedString, Span),

    /// Variable must be consumed before end of scope
    MustConsumeBeforeEndOfScope(InternedString, Span),

    /// Borrow must not outlive the borrowed value
    BorrowMustNotOutlive(BorrowId, InternedString, Span),

    /// Resource must be cleaned up
    ResourceMustBeCleanedUp(ResourceId, Span),

    /// Two borrows cannot coexist
    BorrowsCannotCoexist(BorrowId, BorrowId, Span),

    /// Variable must be moved
    MustMove(InternedString, Span),

    /// Variable cannot be copied
    CannotCopy(InternedString, Span),
}

/// Scope information for tracking variable lifetimes
#[derive(Debug, Clone, PartialEq)]
pub struct ScopeInfo {
    /// Variables declared in this scope
    pub declared_vars: HashSet<InternedString>,

    /// Variables that must be consumed before scope ends
    pub must_consume: HashSet<InternedString>,

    /// Active borrows in this scope
    pub active_borrows: HashSet<BorrowId>,

    /// Scope kind
    pub scope_kind: ScopeKind,

    /// Span of the scope
    pub span: Span,
}

/// Different kinds of scopes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeKind {
    /// Function scope
    Function,

    /// Block scope
    Block,

    /// Loop scope
    Loop,

    /// Conditional scope
    Conditional,

    /// Match arm scope
    MatchArm,

    /// Async scope
    Async,
}

/// Lifetime constraints for borrow checking
#[derive(Debug, Clone, PartialEq)]
pub enum LifetimeConstraint {
    /// One lifetime must outlive another
    Outlives(Lifetime, Lifetime, Span),

    /// Lifetime must be at least as long as a scope
    OutlivesScope(Lifetime, ScopeKind, Span),

    /// Two lifetimes must be equal
    Equal(Lifetime, Lifetime, Span),

    /// Lifetime is bounded by another
    Bounded(Lifetime, Lifetime, Span),
}

/// Lifetime scope information
#[derive(Debug, Clone, PartialEq)]
pub struct LifetimeScope {
    pub lifetime: Lifetime,
    pub scope_kind: ScopeKind,
    pub span: Span,
    pub parent: Option<Lifetime>,
}

/// Unique identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BorrowId(u32);

impl ResourceId {
    pub fn next() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_ID: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl BorrowId {
    pub fn next() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_ID: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Linear type checking errors
#[derive(Debug, Clone, PartialEq)]
pub enum LinearTypeError {
    /// Variable used more than once (linear violation)
    UsedMoreThanOnce {
        var: InternedString,
        first_use: Span,
        second_use: Span,
        linearity: LinearityKind,
    },

    /// Variable not used (linear violation)
    NotUsed {
        var: InternedString,
        declaration: Span,
        linearity: LinearityKind,
    },

    /// Variable used after move
    UsedAfterMove {
        var: InternedString,
        move_span: Span,
        use_span: Span,
    },

    /// Borrow outlives borrowed value
    BorrowOutlivesValue {
        borrow_id: BorrowId,
        borrowed_var: InternedString,
        borrow_span: Span,
        end_of_value_span: Span,
    },

    /// Multiple mutable borrows
    MultipleMutableBorrows {
        var: InternedString,
        first_borrow: Span,
        second_borrow: Span,
    },

    /// Use while borrowed
    UseWhileBorrowed {
        var: InternedString,
        borrow_span: Span,
        use_span: Span,
        borrow_kind: BorrowKind,
    },

    /// Resource not cleaned up
    ResourceNotCleanedUp {
        resource_id: ResourceId,
        resource_type: Type,
        allocation_span: Span,
    },

    /// Cannot copy linear type
    CannotCopy {
        var: InternedString,
        copy_span: Span,
        linearity: LinearityKind,
    },

    /// Invalid borrow
    InvalidBorrow {
        var: InternedString,
        borrow_kind: BorrowKind,
        reason: String,
        span: Span,
    },

    /// Constraint violation
    ConstraintViolation {
        constraint: LinearConstraint,
        violation_span: Span,
        reason: String,
    },
}

/// Result type for linear type checking
pub type LinearTypeResult<T> = Result<T, LinearTypeError>;

impl LinearTypeChecker {
    pub fn new() -> Self {
        Self {
            variable_usage: HashMap::new(),
            resource_tracker: ResourceTracker::new(),
            borrow_checker: BorrowChecker::new(),
            linear_types: HashMap::new(),
            linear_constraints: Vec::new(),
            scope_stack: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Check linear type constraints for a program
    pub fn check_program(&mut self, program: &TypedProgram) -> LinearTypeResult<()> {
        self.enter_scope(ScopeKind::Function, Span::new(0, 0));

        for declaration in &program.declarations {
            self.check_declaration(&declaration.node)?;
        }

        self.exit_scope()?;

        // Check for unused linear variables
        self.check_unused_linear_variables()?;

        // Check resource cleanup
        self.check_resource_cleanup()?;

        Ok(())
    }

    /// Check a declaration for linear type constraints
    fn check_declaration(&mut self, decl: &TypedDeclaration) -> LinearTypeResult<()> {
        match decl {
            TypedDeclaration::Function(func) => self.check_function(func),
            TypedDeclaration::Variable(var) => self.check_variable_declaration(var),
            _ => Ok(()), // Other declarations don't have linear constraints
        }
    }

    /// Check a function for linear constraints
    fn check_function(&mut self, func: &TypedFunction) -> LinearTypeResult<()> {
        self.enter_scope(ScopeKind::Function, Span::new(0, 0)); // TODO: get actual span

        // Register parameters
        for param in &func.params {
            let linearity = self.get_type_linearity(&param.ty);
            self.register_variable(param.name, linearity, param.span);
        }

        // Check function body
        if let Some(ref body) = func.body {
            self.check_block(body)?;
        }

        self.exit_scope()?;
        Ok(())
    }

    /// Check a block of statements
    fn check_block(&mut self, block: &TypedBlock) -> LinearTypeResult<()> {
        self.enter_scope(ScopeKind::Block, block.span);
        for stmt_node in &block.statements {
            self.check_statement(&stmt_node.node)?;
        }
        self.exit_scope()?;
        Ok(())
    }

    /// Check a variable declaration
    fn check_variable_declaration(&mut self, var: &TypedVariable) -> LinearTypeResult<()> {
        let linearity = self.get_type_linearity(&var.ty);
        self.register_variable(var.name, linearity, Span::new(0, 0)); // TODO: get actual span

        // If there's an initializer, check it
        if let Some(init) = &var.initializer {
            self.check_expression(&init.node)?;
        }

        Ok(())
    }

    /// Check a statement for linear constraints
    fn check_statement(&mut self, stmt: &TypedStatement) -> LinearTypeResult<()> {
        match stmt {
            TypedStatement::Expression(expr) => self.check_expression(&expr.node),
            TypedStatement::Block(block) => self.check_block(block),
            TypedStatement::If(if_stmt) => {
                self.check_expression(&if_stmt.condition.node)?;

                self.enter_scope(ScopeKind::Conditional, if_stmt.span);
                self.check_block(&if_stmt.then_block)?;
                self.exit_scope()?;

                if let Some(else_block) = &if_stmt.else_block {
                    self.enter_scope(ScopeKind::Conditional, if_stmt.span);
                    self.check_block(else_block)?;
                    self.exit_scope()?;
                }

                Ok(())
            }
            TypedStatement::While(while_stmt) => {
                self.check_expression(&while_stmt.condition.node)?;

                self.enter_scope(ScopeKind::Loop, while_stmt.span);
                self.check_block(&while_stmt.body)?;
                self.exit_scope()
            }
            TypedStatement::Let(let_stmt) => {
                let linearity = self.get_type_linearity(&let_stmt.ty);
                self.register_variable(let_stmt.name, linearity, let_stmt.span);

                if let Some(init) = &let_stmt.initializer {
                    self.check_expression(&init.node)?;
                }

                Ok(())
            }
            TypedStatement::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    self.check_expression(&expr.node)?;
                }
                Ok(())
            }
            _ => {
                // Other statement types don't have linear constraints
                Ok(())
            }
        }
    }

    /// Check an expression for linear constraints
    fn check_expression(&mut self, expr: &TypedExpression) -> LinearTypeResult<()> {
        match expr {
            TypedExpression::Variable(var) => {
                self.use_variable(*var, Span::new(0, 0)) // TODO: get actual span
            }
            TypedExpression::Call(call) => {
                self.check_expression(&call.callee.node)?;
                for arg in &call.positional_args {
                    self.check_expression(&arg.node)?;
                }
                Ok(())
            }
            TypedExpression::Field(field_access) => {
                self.check_expression(&field_access.object.node)
            }
            TypedExpression::Binary(binary) => {
                self.check_expression(&binary.left.node)?;
                self.check_expression(&binary.right.node)
            }
            TypedExpression::Unary(unary) => self.check_expression(&unary.operand.node),
            TypedExpression::If(if_expr) => {
                self.check_expression(&if_expr.condition.node)?;
                self.check_expression(&if_expr.then_branch.node)?;
                self.check_expression(&if_expr.else_branch.node)
            }
            TypedExpression::Literal(_) => Ok(()),
            TypedExpression::Array(elements) => {
                for elem in elements {
                    self.check_expression(&elem.node)?;
                }
                Ok(())
            }
            TypedExpression::Index(index_expr) => {
                self.check_expression(&index_expr.object.node)?;
                self.check_expression(&index_expr.index.node)
            }
            _ => {
                // Other expression types don't have linear constraints
                Ok(())
            }
        }
    }

    /// Register a new variable with linearity information
    fn register_variable(&mut self, name: InternedString, linearity: LinearityKind, _span: Span) {
        let usage_info = UsageInfo {
            use_count: 0,
            last_use_span: None,
            is_moved: false,
            is_borrowed: false,
            linearity,
            active_borrows: Vec::new(),
        };

        self.variable_usage.insert(name, usage_info);

        // Add to current scope
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.declared_vars.insert(name);

            // If it's a linear type, it must be consumed
            if matches!(linearity, LinearityKind::Linear | LinearityKind::Affine) {
                scope.must_consume.insert(name);
            }
        }
    }

    /// Record usage of a variable
    fn use_variable(&mut self, name: InternedString, span: Span) -> LinearTypeResult<()> {
        if let Some(usage) = self.variable_usage.get_mut(&name) {
            usage.use_count += 1;
            usage.last_use_span = Some(span);

            // Check linear constraints
            match usage.linearity {
                LinearityKind::Linear => {
                    if usage.use_count > 1 {
                        return Err(LinearTypeError::UsedMoreThanOnce {
                            var: name,
                            first_use: usage.last_use_span.unwrap_or(span),
                            second_use: span,
                            linearity: LinearityKind::Linear,
                        });
                    }
                }
                LinearityKind::Affine => {
                    if usage.use_count > 1 {
                        return Err(LinearTypeError::UsedMoreThanOnce {
                            var: name,
                            first_use: usage.last_use_span.unwrap_or(span),
                            second_use: span,
                            linearity: LinearityKind::Affine,
                        });
                    }
                }
                _ => {} // Other linearities allow multiple uses
            }

            // Check if used after move
            if usage.is_moved {
                return Err(LinearTypeError::UsedAfterMove {
                    var: name,
                    move_span: usage.last_use_span.unwrap_or(span),
                    use_span: span,
                });
            }

            // Check if used while borrowed
            if !usage.active_borrows.is_empty() {
                for &borrow_id in &usage.active_borrows {
                    if let Some(borrow_info) = self.borrow_checker.active_borrows.get(&borrow_id) {
                        if borrow_info.borrow_kind == BorrowKind::Mutable {
                            return Err(LinearTypeError::UseWhileBorrowed {
                                var: name,
                                borrow_span: borrow_info.borrow_span,
                                use_span: span,
                                borrow_kind: borrow_info.borrow_kind,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Enter a new scope
    fn enter_scope(&mut self, kind: ScopeKind, span: Span) {
        let scope = ScopeInfo {
            declared_vars: HashSet::new(),
            must_consume: HashSet::new(),
            active_borrows: HashSet::new(),
            scope_kind: kind,
            span,
        };
        self.scope_stack.push(scope);
    }

    /// Exit the current scope and check constraints
    fn exit_scope(&mut self) -> LinearTypeResult<()> {
        if let Some(scope) = self.scope_stack.pop() {
            // Check that all must-consume variables were consumed
            for var in scope.must_consume {
                if let Some(usage) = self.variable_usage.get(&var) {
                    if usage.use_count == 0 {
                        self.errors.push(LinearTypeError::NotUsed {
                            var,
                            declaration: scope.span,
                            linearity: usage.linearity,
                        });
                    }
                }
            }

            // Clean up variables declared in this scope
            for var in scope.declared_vars {
                self.variable_usage.remove(&var);
            }
        }

        Ok(())
    }

    /// Check for unused linear variables
    fn check_unused_linear_variables(&self) -> LinearTypeResult<()> {
        for (var, usage) in &self.variable_usage {
            if matches!(usage.linearity, LinearityKind::Linear) && usage.use_count == 0 {
                return Err(LinearTypeError::NotUsed {
                    var: *var,
                    declaration: Span::new(0, 0), // Would need better span tracking
                    linearity: usage.linearity,
                });
            }
        }
        Ok(())
    }

    /// Check that all resources are properly cleaned up
    fn check_resource_cleanup(&self) -> LinearTypeResult<()> {
        for (resource_id, resource) in &self.resource_tracker.active_resources {
            if !resource.is_consumed {
                return Err(LinearTypeError::ResourceNotCleanedUp {
                    resource_id: *resource_id,
                    resource_type: resource.resource_type.clone(),
                    allocation_span: resource.allocation_span,
                });
            }
        }
        Ok(())
    }

    /// Get the linearity kind for a type
    fn get_type_linearity(&self, ty: &Type) -> LinearityKind {
        match ty {
            Type::Named { id, .. } => {
                if let Some(info) = self.linear_types.get(id) {
                    info.linearity
                } else {
                    LinearityKind::Unrestricted
                }
            }
            // Function types are generally unrestricted
            Type::Function { .. } => LinearityKind::Unrestricted,
            // Primitives are unrestricted
            Type::Primitive(_) => LinearityKind::Unrestricted,
            // Other types default to unrestricted
            _ => LinearityKind::Unrestricted,
        }
    }

    /// Add a linear type definition
    pub fn add_linear_type(&mut self, type_id: TypeId, info: LinearTypeInfo) {
        self.linear_types.insert(type_id, info);
    }

    /// Create a borrow of a variable
    pub fn create_borrow(
        &mut self,
        var: InternedString,
        kind: BorrowKind,
        is_mutable: bool,
        span: Span,
    ) -> LinearTypeResult<BorrowId> {
        let borrow_id = BorrowId::next();

        // Check borrowing rules
        if let Some(usage) = self.variable_usage.get_mut(&var) {
            // Check if variable is already moved
            if usage.is_moved {
                return Err(LinearTypeError::InvalidBorrow {
                    var,
                    borrow_kind: kind,
                    reason: "Cannot borrow moved value".to_string(),
                    span,
                });
            }

            // Check for conflicting borrows
            if is_mutable && !usage.active_borrows.is_empty() {
                return Err(LinearTypeError::MultipleMutableBorrows {
                    var,
                    first_borrow: span, // Would need better tracking
                    second_borrow: span,
                });
            }

            // Create borrow info
            let borrow_info = BorrowInfo {
                id: borrow_id,
                borrowed_var: var,
                borrow_kind: kind,
                lifetime: None, // Would be inferred
                borrow_span: span,
                is_mutable,
            };

            // Register the borrow
            self.borrow_checker
                .active_borrows
                .insert(borrow_id, borrow_info);
            usage.active_borrows.push(borrow_id);
            usage.is_borrowed = true;

            Ok(borrow_id)
        } else {
            Err(LinearTypeError::InvalidBorrow {
                var,
                borrow_kind: kind,
                reason: "Variable not found".to_string(),
                span,
            })
        }
    }

    /// End a borrow
    pub fn end_borrow(&mut self, borrow_id: BorrowId) -> LinearTypeResult<()> {
        if let Some(borrow_info) = self.borrow_checker.active_borrows.remove(&borrow_id) {
            if let Some(usage) = self.variable_usage.get_mut(&borrow_info.borrowed_var) {
                usage.active_borrows.retain(|&id| id != borrow_id);
                if usage.active_borrows.is_empty() {
                    usage.is_borrowed = false;
                }
            }
        }
        Ok(())
    }
}

impl ResourceTracker {
    pub fn new() -> Self {
        Self {
            active_resources: HashMap::new(),
            dependencies: HashMap::new(),
            cleanup_order: VecDeque::new(),
            allocation_sites: HashMap::new(),
        }
    }

    /// Track a new resource
    pub fn track_resource(&mut self, info: ResourceInfo) {
        self.allocation_sites.insert(info.id, info.allocation_span);
        self.active_resources.insert(info.id, info);
    }

    /// Mark a resource as consumed
    pub fn consume_resource(&mut self, id: ResourceId) {
        if let Some(resource) = self.active_resources.get_mut(&id) {
            resource.is_consumed = true;
        }
    }
}

impl BorrowChecker {
    pub fn new() -> Self {
        Self {
            active_borrows: HashMap::new(),
            borrow_graph: HashMap::new(),
            lifetime_constraints: Vec::new(),
            lifetime_scope: Vec::new(),
        }
    }
}

/// Helper functions for creating common linear types
impl LinearTypeInfo {
    /// Create a linear file handle type
    pub fn file_handle(type_id: TypeId) -> Self {
        Self {
            type_id,
            linearity: LinearityKind::Linear,
            resource_kind: Some(ResourceKind::FileHandle),
            cleanup_behavior: CleanupBehavior::Automatic(Some(InternedString::from_symbol(
                string_interner::Symbol::try_from_usize(1).unwrap(),
            ))),
            borrowing_rules: BorrowingRules {
                allows_shared_borrow: true,
                allows_mutable_borrow: true,
                allows_multiple_shared: false,
                max_borrow_depth: Some(1),
                custom_constraints: Vec::new(),
            },
        }
    }

    /// Create an affine unique pointer type
    pub fn unique_pointer(type_id: TypeId) -> Self {
        Self {
            type_id,
            linearity: LinearityKind::Affine,
            resource_kind: Some(ResourceKind::Memory),
            cleanup_behavior: CleanupBehavior::Automatic(None),
            borrowing_rules: BorrowingRules {
                allows_shared_borrow: true,
                allows_mutable_borrow: true,
                allows_multiple_shared: true,
                max_borrow_depth: None,
                custom_constraints: Vec::new(),
            },
        }
    }

    /// Create a shared reference type
    pub fn shared_reference(type_id: TypeId) -> Self {
        Self {
            type_id,
            linearity: LinearityKind::Shared,
            resource_kind: None,
            cleanup_behavior: CleanupBehavior::None,
            borrowing_rules: BorrowingRules {
                allows_shared_borrow: true,
                allows_mutable_borrow: false,
                allows_multiple_shared: true,
                max_borrow_depth: None,
                custom_constraints: Vec::new(),
            },
        }
    }
}
