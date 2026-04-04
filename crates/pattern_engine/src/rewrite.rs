use crate::bindings::Bindings;
use crate::pattern::Pattern;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedExpression, TypedNode, TypedStatement};
use zyntax_typed_ast::TypedASTBuilder;

/// Unique rewrite identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RewriteId(pub u32);

static NEXT_REWRITE_ID: AtomicU32 = AtomicU32::new(1);

impl RewriteId {
    pub fn fresh() -> Self {
        RewriteId(NEXT_REWRITE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// What a rewrite rule produces.
pub enum RewriteOutput {
    /// Replace the matched expression with a new one.
    ReplaceExpr(TypedNode<TypedExpression>),

    /// Replace the matched statement with a new one.
    ReplaceStmt(TypedNode<TypedStatement>),

    /// Replace the matched declaration with a new one.
    ReplaceDecl(TypedNode<TypedDeclaration>),

    /// Replace with multiple declarations at program scope plus an optional
    /// replacement at the match site.
    Expand {
        declarations: Vec<TypedNode<TypedDeclaration>>,
        replacement: Option<TypedNode<TypedDeclaration>>,
    },

    /// Remove the matched node (only valid for statements/declarations).
    Delete,

    /// Pattern matched but rewrite determined no action needed.
    Unchanged,
}

/// Priority level controlling when a rewrite fires relative to others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub u32);

impl Priority {
    pub const NORMALIZATION: Priority = Priority(100);
    pub const SEMANTIC: Priority = Priority(200);
    pub const OPTIMIZATION: Priority = Priority(300);
    pub const TARGET: Priority = Priority(400);
}

/// Metadata about a rewrite's impact.
#[derive(Debug, Clone)]
pub struct RewriteBenefit {
    /// Whether this rewrite is required for correctness (not optional).
    pub required: bool,
    /// Estimated code size delta (negative = shrinks).
    pub size_delta: i32,
    /// Estimated performance impact (positive = faster).
    pub perf_delta: i32,
}

impl Default for RewriteBenefit {
    fn default() -> Self {
        Self {
            required: false,
            size_delta: 0,
            perf_delta: 0,
        }
    }
}

// --- Concrete rewrite types for each AST node kind ---

pub struct ExprRewrite {
    pub id: RewriteId,
    pub name: &'static str,
    pub pattern: Pattern<TypedExpression>,
    pub apply: Arc<dyn Fn(Bindings, &mut TypedASTBuilder) -> RewriteOutput + Send + Sync>,
    pub priority: Priority,
    pub benefit: RewriteBenefit,
}

pub struct StmtRewrite {
    pub id: RewriteId,
    pub name: &'static str,
    pub pattern: Pattern<TypedStatement>,
    pub apply: Arc<dyn Fn(Bindings, &mut TypedASTBuilder) -> RewriteOutput + Send + Sync>,
    pub priority: Priority,
    pub benefit: RewriteBenefit,
}

pub struct DeclRewrite {
    pub id: RewriteId,
    pub name: &'static str,
    pub pattern: Pattern<TypedDeclaration>,
    pub apply: Arc<dyn Fn(Bindings, &mut TypedASTBuilder) -> RewriteOutput + Send + Sync>,
    pub priority: Priority,
    pub benefit: RewriteBenefit,
}

impl ExprRewrite {
    pub fn new(
        name: &'static str,
        priority: Priority,
        pattern: Pattern<TypedExpression>,
        apply: impl Fn(Bindings, &mut TypedASTBuilder) -> RewriteOutput + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: RewriteId::fresh(),
            name,
            pattern,
            apply: Arc::new(apply),
            priority,
            benefit: RewriteBenefit::default(),
        }
    }

    pub fn with_benefit(mut self, benefit: RewriteBenefit) -> Self {
        self.benefit = benefit;
        self
    }
}

impl StmtRewrite {
    pub fn new(
        name: &'static str,
        priority: Priority,
        pattern: Pattern<TypedStatement>,
        apply: impl Fn(Bindings, &mut TypedASTBuilder) -> RewriteOutput + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: RewriteId::fresh(),
            name,
            pattern,
            apply: Arc::new(apply),
            priority,
            benefit: RewriteBenefit::default(),
        }
    }

    pub fn with_benefit(mut self, benefit: RewriteBenefit) -> Self {
        self.benefit = benefit;
        self
    }
}

impl DeclRewrite {
    pub fn new(
        name: &'static str,
        priority: Priority,
        pattern: Pattern<TypedDeclaration>,
        apply: impl Fn(Bindings, &mut TypedASTBuilder) -> RewriteOutput + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: RewriteId::fresh(),
            name,
            pattern,
            apply: Arc::new(apply),
            priority,
            benefit: RewriteBenefit::default(),
        }
    }

    pub fn with_benefit(mut self, benefit: RewriteBenefit) -> Self {
        self.benefit = benefit;
        self
    }
}
