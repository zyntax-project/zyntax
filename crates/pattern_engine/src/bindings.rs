use std::collections::HashMap;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedExpression, TypedNode, TypedStatement};
use zyntax_typed_ast::Type;

/// The result of a successful pattern match — a typed map from names to matched AST fragments.
#[derive(Debug, Clone, Default)]
pub struct Bindings {
    exprs: HashMap<&'static str, TypedNode<TypedExpression>>,
    stmts: HashMap<&'static str, TypedNode<TypedStatement>>,
    decls: HashMap<&'static str, TypedNode<TypedDeclaration>>,
    types: HashMap<&'static str, Type>,
    spans: HashMap<&'static str, Span>,
}

impl Bindings {
    pub fn new() -> Self {
        Self::default()
    }

    // --- Getters ---

    pub fn expr(&self, name: &'static str) -> Option<&TypedNode<TypedExpression>> {
        self.exprs.get(name)
    }

    pub fn stmt(&self, name: &'static str) -> Option<&TypedNode<TypedStatement>> {
        self.stmts.get(name)
    }

    pub fn decl(&self, name: &'static str) -> Option<&TypedNode<TypedDeclaration>> {
        self.decls.get(name)
    }

    pub fn ty(&self, name: &'static str) -> Option<&Type> {
        self.types.get(name)
    }

    pub fn span(&self, name: &'static str) -> Option<&Span> {
        self.spans.get(name)
    }

    // --- Setters ---

    pub fn bind_expr(&mut self, name: &'static str, node: TypedNode<TypedExpression>) {
        self.exprs.insert(name, node);
    }

    pub fn bind_stmt(&mut self, name: &'static str, node: TypedNode<TypedStatement>) {
        self.stmts.insert(name, node);
    }

    pub fn bind_decl(&mut self, name: &'static str, node: TypedNode<TypedDeclaration>) {
        self.decls.insert(name, node);
    }

    pub fn bind_type(&mut self, name: &'static str, ty: Type) {
        self.types.insert(name, ty);
    }

    pub fn bind_span(&mut self, name: &'static str, span: Span) {
        self.spans.insert(name, span);
    }

    /// Merge another Bindings into this one. Existing keys are overwritten.
    pub fn merge(&mut self, other: Bindings) {
        self.exprs.extend(other.exprs);
        self.stmts.extend(other.stmts);
        self.decls.extend(other.decls);
        self.types.extend(other.types);
        self.spans.extend(other.spans);
    }
}
