//! A small functional layer over the typed-AST node constructors for
//! writing library functions in Rust.
//!
//! Expressions are plain values: names are globally interned and every
//! node carries the same empty span, so nothing needs a builder handle
//! and expressions compose as arguments. A [`Local`] pairs a name with
//! its type so reads and writes of a variable agree.

use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{
    typed_node, TypedBinary, TypedBlock, TypedCall, TypedCast, TypedDeclaration, TypedExpression,
    TypedFunction, TypedIf, TypedIfExpr, TypedIndex, TypedLet, TypedLiteral, TypedMethodCall,
    TypedParameter, TypedStatement, TypedUnary, TypedWhile,
};
use zyntax_typed_ast::{
    BinaryOp, InternedString, Mutability, ParamOwnership, PrimitiveType, Type, TypedNode, UnaryOp,
    Visibility,
};

pub type Expr = TypedNode<TypedExpression>;
pub type Stmt = TypedNode<TypedStatement>;
pub type Decl = TypedNode<TypedDeclaration>;

pub const SPAN: Span = Span::new(0, 0);

/// The module every declaration here is attributed to. A frontend's own
/// declarations carry none, which is how lowering and codegen tell what
/// a program wrote from what the library brought and build the latter
/// only where the former reaches it.
pub const MODULE: &str = "builtins";

pub fn i64() -> Type {
    Type::Primitive(PrimitiveType::I64)
}
pub fn i32() -> Type {
    Type::Primitive(PrimitiveType::I32)
}
pub fn f64() -> Type {
    Type::Primitive(PrimitiveType::F64)
}
/// An address as a number, sized by the target.
pub fn usize() -> Type {
    Type::Primitive(PrimitiveType::USize)
}
pub fn boolean() -> Type {
    Type::Primitive(PrimitiveType::Bool)
}
pub fn string() -> Type {
    Type::Primitive(PrimitiveType::String)
}
pub fn unit() -> Type {
    Type::Primitive(PrimitiveType::Unit)
}
pub fn any() -> Type {
    Type::Any
}

pub fn intern(s: &str) -> InternedString {
    InternedString::new_global(s)
}

fn node(e: TypedExpression, ty: Type) -> Expr {
    typed_node(e, ty, SPAN)
}

// ─── literals ───────────────────────────────────────────────────────

pub fn int(v: i64) -> Expr {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        i64(),
    )
}
pub fn int32(v: i32) -> Expr {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        i32(),
    )
}
pub fn float(v: f64) -> Expr {
    node(TypedExpression::Literal(TypedLiteral::Float(v)), f64())
}
pub fn text(s: &str) -> Expr {
    node(
        TypedExpression::Literal(TypedLiteral::String(intern(s))),
        string(),
    )
}
pub fn bool(v: bool) -> Expr {
    node(TypedExpression::Literal(TypedLiteral::Bool(v)), boolean())
}

// ─── locals ─────────────────────────────────────────────────────────

/// A named variable and its type.
#[derive(Clone)]
pub struct Local {
    pub name: &'static str,
    pub ty: Type,
    /// As a parameter: whether the function keeps what it is passed, so
    /// the caller must not release it afterwards.
    pub owned: bool,
}

pub fn local(name: &'static str, ty: Type) -> Local {
    Local {
        name,
        ty,
        owned: false,
    }
}

/// A parameter the function stores somewhere that outlives the call.
pub fn owned(name: &'static str, ty: Type) -> Local {
    Local {
        name,
        ty,
        owned: true,
    }
}

impl Local {
    /// Read the variable.
    pub fn e(&self) -> Expr {
        node(
            TypedExpression::Variable(intern(self.name)),
            self.ty.clone(),
        )
    }

    /// Declare it with an initial value.
    pub fn decl(&self, init: Expr) -> Stmt {
        typed_node(
            TypedStatement::Let(TypedLet {
                name: intern(self.name),
                ty: self.ty.clone(),
                mutability: Mutability::Mutable,
                initializer: Some(Box::new(init)),
                span: SPAN,
            }),
            Type::Unknown,
            SPAN,
        )
    }

    /// Assign to it.
    pub fn set(&self, value: Expr) -> Stmt {
        let assign = node(
            TypedExpression::Binary(TypedBinary {
                op: BinaryOp::Assign,
                left: Box::new(self.e()),
                right: Box::new(value),
            }),
            self.ty.clone(),
        );
        expr(assign)
    }

    /// `x = x + by`.
    pub fn add_assign(&self, by: Expr) -> Stmt {
        self.set(add(self.e(), by))
    }

    pub fn param(&self) -> TypedParameter {
        let mut p = TypedParameter::regular(
            intern(self.name),
            self.ty.clone(),
            Mutability::Mutable,
            SPAN,
        );
        if self.owned {
            p.ownership = ParamOwnership::Owned;
        }
        p
    }
}

// ─── expressions ────────────────────────────────────────────────────

pub fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    let ty = match op {
        BinaryOp::Eq
        | BinaryOp::Ne
        | BinaryOp::Lt
        | BinaryOp::Le
        | BinaryOp::Gt
        | BinaryOp::Ge
        | BinaryOp::And
        | BinaryOp::Or => boolean(),
        _ => l.ty.clone(),
    };
    node(
        TypedExpression::Binary(TypedBinary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }),
        ty,
    )
}
pub fn add(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Add, l, r)
}
pub fn sub(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Sub, l, r)
}
pub fn mul(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Mul, l, r)
}
pub fn div(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Div, l, r)
}
pub fn rem(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Rem, l, r)
}
pub fn eq(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Eq, l, r)
}
pub fn ne(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Ne, l, r)
}
pub fn lt(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Lt, l, r)
}
pub fn le(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Le, l, r)
}
pub fn gt(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Gt, l, r)
}
pub fn ge(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Ge, l, r)
}
pub fn and(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::And, l, r)
}
pub fn or(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Or, l, r)
}
pub fn bitand(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::BitAnd, l, r)
}
pub fn bitor(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::BitOr, l, r)
}
pub fn bitxor(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::BitXor, l, r)
}
pub fn shl(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Shl, l, r)
}
pub fn shr(l: Expr, r: Expr) -> Expr {
    bin(BinaryOp::Shr, l, r)
}
pub fn not(e: Expr) -> Expr {
    node(
        TypedExpression::Unary(TypedUnary {
            op: UnaryOp::Not,
            operand: Box::new(e),
        }),
        boolean(),
    )
}
pub fn neg(e: Expr) -> Expr {
    let ty = e.ty.clone();
    node(
        TypedExpression::Unary(TypedUnary {
            op: UnaryOp::Minus,
            operand: Box::new(e),
        }),
        ty,
    )
}

/// A call to a function by name.
pub fn call(name: &str, args: Vec<Expr>, ret: Type) -> Expr {
    node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(node(TypedExpression::Variable(intern(name)), Type::Unknown)),
            positional_args: args,
            named_args: Vec::new(),
            type_args: Vec::new(),
        }),
        ret,
    )
}

pub fn mcall(recv: Expr, method: &str, args: Vec<Expr>, ret: Type) -> Expr {
    node(
        TypedExpression::MethodCall(TypedMethodCall {
            receiver: Box::new(recv),
            method: intern(method),
            type_args: Vec::new(),
            positional_args: args,
            named_args: Vec::new(),
        }),
        ret,
    )
}

pub fn idx(xs: Expr, i: Expr, elem: Type) -> Expr {
    node(
        TypedExpression::Index(TypedIndex {
            object: Box::new(xs),
            index: Box::new(i),
        }),
        elem,
    )
}

pub fn cast(e: Expr, ty: Type) -> Expr {
    node(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(e),
            target_type: ty.clone(),
        }),
        ty,
    )
}

pub fn if_expr(c: Expr, t: Expr, e: Expr) -> Expr {
    let ty = t.ty.clone();
    node(
        TypedExpression::If(TypedIfExpr {
            condition: Box::new(c),
            then_branch: Box::new(t),
            else_branch: Box::new(e),
        }),
        ty,
    )
}

/// The null of a pointer type.
pub fn null(ty: Type) -> Expr {
    node(TypedExpression::Literal(TypedLiteral::Null), ty)
}

/// A list literal of the given list type.
pub fn list(items: Vec<Expr>, list_ty: Type) -> Expr {
    node(TypedExpression::Array(items), list_ty)
}

// ─── statements ─────────────────────────────────────────────────────

pub fn expr(e: Expr) -> Stmt {
    typed_node(TypedStatement::Expression(Box::new(e)), Type::Unknown, SPAN)
}

pub fn set_idx(xs: Expr, i: Expr, value: Expr) -> Stmt {
    let elem = value.ty.clone();
    let target = idx(xs, i, elem.clone());
    expr(node(
        TypedExpression::Binary(TypedBinary {
            op: BinaryOp::Assign,
            left: Box::new(target),
            right: Box::new(value),
        }),
        elem,
    ))
}

pub fn ret(e: Expr) -> Stmt {
    typed_node(
        TypedStatement::Return(Some(Box::new(e))),
        Type::Unknown,
        SPAN,
    )
}

pub fn ret_void() -> Stmt {
    typed_node(TypedStatement::Return(None), Type::Unknown, SPAN)
}

pub fn block(stmts: Vec<Stmt>) -> TypedBlock {
    TypedBlock {
        statements: stmts,
        span: SPAN,
    }
}

pub fn if_(c: Expr, then: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
    typed_node(
        TypedStatement::If(TypedIf {
            condition: Box::new(c),
            then_block: block(then),
            else_block: if els.is_empty() {
                None
            } else {
                Some(block(els))
            },
            span: SPAN,
        }),
        Type::Unknown,
        SPAN,
    )
}

pub fn when(c: Expr, then: Vec<Stmt>) -> Stmt {
    if_(c, then, Vec::new())
}

pub fn while_(c: Expr, body: Vec<Stmt>) -> Stmt {
    typed_node(
        TypedStatement::While(TypedWhile {
            condition: Box::new(c),
            body: block(body),
            span: SPAN,
        }),
        Type::Unknown,
        SPAN,
    )
}

/// `let i = from; while i < to { body; i = i + 1 }`.
pub fn for_range(i: &Local, from: Expr, to: Expr, mut body: Vec<Stmt>) -> Vec<Stmt> {
    body.push(i.add_assign(int(1)));
    vec![i.decl(from), while_(lt(i.e(), to), body)]
}

/// `fatal(kind, message)`: report the error, and leave the function
/// with a placeholder result once [`define`] has seen the return type.
/// A frontend that turns errors into exceptions gets control back
/// from `zb_fatal`, so nothing after it may run.
pub fn fatal(kind: &str, message: Expr) -> Stmt {
    expr(call("zb_fatal", vec![text(kind), message], unit()))
}

fn is_fatal(stmt: &Stmt) -> bool {
    match &stmt.node {
        TypedStatement::Expression(e) => match &e.node {
            TypedExpression::Call(c) => matches!(
                &c.callee.node,
                TypedExpression::Variable(n) if n.resolve_global().as_deref() == Some("zb_fatal")
            ),
            _ => false,
        },
        _ => false,
    }
}

/// The value a function hands back when it has reported an error.
fn placeholder(ret_ty: &Type) -> Option<Expr> {
    Some(match ret_ty {
        Type::Primitive(PrimitiveType::Unit) => return None,
        Type::Primitive(PrimitiveType::Bool) => bool(false),
        Type::Primitive(PrimitiveType::I32) => int32(0),
        Type::Primitive(PrimitiveType::F64) => float(0.0),
        Type::Primitive(PrimitiveType::String) => text(""),
        Type::Primitive(_) => int(0),
        // Boxes, lists and code addresses: a null of the type.
        other => node(TypedExpression::Literal(TypedLiteral::Null), other.clone()),
    })
}

fn leave_after_fatal(stmts: Vec<Stmt>, ret_ty: &Type) -> Vec<Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    let mut iter = stmts.into_iter().peekable();
    while let Some(mut stmt) = iter.next() {
        match &mut stmt.node {
            TypedStatement::If(i) => {
                let then = std::mem::take(&mut i.then_block.statements);
                i.then_block.statements = leave_after_fatal(then, ret_ty);
                if let Some(els) = &mut i.else_block {
                    let e = std::mem::take(&mut els.statements);
                    els.statements = leave_after_fatal(e, ret_ty);
                }
            }
            TypedStatement::While(w) => {
                let body = std::mem::take(&mut w.body.statements);
                w.body.statements = leave_after_fatal(body, ret_ty);
            }
            _ => {}
        }
        let fatal_here = is_fatal(&stmt);
        out.push(stmt);
        if fatal_here
            && !matches!(
                iter.peek().map(|s| &s.node),
                Some(TypedStatement::Return(_))
            )
        {
            out.push(match placeholder(ret_ty) {
                Some(value) => ret(value),
                None => ret_void(),
            });
        }
    }
    out
}

// ─── declarations ───────────────────────────────────────────────────

/// A function with a body.
pub fn define(name: &str, params: &[&Local], ret_ty: Type, body: Vec<Stmt>) -> Decl {
    let body = leave_after_fatal(body, &ret_ty);
    typed_node(
        TypedDeclaration::Function(TypedFunction {
            name: intern(name),
            annotations: Vec::new(),
            effects: Vec::new(),
            with_handlers: Vec::new(),
            type_params: Vec::new(),
            params: params.iter().map(|p| p.param()).collect(),
            return_type: ret_ty.clone(),
            body: Some(block(body)),
            visibility: Visibility::Public,
            is_async: false,
            is_fiber: false,
            is_pure: false,
            is_external: false,
            calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
            link_name: None,
            module: Some(intern(MODULE)),
        }),
        ret_ty,
        SPAN,
    )
}

/// An extern bound to a runtime symbol or, when `symbol` is `None`, to
/// an intrinsic the compiler resolves by the extern's own name.
pub fn extern_fn(name: &str, params: &[(&str, Type)], ret_ty: Type, symbol: Option<&str>) -> Decl {
    typed_node(
        TypedDeclaration::Function(TypedFunction {
            name: intern(name),
            annotations: Vec::new(),
            effects: Vec::new(),
            with_handlers: Vec::new(),
            type_params: Vec::new(),
            params: params
                .iter()
                .map(|(n, t)| {
                    TypedParameter::regular(intern(n), t.clone(), Mutability::Immutable, SPAN)
                })
                .collect(),
            return_type: ret_ty.clone(),
            body: None,
            visibility: Visibility::Public,
            is_async: false,
            is_fiber: false,
            is_pure: false,
            is_external: true,
            calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
            link_name: symbol.map(intern),
            module: Some(intern(MODULE)),
        }),
        ret_ty,
        SPAN,
    )
}

/// The names called anywhere in a statement.
pub fn callees_of_stmt(stmt: &Stmt, out: &mut std::collections::BTreeSet<String>) {
    match &stmt.node {
        TypedStatement::Expression(e) => callees_of_expr(e, out),
        TypedStatement::Let(l) => {
            if let Some(init) = &l.initializer {
                callees_of_expr(init, out);
            }
        }
        TypedStatement::Return(Some(e)) => callees_of_expr(e, out),
        TypedStatement::If(i) => {
            callees_of_expr(&i.condition, out);
            for s in &i.then_block.statements {
                callees_of_stmt(s, out);
            }
            if let Some(e) = &i.else_block {
                for s in &e.statements {
                    callees_of_stmt(s, out);
                }
            }
        }
        TypedStatement::While(w) => {
            callees_of_expr(&w.condition, out);
            for s in &w.body.statements {
                callees_of_stmt(s, out);
            }
        }
        _ => {}
    }
}

fn callees_of_expr(e: &Expr, out: &mut std::collections::BTreeSet<String>) {
    match &e.node {
        TypedExpression::Call(c) => {
            if let TypedExpression::Variable(n) = &c.callee.node {
                if let Some(name) = n.resolve_global() {
                    out.insert(name);
                }
            }
            for a in &c.positional_args {
                callees_of_expr(a, out);
            }
        }
        TypedExpression::MethodCall(m) => {
            callees_of_expr(&m.receiver, out);
            for a in &m.positional_args {
                callees_of_expr(a, out);
            }
        }
        TypedExpression::Binary(b) => {
            callees_of_expr(&b.left, out);
            callees_of_expr(&b.right, out);
        }
        TypedExpression::Unary(u) => callees_of_expr(&u.operand, out),
        TypedExpression::Index(i) => {
            callees_of_expr(&i.object, out);
            callees_of_expr(&i.index, out);
        }
        TypedExpression::Cast(c) => callees_of_expr(&c.expr, out),
        TypedExpression::If(i) => {
            callees_of_expr(&i.condition, out);
            callees_of_expr(&i.then_branch, out);
            callees_of_expr(&i.else_branch, out);
        }
        TypedExpression::Array(items) => {
            for a in items {
                callees_of_expr(a, out);
            }
        }
        _ => {}
    }
}
