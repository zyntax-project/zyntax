//! Line restores for Lua lowering.

use super::*;

impl LineRestores<'_> {
    pub(super) fn strip(statements: &mut [St], reads_line: &dyn Fn(&str) -> bool) {
        let mut walk = LineRestores {
            reads_line,
            line: intern(library::LINE),
            pending: intern(library::PENDING),
            exits: Vec::new(),
        };
        walk.stmts(statements, false, true);
    }

    /// Whether the line is read from before `statements` on, given
    /// `later` after them; `strip` removes dead restores on the way.
    pub(super) fn stmts(&mut self, statements: &mut [St], mut later: bool, strip: bool) -> bool {
        for s in statements.iter_mut().rev() {
            later = self.stmt(s, later, strip);
        }
        later
    }

    pub(super) fn stmt(&mut self, s: &mut St, later: bool, strip: bool) -> bool {
        match &mut s.node {
            TypedStatement::Expression(e) => self.expr(e, later, strip),
            TypedStatement::Let(l) => match &mut l.initializer {
                Some(e) => self.expr(e, later, strip),
                None => later,
            },
            TypedStatement::If(i) if self.is_check(&i.condition) => {
                // The raising call before the check reads the line.
                if strip && !later && self.is_restore(i.else_block.as_ref()) {
                    i.else_block = None;
                }
                true
            }
            TypedStatement::If(i) => {
                let then = self.stmts(&mut i.then_block.statements, later, strip);
                let els = match &mut i.else_block {
                    Some(b) => self.stmts(&mut b.statements, later, strip),
                    None => later,
                };
                self.expr(&mut i.condition, then || els, strip)
            }
            TypedStatement::While(w) => {
                // The line read at the head, which the end of the body
                // reaches again: the least answer, from a first walk
                // assuming it is not; inner loops assume it is.
                let head = if strip {
                    self.exits.push(later);
                    let body = self.stmts(&mut w.body.statements, false, false);
                    self.exits.pop();
                    self.expr(&mut w.condition, body || later, false)
                } else {
                    true
                };
                self.exits.push(later);
                let body = self.stmts(&mut w.body.statements, head, strip);
                self.exits.pop();
                self.expr(&mut w.condition, body || later, strip)
            }
            TypedStatement::Block(b) => self.stmts(&mut b.statements, later, strip),
            TypedStatement::Return(v) => match v {
                Some(e) => self.expr(e, false, strip),
                None => false,
            },
            TypedStatement::Break(None) => self.exits.last().copied().unwrap_or(true),
            TypedStatement::Label(_) => later,
            _ => true,
        }
    }

    pub(super) fn expr(&mut self, e: &mut Node, later: bool, strip: bool) -> bool {
        match &mut e.node {
            TypedExpression::Literal(_) => later,
            TypedExpression::Variable(name) => later || *name == self.line,
            TypedExpression::Call(c) => {
                let reads = match &c.callee.node {
                    TypedExpression::Variable(name) => name
                        .resolve_global()
                        .is_some_and(|n| (self.reads_line)(n.as_str())),
                    _ => true,
                };
                let mut later = later || reads;
                for a in c.positional_args.iter_mut().rev() {
                    later = self.expr(a, later, strip);
                }
                if c.named_args.is_empty() { later } else { true }
            }
            TypedExpression::MethodCall(c) => {
                let mut later = later;
                for a in c.positional_args.iter_mut().rev() {
                    later = self.expr(a, later, strip);
                }
                if c.named_args.is_empty() {
                    self.expr(&mut c.receiver, later, strip)
                } else {
                    true
                }
            }
            TypedExpression::Binary(b) => match b.op {
                BinaryOp::Assign => match &b.left.node {
                    // A store of the line: nothing before it is read
                    // through it.
                    TypedExpression::Variable(name) if *name == self.line => {
                        self.expr(&mut b.right, false, strip)
                    }
                    _ => {
                        let later = self.expr(&mut b.left, later, strip);
                        self.expr(&mut b.right, later, strip)
                    }
                },
                BinaryOp::And | BinaryOp::Or => {
                    let right = self.expr(&mut b.right, later, strip);
                    self.expr(&mut b.left, right || later, strip)
                }
                _ => {
                    let later = self.expr(&mut b.right, later, strip);
                    self.expr(&mut b.left, later, strip)
                }
            },
            TypedExpression::Unary(u) => self.expr(&mut u.operand, later, strip),
            TypedExpression::Cast(c) => self.expr(&mut c.expr, later, strip),
            TypedExpression::Field(f) => self.expr(&mut f.object, later, strip),
            TypedExpression::Index(i) => {
                let later = self.expr(&mut i.index, later, strip);
                self.expr(&mut i.object, later, strip)
            }
            TypedExpression::If(i) => {
                let then = self.expr(&mut i.then_branch, later, strip);
                let els = self.expr(&mut i.else_branch, later, strip);
                self.expr(&mut i.condition, then || els, strip)
            }
            TypedExpression::Block(b) => self.stmts(&mut b.statements, later, strip),
            TypedExpression::Array(items) | TypedExpression::Tuple(items) => {
                let mut later = later;
                for item in items.iter_mut().rev() {
                    later = self.expr(item, later, strip);
                }
                later
            }
            TypedExpression::Struct(s) => {
                let mut later = later;
                for f in s.fields.iter_mut().rev() {
                    later = self.expr(&mut f.value, later, strip);
                }
                later
            }
            _ => true,
        }
    }

    /// `pending != nil`: the check after a call that may have raised.
    pub(super) fn is_check(&self, cond: &Node) -> bool {
        matches!(&cond.node, TypedExpression::Binary(b)
            if b.op == BinaryOp::Ne
                && matches!(&b.left.node, TypedExpression::Variable(n) if *n == self.pending)
                && matches!(&b.right.node, TypedExpression::Literal(TypedLiteral::Null)))
    }

    /// A check's else branch that only stores the line.
    pub(super) fn is_restore(&self, els: Option<&TypedBlock>) -> bool {
        let Some(b) = els else {
            return false;
        };
        let [only] = b.statements.as_slice() else {
            return false;
        };
        let TypedStatement::Expression(e) = &only.node else {
            return false;
        };
        matches!(&e.node, TypedExpression::Binary(b)
            if b.op == BinaryOp::Assign
                && matches!(&b.left.node, TypedExpression::Variable(n) if *n == self.line)
                && matches!(&b.right.node, TypedExpression::Literal(TypedLiteral::Integer(_))))
    }
}
