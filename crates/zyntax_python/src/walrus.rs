//! `(name := value)` as an assignment before the statement that holds it.
//! It is read where the walrus is the first thing the statement
//! evaluates: the left operand of a comparison or operator, the first
//! operand of `and` or `or`, the first argument of a call, the subject
//! of an attribute or subscript. The test of an `if` and of a `while` is
//! such a place, and a `while` re-runs the assignment on each pass. A
//! walrus anywhere else is refused, because hoisting it would change
//! what runs and when.

use crate::{Error, Result};
use ruff_python_ast as py;
use ruff_text_size::Ranged;
use std::cell::RefCell;

/// Rewrite every walrus of `body`.
pub(crate) fn rewrite(body: &mut Vec<py::Stmt>, origins: &mut Vec<Option<String>>) -> Result<()> {
    let error: RefCell<Option<Error>> = RefCell::new(None);
    crate::walk::expand_module(body, origins, &mut |s| match statement(s) {
        Ok(stmts) => stmts,
        Err(e) => {
            error.borrow_mut().get_or_insert(e);
            Vec::new()
        }
    });
    match error.into_inner() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The walrus the expression evaluates first, taken out of it: the
/// name it assigns and the value, with the expression now reading the
/// name. `None` when there is none.
fn first(e: &mut py::Expr) -> Option<(py::Expr, py::Expr)> {
    match e {
        py::Expr::Named(n) => {
            let target = (*n.target).clone();
            let value = std::mem::replace(&mut *n.value, name_load("None", n.range));
            let mut read = target.clone();
            set_ctx(&mut read, py::ExprContext::Load);
            *e = read;
            Some((target, value))
        }
        py::Expr::Compare(c) => first(&mut c.left),
        py::Expr::BinOp(b) => first(&mut b.left),
        py::Expr::UnaryOp(u) => first(&mut u.operand),
        py::Expr::BoolOp(b) => b.values.first_mut().and_then(first),
        py::Expr::Attribute(a) => first(&mut a.value),
        py::Expr::Subscript(s) => first(&mut s.value),
        py::Expr::Call(c) => {
            if let Some(found) = first(&mut c.func) {
                return Some(found);
            }
            c.arguments.args.first_mut().and_then(first)
        }
        _ => None,
    }
}

fn name_load(id: &str, range: ruff_text_size::TextRange) -> py::Expr {
    py::Expr::Name(py::ExprName {
        node_index: Default::default(),
        range,
        id: py::name::Name::new(id),
        ctx: py::ExprContext::Load,
    })
}

fn set_ctx(e: &mut py::Expr, ctx: py::ExprContext) {
    if let py::Expr::Name(n) = e {
        n.ctx = ctx;
    }
}

/// Whether any walrus remains in `e`.
fn has_walrus(e: &py::Expr) -> bool {
    use ruff_python_ast::visitor::{Visitor, walk_expr};
    struct V(bool);
    impl<'a> Visitor<'a> for V {
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if matches!(e, py::Expr::Named(_)) {
                self.0 = true;
            }
            walk_expr(self, e);
        }
    }
    let mut v = V(false);
    v.visit_expr(e);
    v.0
}

fn assign(target: py::Expr, value: py::Expr) -> py::Stmt {
    let range = value.range();
    py::Stmt::Assign(py::StmtAssign {
        node_index: Default::default(),
        range,
        targets: vec![target],
        value: Box::new(value),
    })
}

fn refuse<T>(at: &impl Ranged) -> Result<T> {
    Err(Error::unsupported(
        "an assignment expression anywhere but where its statement evaluates first",
        at,
    ))
}

/// Take every leading walrus out of `e`, in the order they run.
fn hoisted(e: &mut py::Expr) -> Result<Vec<py::Stmt>> {
    let mut out = Vec::new();
    while let Some((target, value)) = first(e) {
        // The value may hold a walrus of its own.
        let mut value = value;
        out.extend(hoisted(&mut value)?);
        out.push(assign(target, value));
    }
    if has_walrus(e) {
        return refuse(&*e);
    }
    Ok(out)
}

fn statement(s: py::Stmt) -> Result<Vec<py::Stmt>> {
    match s {
        py::Stmt::If(mut i) => {
            let mut before = hoisted(&mut i.test)?;
            // A walrus in an `elif` runs only when the tests before it
            // fail, so the clause becomes an `else` holding an `if`.
            let mut at = 0;
            while at < i.elif_else_clauses.len() {
                let clause = &mut i.elif_else_clauses[at];
                if let Some(test) = &mut clause.test
                    && has_walrus(test)
                {
                    let tail: Vec<py::ElifElseClause> = i.elif_else_clauses.drain(at..).collect();
                    let mut tail = tail.into_iter();
                    let head = tail.next().expect("a clause");
                    let mut nested = py::StmtIf {
                        node_index: Default::default(),
                        range: head.range,
                        test: Box::new(head.test.expect("an elif has a test")),
                        body: head.body,
                        elif_else_clauses: tail.collect(),
                    };
                    let mut stmts = hoisted(&mut nested.test)?;
                    let inner = statement(py::Stmt::If(nested))?;
                    stmts.extend(inner);
                    i.elif_else_clauses.push(py::ElifElseClause {
                        range: i.range,
                        node_index: Default::default(),
                        test: None,
                        body: stmts.into_iter().collect(),
                    });
                    break;
                }
                if let Some(test) = &clause.test
                    && has_walrus(test)
                {
                    return refuse(test);
                }
                at += 1;
            }
            before.push(py::Stmt::If(i));
            Ok(before)
        }
        py::Stmt::While(mut w) => {
            if !has_walrus(&w.test) {
                return Ok(vec![py::Stmt::While(w)]);
            }
            if !w.orelse.is_empty() {
                return refuse(&*w.test);
            }
            // `while test: body` is `while True: assign; if not test:
            // break; body`.
            let mut head = hoisted(&mut w.test)?;
            let range = w.range;
            let stop = py::Stmt::If(py::StmtIf {
                node_index: Default::default(),
                range,
                test: Box::new(py::Expr::UnaryOp(py::ExprUnaryOp {
                    node_index: Default::default(),
                    range,
                    op: py::UnaryOp::Not,
                    operand: Box::new(std::mem::replace(&mut *w.test, name_load("None", range))),
                })),
                body: vec![py::Stmt::Break(py::StmtBreak {
                    node_index: Default::default(),
                    range,
                })]
                .into_iter()
                .collect(),
                elif_else_clauses: Vec::new(),
            });
            head.push(stop);
            head.extend(std::mem::take(&mut w.body));
            w.body = head.into_iter().collect();
            *w.test = py::Expr::BooleanLiteral(py::ExprBooleanLiteral {
                node_index: Default::default(),
                range,
                value: true,
            });
            Ok(vec![py::Stmt::While(w)])
        }
        py::Stmt::Assign(mut a) => {
            let mut before = hoisted(&mut a.value)?;
            before.push(py::Stmt::Assign(a));
            Ok(before)
        }
        py::Stmt::AugAssign(mut a) => {
            let mut before = hoisted(&mut a.value)?;
            before.push(py::Stmt::AugAssign(a));
            Ok(before)
        }
        py::Stmt::Expr(mut e) => {
            let mut before = hoisted(&mut e.value)?;
            before.push(py::Stmt::Expr(e));
            Ok(before)
        }
        py::Stmt::Return(mut r) => {
            let mut before = match &mut r.value {
                Some(v) => hoisted(v)?,
                None => Vec::new(),
            };
            before.push(py::Stmt::Return(r));
            Ok(before)
        }
        other => Ok(vec![other]),
    }
}
