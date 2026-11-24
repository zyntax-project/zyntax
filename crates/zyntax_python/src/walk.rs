//! Rewrites that replace one statement with several, at any depth.

use ruff_python_ast as py;

/// Every statement of `body` and of the statement lists inside it,
/// innermost first, replaced by what `f` returns for it. `origins` has the
/// module each top-level statement came from.
pub(crate) fn expand_module(
    body: &mut Vec<py::Stmt>,
    origins: &mut Vec<Option<String>>,
    f: &mut dyn FnMut(py::Stmt) -> Vec<py::Stmt>,
) {
    let mut stmts = Vec::with_capacity(body.len());
    let mut kept = Vec::with_capacity(body.len());
    for (s, origin) in std::mem::take(body)
        .into_iter()
        .zip(std::mem::take(origins))
    {
        let mut one = vec![s];
        block(&mut one, f);
        kept.extend(std::iter::repeat_n(origin, one.len()));
        stmts.extend(one);
    }
    *body = stmts;
    *origins = kept;
}

fn block<B>(body: &mut B, f: &mut dyn FnMut(py::Stmt) -> Vec<py::Stmt>)
where
    B: Default + IntoIterator<Item = py::Stmt> + FromIterator<py::Stmt>,
{
    let mut out = Vec::new();
    for mut s in std::mem::take(body) {
        nested(&mut s, f);
        out.extend(f(s));
    }
    *body = out.into_iter().collect();
}

/// The statement lists inside `s`, rewritten.
fn nested(s: &mut py::Stmt, f: &mut dyn FnMut(py::Stmt) -> Vec<py::Stmt>) {
    match s {
        py::Stmt::FunctionDef(d) => block(&mut d.body, f),
        py::Stmt::ClassDef(c) => block(&mut c.body, f),
        py::Stmt::If(i) => {
            block(&mut i.body, f);
            for clause in i.elif_else_clauses.iter_mut() {
                block(&mut clause.body, f);
            }
        }
        py::Stmt::For(l) => {
            block(&mut l.body, f);
            block(&mut l.orelse, f);
        }
        py::Stmt::While(w) => {
            block(&mut w.body, f);
            block(&mut w.orelse, f);
        }
        py::Stmt::With(w) => block(&mut w.body, f),
        py::Stmt::Try(t) => {
            block(&mut t.body, f);
            for h in t.handlers.iter_mut() {
                let py::ExceptHandler::ExceptHandler(h) = h;
                block(&mut h.body, f);
            }
            block(&mut t.orelse, f);
            block(&mut t.finalbody, f);
        }
        py::Stmt::Match(m) => {
            for case in m.cases.iter_mut() {
                block(&mut case.body, f);
            }
        }
        _ => {}
    }
}
