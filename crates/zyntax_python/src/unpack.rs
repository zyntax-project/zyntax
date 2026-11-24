//! Assignment to a target with a `*` in it, written as the statements a
//! program would. `a, *rest, z = xs` holds `list(xs)` and assigns the
//! first, the middle slice and the last from it, after checking it is
//! long enough. A `for` target with a `*` takes each item the same way.

use crate::dataclass::Fill;
use ruff_python_ast as py;
use rustc_hash::FxHashMap as HashMap;

/// Rewrite every starred target of `body`, at any depth. `origins` has
/// the module each top-level statement came from.
pub(crate) fn rewrite(body: &mut Vec<py::Stmt>, origins: &mut Vec<Option<String>>) {
    let mut count = 0;
    let mut stmts = Vec::with_capacity(body.len());
    let mut kept = Vec::with_capacity(body.len());
    for (s, origin) in std::mem::take(body)
        .into_iter()
        .zip(std::mem::take(origins))
    {
        let mut one = vec![s];
        block(&mut one, &mut count);
        kept.extend(std::iter::repeat_n(origin, one.len()));
        stmts.extend(one);
    }
    *body = stmts;
    *origins = kept;
}

fn block<B>(body: &mut B, count: &mut usize)
where
    B: Default + IntoIterator<Item = py::Stmt> + FromIterator<py::Stmt>,
{
    let mut out = Vec::new();
    for mut s in std::mem::take(body) {
        nested(&mut s, count);
        match &mut s {
            py::Stmt::Assign(a) if a.targets.len() == 1 && starred_in(&a.targets[0]).is_some() => {
                let target = a.targets.pop().expect("one target");
                let value = (*a.value).clone();
                out.extend(expand(target, value, count));
            }
            py::Stmt::For(f) if starred_in(&f.target).is_some() => {
                let item = format!("__for{}", *count);
                *count += 1;
                let target = std::mem::replace(&mut *f.target, name(&item, py::ExprContext::Store));
                let head = expand(target, name(&item, py::ExprContext::Load), count);
                let old = std::mem::take(&mut f.body);
                f.body = head.into_iter().chain(old).collect();
                out.push(s);
            }
            _ => out.push(s),
        }
    }
    *body = out.into_iter().collect();
}

/// The statement lists inside `s`, rewritten.
fn nested(s: &mut py::Stmt, count: &mut usize) {
    match s {
        py::Stmt::FunctionDef(f) => block(&mut f.body, count),
        py::Stmt::ClassDef(c) => block(&mut c.body, count),
        py::Stmt::If(i) => {
            block(&mut i.body, count);
            for clause in i.elif_else_clauses.iter_mut() {
                block(&mut clause.body, count);
            }
        }
        py::Stmt::For(f) => {
            block(&mut f.body, count);
            block(&mut f.orelse, count);
        }
        py::Stmt::While(w) => {
            block(&mut w.body, count);
            block(&mut w.orelse, count);
        }
        py::Stmt::With(w) => block(&mut w.body, count),
        py::Stmt::Try(t) => {
            block(&mut t.body, count);
            for h in t.handlers.iter_mut() {
                let py::ExceptHandler::ExceptHandler(h) = h;
                block(&mut h.body, count);
            }
            block(&mut t.orelse, count);
            block(&mut t.finalbody, count);
        }
        py::Stmt::Match(m) => {
            for case in m.cases.iter_mut() {
                block(&mut case.body, count);
            }
        }
        _ => {}
    }
}

/// The elements of a tuple or list target, when one of them is starred.
fn starred_in(target: &py::Expr) -> Option<&[py::Expr]> {
    let elts = match target {
        py::Expr::Tuple(t) => &t.elts,
        py::Expr::List(l) => &l.elts,
        _ => return None,
    };
    elts.iter()
        .any(|e| matches!(e, py::Expr::Starred(_)))
        .then_some(&elts[..])
}

fn name(id: &str, ctx: py::ExprContext) -> py::Expr {
    py::Expr::Name(py::ExprName {
        node_index: Default::default(),
        range: Default::default(),
        id: py::name::Name::new(id),
        ctx,
    })
}

/// The statements that assign `value` to the starred `target`.
fn expand(target: py::Expr, value: py::Expr, count: &mut usize) -> Vec<py::Stmt> {
    let elts = match target {
        py::Expr::Tuple(t) => t.elts,
        py::Expr::List(l) => l.elts,
        other => return vec![assign(other, value)],
    };
    let held = format!("__unpack{}", *count);
    *count += 1;
    let at = elts
        .iter()
        .position(|e| matches!(e, py::Expr::Starred(_)))
        .expect("a starred element");
    let before = at;
    let after = elts.len() - at - 1;
    let mut source = format!(
        "{held} = list(__value)\nif len({held}) < {need}:\n    raise ValueError(\"not enough values to unpack (expected at least {need}, got \" + str(len({held})) + \")\")\n",
        need = before + after
    );
    for i in 0..elts.len() {
        let index = if i < at {
            format!("{i}")
        } else if i == at {
            if after == 0 {
                format!("{before}:")
            } else {
                format!("{before}:len({held}) - {after}")
            }
        } else {
            format!("-{}", elts.len() - i)
        };
        source.push_str(&format!("__target{i} = {held}[{index}]\n"));
    }
    let mut fill: HashMap<String, py::Expr> = HashMap::default();
    fill.insert("__value".to_string(), value);
    for (i, e) in elts.into_iter().enumerate() {
        let e = match e {
            py::Expr::Starred(s) => *s.value,
            other => other,
        };
        fill.insert(format!("__target{i}"), e);
    }
    let fill = Fill(fill);
    let mut stmts = ruff_python_parser::parse_module(&source)
        .expect("an unpacking parses")
        .into_syntax()
        .body;
    for s in stmts.iter_mut() {
        use ruff_python_ast::visitor::transformer::walk_stmt;
        walk_stmt(&fill, s);
    }
    stmts.into_iter().collect()
}

fn assign(target: py::Expr, value: py::Expr) -> py::Stmt {
    let mut s = ruff_python_parser::parse_module("__target = __value\n")
        .expect("an assignment parses")
        .into_syntax()
        .body
        .remove(0);
    let mut fill: HashMap<String, py::Expr> = HashMap::default();
    fill.insert("__target".to_string(), target);
    fill.insert("__value".to_string(), value);
    use ruff_python_ast::visitor::transformer::walk_stmt;
    walk_stmt(&Fill(fill), &mut s);
    s
}
