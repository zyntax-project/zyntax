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
    crate::walk::expand_module(body, origins, &mut |mut s| match &mut s {
        py::Stmt::Assign(a) if a.targets.len() == 1 && starred_in(&a.targets[0]).is_some() => {
            let target = a.targets.pop().expect("one target");
            let value = (*a.value).clone();
            expand(target, value, &mut count)
        }
        py::Stmt::For(f) if starred_in(&f.target).is_some() => {
            let item = format!("__for{count}");
            count += 1;
            let target = std::mem::replace(&mut *f.target, name(&item, py::ExprContext::Store));
            let head = expand(target, name(&item, py::ExprContext::Load), &mut count);
            let old = std::mem::take(&mut f.body);
            f.body = head.into_iter().chain(old).collect();
            vec![s]
        }
        _ => vec![s],
    });
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
