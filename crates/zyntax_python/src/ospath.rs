//! The parts of `os.path` the host does not have, found in a bundled
//! module. Before the program is linked, `os.path.splitext(p)` and the
//! other names that module defines become calls of it, and so does
//! `os.path.join` with other than two arguments; the rest of `os.path`
//! is the host's.

use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr};
use std::cell::Cell;

/// The module the functions are in.
pub(crate) const MODULE: &str = "__ospath";

/// The names of `os.path` the module defines.
const NAMES: &[&str] = &["isabs", "split", "splitext", "normpath", "commonprefix"];

/// Redirect the calls of the program, and import the module when any
/// was.
pub(crate) fn rewrite(main: &mut Vec<py::Stmt>) {
    let used = Cell::new(false);
    let rewriter = Redirect { used: &used };
    for s in main.iter_mut() {
        rewriter.visit_stmt(s);
    }
    if used.get() {
        let import = ruff_python_parser::parse_module(&format!("import {MODULE}\n"))
            .expect("the import parses")
            .into_syntax()
            .body
            .into_iter()
            .next()
            .expect("one statement");
        main.insert(0, import);
    }
}

struct Redirect<'a> {
    used: &'a Cell<bool>,
}

/// Whether `e` is `os.path`.
fn is_os_path(e: &py::Expr) -> bool {
    matches!(e, py::Expr::Attribute(a)
        if a.attr.as_str() == "path"
            && matches!(&*a.value, py::Expr::Name(n) if n.id.as_str() == "os"))
}

impl Transformer for Redirect<'_> {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        use ruff_python_ast::visitor::transformer::walk_stmt;
        // `from os.path import splitext` takes the module's name.
        if let py::Stmt::ImportFrom(i) = s
            && i.level == 0
            && i.module.as_ref().is_some_and(|m| m.as_str() == "os.path")
            && i.names.iter().all(|a| NAMES.contains(&a.name.as_str()))
        {
            i.module = Some(py::Identifier::new(MODULE, i.range));
            return;
        }
        walk_stmt(self, s);
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        walk_expr(self, e);
        let py::Expr::Call(c) = e else {
            return;
        };
        let py::Expr::Attribute(a) = &*c.func else {
            return;
        };
        if !is_os_path(&a.value) {
            return;
        }
        let name = a.attr.as_str();
        // `os.path.join` of two names is the host's.
        let ours = NAMES.contains(&name)
            || (name == "join" && c.arguments.args.len() != 2 && c.arguments.keywords.is_empty());
        if !ours {
            return;
        }
        let range = a.range;
        *c.func = py::Expr::Attribute(py::ExprAttribute {
            node_index: Default::default(),
            range,
            value: Box::new(py::Expr::Name(py::ExprName {
                node_index: Default::default(),
                range,
                id: py::name::Name::new(MODULE),
                ctx: py::ExprContext::Load,
            })),
            attr: py::Identifier::new(name, range),
            ctx: py::ExprContext::Load,
        });
        self.used.set(true);
    }
}
