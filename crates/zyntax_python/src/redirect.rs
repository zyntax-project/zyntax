//! The parts of host modules that bundled modules supply. Before the
//! program is linked, `os.path.splitext(p)`, `json.loads(s)` and the
//! other names a bundled module defines become calls of it, the module
//! is imported, and so is the name in `from json import loads`. What the
//! host module has stays the host's: `os.path.join` of two names,
//! `json.dumps`.

use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use std::cell::RefCell;

/// The module for the text functions of `os.path`.
pub(crate) const OSPATH: &str = "__ospath";
/// The module for `json.loads` and `json.load`.
pub(crate) const JSON: &str = "__json";
/// The module for `int(text, base)`.
pub(crate) const INTBASE: &str = "__intbase";
/// The module for `bytearray`.
pub(crate) const BYTEARRAY: &str = "__bytearray";

const OSPATH_NAMES: &[&str] = &["isabs", "split", "splitext", "normpath", "commonprefix"];
const JSON_NAMES: &[&str] = &["loads", "load", "JSONDecodeError"];

/// Redirect the calls of the program, and import the modules that
/// were used.
pub(crate) fn rewrite(main: &mut Vec<py::Stmt>, big_ints: bool) {
    let used: RefCell<Vec<&'static str>> = RefCell::new(Vec::new());
    // `from os.path import splitext` and `from json import loads`.
    let mut out = Vec::with_capacity(main.len());
    for s in std::mem::take(main) {
        match s {
            py::Stmt::ImportFrom(i) if i.level == 0 => {
                let module = i.module.as_ref().map(|m| m.as_str().to_string());
                let (ours, names): (&str, &[&str]) = match module.as_deref() {
                    Some("os.path") => (OSPATH, OSPATH_NAMES),
                    Some("json") => (JSON, JSON_NAMES),
                    _ => {
                        out.push(py::Stmt::ImportFrom(i));
                        continue;
                    }
                };
                let (ours_names, rest): (Vec<py::Alias>, Vec<py::Alias>) = i
                    .names
                    .iter()
                    .cloned()
                    .partition(|a| names.contains(&a.name.as_str()));
                if ours_names.is_empty() {
                    out.push(py::Stmt::ImportFrom(i));
                    continue;
                }
                let mut bundled = i.clone();
                bundled.module = Some(py::Identifier::new(ours, i.range));
                bundled.names = ours_names;
                out.push(py::Stmt::ImportFrom(bundled));
                if !rest.is_empty() {
                    let mut kept = i;
                    kept.names = rest;
                    out.push(py::Stmt::ImportFrom(kept));
                }
            }
            other => out.push(other),
        }
    }
    *main = out;
    let rewriter = Redirect {
        used: &used,
        big_ints,
    };
    for s in main.iter_mut() {
        rewriter.visit_stmt(s);
    }
    let mut seen: Vec<&str> = Vec::new();
    for module in used.into_inner() {
        if seen.contains(&module) {
            continue;
        }
        seen.push(module);
        let import = ruff_python_parser::parse_module(&format!("import {module}\n"))
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
    used: &'a RefCell<Vec<&'static str>>,
    /// The program has big ints, whose `int(text, base)` is the module's.
    big_ints: bool,
}

/// Whether `e` is `os.path`.
fn is_os_path(e: &py::Expr) -> bool {
    matches!(e, py::Expr::Attribute(a)
        if a.attr.as_str() == "path"
            && matches!(&*a.value, py::Expr::Name(n) if n.id.as_str() == "os"))
}

impl Transformer for Redirect<'_> {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        walk_stmt(self, s);
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        walk_expr(self, e);
        // `bytearray` is the module's class.
        if let py::Expr::Name(n) = e
            && n.id.as_str() == "bytearray"
            && n.ctx == py::ExprContext::Load
        {
            let range = n.range;
            *e = py::Expr::Attribute(py::ExprAttribute {
                node_index: Default::default(),
                range,
                value: Box::new(py::Expr::Name(py::ExprName {
                    node_index: Default::default(),
                    range,
                    id: py::name::Name::new(BYTEARRAY),
                    ctx: py::ExprContext::Load,
                })),
                attr: py::Identifier::new("bytearray", range),
                ctx: py::ExprContext::Load,
            });
            self.used.borrow_mut().push(BYTEARRAY);
            return;
        }
        let py::Expr::Call(c) = e else {
            return;
        };
        // `int(text, base)`.
        if !self.big_ints
            && matches!(&*c.func, py::Expr::Name(n) if n.id.as_str() == "int")
            && c.arguments.args.len() == 2
            && c.arguments.keywords.is_empty()
        {
            let range = c.arguments.range;
            *c.func = py::Expr::Attribute(py::ExprAttribute {
                node_index: Default::default(),
                range,
                value: Box::new(py::Expr::Name(py::ExprName {
                    node_index: Default::default(),
                    range,
                    id: py::name::Name::new(INTBASE),
                    ctx: py::ExprContext::Load,
                })),
                attr: py::Identifier::new("of_base", range),
                ctx: py::ExprContext::Load,
            });
            self.used.borrow_mut().push(INTBASE);
            return;
        }
        let py::Expr::Attribute(a) = &*c.func else {
            return;
        };
        let name = a.attr.as_str();
        let module = if is_os_path(&a.value) {
            // `os.path.join` of two names is the host's.
            let ours = OSPATH_NAMES.contains(&name)
                || (name == "join"
                    && c.arguments.args.len() != 2
                    && c.arguments.keywords.is_empty());
            ours.then_some(OSPATH)
        } else if matches!(&*a.value, py::Expr::Name(n) if n.id.as_str() == "json")
            && JSON_NAMES.contains(&name)
        {
            Some(JSON)
        } else {
            None
        };
        let Some(module) = module else {
            return;
        };
        let range = a.range;
        *c.func = py::Expr::Attribute(py::ExprAttribute {
            node_index: Default::default(),
            range,
            value: Box::new(py::Expr::Name(py::ExprName {
                node_index: Default::default(),
                range,
                id: py::name::Name::new(module),
                ctx: py::ExprContext::Load,
            })),
            attr: py::Identifier::new(name, range),
            ctx: py::ExprContext::Load,
        });
        self.used.borrow_mut().push(module);
    }
}
