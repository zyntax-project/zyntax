//! Python's scoping rules: which names a body binds, which it declares
//! global or nonlocal, and which it reads from an enclosing scope.

use ruff_python_ast as py;
use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
use std::collections::HashSet;

/// What one body does with names.
#[derive(Debug, Default, Clone)]
pub(crate) struct Scope {
    /// Names assigned in this body, which makes them its locals.
    pub bound: HashSet<String>,
    /// Names declared `global`.
    pub globals: HashSet<String>,
    /// Names read here, or free in a nested body, that this body does
    /// not bind: they come from an enclosing scope or the module.
    pub free: HashSet<String>,
    /// Nested function bodies, by name.
    pub children: Vec<(String, Scope)>,
}

impl Scope {
    pub(crate) fn of_function(f: &py::StmtFunctionDef) -> Scope {
        let params = f
            .parameters
            .iter_non_variadic_params()
            .map(|p| p.parameter.name.to_string())
            .collect();
        Self::of_body(params, &f.body)
    }

    pub(crate) fn of_lambda(l: &py::ExprLambda) -> Scope {
        let params = l
            .parameters
            .as_ref()
            .map(|ps| {
                ps.iter_non_variadic_params()
                    .map(|p| p.parameter.name.to_string())
                    .collect()
            })
            .unwrap_or_default();
        let mut collector = Collector::default();
        collector.visit_expr(&l.body);
        collector.finish(params)
    }

    pub(crate) fn of_body(params: Vec<String>, body: &[py::Stmt]) -> Scope {
        let mut collector = Collector::default();
        for s in body {
            collector.visit_stmt(s);
        }
        collector.finish(params)
    }

    /// Every name declared `global` here or in a nested body.
    pub(crate) fn declared_globals(&self) -> HashSet<String> {
        let mut out = self.globals.clone();
        for (_, child) in &self.children {
            out.extend(child.declared_globals());
        }
        out
    }
}

#[derive(Default)]
struct Collector {
    loads: HashSet<String>,
    bound: HashSet<String>,
    globals: HashSet<String>,
    nonlocals: HashSet<String>,
    children: Vec<(String, Scope)>,
}

impl Collector {
    fn finish(self, params: Vec<String>) -> Scope {
        let mut free: HashSet<String> = self.loads;
        for (_, child) in &self.children {
            free.extend(child.free.iter().cloned());
        }
        for name in params.iter().chain(&self.bound).chain(&self.globals) {
            free.remove(name);
        }
        Scope {
            bound: self.bound,
            globals: self.globals,
            free,
            children: self.children,
        }
    }

    fn bind(&mut self, name: &str) {
        if !self.globals.contains(name) && !self.nonlocals.contains(name) {
            self.bound.insert(name.to_string());
        }
    }
}

impl<'a> Visitor<'a> for Collector {
    fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
        match stmt {
            py::Stmt::FunctionDef(f) => {
                self.bind(f.name.as_str());
                for p in f.parameters.iter_non_variadic_params() {
                    if let Some(d) = &p.default {
                        self.visit_expr(d);
                    }
                }
                self.children
                    .push((f.name.to_string(), Scope::of_function(f)));
            }
            py::Stmt::ClassDef(c) => self.bind(c.name.as_str()),
            py::Stmt::Global(g) => {
                for n in &g.names {
                    self.globals.insert(n.to_string());
                    self.bound.remove(n.as_str());
                }
            }
            py::Stmt::Nonlocal(g) => {
                for n in &g.names {
                    self.nonlocals.insert(n.to_string());
                    self.bound.remove(n.as_str());
                }
            }
            py::Stmt::Import(_) | py::Stmt::ImportFrom(_) => {}
            _ => walk_stmt(self, stmt),
        }
    }

    fn visit_expr(&mut self, expr: &'a py::Expr) {
        match expr {
            py::Expr::Name(n) => match n.ctx {
                py::ExprContext::Load => {
                    self.loads.insert(n.id.to_string());
                }
                py::ExprContext::Store | py::ExprContext::Del => self.bind(n.id.as_str()),
                py::ExprContext::Invalid => {}
            },
            py::Expr::Lambda(l) => {
                self.children.push((String::new(), Scope::of_lambda(l)));
            }
            _ => walk_expr(self, expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Scope;

    fn scope_of(src: &str) -> Scope {
        let module = ruff_python_parser::parse_module(src).unwrap().into_syntax();
        Scope::of_body(Vec::new(), &module.body)
    }

    #[test]
    fn a_read_of_an_unbound_name_is_free() {
        let s = scope_of("x = a + 1\ny = x");
        assert!(s.bound.contains("x") && s.bound.contains("y"));
        assert!(s.free.contains("a"));
        assert!(!s.free.contains("x"));
    }

    #[test]
    fn a_nested_body_passes_its_free_names_up() {
        let s = scope_of("def f(n):\n    global total\n    total += n\n    return m");
        let (_, f) = &s.children[0];
        assert!(f.globals.contains("total"));
        assert!(!f.bound.contains("total"));
        assert!(f.free.contains("m"));
        assert!(s.free.contains("m"));
        assert!(s.declared_globals().contains("total"));
    }
}
