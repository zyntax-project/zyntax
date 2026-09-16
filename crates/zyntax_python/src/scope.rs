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
    /// Names declared `nonlocal`.
    pub nonlocals: HashSet<String>,
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
        let mut collector = Collector::default();
        for s in &f.body {
            collector.visit_stmt(s);
        }
        // A default is evaluated where the function is defined, and a
        // call leaving the argument out evaluates it there too, so the
        // names it reads are the function's to reach.
        let mut defaults = Collector::default();
        for p in f.parameters.iter_non_variadic_params() {
            if let Some(d) = &p.default {
                defaults.visit_expr(d);
            }
        }
        let mut scope = collector.finish(params);
        scope.free.extend(defaults.loads);
        scope
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

    /// A generator expression: its loop variables are its own.
    pub(crate) fn of_generator(g: &py::ExprGenerator) -> Scope {
        let mut collector = Collector::default();
        for comp in &g.generators {
            collector.visit_expr(&comp.target);
            collector.visit_expr(&comp.iter);
            for cond in &comp.ifs {
                collector.visit_expr(cond);
            }
        }
        collector.visit_expr(&g.elt);
        collector.finish(Vec::new())
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
            nonlocals: self.nonlocals,
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
            // A class binds its name; its methods are bodies of their
            // own, reading the module the way any function does.
            py::Stmt::ClassDef(c) => {
                self.bind(c.name.as_str());
                for s in &c.body {
                    if let py::Stmt::FunctionDef(m) = s {
                        self.children.push((
                            format!("{}.{}", c.name.as_str(), m.name.as_str()),
                            Scope::of_function(m),
                        ));
                    }
                }
            }
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
            py::Expr::Generator(g) => {
                self.children.push((String::new(), Scope::of_generator(g)));
            }
            _ => walk_expr(self, expr),
        }
    }
}

/// Whether an `except` clause may still hold its exception once it has
/// run: it re-raises it with a bare `raise`, or it does something with
/// the name it bound the exception to other than read from it. Reading
/// is a field access, a conversion to text, or an argument to one of
/// the built-in functions that keep nothing.
pub(crate) fn handler_keeps_exception(body: &[py::Stmt], name: Option<&str>) -> bool {
    let mut k = Keeps { name, keeps: false };
    for s in body {
        k.visit_stmt(s);
    }
    k.keeps
}

struct Keeps<'n> {
    name: Option<&'n str>,
    keeps: bool,
}

impl Keeps<'_> {
    fn is_it(&self, e: &py::Expr) -> bool {
        matches!((e, self.name), (py::Expr::Name(n), Some(name)) if n.id.as_str() == name)
    }
}

/// Built-in functions that read an argument and keep nothing of it.
const READERS: &[&str] = &["print", "str", "repr", "len", "type", "isinstance", "bool"];

impl<'a> Visitor<'a> for Keeps<'_> {
    fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
        match stmt {
            py::Stmt::Raise(r) if r.exc.is_none() => self.keeps = true,
            // A nested body that reads the name may run after the
            // clause is done.
            py::Stmt::FunctionDef(f) => {
                if self
                    .name
                    .is_some_and(|n| Scope::of_function(f).free.contains(n))
                {
                    self.keeps = true;
                }
            }
            _ => walk_stmt(self, stmt),
        }
    }

    fn visit_expr(&mut self, expr: &'a py::Expr) {
        if self.name.is_none() {
            return;
        }
        match expr {
            py::Expr::Name(n) => {
                if self.is_it(expr) && n.ctx == py::ExprContext::Load {
                    self.keeps = true;
                }
            }
            py::Expr::Attribute(a) if self.is_it(&a.value) => {}
            py::Expr::Call(c) => {
                // A method is handed the exception as its receiver.
                if matches!(&*c.func, py::Expr::Attribute(a) if self.is_it(&a.value)) {
                    self.keeps = true;
                    return;
                }
                let reader =
                    matches!(&*c.func, py::Expr::Name(f) if READERS.contains(&f.id.as_str()));
                if reader {
                    for arg in &c.arguments.args {
                        if !self.is_it(arg) {
                            self.visit_expr(arg);
                        }
                    }
                    for kw in &c.arguments.keywords {
                        self.visit_expr(&kw.value);
                    }
                } else {
                    walk_expr(self, expr);
                }
            }
            py::Expr::Lambda(l) => {
                if self
                    .name
                    .is_some_and(|n| Scope::of_lambda(l).free.contains(n))
                {
                    self.keeps = true;
                }
            }
            py::Expr::Generator(g) => {
                if self
                    .name
                    .is_some_and(|n| Scope::of_generator(g).free.contains(n))
                {
                    self.keeps = true;
                }
            }
            _ => walk_expr(self, expr),
        }
    }

    fn visit_interpolated_string_element(&mut self, element: &'a py::InterpolatedStringElement) {
        // An interpolation is a conversion to text.
        if let py::InterpolatedStringElement::Interpolation(e) = element {
            if self.is_it(&e.expression) {
                return;
            }
        }
        ruff_python_ast::visitor::walk_interpolated_string_element(self, element);
    }
}

#[cfg(test)]
mod tests {
    use super::{handler_keeps_exception, Scope};

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

    fn keeps(handler_body: &str, name: Option<&str>) -> bool {
        let src = format!("try:\n    pass\nexcept E as e:\n{handler_body}");
        let module = ruff_python_parser::parse_module(&src)
            .unwrap()
            .into_syntax();
        let ruff_python_ast::Stmt::Try(t) = &module.body[0] else {
            panic!("a try statement");
        };
        let ruff_python_ast::ExceptHandler::ExceptHandler(h) = &t.handlers[0];
        handler_keeps_exception(&h.body, name)
    }

    #[test]
    fn a_handler_that_only_reads_its_exception_keeps_nothing() {
        assert!(!keeps(
            "    print(e)\n    m = e.message\n    s = f'{e}: {str(e)}'\n",
            Some("e")
        ));
        assert!(!keeps("    total += 1\n", None));
        assert!(!keeps("    return 0\n", Some("e")));
    }

    #[test]
    fn a_handler_that_stores_raises_or_captures_its_exception_keeps_it() {
        assert!(keeps("    raise\n", None));
        assert!(keeps("    saved = e\n", Some("e")));
        assert!(keeps("    errors.append(e)\n", Some("e")));
        assert!(keeps("    return e\n", Some("e")));
        assert!(keeps("    f = lambda: e\n", Some("e")));
        assert!(keeps("    e.register()\n", Some("e")));
    }
}
