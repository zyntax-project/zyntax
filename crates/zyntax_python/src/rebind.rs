//! Flow-sensitive local rebinding. A local is typed as the join of
//! everything assigned to it, so `data = array('B', data)` leaves
//! `data` as wide as the parameter it replaced. Where a rebinding in
//! straight-line code is the only binding every later occurrence can
//! see, it is a fresh variable from that statement on, and is renamed
//! to one (`x`, `x$1`, `x$2`, ...) before inference, which then types
//! each version on its own.
//!
//! A rebinding keeps its name when it is inside a loop (the statements
//! before it in the body read it on the next pass), when the name
//! occurs after the block the rebinding is in (a merge reads whichever
//! binding ran), or when the name is read by a nested body that runs
//! later: a `def`, a `lambda`, a class body, or a generator expression,
//! whose outermost iterable alone is evaluated where it is written.
//! Names a `global`, `nonlocal`, `del`, import, `except ... as`,
//! walrus, comprehension target or `match` binds are left alone too.

use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use ruff_python_ast::visitor::{Visitor, walk_expr as walk_expr_ref, walk_stmt as walk_stmt_ref};
use ruff_text_size::TextSize;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// Rewrite every function body in the module, nested ones included.
/// The module body itself is left alone: its variables are the globals
/// every function reads.
pub(crate) fn rewrite(body: &mut [py::Stmt]) {
    for s in body.iter_mut() {
        functions(s);
    }
}

fn functions(s: &mut py::Stmt) {
    match s {
        py::Stmt::FunctionDef(f) => {
            function(f);
            for inner in f.body.iter_mut() {
                functions(inner);
            }
        }
        py::Stmt::ClassDef(c) => {
            for inner in c.body.iter_mut() {
                functions(inner);
            }
        }
        py::Stmt::If(i) => {
            for inner in i.body.iter_mut() {
                functions(inner);
            }
            for clause in i.elif_else_clauses.iter_mut() {
                for inner in clause.body.iter_mut() {
                    functions(inner);
                }
            }
        }
        py::Stmt::For(f) => {
            for inner in f.body.iter_mut().chain(f.orelse.iter_mut()) {
                functions(inner);
            }
        }
        py::Stmt::While(w) => {
            for inner in w.body.iter_mut().chain(w.orelse.iter_mut()) {
                functions(inner);
            }
        }
        py::Stmt::With(w) => {
            for inner in w.body.iter_mut() {
                functions(inner);
            }
        }
        py::Stmt::Try(t) => {
            for inner in t.body.iter_mut() {
                functions(inner);
            }
            for h in t.handlers.iter_mut() {
                let py::ExceptHandler::ExceptHandler(h) = h;
                for inner in h.body.iter_mut() {
                    functions(inner);
                }
            }
            for inner in t.orelse.iter_mut().chain(t.finalbody.iter_mut()) {
                functions(inner);
            }
        }
        _ => {}
    }
}

fn function(f: &mut py::StmtFunctionDef) {
    let mut facts = Facts::default();
    for p in f.parameters.iter_non_variadic_params() {
        let name = p.parameter.name.to_string();
        facts.first.insert(name.clone(), TextSize::default());
        facts.locals.insert(name);
    }
    for s in &f.body {
        facts.visit_stmt(s);
    }
    let mut versions = Versions {
        facts,
        next: HashMap::default(),
    };
    block(&mut f.body, &mut versions, &|_| false);
}

/// One block's statements, in order. A rebinding at `i` renames the
/// statements after it; a conditional or `with` at `i` is entered with
/// the statements after it as what its own rebindings must not reach.
fn block(stmts: &mut [py::Stmt], v: &mut Versions, outer_tail: &dyn Fn(&str) -> bool) {
    for i in 0..stmts.len() {
        let (head, tail) = stmts.split_at_mut(i + 1);
        let s = &mut head[i];
        if let py::Stmt::Assign(a) = s
            && let [py::Expr::Name(target)] = a.targets.as_mut_slice()
            && v.is_candidate(target.id.as_str(), a.range.start())
            && occurs_in(tail, target.id.as_str())
            && !outer_tail(target.id.as_str())
        {
            let from = target.id.to_string();
            let to = v.fresh(&from, a.range.start());
            target.id = py::name::Name::new(&to);
            let renamer = Renamer { from, to };
            for t in tail.iter_mut() {
                renamer.visit_stmt(t);
            }
            continue;
        }
        let inner_tail = |name: &str| occurs_in(tail, name) || outer_tail(name);
        match s {
            py::Stmt::If(i) => {
                block(&mut i.body, v, &inner_tail);
                for clause in i.elif_else_clauses.iter_mut() {
                    block(&mut clause.body, v, &inner_tail);
                }
            }
            py::Stmt::With(w) => block(&mut w.body, v, &inner_tail),
            _ => {}
        }
    }
}

struct Versions {
    facts: Facts,
    /// The versions handed out so far, by base name.
    next: HashMap<String, usize>,
}

impl Versions {
    /// Whether an assignment to `name` at `at` rebinds a local this
    /// pass may version: one bound earlier in the text and read by
    /// nothing that runs later than where it is written.
    fn is_candidate(&self, name: &str, at: TextSize) -> bool {
        self.facts.locals.contains(name)
            && !self.facts.fixed.contains(name)
            && self.facts.first.get(name).is_some_and(|&first| first < at)
    }

    fn fresh(&mut self, name: &str, at: TextSize) -> String {
        let base = name.split('$').next().unwrap_or(name).to_string();
        let n = self.next.entry(base.clone()).or_default();
        *n += 1;
        let fresh = format!("{base}${n}");
        self.facts.first.insert(fresh.clone(), at);
        self.facts.locals.insert(fresh.clone());
        fresh
    }
}

/// What a function body does with names, gathered before any renaming.
#[derive(Default)]
struct Facts {
    /// The earliest offset each name occurs at; a parameter's is zero.
    first: HashMap<String, TextSize>,
    /// Names the body stores to, plus its parameters.
    locals: HashSet<String>,
    /// Names that keep one binding throughout.
    fixed: HashSet<String>,
    /// How many bodies that run later enclose the visit.
    depth: usize,
}

impl Facts {
    fn later(&mut self, visit: impl FnOnce(&mut Self)) {
        self.depth += 1;
        visit(self);
        self.depth -= 1;
    }
}

impl<'a> Visitor<'a> for Facts {
    fn visit_stmt(&mut self, s: &'a py::Stmt) {
        match s {
            py::Stmt::FunctionDef(f) => {
                self.fixed.insert(f.name.to_string());
                self.later(|v| walk_stmt_ref(v, s));
            }
            py::Stmt::ClassDef(c) => {
                self.fixed.insert(c.name.to_string());
                self.later(|v| walk_stmt_ref(v, s));
            }
            py::Stmt::Match(_) => self.later(|v| walk_stmt_ref(v, s)),
            py::Stmt::Global(g) => self.fixed.extend(g.names.iter().map(|n| n.to_string())),
            py::Stmt::Nonlocal(g) => self.fixed.extend(g.names.iter().map(|n| n.to_string())),
            py::Stmt::Delete(_) => self.later(|v| walk_stmt_ref(v, s)),
            py::Stmt::Import(i) => {
                for a in &i.names {
                    self.fixed
                        .insert(a.asname.as_ref().unwrap_or(&a.name).to_string());
                }
            }
            py::Stmt::ImportFrom(i) => {
                for a in &i.names {
                    self.fixed
                        .insert(a.asname.as_ref().unwrap_or(&a.name).to_string());
                }
            }
            _ => walk_stmt_ref(self, s),
        }
    }

    fn visit_except_handler(&mut self, h: &'a py::ExceptHandler) {
        let py::ExceptHandler::ExceptHandler(handler) = h;
        if let Some(name) = &handler.name {
            self.fixed.insert(name.to_string());
        }
        ruff_python_ast::visitor::walk_except_handler(self, h);
    }

    fn visit_expr(&mut self, e: &'a py::Expr) {
        match e {
            py::Expr::Name(n) => {
                let name = n.id.as_str();
                let at = n.range.start();
                match self.first.get(name) {
                    Some(&first) if first <= at => {}
                    _ => {
                        self.first.insert(name.to_string(), at);
                    }
                }
                if n.ctx.is_store() {
                    self.locals.insert(name.to_string());
                }
                if self.depth > 0 {
                    self.fixed.insert(name.to_string());
                }
            }
            py::Expr::Lambda(_) => self.later(|v| walk_expr_ref(v, e)),
            py::Expr::Named(n) => {
                self.later(|v| v.visit_expr(&n.target));
                self.visit_expr(&n.value);
            }
            // The outermost iterable runs where the expression is
            // written; the rest runs when the generator is driven.
            py::Expr::Generator(g) => {
                let (first, rest) = g.generators.split_first().expect("a generator loops");
                self.visit_expr(&first.iter);
                self.later(|v| {
                    v.visit_expr(&first.target);
                    for cond in &first.ifs {
                        v.visit_expr(cond);
                    }
                    for comp in rest {
                        v.visit_comprehension(comp);
                    }
                    v.visit_expr(&g.elt);
                });
            }
            _ => walk_expr_ref(self, e),
        }
    }

    /// A comprehension's targets are its own; the rest is evaluated in
    /// place.
    fn visit_comprehension(&mut self, c: &'a py::Comprehension) {
        self.later(|v| v.visit_expr(&c.target));
        self.visit_expr(&c.iter);
        for cond in &c.ifs {
            self.visit_expr(cond);
        }
    }
}

/// Whether `name` occurs anywhere in the statements.
fn occurs_in(stmts: &[py::Stmt], name: &str) -> bool {
    struct Occurs<'n> {
        name: &'n str,
        found: bool,
    }
    impl<'a> Visitor<'a> for Occurs<'_> {
        fn visit_stmt(&mut self, s: &'a py::Stmt) {
            if !self.found {
                walk_stmt_ref(self, s);
            }
        }
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if self.found {
                return;
            }
            if let py::Expr::Name(n) = e
                && n.id.as_str() == self.name
            {
                self.found = true;
                return;
            }
            walk_expr_ref(self, e);
        }
    }
    let mut o = Occurs { name, found: false };
    for s in stmts {
        o.visit_stmt(s);
        if o.found {
            return true;
        }
    }
    false
}

/// Renames every occurrence of one name, stores and loads alike.
struct Renamer {
    from: String,
    to: String,
}

impl Transformer for Renamer {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        walk_stmt(self, s);
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        if let py::Expr::Name(n) = e
            && n.id.as_str() == self.from
        {
            n.id = py::name::Name::new(&self.to);
            return;
        }
        walk_expr(self, e);
    }
}

#[cfg(test)]
mod tests {
    use super::rewrite;
    use ruff_python_ast as py;

    fn names_in(src: &str) -> Vec<String> {
        let mut module = ruff_python_parser::parse_module(src).unwrap().into_syntax();
        let mut body: Vec<py::Stmt> = module.body.drain(..).collect();
        rewrite(&mut body);
        struct Names(Vec<String>);
        impl<'a> ruff_python_ast::visitor::Visitor<'a> for Names {
            fn visit_expr(&mut self, e: &'a py::Expr) {
                if let py::Expr::Name(n) = e {
                    self.0.push(n.id.to_string());
                }
                ruff_python_ast::visitor::walk_expr(self, e);
            }
        }
        let mut names = Names(Vec::new());
        for s in &body {
            ruff_python_ast::visitor::Visitor::visit_stmt(&mut names, s);
        }
        names.0
    }

    #[test]
    fn a_straight_line_rebinding_is_a_fresh_version() {
        let names = names_in("def f(data):\n    data = g(data)\n    return h(data)\n");
        assert_eq!(names, ["g", "data", "data$1", "h", "data$1"]);
    }

    #[test]
    fn a_rebinding_in_a_loop_keeps_its_name() {
        let names = names_in("def f(x):\n    for i in r:\n        x = g(x)\n    return x\n");
        assert!(names.iter().all(|n| !n.contains('$')), "{names:?}");
    }

    #[test]
    fn a_rebinding_a_later_read_can_reach_keeps_its_name() {
        let names = names_in("def f(x, c):\n    if c:\n        x = g(x)\n    return x\n");
        assert!(names.iter().all(|n| !n.contains('$')), "{names:?}");
        let names =
            names_in("def f(x, c):\n    if c:\n        x = g(x)\n        return x\n    return 0\n");
        assert_eq!(names, ["c", "g", "x", "x$1", "x$1"]);
    }

    #[test]
    fn a_name_a_later_body_reads_keeps_one_binding() {
        for src in [
            "def f(x):\n    k = lambda: x\n    x = g(x)\n    return k()\n",
            "def f(x):\n    k = (x for _ in r)\n    x = g(x)\n    return list(k)\n",
            "def f(x):\n    def k():\n        return x\n    x = g(x)\n    return k()\n",
        ] {
            let names = names_in(src);
            assert!(names.iter().all(|n| !n.contains('$')), "{src}: {names:?}");
        }
        // The outermost iterable is read where the expression is.
        let names = names_in("def f(x):\n    x = g(x)\n    return list(y for y in x)\n");
        assert!(names.contains(&"x$1".to_string()), "{names:?}");
    }

    #[test]
    fn a_second_rebinding_is_the_next_version() {
        let names = names_in("def f(x):\n    x = g(x)\n    x = h(x)\n    return x\n");
        assert_eq!(names, ["g", "x", "x$1", "h", "x$1", "x$2", "x$2"]);
    }
}
