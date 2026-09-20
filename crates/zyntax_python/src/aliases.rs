//! Class aliases. `Canvas = PpmCanvas` names the class under another
//! name; where that name is assigned nothing else in its scope, every
//! read of it is the class, and the program is rewritten to say so
//! before inference. Classes are not values here, so an alias that
//! could not be resolved this way would be a dynamic value that cannot
//! be called.

use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use ruff_python_ast::visitor::{Visitor, walk_stmt as walk_stmt_ref};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cell::RefCell;

/// Rewrite the aliases in the module body and in every function body.
pub(crate) fn rewrite(body: &mut [py::Stmt]) {
    let classes: HashSet<String> = body
        .iter()
        .filter_map(|s| match s {
            py::Stmt::ClassDef(c) => Some(c.name.to_string()),
            _ => None,
        })
        .collect();
    if classes.is_empty() {
        return;
    }
    // Module-level aliases: names no function writes through `global`.
    let mut written_globally = HashSet::default();
    for s in body.iter() {
        globals_declared(s, &mut written_globally);
    }
    scope(body, &classes, &written_globally);
    for s in body.iter_mut() {
        functions(s, &classes);
    }
}

/// Every function body under `s`, innermost last.
fn functions(s: &mut py::Stmt, classes: &HashSet<String>) {
    match s {
        py::Stmt::FunctionDef(f) => {
            let mut declared = HashSet::default();
            for inner in f.body.iter() {
                globals_declared(inner, &mut declared);
            }
            scope(&mut f.body, classes, &declared);
            for inner in f.body.iter_mut() {
                functions(inner, classes);
            }
        }
        py::Stmt::ClassDef(c) => {
            for inner in c.body.iter_mut() {
                functions(inner, classes);
            }
        }
        py::Stmt::If(i) => {
            for inner in i.body.iter_mut() {
                functions(inner, classes);
            }
            for clause in i.elif_else_clauses.iter_mut() {
                for inner in clause.body.iter_mut() {
                    functions(inner, classes);
                }
            }
        }
        py::Stmt::For(f) => {
            for inner in f.body.iter_mut().chain(f.orelse.iter_mut()) {
                functions(inner, classes);
            }
        }
        py::Stmt::While(w) => {
            for inner in w.body.iter_mut().chain(w.orelse.iter_mut()) {
                functions(inner, classes);
            }
        }
        py::Stmt::With(w) => {
            for inner in w.body.iter_mut() {
                functions(inner, classes);
            }
        }
        py::Stmt::Try(t) => {
            for inner in t.body.iter_mut() {
                functions(inner, classes);
            }
            for h in t.handlers.iter_mut() {
                let py::ExceptHandler::ExceptHandler(h) = h;
                for inner in h.body.iter_mut() {
                    functions(inner, classes);
                }
            }
            for inner in t.orelse.iter_mut().chain(t.finalbody.iter_mut()) {
                functions(inner, classes);
            }
        }
        _ => {}
    }
}

/// The names `global` and `nonlocal` statements anywhere under `s`
/// declare, which are written from elsewhere.
fn globals_declared(s: &py::Stmt, out: &mut HashSet<String>) {
    struct G<'o>(&'o mut HashSet<String>);
    impl<'a> Visitor<'a> for G<'_> {
        fn visit_stmt(&mut self, s: &'a py::Stmt) {
            match s {
                py::Stmt::Global(g) => self.0.extend(g.names.iter().map(|n| n.to_string())),
                py::Stmt::Nonlocal(g) => self.0.extend(g.names.iter().map(|n| n.to_string())),
                _ => walk_stmt_ref(self, s),
            }
        }
    }
    G(out).visit_stmt(s);
}

/// One scope's aliases: `x = C` statements whose `x` nothing else in
/// the scope stores to. Nested function bodies are part of the scope
/// for counting (a closure may write through `nonlocal`, which is in
/// `excluded`) and for renaming, since they read the enclosing name.
fn scope(body: &mut [py::Stmt], classes: &HashSet<String>, excluded: &HashSet<String>) {
    let mut stores: HashMap<String, usize> = HashMap::default();
    for s in body.iter() {
        count_stores(s, &mut stores);
    }
    let mut aliases: HashMap<String, String> = HashMap::default();
    for s in body.iter_mut() {
        if let py::Stmt::Assign(a) = s
            && let [py::Expr::Name(target)] = a.targets.as_slice()
            && let py::Expr::Name(class) = &*a.value
            && classes.contains(class.id.as_str())
            && !classes.contains(target.id.as_str())
            && !excluded.contains(target.id.as_str())
            && stores.get(target.id.as_str()) == Some(&1)
        {
            aliases.insert(target.id.to_string(), class.id.to_string());
            *s = py::Stmt::Pass(py::StmtPass {
                node_index: Default::default(),
                range: a.range,
            });
        }
    }
    if aliases.is_empty() {
        return;
    }
    let renamer = Renamer {
        aliases: RefCell::new(aliases),
    };
    for s in body.iter_mut() {
        renamer.visit_stmt(s);
    }
}

/// How many times each name is stored to under `s`, nested bodies
/// included.
fn count_stores(s: &py::Stmt, out: &mut HashMap<String, usize>) {
    struct C<'o>(&'o mut HashMap<String, usize>);
    impl<'a> Visitor<'a> for C<'_> {
        fn visit_stmt(&mut self, s: &'a py::Stmt) {
            match s {
                // A def or class binds its name; an import binds too.
                py::Stmt::FunctionDef(f) => *self.0.entry(f.name.to_string()).or_default() += 1,
                py::Stmt::ClassDef(c) => *self.0.entry(c.name.to_string()).or_default() += 1,
                py::Stmt::Import(i) => {
                    for a in &i.names {
                        let name = a.asname.as_ref().unwrap_or(&a.name).to_string();
                        *self.0.entry(name).or_default() += 1;
                    }
                }
                py::Stmt::ImportFrom(i) => {
                    for a in &i.names {
                        let name = a.asname.as_ref().unwrap_or(&a.name).to_string();
                        *self.0.entry(name).or_default() += 1;
                    }
                }
                _ => {}
            }
            walk_stmt_ref(self, s);
        }
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Name(n) = e
                && n.ctx.is_store()
            {
                *self.0.entry(n.id.to_string()).or_default() += 1;
            }
            ruff_python_ast::visitor::walk_expr(self, e);
        }
        fn visit_parameter(&mut self, p: &'a py::Parameter) {
            *self.0.entry(p.name.to_string()).or_default() += 1;
        }
    }
    C(out).visit_stmt(s);
}

/// Replaces reads of an alias by the class it names. A nested function
/// whose parameter shadows the alias keeps its own name.
struct Renamer {
    aliases: RefCell<HashMap<String, String>>,
}

impl Transformer for Renamer {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        if let py::Stmt::FunctionDef(f) = s {
            let shadowed: Vec<(String, String)> = f
                .parameters
                .iter()
                .filter_map(|p| {
                    let name = p.name().to_string();
                    self.aliases
                        .borrow_mut()
                        .remove(&name)
                        .map(|class| (name, class))
                })
                .collect();
            walk_stmt(self, s);
            self.aliases.borrow_mut().extend(shadowed);
            return;
        }
        walk_stmt(self, s);
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        if let py::Expr::Name(n) = e
            && n.ctx.is_load()
            && let Some(class) = self.aliases.borrow().get(n.id.as_str())
        {
            n.id = py::name::Name::new(class);
            return;
        }
        walk_expr(self, e);
    }
}
