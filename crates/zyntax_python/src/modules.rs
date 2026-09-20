//! A program of several files, made into one body. Each imported
//! module's top-level names are qualified with the module's name
//! (`helper$f`), every reference to them follows, and the module's body
//! runs ahead of the one that imports it. What remains is a single module
//! the rest of the frontend compiles as it compiles any other.

use crate::scope::Scope;
use crate::stdlib;
use crate::{Error, Result};
use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use ruff_text_size::Ranged;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cell::RefCell;

/// Finds a module's source by its dotted name, when it exists.
pub type Resolver<'a> = dyn Fn(&str) -> Option<String> + 'a;

/// Standard modules written in Python and compiled with the program,
/// found when the program's own files do not have the name.
pub(crate) fn bundled(module: &str) -> Option<&'static str> {
    match module {
        "decimal" => Some(include_str!("../stdlib/decimal.py")),
        _ => None,
    }
}

/// A module's source: the program's own file, or the bundled one.
fn source_of(resolve: &Resolver<'_>, module: &str) -> Option<String> {
    resolve(module).or_else(|| bundled(module).map(str::to_string))
}

/// The qualified name of `member` in `module`.
pub(crate) fn qualified(module: &str, member: &str) -> String {
    format!("{}${member}", module.replace('.', "$"))
}

/// What one module imports from the program's own modules.
#[derive(Default)]
struct UserImports {
    /// Alias (or the module's own name) to the module.
    modules: HashMap<String, String>,
    /// Local name to the qualified name it stands for.
    names: HashMap<String, String>,
}

/// A program linked into one body.
pub(crate) struct Linked {
    /// Every statement with the module it was written in, `None` for
    /// the main file, so what is reported about it can name the file.
    pub(crate) statements: Vec<(py::Stmt, Option<String>)>,
    /// The modules loaded, by name with their source, in load order.
    pub(crate) modules: Vec<(String, String)>,
}

/// Link `main`'s body with every module it imports, transitively.
/// Imported modules come first, each once, in the order first reached.
pub(crate) fn link(main: Vec<py::Stmt>, resolve: &Resolver<'_>) -> Result<Linked> {
    let mut linker = Linker {
        resolve,
        done: HashSet::default(),
        in_progress: Vec::new(),
        out: Vec::new(),
        sources: Vec::new(),
        reads: names_read(&main),
    };
    let mut main = main;
    let imports = linker.link_imports(&mut main)?;
    let main_scope = Scope::of_body(Vec::new(), &main);
    Qualifier::new(None, &main_scope, imports).run(&mut main);
    let mut statements = linker.out;
    statements.extend(main.into_iter().map(|s| (s, None)));
    Ok(Linked {
        statements,
        modules: linker.sources,
    })
}

struct Linker<'r> {
    resolve: &'r Resolver<'r>,
    done: HashSet<String>,
    in_progress: Vec<String>,
    out: Vec<(py::Stmt, Option<String>)>,
    sources: Vec<(String, String)>,
    /// Every name the body being linked reads anywhere, so an import
    /// nothing reads is known to be one.
    reads: HashSet<String>,
}

impl Linker<'_> {
    /// Load the program's own modules `body` imports, replacing each
    /// import statement with `pass`, and report what was imported.
    fn link_imports(&mut self, body: &mut [py::Stmt]) -> Result<UserImports> {
        let mut imports = UserImports::default();
        for stmt in body.iter_mut() {
            match stmt {
                py::Stmt::Import(i) => {
                    let mut ours = false;
                    for alias in &i.names {
                        let module = alias.name.id.as_str();
                        if stdlib::is_known(module) {
                            continue;
                        }
                        // A module the program never reads through its
                        // name is imported for nothing; one that is not
                        // here is then no loss.
                        let local = alias
                            .asname
                            .as_ref()
                            .map(|a| a.id.as_str())
                            .unwrap_or(module);
                        if source_of(self.resolve, module).is_none()
                            && !self
                                .reads
                                .contains(local.split('.').next().unwrap_or(local))
                        {
                            ours = true;
                            continue;
                        }
                        self.load(module, alias.range())?;
                        let local = alias
                            .asname
                            .as_ref()
                            .map(|a| a.id.to_string())
                            .unwrap_or_else(|| module.to_string());
                        imports.modules.insert(local, module.to_string());
                        ours = true;
                    }
                    if ours {
                        *stmt = pass(stmt.range());
                    }
                }
                py::Stmt::ImportFrom(f) => {
                    let Some(module) = f.module.as_ref().map(|m| m.id.to_string()) else {
                        continue;
                    };
                    if stdlib::is_known(&module) {
                        continue;
                    }
                    for alias in &f.names {
                        let name = alias.name.id.as_str();
                        let at = alias.range();
                        // `from m import *`: every public name the module
                        // binds at its top level.
                        if name == "*" {
                            self.load(&module, at)?;
                            for public in self.public_names(&module, at)? {
                                imports
                                    .names
                                    .insert(public.clone(), qualified(&module, &public));
                            }
                            continue;
                        }
                        let local = alias
                            .asname
                            .as_ref()
                            .map(|a| a.id.to_string())
                            .unwrap_or_else(|| name.to_string());
                        // `from pkg import mod` names a module of the
                        // package when there is one; otherwise a member.
                        let submodule = format!("{module}.{name}");
                        if source_of(self.resolve, &submodule).is_some() {
                            self.load(&submodule, at)?;
                            imports.modules.insert(local, submodule);
                        } else {
                            self.load(&module, at)?;
                            imports.names.insert(local, qualified(&module, name));
                        }
                    }
                    *stmt = pass(stmt.range());
                }
                // Imports inside a function, a branch or a loop take
                // effect when the program starts, like the ones at the
                // top of the file.
                py::Stmt::FunctionDef(d) => {
                    let inner = self.link_imports(&mut d.body)?;
                    imports.modules.extend(inner.modules);
                    imports.names.extend(inner.names);
                }
                // A branch on the interpreter's version is decided here:
                // the one taken stays, under a test of `True`, and the
                // others' imports are never made.
                py::Stmt::If(i) if let Some(taken) = version_branch(i) => {
                    let mut body = match taken {
                        None => Default::default(),
                        Some(0) => std::mem::take(&mut i.body),
                        Some(n) => std::mem::take(&mut i.elif_else_clauses[n - 1].body),
                    };
                    let inner = self.link_imports(&mut body)?;
                    imports.modules.extend(inner.modules);
                    imports.names.extend(inner.names);
                    *i.test = py::Expr::BooleanLiteral(py::ExprBooleanLiteral {
                        node_index: Default::default(),
                        range: i.test.range(),
                        value: true,
                    });
                    i.body = body;
                    i.elif_else_clauses.clear();
                    if i.body.is_empty() {
                        i.body.push(pass(i.range()));
                    }
                }
                py::Stmt::If(i) => {
                    let inner = self.link_imports(&mut i.body)?;
                    imports.modules.extend(inner.modules);
                    imports.names.extend(inner.names);
                    for clause in i.elif_else_clauses.iter_mut() {
                        let inner = self.link_imports(&mut clause.body)?;
                        imports.modules.extend(inner.modules);
                        imports.names.extend(inner.names);
                    }
                }
                py::Stmt::For(f) => {
                    let inner = self.link_imports(&mut f.body)?;
                    imports.modules.extend(inner.modules);
                    imports.names.extend(inner.names);
                }
                py::Stmt::While(w) => {
                    let inner = self.link_imports(&mut w.body)?;
                    imports.modules.extend(inner.modules);
                    imports.names.extend(inner.names);
                }
                _ => {}
            }
        }
        Ok(imports)
    }

    /// Load a module once: parse it, link its own imports, qualify its
    /// names, and append its body to the output.
    fn load(&mut self, module: &str, at: ruff_text_size::TextRange) -> Result<()> {
        if self.done.contains(module) {
            return Ok(());
        }
        if self.in_progress.iter().any(|m| m == module) {
            return Err(Error::unsupported(
                format!("a circular import of `{module}`"),
                &at,
            ));
        }
        let Some(source) = source_of(self.resolve, module) else {
            return Err(Error::unsupported(
                format!("import of module `{module}`, which was not found"),
                &at,
            ));
        };
        let parsed = ruff_python_parser::parse_module(&source)
            .map_err(|e| Error::syntax(e.error.to_string(), e.location).in_module(module))?;
        if let Some(first) = parsed.errors().first() {
            return Err(Error::syntax(first.error.to_string(), first.location).in_module(module));
        }
        self.in_progress.push(module.to_string());
        let mut body: Vec<py::Stmt> = parsed.into_syntax().body.into_iter().collect();
        // What goes wrong inside the module is reported against it.
        let outer_reads = std::mem::replace(&mut self.reads, names_read(&body));
        let imports = self.link_imports(&mut body);
        self.reads = outer_reads;
        let imports = imports.map_err(|e| e.in_module(module))?;
        // `__name__` is the module's own; the assignment binds it at
        // module level so the qualifier renames every read of it.
        let mut with_name: Vec<py::Stmt> =
            ruff_python_parser::parse_module(&format!("__name__ = {module:?}"))
                .expect("an assignment parses")
                .into_syntax()
                .body
                .into_iter()
                .collect();
        with_name.append(&mut body);
        let mut body = with_name;
        let scope = Scope::of_body(Vec::new(), &body);
        Qualifier::new(Some(module), &scope, imports).run(&mut body);
        self.in_progress.pop();
        self.done.insert(module.to_string());
        self.sources.push((module.to_string(), source));
        self.out
            .extend(body.into_iter().map(|s| (s, Some(module.to_string()))));
        Ok(())
    }
}

impl Linker<'_> {
    /// The names a module binds at its top level that `import *` takes:
    /// those in `__all__` when it defines one, else every name not
    /// starting with an underscore.
    fn public_names(&self, module: &str, at: ruff_text_size::TextRange) -> Result<Vec<String>> {
        let Some(source) = source_of(self.resolve, module) else {
            return Err(Error::unsupported(
                format!("import of module `{module}`, which was not found"),
                &at,
            ));
        };
        let parsed = ruff_python_parser::parse_module(&source)
            .map_err(|e| Error::syntax(e.error.to_string(), e.location).in_module(module))?;
        let body: Vec<py::Stmt> = parsed.into_syntax().body.into_iter().collect();
        for s in &body {
            if let py::Stmt::Assign(a) = s
                && let [py::Expr::Name(target)] = a.targets.as_slice()
                && target.id.as_str() == "__all__"
                && let py::Expr::List(items) = &*a.value
            {
                return Ok(items
                    .elts
                    .iter()
                    .filter_map(|e| match e {
                        py::Expr::StringLiteral(s) => Some(s.value.to_str().to_string()),
                        _ => None,
                    })
                    .collect());
            }
        }
        let scope = Scope::of_body(Vec::new(), &body);
        let mut names: Vec<String> = scope
            .bound
            .iter()
            .chain(&scope.classes)
            .filter(|n| !n.starts_with('_'))
            .cloned()
            .collect();
        names.sort();
        Ok(names)
    }
}

/// Every name read anywhere in `body`, nested bodies included.
fn names_read(body: &[py::Stmt]) -> HashSet<String> {
    use ruff_python_ast::visitor::{Visitor, walk_expr};
    #[derive(Default)]
    struct Reads(HashSet<String>);
    impl<'a> Visitor<'a> for Reads {
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Name(n) = e
                && n.ctx.is_load()
            {
                self.0.insert(n.id.to_string());
            }
            walk_expr(self, e);
        }
    }
    let mut reads = Reads::default();
    for s in body {
        reads.visit_stmt(s);
    }
    reads.0
}

/// The Python this frontend speaks, as `sys.version_info` reports it.
const VERSION: [i64; 3] = [3, 12, 0];

/// Which branch of `i` runs, when its tests compare `sys.version_info`
/// or one of its parts to literals: the index of the clause taken, the
/// `if` being 0, or `None` for none of them. A test of any other shape
/// leaves the statement alone.
fn version_branch(i: &py::StmtIf) -> Option<Option<usize>> {
    let mut tests = vec![Some(&*i.test)];
    for clause in &i.elif_else_clauses {
        tests.push(clause.test.as_ref());
    }
    for (n, test) in tests.iter().enumerate() {
        match test {
            None => return Some(Some(n)),
            Some(test) => {
                if version_test(test)? {
                    return Some(Some(n));
                }
            }
        }
    }
    Some(None)
}

/// `sys.version_info[k] <op> literal`, `sys.version_info <op> (a, b)`
/// or `sys.version_info.major <op> literal`, decided for [`VERSION`].
fn version_test(e: &py::Expr) -> Option<bool> {
    let py::Expr::Compare(c) = e else {
        return None;
    };
    if c.ops.len() != 1 {
        return None;
    }
    let is_version_info = |e: &py::Expr| {
        matches!(e, py::Expr::Attribute(a)
            if a.attr.as_str() == "version_info" && matches!(&*a.value, py::Expr::Name(n) if n.id.as_str() == "sys"))
    };
    let int_of = |e: &py::Expr| match e {
        py::Expr::NumberLiteral(n) => match &n.value {
            py::Number::Int(i) => i.as_i64(),
            _ => None,
        },
        _ => None,
    };
    let (left, right): (Vec<i64>, Vec<i64>) = match &*c.left {
        // sys.version_info[k]
        py::Expr::Subscript(s) if is_version_info(&s.value) => {
            let k = usize::try_from(int_of(&s.slice)?).ok()?;
            (vec![*VERSION.get(k)?], vec![int_of(&c.comparators[0])?])
        }
        // sys.version_info.major / .minor
        py::Expr::Attribute(a) if is_version_info(&a.value) => {
            let k = match a.attr.as_str() {
                "major" => 0,
                "minor" => 1,
                "micro" => 2,
                _ => return None,
            };
            (vec![VERSION[k]], vec![int_of(&c.comparators[0])?])
        }
        // sys.version_info >= (3, 0)
        left if is_version_info(left) => {
            let py::Expr::Tuple(t) = &c.comparators[0] else {
                return None;
            };
            let right: Option<Vec<i64>> = t.elts.iter().map(int_of).collect();
            let right = right?;
            (VERSION[..right.len().min(3)].to_vec(), right)
        }
        _ => return None,
    };
    let order = left.cmp(&right);
    Some(match c.ops[0] {
        py::CmpOp::Lt => order.is_lt(),
        py::CmpOp::LtE => order.is_le(),
        py::CmpOp::Gt => order.is_gt(),
        py::CmpOp::GtE => order.is_ge(),
        py::CmpOp::Eq => order.is_eq(),
        py::CmpOp::NotEq => order.is_ne(),
        _ => return None,
    })
}

fn pass(range: ruff_text_size::TextRange) -> py::Stmt {
    py::Stmt::Pass(py::StmtPass {
        node_index: Default::default(),
        range,
    })
}

/// Rewrites one module's names: its own top-level names to their
/// qualified form when the module has one, and names it imported from
/// other modules to theirs. A name bound by an enclosing function,
/// lambda or comprehension is that scope's and is left alone.
struct Qualifier {
    prefix: Option<String>,
    module_names: HashSet<String>,
    imports: UserImports,
    /// Scopes entered, innermost last. A class body's names are visible
    /// in that body only, not from the methods inside it.
    locals: RefCell<Vec<Frame>>,
}

struct Frame {
    names: HashSet<String>,
    class: bool,
}

impl Qualifier {
    fn new(prefix: Option<&str>, module_scope: &Scope, imports: UserImports) -> Self {
        let mut module_names = module_scope.bound.clone();
        module_names.extend(module_scope.declared_globals());
        module_names.extend(module_scope.classes.iter().cloned());
        Self {
            prefix: prefix.map(str::to_string),
            module_names,
            imports,
            locals: RefCell::new(Vec::new()),
        }
    }

    fn run(&self, body: &mut [py::Stmt]) {
        for stmt in body {
            self.visit_stmt(stmt);
        }
    }

    fn is_local(&self, name: &str) -> bool {
        let mut inside_function = false;
        for frame in self.locals.borrow().iter().rev() {
            if frame.class {
                if inside_function {
                    continue;
                }
            } else {
                inside_function = true;
            }
            if frame.names.contains(name) {
                return true;
            }
        }
        false
    }

    /// What a bare name at this point stands for, if anything else.
    fn rename(&self, name: &str) -> Option<String> {
        if self.is_local(name) {
            return None;
        }
        if let Some(target) = self.imports.names.get(name) {
            return Some(target.clone());
        }
        if let Some(prefix) = &self.prefix
            && self.module_names.contains(name)
        {
            return Some(qualified(prefix, name));
        }
        None
    }

    fn with_scope<R>(&self, names: HashSet<String>, class: bool, f: impl FnOnce() -> R) -> R {
        self.locals.borrow_mut().push(Frame { names, class });
        let out = f();
        self.locals.borrow_mut().pop();
        out
    }

    fn set_name(name: &mut py::Identifier, to: &str) {
        name.id = py::name::Name::new(to);
    }
}

impl Transformer for Qualifier {
    fn visit_stmt(&self, stmt: &mut py::Stmt) {
        match stmt {
            py::Stmt::FunctionDef(f) => {
                if let Some(to) = self.rename(f.name.id.as_str()) {
                    Self::set_name(&mut f.name, &to);
                }
                for d in &mut f.decorator_list {
                    self.visit_decorator(d);
                }
                self.visit_parameters(&mut f.parameters);
                if let Some(returns) = &mut f.returns {
                    self.visit_annotation(returns);
                }
                let scope = Scope::of_function(f);
                let mut locals = scope.bound;
                locals.extend(parameter_names(&f.parameters));
                for g in &scope.globals {
                    locals.remove(g);
                }
                self.with_scope(locals, false, || {
                    for s in &mut f.body {
                        self.visit_stmt(s);
                    }
                });
            }
            py::Stmt::ClassDef(c) => {
                if let Some(to) = self.rename(c.name.id.as_str()) {
                    Self::set_name(&mut c.name, &to);
                }
                if let Some(args) = &mut c.arguments {
                    self.visit_arguments(args);
                }
                // A class body binds its methods; they are not module names.
                let scope = Scope::of_body(Vec::new(), &c.body);
                self.with_scope(scope.bound, true, || {
                    for s in &mut c.body {
                        self.visit_stmt(s);
                    }
                });
            }
            py::Stmt::Global(g) => {
                for name in &mut g.names {
                    if let Some(prefix) = &self.prefix
                        && self.module_names.contains(name.id.as_str())
                    {
                        let to = qualified(prefix, name.id.as_str());
                        Self::set_name(name, &to);
                    }
                }
            }
            _ => walk_stmt(self, stmt),
        }
    }

    fn visit_expr(&self, expr: &mut py::Expr) {
        match expr {
            // `m.x`, or `a.b.x` for `import a.b`, through an imported
            // module is the qualified name.
            py::Expr::Attribute(a) => {
                if let Some(path) = dotted_path(&a.value) {
                    let head = path.split('.').next().unwrap_or(&path);
                    if !self.is_local(head)
                        && let Some(module) = self.imports.modules.get(&path)
                    {
                        let to = qualified(module, a.attr.id.as_str());
                        *expr = py::Expr::Name(py::ExprName {
                            node_index: Default::default(),
                            range: a.range,
                            id: py::name::Name::new(to),
                            ctx: a.ctx,
                        });
                        return;
                    }
                }
                walk_expr(self, expr);
            }
            py::Expr::Name(n) => {
                if let Some(to) = self.rename(n.id.as_str()) {
                    n.id = py::name::Name::new(to);
                }
            }
            py::Expr::Lambda(l) => {
                let mut locals = Scope::of_lambda(l).bound;
                if let Some(ps) = &l.parameters {
                    locals.extend(parameter_names(ps));
                }
                self.with_scope(locals, false, || walk_expr(self, expr));
            }
            py::Expr::ListComp(_)
            | py::Expr::SetComp(_)
            | py::Expr::DictComp(_)
            | py::Expr::Generator(_) => {
                let targets = comprehension_targets(expr);
                self.with_scope(targets, false, || walk_expr(self, expr));
            }
            _ => walk_expr(self, expr),
        }
    }
}

/// `a.b.c` as "a.b.c", when the expression is names all the way down.
fn dotted_path(e: &py::Expr) -> Option<String> {
    match e {
        py::Expr::Name(n) => Some(n.id.to_string()),
        py::Expr::Attribute(a) => Some(format!("{}.{}", dotted_path(&a.value)?, a.attr.id)),
        _ => None,
    }
}

/// Every parameter name, positional, keyword-only, starred or double-starred.
fn parameter_names(ps: &py::Parameters) -> HashSet<String> {
    let mut out: HashSet<String> = ps
        .iter_non_variadic_params()
        .map(|p| p.parameter.name.to_string())
        .collect();
    if let Some(v) = &ps.vararg {
        out.insert(v.name.to_string());
    }
    if let Some(k) = &ps.kwarg {
        out.insert(k.name.to_string());
    }
    out
}

/// The names a comprehension's `for` clauses bind.
fn comprehension_targets(expr: &py::Expr) -> HashSet<String> {
    let generators = match expr {
        py::Expr::ListComp(c) => &c.generators,
        py::Expr::SetComp(c) => &c.generators,
        py::Expr::DictComp(c) => &c.generators,
        py::Expr::Generator(g) => &g.generators,
        _ => return HashSet::default(),
    };
    let mut out = HashSet::default();
    for g in generators {
        collect_target_names(&g.target, &mut out);
    }
    out
}

fn collect_target_names(target: &py::Expr, out: &mut HashSet<String>) {
    match target {
        py::Expr::Name(n) => {
            out.insert(n.id.to_string());
        }
        py::Expr::Tuple(t) => {
            for e in &t.elts {
                collect_target_names(e, out);
            }
        }
        py::Expr::List(l) => {
            for e in &l.elts {
                collect_target_names(e, out);
            }
        }
        py::Expr::Starred(s) => collect_target_names(&s.value, out),
        _ => {}
    }
}
