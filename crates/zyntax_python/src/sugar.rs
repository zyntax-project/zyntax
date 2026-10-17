//! Class-level sugar rewritten away before inference, so the rest of
//! the frontend sees only the shapes it models:
//!
//! - `class C(list): pass` names the builtin; `C` becomes `list`.
//! - `super(C, self).m(...)` is `super().m(...)`, and `super().__init__()`
//!   in a class with no base does nothing.
//! - `@classmethod` and `@staticmethod` methods are module functions
//!   named `C$m`; `cls` in a classmethod is the class, and `C.m(...)`
//!   or `cls.m(...)` calls the function.
//! - `self.__class__.X` in a method of `C` is `C.X`.
//! - `__rmul__ = __mul__` in a class body, naming a plain method the
//!   body defined above, is that method again under the new name.
//! - `map(f, xs)` with `f` a name is `[f(x) for x in xs]`, so the
//!   elements are typed as the calls are.
//! - `functools.reduce(f, xs, start)` is a call of a module function
//!   that folds `xs` in a loop: `acc op x` for `f` an `operator`
//!   function, `f(acc, x)` otherwise, so the fold is typed as the
//!   operation is.
//! - `if p is None: p = e` at the top of a function, `p` a parameter,
//!   is `p$set = e if p is None else p`, and `p` is `p$set` from there
//!   on: the parameter is never rebound, and the name holds the default
//!   or what was passed, never None.

use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use ruff_text_size::Ranged;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cell::{Cell, RefCell};

const BUILTIN_BASES: &[&str] = &["list", "dict", "set", "object"];

/// `origins` names each statement's module, and grows with `body`.
pub(crate) fn rewrite(body: &mut Vec<py::Stmt>, origins: &mut Vec<Option<String>>) {
    // Classes with no base but `object`, for `super().__init__()`.
    let mut rootless: HashSet<String> = HashSet::default();
    // Builtin aliases: `class C(list): pass`.
    let mut aliases: HashMap<String, String> = HashMap::default();
    // Static and class methods, by class then method.
    let mut statics: HashMap<String, HashSet<String>> = HashMap::default();
    let mut hoisted: Vec<(py::Stmt, Option<String>)> = Vec::new();
    for (i, s) in body.iter_mut().enumerate() {
        let py::Stmt::ClassDef(c) = s else {
            continue;
        };
        let origin = origins.get(i).cloned().flatten();
        let class = c.name.to_string();
        let bases = c.bases();
        let builtin_base = match bases {
            [py::Expr::Name(b)] if BUILTIN_BASES.contains(&b.id.as_str()) => Some(b.id.to_string()),
            [] => Some("object".to_string()),
            _ => None,
        };
        if let Some(base) = &builtin_base {
            rootless.insert(class.clone());
            if base != "object" && body_is_empty(&c.body) {
                aliases.insert(class.clone(), base.clone());
                *s = pass(c.range());
                continue;
            }
        }
        let mut kept = Vec::new();
        // The plain methods defined so far, for an alias of one.
        let mut defined: HashMap<String, py::StmtFunctionDef> = HashMap::default();
        for m in std::mem::take(&mut c.body) {
            let alias = match &m {
                py::Stmt::Assign(a)
                    if let [py::Expr::Name(alias)] = &a.targets[..]
                        && let py::Expr::Name(of) = &*a.value
                        && let Some(def) = defined.get(of.id.as_str()) =>
                {
                    let mut def = def.clone();
                    def.name = py::Identifier::new(alias.id.as_str(), alias.range);
                    Some(def)
                }
                _ => None,
            };
            if let Some(def) = alias {
                defined.insert(def.name.to_string(), def.clone());
                kept.push(py::Stmt::FunctionDef(def));
                continue;
            }
            match m {
                py::Stmt::FunctionDef(f) if f.decorator_list.is_empty() => {
                    defined.insert(f.name.to_string(), f.clone());
                    kept.push(py::Stmt::FunctionDef(f));
                }
                py::Stmt::FunctionDef(mut f) if is_static_or_class_method(&f) => {
                    let is_classmethod = decorated_with(&f, "classmethod");
                    f.decorator_list.clear();
                    if is_classmethod {
                        let cls = f
                            .parameters
                            .args
                            .first()
                            .map(|p| p.parameter.name.to_string());
                        if let Some(cls) = cls {
                            f.parameters.args.remove(0);
                            let renamer = ClsRenamer {
                                cls,
                                class: class.clone(),
                            };
                            for stmt in f.body.iter_mut() {
                                renamer.visit_stmt(stmt);
                            }
                        }
                    }
                    statics
                        .entry(class.clone())
                        .or_default()
                        .insert(f.name.to_string());
                    f.name =
                        py::Identifier::new(format!("{class}${}", f.name.as_str()), f.name.range());
                    hoisted.push((py::Stmt::FunctionDef(f), origin.clone()));
                }
                other => kept.push(other),
            }
        }
        c.body = kept.into_iter().collect();
    }
    let rewriter = Rewriter {
        rootless,
        aliases,
        statics,
        method_of: RefCell::new(None),
        map_is_builtin: !binds_module_name(body, "map"),
        map_items: Cell::new(0),
        folds: Folds::of(body, origins),
        origin: RefCell::new(None),
        folders: RefCell::new(Vec::new()),
    };
    for (s, origin) in body.iter_mut().zip(origins.iter()) {
        rewriter.origin.replace(origin.clone());
        rewriter.visit_stmt(s);
    }
    for f in rewriter.folders.take() {
        body.push(folder(&f.name, f.op, f.range));
        origins.push(f.origin);
    }
    // Hoisted functions go last: module functions are collected by
    // name, not by position.
    for (mut s, origin) in hoisted {
        rewriter.visit_stmt(&mut s);
        body.push(s);
        origins.push(origin);
    }
}

fn pass(range: ruff_text_size::TextRange) -> py::Stmt {
    py::Stmt::Pass(py::StmtPass {
        node_index: Default::default(),
        range,
    })
}

fn body_is_empty(body: &[py::Stmt]) -> bool {
    body.iter().all(|s| match s {
        py::Stmt::Pass(_) => true,
        py::Stmt::Expr(e) => matches!(&*e.value, py::Expr::StringLiteral(_)),
        _ => false,
    })
}

fn decorated_with(f: &py::StmtFunctionDef, name: &str) -> bool {
    f.decorator_list
        .iter()
        .any(|d| matches!(&d.expression, py::Expr::Name(n) if n.id.as_str() == name))
}

fn is_static_or_class_method(f: &py::StmtFunctionDef) -> bool {
    f.decorator_list.len() == 1
        && (decorated_with(f, "classmethod") || decorated_with(f, "staticmethod"))
}

fn name_expr(id: &str, range: ruff_text_size::TextRange) -> py::Expr {
    py::Expr::Name(py::ExprName {
        node_index: Default::default(),
        range,
        id: py::name::Name::new(id),
        ctx: py::ExprContext::Load,
    })
}

/// `cls` in a classmethod's body is the class.
struct ClsRenamer {
    cls: String,
    class: String,
}

impl Transformer for ClsRenamer {
    fn visit_expr(&self, e: &mut py::Expr) {
        if let py::Expr::Name(n) = e
            && n.id.as_str() == self.cls
        {
            n.id = py::name::Name::new(&self.class);
            return;
        }
        walk_expr(self, e);
    }
}

struct Rewriter {
    rootless: HashSet<String>,
    aliases: HashMap<String, String>,
    statics: HashMap<String, HashSet<String>>,
    /// The class and `self` name of the method being walked.
    method_of: RefCell<Option<(String, String)>>,
    /// No module-level binding shadows the builtin `map`.
    map_is_builtin: bool,
    /// Comprehension targets made for `map` calls so far.
    map_items: Cell<u32>,
    /// What names `functools.reduce` and the `operator` functions, by
    /// module.
    folds: Folds,
    /// The module of the statement being walked.
    origin: RefCell<Option<String>>,
    /// The fold functions calls were rewritten to.
    folders: RefCell<Vec<Folder>>,
}

/// A fold function a `reduce` call was rewritten to call.
struct Folder {
    name: String,
    /// The module the call is in, which the function joins.
    origin: Option<String>,
    /// The operator folded with; `None` for a function passed.
    op: Option<py::Operator>,
    /// Where the first such call is.
    range: ruff_text_size::TextRange,
}

/// The modules whose `reduce` is `functools.reduce`, the modules that
/// import `functools` and `operator` as themselves, and the `operator`
/// functions each module imports by name.
#[derive(Default)]
struct Folds {
    reduce: HashSet<Option<String>>,
    functools: HashSet<Option<String>>,
    operator: HashSet<Option<String>>,
    operator_names: HashMap<(Option<String>, String), py::Operator>,
}

impl Folds {
    fn of(body: &[py::Stmt], origins: &[Option<String>]) -> Folds {
        let mut f = Folds::default();
        let mut bound: HashMap<(Option<String>, String), u32> = HashMap::default();
        for (s, origin) in body.iter().zip(origins) {
            f.scan(s, origin, &mut bound);
        }
        // A name bound more than once may not be the import.
        let once = |origin: &Option<String>, name: &str, bound: &HashMap<_, u32>| {
            bound.get(&(origin.clone(), name.to_string())) == Some(&1)
        };
        f.reduce.retain(|o| once(o, "reduce", &bound));
        f.functools.retain(|o| once(o, "functools", &bound));
        f.operator.retain(|o| once(o, "operator", &bound));
        f.operator_names.retain(|(o, n), _| once(o, n, &bound));
        f
    }

    /// Note what one module-level statement binds, looking into the
    /// branches of a module-level `if` or `try`.
    fn scan(
        &mut self,
        s: &py::Stmt,
        origin: &Option<String>,
        bound: &mut HashMap<(Option<String>, String), u32>,
    ) {
        let mut bind =
            |name: &str| *bound.entry((origin.clone(), name.to_string())).or_default() += 1;
        match s {
            py::Stmt::Import(i) => {
                for a in &i.names {
                    let name = a.asname.as_ref().map_or(a.name.as_str(), |n| n.as_str());
                    bind(name.split('.').next().unwrap_or(name));
                    if a.asname.is_none() && a.name.as_str() == "functools" {
                        self.functools.insert(origin.clone());
                    }
                    if a.asname.is_none() && a.name.as_str() == "operator" {
                        self.operator.insert(origin.clone());
                    }
                }
            }
            py::Stmt::ImportFrom(i) => {
                let from = i.module.as_ref().map(|m| m.as_str());
                for a in &i.names {
                    let name = a.asname.as_ref().map_or(a.name.as_str(), |n| n.as_str());
                    bind(name);
                    match from {
                        Some("functools") if a.name.as_str() == "reduce" && name == "reduce" => {
                            self.reduce.insert(origin.clone());
                        }
                        Some("operator") => {
                            if let Some(op) = operator_function(a.name.as_str()) {
                                self.operator_names
                                    .insert((origin.clone(), name.to_string()), op);
                            }
                        }
                        _ => {}
                    }
                }
            }
            py::Stmt::FunctionDef(d) => bind(d.name.as_str()),
            py::Stmt::ClassDef(c) => bind(c.name.as_str()),
            py::Stmt::Assign(a) => {
                for t in &a.targets {
                    let mut names = Vec::new();
                    target_names(t, &mut names);
                    for n in names {
                        bind(n);
                    }
                }
            }
            py::Stmt::AnnAssign(a) => {
                if let py::Expr::Name(n) = &*a.target {
                    bind(n.id.as_str());
                }
            }
            py::Stmt::AugAssign(a) => {
                if let py::Expr::Name(n) = &*a.target {
                    bind(n.id.as_str());
                }
            }
            py::Stmt::For(l) => {
                let mut names = Vec::new();
                target_names(&l.target, &mut names);
                for n in names {
                    bind(n);
                }
                for s in l.body.iter().chain(&l.orelse) {
                    self.scan(s, origin, bound);
                }
            }
            py::Stmt::If(i) => {
                for s in &i.body {
                    self.scan(s, origin, bound);
                }
                for clause in &i.elif_else_clauses {
                    for s in &clause.body {
                        self.scan(s, origin, bound);
                    }
                }
            }
            py::Stmt::Try(t) => {
                for s in t.body.iter().chain(&t.orelse).chain(&t.finalbody) {
                    self.scan(s, origin, bound);
                }
                for h in &t.handlers {
                    let py::ExceptHandler::ExceptHandler(h) = h;
                    for s in &h.body {
                        self.scan(s, origin, bound);
                    }
                }
            }
            py::Stmt::With(w) => {
                for s in &w.body {
                    self.scan(s, origin, bound);
                }
            }
            py::Stmt::While(w) => {
                for s in w.body.iter().chain(&w.orelse) {
                    self.scan(s, origin, bound);
                }
            }
            _ => {}
        }
    }

    /// Whether `func` names `functools.reduce` in `origin`.
    fn is_reduce(&self, func: &py::Expr, origin: &Option<String>) -> bool {
        match func {
            py::Expr::Name(n) => n.id.as_str() == "reduce" && self.reduce.contains(origin),
            py::Expr::Attribute(a) => {
                a.attr.as_str() == "reduce"
                    && is_name(&a.value, "functools")
                    && self.functools.contains(origin)
            }
            _ => false,
        }
    }

    /// The operator `f` names in `origin`, when it is an `operator`
    /// function.
    fn operator_of(&self, f: &py::Expr, origin: &Option<String>) -> Option<py::Operator> {
        match f {
            py::Expr::Name(n) => self
                .operator_names
                .get(&(origin.clone(), n.id.to_string()))
                .copied(),
            py::Expr::Attribute(a)
                if is_name(&a.value, "operator") && self.operator.contains(origin) =>
            {
                operator_function(a.attr.as_str())
            }
            _ => None,
        }
    }
}

/// The names an assignment target binds.
fn target_names<'e>(t: &'e py::Expr, out: &mut Vec<&'e str>) {
    match t {
        py::Expr::Name(n) => out.push(n.id.as_str()),
        py::Expr::Tuple(t) => t.elts.iter().for_each(|e| target_names(e, out)),
        py::Expr::List(l) => l.elts.iter().for_each(|e| target_names(e, out)),
        py::Expr::Starred(s) => target_names(&s.value, out),
        _ => {}
    }
}

/// The `operator` module's function for `op`.
fn op_function_name(op: py::Operator) -> &'static str {
    match op {
        py::Operator::Add => "add",
        py::Operator::Sub => "sub",
        py::Operator::Mult => "mul",
        py::Operator::Div => "truediv",
        py::Operator::FloorDiv => "floordiv",
        py::Operator::Mod => "mod",
        py::Operator::Pow => "pow",
        py::Operator::MatMult => "matmul",
        py::Operator::BitAnd => "and_",
        py::Operator::BitOr => "or_",
        py::Operator::BitXor => "xor",
        py::Operator::LShift => "lshift",
        py::Operator::RShift => "rshift",
    }
}

/// The binary operator the `operator` module's function `name` is.
fn operator_function(name: &str) -> Option<py::Operator> {
    Some(match name {
        "add" => py::Operator::Add,
        "sub" => py::Operator::Sub,
        "mul" => py::Operator::Mult,
        "truediv" => py::Operator::Div,
        "floordiv" => py::Operator::FloorDiv,
        "mod" => py::Operator::Mod,
        "pow" => py::Operator::Pow,
        "and_" => py::Operator::BitAnd,
        "or_" => py::Operator::BitOr,
        "xor" => py::Operator::BitXor,
        "lshift" => py::Operator::LShift,
        "rshift" => py::Operator::RShift,
        _ => return None,
    })
}

/// The module function a `reduce` call folding with `op` (or, for
/// `None`, with a function it is passed) is rewritten to call, its
/// every node at `range`.
fn folder(name: &str, op: Option<py::Operator>, range: ruff_text_size::TextRange) -> py::Stmt {
    let source = match op {
        Some(op) => format!(
            "def f(xs, acc):\n    for x in xs:\n        acc = acc {} x\n    return acc\n",
            op.as_str()
        ),
        None => "def f(f, xs, acc):\n    for x in xs:\n        acc = f(acc, x)\n    return acc\n"
            .to_string(),
    };
    let mut parsed = ruff_python_parser::parse_module(&source)
        .expect("a fold parses")
        .into_syntax();
    let mut def = parsed.body.remove(0);
    if let py::Stmt::FunctionDef(f) = &mut def {
        f.name = py::Identifier::new(name, range);
    }
    Relocate(range).visit_stmt(&mut def);
    def
}

/// Every node of what it walks at one range.
struct Relocate(ruff_text_size::TextRange);

impl Transformer for Relocate {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        match s {
            py::Stmt::FunctionDef(f) => {
                f.range = self.0;
                f.parameters.range = self.0;
                for p in f.parameters.args.iter_mut() {
                    p.range = self.0;
                    p.parameter.range = self.0;
                    p.parameter.name = py::Identifier::new(p.parameter.name.as_str(), self.0);
                }
            }
            py::Stmt::For(l) => l.range = self.0,
            py::Stmt::Assign(a) => a.range = self.0,
            py::Stmt::Return(r) => r.range = self.0,
            _ => {}
        }
        walk_stmt(self, s);
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        py::relocate::relocate_expr(e, self.0);
    }
}

impl Rewriter {
    fn is_static(&self, class: &str, method: &str) -> bool {
        self.statics
            .get(class)
            .is_some_and(|methods| methods.contains(method))
    }
}

impl Transformer for Rewriter {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        match s {
            py::Stmt::FunctionDef(f) => {
                bind_none_defaults(f);
                walk_stmt(self, s);
            }
            py::Stmt::ClassDef(c) => {
                let class = c.name.to_string();
                for m in c.body.iter_mut() {
                    if let py::Stmt::FunctionDef(f) = m {
                        bind_none_defaults(f);
                        let this = f
                            .parameters
                            .args
                            .first()
                            .map(|p| p.parameter.name.to_string())
                            .unwrap_or_default();
                        let was = self.method_of.replace(Some((class.clone(), this)));
                        // `super().__init__()` with no base to reach.
                        if self.rootless.contains(&class) {
                            for stmt in f.body.iter_mut() {
                                if let py::Stmt::Expr(e) = stmt
                                    && let py::Expr::Call(call) = &*e.value
                                    && let py::Expr::Attribute(a) = &*call.func
                                    && a.attr.as_str() == "__init__"
                                    && is_super(&a.value)
                                {
                                    *stmt = pass(e.range());
                                }
                            }
                        }
                        walk_stmt(self, m);
                        self.method_of.replace(was);
                    } else {
                        walk_stmt(self, m);
                    }
                }
            }
            _ => walk_stmt(self, s),
        }
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        match e {
            // `C` for `class C(list): pass` is `list`.
            py::Expr::Name(n) if n.ctx.is_load() => {
                if let Some(builtin) = self.aliases.get(n.id.as_str()) {
                    n.id = py::name::Name::new(builtin);
                }
            }
            // `super(C, self)` is `super()`.
            py::Expr::Call(c) if is_name(&c.func, "super") && !c.arguments.args.is_empty() => {
                c.arguments.args = Vec::new().into_boxed_slice();
            }
            // `C.m(...)` for a static or class method is `C$m(...)`.
            py::Expr::Call(c)
                if let py::Expr::Attribute(a) = &*c.func
                    && let py::Expr::Name(class) = &*a.value
                    && self.is_static(class.id.as_str(), a.attr.as_str()) =>
            {
                let range = c.func.range();
                let target = format!("{}${}", class.id.as_str(), a.attr.as_str());
                *c.func = name_expr(&target, range);
                for arg in c.arguments.args.iter_mut() {
                    self.visit_expr(arg);
                }
                for k in c.arguments.keywords.iter_mut() {
                    self.visit_expr(&mut k.value);
                }
            }
            // `self.__class__.X` in a method of C is `C.X`.
            py::Expr::Attribute(outer)
                if let py::Expr::Attribute(inner) = &*outer.value
                    && inner.attr.as_str() == "__class__"
                    && let py::Expr::Name(this) = &*inner.value
                    && self
                        .method_of
                        .borrow()
                        .as_ref()
                        .is_some_and(|(_, me)| me == this.id.as_str()) =>
            {
                let class = self.method_of.borrow().as_ref().map(|(c, _)| c.clone());
                if let Some(class) = class {
                    *outer.value = name_expr(&class, inner.range());
                }
            }
            // `reduce(f, xs, start)` is a call of a fold function.
            py::Expr::Call(c)
                if c.arguments.keywords.is_empty()
                    && c.arguments.args.len() == 3
                    && !c
                        .arguments
                        .args
                        .iter()
                        .any(|a| matches!(a, py::Expr::Starred(_)))
                    && self.folds.is_reduce(&c.func, &self.origin.borrow()) =>
            {
                for arg in c.arguments.args.iter_mut() {
                    self.visit_expr(arg);
                }
                let origin = self.origin.borrow().clone();
                let op = self.folds.operator_of(&c.arguments.args[0], &origin);
                let name = match op {
                    Some(op) => format!("reduce${}", op_function_name(op)),
                    None => "reduce$call".to_string(),
                };
                let mut folders = self.folders.borrow_mut();
                if !folders.iter().any(|f| f.name == name && f.origin == origin) {
                    folders.push(Folder {
                        name: name.clone(),
                        origin,
                        op,
                        range: c.range(),
                    });
                }
                if op.is_some() {
                    let mut args = std::mem::take(&mut c.arguments.args).into_vec();
                    args.remove(0);
                    c.arguments.args = args.into_boxed_slice();
                }
                let range = c.func.range();
                *c.func = name_expr(&name, range);
            }
            // `map(f, xs)` is `[f(x) for x in xs]`: eager, as the
            // frontend's `map` is, with `f` named once per element.
            py::Expr::Call(c)
                if self.map_is_builtin
                    && is_name(&c.func, "map")
                    && c.arguments.keywords.is_empty()
                    && let [py::Expr::Name(_), xs] = &*c.arguments.args
                    && !matches!(xs, py::Expr::Starred(_)) =>
            {
                let range = c.range();
                let n = self.map_items.get();
                self.map_items.set(n + 1);
                let item = format!("map$item{n}");
                let mut args = std::mem::take(&mut c.arguments.args).into_vec();
                let mut iter = args.pop().expect("two arguments");
                let f = args.pop().expect("two arguments");
                self.visit_expr(&mut iter);
                let mut target = name_expr(&item, range);
                if let py::Expr::Name(t) = &mut target {
                    t.ctx = py::ExprContext::Store;
                }
                let elt = py::Expr::Call(py::ExprCall {
                    node_index: Default::default(),
                    range_start: range.start(),
                    func: Box::new(f),
                    arguments: py::Arguments {
                        node_index: Default::default(),
                        range,
                        args: vec![name_expr(&item, range)].into_boxed_slice(),
                        keywords: Default::default(),
                    },
                });
                *e = py::Expr::ListComp(py::ExprListComp {
                    node_index: Default::default(),
                    range,
                    elt: Box::new(elt),
                    generators: vec![py::Comprehension {
                        range,
                        node_index: Default::default(),
                        target,
                        iter,
                        ifs: Vec::new(),
                        is_async: false,
                    }],
                });
            }
            _ => walk_expr(self, e),
        }
    }
}

/// `if p is None: p = e` among the top-level statements of `f`, `p` a
/// parameter, becomes `p$set = e if p is None else p`, and every later
/// `p` in the body becomes `p$set`. Skipped where renaming could change
/// what a name means: `p` read by a nested scope or bound other than
/// by an expression target, or a body that reads its locals by name.
fn bind_none_defaults(f: &mut py::StmtFunctionDef) {
    let params: Vec<String> = f
        .parameters
        .iter_non_variadic_params()
        .map(|p| p.parameter.name.to_string())
        .collect();
    let mut i = 0;
    while i < f.body.len() {
        let found = match &f.body[i] {
            py::Stmt::If(s) if s.elif_else_clauses.is_empty() => match (&*s.test, &s.body[..]) {
                (py::Expr::Compare(c), [py::Stmt::Assign(a)])
                    if matches!(&c.ops[..], [py::CmpOp::Is])
                        && let (py::Expr::Name(n), [py::Expr::NoneLiteral(_)]) =
                            (&*c.left, &c.comparators[..])
                        && let [target] = &a.targets[..]
                        && is_name(target, n.id.as_str())
                        && params.iter().any(|p| p == n.id.as_str()) =>
                {
                    Some(n.id.to_string())
                }
                _ => None,
            },
            _ => None,
        };
        let Some(name) = found else {
            i += 1;
            continue;
        };
        let mut check = Renamable {
            name: &name,
            nested: 0,
            ok: true,
        };
        for stmt in &f.body {
            py::visitor::Visitor::visit_stmt(&mut check, stmt);
        }
        if !check.ok {
            i += 1;
            continue;
        }
        let set = format!("{name}$set");
        let placeholder = pass(f.range());
        let py::Stmt::If(s) = std::mem::replace(&mut f.body[i], placeholder) else {
            unreachable!("matched above");
        };
        let range = s.range;
        let mut body = s.body.into_iter();
        let Some(py::Stmt::Assign(a)) = body.next() else {
            unreachable!("matched above");
        };
        let mut target = name_expr(&set, a.targets[0].range());
        if let py::Expr::Name(t) = &mut target {
            t.ctx = py::ExprContext::Store;
        }
        let value = py::Expr::If(py::ExprIf {
            node_index: Default::default(),
            range,
            test: s.test,
            body: a.value,
            orelse: Box::new(name_expr(&name, range)),
        });
        f.body[i] = py::Stmt::Assign(py::StmtAssign {
            node_index: Default::default(),
            range,
            targets: vec![target],
            value: Box::new(value),
        });
        let renamer = Renamer {
            from: &name,
            to: &set,
        };
        for stmt in f.body[i + 1..].iter_mut() {
            renamer.visit_stmt(stmt);
        }
        i += 1;
    }
}

/// Whether every use of `name` in a function body is an expression
/// renaming can follow.
struct Renamable<'n> {
    name: &'n str,
    /// Depth of nested functions, lambdas and classes.
    nested: u32,
    ok: bool,
}

impl<'a> py::visitor::Visitor<'a> for Renamable<'_> {
    fn visit_stmt(&mut self, s: &'a py::Stmt) {
        let name = self.name;
        let binds = |id: &py::Identifier| id.as_str() == name;
        match s {
            py::Stmt::FunctionDef(_) | py::Stmt::ClassDef(_) => {
                self.nested += 1;
                py::visitor::walk_stmt(self, s);
                self.nested -= 1;
                if let py::Stmt::FunctionDef(f) = s
                    && binds(&f.name)
                {
                    self.ok = false;
                }
                if let py::Stmt::ClassDef(c) = s
                    && binds(&c.name)
                {
                    self.ok = false;
                }
                return;
            }
            py::Stmt::Global(g) if g.names.iter().any(binds) => self.ok = false,
            py::Stmt::Nonlocal(g) if g.names.iter().any(binds) => self.ok = false,
            py::Stmt::Import(im) if im.names.iter().any(|a| alias_binds(a, self.name)) => {
                self.ok = false
            }
            py::Stmt::ImportFrom(im) if im.names.iter().any(|a| alias_binds(a, self.name)) => {
                self.ok = false
            }
            py::Stmt::Match(_) => self.ok = false,
            _ => {}
        }
        py::visitor::walk_stmt(self, s);
    }

    fn visit_expr(&mut self, e: &'a py::Expr) {
        match e {
            py::Expr::Name(n) => {
                let id = n.id.as_str();
                if (id == self.name && self.nested > 0)
                    || matches!(id, "locals" | "vars" | "eval" | "exec")
                {
                    self.ok = false;
                }
            }
            py::Expr::Lambda(_) => {
                self.nested += 1;
                py::visitor::walk_expr(self, e);
                self.nested -= 1;
                return;
            }
            _ => {}
        }
        py::visitor::walk_expr(self, e);
    }

    fn visit_except_handler(&mut self, h: &'a py::ExceptHandler) {
        let py::ExceptHandler::ExceptHandler(h) = h;
        if h.name.as_ref().is_some_and(|n| n.as_str() == self.name) {
            self.ok = false;
        }
        for stmt in &h.body {
            self.visit_stmt(stmt);
        }
        if let Some(t) = &h.type_ {
            self.visit_expr(t);
        }
    }
}

fn alias_binds(a: &py::Alias, name: &str) -> bool {
    let bound = a.asname.as_ref().map_or(a.name.as_str(), |n| n.as_str());
    bound.split('.').next() == Some(name)
}

/// Every `from` in what it walks is `to`.
struct Renamer<'n> {
    from: &'n str,
    to: &'n str,
}

impl Transformer for Renamer<'_> {
    fn visit_expr(&self, e: &mut py::Expr) {
        if let py::Expr::Name(n) = e
            && n.id.as_str() == self.from
        {
            n.id = py::name::Name::new(self.to);
            return;
        }
        walk_expr(self, e);
    }
}

/// Whether a module-level statement binds `name`.
fn binds_module_name(body: &[py::Stmt], name: &str) -> bool {
    body.iter().any(|s| match s {
        py::Stmt::FunctionDef(f) => f.name.as_str() == name,
        py::Stmt::ClassDef(c) => c.name.as_str() == name,
        py::Stmt::Assign(a) => a.targets.iter().any(|t| is_name(t, name)),
        py::Stmt::AnnAssign(a) => is_name(&a.target, name),
        py::Stmt::Import(i) => i
            .names
            .iter()
            .any(|a| a.asname.as_ref().map_or(a.name.as_str(), |n| n.as_str()) == name),
        py::Stmt::ImportFrom(i) => i
            .names
            .iter()
            .any(|a| a.asname.as_ref().map_or(a.name.as_str(), |n| n.as_str()) == name),
        _ => false,
    })
}

fn is_name(e: &py::Expr, name: &str) -> bool {
    matches!(e, py::Expr::Name(n) if n.id.as_str() == name)
}

fn is_super(e: &py::Expr) -> bool {
    matches!(e, py::Expr::Call(c) if is_name(&c.func, "super"))
}
