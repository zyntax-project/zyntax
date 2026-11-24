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
//! - `list()` is `[]`, a list whose kind is what the program puts in.
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
    // Property getters and setters by name, and what each class says
    // about them: the classes it derives from, the properties it
    // defines, and the attributes its methods assign on `self`.
    let mut getters: HashSet<String> = HashSet::default();
    let mut setters: HashSet<String> = HashSet::default();
    let mut class_info: Vec<ClassInfo> = Vec::new();
    // The fields of the dataclasses so far, which one derived from them
    // starts with.
    let mut dataclass_fields: HashMap<String, Vec<crate::dataclass::Field>> = HashMap::default();
    for (i, s) in body.iter_mut().enumerate() {
        let py::Stmt::ClassDef(c) = s else {
            continue;
        };
        if let Some(fields) = crate::dataclass::rewrite_class(c, &dataclass_fields) {
            dataclass_fields.insert(c.name.to_string(), fields);
        }
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
        let mut info = ClassInfo {
            bases: bases
                .iter()
                .filter_map(|b| match b {
                    py::Expr::Name(n) => Some(n.id.to_string()),
                    _ => None,
                })
                .collect(),
            name: class.clone(),
            ..ClassInfo::default()
        };
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
                py::Stmt::FunctionDef(mut f) if property_accessor(&f).is_some() => {
                    let (name, is_getter) = property_accessor(&f).expect("picked");
                    f.decorator_list.clear();
                    let suffix = if is_getter { "get" } else { "set" };
                    f.name = py::Identifier::new(format!("{name}${suffix}"), f.name.range());
                    if is_getter {
                        getters.insert(name.clone());
                    } else {
                        setters.insert(name.clone());
                    }
                    info.props.insert(name);
                    kept.push(py::Stmt::FunctionDef(f));
                }
                other => kept.push(other),
            }
        }
        for m in &kept {
            if let py::Stmt::FunctionDef(f) = m
                && let Some(this) = f.parameters.args.first().map(|p| p.parameter.name.as_str())
            {
                assigned_on(&f.body, this, &mut info.assigns);
            }
        }
        class_info.push(info);
        // A class that defines `__iter__` or `__next__` gets the method
        // that lists what it iterates, which the library calls to
        // iterate an instance it holds boxed.
        let defines = |name: &str| {
            kept.iter()
                .any(|m| matches!(m, py::Stmt::FunctionDef(f) if f.name.as_str() == name))
        };
        if (defines("__iter__") || defines("__next__")) && !defines("__iter_all__") {
            // An iterator class is read through its `__next__` until
            // StopIteration; any other through the iterator `__iter__`
            // returns.
            // The iterator class also has the generator that pulls from
            // it, which a `for` loop advances item by item.
            let source = if defines("__next__") {
                "def __iter_all__(self):\n    out = []\n    while True:\n        try:\n            out.append(self.__next__())\n        except StopIteration:\n            break\n    return out\n\ndef __iter_gen__(self):\n    while True:\n        try:\n            item = self.__next__()\n        except StopIteration:\n            return\n        yield item\n"
            } else {
                "def __iter_all__(self):\n    return list(self.__iter__())\n"
            };
            if let Ok(parsed) = ruff_python_parser::parse_module(source) {
                kept.extend(parsed.into_syntax().body);
            }
        }
        c.body = kept.into_iter().collect();
    }
    // The decorator and `field` were consumed: the import of them is
    // no module the program needs.
    if !dataclass_fields.is_empty() {
        // The import of a module the program lacks is the binding
        // `name = __import__("dataclasses")`, or one of its members.
        fn names_dataclasses(e: &py::Expr) -> bool {
            match e {
                py::Expr::Attribute(a) => names_dataclasses(&a.value),
                py::Expr::Call(c) => {
                    is_name(&c.func, "__import__")
                        && matches!(c.arguments.args.first(),
                            Some(py::Expr::StringLiteral(l)) if l.value.to_str() == "dataclasses")
                }
                _ => false,
            }
        }
        let is_import = |s: &py::Stmt| match s {
            py::Stmt::Assign(a) => names_dataclasses(&a.value),
            py::Stmt::ImportFrom(f) => {
                f.level == 0
                    && f.module
                        .as_ref()
                        .is_some_and(|m| m.as_str() == "dataclasses")
            }
            py::Stmt::Import(i) => i.names.iter().all(|a| a.name.as_str() == "dataclasses"),
            _ => false,
        };
        let mut kept = Vec::with_capacity(body.len());
        let mut kept_origins = Vec::with_capacity(body.len());
        for (s, origin) in std::mem::take(body).into_iter().zip(origins.iter()) {
            if !is_import(&s) {
                kept.push(s);
                kept_origins.push(origin.clone());
            }
        }
        *body = kept;
        *origins = kept_origins;
    }
    // A property is not rewritten where a class that does not own it
    // keeps a plain attribute of the name.
    let owns = |class: &str, prop: &str| {
        let mut seen: Vec<&str> = vec![class];
        let mut at = 0;
        while at < seen.len() {
            let Some(info) = class_info.iter().find(|c| c.name == seen[at]) else {
                at += 1;
                continue;
            };
            if info.props.contains(prop) {
                return true;
            }
            for b in &info.bases {
                if !seen.contains(&b.as_str()) {
                    seen.push(b.as_str());
                }
            }
            at += 1;
        }
        false
    };
    let conflicts: HashSet<String> = getters
        .iter()
        .chain(&setters)
        .filter(|p| {
            class_info
                .iter()
                .any(|c| c.assigns.contains(*p) && !owns(&c.name, p))
        })
        .cloned()
        .collect();
    getters.retain(|p| !conflicts.contains(p));
    setters.retain(|p| !conflicts.contains(p));
    let classes: HashSet<String> = body
        .iter()
        .filter_map(|s| match s {
            py::Stmt::ClassDef(c) => Some(c.name.to_string()),
            _ => None,
        })
        .collect();
    let rewriter = Rewriter {
        getters,
        setters,
        classes,
        rootless,
        aliases,
        statics,
        method_of: RefCell::new(None),
        map_is_builtin: !binds_module_name(body, "map"),
        list_is_builtin: !binds_module_name(body, "list"),
        iter_is_builtin: !binds_module_name(body, "iter"),
        imported_modules: body
            .iter()
            .filter_map(|s| match s {
                py::Stmt::Import(i) => Some(i),
                _ => None,
            })
            .flat_map(|i| i.names.iter())
            .map(|a| {
                a.asname
                    .as_ref()
                    .map_or_else(
                        || a.name.split('.').next().unwrap_or_default(),
                        |n| n.as_str(),
                    )
                    .to_string()
            })
            .collect(),
        with_items: Cell::new(0),
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

/// A property accessor: `@property` is the getter, `@name.setter` on
/// `def name` the setter. Returns the property's name and which.
fn property_accessor(f: &py::StmtFunctionDef) -> Option<(String, bool)> {
    let [d] = &*f.decorator_list else {
        return None;
    };
    match &d.expression {
        py::Expr::Name(n) if n.id.as_str() == "property" => Some((f.name.to_string(), true)),
        py::Expr::Attribute(a)
            if a.attr.as_str() == "setter"
                && matches!(&*a.value, py::Expr::Name(n) if n.id.as_str() == f.name.as_str()) =>
        {
            Some((f.name.to_string(), false))
        }
        _ => None,
    }
}

fn is_static_or_class_method(f: &py::StmtFunctionDef) -> bool {
    f.decorator_list.len() == 1
        && (decorated_with(f, "classmethod") || decorated_with(f, "staticmethod"))
}

/// `receiver.method(args)`.
fn method_call(
    receiver: py::Expr,
    method: &str,
    args: Vec<py::Expr>,
    range: ruff_text_size::TextRange,
) -> py::Expr {
    py::Expr::Call(py::ExprCall {
        node_index: Default::default(),
        range_start: range.start(),
        func: Box::new(py::Expr::Attribute(py::ExprAttribute {
            node_index: Default::default(),
            range,
            value: Box::new(receiver),
            attr: py::Identifier::new(method, range),
            ctx: py::ExprContext::Load,
        })),
        arguments: py::Arguments {
            node_index: Default::default(),
            range,
            args: args.into_boxed_slice(),
            keywords: Default::default(),
        },
    })
}

/// Whether `e` reads the same twice: names and attributes of names.
pub(crate) fn is_pure_path(e: &py::Expr) -> bool {
    match e {
        py::Expr::Name(_) => true,
        py::Expr::Attribute(a) => is_pure_path(&a.value),
        _ => false,
    }
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

/// What `rewrite` learns of one class for its properties.
#[derive(Default)]
struct ClassInfo {
    name: String,
    bases: Vec<String>,
    props: HashSet<String>,
    assigns: HashSet<String>,
}

/// The attributes `body` assigns on the name `this`.
fn assigned_on(body: &[py::Stmt], this: &str, out: &mut HashSet<String>) {
    use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
    struct Assigns<'a>(&'a str, &'a mut HashSet<String>);
    impl<'a> Visitor<'a> for Assigns<'_> {
        fn visit_stmt(&mut self, s: &'a py::Stmt) {
            walk_stmt(self, s);
        }
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Attribute(a) = e
                && matches!(a.ctx, py::ExprContext::Store)
                && is_name(&a.value, self.0)
            {
                self.1.insert(a.attr.to_string());
            }
            walk_expr(self, e);
        }
    }
    let mut v = Assigns(this, out);
    for s in body {
        v.visit_stmt(s);
    }
}

struct Rewriter {
    /// The names of properties with a getter, and with a setter.
    getters: HashSet<String>,
    setters: HashSet<String>,
    /// The classes the module defines.
    classes: HashSet<String>,
    rootless: HashSet<String>,
    aliases: HashMap<String, String>,
    statics: HashMap<String, HashSet<String>>,
    /// The class and `self` name of the method being walked.
    method_of: RefCell<Option<(String, String)>>,
    /// No module-level binding shadows the builtin `map`.
    map_is_builtin: bool,
    /// No module-level binding shadows the builtin `list`.
    list_is_builtin: bool,
    /// No module-level binding shadows the builtin `iter`.
    iter_is_builtin: bool,
    /// The names `import` binds to modules at the top of the program.
    imported_modules: HashSet<String>,
    /// `with` statements made into try statements so far.
    with_items: Cell<u32>,
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
    /// Whether `e` names a module or a class of the program, whose
    /// attributes are not an instance's.
    fn names_module_or_class(&self, e: &py::Expr) -> bool {
        matches!(e, py::Expr::Name(n)
            if self.imported_modules.contains(n.id.as_str()) || self.classes.contains(n.id.as_str()))
    }

    /// Whether `with` on `e` may be a context manager the program
    /// defines: not a file opened in place, nor a call of a module's.
    fn is_program_context(&self, e: &py::Expr) -> bool {
        match e {
            py::Expr::Call(c) => match &*c.func {
                py::Expr::Name(f) => f.id.as_str() != "open",
                py::Expr::Attribute(a) => !matches!(
                    &*a.value,
                    py::Expr::Name(m) if self.imported_modules.contains(m.id.as_str())
                ),
                _ => true,
            },
            _ => true,
        }
    }

    /// `with a as x, b as y: body` as
    /// `m = a; x = m.__enter__(); try: ... except BaseException as e:
    /// seen = True; if not m.__exit__(type(e), e, None): raise;
    /// finally: if not seen: m.__exit__(None, None, None)`, nested.
    fn desugar_with(&self, w: &mut py::StmtWith) -> py::Stmt {
        let range = w.range();
        let mut body = std::mem::take(&mut w.body);
        let items = std::mem::take(&mut w.items);
        for item in items.into_iter().rev() {
            let n = self.with_items.get();
            self.with_items.set(n + 1);
            let source = format!(
                "__with_mgr_{n} = None\n\
                 __with_target_{n} = __with_mgr_{n}.__enter__()\n\
                 __with_seen_{n} = False\n\
                 try:\n    pass\n\
                 except BaseException as __with_exc_{n}:\n\
                 \x20   __with_seen_{n} = True\n\
                 \x20   if not __with_mgr_{n}.__exit__(type(__with_exc_{n}), __with_exc_{n}, None):\n\
                 \x20       raise\n\
                 finally:\n\
                 \x20   if not __with_seen_{n}:\n\
                 \x20       __with_mgr_{n}.__exit__(None, None, None)\n"
            );
            let mut parsed = ruff_python_parser::parse_module(&source)
                .expect("the with template parses")
                .into_syntax()
                .body;
            if let py::Stmt::Assign(a) = &mut parsed[0] {
                *a.value = item.context_expr;
            }
            match item.optional_vars {
                Some(target) => {
                    if let py::Stmt::Assign(a) = &mut parsed[1] {
                        a.targets = vec![*target];
                    }
                }
                None => {
                    // The entered value is dropped.
                    if let py::Stmt::Assign(a) = &parsed[1] {
                        let call = (*a.value).clone();
                        parsed[1] = py::Stmt::Expr(py::StmtExpr {
                            node_index: Default::default(),
                            range: a.range,
                            value: Box::new(call),
                        });
                    }
                }
            }
            if let py::Stmt::Try(t) = &mut parsed[3] {
                t.body = body;
            }
            body = parsed;
        }
        py::Stmt::If(py::StmtIf {
            node_index: Default::default(),
            range,
            test: Box::new(py::Expr::BooleanLiteral(py::ExprBooleanLiteral {
                node_index: Default::default(),
                range,
                value: true,
            })),
            body,
            elif_else_clauses: Vec::new(),
        })
    }

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
                narrow_isinstance_branches(f, &self.classes);
                walk_stmt(self, s);
            }
            py::Stmt::ClassDef(c) => {
                let class = c.name.to_string();
                for m in c.body.iter_mut() {
                    if let py::Stmt::FunctionDef(f) = m {
                        bind_none_defaults(f);
                        narrow_isinstance_branches(f, &self.classes);
                        if f.name.as_str() == "__iter__" && self.iter_is_builtin {
                            unwrap_iter_returns(&mut f.body);
                        }
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
            // `obj.p = v` for a property with a setter is `obj.p$set(v)`.
            py::Stmt::Assign(a)
                if a.targets.len() == 1
                    && let py::Expr::Attribute(t) = &a.targets[0]
                    && self.setters.contains(t.attr.as_str())
                    && !is_super(&t.value)
                    && !self.names_module_or_class(&t.value) =>
            {
                let range = a.range;
                let placeholder = name_expr("_", range);
                let py::Expr::Attribute(target) =
                    std::mem::replace(&mut a.targets[0], placeholder.clone())
                else {
                    unreachable!("matched an attribute")
                };
                let value = std::mem::replace(&mut *a.value, placeholder);
                *s = py::Stmt::Expr(py::StmtExpr {
                    node_index: Default::default(),
                    range,
                    value: Box::new(method_call(
                        *target.value,
                        &format!("{}$set", target.attr),
                        vec![value],
                        range,
                    )),
                });
                walk_stmt(self, s);
            }
            // `obj.p += v` is `obj.p$set(obj.p + v)`.
            py::Stmt::AugAssign(a)
                if let py::Expr::Attribute(t) = &*a.target
                    && self.setters.contains(t.attr.as_str())
                    && self.getters.contains(t.attr.as_str())
                    && is_pure_path(&t.value)
                    && !is_super(&t.value)
                    && !self.names_module_or_class(&t.value) =>
            {
                let range = a.range;
                let placeholder = name_expr("_", range);
                let py::Expr::Attribute(target) =
                    std::mem::replace(&mut *a.target, placeholder.clone())
                else {
                    unreachable!("matched an attribute")
                };
                let value = std::mem::replace(&mut *a.value, placeholder);
                let read = py::Expr::Attribute(py::ExprAttribute {
                    node_index: Default::default(),
                    range,
                    value: target.value.clone(),
                    attr: target.attr.clone(),
                    ctx: py::ExprContext::Load,
                });
                let sum = py::Expr::BinOp(py::ExprBinOp {
                    node_index: Default::default(),
                    range,
                    left: Box::new(read),
                    op: a.op,
                    right: Box::new(value),
                });
                *s = py::Stmt::Expr(py::StmtExpr {
                    node_index: Default::default(),
                    range,
                    value: Box::new(method_call(
                        *target.value,
                        &format!("{}$set", target.attr),
                        vec![sum],
                        range,
                    )),
                });
                walk_stmt(self, s);
            }
            // `with` on a context manager of the program's own is a try
            // statement that calls its `__enter__` and `__exit__`.
            py::Stmt::With(w)
                if !w.is_async
                    && w.items
                        .iter()
                        .all(|i| self.is_program_context(&i.context_expr)) =>
            {
                *s = self.desugar_with(w);
                walk_stmt(self, s);
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
            // `list()` is `[]`.
            py::Expr::Call(c)
                if self.list_is_builtin
                    && let py::Expr::Name(f) = &*c.func
                    && (f.id.as_str() == "list"
                        || self.aliases.get(f.id.as_str()).is_some_and(|b| b == "list"))
                    && c.arguments.args.is_empty()
                    && c.arguments.keywords.is_empty() =>
            {
                *e = py::Expr::List(py::ExprList {
                    node_index: Default::default(),
                    range: c.range(),
                    elts: Vec::new(),
                    ctx: py::ExprContext::Load,
                });
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
            // A call of an attribute is a method call: the receiver and
            // the arguments are rewritten, not the callee.
            py::Expr::Call(c)
                if !self.getters.is_empty() && matches!(&*c.func, py::Expr::Attribute(_)) =>
            {
                if let py::Expr::Attribute(a) = &mut *c.func {
                    self.visit_expr(&mut a.value);
                }
                for arg in c.arguments.args.iter_mut() {
                    self.visit_expr(arg);
                }
                for k in c.arguments.keywords.iter_mut() {
                    self.visit_expr(&mut k.value);
                }
            }
            // `obj.p` of a property is `obj.p$get()`.
            py::Expr::Attribute(a)
                if self.getters.contains(a.attr.as_str())
                    && a.ctx.is_load()
                    && !is_super(&a.value)
                    && !self.names_module_or_class(&a.value) =>
            {
                self.visit_expr(&mut a.value);
                let range = a.range;
                let receiver = std::mem::replace(&mut *a.value, name_expr("_", range));
                let method = format!("{}$get", a.attr);
                *e = method_call(receiver, &method, Vec::new(), range);
            }
            // `iter(xs)` is the generator `(x for x in xs)`: an iterator
            // that `next` and a `for` loop advance.
            py::Expr::Call(c)
                if self.iter_is_builtin
                    && is_name(&c.func, "iter")
                    && c.arguments.keywords.is_empty()
                    && let [only] = &*c.arguments.args
                    && !matches!(only, py::Expr::Starred(_)) =>
            {
                let range = c.range();
                let n = self.map_items.get();
                self.map_items.set(n + 1);
                let item = format!("iter$item{n}");
                let mut iter = std::mem::take(&mut c.arguments.args)
                    .into_vec()
                    .pop()
                    .expect("one argument");
                self.visit_expr(&mut iter);
                let mut target = name_expr(&item, range);
                if let py::Expr::Name(t) = &mut target {
                    t.ctx = py::ExprContext::Store;
                }
                *e = py::Expr::Generator(py::ExprGenerator {
                    node_index: Default::default(),
                    range,
                    elt: Box::new(name_expr(&item, range)),
                    generators: vec![py::Comprehension {
                        range,
                        node_index: Default::default(),
                        target,
                        iter,
                        ifs: Vec::new(),
                        is_async: false,
                    }],
                    parenthesized: true,
                });
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

/// `return iter(xs)` in an `__iter__` is `return xs`: the sequence is
/// iterated once by whoever called, as the iterator would be.
fn unwrap_iter_returns(stmts: &mut [py::Stmt]) {
    for s in stmts {
        match s {
            py::Stmt::Return(r) => {
                if let Some(value) = &mut r.value
                    && let py::Expr::Call(c) = &mut **value
                    && is_name(&c.func, "iter")
                    && c.arguments.keywords.is_empty()
                    && c.arguments.args.len() == 1
                    && !matches!(c.arguments.args[0], py::Expr::Starred(_))
                {
                    let range = c.range();
                    let inner = std::mem::replace(
                        &mut c.arguments.args[0],
                        py::Expr::NoneLiteral(py::ExprNoneLiteral {
                            node_index: Default::default(),
                            range,
                        }),
                    );
                    **value = inner;
                }
            }
            py::Stmt::If(i) => {
                unwrap_iter_returns(&mut i.body);
                for clause in &mut i.elif_else_clauses {
                    unwrap_iter_returns(&mut clause.body);
                }
            }
            py::Stmt::For(f) => unwrap_iter_returns(&mut f.body),
            py::Stmt::While(w) => unwrap_iter_returns(&mut w.body),
            py::Stmt::With(w) => unwrap_iter_returns(&mut w.body),
            py::Stmt::Try(t) => {
                unwrap_iter_returns(&mut t.body);
                for h in &mut t.handlers {
                    let py::ExceptHandler::ExceptHandler(h) = h;
                    unwrap_iter_returns(&mut h.body);
                }
                unwrap_iter_returns(&mut t.orelse);
                unwrap_iter_returns(&mut t.finalbody);
            }
            _ => {}
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

/// `if isinstance(x, C):` and `elif isinstance(x, C):` in `f`, with `x` a
/// name the branch does not bind and `C` a class of the module: the branch
/// starts with `x$isN = x` and `assert isinstance(x$isN, C)`, the pair the
/// typer reads as an instance of `C`, and reads `x$isN` for `x`. Skipped
/// where renaming could change what a name means (see [`Renamable`]).
fn narrow_isinstance_branches(f: &mut py::StmtFunctionDef, classes: &HashSet<String>) {
    let mut renamable: HashMap<String, bool> = HashMap::default();
    let mut tested = Vec::new();
    isinstance_tests(&f.body, classes, &mut tested);
    for name in tested {
        if renamable.contains_key(&name) {
            continue;
        }
        let mut check = Renamable {
            name: &name,
            nested: 0,
            ok: true,
        };
        for stmt in &f.body {
            py::visitor::Visitor::visit_stmt(&mut check, stmt);
        }
        let ok = check.ok;
        renamable.insert(name, ok);
    }
    let mut count = 0;
    narrow_in(&mut f.body, classes, &renamable, &mut count);
}

/// The narrowed name of an `isinstance(x, C)` test, `C` a class of the
/// module.
fn isinstance_test<'e>(test: &'e py::Expr, classes: &HashSet<String>) -> Option<&'e str> {
    let py::Expr::Call(c) = test else {
        return None;
    };
    if !is_name(&c.func, "isinstance") || !c.arguments.keywords.is_empty() {
        return None;
    }
    match &c.arguments.args[..] {
        [py::Expr::Name(x), py::Expr::Name(class)] if classes.contains(class.id.as_str()) => {
            Some(x.id.as_str())
        }
        _ => None,
    }
}

/// The names `isinstance` tests narrow in `stmts`, outside nested scopes.
fn isinstance_tests(stmts: &[py::Stmt], classes: &HashSet<String>, out: &mut Vec<String>) {
    for s in stmts {
        match s {
            py::Stmt::If(i) => {
                out.extend(isinstance_test(&i.test, classes).map(str::to_string));
                isinstance_tests(&i.body, classes, out);
                for c in &i.elif_else_clauses {
                    if let Some(t) = &c.test {
                        out.extend(isinstance_test(t, classes).map(str::to_string));
                    }
                    isinstance_tests(&c.body, classes, out);
                }
            }
            py::Stmt::While(w) => {
                isinstance_tests(&w.body, classes, out);
                isinstance_tests(&w.orelse, classes, out);
            }
            py::Stmt::For(f) => {
                isinstance_tests(&f.body, classes, out);
                isinstance_tests(&f.orelse, classes, out);
            }
            py::Stmt::With(w) => isinstance_tests(&w.body, classes, out),
            py::Stmt::Try(t) => {
                isinstance_tests(&t.body, classes, out);
                for h in &t.handlers {
                    let py::ExceptHandler::ExceptHandler(h) = h;
                    isinstance_tests(&h.body, classes, out);
                }
                isinstance_tests(&t.orelse, classes, out);
                isinstance_tests(&t.finalbody, classes, out);
            }
            _ => {}
        }
    }
}

fn narrow_in(
    stmts: &mut [py::Stmt],
    classes: &HashSet<String>,
    renamable: &HashMap<String, bool>,
    count: &mut u32,
) {
    for s in stmts.iter_mut() {
        match s {
            py::Stmt::If(i) => {
                narrow_branch(&i.test, &mut i.body, classes, renamable, count);
                narrow_in(&mut i.body, classes, renamable, count);
                for c in i.elif_else_clauses.iter_mut() {
                    if let Some(t) = &c.test {
                        narrow_branch(t, &mut c.body, classes, renamable, count);
                    }
                    narrow_in(&mut c.body, classes, renamable, count);
                }
            }
            py::Stmt::While(w) => {
                narrow_in(&mut w.body, classes, renamable, count);
                narrow_in(&mut w.orelse, classes, renamable, count);
            }
            py::Stmt::For(f) => {
                narrow_in(&mut f.body, classes, renamable, count);
                narrow_in(&mut f.orelse, classes, renamable, count);
            }
            py::Stmt::With(w) => narrow_in(&mut w.body, classes, renamable, count),
            py::Stmt::Try(t) => {
                narrow_in(&mut t.body, classes, renamable, count);
                for h in t.handlers.iter_mut() {
                    let py::ExceptHandler::ExceptHandler(h) = h;
                    narrow_in(&mut h.body, classes, renamable, count);
                }
                narrow_in(&mut t.orelse, classes, renamable, count);
                narrow_in(&mut t.finalbody, classes, renamable, count);
            }
            _ => {}
        }
    }
}

fn narrow_branch<B>(
    test: &py::Expr,
    body: &mut B,
    classes: &HashSet<String>,
    renamable: &HashMap<String, bool>,
    count: &mut u32,
) where
    B: Default
        + FromIterator<py::Stmt>
        + IntoIterator<Item = py::Stmt>
        + std::ops::DerefMut<Target = [py::Stmt]>,
{
    let Some(x) = isinstance_test(test, classes) else {
        return;
    };
    if !renamable.get(x).copied().unwrap_or(false) || binds_name(body, x) {
        return;
    }
    let to = format!("{x}$is{count}");
    *count += 1;
    let renamer = Renamer { from: x, to: &to };
    for stmt in body.iter_mut() {
        renamer.visit_stmt(stmt);
    }
    let range = test.range();
    let mut target = name_expr(&to, range);
    if let py::Expr::Name(t) = &mut target {
        t.ctx = py::ExprContext::Store;
    }
    let mut asserted = test.clone();
    renamer.visit_expr(&mut asserted);
    let rest = std::mem::take(body);
    *body = [
        py::Stmt::Assign(py::StmtAssign {
            node_index: Default::default(),
            range,
            targets: vec![target],
            value: Box::new(name_expr(x, range)),
        }),
        py::Stmt::Assert(py::StmtAssert {
            node_index: Default::default(),
            range,
            test: Box::new(asserted),
            msg: None,
        }),
    ]
    .into_iter()
    .chain(rest)
    .collect();
}

/// Whether `stmts` bind `name` anywhere, nested blocks included.
fn binds_name(stmts: &[py::Stmt], name: &str) -> bool {
    struct Binds<'n> {
        name: &'n str,
        found: bool,
    }
    impl<'a> py::visitor::Visitor<'a> for Binds<'_> {
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Name(n) = e
                && n.id.as_str() == self.name
                && !n.ctx.is_load()
            {
                self.found = true;
            }
            py::visitor::walk_expr(self, e);
        }
    }
    let mut b = Binds { name, found: false };
    for s in stmts {
        py::visitor::Visitor::visit_stmt(&mut b, s);
    }
    b.found
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
