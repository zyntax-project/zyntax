//! `*args` as one tuple parameter. A function that takes `*rest` gets a
//! positional parameter `rest` after its others, and each call the
//! frontend can resolve to it packs the surplus arguments into a tuple
//! for that parameter. The program is rewritten to say so before
//! inference, so the rest of the frontend sees ordinary parameters.
//!
//! A call is resolved by name: a function, a class (its `__init__`),
//! `Class.method`, `super().method`, or a method name every definition
//! of which agrees. A function that takes `*args` and is used as a value
//! is refused, because a call through the value cannot be packed.

use crate::dataclass::Fill;
use crate::{Error, Result};
use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use ruff_python_ast::visitor::{Visitor, walk_expr as read_expr, walk_stmt as read_stmt};
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cell::RefCell;

/// Method names of the builtin types: a call to one of these cannot be
/// told from a call to a method of the same name that takes `*args`.
const BUILTIN_METHODS: &[&str] = &[
    "add",
    "append",
    "clear",
    "close",
    "copy",
    "count",
    "difference",
    "discard",
    "endswith",
    "extend",
    "find",
    "flush",
    "format",
    "get",
    "index",
    "insert",
    "intersection",
    "items",
    "join",
    "keys",
    "lower",
    "pop",
    "read",
    "readline",
    "readlines",
    "remove",
    "replace",
    "reverse",
    "setdefault",
    "sort",
    "split",
    "startswith",
    "strip",
    "union",
    "update",
    "upper",
    "values",
    "write",
];

/// A function that takes `*rest`.
#[derive(Clone, PartialEq, Debug)]
struct Variadic {
    /// The positional parameters before `*rest`.
    positional: usize,
    /// How many of them a call through an instance, or through the
    /// class, does not give: the receiver.
    via_instance: usize,
    via_class: usize,
    /// The name of the parameter the surplus goes to.
    name: String,
}

#[derive(Default)]
struct ClassEntry {
    bases: Vec<String>,
    /// Each method of the class body, with its `*args` if it has one.
    methods: HashMap<String, Option<Variadic>>,
}

/// What the program defines, by name.
#[derive(Default)]
struct Tables {
    functions: HashMap<String, Vec<Option<Variadic>>>,
    classes: HashMap<String, ClassEntry>,
    methods: HashMap<String, Vec<Option<Variadic>>>,
}

impl Tables {
    /// The method `name` of `class` or of the first base that has one.
    fn lookup(&self, class: &str, name: &str, depth: usize) -> Option<Option<Variadic>> {
        let entry = self.classes.get(class)?;
        if let Some(m) = entry.methods.get(name) {
            return Some(m.clone());
        }
        if depth > 16 {
            return None;
        }
        entry
            .bases
            .iter()
            .find_map(|b| self.lookup(b, name, depth + 1))
    }
}

/// The `*args` of `f`, when it has them.
fn variadic(f: &py::StmtFunctionDef, method: bool) -> Result<Option<Variadic>> {
    let Some(rest) = &f.parameters.vararg else {
        return Ok(None);
    };
    let (mut via_instance, mut via_class) = (usize::from(method), 0);
    for d in &f.decorator_list {
        match &d.expression {
            py::Expr::Name(n) if method && n.id.as_str() == "staticmethod" => via_instance = 0,
            py::Expr::Name(n) if method && n.id.as_str() == "classmethod" => via_class = 1,
            other => {
                return Err(Error::unsupported(
                    "a decorated function that takes *args",
                    other,
                ));
            }
        }
    }
    Ok(Some(Variadic {
        positional: f.parameters.posonlyargs.len() + f.parameters.args.len(),
        via_instance,
        via_class,
        name: rest.name.to_string(),
    }))
}

/// Rewrite every function of `body` that takes `*args`, and the calls
/// to it.
pub(crate) fn rewrite(body: &mut [py::Stmt]) -> Result<()> {
    let mut collect = Collect {
        tables: Tables::default(),
        error: None,
    };
    for s in body.iter() {
        collect.visit_stmt(s);
    }
    if let Some(e) = collect.error {
        return Err(e);
    }
    let tables = collect.tables;
    let any = |defs: &HashMap<String, Vec<Option<Variadic>>>| {
        defs.iter()
            .filter(|(_, d)| d.iter().any(Option::is_some))
            .map(|(n, _)| n.clone())
            .collect::<HashSet<String>>()
    };
    let functions = any(&tables.functions);
    let methods = any(&tables.methods);
    if functions.is_empty() && methods.is_empty() {
        return Ok(());
    }
    let mut values = Values {
        functions: &functions,
        methods: &methods,
        error: None,
    };
    for s in body.iter() {
        values.visit_stmt(s);
    }
    if let Some(e) = values.error {
        return Err(e);
    }
    let rewriter = Rewriter {
        tables: &tables,
        classes: RefCell::new(Vec::new()),
        error: RefCell::new(None),
    };
    for s in body.iter_mut() {
        rewriter.visit_stmt(s);
    }
    match rewriter.error.into_inner() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Finds the functions, classes and methods the program defines.
struct Collect {
    tables: Tables,
    error: Option<Error>,
}

impl Collect {
    fn note(&mut self, f: &py::StmtFunctionDef, method: bool) -> Option<Variadic> {
        match variadic(f, method) {
            Ok(v) => v,
            Err(e) => {
                self.error.get_or_insert(e);
                None
            }
        }
    }
}

impl<'a> Visitor<'a> for Collect {
    fn visit_stmt(&mut self, s: &'a py::Stmt) {
        match s {
            py::Stmt::FunctionDef(f) => {
                let v = self.note(f, false);
                self.tables
                    .functions
                    .entry(f.name.to_string())
                    .or_default()
                    .push(v);
                read_stmt(self, s);
            }
            py::Stmt::ClassDef(c) => {
                let mut entry = ClassEntry::default();
                if let Some(args) = &c.arguments {
                    for b in &args.args {
                        if let py::Expr::Name(n) = b {
                            entry.bases.push(n.id.to_string());
                        }
                    }
                }
                for m in &c.body {
                    match m {
                        py::Stmt::FunctionDef(f) => {
                            let v = self.note(f, true);
                            entry.methods.insert(f.name.to_string(), v.clone());
                            self.tables
                                .methods
                                .entry(f.name.to_string())
                                .or_default()
                                .push(v);
                            for inner in &f.body {
                                self.visit_stmt(inner);
                            }
                        }
                        other => self.visit_stmt(other),
                    }
                }
                self.tables.classes.insert(c.name.to_string(), entry);
            }
            _ => read_stmt(self, s),
        }
    }
}

/// Refuses a function that takes `*args` named anywhere but as the
/// function of a call.
struct Values<'t> {
    functions: &'t HashSet<String>,
    methods: &'t HashSet<String>,
    error: Option<Error>,
}

impl<'a> Visitor<'a> for Values<'_> {
    fn visit_expr(&mut self, e: &'a py::Expr) {
        match e {
            py::Expr::Call(c) => {
                match &*c.func {
                    py::Expr::Name(_) => {}
                    py::Expr::Attribute(a) => self.visit_expr(&a.value),
                    other => self.visit_expr(other),
                }
                for a in &c.arguments.args {
                    self.visit_expr(a);
                }
                for k in &c.arguments.keywords {
                    self.visit_expr(&k.value);
                }
            }
            py::Expr::Name(n)
                if n.ctx == py::ExprContext::Load && self.functions.contains(n.id.as_str()) =>
            {
                self.error.get_or_insert(Error::unsupported(
                    "a function that takes *args used as a value",
                    e,
                ));
            }
            py::Expr::Attribute(a)
                if a.ctx == py::ExprContext::Load && self.methods.contains(a.attr.as_str()) =>
            {
                self.error.get_or_insert(Error::unsupported(
                    "a method that takes *args used as a value",
                    e,
                ));
            }
            _ => read_expr(self, e),
        }
    }
}

/// Replaces each `*rest` parameter and packs the arguments of the calls
/// to its function.
struct Rewriter<'t> {
    tables: &'t Tables,
    /// The classes the walk is inside, innermost last.
    classes: RefCell<Vec<String>>,
    error: RefCell<Option<Error>>,
}

impl Rewriter<'_> {
    fn fail(&self, what: impl Into<String>, at: &impl Ranged) {
        self.error
            .borrow_mut()
            .get_or_insert_with(|| Error::unsupported(what.into(), at));
    }

    /// The `*args` all of `defs` share, when any has one.
    fn agree(&self, defs: &[Option<Variadic>], what: &str, at: &impl Ranged) -> Option<Variadic> {
        let first = defs.iter().flatten().next()?;
        if defs.iter().all(|d| d.as_ref() == Some(first)) {
            return Some(first.clone());
        }
        self.fail(
            format!("`{what}` defined with *args in one place and differently in another"),
            at,
        );
        None
    }

    /// What a call to `func` takes `*args` of, and how many positional
    /// arguments the call gives before them.
    fn resolve(&self, func: &py::Expr, at: &impl Ranged) -> Option<(Variadic, usize)> {
        let tables = self.tables;
        match func {
            py::Expr::Name(n) => {
                let id = n.id.as_str();
                if let Some(defs) = tables.functions.get(id) {
                    let v = self.agree(defs, id, at)?;
                    let fixed = v.positional;
                    return Some((v, fixed));
                }
                let v = tables.lookup(id, "__init__", 0)??;
                let fixed = v.positional - v.via_instance;
                Some((v, fixed))
            }
            py::Expr::Attribute(a) => {
                let attr = a.attr.as_str();
                match &*a.value {
                    py::Expr::Name(c) if tables.classes.contains_key(c.id.as_str()) => {
                        let v = tables.lookup(c.id.as_str(), attr, 0)??;
                        let fixed = v.positional - v.via_class;
                        Some((v, fixed))
                    }
                    py::Expr::Call(s) if is_super(s) => {
                        let class = self.classes.borrow().last().cloned()?;
                        let v = tables
                            .classes
                            .get(&class)?
                            .bases
                            .iter()
                            .find_map(|b| tables.lookup(b, attr, 0))??;
                        let fixed = v.positional - v.via_instance;
                        Some((v, fixed))
                    }
                    _ => {
                        let v = self.agree(tables.methods.get(attr)?, attr, at)?;
                        if BUILTIN_METHODS.contains(&attr) {
                            self.fail(
                                format!(
                                    "a method named `{attr}` that takes *args: builtin types have one of that name"
                                ),
                                at,
                            );
                            return None;
                        }
                        let fixed = v.positional - v.via_instance;
                        Some((v, fixed))
                    }
                }
            }
            _ => None,
        }
    }

    /// The call `f(a, *xs)` with `xs` a name or attribute path and `fixed`
    /// parameters before the tuple: `args` becomes `a, xs[0], ...` and
    /// the tuple is what follows.
    fn spread(&self, args: &mut Vec<py::Expr>, at: usize, fixed: usize) -> Option<py::Expr> {
        if at + 1 != args.len() {
            return None;
        }
        let Some(py::Expr::Starred(star)) = args.pop() else {
            return None;
        };
        let seq = *star.value;
        if !crate::sugar::is_pure_path(&seq) {
            return None;
        }
        let need = fixed - at;
        for i in 0..need {
            args.push(substitute(&format!("__vs[{i}]"), &seq));
        }
        Some(substitute(&format!("tuple(__vs[{need}:])"), &seq))
    }

    /// Replace the `*rest` parameter of `f` with a positional one.
    fn parameter(&self, f: &mut py::StmtFunctionDef, method: bool) {
        if variadic(f, method).ok().flatten().is_none() {
            return;
        }
        let Some(rest) = f.parameters.vararg.take() else {
            return;
        };
        let range = rest.range;
        let ordered = f
            .parameters
            .posonlyargs
            .iter()
            .chain(f.parameters.args.iter())
            .any(|p| p.default.is_some());
        let default = ordered.then(|| {
            Box::new(py::Expr::Tuple(py::ExprTuple {
                node_index: Default::default(),
                range,
                elts: Vec::new(),
                ctx: py::ExprContext::Load,
                parenthesized: true,
            }))
        });
        f.parameters.args.push(py::ParameterWithDefault {
            range,
            node_index: Default::default(),
            parameter: py::Parameter {
                annotation: None,
                ..*rest
            },
            default,
        });
    }
}

fn is_super(c: &py::ExprCall) -> bool {
    matches!(&*c.func, py::Expr::Name(n) if n.id.as_str() == "super") && c.arguments.args.is_empty()
}

/// The expression `source` with `__vs` replaced by `value`.
fn substitute(source: &str, value: &py::Expr) -> py::Expr {
    let mut e = *ruff_python_parser::parse_expression(source)
        .expect("a spread argument parses")
        .into_syntax()
        .body;
    let mut fill = HashMap::default();
    fill.insert("__vs".to_string(), value.clone());
    Fill(fill).visit_expr(&mut e);
    e
}

/// The tuple of the surplus arguments of a call.
fn pack(extras: Vec<py::Expr>, range: TextRange) -> py::Expr {
    if !extras.iter().any(|e| matches!(e, py::Expr::Starred(_))) {
        return py::Expr::Tuple(py::ExprTuple {
            node_index: Default::default(),
            range,
            elts: extras,
            ctx: py::ExprContext::Load,
            parenthesized: true,
        });
    }
    // `tuple([a, b] + list(xs) + [c])`.
    let mut fill: HashMap<String, py::Expr> = HashMap::default();
    let mut parts: Vec<String> = Vec::new();
    let mut run: Vec<String> = Vec::new();
    let mut star: Option<String> = None;
    let count = extras.len();
    for (i, e) in extras.into_iter().enumerate() {
        let name = format!("__va{i}");
        match e {
            py::Expr::Starred(s) => {
                if !run.is_empty() {
                    parts.push(format!("[{}]", run.join(", ")));
                    run.clear();
                }
                fill.insert(name.clone(), *s.value);
                parts.push(format!("list({name})"));
                star = Some(name);
            }
            other => {
                fill.insert(name.clone(), other);
                run.push(name);
            }
        }
    }
    if !run.is_empty() {
        parts.push(format!("[{}]", run.join(", ")));
    }
    let source = match (&star, count) {
        (Some(only), 1) => format!("tuple({only})"),
        _ => format!("tuple({})", parts.join(" + ")),
    };
    let mut e = *ruff_python_parser::parse_expression(&source)
        .expect("a packed tuple parses")
        .into_syntax()
        .body;
    Fill(fill).visit_expr(&mut e);
    e
}

impl Transformer for Rewriter<'_> {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        match s {
            py::Stmt::ClassDef(c) => {
                for m in c.body.iter_mut() {
                    if let py::Stmt::FunctionDef(f) = m {
                        self.parameter(f, true);
                    }
                }
                self.classes.borrow_mut().push(c.name.to_string());
                walk_stmt(self, s);
                self.classes.borrow_mut().pop();
            }
            py::Stmt::FunctionDef(f) => {
                self.parameter(f, false);
                walk_stmt(self, s);
            }
            _ => walk_stmt(self, s),
        }
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        walk_expr(self, e);
        let py::Expr::Call(c) = e else {
            return;
        };
        let range = c.range();
        let Some((v, fixed)) = self.resolve(&c.func, &range) else {
            return;
        };
        let mut args = std::mem::take(&mut c.arguments.args).into_vec();
        let packed = match args
            .iter()
            .position(|a| matches!(a, py::Expr::Starred(_)))
            .filter(|at| *at < fixed)
        {
            // `f(*xs)` gives the parameters before the tuple from the
            // front of `xs` and the tuple from the rest.
            Some(at) => match self.spread(&mut args, at, fixed) {
                Some(rest) => rest,
                None => {
                    c.arguments.args = args.into_boxed_slice();
                    self.fail(
                        "a `*` argument of a function that takes *args, unless it is a name and the last argument",
                        &range,
                    );
                    return;
                }
            },
            None => {
                let extras = args.split_off(fixed.min(args.len()));
                pack(extras, range)
            }
        };
        if args.len() < fixed {
            // The parameters before the tuple are given by keyword.
            c.arguments.keywords.push(py::Keyword {
                range,
                node_index: Default::default(),
                arg: Some(py::Identifier::new(v.name.as_str(), range)),
                value: packed,
            });
        } else {
            args.push(packed);
        }
        c.arguments.args = args.into_boxed_slice();
    }
}
