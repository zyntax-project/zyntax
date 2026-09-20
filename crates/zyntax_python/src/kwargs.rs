//! `**kwargs` as keyword parameters. A function that takes `**kwargs`
//! and reads it only through `kwargs.get('k', default)` or by passing
//! `**kwargs` on to another such function is a function of keyword
//! parameters with defaults: the set of keys it reads, and those the
//! functions it forwards to read. The program is rewritten to say so
//! before inference, so each key is a typed parameter and no dict is
//! built. A `**kwargs` used any other way is refused.

use crate::{Error, Result};
use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use ruff_text_size::Ranged;
use rustc_hash::FxHashMap as HashMap;
use std::cell::RefCell;

/// A function that takes `**kwargs`: a module function by name, or a
/// method by class and name.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Target {
    Function(String),
    Method(String, String),
}

/// One key a target reads, with the default it reads it with.
#[derive(Clone, Debug)]
struct Key {
    name: String,
    default: py::Expr,
}

/// What one target does with its `**kwargs`.
#[derive(Default, Debug)]
struct Uses {
    keys: Vec<Key>,
    forwards: Vec<(Target, ruff_text_size::TextRange)>,
}

/// Rewrite every `**kwargs` function in `body`.
pub(crate) fn rewrite(body: &mut [py::Stmt]) -> Result<()> {
    let mut defs: Vec<(Target, &mut py::StmtFunctionDef)> = Vec::new();
    for s in body.iter_mut() {
        match s {
            py::Stmt::FunctionDef(f) if f.parameters.kwarg.is_some() => {
                defs.push((Target::Function(f.name.to_string()), f));
            }
            py::Stmt::ClassDef(c) => {
                let class = c.name.to_string();
                for m in c.body.iter_mut() {
                    if let py::Stmt::FunctionDef(f) = m
                        && f.parameters.kwarg.is_some()
                    {
                        defs.push((Target::Method(class.clone(), f.name.to_string()), f));
                    }
                }
            }
            _ => {}
        }
    }
    if defs.is_empty() {
        return Ok(());
    }
    // What each target reads of its own.
    let mut uses: HashMap<Target, Uses> = HashMap::default();
    for (target, f) in &defs {
        if f.parameters.vararg.is_some() {
            return Err(Error::unsupported(
                "*args alongside **kwargs",
                &*f.parameters,
            ));
        }
        let kw = f
            .parameters
            .kwarg
            .as_ref()
            .expect("picked")
            .name
            .to_string();
        let found = Finder {
            kw,
            uses: RefCell::new(Uses::default()),
            error: RefCell::new(None),
        };
        for s in &f.body {
            found.stmt(s);
        }
        if let Some(e) = found.error.into_inner() {
            return Err(e);
        }
        uses.insert(target.clone(), found.uses.into_inner());
    }
    // Every key a target's callees read is one it takes too.
    let targets: Vec<Target> = uses.keys().cloned().collect();
    loop {
        let mut changed = false;
        for t in &targets {
            let forwards = uses[t].forwards.clone();
            for (f, at) in forwards {
                let Some(theirs) = uses.get(&f).map(|u| u.keys.clone()) else {
                    return Err(Error::unsupported(
                        format!(
                            "`**kwargs` passed to {}, which takes no `**kwargs`",
                            describe(&f)
                        ),
                        &at,
                    ));
                };
                let mine = uses.get_mut(t).expect("known");
                for k in theirs {
                    if !mine.keys.iter().any(|m| m.name == k.name) {
                        mine.keys.push(k);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    // The keys each callee takes, for rewriting the forwarding calls.
    let key_names: HashMap<Target, Vec<String>> = uses
        .iter()
        .map(|(t, u)| (t.clone(), u.keys.iter().map(|k| k.name.clone()).collect()))
        .collect();
    for (target, f) in defs {
        let kw = f.parameters.kwarg.take().expect("picked").name.to_string();
        let mut keys = uses.remove(&target).expect("known").keys;
        keys.sort_by(|a, b| a.name.cmp(&b.name));
        let range = f.parameters.range();
        for k in keys {
            f.parameters.kwonlyargs.push(py::ParameterWithDefault {
                range,
                node_index: Default::default(),
                parameter: py::Parameter {
                    range,
                    node_index: Default::default(),
                    name: py::Identifier::new(k.name.as_str(), range),
                    annotation: None,
                },
                default: Some(Box::new(k.default)),
            });
        }
        let rewriter = Rewriter {
            kw,
            keys: &key_names,
        };
        for s in f.body.iter_mut() {
            rewriter.visit_stmt(s);
        }
    }
    Ok(())
}

fn describe(t: &Target) -> String {
    match t {
        Target::Function(f) => format!("`{f}`"),
        Target::Method(c, m) => format!("`{c}.{m}`"),
    }
}

/// The target a call's function names, when it is one this module
/// rewrites: a function, a class (its `__init__`), or `Class.method`.
fn callee_target(func: &py::Expr) -> Option<Target> {
    match func {
        py::Expr::Name(n) => Some(Target::Function(n.id.to_string())),
        py::Expr::Attribute(a) => match &*a.value {
            py::Expr::Name(c) => Some(Target::Method(c.id.to_string(), a.attr.to_string())),
            _ => None,
        },
        _ => None,
    }
}

/// Whether an expression is a constant a parameter default may be:
/// a literal, or a tuple of them.
fn is_constant(e: &py::Expr) -> bool {
    match e {
        py::Expr::NumberLiteral(_)
        | py::Expr::StringLiteral(_)
        | py::Expr::BytesLiteral(_)
        | py::Expr::BooleanLiteral(_)
        | py::Expr::NoneLiteral(_) => true,
        py::Expr::UnaryOp(u) => is_constant(&u.operand),
        py::Expr::Tuple(t) => t.elts.iter().all(is_constant),
        _ => false,
    }
}

/// Collects what a body does with its `**kwargs` name.
struct Finder {
    kw: String,
    uses: RefCell<Uses>,
    error: RefCell<Option<Error>>,
}

impl Finder {
    fn fail(&self, what: &str, at: &impl Ranged) {
        let mut error = self.error.borrow_mut();
        if error.is_none() {
            *error = Some(Error::unsupported(what.to_string(), at));
        }
    }

    fn is_kw(&self, e: &py::Expr) -> bool {
        matches!(e, py::Expr::Name(n) if n.id.as_str() == self.kw)
    }

    fn stmt(&self, s: &py::Stmt) {
        use ruff_python_ast::visitor::{Visitor, walk_stmt};
        struct V<'f>(&'f Finder);
        impl<'a> Visitor<'a> for V<'_> {
            fn visit_stmt(&mut self, s: &'a py::Stmt) {
                walk_stmt(self, s);
            }
            fn visit_expr(&mut self, e: &'a py::Expr) {
                self.0.expr(e, self);
            }
        }
        let mut v = V(self);
        v.visit_stmt(s);
    }

    fn expr<'a>(&self, e: &'a py::Expr, v: &mut impl ruff_python_ast::visitor::Visitor<'a>) {
        use ruff_python_ast::visitor::walk_expr;
        match e {
            // `kwargs.get('k', default)`.
            py::Expr::Call(c)
                if let py::Expr::Attribute(a) = &*c.func
                    && self.is_kw(&a.value)
                    && a.attr.as_str() == "get" =>
            {
                let args = &c.arguments.args;
                let Some(py::Expr::StringLiteral(key)) = args.first() else {
                    self.fail("`**kwargs.get` with a key that is not a string literal", c);
                    return;
                };
                if args.len() > 2 || !c.arguments.keywords.is_empty() {
                    self.fail("`**kwargs.get` with these arguments", c);
                    return;
                }
                let default = match args.get(1) {
                    Some(d) if is_constant(d) => d.clone(),
                    Some(d) => {
                        self.fail("`**kwargs.get` with a default that is not a constant", d);
                        return;
                    }
                    None => py::Expr::NoneLiteral(py::ExprNoneLiteral {
                        node_index: Default::default(),
                        range: c.range(),
                    }),
                };
                let name = key.value.to_str().to_string();
                let mut uses = self.uses.borrow_mut();
                if uses.keys.iter().any(|k| k.name == name) {
                    self.fail("`**kwargs.get` of one key twice", c);
                    return;
                }
                uses.keys.push(Key { name, default });
            }
            // `f(..., **kwargs)`.
            py::Expr::Call(c)
                if c.arguments
                    .keywords
                    .iter()
                    .any(|k| k.arg.is_none() && self.is_kw(&k.value)) =>
            {
                match callee_target(&c.func) {
                    Some(t) => self.uses.borrow_mut().forwards.push((t, c.range())),
                    None => {
                        self.fail(
                            "`**kwargs` passed to a call that is not a function or method by name",
                            c,
                        );
                        return;
                    }
                }
                for a in &c.arguments.args {
                    v.visit_expr(a);
                }
                for k in &c.arguments.keywords {
                    if !(k.arg.is_none() && self.is_kw(&k.value)) {
                        v.visit_expr(&k.value);
                    }
                }
            }
            _ if self.is_kw(e) => {
                self.fail(
                    "`**kwargs` used other than through .get or passed on with **",
                    e,
                );
            }
            _ => walk_expr(v, e),
        }
    }
}

/// Replaces the reads and forwards a [`Finder`] accepted.
struct Rewriter<'k> {
    kw: String,
    keys: &'k HashMap<Target, Vec<String>>,
}

impl Rewriter<'_> {
    fn name(&self, id: &str, range: ruff_text_size::TextRange) -> py::Expr {
        py::Expr::Name(py::ExprName {
            node_index: Default::default(),
            range,
            id: py::name::Name::new(id),
            ctx: py::ExprContext::Load,
        })
    }
}

impl Transformer for Rewriter<'_> {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        walk_stmt(self, s);
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        let is_kw = |x: &py::Expr| matches!(x, py::Expr::Name(n) if n.id.as_str() == self.kw);
        if let py::Expr::Call(c) = e {
            // `kwargs.get('k', d)` is the parameter `k`.
            if let py::Expr::Attribute(a) = &*c.func
                && is_kw(&a.value)
                && a.attr.as_str() == "get"
                && let Some(py::Expr::StringLiteral(key)) = c.arguments.args.first()
            {
                let range = c.range();
                *e = self.name(key.value.to_str(), range);
                return;
            }
            // `f(..., **kwargs)` passes each key f takes by name.
            if let Some(at) = c
                .arguments
                .keywords
                .iter()
                .position(|k| k.arg.is_none() && is_kw(&k.value))
            {
                let range = c.arguments.keywords[at].range();
                c.arguments.keywords.remove(at);
                let names = callee_target(&c.func)
                    .and_then(|t| self.keys.get(&t))
                    .cloned()
                    .unwrap_or_default();
                for k in names {
                    c.arguments.keywords.push(py::Keyword {
                        range,
                        node_index: Default::default(),
                        arg: Some(py::Identifier::new(k.as_str(), range)),
                        value: self.name(&k, range),
                    });
                }
            }
        }
        walk_expr(self, e);
    }
}
