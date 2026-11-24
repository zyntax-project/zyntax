//! Displays and calls that spread other values in, written as the
//! concatenation or merge. `[a, *xs]`, `(*xs, b)` and `{*xs}` are the
//! lists of their parts added together. `{**a, "k": v}`, `dict(a, k=v)`
//! and `d.update(a, k=v)` are a dict display or a call to a module
//! function that copies its first operand and updates the copy with the
//! second. The rest of the frontend sees only plain displays,
//! `dict(x)` and `update` with one operand.

use crate::dataclass::Fill;
use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::FxHashMap as HashMap;
use std::cell::Cell;

/// The function every merge calls.
const MERGE: &str = "dict$merge";

const HELPER: &str = "def merge(a, b):\n    r = dict(a)\n    r.update(b)\n    return r\n";

/// One operand of a merge: entries written out, or a dict (or pairs)
/// spread in.
enum Part {
    Items(Vec<py::DictItem>),
    Spread(py::Expr),
}

/// Rewrite every merge of `body`.
pub(crate) fn rewrite(body: &mut Vec<py::Stmt>, origins: &mut Vec<Option<String>>) {
    let rewriter = Rewriter {
        own_update: defines_update(body),
        used: Cell::new(false),
    };
    for s in body.iter_mut() {
        rewriter.visit_stmt(s);
    }
    if rewriter.used.get() {
        let mut helper = ruff_python_parser::parse_module(HELPER)
            .expect("the merge helper parses")
            .into_syntax()
            .body
            .into_iter()
            .next()
            .expect("one function");
        if let py::Stmt::FunctionDef(f) = &mut helper {
            f.name = py::Identifier::new(MERGE, f.name.range);
        }
        body.push(helper);
        origins.push(None);
    }
}

/// Whether the program has a method named `update`, which a call with
/// keywords may then be a call of.
fn defines_update(body: &[py::Stmt]) -> bool {
    use ruff_python_ast::visitor::{Visitor, walk_stmt as read};
    struct V(bool);
    impl<'a> Visitor<'a> for V {
        fn visit_stmt(&mut self, s: &'a py::Stmt) {
            if let py::Stmt::ClassDef(c) = s
                && c.body
                    .iter()
                    .any(|m| matches!(m, py::Stmt::FunctionDef(f) if f.name.as_str() == "update"))
            {
                self.0 = true;
            }
            read(self, s);
        }
    }
    let mut v = V(false);
    for s in body {
        v.visit_stmt(s);
    }
    v.0
}

struct Rewriter {
    own_update: bool,
    used: Cell<bool>,
}

fn name(id: &str, range: TextRange) -> py::Expr {
    py::Expr::Name(py::ExprName {
        node_index: Default::default(),
        range,
        id: py::name::Name::new(id),
        ctx: py::ExprContext::Load,
    })
}

fn call(func: &str, args: Vec<py::Expr>, range: TextRange) -> py::Expr {
    py::Expr::Call(py::ExprCall {
        node_index: Default::default(),
        range_start: range.start(),
        func: Box::new(name(func, range)),
        arguments: py::Arguments {
            node_index: Default::default(),
            range,
            args: args.into_boxed_slice(),
            keywords: Default::default(),
        },
    })
}

fn display(items: Vec<py::DictItem>, range: TextRange) -> py::Expr {
    py::Expr::Dict(py::ExprDict {
        node_index: Default::default(),
        range,
        items,
    })
}

fn key(k: &str, range: TextRange) -> py::Expr {
    py::Expr::StringLiteral(py::ExprStringLiteral {
        node_index: Default::default(),
        range,
        value: py::StringLiteralValue::single(py::StringLiteral {
            range,
            node_index: Default::default(),
            value: k.to_string().into_boxed_str(),
            flags: py::StringLiteralFlags::empty(),
        }),
    })
}

/// The operands of a dict display, in order.
fn parts_of_items(items: Vec<py::DictItem>) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut run = Vec::new();
    for item in items {
        match item.key {
            Some(_) => run.push(item),
            None => {
                if !run.is_empty() {
                    parts.push(Part::Items(std::mem::take(&mut run)));
                }
                parts.push(Part::Spread(item.value));
            }
        }
    }
    if !run.is_empty() {
        parts.push(Part::Items(run));
    }
    parts
}

/// The operands of `f(positional, k=v, **spread)`, in order.
fn parts_of_call(positional: Option<py::Expr>, keywords: Vec<py::Keyword>) -> Vec<Part> {
    let mut parts: Vec<Part> = positional.into_iter().map(Part::Spread).collect();
    let mut run = Vec::new();
    for k in keywords {
        match k.arg {
            Some(arg) => run.push(py::DictItem {
                key: Some(key(arg.as_str(), arg.range)),
                value: k.value,
            }),
            None => {
                if !run.is_empty() {
                    parts.push(Part::Items(std::mem::take(&mut run)));
                }
                parts.push(Part::Spread(k.value));
            }
        }
    }
    if !run.is_empty() {
        parts.push(Part::Items(run));
    }
    parts
}

/// What a display with `*` elements is made into.
#[derive(Clone, Copy)]
pub(crate) enum Wrap {
    List,
    Tuple,
    Set,
}

/// The display `elts`, some of them `*xs`, as the list of its parts
/// added together: `tuple([a] + list(xs) + [b])`.
pub(crate) fn display_of(wrap: Wrap, elts: Vec<py::Expr>) -> py::Expr {
    let mut fill: HashMap<String, py::Expr> = HashMap::default();
    let mut parts: Vec<String> = Vec::new();
    let mut run: Vec<String> = Vec::new();
    for (i, e) in elts.into_iter().enumerate() {
        let name = format!("__sp{i}");
        match e {
            py::Expr::Starred(star) => {
                if !run.is_empty() {
                    parts.push(format!("[{}]", run.join(", ")));
                    run.clear();
                }
                fill.insert(name.clone(), *star.value);
                parts.push(format!("list({name})"));
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
    let joined = parts.join(" + ");
    let source = match (wrap, parts.len()) {
        (Wrap::List, _) => joined,
        // `tuple(xs)` and `set(xs)` take the one operand as it is.
        (Wrap::Tuple, 1) if joined.starts_with("list(") => format!("tuple{}", &joined[4..]),
        (Wrap::Set, 1) if joined.starts_with("list(") => format!("set{}", &joined[4..]),
        (Wrap::Tuple, _) => format!("tuple({joined})"),
        (Wrap::Set, _) => format!("set({joined})"),
    };
    let mut e = *ruff_python_parser::parse_expression(&source)
        .expect("a spread display parses")
        .into_syntax()
        .body;
    Fill(fill).visit_expr(&mut e);
    e
}

fn has_star(elts: &[py::Expr]) -> bool {
    elts.iter().any(|e| matches!(e, py::Expr::Starred(_)))
}

impl Rewriter {
    /// The dict the operands merge into, later ones winning.
    fn fold(&self, parts: Vec<Part>, range: TextRange) -> py::Expr {
        let operand = |p: Part| match p {
            Part::Items(items) => display(items, range),
            Part::Spread(e) => e,
        };
        let mut parts = parts.into_iter();
        let first = parts.next().expect("a merge has an operand");
        let rest: Vec<Part> = parts.collect();
        if rest.is_empty() {
            return match first {
                Part::Items(items) => display(items, range),
                Part::Spread(e) => call("dict", vec![e], range),
            };
        }
        self.used.set(true);
        let mut merged = operand(first);
        for p in rest {
            merged = call(MERGE, vec![merged, operand(p)], range);
        }
        merged
    }
}

/// A call's one positional argument, when it has at most one and it is
/// not a `*` argument.
fn positional(c: &mut py::ExprCall) -> Option<Option<py::Expr>> {
    match &*c.arguments.args {
        [] => Some(None),
        [py::Expr::Starred(_)] => None,
        [_] => Some(std::mem::take(&mut c.arguments.args).into_vec().pop()),
        _ => None,
    }
}

impl Transformer for Rewriter {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        walk_stmt(self, s);
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        walk_expr(self, e);
        let range = e.range();
        match e {
            py::Expr::List(l) if l.ctx == py::ExprContext::Load && has_star(&l.elts) => {
                *e = display_of(Wrap::List, std::mem::take(&mut l.elts));
            }
            py::Expr::Tuple(t) if t.ctx == py::ExprContext::Load && has_star(&t.elts) => {
                *e = display_of(Wrap::Tuple, std::mem::take(&mut t.elts));
            }
            py::Expr::Set(s) if has_star(&s.elts) => {
                *e = display_of(Wrap::Set, std::mem::take(&mut s.elts));
            }
            py::Expr::Dict(d) if d.items.iter().any(|i| i.key.is_none()) => {
                let parts = parts_of_items(std::mem::take(&mut d.items));
                *e = self.fold(parts, range);
            }
            py::Expr::Call(c) if !c.arguments.keywords.is_empty() => {
                let is_dict = matches!(&*c.func, py::Expr::Name(n) if n.id.as_str() == "dict");
                let is_update = !self.own_update
                    && matches!(&*c.func, py::Expr::Attribute(a) if a.attr.as_str() == "update");
                if !is_dict && !is_update {
                    return;
                }
                let Some(pos) = positional(c) else {
                    return;
                };
                let keywords = std::mem::take(&mut c.arguments.keywords).into_iter();
                let parts = parts_of_call(pos, keywords.collect());
                let merged = self.fold(parts, range);
                if is_dict {
                    *e = merged;
                } else {
                    c.arguments.args = vec![merged].into_boxed_slice();
                }
            }
            _ => {}
        }
    }
}
