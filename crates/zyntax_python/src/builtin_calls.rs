//! Calls through the name of a builtin type, rewritten to the code they
//! stand for before inference, so the rest of the frontend sees only the
//! shapes it models:
//!
//! - `str.upper(s)` is `s.upper()`: the type's method called on its
//!   first argument.
//! - `dict.fromkeys(xs, v)` is `{k: v for k in xs}`; a value that is not
//!   a constant or a name is evaluated once, as the call does, by
//!   iterating over a list that holds it.
//! - `str.maketrans(a, b, z)` is the dict `str.translate` reads: the
//!   ordinal of each character of `a` mapped to the ordinal of the one of
//!   `b` at its place, and each one of `z` to None.
//!
//! A builtin type name that the program binds itself is left alone.

use crate::dataclass::Fill;
use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use ruff_text_size::Ranged;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cell::Cell;

/// The types whose methods can be called through the type's name.
const TYPES: [&str; 6] = ["str", "bytes", "list", "dict", "set", "tuple"];

/// Rewrite every such call of `body`.
pub(crate) fn rewrite(body: &mut [py::Stmt]) {
    let shadowed: HashSet<&'static str> = TYPES
        .into_iter()
        .filter(|t| crate::sugar::binds_module_name(body, t))
        .collect();
    let rewriter = Rewriter {
        shadowed,
        made: Cell::new(0),
    };
    for s in body.iter_mut() {
        rewriter.visit_stmt(s);
    }
}

struct Rewriter {
    shadowed: HashSet<&'static str>,
    /// How many names have been made, for the next to differ.
    made: Cell<usize>,
}

fn name_expr(id: &str, range: ruff_text_size::TextRange, ctx: py::ExprContext) -> py::Expr {
    py::Expr::Name(py::ExprName {
        node_index: Default::default(),
        range,
        id: py::name::Name::new(id),
        ctx,
    })
}

impl Rewriter {
    /// The type and method of `T.method`, for `T` a builtin type name the
    /// program leaves alone.
    fn through_type<'e>(&self, func: &'e py::Expr) -> Option<(&'e str, &'e str)> {
        let py::Expr::Attribute(a) = func else {
            return None;
        };
        let py::Expr::Name(t) = &*a.value else {
            return None;
        };
        let ty = t.id.as_str();
        (TYPES.contains(&ty) && !self.shadowed.contains(ty)).then_some((ty, a.attr.as_str()))
    }

    fn fromkeys(&self, c: &mut py::ExprCall) -> py::Expr {
        let range = c.range();
        let n = self.made.get();
        self.made.set(n + 1);
        let (key, held) = (format!("fromkeys$key{n}"), format!("fromkeys$value{n}"));
        let mut args = std::mem::take(&mut c.arguments.args).into_vec();
        let value = (args.len() == 2).then(|| args.pop().expect("two arguments"));
        let iter = args.pop().expect("keys");
        let store = |id: &str| name_expr(id, range, py::ExprContext::Store);
        let comprehension = |target, iter| py::Comprehension {
            range,
            node_index: Default::default(),
            target,
            iter,
            ifs: Vec::new(),
            is_async: false,
        };
        let mut generators = Vec::new();
        let element = match value {
            None => py::Expr::NoneLiteral(py::ExprNoneLiteral {
                node_index: Default::default(),
                range,
            }),
            Some(v)
                if matches!(
                    v,
                    py::Expr::NumberLiteral(_)
                        | py::Expr::StringLiteral(_)
                        | py::Expr::BytesLiteral(_)
                        | py::Expr::BooleanLiteral(_)
                        | py::Expr::NoneLiteral(_)
                        | py::Expr::Name(_)
                ) =>
            {
                v
            }
            Some(v) => {
                generators.push(comprehension(
                    store(&held),
                    py::Expr::List(py::ExprList {
                        node_index: Default::default(),
                        range,
                        elts: vec![v],
                        ctx: py::ExprContext::Load,
                    }),
                ));
                name_expr(&held, range, py::ExprContext::Load)
            }
        };
        generators.push(comprehension(store(&key), iter));
        py::Expr::DictComp(py::ExprDictComp {
            node_index: Default::default(),
            range,
            key: Some(Box::new(name_expr(&key, range, py::ExprContext::Load))),
            value: Box::new(element),
            generators,
        })
    }

    fn maketrans(&self, c: &mut py::ExprCall) -> py::Expr {
        let mut args = std::mem::take(&mut c.arguments.args).into_vec();
        let deleted = (args.len() == 3).then(|| args.pop().expect("three arguments"));
        let to = args.pop().expect("two arguments");
        let from = args.pop().expect("two arguments");
        let empty =
            |e: &py::Expr| matches!(e, py::Expr::StringLiteral(s) if s.value.to_str().is_empty());
        let pairs = "{ord(__x): ord(__y) for __x, __y in zip(__a, __b)}";
        let drops = "{ord(__c): None for __c in __z}";
        let template = match (empty(&from) && empty(&to), deleted.is_some()) {
            (true, true) => drops.to_string(),
            (false, true) => format!("{{**{pairs}, **{drops}}}"),
            _ => pairs.to_string(),
        };
        let mut table = *ruff_python_parser::parse_expression(&template)
            .expect("a translation table template parses")
            .into_syntax()
            .body;
        let mut fill: HashMap<String, py::Expr> = HashMap::default();
        for (name, value) in [("__a", from), ("__b", to)]
            .into_iter()
            .chain(deleted.map(|z| ("__z", z)))
        {
            fill.insert(name.to_string(), value);
        }
        Fill(fill).visit_expr(&mut table);
        table
    }
}

impl Transformer for Rewriter {
    fn visit_expr(&self, e: &mut py::Expr) {
        walk_expr(self, e);
        let py::Expr::Call(c) = e else {
            return;
        };
        let Some((ty, method)) = self.through_type(&c.func) else {
            return;
        };
        let plain = c.arguments.keywords.is_empty()
            && !c
                .arguments
                .args
                .iter()
                .any(|a| matches!(a, py::Expr::Starred(_)));
        match (ty, method, c.arguments.args.len()) {
            ("dict", "fromkeys", 1 | 2) if plain => *e = self.fromkeys(c),
            ("str", "maketrans", 2 | 3) if plain => *e = self.maketrans(c),
            (_, _, 1..) if crate::diagnostics::is_unbound_method(ty, method) => {
                if matches!(c.arguments.args[0], py::Expr::Starred(_)) {
                    return;
                }
                let mut args = std::mem::take(&mut c.arguments.args).into_vec();
                let receiver = args.remove(0);
                c.arguments.args = args.into_boxed_slice();
                if let py::Expr::Attribute(a) = &mut *c.func {
                    *a.value = receiver;
                }
            }
            _ => {}
        }
    }

    fn visit_stmt(&self, s: &mut py::Stmt) {
        walk_stmt(self, s);
    }
}
