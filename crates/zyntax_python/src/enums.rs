//! `Enum`, `IntEnum` and `StrEnum` as the class a program would write.
//! The members of `class Color(Enum)` are instances of the class made
//! after it and kept as class attributes and in a list. `Color(2)` and
//! `Color["RED"]` call functions that search the list, and the class
//! named where a program iterates (`for c in Color`, `list(Color)`,
//! `len(Color)`) is the list.
//!
//! The values of an `auto()` count from 1, and `StrEnum` takes the
//! lowered name. An enum that defines `__init__` or is a `Flag` is not
//! read.

use crate::dataclass::Fill;
use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr};
use rustc_hash::FxHashMap as HashMap;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Plain,
    Int,
    Str,
}

fn base_kind(e: &py::Expr) -> Option<Kind> {
    let name = match e {
        py::Expr::Name(n) => n.id.as_str(),
        py::Expr::Attribute(a) => a.attr.as_str(),
        _ => return None,
    };
    Some(match name {
        "Enum" => Kind::Plain,
        "IntEnum" => Kind::Int,
        "StrEnum" => Kind::Str,
        _ => return None,
    })
}

/// The members of a class with their values, and the rest of its body.
type Members = (Vec<(String, py::Expr)>, Vec<py::Stmt>);

/// The members a class lists, with each value: an expression of the
/// program, or the counter `auto()` asks for.
fn members(c: &mut py::StmtClassDef, kind: Kind) -> Option<Members> {
    let mut found: Vec<(String, py::Expr)> = Vec::new();
    let mut rest: Vec<py::Stmt> = Vec::new();
    let mut counter: i64 = 1;
    for s in std::mem::take(&mut c.body) {
        match s {
            py::Stmt::Assign(a)
                if a.targets.len() == 1
                    && matches!(&a.targets[0], py::Expr::Name(n)
                        if !n.id.as_str().starts_with('_')) =>
            {
                let py::Expr::Name(n) = &a.targets[0] else {
                    unreachable!("matched a name")
                };
                let name = n.id.to_string();
                let value = match &*a.value {
                    py::Expr::Call(call)
                        if matches!(&*call.func, py::Expr::Name(f) if f.id.as_str() == "auto")
                            && call.arguments.args.is_empty() =>
                    {
                        let v = if kind == Kind::Str {
                            string(&name.to_lowercase())
                        } else {
                            int(counter)
                        };
                        counter += 1;
                        v
                    }
                    other => {
                        if let py::Expr::NumberLiteral(num) = other
                            && let py::Number::Int(i) = &num.value
                            && let Some(v) = i.as_i64()
                        {
                            counter = v + 1;
                        }
                        other.clone()
                    }
                };
                found.push((name, value));
            }
            // A docstring or a pass.
            py::Stmt::Pass(_) => {}
            py::Stmt::Expr(e) if matches!(*e.value, py::Expr::StringLiteral(_)) => {}
            other => rest.push(other),
        }
    }
    if rest.iter().any(|s| {
        matches!(s, py::Stmt::FunctionDef(f)
            if f.name.as_str() == "__init__" || f.name.as_str() == "__new__")
    }) {
        return None;
    }
    Some((found, rest))
}

fn int(v: i64) -> py::Expr {
    py::Expr::NumberLiteral(py::ExprNumberLiteral {
        node_index: Default::default(),
        range: Default::default(),
        value: py::Number::Int(py::Int::from(v as u64)),
    })
}

fn string(s: &str) -> py::Expr {
    py::Expr::StringLiteral(py::ExprStringLiteral {
        node_index: Default::default(),
        range: Default::default(),
        value: py::StringLiteralValue::single(py::StringLiteral {
            range: Default::default(),
            node_index: Default::default(),
            value: s.to_string().into_boxed_str(),
            flags: py::StringLiteralFlags::empty(),
        }),
    })
}

fn parse(source: &str) -> Vec<py::Stmt> {
    ruff_python_parser::parse_module(source)
        .expect("an enum expansion parses")
        .into_syntax()
        .body
        .into_iter()
        .collect()
}

/// Rewrite every enum of `body`.
pub(crate) fn rewrite(body: &mut Vec<py::Stmt>, origins: &mut Vec<Option<String>>) {
    let mut names: Vec<String> = Vec::new();
    let mut out = Vec::with_capacity(body.len());
    let mut kept = Vec::with_capacity(body.len());
    for (s, origin) in std::mem::take(body)
        .into_iter()
        .zip(std::mem::take(origins))
    {
        match s {
            py::Stmt::ClassDef(mut c) => {
                let kind = c
                    .arguments
                    .as_ref()
                    .and_then(|a| a.args.iter().find_map(base_kind));
                let expanded = kind.and_then(|k| expand(&mut c, k));
                match expanded {
                    Some(stmts) => {
                        names.push(c.name.to_string());
                        kept.extend(std::iter::repeat_n(origin, stmts.len()));
                        out.extend(stmts);
                    }
                    None => {
                        kept.push(origin);
                        out.push(py::Stmt::ClassDef(c));
                    }
                }
            }
            other => {
                kept.push(origin);
                out.push(other);
            }
        }
    }
    *body = out;
    *origins = kept;
    if names.is_empty() {
        return;
    }
    let rewriter = Uses { enums: names };
    for s in body.iter_mut() {
        rewriter.visit_stmt_top(s);
    }
    // The imports the program made for the enum names.
    crate::decorators::drop_imports(body, origins);
}

/// The statements the enum class `c` becomes.
fn expand(c: &mut py::StmtClassDef, kind: Kind) -> Option<Vec<py::Stmt>> {
    let mut original = c.clone();
    let (found, rest) = members(&mut original, kind)?;
    let class = c.name.to_string();
    let user: Vec<String> = rest
        .iter()
        .filter_map(|s| match s {
            py::Stmt::FunctionDef(f) => Some(f.name.to_string()),
            _ => None,
        })
        .collect();
    let shown = match kind {
        Kind::Plain => format!("\"{class}.\" + self.name"),
        Kind::Int => "str(self.value)".to_string(),
        Kind::Str => "self.value".to_string(),
    };
    let mut methods = format!(
        "class __Template:\n\
         \x20   def __init__(self, name, value):\n        self.name = name\n        self.value = value\n\
         \x20   def __repr__(self):\n        return \"<{class}.\" + self.name + \": \" + repr(self.value) + \">\"\n\
         \x20   def __str__(self):\n        return {shown}\n"
    );
    if kind == Kind::Int {
        methods.push_str(&format!(
            "    def __int__(self):\n        return self.value\n\
             \x20   def __index__(self):\n        return self.value\n\
             \x20   def __eq__(self, other):\n        if isinstance(other, {class}):\n            return self.value == other.value\n        return self.value == other\n\
             \x20   def __ne__(self, other):\n        return not self.__eq__(other)\n\
             \x20   def __lt__(self, other):\n        if isinstance(other, {class}):\n            return self.value < other.value\n        return self.value < other\n\
             \x20   def __le__(self, other):\n        if isinstance(other, {class}):\n            return self.value <= other.value\n        return self.value <= other\n\
             \x20   def __gt__(self, other):\n        if isinstance(other, {class}):\n            return self.value > other.value\n        return self.value > other\n\
             \x20   def __ge__(self, other):\n        if isinstance(other, {class}):\n            return self.value >= other.value\n        return self.value >= other\n"
        ));
    }
    if kind == Kind::Int {
        // An int enum member does arithmetic as its value does.
        for (name, op) in [
            ("add", "+"),
            ("sub", "-"),
            ("mul", "*"),
            ("floordiv", "//"),
            ("mod", "%"),
            ("truediv", "/"),
            ("and", "&"),
            ("or", "|"),
            ("xor", "^"),
            ("lshift", "<<"),
            ("rshift", ">>"),
        ] {
            methods.push_str(&format!(
                "    def __{name}__(self, other):\n        if isinstance(other, {class}):\n            return self.value {op} other.value\n        return self.value {op} other\n\
                 \x20   def __r{name}__(self, other):\n        return other {op} self.value\n"
            ));
        }
    }
    let template = match parse(&methods).into_iter().next() {
        Some(py::Stmt::ClassDef(t)) => t,
        _ => return None,
    };
    let mut class_body: Vec<py::Stmt> = rest;
    for m in template.body {
        if let py::Stmt::FunctionDef(f) = &m
            && user.iter().any(|n| n == f.name.as_str())
        {
            continue;
        }
        class_body.push(m);
    }
    if class_body.is_empty() {
        class_body.push(py::Stmt::Pass(py::StmtPass {
            node_index: Default::default(),
            range: Default::default(),
        }));
    }
    c.decorator_list.clear();
    c.arguments = None;
    c.body = class_body.into_iter().collect();
    let mut out = vec![py::Stmt::ClassDef(c.clone())];
    let mut fill: HashMap<String, py::Expr> = HashMap::default();
    let mut source = String::new();
    for (i, (name, value)) in found.iter().enumerate() {
        source.push_str(&format!(
            "{class}.{name} = {class}(\"{name}\", __enum_value_{i})\n"
        ));
        fill.insert(format!("__enum_value_{i}"), value.clone());
    }
    let listed = found
        .iter()
        .map(|(n, _)| format!("{class}.{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    source.push_str(&format!("__enum_members_{class} = [{listed}]\n"));
    source.push_str(&format!(
        "def __enum_call_{class}(value):\n    if isinstance(value, {class}):\n        return value\n    for member in __enum_members_{class}:\n        if member.value == value:\n            return member\n    raise ValueError(repr(value) + \" is not a valid {class}\")\n\
         def __enum_item_{class}(name):\n    for member in __enum_members_{class}:\n        if member.name == name:\n            return member\n    raise KeyError(name)\n"
    ));
    let fill = Fill(fill);
    for mut s in parse(&source) {
        use ruff_python_ast::visitor::transformer::walk_stmt;
        walk_stmt(&fill, &mut s);
        out.push(s);
    }
    Some(out)
}

/// Rewrites what names an enum class: a call with one argument, a
/// subscript, and the class where it is iterated.
struct Uses {
    enums: Vec<String>,
}

impl Uses {
    fn is_enum(&self, e: &py::Expr) -> Option<&str> {
        match e {
            py::Expr::Name(n) => self
                .enums
                .iter()
                .find(|c| c.as_str() == n.id.as_str())
                .map(String::as_str),
            _ => None,
        }
    }

    fn members_of(&self, class: &str, range: ruff_text_size::TextRange) -> py::Expr {
        py::Expr::Name(py::ExprName {
            node_index: Default::default(),
            range,
            id: py::name::Name::new(format!("__enum_members_{class}")),
            ctx: py::ExprContext::Load,
        })
    }

    fn visit_stmt_top(&self, s: &mut py::Stmt) {
        use ruff_python_ast::visitor::transformer::walk_stmt;
        match s {
            py::Stmt::For(f) => {
                if let Some(class) = self.is_enum(&f.iter) {
                    let range = ruff_text_size::Ranged::range(&*f.iter);
                    *f.iter = self.members_of(class, range);
                }
                walk_stmt(self, s);
            }
            _ => walk_stmt(self, s),
        }
    }
}

/// The builtins that read what they are given as a sequence.
const ITERATING: &[&str] = &[
    "list",
    "tuple",
    "set",
    "frozenset",
    "sorted",
    "reversed",
    "len",
    "enumerate",
    "max",
    "min",
    "sum",
    "any",
    "all",
    "iter",
    "next",
    "zip",
    "map",
    "filter",
];

impl Transformer for Uses {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        self.visit_stmt_top(s);
    }

    fn visit_expr(&self, e: &mut py::Expr) {
        walk_expr(self, e);
        let range = ruff_text_size::Ranged::range(e);
        match e {
            // `Color(2)` looks the member up by value.
            py::Expr::Call(c)
                if c.arguments.args.len() == 1
                    && c.arguments.keywords.is_empty()
                    && self.is_enum(&c.func).is_some() =>
            {
                let class = self.is_enum(&c.func).expect("an enum").to_string();
                *c.func = py::Expr::Name(py::ExprName {
                    node_index: Default::default(),
                    range,
                    id: py::name::Name::new(format!("__enum_call_{class}")),
                    ctx: py::ExprContext::Load,
                });
            }
            // `Color["RED"]` looks it up by name.
            py::Expr::Subscript(s) if s.ctx == py::ExprContext::Load => {
                if let Some(class) = self.is_enum(&s.value).map(str::to_string) {
                    let call = py::Expr::Call(py::ExprCall {
                        node_index: Default::default(),
                        range_start: range.start(),
                        func: Box::new(py::Expr::Name(py::ExprName {
                            node_index: Default::default(),
                            range,
                            id: py::name::Name::new(format!("__enum_item_{class}")),
                            ctx: py::ExprContext::Load,
                        })),
                        arguments: py::Arguments {
                            node_index: Default::default(),
                            range,
                            args: vec![(*s.slice).clone()].into_boxed_slice(),
                            keywords: Default::default(),
                        },
                    });
                    *e = call;
                }
            }
            // A builtin that iterates reads the list of members.
            py::Expr::Call(c) if matches!(&*c.func, py::Expr::Name(f) if ITERATING.contains(&f.id.as_str())) => {
                for a in c.arguments.args.iter_mut() {
                    if let Some(class) = self.is_enum(a).map(str::to_string) {
                        *a = self.members_of(&class, range);
                    }
                }
            }
            // `[... for c in Color]`.
            py::Expr::ListComp(_) | py::Expr::SetComp(_) | py::Expr::Generator(_) => {
                let generators = match e {
                    py::Expr::ListComp(c) => &mut c.generators,
                    py::Expr::SetComp(c) => &mut c.generators,
                    py::Expr::Generator(c) => &mut c.generators,
                    _ => unreachable!("a comprehension"),
                };
                for g in generators.iter_mut() {
                    if let Some(class) = self.is_enum(&g.iter).map(str::to_string) {
                        g.iter = self.members_of(&class, range);
                    }
                }
            }
            _ => {}
        }
    }
}
