//! `@dataclass` as the methods it generates. A class decorated with it
//! has its annotated fields, in order, made into `__init__`, `__repr__`
//! and `__eq__` (and the ordering and hash methods the options ask for),
//! written as the source a program would, so the rest of the frontend
//! sees a class and no decorator.

use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr};
use rustc_hash::FxHashMap as HashMap;

/// One field of a dataclass.
#[derive(Clone)]
pub(crate) struct Field {
    name: String,
    /// The value of a field with a default.
    default: Option<py::Expr>,
    /// The call that makes the value of a `default_factory` field.
    factory: Option<py::Expr>,
}

#[derive(Default)]
struct Options {
    init: bool,
    repr: bool,
    eq: bool,
    order: bool,
    frozen: bool,
}

/// Whether `e` names the decorator: `dataclass` or `dataclasses.dataclass`.
fn is_dataclass_name(e: &py::Expr) -> bool {
    match e {
        py::Expr::Name(n) => n.id.as_str() == "dataclass",
        py::Expr::Attribute(a) => {
            a.attr.as_str() == "dataclass"
                && matches!(&*a.value, py::Expr::Name(m) if m.id.as_str() == "dataclasses")
        }
        _ => false,
    }
}

/// The options of the `@dataclass` or `@dataclass(...)` decorating `c`,
/// when it is the only decorator.
fn options(c: &py::StmtClassDef) -> Option<Options> {
    let [d] = &*c.decorator_list else {
        return None;
    };
    let mut o = Options {
        init: true,
        repr: true,
        eq: true,
        ..Options::default()
    };
    match &d.expression {
        e if is_dataclass_name(e) => Some(o),
        py::Expr::Call(call) if is_dataclass_name(&call.func) && call.arguments.args.is_empty() => {
            for k in &call.arguments.keywords {
                let name = k.arg.as_ref()?.as_str();
                let py::Expr::BooleanLiteral(b) = &k.value else {
                    return None;
                };
                match name {
                    "init" => o.init = b.value,
                    "repr" => o.repr = b.value,
                    "eq" => o.eq = b.value,
                    "order" => o.order = b.value,
                    "frozen" => o.frozen = b.value,
                    // Other options change nothing this generates.
                    "unsafe_hash" | "match_args" | "kw_only" | "slots" | "weakref_slot" => {}
                    _ => return None,
                }
            }
            Some(o)
        }
        _ => None,
    }
}

/// `field(default=..., default_factory=...)`.
fn field_call(e: &py::Expr) -> Option<&py::ExprCall> {
    let py::Expr::Call(c) = e else {
        return None;
    };
    let named = match &*c.func {
        py::Expr::Name(n) => n.id.as_str() == "field",
        py::Expr::Attribute(a) => {
            a.attr.as_str() == "field"
                && matches!(&*a.value, py::Expr::Name(m) if m.id.as_str() == "dataclasses")
        }
        _ => false,
    };
    named.then_some(c)
}

/// Whether the annotation is `ClassVar` or `ClassVar[...]`.
fn is_class_var(annotation: &py::Expr) -> bool {
    match annotation {
        py::Expr::Name(n) => n.id.as_str() == "ClassVar",
        py::Expr::Subscript(s) => is_class_var(&s.value),
        py::Expr::Attribute(a) => a.attr.as_str() == "ClassVar",
        _ => false,
    }
}

/// Replaces the placeholder names of a parsed method with the
/// expressions they stand for.
struct Fill(HashMap<String, py::Expr>);

impl Transformer for Fill {
    fn visit_expr(&self, e: &mut py::Expr) {
        if let py::Expr::Name(n) = e
            && let Some(replacement) = self.0.get(n.id.as_str())
        {
            *e = replacement.clone();
            return;
        }
        walk_expr(self, e);
    }
}

/// Make the methods of `c`, a class decorated with `@dataclass`, and
/// drop its decorator and field declarations. `known` has the fields of
/// the dataclasses defined before it, which a derived one starts with.
/// The fields are returned for the classes derived from this one.
pub(crate) fn rewrite_class(
    c: &mut py::StmtClassDef,
    known: &HashMap<String, Vec<Field>>,
) -> Option<Vec<Field>> {
    let options = options(c)?;
    let class = c.name.to_string();
    let mut fields: Vec<Field> = Vec::new();
    // The fields of the base dataclass come first.
    if let Some(args) = &c.arguments {
        for base in &args.args {
            if let py::Expr::Name(b) = base
                && let Some(inherited) = known.get(b.id.as_str())
            {
                fields.extend(inherited.iter().cloned());
            }
        }
    }
    let mut body = Vec::new();
    for s in std::mem::take(&mut c.body) {
        match s {
            py::Stmt::AnnAssign(a)
                if matches!(&*a.target, py::Expr::Name(_)) && !is_class_var(&a.annotation) =>
            {
                let py::Expr::Name(target) = &*a.target else {
                    unreachable!("matched a name")
                };
                let mut field = Field {
                    name: target.id.to_string(),
                    default: None,
                    factory: None,
                };
                if let Some(v) = a.value.as_deref() {
                    match field_call(v) {
                        Some(call) => {
                            for k in &call.arguments.keywords {
                                match k.arg.as_ref().map(|a| a.as_str()) {
                                    Some("default") => field.default = Some(k.value.clone()),
                                    Some("default_factory") => {
                                        field.factory = Some(k.value.clone())
                                    }
                                    _ => {}
                                }
                            }
                        }
                        None => field.default = Some(v.clone()),
                    }
                }
                // A field redeclared replaces the inherited one in place.
                match fields.iter_mut().find(|f| f.name == field.name) {
                    Some(slot) => *slot = field,
                    None => fields.push(field),
                }
            }
            other => body.push(other),
        }
    }
    c.decorator_list.clear();
    let defines = |name: &str| {
        body.iter()
            .any(|s| matches!(s, py::Stmt::FunctionDef(f) if f.name.as_str() == name))
    };
    let mut methods: Vec<py::Stmt> = Vec::new();
    let mut fill: HashMap<String, py::Expr> = HashMap::default();
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();

    if options.init && !defines("__init__") {
        let mut params = String::new();
        let mut assigns = String::new();
        for (i, f) in fields.iter().enumerate() {
            params.push_str(", ");
            params.push_str(&f.name);
            if let Some(d) = &f.default {
                params.push_str(&format!("=__dc_default_{i}"));
                fill.insert(format!("__dc_default_{i}"), d.clone());
            } else if let Some(factory) = &f.factory {
                params.push_str("=None");
                assigns.push_str(&format!(
                    "    if {n} is None:\n        {n} = __dc_factory_{i}()\n",
                    n = f.name
                ));
                fill.insert(format!("__dc_factory_{i}"), factory.clone());
            }
            assigns.push_str(&format!("    self.{n} = {n}\n", n = f.name));
        }
        let post = if defines("__post_init__") {
            "    self.__post_init__()\n"
        } else {
            ""
        };
        methods.push(parse(&format!(
            "def __init__(self{params}):\n{assigns}{post}    pass\n"
        )));
    }
    if options.repr && !defines("__repr__") {
        let shown: Vec<String> = names
            .iter()
            .map(|n| format!("{n}={{self.{n}!r}}"))
            .collect();
        methods.push(parse(&format!(
            "def __repr__(self):\n    return f\"{class}({})\"\n",
            shown.join(", ")
        )));
    }
    let tuple = |who: &str| {
        let items: Vec<String> = names.iter().map(|n| format!("{who}.{n}")).collect();
        match items.len() {
            0 => "()".to_string(),
            1 => format!("({},)", items[0]),
            _ => format!("({})", items.join(", ")),
        }
    };
    if options.eq && !defines("__eq__") {
        methods.push(parse(&format!(
            "def __eq__(self, other):\n    if not isinstance(other, {class}):\n        return False\n    return {} == {}\n",
            tuple("self"),
            tuple("other")
        )));
        if options.frozen && !defines("__hash__") {
            methods.push(parse(&format!(
                "def __hash__(self):\n    return hash({})\n",
                tuple("self")
            )));
        }
    }
    if options.order {
        for (name, op) in [
            ("__lt__", "<"),
            ("__le__", "<="),
            ("__gt__", ">"),
            ("__ge__", ">="),
        ] {
            if !defines(name) {
                methods.push(parse(&format!(
                    "def {name}(self, other):\n    return {} {op} {}\n",
                    tuple("self"),
                    tuple("other")
                )));
            }
        }
    }
    let fill = Fill(fill);
    for m in &mut methods {
        fill.visit_stmt_in(m);
    }
    // The source ends each `__init__` in `pass`, which is dropped where
    // the body has statements.
    for m in &mut methods {
        if let py::Stmt::FunctionDef(f) = m
            && f.body.len() > 1
            && matches!(f.body.last(), Some(py::Stmt::Pass(_)))
        {
            f.body.pop();
        }
    }
    body.extend(methods);
    c.body = body.into_iter().collect();
    Some(fields)
}

impl Fill {
    fn visit_stmt_in(&self, s: &mut py::Stmt) {
        use ruff_python_ast::visitor::transformer::walk_stmt;
        walk_stmt(self, s);
    }
}

fn parse(source: &str) -> py::Stmt {
    ruff_python_parser::parse_module(source)
        .expect("a dataclass method parses")
        .into_syntax()
        .body
        .into_iter()
        .next()
        .expect("one method")
}
