//! `namedtuple` and `typing.NamedTuple` as the class a program would
//! write: the fields are attributes set by `__init__`, and the class is
//! iterable, indexable, comparable and prints as the tuple type does.
//! `P = namedtuple("P", ["x", "y"])` and `class P(NamedTuple)` with
//! annotated fields are the forms read; the field names must be literals.

use crate::dataclass::Fill;
use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::Transformer;
use rustc_hash::FxHashMap as HashMap;

/// One field with the default it takes, when it has one.
struct Field {
    name: String,
    default: Option<py::Expr>,
}

/// What `_replace` takes for a field it was not given.
const MISSING: &str = "__NT_MISSING";

/// Rewrite every named tuple of `body`.
pub(crate) fn rewrite(body: &mut Vec<py::Stmt>, origins: &mut Vec<Option<String>>) {
    let mut made = false;
    crate::walk::expand_module(body, origins, &mut |s| match s {
        py::Stmt::Assign(a) => match call_form(&a) {
            Some(class) => {
                made = true;
                vec![class]
            }
            None => vec![py::Stmt::Assign(a)],
        },
        py::Stmt::ClassDef(mut c) => {
            let before = c.arguments.as_ref().map_or(0, |a| a.args.len());
            class_form(&mut c);
            made |= c.arguments.as_ref().map_or(0, |a| a.args.len()) < before;
            vec![py::Stmt::ClassDef(c)]
        }
        other => vec![other],
    });
    if made {
        let sentinel = ruff_python_parser::parse_module(&format!(
            "class __NTMissing:\n    pass\n{MISSING} = __NTMissing()\n"
        ))
        .expect("the sentinel parses")
        .into_syntax()
        .body;
        for (at, s) in sentinel.into_iter().enumerate() {
            body.insert(at, s);
            origins.insert(at, None);
        }
    }
}

/// Whether `e` names `namedtuple`, bare, qualified by the bundled
/// module, or through `collections`.
fn is_namedtuple(e: &py::Expr) -> bool {
    match e {
        py::Expr::Name(n) => matches!(n.id.as_str(), "namedtuple" | "collections$namedtuple"),
        py::Expr::Attribute(a) => {
            a.attr.as_str() == "namedtuple"
                && matches!(&*a.value, py::Expr::Name(m)
                    if matches!(m.id.as_str(), "collections" | "collections$"))
        }
        _ => false,
    }
}

fn is_named_tuple_base(e: &py::Expr) -> bool {
    match e {
        py::Expr::Name(n) => matches!(n.id.as_str(), "NamedTuple" | "typing$NamedTuple"),
        py::Expr::Attribute(a) => a.attr.as_str() == "NamedTuple",
        _ => false,
    }
}

/// The field names a literal lists: `["x", "y"]`, `("x", "y")`, or the
/// string `"x y"` or `"x, y"`.
fn field_names(e: &py::Expr) -> Option<Vec<String>> {
    match e {
        py::Expr::StringLiteral(s) => Some(
            s.value
                .to_str()
                .split([',', ' '])
                .filter(|n| !n.is_empty())
                .map(str::to_string)
                .collect(),
        ),
        py::Expr::List(l) => strings(&l.elts),
        py::Expr::Tuple(t) => strings(&t.elts),
        _ => None,
    }
}

fn strings(elts: &[py::Expr]) -> Option<Vec<String>> {
    elts.iter()
        .map(|e| match e {
            py::Expr::StringLiteral(s) => Some(s.value.to_str().to_string()),
            _ => None,
        })
        .collect()
}

/// `P = namedtuple("P", fields, defaults=(...))` as a class.
fn call_form(a: &py::StmtAssign) -> Option<py::Stmt> {
    let [py::Expr::Name(target)] = &a.targets[..] else {
        return None;
    };
    let py::Expr::Call(c) = &*a.value else {
        return None;
    };
    if !is_namedtuple(&c.func) || c.arguments.args.len() != 2 {
        return None;
    }
    let names = field_names(&c.arguments.args[1])?;
    let mut defaults: Vec<py::Expr> = Vec::new();
    for k in &c.arguments.keywords {
        match k.arg.as_ref().map(|a| a.as_str()) {
            Some("defaults") => match &k.value {
                py::Expr::Tuple(t) => defaults = t.elts.clone(),
                py::Expr::List(l) => defaults = l.elts.clone(),
                _ => return None,
            },
            Some("rename") | Some("module") => {}
            _ => return None,
        }
    }
    let first_default = names.len().checked_sub(defaults.len())?;
    let mut defaults = defaults.into_iter();
    let fields: Vec<Field> = names
        .into_iter()
        .enumerate()
        .map(|(i, name)| Field {
            name,
            default: if i >= first_default {
                defaults.next()
            } else {
                None
            },
        })
        .collect();
    let class_name = target.id.to_string();
    let mut class = parse_class(&format!("class {class_name}:\n    pass\n"));
    let methods = methods(&class_name, &fields);
    class.body = methods.into_iter().collect();
    Some(py::Stmt::ClassDef(class))
}

/// `class P(NamedTuple)` with annotated fields: the fields are removed
/// from the body and the methods added to it.
fn class_form(c: &mut py::StmtClassDef) {
    let Some(args) = &mut c.arguments else {
        return;
    };
    let before = args.args.len();
    args.args = std::mem::take(&mut args.args)
        .into_vec()
        .into_iter()
        .filter(|b| !is_named_tuple_base(b))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    if args.args.len() == before {
        return;
    }
    let mut fields = Vec::new();
    let mut kept = Vec::new();
    for s in std::mem::take(&mut c.body) {
        match s {
            py::Stmt::AnnAssign(a) if matches!(&*a.target, py::Expr::Name(_)) => {
                let py::Expr::Name(n) = &*a.target else {
                    unreachable!("matched a name")
                };
                fields.push(Field {
                    name: n.id.to_string(),
                    default: a.value.as_deref().cloned(),
                });
            }
            other => kept.push(other),
        }
    }
    let own: Vec<String> = kept
        .iter()
        .filter_map(|s| match s {
            py::Stmt::FunctionDef(f) => Some(f.name.to_string()),
            _ => None,
        })
        .collect();
    let class_name = c.name.to_string();
    for m in methods(&class_name, &fields) {
        if let py::Stmt::FunctionDef(f) = &m
            && own.iter().any(|n| n == f.name.as_str())
        {
            continue;
        }
        kept.push(m);
    }
    c.body = kept.into_iter().collect();
}

fn parse_class(source: &str) -> py::StmtClassDef {
    match ruff_python_parser::parse_module(source)
        .expect("a named tuple class parses")
        .into_syntax()
        .body
        .into_iter()
        .next()
        .expect("one class")
    {
        py::Stmt::ClassDef(c) => c,
        _ => unreachable!("a class"),
    }
}

/// The statements of the class body for `fields`.
fn methods(class: &str, fields: &[Field]) -> Vec<py::Stmt> {
    let n = fields.len();
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    let tuple_of = |who: &str| -> String {
        match n {
            0 => "()".to_string(),
            1 => format!("({who}.{},)", names[0]),
            _ => format!(
                "({})",
                names
                    .iter()
                    .map(|f| format!("{who}.{f}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    };
    let mut fill: HashMap<String, py::Expr> = HashMap::default();
    let mut params = String::new();
    let mut sets = String::new();
    for (i, f) in fields.iter().enumerate() {
        params.push_str(&format!(", {}", f.name));
        if let Some(d) = &f.default {
            params.push_str(&format!("=__nt_default_{i}"));
            fill.insert(format!("__nt_default_{i}"), d.clone());
        }
        sets.push_str(&format!("        self.{0} = {0}\n", f.name));
    }
    if n == 0 {
        sets.push_str("        pass\n");
    }
    let shown = names
        .iter()
        .map(|f| format!("\"{f}=\" + repr(self.{f})"))
        .collect::<Vec<_>>()
        .join(" + \", \" + ");
    let repr_body = if n == 0 {
        format!("\"{class}()\"")
    } else {
        format!("\"{class}(\" + {shown} + \")\"")
    };
    let yields = if n == 0 {
        "        return\n        yield\n".to_string()
    } else {
        names
            .iter()
            .map(|f| format!("        yield self.{f}\n"))
            .collect::<String>()
    };
    let gets = names
        .iter()
        .enumerate()
        .map(|(i, f)| format!("        if i == {i}:\n            return self.{f}\n"))
        .collect::<String>();
    let quoted = names
        .iter()
        .map(|f| format!("\"{f}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let dict_items = names
        .iter()
        .map(|f| format!("\"{f}\": self.{f}"))
        .collect::<Vec<_>>()
        .join(", ");
    let replace_params = names
        .iter()
        .map(|f| format!(", {f}={MISSING}"))
        .collect::<String>();
    let replace_fields = names
        .iter()
        .map(|f| format!("{f} if {f} is not {MISSING} else self.{f}"))
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "class __Template:\n\
         \x20   _fields = ({quoted}{trailing})\n\
         \x20   __match_args__ = ({quoted}{trailing})\n\
         \x20   def __init__(self{params}):\n{sets}\
         \x20   def __repr__(self):\n        return {repr_body}\n\
         \x20   def __iter__(self):\n{yields}\
         \x20   def __len__(self):\n        return {n}\n\
         \x20   def __getitem__(self, i):\n        if i < 0:\n            i += {n}\n{gets}        raise IndexError(\"tuple index out of range\")\n\
         \x20   def _tuple(self):\n        return {own}\n\
         \x20   def __eq__(self, other):\n        if isinstance(other, {class}):\n            return {own} == {other}\n        if isinstance(other, tuple):\n            return {own} == other\n        return False\n\
         \x20   def __ne__(self, other):\n        return not self.__eq__(other)\n\
         \x20   def __lt__(self, other):\n        if isinstance(other, {class}):\n            return {own} < {other}\n        return {own} < other\n\
         \x20   def __le__(self, other):\n        if isinstance(other, {class}):\n            return {own} <= {other}\n        return {own} <= other\n\
         \x20   def __gt__(self, other):\n        if isinstance(other, {class}):\n            return {own} > {other}\n        return {own} > other\n\
         \x20   def __ge__(self, other):\n        if isinstance(other, {class}):\n            return {own} >= {other}\n        return {own} >= other\n\
         \x20   def _asdict(self):\n        return {{{dict_items}}}\n\
         \x20   def _replace(self{replace_params}):\n        return {class}({replace_fields})\n",
        trailing = if n == 1 { "," } else { "" },
        own = tuple_of("self"),
        other = tuple_of("other"),
    );
    let template = parse_class(&source);
    let fill = Fill(fill);
    let mut out: Vec<py::Stmt> = Vec::new();
    for mut s in template.body {
        if let py::Stmt::FunctionDef(f) = &mut s {
            for p in f
                .parameters
                .posonlyargs
                .iter_mut()
                .chain(f.parameters.args.iter_mut())
            {
                if let Some(d) = &mut p.default {
                    fill.visit_expr(d);
                }
            }
        }
        out.push(s);
    }
    out
}
