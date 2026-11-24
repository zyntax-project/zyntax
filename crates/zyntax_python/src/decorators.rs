//! Decorators of the standard library, written as the code they make.
//! A module function under `@lru_cache` or `@cache` is a wrapper that
//! looks its arguments up in a dict kept for it. One under
//! `@contextmanager` is a function that makes an instance of a class
//! holding the generator, whose `__enter__` runs it to its `yield` and
//! whose `__exit__` runs the rest.
//!
//! A context manager's code after the `yield` runs when the block
//! raises too, where Python raises the exception at the `yield`.

use crate::{Error, Result};
use ruff_python_ast as py;
use ruff_text_size::Ranged;

/// The names the program imports for what this pass rewrites.
const IMPORTED: &[(&str, &[&str])] = &[
    ("functools", &["lru_cache", "cache"]),
    ("contextlib", &["contextmanager"]),
    ("abc", &["ABC", "ABCMeta", "abstractmethod"]),
    ("enum", &["Enum", "IntEnum", "StrEnum", "auto", "unique"]),
];

fn is_imported(module: &str, name: &str) -> bool {
    IMPORTED
        .iter()
        .any(|(m, names)| *m == module && names.contains(&name))
}

enum Kind {
    Cache,
    ContextManager,
}

/// The decorator of `f`, when it is one of these and the only one.
fn kind(f: &py::StmtFunctionDef) -> Option<Kind> {
    let [d] = &*f.decorator_list else {
        return None;
    };
    let head = match &d.expression {
        py::Expr::Call(c) => &*c.func,
        other => other,
    };
    let name = match head {
        py::Expr::Name(n) => n.id.as_str(),
        py::Expr::Attribute(a)
            if matches!(&*a.value, py::Expr::Name(m)
                if matches!(m.id.as_str(), "functools" | "contextlib")) =>
        {
            a.attr.as_str()
        }
        _ => return None,
    };
    match (name, matches!(&d.expression, py::Expr::Call(_))) {
        ("lru_cache" | "cache", _) => Some(Kind::Cache),
        ("contextmanager", false) => Some(Kind::ContextManager),
        _ => None,
    }
}

/// Rewrite every decorated function of `body`.
pub(crate) fn rewrite(body: &mut Vec<py::Stmt>, origins: &mut Vec<Option<String>>) -> Result<()> {
    // A body of `...` is a body of `pass`.
    crate::walk::expand_module(body, origins, &mut |s| match s {
        py::Stmt::Expr(e) if matches!(&*e.value, py::Expr::EllipsisLiteral(_)) => {
            vec![py::Stmt::Pass(py::StmtPass {
                node_index: Default::default(),
                range: e.range,
            })]
        }
        other => vec![other],
    });
    let abstracts = abstract_classes(body);
    if !abstracts
        && !body
            .iter()
            .any(|s| matches!(s, py::Stmt::FunctionDef(f) if kind(f).is_some()))
    {
        return Ok(());
    }
    let mut out = Vec::with_capacity(body.len());
    let mut kept = Vec::with_capacity(body.len());
    for (s, origin) in std::mem::take(body)
        .into_iter()
        .zip(std::mem::take(origins))
    {
        let made = match s {
            py::Stmt::FunctionDef(mut f) => match kind(&f) {
                Some(k) => {
                    f.decorator_list.clear();
                    expand(f, k)?
                }
                None => vec![py::Stmt::FunctionDef(f)],
            },
            other => vec![other],
        };
        kept.extend(std::iter::repeat_n(origin, made.len()));
        out.extend(made);
    }
    *body = out;
    *origins = kept;
    drop_imports(body, origins);
    Ok(())
}

/// The names of the parameters of `ps`, and how a call passes them on.
fn forward(ps: &py::Parameters) -> Result<(Vec<String>, String)> {
    if ps.vararg.is_some() || ps.kwarg.is_some() {
        return Err(Error::unsupported(
            "a decorated function that takes *args or **kwargs",
            ps,
        ));
    }
    let names: Vec<String> = ps
        .posonlyargs
        .iter()
        .chain(ps.args.iter())
        .chain(ps.kwonlyargs.iter())
        .map(|p| p.parameter.name.to_string())
        .collect();
    let mut args: Vec<String> = ps
        .posonlyargs
        .iter()
        .chain(ps.args.iter())
        .map(|p| p.parameter.name.to_string())
        .collect();
    args.extend(ps.kwonlyargs.iter().map(|p| {
        let n = p.parameter.name.as_str();
        format!("{n}={n}")
    }));
    Ok((names, args.join(", ")))
}

fn parse(source: &str) -> Vec<py::Stmt> {
    ruff_python_parser::parse_module(source)
        .expect("a decorator expansion parses")
        .into_syntax()
        .body
        .into_iter()
        .collect()
}

fn expand(f: py::StmtFunctionDef, k: Kind) -> Result<Vec<py::Stmt>> {
    let name = f.name.to_string();
    let (names, args) = forward(&f.parameters)?;
    match k {
        Kind::Cache => {
            let key = match names.len() {
                0 => "()".to_string(),
                1 => names[0].clone(),
                _ => format!("({})", names.join(", ")),
            };
            let mut stmts = parse(&format!(
                "__cache_{name} = {{}}\n\
                 def {name}():\n\
                 \x20   __key = {key}\n\
                 \x20   if __key in __cache_{name}:\n\
                 \x20       return __cache_{name}[__key]\n\
                 \x20   __result = __cached_{name}({args})\n\
                 \x20   __cache_{name}[__key] = __result\n\
                 \x20   return __result\n"
            ));
            if let py::Stmt::FunctionDef(wrapper) = &mut stmts[1] {
                wrapper.parameters = f.parameters.clone();
            }
            let mut inner = f;
            inner.name = py::Identifier::new(format!("__cached_{name}"), inner.name.range);
            stmts.push(py::Stmt::FunctionDef(inner));
            Ok(stmts)
        }
        Kind::ContextManager => {
            let mut stmts = parse(&format!(
                "def {name}():\n\
                 \x20   return __CM_{name}({args})\n\
                 class __CM_{name}:\n\
                 \x20   def __init__(self):\n\
                 \x20       self.gen = __gen_{name}({args})\n\
                 \x20   def __enter__(self):\n\
                 \x20       return next(self.gen)\n\
                 \x20   def __exit__(self, kind, error, trace):\n\
                 \x20       next(self.gen, None)\n\
                 \x20       return False\n"
            ));
            if let py::Stmt::FunctionDef(wrapper) = &mut stmts[0] {
                wrapper.parameters = f.parameters.clone();
            }
            if let py::Stmt::ClassDef(class) = &mut stmts[1]
                && let Some(py::Stmt::FunctionDef(init)) = class.body.first_mut()
            {
                let mut ps = (*f.parameters).clone();
                let range = ps.range();
                ps.args.insert(
                    0,
                    py::ParameterWithDefault {
                        range,
                        node_index: Default::default(),
                        parameter: py::Parameter {
                            range,
                            node_index: Default::default(),
                            name: py::Identifier::new("self", range),
                            annotation: None,
                        },
                        default: None,
                    },
                );
                *init.parameters = ps;
            }
            let mut inner = f;
            inner.name = py::Identifier::new(format!("__gen_{name}"), inner.name.range);
            stmts.push(py::Stmt::FunctionDef(inner));
            Ok(stmts)
        }
    }
}

/// Whether `e` is `name` or `abc.name`.
fn is_abc_name(e: &py::Expr, name: &str) -> bool {
    match e {
        py::Expr::Name(n) => n.id.as_str() == name,
        py::Expr::Attribute(a) => {
            a.attr.as_str() == name
                && matches!(&*a.value, py::Expr::Name(m) if m.id.as_str() == "abc")
        }
        _ => false,
    }
}

/// What the classes seen so far need an instance to define.
#[derive(Default)]
struct Known {
    abstract_methods: Vec<String>,
    /// The class whose `__init__` an instance of this one runs.
    init_owner: Option<String>,
}

/// Make `ABC` and `@abstractmethod` plain: a class with a method left
/// abstract refuses to be instantiated itself, in the `__init__` its
/// instances run. Whether any class used them.
fn abstract_classes(body: &mut [py::Stmt]) -> bool {
    let mut known: std::collections::HashMap<String, Known> = std::collections::HashMap::new();
    let mut any = false;
    // The guards each class's `__init__` gets, by that class.
    let mut guards: Vec<(String, String, Vec<String>)> = Vec::new();
    let mut synthesized: Vec<String> = Vec::new();
    for s in body.iter_mut() {
        let py::Stmt::ClassDef(c) = s else {
            continue;
        };
        let name = c.name.to_string();
        let mut own: Vec<String> = Vec::new();
        let mut concrete: Vec<String> = Vec::new();
        for m in c.body.iter_mut() {
            let py::Stmt::FunctionDef(f) = m else {
                continue;
            };
            let before = f.decorator_list.len();
            f.decorator_list
                .retain(|d| !is_abc_name(&d.expression, "abstractmethod"));
            if f.decorator_list.len() < before {
                own.push(f.name.to_string());
                any = true;
            } else {
                concrete.push(f.name.to_string());
            }
        }
        let mut inherited: Vec<String> = Vec::new();
        let mut init_owner = None;
        if let Some(args) = &mut c.arguments {
            let mut kept = Vec::new();
            for b in std::mem::take(&mut args.args).into_vec() {
                if is_abc_name(&b, "ABC") {
                    any = true;
                    continue;
                }
                if let py::Expr::Name(n) = &b
                    && let Some(k) = known.get(n.id.as_str())
                {
                    inherited.extend(k.abstract_methods.iter().cloned());
                    init_owner = init_owner.or_else(|| k.init_owner.clone());
                }
                kept.push(b);
            }
            args.args = kept.into_boxed_slice();
            let before = args.keywords.len();
            args.keywords.retain(|k| {
                !(k.arg.as_ref().is_some_and(|a| a.as_str() == "metaclass")
                    && is_abc_name(&k.value, "ABCMeta"))
            });
            any |= args.keywords.len() < before;
        }
        if concrete.iter().any(|n| n == "__init__") {
            init_owner = Some(name.clone());
        }
        let mut methods: Vec<String> = inherited
            .into_iter()
            .filter(|m| !concrete.contains(m))
            .chain(own)
            .collect();
        methods.sort();
        methods.dedup();
        if !methods.is_empty() && c.decorator_list.is_empty() {
            let owner = match &init_owner {
                Some(o) => o.clone(),
                None => {
                    // No class of the chain has one: this one gets one.
                    synthesized.push(name.clone());
                    init_owner = Some(name.clone());
                    name.clone()
                }
            };
            guards.push((owner, name.clone(), methods.clone()));
        }
        known.insert(
            name,
            Known {
                abstract_methods: methods,
                init_owner,
            },
        );
    }
    for s in body.iter_mut() {
        let py::Stmt::ClassDef(c) = s else {
            continue;
        };
        let name = c.name.to_string();
        if synthesized.contains(&name) {
            c.body.extend(parse("def __init__(self):\n    pass\n"));
        }
        for (owner, class, methods) in guards.iter().rev() {
            if *owner == name {
                guard(c, class, methods);
            }
        }
    }
    any
}

/// Make `__init__` of `c` refuse an instance of exactly `class`.
fn guard(c: &mut py::StmtClassDef, class: &str, methods: &[String]) {
    let named: Vec<String> = methods.iter().map(|m| format!("'{m}'")).collect();
    let message = format!(
        "Can't instantiate abstract class {class} without an implementation for abstract method{} {}",
        if methods.len() == 1 { "" } else { "s" },
        named.join(", ")
    );
    let check = parse(&format!(
        "if type(self) is {class}:\n    raise TypeError({message:?})\n"
    ));
    for m in c.body.iter_mut() {
        if let py::Stmt::FunctionDef(f) = m
            && f.name.as_str() == "__init__"
        {
            let old = std::mem::take(&mut f.body);
            f.body = check.into_iter().chain(old).collect();
            return;
        }
    }
}

/// Remove the imports of the decorators this pass made unnecessary.
pub(crate) fn drop_imports(body: &mut Vec<py::Stmt>, origins: &mut Vec<Option<String>>) {
    let mut kept = Vec::with_capacity(body.len());
    let mut kept_origins = Vec::with_capacity(body.len());
    let mut dropped: Vec<(String, bool)> = Vec::new();
    for (mut s, origin) in std::mem::take(body)
        .into_iter()
        .zip(std::mem::take(origins))
    {
        if let py::Stmt::ImportFrom(i) = &mut s
            && i.level == 0
            && let Some(m) = i.module.as_ref().map(|m| m.as_str().to_string())
        {
            let names: Vec<py::Alias> = i
                .names
                .iter()
                .filter(|a| !is_imported(&m, a.name.as_str()))
                .cloned()
                .collect();
            if names.is_empty() {
                continue;
            }
            i.names = names;
        }
        // The import of a module the program lacks is the binding
        // `name = __import__("contextlib")`, or one of its members.
        if let py::Stmt::Assign(a) = &s
            && let Some((module, member)) = foreign_import(&a.value)
            && IMPORTED.iter().any(|(m, _)| *m == module)
            && member.is_none_or(|m| is_imported(module, m))
        {
            dropped.push((module.to_string(), member.is_none()));
            continue;
        }
        kept.push(s);
        kept_origins.push(origin);
    }
    // `import abc` and `import contextlib` are not needed once only
    // decorators and bases used them.
    for (module, whole) in &dropped {
        if *whole && uses_name(&kept, module) {
            // Put the binding back: the program reads the module.
            let at = 0;
            let binding = parse(&format!("{module} = __import__({module:?})\n"));
            for (n, b) in binding.into_iter().enumerate() {
                kept.insert(at + n, b);
                kept_origins.insert(at + n, None);
            }
        }
    }
    *body = kept;
    *origins = kept_origins;
}

/// `__import__("m")` or `__import__("m").member`, as the module and member.
fn foreign_import(e: &py::Expr) -> Option<(&str, Option<&str>)> {
    match e {
        py::Expr::Attribute(a) => {
            let (module, _) = foreign_import(&a.value)?;
            Some((module, Some(a.attr.as_str())))
        }
        py::Expr::Call(c) if matches!(&*c.func, py::Expr::Name(n) if n.id.as_str() == "__import__") => {
            match c.arguments.args.first() {
                Some(py::Expr::StringLiteral(l)) => Some((l.value.to_str(), None)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether any expression of `body` is the name `id`.
fn uses_name(body: &[py::Stmt], id: &str) -> bool {
    use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
    struct V<'n>(&'n str, bool);
    impl<'a> Visitor<'a> for V<'_> {
        fn visit_stmt(&mut self, s: &'a py::Stmt) {
            walk_stmt(self, s);
        }
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if matches!(e, py::Expr::Name(n) if n.id.as_str() == self.0) {
                self.1 = true;
            }
            walk_expr(self, e);
        }
    }
    let mut v = V(id, false);
    for s in body {
        v.visit_stmt(s);
    }
    v.1
}
