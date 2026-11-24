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
    if !body
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

/// Remove the imports of the decorators this pass made unnecessary.
fn drop_imports(body: &mut Vec<py::Stmt>, origins: &mut Vec<Option<String>>) {
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
            && matches!(m.as_str(), "functools" | "contextlib")
        {
            let names: Vec<py::Alias> = i
                .names
                .iter()
                .filter(|a| !matches!(a.name.as_str(), "lru_cache" | "cache" | "contextmanager"))
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
            && matches!(module, "functools" | "contextlib")
            && member.is_none_or(|m| matches!(m, "lru_cache" | "cache" | "contextmanager"))
        {
            dropped.push((module.to_string(), member.is_none()));
            continue;
        }
        kept.push(s);
        kept_origins.push(origin);
    }
    // `import contextlib` is not needed once only decorators used it.
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
