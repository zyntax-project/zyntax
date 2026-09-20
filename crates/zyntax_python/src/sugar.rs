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

use ruff_python_ast as py;
use ruff_python_ast::visitor::transformer::{Transformer, walk_expr, walk_stmt};
use ruff_text_size::Ranged;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cell::RefCell;

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
    for (i, s) in body.iter_mut().enumerate() {
        let py::Stmt::ClassDef(c) = s else {
            continue;
        };
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
        let mut kept = Vec::new();
        for m in std::mem::take(&mut c.body) {
            match m {
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
                other => kept.push(other),
            }
        }
        c.body = kept.into_iter().collect();
    }
    let rewriter = Rewriter {
        rootless,
        aliases,
        statics,
        method_of: RefCell::new(None),
    };
    for s in body.iter_mut() {
        rewriter.visit_stmt(s);
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

fn is_static_or_class_method(f: &py::StmtFunctionDef) -> bool {
    f.decorator_list.len() == 1
        && (decorated_with(f, "classmethod") || decorated_with(f, "staticmethod"))
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

struct Rewriter {
    rootless: HashSet<String>,
    aliases: HashMap<String, String>,
    statics: HashMap<String, HashSet<String>>,
    /// The class and `self` name of the method being walked.
    method_of: RefCell<Option<(String, String)>>,
}

impl Rewriter {
    fn is_static(&self, class: &str, method: &str) -> bool {
        self.statics
            .get(class)
            .is_some_and(|methods| methods.contains(method))
    }
}

impl Transformer for Rewriter {
    fn visit_stmt(&self, s: &mut py::Stmt) {
        match s {
            py::Stmt::ClassDef(c) => {
                let class = c.name.to_string();
                for m in c.body.iter_mut() {
                    if let py::Stmt::FunctionDef(f) = m {
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
            _ => walk_expr(self, e),
        }
    }
}

fn is_name(e: &py::Expr, name: &str) -> bool {
    matches!(e, py::Expr::Name(n) if n.id.as_str() == name)
}

fn is_super(e: &py::Expr) -> bool {
    matches!(e, py::Expr::Call(c) if is_name(&c.func, "super"))
}
