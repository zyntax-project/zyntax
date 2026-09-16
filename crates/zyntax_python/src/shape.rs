//! Shapes retained across Python containers for unpacking in closed functions.
//! Runtime values keep their ordinary list, tuple, and dict representation.

use std::collections::HashMap;

use ruff_python_ast as py;

use crate::types::{Elem, Item, Module, Ty};

#[derive(Clone, Debug, PartialEq, Eq)]
enum Shape {
    Bottom,
    Dynamic,
    Scalar(Ty),
    Tuple(Vec<Shape>),
    List(Box<Shape>),
    Dict(Box<Shape>),
}

impl Shape {
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, x) | (x, Self::Bottom) => x.clone(),
            (Self::Scalar(a), Self::Scalar(b)) if a == b => self.clone(),
            (Self::Tuple(a), Self::Tuple(b)) if a.len() == b.len() => {
                Self::Tuple(a.iter().zip(b).map(|(x, y)| x.join(y)).collect())
            }
            (Self::List(a), Self::List(b)) => Self::List(Box::new(a.join(b))),
            (Self::Dict(a), Self::Dict(b)) => Self::Dict(Box::new(a.join(b))),
            _ => Self::Dynamic,
        }
    }

    fn ty(&self) -> Ty {
        match self {
            Self::Scalar(t) => *t,
            Self::Tuple(_) => Ty::Tuple,
            Self::List(inner) => Ty::List(Elem::of(inner.ty())),
            Self::Dict(_) => Ty::Dict,
            Self::Bottom => Ty::Unknown,
            Self::Dynamic => Ty::Object,
        }
    }

    fn element(&self) -> Self {
        match self {
            Self::List(inner) => (**inner).clone(),
            Self::Tuple(fields) => fields
                .iter()
                .fold(Self::Bottom, |acc, field| acc.join(field)),
            _ => Self::Dynamic,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Profile {
    params: Vec<Shape>,
    ret: Shape,
    post_params: Vec<Shape>,
}

struct Analyzer<'a> {
    profiles: &'a HashMap<String, Profile>,
    globals: &'a HashMap<String, Shape>,
    passed: &'a mut HashMap<String, Vec<Shape>>,
    env: HashMap<String, Shape>,
    ret: Shape,
}

impl Analyzer<'_> {
    fn write_item(&mut self, target: &py::Expr, value: Shape) {
        let py::Expr::Subscript(sub) = target else {
            return;
        };
        let py::Expr::Name(n) = &*sub.value else {
            return;
        };
        let name = n.id.as_str();
        let updated = match self.name(name) {
            Shape::List(inner) => Shape::List(Box::new(inner.join(&value))),
            Shape::Dict(inner) => Shape::Dict(Box::new(inner.join(&value))),
            _ => Shape::Dynamic,
        };
        self.env.insert(name.to_string(), updated);
    }

    fn name(&self, name: &str) -> Shape {
        self.env
            .get(name)
            .or_else(|| self.globals.get(name))
            .cloned()
            .unwrap_or(Shape::Dynamic)
    }

    fn expr(&mut self, expr: &py::Expr) -> Shape {
        match expr {
            py::Expr::NumberLiteral(n) => match &n.value {
                py::Number::Int(_) => Shape::Scalar(Ty::Int),
                py::Number::Float(_) => Shape::Scalar(Ty::Float),
                _ => Shape::Dynamic,
            },
            py::Expr::StringLiteral(_) => Shape::Scalar(Ty::Str),
            py::Expr::BooleanLiteral(_) => Shape::Scalar(Ty::Bool),
            py::Expr::Name(n) => self.name(n.id.as_str()),
            py::Expr::Tuple(t) => Shape::Tuple(t.elts.iter().map(|e| self.expr(e)).collect()),
            py::Expr::List(l) => Shape::List(Box::new(
                l.elts
                    .iter()
                    .map(|e| self.expr(e))
                    .fold(Shape::Bottom, |a, b| a.join(&b)),
            )),
            py::Expr::Dict(d) => {
                Shape::Dict(Box::new(if d.items.iter().any(|item| item.key.is_none()) {
                    Shape::Dynamic
                } else {
                    d.items
                        .iter()
                        .map(|item| self.expr(&item.value))
                        .fold(Shape::Bottom, |a, b| a.join(&b))
                }))
            }
            py::Expr::Subscript(s) => {
                let source = self.expr(&s.value);
                if matches!(&*s.slice, py::Expr::Slice(_)) {
                    match source {
                        Shape::List(_) => source,
                        _ => Shape::Dynamic,
                    }
                } else {
                    match source {
                        Shape::Dict(inner) | Shape::List(inner) => *inner,
                        Shape::Tuple(fields) => {
                            let py::Expr::NumberLiteral(n) = &*s.slice else {
                                return Shape::Dynamic;
                            };
                            let py::Number::Int(i) = &n.value else {
                                return Shape::Dynamic;
                            };
                            i.as_i64()
                                .and_then(|i| fields.get(i as usize).cloned())
                                .unwrap_or(Shape::Dynamic)
                        }
                        _ => Shape::Dynamic,
                    }
                }
            }
            py::Expr::UnaryOp(u) => match u.op {
                py::UnaryOp::UAdd | py::UnaryOp::USub => self.expr(&u.operand),
                py::UnaryOp::Not => Shape::Scalar(Ty::Bool),
                _ => Shape::Dynamic,
            },
            py::Expr::BinOp(b) => {
                let left = self.expr(&b.left).ty();
                let right = self.expr(&b.right).ty();
                if !matches!(left, Ty::Int | Ty::Float | Ty::Bool)
                    || !matches!(right, Ty::Int | Ty::Float | Ty::Bool)
                {
                    Shape::Dynamic
                } else {
                    match b.op {
                        py::Operator::Add
                        | py::Operator::Sub
                        | py::Operator::Mult
                        | py::Operator::Div
                        | py::Operator::FloorDiv
                        | py::Operator::Mod => Shape::Scalar(
                            if b.op == py::Operator::Div || left == Ty::Float || right == Ty::Float
                            {
                                Ty::Float
                            } else {
                                Ty::Int
                            },
                        ),
                        py::Operator::Pow if left == Ty::Float || right == Ty::Float => {
                            Shape::Scalar(Ty::Float)
                        }
                        _ => Shape::Dynamic,
                    }
                }
            }
            py::Expr::Call(c) => {
                let args: Vec<_> = c.arguments.args.iter().map(|e| self.expr(e)).collect();
                match &*c.func {
                    py::Expr::Name(n) => match n.id.as_str() {
                        "list" => match args.first() {
                            Some(Shape::List(_)) => args[0].clone(),
                            Some(x) => Shape::List(Box::new(x.element())),
                            None => Shape::List(Box::new(Shape::Bottom)),
                        },
                        "range" | "xrange" => Shape::List(Box::new(Shape::Scalar(Ty::Int))),
                        name => {
                            if let Some(profile) = self.profiles.get(name) {
                                let slots = self
                                    .passed
                                    .entry(name.to_string())
                                    .or_insert_with(|| vec![Shape::Bottom; profile.params.len()]);
                                for (slot, arg) in slots.iter_mut().zip(&args) {
                                    *slot = slot.join(arg);
                                }
                                if !c.arguments.keywords.is_empty() {
                                    for slot in slots.iter_mut() {
                                        *slot = Shape::Dynamic;
                                    }
                                }
                                for (arg, post) in c.arguments.args.iter().zip(&profile.post_params)
                                {
                                    if let py::Expr::Name(n) = arg {
                                        let old = self.name(n.id.as_str());
                                        if *post != Shape::Bottom && *post != old {
                                            self.env.insert(n.id.to_string(), old.join(post));
                                        }
                                    }
                                }
                                profile.ret.clone()
                            } else {
                                if !matches!(
                                    name,
                                    "len"
                                        | "print"
                                        | "str"
                                        | "repr"
                                        | "sum"
                                        | "min"
                                        | "max"
                                        | "sorted"
                                        | "reversed"
                                        | "tuple"
                                        | "int"
                                        | "float"
                                ) {
                                    for arg in &c.arguments.args {
                                        if let py::Expr::Name(n) = arg {
                                            if matches!(
                                                self.name(n.id.as_str()),
                                                Shape::List(_) | Shape::Dict(_)
                                            ) {
                                                self.env.insert(n.id.to_string(), Shape::Dynamic);
                                            }
                                        }
                                    }
                                }
                                Shape::Dynamic
                            }
                        }
                    },
                    py::Expr::Attribute(a) => {
                        let recv = self.expr(&a.value);
                        match (recv, a.attr.as_str()) {
                            (Shape::Dict(inner), "values") => Shape::List(inner),
                            (Shape::List(inner), "copy") => Shape::List(inner),
                            (_, "len" | "count" | "index" | "copy") => Shape::Dynamic,
                            (_, _) => {
                                if let py::Expr::Name(n) = &*a.value {
                                    if matches!(
                                        self.name(n.id.as_str()),
                                        Shape::List(_) | Shape::Dict(_)
                                    ) {
                                        self.env.insert(n.id.to_string(), Shape::Dynamic);
                                    }
                                }
                                Shape::Dynamic
                            }
                        }
                    }
                    _ => Shape::Dynamic,
                }
            }
            _ => Shape::Dynamic,
        }
    }

    fn bind(&mut self, target: &py::Expr, shape: Shape) {
        match target {
            py::Expr::Name(n) => {
                let name = n.id.to_string();
                let old = self.env.get(&name).cloned().unwrap_or(Shape::Bottom);
                self.env.insert(name, old.join(&shape));
            }
            py::Expr::Tuple(t) => self.bind_fields(&t.elts, shape),
            py::Expr::List(l) => self.bind_fields(&l.elts, shape),
            _ => {}
        }
    }

    fn bind_fields(&mut self, targets: &[py::Expr], shape: Shape) {
        match shape {
            Shape::Tuple(fields) if fields.len() == targets.len() => {
                for (target, shape) in targets.iter().zip(fields) {
                    self.bind(target, shape);
                }
            }
            Shape::List(inner) => {
                for target in targets {
                    self.bind(target, (*inner).clone());
                }
            }
            _ => {
                for target in targets {
                    self.bind(target, Shape::Dynamic);
                }
            }
        }
    }

    fn block(&mut self, body: &[py::Stmt]) {
        for stmt in body {
            match stmt {
                py::Stmt::Assign(a) => {
                    let mut shape = self.expr(&a.value);
                    if let py::Expr::Name(source) = &*a.value {
                        if a.targets
                            .iter()
                            .any(|target| matches!(target, py::Expr::Name(n) if n.id != source.id))
                            && matches!(shape, Shape::List(_) | Shape::Dict(_))
                        {
                            self.env.insert(source.id.to_string(), Shape::Dynamic);
                            shape = Shape::Dynamic;
                        }
                    }
                    for target in &a.targets {
                        self.bind(target, shape.clone());
                        self.write_item(target, shape.clone());
                    }
                }
                py::Stmt::AnnAssign(a) => {
                    if let Some(value) = &a.value {
                        let shape = self.expr(value);
                        self.bind(&a.target, shape.clone());
                        self.write_item(&a.target, shape);
                    }
                }
                py::Stmt::AugAssign(a) => {
                    let left = self.expr(&a.target);
                    let right = self.expr(&a.value);
                    let value = match (left.ty(), right.ty()) {
                        (Ty::Float, Ty::Float | Ty::Int | Ty::Bool)
                        | (Ty::Int | Ty::Bool, Ty::Float) => Shape::Scalar(Ty::Float),
                        (Ty::Int | Ty::Bool, Ty::Int | Ty::Bool) => Shape::Scalar(Ty::Int),
                        _ => Shape::Dynamic,
                    };
                    self.bind(&a.target, value.clone());
                    self.write_item(&a.target, value);
                }
                py::Stmt::For(f) => {
                    let before = self.env.clone();
                    let element = self.expr(&f.iter).element();
                    self.bind(&f.target, element);
                    self.block(&f.body);
                    for (name, shape) in before {
                        let current = self.env.get(&name).cloned().unwrap_or(Shape::Bottom);
                        self.env.insert(name, shape.join(&current));
                    }
                    self.block(&f.orelse);
                }
                py::Stmt::If(i) => {
                    let before = self.env.clone();
                    self.block(&i.body);
                    let then_env = self.env.clone();
                    self.env = before;
                    for clause in &i.elif_else_clauses {
                        self.block(&clause.body);
                    }
                    for (name, shape) in then_env {
                        let current = self.env.get(&name).cloned().unwrap_or(Shape::Bottom);
                        self.env.insert(name, shape.join(&current));
                    }
                }
                py::Stmt::Expr(e) => {
                    if let py::Expr::Call(c) = &*e.value {
                        if let py::Expr::Attribute(a) = &*c.func {
                            if a.attr.as_str() == "append" {
                                if let py::Expr::Name(n) = &*a.value {
                                    if let Some(arg) = c.arguments.args.first() {
                                        let added = self.expr(arg);
                                        let old = self.name(n.id.as_str());
                                        if let Shape::List(inner) = old {
                                            self.env.insert(
                                                n.id.to_string(),
                                                Shape::List(Box::new(inner.join(&added))),
                                            );
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    self.expr(&e.value);
                }
                py::Stmt::Return(r) => {
                    let shape = r
                        .value
                        .as_ref()
                        .map(|e| self.expr(e))
                        .unwrap_or(Shape::Dynamic);
                    self.ret = self.ret.join(&shape);
                }
                _ => {}
            }
        }
    }
}

/// Shapes proven from closed calls and container construction. These
/// refine locals, not function ABIs.
pub(crate) fn infer(
    module: &Module,
    items: &[Item<'_>],
    entry: &[py::Stmt],
) -> HashMap<String, HashMap<String, Ty>> {
    let mut profiles: HashMap<String, Profile> = items
        .iter()
        .map(|item| {
            let sig = &module.funcs[&item.name];
            (
                item.name.clone(),
                Profile {
                    params: vec![Shape::Bottom; sig.params.len()],
                    ret: Shape::Bottom,
                    post_params: vec![Shape::Bottom; sig.params.len()],
                },
            )
        })
        .collect();
    let mut globals = HashMap::new();
    let mut locals = HashMap::new();
    let mut converged = false;
    for _ in 0..16 {
        let mut passed = HashMap::new();
        let mut next_profiles = profiles.clone();
        let mut next_locals = HashMap::new();
        for item in items {
            let sig = &module.funcs[&item.name];
            let profile = &profiles[&item.name];
            let env = sig
                .params
                .iter()
                .zip(&profile.params)
                .map(|((name, _), shape)| (name.clone(), shape.clone()))
                .collect();
            let mut analyzer = Analyzer {
                profiles: &profiles,
                globals: &globals,
                passed: &mut passed,
                env,
                ret: Shape::Bottom,
            };
            analyzer.block(&item.def.body);
            next_profiles.get_mut(&item.name).unwrap().ret = analyzer.ret.clone();
            next_profiles.get_mut(&item.name).unwrap().post_params = sig
                .params
                .iter()
                .map(|(name, _)| analyzer.env.get(name).cloned().unwrap_or(Shape::Bottom))
                .collect();
            next_locals.insert(item.name.clone(), analyzer.env);
        }
        let mut entry_analyzer = Analyzer {
            profiles: &profiles,
            globals: &globals,
            passed: &mut passed,
            env: HashMap::new(),
            ret: Shape::Bottom,
        };
        entry_analyzer.block(entry);
        let next_globals = entry_analyzer.env;
        for item in items {
            let sig = &module.funcs[&item.name];
            let profile = next_profiles.get_mut(&item.name).unwrap();
            for (i, (_, ty)) in sig.params.iter().enumerate() {
                let mut shape = if module.closed.contains(&item.name) {
                    Shape::Bottom
                } else {
                    Shape::Dynamic
                };
                if let Some(default) = sig.defaults.get(i).and_then(|d| d.as_ref()) {
                    let mut analyzer = Analyzer {
                        profiles: &profiles,
                        globals: &next_globals,
                        passed: &mut passed,
                        env: HashMap::new(),
                        ret: Shape::Bottom,
                    };
                    shape = shape.join(&analyzer.expr(default));
                }
                if let Some(given) = passed.get(&item.name).and_then(|v| v.get(i)) {
                    shape = shape.join(given);
                }
                if shape == Shape::Bottom && *ty != Ty::Unknown {
                    shape = Shape::Scalar(*ty);
                }
                profile.params[i] = shape;
            }
        }
        if next_profiles == profiles && next_globals == globals {
            locals = next_locals;
            converged = true;
            break;
        }
        profiles = next_profiles;
        globals = next_globals;
        locals = next_locals;
    }
    if !converged {
        return HashMap::new();
    }
    let refined: HashMap<_, _> = locals
        .into_iter()
        .map(|(name, shapes)| {
            let vars: HashMap<String, Ty> = shapes
                .into_iter()
                .filter_map(|(name, shape)| {
                    let ty = shape.ty();
                    matches!(ty, Ty::Float | Ty::List(Elem::Float)).then_some((name, ty))
                })
                .collect();
            (name, vars)
        })
        .collect();
    if std::env::var_os("ZYNTAX_TRACE_SHAPES").is_some() {
        let mut entries: Vec<_> = refined.iter().collect();
        entries.sort_by_key(|(name, _)| *name);
        for (name, vars) in entries {
            if !vars.is_empty() {
                eprintln!("[shapes] {name}: {vars:?}");
            }
        }
    }
    refined
}
