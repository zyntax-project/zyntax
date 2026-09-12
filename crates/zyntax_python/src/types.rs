//! The static types a Python expression can have here, and how they are
//! inferred.
//!
//! Python has no static types; this frontend gives every expression one
//! anyway, because the IR needs it and because the fast path depends on
//! it. The lattice is flat: the primitives the IR can hold unboxed, and
//! `Object` for a value the runtime owns. A name assigned two different
//! primitives is `Object`, not their least upper bound, since `x = 1`
//! followed by `x = 2.5` must still print `1` between the two.
//!
//! Inference runs to a fixed point twice: over the module's function
//! signatures, so an unannotated return type is the join of what the
//! body returns, and inside each function over its locals, so a name
//! gets the join of everything assigned to it anywhere in the body.

use ruff_python_ast as py;
use std::collections::HashMap;
use zyntax_typed_ast::typed_ast::TypedFunction;

/// A static type. `Unknown` is the bottom of the join and never
/// survives inference; a name nothing assigns to is a runtime error in
/// Python and an `Object` here.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum Ty {
    Int,
    Float,
    Bool,
    Str,
    None,
    /// A list whose elements are all of one kind.
    List(Elem),
    /// A tuple: an immutable list of dynamic values.
    Tuple,
    /// A dynamic value: a boxed `Any`.
    Object,
    #[default]
    Unknown,
}

/// The element kinds a list is instantiated for. Anything else in a
/// list is a dynamic value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Elem {
    Int,
    Float,
    Str,
    Object,
}

impl Elem {
    /// The element kind a value of `ty` is stored as.
    pub(crate) fn of(ty: Ty) -> Elem {
        match ty {
            Ty::Int => Elem::Int,
            Ty::Float => Elem::Float,
            Ty::Str => Elem::Str,
            _ => Elem::Object,
        }
    }

    pub(crate) fn ty(self) -> Ty {
        match self {
            Elem::Int => Ty::Int,
            Elem::Float => Ty::Float,
            Elem::Str => Ty::Str,
            Elem::Object => Ty::Object,
        }
    }

    /// The suffix of the library functions for this kind.
    pub(crate) fn suffix(self) -> &'static str {
        match self {
            Elem::Int => "i64",
            Elem::Float => "f64",
            Elem::Str => "str",
            Elem::Object => "any",
        }
    }
}

impl Ty {
    /// The join of two assignments to one name.
    pub(crate) fn join(self, other: Ty) -> Ty {
        match (self, other) {
            (Ty::Unknown, t) | (t, Ty::Unknown) => t,
            (a, b) if a == b => a,
            _ => Ty::Object,
        }
    }

    pub(crate) fn is_numeric(self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Bool)
    }

    /// The element type of a sequence, when it is one.
    pub(crate) fn element(self) -> Option<Ty> {
        match self {
            Ty::List(e) => Some(e.ty()),
            Ty::Tuple => Some(Ty::Object),
            Ty::Str => Some(Ty::Str),
            _ => None,
        }
    }

    /// The type an arithmetic result has when both operands are known
    /// primitives. Bools count as ints, as in Python.
    fn arith(self, other: Ty) -> Ty {
        match (self, other) {
            (Ty::Float, x) | (x, Ty::Float) if x.is_numeric() => Ty::Float,
            (a, b) if a.is_numeric() && b.is_numeric() => Ty::Int,
            (Ty::Unknown, _) | (_, Ty::Unknown) => Ty::Unknown,
            _ => Ty::Object,
        }
    }
}

/// A function's signature as far as inference knows it.
#[derive(Clone, Debug)]
pub(crate) struct Sig {
    pub(crate) params: Vec<(String, Ty)>,
    pub(crate) ret: Ty,
    /// Each parameter's default, evaluated at the call site that leaves
    /// the parameter out.
    pub(crate) defaults: Vec<Option<py::Expr>>,
}

/// What the module declares: every `def` by name, and the library's
/// `List<T>` so list types can be spelled the way the library spells
/// them.
#[derive(Default, Debug)]
pub(crate) struct Module {
    pub(crate) funcs: HashMap<String, Sig>,
    /// Module-level variables a function reads or declares `global`,
    /// with the join of everything assigned to them anywhere.
    pub(crate) globals: HashMap<String, Ty>,
    /// Functions the lowering produced from nested defs and lambdas,
    /// and the value adapters of module functions, to be declared with
    /// the program.
    pub(crate) lifted: std::cell::RefCell<Vec<TypedFunction>>,
    /// Module functions used as values, which need an adapter.
    pub(crate) adapters: std::cell::RefCell<std::collections::BTreeSet<String>>,
    /// A counter for names no Python program can spell.
    pub(crate) counter: std::cell::Cell<usize>,
    pub(crate) list_type: Option<zyntax_typed_ast::TypeId>,
}

/// One function's inferred locals.
#[derive(Default, Debug, Clone)]
pub(crate) struct Locals {
    pub(crate) vars: HashMap<String, Ty>,
    pub(crate) ret: Ty,
    /// Names this body declares `global`, and what it assigns to them.
    pub(crate) global_writes: HashMap<String, Ty>,
    /// Names this body declares `nonlocal`, and what it assigns to them.
    pub(crate) nonlocal_writes: HashMap<String, Ty>,
}

pub(crate) fn annotation(e: &py::Expr) -> Ty {
    match e {
        py::Expr::Name(n) => match n.id.as_str() {
            "int" => Ty::Int,
            "float" => Ty::Float,
            "bool" => Ty::Bool,
            "str" => Ty::Str,
            "None" => Ty::None,
            _ => Ty::Object,
        },
        py::Expr::NoneLiteral(_) => Ty::None,
        _ => Ty::Object,
    }
}

/// Signature from the annotations alone; an unannotated return is
/// `Unknown` until the body says.
pub(crate) fn declared_sig(f: &py::StmtFunctionDef) -> Sig {
    let params = f
        .parameters
        .iter_non_variadic_params()
        .map(|p| {
            let ty = p
                .parameter
                .annotation
                .as_deref()
                .map(annotation)
                .unwrap_or(Ty::Object);
            (p.parameter.name.to_string(), ty)
        })
        .collect();
    let defaults = f
        .parameters
        .iter_non_variadic_params()
        .map(|p| p.default.as_deref().cloned())
        .collect();
    Sig {
        params,
        ret: f.returns.as_deref().map(annotation).unwrap_or(Ty::Unknown),
        defaults,
    }
}

/// Infer the module's signatures to a fixed point.
pub(crate) fn infer_module(known: &Module, defs: &[&py::StmtFunctionDef]) -> HashMap<String, Sig> {
    let mut module = Module {
        funcs: HashMap::new(),
        globals: known.globals.clone(),
        list_type: known.list_type,
        ..Default::default()
    };
    for f in defs {
        module.funcs.insert(f.name.to_string(), declared_sig(f));
    }
    for _ in 0..8 {
        let mut changed = false;
        for f in defs {
            if f.returns.is_some() {
                continue;
            }
            let sig = module.funcs[f.name.as_str()].clone();
            let locals = infer_locals(&module, &sig, &f.body);
            let ret = if locals.ret == Ty::Unknown {
                Ty::None
            } else {
                locals.ret
            };
            if ret != sig.ret {
                module.funcs.get_mut(f.name.as_str()).unwrap().ret = ret;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Whatever recursion left undecided is dynamic.
    for sig in module.funcs.values_mut() {
        if sig.ret == Ty::Unknown {
            sig.ret = Ty::Object;
        }
    }
    module.funcs
}

/// Infer one body's locals: parameters as declared, every other name
/// the join of what is assigned to it, iterated until stable.
pub(crate) fn infer_locals(module: &Module, sig: &Sig, body: &[py::Stmt]) -> Locals {
    infer_locals_seeded(module, sig, body, &HashMap::new())
}

/// [`infer_locals`] with the variables captured from an enclosing scope
/// already typed. A captured variable a nested body assigns under
/// `nonlocal` takes the join of both scopes' assignments.
pub(crate) fn infer_locals_seeded(
    module: &Module,
    sig: &Sig,
    body: &[py::Stmt],
    seeds: &HashMap<String, Ty>,
) -> Locals {
    let mut locals = Locals::default();
    for (name, ty) in seeds {
        locals.vars.insert(name.clone(), *ty);
    }
    for (name, ty) in &sig.params {
        locals.vars.insert(name.clone(), *ty);
    }
    let scope = crate::scope::Scope::of_body(Vec::new(), body);
    for name in &scope.globals {
        locals.global_writes.insert(name.clone(), Ty::Unknown);
    }
    for name in &scope.nonlocals {
        locals.nonlocal_writes.insert(name.clone(), Ty::Unknown);
        locals.vars.remove(name);
    }
    for _ in 0..8 {
        let before = locals.clone();
        let mut walker = Walker {
            module,
            locals: &mut locals,
            params: &sig.params,
            seeds,
        };
        walker.stmts(body);
        // What nested bodies assign to this body's variables.
        for (_, child) in &scope.children {
            if child.nonlocals.is_empty() {
                continue;
            }
            for (name, ty) in child_nonlocal_writes(module, body, &locals.vars) {
                if let Some(own) = locals.vars.get_mut(&name) {
                    *own = own.join(ty);
                }
            }
        }
        if locals.vars == before.vars
            && locals.ret == before.ret
            && locals.global_writes == before.global_writes
            && locals.nonlocal_writes == before.nonlocal_writes
        {
            break;
        }
    }
    for ty in locals.vars.values_mut() {
        if *ty == Ty::Unknown {
            *ty = Ty::Object;
        }
    }
    locals
}

/// The `nonlocal` assignments of the defs directly inside `body`, typed
/// with `vars` as their enclosing scope.
fn child_nonlocal_writes(
    module: &Module,
    body: &[py::Stmt],
    vars: &HashMap<String, Ty>,
) -> Vec<(String, Ty)> {
    let mut out = Vec::new();
    for s in body {
        let py::Stmt::FunctionDef(f) = s else {
            continue;
        };
        let sig = declared_sig(f);
        let child = infer_locals_seeded(module, &sig, &f.body, vars);
        out.extend(child.nonlocal_writes);
    }
    out
}

struct Walker<'a> {
    module: &'a Module,
    locals: &'a mut Locals,
    params: &'a [(String, Ty)],
    seeds: &'a HashMap<String, Ty>,
}

impl Walker<'_> {
    fn assign(&mut self, name: &str, ty: Ty) {
        // A declared global or nonlocal is another scope's variable.
        if let Some(written) = self.locals.global_writes.get_mut(name) {
            *written = written.join(ty);
            return;
        }
        if let Some(written) = self.locals.nonlocal_writes.get_mut(name) {
            *written = written.join(ty);
            return;
        }
        // A captured variable is typed by the scope that owns it.
        if self.seeds.contains_key(name) {
            return;
        }

        // A parameter keeps its declared type unless the body assigns it
        // another, in which case it lives as an object from the start.
        if let Some((_, declared)) = self.params.iter().find(|(n, _)| n == name) {
            if *declared != Ty::Object && ty != *declared && ty != Ty::Unknown {
                self.locals.vars.insert(name.to_string(), Ty::Object);
            }
            return;
        }
        let joined = self
            .locals
            .vars
            .get(name)
            .copied()
            .unwrap_or(Ty::Unknown)
            .join(ty);
        self.locals.vars.insert(name.to_string(), joined);
    }

    fn target(&mut self, target: &py::Expr, ty: Ty) {
        match target {
            py::Expr::Name(n) => self.assign(n.id.as_str(), ty),
            // Unpacking gives every name an element, whose type only the
            // runtime knows.
            py::Expr::Tuple(t) => {
                for e in &t.elts {
                    self.target(e, Ty::Object);
                }
            }
            py::Expr::List(l) => {
                for e in &l.elts {
                    self.target(e, Ty::Object);
                }
            }
            _ => {}
        }
    }

    fn stmts(&mut self, stmts: &[py::Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &py::Stmt) {
        match s {
            py::Stmt::Assign(a) => {
                let ty = self.expr(&a.value);
                for t in &a.targets {
                    self.target(t, ty);
                }
            }
            py::Stmt::AnnAssign(a) => {
                if let Some(v) = &a.value {
                    let ty = self.expr(v);
                    self.target(&a.target, ty);
                }
            }
            py::Stmt::AugAssign(a) => {
                let lhs = self.expr(&a.target);
                let rhs = self.expr(&a.value);
                let ty = binop(a.op, lhs, rhs, &a.value);
                self.target(&a.target, ty);
            }
            py::Stmt::Return(r) => {
                let ty = match &r.value {
                    Some(v) => self.expr(v),
                    None => Ty::None,
                };
                self.locals.ret = self.locals.ret.join(ty);
            }
            py::Stmt::For(f) => {
                let iter = self.expr(&f.iter);
                let item = match (&*f.iter, iter) {
                    (py::Expr::Call(c), _) if is_name(&c.func, "range") => Ty::Int,
                    (_, t) => t.element().unwrap_or(Ty::Object),
                };
                self.target(&f.target, item);
                self.stmts(&f.body);
                self.stmts(&f.orelse);
            }
            py::Stmt::While(w) => {
                self.stmts(&w.body);
                self.stmts(&w.orelse);
            }
            py::Stmt::If(i) => {
                self.stmts(&i.body);
                for clause in &i.elif_else_clauses {
                    self.stmts(&clause.body);
                }
            }
            py::Stmt::Try(t) => {
                self.stmts(&t.body);
                for h in &t.handlers {
                    let py::ExceptHandler::ExceptHandler(h) = h;
                    if let Some(name) = &h.name {
                        self.assign(name.as_str(), Ty::Object);
                    }
                    self.stmts(&h.body);
                }
                self.stmts(&t.orelse);
                self.stmts(&t.finalbody);
            }
            py::Stmt::With(w) => self.stmts(&w.body),
            py::Stmt::FunctionDef(f) => self.assign(f.name.as_str(), Ty::Object),
            py::Stmt::ClassDef(c) => self.assign(c.name.as_str(), Ty::Object),
            _ => {}
        }
    }

    fn expr(&self, e: &py::Expr) -> Ty {
        Typer {
            module: self.module,
            vars: &self.locals.vars,
            outer: self.seeds,
        }
        .expr(e)
    }
}

/// Expression typing against a fixed environment. The lowering uses
/// the same rules, so what it emits agrees with what inference
/// assumed.
pub(crate) struct Typer<'a> {
    pub(crate) module: &'a Module,
    pub(crate) vars: &'a HashMap<String, Ty>,
    /// Variables of the enclosing function this body captured.
    pub(crate) outer: &'a HashMap<String, Ty>,
}

/// Give the names in an assignment target a type, in a scratch
/// environment.
pub(crate) fn bind_target(vars: &mut HashMap<String, Ty>, target: &py::Expr, ty: Ty) {
    match target {
        py::Expr::Name(n) => {
            vars.insert(n.id.to_string(), ty);
        }
        py::Expr::Tuple(t) => {
            for e in &t.elts {
                bind_target(vars, e, Ty::Object);
            }
        }
        _ => {}
    }
}

pub(crate) fn is_name(e: &py::Expr, name: &str) -> bool {
    matches!(e, py::Expr::Name(n) if n.id.as_str() == name)
}

/// The number the library's dynamic arithmetic switches on.
pub(crate) fn arith_code(op: py::Operator) -> i64 {
    match op {
        py::Operator::Add => 0,
        py::Operator::Sub => 1,
        py::Operator::Mult | py::Operator::MatMult => 2,
        py::Operator::Div => 3,
        py::Operator::FloorDiv => 4,
        py::Operator::Mod => 5,
        py::Operator::Pow => 6,
        py::Operator::BitAnd => 7,
        py::Operator::BitOr => 8,
        py::Operator::BitXor => 9,
        py::Operator::LShift => 10,
        py::Operator::RShift => 11,
    }
}

/// What `left op right` produces. `/` is always a float on numbers,
/// `**` with a negative literal exponent too.
pub(crate) fn binop(op: py::Operator, l: Ty, r: Ty, right: &py::Expr) -> Ty {
    match op {
        py::Operator::Div if l.is_numeric() && r.is_numeric() => Ty::Float,
        // `int ** int` is an int only when the exponent is visibly not
        // negative; otherwise Python's answer may be a float, and the
        // value decides at run time.
        py::Operator::Pow if l.is_numeric() && r.is_numeric() => {
            if l == Ty::Float || r == Ty::Float {
                Ty::Float
            } else if nonnegative_literal(right) {
                Ty::Int
            } else if l == Ty::Unknown || r == Ty::Unknown {
                Ty::Unknown
            } else {
                Ty::Object
            }
        }
        py::Operator::Add if l == Ty::Str && r == Ty::Str => Ty::Str,
        py::Operator::Add if matches!(l, Ty::List(_)) && l == r => l,
        py::Operator::Add if l == Ty::Tuple && r == Ty::Tuple => Ty::Tuple,
        py::Operator::Mult
            if matches!(l, Ty::List(_) | Ty::Tuple) && matches!(r, Ty::Int | Ty::Bool) =>
        {
            l
        }
        py::Operator::Mult
            if matches!(r, Ty::List(_) | Ty::Tuple) && matches!(l, Ty::Int | Ty::Bool) =>
        {
            r
        }
        py::Operator::Mult
            if (l == Ty::Str && matches!(r, Ty::Int | Ty::Bool))
                || (matches!(l, Ty::Int | Ty::Bool) && r == Ty::Str) =>
        {
            Ty::Str
        }
        _ if l == Ty::Str || r == Ty::Str => {
            if l == Ty::Unknown || r == Ty::Unknown {
                Ty::Unknown
            } else {
                Ty::Object
            }
        }
        _ => l.arith(r),
    }
}

fn nonnegative_literal(e: &py::Expr) -> bool {
    matches!(e, py::Expr::NumberLiteral(n) if matches!(n.value, py::Number::Int(_)))
        || matches!(e, py::Expr::UnaryOp(u) if u.op == py::UnaryOp::UAdd
            && matches!(&*u.operand, py::Expr::NumberLiteral(_)))
}

impl Typer<'_> {
    pub(crate) fn expr(&self, e: &py::Expr) -> Ty {
        match e {
            py::Expr::NumberLiteral(n) => match &n.value {
                py::Number::Int(_) => Ty::Int,
                py::Number::Float(_) => Ty::Float,
                py::Number::Complex { .. } => Ty::Object,
            },
            py::Expr::BooleanLiteral(_) => Ty::Bool,
            py::Expr::NoneLiteral(_) => Ty::None,
            py::Expr::StringLiteral(_) | py::Expr::FString(_) => Ty::Str,
            py::Expr::Name(n) => self
                .vars
                .get(n.id.as_str())
                .or_else(|| self.outer.get(n.id.as_str()))
                .or_else(|| self.module.globals.get(n.id.as_str()))
                .copied()
                .unwrap_or(Ty::Object),
            py::Expr::BinOp(b) => {
                let l = self.expr(&b.left);
                let r = self.expr(&b.right);
                binop(b.op, l, r, &b.right)
            }
            py::Expr::UnaryOp(u) => match u.op {
                py::UnaryOp::Not => Ty::Bool,
                py::UnaryOp::Invert => match self.expr(&u.operand) {
                    Ty::Int | Ty::Bool => Ty::Int,
                    Ty::Unknown => Ty::Unknown,
                    _ => Ty::Object,
                },
                py::UnaryOp::UAdd | py::UnaryOp::USub => match self.expr(&u.operand) {
                    Ty::Bool => Ty::Int,
                    t => t,
                },
            },
            py::Expr::Compare(_) => Ty::Bool,
            py::Expr::BoolOp(b) => {
                let mut acc = Ty::Unknown;
                for v in &b.values {
                    acc = acc.join(self.expr(v));
                }
                acc
            }
            py::Expr::If(i) => self.expr(&i.body).join(self.expr(&i.orelse)),
            py::Expr::Call(c) => self.call(c),
            py::Expr::Subscript(s) => {
                let seq = self.expr(&s.value);
                if matches!(&*s.slice, py::Expr::Slice(_)) {
                    match seq {
                        Ty::Str | Ty::List(_) | Ty::Tuple => seq,
                        _ => Ty::Object,
                    }
                } else {
                    seq.element().unwrap_or(Ty::Object)
                }
            }
            py::Expr::List(l) => Ty::List(self.elem_of(l.elts.iter())),
            py::Expr::ListComp(c) => {
                let mut vars = self.vars.clone();
                for g in &c.generators {
                    bind_target(
                        &mut vars,
                        &g.target,
                        self.expr(&g.iter).element().unwrap_or(Ty::Object),
                    );
                }
                let inner = Typer {
                    module: self.module,
                    vars: &vars,
                    outer: self.outer,
                };
                Ty::List(Elem::of(inner.expr(&c.elt)))
            }
            py::Expr::Tuple(_) => Ty::Tuple,
            _ => Ty::Object,
        }
    }

    /// The element kind of a literal: the one kind every element has,
    /// or dynamic when they differ.
    fn elem_of<'e>(&self, elts: impl Iterator<Item = &'e py::Expr>) -> Elem {
        let mut kind: Option<Elem> = None;
        for e in elts {
            let k = Elem::of(self.expr(e));
            kind = Some(match kind {
                None => k,
                Some(prev) if prev == k => k,
                Some(_) => return Elem::Object,
            });
        }
        kind.unwrap_or(Elem::Object)
    }

    fn call(&self, c: &py::ExprCall) -> Ty {
        let args = &c.arguments.args;
        let arg = |i: usize| args.get(i).map(|a| self.expr(a)).unwrap_or(Ty::Unknown);
        match &*c.func {
            py::Expr::Name(n) => {
                let name = n.id.as_str();
                if let Some(sig) = self.module.funcs.get(name) {
                    return sig.ret;
                }
                match name {
                    "print" => Ty::None,
                    // A range is iterated as ints.
                    "range" => Ty::List(Elem::Int),
                    "len" | "int" | "ord" | "hash" | "id" => Ty::Int,
                    "sorted" | "reversed" | "list" => match arg(0) {
                        Ty::List(e) => Ty::List(e),
                        Ty::Str => Ty::List(Elem::Str),
                        Ty::Tuple => Ty::List(Elem::Object),
                        _ => match args.first() {
                            Some(py::Expr::Call(c)) if is_name(&c.func, "range") => {
                                Ty::List(Elem::Int)
                            }
                            _ => Ty::List(Elem::Object),
                        },
                    },
                    "tuple" => Ty::Tuple,
                    "sum" => match arg(0) {
                        Ty::List(Elem::Int) => Ty::Int,
                        Ty::List(Elem::Float) => Ty::Float,
                        _ => Ty::Object,
                    },
                    "min" | "max" if args.len() == 1 => match arg(0) {
                        Ty::List(e) => e.ty(),
                        _ => Ty::Object,
                    },
                    "divmod" => Ty::Tuple,
                    "type" => Ty::Str,
                    "str" | "repr" | "input" | "chr" => Ty::Str,
                    "float" => Ty::Float,
                    "bool" | "isinstance" | "callable" | "hasattr" => Ty::Bool,
                    "abs" => match arg(0) {
                        Ty::Int | Ty::Bool => Ty::Int,
                        Ty::Float => Ty::Float,
                        Ty::Unknown => Ty::Unknown,
                        _ => Ty::Object,
                    },
                    "round" if args.len() == 1 => match arg(0) {
                        Ty::Int | Ty::Float | Ty::Bool => Ty::Int,
                        Ty::Unknown => Ty::Unknown,
                        _ => Ty::Object,
                    },
                    "min" | "max" if args.len() >= 2 => {
                        let mut acc = Ty::Unknown;
                        for a in args.iter() {
                            acc = acc.join(self.expr(a));
                        }
                        acc
                    }
                    "pow" if args.len() == 2 => binop(py::Operator::Pow, arg(0), arg(1), &args[1]),
                    _ => Ty::Object,
                }
            }
            // A method on a list.
            py::Expr::Attribute(a) if matches!(self.expr(&a.value), Ty::List(_)) => {
                let Ty::List(e) = self.expr(&a.value) else {
                    unreachable!()
                };
                match a.attr.as_str() {
                    "pop" => e.ty(),
                    "index" | "count" => Ty::Int,
                    "copy" => Ty::List(e),
                    _ => Ty::None,
                }
            }
            // A method on a string, when the receiver is known to be one.
            py::Expr::Attribute(a) if self.expr(&a.value) == Ty::Str => match a.attr.as_str() {
                "upper" | "lower" | "strip" | "lstrip" | "rstrip" | "replace" | "join"
                | "capitalize" | "title" | "swapcase" | "format" | "zfill" | "center" | "ljust"
                | "rjust" => Ty::Str,
                "find" | "rfind" | "index" | "rindex" | "count" => Ty::Int,
                "startswith" | "endswith" | "isdigit" | "isalpha" | "isalnum" | "isspace"
                | "isupper" | "islower" => Ty::Bool,
                "split" | "rsplit" | "splitlines" => Ty::List(Elem::Str),
                _ => Ty::Object,
            },
            _ => Ty::Object,
        }
    }
}

pub(crate) fn stmt_kind(s: &py::Stmt) -> &'static str {
    match s {
        py::Stmt::FunctionDef(_) => "def",
        py::Stmt::ClassDef(_) => "class",
        py::Stmt::Return(_) => "return",
        py::Stmt::Delete(_) => "del",
        py::Stmt::TypeAlias(_) => "type alias",
        py::Stmt::Assign(_) => "assignment",
        py::Stmt::AugAssign(_) => "augmented assignment",
        py::Stmt::AnnAssign(_) => "annotated assignment",
        py::Stmt::For(_) => "for",
        py::Stmt::While(_) => "while",
        py::Stmt::If(_) => "if",
        py::Stmt::With(_) => "with",
        py::Stmt::Match(_) => "match",
        py::Stmt::Raise(_) => "raise",
        py::Stmt::Try(_) => "try",
        py::Stmt::Assert(_) => "assert",
        py::Stmt::Import(_) | py::Stmt::ImportFrom(_) => "import",
        py::Stmt::Global(_) => "global",
        py::Stmt::Nonlocal(_) => "nonlocal",
        py::Stmt::Expr(_) => "expression statement",
        py::Stmt::Pass(_) => "pass",
        py::Stmt::Break(_) => "break",
        py::Stmt::Continue(_) => "continue",
        py::Stmt::IpyEscapeCommand(_) => "IPython escape",
    }
}

pub(crate) fn expr_kind(e: &py::Expr) -> &'static str {
    match e {
        py::Expr::BoolOp(_) => "boolean operator",
        py::Expr::Named(_) => "walrus",
        py::Expr::BinOp(_) => "binary operator",
        py::Expr::UnaryOp(_) => "unary operator",
        py::Expr::Lambda(_) => "lambda",
        py::Expr::If(_) => "conditional expression",
        py::Expr::Dict(_) => "dict literal",
        py::Expr::Set(_) => "set literal",
        py::Expr::ListComp(_) => "list comprehension",
        py::Expr::SetComp(_) => "set comprehension",
        py::Expr::DictComp(_) => "dict comprehension",
        py::Expr::Generator(_) => "generator expression",
        py::Expr::Await(_) => "await",
        py::Expr::Yield(_) | py::Expr::YieldFrom(_) => "yield",
        py::Expr::Compare(_) => "comparison",
        py::Expr::Call(_) => "call",
        py::Expr::FString(_) => "f-string",
        py::Expr::TString(_) => "t-string",
        py::Expr::StringLiteral(_) => "string",
        py::Expr::BytesLiteral(_) => "bytes",
        py::Expr::NumberLiteral(_) => "number",
        py::Expr::BooleanLiteral(_) => "bool",
        py::Expr::NoneLiteral(_) => "None",
        py::Expr::EllipsisLiteral(_) => "...",
        py::Expr::Attribute(_) => "attribute access",
        py::Expr::Subscript(_) => "subscript",
        py::Expr::Starred(_) => "starred expression",
        py::Expr::Name(_) => "name",
        py::Expr::List(_) => "list literal",
        py::Expr::Tuple(_) => "tuple",
        py::Expr::Slice(_) => "slice",
        py::Expr::IpyEscapeCommand(_) => "IPython escape",
    }
}
