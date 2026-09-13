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
    /// A dict: keys and values, dynamic, in insertion order.
    Dict,
    /// A set of dynamic values.
    Set,
    /// An instance of the module's class at this index.
    Class(u16),
    /// A generator: a fiber yielding dynamic values.
    Gen,
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
    /// Instances of one class, held by address, so an element is the
    /// instance itself rather than a box to open.
    Class(u16),
    Object,
}

impl Elem {
    /// The element kind a value of `ty` is stored as.
    pub(crate) fn of(ty: Ty) -> Elem {
        match ty {
            Ty::Int => Elem::Int,
            Ty::Float => Elem::Float,
            Ty::Str => Elem::Str,
            Ty::Class(k) => Elem::Class(k),
            _ => Elem::Object,
        }
    }

    pub(crate) fn ty(self) -> Ty {
        match self {
            Elem::Int => Ty::Int,
            Elem::Float => Ty::Float,
            Elem::Str => Ty::Str,
            Elem::Class(k) => Ty::Class(k),
            Elem::Object => Ty::Object,
        }
    }

    /// The suffix of the library functions for this kind.
    pub(crate) fn suffix(self) -> &'static str {
        match self {
            Elem::Int => "i64",
            Elem::Float => "f64",
            Elem::Str => "str",
            Elem::Class(_) => "ptr",
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
            Ty::Tuple | Ty::Dict | Ty::Set | Ty::Gen => Some(Ty::Object),
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
    /// The module's classes, bases before subclasses.
    pub(crate) classes: Vec<ClassInfo>,
    pub(crate) class_index: HashMap<String, usize>,
    /// Attribute names read and written on dynamic receivers, and
    /// methods called on them with an argument count: each needs a
    /// dispatcher over the classes that have it.
    pub(crate) attr_reads: std::cell::RefCell<std::collections::BTreeSet<String>>,
    pub(crate) attr_writes: std::cell::RefCell<std::collections::BTreeSet<String>>,
    pub(crate) dyn_methods: std::cell::RefCell<std::collections::BTreeSet<(String, usize)>>,
    /// Library functions that can raise.
    pub(crate) fallible: std::collections::BTreeSet<String>,
    pub(crate) list_type: Option<zyntax_typed_ast::TypeId>,
    /// What `__name__` is in this module.
    pub(crate) name: String,
    /// `import m [as n]`: the name a module goes by, to the module.
    pub(crate) imports: HashMap<String, String>,
    /// `from m import x [as y]`: the local name, to the module and the
    /// member.
    pub(crate) from_names: HashMap<String, (String, String)>,
    /// The program's own modules, to the index of their source file.
    pub(crate) files: HashMap<String, u32>,
}

impl Module {
    /// The source file a module's statements are in; the main file when
    /// `module` is `None`.
    pub(crate) fn file_of(&self, module: Option<&str>) -> u32 {
        module.and_then(|m| self.files.get(m).copied()).unwrap_or(0)
    }

    /// What a module-qualified name stands for, when `alias` names an
    /// imported module and nothing shadows it.
    pub(crate) fn module_member(&self, alias: &str, name: &str) -> Option<crate::stdlib::Member> {
        let module = self.imports.get(alias)?;
        crate::stdlib::member(module, name)
    }

    /// What a name brought in by `from m import x` stands for.
    pub(crate) fn imported_name(&self, name: &str) -> Option<crate::stdlib::Member> {
        let (module, member) = self.from_names.get(name)?;
        crate::stdlib::member(module, member)
    }
}

/// The type of what a module member evaluates to, or of what calling it
/// returns.
pub(crate) fn member_ty(member: crate::stdlib::Member) -> Ty {
    match member {
        crate::stdlib::Member::Func { ret, .. } => ret,
        crate::stdlib::Member::Float(_) => Ty::Float,
        crate::stdlib::Member::Int(_) => Ty::Int,
        crate::stdlib::Member::Value { ty, .. } => ty,
    }
}

/// A class: its place in the hierarchy, its layout and its methods.
#[derive(Debug, Clone, Default)]
pub(crate) struct ClassInfo {
    pub(crate) name: String,
    pub(crate) base: Option<usize>,
    /// Every field in layout order: the class tag first, then the
    /// base's fields, then this class's own.
    pub(crate) fields: Vec<(String, Ty)>,
    /// The methods this class itself defines.
    pub(crate) methods: Vec<String>,
    pub(crate) type_id: Option<zyntax_typed_ast::TypeId>,
}

/// The name of the function a method lowers to.
pub(crate) fn method_fn(class: &str, method: &str) -> String {
    format!("{class}${method}")
}

impl Module {
    /// The index of a field on class `k`, inherited or own, and its type.
    pub(crate) fn field(&self, k: usize, name: &str) -> Option<(usize, Ty)> {
        self.classes[k]
            .fields
            .iter()
            .position(|(f, _)| f == name)
            .map(|i| (i, self.classes[k].fields[i].1))
    }

    /// The class in `k`'s chain that defines `method`, nearest first.
    pub(crate) fn method_owner(&self, k: usize, method: &str) -> Option<usize> {
        let mut at = Some(k);
        while let Some(c) = at {
            if self.classes[c].methods.iter().any(|m| m == method) {
                return Some(c);
            }
            at = self.classes[c].base;
        }
        None
    }

    /// The signature and function name of `method` as `k` sees it.
    pub(crate) fn method_sig(&self, k: usize, method: &str) -> Option<(&Sig, String)> {
        let owner = self.method_owner(k, method)?;
        let name = method_fn(&self.classes[owner].name, method);
        self.funcs.get(&name).map(|sig| (sig, name))
    }

    /// Whether `k` is `base` or derives from it.
    pub(crate) fn is_subclass(&self, k: usize, base: usize) -> bool {
        let mut at = Some(k);
        while let Some(c) = at {
            if c == base {
                return true;
            }
            at = self.classes[c].base;
        }
        false
    }

    /// The classes deriving from `owner` that define `method` themselves.
    pub(crate) fn overriders(&self, owner: usize, method: &str) -> Vec<usize> {
        (0..self.classes.len())
            .filter(|&c| {
                c != owner
                    && self.is_subclass(c, owner)
                    && self.classes[c].methods.iter().any(|m| m == method)
            })
            .collect()
    }
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
    /// Attributes assigned on the first parameter (`self.x = ...`), and
    /// what is assigned to them.
    pub(crate) field_writes: HashMap<String, Ty>,
}

/// An annotation, with the module's class names known.
pub(crate) fn annotation_in(classes: &HashMap<String, usize>, e: &py::Expr) -> Ty {
    match e {
        py::Expr::Name(n) => match n.id.as_str() {
            "int" => Ty::Int,
            "float" => Ty::Float,
            "bool" => Ty::Bool,
            "str" => Ty::Str,
            "None" => Ty::None,
            "list" | "List" | "Sequence" | "Iterable" => Ty::List(Elem::Object),
            "dict" | "Dict" | "Mapping" => Ty::Dict,
            "set" | "Set" => Ty::Set,
            "tuple" | "Tuple" => Ty::Tuple,
            other => classes
                .get(other)
                .map(|k| Ty::Class(*k as u16))
                .unwrap_or(Ty::Object),
        },
        py::Expr::NoneLiteral(_) => Ty::None,
        // `list[int]` and friends: the outer name decides.
        py::Expr::Subscript(sub) => match annotation_in(classes, &sub.value) {
            Ty::List(_) => match annotation_in(classes, &sub.slice) {
                Ty::Int => Ty::List(Elem::Int),
                Ty::Float => Ty::List(Elem::Float),
                Ty::Str => Ty::List(Elem::Str),
                Ty::Class(k) => Ty::List(Elem::Class(k)),
                _ => Ty::List(Elem::Object),
            },
            other => other,
        },
        _ => Ty::Object,
    }
}

/// Signature from the annotations alone; an unannotated return is
/// `Unknown` until the body says.
pub(crate) fn declared_sig(f: &py::StmtFunctionDef) -> Sig {
    declared_sig_in(&HashMap::new(), f, None)
}

/// [`declared_sig`] with class names resolved, and `self` typed as the
/// class a method belongs to.
pub(crate) fn declared_sig_in(
    classes: &HashMap<String, usize>,
    f: &py::StmtFunctionDef,
    class: Option<usize>,
) -> Sig {
    let params = f
        .parameters
        .iter_non_variadic_params()
        .enumerate()
        .map(|(i, p)| {
            let ty = match (i, class, &p.parameter.annotation) {
                (0, Some(k), None) => Ty::Class(k as u16),
                (_, _, Some(a)) => annotation_in(classes, a),
                _ => Ty::Object,
            };
            (p.parameter.name.to_string(), ty)
        })
        .collect();
    let defaults = f
        .parameters
        .iter_non_variadic_params()
        .map(|p| p.default.as_deref().cloned())
        .collect();
    let ret = if is_generator(&f.body) {
        Ty::Gen
    } else {
        f.returns
            .as_deref()
            .map(|r| annotation_in(classes, r))
            .unwrap_or(Ty::Unknown)
    };
    Sig {
        params,
        ret,
        defaults,
    }
}

/// Whether a body yields, making its function a generator. Nested
/// functions yield for themselves.
pub(crate) fn is_generator(body: &[py::Stmt]) -> bool {
    use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
    #[derive(Default)]
    struct Finder(bool);
    impl<'a> Visitor<'a> for Finder {
        fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
            if !matches!(stmt, py::Stmt::FunctionDef(_) | py::Stmt::ClassDef(_)) {
                walk_stmt(self, stmt);
            }
        }
        fn visit_expr(&mut self, expr: &'a py::Expr) {
            match expr {
                py::Expr::Yield(_) | py::Expr::YieldFrom(_) => self.0 = true,
                py::Expr::Lambda(_) | py::Expr::Generator(_) => {}
                _ => walk_expr(self, expr),
            }
        }
    }
    let mut finder = Finder::default();
    for s in body {
        finder.visit_stmt(s);
    }
    finder.0
}

/// Infer the module's signatures to a fixed point.
/// A function the module defines: the name it lowers to, the class it
/// is a method of, and its definition.
pub(crate) struct Item<'a> {
    pub(crate) name: String,
    pub(crate) class: Option<usize>,
    pub(crate) def: &'a py::StmtFunctionDef,
    /// The program's module the definition is written in; `None` for
    /// the main file.
    pub(crate) module: Option<String>,
}

/// Infer the module's signatures to a fixed point. Class layouts are
/// taken from `known` and refined from what methods assign to `self`.
pub(crate) fn infer_module(
    known: &Module,
    items: &[Item<'_>],
) -> (HashMap<String, Sig>, Vec<ClassInfo>) {
    let mut module = Module {
        funcs: HashMap::new(),
        globals: known.globals.clone(),
        list_type: known.list_type,
        classes: known.classes.clone(),
        class_index: known.class_index.clone(),
        ..Default::default()
    };
    for item in items {
        module.funcs.insert(
            item.name.clone(),
            declared_sig_in(&module.class_index, item.def, item.class),
        );
    }
    for _ in 0..8 {
        let mut changed = false;
        for item in items {
            let sig = module.funcs[&item.name].clone();
            let locals = infer_locals(&module, &sig, &item.def.body);
            if item.def.returns.is_none() && sig.ret != Ty::Gen {
                let ret = if locals.ret == Ty::Unknown {
                    Ty::None
                } else {
                    locals.ret
                };
                if ret != sig.ret {
                    module.funcs.get_mut(&item.name).unwrap().ret = ret;
                    changed = true;
                }
            }
            // What a method assigns to `self.x` types the field on its
            // class, and on every class deriving from it.
            if let Some(k) = item.class {
                for (field, ty) in &locals.field_writes {
                    changed |= widen_field(&mut module.classes, k, field, *ty);
                }
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
    for class in &mut module.classes {
        for (_, ty) in &mut class.fields {
            if *ty == Ty::Unknown {
                *ty = Ty::Object;
            }
        }
    }
    normalize_layouts(&mut module.classes);
    (module.funcs, module.classes)
}

/// Lay each class out as its base's fields followed by its own, so an
/// instance reads correctly through a base-typed reference. Bases come
/// before their subclasses in the list.
fn normalize_layouts(classes: &mut [ClassInfo]) {
    for c in 0..classes.len() {
        let Some(b) = classes[c].base else {
            continue;
        };
        let mut fields = classes[b].fields.clone();
        for (name, ty) in std::mem::take(&mut classes[c].fields) {
            match fields.iter_mut().find(|(f, _)| *f == name) {
                Some(slot) => slot.1 = slot.1.join(ty),
                None => fields.push((name, ty)),
            }
        }
        classes[c].fields = fields;
    }
}

/// Join `ty` into field `name` of class `k` and of every subclass, adding
/// the field where it is new. Returns whether anything changed.
fn widen_field(classes: &mut [ClassInfo], k: usize, name: &str, ty: Ty) -> bool {
    let mut changed = false;
    let targets: Vec<usize> = (0..classes.len())
        .filter(|&c| {
            let mut at = Some(c);
            while let Some(x) = at {
                if x == k {
                    return true;
                }
                at = classes[x].base;
            }
            false
        })
        .collect();
    for c in targets {
        match classes[c].fields.iter().position(|(f, _)| f == name) {
            Some(i) => {
                let joined = classes[c].fields[i].1.join(ty);
                if joined != classes[c].fields[i].1 {
                    classes[c].fields[i].1 = joined;
                    changed = true;
                }
            }
            None => {
                classes[c].fields.push((name.to_string(), ty));
                changed = true;
            }
        }
    }
    changed
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
            // `self.x = v` in a method declares the field.
            py::Expr::Attribute(a)
                if matches!(&*a.value, py::Expr::Name(n)
                    if self.params.first().is_some_and(|(p, _)| p == n.id.as_str())) =>
            {
                let joined = self
                    .locals
                    .field_writes
                    .get(a.attr.as_str())
                    .copied()
                    .unwrap_or(Ty::Unknown)
                    .join(ty);
                self.locals.field_writes.insert(a.attr.to_string(), joined);
            }
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
                        // `except E as e` binds an instance of E.
                        let ty = match h.type_.as_deref() {
                            Some(py::Expr::Name(n)) => self
                                .module
                                .class_index
                                .get(n.id.as_str())
                                .map(|k| Ty::Class(*k as u16))
                                .unwrap_or(Ty::Object),
                            _ => Ty::Object,
                        };
                        self.assign(name.as_str(), ty);
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

/// `super()` with no arguments.
pub(crate) fn is_super_call(e: &py::Expr) -> bool {
    matches!(e, py::Expr::Call(c) if is_name(&c.func, "super") && c.arguments.args.is_empty())
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
        py::Operator::BitAnd | py::Operator::BitOr | py::Operator::Sub | py::Operator::BitXor
            if l == Ty::Set && r == Ty::Set =>
        {
            Ty::Set
        }
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
            py::Expr::Name(n) => {
                let name = n.id.as_str();
                if let Some(ty) = self
                    .vars
                    .get(name)
                    .or_else(|| self.outer.get(name))
                    .or_else(|| self.module.globals.get(name))
                {
                    return *ty;
                }
                if name == "__name__" {
                    return Ty::Str;
                }
                match self.module.imported_name(name) {
                    Some(m) => member_ty(m),
                    None => Ty::Object,
                }
            }
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
            py::Expr::Attribute(a) => {
                if let Some(m) = self.module_member_of(&a.value, a.attr.as_str()) {
                    return member_ty(m);
                }
                match self.expr(&a.value) {
                    Ty::Class(k) => self
                        .module
                        .field(k as usize, a.attr.as_str())
                        .map(|(_, ty)| ty)
                        .unwrap_or(Ty::Object),
                    _ => Ty::Object,
                }
            }
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
            py::Expr::Dict(_) | py::Expr::DictComp(_) => Ty::Dict,
            py::Expr::Set(_) | py::Expr::SetComp(_) => Ty::Set,
            py::Expr::Generator(_) => Ty::Gen,
            py::Expr::Yield(_) | py::Expr::YieldFrom(_) => Ty::None,
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

    /// The module member `value.attr` names, when `value` is an imported
    /// module's name and no variable shadows it.
    pub(crate) fn module_member_of(
        &self,
        value: &py::Expr,
        attr: &str,
    ) -> Option<crate::stdlib::Member> {
        let py::Expr::Name(m) = value else {
            return None;
        };
        let alias = m.id.as_str();
        if self.vars.contains_key(alias)
            || self.outer.contains_key(alias)
            || self.module.globals.contains_key(alias)
        {
            return None;
        }
        self.module.module_member(alias, attr)
    }

    fn call(&self, c: &py::ExprCall) -> Ty {
        let args = &c.arguments.args;
        let arg = |i: usize| args.get(i).map(|a| self.expr(a)).unwrap_or(Ty::Unknown);
        if let py::Expr::Attribute(a) = &*c.func {
            if let Some(m) = self.module_member_of(&a.value, a.attr.as_str()) {
                return member_ty(m);
            }
        }
        match &*c.func {
            py::Expr::Name(n) => {
                let name = n.id.as_str();
                if let Some(k) = self.module.class_index.get(name) {
                    return Ty::Class(*k as u16);
                }
                if let Some(sig) = self.module.funcs.get(name) {
                    return sig.ret;
                }
                if !self.vars.contains_key(name) && !self.outer.contains_key(name) {
                    if let Some(m) = self.module.imported_name(name) {
                        return member_ty(m);
                    }
                }
                match name {
                    "print" => Ty::None,
                    // A range is iterated as ints.
                    "range" => Ty::List(Elem::Int),
                    "len" | "int" | "ord" | "hash" | "id" => Ty::Int,
                    "next" => Ty::Object,
                    "sorted" | "reversed" | "list" => match arg(0) {
                        Ty::List(e) => Ty::List(e),
                        Ty::Str => Ty::List(Elem::Str),
                        Ty::Tuple | Ty::Dict | Ty::Set | Ty::Gen => Ty::List(Elem::Object),
                        _ => match args.first() {
                            Some(py::Expr::Call(c)) if is_name(&c.func, "range") => {
                                Ty::List(Elem::Int)
                            }
                            _ => Ty::List(Elem::Object),
                        },
                    },
                    "tuple" => Ty::Tuple,
                    "dict" => Ty::Dict,
                    "set" => Ty::Set,
                    // Pairs and mapped values are dynamic; the lists are eager.
                    "enumerate" | "zip" | "map" | "filter" => Ty::List(Elem::Object),
                    "any" | "all" => Ty::Bool,
                    "sum" => {
                        let items = match arg(0) {
                            Ty::List(Elem::Int) => Ty::Int,
                            Ty::List(Elem::Float) => Ty::Float,
                            _ => Ty::Object,
                        };
                        match args.get(1) {
                            None => items,
                            Some(start) => match (items, self.expr(start)) {
                                (Ty::Int, Ty::Int | Ty::Bool) => Ty::Int,
                                (Ty::Int | Ty::Float, Ty::Float)
                                | (Ty::Float, Ty::Int | Ty::Bool) => Ty::Float,
                                _ => Ty::Object,
                            },
                        }
                    }
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
                    // With digits, the result keeps the argument's type.
                    "round" if args.len() == 2 => match arg(0) {
                        Ty::Int | Ty::Bool => Ty::Int,
                        Ty::Float => Ty::Float,
                        Ty::Unknown => Ty::Unknown,
                        _ => Ty::Object,
                    },
                    "pow" if args.len() == 3 => Ty::Int,
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
            // A method on an instance: what the defining class says.
            py::Expr::Attribute(a) if matches!(self.expr(&a.value), Ty::Class(_)) => {
                let Ty::Class(k) = self.expr(&a.value) else {
                    unreachable!()
                };
                match self.module.method_sig(k as usize, a.attr.as_str()) {
                    Some((sig, _)) => sig.ret,
                    None => Ty::Object,
                }
            }
            // `super().m(...)`: the base's method.
            py::Expr::Attribute(a) if is_super_call(&a.value) => {
                match self.vars.get("self").copied() {
                    Some(Ty::Class(k)) => {
                        let base = self.module.classes[k as usize].base;
                        match base.and_then(|b| self.module.method_sig(b, a.attr.as_str())) {
                            Some((sig, _)) => sig.ret,
                            None => Ty::Object,
                        }
                    }
                    _ => Ty::Object,
                }
            }
            py::Expr::Attribute(a) if self.expr(&a.value) == Ty::Dict => match a.attr.as_str() {
                "keys" | "values" | "items" => Ty::List(Elem::Object),
                "copy" => Ty::Dict,
                "clear" | "update" => Ty::None,
                _ => Ty::Object,
            },
            py::Expr::Attribute(a) if self.expr(&a.value) == Ty::Set => match a.attr.as_str() {
                "add" | "remove" | "discard" | "clear" | "update" => Ty::None,
                "union" | "intersection" | "difference" | "symmetric_difference" | "copy" => {
                    Ty::Set
                }
                "issubset" | "issuperset" | "isdisjoint" => Ty::Bool,
                _ => Ty::Object,
            },
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
