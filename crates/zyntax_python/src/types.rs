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
//! An unannotated parameter is dynamic, as Python has it, unless every
//! call of its function is in view; then it is the join of what the
//! calls pass and what the body assigns to it, which is exact where the
//! program is consistent and dynamic where it is not.

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
    /// A function value whose function is known: the closure at this
    /// index of the module's table. Carried as the record every function
    /// value is, so it is a dynamic value wherever one is needed; where
    /// it is called, the call is direct and typed.
    Closure(u16),
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
            Ty::Unknown => Some(Ty::Unknown),
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
    /// Functions every call of which is in view, so an unannotated
    /// parameter can be typed by what is passed; see [`closed_items`].
    pub(crate) closed: std::collections::HashSet<String>,
    /// Every lambda and nested `def` in the program, by the function
    /// holding it and its position in the source, with what inference
    /// knows of each. Filled once before inference; the signatures are
    /// refined during it.
    pub(crate) closures: std::cell::RefCell<Vec<ClosureInfo>>,
    /// Closure index by `(file, start of range)`.
    pub(crate) closure_index: HashMap<(u32, u32), u16>,
}

/// A lambda or nested `def`: a function value whose function is known
/// wherever the value's type is.
#[derive(Debug, Clone)]
pub(crate) struct ClosureInfo {
    /// The function it lowers to. The record's code is the adapter of
    /// the same name that unboxes for a call through a value; a direct
    /// call names [`Self::typed_name`].
    pub(crate) name: String,
    /// Its parameters as inferred, and what it returns.
    pub(crate) sig: Sig,
    /// Which parameters carry no annotation and so are typed by the
    /// calls.
    pub(crate) inferred: Vec<bool>,
    /// Whether the return type is inferred from the body rather than
    /// annotated.
    pub(crate) ret_inferred: bool,
    /// Whether a value of this closure reaches anywhere but a call, a
    /// local, or a return. Then a call through it may come from code
    /// the inference cannot see, and its parameters stay dynamic.
    pub(crate) escapes: bool,
}

impl ClosureInfo {
    /// The direct entry: typed parameters after the record.
    pub(crate) fn typed_name(&self) -> String {
        format!("{}$typed", self.name)
    }

    /// Back to what the source declares, for another round of inference.
    fn reset(&mut self) {
        for (flag, (_, ty)) in self.inferred.iter().zip(self.sig.params.iter_mut()) {
            if *flag {
                *ty = Ty::Unknown;
            }
        }
        if self.ret_inferred {
            self.sig.ret = Ty::Unknown;
        }
        self.escapes = false;
    }
}

/// Run `f` with `file` as the file the statements being typed are in.
pub(crate) fn in_file<T>(file: u32, f: impl FnOnce() -> T) -> T {
    crate::lower::set_current_file(file);
    let out = f();
    crate::lower::set_current_file(0);
    out
}

/// Every lambda and nested `def` under `body`, at any depth, as closures
/// of the file `file`. Generators keep their own lowering and are not
/// closures here. `owner` names the function the body belongs to; a
/// closure inside a closure is named after the outer one in turn.
pub(crate) fn collect_closures(
    module: &mut Module,
    owner: &str,
    file: u32,
    body: &[py::Stmt],
    classes: &HashMap<String, usize>,
) {
    use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
    struct Finder<'m, 'c> {
        module: &'m mut Module,
        classes: &'c HashMap<String, usize>,
        owner: Vec<String>,
        file: u32,
    }
    impl Finder<'_, '_> {
        fn add(
            &mut self,
            start: u32,
            kind: &str,
            sig: Sig,
            inferred: Vec<bool>,
            ret_inferred: bool,
        ) -> String {
            let index = self.module.closures.borrow().len();
            let name = format!(
                "{}${kind}${index}",
                self.owner.last().cloned().unwrap_or_default()
            );
            self.module.closures.borrow_mut().push(ClosureInfo {
                name: name.clone(),
                sig,
                inferred,
                ret_inferred,
                escapes: false,
            });
            self.module
                .closure_index
                .insert((self.file, start), index as u16);
            name
        }
    }
    impl<'a> Visitor<'a> for Finder<'_, '_> {
        fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
            match stmt {
                // A nested def that yields is a generator, lowered on its
                // own terms; one with variadics is not compiled at all.
                // A default is evaluated where the def is, which a call
                // from elsewhere cannot see.
                py::Stmt::FunctionDef(f)
                    if !is_generator(&f.body)
                        && f.parameters.vararg.is_none()
                        && f.parameters.kwarg.is_none()
                        && f.decorator_list.is_empty()
                        && f.parameters
                            .iter_non_variadic_params()
                            .all(|p| p.default.is_none()) =>
                {
                    let mut sig = declared_sig_in(self.classes, f, None);
                    let inferred: Vec<bool> = f
                        .parameters
                        .iter_non_variadic_params()
                        .map(|p| p.parameter.annotation.is_none())
                        .collect();
                    for (flag, (_, ty)) in inferred.iter().zip(sig.params.iter_mut()) {
                        if *flag {
                            *ty = Ty::Unknown;
                        }
                    }
                    if f.returns.is_none() {
                        sig.ret = Ty::Unknown;
                    }
                    let name = self.add(
                        f.range.start().to_u32(),
                        f.name.as_str(),
                        sig,
                        inferred,
                        f.returns.is_none(),
                    );
                    self.owner.push(name);
                    walk_stmt(self, stmt);
                    self.owner.pop();
                }
                py::Stmt::ClassDef(_) => {}
                _ => walk_stmt(self, stmt),
            }
        }
        fn visit_expr(&mut self, expr: &'a py::Expr) {
            if let py::Expr::Lambda(l) = expr {
                let params: Vec<(String, Ty)> = l
                    .parameters
                    .as_ref()
                    .map(|ps| {
                        ps.iter_non_variadic_params()
                            .map(|p| (p.parameter.name.to_string(), Ty::Unknown))
                            .collect()
                    })
                    .unwrap_or_default();
                let variadic = l
                    .parameters
                    .as_ref()
                    .is_some_and(|ps| ps.vararg.is_some() || ps.kwarg.is_some());
                let with_default = l
                    .parameters
                    .as_ref()
                    .is_some_and(|ps| ps.iter_non_variadic_params().any(|p| p.default.is_some()));
                if !variadic && !with_default {
                    let n = params.len();
                    let sig = Sig {
                        params,
                        ret: Ty::Unknown,
                        defaults: vec![None; n],
                    };
                    let name =
                        self.add(l.range.start().to_u32(), "lambda", sig, vec![true; n], true);
                    self.owner.push(name);
                    walk_expr(self, expr);
                    self.owner.pop();
                    return;
                }
            }
            walk_expr(self, expr);
        }
    }
    let mut finder = Finder {
        module,
        classes,
        owner: vec![owner.to_string()],
        file,
    };
    for s in body {
        finder.visit_stmt(s);
    }
}

/// The closures directly inside `body`: not those inside a nested
/// function, which are that function's. `files` gives the file of each
/// top-level statement where the body spans several; empty otherwise.
fn closures_directly_in(
    module: &Module,
    body: &[py::Stmt],
    files: &[u32],
) -> Vec<(u16, ClosureDef)> {
    use ruff_python_ast::visitor::Visitor;
    let mut d = Direct {
        module,
        found: Vec::new(),
    };
    for (i, s) in body.iter().enumerate() {
        match files.get(i) {
            Some(&file) => in_file(file, || d.visit_stmt(s)),
            None => d.visit_stmt(s),
        }
    }
    d.found
}

/// [`closures_directly_in`] for a lambda's body.
fn closures_directly_in_expr(module: &Module, body: &py::Expr) -> Vec<(u16, ClosureDef)> {
    use ruff_python_ast::visitor::Visitor;
    let mut d = Direct {
        module,
        found: Vec::new(),
    };
    d.visit_expr(body);
    d.found
}

struct Direct<'m> {
    module: &'m Module,
    found: Vec<(u16, ClosureDef)>,
}

impl<'a> ruff_python_ast::visitor::Visitor<'a> for Direct<'_> {
    fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
        use ruff_python_ast::visitor::walk_stmt;
        match stmt {
            py::Stmt::FunctionDef(f) => {
                if let Some(k) = self
                    .module
                    .closure_at(crate::lower::current_file(), f.range.start().to_u32())
                {
                    self.found.push((k, ClosureDef::Def(f.clone())));
                }
            }
            py::Stmt::ClassDef(_) => {}
            _ => walk_stmt(self, stmt),
        }
    }
    fn visit_expr(&mut self, expr: &'a py::Expr) {
        use ruff_python_ast::visitor::walk_expr;
        match expr {
            py::Expr::Lambda(l) => {
                if let Some(k) = self
                    .module
                    .closure_at(crate::lower::current_file(), l.range.start().to_u32())
                {
                    self.found.push((k, ClosureDef::Lambda(l.clone())));
                }
            }
            py::Expr::ListComp(_)
            | py::Expr::SetComp(_)
            | py::Expr::DictComp(_)
            | py::Expr::Generator(_) => {}
            _ => walk_expr(self, expr),
        }
    }
}

/// Type the closures directly inside `body` against `vars`, its
/// variables; see [`infer_closure`]. Returns whether any signature
/// changed.
fn infer_closures_in(
    module: &Module,
    body: &[py::Stmt],
    files: &[u32],
    vars: &HashMap<String, Ty>,
) -> bool {
    let mut changed = false;
    for (k, def) in closures_directly_in(module, body, files) {
        changed |= infer_closure(module, k, &def, vars);
    }
    changed
}

/// Type closure `k`'s body with `vars` as the scope holding it: what it
/// returns, what it assigns to its own parameters, and the closures
/// inside it in turn. Only the names the body reads from outside are
/// seeded, so a name it binds itself is its own. Returns whether its
/// signature changed.
fn infer_closure(module: &Module, k: u16, def: &ClosureDef, vars: &HashMap<String, Ty>) -> bool {
    let sig = module.closures.borrow()[k as usize].sig.clone();
    let seeds_for = |scope: &crate::scope::Scope| -> HashMap<String, Ty> {
        vars.iter()
            .filter(|(n, _)| scope.free.contains(*n))
            .map(|(n, t)| (n.clone(), *t))
            .collect()
    };
    let mut changed = false;
    let (ret, param_writes) = match def {
        ClosureDef::Lambda(l) => {
            let seeds = seeds_for(&crate::scope::Scope::of_lambda(l));
            let params: HashMap<String, Ty> = sig.params.iter().cloned().collect();
            let ret = Typer {
                module,
                vars: &params,
                outer: &seeds,
            }
            .expr(&l.body);
            let mut inner = seeds;
            inner.extend(params);
            for (j, nested) in closures_directly_in_expr(module, &l.body) {
                changed |= infer_closure(module, j, &nested, &inner);
            }
            (ret, HashMap::new())
        }
        ClosureDef::Def(f) => {
            let seeds = seeds_for(&crate::scope::Scope::of_function(f));
            let locals = infer_locals_with(module, &sig, &f.body, &seeds, &[], false);
            let ret = if locals.returns { locals.ret } else { Ty::None };
            let mut inner = seeds;
            inner.extend(locals.vars);
            changed |= infer_closures_in(module, &f.body, &[], &inner);
            (ret, locals.param_writes)
        }
    };
    let mut closures = module.closures.borrow_mut();
    let c = &mut closures[k as usize];
    if c.ret_inferred && c.sig.ret != ret {
        c.sig.ret = ret;
        changed = true;
    }
    for (i, (name, ty)) in c.sig.params.iter_mut().enumerate() {
        if let (true, Some(written)) = (c.inferred[i], param_writes.get(name)) {
            let joined = ty.join(*written);
            if joined != *ty {
                *ty = joined;
                changed = true;
            }
        }
    }
    changed
}

/// A closure's definition, for inferring its body.
#[derive(Debug, Clone)]
pub(crate) enum ClosureDef {
    Lambda(py::ExprLambda),
    Def(py::StmtFunctionDef),
}

impl Module {
    /// The closure a lambda or nested def is, from the function holding
    /// it and the start of its range.
    pub(crate) fn closure_at(&self, file: u32, start: u32) -> Option<u16> {
        self.closure_index.get(&(file, start)).copied()
    }

    /// What calling closure `k` returns.
    pub(crate) fn closure_ret(&self, k: u16) -> Ty {
        self.closures
            .borrow()
            .get(k as usize)
            .map(|c| c.sig.ret)
            .unwrap_or(Ty::Object)
    }

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
    /// What the body assigns to its own parameters.
    pub(crate) param_writes: HashMap<String, Ty>,
    /// Whether the body has a `return`; without one it returns None.
    pub(crate) returns: bool,
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

/// What [`infer_module`] decided.
pub(crate) struct Inferred {
    pub(crate) funcs: HashMap<String, Sig>,
    pub(crate) classes: Vec<ClassInfo>,
    pub(crate) closures: Vec<ClosureInfo>,
    /// The entry body's locals, against the signatures above.
    pub(crate) entry: Locals,
}

/// The items every call of which is in view: module functions never
/// mentioned but as a callee, and constructors never reached but
/// through their class. An unannotated parameter of one is typed by
/// what is passed to it, everywhere it is passed; any other stays
/// dynamic, as Python has it.
pub(crate) fn closed_items(
    body: &[py::Stmt],
    items: &[Item<'_>],
) -> std::collections::HashSet<String> {
    use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
    #[derive(Default)]
    struct Mentions {
        /// Names used as anything but a callee: values, assignment
        /// targets, deletions.
        names: std::collections::HashSet<String>,
        /// `x.__init__` on anything but `super()`: a constructor
        /// reached without its class.
        init: bool,
    }
    impl<'a> Visitor<'a> for Mentions {
        fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
            if let py::Stmt::FunctionDef(f) = stmt {
                // A decorator receives the function as a value.
                if !f.decorator_list.is_empty() {
                    self.names.insert(f.name.to_string());
                }
            }
            walk_stmt(self, stmt);
        }
        fn visit_expr(&mut self, expr: &'a py::Expr) {
            match expr {
                py::Expr::Call(c) => {
                    match &*c.func {
                        py::Expr::Name(_) => {}
                        other => self.visit_expr(other),
                    }
                    for a in &c.arguments.args {
                        self.visit_expr(a);
                    }
                    for k in &c.arguments.keywords {
                        self.visit_expr(&k.value);
                    }
                }
                py::Expr::Name(n) => {
                    self.names.insert(n.id.to_string());
                }
                py::Expr::Attribute(a) if a.attr.as_str() == "__init__" => {
                    if !is_super_call(&a.value) {
                        self.init = true;
                    }
                    walk_expr(self, expr);
                }
                _ => walk_expr(self, expr),
            }
        }
    }
    let mut seen = Mentions::default();
    for s in body {
        seen.visit_stmt(s);
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for item in items {
        *counts.entry(item.name.as_str()).or_default() += 1;
    }
    items
        .iter()
        .filter(|item| counts[item.name.as_str()] == 1)
        .filter(|item| item.def.parameters.vararg.is_none() && item.def.parameters.kwarg.is_none())
        .filter(|item| match item.class {
            None => !seen.names.contains(&item.name),
            // The exception classes the library and the lowering raise
            // by name are constructed out of view.
            Some(_) => {
                item.def.name.as_str() == "__init__"
                    && !seen.init
                    && !crate::prelude::EXCEPTION_KINDS
                        .iter()
                        .any(|kind| item.name == method_fn(kind, "__init__"))
            }
        })
        .map(|item| item.name.clone())
        .collect()
}

/// Infer the module's signatures to a fixed point. Class layouts are
/// taken from `known` and refined from what methods assign to `self`;
/// the unannotated parameters of `known.closed` functions are the join
/// of what every call passes and what the body assigns to them.
/// Nothing undecided is settled before the end, so a value not yet
/// typed contributes nothing to a join rather than making it dynamic.
pub(crate) fn infer_module(
    known: &Module,
    items: &[Item<'_>],
    entry: &[py::Stmt],
    entry_files: &[u32],
) -> Inferred {
    let mut closures = known.closures.borrow().clone();
    for c in &mut closures {
        c.reset();
    }
    let mut module = Module {
        funcs: HashMap::new(),
        globals: known.globals.clone(),
        list_type: known.list_type,
        classes: known.classes.clone(),
        class_index: known.class_index.clone(),
        closed: known.closed.clone(),
        closures: std::cell::RefCell::new(closures),
        closure_index: known.closure_index.clone(),
        files: known.files.clone(),
        ..Default::default()
    };
    // Which parameters of each function are inferred, by position.
    let mut inferring: HashMap<String, Vec<bool>> = HashMap::new();
    for item in items {
        let mut sig = declared_sig_in(&module.class_index, item.def, item.class);
        if known.closed.contains(&item.name) {
            let flags: Vec<bool> = item
                .def
                .parameters
                .iter_non_variadic_params()
                .enumerate()
                .map(|(i, p)| p.parameter.annotation.is_none() && !(i == 0 && item.class.is_some()))
                .collect();
            for (flag, (_, ty)) in flags.iter().zip(sig.params.iter_mut()) {
                if *flag {
                    *ty = Ty::Unknown;
                }
            }
            inferring.insert(item.name.clone(), flags);
        }
        module.funcs.insert(item.name.clone(), sig);
    }
    let entry_sig = Sig {
        params: Vec::new(),
        ret: Ty::None,
        defaults: Vec::new(),
    };
    let mut entry_locals = Locals::default();
    for _ in 0..32 {
        let mut changed = false;
        let mut passed: Vec<(Target, usize, Ty)> = Vec::new();
        let mut escaped: Vec<u16> = Vec::new();
        for item in items {
            let sig = module.funcs[&item.name].clone();
            let file = module.file_of(item.module.as_deref());
            let locals = in_file(file, || {
                let locals = infer_locals_open(&module, &sig, &item.def.body, &[]);
                changed |= infer_closures_in(&module, &item.def.body, &[], &locals.vars);
                locals
            });
            if item.def.returns.is_none() && sig.ret != Ty::Gen {
                let ret = if locals.returns { locals.ret } else { Ty::None };
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
            if let Some(flags) = inferring.get(&item.name) {
                for (i, (name, _)) in sig.params.iter().enumerate() {
                    if let (true, Some(ty)) = (flags[i], locals.param_writes.get(name)) {
                        passed.push((Target::Item(item.name.clone()), i, *ty));
                    }
                }
            }
            in_file(file, || {
                Calls {
                    module: &module,
                    vars: &locals.vars,
                    class: item.class,
                    opaque: false,
                    passed: &mut passed,
                    allow_closure: false,
                    escaped: &mut escaped,
                    files: &[],
                    no_outer: HashMap::new(),
                }
                .stmts(&item.def.body)
            });
        }
        entry_locals = infer_locals_open(&module, &entry_sig, entry, entry_files);
        changed |= infer_closures_in(&module, entry, entry_files, &entry_locals.vars);
        Calls {
            module: &module,
            vars: &entry_locals.vars,
            class: None,
            opaque: false,
            passed: &mut passed,
            allow_closure: false,
            escaped: &mut escaped,
            files: entry_files,
            no_outer: HashMap::new(),
        }
        .stmts(entry);
        for (callee, index, ty) in passed {
            let slot = match &callee {
                Target::Item(name) => {
                    let Some(flags) = inferring.get(name) else {
                        continue;
                    };
                    if !flags[index] {
                        continue;
                    }
                    let slot = &mut module.funcs.get_mut(name).unwrap().params[index].1;
                    let joined = slot.join(ty);
                    if joined != *slot {
                        *slot = joined;
                        changed = true;
                    }
                    continue;
                }
                Target::Closure(k) => *k,
            };
            let mut closures = module.closures.borrow_mut();
            let c = &mut closures[slot as usize];
            if !c.inferred[index] {
                continue;
            }
            // A parameter only ever passed None is dynamic: the IR has no
            // value of that type to pass.
            let joined = match c.sig.params[index].1.join(ty) {
                Ty::None => Ty::Object,
                t => t,
            };
            if joined != c.sig.params[index].1 {
                c.sig.params[index].1 = joined;
                changed = true;
            }
        }
        // An escaped closure may be called from code out of view, so
        // its inferred parameters are dynamic; the body is retyped
        // against that on the next round.
        for k in escaped {
            let mut closures = module.closures.borrow_mut();
            let c = &mut closures[k as usize];
            if !c.escapes {
                c.escapes = true;
                changed = true;
            }
            for (flag, (_, ty)) in c.inferred.iter().zip(c.sig.params.iter_mut()) {
                if *flag && *ty != Ty::Object {
                    *ty = Ty::Object;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Whatever recursion left undecided is dynamic. So is a parameter
    // only ever passed None: the IR has no value of that type to pass.
    for (name, sig) in module.funcs.iter_mut() {
        if sig.ret == Ty::Unknown {
            sig.ret = Ty::Object;
        }
        if let Some(flags) = inferring.get(name) {
            for (flag, (_, ty)) in flags.iter().zip(sig.params.iter_mut()) {
                if *flag && matches!(ty, Ty::Unknown | Ty::None) {
                    *ty = Ty::Object;
                }
            }
        }
    }
    for class in &mut module.classes {
        for (_, ty) in &mut class.fields {
            if *ty == Ty::Unknown {
                *ty = Ty::Object;
            }
        }
    }
    let mut closures = module.closures.take();
    for c in &mut closures {
        if c.sig.ret == Ty::Unknown {
            c.sig.ret = Ty::Object;
        }
        for (flag, (_, ty)) in c.inferred.iter().zip(c.sig.params.iter_mut()) {
            if *flag && *ty == Ty::Unknown {
                *ty = Ty::Object;
            }
        }
    }
    normalize_layouts(&mut module.classes);
    settle(&mut entry_locals);
    Inferred {
        funcs: module.funcs,
        classes: module.classes,
        closures,
        entry: entry_locals,
    }
}

/// The calls a body makes to the module's own functions and
/// constructors, and what each passes to which parameter.
struct Calls<'a> {
    module: &'a Module,
    vars: &'a HashMap<String, Ty>,
    /// The class whose method this body is, for `super()`.
    class: Option<usize>,
    /// Inside a nested function or lambda, whose variables are not in
    /// `vars`: every argument counts as dynamic.
    opaque: bool,
    passed: &'a mut Vec<(Target, usize, Ty)>,
    /// Whether the expression about to be visited may be a closure
    /// value without that counting as the closure escaping: a callee, a
    /// value bound to a name, a returned value.
    allow_closure: bool,
    /// Closures a value of which reached anywhere else.
    escaped: &'a mut Vec<u16>,
    /// The file of each top-level statement, where they span several.
    files: &'a [u32],
    /// A body typed here has no enclosing scope of its own.
    no_outer: HashMap<String, Ty>,
}

/// What a call reaches: a function of the module (or a constructor), or
/// the closure at an index of the module's table.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    Item(String),
    Closure(u16),
}

impl Calls<'_> {
    fn stmts(&mut self, stmts: &[py::Stmt]) {
        use ruff_python_ast::visitor::Visitor;
        for (i, s) in stmts.iter().enumerate() {
            match self.files.get(i) {
                Some(&file) => in_file(file, || self.visit_stmt(s)),
                None => self.visit_stmt(s),
            }
        }
    }

    /// Walk a nested scope, whose variables are not in `vars`.
    fn nested(&mut self, walk: impl FnOnce(&mut Self)) {
        let was = self.opaque;
        self.opaque = true;
        walk(self);
        self.opaque = was;
    }

    /// The function a call reaches and the index of its first parameter
    /// the arguments fill: a constructor's `self` is not passed.
    fn callee(&self, func: &py::Expr) -> Option<(Target, usize)> {
        // A name bound here is a value, whatever else it spells.
        let bound_here = |name: &str| self.vars.contains_key(name);
        match func {
            py::Expr::Name(n) if !bound_here(n.id.as_str()) => {
                let name = n.id.as_str();
                if let Some(&k) = self.module.class_index.get(name) {
                    let (_, init) = self.module.method_sig(k, "__init__")?;
                    return Some((Target::Item(init), 1));
                }
                if self.module.funcs.contains_key(name) {
                    return Some((Target::Item(name.to_string()), 0));
                }
                self.closure_of(func).map(|k| (Target::Closure(k), 0))
            }
            py::Expr::Attribute(a) if a.attr.as_str() == "__init__" && is_super_call(&a.value) => {
                let base = self.module.classes[self.class?].base?;
                let (_, init) = self.module.method_sig(base, "__init__")?;
                Some((Target::Item(init), 1))
            }
            other => self.closure_of(other).map(|k| (Target::Closure(k), 0)),
        }
    }

    fn arg_ty(&self, e: &py::Expr) -> Ty {
        if self.opaque {
            return Ty::Object;
        }
        self.typer().expr(e)
    }

    fn typer(&self) -> Typer<'_> {
        Typer {
            module: self.module,
            vars: self.vars,
            outer: &self.no_outer,
        }
    }

    /// The closure an expression is a value of, if inference knows. In a
    /// nested scope only the enclosing body's variables are known, which
    /// is what a nested body captures; a name it binds itself is read as
    /// the enclosing one's, which at worst makes a closure dynamic.
    fn closure_of(&self, e: &py::Expr) -> Option<u16> {
        match self.typer().callee_ty(e) {
            Ty::Closure(k) => Some(k),
            _ => None,
        }
    }

    fn escape(&mut self, k: u16) {
        if !self.escaped.contains(&k) {
            self.escaped.push(k);
        }
    }

    /// Record what a call passes to each parameter from `first` on. A
    /// parameter left out takes its default; a call that cannot be
    /// matched to the parameters makes them all dynamic.
    fn record(
        &mut self,
        callee: Target,
        first: usize,
        args: &[py::Expr],
        keywords: &[py::Keyword],
    ) {
        let sig = match &callee {
            Target::Item(name) => self.module.funcs[name].clone(),
            Target::Closure(k) => match self.module.closures.borrow().get(*k as usize) {
                Some(c) => c.sig.clone(),
                None => return,
            },
        };
        let n = sig.params.len();
        let mut given: Vec<Option<Ty>> = vec![None; n];
        let mut matched = !args.iter().any(|a| matches!(a, py::Expr::Starred(_)))
            && keywords.iter().all(|k| k.arg.is_some())
            && first + args.len() <= n;
        if matched {
            for (i, a) in args.iter().enumerate() {
                given[first + i] = Some(self.arg_ty(a));
            }
            for k in keywords {
                let name = k.arg.as_ref().unwrap().as_str();
                match sig.params.iter().position(|(p, _)| p == name) {
                    Some(i) if i >= first && given[i].is_none() => {
                        given[i] = Some(self.arg_ty(&k.value));
                    }
                    _ => matched = false,
                }
            }
        }
        for i in first..n {
            let ty = if !matched {
                Ty::Object
            } else {
                match (given[i], &sig.defaults[i]) {
                    (Some(t), _) => t,
                    (None, Some(d)) => Typer {
                        module: self.module,
                        vars: &HashMap::new(),
                        outer: &HashMap::new(),
                    }
                    .expr(d),
                    // Missing with no default: the call fails before the
                    // function runs.
                    (None, None) => continue,
                }
            };
            self.passed.push((callee.clone(), i, ty));
        }
    }
}

impl<'ast> ruff_python_ast::visitor::Visitor<'ast> for Calls<'_> {
    fn visit_stmt(&mut self, stmt: &'ast py::Stmt) {
        use ruff_python_ast::visitor::walk_stmt;
        match stmt {
            // Defaults are evaluated where the function is defined.
            py::Stmt::FunctionDef(f) => {
                for p in f.parameters.iter_non_variadic_params() {
                    if let Some(d) = &p.default {
                        self.visit_expr(d);
                    }
                }
                self.nested(|c| c.visit_body(&f.body));
            }
            py::Stmt::ClassDef(c) => self.nested(|v| v.visit_body(&c.body)),
            // `raise C`: the class is called with nothing.
            py::Stmt::Raise(r) => {
                if let Some(py::Expr::Name(n)) = r.exc.as_deref() {
                    if let Some(&k) = self.module.class_index.get(n.id.as_str()) {
                        if let Some((_, name)) = self.module.method_sig(k, "__init__") {
                            self.record(Target::Item(name), 1, &[], &[]);
                        }
                    }
                }
                walk_stmt(self, stmt);
            }
            // A closure bound to a name, or returned, is still in view.
            py::Stmt::Assign(a) if a.targets.iter().all(|t| matches!(t, py::Expr::Name(_))) => {
                self.allow_closure = true;
                self.visit_expr(&a.value);
            }
            py::Stmt::AnnAssign(a) if matches!(&*a.target, py::Expr::Name(_)) => {
                if let Some(v) = &a.value {
                    self.allow_closure = true;
                    self.visit_expr(v);
                }
            }
            py::Stmt::Return(r) => {
                if let Some(v) = &r.value {
                    self.allow_closure = true;
                    self.visit_expr(v);
                }
            }
            _ => walk_stmt(self, stmt),
        }
    }

    fn visit_expr(&mut self, expr: &'ast py::Expr) {
        use ruff_python_ast::visitor::walk_expr;
        // A closure value anywhere but a callee, a binding or a return
        // has left the inference's sight.
        let allowed = std::mem::take(&mut self.allow_closure);
        if !allowed {
            if let Some(k) = self.closure_of(expr) {
                self.escape(k);
            }
        }
        match expr {
            py::Expr::Call(c) => {
                if let Some((target, first)) = self.callee(&c.func) {
                    self.record(target, first, &c.arguments.args, &c.arguments.keywords);
                }
                self.allow_closure = true;
                self.visit_expr(&c.func);
                for a in &c.arguments.args {
                    self.visit_expr(a);
                }
                for k in &c.arguments.keywords {
                    self.visit_expr(&k.value);
                }
            }
            py::Expr::Lambda(l) => {
                if let Some(params) = &l.parameters {
                    for p in params.iter_non_variadic_params() {
                        if let Some(d) = &p.default {
                            self.visit_expr(d);
                        }
                    }
                }
                self.nested(|c| {
                    c.allow_closure = true;
                    c.visit_expr(&l.body);
                });
            }
            // A comprehension's own variables are not in `vars`.
            py::Expr::ListComp(_)
            | py::Expr::SetComp(_)
            | py::Expr::DictComp(_)
            | py::Expr::Generator(_) => self.nested(|c| walk_expr(c, expr)),
            _ => walk_expr(self, expr),
        }
    }
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
/// the join of what is assigned to it, iterated until stable. A name
/// nothing decides is dynamic.
pub(crate) fn infer_locals(module: &Module, sig: &Sig, body: &[py::Stmt]) -> Locals {
    infer_locals_seeded(module, sig, body, &HashMap::new())
}

/// [`infer_locals`] for the entry body, whose statements come from
/// several files: `files` names each top-level statement's.
pub(crate) fn infer_locals_entry(
    module: &Module,
    sig: &Sig,
    body: &[py::Stmt],
    files: &[u32],
) -> Locals {
    infer_locals_with(module, sig, body, &HashMap::new(), files, true)
}

/// [`infer_locals`] leaving what is undecided undecided, for the
/// module-wide fixed point: a value another function has yet to type
/// must not settle as dynamic here and poison every join it reaches.
fn infer_locals_open(module: &Module, sig: &Sig, body: &[py::Stmt], files: &[u32]) -> Locals {
    infer_locals_with(module, sig, body, &HashMap::new(), files, false)
}

/// Whatever inference left undecided is dynamic.
fn settle(locals: &mut Locals) {
    for ty in locals.vars.values_mut() {
        if *ty == Ty::Unknown {
            *ty = Ty::Object;
        }
    }
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
    infer_locals_with(module, sig, body, seeds, &[], true)
}

/// `files` names the file of each top-level statement where the body
/// spans several, so a lambda or nested def is looked up in the right
/// one; empty for a body from one file.
fn infer_locals_with(
    module: &Module,
    sig: &Sig,
    body: &[py::Stmt],
    seeds: &HashMap<String, Ty>,
    files: &[u32],
    settled: bool,
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
        for (i, s) in body.iter().enumerate() {
            match files.get(i) {
                Some(&file) => in_file(file, || walker.stmt(s)),
                None => walker.stmt(s),
            }
        }
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
    // A body control can fall off the end of returns None there too.
    if locals.returns && !terminates(body) {
        locals.ret = locals.ret.join(Ty::None);
    }
    if settled {
        settle(&mut locals);
    }
    locals
}

/// Whether control never reaches the end of `stmts`: some statement in
/// the list leaves the function on every path. Anything not shown to
/// leave is taken to fall through.
fn terminates(stmts: &[py::Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        py::Stmt::Return(_) | py::Stmt::Raise(_) => true,
        py::Stmt::If(i) => {
            terminates(&i.body)
                && i.elif_else_clauses.iter().all(|c| terminates(&c.body))
                && i.elif_else_clauses.iter().any(|c| c.test.is_none())
        }
        // `while True` with no break of its own never falls out.
        py::Stmt::While(w) => {
            (is_true_literal(&w.test) && !breaks(&w.body)) || terminates(&w.orelse)
        }
        py::Stmt::For(f) => !breaks(&f.body) && terminates(&f.orelse),
        py::Stmt::Try(t) => {
            terminates(&t.finalbody)
                || (terminates(&t.body) && t.handlers.iter().all(|h| terminates(handler_body(h))))
        }
        py::Stmt::With(w) => terminates(&w.body),
        _ => false,
    })
}

fn is_true_literal(e: &py::Expr) -> bool {
    match e {
        py::Expr::BooleanLiteral(b) => b.value,
        py::Expr::NumberLiteral(n) => match &n.value {
            py::Number::Int(i) => i.as_i64().is_some_and(|v| v != 0),
            _ => false,
        },
        _ => false,
    }
}

fn handler_body(h: &py::ExceptHandler) -> &[py::Stmt] {
    match h {
        py::ExceptHandler::ExceptHandler(h) => &h.body,
    }
}

/// Whether `stmts` has a `break` leaving the loop they are the body
/// of: not one inside a nested loop, which leaves that loop.
fn breaks(stmts: &[py::Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        py::Stmt::Break(_) => true,
        py::Stmt::If(i) => breaks(&i.body) || i.elif_else_clauses.iter().any(|c| breaks(&c.body)),
        py::Stmt::Try(t) => {
            breaks(&t.body)
                || breaks(&t.orelse)
                || breaks(&t.finalbody)
                || t.handlers.iter().any(|h| breaks(handler_body(h)))
        }
        py::Stmt::With(w) => breaks(&w.body),
        // A loop's `else` runs in the enclosing loop's body.
        py::Stmt::While(w) => breaks(&w.orelse),
        py::Stmt::For(f) => breaks(&f.orelse),
        _ => false,
    })
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
        // One still being inferred takes the assignment into its type.
        if let Some((_, declared)) = self.params.iter().find(|(n, _)| n == name) {
            let written = self
                .locals
                .param_writes
                .get(name)
                .copied()
                .unwrap_or(Ty::Unknown)
                .join(ty);
            self.locals.param_writes.insert(name.to_string(), written);
            if !matches!(*declared, Ty::Object | Ty::Unknown)
                && ty != *declared
                && ty != Ty::Unknown
            {
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
                self.locals.returns = true;
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
            py::Stmt::FunctionDef(f) => {
                let ty = self
                    .module
                    .closure_at(crate::lower::current_file(), f.range.start().to_u32())
                    .map(Ty::Closure)
                    .unwrap_or(Ty::Object);
                self.assign(f.name.as_str(), ty)
            }
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
                    Ty::Unknown => Ty::Unknown,
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
            // A lambda is the closure it defines, where inference knows
            // it; otherwise a function value like any other.
            py::Expr::Lambda(l) => self
                .module
                .closure_at(crate::lower::current_file(), l.range.start().to_u32())
                .map(Ty::Closure)
                .unwrap_or(Ty::Object),
            _ => Ty::Object,
        }
    }

    /// The type of a call's callee: a variable's, a lambda's, or any
    /// other expression's. A module function or class named directly is
    /// not a value here.
    pub(crate) fn callee_ty(&self, func: &py::Expr) -> Ty {
        match func {
            py::Expr::Name(n) => {
                let name = n.id.as_str();
                self.vars
                    .get(name)
                    .or_else(|| self.outer.get(name))
                    .or_else(|| self.module.globals.get(name))
                    .copied()
                    .unwrap_or(Ty::Object)
            }
            other => self.expr(other),
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
        // A call through a value whose function is known returns what
        // that function returns.
        if let Ty::Closure(k) = self.callee_ty(&c.func) {
            return self.module.closure_ret(k);
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
            // A method on a value not yet typed.
            py::Expr::Attribute(a) if self.expr(&a.value) == Ty::Unknown => Ty::Unknown,
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
