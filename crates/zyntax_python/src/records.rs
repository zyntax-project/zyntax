//! Dict literals whose keys are distinct string literals, as records.
//!
//! Each such key list is a shape, and each shape a class `dict$<n>`
//! laid out after the program's classes: a field per key, in literal
//! order, so a literal is one allocation and `rec["k"]` is a field
//! access. A record is only ever used where its class stands in for a
//! dict exactly: built by its literal, read and written by literal keys
//! of its shape, and passed around typed. Anything else the lowering
//! meets a record in (a box, a dynamic key, a method, a comparison, a
//! builtin) demotes the shape, and the program is typed again with its
//! literals as dicts. A boxed record never exists, so nothing dynamic
//! can ask one to grow.

use crate::types::{self, ClassInfo, Elem, Ty};
use ruff_python_ast as py;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cell::{Cell, RefCell};

#[derive(Clone, Debug)]
pub(crate) struct RecordInfo {
    /// The keys in literal order; key `i` is field `k$i`.
    pub(crate) keys: Vec<String>,
    /// The record's class.
    pub(crate) class: usize,
}

thread_local! {
    static RECORDS: RefCell<Vec<RecordInfo>> = const { RefCell::new(Vec::new()) };
    static SHAPES: RefCell<HashMap<Vec<String>, usize>> = RefCell::new(HashMap::default());
    /// Field writes found while typing literals and literal-key stores,
    /// drained by the inference rounds.
    static WRITES: RefCell<Vec<(usize, String, Ty)>> = const { RefCell::new(Vec::new()) };
    /// Records the lowering found used as something other than a record.
    static DEMOTED: RefCell<HashSet<usize>> = RefCell::new(HashSet::default());
    /// Whether the lowering in progress is the program's, whose uses of
    /// a record decide its demotion; generated hooks' are not.
    static WATCH: Cell<bool> = const { Cell::new(false) };
    /// Typing that happens before a body's locals are bound notes no
    /// field writes while this is above zero.
    static QUIET: Cell<u32> = const { Cell::new(0) };
    /// Whether the program reads its globals as a dict (`globals()`,
    /// `exec`): every record a global holds then reaches a box.
    static GLOBALS_ESCAPE: Cell<bool> = const { Cell::new(false) };
}

/// Field writes are not noted while one of these is alive.
pub(crate) struct Quiet;

impl Quiet {
    pub(crate) fn new() -> Self {
        QUIET.with(|q| q.set(q.get() + 1));
        Quiet
    }
}

impl Drop for Quiet {
    fn drop(&mut self) {
        QUIET.with(|q| q.set(q.get() - 1));
    }
}

/// The key list of a literal that can be a record: every key a string
/// literal, none twice, and at least one.
pub(crate) fn literal_keys(d: &py::ExprDict) -> Option<Vec<String>> {
    if d.items.is_empty() {
        return None;
    }
    let mut keys = Vec::with_capacity(d.items.len());
    for item in &d.items {
        match &item.key {
            Some(py::Expr::StringLiteral(s)) => {
                let k = s.value.to_str().to_string();
                if keys.contains(&k) {
                    return None;
                }
                keys.push(k);
            }
            _ => return None,
        }
    }
    Some(keys)
}

/// Every record shape of the program not in `demoted`, as classes after
/// the `first` classes the program declares.
pub(crate) fn collect(
    body: &[py::Stmt],
    demoted: &HashSet<Vec<String>>,
    first: usize,
) -> Vec<ClassInfo> {
    use ruff_python_ast::visitor::{Visitor, walk_expr};
    struct Find<'a> {
        demoted: &'a HashSet<Vec<String>>,
        shapes: Vec<Vec<String>>,
        index: HashMap<Vec<String>, usize>,
    }
    impl<'a> Visitor<'a> for Find<'_> {
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Dict(d) = e
                && let Some(keys) = literal_keys(d)
                && !self.demoted.contains(&keys)
                && !self.index.contains_key(&keys)
            {
                self.index.insert(keys.clone(), self.shapes.len());
                self.shapes.push(keys);
            }
            walk_expr(self, e);
        }
    }
    let mut find = Find {
        demoted,
        shapes: Vec::new(),
        index: HashMap::default(),
    };
    for s in body {
        find.visit_stmt(s);
    }
    let mut classes = Vec::with_capacity(find.shapes.len());
    let mut records = Vec::with_capacity(find.shapes.len());
    for (n, keys) in find.shapes.into_iter().enumerate() {
        let mut fields = vec![("$class".to_string(), Ty::Int)];
        fields.extend((0..keys.len()).map(|i| (field_name(i), Ty::Unknown)));
        classes.push(ClassInfo {
            name: format!("dict${n}"),
            base: None,
            descendants: 1,
            fields,
            methods: Vec::new(),
            type_id: None,
            module: None,
        });
        records.push(RecordInfo {
            keys,
            class: first + n,
        });
    }
    RECORDS.with(|r| *r.borrow_mut() = records);
    SHAPES.with(|s| *s.borrow_mut() = find.index);
    WRITES.with(|w| w.borrow_mut().clear());
    DEMOTED.with(|d| d.borrow_mut().clear());
    WATCH.with(|w| w.set(false));
    GLOBALS_ESCAPE.with(|g| g.set(false));
    classes
}

pub(crate) fn field_name(i: usize) -> String {
    format!("k${i}")
}

/// The class a literal builds, when it is a record.
pub(crate) fn of_literal(d: &py::ExprDict) -> Option<usize> {
    if RECORDS.with(|r| r.borrow().is_empty()) {
        return None;
    }
    let keys = literal_keys(d)?;
    let n = SHAPES.with(|s| s.borrow().get(&keys).copied())?;
    Some(RECORDS.with(|r| r.borrow()[n].class))
}

/// The record class `k` is, by its index among the records.
fn record_of_class(k: usize) -> Option<usize> {
    RECORDS.with(|r| r.borrow().iter().position(|rec| rec.class == k))
}

pub(crate) fn is_record(k: usize) -> bool {
    record_of_class(k).is_some()
}

/// The field `key` names on record class `k`, when `key` is a string
/// literal of its shape.
pub(crate) fn field_of(k: usize, key: &str) -> Option<String> {
    let n = record_of_class(k)?;
    RECORDS.with(|r| {
        r.borrow()[n]
            .keys
            .iter()
            .position(|x| x == key)
            .map(field_name)
    })
}

/// A literal's value or a literal-key store typed field `field` of
/// record class `k` as `ty`.
pub(crate) fn note_write(k: usize, field: String, ty: Ty) {
    if QUIET.with(|q| q.get()) > 0 {
        return;
    }
    WRITES.with(|w| w.borrow_mut().push((k, field, ty)));
}

pub(crate) fn take_writes() -> Vec<(usize, String, Ty)> {
    WRITES.with(|w| std::mem::take(&mut *w.borrow_mut()))
}

pub(crate) fn watch(on: bool) {
    WATCH.with(|w| w.set(on));
}

/// Whether the program has any record shape.
pub(crate) fn any() -> bool {
    RECORDS.with(|r| !r.borrow().is_empty())
}

pub(crate) fn escape_globals() {
    GLOBALS_ESCAPE.with(|g| g.set(true));
}

/// Demote the records the globals hold, when the program reads them as
/// a dict.
pub(crate) fn demote_globals(module: &types::Module) {
    if GLOBALS_ESCAPE.with(|g| g.get()) {
        for ty in module.globals.values() {
            demote_in_inference(module, *ty);
        }
    }
}

/// Demote every record a value of `ty` holds or is: an edge inference
/// found.
pub(crate) fn demote_in_inference(module: &types::Module, ty: Ty) {
    if !any() {
        return;
    }
    let mut seen = HashSet::default();
    reach(module, ty, &mut seen);
}

/// Demote every record a value of `ty` holds or is, when the program's
/// lowering is what met it.
pub(crate) fn demote_reached(module: &types::Module, ty: Ty) {
    if !WATCH.with(|w| w.get()) || RECORDS.with(|r| r.borrow().is_empty()) {
        return;
    }
    let mut seen = HashSet::default();
    reach(module, ty, &mut seen);
}

fn reach(module: &types::Module, ty: Ty, seen: &mut HashSet<Ty>) {
    if !seen.insert(ty) {
        return;
    }
    match ty {
        Ty::Class(k) => {
            if let Some(n) = record_of_class(k as usize) {
                DEMOTED.with(|d| d.borrow_mut().insert(n));
            }
            if let Some(class) = module.classes.get(k as usize) {
                for (_, t) in &class.fields {
                    reach(module, *t, seen);
                }
            }
        }
        Ty::List(e) => match e {
            Elem::Object | Elem::Int | Elem::Float | Elem::Str | Elem::Array(_) => {}
            e => reach(module, e.ty(), seen),
        },
        Ty::Tuple(k) => {
            for t in types::tuple_shape(k) {
                reach(module, t, seen);
            }
        }
        Ty::Dict(k) => {
            let (key, value) = types::dict_shape(k);
            reach(module, key, seen);
            reach(module, value, seen);
        }
        Ty::Set(k) => reach(module, types::set_shape(k).0, seen),
        _ => {}
    }
}

pub(crate) fn any_demoted() -> bool {
    DEMOTED.with(|d| !d.borrow().is_empty())
}

/// The key lists of the records the lowering demoted.
pub(crate) fn demoted() -> Vec<Vec<String>> {
    let found: Vec<usize> = DEMOTED.with(|d| d.borrow().iter().copied().collect());
    RECORDS.with(|r| {
        let r = r.borrow();
        found.into_iter().map(|n| r[n].keys.clone()).collect()
    })
}

/// Every record's key list.
pub(crate) fn all_shapes() -> Vec<Vec<String>> {
    RECORDS.with(|r| r.borrow().iter().map(|rec| rec.keys.clone()).collect())
}
