//! Static types for a dynamically typed program.
//!
//! A variable's type is the join of everything assigned to it, over the
//! whole chunk; a function's parameter types are the join of what its
//! direct calls pass; its result is the join of what it returns. The
//! join of two different types is the dynamic value, so a variable
//! holding an integer everywhere is an `i64` and one holding an
//! integer here and a string there is boxed. Rounds repeat until
//! nothing changes: a recursive function's result depends on itself.

use std::collections::{BTreeSet, HashMap, HashSet};

use full_moon::ast::{self, BinOp, Block, Expression, Prefix, Stmt, Suffix, UnOp, Var};
use indexmap::IndexMap;

use crate::library::stdlib::{BUILTINS, Builtin, Ret};
use crate::scope::{Binding, CHUNK, FuncId, Scopes, VarId};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Ty {
    Nil,
    Bool,
    Int,
    Float,
    /// An integer or a float, whichever it holds: the join of the two,
    /// carried unboxed with a tag.
    Number,
    /// Nil, a boolean, an integer or a float: the join of any of them,
    /// carried unboxed with a tag. Arithmetic on it raises for the
    /// first two, as on the dynamic value.
    Scalar,
    Str,
    /// A table, held by pointer; never nil.
    Table,
    /// This function, as its record, or nil: a closure over whatever
    /// it captured. A call through it is a direct call, once the value
    /// is seen not to be nil.
    Func(FuncId),
    /// A table of this shape, or nil: its fields are slots of the
    /// object, typed, read and written without a lookup.
    Shape(ShapeId),
    /// A dynamic value: a boxed `Any`.
    Any,
    /// Nothing known yet: the bottom of the join.
    #[default]
    Unknown,
}

impl Ty {
    pub fn join(self, other: Ty) -> Ty {
        match (self, other) {
            (Ty::Unknown, t) | (t, Ty::Unknown) => t,
            (a, b) if a == b => a,
            (a, b) if a.is_number() && b.is_number() => Ty::Number,
            (a, b) if a.is_scalar() && b.is_scalar() => Ty::Scalar,
            // A shape or a function holds nil too.
            (Ty::Shape(k), Ty::Nil) | (Ty::Nil, Ty::Shape(k)) => Ty::Shape(k),
            (Ty::Func(f), Ty::Nil) | (Ty::Nil, Ty::Func(f)) => Ty::Func(f),
            _ => Ty::Any,
        }
    }

    /// Whether a value of this type may be nil.
    pub fn may_be_nil(self) -> bool {
        matches!(
            self,
            Ty::Nil | Ty::Scalar | Ty::Any | Ty::Unknown | Ty::Shape(_) | Ty::Func(_)
        )
    }

    /// Carried unboxed as a tagged scalar, or convertible to one.
    pub fn is_scalar(self) -> bool {
        matches!(
            self,
            Ty::Nil | Ty::Bool | Ty::Int | Ty::Float | Ty::Number | Ty::Scalar
        )
    }

    /// An integer, a float, or one or the other: arithmetic on it
    /// needs no box.
    pub fn is_number(self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Number)
    }

    /// Whether a value of this type is always true in a condition.
    pub fn always_truthy(self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Number | Ty::Str | Ty::Table)
    }

    /// What is known once inference has settled: a variable nothing
    /// assigned holds nil.
    pub fn settled(self) -> Ty {
        match self {
            Ty::Unknown => Ty::Nil,
            t => t,
        }
    }
}

/// What a function returns: a fixed number of values, each typed, or
/// a dynamic value that may hold several; or nothing known yet, of a
/// callee no round has typed, which decides nothing until it has.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Returns {
    Fixed(Vec<Ty>),
    Dynamic,
    Unknown,
}

impl Returns {
    /// The type of the call in single-value position.
    pub fn first(&self) -> Ty {
        match self {
            Returns::Fixed(v) => v.first().copied().unwrap_or(Ty::Nil),
            Returns::Dynamic => Ty::Any,
            Returns::Unknown => Ty::Unknown,
        }
    }

    pub fn settled(self) -> Returns {
        match self {
            Returns::Fixed(v) => Returns::Fixed(v.into_iter().map(Ty::settled).collect()),
            Returns::Dynamic | Returns::Unknown => Returns::Dynamic,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Sig {
    pub params: Vec<Ty>,
    pub returns: Returns,
}

/// A shape: the layout of the tables made by constructors with one
/// set of constant string keys (an empty constructor has a shape of
/// its own per site).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ShapeId(pub u32);

#[derive(Clone, Default, PartialEq, Debug)]
pub struct ShapeInfo {
    /// The fields in slot order: the constructor's keys, then those
    /// that stores add; each with the join of what is stored in it.
    pub fields: IndexMap<String, Ty>,
    /// How many of the fields every constructor sets: the first
    /// `born` are present from birth, the rest may be absent.
    pub born: usize,
    /// The join of what is stored under keys that are not strings:
    /// the constructor's positional values, and stores under numbers.
    /// What a read under such a key yields, when the table has no
    /// metatable to ask instead.
    pub element: Ty,
    /// The shapes of the metatables `setmetatable` gives tables of
    /// this shape.
    pub classes: BTreeSet<ShapeId>,
    /// Whether a table of this shape may get a metatable the types do
    /// not follow.
    pub unknown_meta: bool,
    /// Whether a table of this shape is lost into a dynamic value or
    /// the library, where stores the types do not see may reach it.
    pub escapes: bool,
    /// Whether a store with a key not known at compile time, that may
    /// be a string, may reach a table of this shape: then every field
    /// takes what it stores, and a name that is no field may be held.
    pub dynamic_keys: bool,
    /// The join of what stores under keys not known at compile time
    /// put in a table of this shape: what a name that is no field may
    /// hold, when `dynamic_keys`.
    pub dyn_value: Ty,
    /// Reads out of a table of this shape whose results the types do
    /// not follow: what they may yield escapes.
    pub untyped_reads: Reads,
    /// The line the first constructor is on, for naming.
    pub line: usize,
}

/// Reads out of a table that hand what they find to code the types do
/// not follow: under keys not known at compile time, or by a library
/// function that walks the table.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Reads {
    /// Under a key that may be a string: any field, or any name.
    pub names: bool,
    /// Under a key that may be a number: any element.
    pub elements: bool,
    /// Going on through `__index` where the table lacks the key.
    pub through_meta: bool,
}

impl Reads {
    /// Reads under a key of type `key_ty`.
    pub fn under(key_ty: Ty, through_meta: bool) -> Reads {
        Reads {
            names: key_may_be_string(key_ty),
            elements: key_may_be_index(key_ty),
            through_meta,
        }
    }

    pub fn union(self, other: Reads) -> Reads {
        Reads {
            names: self.names || other.names,
            elements: self.elements || other.elements,
            through_meta: self.through_meta || other.through_meta,
        }
    }
}

impl ShapeInfo {
    pub fn field(&self, name: &str) -> Option<(usize, Ty)> {
        self.fields.get_full(name).map(|(i, _, t)| (i, *t))
    }

    /// Whether a table of this shape always holds the field: set by
    /// every constructor and never assigned something that may be nil.
    pub fn always_present(&self, name: &str) -> bool {
        match self.field(name) {
            Some((i, ty)) => i < self.born && !ty.may_be_nil(),
            None => false,
        }
    }
}

/// What the types know of a shape's metamethod for an event.
#[derive(Clone, PartialEq, Debug)]
pub enum MetaTargets {
    /// Some metatable's handler is not known: the runtime decides.
    None,
    /// Every metatable has a handler, but one is not typed yet.
    Unsettled,
    /// The handler of each class.
    Funcs(Vec<(ShapeId, FuncId)>),
}

/// The metamethod an arithmetic operator dispatches to, when an
/// operand is a table.
pub fn arith_event(op: &BinOp) -> Option<&'static str> {
    Some(match op {
        BinOp::Plus(_) => "__add",
        BinOp::Minus(_) => "__sub",
        BinOp::Star(_) => "__mul",
        BinOp::Slash(_) => "__div",
        BinOp::Percent(_) => "__mod",
        BinOp::Caret(_) => "__pow",
        BinOp::DoubleSlash(_) => "__idiv",
        _ => return None,
    })
}

/// The metamethods the typed dispatch handles: a table operand's
/// handler is called directly when the types know it, so the runtime
/// never reaches it with values of its own.
pub fn typed_event(event: &str) -> bool {
    matches!(
        event,
        "__add" | "__sub" | "__mul" | "__div" | "__mod" | "__pow" | "__idiv" | "__unm"
    )
}

/// Whether a value of this type can carry the handler for an
/// arithmetic event of its own: a table can, a string does (the string
/// metatable's, which defers to the other operand's), a number, a
/// boolean or nil cannot.
fn may_handle_arith(ty: Ty) -> bool {
    !matches!(
        ty,
        Ty::Int | Ty::Float | Ty::Number | Ty::Scalar | Ty::Bool | Ty::Nil | Ty::Str
    )
}

/// One step of a field lookup through metatables, as the runtime
/// takes it from the table in hand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hop {
    /// To the table's metatable, a table of this shape.
    Meta(ShapeId),
    /// The table, of this shape, lacks the field: the lookup goes on.
    Absent(ShapeId),
    /// Through the `__index` field of the table, of this shape, which
    /// holds a table.
    Index(ShapeId),
}

/// Where a lookup of a name may end: the field of a table of shape
/// `shape`, reached by `hops`, holding `ty`; `sure` when the field is
/// always there.
#[derive(Clone, PartialEq, Debug)]
pub struct Lookup {
    pub hops: Vec<Hop>,
    pub shape: ShapeId,
    pub ty: Ty,
    pub sure: bool,
}

#[derive(Clone, Default, PartialEq, Debug)]
pub struct Inferred {
    pub funcs: HashMap<FuncId, Sig>,
    pub vars: HashMap<VarId, Ty>,
    pub globals: HashMap<String, Ty>,
    /// Functions that may be called through a value the types do not
    /// follow: one lost into a dynamic value, or a table, or passed to
    /// the library. Their parameters take anything. Only ever grows.
    pub escaping: HashSet<FuncId>,
    /// The shapes, by id.
    pub shapes: Vec<ShapeInfo>,
    /// The shape of the constructors with these keys, sorted.
    pub shape_by_keys: HashMap<Vec<String>, ShapeId>,
    /// The shape of an empty constructor, by the byte offset of its
    /// brace.
    pub shape_by_site: HashMap<usize, ShapeId>,
    /// Constant-key stores through a receiver the types do not know:
    /// the key, and the join of what is stored. Such a store may reach
    /// any escaping shape with the field.
    pub blind_stores: HashMap<String, Ty>,
    /// Whether a store with a key not known at compile time, that may
    /// be a string, reaches a receiver the types do not know.
    pub blind_dynamic_stores: bool,
    /// The join of what is stored under keys that are not strings
    /// through receivers the types do not know: it may land in the
    /// array part of any escaping shape.
    pub blind_element: Ty,
    /// Whether `setmetatable` is applied to a receiver the types do
    /// not know: any escaping shape may get any metatable.
    pub blind_setmetatable: bool,
}

impl Inferred {
    pub fn var(&self, v: VarId) -> Ty {
        self.vars.get(&v).copied().unwrap_or(Ty::Unknown)
    }
    pub fn global(&self, name: &str) -> Ty {
        self.globals.get(name).copied().unwrap_or(Ty::Unknown)
    }
    pub fn sig(&self, f: FuncId) -> Option<&Sig> {
        self.funcs.get(&f)
    }
    pub fn shape(&self, k: ShapeId) -> &ShapeInfo {
        &self.shapes[k.0 as usize]
    }

    /// Whether a table of shape `k` may hold `name` other than in a
    /// field of its shape: stored under a key not known at compile
    /// time, or through a receiver the types do not know.
    pub fn may_hold_unnamed(&self, k: ShapeId, name: &str) -> bool {
        let info = self.shape(k);
        info.field(name).is_none()
            && (info.dynamic_keys || (info.escapes && self.blind_stores.contains_key(name)))
    }

    /// What a table of shape `k` itself holds under `name` when that
    /// is no field of its shape: nil, or what such stores put there.
    fn unnamed_ty(&self, k: ShapeId, name: &str) -> Ty {
        let info = self.shape(k);
        let mut ty = Ty::Nil;
        // A store under a field's name lands in its slot.
        if info.field(name).is_some() {
            return ty;
        }
        if info.dynamic_keys {
            ty = join_read(ty, info.dyn_value);
        }
        if info.escapes
            && let Some(stored) = self.blind_stores.get(name)
        {
            ty = join_read(ty, *stored);
        }
        ty
    }

    /// Whether `#t` on a table of shape `k` is the length of its array
    /// part: no metatable it may have holds a `__len`.
    pub fn plain_len(&self, k: ShapeId) -> bool {
        let info = self.shape(k);
        !info.unknown_meta
            && info.classes.iter().all(|c| {
                self.shape(*c).field("__len").is_none() && !self.may_hold_unnamed(*c, "__len")
            })
    }

    /// The functions `name` may resolve to for a table whose metatable
    /// has shape `k`, looked up as Lua does when the table itself
    /// lacks it: in the metatable's `__index` table, then in that
    /// table's own metatable's `__index`, and so on. Every function
    /// along the chain is one of them, since a field set later than
    /// the call lets the lookup go on past it; the call site tells
    /// them apart by the value it finds. `None` when a field on the
    /// way is not a function, or the chain leads where the types do
    /// not follow.
    pub fn resolve_method(&self, k: ShapeId, name: &str) -> Option<Vec<Ty>> {
        let mut table = match self.shape(k).field("__index") {
            Some((_, Ty::Shape(t))) => t,
            Some((_, Ty::Unknown)) => return Some(vec![Ty::Unknown]),
            _ => return None,
        };
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        loop {
            if !seen.insert(table) {
                break;
            }
            let info = self.shape(table);
            match info.field(name) {
                Some((_, ty @ (Ty::Func(_) | Ty::Unknown))) => {
                    out.push(ty);
                    if info.always_present(name) {
                        break;
                    }
                }
                Some(_) => return None,
                None if self.may_hold_unnamed(table, name) => return None,
                None => {}
            }
            // Behind this table: nothing, when it has no metatable.
            if !info.unknown_meta && info.classes.is_empty() {
                break;
            }
            table = self.index_behind(table)?;
        }
        Some(out)
    }

    /// Where reading `name` from a table of shape `k` may end, as the
    /// runtime looks it up: the table's own field, then, when that is
    /// absent, its metatable's `__index` table and the tables behind
    /// it. None when the chain is not one the types follow to its end
    /// (a metatable they do not know, an `__index` that is not a
    /// table, a class with several metatables); an empty list when
    /// the name is nowhere along it.
    pub fn lookups(&self, k: ShapeId, name: &str) -> Option<Vec<Lookup>> {
        let mut out = Vec::new();
        let mut hops = Vec::new();
        let info = self.shape(k);
        if let Some((_, ty)) = info.field(name) {
            let sure = info.always_present(name);
            out.push(Lookup {
                hops: hops.clone(),
                shape: k,
                ty,
                sure,
            });
            if sure {
                return Some(out);
            }
            hops.push(Hop::Absent(k));
        }
        if info.unknown_meta || self.may_hold_unnamed(k, name) {
            return None;
        }
        for class in &info.classes {
            let mut hops = hops.clone();
            hops.push(Hop::Meta(*class));
            self.lookups_behind(*class, name, hops, &mut out)?;
        }
        Some(out)
    }

    /// The lookups of `name` that go through the `__index` of a
    /// metatable of shape `class`.
    fn lookups_behind(
        &self,
        class: ShapeId,
        name: &str,
        mut hops: Vec<Hop>,
        out: &mut Vec<Lookup>,
    ) -> Option<()> {
        let mut table = match self.shape(class).field("__index") {
            Some((_, Ty::Shape(t))) => t,
            // A metatable without `__index` answers nothing.
            None if !self.may_hold_unnamed(class, "__index") => return Some(()),
            _ => return None,
        };
        hops.push(Hop::Index(class));
        let mut seen = BTreeSet::new();
        loop {
            if !seen.insert(table) {
                return None;
            }
            let info = self.shape(table);
            if let Some((_, ty)) = info.field(name) {
                let sure = info.always_present(name);
                out.push(Lookup {
                    hops: hops.clone(),
                    shape: table,
                    ty,
                    sure,
                });
                if sure {
                    return Some(());
                }
                hops.push(Hop::Absent(table));
            }
            if self.may_hold_unnamed(table, name) {
                return None;
            }
            if !info.unknown_meta && info.classes.is_empty() {
                return Some(());
            }
            if info.unknown_meta || info.classes.len() != 1 {
                return None;
            }
            let behind = *info.classes.iter().next()?;
            hops.push(Hop::Meta(behind));
            table = match self.shape(behind).field("__index") {
                Some((_, Ty::Shape(next))) => next,
                _ => return None,
            };
            hops.push(Hop::Index(behind));
        }
    }

    /// What reading a table of shape `k` under a key of type `key_ty`
    /// yields: its element type, or nil, when the key cannot be a
    /// string and the table has no metatable to ask; else anything.
    pub fn element_read_ty(&self, k: ShapeId, key_ty: Ty) -> Ty {
        let info = self.shape(k);
        if !key_is_number(key_ty) || info.unknown_meta || !info.classes.is_empty() {
            return Ty::Any;
        }
        join_read(info.element, Ty::Nil)
    }

    /// The element type a `for` over `ipairs(t)` binds, for `t` of type
    /// `ty`: the shape's, when its reads are typed.
    pub fn ipairs_value_ty(&self, ty: Ty) -> Ty {
        match ty {
            Ty::Shape(k) if self.element_read_ty(k, Ty::Int) != Ty::Any => self.shape(k).element,
            Ty::Unknown => Ty::Unknown,
            _ => Ty::Any,
        }
    }

    /// The functions the metamethod `event` of a table of shape `k`
    /// may be: one per class, read raw from the metatable as the
    /// runtime reads it. None when the types do not know every
    /// metatable's handler; Unsettled while some class's handler is
    /// not typed yet.
    pub fn metamethod_targets(&self, k: ShapeId, event: &str) -> MetaTargets {
        let info = self.shape(k);
        if info.unknown_meta || info.classes.is_empty() {
            return MetaTargets::None;
        }
        let mut out = Vec::new();
        for class in &info.classes {
            match self.shape(*class).field(event) {
                Some((_, Ty::Func(f))) => out.push((*class, f)),
                Some((_, Ty::Unknown)) => return MetaTargets::Unsettled,
                _ => return MetaTargets::None,
            }
        }
        MetaTargets::Funcs(out)
    }

    /// The table a lookup goes on to when a table of shape `t` lacks a
    /// field: its one class's `__index` table, when the types know it.
    fn index_behind(&self, t: ShapeId) -> Option<ShapeId> {
        let info = self.shape(t);
        if info.unknown_meta || info.classes.len() != 1 {
            return None;
        }
        let class = *info.classes.iter().next()?;
        match self.shape(class).field("__index") {
            Some((_, Ty::Shape(next))) => Some(next),
            _ => None,
        }
    }

    /// What reading field `name` of a table of shape `k` yields when
    /// the field is absent: what the table holds under a name that is
    /// no field, else what its metatables' `__index` chains hold, nil
    /// when it has no metatable, anything when the types do not know
    /// what it has.
    pub fn absent_field_ty(&self, k: ShapeId, name: &str) -> Ty {
        let info = self.shape(k);
        if info.unknown_meta {
            return Ty::Any;
        }
        let mut ty = self.unnamed_ty(k, name);
        for class in &info.classes {
            ty = join_read(ty, self.index_chain_ty(*class, name, 0));
            if ty == Ty::Any {
                break;
            }
        }
        ty
    }

    /// What `name` yields looked up in a table of shape `class` as a
    /// metatable's `__index`: the field when it holds one, else what
    /// the class's own metatable gives, else nil.
    fn index_chain_ty(&self, class: ShapeId, name: &str, depth: usize) -> Ty {
        if depth > 8 {
            return Ty::Any;
        }
        let info = self.shape(class);
        let own = match info.field(name) {
            Some((_, Ty::Unknown)) => return Ty::Unknown,
            Some((_, ty)) if info.always_present(name) => return ty,
            Some((_, ty)) => Some(ty),
            None if self.may_hold_unnamed(class, name) => Some(self.unnamed_ty(class, name)),
            None => None,
        };
        // The field may be absent: what lies behind it.
        let behind = if info.unknown_meta {
            Ty::Any
        } else {
            let mut t = Ty::Nil;
            for c in &info.classes {
                t = join_read(
                    t,
                    match self.shape(*c).field("__index") {
                        Some((_, Ty::Shape(next))) => self.index_chain_ty(next, name, depth + 1),
                        Some((_, Ty::Unknown)) => Ty::Unknown,
                        None => Ty::Nil,
                        Some(_) => Ty::Any,
                    },
                );
            }
            t
        };
        match own {
            Some(own) => join_read(own, behind),
            None => behind,
        }
    }

    /// What reading field `name` of a table of shape `k` yields. A
    /// field nothing has stored in yet decides nothing yet.
    pub fn field_read_ty(&self, k: ShapeId, name: &str) -> Ty {
        let info = self.shape(k);
        match info.field(name) {
            Some((_, Ty::Unknown)) => Ty::Unknown,
            Some((_, ty)) if info.always_present(name) => ty,
            Some((_, ty)) => join_read(ty, self.absent_field_ty(k, name)),
            None => self.absent_field_ty(k, name),
        }
    }
}

/// The builtin a global name is, when the program never assigns it.
pub fn builtin_named(scopes: &Scopes, name: &str) -> Option<&'static Builtin> {
    if scopes.global_writes.contains_key(name) {
        return None;
    }
    BUILTINS.iter().find(|b| b.lib.is_empty() && b.name == name)
}

/// The library function `lib.name` is, when `lib` is the untouched
/// global of that library.
pub fn builtin_member(scopes: &Scopes, lib: &str, name: &str) -> Option<&'static Builtin> {
    if scopes.global_writes.contains_key(lib) {
        return None;
    }
    BUILTINS.iter().find(|b| b.lib == lib && b.name == name)
}

/// The result of a `math` function whose arguments' types decide it:
/// `math.abs` of an integer is an integer, `math.max` of floats a
/// float. `None` when they do not, and the function's declared result
/// stands; unknown when an argument is not known yet.
pub fn math_result(b: &Builtin, args: &[Ty]) -> Option<Ty> {
    if b.lib != "math" {
        return None;
    }
    let numbers = |args: &[Ty]| -> Option<Ty> {
        // All integers, or all floats.
        let first = *args.first()?;
        if !first.is_number() || args.iter().any(|t| *t != first) {
            return None;
        }
        Some(first)
    };
    let result = match (b.name, args) {
        ("abs" | "max" | "min", args) => numbers(args),
        ("floor" | "ceil", [Ty::Int]) => Some(Ty::Int),
        ("fmod", [Ty::Int, Ty::Int]) => Some(Ty::Int),
        ("fmod", [Ty::Int | Ty::Float, Ty::Int | Ty::Float]) => Some(Ty::Float),
        ("floor" | "ceil" | "fmod", _) => None,
        _ => return None,
    };
    if args.contains(&Ty::Unknown) {
        return Some(Ty::Unknown);
    }
    result
}

pub fn ret_ty(r: Ret) -> Ty {
    match r {
        Ret::Unit => Ty::Nil,
        Ret::Bool => Ty::Bool,
        Ret::Int => Ty::Int,
        Ret::Float => Ty::Float,
        Ret::Str => Ty::Str,
        Ret::Any | Ret::Multi => Ty::Any,
        Ret::Table => Ty::Table,
    }
}

/// What a numeral literal is: an integer, or a float when it has a
/// fraction, an exponent, or does not fit.
pub fn numeral_ty(text: &str) -> Ty {
    match crate::host::parse_numeral(text) {
        crate::host::Numeral::Int(_) => Ty::Int,
        _ => Ty::Float,
    }
}

/// Types expressions against what the last round settled.
pub struct Typer<'a> {
    pub scopes: &'a Scopes,
    pub known: &'a Inferred,
}

impl<'a> Typer<'a> {
    fn name_ty(&self, token: &full_moon::tokenizer::TokenReference) -> Ty {
        match self.scopes.binding(token) {
            Some(b) => self.binding_ty(b),
            None => Ty::Any,
        }
    }

    pub fn binding_ty(&self, binding: &Binding) -> Ty {
        match binding {
            Binding::Local(v) | Binding::Upvalue(v) => self.known.var(*v),
            Binding::Global(name) => self.global_ty(name),
            // A free name in the scope of a local `_ENV`: its field.
            Binding::Field(env, name, _) => self.field_ty(self.known.var(*env), name),
        }
    }

    pub fn global_ty(&self, name: &str) -> Ty {
        // An entry of the globals table can be anything.
        if self.scopes.dynamic_globals {
            return Ty::Any;
        }
        if builtin_named(self.scopes, name).is_some() {
            return Ty::Any;
        }
        if !self.scopes.global_writes.contains_key(name) {
            return Ty::Nil;
        }
        let assigned = self.known.global(name);
        if self.scopes.globals_initialized.contains(name) {
            assigned
        } else {
            assigned.join(Ty::Nil)
        }
    }

    /// The function a callee expression is, when its type says.
    pub fn known_callee(&self, prefix: &Prefix) -> Option<FuncId> {
        match self.prefix_ty(prefix) {
            Ty::Func(f) => Some(f),
            _ => None,
        }
    }

    /// The builtin a callee names: `print`, or `string.format` where
    /// `string` is the library's table.
    pub fn builtin_callee(
        &self,
        prefix: &Prefix,
        suffixes: &[&Suffix],
    ) -> Option<&'static Builtin> {
        let Prefix::Name(token) = prefix else {
            return None;
        };
        let Some(Binding::Global(name)) = self.scopes.binding(token) else {
            return None;
        };
        match suffixes {
            [Suffix::Call(ast::Call::AnonymousCall(_))] => builtin_named(self.scopes, name),
            [
                Suffix::Index(ast::Index::Dot { name: member, .. }),
                Suffix::Call(ast::Call::AnonymousCall(_)),
            ] => builtin_member(self.scopes, name, &ident(member)),
            _ => None,
        }
    }

    /// What a call returns. `suffixes` are the callee's suffixes,
    /// ending in the call being typed.
    pub fn call_returns(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Returns {
        let Some(Suffix::Call(last)) = suffixes.last() else {
            return Returns::Dynamic;
        };
        if let Some(b) = self.builtin_callee(prefix, suffixes) {
            if let ast::Call::AnonymousCall(args) = last
                && let Some(t) = self.math_call_ty(b, args)
            {
                return Returns::Fixed(vec![t]);
            }
            if let ast::Call::AnonymousCall(args) = last
                && let Some(r) = self.metatable_call_returns(b, args)
            {
                return r;
            }
            return match b.ret {
                Ret::Multi => Returns::Dynamic,
                r => Returns::Fixed(vec![ret_ty(r)]),
            };
        }
        // The receiver of the last call is the prefix through every
        // suffix before it.
        let receiver = self.suffixed_ty(prefix, &suffixes[..suffixes.len() - 1]);
        match last {
            ast::Call::MethodCall(m) => self.method_returns(receiver, &ident(m.name())),
            ast::Call::AnonymousCall(_) => self.callee_returns(receiver),
            _ => Returns::Dynamic,
        }
    }

    /// What a call through a value of this type returns.
    fn callee_returns(&self, callee: Ty) -> Returns {
        match callee {
            // A function no round has settled yet returns nothing
            // known: the value stays undecided until it has.
            Ty::Func(f) => self
                .known
                .sig(f)
                .map(|s| s.returns.clone())
                .unwrap_or(Returns::Unknown),
            Ty::Unknown => Returns::Unknown,
            _ => Returns::Dynamic,
        }
    }

    /// What `receiver:name(...)` returns: the joined results of the
    /// functions the receiver's classes resolve it to, when every one
    /// does; a string method's declared result.
    pub fn method_returns(&self, receiver: Ty, name: &str) -> Returns {
        match receiver {
            Ty::Str => match builtin_member(self.scopes, "string", name) {
                Some(b) => match b.ret {
                    Ret::Multi => Returns::Dynamic,
                    r => Returns::Fixed(vec![ret_ty(r)]),
                },
                None => Returns::Dynamic,
            },
            Ty::Shape(k) => match self.method_targets(k, name) {
                Some(targets) => {
                    let mut out = Returns::Unknown;
                    for target in targets {
                        let r = self.callee_returns(target);
                        out = match (out, r) {
                            (Returns::Unknown, r) | (r, Returns::Unknown) => r,
                            (Returns::Fixed(x), Returns::Fixed(y)) if x.len() == y.len() => {
                                Returns::Fixed(
                                    x.into_iter().zip(y).map(|(a, b)| a.join(b)).collect(),
                                )
                            }
                            _ => Returns::Dynamic,
                        };
                    }
                    out
                }
                None => Returns::Dynamic,
            },
            Ty::Unknown => Returns::Unknown,
            _ => Returns::Dynamic,
        }
    }

    /// The functions `receiver:name` may be for a receiver of shape
    /// `k`: one per class the shape's tables get, when the field is
    /// not one of the shape's own, every class resolves it, and no
    /// other metatable can reach the tables. Each is a function or,
    /// not settled yet, unknown.
    pub fn method_targets(&self, k: ShapeId, name: &str) -> Option<Vec<Ty>> {
        let info = self.known.shapes.get(k.0 as usize)?;
        let mut out = Vec::new();
        // The table's own field first; one that is always there is
        // all there is.
        match info.field(name) {
            Some((_, ty @ (Ty::Func(_) | Ty::Unknown))) => {
                out.push(ty);
                if info.always_present(name) {
                    return Some(out);
                }
            }
            Some(_) => return None,
            None if self.known.may_hold_unnamed(k, name) => return None,
            None => {}
        }
        if info.unknown_meta {
            return None;
        }
        if info.classes.is_empty() {
            return if out.is_empty() { None } else { Some(out) };
        }
        for class in &info.classes {
            out.extend(self.known.resolve_method(*class, name)?);
        }
        out.sort_by_key(|t| match t {
            Ty::Func(f) => f.0 as i64,
            _ => -1,
        });
        out.dedup();
        Some(out)
    }

    /// `setmetatable(t, m)` returns `t`; `getmetatable(t)` is anything.
    fn metatable_call_returns(&self, b: &Builtin, args: &ast::FunctionArgs) -> Option<Returns> {
        if !b.lib.is_empty() || b.name != "setmetatable" {
            return None;
        }
        let ast::FunctionArgs::Parentheses { arguments, .. } = args else {
            return None;
        };
        let first = arguments.iter().next()?;
        match self.ty_of(first) {
            t @ Ty::Shape(_) => Some(Returns::Fixed(vec![t])),
            Ty::Unknown => Some(Returns::Unknown),
            _ => None,
        }
    }

    /// What `receiver.name` yields.
    pub fn field_ty(&self, receiver: Ty, name: &str) -> Ty {
        match receiver {
            Ty::Shape(k) => match self.known.shapes.get(k.0 as usize) {
                Some(_) => self.known.field_read_ty(k, name),
                None => Ty::Unknown,
            },
            Ty::Unknown => Ty::Unknown,
            _ => Ty::Any,
        }
    }

    /// What a `math` call's arguments make its result, when they do.
    /// The arguments are each expression's one value, the last one's
    /// several; a last argument of no fixed count decides nothing.
    fn math_call_ty(&self, b: &Builtin, args: &ast::FunctionArgs) -> Option<Ty> {
        if b.lib != "math" {
            return None;
        }
        let ast::FunctionArgs::Parentheses { arguments, .. } = args else {
            return None;
        };
        let exprs: Vec<&Expression> = arguments.iter().collect();
        let mut tys = Vec::with_capacity(exprs.len());
        for (i, e) in exprs.iter().enumerate() {
            if i + 1 == exprs.len() {
                match multi_returns(self, e) {
                    Some(Returns::Fixed(more)) => tys.extend(more),
                    Some(Returns::Dynamic) => return None,
                    Some(Returns::Unknown) => tys.push(Ty::Unknown),
                    None => tys.push(self.ty_of(e)),
                }
            } else {
                tys.push(self.ty_of(e));
            }
        }
        math_result(b, &tys)
    }

    fn prefix_ty(&self, p: &Prefix) -> Ty {
        match p {
            Prefix::Name(token) => self.name_ty(token),
            Prefix::Expression(e) => self.ty_of(e),
            _ => Ty::Any,
        }
    }

    /// The global `_G.name` names, when `_G` is the library's table.
    pub fn global_member(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Option<String> {
        let Prefix::Name(token) = prefix else {
            return None;
        };
        let Some(Binding::Global(g)) = self.scopes.binding(token) else {
            return None;
        };
        if !Scopes::is_globals_name(g) || self.scopes.global_writes.contains_key(g) {
            return None;
        }
        // Once `_ENV` is assigned, `_G` is a global like any other,
        // found in whatever the environment is.
        if g == "_G" && self.scopes.global_writes.contains_key("_ENV") {
            return None;
        }
        match suffixes {
            [Suffix::Index(ast::Index::Dot { name, .. })] => Some(ident(name)),
            [Suffix::Index(ast::Index::Brackets { expression, .. })] => {
                crate::scope::literal_string(expression)
            }
            _ => None,
        }
    }

    /// The type of a prefix followed by some of its suffixes.
    pub fn suffixed_ty(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Ty {
        if suffixes.is_empty() {
            return self.prefix_ty(prefix);
        }
        if let Some(name) = self.global_member(prefix, suffixes) {
            return self.global_ty(&name);
        }
        match suffixes.last().unwrap() {
            Suffix::Call(_) => self.call_returns(prefix, suffixes).first(),
            Suffix::Index(index) => {
                let receiver = self.suffixed_ty(prefix, &suffixes[..suffixes.len() - 1]);
                match constant_key(index) {
                    Some(name) => self.field_ty(receiver, &name),
                    None => match (receiver, index) {
                        (Ty::Shape(k), ast::Index::Brackets { expression, .. }) => {
                            self.known.element_read_ty(k, self.ty_of(expression))
                        }
                        (Ty::Unknown, _) => Ty::Unknown,
                        _ => Ty::Any,
                    },
                }
            }
            _ => Ty::Any,
        }
    }

    /// The type of an expression in single-value position.
    pub fn ty_of(&self, e: &Expression) -> Ty {
        match e {
            Expression::Number(t) => numeral_ty(&t.token().to_string()),
            Expression::String(_) => Ty::Str,
            Expression::Symbol(t) => match t.token().to_string().trim() {
                "true" | "false" => Ty::Bool,
                "nil" => Ty::Nil,
                _ => Ty::Any,
            },
            Expression::Parentheses { expression, .. } => self.ty_of(expression),
            Expression::Function(f) => Ty::Func(self.scopes.function_of(f.body())),
            Expression::TableConstructor(t) => match constructor_shape(self.known, t) {
                Some(k) => Ty::Shape(k),
                None => Ty::Table,
            },
            Expression::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                self.suffixed_ty(c.prefix(), &suffixes)
            }
            Expression::Var(v) => match v {
                Var::Name(token) => self.name_ty(token),
                Var::Expression(v) => {
                    let suffixes: Vec<&Suffix> = v.suffixes().collect();
                    self.suffixed_ty(v.prefix(), &suffixes)
                }
                _ => Ty::Any,
            },
            Expression::UnaryOperator { unop, expression } => {
                let t = self.ty_of(expression);
                if t == Ty::Unknown && !matches!(unop, UnOp::Not(_)) {
                    return Ty::Unknown;
                }
                match unop {
                    UnOp::Not(_) => Ty::Bool,
                    UnOp::Minus(_) => match t {
                        Ty::Int | Ty::Float | Ty::Number => t,
                        Ty::Scalar | Ty::Nil | Ty::Bool => Ty::Number,
                        Ty::Shape(_) => self.metamethod_result("__unm", t, t).unwrap_or(Ty::Any),
                        _ => Ty::Any,
                    },
                    UnOp::Hash(_) => match t {
                        Ty::Str => Ty::Int,
                        Ty::Table | Ty::Shape(_) if !self.scopes.len_meta => Ty::Int,
                        _ => Ty::Any,
                    },
                    UnOp::Tilde(_) => match t {
                        Ty::Int => Ty::Int,
                        _ => Ty::Any,
                    },
                    _ => Ty::Any,
                }
            }
            Expression::BinaryOperator { lhs, binop, rhs } => {
                let a = self.ty_of(lhs);
                let b = self.ty_of(rhs);
                if let Some(event) = arith_event(binop)
                    && let Some(ty) = self.metamethod_result(event, a, b)
                {
                    return ty;
                }
                binary_ty(binop, a, b)
            }
            _ => Ty::Any,
        }
    }

    /// What arithmetic on a table operand yields: the joined first
    /// results of the handlers the types know, the left operand's
    /// taking precedence as the runtime's do. None when the runtime
    /// decides.
    pub fn metamethod_result(&self, event: &str, a: Ty, b: Ty) -> Option<Ty> {
        let sides = self.metamethod_sides(event, a, b)?;
        let mut out: Option<Ty> = None;
        for (_, _, f) in sides {
            let r = self.callee_returns(Ty::Func(f)).first();
            out = Some(match out {
                None => r,
                Some(so_far) => join_read(so_far, r),
            });
        }
        Some(out.unwrap_or(Ty::Unknown))
    }

    /// The handlers arithmetic on `a` and `b` may call, in the order
    /// the runtime tries them: the left operand's classes' when it is a
    /// shaped table, then the right's when the left cannot carry a
    /// handler of its own. Each with the operand's side and class.
    /// None when an operand the types do not follow may answer first,
    /// or when neither side is a shaped table with handlers.
    pub fn metamethod_sides(
        &self,
        event: &str,
        a: Ty,
        b: Ty,
    ) -> Option<Vec<(bool, ShapeId, FuncId)>> {
        let mut out = Vec::new();
        let mut left_settled = false;
        if let Ty::Shape(k) = a {
            match self.known.metamethod_targets(k, event) {
                MetaTargets::Funcs(funcs) => {
                    out.extend(funcs.into_iter().map(|(c, f)| (true, c, f)));
                    left_settled = true;
                }
                MetaTargets::Unsettled => return Some(Vec::new()),
                MetaTargets::None => {}
            }
        }
        if let Ty::Shape(k) = b
            && (left_settled || !may_handle_arith(a))
        {
            match self.known.metamethod_targets(k, event) {
                MetaTargets::Funcs(funcs) => {
                    out.extend(funcs.into_iter().map(|(c, f)| (false, c, f)));
                }
                MetaTargets::Unsettled => return Some(Vec::new()),
                MetaTargets::None => {}
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

/// The identifier a name token holds.
pub fn ident(token: &full_moon::tokenizer::TokenReference) -> String {
    match token.token().token_type() {
        full_moon::tokenizer::TokenType::Identifier { identifier } => identifier.to_string(),
        _ => token.token().to_string(),
    }
}

/// The type of `a and b` or `a or b`.
pub fn logical_ty(is_and: bool, a: Ty, b: Ty) -> Ty {
    if a == Ty::Unknown || b == Ty::Unknown {
        return Ty::Unknown;
    }
    if is_and {
        if a.always_truthy() {
            b
        } else if a == Ty::Nil {
            Ty::Nil
        } else if a == Ty::Bool && b == Ty::Bool {
            Ty::Bool
        } else {
            a.join(b)
        }
    } else if a.always_truthy() {
        a
    } else if a == Ty::Nil {
        b
    } else if a == Ty::Bool && b == Ty::Bool {
        Ty::Bool
    } else {
        a.join(b)
    }
}

pub fn binary_ty(op: &BinOp, a: Ty, b: Ty) -> Ty {
    // Nothing known on one side settles nothing yet, except that a
    // comparison is a boolean whatever its operands.
    let comparison = matches!(
        op,
        BinOp::TwoEqual(_)
            | BinOp::TildeEqual(_)
            | BinOp::LessThan(_)
            | BinOp::LessThanEqual(_)
            | BinOp::GreaterThan(_)
            | BinOp::GreaterThanEqual(_)
    );
    if (a == Ty::Unknown || b == Ty::Unknown) && !comparison {
        return Ty::Unknown;
    }
    match op {
        BinOp::Plus(_)
        | BinOp::Minus(_)
        | BinOp::Star(_)
        | BinOp::Percent(_)
        | BinOp::DoubleSlash(_) => {
            // A float operand makes a float; two integers an integer;
            // a number whose kind is not known keeps it open. A scalar
            // that may not be a number raises, or is a number.
            if !(a.is_number() && b.is_number()) {
                if a.is_scalar() && b.is_scalar() {
                    Ty::Number
                } else {
                    Ty::Any
                }
            } else if a == Ty::Int && b == Ty::Int {
                Ty::Int
            } else if a == Ty::Float || b == Ty::Float {
                Ty::Float
            } else {
                Ty::Number
            }
        }
        BinOp::Slash(_) | BinOp::Caret(_) => {
            if a.is_scalar() && b.is_scalar() {
                Ty::Float
            } else {
                Ty::Any
            }
        }
        BinOp::Ampersand(_)
        | BinOp::Pipe(_)
        | BinOp::Tilde(_)
        | BinOp::DoubleLessThan(_)
        | BinOp::DoubleGreaterThan(_) => {
            if a.is_number() && b.is_number() {
                Ty::Int
            } else {
                Ty::Any
            }
        }
        BinOp::TwoDots(_) => {
            let text = |t: Ty| t == Ty::Str || t.is_number();
            if text(a) && text(b) { Ty::Str } else { Ty::Any }
        }
        BinOp::TwoEqual(_)
        | BinOp::TildeEqual(_)
        | BinOp::LessThan(_)
        | BinOp::LessThanEqual(_)
        | BinOp::GreaterThan(_)
        | BinOp::GreaterThanEqual(_) => Ty::Bool,
        BinOp::And(_) => logical_ty(true, a, b),
        BinOp::Or(_) => logical_ty(false, a, b),
        _ => Ty::Any,
    }
}

/// Whether the last statement of a block can be run past, so the
/// function returns nothing there.
pub fn falls_through(block: &Block) -> bool {
    match block.last_stmt() {
        Some(ast::LastStmt::Return(_)) => false,
        Some(ast::LastStmt::Break(_)) => false,
        _ => match block.stmts().last() {
            Some(Stmt::If(i)) => {
                let Some(els) = i.else_block() else {
                    return true;
                };
                falls_through(i.block())
                    || i.else_if()
                        .is_some_and(|e| e.iter().any(|e| falls_through(e.block())))
                    || falls_through(els)
            }
            Some(Stmt::Do(d)) => falls_through(d.block()),
            _ => true,
        },
    }
}

/// One round of inference over the chunk, from what the last round
/// settled.
struct Round<'a> {
    scopes: &'a Scopes,
    known: &'a Inferred,
    out: Inferred,
    /// The function being walked.
    func: FuncId,
    /// What the function being walked returns, joined so far.
    returns: Option<Returns>,
    /// Field names read, or called as methods, through a receiver the
    /// types do not know: a function of that name in a shape the
    /// program may reach through such a receiver escapes.
    blind_reads: HashSet<String>,
    /// Reads under keys not known at compile time, or walks by the
    /// library, through receivers the types do not know: they reach
    /// every escaping shape.
    blind_keyed_reads: Reads,
    /// The call a generic `for` being walked takes its iterator from,
    /// by the position of its callee's name: a library iterator it
    /// returns is called by the loop alone, with the table it was
    /// given.
    loop_iterator: Option<usize>,
    /// Whether `getmetatable` is applied to a receiver the types do
    /// not know, or whether its result is a class: classes escape.
    blind_getmetatable: bool,
}

impl<'a> Round<'a> {
    fn typer(&self) -> Typer<'_> {
        Typer {
            scopes: self.scopes,
            known: self.known,
        }
    }

    /// A function reached through this value may be called with
    /// anything from here on.
    fn escape(&mut self, ty: Ty) {
        match ty {
            Ty::Func(f) => {
                if self.out.escaping.insert(f) && std::env::var_os("ZYNTAX_TRACE_ESCAPES").is_some()
                {
                    eprintln!(
                        "[escape] function {} ({:?}) in {}",
                        self.scopes.func(f).name,
                        f,
                        std::backtrace::Backtrace::force_capture()
                            .to_string()
                            .lines()
                            .filter(|l| l.contains("types.rs:"))
                            .take(4)
                            .map(|l| l.trim().to_string())
                            .collect::<Vec<_>>()
                            .join(" <- ")
                    );
                }
            }
            Ty::Shape(k) => {
                if !self.out.shapes[k.0 as usize].escapes
                    && std::env::var_os("ZYNTAX_TRACE_ESCAPES").is_some()
                {
                    eprintln!(
                        "[escape] shape {} in {}",
                        k.0,
                        std::backtrace::Backtrace::force_capture()
                            .to_string()
                            .lines()
                            .filter(|l| l.contains("types.rs:"))
                            .take(4)
                            .map(|l| l.trim().to_string())
                            .collect::<Vec<_>>()
                            .join(" <- ")
                    );
                }
                self.out.shapes[k.0 as usize].escapes = true;
            }
            _ => {}
        }
    }

    /// The shape of a constructor's tables, made if none is.
    fn shape_for(&mut self, t: &ast::TableConstructor) -> ShapeId {
        let keys = constructor_keys(t);
        let line = t.braces().tokens().0.token().start_position().line();
        let make = |shapes: &mut Vec<ShapeInfo>, keys: &[String]| {
            let id = ShapeId(shapes.len() as u32);
            let mut fields = IndexMap::new();
            for k in keys {
                fields.entry(k.clone()).or_insert(Ty::Unknown);
            }
            let born = fields.len();
            shapes.push(ShapeInfo {
                fields,
                born,
                line,
                ..Default::default()
            });
            id
        };
        if keys.is_empty() {
            let site = constructor_site(t);
            if let Some(k) = self.out.shape_by_site.get(&site) {
                return *k;
            }
            let id = make(&mut self.out.shapes, &keys);
            self.out.shape_by_site.insert(site, id);
            id
        } else {
            let mut sorted = keys.clone();
            sorted.sort();
            sorted.dedup();
            if let Some(k) = self.out.shape_by_keys.get(&sorted) {
                return *k;
            }
            let id = make(&mut self.out.shapes, &sorted);
            self.out.shape_by_keys.insert(sorted, id);
            id
        }
    }

    /// A store of `ty` into field `name` of the tables of shape `k`.
    fn store_field(&mut self, k: ShapeId, name: &str, ty: Ty) {
        let slot = self.out.shapes[k.0 as usize]
            .fields
            .get(name)
            .copied()
            .unwrap_or(Ty::Unknown);
        let joined = self.join_into(slot, ty);
        self.out.shapes[k.0 as usize]
            .fields
            .insert(name.to_string(), joined);
    }

    /// A store of `ty` under a key that is not a string into the
    /// tables of shape `k`.
    fn store_element(&mut self, k: ShapeId, ty: Ty) {
        let element = self.out.shapes[k.0 as usize].element;
        let joined = self.join_into(element, ty);
        self.out.shapes[k.0 as usize].element = joined;
    }

    /// A store through `receiver` under `key` of a value of type `ty`:
    /// into the shape's field or element, or blind.
    fn store_through(&mut self, receiver: Ty, key: Option<String>, key_ty: Ty, ty: Ty) {
        match (receiver, key) {
            (Ty::Shape(k), Some(name)) => self.store_field(k, &name, ty),
            // Under a number the value is an element and stays typed.
            // Under a string it may land in any field, or under a name
            // that is none; under anything else it is lost in the hash
            // part.
            (Ty::Shape(k), None) => {
                if key_may_be_index(key_ty) {
                    self.store_element(k, ty);
                }
                if key_may_be_string(key_ty) {
                    self.out.shapes[k.0 as usize].dynamic_keys = true;
                    let names: Vec<String> = self.out.shapes[k.0 as usize]
                        .fields
                        .keys()
                        .cloned()
                        .collect();
                    for name in names {
                        self.store_field(k, &name, ty);
                    }
                    let held = self.out.shapes[k.0 as usize].dyn_value;
                    self.out.shapes[k.0 as usize].dyn_value = self.join_into(held, ty);
                }
                if !key_is_number(key_ty) {
                    self.escape(ty);
                }
            }
            (Ty::Unknown, _) => {}
            (_, Some(name)) => {
                let slot = self
                    .out
                    .blind_stores
                    .get(&name)
                    .copied()
                    .unwrap_or(Ty::Unknown);
                let joined = self.join_into(slot, ty);
                self.out.blind_stores.insert(name, joined);
                self.escape(ty);
            }
            (_, None) => {
                if key_may_be_string(key_ty) {
                    self.out.blind_dynamic_stores = true;
                }
                if key_may_be_index(key_ty) {
                    let element = self.out.blind_element;
                    self.out.blind_element = self.join_into(element, ty);
                }
                self.escape(ty);
            }
        }
    }

    /// A method called through a value: every function `name` may
    /// reach from a table of shape `k`, in the table or along its
    /// classes' `__index` chains, escapes.
    fn escape_method(&mut self, k: ShapeId, name: &str) {
        let mut seen = BTreeSet::new();
        let mut todo = vec![k];
        while let Some(t) = todo.pop() {
            if !seen.insert(t) || t.0 as usize >= self.out.shapes.len() {
                continue;
            }
            let info = &self.out.shapes[t.0 as usize];
            let own = info.field(name).map(|(_, ty)| ty);
            let classes: Vec<ShapeId> = info.classes.iter().copied().collect();
            if let Some(ty) = own {
                self.escape(ty);
            }
            for c in classes {
                todo.push(c);
                if let Some((_, Ty::Shape(next))) = self.out.shapes[c.0 as usize].field("__index") {
                    todo.push(next);
                }
            }
        }
    }

    /// A read of field `name` through `receiver`, noted when the
    /// receiver is not a shape.
    fn read_through(&mut self, receiver: Ty, name: &str) {
        if !matches!(receiver, Ty::Shape(_) | Ty::Unknown | Ty::Str) {
            self.blind_reads.insert(name.to_string());
        }
    }

    /// `receiver[key]` for a key of type `key_ty` not known at compile
    /// time: unless the types know what it yields, what it may yield
    /// escapes.
    fn keyed_read(&mut self, receiver: Ty, key_ty: Ty) {
        if let Ty::Shape(k) = receiver
            && self.known.element_read_ty(k, key_ty) != Ty::Any
        {
            return;
        }
        self.untyped_read(receiver, Reads::under(key_ty, true));
    }

    /// Reads out of `receiver` whose results the types do not follow:
    /// noted on its shape, or, through a receiver the types do not
    /// know, against every escaping shape.
    fn untyped_read(&mut self, receiver: Ty, reads: Reads) {
        match receiver {
            Ty::Shape(k) => {
                let info = &mut self.out.shapes[k.0 as usize];
                info.untyped_reads = info.untyped_reads.union(reads);
            }
            // A string's reads reach the string library, not a table.
            Ty::Unknown | Ty::Str => {}
            _ => self.blind_keyed_reads = self.blind_keyed_reads.union(reads),
        }
    }

    /// What reaches the shapes the types lost track of: a table of an
    /// escaping shape may take the stores made through receivers the
    /// types do not know, get any metatable set that way, and have
    /// its fields read and its methods called that way. What such a
    /// table holds escapes with it; its classes are reached through
    /// `getmetatable`.
    fn settle_shapes(&mut self) {
        let dynamic_code = self.scopes.dynamic_code;
        // The library's readers are reached through the globals table,
        // or by code the chunk does not contain.
        if dynamic_code || self.scopes.dynamic_globals {
            self.loose_reader();
        }
        loop {
            let mut changed = false;
            // A function under a metamethod's name is called by the
            // runtime, with dynamic values, wherever its table serves
            // as a metatable: it escapes, unless the event is one the
            // typed dispatch handles and no table of the shapes it is
            // the metatable of ever reaches the runtime's own dispatch,
            // that is, none of them escapes (nor the class itself).
            for k in 0..self.out.shapes.len() {
                let metamethods: Vec<(String, Ty)> = self.out.shapes[k]
                    .fields
                    .iter()
                    .filter(|(name, _)| name.starts_with("__"))
                    .map(|(name, ty)| (name.clone(), *ty))
                    .collect();
                let class = ShapeId(k as u32);
                let reached = self.out.shapes[k].escapes
                    || self
                        .out
                        .shapes
                        .iter()
                        .any(|s| s.escapes && s.classes.contains(&class));
                for (name, ty) in metamethods {
                    // An `__index` table is looked up in, not handed
                    // out: it escapes with the tables whose lookups
                    // reach it, below.
                    if name == "__index" && matches!(ty, Ty::Shape(_)) {
                        continue;
                    }
                    if reached || !typed_event(&name) {
                        changed |= self.escape_new(ty);
                    }
                }
            }
            for k in 0..self.out.shapes.len() {
                changed |= self.settle_reads(k);
                if !self.out.shapes[k].escapes {
                    continue;
                }
                // A table of the shape may be reached through anything.
                let fields: Vec<(String, Ty)> = self.out.shapes[k]
                    .fields
                    .iter()
                    .map(|(n, t)| (n.clone(), *t))
                    .collect();
                for (name, ty) in &fields {
                    match ty {
                        Ty::Shape(other) if !self.out.shapes[other.0 as usize].escapes => {
                            self.out.shapes[other.0 as usize].escapes = true;
                            changed = true;
                        }
                        Ty::Func(f)
                            if (self.blind_reads.contains(name)
                                || self.blind_getmetatable
                                || dynamic_code)
                                && !self.out.escaping.contains(f) =>
                        {
                            self.escape(Ty::Func(*f));
                            changed = true;
                        }
                        _ => {}
                    }
                }
                if self.out.blind_setmetatable && !self.out.shapes[k].unknown_meta {
                    self.out.shapes[k].unknown_meta = true;
                    changed = true;
                }
                if (self.out.blind_dynamic_stores || dynamic_code)
                    && !self.out.shapes[k].dynamic_keys
                {
                    self.out.shapes[k].dynamic_keys = true;
                    changed = true;
                }
                // Its array part takes the stores made blind, and what
                // it holds escapes with it.
                let element = self.out.shapes[k].element;
                let reaching = if dynamic_code {
                    Ty::Any
                } else {
                    self.out.blind_element
                };
                let joined = self.join_into(element, reaching);
                if joined != element {
                    self.out.shapes[k].element = joined;
                    changed = true;
                }
                if joined != Ty::Any && !matches!(joined, Ty::Unknown) {
                    changed |= self.escape_new(joined);
                }
                if self.blind_getmetatable || dynamic_code {
                    let classes: Vec<ShapeId> =
                        self.out.shapes[k].classes.iter().copied().collect();
                    for c in classes {
                        if !self.out.shapes[c.0 as usize].escapes {
                            self.out.shapes[c.0 as usize].escapes = true;
                            changed = true;
                        }
                    }
                }
                // Its classes' methods are reached through it, and the
                // tables its lookups go on to.
                let classes: Vec<ShapeId> = self.out.shapes[k].classes.iter().copied().collect();
                for c in classes {
                    if let Some((_, ty @ Ty::Shape(_))) =
                        self.out.shapes[c.0 as usize].field("__index")
                    {
                        changed |= self.escape_new(ty);
                    }
                    let methods: Vec<(String, Ty)> = self.out.shapes[c.0 as usize]
                        .fields
                        .iter()
                        .map(|(n, t)| (n.clone(), *t))
                        .collect();
                    for (name, ty) in methods {
                        if let Ty::Func(f) = ty
                            && (self.blind_reads.contains(&name) || dynamic_code)
                            && !self.out.escaping.contains(&f)
                        {
                            self.escape(Ty::Func(f));
                            changed = true;
                        }
                    }
                }
                let blind: Vec<(String, Ty)> = self
                    .out
                    .blind_stores
                    .iter()
                    .map(|(n, t)| (n.clone(), *t))
                    .collect();
                for (name, ty) in blind {
                    if let Some(slot) = self.out.shapes[k].fields.get(&name).copied() {
                        let joined = self.join_into(slot, ty);
                        if joined != slot {
                            self.out.shapes[k].fields.insert(name, joined);
                            changed = true;
                        }
                    }
                }
                if self.out.shapes[k].dynamic_keys {
                    let names: Vec<String> = self.out.shapes[k].fields.keys().cloned().collect();
                    for name in names {
                        let slot = self.out.shapes[k].fields[&name];
                        if slot != Ty::Any {
                            self.escape(slot);
                            self.out.shapes[k].fields.insert(name, Ty::Any);
                            changed = true;
                        }
                    }
                    let held = self.out.shapes[k].dyn_value;
                    if held != Ty::Any {
                        self.escape(held);
                        self.out.shapes[k].dyn_value = Ty::Any;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// What the untyped reads of a table of shape `k` may yield
    /// escapes, and those that go on through `__index` reach the
    /// tables behind it; an escaping shape takes the reads made
    /// through receivers the types do not know. Whether anything
    /// changed.
    fn settle_reads(&mut self, k: usize) -> bool {
        let mut changed = false;
        if self.out.shapes[k].escapes {
            let reads = self.out.shapes[k]
                .untyped_reads
                .union(self.blind_keyed_reads);
            if reads != self.out.shapes[k].untyped_reads {
                self.out.shapes[k].untyped_reads = reads;
                changed = true;
            }
        }
        let reads = self.out.shapes[k].untyped_reads;
        let mut held = Vec::new();
        if reads.names {
            held.extend(self.out.shapes[k].fields.values().copied());
            held.push(self.out.shapes[k].dyn_value);
        }
        if reads.elements {
            held.push(self.out.shapes[k].element);
        }
        for ty in held {
            changed |= self.escape_new(ty);
        }
        if reads.through_meta && (reads.names || reads.elements) {
            // A metatable the types do not follow was lost to them, and
            // its `__index` with it: the reads reach every escaping
            // shape.
            if self.out.shapes[k].unknown_meta {
                let blind = self.blind_keyed_reads.union(reads);
                if blind != self.blind_keyed_reads {
                    self.blind_keyed_reads = blind;
                    changed = true;
                }
            }
            let classes: Vec<ShapeId> = self.out.shapes[k].classes.iter().copied().collect();
            for c in classes {
                if let Some((_, Ty::Shape(next))) = self.out.shapes[c.0 as usize].field("__index") {
                    let info = &mut self.out.shapes[next.0 as usize];
                    let joined = info.untyped_reads.union(reads);
                    if joined != info.untyped_reads {
                        info.untyped_reads = joined;
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    /// `escape`, and whether it made the function or shape escape.
    fn escape_new(&mut self, ty: Ty) -> bool {
        let new = match ty {
            Ty::Func(f) => !self.out.escaping.contains(&f),
            Ty::Shape(k) => !self.out.shapes[k.0 as usize].escapes,
            _ => false,
        };
        self.escape(ty);
        new
    }

    /// Whether the function's parameters take anything: decided by the
    /// escapes found at the last settled typing, not by this round's,
    /// so that a type an early round got wrong cannot fix a function's
    /// parameters for good.
    fn is_escaping(&self, f: FuncId) -> bool {
        self.scopes.func(f).escapes || self.known.escaping.contains(&f)
    }

    /// `value` joined into what `slot` holds: a function joined with
    /// anything but itself is lost to the types, and escapes.
    fn join_into(&mut self, slot: Ty, value: Ty) -> Ty {
        let joined = slot.join(value);
        // A side the join did not keep as it was is lost to the types.
        if joined != slot {
            self.escape(slot);
        }
        if joined != value {
            self.escape(value);
        }
        joined
    }

    fn join_returns(&mut self, a: Returns, b: Returns) -> Returns {
        match (a, b) {
            (Returns::Unknown, r) | (r, Returns::Unknown) => r,
            (Returns::Fixed(x), Returns::Fixed(y)) if x.len() == y.len() => Returns::Fixed(
                x.into_iter()
                    .zip(y)
                    .map(|(p, q)| self.join_into(p, q))
                    .collect(),
            ),
            (a, b) => {
                self.escape_returns(&a);
                self.escape_returns(&b);
                Returns::Dynamic
            }
        }
    }

    fn escape_returns(&mut self, r: &Returns) {
        if let Returns::Fixed(types) = r {
            for t in types {
                self.escape(*t);
            }
        }
    }

    fn assign_var(&mut self, v: VarId, ty: Ty) {
        let joined = self.join_into(self.out.var(v), ty);
        self.out.vars.insert(v, joined);
    }

    fn assign_binding(&mut self, binding: &Binding, ty: Ty) {
        match binding {
            Binding::Local(v) | Binding::Upvalue(v) => self.assign_var(*v, ty),
            Binding::Global(name) => {
                // In a program that reaches its globals through a
                // table, a global is an entry of it.
                if self.scopes.dynamic_globals {
                    self.escape(ty);
                }
                let joined = self.join_into(self.out.global(name), ty);
                self.out.globals.insert(name.clone(), joined);
            }
            // A free name in the scope of a local `_ENV`: a store into
            // its field.
            Binding::Field(env, name, _) => {
                let receiver = self.known.var(*env);
                self.store_through(receiver, Some(name.clone()), Ty::Str, ty);
            }
        }
    }

    /// An expression whose value goes where the types do not follow
    /// it: a function there escapes.
    fn value(&mut self, e: &Expression) {
        self.expr(e);
        let ty = self.typer().ty_of(e);
        self.escape(ty);
    }

    /// The types of `n` targets assigned from `exprs`, Lua's way: the
    /// last expression's several values fill the rest, missing ones
    /// are nil.
    fn assigned_types(&self, exprs: &[&Expression], n: usize) -> Vec<Ty> {
        let typer = self.typer();
        let mut out = Vec::with_capacity(n);
        for (i, e) in exprs.iter().enumerate() {
            if out.len() >= n {
                break;
            }
            let last = i == exprs.len() - 1;
            if last && out.len() + 1 < n {
                // The last expression supplies the rest.
                match multi_returns(&typer, e) {
                    Some(Returns::Fixed(types)) => {
                        for k in 0..(n - out.len()) {
                            out.push(types.get(k).copied().unwrap_or(Ty::Nil));
                        }
                    }
                    Some(Returns::Dynamic) => {
                        while out.len() < n {
                            out.push(Ty::Any);
                        }
                    }
                    Some(Returns::Unknown) => {
                        while out.len() < n {
                            out.push(Ty::Unknown);
                        }
                    }
                    None => {
                        out.push(typer.ty_of(e));
                    }
                }
            } else {
                out.push(typer.ty_of(e));
            }
        }
        while out.len() < n {
            out.push(Ty::Nil);
        }
        out
    }

    fn block(&mut self, block: &Block) {
        for stmt in block.stmts() {
            self.stmt(stmt);
        }
        if let Some(ast::LastStmt::Return(r)) = block.last_stmt() {
            let exprs: Vec<&Expression> = r.returns().iter().collect();
            for e in &exprs {
                self.expr(e);
            }
            let typer = self.typer();
            let returns = match exprs.last() {
                None => Returns::Fixed(Vec::new()),
                Some(last) => {
                    let mut types: Vec<Ty> = exprs[..exprs.len() - 1]
                        .iter()
                        .map(|e| typer.ty_of(e))
                        .collect();
                    match multi_returns(&typer, last) {
                        Some(Returns::Fixed(rest)) => {
                            types.extend(rest);
                            Returns::Fixed(types)
                        }
                        Some(Returns::Dynamic) => Returns::Dynamic,
                        // A tail not typed yet: this return decides
                        // nothing this round.
                        Some(Returns::Unknown) => Returns::Unknown,
                        None => {
                            types.push(typer.ty_of(last));
                            Returns::Fixed(types)
                        }
                    }
                }
            };
            self.returns = Some(match self.returns.take() {
                Some(r) => self.join_returns(r, returns),
                None => returns,
            });
        }
    }

    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Assignment(a) => {
                let exprs: Vec<&Expression> = a.expressions().iter().collect();
                let targets: Vec<&Var> = a.variables().iter().collect();
                // Stored into a variable or a shape's field, a value
                // keeps its type; into a table, it is on its own.
                let kept = targets.iter().all(|t| match t {
                    Var::Name(_) => true,
                    Var::Expression(v) => {
                        let suffixes: Vec<&Suffix> = v.suffixes().collect();
                        if self.typer().global_member(v.prefix(), &suffixes).is_some() {
                            return true;
                        }
                        let receiver =
                            self.typer().suffixed_ty(v.prefix(), &suffixes[..suffixes.len() - 1]);
                        matches!(receiver, Ty::Shape(_) | Ty::Unknown)
                            && matches!(suffixes.last(), Some(Suffix::Index(i)) if constant_key(i).is_some())
                    }
                    _ => false,
                });
                for e in &exprs {
                    if kept {
                        self.expr(e);
                    } else {
                        self.value(e);
                    }
                }
                let types = self.assigned_types(&exprs, targets.len());
                for (target, ty) in targets.iter().zip(types) {
                    match target {
                        Var::Name(token) => {
                            if let Some(b) = self.scopes.binding(token).cloned() {
                                self.assign_binding(&b, ty);
                            }
                        }
                        Var::Expression(v) => {
                            let suffixes: Vec<&Suffix> = v.suffixes().collect();
                            if let Some(name) = self.typer().global_member(v.prefix(), &suffixes) {
                                self.assign_binding(&Binding::Global(name), ty);
                                continue;
                            }
                            self.chain(v.prefix(), &suffixes[..suffixes.len() - 1]);
                            let receiver = self
                                .typer()
                                .suffixed_ty(v.prefix(), &suffixes[..suffixes.len() - 1]);
                            if let Some(Suffix::Index(index)) = suffixes.last() {
                                let key = constant_key(index);
                                let key_ty = match index {
                                    ast::Index::Brackets { expression, .. } => {
                                        self.value(expression);
                                        self.typer().ty_of(expression)
                                    }
                                    _ => Ty::Str,
                                };
                                self.store_through(receiver, key, key_ty, ty);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Stmt::Do(d) => self.block(d.block()),
            Stmt::FunctionCall(c) => self.call(c),
            Stmt::FunctionDeclaration(f) => {
                let names: Vec<_> = f.name().names().iter().collect();
                let id = self.scopes.function_of(f.body());
                let method = f.name().method_name();
                if names.len() == 1
                    && method.is_none()
                    && let Some(b) = self.scopes.binding(names[0]).cloned()
                {
                    self.assign_binding(&b, Ty::Func(id));
                } else {
                    // `function a.b.c()` or `function a.b:c()`: stored in
                    // the field `c` of `a.b`, whatever that is.
                    let last = match method {
                        Some(m) => ident(m),
                        None => ident(names[names.len() - 1]),
                    };
                    let path_len = if method.is_some() {
                        names.len()
                    } else {
                        names.len() - 1
                    };
                    let mut receiver = match self.scopes.binding(names[0]) {
                        Some(b) => self.typer().binding_ty(b),
                        None => Ty::Any,
                    };
                    for name in &names[1..path_len] {
                        receiver = self.typer().field_ty(receiver, &ident(name));
                    }
                    self.store_through(receiver, Some(last), Ty::Str, Ty::Func(id));
                }
                self.function(f.body());
            }
            Stmt::GenericFor(f) => {
                let exprs: Vec<&Expression> = f.expressions().iter().collect();
                // The iterator is called by the loop, through its value.
                let outer = self.loop_iterator.take();
                if let [Expression::FunctionCall(c)] = exprs.as_slice()
                    && let Prefix::Name(token) = c.prefix()
                {
                    self.loop_iterator = Some(crate::scope::pos_of(token));
                }
                for e in &exprs {
                    self.value(e);
                }
                self.loop_iterator = outer;
                let names: Vec<VarId> = f.names().iter().map(|n| self.scopes.declared(n)).collect();
                // `ipairs` gives an integer key; anything else, dynamic
                // values.
                let ipairs = exprs.len() == 1
                    && matches!(exprs[0], Expression::FunctionCall(c)
                        if self.typer().builtin_callee(c.prefix(), &c.suffixes().collect::<Vec<_>>())
                            .is_some_and(|b| b.name == "ipairs" && b.lib.is_empty()));
                let element = if ipairs {
                    let table = match exprs[0] {
                        Expression::FunctionCall(c) => match c.suffixes().last() {
                            Some(Suffix::Call(ast::Call::AnonymousCall(
                                ast::FunctionArgs::Parentheses { arguments, .. },
                            ))) => arguments
                                .iter()
                                .next()
                                .map(|a| self.typer().ty_of(a))
                                .unwrap_or(Ty::Nil),
                            _ => Ty::Any,
                        },
                        _ => Ty::Any,
                    };
                    self.known.ipairs_value_ty(table)
                } else {
                    Ty::Any
                };
                for (i, v) in names.iter().enumerate() {
                    let ty = match i {
                        0 if ipairs => Ty::Int,
                        1 if ipairs => element,
                        _ => Ty::Any,
                    };
                    self.assign_var(*v, ty);
                }
                self.block(f.block());
            }
            Stmt::If(i) => {
                self.value(i.condition());
                self.block(i.block());
                if let Some(elseifs) = i.else_if() {
                    for e in elseifs {
                        self.value(e.condition());
                        self.block(e.block());
                    }
                }
                if let Some(b) = i.else_block() {
                    self.block(b);
                }
            }
            Stmt::LocalAssignment(l) => {
                let exprs: Vec<&Expression> = l.expressions().iter().collect();
                for e in &exprs {
                    self.expr(e);
                }
                let names: Vec<VarId> = l.names().iter().map(|n| self.scopes.declared(n)).collect();
                let types = if exprs.is_empty() {
                    vec![Ty::Nil; names.len()]
                } else {
                    self.assigned_types(&exprs, names.len())
                };
                for (v, ty) in names.iter().zip(types) {
                    self.assign_var(*v, ty);
                }
            }
            Stmt::LocalFunction(f) => {
                let v = self.scopes.declared(f.name());
                let id = self.scopes.function_of(f.body());
                self.assign_var(v, Ty::Func(id));
                self.function(f.body());
            }
            Stmt::NumericFor(f) => {
                self.value(f.start());
                self.value(f.end());
                if let Some(s) = f.step() {
                    self.value(s);
                }
                let typer = self.typer();
                let start = typer.ty_of(f.start());
                let step = f.step().map(|s| typer.ty_of(s)).unwrap_or(Ty::Int);
                let ty = if start == Ty::Float || step == Ty::Float {
                    Ty::Float
                } else {
                    Ty::Int
                };
                let v = self.scopes.declared(f.index_variable());
                self.assign_var(v, ty);
                self.block(f.block());
            }
            Stmt::Repeat(r) => {
                self.block(r.block());
                self.value(r.until());
            }
            Stmt::While(w) => {
                self.value(w.condition());
                self.block(w.block());
            }
            _ => {}
        }
    }

    /// A nested function: walked as its own, its signature recorded.
    fn function(&mut self, body: &ast::FunctionBody) {
        let id = self.scopes.function_of(body);
        let info = self.scopes.func(id);
        let outer_func = self.func;
        let outer_returns = self.returns.take();
        self.func = id;
        // Parameters: what direct calls pass, unless the function is
        // called through values too, when anything may arrive.
        let escapes = self.is_escaping(id);
        let param_tys: Vec<Ty> = if escapes {
            vec![Ty::Any; info.params.len()]
        } else {
            info.params
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    self.known
                        .sig(id)
                        .and_then(|s| s.params.get(i).copied())
                        .unwrap_or(Ty::Unknown)
                })
                .collect()
        };
        for (v, ty) in info.params.iter().zip(&param_tys) {
            self.assign_var(*v, *ty);
        }
        self.block(body.block());
        let mut returns = self.returns.take().unwrap_or(Returns::Fixed(Vec::new()));
        if crate::types::falls_through(body.block()) {
            returns = self.join_returns(returns, Returns::Fixed(Vec::new()));
        }
        // What an escaping function returns leaves through its record,
        // as dynamic values.
        if escapes {
            self.escape_returns(&returns);
        }
        // The parameters stay what this round's calls joined into them;
        // only the result is settled here.
        let n = info.params.len();
        let entry = self.out.funcs.entry(id).or_insert_with(|| Sig {
            params: vec![Ty::Unknown; n],
            returns: Returns::Fixed(Vec::new()),
        });
        entry.returns = returns;
        if escapes {
            entry.params = vec![Ty::Any; n];
        }
        self.func = outer_func;
        self.returns = outer_returns;
    }

    fn call(&mut self, c: &ast::FunctionCall) {
        let suffixes: Vec<&Suffix> = c.suffixes().collect();
        // `setmetatable(t, m)` links a shape to its class; `rawset`
        // stores; neither loses its table.
        if let Some(b) = self.typer().builtin_callee(c.prefix(), &suffixes)
            && let Some(Suffix::Call(ast::Call::AnonymousCall(ast::FunctionArgs::Parentheses {
                arguments,
                ..
            }))) = suffixes.last()
        {
            let args: Vec<&Expression> = arguments.iter().collect();
            if b.lib.is_empty()
                && matches!(
                    b.name,
                    "setmetatable" | "getmetatable" | "rawset" | "rawget"
                )
            {
                self.metatable_builtin(b.name, &args);
                return;
            }
            // `table.sort` of a table with no metatable by a known
            // function calls it with the elements: a direct call, the
            // comparator kept as it is.
            if b.lib == "table"
                && b.name == "sort"
                && args.len() == 2
                && let Ty::Shape(k) = self.typer().ty_of(args[0])
                && let Ty::Func(f) = self.typer().ty_of(args[1])
                && !self.known.shape(k).unknown_meta
                && self.known.shape(k).classes.is_empty()
            {
                self.expr(args[0]);
                self.expr(args[1]);
                let element = self.known.shape(k).element;
                self.record_call_types(f, vec![element, element]);
                return;
            }
            // `table.insert` and `table.move` store elements into their
            // table; the rest of the call takes values as any library
            // call does.
            if b.lib == "table" && matches!(b.name, "insert" | "move") {
                self.chain(c.prefix(), &suffixes);
                let tys: Vec<Ty> = args.iter().map(|a| self.typer().ty_of(a)).collect();
                match b.name {
                    "insert" if tys.len() >= 2 => {
                        self.store_through(tys[0], None, Ty::Int, tys[tys.len() - 1]);
                    }
                    // What it reads of a table with a metatable may come
                    // through `__index`.
                    "move" if tys.len() >= 4 => {
                        let element = match tys[0] {
                            Ty::Shape(k)
                                if !self.known.shape(k).unknown_meta
                                    && self.known.shape(k).classes.is_empty() =>
                            {
                                self.known.shape(k).element
                            }
                            Ty::Unknown => Ty::Unknown,
                            source => {
                                self.untyped_read(source, Reads::under(Ty::Int, true));
                                Ty::Any
                            }
                        };
                        let into = tys.get(4).copied().unwrap_or(tys[0]);
                        self.store_through(into, None, Ty::Int, element);
                    }
                    _ => {}
                }
                return;
            }
        }
        self.chain(c.prefix(), &suffixes);
    }

    /// What a direct call of the library function `b` reads out of
    /// its table argument and hands out untyped.
    fn library_reads_of(&mut self, b: &Builtin, prefix: &Prefix, args: &ast::FunctionArgs) {
        if let Some(reads) = library_reads(b) {
            let table = match args {
                ast::FunctionArgs::Parentheses { arguments, .. } => match arguments.iter().next() {
                    Some(first) => self.typer().ty_of(first),
                    None => Ty::Nil,
                },
                ast::FunctionArgs::TableConstructor(t) => match constructor_shape(self.known, t) {
                    Some(k) => Ty::Shape(k),
                    None => Ty::Table,
                },
                _ => Ty::Str,
            };
            // A typed `ipairs` binds each element as its type; what
            // its iterator hands elsewhere leaves with the table,
            // which the call takes as a value.
            let typed = b.name == "ipairs" && self.known.ipairs_value_ty(table) != Ty::Any;
            if !typed {
                self.untyped_read(table, reads);
            }
        }
        // The iterator `pairs` or `ipairs` returns reads whatever
        // table it is given; outside a loop's header, any.
        let header = match prefix {
            Prefix::Name(token) => self.loop_iterator == Some(crate::scope::pos_of(token)),
            _ => false,
        };
        if b.lib.is_empty() && matches!(b.name, "pairs" | "ipairs") && !header {
            self.loose_reader();
        }
    }

    /// A library function that reads the tables it is given is held
    /// as a value: whatever it is called with, it may hand out as
    /// dynamic values what a table of any escaping shape holds.
    fn loose_reader(&mut self) {
        self.blind_keyed_reads = self.blind_keyed_reads.union(Reads {
            names: true,
            elements: true,
            through_meta: true,
        });
    }

    /// Whether the global `name` is a library function that reads a
    /// table it is given, or the library table holding such functions.
    fn is_reader(&self, name: &str) -> bool {
        builtin_named(self.scopes, name).is_some_and(|b| library_reads(b).is_some())
            || (name == "table" && !self.scopes.global_writes.contains_key(name))
    }

    /// `setmetatable`, `getmetatable`, `rawset`, `rawget`: what they do
    /// to the shapes involved.
    fn metatable_builtin(&mut self, name: &str, args: &[&Expression]) {
        for a in args {
            self.expr(a);
        }
        let tys: Vec<Ty> = args.iter().map(|a| self.typer().ty_of(a)).collect();
        let receiver = tys.first().copied().unwrap_or(Ty::Nil);
        match name {
            "setmetatable" => {
                let meta = tys.get(1).copied().unwrap_or(Ty::Nil);
                match (receiver, meta) {
                    (Ty::Shape(k), Ty::Shape(c)) => {
                        self.out.shapes[k.0 as usize].classes.insert(c);
                    }
                    (Ty::Shape(_), Ty::Nil | Ty::Unknown) => {}
                    (Ty::Shape(k), _) => {
                        self.out.shapes[k.0 as usize].unknown_meta = true;
                        self.escape(meta);
                    }
                    (Ty::Unknown, _) => {}
                    (_, _) => {
                        self.out.blind_setmetatable = true;
                        self.escape(meta);
                    }
                }
                for t in tys.iter().skip(2) {
                    self.escape(*t);
                }
            }
            "getmetatable" => {
                match receiver {
                    Ty::Shape(k) => {
                        let classes = self.out.shapes[k.0 as usize].classes.clone();
                        for c in classes {
                            self.escape(Ty::Shape(c));
                        }
                    }
                    Ty::Unknown => {}
                    _ => self.blind_getmetatable = true,
                }
                for t in tys.iter().skip(1) {
                    self.escape(*t);
                }
            }
            "rawset" => {
                let key = args.get(1).and_then(|k| crate::scope::literal_string(k));
                let key_ty = tys.get(1).copied().unwrap_or(Ty::Nil);
                let value = tys.get(2).copied().unwrap_or(Ty::Nil);
                self.store_through(receiver, key, key_ty, value);
                for t in tys.iter().skip(3) {
                    self.escape(*t);
                }
            }
            // `rawget` hands out what it finds as a dynamic value.
            _ => {
                let key_ty = tys.get(1).copied().unwrap_or(Ty::Nil);
                match args.get(1).and_then(|k| crate::scope::literal_string(k)) {
                    Some(key) if !matches!(receiver, Ty::Shape(_)) => {
                        self.read_through(receiver, &key)
                    }
                    _ => self.untyped_read(receiver, Reads::under(key_ty, false)),
                }
                for t in tys.iter().skip(1) {
                    self.escape(*t);
                }
            }
        }
    }

    /// A prefix and its suffixes, left to right: what each call passes
    /// is recorded against the function it reaches, and what each
    /// field read or method call goes through is noted.
    fn chain(&mut self, prefix: &Prefix, suffixes: &[&Suffix]) {
        self.prefix(prefix);
        // A reader taken out of the `table` library as a value.
        if let Prefix::Name(token) = prefix
            && let Some(Binding::Global(g)) = self.scopes.binding(token)
            && g == "table"
            && self.is_reader(g)
            && let Some(Suffix::Index(index)) = suffixes.first()
            && !(suffixes.len() >= 2
                && self
                    .typer()
                    .builtin_callee(prefix, &suffixes[..2])
                    .is_some())
            && constant_key(index).is_none_or(|m| {
                builtin_member(self.scopes, "table", &m).is_some_and(|b| library_reads(b).is_some())
            })
        {
            self.loose_reader();
        }
        for (i, s) in suffixes.iter().enumerate() {
            let receiver = self.typer().suffixed_ty(prefix, &suffixes[..i]);
            match s {
                Suffix::Call(ast::Call::AnonymousCall(args)) => {
                    if let Some(b) = self.typer().builtin_callee(prefix, &suffixes[..=i]) {
                        self.library_reads_of(b, prefix, args);
                    }
                    // The arguments of a direct call feed the callee's
                    // parameters, and so keep their types; a callee not
                    // typed yet may still turn out known. Anything else
                    // takes values.
                    let direct = matches!(receiver, Ty::Func(_) | Ty::Unknown);
                    self.args(args, direct);
                    if let Ty::Func(f) = receiver {
                        self.record_call(f, None, args);
                    }
                }
                Suffix::Call(ast::Call::MethodCall(m)) => {
                    let name = ident(m.name());
                    let targets = match receiver {
                        Ty::Shape(k) => self.typer().method_targets(k, &name),
                        Ty::Unknown => Some(Vec::new()),
                        _ => None,
                    };
                    match targets {
                        Some(targets) => {
                            self.args(m.args(), true);
                            for target in targets {
                                if let Ty::Func(f) = target {
                                    self.record_call(f, Some(receiver), m.args());
                                }
                            }
                        }
                        None => {
                            self.read_through(receiver, &name);
                            if let Ty::Shape(k) = receiver {
                                self.escape_method(k, &name);
                            }
                            self.args(m.args(), false);
                        }
                    }
                }
                Suffix::Index(index) => {
                    if let ast::Index::Brackets { expression, .. } = index {
                        self.value(expression);
                    }
                    match (constant_key(index), index) {
                        (Some(name), _) => self.read_through(receiver, &name),
                        (None, ast::Index::Brackets { expression, .. }) => {
                            let key_ty = self.typer().ty_of(expression);
                            self.keyed_read(receiver, key_ty);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    /// A direct call to `f`: what it passes joins the parameters.
    /// `receiver` is the object of a method call, passed first.
    fn record_call(&mut self, f: FuncId, receiver: Option<Ty>, args: &ast::FunctionArgs) {
        let exprs: Vec<&Expression> = match args {
            ast::FunctionArgs::Parentheses { arguments, .. } => arguments.iter().collect(),
            _ => Vec::new(),
        };
        let n = self.scopes.func(f).params.len();
        let mut types: Vec<Ty> = receiver.into_iter().collect();
        let wanted = n.saturating_sub(types.len());
        types.extend(match args {
            ast::FunctionArgs::String(_) => vec![Ty::Str],
            ast::FunctionArgs::TableConstructor(t) => vec![match constructor_shape(&self.out, t) {
                Some(k) => Ty::Shape(k),
                None => Ty::Table,
            }],
            _ => self.assigned_types(&exprs, wanted.max(exprs.len())),
        });
        self.record_call_types(f, types);
    }

    /// A direct call to `f` with arguments of `types`: they join the
    /// parameters; the extras go into `...` as values, or are dropped.
    fn record_call_types(&mut self, f: FuncId, mut types: Vec<Ty>) {
        let info = self.scopes.func(f);
        let n = info.params.len();
        let is_vararg = info.is_vararg;
        for t in types.iter().skip(n) {
            if is_vararg {
                self.escape(*t);
            }
        }
        types.resize(n, Ty::Nil);
        let params = self
            .out
            .funcs
            .get(&f)
            .map(|s| s.params.clone())
            .unwrap_or_else(|| vec![Ty::Unknown; n]);
        let params: Vec<Ty> = params
            .into_iter()
            .zip(types)
            .map(|(p, t)| self.join_into(p, t))
            .collect();
        let sig = self.out.funcs.entry(f).or_insert_with(|| Sig {
            params: vec![Ty::Unknown; n],
            returns: Returns::Fixed(Vec::new()),
        });
        sig.params = params;
    }

    fn prefix(&mut self, p: &Prefix) {
        if let Prefix::Expression(e) = p {
            self.expr(e);
        }
    }

    fn args(&mut self, args: &ast::FunctionArgs, kept: bool) {
        match args {
            ast::FunctionArgs::Parentheses { arguments, .. } => {
                for a in arguments {
                    if kept {
                        self.expr(a);
                    } else {
                        self.value(a);
                    }
                }
            }
            ast::FunctionArgs::TableConstructor(t) => self.table(t),
            _ => {}
        }
    }

    /// A constructor: its tables have a shape, and a constant-key
    /// field's value joins the shape's field. The rest are values.
    fn table(&mut self, t: &ast::TableConstructor) {
        let shape = self.shape_for(t);
        let count = t.fields().len();
        for (i, field) in t.fields().iter().enumerate() {
            match field {
                ast::Field::ExpressionKey { key, value, .. } => {
                    match crate::scope::literal_string(key) {
                        Some(name) => {
                            self.expr(value);
                            let ty = self.typer().ty_of(value);
                            self.store_field(shape, &name, ty);
                        }
                        None => {
                            self.value(key);
                            self.value(value);
                        }
                    }
                }
                ast::Field::NameKey { key, value, .. } => {
                    self.expr(value);
                    let ty = self.typer().ty_of(value);
                    self.store_field(shape, &ident(key), ty);
                }
                // A positional value is an element; the last one's
                // several values all are.
                ast::Field::NoKey(e) => {
                    self.expr(e);
                    let last = i + 1 == count;
                    let types = match multi_returns(&self.typer(), e) {
                        Some(Returns::Fixed(types)) if last => types,
                        Some(Returns::Dynamic) if last => vec![Ty::Any],
                        Some(Returns::Unknown) if last => vec![Ty::Unknown],
                        _ => vec![self.typer().ty_of(e)],
                    };
                    for ty in types {
                        self.store_element(shape, ty);
                    }
                }
                _ => {}
            }
        }
    }

    /// Walk an expression for the functions and calls inside it, in a
    /// place that keeps the value's type: assigned to a variable,
    /// returned, or passed to a known function.
    fn expr(&mut self, e: &Expression) {
        match e {
            Expression::BinaryOperator { lhs, binop, rhs } => {
                // Arithmetic the types dispatch to a handler passes
                // the operands to it; anything else takes values.
                let a = self.typer().ty_of(lhs);
                let b = self.typer().ty_of(rhs);
                let sides =
                    arith_event(binop).and_then(|event| self.typer().metamethod_sides(event, a, b));
                match sides {
                    Some(sides) => {
                        self.expr(lhs);
                        self.expr(rhs);
                        for (_, _, f) in sides {
                            self.record_call_types(f, vec![a, b]);
                        }
                    }
                    None => {
                        self.value(lhs);
                        self.value(rhs);
                    }
                }
            }
            Expression::Parentheses { expression, .. } => self.expr(expression),
            Expression::UnaryOperator { unop, expression } => {
                let t = self.typer().ty_of(expression);
                let sides = match unop {
                    UnOp::Minus(_) => self.typer().metamethod_sides("__unm", t, t),
                    _ => None,
                };
                match sides {
                    Some(sides) => {
                        self.expr(expression);
                        for (_, _, f) in sides {
                            self.record_call_types(f, vec![t, t]);
                        }
                    }
                    None => self.value(expression),
                }
            }
            Expression::Function(f) => self.function(f.body()),
            Expression::FunctionCall(c) => self.call(c),
            Expression::TableConstructor(t) => self.table(t),
            Expression::Var(Var::Expression(v)) => {
                let suffixes: Vec<&Suffix> = v.suffixes().collect();
                self.chain(v.prefix(), &suffixes);
            }
            Expression::Var(Var::Name(token)) => {
                if let Some(Binding::Global(name)) = self.scopes.binding(token)
                    && self.is_reader(name)
                {
                    self.loose_reader();
                }
            }
            _ => {}
        }
    }
}

/// The join of two things a read may yield, where one not decided yet
/// leaves the read undecided.
fn join_read(a: Ty, b: Ty) -> Ty {
    if a == Ty::Unknown || b == Ty::Unknown {
        Ty::Unknown
    } else {
        a.join(b)
    }
}

/// What a library function reads out of the table it takes first and
/// hands to its caller, or to a function it calls, as dynamic values:
/// `pairs` and `next` every field and element, raw; `ipairs`,
/// `table.unpack`, `table.remove` and `table.sort` elements, through
/// `__index`. None for one that hands out nothing of a table.
fn library_reads(b: &Builtin) -> Option<Reads> {
    match (b.lib, b.name) {
        ("", "pairs" | "next") => Some(Reads {
            names: true,
            elements: true,
            through_meta: false,
        }),
        ("", "ipairs") | ("table", "unpack" | "remove" | "sort") => {
            Some(Reads::under(Ty::Int, true))
        }
        _ => None,
    }
}

/// Whether a key of this type is a number: then a read under it is
/// of the array part, or of the hash part's numbers.
fn key_is_number(ty: Ty) -> bool {
    matches!(ty, Ty::Int | Ty::Float | Ty::Number)
}

/// Whether a key of this type may be a number: then a store under it
/// may land in a shape's array part.
fn key_may_be_index(ty: Ty) -> bool {
    !matches!(
        ty,
        Ty::Str | Ty::Bool | Ty::Nil | Ty::Table | Ty::Shape(_) | Ty::Func(_)
    )
}

/// Whether a key of this type may be a string: then a store under it
/// may land on a shape's field.
fn key_may_be_string(ty: Ty) -> bool {
    !matches!(
        ty,
        Ty::Int
            | Ty::Float
            | Ty::Number
            | Ty::Scalar
            | Ty::Bool
            | Ty::Nil
            | Ty::Table
            | Ty::Func(_)
            | Ty::Shape(_)
    )
}

/// The constant string an index suffix names: `.name` or `["name"]`.
pub fn constant_key(index: &ast::Index) -> Option<String> {
    match index {
        ast::Index::Dot { name, .. } => Some(ident(name)),
        ast::Index::Brackets { expression, .. } => crate::scope::literal_string(expression),
        _ => None,
    }
}

/// The constant string keys of a constructor, in order, and whether
/// the constructor has other fields too. `None` when it has an entry
/// that is neither a constant key nor positional.
pub fn constructor_keys(t: &ast::TableConstructor) -> Vec<String> {
    let mut keys = Vec::new();
    for field in t.fields() {
        match field {
            ast::Field::NameKey { key, .. } => keys.push(ident(key)),
            ast::Field::ExpressionKey { key, .. } => {
                if let Some(k) = crate::scope::literal_string(key) {
                    keys.push(k);
                }
            }
            _ => {}
        }
    }
    keys
}

/// The byte offset of a constructor's opening brace: what tells one
/// empty constructor from another.
pub fn constructor_site(t: &ast::TableConstructor) -> usize {
    crate::scope::pos_of(t.braces().tokens().0)
}

/// The shape a constructor's tables have, if one is known.
pub fn constructor_shape(known: &Inferred, t: &ast::TableConstructor) -> Option<ShapeId> {
    let keys = constructor_keys(t);
    if keys.is_empty() {
        known.shape_by_site.get(&constructor_site(t)).copied()
    } else {
        let mut sorted = keys;
        sorted.sort();
        sorted.dedup();
        known.shape_by_keys.get(&sorted).copied()
    }
}

/// What an expression in last position of a list supplies when it may
/// supply several values: a call's returns, or every vararg. `None`
/// for an expression of one value.
pub fn multi_returns(typer: &Typer<'_>, e: &Expression) -> Option<Returns> {
    match e {
        Expression::FunctionCall(c) => {
            let suffixes: Vec<&Suffix> = c.suffixes().collect();
            Some(typer.call_returns(c.prefix(), &suffixes))
        }
        Expression::Var(Var::Expression(v)) => {
            let suffixes: Vec<&Suffix> = v.suffixes().collect();
            if matches!(suffixes.last(), Some(Suffix::Call(_))) {
                Some(typer.call_returns(v.prefix(), &suffixes))
            } else {
                None
            }
        }
        Expression::Symbol(t) if t.token().to_string().trim() == "..." => Some(Returns::Dynamic),
        _ => None,
    }
}

/// Infer the chunk: rounds until the types settle, then everything
/// unknown is nil.
pub fn infer(scopes: &Scopes, ast: &ast::Ast) -> Inferred {
    let mut known = Inferred::default();
    // Every constructor's shape exists before the first round, so a
    // constructor is a shape from the start rather than a table for
    // a round, which would lose what is made from it for good.
    // A name bound straight to a constructor names its shape, and a
    // field stored through the name (`Class.method = f`, `function
    // Class.new()`) is a field of the shape from the start, so a read
    // of it before the store's round is undecided rather than nil.
    {
        struct Constructors<'a> {
            round: Round<'a>,
            named: HashMap<Named, ShapeId>,
        }
        #[derive(Clone, PartialEq, Eq, Hash)]
        enum Named {
            Var(VarId),
            Global(String),
        }
        impl Constructors<'_> {
            /// The variable or global a name token is: declared here,
            /// or used.
            fn named(&self, token: &full_moon::tokenizer::TokenReference) -> Option<Named> {
                if let Some(v) = self.round.scopes.decls.get(&crate::scope::pos_of(token)) {
                    return Some(Named::Var(*v));
                }
                match self.round.scopes.binding(token)? {
                    Binding::Local(v) | Binding::Upvalue(v) => Some(Named::Var(*v)),
                    Binding::Global(name) => Some(Named::Global(name.clone())),
                    Binding::Field(..) => None,
                }
            }
            fn constructor_of(e: &Expression) -> Option<&ast::TableConstructor> {
                match e {
                    Expression::TableConstructor(t) => Some(t),
                    Expression::Parentheses { expression, .. } => Self::constructor_of(expression),
                    // `setmetatable({...}, m)` is the constructor.
                    Expression::FunctionCall(c) => {
                        let suffixes: Vec<&Suffix> = c.suffixes().collect();
                        let Prefix::Name(name) = c.prefix() else {
                            return None;
                        };
                        if ident(name) != "setmetatable" {
                            return None;
                        }
                        let [
                            Suffix::Call(ast::Call::AnonymousCall(
                                ast::FunctionArgs::Parentheses { arguments, .. },
                            )),
                        ] = suffixes.as_slice()
                        else {
                            return None;
                        };
                        Self::constructor_of(arguments.iter().next()?)
                    }
                    _ => None,
                }
            }
            fn bind(&mut self, token: &full_moon::tokenizer::TokenReference, e: &Expression) {
                if let Some(t) = Self::constructor_of(e)
                    && let Some(n) = self.named(token)
                {
                    let k = self.round.shape_for(t);
                    self.named.insert(n, k);
                }
            }
            fn field_of(&mut self, token: &full_moon::tokenizer::TokenReference, name: &str) {
                if let Some(n) = self.named(token)
                    && let Some(k) = self.named.get(&n).copied()
                {
                    self.round.out.shapes[k.0 as usize]
                        .fields
                        .entry(name.to_string())
                        .or_insert(Ty::Unknown);
                }
            }
        }
        impl full_moon::visitors::Visitor for Constructors<'_> {
            fn visit_table_constructor(&mut self, t: &ast::TableConstructor) {
                self.round.shape_for(t);
            }
            fn visit_local_assignment(&mut self, l: &ast::LocalAssignment) {
                for (name, e) in l.names().iter().zip(l.expressions().iter()) {
                    self.bind(name, e);
                }
            }
            fn visit_assignment(&mut self, a: &ast::Assignment) {
                for (target, e) in a.variables().iter().zip(a.expressions().iter()) {
                    match target {
                        Var::Name(name) => self.bind(name, e),
                        Var::Expression(v) => {
                            let suffixes: Vec<&Suffix> = v.suffixes().collect();
                            if let (Prefix::Name(name), [Suffix::Index(index)]) =
                                (v.prefix(), suffixes.as_slice())
                                && let Some(key) = constant_key(index)
                            {
                                self.field_of(name, &key);
                            }
                        }
                        _ => {}
                    }
                }
            }
            fn visit_function_declaration(&mut self, f: &ast::FunctionDeclaration) {
                let names: Vec<_> = f.name().names().iter().collect();
                match (names.as_slice(), f.name().method_name()) {
                    ([owner, field], None) => self.field_of(owner, &ident(field)),
                    ([owner], Some(method)) => self.field_of(owner, &ident(method)),
                    _ => {}
                }
            }
        }
        let empty = Inferred::default();
        let mut visitor = Constructors {
            round: Round {
                scopes,
                known: &empty,
                out: Inferred::default(),
                func: CHUNK,
                returns: None,
                blind_reads: HashSet::new(),
                blind_keyed_reads: Reads::default(),
                loop_iterator: None,
                blind_getmetatable: false,
            },
            named: HashMap::new(),
        };
        full_moon::visitors::Visitor::visit_ast(&mut visitor, ast);
        known.shapes = visitor.round.out.shapes;
        known.shape_by_keys = visitor.round.out.shape_by_keys;
        known.shape_by_site = visitor.round.out.shape_by_site;
    }
    // Two fixed points, one inside the other. The inner one types the
    // chunk with a given set of escaping functions, whose parameters
    // take anything, and notes what else escapes; the outer one adds
    // those and types again. An escape is thus always read off a
    // settled typing, never off a round that had a type wrong.
    let mut escaping: HashSet<FuncId> = HashSet::new();
    loop {
        known = infer_given(scopes, ast, &known, &escaping);
        if known.escaping.is_subset(&escaping) {
            break;
        }
        escaping.extend(known.escaping.iter().copied());
    }
    for ty in known.vars.values_mut() {
        *ty = ty.settled();
    }
    for ty in known.globals.values_mut() {
        *ty = ty.settled();
    }
    for sig in known.funcs.values_mut() {
        for p in &mut sig.params {
            *p = p.settled();
        }
        sig.returns = sig.returns.clone().settled();
    }
    for shape in &mut known.shapes {
        for ty in shape.fields.values_mut() {
            *ty = ty.settled();
        }
    }
    trace_types(scopes, &known);
    known
}

/// Rounds until the types settle, with `escaping` the functions whose
/// parameters take anything. `seed` carries the shapes' identities.
fn infer_given(
    scopes: &Scopes,
    ast: &ast::Ast,
    seed: &Inferred,
    escaping: &HashSet<FuncId>,
) -> Inferred {
    let mut known = Inferred {
        shapes: seed.shapes.clone(),
        shape_by_keys: seed.shape_by_keys.clone(),
        shape_by_site: seed.shape_by_site.clone(),
        escaping: escaping.clone(),
        ..Default::default()
    };
    for _ in 0..16 {
        let mut round = Round {
            scopes,
            known: &known,
            out: Inferred::default(),
            func: CHUNK,
            returns: None,
            blind_reads: HashSet::new(),
            blind_keyed_reads: Reads::default(),
            loop_iterator: None,
            blind_getmetatable: false,
        };
        // The signatures from the last round carry over so a recursive
        // call in this one sees them; params are rejoined from calls.
        // The escapes given are the round's starting point; what it
        // finds is added for the next fixed point. The shapes keep
        // their ids and their fields' order; what is stored in them,
        // their classes and what reaches them are found again.
        round.out.escaping = escaping.clone();
        round.out.shape_by_keys = known.shape_by_keys.clone();
        round.out.shape_by_site = known.shape_by_site.clone();
        round.out.shapes = known
            .shapes
            .iter()
            .map(|s| ShapeInfo {
                fields: s.fields.keys().map(|k| (k.clone(), Ty::Unknown)).collect(),
                born: s.born,
                line: s.line,
                ..Default::default()
            })
            .collect();
        for (f, sig) in &known.funcs {
            round.out.funcs.insert(
                *f,
                Sig {
                    params: vec![Ty::Unknown; sig.params.len()],
                    returns: sig.returns.clone(),
                },
            );
        }
        round.block(ast.nodes());
        round.settle_shapes();
        let out = round.out;
        if std::env::var_os("ZYNTAX_TRACE_ROUNDS").is_some() {
            eprintln!(
                "[round] shapes {:?}",
                out.shapes.iter().map(|s| &s.fields).collect::<Vec<_>>()
            );
            let mut funcs: Vec<(&FuncId, &Sig)> = out.funcs.iter().collect();
            funcs.sort_by_key(|(f, _)| **f);
            for (f, sig) in funcs {
                eprintln!(
                    "[round]   {} {:?} -> {:?}",
                    scopes.func(*f).name,
                    sig.params,
                    sig.returns
                );
            }
            let mut vars: Vec<(&VarId, &Ty)> = out.vars.iter().collect();
            vars.sort_by_key(|(v, _)| **v);
            eprintln!(
                "[round]   vars {:?}",
                vars.iter()
                    .map(|(v, t)| format!("{}={:?}", scopes.var(**v).name, t))
                    .collect::<Vec<_>>()
            );
        }
        if out == known {
            break;
        }
        known = out;
    }
    known
}

fn trace_types(scopes: &Scopes, known: &Inferred) {
    if std::env::var_os("ZYNTAX_TRACE_TYPES").is_some() {
        for (i, shape) in known.shapes.iter().enumerate() {
            eprintln!(
                "[types] shape {i}@{}: {:?} born={} element={:?} classes={:?}{}{}{}{}",
                shape.line,
                shape.fields,
                shape.born,
                shape.element,
                shape.classes,
                if shape.escapes { " escapes" } else { "" },
                if shape.unknown_meta {
                    " unknown-meta"
                } else {
                    ""
                },
                if shape.dynamic_keys {
                    format!(" dynamic-keys={:?}", shape.dyn_value)
                } else {
                    String::new()
                },
                match shape.untyped_reads {
                    Reads {
                        names: false,
                        elements: false,
                        ..
                    } => String::new(),
                    r => format!(
                        " untyped-reads{}{}{}",
                        if r.names { ":names" } else { "" },
                        if r.elements { ":elements" } else { "" },
                        if r.through_meta { ":meta" } else { "" },
                    ),
                },
            );
        }
        eprintln!(
            "[types] blind stores {:?} dynamic={} element={:?} setmetatable={}",
            known.blind_stores,
            known.blind_dynamic_stores,
            known.blind_element,
            known.blind_setmetatable
        );
        let mut escaping: Vec<String> = known
            .escaping
            .iter()
            .map(|f| format!("{}@{}", scopes.func(*f).name, scopes.func(*f).line))
            .collect();
        escaping.sort();
        eprintln!("[types] escaping {escaping:?}");
        let mut funcs: Vec<(&FuncId, &Sig)> = known.funcs.iter().collect();
        funcs.sort_by_key(|(f, _)| **f);
        for (f, sig) in funcs {
            let info = scopes.func(*f);
            eprintln!(
                "[types] {}@{} ({:?}) -> {:?}",
                info.name, info.line, sig.params, sig.returns
            );
        }
        let mut vars: Vec<(&VarId, &Ty)> = known.vars.iter().collect();
        vars.sort_by_key(|(v, _)| **v);
        for (v, ty) in vars {
            let info = scopes.var(*v);
            eprintln!("[types] local {} #{}: {ty:?}", info.name, v.0);
        }
        let mut globals: Vec<(&String, &Ty)> = known.globals.iter().collect();
        globals.sort_by(|a, b| a.0.cmp(b.0));
        for (name, ty) in globals {
            eprintln!("[types] global {name}: {ty:?}");
        }
    }
}
