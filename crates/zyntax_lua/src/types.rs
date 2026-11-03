//! Static types for a dynamically typed program.
//!
//! A variable's type is the join of everything assigned to it, over the
//! whole chunk; a function's parameter types are the join of what its
//! direct calls pass; its result is the join of what it returns. The
//! join of two different types is the dynamic value, so a variable
//! holding an integer everywhere is an `i64` and one holding an
//! integer here and a string there is boxed. Rounds repeat until
//! nothing changes: a recursive function's result depends on itself.

mod builtins;
mod calls;
mod constructors;
mod expressions;
mod infer;
mod keys;
mod operators;
mod settle;
mod shapes;
mod statements;
mod stores;

pub(crate) use builtins::*;
pub(crate) use infer::*;
pub(crate) use keys::*;
pub(crate) use operators::*;

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
    /// An integer or nil: a scalar whose tag is only ever nil or
    /// integer. Arithmetic on it raises for nil, else is the integer's.
    IntOrNil,
    /// A float or nil, as `IntOrNil` is for an integer.
    FloatOrNil,
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
            (Ty::Int | Ty::IntOrNil, Ty::Nil | Ty::IntOrNil)
            | (Ty::Nil | Ty::IntOrNil, Ty::Int | Ty::IntOrNil) => Ty::IntOrNil,
            (Ty::Float | Ty::FloatOrNil, Ty::Nil | Ty::FloatOrNil)
            | (Ty::Nil | Ty::FloatOrNil, Ty::Float | Ty::FloatOrNil) => Ty::FloatOrNil,
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
            Ty::Nil
                | Ty::Scalar
                | Ty::IntOrNil
                | Ty::FloatOrNil
                | Ty::Any
                | Ty::Unknown
                | Ty::Shape(_)
                | Ty::Func(_)
        )
    }

    /// Carried unboxed as a tagged scalar, or convertible to one.
    pub fn is_scalar(self) -> bool {
        matches!(
            self,
            Ty::Nil
                | Ty::Bool
                | Ty::Int
                | Ty::Float
                | Ty::Number
                | Ty::Scalar
                | Ty::IntOrNil
                | Ty::FloatOrNil
        )
    }

    /// Carried as a tagged scalar that may be nil: arithmetic and
    /// ordering on it check the tag first.
    pub fn is_tagged(self) -> bool {
        matches!(self, Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil)
    }

    /// The number an integer-or-nil or float-or-nil holds when it is
    /// not nil; any other type itself.
    pub fn underlying(self) -> Ty {
        match self {
            Ty::IntOrNil => Ty::Int,
            Ty::FloatOrNil => Ty::Float,
            t => t,
        }
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

/// The receiver at a tracked metatable write.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MetaReceiver {
    Built,
    Local(VarId),
    Other,
}

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
    /// Receiver provenance of every tracked metatable write.
    pub meta_receivers: BTreeSet<MetaReceiver>,
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
    /// The join of what stores under keys not known at compile time,
    /// that may be no number, put in a table of this shape: what a name
    /// that is no field may hold, when `dynamic_keys`, and what a key
    /// that is neither a string nor a number may.
    pub dyn_value: Ty,
    /// Reads out of a table of this shape whose results the types do
    /// not follow: what they may yield escapes.
    pub untyped_reads: Reads,
    /// The line the first constructor is on, for naming.
    pub line: usize,
    /// The constructors making tables of this shape, by the byte offset
    /// of their brace.
    pub sites: BTreeSet<usize>,
    /// Whether one of them may run more than once: it stands in a
    /// function or a loop, or in a chunk that jumps or is called.
    pub remade: bool,
    /// The fields a store may leave nil: given a value that may be nil,
    /// at birth or later.
    pub nil_stores: BTreeSet<String>,
    /// The constructors built inside `setmetatable` with a metatable
    /// that is never nil: their tables have it from birth.
    pub born_meta: BTreeSet<usize>,
    /// Whether `setmetatable` may leave a table of this shape without a
    /// metatable.
    pub meta_cleared: bool,
}

/// Reads out of a table that hand what they find to code the types do
/// not follow: under keys not known at compile time, or by a library
/// function that walks the table.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Reads {
    /// Under a key that may be no number: any field, any name, or
    /// what is held under a key of another kind.
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
            names: !key_is_number(key_ty),
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
    /// The `__index` field of the table, of shape `.0`, holds function
    /// `.1`: the lookup ends in calling it with the table the last
    /// `Meta` hop started from and the name.
    Handler(ShapeId, FuncId),
}

/// Where a lookup of a name may end: the field of a table of shape
/// `shape`, reached by `hops`, holding `ty`; `sure` when the field is
/// always there. A lookup whose last hop is a `Handler` ends in the
/// handler's call instead: `shape` is then the table it is called
/// with, and `ty` its first result.
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
    /// What the chunk's `return` statements return; `None` when it has
    /// none.
    pub chunk: Option<Returns>,
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
    /// Whether a read of a field or element nothing typed has been
    /// stored in yields nil, as a table with nothing stored there
    /// does, rather than deciding nothing. Set once the rounds settle
    /// with such reads undecided: a store of what such a read yields
    /// is then typed from it.
    pub settling: bool,
    /// The locals that never hold nil: declared with a value that is
    /// never nil and never assigned again.
    pub non_nil_vars: HashSet<VarId>,
}

/// What a builtin does with the tables among its arguments, beyond
/// looking at them: whether they may be borrowed, so a table passed
/// keeps its shape's types.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Borrows {
    /// It may keep its arguments or hand them to code: they escape.
    No,
    /// It looks at its arguments raw and keeps none: all are borrowed.
    Raw,
    /// It reads its first argument's elements as an index does, so a
    /// metatable's `__index` may be called with it: borrowed when the
    /// table has no metatable.
    Indexed,
    /// It walks its first argument for a generic `for`: borrowed when
    /// the loop's bindings are typed from the table's shape.
    Walk,
    /// `select`: its arguments are borrowed when it counts them.
    Count,
}

/// Types expressions against what the last round settled.
pub struct Typer<'a> {
    pub scopes: &'a Scopes,
    pub known: &'a Inferred,
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
    /// How many loops enclose what is being walked in its function.
    loops: usize,
    /// Whether the main chunk's own code holds a `goto` or a label, so
    /// a statement of it may run more than once.
    chunk_jumps: bool,
}
