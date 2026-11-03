//! Lua's AST rewritten as typed AST.
//!
//! Every expression is lowered with the static type inference gave it:
//! arithmetic on two integers is an `i64` add, on two dynamic values a
//! call into the library's `zl_arith`. Crossings are explicit: a typed
//! value is boxed where a dynamic one is needed, and a dynamic value
//! known to hold a type is read back without a check, because
//! inference only settles a type it saw every assignment of.
//!
//! Functions become typed entries, called directly where the callee is
//! known; one used as a value gets a record of the shape every function
//! value has, whose code adapts the record's dynamic arguments to the
//! typed entry. Closures capture through cells or copies in the record.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};

use full_moon::ast::{self, BinOp, Block, Expression, Prefix, Stmt, Suffix, UnOp, Var};
use full_moon::tokenizer::TokenReference;

use zyntax_builtins::functions::{VARIADIC_ARITY, arity_word};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{
    BinaryOp, ParamOwnership, TypedBinary, TypedBlock, TypedCall, TypedCast, TypedDeclaration,
    TypedExpression, TypedFieldAccess, TypedFieldInit, TypedFunction, TypedIf, TypedIfExpr,
    TypedIndex, TypedLet, TypedLiteral, TypedParameter, TypedStatement, TypedStructLiteral,
    TypedUnary, TypedVariable, TypedWhile, UnaryOp,
};
use zyntax_typed_ast::{
    InternedString, Mutability, PrimitiveType, Type, TypedNode, TypedProgram, Visibility,
};

use crate::library::stdlib::{Builtin, Param, Ret};
use crate::library::values::*;
use crate::library::{self, Types};
use crate::scope::{Binding, CHUNK, FuncId, Scopes, VarId};
use crate::types::{self, Hop, Inferred, Returns, ShapeId, Ty, Typer, ident};
use crate::{ENTRY, Error, Library, Result, intern, prim};
use zyntax_typed_ast::TypeId;

mod binary;
mod blocks;
mod builders;
mod builtin_args;
mod builtins;
mod coerce;
mod compare;
mod constructors;
mod context;
mod debug;
mod direct_calls;
mod dispatch;
mod dynamic_for;
mod errors;
mod exact;
mod expressions;
mod field_read;
mod field_write;
mod finders;
mod functions;
mod iterators;
mod line_restores;
mod loaded;
mod method_calls;
mod method_lookup;
mod modules;
mod numeric;
mod numeric_for;
mod pairs;
mod program;
mod program_debug;
mod scalars;
mod shape_hooks;
mod slots;
mod sort;
mod statements;
mod strings;
mod suffixes;
mod unary;
mod upvalues;
mod value_calls;
mod variables;

use builders::*;
use context::*;
pub(crate) use loaded::{error_text_program, loaded_program, loaded_types};
use modules::*;
pub(crate) use program::program;
use program_debug::*;
use shape_hooks::*;
pub(crate) use strings::string_bytes;
use strings::*;
use upvalues::*;

type Node = TypedNode<TypedExpression>;
type St = TypedNode<TypedStatement>;

/// A lowered expression and the static type it has.
#[derive(Clone)]
struct Val {
    node: Node,
    ty: Ty,
}

/// What a value is called in a type error about it, `local 'x'`,
/// `field 'y'` and the like; nothing for a temporary.
type Desc = Option<String>;

/// What a site's check describes a type error with: the names of the
/// operands, and whether the site is a call, whose error names the
/// callee, rather than an operator or index, whose error names an
/// operand.
struct Described {
    operands: [Desc; 2],
    call: bool,
}

impl Described {
    const NONE: Described = Described {
        operands: [None, None],
        call: false,
    };
    fn operand(desc: Desc) -> Described {
        Described {
            operands: [desc, None],
            call: false,
        }
    }
    fn operands(left: Desc, right: Desc) -> Described {
        Described {
            operands: [left, right],
            call: false,
        }
    }
    fn callee(desc: Desc) -> Described {
        Described {
            operands: [desc, None],
            call: true,
        }
    }
}

/// What an expression in last position produces: a fixed number of
/// values, or one dynamic value that may hold several, or nothing: a
/// call that runs for its effects.
enum Multi {
    Fixed(Vec<Val>),
    Dynamic(Node),
    None(Node),
}

/// Where a variable lives in the function being lowered.
#[derive(Clone)]
enum Storage {
    /// A local of the function, or a parameter.
    Local(InternedString, Ty),
    /// A one-element list shared with closures.
    Cell(InternedString, Ty),
    /// A module-level variable.
    Module(InternedString, Ty),
}

/// Where the record of a captured variable sits: `env[RECORD_CELLS_AT +
/// i]` for the `i`th capture of the function. Before the captures, after
/// the code and the arity every record has, sits the function's number:
/// what tells a call site which known function a value is, since the
/// code's address changes as the function is compiled again.
const RECORD_CELLS_AT: usize = 3;

/// The whole program as the lowerers share it.
struct Module<'a> {
    scopes: &'a Scopes,
    inferred: &'a Inferred,
    types: Types,
    /// The chunk's name, as error positions spell it.
    chunk: &'a str,
    /// What this chunk's symbols carry after `lua$`: nothing for the
    /// main chunk, `m$<name>$` for a required file, so two files'
    /// functions and locals never share a name.
    tag: String,
    /// This chunk's number in the bits above `LINE_BITS` of a stored
    /// line: 0 for the main chunk.
    chunk_index: i64,
    /// The module variable holding this chunk's globals table when it
    /// has one of its own (a chunk loaded with an env), else the
    /// program's.
    env_var: Option<InternedString>,
    /// Byte offsets where each line starts, for positions.
    line_starts: Vec<usize>,
    /// Loaded from a binary chunk without debug information: its
    /// statements are on no line and its variables have no names.
    stripped: bool,
    /// The library functions that can raise.
    fallible: HashSet<&'static str>,
    /// The library functions that may run the program's code before
    /// they return.
    reentrant: HashSet<&'static str>,
    /// The program's functions that can raise, once a first lowering
    /// has found out; every one, before.
    raising: Option<HashSet<FuncId>>,
    /// The program's functions whose calls cannot go deeper than their
    /// own callees, once a first lowering has found out: no recursion
    /// through direct calls and no call through a value or the
    /// library. They keep no count of their depth.
    bounded: Option<HashSet<FuncId>>,
    /// Functions lowered so far, in the order they were reached.
    functions: RefCell<Vec<TypedFunction>>,
    /// Module-level variables: globals, and chunk locals every function
    /// reaches. Name and type.
    module_vars: RefCell<Vec<(InternedString, Ty)>>,
    /// What each function's lowering found about its raising.
    facts: RefCell<HashMap<FuncId, RaiseFact>>,
    /// Constructor facts proved with the field presence at their declaration.
    fresh_classes: RefCell<HashMap<FuncId, ShapeId>>,
    /// Locals initialized by distinct fresh calls, for metatable-write isolation.
    fresh_locals: RefCell<HashSet<VarId>>,
    /// Functions whose first result is their unchanged first argument.
    returns_first: RefCell<HashSet<FuncId>>,
    /// How each shape's tables are laid out, by shape; none for a shape
    /// whose tables are plain.
    layouts: Vec<Option<ShapeLayout>>,
    /// The finder made for each (shape, name, mode) looked up through
    /// metatables so far; none when its lookups cannot be checked.
    finders: RefCell<HashMap<(ShapeId, String, FinderMode), Option<Finder>>>,
    /// The helpers made here that may run the program's code before
    /// they return: a call through them is a dynamic call.
    reentrant_helpers: RefCell<HashSet<String>>,
    /// The sort made for each (shape, comparator) `table.sort` is
    /// called with.
    sorts: RefCell<HashMap<(ShapeId, FuncId), String>>,
    /// What the debug library is told of the chunk, when the program
    /// keeps its call stack.
    debug: Option<DebugInfo>,
    /// Whether this chunk's outermost code runs once, as a program's
    /// main chunk does: a field stored there in a table made once is
    /// then there for good wherever the store has run.
    runs_once: bool,
}

/// What a chunk of a program that keeps its call stack records: the
/// records the host reads (see `host_debug`), and how many call sites
/// are numbered so far.
#[derive(Default)]
struct DebugInfo {
    /// Whether frames spill their locals at each call, for
    /// `debug.getlocal`.
    locals: bool,
    /// Whether a call may change the caller's locals, which it reads
    /// back afterwards.
    setlocal: bool,
    records: RefCell<Vec<String>>,
    sites: std::cell::Cell<i64>,
}

/// How a shape's tables are laid out: the table header, then one slot
/// per field, each stored as its type. The struct is registered as
/// `name` under `id`; `gid` is the number a table carries in its
/// header, unique across the program's chunks.
struct ShapeLayout {
    id: TypeId,
    name: String,
    gid: i64,
    slots: Vec<Slot>,
}

/// A table slot. A `Number` slot is two words, `s{bit}i` and
/// `s{bit}f`, with its kind in bit `kbit` of the header's `present`
/// word. While its presence bit is set, the kind bit is set exactly
/// when the value is the integer in `i` and clear exactly when it is
/// the float in `f`; the other word is unspecified (a float store
/// writes `f` alone), so every reader selects its word by the kind.
#[derive(Clone)]
struct Slot {
    name: String,
    /// The slot's bit in the header's `present` word.
    bit: usize,
    kind: SlotKind,
    /// A `Number` slot's kind bit in the `present` word, above every
    /// presence bit.
    kbit: Option<usize>,
}

/// What a slot stores.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlotKind {
    Int,
    Float,
    Bool,
    Str,
    Table,
    /// An integer or a float: two fields and a kind bit.
    Number,
    /// A tagged scalar: three fields.
    Scalar,
    /// A boxed value, null for nil.
    Any,
}

impl Slot {
    fn kind_of(ty: Ty) -> SlotKind {
        match ty {
            Ty::Int => SlotKind::Int,
            Ty::Float => SlotKind::Float,
            Ty::Bool => SlotKind::Bool,
            Ty::Str => SlotKind::Str,
            Ty::Table | Ty::Shape(_) => SlotKind::Table,
            Ty::Number => SlotKind::Number,
            Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => SlotKind::Scalar,
            Ty::Nil | Ty::Func(_) | Ty::Any | Ty::Unknown => SlotKind::Any,
        }
    }

    /// The type a slot's value is read as, and stored as: its kind's.
    fn stored_ty(&self) -> Ty {
        match self.kind {
            SlotKind::Int => Ty::Int,
            SlotKind::Float => Ty::Float,
            SlotKind::Bool => Ty::Bool,
            SlotKind::Str => Ty::Str,
            SlotKind::Table => Ty::Table,
            SlotKind::Number => Ty::Number,
            SlotKind::Scalar => Ty::Scalar,
            SlotKind::Any => Ty::Any,
        }
    }

    /// The struct fields the slot occupies.
    fn fields(&self, table_ty: &Type) -> Vec<(String, Type)> {
        let base = format!("s{}", self.bit);
        match self.kind {
            SlotKind::Int => vec![(base, prim(PrimitiveType::I64))],
            SlotKind::Float => vec![(base, prim(PrimitiveType::F64))],
            SlotKind::Bool => vec![(base, prim(PrimitiveType::Bool))],
            SlotKind::Str => vec![(base, prim(PrimitiveType::String))],
            SlotKind::Table => vec![(base, table_ty.clone())],
            SlotKind::Any => vec![(base, Type::Any)],
            SlotKind::Number => vec![
                (format!("{base}i"), prim(PrimitiveType::I64)),
                (format!("{base}f"), prim(PrimitiveType::F64)),
            ],
            SlotKind::Scalar => vec![
                (format!("{base}t"), prim(PrimitiveType::I64)),
                (format!("{base}i"), prim(PrimitiveType::I64)),
                (format!("{base}f"), prim(PrimitiveType::F64)),
            ],
        }
    }

    fn mask(&self) -> i64 {
        1i64 << self.bit
    }

    /// The mask of a `Number` slot's kind bit.
    fn kmask(&self) -> i64 {
        1i64 << self.kbit.expect("a number slot has a kind bit")
    }
}

/// The most slots a shape has: the bits of the header's `present` word.
const MAX_SLOTS: usize = 63;

/// The lookups of a name from a shape's tables, checked by one
/// function, `helper(t)`, in the order the runtime looks. Its answer
/// depends on the mode. `ends` are where the lookups may end, each
/// with the shape of the table it ends at.
#[derive(Clone)]
struct Finder {
    helper: String,
    ends: Vec<(ShapeId, End)>,
}

/// How a call site finds which end its lookup takes.
enum Dispatch {
    /// By calling the `Which` finder.
    Finder(String),
    /// Decided where it stands.
    Folded(FoldPlan),
}

/// A call site's lookup decided where it stands.
struct FoldPlan {
    /// The end taken by the metatable's shape, by shape id, none taking
    /// any; or none when every table takes end 0.
    arms: Option<Vec<(i64, i64)>>,
    /// Whether the ends are fields of the metatable, rather than of the
    /// table itself.
    via_meta: bool,
}

/// Where a lookup surely ends at the place it is made.
enum Folded {
    /// In the table's own field, holding this type.
    Own(Ty),
    /// By the shape of the table's metatable: for each, the table the
    /// lookup ends in, by shape, and the field's type; none when the
    /// name is nowhere along the way.
    Classes(Vec<(ShapeId, Option<(ShapeId, Ty)>)>),
}

/// What a finder answers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum FinderMode {
    /// The table holding the field, reached through metatables, or
    /// null: the lookups through the table's own field are left out,
    /// and a nil receiver gives null.
    Table,
    /// The index in `ends` of the end the lookup takes, or -1 when it
    /// takes none. Reads no value, calls nothing; the receiver is
    /// not nil.
    Which,
    /// The table the lookup ends at, or null: the one holding the
    /// field, or the one an `__index` function is called with. The
    /// receiver is not nil.
    Where,
}

/// Where a lookup ends.
#[derive(Clone, Copy, PartialEq, Debug)]
enum End {
    /// The field, in a slot, holding this type.
    Slot(Ty),
    /// A call of the `__index` function `f` held by a metatable of
    /// shape `class`, whose first result has type `ty`.
    Handler { class: ShapeId, f: FuncId, ty: Ty },
}

/// What one lowering of a function found: whether it raises itself
/// (its depth check aside), whether it checks its depth, whether it
/// calls anything that may run the program's code before returning,
/// and which functions it calls directly.
#[derive(Clone, Default)]
struct RaiseFact {
    own: bool,
    checks_depth: bool,
    dynamic: bool,
    callees: HashSet<FuncId>,
}

/// A function being lowered.
struct Lowerer<'m, 'a> {
    m: &'m Module<'a>,
    func: FuncId,
    storage: HashMap<VarId, Storage>,
    /// The list of extra arguments, for a variadic function.
    varargs: Option<InternedString>,
    returns: Returns,
    temps: usize,
    /// Names already bound by a `let` in this function, so a second
    /// declaration of a Lua local of the same name is a fresh symbol.
    bound: std::collections::HashSet<InternedString>,
    /// Whether this function checks for an error anywhere, so it may
    /// leave with one pending.
    raised: bool,
    /// The program functions this one calls directly and that may
    /// raise into it.
    raise_callees: HashSet<FuncId>,
    /// Whether the statement being lowered checks for an error, so its
    /// line is recorded ahead of it.
    line_needed: bool,
    /// Whether the body reads the line it was entered at, for
    /// `error(v, 2)`.
    entry_line: bool,
    /// The `<close>` variables in scope, each with the depth of the
    /// block declaring it, innermost last. Leaving a block closes its
    /// variables in reverse order.
    tbc: Vec<(usize, Closable)>,
    /// Whether this function counts itself in the call depth: every
    /// program function does, so that recursion without end is an
    /// error to catch; a chunk does not.
    counts_depth: bool,
    /// How many blocks are open, the function's body counting as one.
    depth: usize,
    /// The depth of each enclosing loop's body, and where the loop
    /// ends: a `break` closes from that line.
    loop_depths: Vec<(usize, Span)>,
    /// Whether this is a function of the program that keeps a frame on
    /// the debug library's stack: set by the lowering of a function or
    /// chunk body in a program that keeps one, never for a helper.
    frames: bool,
    /// The locals in scope, in the order Lua numbers them.
    live: Vec<Live>,
    /// The values of the hidden `for` locals in scope, boxed, which
    /// [`Live::ForState`] indexes.
    for_states: Vec<Node>,
    /// The lines statements of this function start on.
    lines: std::collections::BTreeSet<i64>,
    /// The span of the call a `return` makes as a tail call.
    tail_span: Option<Span>,
    /// The `end` of a function's body, where the variables its
    /// outermost block declares are closed.
    end_span: Option<Span>,
    /// Fields of tables made once that are surely there: on every path
    /// to this point a store of a value that is never nil put them
    /// there, and they are never taken away.
    present: BTreeSet<(ShapeId, String)>,
    exact_classes: HashMap<VarId, ShapeId>,
}

/// A local in scope, as `debug.getlocal` numbers them: a variable, or
/// the hidden state of a `for` loop, its value in `for_states`.
#[derive(Clone, Copy)]
enum Live {
    Var(VarId),
    ForState(usize),
}

/// What leaving a block closes: a `<close>` variable, or the closing
/// value of a generic `for`, held in a temporary. A loop's closing
/// value is entered at its body's depth but outlives each pass of the
/// body: the loop closes it once it ends, and only a jump out of the
/// loop past that closes it on the way.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Closable {
    Var(VarId),
    Loop(InternedString),
}

/// A target of a multiple assignment: a name or a global, stored as
/// a whole, or an indexed place whose table and key were evaluated
/// ahead of the values.
enum Prepared<'e> {
    Whole(&'e Var),
    Index(Box<(Val, Val, Desc)>),
}

/// What is known of a `//` or `%` divisor from its literal.
#[derive(Clone, Copy)]
struct Divisor {
    /// A literal that is not 0 or -1: the integer path cannot raise.
    plain: bool,
    /// A literal positive power of two.
    pow2: Option<i64>,
}

/// What a dispatch yields on every path: a fixed number of results,
/// each typed, or a dynamic value.
enum Yield {
    Fixed(Vec<Ty>),
    Dynamic,
}

/// How a call through a value tells which known function it is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Guard {
    /// The value's code against each function's.
    ByCode,
    /// The value is not nil: it can only be the one function.
    NotNil,
}

/// How a call names what it calls, for the error calling nil raises:
/// `obj:name()` a method, `obj.name()` a field. The value is what
/// `zl_raise_call_nil` takes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CallKind {
    Method = 0,
    Field = 1,
}

/// How two numbers are compared.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Compare {
    Eq,
    Lt,
    Le,
}

thread_local! {
    /// Every name a chunk being lowered refers to, while one is: what
    /// it needs declared of the library.
    static NAMED: std::cell::RefCell<Option<std::collections::HashSet<InternedString>>> =
        const { std::cell::RefCell::new(None) };
}

/// Collect the library symbols lowering refers to, restoring an outer
/// collection when lowering returns early with an error.
struct NamedReferences {
    previous: Option<HashSet<InternedString>>,
    active: bool,
}

impl Drop for NamedReferences {
    fn drop(&mut self) {
        if self.active {
            NAMED.with(|named| {
                named.replace(self.previous.take());
            });
        }
    }
}

/// The tags of a run-time scalar. A boolean's truth is its tag, so a
/// scalar is true from `TAG_TRUE` up and a number from `TAG_INT` up.
const TAG_NIL: i64 = 0;
const TAG_FALSE: i64 = 1;
const TAG_TRUE: i64 = 2;
const TAG_INT: i64 = 3;
const TAG_FLOAT: i64 = 4;

/// The parts of a scalar that is a plain read.
struct NumberParts {
    tag: Node,
    int: Node,
    float: Node,
}

/// The chunk's statements as a function; the entry names the chunk,
/// runs it, and reports an error nothing caught.
const CHUNK_FN: &str = "lua$chunk";

/// The restores of the line after checked calls that nothing reads:
/// removed where every path from the restore stores the line again, or
/// returns, before a raise or anything else that reads it. Walks a
/// function body backwards, tracking whether the line is still to be
/// read (`later`); what reads it is a check (the raising call before
/// it), a read of the line itself, and a library call that may raise
/// or run the program's code.
struct LineRestores<'a> {
    reads_line: &'a dyn Fn(&str) -> bool,
    line: InternedString,
    pending: InternedString,
    /// Whether the line is read after each enclosing loop, innermost
    /// last: where a `break` goes.
    exits: Vec<bool>,
}

/// Set by a `return` at the chunk's outermost level when the chunk
/// runs as segments, so the driver stops.
const RETURNED: &str = "lua$returned";

/// The line a function was entered at: the caller's, for `error(v, 2)`.
const ENTRY_LINE: &str = "$entry_line";
/// The script's arguments, `...` at the main chunk.
const MAIN_VARARGS: &str = "lua$varargs";
/// The main chunk's environment, when the program assigns `_ENV`.
const MAIN_ENV: &str = "lua$env";

/// A file the program requires, parsed and resolved on its own.
struct Loaded {
    name: String,
    file: String,
    source: String,
    ast: ast::Ast,
    scopes: Scopes,
}

/// Whether a program keeps its call stack for the debug library, and
/// what else it records for it.
#[derive(Clone, Copy, Default)]
struct DebugMode {
    on: bool,
    locals: bool,
    setlocal: bool,
    /// Upvalues may be set or joined: captures are shared through cells.
    rebinds: bool,
}

/// The debug mode of the program running, for chunks it loads.
static PROGRAM_DEBUG: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The entry the host reports an uncaught table error through, in the
/// program [`error_text_program`] builds.
pub(crate) const ERROR_TEXT_ENTRY: &str = "lua$error_text";

/// What a program is entered for.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Entry {
    /// Running the chunk, as `lua` runs a script: an error nothing caught
    /// ends the program, and the state closes when the chunk returns.
    Program,
    /// Opening the state for a host (`HOST_ENTRY`): the globals kept in
    /// the globals table, where the C API reaches them, and nothing run
    /// or closed.
    Host,
}
