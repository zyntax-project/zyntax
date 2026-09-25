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
use std::collections::{HashMap, HashSet};

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

/// The layouts of a chunk's shapes: a shape with fields, and not too
/// many, that no weak metatable may reach, gets slots. With `slots`
/// off (a chunk compiled at run time, whose shapes the program's hooks
/// do not know), none does. Number slots take kind bits from bit 63
/// down while those stay above the presence bits; a Number slot left
/// without one is a Scalar slot.
fn shape_layouts(inferred: &Inferred, chunk_index: i64, slots: bool) -> Vec<Option<ShapeLayout>> {
    let trace = std::env::var_os("ZYNTAX_TRACE_TYPES").is_some();
    inferred
        .shapes
        .iter()
        .enumerate()
        .map(|(k, info)| {
            if !slots
                || info.fields.is_empty()
                || info.fields.len() > MAX_SLOTS
                || inferred.may_be_weak(ShapeId(k as u32))
            {
                return None;
            }
            let gid = (chunk_index << 20) + k as i64 + 1;
            let mut kbits = (info.fields.len()..64).rev();
            let slots = info
                .fields
                .iter()
                .enumerate()
                .map(|(bit, (name, ty))| {
                    let mut kind = Slot::kind_of(ty.settled());
                    let kbit = match kind {
                        SlotKind::Number => kbits.next(),
                        _ => None,
                    };
                    if kind == SlotKind::Number && kbit.is_none() {
                        if trace {
                            eprintln!(
                                "[types] shape {k}: field {name} has no kind bit, a scalar slot"
                            );
                        }
                        kind = SlotKind::Scalar;
                    }
                    Slot {
                        name: name.clone(),
                        bit,
                        kind,
                        kbit,
                    }
                })
                .collect();
            Some(ShapeLayout {
                id: TypeId::next(),
                name: format!("LuaTable${gid}"),
                gid,
                slots,
            })
        })
        .collect()
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

/// The functions whose depth is bounded by the program's text: not
/// escaping, calling nothing through a value or the library, and not
/// reaching themselves through direct calls. Their depth needs no
/// count, and their depth check cannot raise.
fn bounded_functions(
    facts: &HashMap<FuncId, RaiseFact>,
    escaping: impl Fn(FuncId) -> bool,
) -> HashSet<FuncId> {
    let mut unbounded: HashSet<FuncId> = facts
        .iter()
        .filter(|(f, fact)| fact.dynamic || escaping(**f))
        .map(|(f, _)| *f)
        .collect();
    // A function on a cycle of direct calls.
    for f in facts.keys() {
        let mut seen = HashSet::new();
        let mut todo: Vec<FuncId> = facts[f].callees.iter().copied().collect();
        while let Some(g) = todo.pop() {
            if g == *f {
                unbounded.insert(*f);
                break;
            }
            if seen.insert(g)
                && let Some(fact) = facts.get(&g)
            {
                todo.extend(fact.callees.iter().copied());
            }
        }
    }
    loop {
        let before = unbounded.len();
        for (f, fact) in facts {
            if fact.callees.iter().any(|c| unbounded.contains(c)) {
                unbounded.insert(*f);
            }
        }
        if unbounded.len() == before {
            break;
        }
    }
    facts
        .keys()
        .copied()
        .filter(|f| !unbounded.contains(f))
        .collect()
}

/// The functions that may raise: those that check for an error
/// themselves, or their depth without a bound on it, and those calling
/// one of them, and so on.
fn raising_functions(
    facts: &HashMap<FuncId, RaiseFact>,
    bounded: &HashSet<FuncId>,
) -> HashSet<FuncId> {
    let mut raising: HashSet<FuncId> = facts
        .iter()
        .filter(|(f, fact)| fact.own || (fact.checks_depth && !bounded.contains(f)))
        .map(|(f, _)| *f)
        .collect();
    loop {
        let before = raising.len();
        for (f, fact) in facts {
            if fact.callees.iter().any(|c| raising.contains(c)) {
                raising.insert(*f);
            }
        }
        if raising.len() == before {
            return raising;
        }
    }
}

impl<'a> Module<'a> {
    fn typer(&self) -> Typer<'_> {
        Typer {
            scopes: self.scopes,
            known: self.inferred,
        }
    }

    /// The line a span starts on, counted from one, as positions and
    /// hooks see it: no line in a stripped chunk.
    fn line_of(&self, span: Span) -> i64 {
        let line = if self.stripped {
            library::STRIPPED_LINE
        } else {
            self.source_line(span)
        };
        (self.chunk_index << library::LINE_BITS) | line
    }

    /// The line a span starts on in the source, stripped or not.
    fn source_line(&self, span: Span) -> i64 {
        self.line_starts
            .partition_point(|&start| start <= span.start) as i64
    }

    /// Whether a program function may raise, as far as is known.
    fn raises(&self, f: FuncId) -> bool {
        match &self.raising {
            Some(set) => set.contains(&f),
            None => true,
        }
    }

    /// The struct of each slotted shape, declared and registered, and
    /// its layout handed on for the hooks.
    fn declare_shapes(
        &mut self,
        declarations: &mut Vec<TypedNode<TypedDeclaration>>,
        registry: &mut zyntax_typed_ast::TypeRegistry,
        hooks: &mut Vec<ShapeLayout>,
    ) {
        for layout in std::mem::take(&mut self.layouts).into_iter().flatten() {
            let fields = self.layout_fields(&layout);
            library::declare_reference_struct(registry, layout.id, &layout.name, &fields);
            declarations.push(library::reference_struct_class(&layout.name, &fields));
            hooks.push(layout);
        }
    }

    /// Whether tables of shape `k` never get a metatable.
    fn plain_shape(&self, k: ShapeId) -> bool {
        let info = self.inferred.shape(k);
        !info.unknown_meta && info.classes.is_empty()
    }

    /// How a shape's tables are laid out, when they have slots.
    fn layout(&self, k: ShapeId) -> Option<&ShapeLayout> {
        self.layouts.get(k.0 as usize).and_then(|l| l.as_ref())
    }

    /// The struct type of a layout's tables, as a pointer.
    fn shape_ty(&self, layout: &ShapeLayout) -> Type {
        library::table_ty(layout.id)
    }

    /// Every field of a layout's struct: the table header, then the
    /// slots.
    fn layout_fields(&self, layout: &ShapeLayout) -> Vec<(String, Type)> {
        let table_ty = self.types.table();
        let mut fields = library::table_header_fields(self.types.table_type);
        for slot in &layout.slots {
            fields.extend(slot.fields(&table_ty));
        }
        fields
    }

    /// Whether a function's depth is known bounded, so it keeps no
    /// count of it. Nothing is, until a first lowering has looked.
    fn is_bounded(&self, f: FuncId) -> bool {
        self.bounded.as_ref().is_some_and(|set| set.contains(&f))
    }

    /// Whether the program's code may run inside this lowered body
    /// before it returns: a call through a value, or into a library
    /// function that makes one.
    fn calls_dynamically(&self, statements: &[St]) -> bool {
        let mut names = std::collections::BTreeSet::new();
        for s in statements {
            library::callee_names(s, &mut names);
        }
        let helpers = self.reentrant_helpers.borrow();
        names
            .iter()
            .any(|n| self.reentrant.contains(n.as_str()) || helpers.contains(n))
    }

    /// The IR type of a static type.
    fn ir(&self, ty: Ty) -> Type {
        match ty {
            Ty::Bool => prim(PrimitiveType::Bool),
            Ty::Int => prim(PrimitiveType::I64),
            Ty::Float => prim(PrimitiveType::F64),
            Ty::Number | Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => number_type(),
            Ty::Str => prim(PrimitiveType::String),
            // A table of a known shape is a table, or null for nil.
            Ty::Table | Ty::Shape(_) => self.types.table(),
            // A known function is still the record every function value
            // is; what is known is where calls through it go.
            Ty::Func(_) | Ty::Nil | Ty::Any | Ty::Unknown => Type::Any,
        }
    }

    fn anys(&self) -> Type {
        self.types.anys()
    }

    /// The symbol of a global's module variable.
    fn global_symbol(name: &str) -> InternedString {
        intern(&format!("lua$g${name}"))
    }

    /// The symbol of a captured chunk local's module variable.
    fn module_local_symbol(&self, v: VarId) -> InternedString {
        intern(&format!(
            "lua$l${}{}${}",
            self.tag,
            self.scopes.var(v).name,
            v.0
        ))
    }

    fn declare_module_var(&self, name: InternedString, ty: Ty) {
        let mut vars = self.module_vars.borrow_mut();
        if !vars.iter().any(|(n, _)| *n == name) {
            vars.push((name, ty));
        }
    }

    /// The typed entry's name for a function.
    fn entry_name(&self, f: FuncId) -> String {
        let info = self.scopes.func(f);
        if f == CHUNK {
            return if self.tag.is_empty() {
                ENTRY.to_string()
            } else {
                format!("lua${}chunk", self.tag)
            };
        }
        if info.top_level
            && let Some((name, _)) = self
                .scopes
                .global_functions
                .iter()
                .find(|(_, id)| **id == f)
        {
            return format!("lua${}{name}", self.tag);
        }
        let name = if info.name.is_empty() {
            "anon".to_string()
        } else {
            info.name.replace(['.', ':'], "$")
        };
        format!("lua${}{name}${}", self.tag, f.0)
    }

    /// The restores of the line in a body that nothing reads.
    fn strip_line_restores(&self, statements: &mut [St]) {
        let reads_line = |name: &str| self.fallible.contains(name) || self.reentrant.contains(name);
        LineRestores::strip(statements, &reads_line);
    }

    /// Whether a function's typed entry takes its record as `env`: a
    /// nested function capturing something, which the record holds.
    fn takes_env(&self, f: FuncId) -> bool {
        let info = self.scopes.func(f);
        !info.top_level && !info.captures.is_empty()
    }

    /// The number a function's values carry, unique across the
    /// program's chunks: the chunk's number above the low 32 bits.
    fn func_key(&self, f: FuncId) -> i64 {
        (self.chunk_index << 32) | f.0 as i64
    }

    /// A record for the debug library's host, when the program keeps
    /// its call stack.
    fn debug_record(&self, fields: &[String]) {
        if let Some(debug) = &self.debug {
            debug.records.borrow_mut().push(
                fields
                    .iter()
                    .map(|f| sanitize(f))
                    .collect::<Vec<_>>()
                    .join("\x1f"),
            );
        }
    }

    /// The record code's name for a function.
    fn code_name(&self, f: FuncId) -> String {
        format!("{}$fn", self.entry_name(f))
    }

    fn sig(&self, f: FuncId) -> types::Sig {
        self.inferred.sig(f).cloned().unwrap_or_else(|| types::Sig {
            params: vec![Ty::Nil; self.scopes.func(f).params.len()],
            returns: Returns::Fixed(Vec::new()),
        })
    }

    /// The IR return type of a function.
    fn return_ir(&self, returns: &Returns) -> Type {
        match returns {
            Returns::Fixed(v) if v.is_empty() => prim(PrimitiveType::Unit),
            Returns::Fixed(v) if v.len() == 1 => self.ir(v[0]),
            Returns::Fixed(_) => self.anys(),
            Returns::Dynamic | Returns::Unknown => Type::Any,
        }
    }
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

fn unsupported<T>(what: impl Into<String>, span: Span) -> Result<T> {
    Err(Error::unsupported(what, span))
}

fn span_of<N: full_moon::node::Node>(node: &N) -> Span {
    match node.range() {
        Some((a, b)) => Span::new(a.bytes(), b.bytes()),
        None => Span::new(0, 0),
    }
}

fn node(x: TypedExpression, ty: Type, span: Span) -> Node {
    TypedNode::new(x, ty, span)
}

fn stmt(s: TypedStatement, span: Span) -> St {
    TypedNode::new(s, Type::Unknown, span)
}

fn expr_stmt(e: Node) -> St {
    let span = e.span;
    stmt(TypedStatement::Expression(Box::new(e)), span)
}

thread_local! {
    /// Every name a chunk being lowered refers to, while one is: what
    /// it needs declared of the library.
    static NAMED: std::cell::RefCell<Option<std::collections::HashSet<InternedString>>> =
        const { std::cell::RefCell::new(None) };
}

fn var(name: InternedString, ty: Type, span: Span) -> Node {
    NAMED.with(|named| {
        if let Some(named) = named.borrow_mut().as_mut() {
            named.insert(name);
        }
    });
    node(TypedExpression::Variable(name), ty, span)
}

fn int_lit(v: i64, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        prim(PrimitiveType::I64),
        span,
    )
}

fn int32_lit(v: i32, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        prim(PrimitiveType::I32),
        span,
    )
}

fn float_lit(v: f64, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Float(v)),
        prim(PrimitiveType::F64),
        span,
    )
}

fn bool_lit(v: bool, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Bool(v)),
        prim(PrimitiveType::Bool),
        span,
    )
}

fn str_lit(s: &str, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::String(intern(s))),
        prim(PrimitiveType::String),
        span,
    )
}

fn null(ty: Type, span: Span) -> Node {
    node(TypedExpression::Literal(TypedLiteral::Null), ty, span)
}

fn nil(span: Span) -> Node {
    null(Type::Any, span)
}

fn call(name: &str, args: Vec<Node>, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(var(intern(name), Type::Unknown, span)),
            positional_args: args,
            named_args: Vec::new(),
            type_args: Vec::new(),
        }),
        ty,
        span,
    )
}

fn binary(op: BinaryOp, left: Node, right: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Binary(TypedBinary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }),
        ty,
        span,
    )
}

fn unary(op: UnaryOp, operand: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Unary(TypedUnary {
            op,
            operand: Box::new(operand),
        }),
        ty,
        span,
    )
}

fn if_value(cond: Node, then: Node, els: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::If(TypedIfExpr {
            condition: Box::new(cond),
            then_branch: Box::new(then),
            else_branch: Box::new(els),
        }),
        ty,
        span,
    )
}

fn cast(value: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(value),
            target_type: ty.clone(),
        }),
        ty,
        span,
    )
}

fn index(xs: Node, i: Node, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Index(TypedIndex {
            object: Box::new(xs),
            index: Box::new(i),
        }),
        ty,
        span,
    )
}

/// `obj.field`, a struct field read.
fn field(obj: Node, name: &str, ty: Type, span: Span) -> Node {
    node(
        TypedExpression::Field(TypedFieldAccess {
            object: Box::new(obj),
            field: intern(name),
        }),
        ty,
        span,
    )
}

/// `t` as a pointer to the struct of a shape's tables.
fn as_shape(t: Node, shape_ty: Type, span: Span) -> Node {
    cast(cast(t, prim(PrimitiveType::I64), span), shape_ty, span)
}

fn assign(target: Node, value: Node, span: Span) -> St {
    let ty = target.ty.clone();
    expr_stmt(binary(BinaryOp::Assign, target, value, ty, span))
}

fn let_(name: InternedString, ty: Type, value: Node, span: Span) -> St {
    stmt(
        TypedStatement::Let(TypedLet {
            name,
            ty,
            mutability: Mutability::Mutable,
            initializer: Some(Box::new(value)),
            span,
        }),
        span,
    )
}

fn if_(cond: Node, then: Vec<St>, els: Option<Vec<St>>, span: Span) -> St {
    stmt(
        TypedStatement::If(TypedIf {
            condition: Box::new(cond),
            then_block: TypedBlock {
                statements: then,
                span,
            },
            else_block: els.map(|statements| TypedBlock { statements, span }),
            span,
        }),
        span,
    )
}

fn while_(cond: Node, body: Vec<St>, span: Span) -> St {
    stmt(
        TypedStatement::While(TypedWhile {
            condition: Box::new(cond),
            body: TypedBlock {
                statements: body,
                span,
            },
            span,
        }),
        span,
    )
}

fn ret(value: Option<Node>, span: Span) -> St {
    stmt(TypedStatement::Return(value.map(Box::new)), span)
}

/// Statements followed by a value, as one expression.
/// A scalar whose kind is decided at run time: a tag, the integer it
/// holds and the float it holds. A value struct, so it lives in
/// registers; the field the tag does not select is unspecified, so
/// every reader selects by the tag. A number is one whose tag is never
/// nil or boolean.
fn number_type() -> Type {
    Type::Tuple(vec![
        prim(PrimitiveType::I64),
        prim(PrimitiveType::I64),
        prim(PrimitiveType::F64),
    ])
}

/// Whether arithmetic or ordering on `a` and `b` takes the numbers'
/// own operation once a nil operand has raised: each is a number of
/// known kind or such a number or nil, and one may be nil.
fn or_nil_operands(a: Ty, b: Ty) -> bool {
    let known = |t: Ty| matches!(t.underlying(), Ty::Int | Ty::Float);
    (matches!(a, Ty::IntOrNil | Ty::FloatOrNil) || matches!(b, Ty::IntOrNil | Ty::FloatOrNil))
        && known(a)
        && known(b)
}

/// Whether `a == b` is decided by tags and one number part: one side
/// is an integer or nil and the other an integer, nil or the same; or
/// the like for floats.
fn or_nil_comparable(a: Ty, b: Ty) -> bool {
    let fits = |x: Ty, y: Ty| match x {
        Ty::IntOrNil => matches!(y, Ty::IntOrNil | Ty::Int | Ty::Nil),
        Ty::FloatOrNil => matches!(y, Ty::FloatOrNil | Ty::Float | Ty::Nil),
        _ => false,
    };
    fits(a, b) || fits(b, a)
}

/// The tags of a run-time scalar. A boolean's truth is its tag, so a
/// scalar is true from `TAG_TRUE` up and a number from `TAG_INT` up.
const TAG_NIL: i64 = 0;
const TAG_FALSE: i64 = 1;
const TAG_TRUE: i64 = 2;
const TAG_INT: i64 = 3;
const TAG_FLOAT: i64 = 4;

fn number_value(tag: Node, int: Node, float: Node, span: Span) -> Node {
    node(
        TypedExpression::Tuple(vec![tag, int, float]),
        number_type(),
        span,
    )
}

fn number_of_int(n: Node, span: Span) -> Node {
    number_value(int_lit(TAG_INT, span), n, float_lit(0.0, span), span)
}

fn number_of_float(f: Node, span: Span) -> Node {
    number_value(int_lit(TAG_FLOAT, span), int_lit(0, span), f, span)
}

fn scalar_of_nil(span: Span) -> Node {
    number_value(
        int_lit(TAG_NIL, span),
        int_lit(0, span),
        float_lit(0.0, span),
        span,
    )
}

fn scalar_of_bool(b: Node, span: Span) -> Node {
    let tag = if_value(
        b,
        int_lit(TAG_TRUE, span),
        int_lit(TAG_FALSE, span),
        prim(PrimitiveType::I64),
        span,
    );
    number_value(tag, int_lit(0, span), float_lit(0.0, span), span)
}

/// The tag an integer-or-float result carries, from whether it is
/// the integer.
fn tag_of_is_int(is_int: Node, span: Span) -> Node {
    if_value(
        is_int,
        int_lit(TAG_INT, span),
        int_lit(TAG_FLOAT, span),
        prim(PrimitiveType::I64),
        span,
    )
}

/// The parts of a scalar that is a plain read.
struct NumberParts {
    tag: Node,
    int: Node,
    float: Node,
}

impl NumberParts {
    fn of(n: &Node, span: Span) -> NumberParts {
        let part = |i: i64, ty: PrimitiveType| index(n.clone(), int_lit(i, span), prim(ty), span);
        NumberParts {
            tag: part(0, PrimitiveType::I64),
            int: part(1, PrimitiveType::I64),
            float: part(2, PrimitiveType::F64),
        }
    }

    fn has_tag(&self, tag: i64, span: Span) -> Node {
        binary(
            BinaryOp::Eq,
            self.tag.clone(),
            int_lit(tag, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    fn is_int(&self, span: Span) -> Node {
        self.has_tag(TAG_INT, span)
    }

    /// Whether the value is a number at all.
    fn is_numeric(&self, span: Span) -> Node {
        binary(
            BinaryOp::Ge,
            self.tag.clone(),
            int_lit(TAG_INT, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    /// Whether the value is true: anything but nil and false.
    fn is_true(&self, span: Span) -> Node {
        binary(
            BinaryOp::Ge,
            self.tag.clone(),
            int_lit(TAG_TRUE, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    /// Whether both values are numbers, without a short-circuit: the
    /// tags are small, so their product says.
    fn both_numeric(&self, other: &NumberParts, span: Span) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        binary(
            BinaryOp::Ge,
            binary(
                BinaryOp::Mul,
                self.tag.clone(),
                other.tag.clone(),
                i64_t,
                span,
            ),
            int_lit(TAG_INT * TAG_INT, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    /// Whether two numbers are both integers, without a short-circuit:
    /// the tags of numbers are `TAG_INT` and `TAG_FLOAT`, and only two
    /// `TAG_INT`s or together to `TAG_INT`.
    fn both_int(&self, other: &NumberParts, span: Span) -> Node {
        binary(
            BinaryOp::Eq,
            binary(
                BinaryOp::BitOr,
                self.tag.clone(),
                other.tag.clone(),
                prim(PrimitiveType::I64),
                span,
            ),
            int_lit(TAG_INT, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    /// The value as a float, whichever number it holds: both arms are
    /// plain reads, so this is a select.
    fn as_float(&self, span: Span) -> Node {
        if_value(
            self.is_int(span),
            cast(self.int.clone(), prim(PrimitiveType::F64), span),
            self.float.clone(),
            prim(PrimitiveType::F64),
            span,
        )
    }
}

fn block_value(statements: Vec<St>, value: Node, span: Span) -> Node {
    if statements.is_empty() {
        return value;
    }
    let ty = value.ty.clone();
    let mut statements = statements;
    statements.push(expr_stmt(value));
    node(
        TypedExpression::Block(TypedBlock { statements, span }),
        ty,
        span,
    )
}

/// `x % n` for an integer `x` and a positive power of two `n`: the low
/// bits, which are the floored modulo whatever `x`'s sign.
fn pow2_mod(x: Node, n: i64, span: Span) -> Node {
    let i64_t = prim(PrimitiveType::I64);
    binary(BinaryOp::BitAnd, x, int_lit(n - 1, span), i64_t, span)
}

/// `x // n` for an integer `x` and a positive power of two `n`: an
/// arithmetic shift, which rounds toward negative infinity.
fn pow2_floordiv(x: Node, n: i64, span: Span) -> Node {
    let i64_t = prim(PrimitiveType::I64);
    let shift = int_lit(n.trailing_zeros() as i64, span);
    binary(BinaryOp::Shr, x, shift, i64_t, span)
}

/// The length of a list.
fn list_len(xs: Node, span: Span) -> Node {
    node(
        TypedExpression::MethodCall(zyntax_typed_ast::typed_ast::TypedMethodCall {
            receiver: Box::new(xs),
            method: intern("len"),
            type_args: Vec::new(),
            positional_args: Vec::new(),
            named_args: Vec::new(),
        }),
        prim(PrimitiveType::I64),
        span,
    )
}

/// The address of function `name`.
fn code_of(name: &str, span: Span) -> Node {
    var(intern(name), prim(PrimitiveType::USize), span)
}

fn parameter(name: InternedString, ty: Type, span: Span) -> TypedParameter {
    let mut p = TypedParameter::regular(name, ty.clone(), Mutability::Mutable, span);
    // A dynamic parameter is dynamic by the language's rules, not by
    // omission, so the lowering does not warn about it.
    if ty == Type::Any {
        p.attributes
            .push(zyntax_typed_ast::typed_ast::ParameterAttribute {
                name: intern("dynamic"),
                args: Vec::new(),
                span,
            });
    }
    // Anything on the heap that a function receives may be kept by
    // it: a table stored into another, a value captured.
    if !matches!(
        ty,
        Type::Primitive(PrimitiveType::I64)
            | Type::Primitive(PrimitiveType::F64)
            | Type::Primitive(PrimitiveType::Bool)
            | Type::Primitive(PrimitiveType::Unit)
    ) {
        p.ownership = ParamOwnership::Shared;
    }
    p
}

fn typed_function(
    name: &str,
    params: Vec<TypedParameter>,
    return_type: Type,
    statements: Vec<St>,
    span: Span,
) -> TypedFunction {
    TypedFunction {
        name: intern(name),
        annotations: vec![library::strict_fp()],
        effects: Vec::new(),
        with_handlers: Vec::new(),
        type_params: Vec::new(),
        params,
        return_type,
        body: Some(TypedBlock { statements, span }),
        visibility: Visibility::Public,
        is_async: false,
        is_fiber: false,
        is_pure: false,
        is_external: false,
        calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
        link_name: None,
        module: None,
    }
}

/// The bytes of a string literal token, escapes decoded, and a byte the
/// source could not spell as text restored from its stand-in.
pub(crate) fn string_bytes(token: &TokenReference) -> std::result::Result<Vec<u8>, String> {
    let bytes = literal_bytes(token)?;
    Ok(restore_bytes(bytes))
}

/// Private-use characters standing for bytes (see `source_text`) back
/// to the bytes.
fn restore_bytes(bytes: Vec<u8>) -> Vec<u8> {
    // U+F700..U+F7FF are EF 9C 80 .. EF 9F BF.
    if !bytes
        .windows(2)
        .any(|w| w[0] == 0xEF && (0x9C..=0x9F).contains(&w[1]))
    {
        return bytes;
    }
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 2 < bytes.len()
            && bytes[i] == 0xEF
            && (0x9C..=0x9F).contains(&bytes[i + 1])
            && (0x80..=0xBF).contains(&bytes[i + 2])
        {
            let code = ((bytes[i] as u32 & 0x0F) << 12)
                | ((bytes[i + 1] as u32 & 0x3F) << 6)
                | (bytes[i + 2] as u32 & 0x3F);
            if (crate::ESCAPED_BYTES..crate::ESCAPED_BYTES + 0x100).contains(&code) {
                out.push((code - crate::ESCAPED_BYTES) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

fn literal_bytes(token: &TokenReference) -> std::result::Result<Vec<u8>, String> {
    use full_moon::tokenizer::{StringLiteralQuoteType, TokenType};
    let TokenType::StringLiteral {
        literal,
        quote_type,
        ..
    } = token.token().token_type()
    else {
        return Ok(token.token().to_string().into_bytes());
    };
    match quote_type {
        StringLiteralQuoteType::Brackets => {
            // A long string starts after its first line break.
            let s = literal.as_str();
            let s = s
                .strip_prefix("\r\n")
                .or_else(|| s.strip_prefix('\n'))
                .or_else(|| s.strip_prefix('\r'))
                .unwrap_or(s);
            Ok(s.as_bytes().to_vec())
        }
        _ => decode_escapes(literal.as_str()),
    }
}

/// A string literal as a node: text when it is UTF-8, otherwise built
/// at run time from its bytes spelled in hex, since a literal in the
/// typed AST is text.
fn string_literal(bytes: Vec<u8>, span: Span) -> Node {
    match String::from_utf8(bytes) {
        Ok(text) => str_lit(&text, span),
        Err(e) => {
            let hex: String = e.as_bytes().iter().map(|b| format!("{b:02x}")).collect();
            call(
                "zl_bytes",
                vec![str_lit(&hex, span)],
                prim(PrimitiveType::String),
                span,
            )
        }
    }
}

/// Lua's escapes: `\n` and the others, `\ddd`, `\xXX`, `\u{XXX}`, `\z`,
/// and a backslash before a line break.
fn decode_escapes(s: &str) -> std::result::Result<Vec<u8>, String> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b != b'\\' {
            out.push(b);
            i += 1;
            continue;
        }
        i += 1;
        let Some(&c) = bytes.get(i) else {
            return Err("unfinished string".to_string());
        };
        i += 1;
        match c {
            b'n' => out.push(b'\n'),
            b't' => out.push(b'\t'),
            b'r' => out.push(b'\r'),
            b'a' => out.push(7),
            b'b' => out.push(8),
            b'f' => out.push(12),
            b'v' => out.push(11),
            b'\\' => out.push(b'\\'),
            b'"' => out.push(b'"'),
            b'\'' => out.push(b'\''),
            b'\n' => {
                out.push(b'\n');
                if bytes.get(i) == Some(&b'\r') {
                    i += 1;
                }
            }
            b'\r' => {
                out.push(b'\n');
                if bytes.get(i) == Some(&b'\n') {
                    i += 1;
                }
            }
            b'x' => {
                let hex = s.get(i..i + 2).ok_or("hexadecimal digit expected")?;
                let v = u8::from_str_radix(hex, 16).map_err(|_| "hexadecimal digit expected")?;
                out.push(v);
                i += 2;
            }
            b'z' => {
                // Lua's whitespace: a vertical tab counts.
                while i < bytes.len()
                    && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
                {
                    i += 1;
                }
            }
            b'u' => {
                if bytes.get(i) != Some(&b'{') {
                    return Err("missing '{' in \\u{xxxx}".to_string());
                }
                let end = s[i..].find('}').ok_or("missing '}' in \\u{xxxx}")? + i;
                let digits = &s[i + 1..end];
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err("hexadecimal digit expected".to_string());
                }
                // Lua's own UTF-8 reaches 0x7FFFFFFF in six bytes.
                let code = u64::from_str_radix(digits, 16).unwrap_or(u64::MAX);
                if code > 0x7FFF_FFFF {
                    return Err("UTF-8 value too large".to_string());
                }
                crate::host::utf8_encode(code as u32, &mut out);
                i = end + 1;
            }
            d if d.is_ascii_digit() => {
                let mut v: u32 = (d - b'0') as u32;
                let mut n = 1;
                while n < 3 && i < bytes.len() && bytes[i].is_ascii_digit() {
                    v = v * 10 + (bytes[i] - b'0') as u32;
                    i += 1;
                    n += 1;
                }
                if v > 255 {
                    return Err("decimal escape too large".to_string());
                }
                out.push(v as u8);
            }
            other => return Err(format!("invalid escape sequence '\\{}'", other as char)),
        }
    }
    Ok(out)
}

impl<'m, 'a> Lowerer<'m, 'a> {
    fn new(m: &'m Module<'a>, func: FuncId) -> Self {
        let returns = m.sig(func).returns;
        Lowerer {
            m,
            func,
            storage: HashMap::new(),
            varargs: None,
            returns,
            temps: 0,
            bound: std::collections::HashSet::new(),
            raised: false,
            raise_callees: HashSet::new(),
            line_needed: false,
            entry_line: false,
            tbc: Vec::new(),
            counts_depth: false,
            depth: 0,
            loop_depths: Vec::new(),
            frames: false,
            live: Vec::new(),
            for_states: Vec::new(),
            lines: std::collections::BTreeSet::new(),
            tail_span: None,
            end_span: None,
        }
    }

    fn typer(&self) -> Typer<'_> {
        self.m.typer()
    }

    fn ir(&self, ty: Ty) -> Type {
        self.m.ir(ty)
    }

    fn scopes(&self) -> &'a Scopes {
        self.m.scopes
    }

    /// A local no Lua program can spell.
    fn temp(&mut self) -> InternedString {
        self.temps += 1;
        intern(&format!("$t{}", self.temps))
    }

    /// The symbol a Lua local is lowered to: its name, or its name and
    /// id once another local of that name has been declared here.
    fn local_symbol(&mut self, v: VarId) -> InternedString {
        let name = &self.scopes().var(v).name;
        let plain = intern(name);
        let symbol = if self.bound.contains(&plain) {
            intern(&format!("{name}${}", v.0))
        } else {
            plain
        };
        self.bound.insert(symbol);
        symbol
    }

    // ─── values ─────────────────────────────────────────────────

    fn nil_val(&self, span: Span) -> Val {
        Val {
            node: nil(span),
            ty: Ty::Nil,
        }
    }

    /// `v` as a value of `target`.
    fn coerce(&mut self, v: Val, target: Ty) -> Node {
        let span = v.node.span;
        let target = if target == Ty::Unknown {
            Ty::Nil
        } else {
            target
        };
        // A scalar that may only be nil besides holds its number in the
        // number's part; any other conversion is the scalar's.
        match (v.ty, target) {
            (Ty::IntOrNil, Ty::Int) => {
                let (pre, n) = self.number_parts(v);
                return block_value(pre, n.int, span);
            }
            (Ty::FloatOrNil, Ty::Float) => {
                let (pre, n) = self.number_parts(v);
                return block_value(pre, n.float, span);
            }
            // A dynamic value known to be the number or nil: nil, or
            // the payload.
            (Ty::Any, Ty::IntOrNil | Ty::FloatOrNil) => {
                let mut pre = Vec::new();
                let b = self.hold(v, &mut pre);
                let is_nil = binary(
                    BinaryOp::Eq,
                    b.node.clone(),
                    nil(span),
                    prim(PrimitiveType::Bool),
                    span,
                );
                let number = if target == Ty::IntOrNil {
                    let payload = call(
                        "zb_box_payload_i64",
                        vec![b.node],
                        prim(PrimitiveType::I64),
                        span,
                    );
                    number_of_int(payload, span)
                } else {
                    let payload = call(
                        "zb_box_payload_f64",
                        vec![b.node],
                        prim(PrimitiveType::F64),
                        span,
                    );
                    number_of_float(payload, span)
                };
                let value = if_value(is_nil, scalar_of_nil(span), number, number_type(), span);
                return block_value(pre, value, span);
            }
            (Ty::IntOrNil | Ty::FloatOrNil, Ty::Any) => {
                let int = v.ty == Ty::IntOrNil;
                let (pre, n) = self.number_parts(v);
                let boxed = if int {
                    call("zb_box_i64", vec![n.int.clone()], Type::Any, span)
                } else {
                    call("zb_box_f64", vec![n.float.clone()], Type::Any, span)
                };
                let value = if_value(n.has_tag(TAG_NIL, span), nil(span), boxed, Type::Any, span);
                return block_value(pre, value, span);
            }
            _ => {}
        }
        let as_scalar = |t: Ty| if t.is_tagged() { Ty::Scalar } else { t };
        let v = Val {
            node: v.node,
            ty: as_scalar(v.ty),
        };
        let target = as_scalar(target);
        match (v.ty, target) {
            (a, b) if a == b => v.node,
            (Ty::Unknown, _) => v.node,
            (Ty::Nil, Ty::Any) => v.node,
            (Ty::Func(_), Ty::Any) | (Ty::Any, Ty::Func(_)) | (Ty::Func(_), Ty::Func(_)) => v.node,
            (Ty::Int, Ty::Float) => cast(v.node, prim(PrimitiveType::F64), span),
            (Ty::Int, Ty::Number) => number_of_int(v.node, span),
            (Ty::Float, Ty::Number) => number_of_float(v.node, span),
            (Ty::Number, Ty::Float) => {
                let (pre, n) = self.number_parts(v);
                block_value(pre, n.as_float(span), span)
            }
            (Ty::Number, Ty::Int) => {
                let (pre, n) = self.number_parts(v);
                let value = if_value(
                    n.is_int(span),
                    n.int,
                    cast(n.float, prim(PrimitiveType::I64), span),
                    prim(PrimitiveType::I64),
                    span,
                );
                block_value(pre, value, span)
            }
            (Ty::Number, Ty::Any) => {
                let (pre, n) = self.number_parts(v);
                let value = if_value(
                    n.is_int(span),
                    call("zb_box_i64", vec![n.int], Type::Any, span),
                    call("zb_box_f64", vec![n.float], Type::Any, span),
                    Type::Any,
                    span,
                );
                block_value(pre, value, span)
            }
            // A dynamic value known to hold a number: its kind is read
            // off the box with its payload.
            (Ty::Any, Ty::Number) => {
                let mut pre = Vec::new();
                let b = self.hold(v, &mut pre);
                let is_int = binary(
                    BinaryOp::Ne,
                    call(
                        "zb_any_category",
                        vec![b.node.clone()],
                        prim(PrimitiveType::I64),
                        span,
                    ),
                    int_lit(library::FLOAT, span),
                    prim(PrimitiveType::Bool),
                    span,
                );
                let tag = tag_of_is_int(is_int, span);
                let int = call(
                    "zb_box_payload_i64",
                    vec![b.node.clone()],
                    prim(PrimitiveType::I64),
                    span,
                );
                let float = call(
                    "zb_box_payload_f64",
                    vec![b.node],
                    prim(PrimitiveType::F64),
                    span,
                );
                block_value(pre, number_value(tag, int, float, span), span)
            }
            // A number is a scalar as it is; the other scalars take
            // their tag.
            (Ty::Number, Ty::Scalar) | (Ty::Scalar, Ty::Number) => v.node,
            (Ty::Nil, Ty::Scalar) => {
                block_value(vec![expr_stmt(v.node)], scalar_of_nil(span), span)
            }
            (Ty::Bool, Ty::Scalar) => scalar_of_bool(v.node, span),
            (Ty::Int, Ty::Scalar) => number_of_int(v.node, span),
            (Ty::Float, Ty::Scalar) => number_of_float(v.node, span),
            (Ty::Scalar, Ty::Any) => {
                let (pre, n) = self.number_parts(v);
                let bool_t = prim(PrimitiveType::Bool);
                let is_true = n.has_tag(TAG_TRUE, span);
                let is_bool = binary(
                    BinaryOp::Lt,
                    n.tag.clone(),
                    int_lit(TAG_INT, span),
                    bool_t,
                    span,
                );
                let value = if_value(
                    n.has_tag(TAG_NIL, span),
                    nil(span),
                    if_value(
                        is_bool,
                        call("zb_box_bool", vec![is_true], Type::Any, span),
                        if_value(
                            n.has_tag(TAG_INT, span),
                            call("zb_box_i64", vec![n.int], Type::Any, span),
                            call("zb_box_f64", vec![n.float], Type::Any, span),
                            Type::Any,
                            span,
                        ),
                        Type::Any,
                        span,
                    ),
                    Type::Any,
                    span,
                );
                block_value(pre, value, span)
            }
            // A dynamic value known to be a scalar: its tag is read off
            // the box.
            (Ty::Any, Ty::Scalar) => {
                let mut pre = Vec::new();
                let b = self.hold(v, &mut pre);
                let i64_t = prim(PrimitiveType::I64);
                let bool_t = prim(PrimitiveType::Bool);
                let category = call("zb_any_category", vec![b.node.clone()], i64_t.clone(), span);
                let is_cat = |c: i64| {
                    binary(
                        BinaryOp::Eq,
                        category.clone(),
                        int_lit(c, span),
                        bool_t.clone(),
                        span,
                    )
                };
                let is_nil = binary(
                    BinaryOp::Eq,
                    b.node.clone(),
                    nil(span),
                    bool_t.clone(),
                    span,
                );
                let truth = binary(
                    BinaryOp::Ne,
                    call(
                        "zb_box_payload_bool",
                        vec![b.node.clone()],
                        prim(PrimitiveType::I32),
                        span,
                    ),
                    int32_lit(0, span),
                    bool_t.clone(),
                    span,
                );
                let as_bool = scalar_of_bool(truth, span);
                let as_float = number_of_float(
                    call(
                        "zb_box_payload_f64",
                        vec![b.node.clone()],
                        prim(PrimitiveType::F64),
                        span,
                    ),
                    span,
                );
                let as_int =
                    number_of_int(call("zb_box_payload_i64", vec![b.node], i64_t, span), span);
                let value = if_value(
                    is_nil,
                    scalar_of_nil(span),
                    if_value(
                        is_cat(library::BOOL),
                        as_bool,
                        if_value(
                            is_cat(library::FLOAT),
                            as_float,
                            as_int,
                            number_type(),
                            span,
                        ),
                        number_type(),
                        span,
                    ),
                    number_type(),
                    span,
                );
                block_value(pre, value, span)
            }
            // A scalar the types know the kind of.
            (Ty::Scalar, Ty::Int) => {
                let (pre, n) = self.number_parts(v);
                let value = if_value(
                    n.is_int(span),
                    n.int,
                    cast(n.float, prim(PrimitiveType::I64), span),
                    prim(PrimitiveType::I64),
                    span,
                );
                block_value(pre, value, span)
            }
            (Ty::Scalar, Ty::Float) => {
                let (pre, n) = self.number_parts(v);
                block_value(pre, n.as_float(span), span)
            }
            (Ty::Scalar, Ty::Bool) => {
                let (pre, n) = self.number_parts(v);
                block_value(pre, n.has_tag(TAG_TRUE, span), span)
            }
            (Ty::Scalar, Ty::Nil) => block_value(vec![expr_stmt(v.node)], nil(span), span),
            // A shaped table is a table; null is nil. Only a table slot
            // takes a shaped value as a table, and it keeps nil as null
            // with its present bit clear.
            (Ty::Table, Ty::Shape(_))
            | (Ty::Shape(_), Ty::Shape(_))
            | (Ty::Shape(_), Ty::Table) => v.node,
            (Ty::Nil, Ty::Shape(_)) => block_value(
                vec![expr_stmt(v.node)],
                null(self.ir(Ty::Table), span),
                span,
            ),
            (Ty::Shape(_), Ty::Any) => {
                let mut pre = Vec::new();
                let t = self.hold(v, &mut pre);
                let is_null = binary(
                    BinaryOp::Eq,
                    t.node.clone(),
                    null(self.ir(Ty::Table), span),
                    prim(PrimitiveType::Bool),
                    span,
                );
                let boxed = self.box_table(t.node);
                block_value(
                    pre,
                    if_value(is_null, nil(span), boxed, Type::Any, span),
                    span,
                )
            }
            (Ty::Any, Ty::Shape(_)) => {
                let mut pre = Vec::new();
                let b = self.hold(v, &mut pre);
                let is_nil = binary(
                    BinaryOp::Eq,
                    b.node.clone(),
                    nil(span),
                    prim(PrimitiveType::Bool),
                    span,
                );
                let table_t = self.ir(Ty::Table);
                let unboxed = self.unbox_table(b.node);
                block_value(
                    pre,
                    if_value(is_nil, null(table_t.clone(), span), unboxed, table_t, span),
                    span,
                )
            }
            (Ty::Shape(_), Ty::Nil) => block_value(vec![expr_stmt(v.node)], nil(span), span),
            (Ty::Bool, Ty::Any) => call("zb_box_bool", vec![v.node], Type::Any, span),
            (Ty::Int, Ty::Any) => call("zb_box_i64", vec![v.node], Type::Any, span),
            (Ty::Float, Ty::Any) => call("zb_box_f64", vec![v.node], Type::Any, span),
            (Ty::Str, Ty::Any) => call("zb_box_str", vec![v.node], Type::Any, span),
            (Ty::Table, Ty::Any) => self.box_table(v.node),
            // A dynamic value known to hold the type: read straight out.
            (Ty::Any, Ty::Bool) => binary(
                BinaryOp::Ne,
                call(
                    "zb_box_payload_bool",
                    vec![v.node],
                    prim(PrimitiveType::I32),
                    span,
                ),
                int32_lit(0, span),
                prim(PrimitiveType::Bool),
                span,
            ),
            (Ty::Any, Ty::Int) => call(
                "zb_box_payload_i64",
                vec![v.node],
                prim(PrimitiveType::I64),
                span,
            ),
            (Ty::Any, Ty::Float) => call(
                "zb_box_payload_f64",
                vec![v.node],
                prim(PrimitiveType::F64),
                span,
            ),
            (Ty::Any, Ty::Str) => call(
                "zb_box_get_str",
                vec![v.node],
                prim(PrimitiveType::String),
                span,
            ),
            (Ty::Any, Ty::Table) => self.unbox_table(v.node),
            (Ty::Any, Ty::Nil) => v.node,
            // A nil where a typed value was expected: the value's zero,
            // after the expression ran. Inference joins nil into the
            // type, so this is only reached for a slot never read.
            (Ty::Nil, t) => {
                let zero = match t {
                    Ty::Int => int_lit(0, span),
                    Ty::Float => float_lit(0.0, span),
                    Ty::Number => number_of_int(int_lit(0, span), span),
                    Ty::Scalar => scalar_of_nil(span),
                    Ty::Bool => bool_lit(false, span),
                    Ty::Str => str_lit("", span),
                    Ty::Table | Ty::Shape(_) => null(self.ir(Ty::Table), span),
                    _ => nil(span),
                };
                block_value(vec![expr_stmt(v.node)], zero, span)
            }
            (Ty::Float, Ty::Int) => cast(v.node, prim(PrimitiveType::I64), span),
            (a, b) => {
                // Anything else goes through the box.
                let boxed = self.coerce(v, Ty::Any);
                if b == Ty::Any {
                    return boxed;
                }
                let _ = a;
                self.coerce(
                    Val {
                        node: boxed,
                        ty: Ty::Any,
                    },
                    b,
                )
            }
        }
    }

    fn boxed(&mut self, v: Val) -> Node {
        self.coerce(v, Ty::Any)
    }

    fn box_table(&mut self, t: Node) -> Node {
        let span = t.span;
        call(
            "zb_box_instance",
            vec![
                cast(t, prim(PrimitiveType::I64), span),
                int32_lit(library::table_tag() as i32, span),
            ],
            Type::Any,
            span,
        )
    }

    fn unbox_table(&mut self, x: Node) -> Node {
        let span = x.span;
        cast(
            call(
                "zb_unbox_instance_raw",
                vec![x],
                prim(PrimitiveType::I64),
                span,
            ),
            self.ir(Ty::Table),
            span,
        )
    }

    /// Whether the node is a plain read, safe to repeat.
    fn is_simple(node: &Node) -> bool {
        matches!(
            node.node,
            TypedExpression::Variable(_) | TypedExpression::Literal(_)
        )
    }

    /// The value copied into a temporary, unless it is a literal: a
    /// variable read now, before a statement that may assign it.
    fn snapshot(&mut self, v: Val, pre: &mut Vec<St>) -> Val {
        if matches!(v.node.node, TypedExpression::Literal(_)) {
            return v;
        }
        let span = v.node.span;
        let name = self.temp();
        let ty = self.ir(v.ty);
        pre.push(let_(name, ty.clone(), v.node, span));
        Val {
            node: var(name, ty, span),
            ty: v.ty,
        }
    }

    /// The value held in a temporary, so it can be read more than
    /// once: the statement binding it, and the read.
    fn hold(&mut self, v: Val, pre: &mut Vec<St>) -> Val {
        if Self::is_simple(&v.node) {
            return v;
        }
        let span = v.node.span;
        let name = self.temp();
        let ty = self.ir(v.ty);
        pre.push(let_(name, ty.clone(), v.node, span));
        Val {
            node: var(name, ty, span),
            ty: v.ty,
        }
    }

    /// `x % n` for a float `x` and a positive power of two `n`: the
    /// quotient, its floor and the product are all exact, and so is the
    /// difference; a zero keeps `x`'s sign, as fmod's does.
    fn float_mod_pow2(&mut self, x: Node, n: i64, span: Span) -> Node {
        let f64_t = prim(PrimitiveType::F64);
        let mut pre = Vec::new();
        let x = self
            .hold(
                Val {
                    node: x,
                    ty: Ty::Float,
                },
                &mut pre,
            )
            .node;
        let quotient = binary(
            BinaryOp::Mul,
            x.clone(),
            float_lit(1.0 / n as f64, span),
            f64_t.clone(),
            span,
        );
        let floored = call("floor", vec![quotient], f64_t.clone(), span);
        let product = binary(
            BinaryOp::Mul,
            floored,
            float_lit(n as f64, span),
            f64_t.clone(),
            span,
        );
        let r = self
            .hold(
                Val {
                    node: binary(BinaryOp::Sub, x.clone(), product, f64_t.clone(), span),
                    ty: Ty::Float,
                },
                &mut pre,
            )
            .node;
        let zero = binary(
            BinaryOp::Eq,
            r.clone(),
            float_lit(0.0, span),
            prim(PrimitiveType::Bool),
            span,
        );
        let signed_zero = binary(BinaryOp::Mul, x, float_lit(0.0, span), f64_t.clone(), span);
        block_value(pre, if_value(zero, signed_zero, r, f64_t, span), span)
    }

    // ─── slots ───────────────────────────────────────────────────
    // A shaped table's constant-key fields sit in typed slots after
    // the table header, with a bit in the header's `present` word for
    // each that holds a value.

    /// The slot a field has in a shape's tables, when they have slots.
    fn slot_of(&self, k: ShapeId, name: &str) -> Option<(&'m ShapeLayout, Slot)> {
        let layout = self.m.layout(k)?;
        let slot = layout.slots.iter().find(|s| s.name == name)?;
        Some((layout, slot.clone()))
    }

    /// The finder for `name` looked up from a table of shape `k`, in
    /// `mode`, made on first use: a function checking each lookup in
    /// the order the runtime looks, on to the next where a check fails.
    /// None when the types do not follow the lookups to their ends,
    /// when a table on the way has no slots, or when the name is
    /// nowhere along them.
    fn finder(&mut self, k: ShapeId, name: &str, mode: FinderMode, span: Span) -> Option<Finder> {
        let key = (k, name.to_string(), mode);
        if let Some(found) = self.m.finders.borrow().get(&key) {
            return found.clone();
        }
        let made = self.make_finder(k, name, mode, span);
        self.m.finders.borrow_mut().insert(key, made.clone());
        made
    }

    /// A `Table` finder covers the lookups through metatables to a
    /// slot; `Which` and `Where` cover every lookup, the table's own
    /// field and `__index` functions included.
    fn make_finder(
        &mut self,
        k: ShapeId,
        name: &str,
        mode: FinderMode,
        span: Span,
    ) -> Option<Finder> {
        let mut lookups: Vec<types::Lookup> = self.m.inferred.lookups(k, name)?;
        if mode == FinderMode::Table {
            if lookups
                .iter()
                .any(|l| matches!(l.hops.last(), Some(Hop::Handler(..))))
            {
                return None;
            }
            lookups.retain(|l| !l.hops.is_empty());
        }
        if lookups.is_empty() {
            return None;
        }
        let table_t = self.ir(Ty::Table);
        let bool_t = prim(PrimitiveType::Bool);
        let i64_t = prim(PrimitiveType::I64);
        let t = || var(intern("t"), table_t.clone(), span);
        let not_null = |x: &Node| {
            binary(
                BinaryOp::Ne,
                x.clone(),
                null(table_t.clone(), span),
                bool_t.clone(),
                span,
            )
        };
        let mut ends: Vec<(ShapeId, End)> = Vec::new();
        let mut body = Vec::new();
        if mode == FinderMode::Table {
            body.push(if_(
                binary(
                    BinaryOp::Eq,
                    t(),
                    null(table_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                vec![ret(Some(null(table_t.clone(), span)), span)],
                None,
                span,
            ));
        }
        for lookup in &lookups {
            let end = match lookup.hops.last() {
                Some(Hop::Handler(class, f)) => End::Handler {
                    class: *class,
                    f: *f,
                    ty: lookup.ty,
                },
                _ => {
                    self.slot_of(lookup.shape, name)?;
                    End::Slot(lookup.ty)
                }
            };
            let at = match ends.iter().position(|e| *e == (lookup.shape, end)) {
                Some(i) => i,
                None => {
                    ends.push((lookup.shape, end));
                    ends.len() - 1
                }
            };
            // The table each hop starts from: `t`, then a temporary per
            // hop that moves on. `origin` is the one the last `Meta`
            // hop started from.
            let mut x = t();
            let mut origin = t();
            let mut hops: Vec<(Hop, Node, Node)> = Vec::new();
            for hop in &lookup.hops {
                let to = match hop {
                    Hop::Meta(_) | Hop::Index(_) => var(self.temp(), table_t.clone(), span),
                    Hop::Absent(_) | Hop::Handler(..) => x.clone(),
                };
                if let Hop::Meta(_) = hop {
                    origin = x.clone();
                }
                hops.push((*hop, x, to.clone()));
                x = to;
            }
            // Inside out: the end, then each hop around it.
            let found = match (mode, end) {
                (FinderMode::Which, _) => int_lit(at as i64, span),
                (_, End::Handler { .. }) => origin,
                _ => x.clone(),
            };
            let mut inner = match end {
                End::Slot(_) if !lookup.sure => {
                    let (_, slot) = self.slot_of(lookup.shape, name)?;
                    vec![if_(
                        self.slot_present(&x, &slot, span),
                        vec![ret(Some(found), span)],
                        None,
                        span,
                    )]
                }
                _ => vec![ret(Some(found), span)],
            };
            for (hop, from, to) in hops.into_iter().rev() {
                inner = match hop {
                    Hop::Meta(m) => {
                        let TypedExpression::Variable(to_name) = &to.node else {
                            unreachable!("a hop's table is a variable")
                        };
                        let layout = self.m.layout(m)?;
                        let shape_is = binary(
                            BinaryOp::Eq,
                            field(to.clone(), "shape", i64_t.clone(), span),
                            int_lit(layout.gid, span),
                            bool_t.clone(),
                            span,
                        );
                        vec![
                            let_(
                                *to_name,
                                table_t.clone(),
                                field(from, "meta", table_t.clone(), span),
                                span,
                            ),
                            if_(
                                not_null(&to),
                                vec![if_(shape_is, inner, None, span)],
                                None,
                                span,
                            ),
                        ]
                    }
                    Hop::Absent(a) => {
                        let (_, slot) = self.slot_of(a, name)?;
                        let absent = binary(
                            BinaryOp::Eq,
                            binary(
                                BinaryOp::BitAnd,
                                self.present_of(&from, span),
                                int_lit(slot.mask(), span),
                                i64_t.clone(),
                                span,
                            ),
                            int_lit(0, span),
                            bool_t.clone(),
                            span,
                        );
                        vec![if_(absent, inner, None, span)]
                    }
                    Hop::Index(i) => {
                        let TypedExpression::Variable(to_name) = &to.node else {
                            unreachable!("a hop's table is a variable")
                        };
                        let (layout, slot) = self.slot_of(i, "__index")?;
                        if slot.kind != SlotKind::Table {
                            return None;
                        }
                        let mut through = vec![let_(
                            *to_name,
                            table_t.clone(),
                            self.slot_read(&from, layout, &slot, span).node,
                            span,
                        )];
                        through.extend(inner);
                        vec![if_(
                            self.slot_present(&from, &slot, span),
                            through,
                            None,
                            span,
                        )]
                    }
                    // The metatable holds the function: the lookup ends
                    // in its call, else in nil.
                    Hop::Handler(class, _) => {
                        let (_, slot) = self.slot_of(class, "__index")?;
                        if slot.kind != SlotKind::Any {
                            return None;
                        }
                        vec![if_(
                            self.slot_present(&from, &slot, span),
                            inner,
                            None,
                            span,
                        )]
                    }
                };
            }
            body.extend(inner);
        }
        // Nothing found.
        let (rest, returns, flavor) = match mode {
            FinderMode::Table => (null(table_t.clone(), span), table_t.clone(), "find"),
            FinderMode::Which => (int_lit(-1, span), i64_t, "which"),
            FinderMode::Where => (null(table_t.clone(), span), table_t.clone(), "where"),
        };
        body.push(ret(Some(rest), span));
        let helper = format!("lua${}{flavor}${}${}", self.m.tag, k.0, name);
        self.m.functions.borrow_mut().push(typed_function(
            &helper,
            vec![parameter(intern("t"), table_t, span)],
            returns,
            body,
            span,
        ));
        Some(Finder { helper, ends })
    }

    /// The sort for `table.sort` over tables of shape `k` by the
    /// function `f`: the library's quicksort over the array part, the
    /// comparator called directly with the elements as their type.
    /// `lua$sort$<k>$<f>(arr, lo, hi, line, rec)`, `line` the sort's
    /// own for the error it raises, `rec` the comparator's record.
    fn sort_helper(&mut self, k: ShapeId, f: FuncId, span: Span) -> String {
        let key = (k, f);
        if let Some(helper) = self.m.sorts.borrow().get(&key) {
            return helper.clone();
        }
        let helper = format!("lua${}sort${}${}", self.m.tag, k.0, f.0);
        let mut lowerer = Lowerer::new(self.m, CHUNK);
        lowerer.returns = Returns::Fixed(Vec::new());
        let anys = self.m.anys();
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let unit = prim(PrimitiveType::Unit);
        let arr = || var(intern("arr"), anys.clone(), span);
        let lo = || var(intern("lo"), i64_t.clone(), span);
        let hi = || var(intern("hi"), i64_t.clone(), span);
        let line = || var(intern("line"), i64_t.clone(), span);
        let rec = || var(intern("rec"), Type::Any, span);
        let i = intern("i");
        let j = intern("j");
        let pivot = intern("pivot");
        let iv = || var(i, i64_t.clone(), span);
        let jv = || var(j, i64_t.clone(), span);
        let at = |x: Node| index(arr(), x, Type::Any, span);
        let record = Val {
            node: rec(),
            ty: Ty::Any,
        };
        // `less(a, b)`: the comparator's first result, as a truth.
        let less = |lowerer: &mut Lowerer<'m, 'a>, a: Node, b: Node| -> Node {
            let multi = lowerer.direct_call_vals(
                f,
                Some(record.clone()),
                Vec::new(),
                vec![
                    Val {
                        node: a,
                        ty: Ty::Any,
                    },
                    Val {
                        node: b,
                        ty: Ty::Any,
                    },
                ],
                None,
                &None,
                span,
            );
            let first = match multi {
                Multi::Fixed(vals) => vals
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| lowerer.nil_val(span)),
                Multi::Dynamic(node) => Val {
                    node: call("zl_first", vec![node], Type::Any, span),
                    ty: Ty::Any,
                },
                Multi::None(node) => Val {
                    node: block_value(vec![expr_stmt(node)], nil(span), span),
                    ty: Ty::Nil,
                },
            };
            lowerer.truthy(first)
        };
        let step =
            |x: Node, by: i64| binary(BinaryOp::Add, x, int_lit(by, span), i64_t.clone(), span);
        let cmp = |op: BinaryOp, a: Node, b: Node| binary(op, a, b, bool_t.clone(), span);
        let less_i = less(&mut lowerer, at(iv()), var(pivot, Type::Any, span));
        let less_j = less(&mut lowerer, var(pivot, Type::Any, span), at(jv()));
        let tmp = intern("tmp");
        let body = vec![
            if_(
                cmp(BinaryOp::Ge, lo(), hi()),
                vec![ret(None, span)],
                None,
                span,
            ),
            let_(
                pivot,
                Type::Any,
                at(binary(
                    BinaryOp::Div,
                    binary(BinaryOp::Add, lo(), hi(), i64_t.clone(), span),
                    int_lit(2, span),
                    i64_t.clone(),
                    span,
                )),
                span,
            ),
            let_(i, i64_t.clone(), lo(), span),
            let_(j, i64_t.clone(), hi(), span),
            while_(
                cmp(BinaryOp::Le, iv(), jv()),
                vec![
                    // An order that never settles walks off the range:
                    // the comparator is not one.
                    while_(
                        binary(
                            BinaryOp::And,
                            cmp(BinaryOp::Le, iv(), hi()),
                            less_i,
                            bool_t.clone(),
                            span,
                        ),
                        vec![assign(iv(), step(iv(), 1), span)],
                        span,
                    ),
                    while_(
                        binary(
                            BinaryOp::And,
                            cmp(BinaryOp::Ge, jv(), lo()),
                            less_j,
                            bool_t.clone(),
                            span,
                        ),
                        vec![assign(jv(), step(jv(), -1), span)],
                        span,
                    ),
                    if_(
                        binary(
                            BinaryOp::Or,
                            cmp(BinaryOp::Gt, iv(), hi()),
                            cmp(BinaryOp::Lt, jv(), lo()),
                            bool_t.clone(),
                            span,
                        ),
                        vec![
                            assign(
                                var(intern(library::LINE), i64_t.clone(), span),
                                line(),
                                span,
                            ),
                            expr_stmt(call(
                                "zb_fatal",
                                vec![
                                    str_lit("error", span),
                                    str_lit("invalid order function for sorting", span),
                                ],
                                unit.clone(),
                                span,
                            )),
                            ret(None, span),
                        ],
                        None,
                        span,
                    ),
                    if_(
                        cmp(BinaryOp::Le, iv(), jv()),
                        vec![
                            let_(tmp, Type::Any, at(iv()), span),
                            assign(at(iv()), at(jv()), span),
                            assign(at(jv()), var(tmp, Type::Any, span), span),
                            assign(iv(), step(iv(), 1), span),
                            assign(jv(), step(jv(), -1), span),
                        ],
                        None,
                        span,
                    ),
                ],
                span,
            ),
            if_(
                binary(
                    BinaryOp::Ne,
                    Lowerer::pending(span),
                    nil(span),
                    bool_t.clone(),
                    span,
                ),
                vec![ret(None, span)],
                None,
                span,
            ),
            expr_stmt(call(
                &helper,
                vec![arr(), lo(), jv(), line(), rec()],
                unit.clone(),
                span,
            )),
            expr_stmt(call(
                &helper,
                vec![arr(), iv(), hi(), line(), rec()],
                unit,
                span,
            )),
            ret(None, span),
        ];
        let params = vec![
            parameter(intern("arr"), anys, span),
            parameter(intern("lo"), i64_t.clone(), span),
            parameter(intern("hi"), i64_t.clone(), span),
            parameter(intern("line"), i64_t, span),
            parameter(intern("rec"), Type::Any, span),
        ];
        self.m.functions.borrow_mut().push(typed_function(
            &helper,
            params,
            prim(PrimitiveType::Unit),
            body,
            span,
        ));
        self.m.sorts.borrow_mut().insert(key, helper.clone());
        helper
    }

    /// The value a finder found, read from `found` (held, not null) as
    /// `ty`: the slot the field has in the shape of `found`.
    fn found_value(
        &mut self,
        finder: &Finder,
        found: &Node,
        name: &str,
        ty: Ty,
        span: Span,
    ) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let mut value: Option<Node> = None;
        for (shape, _) in finder.ends.iter().rev() {
            let Some((layout, slot)) = self.slot_of(*shape, name) else {
                continue;
            };
            let read = self.slot_read(found, layout, &slot, span);
            let read = self.coerce(read, ty);
            value = Some(match value {
                None => read,
                Some(rest) => if_value(
                    binary(
                        BinaryOp::Eq,
                        field(found.clone(), "shape", i64_t.clone(), span),
                        int_lit(layout.gid, span),
                        bool_t.clone(),
                        span,
                    ),
                    read,
                    rest,
                    self.ir(ty),
                    span,
                ),
            });
        }
        value.unwrap_or_else(|| self.zero_of(ty, span))
    }

    /// The `present` word of table `t`.
    fn present_of(&self, t: &Node, span: Span) -> Node {
        field(t.clone(), "present", prim(PrimitiveType::I64), span)
    }

    /// Whether slot `slot` of table `t` holds a value.
    fn slot_present(&self, t: &Node, slot: &Slot, span: Span) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        binary(
            BinaryOp::Ne,
            binary(
                BinaryOp::BitAnd,
                self.present_of(t, span),
                int_lit(slot.mask(), span),
                i64_t,
                span,
            ),
            int_lit(0, span),
            prim(PrimitiveType::Bool),
            span,
        )
    }

    /// The value in slot `slot` of table `t`, as the slot stores it.
    fn slot_read(&self, t: &Node, layout: &ShapeLayout, slot: &Slot, span: Span) -> Val {
        let shaped = as_shape(t.clone(), self.m.shape_ty(layout), span);
        let base = format!("s{}", slot.bit);
        let node = match slot.kind {
            SlotKind::Number => return self.number_slot_read(t, layout, slot, Ty::Number, span),
            SlotKind::Scalar => number_value(
                field(
                    shaped.clone(),
                    &format!("{base}t"),
                    prim(PrimitiveType::I64),
                    span,
                ),
                field(
                    shaped.clone(),
                    &format!("{base}i"),
                    prim(PrimitiveType::I64),
                    span,
                ),
                field(shaped, &format!("{base}f"), prim(PrimitiveType::F64), span),
                span,
            ),
            _ => field(shaped, &base, self.ir(slot.stored_ty()), span),
        };
        Val {
            node,
            ty: slot.stored_ty(),
        }
    }

    /// The value in slot `slot` of table `t` read as `ty`: a Number
    /// slot's float word alone for `Float` and its integer word alone
    /// for `Int`, which the caller knows the slot to hold; else as the
    /// slot stores it.
    fn slot_read_as(&self, t: &Node, layout: &ShapeLayout, slot: &Slot, ty: Ty, span: Span) -> Val {
        match slot.kind {
            SlotKind::Number => self.number_slot_read(t, layout, slot, ty, span),
            _ => self.slot_read(t, layout, slot, span),
        }
    }

    /// A word of Number slot `slot` of table `t`: `i` or `f`.
    fn number_slot_word(
        &self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        word: char,
        span: Span,
    ) -> Node {
        let shaped = as_shape(t.clone(), self.m.shape_ty(layout), span);
        let ty = match word {
            'i' => prim(PrimitiveType::I64),
            _ => prim(PrimitiveType::F64),
        };
        field(shaped, &format!("s{}{word}", slot.bit), ty, span)
    }

    /// Number slot `slot` of table `t` read as `ty`: one word for
    /// `Float` or `Int`, else the number its kind bit tells.
    fn number_slot_read(
        &self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        ty: Ty,
        span: Span,
    ) -> Val {
        let i64_t = prim(PrimitiveType::I64);
        match ty {
            Ty::Float => Val {
                node: self.number_slot_word(t, layout, slot, 'f', span),
                ty: Ty::Float,
            },
            Ty::Int => Val {
                node: self.number_slot_word(t, layout, slot, 'i', span),
                ty: Ty::Int,
            },
            _ => {
                let is_int = binary(
                    BinaryOp::Ne,
                    binary(
                        BinaryOp::BitAnd,
                        self.present_of(t, span),
                        int_lit(slot.kmask(), span),
                        i64_t,
                        span,
                    ),
                    int_lit(0, span),
                    prim(PrimitiveType::Bool),
                    span,
                );
                Val {
                    node: number_value(
                        tag_of_is_int(is_int, span),
                        self.number_slot_word(t, layout, slot, 'i', span),
                        self.number_slot_word(t, layout, slot, 'f', span),
                        span,
                    ),
                    ty: Ty::Number,
                }
            }
        }
    }

    /// Statements storing `value` into slot `slot` of table `t`, and
    /// setting its bit; a value that may be nil clears the bit when it
    /// is. With `settled` the bit is set for good (a field born with
    /// the table that nothing clears) and left alone. A Number slot
    /// takes a value of its own type, `Int` or `Float`, as it is.
    fn slot_store(
        &mut self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        value: Val,
        settled: bool,
        span: Span,
    ) -> Vec<St> {
        if slot.kind == SlotKind::Number {
            return self.number_slot_store(t, layout, slot, value, settled, span);
        }
        let i64_t = prim(PrimitiveType::I64);
        let shaped = as_shape(t.clone(), self.m.shape_ty(layout), span);
        let base = format!("s{}", slot.bit);
        let mut out = Vec::new();
        let value = self.coerce(value, slot.stored_ty());
        let value = self
            .hold(
                Val {
                    node: value,
                    ty: slot.stored_ty(),
                },
                &mut out,
            )
            .node;
        match slot.kind {
            SlotKind::Scalar => {
                let parts = NumberParts::of(&value, span);
                out.push(assign(
                    field(shaped.clone(), &format!("{base}t"), i64_t.clone(), span),
                    parts.tag,
                    span,
                ));
                out.push(assign(
                    field(shaped.clone(), &format!("{base}i"), i64_t.clone(), span),
                    parts.int,
                    span,
                ));
                out.push(assign(
                    field(shaped, &format!("{base}f"), prim(PrimitiveType::F64), span),
                    parts.float,
                    span,
                ));
            }
            _ => out.push(assign(
                field(shaped, &base, self.ir(slot.stored_ty()), span),
                value.clone(),
                span,
            )),
        }
        if settled {
            return out;
        }
        let present = self.present_of(t, span);
        let set = binary(
            BinaryOp::BitOr,
            present.clone(),
            int_lit(slot.mask(), span),
            i64_t.clone(),
            span,
        );
        let updated = match self.slot_nil(&value, slot, span) {
            Some(is_nil) => {
                let cleared = binary(
                    BinaryOp::BitAnd,
                    present.clone(),
                    int_lit(!slot.mask(), span),
                    i64_t.clone(),
                    span,
                );
                if_value(is_nil, cleared, set, i64_t, span)
            }
            None => set,
        };
        out.push(assign(present, updated, span));
        out
    }

    /// Statements storing `value` into Number slot `slot` of table `t`:
    /// a float writes `f` and clears the kind bit, an integer writes
    /// `i` and sets it, a number of run-time kind writes both words
    /// and sets the bit from its tag. The presence bit is set unless
    /// `settled`.
    fn number_slot_store(
        &mut self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        value: Val,
        settled: bool,
        span: Span,
    ) -> Vec<St> {
        let i64_t = prim(PrimitiveType::I64);
        let mut out = Vec::new();
        let int_word = self.number_slot_word(t, layout, slot, 'i', span);
        let float_word = self.number_slot_word(t, layout, slot, 'f', span);
        let present = self.present_of(t, span);
        let or = |a: Node, b: Node| binary(BinaryOp::BitOr, a, b, i64_t.clone(), span);
        let with_mask = |p: Node| {
            if settled {
                p
            } else {
                or(p, int_lit(slot.mask(), span))
            }
        };
        let without_kind = |p: Node| {
            binary(
                BinaryOp::BitAnd,
                p,
                int_lit(!slot.kmask(), span),
                i64_t.clone(),
                span,
            )
        };
        match value.ty {
            Ty::Float => {
                out.push(assign(float_word, value.node, span));
                out.push(assign(
                    present.clone(),
                    without_kind(with_mask(present.clone())),
                    span,
                ));
            }
            Ty::Int => {
                out.push(assign(int_word, value.node, span));
                let bits = or(present.clone(), int_lit(slot.kmask(), span));
                out.push(assign(present.clone(), with_mask(bits), span));
            }
            _ => {
                let value = Val {
                    node: self.coerce(value, Ty::Number),
                    ty: Ty::Number,
                };
                let (pre, n) = self.number_parts(value);
                out.extend(pre);
                let kind = if_value(
                    n.is_int(span),
                    int_lit(slot.kmask(), span),
                    int_lit(0, span),
                    i64_t.clone(),
                    span,
                );
                out.push(assign(int_word, n.int, span));
                out.push(assign(float_word, n.float, span));
                let bits = or(without_kind(with_mask(present.clone())), kind);
                out.push(assign(present, bits, span));
            }
        }
        out
    }

    /// A new table's Number slot `slot` holding `value`, or nothing:
    /// its `present` bits and its two words' initializers, plain reads.
    fn number_slot_init(
        &mut self,
        slot: &Slot,
        value: Option<Val>,
        pre: &mut Vec<St>,
        span: Span,
    ) -> (Node, Vec<TypedFieldInit>) {
        let i64_t = prim(PrimitiveType::I64);
        let (bits, int, float) = match value.map(|v| self.slot_value(v, slot)) {
            None => (int_lit(0, span), int_lit(0, span), float_lit(0.0, span)),
            Some(v) if v.ty == Ty::Int => (
                int_lit(slot.mask() | slot.kmask(), span),
                self.hold(v, pre).node,
                float_lit(0.0, span),
            ),
            Some(v) if v.ty == Ty::Float => (
                int_lit(slot.mask(), span),
                int_lit(0, span),
                self.hold(v, pre).node,
            ),
            Some(v) => {
                let (held, n) = self.number_parts(v);
                pre.extend(held);
                let kind = if_value(
                    n.is_int(span),
                    int_lit(slot.mask() | slot.kmask(), span),
                    int_lit(slot.mask(), span),
                    i64_t.clone(),
                    span,
                );
                (kind, n.int, n.float)
            }
        };
        let words = [("i", int), ("f", float)]
            .into_iter()
            .map(|(suffix, value)| TypedFieldInit {
                name: intern(&format!("s{}{suffix}", slot.bit)),
                value: Box::new(value),
            })
            .collect();
        (bits, words)
    }

    /// `v` as a store into `slot` takes it: an integer or a float as it
    /// is for a Number slot, else of the slot's stored type.
    fn slot_value(&mut self, v: Val, slot: &Slot) -> Val {
        match (slot.kind, v.ty) {
            (SlotKind::Number, Ty::Int | Ty::Float | Ty::Number) => v,
            _ => Val {
                node: self.coerce(v, slot.stored_ty()),
                ty: slot.stored_ty(),
            },
        }
    }

    /// Whether a value of the slot's stored type is nil, when it can be.
    fn slot_nil(&self, value: &Node, slot: &Slot, span: Span) -> Option<Node> {
        let bool_t = prim(PrimitiveType::Bool);
        match slot.kind {
            SlotKind::Scalar => Some(NumberParts::of(value, span).has_tag(TAG_NIL, span)),
            SlotKind::Any => Some(binary(BinaryOp::Eq, value.clone(), nil(span), bool_t, span)),
            SlotKind::Table => Some(binary(
                BinaryOp::Eq,
                value.clone(),
                null(self.ir(Ty::Table), span),
                bool_t,
                span,
            )),
            _ => None,
        }
    }

    /// The parts of a number, held first unless it is a plain read.
    fn number_parts(&mut self, v: Val) -> (Vec<St>, NumberParts) {
        let mut pre = Vec::new();
        let parts = self.number_parts_into(v, &mut pre);
        (pre, parts)
    }

    /// The parts of number `v`, with what computes them pushed on
    /// `pre`. A number built in place, alone or at the end of a block,
    /// is held part by part, so no aggregate is made only to be taken
    /// apart.
    fn number_parts_into(&mut self, v: Val, pre: &mut Vec<St>) -> NumberParts {
        let span = v.node.span;
        let ty = v.ty;
        match v.node.node {
            TypedExpression::Block(mut block)
                if matches!(
                    block.statements.last().map(|s| &s.node),
                    Some(TypedStatement::Expression(_))
                ) =>
            {
                let Some(St {
                    node: TypedStatement::Expression(tail),
                    ..
                }) = block.statements.pop()
                else {
                    unreachable!("the block ends in an expression")
                };
                pre.extend(block.statements);
                self.number_parts_into(Val { node: *tail, ty }, pre)
            }
            TypedExpression::Tuple(parts) if parts.len() == 3 => {
                let mut held = parts.into_iter().zip([Ty::Int, Ty::Int, Ty::Float]);
                let mut next = || {
                    let (node, ty) = held.next().expect("three parts");
                    self.hold(Val { node, ty }, pre).node
                };
                NumberParts {
                    tag: next(),
                    int: next(),
                    float: next(),
                }
            }
            node => {
                let held = self.hold(
                    Val {
                        node: Node {
                            node,
                            ty: v.node.ty,
                            span,
                        },
                        ty,
                    },
                    pre,
                );
                NumberParts::of(&held.node, span)
            }
        }
    }

    /// A list literal whose items are plain reads: any other item is
    /// bound ahead of the literal, since the lowering stores the elements
    /// in one block and an item may carry control flow (a check after a
    /// call) below a wrapper.
    fn array_of(&mut self, items: Vec<Node>, pre: &mut Vec<St>, span: Span) -> Node {
        let mut plain = Vec::with_capacity(items.len());
        for item in items {
            if Self::is_simple(&item) {
                plain.push(item);
            } else {
                let name = self.temp();
                let ty = item.ty.clone();
                pre.push(let_(name, ty.clone(), item, span));
                plain.push(var(name, ty, span));
            }
        }
        node(TypedExpression::Array(plain), self.m.anys(), span)
    }

    /// The truth of a value, as a boolean.
    fn truthy(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Bool => v.node,
            // False when nil or false: the tags below true.
            Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => {
                let (pre, n) = self.number_parts(v);
                block_value(pre, n.is_true(span), span)
            }
            // A shaped table is nil when null.
            Ty::Shape(_) => binary(
                BinaryOp::Ne,
                v.node,
                null(self.ir(Ty::Table), span),
                prim(PrimitiveType::Bool),
                span,
            ),
            Ty::Nil => block_value(vec![expr_stmt(v.node)], bool_lit(false, span), span),
            // A function value may be nil.
            Ty::Func(_) => binary(
                BinaryOp::Ne,
                v.node,
                nil(span),
                prim(PrimitiveType::Bool),
                span,
            ),
            Ty::Int | Ty::Float | Ty::Number | Ty::Str | Ty::Table => {
                if Self::is_simple(&v.node) {
                    bool_lit(true, span)
                } else {
                    block_value(vec![expr_stmt(v.node)], bool_lit(true, span), span)
                }
            }
            _ => call("zl_truthy", vec![v.node], prim(PrimitiveType::Bool), span),
        }
    }

    // ─── errors ─────────────────────────────────────────────────

    /// The placeholder this function leaves with once an error is
    /// pending: whoever called it checks next.
    fn placeholder_return(&mut self, span: Span) -> St {
        if !self.tbc.is_empty() {
            // The error has left the function when its `<close>`
            // variables are closed: its frame goes first.
            let mut statements = self.debug_call("zl_dbg_leave", Vec::new(), span);
            self.closes_from(1, span, &mut statements);
            statements.push(self.placeholder_return_plain(span, false));
            return stmt(TypedStatement::Block(TypedBlock { statements, span }), span);
        }
        self.placeholder_return_plain(span, true)
    }

    /// Leaving with the placeholder, the depth uncounted and, when
    /// `pop`, the frame popped.
    fn placeholder_return_plain(&mut self, span: Span, pop: bool) -> St {
        let leave = self.placeholder_value(span);
        if self.counts_depth || (self.frames && pop) {
            let mut statements = Vec::new();
            if self.counts_depth {
                statements.push(depth_step(-1, span));
            }
            if pop {
                statements.extend(self.debug_call("zl_dbg_leave", Vec::new(), span));
            }
            statements.push(leave);
            return stmt(TypedStatement::Block(TypedBlock { statements, span }), span);
        }
        leave
    }

    fn placeholder_value(&mut self, span: Span) -> St {
        let value = match self.returns.clone() {
            Returns::Fixed(types) if types.is_empty() => None,
            Returns::Fixed(types) if types.len() == 1 => {
                Some(self.zero_of(types[0].settled(), span))
            }
            Returns::Fixed(_) => Some(node(
                TypedExpression::Array(Vec::new()),
                self.m.anys(),
                span,
            )),
            Returns::Dynamic | Returns::Unknown => Some(nil(span)),
        };
        ret(value, span)
    }

    fn zero_of(&mut self, ty: Ty, span: Span) -> Node {
        match ty {
            Ty::Int => int_lit(0, span),
            Ty::Float => float_lit(0.0, span),
            Ty::Number => number_of_int(int_lit(0, span), span),
            Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => scalar_of_nil(span),
            Ty::Bool => bool_lit(false, span),
            Ty::Str => str_lit("", span),
            Ty::Shape(_) => null(self.ir(Ty::Table), span),
            Ty::Table => call("zl_table_new", vec![], self.ir(Ty::Table), span),
            _ => nil(span),
        }
    }

    fn pending(span: Span) -> Node {
        var(intern(library::PENDING), Type::Any, span)
    }

    /// `if an error is pending, leave`; then the line is this one
    /// again, since a callee sets its own (unless nothing reads it
    /// before it is stored again: see [`LineRestores`]). A type error's
    /// note of which operand it is about is cleared on the way: it was
    /// a deeper site's.
    fn pending_check(&mut self, span: Span) -> St {
        self.pending_check_described(span, &Described::NONE)
    }

    /// The same, the error described first: a type error about an
    /// operand of this site gets that operand's description appended,
    /// `(local 'x')` and the like, the way the reference names the
    /// variable.
    fn pending_check_described(&mut self, span: Span, descs: &Described) -> St {
        let cond = binary(
            BinaryOp::Ne,
            Self::pending(span),
            nil(span),
            prim(PrimitiveType::Bool),
            span,
        );
        let leave = self.leave_described(span, descs);
        let restore = self.set_line(span);
        if_(cond, leave, Some(vec![restore]), span)
    }

    /// Leaving with the pending error, described as above.
    fn leave_described(&mut self, span: Span, descs: &Described) -> Vec<St> {
        self.raised = true;
        self.line_needed = true;
        let settle = if descs.operands.iter().all(Option::is_none) {
            assign(
                var(intern(library::VARINFO), prim(PrimitiveType::I64), span),
                int_lit(0, span),
                span,
            )
        } else {
            let text = |d: &Desc| {
                let d = d.as_deref().map(|d| format!(" ({d})")).unwrap_or_default();
                str_lit(&d, span)
            };
            assign(
                Self::pending(span),
                call(
                    "zl_annotate",
                    vec![
                        Self::pending(span),
                        text(&descs.operands[0]),
                        text(&descs.operands[1]),
                        bool_lit(descs.call, span),
                    ],
                    Type::Any,
                    span,
                ),
                span,
            )
        };
        let leave = self.placeholder_return(span);
        vec![settle, leave]
    }

    /// `zl_depth += 1; if zl_depth > limit { raise; leave }`: one more
    /// frame, refused past what the reference allows.
    fn stack_check(&mut self, span: Span) -> St {
        let i64_t = prim(PrimitiveType::I64);
        let depth = var(intern(library::DEPTH), i64_t, span);
        let over = binary(
            BinaryOp::Gt,
            depth,
            int_lit(library::MAX_DEPTH, span),
            prim(PrimitiveType::Bool),
            span,
        );
        let leave = self.placeholder_return(span);
        let statements = vec![
            depth_step(1, span),
            if_(
                over,
                vec![
                    expr_stmt(call(
                        "zl_stack_overflow",
                        vec![],
                        prim(PrimitiveType::Unit),
                        span,
                    )),
                    leave,
                ],
                None,
                span,
            ),
        ];
        stmt(TypedStatement::Block(TypedBlock { statements, span }), span)
    }

    /// `zl_line = <the line of span>`.
    fn set_line(&mut self, span: Span) -> St {
        let line = self.m.line_of(span);
        assign(
            var(intern(library::LINE), prim(PrimitiveType::I64), span),
            int_lit(line, span),
            span,
        )
    }

    // ─── the debug library's view ───────────────────────────────

    /// `zl_line = <line>`, then the line and count hooks when a hook
    /// is set: a statement starts.
    fn debug_line(&mut self, line: i64, span: Span) -> Vec<St> {
        let i64_t = prim(PrimitiveType::I64);
        self.lines.insert(line & ((1i64 << library::LINE_BITS) - 1));
        vec![
            assign(
                var(intern(library::LINE), i64_t.clone(), span),
                int_lit(line, span),
                span,
            ),
            self.if_hooked(
                vec![expr_stmt(call(
                    "zl_dbg_line",
                    vec![int_lit(line, span)],
                    prim(PrimitiveType::Unit),
                    span,
                ))],
                span,
            ),
        ]
    }

    /// `statements` when a hook is set.
    fn if_hooked(&self, statements: Vec<St>, span: Span) -> St {
        let i64_t = prim(PrimitiveType::I64);
        if_(
            binary(
                BinaryOp::Ne,
                var(intern(library::debug::HOOK_MASK), i64_t.clone(), span),
                int_lit(0, span),
                prim(PrimitiveType::Bool),
                span,
            ),
            statements,
            None,
            span,
        )
    }

    /// A call of a library function the debug library's hooks and
    /// frames go through, when this function keeps a frame.
    fn debug_call(&self, name: &str, args: Vec<Node>, span: Span) -> Vec<St> {
        if !self.frames {
            return Vec::new();
        }
        vec![expr_stmt(call(name, args, prim(PrimitiveType::Unit), span))]
    }

    /// A loop is about to run its body again: the line its head is on
    /// runs (`zl_dbg_loop`), or for a `while`, the jump back lands on
    /// its head (`back`).
    fn debug_loop_back(&self, head: Span, back: bool, span: Span) -> Vec<St> {
        if !self.frames {
            return Vec::new();
        }
        let line = self.m.line_of(head);
        let name = if back {
            "zl_dbg_loop_head"
        } else {
            "zl_dbg_loop"
        };
        vec![self.if_hooked(
            vec![expr_stmt(call(
                name,
                vec![int_lit(line, span)],
                prim(PrimitiveType::Unit),
                span,
            ))],
            span,
        )]
    }

    /// A loop whose head is on `head` was left; `end_line` is told to
    /// a line hook as the line after it, for a generic `for`.
    fn debug_loop_exit(&self, head: Span, end_line: Option<i64>, span: Span) -> Vec<St> {
        if !self.frames {
            return Vec::new();
        }
        let line = self.m.line_of(head);
        let mut hooked = vec![expr_stmt(call(
            "zl_dbg_loop_exit",
            vec![int_lit(line, span)],
            prim(PrimitiveType::Unit),
            span,
        ))];
        if let Some(end) = end_line {
            hooked.push(expr_stmt(call(
                "zl_dbg_line",
                vec![int_lit(end, span)],
                prim(PrimitiveType::Unit),
                span,
            )));
        }
        vec![self.if_hooked(hooked, span)]
    }

    /// `zl_dbg_enter(key)`: this function's frame, pushed.
    fn debug_enter(&self, f: FuncId, span: Span) -> St {
        expr_stmt(call(
            "zl_dbg_enter",
            vec![int_lit(self.m.func_key(f), span)],
            prim(PrimitiveType::Unit),
            span,
        ))
    }

    /// What the debug library is told of this function: where it is,
    /// its parameters and upvalues, the lines its statements start on
    /// and its parameters' names. A stripped function has no active
    /// lines, no parameter names and upvalues named `(no name)`.
    fn debug_function_records(&mut self, id: FuncId, line: i64, last_line: i64) {
        let mask = (1i64 << library::LINE_BITS) - 1;
        let info = self.scopes().func(id).clone();
        let stripped = self.m.stripped;
        let mut f = vec![
            "F".to_string(),
            id.0.to_string(),
            (line & mask).to_string(),
            (last_line & mask).to_string(),
            info.params.len().to_string(),
            if info.is_vararg { "1" } else { "0" }.to_string(),
        ];
        f.extend(info.upvalues.iter().map(|u| match u {
            _ if stripped => "(no name)".to_string(),
            crate::scope::Upvalue::Var(v) => self.scopes().var(*v).name.clone(),
            crate::scope::Upvalue::Env => "_ENV".to_string(),
        }));
        self.m.debug_record(&f);
        let mut lines = std::mem::take(&mut self.lines);
        if stripped {
            lines.clear();
        } else if id != CHUNK {
            lines.insert(last_line & mask);
        }
        let mut a = vec!["A".to_string(), id.0.to_string()];
        a.extend(lines.iter().map(|l| l.to_string()));
        self.m.debug_record(&a);
        let mut l = vec!["L".to_string(), id.0.to_string(), "0".to_string()];
        if !stripped {
            l.extend(
                info.params
                    .iter()
                    .map(|v| self.scopes().var(*v).name.clone()),
            );
        }
        self.m.debug_record(&l);
    }

    /// The names of the locals in scope, as `debug.getlocal` gives them:
    /// `(temporary)` each in a stripped chunk.
    fn live_names(&self) -> Vec<String> {
        self.live
            .iter()
            .map(|l| match l {
                _ if self.m.stripped => "(temporary)".to_string(),
                Live::Var(v) => self.scopes().var(*v).name.clone(),
                Live::ForState(_) => "(for state)".to_string(),
            })
            .collect()
    }

    /// The live locals kept in cells, a bit each by slot (the first 63
    /// slots): a spill holds such a local's cell, not its value, so a
    /// `debug.setlocal` reaches the closures sharing it at once.
    fn live_cells(&mut self) -> i64 {
        let mut mask = 0i64;
        for (i, l) in self.live.clone().into_iter().enumerate().take(63) {
            if let Live::Var(v) = l
                && matches!(self.storage_of(v), Storage::Cell(..))
            {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// A call site of this function, numbered for the debug library,
    /// with what the callee is called there (`desc`); the locals in
    /// scope recorded for `debug.getlocal`. None when this function
    /// keeps no frame.
    fn debug_site(&mut self, desc: &Desc, span: Span) -> Option<i64> {
        self.debug_site_as(desc, "", span)
    }

    /// [`Self::debug_site`] for a callee a traceback names `global`
    /// whatever the site calls it (a library function), unless that is
    /// empty.
    fn debug_site_as(&mut self, desc: &Desc, global: &str, span: Span) -> Option<i64> {
        if !self.frames {
            return None;
        }
        let m = self.m;
        let debug = m.debug.as_ref()?;
        let k = debug.sites.get() + 1;
        debug.sites.set(k);
        let (namewhat, name) = match desc.as_deref().and_then(|d| d.split_once(" '")) {
            Some((what, rest)) => (what.to_string(), rest.trim_end_matches('\'').to_string()),
            None => (String::new(), String::new()),
        };
        let global = if !global.is_empty() {
            global.to_string()
        } else if namewhat == "global" {
            name.clone()
        } else {
            String::new()
        };
        m.debug_record(&["S".to_string(), k.to_string(), namewhat, name, global]);
        if debug.locals {
            let mut record = vec!["L".to_string(), self.func.0.to_string(), k.to_string()];
            record.extend(self.live_names());
            m.debug_record(&record);
            let cells = self.live_cells();
            if cells != 0 {
                m.debug_record(&[
                    "C".to_string(),
                    self.func.0.to_string(),
                    k.to_string(),
                    cells.to_string(),
                ]);
            }
        }
        let tail = if self.tail_span == Some(span) {
            library::debug::TAIL_SITE
        } else {
            0
        };
        Some((m.chunk_index << 32) | k | tail)
    }

    /// What a call site does before its call, the arguments evaluated:
    /// the locals spilled (for `debug.getlocal`), the site stored, and
    /// the line the call starts on, which the arguments may have moved
    /// the frame off. The spill list's variable is returned, for
    /// [`Self::after_call`].
    fn before_call(
        &mut self,
        desc: &Desc,
        span: Span,
        pre: &mut Vec<St>,
    ) -> Option<InternedString> {
        let site = self.debug_site(desc, span)?;
        let spill = self.spill(site, span, pre);
        pre.push(self.set_line(span));
        pre.push(assign(
            var(
                intern(library::debug::DBG_SITE),
                prim(PrimitiveType::I64),
                span,
            ),
            int_lit(site, span),
            span,
        ));
        spill
    }

    /// The locals in scope (and a variadic function's extra arguments,
    /// after them) as a list handed to the host for the frame, when
    /// the program reads locals.
    fn spill(&mut self, site: i64, span: Span, pre: &mut Vec<St>) -> Option<InternedString> {
        if !self.debug_locals() {
            return None;
        }
        let mut items = Vec::with_capacity(self.live.len() + 1);
        let cells = self.live_cells();
        for (i, l) in self.live.clone().into_iter().enumerate() {
            items.push(match l {
                Live::Var(v) if i < 63 && cells & (1 << i) != 0 => self.capture_value(v, span),
                Live::Var(v) => {
                    let value = self.read_var(v, span);
                    self.boxed(value)
                }
                Live::ForState(i) => self.for_states[i].clone(),
            });
        }
        if let Some(varargs) = self.varargs
            && self.func != CHUNK
        {
            items.push(call(
                "zb_box_tuple",
                vec![var(varargs, self.m.anys(), span)],
                Type::Any,
                span,
            ));
        }
        let list = self.array_of(items, pre, span);
        let name = self.temp();
        pre.push(let_(
            name,
            Type::Any,
            call("zb_list_box_any", vec![list], Type::Any, span),
            span,
        ));
        pre.push(expr_stmt(call(
            "zl_dbg_spill",
            vec![var(name, Type::Any, span), int_lit(site, span)],
            prim(PrimitiveType::Unit),
            span,
        )));
        Some(name)
    }

    /// What a call site does once its call returned: the locals read
    /// back from the list they were spilled to, when a callee may
    /// have set them, and the list released.
    fn after_call(&mut self, spill: Option<InternedString>, span: Span) -> Vec<St> {
        let Some(spill) = spill else {
            return Vec::new();
        };
        let i64_t = prim(PrimitiveType::I64);
        let unspill = call(
            "zl_dbg_unspill",
            vec![var(spill, Type::Any, span)],
            i64_t.clone(),
            span,
        );
        if !self.m.debug.as_ref().is_some_and(|d| d.setlocal) {
            return vec![expr_stmt(unspill)];
        }
        // Each local a callee set is read back; the others keep what
        // the call left in them. Straight-line, as the call may sit in
        // an expression. A local in a cell was set through it.
        let set = self.temp();
        let mut out = vec![let_(set, i64_t.clone(), unspill, span)];
        let cells = self.live_cells();
        for (i, l) in self.live.clone().into_iter().enumerate().take(63) {
            let Live::Var(v) = l else {
                continue;
            };
            if cells & (1 << i) != 0 {
                continue;
            }
            let current = self.read_var(v, span);
            let current = self.boxed(current);
            let value = Val {
                node: call(
                    "zl_dbg_pick",
                    vec![
                        var(spill, Type::Any, span),
                        var(set, i64_t.clone(), span),
                        int_lit(i as i64, span),
                        current,
                    ],
                    Type::Any,
                    span,
                ),
                ty: Ty::Any,
            };
            out.push(self.write_var(v, value, span));
        }
        out
    }

    /// Whether a call node names a library function that can raise.
    fn call_can_raise(&self, node: &Node) -> bool {
        let TypedExpression::Call(c) = &node.node else {
            return false;
        };
        let TypedExpression::Variable(name) = &c.callee.node else {
            return false;
        };
        name.resolve_global()
            .is_some_and(|n| self.m.fallible.contains(n.as_str()))
    }

    /// A value from a call that may have raised: held, then checked
    /// before anything uses it. A call that cannot raise is left alone.
    fn guarded_described(&mut self, v: Val, descs: &Described) -> Val {
        if !self.call_can_raise(&v.node) {
            return v;
        }
        self.guard_described(v, descs)
    }

    fn guard(&mut self, v: Val) -> Val {
        self.guard_described(v, &Described::NONE)
    }

    fn guard_described(&mut self, v: Val, descs: &Described) -> Val {
        let span = v.node.span;
        let mut pre = Vec::new();
        let held = if v.node.ty == prim(PrimitiveType::Unit) {
            pre.push(expr_stmt(v.node));
            Val {
                node: nil(span),
                ty: Ty::Nil,
            }
        } else {
            self.hold(v, &mut pre)
        };
        pre.push(self.pending_check_described(span, descs));
        Val {
            node: block_value(pre, held.node, span),
            ty: held.ty,
        }
    }

    /// A statement calling something that may have raised, checked.
    fn guarded_stmt(&mut self, node: Node, descs: &Described) -> St {
        if !self.call_can_raise(&node) {
            return expr_stmt(node);
        }
        let span = node.span;
        let mut statements = vec![expr_stmt(node)];
        statements.push(self.pending_check_described(span, descs));
        stmt(TypedStatement::Block(TypedBlock { statements, span }), span)
    }

    // ─── what a value is called ─────────────────────────────────
    // The reference names the variable a type error is about: a local,
    // an upvalue, a global, a field, a constant or a method. These give
    // the same name for an expression, appended by the site's check.

    /// A stripped chunk knows no local's name, calls every upvalue `?`
    /// and cannot tell the environment from another table.
    fn describe_name(&self, token: &TokenReference) -> Desc {
        let name = ident(token);
        let stripped = self.m.stripped;
        Some(match self.scopes().binding(token) {
            Some(Binding::Local(_)) if stripped => return None,
            Some(Binding::Upvalue(_)) if stripped => "upvalue '?'".to_string(),
            Some(Binding::Local(_)) => format!("local '{name}'"),
            Some(Binding::Upvalue(_)) => format!("upvalue '{name}'"),
            _ if stripped => format!("field '{name}'"),
            _ => format!("global '{name}'"),
        })
    }

    /// The `_ENV` a free name is a field of, described as the variable
    /// it is at the use: what an index error on it names.
    fn env_var_desc(&self, upvalue: bool) -> Desc {
        match (upvalue, self.m.stripped) {
            (true, false) => Some("upvalue '_ENV'".to_string()),
            (true, true) => Some("upvalue '?'".to_string()),
            (false, false) => Some("local '_ENV'".to_string()),
            (false, true) => None,
        }
    }

    fn describe(&self, e: &Expression) -> Desc {
        match e {
            Expression::Var(Var::Name(token)) => self.describe_name(token),
            Expression::Var(Var::Expression(v)) => {
                let suffixes: Vec<&Suffix> = v.suffixes().collect();
                self.describe_chain(v.prefix(), &suffixes)
            }
            Expression::Parentheses { expression, .. } => self.describe(expression),
            Expression::String(t) => {
                let bytes = string_bytes(t).ok()?;
                Some(format!("constant '{}'", String::from_utf8_lossy(&bytes)))
            }
            _ => None,
        }
    }

    /// What a prefix with suffixes applied names: the last suffix, or
    /// the prefix itself without any.
    fn describe_chain(&self, prefix: &Prefix, suffixes: &[&Suffix]) -> Desc {
        let Some((last, init)) = suffixes.split_last() else {
            return match prefix {
                Prefix::Name(token) => self.describe_name(token),
                Prefix::Expression(e) => self.describe(e),
                _ => None,
            };
        };
        // `_ENV.x` is the global `x`.
        if init.is_empty()
            && let Prefix::Name(token) = prefix
            && ident(token) == "_ENV"
            && let Suffix::Index(ast::Index::Dot { name, .. }) = last
            && !self.m.stripped
        {
            return Some(format!("global '{}'", ident(name)));
        }
        Self::describe_suffix(last)
    }

    /// What an index suffix names its result: a field, by the key when
    /// that is a constant string, `integer index` for a small integer
    /// and `?` for any other key, as the reference has it.
    fn describe_suffix(s: &Suffix) -> Desc {
        let key = match s {
            Suffix::Index(ast::Index::Dot { name, .. }) => ident(name),
            Suffix::Index(ast::Index::Brackets { expression, .. }) => match expression {
                Expression::String(t) => match string_bytes(t) {
                    Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                    Err(_) => "?".to_string(),
                },
                Expression::Number(t) => {
                    match crate::host::parse_numeral(t.token().to_string().trim()) {
                        crate::host::Numeral::Int(0..=255) => "integer index".to_string(),
                        _ => "?".to_string(),
                    }
                }
                _ => "?".to_string(),
            },
            _ => return None,
        };
        Some(format!("field '{key}'"))
    }

    // ─── variables ──────────────────────────────────────────────

    fn var_ty(&self, v: VarId) -> Ty {
        self.m.inferred.var(v).settled()
    }

    /// Where a variable of this function or an enclosing one lives,
    /// deciding for one not seen yet.
    fn storage_of(&mut self, v: VarId) -> Storage {
        if let Some(s) = self.storage.get(&v) {
            return s.clone();
        }
        let info = self.scopes().var(v);
        let ty = self.var_ty(v);
        let s = if info.is_module_var() {
            let symbol = self.m.module_local_symbol(v);
            self.m.declare_module_var(symbol, ty);
            Storage::Module(symbol, ty)
        } else if info.needs_cell() {
            let symbol = intern(&format!("{}$cell{}", info.name, v.0));
            Storage::Cell(symbol, ty)
        } else {
            let symbol = self.local_symbol(v);
            Storage::Local(symbol, ty)
        };
        self.storage.insert(v, s.clone());
        s
    }

    /// Read a variable.
    fn read_var(&mut self, v: VarId, span: Span) -> Val {
        match self.storage_of(v) {
            Storage::Local(name, ty) | Storage::Module(name, ty) => Val {
                node: var(name, self.ir(ty), span),
                ty,
            },
            Storage::Cell(name, ty) => {
                let cell = var(name, self.m.anys(), span);
                let element = index(cell, int_lit(0, span), Type::Any, span);
                let node = self.coerce(
                    Val {
                        node: element,
                        ty: Ty::Any,
                    },
                    ty,
                );
                Val { node, ty }
            }
        }
    }

    /// Store into a variable.
    /// `name = v` where `name` is a field of a local or upvalue `_ENV`.
    fn write_field(&mut self, env: VarId, name: &str, upvalue: bool, value: Val, span: Span) -> St {
        let obj = self.read_var(env, span);
        let key = Val {
            node: str_lit(name, span),
            ty: Ty::Str,
        };
        self.index_write(obj, key, value, self.env_var_desc(upvalue), span)
    }

    fn write_var(&mut self, v: VarId, value: Val, span: Span) -> St {
        match self.storage_of(v) {
            Storage::Local(name, ty) | Storage::Module(name, ty) => {
                let value = self.coerce(value, ty);
                assign(var(name, self.ir(ty), span), value, span)
            }
            Storage::Cell(name, ty) => {
                let value = self.coerce(value, ty);
                let value = self.coerce(Val { node: value, ty }, Ty::Any);
                let cell = var(name, self.m.anys(), span);
                assign(index(cell, int_lit(0, span), Type::Any, span), value, span)
            }
        }
    }

    /// Declare a local with a value; its first declaration is a `let`,
    /// a cell's is the cell.
    fn declare_var(&mut self, v: VarId, value: Val, span: Span) -> St {
        match self.storage_of(v) {
            Storage::Local(name, ty) => {
                let value = self.coerce(value, ty);
                let_(name, self.ir(ty), value, span)
            }
            Storage::Module(name, ty) => {
                let value = self.coerce(value, ty);
                assign(var(name, self.ir(ty), span), value, span)
            }
            Storage::Cell(name, ty) => {
                let value = self.coerce(value, ty);
                let value = self.coerce(Val { node: value, ty }, Ty::Any);
                let mut pre = Vec::new();
                let cell = self.array_of(vec![value], &mut pre, span);
                let_(name, self.m.anys(), block_value(pre, cell, span), span)
            }
        }
    }

    /// Read a name.
    fn read_name(&mut self, token: &TokenReference) -> Result<Val> {
        let span = span_of(token);
        let binding = self
            .scopes()
            .binding(token)
            .cloned()
            .unwrap_or_else(|| Binding::Global(ident(token)));
        match binding {
            Binding::Local(v) | Binding::Upvalue(v) => Ok(self.read_var(v, span)),
            Binding::Global(name) => self.read_global(&name, span),
            Binding::Field(v, name, upvalue) => {
                let env = self.read_var(v, span);
                let key = Val {
                    node: str_lit(&name, span),
                    ty: Ty::Str,
                };
                Ok(self.field_read(env, key, &name, self.env_var_desc(upvalue), span))
            }
        }
    }

    fn read_global(&mut self, name: &str, span: Span) -> Result<Val> {
        if self.scopes().dynamic_globals {
            // `_ENV` is the environment itself, not an entry in it.
            if name == "_ENV" {
                let g = self.globals_table(span);
                return Ok(match g.ty {
                    Ty::Any => g,
                    _ => Val {
                        node: self.box_table(g.node),
                        ty: Ty::Any,
                    },
                });
            }
            let key = Val {
                node: str_lit(name, span),
                ty: Ty::Str,
            };
            let g = self.globals_table(span);
            let desc = self.m.env_var.and_then(|_| self.env_var_desc(true));
            return Ok(self.index_read(g, key, desc, span));
        }
        if let Some(f) = self.scopes().known_global_function(name) {
            return Ok(self.function_value(f, span));
        }
        if !self.scopes().global_writes.contains_key(name) {
            if let Some(v) = self.preset_value(name, span) {
                return Ok(v);
            }
            if name == "_G" || name == "_ENV" {
                return unsupported("`_G` other than as `_G.name`", span);
            }
            return Ok(self.nil_val(span));
        }
        let ty = self.typer().global_ty(name);
        let symbol = Module::global_symbol(name);
        self.m.declare_module_var(symbol, ty);
        Ok(Val {
            node: var(symbol, self.ir(ty), span),
            ty,
        })
    }

    fn write_global(&mut self, name: &str, value: Val, span: Span) -> Result<St> {
        if self.scopes().dynamic_globals {
            // `_ENV = v`: the chunk's environment from here on.
            if name == "_ENV" {
                let Some(env) = self.m.env_var else {
                    return unsupported("assigning `_ENV`", span);
                };
                let v = self.boxed(value);
                return Ok(assign(var(env, Type::Any, span), v, span));
            }
            let key = Val {
                node: str_lit(name, span),
                ty: Ty::Str,
            };
            let g = self.globals_table(span);
            let desc = self.m.env_var.and_then(|_| self.env_var_desc(true));
            return Ok(self.index_write(g, key, value, desc, span));
        }
        if self.scopes().known_global_function(name).is_some() {
            // The one declaration of a known function is its typed
            // entry; nothing is stored.
            return Ok(stmt(
                TypedStatement::Block(TypedBlock {
                    statements: Vec::new(),
                    span,
                }),
                span,
            ));
        }
        let ty = self.typer().global_ty(name);
        let symbol = Module::global_symbol(name);
        self.m.declare_module_var(symbol, ty);
        let value = self.coerce(value, ty);
        Ok(assign(var(symbol, self.ir(ty), span), value, span))
    }

    /// What the global `name` holds when the program starts, when it
    /// holds anything.
    fn preset_value(&mut self, name: &str, span: Span) -> Option<Val> {
        if let Some(b) = types::builtin_global(name) {
            return Some(self.builtin_value(b, span));
        }
        let node = if crate::library::stdlib::LIBS.contains(&name) {
            call(
                &crate::library::stdlib::lib_table_fn(name),
                vec![],
                Type::Any,
                span,
            )
        } else if name == "arg" {
            call("zl_arg_table", vec![], Type::Any, span)
        } else if name == "_VERSION" {
            return Some(Val {
                node: str_lit(crate::library::stdlib::LUA_VERSION, span),
                ty: Ty::Str,
            });
        } else {
            return None;
        };
        Some(Val { node, ty: Ty::Any })
    }

    /// Statements giving each global the program assigns, and may read
    /// before it does, the value it holds when the program starts.
    fn preset_globals(&mut self, span: Span) -> Vec<St> {
        let scopes = self.scopes();
        if scopes.dynamic_globals {
            return Vec::new();
        }
        let names: Vec<String> = scopes
            .global_writes
            .keys()
            .filter(|name| {
                types::preset_global(name) && !scopes.globals_initialized.contains(*name)
            })
            .cloned()
            .collect();
        let mut out = Vec::new();
        for name in names {
            let Some(value) = self.preset_value(&name, span) else {
                continue;
            };
            let ty = self.typer().global_ty(&name);
            let symbol = Module::global_symbol(&name);
            self.m.declare_module_var(symbol, ty);
            let value = self.coerce(value, ty);
            out.push(assign(var(symbol, self.ir(ty), span), value, span));
        }
        out
    }

    /// The globals table, when the program reaches its globals through
    /// one: every global is an entry, the builtins included. A chunk
    /// with an environment of its own reads it as a value, since
    /// `_ENV = v` may make it anything.
    fn globals_table(&mut self, span: Span) -> Val {
        match self.m.env_var {
            Some(env) => Val {
                node: var(env, Type::Any, span),
                ty: Ty::Any,
            },
            None => Val {
                node: var(intern(library::GLOBALS), self.ir(Ty::Table), span),
                ty: Ty::Table,
            },
        }
    }

    /// A builtin as a function value.
    fn builtin_value(&mut self, b: &Builtin, span: Span) -> Val {
        let index = crate::library::stdlib::builtin_index(b);
        Val {
            node: call(
                "zl_builtin_value",
                vec![
                    int_lit(index as i64, span),
                    code_of(&crate::library::stdlib::wrapper_name(b), span),
                    int_lit(VARIADIC_ARITY, span),
                ],
                Type::Any,
                span,
            ),
            ty: Ty::Any,
        }
    }

    // ─── functions as values ────────────────────────────────────

    /// The record of a function: its code, its arity and what it
    /// captures, made where the function is defined or named.
    fn function_value(&mut self, f: FuncId, span: Span) -> Val {
        let info = self.scopes().func(f);
        let arity = if info.is_vararg {
            VARIADIC_ARITY
        } else {
            arity_word(info.params.len(), info.params.len())
        };
        let captures = info.captures.clone();
        let mut cells = Vec::with_capacity(captures.len() + 1);
        cells.push(call(
            "zb_box_i64",
            vec![int_lit(self.m.func_key(f), span)],
            Type::Any,
            span,
        ));
        for v in captures {
            cells.push(self.capture_value(v, span));
        }
        let mut pre = Vec::new();
        let cells = self.array_of(cells, &mut pre, span);
        let record = call(
            "zb_func_new",
            vec![
                code_of(&self.m.code_name(f), span),
                int_lit(arity, span),
                cells,
            ],
            Type::Any,
            span,
        );
        // The debug library answers `getinfo(level, "f")` with the
        // last value made of each function.
        let record = if self.m.debug.is_some() {
            let name = self.temp();
            pre.push(let_(name, Type::Any, record, span));
            pre.push(expr_stmt(call(
                "zl_dbg_closure",
                vec![
                    int_lit(self.m.func_key(f), span),
                    var(name, Type::Any, span),
                ],
                prim(PrimitiveType::Unit),
                span,
            )));
            var(name, Type::Any, span)
        } else {
            record
        };
        Val {
            node: block_value(pre, record, span),
            ty: Ty::Any,
        }
    }

    /// What a closure's record holds for a captured variable: the cell
    /// itself, boxed, or the value copied.
    fn capture_value(&mut self, v: VarId, span: Span) -> Node {
        match self.storage_of(v) {
            Storage::Cell(name, _) => call(
                "zb_list_box_any",
                vec![var(name, self.m.anys(), span)],
                Type::Any,
                span,
            ),
            Storage::Local(..) | Storage::Module(..) => {
                let value = self.read_var(v, span);
                self.boxed(value)
            }
        }
    }

    /// The prologue of a function taking its record: each capture
    /// read out of `env`.
    fn env_prologue(&mut self, span: Span) -> Vec<St> {
        let mut out = Vec::new();
        let captures = self.scopes().func(self.func).captures.clone();
        for (i, v) in captures.iter().enumerate() {
            let slot = index(
                var(intern("env"), self.m.anys(), span),
                int_lit((RECORD_CELLS_AT + i) as i64, span),
                Type::Any,
                span,
            );
            let ty = self.var_ty(*v);
            let info = self.scopes().var(*v);
            if info.needs_cell() {
                let symbol = intern(&format!("{}$cell{}", info.name, v.0));
                self.storage.insert(*v, Storage::Cell(symbol, ty));
                out.push(let_(
                    symbol,
                    self.m.anys(),
                    call("zb_unbox_list_raw_any", vec![slot], self.m.anys(), span),
                    span,
                ));
            } else {
                let symbol = self.local_symbol(*v);
                self.storage.insert(*v, Storage::Local(symbol, ty));
                let value = self.coerce(
                    Val {
                        node: slot,
                        ty: Ty::Any,
                    },
                    ty,
                );
                out.push(let_(symbol, self.ir(ty), value, span));
            }
        }
        out
    }

    // ─── expressions ────────────────────────────────────────────

    /// An expression in single-value position.
    fn expr(&mut self, e: &Expression) -> Result<Val> {
        let span = span_of(e);
        match e {
            Expression::Number(t) => {
                let text = t.token().to_string();
                Ok(match crate::host::parse_numeral(text.trim()) {
                    crate::host::Numeral::Int(v) => Val {
                        node: int_lit(v, span),
                        ty: Ty::Int,
                    },
                    crate::host::Numeral::Float(v) => Val {
                        node: float_lit(v, span),
                        ty: Ty::Float,
                    },
                    crate::host::Numeral::None => {
                        return Err(Error::Syntax {
                            message: format!("malformed number near '{}'", text.trim()),
                            span: (span.start, span.end),
                        });
                    }
                })
            }
            Expression::String(t) => {
                let bytes = string_bytes(t).map_err(|message| Error::Syntax {
                    message,
                    span: (span.start, span.end),
                })?;
                Ok(Val {
                    node: string_literal(bytes, span),
                    ty: Ty::Str,
                })
            }
            Expression::Symbol(t) => match t.token().to_string().trim() {
                "true" => Ok(Val {
                    node: bool_lit(true, span),
                    ty: Ty::Bool,
                }),
                "false" => Ok(Val {
                    node: bool_lit(false, span),
                    ty: Ty::Bool,
                }),
                "nil" => Ok(self.nil_val(span)),
                "..." => {
                    let Some(varargs) = self.varargs else {
                        return unsupported("`...` outside a vararg function", span);
                    };
                    Ok(Val {
                        node: call(
                            "zl_value_at",
                            vec![var(varargs, self.m.anys(), span), int_lit(1, span)],
                            Type::Any,
                            span,
                        ),
                        ty: Ty::Any,
                    })
                }
                other => unsupported(format!("the symbol `{other}`"), span),
            },
            Expression::Parentheses { expression, .. } => {
                // Parentheses truncate to one value.
                self.expr(expression)
            }
            Expression::Function(f) => {
                let id = self.scopes().function_of(f.body());
                self.lower_function(id, f.body(), false)?;
                let record = self.function_value(id, span);
                Ok(Val {
                    node: record.node,
                    ty: Ty::Func(id),
                })
            }
            Expression::TableConstructor(t) => self.table_constructor(t, span),
            Expression::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                let multi = self.suffixed(c.prefix(), &suffixes, span)?;
                Ok(self.first_of(multi, span))
            }
            Expression::Var(v) => match v {
                Var::Name(token) => self.read_name(token),
                Var::Expression(v) => {
                    let suffixes: Vec<&Suffix> = v.suffixes().collect();
                    let multi = self.suffixed(v.prefix(), &suffixes, span)?;
                    Ok(self.first_of(multi, span))
                }
                _ => unsupported("this variable form", span),
            },
            Expression::UnaryOperator { unop, expression } => {
                let v = self.expr(expression)?;
                let desc = self.describe(expression);
                self.unary_op(unop, v, desc, span)
            }
            Expression::BinaryOperator { lhs, binop, rhs } => self.binary_op(binop, lhs, rhs, span),
            _ => unsupported("this expression form", span),
        }
    }

    /// The single value of what may be several.
    fn first_of(&mut self, multi: Multi, span: Span) -> Val {
        match multi {
            Multi::Fixed(mut vals) => {
                if vals.is_empty() {
                    self.nil_val(span)
                } else if vals.len() == 1 {
                    vals.pop().unwrap()
                } else {
                    // Several values of one call: the first carries the
                    // call, the rest are reads of its result.
                    vals.swap_remove(0)
                }
            }
            Multi::Dynamic(node) => Val {
                node: call("zl_first", vec![node], Type::Any, span),
                ty: Ty::Any,
            },
            Multi::None(node) => Val {
                node: block_value(vec![expr_stmt(node)], nil(span), span),
                ty: Ty::Nil,
            },
        }
    }

    /// An expression in last position, where it may supply several
    /// values.
    fn expr_multi(&mut self, e: &Expression) -> Result<Multi> {
        let span = span_of(e);
        match e {
            Expression::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                self.suffixed(c.prefix(), &suffixes, span)
            }
            Expression::Var(Var::Expression(v))
                if matches!(v.suffixes().last(), Some(Suffix::Call(_))) =>
            {
                let suffixes: Vec<&Suffix> = v.suffixes().collect();
                self.suffixed(v.prefix(), &suffixes, span)
            }
            Expression::Symbol(t) if t.token().to_string().trim() == "..." => {
                let Some(varargs) = self.varargs else {
                    return unsupported("`...` outside a vararg function", span);
                };
                Ok(Multi::Dynamic(call(
                    "zl_values_from",
                    vec![var(varargs, self.m.anys(), span), int_lit(1, span)],
                    Type::Any,
                    span,
                )))
            }
            _ => Ok(Multi::Fixed(vec![self.expr(e)?])),
        }
    }

    /// A list of expressions as the values it supplies, the last one
    /// expanded: statements to run first, the fixed values, and the
    /// dynamic tail if any. A last call supplying no values still runs,
    /// after the values before it.
    fn expr_list(&mut self, exprs: &[&Expression]) -> Result<(Vec<St>, Vec<Val>, Option<Node>)> {
        let mut pre = Vec::new();
        let mut vals: Vec<Val> = Vec::new();
        let mut tail = None;
        for (i, e) in exprs.iter().enumerate() {
            if i + 1 == exprs.len() {
                match self.expr_multi(e)? {
                    Multi::Fixed(more) => vals.extend(more),
                    Multi::Dynamic(node) => tail = Some(node),
                    Multi::None(node) => {
                        let held = std::mem::take(&mut vals);
                        for v in held {
                            let h = self.hold(v, &mut pre);
                            vals.push(h);
                        }
                        pre.push(expr_stmt(node));
                    }
                }
            } else {
                vals.push(self.expr(e)?);
            }
        }
        Ok((pre, vals, tail))
    }

    /// The values of an expression list as a `List<Any>`.
    fn packed_list(&mut self, exprs: &[&Expression], span: Span) -> Result<Node> {
        let (mut pre, vals, tail) = self.expr_list(exprs)?;
        let mut items = Vec::with_capacity(vals.len());
        for v in vals {
            items.push(self.boxed(v));
        }
        let list = self.array_of(items, &mut pre, span);
        Ok(match tail {
            None => block_value(pre, list, span),
            Some(tail) => {
                let name = self.temp();
                pre.push(let_(name, self.m.anys(), list, span));
                pre.push(expr_stmt(call(
                    "zl_append_values",
                    vec![var(name, self.m.anys(), span), tail],
                    prim(PrimitiveType::Unit),
                    span,
                )));
                block_value(pre, var(name, self.m.anys(), span), span)
            }
        })
    }

    /// `n` values from an expression list, Lua's way: the last
    /// expression's several values fill the rest, missing ones are nil.
    fn adjusted(
        &mut self,
        exprs: &[&Expression],
        n: usize,
        span: Span,
    ) -> Result<(Vec<St>, Vec<Val>)> {
        let listed = self.expr_list(exprs)?;
        Ok(self.adjust(listed, n, span))
    }

    /// An expression list's values, as [`Self::expr_list`] gives them,
    /// adjusted to `n`.
    fn adjust(
        &mut self,
        listed: (Vec<St>, Vec<Val>, Option<Node>),
        n: usize,
        span: Span,
    ) -> (Vec<St>, Vec<Val>) {
        let (mut pre, vals, tail) = listed;
        let mut out: Vec<Val> = Vec::with_capacity(n);
        for v in vals {
            if out.len() < n {
                let held = self.hold(v, &mut pre);
                out.push(held);
            } else {
                // Extra values run for their effects.
                pre.push(expr_stmt(v.node));
            }
        }
        if let Some(tail) = tail {
            if out.len() < n {
                let name = self.temp();
                pre.push(let_(
                    name,
                    self.m.anys(),
                    call("zl_values", vec![tail], self.m.anys(), span),
                    span,
                ));
                let mut k = 1;
                while out.len() < n {
                    out.push(Val {
                        node: call(
                            "zl_value_at",
                            vec![var(name, self.m.anys(), span), int_lit(k, span)],
                            Type::Any,
                            span,
                        ),
                        ty: Ty::Any,
                    });
                    k += 1;
                }
            } else {
                pre.push(expr_stmt(tail));
            }
        }
        while out.len() < n {
            out.push(self.nil_val(span));
        }
        (pre, out)
    }

    // ─── operators ──────────────────────────────────────────────

    fn unary_op(&mut self, op: &UnOp, v: Val, desc: Desc, span: Span) -> Result<Val> {
        let descs = Described::operand(desc);
        Ok(match op {
            UnOp::Not(_) => {
                let t = self.truthy(v);
                Val {
                    node: unary(UnaryOp::Not, t, prim(PrimitiveType::Bool), span),
                    ty: Ty::Bool,
                }
            }
            UnOp::Minus(_) => match v.ty {
                Ty::Int => Val {
                    node: binary(
                        BinaryOp::Sub,
                        int_lit(0, span),
                        v.node,
                        prim(PrimitiveType::I64),
                        span,
                    ),
                    ty: Ty::Int,
                },
                Ty::Float => Val {
                    node: unary(UnaryOp::Minus, v.node, prim(PrimitiveType::F64), span),
                    ty: Ty::Float,
                },
                Ty::Number | Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => {
                    let mut pre = Vec::new();
                    let held = self.hold(v.clone(), &mut pre);
                    let n = NumberParts::of(&held.node, span);
                    // A scalar that is not a number raises, as the
                    // dynamic value does.
                    if v.ty != Ty::Number {
                        let boxed = self.coerce(held.clone(), Ty::Any);
                        pre.push(if_(
                            n.is_numeric(span),
                            Vec::new(),
                            Some(vec![
                                expr_stmt(call("zl_unm", vec![boxed], Type::Any, span)),
                                self.pending_check_described(span, &descs),
                            ]),
                            span,
                        ));
                    }
                    let int = binary(
                        BinaryOp::Sub,
                        int_lit(0, span),
                        n.int,
                        prim(PrimitiveType::I64),
                        span,
                    );
                    let float = unary(UnaryOp::Minus, n.float, prim(PrimitiveType::F64), span);
                    Val {
                        node: block_value(pre, number_value(n.tag, int, float, span), span),
                        ty: Ty::Number,
                    }
                }
                Ty::Shape(_)
                    if let Some(sides) = self.typer().metamethod_sides("__unm", v.ty, v.ty)
                        && !sides.is_empty() =>
                {
                    // The one operand is both: the handler takes it twice.
                    let sides: Vec<(bool, ShapeId, FuncId)> =
                        sides.into_iter().filter(|(left, _, _)| *left).collect();
                    let mut pre = Vec::new();
                    let held = self.hold(v, &mut pre);
                    let b = self.boxed(held.clone());
                    let generic = self.guard_described(
                        Val {
                            node: call("zl_unm", vec![b], Type::Any, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    );
                    let ty = self
                        .typer()
                        .metamethod_result("__unm", held.ty, held.ty)
                        .unwrap_or(Ty::Any)
                        .settled();
                    let value = self.metamethod_dispatch(
                        "__unm",
                        sides,
                        vec![held.clone(), held],
                        generic,
                        ty,
                        span,
                    );
                    Val {
                        node: block_value(pre, value, span),
                        ty,
                    }
                }
                _ => {
                    let b = self.boxed(v);
                    self.guard_described(
                        Val {
                            node: call("zl_unm", vec![b], Type::Any, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    )
                }
            },
            UnOp::Hash(_) => match v.ty {
                Ty::Str => Val {
                    node: call("zb_str_len", vec![v.node], prim(PrimitiveType::I64), span),
                    ty: Ty::Int,
                },
                Ty::Table if !self.scopes().len_meta => Val {
                    node: call("zl_table_len", vec![v.node], prim(PrimitiveType::I64), span),
                    ty: Ty::Int,
                },
                // A shaped table is a table unless it is nil, when the
                // dynamic path raises as Lua does. With no `__len` in
                // any metatable it may have, its length is its array
                // part's.
                Ty::Shape(k) if !self.scopes().len_meta => {
                    let i64_t = prim(PrimitiveType::I64);
                    let mut pre = Vec::new();
                    let t = self.hold(v, &mut pre);
                    let is_null = binary(
                        BinaryOp::Eq,
                        t.node.clone(),
                        null(self.ir(Ty::Table), span),
                        prim(PrimitiveType::Bool),
                        span,
                    );
                    let on_nil = self.guard_described(
                        Val {
                            node: call("zl_len_any", vec![nil(span)], Type::Any, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    );
                    let on_nil = self.coerce(on_nil, Ty::Int);
                    let len = if self.m.inferred.plain_len(k) {
                        list_len(
                            call(
                                "zb_unbox_list_raw_any",
                                vec![field(t.node, "arr", Type::Any, span)],
                                self.m.anys(),
                                span,
                            ),
                            span,
                        )
                    } else {
                        call("zl_table_len", vec![t.node], i64_t.clone(), span)
                    };
                    Val {
                        node: block_value(pre, if_value(is_null, on_nil, len, i64_t, span), span),
                        ty: Ty::Int,
                    }
                }
                _ => {
                    let b = self.boxed(v);
                    self.guard_described(
                        Val {
                            node: call("zl_len_any", vec![b], Type::Any, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    )
                }
            },
            UnOp::Tilde(_) => match v.ty {
                Ty::Int => Val {
                    node: binary(
                        BinaryOp::BitXor,
                        v.node,
                        int_lit(-1, span),
                        prim(PrimitiveType::I64),
                        span,
                    ),
                    ty: Ty::Int,
                },
                _ => {
                    let b = self.boxed(v);
                    self.guard_described(
                        Val {
                            node: call("zl_bnot", vec![b], Type::Any, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    )
                }
            },
            _ => return unsupported("this unary operator", span),
        })
    }

    /// An operator's handler, called directly: for each side that is a
    /// shaped table and each of its classes, in the order the runtime
    /// tries them, when the operand's metatable is that class and the
    /// handler is present in it; else `generic`, the runtime's own
    /// dispatch. The operands are held; every path yields `ty`.
    fn metamethod_dispatch(
        &mut self,
        event: &str,
        sides: Vec<(bool, ShapeId, FuncId)>,
        operands: Vec<Val>,
        generic: Val,
        ty: Ty,
        span: Span,
    ) -> Node {
        let table_t = self.ir(Ty::Table);
        let bool_t = prim(PrimitiveType::Bool);
        let i64_t = prim(PrimitiveType::I64);
        let metamethod = Some(format!("metamethod '{}'", event.trim_start_matches("__")));
        let mut pre = Vec::new();
        let mut value = self.coerce(generic, ty);
        for (left, class, f) in sides.into_iter().rev() {
            let Some((layout, slot)) = self.slot_of(class, event) else {
                continue;
            };
            let operand = if left { &operands[0] } else { &operands[1] };
            let not_null = |x: &Node| {
                binary(
                    BinaryOp::Ne,
                    x.clone(),
                    null(table_t.clone(), span),
                    bool_t.clone(),
                    span,
                )
            };
            let meta = self.hold(
                Val {
                    node: if_value(
                        not_null(&operand.node),
                        field(operand.node.clone(), "meta", table_t.clone(), span),
                        null(table_t.clone(), span),
                        table_t.clone(),
                        span,
                    ),
                    ty: Ty::Table,
                },
                &mut pre,
            );
            let is_class = binary(
                BinaryOp::And,
                binary(
                    BinaryOp::Eq,
                    field(meta.node.clone(), "shape", i64_t.clone(), span),
                    int_lit(layout.gid, span),
                    bool_t.clone(),
                    span,
                ),
                self.slot_present(&meta.node, &slot, span),
                bool_t.clone(),
                span,
            );
            let ok = self.hold(
                Val {
                    node: if_value(
                        not_null(&meta.node),
                        is_class,
                        bool_lit(false, span),
                        bool_t.clone(),
                        span,
                    ),
                    ty: Ty::Bool,
                },
                &mut pre,
            );
            let record = self.slot_read(&meta.node, layout, &slot, span);
            let multi = self.direct_call_vals(
                f,
                Some(record),
                Vec::new(),
                operands.clone(),
                None,
                &metamethod,
                span,
            );
            let first = match multi {
                Multi::Fixed(vals) => vals
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| self.nil_val(span)),
                Multi::Dynamic(node) => Val {
                    node: call("zl_first", vec![node], Type::Any, span),
                    ty: Ty::Any,
                },
                Multi::None(node) => Val {
                    node: block_value(vec![expr_stmt(node)], nil(span), span),
                    ty: Ty::Nil,
                },
            };
            // Held: the call's check leaves its block, and a branch's
            // value is read after it.
            let mut arm_pre = Vec::new();
            let arm = self.coerce(first, ty);
            let arm = self.hold(Val { node: arm, ty }, &mut arm_pre);
            value = if_value(
                ok.node,
                block_value(arm_pre, arm.node, span),
                value,
                self.ir(ty),
                span,
            );
        }
        block_value(pre, value, span)
    }

    fn binary_op(
        &mut self,
        op: &BinOp,
        lhs: &Expression,
        rhs: &Expression,
        span: Span,
    ) -> Result<Val> {
        // `and` and `or` evaluate their right side conditionally.
        match op {
            BinOp::And(_) | BinOp::Or(_) => {
                return self.logical(matches!(op, BinOp::And(_)), lhs, rhs, span);
            }
            _ => {}
        }
        let a = self.expr(lhs)?;
        let b = self.expr(rhs)?;
        let descs = Described::operands(self.describe(lhs), self.describe(rhs));
        let v = self.binary_vals(op, a, b, &descs, span)?;
        Ok(self.guarded_described(v, &descs))
    }

    fn binary_vals(
        &mut self,
        op: &BinOp,
        a: Val,
        b: Val,
        descs: &Described,
        span: Span,
    ) -> Result<Val> {
        // A shaped table operand whose handler the types know.
        if let Some(event) = types::arith_event(op)
            && let Some(sides) = self.typer().metamethod_sides(event, a.ty, b.ty)
            && !sides.is_empty()
        {
            let code = match op {
                BinOp::Plus(_) => OP_ADD,
                BinOp::Minus(_) => OP_SUB,
                BinOp::Star(_) => OP_MUL,
                BinOp::Slash(_) => OP_DIV,
                BinOp::Percent(_) => OP_MOD,
                BinOp::Caret(_) => OP_POW,
                _ => OP_IDIV,
            };
            let mut pre = Vec::new();
            let a = self.hold(a, &mut pre);
            let b = self.hold(b, &mut pre);
            let x = self.boxed(a.clone());
            let y = self.boxed(b.clone());
            let generic = self.guard_described(
                Val {
                    node: call("zl_arith", vec![int_lit(code, span), x, y], Type::Any, span),
                    ty: Ty::Any,
                },
                descs,
            );
            let ty = self
                .typer()
                .metamethod_result(event, a.ty, b.ty)
                .unwrap_or(Ty::Any)
                .settled();
            let value = self.metamethod_dispatch(event, sides, vec![a, b], generic, ty, span);
            return Ok(Val {
                node: block_value(pre, value, span),
                ty,
            });
        }
        let ints = a.ty == Ty::Int && b.ty == Ty::Int;
        let numbers = a.ty.is_number() && b.ty.is_number();
        // A literal divisor that is not 0 or -1 makes `//` and `%`
        // plain arithmetic: nothing to raise, nothing to check after.
        let literal_divisor = match &b.node.node {
            TypedExpression::Literal(TypedLiteral::Integer(n)) if b.ty == Ty::Int => {
                Some(*n as i64)
            }
            _ => None,
        };
        let plain_divisor = literal_divisor.is_some_and(|n| n != 0 && n != -1);
        // A positive power of two divides a float exactly, so its
        // modulo is three operations instead of fmod; an integer's
        // floor division and modulo by it are an arithmetic shift and a
        // mask, for either sign.
        let pow2_divisor = literal_divisor.filter(|n| *n > 0 && n & (n - 1) == 0);
        let divisor = Divisor {
            plain: plain_divisor,
            pow2: pow2_divisor,
        };
        // A number whose kind is not known decides it at run time.
        if numbers
            && (a.ty == Ty::Number || b.ty == Ty::Number)
            && let Some(v) = self.number_binary(op, a.clone(), b.clone(), divisor, span)
        {
            return Ok(v);
        }
        // A scalar that may not be a number: arithmetic and ordering
        // raise for nil and booleans as the dynamic value does, then
        // take the number path.
        let arithmetic_or_order = matches!(
            op,
            BinOp::Plus(_)
                | BinOp::Minus(_)
                | BinOp::Star(_)
                | BinOp::Slash(_)
                | BinOp::Caret(_)
                | BinOp::DoubleSlash(_)
                | BinOp::Percent(_)
                | BinOp::LessThan(_)
                | BinOp::LessThanEqual(_)
                | BinOp::GreaterThan(_)
                | BinOp::GreaterThanEqual(_)
        );
        if arithmetic_or_order && or_nil_operands(a.ty, b.ty) {
            return self.or_nil_binary(op, a, b, descs, span);
        }
        if arithmetic_or_order
            && (a.ty.is_tagged() || b.ty.is_tagged())
            && a.ty.is_scalar()
            && b.ty.is_scalar()
        {
            return Ok(self.scalar_binary(op, a, b, divisor, descs, span));
        }
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let bool_t = prim(PrimitiveType::Bool);
        let as_float = |this: &mut Self, v: Val| this.coerce(v, Ty::Float);
        let int_result = |n: Node| Val {
            node: n,
            ty: Ty::Int,
        };
        let float_result = |n: Node| Val {
            node: n,
            ty: Ty::Float,
        };
        let bool_result = |n: Node| Val {
            node: n,
            ty: Ty::Bool,
        };
        let arith = |this: &mut Self, code: i64, a: Val, b: Val| {
            let a = this.boxed(a);
            let b = this.boxed(b);
            Val {
                node: call("zl_arith", vec![int_lit(code, span), a, b], Type::Any, span),
                ty: Ty::Any,
            }
        };
        Ok(match op {
            BinOp::Plus(_) | BinOp::Minus(_) | BinOp::Star(_) => {
                let bop = match op {
                    BinOp::Plus(_) => BinaryOp::Add,
                    BinOp::Minus(_) => BinaryOp::Sub,
                    _ => BinaryOp::Mul,
                };
                let code = match op {
                    BinOp::Plus(_) => OP_ADD,
                    BinOp::Minus(_) => OP_SUB,
                    _ => OP_MUL,
                };
                if ints {
                    int_result(binary(bop, a.node, b.node, i64_t, span))
                } else if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(binary(bop, x, y, f64_t, span))
                } else {
                    arith(self, code, a, b)
                }
            }
            BinOp::Slash(_) => {
                if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(binary(BinaryOp::Div, x, y, f64_t, span))
                } else {
                    arith(self, OP_DIV, a, b)
                }
            }
            BinOp::Caret(_) => {
                if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(call("zl_pow", vec![x, y], f64_t, span))
                } else {
                    arith(self, OP_POW, a, b)
                }
            }
            BinOp::DoubleSlash(_) => {
                if let (true, Some(n)) = (ints, pow2_divisor) {
                    int_result(pow2_floordiv(a.node, n, span))
                } else if ints && plain_divisor {
                    int_result(call("zb_floordiv_i64", vec![a.node, b.node], i64_t, span))
                } else if ints {
                    int_result(call("zl_idiv_i64", vec![a.node, b.node], i64_t, span))
                } else if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(call("zl_idiv_f64", vec![x, y], f64_t, span))
                } else {
                    arith(self, OP_IDIV, a, b)
                }
            }
            BinOp::Percent(_) => {
                if let (true, Some(n)) = (ints, pow2_divisor) {
                    int_result(pow2_mod(a.node, n, span))
                } else if ints && plain_divisor {
                    int_result(call("zb_mod_i64", vec![a.node, b.node], i64_t, span))
                } else if ints {
                    int_result(call("zl_mod_i64", vec![a.node, b.node], i64_t, span))
                } else if let (true, Some(n)) = (numbers, pow2_divisor) {
                    let x = as_float(self, a);
                    float_result(self.float_mod_pow2(x, n, span))
                } else if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(call("zl_mod_f64", vec![x, y], f64_t, span))
                } else {
                    arith(self, OP_MOD, a, b)
                }
            }
            BinOp::Ampersand(_) | BinOp::Pipe(_) | BinOp::Tilde(_) => {
                let (bop, code) = match op {
                    BinOp::Ampersand(_) => (BinaryOp::BitAnd, OP_BAND),
                    BinOp::Pipe(_) => (BinaryOp::BitOr, OP_BOR),
                    _ => (BinaryOp::BitXor, OP_BXOR),
                };
                if ints {
                    int_result(binary(bop, a.node, b.node, i64_t, span))
                } else {
                    arith(self, code, a, b)
                }
            }
            BinOp::DoubleLessThan(_) | BinOp::DoubleGreaterThan(_) => {
                let (f, code) = match op {
                    BinOp::DoubleLessThan(_) => ("zl_shl_i64", OP_SHL),
                    _ => ("zl_shr_i64", OP_SHR),
                };
                if ints {
                    int_result(call(f, vec![a.node, b.node], i64_t, span))
                } else {
                    arith(self, code, a, b)
                }
            }
            BinOp::TwoDots(_) => {
                let text = |t: Ty| matches!(t, Ty::Str | Ty::Int | Ty::Float);
                if text(a.ty) && text(b.ty) {
                    let x = self.text_of(a);
                    let y = self.text_of(b);
                    Val {
                        node: binary(BinaryOp::Add, x, y, prim(PrimitiveType::String), span),
                        ty: Ty::Str,
                    }
                } else {
                    let x = self.boxed(a);
                    let y = self.boxed(b);
                    Val {
                        node: call("zl_concat", vec![x, y], Type::Any, span),
                        ty: Ty::Any,
                    }
                }
            }
            BinOp::TwoEqual(_) | BinOp::TildeEqual(_) => {
                let negate = matches!(op, BinOp::TildeEqual(_));
                let eq = self.equal(a, b, span);
                bool_result(if negate {
                    unary(UnaryOp::Not, eq, bool_t, span)
                } else {
                    eq
                })
            }
            BinOp::LessThan(_)
            | BinOp::LessThanEqual(_)
            | BinOp::GreaterThan(_)
            | BinOp::GreaterThanEqual(_) => {
                // `a > b` is `b < a`.
                let (a, b, or_equal) = match op {
                    BinOp::LessThan(_) => (a, b, false),
                    BinOp::LessThanEqual(_) => (a, b, true),
                    BinOp::GreaterThan(_) => (b, a, false),
                    _ => (b, a, true),
                };
                bool_result(self.less(a, b, or_equal, span))
            }
            _ => return unsupported("this binary operator", span),
        })
    }

    /// Arithmetic or ordering on scalars of which at least one may be
    /// nil or a boolean: the dynamic operation runs when either is,
    /// which raises; then both are numbers and the number path runs.
    fn scalar_binary(
        &mut self,
        op: &BinOp,
        a: Val,
        b: Val,
        divisor: Divisor,
        descs: &Described,
        span: Span,
    ) -> Val {
        let mut pre = Vec::new();
        let a = Val {
            node: self.coerce(a, Ty::Scalar),
            ty: Ty::Scalar,
        };
        let b = Val {
            node: self.coerce(b, Ty::Scalar),
            ty: Ty::Scalar,
        };
        let ha = self.hold(a, &mut pre);
        let hb = self.hold(b, &mut pre);
        let x = NumberParts::of(&ha.node, span);
        let y = NumberParts::of(&hb.node, span);
        let numeric = x.both_numeric(&y, span);
        let raise = self.scalar_raise(op, &ha, &hb, span);
        pre.push(if_(
            numeric,
            Vec::new(),
            Some(vec![
                expr_stmt(raise),
                self.pending_check_described(span, descs),
            ]),
            span,
        ));
        let na = Val {
            node: ha.node,
            ty: Ty::Number,
        };
        let nb = Val {
            node: hb.node,
            ty: Ty::Number,
        };
        let v = self
            .number_binary(op, na, nb, divisor, span)
            .expect("the number path takes every arithmetic and ordering operator");
        Val {
            node: block_value(pre, v.node, span),
            ty: v.ty,
        }
    }

    /// The dynamic operation on held scalars `a` and `b` that are not
    /// both numbers: it raises, as on the dynamic values.
    fn scalar_raise(&mut self, op: &BinOp, a: &Val, b: &Val, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let boxed_a = self.coerce(a.clone(), Ty::Any);
        let boxed_b = self.coerce(b.clone(), Ty::Any);
        match op {
            BinOp::LessThan(_) => call("zl_lt", vec![boxed_a, boxed_b], bool_t, span),
            BinOp::LessThanEqual(_) => call("zl_le", vec![boxed_a, boxed_b], bool_t, span),
            BinOp::GreaterThan(_) => call("zl_lt", vec![boxed_b, boxed_a], bool_t, span),
            BinOp::GreaterThanEqual(_) => call("zl_le", vec![boxed_b, boxed_a], bool_t, span),
            _ => {
                let code = match op {
                    BinOp::Plus(_) => OP_ADD,
                    BinOp::Minus(_) => OP_SUB,
                    BinOp::Star(_) => OP_MUL,
                    BinOp::Slash(_) => OP_DIV,
                    BinOp::Caret(_) => OP_POW,
                    BinOp::DoubleSlash(_) => OP_IDIV,
                    _ => OP_MOD,
                };
                call(
                    "zl_arith",
                    vec![int_lit(code, span), boxed_a, boxed_b],
                    Type::Any,
                    span,
                )
            }
        }
    }

    /// Arithmetic or ordering where each operand is a number of known
    /// kind or such a number or nil: a nil operand takes the dynamic
    /// operation, which raises; else the numbers' own operation runs.
    fn or_nil_binary(
        &mut self,
        op: &BinOp,
        a: Val,
        b: Val,
        descs: &Described,
        span: Span,
    ) -> Result<Val> {
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let mut pre = Vec::new();
        let ha = self.hold(a, &mut pre);
        let hb = self.hold(b, &mut pre);
        // The tags of the sides that may be nil, and the tag each has
        // when it is its number.
        let mut tags = Vec::new();
        let mut number = |h: &Val| -> Val {
            let n = NumberParts::of(&h.node, span);
            match h.ty {
                Ty::IntOrNil => {
                    tags.push((n.tag, TAG_INT));
                    Val {
                        node: n.int,
                        ty: Ty::Int,
                    }
                }
                Ty::FloatOrNil => {
                    tags.push((n.tag, TAG_FLOAT));
                    Val {
                        node: n.float,
                        ty: Ty::Float,
                    }
                }
                _ => h.clone(),
            }
        };
        let x = number(&ha);
        let y = number(&hb);
        // Both numbers without a short-circuit: tags are at most
        // `TAG_FLOAT`, so a product of two says which pair they are.
        let (tag, expected) = tags
            .into_iter()
            .reduce(|(p, e), (q, f)| (binary(BinaryOp::Mul, p, q, i64_t.clone(), span), e * f))
            .expect("an operand that may be nil");
        let numeric = binary(BinaryOp::Eq, tag, int_lit(expected, span), bool_t, span);
        let raise = self.scalar_raise(op, &ha, &hb, span);
        pre.push(if_(
            numeric,
            Vec::new(),
            Some(vec![
                expr_stmt(raise),
                self.pending_check_described(span, descs),
            ]),
            span,
        ));
        let v = self.binary_vals(op, x, y, descs, span)?;
        Ok(Val {
            node: block_value(pre, v.node, span),
            ty: v.ty,
        })
    }

    /// `a == b` where one side is a number of known kind or nil and the
    /// other that number, nil, or the same: the tags and the number's
    /// part decide, since a scalar's unused parts are zero.
    fn or_nil_eq(&mut self, a: Val, b: Val, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let mut pre = Vec::new();
        let a = self.hold(a, &mut pre);
        let b = self.hold(b, &mut pre);
        let (a, b) = if a.ty.is_tagged() { (a, b) } else { (b, a) };
        let (tag, part_ty) = match a.ty {
            Ty::IntOrNil => (TAG_INT, Ty::Int),
            _ => (TAG_FLOAT, Ty::Float),
        };
        let part = |n: NumberParts| if part_ty == Ty::Int { n.int } else { n.float };
        let x = NumberParts::of(&a.node, span);
        let eq = match b.ty {
            Ty::Nil => x.has_tag(TAG_NIL, span),
            t if t == part_ty => {
                let is = x.has_tag(tag, span);
                let same = binary(BinaryOp::Eq, part(x), b.node, bool_t.clone(), span);
                binary(BinaryOp::And, is, same, bool_t, span)
            }
            _ => {
                let y = NumberParts::of(&b.node, span);
                let tags = binary(
                    BinaryOp::Eq,
                    x.tag.clone(),
                    y.tag.clone(),
                    bool_t.clone(),
                    span,
                );
                let same = binary(BinaryOp::Eq, part(x), part(y), bool_t.clone(), span);
                binary(BinaryOp::And, tags, same, bool_t, span)
            }
        };
        block_value(pre, eq, span)
    }

    /// Equality of scalars: numbers compare as numbers, anything else
    /// by tag and payload.
    fn scalar_eq(&mut self, a: Val, b: Val, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let a = Val {
            node: self.coerce(a, Ty::Scalar),
            ty: Ty::Scalar,
        };
        let b = Val {
            node: self.coerce(b, Ty::Scalar),
            ty: Ty::Scalar,
        };
        let (mut pre, x) = self.number_parts(a);
        let (mut pre_b, y) = self.number_parts(b);
        pre.append(&mut pre_b);
        let numeric = x.both_numeric(&y, span);
        let as_numbers = self.number_compare_parts(&x, &y, Compare::Eq, span);
        let same = binary(
            BinaryOp::And,
            binary(
                BinaryOp::Eq,
                x.tag.clone(),
                y.tag.clone(),
                bool_t.clone(),
                span,
            ),
            binary(
                BinaryOp::Eq,
                x.int.clone(),
                y.int.clone(),
                bool_t.clone(),
                span,
            ),
            bool_t.clone(),
            span,
        );
        block_value(pre, if_value(numeric, as_numbers, same, bool_t, span), span)
    }

    /// Arithmetic where at least one operand's kind is decided at run
    /// time: both integers takes the integer path, anything else the
    /// float path, and the result says which it took. Bitwise
    /// operators and concatenation are not handled here.
    fn number_binary(
        &mut self,
        op: &BinOp,
        a: Val,
        b: Val,
        divisor: Divisor,
        span: Span,
    ) -> Option<Val> {
        let plain_divisor = divisor.plain;
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let bool_t = prim(PrimitiveType::Bool);
        let both_int = |x: &NumberParts, y: &NumberParts| x.both_int(y, span);
        let a = Val {
            node: self.coerce(a, Ty::Number),
            ty: Ty::Number,
        };
        let b = Val {
            node: self.coerce(b, Ty::Number),
            ty: Ty::Number,
        };
        let (mut pre, x) = self.number_parts(a);
        let (mut pre_b, y) = self.number_parts(b);
        pre.append(&mut pre_b);
        let number = |int: Node, float: Node, is_int: Node| Val {
            node: block_value(
                pre.clone(),
                number_value(tag_of_is_int(is_int, span), int, float, span),
                span,
            ),
            ty: Ty::Number,
        };
        Some(match op {
            BinOp::Plus(_) | BinOp::Minus(_) | BinOp::Star(_) => {
                let bop = match op {
                    BinOp::Plus(_) => BinaryOp::Add,
                    BinOp::Minus(_) => BinaryOp::Sub,
                    _ => BinaryOp::Mul,
                };
                // Both paths are pure, so both run and the tag picks.
                let int = binary(bop, x.int.clone(), y.int.clone(), i64_t, span);
                let float = binary(bop, x.as_float(span), y.as_float(span), f64_t, span);
                number(int, float, both_int(&x, &y))
            }
            BinOp::Slash(_) | BinOp::Caret(_) => {
                let (l, r) = (x.as_float(span), y.as_float(span));
                let float = if matches!(op, BinOp::Slash(_)) {
                    binary(BinaryOp::Div, l, r, f64_t, span)
                } else {
                    call("zl_pow", vec![l, r], f64_t, span)
                };
                Val {
                    node: block_value(pre, float, span),
                    ty: Ty::Float,
                }
            }
            BinOp::DoubleSlash(_) | BinOp::Percent(_) => {
                // The integer path raises on a zero divisor, so only
                // the path taken runs.
                let (fi, ff) = match (op, plain_divisor) {
                    (BinOp::DoubleSlash(_), true) => ("zb_floordiv_i64", "zl_idiv_f64"),
                    (BinOp::DoubleSlash(_), false) => ("zl_idiv_i64", "zl_idiv_f64"),
                    (_, true) => ("zb_mod_i64", "zl_mod_f64"),
                    (_, false) => ("zl_mod_i64", "zl_mod_f64"),
                };
                let is_int = both_int(&x, &y);
                let int_part = match (op, divisor.pow2) {
                    (BinOp::Percent(_), Some(n)) => pow2_mod(x.int.clone(), n, span),
                    (_, Some(n)) => pow2_floordiv(x.int.clone(), n, span),
                    _ => call(fi, vec![x.int.clone(), y.int.clone()], i64_t.clone(), span),
                };
                let int = if_value(is_int.clone(), int_part, int_lit(0, span), i64_t, span);
                let float_part = match (op, divisor.pow2) {
                    (BinOp::Percent(_), Some(n)) => self.float_mod_pow2(x.as_float(span), n, span),
                    _ => call(
                        ff,
                        vec![x.as_float(span), y.as_float(span)],
                        f64_t.clone(),
                        span,
                    ),
                };
                let float = if_value(
                    is_int.clone(),
                    float_lit(0.0, span),
                    float_part,
                    f64_t,
                    span,
                );
                // The call is below the value, where the check after a
                // call would not find it.
                let v = number(int, float, is_int);
                if plain_divisor { v } else { self.guard(v) }
            }
            BinOp::TwoEqual(_) | BinOp::TildeEqual(_) => {
                let eq = self.number_compare_parts(&x, &y, Compare::Eq, span);
                let v = if matches!(op, BinOp::TildeEqual(_)) {
                    unary(UnaryOp::Not, eq, bool_t, span)
                } else {
                    eq
                };
                Val {
                    node: block_value(pre, v, span),
                    ty: Ty::Bool,
                }
            }
            BinOp::LessThan(_)
            | BinOp::LessThanEqual(_)
            | BinOp::GreaterThan(_)
            | BinOp::GreaterThanEqual(_) => {
                let (l, r, how) = match op {
                    BinOp::LessThan(_) => (&x, &y, Compare::Lt),
                    BinOp::LessThanEqual(_) => (&x, &y, Compare::Le),
                    BinOp::GreaterThan(_) => (&y, &x, Compare::Lt),
                    _ => (&y, &x, Compare::Le),
                };
                let v = self.number_compare_parts(l, r, how, span);
                Val {
                    node: block_value(pre, v, span),
                    ty: Ty::Bool,
                }
            }
            _ => return None,
        })
    }

    /// `a` against `b`, at least one of them a number of run-time kind.
    fn number_compare(&mut self, a: Val, b: Val, how: Compare, span: Span) -> Node {
        let a = Val {
            node: self.coerce(a, Ty::Number),
            ty: Ty::Number,
        };
        let b = Val {
            node: self.coerce(b, Ty::Number),
            ty: Ty::Number,
        };
        let (mut pre, x) = self.number_parts(a);
        let (mut pre_b, y) = self.number_parts(b);
        pre.append(&mut pre_b);
        let v = self.number_compare_parts(&x, &y, how, span);
        block_value(pre, v, span)
    }

    /// Two integers compare as integers, two floats as floats, and an
    /// integer against a float exactly, as Lua does.
    fn number_compare_parts(
        &mut self,
        x: &NumberParts,
        y: &NumberParts,
        how: Compare,
        span: Span,
    ) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let (op, int_float, float_int) = match how {
            Compare::Eq => (BinaryOp::Eq, "zl_eq_if", "zl_eq_if"),
            Compare::Lt => (BinaryOp::Lt, "zl_lt_if", "zl_lt_fi"),
            Compare::Le => (BinaryOp::Le, "zl_le_if", "zl_le_fi"),
        };
        let ints = binary(op, x.int.clone(), y.int.clone(), bool_t.clone(), span);
        let floats = binary(op, x.float.clone(), y.float.clone(), bool_t.clone(), span);
        // `zl_eq_if` takes the integer first whichever side it is on.
        let x_int = call(
            int_float,
            vec![x.int.clone(), y.float.clone()],
            bool_t.clone(),
            span,
        );
        let y_int = if how == Compare::Eq {
            call(
                float_int,
                vec![y.int.clone(), x.float.clone()],
                bool_t.clone(),
                span,
            )
        } else {
            call(
                float_int,
                vec![x.float.clone(), y.int.clone()],
                bool_t.clone(),
                span,
            )
        };
        // Two numbers of one kind compare by a select; an integer
        // against a float takes the exact helper.
        let same = if_value(x.is_int(span), ints, floats, bool_t.clone(), span);
        let mixed = if_value(x.is_int(span), x_int, y_int, bool_t.clone(), span);
        let same_kind = binary(
            BinaryOp::Eq,
            x.tag.clone(),
            y.tag.clone(),
            bool_t.clone(),
            span,
        );
        if_value(same_kind, same, mixed, bool_t, span)
    }

    /// The text a string or number contributes to `..`.
    fn text_of(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Str => v.node,
            Ty::Int => call(
                "zb_str_of_int",
                vec![v.node],
                prim(PrimitiveType::String),
                span,
            ),
            Ty::Float => call(
                "zl_float_str",
                vec![v.node],
                prim(PrimitiveType::String),
                span,
            ),
            Ty::Number => {
                let str_t = prim(PrimitiveType::String);
                let (pre, n) = self.number_parts(v);
                let text = if_value(
                    n.is_int(span),
                    call("zb_str_of_int", vec![n.int], str_t.clone(), span),
                    call("zl_float_str", vec![n.float], str_t.clone(), span),
                    str_t,
                    span,
                );
                block_value(pre, text, span)
            }
            _ => {
                let b = self.boxed(v);
                self.guard(Val {
                    node: call("zl_concat_text", vec![b], prim(PrimitiveType::String), span),
                    ty: Ty::Str,
                })
                .node
            }
        }
    }

    fn equal(&mut self, a: Val, b: Val, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        if or_nil_comparable(a.ty, b.ty) {
            return self.or_nil_eq(a, b, span);
        }
        match (a.ty, b.ty) {
            (Ty::Int, Ty::Int) | (Ty::Float, Ty::Float) | (Ty::Bool, Ty::Bool) => {
                binary(BinaryOp::Eq, a.node, b.node, bool_t, span)
            }
            (Ty::Int, Ty::Float) => call("zl_eq_if", vec![a.node, b.node], bool_t, span),
            (Ty::Float, Ty::Int) => call("zl_eq_if", vec![b.node, a.node], bool_t, span),
            (x, y) if x.is_number() && y.is_number() => {
                self.number_compare(a, b, Compare::Eq, span)
            }
            (x, y) if (x.is_tagged() || y.is_tagged()) && x.is_scalar() && y.is_scalar() => {
                self.scalar_eq(a, b, span)
            }
            // Tables are equal when they are the same table, `__eq`
            // aside; a shaped table may be nil.
            (Ty::Shape(_), Ty::Nil) => binary(
                BinaryOp::Eq,
                a.node,
                block_value(
                    vec![expr_stmt(b.node)],
                    null(self.ir(Ty::Table), span),
                    span,
                ),
                bool_t,
                span,
            ),
            (Ty::Nil, Ty::Shape(_)) => binary(
                BinaryOp::Eq,
                block_value(
                    vec![expr_stmt(a.node)],
                    null(self.ir(Ty::Table), span),
                    span,
                ),
                b.node,
                bool_t,
                span,
            ),
            (Ty::Func(_), Ty::Nil) | (Ty::Nil, Ty::Func(_)) => {
                binary(BinaryOp::Eq, a.node, b.node, bool_t, span)
            }
            (Ty::Str, Ty::Str) => call("zb_str_eq", vec![a.node, b.node], bool_t, span),
            (Ty::Nil, Ty::Nil) => block_value(
                vec![expr_stmt(a.node), expr_stmt(b.node)],
                bool_lit(true, span),
                span,
            ),
            // Two different known types are never equal; the operands
            // still run.
            (x, y)
                if x != Ty::Any
                    && y != Ty::Any
                    && x != y
                    && !matches!(x, Ty::Table | Ty::Shape(_) | Ty::Func(_))
                    && !matches!(y, Ty::Table | Ty::Shape(_) | Ty::Func(_)) =>
            {
                block_value(
                    vec![expr_stmt(a.node), expr_stmt(b.node)],
                    bool_lit(false, span),
                    span,
                )
            }
            _ => {
                let x = self.boxed(a);
                let y = self.boxed(b);
                call("zl_eq", vec![x, y], bool_t, span)
            }
        }
    }

    fn less(&mut self, a: Val, b: Val, or_equal: bool, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let op = if or_equal { BinaryOp::Le } else { BinaryOp::Lt };
        if a.ty.is_number() && b.ty.is_number() {
            if a.ty == Ty::Number || b.ty == Ty::Number {
                let how = if or_equal { Compare::Le } else { Compare::Lt };
                return self.number_compare(a, b, how, span);
            }
            if a.ty == b.ty {
                return binary(op, a.node, b.node, bool_t, span);
            }
            // An integer against a float compares exactly.
            let f = match (a.ty, or_equal) {
                (Ty::Int, false) => "zl_lt_if",
                (Ty::Int, true) => "zl_le_if",
                (_, false) => "zl_lt_fi",
                _ => "zl_le_fi",
            };
            return call(f, vec![a.node, b.node], bool_t, span);
        }
        if a.ty == Ty::Str && b.ty == Ty::Str {
            let cmp = call(
                "zb_str_cmp",
                vec![a.node, b.node],
                prim(PrimitiveType::I32),
                span,
            );
            return binary(op, cmp, int32_lit(0, span), bool_t, span);
        }
        let x = self.boxed(a);
        let y = self.boxed(b);
        call(
            if or_equal { "zl_le" } else { "zl_lt" },
            vec![x, y],
            bool_t,
            span,
        )
    }

    /// `a and b` is `a` when that is false, else `b`; `a or b` the
    /// other way round. The right side runs only when needed.
    fn logical(
        &mut self,
        is_and: bool,
        lhs: &Expression,
        rhs: &Expression,
        span: Span,
    ) -> Result<Val> {
        let a = self.expr(lhs)?;
        let b = self.expr(rhs)?;
        let bool_t = prim(PrimitiveType::Bool);
        if a.ty == Ty::Bool && b.ty == Ty::Bool {
            let op = if is_and { BinaryOp::And } else { BinaryOp::Or };
            return Ok(Val {
                node: binary(op, a.node, b.node, bool_t, span),
                ty: Ty::Bool,
            });
        }
        // The result type is the join; `a` is read twice, so it is held.
        let ty = types::logical_ty(is_and, a.ty, b.ty).settled();
        let mut pre = Vec::new();
        let a = self.hold(a, &mut pre);
        let test = self.truthy(a.clone());
        let a_node = self.coerce(a, ty);
        let b_node = self.coerce(b, ty);
        let (then, els) = if is_and {
            (b_node, a_node)
        } else {
            (a_node, b_node)
        };
        let value = if_value(test, then, els, self.ir(ty), span);
        Ok(Val {
            node: block_value(pre, value, span),
            ty,
        })
    }

    // ─── tables ─────────────────────────────────────────────────

    fn table_constructor(&mut self, t: &ast::TableConstructor, span: Span) -> Result<Val> {
        self.table_constructor_with(t, None, span)
    }

    /// A constructor; with `meta`, the table is born with that
    /// metatable, which the caller has checked the constructor's shape
    /// has slots for.
    fn table_constructor_with(
        &mut self,
        t: &ast::TableConstructor,
        meta: Option<&Expression>,
        span: Span,
    ) -> Result<Val> {
        let fields: Vec<&ast::Field> = t.fields().iter().collect();
        let positional: Vec<&Expression> = fields
            .iter()
            .filter_map(|f| match f {
                ast::Field::NoKey(e) => Some(e),
                _ => None,
            })
            .collect();
        // No positional values: the shared empty array part, made by
        // no one.
        let arr = if positional.is_empty() {
            None
        } else {
            Some(self.packed_list(&positional, span)?)
        };
        let table_t = self.ir(Ty::Table);
        // A constructor of a slotted shape lays its table out itself.
        if let Some(k) = types::constructor_shape(self.m.inferred, t)
            && self.m.layout(k).is_some()
        {
            return self.shaped_constructor(k, &fields, arr, positional.len(), meta, span);
        }
        debug_assert!(meta.is_none(), "a metatable at birth needs a slotted shape");
        let table = match arr {
            Some(arr) => call("zl_table_with_arr", vec![arr], table_t.clone(), span),
            None => call("zl_table_new", vec![], table_t.clone(), span),
        };
        let keyed: Vec<&ast::Field> = fields
            .iter()
            .copied()
            .filter(|f| !matches!(f, ast::Field::NoKey(_)))
            .collect();
        if keyed.is_empty() {
            return Ok(Val {
                node: table,
                ty: Ty::Table,
            });
        }
        let name = self.temp();
        let mut pre = vec![let_(name, table_t.clone(), table, span)];
        for f in keyed {
            let tb = var(name, table_t.clone(), span);
            match f {
                ast::Field::NameKey { key, value, .. } => {
                    let v = self.expr(value)?;
                    let k = Val {
                        node: str_lit(&ident(key), span),
                        ty: Ty::Str,
                    };
                    pre.push(self.raw_store(tb, k, v, span));
                }
                ast::Field::ExpressionKey { key, value, .. } => {
                    let k = self.expr(key)?;
                    let v = self.expr(value)?;
                    pre.push(self.raw_store(tb, k, v, span));
                }
                _ => {}
            }
        }
        Ok(Val {
            node: block_value(pre, var(name, table_t, span), span),
            ty: Ty::Table,
        })
    }

    /// A constructor whose tables have shape `k`, laid out with its
    /// slots: the header, then each constant-key field in its slot,
    /// present when its value is not nil. Fields under other keys are
    /// stored afterwards, as into a plain table. `meta`, evaluated
    /// after the fields, is the metatable the table is born with.
    fn shaped_constructor(
        &mut self,
        k: ShapeId,
        fields: &[&ast::Field],
        arr: Option<Node>,
        positional: usize,
        meta: Option<&Expression>,
        span: Span,
    ) -> Result<Val> {
        let table_t = self.ir(Ty::Table);
        let i64_t = prim(PrimitiveType::I64);
        let layout = self.m.layout(k).expect("a slotted shape");
        let shape_ty = self.m.shape_ty(layout);
        let gid = layout.gid;
        let slot_of = |name: &str| layout.slots.iter().find(|s| s.name == name).cloned();
        // The values, in source order, kept apart from the slots they
        // go in.
        let mut pre = Vec::new();
        let mut slot_values: Vec<(Slot, Val)> = Vec::new();
        let mut other: Vec<(Val, Val)> = Vec::new();
        for f in fields {
            match f {
                ast::Field::NameKey { key, value, .. } => {
                    let v = self.expr(value)?;
                    let v = self.hold(v, &mut pre);
                    let name = ident(key);
                    match slot_of(&name) {
                        Some(slot) => slot_values.push((slot, v)),
                        None => other.push((
                            Val {
                                node: str_lit(&name, span),
                                ty: Ty::Str,
                            },
                            v,
                        )),
                    }
                }
                ast::Field::ExpressionKey { key, value, .. } => {
                    let kv = self.expr(key)?;
                    let kv = self.hold(kv, &mut pre);
                    let v = self.expr(value)?;
                    let v = self.hold(v, &mut pre);
                    match crate::scope::literal_string(key).and_then(|n| slot_of(&n)) {
                        Some(slot) => slot_values.push((slot, v)),
                        None => other.push((kv, v)),
                    }
                }
                _ => {}
            }
        }
        // A table the lowering has just made has no metatable to
        // protect: `setmetatable` of it with a class that holds neither
        // `__gc` nor `__mode` is the header's store; anything else goes
        // through the library, which checks it and tells the collector.
        let (meta, meta_checked) = match meta {
            Some(e) => {
                let m = self.expr(e)?;
                let m = self.hold(m, &mut pre);
                let inferred = self.m.inferred;
                match m.ty {
                    Ty::Shape(c)
                        if !inferred.class_may_hold(c, "__gc")
                            && !inferred.class_may_hold(c, "__mode") =>
                    {
                        (m.node, None)
                    }
                    _ => (null(table_t.clone(), span), Some(self.boxed(m))),
                }
            }
            None => (null(table_t.clone(), span), None),
        };
        // A key given twice: the last value stands, as in a plain table.
        let mut inits: Vec<TypedFieldInit> = Vec::new();
        let mut present: Node = int_lit(0, span);
        let mut filled: Vec<usize> = Vec::new();
        let layout = self.m.layout(k).expect("a slotted shape");
        for slot in &layout.slots {
            let value = slot_values
                .iter()
                .rev()
                .find(|(s, _)| s.bit == slot.bit)
                .map(|(_, v)| v.clone());
            if slot.kind == SlotKind::Number {
                let (bits, words) = self.number_slot_init(slot, value, &mut pre, span);
                present = binary(BinaryOp::BitOr, present, bits, i64_t.clone(), span);
                inits.extend(words);
                continue;
            }
            let stored = match value {
                Some(v) => {
                    filled.push(slot.bit);
                    let stored = self.coerce(v, slot.stored_ty());
                    let stored = self
                        .hold(
                            Val {
                                node: stored,
                                ty: slot.stored_ty(),
                            },
                            &mut pre,
                        )
                        .node;
                    let bit = match self.slot_nil(&stored, slot, span) {
                        Some(is_nil) => if_value(
                            is_nil,
                            int_lit(0, span),
                            int_lit(slot.mask(), span),
                            i64_t.clone(),
                            span,
                        ),
                        None => int_lit(slot.mask(), span),
                    };
                    present = binary(BinaryOp::BitOr, present, bit, i64_t.clone(), span);
                    stored
                }
                // An absent slot holds nothing: null for a table, never
                // a table made for it.
                None => match slot.kind {
                    SlotKind::Table => null(table_t.clone(), span),
                    _ => self.zero_of(slot.stored_ty(), span),
                },
            };
            match slot.kind {
                SlotKind::Scalar => {
                    let parts = NumberParts::of(&stored, span);
                    for (suffix, part) in [("t", parts.tag), ("i", parts.int), ("f", parts.float)] {
                        inits.push(TypedFieldInit {
                            name: intern(&format!("s{}{suffix}", slot.bit)),
                            value: Box::new(part),
                        });
                    }
                }
                _ => inits.push(TypedFieldInit {
                    name: intern(&format!("s{}", slot.bit)),
                    value: Box::new(stored),
                }),
            }
        }
        // The struct's initializers are plain reads: the presence word
        // and the array part are settled ahead of it.
        let present = self
            .hold(
                Val {
                    node: present,
                    ty: Ty::Int,
                },
                &mut pre,
            )
            .node;
        let arr = self
            .hold(
                Val {
                    node: match arr {
                        Some(arr) => call("zl_arr_box", vec![arr], Type::Any, span),
                        // The shared empty one, made on the first table.
                        None => {
                            let shared = var(intern(library::ARR_EMPTY), Type::Any, span);
                            if_value(
                                binary(
                                    BinaryOp::Ne,
                                    shared.clone(),
                                    nil(span),
                                    prim(PrimitiveType::Bool),
                                    span,
                                ),
                                shared,
                                call("zl_arr_shared", vec![], Type::Any, span),
                                Type::Any,
                                span,
                            )
                        }
                    },
                    ty: Ty::Any,
                },
                &mut pre,
            )
            .node;
        let header = [
            ("arr", arr),
            ("hash", nil(span)),
            ("meta", meta),
            ("high", int_lit(positional as i64, span)),
            ("shape", int_lit(gid, span)),
            ("present", present),
        ];
        let mut all: Vec<TypedFieldInit> = header
            .into_iter()
            .map(|(name, value)| TypedFieldInit {
                name: intern(name),
                value: Box::new(value),
            })
            .collect();
        all.extend(inits);
        let object = node(
            TypedExpression::Struct(TypedStructLiteral {
                name: intern(&layout.name),
                fields: all,
            }),
            shape_ty,
            span,
        );
        // The table as every holder sees it.
        let table = as_shape(object, table_t.clone(), span);
        let _ = filled;
        if other.is_empty() && meta_checked.is_none() {
            return Ok(Val {
                node: block_value(pre, table, span),
                ty: Ty::Shape(k),
            });
        }
        let name = self.temp();
        pre.push(let_(name, table_t.clone(), table, span));
        for (key, value) in other {
            let tb = var(name, table_t.clone(), span);
            pre.push(self.raw_store(tb, key, value, span));
        }
        if let Some(m) = meta_checked {
            let set = call(
                "zl_setmetatable",
                vec![var(name, table_t.clone(), span), m],
                table_t.clone(),
                span,
            );
            pre.push(self.guarded_stmt(set, &Described::NONE));
        }
        Ok(Val {
            node: block_value(pre, var(name, table_t, span), span),
            ty: Ty::Shape(k),
        })
    }

    /// `t[k] = v` without metamethods, on a table.
    fn raw_store(&mut self, tb: Node, k: Val, v: Val, span: Span) -> St {
        let v = self.boxed(v);
        let unit = prim(PrimitiveType::Unit);
        let node = if let Some(kb) = self.constant_key(&k) {
            call("zl_rawset_key", vec![tb, kb, v], unit, span)
        } else {
            match k.ty {
                Ty::Int => call("zl_rawseti", vec![tb, k.node, v], unit, span),
                Ty::Str => call("zl_rawset_str", vec![tb, k.node, v], unit, span),
                _ => {
                    let k = self.boxed(k);
                    call("zl_rawset", vec![tb, k, v], unit, span)
                }
            }
        };
        self.guarded_stmt(node, &Described::NONE)
    }

    /// The name a constant string key spells.
    fn literal_key(&self, key: &Val) -> Option<String> {
        match &key.node.node {
            TypedExpression::Literal(TypedLiteral::String(s)) if key.ty == Ty::Str => {
                s.resolve_global()
            }
            _ => None,
        }
    }

    /// A name the program spells, boxed once and shared: the compiler
    /// keeps a boxed string literal as one constant.
    fn constant_key(&mut self, key: &Val) -> Option<Node> {
        match &key.node.node {
            TypedExpression::Literal(TypedLiteral::String(_)) if key.ty == Ty::Str => {
                Some(self.boxed(key.clone()))
            }
            _ => None,
        }
    }

    /// `obj.name`: a read whose result the types may know, through a
    /// shaped receiver; from the slot when the receiver's shape has one
    /// for the name and it holds a value, else the dynamic path, which
    /// raises for nil and consults the metatable for an absent field.
    fn field_read(&mut self, obj: Val, key: Val, name: &str, desc: Desc, span: Span) -> Val {
        let Ty::Shape(k) = obj.ty else {
            return self.index_read(obj, key, desc, span);
        };
        let ty = self.typer().field_ty(Ty::Shape(k), name).settled();
        let table_t = self.ir(Ty::Table);
        let bool_t = prim(PrimitiveType::Bool);
        let mut pre = Vec::new();
        let t = self.hold(obj, &mut pre);
        let not_null = |x: &Node| {
            binary(
                BinaryOp::Ne,
                x.clone(),
                null(table_t.clone(), span),
                bool_t.clone(),
                span,
            )
        };
        let info = self.m.inferred.shape(k);
        let own = info.field(name).map(|(_, ty)| ty);
        let sure = info.always_present(name);
        let slot = if own.is_some() {
            self.slot_of(k, name)
        } else {
            None
        };
        let finder = if sure {
            None
        } else {
            self.finder(k, name, FinderMode::Table, span)
        };
        // Whether every place the field may be found is checked here:
        // the table's own slot, and the finder for each lookup that
        // goes through a metatable. Then what nothing finds is nil.
        let closed = !sure
            && slot.is_some()
            && self.m.inferred.lookups(k, name).is_some_and(|lookups| {
                finder.is_some() || lookups.iter().all(|l| l.hops.is_empty())
            });
        // Where nothing below finds the field. A slot the table always
        // holds is only missed by a nil receiver, which raises; so is
        // a closed lookup, which else gives nil; anything else takes
        // the general read, which raises for nil and consults a
        // metatable the types do not follow.
        let slot_desc = desc.clone();
        let mut value = if sure && slot.is_some() {
            self.index_nil(desc, ty, span)
        } else if closed {
            let raise = self.index_nil(desc, ty, span);
            let absent = self.coerce(
                Val {
                    node: nil(span),
                    ty: Ty::Nil,
                },
                ty,
            );
            if_value(not_null(&t.node), absent, raise, self.ir(ty), span)
        } else {
            let general = self.index_read(t.clone(), key, desc, span);
            self.coerce(general, ty)
        };
        // Then through the metatables, by the finder.
        if let Some(finder) = finder {
            let found = self.hold(
                Val {
                    node: call(&finder.helper, vec![t.node.clone()], table_t.clone(), span),
                    ty: Ty::Table,
                },
                &mut pre,
            );
            let read = self.found_value(&finder, &found.node, name, ty, span);
            value = if_value(not_null(&found.node), read, value, self.ir(ty), span);
        }
        // The table's own slot first.
        if let Some((layout, slot)) = &slot
            && sure
            && slot.kind == SlotKind::Number
            && ty == Ty::Number
        {
            value = self.sure_number_read(&t.node, layout, slot, slot_desc, span);
        } else if let Some((layout, slot)) = slot {
            let fast = if sure {
                not_null(&t.node)
            } else {
                binary(
                    BinaryOp::And,
                    not_null(&t.node),
                    self.slot_present(&t.node, &slot, span),
                    bool_t,
                    span,
                )
            };
            let from_slot = self.slot_read_as(&t.node, layout, &slot, ty, span);
            let from_slot = self.coerce(from_slot, ty);
            value = if_value(fast, from_slot, value, self.ir(ty), span);
        }
        Val {
            node: block_value(pre, value, span),
            ty,
        }
    }

    /// Number slot `slot` of table `t`, which every table of its shape
    /// holds: a nil `t` raises in computing the tag, ahead of the two
    /// words' reads, so the number is built from its parts with no
    /// merge of whole numbers.
    fn sure_number_read(
        &mut self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        desc: Desc,
        span: Span,
    ) -> Node {
        let read = self.number_slot_read(t, layout, slot, Ty::Number, span);
        let TypedExpression::Tuple(parts) = read.node.node else {
            unreachable!("a number slot reads as its three parts")
        };
        let [tag, int, float]: [Node; 3] = parts.try_into().expect("three parts");
        let not_null = binary(
            BinaryOp::Ne,
            t.clone(),
            null(self.ir(Ty::Table), span),
            prim(PrimitiveType::Bool),
            span,
        );
        let raise = self.index_nil(desc, Ty::Int, span);
        let tag = if_value(not_null, tag, raise, prim(PrimitiveType::I64), span);
        number_value(tag, int, float, span)
    }

    /// Indexing a nil receiver: the error, raised and left through,
    /// then a value of `ty` no one reads. The raise always leaves an
    /// error pending, so nothing past it runs.
    fn index_nil(&mut self, desc: Desc, ty: Ty, span: Span) -> Node {
        let descs = Described::operand(desc);
        let leave = self.leave_described(span, &descs);
        let raise = vec![
            expr_stmt(call("zl_index_nil", vec![], Type::Any, span)),
            if_(bool_lit(true, span), leave, None, span),
        ];
        let unread = match ty {
            Ty::Table => null(self.ir(Ty::Table), span),
            _ => self.zero_of(ty, span),
        };
        block_value(raise, unread, span)
    }

    /// `obj[key]`, with `__index`; `desc` is what `obj` is called.
    fn index_read(&mut self, obj: Val, key: Val, desc: Desc, span: Span) -> Val {
        let descs = Described::operand(desc);
        if let Some(k) = self.constant_key(&key) {
            let node = match obj.ty {
                Ty::Table => call("zl_table_index_key", vec![obj.node, k], Type::Any, span),
                // A shaped table is a table unless it is nil, when the
                // dynamic path raises as Lua does.
                Ty::Shape(_) => {
                    let mut pre = Vec::new();
                    let t = self.hold(obj, &mut pre);
                    let is_null = binary(
                        BinaryOp::Eq,
                        t.node.clone(),
                        null(self.ir(Ty::Table), span),
                        prim(PrimitiveType::Bool),
                        span,
                    );
                    let read = if_value(
                        is_null,
                        call("zl_index_key", vec![nil(span), k.clone()], Type::Any, span),
                        call("zl_table_index_key", vec![t.node, k], Type::Any, span),
                        Type::Any,
                        span,
                    );
                    let v = self.guard_described(
                        Val {
                            node: block_value(pre, read, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    );
                    return v;
                }
                _ => {
                    let o = self.boxed(obj);
                    call("zl_index_key", vec![o, k], Type::Any, span)
                }
            };
            return self.guarded_described(Val { node, ty: Ty::Any }, &descs);
        }
        if let Ty::Shape(s) = obj.ty {
            return self.shape_index_read(obj, s, key, &descs, span);
        }
        let node = match obj.ty {
            Ty::Table => match key.ty {
                Ty::Int => call("zl_table_geti", vec![obj.node, key.node], Type::Any, span),
                Ty::Str => call(
                    "zl_table_index_str",
                    vec![obj.node, key.node],
                    Type::Any,
                    span,
                ),
                _ => {
                    let k = self.boxed(key);
                    call("zl_table_index", vec![obj.node, k], Type::Any, span)
                }
            },
            _ => {
                let o = self.boxed(obj);
                match key.ty {
                    Ty::Int => call("zl_geti", vec![o, key.node], Type::Any, span),
                    Ty::Str => call("zl_index_str", vec![o, key.node], Type::Any, span),
                    _ => {
                        let k = self.boxed(key);
                        call("zl_index", vec![o, k], Type::Any, span)
                    }
                }
            }
        };
        self.guarded_described(Val { node, ty: Ty::Any }, &descs)
    }

    /// `obj[key]` for `obj` a table of shape `s` and a key that is not a
    /// constant string. Under an integer, when the types say what the
    /// array part holds and no metatable could answer instead, an
    /// element in range is read in place; everything else takes the
    /// general path, which raises for a nil receiver.
    fn shape_index_read(
        &mut self,
        obj: Val,
        s: ShapeId,
        key: Val,
        descs: &Described,
        span: Span,
    ) -> Val {
        let table_t = self.ir(Ty::Table);
        let bool_t = prim(PrimitiveType::Bool);
        let i64_t = prim(PrimitiveType::I64);
        let mut pre = Vec::new();
        let t = self.hold(obj, &mut pre);
        let k = self.hold(key, &mut pre);
        let is_null = binary(
            BinaryOp::Eq,
            t.node.clone(),
            null(table_t.clone(), span),
            bool_t.clone(),
            span,
        );
        let (on_table, on_nil) = match k.ty {
            Ty::Int => (
                call(
                    "zl_table_geti",
                    vec![t.node.clone(), k.node.clone()],
                    Type::Any,
                    span,
                ),
                call("zl_geti", vec![nil(span), k.node.clone()], Type::Any, span),
            ),
            Ty::Str => (
                call(
                    "zl_table_index_str",
                    vec![t.node.clone(), k.node.clone()],
                    Type::Any,
                    span,
                ),
                call(
                    "zl_index_str",
                    vec![nil(span), k.node.clone()],
                    Type::Any,
                    span,
                ),
            ),
            _ => {
                let boxed = self.boxed(k.clone());
                let boxed = self
                    .hold(
                        Val {
                            node: boxed,
                            ty: Ty::Any,
                        },
                        &mut pre,
                    )
                    .node;
                (
                    call(
                        "zl_table_index",
                        vec![t.node.clone(), boxed.clone()],
                        Type::Any,
                        span,
                    ),
                    call("zl_index", vec![nil(span), boxed], Type::Any, span),
                )
            }
        };
        let general = self.guard_described(
            Val {
                node: if_value(is_null, on_nil, on_table, Type::Any, span),
                ty: Ty::Any,
            },
            descs,
        );
        let ty = self.m.inferred.element_read_ty(s, k.ty).settled();
        // Without a metatable to ask, an element in range is the
        // answer whatever its kind; a float key names an element only
        // when it is integral, and the general path decides that.
        let info = self.m.inferred.shape(s);
        let plain = !info.unknown_meta && info.classes.is_empty();
        if !plain {
            let node = if ty == Ty::Any {
                general.node
            } else {
                self.coerce(general, ty)
            };
            return Val {
                node: block_value(pre, node, span),
                ty,
            };
        }
        if k.ty != Ty::Int {
            let general = self.coerce(general, ty);
            return Val {
                node: block_value(pre, general, span),
                ty,
            };
        }
        // The array part: `1..len` holds elements `0..len - 1`.
        let arr = || {
            call(
                "zb_unbox_list_raw_any",
                vec![field(t.node.clone(), "arr", Type::Any, span)],
                self.m.anys(),
                span,
            )
        };
        let in_range = binary(
            BinaryOp::And,
            binary(
                BinaryOp::And,
                binary(
                    BinaryOp::Ne,
                    t.node.clone(),
                    null(table_t, span),
                    bool_t.clone(),
                    span,
                ),
                binary(
                    BinaryOp::Ge,
                    k.node.clone(),
                    int_lit(1, span),
                    bool_t.clone(),
                    span,
                ),
                bool_t.clone(),
                span,
            ),
            binary(
                BinaryOp::Le,
                k.node.clone(),
                list_len(arr(), span),
                bool_t.clone(),
                span,
            ),
            bool_t,
            span,
        );
        let element = index(
            arr(),
            binary(BinaryOp::Sub, k.node.clone(), int_lit(1, span), i64_t, span),
            Type::Any,
            span,
        );
        let element = self.coerce(
            Val {
                node: element,
                ty: Ty::Any,
            },
            ty,
        );
        let general = self.coerce(general, ty);
        Val {
            node: block_value(
                pre,
                if_value(in_range, element, general, self.ir(ty), span),
                span,
            ),
            ty,
        }
    }

    /// Whether a value of its type is nil.
    fn is_nil_val(&mut self, v: Val) -> Node {
        let span = v.node.span;
        let bool_t = prim(PrimitiveType::Bool);
        match v.ty {
            Ty::Nil => block_value(vec![expr_stmt(v.node)], bool_lit(true, span), span),
            Ty::Bool | Ty::Int | Ty::Float | Ty::Number | Ty::Str | Ty::Table => {
                if Self::is_simple(&v.node) {
                    bool_lit(false, span)
                } else {
                    block_value(vec![expr_stmt(v.node)], bool_lit(false, span), span)
                }
            }
            Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => {
                let (pre, n) = self.number_parts(v);
                block_value(pre, n.has_tag(TAG_NIL, span), span)
            }
            Ty::Shape(_) => binary(
                BinaryOp::Eq,
                v.node,
                null(self.ir(Ty::Table), span),
                bool_t,
                span,
            ),
            Ty::Func(_) | Ty::Any | Ty::Unknown => {
                binary(BinaryOp::Eq, v.node, nil(span), bool_t, span)
            }
        }
    }

    /// `obj[key] = value`, with `__newindex`; `desc` is what `obj` is
    /// called.
    fn index_write(&mut self, obj: Val, key: Val, value: Val, desc: Desc, span: Span) -> St {
        let descs = Described::operand(desc);
        let unit = prim(PrimitiveType::Unit);
        // A shaped receiver's slot takes the value when the field is
        // there already or no metatable could take it instead; else
        // the dynamic path, which raises for nil and asks `__newindex`.
        if let (Ty::Shape(shape), Some(name)) = (obj.ty, self.literal_key(&key)) {
            let found = self.m.layout(shape).and_then(|layout| {
                layout
                    .slots
                    .iter()
                    .find(|s| s.name == name)
                    .map(|s| (layout, s))
            });
            if let Some((layout, slot)) = found {
                let bool_t = prim(PrimitiveType::Bool);
                let mut pre = Vec::new();
                let t = self.hold(obj, &mut pre);
                let stored = self.slot_value(value, slot);
                let stored = self.hold(stored, &mut pre);
                let not_null = binary(
                    BinaryOp::Ne,
                    t.node.clone(),
                    null(self.ir(Ty::Table), span),
                    bool_t.clone(),
                    span,
                );
                // No metatable the types know of for this shape holds a
                // `__newindex`: nothing can take the store instead.
                let info = self.m.inferred.shape(shape);
                let no_newindex = !info.unknown_meta
                    && info
                        .classes
                        .iter()
                        .all(|c| self.m.inferred.shape(*c).field("__newindex").is_none());
                let fast = if no_newindex {
                    not_null
                } else {
                    let no_meta = binary(
                        BinaryOp::Eq,
                        field(t.node.clone(), "meta", self.ir(Ty::Table), span),
                        null(self.ir(Ty::Table), span),
                        bool_t.clone(),
                        span,
                    );
                    binary(
                        BinaryOp::And,
                        not_null,
                        binary(
                            BinaryOp::Or,
                            self.slot_present(&t.node, slot, span),
                            no_meta,
                            bool_t.clone(),
                            span,
                        ),
                        bool_t,
                        span,
                    )
                };
                // A field every table of the shape is born with, that
                // nothing stores nil in and no store under a computed
                // key reaches, keeps its bit set.
                let settled = info.always_present(&name) && !info.dynamic_keys;
                let then = self.slot_store(&t.node, layout, slot, stored.clone(), settled, span);
                let boxed = self.coerce(stored, Ty::Any);
                let boxed_t = self.coerce(t, Ty::Any);
                let k = self.constant_key(&key).expect("a constant key");
                let els = vec![
                    expr_stmt(call("zl_setindex_key", vec![boxed_t, k, boxed], unit, span)),
                    self.pending_check_described(span, &descs),
                ];
                pre.push(if_(fast, then, Some(els), span));
                return stmt(
                    TypedStatement::Block(TypedBlock {
                        statements: pre,
                        span,
                    }),
                    span,
                );
            }
        }
        let v = self.boxed(value);
        if let Some(k) = self.constant_key(&key) {
            let node = match obj.ty {
                Ty::Table => call("zl_table_setindex_key", vec![obj.node, k, v], unit, span),
                // A shaped table is a table unless it is nil, when the
                // dynamic path raises as Lua does.
                Ty::Shape(_) => {
                    let mut pre = Vec::new();
                    let t = self.hold(obj, &mut pre);
                    let held_v = self.hold(
                        Val {
                            node: v,
                            ty: Ty::Any,
                        },
                        &mut pre,
                    );
                    let is_null = binary(
                        BinaryOp::Eq,
                        t.node.clone(),
                        null(self.ir(Ty::Table), span),
                        prim(PrimitiveType::Bool),
                        span,
                    );
                    let store = if_(
                        is_null,
                        vec![expr_stmt(call(
                            "zl_setindex_key",
                            vec![nil(span), k.clone(), held_v.node.clone()],
                            unit.clone(),
                            span,
                        ))],
                        Some(vec![expr_stmt(call(
                            "zl_table_setindex_key",
                            vec![t.node, k, held_v.node],
                            unit,
                            span,
                        ))]),
                        span,
                    );
                    pre.push(store);
                    pre.push(self.pending_check_described(span, &descs));
                    return stmt(
                        TypedStatement::Block(TypedBlock {
                            statements: pre,
                            span,
                        }),
                        span,
                    );
                }
                _ => {
                    let o = self.boxed(obj);
                    call("zl_setindex_key", vec![o, k, v], unit, span)
                }
            };
            return self.guarded_stmt(node, &descs);
        }
        // A shaped receiver is a table unless it is nil, when the dynamic
        // path raises as Lua does.
        if let Ty::Shape(_) = obj.ty {
            let table_t = self.ir(Ty::Table);
            let mut pre = Vec::new();
            let t = self.hold(obj, &mut pre);
            let held_v = self
                .hold(
                    Val {
                        node: v,
                        ty: Ty::Any,
                    },
                    &mut pre,
                )
                .node;
            let (on_table, on_nil) = match key.ty {
                Ty::Int => {
                    let k = self.hold(key, &mut pre);
                    (
                        call(
                            "zl_table_seti",
                            vec![t.node.clone(), k.node.clone(), held_v.clone()],
                            unit.clone(),
                            span,
                        ),
                        call(
                            "zl_seti",
                            vec![nil(span), k.node, held_v],
                            unit.clone(),
                            span,
                        ),
                    )
                }
                Ty::Str => {
                    let k = self.hold(key, &mut pre);
                    (
                        call(
                            "zl_table_setindex_str",
                            vec![t.node.clone(), k.node.clone(), held_v.clone()],
                            unit.clone(),
                            span,
                        ),
                        call(
                            "zl_setindex_str",
                            vec![nil(span), k.node, held_v],
                            unit.clone(),
                            span,
                        ),
                    )
                }
                _ => {
                    let k = self.boxed(key);
                    let k = self
                        .hold(
                            Val {
                                node: k,
                                ty: Ty::Any,
                            },
                            &mut pre,
                        )
                        .node;
                    (
                        call(
                            "zl_table_setindex",
                            vec![t.node.clone(), k.clone(), held_v.clone()],
                            unit.clone(),
                            span,
                        ),
                        call(
                            "zl_setindex",
                            vec![nil(span), k, held_v],
                            unit.clone(),
                            span,
                        ),
                    )
                }
            };
            let is_null = binary(
                BinaryOp::Eq,
                t.node,
                null(table_t, span),
                prim(PrimitiveType::Bool),
                span,
            );
            pre.push(if_(
                is_null,
                vec![expr_stmt(on_nil)],
                Some(vec![expr_stmt(on_table)]),
                span,
            ));
            pre.push(self.pending_check_described(span, &descs));
            return stmt(
                TypedStatement::Block(TypedBlock {
                    statements: pre,
                    span,
                }),
                span,
            );
        }
        let node = match obj.ty {
            Ty::Table => match key.ty {
                Ty::Int => call("zl_table_seti", vec![obj.node, key.node, v], unit, span),
                Ty::Str => call(
                    "zl_table_setindex_str",
                    vec![obj.node, key.node, v],
                    unit,
                    span,
                ),
                _ => {
                    let k = self.boxed(key);
                    call("zl_table_setindex", vec![obj.node, k, v], unit, span)
                }
            },
            _ => {
                let o = self.boxed(obj);
                match key.ty {
                    Ty::Int => call("zl_seti", vec![o, key.node, v], unit, span),
                    Ty::Str => call("zl_setindex_str", vec![o, key.node, v], unit, span),
                    _ => {
                        let k = self.boxed(key);
                        call("zl_setindex", vec![o, k, v], unit, span)
                    }
                }
            }
        };
        self.guarded_stmt(node, &descs)
    }

    // ─── calls ──────────────────────────────────────────────────

    /// A prefix followed by suffixes: indexing and calls, left to
    /// right. The last suffix may produce several values.
    fn suffixed(&mut self, prefix: &Prefix, suffixes: &[&Suffix], span: Span) -> Result<Multi> {
        // `_G.name` is the global.
        if let Some(name) = self.typer().global_member(prefix, suffixes) {
            let v = self.read_global(&name, span)?;
            return Ok(Multi::Fixed(vec![v]));
        }
        if let Some(name) = self
            .typer()
            .global_member(prefix, &suffixes[..suffixes.len().min(1)])
        {
            let mut multi = None;
            let head = self.read_global(&name, span)?;
            let desc = self.describe_chain(prefix, &suffixes[..1]);
            return self.suffixes_from(head, desc, suffixes, 1, &mut multi, span);
        }
        // A direct call to a known function or a builtin, possibly
        // followed by more suffixes on its result: only the last call
        // of a `return` is a tail call.
        let (mut multi, first) = self.suffixed_head(prefix, suffixes, span)?;
        if first == suffixes.len()
            && let Some(m) = multi
        {
            return Ok(m);
        }
        let (current, desc) = match multi.take() {
            Some(m) => (self.first_of(m, span), None),
            None => match prefix {
                Prefix::Name(token) => (self.read_name(token)?, self.describe_name(token)),
                Prefix::Expression(e) => (self.expr(e)?, self.describe(e)),
                _ => return unsupported("this prefix", span),
            },
        };
        self.suffixes_from(current, desc, suffixes, first, &mut multi, span)
    }

    /// The first suffix of a chain applied as a direct call to a known
    /// function or a builtin, when it is one: the call and how many
    /// suffixes it took.
    fn suffixed_head(
        &mut self,
        prefix: &Prefix,
        suffixes: &[&Suffix],
        span: Span,
    ) -> Result<(Option<Multi>, usize)> {
        let mut multi: Option<Multi> = None;
        let mut first = 0;
        let tail = self.tail_span;
        if let Some(Suffix::Call(ast::Call::AnonymousCall(args))) = suffixes.first()
            && let Some(f) = self.typer().known_callee(prefix)
        {
            if suffixes.len() > 1 {
                self.tail_span = None;
            }
            // A global declared once as this function is the function,
            // and so is a local that holds a function from its
            // declaration on; any other name holds its value, which may
            // be nil by now.
            let declared = match prefix {
                Prefix::Name(token) => match self.scopes().binding(token) {
                    Some(Binding::Global(name)) => {
                        self.scopes().known_global_function(name) == Some(f)
                    }
                    Some(Binding::Local(v) | Binding::Upvalue(v)) => {
                        self.scopes().always_function(*v)
                    }
                    _ => false,
                },
                _ => false,
            };
            if declared {
                // The record holds the captures, when there are any.
                let record = match prefix {
                    Prefix::Name(token) if self.m.takes_env(f) => Some(self.read_name(token)?),
                    _ => None,
                };
                let desc = match prefix {
                    Prefix::Name(token) => self.describe_name(token),
                    _ => None,
                };
                multi = Some(self.direct_call(f, record, None, args, desc, span)?);
            } else {
                let callee = match prefix {
                    Prefix::Name(token) => self.read_name(token)?,
                    Prefix::Expression(e) => self.expr(e)?,
                    _ => return unsupported("this prefix", span),
                };
                let desc = match prefix {
                    Prefix::Name(token) => self.describe_name(token),
                    Prefix::Expression(e) => self.describe(e),
                    _ => None,
                };
                let returns = self.m.sig(f).returns.clone();
                let (pre, vals, tail) = self.call_values(None, args, span)?;
                multi = Some(self.dispatch(
                    callee,
                    Guard::NotNil,
                    &[f],
                    returns,
                    pre,
                    vals,
                    tail,
                    desc,
                    span,
                ));
            }
            self.tail_span = tail;
            first = 1;
        }
        if multi.is_none() {
            let head = suffixes.len().min(2);
            for n in (1..=head).rev() {
                if let Some(b) = self.typer().builtin_callee(prefix, &suffixes[..n]) {
                    let Some(Suffix::Call(ast::Call::AnonymousCall(args))) = suffixes[..n].last()
                    else {
                        unreachable!("a builtin callee ends in a call");
                    };
                    if suffixes.len() > n {
                        self.tail_span = None;
                    }
                    let called = self.builtin_call(b, None, args, span);
                    self.tail_span = tail;
                    multi = Some(called?);
                    first = n;
                    break;
                }
            }
        }
        Ok((multi, first))
    }

    /// The suffixes from `first` on, applied to `current`, which `desc`
    /// names.
    fn suffixes_from(
        &mut self,
        mut current: Val,
        mut desc: Desc,
        suffixes: &[&Suffix],
        first: usize,
        multi: &mut Option<Multi>,
        span: Span,
    ) -> Result<Multi> {
        let mut called = false;
        for (i, s) in suffixes.iter().enumerate().skip(first) {
            // The call a field call's lookup made already.
            if std::mem::take(&mut called) {
                if i + 1 == suffixes.len() {
                    return Ok(multi.take().expect("a field call's results"));
                }
                continue;
            }
            if let Some(m) = multi.take() {
                current = self.first_of(m, span);
                desc = None;
            }
            let last = i + 1 == suffixes.len();
            if let Suffix::Index(ast::Index::Dot { name, .. }) = s
                && let Some(next) = suffixes.get(i + 1)
                && let Some(m) =
                    self.field_call(&current, &ident(name), next, desc.clone(), span)?
            {
                *multi = Some(m);
                called = true;
                desc = None;
                continue;
            }
            match s {
                Suffix::Index(ast::Index::Dot { name, .. }) => {
                    let key = Val {
                        node: str_lit(&ident(name), span),
                        ty: Ty::Str,
                    };
                    current = self.field_read(current, key, &ident(name), desc, span);
                    desc = Self::describe_suffix(s);
                }
                Suffix::Index(ast::Index::Brackets { expression, .. }) => {
                    let key = self.expr(expression)?;
                    current = match crate::scope::literal_string(expression) {
                        Some(name) => self.field_read(current, key, &name, desc, span),
                        None => self.index_read(current, key, desc, span),
                    };
                    desc = Self::describe_suffix(s);
                }
                Suffix::Call(ast::Call::AnonymousCall(args)) => {
                    let tail = self.tail_span.take_if(|_| !last);
                    let m = self.value_call(current.clone(), None, args, desc, span);
                    self.tail_span = self.tail_span.or(tail);
                    let m = m?;
                    if last {
                        return Ok(m);
                    }
                    *multi = Some(m);
                    desc = None;
                }
                Suffix::Call(ast::Call::MethodCall(mc)) => {
                    let tail = self.tail_span.take_if(|_| !last);
                    let m = self.method_call(current.clone(), mc, desc, span);
                    self.tail_span = self.tail_span.or(tail);
                    let m = m?;
                    if last {
                        return Ok(m);
                    }
                    *multi = Some(m);
                    desc = None;
                }
                _ => return unsupported("this suffix", span),
            }
        }
        Ok(Multi::Fixed(vec![current]))
    }

    /// `obj.name(args)` for a receiver of a shape whose lookups of
    /// `name` all end in slots holding functions the types resolve the
    /// call to: called directly, by the end the lookup takes. None for
    /// any other call; `desc` names `obj`.
    fn field_call(
        &mut self,
        obj: &Val,
        name: &str,
        call_suffix: &Suffix,
        desc: Desc,
        span: Span,
    ) -> Result<Option<Multi>> {
        let (Ty::Shape(k), Suffix::Call(ast::Call::AnonymousCall(args))) = (obj.ty, call_suffix)
        else {
            return Ok(None);
        };
        let (returns, funcs) = match self.typer().field_ty(Ty::Shape(k), name).settled() {
            Ty::Func(f) => (self.m.sig(f).returns.clone(), vec![f]),
            _ => {
                let Some(targets) = self.typer().method_targets(k, name) else {
                    return Ok(None);
                };
                let returns = self.typer().method_returns(Ty::Shape(k), name).settled();
                let funcs = targets
                    .iter()
                    .filter_map(|t| match t {
                        Ty::Func(f) => Some(*f),
                        _ => None,
                    })
                    .collect();
                (returns, funcs)
            }
        };
        let Some(which) = self.finder(k, name, FinderMode::Which, span) else {
            return Ok(None);
        };
        let known = which.ends.iter().all(|(_, end)| match end {
            End::Slot(Ty::Func(f)) => funcs.contains(f),
            _ => false,
        });
        if !known {
            return Ok(None);
        }
        self.which_call(
            obj.clone(),
            k,
            name,
            CallKind::Field,
            returns,
            args,
            desc,
            span,
        )
    }

    fn args_exprs<'e>(&self, args: &'e ast::FunctionArgs) -> Vec<&'e Expression> {
        match args {
            ast::FunctionArgs::Parentheses { arguments, .. } => arguments.iter().collect(),
            _ => Vec::new(),
        }
    }

    /// The single argument a call written `f "x"` or `f {…}` passes.
    fn literal_arg(&mut self, args: &ast::FunctionArgs, span: Span) -> Result<Option<Val>> {
        Ok(match args {
            ast::FunctionArgs::String(t) => {
                let bytes = string_bytes(t).map_err(|message| Error::Syntax {
                    message,
                    span: (span.start, span.end),
                })?;
                Some(Val {
                    node: string_literal(bytes, span),
                    ty: Ty::Str,
                })
            }
            ast::FunctionArgs::TableConstructor(t) => Some(self.table_constructor(t, span)?),
            _ => None,
        })
    }

    /// The values a call's arguments supply: `receiver` first for a
    /// method call.
    fn call_values(
        &mut self,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        span: Span,
    ) -> Result<(Vec<St>, Vec<Val>, Option<Node>)> {
        let mut vals = Vec::new();
        if let Some(r) = receiver {
            vals.push(r);
        }
        if let Some(v) = self.literal_arg(args, span)? {
            vals.push(v);
            return Ok((Vec::new(), vals, None));
        }
        let exprs = self.args_exprs(args);
        let (mut pre, more, tail) = self.expr_list(&exprs)?;
        if !pre.is_empty() {
            // The receiver runs before the arguments.
            let held = std::mem::take(&mut vals);
            let mut first = Vec::new();
            for v in held {
                let h = self.hold(v, &mut first);
                vals.push(h);
            }
            first.append(&mut pre);
            pre = first;
        }
        vals.extend(more);
        Ok((pre, vals, tail))
    }

    /// A call to a known function: its typed entry, the arguments
    /// adjusted to its parameters. `record` is the function's value,
    /// when the call went through one, which holds its captures;
    /// `receiver` is the object of a method call, passed first.
    fn direct_call(
        &mut self,
        f: FuncId,
        record: Option<Val>,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        desc: Desc,
        span: Span,
    ) -> Result<Multi> {
        // The function's value is read before its arguments run.
        let mut pre = Vec::new();
        let record = record.map(|r| self.hold(r, &mut pre));
        let (mut arg_pre, vals, tail) = self.call_values(receiver, args, span)?;
        pre.append(&mut arg_pre);
        Ok(self.direct_call_vals(f, record, pre, vals, tail, &desc, span))
    }

    /// [`Self::direct_call`] with the arguments evaluated: `vals`, and
    /// `tail` when the last supplies several values.
    #[allow(clippy::too_many_arguments)]
    fn direct_call_vals(
        &mut self,
        f: FuncId,
        record: Option<Val>,
        mut pre: Vec<St>,
        vals: Vec<Val>,
        tail: Option<Node>,
        desc: &Desc,
        span: Span,
    ) -> Multi {
        let info = self.scopes().func(f);
        let sig = self.m.sig(f);
        let n = info.params.len();
        let is_vararg = info.is_vararg;
        let has_env = self.m.takes_env(f);
        // A tail of several values is read through a list.
        let tail_name = match tail {
            Some(tail) => {
                let name = self.temp();
                pre.push(let_(
                    name,
                    self.m.anys(),
                    call("zl_values", vec![tail], self.m.anys(), span),
                    span,
                ));
                Some(name)
            }
            None => None,
        };
        let mut lowered: Vec<Node> = Vec::new();
        if has_env {
            // A nested known function's record holds its captures; the
            // typed entry takes it as `env`. The record is the value
            // the call went through, else it lives in the function's
            // variable.
            let record = match record {
                Some(r) => r,
                None => match self
                    .scopes()
                    .local_functions
                    .iter()
                    .find(|(_, id)| **id == f)
                {
                    Some((v, _)) => self.read_var(*v, span),
                    None => self.function_value(f, span),
                },
            };
            lowered.push(call(
                "zb_unbox_list_raw_any",
                vec![record.node],
                self.m.anys(),
                span,
            ));
        }
        let mut fixed = vals.into_iter();
        let mut from_tail: i64 = 0;
        for i in 0..n {
            let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
            let v = match fixed.next() {
                Some(v) => v,
                None => match tail_name {
                    Some(t) => {
                        from_tail += 1;
                        Val {
                            node: call(
                                "zl_value_at",
                                vec![var(t, self.m.anys(), span), int_lit(from_tail, span)],
                                Type::Any,
                                span,
                            ),
                            ty: Ty::Any,
                        }
                    }
                    None => self.nil_val(span),
                },
            };
            let v = self.hold(v, &mut pre);
            lowered.push(self.coerce(v, param_ty));
        }
        // The extras: the variadic list, or run for their effects.
        let extras: Vec<Val> = fixed.collect();
        if is_vararg {
            let mut items = Vec::with_capacity(extras.len());
            for v in extras {
                // Evaluated ahead of the call site's own work.
                let v = if self.frames {
                    self.hold(v, &mut pre)
                } else {
                    v
                };
                items.push(self.boxed(v));
            }
            let list = self.array_of(items, &mut pre, span);
            match tail_name {
                Some(t) => {
                    let list_name = self.temp();
                    pre.push(let_(list_name, self.m.anys(), list, span));
                    let rest = call(
                        "zl_slice",
                        vec![
                            var(t, self.m.anys(), span),
                            int_lit(from_tail, span),
                            list_len(var(t, self.m.anys(), span), span),
                        ],
                        self.m.anys(),
                        span,
                    );
                    pre.push(expr_stmt(call(
                        "zb_list_extend_any",
                        vec![var(list_name, self.m.anys(), span), rest],
                        prim(PrimitiveType::Unit),
                        span,
                    )));
                    lowered.push(var(list_name, self.m.anys(), span));
                }
                None => lowered.push(list),
            }
        } else {
            for v in extras {
                pre.push(expr_stmt(v.node));
            }
        }
        let spill = self.before_call(desc, span, &mut pre);
        let value = call(
            &self.m.entry_name(f),
            lowered,
            self.m.return_ir(&sig.returns),
            span,
        );
        let after = self.after_call(spill, span);
        let raises = self.m.raises(f);
        if raises {
            self.raise_callees.insert(f);
        }
        self.call_result(value, &sig.returns, pre, after, raises, span)
    }

    /// The values a typed entry returned, checked for an error when the
    /// callee may raise.
    /// `after` runs once the call returned, ahead of its check.
    fn call_result(
        &mut self,
        value: Node,
        returns: &Returns,
        pre: Vec<St>,
        after: Vec<St>,
        raises: bool,
        span: Span,
    ) -> Multi {
        match returns {
            Returns::Fixed(types) if types.is_empty() => {
                let mut pre = pre;
                pre.push(expr_stmt(value));
                pre.extend(after);
                if raises {
                    pre.push(self.pending_check(span));
                }
                Multi::None(block_value(pre, nil(span), span))
            }
            Returns::Fixed(types) if types.len() == 1 => {
                let ty = types[0].settled();
                let mut pre = pre;
                let value = if raises || !after.is_empty() {
                    let held = self.hold(Val { node: value, ty }, &mut pre);
                    pre.extend(after);
                    if raises {
                        pre.push(self.pending_check(span));
                    }
                    held.node
                } else {
                    value
                };
                Multi::Fixed(vec![Val {
                    node: block_value(pre, value, span),
                    ty,
                }])
            }
            Returns::Fixed(types) => {
                // Several values: a list, each read out as its type.
                let name = self.temp();
                let mut pre = pre;
                pre.push(let_(name, self.m.anys(), value, span));
                pre.extend(after);
                if raises {
                    pre.push(self.pending_check(span));
                }
                let mut vals = Vec::with_capacity(types.len());
                for (i, ty) in types.iter().enumerate() {
                    let ty = ty.settled();
                    let element = index(
                        var(name, self.m.anys(), span),
                        int_lit(i as i64, span),
                        Type::Any,
                        span,
                    );
                    let read = self.coerce(
                        Val {
                            node: element,
                            ty: Ty::Any,
                        },
                        ty,
                    );
                    let read = if i == 0 {
                        block_value(std::mem::take(&mut pre), read, span)
                    } else {
                        read
                    };
                    vals.push(Val { node: read, ty });
                }
                Multi::Fixed(vals)
            }
            Returns::Dynamic | Returns::Unknown => {
                let mut pre = pre;
                let value = if after.is_empty() {
                    value
                } else {
                    let held = self.hold(
                        Val {
                            node: value,
                            ty: Ty::Any,
                        },
                        &mut pre,
                    );
                    pre.extend(after);
                    held.node
                };
                let value = if raises {
                    self.guard(Val {
                        node: value,
                        ty: Ty::Any,
                    })
                    .node
                } else {
                    value
                };
                Multi::Dynamic(block_value(pre, value, span))
            }
        }
    }

    /// A call through a function value, which `desc` names.
    fn value_call(
        &mut self,
        callee: Val,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        desc: Desc,
        span: Span,
    ) -> Result<Multi> {
        // A value the types know the function of calls it directly,
        // once it is seen not to be nil.
        if let Ty::Func(f) = callee.ty {
            let returns = self.m.sig(f).returns.clone();
            let (pre, vals, tail) = self.call_values(receiver, args, span)?;
            return Ok(self.dispatch(
                callee,
                Guard::NotNil,
                &[f],
                returns,
                pre,
                vals,
                tail,
                desc,
                span,
            ));
        }
        // A value known to be nil cannot be called; the error names
        // what it was, as Lua's does.
        if callee.ty == Ty::Nil {
            let (mut pre, _, _) = self.call_values(receiver, args, span)?;
            let what = match desc {
                Some(d) => format!("attempt to call a nil value ({d})"),
                None => "attempt to call a nil value".to_string(),
            };
            let error = call(
                "zb_fatal",
                vec![str_lit("error", span), str_lit(&what, span)],
                prim(PrimitiveType::Unit),
                span,
            );
            pre.push(self.guarded_stmt(error, &Described::NONE));
            return Ok(Multi::None(block_value(pre, nil(span), span)));
        }
        let f = self.boxed(callee);
        let method = receiver.is_some();
        let (pre, vals, tail) = self.call_values(receiver, args, span)?;
        Ok(self.value_call_vals(f, pre, vals, tail, desc, method, span))
    }

    /// [`Self::value_call`] with the callee boxed and the arguments
    /// evaluated. A `method` call enters through the method-site
    /// entries, which tell a library function its first argument is
    /// `self`.
    #[allow(clippy::too_many_arguments)]
    fn value_call_vals(
        &mut self,
        f: Node,
        mut pre: Vec<St>,
        vals: Vec<Val>,
        tail: Option<Node>,
        desc: Desc,
        method: bool,
        span: Span,
    ) -> Multi {
        let entry = if method {
            "zl_apply_method"
        } else {
            "zl_apply"
        };
        // A call site that keeps the debug library's record evaluates
        // everything ahead of storing its number.
        let frames = self.frames;
        let f = if pre.is_empty() && !frames {
            f
        } else {
            self.hold(
                Val {
                    node: f,
                    ty: Ty::Any,
                },
                &mut pre,
            )
            .node
        };
        let vals: Vec<Val> = if frames {
            vals.into_iter().map(|v| self.hold(v, &mut pre)).collect()
        } else {
            vals
        };
        let tail = match tail {
            Some(t) if frames => Some(
                self.hold(
                    Val {
                        node: t,
                        ty: Ty::Any,
                    },
                    &mut pre,
                )
                .node,
            ),
            t => t,
        };
        let descs = Described::callee(desc.clone());
        let called = if tail.is_none() && vals.len() <= zyntax_builtins::functions::MAX_CALL_ARITY {
            let mut lowered = vec![f];
            let n = vals.len();
            for v in vals {
                lowered.push(self.boxed(v));
            }
            call(&format!("{entry}_{n}"), lowered, Type::Any, span)
        } else {
            let mut items = Vec::with_capacity(vals.len());
            for v in vals {
                items.push(self.boxed(v));
            }
            let list = self.array_of(items, &mut pre, span);
            let list = match tail {
                None => list,
                Some(tail) => {
                    let name = self.temp();
                    pre.push(let_(name, self.m.anys(), list, span));
                    pre.push(expr_stmt(call(
                        "zl_append_values",
                        vec![var(name, self.m.anys(), span), tail],
                        prim(PrimitiveType::Unit),
                        span,
                    )));
                    var(name, self.m.anys(), span)
                }
            };
            call(&format!("{entry}_packed"), vec![f, list], Type::Any, span)
        };
        let spill = self.before_call(&desc, span, &mut pre);
        let mut after = self.after_call(spill, span);
        // A callee of the library's enters no frame to take the site.
        if frames {
            after.push(assign(
                var(
                    intern(library::debug::DBG_SITE),
                    prim(PrimitiveType::I64),
                    span,
                ),
                int_lit(0, span),
                span,
            ));
        }
        let called = if after.is_empty() {
            called
        } else {
            let held = self.hold(
                Val {
                    node: called,
                    ty: Ty::Any,
                },
                &mut pre,
            );
            pre.extend(after);
            held.node
        };
        let v = self.guard_described(
            Val {
                node: called,
                ty: Ty::Any,
            },
            &descs,
        );
        Multi::Dynamic(block_value(pre, v.node, span))
    }

    /// `obj:name(args)`: `obj.name(obj, args)` with `obj` evaluated
    /// once; `desc` names `obj`.
    fn method_call(
        &mut self,
        obj: Val,
        mc: &ast::MethodCall,
        desc: Desc,
        span: Span,
    ) -> Result<Multi> {
        let name = ident(mc.name());
        let mut pre = Vec::new();
        let obj = self.hold(obj, &mut pre);
        // A string's methods are the string library's.
        if obj.ty == Ty::Str
            && let Some(b) = types::builtin_member(self.scopes(), "string", &name)
        {
            let multi = self.builtin_call(b, Some(obj), mc.args(), span)?;
            return Ok(self.prefixed(pre, multi, span));
        }
        let key = Val {
            node: str_lit(&name, span),
            ty: Ty::Str,
        };
        let method = Some(format!("method '{name}'"));
        // A method the types resolve to known functions is called
        // directly, whichever of them the receiver's class holds: told
        // by which lookup ends it when every table on its way has
        // slots, else by the code of the value found.
        if let Ty::Shape(k) = obj.ty
            && let Some(targets) = self.typer().method_targets(k, &name)
        {
            let funcs: Vec<FuncId> = targets
                .iter()
                .filter_map(|t| match t {
                    Ty::Func(f) => Some(*f),
                    _ => None,
                })
                .collect();
            if !funcs.is_empty() {
                let returns = self.typer().method_returns(Ty::Shape(k), &name).settled();
                if let Some(multi) = self.which_call(
                    obj.clone(),
                    k,
                    &name,
                    CallKind::Method,
                    returns.clone(),
                    mc.args(),
                    desc.clone(),
                    span,
                )? {
                    return Ok(self.prefixed(pre, multi, span));
                }
                let (arg_pre, vals, tail) = self.call_values(Some(obj.clone()), mc.args(), span)?;
                let callee = self.index_read(obj, key, desc, span);
                let multi = self.dispatch(
                    callee,
                    Guard::ByCode,
                    &funcs,
                    returns,
                    arg_pre,
                    vals,
                    tail,
                    method,
                    span,
                );
                return Ok(self.prefixed(pre, multi, span));
            }
        }
        let callee = self.index_read(obj.clone(), key, desc, span);
        let multi = self.value_call(callee, Some(obj), mc.args(), method, span)?;
        Ok(self.prefixed(pre, multi, span))
    }

    /// `obj:name(args)`, or `obj.name(args)` by `kind`, for a receiver
    /// of shape `k` whose lookups of `name` all end in slots holding
    /// known functions or in `__index` functions giving one: the
    /// receiver checked
    /// not nil, then the `Which` finder says which end the lookup
    /// takes, and that end's function is called directly. Everything
    /// the call goes through (the end, a closure's record, what an
    /// `__index` function gives) is found before the arguments run,
    /// as Lua binds the callee first. A lookup that ends nowhere
    /// raises. None when the lookups are not all of that kind; `desc`
    /// names `obj`.
    #[allow(clippy::too_many_arguments)]
    fn which_call(
        &mut self,
        obj: Val,
        k: ShapeId,
        name: &str,
        kind: CallKind,
        returns: Returns,
        args: &ast::FunctionArgs,
        desc: Desc,
        span: Span,
    ) -> Result<Option<Multi>> {
        let Some(which) = self.finder(k, name, FinderMode::Which, span) else {
            return Ok(None);
        };
        let direct = which.ends.iter().all(|(_, end)| match end {
            End::Slot(ty) => matches!(ty, Ty::Func(_)),
            End::Handler { ty, .. } => kind == CallKind::Method && matches!(ty, Ty::Func(_)),
        });
        if !direct {
            return Ok(None);
        }
        let takes_place = which.ends.iter().any(|(_, end)| match *end {
            End::Slot(Ty::Func(f)) => self.m.takes_env(f),
            End::Slot(_) => false,
            End::Handler { .. } => true,
        });
        let place = if takes_place {
            match self.finder(k, name, FinderMode::Where, span) {
                Some(place) => Some(place),
                None => return Ok(None),
            }
        } else {
            None
        };
        let table_t = self.ir(Ty::Table);
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let mut head = Vec::new();
        let obj = self.hold(obj, &mut head);
        // Indexing a nil receiver raises before anything is looked up.
        let mut raise = vec![
            self.set_line(span),
            expr_stmt(call("zl_index_nil", vec![], Type::Any, span)),
        ];
        raise.extend(self.leave_described(span, &Described::operand(desc)));
        head.push(if_(
            binary(
                BinaryOp::Eq,
                obj.node.clone(),
                null(table_t.clone(), span),
                bool_t.clone(),
                span,
            ),
            raise,
            None,
            span,
        ));
        let w = self.hold(
            Val {
                node: call(&which.helper, vec![obj.node.clone()], i64_t.clone(), span),
                ty: Ty::Int,
            },
            &mut head,
        );
        let place = place.map(|place| {
            self.hold(
                Val {
                    node: call(&place.helper, vec![obj.node.clone()], table_t.clone(), span),
                    ty: Ty::Table,
                },
                &mut head,
            )
        });
        let is_end = |i: usize| {
            binary(
                BinaryOp::Eq,
                w.node.clone(),
                int_lit(i as i64, span),
                bool_t.clone(),
                span,
            )
        };
        // What each end calls through: a closure's record read from its
        // slot, or what the `__index` function gives for the name.
        let mut through: Vec<Option<Val>> = Vec::with_capacity(which.ends.len());
        for (i, (shape, end)) in which.ends.iter().enumerate() {
            let found = match (*end, &place) {
                (End::Slot(Ty::Func(f)), Some(place)) if self.m.takes_env(f) => {
                    let (layout, slot) = self
                        .slot_of(*shape, name)
                        .expect("a finder's slot end has a slot");
                    let record = self.slot_read(&place.node, layout, &slot, span);
                    Some(Val {
                        node: if_value(is_end(i), record.node, nil(span), Type::Any, span),
                        ty: Ty::Any,
                    })
                }
                (End::Handler { class, f, ty }, Some(place)) => {
                    let (layout, slot) = self
                        .slot_of(class, "__index")
                        .expect("a finder's handler end has a slot");
                    let meta = field(place.node.clone(), "meta", table_t.clone(), span);
                    let record = self.slot_read(&meta, layout, &slot, span);
                    let args = vec![
                        Val {
                            node: place.node.clone(),
                            ty: Ty::Shape(*shape),
                        },
                        Val {
                            node: str_lit(name, span),
                            ty: Ty::Str,
                        },
                    ];
                    let multi = self.direct_call_vals(
                        f,
                        Some(record),
                        Vec::new(),
                        args,
                        None,
                        &Some("metamethod 'index'".to_string()),
                        span,
                    );
                    let ty = ty.settled();
                    let first = self.first_of(multi, span);
                    let first = self.coerce(first, ty);
                    let unread = self.zero_of(ty, span);
                    Some(Val {
                        node: if_value(is_end(i), first, unread, self.ir(ty), span),
                        ty,
                    })
                }
                _ => None,
            };
            through.push(found.map(|v| self.hold(v, &mut head)));
        }
        // Then the arguments, each held for the arm that takes it.
        let receiver = (kind == CallKind::Method).then(|| obj.clone());
        let (mut arg_pre, vals, tail) = self.call_values(receiver, args, span)?;
        head.append(&mut arg_pre);
        let vals: Vec<Val> = vals.into_iter().map(|v| self.hold(v, &mut head)).collect();
        let tail = tail.map(|t| {
            self.hold(
                Val {
                    node: t,
                    ty: Ty::Any,
                },
                &mut head,
            )
            .node
        });
        let shape = match &returns {
            Returns::Fixed(types) => Yield::Fixed(types.iter().map(|t| t.settled()).collect()),
            Returns::Dynamic | Returns::Unknown => Yield::Dynamic,
        };
        let raise = self.call_nil(name, kind, span);
        let mut value = self.yield_as(Multi::None(raise), &shape, span);
        let callee_desc = Some(match kind {
            CallKind::Method => format!("method '{name}'"),
            CallKind::Field => format!("field '{name}'"),
        });
        for (i, (_, end)) in which.ends.iter().enumerate().rev() {
            let arm = match *end {
                End::Slot(Ty::Func(f)) => {
                    let multi = self.direct_call_vals(
                        f,
                        through[i].clone(),
                        Vec::new(),
                        vals.clone(),
                        tail.clone(),
                        &callee_desc,
                        span,
                    );
                    self.yield_as(multi, &shape, span)
                }
                End::Handler {
                    ty: Ty::Func(g), ..
                } => {
                    let callee = through[i].clone().expect("a handler end's value is held");
                    self.handler_arm(callee, g, name, &shape, &vals, &tail, &callee_desc, span)
                }
                _ => unreachable!("a which call's ends hold functions"),
            };
            let ty = arm.ty.clone();
            value = if_value(is_end(i), arm, value, ty, span);
        }
        Ok(Some(self.yielded(
            block_value(head, value, span),
            &shape,
            span,
        )))
    }

    /// The arm of a method call whose lookup ends in an `__index`
    /// function, which gave `callee`, known to be the function `g` or
    /// nil: `g` called directly, or nil called, which raises.
    #[allow(clippy::too_many_arguments)]
    fn handler_arm(
        &mut self,
        callee: Val,
        g: FuncId,
        name: &str,
        shape: &Yield,
        vals: &[Val],
        tail: &Option<Node>,
        desc: &Desc,
        span: Span,
    ) -> Node {
        let multi = self.direct_call_vals(
            g,
            Some(callee.clone()),
            Vec::new(),
            vals.to_vec(),
            tail.clone(),
            desc,
            span,
        );
        let then = self.yield_as(multi, shape, span);
        let raise = self.call_nil(name, CallKind::Method, span);
        let otherwise = self.yield_as(Multi::None(raise), shape, span);
        let ty = then.ty.clone();
        let bound = binary(
            BinaryOp::Ne,
            callee.node,
            nil(span),
            prim(PrimitiveType::Bool),
            span,
        );
        if_value(bound, then, otherwise, ty, span)
    }

    /// Calling nil, found for `name`: the error, raised and left
    /// through.
    fn call_nil(&mut self, name: &str, kind: CallKind, span: Span) -> Node {
        let mut raise = vec![
            self.set_line(span),
            expr_stmt(call(
                "zl_raise_call_nil",
                vec![str_lit(name, span), int_lit(kind as i64, span)],
                prim(PrimitiveType::Unit),
                span,
            )),
        ];
        let leave = self.leave_described(span, &Described::NONE);
        raise.push(if_(bool_lit(true, span), leave, None, span));
        block_value(raise, nil(span), span)
    }

    /// A call through `callee`, a value the types say is one of
    /// `funcs`: each is called directly when the guard says it is the
    /// one, and anything else (nil, above all) is called through the
    /// value, which raises as Lua does. Every path yields the call's
    /// results the same way, typed as the call is. The arguments
    /// (`vals` and `tail`, with `pre` before them) run once.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &mut self,
        callee: Val,
        guard: Guard,
        funcs: &[FuncId],
        returns: Returns,
        mut pre: Vec<St>,
        vals: Vec<Val>,
        tail: Option<Node>,
        desc: Desc,
        span: Span,
    ) -> Multi {
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        // The callee is read before the arguments run.
        let mut head = Vec::new();
        let callee = self.hold(callee, &mut head);
        let code = match guard {
            Guard::ByCode => Some(self.hold(
                Val {
                    node: call("zl_func_id", vec![callee.node.clone()], i64_t.clone(), span),
                    ty: Ty::Int,
                },
                &mut head,
            )),
            Guard::NotNil => None,
        };
        head.append(&mut pre);
        let arms: Vec<(Node, FuncId, Val)> = funcs
            .iter()
            .map(|f| {
                let is_f = match &code {
                    Some(code) => binary(
                        BinaryOp::Eq,
                        code.node.clone(),
                        int_lit(self.m.func_key(*f), span),
                        bool_t.clone(),
                        span,
                    ),
                    None => binary(
                        BinaryOp::Ne,
                        callee.node.clone(),
                        nil(span),
                        bool_t.clone(),
                        span,
                    ),
                };
                (is_f, *f, callee.clone())
            })
            .collect();
        self.dispatch_arms(callee, arms, returns, head, vals, tail, desc, span)
    }

    /// The arms of a dispatch around its fallback: `arms` are guards,
    /// in order, each with the function it calls and the record the
    /// call goes through; `callee` (held) is what the fallback calls.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_arms(
        &mut self,
        callee: Val,
        arms: Vec<(Node, FuncId, Val)>,
        returns: Returns,
        mut pre: Vec<St>,
        vals: Vec<Val>,
        tail: Option<Node>,
        desc: Desc,
        span: Span,
    ) -> Multi {
        let vals: Vec<Val> = vals.into_iter().map(|v| self.hold(v, &mut pre)).collect();
        let tail = tail.map(|t| {
            self.hold(
                Val {
                    node: t,
                    ty: Ty::Any,
                },
                &mut pre,
            )
            .node
        });
        // What every path yields: the results as one value.
        let shape = match &returns {
            Returns::Fixed(types) => Yield::Fixed(types.iter().map(|t| t.settled()).collect()),
            Returns::Dynamic | Returns::Unknown => Yield::Dynamic,
        };
        // The fallback first, then each function's arm around it.
        let boxed_callee = self.boxed(callee.clone());
        let fallback = self.value_call_vals(
            boxed_callee,
            Vec::new(),
            vals.clone(),
            tail.clone(),
            desc.clone(),
            false,
            span,
        );
        let mut value = self.yield_as(fallback, &shape, span);
        for (is_f, f, record) in arms.into_iter().rev() {
            let arm = self.direct_call_vals(
                f,
                Some(record),
                Vec::new(),
                vals.clone(),
                tail.clone(),
                &desc,
                span,
            );
            let then = self.yield_as(arm, &shape, span);
            let ty = then.ty.clone();
            value = if_value(is_f, then, value, ty, span);
        }
        self.yielded(block_value(pre, value, span), &shape, span)
    }

    /// The results of `multi` as the one value a dispatch arm yields:
    /// the value itself for one result or a dynamic one, a list of the
    /// boxed results for several, nil for none.
    fn yield_as(&mut self, multi: Multi, shape: &Yield, span: Span) -> Node {
        match shape {
            Yield::Dynamic => match multi {
                Multi::Dynamic(node) => node,
                Multi::None(node) => block_value(vec![expr_stmt(node)], nil(span), span),
                Multi::Fixed(vals) => {
                    let mut pre = Vec::new();
                    let mut items = Vec::with_capacity(vals.len());
                    for v in vals {
                        items.push(self.boxed(v));
                    }
                    let list = self.array_of(items, &mut pre, span);
                    block_value(pre, call("zb_box_tuple", vec![list], Type::Any, span), span)
                }
            },
            Yield::Fixed(types) => {
                // The results, each as its type.
                let mut pre = Vec::new();
                let vals: Vec<Val> = match multi {
                    Multi::Fixed(vals) => {
                        let mut vals = vals.into_iter();
                        let mut out = Vec::with_capacity(types.len());
                        for ty in types {
                            let v = vals.next().unwrap_or_else(|| self.nil_val(span));
                            let v = Val {
                                node: self.coerce(v, *ty),
                                ty: *ty,
                            };
                            out.push(self.hold(v, &mut pre));
                        }
                        for v in vals {
                            pre.push(expr_stmt(v.node));
                        }
                        out
                    }
                    Multi::None(node) => {
                        pre.push(expr_stmt(node));
                        types
                            .iter()
                            .map(|ty| Val {
                                node: self.zero_of(*ty, span),
                                ty: *ty,
                            })
                            .collect()
                    }
                    Multi::Dynamic(node) => {
                        let list = self.temp();
                        pre.push(let_(
                            list,
                            self.m.anys(),
                            call("zl_values", vec![node], self.m.anys(), span),
                            span,
                        ));
                        types
                            .iter()
                            .enumerate()
                            .map(|(i, ty)| {
                                let element = Val {
                                    node: call(
                                        "zl_value_at",
                                        vec![
                                            var(list, self.m.anys(), span),
                                            int_lit(i as i64 + 1, span),
                                        ],
                                        Type::Any,
                                        span,
                                    ),
                                    ty: Ty::Any,
                                };
                                Val {
                                    node: self.coerce(element, *ty),
                                    ty: *ty,
                                }
                            })
                            .collect()
                    }
                };
                match types.len() {
                    0 => block_value(pre, nil(span), span),
                    1 => block_value(pre, vals.into_iter().next().unwrap().node, span),
                    _ => {
                        let mut items = Vec::with_capacity(vals.len());
                        for v in vals {
                            items.push(self.boxed(v));
                        }
                        let list = self.array_of(items, &mut pre, span);
                        block_value(pre, list, span)
                    }
                }
            }
        }
    }

    /// The one value a dispatch yielded, back as the call's results.
    fn yielded(&mut self, value: Node, shape: &Yield, span: Span) -> Multi {
        match shape {
            Yield::Dynamic => Multi::Dynamic(value),
            Yield::Fixed(types) => match types.len() {
                0 => Multi::None(value),
                1 => Multi::Fixed(vec![Val {
                    node: value,
                    ty: types[0],
                }]),
                _ => {
                    let name = self.temp();
                    let mut pre = vec![let_(name, self.m.anys(), value, span)];
                    let mut vals = Vec::with_capacity(types.len());
                    for (i, ty) in types.iter().enumerate() {
                        let element = Val {
                            node: index(
                                var(name, self.m.anys(), span),
                                int_lit(i as i64, span),
                                Type::Any,
                                span,
                            ),
                            ty: Ty::Any,
                        };
                        let read = self.coerce(element, *ty);
                        let read = if i == 0 {
                            block_value(std::mem::take(&mut pre), read, span)
                        } else {
                            read
                        };
                        vals.push(Val {
                            node: read,
                            ty: *ty,
                        });
                    }
                    Multi::Fixed(vals)
                }
            },
        }
    }

    /// Statements run before several values.
    fn prefixed(&mut self, pre: Vec<St>, multi: Multi, span: Span) -> Multi {
        if pre.is_empty() {
            return multi;
        }
        match multi {
            Multi::Dynamic(node) => Multi::Dynamic(block_value(pre, node, span)),
            Multi::None(node) => Multi::None(block_value(pre, node, span)),
            Multi::Fixed(mut vals) => {
                if vals.is_empty() {
                    Multi::Fixed(vec![Val {
                        node: block_value(pre, nil(span), span),
                        ty: Ty::Nil,
                    }])
                } else {
                    let first = vals.remove(0);
                    let mut out = vec![Val {
                        node: block_value(pre, first.node, span),
                        ty: first.ty,
                    }];
                    out.extend(vals);
                    Multi::Fixed(out)
                }
            }
        }
    }

    /// A call to a library function: each argument as the
    /// implementation takes it.
    fn builtin_call(
        &mut self,
        b: &Builtin,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        span: Span,
    ) -> Result<Multi> {
        // `rawget(_G, "x")` and `rawset(_G, "x", v)` are the global.
        if b.lib.is_empty() && (b.name == "rawget" || b.name == "rawset") {
            let exprs = self.args_exprs(args);
            if let Some(Expression::Var(Var::Name(token))) = exprs.first() {
                let is_g = matches!(self.scopes().binding(token), Some(Binding::Global(g)) if Scopes::is_globals_name(g))
                    && !self.scopes().global_writes.contains_key(&ident(token))
                    && !self.scopes().dynamic_globals;
                if is_g {
                    let name = exprs.get(1).and_then(|e| crate::scope::literal_string(e));
                    let Some(name) = name else {
                        return unsupported("`_G` indexed by anything but a string literal", span);
                    };
                    if b.name == "rawget" {
                        let v = self.read_global(&name, span)?;
                        return Ok(Multi::Fixed(vec![v]));
                    }
                    let Some(value) = exprs.get(2) else {
                        return unsupported("`rawset(_G, name)` without a value", span);
                    };
                    let v = self.expr(value)?;
                    let st = self.write_global(&name, v, span)?;
                    return Ok(Multi::None(block_value(vec![st], nil(span), span)));
                }
            }
        }
        // `setmetatable` of a table a constructor is making: the table
        // is born with its metatable.
        if b.lib.is_empty()
            && b.name == "setmetatable"
            && receiver.is_none()
            && let [first, mt] = self.args_exprs(args).as_slice()
            && let Expression::TableConstructor(t) = first
            && let Some(k) = types::constructor_shape(self.m.inferred, t)
            && self.m.layout(k).is_some()
        {
            let v = self.table_constructor_with(t, Some(mt), span_of(first))?;
            return Ok(Multi::Fixed(vec![v]));
        }
        let is_method = receiver.is_some();
        // `error(v, level)` above level 1 positions at the caller: the
        // line this function was entered at.
        if b.lib.is_empty() && b.name == "error" {
            let exprs = self.args_exprs(args);
            let level = exprs.get(1).and_then(|e| match e {
                Expression::Number(n) => {
                    match crate::host::parse_numeral(n.token().to_string().trim()) {
                        crate::host::Numeral::Int(v) => Some(v),
                        _ => None,
                    }
                }
                _ => None,
            });
            if level.is_some_and(|l| l >= 2) {
                self.entry_line = true;
                let (mut pre, vals, _) = self.call_values(None, args, span)?;
                let v = vals.into_iter().next().map(|v| self.boxed(v));
                let v = v.unwrap_or_else(|| nil(span));
                let call = call(
                    "zl_error_at",
                    vec![v, var(intern(ENTRY_LINE), prim(PrimitiveType::I64), span)],
                    prim(PrimitiveType::Unit),
                    span,
                );
                let after = self.around_builtin(b, is_method, span, &mut pre);
                pre.push(expr_stmt(call));
                pre.extend(after);
                pre.push(self.pending_check(span));
                return Ok(Multi::None(block_value(pre, nil(span), span)));
            }
        }
        let (mut pre, vals, tail) = self.call_values(receiver, args, span)?;
        let mut vals: Vec<Val> = vals.into_iter().map(|v| self.hold(v, &mut pre)).collect();
        if tail.is_none()
            && let Some(m) = self.math_call(b, &mut vals, &mut pre, span)
        {
            return Ok(m);
        }
        // `tostring` of a string or a number is its text, made as `..`
        // makes it.
        if b.lib.is_empty()
            && b.name == "tostring"
            && tail.is_none()
            && vals.len() == 1
            && matches!(vals[0].ty, Ty::Str | Ty::Int | Ty::Float | Ty::Number)
        {
            let text = self.text_of(vals.remove(0));
            return Ok(Multi::Fixed(vec![Val {
                node: block_value(pre, text, span),
                ty: Ty::Str,
            }]));
        }
        // `table.sort` of a table with no metatable by a function the
        // types know: a sort of its own, calling the comparator
        // directly with the elements as their type.
        if b.lib == "table"
            && b.name == "sort"
            && tail.is_none()
            && vals.len() == 2
            && let Ty::Shape(k) = vals[0].ty
            && let Ty::Func(f) = vals[1].ty
            && self.m.plain_shape(k)
        {
            let helper = self.sort_helper(k, f, span);
            self.raise_callees.insert(f);
            let table_t = self.ir(Ty::Table);
            let bool_t = prim(PrimitiveType::Bool);
            let i64_t = prim(PrimitiveType::I64);
            let unit = prim(PrimitiveType::Unit);
            let comp = vals.pop().expect("the comparator");
            let t = vals.pop().expect("the table");
            let t_null = binary(
                BinaryOp::Eq,
                t.node.clone(),
                null(table_t, span),
                bool_t.clone(),
                span,
            );
            let comp_nil = binary(
                BinaryOp::Eq,
                comp.node.clone(),
                nil(span),
                bool_t.clone(),
                span,
            );
            let boxed_t = self.boxed(t.clone());
            let general = call(
                "zl_table_sort",
                vec![boxed_t, comp.node.clone()],
                unit.clone(),
                span,
            );
            let arr = self.temp();
            let typed = vec![
                let_(
                    arr,
                    self.m.anys(),
                    call("zl_arr_own", vec![t.node], self.m.anys(), span),
                    span,
                ),
                expr_stmt(call(
                    &helper,
                    vec![
                        var(arr, self.m.anys(), span),
                        int_lit(0, span),
                        binary(
                            BinaryOp::Sub,
                            list_len(var(arr, self.m.anys(), span), span),
                            int_lit(1, span),
                            i64_t.clone(),
                            span,
                        ),
                        var(intern(library::LINE), i64_t, span),
                        comp.node,
                    ],
                    unit,
                    span,
                )),
            ];
            // A nil table raises through the general path; a nil
            // comparator sorts by `<` there too.
            pre.push(if_(
                binary(BinaryOp::Or, t_null, comp_nil, bool_t, span),
                vec![expr_stmt(general)],
                Some(typed),
                span,
            ));
            pre.push(self.pending_check(span));
            return Ok(Multi::None(block_value(pre, nil(span), span)));
        }
        // `setmetatable` of tables the types know: neither is boxed.
        if b.lib.is_empty()
            && b.name == "setmetatable"
            && tail.is_none()
            && vals.len() == 2
            && vals
                .iter()
                .all(|v| matches!(v.ty, Ty::Table | Ty::Shape(_)))
        {
            let table_t = self.ir(Ty::Table);
            let ty = vals[0].ty;
            let node = call(
                "zl_setmetatable_tables",
                vals.into_iter().map(|v| v.node).collect(),
                table_t,
                span,
            );
            let v = self.guard(Val {
                node,
                ty: Ty::Table,
            });
            return Ok(Multi::Fixed(vec![Val {
                node: block_value(pre, v.node, span),
                ty,
            }]));
        }
        // A tail of several values fills what follows through a list.
        let tail_list = match tail {
            Some(tail) => {
                let name = self.temp();
                pre.push(let_(
                    name,
                    self.m.anys(),
                    call("zl_values", vec![tail], self.m.anys(), span),
                    span,
                ));
                Some(name)
            }
            None => None,
        };
        let mut lowered = Vec::with_capacity(b.params.len());
        let mut any_bytes = false;
        let mut consumed = 0;
        for (i, p) in b.params.iter().enumerate() {
            if let Param::Rest = p {
                // Everything left, boxed, plus the tail.
                let mut items = Vec::new();
                for v in vals.drain(..) {
                    items.push(self.boxed(v));
                }
                let list = self.array_of(items, &mut pre, span);
                // The tail's values the parameters before took are
                // not the rest.
                let list = match tail_list {
                    Some(name) => {
                        let list_name = self.temp();
                        pre.push(let_(list_name, self.m.anys(), list, span));
                        let tail = var(name, self.m.anys(), span);
                        let rest = if consumed == 0 {
                            tail
                        } else {
                            call(
                                "zl_slice",
                                vec![
                                    tail.clone(),
                                    int_lit(consumed as i64, span),
                                    list_len(tail, span),
                                ],
                                self.m.anys(),
                                span,
                            )
                        };
                        pre.push(expr_stmt(call(
                            "zb_list_extend_any",
                            vec![var(list_name, self.m.anys(), span), rest],
                            prim(PrimitiveType::Unit),
                            span,
                        )));
                        var(list_name, self.m.anys(), span)
                    }
                    None => list,
                };
                lowered.push(list);
                consumed = usize::MAX;
                break;
            }
            let v = if !vals.is_empty() {
                Some(vals.remove(0))
            } else if let Some(name) = tail_list {
                consumed += 1;
                Some(Val {
                    node: call(
                        "zl_value_at",
                        vec![
                            var(name, self.m.anys(), span),
                            int_lit(consumed as i64, span),
                        ],
                        Type::Any,
                        span,
                    ),
                    ty: Ty::Any,
                })
            } else {
                None
            };
            // A method's receiver is not counted, and is not "bad
            // argument #0".
            let what = match (is_method, i) {
                (true, 0) => format!("calling '{}' on bad self", b.name),
                (true, i) => format!("bad argument #{i} to '{}'", b.name),
                (false, i) => format!("bad argument #{} to '{}'", i + 1, b.name),
            };
            let what = str_lit(&what, span);
            let arg = self.builtin_arg(p, v, what, span);
            any_bytes |= *p == Param::Bytes && arg.ty == Type::Any;
            lowered.push(arg);
        }
        // Arguments past the parameters run for their effects.
        if consumed != usize::MAX {
            for v in vals {
                pre.push(expr_stmt(v.node));
            }
        }
        let ret_ir = crate::library::stdlib::ret_type(b.ret, &self.m.types);
        // An argument's conversion can raise as well as the function;
        // it is checked before the call, which might otherwise take the
        // error as its own (`pcall`), each argument bound in order.
        if lowered.iter().any(|a| self.call_can_raise(a)) {
            for a in lowered.iter_mut() {
                let raises = self.call_can_raise(a);
                let name = self.temp();
                let ty = a.ty.clone();
                let node = std::mem::replace(a, var(name, ty.clone(), span));
                pre.push(let_(name, ty, node, span));
                if raises {
                    pre.push(self.pending_check(span));
                }
            }
        }
        let func = if any_bytes {
            crate::library::stdlib::any_form(b.func)
        } else {
            b.func.to_string()
        };
        let value = call(&func, lowered, ret_ir, span);
        let raises = self.call_can_raise(&value);
        let ty = match b.ret {
            Ret::Unit => Ty::Nil,
            Ret::Multi => Ty::Any,
            r => types::ret_ty(r),
        };
        let after = self.around_builtin(b, is_method, span, &mut pre);
        let v = if after.is_empty() {
            Val { node: value, ty }
        } else if b.ret == Ret::Unit {
            pre.push(expr_stmt(value));
            pre.extend(after);
            if raises {
                pre.push(self.pending_check(span));
            }
            return Ok(Multi::None(block_value(pre, nil(span), span)));
        } else {
            let held = self.hold(Val { node: value, ty }, &mut pre);
            pre.extend(after);
            held
        };
        let v = if raises { self.guard(v) } else { v };
        Ok(match b.ret {
            Ret::Unit => Multi::None(block_value(pre, v.node, span)),
            Ret::Multi => Multi::Dynamic(block_value(pre, v.node, span)),
            _ => Multi::Fixed(vec![Val {
                node: block_value(pre, v.node, span),
                ty: v.ty,
            }]),
        })
    }

    /// What a call of a library function does around it in a function
    /// that keeps a frame: one that may call the program back is a
    /// frame of its own, named as the reference names it; before one
    /// that reads the caller's locals, or calls something that may,
    /// the locals are spilled. Pushes what comes before the call onto
    /// `pre` and returns what comes after it.
    fn around_builtin(
        &mut self,
        b: &Builtin,
        is_method: bool,
        span: Span,
        pre: &mut Vec<St>,
    ) -> Vec<St> {
        if !self.frames {
            return Vec::new();
        }
        // The frame is at the line the call starts on while it runs.
        pre.push(self.set_line(span));
        // The debug library's own functions are the level 0 it
        // describes, not a frame of the stack. `error` is the level an
        // error is raised at.
        let own_frame = (self.m.reentrant.contains(b.func) && b.lib != "debug")
            || (b.lib.is_empty() && b.name == "error");
        let reads_frame = b.lib == "debug" && matches!(b.name, "getlocal" | "setlocal");
        if !own_frame && !reads_frame {
            return Vec::new();
        }
        let global = if b.lib.is_empty() {
            b.name.to_string()
        } else {
            format!("{}.{}", b.lib, b.name)
        };
        let namewhat = if is_method {
            "method"
        } else if b.lib.is_empty() {
            "global"
        } else {
            "field"
        };
        let desc = Some(format!("{namewhat} '{}'", b.name));
        let Some(site) = self.debug_site_as(&desc, &global, span) else {
            return Vec::new();
        };
        let spill = self.spill(site, span, pre);
        let mut after = Vec::new();
        if own_frame {
            pre.push(expr_stmt(call(
                "zl_dbg_enter_c",
                vec![int_lit(site, span)],
                prim(PrimitiveType::Unit),
                span,
            )));
            after.push(expr_stmt(call(
                "zl_dbg_leave",
                vec![],
                prim(PrimitiveType::Unit),
                span,
            )));
        }
        after.extend(self.after_call(spill, span));
        after
    }

    /// A `math` call whose arguments' types decide its result, as the
    /// typer decided it too: the arithmetic inline, nothing boxed on
    /// the way in or out. `None` leaves the call to the library.
    fn math_call(
        &mut self,
        b: &Builtin,
        vals: &mut Vec<Val>,
        pre: &mut Vec<St>,
        span: Span,
    ) -> Option<Multi> {
        if b.lib != "math" {
            return None;
        }
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let bool_t = prim(PrimitiveType::Bool);
        let tys: Vec<Ty> = vals.iter().map(|v| v.ty).collect();
        let ty = match types::math_result(b, &tys) {
            Some(t @ (Ty::Int | Ty::Float | Ty::Number)) => t,
            _ => return None,
        };
        let ir = self.ir(ty);
        let mut args = std::mem::take(vals).into_iter();
        let mut arg = || args.next().expect("an argument");
        let node = match (b.name, ty) {
            ("abs", Ty::Int) => {
                let x = arg().node;
                let negative = binary(BinaryOp::Lt, x.clone(), int_lit(0, span), bool_t, span);
                let negated = binary(BinaryOp::Sub, int_lit(0, span), x.clone(), i64_t, span);
                if_value(negative, negated, x, ir, span)
            }
            ("abs", Ty::Float) => call("abs", vec![arg().node], f64_t, span),
            // Each part's own absolute value: the part the kind does
            // not use stays zero.
            ("abs", Ty::Number) => {
                let (mut held, n) = self.number_parts(arg());
                pre.append(&mut held);
                let negative = binary(BinaryOp::Lt, n.int.clone(), int_lit(0, span), bool_t, span);
                let negated = binary(
                    BinaryOp::Sub,
                    int_lit(0, span),
                    n.int.clone(),
                    i64_t.clone(),
                    span,
                );
                let int = if_value(negative, negated, n.int, i64_t, span);
                let float = call("abs", vec![n.float], f64_t, span);
                number_value(n.tag, int, float, span)
            }
            ("floor" | "ceil", Ty::Int) => arg().node,
            // A float rounds to the integer it makes when that fits,
            // else stays a float; an integer is itself.
            ("floor" | "ceil", Ty::Number) => {
                let x = arg();
                let round = if b.name == "floor" {
                    "floor"
                } else {
                    "zl_ceil_f64"
                };
                if x.ty == Ty::Float {
                    let r = call(round, vec![x.node], f64_t, span);
                    self.number_of_integral(r, pre, span)
                } else {
                    let (mut held, n) = self.number_parts(x);
                    pre.append(&mut held);
                    let r = call(round, vec![n.float.clone()], f64_t, span);
                    let mut inner = Vec::new();
                    let rounded = self.number_of_integral(r, &mut inner, span);
                    let rounded = block_value(inner, rounded, span);
                    if_value(
                        n.is_int(span),
                        number_value(n.tag.clone(), n.int, n.float, span),
                        rounded,
                        ir,
                        span,
                    )
                }
            }
            ("max" | "min", _) => {
                // The best so far gives way when it is less than the
                // next (`max`), or the next is less (`min`): compared
                // with a NaN neither is, so the first stays.
                let mut best = arg();
                if ty == Ty::Number {
                    let node = self.coerce(best, ty);
                    best = self.hold(Val { node, ty }, pre);
                }
                for next in args {
                    let next = if ty == Ty::Number {
                        let node = self.coerce(next, ty);
                        self.hold(Val { node, ty }, pre)
                    } else {
                        next
                    };
                    let (l, r) = if b.name == "max" {
                        (best.clone(), next.clone())
                    } else {
                        (next.clone(), best.clone())
                    };
                    let moves = if ty == Ty::Number {
                        self.number_compare(l, r, Compare::Lt, span)
                    } else {
                        binary(BinaryOp::Lt, l.node, r.node, bool_t.clone(), span)
                    };
                    let picked = Val {
                        node: if_value(moves, next.node, best.node, ir.clone(), span),
                        ty,
                    };
                    best = self.hold(picked, pre);
                }
                best.node
            }
            ("fmod", Ty::Int) => {
                let (a, b) = (arg().node, arg().node);
                call("zl_fmod_i64", vec![a, b], i64_t, span)
            }
            ("fmod", Ty::Float) => {
                let (a, b) = (arg(), arg());
                let a = self.coerce(a, Ty::Float);
                let b = self.coerce(b, Ty::Float);
                binary(BinaryOp::Rem, a, b, f64_t, span)
            }
            _ => unreachable!("math_result decided the shape"),
        };
        let v = Val { node, ty };
        let v = if self.call_can_raise(&v.node) {
            self.guard(v)
        } else {
            v
        };
        let node = block_value(std::mem::take(pre), v.node, span);
        Some(Multi::Fixed(vec![Val { node, ty }]))
    }

    /// An integral float as a number: the integer when it fits, else
    /// the float; `pre` takes the float, held.
    fn number_of_integral(&mut self, r: Node, pre: &mut Vec<St>, span: Span) -> Node {
        let f64_t = prim(PrimitiveType::F64);
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let r = self
            .hold(
                Val {
                    node: r,
                    ty: Ty::Float,
                },
                pre,
            )
            .node;
        // `-2^63 <= r < 2^63`, false for a NaN.
        let fits = binary(
            BinaryOp::And,
            binary(
                BinaryOp::Ge,
                r.clone(),
                float_lit(-9223372036854775808.0, span),
                bool_t.clone(),
                span,
            ),
            binary(
                BinaryOp::Lt,
                r.clone(),
                float_lit(9223372036854775808.0, span),
                bool_t.clone(),
                span,
            ),
            bool_t,
            span,
        );
        let fits = self
            .hold(
                Val {
                    node: fits,
                    ty: Ty::Bool,
                },
                pre,
            )
            .node;
        let int = if_value(
            fits.clone(),
            cast(r.clone(), i64_t.clone(), span),
            int_lit(0, span),
            i64_t,
            span,
        );
        let float = if_value(fits.clone(), float_lit(0.0, span), r, f64_t, span);
        number_value(tag_of_is_int(fits, span), int, float, span)
    }

    /// One argument of a builtin as its parameter takes it.
    fn builtin_arg(&mut self, p: &Param, v: Option<Val>, what: Node, span: Span) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let str_t = prim(PrimitiveType::String);
        match p {
            Param::Any => match v {
                Some(v) => self.boxed(v),
                None => nil(span),
            },
            Param::Value => match v {
                Some(v) => self.boxed(v),
                None => call("zl_arg_value_missing", vec![what], Type::Any, span),
            },
            Param::Expected(kind) => match v {
                Some(v) => self.boxed(v),
                None => call(
                    "zl_arg_expected_missing",
                    vec![what, str_lit(kind, span)],
                    Type::Any,
                    span,
                ),
            },
            Param::Int => match v {
                Some(v) if v.ty == Ty::Int => v.node,
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_int", vec![b, what], i64_t, span)
                }
                None => call("zl_arg_int_missing", vec![what], i64_t, span),
            },
            Param::Float => match v {
                Some(v) if v.ty.is_number() => self.coerce(v, Ty::Float),
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_float", vec![b, what], f64_t, span)
                }
                None => call("zl_arg_float_missing", vec![what], f64_t, span),
            },
            Param::Str => match v {
                Some(v) if v.ty == Ty::Str => v.node,
                Some(v) if v.ty.is_number() => self.text_of(v),
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_str", vec![b, what], str_t, span)
                }
                None => call("zl_arg_str_missing", vec![what], str_t, span),
            },
            // A string by its type is the typed implementation's; any
            // other value, possibly a buffer, is its `$any` form's.
            Param::Bytes => match v {
                Some(v) if v.ty == Ty::Str => v.node,
                Some(v) if v.ty.is_number() => self.text_of(v),
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_bytes", vec![b, what], Type::Any, span)
                }
                None => call("zl_arg_str_missing", vec![what], str_t, span),
            },
            Param::Table => match v {
                Some(v) if v.ty == Ty::Table => v.node,
                // A shaped table is one unless it is nil, when the
                // check raises as for any other value.
                Some(v) if matches!(v.ty, Ty::Shape(_)) => {
                    let table_t = self.ir(Ty::Table);
                    let mut pre = Vec::new();
                    let t = self.hold(v, &mut pre);
                    let is_null = binary(
                        BinaryOp::Eq,
                        t.node.clone(),
                        null(table_t.clone(), span),
                        prim(PrimitiveType::Bool),
                        span,
                    );
                    block_value(
                        pre,
                        if_value(
                            is_null,
                            call("zl_as_table", vec![nil(span), what], table_t.clone(), span),
                            t.node,
                            table_t,
                            span,
                        ),
                        span,
                    )
                }
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_as_table", vec![b, what], self.ir(Ty::Table), span)
                }
                None => call("zl_as_table_missing", vec![what], self.ir(Ty::Table), span),
            },
            Param::OptInt(d) => match v {
                Some(v) if v.ty == Ty::Int => v.node,
                Some(v) if v.ty == Ty::Nil => {
                    block_value(vec![expr_stmt(v.node)], int_lit(*d, span), span)
                }
                Some(v) => {
                    let b = self.boxed(v);
                    call(
                        "zl_arg_opt_int",
                        vec![b, int_lit(*d, span), what],
                        i64_t,
                        span,
                    )
                }
                None => int_lit(*d, span),
            },
            Param::OptStr(d) => match v {
                Some(v) if v.ty == Ty::Str => v.node,
                Some(v) if v.ty == Ty::Nil => {
                    block_value(vec![expr_stmt(v.node)], str_lit(d, span), span)
                }
                Some(v) => {
                    let b = self.boxed(v);
                    call(
                        "zl_arg_opt_str",
                        vec![b, str_lit(d, span), what],
                        str_t,
                        span,
                    )
                }
                None => str_lit(d, span),
            },
            Param::Rest => unreachable!("the rest is gathered by the caller"),
        }
    }

    // ─── statements ─────────────────────────────────────────────

    /// The chunk's outermost block as segments of at most
    /// `SEGMENT_STATEMENTS` statements each, the final `return` in the
    /// last.
    fn segments(&mut self, block: &Block) -> Result<Vec<Vec<St>>> {
        self.depth = 1;
        let mut segments = Vec::new();
        let mut out = Vec::new();
        for s in block.stmts() {
            let at = out.len();
            self.line_needed = false;
            self.stmt(s, &mut out)?;
            self.record_stmt_line(Some(s), span_of(s), at, &mut out);
            if out.len() >= crate::scope::SEGMENT_STATEMENTS {
                segments.push(std::mem::take(&mut out));
            }
        }
        if let Some(last) = block.last_stmt() {
            let span = span_of(last);
            let at = out.len();
            self.line_needed = false;
            match last {
                ast::LastStmt::Return(r) => {
                    let exprs: Vec<&Expression> = r.returns().iter().collect();
                    self.return_stmt(&exprs, span, &mut out)?;
                }
                _ => return unsupported("this statement", span),
            }
            self.record_stmt_line(None, span, at, &mut out);
        } else {
            let span = span_of(block);
            self.closes_checked(
                1,
                true,
                Span::new(span.end.saturating_sub(1), span.end),
                &mut out,
            );
        }
        if !out.is_empty() || segments.is_empty() {
            segments.push(out);
        }
        Ok(segments)
    }

    fn block(&mut self, block: &Block) -> Result<Vec<St>> {
        self.depth += 1;
        let live = self.live.len();
        let mut out = Vec::new();
        for s in block.stmts() {
            let at = out.len();
            self.line_needed = false;
            self.stmt(s, &mut out)?;
            self.record_stmt_line(Some(s), span_of(s), at, &mut out);
        }
        match block.last_stmt() {
            Some(last) => {
                let span = span_of(last);
                let at = out.len();
                self.line_needed = false;
                match last {
                    ast::LastStmt::Break(_) => {
                        // The loop's body and everything within it is left.
                        let (loop_depth, end) =
                            self.loop_depths.last().copied().unwrap_or((1, span));
                        self.closes_checked(loop_depth, false, end, &mut out);
                        out.push(stmt(TypedStatement::Break(None), span));
                    }
                    ast::LastStmt::Return(r) => {
                        let exprs: Vec<&Expression> = r.returns().iter().collect();
                        self.return_stmt(&exprs, span, &mut out)?;
                    }
                    _ => return unsupported("this statement", span),
                }
                self.record_stmt_line(None, span, at, &mut out);
            }
            None => {
                // A function's body closes at its `end`, any other
                // block at its last token.
                let end = match self.end_span.filter(|_| self.depth == 1) {
                    Some(end) => end,
                    None => {
                        let span = span_of(block);
                        Span::new(span.end.saturating_sub(1), span.end)
                    }
                };
                self.closes_checked(self.depth, false, end, &mut out);
            }
        }
        let depth = self.depth;
        self.tbc
            .retain(|(d, c)| *d < depth || (*d == depth && matches!(c, Closable::Loop(_))));
        self.depth -= 1;
        self.live.truncate(live);
        Ok(out)
    }

    /// What leaving to `depth` closes, innermost first: everything
    /// entered at `depth` or deeper, a loop's closing value at `depth`
    /// itself only when `loops` (the jump leaves that loop).
    fn closing(&self, depth: usize, loops: bool) -> Vec<(usize, Closable)> {
        self.tbc
            .iter()
            .rev()
            .filter(|(d, c)| {
                *d > depth || (*d == depth && (loops || matches!(c, Closable::Var(_))))
            })
            .copied()
            .collect()
    }

    /// Close everything entered at `depth` or deeper, innermost first.
    /// Each handler sees the error pending, if any. The entries stay in
    /// scope for the paths that go on.
    fn closes_from(&mut self, depth: usize, span: Span, out: &mut Vec<St>) {
        let entries = self.closing(depth, true);
        self.emit_closes(&entries, span, out);
    }

    fn emit_closes(&mut self, entries: &[(usize, Closable)], span: Span, out: &mut Vec<St>) {
        if !entries.is_empty() {
            // A handler may raise, and its error is this function's.
            self.raised = true;
        }
        for (_, c) in entries {
            let value = match *c {
                Closable::Var(v) => {
                    let value = self.read_var(v, span);
                    self.boxed(value)
                }
                Closable::Loop(name) => var(name, Type::Any, span),
            };
            out.push(expr_stmt(call(
                "zl_close",
                vec![value],
                prim(PrimitiveType::Unit),
                span,
            )));
        }
    }

    /// The closes of a jump or a block's end, on the line of `at`. When
    /// a handler raised, the error then leaves at once, closing only
    /// what is still open.
    fn closes_checked(&mut self, depth: usize, loops: bool, at: Span, out: &mut Vec<St>) {
        let entries = self.closing(depth, loops);
        if entries.is_empty() {
            return;
        }
        out.push(self.set_line(at));
        self.emit_closes(&entries, at, out);
        let scope = self.tbc.clone();
        self.tbc.retain(|e| !entries.contains(e));
        let check = self.pending_check(at);
        self.tbc = scope;
        out.push(check);
    }

    /// The line a statement starts, stored ahead of it. A function
    /// that keeps a frame stores every statement's line, the one its
    /// code runs on (a function definition's is its `end`'s), and
    /// tells the hooks; a block's own line runs nothing.
    fn record_stmt_line(&mut self, s: Option<&Stmt>, span: Span, at: usize, out: &mut Vec<St>) {
        if !self.frames {
            self.record_line(span, at, out);
            return;
        }
        let line = match s {
            Some(Stmt::Do(_) | Stmt::Repeat(_)) => {
                self.line_needed = false;
                return;
            }
            Some(Stmt::LocalFunction(f)) => self.m.line_of(span_of(f.body().end_token())),
            Some(Stmt::FunctionDeclaration(f)) => self.m.line_of(span_of(f.body().end_token())),
            _ => self.m.line_of(span),
        };
        self.line_needed = false;
        let statements = self.debug_line(line, span);
        out.splice(at..at, statements);
    }

    /// A statement that checks for an error stores its line first, for
    /// the position the error's message carries.
    fn record_line(&mut self, span: Span, at: usize, out: &mut Vec<St>) {
        if !self.line_needed {
            return;
        }
        self.line_needed = false;
        let st = self.set_line(span);
        out.insert(at, st);
    }

    fn return_stmt(&mut self, exprs: &[&Expression], span: Span, out: &mut Vec<St>) -> Result<()> {
        // `return f(...)` is a tail call, unless something is to be
        // closed after it.
        let tail = match exprs {
            _ if !self.tbc.is_empty() => None,
            [e @ Expression::FunctionCall(_)] => Some(span_of(*e)),
            [e @ Expression::Var(Var::Expression(v))]
                if matches!(v.suffixes().last(), Some(Suffix::Call(_))) =>
            {
                Some(span_of(*e))
            }
            _ => None,
        };
        let outer_tail = std::mem::replace(&mut self.tail_span, tail);
        let values = self.return_values(exprs, span, out);
        self.tail_span = outer_tail;
        values?;
        if self.tbc.is_empty() && !self.counts_depth && !self.frames {
            return Ok(());
        }
        // The values are computed before anything is closed.
        let Some(last) = out.pop() else {
            return Ok(());
        };
        let TypedStatement::Return(value) = last.node else {
            out.push(last);
            return Ok(());
        };
        let value = value.map(|value| {
            let name = self.temp();
            let ty = value.ty.clone();
            out.push(let_(name, ty.clone(), *value, span));
            var(name, ty, span)
        });
        self.closes_from(1, span, out);
        if self.counts_depth {
            out.push(depth_step(-1, span));
        }
        out.extend(self.debug_call("zl_dbg_leave", Vec::new(), span));
        out.push(ret(value, span));
        Ok(())
    }

    fn return_values(
        &mut self,
        exprs: &[&Expression],
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let returns = self.returns.clone();
        match returns {
            Returns::Fixed(types) if types.is_empty() => {
                for e in exprs {
                    let v = self.expr(e)?;
                    out.push(expr_stmt(v.node));
                }
                if self.func == CHUNK && self.scopes().split_chunk {
                    let flag = intern(RETURNED);
                    self.m.declare_module_var(flag, Ty::Bool);
                    out.push(assign(
                        var(flag, prim(PrimitiveType::Bool), span),
                        bool_lit(true, span),
                        span,
                    ));
                }
                out.push(ret(None, span));
            }
            Returns::Fixed(types) if types.len() == 1 => {
                let (pre, mut vals) = self.adjusted(exprs, 1, span)?;
                out.extend(pre);
                let v = vals.pop().unwrap();
                let value = self.coerce(v, types[0].settled());
                out.push(ret(Some(value), span));
            }
            Returns::Fixed(types) => {
                let (pre, vals) = self.adjusted(exprs, types.len(), span)?;
                out.extend(pre);
                let mut items = Vec::with_capacity(vals.len());
                for v in vals {
                    items.push(self.boxed(v));
                }
                let mut pre = Vec::new();
                let list = self.array_of(items, &mut pre, span);
                out.extend(pre);
                out.push(ret(Some(list), span));
            }
            Returns::Dynamic | Returns::Unknown => {
                let (pre, vals, tail) = self.expr_list(exprs)?;
                out.extend(pre);
                let value = match (vals.len(), tail) {
                    (1, None) => {
                        let v = vals.into_iter().next().unwrap();
                        self.boxed(v)
                    }
                    (0, Some(tail)) => tail,
                    (_, tail) => {
                        let mut items = Vec::with_capacity(vals.len());
                        for v in vals {
                            items.push(self.boxed(v));
                        }
                        let mut pre = Vec::new();
                        let list = self.array_of(items, &mut pre, span);
                        out.extend(pre);
                        match tail {
                            None => call("zl_pack", vec![list], Type::Any, span),
                            Some(tail) => {
                                let name = self.temp();
                                let pre = vec![
                                    let_(name, self.m.anys(), list, span),
                                    expr_stmt(call(
                                        "zl_append_values",
                                        vec![var(name, self.m.anys(), span), tail],
                                        prim(PrimitiveType::Unit),
                                        span,
                                    )),
                                ];
                                block_value(
                                    pre,
                                    call(
                                        "zl_pack",
                                        vec![var(name, self.m.anys(), span)],
                                        Type::Any,
                                        span,
                                    ),
                                    span,
                                )
                            }
                        }
                    }
                };
                out.push(ret(Some(value), span));
            }
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt, out: &mut Vec<St>) -> Result<()> {
        self.stmt_of(s, out)?;
        // The locals a declaration makes are in scope from the next
        // statement on.
        match s {
            Stmt::LocalAssignment(l) => {
                for name in l.names() {
                    let v = self.scopes().declared(name);
                    if !self.scopes().var(v).folded {
                        self.live.push(Live::Var(v));
                    }
                }
            }
            Stmt::LocalFunction(f) => {
                let v = self.scopes().declared(f.name());
                self.live.push(Live::Var(v));
            }
            _ => {}
        }
        Ok(())
    }

    fn stmt_of(&mut self, s: &Stmt, out: &mut Vec<St>) -> Result<()> {
        let span = span_of(s);
        match s {
            Stmt::LocalAssignment(l) => {
                let names: Vec<&TokenReference> = l.names().iter().collect();
                let exprs: Vec<&Expression> = l.expressions().iter().collect();
                let (pre, vals) = if exprs.is_empty() {
                    (
                        Vec::new(),
                        names.iter().map(|_| self.nil_val(span)).collect(),
                    )
                } else {
                    self.adjusted(&exprs, names.len(), span)?
                };
                out.extend(pre);
                for (name, v) in names.iter().zip(vals) {
                    let id = self.scopes().declared(name);
                    out.push(self.declare_var(id, v, span));
                    // A `<close>` variable must hold something closable,
                    // and is closed when its block is left.
                    if self.scopes().var(id).attribute.as_deref() == Some("close") {
                        let value = self.read_var(id, span);
                        let value = self.boxed(value);
                        let check = call(
                            "zl_closable",
                            vec![value, str_lit(&ident(name), span)],
                            prim(PrimitiveType::Unit),
                            span,
                        );
                        let st = self.guarded_stmt(check, &Described::NONE);
                        out.push(st);
                        self.tbc.push((self.depth, Closable::Var(id)));
                    }
                }
            }
            Stmt::Assignment(a) => {
                let targets: Vec<&Var> = a.variables().iter().collect();
                let exprs: Vec<&Expression> = a.expressions().iter().collect();
                if targets.len() == 1 && exprs.len() == 1 {
                    // The common case: no temporaries.
                    let v = self.expr(exprs[0])?;
                    let st = self.assign_target(targets[0], v, span)?;
                    out.push(st);
                    return Ok(());
                }
                // Every target's table and key, then every right side,
                // run before any target is stored.
                let mut pre = Vec::new();
                let mut prepared = Vec::with_capacity(targets.len());
                for target in &targets {
                    prepared.push(self.prepare_target(target, &mut pre, span)?);
                }
                let (mut values_pre, vals) = self.adjusted(&exprs, targets.len(), span)?;
                // A value that is a variable is read now: a store may
                // assign it before the value is used.
                let vals: Vec<Val> = vals
                    .into_iter()
                    .map(|v| self.snapshot(v, &mut values_pre))
                    .collect();
                out.extend(pre);
                out.extend(values_pre);
                for (target, v) in prepared.into_iter().zip(vals) {
                    let st = self.store_prepared(target, v, span)?;
                    out.push(st);
                }
            }
            Stmt::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                let multi = self.suffixed(c.prefix(), &suffixes, span)?;
                match multi {
                    Multi::Fixed(vals) => {
                        for v in vals {
                            out.push(expr_stmt(v.node));
                        }
                    }
                    Multi::Dynamic(node) | Multi::None(node) => out.push(expr_stmt(node)),
                }
            }
            Stmt::Do(d) => {
                let body = self.block(d.block())?;
                out.push(stmt(
                    TypedStatement::Block(TypedBlock {
                        statements: body,
                        span,
                    }),
                    span,
                ));
            }
            Stmt::If(i) => {
                let st = self.if_stmt(i, span)?;
                out.push(st);
            }
            Stmt::While(w) => {
                let cond = self.expr(w.condition())?;
                let cond = self.truthy(cond);
                let mut body = self.loop_body(w.block(), span_of(w.end_token()))?;
                body.extend(self.debug_loop_back(span_of(w.while_token()), true, span));
                out.push(while_(cond, body, span));
            }
            Stmt::Repeat(r) => {
                // `repeat body until c` is a loop leaving once `c` holds;
                // the condition sees the body's locals.
                // A `break` closes from the condition's last token.
                let until = span_of(r.until());
                let until = Span::new(until.end.saturating_sub(1), until.end);
                let mut body = self.loop_body(r.block(), until)?;
                if self.frames {
                    let line = self.m.line_of(span_of(r.until_token()));
                    body.extend(self.debug_line(line, span));
                }
                let cond = self.expr(r.until())?;
                let cond = self.truthy(cond);
                body.push(if_(
                    cond,
                    vec![stmt(TypedStatement::Break(None), span)],
                    None,
                    span,
                ));
                body.extend(self.debug_call("zl_dbg_back", Vec::new(), span));
                out.push(while_(bool_lit(true, span), body, span));
            }
            Stmt::NumericFor(f) => self.numeric_for(f, span, out)?,
            Stmt::GenericFor(f) => self.generic_for(f, span, out)?,
            Stmt::LocalFunction(f) => {
                let v = self.scopes().declared(f.name());
                let id = self.scopes().function_of(f.body());
                self.lower_function(id, f.body(), false)?;
                // The record is made once, here, and lives in the
                // variable; a direct call reads its captures from it. A
                // function capturing its own variable needs the cell
                // before the record.
                if matches!(self.storage_of(v), Storage::Cell(..)) {
                    let placeholder = self.nil_val(span);
                    out.push(self.declare_var(v, placeholder, span));
                    let record = self.function_value(id, span);
                    out.push(self.write_var(v, record, span));
                } else {
                    let record = self.function_value(id, span);
                    out.push(self.declare_var(v, record, span));
                }
            }
            Stmt::FunctionDeclaration(f) => {
                let id = self.scopes().function_of(f.body());
                let is_method = f.name().method_name().is_some();
                self.lower_function(id, f.body(), is_method)?;
                let names: Vec<&TokenReference> = f.name().names().iter().collect();
                if names.len() == 1 && !is_method {
                    let token = names[0];
                    let binding = self
                        .scopes()
                        .binding(token)
                        .cloned()
                        .unwrap_or_else(|| Binding::Global(ident(token)));
                    if let Binding::Global(name) = &binding
                        && self.scopes().known_global_function(name) == Some(id)
                    {
                        // The typed entry is the function.
                        return Ok(());
                    }
                    let record = self.function_value(id, span);
                    let st = match binding {
                        Binding::Local(v) | Binding::Upvalue(v) => self.write_var(v, record, span),
                        Binding::Global(name) => self.write_global(&name, record, span)?,
                        Binding::Field(v, name, upvalue) => {
                            self.write_field(v, &name, upvalue, record, span)
                        }
                    };
                    out.push(st);
                    return Ok(());
                }
                // `function a.b.c()` / `function a.b:m()`: index down to
                // the holder, then store.
                let mut obj = self.read_name(names[0])?;
                let mut desc = self.describe_name(names[0]);
                let mut pre = Vec::new();
                for name in &names[1..names.len() - if is_method { 0 } else { 1 }] {
                    let key = Val {
                        node: str_lit(&ident(name), span),
                        ty: Ty::Str,
                    };
                    obj = self.index_read(obj, key, desc, span);
                    desc = Some(format!("field '{}'", ident(name)));
                    obj = self.hold(obj, &mut pre);
                }
                let last = match f.name().method_name() {
                    Some(m) => ident(m),
                    None => ident(names[names.len() - 1]),
                };
                let record = self.function_value(id, span);
                let key = Val {
                    node: str_lit(&last, span),
                    ty: Ty::Str,
                };
                out.extend(pre);
                let st = self.index_write(obj, key, record, desc, span);
                out.push(st);
            }
            Stmt::Goto(g) => {
                let Some(&id) = self
                    .scopes()
                    .gotos
                    .get(&crate::scope::pos_of(g.goto_token()))
                else {
                    return Err(Error::Syntax {
                        message: format!("no visible label '{}' for goto", ident(g.label_name())),
                        span: (span.start, span.end),
                    });
                };
                let target_depth = self.scopes().label_depths[&id];
                self.closes_checked(target_depth + 1, true, span, out);
                out.push(stmt(TypedStatement::Goto(label_name(id)), span));
            }
            Stmt::Label(l) => {
                let id = self.scopes().labels[&crate::scope::pos_of(l.name())];
                out.push(stmt(TypedStatement::Label(label_name(id)), span));
            }
            _ => return unsupported("this statement", span),
        }
        Ok(())
    }

    /// Store into an assignment target.
    /// A target of a multiple assignment with its table and key
    /// evaluated ahead of the values, as the reference does.
    fn prepare_target<'e>(
        &mut self,
        target: &'e Var,
        pre: &mut Vec<St>,
        span: Span,
    ) -> Result<Prepared<'e>> {
        let Var::Expression(ve) = target else {
            return Ok(Prepared::Whole(target));
        };
        let suffixes: Vec<&Suffix> = ve.suffixes().collect();
        if self.typer().global_member(ve.prefix(), &suffixes).is_some() {
            return Ok(Prepared::Whole(target));
        }
        let Some((last, init)) = suffixes.split_last() else {
            return unsupported("this assignment target", span);
        };
        let obj_multi = self.suffixed(ve.prefix(), init, span)?;
        let obj = self.first_of(obj_multi, span);
        let obj = self.snapshot(obj, pre);
        let desc = self.describe_chain(ve.prefix(), init);
        let key = match last {
            Suffix::Index(ast::Index::Dot { name, .. }) => Val {
                node: str_lit(&ident(name), span),
                ty: Ty::Str,
            },
            Suffix::Index(ast::Index::Brackets { expression, .. }) => self.expr(expression)?,
            _ => return unsupported("this assignment target", span),
        };
        let key = self.snapshot(key, pre);
        Ok(Prepared::Index(Box::new((obj, key, desc))))
    }

    fn store_prepared(&mut self, target: Prepared<'_>, v: Val, span: Span) -> Result<St> {
        match target {
            Prepared::Whole(target) => self.assign_target(target, v, span),
            Prepared::Index(place) => {
                let (obj, key, desc) = *place;
                Ok(self.index_write(obj, key, v, desc, span))
            }
        }
    }

    fn assign_target(&mut self, target: &Var, v: Val, span: Span) -> Result<St> {
        match target {
            Var::Name(token) => {
                let binding = self
                    .scopes()
                    .binding(token)
                    .cloned()
                    .unwrap_or_else(|| Binding::Global(ident(token)));
                match binding {
                    Binding::Local(id) | Binding::Upvalue(id) => {
                        if self.scopes().var(id).attribute.as_deref() == Some("const") {
                            return Err(Error::Syntax {
                                message: format!(
                                    "attempt to assign to const variable '{}'",
                                    ident(token)
                                ),
                                span: (span.start, span.end),
                            });
                        }
                        Ok(self.write_var(id, v, span))
                    }
                    Binding::Global(name) => self.write_global(&name, v, span),
                    Binding::Field(id, name, upvalue) => {
                        Ok(self.write_field(id, &name, upvalue, v, span))
                    }
                }
            }
            Var::Expression(ve) => {
                let suffixes: Vec<&Suffix> = ve.suffixes().collect();
                if let Some(name) = self.typer().global_member(ve.prefix(), &suffixes) {
                    return self.write_global(&name, v, span);
                }
                let Some((last, init)) = suffixes.split_last() else {
                    return unsupported("this assignment target", span);
                };
                let obj_multi = self.suffixed(ve.prefix(), init, span)?;
                let obj = self.first_of(obj_multi, span);
                let desc = self.describe_chain(ve.prefix(), init);
                let key = match last {
                    Suffix::Index(ast::Index::Dot { name, .. }) => Val {
                        node: str_lit(&ident(name), span),
                        ty: Ty::Str,
                    },
                    Suffix::Index(ast::Index::Brackets { expression, .. }) => {
                        self.expr(expression)?
                    }
                    _ => return unsupported("this assignment target", span),
                };
                Ok(self.index_write(obj, key, v, desc, span))
            }
            _ => unsupported("this assignment target", span),
        }
    }

    fn if_stmt(&mut self, i: &ast::If, span: Span) -> Result<St> {
        let cond = self.expr(i.condition())?;
        let cond = self.truthy(cond);
        let then = self.block(i.block())?;
        // `elseif` chains nest as else-ifs.
        let mut els: Option<Vec<St>> = i.else_block().map(|b| self.block(b)).transpose()?;
        if let Some(elseifs) = i.else_if() {
            for e in elseifs.iter().rev() {
                let c = self.expr(e.condition())?;
                let c = self.truthy(c);
                let body = self.block(e.block())?;
                els = Some(vec![if_(c, body, els, span_of(e))]);
            }
        }
        Ok(if_(cond, then, els, span))
    }

    /// A loop body. A label ending it is where a `goto` continues to,
    /// and lands ahead of whatever the loop does after the body.
    fn loop_body(&mut self, block: &Block, end: Span) -> Result<Vec<St>> {
        self.loop_depths.push((self.depth + 1, end));
        let body = self.block(block);
        self.loop_depths.pop();
        body
    }

    /// A `for` loop's body: the loop's hidden locals, `states` their
    /// values boxed (read only by `debug.getlocal`), then its
    /// variables, in scope. `end` is the loop's `end`.
    fn for_body(
        &mut self,
        block: &Block,
        states: Vec<Node>,
        vars: &[VarId],
        end: Span,
    ) -> Result<Vec<St>> {
        let live = self.live.len();
        let held = self.for_states.len();
        for state in states {
            self.live.push(Live::ForState(self.for_states.len()));
            self.for_states.push(state);
        }
        self.live.extend(vars.iter().map(|v| Live::Var(*v)));
        let body = self.loop_body(block, end);
        self.live.truncate(live);
        self.for_states.truncate(held);
        body
    }

    /// Whether the program reads locals through the debug library, so
    /// call sites spill them.
    fn debug_locals(&self) -> bool {
        self.m.debug.as_ref().is_some_and(|d| d.locals)
    }

    fn numeric_for(&mut self, f: &ast::NumericFor, span: Span, out: &mut Vec<St>) -> Result<()> {
        let v = self.scopes().declared(f.index_variable());
        let start = self.expr(f.start())?;
        let limit = self.expr(f.end())?;
        let step = f.step().map(|s| self.expr(s)).transpose()?;
        let loop_ty = types::numeric_for_ty(start.ty, step.as_ref().map_or(Ty::Int, |s| s.ty));
        if !matches!(loop_ty, Ty::Int | Ty::Float) {
            return self.numeric_for_dynamic(f, v, (start, limit, step), span, out);
        }
        let is_float = loop_ty == Ty::Float;
        let num_ty = if is_float { Ty::Float } else { Ty::Int };
        let ir = self.ir(num_ty);
        let bool_t = prim(PrimitiveType::Bool);
        // A limit or start of another type is converted as Lua would:
        // a float limit of an integer loop is the last integer the
        // loop may reach, which depends on the step's direction.
        let start_node = match start.ty {
            t if t == num_ty => start.node,
            Ty::Int | Ty::Float => self.coerce(start, num_ty),
            _ => {
                let b = self.boxed(start);
                let node = if is_float {
                    call(
                        "zl_for_float",
                        vec![b, str_lit("initial value", span)],
                        ir.clone(),
                        span,
                    )
                } else {
                    call(
                        "zl_for_int",
                        vec![b, str_lit("initial value", span)],
                        ir.clone(),
                        span,
                    )
                };
                self.guard(Val { node, ty: num_ty }).node
            }
        };
        let limit_literal = match &limit.node.node {
            TypedExpression::Literal(TypedLiteral::Integer(n)) if limit.ty == Ty::Int => {
                Some(*n as i64)
            }
            _ => None,
        };
        // A limit converted with the step in hand is evaluated before
        // the step all the same.
        enum Limit {
            Ready(Node),
            Float(Node),
            Dynamic(Node),
        }
        let limit = match limit.ty {
            t if t == num_ty => Limit::Ready(limit.node),
            Ty::Int => Limit::Ready(self.coerce(limit, num_ty)),
            Ty::Float => Limit::Float(limit.node),
            _ if is_float => {
                let b = self.boxed(limit);
                let node = call(
                    "zl_for_float",
                    vec![b, str_lit("limit", span)],
                    ir.clone(),
                    span,
                );
                Limit::Ready(self.guard(Val { node, ty: num_ty }).node)
            }
            _ => Limit::Dynamic(self.boxed(limit)),
        };
        // The step's sign, when it is a literal, picks the test.
        let (step_node, step_literal): (Node, Option<f64>) = match step {
            None => (
                if is_float {
                    float_lit(1.0, span)
                } else {
                    int_lit(1, span)
                },
                Some(1.0),
            ),
            Some(s) => {
                let literal = match &s.node.node {
                    TypedExpression::Literal(TypedLiteral::Integer(n)) => Some(*n as f64),
                    TypedExpression::Literal(TypedLiteral::Float(x)) => Some(*x),
                    _ => None,
                };
                let n = match s.ty {
                    t if t == num_ty => s.node,
                    Ty::Int | Ty::Float => self.coerce(s, num_ty),
                    _ => {
                        let b = self.boxed(s);
                        let node = if is_float {
                            call(
                                "zl_for_float",
                                vec![b, str_lit("step", span)],
                                ir.clone(),
                                span,
                            )
                        } else {
                            call(
                                "zl_for_int",
                                vec![b, str_lit("step", span)],
                                ir.clone(),
                                span,
                            )
                        };
                        self.guard(Val { node, ty: num_ty }).node
                    }
                };
                (n, literal)
            }
        };
        let counter = self.temp();
        let lim = self.temp();
        let st = self.temp();
        out.push(let_(counter, ir.clone(), start_node, span));
        let converted = match limit {
            Limit::Ready(node) => {
                out.push(let_(lim, ir.clone(), node, span));
                out.push(let_(st, ir.clone(), step_node, span));
                false
            }
            Limit::Float(node) => {
                let raw = self.temp();
                let f64_t = prim(PrimitiveType::F64);
                out.push(let_(raw, f64_t.clone(), node, span));
                out.push(let_(st, ir.clone(), step_node, span));
                let node = call(
                    "zl_for_limit_f",
                    vec![var(raw, f64_t, span), var(st, ir.clone(), span)],
                    ir.clone(),
                    span,
                );
                out.push(let_(lim, ir.clone(), node, span));
                true
            }
            Limit::Dynamic(b) => {
                let raw = self.temp();
                out.push(let_(raw, Type::Any, b, span));
                out.push(let_(st, ir.clone(), step_node, span));
                let node = call(
                    "zl_for_limit",
                    vec![var(raw, Type::Any, span), var(st, ir.clone(), span)],
                    ir.clone(),
                    span,
                );
                let node = self.guard(Val { node, ty: num_ty }).node;
                out.push(let_(lim, ir.clone(), node, span));
                true
            }
        };
        if converted {
            // A limit past the integers against the step: the counter
            // is put past the (clamped) limit, so the loop does not run.
            let past = if_value(
                binary(
                    BinaryOp::Gt,
                    var(st, ir.clone(), span),
                    int_lit(0, span),
                    bool_t.clone(),
                    span,
                ),
                binary(
                    BinaryOp::Add,
                    var(lim, ir.clone(), span),
                    int_lit(1, span),
                    ir.clone(),
                    span,
                ),
                binary(
                    BinaryOp::Sub,
                    var(lim, ir.clone(), span),
                    int_lit(1, span),
                    ir.clone(),
                    span,
                ),
                ir.clone(),
                span,
            );
            out.push(if_(
                var(intern(library::FOR_SKIP), bool_t.clone(), span),
                vec![assign(var(counter, ir.clone(), span), past, span)],
                None,
                span,
            ));
        }
        let zero = if is_float {
            float_lit(0.0, span)
        } else {
            int_lit(0, span)
        };
        let counter_v = || var(counter, ir.clone(), span);
        let lim_v = || var(lim, ir.clone(), span);
        let st_v = || var(st, ir.clone(), span);
        // A zero step is an error, a literal one at once.
        if step_literal.is_none_or(|x| x == 0.0) {
            let raise = expr_stmt(call(
                "zb_fatal",
                vec![str_lit("error", span), str_lit("'for' step is zero", span)],
                prim(PrimitiveType::Unit),
                span,
            ));
            let leave = self.pending_check(span);
            if step_literal.is_none() {
                out.push(if_(
                    binary(BinaryOp::Eq, st_v(), zero.clone(), bool_t.clone(), span),
                    vec![raise, leave],
                    None,
                    span,
                ));
            } else {
                out.push(raise);
                out.push(leave);
            }
        }
        let cond = match step_literal {
            Some(x) if x > 0.0 => binary(BinaryOp::Le, counter_v(), lim_v(), bool_t.clone(), span),
            Some(_) => binary(BinaryOp::Ge, counter_v(), lim_v(), bool_t.clone(), span),
            None => binary(
                BinaryOp::Or,
                binary(
                    BinaryOp::And,
                    binary(BinaryOp::Gt, st_v(), zero.clone(), bool_t.clone(), span),
                    binary(BinaryOp::Le, counter_v(), lim_v(), bool_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                binary(
                    BinaryOp::And,
                    binary(BinaryOp::Lt, st_v(), zero, bool_t.clone(), span),
                    binary(BinaryOp::Ge, counter_v(), lim_v(), bool_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                bool_t.clone(),
                span,
            ),
        };
        // An integer counter must not run past the limit into a wrap:
        // the loop leaves at the limit itself, unless the limit is a
        // literal the step cannot carry past the range.
        let bounded = !is_float
            && !matches!(
                (limit_literal, step_literal),
                (Some(l), Some(s)) if (s as i64).checked_add(l).is_some() && l.checked_add(s as i64).is_some()
                    && (s as i64) != i64::MIN
            );
        // The body's variable is a copy of the counter each iteration.
        let mut body = vec![self.declare_var(
            v,
            Val {
                node: counter_v(),
                ty: num_ty,
            },
            span,
        )];
        let states = self.numeric_states((counter_v(), lim_v(), st_v()), is_float, span);
        let mut inner = self.for_body(f.block(), states, &[v], span_of(f.end_token()))?;
        inner.extend(self.debug_loop_back(span_of(f.for_token()), false, span));
        let mut after = Vec::new();
        if bounded {
            // Whether the step fits between the counter and the limit:
            // the distance and the step's size as unsigned values, so
            // neither wraps.
            let u64_t = prim(PrimitiveType::U64);
            let distance = |from: Node, to: Node| {
                cast(
                    binary(BinaryOp::Sub, to, from, ir.clone(), span),
                    u64_t.clone(),
                    span,
                )
            };
            let up = || distance(counter_v(), lim_v());
            let down = || distance(lim_v(), counter_v());
            let size = |negated: bool| {
                let s = if negated {
                    binary(BinaryOp::Sub, int_lit(0, span), st_v(), ir.clone(), span)
                } else {
                    st_v()
                };
                cast(s, u64_t.clone(), span)
            };
            let short =
                |gap: Node, step: Node| binary(BinaryOp::Lt, gap, step, bool_t.clone(), span);
            let done = match step_literal {
                Some(x) if x == 1.0 || x == -1.0 => {
                    binary(BinaryOp::Eq, counter_v(), lim_v(), bool_t.clone(), span)
                }
                Some(x) if x > 0.0 => short(up(), size(false)),
                Some(_) => short(down(), size(true)),
                None => if_value(
                    binary(BinaryOp::Gt, st_v(), int_lit(0, span), bool_t.clone(), span),
                    short(up(), size(false)),
                    short(down(), size(true)),
                    bool_t.clone(),
                    span,
                ),
            };
            after.push(if_(
                done,
                vec![stmt(TypedStatement::Break(None), span)],
                None,
                span,
            ));
        }
        after.push(assign(
            counter_v(),
            binary(BinaryOp::Add, counter_v(), st_v(), ir.clone(), span),
            span,
        ));
        body.extend(inner);
        body.extend(after);
        out.push(while_(cond, body, span));
        out.extend(self.debug_loop_exit(span_of(f.for_token()), None, span));
        Ok(())
    }

    /// A numeric loop's hidden locals as Lua keeps them, for
    /// `debug.getlocal`: the index, then for an integer loop the
    /// iterations left after this one and for a float loop the limit,
    /// then the step. `parts` are the counter, the limit and the step.
    fn numeric_states(
        &mut self,
        parts: (Node, Node, Node),
        is_float: bool,
        span: Span,
    ) -> Vec<Node> {
        if !self.debug_locals() {
            return vec![nil(span); 3];
        }
        let (counter, lim, st) = parts;
        let num_ty = if is_float { Ty::Float } else { Ty::Int };
        let rest = if is_float {
            lim
        } else {
            Self::iterations_left(counter.clone(), lim, st.clone(), span)
        };
        [counter, rest, st]
            .into_iter()
            .map(|node| self.boxed(Val { node, ty: num_ty }))
            .collect()
    }

    /// How many more times an integer loop runs, counted as Lua counts
    /// it: the distance to the limit over the step's size, unsigned.
    fn iterations_left(counter: Node, lim: Node, st: Node, span: Span) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        let u64_t = prim(PrimitiveType::U64);
        let bool_t = prim(PrimitiveType::Bool);
        let up = binary(BinaryOp::Gt, st.clone(), int_lit(0, span), bool_t, span);
        let distance = if_value(
            up.clone(),
            binary(
                BinaryOp::Sub,
                lim.clone(),
                counter.clone(),
                i64_t.clone(),
                span,
            ),
            binary(BinaryOp::Sub, counter, lim, i64_t.clone(), span),
            i64_t.clone(),
            span,
        );
        // `-(step + 1) + 1`, so the smallest integer step does not wrap.
        let down = binary(
            BinaryOp::Add,
            cast(
                binary(
                    BinaryOp::Sub,
                    int_lit(-1, span),
                    st.clone(),
                    i64_t.clone(),
                    span,
                ),
                u64_t.clone(),
                span,
            ),
            cast(int_lit(1, span), u64_t.clone(), span),
            u64_t.clone(),
            span,
        );
        let size = if_value(up, cast(st, u64_t.clone(), span), down, u64_t.clone(), span);
        cast(
            binary(
                BinaryOp::Div,
                cast(distance, u64_t.clone(), span),
                size,
                u64_t,
                span,
            ),
            i64_t,
            span,
        )
    }

    /// A numeric `for` whose start or step is not known to be an
    /// integer or a float: an integer loop when both are integers when
    /// it starts, else a float loop, the variable a number either way.
    /// `values` are the start, the limit and the step, lowered in that
    /// order.
    fn numeric_for_dynamic(
        &mut self,
        f: &ast::NumericFor,
        v: VarId,
        values: (Val, Val, Option<Val>),
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let bool_t = prim(PrimitiveType::Bool);
        let (start, limit, step) = values;
        let step = step.unwrap_or(Val {
            node: int_lit(1, span),
            ty: Ty::Int,
        });
        let mut boxed = Vec::new();
        for value in [start, limit, step] {
            let b = self.boxed(value);
            let name = self.temp();
            out.push(let_(name, Type::Any, b, span));
            boxed.push(var(name, Type::Any, span));
        }
        let [start, limit, step] = <[Node; 3]>::try_from(boxed).expect("three values");
        let ints = self.temp();
        out.push(let_(
            ints,
            bool_t.clone(),
            call(
                "zl_for_ints",
                vec![start.clone(), step.clone()],
                bool_t.clone(),
                span,
            ),
            span,
        ));
        let ints_v = || var(ints, bool_t.clone(), span);
        let [ci, li, si, cf, lf, sf] = [(); 6].map(|_| self.temp());
        for (name, ty, zero) in [
            (ci, &i64_t, int_lit(0, span)),
            (li, &i64_t, int_lit(0, span)),
            (si, &i64_t, int_lit(0, span)),
            (cf, &f64_t, float_lit(0.0, span)),
            (lf, &f64_t, float_lit(0.0, span)),
            (sf, &f64_t, float_lit(0.0, span)),
        ] {
            out.push(let_(name, ty.clone(), zero, span));
        }
        let int_v = |name| var(name, i64_t.clone(), span);
        let float_v = |name| var(name, f64_t.clone(), span);
        let converted = |this: &mut Self, helper: &str, value: &Node, what: &str, ty: Ty| {
            let node = call(
                helper,
                vec![value.clone(), str_lit(what, span)],
                this.ir(ty),
                span,
            );
            this.guard(Val { node, ty }).node
        };
        let step_zero = |this: &mut Self, is_zero: Node| {
            let raise = expr_stmt(call(
                "zb_fatal",
                vec![str_lit("error", span), str_lit("'for' step is zero", span)],
                prim(PrimitiveType::Unit),
                span,
            ));
            let leave = this.pending_check(span);
            if_(is_zero, vec![raise, leave], None, span)
        };
        // Lua's order: the integer loop checks the step and then the
        // limit; the float loop converts the limit, the step and the
        // start, then checks the step.
        let mut int_prep = vec![assign(
            int_v(si),
            converted(self, "zl_for_int", &step, "step", Ty::Int),
            span,
        )];
        int_prep.push(step_zero(
            self,
            binary(
                BinaryOp::Eq,
                int_v(si),
                int_lit(0, span),
                bool_t.clone(),
                span,
            ),
        ));
        int_prep.push(assign(
            int_v(ci),
            converted(self, "zl_for_int", &start, "initial value", Ty::Int),
            span,
        ));
        let lim = call(
            "zl_for_limit",
            vec![limit.clone(), int_v(si)],
            i64_t.clone(),
            span,
        );
        let lim = self.guard(Val {
            node: lim,
            ty: Ty::Int,
        });
        int_prep.push(assign(int_v(li), lim.node, span));
        // A limit past the integers against the step: the counter is
        // put past the (clamped) limit, so the loop does not run.
        let past = if_value(
            binary(
                BinaryOp::Gt,
                int_v(si),
                int_lit(0, span),
                bool_t.clone(),
                span,
            ),
            binary(
                BinaryOp::Add,
                int_v(li),
                int_lit(1, span),
                i64_t.clone(),
                span,
            ),
            binary(
                BinaryOp::Sub,
                int_v(li),
                int_lit(1, span),
                i64_t.clone(),
                span,
            ),
            i64_t.clone(),
            span,
        );
        int_prep.push(if_(
            var(intern(library::FOR_SKIP), bool_t.clone(), span),
            vec![assign(int_v(ci), past, span)],
            None,
            span,
        ));
        let mut float_prep = vec![
            assign(
                float_v(lf),
                converted(self, "zl_for_float", &limit, "limit", Ty::Float),
                span,
            ),
            assign(
                float_v(sf),
                converted(self, "zl_for_float", &step, "step", Ty::Float),
                span,
            ),
            assign(
                float_v(cf),
                converted(self, "zl_for_float", &start, "initial value", Ty::Float),
                span,
            ),
        ];
        float_prep.push(step_zero(
            self,
            binary(
                BinaryOp::Eq,
                float_v(sf),
                float_lit(0.0, span),
                bool_t.clone(),
                span,
            ),
        ));
        out.push(if_(ints_v(), int_prep, Some(float_prep), span));
        // Whether the counter is within the limit, the step's sign
        // deciding which side.
        let within = |counter: Node, lim: Node, st: Node, zero: Node| {
            binary(
                BinaryOp::Or,
                binary(
                    BinaryOp::And,
                    binary(BinaryOp::Gt, st.clone(), zero.clone(), bool_t.clone(), span),
                    binary(
                        BinaryOp::Le,
                        counter.clone(),
                        lim.clone(),
                        bool_t.clone(),
                        span,
                    ),
                    bool_t.clone(),
                    span,
                ),
                binary(
                    BinaryOp::And,
                    binary(BinaryOp::Lt, st, zero, bool_t.clone(), span),
                    binary(BinaryOp::Ge, counter, lim, bool_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                bool_t.clone(),
                span,
            )
        };
        let cond = if_value(
            ints_v(),
            within(int_v(ci), int_v(li), int_v(si), int_lit(0, span)),
            within(float_v(cf), float_v(lf), float_v(sf), float_lit(0.0, span)),
            bool_t.clone(),
            span,
        );
        let value = if_value(
            ints_v(),
            number_of_int(int_v(ci), span),
            number_of_float(float_v(cf), span),
            number_type(),
            span,
        );
        let mut body = vec![self.declare_var(
            v,
            Val {
                node: value,
                ty: Ty::Number,
            },
            span,
        )];
        let states = if self.debug_locals() {
            let parts = [
                (int_v(ci), float_v(cf)),
                (
                    Self::iterations_left(int_v(ci), int_v(li), int_v(si), span),
                    float_v(lf),
                ),
                (int_v(si), float_v(sf)),
            ];
            parts
                .into_iter()
                .map(|(int, float)| {
                    let node = if_value(
                        ints_v(),
                        number_of_int(int, span),
                        number_of_float(float, span),
                        number_type(),
                        span,
                    );
                    self.boxed(Val {
                        node,
                        ty: Ty::Number,
                    })
                })
                .collect()
        } else {
            vec![nil(span); 3]
        };
        body.extend(self.for_body(f.block(), states, &[v], span_of(f.end_token()))?);
        // The integer counter leaves when the step would carry it past
        // the limit, so it never wraps: the distance and the step's
        // size compared as unsigned values.
        let u64_t = prim(PrimitiveType::U64);
        let unsigned = |n: Node| cast(n, u64_t.clone(), span);
        let short = |gap: Node, size: Node| {
            binary(
                BinaryOp::Lt,
                unsigned(gap),
                unsigned(size),
                bool_t.clone(),
                span,
            )
        };
        let done = if_value(
            binary(
                BinaryOp::Gt,
                int_v(si),
                int_lit(0, span),
                bool_t.clone(),
                span,
            ),
            short(
                binary(BinaryOp::Sub, int_v(li), int_v(ci), i64_t.clone(), span),
                int_v(si),
            ),
            short(
                binary(BinaryOp::Sub, int_v(ci), int_v(li), i64_t.clone(), span),
                binary(
                    BinaryOp::Sub,
                    int_lit(0, span),
                    int_v(si),
                    i64_t.clone(),
                    span,
                ),
            ),
            bool_t.clone(),
            span,
        );
        let int_step = vec![
            if_(
                done,
                vec![stmt(TypedStatement::Break(None), span)],
                None,
                span,
            ),
            assign(
                int_v(ci),
                binary(BinaryOp::Add, int_v(ci), int_v(si), i64_t.clone(), span),
                span,
            ),
        ];
        let float_step = vec![assign(
            float_v(cf),
            binary(BinaryOp::Add, float_v(cf), float_v(sf), f64_t.clone(), span),
            span,
        )];
        body.push(if_(ints_v(), int_step, Some(float_step), span));
        out.push(while_(cond, body, span));
        Ok(())
    }

    fn generic_for(&mut self, f: &ast::GenericFor, span: Span, out: &mut Vec<St>) -> Result<()> {
        let names: Vec<VarId> = f
            .names()
            .iter()
            .map(|n| self.scopes().declared(n))
            .collect();
        let exprs: Vec<&Expression> = f.expressions().iter().collect();
        // `pairs(t)`, `ipairs(t)` and `next, t` walk the table itself.
        if let [Expression::FunctionCall(c)] = exprs.as_slice() {
            let suffixes: Vec<&Suffix> = c.suffixes().collect();
            if let Some(b) = self.typer().builtin_callee(c.prefix(), &suffixes)
                && b.lib.is_empty()
                && (b.name == "pairs" || b.name == "ipairs")
            {
                let Some(Suffix::Call(ast::Call::AnonymousCall(args))) = suffixes.last() else {
                    unreachable!("a builtin callee ends in a call")
                };
                let arg_exprs = self.args_exprs(args);
                if let (1, None) = (arg_exprs.len(), self.literal_arg(args, span)?) {
                    let t = self.expr(arg_exprs[0])?;
                    return if b.name == "ipairs" {
                        self.ipairs_loop(&names, t, f, span, out)
                    } else {
                        self.pairs_loop(&names, t, f, span, out)
                    };
                }
            }
        }
        if let [Expression::Var(Var::Name(token)), table] = exprs.as_slice()
            && let Some(Binding::Global(name)) = self.scopes().binding(token)
            && name == "next"
            && types::builtin_named(self.scopes(), "next").is_some()
        {
            let t = self.expr(table)?;
            return self.pairs_loop(&names, t, f, span, out);
        }
        // The general protocol: `f, s, control` and a closing value,
        // then `f(s, control)` until its first value is nil. A closing
        // value comes from a fourth expression that is not nil, or from
        // the last expression's values when fewer are written.
        let listed = self.expr_list(&exprs)?;
        let (_, fixed, tail) = &listed;
        let closes = match fixed.get(3) {
            Some(v) => v.ty != Ty::Nil,
            None => tail.is_some(),
        };
        let (pre, vals) = self.adjust(listed, if closes { 4 } else { 3 }, span);
        out.extend(pre);
        let mut it = vals.into_iter();
        let (fv, sv, cv) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        let fname = self.temp();
        let sname = self.temp();
        let cname = self.temp();
        let fb = self.boxed(fv);
        let sb = self.boxed(sv);
        let cb = self.boxed(cv);
        out.push(let_(fname, Type::Any, fb, span));
        out.push(let_(sname, Type::Any, sb, span));
        out.push(let_(cname, Type::Any, cb, span));
        // The closing value is checked once, before the first call, and
        // closed however the loop is left.
        let closing = match it.next() {
            Some(value) => {
                let name = self.temp();
                let value = self.boxed(value);
                out.push(let_(name, Type::Any, value, span));
                let check = call(
                    "zl_closable",
                    vec![var(name, Type::Any, span), str_lit("(for state)", span)],
                    prim(PrimitiveType::Unit),
                    span,
                );
                let st = self.guarded_stmt(check, &Described::NONE);
                out.push(st);
                self.tbc.push((self.depth + 1, Closable::Loop(name)));
                Some(name)
            }
            None => None,
        };
        let vals_name = self.temp();
        let step = call(
            "zl_call_2",
            vec![
                var(fname, Type::Any, span),
                var(sname, Type::Any, span),
                var(cname, Type::Any, span),
            ],
            Type::Any,
            span,
        );
        let step = self
            .guard(Val {
                node: step,
                ty: Ty::Any,
            })
            .node;
        let mut body = vec![let_(
            vals_name,
            self.m.anys(),
            call("zl_values", vec![step], self.m.anys(), span),
            span,
        )];
        let first = call(
            "zl_value_at",
            vec![var(vals_name, self.m.anys(), span), int_lit(1, span)],
            Type::Any,
            span,
        );
        body.push(assign(var(cname, Type::Any, span), first, span));
        body.push(if_(
            binary(
                BinaryOp::Eq,
                var(cname, Type::Any, span),
                nil(span),
                prim(PrimitiveType::Bool),
                span,
            ),
            vec![stmt(TypedStatement::Break(None), span)],
            None,
            span,
        ));
        for (i, v) in names.iter().enumerate() {
            let value = if i == 0 {
                var(cname, Type::Any, span)
            } else {
                call(
                    "zl_value_at",
                    vec![
                        var(vals_name, self.m.anys(), span),
                        int_lit(i as i64 + 1, span),
                    ],
                    Type::Any,
                    span,
                )
            };
            body.push(self.declare_var(
                *v,
                Val {
                    node: value,
                    ty: Ty::Any,
                },
                span,
            ));
        }
        let states = vec![
            var(fname, Type::Any, span),
            var(sname, Type::Any, span),
            var(cname, Type::Any, span),
            closing.map_or_else(|| nil(span), |name| var(name, Type::Any, span)),
        ];
        body.extend(self.for_body(f.block(), states, &names, span_of(f.end_token()))?);
        body.extend(self.debug_loop_back(span_of(f.for_token()), false, span));
        out.push(while_(bool_lit(true, span), body, span));
        if closing.is_some() {
            self.closes_checked(self.depth + 1, true, span_of(f.end_token()), out);
            self.tbc.pop();
        }
        let end = self.m.line_of(span_of(f.end_token()));
        out.extend(self.debug_loop_exit(span_of(f.for_token()), Some(end), span));
        Ok(())
    }

    /// `for i, v in ipairs(t)`: `t[1], t[2], …` until nil.
    fn ipairs_loop(
        &mut self,
        names: &[VarId],
        t: Val,
        f: &ast::GenericFor,
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let block = f.block();
        let i64_t = prim(PrimitiveType::I64);
        let tname = self.temp();
        let counter = self.temp();
        let value = self.temp();
        let t_ir = self.ir(t.ty);
        let t_ty = t.ty;
        out.push(let_(tname, t_ir.clone(), t.node, span));
        out.push(let_(counter, i64_t.clone(), int_lit(1, span), span));
        let read = self.index_read(
            Val {
                node: var(tname, t_ir, span),
                ty: t_ty,
            },
            Val {
                node: var(counter, i64_t.clone(), span),
                ty: Ty::Int,
            },
            None,
            span,
        );
        // The element as the read types it; the loop ends at nil.
        let read_ty = read.ty;
        let read_ir = self.ir(read_ty);
        let mut body = vec![let_(value, read_ir.clone(), read.node, span)];
        let at_end = self.is_nil_val(Val {
            node: var(value, read_ir.clone(), span),
            ty: read_ty,
        });
        body.push(if_(
            at_end,
            vec![stmt(TypedStatement::Break(None), span)],
            None,
            span,
        ));
        if let Some(k) = names.first() {
            body.push(self.declare_var(
                *k,
                Val {
                    node: var(counter, i64_t.clone(), span),
                    ty: Ty::Int,
                },
                span,
            ));
        }
        if let Some(v) = names.get(1) {
            body.push(self.declare_var(
                *v,
                Val {
                    node: var(value, read_ir, span),
                    ty: read_ty,
                },
                span,
            ));
        }
        for v in names.iter().skip(2) {
            let n = self.nil_val(span);
            body.push(self.declare_var(*v, n, span));
        }
        // What `ipairs(t)` would have given: its iterator, `t`, and the
        // index as the control.
        let states = if self.debug_locals() {
            let t = self.boxed(Val {
                node: var(tname, self.ir(t_ty), span),
                ty: t_ty,
            });
            let iter = self.temp();
            let triple = call(
                "zl_values",
                vec![call("zl_ipairs", vec![t.clone()], Type::Any, span)],
                self.m.anys(),
                span,
            );
            out.push(let_(
                iter,
                Type::Any,
                call(
                    "zl_value_at",
                    vec![triple, int_lit(1, span)],
                    Type::Any,
                    span,
                ),
                span,
            ));
            let index = self.boxed(Val {
                node: var(counter, i64_t.clone(), span),
                ty: Ty::Int,
            });
            vec![var(iter, Type::Any, span), t, index, nil(span)]
        } else {
            vec![nil(span); 4]
        };
        let mut inner = self.for_body(block, states, names, span_of(f.end_token()))?;
        inner.extend(self.debug_loop_back(span_of(f.for_token()), false, span));
        let increment = assign(
            var(counter, i64_t.clone(), span),
            binary(
                BinaryOp::Add,
                var(counter, i64_t.clone(), span),
                int_lit(1, span),
                i64_t,
                span,
            ),
            span,
        );
        self.push_loop_with_increment(body, inner, increment, span, out);
        let end = self.m.line_of(span_of(f.end_token()));
        out.extend(self.debug_loop_exit(span_of(f.for_token()), Some(end), span));
        Ok(())
    }

    /// `for k, v in pairs(t)`: every position holding a value, walked
    /// in place. A table whose metatable has `__pairs` is walked as
    /// that says instead: `f, s, c` from the handler, then `f(s, c)`
    /// until its first value is nil. The body is lowered once, so one
    /// loop serves both, choosing its step by the handler's presence.
    fn pairs_loop(
        &mut self,
        names: &[VarId],
        t: Val,
        f: &ast::GenericFor,
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let block = f.block();
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let table_t = self.ir(Ty::Table);
        let anys_t = self.m.anys();
        let tname = self.temp();
        let t = if t.ty == Ty::Table {
            t.node
        } else {
            let b = self.boxed(t);
            let node = call(
                "zl_as_table",
                vec![b, str_lit("for iterator", span)],
                table_t.clone(),
                span,
            );
            self.guard(Val {
                node,
                ty: Ty::Table,
            })
            .node
        };
        out.push(let_(tname, table_t.clone(), t, span));
        let tv = || var(tname, table_t.clone(), span);
        // The handler, or nil.
        let handler = self.temp();
        out.push(let_(
            handler,
            Type::Any,
            call(
                "zl_meta",
                vec![tv(), str_lit("__pairs", span)],
                Type::Any,
                span,
            ),
            span,
        ));
        let hv = || var(handler, Type::Any, span);
        let walked = || binary(BinaryOp::Eq, hv(), nil(span), bool_t.clone(), span);
        // The walk: the position, and the array part's length at the
        // last step, since a body that removes elements shortens it,
        // moving the positions after.
        let pos = self.temp();
        let seen = self.temp();
        out.push(let_(pos, i64_t.clone(), int_lit(-1, span), span));
        out.push(let_(seen, i64_t.clone(), int_lit(0, span), span));
        // The protocol: `f`, `s` and the control `c`.
        let (fname, sname, cname) = (self.temp(), self.temp(), self.temp());
        for name in [fname, sname, cname] {
            out.push(let_(name, Type::Any, nil(span), span));
        }
        let array_len = || call("zl_len", vec![tv()], i64_t.clone(), span);
        let value_at = |vals: InternedString, i: i64| {
            call(
                "zl_value_at",
                vec![var(vals, anys_t.clone(), span), int_lit(i, span)],
                Type::Any,
                span,
            )
        };
        let triple = self.temp();
        let from_handler = {
            let b = self.box_table(tv());
            let called = call("zl_call_1", vec![hv(), b], Type::Any, span);
            let called = self
                .guard(Val {
                    node: called,
                    ty: Ty::Any,
                })
                .node;
            vec![
                let_(
                    triple,
                    anys_t.clone(),
                    call("zl_values", vec![called], anys_t.clone(), span),
                    span,
                ),
                assign(var(fname, Type::Any, span), value_at(triple, 1), span),
                assign(var(sname, Type::Any, span), value_at(triple, 2), span),
                assign(var(cname, Type::Any, span), value_at(triple, 3), span),
            ]
        };
        let mut walk = vec![
            assign(
                var(pos, i64_t.clone(), span),
                call(
                    "zl_next_pos",
                    vec![tv(), int_lit(0, span)],
                    i64_t.clone(),
                    span,
                ),
                span,
            ),
            assign(var(seen, i64_t.clone(), span), array_len(), span),
        ];
        // The walk keeps what `pairs(t)` would have given, `next` and
        // `t`, where `debug.getlocal` reads the loop's state.
        if self.debug_locals() {
            let next = crate::library::stdlib::BUILTINS
                .iter()
                .find(|b| b.lib.is_empty() && b.name == "next")
                .expect("the base library has next");
            let next = self.builtin_value(next, span).node;
            let t = self.box_table(tv());
            walk.push(assign(var(fname, Type::Any, span), next, span));
            walk.push(assign(var(sname, Type::Any, span), t, span));
        }
        out.push(if_(walked(), walk, Some(from_handler), span));
        // Each iteration's key and value.
        let (key, value) = (self.temp(), self.temp());
        let mut body = vec![
            let_(key, Type::Any, nil(span), span),
            let_(value, Type::Any, nil(span), span),
        ];
        let leave = || stmt(TypedStatement::Break(None), span);
        let from_position = vec![
            if_(
                binary(
                    BinaryOp::Lt,
                    var(pos, i64_t.clone(), span),
                    int_lit(0, span),
                    bool_t.clone(),
                    span,
                ),
                vec![leave()],
                None,
                span,
            ),
            // The value first: making the key may collect, which clears
            // a weak value nothing holds yet.
            assign(
                var(value, Type::Any, span),
                call(
                    "zl_pos_value",
                    vec![tv(), var(pos, i64_t.clone(), span)],
                    Type::Any,
                    span,
                ),
                span,
            ),
            // A collection since the step found this position cleared
            // a weak value there: the walk goes on from the next live
            // position, found and read with nothing between that can
            // collect.
            if_(
                binary(
                    BinaryOp::Eq,
                    var(value, Type::Any, span),
                    nil(span),
                    bool_t.clone(),
                    span,
                ),
                vec![
                    assign(
                        var(pos, i64_t.clone(), span),
                        call(
                            "zl_next_pos",
                            vec![
                                tv(),
                                binary(
                                    BinaryOp::Add,
                                    var(pos, i64_t.clone(), span),
                                    int_lit(1, span),
                                    i64_t.clone(),
                                    span,
                                ),
                            ],
                            i64_t.clone(),
                            span,
                        ),
                        span,
                    ),
                    if_(
                        binary(
                            BinaryOp::Lt,
                            var(pos, i64_t.clone(), span),
                            int_lit(0, span),
                            bool_t.clone(),
                            span,
                        ),
                        vec![leave()],
                        None,
                        span,
                    ),
                    assign(
                        var(value, Type::Any, span),
                        call(
                            "zl_pos_value",
                            vec![tv(), var(pos, i64_t.clone(), span)],
                            Type::Any,
                            span,
                        ),
                        span,
                    ),
                ],
                None,
                span,
            ),
            assign(
                var(key, Type::Any, span),
                call(
                    "zl_pos_key",
                    vec![tv(), var(pos, i64_t.clone(), span)],
                    Type::Any,
                    span,
                ),
                span,
            ),
        ];
        let vals = self.temp();
        let from_protocol = {
            let step = call(
                "zl_call_2",
                vec![
                    var(fname, Type::Any, span),
                    var(sname, Type::Any, span),
                    var(cname, Type::Any, span),
                ],
                Type::Any,
                span,
            );
            let step = self
                .guard(Val {
                    node: step,
                    ty: Ty::Any,
                })
                .node;
            vec![
                let_(
                    vals,
                    anys_t.clone(),
                    call("zl_values", vec![step], anys_t.clone(), span),
                    span,
                ),
                assign(var(key, Type::Any, span), value_at(vals, 1), span),
                if_(
                    binary(
                        BinaryOp::Eq,
                        var(key, Type::Any, span),
                        nil(span),
                        bool_t.clone(),
                        span,
                    ),
                    vec![leave()],
                    None,
                    span,
                ),
                assign(var(value, Type::Any, span), value_at(vals, 2), span),
                assign(var(cname, Type::Any, span), var(key, Type::Any, span), span),
            ]
        };
        body.push(if_(walked(), from_position, Some(from_protocol), span));
        for (i, v) in names.iter().enumerate() {
            let node = match i {
                0 => var(key, Type::Any, span),
                1 => var(value, Type::Any, span),
                _ => nil(span),
            };
            let val = Val {
                node,
                ty: if i < 2 { Ty::Any } else { Ty::Nil },
            };
            body.push(self.declare_var(*v, val, span));
        }
        let states = vec![
            var(fname, Type::Any, span),
            var(sname, Type::Any, span),
            var(key, Type::Any, span),
            nil(span),
        ];
        let mut inner = self.for_body(block, states, names, span_of(f.end_token()))?;
        inner.extend(self.debug_loop_back(span_of(f.for_token()), false, span));
        let step = assign(
            var(pos, i64_t.clone(), span),
            call(
                "zl_next_pos_from",
                vec![
                    tv(),
                    binary(
                        BinaryOp::Add,
                        var(pos, i64_t.clone(), span),
                        int_lit(1, span),
                        i64_t.clone(),
                        span,
                    ),
                    var(seen, i64_t.clone(), span),
                ],
                i64_t.clone(),
                span,
            ),
            span,
        );
        let remember = assign(var(seen, i64_t.clone(), span), array_len(), span);
        let advance = if_(walked(), vec![step, remember], None, span);
        let mut whole = body;
        whole.extend(inner);
        whole.push(advance);
        out.push(while_(bool_lit(true, span), whole, span));
        let end = self.m.line_of(span_of(f.end_token()));
        out.extend(self.debug_loop_exit(span_of(f.for_token()), Some(end), span));
        Ok(())
    }

    /// A `while true` loop whose body ends in an increment; a label
    /// ending the body lands ahead of it.
    fn push_loop_with_increment(
        &mut self,
        head: Vec<St>,
        inner: Vec<St>,
        increment: St,
        span: Span,
        out: &mut Vec<St>,
    ) {
        let mut body = head;
        body.extend(inner);
        body.push(increment);
        out.push(while_(bool_lit(true, span), body, span));
    }

    // ─── functions ──────────────────────────────────────────────

    /// Lower a function's body into its typed entry, and its record
    /// code if it is used as a value. Nested functions that capture
    /// something take their record as `env`.
    fn lower_function(
        &mut self,
        id: FuncId,
        body: &ast::FunctionBody,
        _is_method: bool,
    ) -> Result<()> {
        let span = span_of(body);
        let info = self.scopes().func(id).clone();
        let sig = self.m.sig(id);
        let has_env = self.m.takes_env(id);
        let mut child = Lowerer::new(self.m, id);
        child.frames = self.m.debug.is_some();
        child.live = info.params.iter().map(|v| Live::Var(*v)).collect();
        let mut params = Vec::new();
        let mut statements = Vec::new();
        if has_env {
            params.push(parameter(intern("env"), self.m.anys(), span));
            statements.extend(child.env_prologue(span));
        }
        for (i, v) in info.params.iter().enumerate() {
            let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
            let local_ty = child.var_ty(*v);
            let vinfo = self.scopes().var(*v).clone();
            let symbol = child.local_symbol(*v);
            let arg = intern(&format!("{}$arg", vinfo.name));
            params.push(parameter(arg, child.ir(param_ty), span));
            // The parameter as the body's variable: its own type, or a cell.
            let value = Val {
                node: var(arg, child.ir(param_ty), span),
                ty: param_ty,
            };
            if vinfo.needs_cell() {
                let cell_symbol = intern(&format!("{}$cell{}", vinfo.name, v.0));
                child
                    .storage
                    .insert(*v, Storage::Cell(cell_symbol, local_ty));
                statements.push(child.declare_var(*v, value, span));
            } else {
                child.storage.insert(*v, Storage::Local(symbol, local_ty));
                let value = child.coerce(value, local_ty);
                statements.push(let_(symbol, child.ir(local_ty), value, span));
            }
        }
        if info.is_vararg {
            let name = intern("$varargs");
            params.push(parameter(name, self.m.anys(), span));
            child.varargs = Some(name);
        }
        // The function counts itself on the stack, so an unbounded
        // recursion is an error to catch; every return uncounts it. One
        // whose depth is bounded by the text keeps no count.
        child.counts_depth = !self.m.is_bounded(id);
        child.end_span = Some(span_of(body.end_token()));
        let body_statements = child.block(body.block())?;
        if child.entry_line {
            statements.push(entry_line_save(span));
        }
        if child.frames {
            statements.push(child.debug_enter(id, span));
        }
        if child.counts_depth {
            statements.push(child.stack_check(span));
        }
        statements.extend(body_statements);
        // Falling off the end returns nothing, from the line of `end`.
        let last_line = self.m.line_of(span_of(body.end_token()));
        if types::falls_through(body.block()) {
            if child.frames {
                statements.extend(child.debug_line(last_line, span));
            }
            child.return_stmt(&[], span, &mut statements)?;
        }
        if child.frames {
            let line = self
                .m
                .source_line(span_of(body.parameters_parentheses().tokens().0));
            let last_line = self.m.source_line(span_of(body.end_token()));
            child.debug_function_records(id, line, last_line);
        }
        self.m.strip_line_restores(&mut statements);
        let entry = self.m.entry_name(id);
        let function = typed_function(
            &entry,
            params,
            self.m.return_ir(&sig.returns),
            statements,
            span,
        );
        let dynamic = self.m.calls_dynamically(
            &function
                .body
                .as_ref()
                .map(|b| b.statements.clone())
                .unwrap_or_default(),
        );
        self.m.facts.borrow_mut().insert(
            id,
            RaiseFact {
                own: child.raised,
                checks_depth: child.counts_depth,
                dynamic,
                callees: child.raise_callees.clone(),
            },
        );
        self.m.functions.borrow_mut().push(function);
        // The record code, whether or not anything takes the function
        // as a value: unreached, it costs nothing.
        self.record_code(id, &sig, span);
        Ok(())
    }

    /// The record code of a function: the shape every function value
    /// has, calling the typed entry.
    fn record_code(&mut self, id: FuncId, sig: &types::Sig, span: Span) {
        let info = self.scopes().func(id).clone();
        let mut params = vec![parameter(intern("env"), self.m.anys(), span)];
        let mut args: Vec<Node> = Vec::new();
        if self.m.takes_env(id) {
            args.push(var(intern("env"), self.m.anys(), span));
        }
        let mut statements = Vec::new();
        let n = info.params.len();
        let mut child = Lowerer::new(self.m, id);
        if info.is_vararg {
            // Variadic: one packed list, split into the parameters and
            // the rest.
            params.push(parameter(intern("packed"), Type::Any, span));
            let all = intern("$all");
            statements.push(let_(
                all,
                self.m.anys(),
                call(
                    "zl_values",
                    vec![var(intern("packed"), Type::Any, span)],
                    self.m.anys(),
                    span,
                ),
                span,
            ));
            for i in 0..n {
                let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
                let value = call(
                    "zl_value_at",
                    vec![var(all, self.m.anys(), span), int_lit(i as i64 + 1, span)],
                    Type::Any,
                    span,
                );
                args.push(child.coerce(
                    Val {
                        node: value,
                        ty: Ty::Any,
                    },
                    param_ty,
                ));
            }
            args.push(call(
                "zl_slice",
                vec![
                    var(all, self.m.anys(), span),
                    int_lit(n as i64, span),
                    list_len(var(all, self.m.anys(), span), span),
                ],
                self.m.anys(),
                span,
            ));
        } else {
            for i in 0..n {
                let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
                let arg = intern(&format!("a{i}"));
                params.push(parameter(arg, Type::Any, span));
                args.push(child.coerce(
                    Val {
                        node: var(arg, Type::Any, span),
                        ty: Ty::Any,
                    },
                    param_ty,
                ));
            }
        }
        let value = call(
            &self.m.entry_name(id),
            args,
            self.m.return_ir(&sig.returns),
            span,
        );
        // The result as one dynamic value.
        let result = match &sig.returns {
            Returns::Fixed(types) if types.is_empty() => block_value(
                vec![expr_stmt(value)],
                call("zl_none", vec![], Type::Any, span),
                span,
            ),
            Returns::Fixed(types) if types.len() == 1 => child.coerce(
                Val {
                    node: value,
                    ty: types[0].settled(),
                },
                Ty::Any,
            ),
            Returns::Fixed(_) => call("zb_box_tuple", vec![value], Type::Any, span),
            Returns::Dynamic | Returns::Unknown => value,
        };
        statements.push(ret(Some(result), span));
        let function = typed_function(&self.m.code_name(id), params, Type::Any, statements, span);
        self.m.functions.borrow_mut().push(function);
    }
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

impl LineRestores<'_> {
    fn strip(statements: &mut [St], reads_line: &dyn Fn(&str) -> bool) {
        let mut walk = LineRestores {
            reads_line,
            line: intern(library::LINE),
            pending: intern(library::PENDING),
            exits: Vec::new(),
        };
        walk.stmts(statements, false, true);
    }

    /// Whether the line is read from before `statements` on, given
    /// `later` after them; `strip` removes dead restores on the way.
    fn stmts(&mut self, statements: &mut [St], mut later: bool, strip: bool) -> bool {
        for s in statements.iter_mut().rev() {
            later = self.stmt(s, later, strip);
        }
        later
    }

    fn stmt(&mut self, s: &mut St, later: bool, strip: bool) -> bool {
        match &mut s.node {
            TypedStatement::Expression(e) => self.expr(e, later, strip),
            TypedStatement::Let(l) => match &mut l.initializer {
                Some(e) => self.expr(e, later, strip),
                None => later,
            },
            TypedStatement::If(i) if self.is_check(&i.condition) => {
                // The raising call before the check reads the line.
                if strip && !later && self.is_restore(i.else_block.as_ref()) {
                    i.else_block = None;
                }
                true
            }
            TypedStatement::If(i) => {
                let then = self.stmts(&mut i.then_block.statements, later, strip);
                let els = match &mut i.else_block {
                    Some(b) => self.stmts(&mut b.statements, later, strip),
                    None => later,
                };
                self.expr(&mut i.condition, then || els, strip)
            }
            TypedStatement::While(w) => {
                // The line read at the head, which the end of the body
                // reaches again: the least answer, from a first walk
                // assuming it is not; inner loops assume it is.
                let head = if strip {
                    self.exits.push(later);
                    let body = self.stmts(&mut w.body.statements, false, false);
                    self.exits.pop();
                    self.expr(&mut w.condition, body || later, false)
                } else {
                    true
                };
                self.exits.push(later);
                let body = self.stmts(&mut w.body.statements, head, strip);
                self.exits.pop();
                self.expr(&mut w.condition, body || later, strip)
            }
            TypedStatement::Block(b) => self.stmts(&mut b.statements, later, strip),
            TypedStatement::Return(v) => match v {
                Some(e) => self.expr(e, false, strip),
                None => false,
            },
            TypedStatement::Break(None) => self.exits.last().copied().unwrap_or(true),
            TypedStatement::Label(_) => later,
            _ => true,
        }
    }

    fn expr(&mut self, e: &mut Node, later: bool, strip: bool) -> bool {
        match &mut e.node {
            TypedExpression::Literal(_) => later,
            TypedExpression::Variable(name) => later || *name == self.line,
            TypedExpression::Call(c) => {
                let reads = match &c.callee.node {
                    TypedExpression::Variable(name) => name
                        .resolve_global()
                        .is_some_and(|n| (self.reads_line)(n.as_str())),
                    _ => true,
                };
                let mut later = later || reads;
                for a in c.positional_args.iter_mut().rev() {
                    later = self.expr(a, later, strip);
                }
                if c.named_args.is_empty() { later } else { true }
            }
            TypedExpression::MethodCall(c) => {
                let mut later = later;
                for a in c.positional_args.iter_mut().rev() {
                    later = self.expr(a, later, strip);
                }
                if c.named_args.is_empty() {
                    self.expr(&mut c.receiver, later, strip)
                } else {
                    true
                }
            }
            TypedExpression::Binary(b) => match b.op {
                BinaryOp::Assign => match &b.left.node {
                    // A store of the line: nothing before it is read
                    // through it.
                    TypedExpression::Variable(name) if *name == self.line => {
                        self.expr(&mut b.right, false, strip)
                    }
                    _ => {
                        let later = self.expr(&mut b.left, later, strip);
                        self.expr(&mut b.right, later, strip)
                    }
                },
                BinaryOp::And | BinaryOp::Or => {
                    let right = self.expr(&mut b.right, later, strip);
                    self.expr(&mut b.left, right || later, strip)
                }
                _ => {
                    let later = self.expr(&mut b.right, later, strip);
                    self.expr(&mut b.left, later, strip)
                }
            },
            TypedExpression::Unary(u) => self.expr(&mut u.operand, later, strip),
            TypedExpression::Cast(c) => self.expr(&mut c.expr, later, strip),
            TypedExpression::Field(f) => self.expr(&mut f.object, later, strip),
            TypedExpression::Index(i) => {
                let later = self.expr(&mut i.index, later, strip);
                self.expr(&mut i.object, later, strip)
            }
            TypedExpression::If(i) => {
                let then = self.expr(&mut i.then_branch, later, strip);
                let els = self.expr(&mut i.else_branch, later, strip);
                self.expr(&mut i.condition, then || els, strip)
            }
            TypedExpression::Block(b) => self.stmts(&mut b.statements, later, strip),
            TypedExpression::Array(items) | TypedExpression::Tuple(items) => {
                let mut later = later;
                for item in items.iter_mut().rev() {
                    later = self.expr(item, later, strip);
                }
                later
            }
            TypedExpression::Struct(s) => {
                let mut later = later;
                for f in s.fields.iter_mut().rev() {
                    later = self.expr(&mut f.value, later, strip);
                }
                later
            }
            _ => true,
        }
    }

    /// `pending != nil`: the check after a call that may have raised.
    fn is_check(&self, cond: &Node) -> bool {
        matches!(&cond.node, TypedExpression::Binary(b)
            if b.op == BinaryOp::Ne
                && matches!(&b.left.node, TypedExpression::Variable(n) if *n == self.pending)
                && matches!(&b.right.node, TypedExpression::Literal(TypedLiteral::Null)))
    }

    /// A check's else branch that only stores the line.
    fn is_restore(&self, els: Option<&TypedBlock>) -> bool {
        let Some(b) = els else {
            return false;
        };
        let [only] = b.statements.as_slice() else {
            return false;
        };
        let TypedStatement::Expression(e) = &only.node else {
            return false;
        };
        matches!(&e.node, TypedExpression::Binary(b)
            if b.op == BinaryOp::Assign
                && matches!(&b.left.node, TypedExpression::Variable(n) if *n == self.line)
                && matches!(&b.right.node, TypedExpression::Literal(TypedLiteral::Integer(_))))
    }
}

/// A label's name in the typed program, from its number.
fn label_name(id: u32) -> InternedString {
    intern(&format!("$label{id}"))
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

/// `zl_depth += by`.
fn depth_step(by: i64, span: Span) -> St {
    let i64_t = prim(PrimitiveType::I64);
    let depth = var(intern(library::DEPTH), i64_t.clone(), span);
    assign(
        depth.clone(),
        binary(BinaryOp::Add, depth, int_lit(by, span), i64_t, span),
        span,
    )
}

fn entry_line_save(span: Span) -> St {
    let i64_t = prim(PrimitiveType::I64);
    let_(
        intern(ENTRY_LINE),
        i64_t.clone(),
        var(intern(library::LINE), i64_t, span),
        span,
    )
}

/// A field of a record for the debug library's host: no record or
/// field separator inside.
fn sanitize(field: &str) -> String {
    field.replace(['\n', '\x1f'], "?")
}

/// The chunk name as positions spell it (`luaO_chunkid`): a file name
/// longer than the reference's buffer keeps its tail after `...`.
fn chunk_id(file: &str) -> String {
    const IDSIZE: usize = 60;
    const KEPT: usize = IDSIZE - "...".len() - 1;
    if file.len() < IDSIZE {
        return file.to_string();
    }
    let tail = file.len() - KEPT;
    let mut cut = tail;
    while !file.is_char_boundary(cut) {
        cut += 1;
    }
    format!("...{}", &file[cut..])
}

/// A file the program requires, parsed and resolved on its own.
struct Loaded {
    name: String,
    file: String,
    source: String,
    ast: ast::Ast,
    scopes: Scopes,
}

/// The files `require`d by name, transitively, each parsed once. A
/// name that is a standard library or has no file is left to
/// `require` at run time.
/// The files `require` names, loaded in turn, and whether every name
/// was found: one that was not is loaded while the program runs.
fn load_required(first: &[String], main_file: &str) -> Result<(Vec<Loaded>, bool)> {
    let dir = std::path::Path::new(main_file)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let mut loaded: Vec<Loaded> = Vec::new();
    let mut queue: Vec<String> = first.to_vec();
    let mut seen: HashSet<String> = HashSet::new();
    let mut all_found = true;
    while let Some(name) = queue.pop() {
        if !seen.insert(name.clone())
            || crate::library::stdlib::LIBS.contains(&name.as_str())
            || name == "_G"
        {
            continue;
        }
        let relative = format!("{}.lua", name.replace('.', "/"));
        let path = dir.join(&relative);
        // A binary chunk is left to `require` at run time, which
        // reads it back.
        let Some(bytes) = std::fs::read(&path).ok().filter(|b| !crate::is_binary(b)) else {
            all_found = false;
            continue;
        };
        let text = crate::source_text(&bytes);
        let file = path.display().to_string();
        // Positioned in the file it is in, not the program's.
        let (ast, source) = crate::parse_chunk(&text, crate::LOAD_LEVEL).map_err(|e| {
            Error::Rejected(crate::parse::SyntaxError {
                line: None,
                message: e.one_line(&file, &text),
                offset: 0,
            })
        })?;
        let source = source.into_owned();
        let scopes = crate::scope::resolve(&ast);
        queue.extend(scopes.requires.iter().cloned());
        loaded.push(Loaded {
            name,
            file,
            source,
            ast,
            scopes,
        });
    }
    Ok((loaded, all_found))
}

/// Whether `#` on a table may be a `__len` metamethod's value, of
/// any type: a source of the program names the metamethod, or code
/// the program loads while it runs might.
fn len_meta_possible(source: &str, scopes: &Scopes, loaded: &[Loaded], all_found: bool) -> bool {
    let names_it = |s: &str| s.contains("__len");
    !all_found
        || scopes.dynamic_code
        || names_it(source)
        || loaded
            .iter()
            .any(|m| m.scopes.dynamic_code || names_it(&m.source))
}

fn line_starts_of(source: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(source.match_indices('\n').map(|(i, _)| i + 1))
        .collect()
}

/// A module's variables and functions as declarations.
fn declare(module: &Module<'_>, declarations: &mut Vec<TypedNode<TypedDeclaration>>) {
    for (name, ty) in module.module_vars.borrow().iter() {
        declarations.push(TypedNode::new(
            TypedDeclaration::Variable(TypedVariable {
                name: *name,
                ty: module.ir(*ty),
                mutability: Mutability::Mutable,
                initializer: None,
                visibility: Visibility::Public,
            }),
            Type::Unknown,
            Span::new(0, 0),
        ));
    }
    for f in module.functions.borrow().iter() {
        let span = f.body.as_ref().map(|b| b.span).unwrap_or(Span::new(0, 0));
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(f.clone()),
            Type::Unknown,
            span,
        ));
    }
}

/// A chunk besides the main one (a required file, a loaded string) as
/// a function `lua$<tag>chunk($varargs) -> Any` returning its values,
/// with a variadic record code `lua$<tag>chunk$fn` a program calls it
/// through. Its declarations are appended; the record code's name is
/// returned.
/// `env_var` names the chunk's environment when it has one of its
/// own; `own_env` says the chunk starts it as the globals table,
/// rather than being handed one.
#[allow(clippy::too_many_arguments)]
fn chunk_module(
    scopes: &Scopes,
    inferred: &Inferred,
    ast: &ast::Ast,
    source: &str,
    file: &str,
    tag: &str,
    chunk_index: i64,
    env_var: Option<InternedString>,
    own_env: bool,
    library: &Library,
    slots: bool,
    declarations: &mut Vec<TypedNode<TypedDeclaration>>,
    hooks: &mut Vec<ShapeLayout>,
    registry: &mut zyntax_typed_ast::TypeRegistry,
    debug: DebugMode,
    debug_names: (&str, &str),
    stripped: bool,
) -> Result<(String, Option<St>)> {
    let mut module = Module {
        scopes,
        inferred,
        types: library.types.clone(),
        chunk: file,
        tag: tag.to_string(),
        chunk_index,
        env_var,
        line_starts: line_starts_of(source),
        stripped,
        fallible: crate::fallible::FALLIBLE.iter().copied().collect(),
        reentrant: crate::fallible::REENTRANT.iter().copied().collect(),
        raising: None,
        bounded: None,
        functions: RefCell::new(Vec::new()),
        module_vars: RefCell::new(Vec::new()),
        facts: RefCell::new(HashMap::new()),
        finders: RefCell::new(HashMap::new()),
        reentrant_helpers: RefCell::new(HashSet::new()),
        sorts: RefCell::new(HashMap::new()),
        debug: debug.info(),
        layouts: shape_layouts(inferred, chunk_index, slots),
    };
    let span = Span::new(0, source.len());
    let lower = |module: &Module<'_>| -> Result<Vec<St>> {
        let mut main = Lowerer::new(module, CHUNK);
        main.returns = Returns::Dynamic;
        main.varargs = Some(intern("$varargs"));
        main.frames = module.debug.is_some();
        let mut statements = main.block(ast.nodes())?;
        if main.frames {
            statements.insert(0, main.debug_enter(CHUNK, span));
        }
        if main.entry_line {
            statements.insert(0, entry_line_save(span));
        }
        if let (Some(env), true) = (module.env_var, own_env) {
            statements.insert(
                0,
                assign(
                    var(env, Type::Any, span),
                    call("zl_globals_value", vec![], Type::Any, span),
                    span,
                ),
            );
        }
        if types::falls_through(ast.nodes()) {
            main.return_stmt(&[], span, &mut statements)?;
        }
        if main.frames {
            main.debug_function_records(CHUNK, 0, 0);
        }
        module.strip_line_restores(&mut statements);
        module.facts.borrow_mut().insert(
            CHUNK,
            RaiseFact {
                own: main.raised,
                checks_depth: false,
                dynamic: true,
                callees: main.raise_callees.clone(),
            },
        );
        Ok(statements)
    };
    lower(&module)?;
    let bounded = bounded_functions(&module.facts.borrow(), |f| {
        module.inferred.escaping.contains(&f) || module.scopes.func(f).escapes
    });
    let raising = raising_functions(&module.facts.borrow(), &bounded);
    module.bounded = Some(bounded);
    module.raising = Some(raising);
    module.functions.borrow_mut().clear();
    module.module_vars.borrow_mut().clear();
    module.facts.borrow_mut().clear();
    module.finders.borrow_mut().clear();
    module.reentrant_helpers.borrow_mut().clear();
    module.sorts.borrow_mut().clear();
    if let Some(debug) = &module.debug {
        debug.records.borrow_mut().clear();
        debug.sites.set(0);
    }
    let statements = lower(&module)?;
    let chunk_name = format!("lua${tag}chunk");
    let code_name = format!("{chunk_name}$fn");
    module.functions.borrow_mut().push(typed_function(
        &chunk_name,
        vec![parameter(intern("$varargs"), module.anys(), span)],
        Type::Any,
        statements,
        span,
    ));
    // Its record code: a variadic taking the packed arguments.
    module.functions.borrow_mut().push(typed_function(
        &code_name,
        vec![
            parameter(intern("env"), module.anys(), span),
            parameter(intern("packed"), Type::Any, span),
        ],
        Type::Any,
        vec![ret(
            Some(call(
                &chunk_name,
                vec![call(
                    "zl_values",
                    vec![var(intern("packed"), Type::Any, span)],
                    module.anys(),
                    span,
                )],
                Type::Any,
                span,
            )),
            span,
        )],
        span,
    ));
    let register = module
        .debug
        .is_some()
        .then(|| debug_register(&module, debug_names, span));
    if let Some(env) = env_var {
        module.declare_module_var(env, Ty::Any);
    }
    declare(&module, declarations);
    module.declare_shapes(declarations, registry, hooks);
    Ok((code_name, register))
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

impl DebugMode {
    fn of(scopes: &Scopes) -> Self {
        DebugMode {
            on: scopes.debug,
            locals: scopes.debug_getlocal || scopes.debug_setlocal,
            setlocal: scopes.debug_setlocal,
            rebinds: scopes.debug_rebinds,
        }
    }

    fn join(self, other: DebugMode) -> Self {
        DebugMode {
            on: self.on || other.on,
            locals: self.locals || other.locals,
            setlocal: self.setlocal || other.setlocal,
            rebinds: self.rebinds || other.rebinds,
        }
    }

    fn info(self) -> Option<DebugInfo> {
        self.on.then(|| DebugInfo {
            locals: self.locals,
            setlocal: self.setlocal,
            ..Default::default()
        })
    }

    /// Kept for the chunks `load` compiles while the program runs.
    fn remember(self) {
        let bits = self.on as u8
            | (self.locals as u8) << 1
            | (self.setlocal as u8) << 2
            | (self.rebinds as u8) << 3;
        PROGRAM_DEBUG.store(bits, std::sync::atomic::Ordering::Relaxed);
    }

    fn program() -> Self {
        let bits = PROGRAM_DEBUG.load(std::sync::atomic::Ordering::Relaxed);
        DebugMode {
            on: bits & 1 != 0,
            locals: bits & 2 != 0,
            setlocal: bits & 4 != 0,
            rebinds: bits & 8 != 0,
        }
    }
}

/// The debug mode of the program running, for chunks it loads.
static PROGRAM_DEBUG: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The call registering a chunk with the debug library: its number,
/// its source and short names, what its lowering recorded, and its
/// upvalue accessor, which is added to its functions.
fn debug_register(module: &Module<'_>, (source, short): (&str, &str), span: Span) -> St {
    let accessor = upvalue_accessor(module, span);
    let name = upvalue_accessor_name(&module.tag);
    module.functions.borrow_mut().push(accessor);
    let meta = module
        .debug
        .as_ref()
        .map(|d| d.records.borrow().join("\n"))
        .unwrap_or_default();
    expr_stmt(call(
        "zl_dbg_register",
        vec![
            int_lit(module.chunk_index, span),
            str_lit(source, span),
            str_lit(short, span),
            str_lit(&meta, span),
            call(
                "zl_func_of",
                vec![
                    code_of(&name, span),
                    int_lit(library::debug::UP_ARITY, span),
                ],
                Type::Any,
                span,
            ),
        ],
        prim(PrimitiveType::Unit),
        span,
    ))
}

/// Statements returning a chunk as a function value: its record code,
/// variadic, and its key, kept for the debug library when the program
/// keeps its call stack.
fn chunk_value(code_name: &str, chunk_index: i64, debug: bool, anys: Type, span: Span) -> Vec<St> {
    let key = chunk_index << 32;
    let record = call(
        "zb_func_new",
        vec![
            code_of(code_name, span),
            int_lit(VARIADIC_ARITY, span),
            node(
                TypedExpression::Array(vec![call(
                    "zb_box_i64",
                    vec![int_lit(key, span)],
                    Type::Any,
                    span,
                )]),
                anys,
                span,
            ),
        ],
        Type::Any,
        span,
    );
    let name = intern("$chunk");
    let mut out = vec![let_(name, Type::Any, record, span)];
    if debug {
        out.push(expr_stmt(call(
            "zl_dbg_closure",
            vec![int_lit(key, span), var(name, Type::Any, span)],
            prim(PrimitiveType::Unit),
            span,
        )));
    }
    out.push(ret(Some(var(name, Type::Any, span)), span));
    out
}

/// The import every program links the library through.
fn library_import() -> TypedNode<TypedDeclaration> {
    TypedNode::new(
        TypedDeclaration::Import(zyntax_typed_ast::typed_ast::TypedImport {
            language: Some(intern("lua")),
            module_path: vec![intern(crate::policy::LIBRARY_MODULE)],
            items: Vec::new(),
            span: Span::new(0, 0),
        }),
        Type::Unknown,
        Span::new(0, 0),
    )
}

/// The name of a chunk's upvalue accessor.
fn upvalue_accessor_name(tag: &str) -> String {
    format!("lua${tag}dbg$upvalues")
}

/// A chunk's upvalue accessor, as a function value's code: given a
/// function's key, the function, the upvalue's number, a value and
/// what is asked (`debug::UP_*`), it reads or writes the upvalue where
/// the function keeps it (a cell or a copy in its record, a module
/// variable, the environment), gives its identity, joins it to
/// another's cell or hands over its own.
fn upvalue_accessor(module: &Module<'_>, span: Span) -> TypedFunction {
    use crate::library::debug::{UP_CELL, UP_GET, UP_ID, UP_JOIN, UP_SET};
    use crate::scope::Upvalue;
    let i64_t = prim(PrimitiveType::I64);
    let bool_t = prim(PrimitiveType::Bool);
    let anys = module.anys();
    let arg = |i: usize| var(intern(&format!("a{i}")), Type::Any, span);
    let int_of = |x: Node| call("zb_box_get_i64", vec![x], i64_t.clone(), span);
    let is = |x: Node, v: i64| binary(BinaryOp::Eq, x, int_lit(v, span), bool_t.clone(), span);
    let fid = || var(intern("fid"), i64_t.clone(), span);
    let n = || var(intern("n"), i64_t.clone(), span);
    let how = || var(intern("how"), i64_t.clone(), span);
    let rec = || var(intern("rec"), anys.clone(), span);
    let value = arg(3);
    let mut lowerer = Lowerer::new(module, CHUNK);
    // An upvalue's identity as a value: a light userdata, equal to
    // another exactly when the two identities are.
    let upvalue_id = |identity: Node| {
        call(
            "zb_box_instance_raw",
            vec![identity, int32_lit(library::light_tag() as i32, span)],
            Type::Any,
            span,
        )
    };
    // An identity for an upvalue with no cell: one per variable of the
    // chunk, never an address.
    let constant_id = |k: i64| upvalue_id(int_lit(((module.chunk_index + 1) << 40) | k, span));
    let mut body = vec![
        let_(
            intern("fid"),
            i64_t.clone(),
            binary(
                BinaryOp::BitAnd,
                int_of(arg(0)),
                int_lit(0xffff_ffff, span),
                i64_t.clone(),
                span,
            ),
            span,
        ),
        let_(
            intern("rec"),
            anys.clone(),
            call("zb_unbox_list_raw_any", vec![arg(1)], anys.clone(), span),
            span,
        ),
        let_(intern("n"), i64_t.clone(), int_of(arg(2)), span),
        let_(intern("how"), i64_t.clone(), int_of(arg(4)), span),
    ];
    for (k, info) in module.scopes.funcs.iter().enumerate() {
        if info.upvalues.is_empty() {
            continue;
        }
        let mut arms = Vec::new();
        for (i, up) in info.upvalues.iter().enumerate() {
            let mut answers: Vec<St> = Vec::new();
            match up {
                Upvalue::Var(v) => {
                    let vinfo = module.scopes.var(*v);
                    if vinfo.is_module_var() {
                        let ty = module.inferred.var(*v).settled();
                        let symbol = module.module_local_symbol(*v);
                        let read = lowerer.coerce(
                            Val {
                                node: var(symbol, module.ir(ty), span),
                                ty,
                            },
                            Ty::Any,
                        );
                        let stored = lowerer.coerce(
                            Val {
                                node: value.clone(),
                                ty: Ty::Any,
                            },
                            ty,
                        );
                        answers.push(if_(
                            is(how(), UP_GET),
                            vec![ret(Some(read), span)],
                            None,
                            span,
                        ));
                        answers.push(if_(
                            is(how(), UP_SET),
                            vec![assign(var(symbol, module.ir(ty), span), stored, span)],
                            None,
                            span,
                        ));
                        answers.push(if_(
                            is(how(), UP_ID),
                            vec![ret(Some(constant_id(v.0 as i64 + 1)), span)],
                            None,
                            span,
                        ));
                    } else if let Some(j) = info.captures.iter().position(|c| c == v) {
                        let slot = || {
                            index(
                                rec(),
                                int_lit((RECORD_CELLS_AT + j) as i64, span),
                                Type::Any,
                                span,
                            )
                        };
                        if vinfo.needs_cell() {
                            let cell = || {
                                index(
                                    call("zb_unbox_list_raw_any", vec![slot()], anys.clone(), span),
                                    int_lit(0, span),
                                    Type::Any,
                                    span,
                                )
                            };
                            answers.push(if_(
                                is(how(), UP_GET),
                                vec![ret(Some(cell()), span)],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                is(how(), UP_SET),
                                vec![assign(cell(), value.clone(), span)],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                is(how(), UP_ID),
                                vec![ret(
                                    Some(upvalue_id(call(
                                        "zb_unbox_instance_raw",
                                        vec![slot()],
                                        i64_t.clone(),
                                        span,
                                    ))),
                                    span,
                                )],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                binary(
                                    BinaryOp::And,
                                    is(how(), UP_JOIN),
                                    binary(
                                        BinaryOp::Ne,
                                        value.clone(),
                                        nil(span),
                                        bool_t.clone(),
                                        span,
                                    ),
                                    bool_t.clone(),
                                    span,
                                ),
                                vec![assign(slot(), value.clone(), span)],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                is(how(), UP_CELL),
                                vec![ret(Some(slot()), span)],
                                None,
                                span,
                            ));
                        } else {
                            // A chunk that names `debug.upvalueid` keeps
                            // every capture in a cell, so a copy is asked
                            // its identity only from another chunk: it
                            // answers the variable's, one for all its
                            // closure instances.
                            answers.push(if_(
                                is(how(), UP_GET),
                                vec![ret(Some(slot()), span)],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                is(how(), UP_SET),
                                vec![assign(slot(), value.clone(), span)],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                is(how(), UP_ID),
                                vec![ret(Some(constant_id(v.0 as i64 + 1)), span)],
                                None,
                                span,
                            ));
                        }
                    }
                }
                Upvalue::Env => {
                    let env = match module.env_var {
                        Some(env) => var(env, Type::Any, span),
                        None => call("zl_globals_value", vec![], Type::Any, span),
                    };
                    answers.push(if_(
                        is(how(), UP_GET),
                        vec![ret(Some(env), span)],
                        None,
                        span,
                    ));
                    if let Some(env) = module.env_var {
                        answers.push(if_(
                            is(how(), UP_SET),
                            vec![assign(var(env, Type::Any, span), value.clone(), span)],
                            None,
                            span,
                        ));
                    }
                    answers.push(if_(
                        is(how(), UP_ID),
                        vec![ret(Some(constant_id(0)), span)],
                        None,
                        span,
                    ));
                }
            }
            answers.push(ret(Some(nil(span)), span));
            arms.push(if_(is(n(), i as i64 + 1), answers, None, span));
        }
        body.push(if_(is(fid(), k as i64), arms, None, span));
    }
    body.push(ret(Some(nil(span)), span));
    let mut params = vec![parameter(intern("env"), anys, span)];
    params.extend(
        (0..library::debug::UP_ARITY as usize)
            .map(|i| parameter(intern(&format!("a{i}")), Type::Any, span)),
    );
    typed_function(
        &upvalue_accessor_name(&module.tag),
        params,
        Type::Any,
        body,
        span,
    )
}

/// The name of the program's function for a library hook.
fn shape_hook_name(hook: &str) -> String {
    format!("lua$shape${hook}")
}

/// The hooks the library reads and writes slots through, over every
/// slotted shape of the program: each is a chain over the shapes'
/// numbers, then over the slots. In the order of
/// [`library::SHAPE_HOOKS`]; none when the program has no slotted
/// shape.
fn shape_hooks(module: &Module<'_>, layouts: &[ShapeLayout], span: Span) -> Vec<TypedFunction> {
    if layouts.is_empty() {
        return Vec::new();
    }
    let table_t = module.ir(Ty::Table);
    let i64_t = prim(PrimitiveType::I64);
    let str_t = prim(PrimitiveType::String);
    let bool_t = prim(PrimitiveType::Bool);
    let t = || var(intern("t"), table_t.clone(), span);
    let i = || var(intern("i"), i64_t.clone(), span);
    let shape_is = |gid: i64| {
        binary(
            BinaryOp::Eq,
            field(t(), "shape", i64_t.clone(), span),
            int_lit(gid, span),
            bool_t.clone(),
            span,
        )
    };
    let slot_is = |bit: usize| {
        binary(
            BinaryOp::Eq,
            i(),
            int_lit(bit as i64, span),
            bool_t.clone(),
            span,
        )
    };
    let mut lowerer = Lowerer::new(module, CHUNK);
    let param = |name: &str, ty: Type| parameter(intern(name), ty, span);

    // zl_shape_index(t, name) -> i64
    let mut index_body = Vec::new();
    for layout in layouts {
        let mut arms = Vec::new();
        for slot in &layout.slots {
            arms.push(if_(
                call(
                    "zb_str_eq",
                    vec![
                        var(intern("name"), str_t.clone(), span),
                        str_lit(&slot.name, span),
                    ],
                    bool_t.clone(),
                    span,
                ),
                vec![ret(Some(int_lit(slot.bit as i64, span)), span)],
                None,
                span,
            ));
        }
        arms.push(ret(Some(int_lit(-1, span)), span));
        index_body.push(if_(shape_is(layout.gid), arms, None, span));
    }
    index_body.push(ret(Some(int_lit(-1, span)), span));
    let index = typed_function(
        &shape_hook_name("index"),
        vec![param("t", table_t.clone()), param("name", str_t.clone())],
        i64_t.clone(),
        index_body,
        span,
    );

    // zl_shape_load(t, i) -> Any
    let mut load_body = Vec::new();
    for layout in layouts {
        let mut arms = Vec::new();
        for slot in &layout.slots {
            let read = lowerer.slot_read(&t(), layout, slot, span);
            let boxed = lowerer.coerce(read, Ty::Any);
            let present = lowerer.slot_present(&t(), slot, span);
            arms.push(if_(
                slot_is(slot.bit),
                vec![ret(
                    Some(if_value(present, boxed, nil(span), Type::Any, span)),
                    span,
                )],
                None,
                span,
            ));
        }
        load_body.push(if_(shape_is(layout.gid), arms, None, span));
    }
    load_body.push(ret(Some(nil(span)), span));
    let load = typed_function(
        &shape_hook_name("load"),
        vec![param("t", table_t.clone()), param("i", i64_t.clone())],
        Type::Any,
        load_body,
        span,
    );

    // zl_shape_store(t, i, v) -> i64: 0 stored, 1 not of the slot's kind
    let v = || var(intern("v"), Type::Any, span);
    let category = || call("zb_any_category", vec![v()], i64_t.clone(), span);
    let is_cat = |c: i64| {
        binary(
            BinaryOp::Eq,
            category(),
            int_lit(c, span),
            bool_t.clone(),
            span,
        )
    };
    let mut store_body = Vec::new();
    for layout in layouts {
        let mut arms = Vec::new();
        for slot in &layout.slots {
            // nil clears the slot's bit; nothing else is written.
            let mask_off = binary(
                BinaryOp::BitAnd,
                lowerer.present_of(&t(), span),
                int_lit(!slot.mask(), span),
                i64_t.clone(),
                span,
            );
            let mut on_nil = vec![assign(lowerer.present_of(&t(), span), mask_off, span)];
            if slot.kind == SlotKind::Any {
                let shaped = as_shape(t(), module.shape_ty(layout), span);
                on_nil.push(assign(
                    field(shaped, &format!("s{}", slot.bit), Type::Any, span),
                    nil(span),
                    span,
                ));
            }
            on_nil.push(ret(Some(int_lit(0, span)), span));
            let fits = match slot.kind {
                SlotKind::Int => binary(
                    BinaryOp::Or,
                    is_cat(library::INT),
                    is_cat(library::UINT),
                    bool_t.clone(),
                    span,
                ),
                SlotKind::Float => is_cat(library::FLOAT),
                SlotKind::Number => binary(
                    BinaryOp::Or,
                    binary(
                        BinaryOp::Or,
                        is_cat(library::INT),
                        is_cat(library::UINT),
                        bool_t.clone(),
                        span,
                    ),
                    is_cat(library::FLOAT),
                    bool_t.clone(),
                    span,
                ),
                SlotKind::Bool => is_cat(library::BOOL),
                SlotKind::Str => is_cat(library::STR),
                SlotKind::Table => binary(
                    BinaryOp::Eq,
                    cast(
                        call("zb_box_tag", vec![v()], prim(PrimitiveType::I32), span),
                        i64_t.clone(),
                        span,
                    ),
                    int_lit(library::table_tag(), span),
                    bool_t.clone(),
                    span,
                ),
                SlotKind::Scalar => binary(
                    BinaryOp::Lt,
                    category(),
                    int_lit(library::STR, span),
                    bool_t.clone(),
                    span,
                ),
                SlotKind::Any => bool_lit(true, span),
            };
            let stored = Val {
                node: v(),
                ty: Ty::Any,
            };
            let mut write = lowerer.slot_store(&t(), layout, slot, stored, false, span);
            write.push(ret(Some(int_lit(0, span)), span));
            arms.push(if_(
                slot_is(slot.bit),
                vec![
                    if_(
                        binary(BinaryOp::Eq, v(), nil(span), bool_t.clone(), span),
                        on_nil,
                        None,
                        span,
                    ),
                    if_(
                        fits,
                        write,
                        Some(vec![ret(Some(int_lit(1, span)), span)]),
                        span,
                    ),
                ],
                None,
                span,
            ));
        }
        store_body.push(if_(shape_is(layout.gid), arms, None, span));
    }
    store_body.push(ret(Some(int_lit(1, span)), span));
    let store = typed_function(
        &shape_hook_name("store"),
        vec![
            param("t", table_t.clone()),
            param("i", i64_t.clone()),
            param("v", Type::Any),
        ],
        i64_t.clone(),
        store_body,
        span,
    );

    // zl_shape_count(t) -> i64
    let mut count_body = Vec::new();
    for layout in layouts {
        count_body.push(if_(
            shape_is(layout.gid),
            vec![ret(Some(int_lit(layout.slots.len() as i64, span)), span)],
            None,
            span,
        ));
    }
    count_body.push(ret(Some(int_lit(0, span)), span));
    let count = typed_function(
        &shape_hook_name("count"),
        vec![param("t", table_t.clone())],
        i64_t.clone(),
        count_body,
        span,
    );

    // zl_shape_key(t, i) -> String
    let mut key_body = Vec::new();
    for layout in layouts {
        let mut arms = Vec::new();
        for slot in &layout.slots {
            arms.push(if_(
                slot_is(slot.bit),
                vec![ret(Some(str_lit(&slot.name, span)), span)],
                None,
                span,
            ));
        }
        key_body.push(if_(shape_is(layout.gid), arms, None, span));
    }
    key_body.push(ret(Some(str_lit("", span)), span));
    let key = typed_function(
        &shape_hook_name("key"),
        vec![param("t", table_t), param("i", i64_t.clone())],
        str_t,
        key_body,
        span,
    );
    vec![index, load, store, count, key]
}

/// A chunk `load` compiles while the program runs, as a program of its
/// own: the chunk's function, and `lua$l<k>$init(env) -> Any`, which
/// registers the chunk's name for positions, takes the globals table
/// the chunk reads (the program's when `env` is nil) and returns the
/// chunk as a function value.
pub(crate) fn loaded_program(
    ast: &ast::Ast,
    source: &str,
    chunk_name: &str,
    index: i64,
    stripped: bool,
    library: &Library,
) -> Result<TypedProgram> {
    let (scopes, inferred) = loaded_types(ast);
    NAMED.with(|named| *named.borrow_mut() = Some(Default::default()));
    let lowered = loaded_declarations(
        &scopes, &inferred, ast, source, chunk_name, index, stripped, library,
    );
    let named = NAMED
        .with(|named| named.borrow_mut().take())
        .unwrap_or_default();
    let (mut declarations, registry) = lowered?;
    // The library's functions the chunk names, and no others: the rest
    // are already where the chunk will run.
    let mut import = library_import();
    if let TypedDeclaration::Import(i) = &mut import.node {
        i.items = named
            .into_iter()
            .map(|name| zyntax_typed_ast::typed_ast::TypedImportItem::Named { name, alias: None })
            .collect();
    }
    declarations.push(import);
    let span = Span::new(0, source.len());
    Ok(TypedProgram {
        declarations,
        language: Some(intern("lua")),
        span,
        source_files: vec![zyntax_typed_ast::source::SourceFile::new(
            chunk_name.to_string(),
            source.to_string(),
        )],
        type_registry: registry,
    })
}

/// The scopes and types of a chunk `load` compiles.
pub(crate) fn loaded_types(ast: &ast::Ast) -> (Scopes, Inferred) {
    let mut scopes = crate::scope::resolve_loaded(ast, DebugMode::program().rebinds);
    scopes.dynamic_globals = true;
    scopes.len_meta = true;
    // The chunk is a function value, called by whoever `load` gave it to.
    scopes.funcs[CHUNK.0 as usize].escapes = true;
    let inferred = types::infer(&scopes, ast);
    (scopes, inferred)
}

/// The declarations of a loaded chunk: its functions and its `init`.
#[allow(clippy::too_many_arguments)]
fn loaded_declarations(
    scopes: &Scopes,
    inferred: &Inferred,
    ast: &ast::Ast,
    source: &str,
    chunk_name: &str,
    index: i64,
    stripped: bool,
    library: &Library,
) -> Result<(
    Vec<TypedNode<TypedDeclaration>>,
    zyntax_typed_ast::TypeRegistry,
)> {
    let tag = format!("l{index}$");
    let env_var = intern(&format!("lua$l{index}$env"));
    let mut declarations = Vec::new();
    // Its tables are plain: the program's hooks know nothing of a
    // chunk's shapes, and it defines none of its own.
    let mut registry = library.type_registry.clone();
    let mut hooks = Vec::new();
    // A chunk of a program that keeps its call stack keeps it too, and
    // a chunk that reaches the debug library has the chunks loaded
    // after it keep what it may ask of their functions.
    let debug = DebugMode::of(scopes).join(DebugMode::program());
    debug.remember();
    let (code_name, register) = chunk_module(
        scopes,
        inferred,
        ast,
        source,
        chunk_name,
        &tag,
        index,
        Some(env_var),
        false,
        library,
        false,
        &mut declarations,
        &mut hooks,
        &mut registry,
        debug,
        (chunk_name, chunk_name),
        stripped,
    )?;
    let span = Span::new(0, source.len());
    let env = intern("env");
    let mut statements = vec![expr_stmt(call(
        "zl_chunk_add",
        vec![
            string_literal(crate::source_bytes(chunk_name).into_owned(), span),
            int_lit(index, span),
        ],
        prim(PrimitiveType::Unit),
        span,
    ))];
    statements.extend(register);
    statements.push(assign(
        var(env_var, Type::Any, span),
        var(env, Type::Any, span),
        span,
    ));
    statements.extend(chunk_value(
        &code_name,
        index,
        debug.on,
        library.types.anys(),
        span,
    ));
    let init = typed_function(
        &format!("lua${tag}init"),
        vec![parameter(env, Type::Any, span)],
        Type::Any,
        statements,
        span,
    );
    declarations.push(TypedNode::new(
        TypedDeclaration::Function(init),
        Type::Unknown,
        span,
    ));
    Ok((declarations, registry))
}

/// The entry the host reports an uncaught table error through, in the
/// program [`error_text_program`] builds.
pub(crate) const ERROR_TEXT_ENTRY: &str = "lua$error_text";

/// A program of one function, `lua$error_text(err) -> Any`, entering
/// the library's text for an uncaught table error. Compiled by the host
/// only when such an error is reported, so no chunk reaches it.
pub(crate) fn error_text_program(library: &Library) -> TypedProgram {
    let span = Span::new(0, 0);
    let err = intern("err");
    let entry = typed_function(
        ERROR_TEXT_ENTRY,
        vec![parameter(err, Type::Any, span)],
        Type::Any,
        vec![ret(
            Some(call(
                library::ERROR_TEXT,
                vec![var(err, Type::Any, span)],
                Type::Any,
                span,
            )),
            span,
        )],
        span,
    );
    TypedProgram {
        declarations: vec![
            TypedNode::new(TypedDeclaration::Function(entry), Type::Unknown, span),
            library_import(),
        ],
        language: Some(intern("lua")),
        span,
        source_files: Vec::new(),
        type_registry: library.type_registry.clone(),
    }
}

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

/// The whole chunk as a program.
pub(crate) fn program(
    ast: &ast::Ast,
    source: &str,
    file: &str,
    library: &Library,
    entry_kind: Entry,
) -> Result<TypedProgram> {
    let started = std::time::Instant::now();
    let mut scopes = crate::scope::resolve(ast);
    let (mut loaded, all_found) = load_required(&scopes.requires, file)?;
    // Files share their globals through the table, as does code run
    // at run time, which may write any global the program reads.
    let shared = entry_kind == Entry::Host
        || !loaded.is_empty()
        || !all_found
        || scopes.dynamic_code
        || loaded.iter().any(|m| m.scopes.dynamic_code);
    if shared {
        scopes.dynamic_globals = true;
        for m in &mut loaded {
            m.scopes.dynamic_globals = true;
        }
    }
    // A table one file hands another may get its metatable there, where
    // the first file's types do not see it.
    let sets_metatables = |s: &Scopes| {
        s.unseen_metatables || s.globals.contains("setmetatable") || s.globals.contains("debug")
    };
    if (!loaded.is_empty() || !all_found)
        && (!all_found
            || sets_metatables(&scopes)
            || loaded.iter().any(|m| sets_metatables(&m.scopes)))
    {
        scopes.unseen_metatables = true;
        for m in &mut loaded {
            m.scopes.unseen_metatables = true;
        }
    }
    let len_meta = len_meta_possible(source, &scopes, &loaded, all_found);
    scopes.len_meta = len_meta;
    for m in &mut loaded {
        m.scopes.len_meta = len_meta;
    }
    crate::trace_phase("scopes", started);
    let started = std::time::Instant::now();
    let inferred = types::infer(&scopes, ast);
    let module_inferred: Vec<Inferred> = loaded
        .iter()
        .map(|m| types::infer(&m.scopes, &m.ast))
        .collect();
    crate::trace_phase("infer", started);
    let started = std::time::Instant::now();
    // Every file of a program that reaches the debug library keeps the
    // call stack, and so does every chunk it loads.
    let debug = loaded.iter().fold(DebugMode::of(&scopes), |d, m| {
        d.join(DebugMode::of(&m.scopes))
    });
    debug.remember();
    let line_starts = line_starts_of(source);
    // A program that assigns `_ENV` has an environment of its own,
    // the globals table until then.
    let env_var = scopes
        .global_writes
        .contains_key("_ENV")
        .then(|| intern(MAIN_ENV));
    let mut module = Module {
        scopes: &scopes,
        inferred: &inferred,
        types: library.types.clone(),
        chunk: file,
        tag: String::new(),
        chunk_index: 0,
        env_var,
        line_starts,
        stripped: false,
        fallible: crate::fallible::FALLIBLE.iter().copied().collect(),
        reentrant: crate::fallible::REENTRANT.iter().copied().collect(),
        raising: None,
        bounded: None,
        functions: RefCell::new(Vec::new()),
        module_vars: RefCell::new(Vec::new()),
        facts: RefCell::new(HashMap::new()),
        finders: RefCell::new(HashMap::new()),
        reentrant_helpers: RefCell::new(HashSet::new()),
        sorts: RefCell::new(HashMap::new()),
        debug: debug.info(),
        layouts: shape_layouts(&inferred, 0, true),
    };
    let span = Span::new(0, source.len());
    // Lowered twice: the first time finds which functions may raise, so
    // the second checks after calls to those alone.
    let lower_chunk = |module: &Module<'_>| -> Result<Vec<St>> {
        let mut main = Lowerer::new(module, CHUNK);
        main.frames = module.debug.is_some();
        main.returns = Returns::Fixed(Vec::new());
        // `...` at the main chunk: the script's arguments, in a module
        // variable since a segment is a function of its own.
        main.varargs = Some(intern(MAIN_VARARGS));
        let mut statements = if scopes.split_chunk {
            // Each segment is a function; the chunk calls them in turn
            // and stops at an error or a `return`.
            let segments = main.segments(ast.nodes())?;
            let unit = prim(PrimitiveType::Unit);
            let mut driver = Vec::new();
            let returned = module
                .module_vars
                .borrow()
                .iter()
                .any(|(n, _)| *n == intern(RETURNED));
            for (k, mut body) in segments.into_iter().enumerate() {
                if main.entry_line {
                    body.insert(0, entry_line_save(span));
                }
                body.push(ret(None, span));
                module.strip_line_restores(&mut body);
                let name = format!("{CHUNK_FN}${k}");
                module.functions.borrow_mut().push(typed_function(
                    &name,
                    Vec::new(),
                    unit.clone(),
                    body,
                    span,
                ));
                driver.push(expr_stmt(call(&name, vec![], unit.clone(), span)));
                driver.push(if_(
                    binary(
                        BinaryOp::Ne,
                        Lowerer::pending(span),
                        nil(span),
                        prim(PrimitiveType::Bool),
                        span,
                    ),
                    vec![ret(None, span)],
                    None,
                    span,
                ));
                if returned {
                    driver.push(if_(
                        var(intern(RETURNED), prim(PrimitiveType::Bool), span),
                        vec![ret(None, span)],
                        None,
                        span,
                    ));
                }
            }
            driver.push(ret(None, span));
            driver
        } else {
            let mut statements = main.block(ast.nodes())?;
            if main.entry_line {
                statements.insert(0, entry_line_save(span));
            }
            if types::falls_through(ast.nodes()) {
                statements.push(ret(None, span));
            }
            module.strip_line_restores(&mut statements);
            statements
        };
        let presets = main.preset_globals(span);
        statements.splice(0..0, presets);
        if main.frames {
            statements.insert(0, main.debug_enter(CHUNK, span));
            main.debug_function_records(CHUNK, 0, 0);
        }
        if let Some(env) = module.env_var {
            statements.insert(
                0,
                assign(
                    var(env, Type::Any, span),
                    call("zl_globals_value", vec![], Type::Any, span),
                    span,
                ),
            );
        }
        statements.insert(
            0,
            assign(
                var(intern(MAIN_VARARGS), module.anys(), span),
                call("zl_script_args", vec![], module.anys(), span),
                span,
            ),
        );
        module.facts.borrow_mut().insert(
            CHUNK,
            RaiseFact {
                own: main.raised,
                checks_depth: false,
                dynamic: true,
                callees: main.raise_callees.clone(),
            },
        );
        Ok(statements)
    };
    lower_chunk(&module)?;
    let bounded = bounded_functions(&module.facts.borrow(), |f| {
        module.inferred.escaping.contains(&f) || module.scopes.func(f).escapes
    });
    let raising = raising_functions(&module.facts.borrow(), &bounded);
    module.bounded = Some(bounded);
    module.raising = Some(raising);
    module.functions.borrow_mut().clear();
    module.module_vars.borrow_mut().clear();
    module.facts.borrow_mut().clear();
    module.finders.borrow_mut().clear();
    module.reentrant_helpers.borrow_mut().clear();
    module.sorts.borrow_mut().clear();
    if let Some(debug) = &module.debug {
        debug.records.borrow_mut().clear();
        debug.sites.set(0);
    }
    crate::trace_phase("lower 1", started);
    let started = std::time::Instant::now();
    let statements = lower_chunk(&module)?;
    crate::trace_phase("lower 2", started);
    let chunk_fn = typed_function(
        CHUNK_FN,
        Vec::new(),
        prim(PrimitiveType::Unit),
        statements,
        span,
    );
    if let Some(env) = module.env_var {
        module.declare_module_var(env, Ty::Any);
    }
    let register = debug.on.then(|| {
        let source_name = format!("@{file}");
        debug_register(&module, (&source_name, &chunk_id(module.chunk)), span)
    });
    let mut declarations = Vec::new();
    declare(&module, &mut declarations);
    let mut registry = library.type_registry.clone();
    let mut hooks = Vec::new();
    module.declare_shapes(&mut declarations, &mut registry, &mut hooks);
    declarations.push(TypedNode::new(
        TypedDeclaration::Variable(TypedVariable {
            name: intern(MAIN_VARARGS),
            ty: module.anys(),
            mutability: Mutability::Mutable,
            initializer: None,
            visibility: Visibility::Public,
        }),
        Type::Unknown,
        Span::new(0, 0),
    ));

    // Each required file is a chunk of its own, a function the program
    // enters through `package.preload`, and named in positions by its
    // number.
    let mut preloads: Vec<St> = Vec::new();
    preloads.extend(register);
    for (k, m) in loaded.iter().enumerate() {
        let tag = format!("m${}$", m.name.replace('.', "$"));
        let env_var = m
            .scopes
            .global_writes
            .contains_key("_ENV")
            .then(|| intern(&format!("lua${tag}env")));
        // Positions name the file as `require` found it.
        let found_as = format!("./{}.lua", m.name.replace('.', "/"));
        let source_name = format!("@{found_as}");
        crate::dump::note_chunk(k as i64 + 1, &m.source, &source_name, false);
        let (code_name, register) = chunk_module(
            &m.scopes,
            &module_inferred[k],
            &m.ast,
            &m.source,
            &m.file,
            &tag,
            k as i64 + 1,
            env_var,
            true,
            library,
            true,
            &mut declarations,
            &mut hooks,
            &mut registry,
            debug,
            (&source_name, &chunk_id(&found_as)),
            false,
        )?;
        preloads.push(expr_stmt(call(
            "zl_chunk_add",
            vec![
                str_lit(&chunk_id(&found_as), span),
                int_lit(k as i64 + 1, span),
            ],
            prim(PrimitiveType::Unit),
            span,
        )));
        preloads.extend(register);
        // The chunk as a function value, under its key.
        let key = (k as i64 + 1) << 32;
        let record = call(
            "zb_func_new",
            vec![
                code_of(&code_name, span),
                int_lit(VARIADIC_ARITY, span),
                node(
                    TypedExpression::Array(vec![call(
                        "zb_box_i64",
                        vec![int_lit(key, span)],
                        Type::Any,
                        span,
                    )]),
                    module.anys(),
                    span,
                ),
            ],
            Type::Any,
            span,
        );
        let chunk_var = intern(&format!("$chunk{k}"));
        preloads.push(let_(chunk_var, Type::Any, record, span));
        if debug.on {
            preloads.push(expr_stmt(call(
                "zl_dbg_closure",
                vec![int_lit(key, span), var(chunk_var, Type::Any, span)],
                prim(PrimitiveType::Unit),
                span,
            )));
        }
        preloads.push(expr_stmt(call(
            "zl_preload_module",
            vec![str_lit(&m.name, span), var(chunk_var, Type::Any, span)],
            prim(PrimitiveType::Unit),
            span,
        )));
    }

    let hooks = shape_hooks(&module, &hooks, span);
    let mut entry_body = Vec::new();
    if !hooks.is_empty() {
        entry_body.push(expr_stmt(call(
            library::SHAPE_HOOKS_INSTALL,
            library::SHAPE_HOOKS
                .iter()
                .map(|hook| code_of(&shape_hook_name(hook), span))
                .collect(),
            prim(PrimitiveType::Unit),
            span,
        )));
    }
    entry_body.extend([
        assign(
            var(intern(library::CHUNK), prim(PrimitiveType::String), span),
            str_lit(&chunk_id(module.chunk), span),
            span,
        ),
        if scopes.dynamic_globals {
            stmt(
                TypedStatement::Block(TypedBlock {
                    statements: vec![
                        assign(
                            var(intern(library::GLOBALS), module.ir(Ty::Table), span),
                            call("zl_globals_table", vec![], module.ir(Ty::Table), span),
                            span,
                        ),
                        assign(
                            var(
                                intern(library::capi::SHARED),
                                prim(PrimitiveType::I64),
                                span,
                            ),
                            int_lit(1, span),
                            span,
                        ),
                    ],
                    span,
                }),
                span,
            )
        } else {
            stmt(
                TypedStatement::Block(TypedBlock {
                    statements: Vec::new(),
                    span,
                }),
                span,
            )
        },
    ]);
    entry_body.extend(preloads);
    if entry_kind == Entry::Program {
        entry_body.extend([
            expr_stmt(call(CHUNK_FN, vec![], prim(PrimitiveType::Unit), span)),
            expr_stmt(call(
                "zl_report_pending",
                vec![],
                prim(PrimitiveType::Unit),
                span,
            )),
        ]);
    }
    entry_body.push(ret(None, span));
    let entry = typed_function(
        match entry_kind {
            Entry::Program => ENTRY,
            Entry::Host => crate::HOST_ENTRY,
        },
        Vec::new(),
        prim(PrimitiveType::Unit),
        entry_body,
        span,
    );
    declarations.push(TypedNode::new(
        TypedDeclaration::Function(chunk_fn),
        Type::Unknown,
        span,
    ));
    declarations.push(TypedNode::new(
        TypedDeclaration::Function(entry),
        Type::Unknown,
        span,
    ));
    for hook in hooks {
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(hook),
            Type::Unknown,
            span,
        ));
    }
    // The library itself arrives by import: its declarations for
    // typing, its HIR to link against.
    declarations.push(library_import());
    let mut source_files = vec![zyntax_typed_ast::source::SourceFile::new(
        file.to_string(),
        source.to_string(),
    )];
    for m in &loaded {
        source_files.push(zyntax_typed_ast::source::SourceFile::new(
            m.file.clone(),
            m.source.clone(),
        ));
    }
    Ok(TypedProgram {
        declarations,
        language: Some(intern("lua")),
        span,
        source_files,
        type_registry: registry,
    })
}
