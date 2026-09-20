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
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
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
    /// A byte string: stored as a string is, and boxed under its own
    /// category, so nothing reads it as text.
    Bytes,
    /// An open file, in text or binary mode: the record the library
    /// keeps for it.
    File(Mode),
    None,
    /// A list whose elements are all of one kind.
    List(Elem),
    /// A tuple of the shape at this index of the shape table: the
    /// number of elements and the type of each, a value struct. A tuple
    /// no shape is settled for is a dynamic value.
    Tuple(u16),
    /// A dict of the shape at this index of the dict shape table: what
    /// its keys and values are typed as. Storage is dynamic whatever
    /// the shape; the shape types what is read out.
    Dict(u16),
    /// A set of dynamic values.
    Set,
    /// An instance of the module's class at this index, or None: the
    /// value is a pointer, and None is the null one. Reading through
    /// None raises AttributeError, where the lowering does not know the
    /// value is an instance.
    Class(u16),
    /// A generator: a fiber yielding dynamic values.
    Gen,
    /// A function value whose function is known: the closure at this
    /// index of the module's table. Carried as the record every function
    /// value is, so it is a dynamic value wherever one is needed; where
    /// it is called, the call is direct and typed.
    Closure(u16),
    /// A method of a list or an instance bound to a local, at this
    /// index of the module's table: the record every bound method is,
    /// so it is a dynamic value wherever one is needed; where it is
    /// called, the call is the method's on the receiver as it is.
    Bound(u16),
    /// A builtin function named as a value, the one at this index of
    /// [`BUILTIN_VALUES`]: `xrange = range`. A call through the name is
    /// the builtin's call.
    Builtin(u8),
    /// A dynamic value: a boxed `Any`.
    Object,
    #[default]
    Unknown,
}

/// What a file reads and writes: text as strings, binary as bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Mode {
    Text,
    Binary,
}

impl Mode {
    /// The mode of `open(..., mode)`: binary when the mode names `b`.
    pub(crate) fn of(mode: &str) -> Mode {
        if mode.contains('b') {
            Mode::Binary
        } else {
            Mode::Text
        }
    }

    /// What the file's contents are typed as.
    pub(crate) fn content(self) -> Ty {
        match self {
            Mode::Text => Ty::Str,
            Mode::Binary => Ty::Bytes,
        }
    }
}

/// The element kinds a list is instantiated for. Anything else in a
/// list is a dynamic value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Elem {
    Int,
    Float,
    Str,
    /// Instances of one class, held by address, so an element is the
    /// instance itself rather than a box to open.
    Class(u16),
    /// Tuples of one shape, held as the struct itself; the list's
    /// functions are generated for the shape.
    Tuple(u16),
    /// The storage of an `array.array` of this typecode: a list whose
    /// elements are stored at the typecode's width and read as the
    /// number the typecode stands for. The list is the array.
    Array(Code),
    Object,
}

/// A typecode of the `array` module, `u` and `w` aside: those hold
/// characters, which are strings here.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Code {
    B,
    UB,
    H,
    UH,
    I,
    UI,
    L,
    UL,
    Q,
    UQ,
    F,
    D,
}

impl Code {
    pub(crate) const ALL: [Code; 12] = [
        Code::B,
        Code::UB,
        Code::H,
        Code::UH,
        Code::I,
        Code::UI,
        Code::L,
        Code::UL,
        Code::Q,
        Code::UQ,
        Code::F,
        Code::D,
    ];

    /// The typecode spelled `letter`, if it is one an array can hold.
    pub(crate) fn of(letter: &str) -> Option<Code> {
        Code::ALL.iter().copied().find(|c| c.letter() == letter)
    }

    pub(crate) fn letter(self) -> &'static str {
        match self {
            Code::B => "b",
            Code::UB => "B",
            Code::H => "h",
            Code::UH => "H",
            Code::I => "i",
            Code::UI => "I",
            Code::L => "l",
            Code::UL => "L",
            Code::Q => "q",
            Code::UQ => "Q",
            Code::F => "f",
            Code::D => "d",
        }
    }

    /// The kind an element is stored as. `l` and `L` are the 8 bytes
    /// they are on every platform this runs on.
    pub(crate) fn storage(self) -> zyntax_builtins::Kind {
        use zyntax_builtins::Kind;
        match self {
            Code::B => Kind::I8,
            Code::UB => Kind::U8,
            Code::H => Kind::I16,
            Code::UH => Kind::U16,
            Code::I => Kind::I32,
            Code::UI => Kind::U32,
            Code::L | Code::Q => Kind::Int,
            Code::UL | Code::UQ => Kind::U64,
            Code::F => Kind::F32,
            Code::D => Kind::Float,
        }
    }

    /// What an element reads as.
    pub(crate) fn item(self) -> Ty {
        match self {
            Code::F | Code::D => Ty::Float,
            _ => Ty::Int,
        }
    }

    /// Whether storing an element narrows the number it reads as: a
    /// range check on an integer, a rounding on a float.
    pub(crate) fn narrows(self) -> bool {
        self.storage().wide() != self.storage()
    }

    /// Bytes per element, `a.itemsize`.
    pub(crate) fn itemsize(self) -> i64 {
        match self {
            Code::B | Code::UB => 1,
            Code::H | Code::UH => 2,
            Code::I | Code::UI | Code::F => 4,
            _ => 8,
        }
    }

    /// The tag a boxed array of this typecode carries.
    pub(crate) fn tag(self) -> i64 {
        zyntax_builtins::array_tag(self.storage(), self.letter().as_bytes()[0])
    }
}

impl Elem {
    /// The element kind a value of `ty` is stored as.
    pub(crate) fn of(ty: Ty) -> Elem {
        match ty {
            Ty::Int => Elem::Int,
            Ty::Float => Elem::Float,
            Ty::Str => Elem::Str,
            Ty::Class(k) => Elem::Class(k),
            Ty::Tuple(k) => Elem::Tuple(k),
            _ => Elem::Object,
        }
    }

    /// What an element reads as.
    pub(crate) fn ty(self) -> Ty {
        match self {
            Elem::Int => Ty::Int,
            Elem::Float => Ty::Float,
            Elem::Str => Ty::Str,
            Elem::Class(k) => Ty::Class(k),
            Elem::Tuple(k) => Ty::Tuple(k),
            Elem::Array(c) => c.item(),
            Elem::Object => Ty::Object,
        }
    }

    /// The typecode, for the storage of an array.
    pub(crate) fn code(self) -> Option<Code> {
        match self {
            Elem::Array(c) => Some(c),
            _ => None,
        }
    }

    /// The suffix of the library functions for this kind.
    pub(crate) fn suffix(self) -> String {
        match self {
            Elem::Int => "i64".to_string(),
            Elem::Float => "f64".to_string(),
            Elem::Str => "str".to_string(),
            Elem::Class(_) => "ptr".to_string(),
            Elem::Tuple(k) => tuple_suffix(k),
            Elem::Array(c) => c.storage().suffix().to_string(),
            Elem::Object => "any".to_string(),
        }
    }

    /// The tag a boxed list of this kind carries.
    pub(crate) fn list_tag(self) -> i64 {
        match self {
            Elem::Int => zyntax_builtins::Kind::Int.list_tag(),
            Elem::Float => zyntax_builtins::Kind::Float.list_tag(),
            Elem::Str => zyntax_builtins::Kind::Str.list_tag(),
            Elem::Class(_) => zyntax_builtins::Kind::Ptr.list_tag(),
            Elem::Tuple(k) => zyntax_builtins::lists::shape_list_tag(k),
            Elem::Array(c) => c.tag(),
            Elem::Object => zyntax_builtins::Kind::Any.list_tag(),
        }
    }
}

/// The suffix of the library functions generated for tuple shape `k`.
pub(crate) fn tuple_suffix(k: u16) -> String {
    format!("t{k}")
}

/// The builtin functions a name can be bound to and called through.
pub(crate) const BUILTIN_VALUES: &[&str] = &[
    "range",
    "len",
    "zip",
    "enumerate",
    "sorted",
    "reversed",
    "list",
    "tuple",
    "dict",
    "set",
    "frozenset",
    "min",
    "max",
    "sum",
    "abs",
    "int",
    "float",
    "str",
    "repr",
    "bool",
    "print",
    "any",
    "all",
    "map",
    "filter",
    "hash",
    "ord",
    "chr",
    "round",
    "isinstance",
    "next",
    "iter",
    "pow",
    "divmod",
    "id",
];

pub(crate) fn builtin_index(name: &str) -> Option<u8> {
    BUILTIN_VALUES
        .iter()
        .position(|b| *b == name)
        .map(|i| i as u8)
}

thread_local! {
    /// The tuple shapes of the program being compiled, by index. A
    /// shape is interned once and its index never changes, so a
    /// `Ty::Tuple` decided in one inference round names the same shape
    /// in the next and in the lowering.
    static TUPLE_SHAPES: std::cell::RefCell<Vec<Vec<Ty>>> = const { std::cell::RefCell::new(Vec::new()) };
    /// The shapes the lowering used as a list's element, which get the
    /// library's list functions generated for them.
    static TUPLE_LISTS: std::cell::RefCell<std::collections::BTreeSet<u16>> =
        const { std::cell::RefCell::new(std::collections::BTreeSet::new()) };
    /// The dict shapes of the program being compiled: key and value
    /// types, by index, interned like tuple shapes.
    static DICT_SHAPES: std::cell::RefCell<Vec<(Ty, Ty)>> = const { std::cell::RefCell::new(Vec::new()) };
    /// The storage kinds of the arrays the lowering used, as positions
    /// in `Kind::ALL`. A kind the library does not carry gets its list
    /// functions generated with the program; every kind gets its arms
    /// in the hooks the dynamic layer reaches a boxed array through.
    static ARRAY_KINDS: std::cell::RefCell<std::collections::BTreeSet<usize>> =
        const { std::cell::RefCell::new(std::collections::BTreeSet::new()) };
}

/// Forget every shape: the start of a program.
pub(crate) fn reset_tuple_shapes() {
    TUPLE_SHAPES.with(|t| t.borrow_mut().clear());
    TUPLE_LISTS.with(|t| t.borrow_mut().clear());
    DICT_SHAPES.with(|t| t.borrow_mut().clear());
    ARRAY_KINDS.with(|t| t.borrow_mut().clear());
}

/// Record that arrays stored as `kind` are used.
pub(crate) fn note_array_kind(kind: zyntax_builtins::Kind) {
    let index = zyntax_builtins::Kind::ALL
        .iter()
        .position(|k| *k == kind)
        .expect("an array storage kind");
    ARRAY_KINDS.with(|t| t.borrow_mut().insert(index));
}

/// The storage kinds noted so far, and those of arrays held in tuple
/// shapes, in tag order.
pub(crate) fn array_kinds() -> Vec<zyntax_builtins::Kind> {
    for shape in TUPLE_SHAPES.with(|t| t.borrow().clone()) {
        for field in shape {
            if let Ty::List(Elem::Array(c)) = field {
                note_array_kind(c.storage());
            }
        }
    }
    ARRAY_KINDS.with(|t| {
        t.borrow()
            .iter()
            .map(|&i| zyntax_builtins::Kind::ALL[i])
            .collect()
    })
}

/// The dict type with these key and value types.
pub(crate) fn dict_of(key: Ty, value: Ty) -> Ty {
    DICT_SHAPES.with(|t| {
        let mut table = t.borrow_mut();
        if let Some(k) = table.iter().position(|shape| *shape == (key, value)) {
            return Ty::Dict(k as u16);
        }
        let k = u16::try_from(table.len()).expect("fewer than 65536 dict shapes in a program");
        table.push((key, value));
        Ty::Dict(k)
    })
}

/// The key and value types of dict shape `k`.
pub(crate) fn dict_shape(k: u16) -> (Ty, Ty) {
    DICT_SHAPES.with(|t| t.borrow()[k as usize])
}

/// A dict of dynamic keys and values.
pub(crate) fn dynamic_dict() -> Ty {
    dict_of(Ty::Object, Ty::Object)
}

/// Record that a list of tuples of shape `k` is used.
pub(crate) fn note_tuple_list(k: u16) {
    TUPLE_LISTS.with(|t| t.borrow_mut().insert(k));
}

/// The shapes noted as list elements so far.
pub(crate) fn tuple_lists() -> std::collections::BTreeSet<u16> {
    TUPLE_LISTS.with(|t| t.borrow().clone())
}

/// How many shapes the program has interned.
pub(crate) fn tuple_shape_count() -> usize {
    TUPLE_SHAPES.with(|t| t.borrow().len())
}

/// The tuple type of these element types. The empty tuple is a dynamic
/// value: a struct of no fields has no value to hold. An element with
/// no value of its own to store (None) or none a field can hold (a
/// generator) is stored boxed.
pub(crate) fn tuple_of(elems: Vec<Ty>) -> Ty {
    if elems.is_empty() {
        return Ty::Object;
    }
    let elems: Vec<Ty> = elems
        .into_iter()
        .map(|t| match t {
            Ty::None | Ty::Gen => Ty::Object,
            other => other,
        })
        .collect();
    TUPLE_SHAPES.with(|t| {
        let mut table = t.borrow_mut();
        if let Some(k) = table.iter().position(|shape| *shape == elems) {
            return Ty::Tuple(k as u16);
        }
        let k = u16::try_from(table.len()).expect("fewer than 65536 tuple shapes in a program");
        table.push(elems);
        Ty::Tuple(k)
    })
}

/// The element types of the tuple shape `k`.
pub(crate) fn tuple_shape(k: u16) -> Vec<Ty> {
    TUPLE_SHAPES.with(|t| t.borrow()[k as usize].clone())
}

impl Ty {
    /// The join of two assignments to one name.
    pub(crate) fn join(self, other: Ty) -> Ty {
        match (self, other) {
            (Ty::Unknown, t) | (t, Ty::Unknown) => t,
            (a, b) if a == b => a,
            // An instance is a pointer, and None is the null one.
            (Ty::None, Ty::Class(k)) | (Ty::Class(k), Ty::None) => Ty::Class(k),
            // Two shapes of one arity join element by element.
            (Ty::Tuple(a), Ty::Tuple(b)) => {
                let (a, b) = (tuple_shape(a), tuple_shape(b));
                if a.len() != b.len() {
                    return Ty::Object;
                }
                tuple_of(a.into_iter().zip(b).map(|(x, y)| x.join(y)).collect())
            }
            // Two dicts join key with key and value with value.
            (Ty::Dict(a), Ty::Dict(b)) => {
                let ((ka, va), (kb, vb)) = (dict_shape(a), dict_shape(b));
                dict_of(ka.join(kb), va.join(vb))
            }
            // Two lists of shapes are one list seen before and after
            // its elements were typed; of two shapes decided apart
            // they are lists of two kinds.
            (Ty::List(_), Ty::List(_)) if self.refined_by(other) => other,
            (Ty::List(_), Ty::List(_)) if other.refined_by(self) => self,
            _ => Ty::Object,
        }
    }

    /// Whether `other` is this type with some of what was still
    /// undecided in it decided.
    fn refined_by(self, other: Ty) -> bool {
        match (self, other) {
            (a, b) if a == b => true,
            (Ty::Unknown, _) => true,
            (Ty::Tuple(a), Ty::Tuple(b)) => {
                let (a, b) = (tuple_shape(a), tuple_shape(b));
                a.len() == b.len() && a.into_iter().zip(b).all(|(x, y)| x.refined_by(y))
            }
            (Ty::Dict(a), Ty::Dict(b)) => {
                let ((ka, va), (kb, vb)) = (dict_shape(a), dict_shape(b));
                ka.refined_by(kb) && va.refined_by(vb)
            }
            (Ty::List(Elem::Tuple(a)), Ty::List(Elem::Tuple(b))) => {
                Ty::Tuple(a).refined_by(Ty::Tuple(b))
            }
            _ => false,
        }
    }

    /// The shape of a tuple type.
    pub(crate) fn tuple_elems(self) -> Option<Vec<Ty>> {
        match self {
            Ty::Tuple(k) => Some(tuple_shape(k)),
            _ => None,
        }
    }

    /// This type once inference is over: whatever it left undecided is
    /// dynamic, inside a tuple's shape as well.
    pub(crate) fn settled(self) -> Ty {
        match self {
            Ty::Unknown => Ty::Object,
            Ty::Tuple(k) => tuple_of(tuple_shape(k).into_iter().map(Ty::settled).collect()),
            Ty::Dict(k) => {
                let (key, value) = dict_shape(k);
                dict_of(key.settled(), value.settled())
            }
            Ty::List(Elem::Tuple(k)) => Ty::List(Elem::of(Ty::Tuple(k).settled())),
            other => other,
        }
    }

    pub(crate) fn is_numeric(self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Bool)
    }

    /// The element type of a sequence, when it is one: for a tuple,
    /// the kind its elements are read as when they are read by a
    /// position only the runtime knows, which is the list kind the join
    /// of its elements is stored as.
    pub(crate) fn element(self) -> Option<Ty> {
        match self {
            Ty::List(e) => Some(e.ty()),
            Ty::Tuple(k) => {
                let joined = tuple_shape(k).into_iter().fold(Ty::Unknown, Ty::join);
                Some(match joined {
                    Ty::Unknown => Ty::Unknown,
                    other => Elem::of(other).ty(),
                })
            }
            // A dict iterates as its keys.
            Ty::Dict(k) => Some(match dict_shape(k).0 {
                Ty::Unknown => Ty::Unknown,
                key => Elem::of(key).ty(),
            }),
            Ty::Set | Ty::Gen => Some(Ty::Object),
            Ty::Str => Some(Ty::Str),
            // Bytes iterate as their byte values; a file as its lines.
            Ty::Bytes => Some(Ty::Int),
            Ty::File(mode) => Some(mode.content()),
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
    /// The attributes the classes declare (`class C: X = 1`, `C.X = v`).
    pub(crate) class_attrs: std::sync::Arc<crate::class_attrs::ClassAttrs>,
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
    pub(crate) closed: HashSet<String>,
    /// Whether inference is over: while it runs, a field a class does
    /// not yet declare may still be declared this round, and reads of
    /// it are undecided rather than dynamic.
    pub(crate) settled: std::cell::Cell<bool>,
    /// Every lambda and nested `def` in the program, by the function
    /// holding it and its position in the source, with what inference
    /// knows of each. Filled once before inference; the signatures are
    /// refined during it.
    pub(crate) closures: std::cell::RefCell<Vec<ClosureInfo>>,
    /// Closure index by `(file, start of range)`.
    pub(crate) closure_index: HashMap<(u32, u32), u16>,
    /// Every `x = recv.method` a call through `x` can be lowered as a
    /// call on `recv`; see [`BoundInfo`].
    pub(crate) bounds: Vec<BoundInfo>,
    /// Bound method index by `(file, start of the attribute's range)`.
    pub(crate) bound_index: HashMap<(u32, u32), u16>,
    /// For each function, by name, what its body does with each
    /// list-typed parameter: `None` where the list may be kept or
    /// written with anything, else the join of the element kinds
    /// written into it (`Unknown` for a list only read). A caller
    /// passing a list it is still typing reads this; see
    /// [`decide_list`].
    pub(crate) list_params: HashMap<String, Vec<ListFact>>,
    /// Methods, by plain name, called somewhere on a receiver whose
    /// class is not known: their parameters stay dynamic, so a call
    /// from out of view passes what it has. The others are typed by
    /// the calls in view, as module functions are.
    pub(crate) dynamic_methods: HashSet<String>,
    /// Fields a constructor binds to an unkinded list literal, by class
    /// and name, and the kind the program's writes into them decided
    /// last round (`Unknown` while undecided); see [`decide_list`].
    pub(crate) list_fields: HashSet<(usize, String)>,
    pub(crate) field_lists: HashMap<(usize, String), Ty>,
    /// What lowering each function found about its raising, by the
    /// name it lowers to; see [`RaiseFact`].
    pub(crate) raise_facts: std::cell::RefCell<std::collections::BTreeMap<String, RaiseFact>>,
    /// Functions of the program that never leave with an exception
    /// pending, so a call to one needs no check after it.
    pub(crate) non_raising: HashSet<String>,
    /// Items with a parameter typed as an instance, which are lowered a
    /// second time under [`trusted_name`] with those parameters taken
    /// to be instances; a call whose every such argument is known to be
    /// one goes there.
    pub(crate) trusted: HashSet<String>,
    /// Functions whose result is an instance and never None; see
    /// [`returning_instances`].
    pub(crate) returns_instance: HashSet<String>,
    /// Exception classes the lowering raises by name, each getting a
    /// cold `py$raise$Class(message)` that builds the instance and
    /// leaves it pending; see `classes::raisers`.
    pub(crate) raisers: std::cell::RefCell<std::collections::BTreeSet<String>>,
}

/// The name of the function raising exception class `class` with a
/// message.
pub(crate) fn raiser_name(class: &str) -> String {
    format!("py$raise${class}")
}

/// The name of the variant of `name` whose instance-typed parameters
/// are trusted not to be None.
pub(crate) fn trusted_name(name: &str) -> String {
    format!("{name}$trusted")
}

/// Whether `sig` has a parameter typed as an instance, `self` aside.
pub(crate) fn has_instance_params(sig: &Sig, is_method: bool) -> bool {
    sig.params
        .iter()
        .skip(usize::from(is_method))
        .any(|(_, t)| matches!(t, Ty::Class(_)))
}

/// The items whose result is an instance on every path: each `return`
/// hands back a constructor call, `self`, or the result of another such
/// item, and the body cannot fall off its end. Decided together, as the
/// greatest set consistent with itself.
pub(crate) fn returning_instances(module: &Module, items: &[Item<'_>]) -> HashSet<String> {
    fn returns_of<'a>(body: &'a [py::Stmt], out: &mut Vec<Option<&'a py::Expr>>) {
        use ruff_python_ast::visitor::{Visitor, walk_stmt};
        struct Returns<'a, 'b> {
            out: &'b mut Vec<Option<&'a py::Expr>>,
        }
        impl<'a> Visitor<'a> for Returns<'a, '_> {
            fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
                match stmt {
                    py::Stmt::Return(r) => self.out.push(r.value.as_deref()),
                    // A nested function's returns are its own.
                    py::Stmt::FunctionDef(_) | py::Stmt::ClassDef(_) => {}
                    other => walk_stmt(self, other),
                }
            }
        }
        for s in body {
            Returns { out }.visit_stmt(s);
        }
    }
    let mut quiet: HashSet<String> = items
        .iter()
        .filter(|item| {
            matches!(
                module.funcs.get(&item.name).map(|s| s.ret),
                Some(Ty::Class(_))
            ) && terminates(&item.def.body)
        })
        .map(|item| item.name.clone())
        .collect();
    loop {
        let demoted: Vec<String> = items
            .iter()
            .filter(|item| quiet.contains(&item.name))
            .filter(|item| {
                let mut returns = Vec::new();
                returns_of(&item.def.body, &mut returns);
                let self_name = item
                    .class
                    .and_then(|_| item.def.parameters.iter_non_variadic_params().next())
                    .map(|p| p.parameter.name.to_string());
                returns.is_empty()
                    || returns.iter().any(|r| match r {
                        None => true,
                        Some(py::Expr::Call(c)) => match &*c.func {
                            py::Expr::Name(n) => {
                                let n = n.id.as_str();
                                !(module.class_index.contains_key(n) || quiet.contains(n))
                            }
                            _ => true,
                        },
                        Some(py::Expr::Name(n)) => self_name.as_deref() != Some(n.id.as_str()),
                        Some(_) => true,
                    })
            })
            .map(|item| item.name.clone())
            .collect();
        if demoted.is_empty() {
            return quiet;
        }
        for n in demoted {
            quiet.remove(&n);
        }
    }
}

/// How a function of the program can come to raise: on its own (a
/// `raise`, a check after a library call or a dynamic operation), or
/// through one of the program's functions it calls. A function raises
/// if it raises on its own or any callee does; one not in the table is
/// taken to raise.
#[derive(Debug, Clone, Default)]
pub(crate) struct RaiseFact {
    pub(crate) own: bool,
    pub(crate) callees: std::collections::BTreeSet<String>,
}

/// The functions that never raise: the greatest fixed point over the
/// facts, starting from every listed function and demoting each that
/// raises on its own or calls a function not (or no longer) in the set.
pub(crate) fn non_raising(
    facts: &std::collections::BTreeMap<String, RaiseFact>,
) -> HashSet<String> {
    let mut quiet: HashSet<String> = facts
        .iter()
        .filter(|(_, f)| !f.own)
        .map(|(n, _)| n.clone())
        .collect();
    loop {
        let demoted: Vec<String> = quiet
            .iter()
            .filter(|n| facts[*n].callees.iter().any(|c| !quiet.contains(c)))
            .cloned()
            .collect();
        if demoted.is_empty() {
            return quiet;
        }
        for n in demoted {
            quiet.remove(&n);
        }
    }
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
    /// Whether the body may read a variable of an enclosing function.
    /// Then the typed entry takes the record, to reach the cells; a
    /// closure over nothing takes its parameters alone. Decided from the
    /// names in view, so it may say yes where the lowering finds no
    /// cell, never no where it finds one.
    pub(crate) captures: bool,
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

/// `x = recv.method`: a method bound to a local, over a receiver that
/// is a local of the same function. The record `x` holds serves every
/// use of it; a call through `x` is a call of the method on `recv`,
/// which is the object the binding saw because neither name is bound
/// again in the function and the binding sits in no loop.
#[derive(Debug, Clone)]
pub(crate) struct BoundInfo {
    /// The receiver, as written at the binding.
    pub(crate) receiver: py::ExprName,
    pub(crate) method: String,
}

/// The arity of a method a list or an instance value can be bound to,
/// which is what the lowering makes a record for; `None` is any other
/// attribute.
pub(crate) fn bound_method_arity(module: &Module, receiver: Ty, attr: &str) -> Option<i64> {
    match receiver {
        Ty::List(_) => match attr {
            "append" | "remove" | "index" | "count" | "extend" => Some(1),
            "insert" => Some(2),
            "sort" | "reverse" | "copy" | "clear" => Some(0),
            "pop" => Some(zyntax_builtins::functions::VARIADIC_ARITY),
            _ => None,
        },
        Ty::Class(k) => module
            .method_sig(k as usize, attr)
            .map(|(sig, _)| (sig.params.len() - 1) as i64),
        _ => None,
    }
}

/// Register every `x = recv.method` of `body` whose call sites can be
/// lowered direct: `x` and `recv` are each bound once in the function
/// (`recv` may be a parameter bound nowhere), neither is declared
/// `global` or `nonlocal`, and the binding is under no loop. Nested
/// bodies are walked for what they bind, not for bindings of their own.
pub(crate) fn collect_bound_methods(
    module: &mut Module,
    file: u32,
    body: &[py::Stmt],
    params: &[String],
) {
    use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
    #[derive(Default)]
    struct Stores(HashMap<String, usize>);
    impl Stores {
        fn note(&mut self, name: &str, times: usize) {
            *self.0.entry(name.to_string()).or_insert(0) += times;
        }
    }
    impl<'a> Visitor<'a> for Stores {
        fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
            match stmt {
                py::Stmt::FunctionDef(f) => self.note(f.name.as_str(), 1),
                py::Stmt::ClassDef(c) => self.note(c.name.as_str(), 1),
                py::Stmt::Import(i) => {
                    for a in &i.names {
                        let bound = a.asname.as_ref().unwrap_or(&a.name);
                        self.note(bound.split('.').next().unwrap_or_default(), 1);
                    }
                }
                py::Stmt::ImportFrom(i) => {
                    for a in &i.names {
                        self.note(a.asname.as_ref().unwrap_or(&a.name).as_str(), 1);
                    }
                }
                py::Stmt::Global(g) => {
                    for n in &g.names {
                        self.note(n.as_str(), 2);
                    }
                }
                py::Stmt::Nonlocal(g) => {
                    for n in &g.names {
                        self.note(n.as_str(), 2);
                    }
                }
                py::Stmt::Try(t) => {
                    for h in &t.handlers {
                        let py::ExceptHandler::ExceptHandler(h) = h;
                        if let Some(n) = &h.name {
                            self.note(n.as_str(), 1);
                        }
                    }
                }
                _ => {}
            }
            walk_stmt(self, stmt);
        }
        fn visit_expr(&mut self, expr: &'a py::Expr) {
            if let py::Expr::Name(n) = expr
                && !matches!(n.ctx, py::ExprContext::Load)
            {
                self.note(n.id.as_str(), 1);
            }
            walk_expr(self, expr);
        }
    }
    let mut stores = Stores::default();
    for s in body {
        stores.visit_stmt(s);
    }
    let bound_once = |name: &str| {
        let times = stores.0.get(name).copied().unwrap_or(0);
        if params.iter().any(|p| p == name) {
            times == 0
        } else {
            times == 1
        }
    };
    // The bindings themselves: statements of the body under any
    // branch, never under a loop or in a nested body.
    fn bindings<'a>(stmts: &'a [py::Stmt], out: &mut Vec<&'a py::StmtAssign>) {
        for s in stmts {
            match s {
                py::Stmt::Assign(a) => out.push(a),
                py::Stmt::If(i) => {
                    bindings(&i.body, out);
                    for c in &i.elif_else_clauses {
                        bindings(&c.body, out);
                    }
                }
                py::Stmt::Try(t) => {
                    bindings(&t.body, out);
                    for h in &t.handlers {
                        let py::ExceptHandler::ExceptHandler(h) = h;
                        bindings(&h.body, out);
                    }
                    bindings(&t.orelse, out);
                    bindings(&t.finalbody, out);
                }
                _ => {}
            }
        }
    }
    let mut found = Vec::new();
    bindings(body, &mut found);
    for a in found {
        let [py::Expr::Name(x)] = a.targets.as_slice() else {
            continue;
        };
        let py::Expr::Attribute(attr) = &*a.value else {
            continue;
        };
        let py::Expr::Name(recv) = &*attr.value else {
            continue;
        };
        if x.id == recv.id || !bound_once(x.id.as_str()) || !bound_once(recv.id.as_str()) {
            continue;
        }
        let index = module.bounds.len();
        module.bounds.push(BoundInfo {
            receiver: recv.clone(),
            method: attr.attr.to_string(),
        });
        module
            .bound_index
            .insert((file, attr.range.start().to_u32()), index as u16);
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
/// `visible` is every variable of the body's function: its parameters
/// and what it binds.
pub(crate) fn collect_closures(
    module: &mut Module,
    owner: &str,
    file: u32,
    body: &[py::Stmt],
    classes: &HashMap<String, usize>,
    visible: HashSet<String>,
) {
    use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
    struct Finder<'m, 'c> {
        module: &'m mut Module,
        classes: &'c HashMap<String, usize>,
        owner: Vec<String>,
        /// The variables of the enclosing functions, innermost last.
        visible: Vec<HashSet<String>>,
        file: u32,
    }
    impl Finder<'_, '_> {
        /// Register the closure and enter it.
        fn enter(
            &mut self,
            start: u32,
            kind: &str,
            sig: Sig,
            inferred: Vec<bool>,
            ret_inferred: bool,
            scope: &crate::scope::Scope,
        ) {
            let index = self.module.closures.borrow().len();
            let name = format!(
                "{}${kind}${index}",
                self.owner.last().cloned().unwrap_or_default()
            );
            let outer = self.visible.last().expect("an enclosing scope");
            let captures = !scope.free.is_disjoint(outer);
            let mut inner = outer.clone();
            inner.extend(scope.bound.iter().cloned());
            inner.extend(sig.params.iter().map(|(n, _)| n.clone()));
            self.module.closures.borrow_mut().push(ClosureInfo {
                name: name.clone(),
                sig,
                inferred,
                ret_inferred,
                captures,
                escapes: false,
            });
            self.module
                .closure_index
                .insert((self.file, start), index as u16);
            self.owner.push(name);
            self.visible.push(inner);
        }

        fn leave(&mut self) {
            self.owner.pop();
            self.visible.pop();
        }
    }
    impl<'a> Visitor<'a> for Finder<'_, '_> {
        fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
            match stmt {
                // A nested def that yields is a generator: its result is
                // the generator, whatever its body returns. One with
                // variadics is not compiled at all. A default is
                // evaluated where the def is, which a call from
                // elsewhere cannot see.
                py::Stmt::FunctionDef(f)
                    if f.parameters.vararg.is_none()
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
                    let generator = is_generator(&f.body);
                    if f.returns.is_none() && !generator {
                        sig.ret = Ty::Unknown;
                    }
                    self.enter(
                        f.range.start().to_u32(),
                        f.name.as_str(),
                        sig,
                        inferred,
                        f.returns.is_none() && !generator,
                        &crate::scope::Scope::of_function(f),
                    );
                    walk_stmt(self, stmt);
                    self.leave();
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
                    self.enter(
                        l.range.start().to_u32(),
                        "lambda",
                        sig,
                        vec![true; n],
                        true,
                        &crate::scope::Scope::of_lambda(l),
                    );
                    walk_expr(self, expr);
                    self.leave();
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
        visible: vec![visible],
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
            (ret, HashMap::default())
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

    /// The bound method an attribute read is, from the function holding
    /// it and the start of its range.
    pub(crate) fn bound_at(&self, file: u32, start: u32) -> Option<u16> {
        self.bound_index.get(&(file, start)).copied()
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
        crate::stdlib::Member::Str(_) => Ty::Str,
        crate::stdlib::Member::Value { ty, .. } => ty,
        // The type as a value; a call of it is typed by its typecode.
        crate::stdlib::Member::ArrayType => Ty::Object,
        // As a value a function; a call is typed by its arguments.
        crate::stdlib::Member::Binary(_) => Ty::Object,
    }
}

/// What a call of `array.array` with these arguments builds: the array
/// of the typecode named, when the typecode is a literal. Anything
/// else has no static element type and the lowering refuses it.
pub(crate) fn array_call_ty(args: &[py::Expr]) -> Ty {
    match array_call_code(args) {
        Some(code) => Ty::List(Elem::Array(code)),
        None => Ty::Object,
    }
}

/// The typecode an `array.array` call spells as its first argument.
pub(crate) fn array_call_code(args: &[py::Expr]) -> Option<Code> {
    match args.first() {
        Some(py::Expr::StringLiteral(s)) => crate::stdlib::array_code(s.value.to_str()).ok(),
        _ => None,
    }
}

/// A class: its place in the hierarchy, its layout and its methods.
#[derive(Debug, Clone, Default)]
pub(crate) struct ClassInfo {
    pub(crate) name: String,
    pub(crate) base: Option<usize>,
    /// How many classes the class and everything deriving from it are:
    /// they hold the indices from this one's for that many, since
    /// classes are numbered in preorder of the hierarchy.
    pub(crate) descendants: usize,
    /// Every field in layout order: the class tag first, then the
    /// base's fields, then this class's own.
    pub(crate) fields: Vec<(String, Ty)>,
    /// The methods this class itself defines.
    pub(crate) methods: Vec<String>,
    pub(crate) type_id: Option<zyntax_typed_ast::TypeId>,
    /// The module the class was declared in; `None` for the main file.
    pub(crate) module: Option<String>,
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

    /// The class attribute `name` as class `k` sees it, when `k` has no
    /// field of that name.
    pub(crate) fn class_attr(
        &self,
        k: usize,
        name: &str,
    ) -> Option<&crate::class_attrs::ClassAttr> {
        if self.field(k, name).is_some() {
            return None;
        }
        let bases: Vec<Option<usize>> = self.classes.iter().map(|c| c.base).collect();
        self.class_attrs.lookup(&bases, k, name)
    }

    /// The type a class attribute has here: its constant's, else its
    /// module variable's.
    pub(crate) fn class_attr_ty(&self, attr: &crate::class_attrs::ClassAttr) -> Ty {
        match &attr.constant {
            Some(c) => c.ty(),
            None => self
                .globals
                .get(&attr.global)
                .copied()
                .unwrap_or(Ty::Unknown),
        }
    }

    /// The class `e` names, as a call of the name would construct: a
    /// class's name is the class wherever it is written.
    pub(crate) fn class_of_expr(&self, e: &py::Expr) -> Option<usize> {
        crate::class_attrs::class_named(&self.class_index, e)
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

    /// What a call of `method` on an instance of `k` returns: the join
    /// over the method `k` sees and every override an instance of a
    /// subclass of `k` would reach, since the call goes to whichever the
    /// instance's class holds.
    pub(crate) fn dispatched_ret(&self, k: usize, method: &str) -> Option<Ty> {
        let (sig, _) = self.method_sig(k, method)?;
        let mut ret = sig.ret;
        for sub in self.overriders(k, method) {
            if let Some((sub_sig, _)) = self.method_sig(sub, method) {
                ret = self.join_classes(ret, sub_sig.ret);
            }
        }
        Some(ret)
    }

    /// [`Ty::join`] knowing the hierarchy: two instance types join to
    /// the nearest class both derive from, when there is one.
    pub(crate) fn join_classes(&self, a: Ty, b: Ty) -> Ty {
        if let (Ty::Class(x), Ty::Class(y)) = (a, b) {
            let mut at = Some(x as usize);
            while let Some(c) = at {
                if self.is_subclass(y as usize, c) {
                    return Ty::Class(c as u16);
                }
                at = self.classes[c].base;
            }
        }
        a.join(b)
    }

    /// Whether `k` is `base` or derives from it.
    pub(crate) fn is_subclass(&self, k: usize, base: usize) -> bool {
        k >= base && k < base + self.classes[base].descendants
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
    /// Fields written through `self`, in the order first written, which
    /// is the order they are laid out in.
    pub(crate) field_writes: indexmap::IndexMap<String, Ty>,
    /// Fields written through an instance of a known class other than
    /// `self`: the class, the field, what is written.
    pub(crate) other_field_writes: Vec<(usize, String, Ty)>,
    /// Locals bound once and then asserted to be an instance of a class
    /// on the next line: they have that class, and the binding checks.
    pub(crate) narrowed: HashMap<String, u16>,
    /// What the body assigns to its own parameters.
    pub(crate) param_writes: HashMap<String, Ty>,
    /// Whether the body has a `return`; without one it returns None.
    pub(crate) returns: bool,
}

/// Whether an annotation asks for a dynamic value: `Any`, `typing.Any`
/// or `object`.
pub(crate) fn is_dynamic_annotation(e: &py::Expr) -> bool {
    match e {
        py::Expr::Name(n) => matches!(n.id.as_str(), "Any" | "object"),
        py::Expr::Attribute(a) => a.attr.as_str() == "Any",
        _ => false,
    }
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
            "dict" | "Dict" | "Mapping" => dynamic_dict(),
            "set" | "Set" => Ty::Set,
            "tuple" | "Tuple" => Ty::Object,
            other => classes
                .get(other)
                .map(|k| Ty::Class(*k as u16))
                .unwrap_or(Ty::Object),
        },
        py::Expr::NoneLiteral(_) => Ty::None,
        // `list[int]` and friends: the outer name decides.
        // `tuple[int, str]` is that shape.
        py::Expr::Subscript(sub) => match annotation_in(classes, &sub.value) {
            Ty::List(_) => match annotation_in(classes, &sub.slice) {
                Ty::Int => Ty::List(Elem::Int),
                Ty::Float => Ty::List(Elem::Float),
                Ty::Str => Ty::List(Elem::Str),
                Ty::Class(k) => Ty::List(Elem::Class(k)),
                _ => Ty::List(Elem::Object),
            },
            Ty::Object if matches!(&*sub.value, py::Expr::Name(n) if matches!(n.id.as_str(), "tuple" | "Tuple")) => {
                match &*sub.slice {
                    py::Expr::Tuple(t)
                        if !t
                            .elts
                            .iter()
                            .any(|e| matches!(e, py::Expr::EllipsisLiteral(_))) =>
                    {
                        tuple_of(t.elts.iter().map(|e| annotation_in(classes, e)).collect())
                    }
                    _ => Ty::Object,
                }
            }
            other => other,
        },
        _ => Ty::Object,
    }
}

/// The type of `x: T = v`. A typed value keeps its type: the annotation
/// converts nothing, `x: float = 3` binds the int. A dynamic value takes
/// the annotation, read back with a check as an annotated parameter is,
/// so that the annotation is what makes `v: float = bag.payload` a
/// float.
pub(crate) fn annotated_value(
    classes: &HashMap<String, usize>,
    annotation: &py::Expr,
    value: Ty,
) -> Ty {
    if value != Ty::Object {
        return value;
    }
    match annotation_in(classes, annotation) {
        Ty::Object | Ty::None | Ty::Unknown => value,
        declared => declared,
    }
}

/// The element kind an empty list literal takes from the annotation
/// on its binding, `xs: list[str] = []`, when the annotation names
/// one: the literal has no element to say otherwise.
pub(crate) fn annotated_empty_list(
    classes: &HashMap<String, usize>,
    annotation: &py::Expr,
    value: &py::Expr,
) -> Option<Elem> {
    match value {
        py::Expr::List(l) if l.elts.is_empty() => match annotation_in(classes, annotation) {
            Ty::List(e) if e != Elem::Object => Some(e),
            _ => None,
        },
        _ => None,
    }
}

/// Signature from the annotations alone; an unannotated return is
/// `Unknown` until the body says.
pub(crate) fn declared_sig(f: &py::StmtFunctionDef) -> Sig {
    declared_sig_in(&HashMap::default(), f, None)
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
    use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
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
    pub(crate) list_params: HashMap<String, Vec<ListFact>>,
    pub(crate) dynamic_methods: HashSet<String>,
    pub(crate) list_fields: HashSet<(usize, String)>,
    pub(crate) field_lists: HashMap<(usize, String), Ty>,
}

/// The items every call of which is in view: module functions never
/// mentioned but as a callee, and constructors never reached but
/// through their class. An unannotated parameter of one is typed by
/// what is passed to it, everywhere it is passed; any other stays
/// dynamic, as Python has it.
pub(crate) fn closed_items(body: &[py::Stmt], items: &[Item<'_>]) -> HashSet<String> {
    use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
    #[derive(Default)]
    struct Mentions {
        /// Names used as anything but a callee: values, assignment
        /// targets, deletions.
        names: HashSet<String>,
        /// `x.__init__` on anything but `super()`: a constructor
        /// reached without its class.
        init: bool,
        /// Attribute names read as values rather than called.
        valued_methods: HashSet<String>,
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
                        // A method called is not a method valued.
                        py::Expr::Attribute(a) if a.attr.as_str() != "__init__" => {
                            self.visit_expr(&a.value)
                        }
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
                // `x.__init__` on anything but `super()` or a class
                // named outright, whose call is in view.
                py::Expr::Attribute(a) if a.attr.as_str() == "__init__" => {
                    if !is_super_call(&a.value) && !matches!(&*a.value, py::Expr::Name(_)) {
                        self.init = true;
                    }
                    walk_expr(self, expr);
                }
                // An attribute read that is not a call: a method of
                // that name is a value here.
                py::Expr::Attribute(a) => {
                    self.valued_methods.insert(a.attr.to_string());
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
    let mut counts: HashMap<&str, usize> = HashMap::default();
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
            // An operator method is reached from the operators on its
            // class, which are in view, and from dynamic arithmetic,
            // which reads the operand back as the type inferred here.
            // Any other method is closed unless it is read as a value
            // somewhere; a call on an unknown receiver is found during
            // inference and opens it again.
            Some(_) => {
                (item.def.name.as_str() == "__init__"
                    && !seen.init
                    && !crate::prelude::EXCEPTION_KINDS
                        .iter()
                        .any(|kind| item.name == method_fn(kind, "__init__")))
                    || is_operator_method(item.def.name.as_str())
                    || (!item.def.name.starts_with("__")
                        && !seen.valued_methods.contains(item.def.name.as_str()))
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
        funcs: HashMap::default(),
        globals: known.globals.clone(),
        list_type: known.list_type,
        classes: known.classes.clone(),
        class_index: known.class_index.clone(),
        class_attrs: known.class_attrs.clone(),
        closed: known.closed.clone(),
        closures: std::cell::RefCell::new(closures),
        closure_index: known.closure_index.clone(),
        bounds: known.bounds.clone(),
        bound_index: known.bound_index.clone(),
        list_params: known.list_params.clone(),
        dynamic_methods: known.dynamic_methods.clone(),
        list_fields: known.list_fields.clone(),
        field_lists: known.field_lists.clone(),
        files: known.files.clone(),
        imports: known.imports.clone(),
        from_names: known.from_names.clone(),
        ..Default::default()
    };
    // The fields a constructor binds to an unkinded literal.
    for item in items {
        let Some(k) = item.class else {
            continue;
        };
        let Some(first) = item.def.parameters.iter_non_variadic_params().next() else {
            continue;
        };
        let this = first.parameter.name.as_str();
        for stmt in &item.def.body {
            let py::Stmt::Assign(a) = stmt else {
                continue;
            };
            if unkinded_list(&a.value).is_none() {
                continue;
            }
            for t in &a.targets {
                if let py::Expr::Attribute(attr) = t
                    && is_name(&attr.value, this)
                {
                    module.list_fields.insert((k, attr.attr.to_string()));
                }
            }
        }
    }
    // Which parameters of each function are inferred, by position. A
    // method is, like a closed function, unless a call of a method of
    // its name on an unknown receiver was seen last time round.
    let mut inferring: HashMap<String, Vec<bool>> = HashMap::default();
    for item in items {
        let mut sig = declared_sig_in(&module.class_index, item.def, item.class);
        let closed = known.closed.contains(&item.name)
            && !(item.class.is_some() && known.dynamic_methods.contains(item.def.name.as_str()));
        if closed {
            let flags: Vec<bool> = item
                .def
                .parameters
                .iter_non_variadic_params()
                .enumerate()
                .map(|(i, p)| p.parameter.annotation.is_none() && !(i == 0 && item.class.is_some()))
                .collect();
            // A default is one of the values the parameter takes,
            // whether or not a call leaves it out.
            let defaults: Vec<Ty> = sig
                .defaults
                .iter()
                .map(|d| match d {
                    Some(d) => Typer {
                        module: &module,
                        vars: &HashMap::default(),
                        outer: &HashMap::default(),
                    }
                    .expr(d),
                    None => Ty::Unknown,
                })
                .collect();
            for ((flag, (_, ty)), default) in flags.iter().zip(sig.params.iter_mut()).zip(defaults)
            {
                if *flag {
                    *ty = default;
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
    let field_keys: Vec<String> = module
        .list_fields
        .iter()
        .map(|(k, f)| field_key(*k, f))
        .collect();
    let trace_rounds = std::env::var_os("ZYNTAX_TRACE_TYPES_ROUNDS").is_some();
    let mut inner_rounds = 0;
    let index_of: HashMap<&str, usize> = items
        .iter()
        .enumerate()
        .map(|(i, item)| (item.name.as_str(), i))
        .collect();
    for _ in 0..32 {
        inner_rounds += 1;
        let mut changed = false;
        let mut escaped: Vec<u16> = Vec::new();
        let mut dynamic_methods = HashSet::default();
        // Field writes per item, so an item inferred again this round
        // replaces its earlier reading rather than adding to it.
        let mut item_field_rounds: Vec<HashMap<String, FieldRound>> =
            vec![HashMap::default(); items.len()];
        let mut entry_field_rounds: HashMap<String, FieldRound> = HashMap::default();
        // Every item once, the module body, then whichever items a call
        // typed a parameter of since they were inferred, until none is
        // left: a parameter type that reaches its callee mid-round saves
        // the round the callee would otherwise wait for.
        let mut queue: std::collections::VecDeque<usize> = (0..items.len()).collect();
        let mut queued: Vec<bool> = vec![true; items.len()];
        let mut entry_done = false;
        loop {
            let mut passed: Vec<(Target, usize, Ty)> = Vec::new();
            if let Some(i) = queue.pop_front() {
                queued[i] = false;
                let item = &items[i];
                let sig = module.funcs[&item.name].clone();
                let file = module.file_of(item.module.as_deref());
                let locals = in_file(file, || {
                    let locals = infer_locals_open(&module, &sig, &item.def.body, &[]);
                    changed |= infer_closures_in(&module, &item.def.body, &[], &locals.vars);
                    locals
                });
                let facts = in_file(file, || {
                    list_param_facts(&module, &item.def.body, &locals.vars, &sig.params)
                });
                if !field_keys.is_empty() {
                    let rounds = &mut item_field_rounds[i];
                    rounds.clear();
                    in_file(file, || {
                        field_sites_into(&module, &item.def.body, &locals.vars, &field_keys, rounds)
                    });
                }
                if module.list_params.get(&item.name) != Some(&facts) {
                    module.list_params.insert(item.name.clone(), facts);
                    changed = true;
                    if trace_rounds {
                        eprintln!("[types] inner {inner_rounds}: list params of {}", item.name);
                    }
                }
                if item.def.returns.is_none() && sig.ret != Ty::Gen {
                    let ret = if locals.returns { locals.ret } else { Ty::None };
                    if ret != sig.ret {
                        if trace_rounds {
                            eprintln!(
                                "[types] inner {inner_rounds}: {} returns {ret:?} (was {:?})",
                                item.name, sig.ret
                            );
                        }
                        module.funcs.get_mut(&item.name).unwrap().ret = ret;
                        changed = true;
                    }
                }
                // What a method assigns to `self.x` types the field on its
                // class, and on every class deriving from it.
                if let Some(k) = item.class {
                    for (field, ty) in &locals.field_writes {
                        if trace_rounds && *ty == Ty::Object {
                            eprintln!("[types] inner: {} writes self.{field} as Object", item.name);
                        }
                        changed |= widen_field(&mut module.classes, k, field, *ty);
                    }
                }
                for (k, field, ty) in &locals.other_field_writes {
                    if trace_rounds {
                        eprintln!(
                            "[types] inner: {} writes {}.{field} as {ty:?}",
                            item.name, module.classes[*k].name
                        );
                    }
                    changed |= widen_field(&mut module.classes, *k, field, *ty);
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
                        comp_vars: None,
                        passed: &mut passed,
                        allow_closure: false,
                        escaped: &mut escaped,
                        dynamic_methods: &mut dynamic_methods,
                        settled: false,
                        files: &[],
                        no_outer: HashMap::default(),
                    }
                    .stmts(&item.def.body)
                });
            } else if !entry_done {
                entry_done = true;
                entry_locals = infer_locals_open(&module, &entry_sig, entry, entry_files);
                for (k, field, ty) in &entry_locals.other_field_writes {
                    changed |= widen_field(&mut module.classes, *k, field, *ty);
                }
                if !field_keys.is_empty() {
                    field_sites_into(
                        &module,
                        entry,
                        &entry_locals.vars,
                        &field_keys,
                        &mut entry_field_rounds,
                    );
                }
                changed |= infer_closures_in(&module, entry, entry_files, &entry_locals.vars);
                Calls {
                    module: &module,
                    vars: &entry_locals.vars,
                    class: None,
                    opaque: false,
                    comp_vars: None,
                    passed: &mut passed,
                    allow_closure: false,
                    escaped: &mut escaped,
                    dynamic_methods: &mut dynamic_methods,
                    settled: false,
                    files: entry_files,
                    no_outer: HashMap::default(),
                }
                .stmts(entry);
            } else {
                break;
            }
            for (callee, index, ty) in passed {
                let slot = match &callee {
                    Target::Item(name) => {
                        let Some(flags) = inferring.get(name) else {
                            continue;
                        };
                        if !flags[index] {
                            continue;
                        }
                        let current = module.funcs[name].params[index].1;
                        let joined = module.join_classes(current, ty);
                        if joined != current {
                            if trace_rounds {
                                eprintln!(
                                    "[types] inner {inner_rounds}: {name} param {index} {joined:?} (was {current:?})"
                                );
                            }
                            module.funcs.get_mut(name).unwrap().params[index].1 = joined;
                            changed = true;
                            if let Some(&j) = index_of.get(name.as_str())
                                && !queued[j]
                            {
                                queued[j] = true;
                                queue.push_back(j);
                            }
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
        }
        if !field_keys.is_empty() {
            let mut field_rounds: HashMap<String, FieldRound> = entry_field_rounds;
            for rounds in item_field_rounds {
                for (key, round) in rounds {
                    let into = field_rounds.entry(key).or_default();
                    into.kept |= round.kept;
                    into.none |= round.none;
                    into.written.extend(round.written);
                }
            }
            for (k, f) in module.list_fields.clone() {
                let decided = field_rounds
                    .get(&field_key(k, &f))
                    .map(FieldRound::decide)
                    .unwrap_or(Ty::List(Elem::Object));
                if module.field_lists.get(&(k, f.clone())) != Some(&decided) {
                    module.field_lists.insert((k, f), decided);
                    changed = true;
                }
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
        if let Ok(watch) = std::env::var("ZYNTAX_TRACE_TYPES_ROUNDS") {
            for name in watch.split(',') {
                if let Some(sig) = module.funcs.get(name) {
                    eprintln!("[types] inner: {name} {:?} -> {:?}", sig.params, sig.ret);
                }
                for class in &module.classes {
                    if class.name == name {
                        eprintln!("[types] inner: class {name} {:?}", class.fields);
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    if trace_rounds {
        eprintln!(
            "[types] infer_module: {inner_rounds} inner rounds over {} items",
            items.len()
        );
    }
    // The methods called on receivers that never got a type, now that
    // nothing more will be reached.
    let mut dynamic_methods = HashSet::default();
    {
        let mut passed: Vec<(Target, usize, Ty)> = Vec::new();
        let mut escaped: Vec<u16> = Vec::new();
        for item in items {
            let sig = module.funcs[&item.name].clone();
            let file = module.file_of(item.module.as_deref());
            let locals = in_file(file, || {
                infer_locals_open(&module, &sig, &item.def.body, &[])
            });
            in_file(file, || {
                Calls {
                    module: &module,
                    vars: &locals.vars,
                    class: item.class,
                    opaque: false,
                    comp_vars: None,
                    passed: &mut passed,
                    allow_closure: false,
                    escaped: &mut escaped,
                    dynamic_methods: &mut dynamic_methods,
                    settled: true,
                    files: &[],
                    no_outer: HashMap::default(),
                }
                .stmts(&item.def.body)
            });
        }
        Calls {
            module: &module,
            vars: &entry_locals.vars,
            class: None,
            opaque: false,
            comp_vars: None,
            passed: &mut passed,
            allow_closure: false,
            escaped: &mut escaped,
            dynamic_methods: &mut dynamic_methods,
            settled: true,
            files: entry_files,
            no_outer: HashMap::default(),
        }
        .stmts(entry);
    }
    // Whatever recursion left undecided is dynamic. So is a parameter
    // only ever passed None: the IR has no value of that type to pass.
    for (name, sig) in module.funcs.iter_mut() {
        sig.ret = sig.ret.settled();
        if let Some(flags) = inferring.get(name) {
            for (flag, (_, ty)) in flags.iter().zip(sig.params.iter_mut()) {
                if *flag {
                    *ty = match *ty {
                        Ty::None => Ty::Object,
                        t => t.settled(),
                    };
                }
            }
        }
    }
    for class in &mut module.classes {
        for (_, ty) in &mut class.fields {
            *ty = ty.settled();
        }
    }
    let mut closures = module.closures.take();
    for c in &mut closures {
        c.sig.ret = c.sig.ret.settled();
        for (flag, (_, ty)) in c.inferred.iter().zip(c.sig.params.iter_mut()) {
            if *flag {
                *ty = ty.settled();
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
        list_params: module.list_params,
        dynamic_methods,
        list_fields: module.list_fields,
        field_lists: module.field_lists,
    }
}

/// The calls a body makes to the module's own functions and
/// constructors, and what each passes to which parameter.
struct Calls<'a> {
    module: &'a Module,
    vars: &'a HashMap<String, Ty>,
    /// Inside a comprehension: the body's variables with the
    /// comprehension's own bound to what its iterables yield.
    comp_vars: Option<HashMap<String, Ty>>,
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
    /// Methods called on a receiver whose class is not known.
    dynamic_methods: &'a mut HashSet<String>,
    /// Whether a receiver still untyped counts as unknown: only once the
    /// rounds are over, since a type not yet reached is not a dynamic
    /// value, and treating it as one would keep it from being reached.
    settled: bool,
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
            // A call through a bound method of an instance reaches the
            // class's method, with the receiver as its first argument.
            py::Expr::Name(_) if matches!(self.typer().callee_ty(func), Ty::Bound(_)) => {
                let Ty::Bound(k) = self.typer().callee_ty(func) else {
                    unreachable!()
                };
                let info = &self.module.bounds[k as usize];
                let Ty::Class(c) = self.typer().expr(&py::Expr::Name(info.receiver.clone())) else {
                    return None;
                };
                let (_, name) = self.module.method_sig(c as usize, &info.method)?;
                Some((Target::Item(name), 1))
            }
            py::Expr::Attribute(a) if is_super_call(&a.value) => {
                let base = self.module.classes[self.class?].base?;
                let (_, name) = self.module.method_sig(base, a.attr.as_str())?;
                Some((Target::Item(name), 1))
            }
            // `Class.method(obj, ...)`: every argument, the receiver first.
            py::Expr::Attribute(a) if self.typer().class_named(&a.value).is_some() => {
                let k = self.typer().class_named(&a.value)?;
                let (_, name) = self.module.method_sig(k, a.attr.as_str())?;
                Some((Target::Item(name), 0))
            }
            // A method on an instance of a known class: the class's
            // own; each overriding one is recorded beside it, since
            // the call reaches whichever the instance's class holds.
            py::Expr::Attribute(a) if matches!(self.arg_ty(&a.value), Ty::Class(_)) => {
                let Ty::Class(k) = self.arg_ty(&a.value) else {
                    unreachable!()
                };
                let (_, name) = self.module.method_sig(k as usize, a.attr.as_str())?;
                Some((Target::Item(name), 1))
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
            vars: self.comp_vars.as_ref().unwrap_or(self.vars),
            outer: &self.no_outer,
        }
    }

    /// Walk a comprehension with its variables bound: what it calls is
    /// typed as the body's own calls are.
    fn comprehension(&mut self, generators: &[py::Comprehension], walk: impl FnOnce(&mut Self)) {
        let vars = self.typer().comprehension_vars(generators);
        let was = self.comp_vars.replace(vars);
        walk(self);
        self.comp_vars = was;
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

    /// `left op right` where `left` is an instance: a call of the
    /// class's method for `op` with `right` as its one argument.
    fn operator_site(&mut self, op: py::Operator, left: &py::Expr, right: &py::Expr) {
        if self.opaque {
            return;
        }
        let Ty::Class(k) = self.typer().expr(left) else {
            return;
        };
        let Some((_, name)) = self.module.method_sig(k as usize, dunder_name(op)) else {
            return;
        };
        let ty = self.arg_ty(right);
        self.passed.push((Target::Item(name), 1, ty));
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
        // A starred tuple of known shape is its elements, one by one.
        let spread = spread_starred(args, |e| self.arg_ty(e));
        let args: &[py::Expr] = match &spread {
            Some(expanded) => expanded,
            None => args,
        };
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
                        vars: &HashMap::default(),
                        outer: &HashMap::default(),
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
            // A module class's methods are items, visited as such; only
            // what its body declares is evaluated here.
            py::Stmt::ClassDef(c) if self.module.class_index.contains_key(c.name.as_str()) => {
                for s in &c.body {
                    match s {
                        py::Stmt::Assign(a) => self.visit_expr(&a.value),
                        py::Stmt::AnnAssign(a) => {
                            if let Some(v) = &a.value {
                                self.visit_expr(v);
                            }
                        }
                        _ => {}
                    }
                }
            }
            py::Stmt::ClassDef(c) => self.nested(|v| v.visit_body(&c.body)),
            // `raise C`: the class is called with nothing.
            py::Stmt::Raise(r) => {
                if let Some(py::Expr::Name(n)) = r.exc.as_deref()
                    && let Some(&k) = self.module.class_index.get(n.id.as_str())
                    && let Some((_, name)) = self.module.method_sig(k, "__init__")
                {
                    self.record(Target::Item(name), 1, &[], &[]);
                }
                walk_stmt(self, stmt);
            }
            py::Stmt::AugAssign(a) => {
                self.operator_site(a.op, &a.target, &a.value);
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
        if !allowed && let Some(k) = self.closure_of(expr) {
            self.escape(k);
        }
        // An operator on an instance passes the right operand to the
        // class's method for it.
        if let py::Expr::BinOp(b) = expr {
            self.operator_site(b.op, &b.left, &b.right);
        }
        match expr {
            py::Expr::Call(c) => {
                if let Some((target, first)) = self.callee(&c.func) {
                    self.record(target, first, &c.arguments.args, &c.arguments.keywords);
                }
                if let py::Expr::Attribute(a) = &*c.func
                    && !is_super_call(&a.value)
                    && !self.opaque
                    && self.typer().class_named(&a.value).is_none()
                {
                    match self.arg_ty(&a.value) {
                        Ty::Class(k) => {
                            for sub in self.module.overriders(k as usize, a.attr.as_str()) {
                                if let Some((_, name)) =
                                    self.module.method_sig(sub, a.attr.as_str())
                                {
                                    self.record(
                                        Target::Item(name),
                                        1,
                                        &c.arguments.args,
                                        &c.arguments.keywords,
                                    );
                                }
                            }
                        }
                        // Not an instance of a known class: any
                        // method of the name may be reached with
                        // whatever this passes.
                        ty if (ty == Ty::Object || (ty == Ty::Unknown && self.settled))
                            && self
                                .module
                                .classes
                                .iter()
                                .any(|c| c.methods.iter().any(|m| m == a.attr.as_str())) =>
                        {
                            self.dynamic_methods.insert(a.attr.to_string());
                        }
                        _ => {}
                    }
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
            // A comprehension's own variables are bound from its
            // iterables for the walk of it.
            py::Expr::ListComp(py::ExprListComp { generators, .. })
            | py::Expr::SetComp(py::ExprSetComp { generators, .. })
            | py::Expr::DictComp(py::ExprDictComp { generators, .. })
            | py::Expr::Generator(py::ExprGenerator { generators, .. }) => {
                self.comprehension(generators, |c| walk_expr(c, expr))
            }
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
    // A field an ancestor declares is the ancestor's: its layout, and
    // every class deriving from it, must agree on the field's type, or
    // the ancestor's methods read the wrong width through a subclass
    // instance.
    let mut k = k;
    while let Some(base) = classes[k].base {
        if classes[base].fields.iter().any(|(f, _)| f == name) {
            k = base;
        } else {
            break;
        }
    }
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
    // Two instance types join to the nearest class both derive from.
    let bases: Vec<Option<usize>> = classes.iter().map(|c| c.base).collect();
    let join = |a: Ty, b: Ty| -> Ty {
        if let (Ty::Class(x), Ty::Class(y)) = (a, b) {
            let mut at = Some(x as usize);
            while let Some(c) = at {
                let mut y_at = Some(y as usize);
                while let Some(yc) = y_at {
                    if yc == c {
                        return Ty::Class(c as u16);
                    }
                    y_at = bases[yc];
                }
                at = bases[c];
            }
        }
        a.join(b)
    };
    for c in targets {
        match classes[c].fields.iter().position(|(f, _)| f == name) {
            Some(i) => {
                let joined = join(classes[c].fields[i].1, ty);
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
    infer_locals_seeded(module, sig, body, &HashMap::default())
}

/// [`infer_locals`] for the entry body, whose statements come from
/// several files: `files` names each top-level statement's.
pub(crate) fn infer_locals_entry(
    module: &Module,
    sig: &Sig,
    body: &[py::Stmt],
    files: &[u32],
) -> Locals {
    infer_locals_with(module, sig, body, &HashMap::default(), files, true)
}

/// [`infer_locals`] leaving what is undecided undecided, for the
/// module-wide fixed point: a value another function has yet to type
/// must not settle as dynamic here and poison every join it reaches.
fn infer_locals_open(module: &Module, sig: &Sig, body: &[py::Stmt], files: &[u32]) -> Locals {
    infer_locals_with(module, sig, body, &HashMap::default(), files, false)
}

/// Whatever inference left undecided is dynamic.
fn settle(locals: &mut Locals) {
    for ty in locals.vars.values_mut() {
        *ty = ty.settled();
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
    if !module.class_attrs.is_empty() {
        for (k, name) in class_attr_writes(&module.class_index, body) {
            if let Some(attr) = module.class_attr(k, &name) {
                locals
                    .global_writes
                    .insert(attr.global.clone(), Ty::Unknown);
            }
        }
    }
    for name in &scope.nonlocals {
        locals.nonlocal_writes.insert(name.clone(), Ty::Unknown);
        locals.vars.remove(name);
    }
    let fills = list_sites(
        module,
        body,
        &locals.vars,
        unkinded_locals(body, &sig.params, seeds, &scope),
        true,
    );
    locals.narrowed = asserted_classes(body, &module.class_index);
    for _ in 0..8 {
        let before = locals.clone();
        // The result is decided afresh each round, from the types known
        // now, as an unkinded list is.
        locals.ret = Ty::Unknown;
        locals.returns = false;
        let mut walker = Walker {
            module,
            locals: &mut locals,
            params: &sig.params,
            seeds,
            fills: &fills,
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

pub(crate) fn is_empty_list(e: &py::Expr) -> bool {
    matches!(e, py::Expr::List(l) if l.elts.is_empty())
}

/// `[]`, `[None]` or `[None] * n`: a list literal that says nothing
/// about its elements' kind, which is then the kind of what the body
/// puts in. `[None]` counts as one write of None, which any instance
/// kind admits. Says the count of Nones to build.
pub(crate) fn unkinded_list(e: &py::Expr) -> Option<Option<&py::Expr>> {
    let nones = |l: &py::ExprList| l.elts.iter().all(|e| matches!(e, py::Expr::NoneLiteral(_)));
    match e {
        py::Expr::List(l) if l.elts.is_empty() => Some(None),
        py::Expr::List(l) if l.elts.len() == 1 && nones(l) => Some(None),
        py::Expr::BinOp(b) if b.op == py::Operator::Mult => match (&*b.left, &*b.right) {
            (py::Expr::List(l), n) if l.elts.len() == 1 && nones(l) => Some(Some(n)),
            _ => None,
        },
        _ => None,
    }
}

/// How a body uses a list bound to one of its names, collected once
/// from the syntax; what is written in is typed as inference goes.
#[derive(Debug, Default)]
pub(crate) struct ListSites<'ast> {
    /// The list may be kept, aliased, passed to something with no facts,
    /// or written with something not accounted for: its kind is not
    /// this body's to decide.
    pub(crate) kept: bool,
    /// The expressions whose values are put in as elements.
    pub(crate) elements: Vec<&'ast py::Expr>,
    /// The expressions whose elements are put in.
    pub(crate) sequences: Vec<&'ast py::Expr>,
    /// Whether a literal `[None]` seeded it.
    pub(crate) none: bool,
    /// Functions of the module it is passed to, with the parameter's
    /// position; what they write in is what their facts say.
    pub(crate) passed_to: Vec<(String, usize)>,
    /// Written through a receiver not yet typed: nothing can be decided
    /// this round.
    pub(crate) undecided: bool,
}

/// What a function does with a parameter, for callers typing the list
/// they pass; see [`Module::list_params`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ListFact {
    /// Kept, aliased, or written with what cannot be accounted for.
    Kept,
    /// Only read.
    Reads,
    /// Written with elements of this type; `Unknown` while the writes
    /// are not yet typed.
    Writes(Ty),
}

/// The key a class's field goes by among list sites.
pub(crate) fn field_key(class: usize, field: &str) -> String {
    format!("{class}#{field}")
}

/// The uses of each of `names`, list-typed variables of `body`. A read
/// is an index, a slice, an iteration, a measure, a join, a membership
/// test, a truth test, or a call of one of the reading methods; a write
/// is `append`, `insert`, `extend`, `+=`, an element or slice store, or
/// a pass to a function of the module. Anything else, in the body or a
/// nested one, keeps the list. Returning a name is a read when
/// `returns_read`: a local built here leaves as the list it was decided
/// to be, while a parameter or a field returned whole is an alias the
/// caller may write through.
pub(crate) fn list_sites<'ast>(
    module: &Module,
    body: &'ast [py::Stmt],
    vars: &HashMap<String, Ty>,
    names: Vec<String>,
    returns_read: bool,
) -> HashMap<String, ListSites<'ast>> {
    use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
    struct Uses<'a, 'm, 'ast> {
        module: &'m Module,
        typer: Typer<'m>,
        sites: HashMap<String, ListSites<'ast>>,
        /// Whether the current expression is one of the allowed uses.
        allowed: &'a std::cell::Cell<bool>,
        returns_read: bool,
    }
    impl<'ast> Uses<'_, '_, 'ast> {
        fn site(&mut self, name: &str) -> &mut ListSites<'ast> {
            self.sites.get_mut(name).expect("a name being followed")
        }
        fn drop_name(&mut self, name: &str) {
            self.site(name).kept = true;
        }
        /// The followed fields of this name, when an attribute store
        /// goes through a receiver whose class is not known: every one
        /// of them may be the receiver's.
        fn fields_named(&self, e: &py::Expr) -> Vec<(String, bool)> {
            let py::Expr::Attribute(a) = e else {
                return Vec::new();
            };
            let receiver = self.typer.expr(&a.value);
            if !matches!(receiver, Ty::Unknown | Ty::Object) {
                return Vec::new();
            }
            let suffix = format!("#{}", a.attr.as_str());
            self.sites
                .keys()
                .filter(|k| k.ends_with(&suffix))
                .map(|k| (k.clone(), receiver == Ty::Unknown))
                .collect()
        }

        /// The key of a followed list: a name, or a field of an
        /// instance whose class is known.
        fn is_candidate(&self, e: &py::Expr) -> Option<String> {
            match e {
                py::Expr::Name(n) if self.sites.contains_key(n.id.as_str()) => {
                    Some(n.id.to_string())
                }
                py::Expr::Attribute(a) => {
                    let Ty::Class(k) = self.typer.expr(&a.value) else {
                        return None;
                    };
                    let key = field_key(k as usize, a.attr.as_str());
                    self.sites.contains_key(&key).then_some(key)
                }
                _ => None,
            }
        }
    }
    const READS: &[&str] = &[
        "pop", "sort", "reverse", "clear", "copy", "index", "count", "remove",
    ];
    const READERS: &[&str] = &[
        "len",
        "sorted",
        "reversed",
        "enumerate",
        "list",
        "tuple",
        "set",
        "sum",
        "min",
        "max",
        "str",
        "repr",
        "print",
        "bool",
        "any",
        "all",
    ];
    impl<'ast> Visitor<'ast> for Uses<'_, '_, 'ast> {
        fn visit_stmt(&mut self, stmt: &'ast py::Stmt) {
            match stmt {
                py::Stmt::Assign(a) => {
                    for t in &a.targets {
                        if let Some(name) = self.is_candidate(t) {
                            match unkinded_list(&a.value) {
                                None => self.drop_name(&name),
                                Some(count) => {
                                    if !is_empty_list(&a.value) {
                                        self.site(&name).none = true;
                                    }
                                    if let Some(n) = count {
                                        self.visit_expr(n);
                                    }
                                }
                            }
                            continue;
                        }
                        // `xs[i] = v` writes an element; `xs[a:b] = ys`
                        // writes ys's elements.
                        if let py::Expr::Subscript(sub) = t {
                            for (key, unknown) in self.fields_named(&sub.value) {
                                let site = self.site(&key);
                                if unknown {
                                    site.undecided = true;
                                } else if matches!(&*sub.slice, py::Expr::Slice(_)) {
                                    site.sequences.push(&a.value);
                                } else {
                                    site.elements.push(&a.value);
                                }
                            }
                            if let Some(name) = self.is_candidate(&sub.value) {
                                if matches!(&*sub.slice, py::Expr::Slice(_)) {
                                    self.site(&name).sequences.push(&a.value);
                                } else {
                                    self.site(&name).elements.push(&a.value);
                                }
                                self.visit_expr(&sub.slice);
                                self.visit_expr(&a.value);
                                return;
                            }
                        }
                    }
                    // The value is read whatever the targets are.
                    for t in &a.targets {
                        if self.is_candidate(t).is_none() {
                            self.visit_expr(t);
                        }
                    }
                    self.visit_expr(&a.value);
                }
                py::Stmt::AugAssign(a) => {
                    if let Some(name) = self.is_candidate(&a.target) {
                        if a.op == py::Operator::Add {
                            self.site(&name).sequences.push(&a.value);
                            self.visit_expr(&a.value);
                        } else {
                            self.drop_name(&name);
                        }
                        return;
                    }
                    walk_stmt(self, stmt);
                }
                py::Stmt::For(f) => {
                    if let py::Expr::Name(n) = &*f.target
                        && self.sites.contains_key(n.id.as_str())
                    {
                        self.drop_name(n.id.as_str());
                    }
                    if self.is_candidate(&f.iter).is_some() {
                        self.allowed.set(true);
                    }
                    walk_stmt(self, stmt);
                }
                py::Stmt::If(i) => {
                    if self.is_candidate(&i.test).is_some() {
                        self.allowed.set(true);
                    }
                    walk_stmt(self, stmt);
                }
                py::Stmt::While(w) => {
                    if self.is_candidate(&w.test).is_some() {
                        self.allowed.set(true);
                    }
                    walk_stmt(self, stmt);
                }
                py::Stmt::Delete(d) => {
                    for t in &d.targets {
                        if let Some(name) = self.is_candidate(t) {
                            self.drop_name(&name);
                        }
                    }
                    walk_stmt(self, stmt);
                }
                py::Stmt::Return(r)
                    if self.returns_read
                        && matches!(r.value.as_deref(), Some(py::Expr::Name(n)) if self.sites.contains_key(n.id.as_str())) =>
                    {}
                _ => walk_stmt(self, stmt),
            }
        }

        fn visit_expr(&mut self, expr: &'ast py::Expr) {
            let allowed = self.allowed.replace(false);
            match expr {
                py::Expr::Name(n) => {
                    if !allowed && self.sites.contains_key(n.id.as_str()) {
                        self.drop_name(n.id.as_str());
                    }
                }
                // A followed field read whole is kept; any other
                // attribute read reads its object.
                py::Expr::Attribute(a) if self.is_candidate(expr).is_some() => {
                    if !allowed {
                        let key = self.is_candidate(expr).unwrap();
                        self.drop_name(&key);
                    }
                    self.visit_expr(&a.value);
                }
                py::Expr::Call(c) => {
                    let args = &c.arguments.args;
                    match &*c.func {
                        py::Expr::Attribute(a) if !self.fields_named(&a.value).is_empty() => {
                            let method = a.attr.as_str();
                            for (key, unknown) in self.fields_named(&a.value) {
                                let site = self.site(&key);
                                match (method, args.len()) {
                                    _ if unknown => site.undecided = true,
                                    ("append", 1) => site.elements.push(&args[0]),
                                    ("insert", 2) => site.elements.push(&args[1]),
                                    ("extend", 1) => site.sequences.push(&args[0]),
                                    _ if READS.contains(&method) => {}
                                    _ => site.kept = true,
                                }
                            }
                            for arg in args {
                                self.visit_expr(arg);
                            }
                        }
                        py::Expr::Attribute(a) if self.is_candidate(&a.value).is_some() => {
                            let name = self.is_candidate(&a.value).unwrap();
                            let method = a.attr.as_str();
                            match (method, args.len()) {
                                ("append", 1) => self.site(&name).elements.push(&args[0]),
                                ("insert", 2) => self.site(&name).elements.push(&args[1]),
                                ("extend", 1) => self.site(&name).sequences.push(&args[0]),
                                _ if READS.contains(&method) => {}
                                _ => self.drop_name(&name),
                            }
                            for arg in args {
                                self.visit_expr(arg);
                            }
                            for k in &c.arguments.keywords {
                                self.visit_expr(&k.value);
                            }
                        }
                        // `sep.join(xs)`, `len(xs)` and the like read it.
                        py::Expr::Attribute(a)
                            if a.attr.as_str() == "join"
                                && args.len() == 1
                                && self.is_candidate(&args[0]).is_some() =>
                        {
                            self.visit_expr(&a.value);
                        }
                        py::Expr::Name(f) if READERS.contains(&f.id.as_str()) => {
                            for arg in args {
                                if self.is_candidate(arg).is_none() {
                                    self.visit_expr(arg);
                                }
                            }
                            for k in &c.arguments.keywords {
                                self.visit_expr(&k.value);
                            }
                        }
                        // A function of the module does with the list
                        // what its facts say of that parameter.
                        py::Expr::Name(f)
                            if self.module.funcs.contains_key(f.id.as_str())
                                && !self.sites.contains_key(f.id.as_str())
                                && c.arguments.keywords.is_empty() =>
                        {
                            for (i, arg) in args.iter().enumerate() {
                                match self.is_candidate(arg) {
                                    Some(name) => {
                                        self.site(&name).passed_to.push((f.id.to_string(), i));
                                    }
                                    None => self.visit_expr(arg),
                                }
                            }
                        }
                        _ => walk_expr(self, expr),
                    }
                }
                py::Expr::Subscript(sub) => {
                    if self.is_candidate(&sub.value).is_some() {
                        self.visit_expr(&sub.slice);
                        return;
                    }
                    walk_expr(self, expr);
                }
                py::Expr::Compare(c) => {
                    // `v in xs` reads it; comparing it is a use.
                    if c.ops.len() == 1
                        && matches!(c.ops[0], py::CmpOp::In | py::CmpOp::NotIn)
                        && self.is_candidate(&c.comparators[0]).is_some()
                    {
                        self.visit_expr(&c.left);
                        return;
                    }
                    walk_expr(self, expr);
                }
                py::Expr::UnaryOp(u) if u.op == py::UnaryOp::Not => {
                    if self.is_candidate(&u.operand).is_some() {
                        return;
                    }
                    walk_expr(self, expr);
                }
                py::Expr::BoolOp(b) => {
                    for v in &b.values {
                        if self.is_candidate(v).is_none() {
                            self.visit_expr(v);
                        }
                    }
                }
                py::Expr::Lambda(_)
                | py::Expr::ListComp(_)
                | py::Expr::SetComp(_)
                | py::Expr::DictComp(_)
                | py::Expr::Generator(_) => {
                    // A nested scope reads the name as a capture.
                    let mut names = NamesIn::default();
                    names.visit_expr(expr);
                    for n in names.0 {
                        if self.sites.contains_key(&n) {
                            self.drop_name(&n);
                        }
                    }
                }
                _ => walk_expr(self, expr),
            }
        }
    }
    #[derive(Default)]
    struct NamesIn(Vec<String>);
    impl<'ast> Visitor<'ast> for NamesIn {
        fn visit_expr(&mut self, expr: &'ast py::Expr) {
            if let py::Expr::Name(n) = expr {
                self.0.push(n.id.to_string());
            }
            ruff_python_ast::visitor::walk_expr(self, expr);
        }
    }

    let allowed = std::cell::Cell::new(false);
    let no_outer = HashMap::default();
    let mut uses = Uses {
        module,
        typer: Typer {
            module,
            vars,
            outer: &no_outer,
        },
        sites: names
            .into_iter()
            .map(|n| (n, ListSites::default()))
            .collect(),
        allowed: &allowed,
        returns_read,
    };
    for s in body {
        if let py::Stmt::FunctionDef(_) | py::Stmt::ClassDef(_) = s {
            continue;
        }
        uses.visit_stmt(s);
    }
    uses.sites
}

/// `x = v` followed by `assert isinstance(x, C)`, for a local `x` bound
/// nowhere else in the body: `x` is a `C`, and the binding does the
/// checking the assert asked for.
pub(crate) fn asserted_classes(
    body: &[py::Stmt],
    classes: &HashMap<String, usize>,
) -> HashMap<String, u16> {
    use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
    fn pairs(stmts: &[py::Stmt], classes: &HashMap<String, usize>, out: &mut Vec<(String, u16)>) {
        for (i, s) in stmts.iter().enumerate() {
            match s {
                py::Stmt::Assign(a) => {
                    let [py::Expr::Name(x)] = a.targets.as_slice() else {
                        continue;
                    };
                    let Some(py::Stmt::Assert(t)) = stmts.get(i + 1) else {
                        continue;
                    };
                    let py::Expr::Call(c) = &*t.test else {
                        continue;
                    };
                    if !is_name(&c.func, "isinstance") || c.arguments.args.len() != 2 {
                        continue;
                    }
                    let (py::Expr::Name(checked), py::Expr::Name(class)) =
                        (&c.arguments.args[0], &c.arguments.args[1])
                    else {
                        continue;
                    };
                    if checked.id != x.id {
                        continue;
                    }
                    if let Some(&k) = classes.get(class.id.as_str()) {
                        out.push((x.id.to_string(), k as u16));
                    }
                }
                py::Stmt::If(i) => {
                    pairs(&i.body, classes, out);
                    for c in &i.elif_else_clauses {
                        pairs(&c.body, classes, out);
                    }
                }
                py::Stmt::While(w) => pairs(&w.body, classes, out),
                py::Stmt::For(f) => pairs(&f.body, classes, out),
                py::Stmt::Try(t) => {
                    pairs(&t.body, classes, out);
                    for h in &t.handlers {
                        let py::ExceptHandler::ExceptHandler(h) = h;
                        pairs(&h.body, classes, out);
                    }
                    pairs(&t.orelse, classes, out);
                    pairs(&t.finalbody, classes, out);
                }
                _ => {}
            }
        }
    }
    let mut found = Vec::new();
    pairs(body, classes, &mut found);
    if found.is_empty() {
        return HashMap::default();
    }
    // Only a name stored once, by that assignment.
    #[derive(Default)]
    struct Stores(HashMap<String, usize>);
    impl<'a> Visitor<'a> for Stores {
        fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
            match stmt {
                py::Stmt::FunctionDef(f) => {
                    *self.0.entry(f.name.to_string()).or_default() += 1;
                }
                py::Stmt::ClassDef(c) => {
                    *self.0.entry(c.name.to_string()).or_default() += 1;
                }
                py::Stmt::Global(g) => {
                    for n in &g.names {
                        *self.0.entry(n.to_string()).or_default() += 2;
                    }
                }
                py::Stmt::Nonlocal(g) => {
                    for n in &g.names {
                        *self.0.entry(n.to_string()).or_default() += 2;
                    }
                }
                _ => {}
            }
            walk_stmt(self, stmt);
        }
        fn visit_expr(&mut self, expr: &'a py::Expr) {
            if let py::Expr::Name(n) = expr
                && !matches!(n.ctx, py::ExprContext::Load)
            {
                *self.0.entry(n.id.to_string()).or_default() += 1;
            }
            walk_expr(self, expr);
        }
    }
    let mut stores = Stores::default();
    for s in body {
        stores.visit_stmt(s);
    }
    found
        .into_iter()
        .filter(|(name, _)| stores.0.get(name).copied() == Some(1))
        .collect()
}

/// The locals of `body` bound only by unkinded literals, never in a
/// nested body: the ones whose kind the body's writes decide. A
/// parameter is bound by the call.
fn unkinded_locals(
    body: &[py::Stmt],
    params: &[(String, Ty)],
    seeds: &HashMap<String, Ty>,
    scope: &crate::scope::Scope,
) -> Vec<String> {
    use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
    /// Stores of each name, and how many of them bind an unkinded literal.
    #[derive(Default)]
    struct Stores(HashMap<String, (usize, usize)>);
    impl<'a> Visitor<'a> for Stores {
        fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
            match stmt {
                py::Stmt::Assign(a) if unkinded_list(&a.value).is_some() => {
                    for t in &a.targets {
                        if let py::Expr::Name(n) = t {
                            let e = self.0.entry(n.id.to_string()).or_default();
                            e.1 += 1;
                        }
                    }
                }
                py::Stmt::FunctionDef(f) => {
                    self.0.entry(f.name.to_string()).or_default().0 += 1;
                    return;
                }
                py::Stmt::ClassDef(c) => {
                    self.0.entry(c.name.to_string()).or_default().0 += 1;
                    return;
                }
                py::Stmt::Import(i) => {
                    for a in &i.names {
                        let bound = a.asname.as_ref().unwrap_or(&a.name);
                        let first = bound.split('.').next().unwrap_or_default();
                        self.0.entry(first.to_string()).or_default().0 += 1;
                    }
                }
                py::Stmt::ImportFrom(i) => {
                    for a in &i.names {
                        let bound = a.asname.as_ref().unwrap_or(&a.name);
                        self.0.entry(bound.to_string()).or_default().0 += 1;
                    }
                }
                py::Stmt::Try(t) => {
                    for h in &t.handlers {
                        let py::ExceptHandler::ExceptHandler(h) = h;
                        if let Some(n) = &h.name {
                            self.0.entry(n.to_string()).or_default().0 += 1;
                        }
                    }
                }
                _ => {}
            }
            walk_stmt(self, stmt);
        }
        fn visit_expr(&mut self, expr: &'a py::Expr) {
            if let py::Expr::Name(n) = expr
                && !matches!(n.ctx, py::ExprContext::Load)
            {
                self.0.entry(n.id.to_string()).or_default().0 += 1;
            }
            walk_expr(self, expr);
        }
    }
    let mut stores = Stores::default();
    for s in body {
        stores.visit_stmt(s);
    }
    let mut out: Vec<String> = stores
        .0
        .iter()
        .filter(|(name, (all, unkinded))| {
            *unkinded > 0
                && all == unkinded
                && scope.bound.contains(*name)
                && !params.iter().any(|(p, _)| p == *name)
                && !seeds.contains_key(*name)
                && !scope.globals.contains(*name)
                && !scope.nonlocals.contains(*name)
                && scope
                    .children
                    .iter()
                    .all(|(_, c)| !c.free.contains(*name) && !c.bound.contains(*name))
        })
        .map(|(name, _)| name.clone())
        .collect();
    out.sort();
    out
}

/// What a body puts into a list with these uses, against the types
/// known so far: `None` when the list is kept; `Some(None)` when
/// nothing is written in; `Some(Some(Unknown))` while something
/// written in is not yet typed; else the join of what is written.
fn written_into(module: &Module, sites: &ListSites<'_>, typer: &Typer<'_>) -> Option<Option<Ty>> {
    if sites.kept {
        return None;
    }
    let mut written = if sites.none { Some(Ty::None) } else { None };
    let mut undecided = sites.undecided;
    let mut take = |ty: Ty| {
        if ty == Ty::Unknown {
            undecided = true;
        } else {
            written = Some(written.unwrap_or(Ty::Unknown).join(ty));
        }
    };
    for e in &sites.elements {
        take(typer.expr(e));
    }
    for e in &sites.sequences {
        take(typer.expr(e).element().unwrap_or(Ty::Object));
    }
    for (callee, index) in &sites.passed_to {
        match module
            .list_params
            .get(callee)
            .and_then(|facts| facts.get(*index).copied())
        {
            Some(ListFact::Reads) => {}
            Some(ListFact::Writes(ty)) => take(ty),
            Some(ListFact::Kept) | None => return None,
        }
    }
    if undecided {
        return Some(Some(Ty::Unknown));
    }
    Some(written)
}

/// The kind a list bound to an unkinded literal has: `Unknown` while
/// what goes in is not yet typed, a list of anything once it is kept,
/// written with more than one kind, or never written at all.
fn decide_list(module: &Module, sites: &ListSites<'_>, typer: &Typer<'_>) -> Ty {
    match written_into(module, sites, typer) {
        None | Some(None) => Ty::List(Elem::Object),
        Some(Some(Ty::Unknown)) => Ty::Unknown,
        Some(Some(ty)) => Ty::List(Elem::of(ty)),
    }
}

/// What the program's bodies do with a field bound to an unkinded
/// list, gathered over one round of inference.
#[derive(Default, Debug, Clone)]
struct FieldRound {
    kept: bool,
    none: bool,
    /// What is written in, typed where it is written; `Unknown` for a
    /// write not yet typed.
    written: Vec<Ty>,
}

impl FieldRound {
    /// The list's kind: as [`decide_list`] decides a local's.
    fn decide(&self) -> Ty {
        if self.kept {
            return Ty::List(Elem::Object);
        }
        let mut joined = if self.none { Some(Ty::None) } else { None };
        for ty in &self.written {
            if *ty == Ty::Unknown {
                return Ty::Unknown;
            }
            joined = Some(joined.unwrap_or(Ty::Unknown).join(*ty));
        }
        match joined {
            None => Ty::List(Elem::Object),
            Some(ty) => Ty::List(Elem::of(ty)),
        }
    }
}

/// Add what `body` does with each field in `keys` to `rounds`.
fn field_sites_into(
    module: &Module,
    body: &[py::Stmt],
    vars: &HashMap<String, Ty>,
    keys: &[String],
    rounds: &mut HashMap<String, FieldRound>,
) {
    let sites = list_sites(module, body, vars, keys.to_vec(), false);
    let no_outer = HashMap::default();
    let typer = Typer {
        module,
        vars,
        outer: &no_outer,
    };
    for (key, site) in sites {
        let round = rounds.entry(key).or_default();
        // A field handed to a function is out of sight.
        round.kept |= site.kept || !site.passed_to.is_empty();
        round.none |= site.none;
        if site.undecided {
            round.written.push(Ty::Unknown);
        }
        for e in &site.elements {
            round.written.push(typer.expr(e));
        }
        for e in &site.sequences {
            round
                .written
                .push(typer.expr(e).element().unwrap_or(Ty::Object));
        }
    }
}

/// What a body does with each of its parameters, in parameter order;
/// see [`Module::list_params`].
fn list_param_facts(
    module: &Module,
    body: &[py::Stmt],
    vars: &HashMap<String, Ty>,
    params: &[(String, Ty)],
) -> Vec<ListFact> {
    let scope = crate::scope::Scope::of_body(Vec::new(), body);
    let followed: Vec<String> = params
        .iter()
        .filter(|(name, _)| {
            !scope.bound.contains(name)
                && !scope.globals.contains(name)
                && !scope.nonlocals.contains(name)
                && scope
                    .children
                    .iter()
                    .all(|(_, c)| !c.free.contains(name) && !c.bound.contains(name))
        })
        .map(|(name, _)| name.clone())
        .collect();
    let sites = list_sites(module, body, vars, followed, false);
    let no_outer = HashMap::default();
    let typer = Typer {
        module,
        vars,
        outer: &no_outer,
    };
    params
        .iter()
        .map(|(name, _)| match sites.get(name) {
            None => ListFact::Kept,
            Some(site) => match written_into(module, site, &typer) {
                None => ListFact::Kept,
                Some(None) => ListFact::Reads,
                Some(Some(ty)) => ListFact::Writes(ty),
            },
        })
        .collect()
}

/// Whether control never reaches the end of `stmts`: some statement in
/// the list leaves the function on every path. Anything not shown to
/// leave is taken to fall through.
pub(crate) fn terminates(stmts: &[py::Stmt]) -> bool {
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
    /// The uses of each local bound only to unkinded list literals,
    /// whose kind is decided from them at each such binding.
    fills: &'a HashMap<String, ListSites<'a>>,
}

impl Walker<'_> {
    fn typer(&self) -> Typer<'_> {
        Typer {
            module: self.module,
            vars: &self.locals.vars,
            outer: self.seeds,
        }
    }

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
            // A value the declared type admits, None into an instance,
            // a subclass instance into a base, changes nothing. A dict
            // written with another kind widens at the call sites, through
            // what was recorded above, and is read here as declared.
            if !matches!(*declared, Ty::Object | Ty::Unknown)
                && ty != Ty::Unknown
                && !matches!((*declared, ty), (Ty::Dict(_), Ty::Dict(_)))
                && self.module.join_classes(*declared, ty) != *declared
            {
                self.locals.vars.insert(name.to_string(), Ty::Object);
            }
            return;
        }
        let current = self.locals.vars.get(name).copied().unwrap_or(Ty::Unknown);
        let joined = self.module.join_classes(current, ty);
        self.locals.vars.insert(name.to_string(), joined);
    }

    /// `obj.x = v`: on an instance of a known class widens that class's
    /// field; on a dynamic value, the field of every class that has one,
    /// since any of them may be the receiver.
    fn field_write(&mut self, a: &py::ExprAttribute, ty: Ty) {
        match self.expr(&a.value) {
            Ty::Class(k) => {
                self.locals
                    .other_field_writes
                    .push((k as usize, a.attr.to_string(), ty));
            }
            Ty::Object => {
                for (k, class) in self.module.classes.iter().enumerate() {
                    if class.fields.iter().any(|(f, _)| f == a.attr.as_str()) {
                        self.locals
                            .other_field_writes
                            .push((k, a.attr.to_string(), ty));
                    }
                }
            }
            _ => {}
        }
    }

    fn target(&mut self, target: &py::Expr, ty: Ty) {
        match target {
            py::Expr::Name(n) => self.assign(n.id.as_str(), ty),
            // `self.x = v` in a method declares the field, in the order
            // the fields are laid out; the first parameter of a module
            // function is not `self`, so the write also goes the way any
            // other instance's does.
            py::Expr::Attribute(a)
                if matches!(&*a.value, py::Expr::Name(n)
                    if self.params.first().is_some_and(|(p, _)| p == n.id.as_str())) =>
            {
                let current = self
                    .locals
                    .field_writes
                    .get(a.attr.as_str())
                    .copied()
                    .unwrap_or(Ty::Unknown);
                let joined = self.module.join_classes(current, ty);
                self.locals.field_writes.insert(a.attr.to_string(), joined);
                self.field_write(a, ty);
            }
            // `C.X = v`: the class attribute's module variable.
            py::Expr::Attribute(a) if let Some(k) = self.module.class_of_expr(&a.value) => {
                if let Some(attr) = self.module.class_attr(k, a.attr.as_str()) {
                    let global = attr.global.clone();
                    self.assign(&global, ty);
                }
            }
            py::Expr::Attribute(a) => self.field_write(a, ty),
            // `d[k] = v` on a dict held by a name widens the dict's keys
            // by the key's type and its values by the value's.
            py::Expr::Subscript(sub) => {
                if let py::Expr::Name(n) = &*sub.value
                    && let Ty::Dict(k) = self.expr(&sub.value)
                {
                    let (key, value) = dict_shape(k);
                    let written = self.expr(&sub.slice);
                    let widened = dict_of(key.join(written), value.join(ty));
                    if widened != Ty::Dict(k) {
                        self.assign(n.id.as_str(), widened);
                    }
                }
            }
            // Unpacking gives every name an element: its own from a
            // tuple of that arity, the element type from a list, and
            // otherwise whatever the runtime finds.
            py::Expr::Tuple(t) => {
                for (e, ty) in t.elts.iter().zip(unpacked(ty, &t.elts)) {
                    self.target(e, ty);
                }
            }
            py::Expr::List(l) => {
                for (e, ty) in l.elts.iter().zip(unpacked(ty, &l.elts)) {
                    self.target(e, ty);
                }
            }
            _ => {}
        }
    }

    /// A tuple literal gives each unpacked name the type of its own
    /// expression. No tuple escapes this assignment, so the lowering can
    /// bind those values directly without putting them through `Any`.
    fn target_tuple_literal(&mut self, target: &py::Expr, value: &py::Expr) -> bool {
        let py::Expr::Tuple(source) = value else {
            return false;
        };
        let targets = match target {
            py::Expr::Tuple(t) => &t.elts,
            py::Expr::List(l) => &l.elts,
            _ => return false,
        };
        if targets.len() != source.elts.len() {
            return false;
        }
        if targets
            .iter()
            .any(|target| matches!(target, py::Expr::Tuple(_) | py::Expr::List(_)))
        {
            return false;
        }
        for (target, value) in targets.iter().zip(&source.elts) {
            let ty = self.expr(value);
            self.target(target, ty);
        }
        true
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
                    if a.targets.len() == 1 && self.target_tuple_literal(t, &a.value) {
                        continue;
                    }
                    // An unkinded literal is the list its writes make
                    // it, decided afresh from the types known now.
                    if let py::Expr::Name(n) = t {
                        if let Some(sites) = self.fills.get(n.id.as_str()) {
                            let decided = decide_list(self.module, sites, &self.typer());
                            self.locals.vars.remove(n.id.as_str());
                            self.assign(n.id.as_str(), decided);
                            continue;
                        }
                        // The assert on the next line says what it is.
                        if let Some(&k) = self.locals.narrowed.get(n.id.as_str()) {
                            self.assign(n.id.as_str(), Ty::Class(k));
                            continue;
                        }
                    }
                    // `self.f = []` takes the kind the program's writes
                    // into the field decided, nothing until they have.
                    if let py::Expr::Attribute(attr) = t
                        && unkinded_list(&a.value).is_some()
                        && let Ty::Class(k) = self.expr(&attr.value)
                    {
                        let key = (k as usize, attr.attr.to_string());
                        if self.module.list_fields.contains(&key) {
                            let decided = self
                                .module
                                .field_lists
                                .get(&key)
                                .copied()
                                .unwrap_or(Ty::Unknown);
                            self.target(t, decided);
                            continue;
                        }
                    }
                    self.target(t, ty);
                }
            }
            py::Stmt::AnnAssign(a) => {
                if let Some(v) = &a.value {
                    // `x: Any = v` asks for a dynamic variable, as an
                    // unannotated parameter is one.
                    let ty = if is_dynamic_annotation(&a.annotation) {
                        Ty::Object
                    } else if let Some(e) =
                        annotated_empty_list(&self.module.class_index, &a.annotation, v)
                    {
                        Ty::List(e)
                    } else {
                        annotated_value(&self.module.class_index, &a.annotation, self.expr(v))
                    };
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
                self.expr(&f.iter);
                let item = self.typer().item_ty(&f.iter);
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
            // `with e as x` binds x to e: the file itself.
            py::Stmt::With(w) => {
                for item in &w.items {
                    let ty = self.expr(&item.context_expr);
                    if let Some(target) = &item.optional_vars {
                        self.target(target, ty);
                    }
                }
                self.stmts(&w.body)
            }
            py::Stmt::FunctionDef(f) => {
                let ty = self
                    .module
                    .closure_at(crate::lower::current_file(), f.range.start().to_u32())
                    .map(Ty::Closure)
                    .unwrap_or(Ty::Object);
                self.assign(f.name.as_str(), ty)
            }
            // A class of the module is not a variable: its name is the
            // class wherever it is written. What its body declares and
            // does not fix is written to its module variable here.
            py::Stmt::ClassDef(c) => {
                let Some(&k) = self.module.class_index.get(c.name.as_str()) else {
                    self.assign(c.name.as_str(), Ty::Object);
                    return;
                };
                if let Ok(declared) = crate::class_attrs::declared_in(c) {
                    for d in declared {
                        let Some(attr) = self.module.class_attr(k, &d.name) else {
                            continue;
                        };
                        if attr.constant.is_some() {
                            continue;
                        }
                        let global = attr.global.clone();
                        let ty = self.expr(d.value);
                        self.assign(&global, ty);
                    }
                }
            }
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
            for (e, ty) in t.elts.iter().zip(unpacked(ty, &t.elts)) {
                bind_target(vars, e, ty);
            }
        }
        py::Expr::List(l) => {
            for (e, ty) in l.elts.iter().zip(unpacked(ty, &l.elts)) {
                bind_target(vars, e, ty);
            }
        }
        _ => {}
    }
}

/// What each of `targets` receives when a value of type `ty` is
/// unpacked into them: the shape's own elements from a tuple of that
/// arity, the element type of a list or string, and a dynamic value
/// otherwise or wherever a starred target takes a slice.
pub(crate) fn unpacked(ty: Ty, targets: &[py::Expr]) -> Vec<Ty> {
    let starred = targets.iter().any(|t| matches!(t, py::Expr::Starred(_)));
    let each = match ty {
        Ty::Tuple(k) if !starred => {
            let shape = tuple_shape(k);
            if shape.len() == targets.len() {
                return shape;
            }
            Ty::Object
        }
        Ty::List(e) if !starred => e.ty(),
        Ty::Str if !starred => Ty::Str,
        _ => Ty::Object,
    };
    vec![each; targets.len()]
}

/// Whether `body` stores to a class attribute, `C.X = v`, which makes
/// the function a writer of the module variable behind it.
pub(crate) fn writes_class_attrs(class_index: &HashMap<String, usize>, body: &[py::Stmt]) -> bool {
    !class_attr_writes(class_index, body).is_empty()
}

/// The classes and attribute names `body` stores to as `C.X = v`.
pub(crate) fn class_attr_writes(
    class_index: &HashMap<String, usize>,
    body: &[py::Stmt],
) -> Vec<(usize, String)> {
    use py::visitor::{Visitor, walk_expr, walk_stmt};
    struct Finder<'a> {
        classes: &'a HashMap<String, usize>,
        found: Vec<(usize, String)>,
    }
    impl Finder<'_> {
        fn target(&mut self, t: &py::Expr) {
            match t {
                py::Expr::Attribute(a) => {
                    if let Some(k) = crate::class_attrs::class_named(self.classes, &a.value) {
                        self.found.push((k, a.attr.to_string()));
                    }
                }
                py::Expr::Tuple(t) => t.elts.iter().for_each(|e| self.target(e)),
                py::Expr::List(l) => l.elts.iter().for_each(|e| self.target(e)),
                py::Expr::Starred(s) => self.target(&s.value),
                _ => {}
            }
        }
    }
    impl<'a> Visitor<'a> for Finder<'_> {
        fn visit_stmt(&mut self, s: &'a py::Stmt) {
            match s {
                py::Stmt::Assign(a) => a.targets.iter().for_each(|t| self.target(t)),
                py::Stmt::AugAssign(a) => self.target(&a.target),
                py::Stmt::AnnAssign(a) => self.target(&a.target),
                py::Stmt::For(f) => self.target(&f.target),
                // A nested function's stores are its own.
                py::Stmt::FunctionDef(_) => return,
                _ => {}
            }
            walk_stmt(self, s);
        }
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Named(n) = e {
                self.target(&n.target);
            }
            walk_expr(self, e);
        }
    }
    let mut f = Finder {
        classes: class_index,
        found: Vec::new(),
    };
    for s in body {
        f.visit_stmt(s);
    }
    f.found
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

/// Every operator a class can define a method for.
pub(crate) const OPERATORS: [py::Operator; 13] = [
    py::Operator::Add,
    py::Operator::Sub,
    py::Operator::Mult,
    py::Operator::Div,
    py::Operator::FloorDiv,
    py::Operator::Mod,
    py::Operator::Pow,
    py::Operator::MatMult,
    py::Operator::BitAnd,
    py::Operator::BitOr,
    py::Operator::BitXor,
    py::Operator::LShift,
    py::Operator::RShift,
];

/// Whether `name` is the method of one of [`OPERATORS`].
pub(crate) fn is_operator_method(name: &str) -> bool {
    OPERATORS.iter().any(|op| dunder_name(*op) == name)
}

/// The method a class defines to take part in `op`.
pub(crate) fn dunder_name(op: py::Operator) -> &'static str {
    match op {
        py::Operator::Add => "__add__",
        py::Operator::Sub => "__sub__",
        py::Operator::Mult => "__mul__",
        py::Operator::Div => "__truediv__",
        py::Operator::FloorDiv => "__floordiv__",
        py::Operator::Mod => "__mod__",
        py::Operator::Pow => "__pow__",
        py::Operator::MatMult => "__matmul__",
        py::Operator::BitAnd => "__and__",
        py::Operator::BitOr => "__or__",
        py::Operator::BitXor => "__xor__",
        py::Operator::LShift => "__lshift__",
        py::Operator::RShift => "__rshift__",
    }
}

/// What `left op right` produces. `/` is always a float on numbers,
/// `**` with a negative literal exponent too.
pub(crate) fn binop(op: py::Operator, l: Ty, r: Ty, right: &py::Expr) -> Ty {
    match op {
        // A `%` format.
        py::Operator::Mod if l == Ty::Str => Ty::Str,
        py::Operator::Mod if l == Ty::Bytes => Ty::Bytes,
        py::Operator::Add if l == Ty::Bytes && r == Ty::Bytes => Ty::Bytes,
        py::Operator::Mult
            if (l == Ty::Bytes && matches!(r, Ty::Int | Ty::Bool))
                || (matches!(l, Ty::Int | Ty::Bool) && r == Ty::Bytes) =>
        {
            Ty::Bytes
        }
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
        // Two shapes concatenate into one.
        py::Operator::Add if let (Ty::Tuple(a), Ty::Tuple(b)) = (l, r) => {
            let mut elems = tuple_shape(a);
            elems.extend(tuple_shape(b));
            tuple_of(elems)
        }
        py::Operator::Mult if matches!(l, Ty::List(_)) && matches!(r, Ty::Int | Ty::Bool) => l,
        py::Operator::Mult if matches!(r, Ty::List(_)) && matches!(l, Ty::Int | Ty::Bool) => r,
        // A repeated tuple has no shape the count does not decide.
        py::Operator::Mult
            if (matches!(l, Ty::Tuple(_)) && matches!(r, Ty::Int | Ty::Bool))
                || (matches!(r, Ty::Tuple(_)) && matches!(l, Ty::Int | Ty::Bool)) =>
        {
            Ty::Object
        }
        py::Operator::Mult
            if (l == Ty::Str && matches!(r, Ty::Int | Ty::Bool))
                || (matches!(l, Ty::Int | Ty::Bool) && r == Ty::Str) =>
        {
            Ty::Str
        }
        _ if matches!(l, Ty::Str | Ty::Bytes) || matches!(r, Ty::Str | Ty::Bytes) => {
            if l == Ty::Unknown || r == Ty::Unknown {
                Ty::Unknown
            } else {
                Ty::Object
            }
        }
        _ => l.arith(r),
    }
}

/// Call arguments with each `*name` whose tuple shape is known spread
/// into that many element reads, and no other starred argument. None
/// when there is no starred argument, or one that is not such a name.
pub(crate) fn spread_starred(
    args: &[py::Expr],
    ty_of: impl Fn(&py::Expr) -> Ty,
) -> Option<Vec<py::Expr>> {
    if !args.iter().any(|a| matches!(a, py::Expr::Starred(_))) {
        return None;
    }
    let mut out = Vec::with_capacity(args.len() + 2);
    for a in args {
        let py::Expr::Starred(s) = a else {
            out.push(a.clone());
            continue;
        };
        // Only a name is read more than once without effect.
        let py::Expr::Name(_) = &*s.value else {
            return None;
        };
        let Ty::Tuple(k) = ty_of(&s.value) else {
            return None;
        };
        for i in 0..tuple_shape(k).len() {
            out.push(py::Expr::Subscript(py::ExprSubscript {
                node_index: Default::default(),
                range: s.range,
                value: s.value.clone(),
                slice: Box::new(py::Expr::NumberLiteral(py::ExprNumberLiteral {
                    node_index: Default::default(),
                    range: s.range,
                    value: py::Number::Int(py::Int::from(i as u64)),
                })),
                ctx: py::ExprContext::Load,
            }));
        }
    }
    Some(out)
}

/// The mode `open(path, mode)` names, when it is a literal; `r` when
/// left out. None when it is computed.
pub(crate) fn open_mode(c: &py::ExprCall) -> Option<Mode> {
    let mode = c
        .arguments
        .args
        .get(1)
        .or_else(|| c.arguments.find_keyword("mode").map(|k| &k.value));
    match mode {
        None => Some(Mode::Text),
        Some(py::Expr::StringLiteral(s)) => Some(Mode::of(s.value.to_str())),
        Some(_) => None,
    }
}

/// The value of an integer literal, possibly negated.
pub(crate) fn int_literal(e: &py::Expr) -> Option<i64> {
    match e {
        py::Expr::NumberLiteral(n) => match &n.value {
            py::Number::Int(i) => i.as_i64(),
            _ => None,
        },
        py::Expr::UnaryOp(u) if u.op == py::UnaryOp::USub => int_literal(&u.operand)?.checked_neg(),
        py::Expr::UnaryOp(u) if u.op == py::UnaryOp::UAdd => int_literal(&u.operand),
        _ => None,
    }
}

/// The element a literal subscript names in a tuple of `len` elements,
/// counting from the end when negative; None when it is not a literal
/// or is out of range.
pub(crate) fn constant_index(e: &py::Expr, len: usize) -> Option<usize> {
    let i = int_literal(e)?;
    let i = if i < 0 { i + len as i64 } else { i };
    (0..len as i64).contains(&i).then_some(i as usize)
}

/// The indexes a slice with literal bounds selects from a sequence of
/// `len` elements, by Python's rules.
pub(crate) fn tuple_slice(len: usize, slice: &py::ExprSlice) -> Option<Vec<usize>> {
    let n = len as i64;
    let literal = |e: &Option<Box<py::Expr>>| -> Option<Option<i64>> {
        match e {
            None => Some(None),
            Some(e) => int_literal(e).map(Some),
        }
    };
    let step = literal(&slice.step)?.unwrap_or(1);
    if step == 0 {
        return None;
    }
    let (lo, hi) = if step > 0 { (0, n) } else { (-1, n - 1) };
    let bound = |v: Option<i64>, default: i64| match v {
        None => default,
        Some(v) if v < 0 => (v + n).clamp(lo, hi),
        Some(v) => v.clamp(lo, hi),
    };
    let (start, stop) = if step > 0 {
        (
            bound(literal(&slice.lower)?, 0),
            bound(literal(&slice.upper)?, n),
        )
    } else {
        (
            bound(literal(&slice.lower)?, n - 1),
            bound(literal(&slice.upper)?, -1),
        )
    };
    let mut out = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        out.push(i as usize);
        i += step;
    }
    Some(out)
}

fn nonnegative_literal(e: &py::Expr) -> bool {
    matches!(e, py::Expr::NumberLiteral(n) if matches!(n.value, py::Number::Int(_)))
        || matches!(e, py::Expr::UnaryOp(u) if u.op == py::UnaryOp::UAdd
            && matches!(&*u.operand, py::Expr::NumberLiteral(_)))
}

impl Typer<'_> {
    /// The element type of `elt` under the loop variables of
    /// `generators`: what a comprehension over them produces, or
    /// `Unknown` while that is not yet typed.
    pub(crate) fn comprehension_elem(
        &self,
        generators: &[py::Comprehension],
        elt: &py::Expr,
    ) -> Ty {
        let vars = self.comprehension_vars(generators);
        let inner = Typer {
            module: self.module,
            vars: &vars,
            outer: self.outer,
        };
        inner.expr(elt)
    }

    /// The key and value types of a dict comprehension.
    fn comprehension_pair(
        &self,
        generators: &[py::Comprehension],
        key: &py::Expr,
        value: &py::Expr,
    ) -> (Ty, Ty) {
        let vars = self.comprehension_vars(generators);
        let inner = Typer {
            module: self.module,
            vars: &vars,
            outer: self.outer,
        };
        (inner.expr(key), inner.expr(value))
    }

    /// The variables in scope inside a comprehension: this scope's, with
    /// each generator's target bound to what one round of it yields,
    /// where a later generator sees the earlier targets.
    pub(crate) fn comprehension_vars(
        &self,
        generators: &[py::Comprehension],
    ) -> HashMap<String, Ty> {
        let mut vars = self.vars.clone();
        for g in generators {
            let item = Typer {
                module: self.module,
                vars: &vars,
                outer: self.outer,
            }
            .item_ty(&g.iter);
            bind_target(&mut vars, &g.target, item);
        }
        vars
    }

    /// What one round of iterating `iter` binds: an int from `range`,
    /// the (index, element) pair from `enumerate`, else an element of
    /// the iterable.
    pub(crate) fn item_ty(&self, iter: &py::Expr) -> Ty {
        if let py::Expr::Call(c) = iter
            && let py::Expr::Name(n) = &*c.func
            && !self.is_variable(n.id.as_str())
            && c.arguments.keywords.is_empty()
        {
            match (n.id.as_str(), c.arguments.args.len()) {
                ("range", 1..=3) => return Ty::Int,
                ("enumerate", 1 | 2) => {
                    let item = self
                        .expr(&c.arguments.args[0])
                        .element()
                        .unwrap_or(Ty::Object);
                    return tuple_of(vec![Ty::Int, item]);
                }
                _ => {}
            }
        }
        self.expr(iter).element().unwrap_or(Ty::Object)
    }

    /// Whether `name` is a variable of this scope, an enclosing one or
    /// the module, so a builtin of that name is shadowed.
    fn is_variable(&self, name: &str) -> bool {
        self.vars.contains_key(name)
            || self.outer.contains_key(name)
            || self.module.globals.contains_key(name)
    }

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
            py::Expr::BytesLiteral(_) => Ty::Bytes,
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
                if let Some(m) = self.module.imported_name(name) {
                    return if matches!(m, crate::stdlib::Member::Func { .. }) {
                        Ty::Object
                    } else {
                        member_ty(m)
                    };
                }
                if !self.module.funcs.contains_key(name)
                    && !self.module.class_index.contains_key(name)
                    && let Some(k) = builtin_index(name)
                {
                    return Ty::Builtin(k);
                }
                Ty::Object
            }
            py::Expr::BinOp(b) => {
                let l = self.expr(&b.left);
                let r = self.expr(&b.right);
                // An instance takes part through its class's method, and
                // the result is what that method returns.
                if let Ty::Class(k) = l
                    && let Some((sig, _)) = self.module.method_sig(k as usize, dunder_name(b.op))
                {
                    return sig.ret;
                }
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
                    return if matches!(m, crate::stdlib::Member::Func { .. }) {
                        Ty::Object
                    } else {
                        member_ty(m)
                    };
                }
                if let Some(k) = self
                    .module
                    .bound_at(crate::lower::current_file(), a.range.start().to_u32())
                {
                    let receiver = self.expr(&a.value);
                    if bound_method_arity(self.module, receiver, a.attr.as_str()).is_some() {
                        return Ty::Bound(k);
                    }
                }
                if let Some(k) = self.module.class_of_expr(&a.value) {
                    return match self.module.class_attr(k, a.attr.as_str()) {
                        Some(attr) => self.module.class_attr_ty(attr),
                        None => Ty::Object,
                    };
                }
                match self.expr(&a.value) {
                    Ty::Class(k) => self
                        .module
                        .field(k as usize, a.attr.as_str())
                        .map(|(_, ty)| ty)
                        .or_else(|| {
                            self.module
                                .class_attr(k as usize, a.attr.as_str())
                                .map(|attr| self.module.class_attr_ty(attr))
                        })
                        .unwrap_or(if self.module.settled.get() {
                            Ty::Object
                        } else {
                            Ty::Unknown
                        }),
                    // An attribute of None raises; its type is what the
                    // other paths to the name decide.
                    Ty::Unknown | Ty::None => Ty::Unknown,
                    // An array's typecode and element size.
                    Ty::List(Elem::Array(_)) if a.attr.as_str() == "typecode" => Ty::Str,
                    Ty::List(Elem::Array(_)) if a.attr.as_str() == "itemsize" => Ty::Int,
                    _ => Ty::Object,
                }
            }
            py::Expr::Subscript(s) => {
                let seq = self.expr(&s.value);
                if seq == Ty::Unknown {
                    return Ty::Unknown;
                }
                if let py::Expr::Slice(slice) = &*s.slice {
                    match seq {
                        Ty::Str | Ty::Bytes | Ty::List(_) => seq,
                        // A slice of a tuple with literal bounds is the
                        // shape those bounds cut out.
                        Ty::Tuple(k) => {
                            let shape = tuple_shape(k);
                            tuple_slice(shape.len(), slice)
                                .map(|picked| tuple_of(picked.iter().map(|&i| shape[i]).collect()))
                                .unwrap_or(Ty::Object)
                        }
                        _ => Ty::Object,
                    }
                } else if let Ty::Tuple(k) = seq {
                    let shape = tuple_shape(k);
                    match constant_index(&s.slice, shape.len()) {
                        Some(i) => shape[i],
                        None => seq.element().unwrap_or(Ty::Object),
                    }
                } else if let Ty::Dict(k) = seq {
                    dict_shape(k).1
                } else {
                    seq.element().unwrap_or(Ty::Object)
                }
            }
            py::Expr::List(l) => match self.elem_of(l.elts.iter()) {
                Some(e) => Ty::List(e),
                None => Ty::Unknown,
            },
            py::Expr::ListComp(c) => match self.comprehension_elem(&c.generators, &c.elt) {
                Ty::Unknown => Ty::Unknown,
                elem => Ty::List(Elem::of(elem)),
            },
            // A starred element spreads a sequence of a length only the
            // runtime knows.
            py::Expr::Tuple(t) => {
                if t.elts.iter().any(|e| matches!(e, py::Expr::Starred(_))) {
                    Ty::Object
                } else {
                    tuple_of(t.elts.iter().map(|e| self.expr(e)).collect())
                }
            }
            // A literal's shape is the join of its keys and of its values;
            // a spread brings keys and values of every kind.
            py::Expr::Dict(d) => {
                if d.items.iter().any(|item| item.key.is_none()) {
                    return dynamic_dict();
                }
                let mut key = Ty::Unknown;
                let mut value = Ty::Unknown;
                for item in &d.items {
                    if let Some(k) = &item.key {
                        key = key.join(self.expr(k));
                    }
                    value = value.join(self.expr(&item.value));
                }
                dict_of(key, value)
            }
            py::Expr::DictComp(c) => {
                let Some(key) = &c.key else {
                    return dynamic_dict();
                };
                let (k, v) = self.comprehension_pair(&c.generators, key, &c.value);
                dict_of(k, v)
            }
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
    /// or dynamic when they differ; `None` while an element is not yet
    /// typed.
    fn elem_of<'e>(&self, elts: impl Iterator<Item = &'e py::Expr>) -> Option<Elem> {
        let mut kind: Option<Elem> = None;
        for e in elts {
            let ty = self.expr(e);
            if ty == Ty::Unknown {
                return None;
            }
            let k = Elem::of(ty);
            kind = Some(match kind {
                None => k,
                Some(prev) if prev == k => k,
                Some(_) => return Some(Elem::Object),
            });
        }
        Some(kind.unwrap_or(Elem::Object))
    }

    /// The class a bare name is, when no variable shadows it.
    pub(crate) fn class_named(&self, e: &py::Expr) -> Option<usize> {
        let py::Expr::Name(n) = e else {
            return None;
        };
        let name = n.id.as_str();
        if self.vars.contains_key(name)
            || self.outer.contains_key(name)
            || self.module.globals.contains_key(name)
        {
            return None;
        }
        self.module.class_index.get(name).copied()
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

    /// What a call of a module member returns.
    fn member_call_ty(&self, m: crate::stdlib::Member, c: &py::ExprCall) -> Ty {
        match m {
            crate::stdlib::Member::ArrayType => array_call_ty(&c.arguments.args),
            crate::stdlib::Member::Binary(op) if c.arguments.args.len() == 2 => {
                let l = self.expr(&c.arguments.args[0]);
                let r = self.expr(&c.arguments.args[1]);
                binop(op, l, r, &c.arguments.args[1])
            }
            other => member_ty(other),
        }
    }

    fn call(&self, c: &py::ExprCall) -> Ty {
        if let py::Expr::Attribute(a) = &*c.func
            && is_name(&a.value, "frozenset")
            && a.attr.as_str() == "union"
        {
            return Ty::Set;
        }
        if let py::Expr::Attribute(a) = &*c.func
            && let Some(m) = self.module_member_of(&a.value, a.attr.as_str())
        {
            return self.member_call_ty(m, c);
        }
        // `Class.method(obj, ...)` is the method.
        if let py::Expr::Attribute(a) = &*c.func
            && let Some(k) = self.class_named(&a.value)
        {
            return match self.module.method_sig(k, a.attr.as_str()) {
                Some((sig, _)) => sig.ret,
                None => Ty::Object,
            };
        }
        // A call through a value whose function is known returns what
        // that function returns.
        match self.callee_ty(&c.func) {
            Ty::Closure(k) => return self.module.closure_ret(k),
            Ty::Bound(k) => {
                let info = &self.module.bounds[k as usize];
                let receiver = self.expr(&py::Expr::Name(info.receiver.clone()));
                return self.method_ret(receiver, &info.method);
            }
            Ty::Builtin(k) => return self.builtin_call(BUILTIN_VALUES[k as usize], c),
            // `instance(...)` is its class's `__call__`.
            Ty::Class(k) if !matches!(&*c.func, py::Expr::Name(n) if self.module.class_index.contains_key(n.id.as_str())) =>
            {
                return self
                    .module
                    .dispatched_ret(k as usize, "__call__")
                    .unwrap_or(Ty::Object);
            }
            _ => {}
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
                if !self.vars.contains_key(name)
                    && !self.outer.contains_key(name)
                    && let Some(m) = self.module.imported_name(name)
                {
                    return self.member_call_ty(m, c);
                }
                self.builtin_call(name, c)
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
            // `d.get(k)` is a value or None; with a default, a value or
            // the default.
            py::Expr::Attribute(a)
                if a.attr.as_str() == "get"
                    && let Ty::Dict(k) = self.expr(&a.value) =>
            {
                let value = dict_shape(k).1;
                match c.arguments.args.get(1) {
                    Some(default) => value.join(self.expr(default)),
                    None => value.join(Ty::None),
                }
            }
            // A method on a value whose type is known, or not yet.
            py::Expr::Attribute(a) => self.method_ret(self.expr(&a.value), a.attr.as_str()),
            _ => Ty::Object,
        }
    }

    /// What a call of the builtin `name` returns.
    fn builtin_call(&self, name: &str, c: &py::ExprCall) -> Ty {
        let args = &c.arguments.args;
        let arg = |i: usize| args.get(i).map(|a| self.expr(a)).unwrap_or(Ty::Unknown);
        match name {
            "print" => Ty::None,
            // A range is iterated as ints.
            "range" => Ty::List(Elem::Int),
            "len" | "int" | "ord" | "hash" | "id" => Ty::Int,
            "next" => Ty::Object,
            "sorted" | "reversed" | "list" => match arg(0) {
                // An array's elements make a list of what they read as.
                Ty::List(Elem::Array(c)) => Ty::List(Elem::of(c.item())),
                Ty::List(e) => Ty::List(e),
                Ty::Str => Ty::List(Elem::Str),
                Ty::Bytes => Ty::List(Elem::Int),
                Ty::Tuple(_) => Ty::List(Elem::of(arg(0).element().unwrap_or(Ty::Object))),
                // A list of a dict is its keys.
                Ty::Dict(k) => Ty::List(Elem::of(dict_shape(k).0)),
                Ty::Set | Ty::Gen => Ty::List(Elem::Object),
                _ => match args.first() {
                    Some(py::Expr::Call(c)) if is_name(&c.func, "range") => Ty::List(Elem::Int),
                    _ => Ty::List(Elem::Object),
                },
            },
            // A tuple built from a sequence has the length of that
            // sequence, which is not a shape.
            "tuple" => match arg(0) {
                Ty::Tuple(k) => Ty::Tuple(k),
                _ => Ty::Object,
            },
            "dict" => match arg(0) {
                Ty::Dict(k) => Ty::Dict(k),
                Ty::Unknown => dict_of(Ty::Unknown, Ty::Unknown),
                _ => dynamic_dict(),
            },
            "set" | "frozenset" => Ty::Set,
            // Pairs and mapped values are dynamic; the lists are eager.
            "enumerate" | "zip" | "map" | "filter" => Ty::List(Elem::Object),
            "any" | "all" => Ty::Bool,
            "sum" => {
                let items = match arg(0) {
                    Ty::List(Elem::Int) | Ty::Bytes => Ty::Int,
                    Ty::List(Elem::Float) => Ty::Float,
                    t @ Ty::Tuple(_) => match t.element() {
                        Some(Ty::Int | Ty::Bool) => Ty::Int,
                        Some(Ty::Float) => Ty::Float,
                        _ => Ty::Object,
                    },
                    _ => Ty::Object,
                };
                match args.get(1) {
                    None => items,
                    Some(start) => match (items, self.expr(start)) {
                        (Ty::Int, Ty::Int | Ty::Bool) => Ty::Int,
                        (Ty::Int | Ty::Float, Ty::Float) | (Ty::Float, Ty::Int | Ty::Bool) => {
                            Ty::Float
                        }
                        _ => Ty::Object,
                    },
                }
            }
            "min" | "max" if args.len() == 1 => match arg(0) {
                Ty::List(e) => e.ty(),
                t @ Ty::Tuple(_) => Elem::of(t.element().unwrap_or(Ty::Object)).ty(),
                _ => Ty::Object,
            },
            "divmod" => {
                let q = arg(0).arith(arg(1));
                tuple_of(vec![q, q])
            }
            "type" => Ty::Str,
            "str" | "repr" | "input" | "chr" => Ty::Str,
            "bytes" => Ty::Bytes,
            // The mode decides what is read and written; it is a literal
            // or the file is a text one.
            "open" => Ty::File(open_mode(c).unwrap_or(Mode::Text)),
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

    /// What a method call on a value of type `receiver` returns.
    fn method_ret(&self, receiver: Ty, attr: &str) -> Ty {
        match receiver {
            Ty::Unknown | Ty::None => Ty::Unknown,
            Ty::List(e) => match attr {
                "pop" => e.ty(),
                "index" | "count" => Ty::Int,
                "copy" => Ty::List(e),
                // An array's elements as the list of what they read as,
                // or as the bytes they are stored as.
                "tolist" => Ty::List(Elem::of(e.ty())),
                "tobytes" if e.code().is_some() => Ty::Bytes,
                _ => Ty::None,
            },
            Ty::Tuple(_) => match attr {
                "index" | "count" => Ty::Int,
                _ => Ty::Object,
            },
            // What the defining class says, joined with every override
            // an instance of a subclass would reach.
            Ty::Class(k) => self
                .module
                .dispatched_ret(k as usize, attr)
                .unwrap_or(Ty::Object),
            Ty::Dict(k) => {
                let (key, value) = dict_shape(k);
                match attr {
                    "keys" => Ty::List(Elem::of(key)),
                    "values" => Ty::List(Elem::of(value)),
                    "items" => Ty::List(Elem::of(tuple_of(vec![key, value]))),
                    "pop" | "setdefault" => value,
                    "copy" => receiver,
                    "clear" | "update" => Ty::None,
                    _ => Ty::Object,
                }
            }
            Ty::Set => match attr {
                "add" | "remove" | "discard" | "clear" | "update" => Ty::None,
                "union" | "intersection" | "difference" | "symmetric_difference" | "copy" => {
                    Ty::Set
                }
                "issubset" | "issuperset" | "isdisjoint" => Ty::Bool,
                _ => Ty::Object,
            },
            Ty::Str => match attr {
                "upper" | "lower" | "strip" | "lstrip" | "rstrip" | "replace" | "join"
                | "capitalize" | "title" | "swapcase" | "format" | "zfill" | "center" | "ljust"
                | "rjust" => Ty::Str,
                "find" | "rfind" | "index" | "rindex" | "count" => Ty::Int,
                "startswith" | "endswith" | "isdigit" | "isalpha" | "isalnum" | "isspace"
                | "isupper" | "islower" => Ty::Bool,
                "split" | "rsplit" | "splitlines" => Ty::List(Elem::Str),
                "encode" => Ty::Bytes,
                _ => Ty::Object,
            },
            Ty::Bytes => match attr {
                "decode" => Ty::Str,
                _ => Ty::Object,
            },
            Ty::File(mode) => match attr {
                "read" | "readline" => mode.content(),
                "readlines" => Ty::List(Elem::of(mode.content())),
                "write" | "close" | "flush" => Ty::None,
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
