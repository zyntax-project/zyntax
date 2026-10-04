//! The typed interface an embedding host gives its modules, classes and
//! functions, which a frontend checks a program's uses against.

/// A type in an embedding host's description of what it exposes to a
/// program. Values travel through the foreign-object protocol
/// ([`crate::foreign`]); the description supplies the static signatures a
/// frontend types and lowers calls against.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum HostType {
    #[default]
    Void,
    Bool,
    Int,
    Float,
    Str,
    Bytes,
    Object(String),
    Function {
        params: Vec<HostType>,
        ret: Box<HostType>,
    },
    Dynamic,
}

/// A callable member exposed by an embedding host.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct HostMethod {
    pub name: String,
    /// An embedder-defined member key. Zero keeps name-based dispatch.
    pub key: u64,
    pub params: Vec<HostType>,
    pub ret: HostType,
    pub is_static: bool,
    /// The native function the member is, called directly in place of
    /// the foreign protocol.
    pub native: Option<NativeBinding>,
}

/// A field exposed by an embedding host.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct HostField {
    pub name: String,
    /// An embedder-defined member key. Zero keeps name-based dispatch.
    pub key: u64,
    pub ty: HostType,
    pub is_static: bool,
    /// Whether a program may store to it.
    pub writable: bool,
    /// Where the field lies in the object, read and written in place in
    /// place of the foreign protocol.
    pub native: Option<NativeField>,
}

/// A class exposed by an embedding host.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct HostClass {
    pub name: String,
    pub type_name: String,
    pub fields: Vec<HostField>,
    pub methods: Vec<HostMethod>,
    pub constructor: Option<HostMethod>,
    /// Whether a program holds the class's objects as the host's word
    /// rather than as a foreign box.
    pub word: bool,
}

/// How an object operand of a native call reaches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativePass {
    /// The object's word itself.
    Word,
    /// The word stored at this byte offset in the object.
    Indirect(u32),
}

/// An operand or result of a native call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeType {
    Void,
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    /// A word whose meaning is the native function's.
    Word,
    /// As an operand, two: the string's bytes and their length. As a
    /// result, a host string, which `$Host$text_to_string` converts.
    Str,
    /// An object of the class whose `type_name` this is: as an operand,
    /// passed as `pass` says; as a result, the object's word.
    Object {
        type_name: String,
        pass: NativePass,
    },
}

/// A host member that is a native function the program calls directly:
/// the symbol the embedder registered, and its signature. The receiver,
/// if any, is the first operand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeBinding {
    /// The name the call links against.
    pub symbol: String,
    /// The function's address, for a host that loads it after the
    /// runtime was set up: the runtime resolves `symbol` to it. 0 when
    /// the host registered `symbol` itself.
    pub address: usize,
    pub receiver: Option<NativePass>,
    pub params: Vec<NativeType>,
    pub ret: NativeType,
    /// Whether the function can leave an error pending, which the call
    /// site then checks for.
    pub may_raise: bool,
}

/// A host field at a fixed place in its object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeField {
    /// Byte offset from the object's payload.
    pub offset: u32,
    pub ty: NativeType,
    /// How the payload is reached from the object.
    pub pass: NativePass,
}

/// A module exposed by an embedding host.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct HostModule {
    pub name: String,
    pub classes: Vec<HostClass>,
    pub functions: Vec<HostMethod>,
}

/// Finds the typed interface of a module owned by the embedding host.
pub type HostResolver<'a> = dyn Fn(&str) -> Option<HostModule> + 'a;
