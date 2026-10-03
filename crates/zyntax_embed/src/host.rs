//! The typed interface an embedding host gives its modules, classes and
//! functions, which a frontend checks a program's uses against.

/// A type in an embedding host's description of what it exposes to a
/// program. Values travel through the foreign-object protocol
/// ([`crate::foreign`]); the description supplies the static signatures a
/// frontend types and lowers calls against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostType {
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostMethod {
    pub name: String,
    /// An embedder-defined member key. Zero keeps name-based dispatch.
    pub key: u64,
    pub params: Vec<HostType>,
    pub ret: HostType,
    pub is_static: bool,
}

/// A field exposed by an embedding host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostField {
    pub name: String,
    /// An embedder-defined member key. Zero keeps name-based dispatch.
    pub key: u64,
    pub ty: HostType,
    pub is_static: bool,
}

/// A class exposed by an embedding host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostClass {
    pub name: String,
    pub type_name: String,
    pub fields: Vec<HostField>,
    pub methods: Vec<HostMethod>,
    pub constructor: Option<HostMethod>,
}

/// A module exposed by an embedding host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostModule {
    pub name: String,
    pub classes: Vec<HostClass>,
    pub functions: Vec<HostMethod>,
}

/// Finds the typed interface of a module owned by the embedding host.
pub type HostResolver<'a> = dyn Fn(&str) -> Option<HostModule> + 'a;
