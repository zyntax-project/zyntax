//! The built-in library the frontends share, as typed AST.
//!
//! Lists, strings, dynamic values and printing are the same in every
//! language on this compiler; what differs is spelling: whether a true
//! value prints as `True` or `true`, what the absent value is called,
//! how a string is quoted. So the mechanics are built here once as typed
//! AST (see [`build`]), and a frontend hands in a [`Policy`] for the
//! spellings and merges the result into its program. Everything then
//! compiles and inlines with the program; nothing is called across a
//! runtime boundary except the string plugin's primitives, which are the
//! same primitives every frontend's strings rest on.

pub mod build;
mod dicts;
mod dynamic;
mod format;
pub mod functions;
mod io;
mod iteration;
pub mod lists;
mod math;
mod random;
mod strings;

use zyntax_typed_ast::typed_builder::TypedASTBuilder;
use zyntax_typed_ast::{TypeId, TypeRegistry};

pub use build::{Decl, MODULE};

/// How a language spells the values the library prints.
#[derive(Clone, Debug)]
pub struct Policy {
    pub true_text: &'static str,
    pub false_text: &'static str,
    pub none_text: &'static str,
    /// Whether a string's repr prefers single quotes (Python) or double
    /// quotes (most others).
    pub single_quotes: bool,
    /// Whether a whole float prints with a fraction (`3.0`) or bare (`3`).
    pub float_fraction: bool,
    /// What `type(x)` calls each kind of dynamic value.
    pub type_names: TypeNames,
    /// Whether the frontend defines `zb_hook_instance_str`,
    /// `zb_hook_instance_type` and `zb_hook_instance_eq` for boxed
    /// instances of its own classes (kinds from [`INSTANCE_KIND_BASE`]).
    /// When it does not, the library's defaults stand in.
    pub instance_hooks: bool,
    /// Whether the frontend defines `zb_hook_raise(kind, message)` and
    /// turns a library error into an exception it can catch. When it
    /// does, every library function that can fail returns a placeholder
    /// after the hook, and the frontend checks for the pending exception
    /// after calling one; see [`Library::fallible`].
    pub exceptions: bool,
    /// Whether a boolean equals the number it stands for, as in Python
    /// (`True == 1`, one dict key), or is its own kind of value that
    /// equals a boolean only, as in Lua.
    pub bool_is_number: bool,
}

/// The names of the dynamic value kinds, as a language spells them.
#[derive(Clone, Debug)]
pub struct TypeNames {
    pub none: &'static str,
    pub bool: &'static str,
    pub int: &'static str,
    pub float: &'static str,
    pub str: &'static str,
    pub list: &'static str,
    pub tuple: &'static str,
    pub dict: &'static str,
    pub set: &'static str,
    pub function: &'static str,
    pub object: &'static str,
}

/// The element kinds a list is instantiated for. Anything else in a
/// list is a dynamic value. The kinds after `Any` are the storage of
/// typed arrays: their elements read as an `Int` or a `Float`, and are
/// stored at the width the kind names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Int,
    Float,
    Str,
    /// The address of an instance of one of the frontend's classes.
    /// Stored as a plain word; the frontend casts at either end, and
    /// boxes one through `zb_hook_box_instance` when it becomes dynamic.
    Ptr,
    Any,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    U64,
    F32,
}

impl Kind {
    /// The kinds the library itself instantiates. The array storage
    /// kinds are generated into a program by the frontend that uses
    /// them, since most programs use none.
    pub const LIBRARY: [Kind; 5] = [Kind::Int, Kind::Float, Kind::Str, Kind::Ptr, Kind::Any];

    /// Every kind, in the order their tags are numbered.
    pub const ALL: [Kind; 13] = [
        Kind::Int,
        Kind::Float,
        Kind::Str,
        Kind::Ptr,
        Kind::Any,
        Kind::I8,
        Kind::U8,
        Kind::I16,
        Kind::U16,
        Kind::I32,
        Kind::U32,
        Kind::U64,
        Kind::F32,
    ];

    /// The suffix on this kind's functions: `zb_list_get_i64`.
    pub fn suffix(self) -> &'static str {
        match self {
            Kind::Int => "i64",
            Kind::Float => "f64",
            Kind::Str => "str",
            Kind::Ptr => "ptr",
            Kind::Any => "any",
            Kind::I8 => "i8",
            Kind::U8 => "u8",
            Kind::I16 => "i16",
            Kind::U16 => "u16",
            Kind::I32 => "i32",
            Kind::U32 => "u32",
            Kind::U64 => "u64",
            Kind::F32 => "f32",
        }
    }

    pub fn ty(self) -> zyntax_typed_ast::Type {
        use zyntax_typed_ast::type_registry::PrimitiveType as P;
        match self {
            Kind::Int => build::i64(),
            Kind::Float => build::f64(),
            Kind::Str => build::string(),
            Kind::Ptr => build::usize(),
            Kind::Any => build::any(),
            Kind::I8 => zyntax_typed_ast::Type::Primitive(P::I8),
            Kind::U8 => zyntax_typed_ast::Type::Primitive(P::U8),
            Kind::I16 => zyntax_typed_ast::Type::Primitive(P::I16),
            Kind::U16 => zyntax_typed_ast::Type::Primitive(P::U16),
            Kind::I32 => zyntax_typed_ast::Type::Primitive(P::I32),
            Kind::U32 => zyntax_typed_ast::Type::Primitive(P::U32),
            Kind::U64 => zyntax_typed_ast::Type::Primitive(P::U64),
            Kind::F32 => zyntax_typed_ast::Type::Primitive(P::F32),
        }
    }

    /// The kinds whose elements are integers stored narrower than, or
    /// unsigned at, the word an `Int` is.
    pub fn is_narrow_int(self) -> bool {
        matches!(
            self,
            Kind::I8 | Kind::U8 | Kind::I16 | Kind::U16 | Kind::I32 | Kind::U32 | Kind::U64
        )
    }

    /// The kind an element of this kind reads as: `Int` or `Float` for
    /// the array storage kinds, the kind itself otherwise.
    pub fn wide(self) -> Kind {
        match self {
            Kind::F32 => Kind::Float,
            k if k.is_narrow_int() => Kind::Int,
            k => k,
        }
    }

    /// The box tag of a list of this kind: the custom category in the
    /// low byte, the kind above it.
    pub fn list_tag(self) -> i64 {
        let index = Kind::ALL.iter().position(|k| *k == self).unwrap() as i64;
        ((index + 1) << 8) | 255
    }
}

/// The kinds above the list kinds, in the order they are numbered.
const LIST_KINDS: i64 = Kind::ALL.len() as i64;
/// The box tag of a tuple: a list of dynamic values that prints and
/// compares as a tuple.
pub const TUPLE_TAG: i64 = ((LIST_KINDS + 1) << 8) | 255;
/// The box tag of a dict: keys and values alternating in one list.
pub const DICT_TAG: i64 = ((LIST_KINDS + 2) << 8) | 255;
/// The box tag of a set: a list of distinct values.
pub const SET_TAG: i64 = ((LIST_KINDS + 3) << 8) | 255;
/// The box tag of a function value: a record of dynamic values, see
/// [`functions`].
pub const FUNC_TAG: i64 = ((LIST_KINDS + 4) << 8) | 255;
/// The box tag of a bare code address inside a function record.
pub const CODE_TAG: i64 = ((LIST_KINDS + 5) << 8) | 255;
/// Kinds from here up are instances of a frontend's classes, in the
/// order the frontend numbers them.
pub const INSTANCE_KIND_BASE: i64 = 32;
/// Typed arrays are kinds a frontend registers, numbered from here
/// within the frontend's range (`lists::SHAPE_KIND_BASE` and up), so the
/// dynamic layer reaches them through the same hooks as a list of a
/// registered tuple shape. A kind holds the storage's list kind number
/// in its low byte and the typecode letter above; see [`array_tag`].
pub const ARRAY_KIND_BASE: i64 = lists::SHAPE_KIND_BASE + (1 << 19);

/// The box tag of an array stored as a list of `storage` under
/// typecode `letter`, distinct from a list stored the same way and from
/// an array of another typecode.
pub fn array_tag(storage: Kind, letter: u8) -> i64 {
    let kind = ARRAY_KIND_BASE + ((letter as i64) << 8) + (storage.list_tag() >> 8);
    (kind << 8) | 255
}
/// The box category of None, the low byte of its tag.
pub const NONE_CATEGORY: i64 = dynamic::NONE;

/// The box tag of an instance of the frontend's class `index`.
pub fn instance_tag(index: usize) -> i64 {
    ((INSTANCE_KIND_BASE + index as i64) << 8) | 255
}

/// What the library contributes to a program.
pub struct Library {
    pub declarations: Vec<Decl>,
    /// The registry the list type is declared in; a frontend adopts it.
    pub type_registry: TypeRegistry,
    /// `List<T>`, for spelling list types as the library does.
    pub list_type: TypeId,
    /// The functions that can report an error: those calling `zb_fatal`,
    /// and those calling one of them, and so on.
    pub fallible: std::collections::BTreeSet<String>,
}

/// The library for one language's spellings.
pub fn library(policy: &Policy) -> Library {
    let mut b = TypedASTBuilder::new();
    let list_type = lists::declare_list_type(&mut b);
    let mut declarations = Vec::new();
    declarations.extend(io::declarations(policy, list_type));
    declarations.extend(strings::declarations(policy, list_type));
    declarations.extend(format::declarations());
    declarations.extend(dynamic::declarations(policy, list_type));
    declarations.extend(lists::declarations(policy, list_type));
    declarations.extend(functions::declarations(list_type));
    declarations.extend(dicts::declarations(list_type));
    declarations.extend(iteration::declarations(list_type));
    declarations.extend(math::declarations());
    declarations.extend(random::declarations(list_type));
    // The hooks a frontend defines are declared here as externs, so the
    // library lowers on its own; the frontend's definition takes the
    // declaration's place when the two meet in a program.
    if policy.instance_hooks {
        declarations.extend(dynamic::extern_instance_hooks());
    } else {
        declarations.extend(dynamic::default_instance_hooks(policy));
    }
    declarations.extend(lists::ptr_declarations(policy));
    declarations.extend(lists::shape_hook_declarations(policy, list_type));
    if policy.exceptions {
        declarations.push(build::extern_fn(
            "zb_hook_raise",
            &[("kind", build::string()), ("message", build::string())],
            build::unit(),
            None,
        ));
    }
    let fallible = fallible_functions(&declarations);
    Library {
        declarations,
        type_registry: b.registry,
        list_type,
        fallible,
    }
}

/// Every function that reaches `zb_fatal`, through any number of calls.
fn fallible_functions(declarations: &[Decl]) -> std::collections::BTreeSet<String> {
    use std::collections::{BTreeMap, BTreeSet};
    use zyntax_typed_ast::typed_ast::TypedDeclaration;
    let mut calls: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for d in declarations {
        if let TypedDeclaration::Function(f) = &d.node {
            if let (Some(name), Some(body)) = (f.name.resolve_global(), &f.body) {
                let mut callees = BTreeSet::new();
                for s in &body.statements {
                    build::callees_of_stmt(s, &mut callees);
                }
                calls.insert(name, callees);
            }
        }
    }
    let mut fallible: BTreeSet<String> = ["zb_fatal".to_string()].into_iter().collect();
    loop {
        let before = fallible.len();
        for (name, callees) in &calls {
            if callees.iter().any(|c| fallible.contains(c)) {
                fallible.insert(name.clone());
            }
        }
        if fallible.len() == before {
            break;
        }
    }
    fallible
}

/// `List<elem>` as the library declares it.
pub fn list_of(list_type: TypeId, elem: zyntax_typed_ast::Type) -> zyntax_typed_ast::Type {
    zyntax_typed_ast::Type::Named {
        id: list_type,
        type_args: vec![elem],
        const_args: Vec::new(),
        variance: Vec::new(),
        nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_library_builds_for_a_python_policy() {
        let lib = library(&Policy {
            true_text: "True",
            false_text: "False",
            none_text: "None",
            single_quotes: true,
            float_fraction: true,
            instance_hooks: false,
            exceptions: false,
            bool_is_number: true,
            type_names: TypeNames {
                none: "NoneType",
                bool: "bool",
                int: "int",
                float: "float",
                str: "str",
                list: "list",
                tuple: "tuple",
                dict: "dict",
                set: "set",
                function: "function",
                object: "object",
            },
        });
        assert!(lib.declarations.len() > 100);
        let names: Vec<String> = lib
            .declarations
            .iter()
            .filter_map(|d| match &d.node {
                zyntax_typed_ast::typed_ast::TypedDeclaration::Function(f) => {
                    f.name.resolve_global()
                }
                _ => None,
            })
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "a name is declared twice");
        assert!(names.iter().any(|n| n == "zb_list_get_i64"));
        assert!(names.iter().any(|n| n == "zb_float_repr"));
    }
}
