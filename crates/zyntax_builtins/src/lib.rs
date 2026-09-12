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
mod dynamic;
mod format;
pub mod functions;
mod io;
mod lists;
mod strings;

use zyntax_typed_ast::typed_builder::TypedASTBuilder;
use zyntax_typed_ast::{TypeId, TypeRegistry};

pub use build::Decl;

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
    pub function: &'static str,
    pub object: &'static str,
}

/// The element kinds a list is instantiated for. Anything else in a
/// list is a dynamic value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Int,
    Float,
    Str,
    Any,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Int, Kind::Float, Kind::Str, Kind::Any];

    /// The suffix on this kind's functions: `zb_list_get_i64`.
    pub fn suffix(self) -> &'static str {
        match self {
            Kind::Int => "i64",
            Kind::Float => "f64",
            Kind::Str => "str",
            Kind::Any => "any",
        }
    }

    pub fn ty(self) -> zyntax_typed_ast::Type {
        match self {
            Kind::Int => build::i64(),
            Kind::Float => build::f64(),
            Kind::Str => build::string(),
            Kind::Any => build::any(),
        }
    }

    /// The box tag of a list of this kind: the custom category in the
    /// low byte, the kind above it.
    pub fn list_tag(self) -> i64 {
        let index = Kind::ALL.iter().position(|k| *k == self).unwrap() as i64;
        ((index + 1) << 8) | 255
    }
}

/// The box tag of a tuple: a list of dynamic values that prints and
/// compares as a tuple.
pub const TUPLE_TAG: i64 = (5 << 8) | 255;
/// The box tag of a function value: a record of dynamic values, see
/// [`functions`].
pub const FUNC_TAG: i64 = (8 << 8) | 255;
/// The box tag of a bare code address inside a function record.
pub const CODE_TAG: i64 = (9 << 8) | 255;

/// What the library contributes to a program.
pub struct Library {
    pub declarations: Vec<Decl>,
    /// The registry the list type is declared in; a frontend adopts it.
    pub type_registry: TypeRegistry,
    /// `List<T>`, for spelling list types as the library does.
    pub list_type: TypeId,
}

/// The library for one language's spellings.
pub fn library(policy: &Policy) -> Library {
    let mut b = TypedASTBuilder::new();
    let list_type = lists::declare_list_type(&mut b);
    let mut declarations = Vec::new();
    declarations.extend(io::declarations());
    declarations.extend(strings::declarations(policy));
    declarations.extend(format::declarations());
    declarations.extend(dynamic::declarations(policy, list_type));
    declarations.extend(lists::declarations(policy, list_type));
    declarations.extend(functions::declarations(list_type));
    Library {
        declarations,
        type_registry: b.registry,
        list_type,
    }
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
            type_names: TypeNames {
                none: "NoneType",
                bool: "bool",
                int: "int",
                float: "float",
                str: "str",
                list: "list",
                tuple: "tuple",
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
