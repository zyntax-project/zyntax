//! The Python prelude: ZynML source compiled into every Python program.
//!
//! What Python defines above the IR (how a float prints, what `str()`
//! of a bool is, string operators) is written once in ZynML and merged
//! into the program's `TypedProgram`, so it is compiled and inlined with
//! the program rather than called across a runtime boundary. The
//! frontend refers to its functions by name.

use std::sync::OnceLock;
use zyntax_embed::Grammar2;
use zyntax_typed_ast::typed_ast::TypedDeclaration;
use zyntax_typed_ast::{InternedString, TypedNode};

const SOURCE: &str = include_str!("../prelude/python.zynml");

/// The `extern def` names the prelude uses and the runtime symbols they
/// bind to. The parser leaves an extern's link name empty; this is where
/// each one is filled in.
const EXTERNS: &[(&str, &str)] = &[
    ("zrtl_string_length", "$String$length"),
    ("zrtl_string_char_count", "$String$char_count"),
    ("zrtl_string_index_of", "$String$index_of"),
    ("zrtl_string_substring", "$String$substring"),
    ("zrtl_string_char_at", "$String$char_at"),
    ("zrtl_string_repeat", "$String$repeat"),
    ("zrtl_string_from_int", "$String$from_int"),
    ("zrtl_string_from_float", "$String$from_float"),
    ("zrtl_string_equals", "$String$equals"),
    ("zrtl_string_compare", "$String$compare"),
    ("zrtl_string_contains", "$String$contains"),
    ("zrtl_string_to_upper", "$String$to_upper"),
    ("zrtl_string_to_lower", "$String$to_lower"),
    ("zrtl_string_trim", "$String$trim"),
    ("zrtl_string_trim_start", "$String$trim_start"),
    ("zrtl_string_trim_end", "$String$trim_end"),
    ("zrtl_string_starts_with", "$String$starts_with"),
    ("zrtl_string_ends_with", "$String$ends_with"),
    ("zrtl_string_replace_all", "$String$replace_all"),
    ("zrtl_string_count", "$String$count"),
    ("zrtl_string_parse_int", "$String$parse_int"),
    ("zrtl_string_parse_float", "$String$parse_float"),
    ("zrtl_box_tag", "zyntax_box_get_tag"),
    ("zrtl_box_i64", "zyntax_box_get_i64"),
    ("zrtl_box_f64", "zyntax_box_get_f64"),
    ("zrtl_box_bool", "zyntax_box_get_bool"),
    ("zrtl_box_str", "zyntax_box_get_opaque"),
    ("zrtl_math_pow", "$Math$pow"),
    ("zrtl_math_floor", "$Math$floor"),
];

fn grammar() -> &'static Grammar2 {
    static GRAMMAR: OnceLock<Grammar2> = OnceLock::new();
    GRAMMAR.get_or_init(|| {
        Grammar2::from_source(zynml::ZYNML_GRAMMAR).expect("the ZynML grammar compiles")
    })
}

/// The prelude's declarations, ready to merge into a program.
pub(crate) fn declarations() -> Result<Vec<TypedNode<TypedDeclaration>>, String> {
    let program = grammar()
        .parse(SOURCE)
        .map_err(|e| format!("the Python prelude does not parse: {e}"))?;
    let mut declarations = program.declarations;
    for decl in &mut declarations {
        if let TypedDeclaration::Function(f) = &mut decl.node {
            if f.is_external {
                let name = f.name.resolve_global().unwrap_or_default();
                match EXTERNS.iter().find(|(n, _)| *n == name) {
                    Some((_, symbol)) => f.link_name = Some(InternedString::new_global(symbol)),
                    None => return Err(format!("prelude extern `{name}` has no symbol")),
                }
            }
        }
    }
    Ok(declarations)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_prelude_parses_and_every_extern_is_bound() {
        let decls = super::declarations().expect("prelude");
        assert!(decls.len() > 20);
    }
}
