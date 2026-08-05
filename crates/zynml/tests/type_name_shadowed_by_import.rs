//! A type declared here keeps its name when a module it imports only ever
//! referenced that name.
//!
//! The prelude's `FilterIterator<I, P, T>` mentions `P` without declaring
//! it, so the prelude's registry holds `P` as a placeholder with no fields.
//! Merging that registry must not let the placeholder take the name from a
//! struct declared in the importing module, or every field access on it
//! resolves against something fieldless.

use zynml::ZynML;

/// `P` collides with a prelude generic parameter; `Particle` does not.
///
/// Returns the functions that compiled: a field access resolved against a
/// fieldless placeholder fails analysis, and the function carrying it is
/// dropped rather than reported as an error.
fn compiled_functions(type_name: &str) -> Vec<String> {
    let source = format!(
        r#"
import prelude

struct {type_name} {{
    x: f64,
    y: f64
}}

def total(): i64 {{
    let a = {type_name} {{ x: 20.0, y: 1.0 }}
    let b = {type_name} {{ x: 0.0, y: 0.0 }}
    let mut ps = [a, b]
    let mut s = ps[0]
    s.y = s.y + 21.0
    ps[0] = s
    let q = ps[0]
    return (q.x + q.y) as i64
}}
"#
    );

    let mut zynml = ZynML::new().expect("runtime should start");
    zynml
        .load_source(&source)
        .expect("source should load and compile")
}

#[test]
fn a_type_named_like_a_prelude_generic_param_keeps_its_fields() {
    let compiled = compiled_functions("P");
    assert!(
        compiled.iter().any(|f| f == "total"),
        "`P` is a generic parameter in the prelude, but this module declares it; \
         compiled: {compiled:?}"
    );
}

#[test]
fn a_type_whose_name_the_prelude_never_mentions_still_works() {
    let compiled = compiled_functions("Particle");
    assert!(compiled.iter().any(|f| f == "total"), "{compiled:?}");
}
