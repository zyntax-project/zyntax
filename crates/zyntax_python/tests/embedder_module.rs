//! A module an embedder imports hands back an exception nothing in it
//! caught: the entry returns with the exception pending, for the host to
//! take, where a program reports it and ends the process.

use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};

fn load(source: &str) -> TieredRuntime {
    let program = zyntax_python::parse_module_with_host(source, "<module>", &|_| None, &|_| None)
        .expect("should lower");
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    zyntax_python::register_runtime(&mut rt).expect("runtime plugins");
    rt.compile_typed_program(program).expect("should compile");
    rt
}

#[test]
fn a_module_body_that_raises_leaves_its_exception_for_the_host() {
    let rt = load(
        r#"
count = 1
raise ValueError("the module cannot load")
"#,
    );
    rt.call_raw(zyntax_python::ENTRY, &[])
        .expect("the entry returns");
    assert!(rt.take_pending_error().is_some());
    assert_eq!(rt.take_pending_error(), None);
}

#[test]
fn a_module_body_that_finishes_leaves_nothing_pending() {
    let rt = load(
        r#"
count = 1
"#,
    );
    rt.call_raw(zyntax_python::ENTRY, &[])
        .expect("the entry returns");
    assert_eq!(rt.take_pending_error(), None);
}

#[test]
fn a_function_the_host_calls_leaves_its_exception_for_the_host() {
    let rt = load(
        r#"
def checked(x: int) -> int:
    if x > 1:
        raise ValueError("too big")
    return x
"#,
    );
    rt.call_raw(zyntax_python::ENTRY, &[])
        .expect("the entry returns");
    rt.call_raw("checked", &[ZyntaxValue::Int(5)])
        .expect("the call returns");
    assert!(rt.take_pending_error().is_some());
    assert_eq!(
        rt.call_raw("checked", &[ZyntaxValue::Int(1)])
            .expect("the call returns"),
        ZyntaxValue::Int(1)
    );
    assert_eq!(rt.take_pending_error(), None);
}
