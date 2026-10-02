//! A host describes what a module it imported raised, with the module's
//! own description: the exception's type and its text.
//!
//! One test in its own process: a function reached by its pointer compiles
//! on its first call through the process's lazy compiler, which serves one
//! runtime at a time.

use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};

fn load(source: &str) -> TieredRuntime {
    let program = zyntax_python::parse_module_with_host(source, "<module>", &|_| None, &|_| None)
        .expect("should lower");
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    zyntax_python::register_runtime(&mut rt).expect("runtime plugins");
    rt.compile_typed_program(program).expect("should compile");
    rt
}

/// The module's own description of what it raised: the type's name and
/// the exception's text.
fn describe(rt: &TieredRuntime, error: u64) -> String {
    let describe = rt
        .get_function_ptr(zyntax_python::DESCRIBE)
        .expect("an embedder's module describes");
    let describe: extern "C" fn(u64) -> *const u8 = unsafe { std::mem::transmute(describe) };
    let text = describe(error);
    let text = unsafe { zyntax_embed::ZyntaxString::from_ptr(text.cast()) }.expect("a string");
    String::from_utf8_lossy(text.as_bytes()).into_owned()
}

#[test]
fn the_host_describes_what_a_module_raised() {
    let rt = load(
        r#"
class Overdrawn(Exception):
    pass

def withdraw(x: int) -> int:
    if x > 10:
        raise Overdrawn("short by " + str(x - 10))
    if x < 0:
        raise ValueError()
    return x
"#,
    );
    rt.call_raw(zyntax_python::ENTRY, &[])
        .expect("the entry returns");
    rt.call_raw("withdraw", &[ZyntaxValue::Int(12)])
        .expect("the call returns");
    let error = rt.take_pending_error().expect("withdraw raised");
    assert_eq!(describe(&rt, error), "Overdrawn: short by 2");
    rt.call_raw("withdraw", &[ZyntaxValue::Int(-1)])
        .expect("the call returns");
    let error = rt.take_pending_error().expect("withdraw raised");
    assert_eq!(describe(&rt, error), "ValueError");
}
