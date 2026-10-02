//! A host calls a function value a Python program made, as the library's
//! own call through a value does.

use zyntax_embed::foreign::{self, Any, Value};
use zyntax_embed::{TieredConfig, TieredRuntime};

fn load(source: &str) -> TieredRuntime {
    let program = zyntax_python::parse_module_with_host(source, "<module>", &|_| None, &|_| None)
        .expect("should lower");
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    zyntax_python::register_runtime(&mut rt).expect("runtime plugins");
    rt.compile_typed_program(program).expect("should compile");
    rt.call_raw(zyntax_python::ENTRY, &[])
        .expect("the body runs");
    rt
}

const SOURCE: &str = r#"
def make_adder(k: float):
    return lambda x: x + k

def make_scale():
    def scale(a, b):
        return a * b
    return scale
"#;

// One runtime for every case: a function reached by its pointer compiles
// on its first call through the process's lazy compiler, which is one
// runtime's at a time.
#[test]
fn a_host_calls_function_values_a_program_made() {
    let rt = load(SOURCE);

    // A lambda, with what it captured.
    let make = rt.get_function_ptr("make_adder").expect("compiled");
    let make: extern "C" fn(f64) -> Any = unsafe { std::mem::transmute(make) };
    let add = make(1.5);
    unsafe {
        assert!(foreign::is_function(add));
        assert_eq!(foreign::function_arity(add), Some((1, 1)));
        let result = foreign::call_function(add, &[foreign::float(2.0)]).expect("called");
        assert!(matches!(foreign::read(result), Value::Float(f) if f == 3.5));
    }

    // A nested function of two arguments.
    let make = rt.get_function_ptr("make_scale").expect("compiled");
    let make: extern "C" fn() -> Any = unsafe { std::mem::transmute(make) };
    let scale = make();
    unsafe {
        let result =
            foreign::call_function(scale, &[foreign::int(6), foreign::int(7)]).expect("called");
        assert!(matches!(foreign::read(result), Value::Int(42)));
    }

    // The wrong number of arguments, and a value that is no function.
    let error = unsafe { foreign::call_function(add, &[]) }.expect_err("one argument is missing");
    assert_eq!(error.kind, "TypeError");
    let error = unsafe { foreign::call_function(foreign::int(3), &[]) }
        .expect_err("an int is not a function");
    assert_eq!(error.kind, "TypeError");
}
