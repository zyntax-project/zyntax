//! A program the host enters only through its module body builds what
//! that body reaches, of its own functions and of what the frontend
//! generated for it; one the host may call into by name builds all of
//! its own.

use zyntax_embed::{TieredConfig, TieredRuntime};
use zyntax_typed_ast::TypedDeclaration;

fn compiled(source: &str, closed: bool) -> Vec<String> {
    let program = zyntax_python::parse_program(source).expect("lowers");
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    zyntax_python::register_runtime(&mut rt).expect("runtime plugins");
    if closed {
        rt.enter_only_through_entry_points();
    }
    rt.compile_typed_program(program).expect("compiles");
    rt.functions().into_iter().map(str::to_string).collect()
}

const PIPELINE: &str = r#"
def main() -> int:
    n = 1500
    xs = [(i * 2654435761) % 1000003 for i in range(n)]
    evens = [x for x in xs if x % 2 == 0]
    squares = sum(x * x % 1000 for x in evens)
    pairs = [(x % 1000, x) for x in evens[:200]]
    pairs.sort(key=lambda p: p[0])
    acc = squares
    for k, v in pairs[:10]:
        acc += k * 31 + v
    return acc % 1000000007

print(main())
"#;

#[test]
fn pipeline_compile_reaches_few_shape_functions() {
    let of_shape = |name: &str| name.ends_with("_t0");
    let program = zyntax_python::parse_program(PIPELINE).expect("lowers");
    let declared = program
        .declarations
        .iter()
        .filter(|d| match &d.node {
            TypedDeclaration::Function(f) => f.name.resolve_global().is_some_and(|n| of_shape(&n)),
            _ => false,
        })
        .count();
    let built = compiled(PIPELINE, true)
        .into_iter()
        .filter(|n| of_shape(n))
        .count();
    assert!(
        declared > 40,
        "the shape's functions are declared: {declared}"
    );
    assert!(
        built > 0 && built * 3 < declared,
        "{built} of the shape's {declared} functions built"
    );
}

const UNREACHED: &str = r#"
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def norm(self) -> int:
        return self.x * self.x + self.y * self.y

    def never(self) -> int:
        return self.x - self.y

def used(n: int) -> int:
    return Point(n, n + 1).norm()

def unused(n: int) -> int:
    return n * 7

print(used(3))
"#;

#[test]
fn a_closed_program_builds_only_what_its_body_reaches() {
    let closed = compiled(UNREACHED, true);
    assert!(closed.iter().any(|n| n == "used"), "{closed:?}");
    assert!(!closed.iter().any(|n| n == "unused"), "{closed:?}");
    assert!(!closed.iter().any(|n| n.ends_with("never")), "{closed:?}");

    let open = compiled(UNREACHED, false);
    assert!(open.iter().any(|n| n == "unused"), "{open:?}");
}
