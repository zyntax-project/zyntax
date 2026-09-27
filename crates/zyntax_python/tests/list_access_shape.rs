//! A typed list's subscripts and loops lower to direct slot reads and
//! stores behind the frontend's own index check: no call to the list
//! library's checked get, set or index normalisation.

use zyntax_typed_ast::{InternedString, TypedDeclaration};

const MATMULT: &str = r#"
def matmult(y: list[float], val: list[float], row: list[int], col: list[int], x: list[float], n: int) -> None:
    M = len(row) - 1
    for reps in range(n):
        for r in range(M):
            s = 0.0
            for i in range(row[r], row[r + 1]):
                s += x[col[i]] * val[i]
            y[r] = s


def scale(xs: list[float], k: float) -> float:
    t = 0.0
    for x in xs:
        t += x
    for i in range(len(xs)):
        xs[i] *= k
    return t


y = [0.0] * 3
matmult(y, [1.0, 2.0, 3.0], [0, 1, 2, 3], [0, 1, 2], [1.0, 1.0, 1.0], 2)
print(y, scale(y, 2.0))
"#;

#[test]
fn typed_list_access_calls_no_checked_library_function() {
    let program = zyntax_python::parse_program(MATMULT).expect("lowers");
    let library = ["get", "set", "norm"]
        .iter()
        .flat_map(|op| {
            ["i64", "f64"]
                .iter()
                .map(move |k| format!("zb_list_{op}_{k}"))
        })
        .map(|name| (format!("{:?}", InternedString::new_global(&name)), name))
        .collect::<Vec<_>>();
    let mut seen = 0;
    for d in &program.declarations {
        let TypedDeclaration::Function(f) = &d.node else {
            continue;
        };
        let name = f.name.resolve_global().unwrap_or_default();
        if !matches!(name.as_str(), "matmult" | "scale") {
            continue;
        }
        seen += 1;
        let body = format!("{:?}", f.body);
        for (symbol, callee) in &library {
            assert!(!body.contains(symbol.as_str()), "{name} calls {callee}");
        }
    }
    assert_eq!(seen, 2, "both functions are lowered");
}
