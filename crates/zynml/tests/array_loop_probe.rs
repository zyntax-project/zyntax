use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

fn run(src: &str) -> String {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(src, "<probe>").expect("parse");
    let rt = ZyntaxRuntime::new().expect("rt");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt.lower_typed_program(program, builtins).expect("lower");
    let mut rt = ZyntaxRuntime::new().expect("rt");
    rt.compile_module(&module).expect("compile");
    let r = rt.call_function_raw("main", vec![]).expect("call main");
    format!("{:?}", r)
}

#[ignore = "kernel probe — slow; run with cargo test -- --ignored"]
#[test]
fn array_sum_loop() {
    let src = r#"
def main(): i64 {
    let xs = [10, 20, 30, 40, 50]
    let mut total: i64 = 0
    let mut i: i64 = 0
    while i < 5 {
        total = total + xs[i]
        i = i + 1
    }
    return total
}
"#;
    let r = run(src);
    eprintln!("array_sum_loop = {r}");
    assert_eq!(r, "Int(150)");
}

// `xs[i] = value` is rejected at SSA lowering today — the typed-AST
// classifies `xs` as `Type::Named(List<T>)` rather than `Type::Array`,
// so the assign path at
// `crates/compiler/src/ssa.rs::TypedExpression::Index` errors with
// "Cannot index into non-array type" and the whole function is
// silently dropped (call site sees "unknown function 'main'"). Kept
// as `#[ignore]` documentation; when the lowering is fixed this can
// flip to `#[test]` and pin the right answer.
#[ignore = "kernel probe — slow; run with cargo test -- --ignored"]
#[test]
#[ignore = "index-assign on List<T> not yet supported by SSA lowering"]
fn array_index_assign() {
    let src = r#"
def main(): i64 {
    let mut xs = [10, 20, 30]
    xs[1] = 99
    return xs[1]
}
"#;
    let r = run(src);
    eprintln!("array_index_assign = {r}");
    assert_eq!(r, "Int(99)");
}
