use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::hir_dump::dump_module;
use zyntax_embed::ZyntaxRuntime;

#[ignore = "kernel probe — slow; run with cargo test -- --ignored"]
#[test]
#[ignore = "diagnostic only"]
fn dump_slice8_hir() {
    let src = r#"
struct Body { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64 }

def main(): i64 {
    let a = Body { x: 1.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0 }
    let bodies = [a]
    let mut i: i64 = 0
    while i < 3 {
        let mut b = bodies[0]
        b.x = b.x + 1.0
        bodies[0] = b
        i = i + 1
    }
    return 1
}
"#;
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(src, "<dump>").expect("parse");
    let rt = ZyntaxRuntime::new().expect("rt");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt.lower_typed_program(program, builtins).expect("lower");
    eprintln!("{}", dump_module(&module));
}
