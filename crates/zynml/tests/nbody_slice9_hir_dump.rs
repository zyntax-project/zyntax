use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::hir_dump::dump_module;
use zyntax_embed::ZyntaxRuntime;

#[test]
#[ignore = "diagnostic only"]
fn dump_slice9_hir() {
    let src = r#"
struct Body { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64 }

def main(): i64 {
    let a = Body { x: 0.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0 }
    let b = Body { x: 1.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0 }
    let bodies = [a, b]
    let mut t: i64 = 0
    while t < 3 {
        let mut i: i64 = 0
        while i < 2 {
            let mut x = bodies[i]
            let mut j: i64 = i + 1
            while j < 2 {
                let mut y = bodies[j]
                x.x = x.x + y.x
                y.x = y.x - 1.0
                bodies[j] = y
                j = j + 1
            }
            bodies[i] = x
            i = i + 1
        }
        t = t + 1
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
    let module = match rt.lower_typed_program(program, builtins) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("lower err: {e:?}");
            return;
        }
    };
    eprintln!("{}", dump_module(&module));
}
