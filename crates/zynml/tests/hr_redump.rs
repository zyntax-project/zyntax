use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

#[test]
fn redump_first_step() {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let src = r#"
effect Event { def next_event(): i64 }
handler Feed for Event { def next_event(): i64 { return 3 } }

@effect(Event)
fiber def machine(): i64 {
    yield next_event()
}

def first_step(): i64 {
    let mut out: i64 = 0
    with Feed {
        let f = machine()
        match f.next() {
            case Some(v) { out = v }
            case None() { }
        }
    }
    return out
}
"#;
    let program = grammar.parse_with_filename(src, "<p>").expect("parse");
    let rt = ZyntaxRuntime::new().expect("rt");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let m = rt.lower_typed_program(program, builtins).expect("lower");
    for func in m.functions.values() {
        if func.name.resolve_global().as_deref() == Some("first_step") {
            eprintln!("{}", zyntax_compiler::hir_dump::dump_function(func, &m));
        }
    }
}
