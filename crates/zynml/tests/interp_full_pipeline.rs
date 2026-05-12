//! Full-pipeline integration test through the single `ZyntaxRuntime`:
//! parses + lowers the struct/impl pipeline AND executes through the
//! BC interpreter. The interp is `ZyntaxRuntime`'s execution engine —
//! one runtime, not two.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

fn compile_and_install(source: &str) -> ZyntaxRuntime {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<test>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.compile_typed_program(program).expect("compile");
    rt
}

#[test]
fn full_pipeline_runs_simple_struct_method() {
    let rt = compile_and_install(
        r#"
        struct Point {
            x: i64,
            y: i64
        }

        impl Point {
            def sum(self): i64 {
                return self.x + self.y
            }
        }

        def make_and_sum(): i64 {
            let p = Point { x: 10, y: 32 }
            return p.sum()
        }
        "#,
    );

    let r = rt
        .call_function_raw("make_and_sum", vec![])
        .expect("call should succeed");
    assert_eq!(r.as_i64(), Some(42), "Point{{10, 32}}.sum() = {:?}", r);
}

#[test]
fn full_pipeline_runs_multi_struct_with_method_chain() {
    let rt = compile_and_install(
        r#"
        struct Vec2 {
            x: i64,
            y: i64
        }

        struct Triangle {
            a: Vec2,
            b: Vec2,
            c: Vec2
        }

        def vertex_sum(t: Triangle): i64 {
            return t.a.x + t.a.y + t.b.x + t.b.y + t.c.x + t.c.y
        }

        def main(): i64 {
            let t = Triangle {
                a: Vec2 { x: 1, y: 2 },
                b: Vec2 { x: 3, y: 4 },
                c: Vec2 { x: 5, y: 6 }
            }
            return vertex_sum(t)
        }
        "#,
    );

    let r = rt
        .call_function_raw("main", vec![])
        .expect("call should succeed");
    assert_eq!(r.as_i64(), Some(21), "1+2+3+4+5+6 = {:?}", r);
}
