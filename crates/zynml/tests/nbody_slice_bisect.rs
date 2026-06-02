//! Bisect harness — try progressively richer versions of the
//! n-body kernel and report which slice is the one that breaks
//! the SSA lowering / BC interp / Cranelift backend. Output goes
//! to `eprintln`; each slice runs independently so a panic in one
//! doesn't stop the next.
//!
//! Run with: `cargo test --release --test nbody_slice_bisect -- --nocapture --test-threads=1`

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

fn try_run(label: &str, src: &str) {
    let grammar = match Grammar2::from_source(ZYNML_GRAMMAR) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("[{label}] grammar FAIL: {e:?}");
            return;
        }
    };
    let program = match grammar.parse_with_filename(src, "<bisect>") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[{label}] parse FAIL: {e:?}");
            return;
        }
    };
    let rt = match ZyntaxRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[{label}] rt FAIL: {e:?}");
            return;
        }
    };
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = match rt.lower_typed_program(program, builtins) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[{label}] lower FAIL: {e:?}");
            return;
        }
    };
    let func_names: Vec<String> = module
        .functions
        .values()
        .filter_map(|f| f.name.resolve_global())
        .collect();
    eprintln!("[{label}] module functions: {func_names:?}");
    let mut rt = ZyntaxRuntime::new().unwrap();
    let compile_outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.compile_module(&module)));
    match compile_outcome {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("[{label}] compile_module FAIL: {e:?}");
            return;
        }
        Err(p) => {
            let msg = if let Some(s) = p.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = p.downcast_ref::<String>() {
                s.clone()
            } else {
                "<opaque>".to_string()
            };
            eprintln!("[{label}] compile_module PANIC: {msg}");
            return;
        }
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rt.call_function_raw("main", vec![])
    }));
    match result {
        Ok(Ok(r)) => eprintln!("[{label}] OK = {r:?}"),
        Ok(Err(e)) => eprintln!("[{label}] call FAIL: {e:?}"),
        Err(p) => {
            let msg = if let Some(s) = p.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = p.downcast_ref::<String>() {
                s.clone()
            } else {
                "<opaque>".to_string()
            };
            eprintln!("[{label}] PANIC: {msg}");
        }
    }
}

#[test]
fn nbody_slices() {
    // ── Slice 1: just a Body struct, no array. Construct and
    //    return a field.
    try_run(
        "1-struct-only",
        r#"
struct Body { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64 }

def main(): i64 {
    let b = Body {
        x: 1.0, y: 2.0, z: 3.0,
        vx: 0.0, vy: 0.0, vz: 0.0,
        mass: 39.5
    }
    if b.x > 0.5 { return 1 }
    return 0
}
"#,
    );

    // ── Slice 2: array of Body literal, return length.
    try_run(
        "2-array-of-struct",
        r#"
struct Body { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64 }

def main(): i64 {
    let a = Body { x: 1.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0 }
    let b = Body { x: 2.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 2.0 }
    let bodies = [a, b]
    return 2
}
"#,
    );

    // ── Slice 3: read bodies[0].x — array read + field access on
    //    the returned struct value.
    try_run(
        "3-array-read-field",
        r#"
struct Body { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64 }

def main(): i64 {
    let a = Body { x: 1.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0 }
    let bodies = [a]
    let b0 = bodies[0]
    if b0.x > 0.5 { return 1 }
    return 0
}
"#,
    );

    // ── Slice 4: bodies[0] = b — pure index-assign.
    try_run(
        "4-index-assign",
        r#"
struct Body { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64 }

def main(): i64 {
    let a = Body { x: 1.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0 }
    let bodies = [a]
    let mut b = bodies[0]
    b.x = 99.0
    bodies[0] = b
    return 1
}
"#,
    );

    // ── Slice 5: same as 4 but bodies is `let mut`.
    try_run(
        "5-index-assign-mutable-binding",
        r#"
struct Body { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64 }

def main(): i64 {
    let a = Body { x: 1.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0 }
    let mut bodies = [a]
    let mut b = bodies[0]
    b.x = 99.0
    bodies[0] = b
    return 1
}
"#,
    );

    // ── Slice 6: function taking Array<Body>, no index-assign.
    try_run(
        "6-fn-takes-array",
        r#"
struct Body { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64 }

def first_x(bodies: Array<Body>): f64 {
    let b = bodies[0]
    return b.x
}

def main(): i64 {
    let a = Body { x: 7.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0 }
    let bodies = [a]
    let r = first_x(bodies)
    if r > 5.0 { return 1 }
    return 0
}
"#,
    );

    // ── Slice 7: function taking Array<Body> with index-assign.
    try_run(
        "7-fn-index-assign",
        r#"
struct Body { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64 }

def bump(bodies: Array<Body>): i64 {
    let mut b = bodies[0]
    b.x = 99.0
    bodies[0] = b
    return 1
}

def main(): i64 {
    let a = Body { x: 1.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0 }
    let bodies = [a]
    return bump(bodies)
}
"#,
    );
}
