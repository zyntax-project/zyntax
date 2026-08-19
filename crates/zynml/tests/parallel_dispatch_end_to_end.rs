//! A loop spread across cores computes what it computed on one.
//!
//! The analysis decides what may be spread and the runtime hands out
//! the bands; this covers the piece between them, which moves a loop
//! into a function of its own and leaves a call behind. What matters is
//! that the answer does not change, so these run the same kernel both
//! ways and compare, rather than inspecting the shape it was rewritten
//! into.
//!
//! The switch is an environment variable, which is process-wide, so the
//! two halves of a comparison cannot run at once.

use std::sync::mpsc;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::Duration;
use zynml::{Grammar2, ZynML, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::profiling::ProfileConfig;
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};

fn serialised() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

const PRE: &str = "import prelude\nimport simd\n";

/// Compile and run `main` natively, with the dispatch on or off.
///
/// The bytecode interpreter is not a target for this: a dispatch hands
/// a function pointer to the runtime and an interpreter has no address
/// to hand over, which is why the pass runs only where a backend is
/// going to compile the module. So the JIT is installed rather than
/// left to warm up.
fn answer(body: &str, spread: bool) -> ZyntaxValue {
    let src = format!("{PRE}{body}");
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        std::env::set_var("ZYNTAX_PARALLEL_LOOPS", if spread { "1" } else { "0" });
        let got = (|| -> Result<ZyntaxValue, String> {
            let mut zynml = ZynML::new().map_err(|e| format!("{e:?}"))?;
            let mut cfg = TieredConfig::default();
            cfg.profile_config = ProfileConfig {
                warm_threshold: 0,
                hot_threshold: u32::MAX as u64,
                ..ProfileConfig::default()
            };
            zynml.load_source(&src).map_err(|e| format!("{e:?}"))?;
            zynml
                .runtime_mut()
                .install_interp_jit_with(cfg)
                .map_err(|e| format!("{e:?}"))?;
            zynml
                .runtime()
                .call_function_raw("main", vec![])
                .map_err(|e| format!("{e:?}"))
        })();
        let _ = tx.send(got);
    });
    rx.recv_timeout(Duration::from_secs(180))
        .expect("the kernel should finish")
        .expect("the kernel should run")
}

/// A counted loop with a loop inside it, over storage the caller owns.
/// Every row belongs to one iteration, so the rows may be filled in any
/// order.
const ROWS: &str = r#"
def fill_rows(mut o: Ptr<f32>, rows: i64, cols: i64): i64 {
    let mut i: i64 = 0
    while i < rows {
        let mut j: i64 = 0
        while j < cols {
            o[i * cols + j] = ((i * 2 + j) as f32)
            j = j + 1
        }
        i = i + 1
    }
    return 0
}
def main(): i64 {
    let rows: i64 = 96
    let cols: i64 = 64
    let o: Ptr<f32> = alloc_f32(rows * cols)
    let w: i64 = fill_rows(o, rows, cols)
    let mut total: f32 = 0.0
    let mut i: i64 = 0
    while i < rows * cols { total = total + o[i]  i = i + 1 }
    free(o)
    return (total as i64)
}
"#;

/// Spreading the rows across cores does not change what is in them.
#[test]
fn a_spread_loop_gives_the_answer_the_serial_one_gave() {
    let _guard = serialised();
    let serial = answer(ROWS, false);
    let spread = answer(ROWS, true);
    assert_eq!(
        spread, serial,
        "the dispatched kernel should agree with the serial one"
    );
    // 96 rows of 64 holding 2i + j: 128 * sum(i) + 96 * sum(j).
    assert_eq!(serial, ZyntaxValue::Int(777_216));
}

/// Running it again gives the same answer, which a race would not
/// reliably do.
#[test]
fn the_spread_answer_is_the_same_every_time() {
    let _guard = serialised();
    let first = answer(ROWS, true);
    for _ in 0..3 {
        assert_eq!(answer(ROWS, true), first, "the answer moved between runs");
    }
}

/// A loop reusing one scratch buffer on every iteration is not
/// independent, whatever it looks like from the outside, and the
/// analysis is what has to say so.
#[test]
fn a_loop_sharing_scratch_across_iterations_is_refused() {
    let src = format!(
        "{PRE}
def run(mut out: Ptr<f32>, src: Ptr<f32>, mut scratch: Ptr<f32>, rounds: i64, n: i64): i64 {{
    let mut r: i64 = 0
    while r < rounds {{
        let mut i: i64 = 0
        while i < n {{ scratch[i] = src[i] * (r as f32)  i = i + 1 }}
        out[r] = scratch[0]
        r = r + 1
    }}
    return 0
}}"
    );
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(&src, "<sc>").expect("parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
        _ => Ok(None),
    }));
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut module = rt.lower_typed_program(program, builtins).expect("lower");
    zyntax_compiler::run_interp_safe_opts(&mut module);

    // The rounds loop is the one under test. The pass only ever
    // dispatches a loop with a loop inside it, so the inner copy, which
    // genuinely is independent, is not a candidate either way; nothing
    // being dispatched is therefore a statement about the outer one.
    let _guard = serialised();
    std::env::set_var("ZYNTAX_PARALLEL_LOOPS", "1");
    let stats = zyntax_compiler::parallel_dispatch::run_module(&mut module);
    std::env::set_var("ZYNTAX_PARALLEL_LOOPS", "0");
    assert_eq!(
        stats.dispatched, 0,
        "iterations that all write the same scratch must not be spread"
    );
}
