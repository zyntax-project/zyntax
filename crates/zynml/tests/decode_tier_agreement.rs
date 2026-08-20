//! The decode kernel answers the same whether it is interpreted or
//! compiled.
//!
//! It did not. Compiled, the second decode step contributed nothing:
//! 1743 interpreted against 876 compiled, and at the kernel's own
//! sixty-four steps the residual stream then left the representable
//! range and the conversion at the end trapped.
//!
//! The cause was a store written at the wrong width. Cranelift takes an
//! arithmetic instruction's type from its operands and a store's width
//! from the value being stored, and neither consults the type the HIR
//! declared for the result. `inv = 1.0 / total` is an `f64` divide
//! because the literal is one, so `acc2 * inv` came out `f64` even
//! though the source and the HIR both say `f32`, and storing it wrote
//! eight bytes into a four-byte element. Each element the attention
//! loop wrote was overwritten by the next iteration's high half, and
//! the last one ran off the end of the buffer.
//!
//! `float_store_width.rs` holds the narrow case. This is the whole
//! kernel, kept because the shape that reached the fault needed several
//! things at once: a loop the vectorizer declines because one operand
//! is strided, a value that stays wide because no cast or phi edge
//! happened to narrow it, and a store with no annotated binding to
//! force the issue. That combination is easier to keep than to
//! reconstruct.

use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use zynml::ZynML;
use zyntax_compiler::profiling::ProfileConfig;
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::ZyntaxValue;

/// Two steps: enough for the second one's absence to show, few enough
/// to run quickly.
fn source(steps: u32) -> String {
    let f = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/bench_llm_decode.zynml");
    std::fs::read_to_string(f)
        .expect("the decode kernel should be readable")
        .replace("let steps: i64 = 64", &format!("let steps: i64 = {steps}"))
}

/// Run `main` with the given warm-up threshold. Zero compiles the
/// function before it runs; a threshold nothing reaches leaves it to
/// the bytecode interpreter.
fn answer(src: String, warm: u64) -> ZyntaxValue {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let got = (|| -> Result<ZyntaxValue, String> {
            let mut z = ZynML::new().map_err(|e| format!("{e:?}"))?;
            let mut cfg = TieredConfig::default();
            cfg.profile_config = ProfileConfig {
                warm_threshold: warm,
                hot_threshold: u32::MAX as u64,
                ..Default::default()
            };
            z.load_source(&src).map_err(|e| format!("{e:?}"))?;
            z.runtime_mut()
                .install_interp_jit_with(cfg)
                .map_err(|e| format!("{e:?}"))?;
            z.runtime()
                .call_function_raw("main", vec![])
                .map_err(|e| format!("{e:?}"))
        })();
        let _ = tx.send(got);
    });
    rx.recv_timeout(Duration::from_secs(180))
        .expect("the kernel should finish")
        .expect("the kernel should run")
}

/// Compiling the kernel does not change what it computes.
#[test]
fn the_tiers_agree_on_the_decode_kernel() {
    let src = source(2);
    let interpreted = answer(src.clone(), u32::MAX as u64);
    let compiled = answer(src, 0);
    assert_eq!(
        compiled, interpreted,
        "compiling the kernel should not change what it computes"
    );
    assert_eq!(interpreted, ZyntaxValue::Int(1743));
}

/// Every step contributes. A step that computed nothing was how the
/// wrong store first showed: the answer for two steps equalled the
/// answer for one.
#[test]
fn each_step_moves_the_answer() {
    let mut last = 0i64;
    for steps in [1u32, 2, 3] {
        let got = answer(source(steps), 0);
        let ZyntaxValue::Int(n) = got else {
            panic!("expected an integer, got {got:?}");
        };
        assert!(
            n > last,
            "step {steps} left the total at {n}, no higher than {last} for the step before"
        );
        last = n;
    }
}

/// The whole kernel at its own length. Before the fix this trapped:
/// the clobbered elements grew without bound until the conversion at
/// the end had nothing representable to convert.
///
/// Compiled only. Interpreting sixty-four steps takes two and a half
/// minutes and would say nothing the two-step comparison above has not
/// already said; what is worth checking here is that the kernel reaches
/// its end at all, and with the value the interpreter gave when it was
/// asked once.
#[test]
fn sixty_four_steps_finish_without_trapping() {
    assert_eq!(answer(source(64), 0), ZyntaxValue::Int(55842));
}
