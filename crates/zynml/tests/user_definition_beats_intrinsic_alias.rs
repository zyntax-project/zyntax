//! A function the program defines wins the call site over a built-in
//! that happens to share its name.
//!
//! The SSA Call handler consulted a name-keyed alias table (`sqrt`,
//! `abs`, `free`) before it looked the callee up among the program's
//! own functions, so `def free(...)` was rewritten into the
//! deallocation intrinsic at every call site. The intrinsic yields no
//! SSA value, and anything reading the call's result then referred to a
//! register nothing defined -- a backend value-map failure for `free`,
//! and a silently wrong answer for `sqrt` and `abs`.
//!
//! The stdlib entries the aliases stand in for are all `extern def`, so
//! gating the table on "declared here with a body" leaves them routed
//! to their hardware instruction. The last case checks that.
//!
//! No list appears in any of these: a list literal reads its elements
//! through a separate path with its own history, and mixing the two
//! shapes would make a failure ambiguous.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use zynml::{ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::profiling::ProfileConfig;
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::{LanguageGrammar, ZyntaxRuntime, ZyntaxValue};

/// Run `main`, compiled when `warm` is zero and interpreted when it is
/// a threshold nothing reaches.
fn answer(src: String, warm: u64) -> ZyntaxValue {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let got = (|| -> Result<ZyntaxValue, String> {
            let mut rt = ZyntaxRuntime::new().map_err(|e| format!("{e:?}"))?;
            rt.add_import_resolver(Box::new(|m| match m {
                "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
                "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
                _ => Ok(None),
            }));
            let g = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).map_err(|e| format!("{e:?}"))?;
            rt.register_grammar("zynml", g);
            rt.load_module("zynml", &src)
                .map_err(|e| format!("{e:?}"))?;
            let mut cfg = TieredConfig::default();
            cfg.profile_config = ProfileConfig {
                warm_threshold: warm,
                hot_threshold: u32::MAX as u64,
                ..Default::default()
            };
            rt.install_interp_jit_with(cfg)
                .map_err(|e| format!("{e:?}"))?;
            rt.call_function_raw("main", vec![])
                .map_err(|e| format!("{e:?}"))
        })();
        let _ = tx.send(got);
    });
    rx.recv_timeout(Duration::from_secs(180))
        .expect("the kernel should finish")
        .expect("the kernel should run")
}

/// Takes no arguments at all, so the rewritten intrinsic had nothing to
/// free and produced no value for the binding to read.
const OWN_FREE: &str = r#"
import prelude
def free(): i64 {
    return 7
}
def main(): i64 {
    let v: i64 = free()
    return v
}
"#;

/// The same shadowing without a crash: the answer was the hardware
/// square root of the argument rather than the argument. Still failing
/// -- the call reaches the right body now, but the type checker bound
/// the call's type to the prelude's `f64` signature before lowering
/// ran, so the result comes back as the bits of a float. That is a
/// second name-collision channel, upstream of the one this file covers.
const OWN_SQRT: &str = r#"
import prelude
def sqrt(x: i64): i64 {
    return x
}
def main(): i64 {
    let v: i64 = sqrt(49)
    return v
}
"#;

/// `abs` shadowed, and read twice, so a single call site cannot pass by
/// luck.
const OWN_ABS: &str = r#"
import prelude
def abs(x: i64): i64 {
    return x + 1
}
def main(): i64 {
    return abs(7) * 10 + abs(0)
}
"#;

/// The stdlib's own `extern def sqrt` has no body here, so it keeps its
/// hardware instruction. 3.0 * 3.0 is exact in binary floating point.
const STDLIB_SQRT_STILL_ROUTES: &str = r#"
import prelude
def main(): f64 {
    return sqrt(9.0)
}
"#;

fn both_tiers(src: &str, want: ZyntaxValue) {
    assert_eq!(answer(src.to_string(), 0), want, "compiled");
    assert_eq!(
        answer(src.to_string(), u32::MAX as u64),
        want,
        "interpreted"
    );
}

#[test]
fn a_program_that_defines_free_calls_its_own() {
    both_tiers(OWN_FREE, ZyntaxValue::Int(7));
}

#[test]
#[ignore = "the type checker still binds the call to the prelude's f64 sqrt"]
fn a_program_that_defines_sqrt_calls_its_own() {
    both_tiers(OWN_SQRT, ZyntaxValue::Int(49));
}

#[test]
fn a_program_that_defines_abs_calls_its_own() {
    both_tiers(OWN_ABS, ZyntaxValue::Int(81));
}

#[test]
fn the_stdlib_alias_still_reaches_the_instruction() {
    both_tiers(STDLIB_SQRT_STILL_ROUTES, ZyntaxValue::Float(3.0));
}
