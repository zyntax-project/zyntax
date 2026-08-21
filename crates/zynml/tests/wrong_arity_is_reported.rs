//! A call the callee cannot accept is an error, not a quiet nothing.
//!
//! Cranelift's verifier already rejected a mismatched argument count,
//! but the response was to skip the function, and calling a function
//! that was skipped is a no-op. So a program with a mistyped call ran,
//! did nothing, and reported success. Not even the statement before the
//! bad call executed, which is what made it read as "never called"
//! rather than "failed to build".
//!
//! It cost two whole sections of `examples/tensor_ops.zynml`, which
//! called `transpose` and `shape` with the wrong counts and silently
//! did not run for as long as that was true.

use std::path::Path;
use zynml::{ZynML, ZynMLConfig};

fn load(src: &str) -> Result<(), String> {
    let plugins = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../plugins/target/zrtl");
    let cfg = ZynMLConfig {
        plugins_dir: plugins.to_string_lossy().to_string(),
        ..ZynMLConfig::default()
    };
    let mut z = ZynML::with_config(cfg).map_err(|e| format!("{e:?}"))?;
    z.load_source(src).map_err(|e| format!("{e:?}"))?;
    Ok(())
}

/// Too few arguments, in a function that is not the entry point.
///
/// The entry point had always reported this, as "Function not found:
/// main" at the call. Any other function was silent.
#[test]
fn too_few_arguments_is_refused() {
    let err = load(
        r#"
import prelude
import tensor

def takes_two(): i64 {
    let t = Tensor::arange(0.0, 6.0, 1.0)
    return t.shape()
}

def main(): i64 { return takes_two() }
"#,
    )
    .expect_err("a call the callee cannot accept should be refused");
    assert!(
        err.contains("takes 2 arguments but is called with 1"),
        "the error should say what the mismatch is, got: {err}"
    );
    assert!(
        err.contains("takes_two"),
        "the error should name the function the call is in, got: {err}"
    );
}

/// Too many, which the same check has to catch from the other side.
#[test]
fn too_many_arguments_is_refused() {
    let err = load(
        r#"
import prelude

def one(a: i64): i64 { return a }
def caller(): i64 { return one(1, 2) }
def main(): i64 { return caller() }
"#,
    )
    .expect_err("a call the callee cannot accept should be refused");
    assert!(
        err.contains("is called with 2") || err.contains("takes 1 argument"),
        "the error should describe the mismatch, got: {err}"
    );
}

/// A program whose calls all agree still builds. The check runs over
/// every call in the module, so a false positive here would refuse
/// working code.
#[test]
fn correct_arity_still_builds() {
    load(
        r#"
import prelude
import tensor

def shaped(): i64 {
    let t = Tensor::arange(0.0, 6.0, 1.0)
    return t.shape(0)
}

def main(): i64 { return shaped() }
"#,
    )
    .expect("a program whose calls agree should build");
}

/// A default the compiler fills must not read as a mismatch.
///
/// An ordinary `def` has its defaults applied before HIR, so the counts
/// agree by the time this check sees them. This is the false positive
/// that would matter: refusing it would refuse working programs.
#[test]
fn a_filled_default_is_not_a_mismatch() {
    load(
        r#"
import prelude

def greet(a: i64, b: i64 = 5): i64 { return a + b }
def caller(): i64 { return greet(1) }
def main(): i64 { return caller() }
"#,
    )
    .expect("a default the compiler fills should not read as a mismatch");
}

/// A default on an `extern def` is NOT filled, and is reported.
///
/// `Tensor::arange` declares `step: f64 = 1.0` and passing two of three
/// reaches the backend two-of-three. Before this check that was a
/// function Cranelift refused and the runtime skipped, surfacing as
/// "Function not found: main" from the entry point and as nothing at
/// all anywhere else. Reporting it is an improvement, not the fix: the
/// default should be filled. This pins the current behaviour so that
/// filling it is a visible change rather than a silent one.
#[test]
fn an_unfilled_extern_default_is_reported_rather_than_skipped() {
    let err = load(
        r#"
import prelude
import tensor

def uses_default(): i64 {
    let t = Tensor::arange(0.0, 6.0)
    return t.numel()
}

def main(): i64 { return uses_default() }
"#,
    )
    .expect_err("an unfilled extern default currently reaches the backend short");
    assert!(
        err.contains("takes 3 arguments but is called with 2"),
        "it should say what is missing, got: {err}"
    );
}
