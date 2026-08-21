//! A list element read through an element-indexed GEP survives the
//! heap scalar-replacement pass.
//!
//! `scalar_replace_alloc` forwards a store into a later load of the
//! same field by keying both on a byte offset it derives from the GEP.
//! It derived that offset by summing the indices raw, which is only the
//! right answer for the byte-offset GEPs `aggregate_split` emits. A
//! list literal is filled through those, but read back through an
//! element-indexed `gep *i64, [1]`, which the backends scale by eight
//! and the pass read as one. No store landed on offset one, so the read
//! resolved to the `Undef` the pass mints for an unwritten field.
//!
//! Only the second element onward could go wrong: index zero scales to
//! zero either way. And it only surfaced once the callee was inlined
//! into the allocating function, since the pass needs the malloc and
//! every use in one block. That is why printing the values inside the
//! callee "fixed" it -- the calls kept the callee from inlining.
//!
//! Both tiers are checked: the Cranelift backend materialised the
//! `Undef` as zero and answered 300, while the bytecode interpreter
//! refused it outright with "expected integer, got Undef". Neither
//! backend was at fault; they were handed a module that had already
//! lost the value.

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

/// The list is built in `main` and read in a callee small enough to
/// inline, so the malloc and both reads end up in one block.
const ACROSS_A_CALL: &str = r#"
import prelude
def bound(xs: List<i64>): i64 {
    let a: i64 = xs[0]
    let b: i64 = xs[1]
    return a * 100 + b
}
def main(): i64 {
    let ys: List<i64> = [3, 2]
    let r: i64 = bound(ys)
    return r
}
"#;

/// The same reads without a call, which was already right: `main`
/// allocates and reads in one block either way.
const IN_PLACE: &str = r#"
import prelude
def main(): i64 {
    let ys: List<i64> = [3, 2]
    let a: i64 = ys[0]
    let b: i64 = ys[1]
    return a * 100 + b
}
"#;

/// Four elements, so an offset that happened to survive two would not.
const FOUR_ELEMENTS: &str = r#"
import prelude
def pick(xs: List<i64>): i64 {
    return xs[0] * 1000 + xs[1] * 100 + xs[2] * 10 + xs[3]
}
def main(): i64 {
    let ys: List<i64> = [5, 6, 7, 8]
    return pick(ys)
}
"#;

#[test]
fn a_list_read_across_an_inlined_call_keeps_every_element() {
    assert_eq!(answer(ACROSS_A_CALL.to_string(), 0), ZyntaxValue::Int(302));
    assert_eq!(
        answer(ACROSS_A_CALL.to_string(), u32::MAX as u64),
        ZyntaxValue::Int(302)
    );
}

#[test]
fn a_list_read_in_place_keeps_every_element() {
    assert_eq!(answer(IN_PLACE.to_string(), 0), ZyntaxValue::Int(302));
    assert_eq!(
        answer(IN_PLACE.to_string(), u32::MAX as u64),
        ZyntaxValue::Int(302)
    );
}

#[test]
fn four_elements_survive_the_call() {
    assert_eq!(answer(FOUR_ELEMENTS.to_string(), 0), ZyntaxValue::Int(5678));
    assert_eq!(
        answer(FOUR_ELEMENTS.to_string(), u32::MAX as u64),
        ZyntaxValue::Int(5678)
    );
}
