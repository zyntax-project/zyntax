//! A float declared `f32` is an `f32`, and the two things that go wrong
//! when it is not.
//!
//! Cranelift takes an arithmetic instruction's type from its operands
//! and a store's width from the value being stored. Neither consults
//! the type the HIR declared for the result, so an expression that the
//! source declares `f32` but that has one `f64` operand produces an
//! `f64`, and storing it writes eight bytes into a four-byte element.
//! The extra four bytes land on the next element.
//!
//! That is memory corruption rather than a rounding difference, and
//! nothing about it is visible in the HIR: the same module answers
//! correctly through the bytecode interpreter, which takes the store
//! width from the value's declared type instead.
//!
//! The literal is what widens it here. `1.0` is an `f64`, so
//! `1.0 / total` is an `f64` divide however the binding is annotated,
//! and everything computed from it stays wide.
//!
//! The second consequence is louder and was found second. A loop that
//! vectorizes takes its lane count from the first element type in the
//! body, the `f32` load, and widens every instruction to it. The `f64`
//! then becomes a `Vector(F64, 4)`: two hundred and fifty six bits, a
//! shape no Cranelift lane type encodes. The type lookup misses, a
//! default stands in for it, and the demote that follows is built over
//! a vector, which Cranelift refuses at verification time. That one
//! kills the process rather than the answer.
//!
//! Both are the same disagreement. A binding narrows to the width its
//! annotation declares now, so neither shape arises.

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

/// Every element written by a scaled store holds what was written.
///
/// `inv` is divided from a runtime value, so it is an `f64` divide
/// whatever the binding says, and the product stored into each element
/// is wide. Writing it eight bytes at a time overwrites the neighbour.
///
/// The inner loop reads its source strided, which keeps the vectorizer
/// away: widened, the store would go through a vector path that does
/// not have this fault. That is also why the kernel this came from hit
/// it in exactly one loop out of several.
const SCALED_WRITE: &str = r#"
import prelude
import simd
def scaled(mut out: Ptr<f32>, src: Ptr<f32>, rows: i64, cols: i64, total: f32): i64 {
    let inv: f32 = 1.0 / total
    let mut i: i64 = 0
    while i < rows {
        let mut acc: f32 = 0.0
        let mut t: i64 = 0
        while t < cols {
            acc = acc + src[t * rows + i]
            t = t + 1
        }
        out[i] = acc * inv
        i = i + 1
    }
    return rows
}
def main(): i64 {
    let rows: i64 = 64
    let cols: i64 = 8
    let src: Ptr<f32> = alloc_f32(rows * cols)
    let out: Ptr<f32> = alloc_f32(rows)
    let mut i: i64 = 0
    while i < rows * cols { src[i] = ((i % 4) as f32) + 1.0  i = i + 1 }
    let mut z: i64 = 0
    while z < rows { out[z] = 0.0  z = z + 1 }
    let w: i64 = scaled(out, src, rows, cols, 4.0)
    let mut total: f32 = 0.0
    let mut j: i64 = 0
    while j < rows { total = total + out[j]  j = j + 1 }
    free(src) free(out)
    return ((total * 16.0) as i64)
}
"#;

#[test]
fn a_scaled_store_writes_one_element_not_two() {
    let interpreted = answer(SCALED_WRITE.to_string(), u32::MAX as u64);
    let compiled = answer(SCALED_WRITE.to_string(), 0);
    println!("\n  interpreted {interpreted:?}   compiled {compiled:?}");
    assert_eq!(
        compiled, interpreted,
        "compiling the kernel should not change what it computes"
    );
}

/// The same expression in a loop the vectorizer widens.
///
/// Nothing is stored at the wrong width here because the whole
/// function never compiled: the widened `f64` reached a demote built
/// over a vector and Cranelift refused it, taking the process with it.
/// Reading the answer is therefore the smaller half of what this
/// checks; that it runs at all is the rest.
const VECTORIZED: &str = r#"
import prelude
import simd
def scaled(mut out: Ptr<f32>, src: Ptr<f32>, n: i64, total: f32): i64 {
    let inv: f32 = 1.0 / total
    let mut i: i64 = 0
    while i < n {
        out[i] = src[i] * inv
        i = i + 1
    }
    return n
}
def main(): i64 {
    let n: i64 = 64
    let s: Ptr<f32> = alloc_f32(n)
    let o: Ptr<f32> = alloc_f32(n)
    let mut i: i64 = 0
    while i < n { s[i] = ((i % 8) as f32) + 1.0  o[i] = 0.0  i = i + 1 }
    let w: i64 = scaled(o, s, n, 4.0)
    let mut t: f32 = 0.0
    let mut j: i64 = 0
    while j < n { t = t + o[j]  j = j + 1 }
    free(s) free(o)
    return ((t * 16.0) as i64)
}
"#;

#[test]
fn a_widened_scale_compiles_and_agrees() {
    let interpreted = answer(VECTORIZED.to_string(), u32::MAX as u64);
    let compiled = answer(VECTORIZED.to_string(), 0);
    assert_eq!(compiled, interpreted);
    // 64 elements holding (i % 8) + 1, quartered, summed, times 16:
    // eight blocks of 1+2+..+8 = 36, so 8 * 36 / 4 * 16.
    assert_eq!(interpreted, ZyntaxValue::Int(1152));
}
