//! A tensor written in ZynML rather than behind an FFI boundary.
//!
//! `Tensor` today is an `extern struct` whose every operation is a
//! `Call::Symbol` into a Rust plugin, so `a + b` costs a call before it
//! costs any arithmetic. Written in ZynML the same operation is a loop of
//! vector loads, an add and a store, all of it HIR the optimiser can see
//! through.
//!
//! What these pin down is that the language can express it: a struct
//! owning a typed buffer, returned by value from a constructor, swept
//! with whole-vector loads and stores, and reduced. The buffer is
//! `alloc_f32`, so a tensor is the struct plus the memory it points at,
//! and returning one has to hand back both.

use std::time::{Duration, Instant};

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::ZyntaxRuntime;

/// Every kernel below shares this definition. `Tensor` is an ordinary
/// ZynML struct: a typed buffer and the number of elements in it.
const TENSOR: &str = r#"
import prelude
import simd

struct Tensor {
    data: Ptr<f32>,
    len: i64
}

// A tensor of `n` elements, every one of them `v`. `n` is a whole
// number of vectors; a tail loop is the next thing this needs.
def tensor_full(n: i64, v: f32): Tensor {
    let buf: Ptr<f32> = alloc_f32(n)
    let fill: f32x4 = f32x4::splat(v)
    let mut off: i64 = 0
    while off < n {
        vstore_f32x4(buf + off, fill)
        off = off + 4
    }
    return Tensor { data: buf, len: n }
}

// Element-wise sum into a fresh tensor.
def tensor_add(a: Tensor, b: Tensor): Tensor {
    let out: Ptr<f32> = alloc_f32(a.len)
    let mut off: i64 = 0
    while off < a.len {
        let av: f32x4 = vload_f32x4(a.data + off)
        let bv: f32x4 = vload_f32x4(b.data + off)
        vstore_f32x4(out + off, av + bv)
        off = off + 4
    }
    return Tensor { data: out, len: a.len }
}

// Element-wise product into a fresh tensor.
def tensor_mul(a: Tensor, b: Tensor): Tensor {
    let out: Ptr<f32> = alloc_f32(a.len)
    let mut off: i64 = 0
    while off < a.len {
        let av: f32x4 = vload_f32x4(a.data + off)
        let bv: f32x4 = vload_f32x4(b.data + off)
        vstore_f32x4(out + off, av * bv)
        off = off + 4
    }
    return Tensor { data: out, len: a.len }
}

// Add every element, keeping four partial sums until the end.
def tensor_sum(t: Tensor): f32 {
    let mut acc: f32x4 = f32x4::splat(0.0)
    let mut off: i64 = 0
    while off < t.len {
        acc = acc + vload_f32x4(t.data + off)
        off = off + 4
    }
    return acc.sum()
}

def tensor_free(t: Tensor) {
    free(t.data)
}
"#;

fn build(kernel: &str) -> ZyntaxRuntime {
    let source = format!("{TENSOR}\n{kernel}");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(&source, "<tensor_in_zynml>")
        .expect("source should parse");

    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
        _ => Ok(None),
    }));
    rt.compile_typed_program(program).expect("should compile");
    rt
}

fn run(kernel: &str, entry: &str) -> zyntax_compiler::value::ZyntaxValue {
    build(kernel)
        .call_function_raw(entry, vec![])
        .expect("should run")
}

fn poll_until(deadline: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    cond()
}

/// A constructor hands back both the struct and the buffer it owns.
///
/// This is the shape that used to return the callee's dead frame, so
/// two tensors shared one buffer. 8 elements of 2.5 is 20.
#[test]
fn a_constructed_tensor_owns_its_buffer() {
    let got = run(
        r#"
        def main(): f32 {
            let t: Tensor = tensor_full(8, 2.5)
            let s: f32 = tensor_sum(t)
            tensor_free(t)
            return s
        }
        "#,
        "main",
    );
    assert_eq!(got.as_float(), Some(20.0), "got {got:?}");
}

/// Two tensors built separately stay separate: 8*1.5 + 8*3.0 = 36.
///
/// Independent from the test above in the way that matters, because a
/// shared destination would make the second constructor overwrite the
/// first and both sums would read the same buffer.
#[test]
fn two_tensors_do_not_share_a_buffer() {
    let got = run(
        r#"
        def main(): f32 {
            let a: Tensor = tensor_full(8, 1.5)
            let b: Tensor = tensor_full(8, 3.0)
            let s: f32 = tensor_sum(a) + tensor_sum(b)
            tensor_free(a)
            tensor_free(b)
            return s
        }
        "#,
        "main",
    );
    assert_eq!(got.as_float(), Some(36.0), "got {got:?}");
}

/// Element-wise add through a whole chain: (1.5 + 3.0) * 8 = 36.
#[test]
fn elementwise_add_over_a_buffer() {
    let got = run(
        r#"
        def main(): f32 {
            let a: Tensor = tensor_full(8, 1.5)
            let b: Tensor = tensor_full(8, 3.0)
            let c: Tensor = tensor_add(a, b)
            let s: f32 = tensor_sum(c)
            tensor_free(a)
            tensor_free(b)
            tensor_free(c)
            return s
        }
        "#,
        "main",
    );
    assert_eq!(got.as_float(), Some(36.0), "got {got:?}");
}

/// Element-wise multiply, and a result feeding another operation:
/// (2.0 * 3.0) * 16 = 96.
#[test]
fn elementwise_multiply_chains_into_a_reduction() {
    let got = run(
        r#"
        def main(): f32 {
            let a: Tensor = tensor_full(16, 2.0)
            let b: Tensor = tensor_full(16, 3.0)
            let c: Tensor = tensor_mul(a, b)
            let s: f32 = tensor_sum(c)
            tensor_free(a)
            tensor_free(b)
            tensor_free(c)
            return s
        }
        "#,
        "main",
    );
    assert_eq!(got.as_float(), Some(96.0), "got {got:?}");
}

/// The same answers once the functions are compiled rather than
/// interpreted.
///
/// Worth its own test because the two execution paths represent a
/// returned struct differently. The interpreter carries one as a value
/// and never had trouble with it; the compiled path builds it in memory
/// and has to put that memory somewhere its caller can still read. Every
/// case above would pass on the interpreter alone while the compiled
/// tensor handed back a buffer belonging to a frame that was gone.
#[test]
fn the_compiled_tensor_agrees_with_the_interpreted_one() {
    let kernel = r#"
        def main(): f32 {
            let a: Tensor = tensor_full(8, 1.5)
            let b: Tensor = tensor_full(8, 3.0)
            let c: Tensor = tensor_add(a, b)
            let s: f32 = tensor_sum(c) + tensor_sum(a)
            tensor_free(a)
            tensor_free(b)
            tensor_free(c)
            return s
        }
        "#;

    // (1.5 + 3.0) * 8 = 36, plus 1.5 * 8 = 12, so 48. Reading `a` after
    // `c` was built from it is what a shared destination would break.
    const EXPECTED: f64 = 48.0;

    let mut rt = build(kernel);
    let mut cfg = TieredConfig::default();
    cfg.profile_config.warm_threshold = 1;
    rt.install_interp_jit_with(cfg).expect("install interp JIT");

    for _ in 0..4 {
        let got = rt.call_function_raw("main", vec![]).expect("call");
        assert_eq!(got.as_float(), Some(EXPECTED), "interpreted: {got:?}");
    }

    let func_ids = rt.interp_registered_function_ids();
    let tiered_up = poll_until(Duration::from_millis(4000), || {
        func_ids.iter().any(|fid| rt.interp_function_compiled(*fid))
    });
    assert!(tiered_up, "nothing tiered up to Cranelift");

    let got = rt.call_function_raw("main", vec![]).expect("call");
    assert_eq!(got.as_float(), Some(EXPECTED), "compiled: {got:?}");
}

/// A tensor that outlives the expression that made it, passed onward
/// and read after other tensors have been built on top of it.
///
/// `a` is constructed first and read last, with two more constructors
/// and two buffers allocated in between: 4*1.0 = 4.
#[test]
fn a_tensor_survives_later_allocations() {
    let got = run(
        r#"
        def main(): f32 {
            let a: Tensor = tensor_full(4, 1.0)
            let b: Tensor = tensor_full(4, 9.0)
            let c: Tensor = tensor_add(b, b)
            let d: Tensor = tensor_mul(c, b)
            let s: f32 = tensor_sum(a)
            tensor_free(a)
            tensor_free(b)
            tensor_free(c)
            tensor_free(d)
            return s
        }
        "#,
        "main",
    );
    assert_eq!(got.as_float(), Some(4.0), "got {got:?}");
}
