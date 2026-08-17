//! A struct returned by value outlives the function that built it.
//!
//! An aggregate is a value in the IR: `insertvalue` builds one and
//! `return` hands it back, with no memory named anywhere. A backend that
//! represents that value as the address of some bytes has to put the
//! bytes somewhere the caller can still read, which its own frame is
//! not. So the caller provides the memory and passes its address, and
//! two calls get two destinations.
//!
//! What this pins down is that the second call does not land on the
//! first one's result. The struct carries a pointer because that is the
//! shape a tensor has, and a pointer is what makes sharing visible: with
//! one buffer between them both names reach the same elements and the
//! writes pile up.

use std::path::Path;
use zynml::{ZynML, ZynMLConfig};

fn runtime() -> Option<ZynML> {
    let plugins = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("plugins")
        .join("target")
        .join("zrtl");
    let config = ZynMLConfig {
        plugins_dir: plugins.to_string_lossy().to_string(),
        ..ZynMLConfig::default()
    };
    ZynML::with_config(config).ok()
}

/// Two calls, two buffers, two answers.
///
/// Both buffers are freed, which is the other half of the claim: two
/// names reaching one allocation frees it twice and the allocator aborts
/// rather than returning.
#[test]
fn two_calls_returning_a_struct_give_two_independent_values() {
    let Some(mut zynml) = runtime() else {
        return;
    };

    let source = r#"
        import prelude
        import simd

        struct Buf {
            data: Ptr<f32>,
            len: i64
        }

        def make(n: i64): Buf {
            return Buf { data: alloc_f32(n), len: n }
        }

        def main(): i64 {
            let a: Buf = make(4)
            let b: Buf = make(4)
            vstore_f32x4(a.data, f32x4::splat(2.0))
            vstore_f32x4(b.data, f32x4::splat(3.0))
            let sa: f32 = vload_f32x4(a.data).sum()
            let sb: f32 = vload_f32x4(b.data).sum()
            free(a.data)
            free(b.data)
            // 4*2 + 4*3 while they are separate; 24 once they are one.
            return ((sa + sb) as i64)
        }
    "#;

    zynml.load_source(source).expect("should compile");
    let got: i64 = zynml.call_with_result("main").expect("should run");
    assert_eq!(
        got, 20,
        "two calls should give two buffers: 4*2 + 4*3. Got {got}, \
         which means both names reached one buffer."
    );
}

/// The length field travels too, not only the pointer.
///
/// A copy that moved the first word and stopped would still pass the
/// test above, since the sums only read through `data`.
#[test]
fn every_field_of_a_returned_struct_arrives() {
    let Some(mut zynml) = runtime() else {
        return;
    };

    let source = r#"
        import prelude
        import simd

        struct Buf {
            data: Ptr<f32>,
            len: i64
        }

        def make(n: i64): Buf {
            return Buf { data: alloc_f32(n), len: n }
        }

        def main(): i64 {
            let a: Buf = make(4)
            let b: Buf = make(7)
            free(a.data)
            free(b.data)
            return a.len * 10 + b.len
        }
    "#;

    zynml.load_source(source).expect("should compile");
    let got: i64 = zynml.call_with_result("main").expect("should run");
    assert_eq!(got, 47, "a.len should be 4 and b.len should be 7");
}
