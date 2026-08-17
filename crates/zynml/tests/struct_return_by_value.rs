//! A struct returned by value must outlive the function that built it.
//!
//! It does not. A function whose return type is an aggregate builds it
//! in an explicit stack slot and returns the address of that slot:
//!
//! ```text
//! function u0:0(i64) -> i64 apple_aarch64 {
//!     ss0 = explicit_slot 16
//! block0(v0: i64):
//!     v2 = stack_addr.i64 ss0
//!     store v4, v6          ; the pointer field
//!     store v0, v8          ; the length field
//!     return v2             ; the address of a frame about to be popped
//! }
//! ```
//!
//! The frame is gone when the caller reads it, and two calls hand back
//! the same slot address, so two values that should be independent are
//! one. Calling such a function twice and keeping both results gives a
//! pair that alias, and whatever runs next writes over them.
//!
//! Fixing it means the caller providing the destination: a hidden
//! pointer parameter that the callee writes through, which is what a C
//! ABI does with an aggregate too large for registers. That changes
//! signature building, the entry block's parameters, how a return is
//! emitted and every call site, in Cranelift and LLVM and the bytecode
//! interpreter alike, so it is written down here rather than done
//! halfway.
//!
//! Freeing both buffers aborts in the allocator, which is the same
//! fault wearing a louder coat: one address freed twice.
//!
//! Ignored because it fails. Remove the attribute with the fix.

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

/// Two calls, two buffers, two answers. The struct carries a pointer
/// because that is the shape a tensor has, and the pointer is what
/// makes the aliasing visible: both writes land in one buffer.
#[test]
#[ignore = "returns the address of the callee's own stack frame; needs a caller-provided destination"]
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
            // Deliberately not freed. Both names reach one buffer, so
            // freeing both aborts in the allocator and the number that
            // shows what went wrong never gets printed.
            // 8 and 12 while they are separate; 24 once they are one.
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
