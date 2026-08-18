//! The vector shapes reach the machine's vector instructions.
//!
//! A kernel that vectorises in the IR and then lowers to scalar code has
//! gained nothing, and nothing about the HIR says which happened. These
//! read the native disassembly and assert on the instruction that should
//! be there, per architecture, so a shape that stops folding is caught
//! where it stops rather than as a number in a benchmark months later.
//!
//! `qdot_codegen_verify.rs` covers the dot-product path; this covers the
//! elementwise, fused-multiply-add and reduction shapes.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_embed::ZyntaxRuntime;

/// Compile one function through the optimiser and return its native
/// disassembly, or `None` where this build cannot produce one.
fn disassemble(src: &str, name: &str) -> Option<String> {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(src, "<codegen>")
        .expect("parse");
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
    // The vectorizer lives here, so the shapes below depend on it having run.
    zyntax_compiler::run_interp_safe_opts(&mut module);

    let func = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(name))
        .expect("the kernel should be lowered")
        .clone();

    let mut backend = CraneliftBackend::new().expect("backend");
    backend.set_capture_ir(true);
    backend.compile_function(func.id, &func).expect("compile");
    backend
        .take_captured_ir()
        .and_then(|(_clif, disasm)| disasm)
        .filter(|d| !d.is_empty())
}

/// Assert the disassembly contains whichever mnemonic this architecture
/// uses for the shape. Skips where no disassembly is available rather
/// than asserting on nothing.
fn assert_folds(src: &str, name: &str, aarch64: &str, x86_64: &str) {
    let Some(disasm) = disassemble(src, name) else {
        eprintln!("no disassembly from this build; nothing to assert for {name}");
        return;
    };
    let wanted = if cfg!(target_arch = "aarch64") {
        aarch64
    } else if cfg!(target_arch = "x86_64") {
        x86_64
    } else {
        eprintln!("no expected mnemonic recorded for this architecture");
        return;
    };
    assert!(
        disasm.contains(wanted),
        "{name} should reach `{wanted}`; disassembly was:\n{disasm}"
    );
}

const ELEMENTWISE: &str = r#"
import prelude
import simd
def vadd(out: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { out[i] = a[i] + b[i] i = i + 1 }
    return n
}
"#;

const HAND_WRITTEN: &str = r#"
import prelude
import simd
def hv(out: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, n: i64): i64 {
    let mut i: i64 = 0
    while i + 4 <= n {
        vstore_f32x4(out + i, vload_f32x4(a + i) * vload_f32x4(b + i))
        i = i + 4
    }
    return n
}
"#;

const FMA: &str = r#"
import prelude
import simd
def fma4(a: Ptr<f32>, b: Ptr<f32>, c: Ptr<f32>): f32 {
    let x: f32x4 = vload_f32x4(a)
    let y: f32x4 = vload_f32x4(b)
    let z: f32x4 = vload_f32x4(c)
    let r: f32x4 = x * y + z
    return r[0]
}
"#;

const INT_REDUCE: &str = r#"
import prelude
import simd
def isum(a: Ptr<i32>): i32 {
    let v: i32x4 = vload_i32x4(a)
    return v.sum()
}
"#;

/// A scalar loop the vectorizer rewrote must add four lanes at a time,
/// not one.
#[test]
fn an_auto_vectorized_loop_adds_four_lanes_at_a_time() {
    assert_folds(ELEMENTWISE, "vadd", "fadd v", "vaddps");
}

/// The same for a loop written with the intrinsics directly.
#[test]
fn hand_written_vector_ops_multiply_four_lanes_at_a_time() {
    assert_folds(HAND_WRITTEN, "hv", "fmul v", "vmulps");
}

/// A multiply feeding an add becomes one instruction, not two.
#[test]
fn a_multiply_and_add_fuse() {
    assert_folds(FMA, "fma4", "fmla v", "vfmadd");
}

/// Integer horizontal add has a hardware idiom on both, and uses it.
///
/// The float reduction deliberately does not: `faddp` / a shuffle tree
/// reassociates, and a float sum is not associative, so the serial chain
/// is what keeps the result the one the source asked for.
#[test]
fn an_integer_reduction_uses_the_horizontal_add() {
    assert_folds(INT_REDUCE, "isum", "addv", "vphaddd");
}
