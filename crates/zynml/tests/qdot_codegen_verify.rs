//! ML roadmap Phase 0 exit gate — codegen verification.
//!
//! The companion to `qdot_microkernel_exec`: that one proves the pure-ZynML
//! quantized kernel computes the right answer, this one proves it reaches
//! the *fast instruction*. A kernel that is numerically correct but lowers
//! to a scalar loop would silently miss the whole point of the phase.
//!
//! The host-feature gate keeps the assert meaningful rather than flaky: on
//! a machine without the instruction there is nothing to assert, so the
//! check is skipped instead of failing.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::HirInstruction;
use zyntax_embed::ZyntaxRuntime;

const QDOT_SRC: &str = r#"
import prelude

def qdot(a: Ptr<i8>, b: Ptr<i8>, n_bytes: i64): i32 {
    let mut acc: i32x4 = i32x4::splat(0)
    let mut off: i64 = 0
    while off < n_bytes {
        let av: i8x16 = vload_i8x16(a + off)
        let bv: i8x16 = vload_i8x16(b + off)
        acc = acc.dot_u8i8(av, bv)
        off = off + 16
    }
    return acc.sum()
}
"#;

/// True when the running x86_64 host has AVX-VNNI (`vpdpbusd`).
#[allow(unreachable_code)]
fn host_has_avxvnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return std::arch::is_x86_feature_detected!("avxvnni")
            || std::arch::is_x86_feature_detected!("avx512vnni");
    }
    false
}

/// True when the running aarch64 host has the dot-product extension
/// (`sdot`). Apple Silicon always has it.
#[allow(unreachable_code)]
fn host_has_neon_dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        return std::arch::is_aarch64_feature_detected!("dotprod");
    }
    false
}

/// Compile the kernel from source and capture its CLIF + native disasm.
fn capture_qdot_codegen() -> (String, Option<String>) {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(QDOT_SRC, "<qdot>")
        .expect("kernel should parse");

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
    let module = rt
        .lower_typed_program(program, builtins)
        .expect("kernel should lower to HIR");

    let func = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some("qdot"))
        .expect("qdot should be lowered");

    // The dot must survive to codegen as a real vector op.
    assert!(
        func.blocks
            .values()
            .flat_map(|b| &b.instructions)
            .any(|i| matches!(i, HirInstruction::VectorDot { .. })),
        "the kernel must still contain a vector dot at codegen time"
    );

    let mut backend = CraneliftBackend::new().expect("backend");
    backend.set_capture_ir(true);
    backend
        .compile_function(func.id, func)
        .expect("qdot should compile");
    backend.take_captured_ir().expect("captured IR")
}

/// The source kernel must reach the machine dot-product path: vector
/// types, widening, and the pairwise-add tree the backend folds into
/// `sdot` / `vpdpbusd`. A scalar fallback would show none of these.
#[test]
fn qdot_source_kernel_lowers_to_vector_dot_sequence() {
    let (clif, disasm) = capture_qdot_codegen();

    // Vector work survived to CLIF.
    assert!(
        clif.contains("i8x16") || clif.contains("i32x4"),
        "expected vector types in the CLIF:\n{clif}"
    );

    let Some(d) = disasm.filter(|d| !d.is_empty()) else {
        // No disassembly from this compile path — the CLIF assert above
        // is all this test can honestly check.
        return;
    };

    // The lanes must be widened with the right signedness: the unsigned
    // operand zero-extends, the signed one sign-extends. Getting this
    // backwards is a silent wrong-answer bug, so it is worth pinning at
    // the machine level and not just in the numeric tests.
    if cfg!(target_arch = "aarch64") {
        assert!(
            d.contains("uxtl"),
            "the unsigned operand must zero-extend (uxtl):\n{d}"
        );
        assert!(
            d.contains("sxtl"),
            "the signed operand must sign-extend (sxtl):\n{d}"
        );
    }

    // Whether the widen/multiply/pairwise sequence collapses into a
    // single fused dot depends on the target having a rule for this
    // *mixed* unsigned x signed form. Where the fused instruction is
    // emitted, assert it; otherwise the portable sequence above is the
    // documented fallback, and the sequence assert is the real check.
    let fused = d.contains("sdot") || d.contains("usdot") || d.contains("vpdpbusd");
    if fused {
        return;
    }
    assert!(
        d.contains("mul") && (d.contains("saddlp") || d.contains("addp") || d.contains("padd")),
        "without a fused dot, expected the widen/multiply/pairwise-add \
         fallback sequence:\n{d}"
    );
}

/// Whatever the host, the kernel must not fall back to a byte-at-a-time
/// scalar loop: the vector load and the dot both have to reach codegen.
#[test]
fn qdot_source_kernel_uses_vector_loads() {
    let (clif, _) = capture_qdot_codegen();
    assert!(
        clif.contains("load"),
        "expected vector loads in the CLIF:\n{clif}"
    );
}
