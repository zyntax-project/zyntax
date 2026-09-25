//! A lambda held in a local and called through it is compiled with the
//! convention its call site uses.
//!
//! A debug build of the Cranelift backend refuses a call whose known
//! callee is declared with another convention. Running the lambda
//! kernel checks that on this host; compiling it for Windows x64, where
//! Cranelift's `Fast` and the platform convention put arguments in
//! different registers and preserve different ones, checks it where a
//! disagreement would change what the program does.

use std::path::Path;
use zynml::ZynML;

const KERNEL: &str = "bench_lambda_call.zynml";

fn kernel_source() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join(KERNEL);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn the_lambda_kernel_returns_its_sum() {
    let mut z = ZynML::new().expect("runtime");
    z.load_source(&kernel_source()).expect("compile the kernel");
    assert_eq!(z.call_with_result::<i64>("main").expect("run"), 35_000_000);
}

/// The kernel's HIR, lowered and optimised as the classic runtime
/// lowers and optimises it before compiling.
#[cfg(target_arch = "x86_64")]
fn kernel_hir(z: &ZynML) -> zyntax_compiler::HirModule {
    let rt = z.runtime();
    let grammar = rt.get_grammar("zynml").expect("the ZynML grammar");
    let program = grammar
        .parse_with_signatures(&kernel_source(), KERNEL, rt.plugin_signatures())
        .expect("parse the kernel");
    let builtins = grammar
        .builtins()
        .functions
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut module = rt
        .lower_typed_program(program, builtins)
        .expect("lower the kernel");
    zyntax_compiler::run_interp_safe_opts(&mut module);
    zyntax_compiler::run_native_only_opts(&mut module);
    module
}

/// The kernel compiled for Windows x64, nothing run, with or without
/// OSR probes. `ZYNTAX_DUMP_CRANELIFT=` prints each function's CLIF
/// with the convention of its signature and of every call it makes.
#[cfg(target_arch = "x86_64")]
fn compile_for_windows_x64(osr_probes: bool) {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let z = ZynML::new().expect("runtime");
    let module = kernel_hir(&z);
    let mut backend =
        CraneliftBackend::for_target("x86_64-pc-windows-msvc").expect("a Windows x64 backend");
    backend.set_emit_osr_probes(osr_probes);
    backend.set_only_compile_reachable(Some(zyntax_compiler::reachable_function_ids(
        &module,
        &["main"],
    )));
    backend
        .compile_module(&module)
        .expect("compile for Windows x64");
    let main = module
        .functions
        .iter()
        .find(|(_, f)| f.name.resolve_global().as_deref() == Some("main"))
        .map(|(id, _)| *id)
        .expect("the kernel has a main");
    assert!(
        backend.get_function_ptr(main).is_some(),
        "main compiled for Windows x64"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn the_lambda_kernel_compiles_for_windows_x64() {
    compile_for_windows_x64(true);
}

/// Without OSR probes, as the classic runtime compiles: the loop counter
/// then lives in a register the platform convention preserves.
#[cfg(target_arch = "x86_64")]
#[test]
fn the_lambda_kernel_compiles_for_windows_x64_without_osr_probes() {
    compile_for_windows_x64(false);
}
