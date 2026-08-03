//! Is MCJIT usable on this host?
//!
//! `llvm_jit_backend` emits an object file, shells out to the linker and
//! dlopens the result — ~370 ms of fixed cost per install — because MCJIT
//! was reported to hit MAP_JIT cross-thread invalidation on Apple Silicon.
//! Per-function lazy promotion is only viable if an in-process engine
//! works, so check the claim directly against the current inkwell/LLVM.

#![cfg(feature = "llvm-backend")]

use inkwell::context::Context;
use inkwell::OptimizationLevel;

type AddFn = unsafe extern "C" fn(i64, i64) -> i64;

/// Build `add(a, b) = a + b` and hand back an engine plus its JIT'd pointer.
fn jit_add(context: &Context) -> (inkwell::execution_engine::ExecutionEngine<'_>, usize) {
    let module = context.create_module("mcjit_probe");
    let builder = context.create_builder();
    let i64t = context.i64_type();
    let fn_ty = i64t.fn_type(&[i64t.into(), i64t.into()], false);
    let func = module.add_function("add", fn_ty, None);
    let entry = context.append_basic_block(func, "entry");
    builder.position_at_end(entry);
    let a = func.get_nth_param(0).unwrap().into_int_value();
    let b = func.get_nth_param(1).unwrap().into_int_value();
    let sum = builder.build_int_add(a, b, "sum").unwrap();
    builder.build_return(Some(&sum)).unwrap();

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::Default)
        .expect("MCJIT execution engine should be creatable");
    let addr = engine
        .get_function_address("add")
        .expect("JIT'd function should have an address");
    (engine, addr as usize)
}

#[test]
fn mcjit_runs_on_the_compiling_thread() {
    let context = Context::create();
    let (_engine, addr) = jit_add(&context);
    let f: AddFn = unsafe { std::mem::transmute(addr) };
    assert_eq!(unsafe { f(20, 22) }, 42);
}

/// The reported failure: code written by one thread, executed by another.
#[test]
fn mcjit_code_runs_on_a_different_thread() {
    let context = Context::create();
    let (_engine, addr) = jit_add(&context);

    let handle = std::thread::spawn(move || {
        let f: AddFn = unsafe { std::mem::transmute(addr) };
        unsafe { f(20, 22) }
    });
    assert_eq!(handle.join().expect("callee thread should not fault"), 42);
}

/// The inverse, which is the shape a background promotion thread produces:
/// compiled off-thread, called on the thread that was already running.
#[test]
fn mcjit_code_compiled_off_thread_runs_on_main() {
    let addr = std::thread::spawn(|| {
        let context = Context::create();
        let (engine, addr) = jit_add(&context);
        // Keep the engine alive for the process; dropping it frees the code.
        std::mem::forget(engine);
        std::mem::forget(context);
        addr
    })
    .join()
    .expect("compiling thread should not fault");

    let f: AddFn = unsafe { std::mem::transmute(addr) };
    assert_eq!(unsafe { f(20, 22) }, 42);
}

/// Install latency, which is what decides whether per-function promotion is
/// affordable. Reported so a change in either direction is visible.
#[test]
fn mcjit_install_latency() {
    let context = Context::create();
    // Warm LLVM's target machinery so the number reflects steady state.
    let (_warm, _) = jit_add(&context);

    let t0 = std::time::Instant::now();
    let (_engine, addr) = jit_add(&context);
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    let f: AddFn = unsafe { std::mem::transmute(addr) };
    assert_eq!(unsafe { f(1, 2) }, 3);
    eprintln!("[measure] MCJIT compile + install: {elapsed:.2} ms");
    // The object-file path costs ~370 ms per install by its own accounting,
    // so anything in that range means the in-process engine bought nothing.
    assert!(
        elapsed < 100.0,
        "in-process install should be far cheaper than link + dlopen, got {elapsed:.2} ms"
    );
}
