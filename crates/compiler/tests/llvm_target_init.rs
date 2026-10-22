#![cfg(feature = "llvm-backend")]

#[test]
fn simultaneous_first_registration_and_backend_creation_are_idempotent() {
    let barrier = std::sync::Barrier::new(16);
    std::thread::scope(|scope| {
        for _ in 0..16 {
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..8 {
                    inkwell::targets::Target::initialize_all(&Default::default());
                    let context = inkwell::context::Context::create();
                    let backend = zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context);
                    assert!(backend.is_ok(), "{}", backend.err().unwrap());
                    inkwell::targets::Target::from_triple(
                        &inkwell::targets::TargetMachine::get_default_triple(),
                    )
                    .unwrap();
                }
            });
        }
    });
}
