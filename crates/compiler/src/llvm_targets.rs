//! The LLVM targets this build links: the host's, and no other.
//!
//! llvm-sys is built with `disable-alltargets-init`, so it compiles
//! none of its C wrappers around `llvm-c/Target.h`. Those wrappers sit
//! in one object, and its `LLVM_InitializeAll*` functions name every
//! target LLVM was configured with, so linking any one wrapper linked
//! every target and ran all their static initialisers at each process
//! start. inkwell still calls the wrappers by name (an execution
//! engine initialises the native target itself), so they are defined
//! here. Each `All` wrapper initialises the host target, which is every
//! target this build links.

use inkwell::llvm_sys::prelude::LLVMBool;

#[cfg(target_arch = "aarch64")]
mod host {
    pub use inkwell::llvm_sys::target::{
        LLVMInitializeAArch64AsmParser as asm_parser,
        LLVMInitializeAArch64AsmPrinter as asm_printer,
        LLVMInitializeAArch64Disassembler as disassembler, LLVMInitializeAArch64Target as target,
        LLVMInitializeAArch64TargetInfo as info, LLVMInitializeAArch64TargetMC as mc,
    };
}

#[cfg(target_arch = "x86_64")]
mod host {
    pub use inkwell::llvm_sys::target::{
        LLVMInitializeX86AsmParser as asm_parser, LLVMInitializeX86AsmPrinter as asm_printer,
        LLVMInitializeX86Disassembler as disassembler, LLVMInitializeX86Target as target,
        LLVMInitializeX86TargetInfo as info, LLVMInitializeX86TargetMC as mc,
    };
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
compile_error!("the LLVM backend links the host's LLVM target, and has none for this architecture");

/// Register the host target with LLVM, with its assembly printer,
/// parser and disassembler: everything an execution engine or an
/// object emitter here needs. Safe to call more than once. Calling the
/// wrappers from here keeps them linked whatever order a linker reads
/// the archives in; a binary that drives inkwell itself calls this
/// first, or it does not link.
pub fn initialize_host() {
    LLVM_InitializeNativeTarget();
    LLVM_InitializeNativeAsmPrinter();
    LLVM_InitializeNativeAsmParser();
    LLVM_InitializeNativeDisassembler();
}

/// Native and All wrappers share one process-wide registration.
fn initialize() {
    static INITIALIZED: std::sync::Once = std::sync::Once::new();
    INITIALIZED.call_once(|| {
        // SAFETY: registration finishes once before any caller proceeds.
        unsafe {
            host::info();
            host::target();
            host::mc();
            host::asm_parser();
            host::asm_printer();
            host::disassembler();
        }
    });
}

/// `LLVMInitializeNativeTarget`: the host's target info, target and
/// machine-code layer. Returns 0, LLVM's success.
#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeNativeTarget() -> LLVMBool {
    initialize();
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeNativeAsmParser() -> LLVMBool {
    initialize();
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeNativeAsmPrinter() -> LLVMBool {
    initialize();
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeNativeDisassembler() -> LLVMBool {
    initialize();
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllTargetInfos() {
    initialize();
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllTargets() {
    initialize();
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllTargetMCs() {
    initialize();
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllAsmPrinters() {
    initialize();
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllAsmParsers() {
    initialize();
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllDisassemblers() {
    initialize();
}
