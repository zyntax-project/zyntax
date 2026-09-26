//! A call between two functions of one JIT module reaches its callee
//! however much address space the process maps between their compiles.
//!
//! A call inside a module is PC-relative, which reaches 2 GiB on x86-64.
//! A program that maps gigabytes between two compiles (a coroutine's
//! stack is 64 MiB, and a deep chain of them holds many) must not leave
//! the second function out of the first's reach.
#![cfg(unix)]

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::{
    BinaryOp, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirInstruction,
    HirTerminator, HirType, HirValueKind,
};
use zyntax_typed_ast::InternedString;

fn returns_i64() -> HirFunctionSignature {
    HirFunctionSignature {
        params: vec![],
        returns: vec![HirType::I64],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    }
}

/// `fn callee() -> i64 { 42 }`
fn callee() -> HirFunction {
    let mut f = HirFunction::new(InternedString::new_global("callee"), returns_i64());
    let k = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(42)));
    let entry = f.entry_block;
    f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![k] };
    f
}

/// `fn caller() -> i64 { callee() + 1 }`
fn caller(callee: &HirFunction) -> HirFunction {
    let mut f = HirFunction::new(InternedString::new_global("caller"), returns_i64());
    let got = f.create_value(HirType::I64, HirValueKind::Instruction);
    let one = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(1)));
    let sum = f.create_value(HirType::I64, HirValueKind::Instruction);
    let entry = f.entry_block;
    let blk = f.blocks.get_mut(&entry).unwrap();
    blk.instructions.push(HirInstruction::Call {
        result: Some(got),
        callee: HirCallable::Function(callee.id),
        args: vec![],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    blk.instructions.push(HirInstruction::Binary {
        op: BinaryOp::Add,
        result: sum,
        ty: HirType::I64,
        left: got,
        right: one,
    });
    blk.terminator = HirTerminator::Return { values: vec![sum] };
    f
}

/// Address space mapped inaccessible, as a fiber stack's reservation
/// is, and unmapped when dropped.
struct Reservation(*mut libc::c_void, usize);

impl Reservation {
    fn new(bytes: usize) -> Self {
        // SAFETY: an anonymous mapping at an address of the system's
        // choosing, touching nothing that exists.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED, "reserving {bytes} bytes");
        Reservation(at, bytes)
    }

    fn start(&self) -> usize {
        self.0 as usize
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `new`, unmapped once.
        unsafe { libc::munmap(self.0, self.1) };
    }
}

/// Map the gaps above `floor` a new mapping could take, so that the
/// next one lands below it. Linux places a mapping in the highest gap
/// it fits, so one that lands below `floor` shows no gap of its size is
/// left above; elsewhere this only approximates that.
fn fill_gaps_above(floor: usize) -> Vec<Reservation> {
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let mut held = Vec::new();
    let mut size = 1 << 30;
    while size >= page {
        for _ in 0..4096 {
            let gap = Reservation::new(size);
            if gap.start() < floor {
                break;
            }
            held.push(gap);
        }
        size /= 4;
    }
    held
}

#[test]
fn a_call_reaches_a_function_compiled_before_gigabytes_were_mapped() {
    let mut backend = CraneliftBackend::new().expect("backend");
    let callee = callee();
    backend
        .compile_function(callee.id, &callee)
        .expect("callee");
    backend.finalize_definitions().expect("callee finalized");

    // More than the reach of a PC-relative call, mapped after the
    // callee, and nothing left above it for the caller's code to take.
    let wide = Reservation::new(4 << 30);
    let gaps = fill_gaps_above(wide.start());

    // Without a direct call to relocate this would prove nothing: a
    // backend that routes calls through cells reaches any address.
    let caller = caller(&callee);
    backend
        .compile_function(caller.id, &caller)
        .expect("caller");
    backend.finalize_definitions().expect("caller finalized");

    let at = |id| backend.get_function_ptr(id).expect("compiled") as usize;
    let (to, from) = (at(callee.id), at(caller.id));
    assert!(
        to.abs_diff(from) < 1 << 31,
        "callee at {to:#x} and caller at {from:#x} are more than 2 GiB apart"
    );
    // SAFETY: compiled above with this signature, and finalized.
    let run: extern "C" fn() -> i64 = unsafe { std::mem::transmute(from) };
    assert_eq!(run(), 43);
    drop((gaps, wide));
}
