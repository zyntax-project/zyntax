#![cfg(all(feature = "llvm-backend", feature = "cranelift-backend"))]

mod common;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use zyntax_compiler::{
    beadie_adapter::{ZyntaxFunctionDef, build_llvm_backend},
    hir::*,
    osr, reload,
};
use zyntax_typed_ast::InternedString;

static REENTRIES: AtomicUsize = AtomicUsize::new(0);
extern "C" fn baseline_zero(n: i32) -> i32 {
    assert_eq!(n, 0);
    REENTRIES.fetch_add(1, Ordering::Relaxed);
    0
}

#[test]
fn a_late_llvm_resume_point_keeps_the_entry_and_reenters_through_its_cell() {
    let (mut f, header) = common::counted_loop();
    let id = f.id;
    let n = f.signature.params[0].id;
    let i = f.blocks[&header].phis[0].result;
    let sum = f.blocks[&header].phis[1].result;
    let body = f.blocks[&header].successors[0];
    let zero = f.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(0)));
    f.signature.is_pure = false;
    f.blocks.get_mut(&body).unwrap().instructions.insert(
        0,
        HirInstruction::Call {
            result: None,
            callee: HirCallable::Function(id),
            args: vec![zero],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
    );
    let layout = osr::osr_layout(&f, header).unwrap();
    let mut module = HirModule::new(InternedString::new_global("resume_only"));
    module.functions.insert(id, f.clone());
    let module = Arc::new(module);
    let (backend, _context) = build_llvm_backend().unwrap();
    let key = reload::next_backend_key();
    reload::set_call_target(key, id, baseline_zero as *const () as usize);
    backend.with_lock(|b| {
        b.set_use_mcjit(true);
        b.set_cross_tier_links(key, Arc::new(|_| None));
    });
    let bead = osr::next_bead_id();
    let def = ZyntaxFunctionDef {
        id,
        function: Arc::new(f),
        module,
        tier: 1,
        bead_id: bead,
    };
    let helpers = backend.resume_points(&def, [layout.site_key()].into_iter().collect());
    assert_eq!(helpers.len(), 1);
    assert!(
        backend.with_lock(|b| b.get_function_pointer(id)).is_none(),
        "making a resume point must not compile an unused normal entry"
    );
    assert_eq!(helpers[0].1, osr::helper_for(bead, layout.site_key()));
    let mut frame = vec![0u64; (layout.frame.size as usize).div_ceil(8)];
    for (v, offset) in layout.live_ins.iter().zip(&layout.frame.offsets) {
        let value: i32 = if *v == n {
            5
        } else if *v == i {
            2
        } else if *v == sum {
            1
        } else {
            panic!("unexpected live-in {v:?}")
        };
        // SAFETY: the allocated frame has the layout's size and alignment.
        unsafe {
            frame
                .as_mut_ptr()
                .cast::<u8>()
                .add(*offset as usize)
                .cast::<i32>()
                .write(value);
        }
    }
    // SAFETY: the counted loop's resume helper returns i32 from its frame pointer.
    let resume: extern "C" fn(*const u8) -> i32 = unsafe { std::mem::transmute(helpers[0].1) };
    assert_eq!(resume(frame.as_ptr().cast()), 10);
    assert_eq!(REENTRIES.load(Ordering::Relaxed), 3);
    let entry = backend.with_lock(|b| {
        b.set_compile_tier(0);
        b.compile_function(id, &def.function).unwrap();
        b.get_function_pointer(id).unwrap()
    });
    // SAFETY: the counted-loop fixture has an i32 -> i32 entry.
    let call: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(entry) };
    assert_eq!(call(5), 10);
    assert_eq!(
        backend
            .resume_points(&def, [layout.site_key()].into_iter().collect())
            .len(),
        1
    );
    assert_eq!(
        backend.with_lock(|b| b.get_function_pointer(id)),
        Some(entry)
    );
    assert_eq!(call(5), 10);
}
