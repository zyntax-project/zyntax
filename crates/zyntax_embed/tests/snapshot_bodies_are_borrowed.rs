//! Loading an embedded snapshot reads its lowered functions in place:
//! the bytes of their bodies are never copied onto the heap.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use zyntax_compiler::hir::{
    HirFunction, HirFunctionSignature, HirId, HirModule, HirType, HirValue, HirValueKind,
};
use zyntax_embed::{Snapshot, SnapshotBuilder};

/// Counts the bytes allocated on a thread while it is counting.
struct Counting;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Bytes allocated on this thread while `f` runs.
fn allocated_by<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOCATED.load(Ordering::Relaxed);
    COUNTING.with(|c| c.set(true));
    let out = f();
    COUNTING.with(|c| c.set(false));
    (out, ALLOCATED.load(Ordering::Relaxed) - before)
}

/// A module of `functions` functions, each with a body of `values`
/// values.
fn library(functions: usize, values: usize) -> HirModule {
    let mut arena = zyntax_typed_ast::AstArena::new();
    let mut module = HirModule::new(arena.intern_string("lib"));
    for f in 0..functions {
        let signature = HirFunctionSignature {
            params: vec![],
            returns: vec![],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        };
        let mut function = HirFunction::new(arena.intern_string(format!("f{f}")), signature);
        for v in 0..values {
            let id = HirId::new();
            function.values.insert(
                id,
                HirValue {
                    id,
                    ty: HirType::I64,
                    kind: HirValueKind::Parameter(v as u32),
                    uses: std::collections::HashSet::new(),
                    span: None,
                },
            );
        }
        module.functions.insert(function.id, function);
    }
    module
}

#[test]
fn snapshot_bodies_are_borrowed() {
    let bytes = SnapshotBuilder::new("demo")
        .module_lowered(
            "lib",
            zyntax_typed_ast::TypedProgram::default(),
            &library(64, 256),
        )
        .expect("module")
        .encode()
        .expect("encode");
    let image_len = bytes.len();
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());

    let (hir, allocated) = allocated_by(|| {
        let snapshot = Snapshot::load(bytes).expect("load");
        let module = snapshot.module("lib").expect("decodes").expect("present");
        module.hir().cloned().expect("the HIR came with the module")
    });
    assert!(
        allocated < image_len / 8,
        "loading allocated {allocated} bytes for an image of {image_len}: the bodies were copied"
    );

    // What was left in place still decodes.
    let f0 = hir.by_name("f0").expect("f0").id;
    let body = hir.function(f0).expect("the body decodes");
    assert_eq!(body.values.len(), 256);
}
