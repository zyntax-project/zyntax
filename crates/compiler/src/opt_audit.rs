//! Per-function counts of the optimiser's work, for tests that hold
//! each body to one run of the pipeline and every compile above the
//! baseline to that body. Off until [`enable`]; while off, each hook
//! costs one relaxed load.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::hir::{HirFunction, HirId, HirModule};

static ON: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
struct Counts {
    pipeline: HashMap<HirId, usize>,
    finishing: HashMap<HirId, usize>,
    llvm: HashMap<HirId, Vec<usize>>,
}

fn counts() -> &'static Mutex<Counts> {
    static C: OnceLock<Mutex<Counts>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(Counts::default()))
}

/// Start counting, for the rest of the process.
pub fn enable() {
    ON.store(true, Ordering::Relaxed);
}

fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// A run of the whole pipeline over `module`: one for each function it
/// optimises, those not through it already.
pub(crate) fn note_pipeline(module: &HirModule) {
    if !on() {
        return;
    }
    let mut c = counts().lock().unwrap();
    for (id, f) in &module.functions {
        if !f.attributes.optimized && !f.is_external {
            *c.pipeline.entry(*id).or_default() += 1;
        }
    }
}

/// The incremental passes run over `id`, a body that arrived optimised.
pub(crate) fn note_finishing(id: HirId) {
    if on() {
        *counts().lock().unwrap().finishing.entry(id).or_default() += 1;
    }
}

/// `body` handed to the LLVM tier as the HIR of `id`.
#[cfg_attr(not(feature = "llvm-backend"), allow(dead_code))]
pub(crate) fn note_llvm_body(id: HirId, body: &Arc<HirFunction>) {
    if on() {
        counts()
            .lock()
            .unwrap()
            .llvm
            .entry(id)
            .or_default()
            .push(Arc::as_ptr(body) as usize);
    }
}

/// How many runs of the whole pipeline optimised `id`.
pub fn pipeline_runs(id: HirId) -> usize {
    counts()
        .lock()
        .unwrap()
        .pipeline
        .get(&id)
        .copied()
        .unwrap_or(0)
}

/// How many times the incremental passes ran over `id`.
pub fn finishing_runs(id: HirId) -> usize {
    counts()
        .lock()
        .unwrap()
        .finishing
        .get(&id)
        .copied()
        .unwrap_or(0)
}

/// The address of each body the LLVM tier was handed for `id`, in the
/// order it was handed them.
pub fn llvm_bodies(id: HirId) -> Vec<usize> {
    counts()
        .lock()
        .unwrap()
        .llvm
        .get(&id)
        .cloned()
        .unwrap_or_default()
}
