//! Function-level hot reload: content fingerprints, the call cells
//! reloadable calls dispatch through, and the report a reload returns.
//!
//! A reloadable call does not name its callee's code directly; it loads
//! the callee's current entry out of a per-function cell and calls
//! through it. Replacing a function is then one store: every caller —
//! compiled before or after the edit — observes the new entry on its
//! next call. Cells live for the whole process, like the OSR helper
//! slots they are modeled on, so a stale frame can still complete
//! through the pointer it already loaded.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{OnceLock, RwLock};

use crate::hir::{HirFunction, HirId, HirModule};

/// Fingerprint of a function's behavior.
///
/// Hashes the canonical dump, which names values and blocks by position
/// and callees by name — so it is stable across parses (fresh `HirId`s
/// hash identically) and across edits that do not change the lowered
/// function.
pub fn function_fingerprint(func: &HirFunction, module: &HirModule) -> u64 {
    use std::hash::{Hash, Hasher};
    let dump = crate::hir_dump::dump_function(func, module);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    dump.hash(&mut hasher);
    hasher.finish()
}

/// A backend's identity in the cell registry. One process can hold
/// several backends (tests do), and their `HirId` spaces overlap.
pub fn next_backend_key() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn cells() -> &'static RwLock<HashMap<(u64, HirId), Box<AtomicUsize>>> {
    static CELLS: OnceLock<RwLock<HashMap<(u64, HirId), Box<AtomicUsize>>>> = OnceLock::new();
    CELLS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Address of the cell a reloadable call to `func` loads. Created on
/// first use; the address is stable for the life of the process.
pub fn call_cell_addr(backend: u64, func: HirId) -> usize {
    if let Some(cell) = cells().read().unwrap().get(&(backend, func)) {
        return cell.as_ref() as *const AtomicUsize as usize;
    }
    let mut map = cells().write().unwrap();
    let cell = map
        .entry((backend, func))
        .or_insert_with(|| Box::new(AtomicUsize::new(0)));
    cell.as_ref() as *const AtomicUsize as usize
}

/// Publish `ptr` as the current entry of `func`.
pub fn set_call_target(backend: u64, func: HirId, ptr: usize) {
    let addr = call_cell_addr(backend, func);
    // SAFETY: `addr` points at an `AtomicUsize` boxed into the
    // process-lifetime registry above.
    unsafe { &*(addr as *const AtomicUsize) }.store(ptr, Ordering::Release);
}

/// Current entry of `func`, or 0 if none was ever published.
pub fn call_target(backend: u64, func: HirId) -> usize {
    let addr = call_cell_addr(backend, func);
    // SAFETY: as in `set_call_target`.
    unsafe { &*(addr as *const AtomicUsize) }.load(Ordering::Acquire)
}

/// What a reload did, per function, by name.
#[derive(Debug, Default, Clone)]
pub struct ReloadReport {
    /// Changed functions whose new code is now live.
    pub reloaded: Vec<String>,
    /// Functions the edit introduced.
    pub added: Vec<String>,
    /// Functions the edit removed. Their old code stays callable —
    /// anything still holding a reference completes safely.
    pub removed_retained: Vec<String>,
    /// Functions whose fingerprint did not change.
    pub unchanged: usize,
    /// Functions that failed to recompile, with the error. The
    /// previous code keeps running for each.
    pub failed: Vec<(String, String)>,
    /// Reloaded functions whose running loops can finish in the edited
    /// code: at least one resume point was published against the sites
    /// the old code is probing.
    pub resume_published: Vec<String>,
    /// Loops that could not get a resume point, with the reason. The
    /// enclosing call still completes on the old code; the *next* call
    /// runs the edit.
    pub resume_fell_back: Vec<(String, String)>,
    /// Reloaded functions whose dispatch-table slots were patched, so
    /// effect scopes already entered reach the edited implementation
    /// at their next perform.
    pub dispatch_patched: Vec<String>,
    /// True when a compile failure aborted the whole reload: nothing
    /// was swapped, published, or patched, and the running generation
    /// is untouched. `failed` holds the reasons. Per-function declines
    /// (unsupported shapes) do not abort — they are skips, and the
    /// rest of the edit set still applies.
    pub aborted: bool,
}

impl ReloadReport {
    /// True when the edit changed nothing that compiles differently.
    pub fn is_noop(&self) -> bool {
        self.reloaded.is_empty()
            && self.added.is_empty()
            && self.removed_retained.is_empty()
            && self.failed.is_empty()
    }
}
