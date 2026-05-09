//! # On-Stack Replacement (OSR) infrastructure
//!
//! Lets a long-running tier-0 (Cranelift baseline) function transfer its
//! live execution into newly-compiled tier-1 code at a loop header, instead
//! of waiting for the function to return.
//!
//! ## Architecture (3 layers)
//!
//! 1. **Runtime probe** (this module). A globally-registered C ABI function
//!    `osr_probe(bead_id, site) -> *mut ()` that JIT'd code calls at
//!    back-edges. It looks up the corresponding [`beadie::Bead`] and asks
//!    for an [`OsrEntry`] matching the encoded site key.
//!
//! 2. **Tier-0 codegen** (cranelift_backend, increment 3). At each HIR loop
//!    header, emit a sampling counter + probe call + indirect call to the
//!    helper if non-null. Cap the per-site cost at ~one cache-line worth of
//!    instructions amortized over 64 iterations.
//!
//! 3. **Tier-1 codegen** (cranelift_backend, increment 5). For each
//!    eligible loop header, emit a separate Cranelift function with
//!    signature `(i64, i64, i64, i64) -> i64`. Args are bit-cast as
//!    needed and used as the loop's live-in values; the helper jumps to
//!    the loop header in tier-1 code.
//!
//! ## Site key encoding
//!
//! `HirId` is a 128-bit UUID, so we can't pack it directly into 64 bits.
//! Instead, the site key uses a **block index local to the function** —
//! the 0-based position of the loop header in `HirFunction.blocks`:
//!
//! ```text
//!   bits 63..16: loop header block index (per-function, ≤ 2^48)
//!   bits 15..0 : live-in count (≤ 4 — codegen rejects larger layouts)
//! ```
//!
//! Both tier-0 (probe emitter) and tier-1 (helper emitter) walk
//! `HirFunction.blocks` in the same iteration order, so the index is a
//! stable identifier as long as the same `HirFunction` is being compiled
//! at both tiers.
//!
//! ## Bead identity
//!
//! Beadie's `Arc<Bead>` is the source of truth for a function's
//! compilation state. The registry maps a sequential `u64` `bead_id`
//! (assigned by [`next_bead_id`]) to its bead. JIT'd code embeds the
//! `bead_id` as a constant in the probe call.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use beadie::Bead;

use crate::hir::{HirFunction, HirId, HirTerminator, HirType};

// ─────────────────────────────────────────────────────────────────────────────
// Site-key encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of live-ins a single OSR helper accepts. Loops with more
/// live-ins than this are skipped at codegen time — the running tier-0 frame
/// just runs the loop to completion.
pub const OSR_MAX_LIVE_INS: usize = 4;

/// Pack `(loop_header_block_index, live_in_count)` into a 64-bit site key.
///
/// `block_index` is the 0-based position of the header inside
/// `HirFunction.blocks`. `live_in_count` must fit in 16 bits; in practice
/// it's ≤ [`OSR_MAX_LIVE_INS`].
#[inline]
pub fn encode_osr_site(block_index: u64, live_in_count: u16) -> u64 {
    (block_index << 16) | (live_in_count as u64)
}

/// Unpack a site key. Returns `(block_index, live_in_count)`.
#[inline]
pub fn decode_osr_site(site: u64) -> (u64, u16) {
    let block_index = site >> 16;
    let live_in_count = (site & 0xFFFF) as u16;
    (block_index, live_in_count)
}

/// Block-index lookup: returns the 0-based position of `block_id` inside
/// the function's block iteration order, or `None` if not present.
pub fn block_index_of(function: &HirFunction, block_id: HirId) -> Option<u64> {
    function
        .blocks
        .keys()
        .position(|id| *id == block_id)
        .map(|i| i as u64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bead registry
// ─────────────────────────────────────────────────────────────────────────────

/// Global registry mapping a `bead_id` (the `HirId` of the function's main
/// body) to its [`Arc<Bead>`]. Populated by `TieredBackend::compile_module`
/// at module load and consulted by [`osr_probe`] from JIT'd code on the
/// hot path.
///
/// Returning a `'static` reference to the inner `RwLock` keeps the call
/// site short — `bead_registry().read()...`. The registry is initialized
/// lazily on first access.
pub fn bead_registry() -> &'static RwLock<HashMap<u64, Arc<Bead>>> {
    static REGISTRY: OnceLock<RwLock<HashMap<u64, Arc<Bead>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register `bead` under `bead_id`. Idempotent: re-registering a different
/// bead under the same id replaces it (matches reload semantics).
pub fn register_bead(bead_id: u64, bead: Arc<Bead>) {
    bead_registry().write().unwrap().insert(bead_id, bead);
}

/// Drop a previously-registered bead. No-op if absent.
pub fn unregister_bead(bead_id: u64) {
    bead_registry().write().unwrap().remove(&bead_id);
}

/// Allocate a fresh sequential `bead_id`. Call once per registered
/// function; the value is stored alongside the function's metadata and
/// embedded as a constant into JIT'd probe call sites.
pub fn next_bead_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ─────────────────────────────────────────────────────────────────────────────
// Runtime probe (called from JIT'd code)
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime entry point for tier-0 back-edge probes.
///
/// JIT'd code emits an indirect call to this function (registered with the
/// Cranelift backend's symbol table). On the hot path:
/// 1. look up the bead by `bead_id` (one `RwLock` read + hash lookup)
/// 2. ask the bead for an OSR entry matching `site` (one O(log N) binary
///    search inside the bead's lock-free OSR table)
/// 3. return the helper pointer, or `null` if no match
///
/// Returning a raw `*mut ()` keeps the ABI shape simple for Cranelift —
/// the caller bit-tests for null before dispatching the indirect call.
///
/// # Safety
/// Called from generated code with C ABI. The runtime guarantees `bead_id`
/// values it ever passes correspond to either a live registry entry or a
/// stale one (the latter returns null cleanly).
#[no_mangle]
pub extern "C" fn osr_probe(bead_id: u64, site: u64) -> *mut () {
    if std::env::var_os("ZYNML_OSR_TRACE").is_some() {
        eprintln!("[osr_probe] bead_id={} site=0x{:x}", bead_id, site);
    }
    let registry = bead_registry().read().unwrap();
    match registry.get(&bead_id) {
        Some(bead) => bead.osr_entry(site).unwrap_or(std::ptr::null_mut()),
        None => std::ptr::null_mut(),
    }
}

/// Symbol name JIT'd code uses to reference the probe. Registered with the
/// Cranelift backend's symbol table at construction.
pub const OSR_PROBE_SYMBOL: &str = "__zyntax_osr_probe";

/// Symbol name for [`osr_sample_tick`].
pub const OSR_SAMPLE_TICK_SYMBOL: &str = "__zyntax_osr_sample_tick";

/// Sampling tick called at every back-edge probe site. Returns 0 when the
/// caller should run the (more expensive) probe lookup, non-zero
/// otherwise. Sampling rate is 1/64.
///
/// Encapsulating the counter inside an extern function keeps the JIT IR
/// short — no globals to declare, no atomic ops to emit inline. The
/// function-call overhead is acceptable in tier 0 (which exists to be
/// replaced); tier ≥ 1 doesn't emit this.
///
/// `Ordering::Relaxed` is sufficient: we only need rough sampling, not
/// strict ordering.
#[no_mangle]
pub extern "C" fn osr_sample_tick() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed) & 63
}

/// `(name, function_pointer)` pairs to feed
/// `CraneliftBackend::with_runtime_symbols` so JIT'd code can resolve
/// the OSR runtime functions at link time.
pub fn osr_runtime_symbols() -> [(&'static str, *const u8); 2] {
    [
        (OSR_PROBE_SYMBOL, osr_probe as *const u8),
        (OSR_SAMPLE_TICK_SYMBOL, osr_sample_tick as *const u8),
    ]
}

/// Backwards-compatible alias for callers that only care about the probe.
pub fn osr_probe_symbol() -> (&'static str, *const u8) {
    (OSR_PROBE_SYMBOL, osr_probe as *const u8)
}

// ─────────────────────────────────────────────────────────────────────────────
// HIR back-edge analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Loop-header HirIds for `function`, derived from a DFS over the HIR CFG.
///
/// A back-edge is a CFG edge `pred → header` where `header` is on the
/// current DFS stack (i.e. dominates `pred` in the DFS tree). The set of
/// distinct headers across all back-edges is the set of loop headers.
///
/// Whole-program HIR is per-function, so this returns headers from a single
/// function's blocks. The function's first block is the entry point.
///
/// Returns headers in deterministic discovery order so codegen output is
/// stable across runs.
pub fn find_loop_headers(function: &HirFunction) -> Vec<HirId> {
    if function.blocks.is_empty() {
        return Vec::new();
    }

    let entry = match function.blocks.keys().next() {
        Some(&id) => id,
        None => return Vec::new(),
    };

    let mut visited = HashMap::<HirId, DfsColor>::new();
    let mut headers = Vec::new();
    let mut headers_seen = std::collections::HashSet::new();

    dfs_find_back_edges(function, entry, &mut visited, &mut headers, &mut headers_seen);

    headers
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum DfsColor {
    OnStack,
    Done,
}

fn dfs_find_back_edges(
    function: &HirFunction,
    block_id: HirId,
    visited: &mut HashMap<HirId, DfsColor>,
    headers: &mut Vec<HirId>,
    headers_seen: &mut std::collections::HashSet<HirId>,
) {
    visited.insert(block_id, DfsColor::OnStack);

    let block = match function.blocks.get(&block_id) {
        Some(b) => b,
        None => {
            visited.insert(block_id, DfsColor::Done);
            return;
        }
    };

    for &succ in successors_of(&block.terminator).iter() {
        match visited.get(&succ).copied() {
            Some(DfsColor::OnStack) => {
                // Back-edge to a block currently on the DFS stack — `succ`
                // is a loop header.
                if headers_seen.insert(succ) {
                    headers.push(succ);
                }
            }
            Some(DfsColor::Done) => {
                // Forward / cross edge — not a loop.
            }
            None => {
                dfs_find_back_edges(function, succ, visited, headers, headers_seen);
            }
        }
    }

    visited.insert(block_id, DfsColor::Done);
}

/// Successors of a [`HirTerminator`], in source order.
fn successors_of(term: &HirTerminator) -> smallvec::SmallVec<[HirId; 4]> {
    let mut out = smallvec::SmallVec::new();
    match term {
        HirTerminator::Return { .. } | HirTerminator::Unreachable => {}
        HirTerminator::Branch { target } => out.push(*target),
        HirTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => {
            out.push(*true_target);
            out.push(*false_target);
        }
        HirTerminator::Switch { default, cases, .. } => {
            for (_, t) in cases {
                out.push(*t);
            }
            out.push(*default);
        }
        HirTerminator::Invoke { normal, unwind, .. } => {
            out.push(*normal);
            out.push(*unwind);
        }
        HirTerminator::PatternMatch { patterns, default, .. } => {
            for p in patterns {
                out.push(p.target);
            }
            if let Some(d) = default {
                out.push(*d);
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// OSR layout analysis
// ─────────────────────────────────────────────────────────────────────────────

/// What an OSR helper at a given header needs from the running tier-0 frame.
///
/// `live_ins` are the SSA values the helper expects as i64-typed function
/// args (in order); `live_in_types` records the original HIR type of each
/// for bit-cast at helper entry. `return_type` is the function's return
/// type — the helper returns it via i64 bit-cast and the caller reverses
/// the cast at the dispatch site.
///
/// Initial-cut policy (this increment): live-ins are the **phi results at
/// the loop header**. Loops with no phis have an empty live-in list (still
/// a valid layout — the helper takes zero args and runs the loop). Loops
/// whose body uses values defined outside the loop but not phi-merged at
/// the header are **not** representable yet; the helper would need to
/// re-execute the function prologue to materialize them, deferred to a
/// future increment.
#[derive(Debug, Clone)]
pub struct OsrLayout {
    pub header: HirId,
    pub block_index: u64,
    pub live_ins: Vec<HirId>,
    pub live_in_types: Vec<HirType>,
    pub return_type: HirType,
}

impl OsrLayout {
    /// Encoded site key for this layout — see [`encode_osr_site`].
    pub fn site_key(&self) -> u64 {
        encode_osr_site(self.block_index, self.live_ins.len() as u16)
    }
}

/// Reasons a layout is not representable as an OSR helper. Lets callers
/// log a useful diagnostic when `osr_layout` returns `None`.
#[derive(Debug, Clone, Copy)]
pub enum OsrReject {
    /// More than [`OSR_MAX_LIVE_INS`] live-ins.
    TooManyLiveIns(usize),
    /// A live-in's HIR type doesn't fit in i64.
    LiveInDoesntFit,
    /// Function return type doesn't fit in i64.
    ReturnDoesntFit,
    /// Header doesn't exist in the function.
    NoSuchHeader,
}

/// Compute an [`OsrLayout`] for `header`, or report why it's rejected.
pub fn osr_layout(function: &HirFunction, header: HirId) -> Result<OsrLayout, OsrReject> {
    let block = function
        .blocks
        .get(&header)
        .ok_or(OsrReject::NoSuchHeader)?;

    // Multi-value return functions can't go through the i64 helper ABI.
    let return_type = match function.signature.returns.as_slice() {
        [] => HirType::Void,
        [ty] => ty.clone(),
        _ => return Err(OsrReject::ReturnDoesntFit),
    };
    if !type_fits_i64(&return_type) {
        return Err(OsrReject::ReturnDoesntFit);
    }

    let live_ins: Vec<HirId> = block.phis.iter().map(|p| p.result).collect();
    let live_in_types: Vec<HirType> = block.phis.iter().map(|p| p.ty.clone()).collect();

    if live_ins.len() > OSR_MAX_LIVE_INS {
        return Err(OsrReject::TooManyLiveIns(live_ins.len()));
    }
    if !live_in_types.iter().all(type_fits_i64) {
        return Err(OsrReject::LiveInDoesntFit);
    }

    let block_index = block_index_of(function, header).unwrap_or(u64::MAX);

    Ok(OsrLayout {
        header,
        block_index,
        live_ins,
        live_in_types,
        return_type,
    })
}

/// Whether `ty` can round-trip through an i64 (the OSR helper ABI).
///
/// Scalars ≤ 64 bits and pointers always fit. Structs / arrays / unions
/// / vectors don't — they need stack passing. Void fits (no value to
/// transfer).
pub fn type_fits_i64(ty: &HirType) -> bool {
    matches!(
        ty,
        HirType::Void
            | HirType::Bool
            | HirType::I8
            | HirType::I16
            | HirType::I32
            | HirType::I64
            | HirType::U8
            | HirType::U16
            | HirType::U32
            | HirType::U64
            | HirType::F32
            | HirType::F64
            | HirType::Ptr(_)
            | HirType::Ref { .. }
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_key_roundtrips() {
        let cases = [
            (0u64, 0u16),
            (1, 1),
            (42, 4),
            (0xFFFF, 4),
            (0xFFFF_FFFF_FFFF, 3),
        ];
        for (block_idx, count) in cases {
            let site = encode_osr_site(block_idx, count);
            let (b, c) = decode_osr_site(site);
            assert_eq!(b, block_idx);
            assert_eq!(c, count);
        }
    }

    #[test]
    fn bead_ids_are_unique() {
        let a = next_bead_id();
        let b = next_bead_id();
        let c = next_bead_id();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn registry_get_and_remove() {
        // Registry is process-global — use a unique id to avoid clashes
        // between tests run in parallel.
        let id = u64::MAX - 42;
        // Drop any prior entry from a previous run before asserting.
        unregister_bead(id);
        // Probe must return null when no bead is registered.
        assert!(osr_probe(id, 0).is_null());
    }

    #[test]
    fn type_fits_i64_accepts_scalars_and_pointers() {
        assert!(type_fits_i64(&HirType::Void));
        assert!(type_fits_i64(&HirType::Bool));
        assert!(type_fits_i64(&HirType::I32));
        assert!(type_fits_i64(&HirType::I64));
        assert!(type_fits_i64(&HirType::F64));
        assert!(type_fits_i64(&HirType::Ptr(Box::new(HirType::I8))));
    }

    #[test]
    fn type_fits_i64_rejects_aggregates_and_wide() {
        assert!(!type_fits_i64(&HirType::I128));
        assert!(!type_fits_i64(&HirType::U128));
        assert!(!type_fits_i64(&HirType::Array(Box::new(HirType::I32), 4)));
        assert!(!type_fits_i64(&HirType::Struct(
            crate::hir::HirStructType {
                name: Some(zyntax_typed_ast::InternedString::new_global("Empty")),
                fields: vec![],
                packed: false,
            }
        )));
    }
}
