//! # On-Stack Replacement (OSR) infrastructure
//!
//! Lets a long-running tier-0 (Cranelift baseline) function transfer its
//! live execution into newly-compiled tier-1 code at a loop header, instead
//! of waiting for the function to return.
//!
//! ## Architecture (3 layers)
//!
//! 1. **Runtime registry** (this module). Each bead/site pair owns a stable
//!    helper slot. Baseline code reads it at a loop header; promotion fills
//!    it when the matching helper is ready.
//!
//! 2. **Tier-0 codegen** (cranelift_backend). Count loop-header visits in the
//!    running frame and request promotion after the hot threshold. Each
//!    header also loads its helper slot and transfers if one is installed.
//!
//! 3. **Promoted codegen** (cranelift_backend or llvm_jit_backend). Each
//!    eligible header gets a helper that reads live-ins from a frame and
//!    resumes the loop in promoted code.
//!
//! ## Site key encoding
//!
//! `HirId` is a 128-bit UUID, so we can't pack it directly into 64 bits.
//! Instead, the site key uses a **block index local to the function** —
//! the 0-based position of the loop header in `HirFunction.blocks`:
//!
//! ```text
//!   bits 63..16: loop header block index (per-function, ≤ 2^48)
//!   bits 15..0 : live-in count (bounded by [`OSR_MAX_LIVE_INS`])
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

/// Maximum number of live-ins a single OSR helper accepts. Loops carrying
/// more are skipped at codegen time and the running tier-0 frame just
/// finishes the loop itself.
///
/// They travel in a frame rather than registers, so the cost of a large
/// one is stack bytes and a copy taken once when the transfer happens, not
/// per iteration. Resuming at a deeply nested header legitimately needs
/// many: nbody's headers range from three to ninety-nine, because
/// everything computed before the resume point has to arrive with it.
pub const OSR_MAX_LIVE_INS: usize = 128;

/// Pack `(loop_ordinal, live_in_count)` into a 64-bit site key.
///
/// `loop_ordinal` is the 0-based position of the header among the
/// `HirFunction.blocks`. `live_in_count` must fit in 16 bits; in practice
/// it's ≤ [`OSR_MAX_LIVE_INS`].
#[inline]
pub fn encode_osr_site(loop_ordinal: u64, live_in_count: u16) -> u64 {
    (loop_ordinal << 16) | (live_in_count as u64)
}

/// Unpack a site key. Returns `(loop_ordinal, live_in_count)`.
#[inline]
pub fn decode_osr_site(site: u64) -> (u64, u16) {
    let loop_ordinal = site >> 16;
    let live_in_count = (site & 0xFFFF) as u16;
    (loop_ordinal, live_in_count)
}

/// Block-index lookup: returns the 0-based position of `block_id` inside
/// the function's block iteration order, or `None` if not present.
/// 0-based position of `header` among the function's loop headers, in
/// discovery order. Stable under edits elsewhere in the function, which
/// is what lets a site in running code match a helper compiled from an
/// edited body.
pub fn loop_ordinal_of(function: &HirFunction, header: HirId) -> Option<u64> {
    find_loop_headers(function)
        .iter()
        .position(|h| *h == header)
        .map(|i| i as u64)
}

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

/// Global registry mapping a `bead_id` (a sequential id from
/// [`next_bead_id`]) to its [`Arc<Bead>`]. Populated at module load by
/// `TieredBackend::compile_module` and by the interp-JIT install path in
/// `zyntax_embed`; read by [`osr_probe`].
///
/// Returning a `'static` reference to the inner `RwLock` keeps the call
/// site short — `bead_registry().read()...`. The registry is initialized
/// lazily on first access.
pub fn bead_registry() -> &'static RwLock<HashMap<u64, Arc<Bead>>> {
    static REGISTRY: OnceLock<RwLock<HashMap<u64, Arc<Bead>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register `bead` under `bead_id`. Re-registering under an existing id
/// replaces the previous entry.
pub fn register_bead(bead_id: u64, bead: Arc<Bead>) {
    bead_registry().write().unwrap().insert(bead_id, bead);
}

/// Drop a previously-registered bead. No-op if absent.
pub fn unregister_bead(bead_id: u64) {
    bead_registry().write().unwrap().remove(&bead_id);
    helper_slots()
        .write()
        .unwrap()
        .retain(|(b, _), _| *b != bead_id);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper slots
// ─────────────────────────────────────────────────────────────────────────────

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
#[unsafe(no_mangle)]
pub extern "C" fn osr_probe(bead_id: u64, site: u64) -> *mut () {
    // `std::env::var_os` calls `getenv`, which on macOS takes a
    // global lock per call (`os_unfair_lock` around the env
    // table). JIT'd code emits an `osr_probe` call at every
    // back-edge probe site — at rayzor-scale mandelbrot's
    // ~100 M loop iterations that's 100 M getenv calls and ~50 %
    // of total wall-clock. Resolve the trace gate once at process
    // start instead.
    if osr_trace_enabled() {
        eprintln!("[osr_probe] bead_id={} site=0x{:x}", bead_id, site);
    }
    let registry = bead_registry().read().unwrap();
    match registry.get(&bead_id) {
        Some(bead) => bead.osr_entry(site).unwrap_or(std::ptr::null_mut()),
        None => std::ptr::null_mut(),
    }
}

pub fn osr_trace_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("ZYNTAX_OSR_TRACE").is_some())
}

/// Symbol name JIT'd code uses to reference the probe. Registered with the
/// Cranelift backend's symbol table at construction.
pub const OSR_PROBE_SYMBOL: &str = "__zyntax_osr_probe";

/// `(name, function_pointer)` pairs to feed
/// `CraneliftBackend::with_runtime_symbols` so JIT'd code can resolve
/// the OSR runtime functions at link time.
pub fn osr_runtime_symbols() -> [(&'static str, *const u8); 4] {
    [
        (OSR_PROBE_SYMBOL, osr_probe as *const u8),
        (OSR_TRANSFER_SYMBOL, osr_transfer as *const u8),
        (OSR_REQUEST_SYMBOL, osr_request_promotion as *const u8),
        (LAZY_COMPILE_SYMBOL, lazy_compile as *const u8),
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

    dfs_find_back_edges(
        function,
        entry,
        &mut visited,
        &mut headers,
        &mut headers_seen,
    );

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
        HirTerminator::PatternMatch {
            patterns, default, ..
        } => {
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
/// `live_ins` are the SSA values the helper expects as function args. The
/// **first `phi_count`** entries are the phi results at the header (loop-
/// carried values, used as block params on the helper's jump to the
/// header); the rest are non-phi values defined outside the reachable
/// region but used inside it (e.g. function params or pre-loop locals).
/// All live-ins are received as i64s and bit-cast at helper entry per
/// `live_in_types`.
///
/// `return_type` is the function's return type — the helper returns it
/// directly (no bit-cast at the dispatch site since both sides share the
/// same Cranelift signature for the return).
#[derive(Debug, Clone)]
pub struct OsrLayout {
    pub header: HirId,
    /// Position of `header` among the function's loop headers — the
    /// stable half of the site key.
    pub loop_ordinal: u64,
    pub live_ins: Vec<HirId>,
    pub live_in_types: Vec<HirType>,
    /// Number of leading entries in `live_ins` that are phi results at
    /// `header`. `live_ins[..phi_count]` are passed to the header as
    /// block params; `live_ins[phi_count..]` are stored under their
    /// HirIds in the helper's value_map for the body to consume.
    pub phi_count: usize,
    pub return_type: HirType,
    /// Where each live-in sits in the frame the back-edge hands over.
    pub frame: OsrFrame,
    /// Live-ins the region defines again on its way back to the header
    /// (an enclosing loop's counter, say): the resumed code reads each
    /// as a phi at the header, made by [`resumable`], that merges the
    /// frame's value with the region's own.
    pub repairs: Vec<Repair>,
}

/// One live-in the resumed region redefines, and the phi that stands
/// for it at the header.
#[derive(Debug, Clone)]
pub struct Repair {
    /// The live-in as the frame carries it.
    pub value: HirId,
    /// The phi at the header the resumed code reads instead.
    pub phi: HirId,
    pub ty: HirType,
    /// The phi's incomings, one per predecessor of the header inside the
    /// region: the live-in itself where the region's definition reaches,
    /// the phi where the header's own does.
    pub incoming: Vec<(HirId, HirId)>,
    /// Blocks whose uses of the live-in read the phi: those the region's
    /// definition does not reach.
    pub rename_in: Vec<HirId>,
}

impl OsrLayout {
    /// Whether `id` is a phi at the header the resumed code enters
    /// through, so the frame's value for it becomes the phi's.
    pub fn enters_as_phi(&self, id: HirId) -> bool {
        self.repairs.iter().any(|r| r.value == id)
    }
}

impl OsrLayout {
    /// Encoded site key for this layout — see [`encode_osr_site`].
    pub fn site_key(&self) -> u64 {
        encode_osr_site(self.loop_ordinal, self.live_ins.len() as u16)
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
    /// The reachable region contains an instruction whose uses our
    /// conservative analysis can't enumerate (effects, atomics, trait
    /// method calls, etc.). Helper compile would mishandle it.
    UnsupportedInstruction,
    /// A nonterminal block in the resumed region can also be entered from
    /// outside it. Resuming there needs more than one entry point.
    RegionHasExternalEntry,
}

/// Compute an [`OsrLayout`] for `header`, or report why it's rejected.
///
/// Live-ins are computed in two passes:
///
/// 1. **Phi live-ins** — the phi results at `header`, in HIR order.
///    These are the loop-carried values; the helper's jump to the header
///    passes them as block params.
/// 2. **Non-phi live-ins** — values used by any instruction or
///    terminator in the region reachable from `header`, that are *not*
///    defined within that region and *not* HIR constants/globals
///    (which are rematerialized in the helper's prologue alongside the
///    main entry's prologue). These are typically function params or
///    pre-loop locals.
///
/// The combined live-in count must be ≤ [`OSR_MAX_LIVE_INS`].
pub fn osr_layout(function: &HirFunction, header: HirId) -> Result<OsrLayout, OsrReject> {
    osr_layout_with(function, header, &Dominators::compute(function))
}

/// [`osr_layout`] given the function's dominator tree, so a function
/// with many headers computes it once.
pub fn osr_layout_with(
    function: &HirFunction,
    header: HirId,
    dominators: &Dominators,
) -> Result<OsrLayout, OsrReject> {
    let block = function
        .blocks
        .get(&header)
        .ok_or(OsrReject::NoSuchHeader)?;

    // Multi-value return functions can't go through the helper ABI.
    let return_type = match function.signature.returns.as_slice() {
        [] => HirType::Void,
        [ty] => ty.clone(),
        _ => return Err(OsrReject::ReturnDoesntFit),
    };
    if !type_fits_i64(&return_type) {
        return Err(OsrReject::ReturnDoesntFit);
    }

    // Phi live-ins.
    let mut live_ins: Vec<HirId> = block.phis.iter().map(|p| p.result).collect();
    let mut live_in_types: Vec<HirType> = block.phis.iter().map(|p| p.ty.clone()).collect();
    let phi_count = live_ins.len();

    // Non-phi live-ins. Walk reachable blocks, collect uses minus
    // locally-defined-or-rematerializable values.
    let reachable = reachable_from(function, header);
    let in_region: IdSet = reachable.iter().copied().collect();
    // Dominance inside the region, entered at the header alone: the
    // helper's view of the control flow.
    let region_dom = region_dominators(function, header, &in_region);
    // Only values defined in blocks the header dominates are guaranteed to
    // have been computed by the time the resumed code reads them. Anything
    // else (an enclosing loop's counter, say) must arrive in the frame.
    // What the region defines is local to it; a definition the header
    // does not dominate is live at the header when read outside its own
    // dominance, and is repaired below.
    let local_defs = locally_defined_in(function, &reachable);
    let dominated: IdSet = dominators.dominated_by(header).into_iter().collect();

    let mut seen_extra: IdSet = live_ins.iter().copied().collect();

    for &block_id in &reachable {
        let block = match function.blocks.get(&block_id) {
            Some(b) => b,
            None => continue,
        };
        for inst in &block.instructions {
            // Instruction outside the supported subset: reject the
            // layout. The helper compile would mishandle it.
            if !layout_supports(inst) {
                return Err(OsrReject::UnsupportedInstruction);
            }
            inst.for_each_operand(|used| {
                consider_live_in(
                    function,
                    used,
                    &local_defs,
                    &mut seen_extra,
                    &mut live_ins,
                    &mut live_in_types,
                );
            });
        }
        for used in terminator_uses(&block.terminator) {
            consider_live_in(
                function,
                used,
                &local_defs,
                &mut seen_extra,
                &mut live_ins,
                &mut live_in_types,
            );
        }
        // A phi inside the region can name a value produced before it. The
        // header's own phis are the exception: their incoming values are
        // precisely what resuming replaces, and pulling them in would ask
        // the frame for the entry values it exists to supersede.
        if block_id != header {
            for phi in &block.phis {
                for (value_id, predecessor) in &phi.incoming {
                    if !in_region.contains(predecessor) {
                        continue;
                    }
                    consider_live_in(
                        function,
                        *value_id,
                        &local_defs,
                        &mut seen_extra,
                        &mut live_ins,
                        &mut live_in_types,
                    );
                }
            }
        }
    }

    // A block in the region entered from outside it is on a path the
    // resumed code never takes (an enclosing loop's header, reached from
    // before the loop). What it defines and the loop body reads is live
    // at the header, arrives in the frame, and is defined again when the
    // block runs; the resumed code reads a phi at the header merging the
    // two, so every read must be reached by one of them alone.
    let header_preds: Vec<HirId> = reachable
        .iter()
        .copied()
        .filter(|b| {
            function
                .blocks
                .get(b)
                .is_some_and(|block| successors_of(&block.terminator).contains(&header))
        })
        .collect();
    let mut repairs = Vec::new();
    for &def_block in &reachable {
        // Entered from outside the region: not on every path to the
        // header in the function itself.
        if dominated.contains(&def_block) {
            continue;
        }
        let Some(block) = function.blocks.get(&def_block) else {
            continue;
        };
        let defs: Vec<(HirId, HirType)> = block
            .phis
            .iter()
            .map(|p| (p.result, p.ty.clone()))
            .chain(block.instructions.iter().filter_map(|i| {
                instruction_result(i)
                    .and_then(|r| function.values.get(&r).map(|v| (r, v.ty.clone())))
            }))
            .collect();
        for (value, ty) in defs {
            let reached_by_def =
                |b: &HirId| region_dom.get(b).is_some_and(|d| d.contains(&def_block));
            // A block the definition does not dominate reads the header's
            // phi. That is the value there only if every path from the
            // definition to the block passes the header, where the phi
            // merges again; a block the definition reaches around the
            // header would need a phi of its own.
            let mut around: IdSet = IdSet::default();
            {
                let mut stack: Vec<HirId> = vec![def_block];
                while let Some(b) = stack.pop() {
                    let Some(bb) = function.blocks.get(&b) else {
                        continue;
                    };
                    for succ in successors_of(&bb.terminator) {
                        if succ != header && in_region.contains(&succ) && around.insert(succ) {
                            stack.push(succ);
                        }
                    }
                }
            }
            let mixed = |b: &HirId| !reached_by_def(b) && around.contains(b);
            let mut escapes = false;
            for &b in &reachable {
                let Some(rb) = function.blocks.get(&b) else {
                    continue;
                };
                let mut read = false;
                for inst in &rb.instructions {
                    inst.for_each_operand(|u| read |= u == value);
                }
                read |= terminator_uses(&rb.terminator).contains(&value);
                if read && !reached_by_def(&b) {
                    if mixed(&b) {
                        return Err(OsrReject::RegionHasExternalEntry);
                    }
                    escapes = true;
                }
                for p in &rb.phis {
                    for (v, pred) in &p.incoming {
                        if *v == value && in_region.contains(pred) && !reached_by_def(pred) {
                            if mixed(pred) {
                                return Err(OsrReject::RegionHasExternalEntry);
                            }
                            escapes = true;
                        }
                    }
                }
            }
            if !escapes {
                continue;
            }
            let phi = HirId::new();
            let mut incoming = Vec::with_capacity(header_preds.len());
            for &pred in &header_preds {
                if reached_by_def(&pred) {
                    incoming.push((value, pred));
                } else if mixed(&pred) {
                    return Err(OsrReject::RegionHasExternalEntry);
                } else {
                    incoming.push((phi, pred));
                }
            }
            if !live_ins.contains(&value) {
                live_ins.push(value);
                live_in_types.push(ty.clone());
            }
            let rename_in: Vec<HirId> = reachable
                .iter()
                .copied()
                .filter(|b| !reached_by_def(b))
                .collect();
            if osr_trace_enabled() {
                eprintln!(
                    "[osr] {} header {}: live-in {:?} defined in block {} is read again after the header; renamed in {} blocks",
                    function.name.resolve_global().unwrap_or_default(),
                    block_index_of(function, header).unwrap_or(u64::MAX),
                    value,
                    block_index_of(function, def_block).unwrap_or(u64::MAX),
                    rename_in.len()
                );
            }
            repairs.push(Repair {
                value,
                phi,
                ty,
                incoming,
                rename_in,
            });
        }
    }

    if live_ins.len() > OSR_MAX_LIVE_INS {
        return Err(OsrReject::TooManyLiveIns(live_ins.len()));
    }

    // Live-ins travel through a frame rather than registers, so a type only
    // has to have a layout — an aggregate is copied, not squeezed into a
    // slot. A type with no known size still has nowhere to live.
    if live_in_types.iter().any(|t| frame_size_of(t) == 0) {
        return Err(OsrReject::LiveInDoesntFit);
    }

    let loop_ordinal = loop_ordinal_of(function, header).unwrap_or(u64::MAX);
    let frame = OsrFrame::for_types(&live_in_types);

    Ok(OsrLayout {
        header,
        loop_ordinal,
        live_ins,
        live_in_types,
        phi_count,
        return_type,
        frame,
        repairs,
    })
}

/// The dominators of each block of `region`, with `header` as the only
/// entry: edges from outside the region do not count.
fn region_dominators(
    function: &HirFunction,
    header: HirId,
    region: &IdSet,
) -> HashMap<HirId, IdSet> {
    let mut preds: HashMap<HirId, Vec<HirId>> = HashMap::new();
    for &b in region {
        if let Some(block) = function.blocks.get(&b) {
            for succ in successors_of(&block.terminator) {
                if region.contains(&succ) && succ != header {
                    preds.entry(succ).or_default().push(b);
                }
            }
        }
    }
    let all: IdSet = region.iter().copied().collect();
    let mut dom: HashMap<HirId, IdSet> = region
        .iter()
        .map(|&b| {
            if b == header {
                (b, std::iter::once(b).collect())
            } else {
                (b, all.clone())
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &b in region {
            if b == header {
                continue;
            }
            let mut next: Option<IdSet> = None;
            for p in preds.get(&b).map(|v| v.as_slice()).unwrap_or(&[]) {
                let pd = &dom[p];
                next = Some(match next {
                    None => pd.clone(),
                    Some(acc) => acc.intersection(pd).copied().collect(),
                });
            }
            let mut next = next.unwrap_or_default();
            next.insert(b);
            if next != dom[&b] {
                dom.insert(b, next);
                changed = true;
            }
        }
    }
    dom
}

/// `function` as a helper resumes it at `layout.header`: each repaired
/// live-in becomes a phi at the header, appended in live-in order, and
/// the blocks the header dominates read the phi.
pub fn resumable(function: &HirFunction, layout: &OsrLayout) -> HirFunction {
    if layout.repairs.is_empty() {
        return function.clone();
    }
    let mut f = function.clone();
    for r in &layout.repairs {
        f.values.insert(
            r.phi,
            crate::hir::HirValue {
                id: r.phi,
                ty: r.ty.clone(),
                kind: crate::hir::HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        let map: indexmap::IndexMap<HirId, HirId> = std::iter::once((r.value, r.phi)).collect();
        let renamed: IdSet = r.rename_in.iter().copied().collect();
        for (b, block) in f.blocks.iter_mut() {
            if renamed.contains(b) {
                for inst in &mut block.instructions {
                    inst.replace_uses(&map);
                }
                block.terminator.replace_uses(&map);
            }
            // A phi reads by the edge it comes in on.
            for p in &mut block.phis {
                for (v, pred) in &mut p.incoming {
                    if *v == r.value && renamed.contains(pred) {
                        *v = r.phi;
                    }
                }
            }
        }
        if let Some(header) = f.blocks.get_mut(&layout.header) {
            header.phis.push(crate::hir::HirPhi {
                result: r.phi,
                ty: r.ty.clone(),
                incoming: r.incoming.clone(),
            });
        }
    }
    f
}

/// Add `used` to the live-ins list iff it's used in the loop body but
/// not locally defined and not a constant/undef/global (those are
/// rematerialized in the helper's prologue, not passed as args).
/// A set of ids hashed by the id itself: these sets are built and
/// probed once per instruction of a function, per header.
type IdSet = std::collections::HashSet<HirId, std::hash::BuildHasherDefault<fnv::FnvHasher>>;

fn consider_live_in(
    function: &HirFunction,
    used: HirId,
    local_defs: &IdSet,
    seen: &mut IdSet,
    live_ins: &mut Vec<HirId>,
    live_in_types: &mut Vec<HirType>,
) {
    if !seen.insert(used) {
        return;
    }
    if local_defs.contains(&used) {
        return;
    }
    if let Some(value) = function.values.get(&used) {
        match &value.kind {
            crate::hir::HirValueKind::Constant(_)
            | crate::hir::HirValueKind::Undef
            | crate::hir::HirValueKind::Global(_) => return,
            _ => {
                live_ins.push(used);
                live_in_types.push(value.ty.clone());
            }
        }
    }
    // Unknown HirId — silently drop. The helper compile will fail later
    // if this turns out to be a real reference; the caller's error
    // reporting catches it.
}

fn reachable_from(function: &HirFunction, start: HirId) -> Vec<HirId> {
    let mut order = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![start];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        order.push(id);
        if let Some(block) = function.blocks.get(&id) {
            for succ in successors_of(&block.terminator) {
                if !visited.contains(&succ) {
                    stack.push(succ);
                }
            }
        }
    }
    order
}

fn locally_defined_in(function: &HirFunction, blocks: &[HirId]) -> IdSet {
    let mut defs = IdSet::default();
    for &id in blocks {
        let block = match function.blocks.get(&id) {
            Some(b) => b,
            None => continue,
        };
        for phi in &block.phis {
            defs.insert(phi.result);
        }
        for inst in &block.instructions {
            if let Some(result) = instruction_result(inst) {
                defs.insert(result);
            }
        }
    }
    defs
}

fn instruction_result(inst: &crate::hir::HirInstruction) -> Option<HirId> {
    use crate::hir::HirInstruction as I;
    match inst {
        I::Binary { result, .. }
        | I::Unary { result, .. }
        | I::Alloca { result, .. }
        | I::Load { result, .. }
        | I::Cast { result, .. }
        | I::GetElementPtr { result, .. }
        | I::Select { result, .. }
        | I::ExtractValue { result, .. }
        | I::InsertValue { result, .. }
        | I::VectorSplat { result, .. }
        | I::VectorExtractLane { result, .. }
        | I::VectorInsertLane { result, .. }
        | I::VectorHorizontalReduce { result, .. }
        | I::VectorLoad { result, .. }
        | I::VectorUnaryOp { result, .. }
        | I::VectorMinMax { result, .. } => Some(*result),
        I::Call { result, .. }
        | I::IndirectCall { result, .. }
        | I::CallClosure { result, .. }
        | I::TraitMethodCall { result, .. } => *result,
        _ => None,
    }
}

/// Conservatively enumerate the HirIds an instruction reads. This must
/// over-approximate (missing a use is unsafe — the helper compile would
/// later fail with an unmapped HirId), so we err on the side of
/// rejecting layouts that contain instructions we don't fully understand.
///
/// `Ok(uses)` — confidently enumerated uses.
/// `Err(())` — the instruction is outside our supported subset; the
/// caller should reject the layout.
/// The kinds a helper body is known to lower; their operands are what
/// the instruction itself reports, so a use is never missed by naming
/// fields here. Anything else (effects, atomics, trait method calls,
/// fences, ...) rejects the layout rather than risk a use going unseen.
fn layout_supports(inst: &crate::hir::HirInstruction) -> bool {
    use crate::hir::HirInstruction as I;
    matches!(
        inst,
        I::Binary { .. }
            | I::Unary { .. }
            | I::Alloca { .. }
            | I::Load { .. }
            | I::Store { .. }
            | I::GetElementPtr { .. }
            | I::Cast { .. }
            | I::Select { .. }
            | I::ExtractValue { .. }
            | I::InsertValue { .. }
            | I::Call { .. }
            | I::CallClosure { .. }
            | I::VectorSplat { .. }
            | I::VectorExtractLane { .. }
            | I::VectorInsertLane { .. }
            | I::VectorHorizontalReduce { .. }
            | I::VectorLoad { .. }
            | I::VectorStore { .. }
            | I::VectorUnaryOp { .. }
            | I::VectorMinMax { .. }
    )
}

fn terminator_uses(term: &HirTerminator) -> Vec<HirId> {
    let mut uses = Vec::new();
    match term {
        HirTerminator::Return { values } => uses.extend(values.iter().copied()),
        HirTerminator::CondBranch { condition, .. } => uses.push(*condition),
        HirTerminator::Switch { value, .. } => uses.push(*value),
        HirTerminator::Invoke { args, .. } => uses.extend(args.iter().copied()),
        HirTerminator::PatternMatch { value, .. } => uses.push(*value),
        _ => {}
    }
    uses
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

    /// An unreachable block must not constrain the blocks that list it
    /// as a predecessor.
    ///
    /// The shape is the one a nested loop with an early exit produces:
    /// a latch whose predecessors are a live block and an orphan the
    /// CFG left with no way in. Treating the orphan as dominated only
    /// by itself empties the intersection at the latch, and the loop
    /// header stops dominating its own latch — which reads the back
    /// edge as a forward edge and costs the loop its carried values.
    #[test]
    fn an_unreachable_predecessor_does_not_break_dominance() {
        use crate::hir::{HirBlock, HirFunctionSignature, HirTerminator};
        use zyntax_typed_ast::InternedString;

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
        let mut function = HirFunction::new(InternedString::new_global("f"), signature);
        let entry = function.entry_block;
        function.blocks.clear();

        let header = HirId::new();
        let body = HirId::new();
        let latch = HirId::new();
        let orphan = HirId::new();

        let block = |id: HirId, preds: Vec<HirId>| HirBlock {
            id,
            label: None,
            phis: Vec::new(),
            instructions: Vec::new(),
            terminator: HirTerminator::Unreachable,
            dominance_frontier: Default::default(),
            predecessors: preds,
            successors: Vec::new(),
        };

        // entry → header → body → latch → header, plus an orphan edge
        // into the latch from a block nothing reaches.
        function.blocks.insert(entry, block(entry, vec![]));
        function
            .blocks
            .insert(header, block(header, vec![entry, latch]));
        function.blocks.insert(body, block(body, vec![header]));
        function
            .blocks
            .insert(latch, block(latch, vec![body, orphan]));
        function.blocks.insert(orphan, block(orphan, vec![]));

        let dominated = blocks_dominated_by(&function, header);
        assert!(
            dominated.contains(&latch),
            "the header dominates its latch even when an unreachable block \
             also names the latch as a successor; got {dominated:?}"
        );
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
    fn rejected_promotion_can_be_requested_again() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let id = u64::MAX - 43;
        let attempts = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&attempts);
        set_promotion_requester(move |bead| {
            assert_eq!(bead, id);
            seen.fetch_add(1, Ordering::Relaxed) > 0
        });
        osr_request_promotion(id);
        osr_request_promotion(id);
        osr_request_promotion(id);
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
        requested().write().unwrap().remove(&id);
        set_promotion_requester(|_| false);
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

/// Source-location value used to mark a probe site so codegen can find
/// where it landed.
///
/// `MachSrcLoc` is the only post-codegen mapping from an instruction back
/// to a code offset, so probe emission borrows it as a tag rather than a
/// position. Bit 31 distinguishes ours from a real source offset.
pub fn probe_srcloc_tag(site_key: u64) -> u32 {
    0x8000_0000 | (site_key as u32 & 0x7FFF_FFFF)
}

/// One pointer per `(bead_id, site_key)`, holding the OSR helper for that
/// site or null. Generated code loads it directly at the back-edge.
///
/// Storing the helper rather than a flag removes the runtime lookup from
/// the loop: an armed site branches straight to the helper instead of
/// calling `osr_probe` to find it, so the loop contains no call that
/// returns to it.
fn helper_slots() -> &'static RwLock<HashMap<(u64, u64), Box<AtomicU64>>> {
    static SLOTS: OnceLock<RwLock<HashMap<(u64, u64), Box<AtomicU64>>>> = OnceLock::new();
    SLOTS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Address of the helper slot for `(bead_id, site_key)`, allocating it on
/// first call. Stable for as long as the bead is registered.
pub fn helper_slot_addr(bead_id: u64, site_key: u64) -> *const u8 {
    let mut slots = helper_slots().write().unwrap();
    let slot = slots
        .entry((bead_id, site_key))
        .or_insert_with(|| Box::new(AtomicU64::new(0)));
    (&**slot) as *const AtomicU64 as *const u8
}

/// Publish `helper` for `(bead_id, site_key)`, so back-edges start
/// transferring into it.
pub fn publish_helper(bead_id: u64, site_key: u64, helper: *mut ()) {
    if osr_trace_enabled() {
        eprintln!("[osr] publish bead={bead_id} site=0x{site_key:x} -> {helper:?}");
    }
    let mut slots = helper_slots().write().unwrap();
    slots
        .entry((bead_id, site_key))
        .or_insert_with(|| Box::new(AtomicU64::new(0)))
        .store(helper as u64, Ordering::Release);
}

/// Helper currently published for `(bead_id, site_key)`, or null. Reads
/// what generated code reads.
pub fn helper_for(bead_id: u64, site_key: u64) -> *mut () {
    helper_slots()
        .read()
        .unwrap()
        .get(&(bead_id, site_key))
        .map(|s| s.load(Ordering::Acquire) as *mut ())
        .unwrap_or(std::ptr::null_mut())
}

/// Blocks reachable from `start` by following successors, including
/// `start` itself.
///
/// An OSR helper resumes at a loop header, so this is the region it can
/// still execute — everything else in the function was only on the path
/// leading up to the header.
pub fn blocks_reachable_from(
    function: &HirFunction,
    start: HirId,
) -> std::collections::HashSet<HirId> {
    // Walks terminators rather than `HirBlock::successors`. That cached
    // field is not maintained by the passes that run before codegen — on
    // optimized nbody it reports one successor for a ten-block function —
    // so anything reading it sees almost no CFG at all.
    reachable_from(function, start).into_iter().collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame layout
// ─────────────────────────────────────────────────────────────────────────────

/// Byte size of `ty` as laid out in an OSR frame.
///
/// Fields are placed at their natural alignment and the struct is padded to
/// its own alignment, which is what both backends already assume of an
/// in-memory aggregate. Frames are written by one backend and read by the
/// other, so the two must agree byte for byte.
pub fn frame_size_of(ty: &HirType) -> usize {
    if is_held_by_reference(ty) {
        return 8;
    }
    match ty {
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 | HirType::Ptr(_) => 8,
        HirType::I128 | HirType::U128 => 16,
        HirType::Struct(s) => {
            let mut size = 0usize;
            for field in &s.fields {
                let a = frame_align_of(field);
                size = size.div_ceil(a) * a;
                size += frame_size_of(field);
            }
            let a = frame_align_of(ty);
            size.div_ceil(a) * a
        }
        HirType::Vector(elem, n) => frame_size_of(elem).saturating_mul(*n as usize),
        _ => 8,
    }
}

/// Alignment of `ty` in an OSR frame.
pub fn frame_align_of(ty: &HirType) -> usize {
    if is_held_by_reference(ty) {
        return 8;
    }
    match ty {
        HirType::Struct(s) => s.fields.iter().map(frame_align_of).max().unwrap_or(1),
        HirType::Vector(elem, _) => frame_align_of(elem),
        other => frame_size_of(other).min(16).max(1),
    }
}

/// Whether a value of this type is held as a pointer to its storage rather
/// than as the value itself.
///
/// Such a value travels through the frame as that pointer, never as a
/// copy of what it points at: the storage may be shared (a list header,
/// an object, a stack slot other live-ins address), and the resumed code
/// must keep writing where everything else reads. A backend that holds
/// the struct by value loads it through the pointer on its side.
pub fn is_held_by_reference(ty: &HirType) -> bool {
    match ty {
        HirType::Struct(s) => s.fields.len() > 1,
        HirType::Array(_, _) => true,
        _ => false,
    }
}

/// Where each live-in sits in the frame a back-edge hands to a helper.
///
/// Passing live-ins as arguments forced every one to fit a register, and
/// the backends do not agree on how to hold an aggregate in one: a
/// multi-field struct is a pointer in Cranelift and a value in LLVM. The
/// frame holds scalars as themselves and aggregates as the pointer to
/// their storage; a reader that wants the value loads it from there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OsrFrame {
    /// Byte offset of each live-in, parallel to `OsrLayout::live_ins`.
    pub offsets: Vec<u32>,
    /// Total frame size in bytes.
    pub size: u32,
    /// Alignment the frame must be allocated at.
    pub align: u32,
}

impl OsrFrame {
    /// Lay out `types` in order, each at its natural alignment.
    pub fn for_types(types: &[HirType]) -> Self {
        let mut offsets = Vec::with_capacity(types.len());
        let mut cursor = 0usize;
        let mut align = 1usize;
        for ty in types {
            let a = frame_align_of(ty);
            align = align.max(a);
            cursor = cursor.div_ceil(a) * a;
            offsets.push(cursor as u32);
            cursor += frame_size_of(ty);
        }
        let size = cursor.div_ceil(align.max(1)) * align.max(1);
        Self {
            offsets,
            size: size as u32,
            align: align as u32,
        }
    }
}

/// Blocks that `header` dominates: every path to them from the function
/// entry passes through it.
///
/// A helper resumes at `header`, so only these are guaranteed to have run
/// by the time their values are used. A block that is merely *reachable*
/// from the header may also be reached without it (an enclosing loop's
/// header is the common case) and anything it defines has to arrive as a
/// live-in instead.
pub fn blocks_dominated_by(
    function: &HirFunction,
    header: HirId,
) -> std::collections::HashSet<HirId> {
    Dominators::compute(function).dominated_by(header)
}

/// The dominator tree of a function, over the blocks' predecessor
/// lists: each reachable block's immediate dominator, found by the
/// iterative algorithm over a reverse postorder, which is linear in
/// the blocks for the CFGs here.
///
/// A block nothing reaches has no dominator of its own and counts as
/// dominated by every block, so an edge from one never reads as a
/// forward edge into a loop.
pub struct Dominators {
    /// Reachable blocks in reverse postorder.
    order: Vec<HirId>,
    /// Position in `order` of each reachable block.
    index: HashMap<HirId, usize>,
    /// Immediate dominator of each block in `order`, by position; the
    /// entry is its own.
    idom: Vec<usize>,
    /// Every block of the function, for the unreachable ones.
    all: Vec<HirId>,
}

impl Dominators {
    pub fn compute(function: &HirFunction) -> Self {
        let all: Vec<HirId> = function.blocks.keys().copied().collect();
        let Some(&entry) = all.first() else {
            return Dominators {
                order: Vec::new(),
                index: HashMap::new(),
                idom: Vec::new(),
                all,
            };
        };
        let mut successors: HashMap<HirId, Vec<HirId>> = HashMap::new();
        for (&b, block) in &function.blocks {
            for &p in &block.predecessors {
                successors.entry(p).or_default().push(b);
            }
        }
        // Postorder from the entry, then reversed.
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut stack: Vec<(HirId, usize)> = vec![(entry, 0)];
        visited.insert(entry);
        while let Some((b, next)) = stack.last_mut() {
            let succ = successors.get(b).map(|s| s.as_slice()).unwrap_or(&[]);
            if *next < succ.len() {
                let s = succ[*next];
                *next += 1;
                if visited.insert(s) {
                    stack.push((s, 0));
                }
            } else {
                order.push(*b);
                stack.pop();
            }
        }
        order.reverse();
        let index: HashMap<HirId, usize> = order.iter().enumerate().map(|(i, b)| (*b, i)).collect();
        let preds: Vec<Vec<usize>> = order
            .iter()
            .map(|b| {
                function.blocks[b]
                    .predecessors
                    .iter()
                    .filter_map(|p| index.get(p).copied())
                    .collect()
            })
            .collect();
        let mut idom = vec![usize::MAX; order.len()];
        idom[0] = 0;
        let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
            while a != b {
                while a > b {
                    a = idom[a];
                }
                while b > a {
                    b = idom[b];
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for b in 1..order.len() {
                let mut new = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    new = if new == usize::MAX {
                        p
                    } else {
                        intersect(&idom, p, new)
                    };
                }
                if new != usize::MAX && idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        Dominators {
            order,
            index,
            idom,
            all,
        }
    }

    /// Whether `a` dominates `b`.
    pub fn dominates(&self, a: HirId, b: HirId) -> bool {
        let Some(&a) = self.index.get(&a) else {
            return false;
        };
        let Some(&b) = self.index.get(&b) else {
            // Unreachable: dominated by everything.
            return true;
        };
        let mut b = b;
        loop {
            if b == a {
                return true;
            }
            if b == 0 {
                return false;
            }
            b = self.idom[b];
        }
    }

    /// Every block `header` dominates, itself included.
    pub fn dominated_by(&self, header: HirId) -> std::collections::HashSet<HirId> {
        self.all
            .iter()
            .copied()
            .filter(|b| self.dominates(header, *b))
            .collect()
    }
}

/// Symbol the dispatch path calls once per transfer when tracing is on.
pub const OSR_TRANSFER_SYMBOL: &str = "__zyntax_osr_transfer";

/// Counts transfers per site. Generated code calls this from the dispatch
/// path, which runs exactly when a frame moves into a helper — the one
/// place a transfer can be observed from outside a test.
#[unsafe(no_mangle)]
pub extern "C" fn osr_transfer(site: u64, helper: u64) {
    static COUNTS: OnceLock<RwLock<HashMap<u64, u64>>> = OnceLock::new();
    let counts = COUNTS.get_or_init(|| RwLock::new(HashMap::new()));
    let mut guard = counts.write().unwrap();
    let n = guard.entry(site).or_insert(0);
    *n += 1;
    if *n == 1 {
        let tier = if is_llvm_helper(helper as usize) {
            "llvm"
        } else {
            "cranelift"
        };
        eprintln!("[osr] FIRST TRANSFER at site 0x{site:x} -> {helper:#x} ({tier})");
    }
}

fn llvm_helpers() -> &'static RwLock<std::collections::HashSet<usize>> {
    static H: OnceLock<RwLock<std::collections::HashSet<usize>>> = OnceLock::new();
    H.get_or_init(|| RwLock::new(std::collections::HashSet::new()))
}

/// Record that `addr` is a helper the LLVM tier produced.
///
/// A transfer only reports which tier it landed in if the address can be
/// attributed, and the address is all the dispatch path has.
pub fn note_llvm_helper(addr: usize) {
    llvm_helpers().write().unwrap().insert(addr);
}

/// Whether `addr` was published by the LLVM tier.
pub fn is_llvm_helper(addr: usize) -> bool {
    llvm_helpers().read().unwrap().contains(&addr)
}

// ─────────────────────────────────────────────────────────────────────────────
// Lazy compilation
// ─────────────────────────────────────────────────────────────────────────────

/// Symbol the stub standing in for a function not yet compiled calls:
/// `(bead_id) -> entry`. The stub then calls the entry with its own
/// arguments.
pub const LAZY_COMPILE_SYMBOL: &str = "__zyntax_lazy_compile";

/// Installed by the runtime: compile the function behind a bead now and
/// hand back its entry, publishing it wherever the stub was.
type LazyCompiler = Box<dyn Fn(u64) -> *const u8 + Send + Sync>;

fn lazy_compiler() -> &'static RwLock<Option<LazyCompiler>> {
    static R: OnceLock<RwLock<Option<LazyCompiler>>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(None))
}

/// Register how a function left uncompiled is compiled on its first call.
pub fn set_lazy_compiler(f: impl Fn(u64) -> *const u8 + Send + Sync + 'static) {
    *lazy_compiler().write().unwrap() = Some(Box::new(f));
}

/// Called by a stub on the first call of the function it stands for.
/// The runtime compiles the function and publishes its entry; the stub
/// calls what comes back. With no compiler installed the process
/// cannot continue, since the stub has nothing to call.
///
/// # Safety
/// Called from generated code with C ABI.
#[unsafe(no_mangle)]
pub extern "C" fn lazy_compile(bead_id: u64) -> *const u8 {
    let guard = lazy_compiler().read().unwrap();
    let Some(f) = guard.as_ref() else {
        eprintln!(
            "a function compiled on first call was called before the runtime could compile it (bead {bead_id})"
        );
        std::process::abort();
    };
    let entry = f(bead_id);
    if entry.is_null() {
        eprintln!("a function compiled on first call could not be compiled (bead {bead_id})");
        std::process::abort();
    }
    entry
}

// ─────────────────────────────────────────────────────────────────────────────
// Promotion requests
// ─────────────────────────────────────────────────────────────────────────────

/// Symbol a tier-0 function calls once its loop has stayed hot.
pub const OSR_REQUEST_SYMBOL: &str = "__zyntax_osr_request";

/// Installed by the runtime to queue a top-tier compile for a bead.
type PromotionRequester = Box<dyn Fn(u64) -> bool + Send + Sync>;

fn promotion_requester() -> &'static RwLock<Option<PromotionRequester>> {
    static R: OnceLock<RwLock<Option<PromotionRequester>>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(None))
}

/// Register how a promotion request is fulfilled. The runtime owns the
/// policy — whether to queue, and to which tier.
pub fn set_promotion_requester(f: impl Fn(u64) -> bool + Send + Sync + 'static) {
    *promotion_requester().write().unwrap() = Some(Box::new(f));
}

/// Beads that have already asked, so a function called repeatedly does not
/// queue the same compile over and over.
fn requested() -> &'static RwLock<std::collections::HashSet<u64>> {
    static S: OnceLock<RwLock<std::collections::HashSet<u64>>> = OnceLock::new();
    S.get_or_init(|| RwLock::new(std::collections::HashSet::new()))
}

/// Called when a tier-0 frame has revisited a resumable loop enough times
/// to justify a background compile. Invocation counts alone cannot promote
/// a function that remains in one long-running call.
///
/// # Safety
/// Called from generated code with C ABI.
#[unsafe(no_mangle)]
pub extern "C" fn osr_request_promotion(bead_id: u64) {
    if !requested().write().unwrap().insert(bead_id) {
        return;
    }
    if osr_trace_enabled() {
        eprintln!("[osr] promotion requested for bead={bead_id}");
    }
    let guard = promotion_requester().read().unwrap();
    let submitted = guard.as_ref().is_some_and(|f| f(bead_id));
    drop(guard);
    if !submitted {
        requested().write().unwrap().remove(&bead_id);
    }
}
