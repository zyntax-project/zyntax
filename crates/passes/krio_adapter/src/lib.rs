//! # krio-async adapter for zyntax HIR
//!
//! Implements the host-side traits krio-async expects (`CoroCfg`,
//! `AsyncHooks`, `SuspendingFns`) over zyntax's HIR. Once wired into
//! `crates/compiler/src/async_support.rs`, this lets us replace the
//! ~3,600-line home-grown async state-machine lowering with
//! `krio_async::transform_to_state_machine` plus a small amount of
//! per-host glue.
//!
//! ## Status — Phase A scaffolding
//!
//! Only the type skeletons exist. Every method is `unimplemented!()`.
//! The crate compiles, depends correctly on krio + zyntax_compiler,
//! and lets later phases fill in methods one at a time without
//! re-arranging the workspace.
//!
//! See `memory/krio_adoption_plan.md` for the phased plan.

use std::collections::HashMap;

use krio_async::{AsyncHooks, FnId as KrioFnId, SuspendingFns, SuspensionSite};
use krio_stackless::{CfgId, CoroCfg};
use zyntax_compiler::hir::{HirFunction, HirId, HirModule};

// ─────────────────────────────────────────────────────────────────────────────
// Identity wrappers — krio's `CfgId` requires `Copy + Eq + Ord + Hash + Debug`
// **and** the `block_count() / block_ids()` contract assumes source-order
// IDs. ZynML's `HirId` is a UUID — neither sortable nor source-ordered — so
// we use sequence numbers as the krio-facing IDs and maintain a side table
// `seq → HirId` inside `HirCoroCfg` for mapping back to the real HIR. This
// decouples the krio adapter from any future HirId representation change.
//
// Both `HirBlockId` and `HirLocalId` get `CfgId` automatically via the
// blanket impl in `krio-core` (any `Copy + Eq + Ord + Hash + Debug`).
// ─────────────────────────────────────────────────────────────────────────────

/// Block identifier in krio's view: a 0-based sequence number assigned in
/// source/iteration order over `HirFunction.blocks`. Maps back to the
/// real `HirId` via `HirCoroCfg::block_id_to_hir`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HirBlockId(pub u32);

/// Local identifier in krio's view: a 0-based sequence number for state /
/// bool / mut-bool locals minted by the transform. Their HIR-level
/// representation (a stack slot via `Alloca`) is created lazily on first
/// use; see `HirCoroCfg::_locals`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HirLocalId(pub u32);

/// Function identifier. Each `HirFunction` has its own `HirId`; we use
/// it directly as the krio FnId.
pub type HirFnId = HirId;

// (`KrioFnId` is auto-impl'd for any `Copy + Eq + Hash + Debug`.)
// Sanity check at compile time:
const _: fn() = || {
    fn assert_impl<T: KrioFnId>() {}
    assert_impl::<HirFnId>();
};

// ─────────────────────────────────────────────────────────────────────────────
// HirCoroCfg — the IR view krio's transform mutates.
//
// Wraps a `&mut HirFunction` plus mintable-id state. krio walks block
// IDs in iteration order, splits blocks at suspension points,
// allocates state/bool locals, and rewrites terminators. All of these
// translate to existing HIR mutations.
// ─────────────────────────────────────────────────────────────────────────────

/// Adapter wrapping an `HirFunction` for krio's `CoroCfg` consumption.
/// Owned by callers; mutates the function in place via the trait
/// methods.
pub struct HirCoroCfg<'f> {
    /// The function being rewritten. krio mutates blocks, statements,
    /// and terminators here in place.
    function: &'f mut HirFunction,

    /// `seq_id → HirId` for blocks. Indexed by `HirBlockId`. New
    /// blocks created via `new_block()` / `split_after()` append to
    /// this table.
    block_seq_to_hir: Vec<HirId>,
    /// Reverse map; updated alongside `block_seq_to_hir`.
    block_hir_to_seq: HashMap<HirId, u32>,

    /// Side table: locals allocated by `new_state_local` /
    /// `new_bool_local` / `new_mut_bool_local`. The HIR-level
    /// representation (stack slot via `Alloca`) is created lazily on
    /// first emit so we don't pollute the function entry block with
    /// unused slots if krio happens not to use a particular local.
    _locals: HashMap<HirLocalId, LocalKind>,
    _next_local_seq: u32,
}

#[derive(Debug, Clone, Copy)]
enum LocalKind {
    /// `new_state_local()` — i64 mut. Used for state-id and
    /// poll-result locals.
    StateI64,
    /// `new_bool_local()` — immutable bool. Used for is_done /
    /// is_ready check temporaries.
    Bool,
    /// `new_mut_bool_local()` — mutable bool. Used by the cooperative
    /// executor's `all_done` flag.
    MutBool,
}

impl<'f> HirCoroCfg<'f> {
    pub fn new(function: &'f mut HirFunction) -> Self {
        let block_seq_to_hir: Vec<HirId> = function.blocks.keys().copied().collect();
        let block_hir_to_seq: HashMap<HirId, u32> = block_seq_to_hir
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i as u32))
            .collect();
        Self {
            function,
            block_seq_to_hir,
            block_hir_to_seq,
            _locals: HashMap::new(),
            _next_local_seq: 0,
        }
    }

    /// Map a krio `HirBlockId` (sequence number) back to the real
    /// `HirId` it stands for.
    pub fn block_id_to_hir(&self, bb: HirBlockId) -> HirId {
        self.block_seq_to_hir[bb.0 as usize]
    }

    /// Map a real `HirId` to its krio `HirBlockId`. Returns `None`
    /// for blocks not in the function (defensive).
    pub fn hir_to_block_id(&self, hir: HirId) -> Option<HirBlockId> {
        self.block_hir_to_seq.get(&hir).copied().map(HirBlockId)
    }
}

impl<'f> CoroCfg for HirCoroCfg<'f> {
    type BlockId = HirBlockId;
    type LocalId = HirLocalId;

    // ── Read access ──────────────────────────────────────────────────

    fn block_count(&self) -> usize {
        self.function.blocks.len()
    }

    fn statement_count(&self, _bb: Self::BlockId) -> usize {
        unimplemented!("HirCoroCfg::statement_count — Phase B")
    }

    fn block_ids(&self) -> Vec<Self::BlockId> {
        (0..self.block_seq_to_hir.len() as u32)
            .map(HirBlockId)
            .collect()
    }

    // ── Construction ─────────────────────────────────────────────────

    fn new_block(&mut self) -> Self::BlockId {
        unimplemented!("HirCoroCfg::new_block — Phase B")
    }

    fn new_state_local(&mut self) -> Self::LocalId {
        unimplemented!("HirCoroCfg::new_state_local — Phase B")
    }

    fn new_bool_local(&mut self) -> Self::LocalId {
        unimplemented!("HirCoroCfg::new_bool_local — Phase B")
    }

    fn new_mut_bool_local(&mut self) -> Self::LocalId {
        unimplemented!("HirCoroCfg::new_mut_bool_local — Phase B")
    }

    // ── Statement emission ───────────────────────────────────────────

    fn emit_assign_i64(&mut self, _bb: Self::BlockId, _local: Self::LocalId, _value: i64) {
        unimplemented!("HirCoroCfg::emit_assign_i64 — Phase B")
    }

    fn emit_assign_bool(&mut self, _bb: Self::BlockId, _local: Self::LocalId, _value: bool) {
        unimplemented!("HirCoroCfg::emit_assign_bool — Phase B")
    }

    fn emit_eq_check_i64(
        &mut self,
        _bb: Self::BlockId,
        _dest: Self::LocalId,
        _lhs: Self::LocalId,
        _rhs: i64,
    ) {
        unimplemented!("HirCoroCfg::emit_eq_check_i64 — Phase B")
    }

    // ── Block manipulation ───────────────────────────────────────────

    fn replace_with_nop(&mut self, _bb: Self::BlockId, _idx: usize) {
        unimplemented!("HirCoroCfg::replace_with_nop — Phase B")
    }

    fn split_after(&mut self, _src: Self::BlockId, _idx: usize) -> Self::BlockId {
        unimplemented!("HirCoroCfg::split_after — Phase B")
    }

    fn prepend_assign_i64(&mut self, _bb: Self::BlockId, _local: Self::LocalId, _value: i64) {
        unimplemented!("HirCoroCfg::prepend_assign_i64 — Phase B")
    }

    // ── Terminator manipulation ──────────────────────────────────────

    fn set_goto(&mut self, _bb: Self::BlockId, _target: Self::BlockId) {
        unimplemented!("HirCoroCfg::set_goto — Phase B")
    }

    fn set_branch(
        &mut self,
        _bb: Self::BlockId,
        _cond: Self::LocalId,
        _true_bb: Self::BlockId,
        _false_bb: Self::BlockId,
    ) {
        unimplemented!("HirCoroCfg::set_branch — Phase B")
    }

    fn set_switch(
        &mut self,
        _bb: Self::BlockId,
        _discr: Self::LocalId,
        _targets: Vec<(i64, Self::BlockId)>,
        _otherwise: Self::BlockId,
    ) {
        unimplemented!("HirCoroCfg::set_switch — Phase B")
    }

    fn redirect_targets(
        &mut self,
        _bb: Self::BlockId,
        _from: Self::BlockId,
        _to: Self::BlockId,
    ) {
        unimplemented!("HirCoroCfg::redirect_targets — Phase B")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HirSuspendingFns — taint analysis result over the HIR call graph.
//
// Two flavours of suspending function:
//   * direct: `is_async` flag set on `TypedFunction` / `HirFunction`
//   * transitive: any function that calls a (direct or transitive)
//     suspending callee.
//
// The single `await` expression is the only "yield primitive" in
// ZynML's surface language — there's no `Fiber.yield`. Implementations
// of `is_yield_primitive` therefore answer `true` only for an internal
// sentinel function id (or, equivalently, `false` for every host
// function and `classify` returns `DirectYield` against the same
// sentinel).
// ─────────────────────────────────────────────────────────────────────────────

/// Set of HIR function ids that may yield directly or transitively.
pub struct HirSuspendingFns {
    pub suspending: std::collections::HashSet<HirFnId>,
    pub yield_primitive: HirFnId,
}

impl HirSuspendingFns {
    /// Compute the suspending set by tainting from `is_async` over the
    /// call graph. Filled in during Phase C.
    pub fn from_module(_module: &HirModule, _yield_primitive: HirFnId) -> Self {
        unimplemented!("HirSuspendingFns::from_module — Phase C")
    }
}

impl SuspendingFns for HirSuspendingFns {
    type FnId = HirFnId;

    fn is_suspending(&self, fn_id: Self::FnId) -> bool {
        self.suspending.contains(&fn_id)
    }

    fn is_yield_primitive(&self, fn_id: Self::FnId) -> bool {
        fn_id == self.yield_primitive
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HirAsyncHooks — host classifier for "is statement (bb, idx) a
// suspension site, and if so what kind".
//
// For ZynML the only suspension primitive is `await call(...)`, which
// after HIR construction is two instructions: a `Call` to the suspending
// callee, then an `Await` wrapper. The `classify` impl matches the
// `Call` instruction and returns `CrossFnCall` (with the callee +
// args). DirectYield is reserved for a future internal yield primitive
// — currently unused.
// ─────────────────────────────────────────────────────────────────────────────

/// Hook implementation that classifies HIR statements as suspension sites.
pub struct HirAsyncHooks<'a> {
    /// Suspending-fns oracle. Held by reference so callers can share
    /// one across multiple `transform_to_state_machine` invocations
    /// over different functions in the same module.
    pub suspending: &'a HirSuspendingFns,
}

impl<'a> AsyncHooks for HirAsyncHooks<'a> {
    type Cfg = HirCoroCfg<'a>;
    type FnId = HirFnId;

    fn classify(
        &self,
        _cfg: &Self::Cfg,
        _bb: <Self::Cfg as CoroCfg>::BlockId,
        _idx: usize,
    ) -> Option<SuspensionSite<Self::FnId, <Self::Cfg as CoroCfg>::LocalId>> {
        unimplemented!("HirAsyncHooks::classify — Phase C")
    }
}
