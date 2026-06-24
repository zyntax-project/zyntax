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

use std::collections::{HashMap, HashSet};

use krio_async::{AsyncHooks, FnId as KrioFnId, LivenessMap, SuspendingFns, SuspensionSite};
use krio_stackless::CoroCfg;

pub mod abi_emit;
pub mod emit;
pub mod fiber; // krio-fiber backed `FiberCfg` impl for first-class fiber HIR ops
pub mod orchestrator;
use zyntax_compiler::hir::{
    BinaryOp, HirBlock, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule,
    HirTerminator, HirType, HirValue, HirValueKind, Intrinsic,
};

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
    locals: HashMap<HirLocalId, LocalKind>,
    /// `local_id → alloca_value_id`, set on first emit. The alloca
    /// instruction itself lives at the front of the function entry
    /// block.
    local_pointers: HashMap<HirLocalId, HirId>,
    next_local_seq: u32,
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
            locals: HashMap::new(),
            local_pointers: HashMap::new(),
            next_local_seq: 0,
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

    /// Read-only access to the wrapped `HirFunction`. Useful for
    /// tests + downstream consumers that need to inspect the
    /// post-transform shape.
    pub fn function(&self) -> &HirFunction {
        self.function
    }

    /// Mutable access to the wrapped `HirFunction`. Phase E's save/load
    /// emission needs this to insert HIR instructions adjacent to the
    /// blocks krio's transform produced.
    pub fn function_mut(&mut self) -> &mut HirFunction {
        self.function
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn fresh_local_id(&mut self, kind: LocalKind) -> HirLocalId {
        let id = HirLocalId(self.next_local_seq);
        self.next_local_seq += 1;
        self.locals.insert(id, kind);
        id
    }

    /// Mint a `HirLocalId` for use as a liveness tag — krio's
    /// `LivenessMap` needs a per-value handle, but these aren't
    /// mutable state-machine locals (they round-trip to existing SSA
    /// values via [`HirLiveness::local_to_hir`]). They share the
    /// counter with state-machine locals to keep the LocalId space
    /// globally unique within the function.
    pub(crate) fn fresh_liveness_local_id(&mut self) -> HirLocalId {
        let id = HirLocalId(self.next_local_seq);
        self.next_local_seq += 1;
        // Intentionally NOT inserted into `self.locals` — calling
        // `alloca_for` on a liveness-tag id would panic, which is the
        // intended trap for accidental reuse.
        id
    }

    /// Mint an SSA value id and register it as a constant of `ty`
    /// holding `constant`. Returns the value id.
    fn emit_constant(&mut self, ty: HirType, constant: HirConstant) -> HirId {
        let value_id = HirId::new();
        self.function.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty,
                kind: HirValueKind::Constant(constant),
                uses: HashSet::new(),
                span: None,
            },
        );
        value_id
    }

    /// Register an instruction-result SSA value id of `ty`.
    fn register_inst_value(&mut self, ty: HirType) -> HirId {
        let value_id = HirId::new();
        self.function.values.insert(
            value_id,
            HirValue {
                id: value_id,
                ty,
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );
        value_id
    }

    /// Get-or-create the alloca pointer backing `local`. The alloca
    /// instruction is inserted at the FRONT of the function entry
    /// block (so all locals are visible from anywhere in the body).
    fn alloca_for(&mut self, local: HirLocalId) -> HirId {
        if let Some(&ptr) = self.local_pointers.get(&local) {
            return ptr;
        }
        let kind = *self.locals.get(&local).expect("unknown local id");
        let elem_ty = match kind {
            LocalKind::StateI64 => HirType::I64,
            LocalKind::Bool | LocalKind::MutBool => HirType::Bool,
        };
        let align = match kind {
            LocalKind::StateI64 => 8,
            LocalKind::Bool | LocalKind::MutBool => 1,
        };
        let ptr_id = self.register_inst_value(HirType::Ptr(Box::new(elem_ty.clone())));
        let alloca = HirInstruction::Alloca {
            result: ptr_id,
            ty: elem_ty,
            count: None,
            align,
        };
        // Insert at front of entry block.
        let entry = self.function.entry_block;
        let entry_block = self
            .function
            .blocks
            .get_mut(&entry)
            .expect("entry block not in function.blocks");
        entry_block.instructions.insert(0, alloca);

        self.local_pointers.insert(local, ptr_id);
        ptr_id
    }

    fn block_mut(&mut self, bb: HirBlockId) -> &mut HirBlock {
        let hir = self.block_seq_to_hir[bb.0 as usize];
        self.function
            .blocks
            .get_mut(&hir)
            .expect("block id out of range")
    }
}

impl<'f> CoroCfg for HirCoroCfg<'f> {
    type BlockId = HirBlockId;
    type LocalId = HirLocalId;

    // ── Read access ──────────────────────────────────────────────────

    fn block_count(&self) -> usize {
        self.function.blocks.len()
    }

    fn statement_count(&self, bb: Self::BlockId) -> usize {
        let hir = self.block_seq_to_hir[bb.0 as usize];
        self.function
            .blocks
            .get(&hir)
            .map(|b| b.instructions.len())
            .unwrap_or(0)
    }

    fn block_ids(&self) -> Vec<Self::BlockId> {
        (0..self.block_seq_to_hir.len() as u32)
            .map(HirBlockId)
            .collect()
    }

    // ── Construction ─────────────────────────────────────────────────

    fn new_block(&mut self) -> Self::BlockId {
        let hir_id = HirId::new();
        let block = HirBlock {
            id: hir_id,
            label: None,
            phis: Vec::new(),
            instructions: Vec::new(),
            // Krio overwrites this immediately via set_goto/set_branch/set_switch.
            terminator: HirTerminator::Unreachable,
            dominance_frontier: HashSet::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        };
        self.function.blocks.insert(hir_id, block);
        let seq = self.block_seq_to_hir.len() as u32;
        self.block_seq_to_hir.push(hir_id);
        self.block_hir_to_seq.insert(hir_id, seq);
        HirBlockId(seq)
    }

    fn new_state_local(&mut self) -> Self::LocalId {
        self.fresh_local_id(LocalKind::StateI64)
    }

    fn new_bool_local(&mut self) -> Self::LocalId {
        self.fresh_local_id(LocalKind::Bool)
    }

    fn new_mut_bool_local(&mut self) -> Self::LocalId {
        self.fresh_local_id(LocalKind::MutBool)
    }

    // ── Statement emission ───────────────────────────────────────────

    fn emit_assign_i64(&mut self, bb: Self::BlockId, local: Self::LocalId, value: i64) {
        let ptr = self.alloca_for(local);
        let const_id = self.emit_constant(HirType::I64, HirConstant::I64(value));
        self.block_mut(bb).instructions.push(HirInstruction::Store {
            value: const_id,
            ptr,
            align: 8,
            volatile: false,
        });
    }

    fn emit_assign_bool(&mut self, bb: Self::BlockId, local: Self::LocalId, value: bool) {
        let ptr = self.alloca_for(local);
        let const_id = self.emit_constant(HirType::Bool, HirConstant::Bool(value));
        self.block_mut(bb).instructions.push(HirInstruction::Store {
            value: const_id,
            ptr,
            align: 1,
            volatile: false,
        });
    }

    fn emit_eq_check_i64(
        &mut self,
        bb: Self::BlockId,
        dest: Self::LocalId,
        lhs: Self::LocalId,
        rhs: i64,
    ) {
        // dest = (load lhs) == const(rhs)
        let lhs_ptr = self.alloca_for(lhs);
        let dest_ptr = self.alloca_for(dest);
        let lhs_val = self.register_inst_value(HirType::I64);
        let cmp_result = self.register_inst_value(HirType::Bool);
        let rhs_const = self.emit_constant(HirType::I64, HirConstant::I64(rhs));
        let block = self.block_mut(bb);
        block.instructions.push(HirInstruction::Load {
            result: lhs_val,
            ty: HirType::I64,
            ptr: lhs_ptr,
            align: 8,
            volatile: false,
        });
        block.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Eq,
            result: cmp_result,
            ty: HirType::Bool,
            left: lhs_val,
            right: rhs_const,
        });
        block.instructions.push(HirInstruction::Store {
            value: cmp_result,
            ptr: dest_ptr,
            align: 1,
            volatile: false,
        });
    }

    // ── Block manipulation ───────────────────────────────────────────

    fn replace_with_nop(&mut self, bb: Self::BlockId, idx: usize) {
        // HIR has no Nop variant. We pop the instruction and re-thread —
        // krio uses this to erase markers, which is benign because
        // markers don't define any value other code references. Pop
        // changes statement indices for all subsequent statements in
        // the block, but krio's own algorithm is careful to call this
        // only on indices it isn't tracking by position anymore.
        let block = self.block_mut(bb);
        if idx < block.instructions.len() {
            block.instructions.remove(idx);
        }
    }

    fn split_after(&mut self, src: Self::BlockId, idx: usize) -> Self::BlockId {
        // Move instructions[idx+1..] + terminator into a fresh block.
        // The src block ends at idx (inclusive) with no terminator —
        // krio's caller is expected to set one immediately.
        let new_bb = self.new_block();
        let (tail, term) = {
            let src_block = self.block_mut(src);
            let split_at = (idx + 1).min(src_block.instructions.len());
            let tail: Vec<HirInstruction> = src_block.instructions.drain(split_at..).collect();
            // Take the terminator, leaving Unreachable as a placeholder.
            let term = std::mem::replace(&mut src_block.terminator, HirTerminator::Unreachable);
            (tail, term)
        };
        let new_block = self.block_mut(new_bb);
        new_block.instructions = tail;
        new_block.terminator = term;
        new_bb
    }

    fn prepend_assign_i64(&mut self, bb: Self::BlockId, local: Self::LocalId, value: i64) {
        let ptr = self.alloca_for(local);
        let const_id = self.emit_constant(HirType::I64, HirConstant::I64(value));
        let block = self.block_mut(bb);
        block.instructions.insert(
            0,
            HirInstruction::Store {
                value: const_id,
                ptr,
                align: 8,
                volatile: false,
            },
        );
    }

    // ── Terminator manipulation ──────────────────────────────────────

    fn set_goto(&mut self, bb: Self::BlockId, target: Self::BlockId) {
        let target_hir = self.block_id_to_hir(target);
        self.block_mut(bb).terminator = HirTerminator::Branch { target: target_hir };
    }

    fn set_branch(
        &mut self,
        bb: Self::BlockId,
        cond: Self::LocalId,
        true_bb: Self::BlockId,
        false_bb: Self::BlockId,
    ) {
        // Load the cond bool from its alloca, then CondBranch on the loaded SSA value.
        let cond_ptr = self.alloca_for(cond);
        let cond_val = self.register_inst_value(HirType::Bool);
        let true_hir = self.block_id_to_hir(true_bb);
        let false_hir = self.block_id_to_hir(false_bb);
        let block = self.block_mut(bb);
        block.instructions.push(HirInstruction::Load {
            result: cond_val,
            ty: HirType::Bool,
            ptr: cond_ptr,
            align: 1,
            volatile: false,
        });
        block.terminator = HirTerminator::CondBranch {
            condition: cond_val,
            true_target: true_hir,
            false_target: false_hir,
        };
    }

    fn set_switch(
        &mut self,
        bb: Self::BlockId,
        discr: Self::LocalId,
        targets: Vec<(i64, Self::BlockId)>,
        otherwise: Self::BlockId,
    ) {
        // Load discriminant, then Switch on it.
        let discr_ptr = self.alloca_for(discr);
        let discr_val = self.register_inst_value(HirType::I64);
        let cases: Vec<(HirConstant, HirId)> = targets
            .into_iter()
            .map(|(v, t)| (HirConstant::I64(v), self.block_id_to_hir(t)))
            .collect();
        let default_hir = self.block_id_to_hir(otherwise);
        let block = self.block_mut(bb);
        block.instructions.push(HirInstruction::Load {
            result: discr_val,
            ty: HirType::I64,
            ptr: discr_ptr,
            align: 8,
            volatile: false,
        });
        block.terminator = HirTerminator::Switch {
            value: discr_val,
            default: default_hir,
            cases,
        };
    }

    fn redirect_targets(&mut self, bb: Self::BlockId, from: Self::BlockId, to: Self::BlockId) {
        let from_hir = self.block_id_to_hir(from);
        let to_hir = self.block_id_to_hir(to);
        let block = self.block_mut(bb);
        match &mut block.terminator {
            HirTerminator::Branch { target } => {
                if *target == from_hir {
                    *target = to_hir;
                }
            }
            HirTerminator::CondBranch {
                true_target,
                false_target,
                ..
            } => {
                if *true_target == from_hir {
                    *true_target = to_hir;
                }
                if *false_target == from_hir {
                    *false_target = to_hir;
                }
            }
            HirTerminator::Switch { default, cases, .. } => {
                if *default == from_hir {
                    *default = to_hir;
                }
                for (_, t) in cases.iter_mut() {
                    if *t == from_hir {
                        *t = to_hir;
                    }
                }
            }
            HirTerminator::Invoke { normal, unwind, .. } => {
                if *normal == from_hir {
                    *normal = to_hir;
                }
                if *unwind == from_hir {
                    *unwind = to_hir;
                }
            }
            HirTerminator::PatternMatch {
                patterns, default, ..
            } => {
                for p in patterns.iter_mut() {
                    if p.target == from_hir {
                        p.target = to_hir;
                    }
                }
                if let Some(d) = default {
                    if *d == from_hir {
                        *d = to_hir;
                    }
                }
            }
            HirTerminator::Return { .. } | HirTerminator::Unreachable => {}
        }
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
///
/// In ZynML the only suspension primitive is the `Intrinsic::Await`
/// call, not a host function — so `is_yield_primitive` always returns
/// `false` and the suspending set is purely the seed of `is_async`
/// functions plus their transitive callers.
pub struct HirSuspendingFns {
    pub suspending: HashSet<HirFnId>,
}

impl HirSuspendingFns {
    /// Compute the suspending set by tainting from `is_async` over the
    /// call graph.
    ///
    /// Seeds:
    ///   * direct: `signature.is_async` set on the function.
    ///   * effectful: `signature.effects` is non-empty — the function
    ///     performs at least one algebraic effect, which we treat as a
    ///     suspension point (the handler may resume out-of-line).
    ///   * (M3) any function whose body contains a `PerformEffect` is
    ///     also seeded — even if the `effects` annotation was lost
    ///     (defensive) the perform-site still demands captures-lift.
    ///
    /// Then iterate to a fixed point: a function becomes suspending if
    /// it Calls any suspending function (`HirCallable::Function`
    /// variant only; Indirect / Intrinsic calls don't taint).
    ///
    /// `Intrinsic::Await` and `PerformEffect` are suspension primitives
    /// but they aren't `HirCallable::Function`, so they don't enter the
    /// call-graph fixpoint — `classify` recognises them directly.
    pub fn from_module(module: &HirModule) -> Self {
        let mut suspending: HashSet<HirFnId> = HashSet::new();
        for func in module.functions.values() {
            if func.signature.is_async
                || !func.signature.effects.is_empty()
                || function_performs_any_effect(func)
            {
                suspending.insert(func.id);
            }
        }

        loop {
            let mut changed = false;
            for func in module.functions.values() {
                if suspending.contains(&func.id) {
                    continue;
                }
                if function_calls_any(func, &suspending) {
                    suspending.insert(func.id);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        Self { suspending }
    }
}

fn function_calls_any(func: &HirFunction, suspending: &HashSet<HirFnId>) -> bool {
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Call { callee, .. } = inst {
                if let HirCallable::Function(callee_id) = callee {
                    if suspending.contains(callee_id) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Returns true if `func` contains any `HirInstruction::PerformEffect`.
fn function_performs_any_effect(func: &HirFunction) -> bool {
    func.blocks.values().any(|b| {
        b.instructions
            .iter()
            .any(|i| matches!(i, HirInstruction::PerformEffect { .. }))
    })
}

impl SuspendingFns for HirSuspendingFns {
    type FnId = HirFnId;

    fn is_suspending(&self, fn_id: Self::FnId) -> bool {
        self.suspending.contains(&fn_id)
    }

    fn is_yield_primitive(&self, _fn_id: Self::FnId) -> bool {
        // ZynML's only yield primitive is the Intrinsic::Await call,
        // which isn't a HirCallable::Function — so no host function
        // ever counts as a yield primitive.
        false
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
        cfg: &Self::Cfg,
        bb: <Self::Cfg as CoroCfg>::BlockId,
        idx: usize,
    ) -> Option<SuspensionSite<Self::FnId, <Self::Cfg as CoroCfg>::LocalId>> {
        let hir = cfg.block_seq_to_hir.get(bb.0 as usize)?;
        let block = cfg.function.blocks.get(hir)?;
        let inst = block.instructions.get(idx)?;

        // Pattern 1: `Call { callee: Intrinsic(Await), .. }` — the
        // `await foo(x)` lowering's actual suspension point. Returns
        // DirectYield with no value field — krio uses the value only
        // as informational data the host can later look up; we drive
        // captures lift through `LivenessMap` instead.
        if let HirInstruction::Call {
            callee: HirCallable::Intrinsic(Intrinsic::Await),
            ..
        } = inst
        {
            return Some(SuspensionSite::DirectYield { value: None });
        }

        // Pattern 2 (M3, Phase H): `PerformEffect { effect_id, op_name, .. }`
        // — a call to an algebraic effect operation. From krio's
        // perspective this is structurally identical to await: control
        // transfers to the active handler, which may resume out-of-line
        // (resumable handler) or abort (exception-like). Either way,
        // values live across this site need to be saved into the state
        // machine's frame slots. Captures-lift is driven by the
        // `LivenessMap` populated below in `HirLiveness::build`, so
        // returning `DirectYield` here suffices to register the site.
        if let HirInstruction::PerformEffect { .. } = inst {
            return Some(SuspensionSite::DirectYield { value: None });
        }

        // Pattern 3 (future work): direct cross-fn call to an async
        // function without going through Intrinsic::Await. Today's
        // ZynML lowering always interposes Intrinsic::Await, so this
        // never fires; reserve the shape for an optimised path later.
        // if let HirInstruction::Call { callee: HirCallable::Function(fid), args, result, .. } = inst {
        //     if self.suspending.is_suspending(*fid) {
        //         return Some(SuspensionSite::CrossFnCall { ... });
        //     }
        // }

        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-instruction def/use helpers used by the captures-lift liveness pass.
// These cover the HIR variants the canonical async pipeline produces;
// any case that's neither matched nor a side-effecting non-result inst
// returns conservatively (no result / no uses listed). That's safe for
// captures-lift: missing a use under-saves and would crash; missing a
// def over-saves, which is correct.
// ─────────────────────────────────────────────────────────────────────────────

fn instruction_result(inst: &HirInstruction) -> Option<HirId> {
    use HirInstruction as I;
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
        | I::Atomic { result, .. }
        | I::CreateUnion { result, .. }
        | I::GetUnionDiscriminant { result, .. }
        | I::ExtractUnionValue { result, .. }
        | I::CreateClosure { result, .. }
        | I::CreateTraitObject { result, .. }
        | I::UpcastTraitObject { result, .. }
        | I::AsyncLoadSlot { result, .. } => Some(*result),

        I::Call { result, .. }
        | I::IndirectCall { result, .. }
        | I::CallClosure { result, .. }
        | I::TraitMethodCall { result, .. }
        | I::PerformEffect { result, .. }
        | I::HandleEffect { result, .. } => *result,

        _ => None,
    }
}

fn instruction_uses(inst: &HirInstruction) -> smallvec::SmallVec<[HirId; 4]> {
    use HirInstruction as I;
    let mut uses = smallvec::SmallVec::new();
    match inst {
        I::Binary { left, right, .. } => {
            uses.push(*left);
            uses.push(*right);
        }
        I::Unary { operand, .. } | I::Cast { operand, .. } => uses.push(*operand),
        I::Alloca { count, .. } => {
            if let Some(c) = count {
                uses.push(*c);
            }
        }
        I::Load { ptr, .. } => uses.push(*ptr),
        I::Store { value, ptr, .. } => {
            uses.push(*value);
            uses.push(*ptr);
        }
        I::GetElementPtr { ptr, indices, .. } => {
            uses.push(*ptr);
            for v in indices {
                uses.push(*v);
            }
        }
        I::Select {
            condition,
            true_val,
            false_val,
            ..
        } => {
            uses.push(*condition);
            uses.push(*true_val);
            uses.push(*false_val);
        }
        I::ExtractValue { aggregate, .. } => uses.push(*aggregate),
        I::InsertValue {
            aggregate, value, ..
        } => {
            uses.push(*aggregate);
            uses.push(*value);
        }
        I::Call { args, .. } | I::IndirectCall { args, .. } | I::CallClosure { args, .. } => {
            for a in args {
                uses.push(*a);
            }
        }
        I::TraitMethodCall {
            trait_object, args, ..
        } => {
            uses.push(*trait_object);
            for a in args {
                uses.push(*a);
            }
        }
        I::AsyncSaveSlot { frame, value, .. } => {
            uses.push(*frame);
            uses.push(*value);
        }
        I::AsyncLoadSlot { frame, .. } => uses.push(*frame),
        I::PerformEffect { args, .. } => {
            for a in args {
                uses.push(*a);
            }
        }
        _ => {}
    }
    uses
}

fn terminator_uses(term: &zyntax_compiler::hir::HirTerminator) -> smallvec::SmallVec<[HirId; 4]> {
    use zyntax_compiler::hir::HirTerminator as T;
    let mut uses = smallvec::SmallVec::new();
    match term {
        T::Return { values } => {
            for v in values {
                uses.push(*v);
            }
        }
        T::CondBranch { condition, .. } => uses.push(*condition),
        T::Switch { value, .. } => uses.push(*value),
        T::Invoke { args, .. } => {
            for a in args {
                uses.push(*a);
            }
        }
        T::PatternMatch { value, .. } => uses.push(*value),
        T::Branch { .. } | T::Unreachable => {}
    }
    uses
}

// ─────────────────────────────────────────────────────────────────────────────
// HirLiveness — krio's `LivenessMap` populated from zyntax's existing
// per-block liveness analysis (`zyntax_compiler::analysis::LivenessAnalysis`).
//
// krio expects liveness keyed by `(block_id, statement_idx)` for each
// suspension site, with values as krio `LocalId`s. We map SSA `HirId`s
// to "naked" liveness `HirLocalId`s and persist the round-trip mapping
// so Phase E's save/load emission can recover the SSA value to spill.
// ─────────────────────────────────────────────────────────────────────────────

/// Live-across-suspension data for a single async function, in the
/// shape krio-async's `transform_to_state_machine` consumes.
pub struct HirLiveness {
    /// SSA `HirId` → krio `HirLocalId`. Same SSA value shares the
    /// same LocalId at every suspension site it crosses, which lets
    /// krio's slot allocator assign a single slot for the value's
    /// entire live range.
    pub hir_to_local: HashMap<HirId, HirLocalId>,
    /// Reverse map. Used by Phase E during save/load emission to
    /// recover the SSA `HirId` from the LocalId krio reports back.
    pub local_to_hir: HashMap<HirLocalId, HirId>,
    /// The actual `LivenessMap` to feed into the transform.
    pub map: LivenessMap<HirBlockId, HirLocalId>,
}

impl HirLiveness {
    /// Build a `LivenessMap` covering every `Intrinsic::Await`
    /// suspension site in `cfg.function`.
    ///
    /// At each suspension site `(bb, idx)` the live-across set is:
    ///
    /// ```text
    ///   { v defined before inst[idx] } ∩ { v used at or after inst[idx+1] }
    /// ```
    ///
    /// — exactly what krio-async's contract requires (see
    /// `LivenessMap` doc lines 257-269: "defined before the suspension
    /// and used after it"). Values defined AT the suspension (e.g.
    /// the `await_result` SSA value, written by the runtime on
    /// resume) are intentionally excluded — they're set fresh on
    /// resume, not captured.
    ///
    /// Algorithm per suspension:
    ///   1. Forward pass over `inst[0..idx]` and the function entry
    ///      params: collect the "defined before" set.
    ///   2. Backward pass starting from `live_out[bb]` ∪ terminator
    ///      uses, walking instructions in reverse from `len-1` down
    ///      to `idx+1`, applying `live = use(k) ∪ (live - def(k))`
    ///      at each step. Result is "values entering inst[idx+1]"
    ///      which equals the post-suspension live set.
    ///   3. Intersect; that's the captures-lift set for this site.
    ///
    /// `live_out_per_block` should come from
    /// `zyntax_compiler::analysis::LivenessAnalysis::live_out`.
    pub fn build(
        cfg: &mut HirCoroCfg<'_>,
        live_out_per_block: &HashMap<HirId, HashSet<HirId>>,
    ) -> Self {
        let mut hir_to_local: HashMap<HirId, HirLocalId> = HashMap::new();
        let mut local_to_hir: HashMap<HirLocalId, HirId> = HashMap::new();
        let mut map = LivenessMap::new();

        // Snapshot the suspension sites and their live-across sets
        // first — we need `&mut cfg` for minting LocalIds later, but
        // iterating `cfg.function` and mutating it simultaneously
        // would alias.
        let mut sites: Vec<(HirBlockId, usize, HashSet<HirId>)> = Vec::new();
        for (seq, hir_block_id) in cfg.block_seq_to_hir.iter().enumerate() {
            let block = match cfg.function.blocks.get(hir_block_id) {
                Some(b) => b,
                None => continue,
            };
            // Find every suspension index in this block. Two flavors:
            //   * `Intrinsic::Await` — the `await foo(x)` lowering
            //   * `PerformEffect` — a call to an algebraic effect op
            //     (M3, Phase H — handler may resume out-of-line)
            let suspension_indices: Vec<usize> = block
                .instructions
                .iter()
                .enumerate()
                .filter_map(|(i, inst)| {
                    let is_suspension = matches!(
                        inst,
                        HirInstruction::Call {
                            callee: HirCallable::Intrinsic(Intrinsic::Await),
                            ..
                        }
                    ) || matches!(inst, HirInstruction::PerformEffect { .. });
                    if is_suspension {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            if suspension_indices.is_empty() {
                continue;
            }

            // Compute "defined before idx" for every suspension idx
            // in one forward pass: incrementally accumulate defs.
            // `defined_at_or_before[i]` = set of values defined by
            // inst[0..=i] plus function params plus all phi results
            // (defined at the top of their respective blocks; we
            // approximate dominance conservatively by including ALL
            // phi results in the function — captures-lift will then
            // intersect with `live_post` to actually decide what to
            // save). Without phi inclusion, await-in-loop patterns
            // miss the loop counter / accumulator phis and the
            // resume side reads undefined values.
            let mut defined_at_or_before: Vec<HashSet<HirId>> =
                Vec::with_capacity(block.instructions.len());
            let mut acc: HashSet<HirId> =
                cfg.function.signature.params.iter().map(|p| p.id).collect();
            // Add all phi results across the function (cross-block
            // SSA values).
            for other_block in cfg.function.blocks.values() {
                for phi in &other_block.phis {
                    acc.insert(phi.result);
                }
            }
            for inst in &block.instructions {
                if let Some(def) = instruction_result(inst) {
                    acc.insert(def);
                }
                defined_at_or_before.push(acc.clone());
            }

            // Backward pass: live-after each instruction. live_after[k]
            // = values live just AFTER inst[k] = live_before(inst[k+1]).
            // Seed: live_after[len-1] = live_out[bb] ∪ uses(terminator).
            let mut live_after: Vec<HashSet<HirId>> =
                vec![HashSet::new(); block.instructions.len()];
            let block_live_out = live_out_per_block
                .get(hir_block_id)
                .cloned()
                .unwrap_or_default();
            let mut working: HashSet<HirId> = block_live_out.clone();
            for v in terminator_uses(&block.terminator) {
                working.insert(v);
            }
            // Walk from last instruction back to first.
            for k in (0..block.instructions.len()).rev() {
                // live_after[k] = working state before applying inst[k]'s effects
                live_after[k] = working.clone();
                let inst = &block.instructions[k];
                if let Some(def) = instruction_result(inst) {
                    working.remove(&def);
                }
                for u in instruction_uses(inst) {
                    working.insert(u);
                }
                // After this loop iteration, `working` is live BEFORE
                // inst[k] = live_after[k-1] when k > 0.
            }

            for idx in suspension_indices {
                // live_at_suspension = "live entering inst[idx+1]"
                // = live_after[idx] when idx+1 < len, or
                //   block_live_out + terminator_uses when idx is last.
                let live_post = if idx + 1 < live_after.len() {
                    live_after[idx].clone()
                } else {
                    let mut s = block_live_out.clone();
                    for v in terminator_uses(&block.terminator) {
                        s.insert(v);
                    }
                    s
                };
                let defined_pre = if idx == 0 {
                    // Function params + all phi results (defined at
                    // top of their blocks, conservatively assumed to
                    // dominate this site via SSA).
                    let mut s: HashSet<HirId> =
                        cfg.function.signature.params.iter().map(|p| p.id).collect();
                    for other_block in cfg.function.blocks.values() {
                        for phi in &other_block.phis {
                            s.insert(phi.result);
                        }
                    }
                    s
                } else {
                    defined_at_or_before[idx - 1].clone()
                };
                let live: HashSet<HirId> = live_post.intersection(&defined_pre).copied().collect();
                sites.push((HirBlockId(seq as u32), idx, live));
            }
        }

        for (bb, idx, live_set) in sites {
            // Convert live SSA HirIds → LocalIds, keeping the bijection.
            let mut locals_at_site: Vec<HirLocalId> = Vec::with_capacity(live_set.len());
            // Iterate in deterministic order (by HirId hash via sort)
            // so the LocalId allocation is stable across runs — krio's
            // slot allocator uses BTreeMap, but our LocalId minting is
            // sequential, so insertion order matters for which slot
            // a given SSA value gets. Stable order = stable test output.
            let mut ordered: Vec<HirId> = live_set.into_iter().collect();
            // HirId has no Ord; we sort by its Debug repr as a stable
            // surrogate. This sort is only for testability — the
            // transform doesn't depend on it.
            ordered.sort_by_key(|id| format!("{:?}", id));
            for hir_id in ordered {
                let local = if let Some(&existing) = hir_to_local.get(&hir_id) {
                    existing
                } else {
                    let new_local = cfg.fresh_liveness_local_id();
                    hir_to_local.insert(hir_id, new_local);
                    local_to_hir.insert(new_local, hir_id);
                    new_local
                };
                locals_at_site.push(local);
            }
            map.record(bb, idx, locals_at_site);
        }

        Self {
            hir_to_local,
            local_to_hir,
            map,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use zyntax_compiler::hir::{HirFunction, HirFunctionSignature, ParamAttributes};
    use zyntax_typed_ast::InternedString;

    fn empty_function(name: &str) -> HirFunction {
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: true,
        };
        let mut func = HirFunction::new(InternedString::new_global(name), sig);
        // Add a single entry block with an Unreachable terminator so
        // alloca_for has somewhere to insert.
        let entry_id = HirId::new();
        let entry = HirBlock {
            id: entry_id,
            label: Some(InternedString::new_global("entry")),
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Unreachable,
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        };
        let mut blocks = IndexMap::new();
        blocks.insert(entry_id, entry);
        func.blocks = blocks;
        func.entry_block = entry_id;
        func.is_external = false;
        func
    }

    #[test]
    fn block_count_and_ids_round_trip() {
        let mut f = empty_function("test");
        let cfg = HirCoroCfg::new(&mut f);
        assert_eq!(cfg.block_count(), 1);
        let ids = cfg.block_ids();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], HirBlockId(0));
    }

    #[test]
    fn new_block_appends() {
        let mut f = empty_function("test");
        let mut cfg = HirCoroCfg::new(&mut f);
        let bb = cfg.new_block();
        assert_eq!(bb, HirBlockId(1));
        assert_eq!(cfg.block_count(), 2);
        // hir_to_block_id round trip
        let hir = cfg.block_id_to_hir(bb);
        assert_eq!(cfg.hir_to_block_id(hir), Some(bb));
    }

    #[test]
    fn locals_lazy_alloca_at_entry() {
        let mut f = empty_function("test");
        let entry = f.entry_block;
        let mut cfg = HirCoroCfg::new(&mut f);

        // Mint a state local; alloca shouldn't exist yet.
        let l = cfg.new_state_local();
        assert_eq!(cfg.function.blocks[&entry].instructions.len(), 0);

        // First emit forces the alloca.
        cfg.emit_assign_i64(HirBlockId(0), l, 42);
        let entry_block = &cfg.function.blocks[&entry];
        // Expect: [Alloca, Store]. Const value lives in function.values, not in the block.
        assert_eq!(entry_block.instructions.len(), 2);
        assert!(matches!(
            entry_block.instructions[0],
            HirInstruction::Alloca { .. }
        ));
        assert!(matches!(
            entry_block.instructions[1],
            HirInstruction::Store { .. }
        ));

        // Second emit reuses the alloca, just appends another Store.
        cfg.emit_assign_i64(HirBlockId(0), l, 100);
        let entry_block = &cfg.function.blocks[&entry];
        assert_eq!(entry_block.instructions.len(), 3);
    }

    #[test]
    fn split_after_moves_tail_and_terminator() {
        let mut f = empty_function("test");
        let entry = f.entry_block;
        // Plant 3 instructions + a Branch terminator on the entry block,
        // pointing at a fresh second block.
        let other_id = HirId::new();
        f.blocks.insert(
            other_id,
            HirBlock {
                id: other_id,
                label: None,
                phis: vec![],
                instructions: vec![],
                terminator: HirTerminator::Unreachable,
                dominance_frontier: HashSet::new(),
                predecessors: vec![],
                successors: vec![],
            },
        );
        let entry_block = f.blocks.get_mut(&entry).unwrap();
        for _ in 0..3 {
            // Cheap "no result needed" fences are a poor fit for HIR;
            // use a concrete instruction shape to populate the list.
            let v = HirId::new();
            entry_block.instructions.push(HirInstruction::Alloca {
                result: v,
                ty: HirType::I64,
                count: None,
                align: 8,
            });
        }
        entry_block.terminator = HirTerminator::Branch { target: other_id };

        let mut cfg = HirCoroCfg::new(&mut f);
        let new_bb = cfg.split_after(HirBlockId(0), 0);
        let new_hir = cfg.block_id_to_hir(new_bb);

        // src now has just instructions[..=0] and an Unreachable terminator
        // (krio's caller is expected to set one).
        let src = &cfg.function.blocks[&cfg.block_id_to_hir(HirBlockId(0))];
        assert_eq!(src.instructions.len(), 1);
        assert!(matches!(src.terminator, HirTerminator::Unreachable));

        // The new block has the moved tail (2 instructions) and the original
        // Branch terminator.
        let new_block = &cfg.function.blocks[&new_hir];
        assert_eq!(new_block.instructions.len(), 2);
        assert!(matches!(
            new_block.terminator,
            HirTerminator::Branch { target } if target == other_id
        ));
    }

    #[test]
    fn suspending_fns_taints_transitively() {
        // module shape:
        //   async fn a() i32 { return 1; }
        //   fn b() i32 { return a() + 1; }   ← becomes suspending via taint
        //   fn c() i32 { return 2; }         ← stays sync
        let mut module = HirModule::new(InternedString::new_global("test"));
        let mk_sig = |is_async: bool| HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I32],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async,
            effects: vec![],
            is_pure: false,
        };
        let a_fn = HirFunction::new(InternedString::new_global("a"), mk_sig(true));
        let a_id = a_fn.id;
        let mut b_fn = HirFunction::new(InternedString::new_global("b"), mk_sig(false));
        let b_id = b_fn.id;
        let c_fn = HirFunction::new(InternedString::new_global("c"), mk_sig(false));
        let c_id = c_fn.id;

        // b has a single block with `Call a()` — that's enough to taint it.
        let bb_id = HirId::new();
        let mut bb = HirBlock {
            id: bb_id,
            label: None,
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Unreachable,
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        };
        bb.instructions.push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Function(a_id),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        let mut blocks = IndexMap::new();
        blocks.insert(bb_id, bb);
        b_fn.blocks = blocks;
        b_fn.entry_block = bb_id;

        module.functions.insert(a_id, a_fn);
        module.functions.insert(b_id, b_fn);
        module.functions.insert(c_id, c_fn);

        let s = HirSuspendingFns::from_module(&module);
        assert!(s.is_suspending(a_id), "a is async → suspending");
        assert!(s.is_suspending(b_id), "b calls a → suspending via taint");
        assert!(!s.is_suspending(c_id), "c is sync and calls nothing async");
        assert!(!s.is_yield_primitive(a_id));
    }

    #[test]
    fn classify_returns_direct_yield_for_intrinsic_await() {
        let mut f = empty_function("test");
        let entry = f.entry_block;
        // Plant a Call to Intrinsic::Await on the entry block.
        let result_id = HirId::new();
        f.values.insert(
            result_id,
            HirValue {
                id: result_id,
                ty: HirType::I32,
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );
        let entry_block = f.blocks.get_mut(&entry).unwrap();
        entry_block.instructions.push(HirInstruction::Call {
            result: Some(result_id),
            callee: HirCallable::Intrinsic(Intrinsic::Await),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });

        let cfg = HirCoroCfg::new(&mut f);
        let suspending = HirSuspendingFns {
            suspending: HashSet::new(),
        };
        let hooks = HirAsyncHooks {
            suspending: &suspending,
        };
        let site = hooks.classify(&cfg, HirBlockId(0), 0);
        assert!(matches!(site, Some(SuspensionSite::DirectYield { .. })));
    }

    #[test]
    fn transform_to_state_machine_smoke_test() {
        // Smallest possible end-to-end: an async fn with a single
        // Intrinsic::Await in the entry block. Verifies the trait
        // impls compose correctly with krio_async::transform_to_state_machine
        // — the layout should have a resume_entry created by splitting
        // the entry block at the await, and one yield_block recorded.
        let mut f = empty_function("aw");
        let entry = f.entry_block;
        // Make `live` a function parameter — params are "defined
        // before everything", so the intra-block captures-lift will
        // correctly identify it as crossing the await.
        let live = HirId::new();
        f.values.insert(
            live,
            HirValue {
                id: live,
                ty: HirType::I32,
                kind: HirValueKind::Parameter(0),
                uses: HashSet::new(),
                span: None,
            },
        );
        f.signature.params.push(zyntax_compiler::hir::HirParam {
            id: live,
            name: InternedString::new_global("live"),
            ty: HirType::I32,
            attributes: ParamAttributes::default(),
        });
        let entry_block = f.blocks.get_mut(&entry).unwrap();
        // Plant an await call followed by a Return — krio splits at the await.
        entry_block.instructions.push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Intrinsic(Intrinsic::Await),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        entry_block.terminator = HirTerminator::Return { values: vec![live] };
        // Mark the function async so SuspendingFns picks it up.
        f.signature.is_async = true;
        let fn_id = f.id;

        // Build a one-fn module so SuspendingFns has something to taint over.
        let mut module = HirModule::new(InternedString::new_global("m"));
        module.functions.insert(fn_id, f);
        let suspending = HirSuspendingFns::from_module(&module);
        assert!(suspending.is_suspending(fn_id));

        // Get the function back out for mutation. (Module owns it; pull
        // out, transform, put back — keeps lifetimes clean.)
        let mut f = module.functions.swap_remove(&fn_id).unwrap();
        let mut live_out = HashMap::new();
        let mut s = HashSet::new();
        s.insert(live);
        live_out.insert(entry, s);

        let mut cfg = HirCoroCfg::new(&mut f);
        let liveness = HirLiveness::build(&mut cfg, &live_out);
        assert_eq!(liveness.map.at_site.len(), 1, "one await site");
        let hooks = HirAsyncHooks {
            suspending: &suspending,
        };

        let layout = krio_async::transform_to_state_machine(
            &mut cfg,
            fn_id,
            &suspending,
            &hooks,
            &liveness.map,
        )
        .expect("transform should succeed");

        // resume_entries[0] is always the original entry block.
        assert!(!layout.resume_entries.is_empty());
        assert_eq!(layout.resume_entries[0], HirBlockId(0));
        // We had one DirectYield; expect one resume entry beyond the
        // original (state 1) and one yield block.
        assert_eq!(layout.resume_entries.len(), 2);
        assert_eq!(layout.yield_blocks.len(), 1);
        // The yield block is the original entry (split at the await).
        assert_eq!(layout.yield_blocks[0].0, HirBlockId(0));
        // next_state for the yield is 1 (resumes at resume_entries[1]).
        assert_eq!(layout.yield_blocks[0].1, 1);
        // Block kind should be DirectYield.
        assert!(matches!(
            layout.block_kinds[0].1,
            krio_async::BlockKind::DirectYield
        ));

        // Captures lift: the live SSA value should produce one save
        // (at the yield block) and one load (at the resume entry).
        assert_eq!(layout.yield_saves.len(), 1);
        assert_eq!(layout.resume_loads.len(), 1);
        assert_eq!(layout.yield_saves[0].0, HirBlockId(0));
        // Load lives at resume_entries[1].
        assert_eq!(layout.resume_loads[0].0, layout.resume_entries[1]);
        // One slot, holding our `live` SSA value (via its LocalId mapping).
        assert_eq!(layout.yield_saves[0].1.len(), 1);
        let saved_local = layout.yield_saves[0].1[0].1;
        assert_eq!(liveness.local_to_hir[&saved_local], live);
    }

    #[test]
    fn liveness_map_records_every_await_site_with_live_out_values() {
        // Function with two Await sites in entry block, no actual
        // control-flow complexity. live_out is fed in directly.
        // `live_a` and `live_b` are function params so the intra-block
        // captures-lift sees them as defined-before-suspension.
        let mut f = empty_function("test");
        let entry = f.entry_block;
        let live_a = HirId::new();
        let live_b = HirId::new();
        for (i, (id, name)) in [(live_a, "live_a"), (live_b, "live_b")]
            .into_iter()
            .enumerate()
        {
            f.values.insert(
                id,
                HirValue {
                    id,
                    ty: HirType::I32,
                    kind: HirValueKind::Parameter(i as u32),
                    uses: HashSet::new(),
                    span: None,
                },
            );
            f.signature.params.push(zyntax_compiler::hir::HirParam {
                id,
                name: InternedString::new_global(name),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            });
        }
        // Plant two Intrinsic::Await calls and one regular Call between them.
        let entry_block = f.blocks.get_mut(&entry).unwrap();
        entry_block.instructions.push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Intrinsic(Intrinsic::Await),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        entry_block.instructions.push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Function(HirId::new()),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        entry_block.instructions.push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Intrinsic(Intrinsic::Await),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });

        // live_out = {live_a, live_b}
        let mut live_out = HashMap::new();
        let mut set = HashSet::new();
        set.insert(live_a);
        set.insert(live_b);
        live_out.insert(entry, set);

        let mut cfg = HirCoroCfg::new(&mut f);
        let liveness = HirLiveness::build(&mut cfg, &live_out);

        // Two await sites recorded, both at block 0.
        assert_eq!(liveness.map.at_site.len(), 2);
        assert_eq!(liveness.map.at_site[0].0, (HirBlockId(0), 0));
        assert_eq!(liveness.map.at_site[1].0, (HirBlockId(0), 2));

        // Each site has 2 live LocalIds (one per SSA HirId).
        assert_eq!(liveness.map.at_site[0].1.len(), 2);
        assert_eq!(liveness.map.at_site[1].1.len(), 2);

        // Same SSA HirIds → same LocalIds at both sites (stable mapping).
        assert_eq!(liveness.map.at_site[0].1, liveness.map.at_site[1].1);

        // Round-trip: each LocalId resolves back to one of the
        // original HirIds.
        for local in &liveness.map.at_site[0].1 {
            let hir = liveness.local_to_hir[local];
            assert!(hir == live_a || hir == live_b);
        }
    }

    #[test]
    fn classify_returns_none_for_regular_call() {
        let mut f = empty_function("test");
        let entry = f.entry_block;
        let entry_block = f.blocks.get_mut(&entry).unwrap();
        // A regular Call to some user function — not a suspension site.
        entry_block.instructions.push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Function(HirId::new()),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });

        let cfg = HirCoroCfg::new(&mut f);
        let suspending = HirSuspendingFns {
            suspending: HashSet::new(),
        };
        let hooks = HirAsyncHooks {
            suspending: &suspending,
        };
        assert!(hooks.classify(&cfg, HirBlockId(0), 0).is_none());
    }

    // ─────────────────────────────────────────────────────────────────
    // M3: PerformEffect-as-suspension-site tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn classify_returns_direct_yield_for_perform_effect() {
        // M3: a PerformEffect instruction is a suspension site, just
        // like Intrinsic::Await. Captures-lift needs to fire for live
        // values across the perform-site so resumable handlers see a
        // valid frame on resume.
        let mut f = empty_function("test");
        let entry = f.entry_block;
        let entry_block = f.blocks.get_mut(&entry).unwrap();
        entry_block
            .instructions
            .push(HirInstruction::PerformEffect {
                result: None,
                effect_id: HirId::new(),
                op_name: InternedString::new_global("op"),
                args: vec![],
                return_ty: HirType::Void,
            });

        let cfg = HirCoroCfg::new(&mut f);
        let suspending = HirSuspendingFns {
            suspending: HashSet::new(),
        };
        let hooks = HirAsyncHooks {
            suspending: &suspending,
        };
        let site = hooks.classify(&cfg, HirBlockId(0), 0);
        assert!(
            matches!(site, Some(SuspensionSite::DirectYield { .. })),
            "PerformEffect should classify as DirectYield"
        );
    }

    #[test]
    fn suspending_fns_seeds_effect_annotated_function() {
        // M3: a function with a non-empty `signature.effects` list (set
        // by M1.1's annotation extraction → HIR lowering) is a
        // suspending function in its own right — even with no body
        // calls. The handler may suspend the computation.
        let mut module = HirModule::new(InternedString::new_global("test"));
        let sig_effectful = HirFunctionSignature {
            params: vec![],
            returns: vec![],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![InternedString::new_global("Log")],
            is_pure: false,
        };
        let f = HirFunction::new(InternedString::new_global("perform_log"), sig_effectful);
        let f_id = f.id;
        module.functions.insert(f_id, f);

        let s = HirSuspendingFns::from_module(&module);
        assert!(
            s.is_suspending(f_id),
            "function with effects = [Log] should be seeded suspending"
        );
    }

    #[test]
    fn suspending_fns_seeds_function_with_perform_effect_inst() {
        // M3 defensive seed: even if the `effects` annotation is empty
        // (lost or stripped), the presence of a `PerformEffect` in the
        // body should make the function suspending — the perform-site
        // demands captures-lift regardless.
        let mut module = HirModule::new(InternedString::new_global("test"));
        let sig_plain = HirFunctionSignature {
            params: vec![],
            returns: vec![],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut f = HirFunction::new(InternedString::new_global("performs"), sig_plain);
        // Plant a PerformEffect on the entry block.
        let entry_id = f.entry_block;
        let entry = f.blocks.get_mut(&entry_id).unwrap();
        entry.instructions.push(HirInstruction::PerformEffect {
            result: None,
            effect_id: HirId::new(),
            op_name: InternedString::new_global("op"),
            args: vec![],
            return_ty: HirType::Void,
        });
        let f_id = f.id;
        module.functions.insert(f_id, f);

        let s = HirSuspendingFns::from_module(&module);
        assert!(
            s.is_suspending(f_id),
            "function containing PerformEffect should be seeded suspending"
        );
    }

    #[test]
    fn liveness_map_records_perform_effect_site() {
        // M3: HirLiveness::build must enumerate PerformEffect alongside
        // Intrinsic::Await. Live SSA values across a perform-site get
        // mapped to LocalIds and recorded in the LivenessMap.
        let mut f = empty_function("test");
        let entry = f.entry_block;
        let live = HirId::new();
        f.values.insert(
            live,
            HirValue {
                id: live,
                ty: HirType::I32,
                kind: HirValueKind::Parameter(0),
                uses: HashSet::new(),
                span: None,
            },
        );
        f.signature.params.push(zyntax_compiler::hir::HirParam {
            id: live,
            name: InternedString::new_global("live"),
            ty: HirType::I32,
            attributes: ParamAttributes::default(),
        });
        let entry_block = f.blocks.get_mut(&entry).unwrap();
        entry_block
            .instructions
            .push(HirInstruction::PerformEffect {
                result: None,
                effect_id: HirId::new(),
                op_name: InternedString::new_global("get"),
                args: vec![],
                return_ty: HirType::I32,
            });

        // live_out makes `live` cross the perform-site.
        let mut live_out = HashMap::new();
        let mut set = HashSet::new();
        set.insert(live);
        live_out.insert(entry, set);

        let mut cfg = HirCoroCfg::new(&mut f);
        let liveness = HirLiveness::build(&mut cfg, &live_out);

        assert_eq!(
            liveness.map.at_site.len(),
            1,
            "one PerformEffect site recorded"
        );
        assert_eq!(liveness.map.at_site[0].0, (HirBlockId(0), 0));
        assert_eq!(
            liveness.map.at_site[0].1.len(),
            1,
            "the live `live` param is captured"
        );
        let saved_local = liveness.map.at_site[0].1[0];
        assert_eq!(liveness.local_to_hir[&saved_local], live);
    }

    #[test]
    fn redirect_targets_rewrites_branch() {
        let mut f = empty_function("test");
        let entry = f.entry_block;
        // entry → from_block, both placed in function.blocks
        let from_id = HirId::new();
        let to_id = HirId::new();
        for id in [from_id, to_id] {
            f.blocks.insert(
                id,
                HirBlock {
                    id,
                    label: None,
                    phis: vec![],
                    instructions: vec![],
                    terminator: HirTerminator::Unreachable,
                    dominance_frontier: HashSet::new(),
                    predecessors: vec![],
                    successors: vec![],
                },
            );
        }
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: from_id };

        let mut cfg = HirCoroCfg::new(&mut f);
        let from_bb = cfg.hir_to_block_id(from_id).unwrap();
        let to_bb = cfg.hir_to_block_id(to_id).unwrap();
        cfg.redirect_targets(HirBlockId(0), from_bb, to_bb);
        assert!(matches!(
            cfg.function.blocks[&entry].terminator,
            HirTerminator::Branch { target } if target == to_id
        ));
    }
}
