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

use krio_async::{AsyncHooks, FnId as KrioFnId, SuspendingFns, SuspensionSite};
use krio_stackless::CoroCfg;
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

    // ── Internal helpers ─────────────────────────────────────────────

    fn fresh_local_id(&mut self, kind: LocalKind) -> HirLocalId {
        let id = HirLocalId(self.next_local_seq);
        self.next_local_seq += 1;
        self.locals.insert(id, kind);
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

    fn redirect_targets(
        &mut self,
        bb: Self::BlockId,
        from: Self::BlockId,
        to: Self::BlockId,
    ) {
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
    /// Algorithm: seed = `{ id | function(id).is_async }`; iterate to
    /// fixed point — a function becomes suspending if it Calls any
    /// suspending function (HirCallable::Function variant only;
    /// Indirect/Intrinsic calls don't taint).
    ///
    /// `Intrinsic::Await` is the suspension primitive but it isn't a
    /// `HirCallable::Function`, so it doesn't enter the fixpoint —
    /// `classify` recognises it directly.
    pub fn from_module(module: &HirModule) -> Self {
        let mut suspending: HashSet<HirFnId> = module
            .functions
            .values()
            .filter(|f| f.signature.is_async)
            .map(|f| f.id)
            .collect();

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

        // Pattern 2 (future work): direct cross-fn call to an async
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
        assert!(matches!(entry_block.instructions[0], HirInstruction::Alloca { .. }));
        assert!(matches!(entry_block.instructions[1], HirInstruction::Store { .. }));

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
        f.blocks.get_mut(&entry).unwrap().terminator =
            HirTerminator::Branch { target: from_id };

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
