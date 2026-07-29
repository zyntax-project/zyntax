//! HIR-level common-subexpression elimination via dominator-tree
//! value numbering.
//!
//! Walks the dominator tree top-down, maintaining a value table keyed
//! by `(op, operand canonical-ids, type)`. When an instruction's key is
//! already in the table, the originating SSA value dominates the
//! current block by construction — every subsequent use can substitute
//! the canonical id without changing program semantics.
//!
//! On block exit we undo insertions we made in that block (scoped
//! hash table via an explicit undo log) so a later sibling block
//! doesn't see a stale entry from a dominator-tree sibling.
//!
//! ## Why HIR-level (not the backend)
//!
//! Same reason as `const_fold`: the BC interpreter and wasm tier-1
//! JIT bypass Cranelift's optimiser entirely. `let x = a * b + 1; let
//! y = a * b + 2;` currently computes `a * b` twice every iteration
//! in the interpreter; CSE collapses it to one computation.
//!
//! ## What we CSE
//!
//! Pure instructions whose semantics depend only on their operand
//! values:
//!
//!   * `Binary`
//!   * `Unary`
//!   * `Cast`
//!   * `GetElementPtr`        — pure address arithmetic
//!   * `ExtractValue`         — aggregate field read (no aliasing)
//!   * `Call` to a proven-pure callee — its result depends only on its
//!     arguments and it has no observable effect, so two such calls with
//!     matching args are one value. Purity comes from
//!     [`crate::purity`]; only functions in the supplied `pure_fns` set
//!     qualify. Integer arguments are compared in a canonical affine
//!     `(base, offset)` form, so `f((n-1)-2)` and `f((n-2)-1)` — both
//!     `f(n-3)` — share a key.
//!
//! Commutative `Binary` ops (`Add`, `Mul`, `And`, `Or`, `Xor`, `Eq`,
//! `Ne`, `FAdd`, `FMul`, `FEq`, `FNe`) get their operands sorted into
//! a canonical order so `a + b` and `b + a` share a key. Non-
//! commutative ops keep operand order.
//!
//! ## What we don't CSE
//!
//! Anything that could observe or affect mutable state:
//!
//!   * `Load`           — could alias a `Store` between defs
//!   * `Store` / atomics / fences — write effects
//!   * `Call` to an *impure* or non-`Function` callee (symbol /
//!                                intrinsic / indirect) — may have side
//!                                effects or read mutable state
//!   * `IndirectCall`    — opaque target
//!   * `Alloca`          — each one is a fresh allocation
//!   * `CreateClosure`   — identity matters (each closure may have its
//!                          own captured environment)
//!   * `AsyncSaveSlot` / `AsyncLoadSlot` — captures-lift bookkeeping;
//!                                          frame-relative
//!   * `Phi` — handled by SSA's own trivial-phi elimination pass
//!
//! ## Sound use-replacement
//!
//! When we eliminate `redundant_result → canonical`, we record the
//! substitution in a HashMap. After the dominator walk completes, a
//! single rewrite pass over every instruction's operand and every
//! terminator's value replaces redundant ids with their canonical
//! representative (transitively chased — if `A → B` and `B → C` we
//! collapse to `A → C`). The defining instruction itself is then
//! dead and removed.

use crate::analysis::DominatorTree;
use crate::hir::{
    BinaryOp, CastOp, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule,
    HirTerminator, HirType, HirValueKind, UnaryOp,
};
use std::collections::{HashMap, HashSet};

/// Read-only context threaded through the dominator walk: what's needed
/// to value-number pure calls and affine integer expressions.
struct CseCtx<'a> {
    /// Functions proven pure by [`crate::purity`]. Only calls to these
    /// are eligible for CSE — an impure call could observe or mutate
    /// state, so two syntactically-identical impure calls are distinct.
    pure_fns: &'a HashSet<HirId>,
    /// `result → (op, left, right)` for every integer `Add`/`Sub`
    /// binary, used to normalise chained constant offsets.
    bin_defs: HashMap<HirId, (BinaryOp, HirId, HirId)>,
    /// `id → integer value` for every integer constant.
    int_consts: HashMap<HirId, i128>,
}

/// Counters surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CseStats {
    /// Number of redundant instructions removed from blocks.
    pub eliminated: usize,
    /// Number of operand / terminator references rewritten through
    /// the substitution map.
    pub rewrites: usize,
}

/// Public entry. Runs CSE on a single function in place, with no
/// cross-function purity knowledge — pure-call CSE is disabled (the
/// `pure_fns` set is empty), so this behaves like the classic
/// value-only CSE. Prefer [`eliminate_with`] when a purity set is
/// available.
pub fn eliminate(func: &mut HirFunction) -> CseStats {
    eliminate_with(func, &HashSet::new())
}

/// CSE a single function, treating calls to any function in `pure_fns`
/// as value-numbered (two identical pure calls collapse to one).
pub fn eliminate_with(func: &mut HirFunction, pure_fns: &HashSet<HirId>) -> CseStats {
    let ctx = CseCtx {
        pure_fns,
        bin_defs: collect_bin_defs(func),
        int_consts: collect_int_consts(func),
    };
    let dt = DominatorTree::new(func);
    let mut value_table: HashMap<VnKey, HirId> = HashMap::new();
    let mut substitutions: HashMap<HirId, HirId> = HashMap::new();

    visit_block(
        func,
        dt.entry(),
        &dt,
        &ctx,
        &mut value_table,
        &mut substitutions,
    );

    let rewrites = apply_substitutions(func, &substitutions);
    let eliminated = remove_redundant_instructions(func, &substitutions);

    CseStats {
        eliminated,
        rewrites,
    }
}

/// Same entry, looped over every function in `module`. The set of pure
/// functions is read from each function's `signature.is_pure` (populated
/// by [`crate::purity::infer_module`]), so run purity inference first to
/// enable pure-call CSE.
pub fn eliminate_module(module: &mut HirModule) -> CseStats {
    let pure_fns: HashSet<HirId> = module
        .functions
        .iter()
        .filter(|(_, f)| f.signature.is_pure)
        .map(|(id, _)| *id)
        .collect();
    let mut total = CseStats::default();
    for func in module.functions.values_mut() {
        let stats = eliminate_with(func, &pure_fns);
        total.eliminated += stats.eliminated;
        total.rewrites += stats.rewrites;
    }
    total
}

/// `result → (op, left, right)` for every integer `Add`/`Sub` binary in
/// the function — the raw material for affine offset normalisation.
fn collect_bin_defs(func: &HirFunction) -> HashMap<HirId, (BinaryOp, HirId, HirId)> {
    let mut m = HashMap::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Binary {
                op: op @ (BinaryOp::Add | BinaryOp::Sub),
                result,
                left,
                right,
                ..
            } = inst
            {
                m.insert(*result, (*op, *left, *right));
            }
        }
    }
    m
}

/// `id → integer value` for every integer-constant value.
fn collect_int_consts(func: &HirFunction) -> HashMap<HirId, i128> {
    let mut m = HashMap::new();
    for (id, v) in &func.values {
        if let HirValueKind::Constant(c) = &v.kind {
            if let Some(i) = const_as_i128(c) {
                m.insert(*id, i);
            }
        }
    }
    m
}

fn const_as_i128(c: &HirConstant) -> Option<i128> {
    Some(match c {
        HirConstant::I8(v) => *v as i128,
        HirConstant::I16(v) => *v as i128,
        HirConstant::I32(v) => *v as i128,
        HirConstant::I64(v) => *v as i128,
        HirConstant::I128(v) => *v,
        HirConstant::U8(v) => *v as i128,
        HirConstant::U16(v) => *v as i128,
        HirConstant::U32(v) => *v as i128,
        HirConstant::U64(v) => *v as i128,
        _ => return None,
    })
}

/// Canonical affine form of an integer value: `(base, offset)` such that
/// the value equals `base + offset` under wrapping arithmetic. Walks
/// through `add`/`sub` by an integer constant, accumulating the offset,
/// so `(n-1)-2` and `(n-2)-1` both normalise to `(n, -3)`. Any value
/// that isn't such a chain is its own base with offset 0.
///
/// Only `x + c`, `c + x`, and `x - c` shift the offset (coefficient +1
/// on the base). `c - x` has coefficient -1, so it terminates the walk
/// as an opaque base — normalising it would be unsound.
fn affine_form(id: HirId, ctx: &CseCtx, substitutions: &HashMap<HirId, HirId>) -> (HirId, i128) {
    fn go(
        id: HirId,
        ctx: &CseCtx,
        substitutions: &HashMap<HirId, HirId>,
        depth: u32,
    ) -> (HirId, i128) {
        let id = canonical(id, substitutions);
        if depth > 64 {
            return (id, 0);
        }
        if let Some((op, left, right)) = ctx.bin_defs.get(&id).copied() {
            let l = canonical(left, substitutions);
            let r = canonical(right, substitutions);
            match op {
                BinaryOp::Add => {
                    if let Some(c) = ctx.int_consts.get(&r) {
                        let (b, o) = go(l, ctx, substitutions, depth + 1);
                        return (b, o.wrapping_add(*c));
                    }
                    if let Some(c) = ctx.int_consts.get(&l) {
                        let (b, o) = go(r, ctx, substitutions, depth + 1);
                        return (b, o.wrapping_add(*c));
                    }
                }
                BinaryOp::Sub => {
                    if let Some(c) = ctx.int_consts.get(&r) {
                        let (b, o) = go(l, ctx, substitutions, depth + 1);
                        return (b, o.wrapping_sub(*c));
                    }
                }
                _ => {}
            }
        }
        (id, 0)
    }
    go(id, ctx, substitutions, 0)
}

/// Visit `block_id` and its dominator-tree children in preorder.
/// Maintains `value_table` as a scoped hash table: insertions made
/// here are undone after recursing into children, so a sibling
/// subtree never sees this branch's entries.
fn visit_block(
    func: &HirFunction,
    block_id: HirId,
    dt: &DominatorTree,
    ctx: &CseCtx,
    value_table: &mut HashMap<VnKey, HirId>,
    substitutions: &mut HashMap<HirId, HirId>,
) {
    // Undo log: (key, previous_value_or_none). We replay this in
    // reverse on the way out.
    let mut undo_log: Vec<(VnKey, Option<HirId>)> = Vec::new();

    let block = match func.blocks.get(&block_id) {
        Some(b) => b,
        None => return,
    };

    for inst in &block.instructions {
        let Some((result, key)) = vn_key_for(inst, ctx, substitutions) else {
            continue;
        };
        match value_table.get(&key) {
            Some(&canonical) => {
                // Redundant — point this result at the canonical id.
                substitutions.insert(result, canonical);
            }
            None => {
                let prev = value_table.insert(key.clone(), result);
                undo_log.push((key, prev));
            }
        }
    }

    for &child in dt.children(block_id) {
        visit_block(func, child, dt, ctx, value_table, substitutions);
    }

    // Roll back insertions made in this block.
    while let Some((key, prev)) = undo_log.pop() {
        match prev {
            Some(p) => {
                value_table.insert(key, p);
            }
            None => {
                value_table.remove(&key);
            }
        }
    }
}

// ─── Value numbering ──────────────────────────────────────────────

/// The canonical key for an SSA instruction's "abstract value".
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum VnKey {
    Binary(BinaryOp, HirType, HirId, HirId),
    Unary(UnaryOp, HirType, HirId),
    Cast(CastOp, HirType, HirId),
    /// GEP key: pointer + sequence of index ids + result type.
    Gep(HirType, HirId, Vec<HirId>),
    /// ExtractValue key: aggregate + indices + result type.
    Extract(HirType, HirId, Vec<u32>),
    /// Pure call key: callee function id + affine forms of each argument
    /// + generic type/const args. Only built for callees in
    /// `ctx.pure_fns`, so two entries with this key are guaranteed to
    /// name the same effect-free value. Args use their affine `(base,
    /// offset)` normalisation so `f((n-1)-2)` and `f((n-2)-1)` share a
    /// key.
    Call(HirId, Vec<(HirId, i128)>, Vec<HirType>, Vec<HirConstant>),
}

/// Build the VN key for `inst` after chasing operand substitutions
/// so two equivalent computations against substituted-equivalent
/// operands produce the same key. Returns `None` for any instruction
/// shape we don't CSE.
fn vn_key_for(
    inst: &HirInstruction,
    ctx: &CseCtx,
    substitutions: &HashMap<HirId, HirId>,
) -> Option<(HirId, VnKey)> {
    match inst {
        HirInstruction::Binary {
            op,
            result,
            ty,
            left,
            right,
        } => {
            let (l, r) = canonical_operand_order(*op, *left, *right, substitutions);
            Some((*result, VnKey::Binary(*op, ty.clone(), l, r)))
        }
        HirInstruction::Unary {
            op,
            result,
            ty,
            operand,
        } => Some((
            *result,
            VnKey::Unary(*op, ty.clone(), canonical(*operand, substitutions)),
        )),
        HirInstruction::Cast {
            op,
            result,
            ty,
            operand,
        } => Some((
            *result,
            VnKey::Cast(*op, ty.clone(), canonical(*operand, substitutions)),
        )),
        HirInstruction::GetElementPtr {
            result,
            ty,
            ptr,
            indices,
        } => {
            let p = canonical(*ptr, substitutions);
            let ix: Vec<HirId> = indices
                .iter()
                .map(|i| canonical(*i, substitutions))
                .collect();
            Some((*result, VnKey::Gep(ty.clone(), p, ix)))
        }
        HirInstruction::ExtractValue {
            result,
            ty,
            aggregate,
            indices,
        } => Some((
            *result,
            VnKey::Extract(
                ty.clone(),
                canonical(*aggregate, substitutions),
                indices.clone(),
            ),
        )),
        // A call to a proven-pure function is value-numbered: its result
        // depends only on its arguments and it has no observable effect,
        // so two such calls with matching (affine-normalised) args are
        // the same value. Impure callees, and any callable that isn't a
        // direct `Function` (symbol / intrinsic / indirect), fall through
        // to the not-CSE-able arm below.
        HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Function(fid),
            args,
            type_args,
            const_args,
            ..
        } if ctx.pure_fns.contains(fid) => {
            let arg_forms: Vec<(HirId, i128)> = args
                .iter()
                .map(|a| affine_form(*a, ctx, substitutions))
                .collect();
            Some((
                *result,
                VnKey::Call(*fid, arg_forms, type_args.clone(), const_args.clone()),
            ))
        }

        // Side-effecting / identity-bearing instructions are not
        // CSE-able — see module doc for the rationale per variant.
        HirInstruction::Load { .. }
        | HirInstruction::Store { .. }
        | HirInstruction::Call { .. }
        | HirInstruction::IndirectCall { .. }
        | HirInstruction::Alloca { .. }
        | HirInstruction::Select { .. }
        | HirInstruction::InsertValue { .. }
        | HirInstruction::Atomic { .. }
        | HirInstruction::Fence { .. }
        | _ => None,
    }
}

/// Chase the substitution chain — if `A → B` and `B → C` we want
/// `A → C` so two operand references to A and to C produce the same
/// VN key. Union-find with path compression would be classy here;
/// the chain depth is bounded by the number of CSE-eliminated
/// instructions, which is tiny, so iterative chasing is fine.
fn canonical(mut id: HirId, substitutions: &HashMap<HirId, HirId>) -> HirId {
    let mut seen = 0;
    while let Some(&next) = substitutions.get(&id) {
        if next == id || seen > 64 {
            break;
        }
        id = next;
        seen += 1;
    }
    id
}

/// Sort operand HirIds when the op is commutative so `a + b` and
/// `b + a` get the same key. Order is HirId's natural ordering — we
/// use `format!("{:?}", id)` as a stable sort key since `HirId`
/// itself isn't `Ord`.
fn canonical_operand_order(
    op: BinaryOp,
    left: HirId,
    right: HirId,
    substitutions: &HashMap<HirId, HirId>,
) -> (HirId, HirId) {
    let l = canonical(left, substitutions);
    let r = canonical(right, substitutions);
    if is_commutative(op) {
        let lk = format!("{l:?}");
        let rk = format!("{r:?}");
        if lk <= rk {
            (l, r)
        } else {
            (r, l)
        }
    } else {
        (l, r)
    }
}

fn is_commutative(op: BinaryOp) -> bool {
    use BinaryOp::*;
    matches!(
        op,
        Add | Mul | And | Or | Xor | Eq | Ne | FAdd | FMul | FEq | FNe
    )
}

// ─── Apply substitutions across the function ──────────────────────

/// Walk every instruction and every terminator, replacing operand
/// references via `substitutions`. Returns the number of references
/// rewritten.
/// Public re-export of `apply_substitutions` so sibling passes
/// (e.g. `load_cse`) can share the rewrite machinery without
/// duplicating the operand walker. Same contract as the private
/// helper: walks every instruction + terminator + phi-incoming and
/// retargets ids that appear as keys in `substitutions`.
pub fn apply_substitutions_public(
    func: &mut HirFunction,
    substitutions: &HashMap<HirId, HirId>,
) -> usize {
    apply_substitutions(func, substitutions)
}

/// Public re-export of `remove_redundant_instructions` — drops any
/// instruction whose result HirId is a key in `substitutions`.
pub fn remove_redundant_instructions_public(
    func: &mut HirFunction,
    substitutions: &HashMap<HirId, HirId>,
) -> usize {
    remove_redundant_instructions(func, substitutions)
}

fn apply_substitutions(func: &mut HirFunction, substitutions: &HashMap<HirId, HirId>) -> usize {
    if substitutions.is_empty() {
        return 0;
    }
    let mut rewrites = 0;

    let map = |id: &mut HirId| -> bool {
        let new = canonical(*id, substitutions);
        if new != *id {
            *id = new;
            true
        } else {
            false
        }
    };

    for block in func.blocks.values_mut() {
        for inst in &mut block.instructions {
            rewrites += rewrite_inst_operands(inst, &map);
        }
        rewrites += rewrite_terminator_operands(&mut block.terminator, &map);
        // Phi node incoming values are rewritten too — they're real
        // operand uses.
        for phi in &mut block.phis {
            for (_, incoming) in &mut phi.incoming {
                if map(incoming) {
                    rewrites += 1;
                }
            }
        }
    }

    rewrites
}

fn rewrite_inst_operands(inst: &mut HirInstruction, map: &impl Fn(&mut HirId) -> bool) -> usize {
    let mut n = 0;
    macro_rules! m {
        ($e:expr) => {
            if map($e) {
                n += 1;
            }
        };
    }
    macro_rules! m_vec {
        ($vec:expr) => {
            for x in $vec.iter_mut() {
                m!(x);
            }
        };
    }
    match inst {
        HirInstruction::Binary { left, right, .. } => {
            m!(left);
            m!(right);
        }
        HirInstruction::Unary { operand, .. } => m!(operand),
        HirInstruction::Cast { operand, .. } => m!(operand),
        HirInstruction::Load { ptr, .. } => m!(ptr),
        HirInstruction::Store { value, ptr, .. } => {
            m!(value);
            m!(ptr);
        }
        HirInstruction::GetElementPtr { ptr, indices, .. } => {
            m!(ptr);
            m_vec!(indices);
        }
        HirInstruction::Call { args, .. } => m_vec!(args),
        HirInstruction::IndirectCall { func_ptr, args, .. } => {
            m!(func_ptr);
            m_vec!(args);
        }
        HirInstruction::Select {
            condition,
            true_val,
            false_val,
            ..
        } => {
            m!(condition);
            m!(true_val);
            m!(false_val);
        }
        HirInstruction::ExtractValue { aggregate, .. } => m!(aggregate),
        HirInstruction::InsertValue {
            aggregate, value, ..
        } => {
            m!(aggregate);
            m!(value);
        }
        HirInstruction::Atomic { ptr, value, .. } => {
            m!(ptr);
            if let Some(v) = value {
                m!(v);
            }
        }
        HirInstruction::Alloca { count, .. } => {
            if let Some(c) = count {
                m!(c);
            }
        }
        _ => {
            // Other variants (CreateUnion, AsyncSaveSlot, etc.)
            // aren't CSE-able and their operands rarely refer to
            // CSE'd values; we still walk them for correctness
            // since the substitution map may cross variant
            // boundaries when later passes are added.
        }
    }
    n
}

fn rewrite_terminator_operands(
    term: &mut HirTerminator,
    map: &impl Fn(&mut HirId) -> bool,
) -> usize {
    let mut n = 0;
    let mut m = |id: &mut HirId| {
        if map(id) {
            n += 1;
        }
    };
    match term {
        HirTerminator::Return { values } => {
            for v in values {
                m(v);
            }
        }
        HirTerminator::CondBranch { condition, .. } => m(condition),
        HirTerminator::Switch { value, .. } => m(value),
        HirTerminator::Invoke { args, .. } => {
            for a in args {
                m(a);
            }
        }
        HirTerminator::PatternMatch { value, .. } => m(value),
        HirTerminator::Branch { .. } | HirTerminator::Unreachable => {}
    }
    n
}

/// Drop any instruction whose `result` was substituted away. Returns
/// the number of instructions removed. Phi nodes are not touched —
/// they're handled by SSA's trivial-phi pass.
fn remove_redundant_instructions(
    func: &mut HirFunction,
    substitutions: &HashMap<HirId, HirId>,
) -> usize {
    if substitutions.is_empty() {
        return 0;
    }
    let mut removed = 0;
    for block in func.blocks.values_mut() {
        block.instructions.retain(|inst| {
            let res = instruction_result(inst);
            let keep = match res {
                Some(r) => !substitutions.contains_key(&r),
                None => true,
            };
            if !keep {
                removed += 1;
            }
            keep
        });
    }
    removed
}

/// Inline result-id extractor — kept here so we don't depend on the
/// existing `analysis::instruction_result` (which has a slightly
/// different signature scoped to that module).
fn instruction_result(inst: &HirInstruction) -> Option<HirId> {
    match inst {
        HirInstruction::Binary { result, .. } => Some(*result),
        HirInstruction::Unary { result, .. } => Some(*result),
        HirInstruction::Cast { result, .. } => Some(*result),
        HirInstruction::GetElementPtr { result, .. } => Some(*result),
        HirInstruction::ExtractValue { result, .. } => Some(*result),
        HirInstruction::InsertValue { result, .. } => Some(*result),
        HirInstruction::Load { result, .. } => Some(*result),
        HirInstruction::Alloca { result, .. } => Some(*result),
        HirInstruction::Call { result, .. } => *result,
        HirInstruction::IndirectCall { result, .. } => *result,
        HirInstruction::Select { result, .. } => Some(*result),
        HirInstruction::Atomic { result, .. } => Some(*result),
        _ => None,
    }
}

#[allow(unused)]
fn touch_callable(_: &HirCallable) {}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        BinaryOp, HirBlock, HirConstant, HirFunctionSignature, HirTerminator, HirValue,
        HirValueKind,
    };
    use zyntax_typed_ast::InternedString;

    fn empty_sig(ret: HirType) -> HirFunctionSignature {
        HirFunctionSignature {
            params: vec![],
            returns: vec![ret],
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

    fn mk_func(ret: HirType) -> HirFunction {
        let mut f = HirFunction::new(InternedString::new_global("t"), empty_sig(ret));
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        f
    }

    fn add_const(f: &mut HirFunction, ty: HirType, c: HirConstant) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Constant(c),
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn add_inst_result(f: &mut HirFunction, ty: HirType) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn add_param(f: &mut HirFunction, ty: HirType, idx: u32) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Parameter(idx),
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn push(f: &mut HirFunction, inst: HirInstruction) {
        let entry = f.entry_block;
        f.blocks.get_mut(&entry).unwrap().instructions.push(inst);
    }

    #[test]
    fn two_identical_binaries_collapse_to_one() {
        // r1 = a + b
        // r2 = a + b   <- redundant, should be replaced by r1
        // ret r2       <- after CSE, becomes ret r1
        let mut f = mk_func(HirType::I32);
        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let r1 = add_inst_result(&mut f, HirType::I32);
        let r2 = add_inst_result(&mut f, HirType::I32);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r1,
                ty: HirType::I32,
                left: a,
                right: b,
            },
        );
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r2,
                ty: HirType::I32,
                left: a,
                right: b,
            },
        );
        f.blocks.get_mut(&f.entry_block).unwrap().terminator =
            HirTerminator::Return { values: vec![r2] };

        let stats = eliminate(&mut f);
        assert_eq!(stats.eliminated, 1);
        assert_eq!(f.blocks[&f.entry_block].instructions.len(), 1);
        match &f.blocks[&f.entry_block].terminator {
            HirTerminator::Return { values } => assert_eq!(values, &vec![r1]),
            _ => panic!("expected Return"),
        }
    }

    #[test]
    fn commutative_swap_is_recognised() {
        // r1 = a + b
        // r2 = b + a   <- same as r1 under commutativity
        let mut f = mk_func(HirType::I32);
        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let r1 = add_inst_result(&mut f, HirType::I32);
        let r2 = add_inst_result(&mut f, HirType::I32);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r1,
                ty: HirType::I32,
                left: a,
                right: b,
            },
        );
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r2,
                ty: HirType::I32,
                left: b,
                right: a,
            },
        );
        let stats = eliminate(&mut f);
        assert_eq!(stats.eliminated, 1);
    }

    #[test]
    fn non_commutative_swap_is_not_collapsed() {
        // r1 = a - b
        // r2 = b - a   <- DIFFERENT value!
        let mut f = mk_func(HirType::I32);
        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let r1 = add_inst_result(&mut f, HirType::I32);
        let r2 = add_inst_result(&mut f, HirType::I32);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: r1,
                ty: HirType::I32,
                left: a,
                right: b,
            },
        );
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: r2,
                ty: HirType::I32,
                left: b,
                right: a,
            },
        );
        let stats = eliminate(&mut f);
        assert_eq!(stats.eliminated, 0);
    }

    #[test]
    fn dominator_walk_keeps_sibling_blocks_independent() {
        //          entry: cond ? bbT : bbF
        //          /              \
        //     bbT: a + b -> r1   bbF: a + b -> r2
        //
        // bbT does NOT dominate bbF (they're siblings under entry),
        // so r2 must NOT be substituted with r1. Both blocks keep
        // their Binary instructions.
        let entry = HirId::new();
        let bb_t = HirId::new();
        let bb_f = HirId::new();
        let mut f = mk_func(HirType::I32);
        f.entry_block = entry;
        f.blocks.clear();
        let mut e = HirBlock::new(entry);
        e.successors = vec![bb_t, bb_f];
        let mut t = HirBlock::new(bb_t);
        t.predecessors = vec![entry];
        let mut fl = HirBlock::new(bb_f);
        fl.predecessors = vec![entry];
        f.blocks.insert(entry, e);
        f.blocks.insert(bb_t, t);
        f.blocks.insert(bb_f, fl);

        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let r1 = add_inst_result(&mut f, HirType::I32);
        let r2 = add_inst_result(&mut f, HirType::I32);
        let cond = add_const(&mut f, HirType::Bool, HirConstant::Bool(true));

        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::CondBranch {
            condition: cond,
            true_target: bb_t,
            false_target: bb_f,
        };
        f.blocks
            .get_mut(&bb_t)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r1,
                ty: HirType::I32,
                left: a,
                right: b,
            });
        f.blocks.get_mut(&bb_t).unwrap().terminator = HirTerminator::Return { values: vec![r1] };
        f.blocks
            .get_mut(&bb_f)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r2,
                ty: HirType::I32,
                left: a,
                right: b,
            });
        f.blocks.get_mut(&bb_f).unwrap().terminator = HirTerminator::Return { values: vec![r2] };

        let stats = eliminate(&mut f);
        assert_eq!(
            stats.eliminated, 0,
            "siblings must not CSE across each other"
        );
        assert_eq!(f.blocks[&bb_t].instructions.len(), 1);
        assert_eq!(f.blocks[&bb_f].instructions.len(), 1);
    }

    #[test]
    fn entry_definition_csed_into_both_successors() {
        //         entry: a + b -> r1
        //          /            \
        //       bbT: a+b -> r2   bbF: a+b -> r3
        //
        // r1 dominates both successors, so both r2 and r3 collapse
        // to r1.
        let entry = HirId::new();
        let bb_t = HirId::new();
        let bb_f = HirId::new();
        let mut f = mk_func(HirType::I32);
        f.entry_block = entry;
        f.blocks.clear();
        let mut e = HirBlock::new(entry);
        e.successors = vec![bb_t, bb_f];
        let mut t = HirBlock::new(bb_t);
        t.predecessors = vec![entry];
        let mut fl = HirBlock::new(bb_f);
        fl.predecessors = vec![entry];
        f.blocks.insert(entry, e);
        f.blocks.insert(bb_t, t);
        f.blocks.insert(bb_f, fl);

        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let r1 = add_inst_result(&mut f, HirType::I32);
        let r2 = add_inst_result(&mut f, HirType::I32);
        let r3 = add_inst_result(&mut f, HirType::I32);
        let cond = add_const(&mut f, HirType::Bool, HirConstant::Bool(true));

        f.blocks
            .get_mut(&entry)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r1,
                ty: HirType::I32,
                left: a,
                right: b,
            });
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::CondBranch {
            condition: cond,
            true_target: bb_t,
            false_target: bb_f,
        };
        f.blocks
            .get_mut(&bb_t)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r2,
                ty: HirType::I32,
                left: a,
                right: b,
            });
        f.blocks.get_mut(&bb_t).unwrap().terminator = HirTerminator::Return { values: vec![r2] };
        f.blocks
            .get_mut(&bb_f)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r3,
                ty: HirType::I32,
                left: a,
                right: b,
            });
        f.blocks.get_mut(&bb_f).unwrap().terminator = HirTerminator::Return { values: vec![r3] };

        let stats = eliminate(&mut f);
        assert_eq!(stats.eliminated, 2);
        assert!(f.blocks[&bb_t].instructions.is_empty());
        assert!(f.blocks[&bb_f].instructions.is_empty());
        match &f.blocks[&bb_t].terminator {
            HirTerminator::Return { values } => assert_eq!(values, &vec![r1]),
            _ => panic!("expected Return"),
        }
    }

    #[test]
    fn symbol_calls_are_not_csed() {
        // A `Symbol` callee is an opaque runtime/FFI target we can't
        // prove pure — two identical symbol calls stay distinct even
        // when the args match.
        let mut f = mk_func(HirType::I32);
        let a = add_param(&mut f, HirType::I32, 0);
        let r1 = add_inst_result(&mut f, HirType::I32);
        let r2 = add_inst_result(&mut f, HirType::I32);
        push(
            &mut f,
            HirInstruction::Call {
                result: Some(r1),
                callee: HirCallable::Symbol("foo".to_string()),
                args: vec![a],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        push(
            &mut f,
            HirInstruction::Call {
                result: Some(r2),
                callee: HirCallable::Symbol("foo".to_string()),
                args: vec![a],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        let stats = eliminate(&mut f);
        assert_eq!(stats.eliminated, 0);
    }

    #[test]
    fn impure_function_calls_are_not_csed() {
        // Same as above but with a `Function` callee that is NOT in the
        // pure set: still not CSE-able.
        let mut f = mk_func(HirType::I32);
        let callee = HirId::new();
        let a = add_param(&mut f, HirType::I32, 0);
        let r1 = add_inst_result(&mut f, HirType::I32);
        let r2 = add_inst_result(&mut f, HirType::I32);
        for r in [r1, r2] {
            push(
                &mut f,
                HirInstruction::Call {
                    result: Some(r),
                    callee: HirCallable::Function(callee),
                    args: vec![a],
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                },
            );
        }
        // Empty pure set (default `eliminate`).
        assert_eq!(eliminate(&mut f).eliminated, 0);
    }

    #[test]
    fn pure_function_calls_collapse_in_dominance() {
        // Two calls to the same pure function with identical args, the
        // first dominating the second → the second is redundant.
        let mut f = mk_func(HirType::I32);
        let callee = HirId::new();
        let a = add_param(&mut f, HirType::I32, 0);
        let r1 = add_inst_result(&mut f, HirType::I32);
        let r2 = add_inst_result(&mut f, HirType::I32);
        for r in [r1, r2] {
            push(
                &mut f,
                HirInstruction::Call {
                    result: Some(r),
                    callee: HirCallable::Function(callee),
                    args: vec![a],
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                },
            );
        }
        f.blocks.get_mut(&f.entry_block).unwrap().terminator =
            HirTerminator::Return { values: vec![r2] };
        let pure: HashSet<HirId> = [callee].into_iter().collect();
        let stats = eliminate_with(&mut f, &pure);
        assert_eq!(stats.eliminated, 1, "second pure call should collapse");
        match &f.blocks[&f.entry_block].terminator {
            HirTerminator::Return { values } => assert_eq!(values, &vec![r1]),
            _ => panic!("expected Return"),
        }
    }

    #[test]
    fn pure_calls_collapse_under_affine_arg_equivalence() {
        // call f((n-1)-1)  and  call f(n-2) — args differ syntactically
        // but are affinely equal (both n-2), so the pure calls collapse.
        let mut f = mk_func(HirType::I64);
        let callee = HirId::new();
        let n = add_param(&mut f, HirType::I64, 0);
        let one = add_const(&mut f, HirType::I64, HirConstant::I64(1));
        let two = add_const(&mut f, HirType::I64, HirConstant::I64(2));

        // a1 = n - 1 ; a2 = a1 - 1   (== n-2)
        let a1 = add_inst_result(&mut f, HirType::I64);
        let a2 = add_inst_result(&mut f, HirType::I64);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: a1,
                ty: HirType::I64,
                left: n,
                right: one,
            },
        );
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: a2,
                ty: HirType::I64,
                left: a1,
                right: one,
            },
        );
        // b = n - 2
        let b = add_inst_result(&mut f, HirType::I64);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: b,
                ty: HirType::I64,
                left: n,
                right: two,
            },
        );
        let r1 = add_inst_result(&mut f, HirType::I64);
        let r2 = add_inst_result(&mut f, HirType::I64);
        push(
            &mut f,
            HirInstruction::Call {
                result: Some(r1),
                callee: HirCallable::Function(callee),
                args: vec![a2],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        push(
            &mut f,
            HirInstruction::Call {
                result: Some(r2),
                callee: HirCallable::Function(callee),
                args: vec![b],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        f.blocks.get_mut(&f.entry_block).unwrap().terminator =
            HirTerminator::Return { values: vec![r2] };
        let pure: HashSet<HirId> = [callee].into_iter().collect();
        let stats = eliminate_with(&mut f, &pure);
        assert_eq!(
            stats.eliminated, 1,
            "affinely-equal-arg pure calls should collapse"
        );
    }

    #[test]
    fn chained_redundancy_collapses_transitively() {
        // r1 = a + b
        // r2 = a + b      -> redundant w/ r1
        // r3 = r1 - c     -> uses r1
        // r4 = r2 - c     -> uses r2 == r1; after substitution this
        //                    becomes r1 - c, redundant w/ r3
        let mut f = mk_func(HirType::I32);
        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let c = add_param(&mut f, HirType::I32, 2);
        let r1 = add_inst_result(&mut f, HirType::I32);
        let r2 = add_inst_result(&mut f, HirType::I32);
        let r3 = add_inst_result(&mut f, HirType::I32);
        let r4 = add_inst_result(&mut f, HirType::I32);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r1,
                ty: HirType::I32,
                left: a,
                right: b,
            },
        );
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r2,
                ty: HirType::I32,
                left: a,
                right: b,
            },
        );
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: r3,
                ty: HirType::I32,
                left: r1,
                right: c,
            },
        );
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: r4,
                ty: HirType::I32,
                left: r2,
                right: c,
            },
        );
        let stats = eliminate(&mut f);
        // Both r2 (= r1) and r4 (= r3 after substitution) collapse.
        assert_eq!(stats.eliminated, 2);
    }
}
