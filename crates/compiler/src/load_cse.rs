//! Redundant-load elimination within single-entry regions.
//!
//! Plain `cse::eliminate` skips `Load` because Loads can alias with
//! `Store`s through a memory layer we don't model — two Loads with
//! the same pointer expression aren't equivalent if a Store happened
//! between them. This module fills the gap with the simplest correct
//! alias check we can ship:
//!
//!   * Walk each block top-to-bottom.
//!   * Track `Load { ptr, result }` entries in a per-block "available
//!     loads" map keyed by the pointer's canonical HirId (chasing
//!     substitutions through any prior CSE).
//!   * On a *memory-killing* instruction (`Call`, `IndirectCall`,
//!     `Atomic`, `Fence`), clear the map — we conservatively assume
//!     anything in memory could have changed. A `Store` kills every
//!     load except those proven disjoint by exact struct types or
//!     non-overlapping byte ranges relative to the same SSA pointer.
//!   * On a redundant `Load` (its pointer is already in the map),
//!     record a substitution from the new load result to the
//!     previously-seen load result.
//!
//! Carry available loads through edges whose successor has one predecessor.
//! Joins, loop entries and effectful terminators start with an empty map.
//! This exposes repeated checks along a success path without memory-state
//! intersections or a fixed-point dataflow analysis.
//!
//! After collecting substitutions we sweep the function with the
//! same machinery `cse::apply_substitutions` uses (rewrite operand
//! references, drop the now-orphaned defining Loads).

use crate::cse;
use crate::hir::{
    CastOp, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirTerminator, HirType,
    HirValueKind,
};
use crate::licm::{MemLoc, extract_mem_loc, hir_ty_byte_size, value_byte_size};
use fnv::FnvHashSet;
use std::collections::HashMap;
use zyntax_typed_ast::InternedString;

#[derive(Clone, Copy)]
struct Address {
    root: HirId,
    offset: u64,
}

/// Byte offsets only: struct layout, loaded pointers and phi identities
/// are not inferred. A distinct SSA root may still alias this one.
struct Addresses<'a> {
    function: &'a HirFunction,
    defs: HashMap<HirId, &'a HirInstruction>,
    cache: HashMap<HirId, Option<Address>>,
}

impl Addresses<'_> {
    fn stored_size(&self, value: HirId) -> u32 {
        let size = value_byte_size(self.function, value);
        match self.defs.get(&value) {
            Some(
                HirInstruction::Load { ty, .. }
                | HirInstruction::ExtractValue { ty, .. }
                | HirInstruction::InsertValue { ty, .. },
            ) => {
                // An aggregate's value metadata may describe its pointer carrier.
                let declared = hir_ty_byte_size(ty);
                if size == 0 || declared == 0 {
                    0
                } else {
                    size.max(declared)
                }
            }
            _ => size,
        }
    }

    fn get(&mut self, ptr: HirId, depth: u32) -> Option<Address> {
        if let Some(address) = self.cache.get(&ptr) {
            return *address;
        }
        if depth == 0 {
            return None;
        }
        let address = match self.defs.get(&ptr).copied() {
            Some(HirInstruction::Cast {
                op: CastOp::Bitcast | CastOp::PtrToInt | CastOp::IntToPtr,
                operand,
                ty,
                ..
            }) if matches!(
                ty,
                HirType::Ptr(_) | HirType::Ref { .. } | HirType::I64 | HirType::U64
            ) && (matches!(
                self.function.values.get(operand)?.ty,
                HirType::Ptr(_) | HirType::Ref { .. } | HirType::I64 | HirType::U64
            ) || matches!(
                self.defs.get(operand),
                Some(HirInstruction::GetElementPtr { .. })
            )) =>
            {
                self.get(*operand, depth - 1)
            }
            Some(HirInstruction::GetElementPtr {
                ty,
                ptr: base,
                indices,
                ..
            }) => {
                let offset = byte_offset(self.function, ty, indices);
                match offset {
                    Some(offset) => self.get(*base, depth - 1).map(|a| Address {
                        root: a.root,
                        offset: a.offset.wrapping_add(offset),
                    }),
                    None => Some(Address {
                        root: ptr,
                        offset: 0,
                    }),
                }
            }
            _ => Some(Address {
                root: ptr,
                offset: 0,
            }),
        };
        self.cache.insert(ptr, address);
        address
    }
}

fn byte_offset(function: &HirFunction, ty: &HirType, indices: &[HirId]) -> Option<u64> {
    // Only a single scalar index has the same stride in every backend.
    let [index] = indices else {
        return None;
    };
    let HirValueKind::Constant(c) = &function.values.get(index)?.kind else {
        return None;
    };
    let index = match c {
        HirConstant::I8(n) => *n as i64 as u64,
        HirConstant::I16(n) => *n as i64 as u64,
        HirConstant::I32(n) => *n as i64 as u64,
        HirConstant::I64(n) => *n as u64,
        HirConstant::U64(n) => *n,
        _ => return None,
    };
    let stride = match ty {
        HirType::I8 | HirType::U8 => 1,
        HirType::Ptr(inner) => {
            // Pointer-sized strides depend on the eventual backend target.
            if matches!(&**inner, HirType::Ptr(_) | HirType::Ref { .. }) {
                return None;
            }
            let size = hir_ty_byte_size(inner);
            if size == 0 {
                return None;
            }
            u64::from(size)
        }
        _ => return None,
    };
    Some(index.wrapping_mul(stride))
}

fn separate_bytes(a: Option<Address>, a_size: u32, b: Option<Address>, b_size: u32) -> bool {
    let (Some(a), Some(b)) = (a, b) else {
        return false;
    };
    if a.root != b.root || a_size == 0 || b_size == 0 {
        return false;
    }
    let (Some(a_end), Some(b_end)) = (
        a.offset.checked_add(u64::from(a_size)),
        b.offset.checked_add(u64::from(b_size)),
    ) else {
        return false;
    };
    // The proof must also hold when the target uses 32-bit addresses.
    if a_end > u64::from(u32::MAX) || b_end > u64::from(u32::MAX) {
        return false;
    }
    a_end <= b.offset || b_end <= a.offset
}

/// Public stats — same shape as `CseStats` so callers can compose.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LoadCseStats {
    /// Number of redundant Loads removed from blocks.
    pub eliminated: usize,
}

type Available<'a> = HashMap<HirId, (HirId, &'a HirType, MemLoc, u32)>;

/// A forest of single-predecessor edges. Cycles with no root are unreachable
/// and are processed locally. Edges come from terminators, not cached CFGs.
fn load_regions(func: &HirFunction, enabled: bool) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = func.blocks.len();
    if !enabled {
        return (Vec::new(), (0..n).collect());
    }
    let mut parent = vec![None; n];
    let mut count = vec![0; n];
    if enabled {
        for (i, block) in func.blocks.values().enumerate() {
            let mut targets = block.terminator.targets();
            targets.sort_unstable();
            targets.dedup();
            for target in targets {
                if let Some(j) = func.blocks.get_index_of(&target) {
                    count[j] += 1;
                    parent[j] = Some(i);
                }
            }
        }
    }
    let mut children = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for (i, (id, _)) in func.blocks.iter().enumerate() {
        let from = parent[i].filter(|p| {
            count[i] == 1
                && *id != func.entry_block
                && matches!(
                    func.blocks.get_index(*p).unwrap().1.terminator,
                    HirTerminator::Branch { .. }
                        | HirTerminator::CondBranch { .. }
                        | HirTerminator::Switch { .. }
                )
        });
        if let Some(from) = from {
            children[from].push(i);
        } else {
            roots.push(i);
        }
    }
    (children, roots)
}

/// Reuse loads within each block.
pub fn run(func: &mut HirFunction) -> LoadCseStats {
    run_with(func, &FnvHashSet::default(), false)
}

/// Reuse loads through single-entry successors after structural optimization.
pub fn run_regions(func: &mut HirFunction) -> LoadCseStats {
    run_with(func, &FnvHashSet::default(), true)
}

/// [`run`], knowing the exact struct types
/// (`licm::exact_struct_names`).
fn run_with(
    func: &mut HirFunction,
    exact: &FnvHashSet<InternedString>,
    across: bool,
) -> LoadCseStats {
    let mut substitutions: HashMap<HirId, HirId> = HashMap::new();
    let addr_index = crate::licm::build_addr_index(func, exact);
    let no_subst = indexmap::IndexMap::new();
    let loc = |ptr: HirId, size: u32| -> MemLoc {
        extract_mem_loc(func, ptr, size, &no_subst, &addr_index, exact)
    };
    // `ZYNTAX_DISABLE_LOAD_RANGES=1` retains reloads after disjoint stores; safe to run with.
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ranges =
        !*DISABLED.get_or_init(|| std::env::var_os("ZYNTAX_DISABLE_LOAD_RANGES").is_some());
    let mut addresses: Option<Addresses<'_>> = None;

    // `ZYNTAX_DISABLE_CROSS_BLOCK_LOADS=1` keeps reuse block-local; safe to run with.
    static LOCAL_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let across = across
        && !*LOCAL_ONLY
            .get_or_init(|| std::env::var_os("ZYNTAX_DISABLE_CROSS_BLOCK_LOADS").is_some());
    let (children, roots) = load_regions(func, across);
    let mut visited = vec![false; func.blocks.len()];
    let mut pending: Vec<(usize, Available<'_>)> = Vec::new();
    // The final sweep also covers unreachable cycles without reprocessing roots.
    for root in roots.into_iter().chain(0..func.blocks.len()) {
        if visited[root] {
            continue;
        }
        pending.push((root, Available::new()));
        while let Some((index, mut available)) = pending.pop() {
            if std::mem::replace(&mut visited[index], true) {
                continue;
            }
            let block = func.blocks.get_index(index).unwrap().1;
            for inst in &block.instructions {
                match inst {
                    HirInstruction::Load { volatile: true, .. }
                    | HirInstruction::Store { volatile: true, .. } => available.clear(),
                    HirInstruction::Load {
                        result, ptr, ty, ..
                    } => {
                        let canonical_ptr = chase(*ptr, &substitutions);
                        match available.get(&canonical_ptr) {
                            Some(&(prev_result, previous_ty, _, _)) if previous_ty == ty => {
                                substitutions.insert(*result, prev_result);
                            }
                            _ => {
                                let size = hir_ty_byte_size(ty);
                                let at = loc(*ptr, size);
                                available.insert(canonical_ptr, (*result, ty, at, size));
                            }
                        }
                    }
                    HirInstruction::Store { ptr, value, .. } => {
                        let size = value_byte_size(func, *value);
                        let at = loc(*ptr, size);
                        let canonical_ptr = chase(*ptr, &substitutions);
                        available.retain(|loaded_ptr, (_, _, loaded, loaded_size)| {
                            if at.provably_disjoint(loaded) {
                                return true;
                            }
                            if !ranges || size == 0 || *loaded_size == 0 {
                                return false;
                            }
                            // Build the address index only when a store could kill a load.
                            let addresses = addresses.get_or_insert_with(|| Addresses {
                                function: func,
                                defs: func
                                    .blocks
                                    .values()
                                    .flat_map(|b| &b.instructions)
                                    .filter_map(|i| match i {
                                        HirInstruction::Cast { result, .. }
                                        | HirInstruction::GetElementPtr { result, .. }
                                        | HirInstruction::Load { result, .. }
                                        | HirInstruction::ExtractValue { result, .. }
                                        | HirInstruction::InsertValue { result, .. } => {
                                            Some((*result, i))
                                        }
                                        _ => None,
                                    })
                                    .collect(),
                                cache: HashMap::new(),
                            });
                            let size = addresses.stored_size(*value);
                            let address = addresses.get(canonical_ptr, 32);
                            separate_bytes(
                                address,
                                size,
                                addresses.get(*loaded_ptr, 32),
                                *loaded_size,
                            )
                        });
                    }
                    // Memory-killing ops invalidate everything we
                    // believe about memory contents. Conservative
                    // clearing matches what most early alias-aware CSE
                    // implementations do before they grow up.
                    //
                    // `AsyncSaveSlot` writes to the SM frame so a Load
                    // before it and a Load after it of an
                    // overlapping-frame pointer can legitimately read
                    // different values. Treat it as a memory barrier
                    // (otherwise the krio-emitted state-machine poll-fn
                    // re-uses a stale state Load between yields and
                    // the SM keeps re-parking instead of advancing).
                    //
                    // `CreateClosure` may capture by reference to a
                    // mutable environment; conservatively treat as a
                    // barrier.
                    HirInstruction::VectorStore { .. }
                    | HirInstruction::Call { .. }
                    | HirInstruction::IndirectCall { .. }
                    | HirInstruction::Atomic { .. }
                    | HirInstruction::Fence { .. }
                    | HirInstruction::AsyncSaveSlot { .. }
                    | HirInstruction::CreateClosure { .. }
                    | HirInstruction::CallClosure { .. }
                    | HirInstruction::TraitMethodCall { .. }
                    | HirInstruction::PerformEffect { .. }
                    | HirInstruction::HandleEffect { .. }
                    | HirInstruction::Resume { .. }
                    | HirInstruction::AbortEffect { .. }
                    | HirInstruction::CaptureContinuation { .. }
                    | HirInstruction::FiberNew { .. }
                    | HirInstruction::FiberResume { .. }
                    | HirInstruction::FiberResumeWith { .. }
                    | HirInstruction::FiberYield { .. }
                    | HirInstruction::FiberTransfer { .. }
                    | HirInstruction::FiberCancel { .. }
                    | HirInstruction::FiberDrop { .. } => {
                        available.clear();
                    }
                    _ => {}
                }
            }
            // Bound state copying at forks; straight edges can move the map.
            if available.len() > 64 {
                available.clear();
            }
            if let Some((last, rest)) = children.get(index).and_then(|c| c.split_last()) {
                pending.extend(rest.iter().map(|child| (*child, available.clone())));
                pending.push((*last, available));
            }
        }
    }

    if substitutions.is_empty() {
        return LoadCseStats::default();
    }

    let _rewrites = cse::apply_substitutions_public(func, &substitutions);
    let eliminated = cse::remove_redundant_instructions_public(func, &substitutions);
    LoadCseStats { eliminated }
}

/// Module-level entry.
pub fn run_module(module: &mut HirModule) -> LoadCseStats {
    run_module_with_regions(module, false)
}

pub fn run_module_regions(module: &mut HirModule) -> LoadCseStats {
    run_module_with_regions(module, true)
}

fn run_module_with_regions(module: &mut HirModule, across: bool) -> LoadCseStats {
    let mut total = LoadCseStats::default();
    let exact = crate::licm::exact_struct_names(module);
    for func in module.functions_to_optimize() {
        let s = run_with(func, &exact, across);
        total.eliminated += s.eliminated;
    }
    total
}

fn chase(mut id: HirId, subs: &HashMap<HirId, HirId>) -> HirId {
    let mut seen = 0;
    while let Some(&next) = subs.get(&id) {
        if next == id || seen > 64 {
            break;
        }
        id = next;
        seen += 1;
    }
    id
}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        BinaryOp, HirBlock, HirFunctionSignature, HirTerminator, HirType, HirValue, HirValueKind,
    };
    use std::collections::HashSet;
    use zyntax_typed_ast::InternedString;

    #[test]
    fn byte_ranges_never_discard_an_overlapping_write() {
        let root = HirId::new();
        let offsets = [0, 1, 4, 8, 16, u64::MAX - 16, u64::MAX - 7, u64::MAX];
        for a in offsets {
            for b in offsets {
                for a_size in [0, 1, 2, 4, 8, 16] {
                    for b_size in [0, 1, 2, 4, 8, 16] {
                        if separate_bytes(
                            Some(Address { root, offset: a }),
                            a_size,
                            Some(Address { root, offset: b }),
                            b_size,
                        ) {
                            assert!(a_size > 0 && b_size > 0);
                            for i in 0..a_size {
                                for j in 0..b_size {
                                    assert_ne!(
                                        a.wrapping_add(u64::from(i)),
                                        b.wrapping_add(u64::from(j))
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn aggregate_loads_compare_their_declared_types_not_their_carriers() {
        let (mut f, entry) = mk_func();
        let ty = HirType::Struct(crate::hir::HirStructType {
            name: None,
            fields: vec![HirType::I64; 3],
            packed: false,
        });
        let carrier = HirType::Ptr(Box::new(ty.clone()));
        let ptr = add_param(&mut f, carrier.clone(), 0);
        for _ in 0..2 {
            // An aggregate value may be carried as the address of its bytes.
            let result = add_inst(&mut f, carrier.clone());
            push(
                &mut f,
                entry,
                HirInstruction::Load {
                    result,
                    ty: ty.clone(),
                    ptr,
                    align: 8,
                    volatile: false,
                },
            );
        }
        assert_eq!(run(&mut f).eliminated, 1);
    }

    fn sig() -> HirFunctionSignature {
        HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I64],
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

    fn mk_func() -> (HirFunction, HirId) {
        let mut f = HirFunction::new(InternedString::new_global("t"), sig());
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        (f, entry)
    }

    fn add_param(f: &mut HirFunction, ty: HirType, idx: u32) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Parameter(idx),
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }

    fn add_inst(f: &mut HirFunction, ty: HirType) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }

    fn push(f: &mut HirFunction, entry: HirId, inst: HirInstruction) {
        f.blocks.get_mut(&entry).unwrap().instructions.push(inst);
    }

    #[test]
    fn two_loads_same_ptr_collapse_to_one() {
        let (mut f, entry) = mk_func();
        let p = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 0);
        let l1 = add_inst(&mut f, HirType::I64);
        let l2 = add_inst(&mut f, HirType::I64);
        let sum = add_inst(&mut f, HirType::I64);
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l1,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l2,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: sum,
                ty: HirType::I64,
                left: l1,
                right: l2,
            },
        );
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![sum] };

        let stats = run(&mut f);
        assert_eq!(stats.eliminated, 1);
        // Both operands of the Add should now point at l1.
        let blk = &f.blocks[&entry];
        let add = blk
            .instructions
            .iter()
            .find_map(|i| {
                if let HirInstruction::Binary {
                    op: BinaryOp::Add,
                    left,
                    right,
                    ..
                } = i
                {
                    Some((*left, *right))
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(add, (l1, l1));
        // Only one Load left.
        assert_eq!(
            blk.instructions
                .iter()
                .filter(|i| matches!(i, HirInstruction::Load { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn intervening_store_blocks_collapse() {
        let (mut f, entry) = mk_func();
        let p = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 0);
        let v = add_param(&mut f, HirType::I64, 1);
        let l1 = add_inst(&mut f, HirType::I64);
        let l2 = add_inst(&mut f, HirType::I64);
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l1,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Store {
                value: v,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l2,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return {
            values: vec![l1, l2],
        };

        let stats = run(&mut f);
        assert_eq!(stats.eliminated, 0, "Store between Loads must block CSE");
        assert_eq!(f.blocks[&entry].instructions.len(), 3);
    }

    #[test]
    fn intervening_call_blocks_collapse() {
        let (mut f, entry) = mk_func();
        let p = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 0);
        let l1 = add_inst(&mut f, HirType::I64);
        let l2 = add_inst(&mut f, HirType::I64);
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l1,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Call {
                result: None,
                callee: crate::hir::HirCallable::Symbol("opaque".to_string()),
                args: vec![],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l2,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        let stats = run(&mut f);
        assert_eq!(stats.eliminated, 0, "Call between Loads must block CSE");
    }

    #[test]
    fn different_pointers_do_not_alias_each_other() {
        let (mut f, entry) = mk_func();
        let p1 = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 0);
        let p2 = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 1);
        let l1 = add_inst(&mut f, HirType::I64);
        let l2 = add_inst(&mut f, HirType::I64);
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l1,
                ty: HirType::I64,
                ptr: p1,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: l2,
                ty: HirType::I64,
                ptr: p2,
                align: 8,
                volatile: false,
            },
        );
        let stats = run(&mut f);
        assert_eq!(stats.eliminated, 0, "different ptrs are different loads");
    }

    /// Two loads through `*load_ty` around a store through `*store_ty`,
    /// in a module where `exact` names the exact struct types. How many
    /// loads the pass removed.
    fn typed_store_between_loads(store_ty: &str, load_ty: &str, exact: &[&str]) -> usize {
        let named = |name: &str| {
            HirType::Ptr(Box::new(HirType::Struct(crate::hir::HirStructType {
                name: Some(InternedString::new_global(name)),
                fields: vec![HirType::I64],
                packed: false,
            })))
        };
        let (mut f, entry) = mk_func();
        let stored = add_param(&mut f, named(store_ty), 0);
        let loaded = add_param(&mut f, named(load_ty), 1);
        let v = add_param(&mut f, HirType::I64, 2);
        let l1 = add_inst(&mut f, HirType::I64);
        let l2 = add_inst(&mut f, HirType::I64);
        let sum = add_inst(&mut f, HirType::I64);
        let load = |result| HirInstruction::Load {
            result,
            ty: HirType::I64,
            ptr: loaded,
            align: 8,
            volatile: false,
        };
        push(&mut f, entry, load(l1));
        push(
            &mut f,
            entry,
            HirInstruction::Store {
                value: v,
                ptr: stored,
                align: 8,
                volatile: false,
            },
        );
        push(&mut f, entry, load(l2));
        push(
            &mut f,
            entry,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: sum,
                ty: HirType::I64,
                left: l1,
                right: l2,
            },
        );
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![sum] };
        let mut module = HirModule::new(InternedString::new_global("m"));
        for name in [store_ty, load_ty] {
            let id = zyntax_typed_ast::TypeId::next();
            module.types.insert(id, named(name));
            if exact.contains(&name) {
                module.exact_struct_types.insert(id);
            }
        }
        module.functions.insert(f.id, f);
        run_module(&mut module).eliminated
    }

    /// A store to an object of one exact struct type leaves a load of
    /// another exact type standing; a store that may reach the loaded
    /// object does not.
    #[test]
    fn a_store_to_another_exact_type_keeps_the_load() {
        assert_eq!(typed_store_between_loads("B", "A", &["A", "B"]), 1);
        assert_eq!(typed_store_between_loads("Base", "Sub", &["Sub"]), 0);
        assert_eq!(typed_store_between_loads("A", "A", &["A"]), 0);
    }
}
