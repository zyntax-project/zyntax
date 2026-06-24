//! Dominator-tree analysis over `HirFunction.blocks`.
//!
//! Computes immediate dominators using the Cooper-Harvey-Kennedy
//! iterative algorithm ("A Simple, Fast Dominance Algorithm", Cooper,
//! Harvey, Kennedy, 2001) and exposes the standard dominance queries
//! every downstream optimization pass needs (CSE/GVN, LICM, loop
//! vectorization, code motion, etc.):
//!
//!   * `idom(b)`                  → immediate dominator of `b`, or
//!                                  `None` for the entry / unreachable
//!   * `dominates(a, b)`          → does `a` dominate `b`?  (reflexive)
//!   * `strictly_dominates(a, b)` → does `a` dominate `b` and `a ≠ b`?
//!   * `children(b)`              → dom-tree children of `b` (reverse
//!                                  of `idom`)
//!   * `frontier(b)`              → dominance frontier of `b`
//!   * `rpo()`                    → reverse-postorder block list (the
//!                                  natural iteration order for forward
//!                                  data-flow analyses)
//!
//! ## Why a separate module
//!
//! Two dominator implementations already existed in the tree before
//! this module:
//!
//!   * `ssa.rs::DominanceInfo` — used inside SSA construction; works
//!     on `TypedControlFlowGraph` (the pre-HIR CFG) and isn't reachable
//!     from later passes.
//!   * `cfg.rs::ControlFlowGraph::compute_dominance` — works on the
//!     legacy hand-built `ControlFlowGraph` used by the typed-AST
//!     pipeline; not the same node space as HIR.
//!
//! HIR-level optimization passes (constant folding, CSE, LICM, loop
//! vectorization) need a uniform analysis keyed by `HirId` directly
//! against `HirFunction.blocks`. This module fills that gap. It does
//! *not* cache across passes — each pass that needs it constructs a
//! fresh `DominatorTree`; the algorithm is O(N·E·α(N)) and N is small
//! per function, so the cost is in the noise compared to the
//! optimization work itself.

use crate::hir::{HirFunction, HirId};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

/// Immediate dominators + the standard derived data.
#[derive(Debug, Clone)]
pub struct DominatorTree {
    /// Entry block of the function — the root of the dominator tree.
    entry: HirId,
    /// Reverse-postorder of reachable blocks. `rpo[0] == entry`.
    /// Unreachable blocks are absent — every consumer of this struct
    /// should treat unreachable blocks as "dead" anyway, and walking
    /// them would make the iterative idom step incorrect.
    rpo: Vec<HirId>,
    /// Position in `rpo`. Used by `intersect` to walk the dominator
    /// tree upward.
    rpo_pos: HashMap<HirId, usize>,
    /// `idom[b]` = immediate dominator of `b`. The entry block does
    /// NOT appear here (it has no immediate dominator).
    idom: HashMap<HirId, HirId>,
    /// Reverse of `idom` — dom-tree children of each block. Built
    /// once at construction for O(1) tree traversal.
    children: HashMap<HirId, Vec<HirId>>,
    /// Dominance frontier of each block. `frontier[b]` is the set of
    /// blocks `y` such that `b` dominates a predecessor of `y` but
    /// does not strictly dominate `y` itself — the φ-insertion sites
    /// when `b` defines a value.
    frontier: HashMap<HirId, HashSet<HirId>>,
}

impl DominatorTree {
    /// Build dominator information for `func`. O(N·E·α(N)) in
    /// practice — the iterative loop converges in 1–3 passes for the
    /// CFG shapes the compiler emits.
    pub fn new(func: &HirFunction) -> Self {
        let entry = func.entry_block;
        let rpo = compute_rpo(func, entry);
        let rpo_pos: HashMap<HirId, usize> = rpo.iter().enumerate().map(|(i, b)| (*b, i)).collect();

        let idom = compute_idom(func, entry, &rpo, &rpo_pos);
        let children = build_children(&idom);
        let frontier = compute_frontier(func, &idom, &rpo_pos);

        Self {
            entry,
            rpo,
            rpo_pos,
            idom,
            children,
            frontier,
        }
    }

    /// Entry block — root of the dominator tree.
    pub fn entry(&self) -> HirId {
        self.entry
    }

    /// Immediate dominator of `block`, or `None` if `block` is the
    /// entry or is unreachable.
    pub fn idom(&self, block: HirId) -> Option<HirId> {
        self.idom.get(&block).copied()
    }

    /// Does `dominator` dominate `dominated`? Reflexive (every block
    /// dominates itself). Unreachable blocks dominate no one and are
    /// not dominated by anyone other than themselves.
    pub fn dominates(&self, dominator: HirId, dominated: HirId) -> bool {
        if dominator == dominated {
            return true;
        }
        self.strictly_dominates(dominator, dominated)
    }

    /// Does `dominator` strictly dominate `dominated`? `false` when
    /// `dominator == dominated`.
    pub fn strictly_dominates(&self, dominator: HirId, dominated: HirId) -> bool {
        if dominator == dominated {
            return false;
        }
        // Walk up the dominator tree from `dominated`; if we hit
        // `dominator` we're done.
        let mut cur = dominated;
        while let Some(&next) = self.idom.get(&cur) {
            if next == dominator {
                return true;
            }
            if next == cur {
                // Hit the entry's self-loop sentinel — shouldn't
                // happen in our representation (we omit entry from
                // `idom`) but defensive.
                return false;
            }
            cur = next;
        }
        false
    }

    /// Dom-tree children of `block` — every block whose immediate
    /// dominator is `block`. Returned in deterministic insertion
    /// order (dependent on `func.blocks` iteration order).
    pub fn children(&self, block: HirId) -> &[HirId] {
        self.children
            .get(&block)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Dominance frontier of `block`. Empty when `block` is
    /// unreachable or has no frontier entries.
    pub fn frontier(&self, block: HirId) -> &HashSet<HirId> {
        static EMPTY: once_cell::sync::Lazy<HashSet<HirId>> =
            once_cell::sync::Lazy::new(HashSet::new);
        self.frontier.get(&block).unwrap_or(&EMPTY)
    }

    /// Reverse postorder of reachable blocks — the canonical
    /// iteration order for forward data-flow passes. `rpo()[0]` is
    /// always the entry block.
    pub fn rpo(&self) -> &[HirId] {
        &self.rpo
    }

    /// Position of `block` in RPO. `None` for unreachable blocks —
    /// useful when a downstream pass wants to skip them outright.
    pub fn rpo_position(&self, block: HirId) -> Option<usize> {
        self.rpo_pos.get(&block).copied()
    }

    /// Walk the dominator tree in preorder (every block before its
    /// children). Useful for top-down passes like dominator-based
    /// CSE that need to see definitions before uses.
    pub fn preorder(&self) -> Vec<HirId> {
        let mut out = Vec::with_capacity(self.rpo.len());
        self.preorder_visit(self.entry, &mut out);
        out
    }

    fn preorder_visit(&self, block: HirId, out: &mut Vec<HirId>) {
        out.push(block);
        if let Some(kids) = self.children.get(&block) {
            for &c in kids {
                self.preorder_visit(c, out);
            }
        }
    }
}

/// Compute reverse postorder of reachable blocks via iterative DFS
/// from `entry`. The result is an empty Vec only when the entry
/// itself is missing from `func.blocks` — pathological but defended
/// against.
fn compute_rpo(func: &HirFunction, entry: HirId) -> Vec<HirId> {
    if !func.blocks.contains_key(&entry) {
        return Vec::new();
    }

    // Iterative DFS that records postorder. Recursive form would
    // blow the stack on long single-successor chains in big
    // generated bodies.
    enum Action {
        Enter(HirId),
        Exit(HirId),
    }
    let mut stack = vec![Action::Enter(entry)];
    let mut visited: HashSet<HirId> = HashSet::new();
    let mut postorder: Vec<HirId> = Vec::new();

    while let Some(action) = stack.pop() {
        match action {
            Action::Enter(b) => {
                if !visited.insert(b) {
                    continue;
                }
                // Schedule the exit BEFORE recursing into successors —
                // when the stack unwinds back to this frame we push
                // the block onto the postorder list.
                stack.push(Action::Exit(b));
                if let Some(block) = func.blocks.get(&b) {
                    // Reverse so the leftmost successor ends up on top
                    // of the stack; preserves the natural traversal
                    // order and keeps test output stable across runs.
                    for &succ in block.successors.iter().rev() {
                        if !visited.contains(&succ) && func.blocks.contains_key(&succ) {
                            stack.push(Action::Enter(succ));
                        }
                    }
                }
            }
            Action::Exit(b) => postorder.push(b),
        }
    }

    postorder.reverse();
    postorder
}

/// Cooper-Harvey-Kennedy iterative idom computation. Initialises
/// every reachable block's idom to "undefined", then walks RPO
/// repeatedly until the idom map stabilises. Convergence is fast
/// in practice (1–3 iterations for typical CFGs).
fn compute_idom(
    func: &HirFunction,
    entry: HirId,
    rpo: &[HirId],
    rpo_pos: &HashMap<HirId, usize>,
) -> HashMap<HirId, HirId> {
    let mut idom: IndexMap<HirId, HirId> = IndexMap::new();
    // Sentinel: the entry's "immediate dominator" is itself. We
    // remove the entry from the final result so callers see `None`
    // for it, but during the algorithm we need a marker for "has an
    // idom" vs. "doesn't yet" — using the entry's self-pair fits.
    idom.insert(entry, entry);

    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let block = match func.blocks.get(&b) {
                Some(blk) => blk,
                None => continue,
            };
            // Pick any pred with a known idom as the seed, then fold
            // the remaining preds in via `intersect`.
            let mut new_idom: Option<HirId> = None;
            for &p in &block.predecessors {
                if !idom.contains_key(&p) {
                    continue;
                }
                new_idom = Some(match new_idom {
                    None => p,
                    Some(cur) => intersect(&idom, cur, p, rpo_pos),
                });
            }
            if let Some(ni) = new_idom {
                if idom.get(&b) != Some(&ni) {
                    idom.insert(b, ni);
                    changed = true;
                }
            }
        }
    }

    // Strip the entry's self-pair before returning so the public
    // `idom(entry)` is `None`, matching the standard contract.
    let entry_self = idom.remove(&entry);
    debug_assert!(matches!(entry_self, Some(e) if e == entry));
    idom.into_iter().collect()
}

/// Find the nearest common dominator of `b1` and `b2` by walking
/// both up the tentative dominator tree, choosing whichever side is
/// further from the root in RPO terms each step. Classic CHK
/// "finger" intersect.
fn intersect(
    idom: &IndexMap<HirId, HirId>,
    mut b1: HirId,
    mut b2: HirId,
    rpo_pos: &HashMap<HirId, usize>,
) -> HirId {
    while b1 != b2 {
        while rpo_pos.get(&b1).copied().unwrap_or(usize::MAX)
            > rpo_pos.get(&b2).copied().unwrap_or(usize::MAX)
        {
            match idom.get(&b1) {
                Some(&p) if p != b1 => b1 = p,
                _ => return b2,
            }
        }
        while rpo_pos.get(&b2).copied().unwrap_or(usize::MAX)
            > rpo_pos.get(&b1).copied().unwrap_or(usize::MAX)
        {
            match idom.get(&b2) {
                Some(&p) if p != b2 => b2 = p,
                _ => return b1,
            }
        }
    }
    b1
}

/// Invert `idom` to give a children-of map.
fn build_children(idom: &HashMap<HirId, HirId>) -> HashMap<HirId, Vec<HirId>> {
    let mut out: HashMap<HirId, Vec<HirId>> = HashMap::new();
    for (child, parent) in idom {
        out.entry(*parent).or_default().push(*child);
    }
    out
}

/// Standard CHK dominance-frontier computation: for each block `b`
/// with multiple predecessors, walk each predecessor up the
/// dominator tree adding `b` to every block's frontier until we
/// reach `idom[b]`.
fn compute_frontier(
    func: &HirFunction,
    idom: &HashMap<HirId, HirId>,
    _rpo_pos: &HashMap<HirId, usize>,
) -> HashMap<HirId, HashSet<HirId>> {
    let mut df: HashMap<HirId, HashSet<HirId>> = HashMap::new();

    for (&b, block) in &func.blocks {
        if block.predecessors.len() < 2 {
            continue;
        }
        let b_idom = match idom.get(&b) {
            Some(&id) => id,
            None => continue,
        };
        for &p in &block.predecessors {
            // Walk runner from p up to (but not including) idom[b],
            // adding b to each runner's frontier along the way.
            let mut runner = p;
            while runner != b_idom {
                df.entry(runner).or_default().insert(b);
                match idom.get(&runner) {
                    Some(&next) if next != runner => runner = next,
                    _ => break,
                }
            }
        }
    }

    df
}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirFunction, HirFunctionSignature, HirTerminator, HirType};
    use zyntax_typed_ast::InternedString;

    fn empty_sig() -> HirFunctionSignature {
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

    /// Build a CFG by hand: list of (block_id, successors). Wires
    /// up `predecessors` from the successors automatically.
    fn build_func(name: &str, entry: HirId, edges: &[(HirId, &[HirId])]) -> HirFunction {
        let mut func = HirFunction::new(InternedString::new_global(name), empty_sig());
        func.entry_block = entry;
        func.blocks.clear();
        for &(b, succs) in edges {
            let mut block = HirBlock::new(b);
            block.successors = succs.to_vec();
            block.terminator = HirTerminator::Unreachable;
            func.blocks.insert(b, block);
        }
        // Fill predecessors.
        for &(b, succs) in edges {
            for &s in succs {
                if let Some(target) = func.blocks.get_mut(&s) {
                    target.predecessors.push(b);
                }
            }
        }
        func
    }

    #[test]
    fn straight_line_chain_each_block_idom_is_its_predecessor() {
        let a = HirId::new();
        let b = HirId::new();
        let c = HirId::new();
        let func = build_func("chain", a, &[(a, &[b]), (b, &[c]), (c, &[])]);
        let dt = DominatorTree::new(&func);
        assert_eq!(dt.idom(a), None);
        assert_eq!(dt.idom(b), Some(a));
        assert_eq!(dt.idom(c), Some(b));
        assert!(dt.dominates(a, c));
        assert!(dt.dominates(a, b));
        assert!(!dt.strictly_dominates(c, c));
        assert!(dt.dominates(c, c));
    }

    #[test]
    fn diamond_join_idom_is_split_node() {
        // a ─→ b ─→ d
        //  ╲    ╱
        //   ╲─→ c
        let a = HirId::new();
        let b = HirId::new();
        let c = HirId::new();
        let d = HirId::new();
        let func = build_func(
            "diamond",
            a,
            &[(a, &[b, c]), (b, &[d]), (c, &[d]), (d, &[])],
        );
        let dt = DominatorTree::new(&func);
        assert_eq!(dt.idom(b), Some(a));
        assert_eq!(dt.idom(c), Some(a));
        // d's idom is a (the nearest dominator that's on every path
        // from entry to d). b and c each dominate only their own
        // half of the diamond.
        assert_eq!(dt.idom(d), Some(a));
        assert!(!dt.dominates(b, d));
        assert!(!dt.dominates(c, d));
        // Frontier: d is in df of b AND c (both have d as a join
        // they reach but don't strictly dominate).
        assert!(dt.frontier(b).contains(&d));
        assert!(dt.frontier(c).contains(&d));
    }

    #[test]
    fn natural_loop_back_edge_does_not_break_idom() {
        // a ─→ h ─→ body ─→ exit
        //       ↑           │
        //       └──── back ←┘  (body has edge back to h)
        let a = HirId::new();
        let h = HirId::new();
        let body = HirId::new();
        let exit = HirId::new();
        let func = build_func(
            "loop",
            a,
            &[(a, &[h]), (h, &[body, exit]), (body, &[h]), (exit, &[])],
        );
        let dt = DominatorTree::new(&func);
        assert_eq!(dt.idom(h), Some(a));
        // body and exit are both dominated by h (the loop header).
        assert_eq!(dt.idom(body), Some(h));
        assert_eq!(dt.idom(exit), Some(h));
        assert!(dt.dominates(h, body));
        assert!(dt.dominates(h, exit));
        // Back-edge target (h) shows up in body's dominance frontier.
        assert!(dt.frontier(body).contains(&h));
    }

    #[test]
    fn preorder_walks_dom_tree_top_down() {
        // a ─→ b ─→ d
        //  ╲    ╲
        //   ╲    e
        //    ╲─→ c
        // Dominator tree:  a ─→ {b, c}; b ─→ {d, e}
        let a = HirId::new();
        let b = HirId::new();
        let c = HirId::new();
        let d = HirId::new();
        let e = HirId::new();
        let func = build_func(
            "preorder",
            a,
            &[(a, &[b, c]), (b, &[d, e]), (c, &[]), (d, &[]), (e, &[])],
        );
        let dt = DominatorTree::new(&func);
        let pre = dt.preorder();
        // a is first; b's subtree is visited before c (or vice-versa,
        // but a is always first and every parent precedes its kids).
        assert_eq!(pre[0], a);
        let pos = |x: HirId| pre.iter().position(|&y| y == x).unwrap();
        assert!(pos(b) < pos(d));
        assert!(pos(b) < pos(e));
        assert!(pos(a) < pos(b));
        assert!(pos(a) < pos(c));
    }

    #[test]
    fn unreachable_block_is_absent_from_rpo_and_has_no_idom() {
        let a = HirId::new();
        let b = HirId::new();
        let dead = HirId::new();
        let func = build_func("unreachable", a, &[(a, &[b]), (b, &[]), (dead, &[])]);
        let dt = DominatorTree::new(&func);
        assert!(!dt.rpo().contains(&dead));
        assert_eq!(dt.idom(dead), None);
        assert_eq!(dt.rpo_position(dead), None);
        // Reachable blocks unaffected.
        assert_eq!(dt.rpo()[0], a);
        assert_eq!(dt.idom(b), Some(a));
    }
}
