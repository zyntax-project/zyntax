//! Natural-loop detection on the HIR dominator tree.
//!
//! A *natural loop* with header `h` and back-edge `t → h` is the set
//! of blocks that can reach `t` without going through any block other
//! than `h` (i.e. the maximal set of blocks dominated by `h` that
//! reach `t` via predecessor walks). This is the standard textbook
//! definition (e.g. Muchnick §7.4) and is what LICM / loop
//! vectorization / induction-variable analysis all need.
//!
//! We surface:
//!
//!   * `loops()`           — every natural loop, deduplicated by header
//!                           (when multiple back-edges share a header
//!                           we union the bodies)
//!   * `containing_loop(b)` — innermost loop containing `b`, if any
//!   * `loop_depth(b)`      — number of loops nesting `b` (0 outside
//!                            any loop)
//!   * `is_header(b)`       — convenience for "does any back-edge
//!                            target `b`?"
//!
//! ## Definition recap
//!
//! Given a dominator tree, an edge `t → h` is a *back-edge* iff `h`
//! dominates `t`. The natural loop of that back-edge is
//!
//! ```text
//! L(h, t) = {h} ∪ { x : ∃ path t ← … ← x avoiding h }
//! ```
//!
//! which equals the set of predecessor-reachable nodes from `t`
//! stopping at `h`. The header `h` is by definition the only entry
//! to the loop body — that's the property LICM uses to hoist
//! invariants into the preheader.
//!
//! ## What this module does NOT do
//!
//! Irreducible CFGs (multiple back-edges to different headers in the
//! same loop) are not supported. ZynML's source-level loops always
//! lower to reducible CFGs in the SSA builder, so this matches what
//! we actually see. If an irreducible shape ever appears (e.g. from
//! goto-style rewrites) the detector will see it as several
//! independent natural loops, which is suboptimal but not incorrect
//! for the conservative-hoisting / vectorize-when-safe passes we
//! plan to build on top.

use crate::analysis::dominators::DominatorTree;
use crate::hir::{HirFunction, HirId};
use std::collections::{HashMap, HashSet};

/// A single natural loop.
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    /// Loop header — the single entry to the body. Always present
    /// in `body`. The header is also the back-edge target.
    pub header: HirId,
    /// All blocks in the loop, including the header. A block can
    /// belong to several nested loops; the outer-loop body is a
    /// superset of every inner-loop body. (See `LoopForest::nesting`
    /// for the parent relation.)
    pub body: HashSet<HirId>,
    /// Source blocks of every back-edge that targets `header`. A
    /// single header can have multiple back-edges (e.g. `continue`
    /// branches merged at the header); we union their bodies into
    /// the same `NaturalLoop` to keep the analysis interface "one
    /// loop per header" rather than "one loop per back-edge".
    pub latches: Vec<HirId>,
    /// Blocks outside the loop that have at least one predecessor
    /// inside the loop. Anything an iteration can branch to that
    /// isn't the header.
    pub exits: HashSet<HirId>,
}

/// All natural loops in a function plus the nesting structure.
#[derive(Debug, Default, Clone)]
pub struct LoopForest {
    loops: Vec<NaturalLoop>,
    /// Index in `loops` of the innermost loop containing each block,
    /// or absent if the block isn't in any loop.
    innermost: HashMap<HirId, usize>,
    /// Parent (more outer) loop index per loop index. Outer loops
    /// strictly contain inner loops (their body is a strict superset).
    parent: HashMap<usize, usize>,
}

impl LoopForest {
    /// Detect every natural loop in `func` using `dt`. O(N·E) in the
    /// worst case (N blocks × E back-edges × predecessor flood);
    /// O(N+E) for any function without unusually nested loops.
    pub fn detect(func: &HirFunction, dt: &DominatorTree) -> Self {
        // 1. Enumerate back-edges: every edge `t → h` where `h`
        //    dominates `t`.
        let mut back_edges: Vec<(HirId, HirId)> = Vec::new();
        for &t in dt.rpo() {
            if let Some(block) = func.blocks.get(&t) {
                for &h in &block.successors {
                    if dt.dominates(h, t) {
                        back_edges.push((t, h));
                    }
                }
            }
        }

        // 2. Group by header — multiple back-edges to the same header
        //    form one loop with a union of bodies.
        let mut by_header: HashMap<HirId, Vec<HirId>> = HashMap::new();
        for (t, h) in &back_edges {
            by_header.entry(*h).or_default().push(*t);
        }

        // 3. For each header, flood predecessors from each latch
        //    backwards, stopping at the header. That gives the body.
        let mut loops: Vec<NaturalLoop> = Vec::new();
        for (header, latches) in by_header {
            let body = collect_body(func, header, &latches);
            let exits = collect_exits(func, &body);
            loops.push(NaturalLoop {
                header,
                body,
                latches,
                exits,
            });
        }

        // 4. Derive nesting: an outer loop's body strictly contains
        //    an inner loop's body. Sort by body size ascending so
        //    when we walk we always meet the innermost loops first.
        loops.sort_by_key(|l| l.body.len());

        let mut innermost: HashMap<HirId, usize> = HashMap::new();
        for (i, l) in loops.iter().enumerate() {
            for &b in &l.body {
                // Smaller-body loops are visited first so the FIRST
                // recorded loop for a block is the innermost.
                innermost.entry(b).or_insert(i);
            }
        }

        // Parent = smallest loop whose body strictly contains ours.
        let mut parent: HashMap<usize, usize> = HashMap::new();
        for (i, inner) in loops.iter().enumerate() {
            // Iterate outer candidates (anything larger). Take the
            // smallest container.
            let mut best: Option<usize> = None;
            for (j, outer) in loops.iter().enumerate().skip(i + 1) {
                if outer.body.len() <= inner.body.len() {
                    continue;
                }
                if inner.body.is_subset(&outer.body) {
                    best = Some(match best {
                        Some(prev) if loops[prev].body.len() <= outer.body.len() => prev,
                        _ => j,
                    });
                }
            }
            if let Some(p) = best {
                parent.insert(i, p);
            }
        }

        Self {
            loops,
            innermost,
            parent,
        }
    }

    /// Every detected loop, sorted by body size ascending. The
    /// innermost loops come first — convenient for passes that want
    /// to process them inside-out (e.g. vectorize the inner before
    /// the outer's LICM).
    pub fn loops(&self) -> &[NaturalLoop] {
        &self.loops
    }

    /// Innermost loop containing `block`, or `None` if `block` is
    /// outside every loop.
    pub fn containing_loop(&self, block: HirId) -> Option<&NaturalLoop> {
        self.innermost.get(&block).map(|&i| &self.loops[i])
    }

    /// Number of loops `block` is in. 0 outside any loop.
    pub fn loop_depth(&self, block: HirId) -> usize {
        let mut idx = match self.innermost.get(&block) {
            Some(&i) => i,
            None => return 0,
        };
        let mut depth = 1;
        while let Some(&p) = self.parent.get(&idx) {
            depth += 1;
            idx = p;
        }
        depth
    }

    /// Is `block` the header of any natural loop?
    pub fn is_header(&self, block: HirId) -> bool {
        self.loops.iter().any(|l| l.header == block)
    }

    /// Parent (immediately outer) loop index, if any.
    pub fn parent_of(&self, loop_idx: usize) -> Option<usize> {
        self.parent.get(&loop_idx).copied()
    }
}

/// Flood predecessors backwards from every latch, stopping at the
/// header. The result is the loop body — every block on some path
/// from `header` back to a latch.
fn collect_body(func: &HirFunction, header: HirId, latches: &[HirId]) -> HashSet<HirId> {
    let mut body: HashSet<HirId> = HashSet::new();
    body.insert(header);
    let mut stack: Vec<HirId> = latches.to_vec();
    for &l in latches {
        body.insert(l);
    }
    while let Some(b) = stack.pop() {
        if let Some(block) = func.blocks.get(&b) {
            for &p in &block.predecessors {
                if body.insert(p) {
                    // First time we've seen `p`; flood its preds too.
                    // Stop at header — anything past it is outside
                    // the natural loop.
                    if p != header {
                        stack.push(p);
                    }
                }
            }
        }
    }
    body
}

/// Loop exits: any block OUTSIDE `body` that has a predecessor INSIDE
/// `body`. The successor of an exiting branch.
fn collect_exits(func: &HirFunction, body: &HashSet<HirId>) -> HashSet<HirId> {
    let mut exits = HashSet::new();
    for &b in body {
        if let Some(block) = func.blocks.get(&b) {
            for &s in &block.successors {
                if !body.contains(&s) {
                    exits.insert(s);
                }
            }
        }
    }
    exits
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
            effects: vec![],
            is_pure: false,
        }
    }

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
    fn straight_line_has_no_loops() {
        let a = HirId::new();
        let b = HirId::new();
        let func = build_func("chain", a, &[(a, &[b]), (b, &[])]);
        let dt = DominatorTree::new(&func);
        let lf = LoopForest::detect(&func, &dt);
        assert!(lf.loops().is_empty());
        assert_eq!(lf.loop_depth(a), 0);
        assert_eq!(lf.loop_depth(b), 0);
    }

    #[test]
    fn simple_while_loop_is_detected_once() {
        // entry → header ⇄ body
        //           ↓
        //          exit
        let entry = HirId::new();
        let header = HirId::new();
        let body = HirId::new();
        let exit = HirId::new();
        let func = build_func(
            "while",
            entry,
            &[
                (entry, &[header]),
                (header, &[body, exit]),
                (body, &[header]),
                (exit, &[]),
            ],
        );
        let dt = DominatorTree::new(&func);
        let lf = LoopForest::detect(&func, &dt);
        assert_eq!(lf.loops().len(), 1);
        let l = &lf.loops()[0];
        assert_eq!(l.header, header);
        assert!(l.body.contains(&header));
        assert!(l.body.contains(&body));
        assert!(!l.body.contains(&entry));
        assert!(!l.body.contains(&exit));
        assert_eq!(l.latches, vec![body]);
        assert!(l.exits.contains(&exit));
        assert_eq!(lf.loop_depth(body), 1);
        assert_eq!(lf.loop_depth(entry), 0);
        assert!(lf.is_header(header));
        assert!(!lf.is_header(body));
    }

    #[test]
    fn two_back_edges_to_same_header_become_one_loop() {
        // entry → header → body1 → body2 ⇄ header
        //                            ↓
        //                          header (a `continue` from body1)
        let entry = HirId::new();
        let header = HirId::new();
        let body1 = HirId::new();
        let body2 = HirId::new();
        let exit = HirId::new();
        let func = build_func(
            "two-backedges",
            entry,
            &[
                (entry, &[header]),
                (header, &[body1, exit]),
                (body1, &[body2, header]), // `continue`
                (body2, &[header]),
                (exit, &[]),
            ],
        );
        let dt = DominatorTree::new(&func);
        let lf = LoopForest::detect(&func, &dt);
        assert_eq!(
            lf.loops().len(),
            1,
            "merge multiple back-edges sharing a header into one loop"
        );
        let l = &lf.loops()[0];
        assert_eq!(l.header, header);
        assert!(l.body.contains(&body1));
        assert!(l.body.contains(&body2));
        assert_eq!(l.latches.len(), 2);
    }

    #[test]
    fn nested_loops_have_correct_depth() {
        // entry → outer_header → inner_header ⇄ inner_body
        //              ↑               ↓
        //              └──── outer_latch ←──┘ (also inner exits to outer_latch)
        //              ↓
        //             exit
        let entry = HirId::new();
        let oh = HirId::new();
        let ih = HirId::new();
        let ib = HirId::new();
        let ol = HirId::new();
        let exit = HirId::new();
        let func = build_func(
            "nested",
            entry,
            &[
                (entry, &[oh]),
                (oh, &[ih, exit]),
                (ih, &[ib, ol]),
                (ib, &[ih]),
                (ol, &[oh]),
                (exit, &[]),
            ],
        );
        let dt = DominatorTree::new(&func);
        let lf = LoopForest::detect(&func, &dt);
        assert_eq!(lf.loops().len(), 2);
        assert_eq!(lf.loop_depth(ib), 2, "inner body is in 2 loops");
        assert_eq!(lf.loop_depth(ol), 1, "outer latch is in 1 loop");
        assert_eq!(lf.loop_depth(entry), 0);
        assert!(lf.is_header(oh));
        assert!(lf.is_header(ih));

        // Inner-most-first ordering — passes that vectorize the
        // inner before LICM-ing the outer rely on this.
        let inner_idx = lf
            .loops()
            .iter()
            .position(|l| l.header == ih)
            .expect("inner loop present");
        let outer_idx = lf
            .loops()
            .iter()
            .position(|l| l.header == oh)
            .expect("outer loop present");
        assert!(inner_idx < outer_idx, "innermost loops come first");
        assert_eq!(lf.parent_of(inner_idx), Some(outer_idx));
        assert_eq!(lf.parent_of(outer_idx), None);
    }
}
