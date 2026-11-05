use super::*;

#[cfg(test)]
mod tests;

/// Terminator-derived edges and reachability for one use-index lifetime.
/// Release insertion invalidates the index before another site is analysed.
pub(super) struct ReleaseGraph {
    successors: HashMap<HirId, smallvec::SmallVec<[HirId; 2]>>,
    predecessors: HashMap<HirId, smallvec::SmallVec<[HirId; 2]>>,
    reachable: IdSet,
}

impl ReleaseGraph {
    pub(super) fn of(func: &HirFunction) -> Self {
        let mut successors = HashMap::default();
        let mut predecessors: HashMap<_, smallvec::SmallVec<[HirId; 2]>> = HashMap::default();
        for (&id, block) in &func.blocks {
            let mut targets = smallvec::SmallVec::<[HirId; 2]>::new();
            block.terminator.for_each_target(|target| {
                targets.push(target);
                predecessors.entry(target).or_default().push(id);
            });
            successors.insert(id, targets);
        }
        let mut reachable = IdSet::default();
        let mut work = vec![func.entry_block];
        while let Some(id) = work.pop() {
            if reachable.insert(id) {
                if let Some(targets) = successors.get(&id) {
                    work.extend(targets.iter().copied());
                }
            }
        }
        Self {
            successors,
            predecessors,
            reachable,
        }
    }

    /// A newly live entry makes its predecessors live out. The allocation
    /// block kills entry liveness, even when a back edge reaches it.
    fn liveness(&self, allocation: HirId, uses: &IdSet, transfers: &IdSet) -> (IdSet, IdSet) {
        let mut live_in = IdSet::default();
        let mut live_out = IdSet::default();
        let mut work = Vec::new();
        for &block in self.successors.keys() {
            if !self.reachable.contains(&block) {
                continue;
            }
            let transfer = transfers.contains(&block);
            if transfer {
                live_out.insert(block);
            }
            if block != allocation && (transfer || uses.contains(&block)) {
                live_in.insert(block);
                work.push(block);
            }
        }
        while let Some(block) = work.pop() {
            if let Some(preds) = self.predecessors.get(&block) {
                for &pred in preds {
                    if self.reachable.contains(&pred) {
                        live_out.insert(pred);
                        if pred != allocation && live_in.insert(pred) {
                            work.push(pred);
                        }
                    }
                }
            }
        }
        (live_in, live_out)
    }
}

/// Where an allocation dies, when it outlives the block it was made in.
///
/// A value is live at a point when some use is still ahead of it. Read
/// backwards: a block's exit is live if any successor's entry is, and a
/// block's entry is live if it uses the value or its own exit is. The
/// definition kills it, so nothing above the allocation is ever called
/// live and a loop's back edge cannot carry a claim into the iteration
/// that made it.
///
/// The release goes wherever the value is live and no successor keeps
/// it so. That is one point for a value dying at the end of a loop
/// body, and one per arm for a value dying inside a branch.
///
/// Returns `None` where no such point exists, which is a value that is
/// live on every exit and so is not this function's to release.
pub(super) fn drop_points(
    func: &HirFunction,
    site: &MallocSite,
    derived: &IdSet,
    facts: &ModuleFacts,
    users: &Users,
) -> Option<Vec<Point>> {
    drop_points_transferring_with_uses(
        func,
        site,
        derived,
        facts,
        &IdSet::default(),
        &IdSet::default(),
        users,
    )
}

/// [`drop_points`] where leaving `transfer_out` blocks hands the storage
/// to whoever owns it past the edge, a phi that releases it later: the
/// value is live out of such a block whatever its successors do, so no
/// release is placed in it.
pub(super) fn drop_points_transferring_with_uses(
    func: &HirFunction,
    site: &MallocSite,
    derived: &IdSet,
    facts: &ModuleFacts,
    transfer_out: &IdSet,
    implicit_uses: &IdSet,
    users: &Users,
) -> Option<Vec<Point>> {
    let graph = users.flow.get_or_init(|| ReleaseGraph::of(func));
    let reachable = &graph.reachable;

    // What each block does with the value: the last instruction that
    // uses it, and whether the terminator does.
    let mut last_use: HashMap<HirId, usize> = HashMap::default();
    let mut uses_block: IdSet = implicit_uses.clone();
    let mut uses = users.instruction_uses(func, derived).into_iter().peekable();
    for (pos, (block_id, block)) in func.blocks.iter().enumerate() {
        while let Some((_, idx)) = uses.next_if(|(b, _)| *b == pos) {
            if !reachable.contains(block_id) {
                continue;
            }
            let inst = &block.instructions[idx];
            if *block_id == site.block && idx == site.inst_idx {
                continue;
            }
            match classify_derived_use(inst, derived, facts) {
                UseKind::Use => {
                    uses_block.insert(*block_id);
                    let e = last_use.entry(*block_id).or_insert(idx);
                    *e = (*e).max(idx);
                }
                UseKind::Escape => {
                    if trace_enabled() {
                        eprintln!("[drop]   escapes through {inst:?}");
                    }
                    return None;
                }
                UseKind::None => {}
            }
        }
        if !reachable.contains(block_id) {
            continue;
        }
        match derived
            .iter()
            .map(|d| classify_terminator_use(&block.terminator, *d))
            .fold(UseKind::None, strongest)
        {
            UseKind::Use => {
                uses_block.insert(*block_id);
            }
            UseKind::Escape => {
                if trace_enabled() {
                    eprintln!("[drop]   escapes through terminator {:?}", block.terminator);
                }
                return None;
            }
            UseKind::None => {}
        }
    }
    if uses_block.is_empty() {
        if trace_enabled() {
            eprintln!("[drop]   no block reads it");
        }
        return None;
    }

    let (live_in, live_out) = graph.liveness(site.block, &uses_block, transfer_out);

    // Where it is live and nothing after it is: after the last use in a
    // block no successor of which keeps it, and on each edge out of a
    // block that keeps it into one that does not.
    let mut points = Vec::new();
    for (block_id, block) in &func.blocks {
        if !reachable.contains(block_id) {
            continue;
        }
        let holds = *block_id == site.block || live_in.contains(block_id);
        if !holds {
            continue;
        }
        if live_out.contains(block_id) {
            // Handed on along a transfer edge, or read further on along
            // some successor; the others end its life as they are taken.
            if transfer_out.contains(block_id) {
                continue;
            }
            for &succ in &graph.successors[block_id] {
                if !live_in.contains(&succ) && succ != site.block && reachable.contains(&succ) {
                    points.push(Point::Edge(*block_id, succ));
                }
            }
            continue;
        }
        if !uses_block.contains(block_id) && *block_id != site.block {
            continue;
        }
        // After the last use, or ahead of a terminator that is the only
        // thing using it. A terminator reads the address rather than
        // what is at it, so releasing first is still the right order.
        let at = match last_use.get(block_id) {
            Some(idx) => idx + 1,
            None if *block_id == site.block && !uses_block.contains(block_id) => {
                if trace_enabled() {
                    eprintln!("[drop]   its own block neither reads it nor passes it on");
                }
                return None;
            }
            None => block.instructions.len(),
        };
        points.push(Point::After(*block_id, at.saturating_sub(1)));
    }
    if points.is_empty() && trace_enabled() {
        eprintln!("[drop]   live on every path out");
    }
    (!points.is_empty()).then_some(points)
}
