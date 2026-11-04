//! Rollback state for edge rewrites that leave instructions and block order intact.

use crate::hir::{HirFunction, HirId, HirPhi, HirTerminator, IdHashSet};

pub(super) struct EdgeBackup(Vec<(HirId, Vec<HirPhi>, HirTerminator)>);

impl EdgeBackup {
    pub(super) fn new(func: &HirFunction, blocks: impl IntoIterator<Item = HirId>) -> Self {
        let mut seen = IdHashSet::default();
        Self(
            blocks
                .into_iter()
                .filter(|id| seen.insert(*id))
                .map(|id| {
                    let block = &func.blocks[&id];
                    (id, block.phis.clone(), block.terminator.clone())
                })
                .collect(),
        )
    }

    pub(super) fn restore(self, func: &mut HirFunction) {
        for (id, phis, terminator) in self.0 {
            let block = &mut func.blocks[&id];
            block.phis = phis;
            block.terminator = terminator;
        }
        func.rebuild_cfg_edges();
    }
}
