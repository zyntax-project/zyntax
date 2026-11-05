use super::*;
use crate::hir::{HirFunctionSignature, HirValueKind};
use zyntax_typed_ast::InternedString;

fn function() -> HirFunction {
    HirFunction::new(
        InternedString::new_global("release_flow"),
        HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::Void],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        },
    )
}

// Independent reference: solve the liveness equations over raw terminators.
fn fixed_point(
    f: &HirFunction,
    allocation: HirId,
    uses: &IdSet,
    transfers: &IdSet,
) -> (IdSet, IdSet) {
    let mut reachable = IdSet::default();
    let mut work = vec![f.entry_block];
    while let Some(id) = work.pop() {
        if reachable.insert(id) {
            if let Some(b) = f.blocks.get(&id) {
                work.extend(b.terminator.targets());
            }
        }
    }
    let mut live_in = IdSet::default();
    let mut live_out = IdSet::default();
    loop {
        let mut changed = false;
        for (&id, block) in &f.blocks {
            if !reachable.contains(&id) {
                continue;
            }
            let out = transfers.contains(&id)
                || block
                    .terminator
                    .targets()
                    .iter()
                    .any(|s| live_in.contains(s));
            if out {
                changed |= live_out.insert(id);
            }
            if id != allocation && (uses.contains(&id) || out) {
                changed |= live_in.insert(id);
            }
        }
        if !changed {
            return (live_in, live_out);
        }
    }
}

#[test]
fn worklist_matches_equations_on_cyclic_and_unreachable_graphs() {
    let mut seed = 71u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 32) as usize
    };
    for _ in 0..512 {
        let mut f = function();
        let count = 2 + next() % 7;
        let mut ids = vec![f.entry_block];
        for _ in 1..count {
            ids.push(f.create_block());
        }
        let condition = f.create_value(
            HirType::Bool,
            HirValueKind::Constant(HirConstant::Bool(true)),
        );
        for id in &ids {
            f.blocks[id].terminator = match next() % 3 {
                0 => HirTerminator::Return { values: vec![] },
                1 => HirTerminator::Branch {
                    target: ids[next() % count],
                },
                _ => HirTerminator::CondBranch {
                    condition,
                    true_target: ids[next() % count],
                    false_target: ids[next() % count],
                },
            };
        }
        f.blocks.swap_indices(0, count - 1);
        let graph = ReleaseGraph::of(&f);
        for _ in 0..16 {
            let allocation = ids[next() % count];
            let uses = ids.iter().copied().filter(|_| next() % 3 == 0).collect();
            let transfers = ids.iter().copied().filter(|_| next() % 4 == 0).collect();
            assert_eq!(
                graph.liveness(allocation, &uses, &transfers),
                fixed_point(&f, allocation, &uses, &transfers)
            );
        }
    }
}

#[test]
fn a_long_forward_chain_propagates_back_to_the_allocation() {
    let mut f = function();
    let mut last = f.entry_block;
    for _ in 0..1024 {
        let next = f.create_block();
        f.blocks[&last].terminator = HirTerminator::Branch { target: next };
        last = next;
    }
    f.blocks[&last].terminator = HirTerminator::Return { values: vec![] };
    let uses = IdSet::from_iter([last]);
    let transfers = IdSet::default();
    let at = std::time::Instant::now();
    let expected = fixed_point(&f, f.entry_block, &uses, &transfers);
    let reference = at.elapsed();
    let at = std::time::Instant::now();
    let graph = ReleaseGraph::of(&f);
    let actual = graph.liveness(f.entry_block, &uses, &transfers);
    let worklist = at.elapsed();
    assert_eq!(actual, expected);
    assert_eq!(actual.0.len(), 1024);
    assert!(!actual.0.contains(&f.entry_block));
    assert!(actual.1.contains(&f.entry_block));
    eprintln!("release liveness chain: fixed_point={reference:?}, graph_and_worklist={worklist:?}");
}

fn allocation(f: &mut HirFunction) -> (HirId, MallocSite) {
    let ptr = f.create_value(
        HirType::Ptr(std::sync::Arc::new(HirType::I64)),
        HirValueKind::Instruction,
    );
    let size = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(8)));
    f.blocks[&f.entry_block]
        .instructions
        .push(HirInstruction::Call {
            result: Some(ptr),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
    (
        ptr,
        MallocSite {
            result: ptr,
            block: f.entry_block,
            inst_idx: 0,
            release: Release::Intrinsic,
        },
    )
}

fn load(f: &mut HirFunction, block: HirId, ptr: HirId) {
    let result = f.create_value(HirType::I64, HirValueKind::Instruction);
    f.blocks[&block].instructions.push(HirInstruction::Load {
        result,
        ptr,
        ty: HirType::I64,
        align: 8,
        volatile: false,
    });
}

#[test]
fn splitting_a_release_edge_invalidates_the_cached_graph() {
    let mut f = function();
    let (ptr, site) = allocation(&mut f);
    let entry = f.entry_block;
    let read = f.create_block();
    let exit = f.create_block();
    let condition = f.create_value(
        HirType::Bool,
        HirValueKind::Constant(HirConstant::Bool(true)),
    );
    f.blocks[&entry].terminator = HirTerminator::CondBranch {
        condition,
        true_target: read,
        false_target: exit,
    };
    f.blocks[&read].terminator = HirTerminator::Branch { target: exit };
    f.blocks[&exit].terminator = HirTerminator::Return { values: vec![] };
    load(&mut f, read, ptr);
    let derived = IdSet::from_iter([ptr]);
    let facts = ModuleFacts::default();
    let users = Users::of(&f);
    assert!(users.flow.get().is_none());
    let points = drop_points(&f, &site, &derived, &facts, &users).unwrap();
    assert_eq!(
        points,
        vec![Point::Edge(entry, exit), Point::After(read, 0)]
    );
    let cached = users.flow.get().unwrap() as *const ReleaseGraph;
    assert_eq!(
        drop_points(&f, &site, &derived, &facts, &users),
        Some(points.clone())
    );
    assert_eq!(cached, users.flow.get().unwrap() as *const ReleaseGraph);
    assert_eq!(apply_points(&mut f, points, ptr, Release::Intrinsic), 2);
    assert!(users.stale(&f));
    assert_eq!(f.blocks.len(), 4);
    let fresh = Users::of(&f);
    assert!(fresh.flow.get().is_none());
    assert!(drop_points(&f, &site, &derived, &facts, &fresh).is_none());
    let graph = fresh.flow.get().unwrap();
    let middle = *f.blocks.last().unwrap().0;
    assert!(graph.reachable.contains(&middle));
    assert!(graph.successors[&entry].contains(&middle));
    assert_eq!(graph.successors[&middle].as_slice(), &[exit]);
}

#[test]
fn unreachable_indexed_uses_do_not_hide_later_reachable_uses() {
    let mut f = function();
    let (ptr, site) = allocation(&mut f);
    let dead = f.create_block();
    let read = f.create_block();
    f.blocks[&f.entry_block].terminator = HirTerminator::Branch { target: read };
    f.blocks[&dead].terminator = HirTerminator::Return { values: vec![ptr] };
    f.blocks[&read].terminator = HirTerminator::Return { values: vec![] };
    load(&mut f, dead, ptr);
    load(&mut f, read, ptr);
    assert_eq!(
        drop_points(
            &f,
            &site,
            &IdSet::from_iter([ptr]),
            &ModuleFacts::default(),
            &Users::of(&f)
        ),
        Some(vec![Point::After(read, 0)])
    );
}
