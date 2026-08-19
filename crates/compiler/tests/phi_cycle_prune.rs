//! A loop-carried value nothing reads is dropped, even when two of them
//! name each other.
//!
//! SSA construction gives a loop header a phi for every variable written
//! anywhere inside it. For a nested loop that means the outer header
//! takes a phi for each of the inner loop's variables, the inner header
//! takes that phi as its incoming from outside, and because the inner
//! loop reinitialises before reading, the pair is read by nobody but
//! each other. Each looks used, so a pass that removes only phis it can
//! prove unread removes neither, and every level of nesting keeps a
//! register for a value no instruction ever asks for.

use zyntax_compiler::hir::*;
use zyntax_typed_ast::InternedString;

fn ret_i64_sig() -> HirFunctionSignature {
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

/// Names of the phi results still present in a block, for assertions.
fn phis_in(func: &HirFunction, block: HirId) -> Vec<HirId> {
    func.blocks[&block].phis.iter().map(|p| p.result).collect()
}

/// A two-deep counted loop nest carrying one live counter per level and
/// one dead phi per level, where the two dead ones name each other.
///
/// Returns the function together with the ids that must survive and the
/// ids that must not.
struct Nest {
    func: HirFunction,
    outer_header: HirId,
    inner_header: HirId,
    live: Vec<HirId>,
    dead: Vec<HirId>,
}

fn nested_loop_with_a_dead_cycle() -> Nest {
    let mut f = HirFunction::new(InternedString::new_global("nest"), ret_i64_sig());
    let entry = f.entry_block;
    let outer = f.create_block();
    let inner = f.create_block();
    let inner_body = f.create_block();
    let outer_latch = f.create_block();
    let exit = f.create_block();

    let zero = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
    let one = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(1)));
    let t = f.create_value(
        HirType::Bool,
        HirValueKind::Constant(HirConstant::Bool(true)),
    );

    let i = f.create_value(HirType::I64, HirValueKind::Instruction);
    let i_next = f.create_value(HirType::I64, HirValueKind::Instruction);
    let k = f.create_value(HirType::I64, HirValueKind::Instruction);
    let k_next = f.create_value(HirType::I64, HirValueKind::Instruction);
    let dead_outer = f.create_value(HirType::I64, HirValueKind::Instruction);
    let dead_inner = f.create_value(HirType::I64, HirValueKind::Instruction);

    {
        let b = f.blocks.get_mut(&entry).unwrap();
        b.terminator = HirTerminator::Branch { target: outer };
        b.successors = vec![outer];
    }
    {
        let b = f.blocks.get_mut(&outer).unwrap();
        b.predecessors = vec![entry, outer_latch];
        b.successors = vec![inner, exit];
        b.phis.push(HirPhi {
            result: i,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (i_next, outer_latch)],
        });
        // Reads only `dead_inner`, which reads only this.
        b.phis.push(HirPhi {
            result: dead_outer,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (dead_inner, outer_latch)],
        });
        b.terminator = HirTerminator::CondBranch {
            condition: t,
            true_target: inner,
            false_target: exit,
        };
    }
    {
        let b = f.blocks.get_mut(&inner).unwrap();
        b.predecessors = vec![outer, inner_body];
        b.successors = vec![inner_body, outer_latch];
        b.phis.push(HirPhi {
            result: k,
            ty: HirType::I64,
            incoming: vec![(zero, outer), (k_next, inner_body)],
        });
        b.phis.push(HirPhi {
            result: dead_inner,
            ty: HirType::I64,
            incoming: vec![(dead_outer, outer), (k_next, inner_body)],
        });
        b.terminator = HirTerminator::CondBranch {
            condition: t,
            true_target: inner_body,
            false_target: outer_latch,
        };
    }
    {
        let b = f.blocks.get_mut(&inner_body).unwrap();
        b.predecessors = vec![inner];
        b.successors = vec![inner];
        b.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: k_next,
            ty: HirType::I64,
            left: k,
            right: one,
        });
        b.terminator = HirTerminator::Branch { target: inner };
    }
    {
        let b = f.blocks.get_mut(&outer_latch).unwrap();
        b.predecessors = vec![inner];
        b.successors = vec![outer];
        b.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: i_next,
            ty: HirType::I64,
            left: i,
            right: one,
        });
        b.terminator = HirTerminator::Branch { target: outer };
    }
    {
        let b = f.blocks.get_mut(&exit).unwrap();
        b.predecessors = vec![outer];
        b.terminator = HirTerminator::Return { values: vec![i] };
    }

    Nest {
        func: f,
        outer_header: outer,
        inner_header: inner,
        live: vec![i, k],
        dead: vec![dead_outer, dead_inner],
    }
}

/// Two phis that hold each other up and nothing else reads are both
/// removed, and the counters beside them are kept.
#[test]
fn a_dead_phi_cycle_across_a_loop_nest_is_removed() {
    let mut n = nested_loop_with_a_dead_cycle();
    let stats = zyntax_compiler::phi_prune::run_function(&mut n.func);

    let remaining: Vec<HirId> = phis_in(&n.func, n.outer_header)
        .into_iter()
        .chain(phis_in(&n.func, n.inner_header))
        .collect();

    for d in &n.dead {
        assert!(
            !remaining.contains(d),
            "a phi read only by another dead phi should be gone; \
             {remaining:?} still holds {d:?}"
        );
    }
    for l in &n.live {
        assert!(
            remaining.contains(l),
            "the counters are read by real instructions and must stay; \
             {remaining:?} is missing {l:?}"
        );
    }
    assert_eq!(
        stats.removed, 2,
        "both members of the cycle count as removed"
    );
}

/// Removing them may not leave a surviving phi naming a value that no
/// longer exists, which is what taking one member out at a time would
/// do.
#[test]
fn no_surviving_phi_names_a_removed_one() {
    let mut n = nested_loop_with_a_dead_cycle();
    zyntax_compiler::phi_prune::run_function(&mut n.func);

    let defined: Vec<HirId> = n
        .func
        .blocks
        .values()
        .flat_map(|b| {
            b.phis
                .iter()
                .map(|p| p.result)
                .chain(b.instructions.iter().filter_map(result_of))
        })
        .collect();

    for b in n.func.blocks.values() {
        for phi in &b.phis {
            for (v, _) in &phi.incoming {
                let known = defined.contains(v)
                    || n.func
                        .values
                        .get(v)
                        .is_some_and(|d| matches!(d.kind, HirValueKind::Constant(_)));
                assert!(
                    known,
                    "phi {:?} still names {v:?}, which is gone",
                    phi.result
                );
            }
        }
    }
}

fn result_of(inst: &HirInstruction) -> Option<HirId> {
    match inst {
        HirInstruction::Binary { result, .. } => Some(*result),
        _ => None,
    }
}

/// A phi a real instruction reads is not touched, however the pass
/// walks the graph.
#[test]
fn a_phi_something_reads_is_kept() {
    let mut n = nested_loop_with_a_dead_cycle();
    // Give the outer dead phi a genuine reader; it must then survive,
    // and so must the inner one it names.
    let outer_latch = n.func.blocks[&n.outer_header]
        .phis
        .iter()
        .find(|p| p.result == n.dead[0])
        .and_then(|p| p.incoming.iter().find(|(_, b)| *b != n.func.entry_block))
        .map(|(_, b)| *b)
        .expect("the dead phi should have a latch incoming");
    let sum = n.func.create_value(HirType::I64, HirValueKind::Instruction);
    n.func
        .blocks
        .get_mut(&outer_latch)
        .unwrap()
        .instructions
        .push(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: sum,
            ty: HirType::I64,
            left: n.dead[0],
            right: n.dead[0],
        });

    zyntax_compiler::phi_prune::run_function(&mut n.func);
    let remaining: Vec<HirId> = phis_in(&n.func, n.outer_header)
        .into_iter()
        .chain(phis_in(&n.func, n.inner_header))
        .collect();
    for d in &n.dead {
        assert!(
            remaining.contains(d),
            "{remaining:?} dropped {d:?}, which is reachable from a real reader"
        );
    }
}
