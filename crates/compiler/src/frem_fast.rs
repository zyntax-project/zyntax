//! Float remainder fast path.
//!
//! No target has a floating-point remainder instruction, so `FRem` becomes a libm `fmod`
//! call in every backend. Inside a loop that call dominates programs that take a remainder
//! of whole numbers, which is the common case. A site is expanded into a range-and-integrality
//! test, an integer remainder when both operands are positive integers below 2^63, and the
//! original `FRem` otherwise:
//!
//! * every other operand (zero, negative, fractional, NaN, infinite, huge) takes the exact
//!   `FRem`, so results, signs and negative zeros are unchanged;
//! * the dividend is tested against a bound below 2^63 before it is used as an integer, since
//!   a saturated conversion of a larger value would compare equal to a whole number;
//! * only sites inside a loop are expanded, so cold code keeps its single instruction.

use crate::hir::*;
use std::collections::{HashMap, HashSet};

/// Below `i64::MAX`, so a tested operand converts without saturating.
const LIMIT: f64 = 9.0e18;

fn constant(f: &mut HirFunction, ty: HirType, value: HirConstant) -> HirId {
    f.create_value(ty, HirValueKind::Constant(value))
}

fn instruction(f: &mut HirFunction, ty: HirType) -> HirId {
    f.create_value(ty, HirValueKind::Instruction)
}

/// The divisor when it is a constant whole number a conversion represents exactly.
fn whole_divisor(f: &HirFunction, id: HirId) -> Option<i64> {
    match f.values.get(&id)?.kind {
        HirValueKind::Constant(HirConstant::F64(c))
            if c >= 1.0 && c <= 9_007_199_254_740_992.0 && c.fract() == 0.0 =>
        {
            Some(c as i64)
        }
        _ => None,
    }
}

/// Blocks that lie on a cycle, by strongly connected components.
fn cyclic_blocks(f: &HirFunction) -> HashSet<HirId> {
    let ids: Vec<HirId> = f.blocks.keys().copied().collect();
    let index_of: HashMap<HirId, usize> = ids.iter().enumerate().map(|(i, b)| (*b, i)).collect();
    let successors: Vec<Vec<usize>> = ids
        .iter()
        .map(|b| {
            f.blocks[b]
                .terminator
                .targets()
                .iter()
                .filter_map(|t| index_of.get(t).copied())
                .collect()
        })
        .collect();
    let n = ids.len();
    let (mut order, mut low) = (vec![usize::MAX; n], vec![0usize; n]);
    let (mut on_stack, mut stack) = (vec![false; n], Vec::new());
    let mut counter = 0;
    let mut cyclic = HashSet::new();
    for root in 0..n {
        if order[root] != usize::MAX {
            continue;
        }
        // Iterative Tarjan: (node, next successor to visit).
        let mut work = vec![(root, 0usize)];
        order[root] = counter;
        low[root] = counter;
        counter += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (node, ref mut next)) = work.last_mut() {
            if let Some(&succ) = successors[node].get(*next) {
                *next += 1;
                if order[succ] == usize::MAX {
                    order[succ] = counter;
                    low[succ] = counter;
                    counter += 1;
                    stack.push(succ);
                    on_stack[succ] = true;
                    work.push((succ, 0));
                } else if on_stack[succ] {
                    low[node] = low[node].min(order[succ]);
                }
                continue;
            }
            work.pop();
            if let Some(&(parent, _)) = work.last() {
                low[parent] = low[parent].min(low[node]);
            }
            if low[node] == order[node] {
                let mut component = Vec::new();
                while let Some(member) = stack.pop() {
                    on_stack[member] = false;
                    component.push(member);
                    if member == node {
                        break;
                    }
                }
                if component.len() > 1 || successors[node].contains(&node) {
                    cyclic.extend(component.into_iter().map(|m| ids[m]));
                }
            }
        }
    }
    cyclic
}

struct Site {
    block: HirId,
    index: usize,
}

fn find_site(f: &HirFunction, block: HirId) -> Option<Site> {
    f.blocks[&block]
        .instructions
        .iter()
        .position(|i| {
            matches!(
                i,
                HirInstruction::Binary {
                    op: BinaryOp::FRem,
                    ty: HirType::F64,
                    ..
                }
            )
        })
        .map(|index| Site { block, index })
}

/// Splits `site`; returns the join block, which holds everything after the site.
fn expand(f: &mut HirFunction, site: Site, created: &mut Vec<HirId>) -> HirId {
    let Site { block, index } = site;
    let mut tail_instructions = f
        .blocks
        .get_mut(&block)
        .unwrap()
        .instructions
        .split_off(index);
    let HirInstruction::Binary {
        result,
        left: x,
        right: y,
        ..
    } = tail_instructions.remove(0)
    else {
        unreachable!("find_site returned a float remainder");
    };
    let divisor = whole_divisor(f, y);
    let zero = constant(f, HirType::F64, HirConstant::F64(0.0));
    let limit = constant(f, HirType::F64, HirConstant::F64(LIMIT));

    let slow = f.create_block();
    let fast = f.create_block();
    let tail = f.create_block();
    created.extend([slow, fast, tail]);

    // The original terminator moves to the join block, so its successors' phis follow it.
    let terminator = std::mem::replace(
        &mut f.blocks.get_mut(&block).unwrap().terminator,
        HirTerminator::Unreachable,
    );
    for successor in terminator.targets() {
        for phi in &mut f.blocks.get_mut(&successor).unwrap().phis {
            for (_, predecessor) in &mut phi.incoming {
                if *predecessor == block {
                    *predecessor = tail;
                }
            }
        }
    }

    // One comparison per block, each leaving through `slow` when it fails.
    let mut current = block;
    let mut test = |f: &mut HirFunction, op, left, right| {
        let condition = instruction(f, HirType::Bool);
        f.blocks
            .get_mut(&current)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op,
                result: condition,
                ty: HirType::F64,
                left,
                right,
            });
        let next = f.create_block();
        created.push(next);
        f.blocks.get_mut(&current).unwrap().terminator = HirTerminator::CondBranch {
            condition,
            true_target: next,
            false_target: slow,
        };
        current = next;
    };
    test(f, BinaryOp::FGt, x, zero);
    test(f, BinaryOp::FLt, x, limit);
    if divisor.is_none() {
        test(f, BinaryOp::FGt, y, zero);
        test(f, BinaryOp::FLt, y, limit);
    }

    // Both operands are in range, so the conversions are exact.
    let whole_x = instruction(f, HirType::I64);
    let back_x = instruction(f, HirType::F64);
    let whole_y = match divisor {
        Some(value) => constant(f, HirType::I64, HirConstant::I64(value)),
        None => instruction(f, HirType::I64),
    };
    let back_y = divisor.is_none().then(|| instruction(f, HirType::F64));
    let mut emit = |f: &mut HirFunction, instruction: HirInstruction| {
        f.blocks
            .get_mut(&current)
            .unwrap()
            .instructions
            .push(instruction);
    };
    emit(
        f,
        HirInstruction::Cast {
            result: whole_x,
            ty: HirType::I64,
            op: CastOp::FpToSi,
            operand: x,
        },
    );
    emit(
        f,
        HirInstruction::Cast {
            result: back_x,
            ty: HirType::F64,
            op: CastOp::SiToFp,
            operand: whole_x,
        },
    );
    if let Some(back_y) = back_y {
        emit(
            f,
            HirInstruction::Cast {
                result: whole_y,
                ty: HirType::I64,
                op: CastOp::FpToSi,
                operand: y,
            },
        );
        emit(
            f,
            HirInstruction::Cast {
                result: back_y,
                ty: HirType::F64,
                op: CastOp::SiToFp,
                operand: whole_y,
            },
        );
    }
    let mut integral = |f: &mut HirFunction, converted, original, current_block: &mut HirId| {
        let condition = instruction(f, HirType::Bool);
        f.blocks
            .get_mut(current_block)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::FEq,
                result: condition,
                ty: HirType::F64,
                left: converted,
                right: original,
            });
        let next = f.create_block();
        created.push(next);
        f.blocks.get_mut(current_block).unwrap().terminator = HirTerminator::CondBranch {
            condition,
            true_target: next,
            false_target: slow,
        };
        *current_block = next;
    };
    let mut at = current;
    integral(f, back_x, x, &mut at);
    if let Some(back_y) = back_y {
        integral(f, back_y, y, &mut at);
    }
    f.blocks.get_mut(&at).unwrap().terminator = HirTerminator::Branch { target: fast };

    // Integer remainder: the dividend is positive, so the result has its sign.
    let remainder = instruction(f, HirType::I64);
    let converted = instruction(f, HirType::F64);
    f.blocks.get_mut(&fast).unwrap().instructions = vec![
        HirInstruction::Binary {
            op: BinaryOp::Rem,
            result: remainder,
            ty: HirType::I64,
            left: whole_x,
            right: whole_y,
        },
        HirInstruction::Cast {
            result: converted,
            ty: HirType::F64,
            op: CastOp::SiToFp,
            operand: remainder,
        },
    ];
    f.blocks.get_mut(&fast).unwrap().terminator = HirTerminator::Branch { target: tail };

    // The exact remainder.
    let exact = instruction(f, HirType::F64);
    f.blocks.get_mut(&slow).unwrap().instructions = vec![HirInstruction::Binary {
        op: BinaryOp::FRem,
        result: exact,
        ty: HirType::F64,
        left: x,
        right: y,
    }];
    f.blocks.get_mut(&slow).unwrap().terminator = HirTerminator::Branch { target: tail };

    // The join keeps the original result name, so every use stays valid.
    let join = f.blocks.get_mut(&tail).unwrap();
    join.phis.push(HirPhi {
        result,
        ty: HirType::F64,
        incoming: vec![(converted, fast), (exact, slow)],
    });
    join.instructions = tail_instructions;
    join.terminator = terminator;
    tail
}

pub fn run_function(f: &mut HirFunction) -> usize {
    if !f.blocks.values().flat_map(|b| &b.instructions).any(|i| {
        matches!(
            i,
            HirInstruction::Binary {
                op: BinaryOp::FRem,
                ty: HirType::F64,
                ..
            }
        )
    }) {
        return 0;
    }
    let mut in_loop = cyclic_blocks(f);
    // ZYNTAX_TRACE_FREM_FAST=1 reports each function's remainders and how many sit in loops.
    if std::env::var_os("ZYNTAX_TRACE_FREM_FAST").is_some() {
        let sites = |blocks: &mut dyn Iterator<Item = &HirBlock>| {
            blocks
                .flat_map(|b| &b.instructions)
                .filter(|i| {
                    matches!(
                        i,
                        HirInstruction::Binary {
                            op: BinaryOp::FRem,
                            ty: HirType::F64,
                            ..
                        }
                    )
                })
                .count()
        };
        let all = sites(&mut f.blocks.values());
        let looped = sites(
            &mut f
                .blocks
                .iter()
                .filter(|(id, _)| in_loop.contains(id))
                .map(|(_, b)| b),
        );
        eprintln!(
            "[frem_fast] {}: {all} float remainders, {looped} in loops",
            f.name.resolve_global().unwrap_or_default()
        );
    }
    let mut pending: Vec<HirId> = f
        .blocks
        .keys()
        .copied()
        .filter(|b| in_loop.contains(b))
        .collect();
    let mut expanded = 0;
    while let Some(block) = pending.pop() {
        let Some(site) = find_site(f, block) else {
            continue;
        };
        let mut created = Vec::new();
        let tail = expand(f, site, &mut created);
        // The new blocks sit on the same cycles as the block they came from.
        in_loop.extend(created);
        expanded += 1;
        // The rest of the block may hold more sites.
        pending.push(tail);
    }
    if expanded > 0 {
        f.rebuild_cfg_edges();
    }
    expanded
}

pub fn run_module(m: &mut HirModule) -> usize {
    // ZYNTAX_DISABLE_FREM_FAST_PATH keeps every float remainder a single instruction; safe.
    if std::env::var_os("ZYNTAX_DISABLE_FREM_FAST_PATH").is_some() {
        return 0;
    }
    m.functions_to_optimize().map(run_function).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use zyntax_typed_ast::InternedString;

    /// `loop: r = x % y; branch cond ? loop : exit`, plus an `FRem` outside any loop.
    fn function(divisor: Option<f64>) -> (HirFunction, HirId, HirId, HirId) {
        let mut f = HirFunction::new(
            InternedString::new_global("remainder"),
            HirFunctionSignature {
                params: vec![],
                returns: vec![HirType::F64],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
                is_fiber: false,
                effects: vec![],
                is_pure: false,
            },
        );
        let x = f.create_value(HirType::F64, HirValueKind::Parameter(0));
        let y = match divisor {
            Some(c) => f.create_value(HirType::F64, HirValueKind::Constant(HirConstant::F64(c))),
            None => f.create_value(HirType::F64, HirValueKind::Parameter(1)),
        };
        let inside = instruction(&mut f, HirType::F64);
        let outside = instruction(&mut f, HirType::F64);
        let flag = instruction(&mut f, HirType::Bool);
        let (entry, body, exit) = (HirId::new(), HirId::new(), HirId::new());
        let mut e = HirBlock::new(entry);
        e.instructions = vec![HirInstruction::Binary {
            op: BinaryOp::FRem,
            result: outside,
            ty: HirType::F64,
            left: x,
            right: y,
        }];
        e.terminator = HirTerminator::Branch { target: body };
        let mut b = HirBlock::new(body);
        b.instructions = vec![
            HirInstruction::Binary {
                op: BinaryOp::FRem,
                result: inside,
                ty: HirType::F64,
                left: x,
                right: y,
            },
            HirInstruction::Binary {
                op: BinaryOp::FGt,
                result: flag,
                ty: HirType::F64,
                left: inside,
                right: x,
            },
        ];
        b.terminator = HirTerminator::CondBranch {
            condition: flag,
            true_target: body,
            false_target: exit,
        };
        let mut x_block = HirBlock::new(exit);
        x_block.terminator = HirTerminator::Return {
            values: vec![inside],
        };
        f.blocks = IndexMap::default();
        f.blocks.insert(entry, e);
        f.blocks.insert(body, b);
        f.blocks.insert(exit, x_block);
        f.entry_block = entry;
        (f, entry, body, inside)
    }

    fn remainders(f: &HirFunction) -> usize {
        f.blocks
            .values()
            .flat_map(|b| &b.instructions)
            .filter(|i| {
                matches!(
                    i,
                    HirInstruction::Binary {
                        op: BinaryOp::FRem,
                        ..
                    }
                )
            })
            .count()
    }

    #[test]
    fn a_remainder_in_a_loop_gets_an_integer_path_and_one_outside_does_not() {
        let (mut f, entry, _, inside) = function(None);
        assert_eq!(run_function(&mut f), 1);
        // The exact remainder survives on the slow path, and the one outside is untouched.
        assert_eq!(remainders(&f), 2);
        assert!(f.blocks[&entry].instructions.iter().any(|i| matches!(
            i,
            HirInstruction::Binary {
                op: BinaryOp::FRem,
                ..
            }
        )));
        let join = f
            .blocks
            .values()
            .find(|b| b.phis.iter().any(|p| p.result == inside))
            .expect("the join defines the original result");
        assert_eq!(join.phis[0].incoming.len(), 2);
        let integer_remainders = f
            .blocks
            .values()
            .flat_map(|b| &b.instructions)
            .filter(|i| {
                matches!(
                    i,
                    HirInstruction::Binary {
                        op: BinaryOp::Rem,
                        ty: HirType::I64,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(integer_remainders, 1);
    }

    #[test]
    fn the_original_loop_back_edge_leaves_the_join_block() {
        let (mut f, _, body, inside) = function(None);
        run_function(&mut f);
        let join = f
            .blocks
            .iter()
            .find(|(_, b)| b.phis.iter().any(|p| p.result == inside))
            .map(|(id, _)| *id)
            .unwrap();
        // The loop's branch back to its header now comes from the join block.
        assert!(matches!(
            f.blocks[&join].terminator,
            HirTerminator::CondBranch { true_target, .. } if true_target == body
        ));
        assert!(f.blocks[&body].predecessors.contains(&join));
    }

    #[test]
    fn a_constant_whole_divisor_skips_the_divisor_tests() {
        let (mut general, ..) = function(None);
        let (mut constant_divisor, ..) = function(Some(7.0));
        run_function(&mut general);
        run_function(&mut constant_divisor);
        assert!(constant_divisor.blocks.len() < general.blocks.len());
        // A fractional constant divisor has no integer form and takes the general path.
        let (mut fractional, ..) = function(Some(0.5));
        run_function(&mut fractional);
        assert_eq!(fractional.blocks.len(), general.blocks.len());
    }
}
