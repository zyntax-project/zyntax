//! Allocation-backed pointers are disjoint from static global storage, but
//! need not be disjoint from each other or from pointers loaded from globals.

use crate::hir::*;
use crate::hir::{IdHashMap as HashMap, IdHashSet as HashSet};

/// Available constructors whose every return is backed by allocated storage.
/// Missing bodies and recursive call cycles establish no new facts.
pub(crate) fn returning_functions(module: &HirModule) -> HashSet<HirId> {
    let mut known = HashSet::default();
    let candidates: Vec<_> = module
        .functions
        .values()
        .filter(|f| {
            !f.is_external
                && !f.blocks.is_empty()
                && matches!(f.signature.returns.as_slice(), [HirType::Ptr(_)])
        })
        .collect();
    for _ in 0..3 {
        let found: Vec<_> = candidates
            .iter()
            .filter(|f| !known.contains(&f.id))
            .filter(|f| {
                let backed = values(f, &known);
                let mut any = false;
                let all = f
                    .blocks
                    .values()
                    .filter_map(|b| match &b.terminator {
                        HirTerminator::Return { values } => Some(values),
                        _ => None,
                    })
                    .all(|returned| {
                        any = true;
                        matches!(returned.as_slice(), [v] if backed.contains(v))
                    });
                any && all
            })
            .map(|f| f.id)
            .collect();
        if found.is_empty() {
            break;
        }
        known.extend(found);
    }
    known
}

fn integer_bits(ty: &HirType) -> Option<u32> {
    Some(match ty {
        HirType::I8 | HirType::U8 => 8,
        HirType::I16 | HirType::U16 => 16,
        HirType::I32 | HirType::U32 => 32,
        HirType::I64 | HirType::U64 => 64,
        HirType::I128 | HirType::U128 => 128,
        HirType::ISize | HirType::USize => crate::target_pointer_size() as u32 * 8,
        _ => return None,
    })
}

pub(crate) fn preserves_address(op: CastOp, from: &HirType, to: &HirType) -> bool {
    let bits = crate::target_pointer_size() as u32 * 8;
    match op {
        CastOp::Bitcast => {
            matches!((from, to), (HirType::Ptr(_), HirType::Ptr(_)))
                || integer_bits(from).is_some_and(|n| integer_bits(to) == Some(n))
        }
        CastOp::PtrToInt => matches!(from, HirType::Ptr(_)) && integer_bits(to) == Some(bits),
        CastOp::IntToPtr => matches!(to, HirType::Ptr(_)) && integer_bits(from) == Some(bits),
        _ => false,
    }
}

/// Base allocation addresses and lossless representations of them. A phi or
/// select keeps the fact only when every incoming value has it; pointer loads,
/// offsets, arithmetic and unknown calls do not establish provenance.
pub(crate) fn values(f: &HirFunction, callees: &HashSet<HirId>) -> HashSet<HirId> {
    let mut known: HashSet<_> = f
        .signature
        .params
        .iter()
        .filter(|p| p.attributes.allocation_backed && matches!(p.ty, HirType::Ptr(_)))
        .map(|p| p.id)
        .collect();
    let mut dependencies: HashMap<HirId, Vec<HirId>> = HashMap::default();
    for block in f.blocks.values() {
        for inst in &block.instructions {
            let Some(result) = inst.result_id() else {
                continue;
            };
            match inst {
                HirInstruction::Alloca { .. }
                | HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                    ..
                } => {
                    known.insert(result);
                }
                HirInstruction::Call {
                    callee: HirCallable::Function(id),
                    ..
                } if callees.contains(id) => {
                    known.insert(result);
                }
                HirInstruction::Cast {
                    operand, op, ty, ..
                } if f
                    .values
                    .get(operand)
                    .is_some_and(|v| preserves_address(*op, &v.ty, ty)) =>
                {
                    dependencies.insert(result, vec![*operand]);
                }
                HirInstruction::Select {
                    true_val,
                    false_val,
                    ..
                } => {
                    dependencies.insert(result, vec![*true_val, *false_val]);
                }
                _ => {}
            }
        }
        for phi in &block.phis {
            dependencies.insert(phi.result, phi.incoming.iter().map(|(v, _)| *v).collect());
        }
    }
    known.extend(dependencies.keys());
    loop {
        let before = known.len();
        let invalid: Vec<_> = dependencies
            .iter()
            .filter(|(_, inputs)| inputs.is_empty() || inputs.iter().any(|v| !known.contains(v)))
            .map(|(id, _)| *id)
            .collect();
        for id in invalid {
            known.remove(&id);
        }
        if known.len() == before {
            return known;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::InternedString;

    fn pointer() -> HirType {
        HirType::Ptr(std::sync::Arc::new(HirType::U8))
    }
    fn function(name: &str) -> HirFunction {
        HirFunction::new(
            InternedString::new_global(name),
            HirFunctionSignature {
                params: vec![],
                returns: vec![pointer()],
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
    fn call(f: &mut HirFunction, callee: HirCallable, args: Vec<HirId>) -> HirId {
        let result = f.create_value(pointer(), HirValueKind::Instruction);
        f.blocks
            .get_mut(&f.entry_block)
            .unwrap()
            .instructions
            .push(HirInstruction::Call {
                result: Some(result),
                callee,
                args,
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            });
        f.blocks.get_mut(&f.entry_block).unwrap().terminator = HirTerminator::Return {
            values: vec![result],
        };
        result
    }
    #[test]
    fn constructor_chains_prove_storage_but_unknown_and_global_returns_do_not() {
        let mut ctor = function("ctor");
        ctor.attributes.optimized = true;
        let size = ctor.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(8)));
        call(
            &mut ctor,
            HirCallable::Intrinsic(Intrinsic::Malloc),
            vec![size],
        );
        let mut wrapper = function("wrapper");
        call(&mut wrapper, HirCallable::Function(ctor.id), vec![]);
        let mut global = function("global");
        let g = global.create_value(pointer(), HirValueKind::Global(HirId::new()));
        global
            .blocks
            .get_mut(&global.entry_block)
            .unwrap()
            .terminator = HirTerminator::Return { values: vec![g] };
        let mut unknown = function("unknown");
        call(&mut unknown, HirCallable::Function(HirId::new()), vec![]);
        let expected = HashSet::from_iter([ctor.id, wrapper.id]);
        let mut module = HirModule::new(InternedString::new_global("origins"));
        for f in [ctor, wrapper, global, unknown] {
            module.add_function(f);
        }
        assert_eq!(returning_functions(&module), expected);
    }
    #[test]
    fn joining_an_allocation_with_global_storage_loses_the_proof() {
        let mut f = function("select");
        let size = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(8)));
        let object = call(
            &mut f,
            HirCallable::Intrinsic(Intrinsic::Malloc),
            vec![size],
        );
        let g = f.create_value(pointer(), HirValueKind::Global(HirId::new()));
        let condition = f.create_value(HirType::Bool, HirValueKind::Parameter(0));
        let joined = f.create_value(pointer(), HirValueKind::Instruction);
        f.blocks
            .get_mut(&f.entry_block)
            .unwrap()
            .instructions
            .push(HirInstruction::Select {
                result: joined,
                condition,
                true_val: object,
                false_val: g,
                ty: pointer(),
            });
        let known = values(&f, &HashSet::default());
        assert!(known.contains(&object));
        assert!(!known.contains(&joined));
        assert!(!known.contains(&g));
    }
}
