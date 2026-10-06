//! Non-escaping objects carried by phis become scalar field phis.
//!
//! Every root must be a local fixed-size allocation or null. Stores
//! initialize an allocation in its defining block; writes through a phi
//! are excluded. This makes each object an immutable tuple after it leaves
//! its defining block. Its address may only be used for field accesses,
//! null tests, releases and other phis in the same web.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::hir::{
    BinaryOp, CastOp, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule,
    HirPhi, HirTerminator, HirType, HirValueKind, Intrinsic,
};

#[derive(Clone, Copy)]
enum Root {
    Alloc {
        block: HirId,
        index: usize,
        size: u64,
    },
    Phi,
    Null,
}

#[derive(Clone, Copy)]
struct Address {
    root: HirId,
    offset: u64,
    encoded: bool,
}

struct Plan {
    roots: HashMap<HirId, Root>,
    fields: BTreeMap<u64, HirType>,
    final_values: HashMap<(HirId, u64), HirId>,
    loads: Vec<(HirId, HirId, u64)>,
    local_loads: HashMap<HirId, HirId>,
    null_tests: Vec<(HirId, HirId, BinaryOp, HirType)>,
    remove: HashSet<(HirId, usize)>,
}

pub fn run_module(module: &mut HirModule) -> usize {
    // ZYNTAX_DISABLE_HEAP_SCALARIZE keeps reference objects intact; safe.
    if std::env::var_os("ZYNTAX_DISABLE_HEAP_SCALARIZE").is_some() {
        return 0;
    }
    module.functions_to_optimize().map(run_function).sum()
}

pub fn run_function(f: &mut HirFunction) -> usize {
    let mut eliminated = 0;
    let seeds: Vec<_> = f
        .blocks
        .values()
        .flat_map(|b| &b.phis)
        .filter(|p| matches!(p.ty, HirType::Ptr(_)))
        .map(|p| p.result)
        .collect();
    if seeds.is_empty() {
        return 0;
    }
    let mut index = WebIndex::new(f);
    let mut seen = HashSet::new();
    for seed in seeds {
        if seen.contains(&seed) {
            continue;
        }
        let Some(roots) = web(f, &index, seed, &mut seen) else {
            continue;
        };
        if let Some(plan) = plan(f, roots) {
            eliminated += plan
                .roots
                .values()
                .filter(|r| matches!(r, Root::Alloc { .. }))
                .count();
            apply(f, plan);
            index = WebIndex::new(f);
        }
    }
    eliminated
}

fn integer(f: &HirFunction, id: HirId) -> Option<i64> {
    match f.values.get(&id)?.kind {
        HirValueKind::Constant(HirConstant::I64(v)) => Some(v),
        HirValueKind::Constant(HirConstant::ISize(v)) => Some(v),
        HirValueKind::Constant(HirConstant::I32(v)) => Some(v.into()),
        HirValueKind::Constant(HirConstant::U64(v)) => v.try_into().ok(),
        HirValueKind::Constant(HirConstant::USize(v)) => v.try_into().ok(),
        HirValueKind::Constant(HirConstant::U32(v)) => Some(v.into()),
        _ => None,
    }
}

/// Indexed once per unchanged body, including rejected webs.
struct WebIndex {
    phis: HashMap<HirId, Vec<HirId>>,
    users: HashMap<HirId, Vec<HirId>>,
    allocs: HashMap<HirId, Root>,
}

impl WebIndex {
    fn new(f: &HirFunction) -> Self {
        let mut index = Self {
            phis: HashMap::new(),
            users: HashMap::new(),
            allocs: HashMap::new(),
        };
        for (bid, block) in &f.blocks {
            for phi in &block.phis {
                index
                    .phis
                    .insert(phi.result, phi.incoming.iter().map(|(v, _)| *v).collect());
                for (v, _) in &phi.incoming {
                    index.users.entry(*v).or_default().push(phi.result);
                }
            }
            for (i, inst) in block.instructions.iter().enumerate() {
                if let HirInstruction::Call {
                    result: Some(r),
                    callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                    args,
                    ..
                } = inst
                {
                    if let Some(size) = args.first().and_then(|v| integer(f, *v)).filter(|s| *s > 0)
                    {
                        index.allocs.insert(
                            *r,
                            Root::Alloc {
                                block: *bid,
                                index: i,
                                size: size as u64,
                            },
                        );
                    }
                }
            }
        }
        index
    }
}

fn web(
    f: &HirFunction,
    index: &WebIndex,
    seed: HirId,
    seen: &mut HashSet<HirId>,
) -> Option<HashMap<HirId, Root>> {
    let mut roots = HashMap::new();
    let mut work = vec![seed];
    let mut valid = true;
    while let Some(id) = work.pop() {
        if !seen.insert(id) {
            continue;
        }
        let root = if let Some(incoming) = index.phis.get(&id) {
            work.extend(incoming);
            Some(Root::Phi)
        } else if let Some(root) = index.allocs.get(&id) {
            Some(*root)
        } else if matches!(
            f.values.get(&id).map(|v| &v.kind),
            Some(HirValueKind::Constant(HirConstant::Null(_)))
        ) {
            Some(Root::Null)
        } else {
            None
        };
        if let Some(root) = root {
            roots.insert(id, root);
        } else {
            valid = false;
        }
        valid &= f
            .values
            .get(&id)
            .is_some_and(|v| matches!(v.ty, HirType::Ptr(_)));
        // Include every phi sharing a root, even when another incoming
        // already disqualified the web, so it is not analyzed repeatedly.
        if let Some(users) = index.users.get(&id) {
            work.extend(users);
        }
    }
    (valid && roots.values().any(|r| matches!(r, Root::Alloc { .. }))).then_some(roots)
}

fn bytes(ty: &HirType) -> Option<u64> {
    Some(match ty {
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64
        | HirType::U64
        | HirType::ISize
        | HirType::USize
        | HirType::F64
        | HirType::Ptr(_) => 8,
        _ => return None,
    })
}

fn plan(f: &HirFunction, roots: HashMap<HirId, Root>) -> Option<Plan> {
    let mut addresses: HashMap<_, _> = roots
        .keys()
        .map(|r| {
            (
                *r,
                Address {
                    root: *r,
                    offset: 0,
                    encoded: false,
                },
            )
        })
        .collect();
    let mut remove = HashSet::new();
    loop {
        let before = addresses.len();
        for (bid, block) in &f.blocks {
            for (i, inst) in block.instructions.iter().enumerate() {
                let derived = match inst {
                    HirInstruction::GetElementPtr {
                        result,
                        ty,
                        ptr,
                        indices,
                    } => {
                        let Some(base) = addresses.get(ptr).copied() else {
                            continue;
                        };
                        if base.encoded || !matches!(ty, HirType::U8 | HirType::I8) {
                            return None;
                        }
                        let mut offset = base.offset;
                        for index in indices {
                            offset =
                                offset.checked_add(u64::try_from(integer(f, *index)?).ok()?)?;
                        }
                        Some((*result, Address { offset, ..base }))
                    }
                    HirInstruction::Cast {
                        result,
                        operand,
                        op,
                        ty,
                    } => {
                        let Some(base) = addresses.get(operand).copied() else {
                            continue;
                        };
                        let encoded = match op {
                            CastOp::Bitcast if !base.encoded && matches!(ty, HirType::Ptr(_)) => {
                                false
                            }
                            CastOp::PtrToInt
                                if !base.encoded
                                    && matches!(
                                        ty,
                                        HirType::U64
                                            | HirType::I64
                                            | HirType::USize
                                            | HirType::ISize
                                    ) =>
                            {
                                true
                            }
                            CastOp::IntToPtr if base.encoded && matches!(ty, HirType::Ptr(_)) => {
                                false
                            }
                            _ => return None,
                        };
                        Some((*result, Address { encoded, ..base }))
                    }
                    _ => None,
                };
                if let Some((id, addr)) = derived {
                    addresses.insert(id, addr);
                    remove.insert((*bid, i));
                }
            }
        }
        if before == addresses.len() {
            break;
        }
    }
    let mut p = Plan {
        roots,
        fields: BTreeMap::new(),
        final_values: HashMap::new(),
        loads: vec![],
        local_loads: HashMap::new(),
        null_tests: vec![],
        remove,
    };
    let mut stores = HashMap::new();
    for (bid, block) in &f.blocks {
        for phi in &block.phis {
            if !p.roots.contains_key(&phi.result)
                && phi.incoming.iter().any(|(v, _)| addresses.contains_key(v))
            {
                return None;
            }
        }
        let mut bad_term = false;
        block
            .terminator
            .for_each_operand(|v| bad_term |= addresses.contains_key(&v));
        if let HirTerminator::Invoke {
            callee: HirCallable::Indirect(v),
            ..
        } = &block.terminator
        {
            bad_term |= addresses.contains_key(v);
        }
        if bad_term {
            return None;
        }
        for (i, inst) in block.instructions.iter().enumerate() {
            if p.remove.contains(&(*bid, i)) {
                continue;
            }
            if inst
                .result_id()
                .is_some_and(|r| matches!(p.roots.get(&r), Some(Root::Alloc { .. })))
            {
                p.remove.insert((*bid, i));
                continue;
            }
            match inst {
                HirInstruction::Load {
                    result,
                    ptr,
                    ty,
                    volatile: false,
                    ..
                } if addresses.contains_key(ptr) => {
                    let a = addresses[ptr];
                    if a.encoded {
                        return None;
                    }
                    field(&mut p.fields, a.offset, ty)?;
                    if let Root::Alloc { block: home, .. } = p.roots[&a.root]
                        && home == *bid
                    {
                        let value = *stores.get(&(a.root, a.offset))?;
                        p.local_loads.insert(*result, value);
                    } else {
                        p.loads.push((*result, a.root, a.offset));
                    }
                    p.remove.insert((*bid, i));
                }
                HirInstruction::Store {
                    ptr,
                    value,
                    volatile: false,
                    ..
                } if addresses.contains_key(ptr) => {
                    let a = addresses[ptr];
                    let Root::Alloc {
                        block: home, index, ..
                    } = p.roots[&a.root]
                    else {
                        return None;
                    };
                    if a.encoded || home != *bid || i <= index || addresses.contains_key(value) {
                        return None;
                    }
                    field(&mut p.fields, a.offset, &f.values.get(value)?.ty)?;
                    stores.insert((a.root, a.offset), *value);
                    p.remove.insert((*bid, i));
                }
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Free),
                    args,
                    ..
                } if args.len() == 1 && addresses.contains_key(&args[0]) => {
                    let a = addresses[&args[0]];
                    if a.encoded || a.offset != 0 {
                        return None;
                    }
                    p.remove.insert((*bid, i));
                }
                HirInstruction::Binary {
                    result,
                    op: op @ (BinaryOp::Eq | BinaryOp::Ne),
                    left,
                    right,
                    ty,
                } => {
                    let zero = |v| {
                        integer(f, v) == Some(0)
                            || matches!(
                                f.values.get(&v).map(|v| &v.kind),
                                Some(HirValueKind::Constant(HirConstant::Null(_)))
                            )
                    };
                    let a = if zero(*right) {
                        addresses.get(left)
                    } else if zero(*left) {
                        addresses.get(right)
                    } else {
                        None
                    };
                    if let Some(a) = a {
                        if a.offset != 0 {
                            return None;
                        }
                        p.null_tests.push((*result, a.root, *op, ty.clone()));
                    } else if addresses.contains_key(left) || addresses.contains_key(right) {
                        return None;
                    }
                }
                _ => {
                    let mut escapes = false;
                    inst.for_each_operand(|v| escapes |= addresses.contains_key(&v));
                    if let HirInstruction::Call {
                        callee: HirCallable::Indirect(v),
                        ..
                    } = inst
                    {
                        escapes |= addresses.contains_key(v);
                    }
                    if escapes {
                        return None;
                    }
                }
            }
        }
    }
    // Each allocation has initialized storage, so a null result could
    // not have reached its phi. A bare nullable allocation is left alone.
    if p.fields.is_empty() {
        return None;
    }
    // Distinct field types must not be alternate views of overlapping bytes.
    let mut end = 0;
    for (offset, ty) in &p.fields {
        if *offset < end {
            return None;
        }
        end = offset.checked_add(bytes(ty)?)?;
    }
    for (root, kind) in &p.roots {
        if let Root::Alloc { size, .. } = kind {
            if end > *size {
                return None;
            }
            for offset in p.fields.keys() {
                p.final_values
                    .insert((*root, *offset), *stores.get(&(*root, *offset))?);
            }
        }
    }
    Some(p)
}

fn field(fields: &mut BTreeMap<u64, HirType>, offset: u64, ty: &HirType) -> Option<()> {
    bytes(ty)?;
    if let Some(old) = fields.get(&offset)
        && old != ty
    {
        return None;
    }
    fields.insert(offset, ty.clone());
    Some(())
}

fn apply(f: &mut HirFunction, p: Plan) {
    let mut deleted = HashSet::new();
    let mut values = p.final_values;
    let mut present = HashMap::new();
    let mut subs = p.local_loads;
    for (root, kind) in &p.roots {
        let value = match kind {
            Root::Phi => f.create_value(HirType::I64, HirValueKind::Instruction),
            Root::Alloc { .. } => {
                f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(1)))
            }
            Root::Null => f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0))),
        };
        present.insert(*root, value);
        if !matches!(kind, Root::Alloc { .. }) {
            for (offset, ty) in &p.fields {
                let kind = if matches!(kind, Root::Null) {
                    HirValueKind::Undef
                } else {
                    HirValueKind::Instruction
                };
                values.insert((*root, *offset), f.create_value(ty.clone(), kind));
            }
        }
    }
    for (result, root, offset) in &p.loads {
        subs.insert(*result, values[&(*root, *offset)]);
    }
    let tests: HashMap<_, _> = p
        .null_tests
        .iter()
        .map(|(r, root, op, ty)| (*r, (*root, *op, ty.clone())))
        .collect();
    let zero = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
    for (bid, block) in &mut f.blocks {
        let mut phis = Vec::new();
        for phi in std::mem::take(&mut block.phis) {
            if !matches!(p.roots.get(&phi.result), Some(Root::Phi)) {
                phis.push(phi);
                continue;
            }
            deleted.insert(phi.result);
            for (offset, ty) in &p.fields {
                phis.push(HirPhi {
                    result: values[&(phi.result, *offset)],
                    ty: ty.clone(),
                    incoming: phi
                        .incoming
                        .iter()
                        .map(|(v, b)| (values[&(*v, *offset)], *b))
                        .collect(),
                });
            }
            phis.push(HirPhi {
                result: present[&phi.result],
                ty: HirType::I64,
                incoming: phi.incoming.iter().map(|(v, b)| (present[v], *b)).collect(),
            });
        }
        block.phis = phis;
        block.instructions = std::mem::take(&mut block.instructions)
            .into_iter()
            .enumerate()
            .filter_map(|(i, inst)| {
                if p.remove.contains(&(*bid, i)) {
                    if let Some(result) = inst.result_id() {
                        deleted.insert(result);
                    }
                    return None;
                }
                if let Some(result) = inst.result_id()
                    && let Some((root, op, ty)) = tests.get(&result)
                {
                    return Some(HirInstruction::Binary {
                        result,
                        op: *op,
                        ty: ty.clone(),
                        left: present[root],
                        right: zero,
                    });
                }
                Some(inst)
            })
            .collect();
    }
    crate::cse::apply_substitutions_public(f, &subs);
    f.values.retain(|id, _| !deleted.contains(id));
}
