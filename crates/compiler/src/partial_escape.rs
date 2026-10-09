//! Immutable allocation webs stay in scalar fields until a consuming call.
//!
//! Opaque phi inputs retain their backing pointer and are read at the original
//! load sites. A virtual object is materialized only when its identity escapes;
//! all aliases must be dead after that call, so subsequent mutation or retention
//! is handled entirely by the runtime object.

use crate::hir::*;
use std::collections::{BTreeMap, HashMap, HashSet};

type Site = (HirId, usize);
#[derive(Clone, Copy)]
struct Address {
    root: HirId,
    offset: u64,
    encoded: bool,
}
#[derive(Clone, Copy)]
struct Allocation {
    block: HirId,
    index: usize,
    size: u64,
}
struct Plan {
    roots: HashSet<HirId>,
    allocs: HashMap<HirId, Allocation>,
    addresses: HashMap<HirId, Address>,
    fields: BTreeMap<u64, HirType>,
    stores: HashMap<(HirId, u64), HirId>,
    remove: HashSet<Site>,
    local: HashMap<HirId, HirId>,
    escapes: HashMap<Site, HirId>,
}

pub fn run_module(module: &mut HirModule) -> usize {
    // ZYNTAX_DISABLE_PARTIAL_ESCAPE preserves eager allocations; safe.
    if std::env::var_os("ZYNTAX_DISABLE_PARTIAL_ESCAPE").is_some() {
        return 0;
    }
    module.functions_to_optimize().map(run_function).sum()
}

fn number(f: &HirFunction, id: HirId) -> Option<u64> {
    match &f.values.get(&id)?.kind {
        HirValueKind::Constant(HirConstant::I64(n)) => (*n).try_into().ok(),
        HirValueKind::Constant(HirConstant::U64(n)) => Some(*n),
        HirValueKind::Constant(HirConstant::ISize(n)) => (*n).try_into().ok(),
        HirValueKind::Constant(HirConstant::USize(n)) => Some(*n as u64),
        _ => None,
    }
}
fn size(ty: &HirType) -> Option<u64> {
    Some(match ty {
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 => 8,
        HirType::ISize | HirType::USize | HirType::Ptr(_) => crate::target_pointer_size() as u64,
        _ => return None,
    })
}
fn raw() -> HirType {
    HirType::Ptr(Box::new(HirType::U8))
}
fn field_type(ty: &HirType) -> HirType {
    if matches!(ty, HirType::Ptr(_)) {
        raw()
    } else {
        ty.clone()
    }
}
fn field(fields: &mut BTreeMap<u64, HirType>, offset: u64, ty: &HirType) -> Option<()> {
    size(ty)?;
    let ty = field_type(ty);
    if fields.get(&offset).is_some_and(|old| *old != ty) {
        return None;
    }
    fields.insert(offset, ty);
    Some(())
}
fn zero(f: &HirFunction, id: HirId) -> bool {
    number(f, id) == Some(0)
        || matches!(
            f.values.get(&id).map(|v| &v.kind),
            Some(HirValueKind::Constant(HirConstant::Null(_)))
        )
}

pub fn run_function(f: &mut HirFunction) -> usize {
    if f.blocks.len() > 512 {
        return 0;
    }
    let original: HashSet<_> = f
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .filter_map(|i| match i {
            HirInstruction::Call {
                result: Some(r),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                ..
            } => Some(*r),
            _ => None,
        })
        .collect();
    if original.len() < 2 {
        return 0;
    }
    let mut count = 0;
    let mut seen: HashSet<HirId> = HashSet::new();
    // A rewrite changes phi uses, so rebuild the index after each accepted web.
    for _ in 0..8 {
        let defs: HashMap<_, _> = f
            .blocks
            .values()
            .flat_map(|b| &b.instructions)
            .filter_map(|i| i.result_id().map(|r| (r, i)))
            .collect();
        let origin = |mut id: HirId| {
            for _ in 0..32 {
                id = match defs.get(&id) {
                    Some(HirInstruction::Cast {
                        operand,
                        op: CastOp::Bitcast,
                        ty: HirType::Ptr(_),
                        ..
                    }) if matches!(f.values[operand].ty, HirType::Ptr(_)) => *operand,
                    Some(HirInstruction::Cast {
                        operand,
                        op: CastOp::IntToPtr,
                        ..
                    }) => match defs.get(operand) {
                        Some(HirInstruction::Cast {
                            operand: source,
                            op: CastOp::PtrToInt,
                            ty,
                            ..
                        }) if size(ty)
                            .is_some_and(|n| n >= crate::target_pointer_size() as u64) =>
                        {
                            *source
                        }
                        _ => break,
                    },
                    _ => break,
                };
            }
            id
        };
        let allocs: HashMap<_, _> = f
            .blocks
            .iter()
            .flat_map(|(bid, b)| {
                b.instructions.iter().enumerate().filter_map(|(i, inst)| {
                    if let HirInstruction::Call {
                        result: Some(r),
                        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                        args,
                        ..
                    } = inst
                    {
                        let n = number(f, *args.first()?)?;
                        (n > 0 && n <= 256).then_some((
                            *r,
                            Allocation {
                                block: *bid,
                                index: i,
                                size: n,
                            },
                        ))
                    } else {
                        None
                    }
                })
            })
            .collect();
        let phis: HashMap<_, _> = f
            .blocks
            .values()
            .flat_map(|b| &b.phis)
            .filter(|p| matches!(p.ty, HirType::Ptr(_)))
            .map(|p| (p.result, p))
            .collect();
        let mut users: HashMap<HirId, Vec<HirId>> = HashMap::new();
        for p in phis.values() {
            for (v, _) in &p.incoming {
                users.entry(origin(*v)).or_default().push(p.result);
            }
        }
        let mut accepted = None;
        // Preserve instruction order for deterministic candidate selection.
        let seeds: Vec<_> = f
            .blocks
            .values()
            .flat_map(|b| &b.instructions)
            .filter_map(|i| i.result_id())
            .filter(|r| allocs.contains_key(r))
            .collect();
        for seed in seeds {
            if seen.contains(&seed) || !original.contains(&seed) {
                continue;
            }
            let mut roots = HashSet::new();
            let mut work = vec![seed];
            while let Some(r) = work.pop() {
                if !(allocs.contains_key(&r) || phis.contains_key(&r)) || !roots.insert(r) {
                    continue;
                }
                if let Some(p) = phis.get(&r) {
                    work.extend(p.incoming.iter().map(|(v, _)| origin(*v)));
                }
                if let Some(ps) = users.get(&r) {
                    work.extend(ps);
                }
            }
            seen.extend(roots.iter().copied());
            let selected: HashMap<_, _> = roots
                .iter()
                .filter_map(|r| allocs.get(r).map(|a| (*r, *a)))
                .collect();
            // Do not resink the materialization diamonds this pass creates.
            if selected.keys().any(|r| !original.contains(r))
                || selected.len() < 2
                || roots.len() > 32
                || roots.len() == selected.len()
            {
                continue;
            }
            if let Some(p) = plan(f, roots, selected) {
                accepted = Some(p);
                break;
            }
        }
        let Some(p) = accepted else {
            break;
        };
        count += p.allocs.len();
        apply(f, p);
    }
    count
}

fn plan(
    f: &HirFunction,
    roots: HashSet<HirId>,
    allocs: HashMap<HirId, Allocation>,
) -> Option<Plan> {
    let bytes = allocs.values().next()?.size;
    if allocs.values().any(|a| a.size != bytes) {
        return None;
    }
    let mut p = Plan {
        addresses: roots
            .iter()
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
            .collect(),
        roots,
        allocs,
        fields: BTreeMap::new(),
        stores: HashMap::new(),
        remove: HashSet::new(),
        local: HashMap::new(),
        escapes: HashMap::new(),
    };
    for _ in 0..32 {
        let before = p.addresses.len();
        for (bid, b) in &f.blocks {
            for (i, inst) in b.instructions.iter().enumerate() {
                let (result, address) = match inst {
                    HirInstruction::GetElementPtr {
                        result,
                        ptr,
                        indices,
                        ty,
                    } if p.addresses.contains_key(ptr) => {
                        let mut a = p.addresses[ptr];
                        if a.encoded
                            || !matches!(ty, HirType::U8 | HirType::I8)
                            || indices.len() != 1
                        {
                            return None;
                        }
                        a.offset = a.offset.checked_add(number(f, indices[0])?)?;
                        (*result, a)
                    }
                    HirInstruction::Cast {
                        result,
                        operand,
                        op,
                        ty,
                    } if p.addresses.contains_key(operand) => {
                        let mut a = p.addresses[operand];
                        a.encoded = match op {
                            CastOp::Bitcast if !a.encoded && matches!(ty, HirType::Ptr(_)) => false,
                            CastOp::PtrToInt
                                if !a.encoded
                                    && matches!(
                                        ty,
                                        HirType::I64
                                            | HirType::U64
                                            | HirType::ISize
                                            | HirType::USize
                                    )
                                    && size(ty)? >= crate::target_pointer_size() as u64 =>
                            {
                                true
                            }
                            CastOp::IntToPtr if a.encoded && matches!(ty, HirType::Ptr(_)) => false,
                            _ => return None,
                        };
                        (*result, a)
                    }
                    _ => continue,
                };
                p.addresses.insert(result, address);
                p.remove.insert((*bid, i));
            }
        }
        if before == p.addresses.len() {
            break;
        }
    }
    for (bid, b) in &f.blocks {
        for phi in &b.phis {
            for (v, _) in &phi.incoming {
                if let Some(a) = p.addresses.get(v) {
                    if !p.roots.contains(&phi.result) || a.offset != 0 || a.encoded {
                        return None;
                    }
                }
            }
        }
        let mut term_use = false;
        b.terminator
            .for_each_operand(|v| term_use |= p.addresses.contains_key(&v));
        if let HirTerminator::Invoke {
            callee: HirCallable::Indirect(v),
            ..
        } = &b.terminator
        {
            term_use |= p.addresses.contains_key(v);
        }
        if term_use {
            return None;
        }
        for (i, inst) in b.instructions.iter().enumerate() {
            let site = (*bid, i);
            if p.remove.contains(&site) {
                continue;
            }
            if inst.result_id().is_some_and(|r| p.allocs.contains_key(&r)) {
                p.remove.insert(site);
                continue;
            }
            match inst {
                HirInstruction::Store {
                    ptr,
                    value,
                    volatile: false,
                    ..
                } if p.addresses.contains_key(ptr) => {
                    let a = p.addresses[ptr];
                    let home = p.allocs.get(&a.root)?;
                    if a.encoded
                        || home.block != *bid
                        || i <= home.index
                        || p.addresses.contains_key(value)
                    {
                        return None;
                    }
                    field(&mut p.fields, a.offset, &f.values.get(value)?.ty)?;
                    p.stores.insert((a.root, a.offset), *value);
                    p.remove.insert(site);
                }
                HirInstruction::Load {
                    result,
                    ptr,
                    ty,
                    volatile: false,
                    ..
                } if p.addresses.contains_key(ptr) => {
                    let a = p.addresses[ptr];
                    if a.encoded {
                        return None;
                    }
                    field(&mut p.fields, a.offset, ty)?;
                    if p.allocs.get(&a.root).is_some_and(|home| home.block == *bid) {
                        let v = *p.stores.get(&(a.root, a.offset))?;
                        if f.values[&v].ty != *ty {
                            return None;
                        }
                        p.local.insert(*result, v);
                        p.remove.insert(site);
                    }
                }
                HirInstruction::Binary {
                    op: BinaryOp::Eq | BinaryOp::Ne,
                    left,
                    right,
                    ..
                } if p.addresses.contains_key(left) || p.addresses.contains_key(right) => {
                    let a = if zero(f, *left) {
                        p.addresses.get(right)?
                    } else if zero(f, *right) {
                        p.addresses.get(left)?
                    } else {
                        return None;
                    };
                    if a.offset != 0 {
                        return None;
                    }
                }
                HirInstruction::Call {
                    callee: HirCallable::Symbol(_) | HirCallable::Function(_),
                    args,
                    ..
                } if args.iter().any(|a| p.addresses.contains_key(a)) => {
                    let mut root = None;
                    for arg in args {
                        if let Some(a) = p.addresses.get(arg) {
                            if a.offset != 0 || root.is_some_and(|r| r != a.root) {
                                return None;
                            }
                            root = Some(a.root);
                        }
                    }
                    p.escapes.insert(site, root?);
                }
                _ if inst.any_operand(|v| p.addresses.contains_key(&v)) => return None,
                _ => {}
            }
        }
    }
    if p.escapes.is_empty() || p.fields.is_empty() || p.fields.len() > 12 {
        return None;
    }
    let mut end = 0;
    for (off, ty) in &p.fields {
        if *off != end {
            return None;
        }
        end = off.checked_add(size(ty)?)?;
    }
    if end != bytes {
        return None;
    }
    for r in p.allocs.keys() {
        for off in p.fields.keys() {
            p.stores.get(&(*r, *off))?;
        }
    }
    // Materialization-only objects have no scalar reads to eliminate.
    // In particular, leave the diamonds generated by this pass intact.
    let mut splits = p.escapes.len();
    let read_roots: HashSet<_> = f
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .filter_map(|inst| match inst {
            HirInstruction::Load { result, ptr, .. } if !p.local.contains_key(result) => {
                p.addresses.get(ptr).map(|a| {
                    splits += 1;
                    a.root
                })
            }
            _ => None,
        })
        .collect();
    // Bound the control-flow expansion before constructing diamonds.
    if f.blocks.len() + 3 * splits > 512 {
        return None;
    }
    let mut users: HashMap<HirId, Vec<HirId>> = HashMap::new();
    for phi in f.blocks.values().flat_map(|b| &b.phis) {
        if p.roots.contains(&phi.result) {
            for (v, _) in &phi.incoming {
                if let Some(a) = p.addresses.get(v) {
                    users.entry(a.root).or_default().push(phi.result);
                }
            }
        }
    }
    for root in p.allocs.keys() {
        let mut reached = HashSet::new();
        let mut work = vec![*root];
        while let Some(r) = work.pop() {
            if reached.insert(r) {
                if let Some(next) = users.get(&r) {
                    work.extend(next);
                }
            }
        }
        if reached.is_disjoint(&read_roots) {
            return None;
        }
    }
    consuming_escapes(f, &p).then_some(p)
}

fn consuming_escapes(f: &HirFunction, p: &Plan) -> bool {
    let mut live_in: HashMap<HirId, HashSet<HirId>> =
        f.blocks.keys().map(|b| (*b, HashSet::new())).collect();
    let live_out = |bid: HirId, ins: &HashMap<HirId, HashSet<HirId>>| {
        let mut out = HashSet::new();
        for succ in f.blocks[&bid].terminator.targets() {
            out.extend(&ins[&succ]);
            for phi in &f.blocks[&succ].phis {
                for (v, pred) in &phi.incoming {
                    if *pred == bid && p.addresses.contains_key(v) {
                        out.insert(*v);
                    }
                }
            }
        }
        out
    };
    loop {
        let mut changed = false;
        for (bid, b) in f.blocks.iter().rev() {
            let mut live = live_out(*bid, &live_in);
            for inst in b.instructions.iter().rev() {
                if let Some(r) = inst.result_id() {
                    live.remove(&r);
                }
                inst.for_each_operand(|v| {
                    if p.addresses.contains_key(&v) {
                        live.insert(v);
                    }
                });
            }
            for phi in &b.phis {
                live.remove(&phi.result);
            }
            if live_in[bid] != live {
                live_in.insert(*bid, live);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for (bid, b) in &f.blocks {
        let mut live = live_out(*bid, &live_in);
        for (i, inst) in b.instructions.iter().enumerate().rev() {
            if p.escapes.contains_key(&(*bid, i)) && !live.is_empty() {
                return false;
            }
            if let Some(r) = inst.result_id() {
                live.remove(&r);
            }
            inst.for_each_operand(|v| {
                if p.addresses.contains_key(&v) {
                    live.insert(v);
                }
            });
        }
    }
    true
}

fn value(f: &mut HirFunction, ty: HirType) -> HirId {
    f.create_value(ty, HirValueKind::Instruction)
}
fn constant(f: &mut HirFunction, n: u64) -> HirId {
    f.create_value(HirType::U64, HirValueKind::Constant(HirConstant::U64(n)))
}
fn cast(f: &mut HirFunction, out: &mut Vec<HirInstruction>, v: HirId, ty: HirType) -> HirId {
    if f.values[&v].ty == ty {
        return v;
    }
    let op = if matches!(ty, HirType::Ptr(_)) {
        CastOp::Bitcast
    } else {
        CastOp::PtrToInt
    };
    let r = value(f, ty.clone());
    out.push(HirInstruction::Cast {
        result: r,
        ty,
        op,
        operand: v,
    });
    r
}
fn pointer(
    f: &mut HirFunction,
    out: &mut Vec<HirInstruction>,
    base: HirId,
    offset: u64,
    ty: HirType,
) -> HirId {
    let address = if offset == 0 {
        base
    } else {
        let off = constant(f, offset);
        let r = value(f, raw());
        out.push(HirInstruction::GetElementPtr {
            result: r,
            ty: HirType::U8,
            ptr: base,
            indices: vec![off],
        });
        r
    };
    cast(f, out, address, ty)
}

/// Split at the current instruction; phi predecessor identities follow the tail.
fn diamond(f: &mut HirFunction, from: HirId, condition: HirId) -> (HirId, HirId, HirId) {
    let yes = f.create_block();
    let no = f.create_block();
    let tail = f.create_block();
    let term = std::mem::replace(
        &mut f.blocks.get_mut(&from).unwrap().terminator,
        HirTerminator::CondBranch {
            condition,
            true_target: yes,
            false_target: no,
        },
    );
    for succ in term.targets() {
        for phi in &mut f.blocks.get_mut(&succ).unwrap().phis {
            for (_, pred) in &mut phi.incoming {
                if *pred == from {
                    *pred = tail;
                }
            }
        }
    }
    f.blocks.get_mut(&tail).unwrap().terminator = term;
    for b in [yes, no] {
        f.blocks.get_mut(&b).unwrap().terminator = HirTerminator::Branch { target: tail };
    }
    (yes, no, tail)
}

fn read_region(insts: &[HirInstruction], start: usize, root: HirId, p: &Plan) -> usize {
    let mut end = start;
    let mut reads = 0;
    for (i, inst) in insts.iter().enumerate().skip(start).take(48) {
        match inst {
            HirInstruction::Load {
                ptr,
                volatile: false,
                ..
            } => {
                if let Some(a) = p.addresses.get(ptr) {
                    if a.root != root {
                        break;
                    }
                    reads += 1;
                    end = i + 1;
                }
            }
            HirInstruction::Binary { .. }
            | HirInstruction::Unary { .. }
            | HirInstruction::Cast { .. }
            | HirInstruction::GetElementPtr { .. }
            | HirInstruction::Select { .. }
            | HirInstruction::ExtractValue { .. }
            | HirInstruction::InsertValue { .. } => {
                if !inst
                    .result_id()
                    .is_some_and(|r| p.addresses.contains_key(&r))
                    && inst.any_operand(|v| p.addresses.contains_key(&v))
                {
                    break;
                }
            }
            _ => break,
        }
    }
    if reads >= 2 { end } else { start }
}

struct ReadObject<'a> {
    root: HirId,
    flag: HirId,
    backing: HirId,
    fields: &'a HashMap<(HirId, u64), HirId>,
}

/// Both arms retain instruction order; only values used outside need join phis.
fn guarded_reads(
    f: &mut HirFunction,
    from: HirId,
    site: Site,
    insts: &[HirInstruction],
    object: ReadObject<'_>,
    p: &Plan,
    uses: &HashMap<HirId, usize>,
) -> HirId {
    let (bid, start) = site;
    let ReadObject {
        root,
        flag,
        backing,
        fields,
    } = object;
    let (hot, cold, tail) = diamond(f, from, flag);
    let mut hot_map = indexmap::IndexMap::new();
    let mut cold_map = indexmap::IndexMap::new();
    let mut hot_insts = Vec::new();
    let mut cold_insts = Vec::new();
    let mut internal = HashMap::<HirId, usize>::new();
    for (i, inst) in insts.iter().enumerate() {
        if p.remove.contains(&(bid, start + i)) {
            continue;
        }
        inst.for_each_operand(|v| *internal.entry(v).or_default() += 1);
        let result = inst.result_id().unwrap();
        let ty = f.values[&result].ty.clone();
        if let HirInstruction::Load { ptr, align, .. } = inst
            && let Some(a) = p.addresses.get(ptr)
        {
            debug_assert_eq!(a.root, root);
            let scalar = cast(f, &mut hot_insts, fields[&(root, a.offset)], ty.clone());
            hot_map.insert(result, scalar);
            let ptr = pointer(
                f,
                &mut cold_insts,
                backing,
                a.offset,
                HirType::Ptr(Box::new(ty.clone())),
            );
            let loaded = value(f, ty.clone());
            cold_insts.push(HirInstruction::Load {
                result: loaded,
                ptr,
                ty,
                align: *align,
                volatile: false,
            });
            cold_map.insert(result, loaded);
        } else {
            for (map, out) in [
                (&mut hot_map, &mut hot_insts),
                (&mut cold_map, &mut cold_insts),
            ] {
                let mut copy = inst.clone();
                copy.replace_uses(map);
                let renamed = value(f, ty.clone());
                *copy.result_id_mut().unwrap() = renamed;
                map.insert(result, renamed);
                out.push(copy);
            }
        }
    }
    for (result, hot_value) in hot_map {
        if uses.get(&result).copied().unwrap_or_default()
            > internal.get(&result).copied().unwrap_or_default()
        {
            f.blocks.get_mut(&tail).unwrap().phis.push(HirPhi {
                result,
                ty: f.values[&result].ty.clone(),
                incoming: vec![(hot_value, hot), (cold_map[&result], cold)],
            });
        }
    }
    f.blocks.get_mut(&hot).unwrap().instructions = hot_insts;
    f.blocks.get_mut(&cold).unwrap().instructions = cold_insts;
    tail
}

fn apply(f: &mut HirFunction, p: Plan) {
    let yes = f.create_value(
        HirType::Bool,
        HirValueKind::Constant(HirConstant::Bool(true)),
    );
    let no = f.create_value(
        HirType::Bool,
        HirValueKind::Constant(HirConstant::Bool(false)),
    );
    let null = f.create_value(raw(), HirValueKind::Constant(HirConstant::Null(raw())));
    let mut virtuals = HashMap::new();
    let mut backing = HashMap::new();
    let mut fields = HashMap::new();
    let mut init_casts: HashMap<HirId, Vec<HirInstruction>> = HashMap::new();
    // Allocate every phi result before constructing any cyclic incoming edges.
    let ordered_roots: Vec<_> = f
        .values
        .keys()
        .copied()
        .filter(|r| p.roots.contains(r))
        .collect();
    for r in &ordered_roots {
        let alloc = p.allocs.get(r);
        virtuals.insert(
            *r,
            if alloc.is_some() {
                yes
            } else {
                value(f, HirType::Bool)
            },
        );
        backing.insert(
            *r,
            if alloc.is_some() {
                null
            } else {
                value(f, raw())
            },
        );
        for (off, ty) in &p.fields {
            let v = if alloc.is_some() {
                cast(
                    f,
                    init_casts.entry(*r).or_default(),
                    p.stores[&(*r, *off)],
                    ty.clone(),
                )
            } else {
                value(f, ty.clone())
            };
            fields.insert((*r, *off), v);
        }
    }
    // Opaque inputs never read scalar fields. Choose a constructor constant
    // when available so an invariant field stays constant through its phis.
    let defaults: BTreeMap<_, _> = p
        .fields
        .iter()
        .map(|(off, ty)| {
            let init = ordered_roots
                .iter()
                .filter_map(|r| p.stores.get(&(*r, *off)))
                .find_map(|v| match &f.values[v].kind {
                    HirValueKind::Constant(HirConstant::Null(_)) => {
                        Some(HirConstant::Null(ty.clone()))
                    }
                    HirValueKind::Constant(c) => Some(c.clone()),
                    _ => None,
                });
            let kind = init
                .map(HirValueKind::Constant)
                .unwrap_or(HirValueKind::Undef);
            (*off, f.create_value(ty.clone(), kind))
        })
        .collect();
    let blocks: Vec<_> = f.blocks.keys().copied().collect();
    for bid in &blocks {
        let old = std::mem::take(&mut f.blocks.get_mut(bid).unwrap().phis);
        let mut new = Vec::new();
        for phi in old {
            if !p.roots.contains(&phi.result) {
                new.push(phi);
                continue;
            }
            let mut flags = Vec::new();
            let mut pointers = Vec::new();
            let mut scalars: BTreeMap<u64, Vec<_>> =
                p.fields.keys().map(|off| (*off, Vec::new())).collect();
            for (v, pred) in &phi.incoming {
                if let Some(a) = p.addresses.get(v) {
                    flags.push((virtuals[&a.root], *pred));
                    pointers.push((backing[&a.root], *pred));
                    for off in p.fields.keys() {
                        scalars
                            .get_mut(off)
                            .unwrap()
                            .push((fields[&(a.root, *off)], *pred));
                    }
                } else {
                    flags.push((no, *pred));
                    let mut casts = Vec::new();
                    let ptr = cast(f, &mut casts, *v, raw());
                    f.blocks.get_mut(pred).unwrap().instructions.extend(casts);
                    pointers.push((ptr, *pred));
                    for off in p.fields.keys() {
                        scalars.get_mut(off).unwrap().push((defaults[off], *pred));
                    }
                }
            }
            new.push(HirPhi {
                result: virtuals[&phi.result],
                ty: HirType::Bool,
                incoming: flags,
            });
            new.push(HirPhi {
                result: backing[&phi.result],
                ty: raw(),
                incoming: pointers,
            });
            for (off, incoming) in scalars {
                new.push(HirPhi {
                    result: fields[&(phi.result, off)],
                    ty: p.fields[&off].clone(),
                    incoming,
                });
            }
        }
        f.blocks.get_mut(bid).unwrap().phis = new;
    }
    // ZYNTAX_DISABLE_GUARDED_READS keeps separate field guards; safe.
    let combine_reads = std::env::var_os("ZYNTAX_DISABLE_GUARDED_READS").is_none();
    let mut uses = HashMap::<HirId, usize>::new();
    if combine_reads {
        for block in f.blocks.values() {
            for phi in &block.phis {
                for (v, _) in &phi.incoming {
                    *uses.entry(*v).or_default() += 1;
                }
            }
            for inst in &block.instructions {
                inst.for_each_operand(|v| *uses.entry(v).or_default() += 1);
            }
            block
                .terminator
                .for_each_operand(|v| *uses.entry(v).or_default() += 1);
        }
        // Constructor fields and removed local reads may acquire new uses later.
        for v in fields.values().chain(p.local.values()) {
            *uses.entry(*v).or_default() += 1;
        }
        for inst in init_casts.values().flatten() {
            inst.for_each_operand(|v| *uses.entry(v).or_default() += 1);
        }
    }
    for bid in blocks {
        let insts = std::mem::take(&mut f.blocks.get_mut(&bid).unwrap().instructions);
        let mut current = bid;
        let mut i = 0;
        while i < insts.len() {
            if p.remove.contains(&(bid, i)) {
                i += 1;
                continue;
            }
            if combine_reads
                && let HirInstruction::Load { ptr, .. } = &insts[i]
                && let Some(a) = p.addresses.get(ptr)
            {
                let end = read_region(&insts, i, a.root, &p);
                if end > i {
                    current = guarded_reads(
                        f,
                        current,
                        (bid, i),
                        &insts[i..end],
                        ReadObject {
                            root: a.root,
                            flag: virtuals[&a.root],
                            backing: backing[&a.root],
                            fields: &fields,
                        },
                        &p,
                        &uses,
                    );
                    i = end;
                    continue;
                }
            }
            let site = (bid, i);
            let mut inst = insts[i].clone();
            i += 1;
            let mut out = Vec::new();
            match &mut inst {
                HirInstruction::Load {
                    result,
                    ptr,
                    ty,
                    align,
                    ..
                } if p.addresses.contains_key(ptr) => {
                    let a = p.addresses[ptr];
                    let (hot, cold, tail) = diamond(f, current, virtuals[&a.root]);
                    let scalar = cast(f, &mut out, fields[&(a.root, a.offset)], ty.clone());
                    f.blocks.get_mut(&hot).unwrap().instructions = out;
                    let mut out = Vec::new();
                    let address = pointer(
                        f,
                        &mut out,
                        backing[&a.root],
                        a.offset,
                        HirType::Ptr(Box::new(ty.clone())),
                    );
                    let loaded = value(f, ty.clone());
                    out.push(HirInstruction::Load {
                        result: loaded,
                        ptr: address,
                        ty: ty.clone(),
                        align: *align,
                        volatile: false,
                    });
                    f.blocks.get_mut(&cold).unwrap().instructions = out;
                    f.blocks.get_mut(&tail).unwrap().phis.push(HirPhi {
                        result: *result,
                        ty: ty.clone(),
                        incoming: vec![(scalar, hot), (loaded, cold)],
                    });
                    current = tail;
                    continue;
                }
                HirInstruction::Binary {
                    result,
                    op,
                    left,
                    right,
                    ty,
                } if p.addresses.contains_key(left) || p.addresses.contains_key(right) => {
                    let operand = if p.addresses.contains_key(left) {
                        *left
                    } else {
                        *right
                    };
                    let a = p.addresses[&operand];
                    let ptr = cast(f, &mut out, backing[&a.root], f.values[&operand].ty.clone());
                    let actual = value(f, f.values[&*result].ty.clone());
                    out.push(HirInstruction::Binary {
                        result: actual,
                        op: *op,
                        left: if *left == operand { ptr } else { *left },
                        right: if *right == operand { ptr } else { *right },
                        ty: ty.clone(),
                    });
                    let truth = f.create_value(
                        f.values[&*result].ty.clone(),
                        HirValueKind::Constant(HirConstant::Bool(*op == BinaryOp::Ne)),
                    );
                    inst = HirInstruction::Select {
                        result: *result,
                        condition: virtuals[&a.root],
                        true_val: truth,
                        false_val: actual,
                        ty: f.values[&*result].ty.clone(),
                    };
                }
                HirInstruction::Call { args, .. } if p.escapes.contains_key(&site) => {
                    let root = p.escapes[&site];
                    let (hot, cold, tail) = diamond(f, current, virtuals[&root]);
                    let allocated = value(f, raw());
                    let bytes = constant(f, p.allocs.values().next().unwrap().size);
                    out.push(HirInstruction::Call {
                        result: Some(allocated),
                        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                        args: vec![bytes],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    });
                    for (off, ty) in &p.fields {
                        let ptr = pointer(
                            f,
                            &mut out,
                            allocated,
                            *off,
                            HirType::Ptr(Box::new(ty.clone())),
                        );
                        out.push(HirInstruction::Store {
                            value: fields[&(root, *off)],
                            ptr,
                            align: 1,
                            volatile: false,
                        });
                    }
                    f.blocks.get_mut(&hot).unwrap().instructions = out;
                    let actual = value(f, raw());
                    f.blocks.get_mut(&tail).unwrap().phis.push(HirPhi {
                        result: actual,
                        ty: raw(),
                        incoming: vec![(allocated, hot), (backing[&root], cold)],
                    });
                    current = tail;
                    out = Vec::new();
                    for arg in args {
                        if p.addresses.contains_key(arg) {
                            *arg = cast(f, &mut out, actual, f.values[&*arg].ty.clone());
                        }
                    }
                }
                _ => {}
            }
            out.push(inst);
            f.blocks.get_mut(&current).unwrap().instructions.extend(out);
        }
        // All initializing stores precede transport out of their home block.
        for (r, a) in &p.allocs {
            if a.block == bid {
                f.blocks
                    .get_mut(&current)
                    .unwrap()
                    .instructions
                    .extend(init_casts.remove(r).unwrap_or_default());
            }
        }
    }
    crate::cse::apply_substitutions_public(f, &p.local);
    f.rebuild_cfg_edges();
}
