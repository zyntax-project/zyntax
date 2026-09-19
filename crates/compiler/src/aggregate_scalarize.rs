//! An aggregate value that only ever lives in registers becomes its fields.
//!
//! A tuple or struct built by `insertvalue`, read by `extractvalue` and
//! carried by phis is a value the IR never needs the bytes of. Each backend
//! nevertheless gives it storage: Cranelift a stack slot per definition,
//! and the promotion of an escaping alloca a heap allocation per
//! construction. Replacing every such value by one SSA value per field
//! removes the storage, the copies between slots and the phi of an
//! aggregate, which is what keeps a loop-carried tuple out of an OSR frame
//! and out of the LLVM tier.
//!
//! The pass works on webs: the values connected by `insertvalue` (result
//! to aggregate operand), by aggregate phis and by aggregate selects. A
//! web is scalarized when every use of its members is one the pass can
//! rewrite:
//!
//! * `extractvalue` reads a field: the read becomes the field's value.
//! * `insertvalue`, a phi or a select keeps the value inside the web.
//! * A call argument, a return, a stored value or a value inserted into
//!   another aggregate needs the aggregate whole: it is rebuilt there from
//!   its fields, by `insertvalue` into an undefined value, and consumed at
//!   once, so nothing ever aliases a slot that outlives its use.
//!
//! A use as an address (a load or store through the value, a GEP, a cast)
//! is not a value use, and a web with one is left alone. The roots of a
//! web are what starts an aggregate: an undefined value, an alloca or a
//! heap allocation typed as the struct (whose fields start undefined), or
//! a parameter, a call result, a load or a field read of a wider aggregate
//! (whose fields are read once where it is defined).
//!
//! Turned off with `ZYNTAX_DISABLE_AGGREGATE_SCALARIZE=1`;
//! `ZYNTAX_TRACE_SCALARIZE=1` prints each web found.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use crate::hir::{
    HirBlock, HirCallable, HirFunction, HirId, HirInstruction, HirModule, HirPhi, HirTerminator,
    HirType, HirValue, HirValueKind, Intrinsic,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AggregateScalarizeStats {
    /// Webs rewritten.
    pub webs: usize,
    /// Aggregate values that became fields.
    pub values: usize,
    /// Aggregates rebuilt at a use that needs them whole.
    pub rematerialized: usize,
}

pub fn run_module(module: &mut HirModule) -> AggregateScalarizeStats {
    if std::env::var_os("ZYNTAX_DISABLE_AGGREGATE_SCALARIZE").is_some() {
        return AggregateScalarizeStats::default();
    }
    let mut stats = AggregateScalarizeStats::default();
    for func in module.functions_to_optimize() {
        if func.is_external || func.signature.is_async {
            continue;
        }
        let s = run_function(func);
        stats.webs += s.webs;
        stats.values += s.values;
        stats.rematerialized += s.rematerialized;
    }
    stats
}

pub fn run_function(func: &mut HirFunction) -> AggregateScalarizeStats {
    let mut stats = AggregateScalarizeStats::default();
    // Scalarizing one web can expose another (a nested aggregate's field
    // reads become reads of the inner value), so webs are found afresh
    // after each rewrite.
    let mut skipped: HashSet<HirId> = HashSet::new();
    for _ in 0..64 {
        let Some(web) = find_web(func, &skipped) else {
            break;
        };
        if std::env::var_os("ZYNTAX_TRACE_SCALARIZE").is_some() {
            eprintln!(
                "[scalarize] {}: web of {} {:?}",
                func.name.resolve_global().unwrap_or_default(),
                web.members.len(),
                web.members
                    .iter()
                    .map(|m| (format!("{m:?}"), describe(web.defs[m])))
                    .collect::<Vec<_>>()
            );
        }
        match plan(func, &web) {
            Some(p) => {
                let (values, rebuilt) = apply(func, &web, p);
                stats.webs += 1;
                stats.values += values;
                stats.rematerialized += rebuilt;
            }
            None => skipped.extend(web.members.iter().copied()),
        }
    }
    stats
}

fn describe(def: Def) -> &'static str {
    match def {
        Def::Insert { .. } => "insert",
        Def::Phi { .. } => "phi",
        Def::Select { .. } => "select",
        Def::Fresh { alloc: Some(_) } => "alloc",
        Def::Fresh { alloc: None } => "undef",
        Def::Opaque { after: Some(_) } => "opaque",
        Def::Opaque { after: None } => "opaque-entry",
    }
}

/// How a member of a web is defined.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Def {
    /// `insertvalue aggregate, value, [field]`.
    Insert {
        aggregate: HirId,
        value: HirId,
        field: usize,
        block: HirId,
    },
    Phi {
        block: HirId,
    },
    Select {
        block: HirId,
        condition: HirId,
        true_val: HirId,
        false_val: HirId,
    },
    /// Starts undefined: an `Undef` value, or an alloca or heap
    /// allocation typed as the struct, which `alloc` names.
    Fresh {
        alloc: Option<HirId>,
    },
    /// Defined outside the web's own operations; its fields are read
    /// where it is defined: after the instruction at this position, or
    /// at the entry for a parameter or a constant.
    Opaque {
        after: Option<(HirId, usize)>,
    },
}

struct Web {
    ty: HirType,
    fields: Vec<HirType>,
    /// Every member, in discovery order.
    members: Vec<HirId>,
    defs: HashMap<HirId, Def>,
}

/// The value's struct type, when it is one the pass takes apart. A
/// growable list's header is not: a callee edits it in place at the
/// caller's address, so it is a value with an identity, not fields.
fn struct_of(func: &HirFunction, id: HirId) -> Option<(HirType, Vec<HirType>)> {
    let ty = &func.values.get(&id)?.ty;
    match ty {
        HirType::Struct(st)
            if !st.fields.is_empty() && !crate::abi::is_growable_list_header(st) =>
        {
            Some((ty.clone(), st.fields.clone()))
        }
        _ => None,
    }
}

/// Where and how each value of the function is defined.
fn definitions(func: &HirFunction) -> HashMap<HirId, Def> {
    let mut defs = HashMap::new();
    for (block_id, block) in &func.blocks {
        for phi in &block.phis {
            defs.insert(phi.result, Def::Phi { block: *block_id });
        }
        for (i, inst) in block.instructions.iter().enumerate() {
            let Some(result) = inst.result_id() else {
                continue;
            };
            let def = match inst {
                HirInstruction::InsertValue {
                    aggregate,
                    value,
                    indices,
                    ..
                } if indices.len() == 1 => Def::Insert {
                    aggregate: *aggregate,
                    value: *value,
                    field: indices[0] as usize,
                    block: *block_id,
                },
                HirInstruction::Select {
                    condition,
                    true_val,
                    false_val,
                    ..
                } => Def::Select {
                    block: *block_id,
                    condition: *condition,
                    true_val: *true_val,
                    false_val: *false_val,
                },
                HirInstruction::Alloca { count: None, .. } => Def::Fresh {
                    alloc: Some(result),
                },
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                    ..
                } => Def::Fresh {
                    alloc: Some(result),
                },
                _ => Def::Opaque {
                    after: Some((*block_id, i)),
                },
            };
            defs.insert(result, def);
        }
    }
    for value in func.values.values() {
        if matches!(value.kind, HirValueKind::Undef) {
            defs.insert(value.id, Def::Fresh { alloc: None });
        }
    }
    defs
}

/// A web worth scalarizing that `skipped` does not reach: one with a
/// field read of a built member or a phi in it, so that taking it apart
/// removes something.
fn find_web(func: &HirFunction, skipped: &HashSet<HirId>) -> Option<Web> {
    let defs = definitions(func);
    let mut seen: HashSet<HirId> = HashSet::new();
    // Seeds in a stable order: the phis and instructions of each block.
    let mut seeds: Vec<HirId> = Vec::new();
    for block in func.blocks.values() {
        seeds.extend(block.phis.iter().map(|p| p.result));
        seeds.extend(block.instructions.iter().filter_map(|i| i.result_id()));
    }
    for seed in seeds {
        if seen.contains(&seed) || skipped.contains(&seed) {
            continue;
        }
        let Some((ty, fields)) = struct_of(func, seed) else {
            continue;
        };
        let web = collect_web(func, &defs, seed, ty, fields);
        seen.extend(web.members.iter().copied());
        if web.members.iter().any(|m| skipped.contains(m)) {
            continue;
        }
        if worth_it(func, &web) {
            return Some(web);
        }
    }
    None
}

/// Every value connected to `seed` through insertvalue operands and
/// results, phi incomings and results, and select arms and results.
fn collect_web(
    func: &HirFunction,
    defs: &HashMap<HirId, Def>,
    seed: HirId,
    ty: HirType,
    fields: Vec<HirType>,
) -> Web {
    let mut members: Vec<HirId> = vec![seed];
    let mut set: HashSet<HirId> = HashSet::from([seed]);
    loop {
        let mut grew = false;
        let mut add = |id: HirId, members: &mut Vec<HirId>, set: &mut HashSet<HirId>| {
            if set.insert(id) {
                members.push(id);
                true
            } else {
                false
            }
        };
        // Backwards, to what each member is built from.
        let current: Vec<HirId> = members.clone();
        for m in current {
            let operands: Vec<HirId> = match defs.get(&m).copied() {
                Some(Def::Insert { aggregate, .. }) => vec![aggregate],
                Some(Def::Phi { block }) => func
                    .blocks
                    .get(&block)
                    .and_then(|b| b.phis.iter().find(|p| p.result == m))
                    .map(|p| p.incoming.iter().map(|(v, _)| *v).collect())
                    .unwrap_or_default(),
                Some(Def::Select {
                    true_val,
                    false_val,
                    ..
                }) => vec![true_val, false_val],
                _ => Vec::new(),
            };
            for o in operands {
                grew |= add(o, &mut members, &mut set);
            }
        }
        // Forwards, to what is built from a member.
        for block in func.blocks.values() {
            for phi in &block.phis {
                if phi.incoming.iter().any(|(v, _)| set.contains(v)) {
                    grew |= add(phi.result, &mut members, &mut set);
                }
            }
            for inst in &block.instructions {
                let result = match inst {
                    HirInstruction::InsertValue {
                        result, aggregate, ..
                    } if set.contains(aggregate) => *result,
                    HirInstruction::Select {
                        result,
                        true_val,
                        false_val,
                        ..
                    } if set.contains(true_val) || set.contains(false_val) => *result,
                    _ => continue,
                };
                grew |= add(result, &mut members, &mut set);
            }
        }
        if !grew {
            break;
        }
    }
    let web_defs = members
        .iter()
        .map(|m| {
            (
                *m,
                defs.get(m).copied().unwrap_or(Def::Opaque { after: None }),
            )
        })
        .collect();
    Web {
        ty,
        fields,
        members,
        defs: web_defs,
    }
}

/// Whether scalarizing removes something: a phi of the aggregate, or a
/// field read of a member built by insertion.
fn worth_it(func: &HirFunction, web: &Web) -> bool {
    if web
        .members
        .iter()
        .any(|m| matches!(web.defs[m], Def::Phi { .. }))
    {
        return true;
    }
    let built: HashSet<HirId> = web
        .members
        .iter()
        .copied()
        .filter(|m| matches!(web.defs[m], Def::Insert { .. } | Def::Fresh { .. }))
        .collect();
    if built.is_empty() {
        return false;
    }
    func.blocks.values().any(|b| {
        b.instructions.iter().any(|inst| {
            matches!(inst, HirInstruction::ExtractValue { aggregate, .. } if built.contains(aggregate))
        })
    })
}

/// A use of a member outside the web's own operations.
#[derive(Clone, Copy)]
enum Escape {
    /// Operand of the instruction at this position.
    Inst(HirId, usize),
    /// Operand of this block's terminator.
    Term(HirId),
}

struct Plan {
    /// Field reads to rewrite, by position.
    extracts: Vec<(HirId, usize)>,
    /// Uses needing the value whole: the member and where.
    escapes: Vec<(HirId, Escape)>,
}

/// Every use of every member classified; `None` when one is not a value
/// use the pass can rewrite.
fn plan(func: &HirFunction, web: &Web) -> Option<Plan> {
    let members: HashSet<HirId> = web.members.iter().copied().collect();
    // Members must agree on their type, or a field list means different
    // things to different members.
    for m in &web.members {
        if func.values.get(m).map(|v| &v.ty) != Some(&web.ty) {
            return None;
        }
        if let Def::Insert { field, .. } = web.defs[m]
            && field >= web.fields.len()
        {
            return None;
        }
    }
    // A member with no definition in sight must be a parameter, a
    // constant or a global; anything else is a value some earlier
    // rewrite left dangling, which this pass will not build on.
    for m in &web.members {
        if let Def::Opaque { after: None } = web.defs[m] {
            let kind = func.values.get(m).map(|v| &v.kind);
            if !matches!(
                kind,
                Some(
                    HirValueKind::Parameter(_)
                        | HirValueKind::Constant(_)
                        | HirValueKind::Global(_)
                )
            ) {
                if std::env::var_os("ZYNTAX_TRACE_SCALARIZE").is_some() {
                    eprintln!("[scalarize] {m:?} has no definition ({kind:?}); web skipped");
                }
                return None;
            }
        }
    }
    let mut extracts = Vec::new();
    let mut escapes = Vec::new();
    for (block_id, block) in &func.blocks {
        for phi in &block.phis {
            // A phi reading a member is a member itself.
            if !members.contains(&phi.result)
                && phi.incoming.iter().any(|(v, _)| members.contains(v))
            {
                return None;
            }
        }
        for (i, inst) in block.instructions.iter().enumerate() {
            let uses_member = |id: HirId| members.contains(&id);
            match inst {
                HirInstruction::ExtractValue {
                    aggregate, indices, ..
                } if uses_member(*aggregate) => {
                    if indices.is_empty() || indices[0] as usize >= web.fields.len() {
                        return None;
                    }
                    extracts.push((*block_id, i));
                }
                HirInstruction::InsertValue {
                    result,
                    aggregate,
                    value,
                    indices,
                    ..
                } => {
                    // The web's own link, unless the insertion is deeper
                    // than one field.
                    if uses_member(*aggregate) && (indices.len() != 1 || !members.contains(result))
                    {
                        return None;
                    }
                    if uses_member(*value) {
                        escapes.push((*value, Escape::Inst(*block_id, i)));
                    }
                }
                HirInstruction::Select {
                    result,
                    condition,
                    true_val,
                    false_val,
                    ..
                } => {
                    if uses_member(*condition) {
                        return None;
                    }
                    if (uses_member(*true_val) || uses_member(*false_val))
                        && !members.contains(result)
                    {
                        return None;
                    }
                }
                HirInstruction::Call { callee, args, .. } => {
                    if let HirCallable::Indirect(v) = callee
                        && uses_member(*v)
                    {
                        return None;
                    }
                    // Freeing an allocation the web starts from goes with
                    // the allocation.
                    let frees_root = matches!(callee, HirCallable::Intrinsic(Intrinsic::Free))
                        && args.len() == 1
                        && matches!(web.defs.get(&args[0]), Some(Def::Fresh { alloc: Some(_) }));
                    if frees_root {
                        continue;
                    }
                    let mut here = HashSet::new();
                    for a in args {
                        if uses_member(*a) && here.insert(*a) {
                            escapes.push((*a, Escape::Inst(*block_id, i)));
                        }
                    }
                }
                HirInstruction::IndirectCall { func_ptr, args, .. } => {
                    if uses_member(*func_ptr) {
                        return None;
                    }
                    let mut here = HashSet::new();
                    for a in args {
                        if uses_member(*a) && here.insert(*a) {
                            escapes.push((*a, Escape::Inst(*block_id, i)));
                        }
                    }
                }
                HirInstruction::Store { value, ptr, .. } => {
                    if uses_member(*ptr) {
                        return None;
                    }
                    if uses_member(*value) {
                        escapes.push((*value, Escape::Inst(*block_id, i)));
                    }
                }
                HirInstruction::AsyncSaveSlot { frame, value, .. } => {
                    if uses_member(*frame) {
                        return None;
                    }
                    if uses_member(*value) {
                        escapes.push((*value, Escape::Inst(*block_id, i)));
                    }
                }
                other => {
                    // Anything else reading a member treats it as an
                    // address, or in a way this pass does not follow.
                    let mut bad = false;
                    other.for_each_operand(|id| bad |= members.contains(&id));
                    if bad {
                        return None;
                    }
                }
            }
        }
        match &block.terminator {
            HirTerminator::Return { values } => {
                let mut here = HashSet::new();
                for v in values {
                    if members.contains(v) && here.insert(*v) {
                        escapes.push((*v, Escape::Term(*block_id)));
                    }
                }
            }
            HirTerminator::Invoke { args, .. } => {
                let mut here = HashSet::new();
                for a in args {
                    if members.contains(a) && here.insert(*a) {
                        escapes.push((*a, Escape::Term(*block_id)));
                    }
                }
            }
            HirTerminator::CondBranch { condition, .. } if members.contains(condition) => {
                return None;
            }
            HirTerminator::Switch { value, .. } | HirTerminator::PatternMatch { value, .. }
                if members.contains(value) =>
            {
                return None;
            }
            _ => {}
        }
    }
    Some(Plan { extracts, escapes })
}

fn new_value(func: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty,
            kind,
            uses: HashSet::new(),
            span: None,
        },
    );
    id
}

/// Rewrite the function; returns (members scalarized, aggregates rebuilt).
fn apply(func: &mut HirFunction, web: &Web, plan: Plan) -> (usize, usize) {
    let n = web.fields.len();
    // The field values of each member.
    let mut fields: HashMap<HirId, Vec<HirId>> = HashMap::new();
    // New instructions, each placed before the original instruction at
    // its position; several at one position keep their order.
    let mut inserts: Vec<(HirId, usize, HirInstruction)> = Vec::new();
    let mut delete: HashMap<HirId, HashSet<usize>> = HashMap::new();
    let mut deleted_phis: HashSet<HirId> = HashSet::new();
    let mut deleted_values: Vec<HirId> = Vec::new();

    // Roots, phis and selects get their field ids first; insertions
    // follow from their operands.
    for m in &web.members {
        match web.defs[m] {
            Def::Phi { .. } | Def::Select { .. } => {
                let ids = (0..n)
                    .map(|i| new_value(func, web.fields[i].clone(), HirValueKind::Instruction))
                    .collect();
                fields.insert(*m, ids);
            }
            Def::Fresh { .. } => {
                let ids = (0..n)
                    .map(|i| new_value(func, web.fields[i].clone(), HirValueKind::Undef))
                    .collect();
                fields.insert(*m, ids);
            }
            Def::Opaque { after } => {
                let ids: Vec<HirId> = (0..n)
                    .map(|i| new_value(func, web.fields[i].clone(), HirValueKind::Instruction))
                    .collect();
                let (block, at) = match after {
                    Some((block, i)) => (block, i + 1),
                    None => (func.entry_block, 0),
                };
                for (i, id) in ids.iter().enumerate() {
                    inserts.push((
                        block,
                        at,
                        HirInstruction::ExtractValue {
                            result: *id,
                            ty: web.fields[i].clone(),
                            aggregate: *m,
                            indices: vec![i as u32],
                        },
                    ));
                }
                fields.insert(*m, ids);
            }
            Def::Insert { .. } => {}
        }
    }
    let mut pending: Vec<HirId> = web
        .members
        .iter()
        .copied()
        .filter(|m| matches!(web.defs[m], Def::Insert { .. }))
        .collect();
    while !pending.is_empty() {
        let before = pending.len();
        pending.retain(|m| {
            let Def::Insert {
                aggregate,
                value,
                field,
                ..
            } = web.defs[m]
            else {
                return false;
            };
            let Some(base) = fields.get(&aggregate) else {
                return true;
            };
            let mut ids = base.clone();
            ids[field] = value;
            fields.insert(*m, ids);
            false
        });
        if pending.len() == before {
            // Every aggregate operand was pulled into the web, so this
            // does not happen; leave the function as it is if it does.
            return (0, 0);
        }
    }

    // Field reads become the fields. A field value that is itself a
    // rewritten read follows the chain to what replaces it.
    let mut replacements: IndexMap<HirId, HirId> = IndexMap::new();
    let mut deeper: Vec<(HirId, usize, HirInstruction)> = Vec::new();
    for (block, at) in &plan.extracts {
        let Some(HirInstruction::ExtractValue {
            result,
            ty,
            aggregate,
            indices,
        }) = func.blocks.get(block).and_then(|b| b.instructions.get(*at))
        else {
            continue;
        };
        let field = fields[aggregate][indices[0] as usize];
        if indices.len() == 1 {
            replacements.insert(*result, field);
            delete.entry(*block).or_default().insert(*at);
            deleted_values.push(*result);
        } else {
            deeper.push((
                *block,
                *at,
                HirInstruction::ExtractValue {
                    result: *result,
                    ty: ty.clone(),
                    aggregate: field,
                    indices: indices[1..].to_vec(),
                },
            ));
        }
    }
    let canon = |mut id: HirId| {
        for _ in 0..replacements.len() {
            match replacements.get(&id) {
                Some(&next) => id = next,
                None => break,
            }
        }
        id
    };
    for ids in fields.values_mut() {
        for id in ids.iter_mut() {
            *id = canon(*id);
        }
    }
    for (block, at, inst) in deeper {
        if let Some(slot) = func
            .blocks
            .get_mut(&block)
            .and_then(|b| b.instructions.get_mut(at))
        {
            *slot = inst;
        }
    }

    // Field phis and selects, and the removal of what they replace.
    let mut new_phis: HashMap<HirId, Vec<HirPhi>> = HashMap::new();
    for m in &web.members {
        match web.defs[m] {
            Def::Phi { block } => {
                let Some(phi) = func
                    .blocks
                    .get(&block)
                    .and_then(|b| b.phis.iter().find(|p| p.result == *m))
                    .cloned()
                else {
                    continue;
                };
                for i in 0..n {
                    let incoming = phi
                        .incoming
                        .iter()
                        .map(|(v, b)| (fields[v][i], *b))
                        .collect();
                    new_phis.entry(block).or_default().push(HirPhi {
                        result: fields[m][i],
                        ty: web.fields[i].clone(),
                        incoming,
                    });
                }
                deleted_phis.insert(*m);
                deleted_values.push(*m);
            }
            Def::Select {
                block,
                condition,
                true_val,
                false_val,
            } => {
                let at = position_of(func, block, *m);
                for i in 0..n {
                    inserts.push((
                        block,
                        at,
                        HirInstruction::Select {
                            result: fields[m][i],
                            ty: web.fields[i].clone(),
                            condition,
                            true_val: fields[&true_val][i],
                            false_val: fields[&false_val][i],
                        },
                    ));
                }
                delete.entry(block).or_default().insert(at);
                deleted_values.push(*m);
            }
            Def::Insert { block, .. } => {
                let at = position_of(func, block, *m);
                delete.entry(block).or_default().insert(at);
                deleted_values.push(*m);
            }
            Def::Fresh { alloc: Some(_) } => {
                for (block_id, b) in &func.blocks {
                    for (i, inst) in b.instructions.iter().enumerate() {
                        let is_alloc = inst.result_id() == Some(*m);
                        let is_free = matches!(inst, HirInstruction::Call {
                            callee: HirCallable::Intrinsic(Intrinsic::Free),
                            args,
                            ..
                        } if args.len() == 1 && args[0] == *m);
                        if is_alloc || is_free {
                            delete.entry(*block_id).or_default().insert(i);
                        }
                    }
                }
                deleted_values.push(*m);
            }
            Def::Fresh { alloc: None } | Def::Opaque { .. } => {}
        }
    }

    // Uses that need the whole value: rebuild it there.
    let mut rebuilt = 0usize;
    let mut at_sites: Vec<(Escape, IndexMap<HirId, HirId>)> = Vec::new();
    for (member, site) in &plan.escapes {
        let mut agg = new_value(func, web.ty.clone(), HirValueKind::Undef);
        let (block, at) = match site {
            Escape::Inst(block, at) => (*block, *at),
            Escape::Term(block) => {
                let len = func
                    .blocks
                    .get(block)
                    .map(|b| b.instructions.len())
                    .unwrap_or(0);
                (*block, len)
            }
        };
        for (i, field) in fields[member].iter().enumerate() {
            let next = new_value(func, web.ty.clone(), HirValueKind::Instruction);
            inserts.push((
                block,
                at,
                HirInstruction::InsertValue {
                    result: next,
                    ty: web.ty.clone(),
                    aggregate: agg,
                    value: *field,
                    indices: vec![i as u32],
                },
            ));
            agg = next;
        }
        let mut one = IndexMap::new();
        one.insert(*member, agg);
        at_sites.push((*site, one));
        rebuilt += 1;
    }

    // The renames, then the phis, then the instruction lists.
    for block in func.blocks.values_mut() {
        for inst in &mut block.instructions {
            inst.replace_uses(&replacements);
        }
        block.terminator.replace_uses(&replacements);
        for phi in &mut block.phis {
            for (v, _) in &mut phi.incoming {
                if let Some(&r) = replacements.get(v) {
                    *v = r;
                }
            }
        }
    }
    for (site, one) in at_sites {
        match site {
            Escape::Inst(block, at) => {
                if let Some(inst) = func
                    .blocks
                    .get_mut(&block)
                    .and_then(|b| b.instructions.get_mut(at))
                {
                    inst.replace_uses(&one);
                }
            }
            Escape::Term(block) => {
                if let Some(b) = func.blocks.get_mut(&block) {
                    b.terminator.replace_uses(&one);
                }
            }
        }
    }
    for block in func.blocks.values_mut() {
        block.phis.retain(|p| !deleted_phis.contains(&p.result));
    }
    for (block_id, phis) in new_phis {
        if let Some(block) = func.blocks.get_mut(&block_id) {
            block.phis.extend(phis);
        }
    }
    let mut by_block: HashMap<HirId, Vec<(usize, HirInstruction)>> = HashMap::new();
    for (block, at, inst) in inserts {
        by_block.entry(block).or_default().push((at, inst));
    }
    let blocks: Vec<HirId> = func.blocks.keys().copied().collect();
    for block_id in blocks {
        let dels = delete.remove(&block_id).unwrap_or_default();
        let mut adds = by_block.remove(&block_id).unwrap_or_default();
        if dels.is_empty() && adds.is_empty() {
            continue;
        }
        if let Some(block) = func.blocks.get_mut(&block_id) {
            rebuild_block(block, &dels, &mut adds);
        }
    }
    for id in &deleted_values {
        func.values.shift_remove(id);
    }
    (web.members.len(), rebuilt)
}

/// Position of the instruction defining `id` in `block`.
fn position_of(func: &HirFunction, block: HirId, id: HirId) -> usize {
    func.blocks
        .get(&block)
        .and_then(|b| {
            b.instructions
                .iter()
                .position(|i| i.result_id() == Some(id))
        })
        .unwrap_or(0)
}

/// The block's instructions with `dels` removed and `adds` placed, each
/// before the original instruction at its position.
fn rebuild_block(
    block: &mut HirBlock,
    dels: &HashSet<usize>,
    adds: &mut Vec<(usize, HirInstruction)>,
) {
    adds.sort_by_key(|(at, _)| *at);
    let old = std::mem::take(&mut block.instructions);
    let mut out = Vec::with_capacity(old.len() + adds.len());
    let mut adds = adds.drain(..).peekable();
    for (i, inst) in old.into_iter().enumerate() {
        while adds.peek().is_some_and(|(at, _)| *at <= i) {
            out.push(adds.next().unwrap().1);
        }
        if !dels.contains(&i) {
            out.push(inst);
        }
    }
    out.extend(adds.map(|(_, inst)| inst));
    block.instructions = out;
}
