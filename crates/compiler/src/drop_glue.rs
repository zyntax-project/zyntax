//! Release derived from a type's own fields.
//!
//! [`crate::drop_insert`] pairs one allocation with one free. That is
//! the whole story for a box holding a number, and none of it for a
//! type holding another of its kind: releasing a tree's root has to
//! release the two nodes its fields point at, and those their own, for
//! as deep as the tree goes. Nothing in the IR says so, because the
//! fact lives in the type rather than in any one instruction.
//!
//! This derives it. For every `@reference` type with at least one
//! reference-typed field, a function is synthesised that releases each
//! such field before releasing the object itself. Recursion is by name,
//! so a type holding its own kind produces a function that calls
//! itself, and the null check on each field is what ends it at a leaf.
//!
//! What a program would otherwise write by hand:
//!
//! ```text
//! def release(self) {
//!     if self.left != null { self.left.release(); self.right.release() }
//!     free(self)
//! }
//! ```
//!
//! ## Which fields are released
//!
//! Only a field whose type is a named reference type, which is
//! `Ptr(Struct { name: Some(_), .. })` in HIR. A raw `Ptr<i8>` is a
//! pointer the program manages itself and is left alone, because
//! nothing says the struct is the one that owns it.
//!
//! ## Where the layout comes from
//!
//! The same natural-alignment rule the struct-literal lowering uses:
//! each field aligned to its own size, the running offset carried
//! forward. The two have to agree or a release reads a field from the
//! wrong offset, so both compute it the same way from
//! [`crate::ssa::hir_ty_size`].
//!
//! ## What this does not decide
//!
//! Whether a given allocation is released at all. That stays with
//! `drop_insert`, which knows the lifetimes; this only says what
//! releasing one *means* once that pass has decided to.

use indexmap::IndexMap;

use crate::hir::{
    BinaryOp, CastOp, HirBlock, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirId,
    HirInstruction, HirModule, HirParam, HirStructType, HirTerminator, HirType, HirValue,
    HirValueKind, Intrinsic, ParamAttributes, ParamOwnership,
};
use zyntax_typed_ast::InternedString;

/// Per-run statistics, for telemetry and test assertions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GlueStats {
    /// Named reference types found anywhere in the module.
    pub types_seen: usize,
    /// Functions synthesised, one per type that owns another.
    pub glue_emitted: usize,
}

/// The name a type's release function is given.
///
/// Double-underscored so it cannot collide with a method a program
/// writes: `impl Node { def drop(self) }` mangles to `Node$drop`, which
/// is a different name from this one and stays the program's own.
pub fn glue_name(type_name: InternedString) -> String {
    format!("__drop__{}", type_name.resolve_global().unwrap_or_default())
}

/// The type a function is the release for, if it is one of these.
///
/// Read back off the signature rather than off the name, so a program
/// that happens to define `__drop__Node` itself cannot be mistaken for
/// the compiler's own.
pub fn glue_target(func: &HirFunction) -> Option<InternedString> {
    let p = func.signature.params.first()?;
    if p.ownership != ParamOwnership::Owned || func.signature.params.len() != 1 {
        return None;
    }
    let name = owned_field(&p.ty)?;
    (func.name.resolve_global()? == glue_name(name)).then_some(name)
}

/// Field layout for a struct, by the rule the literal lowering uses.
///
/// Returns each field's byte offset. Kept next to the release that
/// reads through it so the two cannot drift apart.
fn field_offsets(fields: &[HirType]) -> Vec<u64> {
    crate::object_header::field_layout(fields).0
}

/// Whether a field's type is one this owns rather than merely points at.
fn owned_field(ty: &HirType) -> Option<InternedString> {
    match ty {
        HirType::Ptr(inner) => match &**inner {
            HirType::Struct(s) => s.name,
            _ => None,
        },
        _ => None,
    }
}

/// Every named reference type the module mentions, with its fields.
///
/// A type is mentioned in more than one place and not always in full:
/// stopping a self-referential expansion leaves the inner mention with
/// no fields at all. The widest mention is the true one, so the map
/// keeps whichever occurrence carries fields.
fn reference_types(module: &HirModule) -> IndexMap<InternedString, Vec<HirType>> {
    let mut found: IndexMap<InternedString, Vec<HirType>> = IndexMap::new();
    let mut note = |ty: &HirType| {
        if let HirType::Ptr(inner) = ty {
            if let HirType::Struct(HirStructType {
                name: Some(n),
                fields,
                ..
            }) = &**inner
            {
                if !fields.is_empty() {
                    found.entry(*n).or_insert_with(|| fields.clone());
                }
            }
        }
    };
    for func in module.functions.values() {
        for p in &func.signature.params {
            note(&p.ty);
        }
        for r in &func.signature.returns {
            note(r);
        }
        for v in func.values.values() {
            note(&v.ty);
        }
    }
    found
}

/// Whether releasing a type from its own fields is turned on.
///
/// Opt-in, because turning it on changes what an existing program
/// means. A program that releases a type by hand releases it twice once
/// this does too, and the second release is a fault at runtime with
/// nothing said at compile time. The way to state that a call ends the
/// caller's claim is `own`, which a method cannot say about its
/// receiver: `own self` does not parse. Until it does, and until the
/// borrow check can report a release through a borrow, a program that
/// manages its own memory has no way to say so and this stays off.
///
/// One switch for the whole feature, not for the synthesis alone. The
/// transfer and liveness rules in `drop_insert` are the rest of it: a
/// constructor that hands back storage makes its caller responsible for
/// releasing it, and a caller that releases what it already released by
/// hand faults just the same whether or not any glue was emitted.
pub fn enabled() -> bool {
    std::env::var("ZYNTAX_DROP_GLUE").as_deref() == Ok("1")
}

/// Emit a release function for every type that owns another.
///
/// Whether to call this at all is [`enabled`]'s question, asked by the
/// caller: a pass that reads the environment cannot be tested without
/// setting it, and a test that sets it changes what every other test
/// running beside it sees.
pub fn synthesise(module: &mut HirModule) -> GlueStats {
    let types = reference_types(module);
    let mut stats = GlueStats {
        types_seen: types.len(),
        glue_emitted: 0,
    };

    // Which types get one, decided before any is built: a field's
    // release is a call to the field type's glue when it has one and a
    // plain free when it does not, and that cannot be known while the
    // set is still being discovered.
    let owning: Vec<InternedString> = types
        .iter()
        .filter(|(_, fields)| fields.iter().any(|f| owned_field(f).is_some()))
        .map(|(n, _)| *n)
        .collect();
    if owning.is_empty() {
        return stats;
    }

    // Reserved first so a type holding its own kind, or two types
    // holding each other, can name a function that is not built yet.
    let ids: IndexMap<InternedString, HirId> = owning.iter().map(|n| (*n, HirId::new())).collect();

    for name in &owning {
        let fields = &types[name];
        let func = build_glue(*name, fields, &ids);
        module.functions.insert(ids[name], func);
        stats.glue_emitted += 1;
    }
    stats
}

/// Add a constant to a function and hand back its id.
fn constant(func: &mut HirFunction, ty: HirType, k: HirConstant) -> HirId {
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty,
            kind: HirValueKind::Constant(k),
            uses: Default::default(),
            span: None,
        },
    );
    id
}

/// Add an instruction result to a function and hand back its id.
fn result(func: &mut HirFunction, ty: HirType) -> HirId {
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty,
            kind: HirValueKind::Instruction,
            uses: Default::default(),
            span: None,
        },
    );
    id
}

fn empty_block(id: HirId) -> HirBlock {
    HirBlock {
        id,
        label: None,
        phis: Vec::new(),
        instructions: Vec::new(),
        terminator: HirTerminator::Unreachable,
        dominance_frontier: Default::default(),
        predecessors: Vec::new(),
        successors: Vec::new(),
    }
}

/// The release function for one type.
///
/// One pair of blocks per owned field: read the field, and call its
/// release when it is not null. The chain ends in a block that frees
/// the object itself, which every path reaches exactly once.
fn build_glue(
    name: InternedString,
    fields: &[HirType],
    ids: &IndexMap<InternedString, HirId>,
) -> HirFunction {
    let self_ty = HirType::Ptr(Box::new(HirType::Struct(HirStructType {
        name: Some(name),
        fields: fields.to_vec(),
        packed: false,
    })));
    let fn_id = ids[&name];
    let self_id = HirId::new();
    let entry = HirId::new();

    let mut func = HirFunction {
        id: fn_id,
        name: InternedString::new_global(&glue_name(name)),
        signature: HirFunctionSignature {
            params: vec![HirParam {
                id: self_id,
                name: InternedString::new_global("self"),
                ty: self_ty.clone(),
                attributes: ParamAttributes::default(),
                // The release is the end of the claim, which is what
                // `Owned` says. A call to one is therefore not an
                // escape needing a second release after it.
                ownership: ParamOwnership::Owned,
            }],
            returns: vec![],
            type_params: Vec::new(),
            const_params: Vec::new(),
            lifetime_params: Vec::new(),
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: Vec::new(),
            is_pure: false,
        },
        entry_block: entry,
        blocks: Default::default(),
        locals: Default::default(),
        values: Default::default(),
        previous_version: None,
        is_external: false,
        calling_convention: crate::hir::CallingConvention::C,
        attributes: Default::default(),
        link_name: None,
    };
    func.values.insert(
        self_id,
        HirValue {
            id: self_id,
            ty: self_ty,
            kind: HirValueKind::Parameter(0),
            uses: Default::default(),
            span: None,
        },
    );

    let offsets = field_offsets(fields);
    let owned: Vec<(usize, InternedString)> = fields
        .iter()
        .enumerate()
        .filter_map(|(i, f)| owned_field(f).map(|n| (i, n)))
        .collect();

    // The last block: the object's own storage, released after every
    // field it owns has been.
    let tail = HirId::new();

    let mut blocks: Vec<HirBlock> = Vec::new();
    let mut current = entry;
    for (nth, (idx, field_type)) in owned.iter().enumerate() {
        let mut read = empty_block(current);
        let field_ty = fields[*idx].clone();

        let off = constant(
            &mut func,
            HirType::I64,
            HirConstant::I64(offsets[*idx] as i64),
        );
        let gep = result(&mut func, HirType::Ptr(Box::new(HirType::U8)));
        read.instructions.push(HirInstruction::GetElementPtr {
            result: gep,
            ty: HirType::U8,
            ptr: self_id,
            indices: vec![off],
        });
        let slot = result(&mut func, HirType::Ptr(Box::new(field_ty.clone())));
        read.instructions.push(HirInstruction::Cast {
            op: CastOp::Bitcast,
            result: slot,
            ty: HirType::Ptr(Box::new(field_ty.clone())),
            operand: gep,
        });
        let child = result(&mut func, field_ty.clone());
        read.instructions.push(HirInstruction::Load {
            result: child,
            ty: field_ty.clone(),
            ptr: slot,
            align: 8,
            volatile: false,
        });
        let null = constant(&mut func, field_ty, HirConstant::I64(0));
        let present = result(&mut func, HirType::Bool);
        read.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Ne,
            result: present,
            ty: HirType::Bool,
            left: child,
            right: null,
        });

        // Where a field that is absent goes: the next field's read, or
        // the object's own release once there are none left.
        let next = if nth + 1 == owned.len() {
            tail
        } else {
            HirId::new()
        };
        let call = HirId::new();
        read.terminator = HirTerminator::CondBranch {
            condition: present,
            true_target: call,
            false_target: next,
        };
        read.successors = vec![call, next];
        blocks.push(read);

        let mut release = empty_block(call);
        // A field type that owns nothing has no release of its own, so
        // freeing its storage is the whole of releasing it.
        match ids.get(field_type) {
            Some(glue) => release.instructions.push(HirInstruction::Call {
                result: None,
                callee: HirCallable::Function(*glue),
                args: vec![child],
                type_args: Vec::new(),
                const_args: Vec::new(),
                is_tail: false,
            }),
            None => release.instructions.push(HirInstruction::Call {
                result: None,
                callee: HirCallable::Intrinsic(Intrinsic::Free),
                args: vec![child],
                type_args: Vec::new(),
                const_args: Vec::new(),
                is_tail: false,
            }),
        }
        release.terminator = HirTerminator::Branch { target: next };
        release.successors = vec![next];
        release.predecessors = vec![current];
        blocks.push(release);

        current = next;
    }

    let mut last = empty_block(tail);
    last.instructions.push(HirInstruction::Call {
        result: None,
        callee: HirCallable::Intrinsic(Intrinsic::Free),
        args: vec![self_id],
        type_args: Vec::new(),
        const_args: Vec::new(),
        is_tail: false,
    });
    last.terminator = HirTerminator::Return { values: Vec::new() };
    blocks.push(last);

    // Predecessors, now that every block exists.
    let edges: Vec<(HirId, HirId)> = blocks
        .iter()
        .flat_map(|b| b.successors.iter().map(move |s| (b.id, *s)))
        .collect();
    for b in &mut blocks {
        b.predecessors = edges
            .iter()
            .filter(|(_, to)| *to == b.id)
            .map(|(from, _)| *from)
            .collect();
    }
    for b in blocks {
        func.blocks.insert(b.id, b);
    }
    func
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A signature for a function the test only uses as a place to
    /// hang a value of the type under test.
    fn void_sig() -> HirFunctionSignature {
        HirFunctionSignature {
            params: Vec::new(),
            returns: vec![],
            type_params: Vec::new(),
            const_params: Vec::new(),
            lifetime_params: Vec::new(),
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: Vec::new(),
            is_pure: false,
        }
    }

    fn node_type(name: InternedString) -> HirType {
        HirType::Ptr(Box::new(HirType::Struct(HirStructType {
            name: Some(name),
            fields: vec![
                HirType::Ptr(Box::new(HirType::Struct(HirStructType {
                    name: Some(name),
                    fields: Vec::new(),
                    packed: false,
                }))),
                HirType::Ptr(Box::new(HirType::Struct(HirStructType {
                    name: Some(name),
                    fields: Vec::new(),
                    packed: false,
                }))),
                HirType::I64,
            ],
            packed: false,
        })))
    }

    /// A module mentioning a type that holds its own kind gets one
    /// release function, and it calls itself.
    ///
    /// Calling itself is the point: a tree is released to whatever
    /// depth it has, and nothing in the IR says how deep that is.
    #[test]
    fn a_type_holding_its_own_kind_gets_a_recursive_release() {
        let name = InternedString::new_global("Node");
        let mut module = HirModule::new(InternedString::new_global("m"));
        let mut host = HirFunction::new(InternedString::new_global("host"), void_sig());
        let v = HirId::new();
        host.values.insert(
            v,
            HirValue {
                id: v,
                ty: node_type(name),
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        module.functions.insert(HirId::new(), host);

        let stats = synthesise(&mut module);
        assert_eq!(stats.glue_emitted, 1, "one type owns another, so one glue");

        let glue = module
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some("__drop__Node"))
            .expect("the release should be named after its type");

        let calls_itself = glue.blocks.values().any(|b| {
            b.instructions.iter().any(|i| {
                matches!(i, HirInstruction::Call { callee: HirCallable::Function(id), .. }
                    if *id == glue.id)
            })
        });
        assert!(
            calls_itself,
            "a type holding its own kind must release through itself, or a \
             tree is freed one level deep"
        );

        let frees_self = glue.blocks.values().any(|b| {
            b.instructions.iter().any(|i| {
                matches!(
                    i,
                    HirInstruction::Call {
                        callee: HirCallable::Intrinsic(Intrinsic::Free),
                        ..
                    }
                )
            })
        });
        assert!(frees_self, "the object's own storage must be released too");
    }

    /// A type owning nothing gets none. Freeing its storage is the
    /// whole of releasing it, which `drop_insert` already emits.
    #[test]
    fn a_type_owning_nothing_gets_no_release() {
        let name = InternedString::new_global("Point");
        let mut module = HirModule::new(InternedString::new_global("m"));
        let mut host = HirFunction::new(InternedString::new_global("host"), void_sig());
        let v = HirId::new();
        host.values.insert(
            v,
            HirValue {
                id: v,
                ty: HirType::Ptr(Box::new(HirType::Struct(HirStructType {
                    name: Some(name),
                    fields: vec![HirType::F64, HirType::F64],
                    packed: false,
                }))),
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        module.functions.insert(HirId::new(), host);

        let stats = synthesise(&mut module);
        assert_eq!(stats.types_seen, 1);
        assert_eq!(stats.glue_emitted, 0);
    }

    /// A raw pointer field is not released.
    ///
    /// `Ptr<i8>` is storage the program manages itself. Nothing says
    /// the struct is the one that owns it, and releasing it here would
    /// free memory its writer is still using.
    #[test]
    fn a_raw_pointer_field_is_left_alone() {
        let name = InternedString::new_global("Buf");
        let mut module = HirModule::new(InternedString::new_global("m"));
        let mut host = HirFunction::new(InternedString::new_global("host"), void_sig());
        let v = HirId::new();
        host.values.insert(
            v,
            HirValue {
                id: v,
                ty: HirType::Ptr(Box::new(HirType::Struct(HirStructType {
                    name: Some(name),
                    fields: vec![HirType::Ptr(Box::new(HirType::I8)), HirType::I64],
                    packed: false,
                }))),
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        module.functions.insert(HirId::new(), host);

        assert_eq!(synthesise(&mut module).glue_emitted, 0);
    }

    /// The offsets a release reads from are the ones the literal
    /// lowering wrote to.
    ///
    /// They are computed in two places, so this pins them together. A
    /// disagreement would read a child from the wrong offset and free
    /// whatever was there.
    #[test]
    fn field_offsets_follow_natural_alignment() {
        let fields = vec![
            HirType::Ptr(Box::new(HirType::I8)),
            HirType::Ptr(Box::new(HirType::I8)),
            HirType::I64,
        ];
        assert_eq!(field_offsets(&fields), vec![0, 8, 16]);
    }
}
