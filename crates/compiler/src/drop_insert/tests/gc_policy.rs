use super::*;

fn gc_facts() -> ModuleFacts {
    ModuleFacts::default().with_drop_policy(DropPolicy::CollectorOwned)
}

fn calls(f: &HirFunction, callee: &HirCallable) -> usize {
    f.blocks.values().flat_map(|b| &b.instructions).filter(|i| {
        matches!(i, HirInstruction::Call { callee: c, .. } if match (c, callee) {
            (HirCallable::Symbol(a), HirCallable::Symbol(b)) => a == b,
            (HirCallable::Function(a), HirCallable::Function(b)) => a == b,
            (HirCallable::Intrinsic(Intrinsic::Free), HirCallable::Intrinsic(Intrinsic::Free)) => true,
            _ => false,
        })
    }).count()
}

#[test]
fn pool_storage_skips_analysis_but_explicit_frees_remain() {
    let (mut f, ptr) = build_alloc_store_load_return();
    let count = f.blocks[&f.entry_block].instructions.len();
    let stats = run_function(&mut f, &gc_facts());
    assert_eq!(stats.mallocs_scanned, 0);
    assert_eq!(stats.frees_inserted, 0);
    assert_eq!(f.blocks[&f.entry_block].instructions.len(), count);
    let entry = f.entry_block;
    insert_free_after(&mut f, entry, count - 1, ptr, Release::Intrinsic);
    run_function(&mut f, &gc_facts());
    assert_eq!(calls(&f, &HirCallable::Intrinsic(Intrinsic::Free)), 1);
}

#[test]
fn collector_owned_loop_phis_need_no_releases_or_copies() {
    let (mut f, _) = build_accumulator_loop(false);
    let before: usize = f.blocks.values().map(|b| b.instructions.len()).sum();
    assert_eq!(release_owned_phis(&mut f, &gc_facts()), 0);
    assert_eq!(
        f.blocks
            .values()
            .map(|b| b.instructions.len())
            .sum::<usize>(),
        before
    );
    assert_eq!(release_owned_phis(&mut f, &ModuleFacts::default()), 2);
}

#[test]
fn strings_are_collected_without_adopting_them_into_boxes() {
    let (mut module, caller, _) = build_boxing_through_wrapper();
    let facts = facts_of(&module).with_drop_policy(DropPolicy::CollectorOwned);
    let stats = run_module_with(&mut module, &facts);
    assert_eq!(stats.frees_inserted, 0);
    let f = &module.functions[&caller];
    assert_eq!(calls(f, &HirCallable::Symbol(STRING_FREE.into())), 0);
    assert_eq!(calls(f, &HirCallable::Symbol(STRING_INTO_BOX.into())), 0);
}

#[test]
fn boxes_still_invoke_their_dropper() {
    let (mut f, _) = build_box_into_struct_and_read();
    let stats = run_function(&mut f, &gc_facts());
    assert_eq!(stats.frees_inserted, 1);
    assert_eq!(calls(&f, &HirCallable::Symbol(BOX_FREE.into())), 1);
}

#[test]
fn foreign_loop_phis_still_release_their_handles() {
    let (mut f, _) = build_accumulator_loop(false);
    for block in f.blocks.values_mut() {
        for inst in &mut block.instructions {
            if let HirInstruction::Call { callee, .. } = inst {
                if matches!(callee, HirCallable::Intrinsic(Intrinsic::Malloc)) {
                    *callee = HirCallable::Symbol("$Foreign$construct0_word".into());
                }
            }
        }
    }
    assert_eq!(release_owned_phis(&mut f, &gc_facts()), 2);
    assert_eq!(calls(&f, &HirCallable::Symbol(FOREIGN_WORD_FREE.into())), 2);
}

#[test]
fn type_glue_and_list_cleanup_are_retained() {
    let name = InternedString::new_global("OwnedPayload");
    let glue = HirId::new();
    let mut facts = gc_facts();
    facts.glue.insert(name, glue);
    let (mut f, ptr) = build_alloc_store_load_return();
    f.values.get_mut(&ptr).unwrap().ty = HirType::Ptr(std::sync::Arc::new(HirType::Struct(
        crate::hir::HirStructType {
            name: Some(name),
            fields: vec![HirType::I64].into(),
            packed: false,
        },
    )));
    assert_eq!(run_function(&mut f, &facts).frees_inserted, 1);
    assert_eq!(calls(&f, &HirCallable::Function(glue)), 1);

    let (mut f, ptr) = build_alloc_store_load_return();
    f.values.get_mut(&ptr).unwrap().ty =
        HirType::Ptr(std::sync::Arc::new(crate::ssa::list_header_type()));
    assert_eq!(run_function(&mut f, &gc_facts()).frees_inserted, 1);
    assert_eq!(calls(&f, &HirCallable::Intrinsic(Intrinsic::Free)), 2);
}

#[test]
fn runtime_policy_does_not_erase_recorded_ownership() {
    let ctor = HirId::new();
    let caller = HirId::new();
    let mut module = HirModule::new(InternedString::new_global("gc_facts"));
    module.functions.insert(ctor, build_constructor());
    module.functions.insert(caller, build_caller(ctor));
    let facts = facts_of(&module).with_drop_policy(DropPolicy::CollectorOwned);
    assert!(facts.returns_owned.contains(&ctor));
    record_facts(&mut module, &facts);
    run_module_with(&mut module, &facts);
    assert!(!caller_frees(&module, caller));
    let restored = facts_of(&module);
    assert!(restored.returns_owned.contains(&ctor));
    assert_eq!(restored.drop_policy, DropPolicy::EarlyRelease);
    run_module_with(&mut module, &restored);
    assert!(caller_frees(&module, caller));
}
