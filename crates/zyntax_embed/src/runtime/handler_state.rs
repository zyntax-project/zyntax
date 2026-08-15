//! Handler state: synthesize the state struct, constructor, and implicit
//! `self` parameter for every stateful handler.
//!
//! Runs on the parsed program before the type registry is snapshotted.

pub(super) fn synthesize_handler_state(program: &mut zyntax_typed_ast::TypedProgram) {
    use zyntax_typed_ast::source::Span;
    use zyntax_typed_ast::type_registry::{
        FieldDef, Mutability, NullabilityKind, PrimitiveType, TypeDefinition, TypeId, TypeKind,
        TypeMetadata, TypeRegistry, Visibility,
    };
    use zyntax_typed_ast::typed_ast::{
        TypedBlock, TypedDeclaration, TypedField, TypedFieldInit, TypedFunction, TypedParameter,
        TypedStatement, TypedStructLiteral,
    };
    use zyntax_typed_ast::{InternedString, Type, TypedExpression, TypedNode};

    // A handler op is resumable iff it declares a `Resume<_>` parameter; the
    // continuation-lifting backend owns those, so Phase 3 leaves them alone.
    fn is_resume_param(ty: &Type, reg: &TypeRegistry) -> bool {
        matches!(ty, Type::Named { id, .. } if reg
            .get_type_by_id(*id)
            .map(|d| d.name.resolve_global().as_deref() == Some("Resume"))
            .unwrap_or(false))
    }

    struct StateInfo {
        handler: InternedString,
        state_name: InternedString,
        state_id: TypeId,
        fields: Vec<TypedField>,
        resumable: Vec<bool>,
    }

    // Every handler of an effect must take its operations the same
    // way, because a perform site passes its arguments to whichever
    // handler is in scope at run time. So state-ness is a property of
    // the EFFECT: once one handler carries state, the effect's ops take
    // a leading state argument and the handlers without state ignore
    // theirs.
    let stateful_effects: std::collections::HashSet<InternedString> = program
        .declarations
        .iter()
        .filter_map(|d| match &d.node {
            TypedDeclaration::EffectHandler(h) if !h.fields.is_empty() => Some(h.effect_name),
            _ => None,
        })
        .collect();

    // Phase A: collect stateful handlers + per-op resumability.
    let mut infos: Vec<StateInfo> = program
        .declarations
        .iter()
        .filter_map(|d| match &d.node {
            TypedDeclaration::EffectHandler(h) if !h.fields.is_empty() => {
                let state_name = InternedString::new_global(&format!(
                    "{}$state",
                    h.name.resolve_global().unwrap_or_default()
                ));
                let resumable = h
                    .handlers
                    .iter()
                    .map(|imp| {
                        imp.params
                            .iter()
                            .any(|p| is_resume_param(&p.ty, &program.type_registry))
                    })
                    .collect();
                Some(StateInfo {
                    handler: h.name,
                    state_name,
                    state_id: TypeId::next(),
                    fields: h.fields.clone(),
                    resumable,
                })
            }
            _ => None,
        })
        .collect();
    if infos.is_empty() {
        return;
    }

    // Register each `H$state` as a `@reference` struct (heap-allocated, so
    // instances are pointers — exactly the `self` ABI we want).
    for info in &mut infos {
        if let Some(existing) = program.type_registry.get_type_by_name(info.state_name) {
            info.state_id = existing.id;
            continue;
        }
        let field_defs: Vec<FieldDef> = info
            .fields
            .iter()
            .map(|f| FieldDef {
                name: f.name,
                ty: f.ty.clone(),
                visibility: f.visibility,
                mutability: f.mutability,
                is_static: f.is_static,
                span: f.span,
                getter: None,
                setter: None,
                is_synthetic: false,
            })
            .collect();
        let mut metadata = TypeMetadata::default();
        metadata.is_reference = true;
        program.type_registry.register_type(TypeDefinition {
            id: info.state_id,
            module: None,
            name: info.state_name,
            kind: TypeKind::Struct {
                fields: field_defs.clone(),
                is_tuple: false,
            },
            type_params: vec![],
            constraints: vec![],
            fields: field_defs,
            methods: vec![],
            constructors: vec![],
            metadata,
            span: Span::default(),
        });
    }

    let named = |id: TypeId| Type::Named {
        id,
        type_args: vec![],
        const_args: vec![],
        variance: vec![],
        nullability: NullabilityKind::NonNull,
    };

    // Phase B: prepend `self: H$state` to every non-resumable op impl.
    // A stateless handler of a stateful effect gets the same leading
    // slot, typed as a plain machine word it never reads — it only has
    // to accept the argument the perform site passes.
    for d in &mut program.declarations {
        if let TypedDeclaration::EffectHandler(h) = &mut d.node {
            let self_ty = match infos.iter().find(|i| i.handler == h.name) {
                Some(info) => Some(named(info.state_id)),
                None if stateful_effects.contains(&h.effect_name) => {
                    Some(Type::Primitive(PrimitiveType::I64))
                }
                None => None,
            };
            let Some(self_ty) = self_ty else {
                continue;
            };
            let resumable: Vec<bool> = match infos.iter().find(|i| i.handler == h.name) {
                Some(info) => info.resumable.clone(),
                None => h
                    .handlers
                    .iter()
                    .map(|imp| {
                        imp.params
                            .iter()
                            .any(|p| is_resume_param(&p.ty, &program.type_registry))
                    })
                    .collect(),
            };
            for (j, imp) in h.handlers.iter_mut().enumerate() {
                if !resumable[j] {
                    imp.params.insert(
                        0,
                        TypedParameter {
                            name: InternedString::new_global("self"),
                            ty: self_ty.clone(),
                            mutability: Mutability::Mutable,
                            ..Default::default()
                        },
                    );
                }
            }
        }
    }

    // Phase C: synthesize `H$new(): H$state { return H$state { f: init, .. } }`.
    let span = Span::default();
    let mut ctors: Vec<TypedNode<TypedDeclaration>> = Vec::new();
    for info in &infos {
        let field_inits: Vec<TypedFieldInit> = info
            .fields
            .iter()
            .filter_map(|f| {
                f.initializer.as_ref().map(|init| TypedFieldInit {
                    name: f.name,
                    value: init.clone(),
                })
            })
            .collect();
        let literal = TypedNode::new(
            TypedExpression::Struct(TypedStructLiteral {
                name: info.state_name,
                fields: field_inits,
            }),
            named(info.state_id),
            span,
        );
        let ret = TypedNode::new(
            TypedStatement::Return(Some(Box::new(literal))),
            Type::Never,
            span,
        );
        let ctor = TypedFunction {
            name: InternedString::new_global(&format!(
                "{}$new",
                info.handler.resolve_global().unwrap_or_default()
            )),
            params: vec![],
            return_type: named(info.state_id),
            body: Some(TypedBlock {
                statements: vec![ret],
                span,
            }),
            visibility: Visibility::Public,
            is_pure: false,
            ..Default::default()
        };
        ctors.push(TypedNode::new(
            TypedDeclaration::Function(ctor),
            Type::Primitive(PrimitiveType::Unit),
            span,
        ));
    }
    program.declarations.extend(ctors);
}
