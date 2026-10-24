use super::*;

pub(super) struct Namespace {
    interface: Arc<CheckerInterface>,
    type_only: bool,
}

#[derive(Clone, Copy)]
enum NamespaceUse<'ast> {
    Member {
        property: Option<&'ast str>,
        at: oxc_span::Span,
    },
    Object(oxc_span::Span),
}

pub(super) struct NamespaceUses<'ast, 'scope> {
    scoping: &'scope oxc_semantic::Scoping,
    symbols: FxHashSet<oxc_semantic::SymbolId>,
    uses: FxHashMap<oxc_semantic::NodeId, NamespaceUse<'ast>>,
}

fn namespace_root<'a, 's>(
    expression: &'s js::Expression<'a>,
) -> Option<&'s js::IdentifierReference<'a>> {
    match expression {
        js::Expression::Identifier(id) => Some(id),
        js::Expression::ParenthesizedExpression(p) => namespace_root(&p.expression),
        _ => None,
    }
}

impl NamespaceUses<'_, '_> {
    fn contains(&self, id: &js::IdentifierReference<'_>) -> bool {
        id.reference_id
            .get()
            .and_then(|id| self.scoping.get_reference(id).symbol_id())
            .is_some_and(|symbol| self.symbols.contains(&symbol))
    }
}

impl<'ast> Visit<'ast> for NamespaceUses<'ast, '_> {
    fn visit_static_member_expression(&mut self, member: &js::StaticMemberExpression<'ast>) {
        if let Some(root) = namespace_root(&member.object).filter(|id| self.contains(id)) {
            self.uses.insert(
                root.node_id.get(),
                NamespaceUse::Member {
                    property: Some(member.property.name.as_str()),
                    at: member.span,
                },
            );
        }
        walk::walk_static_member_expression(self, member);
    }

    fn visit_computed_member_expression(&mut self, member: &js::ComputedMemberExpression<'ast>) {
        if let Some(root) = namespace_root(&member.object).filter(|id| self.contains(id)) {
            self.uses.insert(
                root.node_id.get(),
                NamespaceUse::Member {
                    property: Lowerer::namespace_key(&member.expression),
                    at: member.span,
                },
            );
        }
        walk::walk_computed_member_expression(self, member);
    }

    fn visit_identifier_reference(&mut self, id: &js::IdentifierReference<'ast>) {
        if self.contains(id) {
            self.uses
                .entry(id.node_id.get())
                .or_insert(NamespaceUse::Object(id.span));
        }
    }
}

impl<'ast, 'scope> NamespaceUses<'ast, 'scope> {
    pub(super) fn collect(
        program: &js::Program<'ast>,
        scoping: &'scope oxc_semantic::Scoping,
    ) -> Self {
        let mut namespace_uses = NamespaceUses {
            scoping,
            symbols: FxHashSet::default(),
            uses: FxHashMap::default(),
        };
        for statement in &program.body {
            if let js::Statement::ImportDeclaration(import) = statement
                && let Some(specifiers) = &import.specifiers
            {
                for specifier in specifiers {
                    if let js::ImportDeclarationSpecifier::ImportNamespaceSpecifier(s) = specifier
                        && let Some(symbol) = s.local.symbol_id.get()
                        && scoping
                            .get_resolved_references(symbol)
                            .any(|r| r.is_value())
                    {
                        namespace_uses.symbols.insert(symbol);
                    }
                }
            }
        }
        // Index only namespace references; the complete semantic node table stays disabled.
        if !namespace_uses.symbols.is_empty() {
            namespace_uses.visit_program(program);
        }
        namespace_uses
    }
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    pub(super) fn import_namespace(
        &mut self,
        specifier: &js::ImportNamespaceSpecifier<'ast>,
        type_only: bool,
        interface: &Arc<CheckerInterface>,
        uses: &NamespaceUses<'ast, '_>,
        items: &mut Vec<TypedImportItem>,
    ) -> Result<()> {
        let Some(symbol) = specifier.local.symbol_id.get() else {
            return Ok(());
        };
        if uses.scoping.get_resolved_reference_ids(symbol).is_empty() {
            return Ok(());
        }
        let index = self.namespaces.len();
        self.namespaces.push(Namespace {
            interface: Arc::clone(interface),
            type_only,
        });
        let references = || uses.scoping.get_resolved_references(symbol);
        for reference in references() {
            self.namespace_references.insert(reference.node_id(), index);
        }
        for reference in references().filter(|r| r.is_value()) {
            let signature = match uses.uses[&reference.node_id()] {
                NamespaceUse::Member { property, at } => {
                    self.namespace_export(&self.namespaces[index], property, at)?
                }
                NamespaceUse::Object(at) => {
                    if type_only {
                        return self.error(
                            ErrorKind::Type,
                            "type-only namespace cannot be used as a value",
                            at,
                        );
                    }
                    return self.unsupported("namespace objects as values", at);
                }
            };
            Self::request_function(items, signature.native);
        }
        Ok(())
    }

    fn namespace_key(expression: &js::Expression<'ast>) -> Option<&'ast str> {
        match expression {
            js::Expression::StringLiteral(s) if !s.lone_surrogates => Some(s.value.as_str()),
            js::Expression::ParenthesizedExpression(p) => Self::namespace_key(&p.expression),
            _ => None,
        }
    }

    fn namespace_object(&self, expression: &js::Expression<'ast>) -> Option<&Namespace> {
        match expression {
            js::Expression::Identifier(id) => self
                .namespace_references
                .get(&id.node_id.get())
                .map(|i| &self.namespaces[*i]),
            js::Expression::ParenthesizedExpression(p) => self.namespace_object(&p.expression),
            _ => None,
        }
    }

    fn namespace_property(
        &self,
        object: &js::Expression<'ast>,
        property: Option<&'ast str>,
        at: oxc_span::Span,
    ) -> Result<Option<&Arc<Signature>>> {
        let Some(namespace) = self.namespace_object(object) else {
            return Ok(None);
        };
        self.namespace_export(namespace, property, at).map(Some)
    }

    fn namespace_export<'s>(
        &self,
        namespace: &'s Namespace,
        property: Option<&'ast str>,
        at: oxc_span::Span,
    ) -> Result<&'s Arc<Signature>> {
        if namespace.type_only {
            return self.error(
                ErrorKind::Type,
                "type-only namespace cannot be used as a value",
                at,
            );
        }
        let Some(property) = property else {
            return self.unsupported("runtime namespace indexing", at);
        };
        let name = self.name(property);
        let binding = self.export_binding(&namespace.interface, name, at)?;
        if binding.type_only {
            return self.error(
                ErrorKind::Type,
                "type-only export cannot be used as a value",
                at,
            );
        }
        let signature =
            namespace.interface.functions.get(&name).ok_or_else(|| {
                Error::new(ErrorKind::Type, "namespace member is not a function", at)
            })?;
        self.validate_signature(signature, at)?;
        Ok(signature)
    }

    pub(super) fn namespace_function(
        &self,
        expression: &js::Expression<'ast>,
    ) -> Result<Option<&Arc<Signature>>> {
        match expression {
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                self.namespace_property(&m.object, Some(m.property.name.as_str()), m.span)
            }
            js::Expression::ComputedMemberExpression(m) if !m.optional => {
                self.namespace_property(&m.object, Self::namespace_key(&m.expression), m.span)
            }
            js::Expression::ParenthesizedExpression(p) => self.namespace_function(&p.expression),
            _ => Ok(None),
        }
    }

    pub(super) fn namespace_type(
        &self,
        name: &js::TSQualifiedName<'ast>,
    ) -> Result<(Type, Option<InternedString>)> {
        let js::TSTypeName::IdentifierReference(root) = &name.left else {
            return self.unsupported("nested namespace types", name.span);
        };
        let namespace = self
            .namespace_references
            .get(&root.node_id.get())
            .map(|i| &self.namespaces[*i])
            .ok_or_else(|| Error::new(ErrorKind::Type, "cannot find namespace", root.span))?;
        let exported = self.name(name.right.name.as_str());
        self.export_binding(&namespace.interface, exported, name.span)?;
        namespace
            .interface
            .types
            .get(&exported)
            .map(|ty| (ty.ty.clone(), ty.view))
            .ok_or_else(|| Error::new(ErrorKind::Type, "namespace member is not a type", name.span))
    }

    pub(super) fn direct_signature(
        &self,
        expression: &js::Expression<'ast>,
    ) -> Result<Option<&Arc<Signature>>> {
        if let js::Expression::Identifier(id) = expression {
            let name = self.name(id.name.as_str());
            if !self.scopes.iter().any(|scope| scope.contains_key(&name)) {
                self.require_value(id.name.as_str(), id.span)?;
                if !self.signatures.contains_key(&name) && library_value(id.name.as_str()) {
                    return self.unsupported(&format!("library value `{}`", id.name), id.span);
                }
                return self.signatures.get(&name).map(Some).ok_or_else(|| {
                    Error::new(
                        ErrorKind::Type,
                        format!("cannot find function `{}`", id.name),
                        id.span,
                    )
                });
            }
        }
        self.namespace_function(expression)
    }

    pub(super) fn function_value(&self, signature: &Signature, at: oxc_span::Span) -> Result<Expr> {
        if !signature.type_params.is_empty() {
            return self.unsupported("generic function values", at);
        }
        if self.arities.contains_key(&signature.native) {
            return self.unsupported(
                "function values with optional, default or rest parameters",
                at,
            );
        }
        let ret = signature
            .ret
            .as_ref()
            .ok_or_else(|| Error::new(ErrorKind::Unsupported, "unresolved function result", at))?;
        let ty = self.function_type(
            signature.params.iter().map(|p| p.ty.clone()),
            ret.clone(),
            at,
        )?;
        Ok(var(signature.native, ty, at))
    }

    pub(super) fn check_namespace_write(
        &self,
        object: &js::Expression<'ast>,
        at: oxc_span::Span,
    ) -> Result<()> {
        if self.namespace_object(object).is_some() {
            self.error(ErrorKind::Type, "cannot assign to a namespace export", at)
        } else {
            Ok(())
        }
    }
}
