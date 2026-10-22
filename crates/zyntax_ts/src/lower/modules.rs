use super::*;

pub(super) fn declaration<'a, 's>(
    statement: &'s js::Statement<'a>,
) -> Option<&'s js::Declaration<'a>> {
    match statement {
        js::Statement::ExportDeclaration(export) => Some(&export.declaration),
        other => other.as_declaration(),
    }
}

pub(super) fn function<'a, 's>(statement: &'s js::Statement<'a>) -> Option<&'s js::Function<'a>> {
    match declaration(statement)? {
        js::Declaration::FunctionDeclaration(f) => Some(f),
        _ => None,
    }
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    fn export_name(&self, name: &js::ModuleExportName<'ast>) -> Result<InternedString> {
        if matches!(name, js::ModuleExportName::StringLiteral(s) if s.lone_surrogates) {
            return self.unsupported("module export names with unpaired surrogates", name.span());
        }
        Ok(self.name(name.name().as_str()))
    }

    pub(super) fn imports(
        &mut self,
        program: &js::Program<'ast>,
        semantic: &oxc_semantic::Semantic<'_>,
        dependencies: &[ModuleDependency<'_>],
    ) -> Result<()> {
        let dependencies: FxHashMap<_, _> = dependencies
            .iter()
            .map(|d| (intern(d.specifier), d.program))
            .collect();
        let mut loaded = FxHashSet::default();
        let mut runtime_imports: FxHashMap<InternedString, usize> = FxHashMap::default();
        for statement in &program.body {
            let js::Statement::ImportDeclaration(import) = statement else {
                continue;
            };
            if import.with_clause.is_some()
                || import.phase.is_some()
                || import.source.lone_surrogates
            {
                return self.unsupported(
                    "import attributes, phases or unpaired surrogates",
                    import.span,
                );
            }
            let source = self.name(import.source.value.as_str());
            let Some(dependency) = dependencies.get(&source) else {
                return self.error(
                    ErrorKind::Type,
                    format!("cannot resolve module `{}`", import.source.value),
                    import.source.span,
                );
            };
            let interface = dependency.checker_interface.as_deref();
            let module = dependency
                .type_registry
                .current_module()
                .or_else(|| interface.map(|i| i.module))
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Type,
                        "dependency has no module identity",
                        import.source.span,
                    )
                })?;
            if let Some(interface) = interface
                && loaded.insert(module)
            {
                self.registry.merge_from(&dependency.type_registry);
                self.views.extend(
                    interface
                        .views
                        .iter()
                        .map(|(name, view)| (*name, Arc::clone(view))),
                );
            }
            let type_only = import.import_kind == js::ImportOrExportKind::Type;
            let mut items = vec![];
            if let Some(specifiers) = &import.specifiers {
                let Some(interface) = interface else {
                    return self.error(
                        ErrorKind::Type,
                        "dependency has no exported checker interface",
                        import.source.span,
                    );
                };
                for specifier in specifiers {
                    let js::ImportDeclarationSpecifier::ImportSpecifier(specifier) = specifier
                    else {
                        return self.unsupported("default or namespace imports", import.span);
                    };
                    let name = self.export_name(&specifier.imported)?;
                    let local = self.name(specifier.local.name.as_str());
                    let type_only =
                        type_only || specifier.import_kind == js::ImportOrExportKind::Type;
                    let exported_type = interface.types.get(&name);
                    if let Some(ty) = exported_type {
                        self.aliases.insert(local, (ty.ty.clone(), ty.view));
                    }
                    if let Some(signature) = interface.functions.get(&name) {
                        if type_only {
                            if exported_type.is_none() {
                                return self.unsupported(
                                    "type-only function values and typeof import types",
                                    specifier.span,
                                );
                            }
                            continue;
                        }
                        if signature.ret.is_none()
                            || signature.array_result.is_some_and(|r| {
                                r.parameter.is_some_and(|p| p >= signature.params.len())
                            })
                        {
                            return self.error(
                                ErrorKind::Type,
                                "invalid dependency function interface",
                                specifier.span,
                            );
                        }
                        if signature
                            .params
                            .iter()
                            .any(|p| p.ownership == ParamOwnership::Owned)
                        {
                            return self.unsupported(
                                "importing an ownership-consuming function",
                                specifier.span,
                            );
                        }
                        self.signatures.insert(local, Arc::clone(signature));
                        let used = specifier.local.symbol_id.get().is_some_and(|symbol| {
                            semantic
                                .scoping()
                                .get_resolved_references(symbol)
                                .any(|r| r.is_value())
                        });
                        if used {
                            let item = TypedImportItem::Named {
                                name: signature.native,
                                alias: None,
                            };
                            if !items.contains(&item) {
                                items.push(item);
                            }
                        }
                    } else if exported_type.is_none() {
                        return self.error(
                            ErrorKind::Type,
                            format!("module does not export `{}`", specifier.imported.name()),
                            specifier.span,
                        );
                    }
                }
            }
            let side_effect = !type_only && import.specifiers.is_none();
            if !items.is_empty() || side_effect {
                if let Some(index) = runtime_imports.get(&module) {
                    let TypedDeclaration::Import(previous) = &mut self.declarations[*index].node
                    else {
                        unreachable!()
                    };
                    for item in items {
                        if !previous.items.contains(&item) {
                            previous.items.push(item);
                        }
                    }
                } else {
                    runtime_imports.insert(module, self.declarations.len());
                    self.declarations.push(node(
                        TypedDeclaration::Import(TypedImport {
                            language: Some(intern("typescript")),
                            module_path: vec![module],
                            items,
                            span: span(import.span),
                        }),
                        unit(),
                        import.span,
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn export_interface(
        &mut self,
        program: &js::Program<'ast>,
    ) -> Result<Option<Arc<CheckerInterface>>> {
        if !program.body.iter().any(|statement| {
            matches!(
                statement,
                js::Statement::ImportDeclaration(_)
                    | js::Statement::ExportDeclaration(_)
                    | js::Statement::ExportNamedDeclaration(_)
            )
        }) {
            return Ok(None);
        }
        let mut interface = CheckerInterface {
            module: self.module.name(),
            ..Default::default()
        };
        for statement in &program.body {
            match statement {
                js::Statement::ExportDeclaration(export) => {
                    let name = match &export.declaration {
                        js::Declaration::FunctionDeclaration(f) => {
                            f.id.as_ref().unwrap().name.as_str()
                        }
                        js::Declaration::TSTypeAliasDeclaration(a) => a.id.name.as_str(),
                        js::Declaration::TSInterfaceDeclaration(i) => i.id.name.as_str(),
                        _ => return self.unsupported("this exported declaration", export.span),
                    };
                    self.publish(
                        &mut interface,
                        self.name(name),
                        self.name(name),
                        false,
                        export.span,
                    )?;
                }
                js::Statement::ExportNamedDeclaration(export) => {
                    for specifier in &export.specifiers {
                        self.publish(
                            &mut interface,
                            self.export_name(&specifier.local)?,
                            self.export_name(&specifier.exported)?,
                            export.export_kind == js::ImportOrExportKind::Type
                                || specifier.export_kind == js::ImportOrExportKind::Type,
                            specifier.span,
                        )?;
                    }
                }
                _ => {}
            }
        }
        let views = interface.types.values().filter_map(|t| t.view).chain(
            interface
                .functions
                .values()
                .flat_map(|f| f.params.iter().filter_map(|p| p.view).chain(f.ret_view)),
        );
        for name in views {
            if let Some(view) = self.views.get(&name) {
                interface.views.insert(name, Arc::clone(view));
            }
        }
        let public: FxHashSet<_> = interface.functions.values().map(|f| f.native).collect();
        for declaration in &mut self.declarations {
            if let TypedDeclaration::Function(f) = &mut declaration.node {
                if public.contains(&f.name) {
                    f.visibility = Visibility::Public;
                }
            }
        }
        Ok(Some(Arc::new(interface)))
    }

    fn publish(
        &self,
        interface: &mut CheckerInterface,
        local: InternedString,
        exported: InternedString,
        type_only: bool,
        at: oxc_span::Span,
    ) -> Result<()> {
        let mut found = false;
        if let Some((ty, view)) = self.aliases.get(&local) {
            interface.types.insert(
                exported,
                CheckerType {
                    ty: ty.clone(),
                    view: *view,
                },
            );
            found = true;
        }
        if !type_only && let Some(signature) = self.signatures.get(&local) {
            interface.functions.insert(exported, Arc::clone(signature));
            found = true;
        }
        if found {
            Ok(())
        } else {
            self.unsupported("export of this binding kind", at)
        }
    }
}
