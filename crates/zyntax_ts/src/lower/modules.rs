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
    if let js::Statement::ExportDefaultDeclaration(export) = statement {
        return match &export.declaration {
            js::ExportDefaultDeclarationKind::FunctionDeclaration(f) => Some(f),
            _ => None,
        };
    }
    match declaration(statement)? {
        js::Declaration::FunctionDeclaration(f) => Some(f),
        _ => None,
    }
}

pub(super) fn function_name<'a>(function: &js::Function<'a>) -> &'a str {
    // Anonymous default declarations need a key that no source identifier can use.
    function
        .id
        .as_ref()
        .map_or("\0default", |id| id.name.as_str())
}

pub(crate) fn dependency_source<'a, 's>(
    statement: &'s js::Statement<'a>,
) -> Option<&'s js::StringLiteral<'a>> {
    match statement {
        js::Statement::ImportDeclaration(i) => Some(&i.source),
        js::Statement::ExportFromDeclaration(e) => Some(&e.source),
        js::Statement::ExportAllDeclaration(e) => Some(&e.source),
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
    ) -> Result<CheckerInterface> {
        let dependencies: FxHashMap<_, _> = dependencies
            .iter()
            .map(|d| (intern(d.specifier), d.program))
            .collect();
        let mut loaded = FxHashSet::default();
        let mut runtime_imports = FxHashMap::default();
        let mut exports = CheckerInterface {
            module: self.module.name(),
            ..Default::default()
        };
        if !program
            .body
            .iter()
            .any(|statement| dependency_source(statement).is_some())
        {
            return Ok(exports);
        }
        // Explicit exports shadow star bindings regardless of declaration order.
        let mut explicit = FxHashSet::default();
        let mut export_uses = FxHashMap::default();
        let default_export = self.name("default");
        for statement in &program.body {
            match statement {
                js::Statement::ExportDeclaration(e) => {
                    let name = match &e.declaration {
                        js::Declaration::FunctionDeclaration(f) => {
                            f.id.as_ref().map(|id| id.name.as_str())
                        }
                        js::Declaration::TSTypeAliasDeclaration(t) => Some(t.id.name.as_str()),
                        js::Declaration::TSInterfaceDeclaration(i) => Some(i.id.name.as_str()),
                        _ => return self.unsupported("this exported declaration", e.span),
                    };
                    if let Some(name) = name {
                        explicit.insert(self.name(name));
                    }
                }
                js::Statement::ExportNamedDeclaration(e) => {
                    for specifier in &e.specifiers {
                        explicit.insert(self.export_name(&specifier.exported)?);
                        if let js::ModuleExportName::IdentifierReference(local) = &specifier.local {
                            export_uses.insert(
                                local.node_id.get(),
                                e.export_kind != js::ImportOrExportKind::Type
                                    && specifier.export_kind != js::ImportOrExportKind::Type,
                            );
                        }
                    }
                }
                js::Statement::ExportFromDeclaration(e) => {
                    for specifier in &e.specifiers {
                        explicit.insert(self.export_name(&specifier.exported)?);
                    }
                }
                js::Statement::ExportDefaultDeclaration(e) => {
                    explicit.insert(default_export);
                    if let js::ExportDefaultDeclarationKind::Identifier(local) = &e.declaration {
                        export_uses.insert(local.node_id.get(), true);
                    }
                }
                _ => {}
            }
        }
        for statement in &program.body {
            let Some(source) = dependency_source(statement) else {
                continue;
            };
            let attributes = match statement {
                js::Statement::ImportDeclaration(i) => i.with_clause.is_some() || i.phase.is_some(),
                js::Statement::ExportFromDeclaration(e) => e.with_clause.is_some(),
                js::Statement::ExportAllDeclaration(e) => {
                    if e.exported.is_some() {
                        return self.unsupported("namespace export objects", e.span);
                    }
                    e.with_clause.is_some()
                }
                _ => unreachable!(),
            };
            if attributes || source.lone_surrogates {
                return self.unsupported(
                    "module attributes, phases or unpaired surrogates",
                    source.span,
                );
            }
            let dependency = dependencies
                .get(&self.name(source.value.as_str()))
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Type,
                        format!("cannot resolve module `{}`", source.value),
                        source.span,
                    )
                })?;
            let interface = dependency.checker_interface.as_deref();
            let module = dependency
                .type_registry
                .current_module()
                .or_else(|| interface.map(|i| i.module))
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Type,
                        "dependency has no module identity",
                        source.span,
                    )
                })?;
            if loaded.insert(module) {
                self.registry.merge_from(&dependency.type_registry);
                if let Some(interface) = interface {
                    self.views.extend(
                        interface
                            .views
                            .iter()
                            .map(|(name, view)| (*name, Arc::clone(view))),
                    );
                }
            }
            let mut items = vec![];
            let runtime = match statement {
                js::Statement::ImportDeclaration(import) => {
                    if let Some(specifiers) = &import.specifiers {
                        let interface = self.require_interface(interface, source.span)?;
                        for specifier in specifiers {
                            let (name, identifier, type_only) = match specifier {
                                js::ImportDeclarationSpecifier::ImportSpecifier(s) => (
                                    self.export_name(&s.imported)?,
                                    &s.local,
                                    s.import_kind == js::ImportOrExportKind::Type,
                                ),
                                js::ImportDeclarationSpecifier::ImportDefaultSpecifier(s) => {
                                    (default_export, &s.local, false)
                                }
                                js::ImportDeclarationSpecifier::ImportNamespaceSpecifier(_) => {
                                    return self
                                        .unsupported("namespace import objects", import.span);
                                }
                            };
                            let local = self.name(identifier.name.as_str());
                            let mut binding =
                                self.export_binding(interface, name, specifier.span())?;
                            binding.type_only |=
                                import.import_kind == js::ImportOrExportKind::Type || type_only;
                            self.imported_exports.insert(local, binding);
                            if let Some(ty) = interface.types.get(&name) {
                                self.aliases.insert(local, (ty.ty.clone(), ty.view));
                            }
                            if let Some(signature) = interface.functions.get(&name) {
                                self.validate_signature(signature, specifier.span())?;
                                self.signatures.insert(local, Arc::clone(signature));
                            }
                            let mut used = false;
                            if let Some(symbol) = identifier.symbol_id.get() {
                                for reference in semantic
                                    .scoping()
                                    .get_resolved_references(symbol)
                                    .filter(|r| r.is_value())
                                {
                                    if let Some(value) = export_uses.get(&reference.node_id()) {
                                        used |= !binding.type_only && *value;
                                    } else if !binding.type_only {
                                        used = true;
                                    }
                                }
                            }
                            if used && let Some(signature) = interface.functions.get(&name) {
                                Self::request_function(&mut items, signature.native);
                            }
                        }
                    }
                    !items.is_empty()
                        || (import.specifiers.is_none()
                            && import.import_kind != js::ImportOrExportKind::Type)
                }
                js::Statement::ExportFromDeclaration(export) => {
                    let interface = self.require_interface(interface, source.span)?;
                    for specifier in &export.specifiers {
                        let name = self.export_name(&specifier.local)?;
                        let exported = self.export_name(&specifier.exported)?;
                        let type_only = export.export_kind == js::ImportOrExportKind::Type
                            || specifier.export_kind == js::ImportOrExportKind::Type;
                        let mut binding = self.export_binding(interface, name, specifier.span)?;
                        binding.type_only |= type_only;
                        self.forward(
                            &mut exports,
                            interface,
                            name,
                            exported,
                            binding,
                            specifier.span,
                        )?;
                        if !binding.type_only
                            && let Some(signature) = interface.functions.get(&name)
                        {
                            Self::request_function(&mut items, signature.native);
                        }
                    }
                    !items.is_empty()
                }
                js::Statement::ExportAllDeclaration(export) => {
                    let interface = self.require_interface(interface, source.span)?;
                    let type_only = export.export_kind == js::ImportOrExportKind::Type;
                    for (&name, &binding) in &interface.exports {
                        if name == default_export || explicit.contains(&name) {
                            continue;
                        }
                        let mut binding = CheckerExport {
                            type_only: binding.type_only || type_only,
                            ..binding
                        };
                        if let Some(previous) = exports.exports.get(&name) {
                            if previous.origin != binding.origin {
                                return self.error(ErrorKind::Type, format!("conflicting star export `{}`; explicitly re-export the intended binding", name.resolve_global().unwrap()), export.span);
                            }
                            binding.type_only &= previous.type_only;
                        }
                        self.forward(&mut exports, interface, name, name, binding, export.span)?;
                        if !type_only
                            && !binding.type_only
                            && let Some(signature) = interface.functions.get(&name)
                        {
                            Self::request_function(&mut items, signature.native);
                        }
                    }
                    !type_only
                }
                _ => unreachable!(),
            };
            if runtime {
                self.runtime_import(module, items, statement.span(), &mut runtime_imports);
            }
        }
        Ok(exports)
    }

    fn require_interface<'i>(
        &self,
        interface: Option<&'i CheckerInterface>,
        at: oxc_span::Span,
    ) -> Result<&'i CheckerInterface> {
        interface.ok_or_else(|| {
            Error::new(
                ErrorKind::Type,
                "dependency has no exported checker interface",
                at,
            )
        })
    }

    fn export_binding(
        &self,
        interface: &CheckerInterface,
        name: InternedString,
        at: oxc_span::Span,
    ) -> Result<CheckerExport> {
        interface.exports.get(&name).copied().ok_or_else(|| {
            Error::new(
                ErrorKind::Type,
                format!(
                    "module does not export `{}`",
                    name.resolve_global().unwrap()
                ),
                at,
            )
        })
    }

    fn validate_signature(&self, signature: &Signature, at: oxc_span::Span) -> Result<()> {
        if signature.ret.is_none()
            || signature
                .array_result
                .is_some_and(|r| r.parameter.is_some_and(|p| p >= signature.params.len()))
        {
            return self.error(ErrorKind::Type, "invalid dependency function interface", at);
        }
        if signature
            .params
            .iter()
            .any(|p| p.ownership == ParamOwnership::Owned)
        {
            return self.unsupported("importing an ownership-consuming function", at);
        }
        Ok(())
    }

    pub(super) fn require_value(&self, name: &'ast str, at: oxc_span::Span) -> Result<()> {
        if self.imported_exports.is_empty() {
            return Ok(());
        }
        let name = self.name(name);
        if self
            .imported_exports
            .get(&name)
            .is_some_and(|binding| binding.type_only)
            && !self.functions.contains_key(&name)
            && !self.scopes.iter().any(|scope| scope.contains_key(&name))
        {
            return self.error(
                ErrorKind::Type,
                "type-only binding cannot be used as a value",
                at,
            );
        }
        Ok(())
    }

    fn forward(
        &self,
        target: &mut CheckerInterface,
        source: &CheckerInterface,
        name: InternedString,
        exported: InternedString,
        binding: CheckerExport,
        at: oxc_span::Span,
    ) -> Result<()> {
        if let Some(ty) = source.types.get(&name) {
            target.types.insert(exported, ty.clone());
        }
        if let Some(signature) = source.functions.get(&name) {
            self.validate_signature(signature, at)?;
            target.functions.insert(exported, Arc::clone(signature));
        }
        target.exports.insert(exported, binding);
        Ok(())
    }

    fn request_function(items: &mut Vec<TypedImportItem>, name: InternedString) {
        let item = TypedImportItem::Named { name, alias: None };
        if !items.contains(&item) {
            items.push(item);
        }
    }

    fn runtime_import(
        &mut self,
        module: InternedString,
        items: Vec<TypedImportItem>,
        at: oxc_span::Span,
        imports: &mut FxHashMap<InternedString, usize>,
    ) {
        if let Some(index) = imports.get(&module) {
            let TypedDeclaration::Import(previous) = &mut self.declarations[*index].node else {
                unreachable!()
            };
            for item in items {
                if !previous.items.contains(&item) {
                    previous.items.push(item);
                }
            }
        } else {
            imports.insert(module, self.declarations.len());
            self.declarations.push(node(
                TypedDeclaration::Import(TypedImport {
                    language: Some(intern("typescript")),
                    module_path: vec![module],
                    items,
                    span: span(at),
                }),
                unit(),
                at,
            ));
        }
    }

    pub(super) fn export_interface(
        &mut self,
        program: &js::Program<'ast>,
        mut interface: CheckerInterface,
    ) -> Result<Option<Arc<CheckerInterface>>> {
        if !program.body.iter().any(|statement| {
            matches!(
                statement,
                js::Statement::ImportDeclaration(_)
                    | js::Statement::ExportDeclaration(_)
                    | js::Statement::ExportNamedDeclaration(_)
                    | js::Statement::ExportFromDeclaration(_)
                    | js::Statement::ExportAllDeclaration(_)
                    | js::Statement::ExportDefaultDeclaration(_)
            )
        }) {
            return Ok(None);
        }
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
                js::Statement::ExportDefaultDeclaration(export) => {
                    let local = match &export.declaration {
                        js::ExportDefaultDeclarationKind::FunctionDeclaration(f) => {
                            self.name(function_name(f))
                        }
                        js::ExportDefaultDeclarationKind::Identifier(id) => {
                            self.name(id.name.as_str())
                        }
                        _ => return self.unsupported("this default export value", export.span),
                    };
                    if !self.signatures.contains_key(&local) && !self.aliases.contains_key(&local) {
                        return self
                            .unsupported("default export of a non-function value", export.span);
                    }
                    self.publish(
                        &mut interface,
                        local,
                        self.name("default"),
                        false,
                        export.span,
                    )?;
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
        let public: FxHashSet<_> = interface
            .functions
            .iter()
            .filter(|(name, _)| !interface.exports[name].type_only)
            .map(|(_, f)| f.native)
            .collect();
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
        if let Some(signature) = self.signatures.get(&local) {
            interface.functions.insert(exported, Arc::clone(signature));
            found = true;
        }
        if found {
            let mut binding = self
                .functions
                .get(&local)
                .map(|_| CheckerExport {
                    origin: self.signatures[&local].native,
                    type_only: false,
                })
                .or_else(|| self.imported_exports.get(&local).copied())
                .unwrap_or_else(|| CheckerExport {
                    origin: self.signatures.get(&local).map_or_else(
                        || self.module.function_name(&local.resolve_global().unwrap()),
                        |f| f.native,
                    ),
                    type_only: !self.signatures.contains_key(&local),
                });
            binding.type_only |= type_only;
            interface.exports.insert(exported, binding);
            Ok(())
        } else {
            self.unsupported("export of this binding kind", at)
        }
    }
}
