//! Top-level bindings that function bodies reach live in module globals.
//!
//! The module body assigns each global where its declaration appears. When code that may call
//! into the module's functions runs before that point, reads and writes from function bodies
//! test a ready flag and raise the temporal dead zone's `ReferenceError`. A binding that a
//! function may read while it still holds `undefined` (declared without a value, or a `var`
//! read before its declaration) has an assigned flag instead, and such a read traps. Flow facts
//! are never recorded for a global: any call may reassign it.

use super::*;
use oxc_semantic::SymbolId;

pub(super) struct Global {
    /// Set by the declaration; present only when a call may run before it.
    pub ready: Option<InternedString>,
    /// Set by the first assignment; present only when a read may come before it.
    pub assigned: Option<InternedString>,
    pub source: InternedString,
}

/// Top-level symbols referenced from a function, arrow or class body.
struct Captures<'s> {
    scoping: &'s oxc_semantic::Scoping,
    candidates: &'s FxHashSet<SymbolId>,
    depth: usize,
    captured: FxHashSet<SymbolId>,
}

impl<'a> Visit<'a> for Captures<'_> {
    fn visit_function(&mut self, it: &js::Function<'a>, flags: oxc_semantic::ScopeFlags) {
        self.depth += 1;
        walk::walk_function(self, it, flags);
        self.depth -= 1;
    }
    fn visit_arrow_function_expression(&mut self, it: &js::ArrowFunctionExpression<'a>) {
        self.depth += 1;
        walk::walk_arrow_function_expression(self, it);
        self.depth -= 1;
    }
    fn visit_class(&mut self, it: &js::Class<'a>) {
        self.depth += 1;
        walk::walk_class(self, it);
        self.depth -= 1;
    }
    fn visit_identifier_reference(&mut self, id: &js::IdentifierReference<'a>) {
        if self.depth == 0 {
            return;
        }
        let Some(reference) = id.reference_id.get() else {
            return;
        };
        let reference = self.scoping.get_reference(reference);
        if let Some(symbol) = reference.symbol_id().filter(|_| reference.is_value()) {
            if self.candidates.contains(&symbol) {
                self.captured.insert(symbol);
            }
        }
    }
}

/// Whether evaluating code may call into the module's functions. Deferred bodies do not run;
/// console, Math, timing and array built-ins do not call back.
struct RunsCode<'s> {
    scoping: &'s oxc_semantic::Scoping,
    arrays: &'s FxHashSet<SymbolId>,
    found: bool,
}

impl RunsCode<'_> {
    fn unbound(&self, id: &js::IdentifierReference<'_>) -> bool {
        id.reference_id
            .get()
            .is_none_or(|r| self.scoping.get_reference(r).symbol_id().is_none())
    }
    fn array_constructor(&self, callee: &js::Expression<'_>) -> bool {
        matches!(callee, js::Expression::Identifier(id) if id.name == "Array" && self.unbound(id))
    }
    fn builtin(&self, callee: &js::Expression<'_>) -> bool {
        let js::Expression::StaticMemberExpression(m) = callee else {
            return false;
        };
        let method = m.property.name.as_str();
        match &m.object {
            js::Expression::Identifier(id) if self.unbound(id) => {
                matches!(
                    id.name.as_str(),
                    "console" | "Math" | "Date" | "performance"
                )
            }
            js::Expression::Identifier(id) => {
                matches!(method, "push" | "fill")
                    && id
                        .reference_id
                        .get()
                        .and_then(|r| self.scoping.get_reference(r).symbol_id())
                        .is_some_and(|symbol| self.arrays.contains(&symbol))
            }
            js::Expression::NewExpression(n) => {
                method == "fill" && self.array_constructor(&n.callee)
            }
            js::Expression::CallExpression(c) => {
                method == "fill" && self.array_constructor(&c.callee)
            }
            _ => false,
        }
    }
}

impl<'a> Visit<'a> for RunsCode<'_> {
    fn visit_function(&mut self, _: &js::Function<'a>, _: oxc_semantic::ScopeFlags) {}
    fn visit_arrow_function_expression(&mut self, _: &js::ArrowFunctionExpression<'a>) {}
    fn visit_class(&mut self, _: &js::Class<'a>) {}
    fn visit_call_expression(&mut self, it: &js::CallExpression<'a>) {
        self.found |= !self.builtin(&it.callee);
        walk::walk_call_expression(self, it);
    }
    fn visit_new_expression(&mut self, it: &js::NewExpression<'a>) {
        self.found |= !self.array_constructor(&it.callee);
        walk::walk_new_expression(self, it);
    }
    fn visit_tagged_template_expression(&mut self, it: &js::TaggedTemplateExpression<'a>) {
        self.found = true;
        walk::walk_tagged_template_expression(self, it);
    }
}

/// A declarator whose type is an array: annotated as one, or initialized by an array literal
/// or constructor.
fn declares_array(d: &js::VariableDeclarator<'_>) -> bool {
    if let Some(annotation) = &d.type_annotation {
        return match &annotation.type_annotation {
            js::TSType::TSArrayType(_) => true,
            js::TSType::TSTypeReference(r) => {
                matches!(&r.type_name, js::TSTypeName::IdentifierReference(n) if n.name == "Array")
            }
            _ => false,
        };
    }
    let mut init = d.init.as_ref();
    while let Some(js::Expression::ParenthesizedExpression(p)) = init {
        init = Some(&p.expression);
    }
    match init {
        Some(js::Expression::ArrayExpression(_)) => true,
        Some(js::Expression::CallExpression(c)) => matches!(
            &c.callee,
            js::Expression::StaticMemberExpression(m) if m.property.name == "fill"
                && matches!(&m.object, js::Expression::NewExpression(_) | js::Expression::CallExpression(_))
        ),
        _ => false,
    }
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    /// Opens the module scope and declares a global for each top-level binding that a function,
    /// arrow or class body reaches. Literal constants that no earlier call can observe stay
    /// substituted.
    pub(super) fn module_bindings(
        &mut self,
        program: &'program js::Program<'ast>,
        semantic: &oxc_semantic::Semantic<'_>,
    ) -> Result<()> {
        self.scopes.push(FxHashMap::default());
        let scoping = semantic.scoping();
        let mut candidates = FxHashSet::default();
        let mut arrays = FxHashSet::default();
        for item in &program.body {
            let js::Statement::VariableDeclaration(declaration) = item else {
                continue;
            };
            for d in &declaration.declarations {
                if let js::BindingPattern::BindingIdentifier(id) = &d.id
                    && let Some(symbol) = id.symbol_id.get()
                {
                    candidates.insert(symbol);
                    if declares_array(d) {
                        arrays.insert(symbol);
                    }
                }
            }
        }
        let mut captures = Captures {
            scoping,
            candidates: &candidates,
            depth: 0,
            captured: FxHashSet::default(),
        };
        if !candidates.is_empty() {
            captures.visit_program(program);
        }
        if captures.captured.is_empty() && self.enums.values().all(|e| e.literal) {
            return Ok(());
        }
        let mut runs = RunsCode {
            scoping,
            arrays: &arrays,
            found: false,
        };
        for item in &program.body {
            let js::Statement::VariableDeclaration(declaration) = item else {
                // A regular enum is an object its declaration creates; a call may read it first.
                if let js::Statement::TSEnumDeclaration(declaration) = item
                    && runs.found
                    && !declaration.r#const
                    && !declaration.declare
                {
                    self.enum_ready(declaration);
                }
                if !runs.found {
                    runs.visit_statement(item);
                }
                continue;
            };
            let kind = declaration.kind.as_str();
            for d in &declaration.declarations {
                // A declarator's own initializer runs before its binding is ready.
                if let Some(init) = &d.init
                    && !runs.found
                {
                    runs.visit_expression(init);
                }
                let js::BindingPattern::BindingIdentifier(id) = &d.id else {
                    continue;
                };
                let Some(symbol) = id.symbol_id.get() else {
                    continue;
                };
                if !captures.captured.contains(&symbol) || self.module_symbols.contains_key(&symbol)
                {
                    continue;
                }
                let name = self.name(id.name.as_str());
                if declaration.declare {
                    self.declare_ambient(id, d, kind, symbol)?;
                    continue;
                }
                if !runs.found && self.constants.contains_key(&name) {
                    continue;
                }
                self.declare_global(id, d, kind, runs.found, symbol)?;
            }
        }
        Ok(())
    }

    fn declare_global(
        &mut self,
        id: &js::BindingIdentifier<'ast>,
        d: &js::VariableDeclarator<'ast>,
        kind: &str,
        guarded: bool,
        symbol: SymbolId,
    ) -> Result<()> {
        if !matches!(kind, "let" | "const" | "var") {
            return self.unsupported("using declarations", d.span);
        }
        let (ty, view) = match (&d.type_annotation, &d.init) {
            (Some(annotation), _) => self.ty(&annotation.type_annotation)?,
            (None, Some(init)) => self.initializer_type(init).ok_or_else(|| {
                Error::new(
                    ErrorKind::Unsupported,
                    "inferring the type of a module binding used in functions is not implemented yet",
                    d.span,
                )
            })?,
            (None, None) => {
                return self.unsupported("uninitialized bindings without a type annotation", d.span);
            }
        };
        let storable = is_scalar(&ty)
            || ty == string()
            || element_type(&ty).is_some()
            || matches!(ty, Type::Named { .. });
        if !storable {
            return self.unsupported("module bindings of this type used in functions", d.span);
        }
        let source = self.name(id.name.as_str());
        let global = self.global_name("global$", id.name.as_str());
        if kind == "const"
            && d.type_annotation.is_none()
            && let Some(init) = &d.init
        {
            self.declare_literal(global, init);
        }
        let var = kind == "var";
        let ready = (guarded && !var).then(|| self.global_name("ready$", id.name.as_str()));
        let assigned = (d.init.is_none() || (guarded && var))
            .then(|| self.global_name("assigned$", id.name.as_str()));
        let flags = ready
            .into_iter()
            .chain(assigned)
            .map(|flag| (flag, boolean()));
        for (name, ty) in std::iter::once((global, ty.clone())).chain(flags) {
            self.declarations.push(node(
                TypedDeclaration::Variable(TypedVariable {
                    name,
                    ty,
                    mutability: Mutability::Mutable,
                    initializer: None,
                    visibility: Visibility::Public,
                }),
                unit(),
                d.span,
            ));
        }
        self.globals.insert(
            global,
            Global {
                ready,
                assigned,
                source,
            },
        );
        self.module_symbols.insert(symbol, global);
        self.scopes[0].insert(
            source,
            Binding {
                name: global,
                symbol: Some(symbol),
                ty,
                view,
                mutable: kind != "const",
                initialized: true,
            },
        );
        Ok(())
    }

    /// A module binding an ambient `declare` names: typed, with no native value to read.
    fn declare_ambient(
        &mut self,
        id: &js::BindingIdentifier<'ast>,
        d: &js::VariableDeclarator<'ast>,
        kind: &str,
        symbol: SymbolId,
    ) -> Result<()> {
        let Some(annotation) = &d.type_annotation else {
            return Ok(());
        };
        let (ty, view) = self.ty(&annotation.type_annotation)?;
        let global = self.global_name("ambient$", id.name.as_str());
        self.ambient.insert(global);
        self.module_symbols.insert(symbol, global);
        let source = self.name(id.name.as_str());
        self.scopes[0].insert(
            source,
            Binding {
                name: global,
                symbol: Some(symbol),
                ty,
                view,
                mutable: kind != "const",
                initialized: true,
            },
        );
        Ok(())
    }

    fn global_name(&mut self, kind: &str, source: &str) -> InternedString {
        self.name_buffer.truncate(self.prefix_len);
        self.name_buffer.push_str(kind);
        crate::modules::append_component(&mut self.name_buffer, source);
        intern(&self.name_buffer)
    }

    /// The type an initializer has before any function body is lowered: literals, operators on
    /// them, earlier module bindings, constructors and calls with declared results.
    fn initializer_type(&self, e: &js::Expression<'ast>) -> Option<(Type, Option<InternedString>)> {
        let scalar = |ty: Type| Some((ty, None));
        match e {
            js::Expression::NumericLiteral(_) => scalar(number()),
            js::Expression::BooleanLiteral(_) => scalar(boolean()),
            js::Expression::StringLiteral(s) if !s.lone_surrogates => scalar(string()),
            js::Expression::TemplateLiteral(_) => scalar(string()),
            js::Expression::ParenthesizedExpression(p) => self.initializer_type(&p.expression),
            js::Expression::UnaryExpression(u) => match u.operator.as_str() {
                "-" | "+" | "~" => scalar(number()),
                "!" => scalar(boolean()),
                "typeof" => scalar(string()),
                _ => None,
            },
            js::Expression::BinaryExpression(b) => match b.operator.as_str() {
                "-" | "*" | "/" | "%" | "**" | "|" | "&" | "^" | "<<" | ">>" | ">>>" => {
                    scalar(number())
                }
                "<" | "<=" | ">" | ">=" | "===" | "!==" | "==" | "!=" => scalar(boolean()),
                "+" => {
                    let (left, _) = self.initializer_type(&b.left)?;
                    let (right, _) = self.initializer_type(&b.right)?;
                    if left == string() || right == string() {
                        scalar(string())
                    } else if left == number() && right == number() {
                        scalar(number())
                    } else {
                        None
                    }
                }
                _ => None,
            },
            js::Expression::Identifier(i) => {
                let name = self.name(i.name.as_str());
                if let Some(binding) = self.scopes[0].get(&name) {
                    return Some((binding.ty.clone(), binding.view));
                }
                self.constants.get(&name).map(|c| (c.ty(), None))
            }
            js::Expression::NewExpression(n) => {
                let view = self.class_of_new(n)?;
                Some((self.classes[&view].ty.clone(), Some(view)))
            }
            js::Expression::CallExpression(c) => {
                if let js::Expression::StaticMemberExpression(m) = &c.callee
                    && m.property.name == "fill"
                {
                    let constructor = match &m.object {
                        js::Expression::NewExpression(n) => (&n.callee, &n.type_arguments),
                        js::Expression::CallExpression(c) => (&c.callee, &c.type_arguments),
                        _ => return None,
                    };
                    if !matches!(constructor.0, js::Expression::Identifier(id) if id.name == "Array")
                    {
                        return None;
                    }
                    let element = match constructor.1 {
                        Some(arguments) if arguments.params.len() == 1 => {
                            self.ty(&arguments.params[0]).ok()?.0
                        }
                        Some(_) => return None,
                        None => {
                            self.initializer_type(c.arguments.first()?.as_expression()?)?
                                .0
                        }
                    };
                    return scalar(array_type(element));
                }
                let js::Expression::Identifier(callee) = &c.callee else {
                    return None;
                };
                let sig = self.signatures.get(&self.name(callee.name.as_str()))?;
                if !sig.type_params.is_empty()
                    || self.scopes[0].contains_key(&self.name(callee.name.as_str()))
                {
                    return None;
                }
                Some((sig.ret.clone()?, sig.ret_view))
            }
            js::Expression::ArrayExpression(a) => {
                let mut element = None;
                for item in &a.elements {
                    let (ty, _) = self.initializer_type(item.as_expression()?)?;
                    if element.as_ref().is_some_and(|e| *e != ty) {
                        return None;
                    }
                    element = Some(ty);
                }
                let element = element?;
                (is_scalar(&element) || element == string()).then(|| (array_type(element), None))
            }
            js::Expression::TSAsExpression(a) => self.ty(&a.type_annotation).ok(),
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                let enumeration = self.enum_access(&m.object)?;
                (enumeration.kind() != enums::Kind::Mixed)
                    .then(|| (enumeration.physical(), Some(enumeration.view)))
            }
            _ => None,
        }
    }

    /// The checks for a global reached from a function body: its temporal dead zone, and for a
    /// read, its first assignment. Calls keep the checks straight-line wherever the access is.
    pub(super) fn global_guard(
        &mut self,
        name: InternedString,
        read: bool,
        at: oxc_span::Span,
    ) -> SmallVec<[Stmt; 2]> {
        let mut checks = SmallVec::new();
        if self.top_level {
            return checks;
        }
        let Some(global) = self.globals.get(&name) else {
            return checks;
        };
        let source = global.source;
        let flags = [
            ("$TS$binding_check", global.ready),
            ("$TS$assigned_check", global.assigned.filter(|_| read)),
        ];
        for (check, flag) in flags {
            if let Some(flag) = flag {
                checks.push(expression(call(
                    intern(check),
                    vec![
                        var(flag, boolean(), at),
                        node(
                            TypedExpression::Literal(TypedLiteral::String(source)),
                            string(),
                            at,
                        ),
                    ],
                    unit(),
                    at,
                )));
            }
        }
        checks
    }

    /// Records a write to a global whose reads check for a first assignment.
    pub(super) fn global_assigned(&self, name: InternedString, at: oxc_span::Span) -> Option<Stmt> {
        let flag = self.globals.get(&name)?.assigned?;
        Some(Self::set_flag(flag, at))
    }

    pub(super) fn set_flag(flag: InternedString, at: oxc_span::Span) -> Stmt {
        expression(binary(
            BinaryOp::Assign,
            var(flag, boolean(), at),
            node(
                TypedExpression::Literal(TypedLiteral::Bool(true)),
                boolean(),
                at,
            ),
            unit(),
            at,
        ))
    }

    /// The declaration of a global in the module body: assigns its value when it has one and
    /// marks the binding ready.
    pub(super) fn declare_in_module(
        &mut self,
        global: InternedString,
        source: InternedString,
        init: Option<&js::Expression<'ast>>,
        at: oxc_span::Span,
    ) -> Result<Vec<Stmt>> {
        let mut out = vec![];
        if let Some(init) = init {
            out.push(self.initialize_global(global, source, init, at)?);
            out.extend(self.global_assigned(global, at));
        } else if !self.scopes[0][&source].initialized {
            // Module-body reads before an assignment are checked like a local's.
            self.unassigned.insert(global, Unassigned::Declared);
        }
        if let Some(ready) = self.globals[&global].ready {
            out.push(Self::set_flag(ready, at));
        }
        self.scopes[0].get_mut(&source).unwrap().initialized = true;
        Ok(out)
    }

    fn initialize_global(
        &mut self,
        global: InternedString,
        source: InternedString,
        init: &js::Expression<'ast>,
        at: oxc_span::Span,
    ) -> Result<Stmt> {
        let (ty, view) = {
            let binding = &self.scopes[0][&source];
            (binding.ty.clone(), binding.view)
        };
        let value = self.expr_expected(init, &ty)?;
        self.expect(self.source_type(init, &value.ty), &ty, at)?;
        self.fresh_properties(init, &ty)?;
        self.check_view(self.view_of(init), view, at)?;
        self.check_enum(init, view, at)?;
        self.enum_facts(global, view, init);
        if matches!(ty, Type::Named { .. }) && value.ty != ty {
            return self.unsupported("assignment changing a binding's physical record layout", at);
        }
        if ty == string() && !self.immortal_string(init) {
            return self.unsupported(
                "module string bindings without a proven static lifetime",
                at,
            );
        }
        let value = self.adapt_value(value, &ty, at)?;
        if self.unassigned.contains_key(&global) {
            self.flow.assign(global);
        }
        Ok(expression(binary(
            BinaryOp::Assign,
            var(global, ty, at),
            value,
            unit(),
            at,
        )))
    }

    fn enum_ready(&mut self, declaration: &js::TSEnumDeclaration<'ast>) {
        let source = self.name(declaration.id.name.as_str());
        if !self.enums.contains_key(&source) {
            return;
        }
        let ready = self.global_name("enum_ready$", declaration.id.name.as_str());
        self.declarations.push(node(
            TypedDeclaration::Variable(TypedVariable {
                name: ready,
                ty: boolean(),
                mutability: Mutability::Mutable,
                initializer: None,
                visibility: Visibility::Public,
            }),
            unit(),
            declaration.span,
        ));
        self.enums.get_mut(&source).unwrap().ready = Some(ready);
    }
}
