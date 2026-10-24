//! Classes: a nominal view over a record layout. Constructors and methods are
//! functions; a method takes the instance as its first parameter.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FunctionRole {
    Plain,
    /// An instance method of the class with this view.
    Method(InternedString),
    Static,
    Constructor(InternedString),
}

impl FunctionRole {
    pub(super) fn class(self) -> Option<InternedString> {
        match self {
            Self::Method(view) | Self::Constructor(view) => Some(view),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub(super) struct ClassField<'ast, 'program> {
    pub name: &'ast str,
    pub ty: PrimitiveType,
    pub readonly: bool,
    pub private: bool,
    pub initializer: Option<&'program js::Expression<'ast>>,
    /// Declared by a constructor parameter.
    pub parameter: bool,
    pub definite: bool,
    pub span: oxc_span::Span,
}

pub(super) struct Member {
    pub key: InternedString,
    pub private: bool,
}

pub(super) struct Class<'ast, 'program> {
    pub ty: Type,
    pub fields: Vec<ClassField<'ast, 'program>>,
    pub constructor: Option<&'program js::Function<'ast>>,
    pub constructor_key: InternedString,
    pub component: String,
    pub methods: FxHashMap<&'ast str, Member>,
    pub statics: FxHashMap<&'ast str, Member>,
    /// Method bodies in source order, with the native-symbol component of each.
    pub bodies: Vec<(
        InternedString,
        String,
        &'program js::Function<'ast>,
        FunctionRole,
    )>,
    pub span: oxc_span::Span,
}

/// Names assigned as `this.name = ...` anywhere in a body.
struct ThisAssignments<'a> {
    names: FxHashSet<&'a str>,
}

impl<'a> Visit<'a> for ThisAssignments<'a> {
    fn visit_assignment_expression(&mut self, it: &js::AssignmentExpression<'a>) {
        if let js::AssignmentTarget::StaticMemberExpression(m) = &it.left {
            if matches!(m.object, js::Expression::ThisExpression(_)) {
                self.names.insert(m.property.name.as_str());
            }
        }
        walk::walk_assignment_expression(self, it);
    }
}

fn is_private(accessibility: Option<js::TSAccessibility>) -> bool {
    matches!(
        accessibility,
        Some(js::TSAccessibility::Private | js::TSAccessibility::Protected)
    )
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    pub(super) fn declare_class(&mut self, class: &'program js::Class<'ast>) -> Result<()> {
        let Some(id) = &class.id else {
            return self.unsupported("anonymous classes", class.span);
        };
        if class.type_parameters.is_some() {
            return self.unsupported("generic classes", class.span);
        }
        if class.heritage.is_some() {
            return self.unsupported("class inheritance", class.span);
        }

        if class.r#abstract
            || class.declare
            || self.declaration_file
            || !class.decorators.is_empty()
        {
            return self.unsupported("abstract, ambient or decorated classes", class.span);
        }
        let name = self.name(id.name.as_str());
        if self.aliases.contains_key(&name) {
            return self.error(ErrorKind::Type, "duplicate type declaration", id.span);
        }
        let source = id.name.as_str();
        let mut fields: Vec<ClassField<'ast, 'program>> = vec![];
        let mut constructor = None;
        let mut methods = FxHashMap::default();
        let mut statics = FxHashMap::default();
        let mut bodies = vec![];
        let mut nominal = false;
        for element in &class.body.body {
            match element {
                js::ClassElement::PropertyDefinition(p) => {
                    if p.r#static {
                        return self.unsupported("static fields", p.span);
                    }
                    if p.computed
                        || p.declare
                        || p.optional
                        || p.r#override
                        || !p.decorators.is_empty()
                    {
                        return self.unsupported("this class field form", p.span);
                    }
                    let field = self.class_key(&p.key)?;
                    let ty = match (&p.type_annotation, &p.value) {
                        (Some(annotation), _) => {
                            let (ty, view) = self.ty(&annotation.type_annotation)?;
                            self.plain_enum(view, "enum-typed fields", p.span)?;
                            let Some(ty) = field_primitive(&ty) else {
                                return self.unsupported("reference-valued object fields", p.span);
                            };
                            ty
                        }
                        (None, Some(js::Expression::NumericLiteral(_))) => PrimitiveType::F64,
                        (None, Some(js::Expression::BooleanLiteral(_))) => PrimitiveType::Bool,
                        (None, Some(js::Expression::StringLiteral(_))) => PrimitiveType::String,
                        _ => return self.unsupported("class field type inference", p.span),
                    };
                    if fields.iter().any(|f| f.name == field) {
                        return self.error(ErrorKind::Type, "duplicate identifier", p.span);
                    }
                    nominal |= is_private(p.accessibility);
                    fields.push(ClassField {
                        name: field,
                        ty,
                        readonly: p.readonly,
                        private: is_private(p.accessibility),
                        initializer: p.value.as_ref(),
                        parameter: false,
                        definite: p.definite,
                        span: p.span,
                    });
                }
                js::ClassElement::MethodDefinition(m) => {
                    if m.computed || m.optional || m.r#override || !m.decorators.is_empty() {
                        return self.unsupported("this class method form", m.span);
                    }
                    match m.kind {
                        js::MethodDefinitionKind::Constructor => {
                            if constructor.is_some() {
                                return self.unsupported("constructor overloads", m.span);
                            }
                            constructor = Some(&*m.value);
                        }
                        js::MethodDefinitionKind::Method => {
                            let method = self.class_key(&m.key)?;
                            if m.r#static && method == "constructor" {
                                return self.error(
                                    ErrorKind::Type,
                                    "'static' modifier cannot appear on a constructor declaration",
                                    m.span,
                                );
                            }
                            if methods.contains_key(method) || statics.contains_key(method) {
                                return self.unsupported("function overloads", m.span);
                            }
                            let private = is_private(m.accessibility);
                            let (table, key) = if m.r#static {
                                (&mut statics, intern(&format!("{source}::{method}")))
                            } else {
                                nominal = true;
                                (&mut methods, intern(&format!("{source}.{method}")))
                            };
                            table.insert(method, Member { key, private });
                            let component = if m.r#static {
                                format!("{source}::{method}")
                            } else {
                                format!("{source}.{method}")
                            };
                            bodies.push((key, component, &*m.value, m.r#static));
                        }
                        js::MethodDefinitionKind::Get | js::MethodDefinitionKind::Set => {
                            return self.unsupported("accessors", m.span);
                        }
                    }
                }
                _ => {
                    return self.unsupported(
                        "class static blocks, accessor properties or index signatures",
                        class.span,
                    );
                }
            }
        }
        if let Some(constructor) = constructor {
            for p in &constructor.params.items {
                if p.accessibility.is_none() && !p.readonly {
                    continue;
                }
                let js::BindingPattern::BindingIdentifier(parameter) = &p.pattern else {
                    return self.unsupported("destructuring parameter properties", p.span);
                };
                if p.optional {
                    return self.unsupported("optional parameter properties", p.span);
                }
                let Some(annotation) = &p.type_annotation else {
                    return self.error(
                        ErrorKind::Type,
                        "parameter implicitly has type any",
                        p.span,
                    );
                };
                let (ty, view) = self.ty(&annotation.type_annotation)?;
                self.plain_enum(view, "enum-typed fields", p.span)?;
                let Some(ty) = field_primitive(&ty) else {
                    return self.unsupported("reference-valued object fields", p.span);
                };
                if fields.iter().any(|f| f.name == parameter.name.as_str()) {
                    return self.error(ErrorKind::Type, "duplicate identifier", p.span);
                }
                nominal |= is_private(p.accessibility);
                fields.push(ClassField {
                    name: parameter.name.as_str(),
                    ty,
                    readonly: p.readonly,
                    private: is_private(p.accessibility),
                    initializer: None,
                    parameter: true,
                    definite: false,
                    span: p.span,
                });
            }
        }
        let mut layout: Fields<'ast> = fields
            .iter()
            .map(|f| {
                (
                    f.name,
                    Field {
                        ty: f.ty,
                        readonly: f.readonly,
                        span: f.span,
                    },
                )
            })
            .collect();
        layout.sort_by_key(|(name, _)| *name);
        let ty = self.layout(layout.iter().map(|(name, field)| (*name, field.ty)));
        let view = self.record_view(source, &ty, layout);
        Arc::get_mut(self.views.get_mut(&view).unwrap())
            .unwrap()
            .nominal = nominal;
        let bodies = bodies
            .into_iter()
            .map(|(key, component, f, is_static)| {
                let role = if is_static {
                    FunctionRole::Static
                } else {
                    FunctionRole::Method(view)
                };
                (key, component, f, role)
            })
            .collect();
        self.check_implements(class, &fields)?;
        self.aliases.insert(name, (ty.clone(), Some(view)));
        self.class_names.insert(name, view);
        self.class_order.push(view);
        self.classes.insert(
            view,
            Class {
                ty,
                fields,
                constructor,
                constructor_key: intern(&format!("{source}\0new")),
                component: format!("{source}\0new"),
                methods,
                statics,
                bodies,
                span: class.span,
            },
        );
        Ok(())
    }

    /// A class implementing record interfaces declares each of their fields, publicly and with
    /// the same type.
    fn check_implements(
        &self,
        class: &js::Class<'ast>,
        fields: &[ClassField<'ast, 'program>],
    ) -> Result<()> {
        for clause in &class.implements {
            let js::TSTypeName::IdentifierReference(name) = &clause.expression else {
                return self.unsupported("implementing qualified type names", clause.span);
            };
            if clause.type_arguments.is_some() {
                return self.unsupported("implementing generic interfaces", clause.span);
            }
            let key = self.name(name.name.as_str());
            let Some((_, view)) = self.aliases.get(&key) else {
                if self.declared_types.contains(&key) {
                    return self.unsupported(UNRESOLVED_REFERENCE, clause.span);
                }
                if library_type(name.name.as_str()) {
                    return self.unsupported(&format!("library type `{}`", name.name), clause.span);
                }
                return self.error(
                    ErrorKind::Type,
                    format!("cannot find name `{}`", name.name),
                    clause.span,
                );
            };
            if self.class_names.contains_key(&key) {
                return self.unsupported("classes implementing classes", clause.span);
            }
            let Some(view) = view.and_then(|v| self.views.get(&v)) else {
                return self.unsupported("implementing this type", clause.span);
            };
            for required in &view.fields {
                let property = required.name.resolve_global().unwrap_or_default();
                let found = fields.iter().find(|f| self.name(f.name) == required.name);
                let matches = found.is_some_and(|f| {
                    !f.private && Some(Type::Primitive(f.ty)) == Some(required.ty.clone())
                });
                if !matches {
                    return self.error(
                        ErrorKind::Type,
                        format!(
                            "class `{}` incorrectly implements `{}`: property `{property}`",
                            class.id.as_ref().map_or("", |id| id.name.as_str()),
                            name.name
                        ),
                        clause.span,
                    );
                }
            }
        }
        Ok(())
    }

    /// The only class with this physical layout; layouts shared by several types are ambiguous.
    pub(super) fn class_by_layout(&self, ty: &Type) -> Option<InternedString> {
        let mut matching = self
            .classes
            .iter()
            .filter(|(_, class)| class.ty == *ty)
            .map(|(view, _)| *view);
        let view = matching.next()?;
        matching.next().is_none().then_some(view)
    }

    /// The class of `array[index]` when `array` is a variable of class-instance elements.
    pub(super) fn element_view(&self, array: &js::Expression<'ast>) -> Option<InternedString> {
        let js::Expression::Identifier(id) = array else {
            return None;
        };
        let binding = self.binding(id.name.as_str(), id.span).ok()?;
        self.class_by_layout(element_type(&binding.ty)?)
    }

    pub(super) fn is_class_instance(&self, ty: &Type) -> bool {
        self.classes.values().any(|class| class.ty == *ty)
    }

    fn class_key(&self, key: &js::PropertyKey<'ast>) -> Result<&'ast str> {
        match key {
            js::PropertyKey::StaticIdentifier(i) => Ok(i.name.as_str()),
            js::PropertyKey::PrivateIdentifier(p) => self.unsupported("private names", p.span),
            _ => self.unsupported("computed property names", key.span()),
        }
    }

    pub(super) fn class_signatures(&mut self) -> Result<()> {
        for view in self.class_order.clone() {
            let class = &self.classes[&view];
            let constructor = class.constructor;
            let key = class.constructor_key;
            let component = class.component.clone();
            let ty = class.ty.clone();
            let bodies = class.bodies.clone();
            match constructor {
                Some(f) => {
                    self.signature_as(f, key, &component, FunctionRole::Constructor(view))?
                }
                None => {
                    self.name_buffer.truncate(self.prefix_len);
                    self.name_buffer.push_str("fn$");
                    crate::modules::append_component(&mut self.name_buffer, &component);
                    let native = intern(&self.name_buffer);
                    self.signatures.insert(
                        key,
                        Arc::new(Signature {
                            native,
                            type_params: vec![],
                            type_rules: None,
                            params: vec![],
                            ret: Some(ty),
                            ret_view: Some(view),
                            array_result: None,
                        }),
                    );
                }
            }
            for (key, component, f, role) in bodies {
                self.signature_as(f, key, &component, role)?;
            }
        }
        Ok(())
    }

    pub(super) fn lower_classes(&mut self) -> Result<()> {
        for view in self.class_order.clone() {
            let class = &self.classes[&view];
            let constructor = class.constructor;
            let key = class.constructor_key;
            let bodies = class.bodies.clone();
            match constructor {
                Some(f) => self.function_as(f, key, FunctionRole::Constructor(view))?,
                None => self.default_constructor(view)?,
            }
            for (key, _, f, role) in bodies {
                self.function_as(f, key, role)?;
            }
        }
        Ok(())
    }

    pub(super) fn this_parameter(&self, view: InternedString) -> CheckerParameter {
        CheckerParameter {
            name: self.name("this"),
            ty: self.classes[&view].ty.clone(),
            view: Some(view),
            ownership: ParamOwnership::Borrowed,
        }
    }

    pub(super) fn bind_this(&mut self, view: InternedString, at: oxc_span::Span) -> TypedParameter {
        let ty = self.classes[&view].ty.clone();
        let unique = self.fresh("this");
        let name = self.name("this");
        self.scopes.last_mut().unwrap().insert(
            name,
            Binding {
                name: unique,
                symbol: None,
                ty: ty.clone(),
                view: Some(view),
                mutable: false,
                initialized: true,
            },
        );
        let mut parameter = TypedParameter::regular(unique, ty, Mutability::Mutable, span(at));
        parameter.ownership = ParamOwnership::Borrowed;
        parameter
    }

    fn field_assignment(
        &self,
        this: InternedString,
        ty: &Type,
        field: &'ast str,
        value_type: PrimitiveType,
        value: Expr,
        at: oxc_span::Span,
    ) -> Stmt {
        let target = node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(var(this, ty.clone(), at)),
                field: self.name(field),
            }),
            Type::Primitive(value_type),
            at,
        );
        expression(binary(BinaryOp::Assign, target, value, unit(), at))
    }

    /// Allocates the instance with zeroed fields, then runs parameter properties and
    /// field initializers in order. Returns the instance's binding name.
    pub(super) fn constructor_prologue(
        &mut self,
        view: InternedString,
        f: Option<&js::Function<'ast>>,
        statements: &mut Vec<Stmt>,
    ) -> Result<InternedString> {
        let ty = self.classes[&view].ty.clone();
        let fields = self.classes[&view].fields.clone();
        let at = f.map_or(self.classes[&view].span, |f| f.span);
        let Type::Named { id, .. } = &ty else {
            unreachable!()
        };
        let struct_name = self.registry.get_type_by_id(*id).unwrap().name;
        let zeros = fields
            .iter()
            .map(|field| TypedFieldInit {
                name: self.name(field.name),
                value: Box::new(match field.ty {
                    PrimitiveType::Bool => node(
                        TypedExpression::Literal(TypedLiteral::Bool(false)),
                        boolean(),
                        at,
                    ),
                    PrimitiveType::String => node(
                        TypedExpression::Literal(TypedLiteral::String(self.name(""))),
                        string(),
                        at,
                    ),
                    _ => num(0.0, at),
                }),
            })
            .collect();
        let this = self.fresh("this");
        statements.push(stmt(
            TypedStatement::Let(TypedLet {
                name: this,
                ty: ty.clone(),
                mutability: Mutability::Mutable,
                initializer: Some(Box::new(node(
                    TypedExpression::Struct(TypedStructLiteral {
                        name: struct_name,
                        fields: zeros,
                    }),
                    ty.clone(),
                    at,
                ))),
                span: span(at),
            }),
            at,
        ));
        let name = self.name("this");
        self.scopes.last_mut().unwrap().insert(
            name,
            Binding {
                name: this,
                symbol: None,
                ty: ty.clone(),
                view: Some(view),
                mutable: false,
                initialized: true,
            },
        );
        if let Some(f) = f {
            for p in &f.params.items {
                if p.accessibility.is_none() && !p.readonly {
                    continue;
                }
                let js::BindingPattern::BindingIdentifier(parameter) = &p.pattern else {
                    unreachable!()
                };
                let (value, field) = {
                    let binding = self.binding(parameter.name.as_str(), p.span)?;
                    let field = fields.iter().find(|f| f.name == parameter.name.as_str());
                    (
                        var(binding.name, binding.ty.clone(), p.span),
                        field.unwrap(),
                    )
                };
                if field.ty == PrimitiveType::String {
                    return self
                        .unsupported("string fields without a proven static lifetime", p.span);
                }
                statements
                    .push(self.field_assignment(this, &ty, field.name, field.ty, value, p.span));
            }
        }
        for field in fields.iter().filter(|f| !f.parameter) {
            let Some(initializer) = field.initializer else {
                continue;
            };
            let expected = Type::Primitive(field.ty);
            let value = self.expr_expected(initializer, &expected)?;
            self.expect(&value.ty, &expected, field.span)?;
            if field.ty == PrimitiveType::String && !self.immortal_string(initializer) {
                return self
                    .unsupported("string fields without a proven static lifetime", field.span);
            }
            statements
                .push(self.field_assignment(this, &ty, field.name, field.ty, value, field.span));
        }
        Ok(this)
    }

    /// TypeScript's strict property initialization: every field without an initializer
    /// is assigned in the constructor.
    pub(super) fn check_field_initialization(
        &self,
        view: InternedString,
        constructor: &js::Function<'ast>,
    ) -> Result<()> {
        let mut assigned = ThisAssignments {
            names: FxHashSet::default(),
        };
        assigned.visit_function_body(constructor.body.as_ref().unwrap());
        self.require_initialized(view, &assigned.names)
    }

    fn require_initialized(&self, view: InternedString, assigned: &FxHashSet<&str>) -> Result<()> {
        for field in &self.classes[&view].fields {
            if field.initializer.is_none()
                && !field.parameter
                && !field.definite
                && !assigned.contains(field.name)
            {
                return self.error(
                    ErrorKind::Type,
                    format!(
                        "property `{}` has no initializer and is not definitely assigned in the constructor",
                        field.name
                    ),
                    field.span,
                );
            }
        }
        Ok(())
    }

    pub(super) fn default_constructor(&mut self, view: InternedString) -> Result<()> {
        let key = self.classes[&view].constructor_key;
        if self.lowered.contains(&key) {
            return Ok(());
        }
        self.require_initialized(view, &FxHashSet::default())?;
        let sig = self.signatures[&key].clone();
        let at = self.classes[&view].span;
        self.scopes.push(FxHashMap::default());
        let outer_vars = std::mem::replace(
            &mut self.vars,
            FunctionVars {
                scope: self.scopes.len() - 1,
                hoisted: vec![],
            },
        );
        let outer_class = self.current_class.replace(view);
        let outer_constructor = std::mem::replace(&mut self.in_constructor, true);
        let outer_returns = self.returns.take();
        let mut statements = vec![];
        let result = self.constructor_prologue(view, None, &mut statements);
        self.returns = outer_returns;
        self.in_constructor = outer_constructor;
        self.current_class = outer_class;
        self.vars = outer_vars;
        self.pop_scope();
        let this = result?;
        let ty = sig.ret.clone().unwrap();
        statements.push(stmt(
            TypedStatement::Return(Some(Box::new(var(this, ty.clone(), at)))),
            at,
        ));
        self.declarations.push(node(
            TypedDeclaration::Function(TypedFunction {
                name: sig.native,
                annotations: vec![strict_fp(self.name("strict_fp"))],
                params: vec![],
                return_type: ty,
                body: Some(TypedBlock {
                    statements,
                    span: span(at),
                }),
                visibility: Visibility::Private,
                ..Default::default()
            }),
            unit(),
            at,
        ));
        self.lowered.insert(key);
        Ok(())
    }

    fn check_member_access(
        &self,
        view: InternedString,
        private: bool,
        name: &str,
        at: oxc_span::Span,
    ) -> Result<()> {
        if private && self.current_class != Some(view) {
            return self.error(
                ErrorKind::Type,
                format!("property `{name}` is private and only accessible within its class"),
                at,
            );
        }
        Ok(())
    }

    /// Field reads and writes through a class-typed receiver honour access modifiers.
    pub(super) fn check_field_access(
        &self,
        object: &js::Expression<'ast>,
        field: &str,
        at: oxc_span::Span,
    ) -> Result<()> {
        let Some(view) = self
            .view_of(object)
            .filter(|v| self.classes.contains_key(v))
        else {
            return Ok(());
        };
        let private = self.classes[&view]
            .fields
            .iter()
            .find(|f| f.name == field)
            .is_some_and(|f| f.private);
        self.check_member_access(view, private, field, at)
    }

    pub(super) fn readonly_field_write_allowed(&self, object: &js::Expression<'ast>) -> bool {
        self.in_constructor && matches!(object, js::Expression::ThisExpression(_))
    }

    /// `receiver.method(...)` and `Class.method(...)`; `None` when not a class member call.
    pub(super) fn method_call(
        &mut self,
        c: &js::CallExpression<'ast>,
        m: &js::StaticMemberExpression<'ast>,
    ) -> Result<Option<Expr>> {
        let name = m.property.name.as_str();
        if let js::Expression::Identifier(id) = &m.object {
            let class_name = self.name(id.name.as_str());
            if !self.scopes.iter().any(|s| s.contains_key(&class_name)) {
                if let Some(view) = self.class_names.get(&class_name).copied() {
                    let Some(member) = self.classes[&view].statics.get(name) else {
                        return self.error(
                            ErrorKind::Type,
                            format!("property `{name}` does not exist on class `{}`", id.name),
                            m.span,
                        );
                    };
                    let (key, private) = (member.key, member.private);
                    self.check_member_access(view, private, name, m.span)?;
                    let sig = self.signatures[&key].clone();
                    return self
                        .call_with_signature(
                            sig,
                            &c.arguments,
                            c.type_arguments.as_deref(),
                            None,
                            c.span,
                        )
                        .map(Some);
                }
            }
        }
        let Some(view) = self
            .view_of(&m.object)
            .filter(|v| self.classes.contains_key(v))
        else {
            return Ok(None);
        };
        let class = &self.classes[&view];
        let Some(member) = class.methods.get(name) else {
            return if class.fields.iter().any(|f| f.name == name) {
                Ok(None)
            } else {
                self.error(
                    ErrorKind::Type,
                    format!("property `{name}` does not exist on this class"),
                    m.span,
                )
            };
        };
        let (key, private) = (member.key, member.private);
        self.check_member_access(view, private, name, m.span)?;
        let receiver = self.expr(&m.object)?;
        let sig = &self.signatures[&key];
        let stripped = Arc::new(Signature {
            params: sig.params[1..].to_vec(),
            ..(**sig).clone()
        });
        self.call_with_signature(
            stripped,
            &c.arguments,
            c.type_arguments.as_deref(),
            Some(receiver),
            c.span,
        )
        .map(Some)
    }

    /// The class view a method call produces, when its signature names one.
    pub(super) fn method_return_view(
        &self,
        callee: &js::Expression<'ast>,
    ) -> Option<InternedString> {
        let js::Expression::StaticMemberExpression(m) = callee else {
            return None;
        };
        let name = m.property.name.as_str();
        if let js::Expression::Identifier(id) = &m.object {
            let class_name = self.name(id.name.as_str());
            if !self.scopes.iter().any(|s| s.contains_key(&class_name)) {
                if let Some(view) = self.class_names.get(&class_name) {
                    let member = self.classes[view].statics.get(name)?;
                    return self.signatures[&member.key].ret_view;
                }
            }
        }
        let view = self
            .view_of(&m.object)
            .filter(|v| self.classes.contains_key(v))?;
        let member = self.classes[&view].methods.get(name)?;
        self.signatures[&member.key].ret_view
    }

    pub(super) fn new_expr(&mut self, n: &js::NewExpression<'ast>) -> Result<Expr> {
        let js::Expression::Identifier(id) = &n.callee else {
            return self.unsupported("new on this callee", n.span);
        };
        let class_name = self.name(id.name.as_str());
        let Some(view) = self
            .class_names
            .get(&class_name)
            .copied()
            .filter(|_| !self.scopes.iter().any(|s| s.contains_key(&class_name)))
        else {
            return self.unsupported("new on a value that is not a class declaration", n.span);
        };
        if n.type_arguments.is_some() {
            return self.error(ErrorKind::Type, "class is not generic", n.span);
        }
        let sig = self.signatures[&self.classes[&view].constructor_key].clone();
        self.call_with_signature(sig, &n.arguments, None, None, n.span)
    }

    pub(super) fn this_expr(&mut self, at: oxc_span::Span) -> Result<Expr> {
        let name = self.name("this");
        if !self.scopes.iter().any(|s| s.contains_key(&name)) {
            return self.unsupported("this outside instance methods and constructors", at);
        }
        let b = self.binding("this", at)?;
        Ok(var(b.name, b.ty.clone(), at))
    }

    /// The class view a `new` expression produces.
    pub(super) fn class_of_new(&self, n: &js::NewExpression<'ast>) -> Option<InternedString> {
        let js::Expression::Identifier(id) = &n.callee else {
            return None;
        };
        self.class_names.get(&self.name(id.name.as_str())).copied()
    }

    /// A value of a class with instance members needs that class, not a lookalike record.
    pub(super) fn check_class_target(
        &self,
        from: Option<InternedString>,
        to: Option<InternedString>,
        at: oxc_span::Span,
    ) -> Result<()> {
        if let Some(to) = to {
            if self.views.get(&to).is_some_and(|v| v.nominal) && from != Some(to) {
                return self.error(
                    ErrorKind::Type,
                    "value is missing members required by the class type",
                    at,
                );
            }
        }
        Ok(())
    }
}
