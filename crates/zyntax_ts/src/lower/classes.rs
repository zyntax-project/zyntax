//! Classes: a nominal view over a record layout. Constructors and methods are
//! functions; a method takes the instance as its first parameter.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FunctionRole {
    Plain,
    /// An instance method of the class with this view.
    Method(InternedString),
    /// A static method of the class with this view.
    Static(InternedString),
    Constructor(InternedString),
    /// The constructor body run on an instance allocated elsewhere: by `new` of the class
    /// itself or of a subclass.
    Initializer(InternedString),
}

impl FunctionRole {
    pub(super) fn class(self) -> Option<InternedString> {
        match self {
            Self::Method(view) | Self::Constructor(view) | Self::Initializer(view) => Some(view),
            _ => None,
        }
    }

    /// The class whose body holds the function, which may use its private members.
    pub(super) fn body_class(self) -> Option<InternedString> {
        match self {
            Self::Static(view) => Some(view),
            role => role.class(),
        }
    }

    /// Whether `this` is the function's first parameter.
    pub(super) fn takes_this(self) -> Option<InternedString> {
        match self {
            Self::Method(view) | Self::Initializer(view) => Some(view),
            _ => None,
        }
    }
}

/// The hidden field holding an instance's class in a hierarchy with subclasses.
pub(super) const CLASS_TAG: &str = "\0class";

/// Who may use a member.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Access {
    Public,
    Protected,
    Private,
}

/// Whether an expression reads `this.name` for one of `names`, outside nested functions.
struct ThisReads<'n> {
    names: &'n FxHashSet<&'n str>,
    found: bool,
}

impl<'a> Visit<'a> for ThisReads<'_> {
    fn visit_static_member_expression(&mut self, it: &js::StaticMemberExpression<'a>) {
        if matches!(it.object, js::Expression::ThisExpression(_))
            && self.names.contains(it.property.name.as_str())
        {
            self.found = true;
        }
        walk::walk_static_member_expression(self, it);
    }

    fn visit_function(&mut self, _: &js::Function<'a>, _: oxc_semantic::ScopeFlags) {}
}

/// Counts a body's `return this` statements and its other returns, outside nested functions.
struct ThisReturns {
    this: usize,
    other: usize,
}

impl<'a> Visit<'a> for ThisReturns {
    fn visit_return_statement(&mut self, it: &js::ReturnStatement<'a>) {
        match &it.argument {
            Some(js::Expression::ThisExpression(_)) => self.this += 1,
            _ => self.other += 1,
        }
    }

    fn visit_function(&mut self, _: &js::Function<'a>, _: oxc_semantic::ScopeFlags) {}

    fn visit_arrow_function_expression(&mut self, _: &js::ArrowFunctionExpression<'a>) {}
}

/// Whether a `super(...)` call occurs.
struct SuperCalls {
    found: bool,
}

impl<'a> Visit<'a> for SuperCalls {
    fn visit_call_expression(&mut self, it: &js::CallExpression<'a>) {
        self.found |= matches!(it.callee, js::Expression::Super(_));
        walk::walk_call_expression(self, it);
    }
}

pub(super) fn access(accessibility: Option<js::TSAccessibility>) -> Access {
    match accessibility {
        Some(js::TSAccessibility::Private) => Access::Private,
        Some(js::TSAccessibility::Protected) => Access::Protected,
        _ => Access::Public,
    }
}

#[derive(Clone)]
pub(super) struct ClassField<'ast, 'program> {
    pub name: &'ast str,
    pub ty: PrimitiveType,
    pub readonly: bool,
    pub access: Access,
    /// The class that declares an inherited field; `None` for the class's own fields.
    pub owner: Option<InternedString>,
    pub initializer: Option<&'program js::Expression<'ast>>,
    /// Declared by a constructor parameter.
    pub parameter: bool,
    /// Redeclares an inherited field, whose layout slot it keeps.
    pub inherited_slot: bool,
    pub definite: bool,
    pub span: oxc_span::Span,
}

pub(super) struct Member {
    /// The implementing function.
    pub key: InternedString,
    pub access: Access,
    /// The class that declares an inherited member; `None` for the class's own members.
    pub owner: Option<InternedString>,
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
    pub base: Option<InternedString>,
    pub constructor_access: Access,
    /// The hidden class field's value, for classes in a hierarchy.
    pub tag: Option<f64>,
    /// The initializer's key and native component, for classes others extend.
    pub init: Option<(InternedString, String)>,
    /// Own and inherited static fields.
    pub static_fields: Vec<statics::StaticField<'ast, 'program>>,
    pub static_elements: Vec<statics::StaticElement<'ast, 'program>>,
    /// Set when the class declaration has run, for classes with static members.
    pub ready: Option<InternedString>,
    pub span: oxc_span::Span,
}

/// Names assigned as `this.name = ...` anywhere in a body.
struct ThisAssignments<'a> {
    source: &'a str,
    names: FxHashSet<&'a str>,
}

impl<'a> Visit<'a> for ThisAssignments<'a> {
    fn visit_assignment_expression(&mut self, it: &js::AssignmentExpression<'a>) {
        match &it.left {
            js::AssignmentTarget::StaticMemberExpression(m)
                if matches!(m.object, js::Expression::ThisExpression(_)) =>
            {
                self.names.insert(m.property.name.as_str());
            }
            js::AssignmentTarget::PrivateFieldExpression(p)
                if matches!(p.object, js::Expression::ThisExpression(_)) =>
            {
                self.names.insert(private_name(self.source, &p.field));
            }
            _ => {}
        }
        walk::walk_assignment_expression(self, it);
    }
}

/// A private name as its source text, `#name`, which no public member can be called.
pub(super) fn private_name<'a>(source: &'a str, name: &js::PrivateIdentifier<'_>) -> &'a str {
    &source[name.span.start as usize..name.span.end as usize]
}

/// A member's access: `#name` is private wherever it is declared.
fn member_access(name: &str, accessibility: Option<js::TSAccessibility>) -> Access {
    if name.starts_with('#') {
        Access::Private
    } else {
        access(accessibility)
    }
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    pub(super) fn declare_class(&mut self, class: &'program js::Class<'ast>) -> Result<()> {
        let Some(id) = &class.id else {
            return self.unsupported("anonymous classes", class.span);
        };
        if class.type_parameters.is_some() {
            return self.unsupported("generic classes", class.span);
        }
        if class.r#abstract
            || class.declare
            || self.declaration_file
            || !class.decorators.is_empty()
        {
            return self.unsupported("abstract, ambient or decorated classes", class.span);
        }
        let base = match &class.heritage {
            Some(heritage) => Some(self.class_base(class, heritage)?),
            None => None,
        };
        let name = self.name(id.name.as_str());
        if self.aliases.contains_key(&name) {
            return self.error(ErrorKind::Type, "duplicate type declaration", id.span);
        }
        let source = id.name.as_str();
        // Every class of a hierarchy carries its class in a hidden field, for dispatch.
        let tag = (base.is_some() || self.extended_classes.contains(&name))
            .then(|| self.class_order.len() as f64);
        let mut static_fields: Vec<statics::StaticField<'ast, 'program>> = base
            .map(|base| {
                self.classes[&base]
                    .static_fields
                    .iter()
                    .map(|f| statics::StaticField {
                        owner: Some(f.owner.unwrap_or(base)),
                        ..f.clone()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut static_elements = vec![];
        let (mut fields, mut methods, mut statics) = match base {
            Some(base) => {
                let parent = &self.classes[&base];
                let inherit = |m: &Member| Member {
                    key: m.key,
                    access: m.access,
                    owner: Some(m.owner.unwrap_or(base)),
                };
                (
                    parent
                        .fields
                        .iter()
                        .map(|f| ClassField {
                            owner: Some(f.owner.unwrap_or(base)),
                            ..f.clone()
                        })
                        .collect::<Vec<_>>(),
                    parent
                        .methods
                        .iter()
                        .map(|(k, m)| (*k, inherit(m)))
                        .collect::<FxHashMap<_, _>>(),
                    parent
                        .statics
                        .iter()
                        .map(|(k, m)| (*k, inherit(m)))
                        .collect::<FxHashMap<_, _>>(),
                )
            }
            None => Default::default(),
        };
        let mut constructor = None;
        let mut constructor_access = Access::Public;
        let mut own_members: FxHashSet<&'ast str> = FxHashSet::default();
        let mut bodies = vec![];
        let mut nominal = !methods.is_empty() || fields.iter().any(|f| f.access != Access::Public);
        for element in &class.body.body {
            match element {
                js::ClassElement::PropertyDefinition(p) => {
                    if p.r#static {
                        let name =
                            self.declare_static_field(source, p, &mut static_fields, &statics)?;
                        static_elements.push(statics::StaticElement::Field(name));
                        continue;
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
                        // A redeclared field's initializer is checked against the inherited
                        // type; a different type is TypeScript's incompatible override.
                        (None, Some(_))
                            if let Some(inherited) =
                                fields.iter().find(|f| f.name == field && f.owner.is_some()) =>
                        {
                            inherited.ty
                        }
                        (None, Some(init)) => {
                            match self.field_initializer_type(init, class, &fields) {
                                Some(ty) => ty,
                                None => {
                                    return self.unsupported("class field type inference", p.span);
                                }
                            }
                        }
                        _ => return self.unsupported("class field type inference", p.span),
                    };
                    if fields.iter().any(|f| f.name == field && f.owner.is_none()) {
                        return self.error(ErrorKind::Type, "duplicate identifier", p.span);
                    }
                    let redeclared = self.check_inherited_member(
                        source,
                        &fields,
                        &methods,
                        field,
                        Some(ty),
                        p.value.is_some(),
                        p.span,
                    )?;
                    let access = member_access(field, p.accessibility);
                    if let Some(index) = redeclared {
                        self.check_field_visibility(source, &fields[index], access, p.span)?;
                    }
                    nominal |= access != Access::Public;
                    let declared = ClassField {
                        name: field,
                        ty,
                        readonly: p.readonly,
                        access,
                        owner: None,
                        initializer: p.value.as_ref(),
                        parameter: false,
                        inherited_slot: redeclared.is_some(),
                        definite: p.definite,
                        span: p.span,
                    };
                    match redeclared {
                        Some(index) => fields[index] = declared,
                        None => fields.push(declared),
                    }
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
                            constructor_access = access(m.accessibility);
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
                            if !own_members.insert(method) {
                                return self.unsupported("function overloads", m.span);
                            }
                            if m.r#static
                                && let Some(field) = static_fields.iter().find(|f| f.name == method)
                            {
                                if field.owner.is_none() {
                                    return self.error(
                                        ErrorKind::Type,
                                        "duplicate identifier",
                                        m.span,
                                    );
                                }
                                return self.error(
                                    ErrorKind::Type,
                                    format!("class `{source}` static side incorrectly extends its base class static side: property `{method}`"),
                                    m.span,
                                );
                            }
                            let access = member_access(method, m.accessibility);
                            let (table, key) = if m.r#static {
                                (&mut statics, intern(&format!("{source}::{method}")))
                            } else {
                                nominal = true;
                                (&mut methods, intern(&format!("{source}.{method}")))
                            };
                            if let Some(inherited) = table.get(method) {
                                if method.starts_with('#') {
                                    return self.unsupported(
                                        "private names redeclared in subclasses",
                                        m.span,
                                    );
                                }
                                self.check_override_access(
                                    source, method, inherited, access, m.span,
                                )?;
                            }
                            if !m.r#static {
                                self.check_inherited_member(
                                    source,
                                    &fields,
                                    &FxHashMap::default(),
                                    method,
                                    None,
                                    true,
                                    m.span,
                                )?;
                            }
                            table.insert(
                                method,
                                Member {
                                    key,
                                    access,
                                    owner: None,
                                },
                            );
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
                js::ClassElement::StaticBlock(block) => {
                    static_elements.push(statics::StaticElement::Block(block));
                }
                _ => {
                    return self.unsupported("accessor properties or index signatures", class.span);
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
                let field = parameter.name.as_str();
                if fields.iter().any(|f| f.name == field && f.owner.is_none()) {
                    return self.error(ErrorKind::Type, "duplicate identifier", p.span);
                }
                let redeclared = self.check_inherited_member(
                    source,
                    &fields,
                    &methods,
                    field,
                    Some(ty),
                    true,
                    p.span,
                )?;
                let access = access(p.accessibility);
                if let Some(index) = redeclared {
                    self.check_field_visibility(source, &fields[index], access, p.span)?;
                }
                nominal |= access != Access::Public;
                let declared = ClassField {
                    name: field,
                    ty,
                    readonly: p.readonly,
                    access,
                    owner: None,
                    initializer: None,
                    parameter: true,
                    inherited_slot: redeclared.is_some(),
                    definite: false,
                    span: p.span,
                };
                match redeclared {
                    Some(index) => fields[index] = declared,
                    None => fields.push(declared),
                }
            }
        }
        // A subclass keeps its base's layout as a prefix, so a base view reads it in place.
        let mut key: SmallVec<[(InternedString, PrimitiveType); 4]> = match base {
            Some(base) => self.layout_key(&self.classes[&base].ty),
            None if tag.is_some() => [(self.name(CLASS_TAG), PrimitiveType::F64)]
                .into_iter()
                .collect(),
            None => SmallVec::new(),
        };
        let mut own: Vec<&ClassField<'ast, 'program>> = fields
            .iter()
            .filter(|f| f.owner.is_none() && !f.inherited_slot)
            .collect();
        own.sort_by_key(|f| f.name);
        key.extend(own.iter().map(|f| (self.name(f.name), f.ty)));
        let ty = self.layout_of(key);
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
        let view = self.record_view(source, &ty, layout);
        Arc::get_mut(self.views.get_mut(&view).unwrap())
            .unwrap()
            .nominal = nominal;
        let bodies = bodies
            .into_iter()
            .map(|(key, component, f, is_static)| {
                let role = if is_static {
                    FunctionRole::Static(view)
                } else {
                    FunctionRole::Method(view)
                };
                (key, component, f, role)
            })
            .collect();
        self.check_implements(class, &fields)?;
        let init = self.extended_classes.contains(&name).then(|| {
            (
                intern(&format!("{source}\0init")),
                format!("{source}\0init"),
            )
        });
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
                base,
                constructor_access,
                tag,
                init,
                static_fields,
                static_elements,
                ready: None,
                span: class.span,
            },
        );
        self.declare_static_globals(view, source);
        Ok(())
    }

    /// The class `class` extends: a class this module declares before it.
    fn class_base(
        &self,
        class: &js::Class<'ast>,
        heritage: &js::ClassHeritage<'ast>,
    ) -> Result<InternedString> {
        let at = heritage.expression.span();
        let not_constructor = || {
            self.error(
                ErrorKind::Type,
                "the class heritage is not a constructor function type",
                at,
            )
        };
        let js::Expression::Identifier(base) = &heritage.expression else {
            return match heritage.expression {
                js::Expression::NullLiteral(_) => self.unsupported("classes extending null", at),
                js::Expression::BooleanLiteral(_)
                | js::Expression::NumericLiteral(_)
                | js::Expression::StringLiteral(_)
                | js::Expression::TemplateLiteral(_) => not_constructor(),
                _ => self.unsupported("class heritage expressions", at),
            };
        };
        if heritage.type_arguments.is_some() {
            return self.unsupported("extending generic classes", at);
        }
        let key = self.name(base.name.as_str());
        if let Some(view) = self.class_names.get(&key) {
            if self.classes[view].span.start > class.span.start {
                return self.error(
                    ErrorKind::Type,
                    format!("class `{}` used before its declaration", base.name),
                    at,
                );
            }
            if self.classes[view].constructor_access == Access::Private {
                return self.error(
                    ErrorKind::Type,
                    format!(
                        "cannot extend class `{}`: its constructor is marked as private",
                        base.name
                    ),
                    at,
                );
            }
            return Ok(*view);
        }
        if self.aliases.contains_key(&key) {
            return self.error(
                ErrorKind::Type,
                format!("cannot extend `{}`, which only refers to a type", base.name),
                at,
            );
        }
        if self.declared_types.contains(&key) {
            return self.unsupported(UNRESOLVED_REFERENCE, at);
        }
        if library_value(base.name.as_str()) {
            return self.unsupported(&format!("extending library class `{}`", base.name), at);
        }
        if self.top_level_values.contains(&key) {
            return self.unsupported("extending values that are not class declarations", at);
        }
        if base.name == "undefined" {
            return not_constructor();
        }
        self.error(
            ErrorKind::Type,
            format!("cannot find name `{}`", base.name),
            at,
        )
    }

    /// The type of an instance field initializer before any body is lowered: literals, operators
    /// on them and fields declared before it.
    fn field_initializer_type(
        &self,
        e: &js::Expression<'ast>,
        class: &js::Class<'ast>,
        fields: &[ClassField<'ast, 'program>],
    ) -> Option<PrimitiveType> {
        let operand = |e| self.field_initializer_type(e, class, fields);
        match e {
            js::Expression::NumericLiteral(_) => Some(PrimitiveType::F64),
            js::Expression::BooleanLiteral(_) => Some(PrimitiveType::Bool),
            js::Expression::StringLiteral(s) if !s.lone_surrogates => Some(PrimitiveType::String),
            js::Expression::TemplateLiteral(_) => Some(PrimitiveType::String),
            js::Expression::ParenthesizedExpression(p) => operand(&p.expression),
            js::Expression::UnaryExpression(u) => match u.operator.as_str() {
                "-" | "+" | "~" => Some(PrimitiveType::F64),
                "!" => Some(PrimitiveType::Bool),
                "typeof" => Some(PrimitiveType::String),
                _ => None,
            },
            js::Expression::BinaryExpression(b) => match b.operator.as_str() {
                "-" | "*" | "/" | "%" | "**" | "|" | "&" | "^" | "<<" | ">>" | ">>>" => {
                    Some(PrimitiveType::F64)
                }
                "<" | "<=" | ">" | ">=" | "===" | "!==" | "==" | "!=" => Some(PrimitiveType::Bool),
                "+" => match (operand(&b.left)?, operand(&b.right)?) {
                    (PrimitiveType::String, _) | (_, PrimitiveType::String) => {
                        Some(PrimitiveType::String)
                    }
                    (PrimitiveType::F64, PrimitiveType::F64) => Some(PrimitiveType::F64),
                    _ => None,
                },
                _ => None,
            },
            js::Expression::StaticMemberExpression(m)
                if !m.optional && matches!(m.object, js::Expression::ThisExpression(_)) =>
            {
                let name = m.property.name.as_str();
                // Parameter properties are assigned before or after field initializers
                // depending on the class field semantics the target selects.
                fields
                    .iter()
                    .find(|f| f.name == name && !f.parameter)
                    .map(|f| f.ty)
            }
            _ => None,
        }
    }

    /// The field names and types of a class layout, in layout order.
    fn layout_key(&self, ty: &Type) -> SmallVec<[(InternedString, PrimitiveType); 4]> {
        let Type::Named { id, .. } = ty else {
            unreachable!("class layouts are named records")
        };
        self.registry
            .get_type_by_id(*id)
            .unwrap()
            .fields
            .iter()
            .map(|field| match field.ty {
                Type::Primitive(primitive) => (field.name, primitive),
                _ => unreachable!("class fields are primitives"),
            })
            .collect()
    }

    /// A member a subclass declares must agree in kind and type with the one it inherits.
    /// Returns the inherited field a field declaration redeclares, which must give it a value.
    #[allow(clippy::too_many_arguments)]
    fn check_inherited_member(
        &self,
        class: &str,
        fields: &[ClassField<'ast, 'program>],
        methods: &FxHashMap<&'ast str, Member>,
        name: &str,
        field_type: Option<PrimitiveType>,
        valued: bool,
        at: oxc_span::Span,
    ) -> Result<Option<usize>> {
        let inherited_field = fields
            .iter()
            .position(|f| f.name == name && f.owner.is_some());
        // A private name is a new member in each class that declares it.
        if name.starts_with('#')
            && (inherited_field.is_some() || methods.get(name).is_some_and(|m| m.owner.is_some()))
        {
            return self.unsupported("private names redeclared in subclasses", at);
        }
        let incompatible = match (inherited_field, field_type) {
            (Some(index), Some(ty)) if fields[index].ty == ty => {
                // Without a value, the redeclaration's effect depends on the class field
                // semantics the compiler options select.
                if !valued {
                    return self.unsupported("redeclared inherited fields without a value", at);
                }
                return Ok(Some(index));
            }
            (Some(_), _) => true,
            (None, Some(_)) => methods.get(name).is_some_and(|m| m.owner.is_some()),
            (None, None) => false,
        };
        if incompatible {
            return self.error(
                ErrorKind::Type,
                format!(
                    "property `{name}` in class `{class}` is not assignable to the same property in its base class"
                ),
                at,
            );
        }
        Ok(None)
    }

    /// A redeclared field keeps the inherited one at least as visible.
    fn check_field_visibility(
        &self,
        class: &str,
        inherited: &ClassField<'ast, 'program>,
        access: Access,
        at: oxc_span::Span,
    ) -> Result<()> {
        let member = Member {
            key: intern(""),
            access: inherited.access,
            owner: inherited.owner,
        };
        self.check_override_access(class, inherited.name, &member, access, at)
    }

    /// An override keeps the inherited member at least as visible, and cannot replace a
    /// private one.
    fn check_override_access(
        &self,
        class: &str,
        name: &str,
        inherited: &Member,
        access: Access,
        at: oxc_span::Span,
    ) -> Result<()> {
        let narrower = match inherited.access {
            Access::Private => true,
            Access::Protected => access == Access::Private,
            Access::Public => access != Access::Public,
        };
        if inherited.owner.is_some() && narrower {
            return self.error(
                ErrorKind::Type,
                format!("class `{class}` incorrectly extends its base class: member `{name}` has another visibility"),
                at,
            );
        }
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
            if view.is_some_and(|v| self.method_views.contains_key(&v)) {
                return self.unsupported("implementing interfaces with methods", clause.span);
            }
            let Some(view) = view.and_then(|v| self.views.get(&v)) else {
                return self.unsupported("implementing this type", clause.span);
            };
            for required in &view.fields {
                let property = required.name.resolve_global().unwrap_or_default();
                let found = fields.iter().find(|f| self.name(f.name) == required.name);
                let matches = found.is_some_and(|f| {
                    f.access == Access::Public
                        && Some(Type::Primitive(f.ty)) == Some(required.ty.clone())
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

    pub(super) fn class_key(&self, key: &js::PropertyKey<'ast>) -> Result<&'ast str> {
        match key {
            js::PropertyKey::StaticIdentifier(i) => Ok(i.name.as_str()),
            js::PropertyKey::PrivateIdentifier(p) => Ok(private_name(self.source_text, p)),
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
            let init = class.init.clone();
            let base = class.base;
            // An implicit derived constructor takes its base constructor's parameters.
            let (forwarded, arity) = match base {
                Some(base) if constructor.is_none() => {
                    let (base_init, _) = self.classes[&base].init.as_ref().unwrap();
                    let sig = &self.signatures[base_init];
                    (
                        sig.params[1..].to_vec(),
                        self.arities.get(&sig.native).copied(),
                    )
                }
                _ => (vec![], None),
            };
            match constructor {
                Some(f) => {
                    self.signature_as(f, key, &component, FunctionRole::Constructor(view))?;
                    if let Some((init, component)) = &init {
                        self.signature_as(f, *init, component, FunctionRole::Initializer(view))?;
                    }
                }
                None => {
                    let native = self.native_name(&component);
                    if let Some(arity) = arity {
                        self.arities.insert(native, arity);
                    }
                    self.signatures.insert(
                        key,
                        Arc::new(Signature {
                            native,
                            type_params: vec![],
                            type_rules: None,
                            params: forwarded.clone(),
                            ret: Some(ty),
                            ret_view: Some(view),
                            array_result: None,
                        }),
                    );
                    if let Some((init, component)) = &init {
                        let native = self.native_name(component);
                        if let Some(arity) = arity {
                            self.arities.insert(native, arity);
                        }
                        let this = self.this_parameter(view);
                        self.signatures.insert(
                            *init,
                            Arc::new(Signature {
                                native,
                                type_params: vec![],
                                type_rules: None,
                                params: std::iter::once(this).chain(forwarded).collect(),
                                ret: Some(unit()),
                                ret_view: None,
                                array_result: None,
                            }),
                        );
                    }
                }
            }
            for (key, component, f, role) in bodies {
                self.signature_as(f, key, &component, role)?;
            }
            self.check_overrides(view)?;
        }
        Ok(())
    }

    /// An overriding method must be assignable to the one it overrides: TypeScript compares
    /// methods bivariantly, so distinct primitive parameter or result types never agree.
    fn check_overrides(&self, view: InternedString) -> Result<()> {
        let class = &self.classes[&view];
        let Some(base) = class.base else {
            return Ok(());
        };
        let distinct = |a: &Type, b: &Type| {
            matches!((a, b), (Type::Primitive(_), Type::Primitive(_))) && a != b
        };
        // Own methods in source order, so the first incompatible override is reported.
        for (key, _, f, _) in &class.bodies {
            let Some((name, _)) = class.methods.iter().find(|(_, m)| m.key == *key) else {
                continue;
            };
            let Some(inherited) = self.classes[&base].methods.get(name) else {
                continue;
            };
            let (own, theirs) = (&self.signatures[key], &self.signatures[&inherited.key]);
            let parameters = own.params[1..]
                .iter()
                .zip(&theirs.params[1..])
                .any(|(a, b)| distinct(&a.ty, &b.ty));
            let result = match (&own.ret, &theirs.ret) {
                (Some(a), Some(b)) if *b != unit() => distinct(a, b) || *a == unit(),
                _ => false,
            };
            if parameters || result {
                let at = f.span;
                return self.error(
                    ErrorKind::Type,
                    format!("property `{name}` in a derived class is not assignable to the same property in its base class"),
                    at,
                );
            }
        }
        Ok(())
    }

    fn native_name(&mut self, component: &str) -> InternedString {
        self.name_buffer.truncate(self.prefix_len);
        self.name_buffer.push_str("fn$");
        crate::modules::append_component(&mut self.name_buffer, component);
        intern(&self.name_buffer)
    }

    pub(super) fn lower_classes(&mut self) -> Result<()> {
        for view in self.class_order.clone() {
            let class = &self.classes[&view];
            let constructor = class.constructor;
            let key = class.constructor_key;
            let bodies = class.bodies.clone();
            let init = class.init.clone();
            match constructor {
                Some(f) => {
                    self.function_as(f, key, FunctionRole::Constructor(view))?;
                    if let Some((init, _)) = init {
                        self.function_as(f, init, FunctionRole::Initializer(view))?;
                    }
                }
                None => {
                    self.default_constructor(view, false)?;
                    if init.is_some() {
                        self.default_constructor(view, true)?;
                    }
                }
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
        let this = self.allocate_instance(view, f, statements);
        if self.classes[&view].base.is_none() {
            let initialized = self.initialize_fields(view, f, this)?;
            statements.extend(initialized);
        }
        Ok(this)
    }

    /// Binds `this` to a new instance whose fields are zero and whose class field names `view`.
    fn allocate_instance(
        &mut self,
        view: InternedString,
        f: Option<&js::Function<'ast>>,
        statements: &mut Vec<Stmt>,
    ) -> InternedString {
        let class = &self.classes[&view];
        let ty = class.ty.clone();
        let tag = class.tag;
        let at = f.map_or(class.span, |f| f.span);
        let Type::Named { id, .. } = &ty else {
            unreachable!()
        };
        let struct_name = self.registry.get_type_by_id(*id).unwrap().name;
        let class_tag = self.name(CLASS_TAG);
        // Struct literals initialize fields in layout order.
        let zeros: Vec<TypedFieldInit> = self
            .layout_key(&ty)
            .into_iter()
            .map(|(name, ty)| TypedFieldInit {
                name,
                value: Box::new(match ty {
                    _ if name == class_tag => num(tag.unwrap(), at),
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
        self.bind_instance(view, this);
        this
    }

    fn bind_instance(&mut self, view: InternedString, this: InternedString) {
        let ty = self.classes[&view].ty.clone();
        let name = self.name("this");
        self.scopes.last_mut().unwrap().insert(
            name,
            Binding {
                name: this,
                symbol: None,
                ty,
                view: Some(view),
                mutable: false,
                initialized: true,
            },
        );
    }

    /// Parameter properties, then the class's own field initializers, in order.
    pub(super) fn initialize_fields(
        &mut self,
        view: InternedString,
        f: Option<&js::Function<'ast>>,
        this: InternedString,
    ) -> Result<Vec<Stmt>> {
        let ty = self.classes[&view].ty.clone();
        let mut fields: Vec<_> = self.classes[&view]
            .fields
            .iter()
            .filter(|f| f.owner.is_none())
            .cloned()
            .collect();
        // Initializers run in source order; a redeclared field keeps its inherited position.
        fields.sort_by_key(|f| f.span.start);
        let mut statements = vec![];
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
        let parameters: FxHashSet<&str> = fields
            .iter()
            .filter(|f| f.parameter)
            .map(|f| f.name)
            .collect();
        for field in fields.iter().filter(|f| !f.parameter) {
            let Some(initializer) = field.initializer else {
                continue;
            };
            // Parameter properties are assigned before or after field initializers depending on
            // the class field semantics the target selects.
            let mut reads = ThisReads {
                names: &parameters,
                found: false,
            };
            reads.visit_expression(initializer);
            if reads.found {
                return self.unsupported(
                    "field initializers reading parameter properties",
                    field.span,
                );
            }
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
        Ok(statements)
    }

    /// `super(...)` in the constructor of `view`: the base initializer on this instance, then
    /// this class's parameter properties and field initializers.
    pub(super) fn super_call(
        &mut self,
        view: InternedString,
        f: Option<&js::Function<'ast>>,
        arguments: &[js::Argument<'ast>],
        at: oxc_span::Span,
    ) -> Result<Vec<Stmt>> {
        let base = self.classes[&view].base.unwrap();
        let (init, _) = self.classes[&base].init.clone().unwrap();
        let this = self.binding("this", at)?.name;
        let receiver = self.upcast(var(this, self.classes[&view].ty.clone(), at), base, at);
        let sig = &self.signatures[&init];
        let stripped = Arc::new(Signature {
            params: sig.params[1..].to_vec(),
            ..(**sig).clone()
        });
        let call = self.call_with_signature(stripped, arguments, None, Some(receiver), at)?;
        let mut statements = vec![expression(call)];
        statements.extend(self.initialize_fields(view, f, this)?);
        Ok(statements)
    }

    /// The statements of a derived class's constructor: those before `super(...)` cannot use
    /// `this`; the call runs the base initializer and then this class's field initializers.
    pub(super) fn derived_constructor_body(
        &mut self,
        view: InternedString,
        f: &js::Function<'ast>,
        body: &[js::Statement<'ast>],
        statements: &mut Vec<Stmt>,
    ) -> Result<()> {
        let position = body.iter().position(|s| {
            matches!(s, js::Statement::ExpressionStatement(e)
                if matches!(&e.expression, js::Expression::CallExpression(c)
                    if matches!(c.callee, js::Expression::Super(_))))
        });
        let Some(position) = position else {
            let mut nested = SuperCalls { found: false };
            for s in body {
                nested.visit_statement(s);
            }
            if nested.found {
                return self.unsupported("super calls inside nested statements", f.span);
            }
            return self.error(
                ErrorKind::Type,
                "constructors for derived classes must contain a 'super' call",
                f.span,
            );
        };
        let outer = std::mem::replace(&mut self.before_super, true);
        let before: Result<Vec<_>> = body[..position].iter().map(|s| self.statement(s)).collect();
        self.before_super = outer;
        statements.extend(before?.into_iter().flatten());
        let js::Statement::ExpressionStatement(e) = &body[position] else {
            unreachable!()
        };
        let js::Expression::CallExpression(call) = &e.expression else {
            unreachable!()
        };
        statements.extend(self.super_call(view, Some(f), &call.arguments, call.span)?);
        for s in &body[position + 1..] {
            statements.extend(self.statement(s)?);
        }
        Ok(())
    }

    /// `super(...)` anywhere but a derived constructor's top-level statements.
    pub(super) fn misplaced_super_call(
        &self,
        keyword: oxc_span::Span,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        let derived = self
            .current_class
            .and_then(|view| self.classes.get(&view))
            .is_some_and(|class| class.base.is_some());
        if !derived {
            return self.error(
                ErrorKind::Type,
                "'super' can only be referenced in a derived class",
                keyword,
            );
        }
        if !self.in_constructor {
            return self.error(
                ErrorKind::Type,
                "super calls are not permitted outside constructors",
                at,
            );
        }
        self.unsupported(
            "super calls outside a constructor's top-level statements",
            at,
        )
    }

    /// `super.name` outside a call: fields are not reachable through `super`.
    pub(super) fn super_member(&mut self, m: &js::StaticMemberExpression<'ast>) -> Result<Expr> {
        let name = m.property.name.as_str();
        if self.static_this.is_some() {
            if let Some(read) = self.static_field_read(m)? {
                return Ok(read);
            }
            return self.unsupported("super property values in static code", m.span);
        }
        let base = self
            .current_class
            .and_then(|view| self.classes.get(&view))
            .and_then(|class| class.base);
        let Some(base) = base else {
            return self.error(
                ErrorKind::Type,
                "'super' can only be referenced in a derived class",
                m.object.span(),
            );
        };
        if self.classes[&base].fields.iter().any(|f| f.name == name) {
            return self.error(
                ErrorKind::Type,
                format!("class field `{name}` defined by the parent class is not accessible in the child class via super"),
                m.span,
            );
        }
        if !self.classes[&base].methods.contains_key(name) {
            return self.error(
                ErrorKind::Type,
                format!("property `{name}` does not exist on the base class"),
                m.span,
            );
        }
        self.unsupported("super property values", m.span)
    }

    /// Records names used as `extends` targets and the module's top-level values, before
    /// any class is declared.
    pub(super) fn class_heritage(&mut self, program: &js::Program<'ast>) {
        for statement in &program.body {
            if let js::Statement::ImportDeclaration(import) = statement {
                for specifier in import.specifiers.iter().flatten() {
                    let local = self.name(specifier.local().name.as_str());
                    self.top_level_values.insert(local);
                }
                continue;
            }
            let Some(declaration) = modules::declaration(statement) else {
                continue;
            };
            match declaration {
                js::Declaration::ClassDeclaration(class) => {
                    if let Some(js::ClassHeritage {
                        expression: js::Expression::Identifier(base),
                        ..
                    }) = &class.heritage
                    {
                        let base = self.name(base.name.as_str());
                        self.extended_classes.insert(base);
                    }
                }
                js::Declaration::FunctionDeclaration(function) => {
                    if let Some(id) = &function.id {
                        let name = self.name(id.name.as_str());
                        self.top_level_values.insert(name);
                    }
                }
                js::Declaration::VariableDeclaration(variables) => {
                    for d in &variables.declarations {
                        if let js::BindingPattern::BindingIdentifier(id) = &d.id {
                            let name = self.name(id.name.as_str());
                            self.top_level_values.insert(name);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// TypeScript's strict property initialization: every field without an initializer
    /// is assigned in the constructor.
    pub(super) fn check_field_initialization(
        &self,
        view: InternedString,
        constructor: &js::Function<'ast>,
    ) -> Result<()> {
        let mut assigned = ThisAssignments {
            source: self.source_text,
            names: FxHashSet::default(),
        };
        assigned.visit_function_body(constructor.body.as_ref().unwrap());
        self.require_initialized(view, &assigned.names)
    }

    fn require_initialized(&self, view: InternedString, assigned: &FxHashSet<&str>) -> Result<()> {
        for field in &self.classes[&view].fields {
            if field.owner.is_none()
                && field.initializer.is_none()
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

    /// The constructor of a class that declares none: `new`, or the initializer subclasses
    /// call. A subclass's implicit constructor calls its base's with no arguments.
    pub(super) fn default_constructor(
        &mut self,
        view: InternedString,
        initializer: bool,
    ) -> Result<()> {
        let key = if initializer {
            self.classes[&view].init.as_ref().unwrap().0
        } else {
            self.classes[&view].constructor_key
        };
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
        let outer_body = self.class_body.replace(view);
        let outer_constructor = std::mem::replace(&mut self.in_constructor, true);
        let outer_returns = self.returns.take();
        let mut statements = vec![];
        let mut params = vec![];
        let result = (|| {
            let this = if initializer {
                let parameter = self.bind_this(view, at);
                let this = parameter.name;
                params.push(parameter);
                this
            } else {
                self.allocate_instance(view, None, &mut statements)
            };
            let mut arguments = vec![];
            for forwarded in &sig.params[usize::from(initializer)..] {
                let unique = self.fresh("argument");
                let mut parameter = TypedParameter::regular(
                    unique,
                    forwarded.ty.clone(),
                    Mutability::Mutable,
                    span(at),
                );
                parameter.ownership = ParamOwnership::Borrowed;
                params.push(parameter);
                arguments.push(var(unique, forwarded.ty.clone(), at));
            }
            let initialized = match self.classes[&view].base {
                Some(base) => {
                    let (init, _) = self.classes[&base].init.clone().unwrap();
                    let native = self.signatures[&init].native;
                    let receiver =
                        self.upcast(var(this, self.classes[&view].ty.clone(), at), base, at);
                    let call = node(
                        TypedExpression::Call(TypedCall {
                            callee: Box::new(var(native, Type::Any, at)),
                            positional_args: std::iter::once(receiver).chain(arguments).collect(),
                            named_args: vec![],
                            type_args: vec![],
                        }),
                        unit(),
                        at,
                    );
                    let mut statements = vec![expression(call)];
                    statements.extend(self.initialize_fields(view, None, this)?);
                    statements
                }
                None => self.initialize_fields(view, None, this)?,
            };
            statements.extend(initialized);
            Ok(this)
        })();
        self.returns = outer_returns;
        self.in_constructor = outer_constructor;
        self.current_class = outer_class;
        self.class_body = outer_body;
        self.vars = outer_vars;
        self.pop_scope();
        let this = result?;
        let ty = sig.ret.clone().unwrap();
        if !initializer {
            statements.push(stmt(
                TypedStatement::Return(Some(Box::new(var(this, ty.clone(), at)))),
                at,
            ));
        }
        self.declarations.push(node(
            TypedDeclaration::Function(TypedFunction {
                name: sig.native,
                annotations: vec![strict_fp(self.name("strict_fp"))],
                params,
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

    /// A private member is used only in its class's body; a protected one also in subclasses.
    pub(super) fn check_member_access(
        &self,
        owner: InternedString,
        access: Access,
        name: &str,
        at: oxc_span::Span,
    ) -> Result<()> {
        let allowed = match access {
            Access::Public => true,
            Access::Private => self.class_body == Some(owner),
            Access::Protected => self
                .class_body
                .is_some_and(|class| self.is_subclass(class, owner)),
        };
        if !allowed {
            let kind = if access == Access::Private {
                "private and only accessible within its class"
            } else {
                "protected and only accessible within its class and subclasses"
            };
            return self.error(ErrorKind::Type, format!("property `{name}` is {kind}"), at);
        }
        Ok(())
    }

    /// Whether `class` is `ancestor` or extends it.
    pub(super) fn is_subclass(&self, class: InternedString, ancestor: InternedString) -> bool {
        let mut current = Some(class);
        while let Some(view) = current {
            if view == ancestor {
                return true;
            }
            current = self.classes.get(&view).and_then(|c| c.base);
        }
        false
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
        let Some(declared) = self.classes[&view].fields.iter().find(|f| f.name == field) else {
            return Ok(());
        };
        self.check_member_access(declared.owner.unwrap_or(view), declared.access, field, at)
    }

    pub(super) fn readonly_field_write_allowed(&self, object: &js::Expression<'ast>) -> bool {
        self.in_constructor && matches!(object, js::Expression::ThisExpression(_))
    }

    /// `receiver.method(...)`, `super.method(...)` and `Class.method(...)`; `None` when not a
    /// class member call.
    pub(super) fn method_call(
        &mut self,
        c: &js::CallExpression<'ast>,
        m: &js::StaticMemberExpression<'ast>,
    ) -> Result<Option<Expr>> {
        self.member_method_call(c, &m.object, m.property.name.as_str(), m.span)
    }

    /// `object.name(...)` for a class member `name`, public or private.
    pub(super) fn member_method_call(
        &mut self,
        c: &js::CallExpression<'ast>,
        object: &js::Expression<'ast>,
        name: &str,
        at: oxc_span::Span,
    ) -> Result<Option<Expr>> {
        if let Some((view, through_this)) = self.static_receiver(object) {
            let Some(member) = self.classes[&view].statics.get(name) else {
                if self.classes[&view]
                    .static_fields
                    .iter()
                    .any(|f| f.name == name)
                {
                    return Ok(None);
                }
                return self.error(
                    ErrorKind::Type,
                    format!("property `{name}` does not exist on this class"),
                    at,
                );
            };
            // In a static method `this` is whichever class the call names.
            if through_this
                && self.class_order.iter().any(|other| {
                    *other != view
                        && self.is_subclass(*other, view)
                        && self.classes[other].statics[name].owner.is_none()
                })
            {
                return self.unsupported("static this with methods overridden in subclasses", at);
            }
            let (key, access, owner) = (member.key, member.access, member.owner.unwrap_or(view));
            self.check_member_access(owner, access, name, at)?;
            let sig = self.signatures[&key].clone();
            return self
                .call_with_signature(sig, &c.arguments, c.type_arguments.as_deref(), None, c.span)
                .map(Some);
        }
        if let js::Expression::Super(s) = object {
            return self.super_method_call(c, name, s.span).map(Some);
        }
        let Some(view) = self
            .view_of(object)
            .filter(|v| self.classes.contains_key(v))
        else {
            return Ok(None);
        };
        let class = &self.classes[&view];
        let Some(member) = class.methods.get(name) else {
            if class.fields.iter().any(|f| f.name == name) {
                return Ok(None);
            }
            // Errors in the receiver come first, as TypeScript reports them in source order.
            self.expr(object)?;
            return self.error(
                ErrorKind::Type,
                format!("property `{name}` does not exist on this class"),
                at,
            );
        };
        let (key, access, owner) = (member.key, member.access, member.owner.unwrap_or(view));
        self.check_member_access(owner, access, name, at)?;
        let receiver = self.expr(object)?;
        let call = self.member_call(key, owner, receiver, c)?;
        let call = self.dispatch(view, name, key, call, c.span)?;
        // A method whose result is inferred from `return this` has TypeScript's polymorphic
        // `this` type: the result is the receiver, seen through the receiver's class.
        if owner != view && self.returns_this(key) {
            return Ok(Some(self.downcast(call, view, c.span)));
        }
        if owner != view
            && self.functions[&key].return_type.is_none()
            && self.signatures[&key].ret_view == Some(owner)
        {
            return self.unsupported("inferred class results through subclasses", at);
        }
        Ok(Some(call))
    }

    /// `super.method(...)`: the base class's implementation, called on `this`.
    fn super_method_call(
        &mut self,
        c: &js::CallExpression<'ast>,
        name: &str,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        let Some(base) = self
            .current_class
            .and_then(|view| self.classes.get(&view))
            .and_then(|class| class.base)
        else {
            return self.error(
                ErrorKind::Type,
                "'super' can only be referenced in a derived class",
                at,
            );
        };
        let Some(member) = self.classes[&base].methods.get(name) else {
            if self.classes[&base].fields.iter().any(|f| f.name == name) {
                return self.error(
                    ErrorKind::Type,
                    format!("class field `{name}` defined by the parent class is not accessible in the child class via super"),
                    at,
                );
            }
            return self.error(
                ErrorKind::Type,
                format!("property `{name}` does not exist on the base class"),
                at,
            );
        };
        let (key, access, owner) = (member.key, member.access, member.owner.unwrap_or(base));
        self.check_member_access(owner, access, name, at)?;
        let this = self.this_expr(at)?;
        self.member_call(key, owner, this, c)
    }

    /// A call of the method `key` that `owner` declares, on a receiver of `owner` or a subclass.
    fn member_call(
        &mut self,
        key: InternedString,
        owner: InternedString,
        receiver: Expr,
        c: &js::CallExpression<'ast>,
    ) -> Result<Expr> {
        let receiver = self.upcast(receiver, owner, c.span);
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
    }

    /// Whether the method `key` infers its result from `return this` on every path.
    pub(super) fn returns_this(&self, key: InternedString) -> bool {
        let f = self.functions[&key];
        if f.return_type.is_some() {
            return false;
        }
        let mut returns = ThisReturns { this: 0, other: 0 };
        returns.visit_function_body(f.body.as_ref().unwrap());
        returns.this > 0 && returns.other == 0
    }

    /// The receiver returned by a polymorphic-`this` method, as an instance of `class`.
    fn downcast(&self, value: Expr, class: InternedString, at: oxc_span::Span) -> Expr {
        let ty = self.classes[&class].ty.clone();
        if value.ty == ty {
            return value;
        }
        node(
            TypedExpression::Cast(TypedCast {
                expr: Box::new(value),
                target_type: ty.clone(),
            }),
            ty,
            at,
        )
    }

    /// A subclass instance seen through the layout of `class`, which is a prefix of its own.
    pub(super) fn upcast(&self, value: Expr, class: InternedString, at: oxc_span::Span) -> Expr {
        let ty = self.classes[&class].ty.clone();
        if value.ty == ty {
            return value;
        }
        node(
            TypedExpression::Cast(TypedCast {
                expr: Box::new(value),
                target_type: ty.clone(),
            }),
            ty,
            at,
        )
    }

    /// A call of `name` through a receiver of class `view` runs the implementation of the
    /// instance's class: `call` is the static call of `key`, and each subclass overriding it
    /// is selected by the instance's class field.
    fn dispatch(
        &mut self,
        view: InternedString,
        name: &str,
        key: InternedString,
        call: Expr,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        let mut overrides: Vec<(InternedString, InternedString, Vec<f64>)> = vec![];
        for subclass in self.class_order.clone() {
            if subclass == view || !self.is_subclass(subclass, view) {
                continue;
            }
            let class = &self.classes[&subclass];
            let member = &class.methods[name];
            if member.key == key {
                continue;
            }
            let tag = class.tag.expect("subclasses carry their class");
            let owner = member.owner.unwrap_or(subclass);
            match overrides.iter_mut().find(|(k, _, _)| *k == member.key) {
                Some((_, _, tags)) => tags.push(tag),
                None => overrides.push((member.key, owner, vec![tag])),
            }
        }
        if overrides.is_empty() {
            return Ok(call);
        }
        let base_sig = self.signatures[&key].clone();
        for (implementation, _, _) in &overrides {
            let sig = &self.signatures[implementation];
            let same = sig.params.len() == base_sig.params.len()
                && sig.params[1..]
                    .iter()
                    .zip(&base_sig.params[1..])
                    .all(|(a, b)| a.ty == b.ty)
                && sig.ret.is_some()
                && sig.ret == base_sig.ret
                && sig.type_params.is_empty()
                && base_sig.type_params.is_empty()
                && self.arities.get(&sig.native) == self.arities.get(&base_sig.native);
            if !same {
                return self.unsupported("overrides with a different signature", at);
            }
        }
        let ty = call.ty.clone();
        let TypedExpression::Call(TypedCall {
            positional_args, ..
        }) = call.node
        else {
            unreachable!("member calls are direct calls")
        };
        // Evaluate the receiver and arguments once, in order, then select the implementation.
        let mut statements = vec![];
        let mut values = vec![];
        for argument in positional_args {
            let temporary = self.fresh("dispatch");
            let argument_type = argument.ty.clone();
            statements.push(stmt(
                TypedStatement::Let(TypedLet {
                    name: temporary,
                    ty: argument_type.clone(),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(argument)),
                    span: span(at),
                }),
                at,
            ));
            values.push(var(temporary, argument_type, at));
        }
        let receiver = values[0].clone();
        let tag = node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(receiver.clone()),
                field: self.name(CLASS_TAG),
            }),
            number(),
            at,
        );
        let invoke = |native: InternedString, receiver: Expr| {
            node(
                TypedExpression::Call(TypedCall {
                    callee: Box::new(var(native, Type::Any, at)),
                    positional_args: std::iter::once(receiver)
                        .chain(values[1..].iter().cloned())
                        .collect(),
                    named_args: vec![],
                    type_args: vec![],
                }),
                ty.clone(),
                at,
            )
        };
        let mut selected = invoke(base_sig.native, receiver.clone());
        for (implementation, owner, tags) in overrides.into_iter().rev() {
            let condition = tags
                .into_iter()
                .map(|value| binary(BinaryOp::Eq, tag.clone(), num(value, at), boolean(), at))
                .reduce(|a, b| binary(BinaryOp::Or, a, b, boolean(), at))
                .unwrap();
            let native = self.signatures[&implementation].native;
            let receiver = node(
                TypedExpression::Cast(TypedCast {
                    expr: Box::new(receiver.clone()),
                    target_type: self.classes[&owner].ty.clone(),
                }),
                self.classes[&owner].ty.clone(),
                at,
            );
            selected = node(
                TypedExpression::If(TypedIfExpr {
                    condition: Box::new(condition),
                    then_branch: Box::new(invoke(native, receiver)),
                    else_branch: Box::new(selected),
                }),
                ty.clone(),
                at,
            );
        }
        statements.push(expression(selected));
        Ok(node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(at),
            }),
            ty,
            at,
        ))
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
        if member.owner.is_some() && self.returns_this(member.key) {
            return Some(view);
        }
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
        let allowed = match self.classes[&view].constructor_access {
            Access::Public => true,
            Access::Private => self.class_body == Some(view),
            Access::Protected => self
                .class_body
                .is_some_and(|class| self.is_subclass(class, view)),
        };
        if !allowed {
            return self.error(
                ErrorKind::Type,
                format!(
                    "constructor of class `{}` is not accessible outside the class declaration",
                    id.name
                ),
                n.span,
            );
        }
        let sig = self.signatures[&self.classes[&view].constructor_key].clone();
        self.call_with_signature(sig, &n.arguments, None, None, n.span)
    }

    pub(super) fn this_expr(&mut self, at: oxc_span::Span) -> Result<Expr> {
        let name = self.name("this");
        if self.before_super {
            return self.error(
                ErrorKind::Type,
                "'super' must be called before accessing 'this' in the constructor of a derived class",
                at,
            );
        }
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
            let nominal = self.views.get(&to).is_some_and(|v| v.nominal);
            let hierarchy = self.classes.get(&to).is_some_and(|c| c.tag.is_some());
            let instance = from.is_some_and(|from| self.is_subclass(from, to));
            if nominal && !instance {
                return self.error(
                    ErrorKind::Type,
                    "value is missing members required by the class type",
                    at,
                );
            }
            // Instances of a hierarchy carry their class; a lookalike record does not.
            if hierarchy && !instance {
                return self.unsupported("structural values of classes with subclasses", at);
            }
        }
        Ok(())
    }
}
