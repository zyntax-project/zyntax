//! Static class members. Each static field is a module global that the class declaration
//! initializes where it runs in the module body; static blocks run there too. In static code
//! `this` names the class, and `super` its base.

use super::classes::{Access, Member};
use super::*;

#[derive(Clone)]
pub(super) struct StaticField<'ast, 'program> {
    pub name: &'ast str,
    pub ty: PrimitiveType,
    pub readonly: bool,
    pub access: Access,
    /// The class that declares an inherited static field; `None` for the class's own.
    pub owner: Option<InternedString>,
    pub initializer: Option<&'program js::Expression<'ast>>,
    pub global: InternedString,
    /// Set by the first assignment of a field declared without a value.
    pub assigned: Option<InternedString>,
    pub span: oxc_span::Span,
}

/// A class's own static field initializers and blocks, in class body order.
#[derive(Clone, Copy)]
pub(super) enum StaticElement<'ast, 'program> {
    Field(&'ast str),
    Block(&'program js::StaticBlock<'ast>),
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    /// A static field declaration of the class `source`.
    pub(super) fn declare_static_field(
        &mut self,
        source: &str,
        p: &'program js::PropertyDefinition<'ast>,
        fields: &mut Vec<StaticField<'ast, 'program>>,
        statics: &FxHashMap<&'ast str, Member>,
    ) -> Result<&'ast str> {
        if p.computed || p.declare || p.optional || p.r#override || !p.decorators.is_empty() {
            return self.unsupported("this class field form", p.span);
        }
        let field = self.class_key(&p.key)?;
        let inherited = fields
            .iter()
            .position(|f| f.name == field && f.owner.is_some());
        // A private name is a new member in each class that declares it.
        if field.starts_with('#')
            && (inherited.is_some() || statics.get(field).is_some_and(|m| m.owner.is_some()))
        {
            return self.unsupported("private names redeclared in subclasses", p.span);
        }
        let ty = match (&p.type_annotation, &p.value) {
            (Some(annotation), _) => {
                let (ty, view) = self.ty(&annotation.type_annotation)?;
                self.plain_enum(view, "enum-typed fields", p.span)?;
                let Some(ty) = field_primitive(&ty) else {
                    return self.unsupported("reference-valued object fields", p.span);
                };
                ty
            }
            (None, Some(_)) if let Some(index) = inherited => fields[index].ty,
            (None, Some(init)) => match self.static_type(init, source, fields) {
                Some(ty) => ty,
                None => return self.unsupported("class field type inference", p.span),
            },
            (None, None) => return self.unsupported("class field type inference", p.span),
        };
        if fields.iter().any(|f| f.name == field && f.owner.is_none())
            || statics.get(field).is_some_and(|m| m.owner.is_none())
        {
            return self.error(ErrorKind::Type, "duplicate identifier", p.span);
        }
        let access = if field.starts_with('#') {
            Access::Private
        } else {
            classes::access(p.accessibility)
        };
        if let Some(index) = inherited {
            let base = &fields[index];
            let narrower = match base.access {
                Access::Private => true,
                Access::Protected => access == Access::Private,
                Access::Public => access != Access::Public,
            };
            if base.ty != ty || narrower {
                return self.error(
                    ErrorKind::Type,
                    format!("class `{source}` static side incorrectly extends its base class static side: property `{field}`"),
                    p.span,
                );
            }
            fields.remove(index);
        } else if statics.get(field).is_some_and(|m| m.owner.is_some()) {
            return self.error(
                ErrorKind::Type,
                format!("class `{source}` static side incorrectly extends its base class static side: property `{field}`"),
                p.span,
            );
        }
        let global = self.static_global_name("static$", source, field);
        let assigned = p
            .value
            .is_none()
            .then(|| self.static_global_name("assigned$static$", source, field));
        fields.push(StaticField {
            name: field,
            ty,
            readonly: p.readonly,
            access,
            owner: None,
            initializer: p.value.as_ref(),
            global,
            assigned,
            span: p.span,
        });
        Ok(field)
    }

    /// The type of a static initializer before any body is lowered: literals, operators on
    /// them, and the static fields of this class or another declared before it.
    fn static_type(
        &self,
        e: &js::Expression<'ast>,
        source: &str,
        fields: &[StaticField<'ast, 'program>],
    ) -> Option<PrimitiveType> {
        match e {
            js::Expression::NumericLiteral(_) => Some(PrimitiveType::F64),
            js::Expression::BooleanLiteral(_) => Some(PrimitiveType::Bool),
            js::Expression::StringLiteral(s) if !s.lone_surrogates => Some(PrimitiveType::String),
            js::Expression::TemplateLiteral(_) => Some(PrimitiveType::String),
            js::Expression::ParenthesizedExpression(p) => {
                self.static_type(&p.expression, source, fields)
            }
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
                "+" => {
                    let left = self.static_type(&b.left, source, fields)?;
                    let right = self.static_type(&b.right, source, fields)?;
                    match (left, right) {
                        (PrimitiveType::String, _) | (_, PrimitiveType::String) => {
                            Some(PrimitiveType::String)
                        }
                        (PrimitiveType::F64, PrimitiveType::F64) => Some(PrimitiveType::F64),
                        _ => None,
                    }
                }
                _ => None,
            },
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                let name = m.property.name.as_str();
                let own = match &m.object {
                    js::Expression::ThisExpression(_) => true,
                    js::Expression::Super(_) => {
                        return fields
                            .iter()
                            .find(|f| f.name == name && f.owner.is_some())
                            .map(|f| f.ty);
                    }
                    js::Expression::Identifier(id) => id.name == source,
                    _ => return None,
                };
                if own {
                    return fields.iter().find(|f| f.name == name).map(|f| f.ty);
                }
                let js::Expression::Identifier(id) = &m.object else {
                    return None;
                };
                let view = self.class_names.get(&self.name(id.name.as_str()))?;
                self.classes
                    .get(view)?
                    .static_fields
                    .iter()
                    .find(|f| f.name == name)
                    .map(|f| f.ty)
            }
            _ => None,
        }
    }

    fn static_global_name(&mut self, kind: &str, class: &str, field: &str) -> InternedString {
        self.name_buffer.truncate(self.prefix_len);
        self.name_buffer.push_str(kind);
        crate::modules::append_component(&mut self.name_buffer, &format!("{class}.{field}"));
        intern(&self.name_buffer)
    }

    /// The module globals holding the class's own static fields, and the flag its
    /// declaration sets when it runs.
    pub(super) fn declare_static_globals(&mut self, view: InternedString, source: &'ast str) {
        let class = &self.classes[&view];
        let own: Vec<_> = class
            .static_fields
            .iter()
            .filter(|f| f.owner.is_none())
            .cloned()
            .collect();
        if own.is_empty() && class.static_elements.is_empty() {
            return;
        }
        let ready = self.static_global_name("ready$class$", source, "");
        let mut variables = vec![(ready, boolean())];
        for field in &own {
            variables.push((field.global, Type::Primitive(field.ty)));
            variables.extend(field.assigned.map(|flag| (flag, boolean())));
            self.globals.insert(
                field.global,
                globals::Global {
                    ready: Some(ready),
                    assigned: field.assigned,
                    source: self.name(source),
                },
            );
        }
        for (name, ty) in variables {
            self.declarations.push(node(
                TypedDeclaration::Variable(TypedVariable {
                    name,
                    ty,
                    mutability: Mutability::Mutable,
                    initializer: None,
                    visibility: Visibility::Public,
                }),
                unit(),
                self.classes[&view].span,
            ));
        }
        self.classes.get_mut(&view).unwrap().ready = Some(ready);
    }

    /// The class a static member access names: `C.x`, or `this.x` and `super.x` in static code.
    pub(super) fn static_receiver(
        &self,
        object: &js::Expression<'ast>,
    ) -> Option<(InternedString, bool)> {
        match object {
            js::Expression::Identifier(id) => {
                let key = self.name(id.name.as_str());
                if self.scopes.iter().any(|s| s.contains_key(&key)) {
                    return None;
                }
                self.class_names
                    .get(&key)
                    .copied()
                    .filter(|view| self.classes.contains_key(view))
                    .map(|view| (view, false))
            }
            js::Expression::ThisExpression(_) => self.static_this.map(|view| (view, true)),
            js::Expression::Super(_) => self
                .static_this
                .and_then(|view| self.classes[&view].base)
                .map(|base| (base, false)),
            js::Expression::ParenthesizedExpression(p) => self.static_receiver(&p.expression),
            _ => None,
        }
    }

    /// The static field `name` of the class a receiver names, checked for access.
    fn static_field(
        &self,
        object: &js::Expression<'ast>,
        name: &str,
        at: oxc_span::Span,
    ) -> Result<Option<(InternedString, StaticField<'ast, 'program>)>> {
        let Some((view, through_this)) = self.static_receiver(object) else {
            return Ok(None);
        };
        let Some(field) = self.classes[&view]
            .static_fields
            .iter()
            .find(|f| f.name == name)
            .cloned()
        else {
            return Ok(None);
        };
        self.check_member_access(field.owner.unwrap_or(view), field.access, name, at)?;
        if self.classes[&field.owner.unwrap_or(view)].ambient {
            return self.unsupported("values of ambient classes", at);
        }
        if let Some((class, pending)) = &self.static_pending
            && *class == view
            && field.owner.is_none()
            && pending.contains(&field.name)
        {
            return self.error(
                ErrorKind::Type,
                format!("property `{name}` is used before its initialization"),
                at,
            );
        }
        if self.top_level && !self.evaluated_classes.contains(&view) {
            return self.error(
                ErrorKind::Type,
                "class used before its declaration",
                object.span(),
            );
        }
        // In a static method `this` is whichever class the call names.
        if through_this
            && self.class_order.iter().any(|other| {
                *other != view
                    && self.is_subclass(*other, view)
                    && self.classes[other]
                        .static_fields
                        .iter()
                        .any(|f| f.name == name && f.owner.is_none())
            })
        {
            return self.unsupported("static this with fields redeclared in subclasses", at);
        }
        Ok(Some((view, field)))
    }

    /// `C.x` as a value.
    pub(super) fn static_field_read(
        &mut self,
        m: &js::StaticMemberExpression<'ast>,
    ) -> Result<Option<Expr>> {
        self.static_member_read(&m.object, m.property.name.as_str(), m.span)
    }

    /// `object.name` as a value, when the object names a class with that static field.
    pub(super) fn static_member_read(
        &mut self,
        object: &js::Expression<'ast>,
        name: &str,
        at: oxc_span::Span,
    ) -> Result<Option<Expr>> {
        let Some((_, field)) = self.static_field(object, name, at)? else {
            return Ok(None);
        };
        let ty = Type::Primitive(field.ty);
        let global = self.globals.get(&field.global);
        let mut checks = vec![];
        if !self.top_level
            && let Some(ready) = global.and_then(|g| g.ready)
        {
            checks.push((ready, "$TS$binding_check"));
        }
        if let Some(assigned) = field.assigned {
            checks.push((assigned, "$TS$assigned_check"));
        }
        let source = global.map_or_else(|| intern(name), |g| g.source);
        let mut statements: Vec<Stmt> = checks
            .into_iter()
            .map(|(flag, check)| {
                expression(call(
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
                ))
            })
            .collect();
        let value = var(field.global, ty.clone(), at);
        if statements.is_empty() {
            return Ok(Some(value));
        }
        statements.push(expression(value));
        Ok(Some(node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(at),
            }),
            ty,
            at,
        )))
    }

    /// `C.x` as an assignment target: the field's global.
    pub(super) fn static_field_target(
        &mut self,
        object: &js::Expression<'ast>,
        name: &str,
        at: oxc_span::Span,
    ) -> Result<Option<Expr>> {
        let Some((view, field)) = self.static_field(object, name, at)? else {
            return Ok(None);
        };
        if field.readonly {
            return self.error(ErrorKind::Type, "cannot assign to readonly property", at);
        }
        // Assigning through a subclass defines the subclass's own property.
        if field.owner.is_some_and(|owner| owner != view) {
            return self.unsupported("assignments to inherited static fields", at);
        }
        if field.ty == PrimitiveType::String {
            return self.unsupported("string static fields assigned after their declaration", at);
        }
        Ok(Some(var(field.global, Type::Primitive(field.ty), at)))
    }

    /// The module-body statements of a class declaration: its static field initializers and
    /// static blocks in order, then the flag that the class is ready.
    pub(super) fn run_class_declaration(&mut self, class: &js::Class<'ast>) -> Result<Vec<Stmt>> {
        let Some(id) = &class.id else {
            return Ok(vec![]);
        };
        let Some(view) = self.class_names.get(&self.name(id.name.as_str())).copied() else {
            return Ok(vec![]);
        };
        if self.classes[&view].ambient {
            return Ok(vec![]);
        }
        self.evaluated_classes.insert(view);
        let elements = self.classes[&view].static_elements.clone();
        // Static code reads only the fields declared before it.
        let pending: Vec<&'ast str> = elements
            .iter()
            .filter_map(|element| match element {
                StaticElement::Field(name) => Some(*name),
                StaticElement::Block(_) => None,
            })
            .collect();
        let outer_pending = self.static_pending.replace((view, pending));
        let outer_this = self.static_this.replace(view);
        let outer_body = self.class_body.replace(view);
        let result = (|| {
            let mut out = vec![];
            for element in elements {
                match element {
                    StaticElement::Field(name) => {
                        let field = self.classes[&view]
                            .static_fields
                            .iter()
                            .find(|f| f.name == name && f.owner.is_none())
                            .cloned()
                            .unwrap();
                        let Some(init) = field.initializer else {
                            self.static_pending
                                .as_mut()
                                .unwrap()
                                .1
                                .retain(|f| *f != name);
                            continue;
                        };
                        let expected = Type::Primitive(field.ty);
                        let value = self.expr_expected(init, &expected)?;
                        self.expect(&value.ty, &expected, field.span)?;
                        if field.ty == PrimitiveType::String && !self.immortal_string(init) {
                            return self.unsupported(
                                "string fields without a proven static lifetime",
                                field.span,
                            );
                        }
                        out.push(expression(binary(
                            BinaryOp::Assign,
                            var(field.global, expected, field.span),
                            value,
                            unit(),
                            field.span,
                        )));
                        self.static_pending
                            .as_mut()
                            .unwrap()
                            .1
                            .retain(|f| *f != name);
                    }
                    StaticElement::Block(block) => {
                        self.scopes.push(FxHashMap::default());
                        let lowered: Result<Vec<_>> = (|| {
                            self.reserve(&block.body)?;
                            let mut statements = vec![];
                            for s in &block.body {
                                statements.extend(self.statement(s)?);
                            }
                            Ok(statements)
                        })();
                        self.pop_scope();
                        out.push(stmt(
                            TypedStatement::Block(TypedBlock {
                                statements: lowered?,
                                span: span(block.span),
                            }),
                            block.span,
                        ));
                    }
                }
            }
            Ok(out)
        })();
        self.static_this = outer_this;
        self.class_body = outer_body;
        self.static_pending = outer_pending;
        let mut out = result?;
        if let Some(ready) = self.classes[&view].ready {
            out.push(Self::set_flag(ready, class.span));
        }
        Ok(out)
    }
}
