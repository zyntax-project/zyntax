//! Enums: members are compile-time numbers or static strings. An enum value is a native `f64`
//! or string; its enum identity travels as a checker view, and the members a binding may hold
//! are flow facts, as TypeScript narrows a union of member literal types.

use super::*;

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Value {
    Number(f64),
    Text(InternedString),
    /// Evaluated when the declaration runs, or provided by an ambient declaration.
    Computed,
}

pub(super) struct Member {
    pub name: InternedString,
    pub value: Value,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Numeric,
    Text,
    Mixed,
}

pub(super) struct Enum {
    pub source: InternedString,
    pub view: InternedString,
    pub constant: bool,
    pub ambient: bool,
    pub members: Vec<Member>,
    /// Set once the module body has run the declaration, when a call may run before it.
    pub ready: Option<InternedString>,
}

impl Enum {
    pub fn kind(&self) -> Kind {
        let numeric = self
            .members
            .iter()
            .all(|m| matches!(m.value, Value::Number(_) | Value::Computed));
        let text = self
            .members
            .iter()
            .all(|m| matches!(m.value, Value::Text(_)));
        match (numeric, text) {
            (true, _) => Kind::Numeric,
            (false, true) => Kind::Text,
            _ => Kind::Mixed,
        }
    }
    /// Every member is a literal type, so the enum type is their union.
    pub fn union(&self) -> bool {
        !self.members.iter().any(|m| m.value == Value::Computed)
    }
    pub fn physical(&self) -> Type {
        if self.kind() == Kind::Text {
            string()
        } else {
            number()
        }
    }
    pub fn member(&self, name: InternedString) -> Option<usize> {
        self.members.iter().position(|m| m.name == name)
    }
    pub fn all(&self) -> Members {
        Members::all(self.members.len())
    }
    /// The members whose value is the literal `value`.
    pub fn valued(&self, value: Value) -> Members {
        let mut members = Members::none(self.members.len());
        for (index, member) in self.members.iter().enumerate() {
            if member.value == value {
                members.insert(index);
            }
        }
        members
    }
}

/// A set of members of one enum.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Members(SmallVec<[u64; 1]>);

impl Members {
    fn none(count: usize) -> Self {
        Self(smallvec::smallvec![0; count.div_ceil(64)])
    }
    pub fn all(count: usize) -> Self {
        let mut members = Self::none(count);
        for index in 0..count {
            members.insert(index);
        }
        members
    }
    pub fn single(count: usize, index: usize) -> Self {
        let mut members = Self::none(count);
        members.insert(index);
        members
    }
    fn insert(&mut self, index: usize) {
        self.0[index / 64] |= 1 << (index % 64);
    }
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }
    pub fn len(&self) -> u32 {
        self.0.iter().map(|word| word.count_ones()).sum()
    }
    pub fn first(&self) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .find(|(_, word)| **word != 0)
            .map(|(i, word)| i * 64 + word.trailing_zeros() as usize)
    }
    pub fn union(&self, other: &Self) -> Self {
        Self(self.0.iter().zip(&other.0).map(|(a, b)| a | b).collect())
    }
    pub fn intersect(&self, other: &Self) -> Self {
        Self(self.0.iter().zip(&other.0).map(|(a, b)| a & b).collect())
    }
    pub fn without(&self, other: &Self) -> Self {
        Self(self.0.iter().zip(&other.0).map(|(a, b)| a & !b).collect())
    }
}

/// What the checker knows statically about an expression's value for enum rules.
#[derive(Clone, PartialEq)]
pub(super) enum Static {
    Enum {
        view: InternedString,
        members: Members,
    },
    Number(f64),
    Text(InternedString),
    Other,
}

/// ToInt32, for bitwise operators in constant enum expressions.
fn int32(value: f64) -> i32 {
    if !value.is_finite() {
        return 0;
    }
    value.trunc().rem_euclid(4_294_967_296.0) as u64 as u32 as i32
}

fn number_text(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else if value.is_nan() {
        "NaN".to_owned()
    } else if value.is_infinite() {
        if value > 0.0 { "Infinity" } else { "-Infinity" }.to_owned()
    } else {
        dragonbox_ecma::Buffer::new().format(value).to_owned()
    }
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    /// Declares every module-level enum in source order, evaluating constant members.
    pub(super) fn declare_enums(&mut self, program: &'program js::Program<'ast>) -> Result<()> {
        for item in &program.body {
            if let js::Statement::TSEnumDeclaration(declaration) = item {
                let name = self.name(declaration.id.name.as_str());
                self.module_enums.insert(name);
            }
        }
        for item in &program.body {
            let js::Statement::TSEnumDeclaration(declaration) = item else {
                continue;
            };
            self.declare_enum(declaration)?;
        }
        Ok(())
    }

    fn declare_enum(&mut self, declaration: &js::TSEnumDeclaration<'ast>) -> Result<()> {
        let source = self.name(declaration.id.name.as_str());
        if self.enums.contains_key(&source) {
            return self.unsupported("enum declaration merging", declaration.span);
        }
        if self.aliases.contains_key(&source) {
            return self.error(
                ErrorKind::Type,
                "enum declarations can only merge with namespaces or other enums",
                declaration.id.span,
            );
        }
        self.name_buffer.truncate(self.prefix_len);
        self.name_buffer.push_str("enum$");
        crate::modules::append_component(&mut self.name_buffer, declaration.id.name.as_str());
        let view = intern(&self.name_buffer);
        let constant = declaration.r#const;
        let ambient = declaration.declare;
        let mut members: Vec<Member> = Vec::with_capacity(declaration.body.members.len());
        // Later members, here or in later enums, cannot appear in an initializer.
        let later: Vec<&'ast str> = declaration
            .body
            .members
            .iter()
            .filter_map(|m| Self::member_name(&m.id))
            .collect();
        for (index, member) in declaration.body.members.iter().enumerate() {
            let Some(name) = Self::member_name(&member.id) else {
                return self.unsupported("computed enum member names", member.span);
            };
            let key = self.name(name);
            if members.iter().any(|m| m.name == key) {
                return self.error(
                    ErrorKind::Type,
                    format!("duplicate identifier `{name}`"),
                    member.id.span(),
                );
            }
            let value = match &member.initializer {
                Some(init) => {
                    let pending = &later[index..];
                    match self.enum_constant(init, source, &members, pending)? {
                        Some(value) => value,
                        None if constant => {
                            return self.error(
                                ErrorKind::Type,
                                "const enum member initializers must be constant expressions",
                                init.span(),
                            );
                        }
                        None if ambient => {
                            return self.error(
                                ErrorKind::Type,
                                "ambient enum member initializers must be constant expressions",
                                init.span(),
                            );
                        }
                        None => {
                            if self
                                .initializer_type_of(init)
                                .is_some_and(|ty| ty == string())
                            {
                                return self.error(
                                    ErrorKind::Type,
                                    "computed enum member values must be numbers",
                                    init.span(),
                                );
                            }
                            Value::Computed
                        }
                    }
                }
                None if ambient && !constant => Value::Computed,
                None => match members.last().map(|m| m.value) {
                    None => Value::Number(0.0),
                    Some(Value::Number(n)) => Value::Number(n + 1.0),
                    Some(_) => {
                        return self.error(
                            ErrorKind::Type,
                            "enum member must have initializer",
                            member.span,
                        );
                    }
                },
            };
            if constant && let Value::Number(n) = value {
                if n.is_nan() {
                    return self.error(
                        ErrorKind::Type,
                        "const enum member initializer was evaluated to disallowed value NaN",
                        member.span,
                    );
                }
                if n.is_infinite() {
                    return self.error(
                        ErrorKind::Type,
                        "const enum member initializer was evaluated to a non-finite value",
                        member.span,
                    );
                }
            }
            members.push(Member { name: key, value });
        }
        self.enum_views.insert(view, source);
        self.enums.insert(
            source,
            Enum {
                source,
                view,
                constant,
                ambient,
                members,
                ready: None,
            },
        );
        Ok(())
    }

    fn member_name(id: &js::TSEnumMemberName<'ast>) -> Option<&'ast str> {
        match id {
            js::TSEnumMemberName::Identifier(i) => Some(i.name.as_str()),
            js::TSEnumMemberName::String(s) if !s.lone_surrogates => Some(s.value.as_str()),
            _ => None,
        }
    }

    /// A constant enum expression's value; `None` when the expression is not constant.
    fn enum_constant(
        &self,
        e: &js::Expression<'ast>,
        owner: InternedString,
        earlier: &[Member],
        pending: &[&'ast str],
    ) -> Result<Option<Value>> {
        let later = |this: &Self, at| -> Result<Option<Value>> {
            this.error(
                ErrorKind::Type,
                "an enum member initializer cannot reference members declared after it",
                at,
            )
        };
        // A member of this enum: its value, or `None` when the name is not a member.
        let reference =
            |this: &Self, name: &'ast str, at: oxc_span::Span| -> Result<Option<Value>> {
                if let Some(member) = earlier.iter().find(|m| m.name == this.name(name)) {
                    return Ok(Some(member.value));
                }
                if pending.contains(&name) {
                    return later(this, at);
                }
                Ok(None)
            };
        let qualified = |this: &Self, object: &js::Expression<'ast>, member: &'ast str, at| {
            let js::Expression::Identifier(id) = object else {
                return Ok(None);
            };
            let enumeration = this.name(id.name.as_str());
            if enumeration == owner {
                return reference(this, member, at);
            }
            let Some(other) = this.enums.get(&enumeration) else {
                if this.module_enums.contains(&enumeration) {
                    return later(this, at);
                }
                return Ok(None);
            };
            match other.member(this.name(member)) {
                Some(index) => Ok(Some(other.members[index].value)),
                None => this.error(
                    ErrorKind::Type,
                    format!("property `{member}` does not exist on enum `{}`", id.name),
                    at,
                ),
            }
        };
        let value = match e {
            js::Expression::NumericLiteral(n) => Value::Number(n.value),
            js::Expression::StringLiteral(s) if !s.lone_surrogates => {
                Value::Text(self.name(s.value.as_str()))
            }
            js::Expression::TemplateLiteral(t) => {
                let mut text = String::new();
                for (index, quasi) in t.quasis.iter().enumerate() {
                    let Some(cooked) = quasi
                        .value
                        .cooked
                        .as_ref()
                        .filter(|_| !quasi.lone_surrogates)
                    else {
                        return Ok(None);
                    };
                    text.push_str(cooked.as_str());
                    if let Some(expression) = t.expressions.get(index) {
                        match self.enum_constant(expression, owner, earlier, pending)? {
                            Some(Value::Number(n)) => text.push_str(&number_text(n)),
                            Some(Value::Text(s)) => text.push_str(&s.resolve_global().unwrap()),
                            _ => return Ok(None),
                        }
                    }
                }
                Value::Text(intern(&text))
            }
            js::Expression::ParenthesizedExpression(p) => {
                return self.enum_constant(&p.expression, owner, earlier, pending);
            }
            js::Expression::Identifier(i) => {
                if let Some(value) = reference(self, i.name.as_str(), i.span)? {
                    if value == Value::Computed {
                        return Ok(None);
                    }
                    value
                } else if let Some(constant) = self.constants.get(&self.name(i.name.as_str())) {
                    match *constant {
                        Constant::Number(n) => Value::Number(n),
                        Constant::Text(s) => Value::Text(s),
                        Constant::Bool(_) => return Ok(None),
                    }
                } else {
                    match i.name.as_str() {
                        "Infinity" => Value::Number(f64::INFINITY),
                        "NaN" => Value::Number(f64::NAN),
                        _ => return Ok(None),
                    }
                }
            }
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                match qualified(self, &m.object, m.property.name.as_str(), m.span)? {
                    Some(Value::Computed) | None => return Ok(None),
                    Some(value) => value,
                }
            }
            js::Expression::ComputedMemberExpression(m) if !m.optional => {
                let js::Expression::StringLiteral(key) = &m.expression else {
                    return Ok(None);
                };
                match qualified(self, &m.object, key.value.as_str(), m.span)? {
                    Some(Value::Computed) | None => return Ok(None),
                    Some(value) => value,
                }
            }
            js::Expression::UnaryExpression(u) => {
                let Some(operand) = self.enum_constant(&u.argument, owner, earlier, pending)?
                else {
                    return Ok(None);
                };
                let Value::Number(n) = operand else {
                    return match operand {
                        Value::Text(_) if matches!(u.operator.as_str(), "-" | "~") => self.error(
                            ErrorKind::Type,
                            "an arithmetic operand must be a number",
                            u.span,
                        ),
                        _ => Ok(None),
                    };
                };
                match u.operator.as_str() {
                    "+" => Value::Number(n),
                    "-" => Value::Number(-n),
                    "~" => Value::Number(f64::from(!int32(n))),
                    _ => return Ok(None),
                }
            }
            js::Expression::BinaryExpression(b) => {
                let left = self.enum_constant(&b.left, owner, earlier, pending)?;
                let right = self.enum_constant(&b.right, owner, earlier, pending)?;
                let (Some(left), Some(right)) = (left, right) else {
                    return Ok(None);
                };
                if left == Value::Computed || right == Value::Computed {
                    return Ok(None);
                }
                let op = b.operator.as_str();
                match (left, right) {
                    (Value::Number(x), Value::Number(y)) => Value::Number(match op {
                        "+" => x + y,
                        "-" => x - y,
                        "*" => x * y,
                        "/" => x / y,
                        "%" => x % y,
                        "**" => {
                            if x.abs() == 1.0 && y.is_infinite() {
                                f64::NAN
                            } else {
                                x.powf(y)
                            }
                        }
                        "<<" => f64::from(int32(x).wrapping_shl(int32(y) as u32 & 31)),
                        ">>" => f64::from(int32(x).wrapping_shr(int32(y) as u32 & 31)),
                        ">>>" => f64::from((int32(x) as u32).wrapping_shr(int32(y) as u32 & 31)),
                        "&" => f64::from(int32(x) & int32(y)),
                        "|" => f64::from(int32(x) | int32(y)),
                        "^" => f64::from(int32(x) ^ int32(y)),
                        _ => return Ok(None),
                    }),
                    (x, y) if op == "+" => {
                        let text = |v: Value| match v {
                            Value::Number(n) => number_text(n),
                            Value::Text(s) => s.resolve_global().unwrap(),
                            Value::Computed => unreachable!(),
                        };
                        Value::Text(intern(&(text(x) + &text(y))))
                    }
                    _ if matches!(
                        op,
                        "-" | "*" | "/" | "%" | "**" | "<<" | ">>" | ">>>" | "&" | "|" | "^"
                    ) =>
                    {
                        return self.error(
                            ErrorKind::Type,
                            "an arithmetic operand must be a number",
                            b.span,
                        );
                    }
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        Ok(Some(value))
    }

    /// The enum a view names, whether the whole enum or some of its member types.
    pub(super) fn enum_of(&self, view: Option<InternedString>) -> Option<&Enum> {
        let view = view?;
        match self.enum_views.get(&view) {
            Some(source) => self.enums.get(source),
            None => self.enums.get(&self.member_views.borrow().get(&view)?.0),
        }
    }

    /// The members a value of this enum view may hold.
    pub(super) fn declared_members(&self, view: InternedString) -> Option<Members> {
        if let Some((_, members)) = self.member_views.borrow().get(&view) {
            return Some(members.clone());
        }
        self.enum_of(Some(view)).map(Enum::all)
    }

    /// The view of a union of member types; the whole enum when it names every member.
    fn member_view(&self, enumeration: &Enum, members: Members) -> InternedString {
        if members == enumeration.all() {
            return enumeration.view;
        }
        let mut name = enumeration.view.resolve_global().unwrap_or_default();
        for index in 0..enumeration.members.len() {
            if !members
                .intersect(&Members::single(enumeration.members.len(), index))
                .is_empty()
            {
                write!(&mut name, "${index}").unwrap();
            }
        }
        let view = intern(&name);
        self.member_views
            .borrow_mut()
            .entry(view)
            .or_insert((enumeration.source, members));
        view
    }

    /// `E.A` as a type: the member's literal type.
    pub(super) fn enum_member_type(
        &self,
        name: &js::TSQualifiedName<'ast>,
        at: oxc_span::Span,
    ) -> Option<Result<(Type, Option<InternedString>)>> {
        let js::TSTypeName::IdentifierReference(left) = &name.left else {
            return None;
        };
        let enumeration = self.enums.get(&self.name(left.name.as_str()))?;
        let member = name.right.name.as_str();
        let Some(index) = enumeration.member(self.name(member)) else {
            return Some(self.error(
                ErrorKind::Type,
                format!("enum `{}` has no member `{member}`", left.name),
                at,
            ));
        };
        if enumeration.kind() == Kind::Mixed {
            return Some(self.unsupported("mixed numeric and string enum types", at));
        }
        if !enumeration.union() {
            return Some(self.unsupported("member types of enums with computed members", at));
        }
        let view = self.member_view(
            enumeration,
            Members::single(enumeration.members.len(), index),
        );
        Some(Ok((enumeration.physical(), Some(view))))
    }

    /// A union whose alternatives are all member types of one enum.
    pub(super) fn enum_union(
        &self,
        union: &js::TSUnionType<'ast>,
    ) -> Option<Result<(Type, Option<InternedString>)>> {
        let mut found: Option<(InternedString, Members)> = None;
        for member in &union.types {
            let (_, view) = match self.ty(member) {
                Ok(result) => result,
                Err(error) => return found.is_some().then_some(Err(error)),
            };
            let enumeration = self.enum_of(view)?;
            let members = self.declared_members(view?)?;
            found = Some(match found {
                None => (enumeration.source, members),
                Some((source, earlier)) if source == enumeration.source => {
                    (source, earlier.union(&members))
                }
                Some(_) => return Some(self.unsupported("unions of different enums", union.span)),
            });
        }
        let (source, members) = found?;
        let enumeration = &self.enums[&source];
        Some(Ok((
            enumeration.physical(),
            Some(self.member_view(enumeration, members)),
        )))
    }

    /// `E` as a type annotation.
    pub(super) fn enum_type(
        &self,
        source: InternedString,
        at: oxc_span::Span,
    ) -> Option<Result<(Type, Option<InternedString>)>> {
        let enumeration = self.enums.get(&source)?;
        if enumeration.kind() == Kind::Mixed {
            return Some(self.unsupported("mixed numeric and string enum types", at));
        }
        Some(Ok((enumeration.physical(), Some(enumeration.view))))
    }

    /// The enum a member access names, unless a local binding shadows it.
    pub(super) fn enum_access(&self, object: &js::Expression<'ast>) -> Option<&Enum> {
        let js::Expression::Identifier(id) = object else {
            return None;
        };
        let name = self.name(id.name.as_str());
        if self.scopes.iter().any(|s| s.contains_key(&name)) {
            return None;
        }
        self.enums.get(&name)
    }

    fn enum_key(member: &js::Expression<'ast>) -> Option<&'ast str> {
        match member {
            js::Expression::StringLiteral(s) if !s.lone_surrogates => Some(s.value.as_str()),
            _ => None,
        }
    }

    /// `E.A` or `E["A"]`: the member's value, or `None` when the expression is not a member access.
    pub(super) fn enum_member(&mut self, e: &js::Expression<'ast>) -> Option<Result<Expr>> {
        let (object, key, at) = match e {
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                (&m.object, Some(m.property.name.as_str()), m.span)
            }
            js::Expression::ComputedMemberExpression(m) if !m.optional => {
                (&m.object, Self::enum_key(&m.expression), m.span)
            }
            _ => return None,
        };
        let enumeration = self.enum_access(object)?;
        let Some(key) = key else {
            return Some(if enumeration.constant {
                self.error(
                    ErrorKind::Type,
                    "a const enum member can only be accessed using a string literal",
                    at,
                )
            } else {
                self.unsupported("enum reverse mapping", at)
            });
        };
        let Some(index) = enumeration.member(self.name(key)) else {
            return Some(self.error(
                ErrorKind::Type,
                format!("property `{key}` does not exist on enum `{}`", {
                    enumeration.source.resolve_global().unwrap_or_default()
                }),
                at,
            ));
        };
        if enumeration.kind() == Kind::Mixed {
            return Some(self.unsupported("mixed numeric and string enums", at));
        }
        if enumeration.ambient && !enumeration.constant {
            return Some(self.unsupported("values of ambient enums", at));
        }
        let value = match enumeration.members[index].value {
            Value::Number(n) => num(n, at),
            Value::Text(text) => node(
                TypedExpression::Literal(TypedLiteral::String(text)),
                string(),
                at,
            ),
            Value::Computed => return Some(self.unsupported("computed enum members", at)),
        };
        let checks = self.enum_guard(enumeration.source, at);
        if self.top_level
            && !enumeration.constant
            && !self.declared_enums.contains(&enumeration.source)
        {
            return Some(self.error(
                ErrorKind::Type,
                format!(
                    "enum `{}` used before its declaration",
                    enumeration.source.resolve_global().unwrap_or_default()
                ),
                at,
            ));
        }
        Some(Ok(match checks {
            Some(check) => {
                let ty = value.ty.clone();
                node(
                    TypedExpression::Block(TypedBlock {
                        statements: vec![check, expression(value)],
                        span: span(at),
                    }),
                    ty,
                    at,
                )
            }
            None => value,
        }))
    }

    /// The check a function body makes before reading a regular enum's member, when a call may
    /// run before the enum's declaration.
    fn enum_guard(&self, source: InternedString, at: oxc_span::Span) -> Option<Stmt> {
        if self.top_level {
            return None;
        }
        let ready = self.enums[&source].ready?;
        Some(expression(call(
            intern("$TS$enum_check"),
            vec![
                var(ready, boolean(), at),
                node(
                    TypedExpression::Literal(TypedLiteral::String(source)),
                    string(),
                    at,
                ),
            ],
            unit(),
            at,
        )))
    }

    /// `E` used as a value.
    pub(super) fn enum_object(&self, id: &js::IdentifierReference<'ast>) -> Option<Result<Expr>> {
        let name = self.name(id.name.as_str());
        if self.scopes.iter().any(|s| s.contains_key(&name)) {
            return None;
        }
        let enumeration = self.enums.get(&name)?;
        Some(if enumeration.constant {
            self.error(
                ErrorKind::Type,
                "const enums can only be used in property or index access expressions",
                id.span,
            )
        } else {
            self.unsupported("enum objects as values", id.span)
        })
    }

    /// What the checker knows about an expression for enum and literal rules.
    pub(super) fn static_value(&self, e: &js::Expression<'ast>) -> Static {
        match e {
            js::Expression::NumericLiteral(n) => Static::Number(n.value),
            js::Expression::UnaryExpression(u) if u.operator.as_str() == "-" => match &u.argument {
                js::Expression::NumericLiteral(n) => Static::Number(-n.value),
                _ => Static::Other,
            },
            js::Expression::StringLiteral(s) if !s.lone_surrogates => {
                Static::Text(self.name(s.value.as_str()))
            }
            js::Expression::TemplateLiteral(_) => match constants::literal_text(e) {
                Some(text) => Static::Text(intern(&text)),
                None => Static::Other,
            },
            js::Expression::ParenthesizedExpression(p) => self.static_value(&p.expression),
            js::Expression::TSNonNullExpression(n) => self.static_value(&n.expression),
            js::Expression::AssignmentExpression(a) if a.operator.as_str() == "=" => {
                self.static_value(&a.right)
            }
            js::Expression::Identifier(i) => {
                let name = self.name(i.name.as_str());
                if !self.scopes.iter().any(|s| s.contains_key(&name)) {
                    return match self.literal_constants.get(&name) {
                        Some(value) => value.clone(),
                        None => Static::Other,
                    };
                }
                let Ok(binding) = self.binding(i.name.as_str(), i.span) else {
                    return Static::Other;
                };
                if let (Some(members), Some(enumeration)) =
                    (self.flow.members(binding.name), self.enum_of(binding.view))
                {
                    return Static::Enum {
                        view: enumeration.view,
                        members: members.clone(),
                    };
                }
                if let Some(literal) = self.const_literals.get(&binding.name) {
                    return literal.clone();
                }
                match (self.enum_of(binding.view), binding.view) {
                    (Some(enumeration), Some(view)) => Static::Enum {
                        view: enumeration.view,
                        members: self.declared_members(view).unwrap(),
                    },
                    _ => Static::Other,
                }
            }
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                self.static_member(&m.object, Some(m.property.name.as_str()))
            }
            js::Expression::ComputedMemberExpression(m) if !m.optional => {
                self.static_member(&m.object, Self::enum_key(&m.expression))
            }
            js::Expression::ConditionalExpression(c) => {
                match (
                    self.static_value(&c.consequent),
                    self.static_value(&c.alternate),
                ) {
                    (
                        Static::Enum { view, members },
                        Static::Enum {
                            view: other,
                            members: more,
                        },
                    ) if view == other => Static::Enum {
                        view,
                        members: members.union(&more),
                    },
                    (a, b) if a == b => a,
                    _ => Static::Other,
                }
            }
            js::Expression::CallExpression(_) | js::Expression::TSAsExpression(_) => {
                let view = self.view_of(e);
                match (self.enum_of(view), view) {
                    (Some(enumeration), Some(view)) => Static::Enum {
                        view: enumeration.view,
                        members: self.declared_members(view).unwrap(),
                    },
                    _ => Static::Other,
                }
            }
            _ => Static::Other,
        }
    }

    fn static_member(&self, object: &js::Expression<'ast>, key: Option<&'ast str>) -> Static {
        let Some(enumeration) = self.enum_access(object) else {
            return Static::Other;
        };
        let Some(index) = key.and_then(|key| enumeration.member(self.name(key))) else {
            return Static::Other;
        };
        let members = if enumeration.union() {
            Members::single(enumeration.members.len(), index)
        } else {
            enumeration.all()
        };
        Static::Enum {
            view: enumeration.view,
            members,
        }
    }

    fn describe_static(&self, value: &Static) -> String {
        match value {
            Static::Enum { view, members } => {
                let enumeration = self.enum_of(Some(*view)).unwrap();
                let source = enumeration.source.resolve_global().unwrap_or_default();
                if let (1, Some(index)) = (members.len(), members.first()) {
                    format!(
                        "{source}.{}",
                        enumeration.members[index]
                            .name
                            .resolve_global()
                            .unwrap_or_default()
                    )
                } else {
                    source
                }
            }
            Static::Number(n) => number_text(*n),
            Static::Text(s) => format!("\"{}\"", s.resolve_global().unwrap_or_default()),
            Static::Other => "value".into(),
        }
    }

    /// Whether two values can be equal, by their enum identities and literal types.
    fn overlap(&self, a: &Static, b: &Static) -> bool {
        match (a, b) {
            (Static::Other, _) | (_, Static::Other) => true,
            (
                Static::Enum { view, members },
                Static::Enum {
                    view: other,
                    members: more,
                },
            ) => view == other && !members.intersect(more).is_empty(),
            (Static::Enum { view, members }, literal)
            | (literal, Static::Enum { view, members }) => {
                let enumeration = self.enum_of(Some(*view)).unwrap();
                if !enumeration.union() {
                    return enumeration.kind() == Kind::Numeric
                        && matches!(literal, Static::Number(_));
                }
                let value = match literal {
                    Static::Number(n) => Value::Number(*n),
                    Static::Text(s) => Value::Text(*s),
                    _ => unreachable!(),
                };
                !enumeration.valued(value).intersect(members).is_empty()
            }
            (Static::Number(x), Static::Number(y)) => x == y || (x.is_nan() && y.is_nan()),
            (Static::Text(x), Static::Text(y)) => x == y,
            _ => false,
        }
    }

    /// TypeScript rejects an equality comparison between values that can never be equal.
    pub(super) fn check_comparison(
        &self,
        left: &js::Expression<'ast>,
        right: &js::Expression<'ast>,
        at: oxc_span::Span,
    ) -> Result<()> {
        let a = self.static_value(left);
        let b = self.static_value(right);
        if self.overlap(&a, &b) {
            return Ok(());
        }
        self.error(
            ErrorKind::Type,
            format!(
                "comparison has no overlapping types: `{}` and `{}`",
                self.describe_static(&a),
                self.describe_static(&b)
            ),
            at,
        )
    }

    /// Assigning `source` where a value of the enum `target` is expected.
    pub(super) fn check_enum(
        &self,
        source: &js::Expression<'ast>,
        target: Option<InternedString>,
        at: oxc_span::Span,
    ) -> Result<()> {
        let Some(enumeration) = self.enum_of(target) else {
            return Ok(());
        };
        match source {
            js::Expression::ParenthesizedExpression(p) => {
                return self.check_enum(&p.expression, target, at);
            }
            js::Expression::ConditionalExpression(c) => {
                self.check_enum(&c.consequent, target, at)?;
                return self.check_enum(&c.alternate, target, at);
            }
            _ => {}
        }
        let allowed = self.declared_members(target.unwrap()).unwrap();
        let value = self.static_value(source);
        let accepted = match &value {
            Static::Enum { view, members } => {
                *view == enumeration.view && members.without(&allowed).is_empty()
            }
            Static::Number(n) => {
                enumeration.kind() == Kind::Numeric
                    && (!enumeration.union()
                        || !enumeration
                            .valued(Value::Number(*n))
                            .intersect(&allowed)
                            .is_empty())
            }
            Static::Text(_) => false,
            Static::Other => {
                enumeration.kind() == Kind::Numeric && self.enum_of(self.view_of(source)).is_none()
            }
        };
        if accepted {
            return Ok(());
        }
        let source = match &value {
            Static::Other => self
                .enum_of(self.view_of(source))
                .map(|e| e.source.resolve_global().unwrap_or_default())
                .unwrap_or_else(|| "value".into()),
            value => self.describe_static(value),
        };
        self.error(
            ErrorKind::Type,
            format!(
                "type `{source}` is not assignable to `{}`",
                self.describe_static(&Static::Enum {
                    view: enumeration.view,
                    members: allowed,
                })
            ),
            at,
        )
    }

    /// Records what an enum binding holds after `source` is stored in it.
    pub(super) fn enum_facts(
        &mut self,
        name: InternedString,
        view: Option<InternedString>,
        source: &js::Expression<'ast>,
    ) {
        self.flow.forget_members(name);
        let Some(enumeration) = self.enum_of(view) else {
            return;
        };
        if !enumeration.union() {
            return;
        }
        let members = match self.static_value(source) {
            Static::Enum { view, members } if view == enumeration.view => members,
            Static::Number(n) => enumeration.valued(Value::Number(n)),
            Static::Text(s) => enumeration.valued(Value::Text(s)),
            _ => return,
        };
        self.flow.set_members(name, members);
    }

    /// Narrows an enum binding compared by `===`/`!==` with a member or literal.
    pub(super) fn narrow_enum(&mut self, b: &js::BinaryExpression<'ast>, equality: bool) {
        if !self.narrow_equal(&b.left, &b.right, equality) {
            self.narrow_equal(&b.right, &b.left, equality);
        }
    }

    /// Narrows `target`, an enum binding, by its equality (or inequality) with `other`.
    pub(super) fn narrow_equal(
        &mut self,
        target: &js::Expression<'ast>,
        other: &js::Expression<'ast>,
        equality: bool,
    ) -> bool {
        let js::Expression::Identifier(id) = target else {
            return false;
        };
        let Ok(binding) = self.binding(id.name.as_str(), id.span) else {
            return false;
        };
        let (name, view) = (binding.name, binding.view);
        let Some(enumeration) = self.enum_of(view) else {
            return false;
        };
        if !enumeration.union() {
            return false;
        }
        let current = self
            .flow
            .members(name)
            .cloned()
            .or_else(|| self.declared_members(view?))
            .unwrap_or_else(|| enumeration.all());
        let compared = match self.static_value(other) {
            Static::Enum { view, members } if view == enumeration.view => members,
            Static::Number(n) => enumeration.valued(Value::Number(n)),
            Static::Text(s) => enumeration.valued(Value::Text(s)),
            _ => return false,
        };
        let narrowed = if equality {
            current.intersect(&compared)
        } else if compared.len() == 1 {
            current.without(&compared)
        } else {
            return false;
        };
        self.flow.set_members(name, narrowed);
        true
    }

    /// The type a computed member's initializer has, when it can be told without lowering it.
    fn initializer_type_of(&self, e: &js::Expression<'ast>) -> Option<Type> {
        match e {
            js::Expression::StringLiteral(_) | js::Expression::TemplateLiteral(_) => Some(string()),
            js::Expression::ParenthesizedExpression(p) => self.initializer_type_of(&p.expression),
            js::Expression::BinaryExpression(b) if b.operator.as_str() == "+" => {
                let left = self.initializer_type_of(&b.left);
                let right = self.initializer_type_of(&b.right);
                (left == Some(string()) || right == Some(string())).then(string)
            }
            js::Expression::CallExpression(c) => {
                let js::Expression::StaticMemberExpression(m) = &c.callee else {
                    return None;
                };
                (m.property.name == "toString"
                    && self
                        .initializer_type_of(&m.object)
                        .is_some_and(|t| t == string()))
                .then(string)
            }
            _ => None,
        }
    }

    /// Rejects an enum where the frontend keeps only a physical type, which would lose the
    /// enum's assignability rules.
    pub(super) fn plain_enum(
        &self,
        view: Option<InternedString>,
        what: &str,
        at: oxc_span::Span,
    ) -> Result<()> {
        if self.enum_of(view).is_some() {
            return self.unsupported(what, at);
        }
        Ok(())
    }

    /// The enum view of `E.A` or `E["A"]`.
    pub(super) fn enum_view_of_member(
        &self,
        object: &js::Expression<'ast>,
    ) -> Option<InternedString> {
        self.enum_access(object).map(|e| e.view)
    }

    /// Records the literal type of a `const` declared without an annotation.
    pub(super) fn declare_literal(&mut self, name: InternedString, init: &js::Expression<'ast>) {
        let literal = self.static_value(init);
        if !matches!(literal, Static::Other) {
            self.const_literals.insert(name, literal);
        }
    }

    /// The module body reaching an enum declaration: later module code may read it, and
    /// functions that check its readiness see it ready.
    pub(super) fn run_enum_declaration(
        &mut self,
        declaration: &js::TSEnumDeclaration<'ast>,
    ) -> Option<Stmt> {
        let source = self.name(declaration.id.name.as_str());
        self.declared_enums.insert(source);
        let ready = self.enums.get(&source)?.ready?;
        Some(expression(binary(
            BinaryOp::Assign,
            var(ready, boolean(), declaration.span),
            node(
                TypedExpression::Literal(TypedLiteral::Bool(true)),
                boolean(),
                declaration.span,
            ),
            unit(),
            declaration.span,
        )))
    }
}
