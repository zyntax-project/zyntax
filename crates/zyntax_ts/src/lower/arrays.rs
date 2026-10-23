use super::*;
use std::ops::{Deref, DerefMut};

pub(super) fn array_type(element: Type) -> Type {
    Type::Array {
        element_type: Box::new(element),
        size: None,
        nullability: NullabilityKind::NonNull,
    }
}
pub(super) fn element_type(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Array {
            element_type,
            size: None,
            ..
        } => Some(element_type),
        _ => None,
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ArrayRoot(usize);
#[derive(Clone, Copy)]
pub(super) struct ArrayFact {
    root: ArrayRoot,
    minimum: u64,
}
pub(super) use zyntax_typed_ast::checker_interface::ArrayReturn as ArrayResult;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Integral {
    Constant(u64),
    Nonnegative,
}

/// Storage identity and bounds survive aliases; source numbers keep their f64 ABI.
#[derive(Clone, Default)]
pub(super) struct FlowState {
    scalars: FxHashMap<InternedString, ScalarFlow>,
    arrays: FxHashMap<InternedString, ArrayFact>,
    integers: FxHashMap<InternedString, Integral>,
    bounds: FxHashSet<(ArrayRoot, InternedString)>,
    strings: FxHashSet<InternedString>,
    /// Bindings declared without a value that every path to here has assigned.
    assigned: FxHashSet<InternedString>,
    /// The enum members a binding may hold, when narrower than its declared enum.
    members: FxHashMap<InternedString, super::enums::Members>,
}
impl Deref for FlowState {
    type Target = FxHashMap<InternedString, ScalarFlow>;
    fn deref(&self) -> &Self::Target {
        &self.scalars
    }
}
impl DerefMut for FlowState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.scalars
    }
}
impl FlowState {
    pub(super) fn forget(&mut self, name: InternedString) {
        self.scalars.remove(&name);
        self.arrays.remove(&name);
        self.integers.remove(&name);
        self.strings.remove(&name);
        self.bounds.retain(|(_, i)| *i != name);
        self.members.remove(&name);
    }
    pub(super) fn members(&self, name: InternedString) -> Option<&super::enums::Members> {
        self.members.get(&name)
    }
    pub(super) fn set_members(&mut self, name: InternedString, members: super::enums::Members) {
        self.members.insert(name, members);
    }
    pub(super) fn forget_members(&mut self, name: InternedString) {
        self.members.remove(&name);
    }
    /// Facts about module globals do not cross into or out of a function body.
    pub(super) fn forget_global_members(
        &mut self,
        globals: &FxHashMap<InternedString, super::globals::Global>,
    ) {
        self.members.retain(|name, _| !globals.contains_key(name));
    }
    pub(super) fn assign(&mut self, name: InternedString) {
        self.assigned.insert(name);
    }
    pub(super) fn is_assigned(&self, name: InternedString) -> bool {
        self.assigned.contains(&name)
    }
    /// Assignments made in a loop body do not hold once the loop may run zero times.
    pub(super) fn assigned(&self) -> FxHashSet<InternedString> {
        self.assigned.clone()
    }
    pub(super) fn restore_assigned(&mut self, assigned: FxHashSet<InternedString>) {
        self.assigned = assigned;
    }
    pub(super) fn merge(&mut self, other: &Self) {
        self.assigned.retain(|n| other.assigned.contains(n));
        self.scalars.retain(|n, v| other.scalars.get(n) == Some(v));
        self.arrays.retain(|n, v| {
            if let Some(b) = other.arrays.get(n).filter(|b| b.root == v.root) {
                v.minimum = v.minimum.min(b.minimum);
                true
            } else {
                false
            }
        });
        self.integers
            .retain(|n, v| other.integers.get(n) == Some(v));
        self.bounds.retain(|b| other.bounds.contains(b));
        self.strings.retain(|n| other.strings.contains(n));
        self.members
            .retain(|n, members| match other.members.get(n) {
                Some(more) => {
                    *members = members.union(more);
                    true
                }
                None => false,
            });
    }
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    fn fresh_array_identity(&mut self) -> ArrayRoot {
        let root = ArrayRoot(self.next_array_identity);
        self.next_array_identity += 1;
        root
    }
    pub(super) fn array_element(&self, ty: &Type, at: oxc_span::Span) -> Result<Type> {
        let Some(element) = element_type(ty) else {
            return self.error(ErrorKind::Type, "value is not an array", at);
        };
        Ok(element.clone())
    }
    pub(super) fn check_array_element(&self, element: &Type, at: oxc_span::Span) -> Result<()> {
        if is_scalar(element)
            || *element == string()
            || matches!(element, Type::TypeVar(_))
            || self.is_class_instance(element)
        {
            Ok(())
        } else if let Some(inner) = element_type(element) {
            self.check_array_element(inner, at)
        } else {
            self.unsupported("arrays with this element ownership or representation", at)
        }
    }
    pub(super) fn array_literal(
        &mut self,
        a: &js::ArrayExpression<'ast>,
        hint: Option<&Type>,
    ) -> Result<Expr> {
        let mut values = Vec::with_capacity(a.elements.len());
        let mut element = hint.and_then(element_type).cloned();
        let contextual = element.is_some();
        for source in &a.elements {
            let Some(source) = source.as_expression() else {
                return self.unsupported("sparse or spread arrays", a.span);
            };
            self.plain_enum(self.view_of(source), "arrays of enum values", source.span())?;
            let value = if let Some(hint) = &element {
                self.expr_expected(source, hint)?
            } else {
                self.expr(source)?
            };
            if let Some(element) = &element {
                if !contextual && value.ty != *element {
                    return self.unsupported("heterogeneous array elements", source.span());
                }
                self.expect(&value.ty, element, source.span())?;
            } else {
                element = Some(value.ty.clone());
            }
            if value.ty == string() && !self.immortal_string(source) {
                return self.unsupported(
                    "array storage of strings without a proven static lifetime",
                    source.span(),
                );
            }
            values.push(value);
        }
        let Some(element) = element else {
            return self.unsupported("uncontextualized empty array inference", a.span);
        };
        self.check_array_element(&element, a.span)?;
        if matches!(element, Type::TypeVar(_)) {
            return self.unsupported(
                "generic array construction before element ownership is known",
                a.span,
            );
        }
        Ok(node(
            TypedExpression::Array(values),
            array_type(element),
            a.span,
        ))
    }
    pub(super) fn immortal_string(&self, source: &js::Expression<'ast>) -> bool {
        match source {
            js::Expression::StringLiteral(s) => !s.lone_surrogates,
            js::Expression::TemplateLiteral(_) => constants::literal_text(source).is_some(),
            js::Expression::Identifier(i) if self.constant_text(i.name.as_str()) => true,
            js::Expression::Identifier(i) => self.binding(i.name.as_str(), i.span).is_ok_and(|b| {
                self.flow.strings.contains(&b.name)
                    || (matches!(b.ty, Type::Union(_))
                        && self.flow.get(&b.name) == Some(&ScalarFlow::String))
            }),
            js::Expression::ParenthesizedExpression(p) => self.immortal_string(&p.expression),
            // Every write to a string field stores a static string.
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                self.enum_view_of_member(&m.object).is_some()
                    || self.string_field(&m.object, m.property.name.as_str())
            }
            js::Expression::ComputedMemberExpression(m) if !m.optional => {
                Self::string_key(&m.expression).is_some_and(|key| self.string_field(&m.object, key))
            }
            js::Expression::CallExpression(c) => self
                .array_call(c)
                .and_then(|(_, argument)| argument)
                .is_some_and(|a| self.immortal_string(a)),
            js::Expression::ConditionalExpression(c) => {
                self.immortal_string(&c.consequent) && self.immortal_string(&c.alternate)
            }
            _ => false,
        }
    }
    /// Whether `object.name` is a string field of a record or class.
    fn string_field(&self, object: &js::Expression<'ast>, name: &'ast str) -> bool {
        let Some(view) = self.view_of(object).and_then(|v| self.views.get(&v)) else {
            return false;
        };
        view.field(self.name(name))
            .is_some_and(|f| f.ty == string())
    }
    fn array_source(&self, source: &js::Expression<'ast>) -> Option<ArrayFact> {
        match source {
            js::Expression::Identifier(i) => self
                .binding(i.name.as_str(), i.span)
                .ok()
                .and_then(|b| self.flow.arrays.get(&b.name).copied()),
            js::Expression::ParenthesizedExpression(p) => self.array_source(&p.expression),
            js::Expression::TSAsExpression(a) => self.array_source(&a.expression),
            js::Expression::CallExpression(c) => {
                let (result, argument) = self.array_call(c)?;
                let mut fact = self.array_source(argument?)?;
                fact.minimum = fact.minimum.max(result.minimum);
                Some(fact)
            }
            _ => None,
        }
    }
    fn array_call<'e>(
        &self,
        c: &'e js::CallExpression<'ast>,
    ) -> Option<(ArrayResult, Option<&'e js::Expression<'ast>>)> {
        // Argument writes can change a later argument or an earlier captured alias.
        if self.has_binding_write(c.span) {
            return None;
        }
        let result = self
            .direct_signature(&c.callee)
            .ok()
            .flatten()?
            .array_result?;
        let argument = result
            .parameter
            .and_then(|i| c.arguments.get(i))
            .and_then(|a| a.as_expression());
        Some((result, argument))
    }
    fn array_minimum(&self, source: &js::Expression<'ast>) -> u64 {
        if let Some(fact) = self.array_source(source) {
            return fact.minimum;
        }
        if let Some(length) = Self::literal_length(source) {
            return length;
        }
        match source {
            js::Expression::CallExpression(c) => self.array_call(c).map_or(0, |(result, arg)| {
                result.minimum.max(arg.map_or(0, |a| self.array_minimum(a)))
            }),
            js::Expression::ParenthesizedExpression(p) => self.array_minimum(&p.expression),
            _ => 0,
        }
    }
    pub(super) fn array_parameter(&mut self, name: InternedString, ty: &Type) -> Option<ArrayRoot> {
        if element_type(ty).is_none() && !matches!(ty, Type::TypeVar(_)) {
            return None;
        }
        let root = self.fresh_array_identity();
        self.flow
            .arrays
            .insert(name, ArrayFact { root, minimum: 0 });
        Some(root)
    }
    pub(super) fn record_array_return(&mut self, source: &js::Expression<'ast>) {
        let minimum = self.array_minimum(source);
        let parameter = self.array_source(source).and_then(|fact| {
            self.returns
                .as_ref()
                .unwrap()
                .array_parameters
                .iter()
                .position(|p| *p == Some(fact.root))
        });
        let result = ArrayResult { minimum, parameter };
        let context = self.returns.as_mut().unwrap();
        context.array_result = Some(match context.array_result {
            None => result,
            Some(previous) => ArrayResult {
                minimum: previous.minimum.min(minimum),
                parameter: previous.parameter.filter(|p| Some(*p) == parameter),
            },
        });
    }
    fn literal_length(source: &js::Expression<'ast>) -> Option<u64> {
        match source {
            js::Expression::ArrayExpression(a) => Some(a.elements.len() as u64),
            js::Expression::ParenthesizedExpression(p) => Self::literal_length(&p.expression),
            _ => None,
        }
    }
    pub(super) fn value_facts(
        &mut self,
        name: InternedString,
        ty: &Type,
        source: &js::Expression<'ast>,
    ) {
        if self.globals.contains_key(&name) {
            return;
        }
        if element_type(ty).is_some()
            || (matches!(ty, Type::TypeVar(_)) && self.array_source(source).is_some())
        {
            let fact = self.array_source(source).unwrap_or_else(|| ArrayFact {
                root: self.fresh_array_identity(),
                minimum: self.array_minimum(source),
            });
            self.flow.arrays.insert(name, fact);
        }
        if *ty == number() {
            if let Some(integer) = self.integral(source) {
                self.flow.integers.insert(name, integer);
            }
        }
        if *ty == string() && self.immortal_string(source) {
            self.flow.strings.insert(name);
        }
    }
    fn array_identity(&mut self, source: &js::Expression<'ast>) -> Option<ArrayFact> {
        if let Some(fact) = self.array_source(source) {
            return Some(fact);
        }
        match source {
            js::Expression::Identifier(i) => {
                let b = self.binding(i.name.as_str(), i.span).ok()?;
                if element_type(&b.ty).is_none() || self.globals.contains_key(&b.name) {
                    return None;
                }
                let name = b.name;
                let fact = ArrayFact {
                    root: self.fresh_array_identity(),
                    minimum: 0,
                };
                self.flow.arrays.insert(name, fact);
                Some(fact)
            }
            js::Expression::ParenthesizedExpression(p) => self.array_identity(&p.expression),
            _ => None,
        }
    }
    pub(super) fn integral(&self, source: &js::Expression<'ast>) -> Option<Integral> {
        match source {
            js::Expression::NumericLiteral(n)
                if n.value >= 0.0 && n.value <= u32::MAX as f64 && n.value.fract() == 0.0 =>
            {
                Some(Integral::Constant(n.value as u64))
            }
            js::Expression::Identifier(i) => self
                .binding(i.name.as_str(), i.span)
                .ok()
                .and_then(|b| self.flow.integers.get(&b.name).copied())
                .or_else(|| self.constant_integral(i.name.as_str())),
            js::Expression::ParenthesizedExpression(p) => self.integral(&p.expression),
            js::Expression::BinaryExpression(b) if b.operator.as_str() == "+" => {
                match (self.integral(&b.left)?, self.integral(&b.right)?) {
                    (Integral::Constant(a), Integral::Constant(c))
                        if a + c <= u64::from(u32::MAX) =>
                    {
                        Some(Integral::Constant(a + c))
                    }
                    _ => Some(Integral::Nonnegative),
                }
            }
            _ => None,
        }
    }
    pub(super) fn array_index(
        &mut self,
        object: &js::Expression<'ast>,
        index: &js::Expression<'ast>,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        // Capture the reference before evaluating an index that might reassign it.
        let fact = self.array_source(object);
        let minimum = fact
            .map(|f| f.minimum)
            .or_else(|| Self::literal_length(object))
            .unwrap_or(0);
        let object = self.expr(object)?;
        let element = self.array_element(&object.ty, at)?;
        let integral = self.integral(index);
        let bounded = match integral {
            Some(Integral::Constant(i)) => i < minimum,
            Some(Integral::Nonnegative) => {
                if let (Some(fact), js::Expression::Identifier(i)) = (fact, index) {
                    self.binding(i.name.as_str(), i.span)
                        .is_ok_and(|b| self.flow.bounds.contains(&(fact.root, b.name)))
                } else {
                    false
                }
            }
            None => false,
        };
        let index = if let js::Expression::Identifier(id) = index {
            let binding = self.binding(id.name.as_str(), id.span)?;
            if binding.ty == Type::Primitive(PrimitiveType::I64) {
                var(binding.name, binding.ty.clone(), at)
            } else {
                self.expr(index)?
            }
        } else if let Some(offset) = self.integer_offset(index, at) {
            offset
        } else {
            self.expr(index)?
        };
        if !bounded {
            return self.checked_index(object, index, element, at);
        }
        let index = if index.ty == Type::Primitive(PrimitiveType::I64) {
            index
        } else {
            self.expect(&index.ty, &number(), at)?;
            let index = node(
                TypedExpression::Cast(TypedCast {
                    expr: Box::new(index),
                    target_type: Type::Primitive(PrimitiveType::I64),
                }),
                Type::Primitive(PrimitiveType::I64),
                at,
            );
            index
        };
        Ok(node(
            TypedExpression::Index(TypedIndex {
                object: Box::new(object),
                index: Box::new(index),
            }),
            element,
            at,
        ))
    }
    /// An index that is not proven in range: guard it, since TypeScript would read
    /// `undefined` (or grow the array) where a native element read cannot.
    ///
    /// The index is converted once; the conversion is total, so a NaN, infinite or huge float
    /// saturates to a value that fails the bound. A float must also convert back to itself,
    /// which rejects a fraction, and one unsigned compare covers both ends of the range.
    fn checked_index(
        &mut self,
        object: Expr,
        index: Expr,
        element: Type,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        const REASON: &str =
            "array indexing without proven integer bounds (the result may be undefined)";
        if self.options.indexing == Indexing::Proven {
            return self.unsupported(REASON, at);
        }
        let integer = Type::Primitive(PrimitiveType::I64);
        let unsigned = Type::Primitive(PrimitiveType::U64);
        let cast = |value: Expr, ty: &Type| {
            node(
                TypedExpression::Cast(TypedCast {
                    expr: Box::new(value),
                    target_type: ty.clone(),
                }),
                ty.clone(),
                at,
            )
        };
        let integral_source = index.ty == integer;
        if !integral_source {
            self.expect(&index.ty, &number(), at)?;
        }
        let let_ = |name: InternedString, ty: &Type, value: Expr| {
            stmt(
                TypedStatement::Let(TypedLet {
                    name,
                    ty: ty.clone(),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(value)),
                    span: span(at),
                }),
                at,
            )
        };
        let mut receiver = Vec::new();
        let object = if matches!(object.node, TypedExpression::Variable(_)) {
            object
        } else {
            let name = self.fresh("receiver");
            let ty = object.ty.clone();
            receiver.push(let_(name, &ty, object));
            var(name, ty, at)
        };
        let slot = self.fresh("slot");
        let length = self.fresh("length");
        let slot_value = || var(slot, integer.clone(), at);
        let mut statements = vec![let_(
            length,
            &integer,
            node(
                TypedExpression::MethodCall(TypedMethodCall::positional(
                    object.clone(),
                    self.name("len"),
                    vec![],
                )),
                integer.clone(),
                at,
            ),
        )];
        // In range as an unsigned number: a negative slot is a huge one.
        let below = || {
            binary(
                BinaryOp::Lt,
                cast(slot_value(), &unsigned),
                cast(var(length, integer.clone(), at), &unsigned),
                boolean(),
                at,
            )
        };
        let (valid, written) = if integral_source {
            statements.push(let_(slot, &integer, index));
            (below(), cast(slot_value(), &number()))
        } else {
            let written = self.fresh("index");
            statements.push(let_(written, &number(), index));
            let written_value = || var(written, number(), at);
            statements.push(let_(slot, &integer, cast(written_value(), &integer)));
            let whole = binary(
                BinaryOp::Eq,
                cast(slot_value(), &number()),
                written_value(),
                boolean(),
                at,
            );
            (
                binary(BinaryOp::And, below(), whole, boolean(), at),
                written_value(),
            )
        };
        let invalid = node(
            TypedExpression::Unary(TypedUnary {
                op: UnaryOp::Not,
                operand: Box::new(valid),
            }),
            boolean(),
            at,
        );
        let length_float = cast(var(length, integer.clone(), at), &number());
        let trap = self.call("$TS$index_trap", vec![written, length_float], unit(), at);
        statements.push(stmt(
            TypedStatement::If(TypedIf {
                condition: Box::new(invalid),
                then_block: TypedBlock {
                    statements: vec![expression(trap)],
                    span: span(at),
                },
                else_block: None,
                span: span(at),
            }),
            at,
        ));
        statements.push(expression(slot_value()));
        let guarded = node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(at),
            }),
            integer,
            at,
        );
        let read = node(
            TypedExpression::Index(TypedIndex {
                object: Box::new(object),
                index: Box::new(guarded),
            }),
            element.clone(),
            at,
        );
        if receiver.is_empty() {
            return Ok(read);
        }
        receiver.push(expression(read));
        Ok(node(
            TypedExpression::Block(TypedBlock {
                statements: receiver,
                span: span(at),
            }),
            element,
            at,
        ))
    }
    /// `counter + c` or `counter - c` for a native integer counter and a small whole constant,
    /// kept in integers so its guard stays an integer compare.
    fn integer_offset(&self, index: &js::Expression<'ast>, at: oxc_span::Span) -> Option<Expr> {
        let js::Expression::BinaryExpression(b) = index else {
            return None;
        };
        let operation = match b.operator.as_str() {
            "+" => BinaryOp::Add,
            "-" => BinaryOp::Sub,
            _ => return None,
        };
        let (js::Expression::Identifier(id), js::Expression::NumericLiteral(n)) =
            (&b.left, &b.right)
        else {
            return None;
        };
        if n.value.fract() != 0.0 || !(0.0..=2_147_483_648.0).contains(&n.value) {
            return None;
        }
        let binding = self.binding(id.name.as_str(), id.span).ok()?;
        let integer = Type::Primitive(PrimitiveType::I64);
        if binding.ty != integer {
            return None;
        }
        let constant = node(
            TypedExpression::Literal(TypedLiteral::Integer(n.value as i128)),
            integer.clone(),
            at,
        );
        Some(binary(
            operation,
            var(binding.name, integer.clone(), at),
            constant,
            integer,
            at,
        ))
    }
    pub(super) fn array_length(&self, value: Expr, at: oxc_span::Span) -> Expr {
        let count = node(
            TypedExpression::MethodCall(TypedMethodCall::positional(
                value,
                self.name("len"),
                vec![],
            )),
            Type::Primitive(PrimitiveType::I64),
            at,
        );
        node(
            TypedExpression::Cast(TypedCast {
                expr: Box::new(count),
                target_type: number(),
            }),
            number(),
            at,
        )
    }
    fn length_source<'e>(source: &'e js::Expression<'ast>) -> Option<&'e js::Expression<'ast>> {
        match source {
            js::Expression::StaticMemberExpression(m)
                if !m.optional && m.property.name == "length" =>
            {
                Some(&m.object)
            }
            js::Expression::ParenthesizedExpression(p) => Self::length_source(&p.expression),
            _ => None,
        }
    }
    pub(super) fn narrow_array_truthiness(&mut self, test: &js::Expression<'ast>, truth: bool) {
        if truth {
            if let Some(source) = Self::length_source(test) {
                if let Some(fact) = self.array_identity(source) {
                    for alias in self
                        .flow
                        .arrays
                        .values_mut()
                        .filter(|a| a.root == fact.root)
                    {
                        alias.minimum = alias.minimum.max(1);
                    }
                }
            }
        }
    }
    pub(super) fn narrow_array(&mut self, test: &js::BinaryExpression<'ast>, truth: bool) {
        // Only length lower bounds are facts about storage, not arbitrary f64 indices.
        for (length, other, reversed) in [
            (&test.left, &test.right, false),
            (&test.right, &test.left, true),
        ] {
            let Some(source) = Self::length_source(length) else {
                continue;
            };
            let Some(fact) = self.array_identity(source) else {
                continue;
            };
            let mut op = test.operator.as_str();
            if reversed {
                op = match op {
                    "<" => ">",
                    "<=" => ">=",
                    ">" => "<",
                    ">=" => "<=",
                    x => x,
                };
            }
            let minimum = match (self.integral(other), op, truth) {
                (Some(Integral::Constant(i)), ">", true)
                | (Some(Integral::Constant(i)), "<=", false) => Some(i + 1),
                (Some(Integral::Constant(i)), ">=", true)
                | (Some(Integral::Constant(i)), "<", false) => Some(i),
                (Some(Integral::Constant(0)), "!==" | "!=", true)
                | (Some(Integral::Constant(0)), "===" | "==", false) => Some(1),
                (Some(Integral::Constant(i)), "===" | "==", true)
                | (Some(Integral::Constant(i)), "!==" | "!=", false) => Some(i),
                _ => None,
            };
            if let Some(minimum) = minimum {
                for alias in self
                    .flow
                    .arrays
                    .values_mut()
                    .filter(|a| a.root == fact.root)
                {
                    alias.minimum = alias.minimum.max(minimum);
                }
            }
            if (op == ">" && truth) || (op == "<=" && !truth) {
                if matches!(self.integral(other), Some(Integral::Nonnegative)) {
                    if let js::Expression::Identifier(i) = other {
                        if let Ok(binding) = self.binding(i.name.as_str(), i.span)
                            && !self.globals.contains_key(&binding.name)
                        {
                            let name = binding.name;
                            self.flow.bounds.insert((fact.root, name));
                        }
                    }
                }
            }
        }
    }
    pub(super) fn for_induction(&self, f: &js::ForStatement<'ast>) -> Option<InternedString> {
        let js::ForStatementInit::VariableDeclaration(init) = f.init.as_ref()? else {
            return None;
        };
        if init.declarations.len() != 1 {
            return None;
        }
        let d = &init.declarations[0];
        let js::BindingPattern::BindingIdentifier(id) = &d.id else {
            return None;
        };
        self.integral(d.init.as_ref()?)?;
        let binding = self.binding(id.name.as_str(), id.span).ok()?;
        if !binding.mutable {
            return None;
        }
        let update = f.update.as_ref()?;
        let advances = match update {
            js::Expression::UpdateExpression(u) if u.operator.as_str() == "++" => {
                matches!(&u.argument,js::SimpleAssignmentTarget::AssignmentTargetIdentifier(i) if i.name==id.name)
            }
            js::Expression::AssignmentExpression(a) if a.operator.as_str() == "+=" => {
                matches!(&a.left,js::AssignmentTarget::AssignmentTargetIdentifier(i) if i.name==id.name)
                    && matches!(&a.right,js::Expression::NumericLiteral(n) if n.value>0.0 && n.value <= u32::MAX as f64 && n.value.fract()==0.0)
            }
            _ => false,
        };
        if !advances {
            return None;
        }
        let body = f.body.span();
        if binding
            .symbol
            .and_then(|id| self.writes.get(&id))
            .is_some_and(|w| w.iter().any(|p| body.start <= *p && *p < body.end))
        {
            return None;
        }
        Some(binding.name)
    }
    pub(super) fn native_array_induction(
        &mut self,
        f: &js::ForStatement<'ast>,
        induction: Option<InternedString>,
        init: &mut [Stmt],
    ) -> Result<Option<(Expr, Expr)>> {
        let Some(name) = induction else {
            return Ok(None);
        };
        let Some(js::Expression::BinaryExpression(test)) = &f.test else {
            return Ok(None);
        };
        let operator = test.operator.as_str();
        // A length bound keeps the compare in integers; any other bound is a number the
        // counter is converted to for the compare.
        let (index, bound, source) = match operator {
            "<" => (&test.left, &test.right, Self::length_source(&test.right)),
            "<=" => (&test.left, &test.right, None),
            ">" => (&test.right, &test.left, Self::length_source(&test.left)),
            _ => return Ok(None),
        };
        let js::Expression::Identifier(id) = index else {
            return Ok(None);
        };
        if source.is_none() && !matches!(operator, "<" | "<=") {
            return Ok(None);
        }
        if self.binding(id.name.as_str(), id.span)?.name != name {
            return Ok(None);
        }
        let Some(Stmt {
            node: TypedStatement::Let(binding),
            ..
        }) = init.first_mut()
        else {
            return Ok(None);
        };
        let at = f.span;
        let integer = Type::Primitive(PrimitiveType::I64);
        let initializer = binding.initializer.take().unwrap();
        binding.initializer = Some(Box::new(node(
            TypedExpression::Cast(TypedCast {
                expr: initializer,
                target_type: integer.clone(),
            }),
            integer.clone(),
            at,
        )));
        binding.ty = integer.clone();
        for scope in self.scopes.iter_mut().rev() {
            if let Some(binding) = scope.values_mut().find(|b| b.name == name) {
                binding.ty = integer.clone();
                break;
            }
        }
        let condition = if let Some(source) = source {
            let array = self.expr(source)?;
            self.array_element(&array.ty, at)?;
            let length = node(
                TypedExpression::MethodCall(TypedMethodCall::positional(
                    array,
                    self.name("len"),
                    vec![],
                )),
                integer.clone(),
                at,
            );
            binary(
                BinaryOp::Lt,
                var(name, integer.clone(), at),
                length,
                boolean(),
                at,
            )
        } else {
            let limit = self.expr(bound)?;
            self.expect(&limit.ty, &number(), at)?;
            let counter = node(
                TypedExpression::Cast(TypedCast {
                    expr: Box::new(var(name, integer.clone(), at)),
                    target_type: number(),
                }),
                number(),
                at,
            );
            let compare = if operator == "<" {
                BinaryOp::Lt
            } else {
                BinaryOp::Le
            };
            binary(compare, counter, limit, boolean(), at)
        };
        let step = match f.update.as_ref().unwrap() {
            js::Expression::UpdateExpression(_) => 1,
            js::Expression::AssignmentExpression(a) => match &a.right {
                js::Expression::NumericLiteral(n) => n.value as i128,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        let one = node(
            TypedExpression::Literal(TypedLiteral::Integer(step)),
            integer.clone(),
            at,
        );
        let next = binary(
            BinaryOp::Add,
            var(name, integer.clone(), at),
            one,
            integer.clone(),
            at,
        );
        let update = binary(BinaryOp::Assign, var(name, integer, at), next, unit(), at);
        Ok(Some((condition, update)))
    }
    pub(super) fn set_induction(&mut self, name: Option<InternedString>) {
        if let Some(name) = name {
            self.flow.integers.insert(name, Integral::Nonnegative);
        }
    }
    /// `for (const x of array) body`: an index loop over a binding of the array, which reads the
    /// length each time round as the iterator does.
    pub(super) fn for_of(&mut self, f: &js::ForOfStatement<'ast>) -> Result<Vec<Stmt>> {
        let at = f.span;
        if f.r#await {
            return self.unsupported("for await", at);
        }
        let js::ForStatementLeft::VariableDeclaration(declaration) = &f.left else {
            return self.unsupported("for-of over an assignment target", at);
        };
        let kind = declaration.kind.as_str();
        let [declarator] = &declaration.declarations[..] else {
            return self.error(ErrorKind::Type, "for-of requires one binding", at);
        };
        let js::BindingPattern::BindingIdentifier(id) = &declarator.id else {
            return self.unsupported("destructuring in for-of", at);
        };
        if declarator.type_annotation.is_some() || declarator.init.is_some() {
            return self.error(
                ErrorKind::Type,
                "for-of binding cannot have a type annotation or initializer",
                declarator.span,
            );
        }
        if !matches!(kind, "let" | "const") {
            return self.unsupported("for-of over a var binding", at);
        }
        self.open_jump(statements::JumpKind::Loop);
        self.scopes.push(FxHashMap::default());
        let result = self.for_of_scoped(f, declaration, id, kind);
        self.pop_scope();
        result
    }

    fn for_of_scoped(
        &mut self,
        f: &js::ForOfStatement<'ast>,
        declaration: &js::VariableDeclaration<'ast>,
        id: &js::BindingIdentifier<'ast>,
        kind: &str,
    ) -> Result<Vec<Stmt>> {
        let at = f.span;
        self.reserve_variables(declaration)?;
        let iterable = self.expr(&f.right)?;
        if element_type(&iterable.ty).is_none()
            && (iterable.ty == string() || self.is_class_instance(&iterable.ty))
        {
            return self.unsupported("for-of over non-array iterator protocols", f.right.span());
        }
        let element = self.array_element(&iterable.ty, f.right.span())?;
        if matches!(element, Type::TypeVar(_)) {
            return self.unsupported(
                "generic array iteration before element ownership is known",
                at,
            );
        }
        let integer = Type::Primitive(PrimitiveType::I64);
        let array_ty = iterable.ty.clone();
        let array = self.fresh("iterable");
        let position = self.fresh("position");
        let array_value = || var(array, array_ty.clone(), at);
        let position_value = || var(position, integer.clone(), at);
        let let_ = |name: InternedString, ty: &Type, mutability: Mutability, value: Expr| {
            stmt(
                TypedStatement::Let(TypedLet {
                    name,
                    ty: ty.clone(),
                    mutability,
                    initializer: Some(Box::new(value)),
                    span: span(at),
                }),
                at,
            )
        };
        let init = vec![
            let_(array, &array_ty, Mutability::Immutable, iterable),
            let_(
                position,
                &integer,
                Mutability::Mutable,
                node(
                    TypedExpression::Literal(TypedLiteral::Integer(0)),
                    integer.clone(),
                    at,
                ),
            ),
        ];
        let length = node(
            TypedExpression::MethodCall(TypedMethodCall::positional(
                array_value(),
                self.name("len"),
                vec![],
            )),
            integer.clone(),
            at,
        );
        let condition = binary(BinaryOp::Lt, position_value(), length, boolean(), at);
        let one = node(
            TypedExpression::Literal(TypedLiteral::Integer(1)),
            integer.clone(),
            at,
        );
        let next = binary(BinaryOp::Add, position_value(), one, integer.clone(), at);
        let update = binary(BinaryOp::Assign, position_value(), next, unit(), at);

        let assigned = self.flow.assigned();
        let name = self.name(id.name.as_str());
        let unique = self.fresh(id.name.as_str());
        let view = self.class_by_layout(&element);
        self.scopes.last_mut().unwrap().insert(
            name,
            Binding {
                name: unique,
                symbol: id.symbol_id.get(),
                ty: element.clone(),
                view,
                mutable: kind == "let",
                initialized: true,
            },
        );
        self.clear_written_flow(f.span);
        self.set_induction(None);
        if element == string() {
            self.flow.strings.insert(unique);
        }
        self.loop_depth += 1;
        let body = self.block(&f.body);
        self.loop_depth -= 1;
        self.clear_written_flow(f.span);
        self.flow.restore_assigned(assigned);
        let mut body = body?;
        let mut after = vec![];
        self.close_jump(Some(&mut body), &mut after, at);
        let read = node(
            TypedExpression::Index(TypedIndex {
                object: Box::new(array_value()),
                index: Box::new(position_value()),
            }),
            element.clone(),
            at,
        );
        body.statements
            .insert(0, let_(unique, &element, Mutability::Mutable, read));
        let mut out = vec![stmt(
            TypedStatement::ForCStyle(TypedForCStyle {
                init: Some(Box::new(stmt(
                    TypedStatement::Block(TypedBlock {
                        statements: init,
                        span: span(at),
                    }),
                    at,
                ))),
                condition: Some(Box::new(condition)),
                update: Some(Box::new(update)),
                body,
                span: span(at),
            }),
            at,
        )];
        out.extend(after);
        Ok(out)
    }
    /// `new Array(n).fill(x)` and `Array(n).fill(x)`: `n` copies of `x`. A bare `new Array(n)`
    /// has holes TypeScript reads as `undefined`, which a native element cannot hold, so only the
    /// filled form is lowered. An array element is shared, as in JavaScript.
    pub(super) fn array_fill(
        &mut self,
        c: &js::CallExpression<'ast>,
        m: &js::StaticMemberExpression<'ast>,
        hint: Option<&Type>,
    ) -> Result<Option<Expr>> {
        if m.optional || c.optional || m.property.name != "fill" {
            return Ok(None);
        }
        let (callee, arguments, type_arguments) = match &m.object {
            js::Expression::NewExpression(n) => {
                (&n.callee, &n.arguments, n.type_arguments.as_deref())
            }
            js::Expression::CallExpression(k) if !k.optional => {
                (&k.callee, &k.arguments, k.type_arguments.as_deref())
            }
            _ => return Ok(None),
        };
        let js::Expression::Identifier(id) = callee else {
            return Ok(None);
        };
        let global = self.name("Array");
        if id.name != "Array"
            || self.scopes.iter().any(|s| s.contains_key(&global))
            || self.class_names.contains_key(&global)
            || self.imported_exports.contains_key(&global)
            || self.signatures.contains_key(&global)
            || self.constants.contains_key(&global)
            || self.namespace_references.contains_key(&id.node_id.get())
        {
            return Ok(None);
        }
        let at = c.span;
        if c.type_arguments.is_some() {
            return self.error(ErrorKind::Type, "fill takes no type arguments", at);
        }
        let ([length], [fill]) = (&arguments[..], &c.arguments[..]) else {
            return self.unsupported(
                "Array construction other than new Array(length).fill(value)",
                at,
            );
        };
        let (Some(length), Some(fill)) = (length.as_expression(), fill.as_expression()) else {
            return self.unsupported("spread array arguments", at);
        };
        let (declared, declared_view) = match type_arguments {
            Some(arguments) => match &arguments.params[..] {
                [element] => {
                    let (ty, view) = self.ty(element)?;
                    (Some(ty), view)
                }
                _ => return self.error(ErrorKind::Type, "Array takes one type argument", at),
            },
            None => (hint.and_then(element_type).cloned(), None),
        };
        let length_value = self.expr(length)?;
        if length_value.ty != number() {
            return self.unsupported(
                "Array construction with a non-numeric length",
                length.span(),
            );
        }
        let fill_value = match &declared {
            Some(element) => self.expr_expected(fill, element)?,
            None => self.expr(fill)?,
        };
        let element = declared.unwrap_or_else(|| fill_value.ty.clone());
        self.expect(&fill_value.ty, &element, fill.span())?;
        self.check_view(self.view_of(fill), declared_view, fill.span())?;
        self.check_array_element(&element, at)?;
        if matches!(element, Type::TypeVar(_)) {
            return self.unsupported(
                "generic array construction before element ownership is known",
                at,
            );
        }
        if element == string() && !self.immortal_string(fill) {
            return self.unsupported(
                "array storage of strings without a proven static lifetime",
                fill.span(),
            );
        }
        let zero_bytes = match fill.get_inner_expression() {
            js::Expression::NumericLiteral(n) => {
                element == number() && n.value == 0.0 && n.value.is_sign_positive()
            }
            js::Expression::BooleanLiteral(b) => element == boolean() && !b.value,
            _ => false,
        };

        let integer = Type::Primitive(PrimitiveType::I64);
        let literal = |value: i64| {
            node(
                TypedExpression::Literal(TypedLiteral::Integer(i128::from(value))),
                integer.clone(),
                at,
            )
        };
        let cast = |value: Expr, ty: &Type| {
            node(
                TypedExpression::Cast(TypedCast {
                    expr: Box::new(value),
                    target_type: ty.clone(),
                }),
                ty.clone(),
                at,
            )
        };
        let let_ = |name: InternedString, ty: &Type, mutability: Mutability, value: Expr| {
            stmt(
                TypedStatement::Let(TypedLet {
                    name,
                    ty: ty.clone(),
                    mutability,
                    initializer: Some(Box::new(value)),
                    span: span(at),
                }),
                at,
            )
        };
        let written = self.fresh("length");
        let count = self.fresh("count");
        let array_ty = array_type(element.clone());
        let array = self.fresh("filled");
        let written_value = || var(written, number(), at);
        let count_value = || var(count, integer.clone(), at);
        let mut statements = vec![
            let_(written, &number(), Mutability::Immutable, length_value),
            let_(
                count,
                &integer,
                Mutability::Immutable,
                cast(written_value(), &integer),
            ),
        ];
        // A length is a whole number from 0 to 2^32 - 1; the saturating conversion sends NaN and
        // out-of-range values to a count that fails the round trip or the bounds.
        let whole = binary(
            BinaryOp::Eq,
            cast(count_value(), &number()),
            written_value(),
            boolean(),
            at,
        );
        let in_range = binary(
            BinaryOp::And,
            binary(BinaryOp::Ge, count_value(), literal(0), boolean(), at),
            binary(
                BinaryOp::Le,
                count_value(),
                literal(4_294_967_295),
                boolean(),
                at,
            ),
            boolean(),
            at,
        );
        let valid = binary(BinaryOp::And, whole, in_range, boolean(), at);
        let invalid = node(
            TypedExpression::Unary(TypedUnary {
                op: UnaryOp::Not,
                operand: Box::new(valid),
            }),
            boolean(),
            at,
        );
        let trap = self.call("$TS$array_length_trap", vec![written_value()], unit(), at);
        statements.push(stmt(
            TypedStatement::If(TypedIf {
                condition: Box::new(invalid),
                then_block: TypedBlock {
                    statements: vec![expression(trap)],
                    span: span(at),
                },
                else_block: None,
                span: span(at),
            }),
            at,
        ));
        statements.push(let_(
            array,
            &array_ty,
            Mutability::Immutable,
            node(TypedExpression::Array(vec![]), array_ty.clone(), at),
        ));
        let array_value = || var(array, array_ty.clone(), at);
        let (push, resize_filled) = (self.name("push"), self.name("resize_filled"));
        let (fill_name, position) = (self.fresh("fill"), self.fresh("position"));
        let method = |name: InternedString, args: Vec<Expr>| {
            expression(node(
                TypedExpression::MethodCall(TypedMethodCall::positional(array_value(), name, args)),
                unit(),
                at,
            ))
        };
        if zero_bytes {
            statements.push(method(resize_filled, vec![count_value(), literal(0)]));
        } else {
            let position_value = || var(position, integer.clone(), at);
            statements.push(let_(fill_name, &element, Mutability::Immutable, fill_value));
            statements.push(let_(position, &integer, Mutability::Mutable, literal(0)));
            let next = binary(
                BinaryOp::Add,
                position_value(),
                literal(1),
                integer.clone(),
                at,
            );
            let advance = binary(BinaryOp::Assign, position_value(), next, unit(), at);
            statements.push(stmt(
                TypedStatement::While(TypedWhile {
                    condition: Box::new(binary(
                        BinaryOp::Lt,
                        position_value(),
                        count_value(),
                        boolean(),
                        at,
                    )),
                    body: TypedBlock {
                        statements: vec![
                            method(push, vec![var(fill_name, element.clone(), at)]),
                            expression(advance),
                        ],
                        span: span(at),
                    },
                    span: span(at),
                }),
                at,
            ));
        }
        statements.push(expression(array_value()));
        Ok(Some(node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(at),
            }),
            array_ty,
            at,
        )))
    }
    pub(super) fn array_method(
        &mut self,
        c: &js::CallExpression<'ast>,
        m: &js::StaticMemberExpression<'ast>,
    ) -> Result<Expr> {
        let fact = self.array_source(&m.object);
        let receiver = self.expr(&m.object)?;
        self.array_method_on(c, m, fact, receiver)
    }
    pub(super) fn array_method_on(
        &mut self,
        c: &js::CallExpression<'ast>,
        m: &js::StaticMemberExpression<'ast>,
        fact: Option<ArrayFact>,
        receiver: Expr,
    ) -> Result<Expr> {
        let element = self.array_element(&receiver.ty, c.span)?;
        if c.type_arguments.is_some() {
            return self.error(
                ErrorKind::Type,
                "array method takes no type arguments",
                c.span,
            );
        }
        if m.property.name != "push" {
            return self.unsupported("this array method", c.span);
        }
        if matches!(element, Type::TypeVar(_)) {
            return self.unsupported(
                "generic array mutation before element ownership is known",
                c.span,
            );
        }
        let receiver_name = self.fresh("array_receiver");
        let ty = receiver.ty.clone();
        let mut statements = vec![stmt(
            TypedStatement::Let(TypedLet {
                name: receiver_name,
                ty: ty.clone(),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(receiver)),
                span: span(c.span),
            }),
            c.span,
        )];
        let mut args = Vec::with_capacity(c.arguments.len());
        for argument in &c.arguments {
            let Some(source) = argument.as_expression() else {
                return self.unsupported("spread array arguments", c.span);
            };
            let value = self.expr_expected(source, &element)?;
            self.expect(&value.ty, &element, source.span())?;
            if element == string() && !self.immortal_string(source) {
                return self.unsupported(
                    "array storage of strings without a proven static lifetime",
                    source.span(),
                );
            }
            let name = self.fresh("array_argument");
            statements.push(stmt(
                TypedStatement::Let(TypedLet {
                    name,
                    ty: element.clone(),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(value)),
                    span: span(source.span()),
                }),
                source.span(),
            ));
            args.push(var(name, element.clone(), source.span()));
        }
        if let Some(fact) = fact {
            for alias in self
                .flow
                .arrays
                .values_mut()
                .filter(|a| a.root == fact.root)
            {
                alias.minimum = alias.minimum.saturating_add(args.len() as u64);
            }
        }
        for value in args {
            statements.push(expression(node(
                TypedExpression::MethodCall(TypedMethodCall::positional(
                    var(receiver_name, ty.clone(), c.span),
                    self.name("push"),
                    vec![value],
                )),
                unit(),
                c.span,
            )));
        }
        statements.push(expression(
            self.array_length(var(receiver_name, ty, c.span), c.span),
        ));
        Ok(node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(c.span),
            }),
            number(),
            c.span,
        ))
    }
}
