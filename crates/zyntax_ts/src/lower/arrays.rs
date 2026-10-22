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
            js::Expression::Identifier(i) => self.binding(i.name.as_str(), i.span).is_ok_and(|b| {
                self.flow.strings.contains(&b.name)
                    || (matches!(b.ty, Type::Union(_))
                        && self.flow.get(&b.name) == Some(&ScalarFlow::String))
            }),
            js::Expression::ParenthesizedExpression(p) => self.immortal_string(&p.expression),
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
                if element_type(&b.ty).is_none() {
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
                .and_then(|b| self.flow.integers.get(&b.name).copied()),
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
        } else {
            self.expr(index)?
        };
        if !bounded {
            return self.unsupported(
                "array indexing without proven integer bounds (the result may be undefined)",
                at,
            );
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
                        if let Ok(binding) = self.binding(i.name.as_str(), i.span) {
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
        let (index, source) = match test.operator.as_str() {
            "<" => (&test.left, Self::length_source(&test.right)),
            ">" => (&test.right, Self::length_source(&test.left)),
            _ => return Ok(None),
        };
        let (js::Expression::Identifier(id), Some(source)) = (index, source) else {
            return Ok(None);
        };
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
        let condition = binary(
            BinaryOp::Lt,
            var(name, integer.clone(), at),
            length,
            boolean(),
            at,
        );
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
    pub(super) fn array_method(
        &mut self,
        c: &js::CallExpression<'ast>,
        m: &js::StaticMemberExpression<'ast>,
    ) -> Result<Expr> {
        let fact = self.array_source(&m.object);
        let receiver = self.expr(&m.object)?;
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
