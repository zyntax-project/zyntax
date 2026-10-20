//! The names a program's declarations refer to.
//!
//! An import that names its items brings only those of the module's
//! functions, so a frontend that builds the import after lowering asks
//! here what its declarations name. Every name the code holds is
//! reported, the ones it declares included: a name that is not a
//! function of the module brings nothing. Types are not walked; they
//! name types, never functions.

use crate::arena::InternedString;
use crate::typed_ast::*;

/// Call `note` with every name in `declarations`, nested ones included.
pub fn for_each_name(
    declarations: &[TypedNode<TypedDeclaration>],
    note: &mut impl FnMut(InternedString),
) {
    let mut walk = Walk { note };
    for declaration in declarations {
        walk.declaration(&declaration.node);
    }
}

struct Walk<'a, F: FnMut(InternedString)> {
    note: &'a mut F,
}

impl<F: FnMut(InternedString)> Walk<'_, F> {
    fn name(&mut self, name: InternedString) {
        (self.note)(name);
    }

    fn declaration(&mut self, declaration: &TypedDeclaration) {
        match declaration {
            TypedDeclaration::Function(f) => self.function(f),
            TypedDeclaration::Initializer(i) => {
                self.annotations(&i.annotations);
                if let Some(module) = i.module {
                    self.name(module);
                }
                self.block(&i.body);
            }
            TypedDeclaration::Variable(v) => {
                self.name(v.name);
                self.opt_expr(&v.initializer);
            }
            TypedDeclaration::Class(c) => {
                self.name(c.name);
                self.annotations(&c.annotations);
                for field in &c.fields {
                    self.field(field);
                }
                for method in &c.methods {
                    self.method(method);
                }
                for constructor in &c.constructors {
                    self.method_params(&constructor.params);
                    self.block(&constructor.body);
                }
            }
            TypedDeclaration::Interface(i) => {
                self.name(i.name);
                for method in &i.methods {
                    self.name(method.name);
                    self.method_params(&method.params);
                }
            }
            TypedDeclaration::Impl(i) => {
                self.name(i.trait_name);
                for method in &i.methods {
                    self.method(method);
                }
            }
            TypedDeclaration::Enum(e) => {
                self.name(e.name);
                for variant in &e.variants {
                    self.name(variant.name);
                    if let TypedVariantFields::Named(fields) = &variant.fields {
                        for field in fields {
                            self.field(field);
                        }
                    }
                    self.opt_expr(&variant.discriminant);
                }
            }
            TypedDeclaration::TypeAlias(a) => self.name(a.name),
            TypedDeclaration::Module(m) => {
                self.name(m.name);
                for declaration in &m.declarations {
                    self.declaration(&declaration.node);
                }
            }
            TypedDeclaration::Import(_) => {}
            TypedDeclaration::Extern(e) => self.external(e),
            TypedDeclaration::Effect(e) => {
                self.name(e.name);
                for op in &e.operations {
                    self.name(op.name);
                    self.params(&op.params);
                }
            }
            TypedDeclaration::EffectHandler(h) => {
                self.name(h.name);
                self.name(h.effect_name);
                for field in &h.fields {
                    self.field(field);
                }
                for handler in &h.handlers {
                    self.name(handler.op_name);
                    self.params(&handler.params);
                    if let Some(body) = &handler.body {
                        self.block(body);
                    }
                }
            }
        }
    }

    fn function(&mut self, f: &TypedFunction) {
        self.name(f.name);
        self.annotations(&f.annotations);
        for name in f.effects.iter().chain(&f.with_handlers) {
            self.name(*name);
        }
        self.params(&f.params);
        if let Some(body) = &f.body {
            self.block(body);
        }
        if let Some(link) = f.link_name {
            self.name(link);
        }
    }

    fn method(&mut self, m: &TypedMethod) {
        self.name(m.name);
        self.annotations(&m.annotations);
        self.method_params(&m.params);
        if let Some(body) = &m.body {
            self.block(body);
        }
    }

    fn field(&mut self, field: &TypedField) {
        self.name(field.name);
        self.opt_expr(&field.initializer);
    }

    fn external(&mut self, e: &TypedExtern) {
        match e {
            TypedExtern::Class(c) => {
                self.name(c.name);
                self.name(c.runtime_prefix);
                for method in &c.methods {
                    self.name(method.name);
                    if let Some(symbol) = method.symbol {
                        self.name(symbol);
                    }
                    self.params(&method.params);
                }
                for property in &c.properties {
                    self.name(property.name);
                    for symbol in [property.getter_symbol, property.setter_symbol]
                        .into_iter()
                        .flatten()
                    {
                        self.name(symbol);
                    }
                }
            }
            TypedExtern::Struct(s) => {
                self.name(s.name);
                self.name(s.runtime_prefix);
            }
            TypedExtern::Enum(e) => {
                self.name(e.name);
                self.name(e.runtime_prefix);
                for variant in &e.variants {
                    self.name(variant.name);
                    self.name(variant.symbol);
                }
            }
            TypedExtern::TypeDef(t) => self.name(t.name),
        }
    }

    fn annotations(&mut self, annotations: &[TypedAnnotation]) {
        for annotation in annotations {
            self.name(annotation.name);
            for arg in &annotation.args {
                match arg {
                    TypedAnnotationArg::Positional(value) => self.annotation_value(value),
                    TypedAnnotationArg::Named { name, value } => {
                        self.name(*name);
                        self.annotation_value(value);
                    }
                }
            }
        }
    }

    fn annotation_value(&mut self, value: &TypedAnnotationValue) {
        match value {
            TypedAnnotationValue::String(name) | TypedAnnotationValue::Identifier(name) => {
                self.name(*name)
            }
            TypedAnnotationValue::List(values) => {
                for value in values {
                    self.annotation_value(value);
                }
            }
            TypedAnnotationValue::Integer(_)
            | TypedAnnotationValue::Float(_)
            | TypedAnnotationValue::Bool(_) => {}
        }
    }

    fn params(&mut self, params: &[TypedParameter]) {
        for param in params {
            self.name(param.name);
            self.opt_expr(&param.default_value);
            self.param_attributes(&param.attributes);
        }
    }

    fn method_params(&mut self, params: &[TypedMethodParam]) {
        for param in params {
            self.name(param.name);
            self.opt_expr(&param.default_value);
            self.param_attributes(&param.attributes);
        }
    }

    fn param_attributes(&mut self, attributes: &[ParameterAttribute]) {
        for attribute in attributes {
            self.name(attribute.name);
            self.exprs(&attribute.args);
        }
    }

    fn block(&mut self, block: &TypedBlock) {
        for statement in &block.statements {
            self.statement(&statement.node);
        }
    }

    fn statement(&mut self, statement: &TypedStatement) {
        match statement {
            TypedStatement::Continue => {}
            TypedStatement::Label(name) | TypedStatement::Goto(name) => self.name(*name),
            TypedStatement::Expression(e) | TypedStatement::Yield(e) | TypedStatement::Throw(e) => {
                self.expr(e)
            }
            TypedStatement::Return(e) | TypedStatement::Break(e) => self.opt_expr(e),
            TypedStatement::Let(l) => {
                self.name(l.name);
                self.opt_expr(&l.initializer);
            }
            TypedStatement::LetPattern(l) => {
                self.pattern(&l.pattern);
                self.expr(&l.initializer);
            }
            TypedStatement::If(i) => {
                self.expr(&i.condition);
                self.block(&i.then_block);
                if let Some(otherwise) = &i.else_block {
                    self.block(otherwise);
                }
            }
            TypedStatement::While(w) => {
                self.expr(&w.condition);
                self.block(&w.body);
            }
            TypedStatement::Block(b) => self.block(b),
            TypedStatement::For(f) => {
                self.pattern(&f.pattern);
                self.expr(&f.iterator);
                self.block(&f.body);
            }
            TypedStatement::ForCStyle(f) => {
                if let Some(init) = &f.init {
                    self.statement(&init.node);
                }
                self.opt_expr(&f.condition);
                self.opt_expr(&f.update);
                self.block(&f.body);
            }
            TypedStatement::Loop(l) => self.looped(l),
            TypedStatement::Match(m) => {
                self.expr(&m.scrutinee);
                self.arms(&m.arms);
            }
            TypedStatement::Try(t) => {
                self.block(&t.body);
                for catch in &t.catch_clauses {
                    self.pattern(&catch.pattern);
                    self.block(&catch.body);
                }
                if let Some(finally) = &t.finally_block {
                    self.block(finally);
                }
            }
            TypedStatement::Coroutine(c) => {
                if let CoroutineKind::Custom { name } = &c.kind {
                    self.name(*name);
                }
                self.expr(&c.body);
                self.exprs(&c.params);
            }
            TypedStatement::Defer(d) => self.expr(&d.body),
            TypedStatement::Select(s) => {
                for arm in &s.arms {
                    match &arm.operation {
                        TypedSelectOperation::Receive { channel, pattern } => {
                            self.expr(channel);
                            if let Some(pattern) = pattern {
                                self.pattern(pattern);
                            }
                        }
                        TypedSelectOperation::Send { channel, value } => {
                            self.expr(channel);
                            self.expr(value);
                        }
                        TypedSelectOperation::Timeout { duration } => self.expr(duration),
                    }
                    self.block(&arm.body);
                }
                if let Some(default) = &s.default {
                    self.block(default);
                }
            }
            TypedStatement::With(w) => {
                for handler in &w.handlers {
                    self.name(handler.name);
                    self.exprs(&handler.args);
                }
                self.block(&w.body);
            }
        }
    }

    fn looped(&mut self, l: &TypedLoop) {
        match l {
            TypedLoop::ForEach {
                pattern,
                iterator,
                body,
            } => {
                self.pattern(pattern);
                self.expr(iterator);
                self.block(body);
            }
            TypedLoop::ForCStyle {
                init,
                condition,
                update,
                body,
            } => {
                if let Some(init) = init {
                    self.statement(&init.node);
                }
                self.opt_expr(condition);
                self.opt_expr(update);
                self.block(body);
            }
            TypedLoop::While { condition, body } | TypedLoop::DoWhile { body, condition } => {
                self.expr(condition);
                self.block(body);
            }
            TypedLoop::Infinite { body } => self.block(body),
        }
    }

    fn arms(&mut self, arms: &[TypedMatchArm]) {
        for arm in arms {
            self.pattern(&arm.pattern);
            self.opt_expr(&arm.guard);
            self.expr(&arm.body);
        }
    }

    fn exprs(&mut self, exprs: &[TypedNode<TypedExpression>]) {
        for e in exprs {
            self.expr(e);
        }
    }

    fn opt_expr(&mut self, e: &Option<Box<TypedNode<TypedExpression>>>) {
        if let Some(e) = e {
            self.expr(e);
        }
    }

    fn named_args(&mut self, args: &[TypedNamedArg]) {
        for arg in args {
            self.name(arg.name);
            self.expr(&arg.value);
        }
    }

    fn expr(&mut self, e: &TypedNode<TypedExpression>) {
        match &e.node {
            TypedExpression::Literal(TypedLiteral::String(text)) => self.name(*text),
            TypedExpression::Literal(_) => {}
            TypedExpression::Variable(name) => self.name(*name),
            TypedExpression::Binary(b) => {
                self.expr(&b.left);
                self.expr(&b.right);
            }
            TypedExpression::Unary(u) => self.expr(&u.operand),
            TypedExpression::Call(c) => {
                self.expr(&c.callee);
                self.exprs(&c.positional_args);
                self.named_args(&c.named_args);
            }
            TypedExpression::Field(f) => {
                self.expr(&f.object);
                self.name(f.field);
            }
            TypedExpression::Index(i) => {
                self.expr(&i.object);
                self.expr(&i.index);
            }
            TypedExpression::Array(items) | TypedExpression::Tuple(items) => self.exprs(items),
            TypedExpression::Struct(s) => {
                self.name(s.name);
                for field in &s.fields {
                    self.name(field.name);
                    self.expr(&field.value);
                }
            }
            TypedExpression::Lambda(l) => {
                for param in &l.params {
                    self.name(param.name);
                }
                for capture in &l.captures {
                    self.name(capture.name);
                }
                match &l.body {
                    TypedLambdaBody::Expression(e) => self.expr(e),
                    TypedLambdaBody::Block(b) => self.block(b),
                }
            }
            TypedExpression::Match(m) => {
                self.expr(&m.scrutinee);
                self.arms(&m.arms);
            }
            TypedExpression::If(i) => {
                self.expr(&i.condition);
                self.expr(&i.then_branch);
                self.expr(&i.else_branch);
            }
            TypedExpression::Cast(c) => self.expr(&c.expr),
            TypedExpression::Await(e)
            | TypedExpression::Try(e)
            | TypedExpression::Dereference(e) => self.expr(e),
            TypedExpression::Reference(r) => self.expr(&r.expr),
            TypedExpression::Range(r) => {
                self.opt_expr(&r.start);
                self.opt_expr(&r.end);
            }
            TypedExpression::MethodCall(m) => {
                self.expr(&m.receiver);
                self.name(m.method);
                self.exprs(&m.positional_args);
                self.named_args(&m.named_args);
            }
            TypedExpression::Block(b) => self.block(b),
            TypedExpression::ListComprehension(l) => {
                self.expr(&l.output_expr);
                self.name(l.variable);
                self.expr(&l.iterator);
                self.opt_expr(&l.condition);
            }
            TypedExpression::Slice(s) => {
                self.expr(&s.object);
                self.opt_expr(&s.start);
                self.opt_expr(&s.end);
                self.opt_expr(&s.step);
            }
            TypedExpression::ImportModifier(m) => {
                self.name(m.loader);
                self.name(m.path);
                self.name(m.target_type);
            }
            TypedExpression::Compute(c) => {
                self.exprs(&c.args);
                for modifier in &c.modifiers {
                    self.name(modifier.name);
                    self.exprs(&modifier.positional_args);
                    self.named_args(&modifier.named_args);
                }
                for attr in &c.kernel_attrs {
                    self.name(attr.name);
                    self.exprs(&attr.positional_args);
                    self.named_args(&attr.named_args);
                }
                self.block(&c.body);
            }
            TypedExpression::Path(p) => {
                for segment in &p.segments {
                    self.name(*segment);
                }
            }
        }
    }

    fn pattern(&mut self, p: &TypedNode<TypedPattern>) {
        match &p.node {
            TypedPattern::Wildcard | TypedPattern::Literal(_) | TypedPattern::Range { .. } => {}
            TypedPattern::Identifier { name, .. }
            | TypedPattern::Constant(name)
            | TypedPattern::Regex { pattern: name, .. } => self.name(*name),
            TypedPattern::Rest { name, .. } => {
                if let Some(name) = name {
                    self.name(*name);
                }
            }
            TypedPattern::Tuple(items) | TypedPattern::Array(items) | TypedPattern::Or(items) => {
                for item in items {
                    self.pattern(item);
                }
            }
            TypedPattern::Struct { name, fields } => {
                self.name(*name);
                for field in fields {
                    self.name(field.name);
                    self.pattern(&field.pattern);
                }
            }
            TypedPattern::Enum {
                name,
                variant,
                fields,
            } => {
                self.name(*name);
                self.name(*variant);
                for field in fields {
                    self.pattern(field);
                }
            }
            TypedPattern::Slice {
                prefix,
                middle,
                suffix,
            } => {
                for item in prefix.iter().chain(suffix) {
                    self.pattern(item);
                }
                if let Some(middle) = middle {
                    self.pattern(middle);
                }
            }
            TypedPattern::Reference { pattern, .. }
            | TypedPattern::Typed { pattern, .. }
            | TypedPattern::Constructor { pattern, .. }
            | TypedPattern::Error { pattern, .. } => self.pattern(pattern),
            TypedPattern::Box(inner) | TypedPattern::Lazy(inner) | TypedPattern::Async(inner) => {
                self.pattern(inner)
            }
            TypedPattern::Guard { pattern, condition }
            | TypedPattern::When { pattern, condition } => {
                self.pattern(pattern);
                self.expr(condition);
            }
            TypedPattern::At { name, pattern, .. } => {
                self.name(*name);
                self.pattern(pattern);
            }
            TypedPattern::Path { path, args } => {
                for segment in path {
                    self.name(*segment);
                }
                for arg in args.iter().flatten() {
                    self.pattern(arg);
                }
            }
            TypedPattern::Macro { name, args } => {
                self.name(*name);
                for arg in args {
                    self.pattern(arg);
                }
            }
            TypedPattern::Map(map) => {
                for entry in &map.entries {
                    match entry {
                        TypedMapPatternEntry::KeyValue { pattern, .. } => self.pattern(pattern),
                        TypedMapPatternEntry::Rest { name, .. } => {
                            if let Some(name) = name {
                                self.name(*name);
                            }
                        }
                    }
                }
            }
            TypedPattern::View {
                view_function,
                pattern,
            } => {
                self.expr(view_function);
                self.pattern(pattern);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Span;
    use crate::type_registry::Type;

    fn name(s: &str) -> InternedString {
        InternedString::new_global(s)
    }

    fn expr(e: TypedExpression) -> TypedNode<TypedExpression> {
        TypedNode::new(e, Type::Unknown, Span::default())
    }

    #[test]
    fn a_body_names_what_it_calls_reads_and_sends_to() {
        let call = expr(TypedExpression::Call(TypedCall::positional(
            expr(TypedExpression::Variable(name("helper"))),
            vec![expr(TypedExpression::Variable(name("x")))],
        )));
        let send = expr(TypedExpression::MethodCall(TypedMethodCall::positional(
            expr(TypedExpression::Variable(name("obj"))),
            name("len"),
            Vec::new(),
        )));
        let body = TypedBlock {
            statements: vec![
                TypedNode::new(
                    TypedStatement::Expression(Box::new(call)),
                    Type::Unknown,
                    Span::default(),
                ),
                TypedNode::new(
                    TypedStatement::Return(Some(Box::new(send))),
                    Type::Unknown,
                    Span::default(),
                ),
            ],
            span: Span::default(),
        };
        let function = TypedFunction {
            name: name("caller"),
            annotations: Vec::new(),
            effects: Vec::new(),
            with_handlers: Vec::new(),
            type_params: Vec::new(),
            params: Vec::new(),
            return_type: Type::Unknown,
            body: Some(body),
            visibility: Default::default(),
            is_async: false,
            is_fiber: false,
            is_pure: false,
            is_external: false,
            calling_convention: Default::default(),
            link_name: None,
            module: None,
        };
        let declarations = vec![TypedNode::new(
            TypedDeclaration::Function(function),
            Type::Unknown,
            Span::default(),
        )];
        let mut found = std::collections::HashSet::new();
        for_each_name(&declarations, &mut |n| {
            found.insert(n);
        });
        for expected in ["caller", "helper", "x", "obj", "len"] {
            assert!(found.contains(&name(expected)), "{expected} not found");
        }
    }
}
