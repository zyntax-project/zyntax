use std::collections::{BTreeMap, HashMap, HashSet};

use oxc_allocator::Allocator;
use oxc_ast::ast as js;
use oxc_parser::Parser;
use oxc_span::{GetSpan, SourceType};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::{
    CallingConvention, FieldDef, InternedString, Mutability, NullabilityKind, PrimitiveType, Span,
    Type, TypeId, TypeMetadata, TypeRegistry, Visibility,
};

use crate::{Error, ErrorKind, Options, Soundness};

type Result<T> = std::result::Result<T, Error>;
type Expr = TypedNode<TypedExpression>;
type Stmt = TypedNode<TypedStatement>;

fn intern(name: &str) -> InternedString {
    InternedString::new_global(name)
}
fn span(at: oxc_span::Span) -> Span {
    Span::new(at.start as usize, at.end as usize)
}
fn node<T>(value: T, ty: Type, at: oxc_span::Span) -> TypedNode<T> {
    TypedNode::new(value, ty, span(at))
}
fn unit() -> Type {
    Type::Primitive(PrimitiveType::Unit)
}
fn number() -> Type {
    Type::Primitive(PrimitiveType::F64)
}
fn boolean() -> Type {
    Type::Primitive(PrimitiveType::Bool)
}
fn string() -> Type {
    Type::Primitive(PrimitiveType::String)
}
fn num(value: f64, at: oxc_span::Span) -> Expr {
    node(
        TypedExpression::Literal(TypedLiteral::Float(value)),
        number(),
        at,
    )
}
fn var(name: &str, ty: Type, at: oxc_span::Span) -> Expr {
    node(TypedExpression::Variable(intern(name)), ty, at)
}
fn stmt(value: TypedStatement, at: oxc_span::Span) -> Stmt {
    node(value, unit(), at)
}
fn expression(value: Expr) -> Stmt {
    let at = value.span;
    TypedNode::new(TypedStatement::Expression(Box::new(value)), unit(), at)
}
fn binary(op: BinaryOp, left: Expr, right: Expr, ty: Type, at: oxc_span::Span) -> Expr {
    node(
        TypedExpression::Binary(TypedBinary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }),
        ty,
        at,
    )
}
fn call(name: &str, args: Vec<Expr>, ret: Type, at: oxc_span::Span) -> Expr {
    node(
        TypedExpression::Call(TypedCall::positional(var(name, Type::Any, at), args)),
        ret,
        at,
    )
}
fn strict_fp() -> TypedAnnotation {
    TypedAnnotation {
        name: intern("strict_fp"),
        args: vec![],
        span: Span::default(),
    }
}

#[derive(Clone, Debug)]
struct Field {
    ty: Type,
    readonly: bool,
}
#[derive(Clone)]
struct Binding {
    name: String,
    ty: Type,
    view: Option<String>,
    mutable: bool,
    initialized: bool,
}
#[derive(Clone)]
struct Signature {
    params: Vec<(String, Type, Option<String>)>,
    ret: Type,
    ret_view: Option<String>,
}

/// Parse and type the supported TypeScript constructs without JavaScript emission.
pub fn parse_program(source: &str, file: &str, options: Options) -> Result<TypedProgram> {
    let allocator = Allocator::default();
    let parsed = Parser::new(&allocator, source, SourceType::ts()).parse();
    if let Some(error) = parsed.diagnostics.first() {
        return Err(Error::new(
            ErrorKind::Syntax,
            error.to_string(),
            oxc_span::Span::new(0, 0),
        ));
    }
    let semantic = oxc_semantic::SemanticBuilder::new()
        .with_check_syntax_error(true)
        .build(&parsed.program);
    if let Some(error) = semantic.diagnostics.first() {
        return Err(Error::new(
            ErrorKind::Type,
            error.to_string(),
            oxc_span::Span::new(0, 0),
        ));
    }
    Lowerer::new(options).program(&parsed.program, source, file)
}

struct Lowerer {
    options: Options,
    registry: TypeRegistry,
    aliases: HashMap<String, (Type, Option<String>)>,
    views: HashMap<String, BTreeMap<String, Field>>,
    layouts: Vec<(Vec<(String, Type)>, TypeId, String)>,
    signatures: HashMap<String, Signature>,
    scopes: Vec<HashMap<String, Binding>>,
    declarations: Vec<TypedNode<TypedDeclaration>>,
    returns: Option<(Type, Option<String>)>,
    loop_depth: usize,
    next_name: usize,
}

impl Lowerer {
    fn new(options: Options) -> Self {
        Self {
            options,
            registry: TypeRegistry::new(),
            aliases: HashMap::new(),
            views: HashMap::new(),
            layouts: vec![],
            signatures: HashMap::new(),
            scopes: vec![],
            declarations: vec![],
            returns: None,
            loop_depth: 0,
            next_name: 0,
        }
    }

    fn error<T>(
        &self,
        kind: ErrorKind,
        message: impl Into<String>,
        at: oxc_span::Span,
    ) -> Result<T> {
        Err(Error::new(kind, message, at))
    }
    fn unsupported<T>(&self, what: &str, at: oxc_span::Span) -> Result<T> {
        self.error(
            ErrorKind::Unsupported,
            format!("{what} is not implemented yet"),
            at,
        )
    }
    fn fresh(&mut self, name: &str) -> String {
        self.next_name += 1;
        format!("ts${name}${}", self.next_name)
    }

    fn program(
        mut self,
        program: &js::Program<'_>,
        source: &str,
        file: &str,
    ) -> Result<TypedProgram> {
        // Declare types before signatures; resolve forward aliases to a fixed point.
        let mut pending: Vec<_> = program
            .body
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    js::Statement::TSInterfaceDeclaration(_)
                        | js::Statement::TSTypeAliasDeclaration(_)
                )
            })
            .collect();
        while !pending.is_empty() {
            let before = pending.len();
            let mut next = vec![];
            let mut last = None;
            for item in pending {
                match self.type_declaration(item) {
                    Ok(()) => {}
                    Err(e) => {
                        last = Some(e);
                        next.push(item);
                    }
                }
            }
            if next.len() == before {
                return Err(last.unwrap());
            }
            pending = next;
        }
        for item in &program.body {
            if let js::Statement::FunctionDeclaration(f) = item {
                self.signature(f)?;
            }
        }
        self.externs();
        for item in &program.body {
            if let js::Statement::FunctionDeclaration(f) = item {
                self.function(f)?;
            }
        }
        self.scopes.push(HashMap::new());
        self.reserve(&program.body)?;
        let mut body = vec![];
        for item in &program.body {
            if !matches!(
                item,
                js::Statement::FunctionDeclaration(_)
                    | js::Statement::TSInterfaceDeclaration(_)
                    | js::Statement::TSTypeAliasDeclaration(_)
            ) {
                body.extend(self.statement(item)?);
            }
        }
        self.scopes.pop();
        self.declarations.push(node(
            TypedDeclaration::Function(TypedFunction {
                name: intern(crate::ENTRY),
                body: Some(TypedBlock {
                    statements: body,
                    span: span(program.span),
                }),
                return_type: unit(),
                annotations: vec![strict_fp()],
                visibility: Visibility::Public,
                ..Default::default()
            }),
            unit(),
            program.span,
        ));
        Ok(TypedProgram {
            declarations: self.declarations,
            language: Some(intern("typescript")),
            span: span(program.span),
            source_files: vec![zyntax_typed_ast::source::SourceFile::new(
                file.to_string(),
                source.to_string(),
            )],
            type_registry: self.registry,
        })
    }

    fn type_declaration(&mut self, item: &js::Statement<'_>) -> Result<()> {
        match item {
            js::Statement::TSInterfaceDeclaration(i) => {
                if i.type_parameters.is_some() || !i.extends.is_empty() {
                    return self.unsupported("generic or extending interfaces", i.span);
                }
                let name = i.id.name.to_string();
                if self.aliases.contains_key(&name) {
                    return self.unsupported("declaration merging", i.span);
                }
                let fields = self.fields(&i.body.body)?;
                let ty = self.layout(&fields);
                self.views.insert(name.clone(), fields);
                self.aliases.insert(name.clone(), (ty, Some(name)));
            }
            js::Statement::TSTypeAliasDeclaration(a) => {
                if a.type_parameters.is_some() {
                    return self.unsupported("generic aliases", a.span);
                }
                let name = a.id.name.to_string();
                if self.aliases.contains_key(&name) {
                    return self.error(ErrorKind::Type, "duplicate type declaration", a.span);
                }
                let (ty, view) = if let js::TSType::TSTypeLiteral(literal) = &a.type_annotation {
                    let fields = self.fields(&literal.members)?;
                    let ty = self.layout(&fields);
                    self.views.insert(name.clone(), fields);
                    (ty, Some(name.clone()))
                } else {
                    self.ty(&a.type_annotation)?
                };
                self.aliases.insert(name, (ty, view));
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn fields(&mut self, members: &[js::TSSignature<'_>]) -> Result<BTreeMap<String, Field>> {
        let mut fields = BTreeMap::new();
        for member in members {
            let js::TSSignature::TSPropertySignature(p) = member else {
                return self.unsupported("interface method or index signatures", member.span());
            };
            if p.optional || p.computed {
                return self.unsupported("optional or computed fields", p.span);
            }
            let name = self.key(&p.key)?;
            let Some(annotation) = &p.type_annotation else {
                return self.error(ErrorKind::Type, "field requires a type", p.span);
            };
            let (ty, _) = self.ty(&annotation.type_annotation)?;
            if !matches!(
                ty,
                Type::Primitive(PrimitiveType::F64 | PrimitiveType::Bool)
            ) {
                return self.unsupported("reference-valued object fields", p.span);
            }
            if fields
                .insert(
                    name,
                    Field {
                        ty,
                        readonly: p.readonly,
                    },
                )
                .is_some()
            {
                return self.error(ErrorKind::Type, "duplicate property", p.span);
            }
        }
        Ok(fields)
    }

    fn layout(&mut self, fields: &BTreeMap<String, Field>) -> Type {
        let key: Vec<_> = fields
            .iter()
            .map(|(n, f)| (n.clone(), f.ty.clone()))
            .collect();
        let id = if let Some((_, id, _)) = self.layouts.iter().find(|(k, _, _)| k == &key) {
            *id
        } else {
            let name = format!("ts$record${}", self.layouts.len());
            let definitions = key
                .iter()
                .map(|(n, ty)| FieldDef {
                    name: intern(n),
                    ty: ty.clone(),
                    visibility: Visibility::Public,
                    mutability: Mutability::Mutable,
                    is_static: false,
                    span: Span::default(),
                    getter: None,
                    setter: None,
                    is_synthetic: true,
                })
                .collect();
            let id = self.registry.register_struct_type(
                intern(&name),
                vec![],
                definitions,
                vec![],
                vec![],
                TypeMetadata {
                    is_reference: true,
                    ..Default::default()
                },
                Span::default(),
            );
            self.layouts.push((key, id, name));
            id
        };
        Type::Named {
            id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: NullabilityKind::NonNull,
        }
    }

    fn ty(&self, ty: &js::TSType<'_>) -> Result<(Type, Option<String>)> {
        Ok(match ty {
            js::TSType::TSNumberKeyword(_) => (number(), None),
            js::TSType::TSBooleanKeyword(_) => (boolean(), None),
            js::TSType::TSStringKeyword(_) => (string(), None),
            js::TSType::TSVoidKeyword(_) => (unit(), None),
            js::TSType::TSTypeReference(r) if r.type_arguments.is_none() => {
                let js::TSTypeName::IdentifierReference(name) = &r.type_name else {
                    return self.unsupported("qualified type names", r.span);
                };
                self.aliases
                    .get(name.name.as_str())
                    .cloned()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::Type,
                            format!("unknown type `{}`", name.name),
                            r.span,
                        )
                    })?
            }
            js::TSType::TSParenthesizedType(p) => self.ty(&p.type_annotation)?,
            _ => return self.unsupported("this type representation", ty.span()),
        })
    }

    fn key(&self, key: &js::PropertyKey<'_>) -> Result<String> {
        match key {
            js::PropertyKey::StaticIdentifier(i) => Ok(i.name.to_string()),
            js::PropertyKey::StringLiteral(s) if !s.lone_surrogates => Ok(s.value.to_string()),
            _ => self.unsupported("computed property names", key.span()),
        }
    }

    fn signature(&mut self, f: &js::Function<'_>) -> Result<()> {
        let Some(id) = &f.id else {
            return self.unsupported("anonymous declarations", f.span);
        };
        if f.r#async
            || f.generator
            || f.type_parameters.is_some()
            || f.params.rest.is_some()
            || f.this_param.is_some()
            || f.body.is_none()
        {
            return self.unsupported(
                "async, generator, generic, rest or ambient functions",
                f.span,
            );
        }
        if self.signatures.contains_key(id.name.as_str()) {
            return self.unsupported("function overloads", f.span);
        }
        let mut params = vec![];
        let mut names = HashSet::new();
        for p in &f.params.items {
            let js::BindingPattern::BindingIdentifier(name) = &p.pattern else {
                return self.unsupported("destructuring parameters", p.span);
            };
            if p.optional || p.initializer.is_some() {
                return self.unsupported("optional or default parameters", p.span);
            }
            let Some(annotation) = &p.type_annotation else {
                return self.error(ErrorKind::Type, "parameter implicitly has type any", p.span);
            };
            let (ty, view) = self.ty(&annotation.type_annotation)?;
            if !names.insert(name.name.to_string()) {
                return self.error(ErrorKind::Type, "duplicate parameter", p.span);
            }
            params.push((name.name.to_string(), ty, view));
        }
        let Some(ret) = &f.return_type else {
            return self.unsupported("inferred function return types", f.span);
        };
        let (ret, ret_view) = self.ty(&ret.type_annotation)?;
        self.signatures.insert(
            id.name.to_string(),
            Signature {
                params,
                ret,
                ret_view,
            },
        );
        Ok(())
    }

    fn function(&mut self, f: &js::Function<'_>) -> Result<()> {
        let name = f.id.as_ref().unwrap().name.as_str();
        let sig = self.signatures[name].clone();
        self.scopes.push(HashMap::new());
        let mut params = vec![];
        for (name, ty, view) in &sig.params {
            let unique = self.fresh(name);
            self.scopes.last_mut().unwrap().insert(
                name.clone(),
                Binding {
                    name: unique.clone(),
                    ty: ty.clone(),
                    view: view.clone(),
                    mutable: true,
                    initialized: true,
                },
            );
            let mut param = TypedParameter::regular(
                intern(&unique),
                ty.clone(),
                Mutability::Mutable,
                span(f.span),
            );
            param.ownership = ParamOwnership::Borrowed;
            params.push(param);
        }
        self.returns = Some((sig.ret.clone(), sig.ret_view.clone()));
        let mut statements = vec![];
        self.reserve(&f.body.as_ref().unwrap().statements)?;
        for s in &f.body.as_ref().unwrap().statements {
            statements.extend(self.statement(s)?);
        }
        if sig.ret != unit() && !Self::returns(&f.body.as_ref().unwrap().statements) {
            return self.error(ErrorKind::Type, "function lacks an ending return", f.span);
        }
        self.returns = None;
        self.scopes.pop();
        self.declarations.push(node(
            TypedDeclaration::Function(TypedFunction {
                name: intern(&format!("ts$fn${name}")),
                annotations: vec![strict_fp()],
                params,
                return_type: sig.ret,
                body: Some(TypedBlock {
                    statements,
                    span: span(f.span),
                }),
                visibility: Visibility::Private,
                ..Default::default()
            }),
            unit(),
            f.span,
        ));
        Ok(())
    }

    fn returns(statements: &[js::Statement<'_>]) -> bool {
        statements.iter().any(|s| match s {
            js::Statement::ReturnStatement(_) => true,
            js::Statement::BlockStatement(b) => Self::returns(&b.body),
            js::Statement::IfStatement(i) => i.alternate.as_ref().is_some_and(|other| {
                Self::returns(std::slice::from_ref(&i.consequent))
                    && Self::returns(std::slice::from_ref(other))
            }),
            _ => false,
        })
    }

    fn externs(&mut self) {
        for (name, params, ret) in [
            ("$TS$print_number", vec![number()], unit()),
            ("$TS$pow", vec![number(), number()], number()),
            ("$TS$space", vec![], unit()),
            ("$TS$newline", vec![], unit()),
            ("$IO$print", vec![string()], unit()),
            ("$IO$print_bool", vec![boolean()], unit()),
            ("$String$concat", vec![string(), string()], string()),
            ("$String$equals", vec![string(), string()], boolean()),
        ] {
            let params = params
                .into_iter()
                .enumerate()
                .map(|(i, ty)| {
                    let mut p = TypedParameter::regular(
                        intern(&format!("p{i}")),
                        ty,
                        Mutability::Immutable,
                        Span::default(),
                    );
                    p.ownership = ParamOwnership::Borrowed;
                    p
                })
                .collect();
            self.declarations.push(TypedNode::new(
                TypedDeclaration::Function(TypedFunction {
                    name: intern(name),
                    params,
                    return_type: ret,
                    is_external: true,
                    calling_convention: CallingConvention::Cdecl,
                    link_name: Some(intern(name)),
                    ..Default::default()
                }),
                unit(),
                Span::default(),
            ));
        }
    }

    fn reserve(&mut self, statements: &[js::Statement<'_>]) -> Result<()> {
        for statement in statements {
            if let js::Statement::VariableDeclaration(d) = statement {
                self.reserve_variables(d)?;
            }
        }
        Ok(())
    }

    fn reserve_variables(&mut self, declaration: &js::VariableDeclaration<'_>) -> Result<()> {
        for d in &declaration.declarations {
            if let js::BindingPattern::BindingIdentifier(id) = &d.id {
                let name = id.name.to_string();
                if self.scopes.last().unwrap().contains_key(&name) {
                    return self.error(ErrorKind::Type, "duplicate variable", d.span);
                }
                let unique = self.fresh(&name);
                self.scopes.last_mut().unwrap().insert(
                    name,
                    Binding {
                        name: unique,
                        ty: Type::Never,
                        view: None,
                        mutable: true,
                        initialized: false,
                    },
                );
            }
        }
        Ok(())
    }

    fn binding(&self, name: &str, at: oxc_span::Span) -> Result<Binding> {
        let b = self
            .scopes
            .iter()
            .rev()
            .find_map(|s| s.get(name))
            .cloned()
            .ok_or_else(|| Error::new(ErrorKind::Type, format!("cannot find name `{name}`"), at))?;
        if !b.initialized {
            return self.error(
                ErrorKind::Type,
                format!("`{name}` is used before initialization"),
                at,
            );
        }
        Ok(b)
    }

    fn expect(&self, actual: &Type, expected: &Type, at: oxc_span::Span) -> Result<()> {
        if actual == expected {
            Ok(())
        } else {
            if let (Type::Named { id: a, .. }, Type::Named { id: b, .. }) = (actual, expected) {
                if let (Some(a), Some(b)) = (
                    self.registry.get_type_by_id(*a),
                    self.registry.get_type_by_id(*b),
                ) {
                    if b.fields.iter().all(|needed| {
                        a.fields
                            .iter()
                            .any(|field| field.name == needed.name && field.ty == needed.ty)
                    }) {
                        return self
                            .unsupported("structural views with different physical layouts", at);
                    }
                }
            }
            self.error(
                ErrorKind::Type,
                format!("type {actual:?} is not assignable to {expected:?}"),
                at,
            )
        }
    }

    fn view_of(&self, e: &js::Expression<'_>) -> Option<String> {
        match e {
            js::Expression::Identifier(i) => self
                .binding(i.name.as_str(), i.span)
                .ok()
                .and_then(|b| b.view),
            js::Expression::CallExpression(c) => {
                if let js::Expression::Identifier(i) = &c.callee {
                    self.signatures
                        .get(i.name.as_str())
                        .and_then(|s| s.ret_view.clone())
                } else {
                    None
                }
            }
            js::Expression::ParenthesizedExpression(p) => self.view_of(&p.expression),
            js::Expression::TSNonNullExpression(n) => self.view_of(&n.expression),
            js::Expression::ConditionalExpression(c) => self.view_of(&c.consequent),
            js::Expression::TSAsExpression(a) => {
                self.ty(&a.type_annotation).ok().and_then(|(_, view)| view)
            }
            _ => None,
        }
    }

    fn check_view(&self, from: Option<&str>, to: Option<&str>, at: oxc_span::Span) -> Result<()> {
        if self.options.soundness == Soundness::Checked {
            if let Some(from) = from.and_then(|n| self.views.get(n)) {
                let to = to.and_then(|n| self.views.get(n));
                for (name, field) in from {
                    if field.readonly
                        && !to
                            .and_then(|fields| fields.get(name))
                            .is_some_and(|f| f.readonly)
                    {
                        return self.error(ErrorKind::Soundness, format!("readonly field `{name}` cannot become a writable alias in checked mode"), at);
                    }
                }
            }
        }
        Ok(())
    }

    fn block(&mut self, statement: &js::Statement<'_>) -> Result<TypedBlock> {
        self.scopes.push(HashMap::new());
        let statements = match statement {
            js::Statement::BlockStatement(b) => {
                self.reserve(&b.body)?;
                let mut out = vec![];
                for s in &b.body {
                    out.extend(self.statement(s)?);
                }
                out
            }
            s => self.statement(s)?,
        };
        self.scopes.pop();
        Ok(TypedBlock {
            statements,
            span: span(statement.span()),
        })
    }

    fn statement(&mut self, s: &js::Statement<'_>) -> Result<Vec<Stmt>> {
        let at = s.span();
        Ok(match s {
            js::Statement::EmptyStatement(_) => vec![],
            js::Statement::VariableDeclaration(d) => self.variables(d)?,
            js::Statement::ExpressionStatement(e) => vec![expression(self.expr(&e.expression)?)],
            js::Statement::BlockStatement(_) => {
                vec![stmt(TypedStatement::Block(self.block(s)?), at)]
            }
            js::Statement::ReturnStatement(r) => {
                let Some((ret, view)) = self.returns.clone() else {
                    return self.error(ErrorKind::Syntax, "return outside function", at);
                };
                let value = if let Some(arg) = &r.argument {
                    let value = self.expr(arg)?;
                    self.expect(&value.ty, &ret, at)?;
                    self.check_view(self.view_of(arg).as_deref(), view.as_deref(), at)?;
                    Some(Box::new(value))
                } else {
                    self.expect(&unit(), &ret, at)?;
                    None
                };
                vec![stmt(TypedStatement::Return(value), at)]
            }
            js::Statement::IfStatement(i) => {
                let condition = self.condition(&i.test)?;
                let then_block = self.block(&i.consequent)?;
                let else_block = i.alternate.as_ref().map(|s| self.block(s)).transpose()?;
                vec![stmt(
                    TypedStatement::If(TypedIf {
                        condition: Box::new(condition),
                        then_block,
                        else_block,
                        span: span(at),
                    }),
                    at,
                )]
            }
            js::Statement::WhileStatement(w) => {
                let condition = self.condition(&w.test)?;
                self.loop_depth += 1;
                let body = self.block(&w.body)?;
                self.loop_depth -= 1;
                vec![stmt(
                    TypedStatement::While(TypedWhile {
                        condition: Box::new(condition),
                        body,
                        span: span(at),
                    }),
                    at,
                )]
            }
            js::Statement::ForStatement(f) => {
                self.scopes.push(HashMap::new());
                let mut init = vec![];
                if let Some(value) = &f.init {
                    if let js::ForStatementInit::VariableDeclaration(d) = value {
                        self.reserve_variables(d)?;
                        init = self.variables(d)?;
                    } else if let Some(e) = value.as_expression() {
                        init.push(expression(self.expr(e)?));
                    } else {
                        return self.unsupported("for initializer", at);
                    }
                }
                let condition = f
                    .test
                    .as_ref()
                    .map(|e| self.condition(e))
                    .transpose()?
                    .map(Box::new);
                let update = f
                    .update
                    .as_ref()
                    .map(|e| self.expr(e))
                    .transpose()?
                    .map(Box::new);
                self.loop_depth += 1;
                let body = self.block(&f.body)?;
                self.loop_depth -= 1;
                self.scopes.pop();
                vec![stmt(
                    TypedStatement::ForCStyle(TypedForCStyle {
                        init: Some(Box::new(stmt(
                            TypedStatement::Block(TypedBlock {
                                statements: init,
                                span: span(at),
                            }),
                            at,
                        ))),
                        condition,
                        update,
                        body,
                        span: span(at),
                    }),
                    at,
                )]
            }
            js::Statement::BreakStatement(b) if b.label.is_none() && self.loop_depth > 0 => {
                vec![stmt(TypedStatement::Break(None), at)]
            }
            js::Statement::ContinueStatement(c) if c.label.is_none() && self.loop_depth > 0 => {
                vec![stmt(TypedStatement::Continue, at)]
            }
            _ => return self.unsupported("this statement", at),
        })
    }

    fn variables(&mut self, declaration: &js::VariableDeclaration<'_>) -> Result<Vec<Stmt>> {
        let kind = declaration.kind.as_str();
        if !matches!(kind, "let" | "const") {
            return self.unsupported("var or using declarations", declaration.span);
        }
        let mut out = vec![];
        for d in &declaration.declarations {
            let js::BindingPattern::BindingIdentifier(id) = &d.id else {
                return self.unsupported("destructuring declarations", d.span);
            };
            let name = id.name.to_string();
            if self
                .scopes
                .last()
                .unwrap()
                .get(&name)
                .is_some_and(|b| b.initialized)
            {
                return self.error(ErrorKind::Type, "duplicate variable", d.span);
            }
            let Some(init) = &d.init else {
                return self.unsupported("uninitialized bindings", d.span);
            };
            let value = self.expr(init)?;
            let (ty, view) = if let Some(annotation) = &d.type_annotation {
                self.ty(&annotation.type_annotation)?
            } else {
                (value.ty.clone(), self.view_of(init))
            };
            self.expect(&value.ty, &ty, d.span)?;
            self.check_view(self.view_of(init).as_deref(), view.as_deref(), d.span)?;
            let unique = self.fresh(&name);
            self.scopes.last_mut().unwrap().insert(
                name,
                Binding {
                    name: unique.clone(),
                    ty: ty.clone(),
                    view,
                    mutable: kind == "let",
                    initialized: true,
                },
            );
            out.push(stmt(
                TypedStatement::Let(TypedLet {
                    name: intern(&unique),
                    ty,
                    mutability: Mutability::Mutable,
                    initializer: Some(Box::new(value)),
                    span: span(d.span),
                }),
                d.span,
            ));
        }
        Ok(out)
    }

    fn condition(&mut self, e: &js::Expression<'_>) -> Result<Expr> {
        let value = self.expr(e)?;
        if value.ty == boolean() {
            return Ok(value);
        }
        if value.ty == number() {
            // The ordered self-comparison rejects NaN as well as zero.
            let unique = self.fresh("truthy");
            let at = e.span();
            let binding = stmt(
                TypedStatement::Let(TypedLet {
                    name: intern(&unique),
                    ty: number(),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(value)),
                    span: span(at),
                }),
                at,
            );
            let read = var(&unique, number(), at);
            let nonzero = binary(BinaryOp::Ne, read.clone(), num(0.0, at), boolean(), at);
            let ordered = binary(BinaryOp::Eq, read.clone(), read, boolean(), at);
            return Ok(node(
                TypedExpression::Block(TypedBlock {
                    statements: vec![
                        binding,
                        expression(binary(BinaryOp::And, nonzero, ordered, boolean(), at)),
                    ],
                    span: span(at),
                }),
                boolean(),
                at,
            ));
        }
        self.unsupported("truthiness of this type", e.span())
    }

    fn expr(&mut self, e: &js::Expression<'_>) -> Result<Expr> {
        let at = e.span();
        Ok(match e {
            js::Expression::NumericLiteral(n) => num(n.value, at),
            js::Expression::BooleanLiteral(b) => node(
                TypedExpression::Literal(TypedLiteral::Bool(b.value)),
                boolean(),
                at,
            ),
            js::Expression::StringLiteral(s) if !s.lone_surrogates => node(
                TypedExpression::Literal(TypedLiteral::String(intern(s.value.as_str()))),
                string(),
                at,
            ),
            js::Expression::Identifier(i) => {
                if self.scopes.iter().any(|s| s.contains_key(i.name.as_str())) {
                    let b = self.binding(i.name.as_str(), at)?;
                    var(&b.name, b.ty, at)
                } else {
                    match i.name.as_str() {
                        "NaN" => num(f64::NAN, at),
                        "Infinity" => num(f64::INFINITY, at),
                        name => {
                            let b = self.binding(name, at)?;
                            var(&b.name, b.ty, at)
                        }
                    }
                }
            }
            js::Expression::ParenthesizedExpression(p) => self.expr(&p.expression)?,
            js::Expression::UnaryExpression(u) => {
                if u.operator.as_str() == "!" {
                    node(
                        TypedExpression::Unary(TypedUnary {
                            op: UnaryOp::Not,
                            operand: Box::new(self.condition(&u.argument)?),
                        }),
                        boolean(),
                        at,
                    )
                } else {
                    let value = self.expr(&u.argument)?;
                    self.expect(&value.ty, &number(), at)?;
                    let op = match u.operator.as_str() {
                        "+" => UnaryOp::Plus,
                        "-" => UnaryOp::Minus,
                        _ => return self.unsupported("this unary operator", at),
                    };
                    node(
                        TypedExpression::Unary(TypedUnary {
                            op,
                            operand: Box::new(value),
                        }),
                        number(),
                        at,
                    )
                }
            }
            js::Expression::BinaryExpression(b) => self.binary_expr(b)?,
            js::Expression::LogicalExpression(l) => {
                let left = self.expr(&l.left)?;
                let right = self.expr(&l.right)?;
                self.expect(&left.ty, &boolean(), at)?;
                self.expect(&right.ty, &boolean(), at)?;
                let op = match l.operator.as_str() {
                    "&&" => BinaryOp::And,
                    "||" => BinaryOp::Or,
                    _ => return self.unsupported("nullish coalescing", at),
                };
                binary(op, left, right, boolean(), at)
            }
            js::Expression::ConditionalExpression(c) => {
                let condition = self.condition(&c.test)?;
                let a = self.expr(&c.consequent)?;
                let b = self.expr(&c.alternate)?;
                self.expect(&b.ty, &a.ty, at)?;
                if matches!(a.ty, Type::Named { .. })
                    && self.view_of(&c.consequent) != self.view_of(&c.alternate)
                {
                    return self.unsupported(
                        "conditional object views with different readonly fields",
                        at,
                    );
                }
                let ty = a.ty.clone();
                node(
                    TypedExpression::If(TypedIfExpr {
                        condition: Box::new(condition),
                        then_branch: Box::new(a),
                        else_branch: Box::new(b),
                    }),
                    ty,
                    at,
                )
            }
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                let object = self.expr(&m.object)?;
                let ty = self.field_type(&object.ty, m.property.name.as_str(), at)?;
                node(
                    TypedExpression::Field(TypedFieldAccess {
                        object: Box::new(object),
                        field: intern(m.property.name.as_str()),
                    }),
                    ty,
                    at,
                )
            }
            js::Expression::ObjectExpression(o) => self.object(o)?,
            js::Expression::CallExpression(c) if !c.optional => self.call_expr(c)?,
            js::Expression::AssignmentExpression(a) => self.assignment(a)?,
            js::Expression::UpdateExpression(u) => {
                let target = self.simple_target(&u.argument, at)?;
                self.expect(&target.ty, &number(), at)?;
                let (mut statements, target) = self.capture_target(target, at);
                let old = self.fresh("old");
                let save = stmt(
                    TypedStatement::Let(TypedLet {
                        name: intern(&old),
                        ty: number(),
                        mutability: Mutability::Immutable,
                        initializer: Some(Box::new(target.clone())),
                        span: span(at),
                    }),
                    at,
                );
                let value = binary(
                    if u.operator.as_str() == "++" {
                        BinaryOp::Add
                    } else {
                        BinaryOp::Sub
                    },
                    var(&old, number(), at),
                    num(1.0, at),
                    number(),
                    at,
                );
                let assign = binary(BinaryOp::Assign, target.clone(), value, unit(), at);
                let result = if u.prefix {
                    target
                } else {
                    var(&old, number(), at)
                };
                statements.extend([save, expression(assign), expression(result)]);
                node(
                    TypedExpression::Block(TypedBlock {
                        statements,
                        span: span(at),
                    }),
                    number(),
                    at,
                )
            }
            js::Expression::TSAsExpression(a) => {
                let value = self.expr(&a.expression)?;
                let (ty, view) = self.ty(&a.type_annotation)?;
                self.expect(&value.ty, &ty, at)?;
                self.check_view(self.view_of(&a.expression).as_deref(), view.as_deref(), at)?;
                value
            }
            js::Expression::TSNonNullExpression(n) => self.expr(&n.expression)?,
            _ => return self.unsupported("this expression", at),
        })
    }

    fn binary_expr(&mut self, b: &js::BinaryExpression<'_>) -> Result<Expr> {
        let at = b.span;
        let left = self.expr(&b.left)?;
        let right = self.expr(&b.right)?;
        let op = b.operator.as_str();
        if left.ty == string() && right.ty == string() {
            return match op {
                "+" => Ok(call("$String$concat", vec![left, right], string(), at)),
                "===" | "==" => Ok(call("$String$equals", vec![left, right], boolean(), at)),
                "!==" | "!=" => Ok(node(
                    TypedExpression::Unary(TypedUnary {
                        op: UnaryOp::Not,
                        operand: Box::new(call("$String$equals", vec![left, right], boolean(), at)),
                    }),
                    boolean(),
                    at,
                )),
                _ => self.unsupported("this string operator", at),
            };
        }
        if op == "**" {
            self.expect(&left.ty, &number(), at)?;
            self.expect(&right.ty, &number(), at)?;
            return Ok(call("$TS$pow", vec![left, right], number(), at));
        }
        if left.ty != right.ty {
            return self.unsupported("mixed-type operator coercion", at);
        }
        let (operator, ty) = match op {
            "+" => (BinaryOp::Add, number()),
            "-" => (BinaryOp::Sub, number()),
            "*" => (BinaryOp::Mul, number()),
            "/" => (BinaryOp::Div, number()),
            "%" => (BinaryOp::Rem, number()),
            "**" => (BinaryOp::Pow, number()),
            "===" | "==" => (BinaryOp::Eq, boolean()),
            "!==" | "!=" => (BinaryOp::Ne, boolean()),
            "<" => (BinaryOp::Lt, boolean()),
            "<=" => (BinaryOp::Le, boolean()),
            ">" => (BinaryOp::Gt, boolean()),
            ">=" => (BinaryOp::Ge, boolean()),
            _ => return self.unsupported("this binary operator", at),
        };
        if !matches!(op, "===" | "!==" | "==" | "!=") {
            self.expect(&left.ty, &number(), at)?;
        }
        Ok(binary(operator, left, right, ty, at))
    }

    fn field_type(&self, ty: &Type, field: &str, at: oxc_span::Span) -> Result<Type> {
        let Type::Named { id, .. } = ty else {
            return self.error(ErrorKind::Type, "receiver has no fields", at);
        };
        self.registry
            .get_type_by_id(*id)
            .and_then(|d| {
                d.fields
                    .iter()
                    .find(|f| f.name.resolve_global().as_deref() == Some(field))
            })
            .map(|f| f.ty.clone())
            .ok_or_else(|| Error::new(ErrorKind::Type, format!("no field `{field}`"), at))
    }

    fn object(&mut self, object: &js::ObjectExpression<'_>) -> Result<Expr> {
        let mut fields = BTreeMap::new();
        let mut values = vec![];
        let mut statements = vec![];
        for property in &object.properties {
            let js::ObjectPropertyKind::ObjectProperty(p) = property else {
                return self.unsupported("object spread", object.span);
            };
            if p.computed || p.method || p.kind != js::PropertyKind::Init {
                return self.unsupported("computed properties, getters or methods", p.span);
            }
            let name = self.key(&p.key)?;
            let value = self.expr(&p.value)?;
            if !matches!(
                value.ty,
                Type::Primitive(PrimitiveType::F64 | PrimitiveType::Bool)
            ) {
                return self.unsupported("reference-valued object fields", p.span);
            }
            if fields
                .insert(
                    name.clone(),
                    Field {
                        ty: value.ty.clone(),
                        readonly: false,
                    },
                )
                .is_some()
            {
                return self.unsupported("duplicate object keys", p.span);
            }
            let temporary = self.fresh("field");
            let ty = value.ty.clone();
            statements.push(stmt(
                TypedStatement::Let(TypedLet {
                    name: intern(&temporary),
                    ty: ty.clone(),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(value)),
                    span: span(p.span),
                }),
                p.span,
            ));
            values.push((name, var(&temporary, ty, p.span)));
        }
        let ty = self.layout(&fields);
        let Type::Named { id, .. } = &ty else {
            unreachable!()
        };
        let name = self.registry.get_type_by_id(*id).unwrap().name;
        // Evaluate in source order, then arrange values in the physical layout.
        values.sort_by(|a, b| a.0.cmp(&b.0));
        let fields = values
            .into_iter()
            .map(|(name, value)| TypedFieldInit {
                name: intern(&name),
                value: Box::new(value),
            })
            .collect();
        statements.push(expression(node(
            TypedExpression::Struct(TypedStructLiteral { name, fields }),
            ty.clone(),
            object.span,
        )));
        Ok(node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(object.span),
            }),
            ty,
            object.span,
        ))
    }

    fn call_expr(&mut self, c: &js::CallExpression<'_>) -> Result<Expr> {
        let at = c.span;
        if c.type_arguments.is_some() {
            return self.unsupported("generic calls", at);
        }
        let mut args = vec![];
        for arg in &c.arguments {
            let Some(e) = arg.as_expression() else {
                return self.unsupported("spread arguments", at);
            };
            args.push(self.expr(e)?);
        }
        if let js::Expression::StaticMemberExpression(m) = &c.callee {
            if let js::Expression::Identifier(receiver) = &m.object {
                if receiver.name == "console"
                    && m.property.name == "log"
                    && !self.scopes.iter().any(|s| s.contains_key("console"))
                {
                    return self.log(args, at);
                }
            }
        }
        let js::Expression::Identifier(id) = &c.callee else {
            return self.unsupported("indirect or receiver calls", at);
        };
        if self.scopes.iter().any(|s| s.contains_key(id.name.as_str())) {
            return self.error(ErrorKind::Type, "value is not callable", at);
        }
        let Some(sig) = self.signatures.get(id.name.as_str()).cloned() else {
            return self.error(
                ErrorKind::Type,
                format!("cannot find function `{}`", id.name),
                at,
            );
        };
        if args.len() != sig.params.len() {
            return self.error(ErrorKind::Type, "wrong argument count", at);
        }
        for ((arg, source), (_, expected, view)) in args.iter().zip(&c.arguments).zip(&sig.params) {
            self.expect(&arg.ty, expected, at)?;
            self.check_view(
                source
                    .as_expression()
                    .and_then(|e| self.view_of(e))
                    .as_deref(),
                view.as_deref(),
                at,
            )?;
        }
        Ok(call(&format!("ts$fn${}", id.name), args, sig.ret, at))
    }

    fn log(&mut self, args: Vec<Expr>, at: oxc_span::Span) -> Result<Expr> {
        let mut statements = vec![];
        let mut reads = vec![];
        for arg in args {
            let name = self.fresh("log");
            let ty = arg.ty.clone();
            statements.push(stmt(
                TypedStatement::Let(TypedLet {
                    name: intern(&name),
                    ty: ty.clone(),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(arg)),
                    span: span(at),
                }),
                at,
            ));
            reads.push(var(&name, ty, at));
        }
        for (i, value) in reads.into_iter().enumerate() {
            let symbol = if value.ty == number() {
                "$TS$print_number"
            } else if value.ty == boolean() {
                "$IO$print_bool"
            } else if value.ty == string() {
                "$IO$print"
            } else {
                return self.unsupported("console formatting of this type", at);
            };
            if i != 0 {
                statements.push(expression(call("$TS$space", vec![], unit(), at)));
            }
            statements.push(expression(call(symbol, vec![value], unit(), at)));
        }
        statements.push(expression(call("$TS$newline", vec![], unit(), at)));
        Ok(node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(at),
            }),
            unit(),
            at,
        ))
    }

    fn simple_target(
        &mut self,
        target: &js::SimpleAssignmentTarget<'_>,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        match target {
            js::SimpleAssignmentTarget::AssignmentTargetIdentifier(i) => {
                let b = self.binding(i.name.as_str(), at)?;
                if !b.mutable {
                    return self.error(ErrorKind::Type, "cannot assign to const", at);
                }
                Ok(var(&b.name, b.ty, at))
            }
            js::SimpleAssignmentTarget::StaticMemberExpression(m) => self.field_target(m),
            _ => self.unsupported("this assignment target", at),
        }
    }

    fn field_target(&mut self, m: &js::StaticMemberExpression<'_>) -> Result<Expr> {
        if self
            .view_of(&m.object)
            .and_then(|n| self.views.get(&n))
            .and_then(|fs| fs.get(m.property.name.as_str()))
            .is_some_and(|f| f.readonly)
        {
            return self.error(
                ErrorKind::Type,
                "cannot assign to readonly property",
                m.span,
            );
        }
        // Assignment receivers currently require a lexical reference view.
        if !matches!(m.object, js::Expression::Identifier(_)) {
            return self.unsupported("assignment through a computed receiver", m.span);
        }
        let object = self.expr(&m.object)?;
        let ty = self.field_type(&object.ty, m.property.name.as_str(), m.span)?;
        Ok(node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(object),
                field: intern(m.property.name.as_str()),
            }),
            ty,
            m.span,
        ))
    }

    fn assignment(&mut self, a: &js::AssignmentExpression<'_>) -> Result<Expr> {
        let target = match &a.left {
            js::AssignmentTarget::AssignmentTargetIdentifier(i) => {
                let b = self.binding(i.name.as_str(), a.span)?;
                if !b.mutable {
                    return self.error(ErrorKind::Type, "cannot assign to const", a.span);
                }
                var(&b.name, b.ty, a.span)
            }
            js::AssignmentTarget::StaticMemberExpression(m) => self.field_target(m)?,
            _ => return self.unsupported("this assignment target", a.span),
        };
        let (mut statements, target) = self.capture_target(target, a.span);
        let mut value = self.expr(&a.right)?;
        self.expect(&value.ty, &target.ty, a.span)?;
        if let js::AssignmentTarget::AssignmentTargetIdentifier(i) = &a.left {
            let b = self.binding(i.name.as_str(), a.span)?;
            self.check_view(self.view_of(&a.right).as_deref(), b.view.as_deref(), a.span)?;
        }
        let op = a.operator.as_str();
        if op != "=" {
            self.expect(&target.ty, &number(), a.span)?;
            let operation = match op {
                "+=" => BinaryOp::Add,
                "-=" => BinaryOp::Sub,
                "*=" => BinaryOp::Mul,
                "/=" => BinaryOp::Div,
                "%=" => BinaryOp::Rem,
                _ => return self.unsupported("this compound assignment", a.span),
            };
            value = binary(operation, target.clone(), value, number(), a.span);
        }
        let name = self.fresh("assigned");
        let ty = value.ty.clone();
        let save = stmt(
            TypedStatement::Let(TypedLet {
                name: intern(&name),
                ty: ty.clone(),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(value)),
                span: span(a.span),
            }),
            a.span,
        );
        let assignment = binary(
            BinaryOp::Assign,
            target,
            var(&name, ty.clone(), a.span),
            unit(),
            a.span,
        );
        statements.extend([
            save,
            expression(assignment),
            expression(var(&name, ty.clone(), a.span)),
        ]);
        Ok(node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(a.span),
            }),
            ty,
            a.span,
        ))
    }

    // A property reference captures its receiver before evaluating the right side.
    fn capture_target(&mut self, target: Expr, at: oxc_span::Span) -> (Vec<Stmt>, Expr) {
        let TypedExpression::Field(field) = &target.node else {
            return (vec![], target);
        };
        let name = self.fresh("receiver");
        let object = field.object.as_ref().clone();
        let ty = object.ty.clone();
        let save = stmt(
            TypedStatement::Let(TypedLet {
                name: intern(&name),
                ty: ty.clone(),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(object)),
                span: span(at),
            }),
            at,
        );
        let captured = node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(var(&name, ty, at)),
                field: field.field,
            }),
            target.ty,
            at,
        );
        (vec![save], captured)
    }
}
