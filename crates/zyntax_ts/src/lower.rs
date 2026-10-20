use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::rc::Rc;

use oxc_allocator::Allocator;
use oxc_ast::ast as js;
use oxc_ast_visit::{Visit, walk};
use oxc_parser::Parser;
use oxc_span::{GetSpan, SourceType};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::{
    CallingConvention, FieldDef, InternedString, Mutability, NullabilityKind, PrimitiveType, Span,
    Type, TypeId, TypeMetadata, TypeRegistry, Visibility,
};

use crate::{Error, ErrorKind, Options, Soundness};

mod arrays;
mod generics;
mod unions;
use arrays::{ArrayResult, FlowState, array_type, element_type};

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
// Only the empty variant is inhabited; `void` remains a distinct type.
fn undefined() -> Type {
    Type::Optional(Box::new(Type::Never))
}
fn is_undefined(ty: &Type) -> bool {
    matches!(ty, Type::Optional(inner) if **inner == Type::Never)
}
fn is_scalar(ty: &Type) -> bool {
    *ty == number() || *ty == boolean()
}
fn optional_join(a: &Type, b: &Type) -> Option<Type> {
    if a == b {
        return Some(a.clone());
    }
    if is_undefined(a) && is_scalar(b) {
        return Some(Type::Optional(Box::new(b.clone())));
    }
    if is_undefined(b) && is_scalar(a) {
        return Some(Type::Optional(Box::new(a.clone())));
    }
    for (optional, other) in [(a, b), (b, a)] {
        if let Type::Optional(inner) = optional {
            if inner.as_ref() == other || is_undefined(other) {
                return Some(optional.clone());
            }
        }
    }
    unions::primitive_join(a, b)
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarFlow {
    Number,
    Boolean,
    String,
    Undefined,
    Alternatives(u8),
}
impl ScalarFlow {
    fn of(ty: &Type) -> Option<Self> {
        if *ty == number() {
            Some(Self::Number)
        } else if *ty == boolean() {
            Some(Self::Boolean)
        } else if *ty == string() {
            Some(Self::String)
        } else if is_undefined(ty) {
            Some(Self::Undefined)
        } else if let Type::Union(members) = ty {
            let mask = members
                .iter()
                .try_fold(0, |mask, ty| Self::of(ty).map(|v| mask | v.mask()))?;
            Self::from_mask(mask)
        } else {
            None
        }
    }
    fn ty(self) -> Type {
        match self {
            Self::Number => number(),
            Self::Boolean => boolean(),
            Self::String => string(),
            Self::Undefined => undefined(),
            Self::Alternatives(mask) => unions::primitive_type(mask),
        }
    }
    fn mask(self) -> u8 {
        match self {
            Self::Number => 1,
            Self::Boolean => 2,
            Self::String => 4,
            Self::Undefined => 8,
            Self::Alternatives(mask) => mask,
        }
    }
    fn from_mask(mask: u8) -> Option<Self> {
        Some(match mask {
            0 => return None,
            1 => Self::Number,
            2 => Self::Boolean,
            4 => Self::String,
            8 => Self::Undefined,
            mask => Self::Alternatives(mask),
        })
    }
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
fn var(name: InternedString, ty: Type, at: oxc_span::Span) -> Expr {
    node(TypedExpression::Variable(name), ty, at)
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
fn call(name: InternedString, args: Vec<Expr>, ret: Type, at: oxc_span::Span) -> Expr {
    node(
        TypedExpression::Call(TypedCall::positional(var(name, Type::Any, at), args)),
        ret,
        at,
    )
}
fn strict_fp(name: InternedString) -> TypedAnnotation {
    TypedAnnotation {
        name,
        args: vec![],
        span: Span::default(),
    }
}

#[derive(Clone, Debug)]
struct Field {
    ty: PrimitiveType,
    readonly: bool,
}
struct Binding {
    name: InternedString,
    symbol: Option<oxc_semantic::SymbolId>,
    ty: Type,
    view: Option<InternedString>,
    mutable: bool,
    initialized: bool,
}
struct Signature<'ast> {
    native: InternedString,
    type_params: Vec<(InternedString, Type)>,
    params: Vec<(&'ast str, Type, Option<InternedString>)>,
    ret: Option<Type>,
    ret_view: Option<InternedString>,
    array_result: Option<ArrayResult>,
}

struct ReturnContext {
    expected: Option<(Type, Option<InternedString>)>,
    inferred: Option<(Type, Option<InternedString>)>,
    value_returned: bool,
    array_parameters: SmallVec<[Option<arrays::ArrayRoot>; 4]>,
    array_result: Option<ArrayResult>,
    unproven_string: bool,
}

struct WritePositions<'s> {
    scoping: &'s oxc_semantic::Scoping,
    positions: &'s mut FxHashMap<oxc_semantic::SymbolId, SmallVec<[u32; 2]>>,
}
impl<'ast> Visit<'ast> for WritePositions<'_> {
    fn visit_identifier_reference(&mut self, id: &js::IdentifierReference<'ast>) {
        let reference = self.scoping.get_reference(id.reference_id.get().unwrap());
        if reference.is_write() {
            if let Some(symbol) = reference.symbol_id() {
                self.positions
                    .entry(symbol)
                    .or_default()
                    .push(id.span.start);
            }
        }
    }
}

struct Dependency {
    name: InternedString,
    returned: bool,
}

struct CallDependencies<'s> {
    scoping: &'s oxc_semantic::Scoping,
    inferred: &'s FxHashMap<oxc_semantic::SymbolId, InternedString>,
    seen: FxHashSet<oxc_semantic::SymbolId>,
    callees: Vec<Dependency>,
    owner: Option<oxc_semantic::SymbolId>,
    skip_self: bool,
    in_return: bool,
    return_callees: FxHashSet<InternedString>,
    self_calls: bool,
    arrow_depth: usize,
}

impl CallDependencies<'_> {
    fn is_self_call(&self, mut expression: &js::Expression<'_>) -> bool {
        while let js::Expression::ParenthesizedExpression(p) = expression {
            expression = &p.expression;
        }
        let js::Expression::CallExpression(call) = expression else {
            return false;
        };
        let js::Expression::Identifier(id) = &call.callee else {
            return false;
        };
        self.scoping
            .get_reference(id.reference_id.get().unwrap())
            .symbol_id()
            == self.owner
    }
}

impl<'ast> Visit<'ast> for CallDependencies<'_> {
    fn visit_identifier_reference(&mut self, id: &js::IdentifierReference<'ast>) {
        let reference = self.scoping.get_reference(id.reference_id.get().unwrap());
        if !reference.is_value() {
            return;
        }
        if let Some(symbol) = reference.symbol_id() {
            if Some(symbol) == self.owner && self.skip_self {
                return;
            }
            if let Some(name) = self.inferred.get(&symbol) {
                if self.seen.insert(symbol) {
                    self.callees.push(Dependency {
                        name: *name,
                        returned: false,
                    });
                }
            }
        }
    }

    fn visit_call_expression(&mut self, call: &js::CallExpression<'ast>) {
        if self.in_return {
            let mut callee = &call.callee;
            loop {
                callee = match callee {
                    js::Expression::ParenthesizedExpression(p) => &p.expression,
                    js::Expression::TSNonNullExpression(n) => &n.expression,
                    _ => break,
                };
            }
            if let js::Expression::Identifier(id) = callee {
                if let Some(symbol) = self
                    .scoping
                    .get_reference(id.reference_id.get().unwrap())
                    .symbol_id()
                {
                    if !(self.skip_self && Some(symbol) == self.owner) {
                        if let Some(name) = self.inferred.get(&symbol) {
                            self.return_callees.insert(*name);
                        }
                    }
                }
            }
        }
        walk::walk_call_expression(self, call);
    }

    fn visit_function(&mut self, _: &js::Function<'ast>, _: oxc_semantic::ScopeFlags) {}
    fn visit_unary_expression(&mut self, unary: &js::UnaryExpression<'ast>) {
        let saved = self.in_return;
        if unary.operator.as_str() == "void" {
            self.in_return = false;
        }
        walk::walk_unary_expression(self, unary);
        self.in_return = saved;
    }

    fn visit_arrow_function_expression(&mut self, arrow: &js::ArrowFunctionExpression<'ast>) {
        let saved = (self.in_return, self.skip_self);
        self.in_return = false;
        self.skip_self = false;
        self.arrow_depth += 1;
        walk::walk_arrow_function_expression(self, arrow);
        self.arrow_depth -= 1;
        (self.in_return, self.skip_self) = saved;
    }

    fn visit_return_statement(&mut self, statement: &js::ReturnStatement<'ast>) {
        let direct_self = self.arrow_depth == 0
            && statement
                .argument
                .as_ref()
                .is_some_and(|e| self.is_self_call(e));
        let saved = (self.in_return, self.skip_self);
        self.in_return = self.arrow_depth == 0 && !direct_self;
        self.skip_self = direct_self;
        self.self_calls |= direct_self;
        walk::walk_return_statement(self, statement);
        (self.in_return, self.skip_self) = saved;
    }

    fn visit_expression_statement(&mut self, statement: &js::ExpressionStatement<'ast>) {
        let saved = self.skip_self;
        if self.arrow_depth == 0 && self.is_self_call(&statement.expression) {
            self.skip_self = true;
            self.self_calls = true;
        }
        walk::walk_expression_statement(self, statement);
        self.skip_self = saved;
    }
}

/// Parse and type the supported TypeScript constructs without JavaScript emission.
pub fn parse_program(source: &str, file: &str, options: Options) -> Result<TypedProgram> {
    let allocator = Allocator::default();
    let parsed = Parser::new(&allocator, source, SourceType::ts()).parse();
    if let Some(error) = parsed.diagnostics.first() {
        return Err(Error::from_oxc(ErrorKind::Syntax, error));
    }
    let semantic = oxc_semantic::SemanticBuilder::new()
        .with_check_syntax_error(true)
        .build(&parsed.program);
    if let Some(error) = semantic.diagnostics.first() {
        return Err(Error::from_oxc(ErrorKind::Type, error));
    }
    Lowerer::new(options).program(&parsed.program, &semantic.semantic, source, file)
}

struct Lowerer<'ast, 'program> {
    options: Options,
    names: RefCell<FxHashMap<&'ast str, InternedString>>,
    name_buffer: String,
    registry: TypeRegistry,
    aliases: FxHashMap<InternedString, (Type, Option<InternedString>)>,
    views: FxHashMap<InternedString, BTreeMap<&'ast str, Field>>,
    layouts: FxHashMap<Vec<(InternedString, PrimitiveType)>, TypeId>,
    compatible_layouts: FxHashMap<(TypeId, TypeId), bool>,
    signatures: FxHashMap<InternedString, Rc<Signature<'ast>>>,
    active_types: SmallVec<[(InternedString, Type); 2]>,
    functions: FxHashMap<InternedString, &'program js::Function<'ast>>,
    lowered: FxHashSet<InternedString>,
    self_recursive: FxHashSet<InternedString>,
    scopes: Vec<FxHashMap<InternedString, Binding>>,
    scope_floor: usize,
    flow: FlowState,
    writes: FxHashMap<oxc_semantic::SymbolId, SmallVec<[u32; 2]>>,
    write_offsets: Vec<u32>,
    declarations: Vec<TypedNode<TypedDeclaration>>,
    returns: Option<ReturnContext>,
    loop_depth: usize,
    next_name: usize,
    next_array_identity: usize,
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    fn new(options: Options) -> Self {
        Self {
            options,
            names: RefCell::new(FxHashMap::default()),
            name_buffer: String::new(),
            registry: TypeRegistry::new(),
            aliases: FxHashMap::default(),
            views: FxHashMap::default(),
            layouts: FxHashMap::default(),
            compatible_layouts: FxHashMap::default(),
            signatures: FxHashMap::default(),
            active_types: SmallVec::new(),
            functions: FxHashMap::default(),
            lowered: FxHashSet::default(),
            self_recursive: FxHashSet::default(),
            scopes: vec![],
            scope_floor: 0,
            flow: FlowState::default(),
            writes: FxHashMap::default(),
            write_offsets: Vec::new(),
            declarations: vec![],
            returns: None,
            loop_depth: 0,
            next_name: 0,
            next_array_identity: 0,
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
    fn name(&self, name: &'ast str) -> InternedString {
        *self
            .names
            .borrow_mut()
            .entry(name)
            .or_insert_with(|| intern(name))
    }

    fn generated_name(&mut self, prefix: &str, index: usize) -> InternedString {
        self.name_buffer.clear();
        write!(&mut self.name_buffer, "ts${prefix}${index}").unwrap();
        intern(&self.name_buffer)
    }

    fn fresh(&mut self, name: &str) -> InternedString {
        self.next_name += 1;
        self.generated_name(name, self.next_name)
    }

    fn call(&self, name: &'ast str, args: Vec<Expr>, ret: Type, at: oxc_span::Span) -> Expr {
        call(self.name(name), args, ret, at)
    }

    fn program(
        mut self,
        program: &'program js::Program<'ast>,
        semantic: &oxc_semantic::Semantic<'_>,
        source: &str,
        file: &str,
    ) -> Result<TypedProgram> {
        WritePositions {
            scoping: semantic.scoping(),
            positions: &mut self.writes,
        }
        .visit_program(program);
        self.write_offsets = self.writes.values().flatten().copied().collect();
        self.write_offsets.sort_unstable();
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
        let dependencies = self.dependencies(semantic)?;
        let mut visiting = FxHashSet::default();
        for item in &program.body {
            if let js::Statement::FunctionDeclaration(f) = item {
                let mut pending = vec![(self.name(f.id.as_ref().unwrap().name.as_str()), false)];
                while let Some((name, ready)) = pending.pop() {
                    if self.lowered.contains(&name) {
                        continue;
                    }
                    let function = self.functions[&name];
                    if ready {
                        self.function(function)?;
                        visiting.remove(&name);
                    } else {
                        if !visiting.insert(name) {
                            return self.unsupported("cyclic inferred return types", function.span);
                        }
                        pending.push((name, true));
                        if let Some(callees) = dependencies.get(&name) {
                            pending.extend(callees.iter().rev().map(|d| (d.name, false)));
                        }
                    }
                }
            }
        }
        self.scopes.push(FxHashMap::default());
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
        self.pop_scope();
        self.declarations.push(node(
            TypedDeclaration::Function(TypedFunction {
                name: intern(crate::ENTRY),
                body: Some(TypedBlock {
                    statements: body,
                    span: span(program.span),
                }),
                return_type: unit(),
                annotations: vec![strict_fp(self.name("strict_fp"))],
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

    fn dependencies(
        &mut self,
        semantic: &oxc_semantic::Semantic<'_>,
    ) -> Result<FxHashMap<InternedString, Vec<Dependency>>> {
        let inferred = self
            .functions
            .iter()
            .filter(|(name, _)| self.signatures[*name].ret.is_none())
            .map(|(name, f)| (f.id.as_ref().unwrap().symbol_id.get().unwrap(), *name))
            .collect();
        let mut calls = CallDependencies {
            scoping: semantic.scoping(),
            inferred: &inferred,
            seen: FxHashSet::default(),
            callees: vec![],
            owner: None,
            skip_self: false,
            in_return: false,
            return_callees: FxHashSet::default(),
            self_calls: false,
            arrow_depth: 0,
        };
        let mut dependencies = FxHashMap::default();
        for (name, function) in &self.functions {
            calls.seen.clear();
            calls.owner = function.id.as_ref().unwrap().symbol_id.get();
            calls.self_calls = false;
            calls.visit_function_body(function.body.as_ref().unwrap());
            if self.signatures[name].ret.is_none() {
                if calls.self_calls {
                    self.self_recursive.insert(*name);
                }
                for dependency in &mut calls.callees {
                    dependency.returned = calls.return_callees.contains(&dependency.name);
                }
            }
            calls.return_callees.clear();
            if !calls.callees.is_empty() {
                dependencies.insert(*name, std::mem::take(&mut calls.callees));
            }
        }
        let mut done = FxHashSet::default();
        let mut visiting = FxHashSet::default();
        for &name in dependencies.keys() {
            let mut pending = vec![(name, false)];
            while let Some((name, ready)) = pending.pop() {
                if done.contains(&name) {
                    continue;
                }
                if ready {
                    visiting.remove(&name);
                    done.insert(name);
                } else {
                    if !visiting.insert(name) {
                        let id = self.functions[&name].id.as_ref().unwrap();
                        return self.error(ErrorKind::Type,
                            format!("`{}` implicitly has return type any because its return expressions form a cycle; add a return type annotation", id.name), id.span);
                    }
                    pending.push((name, true));
                    if let Some(callees) = dependencies.get(&name) {
                        pending.extend(
                            callees
                                .iter()
                                .filter(|d| d.returned)
                                .map(|d| (d.name, false)),
                        );
                    }
                }
            }
        }
        Ok(dependencies)
    }

    fn type_declaration(&mut self, item: &js::Statement<'ast>) -> Result<()> {
        match item {
            js::Statement::TSInterfaceDeclaration(i) => {
                if i.type_parameters.is_some() || !i.extends.is_empty() {
                    return self.unsupported("generic or extending interfaces", i.span);
                }
                let name = self.name(i.id.name.as_str());
                if self.aliases.contains_key(&name) {
                    return self.unsupported("declaration merging", i.span);
                }
                let fields = self.fields(&i.body.body)?;
                let ty = self.layout(fields.iter().map(|(name, field)| (*name, field.ty)));
                self.views.insert(name, fields);
                self.aliases.insert(name, (ty, Some(name)));
            }
            js::Statement::TSTypeAliasDeclaration(a) => {
                if a.type_parameters.is_some() {
                    return self.unsupported("generic aliases", a.span);
                }
                let name = self.name(a.id.name.as_str());
                if self.aliases.contains_key(&name) {
                    return self.error(ErrorKind::Type, "duplicate type declaration", a.span);
                }
                let (ty, view) = if let js::TSType::TSTypeLiteral(literal) = &a.type_annotation {
                    let fields = self.fields(&literal.members)?;
                    let ty = self.layout(fields.iter().map(|(name, field)| (*name, field.ty)));
                    self.views.insert(name, fields);
                    (ty, Some(name))
                } else {
                    self.ty(&a.type_annotation)?
                };
                self.aliases.insert(name, (ty, view));
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn fields(&mut self, members: &[js::TSSignature<'ast>]) -> Result<BTreeMap<&'ast str, Field>> {
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
            let Type::Primitive(ty @ (PrimitiveType::F64 | PrimitiveType::Bool)) = ty else {
                return self.unsupported("reference-valued object fields", p.span);
            };
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

    fn layout(&mut self, fields: impl Iterator<Item = (&'ast str, PrimitiveType)>) -> Type {
        let key: SmallVec<[(InternedString, PrimitiveType); 4]> =
            fields.map(|(name, ty)| (self.name(name), ty)).collect();
        let id = if let Some(id) = self.layouts.get(key.as_slice()) {
            *id
        } else {
            let name = self.generated_name("record", self.layouts.len());
            let definitions = key
                .iter()
                .map(|(name, ty)| FieldDef {
                    name: *name,
                    ty: Type::Primitive(*ty),
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
                name,
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
            self.layouts.insert(key.into_vec(), id);
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

    fn ty(&self, ty: &js::TSType<'ast>) -> Result<(Type, Option<InternedString>)> {
        Ok(match ty {
            js::TSType::TSNumberKeyword(_) => (number(), None),
            js::TSType::TSBooleanKeyword(_) => (boolean(), None),
            js::TSType::TSStringKeyword(_) => (string(), None),
            js::TSType::TSVoidKeyword(_) => (unit(), None),
            js::TSType::TSUndefinedKeyword(_) => (undefined(), None),
            js::TSType::TSUnionType(u) => {
                let mut members = u.types.iter();
                let (mut ty, view) = self.ty(members.next().unwrap())?;
                if view.is_some() {
                    return self.unsupported("reference-bearing unions", u.span);
                }
                for member in members {
                    let (other, view) = self.ty(member)?;
                    if view.is_some() {
                        return self.unsupported("reference-bearing unions", u.span);
                    }
                    ty = optional_join(&ty, &other).ok_or_else(|| {
                        Error::new(
                            ErrorKind::Unsupported,
                            "this union payload combination is not implemented yet",
                            u.span,
                        )
                    })?;
                }
                (ty, None)
            }
            js::TSType::TSArrayType(a) => {
                let (element, _) = self.ty(&a.element_type)?;
                self.check_array_element(&element, a.span)?;
                (array_type(element), None)
            }
            js::TSType::TSTypeReference(r) if r.type_arguments.is_some() => {
                let js::TSTypeName::IdentifierReference(name) = &r.type_name else {
                    return self.unsupported("qualified type names", r.span);
                };
                if name.name != "Array" {
                    return self.unsupported("this generic type reference", r.span);
                }
                if r.type_arguments.as_ref().unwrap().params.len() != 1 {
                    return self.error(ErrorKind::Type, "Array requires one type argument", r.span);
                }
                let (element, _) = self.ty(&r.type_arguments.as_ref().unwrap().params[0])?;
                self.check_array_element(&element, r.span)?;
                (array_type(element), None)
            }
            js::TSType::TSFunctionType(f) => {
                if f.type_parameters.is_some() || f.this_param.is_some() || f.params.rest.is_some()
                {
                    return self.unsupported("generic, this or rest function types", f.span);
                }
                let mut params = Vec::with_capacity(f.params.items.len());
                for p in &f.params.items {
                    if p.optional || p.initializer.is_some() {
                        return self.unsupported("optional function parameters", p.span);
                    }
                    let Some(annotation) = &p.type_annotation else {
                        return self.error(
                            ErrorKind::Type,
                            "parameter implicitly has type any",
                            p.span,
                        );
                    };
                    params.push(self.ty(&annotation.type_annotation)?.0);
                }
                let ret = self.ty(&f.return_type.type_annotation)?.0;
                (self.function_type(params.into_iter(), ret, f.span)?, None)
            }
            js::TSType::TSTypeReference(r) if r.type_arguments.is_none() => {
                let js::TSTypeName::IdentifierReference(name) = &r.type_name else {
                    return self.unsupported("qualified type names", r.span);
                };
                let name = self.name(name.name.as_str());
                if let Some((_, ty)) = self.active_types.iter().find(|(n, _)| *n == name) {
                    return Ok((ty.clone(), None));
                }
                self.aliases.get(&name).cloned().ok_or_else(|| {
                    Error::new(
                        ErrorKind::Type,
                        format!(
                            "unknown type `{}`",
                            name.resolve_global().unwrap_or_default()
                        ),
                        r.span,
                    )
                })?
            }
            js::TSType::TSParenthesizedType(p) => self.ty(&p.type_annotation)?,
            _ => return self.unsupported("this type representation", ty.span()),
        })
    }

    fn key(&self, key: &js::PropertyKey<'ast>) -> Result<&'ast str> {
        match key {
            js::PropertyKey::StaticIdentifier(i) => Ok(i.name.as_str()),
            js::PropertyKey::StringLiteral(s) if !s.lone_surrogates => Ok(s.value.as_str()),
            _ => self.unsupported("computed property names", key.span()),
        }
    }

    fn signature(&mut self, f: &'program js::Function<'ast>) -> Result<()> {
        let Some(id) = &f.id else {
            return self.unsupported("anonymous declarations", f.span);
        };
        if f.r#async
            || f.generator
            || f.params.rest.is_some()
            || f.this_param.is_some()
            || f.body.is_none()
        {
            return self.unsupported("async, generator, rest or ambient functions", f.span);
        }
        if self.signatures.contains_key(&self.name(id.name.as_str())) {
            return self.unsupported("function overloads", f.span);
        }
        self.active_types.clear();
        let mut type_params = vec![];
        if let Some(parameters) = &f.type_parameters {
            for p in &parameters.params {
                if p.constraint.is_some() || p.default.is_some() || p.r#const {
                    return self
                        .unsupported("constrained, default or const type parameters", p.span);
                }
                let name = self.name(p.name.name.as_str());
                if self.active_types.iter().any(|(n, _)| *n == name) {
                    return self.error(ErrorKind::Type, "duplicate type parameter", p.span);
                }
                let ty = Type::TypeVar(zyntax_typed_ast::TypeVar::parameter(name));
                self.active_types.push((name, ty.clone()));
                type_params.push((name, ty));
            }
        }
        let mut params = vec![];
        let mut names = FxHashSet::default();
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
            if !names.insert(self.name(name.name.as_str())) {
                return self.error(ErrorKind::Type, "duplicate parameter", p.span);
            }
            params.push((name.name.as_str(), ty, view));
        }
        let (ret, ret_view) = match &f.return_type {
            Some(ret) => {
                let (ty, view) = self.ty(&ret.type_annotation)?;
                (Some(ty), view)
            }
            None => (None, None),
        };
        let name = self.name(id.name.as_str());
        self.functions.insert(name, f);
        self.name_buffer.clear();
        write!(&mut self.name_buffer, "ts$fn${}", id.name).unwrap();
        let native = intern(&self.name_buffer);
        self.signatures.insert(
            name,
            Rc::new(Signature {
                native,
                type_params,
                params,
                ret,
                ret_view,
                array_result: None,
            }),
        );
        self.active_types.clear();
        Ok(())
    }

    fn function(&mut self, f: &js::Function<'ast>) -> Result<()> {
        let name = self.name(f.id.as_ref().unwrap().name.as_str());
        if self.lowered.contains(&name) {
            return Ok(());
        }
        let result = self.function_body(f);
        if result.is_ok() {
            self.lowered.insert(name);
        }
        result
    }

    fn function_body(&mut self, f: &js::Function<'ast>) -> Result<()> {
        let name = self.name(f.id.as_ref().unwrap().name.as_str());
        let inferred = self.signatures[&name].ret.is_none();
        let recursive = inferred && self.self_recursive.contains(&name);
        if recursive {
            let sig = Rc::get_mut(self.signatures.get_mut(&name).unwrap()).unwrap();
            if !sig.type_params.is_empty() {
                return self.unsupported("recursive generic return inference", f.span);
            }
            // Direct self returns contribute no candidate to TypeScript inference.
            // Their types are settled after the other returns, without re-lowering.
            sig.ret = Some(Type::Never);
        }
        let sig = self.signatures[&name].clone();
        self.active_types.clear();
        self.active_types.extend(sig.type_params.iter().cloned());
        self.scopes.push(FxHashMap::default());
        let mut params = vec![];
        for ((name, ty, view), parameter) in sig.params.iter().zip(&f.params.items) {
            let js::BindingPattern::BindingIdentifier(id) = &parameter.pattern else {
                unreachable!()
            };
            let unique = self.fresh(name);
            let source_name = self.name(name);
            self.scopes.last_mut().unwrap().insert(
                source_name,
                Binding {
                    name: unique,
                    symbol: id.symbol_id.get(),
                    ty: ty.clone(),
                    view: *view,
                    mutable: true,
                    initialized: true,
                },
            );
            let mut param =
                TypedParameter::regular(unique, ty.clone(), Mutability::Mutable, span(f.span));
            param.ownership = ParamOwnership::Borrowed;
            params.push(param);
        }
        let array_parameters = params
            .iter()
            .map(|p| self.array_parameter(p.name, &p.ty))
            .collect();
        self.returns = Some(ReturnContext {
            expected: if inferred {
                None
            } else {
                sig.ret.clone().map(|ty| (ty, sig.ret_view))
            },
            inferred: None,
            value_returned: false,
            array_parameters,
            array_result: None,
            unproven_string: false,
        });
        let mut statements = vec![];
        self.reserve(&f.body.as_ref().unwrap().statements)?;
        for s in &f.body.as_ref().unwrap().statements {
            statements.extend(self.statement(s)?);
        }
        let context = self.returns.take().unwrap();
        let array_result = context.array_result;
        let (mut ret, ret_view) = match context.expected {
            Some(result) => result,
            None if !context.value_returned => (unit(), None),
            None => context.inferred.unwrap_or((unit(), None)),
        };
        let complete = Self::returns(&f.body.as_ref().unwrap().statements);
        if inferred && ret == Type::Never && !complete {
            ret = unit();
        }
        if ret != unit() && !complete {
            if inferred {
                ret = optional_join(&ret, &undefined()).ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unsupported,
                        "possibly undefined reference return types are not implemented yet",
                        f.span,
                    )
                })?;
            } else if !matches!(ret, Type::Optional(_)) {
                return self.error(ErrorKind::Type, "function lacks an ending return", f.span);
            }
            statements.push(stmt(TypedStatement::Return(None), f.span));
        }
        let native = sig.native;
        if recursive {
            Self::settle_self_calls(&mut statements, native, &ret);
        }
        self.normalize_returns(&mut statements, &ret)?;
        let type_params = sig
            .type_params
            .iter()
            .map(|(name, _)| TypedTypeParam {
                name: *name,
                span: span(f.span),
                ..Default::default()
            })
            .collect();
        drop(sig);
        let sig = Rc::get_mut(self.signatures.get_mut(&name).unwrap())
            .expect("signature handles do not escape body lowering");
        sig.ret = Some(ret.clone());
        sig.ret_view = ret_view;
        sig.array_result = array_result;
        self.pop_scope();
        self.active_types.clear();
        self.declarations.push(node(
            TypedDeclaration::Function(TypedFunction {
                name: native,
                type_params,
                annotations: vec![strict_fp(self.name("strict_fp"))],
                params,
                return_type: ret,
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

    fn record_return(
        &mut self,
        ty: &Type,
        view: Option<InternedString>,
        at: oxc_span::Span,
    ) -> Result<()> {
        let Some((first, first_view)) = &self.returns.as_ref().unwrap().inferred else {
            self.returns.as_mut().unwrap().inferred = Some((ty.clone(), view));
            return Ok(());
        };
        if *ty == Type::Never {
            return Ok(());
        }
        if *first == Type::Never {
            self.returns.as_mut().unwrap().inferred = Some((ty.clone(), view));
            return Ok(());
        }
        if first != ty {
            if let Some(joined) = optional_join(first, ty) {
                if matches!(joined, Type::Union(_))
                    && self.returns.as_ref().unwrap().unproven_string
                {
                    return self.unsupported(
                        "inferred union string payload without a proven static lifetime",
                        at,
                    );
                }
                self.returns.as_mut().unwrap().inferred = Some((joined, None));
                return Ok(());
            }
            return self.unsupported("inferred union return types", at);
        }
        let logical = |view: Option<InternedString>| {
            view.and_then(|name| self.aliases.get(&name))
                .map(|(ty, _)| ty)
        };
        if logical(*first_view).unwrap_or(first) != logical(view).unwrap_or(ty) {
            return self.unsupported("inferred returns with different property views", at);
        }
        let readonly = |view: Option<InternedString>| {
            view.and_then(|name| self.views.get(&name))
                .into_iter()
                .flat_map(|fields| fields.iter().filter(|(_, f)| f.readonly).map(|(n, _)| n))
        };
        if !readonly(*first_view).eq(readonly(view)) {
            return self.unsupported("inferred returns with different readonly views", at);
        }
        Ok(())
    }

    fn settle_self_calls(statements: &mut [Stmt], native: InternedString, result: &Type) {
        for statement in statements {
            match &mut statement.node {
                TypedStatement::Return(Some(e)) | TypedStatement::Expression(e) => {
                    if let TypedExpression::Call(call) = &e.node {
                        if matches!(call.callee.node, TypedExpression::Variable(name) if name == native)
                        {
                            e.ty = result.clone();
                        }
                    }
                }
                TypedStatement::Block(b) => {
                    Self::settle_self_calls(&mut b.statements, native, result)
                }
                TypedStatement::If(i) => {
                    Self::settle_self_calls(&mut i.then_block.statements, native, result);
                    if let Some(b) = &mut i.else_block {
                        Self::settle_self_calls(&mut b.statements, native, result);
                    }
                }
                TypedStatement::While(w) => {
                    Self::settle_self_calls(&mut w.body.statements, native, result)
                }
                TypedStatement::ForCStyle(f) => {
                    Self::settle_self_calls(&mut f.body.statements, native, result)
                }
                _ => {}
            }
        }
    }

    fn returns(statements: &[js::Statement<'ast>]) -> bool {
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
                        self.generated_name("p", i),
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
                    name: self.name(name),
                    params,
                    return_type: ret,
                    is_external: true,
                    calling_convention: CallingConvention::Cdecl,
                    link_name: Some(self.name(name)),
                    ..Default::default()
                }),
                unit(),
                Span::default(),
            ));
        }
    }

    fn reserve(&mut self, statements: &[js::Statement<'ast>]) -> Result<()> {
        for statement in statements {
            if let js::Statement::VariableDeclaration(d) = statement {
                self.reserve_variables(d)?;
            }
        }
        Ok(())
    }

    fn reserve_variables(&mut self, declaration: &js::VariableDeclaration<'ast>) -> Result<()> {
        for d in &declaration.declarations {
            if let js::BindingPattern::BindingIdentifier(id) = &d.id {
                let name = self.name(id.name.as_str());
                if self.scopes.last().unwrap().contains_key(&name) {
                    return self.error(ErrorKind::Type, "duplicate variable", d.span);
                }
                let unique = name;
                self.scopes.last_mut().unwrap().insert(
                    name,
                    Binding {
                        name: unique,
                        symbol: id.symbol_id.get(),
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

    fn binding(&self, name: &'ast str, at: oxc_span::Span) -> Result<&Binding> {
        let key = self.name(name);
        let (scope, b) = self
            .scopes
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, s)| s.get(&key).map(|b| (index, b)))
            .ok_or_else(|| Error::new(ErrorKind::Type, format!("cannot find name `{name}`"), at))?;
        if scope < self.scope_floor {
            return self.unsupported("lexical captures in arrow functions", at);
        }
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
        if actual == expected
            || unions::accepts(actual, expected)
            || (is_undefined(actual) && *expected == unit())
            || matches!(expected, Type::Optional(inner) if actual == inner.as_ref() || is_undefined(actual))
        {
            Ok(())
        } else {
            if *actual == Type::Never {
                return self.unsupported("diverging expressions used as values", at);
            }
            if let (
                Type::Function {
                    params: a,
                    return_type: ar,
                    ..
                },
                Type::Function {
                    params: b,
                    return_type: br,
                    ..
                },
            ) = (actual, expected)
            {
                if a == b && (**br == unit() || ar == br) {
                    return Ok(());
                }
                if a.len() < b.len() {
                    return self.unsupported("function parameter-count adaptation", at);
                }
            }
            if self.record_subtype(actual, expected) {
                return Ok(());
            }
            self.error(
                ErrorKind::Type,
                format!(
                    "type `{}` is not assignable to `{}`",
                    self.type_description(actual),
                    self.type_description(expected)
                ),
                at,
            )
        }
    }

    fn record_subtype(&self, actual: &Type, expected: &Type) -> bool {
        let (Type::Named { id: a, .. }, Type::Named { id: b, .. }) = (actual, expected) else {
            return false;
        };
        let (Some(a), Some(b)) = (
            self.registry.get_type_by_id(*a),
            self.registry.get_type_by_id(*b),
        ) else {
            return false;
        };
        a.metadata.is_reference
            && b.metadata.is_reference
            && b.fields.iter().all(|needed| {
                a.fields
                    .iter()
                    .any(|field| field.name == needed.name && field.ty == needed.ty)
            })
    }

    // Source views constrain properties; the expression still carries its concrete layout.
    fn source_type<'s>(&'s self, source: &js::Expression<'ast>, physical: &'s Type) -> &'s Type {
        self.view_of(source)
            .and_then(|view| self.aliases.get(&view))
            .map_or(physical, |(ty, _)| ty)
    }

    fn fresh_properties(&self, source: &js::Expression<'ast>, expected: &Type) -> Result<()> {
        let source = match source {
            js::Expression::ParenthesizedExpression(p) => {
                return self.fresh_properties(&p.expression, expected);
            }
            js::Expression::ConditionalExpression(c) => {
                self.fresh_properties(&c.consequent, expected)?;
                return self.fresh_properties(&c.alternate, expected);
            }
            js::Expression::TSNonNullExpression(n) => {
                return self.fresh_properties(&n.expression, expected);
            }
            js::Expression::ObjectExpression(object) => object,
            _ => return Ok(()),
        };
        let Type::Named { id, .. } = expected else {
            return Ok(());
        };
        let Some(definition) = self.registry.get_type_by_id(*id) else {
            return Ok(());
        };
        if definition.fields.is_empty() {
            return Ok(());
        }
        for property in &source.properties {
            if let js::ObjectPropertyKind::ObjectProperty(property) = property {
                let name = self.key(&property.key)?;
                if !definition.fields.iter().any(|f| f.name == self.name(name)) {
                    return self.error(
                        ErrorKind::Type,
                        format!("object literal has an excess property `{name}`"),
                        property.span,
                    );
                }
            }
        }
        Ok(())
    }

    fn adapt_value(&mut self, value: Expr, expected: &Type, at: oxc_span::Span) -> Result<Expr> {
        if value.ty != *expected {
            if let Type::Union(members) = expected {
                if matches!(value.ty, Type::Union(_)) {
                    return self.union_match(
                        value,
                        u8::MAX,
                        expected.clone(),
                        at,
                        |this, payload| this.adapt_value(payload, expected, at),
                    );
                }
                if members.contains(&value.ty) {
                    return Ok(node(
                        TypedExpression::Cast(TypedCast {
                            expr: Box::new(value),
                            target_type: expected.clone(),
                        }),
                        expected.clone(),
                        at,
                    ));
                }
            }
            if let Type::Optional(inner) = expected {
                if is_undefined(&value.ty) {
                    let none = var(self.name("None"), expected.clone(), at);
                    if matches!(value.node, TypedExpression::Variable(name) if name == self.name("None"))
                    {
                        return Ok(none);
                    }
                    return Ok(self.sequence(value, none, at));
                }
                if value.ty == **inner {
                    return Ok(call(self.name("Some"), vec![value], expected.clone(), at));
                }
            }
            if *expected == unit() && is_undefined(&value.ty) {
                let unit = node(TypedExpression::Literal(TypedLiteral::Unit), unit(), at);
                return Ok(self.sequence(value, unit, at));
            }
        }
        if value.ty == *expected || !matches!(expected, Type::Named { .. }) {
            return Ok(value);
        }
        let (
            Type::Named { id: actual, .. },
            Type::Named {
                id: expected_id, ..
            },
        ) = (&value.ty, expected)
        else {
            return self.unsupported("this record layout adaptation", at);
        };
        let key = (*actual, *expected_id);
        let compatible = if let Some(compatible) = self.compatible_layouts.get(&key) {
            *compatible
        } else {
            let offsets = |id| {
                use zyntax_compiler::{hir::HirType, object_header};
                let definition = self.registry.get_type_by_id(id).unwrap();
                let fields: SmallVec<[HirType; 5]> = std::iter::once(object_header::header_field())
                    .chain(definition.fields.iter().map(|field| match field.ty {
                        Type::Primitive(PrimitiveType::F64) => HirType::F64,
                        Type::Primitive(PrimitiveType::Bool) => HirType::Bool,
                        _ => unreachable!("TS records have scalar fields"),
                    }))
                    .collect();
                object_header::field_layout(&fields).0
            };
            let a_offsets = offsets(*actual);
            let b_offsets = offsets(*expected_id);
            let a = self.registry.get_type_by_id(*actual).unwrap();
            let b = self.registry.get_type_by_id(*expected_id).unwrap();
            let compatible = b.fields.iter().enumerate().all(|(index, needed)| {
                a.fields.iter().enumerate().any(|(i, field)| {
                    field.name == needed.name
                        && field.ty == needed.ty
                        && a_offsets[i + 1] == b_offsets[index + 1]
                })
            });
            self.compatible_layouts.insert(key, compatible);
            compatible
        };
        if !compatible {
            return self.unsupported(
                "record views requiring different field offsets at a call or return",
                at,
            );
        }
        Ok(node(
            TypedExpression::Cast(TypedCast {
                expr: Box::new(value),
                target_type: expected.clone(),
            }),
            expected.clone(),
            at,
        ))
    }

    fn sequence(&self, value: Expr, result: Expr, at: oxc_span::Span) -> Expr {
        let ty = result.ty.clone();
        node(
            TypedExpression::Block(TypedBlock {
                statements: vec![expression(value), expression(result)],
                span: span(at),
            }),
            ty,
            at,
        )
    }

    fn optional_match(
        &mut self,
        value: Expr,
        some: impl FnOnce(&mut Self, Expr) -> Result<Expr>,
        none: Option<Expr>,
        result: Type,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        let Type::Optional(inner) = &value.ty else {
            unreachable!()
        };
        let name = self.fresh("present");
        let body = some(self, var(name, inner.as_ref().clone(), at))?;
        let mut arms = Vec::with_capacity(1 + usize::from(none.is_some()));
        arms.push(TypedMatchArm {
            pattern: Box::new(node(
                TypedPattern::Enum {
                    name: self.name("Option"),
                    variant: self.name("Some"),
                    fields: vec![node(
                        TypedPattern::Identifier {
                            name,
                            mutability: Mutability::Immutable,
                        },
                        inner.as_ref().clone(),
                        at,
                    )],
                },
                value.ty.clone(),
                at,
            )),
            guard: None,
            body: Box::new(body),
        });
        if let Some(body) = none {
            arms.push(TypedMatchArm {
                pattern: Box::new(node(
                    TypedPattern::Enum {
                        name: self.name("Option"),
                        variant: self.name("None"),
                        fields: vec![],
                    },
                    value.ty.clone(),
                    at,
                )),
                guard: None,
                body: Box::new(body),
            });
        }
        Ok(node(
            TypedExpression::Match(TypedMatchExpr {
                scrutinee: Box::new(value),
                arms,
            }),
            result,
            at,
        ))
    }

    fn flow_read(
        &mut self,
        value: Expr,
        flow: Option<ScalarFlow>,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        let Some(flow) = flow else {
            return Ok(value);
        };
        if ScalarFlow::of(&value.ty) == Some(flow) {
            return Ok(value);
        }
        if flow == ScalarFlow::Undefined {
            return Ok(self.sequence(value, var(self.name("None"), undefined(), at), at));
        }
        if matches!(value.ty, Type::Union(_)) {
            let result = flow.ty();
            return self.union_match(value, flow.mask(), result.clone(), at, |this, payload| {
                this.adapt_value(payload, &result, at)
            });
        }
        self.optional_match(value, |_, payload| Ok(payload), None, flow.ty(), at)
    }

    fn normalize_returns(&mut self, statements: &mut [Stmt], result: &Type) -> Result<()> {
        for statement in statements {
            match &mut statement.node {
                TypedStatement::Return(value)
                    if matches!(result, Type::Optional(_) | Type::Union(_)) =>
                {
                    let at =
                        oxc_span::Span::new(statement.span.start as u32, statement.span.end as u32);
                    let returned = value
                        .take()
                        .map(|e| *e)
                        .unwrap_or_else(|| var(self.name("None"), undefined(), at));
                    *value = Some(Box::new(self.adapt_value(returned, result, at)?));
                }
                TypedStatement::Block(b) => self.normalize_returns(&mut b.statements, result)?,
                TypedStatement::If(i) => {
                    self.normalize_returns(&mut i.then_block.statements, result)?;
                    if let Some(b) = &mut i.else_block {
                        self.normalize_returns(&mut b.statements, result)?;
                    }
                }
                TypedStatement::While(w) => {
                    self.normalize_returns(&mut w.body.statements, result)?
                }
                TypedStatement::ForCStyle(f) => {
                    self.normalize_returns(&mut f.body.statements, result)?
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn typeof_value(&mut self, value: Expr, at: oxc_span::Span) -> Result<Expr> {
        if matches!(value.ty, Type::Union(_)) {
            return self.union_match(value, u8::MAX, string(), at, |this, payload| {
                this.typeof_value(payload, at)
            });
        }
        if matches!(value.ty, Type::Optional(_)) && !is_undefined(&value.ty) {
            let none = node(
                TypedExpression::Literal(TypedLiteral::String(self.name("undefined"))),
                string(),
                at,
            );
            return self.optional_match(
                value,
                |this, payload| this.typeof_value(payload, at),
                Some(none),
                string(),
                at,
            );
        }
        let text = if value.ty == number() {
            "number"
        } else if value.ty == boolean() {
            "boolean"
        } else if is_undefined(&value.ty) || value.ty == unit() {
            "undefined"
        } else if value.ty == string() {
            "string"
        } else if matches!(value.ty, Type::Function { .. }) {
            "function"
        } else if matches!(value.ty, Type::Named { .. } | Type::Array { .. }) {
            "object"
        } else {
            return self.unsupported("typeof this type", at);
        };
        let text = node(
            TypedExpression::Literal(TypedLiteral::String(self.name(text))),
            string(),
            at,
        );
        Ok(self.sequence(value, text, at))
    }

    fn format_value(&mut self, value: Expr, at: oxc_span::Span) -> Result<Expr> {
        if matches!(value.ty, Type::Union(_)) {
            return self.union_match(value, u8::MAX, unit(), at, |this, payload| {
                this.format_value(payload, at)
            });
        }
        if is_undefined(&value.ty) || value.ty == unit() {
            let print = self.call("$TS$print_undefined", vec![], unit(), at);
            return Ok(self.sequence(value, print, at));
        }
        if matches!(value.ty, Type::Optional(_)) {
            let none = self.call("$TS$print_undefined", vec![], unit(), at);
            return self.optional_match(
                value,
                |this, payload| this.format_value(payload, at),
                Some(none),
                unit(),
                at,
            );
        }
        let symbol = if value.ty == number() {
            "$TS$print_number"
        } else if value.ty == boolean() {
            "$IO$print_bool"
        } else if value.ty == string() {
            "$IO$print"
        } else {
            return self.unsupported("console formatting of this type", at);
        };
        Ok(self.call(symbol, vec![value], unit(), at))
    }

    fn pop_scope(&mut self) {
        for binding in self.scopes.pop().unwrap().values() {
            self.flow.forget(binding.name);
        }
    }

    fn clear_written_flow(&mut self, at: oxc_span::Span) {
        for scope in &self.scopes {
            for binding in scope.values() {
                if binding
                    .symbol
                    .and_then(|id| self.writes.get(&id))
                    .is_some_and(|writes| writes.iter().any(|p| at.start <= *p && *p < at.end))
                {
                    self.flow.forget(binding.name);
                }
            }
        }
    }

    fn has_binding_write(&self, at: oxc_span::Span) -> bool {
        let start = self.write_offsets.partition_point(|p| *p < at.start);
        self.write_offsets.get(start).is_some_and(|p| *p < at.end)
    }

    fn merge_flow(&mut self, other: &FlowState) {
        self.flow.merge(other);
    }

    fn global_undefined(&self, e: &js::Expression<'ast>) -> bool {
        match e {
            js::Expression::Identifier(i) if i.name == "undefined" => !self
                .scopes
                .iter()
                .any(|s| s.contains_key(&self.name("undefined"))),
            js::Expression::ParenthesizedExpression(p) => self.global_undefined(&p.expression),
            _ => false,
        }
    }

    fn narrow(&mut self, test: &js::Expression<'ast>, truth: bool) {
        // Replaying a guard after a write would refine a different value.
        if self.has_binding_write(test.span()) {
            return;
        }
        self.narrow_array_truthiness(test, truth);
        if let js::Expression::BinaryExpression(binary) = test {
            self.narrow_array(binary, truth);
        }
        match test {
            js::Expression::ParenthesizedExpression(p) => self.narrow(&p.expression, truth),
            js::Expression::UnaryExpression(u) if u.operator.as_str() == "!" => {
                self.narrow(&u.argument, !truth)
            }
            js::Expression::Identifier(i) if truth => self.narrow_name(i, true),
            js::Expression::LogicalExpression(l)
                if (truth && l.operator.as_str() == "&&")
                    || (!truth && l.operator.as_str() == "||") =>
            {
                self.narrow(&l.left, truth);
                self.clear_written_flow(l.right.span());
                self.narrow(&l.right, truth);
            }
            js::Expression::BinaryExpression(b)
                if matches!(b.operator.as_str(), "===" | "!==" | "==" | "!=") =>
            {
                let present = truth == matches!(b.operator.as_str(), "!==" | "!=");
                if self.global_undefined(&b.right) {
                    if let js::Expression::Identifier(i) = &b.left {
                        self.narrow_name(i, present);
                    }
                } else if self.global_undefined(&b.left) {
                    if let js::Expression::Identifier(i) = &b.right {
                        self.narrow_name(i, present);
                    }
                } else {
                    let equality = truth == matches!(b.operator.as_str(), "===" | "==");
                    for (operand, text) in [(&b.left, &b.right), (&b.right, &b.left)] {
                        if let (
                            js::Expression::UnaryExpression(u),
                            js::Expression::StringLiteral(text),
                        ) = (operand, text)
                        {
                            if u.operator.as_str() == "typeof" {
                                if let js::Expression::Identifier(i) = &u.argument {
                                    self.narrow_type_name(i, text.value.as_str(), equality);
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn narrow_name(&mut self, id: &js::IdentifierReference<'ast>, present: bool) {
        if let Ok(binding) = self.binding(id.name.as_str(), id.span) {
            if let Type::Optional(inner) = &binding.ty {
                let flow = if present {
                    ScalarFlow::of(inner)
                } else {
                    Some(ScalarFlow::Undefined)
                };
                if let Some(flow) = flow {
                    let name = binding.name;
                    self.flow.insert(name, flow);
                }
            }
        }
    }

    fn optional_equal(&mut self, left: Expr, right: Expr, at: oxc_span::Span) -> Result<Expr> {
        if is_undefined(&left.ty) || left.ty == unit() {
            if is_undefined(&right.ty) || right.ty == unit() {
                return Ok(node(
                    TypedExpression::Literal(TypedLiteral::Bool(true)),
                    boolean(),
                    at,
                ));
            }
            if matches!(right.ty, Type::Optional(_)) {
                let no = node(
                    TypedExpression::Literal(TypedLiteral::Bool(false)),
                    boolean(),
                    at,
                );
                let yes = node(
                    TypedExpression::Literal(TypedLiteral::Bool(true)),
                    boolean(),
                    at,
                );
                return self.optional_match(right, |_, _| Ok(no), Some(yes), boolean(), at);
            }
            return Ok(node(
                TypedExpression::Literal(TypedLiteral::Bool(false)),
                boolean(),
                at,
            ));
        }
        if is_undefined(&right.ty) || right.ty == unit() {
            return self.optional_equal(right, left, at);
        }
        if matches!(left.ty, Type::Optional(_)) {
            let no =
                self.optional_equal(var(self.name("None"), undefined(), at), right.clone(), at)?;
            return self.optional_match(
                left,
                |this, payload| this.optional_equal(payload, right, at),
                Some(no),
                boolean(),
                at,
            );
        }
        if matches!(right.ty, Type::Optional(_)) {
            return self.optional_equal(right, left, at);
        }
        Ok(binary(BinaryOp::Eq, left, right, boolean(), at))
    }

    fn type_description(&self, ty: &Type) -> String {
        match ty {
            Type::Primitive(PrimitiveType::F64) => "number".into(),
            Type::Primitive(PrimitiveType::Bool) => "boolean".into(),
            Type::Primitive(PrimitiveType::String) => "string".into(),
            Type::Primitive(PrimitiveType::Unit) => "void".into(),
            Type::Optional(inner) if **inner == Type::Never => "undefined".into(),
            Type::Optional(inner) => format!("{} | undefined", self.type_description(inner)),
            Type::Union(members) => members
                .iter()
                .map(|ty| self.type_description(ty))
                .collect::<Vec<_>>()
                .join(" | "),
            Type::Array { element_type, .. } => {
                format!("{}[]", self.type_description(element_type))
            }
            Type::Never => "never".into(),
            Type::TypeVar(t) => t.name.and_then(|n| n.resolve_global()).unwrap_or_default(),
            Type::Function {
                params,
                return_type,
                ..
            } => {
                let mut description = String::from("(");
                for (index, param) in params.iter().enumerate() {
                    if index != 0 {
                        description.push_str(", ");
                    }
                    description.push_str(&self.type_description(&param.ty));
                }
                write!(
                    &mut description,
                    ") => {}",
                    self.type_description(return_type)
                )
                .unwrap();
                description
            }
            Type::Named { id, .. } => {
                let mut description = String::from("{ ");
                if let Some(definition) = self.registry.get_type_by_id(*id) {
                    for (index, field) in definition.fields.iter().enumerate() {
                        if index != 0 {
                            description.push_str("; ");
                        }
                        write!(
                            &mut description,
                            "{}: {}",
                            field.name.resolve_global().unwrap_or_default(),
                            self.type_description(&field.ty)
                        )
                        .unwrap();
                    }
                }
                description.push_str(" }");
                description
            }
            _ => format!("{ty:?}"),
        }
    }

    fn view_of(&self, e: &js::Expression<'ast>) -> Option<InternedString> {
        match e {
            js::Expression::Identifier(i) => self
                .binding(i.name.as_str(), i.span)
                .ok()
                .and_then(|b| b.view),
            js::Expression::CallExpression(c) => {
                if let js::Expression::Identifier(i) = &c.callee {
                    let sig = self.signatures.get(&self.name(i.name.as_str()))?;
                    if let Some(index) = sig
                        .type_params
                        .iter()
                        .position(|(_, ty)| Some(ty) == sig.ret.as_ref())
                    {
                        if let Some(arguments) = &c.type_arguments {
                            return arguments
                                .params
                                .get(index)
                                .and_then(|ty| self.ty(ty).ok())
                                .and_then(|(_, view)| view);
                        }
                        let arg = sig
                            .params
                            .iter()
                            .position(|(_, ty, _)| Some(ty) == sig.ret.as_ref())?;
                        return c
                            .arguments
                            .get(arg)
                            .and_then(|a| a.as_expression())
                            .and_then(|e| self.view_of(e));
                    }
                    sig.ret_view
                } else {
                    None
                }
            }
            js::Expression::ParenthesizedExpression(p) => self.view_of(&p.expression),
            js::Expression::TSNonNullExpression(n) => self.view_of(&n.expression),
            js::Expression::ConditionalExpression(c) => self.view_of(&c.consequent),
            js::Expression::AssignmentExpression(a) => self.view_of(&a.right),
            js::Expression::TSAsExpression(a) => {
                self.ty(&a.type_annotation).ok().and_then(|(_, view)| view)
            }
            _ => None,
        }
    }

    fn check_view(
        &self,
        from: Option<InternedString>,
        to: Option<InternedString>,
        at: oxc_span::Span,
    ) -> Result<()> {
        if self.options.soundness == Soundness::Checked {
            if let Some(from) = from.and_then(|n| self.views.get(&n)) {
                let to = to.and_then(|n| self.views.get(&n));
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

    fn block(&mut self, statement: &js::Statement<'ast>) -> Result<TypedBlock> {
        self.scopes.push(FxHashMap::default());
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
        self.pop_scope();
        Ok(TypedBlock {
            statements,
            span: span(statement.span()),
        })
    }

    fn statement(&mut self, s: &js::Statement<'ast>) -> Result<Vec<Stmt>> {
        let at = s.span();
        Ok(match s {
            js::Statement::EmptyStatement(_) => vec![],
            js::Statement::VariableDeclaration(d) => self.variables(d)?,
            js::Statement::ExpressionStatement(e) => vec![expression(self.expr(&e.expression)?)],
            js::Statement::BlockStatement(_) => {
                vec![stmt(TypedStatement::Block(self.block(s)?), at)]
            }
            js::Statement::ReturnStatement(r) => {
                if self.returns.is_none() {
                    return self.error(ErrorKind::Syntax, "return outside function", at);
                }
                let hint = self
                    .returns
                    .as_ref()
                    .unwrap()
                    .expected
                    .as_ref()
                    .map(|(ty, _)| ty.clone());
                let mut value = r
                    .argument
                    .as_ref()
                    .map(|arg| match &hint {
                        Some(hint) => self.expr_expected(arg, hint),
                        None => self.expr(arg),
                    })
                    .transpose()?;
                if value.is_none()
                    && hint
                        .as_ref()
                        .is_some_and(|t| matches!(t, Type::Optional(_)))
                {
                    value = Some(var(self.name("None"), undefined(), at));
                }
                if value.as_ref().is_some_and(|v| {
                    element_type(&v.ty).is_some() || matches!(v.ty, Type::TypeVar(_))
                }) {
                    if let Some(source) = &r.argument {
                        self.record_array_return(source);
                    }
                }
                let view = r.argument.as_ref().and_then(|arg| self.view_of(arg));
                if value.as_ref().is_some_and(|v| v.ty == string())
                    && r.argument
                        .as_ref()
                        .is_some_and(|source| !self.immortal_string(source))
                {
                    self.returns.as_mut().unwrap().unproven_string = true;
                }
                let unit = unit();
                let actual = value.as_ref().map_or(&unit, |value| &value.ty);
                if let Some((ret, expected_view)) = self.returns.as_ref().unwrap().expected.clone()
                {
                    let logical = r
                        .argument
                        .as_ref()
                        .map_or(actual, |arg| self.source_type(arg, actual));
                    self.expect(logical, &ret, at)?;
                    self.check_view(view, expected_view, at)?;
                    if let (Some(source), Some(returned)) = (&r.argument, value.take()) {
                        self.fresh_properties(source, &ret)?;
                        value = Some(self.adapt_value(returned, &ret, at)?);
                    }
                } else {
                    self.returns.as_mut().unwrap().value_returned |= value.is_some();
                    let actual = if value.is_none() {
                        &undefined()
                    } else {
                        actual
                    };
                    self.record_return(actual, view, at)?;
                }
                vec![stmt(TypedStatement::Return(value.map(Box::new)), at)]
            }
            js::Statement::IfStatement(i) => {
                let condition = self.condition(&i.test)?;
                let before = self.flow.clone();
                self.narrow(&i.test, true);
                let then_block = self.block(&i.consequent)?;
                let then_flow = std::mem::replace(&mut self.flow, before);
                self.narrow(&i.test, false);
                let else_block = i.alternate.as_ref().map(|s| self.block(s)).transpose()?;
                let then_returns = Self::returns(std::slice::from_ref(&i.consequent));
                let else_returns = i
                    .alternate
                    .as_ref()
                    .is_some_and(|s| Self::returns(std::slice::from_ref(s)));
                if else_returns {
                    self.flow = then_flow;
                } else if !then_returns {
                    self.merge_flow(&then_flow);
                }
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
                self.clear_written_flow(w.span);
                let condition = self.condition(&w.test)?;
                self.narrow(&w.test, true);
                self.loop_depth += 1;
                let body = self.block(&w.body)?;
                self.loop_depth -= 1;
                self.clear_written_flow(w.span);
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
                self.scopes.push(FxHashMap::default());
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
                let repeated = oxc_span::Span::new(
                    f.init.as_ref().map_or(f.span.start, |i| i.span().end),
                    f.span.end,
                );
                let induction = self.for_induction(f);
                let native_induction = self.native_array_induction(f, induction, &mut init)?;
                self.clear_written_flow(repeated);
                self.set_induction(induction);
                let (condition, update) = if let Some((condition, update)) = native_induction {
                    (Some(Box::new(condition)), Some(Box::new(update)))
                } else {
                    let condition = f
                        .test
                        .as_ref()
                        .map(|e| self.condition(e))
                        .transpose()?
                        .map(Box::new);
                    self.clear_written_flow(repeated);
                    let update = f
                        .update
                        .as_ref()
                        .map(|e| self.expr(e))
                        .transpose()?
                        .map(Box::new);
                    (condition, update)
                };
                self.clear_written_flow(repeated);
                self.set_induction(induction);
                if let Some(test) = &f.test {
                    self.narrow(test, true);
                }
                self.loop_depth += 1;
                let body = self.block(&f.body)?;
                self.loop_depth -= 1;
                self.clear_written_flow(repeated);
                self.pop_scope();
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

    fn variables(&mut self, declaration: &js::VariableDeclaration<'ast>) -> Result<Vec<Stmt>> {
        let kind = declaration.kind.as_str();
        if !matches!(kind, "let" | "const") {
            return self.unsupported("var or using declarations", declaration.span);
        }
        let mut out = vec![];
        for d in &declaration.declarations {
            let js::BindingPattern::BindingIdentifier(id) = &d.id else {
                return self.unsupported("destructuring declarations", d.span);
            };
            let name = self.name(id.name.as_str());
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
            let declared = d
                .type_annotation
                .as_ref()
                .map(|a| self.ty(&a.type_annotation))
                .transpose()?;
            let value = match &declared {
                Some((ty, _)) => self.expr_expected(init, ty)?,
                None => self.expr(init)?,
            };
            if let Some((ty, _)) = &declared {
                self.expect(self.source_type(init, &value.ty), ty, d.span)?;
                self.fresh_properties(init, ty)?;
            }
            let (mut ty, view) = declared.unwrap_or_else(|| (value.ty.clone(), self.view_of(init)));
            self.check_view(self.view_of(init), view, d.span)?;
            if matches!(ty, Type::Named { .. }) {
                ty = value.ty.clone();
            }
            let unique = self.fresh(id.name.as_str());
            if matches!(ty, Type::Optional(_) | Type::Union(_)) {
                if let Some(flow) = ScalarFlow::of(&value.ty) {
                    self.flow.insert(unique, flow);
                }
            }
            self.value_facts(unique, &ty, init);
            let value = self.adapt_value(value, &ty, d.span)?;
            self.scopes.last_mut().unwrap().insert(
                name,
                Binding {
                    name: unique,
                    symbol: id.symbol_id.get(),
                    ty: ty.clone(),
                    view,
                    mutable: kind == "let",
                    initialized: true,
                },
            );
            out.push(stmt(
                TypedStatement::Let(TypedLet {
                    name: unique,
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

    fn condition(&mut self, e: &js::Expression<'ast>) -> Result<Expr> {
        let value = self.expr(e)?;
        self.truthy(value, e.span())
    }

    fn truthy(&mut self, value: Expr, at: oxc_span::Span) -> Result<Expr> {
        if matches!(value.ty, Type::Union(_)) {
            return self.union_match(value, u8::MAX, boolean(), at, |this, payload| {
                this.truthy(payload, at)
            });
        }
        if value.ty == unit() {
            return self.error(
                ErrorKind::Type,
                "an expression of type void cannot be tested for truthiness",
                at,
            );
        }
        if is_undefined(&value.ty) {
            let no = node(
                TypedExpression::Literal(TypedLiteral::Bool(false)),
                boolean(),
                at,
            );
            return Ok(self.sequence(value, no, at));
        }
        if matches!(value.ty, Type::Optional(_)) {
            let no = node(
                TypedExpression::Literal(TypedLiteral::Bool(false)),
                boolean(),
                at,
            );
            return self.optional_match(
                value,
                |this, payload| this.truthy(payload, at),
                Some(no),
                boolean(),
                at,
            );
        }
        if value.ty == boolean() {
            return Ok(value);
        }
        if value.ty == string() {
            let empty = node(
                TypedExpression::Literal(TypedLiteral::String(self.name(""))),
                string(),
                at,
            );
            let equal = self.call("$String$equals", vec![value, empty], boolean(), at);
            return Ok(node(
                TypedExpression::Unary(TypedUnary {
                    op: UnaryOp::Not,
                    operand: Box::new(equal),
                }),
                boolean(),
                at,
            ));
        }
        if value.ty == number() {
            // The ordered self-comparison rejects NaN as well as zero.
            let unique = self.fresh("truthy");
            let binding = stmt(
                TypedStatement::Let(TypedLet {
                    name: unique,
                    ty: number(),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(value)),
                    span: span(at),
                }),
                at,
            );
            let read = var(unique, number(), at);
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
        self.unsupported("truthiness of this type", at)
    }

    fn function_type(
        &self,
        params: impl Iterator<Item = Type>,
        ret: Type,
        at: oxc_span::Span,
    ) -> Result<Type> {
        let scalar = |ty: &Type| {
            matches!(
                ty,
                Type::Primitive(PrimitiveType::F64 | PrimitiveType::Bool) | Type::Function { .. }
            )
        };
        let mut parameters = Vec::with_capacity(params.size_hint().0);
        for ty in params {
            if !scalar(&ty) {
                return self.unsupported("aggregate callback parameters", at);
            }
            parameters.push(zyntax_typed_ast::ParamInfo {
                name: None,
                ty,
                is_optional: false,
                is_varargs: false,
                is_keyword_only: false,
                is_positional_only: false,
                is_out: false,
                is_ref: false,
                is_inout: false,
            });
        }
        if !scalar(&ret) && ret != unit() {
            return self.unsupported("aggregate callback results", at);
        }
        Ok(Type::Function {
            params: parameters,
            return_type: Box::new(ret),
            is_varargs: false,
            has_named_params: false,
            has_default_params: false,
            async_kind: zyntax_typed_ast::AsyncKind::Sync,
            calling_convention: CallingConvention::Default,
            nullability: NullabilityKind::NonNull,
        })
    }

    fn expr_expected(&mut self, e: &js::Expression<'ast>, expected: &Type) -> Result<Expr> {
        let value = match e {
            js::Expression::ArrayExpression(a) => self.array_literal(a, Some(expected)),
            js::Expression::ArrowFunctionExpression(a) => self.arrow(a, Some(expected)),
            js::Expression::ParenthesizedExpression(p) => {
                self.expr_expected(&p.expression, expected)
            }
            _ => self.expr(e),
        }?;
        self.check_union_source(e, &value.ty, expected)?;
        Ok(value)
    }

    fn arrow(
        &mut self,
        arrow: &js::ArrowFunctionExpression<'ast>,
        hint: Option<&Type>,
    ) -> Result<Expr> {
        let at = arrow.span;
        if arrow.r#async || arrow.type_parameters.is_some() || arrow.params.rest.is_some() {
            return self.unsupported("async, generic or rest arrow functions", at);
        }
        let (hints, result_hint) = match hint {
            Some(Type::Function {
                params,
                return_type,
                ..
            }) => (params.as_slice(), Some(return_type.as_ref())),
            _ => (&[][..], None),
        };
        let expected = if let Some(annotation) = &arrow.return_type {
            Some(self.ty(&annotation.type_annotation)?)
        } else {
            result_hint
                .filter(|ty| **ty != unit())
                .map(|ty| (ty.clone(), None))
        };
        let outer_returns = self.returns.replace(ReturnContext {
            expected,
            inferred: None,
            value_returned: false,
            array_parameters: SmallVec::new(),
            array_result: None,
            unproven_string: false,
        });
        let outer_loop = std::mem::replace(&mut self.loop_depth, 0);
        let outer_floor = std::mem::replace(&mut self.scope_floor, self.scopes.len());
        self.scopes.push(FxHashMap::default());
        let result = (|| {
            let mut params = Vec::with_capacity(arrow.params.items.len());
            for (index, p) in arrow.params.items.iter().enumerate() {
                let js::BindingPattern::BindingIdentifier(id) = &p.pattern else {
                    return self.unsupported("destructuring arrow parameters", p.span);
                };
                if p.optional || p.initializer.is_some() {
                    return self.unsupported("optional or default arrow parameters", p.span);
                }
                let (ty, view) = match &p.type_annotation {
                    Some(annotation) => self.ty(&annotation.type_annotation)?,
                    None => (
                        hints
                            .get(index)
                            .ok_or_else(|| {
                                Error::new(
                                    ErrorKind::Type,
                                    "parameter implicitly has type any",
                                    p.span,
                                )
                            })?
                            .ty
                            .clone(),
                        None,
                    ),
                };
                let unique = self.fresh(id.name.as_str());
                let source_name = self.name(id.name.as_str());
                self.scopes.last_mut().unwrap().insert(
                    source_name,
                    Binding {
                        name: unique,
                        symbol: id.symbol_id.get(),
                        ty: ty.clone(),
                        view,
                        mutable: true,
                        initialized: true,
                    },
                );
                params.push(TypedParameter::regular(
                    unique,
                    ty,
                    Mutability::Mutable,
                    span(p.span),
                ));
            }
            let mut statements = vec![];
            let complete = if let js::ArrowFunctionBody::FunctionBody(body) = &arrow.body {
                self.reserve(&body.statements)?;
                for statement in &body.statements {
                    statements.extend(self.statement(statement)?);
                }
                Self::returns(&body.statements)
            } else {
                let e = arrow.body.as_expression().unwrap();
                let hint = self
                    .returns
                    .as_ref()
                    .unwrap()
                    .expected
                    .as_ref()
                    .map(|(ty, _)| ty.clone());
                let mut value = match hint {
                    Some(ty) => self.expr_expected(e, &ty)?,
                    None => self.expr(e)?,
                };
                if let Some((ret, view)) = self.returns.as_ref().unwrap().expected.clone() {
                    self.expect(&value.ty, &ret, at)?;
                    self.check_view(self.view_of(e), view, at)?;
                    value = self.adapt_value(value, &ret, at)?;
                } else {
                    self.returns.as_mut().unwrap().value_returned = true;
                    self.record_return(&value.ty, self.view_of(e), at)?;
                }
                statements.push(stmt(TypedStatement::Return(Some(Box::new(value))), at));
                true
            };
            let context = self.returns.take().unwrap();
            let declared_result = context.expected.is_some();
            let (ret, _) = match context.expected {
                Some(result) => result,
                None if !context.value_returned => (unit(), None),
                None => context.inferred.unwrap_or((unit(), None)),
            };
            if ret != unit() && !complete {
                if declared_result {
                    return self.error(
                        ErrorKind::Type,
                        "arrow function lacks an ending return",
                        at,
                    );
                }
                return self.unsupported("possibly undefined arrow results", at);
            }
            let ty = self.function_type(params.iter().map(|p| p.ty.clone()), ret.clone(), at)?;
            Ok((params, statements, ret, ty))
        })();
        self.pop_scope();
        self.scope_floor = outer_floor;
        self.loop_depth = outer_loop;
        self.returns = outer_returns;
        let (params, statements, ret, ty) = result?;
        let native = self.fresh("arrow");
        self.declarations.push(node(
            TypedDeclaration::Function(TypedFunction {
                name: native,
                params,
                return_type: ret,
                body: Some(TypedBlock {
                    statements,
                    span: span(at),
                }),
                annotations: vec![strict_fp(self.name("strict_fp"))],
                visibility: Visibility::Private,
                ..Default::default()
            }),
            unit(),
            at,
        ));
        Ok(var(native, ty, at))
    }

    fn expr(&mut self, e: &js::Expression<'ast>) -> Result<Expr> {
        let at = e.span();
        Ok(match e {
            js::Expression::NumericLiteral(n) => num(n.value, at),
            js::Expression::BooleanLiteral(b) => node(
                TypedExpression::Literal(TypedLiteral::Bool(b.value)),
                boolean(),
                at,
            ),
            js::Expression::StringLiteral(s) if !s.lone_surrogates => node(
                TypedExpression::Literal(TypedLiteral::String(self.name(s.value.as_str()))),
                string(),
                at,
            ),
            js::Expression::Identifier(i) => {
                if self
                    .scopes
                    .iter()
                    .any(|s| s.contains_key(&self.name(i.name.as_str())))
                {
                    let b = self.binding(i.name.as_str(), at)?;
                    let value = var(b.name, b.ty.clone(), at);
                    let flow = self.flow.get(&b.name).copied();
                    let value = self.flow_read(value, flow, at)?;
                    if value.ty == Type::Primitive(PrimitiveType::I64) {
                        node(
                            TypedExpression::Cast(TypedCast {
                                expr: Box::new(value),
                                target_type: number(),
                            }),
                            number(),
                            at,
                        )
                    } else {
                        value
                    }
                } else {
                    match i.name.as_str() {
                        "undefined" => var(self.name("None"), undefined(), at),
                        "NaN" => num(f64::NAN, at),
                        "Infinity" => num(f64::INFINITY, at),
                        name if self.signatures.contains_key(&self.name(name)) => {
                            let sig = &self.signatures[&self.name(name)];
                            if !sig.type_params.is_empty() {
                                return self.unsupported("generic function values", at);
                            }
                            let ret = sig.ret.as_ref().ok_or_else(|| {
                                Error::new(ErrorKind::Unsupported, "unresolved function result", at)
                            })?;
                            let ty = self.function_type(
                                sig.params.iter().map(|(_, ty, _)| ty.clone()),
                                ret.clone(),
                                at,
                            )?;
                            var(sig.native, ty, at)
                        }
                        name => {
                            let b = self.binding(name, at)?;
                            var(b.name, b.ty.clone(), at)
                        }
                    }
                }
            }
            js::Expression::ParenthesizedExpression(p) => self.expr(&p.expression)?,
            js::Expression::ArrowFunctionExpression(a) => self.arrow(a, None)?,
            js::Expression::UnaryExpression(u) => {
                if u.operator.as_str() == "void" {
                    let value = self.expr(&u.argument)?;
                    self.sequence(value, var(self.name("None"), undefined(), at), at)
                } else if u.operator.as_str() == "typeof" {
                    let value = self.expr(&u.argument)?;
                    self.typeof_value(value, at)?
                } else if u.operator.as_str() == "!" {
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
                let before = self.flow.clone();
                let op = l.operator.as_str();
                if op == "??" {
                    self.narrow(&l.left, false);
                    let right = self.expr(&l.right)?;
                    self.merge_flow(&before);
                    if is_undefined(&left.ty) {
                        self.sequence(left, right, at)
                    } else if let Type::Optional(inner) = &left.ty {
                        let ty = optional_join(inner, &right.ty).ok_or_else(|| {
                            Error::new(
                                ErrorKind::Unsupported,
                                "this nullish result union is not implemented yet",
                                at,
                            )
                        })?;
                        let right = self.adapt_value(right, &ty, at)?;
                        self.optional_match(
                            left,
                            |this, payload| this.adapt_value(payload, &ty, at),
                            Some(right),
                            ty.clone(),
                            at,
                        )?
                    } else {
                        left
                    }
                } else {
                    if matches!(left.ty, Type::Optional(_)) {
                        return self.unsupported(
                            "value-selecting logical operations on optional values",
                            at,
                        );
                    }
                    self.expect(&left.ty, &boolean(), at)?;
                    self.narrow(&l.left, op == "&&");
                    let right = self.expr(&l.right)?;
                    if matches!(right.ty, Type::Optional(_)) {
                        return self.unsupported(
                            "value-selecting logical operations on optional values",
                            at,
                        );
                    }
                    self.expect(&right.ty, &boolean(), at)?;
                    self.merge_flow(&before);
                    let op = match op {
                        "&&" => BinaryOp::And,
                        "||" => BinaryOp::Or,
                        _ => unreachable!(),
                    };
                    binary(op, left, right, boolean(), at)
                }
            }
            js::Expression::ConditionalExpression(c) => {
                let condition = self.condition(&c.test)?;
                let before = self.flow.clone();
                self.narrow(&c.test, true);
                let a = self.expr(&c.consequent)?;
                let a_static = a.ty != string() || self.immortal_string(&c.consequent);
                let then_flow = std::mem::replace(&mut self.flow, before);
                self.narrow(&c.test, false);
                let b = self.expr(&c.alternate)?;
                let b_static = b.ty != string() || self.immortal_string(&c.alternate);
                self.merge_flow(&then_flow);
                if a.ty != b.ty
                    && matches!((&a.ty, &b.ty), (Type::Named { .. }, Type::Named { .. }))
                {
                    return self
                        .unsupported("conditional objects with different physical layouts", at);
                }
                let joined = optional_join(&a.ty, &b.ty);
                if joined.is_none() {
                    self.expect(&b.ty, &a.ty, at)?;
                }
                if matches!(a.ty, Type::Named { .. })
                    && self.view_of(&c.consequent) != self.view_of(&c.alternate)
                {
                    return self
                        .unsupported("conditional objects with different property views", at);
                }
                let ty = joined.unwrap_or_else(|| a.ty.clone());
                if matches!(ty, Type::Union(_)) && (!a_static || !b_static) {
                    return self.unsupported(
                        "conditional union string payload without a proven static lifetime",
                        at,
                    );
                }
                let a = self.adapt_value(a, &ty, at)?;
                let b = self.adapt_value(b, &ty, at)?;
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
                if element_type(&object.ty).is_some() && m.property.name == "length" {
                    return Ok(self.array_length(object, at));
                }
                let ty = self.field_type(
                    self.source_type(&m.object, &object.ty),
                    m.property.name.as_str(),
                    at,
                )?;
                node(
                    TypedExpression::Field(TypedFieldAccess {
                        object: Box::new(object),
                        field: self.name(m.property.name.as_str()),
                    }),
                    ty,
                    at,
                )
            }
            js::Expression::ArrayExpression(a) => self.array_literal(a, None)?,
            js::Expression::ComputedMemberExpression(m) if !m.optional => {
                self.array_index(&m.object, &m.expression, at)?
            }
            js::Expression::ObjectExpression(o) => self.object(o)?,
            js::Expression::CallExpression(c) if !c.optional => self.call_expr(c)?,
            js::Expression::AssignmentExpression(a) => self.assignment(a)?,
            js::Expression::UpdateExpression(u) => {
                let target = self.simple_target(&u.argument, at)?;
                let read = self.read_target(&target, at)?;
                self.expect(&read.ty, &number(), at)?;
                let (mut statements, target) = self.capture_target(target, at);
                let read = if matches!(
                    target.node,
                    TypedExpression::Field(_) | TypedExpression::Index(_)
                ) {
                    target.clone()
                } else {
                    read
                };
                let old = self.fresh("old");
                let save = stmt(
                    TypedStatement::Let(TypedLet {
                        name: old,
                        ty: number(),
                        mutability: Mutability::Immutable,
                        initializer: Some(Box::new(read)),
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
                    var(old, number(), at),
                    num(1.0, at),
                    number(),
                    at,
                );
                let value = self.adapt_value(value, &target.ty, at)?;
                self.record_write(&target, &number());
                let assign = binary(BinaryOp::Assign, target.clone(), value, unit(), at);
                let result = if u.prefix {
                    self.read_target(&target, at)?
                } else {
                    var(old, number(), at)
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
                let (ty, view) = self.ty(&a.type_annotation)?;
                let mut value = self.expr_expected(&a.expression, &ty)?;
                let logical = self.source_type(&a.expression, &value.ty);
                if self.record_subtype(&ty, logical) {
                    if !self.record_subtype(&value.ty, &ty) {
                        return self.unsupported(
                            "record assertions requiring unknown physical fields",
                            at,
                        );
                    }
                } else {
                    if matches!(logical, Type::Optional(_))
                        && optional_join(logical, &ty).is_some()
                        && *logical != ty
                    {
                        return self.unsupported("unproven optional narrowing assertions", at);
                    }
                    self.expect(logical, &ty, at)?;
                }
                self.check_view(self.view_of(&a.expression), view, at)?;
                if matches!(ty, Type::Optional(_) | Type::Union(_)) || is_undefined(&value.ty) {
                    self.adapt_value(value, &ty, at)?
                } else {
                    if !matches!(ty, Type::Named { .. }) {
                        value.ty = ty;
                    }
                    value
                }
            }
            js::Expression::TSNonNullExpression(n) => {
                let value = self.expr(&n.expression)?;
                if matches!(value.ty, Type::Optional(_)) {
                    return self.unsupported("unproven non-null assertions on optional values", at);
                }
                value
            }
            _ => return self.unsupported("this expression", at),
        })
    }

    fn binary_expr(&mut self, b: &js::BinaryExpression<'ast>) -> Result<Expr> {
        let at = b.span;
        let left = self.expr(&b.left)?;
        let right = self.expr(&b.right)?;
        let op = b.operator.as_str();
        if matches!(op, "===" | "!==" | "==" | "!=")
            && (matches!(left.ty, Type::Optional(_))
                || matches!(right.ty, Type::Optional(_))
                || left.ty == unit()
                || right.ty == unit())
        {
            let compatible = if left.ty == unit() || right.ty == unit() {
                (left.ty == unit() && (matches!(right.ty, Type::Optional(_)) || right.ty == unit()))
                    || (right.ty == unit() && matches!(left.ty, Type::Optional(_)))
            } else {
                is_undefined(&left.ty)
                    || is_undefined(&right.ty)
                    || optional_join(&left.ty, &right.ty).is_some()
            };
            if !compatible {
                return self.error(ErrorKind::Type, "comparison has no overlapping types", at);
            }
            let mut statements = vec![];
            let mut reads = vec![];
            for value in [left, right] {
                if value.ty == unit() {
                    statements.push(expression(value));
                    reads.push(node(
                        TypedExpression::Literal(TypedLiteral::Unit),
                        unit(),
                        at,
                    ));
                    continue;
                }
                let name = self.fresh("comparison");
                let ty = value.ty.clone();
                statements.push(stmt(
                    TypedStatement::Let(TypedLet {
                        name,
                        ty: ty.clone(),
                        mutability: Mutability::Immutable,
                        initializer: Some(Box::new(value)),
                        span: span(at),
                    }),
                    at,
                ));
                reads.push(var(name, ty, at));
            }
            let right = reads.pop().unwrap();
            let mut value = self.optional_equal(reads.pop().unwrap(), right, at)?;
            if matches!(op, "!==" | "!=") {
                value = node(
                    TypedExpression::Unary(TypedUnary {
                        op: UnaryOp::Not,
                        operand: Box::new(value),
                    }),
                    boolean(),
                    at,
                );
            }
            statements.push(expression(value));
            return Ok(node(
                TypedExpression::Block(TypedBlock {
                    statements,
                    span: span(at),
                }),
                boolean(),
                at,
            ));
        }
        if matches!(left.ty, Type::Optional(_)) || matches!(right.ty, Type::Optional(_)) {
            return self.error(ErrorKind::Type, "operand is possibly undefined", at);
        }
        if matches!(left.ty, Type::Union(_)) || matches!(right.ty, Type::Union(_)) {
            if matches!(op, "===" | "!==") {
                return self.union_comparison(left, right, op == "!==", at);
            }
            if op == "+" && (left.ty == string() || right.ty == string()) {
                return self.unsupported("string coercion of union values", at);
            }
            if matches!(op, "+" | "-" | "*" | "/" | "%" | "**") {
                return self.error(
                    ErrorKind::Type,
                    "arithmetic requires a narrowed union operand",
                    at,
                );
            }
            return self.unsupported("this operator on union values", at);
        }
        if matches!(op, "-" | "*" | "/" | "%") {
            self.expect(&left.ty, &number(), at)?;
            self.expect(&right.ty, &number(), at)?;
        }
        if matches!(left.ty, Type::Function { .. }) || matches!(right.ty, Type::Function { .. }) {
            return self.unsupported("function object identity and operators", at);
        }
        if matches!(left.ty, Type::TypeVar(_)) || matches!(right.ty, Type::TypeVar(_)) {
            if matches!(op, "+" | "-" | "*" | "/" | "%" | "**") {
                return self.error(
                    ErrorKind::Type,
                    "arithmetic requires a concrete numeric or string type",
                    at,
                );
            }
            return self.unsupported("operators on unconstrained type parameters", at);
        }
        if left.ty == string() && right.ty == string() {
            return match op {
                "+" => Ok(self.call("$String$concat", vec![left, right], string(), at)),
                "===" | "==" => Ok(self.call("$String$equals", vec![left, right], boolean(), at)),
                "!==" | "!=" => Ok(node(
                    TypedExpression::Unary(TypedUnary {
                        op: UnaryOp::Not,
                        operand: Box::new(self.call(
                            "$String$equals",
                            vec![left, right],
                            boolean(),
                            at,
                        )),
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
            return Ok(self.call("$TS$pow", vec![left, right], number(), at));
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

    fn field_type(&self, ty: &Type, field: &'ast str, at: oxc_span::Span) -> Result<Type> {
        if matches!(ty, Type::Function { .. }) {
            return self.unsupported("function object properties", at);
        }
        let Type::Named { id, .. } = ty else {
            return self.error(ErrorKind::Type, "receiver has no fields", at);
        };
        let field_name = self.name(field);
        self.registry
            .get_type_by_id(*id)
            .and_then(|d| d.fields.iter().find(|f| f.name == field_name))
            .map(|f| f.ty.clone())
            .ok_or_else(|| Error::new(ErrorKind::Type, format!("no field `{field}`"), at))
    }

    fn object(&mut self, object: &js::ObjectExpression<'ast>) -> Result<Expr> {
        let mut values: SmallVec<[(&'ast str, Expr); 4]> = SmallVec::new();
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
            let temporary = self.fresh("field");
            let ty = value.ty.clone();
            statements.push(stmt(
                TypedStatement::Let(TypedLet {
                    name: temporary,
                    ty: ty.clone(),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(value)),
                    span: span(p.span),
                }),
                p.span,
            ));
            values.push((name, var(temporary, ty, p.span)));
        }
        // Evaluate in source order, then arrange values in the physical layout.
        values.sort_unstable_by_key(|(name, _)| *name);
        if values.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return self.unsupported("duplicate object keys", object.span);
        }
        let ty = self.layout(values.iter().map(|(name, value)| {
            let Type::Primitive(ty) = value.ty else {
                unreachable!()
            };
            (*name, ty)
        }));
        let Type::Named { id, .. } = &ty else {
            unreachable!()
        };
        let name = self.registry.get_type_by_id(*id).unwrap().name;
        let fields = values
            .into_iter()
            .map(|(name, value)| TypedFieldInit {
                name: self.name(name),
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

    fn arguments<'hint>(
        &mut self,
        c: &js::CallExpression<'ast>,
        mut hints: impl Iterator<Item = &'hint Type>,
        mut static_strings: Option<&mut SmallVec<[bool; 4]>>,
    ) -> Result<Vec<Expr>> {
        let mut args = Vec::with_capacity(c.arguments.len());
        for arg in &c.arguments {
            let Some(e) = arg.as_expression() else {
                return self.unsupported("spread arguments", c.span);
            };
            let value = match hints.next() {
                Some(ty) if !generics::unresolved(ty) => self.expr_expected(e, ty)?,
                _ => self.expr(e)?,
            };
            if let Some(proofs) = &mut static_strings {
                proofs.push(value.ty != string() || self.immortal_string(e));
            }
            args.push(value);
        }
        Ok(args)
    }

    fn call_expr(&mut self, c: &js::CallExpression<'ast>) -> Result<Expr> {
        let at = c.span;
        if let js::Expression::StaticMemberExpression(m) = &c.callee {
            if let js::Expression::Identifier(receiver) = &m.object {
                if receiver.name == "console"
                    && m.property.name == "log"
                    && !self
                        .scopes
                        .iter()
                        .any(|s| s.contains_key(&self.name("console")))
                {
                    if c.type_arguments.is_some() {
                        return self.error(
                            ErrorKind::Type,
                            "console.log does not take type arguments",
                            at,
                        );
                    }
                    let args = self.arguments(c, std::iter::empty(), None)?;
                    return self.log(args, at);
                }
            }
        }
        if let js::Expression::StaticMemberExpression(m) = &c.callee {
            let is_array = if let js::Expression::Identifier(id) = &m.object {
                self.binding(id.name.as_str(), id.span)
                    .is_ok_and(|b| element_type(&b.ty).is_some())
            } else {
                false
            };
            if is_array && !m.optional {
                return self.array_method(c, m);
            }
        }
        let direct = match &c.callee {
            js::Expression::Identifier(id)
                if !self
                    .scopes
                    .iter()
                    .any(|s| s.contains_key(&self.name(id.name.as_str()))) =>
            {
                Some(id)
            }
            _ => None,
        };
        let Some(id) = direct else {
            let callee = self.expr(&c.callee)?;
            let Type::Function {
                params,
                return_type,
                ..
            } = &callee.ty
            else {
                return self.error(ErrorKind::Type, "value is not callable", at);
            };
            if c.type_arguments.is_some() || c.arguments.len() != params.len() {
                return self.error(ErrorKind::Type, "wrong function argument count", at);
            }
            let args = self.arguments(c, params.iter().map(|p| &p.ty), None)?;
            for (arg, expected) in args.iter().zip(params) {
                self.expect(&arg.ty, &expected.ty, at)?;
            }
            let ret = return_type.as_ref().clone();
            return Ok(node(
                TypedExpression::Call(TypedCall::positional(callee, args)),
                ret,
                at,
            ));
        };
        let Some(sig) = self.signatures.get(&self.name(id.name.as_str())).cloned() else {
            return self.error(
                ErrorKind::Type,
                format!("cannot find function `{}`", id.name),
                at,
            );
        };
        if c.arguments.len() != sig.params.len() {
            return self.error(ErrorKind::Type, "wrong argument count", at);
        }
        let mut types: SmallVec<[Option<(Type, Option<InternedString>)>; 2]> =
            std::iter::repeat_n(None, sig.type_params.len()).collect();
        let explicit = c.type_arguments.is_some();
        if let Some(arguments) = &c.type_arguments {
            if arguments.params.len() != types.len() {
                return self.error(ErrorKind::Type, "wrong type argument count", at);
            }
            for (slot, ty) in types.iter_mut().zip(&arguments.params) {
                *slot = Some(self.ty(ty)?);
            }
        }
        let hints: SmallVec<[Type; 4]> = sig
            .params
            .iter()
            .map(|(_, ty, view)| {
                if explicit {
                    self.instantiate_type(ty, *view, &sig.type_params, &types, at)
                        .map(|(ty, _)| ty)
                } else {
                    Ok(ty.clone())
                }
            })
            .collect::<Result<_>>()?;
        let mut static_strings = SmallVec::new();
        let args = self.arguments(c, hints.iter(), Some(&mut static_strings))?;
        if !explicit {
            for ((arg, source), (_, expected, _)) in args.iter().zip(&c.arguments).zip(&sig.params)
            {
                let source = source.as_expression().unwrap();
                self.infer_arguments(
                    expected,
                    self.source_type(source, &arg.ty),
                    self.view_of(source),
                    &sig.type_params,
                    &mut types,
                );
            }
        }
        let args = args
            .into_iter()
            .zip(&c.arguments)
            .zip(&sig.params)
            .zip(static_strings)
            .map(|(((arg, source), (_, expected, view)), static_string)| {
                let (expected, view) =
                    self.instantiate_type(expected, *view, &sig.type_params, &types, at)?;
                let source = source.as_expression().unwrap();
                self.expect(self.source_type(source, &arg.ty), &expected, at)?;
                if matches!(expected, Type::Union(_)) && !static_string {
                    return self.unsupported(
                        "union string argument without a proven static lifetime",
                        source.span(),
                    );
                }
                self.fresh_properties(source, &expected)?;
                self.check_view(self.view_of(source), view, at)?;
                self.adapt_value(arg, &expected, at)
            })
            .collect::<Result<Vec<_>>>()?;
        let ret = sig.ret.clone().ok_or_else(|| {
            Error::new(
                ErrorKind::Unsupported,
                "unresolved inferred return type",
                at,
            )
        })?;
        let (ret, _) = self.instantiate_type(&ret, sig.ret_view, &sig.type_params, &types, at)?;
        let type_args = types
            .into_iter()
            .map(|ty| {
                ty.map(|(ty, _)| ty).ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unsupported,
                        "unresolved generic type argument",
                        at,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(node(
            TypedExpression::Call(TypedCall {
                callee: Box::new(var(sig.native, Type::Any, at)),
                positional_args: args,
                named_args: vec![],
                type_args,
            }),
            ret,
            at,
        ))
    }

    fn log(&mut self, args: Vec<Expr>, at: oxc_span::Span) -> Result<Expr> {
        let mut statements = vec![];
        let mut reads = vec![];
        for arg in args {
            let name = self.fresh("log");
            let ty = arg.ty.clone();
            if ty == unit() {
                statements.push(expression(arg));
                reads.push(node(
                    TypedExpression::Literal(TypedLiteral::Unit),
                    unit(),
                    at,
                ));
                continue;
            }
            statements.push(stmt(
                TypedStatement::Let(TypedLet {
                    name,
                    ty: ty.clone(),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(arg)),
                    span: span(at),
                }),
                at,
            ));
            reads.push(var(name, ty, at));
        }
        for (i, value) in reads.into_iter().enumerate() {
            if i != 0 {
                statements.push(expression(self.call("$TS$space", vec![], unit(), at)));
            }
            statements.push(expression(self.format_value(value, at)?));
        }
        statements.push(expression(self.call("$TS$newline", vec![], unit(), at)));
        Ok(node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(at),
            }),
            unit(),
            at,
        ))
    }

    fn read_target(&mut self, target: &Expr, at: oxc_span::Span) -> Result<Expr> {
        let flow = if let TypedExpression::Variable(name) = target.node {
            self.flow.get(&name).copied()
        } else {
            None
        };
        self.flow_read(target.clone(), flow, at)
    }

    fn record_write(&mut self, target: &Expr, ty: &Type) {
        if let TypedExpression::Variable(name) = target.node {
            self.flow.forget(name);
            if matches!(target.ty, Type::Optional(_) | Type::Union(_)) {
                if let Some(flow) = ScalarFlow::of(ty) {
                    self.flow.insert(name, flow);
                }
            }
        }
    }

    fn simple_target(
        &mut self,
        target: &js::SimpleAssignmentTarget<'ast>,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        match target {
            js::SimpleAssignmentTarget::AssignmentTargetIdentifier(i) => {
                let b = self.binding(i.name.as_str(), at)?;
                if !b.mutable {
                    return self.error(ErrorKind::Type, "cannot assign to const", at);
                }
                Ok(var(b.name, b.ty.clone(), at))
            }
            js::SimpleAssignmentTarget::StaticMemberExpression(m) => self.field_target(m),
            js::SimpleAssignmentTarget::ComputedMemberExpression(m) if !m.optional => {
                self.array_index(&m.object, &m.expression, at)
            }
            _ => self.unsupported("this assignment target", at),
        }
    }

    fn field_target(&mut self, m: &js::StaticMemberExpression<'ast>) -> Result<Expr> {
        if let js::Expression::Identifier(id) = &m.object {
            if self
                .binding(id.name.as_str(), id.span)
                .is_ok_and(|b| element_type(&b.ty).is_some())
            {
                return self.unsupported(
                    "array property mutation, including sparse length changes",
                    m.span,
                );
            }
        }
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
        let ty = self.field_type(
            self.source_type(&m.object, &object.ty),
            m.property.name.as_str(),
            m.span,
        )?;
        Ok(node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(object),
                field: self.name(m.property.name.as_str()),
            }),
            ty,
            m.span,
        ))
    }

    fn assignment(&mut self, a: &js::AssignmentExpression<'ast>) -> Result<Expr> {
        let target = match &a.left {
            js::AssignmentTarget::AssignmentTargetIdentifier(i) => {
                let b = self.binding(i.name.as_str(), a.span)?;
                if !b.mutable {
                    return self.error(ErrorKind::Type, "cannot assign to const", a.span);
                }
                var(b.name, b.ty.clone(), a.span)
            }
            js::AssignmentTarget::StaticMemberExpression(m) => self.field_target(m)?,
            js::AssignmentTarget::ComputedMemberExpression(m) if !m.optional => {
                self.array_index(&m.object, &m.expression, a.span)?
            }
            _ => return self.unsupported("this assignment target", a.span),
        };
        if matches!(target.node, TypedExpression::Index(_)) && matches!(target.ty, Type::TypeVar(_))
        {
            return self.unsupported(
                "generic array mutation before element ownership is known",
                a.span,
            );
        }
        let (mut statements, target) = self.capture_target(target, a.span);
        let target_view = if let js::AssignmentTarget::AssignmentTargetIdentifier(i) = &a.left {
            self.binding(i.name.as_str(), a.span)?.view
        } else {
            None
        };
        let logical_target = target_view
            .and_then(|view| self.aliases.get(&view))
            .map_or(&target.ty, |(ty, _)| ty)
            .clone();
        let op = a.operator.as_str();
        let read = if op == "=" {
            None
        } else {
            Some(self.read_target(&target, a.span)?)
        };
        let mut value = self.expr_expected(&a.right, &logical_target)?;
        self.expect(
            self.source_type(&a.right, &value.ty),
            &logical_target,
            a.span,
        )?;
        self.fresh_properties(&a.right, &logical_target)?;
        self.check_view(self.view_of(&a.right), target_view, a.span)?;
        if matches!(target.ty, Type::Named { .. }) && value.ty != target.ty {
            return self.unsupported(
                "assignment changing a binding's physical record layout",
                a.span,
            );
        }
        if let Some(read) = read {
            self.expect(&read.ty, &number(), a.span)?;
            self.expect(&value.ty, &number(), a.span)?;
            let operation = match op {
                "+=" => BinaryOp::Add,
                "-=" => BinaryOp::Sub,
                "*=" => BinaryOp::Mul,
                "/=" => BinaryOp::Div,
                "%=" => BinaryOp::Rem,
                _ => return self.unsupported("this compound assignment", a.span),
            };
            value = binary(operation, read, value, number(), a.span);
        }
        let name = self.fresh("assigned");
        let ty = value.ty.clone();
        let save = stmt(
            TypedStatement::Let(TypedLet {
                name,
                ty: ty.clone(),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(value)),
                span: span(a.span),
            }),
            a.span,
        );
        if let TypedExpression::Variable(name) = target.node {
            self.record_write(&target, &ty);
            if op == "=" {
                self.value_facts(name, &ty, &a.right);
            }
        } else if target.ty == string()
            && matches!(target.node, TypedExpression::Index(_))
            && !self.immortal_string(&a.right)
        {
            return self.unsupported(
                "array storage of strings without a proven static lifetime",
                a.span,
            );
        }
        let stored = self.adapt_value(var(name, ty.clone(), a.span), &target.ty, a.span)?;
        let assignment = binary(BinaryOp::Assign, target, stored, unit(), a.span);
        statements.extend([
            save,
            expression(assignment),
            expression(var(name, ty.clone(), a.span)),
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

    // A reference captures its receiver and index before evaluating the right side.
    fn capture_target(&mut self, target: Expr, at: oxc_span::Span) -> (Vec<Stmt>, Expr) {
        let TypedNode {
            node: target,
            ty,
            span: original_span,
        } = target;
        match target {
            TypedExpression::Index(index) => {
                let (object_save, object) = self.capture_reference_value(index.object, at);
                let (index_save, index) = self.capture_reference_value(index.index, at);
                (
                    vec![object_save, index_save],
                    node(
                        TypedExpression::Index(TypedIndex {
                            object: Box::new(object),
                            index: Box::new(index),
                        }),
                        ty,
                        at,
                    ),
                )
            }
            TypedExpression::Field(field) => {
                let (save, object) = self.capture_reference_value(field.object, at);
                (
                    vec![save],
                    node(
                        TypedExpression::Field(TypedFieldAccess {
                            object: Box::new(object),
                            field: field.field,
                        }),
                        ty,
                        at,
                    ),
                )
            }
            node => (
                vec![],
                TypedNode {
                    node,
                    ty,
                    span: original_span,
                },
            ),
        }
    }

    fn capture_reference_value(&mut self, value: Box<Expr>, at: oxc_span::Span) -> (Stmt, Expr) {
        let name = self.fresh("reference");
        let ty = value.ty.clone();
        let save = stmt(
            TypedStatement::Let(TypedLet {
                name,
                ty: ty.clone(),
                mutability: Mutability::Immutable,
                initializer: Some(value),
                span: span(at),
            }),
            at,
        );
        (save, var(name, ty, at))
    }
}
