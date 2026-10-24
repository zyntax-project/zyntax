use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::fmt::Write;
use std::sync::Arc;

use oxc_allocator::Allocator;
use oxc_ast::ast as js;
use oxc_ast_visit::{Visit, walk};
use oxc_parser::{ParseOptions, Parser};
use oxc_span::{GetSpan, SourceType};
use zyntax_typed_ast::checker_interface::{
    CheckerExport, CheckerField, CheckerFunction as Signature, CheckerInterface, CheckerParameter,
    CheckerType, CheckerTypeParameter, CheckerView,
};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::{
    CallingConvention, FieldDef, InternedString, Mutability, NullabilityKind, PrimitiveType, Span,
    Type, TypeId, TypeMetadata, TypeRegistry, Visibility,
};

use crate::{Error, ErrorKind, Indexing, ModuleDependency, ModuleId, Options, Soundness};

mod arrays;
mod bitwise;
mod classes;
mod constants;
mod enums;
mod generics;
mod globals;
mod math;
mod modules;
mod namespaces;
mod params;
mod statements;
mod timing;
pub(crate) use modules::dependency_source;
mod unions;
use arrays::Integral;
use arrays::{ArrayResult, FlowState, array_type, element_type};
use classes::FunctionRole;
use constants::Constant;

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
/// Values the standard library declares globally, which the frontend does not model yet.
fn library_value(name: &str) -> bool {
    const NAMES: &[&str] = &[
        "Object",
        "Function",
        "String",
        "Number",
        "Boolean",
        "Symbol",
        "BigInt",
        "Date",
        "RegExp",
        "Error",
        "EvalError",
        "RangeError",
        "ReferenceError",
        "SyntaxError",
        "TypeError",
        "URIError",
        "AggregateError",
        "SuppressedError",
        "Array",
        "Map",
        "Set",
        "WeakMap",
        "WeakSet",
        "WeakRef",
        "FinalizationRegistry",
        "Promise",
        "Proxy",
        "Reflect",
        "JSON",
        "Math",
        "Intl",
        "Atomics",
        "Iterator",
        "ArrayBuffer",
        "SharedArrayBuffer",
        "DataView",
        "Int8Array",
        "Uint8Array",
        "Uint8ClampedArray",
        "Int16Array",
        "Uint16Array",
        "Int32Array",
        "Uint32Array",
        "Float16Array",
        "Float32Array",
        "Float64Array",
        "BigInt64Array",
        "BigUint64Array",
        "DisposableStack",
        "AsyncDisposableStack",
        "globalThis",
        "parseInt",
        "parseFloat",
        "isNaN",
        "isFinite",
        "encodeURI",
        "encodeURIComponent",
        "decodeURI",
        "decodeURIComponent",
        "escape",
        "unescape",
        "eval",
    ];
    NAMES.contains(&name)
}

/// Types the standard and DOM libraries declare, which the frontend does not model yet.
fn library_type(name: &str) -> bool {
    const NAMES: &[&str] = &[
        "Object",
        "Function",
        "String",
        "Number",
        "Boolean",
        "Symbol",
        "BigInt",
        "Date",
        "RegExp",
        "Error",
        "EvalError",
        "RangeError",
        "ReferenceError",
        "SyntaxError",
        "TypeError",
        "URIError",
        "AggregateError",
        "Array",
        "ReadonlyArray",
        "Map",
        "Set",
        "ReadonlyMap",
        "ReadonlySet",
        "WeakMap",
        "WeakSet",
        "WeakRef",
        "Promise",
        "PromiseLike",
        "Iterable",
        "Iterator",
        "IterableIterator",
        "IteratorResult",
        "AsyncIterable",
        "AsyncIterator",
        "AsyncIterableIterator",
        "Generator",
        "AsyncGenerator",
        "GeneratorFunction",
        "ArrayLike",
        "Record",
        "Partial",
        "Required",
        "Readonly",
        "Pick",
        "Omit",
        "Exclude",
        "Extract",
        "NonNullable",
        "ReturnType",
        "Parameters",
        "ConstructorParameters",
        "InstanceType",
        "ThisParameterType",
        "OmitThisParameter",
        "ThisType",
        "Awaited",
        "Uppercase",
        "Lowercase",
        "Capitalize",
        "Uncapitalize",
        "ArrayBuffer",
        "SharedArrayBuffer",
        "ArrayBufferLike",
        "ArrayBufferView",
        "DataView",
        "Int8Array",
        "Uint8Array",
        "Uint8ClampedArray",
        "Int16Array",
        "Uint16Array",
        "Int32Array",
        "Uint32Array",
        "Float32Array",
        "Float64Array",
        "BigInt64Array",
        "BigUint64Array",
        "JSON",
        "Math",
        "Intl",
        "Atomics",
        "Reflect",
        "ProxyHandler",
        "PropertyKey",
        "PropertyDescriptor",
        "PropertyDescriptorMap",
        "TemplateStringsArray",
        "IArguments",
        "CallableFunction",
        "NewableFunction",
        "ObjectConstructor",
        "FunctionConstructor",
        "StringConstructor",
        "NumberConstructor",
        "BooleanConstructor",
        "ArrayConstructor",
        "DateConstructor",
        "RegExpConstructor",
        "ErrorConstructor",
        "SymbolConstructor",
        "PromiseConstructor",
        "MapConstructor",
        "SetConstructor",
        "RegExpMatchArray",
        "RegExpExecArray",
        "Console",
        "Window",
        "Document",
        "Element",
        "Node",
        "NodeList",
        "Event",
        "EventTarget",
        "HTMLCollection",
        "CSSStyleDeclaration",
        "Storage",
        "Location",
        "Navigator",
        "Response",
        "Request",
        "Headers",
        "URL",
        "URLSearchParams",
        "Blob",
        "File",
        "FormData",
        "AbortController",
        "AbortSignal",
        "TextEncoder",
        "TextDecoder",
        "Worker",
        "MessageEvent",
        "KeyboardEvent",
        "MouseEvent",
        "CustomEvent",
        "XMLHttpRequest",
        "WebSocket",
        "Image",
        "Audio",
        "Text",
        "Comment",
        "DocumentFragment",
        "MutationObserver",
        "IntersectionObserver",
    ];
    NAMES.contains(&name) || name.starts_with("HTML") || name.starts_with("SVG")
}
/// The native type of a record or class field: a number, boolean or string.
fn field_primitive(ty: &Type) -> Option<PrimitiveType> {
    match ty {
        Type::Primitive(
            ty @ (PrimitiveType::F64 | PrimitiveType::Bool | PrimitiveType::String),
        ) => Some(*ty),
        _ => None,
    }
}
fn is_true(test: &js::Expression<'_>) -> bool {
    matches!(test, js::Expression::BooleanLiteral(b) if b.value)
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
    span: oxc_span::Span,
}
type Fields<'ast> = SmallVec<[(&'ast str, Field); 4]>;
struct Binding {
    name: InternedString,
    symbol: Option<oxc_semantic::SymbolId>,
    ty: Type,
    view: Option<InternedString>,
    mutable: bool,
    initialized: bool,
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

/// `var` bindings that need function-level storage, and `var` uses that cannot be lowered.
struct VarPlan {
    hoisted: FxHashSet<oxc_semantic::SymbolId>,
    rejected: Option<(&'static str, oxc_span::Span)>,
}

/// Classifies `var` bindings by how far their uses reach beyond the declaring block.
struct VarExtents<'s> {
    scoping: &'s oxc_semantic::Scoping,
    /// Enclosing blocks, each marked when it is a `for` header scope.
    containers: Vec<(oxc_span::Span, bool)>,
    declarations:
        FxHashMap<oxc_semantic::SymbolId, SmallVec<[(oxc_span::Span, (oxc_span::Span, bool)); 1]>>,
    /// Reference positions, each marked when it is inside a nested function body.
    references: FxHashMap<oxc_semantic::SymbolId, SmallVec<[(u32, bool); 2]>>,
    functions: usize,
    program: oxc_span::Span,
}

impl VarExtents<'_> {
    fn plan(&self) -> VarPlan {
        let mut plan = VarPlan {
            hoisted: FxHashSet::default(),
            rejected: None,
        };
        let reject = |plan: &mut VarPlan, reason: &'static str, at: oxc_span::Span| {
            if plan
                .rejected
                .is_none_or(|(_, first)| at.start < first.start)
            {
                plan.rejected = Some((reason, at));
            }
        };
        for (symbol, declarations) in &self.declarations {
            let (first, (container, _)) = declarations
                .iter()
                .min_by_key(|(at, _)| at.start)
                .copied()
                .unwrap();
            // A module var reached from a function body is a module global.
            let module =
                container == self.program && declarations.iter().all(|(_, (c, _))| *c == container);
            let references: SmallVec<[u32; 2]> = self
                .references
                .get(symbol)
                .into_iter()
                .flatten()
                .filter(|(_, deferred)| !(module && *deferred))
                .map(|(at, _)| *at)
                .collect();
            if references.iter().any(|at| *at <= first.start) {
                reject(&mut plan, "var used before its declaration", first);
                continue;
            }
            let confined = declarations.iter().all(|(_, (c, _))| *c == container)
                && references.iter().all(|at| *at < container.end);
            if confined {
                continue;
            }
            if declarations.iter().any(|(_, (_, header))| *header) {
                reject(
                    &mut plan,
                    "var declared in a for initializer used outside the loop",
                    first,
                );
            } else {
                plan.hoisted.insert(*symbol);
            }
        }
        plan
    }

    fn contained<T>(
        &mut self,
        span: oxc_span::Span,
        header: bool,
        walk: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.containers.push((span, header));
        let result = walk(self);
        self.containers.pop();
        result
    }
}

impl<'ast> Visit<'ast> for VarExtents<'_> {
    fn visit_program(&mut self, it: &js::Program<'ast>) {
        self.program = it.span;
        self.contained(it.span, false, |this| walk::walk_program(this, it));
    }
    fn visit_function(&mut self, it: &js::Function<'ast>, flags: oxc_semantic::ScopeFlags) {
        self.functions += 1;
        walk::walk_function(self, it, flags);
        self.functions -= 1;
    }
    fn visit_arrow_function_expression(&mut self, it: &js::ArrowFunctionExpression<'ast>) {
        self.functions += 1;
        walk::walk_arrow_function_expression(self, it);
        self.functions -= 1;
    }
    fn visit_class(&mut self, it: &js::Class<'ast>) {
        self.functions += 1;
        walk::walk_class(self, it);
        self.functions -= 1;
    }
    fn visit_block_statement(&mut self, it: &js::BlockStatement<'ast>) {
        self.contained(it.span, false, |this| walk::walk_block_statement(this, it));
    }
    fn visit_function_body(&mut self, it: &js::FunctionBody<'ast>) {
        self.contained(it.span, false, |this| walk::walk_function_body(this, it));
    }
    fn visit_for_statement(&mut self, it: &js::ForStatement<'ast>) {
        self.contained(it.span, true, |this| walk::walk_for_statement(this, it));
    }
    fn visit_for_in_statement(&mut self, it: &js::ForInStatement<'ast>) {
        self.contained(it.span, true, |this| walk::walk_for_in_statement(this, it));
    }
    fn visit_for_of_statement(&mut self, it: &js::ForOfStatement<'ast>) {
        self.contained(it.span, true, |this| walk::walk_for_of_statement(this, it));
    }
    fn visit_switch_statement(&mut self, it: &js::SwitchStatement<'ast>) {
        self.contained(it.span, false, |this| walk::walk_switch_statement(this, it));
    }
    fn visit_variable_declaration(&mut self, it: &js::VariableDeclaration<'ast>) {
        if it.kind.as_str() == "var" {
            let container = *self.containers.last().unwrap();
            for d in &it.declarations {
                if let js::BindingPattern::BindingIdentifier(id) = &d.id {
                    if let Some(symbol) = id.symbol_id.get() {
                        self.declarations
                            .entry(symbol)
                            .or_default()
                            .push((d.span, container));
                    }
                }
            }
        }
        walk::walk_variable_declaration(self, it);
    }
    fn visit_identifier_reference(&mut self, id: &js::IdentifierReference<'ast>) {
        let reference = self.scoping.get_reference(id.reference_id.get().unwrap());
        if let Some(symbol) = reference.symbol_id() {
            self.references
                .entry(symbol)
                .or_default()
                .push((id.span.start, self.functions > 0));
        }
    }
}

/// How a read that may precede its assignment is reported.
#[derive(Clone, Copy)]
enum Unassigned {
    /// Declared without a value: TypeScript rejects the read.
    Declared,
    /// Hoisted out of its block: the read would observe `undefined`.
    Hoisted,
}

/// Function-level storage for `var` bindings declared in nested blocks.
#[derive(Default)]
struct FunctionVars {
    /// Index in `scopes` of the function's outermost scope.
    scope: usize,
    /// Uninitialized mutable declarations, emitted at the top of the body.
    hoisted: Vec<Stmt>,
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
        let Some(owner) = self.owner else {
            return false;
        };
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
            == Some(owner)
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

/// Parse a source file, using `file` as its module identity.
///
/// Resolvers should use [`parse_module`] to separate canonical identity from
/// diagnostic filenames and avoid treating alternate paths as separate modules.
pub fn parse_program(source: &str, file: &str, options: Options) -> Result<TypedProgram> {
    parse_module(source, file, ModuleId::new(file), options)
}

/// Parse using a resolver-provided identity instead of the diagnostic filename.
pub fn parse_module(
    source: &str,
    file: &str,
    module: ModuleId,
    options: Options,
) -> Result<TypedProgram> {
    parse_module_with_dependencies(source, file, module, options, &[])
}

/// Check against dependency interfaces without reading or cloning their executable bodies.
pub fn parse_module_with_dependencies(
    source: &str,
    file: &str,
    module: ModuleId,
    options: Options,
    dependencies: &[ModuleDependency<'_>],
) -> Result<TypedProgram> {
    for dependency in dependencies {
        dependency.program.type_registry.reserve_type_ids();
    }
    let mut program = with_parsed(source, file, std::convert::identity, |program, semantic| {
        lower_parsed(program, semantic, module, options, dependencies)
    })?;
    program
        .source_files
        .push(zyntax_typed_ast::source::SourceFile::new(
            file.to_owned(),
            source.to_owned(),
        ));
    Ok(program)
}

pub(crate) fn with_parsed<T, E>(
    source: &str,
    file: &str,
    error: impl Fn(Error) -> E,
    lower: impl FnOnce(&js::Program<'_>, &oxc_semantic::Semantic<'_>) -> std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    let allocator = Allocator::default();
    let declaration = crate::ambient::is_declaration_file(file);
    // TypeScript checks every file under strict-mode rules.
    let source_type = SourceType::ts()
        .with_module(true)
        .with_typescript_definition(declaration);
    // Declaration files report stray statements as ambient-context errors.
    let options = ParseOptions {
        allow_return_outside_function: declaration,
        ..ParseOptions::default()
    };
    let parsed = Parser::new(&allocator, source, source_type)
        .with_options(options)
        .parse();
    if let Some(diagnostic) = parsed.diagnostics.first() {
        let reported = Error::from_oxc(parse_error_kind(&diagnostic.message), diagnostic);
        let at = oxc_span::Span::new(reported.start as u32, reported.end as u32);
        return Err(error(
            match reference_grammar_form(&diagnostic.message, source, at) {
                Some((kind, message)) => Error::new(kind, message, at),
                None => reported,
            },
        ));
    }
    let semantic = oxc_semantic::SemanticBuilder::new()
        .with_check_syntax_error(true)
        .build(&parsed.program);
    if let Some(diagnostic) = semantic.diagnostics.first() {
        // Legacy numeric literals are scanner errors in TypeScript.
        let kind = if is_legacy_numeric_literal(&diagnostic.message) {
            ErrorKind::Syntax
        } else {
            ErrorKind::Type
        };
        return Err(error(Error::from_oxc(kind, diagnostic)));
    }
    if declaration && let Some(diagnostic) = crate::ambient::check(&parsed.program) {
        return Err(error(diagnostic));
    }
    if let Some(at) = missing_member_separator(&parsed.program, source) {
        return Err(error(Error::new(ErrorKind::Syntax, "';' expected", at)));
    }
    // `@ts-nocheck` turns TypeScript's semantic errors off; native code cannot be built past one.
    let nocheck = parsed
        .program
        .comments
        .iter()
        .find(|c| source[c.span.start as usize..c.span.end as usize].contains("@ts-nocheck"));
    match (lower(&parsed.program, &semantic.semantic), nocheck) {
        (Err(_), Some(comment)) => Err(error(Error::new(
            ErrorKind::Unsupported,
            "type errors in files that disable checking with @ts-nocheck are not implemented yet",
            comment.span,
        ))),
        (result, _) => result,
    }
}

/// Type members on one line need a `;` or `,` between them, which the parser does not require.
fn missing_member_separator(program: &js::Program<'_>, source: &str) -> Option<oxc_span::Span> {
    struct Members<'s> {
        source: &'s str,
        found: Option<oxc_span::Span>,
    }
    impl Members<'_> {
        fn check(&mut self, members: &[js::TSSignature<'_>]) {
            for pair in members.windows(2) {
                let (end, start) = (pair[0].span().end as usize, pair[1].span().start as usize);
                let own = &self.source[pair[0].span().start as usize..end];
                let between = self.source.get(end..start).unwrap_or("");
                if self.found.is_none()
                    && !own.ends_with([';', ','])
                    && !between.contains([';', ',', '\n', '\r'])
                {
                    self.found = Some(pair[1].span());
                }
            }
        }
    }
    impl<'a> Visit<'a> for Members<'_> {
        fn visit_ts_type_literal(&mut self, it: &js::TSTypeLiteral<'a>) {
            self.check(&it.members);
            walk::walk_ts_type_literal(self, it);
        }
        fn visit_ts_interface_body(&mut self, it: &js::TSInterfaceBody<'a>) {
            self.check(&it.body);
            walk::walk_ts_interface_body(self, it);
        }
    }
    let mut members = Members {
        source,
        found: None,
    };
    members.visit_program(program);
    members.found
}

/// A type naming a declaration that has not resolved yet.
const UNRESOLVED_REFERENCE: &str = "recursive or unresolved type references";

// Grammar rules oxc enforces while parsing but TypeScript reports from its checker.
const CHECKER_GRAMMAR_ERRORS: &[&str] = &[
    "ambient context",
    "modifier cannot be used here",
    "modifier must precede",
    "modifier cannot appear on",
    "modifier already seen",
    "clause already seen",
    "Only a single declaration is allowed in a",
    "expression is only allowed in a generator body",
    "is not allowed in property names",
    "may not be declared using a binding pattern",
    "Cannot assign to this expression",
    "Missing initializer in const declaration",
    "statement can only be used within a function body",
    "Computed property names are not allowed in enums",
    "Class name cannot be",
    "Interface name cannot be",
    "Classes may not have a static property named",
    "Classes can only extend a single class",
    "Generators can only be declared at the top level",
    "cannot have an initializer because it is marked",
    "cannot have a question mark and an initializer",
    "A rest parameter cannot be optional",
    "Missing initializer in destructuring declaration",
    "Invalid rest element target",
    "as an identifier in an async context",
    "as an identifier in a generator context",
    "overload signature cannot be declared as a generator",
    "Interface declaration cannot have",
    "accessor must not have any formal parameters",
    "An index signature must have a type annotation",
    "A rest element must be last in a destructuring pattern",
    "A required element cannot follow an optional element",
    "A required parameter cannot follow an optional parameter",
    "modifier can only appear on",
    "cannot have an implementation because it is marked",
    "modifier cannot be used with",
    "Decorators are not valid here",
    "Modifiers cannot appear here",
    "An accessibility modifier cannot be used with",
    "Classes can't have an element named '#constructor'",
    "Classes can't have a field named 'constructor'",
    "A definite assignment assertion '!' is not permitted in this context",
    "Declarations with initializers cannot also have definite assignment assertions",
    "A rest element cannot have an initializer",
    "A rest element cannot follow another rest element",
    "Logical expressions and coalesce expressions cannot be mixed",
    "Tagged template expressions are not permitted in an optional chain",
    "The only valid property accesses on import are",
    "An interface can only extend an identifier/qualified-name",
    "An accessor cannot have type parameters",
    "'extends' clause must precede 'implements' clause",
    "'extends' list cannot be empty",
    "'implements' list cannot be empty",
    "Type annotation cannot appear on a constructor declaration",
    "Type parameters cannot appear on a constructor declaration",
    "Type parameter list cannot be empty",
    "Type argument list cannot be empty",
    "An enum member cannot have a numeric name",
    "An export assignment cannot be used in a namespace",
    "A default export can only be used in an ECMAScript-style module",
    "An index signature parameter type must be",
    "A 'set' accessor must have exactly one parameter",
    "A 'set' accessor cannot have an optional parameter",
    "Only ambient modules can use quoted names",
    "in regular expression literal",
    "statement may not be `async`",
    "Constructor can't have get/set modifier",
    "Constructor can't be a generator",
    "Using declarations may not have binding patterns",
    "'readonly' type modifier is only permitted on array and tuple literal types",
    "modifier cannot be used on a type parameter",
    "String literal expected",
];

fn parse_error_kind(message: &str) -> ErrorKind {
    if CHECKER_GRAMMAR_ERRORS
        .iter()
        .any(|fragment| message.contains(fragment))
    {
        ErrorKind::Type
    } else {
        ErrorKind::Syntax
    }
}

/// A form TypeScript parses and rejects from its checker, or rejects while parsing, where oxc's
/// parser reports a generic message. Returns TypeScript's category and message for the form.
fn reference_grammar_form(
    message: &str,
    source: &str,
    at: oxc_span::Span,
) -> Option<(ErrorKind, &'static str)> {
    let start = (at.start as usize).min(source.len());
    let end = (at.end as usize).clamp(start, source.len());
    let before = source[..start].trim_end();
    let token = source[start..].trim_start();
    let line = before
        .rsplit(['\n', '\r'])
        .next()
        .unwrap_or("")
        .trim_start();
    let ends_statement = token.is_empty() || token.starts_with([';', '}', ')']);
    let checker = |message| Some((ErrorKind::Type, message));
    match message {
        "Unexpected token" => {
            if is_decorator(line) && starts_with_word(token, DECORATED_STATEMENTS) {
                return checker("Decorators are not valid here.");
            }
            if empty_for_declaration(source, start) {
                return checker("Variable declaration list cannot be empty.");
            }
            if ends_statement {
                if ends_with_word(before, "var") || ends_with_word(before, "const") {
                    return checker("Variable declaration list cannot be empty.");
                }
                if ends_with_word(before, "let") {
                    return checker(
                        "Identifier expected. 'let' is a reserved word in strict mode.",
                    );
                }
                if before.ends_with(',') && starts_with_word(line, &["var", "let", "const"]) {
                    return checker("Trailing comma not allowed.");
                }
            }
            if token.starts_with('{') && ends_with_word(before, "implements") {
                return checker("'implements' list cannot be empty.");
            }
            if line == "[" && token.starts_with(']') {
                return checker("An index signature must have exactly one parameter.");
            }
            if token.starts_with('#')
                && enclosing_brace(source, start).is_some_and(|open| {
                    let head = source[..open]
                        .trim_end()
                        .trim_end_matches(is_identifier_char);
                    ends_with_word(head.trim_end(), "enum")
                })
            {
                return checker("An enum member cannot be named with a private identifier.");
            }
            if token.starts_with("...")
                && ["public", "private", "protected", "readonly"]
                    .iter()
                    .any(|modifier| ends_with_word(before, modifier))
            {
                return checker("A parameter property cannot be declared using a rest parameter.");
            }
            let specifier = before.trim_end_matches(is_identifier_char).trim_end();
            if starts_with_word(line, &["import"])
                && specifier
                    .strip_suffix('(')
                    .is_some_and(|callee| ends_with_word(callee.trim_end(), "require"))
            {
                return checker("String literal expected.");
            }
            None
        }
        "Expected `;` but found `Identifier`"
        | "Identifier expected. 'in' is a reserved word that cannot be used here."
            if empty_for_declaration(source, start) =>
        {
            checker("Variable declaration list cannot be empty.")
        }
        "Expected `,` or `}` but found `?`" if in_object_literal(source, start) => {
            checker("An object member cannot be declared optional.")
        }
        "Expected `,` or `}` but found `!`" if in_object_literal(source, start) => {
            checker("A definite assignment assertion '!' is not permitted in this context.")
        }
        _ if message.starts_with("Expected `{` but found") && empty_extends(before) => {
            checker("'extends' list cannot be empty.")
        }
        // TypeScript parses an assignment or update only on a left-hand-side expression.
        "Cannot assign to this expression" => {
            let target = source[start..end].trim();
            let same_line = source[..start].rsplit('\n').next().unwrap_or("").trim_end();
            let (update, operand) = match target
                .strip_prefix("++")
                .or_else(|| target.strip_prefix("--"))
            {
                Some(operand) => (true, operand.trim_start()),
                None => (
                    same_line.ends_with("++") || same_line.ends_with("--"),
                    target,
                ),
            };
            let unary = operand.starts_with(['+', '-', '!', '~'])
                || starts_with_word(operand, &["await", "delete", "typeof", "void"]);
            let postfix = operand.ends_with("++") || operand.ends_with("--");
            if update && (unary || postfix) {
                Some((ErrorKind::Syntax, "Expression expected."))
            } else if (unary || postfix) && assignment_follows(&source[end..]) {
                Some((ErrorKind::Syntax, "';' expected."))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn assignment_follows(rest: &str) -> bool {
    let rest = rest.trim_start();
    let operator = rest
        .find(|c: char| !"=+-*/%<>&|^?!".contains(c))
        .map_or(rest, |end| &rest[..end]);
    operator.ends_with('=') && !matches!(operator, "==" | "===" | "!=" | "!==" | "<=" | ">=")
}

const DECORATED_STATEMENTS: &[&str] = &[
    "enum",
    "function",
    "var",
    "let",
    "const",
    "interface",
    "type",
    "namespace",
    "module",
    "import",
    "declare",
    "async",
];

fn is_identifier_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '$'
}

fn starts_with_word(text: &str, words: &[&str]) -> bool {
    words.iter().any(|word| {
        text.strip_prefix(word)
            .is_some_and(|rest| !rest.starts_with(is_identifier_char))
    })
}

fn ends_with_word(text: &str, word: &str) -> bool {
    text.strip_suffix(word)
        .is_some_and(|rest| !rest.ends_with(|c: char| is_identifier_char(c) || c == '.'))
}

/// `@name` or `@name(...)` alone on its line.
fn is_decorator(line: &str) -> bool {
    let Some(rest) = line.strip_prefix('@') else {
        return false;
    };
    let name = rest
        .find(|c: char| !(is_identifier_char(c) || c == '.'))
        .unwrap_or(rest.len());
    let arguments = &rest[name..];
    name > 0 && (arguments.is_empty() || arguments.starts_with('(') && arguments.ends_with(')'))
}

/// `for (var of x)` and `for (var in x)`: TypeScript reads `of` and `in` as the loop keyword.
fn empty_for_declaration(source: &str, at: usize) -> bool {
    let Some(keyword) = source[..at].rfind("for") else {
        return false;
    };
    if source[..keyword].ends_with(is_identifier_char) {
        return false;
    }
    let head = &source[keyword + 3..];
    let Some(close) = head.find(')') else {
        return false;
    };
    if at > keyword + 3 + close {
        return false;
    }
    let Some(inner) = head[..close].trim_start().strip_prefix('(') else {
        return false;
    };
    let words: Vec<&str> = inner.split_whitespace().collect();
    match words.as_slice() {
        ["var" | "const", "in", ..] => true,
        ["var" | "const", "of", name] => name.chars().all(is_identifier_char),
        _ => false,
    }
}

/// The innermost `{` still open at `at`.
fn enclosing_brace(source: &str, at: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (index, c) in source[..at].char_indices().rev() {
        match c {
            '}' => depth += 1,
            '{' if depth == 0 => return Some(index),
            '{' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// The innermost open `{` before `at` starts an object literal rather than a block or pattern.
fn in_object_literal(source: &str, at: usize) -> bool {
    let Some(open) = enclosing_brace(source, at) else {
        return false;
    };
    let before = source[..open].trim_end();
    if before.ends_with("=>") {
        return false;
    }
    if before.ends_with([':', '=']) || ends_with_word(before, "return") {
        return true;
    }
    // A call argument: `f({ ... })`, but not a parameter list or statement head.
    let Some(callee) = before.strip_suffix('(') else {
        return false;
    };
    let callee = callee.trim_end();
    let name_start = callee
        .rfind(|c: char| !is_identifier_char(c))
        .map_or(0, |i| i + 1);
    let name = &callee[name_start..];
    !name.is_empty()
        && !matches!(
            name,
            "function" | "if" | "while" | "for" | "catch" | "switch" | "with"
        )
        && !ends_with_word(callee[..name_start].trim_end(), "function")
}

/// `class C extends {` or `class C extends implements I {`.
fn empty_extends(before: &str) -> bool {
    let mut search = before;
    while let Some(index) = search.rfind("extends") {
        let rest = &before[index + "extends".len()..];
        if !before[..index].ends_with(is_identifier_char) && !rest.starts_with(is_identifier_char) {
            let rest = rest.trim_start();
            return rest.starts_with('{') || starts_with_word(rest, &["implements"]);
        }
        search = &before[..index];
    }
    false
}

fn is_legacy_numeric_literal(message: &str) -> bool {
    message.starts_with("'0'-prefixed octal literals")
        || message.starts_with("Decimals with leading zeros")
}

pub(crate) fn lower_parsed(
    program: &js::Program<'_>,
    semantic: &oxc_semantic::Semantic<'_>,
    module: ModuleId,
    options: Options,
    dependencies: &[ModuleDependency<'_>],
) -> Result<TypedProgram> {
    Lowerer::new(options, module).program(program, semantic, dependencies)
}

struct Lowerer<'ast, 'program> {
    options: Options,
    module: ModuleId,
    names: RefCell<FxHashMap<&'ast str, InternedString>>,
    name_buffer: String,
    prefix_len: usize,
    registry: TypeRegistry,
    aliases: FxHashMap<InternedString, (Type, Option<InternedString>)>,
    /// Type declarations in this module, resolved or not.
    declared_types: FxHashSet<InternedString>,
    /// Inline object type literals with known layouts, by source offset.
    type_literals: FxHashMap<u32, (Type, Option<InternedString>)>,
    imported_exports: FxHashMap<InternedString, CheckerExport>,
    namespaces: Vec<namespaces::Namespace>,
    namespace_references: FxHashMap<oxc_semantic::NodeId, usize>,
    views: FxHashMap<InternedString, Arc<CheckerView>>,
    layouts: FxHashMap<Vec<(InternedString, PrimitiveType)>, TypeId>,
    compatible_layouts: FxHashMap<(TypeId, TypeId), bool>,
    signatures: FxHashMap<InternedString, Arc<Signature>>,
    /// Signatures whose callers may omit or add arguments, by native name.
    arities: FxHashMap<InternedString, params::Arity>,
    active_types: SmallVec<[(InternedString, Type); 2]>,
    active_type_rules: Option<Arc<[CheckerTypeParameter]>>,
    functions: FxHashMap<InternedString, &'program js::Function<'ast>>,
    lowered: FxHashSet<InternedString>,
    self_recursive: FxHashSet<InternedString>,
    scopes: Vec<FxHashMap<InternedString, Binding>>,
    scope_floor: usize,
    constants: FxHashMap<InternedString, Constant>,
    constant_symbols: FxHashSet<oxc_semantic::SymbolId>,
    /// The literal types of top-level constants: numbers and strings.
    literal_constants: FxHashMap<InternedString, enums::Static>,
    /// The literal type each `const` binding was declared with, by native name.
    const_literals: FxHashMap<InternedString, enums::Static>,
    enums: FxHashMap<InternedString, enums::Enum>,
    enum_views: FxHashMap<InternedString, InternedString>,
    /// Views of member types, `E.A` or `E.A | E.B`: the enum and the members they admit.
    member_views: RefCell<FxHashMap<InternedString, (InternedString, enums::Members)>>,
    /// Enums whose declarations the module body has passed.
    declared_enums: FxHashSet<InternedString>,
    /// Every enum the module declares, including those not evaluated yet.
    module_enums: FxHashSet<InternedString>,
    /// Module globals by native name, and the top-level symbols they store.
    globals: FxHashMap<InternedString, globals::Global>,
    module_symbols: FxHashMap<oxc_semantic::SymbolId, InternedString>,
    /// Lowering the module body itself, outside any function or arrow.
    top_level: bool,
    /// Bindings an ambient `declare` names, by native name; they have no native value.
    ambient: FxHashSet<InternedString>,
    /// Every declaration in a `.d.ts` file is ambient.
    declaration_file: bool,
    classes: FxHashMap<InternedString, classes::Class<'ast, 'program>>,
    class_names: FxHashMap<InternedString, InternedString>,
    class_order: Vec<InternedString>,
    current_class: Option<InternedString>,
    /// The class whose body holds the code being lowered, instance or static.
    class_body: Option<InternedString>,
    in_constructor: bool,
    /// Lowering a derived constructor's statements before its `super(...)` call.
    before_super: bool,
    /// Classes this module's class declarations extend, by source name.
    extended_classes: FxHashSet<InternedString>,
    /// Top-level functions, variables and imports, by source name.
    top_level_values: FxHashSet<InternedString>,
    hoist: FxHashSet<oxc_semantic::SymbolId>,
    vars: FunctionVars,
    /// Hoisted bindings whose reads need a proven assignment on every path.
    unassigned: FxHashMap<InternedString, Unassigned>,
    flow: FlowState,
    writes: FxHashMap<oxc_semantic::SymbolId, SmallVec<[u32; 2]>>,
    write_offsets: Vec<u32>,
    declarations: Vec<TypedNode<TypedDeclaration>>,
    returns: Option<ReturnContext>,
    loop_depth: usize,
    /// Enclosing loops, switches and labeled statements that a jump can leave.
    jumps: Vec<statements::Jump<'ast>>,
    /// A label waiting for the statement it names.
    pending_label: Option<&'ast str>,
    /// Switches whose clauses cover every member their discriminant may hold.
    exhaustive_switches: FxHashSet<u32>,
    next_name: usize,
    next_array_identity: usize,
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    fn new(options: Options, module: ModuleId) -> Self {
        let name_buffer = module.prefix();
        let prefix_len = name_buffer.len();
        let mut registry = TypeRegistry::new();
        registry.set_current_module(Some(module.name()));
        Self {
            options,
            module,
            names: RefCell::new(FxHashMap::default()),
            name_buffer,
            prefix_len,
            registry,
            aliases: FxHashMap::default(),
            type_literals: FxHashMap::default(),
            declared_types: FxHashSet::default(),
            imported_exports: FxHashMap::default(),
            namespaces: vec![],
            namespace_references: FxHashMap::default(),
            views: FxHashMap::default(),
            layouts: FxHashMap::default(),
            compatible_layouts: FxHashMap::default(),
            signatures: FxHashMap::default(),
            arities: FxHashMap::default(),
            active_types: SmallVec::new(),
            active_type_rules: None,
            functions: FxHashMap::default(),
            lowered: FxHashSet::default(),
            self_recursive: FxHashSet::default(),
            scopes: vec![],
            scope_floor: 0,
            constants: FxHashMap::default(),
            constant_symbols: FxHashSet::default(),
            literal_constants: FxHashMap::default(),
            const_literals: FxHashMap::default(),
            enums: FxHashMap::default(),
            enum_views: FxHashMap::default(),
            member_views: RefCell::new(FxHashMap::default()),
            declared_enums: FxHashSet::default(),
            module_enums: FxHashSet::default(),
            globals: FxHashMap::default(),
            module_symbols: FxHashMap::default(),
            top_level: false,
            ambient: FxHashSet::default(),
            declaration_file: false,
            classes: FxHashMap::default(),
            class_names: FxHashMap::default(),
            class_order: vec![],
            current_class: None,
            class_body: None,
            in_constructor: false,
            before_super: false,
            extended_classes: FxHashSet::default(),
            top_level_values: FxHashSet::default(),
            hoist: FxHashSet::default(),
            vars: FunctionVars::default(),
            unassigned: FxHashMap::default(),
            flow: FlowState::default(),
            writes: FxHashMap::default(),
            write_offsets: Vec::new(),
            declarations: vec![],
            returns: None,
            loop_depth: 0,
            jumps: vec![],
            pending_label: None,
            exhaustive_switches: FxHashSet::default(),
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
        self.name_buffer.truncate(self.prefix_len);
        crate::modules::append_component(&mut self.name_buffer, prefix);
        write!(&mut self.name_buffer, "${index}").unwrap();
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
        dependencies: &[ModuleDependency<'_>],
    ) -> Result<TypedProgram> {
        self.declaration_file = program.source_type.is_typescript_definition();
        WritePositions {
            scoping: semantic.scoping(),
            positions: &mut self.writes,
        }
        .visit_program(program);
        self.write_offsets = self.writes.values().flatten().copied().collect();
        self.write_offsets.sort_unstable();
        let mut extents = VarExtents {
            scoping: semantic.scoping(),
            containers: vec![],
            declarations: FxHashMap::default(),
            references: FxHashMap::default(),
            functions: 0,
            program: program.span,
        };
        extents.visit_program(program);
        let plan = extents.plan();
        if let Some((reason, at)) = plan.rejected {
            return self.unsupported(reason, at);
        }
        self.hoist = plan.hoisted;
        let reexports = self.imports(program, semantic, dependencies)?;
        // Enum initializers may name constants; types may name enums.
        self.collect_constants(program);
        self.declare_literal_types(program);
        self.declare_enums(program)?;
        self.inline_type_literals(program);
        self.class_heritage(program);
        // Declare types before signatures; resolve forward aliases to a fixed point.
        let mut pending: Vec<_> = program
            .body
            .iter()
            .filter_map(modules::declaration)
            .filter(|s| {
                matches!(
                    s,
                    js::Declaration::TSInterfaceDeclaration(_)
                        | js::Declaration::TSTypeAliasDeclaration(_)
                        | js::Declaration::ClassDeclaration(_)
                )
            })
            .collect();
        for item in &pending {
            let (name, span) = match item {
                js::Declaration::TSInterfaceDeclaration(i) => (i.id.name.as_str(), i.span),
                js::Declaration::TSTypeAliasDeclaration(a) => (a.id.name.as_str(), a.span),
                js::Declaration::ClassDeclaration(c) => match &c.id {
                    Some(id) => (id.name.as_str(), c.span),
                    None => continue,
                },
                _ => continue,
            };
            let name = self.name(name);
            if !self.declared_types.insert(name) {
                return self.unsupported("declaration merging", span);
            }
        }
        while !pending.is_empty() {
            let before = pending.len();
            let mut next = vec![];
            let mut errors = vec![];
            for item in pending {
                match self.type_declaration(item) {
                    Ok(()) => {}
                    Err(e) => {
                        errors.push(e);
                        next.push(item);
                    }
                }
            }
            if next.len() == before {
                // A program error wins over a gap; a gap is reported where it occurs, not at a
                // declaration waiting on it.
                let at = errors
                    .iter()
                    .position(|e| e.kind != ErrorKind::Unsupported)
                    .or_else(|| {
                        errors
                            .iter()
                            .position(|e| !e.message.starts_with(UNRESOLVED_REFERENCE))
                    })
                    .unwrap_or(errors.len() - 1);
                return Err(errors.swap_remove(at));
            }
            pending = next;
        }
        // Literals whose fields name aliases resolve once the aliases do.
        self.inline_type_literals(program);
        self.collect_constants(program);
        for item in &program.body {
            if let Some(f) = modules::function(item) {
                self.signature(f)?;
            }
        }
        self.class_signatures()?;
        self.externs();
        self.module_bindings(program, semantic)?;
        let dependencies = self.dependencies(semantic)?;
        let mut visiting = FxHashSet::default();
        for item in &program.body {
            if let Some(f) = modules::function(item) {
                let mut pending = vec![(self.name(modules::function_name(f)), false)];
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
        self.lower_classes()?;
        // In the module body a global is unavailable until its declaration runs.
        for binding in self.scopes[0].values_mut() {
            binding.initialized = false;
        }
        self.top_level = true;
        self.scopes.push(FxHashMap::default());
        self.vars = FunctionVars {
            scope: self.scopes.len() - 1,
            hoisted: vec![],
        };
        self.reserve(&program.body)?;
        let mut body = vec![];
        for item in &program.body {
            if matches!(
                item,
                js::Statement::ImportDeclaration(_)
                    | js::Statement::ExportNamedDeclaration(_)
                    | js::Statement::ExportFromDeclaration(_)
                    | js::Statement::ExportAllDeclaration(_)
                    | js::Statement::ExportDefaultDeclaration(_)
            ) {
                continue;
            }
            if let js::Statement::ExportDeclaration(export) = item {
                if !matches!(
                    export.declaration,
                    js::Declaration::FunctionDeclaration(_)
                        | js::Declaration::TSInterfaceDeclaration(_)
                        | js::Declaration::TSTypeAliasDeclaration(_)
                ) {
                    return self.unsupported("this exported declaration", export.span);
                }
                continue;
            }
            if let js::Statement::TSEnumDeclaration(declaration) = item {
                body.extend(self.run_enum_declaration(declaration));
                continue;
            }
            if !matches!(
                item,
                js::Statement::FunctionDeclaration(_)
                    | js::Statement::TSInterfaceDeclaration(_)
                    | js::Statement::TSTypeAliasDeclaration(_)
                    | js::Statement::ClassDeclaration(_)
            ) {
                body.extend(self.statement(item)?);
            }
        }
        self.pop_scope();
        self.top_level = false;
        body.splice(0..0, std::mem::take(&mut self.vars.hoisted));
        if !body.is_empty() {
            self.declarations.push(node(
                TypedDeclaration::Initializer(TypedInitializer {
                    body: TypedBlock {
                        statements: body,
                        span: span(program.span),
                    },
                    annotations: vec![strict_fp(self.name("strict_fp"))],
                    module: Some(self.module.name()),
                }),
                unit(),
                program.span,
            ));
        }
        let checker_interface = self.export_interface(program, reexports)?;
        Ok(TypedProgram {
            checker_interface,
            declarations: self.declarations,
            language: Some(intern("typescript")),
            span: span(program.span),
            source_files: vec![],
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
            .filter_map(|(name, f)| Some((f.id.as_ref()?.symbol_id.get()?, *name)))
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
            calls.owner = function.id.as_ref().and_then(|id| id.symbol_id.get());
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

    fn type_declaration(&mut self, item: &'program js::Declaration<'ast>) -> Result<()> {
        match item {
            js::Declaration::ClassDeclaration(class) => self.declare_class(class)?,
            js::Declaration::TSInterfaceDeclaration(i) => {
                if i.type_parameters.is_some() {
                    return self.unsupported("generic or extending interfaces", i.span);
                }
                let name = self.name(i.id.name.as_str());
                if self.aliases.contains_key(&name) {
                    return self.unsupported("declaration merging", i.span);
                }
                let fields = self.fields(&i.body.body)?;
                if i.extends.is_empty() {
                    let ty = self.layout(fields.iter().map(|(name, field)| (*name, field.ty)));
                    let view = self.record_view(i.id.name.as_str(), &ty, fields);
                    self.aliases.insert(name, (ty, Some(view)));
                } else {
                    let (ty, view) = self.extended_interface(i, fields)?;
                    self.aliases.insert(name, (ty, Some(view)));
                }
            }
            js::Declaration::TSTypeAliasDeclaration(a) => {
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
                    let view = self.record_view(a.id.name.as_str(), &ty, fields);
                    (ty, Some(view))
                } else {
                    self.ty(&a.type_annotation)?
                };
                self.aliases.insert(name, (ty, view));
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Registers every inline object type literal whose fields are known, so an annotation
    /// anywhere names its layout. A literal that needs a generic parameter or another
    /// unsupported field stays unregistered and reports Unsupported where it is used.
    fn inline_type_literals(&mut self, program: &'program js::Program<'ast>) {
        struct Literals<'l, 'ast, 'program> {
            lowerer: &'l mut Lowerer<'ast, 'program>,
        }
        impl<'ast> Visit<'ast> for Literals<'_, 'ast, '_> {
            fn visit_ts_type_literal(&mut self, it: &js::TSTypeLiteral<'ast>) {
                self.lowerer.register_type_literal(it);
                walk::walk_ts_type_literal(self, it);
            }
        }
        Literals { lowerer: self }.visit_program(program);
    }

    fn register_type_literal(&mut self, literal: &js::TSTypeLiteral<'ast>) {
        if self.type_literals.contains_key(&literal.span.start) {
            return;
        }
        let Ok(fields) = self.fields(&literal.members) else {
            return;
        };
        let ty = self.layout(fields.iter().map(|(name, field)| (*name, field.ty)));
        let label = format!("literal{}", literal.span.start);
        let view = self.record_view(&label, &ty, fields);
        self.type_literals
            .insert(literal.span.start, (ty, Some(view)));
    }

    fn fields(&mut self, members: &[js::TSSignature<'ast>]) -> Result<Fields<'ast>> {
        let mut fields = Fields::new();
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
            let (ty, view) = self.ty(&annotation.type_annotation)?;
            self.plain_enum(view, "enum-typed fields", p.span)?;
            let Some(ty) = field_primitive(&ty) else {
                return self.unsupported("reference-valued object fields", p.span);
            };
            fields.push((
                name,
                Field {
                    ty,
                    readonly: p.readonly,
                    span: p.span,
                },
            ));
        }
        fields.sort_by_key(|(name, _)| *name);
        if let Some(pair) = fields.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return self.error(ErrorKind::Type, "duplicate property", pair[1].1.span);
        }
        Ok(fields)
    }

    fn record_view(&mut self, name: &str, ty: &Type, fields: Fields<'ast>) -> InternedString {
        let fields = fields
            .into_iter()
            .map(|(name, field)| CheckerField {
                name: self.name(name),
                ty: Type::Primitive(field.ty),
                readonly: field.readonly,
            })
            .collect();
        self.record_view_of(name, ty, fields)
    }

    fn record_view_of(
        &mut self,
        name: &str,
        ty: &Type,
        fields: Vec<CheckerField>,
    ) -> InternedString {
        self.name_buffer.truncate(self.prefix_len);
        self.name_buffer.push_str("view$");
        crate::modules::append_component(&mut self.name_buffer, name);
        let name = intern(&self.name_buffer);
        self.views.insert(
            name,
            Arc::new(CheckerView {
                ty: ty.clone(),
                fields,
                nominal: false,
            }),
        );
        name
    }

    /// An interface's own fields joined with those of the record interfaces it extends: a field
    /// both declare must have one type.
    fn extended_interface(
        &mut self,
        interface: &js::TSInterfaceDeclaration<'ast>,
        own: Fields<'ast>,
    ) -> Result<(Type, InternedString)> {
        let mut fields: SmallVec<[CheckerField; 4]> = own
            .iter()
            .map(|(name, field)| CheckerField {
                name: self.name(name),
                ty: Type::Primitive(field.ty),
                readonly: field.readonly,
            })
            .collect();
        let declared = fields.len();
        for heritage in &interface.extends {
            let js::TSTypeName::IdentifierReference(base) = &heritage.type_name else {
                return self.unsupported("extending qualified type names", heritage.span);
            };
            if heritage.type_arguments.is_some() {
                return self.unsupported("extending generic interfaces", heritage.span);
            }
            let base_name = self.name(base.name.as_str());
            let Some((_, base_view)) = self.aliases.get(&base_name).cloned() else {
                if self.declared_types.contains(&base_name) {
                    return self.unsupported(UNRESOLVED_REFERENCE, heritage.span);
                }
                if library_type(base.name.as_str()) {
                    return self
                        .unsupported(&format!("library type `{}`", base.name), heritage.span);
                }
                return self.error(
                    ErrorKind::Type,
                    format!("cannot find name `{}`", base.name),
                    heritage.span,
                );
            };
            if self.class_names.contains_key(&base_name) {
                return self.unsupported("interfaces extending classes", heritage.span);
            }
            let Some(view) = base_view.and_then(|v| self.views.get(&v)).cloned() else {
                return self.unsupported("interfaces extending this type", heritage.span);
            };
            for field in &view.fields {
                match fields.iter().position(|f| f.name == field.name) {
                    Some(index) if fields[index].ty != field.ty => {
                        let property = field.name.resolve_global().unwrap_or_default();
                        let message = if index < declared {
                            format!(
                                "interface `{}` incorrectly extends `{}`: property `{property}` has another type",
                                interface.id.name, base.name
                            )
                        } else {
                            format!(
                                "interface `{}` cannot extend two types whose property `{property}` differs",
                                interface.id.name
                            )
                        };
                        return self.error(ErrorKind::Type, message, heritage.span);
                    }
                    Some(_) => {}
                    None => fields.push(field.clone()),
                }
            }
        }
        // Record layouts order fields by name.
        fields.sort_by_key(|f| f.name.resolve_global().unwrap_or_default());
        let key: SmallVec<[(InternedString, PrimitiveType); 4]> = fields
            .iter()
            .map(|f| (f.name, field_primitive(&f.ty).unwrap()))
            .collect();
        let ty = self.layout_of(key);
        let view = self.record_view_of(interface.id.name.as_str(), &ty, fields.into_vec());
        Ok((ty, view))
    }

    fn layout(&mut self, fields: impl Iterator<Item = (&'ast str, PrimitiveType)>) -> Type {
        let key: SmallVec<[(InternedString, PrimitiveType); 4]> =
            fields.map(|(name, ty)| (self.name(name), ty)).collect();
        self.layout_of(key)
    }

    fn layout_of(&mut self, key: SmallVec<[(InternedString, PrimitiveType); 4]>) -> Type {
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
            js::TSType::TSLiteralType(l) if self.literal_type(l).is_some() => {
                return self.literal_type(l).unwrap();
            }
            js::TSType::TSUnionType(u) => {
                if let Some(members) = self.enum_union(u) {
                    return members;
                }
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
                let (element, view) = self.ty(&a.element_type)?;
                self.plain_enum(view, "arrays of enum values", a.span)?;
                self.check_array_element(&element, a.span)?;
                (array_type(element), None)
            }
            js::TSType::TSTypeReference(r) if r.type_arguments.is_some() => {
                let js::TSTypeName::IdentifierReference(name) = &r.type_name else {
                    return self.unsupported("qualified type names", r.span);
                };
                if name.name != "Array" {
                    if self.namespace_references.contains_key(&name.node_id.get()) {
                        return self.error(
                            ErrorKind::Type,
                            "namespace cannot be used as a generic type",
                            r.span,
                        );
                    }
                    return self.unsupported("this generic type reference", r.span);
                }
                if r.type_arguments.as_ref().unwrap().params.len() != 1 {
                    return self.error(ErrorKind::Type, "Array requires one type argument", r.span);
                }
                let (element, view) = self.ty(&r.type_arguments.as_ref().unwrap().params[0])?;
                self.plain_enum(view, "arrays of enum values", r.span)?;
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
                    let (ty, view) = self.ty(&annotation.type_annotation)?;
                    self.plain_enum(view, "enum types in function values", p.span)?;
                    params.push(ty);
                }
                let (ret, view) = self.ty(&f.return_type.type_annotation)?;
                self.plain_enum(view, "enum types in function values", f.span)?;
                (self.function_type(params.into_iter(), ret, f.span)?, None)
            }
            js::TSType::TSTypeReference(r) if r.type_arguments.is_none() => {
                if let js::TSTypeName::QualifiedName(name) = &r.type_name {
                    if let Some(member) = self.enum_member_type(name, r.span) {
                        return member;
                    }
                    return self.namespace_type(name);
                }
                let js::TSTypeName::IdentifierReference(name) = &r.type_name else {
                    return self.unsupported("qualified type names", r.span);
                };
                // `const` is reserved, so it names a type only in `x as const`.
                if name.name == "const" {
                    return self.unsupported("const assertions", r.span);
                }
                let name = self.name(name.name.as_str());
                if let Some((_, ty)) = self.active_types.iter().find(|(n, _)| *n == name) {
                    return Ok((ty.clone(), None));
                }
                if let Some(enumeration) = self.enum_type(name, r.span) {
                    return enumeration;
                }
                if let Some(found) = self.aliases.get(&name) {
                    return Ok(found.clone());
                }
                if self.declared_types.contains(&name) {
                    return self.unsupported(UNRESOLVED_REFERENCE, r.span);
                }
                let source = name.resolve_global().unwrap_or_default();
                if library_type(&source) {
                    return self.unsupported(&format!("library type `{source}`"), r.span);
                }
                return self.error(ErrorKind::Type, format!("unknown type `{source}`"), r.span);
            }
            js::TSType::TSParenthesizedType(p) => self.ty(&p.type_annotation)?,
            js::TSType::TSTypeLiteral(literal) => match self.type_literals.get(&literal.span.start)
            {
                Some(found) => found.clone(),
                None => return self.unsupported("this type representation", ty.span()),
            },
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
        let source_name = modules::function_name(f);
        self.signature_as(f, self.name(source_name), source_name, FunctionRole::Plain)
    }

    /// `component` names the native symbol; `key` identifies the function here.
    fn signature_as(
        &mut self,
        f: &'program js::Function<'ast>,
        key: InternedString,
        component: &str,
        role: FunctionRole,
    ) -> Result<()> {
        if f.r#async || f.generator || f.this_param.is_some() || f.body.is_none() {
            return self.unsupported("async, generator, rest or ambient functions", f.span);
        }
        if self.signatures.contains_key(&key) {
            return self.unsupported("function overloads", f.span);
        }
        self.active_types.clear();
        self.active_type_rules = None;
        let mut type_params = vec![];
        if let Some(parameters) = &f.type_parameters {
            for p in &parameters.params {
                if p.r#const {
                    return self.unsupported("const type parameters", p.span);
                }
                let name = self.name(p.name.name.as_str());
                if self.active_types.iter().any(|(n, _)| *n == name) {
                    return self.error(ErrorKind::Type, "duplicate type parameter", p.span);
                }
                let ty = Type::TypeVar(zyntax_typed_ast::TypeVar::parameter(name));
                self.active_types.push((name, ty.clone()));
                type_params.push((name, ty));
            }
            self.active_type_rules = self.type_parameter_rules(parameters)?;
        }
        let mut params = vec![];
        if let Some(view) = role.takes_this() {
            params.push(self.this_parameter(view));
        }
        let mut names = FxHashSet::default();
        let mut required = 0;
        let mut optional = false;
        for (index, p) in f.params.items.iter().enumerate() {
            let js::BindingPattern::BindingIdentifier(name) = &p.pattern else {
                return self.unsupported("destructuring parameters", p.span);
            };
            let omittable = p.optional || p.initializer.is_some();
            let (ty, view) = if omittable {
                optional |= p.optional;
                self.omittable_parameter(p)?
            } else {
                if optional {
                    return self.error(
                        ErrorKind::Type,
                        "a required parameter cannot follow an optional parameter",
                        p.span,
                    );
                }
                required = index + 1;
                let Some(annotation) = &p.type_annotation else {
                    return self.error(
                        ErrorKind::Type,
                        "parameter implicitly has type any",
                        p.span,
                    );
                };
                self.ty(&annotation.type_annotation)?
            };
            if !names.insert(self.name(name.name.as_str())) {
                return self.error(ErrorKind::Type, "duplicate parameter", p.span);
            }
            params.push(CheckerParameter {
                name: self.name(name.name.as_str()),
                ty,
                view,
                ownership: ParamOwnership::Borrowed,
            });
        }
        let rest = match &f.params.rest {
            Some(rest) => {
                let (name, ty) = self.rest_parameter(rest)?;
                if !names.insert(name) {
                    return self.error(ErrorKind::Type, "duplicate parameter", rest.span);
                }
                params.push(CheckerParameter {
                    name,
                    ty,
                    view: None,
                    ownership: ParamOwnership::Borrowed,
                });
                true
            }
            None => false,
        };
        let arity =
            (rest || required < f.params.items.len()).then_some(params::Arity { required, rest });
        let (ret, ret_view) = match (&f.return_type, role) {
            (_, FunctionRole::Constructor(view)) => {
                (Some(self.classes[&view].ty.clone()), Some(view))
            }
            (_, FunctionRole::Initializer(_)) => (Some(unit()), None),
            (Some(ret), _) => {
                let (ty, view) = self.ty(&ret.type_annotation)?;
                (Some(ty), view)
            }
            (None, _) => (None, None),
        };
        let name = key;
        self.functions.insert(name, f);
        self.name_buffer.truncate(self.prefix_len);
        self.name_buffer.push_str("fn$");
        crate::modules::append_component(&mut self.name_buffer, component);
        let native = intern(&self.name_buffer);
        if let Some(arity) = arity {
            self.arities.insert(native, arity);
        }
        self.signatures.insert(
            name,
            Arc::new(Signature {
                native,
                type_params,
                type_rules: self.active_type_rules.clone(),
                params,
                ret,
                ret_view,
                array_result: None,
            }),
        );
        self.active_types.clear();
        self.active_type_rules = None;
        Ok(())
    }

    fn function(&mut self, f: &js::Function<'ast>) -> Result<()> {
        let name = self.name(modules::function_name(f));
        self.function_as(f, name, FunctionRole::Plain)
    }

    fn function_as(
        &mut self,
        f: &js::Function<'ast>,
        name: InternedString,
        role: FunctionRole,
    ) -> Result<()> {
        if self.lowered.contains(&name) {
            return Ok(());
        }
        let result = self.function_body(f, name, role);
        if result.is_ok() {
            self.lowered.insert(name);
        }
        result
    }

    fn function_body(
        &mut self,
        f: &js::Function<'ast>,
        name: InternedString,
        role: FunctionRole,
    ) -> Result<()> {
        let inferred = self.signatures[&name].ret.is_none();
        let recursive = inferred && self.self_recursive.contains(&name);
        if recursive {
            let sig = Arc::get_mut(self.signatures.get_mut(&name).unwrap()).unwrap();
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
        self.active_type_rules = sig.type_rules.clone();
        self.scopes.push(FxHashMap::default());
        self.flow.forget_global_members(&self.globals);
        let outer_vars = std::mem::replace(
            &mut self.vars,
            FunctionVars {
                scope: self.scopes.len() - 1,
                hoisted: vec![],
            },
        );
        let mut params = vec![];
        let mut prologue = vec![];
        let has_this = role.takes_this().is_some();
        if let Some(view) = role.takes_this() {
            params.push(self.bind_this(view, f.span));
        }
        let symbols = f
            .params
            .items
            .iter()
            .map(|p| match &p.pattern {
                js::BindingPattern::BindingIdentifier(id) => (id.name.as_str(), id.symbol_id.get()),
                _ => unreachable!(),
            })
            .chain(f.params.rest.iter().map(|rest| match &rest.rest.argument {
                js::BindingPattern::BindingIdentifier(id) => (id.name.as_str(), id.symbol_id.get()),
                _ => unreachable!(),
            }));
        let defaults = f
            .params
            .items
            .iter()
            .map(|p| p.initializer.as_deref())
            .chain(f.params.rest.iter().map(|_| None));
        for ((parameter_type, (source, symbol)), default) in sig
            .params
            .iter()
            .skip(usize::from(has_this))
            .zip(symbols)
            .zip(defaults)
        {
            let CheckerParameter { name, ty, view, .. } = parameter_type;
            let unique = self.fresh(source);
            let mut param =
                TypedParameter::regular(unique, ty.clone(), Mutability::Mutable, span(f.span));
            param.ownership = ParamOwnership::Borrowed;
            params.push(param);
            // A default sees the parameters before it, and replaces `undefined` on entry.
            let (bound, bound_type) = match default {
                Some(default) => {
                    let Type::Optional(inner) = ty else {
                        unreachable!("defaulted parameters are optional slots")
                    };
                    let at = default.span();
                    let value = self.expr_expected(default, inner)?;
                    self.expect(self.source_type(default, &value.ty), inner, at)?;
                    let value = self.adapt_value(value, inner, at)?;
                    let resolved = self.fresh(source);
                    let chosen = self.optional_match(
                        var(unique, ty.clone(), at),
                        |_, present| Ok(present),
                        Some(value),
                        inner.as_ref().clone(),
                        at,
                    )?;
                    prologue.push(stmt(
                        TypedStatement::Let(TypedLet {
                            name: resolved,
                            ty: inner.as_ref().clone(),
                            mutability: Mutability::Mutable,
                            initializer: Some(Box::new(chosen)),
                            span: span(at),
                        }),
                        at,
                    ));
                    (resolved, inner.as_ref().clone())
                }
                None => (unique, ty.clone()),
            };
            self.scopes.last_mut().unwrap().insert(
                *name,
                Binding {
                    name: bound,
                    symbol,
                    ty: bound_type,
                    view: *view,
                    mutable: true,
                    initialized: true,
                },
            );
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
        let mut statements = prologue;
        let outer_class = std::mem::replace(&mut self.current_class, role.class());
        let outer_body = std::mem::replace(&mut self.class_body, role.body_class());
        let outer_constructor = std::mem::replace(
            &mut self.in_constructor,
            matches!(
                role,
                FunctionRole::Constructor(_) | FunctionRole::Initializer(_)
            ),
        );
        let constructed = match role {
            FunctionRole::Constructor(view) => {
                Some(self.constructor_prologue(view, Some(f), &mut statements)?)
            }
            FunctionRole::Initializer(view) if self.classes[&view].base.is_none() => {
                let this = self.binding("this", f.span)?.name;
                let initialized = self.initialize_fields(view, Some(f), this)?;
                statements.extend(initialized);
                None
            }
            _ => None,
        };
        let body = &f.body.as_ref().unwrap().statements;
        self.reserve(body)?;
        match role {
            FunctionRole::Constructor(view) | FunctionRole::Initializer(view)
                if self.classes[&view].base.is_some() =>
            {
                self.derived_constructor_body(view, f, body, &mut statements)?;
            }
            _ => {
                for s in body {
                    statements.extend(self.statement(s)?);
                }
            }
        }
        if let FunctionRole::Constructor(view) = role {
            self.check_field_initialization(view, f)?;
        }
        self.current_class = outer_class;
        self.class_body = outer_body;
        self.in_constructor = outer_constructor;
        statements.splice(0..0, std::mem::take(&mut self.vars.hoisted));
        self.vars = outer_vars;
        let context = self.returns.take().unwrap();
        let array_result = context.array_result;
        let (mut ret, ret_view) = match context.expected {
            Some(result) => result,
            None if !context.value_returned => (unit(), None),
            None => context.inferred.unwrap_or((unit(), None)),
        };
        let complete = constructed.is_some() || self.returns(&f.body.as_ref().unwrap().statements);
        if let Some(this) = constructed {
            statements.push(stmt(
                TypedStatement::Return(Some(Box::new(var(this, ret.clone(), f.span)))),
                f.span,
            ));
        }
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
            .enumerate()
            .map(|(index, (name, _))| TypedTypeParam {
                name: *name,
                bounds: sig
                    .type_rules()
                    .get(index)
                    .and_then(|r| r.constraint.as_ref())
                    .map(|c| vec![TypedTypeBound::Subtype(c.ty.clone())])
                    .unwrap_or_default(),
                default: sig
                    .type_rules()
                    .get(index)
                    .and_then(|r| r.default.as_ref())
                    .map(|d| d.ty.clone()),
                span: span(f.span),
                ..Default::default()
            })
            .collect();
        drop(sig);
        let sig = Arc::get_mut(self.signatures.get_mut(&name).unwrap())
            .expect("signature handles do not escape body lowering");
        sig.ret = Some(ret.clone());
        sig.ret_view = ret_view;
        sig.array_result = array_result;
        self.pop_scope();
        self.flow.forget_global_members(&self.globals);
        self.active_types.clear();
        self.active_type_rules = None;
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
        if *first_view != view
            && (self.enum_of(*first_view).is_some() || self.enum_of(view).is_some())
        {
            return self.unsupported("inferred returns mixing enum types", at);
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
            view.and_then(|name| self.views.get(&name))
                .map(|view| &view.ty)
        };
        if logical(*first_view).unwrap_or(first) != logical(view).unwrap_or(ty) {
            return self.unsupported("inferred returns with different property views", at);
        }
        let readonly = |view: Option<InternedString>| {
            view.and_then(|name| self.views.get(&name))
                .into_iter()
                .flat_map(|view| view.fields.iter().filter(|f| f.readonly).map(|f| f.name))
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

    fn returns(&self, statements: &[js::Statement<'ast>]) -> bool {
        statements.iter().any(|s| match s {
            js::Statement::ReturnStatement(_) => true,
            js::Statement::BlockStatement(b) => self.returns(&b.body),
            js::Statement::IfStatement(i) => i.alternate.as_ref().is_some_and(|other| {
                self.returns(std::slice::from_ref(&i.consequent))
                    && self.returns(std::slice::from_ref(other))
            }),
            js::Statement::SwitchStatement(sw) => self.switch_returns(sw),
            js::Statement::DoWhileStatement(d) => {
                self.returns(std::slice::from_ref(&d.body))
                    || (is_true(&d.test) && !Self::may_break(std::slice::from_ref(&d.body)))
            }
            js::Statement::WhileStatement(w) => {
                is_true(&w.test) && !Self::may_break(std::slice::from_ref(&w.body))
            }
            js::Statement::ForStatement(f) => {
                f.test.as_ref().is_none_or(is_true)
                    && !Self::may_break(std::slice::from_ref(&f.body))
            }
            _ => false,
        })
    }

    /// Whether a `break` can leave the loop whose body this is.
    fn may_break(body: &[js::Statement<'ast>]) -> bool {
        struct Breaks {
            nested: usize,
            found: bool,
        }
        impl<'a> Visit<'a> for Breaks {
            fn visit_break_statement(&mut self, it: &js::BreakStatement<'a>) {
                self.found |= it.label.is_some() || self.nested == 0;
            }
            fn visit_while_statement(&mut self, it: &js::WhileStatement<'a>) {
                self.nested += 1;
                walk::walk_while_statement(self, it);
                self.nested -= 1;
            }
            fn visit_for_statement(&mut self, it: &js::ForStatement<'a>) {
                self.nested += 1;
                walk::walk_for_statement(self, it);
                self.nested -= 1;
            }
            fn visit_do_while_statement(&mut self, it: &js::DoWhileStatement<'a>) {
                self.nested += 1;
                walk::walk_do_while_statement(self, it);
                self.nested -= 1;
            }
            fn visit_for_in_statement(&mut self, it: &js::ForInStatement<'a>) {
                self.nested += 1;
                walk::walk_for_in_statement(self, it);
                self.nested -= 1;
            }
            fn visit_for_of_statement(&mut self, it: &js::ForOfStatement<'a>) {
                self.nested += 1;
                walk::walk_for_of_statement(self, it);
                self.nested -= 1;
            }
            fn visit_switch_statement(&mut self, it: &js::SwitchStatement<'a>) {
                self.nested += 1;
                walk::walk_switch_statement(self, it);
                self.nested -= 1;
            }
            fn visit_function(&mut self, _: &js::Function<'a>, _: oxc_semantic::ScopeFlags) {}
            fn visit_arrow_function_expression(&mut self, _: &js::ArrowFunctionExpression<'a>) {}
        }
        let mut breaks = Breaks {
            nested: 0,
            found: false,
        };
        for statement in body {
            breaks.visit_statement(statement);
        }
        breaks.found
    }

    fn externs(&mut self) {
        let math =
            math::primitive_externs().map(|(name, arity)| (name, vec![number(); arity], number()));
        let primitives = [
            ("$TS$print_number", vec![number()], unit()),
            ("$TS$pow", vec![number(), number()], number()),
            ("$TS$index_trap", vec![number(), number()], unit()),
            ("$TS$array_length_trap", vec![number()], unit()),
            ("$TS$binding_check", vec![boolean(), string()], unit()),
            ("$TS$assigned_check", vec![boolean(), string()], unit()),
            ("$TS$enum_check", vec![boolean(), string()], unit()),
            ("$TS$switch_trap", vec![], unit()),
            ("$TS$date_now", vec![], number()),
            ("$TS$performance_now", vec![], number()),
            (
                "$TS$to_int32_slow",
                vec![number()],
                Type::Primitive(PrimitiveType::I64),
            ),
            ("$TS$space", vec![], unit()),
            ("$TS$newline", vec![], unit()),
            ("$IO$print", vec![string()], unit()),
            ("$IO$print_bool", vec![boolean()], unit()),
            ("$TS$number_to_string", vec![number()], string()),
            ("$IO$format_bool", vec![boolean()], string()),
            ("$String$concat", vec![string(), string()], string()),
            ("$String$equals", vec![string(), string()], boolean()),
        ]
        .map(|(name, params, ret)| (name.to_string(), params, ret));
        for (name, params, ret) in primitives.into_iter().chain(math) {
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
                    name: intern(&name),
                    params,
                    return_type: ret,
                    is_external: true,
                    calling_convention: CallingConvention::Cdecl,
                    link_name: Some(intern(&name)),
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
        // `var` bindings are confined to their block and declared where they appear.
        if declaration.kind.as_str() == "var" {
            return Ok(());
        }
        for d in &declaration.declarations {
            if let js::BindingPattern::BindingIdentifier(id) = &d.id {
                if id
                    .symbol_id
                    .get()
                    .is_some_and(|symbol| self.module_symbols.contains_key(&symbol))
                {
                    continue;
                }
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
            .ok_or_else(|| {
                // Substituted constants have no binding; only a write reaches this lookup.
                if self.constants.contains_key(&key) {
                    return Error::new(ErrorKind::Type, "cannot assign to const", at);
                }
                if library_value(name) {
                    return Error::new(
                        ErrorKind::Unsupported,
                        format!("library value `{name}` is not implemented yet"),
                        at,
                    );
                }
                Error::new(ErrorKind::Type, format!("cannot find name `{name}`"), at)
            })?;
        // Scope 0 holds module globals, which every body reaches.
        if scope < self.scope_floor && scope != 0 {
            return self.unsupported("lexical captures in arrow functions", at);
        }
        // A body that runs later checks a global's readiness when it runs.
        if !b.initialized && (scope != 0 || self.top_level) {
            return self.error(
                ErrorKind::Type,
                format!("`{name}` is used before initialization"),
                at,
            );
        }
        Ok(b)
    }

    fn expect(&self, actual: &Type, expected: &Type, at: oxc_span::Span) -> Result<()> {
        if actual != expected && matches!(actual, Type::TypeVar(_)) {
            let constrained = self.constraint_type(actual);
            if constrained != actual {
                return self.expect(constrained, expected, at);
            }
        }
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
            // A record lacks the class field every instance of a hierarchy carries.
            if matches!(actual, Type::Named { .. })
                && self
                    .classes
                    .values()
                    .any(|class| class.tag.is_some() && class.ty == *expected)
            {
                return self.unsupported("structural values of classes with subclasses", at);
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
            .and_then(|view| self.views.get(&view))
            .map_or(physical, |view| &view.ty)
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
                        Type::Primitive(PrimitiveType::String) => {
                            HirType::Ptr(std::sync::Arc::new(HirType::I8))
                        }
                        _ => unreachable!("TS record fields are numbers, booleans or strings"),
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
                    self.narrow_enum(b, equality);
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
                if let Some(view) = self.class_by_layout(ty) {
                    let component = &self.classes[&view].component;
                    return component.trim_end_matches("\0new").to_owned();
                }
                let mut description = String::from("{ ");
                if let Some(definition) = self.registry.get_type_by_id(*id) {
                    let class_tag = self.name(classes::CLASS_TAG);
                    let fields = definition.fields.iter().filter(|f| f.name != class_tag);
                    for (index, field) in fields.enumerate() {
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
                if let Some(sig) = self.direct_signature(&c.callee).ok().flatten() {
                    if let Some(index) = sig
                        .type_params
                        .iter()
                        .position(|(_, ty)| Some(ty) == sig.ret.as_ref())
                    {
                        return self.type_argument_view(sig, c, index);
                    }
                    sig.ret_view
                } else {
                    self.method_return_view(&c.callee)
                }
            }
            js::Expression::NewExpression(n) => self.class_of_new(n),
            js::Expression::ComputedMemberExpression(m) if !m.optional => self
                .enum_view_of_member(&m.object)
                .or_else(|| self.element_view(&m.object)),
            js::Expression::ThisExpression(t) => {
                self.binding("this", t.span).ok().and_then(|b| b.view)
            }
            js::Expression::ParenthesizedExpression(p) => self.view_of(&p.expression),
            js::Expression::TSNonNullExpression(n) => self.view_of(&n.expression),
            js::Expression::ConditionalExpression(c) => {
                let (a, b) = (self.view_of(&c.consequent), self.view_of(&c.alternate));
                if a != b && (self.enum_of(a).is_some() || self.enum_of(b).is_some()) {
                    None
                } else {
                    a
                }
            }
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                self.enum_view_of_member(&m.object)
            }
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
        self.check_class_target(from, to, at)?;
        if self.options.soundness == Soundness::Checked {
            if let Some(from) = from.and_then(|n| self.views.get(&n)) {
                let to = to.and_then(|n| self.views.get(&n));
                for field in &from.fields {
                    let name = field.name.resolve_global().unwrap();
                    if field.readonly
                        && !to
                            .and_then(|view| view.field(field.name))
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
                    if let Some(source) = &r.argument {
                        self.check_enum(source, expected_view, at)?;
                    }
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
                let then_returns = self.returns(std::slice::from_ref(&i.consequent));
                let else_returns = i
                    .alternate
                    .as_ref()
                    .is_some_and(|s| self.returns(std::slice::from_ref(s)));
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
                let assigned = self.flow.assigned();
                self.clear_written_flow(w.span);
                let condition = self.condition(&w.test)?;
                self.narrow(&w.test, true);
                self.open_jump(statements::JumpKind::Loop);
                self.loop_depth += 1;
                let mut body = self.block(&w.body)?;
                self.loop_depth -= 1;
                let mut after = vec![];
                self.close_jump(Some(&mut body), &mut after, at);
                self.clear_written_flow(w.span);
                self.flow.restore_assigned(assigned);
                if !Self::may_break(std::slice::from_ref(&w.body)) {
                    self.narrow(&w.test, false);
                }
                let mut out = vec![stmt(
                    TypedStatement::While(TypedWhile {
                        condition: Box::new(condition),
                        body,
                        span: span(at),
                    }),
                    at,
                )];
                out.extend(after);
                out
            }
            js::Statement::ForStatement(f) => {
                let assigned = self.flow.assigned();
                self.open_jump(statements::JumpKind::Loop);
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
                let mut body = self.block(&f.body)?;
                self.loop_depth -= 1;
                let mut after = vec![];
                self.close_jump(Some(&mut body), &mut after, at);
                self.clear_written_flow(repeated);
                self.pop_scope();
                self.flow.restore_assigned(assigned);
                if let Some(test) = &f.test
                    && !Self::may_break(std::slice::from_ref(&f.body))
                {
                    self.narrow(test, false);
                }
                let mut out = vec![stmt(
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
                )];
                out.extend(after);
                out
            }
            js::Statement::ForOfStatement(f) => self.for_of(f)?,
            js::Statement::ClassDeclaration(c) => {
                return self.unsupported("class declarations in nested scopes", c.span);
            }
            js::Statement::BreakStatement(b) => self.break_statement(b.label.as_ref(), at)?,
            js::Statement::ContinueStatement(c) => self.continue_statement(c.label.as_ref(), at)?,
            js::Statement::DoWhileStatement(d) => self.do_while(d)?,
            js::Statement::SwitchStatement(sw) => self.switch(sw)?,
            js::Statement::LabeledStatement(l) => self.labeled(l)?,
            _ => return self.unsupported("this statement", at),
        })
    }

    fn variables(&mut self, declaration: &js::VariableDeclaration<'ast>) -> Result<Vec<Stmt>> {
        let kind = declaration.kind.as_str();
        if !matches!(kind, "let" | "const" | "var") {
            return self.unsupported("using declarations", declaration.span);
        }
        if declaration.declare {
            return self.ambient_variables(declaration).map(|()| vec![]);
        }
        let mut out = vec![];
        for d in &declaration.declarations {
            let js::BindingPattern::BindingIdentifier(id) = &d.id else {
                return self.unsupported("destructuring declarations", d.span);
            };
            let name = self.name(id.name.as_str());
            if let Some(global) = id
                .symbol_id
                .get()
                .and_then(|symbol| self.module_symbols.get(&symbol))
                .copied()
            {
                if let Some(annotation) = &d.type_annotation
                    && self.ty(&annotation.type_annotation)?.0 != self.scopes[0][&name].ty
                {
                    return self.error(
                        ErrorKind::Type,
                        "subsequent variable declarations must have the same type",
                        d.span,
                    );
                }
                out.extend(self.declare_in_module(global, name, d.init.as_ref(), d.span)?);
                continue;
            }
            let redeclared = self
                .scopes
                .last()
                .unwrap()
                .get(&name)
                .filter(|b| b.initialized)
                .map(|b| (b.symbol, b.ty.clone()));
            if let Some((symbol, _)) = &redeclared
                && !(kind == "var" && *symbol == id.symbol_id.get())
            {
                return self.error(ErrorKind::Type, "duplicate variable", d.span);
            }
            let Some(init) = &d.init else {
                let hoisted = id
                    .symbol_id
                    .get()
                    .is_some_and(|symbol| self.hoist.contains(&symbol));
                out.extend(self.uninitialized(id, d, kind, hoisted, redeclared.map(|r| r.1))?);
                continue;
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
            if let Some((ty, view)) = &declared {
                self.expect(self.source_type(init, &value.ty), ty, d.span)?;
                self.fresh_properties(init, ty)?;
                self.check_enum(init, *view, d.span)?;
            }
            let (mut ty, view) = declared.unwrap_or_else(|| (value.ty.clone(), self.view_of(init)));
            self.check_view(self.view_of(init), view, d.span)?;
            if matches!(ty, Type::Named { .. }) {
                ty = value.ty.clone();
            }
            let symbol = id.symbol_id.get();
            if kind == "var" && symbol.is_some_and(|symbol| self.hoist.contains(&symbol)) {
                let value = self.adapt_value(value, &ty, d.span)?;
                out.push(self.hoisted_var(id.name.as_str(), symbol, ty, view, value, d.span)?);
                continue;
            }
            if let Some((_, earlier)) = &redeclared
                && *earlier != ty
            {
                return self.error(
                    ErrorKind::Type,
                    "subsequent variable declarations must have the same type",
                    d.span,
                );
            }
            let unique = self.fresh(id.name.as_str());
            if matches!(ty, Type::Optional(_) | Type::Union(_)) {
                if let Some(flow) = ScalarFlow::of(&value.ty) {
                    self.flow.insert(unique, flow);
                }
            }
            self.value_facts(unique, &ty, init);
            self.enum_facts(unique, view, init);
            if kind == "const" && d.type_annotation.is_none() {
                self.declare_literal(unique, init);
            }
            let value = self.adapt_value(value, &ty, d.span)?;
            self.scopes.last_mut().unwrap().insert(
                name,
                Binding {
                    name: unique,
                    symbol: id.symbol_id.get(),
                    ty: ty.clone(),
                    view,
                    mutable: kind != "const",
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

    /// String conversion for `+` and template literals; other types stay unsupported.
    fn text_of(&mut self, value: Expr, at: oxc_span::Span) -> Result<Expr> {
        let value = self.constrained_scalar(value, at);
        let ty = value.ty.clone();
        if ty == string() {
            Ok(value)
        } else if ty == number() {
            Ok(self.call("$TS$number_to_string", vec![value], string(), at))
        } else if ty == boolean() {
            Ok(self.call("$IO$format_bool", vec![value], string(), at))
        } else {
            self.unsupported("string conversion of this type", at)
        }
    }

    fn template(&mut self, template: &js::TemplateLiteral<'ast>) -> Result<Expr> {
        let mut parts = vec![];
        for (index, quasi) in template.quasis.iter().enumerate() {
            let Some(cooked) = quasi
                .value
                .cooked
                .as_ref()
                .filter(|_| !quasi.lone_surrogates)
            else {
                return self.unsupported("template text with unpaired surrogates", quasi.span);
            };
            if !cooked.is_empty() {
                parts.push(node(
                    TypedExpression::Literal(TypedLiteral::String(self.name(cooked.as_str()))),
                    string(),
                    quasi.span,
                ));
            }
            if let Some(expression) = template.expressions.get(index) {
                let value = self.expr(expression)?;
                parts.push(self.text_of(value, expression.span())?);
            }
        }
        let at = template.span;
        let mut parts = parts.into_iter();
        let Some(mut result) = parts.next() else {
            return Ok(node(
                TypedExpression::Literal(TypedLiteral::String(self.name(""))),
                string(),
                at,
            ));
        };
        for part in parts {
            result = self.call("$String$concat", vec![result, part], string(), at);
        }
        Ok(result)
    }

    /// A `var` that outlives its block: declared uninitialized at the top of the
    /// function, assigned where the declaration appears.
    fn hoisted_var(
        &mut self,
        source: &'ast str,
        symbol: Option<oxc_semantic::SymbolId>,
        ty: Type,
        view: Option<InternedString>,
        value: Expr,
        at: oxc_span::Span,
    ) -> Result<Stmt> {
        if ty != number() && ty != boolean() {
            return self.unsupported("var of this type declared in a nested block", at);
        }
        let name = self.name(source);
        let scope = self.vars.scope;
        let existing = self.scopes[scope]
            .get(&name)
            .filter(|b| b.symbol == symbol)
            .map(|b| (b.name, b.ty.clone()));
        let unique = match existing {
            Some((_, earlier)) if earlier != ty => {
                return self.error(
                    ErrorKind::Type,
                    "subsequent variable declarations must have the same type",
                    at,
                );
            }
            Some((unique, _)) => unique,
            None => {
                let unique = self.fresh(source);
                self.scopes[scope].insert(
                    name,
                    Binding {
                        name: unique,
                        symbol,
                        ty: ty.clone(),
                        view,
                        mutable: true,
                        initialized: true,
                    },
                );
                self.vars.hoisted.push(stmt(
                    TypedStatement::Let(TypedLet {
                        name: unique,
                        ty: ty.clone(),
                        mutability: Mutability::Mutable,
                        initializer: None,
                        span: span(at),
                    }),
                    at,
                ));
                self.unassigned.insert(unique, Unassigned::Hoisted);
                unique
            }
        };
        let target = var(unique, ty.clone(), at);
        self.record_write(&target, &ty);
        Ok(expression(binary(
            BinaryOp::Assign,
            target,
            value,
            unit(),
            at,
        )))
    }

    /// A declaration with a type and no value: an uninitialized mutable binding
    /// whose reads must follow an assignment on every path.
    fn uninitialized(
        &mut self,
        id: &js::BindingIdentifier<'ast>,
        d: &js::VariableDeclarator<'ast>,
        kind: &str,
        hoisted: bool,
        earlier: Option<Type>,
    ) -> Result<Option<Stmt>> {
        let Some(annotation) = &d.type_annotation else {
            return self.unsupported("uninitialized bindings without a type annotation", d.span);
        };
        let (ty, view) = self.ty(&annotation.type_annotation)?;
        if matches!(
            ty,
            Type::Union(_) | Type::Function { .. } | Type::TypeVar(_)
        ) {
            return self.unsupported("uninitialized bindings of this type", d.span);
        }
        if let Some(earlier) = earlier {
            if earlier != ty {
                return self.error(
                    ErrorKind::Type,
                    "subsequent variable declarations must have the same type",
                    d.span,
                );
            }
            return Ok(None);
        }
        if hoisted {
            return self.unsupported("uninitialized var used outside its block", d.span);
        }
        let unique = self.fresh(id.name.as_str());
        let name = self.name(id.name.as_str());
        self.scopes.last_mut().unwrap().insert(
            name,
            Binding {
                name: unique,
                symbol: id.symbol_id.get(),
                ty: ty.clone(),
                view,
                mutable: kind != "const",
                initialized: true,
            },
        );
        // A type admitting `undefined` starts as `undefined`; reading it unassigned is valid.
        if matches!(ty, Type::Optional(_)) {
            let none = var(self.name("None"), undefined(), d.span);
            if let Some(flow) = ScalarFlow::of(&none.ty) {
                self.flow.insert(unique, flow);
            }
            let value = self.adapt_value(none, &ty, d.span)?;
            return Ok(Some(stmt(
                TypedStatement::Let(TypedLet {
                    name: unique,
                    ty,
                    mutability: Mutability::Mutable,
                    initializer: Some(Box::new(value)),
                    span: span(d.span),
                }),
                d.span,
            )));
        }
        self.unassigned.insert(unique, Unassigned::Declared);
        Ok(Some(stmt(
            TypedStatement::Let(TypedLet {
                name: unique,
                ty,
                mutability: Mutability::Mutable,
                initializer: None,
                span: span(d.span),
            }),
            d.span,
        )))
    }

    /// An ambient binding's value comes from outside the program, which native code cannot read.
    fn require_native(&self, name: InternedString, at: oxc_span::Span) -> Result<()> {
        if self.ambient.contains(&name) {
            return self.unsupported("values of ambient variables", at);
        }
        Ok(())
    }

    /// `declare var x: T`: a typed binding with no native value.
    fn ambient_variables(&mut self, declaration: &js::VariableDeclaration<'ast>) -> Result<()> {
        for d in &declaration.declarations {
            let js::BindingPattern::BindingIdentifier(id) = &d.id else {
                return self.unsupported("destructuring declarations", d.span);
            };
            let name = self.name(id.name.as_str());
            if let Some(global) = id
                .symbol_id
                .get()
                .and_then(|symbol| self.module_symbols.get(&symbol))
            {
                debug_assert!(self.ambient.contains(global));
                self.scopes[0].get_mut(&name).unwrap().initialized = true;
                continue;
            }
            let Some(annotation) = &d.type_annotation else {
                return self.error(ErrorKind::Type, "variable implicitly has type any", d.span);
            };
            let (ty, view) = self.ty(&annotation.type_annotation)?;
            let unique = self.fresh(id.name.as_str());
            self.ambient.insert(unique);
            self.scopes.last_mut().unwrap().insert(
                name,
                Binding {
                    name: unique,
                    symbol: id.symbol_id.get(),
                    ty,
                    view,
                    mutable: declaration.kind.as_str() != "const",
                    initialized: true,
                },
            );
        }
        Ok(())
    }

    fn require_assigned(&self, name: InternedString, at: oxc_span::Span) -> Result<()> {
        // A body that runs later checks a global's first assignment when it runs.
        if !self.top_level && self.globals.contains_key(&name) {
            return Ok(());
        }
        match self.unassigned.get(&name) {
            Some(Unassigned::Declared) if !self.flow.is_assigned(name) => self.error(
                ErrorKind::Type,
                "variable is used before being assigned",
                at,
            ),
            Some(Unassigned::Hoisted) if !self.flow.is_assigned(name) => {
                self.unsupported("a var read that may precede its assignment", at)
            }
            _ => Ok(()),
        }
    }

    /// `&&` and `||` select one of their operands; only boolean operands are lowered so far.
    fn logical_operand(&self, ty: &Type, at: oxc_span::Span) -> Result<()> {
        if *ty == unit() {
            return self.error(
                ErrorKind::Type,
                "an expression of type void cannot be tested for truthiness",
                at,
            );
        }
        if *ty != boolean() {
            return self.unsupported(
                "value-selecting logical operations on non-boolean values",
                at,
            );
        }
        Ok(())
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
            js::Expression::CallExpression(c) if !c.optional => {
                let filled = if let js::Expression::StaticMemberExpression(m) = &c.callee {
                    self.array_fill(c, m, Some(expected))?
                } else {
                    None
                };
                match filled {
                    Some(filled) => Ok(filled),
                    None => self.expr(e),
                }
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
        let outer_jumps = std::mem::take(&mut self.jumps);
        let outer_label = self.pending_label.take();
        let outer_top = std::mem::replace(&mut self.top_level, false);
        let outer_flow = self.flow.clone();
        self.flow.forget_global_members(&self.globals);
        let outer_floor = std::mem::replace(&mut self.scope_floor, self.scopes.len());
        self.scopes.push(FxHashMap::default());
        let outer_vars = std::mem::replace(
            &mut self.vars,
            FunctionVars {
                scope: self.scopes.len() - 1,
                hoisted: vec![],
            },
        );
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
                    Some(annotation) => {
                        let (ty, view) = self.ty(&annotation.type_annotation)?;
                        self.plain_enum(view, "enum types in function values", p.span)?;
                        (ty, view)
                    }
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
                statements.splice(0..0, std::mem::take(&mut self.vars.hoisted));
                self.returns(&body.statements)
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
                    self.check_enum(e, view, at)?;
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
        self.vars = outer_vars;
        self.scope_floor = outer_floor;
        self.top_level = outer_top;
        self.flow = outer_flow;
        self.jumps = outer_jumps;
        self.pending_label = outer_label;
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
            // A template of literal parts is a static string, as TypeScript types it.
            js::Expression::TemplateLiteral(template) => match constants::literal_text(e) {
                Some(text) => node(
                    TypedExpression::Literal(TypedLiteral::String(intern(&text))),
                    string(),
                    at,
                ),
                None => self.template(template)?,
            },
            js::Expression::Identifier(i) => {
                if self.namespace_references.contains_key(&i.node_id.get()) {
                    return self.unsupported("namespace objects as values", at);
                }
                self.require_value(i.name.as_str(), at)?;
                if let Some(constant) = self.captured_constant(i.name.as_str(), at) {
                    return Ok(constant);
                }
                if self
                    .scopes
                    .iter()
                    .any(|s| s.contains_key(&self.name(i.name.as_str())))
                {
                    let unique = self.binding(i.name.as_str(), at)?.name;
                    self.require_native(unique, at)?;
                    self.require_assigned(unique, at)?;
                    let b = self.binding(i.name.as_str(), at)?;
                    let value = var(b.name, b.ty.clone(), at);
                    let flow = self.flow.get(&b.name).copied();
                    let mut value = self.flow_read(value, flow, at)?;
                    let mut checks = self.global_guard(unique, true, at);
                    if !checks.is_empty() {
                        let ty = value.ty.clone();
                        checks.push(expression(value));
                        value = node(
                            TypedExpression::Block(TypedBlock {
                                statements: checks.into_vec(),
                                span: span(at),
                            }),
                            ty,
                            at,
                        );
                    }
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
                    if let Some(object) = self.enum_object(i) {
                        return object;
                    }
                    match i.name.as_str() {
                        "undefined" => var(self.name("None"), undefined(), at),
                        "console" if !self.imported_exports.contains_key(&self.name("console")) => {
                            return self.unsupported("the console object as a value", at);
                        }
                        "NaN" => num(f64::NAN, at),
                        "Infinity" => num(f64::INFINITY, at),
                        name if self.constant_expr(name, at).is_some() => {
                            self.constant_expr(name, at).unwrap()
                        }
                        name if self.signatures.contains_key(&self.name(name)) => {
                            let sig = &self.signatures[&self.name(name)];
                            self.function_value(sig, at)?
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
                } else if u.operator.as_str() == "delete" {
                    let mut operand = &u.argument;
                    while let js::Expression::ParenthesizedExpression(p) = operand {
                        operand = &p.expression;
                    }
                    if !matches!(
                        operand,
                        js::Expression::StaticMemberExpression(_)
                            | js::Expression::ComputedMemberExpression(_)
                            | js::Expression::PrivateFieldExpression(_)
                            | js::Expression::ChainExpression(_)
                    ) {
                        return self.error(
                            ErrorKind::Type,
                            "The operand of a 'delete' operator must be a property reference.",
                            operand.span(),
                        );
                    }
                    // The operand is checked like any read before the gap is reported.
                    self.expr(operand)?;
                    return self.unsupported("delete", at);
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
                    let mut value = self.expr(&u.argument)?;
                    if u.operator.as_str() == "~" {
                        return self.bitwise_not(value, at);
                    }
                    // Unary plus and minus convert any defined operand to a number.
                    if let Err(error) = self.expect(&value.ty, &number(), at) {
                        if matches!(value.ty, Type::Optional(_)) {
                            return Err(error);
                        }
                        if value.ty != boolean() {
                            return self.unsupported("numeric conversion of this type", at);
                        }
                        // Through an integer: false and true are 0 and 1.
                        let integer = Type::Primitive(PrimitiveType::I32);
                        let bits = node(
                            TypedExpression::Cast(TypedCast {
                                expr: Box::new(value),
                                target_type: integer.clone(),
                            }),
                            integer,
                            at,
                        );
                        value = node(
                            TypedExpression::Cast(TypedCast {
                                expr: Box::new(bits),
                                target_type: number(),
                            }),
                            number(),
                            at,
                        );
                    }
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
                    self.logical_operand(&left.ty, at)?;
                    self.narrow(&l.left, op == "&&");
                    let right = self.expr(&l.right)?;
                    if matches!(right.ty, Type::Optional(_)) {
                        return self.unsupported(
                            "value-selecting logical operations on optional values",
                            at,
                        );
                    }
                    self.logical_operand(&right.ty, at)?;
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
                let mut a = self.expr(&c.consequent)?;
                let a_static = a.ty != string() || self.immortal_string(&c.consequent);
                let then_flow = std::mem::replace(&mut self.flow, before);
                self.narrow(&c.test, false);
                let mut b = self.expr(&c.alternate)?;
                let b_static = b.ty != string() || self.immortal_string(&c.alternate);
                self.merge_flow(&then_flow);
                if a.ty != b.ty
                    && matches!((&a.ty, &b.ty), (Type::Named { .. }, Type::Named { .. }))
                {
                    return self
                        .unsupported("conditional objects with different physical layouts", at);
                }
                let mut joined = optional_join(&a.ty, &b.ty);
                if joined.is_none()
                    && (matches!(a.ty, Type::TypeVar(_)) || matches!(b.ty, Type::TypeVar(_)))
                {
                    a = self.constrained_scalar(a, c.consequent.span());
                    b = self.constrained_scalar(b, c.alternate.span());
                    joined = optional_join(&a.ty, &b.ty);
                    if joined.is_none() {
                        return self.unsupported("conditional unions of generic parameters", at);
                    }
                }
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
            js::Expression::StaticMemberExpression(m)
                if matches!(m.object, js::Expression::Super(_)) =>
            {
                return self.super_member(m);
            }
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                if let Some(member) = self.enum_member(e) {
                    return member;
                }
                if let Some(sig) = self.namespace_function(e)? {
                    return self.function_value(sig, at);
                }
                if let Some(constant) = self.math_constant(m) {
                    return Ok(constant);
                }
                self.field_read(&m.object, m.property.name.as_str(), at)?
            }
            js::Expression::ArrayExpression(a) => self.array_literal(a, None)?,
            js::Expression::ComputedMemberExpression(m) if !m.optional => {
                if let Some(member) = self.enum_member(e) {
                    return member;
                }
                if let Some(sig) = self.namespace_function(e)? {
                    return self.function_value(sig, at);
                }
                if let Some(key) = Self::string_key(&m.expression) {
                    return self.field_read(&m.object, key, at);
                }
                self.array_index(&m.object, &m.expression, at)?
            }
            js::Expression::ObjectExpression(o) => self.object(o)?,
            js::Expression::CallExpression(c) if !c.optional => self.call_expr(c)?,
            js::Expression::NewExpression(n) => self.new_expr(n)?,
            js::Expression::ThisExpression(t) => self.this_expr(t.span)?,
            js::Expression::AssignmentExpression(a) => self.assignment(a)?,
            js::Expression::UpdateExpression(u) => {
                let target = self.simple_target(&u.argument, at)?;
                let read = self.read_target(&target, at)?;
                self.expect(&read.ty, &number(), at)?;
                let (mut statements, target) = self.capture_target(target, at);
                if let TypedExpression::Variable(name) = target.node {
                    statements.extend(self.global_guard(name, true, at));
                }
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
                statements.extend([save, expression(assign)]);
                if let TypedExpression::Variable(name) = target.node {
                    statements.extend(self.global_assigned(name, at));
                }
                statements.push(expression(result));
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
        let left = self.constrained_scalar(left, b.left.span());
        let right = self.constrained_scalar(right, b.right.span());
        let op = b.operator.as_str();
        if matches!(op, "===" | "!==" | "==" | "!=") {
            self.check_comparison(&b.left, &b.right, at)?;
        }
        if matches!(op, "===" | "!==" | "==" | "!=")
            && left.ty == string()
            && right.ty == string()
            && let (Some(left), Some(right)) = (
                constants::literal_text(&b.left),
                constants::literal_text(&b.right),
            )
            && left != right
        {
            return self.error(ErrorKind::Type, "comparison has no overlapping types", at);
        }
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
        if op == "+" && (left.ty == string() || right.ty == string()) {
            let left = self.text_of(left, at)?;
            let right = self.text_of(right, at)?;
            return Ok(self.call("$String$concat", vec![left, right], string(), at));
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
        if matches!(op, "|" | "&" | "^" | "<<" | ">>" | ">>>") {
            return self.bitwise(op, left, right, at);
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

    /// `object.name`, or `object["name"]`, which names the same property.
    fn field_read(
        &mut self,
        receiver: &js::Expression<'ast>,
        name: &'ast str,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        let object = self.expr(receiver)?;
        if element_type(&object.ty).is_some() && name == "length" {
            return Ok(self.array_length(object, at));
        }
        self.check_field_access(receiver, name, at)?;
        let ty = self.field_type(self.source_type(receiver, &object.ty), name, at)?;
        Ok(node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(object),
                field: self.name(name),
            }),
            ty,
            at,
        ))
    }

    /// The property a computed member names with a string literal.
    fn string_key(key: &js::Expression<'ast>) -> Option<&'ast str> {
        match key {
            js::Expression::StringLiteral(s) if !s.lone_surrogates => Some(s.value.as_str()),
            js::Expression::TemplateLiteral(t) if t.expressions.is_empty() => {
                let quasi = &t.quasis[0];
                quasi
                    .value
                    .cooked
                    .as_ref()
                    .filter(|_| !quasi.lone_surrogates)
                    .map(|text| text.as_str())
            }
            js::Expression::ParenthesizedExpression(p) => Self::string_key(&p.expression),
            _ => None,
        }
    }

    fn field_type(&self, ty: &Type, field: &'ast str, at: oxc_span::Span) -> Result<Type> {
        if matches!(ty, Type::Function { .. }) {
            return self.unsupported("function object properties", at);
        }
        // Library members of primitives, arrays and unions are not modelled yet.
        if matches!(
            ty,
            Type::Primitive(PrimitiveType::F64 | PrimitiveType::Bool | PrimitiveType::String)
                | Type::Array { .. }
                | Type::Union(_)
        ) {
            return self.unsupported("properties of primitive and array values", at);
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
            self.plain_enum(self.view_of(&p.value), "enum-valued object fields", p.span)?;
            if field_primitive(&value.ty).is_none() {
                return self.unsupported("reference-valued object fields", p.span);
            }
            if value.ty == string() && !self.immortal_string(&p.value) {
                return self.unsupported("string fields without a proven static lifetime", p.span);
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
        list: &[js::Argument<'ast>],
        at: oxc_span::Span,
        mut hints: impl Iterator<Item = &'hint Type>,
        mut static_strings: Option<&mut SmallVec<[bool; 4]>>,
    ) -> Result<Vec<Expr>> {
        let mut args = Vec::with_capacity(list.len());
        for arg in list {
            let Some(e) = arg.as_expression() else {
                return self.unsupported("spread arguments", at);
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
        if let js::Expression::Super(s) = &c.callee {
            return self.misplaced_super_call(s.span, at);
        }
        if let js::Expression::StaticMemberExpression(m) = &c.callee {
            if let Some(filled) = self.array_fill(c, m, None)? {
                return Ok(filled);
            }
        }
        if let js::Expression::StaticMemberExpression(m) = &c.callee {
            if let js::Expression::Identifier(receiver) = &m.object {
                self.require_value(receiver.name.as_str(), receiver.span)?;
                let console = receiver.name == "console"
                    && !self
                        .namespace_references
                        .contains_key(&receiver.node_id.get())
                    && !self.imported_exports.contains_key(&self.name("console"))
                    && !self
                        .scopes
                        .iter()
                        .any(|s| s.contains_key(&self.name("console")));
                if console && m.property.name != "log" {
                    return self.unsupported("console methods other than log", at);
                }
                if console {
                    if c.type_arguments.is_some() {
                        return self.error(
                            ErrorKind::Type,
                            "console.log does not take type arguments",
                            at,
                        );
                    }
                    let args = self.arguments(&c.arguments, c.span, std::iter::empty(), None)?;
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
            // An array reached by an element access, as in `grid[i].push(x)`.
            if !m.optional && matches!(m.object, js::Expression::ComputedMemberExpression(_)) {
                let receiver = self.expr(&m.object)?;
                if element_type(&receiver.ty).is_some() {
                    return self.array_method_on(c, m, None, receiver);
                }
            }
        }
        if let js::Expression::StaticMemberExpression(m) = &c.callee {
            if !m.optional {
                if let Some(call) = self.math_call(c, m)? {
                    return Ok(call);
                }
                if let Some(call) = self.timing_call(c, m)? {
                    return Ok(call);
                }
                if let Some(call) = self.method_call(c, m)? {
                    return Ok(call);
                }
            }
        }
        let direct = self.direct_signature(&c.callee)?.cloned();
        let Some(sig) = direct else {
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
            let args = self.arguments(&c.arguments, c.span, params.iter().map(|p| &p.ty), None)?;
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
        self.call_with_signature(sig, &c.arguments, c.type_arguments.as_deref(), None, at)
    }

    /// A call through a resolved signature; a receiver becomes the first argument.
    fn call_with_signature(
        &mut self,
        sig: Arc<Signature>,
        arguments: &[js::Argument<'ast>],
        type_arguments: Option<&js::TSTypeParameterInstantiation<'ast>>,
        receiver: Option<Expr>,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        let arity = self.arities.get(&sig.native).copied();
        let rest = arity.is_some_and(|a| a.rest);
        let fixed = sig.params.len() - usize::from(rest);
        let required = arity.map_or(fixed, |a| a.required);
        if arguments.len() < required || (!rest && arguments.len() > fixed) {
            return self.error(ErrorKind::Type, "wrong argument count", at);
        }
        let (arguments, extra) = arguments.split_at(arguments.len().min(fixed));
        let mut types: SmallVec<[Option<(Type, Option<InternedString>)>; 2]> =
            std::iter::repeat_n(None, sig.type_params.len()).collect();
        let explicit = type_arguments.is_some();
        if let Some(arguments) = &type_arguments {
            let required = sig
                .type_rules()
                .iter()
                .position(|r| r.default.is_some())
                .unwrap_or(types.len());
            if arguments.params.len() < required || arguments.params.len() > types.len() {
                return self.error(ErrorKind::Type, "wrong type argument count", at);
            }
            for (slot, ty) in types.iter_mut().zip(&arguments.params) {
                *slot = Some(self.ty(ty)?);
            }
            self.complete_type_arguments(&sig, &mut types, at)?;
        }
        let hints: SmallVec<[Type; 4]> = sig
            .params
            .iter()
            .map(|parameter| {
                if explicit {
                    self.instantiate_type(
                        &parameter.ty,
                        parameter.view,
                        &sig.type_params,
                        &types,
                        at,
                    )
                    .map(|(ty, _)| ty)
                } else {
                    Ok(parameter.ty.clone())
                }
            })
            .collect::<Result<_>>()?;
        let mut static_strings = SmallVec::new();
        let args = self.arguments(arguments, at, hints.iter(), Some(&mut static_strings))?;
        if !explicit {
            for ((arg, source), parameter) in args.iter().zip(arguments).zip(&sig.params) {
                let source = source.as_expression().unwrap();
                self.infer_arguments(
                    &parameter.ty,
                    self.source_type(source, &arg.ty),
                    self.view_of(source),
                    &sig.type_params,
                    &mut types,
                );
            }
            self.complete_type_arguments(&sig, &mut types, at)?;
        }
        self.check_type_arguments(&sig, &types, at)?;
        let args = args
            .into_iter()
            .zip(arguments)
            .zip(&sig.params)
            .zip(static_strings)
            .map(|(((arg, source), parameter), static_string)| {
                let (expected, view) = self.instantiate_type(
                    &parameter.ty,
                    parameter.view,
                    &sig.type_params,
                    &types,
                    at,
                )?;
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
                self.check_enum(source, view, at)?;
                self.adapt_value(arg, &expected, at)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut args = args;
        // Omitted arguments are `undefined`; a rest parameter collects the remaining ones.
        for parameter in &sig.params[args.len()..fixed] {
            let (expected, _) =
                self.instantiate_type(&parameter.ty, parameter.view, &sig.type_params, &types, at)?;
            let undefined = var(self.name("None"), undefined(), at);
            args.push(self.adapt_value(undefined, &expected, at)?);
        }
        if rest {
            let parameter = sig.params.last().unwrap();
            let (expected, _) =
                self.instantiate_type(&parameter.ty, parameter.view, &sig.type_params, &types, at)?;
            if element_type(&expected).is_some_and(|e| matches!(e, Type::TypeVar(_))) {
                return self.unsupported("generic rest parameters", at);
            }
            args.push(self.rest_arguments(extra, &expected, at)?);
        }
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
                positional_args: receiver.into_iter().chain(args).collect(),
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
        if let TypedExpression::Variable(name) = target.node {
            self.require_assigned(name, at)?;
        }
        let flow = if let TypedExpression::Variable(name) = target.node {
            self.flow.get(&name).copied()
        } else {
            None
        };
        self.flow_read(target.clone(), flow, at)
    }

    fn record_write(&mut self, target: &Expr, ty: &Type) {
        if let TypedExpression::Variable(name) = target.node {
            if self.unassigned.contains_key(&name) {
                self.flow.assign(name);
            }
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
                let (name, ty) = (b.name, b.ty.clone());
                self.require_native(name, at)?;
                Ok(var(name, ty, at))
            }
            js::SimpleAssignmentTarget::StaticMemberExpression(m) => {
                self.field_target(&m.object, m.property.name.as_str(), m.span)
            }
            js::SimpleAssignmentTarget::ComputedMemberExpression(m)
                if !m.optional && Self::string_key(&m.expression).is_some() =>
            {
                let key = Self::string_key(&m.expression).unwrap();
                self.field_target(&m.object, key, m.span)
            }
            js::SimpleAssignmentTarget::ComputedMemberExpression(m) if !m.optional => {
                self.check_namespace_write(&m.object, at)?;
                self.array_index(&m.object, &m.expression, at)
            }
            _ => self.unsupported("this assignment target", at),
        }
    }

    fn field_target(
        &mut self,
        receiver: &js::Expression<'ast>,
        name: &'ast str,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        self.check_namespace_write(receiver, at)?;
        if let js::Expression::Identifier(id) = receiver {
            if self
                .binding(id.name.as_str(), id.span)
                .is_ok_and(|b| element_type(&b.ty).is_some())
            {
                return self.unsupported(
                    "array property mutation, including sparse length changes",
                    at,
                );
            }
        }
        self.check_field_access(receiver, name, at)?;
        if !self.readonly_field_write_allowed(receiver)
            && self
                .view_of(receiver)
                .and_then(|n| self.views.get(&n))
                .and_then(|view| view.field(self.name(name)))
                .is_some_and(|f| f.readonly)
        {
            return self.error(ErrorKind::Type, "cannot assign to readonly property", at);
        }
        // Assignment receivers currently require a lexical reference view.
        if !matches!(
            receiver,
            js::Expression::Identifier(_) | js::Expression::ThisExpression(_)
        ) {
            return self.unsupported("assignment through a computed receiver", at);
        }
        let object = self.expr(receiver)?;
        let ty = self.field_type(self.source_type(receiver, &object.ty), name, at)?;
        Ok(node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(object),
                field: self.name(name),
            }),
            ty,
            at,
        ))
    }

    fn assignment(&mut self, a: &js::AssignmentExpression<'ast>) -> Result<Expr> {
        let target = match &a.left {
            js::AssignmentTarget::AssignmentTargetIdentifier(i) => {
                let b = self.binding(i.name.as_str(), a.span)?;
                if !b.mutable {
                    return self.error(ErrorKind::Type, "cannot assign to const", a.span);
                }
                let (name, ty) = (b.name, b.ty.clone());
                self.require_native(name, a.span)?;
                var(name, ty, a.span)
            }
            js::AssignmentTarget::StaticMemberExpression(m) => {
                self.field_target(&m.object, m.property.name.as_str(), m.span)?
            }
            js::AssignmentTarget::ComputedMemberExpression(m)
                if !m.optional && Self::string_key(&m.expression).is_some() =>
            {
                let key = Self::string_key(&m.expression).unwrap();
                self.field_target(&m.object, key, m.span)?
            }
            js::AssignmentTarget::ComputedMemberExpression(m) if !m.optional => {
                self.check_namespace_write(&m.object, a.span)?;
                self.array_index(&m.object, &m.expression, a.span)?
            }
            _ => return self.unsupported("this assignment target", a.span),
        };
        let indexed = match &target.node {
            TypedExpression::Index(_) => true,
            TypedExpression::Block(block) => Self::wraps_index(block),
            _ => false,
        };
        if indexed && matches!(target.ty, Type::TypeVar(_)) {
            return self.unsupported(
                "generic array mutation before element ownership is known",
                a.span,
            );
        }
        let (mut statements, target) = self.capture_target(target, a.span);
        if let TypedExpression::Variable(name) = target.node {
            statements.extend(self.global_guard(name, a.operator.as_str() != "=", a.span));
        }
        let target_view = if let js::AssignmentTarget::AssignmentTargetIdentifier(i) = &a.left {
            self.binding(i.name.as_str(), a.span)?.view
        } else {
            None
        };
        let logical_target = target_view
            .and_then(|view| self.views.get(&view))
            .map_or(&target.ty, |view| &view.ty)
            .clone();
        let op = a.operator.as_str();
        if matches!(op, "??=" | "||=" | "&&=") {
            return self.unsupported("logical assignment operators", a.span);
        }
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
        if op == "=" {
            self.check_enum(&a.right, target_view, a.span)?;
        }
        if matches!(target.ty, Type::Named { .. }) && value.ty != target.ty {
            return self.unsupported(
                "assignment changing a binding's physical record layout",
                a.span,
            );
        }
        if let Some(read) = read {
            self.expect(&read.ty, &number(), a.span)?;
            self.expect(&value.ty, &number(), a.span)?;
            let arithmetic = match op {
                "+=" => Some(BinaryOp::Add),
                "-=" => Some(BinaryOp::Sub),
                "*=" => Some(BinaryOp::Mul),
                "/=" => Some(BinaryOp::Div),
                "%=" => Some(BinaryOp::Rem),
                "|=" | "&=" | "^=" | "<<=" | ">>=" | ">>>=" => None,
                _ => return self.unsupported("this compound assignment", a.span),
            };
            value = match arithmetic {
                Some(operation) => binary(operation, read, value, number(), a.span),
                None => self.bitwise(&op[..op.len() - 1], read, value, a.span)?,
            };
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
                self.enum_facts(name, target_view, &a.right);
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
        if matches!(target.node, TypedExpression::Field(_))
            && target.ty == string()
            && (op != "=" || !self.immortal_string(&a.right))
        {
            return self.unsupported("string fields without a proven static lifetime", a.span);
        }
        if let TypedExpression::Variable(name) = target.node
            && self.globals.contains_key(&name)
            && target.ty == string()
            && (op != "=" || !self.immortal_string(&a.right))
        {
            return self.unsupported(
                "module string bindings without a proven static lifetime",
                a.span,
            );
        }
        let stored = self.adapt_value(var(name, ty.clone(), a.span), &target.ty, a.span)?;
        let assigned = match target.node {
            TypedExpression::Variable(global) => self.global_assigned(global, a.span),
            _ => None,
        };
        let assignment = binary(BinaryOp::Assign, target, stored, unit(), a.span);
        statements.extend([save, expression(assignment)]);
        statements.extend(assigned);
        statements.push(expression(var(name, ty.clone(), a.span)));
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
            // An element read that bound its receiver first: the binding runs ahead of the
            // captured target.
            TypedExpression::Block(mut block) if Self::wraps_index(&block) => {
                let Some(TypedNode {
                    node: TypedStatement::Expression(last),
                    ..
                }) = block.statements.pop()
                else {
                    unreachable!("wraps_index checked the last statement");
                };
                let (mut captured, target) = self.capture_target(*last, at);
                block.statements.append(&mut captured);
                (block.statements, target)
            }
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

    fn wraps_index(block: &TypedBlock) -> bool {
        matches!(
            block.statements.last(),
            Some(TypedNode {
                node: TypedStatement::Expression(last),
                ..
            }) if matches!(last.node, TypedExpression::Index(_))
        )
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
