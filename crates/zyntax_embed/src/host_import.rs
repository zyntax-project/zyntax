//! A host module as declarations a program imports.
//!
//! Each class becomes an opaque type whose value is the foreign object's
//! box, and each member a generated function that reaches the object
//! through the foreign-object protocol ([`crate::foreign`]): its
//! arguments boxed and its result unboxed at the types the host declared,
//! a host object passed as the box it already is.
//!
//! The declarations are built as a typed program, so every language that
//! lowers through the embedder imports host modules the same way.

use std::collections::{HashMap, HashSet};

use zyntax_typed_ast::type_registry::{
    FieldDef, Mutability, NullabilityKind, PrimitiveType, Type, TypeDefinition, TypeId, TypeKind,
    TypeMetadata, TypeRegistry, Visibility,
};
use zyntax_typed_ast::typed_ast::{
    BinaryOp, ParameterKind, TypedBinary, TypedBlock, TypedCall, TypedCast, TypedDeclaration,
    TypedExpression, TypedExtern, TypedExternStruct, TypedFieldAccess, TypedFunction, TypedIf,
    TypedImport, TypedImportItem, TypedLet, TypedLiteral, TypedMethod, TypedMethodParam, TypedNode,
    TypedParameter, TypedStatement, TypedTraitImpl,
};
use zyntax_typed_ast::{InternedString, Span, TypedProgram};

use crate::host::{
    HostClass, HostField, HostMethod, HostModule, HostType, NativeBinding, NativePass, NativeType,
};
use crate::runtime::{RuntimeError, RuntimeResult};

/// Answers a module path with the host module it names, if the host has
/// one.
pub type HostModuleResolver = Box<dyn Fn(&str) -> Option<HostModule>>;

/// The most arguments a member may take: the protocol's fixed arities.
const MAX_ARITY: usize = 8;

/// Replace every import a host module answers with that module's
/// declarations. An import no host module answers is left for the
/// import chain.
///
/// An import names its module by its whole path, longest first:
/// `import a.b` is module `a.b`, or else member `b` of module `a`; `from
/// a import X` is module `a.X`, or else member `X` of module `a`.
pub(crate) fn expand_host_imports(
    program: &mut TypedProgram,
    registry: &mut TypeRegistry,
    resolvers: &[HostModuleResolver],
) -> RuntimeResult<()> {
    if resolvers.is_empty() {
        return Ok(());
    }
    let ask = |path: &str| resolvers.iter().find_map(|r| r(path));
    let mut builder = Builder::default();
    let mut kept = Vec::with_capacity(program.declarations.len());
    for decl in std::mem::take(&mut program.declarations) {
        if let TypedDeclaration::Import(import) = &decl.node
            && let Some(modules) = resolve(import, &ask)?
        {
            for module in modules {
                builder.module(&module)?;
            }
            continue;
        }
        kept.push(decl);
    }
    // Ahead of the program's own, as an import's are: what lowers first
    // records the signatures what lowers after it calls through.
    for (name, id, ty) in &builder.views {
        register_view(registry, name, *id, ty);
    }
    let mut declarations = builder.declarations;
    declarations.extend(kept);
    program.declarations = declarations;
    Ok(())
}

/// The host modules `import` names, or `None` when it names none.
fn resolve(
    import: &TypedImport,
    ask: &dyn Fn(&str) -> Option<HostModule>,
) -> RuntimeResult<Option<Vec<HostModule>>> {
    let segments: Vec<String> = import
        .module_path
        .iter()
        .filter_map(|s| s.resolve_global())
        .collect();
    let path = segments.join(".");
    let mut named: Vec<(String, Option<String>)> = Vec::new();
    for item in &import.items {
        match item {
            TypedImportItem::Named { name, alias } => named.push((
                name.resolve_global().unwrap_or_default(),
                alias.and_then(|a| a.resolve_global()),
            )),
            TypedImportItem::Glob => {}
            TypedImportItem::Default(_) => return Ok(None),
        }
    }
    if named.is_empty() {
        if let Some(module) = ask(&path) {
            return Ok(Some(vec![module]));
        }
        let Some((parent, member)) = path.rsplit_once('.') else {
            return Ok(None);
        };
        return match ask(parent) {
            Some(module) => {
                check_member(&module, member)?;
                Ok(Some(vec![module]))
            }
            None => Ok(None),
        };
    }
    let mut modules: Vec<HostModule> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for (name, alias) in &named {
        if let Some(alias) = alias
            && alias != name
        {
            return Err(RuntimeError::Execution(format!(
                "`{name}` is imported from host module `{path}` as `{alias}`, and a host member \
                 is imported under its own name only"
            )));
        }
        let longer = format!("{path}.{name}");
        if let Some(module) = ask(&longer) {
            check_member(&module, name)?;
            modules.push(module);
        } else if let Some(module) = ask(&path) {
            check_member(&module, name)?;
            modules.push(module);
        } else {
            missing.push(name.clone());
        }
    }
    match (modules.is_empty(), missing.is_empty()) {
        (true, _) => Ok(None),
        (false, true) => Ok(Some(modules)),
        (false, false) => Err(RuntimeError::Execution(format!(
            "host module `{path}` has no member {}",
            missing
                .iter()
                .map(|m| format!("`{m}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

fn check_member(module: &HostModule, member: &str) -> RuntimeResult<()> {
    let found = module.classes.iter().any(|c| c.name == member)
        || module.functions.iter().any(|f| f.name == member);
    if found {
        return Ok(());
    }
    let members: Vec<&str> = module
        .classes
        .iter()
        .map(|c| c.name.as_str())
        .chain(module.functions.iter().map(|f| f.name.as_str()))
        .collect();
    Err(RuntimeError::Execution(format!(
        "host module `{}` has no member `{member}`; it has {}",
        module.name,
        if members.is_empty() {
            "none".to_string()
        } else {
            members.join(", ")
        }
    )))
}

/// How a value of a host type crosses into the protocol and back.
#[derive(Clone)]
enum Crossing {
    /// Boxed with `make` on the way in and read with `read` on the way
    /// out; the box is freed after the call unless `owns` says the value
    /// read out still lives in it.
    Boxed {
        ty: Type,
        make: &'static str,
        read: &'static str,
        owns: bool,
    },
    /// Already a box: a host object, or a dynamic value.
    Box(Type),
    /// A host object of a class the program holds as the host's word,
    /// boxed for the protocol and taken back out of a result.
    Word(Type),
    Void,
}

impl Crossing {
    fn ty(&self) -> Type {
        match self {
            Crossing::Boxed { ty, .. } | Crossing::Box(ty) | Crossing::Word(ty) => ty.clone(),
            Crossing::Void => Type::Primitive(PrimitiveType::Unit),
        }
    }

    /// The type the protocol's call is declared to return for it.
    fn wire(&self) -> Type {
        match self {
            Crossing::Box(ty) => ty.clone(),
            _ => Type::Any,
        }
    }
}

/// The runtime functions the generated members call, by the name the
/// program declares them under: (name, link name, parameters, result).
fn helpers() -> Vec<(&'static str, &'static str, Vec<Type>, Type)> {
    let prim = Type::Primitive;
    vec![
        (
            "host$box_i64",
            "zyntax_box_i64",
            vec![prim(PrimitiveType::I64)],
            Type::Any,
        ),
        (
            "host$box_f64",
            "zyntax_box_f64",
            vec![prim(PrimitiveType::F64)],
            Type::Any,
        ),
        (
            "host$box_bool",
            "zyntax_box_bool",
            vec![prim(PrimitiveType::I32)],
            Type::Any,
        ),
        (
            "host$box_str",
            "zyntax_box_str",
            vec![prim(PrimitiveType::String)],
            Type::Any,
        ),
        (
            "host$read_i64",
            "zyntax_box_get_i64",
            vec![Type::Any],
            prim(PrimitiveType::I64),
        ),
        (
            "host$read_f64",
            "zyntax_box_get_f64",
            vec![Type::Any],
            prim(PrimitiveType::F64),
        ),
        (
            "host$read_bool",
            "zyntax_box_get_bool",
            vec![Type::Any],
            prim(PrimitiveType::I32),
        ),
        (
            "host$read_str",
            "zyntax_box_get_opaque",
            vec![Type::Any],
            prim(PrimitiveType::String),
        ),
        (
            "host$free",
            "zyntax_box_free",
            vec![Type::Any],
            prim(PrimitiveType::Unit),
        ),
        (
            "host$import",
            "$Foreign$import",
            vec![prim(PrimitiveType::String)],
            Type::Any,
        ),
        (
            "host$get",
            "$Foreign$get",
            vec![Type::Any, prim(PrimitiveType::String)],
            Type::Any,
        ),
        (
            "host$read_word",
            "zyntax_box_get_opaque",
            vec![Type::Any],
            prim(PrimitiveType::I64),
        ),
        (
            "host$box_word",
            "$Foreign$box_retained_word",
            vec![prim(PrimitiveType::I64)],
            Type::Any,
        ),
        (
            "host$take_word",
            "$Foreign$retain_box_word",
            vec![Type::Any],
            prim(PrimitiveType::I64),
        ),
    ]
}

/// The host functions a natively bound member's call site uses, declared
/// only where a module binds one: a host with no bindings registers none.
fn native_helpers() -> Vec<(&'static str, &'static str, Vec<Type>, Type)> {
    let prim = Type::Primitive;
    vec![
        (
            "host$pending_flag",
            "$Host$pending_flag",
            vec![],
            prim(PrimitiveType::I64),
        ),
        (
            "host$raise_pending",
            "$Host$raise_pending",
            vec![],
            prim(PrimitiveType::Unit),
        ),
        (
            "host$text_to_string",
            "$Host$text_to_string",
            vec![prim(PrimitiveType::I64)],
            prim(PrimitiveType::String),
        ),
    ]
}

/// A view of one value of `ty` at an address: a header-free reference
/// struct whose only field `v` is read and written in place.
fn register_view(registry: &mut TypeRegistry, name: &str, id: TypeId, ty: &Type) {
    let field = FieldDef {
        name: s("v"),
        ty: ty.clone(),
        visibility: Visibility::Public,
        mutability: Mutability::Mutable,
        is_static: false,
        span: span(),
        getter: None,
        setter: None,
        is_synthetic: false,
    };
    let mut metadata = TypeMetadata {
        is_reference: true,
        ..Default::default()
    };
    metadata.custom.insert(
        s(zyntax_compiler::object_header::HEADER_FREE_KEY),
        String::new(),
    );
    registry.register_type(TypeDefinition {
        id,
        module: None,
        name: s(name),
        kind: TypeKind::Struct {
            fields: vec![field.clone()],
            is_tuple: false,
        },
        type_params: vec![],
        constraints: vec![],
        fields: vec![field],
        methods: vec![],
        constructors: vec![],
        metadata,
        span: span(),
    });
}

/// The primitive a native operand of `ty` is passed as.
fn native_prim(ty: &NativeType) -> Option<PrimitiveType> {
    Some(match ty {
        NativeType::Bool | NativeType::I8 => PrimitiveType::I8,
        NativeType::I16 => PrimitiveType::I16,
        NativeType::I32 => PrimitiveType::I32,
        NativeType::I64 | NativeType::Word | NativeType::Str | NativeType::Object { .. } => {
            PrimitiveType::I64
        }
        NativeType::U8 => PrimitiveType::U8,
        NativeType::U16 => PrimitiveType::U16,
        NativeType::U32 => PrimitiveType::U32,
        NativeType::U64 => PrimitiveType::U64,
        NativeType::F32 => PrimitiveType::F32,
        NativeType::F64 => PrimitiveType::F64,
        NativeType::Void => return None,
    })
}

#[derive(Default)]
struct Builder {
    declarations: Vec<TypedNode<TypedDeclaration>>,
    /// Classes declared so far, by name, with the module that declared
    /// them.
    classes: HashMap<String, String>,
    functions: HashSet<String>,
    modules: HashSet<String>,
    helpers: bool,
    native_helpers: bool,
    /// View structs the native lowering reads through, by name: their
    /// type ids and the type of their one field.
    views: Vec<(String, TypeId, Type)>,
    /// Classes held as the host's word, by name.
    word_classes: HashSet<String>,
    /// Native functions declared so far, by the name the program calls.
    natives: HashSet<String>,
    /// Bindings a native body makes ahead of its call.
    pending_binds: Vec<TypedNode<TypedStatement>>,
    fresh: usize,
}

fn span() -> Span {
    Span::default()
}

fn s(text: &str) -> InternedString {
    InternedString::new_global(text)
}

fn node<T>(value: T, ty: Type) -> TypedNode<T> {
    TypedNode::new(value, ty, span())
}

fn unit() -> Type {
    Type::Primitive(PrimitiveType::Unit)
}

fn var(name: &str, ty: Type) -> TypedNode<TypedExpression> {
    node(TypedExpression::Variable(s(name)), ty)
}

fn string(text: &str) -> TypedNode<TypedExpression> {
    node(
        TypedExpression::Literal(TypedLiteral::String(s(text))),
        Type::Primitive(PrimitiveType::String),
    )
}

fn call(
    function: &str,
    args: Vec<TypedNode<TypedExpression>>,
    ty: Type,
) -> TypedNode<TypedExpression> {
    node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(var(function, unit())),
            positional_args: args,
            named_args: vec![],
            type_args: vec![],
        }),
        ty,
    )
}

fn statement(expr: TypedNode<TypedExpression>) -> TypedNode<TypedStatement> {
    node(TypedStatement::Expression(Box::new(expr)), unit())
}

fn bind(name: &str, value: TypedNode<TypedExpression>) -> TypedNode<TypedStatement> {
    let ty = value.ty.clone();
    node(
        TypedStatement::Let(TypedLet {
            name: s(name),
            ty,
            mutability: Mutability::Immutable,
            initializer: Some(Box::new(value)),
            span: span(),
        }),
        unit(),
    )
}

fn ret(value: Option<TypedNode<TypedExpression>>) -> TypedNode<TypedStatement> {
    node(TypedStatement::Return(value.map(Box::new)), Type::Never)
}

fn extern_function(
    name: &str,
    link: &str,
    params: Vec<Type>,
    result: Type,
) -> TypedNode<TypedDeclaration> {
    let mut function = TypedFunction {
        name: s(name),
        params: params
            .into_iter()
            .enumerate()
            .map(|(i, ty)| TypedParameter {
                name: s(&format!("a{i}")),
                ty,
                ..Default::default()
            })
            .collect(),
        return_type: result,
        body: None,
        visibility: Visibility::Public,
        is_external: true,
        link_name: Some(s(link)),
        ..Default::default()
    };
    function.mark_generated();
    node(TypedDeclaration::Function(function), unit())
}

fn cast_to(value: TypedNode<TypedExpression>, ty: Type) -> TypedNode<TypedExpression> {
    if value.ty == ty {
        return value;
    }
    node(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(value),
            target_type: ty.clone(),
        }),
        ty,
    )
}

/// An object, or a string, as the word it is.
fn as_word(value: TypedNode<TypedExpression>) -> TypedNode<TypedExpression> {
    cast_to(value, Type::Primitive(PrimitiveType::I64))
}

fn add(left: TypedNode<TypedExpression>, offset: i64) -> TypedNode<TypedExpression> {
    let ty = left.ty.clone();
    node(
        TypedExpression::Binary(TypedBinary {
            op: BinaryOp::Add,
            left: Box::new(left),
            right: Box::new(node(
                TypedExpression::Literal(TypedLiteral::Integer(offset as i128)),
                Type::Primitive(PrimitiveType::I64),
            )),
        }),
        ty,
    )
}

/// A class's type by its name, as a parsed `extern struct` names it;
/// the import chain resolves it to the extern type it declares.
fn class_type(name: &str) -> Type {
    Type::Unresolved(s(name))
}

/// A member's signature crossed into protocol terms.
struct Signature {
    params: Vec<Crossing>,
    result: Crossing,
}

/// What a member call is made on.
enum Receiver {
    /// The instance the method is called on, as its `self`.
    Instance(Type),
    /// The instance, held as the host's word: boxed for the call.
    Boxing(Type),
    /// The class object, fetched for the call.
    Class(String),
    /// The module object, fetched for the call.
    Module,
}

impl Builder {
    fn module(&mut self, module: &HostModule) -> RuntimeResult<()> {
        if !self.modules.insert(module.name.clone()) {
            return Ok(());
        }
        if !self.helpers {
            self.helpers = true;
            for (name, link, params, result) in helpers() {
                self.declarations
                    .push(extern_function(name, link, params, result));
            }
        }
        for class in &module.classes {
            if let Some(other) = self.classes.insert(class.name.clone(), module.name.clone()) {
                return Err(RuntimeError::Execution(format!(
                    "host class `{}` is declared by both `{other}` and `{}`",
                    class.name, module.name
                )));
            }
            self.declarations.push(node(
                TypedDeclaration::Extern(TypedExtern::Struct(TypedExternStruct {
                    name: s(&class.name),
                    runtime_prefix: s(&format!("${}", class.name)),
                    type_params: vec![],
                })),
                unit(),
            ));
        }
        for class in &module.classes {
            if class.word {
                self.word_classes.insert(class.name.clone());
            }
        }
        for class in &module.classes {
            self.class(module, class)?;
        }
        for function in &module.functions {
            if !self.functions.insert(function.name.clone()) {
                return Err(RuntimeError::Execution(format!(
                    "host function `{}` is declared by two imported modules",
                    function.name
                )));
            }
            let signature = self.signature(module, function, &function.name)?;
            let body = match &function.native {
                Some(binding) => self.native_body(binding, &signature, None, &function.name)?,
                None => self.member_body(
                    module,
                    &Receiver::Module,
                    &format!("{}$host", function.name),
                    "$Foreign$invoke",
                    &function.name,
                    &signature,
                ),
            };
            let mut generated = TypedFunction {
                name: s(&function.name),
                params: signature
                    .params
                    .iter()
                    .enumerate()
                    .map(|(i, p)| TypedParameter {
                        name: s(&format!("p{i}")),
                        ty: p.ty(),
                        ..Default::default()
                    })
                    .collect(),
                return_type: signature.result.ty(),
                body: Some(body),
                visibility: Visibility::Public,
                ..Default::default()
            };
            generated.mark_generated();
            self.declarations
                .push(node(TypedDeclaration::Function(generated), unit()));
        }
        Ok(())
    }

    fn class(&mut self, module: &HostModule, class: &HostClass) -> RuntimeResult<()> {
        let this = class_type(&class.name);
        // The class object, for its constructor and statics.
        let fetch = format!("{}$host$class", class.name);
        let mut fetch_fn = TypedFunction {
            name: s(&fetch),
            params: vec![],
            return_type: Type::Any,
            body: Some(TypedBlock {
                statements: vec![
                    bind(
                        "m",
                        call("host$import", vec![string(&module.name)], Type::Any),
                    ),
                    bind(
                        "c",
                        call(
                            "host$get",
                            vec![var("m", Type::Any), string(&class.name)],
                            Type::Any,
                        ),
                    ),
                    statement(call("host$free", vec![var("m", Type::Any)], unit())),
                    ret(Some(var("c", Type::Any))),
                ],
                span: span(),
            }),
            visibility: Visibility::Public,
            ..Default::default()
        };
        fetch_fn.mark_generated();
        self.declarations
            .push(node(TypedDeclaration::Function(fetch_fn), unit()));

        let mut methods = Vec::new();
        if let Some(constructor) = &class.constructor {
            let signature = self.signature(module, constructor, &format!("{}::new", class.name))?;
            let signature = Signature {
                params: signature.params,
                result: if class.word {
                    Crossing::Word(this.clone())
                } else {
                    Crossing::Box(this.clone())
                },
            };
            let body = match &constructor.native {
                Some(binding) => {
                    self.native_body(binding, &signature, None, &format!("{}::new", class.name))?
                }
                None => self.member_body(
                    module,
                    &Receiver::Class(class.name.clone()),
                    &format!("{}$host$new", class.name),
                    "$Foreign$construct",
                    "",
                    &signature,
                ),
            };
            methods.push(method("new", None, &signature, body));
        }
        for m in &class.methods {
            let signature = self.signature(module, m, &format!("{}.{}", class.name, m.name))?;
            let receiver = if m.is_static {
                Receiver::Class(class.name.clone())
            } else if class.word {
                Receiver::Boxing(this.clone())
            } else {
                Receiver::Instance(this.clone())
            };
            let body = match &m.native {
                Some(binding) => self.native_body(
                    binding,
                    &signature,
                    (!m.is_static).then_some((&this, class.word)),
                    &format!("{}.{}", class.name, m.name),
                )?,
                None => self.member_body(
                    module,
                    &receiver,
                    &format!("{}$host${}", class.name, m.name),
                    "$Foreign$invoke",
                    &m.name,
                    &signature,
                ),
            };
            let self_ty = (!m.is_static).then(|| this.clone());
            methods.push(method(&m.name, self_ty, &signature, body));
        }
        for field in &class.fields {
            methods.extend(self.field(module, class, field)?);
        }
        self.declarations.push(node(
            TypedDeclaration::Impl(TypedTraitImpl {
                trait_name: s(""),
                trait_type_args: vec![],
                for_type: this,
                methods,
                associated_types: vec![],
                module: None,
                span: span(),
            }),
            unit(),
        ));
        Ok(())
    }

    /// `x()` and `set_x(v)` for a field `x`.
    fn field(
        &mut self,
        module: &HostModule,
        class: &HostClass,
        field: &HostField,
    ) -> RuntimeResult<Vec<TypedMethod>> {
        let what = format!("field {}.{}", class.name, field.name);
        let crossing = self.crossing(module, &field.ty, &what)?;
        if matches!(crossing, Crossing::Void) {
            return Err(RuntimeError::Execution(format!("{what} has no type")));
        }
        let this = class_type(&class.name);
        if field.native.is_some() {
            return self.native_field(class, field, crossing);
        }
        let (receiver, receiver_ty) = if field.is_static {
            (Receiver::Class(class.name.clone()), Type::Any)
        } else if class.word {
            (Receiver::Boxing(this.clone()), Type::Any)
        } else {
            (Receiver::Instance(this.clone()), this.clone())
        };
        let self_ty = (!field.is_static).then(|| this.clone());

        let get = format!("{}$host$get${}", class.name, field.name);
        self.declarations.push(extern_function(
            &get,
            "$Foreign$get",
            vec![receiver_ty.clone(), Type::Primitive(PrimitiveType::String)],
            crossing.wire(),
        ));
        let mut statements = Vec::new();
        let target = self.open_receiver(&receiver, module, &mut statements);
        statements.push(bind(
            "r",
            call(
                &get,
                vec![target.clone(), string(&field.name)],
                crossing.wire(),
            ),
        ));
        self.close_receiver(&receiver, &mut statements);
        let value = self.read_result(&crossing, &mut statements);
        statements.push(ret(value));
        let getter_sig = Signature {
            params: vec![],
            result: crossing.clone(),
        };
        let getter = method(
            &field.name,
            self_ty.clone(),
            &getter_sig,
            TypedBlock {
                statements,
                span: span(),
            },
        );

        if !field.writable {
            return Ok(vec![getter]);
        }
        let set = format!("{}$host$set${}", class.name, field.name);
        self.declarations.push(extern_function(
            &set,
            "$Foreign$set",
            vec![
                receiver_ty,
                Type::Primitive(PrimitiveType::String),
                crossing.wire(),
            ],
            unit(),
        ));
        let mut statements = Vec::new();
        let target = self.open_receiver(&receiver, module, &mut statements);
        let (arg, made) = self.cross_in(&crossing, var("p0", crossing.ty()), &mut statements);
        statements.push(statement(call(
            &set,
            vec![target, string(&field.name), arg],
            unit(),
        )));
        if let Some(made) = made {
            statements.push(statement(call(
                "host$free",
                vec![var(&made, Type::Any)],
                unit(),
            )));
        }
        self.close_receiver(&receiver, &mut statements);
        statements.push(ret(None));
        let setter_sig = Signature {
            params: vec![crossing],
            result: Crossing::Void,
        };
        let setter = method(
            &format!("set_{}", field.name),
            self_ty,
            &setter_sig,
            TypedBlock {
                statements,
                span: span(),
            },
        );
        Ok(vec![getter, setter])
    }

    fn signature(
        &self,
        module: &HostModule,
        m: &HostMethod,
        what: &str,
    ) -> RuntimeResult<Signature> {
        if m.params.len() > MAX_ARITY {
            return Err(RuntimeError::Execution(format!(
                "host member `{what}` takes {} arguments, more than the {MAX_ARITY} a host call carries",
                m.params.len()
            )));
        }
        let params = m
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let crossing = self.crossing(module, p, &format!("argument {i} of `{what}`"))?;
                if matches!(crossing, Crossing::Void) {
                    return Err(RuntimeError::Execution(format!(
                        "argument {i} of host member `{what}` is declared void"
                    )));
                }
                Ok(crossing)
            })
            .collect::<RuntimeResult<Vec<_>>>()?;
        let result = self.crossing(module, &m.ret, &format!("the result of `{what}`"))?;
        Ok(Signature { params, result })
    }

    fn crossing(&self, module: &HostModule, ty: &HostType, what: &str) -> RuntimeResult<Crossing> {
        let prim = Type::Primitive;
        Ok(match ty {
            HostType::Void => Crossing::Void,
            HostType::Bool => Crossing::Boxed {
                ty: prim(PrimitiveType::Bool),
                make: "host$box_bool",
                read: "host$read_bool",
                owns: false,
            },
            HostType::Int => Crossing::Boxed {
                ty: prim(PrimitiveType::I64),
                make: "host$box_i64",
                read: "host$read_i64",
                owns: false,
            },
            HostType::Float => Crossing::Boxed {
                ty: prim(PrimitiveType::F64),
                make: "host$box_f64",
                read: "host$read_f64",
                owns: false,
            },
            HostType::Str => Crossing::Boxed {
                ty: prim(PrimitiveType::String),
                make: "host$box_str",
                read: "host$read_str",
                owns: true,
            },
            // A class of this module is its own type; any other host
            // object is a dynamic value.
            HostType::Object(type_name) => match module
                .classes
                .iter()
                .find(|c| &c.type_name == type_name || &c.name == type_name)
            {
                Some(c) if c.word => Crossing::Word(class_type(&c.name)),
                Some(c) => Crossing::Box(class_type(&c.name)),
                None => Crossing::Box(Type::Any),
            },
            HostType::Dynamic => Crossing::Box(Type::Any),
            HostType::Bytes | HostType::Function { .. } => {
                return Err(RuntimeError::Execution(format!(
                    "{what} in host module `{}` is a {} type, which an import does not carry yet",
                    module.name,
                    if matches!(ty, HostType::Bytes) {
                        "bytes"
                    } else {
                        "function"
                    }
                )));
            }
        })
    }

    fn fresh(&mut self) -> String {
        self.fresh += 1;
        format!("h{}", self.fresh)
    }

    /// Bind what the call is made on, returning the expression naming it.
    fn open_receiver(
        &mut self,
        receiver: &Receiver,
        module: &HostModule,
        statements: &mut Vec<TypedNode<TypedStatement>>,
    ) -> TypedNode<TypedExpression> {
        match receiver {
            Receiver::Instance(ty) => var("self", ty.clone()),
            Receiver::Boxing(ty) => {
                statements.push(bind(
                    "c",
                    call(
                        "host$box_word",
                        vec![as_word(var("self", ty.clone()))],
                        Type::Any,
                    ),
                ));
                var("c", Type::Any)
            }
            Receiver::Class(class) => {
                statements.push(bind(
                    "c",
                    call(&format!("{class}$host$class"), vec![], Type::Any),
                ));
                var("c", Type::Any)
            }
            Receiver::Module => {
                statements.push(bind(
                    "c",
                    call("host$import", vec![string(&module.name)], Type::Any),
                ));
                var("c", Type::Any)
            }
        }
    }

    fn close_receiver(&self, receiver: &Receiver, statements: &mut Vec<TypedNode<TypedStatement>>) {
        if !matches!(receiver, Receiver::Instance(_)) {
            statements.push(statement(call(
                "host$free",
                vec![var("c", Type::Any)],
                unit(),
            )));
        }
    }

    /// The argument the protocol takes for `value`, and the box made for
    /// it, which the caller frees once the call is done.
    fn cross_in(
        &mut self,
        crossing: &Crossing,
        value: TypedNode<TypedExpression>,
        statements: &mut Vec<TypedNode<TypedStatement>>,
    ) -> (TypedNode<TypedExpression>, Option<String>) {
        match crossing {
            Crossing::Boxed { make, ty, .. } => {
                let value = if *ty == Type::Primitive(PrimitiveType::Bool) {
                    node(
                        TypedExpression::Cast(TypedCast {
                            expr: Box::new(value),
                            target_type: Type::Primitive(PrimitiveType::I32),
                        }),
                        Type::Primitive(PrimitiveType::I32),
                    )
                } else {
                    value
                };
                let name = self.fresh();
                statements.push(bind(&name, call(make, vec![value], Type::Any)));
                (var(&name, Type::Any), Some(name))
            }
            Crossing::Word(_) => {
                let name = self.fresh();
                statements.push(bind(
                    &name,
                    call("host$box_word", vec![as_word(value)], Type::Any),
                ));
                (var(&name, Type::Any), Some(name))
            }
            Crossing::Box(_) | Crossing::Void => (value, None),
        }
    }

    /// The value the member returns, read out of the result `r`.
    fn read_result(
        &self,
        crossing: &Crossing,
        statements: &mut Vec<TypedNode<TypedStatement>>,
    ) -> Option<TypedNode<TypedExpression>> {
        match crossing {
            Crossing::Void => {
                statements.push(statement(call(
                    "host$free",
                    vec![var("r", Type::Any)],
                    unit(),
                )));
                None
            }
            Crossing::Box(ty) => Some(var("r", ty.clone())),
            Crossing::Word(ty) => {
                statements.push(bind(
                    "w",
                    call(
                        "host$take_word",
                        vec![var("r", Type::Any)],
                        Type::Primitive(PrimitiveType::I64),
                    ),
                ));
                statements.push(statement(call(
                    "host$free",
                    vec![var("r", Type::Any)],
                    unit(),
                )));
                Some(cast_to(
                    var("w", Type::Primitive(PrimitiveType::I64)),
                    ty.clone(),
                ))
            }
            Crossing::Boxed { ty, read, owns, .. } => {
                let read_ty = if *ty == Type::Primitive(PrimitiveType::Bool) {
                    Type::Primitive(PrimitiveType::I32)
                } else {
                    ty.clone()
                };
                statements.push(bind(
                    "v",
                    call(read, vec![var("r", Type::Any)], read_ty.clone()),
                ));
                // A string read out of its box still lives in it.
                if !owns {
                    statements.push(statement(call(
                        "host$free",
                        vec![var("r", Type::Any)],
                        unit(),
                    )));
                }
                let value = var("v", read_ty.clone());
                Some(if read_ty != *ty {
                    node(
                        TypedExpression::Cast(TypedCast {
                            expr: Box::new(value),
                            target_type: ty.clone(),
                        }),
                        ty.clone(),
                    )
                } else {
                    value
                })
            }
        }
    }

    /// The view type reading one `prim` in place.
    fn view(&mut self, prim: PrimitiveType) -> Type {
        let name = format!("host$at${prim:?}");
        let id = match self.views.iter().find(|(n, _, _)| *n == name) {
            Some((_, id, _)) => *id,
            None => {
                let id = TypeId::next();
                self.views.push((name, id, Type::Primitive(prim)));
                id
            }
        };
        Type::Named {
            id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: NullabilityKind::NonNull,
        }
    }

    /// The `prim` at the address `addr`, an i64, as a place to read or
    /// assign.
    fn at(
        &mut self,
        addr: TypedNode<TypedExpression>,
        prim: PrimitiveType,
    ) -> TypedNode<TypedExpression> {
        let view = self.view(prim);
        node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(cast_to(addr, view)),
                field: s("v"),
            }),
            Type::Primitive(prim),
        )
    }

    /// The word of a host object `value`: itself where its class is held
    /// as the word, else the word its box carries.
    fn word_of(
        &self,
        value: TypedNode<TypedExpression>,
        held_as_word: bool,
    ) -> TypedNode<TypedExpression> {
        if held_as_word {
            as_word(value)
        } else {
            call(
                "host$read_word",
                vec![value],
                Type::Primitive(PrimitiveType::I64),
            )
        }
    }

    /// What reaches a native function for an object word.
    fn pass(
        &mut self,
        word: TypedNode<TypedExpression>,
        pass: NativePass,
    ) -> TypedNode<TypedExpression> {
        match pass {
            NativePass::Word => word,
            NativePass::Indirect(k) => self.at(add(word, k as i64), PrimitiveType::I64),
        }
    }

    fn declare_native_helpers(&mut self) {
        if !self.native_helpers {
            self.native_helpers = true;
            for (name, link, params, result) in native_helpers() {
                self.declarations
                    .push(extern_function(name, link, params, result));
            }
        }
    }

    /// The statements that leave an error a native call reported pending
    /// for the program: one load and a branch not taken when none is.
    fn check_pending(&mut self, statements: &mut Vec<TypedNode<TypedStatement>>) {
        let flag = self.fresh();
        statements.push(bind(
            &flag,
            call(
                "host$pending_flag",
                vec![],
                Type::Primitive(PrimitiveType::I64),
            ),
        ));
        let set = self.at(
            var(&flag, Type::Primitive(PrimitiveType::I64)),
            PrimitiveType::U8,
        );
        let condition = node(
            TypedExpression::Binary(TypedBinary {
                op: BinaryOp::Ne,
                left: Box::new(set),
                right: Box::new(cast_to(
                    node(
                        TypedExpression::Literal(TypedLiteral::Integer(0)),
                        Type::Primitive(PrimitiveType::I64),
                    ),
                    Type::Primitive(PrimitiveType::U8),
                )),
            }),
            Type::Primitive(PrimitiveType::Bool),
        );
        statements.push(node(
            TypedStatement::If(TypedIf {
                condition: Box::new(condition),
                then_block: TypedBlock {
                    statements: vec![statement(call("host$raise_pending", vec![], unit()))],
                    span: span(),
                },
                else_block: None,
                span: span(),
            }),
            unit(),
        ));
    }

    /// A native result `r` of `ty`, as the member's result crossing.
    fn from_native(
        &mut self,
        r: TypedNode<TypedExpression>,
        ty: &NativeType,
        result: &Crossing,
    ) -> TypedNode<TypedExpression> {
        match (ty, result) {
            (NativeType::Str, _) => call(
                "host$text_to_string",
                vec![r],
                Type::Primitive(PrimitiveType::String),
            ),
            (NativeType::Object { .. }, Crossing::Box(class)) => {
                // A class held boxed takes the word into a box of its own.
                cast_to(call("host$box_word", vec![r], Type::Any), class.clone())
            }
            (_, crossing) => cast_to(r, crossing.ty()),
        }
    }

    /// The body of a member bound to a native function: one direct call,
    /// its operands passed as the binding says.
    fn native_body(
        &mut self,
        binding: &NativeBinding,
        signature: &Signature,
        receiver: Option<(&Type, bool)>,
        what: &str,
    ) -> RuntimeResult<TypedBlock> {
        if binding.params.len() != signature.params.len() {
            return Err(RuntimeError::Execution(format!(
                "host member `{what}` takes {} arguments, and its native binding `{}` {}",
                signature.params.len(),
                binding.symbol,
                binding.params.len()
            )));
        }
        self.declare_native_helpers();
        if binding.address != 0 {
            zyntax_compiler::late_symbols::register(&binding.symbol, binding.address as *const u8);
        }
        let i64t = Type::Primitive(PrimitiveType::I64);
        let mut operand_types = Vec::new();
        let mut args = Vec::new();
        if let Some(pass) = binding.receiver {
            let Some((self_ty, held_as_word)) = receiver else {
                return Err(RuntimeError::Execution(format!(
                    "`{what}` has no receiver, and its native binding `{}` takes one",
                    binding.symbol
                )));
            };
            let word = self.word_of(var("self", self_ty.clone()), held_as_word);
            args.push(self.pass(word, pass));
            operand_types.push(i64t.clone());
        }
        for (i, (ty, crossing)) in binding.params.iter().zip(&signature.params).enumerate() {
            let value = var(&format!("p{i}"), crossing.ty());
            match ty {
                NativeType::Void => {
                    return Err(RuntimeError::Execution(format!(
                        "argument {i} of `{what}` is bound as void"
                    )));
                }
                NativeType::Str => {
                    // The bytes after the string's header, and their count.
                    let word = self.fresh();
                    self.pending_binds.push(bind(&word, as_word(value)));
                    args.push(add(var(&word, i64t.clone()), 16));
                    let len = self.at(var(&word, i64t.clone()), PrimitiveType::U32);
                    args.push(cast_to(len, i64t.clone()));
                    operand_types.push(i64t.clone());
                    operand_types.push(i64t.clone());
                }
                NativeType::Object { pass, .. } => {
                    let word = self.word_of(value, matches!(crossing, Crossing::Word(_)));
                    args.push(self.pass(word, *pass));
                    operand_types.push(i64t.clone());
                }
                other => {
                    let prim = native_prim(other).expect("a value type");
                    args.push(cast_to(value, Type::Primitive(prim)));
                    operand_types.push(Type::Primitive(prim));
                }
            }
        }
        let ret_ty = native_prim(&binding.ret)
            .map(Type::Primitive)
            .unwrap_or_else(unit);
        let callee = format!("native${}", binding.symbol);
        if !self.natives.contains(&callee) {
            self.natives.insert(callee.clone());
            self.declarations.push(extern_function(
                &callee,
                &binding.symbol,
                operand_types,
                ret_ty.clone(),
            ));
        }
        let mut statements = std::mem::take(&mut self.pending_binds);
        let value = if binding.ret == NativeType::Void {
            statements.push(statement(call(&callee, args, unit())));
            None
        } else {
            statements.push(bind("r", call(&callee, args, ret_ty.clone())));
            Some(())
        };
        if binding.may_raise {
            self.check_pending(&mut statements);
        }
        let result =
            value.map(|()| self.from_native(var("r", ret_ty), &binding.ret, &signature.result));
        statements.push(ret(result));
        Ok(TypedBlock {
            statements,
            span: span(),
        })
    }

    /// `x()` and, for a writable field, `set_x(v)`, for a field at a fixed
    /// place in the object: a load and a store, no call.
    fn native_field(
        &mut self,
        class: &HostClass,
        field: &HostField,
        crossing: Crossing,
    ) -> RuntimeResult<Vec<TypedMethod>> {
        let native = field.native.as_ref().expect("a native field");
        let what = format!("field {}.{}", class.name, field.name);
        if field.is_static {
            return Err(RuntimeError::Execution(format!(
                "{what} is static, and a static field is not bound in place"
            )));
        }
        let this = class_type(&class.name);
        let prim = native_prim(&native.ty)
            .ok_or_else(|| RuntimeError::Execution(format!("{what} is bound as void")))?;
        let word = self.word_of(var("self", this.clone()), class.word);
        let payload = self.pass(word, native.pass);
        let place = self.at(add(payload, native.offset as i64), prim);
        if matches!(native.ty, NativeType::Str) {
            self.declare_native_helpers();
        }
        let value = self.from_native(place.clone(), &native.ty, &crossing);
        let getter = method(
            &field.name,
            Some(this.clone()),
            &Signature {
                params: vec![],
                result: crossing.clone(),
            },
            TypedBlock {
                statements: vec![ret(Some(value))],
                span: span(),
            },
        );
        if !field.writable {
            return Ok(vec![getter]);
        }
        if matches!(native.ty, NativeType::Str | NativeType::Object { .. }) {
            return Err(RuntimeError::Execution(format!(
                "{what} is writable and holds a {}, which is not stored in place yet",
                if matches!(native.ty, NativeType::Str) {
                    "string"
                } else {
                    "object"
                }
            )));
        }
        let assign = node(
            TypedExpression::Binary(TypedBinary {
                op: BinaryOp::Assign,
                left: Box::new(place),
                right: Box::new(cast_to(var("p0", crossing.ty()), Type::Primitive(prim))),
            }),
            Type::Primitive(prim),
        );
        let setter = method(
            &format!("set_{}", field.name),
            Some(this),
            &Signature {
                params: vec![crossing],
                result: Crossing::Void,
            },
            TypedBlock {
                statements: vec![statement(assign), ret(None)],
                span: span(),
            },
        );
        Ok(vec![getter, setter])
    }

    /// The body of a member that calls `entry{n}` on its receiver with
    /// its parameters `p0..`, named `name` (a constructor passes none).
    fn member_body(
        &mut self,
        module: &HostModule,
        receiver: &Receiver,
        wire: &str,
        entry: &str,
        name: &str,
        signature: &Signature,
    ) -> TypedBlock {
        let receiver_ty = match receiver {
            Receiver::Instance(ty) => ty.clone(),
            _ => Type::Any,
        };
        let named = !name.is_empty();
        let mut wire_params = vec![receiver_ty];
        if named {
            wire_params.push(Type::Primitive(PrimitiveType::String));
        }
        wire_params.extend(signature.params.iter().map(|p| match p {
            Crossing::Box(ty) => ty.clone(),
            _ => Type::Any,
        }));
        self.declarations.push(extern_function(
            wire,
            &format!("{entry}{}", signature.params.len()),
            wire_params,
            signature.result.wire(),
        ));

        let mut statements = Vec::new();
        let target = self.open_receiver(receiver, module, &mut statements);
        let mut args = vec![target];
        if named {
            args.push(string(name));
        }
        let mut made = Vec::new();
        for (i, crossing) in signature.params.iter().enumerate() {
            let (arg, boxed) = self.cross_in(
                crossing,
                var(&format!("p{i}"), crossing.ty()),
                &mut statements,
            );
            args.push(arg);
            made.extend(boxed);
        }
        statements.push(bind("r", call(wire, args, signature.result.wire())));
        for boxed in made {
            statements.push(statement(call(
                "host$free",
                vec![var(&boxed, Type::Any)],
                unit(),
            )));
        }
        self.close_receiver(receiver, &mut statements);
        let value = self.read_result(&signature.result, &mut statements);
        statements.push(ret(value));
        TypedBlock {
            statements,
            span: span(),
        }
    }
}

fn method(
    name: &str,
    self_ty: Option<Type>,
    signature: &Signature,
    body: TypedBlock,
) -> TypedMethod {
    let mut params = Vec::new();
    let is_static = self_ty.is_none();
    if let Some(ty) = self_ty {
        params.push(param("self", ty, true));
    }
    for (i, p) in signature.params.iter().enumerate() {
        params.push(param(&format!("p{i}"), p.ty(), false));
    }
    TypedMethod {
        name: s(name),
        annotations: vec![],
        type_params: vec![],
        params,
        return_type: signature.result.ty(),
        body: Some(body),
        visibility: Visibility::Public,
        is_static,
        is_async: false,
        is_override: false,
        span: span(),
    }
}

fn param(name: &str, ty: Type, is_self: bool) -> TypedMethodParam {
    TypedMethodParam {
        name: s(name),
        ty,
        mutability: Mutability::Immutable,
        is_self,
        kind: ParameterKind::Regular,
        default_value: None,
        attributes: vec![],
        span: span(),
        ownership: Default::default(),
    }
}
