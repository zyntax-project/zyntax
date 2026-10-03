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

use zyntax_typed_ast::type_registry::{Mutability, PrimitiveType, Type, Visibility};
use zyntax_typed_ast::typed_ast::{
    ParameterKind, TypedBlock, TypedCall, TypedCast, TypedDeclaration, TypedExpression,
    TypedExtern, TypedExternStruct, TypedFunction, TypedImport, TypedImportItem, TypedLet,
    TypedLiteral, TypedMethod, TypedMethodParam, TypedNode, TypedParameter, TypedStatement,
    TypedTraitImpl,
};
use zyntax_typed_ast::{InternedString, Span, TypedProgram};

use crate::host::{HostClass, HostField, HostMethod, HostModule, HostType};
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
    Void,
}

impl Crossing {
    fn ty(&self) -> Type {
        match self {
            Crossing::Boxed { ty, .. } | Crossing::Box(ty) => ty.clone(),
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
    ]
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
            let body = self.member_body(
                module,
                &Receiver::Module,
                &format!("{}$host", function.name),
                "$Foreign$invoke",
                &function.name,
                &signature,
            );
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
                result: Crossing::Box(this.clone()),
            };
            let body = self.member_body(
                module,
                &Receiver::Class(class.name.clone()),
                &format!("{}$host$new", class.name),
                "$Foreign$construct",
                "",
                &signature,
            );
            methods.push(method("new", None, &signature, body));
        }
        for m in &class.methods {
            let signature = self.signature(module, m, &format!("{}.{}", class.name, m.name))?;
            let receiver = if m.is_static {
                Receiver::Class(class.name.clone())
            } else {
                Receiver::Instance(this.clone())
            };
            let body = self.member_body(
                module,
                &receiver,
                &format!("{}$host${}", class.name, m.name),
                "$Foreign$invoke",
                &m.name,
                &signature,
            );
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
        let (receiver, receiver_ty) = if field.is_static {
            (Receiver::Class(class.name.clone()), Type::Any)
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
            HostType::Object(type_name) => Crossing::Box(
                module
                    .classes
                    .iter()
                    .find(|c| &c.type_name == type_name || &c.name == type_name)
                    .map(|c| class_type(&c.name))
                    .unwrap_or(Type::Any),
            ),
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
