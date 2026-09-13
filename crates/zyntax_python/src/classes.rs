//! Classes as structs. An instance is a heap struct whose first field is
//! the class's index; a method is a function taking the instance first;
//! a method some subclass overrides is called through a dispatcher that
//! switches on that index. Attribute access and method calls on a
//! dynamic receiver go through dispatchers generated per name over every
//! class that has it.

use crate::lower::{
    self, binary, call, callm_name, cast, dispatch_name, field_storage, getattr_name, int_lit, ir,
    new_name, node, setattr_name, str_lit, var, without_self, Lowerer, Node, Val,
};
use crate::scope::Scope;
use crate::types::{method_fn, ClassInfo, Locals, Module, Sig, Ty};
use crate::{intern, Error, Result};
use ruff_python_ast as py;
use ruff_text_size::Ranged;
use std::collections::HashMap;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{FieldDef, TypeMetadata};
use zyntax_typed_ast::typed_ast::{
    TypedAnnotation, TypedBlock, TypedClass, TypedDeclaration, TypedField, TypedFieldAccess,
    TypedFieldInit, TypedFunction, TypedIf, TypedLet, TypedLiteral, TypedParameter, TypedStatement,
    TypedStructLiteral,
};
use zyntax_typed_ast::{
    Mutability, ParamOwnership, ParameterKind, Type, TypeRegistry, TypedNode, Visibility,
};

/// A class as written: its name, base and methods.
pub(crate) struct ClassDef<'a> {
    pub(crate) name: String,
    pub(crate) base: Option<String>,
    pub(crate) methods: Vec<&'a py::StmtFunctionDef>,
    /// Where the class is written, for what is reported about it, and
    /// in which of the program's modules (`None` for the main file).
    pub(crate) range: ruff_text_size::TextRange,
    pub(crate) module: Option<String>,
}

/// The classes of a module body, in order. `origins` names the module
/// each statement of `body` came from.
pub(crate) fn collect<'a>(
    body: &'a [py::Stmt],
    origins: &[Option<String>],
) -> Result<Vec<ClassDef<'a>>> {
    let mut out = Vec::new();
    for (i, stmt) in body.iter().enumerate() {
        let py::Stmt::ClassDef(c) = stmt else {
            continue;
        };
        let module = origins.get(i).cloned().flatten();
        let located = |e: Error| match &module {
            Some(m) => e.in_module(m),
            None => e,
        };
        if !c.decorator_list.is_empty() {
            return Err(located(Error::unsupported("class decorators", c)));
        }
        let mut base = None;
        if let Some(args) = &c.arguments {
            if !args.keywords.is_empty() || args.args.len() > 1 {
                return Err(located(Error::unsupported(
                    "more than one base class or class keywords",
                    c,
                )));
            }
            if let Some(b) = args.args.first() {
                let py::Expr::Name(n) = b else {
                    return Err(located(Error::unsupported(
                        "a base class that is not a name",
                        b,
                    )));
                };
                if n.id.as_str() != "object" {
                    base = Some(n.id.to_string());
                }
            }
        }
        let mut methods = Vec::new();
        for s in &c.body {
            match s {
                py::Stmt::FunctionDef(f) => methods.push(f),
                py::Stmt::Pass(_) => {}
                py::Stmt::Expr(e) if matches!(*e.value, py::Expr::StringLiteral(_)) => {}
                other => {
                    return Err(located(Error::unsupported(
                        "a class body statement other than a method",
                        other,
                    )))
                }
            }
        }
        out.push(ClassDef {
            range: c.range(),
            module: module.clone(),
            name: c.name.to_string(),
            base,
            methods,
        });
    }
    Ok(out)
}

/// The classes as the module first knows them: names, bases, methods,
/// and the class tag as the only field.
pub(crate) fn skeletons(defs: &[ClassDef<'_>]) -> Result<(Vec<ClassInfo>, HashMap<String, usize>)> {
    let mut classes = Vec::new();
    let mut index = HashMap::new();
    for def in defs {
        let base = match &def.base {
            Some(b) => Some(*index.get(b).ok_or_else(|| {
                let e = Error::unsupported(
                    format!(
                        "class {} deriving from `{b}`, which is not a class defined before it",
                        def.name
                    ),
                    &def.range,
                );
                match &def.module {
                    Some(m) => e.in_module(m),
                    None => e,
                }
            })?),
            None => None,
        };
        index.insert(def.name.clone(), classes.len());
        classes.push(ClassInfo {
            name: def.name.clone(),
            base,
            fields: vec![("$class".to_string(), Ty::Int)],
            methods: def.methods.iter().map(|m| m.name.to_string()).collect(),
            type_id: None,
        });
    }
    Ok((classes, index))
}

/// Register each class's struct with the layout inference settled on.
pub(crate) fn register(
    module: &mut Module,
    registry: &mut TypeRegistry,
) -> Vec<TypedNode<TypedDeclaration>> {
    let span = Span::new(0, 0);
    let mut decls = Vec::new();
    let mut ids = Vec::new();
    for class in &mut module.classes {
        let fields: Vec<FieldDef> = class
            .fields
            .iter()
            .map(|(name, ty)| FieldDef {
                name: intern(name),
                ty: ir(field_storage(*ty)),
                visibility: Visibility::Public,
                mutability: Mutability::Mutable,
                is_static: false,
                span,
                getter: None,
                setter: None,
                is_synthetic: false,
            })
            .collect();
        let id = registry.register_struct_type(
            intern(&class.name),
            Vec::new(),
            fields,
            Vec::new(),
            Vec::new(),
            TypeMetadata {
                is_reference: true,
                ..Default::default()
            },
            span,
        );
        class.type_id = Some(id);
        ids.push(id);
        decls.push(TypedNode::new(
            TypedDeclaration::Class(TypedClass {
                name: intern(&class.name),
                type_params: Vec::new(),
                extends: None,
                implements: Vec::new(),
                fields: class
                    .fields
                    .iter()
                    .map(|(name, ty)| TypedField {
                        name: intern(name),
                        ty: ir(field_storage(*ty)),
                        initializer: None,
                        visibility: Visibility::Public,
                        mutability: Mutability::Mutable,
                        is_static: false,
                        span,
                    })
                    .collect(),
                methods: Vec::new(),
                constructors: Vec::new(),
                visibility: Visibility::Public,
                is_abstract: false,
                is_final: false,
                // An instance is a heap object every holder shares.
                annotations: vec![TypedAnnotation {
                    name: intern("reference"),
                    args: Vec::new(),
                    span,
                }],
                span,
            }),
            Type::Unknown,
            span,
        ));
    }
    lower::set_class_types(ids);
    decls
}

fn stmt(e: Node, span: Span) -> TypedNode<TypedStatement> {
    TypedNode::new(TypedStatement::Expression(Box::new(e)), Type::Unknown, span)
}
fn ret(e: Node, span: Span) -> TypedNode<TypedStatement> {
    TypedNode::new(
        TypedStatement::Return(Some(Box::new(e))),
        Type::Unknown,
        span,
    )
}
fn let_(name: &str, ty: Ty, value: Node, span: Span) -> TypedNode<TypedStatement> {
    TypedNode::new(
        TypedStatement::Let(TypedLet {
            name: intern(name),
            ty: ir(ty),
            mutability: Mutability::Mutable,
            initializer: Some(Box::new(value)),
            span,
        }),
        Type::Unknown,
        span,
    )
}
fn when(cond: Node, then: Vec<TypedNode<TypedStatement>>, span: Span) -> TypedNode<TypedStatement> {
    TypedNode::new(
        TypedStatement::If(TypedIf {
            condition: Box::new(cond),
            then_block: TypedBlock {
                statements: then,
                span,
            },
            else_block: None,
            span,
        }),
        Type::Unknown,
        span,
    )
}
fn param(name: &str, ty: Ty, span: Span) -> TypedParameter {
    TypedParameter {
        name: intern(name),
        ty: ir(ty),
        mutability: Mutability::Mutable,
        kind: ParameterKind::Regular,
        default_value: None,
        attributes: lower::dynamic_attribute(ty, span),
        ownership: match ty {
            Ty::Int | Ty::Float | Ty::Bool | Ty::None => ParamOwnership::Copied,
            _ => ParamOwnership::Owned,
        },
        span,
    }
}
fn function(
    name: &str,
    params: Vec<TypedParameter>,
    ret_ty: Ty,
    statements: Vec<TypedNode<TypedStatement>>,
    span: Span,
) -> TypedFunction {
    TypedFunction {
        name: intern(name),
        annotations: Vec::new(),
        effects: Vec::new(),
        with_handlers: Vec::new(),
        type_params: Vec::new(),
        params,
        return_type: ir(ret_ty),
        body: Some(TypedBlock { statements, span }),
        visibility: Visibility::Public,
        is_async: false,
        is_fiber: false,
        is_pure: false,
        is_external: false,
        calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
        link_name: None,
        module: None,
    }
}
fn field(object: Node, name: &str, ty: Ty, span: Span) -> Node {
    node(
        TypedExpression::Field(TypedFieldAccess {
            object: Box::new(object),
            field: intern(name),
        }),
        field_storage(ty),
        span,
    )
}
fn none(span: Span) -> Val {
    Val {
        node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
        ty: Ty::None,
    }
}
fn scratch(module: &Module) -> Lowerer<'_> {
    let sig = Sig {
        params: Vec::new(),
        ret: Ty::None,
        defaults: Vec::new(),
    };
    let mut lowerer = Lowerer::new(
        module,
        "$class",
        sig,
        Locals::default(),
        &Scope::default(),
        Vec::new(),
        HashMap::new(),
    );
    lowerer.guards = false;
    lowerer
}

use zyntax_typed_ast::typed_ast::TypedExpression;

/// The functions every class needs: construction, unboxing, dispatch.
pub(crate) fn generated(module: &Module) -> Vec<TypedFunction> {
    let span = Span::new(0, 0);
    let mut out = Vec::new();
    for (k, class) in module.classes.iter().enumerate() {
        out.push(constructor(module, k, span));
        out.push(unboxer(module, k, span));
        for method in &class.methods {
            let fn_name = method_fn(&class.name, method);
            if !module.overriders(k, method).is_empty() {
                out.push(dispatcher(module, k, method, &fn_name, span));
            }
        }
    }
    for attr in module.attr_reads.borrow().iter() {
        out.push(getattr(module, attr, span));
    }
    for attr in module.attr_writes.borrow().iter() {
        out.push(setattr(module, attr, span));
    }
    for (method, arity) in module.dyn_methods.borrow().iter() {
        out.push(dynamic_call(module, method, *arity, span));
    }
    out.extend(hooks(module, span));
    out
}

/// `C$new(args)`: the struct with its tag and zeroed fields, then
/// `__init__` when the class has one.
fn constructor(module: &Module, k: usize, span: Span) -> TypedFunction {
    let class = &module.classes[k];
    let mut lowerer = scratch(module);
    let mut fields = Vec::new();
    for (name, ty) in &class.fields {
        let value = if name == "$class" {
            int_lit(k as i64, span)
        } else {
            match field_storage(*ty) {
                Ty::Int => int_lit(0, span),
                Ty::Float => node(
                    TypedExpression::Literal(TypedLiteral::Float(0.0)),
                    Ty::Float,
                    span,
                ),
                Ty::Bool => node(
                    TypedExpression::Literal(TypedLiteral::Bool(false)),
                    Ty::Bool,
                    span,
                ),
                Ty::Str => str_lit("", span),
                _ => lowerer.coerce(none(span), Ty::Object),
            }
        };
        fields.push(TypedFieldInit {
            name: intern(name),
            value: Box::new(value),
        });
    }
    let literal = node(
        TypedExpression::Struct(TypedStructLiteral {
            name: intern(&class.name),
            fields,
        }),
        Ty::Class(k as u16),
        span,
    );
    let mut statements = vec![let_("obj", Ty::Class(k as u16), literal, span)];
    let mut params = Vec::new();
    if let Some((sig, fn_name)) = module.method_sig(k, "__init__") {
        let owner = module.method_owner(k, "__init__").expect("found");
        let init = without_self(sig);
        let mut args = vec![lowerer.coerce(
            Val {
                node: var(intern("obj"), Ty::Class(k as u16), span),
                ty: Ty::Class(k as u16),
            },
            Ty::Class(owner as u16),
        )];
        for (name, ty) in &init.params {
            params.push(param(name, *ty, span));
            args.push(var(intern(name), *ty, span));
        }
        statements.push(stmt(call(&fn_name, args, sig.ret, span), span));
    }
    statements.push(ret(var(intern("obj"), Ty::Class(k as u16), span), span));
    function(
        &new_name(&class.name),
        params,
        Ty::Class(k as u16),
        statements,
        span,
    )
}

/// `C$unbox(x)`: the address in a box tagged as `C` or a subclass.
fn unboxer(module: &Module, k: usize, span: Span) -> TypedFunction {
    let class = &module.classes[k];
    let x = var(intern("x"), Ty::Object, span);
    let kind = call("zb_any_kind", vec![x.clone()], Ty::Int, span);
    let mut accepted: Option<Node> = None;
    for c in 0..module.classes.len() {
        if module.is_subclass(c, k) {
            let this = binary(
                BinaryOp::Eq,
                kind.clone(),
                int_lit(zyntax_builtins::INSTANCE_KIND_BASE + c as i64, span),
                Ty::Bool,
                span,
            );
            accepted = Some(match accepted {
                None => this,
                Some(prev) => binary(BinaryOp::Or, prev, this, Ty::Bool, span),
            });
        }
    }
    let accepted = accepted.expect("a class accepts itself");
    let not_accepted = node(
        TypedExpression::Unary(zyntax_typed_ast::typed_ast::TypedUnary {
            op: zyntax_typed_ast::UnaryOp::Not,
            operand: Box::new(accepted),
        }),
        Ty::Bool,
        span,
    );
    let message = binary(
        BinaryOp::Add,
        str_lit(&format!("expected {}, got ", class.name), span),
        call("zb_any_type", vec![x.clone()], Ty::Str, span),
        Ty::Str,
        span,
    );
    let statements = vec![
        when(
            not_accepted,
            vec![stmt(
                call(
                    "zb_fatal",
                    vec![str_lit("TypeError", span), message],
                    Ty::None,
                    span,
                ),
                span,
            )],
            span,
        ),
        ret(
            lower::addr_call("zb_unbox_instance_raw", vec![x], span),
            span,
        ),
    ];
    let mut f = function(
        &format!("{}$unbox", class.name),
        vec![param("x", Ty::Object, span)],
        Ty::Int,
        statements,
        span,
    );
    f.return_type = lower::addr_type();
    f
}

/// `Owner$m$dispatch(self, args)`: the override for the instance's
/// class, or the owner's own method.
fn dispatcher(
    module: &Module,
    owner: usize,
    method: &str,
    fn_name: &str,
    span: Span,
) -> TypedFunction {
    let sig = &module.funcs[fn_name];
    let mut lowerer = scratch(module);
    let self_ty = Ty::Class(owner as u16);
    let mut params = vec![param("self", self_ty, span)];
    for (name, ty) in sig.params.iter().skip(1) {
        params.push(param(name, *ty, span));
    }
    let tag = field(var(intern("self"), self_ty, span), "$class", Ty::Int, span);
    let mut statements = vec![let_("tag", Ty::Int, tag, span)];
    for sub in module.overriders(owner, method) {
        let sub_fn = method_fn(&module.classes[sub].name, method);
        let sub_sig = &module.funcs[&sub_fn];
        let mut args = vec![lowerer.coerce(
            Val {
                node: var(intern("self"), self_ty, span),
                ty: self_ty,
            },
            Ty::Class(sub as u16),
        )];
        for ((name, ty), (_, sub_ty)) in
            sig.params.iter().skip(1).zip(sub_sig.params.iter().skip(1))
        {
            args.push(lowerer.coerce(
                Val {
                    node: var(intern(name), *ty, span),
                    ty: *ty,
                },
                *sub_ty,
            ));
        }
        let result = lowerer.coerce(
            Val {
                node: call(&sub_fn, args, sub_sig.ret, span),
                ty: sub_sig.ret,
            },
            sig.ret,
        );
        statements.push(when(
            binary(
                BinaryOp::Eq,
                var(intern("tag"), Ty::Int, span),
                int_lit(sub as i64, span),
                Ty::Bool,
                span,
            ),
            vec![ret(result, span)],
            span,
        ));
    }
    let mut args = vec![var(intern("self"), self_ty, span)];
    for (name, ty) in sig.params.iter().skip(1) {
        args.push(var(intern(name), *ty, span));
    }
    statements.push(ret(call(fn_name, args, sig.ret, span), span));
    function(&dispatch_name(fn_name), params, sig.ret, statements, span)
}

/// The `if kind == c { ... }` chain over the classes `pick` selects,
/// with the instance unboxed as `obj` of that class.
fn per_class(
    module: &Module,
    x: Node,
    pick: impl Fn(usize) -> bool,
    body: impl Fn(&mut Lowerer<'_>, usize, Node) -> Vec<TypedNode<TypedStatement>>,
    span: Span,
) -> Vec<TypedNode<TypedStatement>> {
    let mut lowerer = scratch(module);
    let mut statements = vec![let_(
        "kind",
        Ty::Int,
        call("zb_any_kind", vec![x.clone()], Ty::Int, span),
        span,
    )];
    for c in 0..module.classes.len() {
        if !pick(c) {
            continue;
        }
        let address = lower::addr_call("zb_unbox_instance_raw", vec![x.clone()], span);
        let obj = cast(address, Ty::Class(c as u16), span);
        let mut then = vec![let_("obj", Ty::Class(c as u16), obj, span)];
        then.extend(body(
            &mut lowerer,
            c,
            var(intern("obj"), Ty::Class(c as u16), span),
        ));
        statements.push(when(
            binary(
                BinaryOp::Eq,
                var(intern("kind"), Ty::Int, span),
                int_lit(zyntax_builtins::INSTANCE_KIND_BASE + c as i64, span),
                Ty::Bool,
                span,
            ),
            then,
            span,
        ));
    }
    statements
}

fn attribute_error(x: Node, attr: &str, span: Span) -> TypedNode<TypedStatement> {
    let message = binary(
        BinaryOp::Add,
        binary(
            BinaryOp::Add,
            str_lit("'", span),
            call("zb_any_type", vec![x], Ty::Str, span),
            Ty::Str,
            span,
        ),
        str_lit(&format!("' object has no attribute '{attr}'"), span),
        Ty::Str,
        span,
    );
    stmt(
        call(
            "zb_fatal",
            vec![str_lit("AttributeError", span), message],
            Ty::None,
            span,
        ),
        span,
    )
}

fn getattr(module: &Module, attr: &str, span: Span) -> TypedFunction {
    let x = var(intern("x"), Ty::Object, span);
    let mut statements = per_class(
        module,
        x.clone(),
        |c| module.field(c, attr).is_some(),
        |lowerer, c, obj| {
            let (_, ty) = module.field(c, attr).expect("picked");
            let value = lowerer.coerce(
                Val {
                    node: field(obj, attr, ty, span),
                    ty: field_storage(ty),
                },
                ty,
            );
            let boxed = lowerer.coerce(Val { node: value, ty }, Ty::Object);
            vec![ret(boxed, span)]
        },
        span,
    );
    statements.push(attribute_error(x.clone(), attr, span));
    statements.push(ret(x, span));
    function(
        &getattr_name(attr),
        vec![param("x", Ty::Object, span)],
        Ty::Object,
        statements,
        span,
    )
}

fn setattr(module: &Module, attr: &str, span: Span) -> TypedFunction {
    let x = var(intern("x"), Ty::Object, span);
    let mut statements = per_class(
        module,
        x.clone(),
        |c| module.field(c, attr).is_some(),
        |lowerer, c, obj| {
            let (_, ty) = module.field(c, attr).expect("picked");
            let value = lowerer.coerce(
                Val {
                    node: var(intern("v"), Ty::Object, span),
                    ty: Ty::Object,
                },
                ty,
            );
            let stored = lowerer.coerce(Val { node: value, ty }, field_storage(ty));
            vec![
                stmt(
                    binary(
                        BinaryOp::Assign,
                        field(obj, attr, ty, span),
                        stored,
                        Ty::None,
                        span,
                    ),
                    span,
                ),
                TypedNode::new(TypedStatement::Return(None), Type::Unknown, span),
            ]
        },
        span,
    );
    statements.push(attribute_error(x, attr, span));
    function(
        &setattr_name(attr),
        vec![param("x", Ty::Object, span), param("v", Ty::Object, span)],
        Ty::None,
        statements,
        span,
    )
}

fn dynamic_call(module: &Module, method: &str, arity: usize, span: Span) -> TypedFunction {
    let x = var(intern("x"), Ty::Object, span);
    let mut params = vec![param("x", Ty::Object, span)];
    for i in 0..arity {
        params.push(param(&format!("a{i}"), Ty::Object, span));
    }
    let mut statements = per_class(
        module,
        x.clone(),
        |c| {
            module
                .method_sig(c, method)
                .is_some_and(|(sig, _)| sig.params.len() == arity + 1)
        },
        |lowerer, c, obj| {
            let (sig, _) = module.method_sig(c, method).expect("picked");
            let args: Vec<Node> = sig
                .params
                .iter()
                .skip(1)
                .enumerate()
                .map(|(i, (_, ty))| {
                    lowerer.coerce(
                        Val {
                            node: var(intern(&format!("a{i}")), Ty::Object, span),
                            ty: Ty::Object,
                        },
                        *ty,
                    )
                })
                .collect();
            let result = lowerer.invoke(c, method, obj, args, span).expect("picked");
            let boxed = lowerer.coerce(result, Ty::Object);
            vec![ret(boxed, span)]
        },
        span,
    );
    statements.extend(builtin_arms(module, method, arity, x.clone(), span));
    statements.push(attribute_error(x.clone(), method, span));
    statements.push(ret(x, span));
    function(
        &callm_name(method, arity),
        params,
        Ty::Object,
        statements,
        span,
    )
}

/// The same method on a boxed string, list, tuple, dict or set: one
/// arm per kind the typed lowering has the method for, found by lowering
/// `s.method(a0, ...)` with `s` of that type. A kind without it gets no
/// arm and falls through to the attribute error.
fn builtin_arms(
    module: &Module,
    method: &str,
    arity: usize,
    x: Node,
    span: Span,
) -> Vec<TypedNode<TypedStatement>> {
    use crate::types::Elem;
    let args: Vec<String> = (0..arity).map(|i| format!("a{i}")).collect();
    let source = format!("s.{method}({})", args.join(", "));
    let Ok(parsed) = ruff_python_parser::parse_expression(&source) else {
        return Vec::new();
    };
    let expr = parsed.into_syntax().body;
    let category = |c: i64| {
        binary(
            BinaryOp::Eq,
            call("zb_any_category", vec![x.clone()], Ty::Int, span),
            int_lit(c, span),
            Ty::Bool,
            span,
        )
    };
    let kind = |k: i64| {
        binary(
            BinaryOp::Eq,
            var(intern("kind"), Ty::Int, span),
            int_lit(k, span),
            Ty::Bool,
            span,
        )
    };
    let list_kind = |k: zyntax_builtins::Kind| kind(k.list_tag() >> 8);
    let receivers: [(Node, Ty); 8] = [
        (category(5), Ty::Str),
        (list_kind(zyntax_builtins::Kind::Int), Ty::List(Elem::Int)),
        (
            list_kind(zyntax_builtins::Kind::Float),
            Ty::List(Elem::Float),
        ),
        (list_kind(zyntax_builtins::Kind::Str), Ty::List(Elem::Str)),
        (
            list_kind(zyntax_builtins::Kind::Any),
            Ty::List(Elem::Object),
        ),
        (kind(zyntax_builtins::TUPLE_TAG >> 8), Ty::Tuple),
        (kind(zyntax_builtins::DICT_TAG >> 8), Ty::Dict),
        (kind(zyntax_builtins::SET_TAG >> 8), Ty::Set),
    ];
    let mut arms = Vec::new();
    for (test, ty) in receivers {
        let mut vars: Vec<(&str, Ty)> = vec![("s", ty)];
        for a in &args {
            vars.push((a.as_str(), Ty::Object));
        }
        let mut lowerer = scratch_with(module, &vars);
        // The receiver is what the arm's test says it is; the method
        // then runs on it, and may leave statements to run ahead of it.
        let receiver = lowerer.trusted(
            Val {
                node: x.clone(),
                ty: Ty::Object,
            },
            ty,
        );
        let mut then = std::mem::take(&mut lowerer.hoisted);
        then.push(let_("s", ty, receiver, span));
        let Ok(value) = lowerer.expr(&expr) else {
            continue;
        };
        let boxed = lowerer.coerce(value, Ty::Object);
        then.extend(std::mem::take(&mut lowerer.hoisted));
        then.push(ret(boxed, span));
        arms.push(when(test, then, span));
    }
    arms
}

fn scratch_with<'m>(module: &'m Module, vars: &[(&str, Ty)]) -> Lowerer<'m> {
    let sig = Sig {
        params: Vec::new(),
        ret: Ty::Object,
        defaults: Vec::new(),
    };
    let mut locals = Locals::default();
    for (name, ty) in vars {
        locals.vars.insert(name.to_string(), *ty);
    }
    Lowerer::new(
        module,
        "$class",
        sig,
        locals,
        &Scope::default(),
        Vec::new(),
        HashMap::new(),
    )
}

/// What the library asks of instances it only sees boxed.
fn hooks(module: &Module, span: Span) -> Vec<TypedFunction> {
    let x = var(intern("x"), Ty::Object, span);
    // str(): `__str__`, `__repr__`, or `<C object>`.
    let mut statements = per_class(
        module,
        x.clone(),
        |_| true,
        |lowerer, c, obj| {
            let text = lowerer.str_of(Val {
                node: obj,
                ty: Ty::Class(c as u16),
            });
            vec![ret(text, span)]
        },
        span,
    );
    statements.push(ret(str_lit("<object>", span), span));
    let str_hook = function(
        "zb_hook_instance_str",
        vec![param("x", Ty::Object, span)],
        Ty::Str,
        statements,
        span,
    );
    // type(): the class name.
    let mut statements = per_class(
        module,
        x.clone(),
        |_| true,
        |_, c, _| vec![ret(str_lit(&module.classes[c].name, span), span)],
        span,
    );
    statements.push(ret(str_lit("object", span), span));
    let type_hook = function(
        "zb_hook_instance_type",
        vec![param("x", Ty::Object, span)],
        Ty::Str,
        statements,
        span,
    );
    // ==: `__eq__` on the left operand's class, else identity.
    let a = var(intern("a"), Ty::Object, span);
    let b = var(intern("b"), Ty::Object, span);
    let mut statements = per_class(
        module,
        a.clone(),
        |c| module.method_sig(c, "__eq__").is_some(),
        |lowerer, c, obj| {
            let (sig, _) = module.method_sig(c, "__eq__").expect("picked");
            let other_ty = sig.params.get(1).map(|(_, t)| *t).unwrap_or(Ty::Object);
            let other = lowerer.coerce(
                Val {
                    node: var(intern("b"), Ty::Object, span),
                    ty: Ty::Object,
                },
                other_ty,
            );
            let result = lowerer
                .invoke(c, "__eq__", obj, vec![other], span)
                .expect("picked");
            let truth = lowerer.truthy(result);
            vec![ret(truth, span)]
        },
        span,
    );
    statements.push(ret(call("zb_any_same", vec![a, b], Ty::Bool, span), span));
    let eq_hook = function(
        "zb_hook_instance_eq",
        vec![param("a", Ty::Object, span), param("b", Ty::Object, span)],
        Ty::Bool,
        statements,
        span,
    );
    vec![
        str_hook,
        type_hook,
        eq_hook,
        box_hook(module, span),
        unbox_hook(module, span),
    ]
}

/// `zb_hook_unbox_instance(x, tag)`: the address in a box holding an
/// instance of the class `tag` names, or of a subclass; anything else
/// is a TypeError. The library calls it for each element of a list
/// becoming a list of one class.
fn unbox_hook(module: &Module, span: Span) -> TypedFunction {
    let x = var(intern("x"), Ty::Object, span);
    let tag = var(intern("tag"), Ty::Int, span);
    let tag_param = TypedParameter {
        name: intern("tag"),
        ty: Type::Primitive(zyntax_typed_ast::PrimitiveType::I32),
        mutability: Mutability::Immutable,
        kind: ParameterKind::Regular,
        default_value: None,
        attributes: Vec::new(),
        ownership: ParamOwnership::Copied,
        span,
    };
    let mut statements = Vec::new();
    for (k, class) in module.classes.iter().enumerate() {
        let matches = binary(
            BinaryOp::Eq,
            cast(tag.clone(), Ty::Int, span),
            int_lit(zyntax_builtins::instance_tag(k) as i64, span),
            Ty::Bool,
            span,
        );
        let address = lower::addr_call(&format!("{}$unbox", class.name), vec![x.clone()], span);
        statements.push(when(matches, vec![ret(address, span)], span));
    }
    statements.push(ret(
        lower::addr_call("zb_unbox_instance_raw", vec![x], span),
        span,
    ));
    let mut f = function(
        "zb_hook_unbox_instance",
        vec![param("x", Ty::Object, span), tag_param],
        Ty::Int,
        statements,
        span,
    );
    f.return_type = lower::addr_type();
    f
}

/// `zb_hook_box_instance(p)`: an instance from its address, boxed under
/// its class's tag. Every instance keeps its class index in its first
/// field, so any class's layout reads it.
fn box_hook(module: &Module, span: Span) -> TypedFunction {
    let p = lower::code_of("p", span);
    let param = TypedParameter {
        name: intern("p"),
        ty: lower::addr_type(),
        mutability: Mutability::Immutable,
        kind: ParameterKind::Regular,
        default_value: None,
        attributes: Vec::new(),
        ownership: ParamOwnership::Copied,
        span,
    };
    let tag = if module.classes.is_empty() {
        // No class exists to be an instance of.
        int_lit(255, span)
    } else {
        let index = field(cast(p.clone(), Ty::Class(0), span), "$class", Ty::Int, span);
        let shifted = binary(
            BinaryOp::Shl,
            binary(
                BinaryOp::Add,
                int_lit(zyntax_builtins::INSTANCE_KIND_BASE, span),
                index,
                Ty::Int,
                span,
            ),
            int_lit(8, span),
            Ty::Int,
            span,
        );
        binary(BinaryOp::BitOr, shifted, int_lit(255, span), Ty::Int, span)
    };
    let boxed = call(
        "zb_box_instance_raw",
        vec![p, cast_i32(tag, span)],
        Ty::Object,
        span,
    );
    function(
        "zb_hook_box_instance",
        vec![param],
        Ty::Object,
        vec![ret(boxed, span)],
        span,
    )
}

/// An int narrowed to the i32 a box tag is.
fn cast_i32(value: Node, span: Span) -> Node {
    let i32_ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::I32);
    TypedNode::new(
        TypedExpression::Cast(zyntax_typed_ast::typed_ast::TypedCast {
            expr: Box::new(value),
            target_type: i32_ty.clone(),
        }),
        i32_ty,
        span,
    )
}

use zyntax_typed_ast::BinaryOp;

/// `zb_hook_raise(kind, message)`: the library's error as the exception
/// class of that name, pending.
pub(crate) fn raise_hook(module: &Module) -> TypedFunction {
    let span = Span::new(0, 0);
    let mut hook = raise_hook_body(module, span);
    // An error path: a caller keeps the call rather than copying the
    // hook's switch over every exception class into its own body.
    hook.annotations.push(TypedAnnotation {
        name: intern("cold"),
        args: Vec::new(),
        span,
    });
    hook
}

fn raise_hook_body(module: &Module, span: Span) -> TypedFunction {
    let mut lowerer = scratch(module);
    let kind = var(intern("kind"), Ty::Str, span);
    let message = var(intern("message"), Ty::Str, span);
    let mut statements = Vec::new();
    for name in crate::prelude::EXCEPTION_KINDS {
        let Some(&k) = module.class_index.get(*name) else {
            continue;
        };
        let instance = Val {
            node: call(
                &new_name(name),
                vec![message.clone()],
                Ty::Class(k as u16),
                span,
            ),
            ty: Ty::Class(k as u16),
        };
        let boxed = lowerer.coerce(instance, Ty::Object);
        let set = binary(
            BinaryOp::Assign,
            var(intern(lower::PENDING), Ty::Object, span),
            boxed,
            Ty::None,
            span,
        );
        let matches = call(
            "zb_str_eq",
            vec![kind.clone(), str_lit(name, span)],
            Ty::Bool,
            span,
        );
        statements.push(when(
            matches,
            vec![
                stmt(set, span),
                TypedNode::new(TypedStatement::Return(None), Type::Unknown, span),
            ],
            span,
        ));
    }
    function(
        "zb_hook_raise",
        vec![
            param("kind", Ty::Str, span),
            param("message", Ty::Str, span),
        ],
        Ty::None,
        statements,
        span,
    )
}
