//! Classes as structs. An instance is a heap struct whose first field is
//! the class's index; a method is a function taking the instance first;
//! a method some subclass overrides is called through a dispatcher that
//! switches on that index. Attribute access and method calls on a
//! dynamic receiver go through dispatchers generated per name over every
//! class that has it.

use crate::lower::{
    self, Lowerer, Node, Val, binary, call, callm_name, cast, dispatch_name, field_storage,
    getattr_name, int_lit, ir, method_call, new_name, node, setattr_name, str_lit, var,
    without_self,
};
use crate::scope::Scope;
use crate::types::{ClassInfo, Locals, Module, Sig, Ty, method_fn};
use crate::{Error, Result, intern};
use ruff_python_ast as py;
use ruff_text_size::Ranged;
use rustc_hash::FxHashMap as HashMap;
use zyntax_typed_ast::TypeId;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{FieldDef, TypeDefinition, TypeKind, TypeMetadata};
use zyntax_typed_ast::typed_ast::{
    TypedAnnotation, TypedBlock, TypedCall, TypedCast, TypedClass, TypedDeclaration, TypedField,
    TypedFieldAccess, TypedFieldInit, TypedFunction, TypedIf, TypedLet, TypedLiteral,
    TypedParameter, TypedStatement, TypedStructLiteral,
};
use zyntax_typed_ast::{
    Mutability, ParamOwnership, ParameterKind, Type, TypeRegistry, TypedNode, Visibility,
};

/// A class as written: its name, base, methods and the attributes its
/// body declares.
pub(crate) struct ClassDef<'a> {
    pub(crate) name: String,
    pub(crate) base: Option<String>,
    pub(crate) methods: Vec<&'a py::StmtFunctionDef>,
    pub(crate) attrs: Vec<crate::class_attrs::Declared<'a>>,
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
        let methods = c
            .body
            .iter()
            .filter_map(|s| match s {
                py::Stmt::FunctionDef(f) => Some(f),
                _ => None,
            })
            .collect();
        let attrs = crate::class_attrs::declared_in(c).map_err(located)?;
        out.push(ClassDef {
            range: c.range(),
            module: module.clone(),
            name: c.name.to_string(),
            base,
            methods,
            attrs,
        });
    }
    Ok(out)
}

/// The classes as the module first knows them: names, bases, methods,
/// and the class tag as the only field.
pub(crate) fn skeletons(defs: &[ClassDef<'_>]) -> Result<(Vec<ClassInfo>, HashMap<String, usize>)> {
    // A base is declared before what derives from it.
    let mut declared: HashMap<&str, usize> = HashMap::default();
    let mut base_of: Vec<Option<usize>> = Vec::with_capacity(defs.len());
    for (i, def) in defs.iter().enumerate() {
        let base = match &def.base {
            Some(b) => Some(*declared.get(b.as_str()).ok_or_else(|| {
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
        declared.insert(def.name.as_str(), i);
        base_of.push(base);
    }
    // Classes are numbered in preorder of the hierarchy, each right
    // after its base and before the next of its base's children, so a
    // class's descendants are a contiguous range after it: one compare
    // pair tells whether an instance is one.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); defs.len()];
    let mut roots = Vec::new();
    for (i, base) in base_of.iter().enumerate() {
        match base {
            Some(b) => children[*b].push(i),
            None => roots.push(i),
        }
    }
    let mut order: Vec<usize> = Vec::with_capacity(defs.len());
    let mut descendants: Vec<usize> = vec![1; defs.len()];
    fn visit(
        i: usize,
        children: &[Vec<usize>],
        order: &mut Vec<usize>,
        descendants: &mut [usize],
    ) -> usize {
        order.push(i);
        let mut size = 1;
        for &c in &children[i] {
            size += visit(c, children, order, descendants);
        }
        descendants[i] = size;
        size
    }
    for r in roots {
        visit(r, &children, &mut order, &mut descendants);
    }
    let mut position: Vec<usize> = vec![0; defs.len()];
    for (k, &i) in order.iter().enumerate() {
        position[i] = k;
    }
    let mut classes = Vec::with_capacity(defs.len());
    let mut index = HashMap::default();
    for &i in &order {
        let def = &defs[i];
        index.insert(def.name.clone(), classes.len());
        classes.push(ClassInfo {
            name: def.name.clone(),
            base: base_of[i].map(|b| position[b]),
            descendants: descendants[i],
            fields: vec![("$class".to_string(), Ty::Int)],
            methods: def.methods.iter().map(|m| m.name.to_string()).collect(),
            type_id: None,
            module: def.module.clone(),
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
    // Every class has its id before any is laid out, so a field holding
    // an instance of a class declared later, or of its own, has a type.
    let ids: Vec<TypeId> = module.classes.iter().map(|_| TypeId::next()).collect();
    lower::set_class_types(ids.clone());
    // The declaration names the class's file, so a reader tells the
    // program's classes from the prelude's and the imported modules'.
    let declared_in: Vec<Span> = module
        .classes
        .iter()
        .map(|class| Span::in_file(0, 0, module.file_of(class.module.as_deref())))
        .collect();
    for (k, class) in module.classes.iter_mut().enumerate() {
        let declared = declared_in[k];
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
        let id = ids[k];
        registry.register_type(TypeDefinition {
            id,
            module: None,
            name: intern(&class.name),
            kind: TypeKind::Struct {
                fields: fields.clone(),
                is_tuple: false,
            },
            type_params: Vec::new(),
            constraints: Vec::new(),
            fields,
            methods: Vec::new(),
            constructors: Vec::new(),
            metadata: TypeMetadata {
                is_reference: true,
                ..Default::default()
            },
            span,
        });
        class.type_id = Some(id);
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
                span: declared,
            }),
            Type::Unknown,
            declared,
        ));
    }
    decls
}

fn ret_void(span: Span) -> TypedNode<TypedStatement> {
    TypedNode::new(TypedStatement::Return(None), Type::Unknown, span)
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
            _ => ParamOwnership::Shared,
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
        HashMap::default(),
    );
    lowerer.guards = false;
    lowerer
}

use zyntax_typed_ast::typed_ast::TypedExpression;

/// How the generated functions raise: a constructor through the
/// `__init__` it calls, a dispatcher through the methods it reaches.
/// Neither raises on its own.
pub(crate) fn raise_facts(
    module: &Module,
    facts: &mut std::collections::BTreeMap<String, crate::types::RaiseFact>,
) {
    for (k, class) in module.classes.iter().enumerate() {
        let mut constructor = crate::types::RaiseFact::default();
        if let Some((_, init)) = module.method_sig(k, "__init__") {
            constructor.callees.insert(init);
        }
        facts.insert(lower::new_name(&class.name), constructor);
        for method in &class.methods {
            let fn_name = method_fn(&class.name, method);
            let overriders = module.overriders(k, method);
            if overriders.is_empty() || method == "__init__" {
                continue;
            }
            let mut dispatcher = crate::types::RaiseFact::default();
            dispatcher.callees.insert(fn_name.clone());
            for sub in overriders {
                dispatcher
                    .callees
                    .insert(method_fn(&module.classes[sub].name, method));
            }
            facts.insert(lower::dispatch_name(&fn_name), dispatcher);
        }
    }
}

/// The functions every class needs: construction, unboxing, dispatch.
pub(crate) fn generated(module: &Module) -> Vec<TypedFunction> {
    let span = Span::new(0, 0);
    let mut out = Vec::new();
    for (k, class) in module.classes.iter().enumerate() {
        out.push(constructor(module, k, span));
        out.push(unboxer(module, k, span));
        // A constructor is called by its class's name, never through
        // an instance of a base, so it has no dispatcher.
        for method in &class.methods {
            let fn_name = method_fn(&class.name, method);
            if method != "__init__" && !module.overriders(k, method).is_empty() {
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
    for class in module.raisers.borrow().iter() {
        out.push(raiser(module, class, span));
    }
    out.extend(hooks(module, span));
    out
}

/// `py$raise$Class(message)`: the instance built and left pending.
/// Cold, so a raise is a call the caller neither inlines nor compiles
/// ahead of its first use.
fn raiser(module: &Module, class: &str, span: Span) -> TypedFunction {
    let mut lowerer = scratch(module);
    let k = module.class_index[class];
    let instance = Val {
        node: call(
            &new_name(class),
            vec![var(intern("message"), Ty::Str, span)],
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
    let mut f = function(
        &crate::types::raiser_name(class),
        vec![param("message", Ty::Str, span)],
        Ty::None,
        vec![
            stmt(set, span),
            TypedNode::new(TypedStatement::Return(None), Type::Unknown, span),
        ],
        span,
    );
    f.annotations.push(TypedAnnotation {
        name: intern("cold"),
        args: Vec::new(),
        span,
    });
    f
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
                Ty::Class(c) => lowerer.coerce(none(span), Ty::Class(c)),
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
    // None is the null instance of any class.
    let is_none = binary(
        BinaryOp::Eq,
        x.clone(),
        node(
            TypedExpression::Literal(TypedLiteral::Null),
            Ty::Object,
            span,
        ),
        Ty::Bool,
        span,
    );
    // The class and everything deriving from it are one range of kinds.
    let kind = call("zb_any_kind", vec![x.clone()], Ty::Int, span);
    let first = zyntax_builtins::INSTANCE_KIND_BASE + k as i64;
    let accepted = binary(
        BinaryOp::And,
        binary(
            BinaryOp::Ge,
            kind.clone(),
            int_lit(first, span),
            Ty::Bool,
            span,
        ),
        binary(
            BinaryOp::Lt,
            kind,
            int_lit(first + class.descendants as i64, span),
            Ty::Bool,
            span,
        ),
        Ty::Bool,
        span,
    );
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
    // The raw read takes a box; a rejected value may be None, so the
    // error path leaves with a null address instead.
    let statements = vec![
        when(
            is_none,
            vec![ret(lower::as_addr(lower::int_lit(0, span), span), span)],
            span,
        ),
        when(
            not_accepted,
            vec![
                stmt(
                    call(
                        "zb_fatal",
                        vec![str_lit("TypeError", span), message],
                        Ty::None,
                        span,
                    ),
                    span,
                ),
                ret(lower::as_addr(lower::int_lit(0, span), span), span),
            ],
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
    let base = &module.funcs[fn_name];
    // The dispatcher returns what any of the methods it reaches may.
    let sig = Sig {
        params: base.params.clone(),
        ret: module.dispatched_ret(owner, method).unwrap_or(base.ret),
        defaults: base.defaults.clone(),
    };
    let sig = &sig;
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
        // An override taking another number of arguments cannot be
        // called with these; the call fails as it would in Python.
        if sub_sig.params.len() != sig.params.len() {
            let message = str_lit(
                &format!(
                    "{}() takes {} positional arguments but {} were given",
                    method,
                    sub_sig.params.len(),
                    sig.params.len()
                ),
                span,
            );
            let fail = stmt(
                call(
                    "zb_fatal",
                    vec![str_lit("TypeError", span), message],
                    Ty::None,
                    span,
                ),
                span,
            );
            let none = lowerer.coerce(
                Val {
                    node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
                    ty: Ty::None,
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
                vec![fail, ret(none, span)],
                span,
            ));
            continue;
        }
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
    let own = lowerer.coerce(
        Val {
            node: call(fn_name, args, base.ret, span),
            ty: base.ret,
        },
        sig.ret,
    );
    statements.push(ret(own, span));
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
        // The box held an instance of this class, so `obj` is one.
        lowerer.assume_instance(intern("obj"));
        let arm = body(
            &mut lowerer,
            c,
            var(intern("obj"), Ty::Class(c as u16), span),
        );
        then.append(&mut lowerer.hoisted);
        then.extend(arm);
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
            // A field only ever given None holds nothing else; a value
            // arriving through a dynamic receiver is refused rather than
            // stored as None.
            let mut out = Vec::new();
            if ty == Ty::None {
                let not_none = binary(
                    BinaryOp::Ne,
                    call(
                        "zb_any_category",
                        vec![var(intern("v"), Ty::Object, span)],
                        Ty::Int,
                        span,
                    ),
                    int_lit(zyntax_builtins::NONE_CATEGORY, span),
                    Ty::Bool,
                    span,
                );
                let message = str_lit(
                    &format!(
                        "attribute '{attr}' of '{}' holds None only",
                        module.classes[c].name
                    ),
                    span,
                );
                out.push(when(
                    not_none,
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
                ));
            }
            let value = lowerer.coerce(
                Val {
                    node: var(intern("v"), Ty::Object, span),
                    ty: Ty::Object,
                },
                ty,
            );
            let stored = lowerer.coerce(Val { node: value, ty }, field_storage(ty));
            out.extend([
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
            ]);
            out
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
    // Each receiver: the test on the box, its type, and the raw read of
    // the payload where the trusted read of the type is not it.
    let mut receivers: Vec<(Node, Ty, Option<String>)> = vec![
        (category(5), Ty::Str, None),
        (
            list_kind(zyntax_builtins::Kind::Int),
            Ty::List(Elem::Int),
            None,
        ),
        (
            list_kind(zyntax_builtins::Kind::Float),
            Ty::List(Elem::Float),
            None,
        ),
        (
            list_kind(zyntax_builtins::Kind::Str),
            Ty::List(Elem::Str),
            None,
        ),
        (
            list_kind(zyntax_builtins::Kind::Any),
            Ty::List(Elem::Object),
            None,
        ),
        (
            kind(zyntax_builtins::DICT_TAG >> 8),
            crate::types::dynamic_dict(),
            None,
        ),
        (kind(zyntax_builtins::SET_TAG >> 8), Ty::Set, None),
    ];
    // A boxed tuple is a list of dynamic values under its own tag, and
    // answers the two methods a tuple has as that list.
    if matches!(method, "count" | "index") {
        receivers.push((
            kind(zyntax_builtins::TUPLE_TAG >> 8),
            Ty::List(Elem::Object),
            Some("zb_unbox_tuple_raw".to_string()),
        ));
    }
    // A list of tuples of each shape the program has.
    for k in crate::types::tuple_lists() {
        let e = Elem::Tuple(k);
        receivers.push((
            kind(e.list_tag() >> 8),
            Ty::List(e),
            Some(format!("zb_unbox_list_raw_{}", e.suffix())),
        ));
    }
    let mut arms = Vec::new();
    for (test, ty, raw) in receivers {
        let mut vars: Vec<(&str, Ty)> = vec![("s", ty)];
        for a in &args {
            vars.push((a.as_str(), Ty::Object));
        }
        let mut lowerer = scratch_with(module, &vars);
        // The receiver is what the arm's test says it is; the method
        // then runs on it, and may leave statements to run ahead of it.
        let receiver = match raw {
            Some(read) => call(&read, vec![x.clone()], ty, span),
            None => lowerer.trusted(
                Val {
                    node: x.clone(),
                    ty: Ty::Object,
                },
                ty,
            ),
        };
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
        HashMap::default(),
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
    // repr(): `__repr__`, or `<C object>`.
    let mut statements = per_class(
        module,
        x.clone(),
        |_| true,
        |lowerer, c, obj| {
            let text = lowerer.repr_of(Val {
                node: obj,
                ty: Ty::Class(c as u16),
            });
            vec![ret(text, span)]
        },
        span,
    );
    statements.push(ret(str_lit("<object>", span), span));
    let repr_hook = function(
        "zb_hook_instance_repr",
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
    let mut out = vec![
        str_hook,
        repr_hook,
        type_hook,
        eq_hook,
        hash_hook(module, span),
        arith_hook(module, span),
        box_hook(module, span),
        unbox_hook(module, span),
    ];
    out.extend(shaped_hooks(span));
    out
}

/// What the library asks of a boxed list whose elements are a tuple
/// shape: its elements as dynamic values, one element, a store and an
/// append. Each dispatches on the box's kind to the list functions
/// generated for the shape; a kind no list here has is a type error.
fn shaped_hooks(span: Span) -> Vec<TypedFunction> {
    use crate::types::Elem;
    let x = var(intern("x"), Ty::Object, span);
    let i = var(intern("i"), Ty::Int, span);
    let v = var(intern("v"), Ty::Object, span);
    let shapes: Vec<u16> = crate::types::tuple_lists().into_iter().collect();
    let kind = call("zb_any_kind", vec![x.clone()], Ty::Int, span);
    let is_shape = |k: u16| {
        binary(
            BinaryOp::Eq,
            kind.clone(),
            int_lit(Elem::Tuple(k).list_tag() >> 8, span),
            Ty::Bool,
            span,
        )
    };
    let raw = |k: u16| {
        call(
            &format!("zb_unbox_list_raw_{}", Elem::Tuple(k).suffix()),
            vec![x.clone()],
            Ty::List(Elem::Tuple(k)),
            span,
        )
    };
    let unknown = || {
        stmt(
            call(
                "zb_fatal",
                vec![
                    str_lit("TypeError", span),
                    str_lit("a list of an unknown kind", span),
                ],
                Ty::None,
                span,
            ),
            span,
        )
    };
    let mut items = Vec::new();
    let mut get = Vec::new();
    let mut set = Vec::new();
    let mut append = Vec::new();
    let mut repr = Vec::new();
    // An array's kind holds its storage kind in the low byte, above
    // `ARRAY_KIND_BASE`; every typecode stored that way shares the arms,
    // and the repr reads the typecode letter out of the kind.
    let base = zyntax_builtins::ARRAY_KIND_BASE;
    let is_array_of = |storage: zyntax_builtins::Kind| {
        binary(
            BinaryOp::And,
            binary(
                BinaryOp::Ge,
                kind.clone(),
                int_lit(base, span),
                Ty::Bool,
                span,
            ),
            binary(
                BinaryOp::Eq,
                binary(
                    BinaryOp::BitAnd,
                    kind.clone(),
                    int_lit(255, span),
                    Ty::Bool,
                    span,
                ),
                int_lit(storage.list_tag() >> 8, span),
                Ty::Bool,
                span,
            ),
            Ty::Bool,
            span,
        )
    };
    let letter = || {
        call(
            "zb_str_chr",
            vec![binary(
                BinaryOp::Shr,
                binary(
                    BinaryOp::Sub,
                    kind.clone(),
                    int_lit(base, span),
                    Ty::Int,
                    span,
                ),
                int_lit(8, span),
                Ty::Int,
                span,
            )],
            Ty::Str,
            span,
        )
    };
    for storage in crate::types::array_kinds() {
        let suffix = storage.suffix();
        let stored = storage.ty();
        let wide = storage.wide().ty();
        let list_ty = zyntax_builtins::list_of(lower::list_type_id(), stored.clone());
        let raw = || {
            TypedNode::new(
                TypedExpression::Call(TypedCall {
                    callee: Box::new(var(
                        intern(&format!("zb_unbox_list_raw_{suffix}")),
                        Ty::Unknown,
                        span,
                    )),
                    positional_args: vec![x.clone()],
                    named_args: Vec::new(),
                    type_args: Vec::new(),
                }),
                list_ty.clone(),
                span,
            )
        };
        let typed_call = |name: &str, args: Vec<Node>, ty: Type| {
            TypedNode::new(
                TypedExpression::Call(TypedCall {
                    callee: Box::new(var(intern(name), Ty::Unknown, span)),
                    positional_args: args,
                    named_args: Vec::new(),
                    type_args: Vec::new(),
                }),
                ty,
                span,
            )
        };
        let widen = |e: Node| {
            TypedNode::new(
                TypedExpression::Cast(TypedCast {
                    expr: Box::new(e),
                    target_type: wide.clone(),
                }),
                wide.clone(),
                span,
            )
        };
        // The number a dynamic value holds, at the stored width: checked
        // by the storage's narrowing where it has one.
        let as_stored = |v: Node| {
            let read = match storage.wide() {
                zyntax_builtins::Kind::Float => call("zb_any_as_f64", vec![v], Ty::Float, span),
                _ => call("zb_any_as_i64", vec![v], Ty::Int, span),
            };
            if storage.wide() == storage {
                return read;
            }
            typed_call(
                &format!("zb_list_narrow_{suffix}"),
                vec![read],
                stored.clone(),
            )
        };
        let boxed = |e: Node| match storage.wide() {
            zyntax_builtins::Kind::Float => call("zb_box_f64", vec![widen(e)], Ty::Object, span),
            _ => call("zb_box_i64", vec![widen(e)], Ty::Object, span),
        };
        items.push(when(
            is_array_of(storage),
            vec![ret(
                call(
                    &format!("zb_list_to_any_{suffix}"),
                    vec![raw()],
                    Ty::List(Elem::Object),
                    span,
                ),
                span,
            )],
            span,
        ));
        get.push(when(
            is_array_of(storage),
            vec![ret(
                boxed(typed_call(
                    &format!("zb_list_get_{suffix}"),
                    vec![raw(), i.clone()],
                    stored.clone(),
                )),
                span,
            )],
            span,
        ));
        set.push(when(
            is_array_of(storage),
            vec![
                stmt(
                    call(
                        &format!("zb_list_set_{suffix}"),
                        vec![raw(), i.clone(), as_stored(v.clone())],
                        Ty::None,
                        span,
                    ),
                    span,
                ),
                ret_void(span),
            ],
            span,
        ));
        append.push(when(
            is_array_of(storage),
            vec![
                stmt(
                    method_call(raw(), "push", vec![as_stored(v.clone())], Ty::None, span),
                    span,
                ),
                ret_void(span),
            ],
            span,
        ));
        let empty = binary(
            BinaryOp::Eq,
            method_call(raw(), "len", vec![], Ty::Int, span),
            int_lit(0, span),
            Ty::Bool,
            span,
        );
        let prefix = |open: &str| {
            binary(
                BinaryOp::Add,
                binary(
                    BinaryOp::Add,
                    str_lit("array('", span),
                    letter(),
                    Ty::Str,
                    span,
                ),
                str_lit(open, span),
                Ty::Str,
                span,
            )
        };
        repr.push(when(
            is_array_of(storage),
            vec![
                when(empty, vec![ret(prefix("')"), span)], span),
                ret(
                    call(
                        &format!("zb_list_items_{suffix}"),
                        vec![raw(), prefix("', ["), str_lit("])", span)],
                        Ty::Str,
                        span,
                    ),
                    span,
                ),
            ],
            span,
        ));
    }
    for &k in &shapes {
        let e = Elem::Tuple(k);
        repr.push(when(
            is_shape(k),
            vec![ret(
                call(
                    &format!("zb_list_repr_{}", e.suffix()),
                    vec![raw(k)],
                    Ty::Str,
                    span,
                ),
                span,
            )],
            span,
        ));
        items.push(when(
            is_shape(k),
            vec![ret(
                call(
                    &format!("zb_list_to_any_{}", e.suffix()),
                    vec![raw(k)],
                    Ty::List(Elem::Object),
                    span,
                ),
                span,
            )],
            span,
        ));
        let element = call(
            &format!("zb_list_get_{}", e.suffix()),
            vec![raw(k), i.clone()],
            Ty::Tuple(k),
            span,
        );
        get.push(when(
            is_shape(k),
            vec![ret(
                call(
                    &format!("zb_tuple_box_{}", e.suffix()),
                    vec![element],
                    Ty::Object,
                    span,
                ),
                span,
            )],
            span,
        ));
        let read = call(
            &format!("zb_tuple_read_{}", e.suffix()),
            vec![v.clone()],
            Ty::Tuple(k),
            span,
        );
        set.push(when(
            is_shape(k),
            vec![
                stmt(
                    call(
                        &format!("zb_list_set_{}", e.suffix()),
                        vec![raw(k), i.clone(), read.clone()],
                        Ty::None,
                        span,
                    ),
                    span,
                ),
                ret_void(span),
            ],
            span,
        ));
        append.push(when(
            is_shape(k),
            vec![
                stmt(
                    method_call(raw(k), "push", vec![read], Ty::None, span),
                    span,
                ),
                ret_void(span),
            ],
            span,
        ));
    }
    items.push(unknown());
    items.push(ret(
        node(
            TypedExpression::Array(Vec::new()),
            Ty::List(Elem::Object),
            span,
        ),
        span,
    ));
    get.push(unknown());
    get.push(ret(
        node(
            TypedExpression::Literal(TypedLiteral::Null),
            Ty::Object,
            span,
        ),
        span,
    ));
    set.push(unknown());
    set.push(ret_void(span));
    append.push(unknown());
    append.push(ret_void(span));
    repr.push(unknown());
    repr.push(ret(str_lit("", span), span));
    vec![
        function(
            "zb_hook_shaped_repr",
            vec![param("x", Ty::Object, span)],
            Ty::Str,
            repr,
            span,
        ),
        function(
            "zb_hook_shaped_items",
            vec![param("x", Ty::Object, span)],
            Ty::List(Elem::Object),
            items,
            span,
        ),
        function(
            "zb_hook_shaped_get",
            vec![param("x", Ty::Object, span), param("i", Ty::Int, span)],
            Ty::Object,
            get,
            span,
        ),
        function(
            "zb_hook_shaped_set",
            vec![
                param("x", Ty::Object, span),
                param("i", Ty::Int, span),
                param("v", Ty::Object, span),
            ],
            Ty::None,
            set,
            span,
        ),
        function(
            "zb_hook_shaped_append",
            vec![param("x", Ty::Object, span), param("v", Ty::Object, span)],
            Ty::None,
            append,
            span,
        ),
    ]
}

/// `zb_hook_instance_hash(x)`: what a dict keys an instance by. A class
/// with `__hash__` answers through it; one with `__eq__` and no
/// `__hash__` is unhashable, as in Python, since equal instances would
/// land in different places; any other class hashes by identity.
fn hash_hook(module: &Module, span: Span) -> TypedFunction {
    let x = var(intern("x"), Ty::Object, span);
    let mut statements = per_class(
        module,
        x.clone(),
        |c| module.method_sig(c, "__hash__").is_some() || module.method_sig(c, "__eq__").is_some(),
        |lowerer, c, obj| {
            if module.method_sig(c, "__hash__").is_none() {
                let message = str_lit(
                    &format!("unhashable type: '{}'", module.classes[c].name),
                    span,
                );
                return vec![
                    stmt(
                        call(
                            "zb_fatal",
                            vec![str_lit("TypeError", span), message],
                            Ty::None,
                            span,
                        ),
                        span,
                    ),
                    ret(int_lit(0, span), span),
                ];
            }
            let result = lowerer
                .invoke(c, "__hash__", obj, vec![], span)
                .expect("picked");
            let value = lowerer.coerce(result, Ty::Int);
            vec![ret(value, span)]
        },
        span,
    );
    statements.push(ret(
        lower::addr_call("zb_unbox_instance_raw", vec![x], span),
        span,
    ));
    function(
        "zb_hook_instance_hash",
        vec![param("x", Ty::Object, span)],
        Ty::Int,
        statements,
        span,
    )
}

/// `zb_hook_instance_arith(code, a, b)`: `a op b` where `a` is an
/// instance, through the operator method its class defines for the
/// operation `code` names; anything else is a TypeError. The library
/// calls it from dynamic arithmetic when either side is an instance.
fn arith_hook(module: &Module, span: Span) -> TypedFunction {
    const OPS: [ruff_python_ast::Operator; 13] = crate::types::OPERATORS;
    let code = var(intern("code"), Ty::Int, span);
    let a = var(intern("a"), Ty::Object, span);
    let b = var(intern("b"), Ty::Object, span);
    let mut statements = per_class(
        module,
        a.clone(),
        |c| {
            OPS.iter().any(|op| {
                module
                    .method_sig(c, crate::types::dunder_name(*op))
                    .is_some()
            })
        },
        |lowerer, c, obj| {
            let mut arms = Vec::new();
            for op in OPS {
                let name = crate::types::dunder_name(op);
                if module.method_sig(c, name).is_none() {
                    continue;
                }
                let other = lower::Val {
                    node: b.clone(),
                    ty: Ty::Object,
                };
                let Some(result) = lowerer.dunder(c, name, obj.clone(), vec![other], span) else {
                    continue;
                };
                let boxed = lowerer.coerce(result, Ty::Object);
                arms.push(when(
                    binary(
                        BinaryOp::Eq,
                        code.clone(),
                        int_lit(crate::types::arith_code(op), span),
                        Ty::Bool,
                        span,
                    ),
                    vec![ret(boxed, span)],
                    span,
                ));
            }
            arms
        },
        span,
    );
    let message = binary(
        BinaryOp::Add,
        binary(
            BinaryOp::Add,
            str_lit("unsupported operand type(s): '", span),
            call("zb_any_type", vec![a], Ty::Str, span),
            Ty::Str,
            span,
        ),
        binary(
            BinaryOp::Add,
            str_lit("' and '", span),
            binary(
                BinaryOp::Add,
                call("zb_any_type", vec![b], Ty::Str, span),
                str_lit("'", span),
                Ty::Str,
                span,
            ),
            Ty::Str,
            span,
        ),
        Ty::Str,
        span,
    );
    statements.push(stmt(
        call(
            "zb_fatal",
            vec![str_lit("TypeError", span), message],
            Ty::None,
            span,
        ),
        span,
    ));
    statements.push(ret(none(span).node, span));
    function(
        "zb_hook_instance_arith",
        vec![
            param("code", Ty::Int, span),
            param("a", Ty::Object, span),
            param("b", Ty::Object, span),
        ],
        Ty::Object,
        statements,
        span,
    )
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
    // A tag of no class: nothing to read out of the value.
    statements.push(stmt(
        call(
            "zb_fatal",
            vec![
                str_lit("TypeError", span),
                str_lit("not an address of that kind", span),
            ],
            Ty::None,
            span,
        ),
        span,
    ));
    statements.push(ret(lower::as_addr(lower::int_lit(0, span), span), span));
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
        vec![p.clone(), cast_i32(tag, span)],
        Ty::Object,
        span,
    );
    // A null address is None.
    let is_null = binary(BinaryOp::Eq, p, lower::int_lit(0, span), Ty::Bool, span);
    let none = node(
        TypedExpression::Literal(TypedLiteral::Null),
        Ty::Object,
        span,
    );
    function(
        "zb_hook_box_instance",
        vec![param],
        Ty::Object,
        vec![when(is_null, vec![ret(none, span)], span), ret(boxed, span)],
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
