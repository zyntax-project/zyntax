//! A Python frontend for Zyntax.
//!
//! Python source is parsed by `ruff_python_parser` and its AST is
//! rewritten into a [`TypedProgram`], which the runtime compiles the
//! same way it compiles anything else. No grammar of our own: Python's
//! surface (indentation, soft keywords, nested f-strings) is a poor fit
//! for a PEG and a solved problem in Ruff's hand-written parser.
//!
//! ## Types
//!
//! Every expression gets a static type from [`types`]: `int` is `i64`,
//! `float` is `f64`, `bool` is `bool`, `str` is `String`, `None` is
//! `Unit`, and a value the pass cannot type is `Any`, the IR's boxed
//! dynamic value. Crossings between them are explicit in what
//! [`lower`] emits. What Python defines above the IR (how a value
//! prints, list and string operations, dynamic dispatch) comes from the
//! shared built-in library with Python's spellings, compiled with the
//! program.

use ruff_python_ast as py;
use ruff_text_size::Ranged;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{TypedBlock, TypedDeclaration, TypedFunction, TypedVariable};
use zyntax_typed_ast::{
    InternedString, Mutability, PrimitiveType, Type, TypedNode, TypedProgram, Visibility,
};

mod classes;
mod format;
mod lower;
mod prelude;
mod scope;
mod types;

/// Why a program could not be turned into a `TypedProgram`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Python syntax error: {0}")]
    Syntax(String),
    /// Something Python allows that this frontend does not compile yet.
    /// Says what it was and where, so the gap is a fact rather than a
    /// guess.
    #[error("{what} is not supported yet (at byte offset {at})")]
    Unsupported { what: String, at: usize },
}

type Result<T> = std::result::Result<T, Error>;

/// The function a module's top-level statements become. A host runs a
/// Python program by calling this.
pub const ENTRY: &str = "__main__";

/// Python's spellings for the built-in library.
const POLICY: zyntax_builtins::Policy = zyntax_builtins::Policy {
    true_text: "True",
    false_text: "False",
    none_text: "None",
    single_quotes: true,
    float_fraction: true,
    instance_hooks: true,
    exceptions: true,
    type_names: zyntax_builtins::TypeNames {
        none: "NoneType",
        bool: "bool",
        int: "int",
        float: "float",
        str: "str",
        list: "list",
        tuple: "tuple",
        dict: "dict",
        set: "set",
        function: "function",
        object: "object",
    },
};

/// Give a runtime what a compiled Python program links against: the IO
/// and string plugins the library's primitives come from, and the name
/// a program is entered through, so only the library the program
/// reaches is built. A host calls this once before compiling a program.
pub fn register_runtime(
    runtime: &mut zyntax_embed::TieredRuntime,
) -> std::result::Result<(), zyntax_embed::RuntimeError> {
    runtime.declare_entry_points([ENTRY]);
    runtime.register_static_plugins([zrtl_io::static_plugin(), zrtl_string::static_plugin()])
}

/// Parse Python source and rewrite it into a `TypedProgram`.
pub fn parse_program(source: &str) -> Result<TypedProgram> {
    let parsed = ruff_python_parser::parse_module(source)
        .map_err(|e| Error::Syntax(format!("{} at {:?}", e.error, e.location)))?;
    if let Some(first) = parsed.errors().first() {
        return Err(Error::Syntax(format!(
            "{} at {:?}",
            first.error, first.location
        )));
    }
    let mut module = parsed.into_syntax();
    // The prelude's declarations come first.
    let prelude = ruff_python_parser::parse_module(prelude::SOURCE)
        .expect("the prelude parses")
        .into_syntax();
    let mut body = prelude.body;
    body.append(&mut module.body);
    module.body = body;

    // A module's body is the program. Statements outside any `def` run
    // top to bottom when the module is executed, so they become the
    // body of the entry point, in order.
    let mut defs: Vec<&py::StmtFunctionDef> = Vec::new();
    let mut top_level: Vec<&py::Stmt> = Vec::new();
    for stmt in &module.body {
        match stmt {
            py::Stmt::FunctionDef(f) => {
                if f.name.as_str() == ENTRY {
                    return Err(Error::Unsupported {
                        what: format!(
                            "a function named `{ENTRY}`; the module body is the program's entry"
                        ),
                        at: f.range().start().to_usize(),
                    });
                }
                defs.push(f);
            }
            // A module docstring declares nothing and runs nothing.
            py::Stmt::Expr(e) if matches!(*e.value, py::Expr::StringLiteral(_)) => {}
            py::Stmt::Pass(_) => {}
            // Classes are declarations; their methods are functions.
            py::Stmt::ClassDef(_) => {}
            other => top_level.push(other),
        }
    }
    let class_defs = classes::collect(&module.body)?;
    let (class_infos, class_index) = classes::skeletons(&class_defs)?;
    let mut items: Vec<types::Item<'_>> = defs
        .iter()
        .map(|f| types::Item {
            name: f.name.to_string(),
            class: None,
            def: f,
        })
        .collect();
    for (k, def) in class_defs.iter().enumerate() {
        for m in &def.methods {
            items.push(types::Item {
                name: types::method_fn(&def.name, m.name.as_str()),
                class: Some(k),
                def: m,
            });
        }
    }

    let mut library = zyntax_builtins::library(&POLICY);
    lower::set_list_type(library.list_type);
    let owned: Vec<py::Stmt> = top_level.iter().map(|s| (*s).clone()).collect();
    let entry_sig = types::Sig {
        params: Vec::new(),
        ret: types::Ty::None,
        defaults: Vec::new(),
    };
    let mut inferred = types::Module {
        list_type: Some(library.list_type),
        classes: class_infos,
        class_index,
        fallible: library.fallible.clone(),
        ..Default::default()
    };
    let global_names = module_globals(&module.body, &defs, &inferred.class_index);
    // A global's type is the join of every assignment to it: the
    // module's own, then those under `global` in each function.
    let main_locals = types::infer_locals(&inferred, &entry_sig, &owned);
    for name in &global_names {
        let ty = main_locals
            .vars
            .get(name)
            .copied()
            .unwrap_or(types::Ty::Unknown);
        inferred.globals.insert(name.clone(), ty);
    }
    for _ in 0..4 {
        let before = inferred.globals.clone();
        let (funcs, class_infos) = types::infer_module(&inferred, &items);
        inferred.funcs = funcs;
        inferred.classes = class_infos;
        for item in &items {
            let sig = inferred.funcs[&item.name].clone();
            let locals = types::infer_locals(&inferred, &sig, &item.def.body);
            for (name, ty) in &locals.global_writes {
                let joined = inferred
                    .globals
                    .get(name)
                    .copied()
                    .unwrap_or(types::Ty::Unknown)
                    .join(*ty);
                inferred.globals.insert(name.clone(), joined);
            }
        }
        if inferred.globals == before {
            break;
        }
    }
    for ty in inferred.globals.values_mut() {
        if *ty == types::Ty::Unknown {
            *ty = types::Ty::Object;
        }
    }
    let mut declarations = classes::register(&mut inferred, &mut library.type_registry);
    // The exception in flight.
    declarations.push(TypedNode::new(
        TypedDeclaration::Variable(TypedVariable {
            name: intern(lower::PENDING),
            ty: Type::Any,
            mutability: Mutability::Mutable,
            initializer: None,
            visibility: Visibility::Public,
        }),
        Type::Unknown,
        Span::new(0, 0),
    ));
    for (name, ty) in &inferred.globals {
        let stored = match ty {
            types::Ty::Int | types::Ty::Float | types::Ty::Bool | types::Ty::Str => *ty,
            _ => types::Ty::Object,
        };
        declarations.push(TypedNode::new(
            TypedDeclaration::Variable(TypedVariable {
                name: intern(name),
                ty: lower::ir(stored),
                mutability: Mutability::Mutable,
                initializer: None,
                visibility: Visibility::Public,
            }),
            Type::Unknown,
            Span::new(0, 0),
        ));
    }
    for item in &items {
        let sig = inferred.funcs[&item.name].clone();
        let locals = types::infer_locals(&inferred, &sig, &item.def.body);
        let scope = scope::Scope::of_function(item.def);
        let mut lowerer = lower::Lowerer::new(
            &inferred,
            &item.name,
            sig,
            locals,
            &scope,
            Vec::new(),
            std::collections::HashMap::new(),
        );
        lowerer.class = item.class;
        let func = lowerer.function_named(item.def, &item.name)?;
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(func),
            Type::Unknown,
            span_of(item.def),
        ));
    }
    if !top_level.is_empty() {
        let mut locals = types::infer_locals(&inferred, &entry_sig, &owned);
        for name in inferred.globals.keys() {
            locals.vars.remove(name);
        }
        let scope = scope::Scope::of_body(Vec::new(), &owned);
        let statements = lower::Lowerer::new(
            &inferred,
            ENTRY,
            entry_sig,
            locals,
            &scope,
            Vec::new(),
            std::collections::HashMap::new(),
        )
        .entry_body(&top_level)?;
        let span = Span::new(
            top_level[0].range().start().to_usize(),
            top_level[top_level.len() - 1].range().end().to_usize(),
        );
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(TypedFunction {
                name: intern(ENTRY),
                annotations: Vec::new(),
                effects: Vec::new(),
                with_handlers: Vec::new(),
                type_params: Vec::new(),
                params: Vec::new(),
                return_type: prim(PrimitiveType::Unit),
                body: Some(TypedBlock { statements, span }),
                visibility: Visibility::Public,
                is_async: false,
                is_fiber: false,
                is_pure: false,
                is_external: false,
                calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
                link_name: None,
                module: None,
            }),
            Type::Unknown,
            span,
        ));
    }

    for name in inferred.adapters.borrow().iter() {
        let sig = &inferred.funcs[name];
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(lower::adapter(&inferred, name, sig)),
            Type::Unknown,
            Span::new(0, 0),
        ));
    }
    for func in inferred.lifted.take() {
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(func),
            Type::Unknown,
            Span::new(0, 0),
        ));
    }
    declarations.push(TypedNode::new(
        TypedDeclaration::Function(classes::raise_hook(&inferred)),
        Type::Unknown,
        Span::new(0, 0),
    ));
    for func in classes::generated(&inferred) {
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(func),
            Type::Unknown,
            Span::new(0, 0),
        ));
    }
    declarations.extend(library.declarations);

    Ok(TypedProgram {
        declarations,
        language: Some(intern("python")),
        span: Span::new(0, source.len()),
        source_files: Vec::new(),
        type_registry: library.type_registry,
    })
}

/// The module-level names that are variables of the module rather than
/// locals of its body: those a function reads or declares `global`.
fn module_globals(
    body: &[py::Stmt],
    defs: &[&py::StmtFunctionDef],
    classes: &std::collections::HashMap<String, usize>,
) -> Vec<String> {
    let module = scope::Scope::of_body(Vec::new(), body);
    let mut functions: std::collections::HashSet<&str> =
        defs.iter().map(|f| f.name.as_str()).collect();
    functions.extend(classes.keys().map(|k| k.as_str()));
    let mut names: std::collections::BTreeSet<String> =
        module.declared_globals().into_iter().collect();
    for (_, child) in &module.children {
        for name in &child.free {
            if module.bound.contains(name) && !functions.contains(name.as_str()) {
                names.insert(name.clone());
            }
        }
    }
    names.into_iter().collect()
}

pub(crate) fn intern(s: &str) -> InternedString {
    InternedString::new_global(s)
}

pub(crate) fn span_of<N: Ranged>(node: &N) -> Span {
    let r = node.range();
    Span::new(r.start().to_usize(), r.end().to_usize())
}

pub(crate) fn prim(p: PrimitiveType) -> Type {
    Type::Primitive(p)
}
