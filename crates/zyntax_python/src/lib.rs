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
mod host;
mod lower;
mod modules;
mod prelude;
mod scope;
mod shape;
mod stdlib;
mod types;

/// Why a program could not be turned into a `TypedProgram`.
///
/// Each carries where in the source it happened, as byte offsets into
/// the main file, or into the module named by `module` when one is.
/// [`Error::render`] shows it against the source.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Python that does not parse.
    #[error("Python syntax error: {message} at {}..{}{}", span.0, span.1, in_module(module))]
    Syntax {
        message: String,
        span: (usize, usize),
        module: Option<String>,
    },
    /// Something Python allows that this frontend does not compile yet.
    /// Says what it was and where, so the gap is a fact rather than a
    /// guess.
    #[error("{what} is not supported yet (at byte offset {}{})", span.0, in_module(module))]
    Unsupported {
        what: String,
        span: (usize, usize),
        module: Option<String>,
    },
    /// The built-in library this crate was built with cannot be read.
    #[error("the built-in library is unreadable: {0}")]
    Library(String),
}

fn in_module(module: &Option<String>) -> String {
    match module {
        Some(m) => format!(" in module `{m}`"),
        None => String::new(),
    }
}

impl Error {
    pub(crate) fn syntax(message: impl Into<String>, at: ruff_text_size::TextRange) -> Self {
        Error::Syntax {
            message: message.into(),
            span: (at.start().to_usize(), at.end().to_usize()),
            module: None,
        }
    }

    pub(crate) fn unsupported(what: impl Into<String>, at: &impl Ranged) -> Self {
        let range = at.range();
        Error::Unsupported {
            what: what.into(),
            span: (range.start().to_usize(), range.end().to_usize()),
            module: None,
        }
    }

    pub(crate) fn unsupported_span(what: impl Into<String>, span: Span) -> Self {
        Error::Unsupported {
            what: what.into(),
            span: (span.start, span.end),
            module: None,
        }
    }

    /// The same error, located in the named module rather than the main
    /// file.
    pub(crate) fn in_module(mut self, name: &str) -> Self {
        match &mut self {
            Error::Syntax { module, .. } | Error::Unsupported { module, .. } => {
                if module.is_none() {
                    *module = Some(name.to_string());
                }
            }
            Error::Library(_) => {}
        }
        self
    }

    /// The module the error is located in, when it is not the main file.
    pub fn module(&self) -> Option<&str> {
        match self {
            Error::Syntax { module, .. } | Error::Unsupported { module, .. } => module.as_deref(),
            Error::Library(_) => None,
        }
    }

    /// The error shown against its source, the way the compiler shows
    /// its own diagnostics. `file` names the source the error's span
    /// refers to: the main file, or the module [`Error::module`] names.
    pub fn render(&self, file: &str, source: &str, use_colors: bool) -> String {
        use zyntax_typed_ast::diagnostics::{Diagnostic, render_diagnostic};
        let (message, label, span) = match self {
            Error::Syntax { message, span, .. } => {
                (format!("syntax error: {message}"), "here", *span)
            }
            Error::Unsupported { what, span, .. } => (
                format!("{what} is not supported yet"),
                "this frontend does not compile this form",
                *span,
            ),
            Error::Library(message) => {
                return format!("error: the built-in library is unreadable: {message}\n");
            }
        };
        // A span has to cover something to be shown; an empty one at the
        // end of the file is drawn on its last byte.
        let end = source.len().max(1);
        let start = span.0.min(end - 1);
        let stop = span.1.max(start + 1).min(end);
        let diagnostic = Diagnostic::error(message).with_primary(Span::new(start, stop), label);
        render_diagnostic(&diagnostic, file, source, use_colors)
    }
}

type Result<T> = std::result::Result<T, Error>;

/// The function a module's top-level statements become. A host runs a
/// Python program by calling this.
pub const ENTRY: &str = "__main__";

mod policy;
pub use host::set_args;
use policy::LIBRARY_MODULE;
pub use policy::POLICY;

/// The built-in library, declared and lowered when this crate was
/// built. A program imports it; the runtime links the HIR and lowers
/// only the program.
const SNAPSHOT: &[u8] = zyntax_embed::include_snapshot!("python");

mod fallible {
    include!(concat!(env!("OUT_DIR"), "/fallible.rs"));
}

/// The snapshot, decoded once per process.
fn snapshot() -> Result<std::sync::Arc<zyntax_embed::Snapshot>> {
    static SNAPSHOT_ONCE: std::sync::OnceLock<
        std::result::Result<std::sync::Arc<zyntax_embed::Snapshot>, String>,
    > = std::sync::OnceLock::new();
    SNAPSHOT_ONCE
        .get_or_init(|| {
            zyntax_embed::Snapshot::load(SNAPSHOT)
                .map(std::sync::Arc::new)
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(Error::Library)
}

/// What the frontend needs to know about the library: the registry its
/// types live in, which type is `List<T>`, and which functions raise.
struct Library {
    type_registry: zyntax_typed_ast::TypeRegistry,
    list_type: zyntax_typed_ast::TypeId,
    fallible: std::collections::BTreeSet<String>,
}

fn library() -> Result<Library> {
    let trace = std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some();
    let t0 = std::time::Instant::now();
    let snapshot = snapshot()?;
    let t1 = std::time::Instant::now();
    let module = snapshot
        .module(LIBRARY_MODULE)
        .map_err(|e| Error::Library(e.to_string()))?
        .ok_or_else(|| Error::Library(format!("the snapshot has no `{LIBRARY_MODULE}`")))?;
    let t2 = std::time::Instant::now();
    let program = module.program();
    let t3 = std::time::Instant::now();
    let type_registry = program.type_registry.clone();
    if trace {
        eprintln!(
            "[LIBRARY] snapshot load {:.2} ms, module {:.2} ms, program {:.2} ms, registry clone {:.2} ms",
            (t1 - t0).as_secs_f64() * 1000.0,
            (t2 - t1).as_secs_f64() * 1000.0,
            (t3 - t2).as_secs_f64() * 1000.0,
            t3.elapsed().as_secs_f64() * 1000.0
        );
    }
    let list_type = type_registry
        .get_type_by_name(intern("List"))
        .map(|def| def.id)
        .ok_or_else(|| Error::Library("the library declares no List type".to_string()))?;
    Ok(Library {
        type_registry,
        list_type,
        fallible: fallible::FALLIBLE.iter().map(|s| s.to_string()).collect(),
    })
}

/// Give a runtime what a compiled Python program links against: the IO,
/// string and math plugins the library's primitives come from, the
/// host's own symbols, and the name a program is entered through, so
/// only the library the program reaches is built. A host calls this
/// once before compiling a program, after [`set_args`] if it has any.
pub fn register_runtime(
    runtime: &mut zyntax_embed::TieredRuntime,
) -> std::result::Result<(), zyntax_embed::RuntimeError> {
    // No Python program releases anything itself, so the compiler
    // releases what it can prove dead and the collector takes the rest.
    runtime.set_automatic_release(true);
    runtime.set_collector(zyntax_embed::Collector::MarkSweep);
    let snapshot = snapshot().map_err(|e| zyntax_embed::RuntimeError::Execution(e.to_string()))?;
    runtime.install_snapshot(snapshot)?;
    runtime.declare_entry_points([ENTRY]);
    runtime.register_static_plugins([
        zrtl_io::static_plugin(),
        zrtl_string::static_plugin(),
        zrtl_math::static_plugin(),
        host::static_plugin(),
    ])
}

/// Parse Python source and rewrite it into a `TypedProgram`. A program
/// that imports its own modules needs [`parse_program_with`].
pub fn parse_program(source: &str) -> Result<TypedProgram> {
    parse_program_with(source, "<python>", &|_| None)
}

/// [`parse_program`] for a program of several files: `modules` finds
/// the source of a module by its dotted name. Each module's body runs
/// once, ahead of the file importing it, and its names are the
/// module's own. `file` names the main source in diagnostics.
pub fn parse_program_with(
    source: &str,
    file: &str,
    modules: &modules::Resolver<'_>,
) -> Result<TypedProgram> {
    let parsed = ruff_python_parser::parse_module(source)
        .map_err(|e| Error::syntax(e.error.to_string(), e.location))?;
    if let Some(first) = parsed.errors().first() {
        return Err(Error::syntax(first.error.to_string(), first.location));
    }
    types::reset_tuple_shapes();
    let mut module = parsed.into_syntax();
    let main: Vec<py::Stmt> = std::mem::take(&mut module.body).into_iter().collect();
    let linked = modules::link(main, modules)?;
    // The program's source files: the main file first, then each module
    // in the order it was loaded; a span names its file by that index.
    let mut source_files = vec![zyntax_typed_ast::source::SourceFile::new(
        file.to_string(),
        source.to_string(),
    )];
    let mut files: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for (name, text) in linked.modules {
        files.insert(name.clone(), source_files.len() as u32);
        source_files.push(zyntax_typed_ast::source::SourceFile::new(name, text));
    }
    // The prelude's declarations come first. `origins` names, for each
    // statement of the body, the program's module it came from.
    let prelude = ruff_python_parser::parse_module(prelude::SOURCE)
        .expect("the prelude parses")
        .into_syntax();
    let mut origins: Vec<Option<String>> = vec![None; prelude.body.len()];
    let mut body = prelude.body;
    for (stmt, origin) in linked.statements {
        body.push(stmt);
        origins.push(origin);
    }
    module.body = body;
    let located = |e: Error, module: Option<&str>| match module {
        Some(m) => e.in_module(m),
        None => e,
    };

    // A module's body is the program. Statements outside any `def` run
    // top to bottom when the module is executed, so they become the
    // body of the entry point, in order.
    let mut defs: Vec<(&py::StmtFunctionDef, Option<&str>)> = Vec::new();
    let mut top_level: Vec<(&py::Stmt, Option<&str>)> = Vec::new();
    for (stmt, origin) in module.body.iter().zip(origins.iter()) {
        let origin = origin.as_deref();
        match stmt {
            py::Stmt::FunctionDef(f) => {
                if f.name.as_str() == ENTRY {
                    return Err(located(
                        Error::unsupported(
                            format!(
                                "a function named `{ENTRY}`; the module body is the program's entry"
                            ),
                            &f,
                        ),
                        origin,
                    ));
                }
                defs.push((f, origin));
            }
            // A module docstring declares nothing and runs nothing.
            py::Stmt::Expr(e) if matches!(*e.value, py::Expr::StringLiteral(_)) => {}
            py::Stmt::Pass(_) => {}
            // Classes are declarations; their methods are functions.
            py::Stmt::ClassDef(_) => {}
            other => top_level.push((other, origin)),
        }
    }
    let class_defs = classes::collect(&module.body, &origins)?;
    let (class_infos, class_index) = classes::skeletons(&class_defs)?;
    let mut items: Vec<types::Item<'_>> = defs
        .iter()
        .map(|(f, origin)| types::Item {
            name: f.name.to_string(),
            class: None,
            def: f,
            module: origin.map(str::to_string),
        })
        .collect();
    for def in &class_defs {
        for m in &def.methods {
            items.push(types::Item {
                name: types::method_fn(&def.name, m.name.as_str()),
                class: Some(class_index[&def.name]),
                def: m,
                module: def.module.clone(),
            });
        }
    }

    let mut library = library()?;
    lower::set_list_type(library.list_type);
    let owned: Vec<py::Stmt> = top_level.iter().map(|(s, _)| (*s).clone()).collect();
    let entry_sig = types::Sig {
        params: Vec::new(),
        ret: types::Ty::None,
        defaults: Vec::new(),
    };
    let (imports, from_names) = collect_imports(&module.body)?;
    let mut inferred = types::Module {
        list_type: Some(library.list_type),
        classes: class_infos,
        class_index,
        fallible: library.fallible.clone(),
        name: ENTRY.to_string(),
        imports,
        from_names,
        files,
        ..Default::default()
    };
    let def_stmts: Vec<&py::StmtFunctionDef> = defs.iter().map(|(f, _)| *f).collect();
    let global_names = module_globals(&module.body, &def_stmts, &inferred.class_index);
    inferred.closed = types::closed_items(&module.body, &items);
    // Every lambda and nested def, so a call through a value of one is
    // a direct call wherever the value's type is known.
    let class_index = inferred.class_index.clone();
    let entry_files: Vec<u32> = top_level
        .iter()
        .map(|(_, origin)| inferred.file_of(*origin))
        .collect();
    for item in &items {
        let file = inferred.file_of(item.module.as_deref());
        let scope = scope::Scope::of_function(item.def);
        let mut visible = scope.bound.clone();
        visible.extend(
            item.def
                .parameters
                .iter_non_variadic_params()
                .map(|p| p.parameter.name.to_string()),
        );
        types::collect_closures(
            &mut inferred,
            &item.name,
            file,
            &item.def.body,
            &class_index,
            visible,
        );
        let params: Vec<String> = item
            .def
            .parameters
            .iter_non_variadic_params()
            .map(|p| p.parameter.name.to_string())
            .collect();
        types::collect_bound_methods(&mut inferred, file, &item.def.body, &params);
    }
    let entry_scope = scope::Scope::of_body(Vec::new(), &owned);
    for (stmt, file) in owned.iter().zip(&entry_files) {
        types::collect_closures(
            &mut inferred,
            ENTRY,
            *file,
            std::slice::from_ref(stmt),
            &class_index,
            entry_scope.bound.clone(),
        );
    }
    // Names the lowering makes up start past the closures' indices.
    inferred.counter.set(inferred.closures.borrow().len());
    // A global's type is the join of every assignment to it: the
    // module's own, then those under `global` in each function. The
    // module's own are retyped each round, as the functions they call
    // become known; nothing settles as dynamic before the end.
    for name in &global_names {
        inferred.globals.insert(name.clone(), types::Ty::Unknown);
    }
    // Each round infers the module afresh from the declared layouts,
    // against the globals and the list-parameter facts the last round
    // settled; nothing else carries over, so what one round decided
    // from less than it knows does not bind the next.
    // Methods called on unknown receivers are found only once the
    // globals and the list facts have settled with every method typed
    // by its calls: opening a method makes more receivers unknown,
    // never fewer, so the set only grows from there, and a set taken
    // earlier would open methods on account of what was not yet known.
    let declared_classes = inferred.classes.clone();
    let mut methods_settling = false;
    for _ in 0..12 {
        let before = (
            inferred.globals.clone(),
            inferred.list_params.clone(),
            inferred.dynamic_methods.clone(),
            inferred.field_lists.clone(),
        );
        inferred.classes = declared_classes.clone();
        let out = types::infer_module(&inferred, &items, &owned, &entry_files);
        inferred.funcs = out.funcs;
        inferred.classes = out.classes;
        inferred.closures = std::cell::RefCell::new(out.closures);
        inferred.list_params = out.list_params;
        let found_dynamic = out.dynamic_methods.clone();
        if methods_settling {
            inferred
                .dynamic_methods
                .extend(found_dynamic.iter().cloned());
        }
        inferred.list_fields = out.list_fields;
        inferred.field_lists = out.field_lists;
        if std::env::var_os("ZYNTAX_TRACE_TYPES").is_some() {
            let mut dynamic: Vec<&String> = inferred.dynamic_methods.iter().collect();
            dynamic.sort();
            eprintln!(
                "[types] round: dynamic methods {}",
                dynamic
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            if let Ok(watch) = std::env::var("ZYNTAX_TRACE_TYPES_WATCH") {
                for name in watch.split(',') {
                    if let Some(sig) = inferred.funcs.get(name) {
                        eprintln!("[types] round: {name} {:?} -> {:?}", sig.params, sig.ret);
                    }
                    for class in &inferred.classes {
                        if class.name == name {
                            eprintln!("[types] round: class {name} {:?}", class.fields);
                        }
                    }
                }
            }
        }
        let mut writes: Vec<(String, types::Ty)> = global_names
            .iter()
            .map(|name| {
                let ty = out
                    .entry
                    .vars
                    .get(name)
                    .copied()
                    .unwrap_or(types::Ty::Unknown);
                (name.clone(), ty)
            })
            .collect();
        for item in &items {
            let sig = inferred.funcs[&item.name].clone();
            let file = inferred.file_of(item.module.as_deref());
            let locals = types::in_file(file, || {
                types::infer_locals(&inferred, &sig, &item.def.body)
            });
            writes.extend(locals.global_writes.iter().map(|(n, t)| (n.clone(), *t)));
        }
        for (name, ty) in writes {
            let joined = inferred
                .globals
                .get(&name)
                .copied()
                .unwrap_or(types::Ty::Unknown)
                .join(ty);
            inferred.globals.insert(name, joined);
        }
        let settled = (
            inferred.globals.clone(),
            inferred.list_params.clone(),
            inferred.dynamic_methods.clone(),
            inferred.field_lists.clone(),
        ) == before;
        if settled && methods_settling {
            break;
        }
        if settled {
            methods_settling = true;
            inferred.dynamic_methods = found_dynamic;
        }
    }
    inferred.settled.set(true);
    for ty in inferred.globals.values_mut() {
        *ty = ty.settled();
    }
    // `ZYNTAX_TRACE_TYPES=1` prints what inference decided: each
    // function's signature, each class's fields, the globals.
    if std::env::var_os("ZYNTAX_TRACE_TYPES").is_some() {
        let mut names: Vec<&String> = inferred.funcs.keys().collect();
        names.sort();
        for name in names {
            let sig = &inferred.funcs[name];
            let params: Vec<String> = sig
                .params
                .iter()
                .map(|(n, t)| format!("{n}: {t:?}"))
                .collect();
            eprintln!("[types] {name}({}) -> {:?}", params.join(", "), sig.ret);
        }
        for class in &inferred.classes {
            let fields: Vec<String> = class
                .fields
                .iter()
                .map(|(n, t)| format!("{n}: {t:?}"))
                .collect();
            eprintln!("[types] class {} {{ {} }}", class.name, fields.join(", "));
        }
        let mut globals: Vec<(&String, &types::Ty)> = inferred.globals.iter().collect();
        globals.sort_by(|a, b| a.0.cmp(b.0));
        for (name, ty) in globals {
            eprintln!("[types] global {name}: {ty:?}");
        }
        if !inferred.dynamic_methods.is_empty() {
            let mut dynamic: Vec<&String> = inferred.dynamic_methods.iter().collect();
            dynamic.sort();
            eprintln!(
                "[types] dynamic methods: {}",
                dynamic
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
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
        let stored = lower::Lowerer::storage(*ty);
        declarations.push(TypedNode::new(
            TypedDeclaration::Variable(TypedVariable {
                name: lower::global_symbol(name),
                ty: lower::ir(stored),
                mutability: Mutability::Mutable,
                initializer: None,
                visibility: Visibility::Public,
            }),
            Type::Unknown,
            Span::new(0, 0),
        ));
    }
    // Which items get a variant that trusts its instance-typed
    // parameters, and which never return None, are read at every call
    // site, so both are settled before any body is lowered.
    inferred.trusted = items
        .iter()
        .filter(|item| {
            inferred
                .funcs
                .get(&item.name)
                .is_some_and(|sig| types::has_instance_params(sig, item.class.is_some()))
        })
        .map(|item| item.name.clone())
        .collect();
    inferred.returns_instance = types::returning_instances(&inferred, &items);
    // The functions are lowered twice. The first time teaches which of
    // them can raise; the second time, a call to one that never does is
    // not followed by a check. Only the second lowering is kept.
    let unpack_shapes = shape::infer(&inferred, &items, &owned);
    lower_items(&inferred, &items, &unpack_shapes)?;
    let mut facts = inferred.raise_facts.take();
    classes::raise_facts(&inferred, &mut facts);
    inferred.non_raising = types::non_raising(&facts);
    inferred.lifted.take();
    inferred.adapters.take();
    inferred.attr_reads.take();
    inferred.attr_writes.take();
    inferred.dyn_methods.take();
    inferred.counter.set(inferred.closures.borrow().len());
    declarations.extend(lower_items(&inferred, &items, &unpack_shapes)?);
    if !top_level.is_empty() {
        let mut locals = types::infer_locals_entry(&inferred, &entry_sig, &owned, &entry_files);
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
            top_level[0].0.range().start().to_usize(),
            top_level[top_level.len() - 1].0.range().end().to_usize(),
        );
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(TypedFunction {
                name: intern(ENTRY),
                annotations: vec![lower::strict_fp_annotation(span)],
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
    for mut func in inferred.lifted.take() {
        if !func
            .annotations
            .iter()
            .any(|a| a.name.resolve_global().as_deref() == Some("strict_fp"))
        {
            func.annotations
                .push(lower::strict_fp_annotation(Span::new(0, 0)));
        }
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
    // The library itself arrives by import: its declarations for
    // typing, its HIR to link against.
    declarations.push(TypedNode::new(
        TypedDeclaration::Import(zyntax_typed_ast::typed_ast::TypedImport {
            language: Some(intern("python")),
            module_path: vec![intern(LIBRARY_MODULE)],
            items: Vec::new(),
            span: Span::new(0, 0),
        }),
        Type::Unknown,
        Span::new(0, 0),
    ));

    Ok(TypedProgram {
        declarations,
        language: Some(intern("python")),
        span: Span::new(0, source.len()),
        source_files,
        type_registry: library.type_registry,
    })
}

/// Module aliases to modules, and local names to the module and member
/// they were imported from.
type Imports = std::collections::HashMap<String, String>;
type FromNames = std::collections::HashMap<String, (String, String)>;

/// Every `import` in the program, wherever it appears: the modules go
/// by their aliases, the names brought in by `from` by theirs. A module
/// this frontend does not know is refused here, before anything is
/// lowered.
fn collect_imports(body: &[py::Stmt]) -> Result<(Imports, FromNames)> {
    let mut imports = Imports::new();
    let mut from_names = FromNames::new();
    fn walk(stmts: &[py::Stmt], imports: &mut Imports, from_names: &mut FromNames) -> Result<()> {
        for s in stmts {
            match s {
                py::Stmt::Import(i) => {
                    for alias in &i.names {
                        let module = alias.name.id.as_str();
                        if !stdlib::is_known(module) {
                            return Err(Error::unsupported(
                                format!("import of module `{module}`"),
                                &alias,
                            ));
                        }
                        let local = alias
                            .asname
                            .as_ref()
                            .map(|a| a.id.to_string())
                            .unwrap_or_else(|| module.to_string());
                        imports.insert(local, module.to_string());
                    }
                }
                py::Stmt::ImportFrom(f) => {
                    let Some(module) = f.module.as_ref().map(|m| m.id.as_str()) else {
                        return Err(Error::unsupported("a relative import".to_string(), &f));
                    };
                    if !stdlib::is_known(module) {
                        return Err(Error::unsupported(
                            format!("import of module `{module}`"),
                            &f,
                        ));
                    }
                    for alias in &f.names {
                        let name = alias.name.id.as_str();
                        if name == "*" {
                            return Err(Error::unsupported(
                                format!("`from {module} import *`"),
                                &alias,
                            ));
                        }
                        // typing's names are annotations, not values;
                        // __future__'s are the language as it is.
                        if module == "typing" {
                            if !stdlib::is_typing_name(name) {
                                return Err(Error::unsupported(format!("`typing.{name}`"), &alias));
                            }
                            continue;
                        }
                        if module == "__future__" {
                            if !stdlib::is_future_name(name) {
                                return Err(Error::unsupported(
                                    format!("`__future__.{name}`"),
                                    &alias,
                                ));
                            }
                            continue;
                        }
                        if stdlib::member(module, name).is_none() {
                            return Err(Error::unsupported(format!("`{module}.{name}`"), &alias));
                        }
                        let local = alias
                            .asname
                            .as_ref()
                            .map(|a| a.id.to_string())
                            .unwrap_or_else(|| name.to_string());
                        from_names.insert(local, (module.to_string(), name.to_string()));
                    }
                }
                py::Stmt::FunctionDef(d) => walk(&d.body, imports, from_names)?,
                py::Stmt::ClassDef(c) => walk(&c.body, imports, from_names)?,
                py::Stmt::If(i) => {
                    walk(&i.body, imports, from_names)?;
                    for clause in &i.elif_else_clauses {
                        walk(&clause.body, imports, from_names)?;
                    }
                }
                py::Stmt::For(f) => {
                    walk(&f.body, imports, from_names)?;
                    walk(&f.orelse, imports, from_names)?;
                }
                py::Stmt::While(w) => {
                    walk(&w.body, imports, from_names)?;
                    walk(&w.orelse, imports, from_names)?;
                }
                py::Stmt::Try(t) => {
                    walk(&t.body, imports, from_names)?;
                    for h in &t.handlers {
                        let py::ExceptHandler::ExceptHandler(h) = h;
                        walk(&h.body, imports, from_names)?;
                    }
                    walk(&t.orelse, imports, from_names)?;
                    walk(&t.finalbody, imports, from_names)?;
                }
                py::Stmt::With(w) => walk(&w.body, imports, from_names)?,
                _ => {}
            }
        }
        Ok(())
    }
    walk(body, &mut imports, &mut from_names)?;
    Ok((imports, from_names))
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

/// The span of a node, in the file being lowered.
/// Lower every function of the program, recording what each one's
/// lowering found about its raising in the module's table.
fn lower_items(
    inferred: &types::Module,
    items: &[types::Item<'_>],
    unpack_shapes: &std::collections::HashMap<String, std::collections::HashMap<String, types::Ty>>,
) -> Result<Vec<TypedNode<TypedDeclaration>>> {
    let mut declarations = Vec::with_capacity(items.len());
    for item in items {
        // The item itself, and the variant trusting its instance-typed
        // parameters where it has any.
        let mut variants = vec![(item.name.clone(), false)];
        if inferred.trusted.contains(&item.name) {
            variants.push((types::trusted_name(&item.name), true));
        }
        for (name, trusted) in variants {
            let sig = inferred.funcs[&item.name].clone();
            lower::set_current_file(inferred.file_of(item.module.as_deref()));
            let mut locals = types::infer_locals(inferred, &sig, &item.def.body);
            if let Some(shapes) = unpack_shapes.get(&item.name) {
                for (name, ty) in shapes {
                    if !sig.params.iter().any(|(param, _)| param == name)
                        && locals.vars.get(name) == Some(&types::Ty::Object)
                    {
                        locals.vars.insert(name.clone(), *ty);
                    }
                }
            }
            let scope = scope::Scope::of_function(item.def);
            let mut lowerer = lower::Lowerer::new(
                inferred,
                &item.name,
                sig,
                locals,
                &scope,
                Vec::new(),
                std::collections::HashMap::new(),
            );
            lowerer.class = item.class;
            lowerer.trusted = trusted;
            let func = lowerer.function_named(item.def, &name).map_err(|e| {
                match item.module.as_deref() {
                    Some(m) => e.in_module(m),
                    None => e,
                }
            })?;
            inferred
                .raise_facts
                .borrow_mut()
                .insert(name, lowerer.raise_fact());
            declarations.push(TypedNode::new(
                TypedDeclaration::Function(func),
                Type::Unknown,
                span_of(item.def),
            ));
            lower::set_current_file(0);
        }
    }
    Ok(declarations)
}

pub(crate) fn span_of<N: Ranged>(node: &N) -> Span {
    let r = node.range();
    Span::in_file(
        r.start().to_usize(),
        r.end().to_usize(),
        lower::current_file(),
    )
}

pub(crate) fn prim(p: PrimitiveType) -> Type {
    Type::Primitive(p)
}
