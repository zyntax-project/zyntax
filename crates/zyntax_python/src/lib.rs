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
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{TypedBlock, TypedDeclaration, TypedFunction, TypedVariable};
use zyntax_typed_ast::{
    InternedString, Mutability, PrimitiveType, Type, TypedNode, TypedProgram, Visibility,
};

mod aliases;
mod bytes;
mod class_attrs;
mod classes;
mod format;
mod host;
mod kwargs;
mod lower;
mod modules;
mod prelude;
mod rebind;
mod scope;
mod shape;
mod stdlib;
mod sugar;
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

/// The name of the prelude's file in a program's `source_files`. A
/// declaration whose span names it is the built-in library's, not the
/// program's.
pub const PRELUDE: &str = "<prelude>";

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
    // Programs declare no effects or handlers, and lower the same
    // without the structural cleanup.
    runtime.set_pattern_rewrites(false);
    let snapshot = snapshot().map_err(|e| zyntax_embed::RuntimeError::Execution(e.to_string()))?;
    runtime.install_snapshot(snapshot)?;
    runtime.declare_entry_points([ENTRY]);
    runtime.register_static_plugins([
        zrtl_io::static_plugin(),
        zrtl_string::static_plugin(),
        zrtl_math::static_plugin(),
        host::static_plugin(),
        zyntax_embed::foreign::static_plugin(),
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
    // `ZYNTAX_TRACE_LOWER_PHASES=1` times the frontend's steps on stderr,
    // the same switch the embedder's phase trace reads.
    let trace = std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some();
    let mut phase = std::time::Instant::now();
    let mut lap = |what: &str| {
        if trace {
            eprintln!(
                "[PY-FRONT] {what:<14} {:8.2} ms",
                phase.elapsed().as_secs_f64() * 1000.0
            );
        }
        phase = std::time::Instant::now();
    };
    let parsed = ruff_python_parser::parse_module(source)
        .map_err(|e| Error::syntax(e.error.to_string(), e.location))?;
    if let Some(first) = parsed.errors().first() {
        return Err(Error::syntax(first.error.to_string(), first.location));
    }
    types::reset_tuple_shapes();
    scope::reset_cache();
    let mut module = parsed.into_syntax();
    let main: Vec<py::Stmt> = std::mem::take(&mut module.body).into_iter().collect();
    let linked = modules::link(main, modules)?;
    // The program's source files: the main file first, then each module
    // in the order it was loaded; a span names its file by that index.
    let mut source_files = vec![zyntax_typed_ast::source::SourceFile::new(
        file.to_string(),
        source.to_string(),
    )];
    let mut files: HashMap<String, u32> = HashMap::default();
    for (name, text) in linked.modules {
        files.insert(name.clone(), source_files.len() as u32);
        source_files.push(zyntax_typed_ast::source::SourceFile::new(name, text));
    }
    // The prelude's declarations come first. `origins` names, for each
    // statement of the body, the program's module it came from.
    let prelude = ruff_python_parser::parse_module(prelude::SOURCE)
        .expect("the prelude parses")
        .into_syntax();
    // The prelude is a file of its own, so a span tells its declarations
    // from the program's.
    files.insert(PRELUDE.to_string(), source_files.len() as u32);
    source_files.push(zyntax_typed_ast::source::SourceFile::new(
        PRELUDE.to_string(),
        prelude::SOURCE.to_string(),
    ));
    let mut origins: Vec<Option<String>> = vec![Some(PRELUDE.to_string()); prelude.body.len()];
    let mut body = prelude.body;
    for (stmt, origin) in linked.statements {
        for stmt in flatten_true_if(stmt) {
            body.push(stmt);
            origins.push(origin.clone());
        }
    }
    // Sugar the frontend does not model is rewritten away before
    // anything is typed: `**kwargs` becomes keyword parameters, class
    // and static methods become module functions, a name given a class
    // becomes the class, a local rebound in straight-line code becomes
    // a fresh version.
    let mut statements: Vec<py::Stmt> = body.into_iter().collect();
    kwargs::rewrite(&mut statements)?;
    sugar::rewrite(&mut statements, &mut origins);
    aliases::rewrite(&mut statements);
    rebind::rewrite(&mut statements);
    module.body = statements.into_iter().collect();
    lap("parse+link");
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
            // A class's methods are functions; what its body declares
            // besides runs where the class statement is.
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

    let bases: Vec<Option<usize>> = class_infos.iter().map(|c| c.base).collect();
    let top_stmts: Vec<&py::Stmt> = top_level.iter().map(|(s, _)| *s).collect();
    let class_attrs = std::sync::Arc::new(class_attrs::collect(
        &class_defs,
        &class_index,
        &bases,
        &top_stmts,
        &items,
    )?);
    lap("classes");
    let mut library = library()?;
    lower::set_list_type(library.list_type);
    lap("library");
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
        class_attrs,
        fallible: library.fallible.clone(),
        name: ENTRY.to_string(),
        imports,
        from_names,
        files,
        file_names: source_files.iter().map(|f| f.name.clone()).collect(),
        ..Default::default()
    };
    let def_stmts: Vec<&py::StmtFunctionDef> = defs.iter().map(|(f, _)| *f).collect();
    let mut global_names = module_globals(&module.body, &def_stmts, &inferred.class_index);
    // A class attribute not fixed as a constant is a module variable.
    global_names.extend(inferred.class_attrs.globals().map(str::to_string));
    inferred.closed = types::closed_items(&module.body, &items);
    inferred.methods = items
        .iter()
        .filter(|item| item.class.is_some())
        .map(|item| item.name.clone())
        .collect();
    inferred.fixed_params = items
        .iter()
        .map(|item| {
            let fixed = item
                .def
                .parameters
                .iter_non_variadic_params()
                .enumerate()
                .map(|(i, p)| p.parameter.annotation.is_some() || (i == 0 && item.class.is_some()))
                .collect();
            (item.name.clone(), fixed)
        })
        .collect();
    // Every lambda and nested def, so a call through a value of one is
    // a direct call wherever the value's type is known.
    let class_index = inferred.class_index.clone();
    let entry_files: Vec<u32> = top_level
        .iter()
        .map(|(_, origin)| inferred.file_of(*origin))
        .collect();
    // Which functions declare a `global`: only those can write one, so
    // only their bodies are re-read for the globals' types each round.
    let mut writes_globals: Vec<bool> = Vec::with_capacity(items.len());
    for item in &items {
        let file = inferred.file_of(item.module.as_deref());
        let scope = scope::Scope::of_function(item.def);
        writes_globals.push(
            !scope.globals.is_empty()
                || (!inferred.class_attrs.is_empty()
                    && types::writes_class_attrs(&inferred.class_index, &item.def.body)),
        );
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
    // become known; nothing settles as dynamic before the end, so a
    // round's value decided from an operand not yet typed is refined
    // by the next round's rather than joined with it.
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
    lap("closures");
    let mut declared_classes = inferred.classes.clone();
    let mut methods_settling = false;
    let mut rounds = 0;
    for _ in 0..12 {
        rounds += 1;
        let before = (
            inferred.globals.clone(),
            inferred.list_params.clone(),
            inferred.dynamic_methods.clone(),
            inferred.field_lists.clone(),
            inferred.dynamic_fields.clone(),
        );
        inferred.classes = declared_classes.clone();
        let out = types::infer_module(&inferred, &items, &owned, &entry_files);
        inferred.funcs = out.funcs;
        inferred.classes = out.classes;
        // A field a round found written on an instance other than
        // `self` (`x.symbol = v` for `x` of a known class) is the class's
        // from then on; its type is decided afresh each round like the
        // others'.
        let mut grew = false;
        for (declared, found) in declared_classes.iter_mut().zip(&inferred.classes) {
            for (name, _) in &found.fields {
                if !declared.fields.iter().any(|(f, _)| f == name) {
                    declared.fields.push((name.clone(), types::Ty::Unknown));
                    grew = true;
                }
            }
        }
        if grew {
            rounds -= 1;
            continue;
        }
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
        inferred.dynamic_fields = out.dynamic_fields;
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
                    .or_else(|| out.entry.global_writes.get(name))
                    .copied()
                    .unwrap_or(types::Ty::Unknown);
                (name.clone(), ty)
            })
            .collect();
        for (item, writes_global) in items.iter().zip(&writes_globals) {
            if !writes_global {
                continue;
            }
            let sig = inferred.funcs[&item.name].clone();
            let file = inferred.file_of(item.module.as_deref());
            let locals = types::in_file(file, || {
                types::infer_locals_open(&inferred, &sig, &item.def.body, &[], false)
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
            inferred.dynamic_fields.clone(),
        ) == before;
        if settled && methods_settling {
            break;
        }
        if settled {
            // The methods round reads the same inputs when this round
            // found no dynamic methods; its answer would be this one.
            if found_dynamic.is_empty() && inferred.dynamic_methods.is_empty() {
                break;
            }
            methods_settling = true;
            inferred.dynamic_methods = found_dynamic;
        }
    }
    inferred.settled.set(true);
    for ty in inferred.globals.values_mut() {
        *ty = ty.settled();
    }
    lap(&format!("infer x{rounds}"));
    types::specialise(&mut inferred, &items, &owned, &entry_files);
    lap("specialise");
    // `ZYNTAX_TRACE_TYPES=1` prints what inference decided: each
    // function's signature and the instances made of it, each class's
    // fields, the globals, and each function's locals as it is lowered.
    if std::env::var_os("ZYNTAX_TRACE_TYPES").is_some() {
        let describe = |sig: &types::Sig| -> String {
            let params: Vec<String> = sig
                .params
                .iter()
                .map(|(n, t)| format!("{n}: {}", t.describe()))
                .collect();
            format!("({}) -> {}", params.join(", "), sig.ret.describe())
        };
        let mut names: Vec<&String> = inferred.funcs.keys().collect();
        names.sort();
        for name in names {
            eprintln!("[types] {name}{}", describe(&inferred.funcs[name]));
            for spec in inferred.specs.iter().filter(|s| s.item == *name) {
                eprintln!("[types]   instance {}{}", spec.name, describe(&spec.sig));
            }
        }
        for class in &inferred.classes {
            let fields: Vec<String> = class
                .fields
                .iter()
                .map(|(n, t)| format!("{n}: {}", t.describe()))
                .collect();
            eprintln!("[types] class {} {{ {} }}", class.name, fields.join(", "));
        }
        let mut globals: Vec<(&String, &types::Ty)> = inferred.globals.iter().collect();
        globals.sort_by(|a, b| a.0.cmp(b.0));
        for (name, ty) in globals {
            eprintln!("[types] global {name}: {}", ty.describe());
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
    // An instance trusts and returns what its item does, for its own
    // signature.
    for spec in &inferred.specs {
        let is_method = inferred.methods.contains(&spec.item);
        if types::has_instance_params(&spec.sig, is_method) {
            inferred.trusted.insert(spec.name.clone());
        }
        if inferred.returns_instance.contains(&spec.item)
            && matches!(spec.sig.ret, types::Ty::Class(_))
        {
            inferred.returns_instance.insert(spec.name.clone());
        }
    }
    // The functions are lowered twice. The first time teaches which of
    // them can raise; the second time, a call to one that never does is
    // not followed by a check. Only the second lowering is kept.
    let unpack_shapes = shape::infer(&inferred, &items, &owned);
    // A module function nothing reaches from the module body is never
    // run by the program, so what it writes need not compile: a Python
    // that never runs it never minds. One that does compile is kept, for
    // a host that calls it by name. A method stays with its class, since
    // the dispatchers over dynamic receivers and the runtime's hooks may
    // name it: one whose name no attribute access spells is replaced by
    // a function that raises the reason, which is what Python does with
    // a body it cannot run.
    let reached = reachable_functions(&owned, &items);
    let attrs_used = attribute_names(&owned, &items);
    let unreached = |item: &types::Item<'_>| match item.class {
        None => !reached.contains(&item.name),
        Some(_) => !attrs_used.contains(item.def.name.as_str()),
    };
    let mut dropped: HashSet<String> = HashSet::default();
    let lower_all = |inferred: &types::Module,
                     dropped: &mut HashSet<String>|
     -> Result<Vec<TypedNode<TypedDeclaration>>> {
        let mut out = Vec::new();
        for item in &items {
            if dropped.contains(&item.name) {
                continue;
            }
            match lower_items(inferred, std::slice::from_ref(item), &unpack_shapes) {
                Ok(d) => out.extend(d),
                // A method stands in for itself as a raise: the class's
                // dispatchers may still name it.
                Err(e) if unreached(item) && item.class.is_some() => {
                    out.extend(lower_stub(inferred, item, &e));
                }
                Err(_) if unreached(item) => {
                    dropped.insert(item.name.clone());
                    continue;
                }
                Err(e) => return Err(e),
            }
            out.extend(lower_specs(inferred, item, &unpack_shapes));
        }
        // What names a dropped function goes with it; nothing reached
        // does, or it would have been reached itself.
        loop {
            let more: Vec<String> = items
                .iter()
                .filter(|i| i.class.is_none() && !reached.contains(&i.name))
                .filter(|i| !dropped.contains(&i.name))
                .filter(|i| reads_any_of(&i.def.body, dropped))
                .map(|i| i.name.clone())
                .collect();
            if more.is_empty() {
                break;
            }
            dropped.extend(more);
        }
        if !dropped.is_empty() && std::env::var_os("ZYNTAX_TRACE_TYPES").is_some() {
            let mut names: Vec<&String> = dropped.iter().collect();
            names.sort();
            eprintln!(
                "[types] not compiled, unreached: {}",
                names
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        Ok(out)
    };
    lower_all(&inferred, &mut dropped)?;
    let mut facts = inferred.raise_facts.take();
    classes::raise_facts(&inferred, &mut facts);
    inferred.non_raising = types::non_raising(&facts);
    inferred.lifted.take();
    inferred.adapters.take();
    inferred.attr_reads.take();
    inferred.attr_writes.take();
    inferred.dyn_methods.take();
    inferred.abstract_calls.take();
    inferred.class_adapters.take();
    inferred.counter.set(inferred.closures.borrow().len());
    // The lowering kept reads dict and set shapes as the first one left
    // them joined.
    types::close_shape_classes();
    let kept = lower_all(&inferred, &mut dropped)?;
    declarations.extend(kept.into_iter().filter(|d| match &d.node {
        TypedDeclaration::Function(f) => f.name.resolve_global().is_none_or(|n| {
            !dropped.contains(&n) && !dropped.contains(n.trim_end_matches("$trusted"))
        }),
        _ => true,
    }));
    // The module body is the entry even when it has no statements: a
    // program is built from its entry, and one without would have the
    // whole library built up front.
    {
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
            HashMap::default(),
        )
        .entry_body(&top_level)?;
        let span = match (top_level.first(), top_level.last()) {
            (Some(first), Some(last)) => Span::new(
                first.0.range().start().to_usize(),
                last.0.range().end().to_usize(),
            ),
            _ => Span::new(0, 0),
        };
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
    for &k in inferred.class_adapters.borrow().iter() {
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(lower::class_adapter(&inferred, k)),
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
    declarations.extend(lower::shape_declarations(&inferred, library.list_type));
    lap("lower");
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
type Imports = HashMap<String, String>;
type FromNames = HashMap<String, (String, String)>;

/// Every `import` in the program, wherever it appears: the modules go
/// by their aliases, the names brought in by `from` by theirs. A module
/// this frontend does not know is refused here, before anything is
/// lowered.
fn collect_imports(body: &[py::Stmt]) -> Result<(Imports, FromNames)> {
    let mut imports = Imports::default();
    let mut from_names = FromNames::default();
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
/// The module functions the module body reaches, through the names
/// read in it, in the methods (every class is compiled), and in each
/// function reached, transitively.
fn reachable_functions(body: &[py::Stmt], items: &[types::Item<'_>]) -> HashSet<String> {
    use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
    #[derive(Default)]
    struct Reads(HashSet<String>);
    impl<'a> Visitor<'a> for Reads {
        fn visit_stmt(&mut self, s: &'a py::Stmt) {
            // A default is evaluated where the function is defined.
            if let py::Stmt::FunctionDef(f) = s {
                for p in f.parameters.iter_non_variadic_params() {
                    if let Some(d) = &p.default {
                        self.visit_expr(d);
                    }
                }
            }
            walk_stmt(self, s);
        }
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Name(n) = e {
                self.0.insert(n.id.to_string());
            }
            walk_expr(self, e);
        }
    }
    let reads_of = |stmts: &[py::Stmt]| {
        let mut r = Reads::default();
        for s in stmts {
            r.visit_stmt(s);
        }
        r.0
    };
    let functions: HashMap<&str, &types::Item<'_>> = items
        .iter()
        .filter(|i| i.class.is_none())
        .map(|i| (i.name.as_str(), i))
        .collect();
    let mut pending: Vec<String> = reads_of(body).into_iter().collect();
    for item in items.iter().filter(|i| i.class.is_some()) {
        pending.extend(reads_of(&item.def.body));
        for p in item.def.parameters.iter_non_variadic_params() {
            if let Some(d) = &p.default {
                let mut r = Reads::default();
                r.visit_expr(d);
                pending.extend(r.0);
            }
        }
    }
    let mut reached: HashSet<String> = HashSet::default();
    while let Some(name) = pending.pop() {
        let Some(item) = functions.get(name.as_str()) else {
            continue;
        };
        if !reached.insert(name) {
            continue;
        }
        pending.extend(reads_of(&item.def.body));
        for p in item.def.parameters.iter_non_variadic_params() {
            if let Some(d) = &p.default {
                let mut r = Reads::default();
                r.visit_expr(d);
                pending.extend(r.0);
            }
        }
    }
    reached
}

/// Every attribute name the program spells, `x.name`, in the module
/// body and in every function.
fn attribute_names(body: &[py::Stmt], items: &[types::Item<'_>]) -> HashSet<String> {
    use ruff_python_ast::visitor::{Visitor, walk_expr};
    #[derive(Default)]
    struct Attrs(HashSet<String>);
    impl<'a> Visitor<'a> for Attrs {
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Attribute(a) = e {
                self.0.insert(a.attr.to_string());
            }
            walk_expr(self, e);
        }
    }
    let mut attrs = Attrs::default();
    for s in body {
        attrs.visit_stmt(s);
    }
    for item in items {
        for s in &item.def.body {
            attrs.visit_stmt(s);
        }
    }
    attrs.0
}

/// Whether `body` reads any of `names`.
fn reads_any_of(body: &[py::Stmt], names: &HashSet<String>) -> bool {
    use ruff_python_ast::visitor::{Visitor, walk_expr};
    struct Reads<'n> {
        names: &'n HashSet<String>,
        found: bool,
    }
    impl<'a> Visitor<'a> for Reads<'_> {
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Name(n) = e
                && self.names.contains(n.id.as_str())
            {
                self.found = true;
            }
            walk_expr(self, e);
        }
    }
    let mut r = Reads {
        names,
        found: false,
    };
    for s in body {
        r.visit_stmt(s);
    }
    r.found
}

fn module_globals(
    body: &[py::Stmt],
    defs: &[&py::StmtFunctionDef],
    classes: &HashMap<String, usize>,
) -> Vec<String> {
    let module = scope::Scope::of_body(Vec::new(), body);
    let mut functions: HashSet<&str> = defs.iter().map(|f| f.name.as_str()).collect();
    functions.extend(classes.keys().map(|k| k.as_str()));
    let mut names: std::collections::BTreeSet<String> =
        module.declared_globals().into_iter().collect();
    for (_, child) in module.children.iter().chain(&module.methods) {
        for name in &child.free {
            if module.bound.contains(name) && !functions.contains(name.as_str()) {
                names.insert(name.clone());
            }
        }
    }
    // `globals()[name]` may reach any module variable by name.
    if uses_globals(body, defs) {
        for name in &module.bound {
            if !functions.contains(name.as_str()) {
                names.insert(name.clone());
            }
        }
    }
    names.into_iter().collect()
}

/// Whether the program calls `globals()` anywhere.
fn uses_globals(body: &[py::Stmt], defs: &[&py::StmtFunctionDef]) -> bool {
    use ruff_python_ast::visitor::{Visitor, walk_expr};
    #[derive(Default)]
    struct Finder(bool);
    impl<'a> Visitor<'a> for Finder {
        fn visit_expr(&mut self, e: &'a py::Expr) {
            if let py::Expr::Call(c) = e
                && matches!(&*c.func, py::Expr::Name(n) if n.id.as_str() == "globals")
            {
                self.0 = true;
            }
            walk_expr(self, e);
        }
    }
    let mut finder = Finder::default();
    for s in body {
        finder.visit_stmt(s);
    }
    for f in defs {
        for s in &f.body {
            finder.visit_stmt(s);
        }
    }
    finder.0
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
    unpack_shapes: &HashMap<String, HashMap<String, types::Ty>>,
) -> Result<Vec<TypedNode<TypedDeclaration>>> {
    let mut declarations = Vec::with_capacity(items.len());
    for item in items {
        let sig = inferred.funcs[&item.name].clone();
        declarations.extend(lower_item_as(
            inferred,
            item,
            &item.name,
            sig,
            unpack_shapes,
        )?);
    }
    Ok(declarations)
}

/// Lower `item`'s body as the function `fn_name` with signature `sig`:
/// the item itself, or one of its per-signature instances. The
/// function, and the variant trusting its instance-typed parameters
/// where it has any.
fn lower_item_as(
    inferred: &types::Module,
    item: &types::Item<'_>,
    fn_name: &str,
    sig: types::Sig,
    unpack_shapes: &HashMap<String, HashMap<String, types::Ty>>,
) -> Result<Vec<TypedNode<TypedDeclaration>>> {
    let is_instance = fn_name != item.name;
    let mut declarations = Vec::with_capacity(2);
    let mut variants = vec![(fn_name.to_string(), false)];
    if inferred.trusted.contains(fn_name) {
        variants.push((types::trusted_name(fn_name), true));
    }
    for (name, trusted) in variants {
        let sig = sig.clone();
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
        if !trusted && std::env::var_os("ZYNTAX_TRACE_TYPES").is_some() {
            let mut vars: Vec<String> = locals
                .vars
                .iter()
                .filter(|(n, _)| !sig.params.iter().any(|(p, _)| p == *n))
                .map(|(n, t)| format!("{n}: {}", t.describe()))
                .collect();
            vars.sort();
            eprintln!("[types] locals {} {{ {} }}", fn_name, vars.join(", "));
        }
        let scope = scope::Scope::of_function(item.def);
        let mut lowerer = lower::Lowerer::new(
            inferred,
            &item.name,
            sig,
            locals,
            &scope,
            Vec::new(),
            HashMap::default(),
        );
        lowerer.class = item.class;
        lowerer.trusted = trusted;
        lowerer.callee_defaults = !is_instance;
        let func =
            lowerer
                .function_named(item.def, &name)
                .map_err(|e| match item.module.as_deref() {
                    Some(m) => e.in_module(m),
                    None => e,
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
    Ok(declarations)
}

/// Lower the per-signature instances of `item`. An instance whose body
/// does not lower against its types forwards to the item's own
/// function instead, so its callers stand.
fn lower_specs(
    inferred: &types::Module,
    item: &types::Item<'_>,
    unpack_shapes: &HashMap<String, HashMap<String, types::Ty>>,
) -> Vec<TypedNode<TypedDeclaration>> {
    let mut declarations = Vec::new();
    for spec in inferred.specs.iter().filter(|s| s.item == item.name) {
        match lower_item_as(inferred, item, &spec.name, spec.sig.clone(), unpack_shapes) {
            Ok(d) => declarations.extend(d),
            Err(e) => {
                if std::env::var_os("ZYNTAX_TRACE_TYPES").is_some() {
                    eprintln!(
                        "[types] instance {} forwards to {}: {e:?}",
                        spec.name, item.name
                    );
                }
                let forwarders = lower::forwarders(inferred, spec, &inferred.funcs[&item.name]);
                for (f, fact) in forwarders {
                    inferred
                        .raise_facts
                        .borrow_mut()
                        .insert(f.name.resolve_global().unwrap_or_default(), fact);
                    declarations.push(TypedNode::new(
                        TypedDeclaration::Function(f),
                        Type::Unknown,
                        span_of(item.def),
                    ));
                }
            }
        }
    }
    declarations
}

/// The declarations of a method that failed to lower, each a function
/// of its signature that raises the reason when called.
fn lower_stub(
    inferred: &types::Module,
    item: &types::Item<'_>,
    error: &Error,
) -> Vec<TypedNode<TypedDeclaration>> {
    let message = match error {
        Error::Unsupported { what, .. } => format!("{what} is not supported"),
        other => format!("{other:?}"),
    };
    let mut variants = vec![(item.name.clone(), false)];
    if inferred.trusted.contains(&item.name) {
        variants.push((types::trusted_name(&item.name), true));
    }
    let span = span_of(item.def);
    let mut out = Vec::new();
    for (name, trusted) in variants {
        let sig = inferred.funcs[&item.name].clone();
        let scope = scope::Scope::of_function(item.def);
        let mut lowerer = lower::Lowerer::new(
            inferred,
            &item.name,
            sig,
            types::Locals::default(),
            &scope,
            Vec::new(),
            HashMap::default(),
        );
        lowerer.class = item.class;
        lowerer.trusted = trusted;
        let func = lowerer.stub_function(&name, &message, span);
        inferred
            .raise_facts
            .borrow_mut()
            .insert(name, lowerer.raise_fact());
        out.push(TypedNode::new(
            TypedDeclaration::Function(func),
            Type::Unknown,
            span,
        ));
    }
    out
}

/// A module-level `if True:` with no other branch (a version branch
/// decided at link time) is its body: what it defines is the module's,
/// not a closure's.
fn flatten_true_if(stmt: py::Stmt) -> Vec<py::Stmt> {
    match stmt {
        py::Stmt::If(i)
            if matches!(&*i.test, py::Expr::BooleanLiteral(b) if b.value)
                && i.elif_else_clauses.is_empty() =>
        {
            i.body.into_iter().flat_map(flatten_true_if).collect()
        }
        other => vec![other],
    }
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

/// What a module exports to another module, by Python's convention: its
/// top-level `def`s and `class`es, in order. With `__all__` assigned a
/// list of string literals, the names it lists and nothing else; without
/// it, every name that does not start with `_`. A host that publishes
/// the module to other languages reads this rather than the program's
/// declarations, which also carry the prelude's and the imported
/// modules'.
pub fn exports(source: &str) -> Result<Vec<zyntax_typed_ast::ExportedSymbol>> {
    use zyntax_typed_ast::{ExportedSymbol, SymbolKind};
    let parsed = ruff_python_parser::parse_module(source)
        .map_err(|e| Error::syntax(e.error.to_string(), e.location))?;
    let module = parsed.into_syntax();
    let mut all: Option<Vec<String>> = None;
    let mut declared: Vec<(String, SymbolKind)> = Vec::new();
    for stmt in &module.body {
        match stmt {
            py::Stmt::FunctionDef(f) => declared.push((f.name.to_string(), SymbolKind::Function)),
            py::Stmt::ClassDef(c) => declared.push((c.name.to_string(), SymbolKind::Class)),
            py::Stmt::Assign(a) => {
                let names_all = a
                    .targets
                    .iter()
                    .any(|t| matches!(t, py::Expr::Name(n) if n.id.as_str() == "__all__"));
                if names_all && let py::Expr::List(list) = &*a.value {
                    all = Some(
                        list.elts
                            .iter()
                            .filter_map(|e| match e {
                                py::Expr::StringLiteral(s) => Some(s.value.to_string()),
                                _ => None,
                            })
                            .collect(),
                    );
                }
            }
            _ => {}
        }
    }
    Ok(declared
        .into_iter()
        .filter(|(name, _)| match &all {
            Some(all) => all.contains(name),
            None => !name.starts_with('_'),
        })
        .map(|(name, kind)| ExportedSymbol {
            name,
            kind,
            is_public: true,
        })
        .collect())
}

/// The methods a class exports, by Python's convention: the `def`s of
/// its body whose names do not start with `_`, in order. `None` when
/// `class` is not declared at the top level of `source`.
pub fn class_exports(source: &str, class: &str) -> Result<Option<Vec<String>>> {
    let parsed = ruff_python_parser::parse_module(source)
        .map_err(|e| Error::syntax(e.error.to_string(), e.location))?;
    let module = parsed.into_syntax();
    for stmt in &module.body {
        if let py::Stmt::ClassDef(c) = stmt
            && c.name.as_str() == class
        {
            return Ok(Some(
                c.body
                    .iter()
                    .filter_map(|s| match s {
                        py::Stmt::FunctionDef(f) if !f.name.starts_with('_') => {
                            Some(f.name.to_string())
                        }
                        _ => None,
                    })
                    .collect(),
            ));
        }
    }
    Ok(None)
}
