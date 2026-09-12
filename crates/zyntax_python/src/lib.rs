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
use zyntax_typed_ast::typed_ast::{TypedBlock, TypedDeclaration, TypedFunction};
use zyntax_typed_ast::{InternedString, PrimitiveType, Type, TypedNode, TypedProgram, Visibility};

mod lower;
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
    type_names: zyntax_builtins::TypeNames {
        none: "NoneType",
        bool: "bool",
        int: "int",
        float: "float",
        str: "str",
        list: "list",
        tuple: "tuple",
        object: "object",
    },
};

/// Give a runtime what a compiled Python program links against: the IO
/// and string plugins the library's primitives come from. A host calls
/// this once before compiling a program.
pub fn register_runtime(
    runtime: &mut zyntax_embed::TieredRuntime,
) -> std::result::Result<(), zyntax_embed::RuntimeError> {
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
    let module = parsed.into_syntax();

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
            other => top_level.push(other),
        }
    }

    let library = zyntax_builtins::library(&POLICY);
    lower::set_list_type(library.list_type);
    let mut inferred = types::infer_module(&defs);
    inferred.list_type = Some(library.list_type);
    let mut declarations = Vec::new();
    for f in &defs {
        let sig = inferred.funcs[f.name.as_str()].clone();
        let locals = types::infer_locals(&inferred, &sig, &f.body);
        let func = lower::Lowerer::new(&inferred, sig, locals).function(f)?;
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(func),
            Type::Unknown,
            span_of(*f),
        ));
    }
    if !top_level.is_empty() {
        let owned: Vec<py::Stmt> = top_level.iter().map(|s| (*s).clone()).collect();
        let sig = types::Sig {
            params: Vec::new(),
            ret: types::Ty::None,
            defaults: 0,
        };
        let locals = types::infer_locals(&inferred, &sig, &owned);
        let statements = lower::Lowerer::new(&inferred, sig, locals).body(&top_level)?;
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

    declarations.extend(library.declarations);

    Ok(TypedProgram {
        declarations,
        language: Some(intern("python")),
        span: Span::new(0, source.len()),
        source_files: Vec::new(),
        type_registry: library.type_registry,
    })
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
