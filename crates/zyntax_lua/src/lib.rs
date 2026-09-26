//! A Lua 5.4 frontend for Zyntax.
//!
//! Lua source is parsed by `full_moon` and its AST is rewritten into a
//! [`TypedProgram`], which the runtime compiles the same way it
//! compiles anything else.
//!
//! ## Types
//!
//! Every expression gets a static type from [`types`]: an integer is
//! `i64`, a float `f64`, a boolean `bool`, a string `String`, a table
//! the library's table struct, and a value the pass cannot type is
//! `Any`, the IR's boxed dynamic value. Crossings between them are
//! explicit in what [`lower`] emits. What Lua defines above the IR
//! (tables, metatables, coroutines, the standard library) comes from
//! the built-in library in [`library`], compiled once into the
//! snapshot this crate carries.

use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::{InternedString, PrimitiveType, Type, TypedProgram};

mod annotation;
mod dump;
mod exports;
mod host;
mod host_debug;
mod host_gc;
mod host_io;
mod host_os;
pub mod library;
mod lower;
mod pack;
mod parse;
pub mod pattern;
mod policy;
mod scope;
mod types;

pub use annotation::{LuaType, Returned};
pub use exports::{
    DeclaredClass, Exported, ExportedFunction, ExportedTable, Exports, Signature, is_metafield,
};
pub use host::{set_args, set_ignore_env};
pub use parse::SyntaxError;
use policy::LIBRARY_MODULE;
pub use policy::POLICY;

/// Why a program could not be turned into a `TypedProgram`.
///
/// Each carries where in the source it happened, as byte offsets.
/// [`Error::render`] shows it against the source.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Lua the reference refuses to read, with its message and line as
    /// the reference gives them.
    #[error("Lua syntax error: {}", .0.message)]
    Rejected(parse::SyntaxError),
    /// Lua that does not parse, found while it is compiled.
    #[error("Lua syntax error: {message} at {}..{}", span.0, span.1)]
    Syntax {
        message: String,
        span: (usize, usize),
    },
    /// Something Lua allows that this frontend does not compile yet.
    /// Says what it was and where, so the gap is a fact rather than a
    /// guess.
    #[error("{what} is not supported yet (at byte offset {})", span.0)]
    Unsupported { what: String, span: (usize, usize) },
    /// The built-in library this crate was built with cannot be read.
    #[error("the built-in library is unreadable: {0}")]
    Library(String),
    /// An annotation that does not fit what its statement declares, at
    /// the comment.
    #[error("{message} (at byte offset {})", span.0)]
    Annotation {
        message: String,
        span: (usize, usize),
    },
}

impl Error {
    pub(crate) fn unsupported(what: impl Into<String>, span: Span) -> Self {
        Error::Unsupported {
            what: what.into(),
            span: (span.start, span.end),
        }
    }

    /// The error on one line, as the reference reports a chunk it cannot
    /// read: `chunk:line: message`. `chunk` is the chunk's name as a
    /// message shows it.
    pub fn one_line(&self, chunk: &str, source: &str) -> String {
        let line_of = |at: usize| parse::line_at(source, at);
        match self {
            Error::Rejected(e) => e.with_chunk(chunk),
            Error::Syntax { message, span } => format!("{chunk}:{}: {message}", line_of(span.0)),
            Error::Unsupported { what, span } => {
                format!("{chunk}:{}: {what} is not supported yet", line_of(span.0))
            }
            Error::Library(message) => format!("the built-in library is unreadable: {message}"),
            Error::Annotation { message, span } => {
                format!("{chunk}:{}: {message}", line_of(span.0))
            }
        }
    }

    /// The error positioned in `original` rather than in the text
    /// [`parse_chunk`] made of it, whose line breaks are all `\n`.
    pub(crate) fn in_original(self, original: &str) -> Self {
        if !original.contains('\r') {
            return self;
        }
        let at = |offset: usize| parse::original_offset(original, offset);
        match self {
            Error::Rejected(mut e) => {
                e.offset = at(e.offset);
                Error::Rejected(e)
            }
            Error::Syntax { message, span } => Error::Syntax {
                message,
                span: (at(span.0), at(span.1)),
            },
            Error::Unsupported { what, span } => Error::Unsupported {
                what,
                span: (at(span.0), at(span.1)),
            },
            e @ Error::Library(_) => e,
            Error::Annotation { message, span } => Error::Annotation {
                message,
                span: (at(span.0), at(span.1)),
            },
        }
    }

    /// The error shown against its source, the way the compiler shows
    /// its own diagnostics.
    pub fn render(&self, file: &str, source: &str, use_colors: bool) -> String {
        use zyntax_typed_ast::diagnostics::{Diagnostic, render_diagnostic};
        let (message, label, span) = match self {
            Error::Rejected(e) => (
                format!("syntax error: {}", e.message),
                "here",
                (e.offset, e.offset),
            ),
            Error::Syntax { message, span } => (format!("syntax error: {message}"), "here", *span),
            Error::Unsupported { what, span } => (
                format!("{what} is not supported yet"),
                "this frontend does not compile this form",
                *span,
            ),
            Error::Library(message) => {
                return format!("error: the built-in library is unreadable: {message}\n");
            }
            Error::Annotation { message, span } => (message.clone(), "this annotation", *span),
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

/// The function a chunk's statements become. A host runs a Lua program
/// by calling this.
pub const ENTRY: &str = "lua$main";

/// The function that opens the state for a host embedding the runtime,
/// which then loads and calls chunks through the C API (see
/// [`open_host`]).
pub const HOST_ENTRY: &str = "lua$host";

/// The built-in library, declared and lowered when this crate was
/// built. A program imports it; the runtime links the HIR and lowers
/// only the program.
const SNAPSHOT: &[u8] = zyntax_embed::include_snapshot!("lua");

/// The library functions that can raise, so a call to one is followed
/// by a check for the error.
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
/// types live in, and which types are `List<T>` and the table.
pub(crate) struct Library {
    type_registry: zyntax_typed_ast::TypeRegistry,
    types: library::Types,
}

/// The library, read from the snapshot once per process.
fn library() -> Result<&'static Library> {
    static LIBRARY: std::sync::OnceLock<std::result::Result<Library, String>> =
        std::sync::OnceLock::new();
    LIBRARY
        .get_or_init(read_library)
        .as_ref()
        .map_err(|message| Error::Library(message.clone()))
}

fn read_library() -> std::result::Result<Library, String> {
    let snapshot = snapshot().map_err(|e| match e {
        Error::Library(message) => message,
        other => other.to_string(),
    })?;
    let module = snapshot
        .module(LIBRARY_MODULE)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("the snapshot has no `{LIBRARY_MODULE}`"))?;
    let program = module.program();
    let type_registry = program.type_registry.clone();
    let list_type = type_registry
        .get_type_by_name(intern("List"))
        .map(|def| def.id)
        .ok_or_else(|| "the library declares no List type".to_string())?;
    let table_type = type_registry
        .get_type_by_name(intern(library::TABLE_TYPE))
        .map(|def| def.id)
        .ok_or_else(|| "the library declares no table type".to_string())?;
    Ok(Library {
        type_registry,
        types: library::Types {
            list_type,
            table_type,
        },
    })
}

/// Give a runtime what a compiled Lua program links against: the IO,
/// string and math plugins the library's primitives come from, the
/// host's own symbols, and the name a program is entered through, so
/// only the library the program reaches is built. A host calls this
/// once before compiling a program, after [`set_args`] if it has any.
/// The runtime `load` compiles chunks with, set by the host before the
/// program runs. A loaded chunk is compiled from inside the program,
/// which the runtime's entry holds a shared reference to; what
/// compiling touches is not what running reads.
static RUNTIME: std::sync::atomic::AtomicPtr<zyntax_embed::TieredRuntime> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

pub fn set_runtime(runtime: &mut zyntax_embed::TieredRuntime) {
    RUNTIME.store(runtime, std::sync::atomic::Ordering::Release);
}

pub(crate) fn runtime() -> Option<&'static mut zyntax_embed::TieredRuntime> {
    let p = RUNTIME.load(std::sync::atomic::Ordering::Acquire);
    // SAFETY: the host set it to a runtime that outlives the program,
    // and runs the program on one thread.
    unsafe { p.as_mut() }
}

/// Compile `source` as a chunk of the running program: the function
/// value it is, or the message a syntax error gives. `raw_name` is the
/// reference's `source` for it; a chunk `stripped` of debug
/// information has no lines or variable names.
pub(crate) fn load_chunk(
    source: &str,
    (chunk_name, raw_name): (&str, &str),
    index: i64,
    env: *const zrtl::DynamicBox,
    stripped: bool,
) -> std::result::Result<*const zrtl::DynamicBox, String> {
    let started = std::time::Instant::now();
    let (ast, source) =
        parse_chunk(source, LOAD_LEVEL).map_err(|e| e.one_line(chunk_name, source))?;
    trace_phase("load:parse", started);
    let started = std::time::Instant::now();
    let library = library().map_err(|e| e.to_string())?;
    trace_phase("load:lib", started);
    let started = std::time::Instant::now();
    let program = lower::loaded_program(&ast, &source, chunk_name, index, stripped, library)
        .map_err(|e| e.one_line(chunk_name, &source))?;
    dump::note_chunk(index, &source, raw_name, stripped);
    trace_phase("load:lower", started);
    let started = std::time::Instant::now();
    let runtime = runtime().ok_or("no runtime to load into")?;
    let init = format!("lua$l{index}$init");
    runtime
        .join_typed_program(program, &[init.as_str()])
        .map_err(|e| e.to_string())?;
    trace_phase("load:compile", started);
    let started = std::time::Instant::now();
    let entry = runtime
        .function_pointer(&init)
        .ok_or("the loaded chunk has no entry")?;
    // SAFETY: `init` was compiled with this signature just above.
    let init: extern "C" fn(*const zrtl::DynamicBox) -> *const zrtl::DynamicBox =
        unsafe { std::mem::transmute(entry) };
    let chunk = init(env);
    trace_phase("load:init", started);
    Ok(chunk)
}

pub fn register_runtime(
    runtime: &mut zyntax_embed::TieredRuntime,
) -> std::result::Result<(), zyntax_embed::RuntimeError> {
    // No Lua program releases anything itself, so the compiler releases
    // what it can prove dead and the collector takes the rest.
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
        zyntax_lua_capi::static_plugin(),
    ])?;
    zyntax_lua_capi::install(zyntax_lua_capi::Hooks {
        resolve: resolve_bridge,
        add_root: zyntax_compiler::collector::add_root_range,
        tags: zyntax_lua_capi::Tags {
            table: library::table_tag() as u32,
            thread: library::thread_tag() as u32,
            function: zyntax_builtins::FUNC_TAG as u32,
            code: zyntax_builtins::CODE_TAG as u32,
            userdata: library::userdata_tag() as u32,
            light: library::light_tag() as u32,
        },
    });
    Ok(())
}

/// Compile the C API's bridge into the running program, the first time
/// a native library opens, and hand over its entries' addresses.
fn resolve_bridge() -> std::result::Result<zyntax_lua_capi::bridge::Resolved, String> {
    use zyntax_lua_capi::bridge::{ENTRIES, Resolved, VERSION, entry_name};
    let lib = library().map_err(|e| e.to_string())?;
    let defined = library::capi::bridge_entry_names(&lib.types);
    let expected: Vec<(String, usize)> = ENTRIES
        .iter()
        .map(|(name, arity)| (entry_name(name), *arity))
        .collect();
    if defined != expected {
        return Err("the library's C API bridge does not match the plugin's".to_string());
    }
    let started = std::time::Instant::now();
    let runtime = runtime().ok_or("no runtime to compile the C API bridge into")?;
    let program =
        library::capi::bridge_program(&lib.types, lib.type_registry.clone(), LIBRARY_MODULE);
    let names: Vec<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    runtime
        .join_typed_program(program, &names)
        .map_err(|e| e.to_string())?;
    let entries = names
        .iter()
        .map(|n| runtime.function_pointer(n).unwrap_or(std::ptr::null()))
        .collect();
    trace_phase("capi:bridge", started);
    Ok(Resolved {
        version: VERSION,
        entries,
    })
}

/// Parse Lua source and rewrite it into a `TypedProgram`. `file` names
/// the source in diagnostics.
/// A source that is not UTF-8 as text the parser takes: each byte that
/// is not part of a valid sequence becomes a private-use character,
/// which a string literal turns back into the byte. Outside literals
/// (comments) the bytes do not matter.
pub fn source_text(bytes: &[u8]) -> std::borrow::Cow<'_, str> {
    match std::str::from_utf8(bytes) {
        Ok(s) => std::borrow::Cow::Borrowed(s),
        Err(_) => {
            let mut out = String::with_capacity(bytes.len() + 16);
            let mut rest = bytes;
            loop {
                match std::str::from_utf8(rest) {
                    Ok(s) => {
                        out.push_str(s);
                        break;
                    }
                    Err(e) => {
                        let good = e.valid_up_to();
                        out.push_str(std::str::from_utf8(&rest[..good]).unwrap_or(""));
                        let bad = e.error_len().unwrap_or(rest.len() - good);
                        for &b in &rest[good..good + bad] {
                            out.push(escaped_byte(b));
                        }
                        rest = &rest[good + bad..];
                    }
                }
            }
            std::borrow::Cow::Owned(out)
        }
    }
}

/// Whether a file's bytes, past any `#` line, are a binary chunk.
pub fn is_binary(bytes: &[u8]) -> bool {
    bytes.first() == Some(&dump::SIGNATURE[0])
}

/// A binary chunk run as a program: the text it holds, calling the
/// function it holds with the script's arguments when it is not a main
/// chunk; or the message refusing it, the chunk named `file`.
pub fn binary_source(bytes: &[u8], file: &str) -> std::result::Result<String, String> {
    let undumped =
        dump::undump(bytes).map_err(|why| format!("{file}: bad binary format ({why})"))?;
    Ok(dump::wrapper_text(&undumped, true))
}

/// The private-use character standing for byte `b` in a source that
/// is not UTF-8.
pub(crate) fn escaped_byte(b: u8) -> char {
    char::from_u32(ESCAPED_BYTES + b as u32).expect("a private-use character")
}
pub(crate) const ESCAPED_BYTES: u32 = 0xF700;

/// The C call depth the reference reads a chunk at: a script from the
/// interpreter's entry, a chunk `load`ed from a script's main chunk.
const MAIN_LEVEL: usize = 1;
pub(crate) const LOAD_LEVEL: usize = 2;

/// `text` as the source it was read from: each private-use character
/// [`source_text`] made of a byte that was not UTF-8 is that byte again.
pub(crate) fn source_bytes(text: &str) -> std::borrow::Cow<'_, [u8]> {
    let escaped = |c: char| (ESCAPED_BYTES..ESCAPED_BYTES + 256).contains(&(c as u32));
    if !text.chars().any(escaped) {
        return std::borrow::Cow::Borrowed(text.as_bytes());
    }
    let mut out = Vec::with_capacity(text.len());
    let mut utf8 = [0; 4];
    for c in text.chars() {
        if escaped(c) {
            out.push((c as u32 - ESCAPED_BYTES) as u8);
        } else {
            out.extend_from_slice(c.encode_utf8(&mut utf8).as_bytes());
        }
    }
    std::borrow::Cow::Owned(out)
}

/// Read a chunk: refused as the reference refuses it, then parsed. The
/// text the tree was parsed from comes with it: every line break in it
/// is a `\n`, and a form `full_moon` does not take is rewritten into
/// one it does, lines unmoved. An error's position is in `text`.
pub(crate) fn parse_chunk(
    text: &str,
    level: usize,
) -> Result<(full_moon::ast::Ast, std::borrow::Cow<'_, str>)> {
    let Some(normal) = parse::with_newline_breaks(text) else {
        return parse_normal(std::borrow::Cow::Borrowed(text), level);
    };
    parse_normal(std::borrow::Cow::Owned(normal), level).map_err(|e| e.in_original(text))
}

/// [`parse_chunk`] on a text whose line breaks are all `\n`.
fn parse_normal(
    text: std::borrow::Cow<'_, str>,
    level: usize,
) -> Result<(full_moon::ast::Ast, std::borrow::Cow<'_, str>)> {
    let bytes = source_bytes(&text);
    let checked = parse::check(&bytes, level).map_err(Error::Rejected)?;
    let rewritten = parse::with_breaks_closed(&bytes, &checked);
    drop(bytes);
    let text = match rewritten {
        Some(rewritten) => std::borrow::Cow::Owned(source_text(&rewritten).into_owned()),
        None => text,
    };
    match full_moon::parse_fallible(&text, full_moon::LuaVersion::lua54()).into_result() {
        Ok(ast) => Ok((ast, text)),
        Err(errors) => {
            let first = errors.into_iter().next().expect("an error");
            let (message, range) = match &first {
                full_moon::Error::AstError(e) => (e.error_message().to_string(), e.range()),
                full_moon::Error::TokenizerError(e) => (e.error().to_string(), e.range()),
            };
            Err(Error::Syntax {
                message,
                span: (range.0.bytes(), range.1.bytes()),
            })
        }
    }
}

pub fn parse_program(source: &str, file: &str) -> Result<TypedProgram> {
    let started = std::time::Instant::now();
    let (ast, normal) = parse_chunk(source, MAIN_LEVEL)?;
    trace_phase("full_moon", started);
    dump::note_chunk(0, source, &format!("@{file}"), false);
    let started = std::time::Instant::now();
    let library = library()?;
    trace_phase("library", started);
    lower::program(&ast, &normal, file, library, lower::Entry::Program)
        .map_err(|e| e.in_original(source))
}

/// What a module whose chunk is `source` exports, by Lua's convention:
/// the value the chunk returns, as the types of a chunk `load` compiles
/// know it. Nothing is compiled or run.
pub fn exports(source: &str) -> Result<Exports> {
    let (ast, _) = parse_chunk(source, LOAD_LEVEL)?;
    let (scopes, inferred) = lower::loaded_types(&ast);
    exports::of(&scopes, &inferred).map_err(|e| e.in_original(source))
}

/// Open the state of `runtime` for a host, as `luaL_newstate` does for a
/// C host: the library and a program of no statements compiled, the
/// globals kept in the globals table, and the C API made ready. The host
/// then loads a chunk with the global `load` and runs it with a
/// protected call, through the C API on the returned `lua_State`. The
/// runtime is [`register_runtime`]'s, and keeps its address while the
/// state is open; once per runtime.
pub fn open_host(
    runtime: &mut zyntax_embed::TieredRuntime,
) -> std::result::Result<zyntax_lua_capi::state::L, String> {
    let (ast, source) = parse_chunk("", MAIN_LEVEL).map_err(|e| e.to_string())?;
    let library = library().map_err(|e| e.to_string())?;
    let program = lower::program(&ast, &source, "=host", library, lower::Entry::Host)
        .map_err(|e| e.to_string())?;
    runtime.declare_entry_points([HOST_ENTRY]);
    set_runtime(runtime);
    runtime
        .compile_typed_program(program)
        .map_err(|e| e.to_string())?;
    let entry = runtime
        .function_pointer(HOST_ENTRY)
        .ok_or("the host program has no entry")?;
    // SAFETY: the entry was compiled just above, taking and returning nothing.
    let open: extern "C" fn() = unsafe { std::mem::transmute(entry) };
    open();
    zyntax_lua_capi::host_state()
}

/// `ZYNTAX_TRACE_LOWER_PHASES=1` times the frontend's steps on stderr.
pub(crate) fn trace_phase(what: &str, since: std::time::Instant) {
    if std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some() {
        eprintln!(
            "[ZYLUA] {what:<10} {:8.2} ms",
            since.elapsed().as_secs_f64() * 1000.0
        );
    }
}

pub(crate) fn intern(s: &str) -> InternedString {
    InternedString::new_global(s)
}

pub(crate) fn prim(p: PrimitiveType) -> Type {
    Type::Primitive(p)
}

#[cfg(test)]
mod capi_tests {
    use super::*;

    /// The library defines the bridge the plugin reads, entry for entry.
    #[test]
    fn the_bridge_matches_the_plugin() {
        use zyntax_lua_capi::bridge::{ENTRIES, entry_name};
        let lib = library().expect("the library");
        let defined = library::capi::bridge_entry_names(&lib.types);
        let expected: Vec<(String, usize)> = ENTRIES
            .iter()
            .map(|(name, arity)| (entry_name(name), *arity))
            .collect();
        assert_eq!(defined, expected);
        assert_eq!(zyntax_lua_capi::bridge::LINE_BITS, library::LINE_BITS);
    }

    #[test]
    fn userdata_kinds_follow_the_library_kinds() {
        let kinds = [
            library::TABLE_KIND,
            library::THREAD_KIND,
            library::NIL_ERROR_KIND,
            library::FILE_KIND,
            library::CLOSING_KIND,
            library::DEAD_KEY_KIND,
            library::LIGHT_KIND,
            library::USERDATA_KIND,
        ];
        let mut sorted = kinds.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), kinds.len(), "every kind is distinct");
    }
}
