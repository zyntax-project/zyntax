//! Errors support.

use super::*;

/// An error nothing caught, on stderr the way `lua` reports it.
pub(super) extern "C" fn host_report_pending(err: *const DynamicBox) {
    use std::io::Write;
    let mut out = b"lua: ".to_vec();
    out.extend(uncaught_text(err));
    out.push(b'\n');
    // A program that keeps its call stack shows where the error was raised.
    if let Some(trace) = crate::host_debug::uncaught_traceback() {
        out.extend(trace.into_bytes());
        out.push(b'\n');
    }
    host_io::host_io_flush_all();
    let _ = std::io::stderr().write_all(&out);
}

/// The end of the program with status 1, after an uncaught error is
/// reported and the state closed. It ends as a program that returns
/// does: every open stream flushed, no destructor run, so a compile
/// thread still running is not raced by the process's teardown.
pub(super) extern "C" fn host_exit_uncaught() {
    host_io::host_io_flush_all();
    // SAFETY: `_exit` does not return, and the streams are flushed.
    unsafe { libc::_exit(1) }
}

/// The message of an uncaught error: a string as it is, a number as
/// `tostring` writes it, a table as its `__tostring` gives it when that
/// is a string (or by the error that raises), a file as it prints;
/// anything else by its type.
pub(super) fn uncaught_text(err: *const DynamicBox) -> Vec<u8> {
    let named = |kind: &str| format!("(error object is a {kind} value)").into_bytes();
    let kind_of = |b: &DynamicBox| b.tag.raw() >> 8;
    let instance = |k: usize| zyntax_builtins::instance_tag(k) as u32 >> 8;
    // SAFETY: the library hands over the pending error, a live box or null.
    let boxed = unsafe { err.as_ref() };
    if boxed.is_some_and(|b| kind_of(b) == instance(crate::library::NIL_ERROR_KIND)) {
        return named("nil");
    }
    match unsafe { read_arg(err) } {
        Arg::Nil => named("nil"),
        Arg::Bool(_) => named("boolean"),
        Arg::Int(n) => n.to_string().into_bytes(),
        Arg::Float(x) => float_text(x).into_bytes(),
        Arg::Str(bytes, _) => bytes.to_vec(),
        Arg::Other(text, _) => {
            let b = boxed.expect("a value that is not nil is a box");
            match type_word(b) {
                "FILE*" => text,
                "table" if kind_of(b) == instance(crate::library::TABLE_KIND) => {
                    match table_error_text(err) {
                        Ok(Some(text)) => text,
                        Ok(None) => named("table"),
                        Err(why) => {
                            eprintln!("zylua: the error's __tostring could not be compiled: {why}");
                            named("table")
                        }
                    }
                }
                "table" => named("userdata"),
                word => named(word),
            }
        }
    }
}

/// What an uncaught table error's `__tostring` gives, when that is a
/// string, or the message of the error it raises. No chunk reaches the
/// library's text for it, so it is compiled here, once, when the first
/// such error is reported.
pub(super) fn table_error_text(err: *const DynamicBox) -> Result<Option<Vec<u8>>, String> {
    use crate::lower::{ERROR_TEXT_ENTRY, error_text_program};
    let runtime = crate::runtime().ok_or("no runtime to compile into")?;
    let entry = match runtime.function_pointer(ERROR_TEXT_ENTRY) {
        Some(entry) => entry,
        None => {
            let library = crate::library().map_err(|e| e.to_string())?;
            runtime.declare_entry_points([ERROR_TEXT_ENTRY]);
            runtime
                .compile_typed_program(error_text_program(library))
                .map_err(|e| e.to_string())?;
            runtime
                .function_pointer(ERROR_TEXT_ENTRY)
                .ok_or("the compiled program has no entry")?
        }
    };
    // SAFETY: the entry was compiled with this signature.
    let text: extern "C" fn(*const DynamicBox) -> *const DynamicBox =
        unsafe { std::mem::transmute(entry) };
    Ok(match unsafe { read_arg(text(err)) } {
        Arg::Str(bytes, _) => Some(bytes.to_vec()),
        _ => None,
    })
}
