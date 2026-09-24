//! Method and field calls on shaped receivers take their typed paths:
//! a program under `typed_paths/` run by the interpreter tier calls no
//! library function that looks a callee up or calls through a value.

use std::path::Path;
use std::process::Command;

/// The library calls a typed method or field call never makes.
const DENIED: &[&str] = &[
    "zl_func_id",
    "zl_table_index_key",
    "zl_shape_index",
    "zl_apply_",
    "lua$value$",
    "lua$miss$",
];

/// `setmetatable` on a table with slots reads its `__metatable` field
/// through the shape: a call the program makes, not a method site.
const SETMETATABLE_ON_SLOTS: &[&str] = &["zl_shape_index"];

/// Runs `name` under the interpreter's call trace; the program's
/// output and the names of the functions it called.
fn traced(name: &str) -> (String, Vec<String>) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("typed_paths")
        .join(name);
    let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
        .arg("run")
        .arg(&script)
        .env("ZYNTAX_TRACE_INTERP", "1")
        .env("ZYNTAX_DISABLE_WARM_UP", "1")
        .output()
        .expect("zylua starts");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "{name} failed:\n{stderr}");
    let calls = stderr
        .lines()
        .filter_map(|l| l.strip_prefix("[interp] call "))
        .map(|l| l.split('(').next().unwrap_or(l).to_string())
        .collect();
    (String::from_utf8_lossy(&output.stdout).into_owned(), calls)
}

fn assert_typed(name: &str, expected: &str, finder: &str, allowed: &[&str]) {
    let (stdout, calls) = traced(name);
    assert_eq!(stdout.trim(), expected, "{name}");
    let denied: Vec<&String> = calls
        .iter()
        .filter(|c| DENIED.iter().any(|d| c.contains(d) && !allowed.contains(d)))
        .collect();
    assert!(denied.is_empty(), "{name} called {denied:?}");
    assert!(
        calls.iter().any(|c| c.contains(finder)),
        "{name} never called a {finder} finder: {calls:?}"
    );
}

#[test]
fn inherited_methods_are_called_by_which_end_the_lookup_takes() {
    assert_typed("method_call.lua", "true", "$which$", SETMETATABLE_ON_SLOTS);
}

#[test]
fn own_and_inherited_methods_are_called_directly() {
    assert_typed("probe_own.lua", "21", "$which$", &[]);
}

#[test]
fn a_class_table_field_is_called_directly() {
    assert_typed("field_call.lua", "12", "$which$", &[]);
}
