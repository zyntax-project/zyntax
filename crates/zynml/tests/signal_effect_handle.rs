//! A signal as an effect handle: the handler owns the storage, and the
//! host only performs get/set through it.
//!
//! The open question here is how a GENERIC handler spells "no value
//! yet". A `signal` declared in a DSL carries the author's default, so
//! `= 0` would do, but a handler field whose type is the effect's type
//! parameter has no context to infer an initialiser from, and the
//! grammar requires one.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime};

fn compile(src: &str, name: &str) -> Result<TieredRuntime, String> {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    let mut rt = TieredRuntime::new(config).map_err(|e| e.to_string())?;
    let g = Grammar2::from_source(ZYNML_GRAMMAR).map_err(|e| format!("{e:?}"))?;
    let program = g
        .parse_with_filename(src, name)
        .map_err(|e| format!("parse: {e:?}"))?;
    rt.compile_typed_program(program)
        .map_err(|e| format!("compile: {e}"))?;
    Ok(rt)
}

fn brief<T>(outcome: &Result<T, String>) -> String {
    match outcome {
        Ok(_) => "ok".to_string(),
        Err(e) => e.chars().take(96).collect::<String>().replace('\n', " "),
    }
}

/// How a generic handler field can be declared.
///
/// Only `set` is in the effect, so the field's declaration is the only
/// thing varying: a `get(): T` over an optional field would also have
/// to unwrap, which would confound "the field form is rejected" with
/// "the read is ill-typed".
#[test]
fn which_generic_field_form_parses() {
    let forms = [
        ("bare", "var content: T"),
        ("Null<T> + null", "var content: Null<T> = null"),
        ("?T + null", "var content: ?T = null"),
        ("T + zero", "var content: T = 0"),
        ("T + default", "var content: T = default"),
    ];
    for (label, field) in forms {
        let src = format!(
            "effect Signal<T> {{\n    def set(val: T)\n}}\n\n\
             handler MintedSignal<T> for Signal<T> {{\n    {field}\n    \
             def set(val: T) {{ self.content = val }}\n}}\n"
        );
        println!("FIELD {label:<22} -> {}", brief(&compile(&src, "g.zyn")));
    }
}

/// Whether a field that can be absent is readable, and as what.
///
/// A `Null<T>` / `?T` field holds "T or nothing", so `get(): T` over it
/// is only well-typed if something unwraps. Each row pairs the field's
/// type with the type `get` claims to return, which tells apart "the
/// optional field works" from "the optional field works but the effect
/// signature has to change with it" — and the latter matters, because
/// the effect is `def get(): T` and a handler cannot widen it.
#[test]
fn what_an_absent_capable_field_reads_back_as() {
    let rows = [
        ("Null<T> read as T", "var content: Null<T> = null", "T"),
        (
            "Null<T> read as Null<T>",
            "var content: Null<T> = null",
            "Null<T>",
        ),
        ("?T read as T", "var content: ?T = null", "T"),
        ("?T read as ?T", "var content: ?T = null", "?T"),
        ("T read as T (control)", "var content: T = 0", "T"),
    ];
    for (label, field, ret) in rows {
        let src = format!(
            "effect Signal<T> {{\n    def get(): {ret}\n    def set(val: T)\n}}\n\n\
             handler MintedSignal<T> for Signal<T> {{\n    {field}\n    \
             def get(): {ret} {{ return self.content }}\n    \
             def set(val: T) {{ self.content = val }}\n}}\n"
        );
        println!("READBACK {label:<26} -> {}", brief(&compile(&src, "g.zyn")));
    }
}

/// Whether the intended generic shape resolves and mints independent
/// instances, and whether the concrete spelling behaves the same.
#[test]
fn report_what_the_substrate_supports() {
    let generic = r#"
effect Signal<T> {
    def get(): Null<T>
    def set(val: T)
}

handler MintedSignal<T> for Signal<T> {
    var content: Null<T> = null
    def get(): Null<T> { return self.content }
    def set(val: T) { self.content = val }
}
"#;

    let concrete = r#"
effect SignalI64 {
    def get(): i64
    def set(val: i64)
}

handler MintedSignalI64 for SignalI64 {
    var content: i64 = 0
    def get(): i64 { return self.content }
    def set(val: i64) { self.content = val }
}

@effect(SignalI64)
def read(): i64 { return get() }

@effect(SignalI64)
def write(v: i64): i64 { set(v) return 0 }
"#;

    for (label, src, handler) in [
        ("GENERIC ", generic, "MintedSignal"),
        ("CONCRETE", concrete, "MintedSignalI64"),
    ] {
        match compile(src, "s.zyn") {
            Ok(mut rt) => {
                let token = rt.get_effect_handler(handler);
                match token {
                    Ok(tok) => {
                        let a = rt.new_handler_instance(tok);
                        let b = rt.new_handler_instance(tok);
                        println!(
                            "{label} compile=ok handler=ok instances={} {}",
                            a.is_ok(),
                            b.is_ok()
                        );
                    }
                    Err(e) => println!("{label} compile=ok handler=ERR {e}"),
                }
            }
            Err(e) => println!("{label} compile=ERR {e}"),
        }
    }
}
