//! A signal as an effect handle: the handler owns the storage, and the
//! host only performs get/set through it.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime};

/// The intended shape, generic over the signal's type.
const GENERIC: &str = r#"
effect Signal<T> {
    def get(): T
    def set(val: T)
}

handler MintedSignal<T> for Signal<T> {
    var content: T = 0
    def get(): T { return self.content }
    def set(val: T) { self.content = val }
}
"#;

/// The same thing with the type written out, to tell "generics are the
/// problem" apart from "this shape is the problem".
const CONCRETE: &str = r#"
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
def write(v: i64) { set(v) }
"#;

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

/// Variants on the generic handler's field declaration. The grammar
/// rejects a bare `var content: T`, expecting `<` or `=` after it.
#[test]
fn which_generic_field_form_parses() {
    let forms = [
        ("bare", "var content: T"),
        ("null-init", "var content: T = Null<T>"),
        ("zero-init", "var content: T = 0"),
        ("default-init", "var content: T = default"),
    ];
    for (label, field) in forms {
        let src = format!(
            "effect Signal<T> {{\n    def get(): T\n    def set(val: T)\n}}\n\nhandler MintedSignal<T> for Signal<T> {{\n    {field}\n    def get(): T {{ return self.content }}\n    def set(val: T) {{ self.content = val }}\n}}\n"
        );
        let outcome = compile(&src, "g.zyn");
        let brief = match &outcome {
            Ok(_) => "ok".to_string(),
            Err(e) => e.chars().take(90).collect::<String>().replace('\n', " "),
        };
        println!("FORM {label:<14} -> {brief}");
    }
}

#[test]
fn report_what_the_substrate_supports() {
    match compile(CONCRETE, "concrete.zyn") {
        Ok(mut rt) => {
            let h = rt.get_effect_handler("MintedSignalI64");
            println!("CONCRETE compile=ok handler={:?}", h.as_ref().err());
            if let Ok(tok) = h {
                let a = rt.new_handler_instance(tok);
                let b = rt.new_handler_instance(tok);
                println!("CONCRETE two instances: {:?} {:?}", a.is_ok(), b.is_ok());
            }
        }
        Err(e) => println!("CONCRETE compile=ERR {e}"),
    }

    match compile(GENERIC, "generic.zyn") {
        Ok(mut rt) => {
            let h = rt.get_effect_handler("MintedSignal");
            println!("GENERIC  compile=ok handler_err={:?}", h.as_ref().err());
            if let Ok(tok) = h {
                let a = rt.new_handler_instance(tok);
                let b = rt.new_handler_instance(tok);
                println!("GENERIC  two instances: {:?} {:?}", a.is_ok(), b.is_ok());
            }
        }
        Err(e) => println!("GENERIC  compile=ERR {e}"),
    }
}
