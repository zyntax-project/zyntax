//! A function is checked against the module it was compiled in.
//!
//! `current_module` is the last module loaded, not the only one. A host
//! that loads several leaves every function from the earlier ones
//! outside it, and a question asked about such a function against
//! `current_module` alone finds no function rather than finding no
//! effects. Those two answers have the same shape and mean opposite
//! things, so the check reported that nothing was needed for exactly
//! the functions it could not see.
//!
//! What that cost: a perform whose effect had no frame went ahead, and
//! the handler op it resolved statically read an implicit `self` that
//! nothing supplied. The failure arrived as a jump to an address that
//! was not code, several frames below the call the host made, with
//! nothing to say which operation was meant.
//!
//! One module hid it, because then `current_module` is the right one.

use zynml::{ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE};
use zyntax_embed::{LanguageGrammar, TieredConfig, TieredRuntime};

/// The effect, its stateful handler, and three frames: the entry
/// declares nothing, the middle declares nothing, and the one that
/// performs declares the effect.
const FIRST: &str = r#"
effect Events {
    def id_pct(): i64
}

handler Play for Events {
    var pct: i64 = 7
    def id_pct(): i64 { return self.pct }
}

@effect(Events)
def readid_pct(): i64 { return id_pct() }

def view(): i64 { return readid_pct() }

def entry(): i64 { return view() }
"#;

/// A second module, so the first one stops being `current_module`. Its
/// contents do not matter; that it arrives after does.
const SECOND: &str = "def unrelated(): i64 { return 1 }";

fn two_module_runtime() -> TieredRuntime {
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        _ => Ok(None),
    }));
    let g = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("grammar");
    rt.register_grammar("zynml", g);
    rt.load_module("zynml", FIRST).expect("first module");
    rt.load_module("zynml", SECOND).expect("second module");
    rt
}

/// The function that performs is in the earlier module, so the check
/// used to look for it in the wrong one and answer that it needed
/// nothing.
#[test]
fn the_performing_function_is_found_in_its_own_module() {
    let rt = two_module_runtime();
    let missing = rt
        .missing_stateful_handler("readid_pct")
        .expect("it declares a stateful effect and no handler is installed");
    assert_eq!(missing.1, "Events");
}

/// And so is a caller several frames above it, which declares nothing
/// of its own.
#[test]
fn a_caller_that_declares_nothing_is_reported_too() {
    let rt = two_module_runtime();
    assert_eq!(
        rt.missing_stateful_handler("entry").map(|m| m.1),
        Some("Events".to_string()),
        "`entry` reaches the perform through `view`"
    );
}

/// The call is refused rather than made. Before this it went ahead and
/// the perform jumped to an address that was not code.
#[test]
fn the_call_is_refused_rather_than_jumping() {
    let rt = two_module_runtime();
    let err = rt
        .call_raw("entry", &[])
        .expect_err("a call reaching an unhandled stateful perform must be refused");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Events"),
        "the refusal should name the effect, got: {msg}"
    );
}

/// With the handler installed there is nothing to report and the call
/// runs, so the check above is about the missing frame rather than
/// about the module the function came from.
#[test]
fn nothing_is_reported_once_a_handler_is_installed() {
    let mut rt = two_module_runtime();
    let token = rt.get_effect_handler("Play").expect("handler");
    let instance = rt.new_handler_instance(token).expect("instance");
    let _frame = rt.push_handler_instance(instance).expect("push");

    assert!(rt.missing_stateful_handler("entry").is_none());

    // Called the way a host on a render path calls: by pointer, which
    // no guard can intercept.
    let ptr = rt.get_function_ptr("entry").expect("a compiled entry");
    let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    assert_eq!(f(), 7, "the perform reaches the handler's own state");
}
