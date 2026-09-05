//! A call is refused for what it reaches, not for what it declares.
//!
//! A perform whose effect has no frame in scope resolves its handler op
//! statically, and a stateful op then reads an implicit `self` that
//! nothing supplied. `call_raw` refuses such a call rather than letting
//! it reach compiled code.
//!
//! It used to ask only what the called function itself declared. An
//! entry point declares its own contract, and a function that performs
//! is often several frames below one that declares nothing, so the
//! check passed and the perform went ahead with no frame: the failure
//! arrived as a jump to an address that was not code, with nothing to
//! say which operation was meant or which function performed it.
//!
//! A host that must call by function pointer cannot be refused by
//! anything, because nothing is asked. `missing_stateful_handler` is
//! the same question asked on its own, for that route.

use zynml::{ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE};
use zyntax_embed::{LanguageGrammar, TieredConfig, TieredRuntime};

const SRC: &str = r#"
effect Ledger {
    def note(): i64
}

handler Tally for Ledger {
    var n: i64 = 0
    def note(): i64 { self.n = self.n + 1  return self.n }
}

// Declares the effect, and performs it.
@effect(Ledger)
def deep(): i64 {
    return note()
}

// Declares nothing, and calls something that performs.
def middle(): i64 {
    return deep()
}

// Declares nothing either. This is what a host calls.
def entry(): i64 {
    return middle()
}
"#;

fn runtime() -> TieredRuntime {
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        _ => Ok(None),
    }));
    let g = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("grammar");
    rt.register_grammar("zynml", g);
    rt.load_module("zynml", SRC).expect("should compile");
    rt
}

/// The entry point declares nothing, so the old check passed it. What
/// it reaches is what decides.
#[test]
fn an_entry_that_declares_nothing_is_still_refused() {
    let rt = runtime();
    let err = rt
        .call_raw("entry", &[])
        .expect_err("a call reaching an unhandled stateful perform should be refused");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Ledger"),
        "the refusal should name the effect that has no frame, got: {msg}"
    );
    assert!(msg.contains("entry"), "and the call it refused, got: {msg}");
}

/// The same question, asked on its own, for a host that calls by
/// pointer and so cannot be refused by anything.
#[test]
fn the_check_is_callable_without_making_the_call() {
    let rt = runtime();
    let missing = rt
        .missing_stateful_handler("entry")
        .expect("the effect reached from `entry` has no handler on this thread");
    assert_eq!(missing.1, "Ledger");

    // And the function that performs it answers the same way, so the
    // walk is not reporting only the frames it started from.
    assert_eq!(
        rt.missing_stateful_handler("deep").map(|m| m.1),
        Some("Ledger".to_string())
    );
}

/// A function that opens the scope itself needs nothing from its
/// caller, so it is not reported. Without this the ordinary way to use
/// a handler would read as an error.
#[test]
fn a_function_that_handles_it_is_not_reported() {
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        _ => Ok(None),
    }));
    let g = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("grammar");
    rt.register_grammar("zynml", g);
    rt.load_module(
        "zynml",
        r#"
effect Ledger {
    def note(): i64
}

handler Tally for Ledger {
    var n: i64 = 0
    def note(): i64 { self.n = self.n + 1  return self.n }
}

@effect(Ledger)
def deep(): i64 {
    return note()
}

def entry(): i64 {
    with Tally {
        return deep()
    }
}
"#,
    )
    .expect("should compile");

    assert!(
        rt.missing_stateful_handler("entry").is_none(),
        "a function that opens the scope around the perform supplies the frame itself"
    );
}
