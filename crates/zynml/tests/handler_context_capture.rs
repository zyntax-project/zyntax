//! Running a stored callback under the handlers that were in scope
//! where it was written.
//!
//! A host that compiles `computed { ... }` or `on_click = || { ... }`
//! into a zero-argument function stores it and calls it on its own
//! schedule. By then the extent that installed the handlers has
//! closed, so a perform inside the body finds nothing in scope unless
//! the context was captured at registration and reinstated at call
//! time.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime};

/// A counter effect whose handler holds the count, so a wrong context
/// is visible as a wrong number rather than as a crash.
const SRC: &str = r#"
effect Counter {
    def bump()
    def total(): i64
}

handler Tally for Counter {
    var n: i64 = 0
    def bump() { self.n = self.n + 1 }
    def total(): i64 { return self.n }
}

@effect(Counter)
def deferred_body(): i64 { bump() return total() }

@effect(Counter)
def read_total(): i64 { return total() }
"#;

fn runtime() -> TieredRuntime {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    let mut rt = TieredRuntime::new(config).expect("runtime should start");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(SRC, "<handler_context_capture>")
        .expect("parse");
    rt.compile_typed_program(program).expect("compile");
    rt
}

/// The shape the request asks for: capture inside the extent, run the
/// body after the extent has closed, and see the handler that was in
/// scope at capture.
#[test]
fn a_callback_runs_under_the_context_captured_where_it_was_registered() {
    let mut rt = runtime();
    let token = rt.get_effect_handler("Tally").expect("resolve Tally");
    let instance = rt.new_handler_instance(token).expect("mint");

    // The extent that installs the handler, and in which the host
    // would register its callback.
    let frame = rt.push_handler_instance(instance).expect("install");
    let context = rt.capture_handler_context();
    rt.pop_effect_handler(frame);

    // Extent closed. What a perform does here is NOT asserted: with no
    // handler in scope the perform site hands the static fallback a null
    // `self` and the process dies on the dereference, so calling it
    // would take the test with it. See `a_perform_with_no_handler_in_
    // scope_crashes` below.

    // Reinstated: the body runs against the state the extent installed.
    let scope = rt.enter_handler_context(&context);
    let first = rt
        .call::<i64>("deferred_body", &[])
        .map_err(|e| e.to_string());
    rt.leave_handler_context(scope);
    assert_eq!(first, Ok(1), "the captured handler counted the bump");

    // And the state persisted, so a second firing continues it rather
    // than starting a fresh machine.
    let scope = rt.enter_handler_context(&context);
    let second = rt
        .call::<i64>("deferred_body", &[])
        .map_err(|e| e.to_string());
    rt.leave_handler_context(scope);
    assert_eq!(second, Ok(2), "the same state, not a second instance");

    rt.release_handler_context(context);
}

/// Entering layers on what is already installed rather than replacing
/// it, which is what lets a callback register another callback.
#[test]
fn entering_layers_on_the_current_stack_and_leaves_it_as_found() {
    let mut rt = runtime();
    let token = rt.get_effect_handler("Tally").expect("resolve Tally");

    let outer = rt.new_handler_instance(token).expect("mint outer");
    let inner = rt.new_handler_instance(token).expect("mint inner");

    // Capture a context holding `inner`.
    let f = rt.push_handler_instance(inner).expect("install inner");
    let captured = rt.capture_handler_context();
    rt.pop_effect_handler(f);

    // Now install `outer` and run the captured context inside it.
    let outer_frame = rt.push_handler_instance(outer).expect("install outer");
    let _ = rt
        .call::<i64>("deferred_body", &[])
        .map_err(|e| e.to_string());

    let scope = rt.enter_handler_context(&captured);
    let inside = rt
        .call::<i64>("deferred_body", &[])
        .map_err(|e| e.to_string());
    rt.leave_handler_context(scope);
    assert_eq!(inside, Ok(1), "the layered context handled the perform");

    // Outer is intact and still holds its own count.
    let after = rt.call::<i64>("read_total", &[]).map_err(|e| e.to_string());
    assert_eq!(after, Ok(1), "leaving restored the stack it found");

    rt.pop_effect_handler(outer_frame);
    rt.release_handler_context(captured);
}

/// A captured context keeps its handler state alive after the owner
/// has let the instance go, so a callback stored past the extent does
/// not read freed state.
#[test]
fn a_captured_context_keeps_its_handler_state_alive() {
    let mut rt = runtime();
    let token = rt.get_effect_handler("Tally").expect("resolve Tally");
    let instance = rt.new_handler_instance(token).expect("mint");

    let frame = rt.push_handler_instance(instance).expect("install");
    let context = rt.capture_handler_context();
    let _ = rt
        .call::<i64>("deferred_body", &[])
        .map_err(|e| e.to_string());
    rt.pop_effect_handler(frame);
    // The owner is done with it; only the capture still names it.
    rt.drop_handler_instance(instance);

    let scope = rt.enter_handler_context(&context);
    let out = rt
        .call::<i64>("deferred_body", &[])
        .map_err(|e| e.to_string());
    rt.leave_handler_context(scope);
    assert_eq!(
        out,
        Ok(2),
        "the state the capture held is still the one read"
    );

    rt.release_handler_context(context);
}

/// A host callback that re-enters the runtime while the context is
/// installed. The request calls this out: entering must not leave a
/// runtime lock held, or a body calling a host extern deadlocks.
#[test]
fn the_runtime_is_reentrant_while_a_context_is_installed() {
    let mut rt = runtime();
    let token = rt.get_effect_handler("Tally").expect("resolve Tally");
    let instance = rt.new_handler_instance(token).expect("mint");
    let frame = rt.push_handler_instance(instance).expect("install");
    let context = rt.capture_handler_context();
    rt.pop_effect_handler(frame);

    let scope = rt.enter_handler_context(&context);
    // Everything here is what a callback body might do: call compiled
    // code, and ask the runtime questions while doing it.
    let a = rt
        .call::<i64>("deferred_body", &[])
        .map_err(|e| e.to_string());
    let names: Vec<String> = rt.functions().into_iter().map(String::from).collect();
    let b = rt.call::<i64>("read_total", &[]).map_err(|e| e.to_string());
    rt.leave_handler_context(scope);

    assert_eq!(a, Ok(1));
    assert_eq!(b, Ok(1), "reading back through the same installed context");
    assert!(
        names.iter().any(|n| n == "deferred_body"),
        "the runtime answered while a context was installed"
    );

    rt.release_handler_context(context);
}

/// A perform with no handler in scope kills the process instead of
/// failing.
///
/// Nothing here involves captured contexts; it is what happens when an
/// effectful function is called outside any extent. The perform site
/// falls back to the statically resolved handler op, and for a stateful
/// handler that op takes an implicit `self` that
/// `__zyntax_effect_lookup_state` returns null for, so the first field
/// access dereferences null.
///
/// This matters to the capture API rather than being caused by it: a
/// host that forgets to reinstate a context, or reinstates one that
/// does not cover the effect, gets a SIGSEGV rather than an error it
/// can report. Ignored because the failure is a signal, which takes the
/// whole test binary with it rather than failing one case.
#[test]
#[ignore = "crashes the test binary: a stateful perform with no handler in scope \
            dereferences a null `self` (SIGSEGV), rather than failing as a value"]
fn a_perform_with_no_handler_in_scope_crashes() {
    let rt = runtime();
    let out = rt
        .call::<i64>("deferred_body", &[])
        .map_err(|e| e.to_string());
    // Never reached today.
    assert!(out.is_err(), "expected a reportable error, got {out:?}");
}
