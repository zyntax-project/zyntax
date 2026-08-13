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

def plain(): i64 { return 7 }

def handles_it_itself(): i64 {
    let mut out: i64 = 0
    with Tally {
        out = deferred_body()
    }
    return out
}
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

    // Extent closed. Calling now is refused rather than run — see
    // `a_perform_with_no_handler_in_scope_fails_as_a_value`.

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

/// A perform whose effect has no handler in scope fails as a value.
///
/// Nothing here involves captured contexts; it is what happens when an
/// effectful function is called outside any extent. Left to reach
/// compiled code, the perform site falls back to the statically
/// resolved handler op, which for a stateful handler reads an implicit
/// `self` that nothing supplied. The call is refused at the boundary
/// instead, so the failure is a value the caller can report.
#[test]
fn a_perform_with_no_handler_in_scope_fails_as_a_value() {
    let rt = runtime();
    let out = rt
        .call::<i64>("deferred_body", &[])
        .map_err(|e| e.to_string());
    let Err(message) = out else {
        panic!("expected a refusal, got {out:?}");
    };
    assert!(
        message.contains("Counter") && message.contains("no handler"),
        "the error should name the effect and say what is missing, got: {message}"
    );
}

/// The refusal is specific to effects whose handler carries state. A
/// function that performs nothing is unaffected, so the guard cannot
/// break ordinary calls.
#[test]
fn a_function_that_performs_nothing_still_calls_with_nothing_installed() {
    let rt = runtime();
    let out = rt.call::<i64>("plain", &[]).map_err(|e| e.to_string());
    assert_eq!(
        out,
        Ok(7),
        "a call with no effects is untouched by the guard"
    );
}

/// A function that installs a handler around its own performs is not
/// refused, even though what it calls needs one.
///
/// This is the ordinary way to use an effect, and it is why the check
/// reads only what the entry itself declares: a callee's effects are
/// the callee's business, and the caller has already supplied for them.
#[test]
fn an_entry_that_opens_its_own_handler_scope_is_not_refused() {
    let rt = runtime();
    let out = rt
        .call::<i64>("handles_it_itself", &[])
        .map_err(|e| e.to_string());
    assert_eq!(
        out,
        Ok(1),
        "the `with` scope inside the entry supplies the handler"
    );
}
