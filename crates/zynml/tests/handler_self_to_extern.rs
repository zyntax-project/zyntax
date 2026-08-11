//! Can a handler op body call out to a host extern, and can it pass
//! `self`?
//!
//! This is the load-bearing detail of a "signal as effect handle"
//! design where a write notifies the host: the handler owns the
//! storage, and `set` tells the host something changed by calling an
//! extern with its own context. Three things have to hold, and they
//! fail independently:
//!
//! 1. a handler op body can call an extern at all,
//! 2. `self` is passable as an argument rather than only usable as a
//!    field-access base,
//! 3. what arrives on the host side is the ADDRESS of the handler's
//!    storage, not a copy — otherwise the host reads a temporary and
//!    every value it sees is stale by construction.
//!
//! Deliberately concrete (no generics) and free of `Optional` / `Null`,
//! so it isolates the extern question from anything else in flight.

use std::sync::Mutex;
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{
    TieredConfig, TieredRuntime, TypeCategory, TypeFlags, TypeTag, ZrtlSigFlags, ZrtlSymbolSig,
    ZyntaxValue,
};

/// What each host extern saw, in call order.
static SEEN_UNIT: Mutex<Vec<()>> = Mutex::new(Vec::new());
static SEEN_INT: Mutex<Vec<i64>> = Mutex::new(Vec::new());
static SEEN_PTR: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// `SEEN_PTR` is process-wide, so a test that counts notifications has
/// to be the only one running.
static EXCLUSIVE: Mutex<()> = Mutex::new(());

/// Take the process-wide lock, ignoring a previous test's panic.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner())
}

extern "C" fn host_notify0() {
    SEEN_UNIT.lock().unwrap().push(());
}

extern "C" fn host_notify_int(v: i64) {
    SEEN_INT.lock().unwrap().push(v);
}

extern "C" fn host_notify_ptr(p: *const u8) {
    SEEN_PTR.lock().unwrap().push(p as usize);
}

fn reset() {
    SEEN_UNIT.lock().unwrap().clear();
    SEEN_INT.lock().unwrap().clear();
    SEEN_PTR.lock().unwrap().clear();
}

/// Read the first `words` u64s at `addr`, refusing anything that can't
/// be a live allocation. A handler that passes a COPY of an i64 field
/// rather than its address shows up here as a small integer, and
/// dereferencing that would abort the test process instead of
/// reporting the finding.
fn peek(addr: usize, words: usize) -> Option<Vec<u64>> {
    if addr < 0x1000 || addr % 8 != 0 {
        return None;
    }
    // Safety: the guard above rejects obvious non-pointers, and the
    // only values reaching here come from a handler region the runtime
    // allocated and still owns for the duration of the call.
    Some(
        (0..words)
            .map(|i| unsafe { *(addr as *const u64).add(i) })
            .collect(),
    )
}

fn ptr_tag() -> TypeTag {
    TypeTag::new(TypeCategory::Pointer, 0, TypeFlags::NONE)
}

fn sig(params: &[TypeTag]) -> ZrtlSymbolSig {
    let mut slots = [TypeTag::VOID; 16];
    slots[..params.len()].copy_from_slice(params);
    ZrtlSymbolSig {
        param_count: params.len() as u8,
        flags: ZrtlSigFlags::NONE,
        return_type: TypeTag::VOID,
        params: slots,
    }
}

/// A runtime with the three host externs published, ready to compile
/// against. Symbols are finalized BEFORE compilation so the extern
/// declarations resolve during lowering rather than at first call.
fn runtime_with_host_externs() -> TieredRuntime {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    let mut rt = TieredRuntime::new(config).expect("runtime should start");

    rt.register_function_typed("host_notify0", host_notify0 as *const u8, sig(&[]));
    rt.register_function_typed(
        "host_notify_int",
        host_notify_int as *const u8,
        sig(&[TypeTag::I64]),
    );
    rt.register_function_typed(
        "host_notify_ptr",
        host_notify_ptr as *const u8,
        sig(&[ptr_tag()]),
    );
    rt.finalize_runtime_symbols()
        .expect("host symbols should publish");
    rt
}

/// Compile `src` and call `main`, reporting every stage as a value so
/// one run maps out what the substrate supports rather than aborting on
/// the first form that doesn't.
fn compile_and_run(src: &str) -> Result<i64, String> {
    let mut rt = runtime_with_host_externs();
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).map_err(|e| format!("grammar: {e:?}"))?;
    let program = grammar
        .parse_with_filename(src, "<handler_self_to_extern>")
        .map_err(|e| format!("parse: {e:?}"))?;
    rt.compile_typed_program(program)
        .map_err(|e| format!("compile: {e}"))?;
    rt.call::<i64>("main", &[])
        .map_err(|e| format!("call: {e}"))
}

fn brief(outcome: &Result<i64, String>) -> String {
    match outcome {
        Ok(v) => format!("ran, main = {v}"),
        Err(e) => e.chars().take(110).collect::<String>().replace('\n', " "),
    }
}

/// The shared shape: a handler holding one i64, written through a
/// `set` op, with `$NOTIFY` spliced into the write.
fn program(notify_decl: &str, notify_call: &str) -> String {
    format!(
        r#"
{notify_decl}

effect SignalI64 {{
    def get(): i64
    def set(val: i64)
}}

handler MintedSignalI64 for SignalI64 {{
    var content: i64 = 0
    def get(): i64 {{ return self.content }}
    def set(val: i64) {{
        self.content = val
        {notify_call}
    }}
}}

@effect(SignalI64)
def write(v: i64) {{ set(v) }}

@effect(SignalI64)
def read(): i64 {{ return get() }}

def main(): i64 {{
    let mut out: i64 = 0
    with MintedSignalI64 {{
        write(7)
        out = read()
    }}
    return out
}}
"#
    )
}

/// Which extern-call form a handler op body accepts. Reported rather
/// than asserted: the point is to learn which of these the substrate
/// supports, and a hard failure on the first one hides the rest.
#[test]
fn which_extern_call_form_works_in_a_handler_body() {
    let _guard = exclusive();
    let forms: [(&str, &str, &str); 5] = [
        ("no extern (control)", "", ""),
        (
            "extern, no args",
            "extern def host_notify0()",
            "host_notify0()",
        ),
        (
            "extern, field value",
            "extern def host_notify_int(v: i64)",
            "host_notify_int(self.content)",
        ),
        (
            "extern, self as Ptr",
            "extern def host_notify_ptr(p: Ptr<i8>)",
            "host_notify_ptr(self)",
        ),
        (
            "extern, self as i64",
            "extern def host_notify_int(v: i64)",
            "host_notify_int(self)",
        ),
    ];

    for (label, decl, call) in forms {
        reset();
        let outcome = compile_and_run(&program(decl, call));
        println!(
            "FORM {label:<22} -> {:<56} unit={} int={:?} ptr={:?}",
            brief(&outcome),
            SEEN_UNIT.lock().unwrap().len(),
            SEEN_INT.lock().unwrap(),
            SEEN_PTR
                .lock()
                .unwrap()
                .iter()
                .map(|p| format!("{p:#x}"))
                .collect::<Vec<_>>(),
        );
    }
}

/// The control: the handler works and `set`/`get` round-trip without
/// any extern involved. If this fails, nothing below means anything.
#[test]
fn the_handler_round_trips_without_an_extern() {
    let _guard = exclusive();
    reset();
    assert_eq!(
        compile_and_run(&program("", "")),
        Ok(7),
        "set(7) then get() should read back 7"
    );
}

/// A handler op body can reach a host extern at all.
#[test]
fn a_handler_body_can_call_a_host_extern() {
    let _guard = exclusive();
    reset();
    let outcome = compile_and_run(&program("extern def host_notify0()", "host_notify0()"));
    assert_eq!(outcome, Ok(7), "the write still lands");
    assert_eq!(
        SEEN_UNIT.lock().unwrap().len(),
        1,
        "one write, one notification"
    );
}

/// The region behind `self` is opaque, and reading it by offset is not
/// a supported way to get at a handler field.
///
/// An earlier version of this test dumped 32 words looking for the
/// written value. It found neither value and then segfaulted: the
/// region is around 16 bytes, so the dump ran off the end into
/// unrelated allocations. Even the two in-bounds words held a
/// high-entropy header and a small address rather than the i64 that
/// had just been written, which matches the synthesized `@reference`
/// state region that `handler_state.rs` describes — fields live behind
/// an indirection, not inline at a fixed offset.
///
/// Kept as a negative result. Anything on the host side that wants a
/// handler's value should own the storage itself (an extern struct
/// whose layout the host defines) rather than read the runtime's
/// region, which is internal and free to change.
#[test]
fn the_region_behind_self_is_opaque_to_the_host() {
    let _guard = exclusive();
    reset();
    const WRITTEN: i64 = 0x1111_2222;

    let src = format!(
        r#"
extern def host_notify_ptr(p: Ptr<i8>)

effect SignalI64 {{
    def get(): i64
    def set(val: i64)
}}

handler MintedSignalI64 for SignalI64 {{
    var content: i64 = 0
    def get(): i64 {{ return self.content }}
    def set(val: i64) {{
        self.content = val
        host_notify_ptr(self)
    }}
}}

@effect(SignalI64)
def write(v: i64): i64 {{ set(v) return 0 }}

def main(): i64 {{
    with MintedSignalI64 {{
        write({WRITTEN})
    }}
    return 0
}}
"#
    );

    let outcome = compile_and_run(&src);
    let seen = SEEN_PTR.lock().unwrap().clone();
    assert!(outcome.is_ok(), "program should run: {outcome:?}");
    assert_eq!(seen.len(), 1, "one write, one notification");

    // Two words only. The region is small, and reading past it is what
    // crashed the previous version of this test.
    let words = peek(seen[0], 2).expect("the pointer should at least be readable");
    println!("OPAQUE first 2 words at {:#x}: {words:x?}", seen[0]);
    assert!(
        !words.contains(&(WRITTEN as u64)),
        "if the value ever DOES appear inline at offset 0 or 8, the \
         layout changed and this negative result needs revisiting"
    );
}

/// Sequential `with` scopes can reuse one address.
///
/// This is why a raw pointer is identity only for as long as the
/// instance is alive: the first region is released at scope exit and
/// the allocator is free to hand the same address to the next one.
/// Recorded as an observation because it constrains the host side —
/// subscribers keyed on an address MUST be dropped at unmount, or a
/// later instance inherits them.
#[test]
fn sequential_scopes_may_reuse_one_address() {
    let _guard = exclusive();
    reset();
    let src = r#"
extern def host_notify_ptr(p: Ptr<i8>)

effect SignalI64 {
    def get(): i64
    def set(val: i64)
}

handler MintedSignalI64 for SignalI64 {
    var content: i64 = 0
    def get(): i64 { return self.content }
    def set(val: i64) {
        self.content = val
        host_notify_ptr(self)
    }
}

@effect(SignalI64)
def write(v: i64): i64 { set(v) return 0 }

def main(): i64 {
    with MintedSignalI64 {
        write(1)
    }
    with MintedSignalI64 {
        write(2)
    }
    return 0
}
"#;

    let outcome = compile_and_run(src);
    let seen = SEEN_PTR.lock().unwrap().clone();
    assert!(outcome.is_ok(), "program should run: {outcome:?}");
    assert_eq!(seen.len(), 2, "one write per scope");
    println!(
        "SEQUENTIAL scopes reused the address: {} ({:#x}, {:#x})",
        seen[0] == seen[1],
        seen[0],
        seen[1]
    );
}

/// Two host-minted instances, both alive, notify from distinct
/// addresses.
///
/// This is the case the design actually needs: two mounted components
/// existing at the same time, not one scope following another. Both
/// instances are minted up front so neither region can be recycled
/// into the other, and each is installed around its own write.
#[test]
fn two_live_host_minted_instances_notify_from_distinct_addresses() {
    let _guard = exclusive();
    reset();
    let src = r#"
extern def host_notify_ptr(p: Ptr<i8>)

effect SignalI64 {
    def get(): i64
    def set(val: i64)
}

handler MintedSignalI64 for SignalI64 {
    var content: i64 = 0
    def get(): i64 { return self.content }
    def set(val: i64) {
        self.content = val
        host_notify_ptr(self)
    }
}

@effect(SignalI64)
def write(v: i64): i64 { set(v) return 0 }
"#;

    let mut rt = runtime_with_host_externs();
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(src, "<two_instances>")
        .expect("parse");
    rt.compile_typed_program(program).expect("compile");

    let token = rt.get_effect_handler("MintedSignalI64").expect("resolve");
    // Mint both BEFORE installing either, so the two regions coexist
    // and the allocator cannot hand out one address twice.
    let a = rt.new_handler_instance(token).expect("mint a");
    let b = rt.new_handler_instance(token).expect("mint b");

    let frame_a = rt.push_handler_instance(a).expect("install a");
    let call_a = rt.call::<i64>("write", &[ZyntaxValue::Int(101)]);
    rt.pop_effect_handler(frame_a);

    let frame_b = rt.push_handler_instance(b).expect("install b");
    let call_b = rt.call::<i64>("write", &[ZyntaxValue::Int(202)]);
    rt.pop_effect_handler(frame_b);

    let seen = SEEN_PTR.lock().unwrap().clone();
    println!(
        "INSTANCES call_a={:?} call_b={:?} ptrs={:x?}",
        call_a.as_ref().err().map(|e| e.to_string()),
        call_b.as_ref().err().map(|e| e.to_string()),
        seen
    );

    assert_eq!(seen.len(), 2, "one notification per instance");
    assert_ne!(
        seen[0], seen[1],
        "two live instances must have two storage regions — \
         if these match, the pointer cannot serve as instance identity"
    );
}
