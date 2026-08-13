//! A handler whose storage is an extern struct the HOST owns.
//!
//! The alternative — a plain `var content: T` on the handler — keeps
//! the value in a runtime-internal region that the host cannot read by
//! offset (see `handler_self_to_extern.rs`). Putting the storage in an
//! extern struct moves the layout to the host side, so reads are
//! ordinary host reads and the write itself is the notification.
//!
//! What has to hold:
//!
//! 1. an extern struct with an impl block compiles, and its methods
//!    reach host symbols by the exact names the host registered,
//! 2. a handler can hold one as a field and initialise it by calling a
//!    static method, so each instance mints its own,
//! 3. `self` passes from a method body into the extern call,
//! 4. two instances address two distinct cells.

use std::sync::Mutex;
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{
    TieredConfig, TieredRuntime, TypeCategory, TypeFlags, TypeTag, ZrtlSigFlags, ZrtlSymbolSig,
    ZyntaxValue,
};

/// Host-owned storage. A real integration would key subscribers off the
/// same cell; here it only has to prove the value round-trips through
/// memory the host allocated and can read without knowing anything
/// about the runtime's layout.
struct Cell {
    value: i64,
}

/// Cells the host has minted, as addresses — a raw pointer is not
/// `Send`, and the address is all the bookkeeping needs.
static CELLS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// Every `set` the host observed, as (cell address, value).
static WRITES: Mutex<Vec<(usize, i64)>> = Mutex::new(Vec::new());
/// `CELLS` and `WRITES` are process-wide, so a test that counts them
/// has to be the only one running. Held for the body of each test that
/// asserts on either.
static EXCLUSIVE: Mutex<()> = Mutex::new(());

/// Take the process-wide lock, ignoring a previous test's panic.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner())
}

extern "C" fn blinc_signal_cell_new() -> *mut Cell {
    let boxed = Box::into_raw(Box::new(Cell { value: 0 }));
    CELLS.lock().unwrap().push(boxed as usize);
    boxed
}

extern "C" fn blinc_signal_cell_get(c: *mut Cell) -> i64 {
    if c.is_null() {
        return -1;
    }
    unsafe { (*c).value }
}

extern "C" fn blinc_signal_cell_set(c: *mut Cell, v: i64) {
    if c.is_null() {
        return;
    }
    unsafe { (*c).value = v };
    WRITES.lock().unwrap().push((c as usize, v));
}

fn reset() {
    // Free anything a previous test minted, so pointer comparisons
    // can't be confused by an address the allocator reused.
    let mut cells = CELLS.lock().unwrap();
    for c in cells.drain(..) {
        drop(unsafe { Box::from_raw(c as *mut Cell) });
    }
    WRITES.lock().unwrap().clear();
}

fn ptr_tag() -> TypeTag {
    TypeTag::new(TypeCategory::Pointer, 0, TypeFlags::NONE)
}

fn sig(params: &[TypeTag], ret: TypeTag) -> ZrtlSymbolSig {
    let mut slots = [TypeTag::VOID; 16];
    slots[..params.len()].copy_from_slice(params);
    ZrtlSymbolSig {
        param_count: params.len() as u8,
        flags: ZrtlSigFlags::NONE,
        return_type: ret,
        params: slots,
    }
}

fn runtime() -> TieredRuntime {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    let mut rt = TieredRuntime::new(config).expect("runtime should start");
    rt.register_function_typed(
        "blinc_signal_cell_new",
        blinc_signal_cell_new as *const u8,
        sig(&[], ptr_tag()),
    );
    rt.register_function_typed(
        "blinc_signal_cell_get",
        blinc_signal_cell_get as *const u8,
        sig(&[ptr_tag()], TypeTag::I64),
    );
    rt.register_function_typed(
        "blinc_signal_cell_set",
        blinc_signal_cell_set as *const u8,
        sig(&[ptr_tag(), TypeTag::I64], TypeTag::VOID),
    );
    rt.finalize_runtime_symbols().expect("publish host symbols");
    rt
}

fn compile(src: &str) -> Result<TieredRuntime, String> {
    let mut rt = runtime();
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).map_err(|e| format!("grammar: {e:?}"))?;
    let program = grammar
        .parse_with_filename(src, "<extern_struct_handler_storage>")
        .map_err(|e| format!("parse: {e:?}"))?;
    rt.compile_typed_program(program)
        .map_err(|e| format!("compile: {e}"))?;
    Ok(rt)
}

fn brief<T>(outcome: &Result<T, String>) -> String {
    match outcome {
        Ok(_) => "ok".to_string(),
        Err(e) => e.chars().take(100).collect::<String>().replace('\n', " "),
    }
}

/// The extern-struct declaration and its impl, generic over T.
const CELL_DECL: &str = r#"
extern struct SignalCell<T>

extern def blinc_signal_cell_new(): SignalCell<i64>
extern def blinc_signal_cell_get(c: SignalCell<i64>): i64
extern def blinc_signal_cell_set(c: SignalCell<i64>, v: i64)

impl<T> SignalCell<T> {
    def init(): SignalCell<T> { extern blinc_signal_cell_new() }
    def get(self): T { extern blinc_signal_cell_get(self) }
    def set(self, v: T) { extern blinc_signal_cell_set(self, v) }
}
"#;

/// How the handler field can name the constructor. Reported, because
/// the call syntax for a static method on a generic extern struct is
/// the part most likely to differ from what the sketch assumes.
#[test]
fn which_constructor_call_form_initialises_the_field() {
    let _guard = exclusive();
    let forms = [
        ("SignalCell.init()", "SignalCell.init()"),
        ("SignalCell<i64>.init()", "SignalCell<i64>.init()"),
        ("extern call direct", "extern blinc_signal_cell_new()"),
    ];
    let mut seen: Vec<(&str, bool)> = Vec::new();
    for (label, ctor) in forms {
        reset();
        let src = format!(
            r#"{CELL_DECL}
effect SignalI64 {{
    def get(): i64
    def set(val: i64)
}}

handler MintedSignalI64 for SignalI64 {{
    var cell: SignalCell<i64> = {ctor}
    def get(): i64 {{ return self.cell.get() }}
    def set(val: i64) {{ self.cell.set(val) }}
}}
"#
        );
        let outcome = compile(&src);
        println!("CTOR {label:<24} -> {}", brief(&outcome));
        seen.push((label, outcome.is_ok()));
    }
    assert_eq!(
        seen,
        vec![
            ("SignalCell.init()", true),
            // Turbofish on a static call is not in the grammar. Pinned
            // so the emitter is not written against a spelling that
            // only looks plausible.
            ("SignalCell<i64>.init()", false),
            ("extern call direct", true),
        ]
    );
}

/// The declaration alone compiles: extern struct plus a generic impl
/// whose method bodies are extern calls.
#[test]
fn an_extern_struct_with_a_generic_impl_compiles() {
    let _guard = exclusive();
    reset();
    let outcome = compile(CELL_DECL);
    assert!(outcome.is_ok(), "{}", brief(&outcome));
}

/// The value round-trips through host-owned memory: a `set` performed
/// through the effect lands in the host's `Cell`, and a `get` reads it
/// back out. No knowledge of the runtime's handler region is involved.
#[test]
fn a_value_round_trips_through_host_owned_storage() {
    let _guard = exclusive();
    reset();
    let src = format!(
        r#"{CELL_DECL}
effect SignalI64 {{
    def get(): i64
    def set(val: i64)
}}

handler MintedSignalI64 for SignalI64 {{
    var cell: SignalCell<i64> = SignalCell.init()
    def get(): i64 {{ return self.cell.get() }}
    def set(val: i64) {{ self.cell.set(val) }}
}}

@effect(SignalI64)
def write(v: i64): i64 {{ set(v) return 0 }}

@effect(SignalI64)
def read(): i64 {{ return get() }}

def main(): i64 {{
    let mut out: i64 = 0
    with MintedSignalI64 {{
        write(41)
        out = read()
    }}
    return out
}}
"#
    );

    let compiled = compile(&src);
    let Ok(rt) = compiled else {
        panic!("compile: {}", brief(&compiled));
    };
    let ran = rt.call::<i64>("main", &[]).map_err(|e| e.to_string());
    let writes = WRITES.lock().unwrap().clone();
    let cells = CELLS.lock().unwrap().len();
    println!("ROUNDTRIP main={ran:?} cells={cells} writes={writes:x?}");

    assert_eq!(ran, Ok(41), "the value read back is the one written");
    assert_eq!(writes.len(), 1, "one set, one host-observed write");
    assert_eq!(writes[0].1, 41, "the host saw the written value");
}

/// Two live instances mint two cells, and each write names its own.
/// This is the identity claim: the cell the host allocated, not an
/// address the runtime's allocator may recycle.
#[test]
fn two_instances_mint_two_cells() {
    let _guard = exclusive();
    reset();
    let src = format!(
        r#"{CELL_DECL}
effect SignalI64 {{
    def get(): i64
    def set(val: i64)
}}

handler MintedSignalI64 for SignalI64 {{
    var cell: SignalCell<i64> = SignalCell.init()
    def get(): i64 {{ return self.cell.get() }}
    def set(val: i64) {{ self.cell.set(val) }}
}}

@effect(SignalI64)
def write(v: i64): i64 {{ set(v) return 0 }}
"#
    );

    let compiled = compile(&src);
    let Ok(mut rt) = compiled else {
        panic!("compile: {}", brief(&compiled));
    };

    let token = rt.get_effect_handler("MintedSignalI64").expect("resolve");
    let a = rt.new_handler_instance(token).expect("mint a");
    let b = rt.new_handler_instance(token).expect("mint b");

    let frame_a = rt.push_handler_instance(a).expect("install a");
    let _ = rt.call::<i64>("write", &[ZyntaxValue::Int(101)]);
    rt.pop_effect_handler(frame_a);

    let frame_b = rt.push_handler_instance(b).expect("install b");
    let _ = rt.call::<i64>("write", &[ZyntaxValue::Int(202)]);
    rt.pop_effect_handler(frame_b);

    let writes = WRITES.lock().unwrap().clone();
    let cells = CELLS.lock().unwrap().len();
    println!("INSTANCES cells={cells} writes={writes:x?}");

    assert_eq!(cells, 2, "each instance ran the field initialiser once");
    assert_eq!(writes.len(), 2, "one write per instance");
    assert_ne!(
        writes[0].0, writes[1].0,
        "each instance wrote to its own cell"
    );
    assert_eq!((writes[0].1, writes[1].1), (101, 202));
}

// ---------------------------------------------------------------------
// Ablation: which construct trips `lowering.rs` symbol lookup
// ---------------------------------------------------------------------

/// Compile without letting a compiler panic abort the run, so one pass
/// maps every combination instead of stopping at the first bad one.
fn try_compile(src: &str) -> Result<(), String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile(src).map(|_| ())));
    std::panic::set_hook(hook);
    match outcome {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(format!("error: {}", e.chars().take(80).collect::<String>())),
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic>".into());
            Err(format!(
                "PANIC: {}",
                msg.chars().take(80).collect::<String>()
            ))
        }
    }
}

/// Extern-struct methods resolve from every caller, generic or not.
///
/// Four sources varying two axes: whether the struct and its impl are
/// generic, and whether the caller is a plain function or a handler
/// field plus handler ops. Handler ops are synthesized functions with
/// their own implicit `self`, so a method call on a FIELD inside one
/// resolves a receiver that the plain-function case never exercises —
/// which is why the handler column is worth keeping separate.
#[test]
fn extern_struct_methods_resolve_from_plain_fns_and_handlers_alike() {
    let _guard = exclusive();
    let concrete_decl = r#"
extern struct SignalCell

impl SignalCell {
    def init(): SignalCell { extern blinc_signal_cell_new() }
    def get(self): i64 { extern blinc_signal_cell_get(self) }
    def set(self, v: i64) { extern blinc_signal_cell_set(self, v) }
}
"#;
    let generic_decl = r#"
extern struct SignalCell<T>

impl<T> SignalCell<T> {
    def init(): SignalCell<T> { extern blinc_signal_cell_new() }
    def get(self): T { extern blinc_signal_cell_get(self) }
    def set(self, v: T) { extern blinc_signal_cell_set(self, v) }
}
"#;

    let plain_user = |ty: &str| {
        format!(
            r#"
def main(): i64 {{
    let c: {ty} = SignalCell.init()
    c.set(41)
    return c.get()
}}
"#
        )
    };
    let handler_user = |ty: &str| {
        format!(
            r#"
effect SignalI64 {{
    def get(): i64
    def set(val: i64)
}}

handler MintedSignalI64 for SignalI64 {{
    var cell: {ty} = SignalCell.init()
    def get(): i64 {{ return self.cell.get() }}
    def set(val: i64) {{ self.cell.set(val) }}
}}
"#
        )
    };

    let rows = [
        (
            "concrete + plain fn",
            concrete_decl,
            plain_user("SignalCell"),
        ),
        (
            "concrete + handler",
            concrete_decl,
            handler_user("SignalCell"),
        ),
        (
            "generic  + plain fn",
            generic_decl,
            plain_user("SignalCell<i64>"),
        ),
        (
            "generic  + handler",
            generic_decl,
            handler_user("SignalCell<i64>"),
        ),
    ];

    for (label, decl, user) in rows {
        reset();
        let src = format!("{decl}{user}");
        let outcome = try_compile(&src);
        println!(
            "ABLATE {label:<22} -> {}",
            match &outcome {
                Ok(()) => "ok".to_string(),
                Err(e) => e.clone(),
            }
        );
        assert!(
            outcome.is_ok(),
            "{label} must resolve: {}",
            outcome.unwrap_err()
        );
    }
}

/// Whether a handler can hold an extern struct at all, with no method
/// call anywhere: the field is initialised by a direct extern call and
/// never read. Separates "extern struct as handler state" from
/// "extern-struct METHODS from a handler body".
#[test]
fn a_handler_field_can_be_an_extern_struct_without_methods() {
    let _guard = exclusive();
    reset();
    let src = r#"
extern struct SignalCell

extern def blinc_signal_cell_new(): SignalCell
extern def blinc_signal_cell_get(c: SignalCell): i64
extern def blinc_signal_cell_set(c: SignalCell, v: i64)

effect SignalI64 {
    def get(): i64
    def set(val: i64)
}

handler MintedSignalI64 for SignalI64 {
    var cell: SignalCell = blinc_signal_cell_new()
    def get(): i64 { return blinc_signal_cell_get(self.cell) }
    def set(val: i64) { blinc_signal_cell_set(self.cell, val) }
}
"#;
    let outcome = try_compile(src);
    println!("BARE-EXTERN-FNS -> {outcome:?}");
    assert!(
        outcome.is_ok(),
        "an extern struct as handler state, with no methods involved: {}",
        outcome.unwrap_err()
    );
}

/// Both ways a handler reaches an impl method work, separately and
/// together: a static method in the field initialiser, and an instance
/// method called on that field from an op body.
///
/// Kept as two axes rather than one combined case because they resolve
/// differently — the initialiser runs in the synthesized constructor
/// while the instance call runs in an op body, and only the latter has
/// to distinguish the handler's own `self` from the field's receiver.
#[test]
fn a_handler_reaches_impl_methods_from_both_the_initialiser_and_op_bodies() {
    let _guard = exclusive();
    let decl = r#"
extern struct SignalCell

impl SignalCell {
    def init(): SignalCell { extern blinc_signal_cell_new() }
    def get(self): i64 { extern blinc_signal_cell_get(self) }
    def set(self, v: i64) { extern blinc_signal_cell_set(self, v) }
}

extern def blinc_signal_cell_new(): SignalCell
extern def blinc_signal_cell_get(c: SignalCell): i64
extern def blinc_signal_cell_set(c: SignalCell, v: i64)

effect SignalI64 {
    def get(): i64
    def set(val: i64)
}
"#;

    // (label, field initialiser, get body, set body)
    let rows = [
        (
            "neither (control)",
            "blinc_signal_cell_new()",
            "return blinc_signal_cell_get(self.cell)",
            "blinc_signal_cell_set(self.cell, val)",
        ),
        (
            "static method in field init",
            "SignalCell.init()",
            "return blinc_signal_cell_get(self.cell)",
            "blinc_signal_cell_set(self.cell, val)",
        ),
        (
            "instance method in op body",
            "blinc_signal_cell_new()",
            "return self.cell.get()",
            "self.cell.set(val)",
        ),
        (
            "both",
            "SignalCell.init()",
            "return self.cell.get()",
            "self.cell.set(val)",
        ),
    ];

    for (label, init, get_body, set_body) in rows {
        reset();
        let src = format!(
            r#"{decl}
handler MintedSignalI64 for SignalI64 {{
    var cell: SignalCell = {init}
    def get(): i64 {{ {get_body} }}
    def set(val: i64) {{ {set_body} }}
}}
"#
        );
        let outcome = try_compile(&src);
        println!(
            "HANDLER-USE {label:<28} -> {}",
            match &outcome {
                Ok(()) => "ok".to_string(),
                Err(e) => e.clone(),
            }
        );
        assert!(
            outcome.is_ok(),
            "{label} must resolve: {}",
            outcome.unwrap_err()
        );
    }
}
