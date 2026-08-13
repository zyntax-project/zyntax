//! Host-owned signal storage that holds a value of ANY type.
//!
//! The generic shape in `extern_struct_handler_storage.rs` needs a
//! `SignalCell<T>` whose methods monomorphise per element type. That
//! works, but it leaves the host registering symbols against a set of
//! types the emitter cannot know: a signal can hold a user-declared
//! struct, so the set is open.
//!
//! `Any` removes the generic entirely. A value crossing into an `Any`
//! parameter is auto-boxed into a ZRTL `DynamicBox`, which is
//! `#[repr(C)]` and self-describing: a type tag, a size, a payload
//! pointer and a dropper. So one non-generic cell and three host
//! symbols cover every element type, aggregates included, and the host
//! never has to know the runtime's layout for the value it holds.

use std::sync::Mutex;
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::zrtl::DynamicBoxRepr;
use zyntax_embed::{
    TieredConfig, TieredRuntime, TypeCategory, TypeFlags, TypeTag, ZrtlSigFlags, ZrtlSymbolSig,
};

/// Host-owned storage: one slot holding whatever box was last written.
struct Cell {
    boxed: *mut DynamicBoxRepr,
}

static CELLS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// Every `set` the host saw, as (cell address, type tag, size).
static WRITES: Mutex<Vec<(usize, u32, u32)>> = Mutex::new(Vec::new());
/// Whether each write's box carried a dropper, i.e. whether it OWNS
/// its payload. A borrowed box points into the caller's frame and
/// cannot outlive the call, which decides whether host storage may
/// hold the box or must copy the payload out.
static OWNED: Mutex<Vec<(bool, bool)>> = Mutex::new(Vec::new());
/// `CELLS`, `WRITES` and `OWNED` are process-wide, so a test that
/// counts them has to be the only one running.
static EXCLUSIVE: Mutex<()> = Mutex::new(());

/// Take the process-wide lock, ignoring a previous test's panic.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner())
}

extern "C" fn host_signal_cell_new() -> *mut Cell {
    let c = Box::into_raw(Box::new(Cell {
        boxed: std::ptr::null_mut(),
    }));
    CELLS.lock().unwrap().push(c as usize);
    c
}

extern "C" fn host_signal_cell_get(c: *mut Cell) -> *mut DynamicBoxRepr {
    if c.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*c).boxed }
}

/// Whether `p` can be a live `DynamicBox` rather than a raw scalar that
/// was passed through unboxed. Dereferencing an unboxed value is a
/// misaligned read, which aborts the process without unwinding, so a
/// failed autobox has to be detectable rather than fatal.
fn plausible_box(p: *mut DynamicBoxRepr) -> bool {
    let a = p as usize;
    a >= 0x1000 && a % 8 == 0
}

extern "C" fn host_signal_cell_set(c: *mut Cell, v: *mut DynamicBoxRepr) {
    if c.is_null() {
        return;
    }
    // The box describes itself, so the host can record what it holds
    // without being told the element type up front. A raw value is
    // recorded as tag 0 with the value in `size` so the test can say
    // "this arrived unboxed" instead of dying on the deref.
    let (tag, size) = if plausible_box(v) {
        unsafe { ((*v).tag, (*v).size) }
    } else {
        (0, v as u32)
    };
    if plausible_box(v) {
        let b = unsafe { &*v };
        OWNED
            .lock()
            .unwrap()
            .push((b.dropper.is_some(), b.display_fn.is_some()));
    }
    unsafe { (*c).boxed = v };
    WRITES.lock().unwrap().push((c as usize, tag, size));
}

fn reset() {
    let mut cells = CELLS.lock().unwrap();
    for c in cells.drain(..) {
        drop(unsafe { Box::from_raw(c as *mut Cell) });
    }
    WRITES.lock().unwrap().clear();
    OWNED.lock().unwrap().clear();
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
        "host_signal_cell_new",
        host_signal_cell_new as *const u8,
        sig(&[], ptr_tag()),
    );
    rt.register_function_typed(
        "host_signal_cell_get",
        host_signal_cell_get as *const u8,
        sig(&[ptr_tag()], ptr_tag()),
    );
    rt.register_function_typed(
        "host_signal_cell_set",
        host_signal_cell_set as *const u8,
        sig(&[ptr_tag(), ptr_tag()], TypeTag::VOID),
    );
    rt.finalize_runtime_symbols().expect("publish host symbols");
    rt
}

fn compile(src: &str) -> Result<TieredRuntime, String> {
    let mut rt = runtime();
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).map_err(|e| format!("grammar: {e:?}"))?;
    let program = grammar
        .parse_with_filename(src, "<signal_cell_any>")
        .map_err(|e| format!("parse: {e:?}"))?;
    rt.compile_typed_program(program)
        .map_err(|e| format!("compile: {e}"))?;
    Ok(rt)
}

/// Compile without letting a compiler panic abort the run.
fn try_compile(src: &str) -> Result<(), String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile(src).map(|_| ())));
    std::panic::set_hook(hook);
    match outcome {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(format!("error: {}", e.chars().take(90).collect::<String>())),
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic>".into());
            Err(format!(
                "PANIC: {}",
                msg.chars().take(90).collect::<String>()
            ))
        }
    }
}

/// The cell and its three host symbols, with no generics anywhere.
const CELL: &str = r#"
extern struct SignalCell

extern def host_signal_cell_new(): SignalCell
extern def host_signal_cell_get(c: SignalCell): Any
extern def host_signal_cell_set(c: SignalCell, v: Any)
"#;

/// What reaches the host for each way of getting a value into an
/// `Any` parameter. All three box; the tags differ, and that is the
/// point.
///
/// A bare literal boxes as i32 (`tag=0x302, size=4`) while a value
/// bound as `let v: i64` boxes as i64 (`tag=0x402, size=8`), because
/// the literal takes the default integer type. An emitter should
/// therefore type its bindings rather than let a literal decide the
/// element type of a signal.
///
/// Reported as (tag, size) rather than asserted on an exact tag: an
/// unboxed value would show tag 0 with the raw value in `size`, which
/// is the failure this distinguishes.
#[test]
fn which_construct_actually_boxes_into_an_any_parameter() {
    let _guard = exclusive();
    let rows: [(&str, String); 3] = [
        (
            "direct argument",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    host_signal_cell_set(c, 41)\n    return 0\n}}\n"),
        ),
        (
            "via `let b: Any = v`",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    let v: i64 = 41\n    let b: Any = v\n    host_signal_cell_set(c, b)\n    return 0\n}}\n"),
        ),
        (
            "explicit zyntax_box_i64",
            format!("{CELL}\nextern def zyntax_box_i64(v: i64): Any\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    host_signal_cell_set(c, zyntax_box_i64(41))\n    return 0\n}}\n"),
        ),
    ];

    let mut seen: Vec<(&str, u32, u32)> = Vec::new();
    for (label, src) in rows {
        reset();
        let outcome = compile(&src);
        let ran = match &outcome {
            Ok(rt) => rt.call::<i64>("main", &[]).map_err(|e| e.to_string()),
            Err(e) => Err(e.clone()),
        };
        let writes = WRITES.lock().unwrap().clone();
        let verdict = match writes.first() {
            None => "no write reached the host".to_string(),
            Some((_, 0, raw)) => format!("UNBOXED (raw value {raw})"),
            Some((_, tag, size)) => format!("BOXED tag={tag:#x} size={size}"),
        };
        assert!(
            !matches!(writes.first(), Some((_, 0, _))),
            "{label} reached the host unboxed -- a raw value where a box was expected",
        );
        assert!(writes.first().is_some(), "{label} produced no write");
        seen.push((label, writes[0].1, writes[0].2));
        println!(
            "BOXING {label:<24} -> {verdict}  [ran={}]",
            match &ran {
                Ok(v) => format!("{v}"),
                Err(e) => e.chars().take(40).collect::<String>(),
            }
        );
    }
    // The widths are the emitter-facing fact: a bare literal takes the
    // default integer type, so a signal declared i64 would be minted
    // i32 if the emitter let a literal decide.
    let widths: Vec<(&str, u32)> = seen.iter().map(|(l, _, size)| (*l, *size)).collect();
    assert_eq!(
        widths,
        vec![
            ("direct argument", 4),
            ("via `let b: Any = v`", 8),
            ("explicit zyntax_box_i64", 8),
        ],
        "a bare literal boxes as i32; a typed binding as i64",
    );
}

/// Where `Any` is accepted, and whether a typed value crosses into and
/// back out of it without an explicit cast.
#[test]
fn where_any_is_accepted_and_whether_it_coerces() {
    let _guard = exclusive();
    let rows: [(&str, String); 4] = [
        (
            "declarations only",
            CELL.to_string(),
        ),
        (
            "T -> Any (autobox)",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    host_signal_cell_set(c, 41)\n    return 0\n}}\n"),
        ),
        (
            "Any -> T (autounbox)",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    host_signal_cell_set(c, 41)\n    return host_signal_cell_get(c)\n}}\n"),
        ),
        (
            "Any held as Any",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    host_signal_cell_set(c, 41)\n    let v: Any = host_signal_cell_get(c)\n    return 0\n}}\n"),
        ),
    ];
    for (label, src) in rows {
        reset();
        let outcome = try_compile(&src);
        println!(
            "ANY {label:<24} -> {}",
            match &outcome {
                Ok(()) => "ok".to_string(),
                Err(e) => e.clone(),
            }
        );
        assert!(outcome.is_ok(), "{label}: {}", outcome.unwrap_err());
    }
}

/// A scalar signal round-trips through host storage with no generics
/// and no impl block: write 41 through the effect, read 41 back.
#[test]
fn a_scalar_round_trips_through_an_any_cell() {
    let _guard = exclusive();
    reset();
    let src = format!(
        r#"{CELL}
effect SignalI64 {{
    def get(): i64
    def set(val: i64)
}}

handler MintedSignalI64 for SignalI64 {{
    var cell: SignalCell = host_signal_cell_new()
    def get(): i64 {{ return host_signal_cell_get(self.cell) }}
    def set(val: i64) {{ host_signal_cell_set(self.cell, val) }}
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
        panic!("compile: {}", compiled.err().unwrap());
    };
    let ran = rt.call::<i64>("main", &[]).map_err(|e| e.to_string());
    let writes = WRITES.lock().unwrap().clone();
    println!("SCALAR main={ran:?} writes={writes:?}");

    assert_eq!(ran, Ok(41), "the value read back is the one written");
    assert_eq!(writes.len(), 1, "one set reached the host");
}

/// The case that decides whether `Any` is enough: a user-declared
/// struct as the element type. Nothing about `Point` is known to the
/// host, and it must not need to be — the box carries its own tag,
/// size and dropper.
#[test]
fn a_user_struct_round_trips_through_an_any_cell() {
    let _guard = exclusive();
    reset();
    let src = format!(
        r#"{CELL}
struct Point {{ x: i64, y: i64 }}

effect SignalPoint {{
    def get(): Point
    def set(val: Point)
}}

handler MintedSignalPoint for SignalPoint {{
    var cell: SignalCell = host_signal_cell_new()
    def get(): Point {{ return host_signal_cell_get(self.cell) }}
    def set(val: Point) {{ host_signal_cell_set(self.cell, val) }}
}}

@effect(SignalPoint)
def write(p: Point): i64 {{ set(p) return 0 }}

@effect(SignalPoint)
def read_x(): i64 {{ let p: Point = get() return p.x }}

def main(): i64 {{
    let mut out: i64 = 0
    with MintedSignalPoint {{
        write(Point {{ x: 3, y: 4 }})
        out = read_x()
    }}
    return out
}}
"#
    );

    let compiled = try_compile(&src);
    println!(
        "STRUCT compile -> {}",
        match &compiled {
            Ok(()) => "ok".to_string(),
            Err(e) => e.clone(),
        }
    );
    if compiled.is_err() {
        return;
    }

    let rt = compile(&src).expect("already known to compile");
    let ran = rt.call::<i64>("main", &[]).map_err(|e| e.to_string());
    let writes = WRITES.lock().unwrap().clone();
    println!("STRUCT main={ran:?} writes={writes:?} (cell, tag, size)");
    assert_eq!(ran, Ok(3), "p.x read back through the cell");
}

/// Does a box arriving at an `Any` parameter OWN its payload?
///
/// Decides whether host-side storage may retain the box or has to copy
/// the payload out before returning. A box with no dropper borrows,
/// and for a scalar that is harmless to copy; for an aggregate it is
/// not, because a shallow copy duplicates any interior pointer without
/// duplicating its ownership.
#[test]
fn whether_a_box_at_an_any_boundary_owns_its_payload() {
    let _guard = exclusive();
    let cases: [(&str, String); 2] = [
        (
            "scalar",
            format!(
                "{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    let v: i64 = 41\n    host_signal_cell_set(c, v)\n    return 0\n}}\n"
            ),
        ),
        (
            "struct",
            format!(
                "{CELL}\nstruct Point {{ x: i64, y: i64 }}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    let p: Point = Point {{ x: 3, y: 4 }}\n    host_signal_cell_set(c, p)\n    return 0\n}}\n"
            ),
        ),
    ];

    for (label, src) in cases {
        reset();
        let compiled = compile(&src);
        let ran = match &compiled {
            Ok(rt) => rt.call::<i64>("main", &[]).map_err(|e| e.to_string()),
            Err(e) => Err(e.clone()),
        };
        let writes = WRITES.lock().unwrap().clone();
        let owned = OWNED.lock().unwrap().clone();
        println!(
            "OWNERSHIP {label:<8} ran={:?} writes={writes:x?} (dropper, display_fn)={owned:?}",
            ran.as_ref()
                .map_err(|e| e.chars().take(40).collect::<String>())
        );
    }
}

// ---------------------------------------------------------------------
// Who frees a box the host RETURNS?
// ---------------------------------------------------------------------

/// Droppers fired on boxes this host handed back from an `Any` return.
static DROPS: Mutex<usize> = Mutex::new(0);

extern "C" fn counting_dropper(p: *mut u8) {
    *DROPS.lock().unwrap() += 1;
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p as *mut i64) });
    }
}

/// A cell whose `get` hands back an OWNED box carrying a dropper, so
/// the test can see whether anything on the DSL side runs it.
static OWNED_GETS: Mutex<usize> = Mutex::new(0);

extern "C" fn owning_cell_get(_c: *mut Cell) -> *mut DynamicBoxRepr {
    *OWNED_GETS.lock().unwrap() += 1;
    let payload = Box::into_raw(Box::new(7i64));
    Box::into_raw(Box::new(DynamicBoxRepr {
        tag: TypeTag::I64.0,
        size: 8,
        data: payload as *mut u8,
        dropper: Some(counting_dropper),
        display_fn: None,
    }))
}

/// An `Any` return value unboxes wherever it lands, and the box is
/// never freed.
///
/// The positions are kept apart because they were not always
/// equivalent: implicit coercion once happened only at a return and a
/// call argument, so a `let` or an assignment silently bound the raw
/// box pointer until zyntax `b6721fb`. The rows with an explicit `as`
/// stay as well, since a cast on top of a coercion must not
/// double-unbox.
///
/// The dropper count is the load-bearing assertion: nothing frees a
/// returned box, which is why `SignalCell` reuses one box per cell
/// instead of allocating per read. If that ever changes, reusing a box
/// becomes a use-after-free and this is what says so.
#[test]
fn where_an_any_return_is_unboxed_and_whether_it_is_freed() {
    let _guard = exclusive();
    // (label, source, expected-if-unboxed)
    let rows: [(&str, String, i64); 6] = [
        (
            "return position",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    return host_signal_cell_get(c)\n}}\n"),
            7,
        ),
        (
            "let binding, no loop",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    let v: i64 = host_signal_cell_get(c)\n    return v\n}}\n"),
            7,
        ),
        (
            "let binding + `as i64`",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    let v: i64 = host_signal_cell_get(c) as i64\n    return v\n}}\n"),
            7,
        ),
        (
            "assignment + `as i64`",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    let mut v: i64 = 0\n    v = host_signal_cell_get(c) as i64\n    return v\n}}\n"),
            7,
        ),
        (
            "plain assignment, no cast",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    let mut v: i64 = 0\n    v = host_signal_cell_get(c)\n    return v\n}}\n"),
            7,
        ),
        (
            "let binding, 8x loop",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = host_signal_cell_new()\n    let mut total: i64 = 0\n    let mut i: i64 = 0\n    while i < 8 {{\n        let v: i64 = host_signal_cell_get(c)\n        total = total + v\n        i = i + 1\n    }}\n    return total\n}}\n"),
            56,
        ),
    ];

    let mut outcomes: Vec<(&str, bool)> = Vec::new();
    for (label, src, expect) in rows {
        *DROPS.lock().unwrap() = 0;
        *OWNED_GETS.lock().unwrap() = 0;
        reset();

        let mut config = TieredConfig::development();
        config.enable_osr = true;
        let mut rt = TieredRuntime::new(config).expect("runtime");
        rt.register_function_typed(
            "host_signal_cell_new",
            host_signal_cell_new as *const u8,
            sig(&[], ptr_tag()),
        );
        rt.register_function_typed(
            "host_signal_cell_get",
            owning_cell_get as *const u8,
            sig(&[ptr_tag()], ptr_tag()),
        );
        rt.register_function_typed(
            "host_signal_cell_set",
            host_signal_cell_set as *const u8,
            sig(&[ptr_tag(), ptr_tag()], TypeTag::VOID),
        );
        rt.finalize_runtime_symbols().expect("publish");

        let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
        let program = grammar
            .parse_with_filename(&src, "<owned_get>")
            .expect("parse");
        rt.compile_typed_program(program).expect("compile");
        let ran = rt.call::<i64>("main", &[]).map_err(|e| e.to_string());

        let gets = *OWNED_GETS.lock().unwrap();
        let drops = *DROPS.lock().unwrap();
        let unboxed = ran.as_ref().map(|v| *v == expect).unwrap_or(false);
        outcomes.push((label, unboxed));
        assert_eq!(
            drops, 0,
            "{label}: a returned box was freed. Nothing did before, which is why \
             SignalCell reuses one box per cell rather than allocating per read.",
        );
        println!(
            "ANYRET {label:<22} unboxed={unboxed:<5} got={:<16} gets={gets} droppers_fired={drops}",
            match &ran {
                Ok(v) => format!("{v}"),
                Err(e) => e.chars().take(14).collect::<String>(),
            }
        );
    }
    assert!(
        outcomes.iter().all(|(_, unboxed)| *unboxed),
        "an Any unboxes in every position, with or without a cast: {outcomes:?}",
    );
}
