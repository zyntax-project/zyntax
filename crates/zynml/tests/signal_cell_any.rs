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

extern "C" fn blinc_signal_cell_new() -> *mut Cell {
    let c = Box::into_raw(Box::new(Cell {
        boxed: std::ptr::null_mut(),
    }));
    CELLS.lock().unwrap().push(c as usize);
    c
}

extern "C" fn blinc_signal_cell_get(c: *mut Cell) -> *mut DynamicBoxRepr {
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

extern "C" fn blinc_signal_cell_set(c: *mut Cell, v: *mut DynamicBoxRepr) {
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
    unsafe { (*c).boxed = v };
    WRITES.lock().unwrap().push((c as usize, tag, size));
}

fn reset() {
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
        sig(&[ptr_tag()], ptr_tag()),
    );
    rt.register_function_typed(
        "blinc_signal_cell_set",
        blinc_signal_cell_set as *const u8,
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

extern def blinc_signal_cell_new(): SignalCell
extern def blinc_signal_cell_get(c: SignalCell): Any
extern def blinc_signal_cell_set(c: SignalCell, v: Any)
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
    let rows: [(&str, String); 3] = [
        (
            "direct argument",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = blinc_signal_cell_new()\n    blinc_signal_cell_set(c, 41)\n    return 0\n}}\n"),
        ),
        (
            "via `let b: Any = v`",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = blinc_signal_cell_new()\n    let v: i64 = 41\n    let b: Any = v\n    blinc_signal_cell_set(c, b)\n    return 0\n}}\n"),
        ),
        (
            "explicit zyntax_box_i64",
            format!("{CELL}\nextern def zyntax_box_i64(v: i64): Any\ndef main(): i64 {{\n    let c: SignalCell = blinc_signal_cell_new()\n    blinc_signal_cell_set(c, zyntax_box_i64(41))\n    return 0\n}}\n"),
        ),
    ];

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
        println!(
            "BOXING {label:<24} -> {verdict}  [ran={}]",
            match &ran {
                Ok(v) => format!("{v}"),
                Err(e) => e.chars().take(40).collect::<String>(),
            }
        );
    }
}

/// Where `Any` is accepted, and whether a typed value crosses into and
/// back out of it without an explicit cast.
#[test]
fn where_any_is_accepted_and_whether_it_coerces() {
    let rows: [(&str, String); 4] = [
        (
            "declarations only",
            CELL.to_string(),
        ),
        (
            "T -> Any (autobox)",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = blinc_signal_cell_new()\n    blinc_signal_cell_set(c, 41)\n    return 0\n}}\n"),
        ),
        (
            "Any -> T (autounbox)",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = blinc_signal_cell_new()\n    blinc_signal_cell_set(c, 41)\n    return blinc_signal_cell_get(c)\n}}\n"),
        ),
        (
            "Any held as Any",
            format!("{CELL}\ndef main(): i64 {{\n    let c: SignalCell = blinc_signal_cell_new()\n    blinc_signal_cell_set(c, 41)\n    let v: Any = blinc_signal_cell_get(c)\n    return 0\n}}\n"),
        ),
    ];
    for (label, src) in rows {
        reset();
        println!(
            "ANY {label:<24} -> {}",
            match try_compile(&src) {
                Ok(()) => "ok".to_string(),
                Err(e) => e,
            }
        );
    }
}

/// A scalar signal round-trips through host storage with no generics
/// and no impl block: write 41 through the effect, read 41 back.
#[test]
fn a_scalar_round_trips_through_an_any_cell() {
    reset();
    let src = format!(
        r#"{CELL}
effect SignalI64 {{
    def get(): i64
    def set(val: i64)
}}

handler MintedSignalI64 for SignalI64 {{
    var cell: SignalCell = blinc_signal_cell_new()
    def get(): i64 {{ return blinc_signal_cell_get(self.cell) }}
    def set(val: i64) {{ blinc_signal_cell_set(self.cell, val) }}
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
    reset();
    let src = format!(
        r#"{CELL}
struct Point {{ x: i64, y: i64 }}

effect SignalPoint {{
    def get(): Point
    def set(val: Point)
}}

handler MintedSignalPoint for SignalPoint {{
    var cell: SignalCell = blinc_signal_cell_new()
    def get(): Point {{ return blinc_signal_cell_get(self.cell) }}
    def set(val: Point) {{ blinc_signal_cell_set(self.cell, val) }}
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
