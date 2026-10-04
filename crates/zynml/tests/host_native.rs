//! Host members bound to native functions: each call a direct call with
//! typed operands, each bound field a load or a store in the object.

use std::cell::Cell;

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::zrtl::{TypeCategory, TypeFlags, TypeTag, ZrtlSigFlags, ZrtlSymbolSig};
use zyntax_embed::foreign::{self, ForeignError};
use zyntax_embed::host::{
    HostClass, HostField, HostMethod, HostModule, HostType, NativeBinding, NativeField, NativePass,
    NativeType,
};
use zyntax_embed::{TieredConfig, TieredRuntime};

/// What a plugin's struct is.
#[repr(C)]
struct Vec2 {
    x: f64,
    y: f64,
}

/// A host object: a descriptor word, then the plugin's payload.
#[repr(C)]
struct Core {
    desc: usize,
    payload: *mut Vec2,
}

extern "C" fn vec2_new(x: f64, y: f64) -> i64 {
    let payload = Box::into_raw(Box::new(Vec2 { x, y }));
    Box::into_raw(Box::new(Core { desc: 0, payload })) as i64
}

extern "C" fn vec2_len(v: *const Vec2) -> f64 {
    let v = unsafe { &*v };
    (v.x * v.x + v.y * v.y).sqrt()
}

extern "C" fn vec2_scale(v: *mut Vec2, by: f64) {
    let v = unsafe { &mut *v };
    v.x *= by;
    v.y *= by;
}

extern "C" fn vec2_dot(a: *const Vec2, b: *const Vec2) -> f64 {
    let (a, b) = unsafe { (&*a, &*b) };
    a.x * b.x + a.y * b.y
}

extern "C" fn twice(n: i32) -> i32 {
    n * 2
}

extern "C" fn count_bytes(data: *const u8, len: i64) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) };
    bytes.iter().filter(|b| **b == b'a').count() as i64
}

/// A host string: what the host's own strings are, here a leaked Rust
/// string, which `$Host$text_to_string` converts.
extern "C" fn shout(data: *const u8, len: i64) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) };
    let text = String::from_utf8_lossy(bytes);
    let text = format!("{text}{text}{text}");
    Box::into_raw(Box::new(text)) as i64
}

extern "C" fn text_to_string(host: i64) -> *mut u8 {
    let text = unsafe { Box::from_raw(host as *mut String) };
    // A runtime string, out of the box the library makes for one.
    unsafe { (*foreign::string(&text)).data }
}

thread_local! {
    static PENDING: Cell<u8> = const { Cell::new(0) };
}

extern "C" fn pending_flag() -> i64 {
    PENDING.with(|p| p.as_ptr() as i64)
}

extern "C" fn raise_pending() {
    PENDING.with(|p| p.set(0));
    foreign::report(ForeignError::new("ValueError", "a negative length"));
}

extern "C" fn checked(n: i64) -> i64 {
    if n < 0 {
        PENDING.with(|p| p.set(1));
        return 0;
    }
    n + 1
}

fn tag(category: TypeCategory, size: u16) -> TypeTag {
    TypeTag::new(category, size, TypeFlags::NONE)
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

fn bound(
    name: &str,
    params: Vec<HostType>,
    ret: HostType,
    is_static: bool,
    native: NativeBinding,
) -> HostMethod {
    HostMethod {
        name: name.into(),
        params,
        ret,
        is_static,
        native: Some(native),
        ..Default::default()
    }
}

fn binding(
    symbol: &str,
    receiver: Option<NativePass>,
    params: Vec<NativeType>,
    ret: NativeType,
) -> NativeBinding {
    NativeBinding {
        symbol: symbol.into(),
        address: 0,
        receiver,
        params,
        ret,
        may_raise: false,
    }
}

fn geometry() -> HostModule {
    let vec2 = HostType::Object("geo.Vec2".into());
    let object = NativeType::Object {
        type_name: "geo.Vec2".into(),
        pass: NativePass::Indirect(8),
    };
    let payload = Some(NativePass::Indirect(8));
    let field = |name: &str, offset: u32| HostField {
        name: name.into(),
        ty: HostType::Float,
        writable: true,
        native: Some(NativeField {
            offset,
            ty: NativeType::F64,
            pass: NativePass::Indirect(8),
        }),
        ..Default::default()
    };
    let mut checked_binding = binding("plug_checked", None, vec![NativeType::I64], NativeType::I64);
    checked_binding.may_raise = true;
    HostModule {
        name: "geo".into(),
        classes: vec![HostClass {
            name: "Vec2".into(),
            type_name: "geo.Vec2".into(),
            word: true,
            fields: vec![field("x", 0), field("y", 8)],
            methods: vec![
                bound(
                    "len",
                    vec![],
                    HostType::Float,
                    false,
                    binding("plug_vec2_len", payload, vec![], NativeType::F64),
                ),
                bound(
                    "scale",
                    vec![HostType::Float],
                    HostType::Void,
                    false,
                    binding(
                        "plug_vec2_scale",
                        payload,
                        vec![NativeType::F64],
                        NativeType::Void,
                    ),
                ),
                bound(
                    "dot",
                    vec![vec2.clone()],
                    HostType::Float,
                    false,
                    binding("plug_vec2_dot", payload, vec![object], NativeType::F64),
                ),
            ],
            constructor: Some(bound(
                "new",
                vec![HostType::Float, HostType::Float],
                vec2,
                true,
                binding(
                    "plug_vec2_new",
                    None,
                    vec![NativeType::F64, NativeType::F64],
                    NativeType::Object {
                        type_name: "geo.Vec2".into(),
                        pass: NativePass::Word,
                    },
                ),
            )),
        }],
        functions: vec![
            // Loaded late: no registration, only an address.
            bound(
                "twice",
                vec![HostType::Int],
                HostType::Int,
                true,
                NativeBinding {
                    address: twice as *const () as usize,
                    ..binding(
                        "plug_late_twice",
                        None,
                        vec![NativeType::I32],
                        NativeType::I32,
                    )
                },
            ),
            bound(
                "count_a",
                vec![HostType::Str],
                HostType::Int,
                true,
                binding(
                    "plug_count_bytes",
                    None,
                    vec![NativeType::Str],
                    NativeType::I64,
                ),
            ),
            bound(
                "shout",
                vec![HostType::Str],
                HostType::Str,
                true,
                binding("plug_shout", None, vec![NativeType::Str], NativeType::Str),
            ),
            bound(
                "checked",
                vec![HostType::Int],
                HostType::Int,
                true,
                checked_binding,
            ),
        ],
    }
}

const SRC: &str = r#"
from geo import Vec2

def lengths(): f64 {
    let v = Vec2::new(3.0, 4.0)
    return v.len()
}

def fields(): f64 {
    let v = Vec2::new(3.0, 4.0)
    v.set_x(6.0)
    v.scale(0.5)
    return v.x() * 10.0 + v.y()
}

def dots(): f64 {
    let a = Vec2::new(1.0, 2.0)
    let b = Vec2::new(3.0, 4.0)
    return a.dot(b)
}

def calls(): i64 {
    return twice(21) * 100 + count_a("banana")
}

def strings(): i64 {
    return count_a(shout("abc"))
}

def raising(n: i64): i64 {
    return checked(n)
}
"#;

fn runtime() -> TieredRuntime {
    let mut rt = TieredRuntime::new(TieredConfig::development()).expect("runtime should start");
    let f64t = TypeTag::F64;
    let ptr = tag(TypeCategory::Pointer, 0);
    let i64t = TypeTag::I64;
    rt.register_function_typed(
        "plug_vec2_new",
        vec2_new as *const u8,
        sig(&[f64t, f64t], i64t),
    );
    rt.register_function_typed("plug_vec2_len", vec2_len as *const u8, sig(&[i64t], f64t));
    rt.register_function_typed(
        "plug_vec2_scale",
        vec2_scale as *const u8,
        sig(&[i64t, f64t], TypeTag::VOID),
    );
    rt.register_function_typed(
        "plug_vec2_dot",
        vec2_dot as *const u8,
        sig(&[i64t, i64t], f64t),
    );
    rt.register_function_typed(
        "plug_count_bytes",
        count_bytes as *const u8,
        sig(&[i64t, i64t], i64t),
    );
    rt.register_function_typed("plug_shout", shout as *const u8, sig(&[i64t, i64t], i64t));
    rt.register_function_typed("plug_checked", checked as *const u8, sig(&[i64t], i64t));
    rt.register_function_typed(
        "$Host$pending_flag",
        pending_flag as *const u8,
        sig(&[], i64t),
    );
    rt.register_function_typed(
        "$Host$raise_pending",
        raise_pending as *const u8,
        sig(&[], TypeTag::VOID),
    );
    rt.register_function_typed(
        "$Host$text_to_string",
        text_to_string as *const u8,
        sig(&[i64t], ptr),
    );
    rt.register_static_plugins([zrtl_io::static_plugin()])
        .expect("plugins");
    rt.finalize_runtime_symbols().expect("symbols");
    rt.add_host_modules(|path| (path == "geo").then(geometry))
        .expect("host modules");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(SRC, "<host_native>")
        .expect("parse");
    rt.compile_typed_program(program).expect("compile");
    rt
}

#[test]
fn native_members_and_fields_run_directly() {
    let rt = runtime();
    let f64_of = |f: &str| rt.call::<f64>(f, &[]).map_err(|e| e.to_string());
    let i64_of = |f: &str| rt.call::<i64>(f, &[]).map_err(|e| e.to_string());
    assert_eq!(f64_of("lengths"), Ok(5.0));
    assert_eq!(f64_of("fields"), Ok(32.0));
    assert_eq!(f64_of("dots"), Ok(11.0));
    assert_eq!(i64_of("calls"), Ok(4203));
    assert_eq!(i64_of("strings"), Ok(3));
    let raising = |n: i64| {
        rt.call::<i64>("raising", &[zyntax_embed::ZyntaxValue::Int(n)])
            .map_err(|e| e.to_string())
    };
    assert_eq!(raising(4), Ok(5));
    assert!(foreign::take_error().is_none());
    assert_eq!(raising(-1), Ok(0));
    let error = foreign::take_error().expect("the native error is left pending");
    assert_eq!(
        (error.kind, error.message.as_str()),
        ("ValueError", "a negative length")
    );
}
