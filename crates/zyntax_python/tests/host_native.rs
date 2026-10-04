//! Python calling host members bound to native functions: each call a
//! direct call with typed operands, each bound field read and written in
//! the object.

use std::cell::Cell;
use std::sync::Mutex;

use zyntax_compiler::zrtl::{TypeTag, ZrtlSigFlags, ZrtlSymbolSig};
use zyntax_embed::foreign::{self, Any, Foreign, ForeignError};
use zyntax_embed::{TieredConfig, TieredRuntime};
use zyntax_python::{
    HostClass, HostField, HostMethod, HostModule, HostType, NativeBinding, NativeField, NativePass,
    NativeType,
};

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

static RECORDED: Mutex<Vec<f64>> = Mutex::new(Vec::new());

extern "C" fn record(v: f64) {
    RECORDED.lock().unwrap().push(v);
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

extern "C" fn thrice(data: *const u8, len: i64) -> i64 {
    let text = String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(data, len as usize) });
    Box::into_raw(Box::new(format!("{text}{text}{text}"))) as i64
}

extern "C" fn text_to_string(host: i64) -> *mut u8 {
    let text = unsafe { Box::from_raw(host as *mut String) };
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

/// The protocol, for what is not bound: the module and its class.
struct Stand;

impl Foreign for Stand {
    fn import(&self, name: &str) -> Result<Option<Any>, ForeignError> {
        Ok((name == "geo").then(|| foreign::boxed(1)))
    }
    fn get(&self, _object: usize, name: &str) -> Result<Any, ForeignError> {
        Ok(foreign::boxed(match name {
            "Vec2" => 2,
            _ => 3,
        }))
    }
    fn set(&self, _: usize, _: &str, _: Any) -> Result<(), ForeignError> {
        Err(ForeignError::new(
            "TypeError",
            "not set through the protocol",
        ))
    }
    fn call(&self, _: usize, _: &[Any]) -> Result<Any, ForeignError> {
        Err(ForeignError::new(
            "TypeError",
            "not called through the protocol",
        ))
    }
    fn invoke(&self, _: usize, name: &str, _: &[Any]) -> Result<Any, ForeignError> {
        Err(ForeignError::new(
            "TypeError",
            format!("{name} through the protocol"),
        ))
    }
    fn retain(&self, object: usize) -> Result<usize, ForeignError> {
        Ok(object)
    }
    fn text(&self, _: usize) -> String {
        "<host>".into()
    }
    fn type_name(&self, _: usize) -> String {
        "host".into()
    }
    fn equals(&self, a: usize, b: usize) -> bool {
        a == b
    }
    fn hash(&self, object: usize) -> i64 {
        object as i64
    }
    fn release(&self, _: usize) {}
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

/// A member and the native function it is bound to.
struct Member {
    name: &'static str,
    params: Vec<HostType>,
    ret: HostType,
    is_static: bool,
}

fn bound(
    member: Member,
    symbol: &str,
    receiver: Option<NativePass>,
    native: Vec<NativeType>,
    native_ret: NativeType,
) -> HostMethod {
    HostMethod {
        name: member.name.into(),
        params: member.params,
        ret: member.ret,
        is_static: member.is_static,
        native: Some(NativeBinding {
            symbol: symbol.into(),
            address: 0,
            receiver,
            params: native,
            ret: native_ret,
            may_raise: false,
        }),
        ..Default::default()
    }
}

fn member(name: &'static str, params: Vec<HostType>, ret: HostType, is_static: bool) -> Member {
    Member {
        name,
        params,
        ret,
        is_static,
    }
}

fn geometry() -> HostModule {
    let vec2 = HostType::Object("geo.Vec2".into());
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
    let mut checked = bound(
        member("checked", vec![HostType::Int], HostType::Int, true),
        "py_checked",
        None,
        vec![NativeType::I64],
        NativeType::I64,
    );
    checked.native.as_mut().unwrap().may_raise = true;
    let mut late_twice = bound(
        member("twice", vec![HostType::Int], HostType::Int, true),
        "py_late_twice",
        None,
        vec![NativeType::I32],
        NativeType::I32,
    );
    late_twice.native.as_mut().unwrap().address = twice as *const () as usize;
    HostModule {
        name: "geo".into(),
        classes: vec![HostClass {
            name: "Vec2".into(),
            type_name: "geo.Vec2".into(),
            word: true,
            fields: vec![field("x", 0), field("y", 8)],
            methods: vec![
                bound(
                    member("len", vec![], HostType::Float, false),
                    "py_vec2_len",
                    payload,
                    vec![],
                    NativeType::F64,
                ),
                bound(
                    member("scale", vec![HostType::Float], HostType::Void, false),
                    "py_vec2_scale",
                    payload,
                    vec![NativeType::F64],
                    NativeType::Void,
                ),
                bound(
                    member("dot", vec![vec2.clone()], HostType::Float, false),
                    "py_vec2_dot",
                    payload,
                    vec![NativeType::Object {
                        type_name: "geo.Vec2".into(),
                        pass: NativePass::Indirect(8),
                    }],
                    NativeType::F64,
                ),
            ],
            constructor: Some(bound(
                member("Vec2", vec![HostType::Float, HostType::Float], vec2, true),
                "py_vec2_new",
                None,
                vec![NativeType::F64, NativeType::F64],
                NativeType::Object {
                    type_name: "geo.Vec2".into(),
                    pass: NativePass::Word,
                },
            )),
        }],
        functions: vec![
            late_twice,
            bound(
                member("count_a", vec![HostType::Str], HostType::Int, true),
                "py_count_bytes",
                None,
                vec![NativeType::Str],
                NativeType::I64,
            ),
            bound(
                member("thrice", vec![HostType::Str], HostType::Str, true),
                "py_thrice",
                None,
                vec![NativeType::Str],
                NativeType::Str,
            ),
            bound(
                member("record", vec![HostType::Float], HostType::Void, true),
                "py_record",
                None,
                vec![NativeType::F64],
                NativeType::Void,
            ),
            checked,
        ],
    }
}

const PROGRAM: &str = r#"
from geo import Vec2, twice, count_a, thrice, record, checked

v = Vec2(3.0, 4.0)
record(v.len())
v.x = 6.0
v.scale(0.5)
record(v.x * 10.0 + v.y)
a = Vec2(1.0, 2.0)
b = Vec2(3.0, 4.0)
record(a.dot(b))
record(float(twice(21) * 100 + count_a("banana")))
record(float(count_a(thrice("abc"))))
record(float(checked(4)))
try:
    checked(-1)
    record(0.0)
except ValueError as e:
    record(-1.0)
"#;

#[test]
fn python_calls_native_members_directly() {
    foreign::install(Box::new(Stand));
    let hosts = |name: &str| (name == "geo").then(geometry);
    let program = zyntax_python::parse_program_with_host(PROGRAM, "native.py", &|_| None, &hosts)
        .expect("lowers");
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    zyntax_python::register_runtime(&mut rt).expect("registered");
    let f64t = TypeTag::F64;
    let i64t = TypeTag::I64;
    rt.register_function_typed(
        "py_record",
        record as *const u8,
        sig(&[f64t], TypeTag::VOID),
    );
    rt.register_function_typed(
        "py_vec2_new",
        vec2_new as *const u8,
        sig(&[f64t, f64t], i64t),
    );
    rt.register_function_typed("py_vec2_len", vec2_len as *const u8, sig(&[i64t], f64t));
    rt.register_function_typed(
        "py_vec2_scale",
        vec2_scale as *const u8,
        sig(&[i64t, f64t], TypeTag::VOID),
    );
    rt.register_function_typed(
        "py_vec2_dot",
        vec2_dot as *const u8,
        sig(&[i64t, i64t], f64t),
    );
    rt.register_function_typed(
        "py_count_bytes",
        count_bytes as *const u8,
        sig(&[i64t, i64t], i64t),
    );
    rt.register_function_typed("py_thrice", thrice as *const u8, sig(&[i64t, i64t], i64t));
    rt.register_function_typed("py_checked", checked as *const u8, sig(&[i64t], i64t));
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
        sig(&[i64t], i64t),
    );
    rt.finalize_runtime_symbols().expect("symbols");
    rt.enter_only_through_entry_points();
    rt.compile_typed_program(program).expect("compiles");
    rt.call_raw(zyntax_python::ENTRY, &[]).expect("runs");
    assert_eq!(
        *RECORDED.lock().unwrap(),
        [5.0, 32.0, 11.0, 4203.0, 3.0, 5.0, -1.0]
    );
    std::mem::forget(rt);
}
