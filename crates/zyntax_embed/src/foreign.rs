//! Foreign objects: values of the program that embeds the runtime, held
//! by a compiled program as dynamic values.
//!
//! A foreign object is a box tagged [`FOREIGN_TAG`] whose `data` is a
//! word the embedder chose ([`boxed`]); releasing the box hands the word
//! back through [`Foreign::release`]. The word is the payload itself and
//! the box's size is zero, so no reader of a payload reads through it.
//!
//! The built-in library reaches a foreign object only through the
//! `$Foreign$*` symbols of [`static_plugin`], each of which asks the
//! [`Foreign`] installed with [`install`]: one per process, set once, as
//! the fiber backend is. Until one is installed no module is foreign, so
//! no foreign object exists to operate on.
//!
//! Values cross as the library's dynamic values ([`Any`]); None is the
//! null box. A box handed to the embedder is borrowed for the call; a
//! box it returns is the program's.

use std::cell::RefCell;
use std::sync::OnceLock;

use zrtl::{DynamicBox, StringConstPtr, StringPtr, TypeCategory, TypeTag};

/// The box tag of a foreign object: the custom category, with the kind
/// `zyntax_builtins::FOREIGN_TAG` gives it.
pub const FOREIGN_TAG: u32 = (20 << 8) | 0xFF;

/// The box tag of a tuple, as `zyntax_builtins::TUPLE_TAG` gives it:
/// several values as one, a list of them.
pub const TUPLE_TAG: u32 = (14 << 8) | 0xFF;

/// The header a list points at, as the library lays one out.
#[repr(C)]
struct ListHeader {
    data: *mut Any,
    len: i64,
    capacity: i64,
}

/// A dynamic value as the library holds it.
pub type Any = *mut DynamicBox;

/// Why an operation on a foreign object failed: a kind the library
/// raises (`"TypeError"`, `"AttributeError"`, `"IndexError"`,
/// `"RuntimeError"`) and the message.
#[derive(Clone, Debug)]
pub struct ForeignError {
    pub kind: &'static str,
    pub message: String,
}

impl ForeignError {
    pub fn new(kind: &'static str, message: impl Into<String>) -> ForeignError {
        ForeignError {
            kind,
            message: message.into(),
        }
    }
}

/// What the embedder does for a program's foreign objects. `object` is
/// the word a box carries.
pub trait Foreign: Send + Sync {
    /// The member `name` of `object`: a field's value, or a method as a
    /// value that takes its receiver first.
    fn get(&self, object: usize, name: &str) -> Result<Any, ForeignError>;

    /// Write the member `name` of `object`.
    fn set(&self, object: usize, name: &str, value: Any) -> Result<(), ForeignError>;

    /// Call `object` with `args`.
    fn call(&self, object: usize, args: &[Any]) -> Result<Any, ForeignError>;

    /// Construct through a host class. A successful result is a non-null box
    /// owned by the program and released when its value dies.
    fn construct(&self, object: usize, args: &[Any]) -> Result<Any, ForeignError> {
        self.call(object, args)
    }

    /// Construct and return the owned opaque word without a dynamic box.
    fn construct_word(&self, object: usize, args: &[Any]) -> Result<usize, ForeignError> {
        let boxed = self.construct(object, args)?;
        unsafe { take_word(boxed) }
            .ok_or_else(|| ForeignError::new("TypeError", "a host constructor returned no object"))
    }

    /// Add one ownership claim to an opaque host-object word.
    fn retain(&self, _object: usize) -> Result<usize, ForeignError> {
        Err(ForeignError::new(
            "TypeError",
            "host object cannot be retained",
        ))
    }

    /// Call the method `name` of `object` with `args`.
    fn invoke(&self, object: usize, name: &str, args: &[Any]) -> Result<Any, ForeignError>;

    /// Read a field whose schema declares a floating-point result without
    /// allocating a dynamic box for that result.
    fn get_float(&self, _object: usize, _name: &str) -> Result<f64, ForeignError> {
        number_of(self.get(_object, _name)?)
    }

    /// Write a schema-declared floating-point field without boxing it.
    fn set_float(&self, _object: usize, _name: &str, _value: f64) -> Result<(), ForeignError> {
        self.set(_object, _name, float(_value))
    }

    /// Call a schema-declared function of float arguments and result.
    fn call_float(&self, _object: usize, _args: &[f64]) -> Result<f64, ForeignError> {
        let args: Vec<Any> = _args.iter().map(|&value| float(value)).collect();
        number_of(self.call(_object, &args)?)
    }

    /// Construct with unboxed float arguments. A successful result is a
    /// non-null box owned by the program and released when its value dies.
    fn construct_float(&self, _object: usize, _args: &[f64]) -> Result<Any, ForeignError> {
        let args: Vec<Any> = _args.iter().map(|&value| float(value)).collect();
        self.construct(_object, &args)
    }

    /// Construct from unboxed floats and return the owned opaque word.
    fn construct_float_word(&self, object: usize, args: &[f64]) -> Result<usize, ForeignError> {
        let boxed = self.construct_float(object, args)?;
        unsafe { take_word(boxed) }
            .ok_or_else(|| ForeignError::new("TypeError", "a host constructor returned no object"))
    }

    /// Invoke a schema-declared method of float arguments and result.
    fn invoke_float(
        &self,
        _object: usize,
        _name: &str,
        _args: &[f64],
    ) -> Result<f64, ForeignError> {
        let args: Vec<Any> = _args.iter().map(|&value| float(value)).collect();
        number_of(self.invoke(_object, _name, &args)?)
    }

    /// Read a float field by an embedder-defined schema key.
    fn get_float_key(&self, _object: usize, _key: u64) -> Result<f64, ForeignError> {
        Err(ForeignError::new(
            "TypeError",
            "the embedder does not recognize keyed float fields",
        ))
    }

    /// Write a float field by an embedder-defined schema key.
    fn set_float_key(&self, _object: usize, _key: u64, _value: f64) -> Result<(), ForeignError> {
        Err(ForeignError::new(
            "TypeError",
            "the embedder does not recognize keyed float fields",
        ))
    }

    /// Invoke a float method by an embedder-defined schema key.
    fn invoke_float_key(
        &self,
        _object: usize,
        _key: u64,
        _args: &[f64],
    ) -> Result<f64, ForeignError> {
        Err(ForeignError::new(
            "TypeError",
            "the embedder does not recognize keyed float methods",
        ))
    }

    /// `object` as text.
    fn text(&self, object: usize) -> String;

    /// The name of `object`'s type.
    fn type_name(&self, object: usize) -> String;

    /// Whether `a` and `b` are the same value.
    fn equals(&self, a: usize, b: usize) -> bool;

    /// A hash that agrees with [`Foreign::equals`].
    fn hash(&self, object: usize) -> i64;

    /// The module `name` names, as a foreign object, or `None` when the
    /// embedder has none by that name.
    fn import(&self, name: &str) -> Result<Option<Any>, ForeignError>;

    /// The program released its box of `object`.
    fn release(&self, object: usize);

    /// The bytes `object` is a buffer of, which the program reads in
    /// place: their address and length, valid while the program holds
    /// `object`. `None` for an object that is not one.
    fn bytes(&self, _object: usize) -> Option<(*const u8, usize)> {
        None
    }

    /// How many values `object` is, when it is several values a call
    /// gave at once: a call or method that returns it gives the program
    /// that many, as its tuple of them. `None` for any other object.
    fn values(&self, _object: usize) -> Option<usize> {
        None
    }

    /// Value `index` of an `object` that [`Foreign::values`] counts, as
    /// a box the program owns.
    fn value(&self, _object: usize, _index: usize) -> Any {
        none()
    }
}

fn number_of(any: Any) -> Result<f64, ForeignError> {
    match unsafe { read(any) } {
        Value::Float(value) => Ok(value),
        Value::Int(value) => Ok(value as f64),
        _ => Err(ForeignError::new(
            "TypeError",
            "a numeric host result was expected",
        )),
    }
}

static INSTALLED: OnceLock<Box<dyn Foreign>> = OnceLock::new();

/// Install the embedder's [`Foreign`]. The first call wins; a later one
/// returns `false` and changes nothing.
pub fn install(foreign: Box<dyn Foreign>) -> bool {
    INSTALLED.set(foreign).is_ok()
}

fn installed() -> Option<&'static dyn Foreign> {
    INSTALLED.get().map(|f| f.as_ref())
}

/// A box of `word`, which must not be zero. The program releases it,
/// and [`Foreign::release`] then hears of `word`.
pub fn boxed(word: usize) -> Any {
    assert_ne!(word, 0, "a foreign object's word is not zero");
    DynamicBox {
        tag: TypeTag(FOREIGN_TAG),
        size: 0,
        data: word as *mut u8,
        dropper: Some(drop_foreign),
        display_fn: None,
    }
    .into_raw()
}

extern "C" fn drop_foreign(word: *mut u8) {
    if let Some(foreign) = installed() {
        foreign.release(word as usize);
    }
}

/// The word of a foreign object's box, or `None` for any other value.
///
/// # Safety
/// `any` must be null or a live box.
pub unsafe fn word(any: Any) -> Option<usize> {
    (!any.is_null() && (*any).tag.0 == FOREIGN_TAG).then(|| (*any).data as usize)
}

/// The box tag of a function value, as `zyntax_builtins::FUNC_TAG` gives
/// it: a record of dynamic values, the code's address first, its arity
/// word second and what it captured after.
pub const FUNC_TAG: u32 = (17 << 8) | 0xFF;

/// The most arguments a call through a function value passes, as
/// `zyntax_builtins::MAX_CALL_ARITY` says.
pub const MAX_CALL_ARITY: usize = 8;

/// The arity word of a function whose code takes its arguments packed in
/// one list.
const VARIADIC_ARITY: i64 = -1;

/// Whether `any` is a function value.
///
/// # Safety
/// `any` must be null or a live box.
pub unsafe fn is_function(any: Any) -> bool {
    !any.is_null() && (*any).tag.0 == FUNC_TAG
}

/// A function value's record: its items, the code first.
///
/// # Safety
/// `f` must be a live function value.
unsafe fn record<'a>(f: Any) -> (*const ListHeader, &'a [Any]) {
    let list = (*f).data as *const ListHeader;
    let items = match list.as_ref() {
        Some(h) if h.len > 0 && !h.data.is_null() => {
            std::slice::from_raw_parts(h.data, h.len as usize)
        }
        _ => &[],
    };
    (list, items)
}

/// How many arguments the function value `f` takes, `(least, most)`, as
/// its arity word says; `None` for any other value and for a function
/// whose code takes its arguments packed in a list.
///
/// # Safety
/// `f` must be null or a live box.
pub unsafe fn function_arity(f: Any) -> Option<(usize, usize)> {
    if !is_function(f) {
        return None;
    }
    let (_, items) = record(f);
    let Value::Int(word) = read(*items.get(1)?) else {
        return None;
    };
    if word == VARIADIC_ARITY {
        return None;
    }
    let most = (word & 0xFFFF) as usize;
    let least = match word >> 16 {
        0 => most,
        n => n as usize - 1,
    };
    Some((least, most))
}

/// Call the function value `f` with `args`, as the library's own call
/// through a value does: the record and each argument as dynamic values,
/// to a dynamic result. An error the call raises is left pending in the
/// runtime's error flag, as any call's is.
///
/// A call passes every argument the code takes: leaving out one with a
/// default needs the frontend's marker for it, and a code taking its
/// arguments packed in a list needs the library to build the list.
///
/// # Safety
/// `f` must be null or a live box, each of `args` a live box or null, and
/// the program that made `f` must still be loaded.
pub unsafe fn call_function(f: Any, args: &[Any]) -> Result<Any, ForeignError> {
    if !is_function(f) {
        return Err(ForeignError::new(
            "TypeError",
            "the value is not a function",
        ));
    }
    let Some((least, most)) = function_arity(f) else {
        return Err(ForeignError::new(
            "TypeError",
            "a function taking its arguments packed is not called from the host",
        ));
    };
    if args.len() != most {
        let takes = if least == most {
            format!("{most}")
        } else {
            format!("{most} (the host passes every one; {least} without defaults)")
        };
        return Err(ForeignError::new(
            "TypeError",
            format!(
                "function takes {takes} arguments but {} were given",
                args.len()
            ),
        ));
    }
    let (list, items) = record(f);
    let code = match items.first() {
        Some(&b) if !b.is_null() && !(*b).data.is_null() => (*b).data as *const u8,
        _ => {
            return Err(ForeignError::new(
                "TypeError",
                "the function value has no code",
            ));
        }
    };
    type R = *const ListHeader;
    let a = |i: usize| args[i];
    Ok(match most {
        0 => std::mem::transmute::<_, extern "C" fn(R) -> Any>(code)(list),
        1 => std::mem::transmute::<_, extern "C" fn(R, Any) -> Any>(code)(list, a(0)),
        2 => std::mem::transmute::<_, extern "C" fn(R, Any, Any) -> Any>(code)(list, a(0), a(1)),
        3 => std::mem::transmute::<_, extern "C" fn(R, Any, Any, Any) -> Any>(code)(
            list,
            a(0),
            a(1),
            a(2),
        ),
        4 => std::mem::transmute::<_, extern "C" fn(R, Any, Any, Any, Any) -> Any>(code)(
            list,
            a(0),
            a(1),
            a(2),
            a(3),
        ),
        5 => std::mem::transmute::<_, extern "C" fn(R, Any, Any, Any, Any, Any) -> Any>(code)(
            list,
            a(0),
            a(1),
            a(2),
            a(3),
            a(4),
        ),
        6 => std::mem::transmute::<_, extern "C" fn(R, Any, Any, Any, Any, Any, Any) -> Any>(code)(
            list,
            a(0),
            a(1),
            a(2),
            a(3),
            a(4),
            a(5),
        ),
        7 => std::mem::transmute::<_, extern "C" fn(R, Any, Any, Any, Any, Any, Any, Any) -> Any>(
            code,
        )(list, a(0), a(1), a(2), a(3), a(4), a(5), a(6)),
        8 => std::mem::transmute::<
            _,
            extern "C" fn(R, Any, Any, Any, Any, Any, Any, Any, Any) -> Any,
        >(code)(list, a(0), a(1), a(2), a(3), a(4), a(5), a(6), a(7)),
        _ => {
            return Err(ForeignError::new(
                "TypeError",
                format!("a call through a value passes at most {MAX_CALL_ARITY} arguments"),
            ));
        }
    })
}

/// Take a foreign box's opaque word and release only its header.
///
/// # Safety
/// `any` must be null or a live box owned by the caller.
unsafe fn take_word(any: Any) -> Option<usize> {
    let word = unsafe { word(any) }?;
    unsafe {
        (*any).data = std::ptr::null_mut();
        (*any).dropper = None;
        DynamicBox::free_raw(any);
    }
    Some(word)
}

/// The bytes the foreign object `any` is a buffer of, read in place;
/// `None` for any other value.
///
/// # Safety
/// `any` must be null or a live box, held while the bytes are read.
pub unsafe fn bytes_of<'a>(any: Any) -> Option<&'a [u8]> {
    let (p, len) = installed()?.bytes(unsafe { word(any) }?)?;
    Some(if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(p, len) }
    })
}

/// A dynamic value as the embedder reads it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value<'a> {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(&'a str),
    Foreign(usize),
    /// Several values as one, a tuple: its items.
    Tuple(&'a [Any]),
    /// A value of the program's own that has no reading here, with its tag.
    Other(u32),
}

/// What `any` holds.
///
/// # Safety
/// `any` must be null or a live box, and outlive what is read.
pub unsafe fn read<'a>(any: Any) -> Value<'a> {
    if any.is_null() {
        return Value::None;
    }
    let b = &*any;
    if b.tag.0 == FOREIGN_TAG {
        return Value::Foreign(b.data as usize);
    }
    if b.tag.0 == TUPLE_TAG {
        let list = b.data as *const ListHeader;
        return Value::Tuple(match list.as_ref() {
            Some(h) if h.len > 0 && !h.data.is_null() => {
                std::slice::from_raw_parts(h.data, h.len as usize)
            }
            _ => &[],
        });
    }
    // A number of any width, read at the size its box records.
    let p = b.data as *const u8;
    match (b.tag.category(), b.size) {
        (TypeCategory::Void, _) => Value::None,
        (_, _) if p.is_null() && b.tag.category() != TypeCategory::String => Value::None,
        (TypeCategory::Bool, _) => Value::Bool(*p != 0),
        (TypeCategory::Int, 1) => Value::Int(*(p as *const i8) as i64),
        (TypeCategory::Int, 2) => Value::Int(*(p as *const i16) as i64),
        (TypeCategory::Int, 4) => Value::Int(*(p as *const i32) as i64),
        (TypeCategory::Int, 8) => Value::Int(*(p as *const i64)),
        (TypeCategory::UInt, 1) => Value::Int(*p as i64),
        (TypeCategory::UInt, 2) => Value::Int(*(p as *const u16) as i64),
        (TypeCategory::UInt, 4) => Value::Int(*(p as *const u32) as i64),
        (TypeCategory::UInt, 8) => Value::Int(*(p as *const u64) as i64),
        (TypeCategory::Float, 4) => Value::Float(*(p as *const f32) as f64),
        (TypeCategory::Float, 8) => Value::Float(*(p as *const f64)),
        (TypeCategory::String, _) => match zrtl::string_as_str(b.data as StringConstPtr) {
            Some(text) => Value::Str(text),
            None => Value::Other(b.tag.0),
        },
        _ => Value::Other(b.tag.0),
    }
}

/// None.
pub fn none() -> Any {
    std::ptr::null_mut()
}

pub fn boolean(v: bool) -> Any {
    DynamicBox::owned_bool(v).into_raw()
}

pub fn int(v: i64) -> Any {
    DynamicBox::owned_i64(v).into_raw()
}

pub fn float(v: f64) -> Any {
    DynamicBox::owned_f64(v).into_raw()
}

/// A string box, laid out as the library's own.
pub fn string(text: &str) -> Any {
    DynamicBox {
        tag: TypeTag::STRING,
        size: std::mem::size_of::<*const u8>() as u32,
        data: zrtl::string_new(text) as *mut u8,
        dropper: Some(drop_string),
        display_fn: None,
    }
    .into_raw()
}

extern "C" fn drop_string(s: *mut u8) {
    // SAFETY: the string `string` made for this box, released once.
    unsafe { zrtl::string_free(s as StringPtr) }
}

thread_local! {
    /// The error the last operation on this thread reported, until the
    /// library takes it.
    static ERROR: RefCell<Option<ForeignError>> = const { RefCell::new(None) };
}

fn fail(error: ForeignError) {
    ERROR.with(|e| *e.borrow_mut() = Some(error));
}

fn no_embedder() -> ForeignError {
    ForeignError::new("TypeError", "no embedder handles foreign objects")
}

/// `f` with the installed [`Foreign`], its error kept for the library.
fn with<T>(default: T, f: impl FnOnce(&dyn Foreign) -> Result<T, ForeignError>) -> T {
    match installed().ok_or_else(no_embedder).and_then(f) {
        Ok(v) => v,
        Err(e) => {
            fail(e);
            default
        }
    }
}

/// # Safety
/// A string the program holds.
unsafe fn text_of<'a>(s: StringConstPtr) -> &'a str {
    zrtl::string_as_str(s).unwrap_or("")
}

/// # Safety
/// `data` holds `len` boxes.
unsafe fn args_of<'a>(data: i64, len: i64) -> &'a [Any] {
    if len <= 0 || data == 0 {
        return &[];
    }
    std::slice::from_raw_parts(data as *const Any, len as usize)
}

unsafe extern "C" fn foreign_get(x: Any, name: StringConstPtr) -> Any {
    let Some(object) = word(x) else {
        fail(no_embedder());
        return none();
    };
    with(none(), |f| f.get(object, text_of(name)))
}

unsafe extern "C" fn foreign_set(x: Any, name: StringConstPtr, value: Any) {
    let Some(object) = word(x) else {
        return fail(no_embedder());
    };
    with((), |f| f.set(object, text_of(name), value))
}

unsafe extern "C" fn foreign_call(x: Any, data: i64, len: i64) -> Any {
    let Some(object) = word(x) else {
        fail(no_embedder());
        return none();
    };
    with(none(), |f| f.call(object, args_of(data, len)))
}

unsafe extern "C" fn foreign_invoke(x: Any, name: StringConstPtr, data: i64, len: i64) -> Any {
    let Some(object) = word(x) else {
        fail(no_embedder());
        return none();
    };
    with(none(), |f| {
        f.invoke(object, text_of(name), args_of(data, len))
    })
}

extern "C" fn foreign_box_retained_word(word: i64) -> Any {
    let retained = with(0, |f| f.retain(word as usize));
    if retained == 0 {
        none()
    } else {
        boxed(retained)
    }
}

unsafe extern "C" fn foreign_retain_box_word(value: Any) -> i64 {
    let Some(word) = (unsafe { word(value) }) else {
        fail(no_embedder());
        return 0;
    };
    with(0, |f| f.retain(word).map(|word| word as i64))
}

extern "C" fn foreign_release_word(word: i64) {
    if word != 0
        && let Some(foreign) = installed()
    {
        foreign.release(word as usize);
    }
}

// Fixed-arity entries let typed frontends hand arguments to the embedder
// directly. The variadic entries above remain for calls whose arity is only
// known at runtime.
macro_rules! fixed_foreign_calls {
    ($call:ident, $construct:ident, $invoke:ident $(, $arg:ident)*) => {
        unsafe extern "C" fn $call(x: Any, $($arg: Any),*) -> Any {
            let Some(object) = (unsafe { word(x) }) else {
                fail(no_embedder());
                return none();
            };
            let args: &[Any] = &[$($arg),*];
            with(none(), |f| f.call(object, args))
        }

        unsafe extern "C" fn $construct(x: Any, $($arg: Any),*) -> Any {
            let Some(object) = (unsafe { word(x) }) else {
                fail(no_embedder());
                return none();
            };
            let args: &[Any] = &[$($arg),*];
            with(none(), |f| f.construct(object, args))
        }

        unsafe extern "C" fn $invoke(
            x: Any,
            name: StringConstPtr,
            $($arg: Any),*
        ) -> Any {
            let Some(object) = (unsafe { word(x) }) else {
                fail(no_embedder());
                return none();
            };
            let args: &[Any] = &[$($arg),*];
            with(none(), |f| f.invoke(object, unsafe { text_of(name) }, args))
        }
    };
}

fixed_foreign_calls!(foreign_call_0, foreign_construct_0, foreign_invoke_0);
fixed_foreign_calls!(foreign_call_1, foreign_construct_1, foreign_invoke_1, a0);
fixed_foreign_calls!(
    foreign_call_2,
    foreign_construct_2,
    foreign_invoke_2,
    a0,
    a1
);
fixed_foreign_calls!(
    foreign_call_3,
    foreign_construct_3,
    foreign_invoke_3,
    a0,
    a1,
    a2
);
fixed_foreign_calls!(
    foreign_call_4,
    foreign_construct_4,
    foreign_invoke_4,
    a0,
    a1,
    a2,
    a3
);
fixed_foreign_calls!(
    foreign_call_5,
    foreign_construct_5,
    foreign_invoke_5,
    a0,
    a1,
    a2,
    a3,
    a4
);
fixed_foreign_calls!(
    foreign_call_6,
    foreign_construct_6,
    foreign_invoke_6,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5
);
fixed_foreign_calls!(
    foreign_call_7,
    foreign_construct_7,
    foreign_invoke_7,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6
);
fixed_foreign_calls!(
    foreign_call_8,
    foreign_construct_8,
    foreign_invoke_8,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7
);

macro_rules! fixed_word_constructs {
    ($construct:ident $(, $arg:ident)*) => {
        unsafe extern "C" fn $construct(x: Any, $($arg: Any),*) -> i64 {
            let Some(object) = (unsafe { word(x) }) else {
                fail(no_embedder());
                return 0;
            };
            let args: &[Any] = &[$($arg),*];
            with(0, |f| f.construct_word(object, args).map(|word| word as i64))
        }
    };
}

fixed_word_constructs!(foreign_construct_word_0);
fixed_word_constructs!(foreign_construct_word_1, a0);
fixed_word_constructs!(foreign_construct_word_2, a0, a1);
fixed_word_constructs!(foreign_construct_word_3, a0, a1, a2);
fixed_word_constructs!(foreign_construct_word_4, a0, a1, a2, a3);
fixed_word_constructs!(foreign_construct_word_5, a0, a1, a2, a3, a4);
fixed_word_constructs!(foreign_construct_word_6, a0, a1, a2, a3, a4, a5);
fixed_word_constructs!(foreign_construct_word_7, a0, a1, a2, a3, a4, a5, a6);
fixed_word_constructs!(foreign_construct_word_8, a0, a1, a2, a3, a4, a5, a6, a7);

unsafe extern "C" fn foreign_get_float(x: Any, name: StringConstPtr) -> f64 {
    let Some(object) = (unsafe { word(x) }) else {
        fail(no_embedder());
        return f64::MAX;
    };
    with(f64::MAX, |f| f.get_float(object, unsafe { text_of(name) }))
}

unsafe extern "C" fn foreign_set_float(x: Any, name: StringConstPtr, value: f64) -> i32 {
    let Some(object) = (unsafe { word(x) }) else {
        fail(no_embedder());
        return 0;
    };
    with(0, |f| {
        f.set_float(object, unsafe { text_of(name) }, value)
            .map(|()| 1)
    })
}

unsafe extern "C" fn foreign_get_float_key(x: Any, key: i64) -> f64 {
    let Some(object) = (unsafe { word(x) }) else {
        fail(no_embedder());
        return f64::MAX;
    };
    with(f64::MAX, |f| f.get_float_key(object, key as u64))
}

unsafe extern "C" fn foreign_set_float_key(x: Any, key: i64, value: f64) -> i32 {
    let Some(object) = (unsafe { word(x) }) else {
        fail(no_embedder());
        return 0;
    };
    with(0, |f| {
        f.set_float_key(object, key as u64, value).map(|()| 1)
    })
}

macro_rules! fixed_float_calls {
    ($call:ident, $construct:ident, $invoke:ident $(, $arg:ident)*) => {
        unsafe extern "C" fn $call(x: Any, $($arg: f64),*) -> f64 {
            let Some(object) = (unsafe { word(x) }) else {
                fail(no_embedder());
                return f64::MAX;
            };
            let args: &[f64] = &[$($arg),*];
            with(f64::MAX, |f| f.call_float(object, args))
        }

        unsafe extern "C" fn $construct(x: Any, $($arg: f64),*) -> Any {
            let Some(object) = (unsafe { word(x) }) else {
                fail(no_embedder());
                return none();
            };
            let args: &[f64] = &[$($arg),*];
            with(none(), |f| f.construct_float(object, args))
        }

        unsafe extern "C" fn $invoke(
            x: Any,
            name: StringConstPtr,
            $($arg: f64),*
        ) -> f64 {
            let Some(object) = (unsafe { word(x) }) else {
                fail(no_embedder());
                return f64::MAX;
            };
            let args: &[f64] = &[$($arg),*];
            with(f64::MAX, |f| f.invoke_float(object, unsafe { text_of(name) }, args))
        }
    };
}

fixed_float_calls!(
    foreign_call_float_0,
    foreign_construct_float_0,
    foreign_invoke_float_0
);
fixed_float_calls!(
    foreign_call_float_1,
    foreign_construct_float_1,
    foreign_invoke_float_1,
    a0
);
fixed_float_calls!(
    foreign_call_float_2,
    foreign_construct_float_2,
    foreign_invoke_float_2,
    a0,
    a1
);
fixed_float_calls!(
    foreign_call_float_3,
    foreign_construct_float_3,
    foreign_invoke_float_3,
    a0,
    a1,
    a2
);
fixed_float_calls!(
    foreign_call_float_4,
    foreign_construct_float_4,
    foreign_invoke_float_4,
    a0,
    a1,
    a2,
    a3
);
fixed_float_calls!(
    foreign_call_float_5,
    foreign_construct_float_5,
    foreign_invoke_float_5,
    a0,
    a1,
    a2,
    a3,
    a4
);

macro_rules! fixed_float_key_invokes {
    ($invoke:ident $(, $arg:ident)*) => {
        unsafe extern "C" fn $invoke(x: Any, key: i64, $($arg: f64),*) -> f64 {
            let Some(object) = (unsafe { word(x) }) else {
                fail(no_embedder());
                return f64::MAX;
            };
            let args: &[f64] = &[$($arg),*];
            with(f64::MAX, |f| f.invoke_float_key(object, key as u64, args))
        }
    };
}

fixed_float_key_invokes!(foreign_invoke_float_key_0);
fixed_float_key_invokes!(foreign_invoke_float_key_1, a0);
fixed_float_key_invokes!(foreign_invoke_float_key_2, a0, a1);
fixed_float_key_invokes!(foreign_invoke_float_key_3, a0, a1, a2);
fixed_float_key_invokes!(foreign_invoke_float_key_4, a0, a1, a2, a3);
fixed_float_key_invokes!(foreign_invoke_float_key_5, a0, a1, a2, a3, a4);
fixed_float_key_invokes!(foreign_invoke_float_key_6, a0, a1, a2, a3, a4, a5);
fixed_float_key_invokes!(foreign_invoke_float_key_7, a0, a1, a2, a3, a4, a5, a6);
fixed_float_key_invokes!(foreign_invoke_float_key_8, a0, a1, a2, a3, a4, a5, a6, a7);
fixed_float_calls!(
    foreign_call_float_6,
    foreign_construct_float_6,
    foreign_invoke_float_6,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5
);
fixed_float_calls!(
    foreign_call_float_7,
    foreign_construct_float_7,
    foreign_invoke_float_7,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6
);
fixed_float_calls!(
    foreign_call_float_8,
    foreign_construct_float_8,
    foreign_invoke_float_8,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7
);

macro_rules! fixed_float_word_constructs {
    ($construct:ident $(, $arg:ident)*) => {
        unsafe extern "C" fn $construct(x: Any, $($arg: f64),*) -> i64 {
            let Some(object) = (unsafe { word(x) }) else {
                fail(no_embedder());
                return 0;
            };
            let args: &[f64] = &[$($arg),*];
            with(0, |f| {
                f.construct_float_word(object, args)
                    .map(|word| word as i64)
            })
        }
    };
}

fixed_float_word_constructs!(foreign_construct_float_word_0);
fixed_float_word_constructs!(foreign_construct_float_word_1, a0);
fixed_float_word_constructs!(foreign_construct_float_word_2, a0, a1);
fixed_float_word_constructs!(foreign_construct_float_word_3, a0, a1, a2);
fixed_float_word_constructs!(foreign_construct_float_word_4, a0, a1, a2, a3);
fixed_float_word_constructs!(foreign_construct_float_word_5, a0, a1, a2, a3, a4);
fixed_float_word_constructs!(foreign_construct_float_word_6, a0, a1, a2, a3, a4, a5);
fixed_float_word_constructs!(foreign_construct_float_word_7, a0, a1, a2, a3, a4, a5, a6);
fixed_float_word_constructs!(
    foreign_construct_float_word_8,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7
);

unsafe extern "C" fn foreign_str(x: Any) -> StringPtr {
    let text = match (word(x), installed()) {
        (Some(object), Some(f)) => f.text(object),
        _ => String::new(),
    };
    zrtl::string_new(&text)
}

unsafe extern "C" fn foreign_type(x: Any) -> StringPtr {
    let name = match (word(x), installed()) {
        (Some(object), Some(f)) => f.type_name(object),
        _ => String::new(),
    };
    zrtl::string_new(&name)
}

unsafe extern "C" fn foreign_eq(a: Any, b: Any) -> i32 {
    match (word(a), word(b), installed()) {
        (Some(a), Some(b), Some(f)) => f.equals(a, b) as i32,
        (Some(a), Some(b), None) => (a == b) as i32,
        _ => 0,
    }
}

unsafe extern "C" fn foreign_hash(x: Any) -> i64 {
    match (word(x), installed()) {
        (Some(object), Some(f)) => f.hash(object),
        (Some(object), None) => object as i64,
        _ => 0,
    }
}

unsafe extern "C" fn foreign_import(name: StringConstPtr) -> Any {
    let Some(f) = installed() else {
        return none();
    };
    match f.import(text_of(name)) {
        Ok(module) => module.unwrap_or_else(none),
        Err(e) => {
            fail(e);
            none()
        }
    }
}

/// The length of the bytes `x` is a buffer of, or -1 when it is none.
unsafe extern "C" fn foreign_bytes_len(x: Any) -> i64 {
    match unsafe { bytes_of(x) } {
        Some(bytes) => bytes.len() as i64,
        None => -1,
    }
}

/// How many values `x` is when it is several values given at once, or -1
/// when it is not.
unsafe extern "C" fn foreign_values_len(x: Any) -> i64 {
    match unsafe { word(x) } {
        Some(object) => with(-1, |f| Ok(f.values(object).map_or(-1, |n| n as i64))),
        None => -1,
    }
}

/// Value `index` of such an `x`.
unsafe extern "C" fn foreign_value(x: Any, index: i64) -> Any {
    let Some(object) = (unsafe { word(x) }) else {
        return none();
    };
    with(none(), |f| Ok(f.value(object, index.max(0) as usize)))
}

/// The kind of the error the last operation reported, or null when it
/// reported none.
extern "C" fn foreign_error_kind() -> StringPtr {
    ERROR.with(|e| match &*e.borrow() {
        Some(error) => zrtl::string_new(error.kind),
        None => std::ptr::null_mut(),
    })
}

/// The message of the error the last operation reported, which it takes.
extern "C" fn foreign_error_message() -> StringPtr {
    let message = ERROR.with(|e| e.borrow_mut().take().map(|e| e.message));
    zrtl::string_new(message.as_deref().unwrap_or(""))
}

static INFO: zrtl::ZrtlInfo = zrtl::ZrtlInfo::new(c"foreign".as_ptr());
static SYMBOLS: &[zrtl::ZrtlSymbol] = &[
    zrtl::ZrtlSymbol::new(
        c"$Foreign$box_retained_word".as_ptr(),
        foreign_box_retained_word as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$retain_box_word".as_ptr(),
        foreign_retain_box_word as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$release_word".as_ptr(),
        foreign_release_word as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_word0".as_ptr(),
        foreign_construct_word_0 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_word1".as_ptr(),
        foreign_construct_word_1 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_word2".as_ptr(),
        foreign_construct_word_2 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_word3".as_ptr(),
        foreign_construct_word_3 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_word4".as_ptr(),
        foreign_construct_word_4 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_word5".as_ptr(),
        foreign_construct_word_5 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_word6".as_ptr(),
        foreign_construct_word_6 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_word7".as_ptr(),
        foreign_construct_word_7 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_word8".as_ptr(),
        foreign_construct_word_8 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float_word0".as_ptr(),
        foreign_construct_float_word_0 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float_word1".as_ptr(),
        foreign_construct_float_word_1 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float_word2".as_ptr(),
        foreign_construct_float_word_2 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float_word3".as_ptr(),
        foreign_construct_float_word_3 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float_word4".as_ptr(),
        foreign_construct_float_word_4 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float_word5".as_ptr(),
        foreign_construct_float_word_5 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float_word6".as_ptr(),
        foreign_construct_float_word_6 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float_word7".as_ptr(),
        foreign_construct_float_word_7 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float_word8".as_ptr(),
        foreign_construct_float_word_8 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Foreign$get".as_ptr(), foreign_get as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$set".as_ptr(), foreign_set as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call".as_ptr(), foreign_call as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke".as_ptr(), foreign_invoke as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call0".as_ptr(), foreign_call_0 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call1".as_ptr(), foreign_call_1 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call2".as_ptr(), foreign_call_2 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call3".as_ptr(), foreign_call_3 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call4".as_ptr(), foreign_call_4 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call5".as_ptr(), foreign_call_5 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call6".as_ptr(), foreign_call_6 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call7".as_ptr(), foreign_call_7 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call8".as_ptr(), foreign_call_8 as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct0".as_ptr(),
        foreign_construct_0 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct1".as_ptr(),
        foreign_construct_1 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct2".as_ptr(),
        foreign_construct_2 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct3".as_ptr(),
        foreign_construct_3 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct4".as_ptr(),
        foreign_construct_4 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct5".as_ptr(),
        foreign_construct_5 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct6".as_ptr(),
        foreign_construct_6 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct7".as_ptr(),
        foreign_construct_7 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct8".as_ptr(),
        foreign_construct_8 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke0".as_ptr(), foreign_invoke_0 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke1".as_ptr(), foreign_invoke_1 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke2".as_ptr(), foreign_invoke_2 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke3".as_ptr(), foreign_invoke_3 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke4".as_ptr(), foreign_invoke_4 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke5".as_ptr(), foreign_invoke_5 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke6".as_ptr(), foreign_invoke_6 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke7".as_ptr(), foreign_invoke_7 as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke8".as_ptr(), foreign_invoke_8 as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$get_float".as_ptr(),
        foreign_get_float as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$set_float".as_ptr(),
        foreign_set_float as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$get_float_key".as_ptr(),
        foreign_get_float_key as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$set_float_key".as_ptr(),
        foreign_set_float_key as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$call_float0".as_ptr(),
        foreign_call_float_0 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$call_float1".as_ptr(),
        foreign_call_float_1 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$call_float2".as_ptr(),
        foreign_call_float_2 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$call_float3".as_ptr(),
        foreign_call_float_3 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$call_float4".as_ptr(),
        foreign_call_float_4 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$call_float5".as_ptr(),
        foreign_call_float_5 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$call_float6".as_ptr(),
        foreign_call_float_6 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$call_float7".as_ptr(),
        foreign_call_float_7 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$call_float8".as_ptr(),
        foreign_call_float_8 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float0".as_ptr(),
        foreign_construct_float_0 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float1".as_ptr(),
        foreign_construct_float_1 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float2".as_ptr(),
        foreign_construct_float_2 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float3".as_ptr(),
        foreign_construct_float_3 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float4".as_ptr(),
        foreign_construct_float_4 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float5".as_ptr(),
        foreign_construct_float_5 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float6".as_ptr(),
        foreign_construct_float_6 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float7".as_ptr(),
        foreign_construct_float_7 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$construct_float8".as_ptr(),
        foreign_construct_float_8 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float0".as_ptr(),
        foreign_invoke_float_0 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float1".as_ptr(),
        foreign_invoke_float_1 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float2".as_ptr(),
        foreign_invoke_float_2 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float3".as_ptr(),
        foreign_invoke_float_3 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float4".as_ptr(),
        foreign_invoke_float_4 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float5".as_ptr(),
        foreign_invoke_float_5 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float6".as_ptr(),
        foreign_invoke_float_6 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float7".as_ptr(),
        foreign_invoke_float_7 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float8".as_ptr(),
        foreign_invoke_float_8 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float_key0".as_ptr(),
        foreign_invoke_float_key_0 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float_key1".as_ptr(),
        foreign_invoke_float_key_1 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float_key2".as_ptr(),
        foreign_invoke_float_key_2 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float_key3".as_ptr(),
        foreign_invoke_float_key_3 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float_key4".as_ptr(),
        foreign_invoke_float_key_4 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float_key5".as_ptr(),
        foreign_invoke_float_key_5 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float_key6".as_ptr(),
        foreign_invoke_float_key_6 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float_key7".as_ptr(),
        foreign_invoke_float_key_7 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$invoke_float_key8".as_ptr(),
        foreign_invoke_float_key_8 as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Foreign$str".as_ptr(), foreign_str as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$type".as_ptr(), foreign_type as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$eq".as_ptr(), foreign_eq as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$hash".as_ptr(), foreign_hash as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$import".as_ptr(), foreign_import as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$bytes_len".as_ptr(),
        foreign_bytes_len as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$values_len".as_ptr(),
        foreign_values_len as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Foreign$value".as_ptr(), foreign_value as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$error_kind".as_ptr(),
        foreign_error_kind as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Foreign$error_message".as_ptr(),
        foreign_error_message as *const u8,
    ),
];

/// The `$Foreign$*` symbols the built-in library's foreign operations
/// link against, as a plugin the runtime links like any other.
pub fn static_plugin() -> zrtl::StaticPlugin {
    zrtl::StaticPlugin {
        info: &INFO,
        symbols: SYMBOLS,
    }
}
