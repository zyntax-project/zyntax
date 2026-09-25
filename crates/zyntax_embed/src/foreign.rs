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

    /// Call the method `name` of `object` with `args`.
    fn invoke(&self, object: usize, name: &str, args: &[Any]) -> Result<Any, ForeignError>;

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

/// A dynamic value as the embedder reads it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value<'a> {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(&'a str),
    Foreign(usize),
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
static SYMBOLS: [zrtl::ZrtlSymbol; 11] = [
    zrtl::ZrtlSymbol::new(c"$Foreign$get".as_ptr(), foreign_get as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$set".as_ptr(), foreign_set as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$call".as_ptr(), foreign_call as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$invoke".as_ptr(), foreign_invoke as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$str".as_ptr(), foreign_str as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$type".as_ptr(), foreign_type as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$eq".as_ptr(), foreign_eq as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$hash".as_ptr(), foreign_hash as *const u8),
    zrtl::ZrtlSymbol::new(c"$Foreign$import".as_ptr(), foreign_import as *const u8),
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
        symbols: &SYMBOLS,
    }
}
