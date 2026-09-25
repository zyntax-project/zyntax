//! Lua values as the runtime boxes them, read and made without the
//! library: nil is a null box, booleans, integers, floats and strings
//! are boxes of the shared categories, and everything else is a box
//! under one of the frontend's tags, which [`crate::install`] hands
//! over.

use std::sync::OnceLock;

use zrtl::{DynamicBox, TypeCategory, TypeTag};

/// A Lua value: a box, or null for nil.
pub type Any = *const DynamicBox;
/// A string's header.
pub type Str = *const i32;
/// A list of values as the compiler lays it out.
pub type List = *mut ListHeader;

/// The header a list value points at.
#[repr(C)]
pub struct ListHeader {
    pub data: *mut Any,
    pub len: i64,
    pub capacity: i64,
}

/// The frontend's box tags for the values this API distinguishes. A
/// value under any other tag of the custom category is userdata to C
/// code: a file, say.
#[derive(Clone, Copy, Debug)]
pub struct Tags {
    pub table: u32,
    pub thread: u32,
    pub function: u32,
    pub code: u32,
    /// Full userdata: the address of a block this crate lays out.
    pub userdata: u32,
    /// Light userdata: the pointer itself.
    pub light: u32,
}

static TAGS: OnceLock<Tags> = OnceLock::new();

pub(crate) fn set_tags(tags: Tags) {
    let _ = TAGS.set(tags);
}

pub(crate) fn tags() -> &'static Tags {
    TAGS.get()
        .expect("the C API is installed before a value is read")
}

// The type codes of lua.h.
pub const LUA_TNONE: i32 = -1;
pub const LUA_TNIL: i32 = 0;
pub const LUA_TBOOLEAN: i32 = 1;
pub const LUA_TLIGHTUSERDATA: i32 = 2;
pub const LUA_TNUMBER: i32 = 3;
pub const LUA_TSTRING: i32 = 4;
pub const LUA_TTABLE: i32 = 5;
pub const LUA_TFUNCTION: i32 = 6;
pub const LUA_TUSERDATA: i32 = 7;
pub const LUA_TTHREAD: i32 = 8;

/// A value's Lua type.
///
/// # Safety
/// `v` is null or a live box.
pub unsafe fn type_of(v: Any) -> i32 {
    let Some(b) = (unsafe { v.as_ref() }) else {
        return LUA_TNIL;
    };
    match b.tag.category() {
        TypeCategory::Void => LUA_TNIL,
        TypeCategory::Bool => LUA_TBOOLEAN,
        TypeCategory::Int | TypeCategory::UInt | TypeCategory::Float => LUA_TNUMBER,
        TypeCategory::String => LUA_TSTRING,
        _ => {
            let t = tags();
            let raw = b.tag.raw();
            if raw == t.table {
                LUA_TTABLE
            } else if raw == t.function || raw == t.code {
                LUA_TFUNCTION
            } else if raw == t.thread {
                LUA_TTHREAD
            } else if raw == t.light {
                LUA_TLIGHTUSERDATA
            } else {
                LUA_TUSERDATA
            }
        }
    }
}

/// Whether `v` is a box under `tag`.
///
/// # Safety
/// `v` is null or a live box.
pub unsafe fn has_tag(v: Any, tag: u32) -> bool {
    unsafe { v.as_ref() }.is_some_and(|b| b.tag.raw() == tag)
}

/// What a number box holds.
pub enum Number {
    Int(i64),
    Float(f64),
}

/// The number a box holds, if it holds one.
///
/// # Safety
/// `v` is null or a live box.
pub unsafe fn number_of(v: Any) -> Option<Number> {
    let b = unsafe { v.as_ref() }?;
    if b.data.is_null() {
        return None;
    }
    match b.tag.category() {
        TypeCategory::Int => Some(Number::Int(unsafe { int_payload(b) })),
        TypeCategory::UInt => Some(Number::Int(unsafe { int_payload(b) })),
        TypeCategory::Float => Some(Number::Float(if b.size == 4 {
            unsafe { *(b.data as *const f32) as f64 }
        } else {
            unsafe { *(b.data as *const f64) }
        })),
        _ => None,
    }
}

/// An integer payload of whatever width the box records.
unsafe fn int_payload(b: &DynamicBox) -> i64 {
    let signed = b.tag.category() == TypeCategory::Int;
    unsafe {
        match (b.size, signed) {
            (1, true) => *(b.data as *const i8) as i64,
            (2, true) => *(b.data as *const i16) as i64,
            (4, true) => *(b.data as *const i32) as i64,
            (1, false) => *b.data as i64,
            (2, false) => *(b.data as *const u16) as i64,
            (4, false) => *(b.data as *const u32) as i64,
            _ => *(b.data as *const i64),
        }
    }
}

/// The truth of a value: everything but nil and false.
///
/// # Safety
/// `v` is null or a live box.
pub unsafe fn truth(v: Any) -> bool {
    let Some(b) = (unsafe { v.as_ref() }) else {
        return false;
    };
    match b.tag.category() {
        TypeCategory::Void => false,
        TypeCategory::Bool => !b.data.is_null() && unsafe { *b.data } != 0,
        _ => true,
    }
}

/// The string a string box holds, or null.
///
/// # Safety
/// `v` is null or a live box.
pub unsafe fn string_of(v: Any) -> Str {
    match unsafe { v.as_ref() } {
        Some(b) if b.tag.category() == TypeCategory::String => b.data as Str,
        _ => std::ptr::null(),
    }
}

/// A string's bytes.
///
/// # Safety
/// `s` is null or a live string.
pub unsafe fn bytes_of<'a>(s: Str) -> &'a [u8] {
    unsafe { zrtl::string::string_as_bytes(s) }
}

/// The pointer a box under a frontend tag carries.
///
/// # Safety
/// `v` is a live box.
pub unsafe fn payload(v: Any) -> *mut u8 {
    unsafe { (*v).data }
}

/// Whether a string's bytes are followed by a zero byte its own
/// storage holds: its layout ends with the bytes, and the byte after
/// them lies in the sixteen-byte granule of the last one, which every
/// string's storage fills. A string that cannot be shown to be is
/// copied by the caller.
///
/// # Safety
/// `s` is a live string.
pub unsafe fn is_terminated(s: Str) -> bool {
    use zrtl::string::{BUILDER, INDEXED, STRING_HEADER_SIZE, string_header};
    let h = unsafe { string_header(s) };
    if h.flags & (INDEXED | BUILDER) != 0 {
        return false;
    }
    let end = STRING_HEADER_SIZE + h.byte_len as usize;
    if end % 16 == 0 {
        return false;
    }
    unsafe { *(s as *const u8).add(end) == 0 }
}

fn alloc(size: usize, align: usize) -> *mut u8 {
    // SAFETY: every size asked for here is at least one.
    let p = unsafe { zrtl::heap::alloc(size.max(1), align) };
    assert!(!p.is_null(), "the heap is exhausted");
    p
}

/// A zeroed block of the program's heap, where the collector sees it.
pub fn zeroed(size: usize) -> *mut u8 {
    let p = alloc(size, 16);
    // SAFETY: the block is `size` bytes.
    unsafe { std::ptr::write_bytes(p, 0, size) };
    p
}

/// A string of `bytes` followed by a zero byte: TEXT when the bytes
/// are UTF-8 and need no character index.
pub fn new_string(bytes: &[u8]) -> Str {
    use zrtl::string::{INDEXED, TEXT, layout_of, string_init};
    let utf8 = zrtl::string::text_char_len(bytes);
    let (char_len, text) = match utf8 {
        Some(n) => {
            let (flags, _) = layout_of(bytes.len(), n, true);
            if flags & INDEXED == 0 {
                (n, true)
            } else {
                (bytes.len(), false)
            }
        }
        None => (bytes.len(), false),
    };
    let (flags, total) = layout_of(bytes.len(), char_len, text);
    debug_assert!(flags & INDEXED == 0);
    debug_assert!(!text || flags & TEXT != 0);
    let p = alloc(total + 1, 16);
    // SAFETY: the block holds the string's layout and one byte more.
    unsafe {
        string_init(p, bytes, char_len as u32, flags);
        *p.add(total) = 0;
    }
    p as Str
}

fn boxed(tag: TypeTag, size: u32, data: *mut u8) -> Any {
    DynamicBox {
        tag,
        size,
        data,
        dropper: None,
        display_fn: None,
    }
    .into_raw()
}

fn scalar<T: Copy>(value: T) -> *mut u8 {
    let p = alloc(std::mem::size_of::<T>(), 8) as *mut T;
    // SAFETY: the block is as large as a `T`.
    unsafe { p.write(value) };
    p as *mut u8
}

pub fn box_int(v: i64) -> Any {
    boxed(TypeTag::I64, 8, scalar(v))
}

pub fn box_float(v: f64) -> Any {
    boxed(TypeTag::F64, 8, scalar(v))
}

pub fn box_bool(v: bool) -> Any {
    boxed(TypeTag::BOOL, 1, scalar(v as u8))
}

pub fn box_string(s: Str) -> Any {
    boxed(
        TypeTag::STRING,
        std::mem::size_of::<usize>() as u32,
        s as *mut u8,
    )
}

pub fn box_bytes(bytes: &[u8]) -> Any {
    box_string(new_string(bytes))
}

/// A box carrying `p` under a frontend `tag`.
pub fn box_pointer(p: *mut u8, tag: u32) -> Any {
    boxed(
        TypeTag::from_raw(tag),
        std::mem::size_of::<usize>() as u32,
        p,
    )
}

/// The items of a list.
///
/// # Safety
/// `list` is null or a live list.
pub unsafe fn items<'a>(list: List) -> &'a [Any] {
    match unsafe { list.as_ref() } {
        Some(h) if h.len > 0 && !h.data.is_null() => unsafe {
            std::slice::from_raw_parts(h.data, h.len as usize)
        },
        _ => &[],
    }
}

/// The list a boxed list or tuple holds.
///
/// # Safety
/// `v` is a live box of a list kind.
pub unsafe fn list_of(v: Any) -> List {
    unsafe { payload(v) as List }
}
