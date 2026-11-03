//! Packing support.

use super::*;

/// One argument as `string.pack` takes it.
pub(super) unsafe fn read_pack_arg<'a>(b: *const DynamicBox) -> crate::pack::PackArg<'a> {
    use crate::pack::PackArg;
    if b.is_null() {
        return PackArg::Other("nil");
    }
    let b = unsafe { &*b };
    match b.tag.category() {
        TypeCategory::Void => PackArg::Other("nil"),
        TypeCategory::Bool => PackArg::Other("boolean"),
        TypeCategory::Int | TypeCategory::UInt => {
            if b.tag == TypeTag::I64 {
                PackArg::Int(b.as_i64().unwrap_or(0))
            } else {
                PackArg::Int(b.as_i32().map(i64::from).unwrap_or(0))
            }
        }
        TypeCategory::Float => PackArg::Float(b.as_f64().unwrap_or(0.0)),
        TypeCategory::String => PackArg::Str(unsafe { bytes_of(b.data as zrtl::StringConstPtr) }),
        _ => PackArg::Other(type_word(b)),
    }
}

/// The packed bytes, or null with the message kept for `pack_error`.
pub(super) extern "C" fn host_pack(
    fmt: zrtl::StringConstPtr,
    args: *const ListHeader,
) -> StringPtr {
    let fmt = unsafe { bytes_of(fmt) };
    let args: Vec<crate::pack::PackArg<'_>> = if args.is_null() {
        Vec::new()
    } else {
        let header = unsafe { &*args };
        let len = header.len.max(0) as usize;
        (0..len)
            .map(|i| unsafe { read_pack_arg(*header.data.add(i)) })
            .collect()
    };
    match crate::pack::pack(fmt, &args) {
        Ok(bytes) => zrtl::string::string_from_bytes(&bytes),
        Err(e) => {
            PACK_ERROR.with(|err| *err.borrow_mut() = e);
            std::ptr::null_mut()
        }
    }
}

pub(super) extern "C" fn host_pack_error() -> StringPtr {
    PACK_ERROR.with(|e| zrtl::string::string_from_bytes(e.borrow().as_bytes()))
}

/// The size, or -1 with the message kept.
pub(super) extern "C" fn host_packsize(fmt: zrtl::StringConstPtr) -> i64 {
    match crate::pack::packsize(unsafe { bytes_of(fmt) }) {
        Ok(n) => n,
        Err(e) => {
            PACK_ERROR.with(|err| *err.borrow_mut() = e);
            -1
        }
    }
}

/// How many values were unpacked, kept for the accessors, or -1 with
/// the message kept. `pos` is a byte offset.
pub(super) extern "C" fn host_unpack(
    fmt: zrtl::StringConstPtr,
    s: zrtl::StringConstPtr,
    pos: i64,
) -> i64 {
    let (fmt, data) = unsafe { (bytes_of(fmt), bytes_of(s)) };
    match crate::pack::unpack(fmt, data, pos.max(0) as usize) {
        Ok((values, next)) => {
            let n = values.len() as i64;
            UNPACKED.with(|u| *u.borrow_mut() = (values, next));
            n
        }
        Err(e) => {
            PACK_ERROR.with(|err| *err.borrow_mut() = e);
            -1
        }
    }
}

/// 0 for an integer, 1 for a float, 2 for a string.
pub(super) extern "C" fn host_unpack_kind(i: i64) -> i64 {
    UNPACKED.with(|u| match u.borrow().0.get(i as usize) {
        Some(crate::pack::Unpacked::Int(_)) => 0,
        Some(crate::pack::Unpacked::Float(_)) => 1,
        _ => 2,
    })
}

pub(super) extern "C" fn host_unpack_int(i: i64) -> i64 {
    UNPACKED.with(|u| match u.borrow().0.get(i as usize) {
        Some(crate::pack::Unpacked::Int(v)) => *v,
        _ => 0,
    })
}

pub(super) extern "C" fn host_unpack_float(i: i64) -> f64 {
    UNPACKED.with(|u| match u.borrow().0.get(i as usize) {
        Some(crate::pack::Unpacked::Float(v)) => *v,
        _ => 0.0,
    })
}

pub(super) extern "C" fn host_unpack_str(i: i64) -> StringPtr {
    UNPACKED.with(|u| match u.borrow().0.get(i as usize) {
        Some(crate::pack::Unpacked::Str(v)) => zrtl::string::string_from_bytes(v),
        _ => zrtl::string::string_from_bytes(b""),
    })
}

/// The position after the last unpacked value, 1-based.
pub(super) extern "C" fn host_unpack_next() -> i64 {
    UNPACKED.with(|u| u.borrow().1 as i64)
}
