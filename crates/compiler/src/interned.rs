//! Boxes every program shares.
//!
//! The two booleans and the small integers are boxed so often, and
//! carry so little, that one box of each serves every use: a value of
//! these kinds is immutable, and nothing tells one box of `3` from
//! another. The boxes live for the process; a release of one is a
//! no-op, which the pool checks for before anything else.
//!
//! The box pass hands compiled code their addresses, so they are laid
//! out exactly as a box it would make: the header and the payload
//! right after it, `data` pointing at the payload.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::zrtl::TypeTag;

/// The smallest and the count of integers that have a shared box.
pub const SMALL_INT_MIN: i64 = -5;
pub const SMALL_INT_COUNT: usize = 262;

/// A box with its payload beside it: the layout the box pass makes.
#[repr(C)]
pub struct StaticBox {
    tag: u32,
    size: u32,
    data: *mut u8,
    dropper: usize,
    display: usize,
    payload: i64,
}

/// Bytes from one box to the next in the tables.
pub const STRIDE: usize = std::mem::size_of::<StaticBox>();

/// One table: the two booleans, then the integers, so that whether an
/// address is a shared box is one subtraction and one compare.
struct Table(Box<[StaticBox]>);

// SAFETY: the table is written once, before any address of it is handed
// out, and never again.
unsafe impl Sync for Table {}
unsafe impl Send for Table {}

static TABLE: OnceLock<Table> = OnceLock::new();
/// The table's address and length in bytes, for the release check, which
/// runs on every release and reads nothing else.
static BASE: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

fn table() -> &'static [StaticBox] {
    &TABLE
        .get_or_init(|| {
            let entries = [(TypeTag::BOOL.0, 1u32, 0i64), (TypeTag::BOOL.0, 1, 1)]
                .into_iter()
                .chain(
                    (0..SMALL_INT_COUNT).map(|i| (TypeTag::I64.0, 8u32, SMALL_INT_MIN + i as i64)),
                );
            let mut boxes: Box<[StaticBox]> = entries
                .map(|(tag, size, v)| StaticBox {
                    tag,
                    size,
                    data: std::ptr::null_mut(),
                    dropper: 0,
                    display: 0,
                    payload: v,
                })
                .collect();
            for b in boxes.iter_mut() {
                b.data = &mut b.payload as *mut i64 as *mut u8;
            }
            BYTES.store(boxes.len() * STRIDE, Ordering::Relaxed);
            BASE.store(boxes.as_ptr() as usize, Ordering::Release);
            Table(boxes)
        })
        .0
}

/// Address of the box of the smallest shared integer; the box of
/// `SMALL_INT_MIN + i` is `STRIDE * i` bytes on.
pub fn small_int_base() -> usize {
    &table()[2] as *const StaticBox as usize
}

/// Address of the box of `b`.
pub fn bool_box(b: bool) -> usize {
    &table()[usize::from(b)] as *const StaticBox as usize
}

/// Whether `p` is one of the shared boxes, which no release may touch.
/// Before the table exists nothing can hold one of its addresses, and
/// the base reads as zero.
#[inline]
pub fn is_interned(p: usize) -> bool {
    p.wrapping_sub(BASE.load(Ordering::Relaxed)) < BYTES.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shared_boxes_read_as_boxes() {
        let three = small_int_base() + STRIDE * (3 - SMALL_INT_MIN) as usize;
        // SAFETY: an address from the table, laid out as a StaticBox.
        let b = unsafe { &*(three as *const StaticBox) };
        assert_eq!(b.tag, TypeTag::I64.0);
        assert_eq!(unsafe { *(b.data as *const i64) }, 3);
        let t = unsafe { &*(bool_box(true) as *const StaticBox) };
        assert_eq!(t.tag, TypeTag::BOOL.0);
        assert_eq!(unsafe { *t.data }, 1);
        assert!(is_interned(three));
        assert!(is_interned(bool_box(false)));
        assert!(!is_interned(three + SMALL_INT_COUNT * STRIDE));
    }
}
