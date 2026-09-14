//! Where the strings and boxes this SDK makes are allocated.
//!
//! A host that manages the program's heap installs its own allocator
//! with [`set_allocator`], so that a string a plugin builds is a block
//! of the same heap as everything the compiled program makes, and
//! whatever reclaims that heap sees it. Until a host does, the system
//! allocator serves. A plugin loaded from a file is its own copy of
//! this module; the loader installs the allocator into it through the
//! `_zrtl_set_allocator` export the plugin macro emits.
//!
//! Install before the first allocation: a block from one allocator
//! must not be released through the other.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Allocate `size` bytes, sixteen-aligned, never null for a size the
/// system could serve.
pub type AllocFn = unsafe extern "C" fn(usize) -> *mut u8;
/// Release a block from the matching [`AllocFn`]; null is ignored.
pub type FreeFn = unsafe extern "C" fn(*mut u8);

static ALLOC: AtomicUsize = AtomicUsize::new(0);
static FREE: AtomicUsize = AtomicUsize::new(0);

/// Route every string and box allocation through `alloc` and `free`.
pub fn set_allocator(alloc: AllocFn, free: FreeFn) {
    ALLOC.store(alloc as usize, Ordering::SeqCst);
    FREE.store(free as usize, Ordering::SeqCst);
}

/// Whether a host has installed an allocator.
pub fn has_allocator() -> bool {
    ALLOC.load(Ordering::Relaxed) != 0
}

/// Allocate `size` bytes at `align`, which is at most sixteen.
///
/// # Safety
/// As `std::alloc::alloc`: `size` must not be zero.
pub unsafe fn alloc(size: usize, align: usize) -> *mut u8 {
    let f = ALLOC.load(Ordering::Relaxed);
    if f != 0 {
        let alloc: AllocFn = std::mem::transmute(f);
        return alloc(size);
    }
    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, align))
}

/// Release a block from [`alloc`] of the same `size` and `align`.
///
/// # Safety
/// `ptr` must have come from [`alloc`] with these arguments and must
/// not be used afterwards.
pub unsafe fn free(ptr: *mut u8, size: usize, align: usize) {
    if ptr.is_null() {
        return;
    }
    let f = FREE.load(Ordering::Relaxed);
    if f != 0 {
        let free: FreeFn = std::mem::transmute(f);
        return free(ptr);
    }
    std::alloc::dealloc(
        ptr,
        std::alloc::Layout::from_size_align_unchecked(size, align),
    );
}
