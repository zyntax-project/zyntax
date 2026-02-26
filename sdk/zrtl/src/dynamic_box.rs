//! DynamicBox - Runtime Boxed Values
//!
//! `DynamicBox` is the standard way to pass opaque/polymorphic values
//! between Zyntax code and native runtime functions.
//!
//! This is equivalent to `ZrtlDynamicBox` in the C SDK.

use crate::type_system::{TypeCategory, TypeTag};

/// Drop function type for custom types (C ABI compatible)
pub type DropFn = extern "C" fn(*mut u8);

/// Display function type for formatting values (C ABI compatible)
/// Takes a pointer to the data and a ZrtlString buffer to write into
/// Returns the formatted string (same as buffer, or newly allocated if needed)
pub type DisplayFn = extern "C" fn(*const u8) -> *const u8;

/// Dynamic boxed value - C ABI compatible
///
/// Layout (32 bytes on 64-bit):
/// - tag: Type tag identifying the contained type
/// - size: Size of the data in bytes
/// - data: Pointer to the actual data
/// - dropper: Optional destructor function
/// - display_fn: Optional display/formatting function
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DynamicBox {
    /// Type tag identifying the type
    pub tag: TypeTag,
    /// Size of data in bytes
    pub size: u32,
    /// Pointer to the data
    pub data: *mut u8,
    /// Optional destructor (null if no cleanup needed)
    pub dropper: Option<DropFn>,
    /// Optional display function for formatting (null if no Display trait)
    pub display_fn: Option<DisplayFn>,
}

impl DynamicBox {
    /// Create a null/empty box
    #[inline]
    pub const fn null() -> Self {
        Self {
            tag: TypeTag::VOID,
            size: 0,
            data: std::ptr::null_mut(),
            dropper: None,
            display_fn: None,
        }
    }

    /// Check if the box is null/empty
    #[inline]
    pub fn is_null(&self) -> bool {
        self.data.is_null()
    }

    /// Get the type category
    #[inline]
    pub fn category(&self) -> TypeCategory {
        self.tag.category()
    }

    /// Check if this matches a category
    #[inline]
    pub fn is_category(&self, cat: TypeCategory) -> bool {
        self.tag.is_category(cat)
    }

    /// Get data as typed pointer (caller must verify type first)
    ///
    /// # Safety
    /// Caller must ensure the actual type matches T
    #[inline]
    pub unsafe fn as_ptr<T>(&self) -> *const T {
        self.data as *const T
    }

    /// Get data as mutable typed pointer
    ///
    /// # Safety
    /// Caller must ensure the actual type matches T
    #[inline]
    pub unsafe fn as_mut_ptr<T>(&mut self) -> *mut T {
        self.data as *mut T
    }

    /// Get data as typed reference
    ///
    /// # Safety
    /// Caller must ensure the actual type matches T
    #[inline]
    pub unsafe fn as_ref<T>(&self) -> Option<&T> {
        if self.data.is_null() {
            None
        } else {
            Some(&*(self.data as *const T))
        }
    }

    /// Get data as mutable typed reference
    ///
    /// # Safety
    /// Caller must ensure the actual type matches T
    #[inline]
    pub unsafe fn as_mut<T>(&mut self) -> Option<&mut T> {
        if self.data.is_null() {
            None
        } else {
            Some(&mut *(self.data as *mut T))
        }
    }

    /// Free the boxed value
    ///
    /// Calls the dropper if present, then clears the box.
    pub fn free(&mut self) {
        if !self.data.is_null() {
            if let Some(dropper) = self.dropper {
                dropper(self.data);
            }
            self.data = std::ptr::null_mut();
            self.size = 0;
            self.tag = TypeTag::VOID;
            self.dropper = None;
        }
    }

    // ========================================================================
    // Type-safe accessors for primitive types
    // ========================================================================

    /// Get as i8 (type-checked)
    pub fn as_i8(&self) -> Option<i8> {
        if self.tag == TypeTag::I8 {
            unsafe { self.as_ref::<i8>().copied() }
        } else {
            None
        }
    }

    /// Get as i16 (type-checked)
    pub fn as_i16(&self) -> Option<i16> {
        if self.tag == TypeTag::I16 {
            unsafe { self.as_ref::<i16>().copied() }
        } else {
            None
        }
    }

    /// Get as i32 (type-checked)
    pub fn as_i32(&self) -> Option<i32> {
        if self.tag == TypeTag::I32 {
            unsafe { self.as_ref::<i32>().copied() }
        } else {
            None
        }
    }

    /// Get as i64 (type-checked)
    pub fn as_i64(&self) -> Option<i64> {
        if self.tag == TypeTag::I64 {
            unsafe { self.as_ref::<i64>().copied() }
        } else {
            None
        }
    }

    /// Get as u8 (type-checked)
    pub fn as_u8(&self) -> Option<u8> {
        if self.tag == TypeTag::U8 {
            unsafe { self.as_ref::<u8>().copied() }
        } else {
            None
        }
    }

    /// Get as u16 (type-checked)
    pub fn as_u16(&self) -> Option<u16> {
        if self.tag == TypeTag::U16 {
            unsafe { self.as_ref::<u16>().copied() }
        } else {
            None
        }
    }

    /// Get as u32 (type-checked)
    pub fn as_u32(&self) -> Option<u32> {
        if self.tag == TypeTag::U32 {
            unsafe { self.as_ref::<u32>().copied() }
        } else {
            None
        }
    }

    /// Get as u64 (type-checked)
    pub fn as_u64(&self) -> Option<u64> {
        if self.tag == TypeTag::U64 {
            unsafe { self.as_ref::<u64>().copied() }
        } else {
            None
        }
    }

    /// Get as f32 (type-checked)
    pub fn as_f32(&self) -> Option<f32> {
        if self.tag == TypeTag::F32 {
            unsafe { self.as_ref::<f32>().copied() }
        } else {
            None
        }
    }

    /// Get as f64 (type-checked)
    pub fn as_f64(&self) -> Option<f64> {
        if self.tag == TypeTag::F64 {
            unsafe { self.as_ref::<f64>().copied() }
        } else {
            None
        }
    }

    /// Get as bool (type-checked)
    pub fn as_bool(&self) -> Option<bool> {
        if self.tag == TypeTag::BOOL {
            unsafe { self.as_ref::<u8>().map(|&v| v != 0) }
        } else {
            None
        }
    }

    // ========================================================================
    // Box creation for stack values (borrowed, no dropper)
    // ========================================================================

    /// Create a box for an i8 value (stack-allocated, borrowed)
    pub fn box_i8(value: &mut i8) -> Self {
        Self {
            tag: TypeTag::I8,
            size: std::mem::size_of::<i8>() as u32,
            data: value as *mut i8 as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    /// Create a box for an i16 value (stack-allocated, borrowed)
    pub fn box_i16(value: &mut i16) -> Self {
        Self {
            tag: TypeTag::I16,
            size: std::mem::size_of::<i16>() as u32,
            data: value as *mut i16 as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    /// Create a box for an i32 value (stack-allocated, borrowed)
    pub fn box_i32(value: &mut i32) -> Self {
        Self {
            tag: TypeTag::I32,
            size: std::mem::size_of::<i32>() as u32,
            data: value as *mut i32 as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    /// Create a box for an i64 value (stack-allocated, borrowed)
    pub fn box_i64(value: &mut i64) -> Self {
        Self {
            tag: TypeTag::I64,
            size: std::mem::size_of::<i64>() as u32,
            data: value as *mut i64 as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    /// Create a box for a u8 value (stack-allocated, borrowed)
    pub fn box_u8(value: &mut u8) -> Self {
        Self {
            tag: TypeTag::U8,
            size: std::mem::size_of::<u8>() as u32,
            data: value as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    /// Create a box for a u16 value (stack-allocated, borrowed)
    pub fn box_u16(value: &mut u16) -> Self {
        Self {
            tag: TypeTag::U16,
            size: std::mem::size_of::<u16>() as u32,
            data: value as *mut u16 as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    /// Create a box for a u32 value (stack-allocated, borrowed)
    pub fn box_u32(value: &mut u32) -> Self {
        Self {
            tag: TypeTag::U32,
            size: std::mem::size_of::<u32>() as u32,
            data: value as *mut u32 as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    /// Create a box for a u64 value (stack-allocated, borrowed)
    pub fn box_u64(value: &mut u64) -> Self {
        Self {
            tag: TypeTag::U64,
            size: std::mem::size_of::<u64>() as u32,
            data: value as *mut u64 as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    /// Create a box for an f32 value (stack-allocated, borrowed)
    pub fn box_f32(value: &mut f32) -> Self {
        Self {
            tag: TypeTag::F32,
            size: std::mem::size_of::<f32>() as u32,
            data: value as *mut f32 as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    /// Create a box for an f64 value (stack-allocated, borrowed)
    pub fn box_f64(value: &mut f64) -> Self {
        Self {
            tag: TypeTag::F64,
            size: std::mem::size_of::<f64>() as u32,
            data: value as *mut f64 as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    /// Create a box for a bool value (stack-allocated, borrowed)
    pub fn box_bool(value: &mut u8) -> Self {
        Self {
            tag: TypeTag::BOOL,
            size: std::mem::size_of::<u8>() as u32,
            data: value as *mut u8,
            dropper: None,
            display_fn: None,
        }
    }

    // ========================================================================
    // Heap-allocated box creation (owned, with dropper)
    // ========================================================================

    /// Allocate a new box with the given tag and size
    pub fn alloc(tag: TypeTag, size: u32) -> Self {
        let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
        let data = unsafe { std::alloc::alloc(layout) };

        Self {
            tag,
            size,
            data,
            dropper: Some(default_dropper),
            display_fn: None,
        }
    }

    /// Create an owned box from a value (moves value to heap)
    pub fn from_value<T: Sized>(tag: TypeTag, value: T) -> Self {
        let boxed = Box::new(value);
        let ptr = Box::into_raw(boxed);

        Self {
            tag,
            size: std::mem::size_of::<T>() as u32,
            data: ptr as *mut u8,
            dropper: Some(drop_box::<T>),
            display_fn: None,
        }
    }

    /// Create an owned i32 box
    pub fn owned_i32(value: i32) -> Self {
        Self::from_value(TypeTag::I32, value)
    }

    /// Create an owned i64 box
    pub fn owned_i64(value: i64) -> Self {
        Self::from_value(TypeTag::I64, value)
    }

    /// Create an owned f32 box
    pub fn owned_f32(value: f32) -> Self {
        Self::from_value(TypeTag::F32, value)
    }

    /// Create an owned f64 box
    pub fn owned_f64(value: f64) -> Self {
        Self::from_value(TypeTag::F64, value)
    }

    /// Create an owned bool box
    pub fn owned_bool(value: bool) -> Self {
        Self::from_value(TypeTag::BOOL, value as u8)
    }

    /// Clone the box (deep copy)
    pub fn clone_box(&self) -> Self {
        if self.is_null() || self.size == 0 {
            return Self::null();
        }

        let mut cloned = Self::alloc(self.tag, self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(self.data, cloned.data, self.size as usize);
        }
        cloned
    }
}

impl Default for DynamicBox {
    fn default() -> Self {
        Self::null()
    }
}

// Default dropper uses std::alloc::dealloc
extern "C" fn default_dropper(ptr: *mut u8) {
    // We can't know the exact layout, so we just leak it
    // Real usage should use typed droppers
    let _ = ptr;
}

// Typed dropper for Box<T>
extern "C" fn drop_box<T>(ptr: *mut u8) {
    unsafe {
        let _ = Box::from_raw(ptr as *mut T);
    }
}

// SAFETY: DynamicBox is carefully managed with explicit ownership
unsafe impl Send for DynamicBox {}
unsafe impl Sync for DynamicBox {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_box() {
        let b = DynamicBox::null();
        assert!(b.is_null());
        assert_eq!(b.size, 0);
    }

    #[test]
    fn test_stack_box() {
        let mut value: i32 = 42;
        let b = DynamicBox::box_i32(&mut value);

        assert!(!b.is_null());
        assert_eq!(b.tag, TypeTag::I32);
        assert_eq!(b.as_i32(), Some(42));

        // Modify through the box
        value = 100;
        assert_eq!(b.as_i32(), Some(100));
    }

    #[test]
    fn test_owned_box() {
        let mut b = DynamicBox::owned_i32(42);

        assert!(!b.is_null());
        assert_eq!(b.as_i32(), Some(42));

        // Free should work
        b.free();
        assert!(b.is_null());
    }

    #[test]
    fn test_clone_box() {
        let original = DynamicBox::owned_i64(123456789);
        let mut cloned = original.clone_box();

        assert_eq!(cloned.as_i64(), Some(123456789));

        // Cloned is independent
        cloned.free();
        assert_eq!(original.as_i64(), Some(123456789));
    }

    #[test]
    fn test_type_mismatch() {
        let b = DynamicBox::owned_i32(42);

        // Wrong type returns None
        assert!(b.as_i64().is_none());
        assert!(b.as_f32().is_none());
        assert!(b.as_bool().is_none());

        // Correct type returns Some
        assert_eq!(b.as_i32(), Some(42));
    }
}
