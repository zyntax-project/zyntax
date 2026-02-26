//! ZrtlArray - Inline header array format
//!
//! Zyntax arrays use an inline header format:
//! ```text
//! Memory layout: [i32 capacity][i32 length][elements...]
//! ```
//!
//! This module provides helpers for working with this format from Rust.

use std::marker::PhantomData;

/// Array pointer type - points to the capacity header
pub type ArrayPtr = *mut i32;
pub type ArrayConstPtr = *const i32;

/// Header size in i32 units (capacity + length)
pub const ARRAY_HEADER_SIZE: usize = 2;

/// Header size in bytes
pub const ARRAY_HEADER_BYTES: usize = ARRAY_HEADER_SIZE * std::mem::size_of::<i32>();

/// Get array capacity from pointer
///
/// # Safety
/// The pointer must be valid.
#[inline]
pub unsafe fn array_capacity(ptr: ArrayConstPtr) -> i32 {
    if ptr.is_null() {
        0
    } else {
        *ptr
    }
}

/// Get array length from pointer
///
/// # Safety
/// The pointer must be valid.
#[inline]
pub unsafe fn array_length(ptr: ArrayConstPtr) -> i32 {
    if ptr.is_null() {
        0
    } else {
        *ptr.add(1)
    }
}

/// Set array length
///
/// # Safety
/// The pointer must be valid and the new length must be <= capacity.
#[inline]
pub unsafe fn set_array_length(ptr: ArrayPtr, len: i32) {
    if !ptr.is_null() {
        *ptr.add(1) = len;
    }
}

/// Get pointer to array elements
///
/// # Safety
/// The pointer must be valid.
#[inline]
pub unsafe fn array_data<T>(ptr: ArrayConstPtr) -> *const T {
    if ptr.is_null() {
        std::ptr::null()
    } else {
        ptr.add(ARRAY_HEADER_SIZE) as *const T
    }
}

/// Get mutable pointer to array elements
///
/// # Safety
/// The pointer must be valid.
#[inline]
pub unsafe fn array_data_mut<T>(ptr: ArrayPtr) -> *mut T {
    if ptr.is_null() {
        std::ptr::null_mut()
    } else {
        ptr.add(ARRAY_HEADER_SIZE) as *mut T
    }
}

/// Calculate allocation size for array
#[inline]
pub const fn array_alloc_size(capacity: usize, elem_size: usize) -> usize {
    ARRAY_HEADER_BYTES + capacity * elem_size
}

/// Create a new array with given capacity
///
/// Returns a pointer that must be freed with `array_free`.
pub fn array_new<T>(initial_capacity: usize) -> ArrayPtr {
    let cap = if initial_capacity > 0 {
        initial_capacity
    } else {
        8
    };
    let size = array_alloc_size(cap, std::mem::size_of::<T>());

    unsafe {
        let layout =
            std::alloc::Layout::from_size_align(size, std::mem::align_of::<T>().max(4)).unwrap();
        let ptr = std::alloc::alloc(layout) as ArrayPtr;

        if !ptr.is_null() {
            *ptr = cap as i32; // capacity
            *ptr.add(1) = 0; // length
        }

        ptr
    }
}

/// Free an array
///
/// # Safety
/// The pointer must have been created by array_new.
pub unsafe fn array_free<T>(ptr: ArrayPtr) {
    if !ptr.is_null() {
        let cap = array_capacity(ptr) as usize;
        let size = array_alloc_size(cap, std::mem::size_of::<T>());
        let layout =
            std::alloc::Layout::from_size_align_unchecked(size, std::mem::align_of::<T>().max(4));
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

/// Push an element to the array
///
/// Returns the new array pointer (may reallocate).
///
/// # Safety
/// The pointer must be valid.
pub unsafe fn array_push<T: Copy>(ptr: ArrayPtr, value: T) -> ArrayPtr {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let cap = array_capacity(ptr) as usize;
    let len = array_length(ptr) as usize;

    // Check if we need to grow
    if len >= cap {
        let new_cap = cap * 2;
        let new_size = array_alloc_size(new_cap, std::mem::size_of::<T>());
        let old_size = array_alloc_size(cap, std::mem::size_of::<T>());

        let layout = std::alloc::Layout::from_size_align_unchecked(
            old_size,
            std::mem::align_of::<T>().max(4),
        );

        let new_ptr = std::alloc::realloc(ptr as *mut u8, layout, new_size) as ArrayPtr;
        if new_ptr.is_null() {
            return std::ptr::null_mut();
        }

        *new_ptr = new_cap as i32;

        // Add element
        let data = array_data_mut::<T>(new_ptr);
        *data.add(len) = value;
        *new_ptr.add(1) = (len + 1) as i32;

        new_ptr
    } else {
        // Add element
        let data = array_data_mut::<T>(ptr);
        *data.add(len) = value;
        *ptr.add(1) = (len + 1) as i32;

        ptr
    }
}

/// Get element from array
///
/// # Safety
/// The pointer must be valid.
pub unsafe fn array_get<T: Copy>(ptr: ArrayConstPtr, index: usize) -> Option<T> {
    if ptr.is_null() {
        return None;
    }

    let len = array_length(ptr) as usize;
    if index >= len {
        return None;
    }

    let data = array_data::<T>(ptr);
    Some(*data.add(index))
}

/// Set element in array
///
/// # Safety
/// The pointer must be valid.
pub unsafe fn array_set<T: Copy>(ptr: ArrayPtr, index: usize, value: T) -> bool {
    if ptr.is_null() {
        return false;
    }

    let len = array_length(ptr) as usize;
    if index >= len {
        return false;
    }

    let data = array_data_mut::<T>(ptr);
    *data.add(index) = value;
    true
}

/// Get array as slice
///
/// # Safety
/// The pointer must be valid.
pub unsafe fn array_as_slice<'a, T>(ptr: ArrayConstPtr) -> &'a [T] {
    if ptr.is_null() {
        return &[];
    }

    let len = array_length(ptr) as usize;
    if len == 0 {
        return &[];
    }

    std::slice::from_raw_parts(array_data::<T>(ptr), len)
}

/// Get array as mutable slice
///
/// # Safety
/// The pointer must be valid.
pub unsafe fn array_as_slice_mut<'a, T>(ptr: ArrayPtr) -> &'a mut [T] {
    if ptr.is_null() {
        return &mut [];
    }

    let len = array_length(ptr) as usize;
    if len == 0 {
        return &mut [];
    }

    std::slice::from_raw_parts_mut(array_data_mut::<T>(ptr), len)
}

/// Array iterator
pub struct ArrayIterator<T> {
    ptr: ArrayConstPtr,
    index: usize,
    _marker: PhantomData<T>,
}

impl<T> ArrayIterator<T> {
    /// Create an iterator for an array
    ///
    /// # Safety
    /// The pointer must be valid for the lifetime of the iterator.
    pub unsafe fn new(ptr: ArrayConstPtr) -> Self {
        Self {
            ptr,
            index: 0,
            _marker: PhantomData,
        }
    }

    /// Check if there are more elements
    pub fn has_next(&self) -> bool {
        if self.ptr.is_null() {
            return false;
        }
        unsafe { self.index < array_length(self.ptr) as usize }
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.index = 0;
    }
}

impl<T: Copy> Iterator for ArrayIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_next() {
            return None;
        }

        unsafe {
            let value = array_get::<T>(self.ptr, self.index);
            self.index += 1;
            value
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.ptr.is_null() {
            return (0, Some(0));
        }
        unsafe {
            let len = array_length(self.ptr) as usize;
            let remaining = len.saturating_sub(self.index);
            (remaining, Some(remaining))
        }
    }
}

impl<T: Copy> ExactSizeIterator for ArrayIterator<T> {}

/// Owned array wrapper that manages memory
pub struct OwnedArray<T> {
    ptr: ArrayPtr,
    _marker: PhantomData<T>,
}

impl<T: Copy> OwnedArray<T> {
    /// Create a new array with default capacity
    pub fn new() -> Option<Self> {
        let ptr = array_new::<T>(8);
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                _marker: PhantomData,
            })
        }
    }

    /// Create with specific capacity
    pub fn with_capacity(capacity: usize) -> Option<Self> {
        let ptr = array_new::<T>(capacity);
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                _marker: PhantomData,
            })
        }
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> ArrayConstPtr {
        self.ptr
    }

    /// Get the mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> ArrayPtr {
        self.ptr
    }

    /// Get the capacity
    pub fn capacity(&self) -> usize {
        unsafe { array_capacity(self.ptr) as usize }
    }

    /// Get the length
    pub fn len(&self) -> usize {
        unsafe { array_length(self.ptr) as usize }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push an element
    pub fn push(&mut self, value: T) -> bool {
        unsafe {
            let new_ptr = array_push(self.ptr, value);
            if new_ptr.is_null() {
                false
            } else {
                self.ptr = new_ptr;
                true
            }
        }
    }

    /// Get an element
    pub fn get(&self, index: usize) -> Option<T> {
        unsafe { array_get(self.ptr, index) }
    }

    /// Set an element
    pub fn set(&mut self, index: usize, value: T) -> bool {
        unsafe { array_set(self.ptr, index, value) }
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { array_as_slice(self.ptr) }
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { array_as_slice_mut(self.ptr) }
    }

    /// Create an iterator
    pub fn iter(&self) -> ArrayIterator<T> {
        unsafe { ArrayIterator::new(self.ptr) }
    }

    /// Release ownership
    pub fn into_raw(self) -> ArrayPtr {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }

    /// Take ownership from raw pointer
    ///
    /// # Safety
    /// The pointer must have been created by array_new.
    pub unsafe fn from_raw(ptr: ArrayPtr) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                _marker: PhantomData,
            })
        }
    }
}

impl<T: Copy> Default for OwnedArray<T> {
    fn default() -> Self {
        Self::new().expect("array allocation failed")
    }
}

impl<T> Drop for OwnedArray<T> {
    fn drop(&mut self) {
        unsafe {
            // Note: We use std::mem::size_of::<T>() which requires T to be Sized,
            // but OwnedArray<T> already has T: Sized implicitly
            let cap = array_capacity(self.ptr) as usize;
            let size = array_alloc_size(cap, std::mem::size_of::<T>());
            let layout = std::alloc::Layout::from_size_align_unchecked(
                size,
                std::mem::align_of::<T>().max(4),
            );
            std::alloc::dealloc(self.ptr as *mut u8, layout);
        }
    }
}

impl<T: Copy + Clone> Clone for OwnedArray<T> {
    fn clone(&self) -> Self {
        let mut new_arr = Self::with_capacity(self.len()).expect("clone allocation failed");
        for value in self.iter() {
            new_arr.push(value);
        }
        new_arr
    }
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for OwnedArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Copy + PartialEq> PartialEq for OwnedArray<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Eq> Eq for OwnedArray<T> {}

impl<T: Copy> FromIterator<T> for OwnedArray<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut arr = Self::with_capacity(lower.max(8)).expect("allocation failed");
        for item in iter {
            arr.push(item);
        }
        arr
    }
}

// SAFETY: OwnedArray owns its memory exclusively
unsafe impl<T: Send> Send for OwnedArray<T> {}
unsafe impl<T: Sync> Sync for OwnedArray<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_new() {
        let arr: OwnedArray<i32> = OwnedArray::new().unwrap();
        assert!(arr.is_empty());
        assert!(arr.capacity() >= 8);
    }

    #[test]
    fn test_array_push() {
        let mut arr: OwnedArray<i32> = OwnedArray::new().unwrap();
        arr.push(1);
        arr.push(2);
        arr.push(3);

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(1));
        assert_eq!(arr.get(1), Some(2));
        assert_eq!(arr.get(2), Some(3));
    }

    #[test]
    fn test_array_grow() {
        let mut arr: OwnedArray<i32> = OwnedArray::with_capacity(2).unwrap();

        for i in 0..100 {
            arr.push(i);
        }

        assert_eq!(arr.len(), 100);
        for i in 0..100 {
            assert_eq!(arr.get(i), Some(i as i32));
        }
    }

    #[test]
    fn test_array_iter() {
        let mut arr: OwnedArray<i32> = OwnedArray::new().unwrap();
        arr.push(10);
        arr.push(20);
        arr.push(30);

        let sum: i32 = arr.iter().sum();
        assert_eq!(sum, 60);
    }

    #[test]
    fn test_array_clone() {
        let mut arr1: OwnedArray<i32> = OwnedArray::new().unwrap();
        arr1.push(1);
        arr1.push(2);

        let arr2 = arr1.clone();

        assert_eq!(arr1, arr2);
        assert_ne!(arr1.as_ptr(), arr2.as_ptr());
    }

    #[test]
    fn test_from_iterator() {
        let arr: OwnedArray<i32> = (0..5).collect();
        assert_eq!(arr.len(), 5);
        assert_eq!(arr.as_slice(), &[0, 1, 2, 3, 4]);
    }
}
