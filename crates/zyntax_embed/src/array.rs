//! ZyntaxArray - Direct interop with Zyntax's array format
//!
//! Zyntax uses a header-prefixed array format:
//! ```text
//! Memory layout: [i32 capacity][i32 length][elements...]
//! ```
//!
//! This module provides `ZyntaxArray` for direct manipulation of these arrays
//! without intermediate conversion to Rust `Vec`.

use std::marker::PhantomData;
use std::ptr::NonNull;

/// Header size for Zyntax arrays (capacity + length)
pub const ARRAY_HEADER_SIZE: usize = std::mem::size_of::<i32>() * 2;

/// A typed array in Zyntax's native format.
///
/// This is a wrapper around a pointer to Zyntax's array format:
/// `[i32 capacity][i32 length][elements...]`
///
/// # Type Parameter
///
/// `T` is the element type. It must be `Copy` and have a known size to ensure
/// safe memory access.
///
/// # Memory Ownership
///
/// `ZyntaxArray` can either own its memory (will free on drop) or borrow
/// from Zyntax runtime (must not be freed by Rust).
pub struct ZyntaxArray<T: Copy> {
    /// Pointer to the capacity field
    ptr: NonNull<i32>,
    /// Whether we own this memory
    owned: bool,
    /// Phantom data for element type
    _marker: PhantomData<T>,
}

impl<T: Copy> ZyntaxArray<T> {
    /// Create a new empty array with the given capacity
    ///
    /// # Panics
    /// Panics if the allocation size overflows or if allocation fails.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1).min(i32::MAX as usize) as i32;
        let elem_bytes = (capacity as usize)
            .checked_mul(std::mem::size_of::<T>())
            .expect("ZyntaxArray: capacity * element size overflows");
        let total_size = ARRAY_HEADER_SIZE
            .checked_add(elem_bytes)
            .expect("ZyntaxArray: total allocation size overflows");

        unsafe {
            let layout = std::alloc::Layout::from_size_align(total_size, 4).unwrap();
            let ptr = std::alloc::alloc(layout) as *mut i32;

            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            // Write header
            *ptr = capacity; // capacity
            *ptr.add(1) = 0; // length

            Self {
                ptr: NonNull::new_unchecked(ptr),
                owned: true,
                _marker: PhantomData,
            }
        }
    }

    /// Create a new empty array with default capacity
    pub fn new() -> Self {
        Self::with_capacity(8)
    }

    /// Create an array from a slice (copies the data)
    pub fn from_slice(slice: &[T]) -> Self {
        let mut arr = Self::with_capacity(slice.len().max(8));
        for &item in slice {
            arr.push(item);
        }
        arr
    }

    /// Create an array from a Vec (copies the data)
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self::from_slice(&vec)
    }

    /// Wrap an existing Zyntax array pointer (borrowed, not owned)
    ///
    /// # Safety
    /// - The pointer must be valid and point to a valid Zyntax array
    /// - The element type T must match the actual array element type
    /// - The memory must remain valid for the lifetime of this ZyntaxArray
    pub unsafe fn from_ptr(ptr: *const i32) -> Option<Self> {
        NonNull::new(ptr as *mut i32).map(|ptr| Self {
            ptr,
            owned: false,
            _marker: PhantomData,
        })
    }

    /// Wrap an existing Zyntax array pointer (takes ownership)
    ///
    /// # Safety
    /// - The pointer must be valid and point to a valid Zyntax array
    /// - The element type T must match the actual array element type
    /// - Ownership is transferred to this ZyntaxArray
    pub unsafe fn from_ptr_owned(ptr: *mut i32) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self {
            ptr,
            owned: true,
            _marker: PhantomData,
        })
    }

    /// Get the raw pointer (for passing to Zyntax functions)
    pub fn as_ptr(&self) -> *const i32 {
        self.ptr.as_ptr()
    }

    /// Get the raw mutable pointer
    pub fn as_mut_ptr(&mut self) -> *mut i32 {
        self.ptr.as_ptr()
    }

    /// Get the capacity (number of elements that can be stored without reallocation)
    pub fn capacity(&self) -> usize {
        unsafe { *self.ptr.as_ptr() as usize }
    }

    /// Get the length (number of elements currently stored)
    pub fn len(&self) -> usize {
        unsafe { *self.ptr.as_ptr().add(1) as usize }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the element data as a slice
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            let len = self.len();
            if len == 0 {
                return &[];
            }
            let data_ptr = (self.ptr.as_ptr() as *const u8).add(ARRAY_HEADER_SIZE) as *const T;
            std::slice::from_raw_parts(data_ptr, len)
        }
    }

    /// Get the element data as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            let len = self.len();
            if len == 0 {
                return &mut [];
            }
            let data_ptr = (self.ptr.as_ptr() as *mut u8).add(ARRAY_HEADER_SIZE) as *mut T;
            std::slice::from_raw_parts_mut(data_ptr, len)
        }
    }

    /// Get element at index (returns None if out of bounds)
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            Some(&self.as_slice()[index])
        } else {
            None
        }
    }

    /// Get mutable element at index (returns None if out of bounds)
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            Some(&mut self.as_mut_slice()[index])
        } else {
            None
        }
    }

    /// Push an element (may reallocate)
    ///
    /// # Panics
    /// Panics if the array is not owned (cannot reallocate borrowed memory)
    pub fn push(&mut self, value: T) {
        if !self.owned {
            panic!("Cannot push to a borrowed ZyntaxArray");
        }

        let len = self.len();
        let cap = self.capacity();

        if len >= cap {
            self.grow();
        }

        unsafe {
            let data_ptr = (self.ptr.as_ptr() as *mut u8).add(ARRAY_HEADER_SIZE) as *mut T;
            *data_ptr.add(len) = value;
            *self.ptr.as_ptr().add(1) = (len + 1) as i32;
        }
    }

    /// Pop an element from the end
    pub fn pop(&mut self) -> Option<T> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        unsafe {
            let new_len = len - 1;
            let data_ptr = (self.ptr.as_ptr() as *const u8).add(ARRAY_HEADER_SIZE) as *const T;
            let value = *data_ptr.add(new_len);
            *self.ptr.as_ptr().add(1) = new_len as i32;
            Some(value)
        }
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        unsafe {
            *self.ptr.as_ptr().add(1) = 0;
        }
    }

    /// Convert to a Vec (copies the data)
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    /// Release ownership of the memory
    pub fn into_raw(mut self) -> *mut i32 {
        let ptr = self.ptr.as_ptr();
        self.owned = false;
        std::mem::forget(self);
        ptr
    }

    /// Grow the array capacity (doubles it)
    fn grow(&mut self) {
        let old_cap = self.capacity();
        let new_cap = (old_cap * 2).max(8).min(i32::MAX as usize);
        let len = self.len();

        let old_size = ARRAY_HEADER_SIZE + old_cap * std::mem::size_of::<T>();
        let new_elem_bytes = new_cap
            .checked_mul(std::mem::size_of::<T>())
            .expect("ZyntaxArray grow: capacity * element size overflows");
        let new_size = ARRAY_HEADER_SIZE
            .checked_add(new_elem_bytes)
            .expect("ZyntaxArray grow: total allocation size overflows");

        unsafe {
            let old_layout = std::alloc::Layout::from_size_align_unchecked(old_size, 4);
            let new_ptr =
                std::alloc::realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_size) as *mut i32;

            if new_ptr.is_null() {
                let new_layout = std::alloc::Layout::from_size_align_unchecked(new_size, 4);
                std::alloc::handle_alloc_error(new_layout);
            }

            *new_ptr = new_cap as i32;
            *new_ptr.add(1) = len as i32;
            self.ptr = NonNull::new_unchecked(new_ptr);
        }
    }

    /// Get the total allocation size
    pub fn allocation_size(&self) -> usize {
        ARRAY_HEADER_SIZE + self.capacity() * std::mem::size_of::<T>()
    }
}

impl<T: Copy> Default for ZyntaxArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> Drop for ZyntaxArray<T> {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                let size = self.allocation_size();
                let layout = std::alloc::Layout::from_size_align_unchecked(size, 4);
                std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T: Copy> Clone for ZyntaxArray<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for ZyntaxArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZyntaxArray")
            .field("capacity", &self.capacity())
            .field("length", &self.len())
            .field("elements", &self.as_slice())
            .finish()
    }
}

impl<T: Copy + PartialEq> PartialEq for ZyntaxArray<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Eq> Eq for ZyntaxArray<T> {}

impl<T: Copy> std::ops::Index<usize> for ZyntaxArray<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("ZyntaxArray index out of bounds")
    }
}

impl<T: Copy> std::ops::IndexMut<usize> for ZyntaxArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
            .expect("ZyntaxArray index out of bounds")
    }
}

impl<T: Copy> From<Vec<T>> for ZyntaxArray<T> {
    fn from(vec: Vec<T>) -> Self {
        Self::from_vec(vec)
    }
}

impl<T: Copy> From<&[T]> for ZyntaxArray<T> {
    fn from(slice: &[T]) -> Self {
        Self::from_slice(slice)
    }
}

impl<T: Copy, const N: usize> From<[T; N]> for ZyntaxArray<T> {
    fn from(arr: [T; N]) -> Self {
        Self::from_slice(&arr)
    }
}

impl<T: Copy> From<ZyntaxArray<T>> for Vec<T> {
    fn from(arr: ZyntaxArray<T>) -> Self {
        arr.to_vec()
    }
}

impl<T: Copy> IntoIterator for ZyntaxArray<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_vec().into_iter()
    }
}

impl<'a, T: Copy> IntoIterator for &'a ZyntaxArray<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T: Copy> IntoIterator for &'a mut ZyntaxArray<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

// Safety: ZyntaxArray's data is just bytes, safe to send across threads
unsafe impl<T: Copy + Send> Send for ZyntaxArray<T> {}
unsafe impl<T: Copy + Sync> Sync for ZyntaxArray<T> {}

impl<T: Copy> Extend<T> for ZyntaxArray<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T: Copy> std::iter::FromIterator<T> for ZyntaxArray<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower).max(8);
        let mut arr = Self::with_capacity(capacity);
        arr.extend(iter);
        arr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_push() {
        let mut arr: ZyntaxArray<i32> = ZyntaxArray::new();
        arr.push(1);
        arr.push(2);
        arr.push(3);

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_from_slice() {
        let arr: ZyntaxArray<i32> = ZyntaxArray::from_slice(&[10, 20, 30]);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], 10);
        assert_eq!(arr[1], 20);
        assert_eq!(arr[2], 30);
    }

    #[test]
    fn test_pop() {
        let mut arr: ZyntaxArray<i32> = [1, 2, 3].into();
        assert_eq!(arr.pop(), Some(3));
        assert_eq!(arr.pop(), Some(2));
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_grow() {
        let mut arr: ZyntaxArray<i32> = ZyntaxArray::with_capacity(2);
        for i in 0..100 {
            arr.push(i);
        }
        assert_eq!(arr.len(), 100);
        assert!(arr.capacity() >= 100);

        for i in 0..100 {
            assert_eq!(arr[i as usize], i);
        }
    }

    #[test]
    fn test_clone() {
        let arr1: ZyntaxArray<i32> = [1, 2, 3].into();
        let arr2 = arr1.clone();

        assert_eq!(arr1, arr2);
        assert_ne!(arr1.as_ptr(), arr2.as_ptr());
    }

    #[test]
    fn test_to_vec() {
        let arr: ZyntaxArray<i32> = [5, 10, 15].into();
        let vec = arr.to_vec();
        assert_eq!(vec, vec![5, 10, 15]);
    }

    #[test]
    fn test_iteration() {
        let arr: ZyntaxArray<i32> = [1, 2, 3].into();

        let sum: i32 = arr.as_slice().iter().sum();
        assert_eq!(sum, 6);

        let doubled: Vec<i32> = arr.into_iter().map(|x| x * 2).collect();
        assert_eq!(doubled, vec![2, 4, 6]);
    }

    #[test]
    fn test_index_mut() {
        let mut arr: ZyntaxArray<i32> = [1, 2, 3].into();
        arr[1] = 42;
        assert_eq!(arr.as_slice(), &[1, 42, 3]);
    }

    #[test]
    fn test_extend() {
        let mut arr: ZyntaxArray<i32> = [1, 2].into();
        arr.extend([3, 4, 5]);
        assert_eq!(arr.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_from_iterator() {
        let arr: ZyntaxArray<i32> = (0..5).collect();
        assert_eq!(arr.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_collect_after_map() {
        let arr: ZyntaxArray<i32> = [1, 2, 3].into();
        let doubled: ZyntaxArray<i32> = arr.as_slice().iter().map(|x| x * 2).collect();
        assert_eq!(doubled.as_slice(), &[2, 4, 6]);
    }
}
