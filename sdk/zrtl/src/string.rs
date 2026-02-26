//! ZrtlString - Inline length-prefixed string format
//!
//! Zyntax strings use an inline length-prefixed format:
//! ```text
//! Memory layout: [i32 length][utf8_bytes...]
//! ```
//!
//! This module provides helpers for working with this format from Rust.

use std::ptr::NonNull;

/// String pointer type - points to the length header
///
/// This is equivalent to `ZrtlStringPtr` in the C SDK.
pub type StringPtr = *mut i32;
pub type StringConstPtr = *const i32;

/// Header size in bytes (just the i32 length)
pub const STRING_HEADER_SIZE: usize = std::mem::size_of::<i32>();

/// Get string length from a string pointer
///
/// # Safety
/// The pointer must be valid and point to a valid string header.
#[inline]
pub unsafe fn string_length(ptr: StringConstPtr) -> i32 {
    if ptr.is_null() {
        0
    } else {
        *ptr
    }
}

/// Get pointer to UTF-8 bytes from a string pointer
///
/// # Safety
/// The pointer must be valid and point to a valid string header.
#[inline]
pub unsafe fn string_data(ptr: StringConstPtr) -> *const u8 {
    if ptr.is_null() {
        std::ptr::null()
    } else {
        (ptr as *const u8).add(STRING_HEADER_SIZE)
    }
}

/// Get mutable pointer to UTF-8 bytes
///
/// # Safety
/// The pointer must be valid and point to a valid string header.
#[inline]
pub unsafe fn string_data_mut(ptr: StringPtr) -> *mut u8 {
    if ptr.is_null() {
        std::ptr::null_mut()
    } else {
        (ptr as *mut u8).add(STRING_HEADER_SIZE)
    }
}

/// Calculate total allocation size for a string of given byte length
#[inline]
pub const fn string_alloc_size(byte_len: usize) -> usize {
    STRING_HEADER_SIZE + byte_len
}

/// Create a new string from a Rust &str
///
/// Returns a pointer that must be freed with `string_free`.
pub fn string_new(s: &str) -> StringPtr {
    let len = s.len() as i32;
    let total_size = string_alloc_size(s.len());

    unsafe {
        let layout = std::alloc::Layout::from_size_align(total_size, 4).unwrap();
        let ptr = std::alloc::alloc(layout) as StringPtr;

        if ptr.is_null() {
            return std::ptr::null_mut();
        }

        // Write length
        *ptr = len;

        // Write string bytes
        if !s.is_empty() {
            std::ptr::copy_nonoverlapping(s.as_ptr(), string_data_mut(ptr), s.len());
        }

        ptr
    }
}

/// Create a new string from bytes with known length
pub fn string_from_bytes(bytes: &[u8]) -> StringPtr {
    let len = bytes.len() as i32;
    let total_size = string_alloc_size(bytes.len());

    unsafe {
        let layout = std::alloc::Layout::from_size_align(total_size, 4).unwrap();
        let ptr = std::alloc::alloc(layout) as StringPtr;

        if ptr.is_null() {
            return std::ptr::null_mut();
        }

        *ptr = len;

        if !bytes.is_empty() {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), string_data_mut(ptr), bytes.len());
        }

        ptr
    }
}

/// Create an empty string
pub fn string_empty() -> StringPtr {
    unsafe {
        let layout = std::alloc::Layout::from_size_align(STRING_HEADER_SIZE, 4).unwrap();
        let ptr = std::alloc::alloc(layout) as StringPtr;
        if !ptr.is_null() {
            *ptr = 0;
        }
        ptr
    }
}

/// Free a string
///
/// # Safety
/// The pointer must have been created by one of the string_new functions.
pub unsafe fn string_free(ptr: StringPtr) {
    if !ptr.is_null() {
        let len = *ptr as usize;
        let total_size = string_alloc_size(len);
        let layout = std::alloc::Layout::from_size_align_unchecked(total_size, 4);
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

/// Copy a string
///
/// # Safety
/// The source pointer must be valid.
pub unsafe fn string_copy(src: StringConstPtr) -> StringPtr {
    if src.is_null() {
        return string_empty();
    }

    let len = string_length(src) as usize;
    let total_size = string_alloc_size(len);

    let layout = std::alloc::Layout::from_size_align(total_size, 4).unwrap();
    let dst = std::alloc::alloc(layout) as StringPtr;

    if !dst.is_null() {
        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, total_size);
    }

    dst
}

/// Compare two strings for equality
///
/// # Safety
/// Both pointers must be valid or null.
pub unsafe fn string_equals(a: StringConstPtr, b: StringConstPtr) -> bool {
    if a == b {
        return true;
    }
    if a.is_null() || b.is_null() {
        return false;
    }

    let len_a = string_length(a);
    let len_b = string_length(b);

    if len_a != len_b {
        return false;
    }

    if len_a == 0 {
        return true;
    }

    let data_a = std::slice::from_raw_parts(string_data(a), len_a as usize);
    let data_b = std::slice::from_raw_parts(string_data(b), len_b as usize);

    data_a == data_b
}

/// Get string as a Rust &str
///
/// # Safety
/// The pointer must be valid and contain valid UTF-8.
pub unsafe fn string_as_str<'a>(ptr: StringConstPtr) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }

    let len = string_length(ptr) as usize;
    if len == 0 {
        return Some("");
    }

    let bytes = std::slice::from_raw_parts(string_data(ptr), len);
    std::str::from_utf8(bytes).ok()
}

/// Get string as bytes slice
///
/// # Safety
/// The pointer must be valid.
pub unsafe fn string_as_bytes<'a>(ptr: StringConstPtr) -> &'a [u8] {
    if ptr.is_null() {
        return &[];
    }

    let len = string_length(ptr) as usize;
    if len == 0 {
        return &[];
    }

    std::slice::from_raw_parts(string_data(ptr), len)
}

/// Non-owning string view (for SDK convenience)
///
/// This does NOT use the inline format - it's just a reference helper.
#[derive(Debug, Clone, Copy)]
pub struct StringView<'a> {
    pub data: &'a str,
}

impl<'a> StringView<'a> {
    /// Create a view from a string pointer
    ///
    /// # Safety
    /// The pointer must be valid and contain valid UTF-8.
    pub unsafe fn from_ptr(ptr: StringConstPtr) -> Option<Self> {
        string_as_str(ptr).map(|data| Self { data })
    }

    /// Get the length in bytes
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> AsRef<str> for StringView<'a> {
    fn as_ref(&self) -> &str {
        self.data
    }
}

impl<'a> std::fmt::Display for StringView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

/// Owned string wrapper that manages memory
pub struct OwnedString {
    ptr: NonNull<i32>,
}

impl OwnedString {
    /// Create from a Rust string
    pub fn new(s: &str) -> Option<Self> {
        let ptr = string_new(s);
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }

    /// Create empty
    pub fn empty() -> Option<Self> {
        let ptr = string_empty();
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> StringConstPtr {
        self.ptr.as_ptr()
    }

    /// Get the mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> StringPtr {
        self.ptr.as_ptr()
    }

    /// Get the length
    pub fn len(&self) -> usize {
        unsafe { string_length(self.as_ptr()) as usize }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get as str
    pub fn as_str(&self) -> Option<&str> {
        unsafe { string_as_str(self.as_ptr()) }
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { string_as_bytes(self.as_ptr()) }
    }

    /// Release ownership (caller must free)
    pub fn into_raw(self) -> StringPtr {
        let ptr = self.ptr.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Take ownership from raw pointer
    ///
    /// # Safety
    /// The pointer must have been created by string_new or similar.
    pub unsafe fn from_raw(ptr: StringPtr) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }
}

impl Drop for OwnedString {
    fn drop(&mut self) {
        unsafe {
            string_free(self.ptr.as_ptr());
        }
    }
}

impl Clone for OwnedString {
    fn clone(&self) -> Self {
        unsafe {
            let ptr = string_copy(self.as_ptr());
            Self {
                ptr: NonNull::new(ptr).expect("string copy failed"),
            }
        }
    }
}

impl std::fmt::Debug for OwnedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_str() {
            Some(s) => write!(f, "OwnedString({:?})", s),
            None => write!(f, "OwnedString(<invalid UTF-8>)"),
        }
    }
}

impl std::fmt::Display for OwnedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_str() {
            Some(s) => write!(f, "{}", s),
            None => write!(f, "<invalid UTF-8>"),
        }
    }
}

impl PartialEq for OwnedString {
    fn eq(&self, other: &Self) -> bool {
        unsafe { string_equals(self.as_ptr(), other.as_ptr()) }
    }
}

impl Eq for OwnedString {}

impl From<&str> for OwnedString {
    fn from(s: &str) -> Self {
        Self::new(s).expect("string allocation failed")
    }
}

impl From<String> for OwnedString {
    fn from(s: String) -> Self {
        Self::new(&s).expect("string allocation failed")
    }
}

// SAFETY: OwnedString owns its memory exclusively
unsafe impl Send for OwnedString {}
unsafe impl Sync for OwnedString {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_new() {
        let s = OwnedString::new("Hello, ZRTL!").unwrap();
        assert_eq!(s.len(), 12);
        assert_eq!(s.as_str(), Some("Hello, ZRTL!"));
    }

    #[test]
    fn test_string_empty() {
        let s = OwnedString::empty().unwrap();
        assert!(s.is_empty());
        assert_eq!(s.as_str(), Some(""));
    }

    #[test]
    fn test_string_clone() {
        let s1 = OwnedString::from("test string");
        let s2 = s1.clone();

        assert_eq!(s1, s2);
        assert_ne!(s1.as_ptr(), s2.as_ptr()); // Different allocations
    }

    #[test]
    fn test_string_equals() {
        let s1 = OwnedString::from("equal");
        let s2 = OwnedString::from("equal");
        let s3 = OwnedString::from("different");

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_raw_functions() {
        unsafe {
            let ptr = string_new("raw test");
            assert_eq!(string_length(ptr), 8);

            let view = string_as_str(ptr).unwrap();
            assert_eq!(view, "raw test");

            string_free(ptr);
        }
    }
}
