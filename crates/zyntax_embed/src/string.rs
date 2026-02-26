//! ZyntaxString - Direct interop with Zyntax's length-prefixed string format
//!
//! Zyntax uses a length-prefixed string format:
//! ```text
//! Memory layout: [i32 length][utf8_bytes...]
//! ```
//!
//! This module provides `ZyntaxString` for direct manipulation of these strings
//! without intermediate conversion to Rust `String`.

use crate::error::{ConversionError, ConversionResult};
use std::ptr::NonNull;

/// A string in Zyntax's native length-prefixed format.
///
/// This is a wrapper around a pointer to Zyntax's string format:
/// `[i32 length][utf8_bytes...]`
///
/// # Memory Ownership
///
/// `ZyntaxString` can either own its memory (will free on drop) or borrow
/// from Zyntax runtime (must not be freed by Rust).
pub struct ZyntaxString {
    /// Pointer to the length field
    ptr: NonNull<i32>,
    /// Whether we own this memory (should free on drop)
    owned: bool,
}

impl ZyntaxString {
    /// Header size in bytes (just the i32 length)
    pub const HEADER_SIZE: usize = std::mem::size_of::<i32>();

    /// Create a new ZyntaxString from a Rust string
    pub fn from_str(s: &str) -> Self {
        let len = s.len() as i32;
        let total_size = Self::HEADER_SIZE + s.len();

        unsafe {
            let layout = std::alloc::Layout::from_size_align(total_size, 4).unwrap();
            let ptr = std::alloc::alloc(layout) as *mut i32;

            if ptr.is_null() {
                panic!("Failed to allocate ZyntaxString");
            }

            // Write length
            *ptr = len;

            // Write string bytes
            if !s.is_empty() {
                std::ptr::copy_nonoverlapping(
                    s.as_ptr(),
                    (ptr as *mut u8).add(Self::HEADER_SIZE),
                    s.len(),
                );
            }

            Self {
                ptr: NonNull::new_unchecked(ptr),
                owned: true,
            }
        }
    }

    /// Create an empty ZyntaxString
    pub fn empty() -> Self {
        Self::from_str("")
    }

    /// Wrap an existing Zyntax string pointer (borrowed, not owned)
    ///
    /// # Safety
    /// - The pointer must be valid and point to a valid Zyntax string
    /// - The memory must remain valid for the lifetime of this ZyntaxString
    /// - The caller retains ownership and must free the memory
    pub unsafe fn from_ptr(ptr: *const i32) -> Option<Self> {
        NonNull::new(ptr as *mut i32).map(|ptr| Self { ptr, owned: false })
    }

    /// Wrap an existing Zyntax string pointer (takes ownership)
    ///
    /// # Safety
    /// - The pointer must be valid and point to a valid Zyntax string
    /// - The memory must have been allocated with the same allocator
    /// - Ownership is transferred to this ZyntaxString
    pub unsafe fn from_ptr_owned(ptr: *mut i32) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self { ptr, owned: true })
    }

    /// Get the raw pointer (for passing to Zyntax functions)
    pub fn as_ptr(&self) -> *const i32 {
        self.ptr.as_ptr()
    }

    /// Get the raw mutable pointer
    pub fn as_mut_ptr(&mut self) -> *mut i32 {
        self.ptr.as_ptr()
    }

    /// Get the length in bytes
    pub fn len(&self) -> usize {
        unsafe { *self.ptr.as_ptr() as usize }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the string data as a byte slice
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            let len = self.len();
            if len == 0 {
                return &[];
            }
            let data_ptr = (self.ptr.as_ptr() as *const u8).add(Self::HEADER_SIZE);
            std::slice::from_raw_parts(data_ptr, len)
        }
    }

    /// Get the string as a str (returns error if not valid UTF-8)
    pub fn as_str(&self) -> ConversionResult<&str> {
        std::str::from_utf8(self.as_bytes()).map_err(ConversionError::from)
    }

    /// Convert to a Rust String (copies the data)
    pub fn to_string(&self) -> ConversionResult<String> {
        self.as_str().map(|s| s.to_owned())
    }

    /// Convert to a Rust String, consuming self
    ///
    /// If the string is owned, this avoids a copy by directly constructing
    /// a String from the bytes (after validation).
    pub fn into_string(self) -> ConversionResult<String> {
        // For now, just copy - a more optimized version could reuse memory
        self.to_string()
    }

    /// Release ownership of the memory (caller becomes responsible for freeing)
    ///
    /// Returns the raw pointer. After calling this, the ZyntaxString will not
    /// free the memory on drop.
    pub fn into_raw(mut self) -> *mut i32 {
        let ptr = self.ptr.as_ptr();
        self.owned = false;
        std::mem::forget(self);
        ptr
    }

    /// Get the total allocation size
    pub fn allocation_size(&self) -> usize {
        Self::HEADER_SIZE + self.len()
    }
}

impl Drop for ZyntaxString {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                let size = Self::HEADER_SIZE + self.len();
                let layout = std::alloc::Layout::from_size_align_unchecked(size, 4);
                std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl Clone for ZyntaxString {
    fn clone(&self) -> Self {
        // Always create an owned copy
        if let Ok(s) = self.as_str() {
            Self::from_str(s)
        } else {
            // For invalid UTF-8, copy the raw bytes
            let bytes = self.as_bytes();
            let total_size = Self::HEADER_SIZE + bytes.len();

            unsafe {
                let layout = std::alloc::Layout::from_size_align(total_size, 4).unwrap();
                let ptr = std::alloc::alloc(layout) as *mut i32;

                if ptr.is_null() {
                    panic!("Failed to allocate ZyntaxString clone");
                }

                *ptr = bytes.len() as i32;
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    (ptr as *mut u8).add(Self::HEADER_SIZE),
                    bytes.len(),
                );

                Self {
                    ptr: NonNull::new_unchecked(ptr),
                    owned: true,
                }
            }
        }
    }
}

impl std::fmt::Debug for ZyntaxString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_str() {
            Ok(s) => write!(f, "ZyntaxString({:?})", s),
            Err(_) => write!(f, "ZyntaxString(<invalid UTF-8, {} bytes>)", self.len()),
        }
    }
}

impl std::fmt::Display for ZyntaxString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_str() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<invalid UTF-8>"),
        }
    }
}

impl PartialEq for ZyntaxString {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for ZyntaxString {}

impl PartialEq<str> for ZyntaxString {
    fn eq(&self, other: &str) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialEq<&str> for ZyntaxString {
    fn eq(&self, other: &&str) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialEq<String> for ZyntaxString {
    fn eq(&self, other: &String) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl From<&str> for ZyntaxString {
    fn from(s: &str) -> Self {
        Self::from_str(s)
    }
}

impl From<String> for ZyntaxString {
    fn from(s: String) -> Self {
        Self::from_str(&s)
    }
}

impl TryFrom<ZyntaxString> for String {
    type Error = ConversionError;

    fn try_from(value: ZyntaxString) -> Result<Self, Self::Error> {
        value.into_string()
    }
}

// Safety: ZyntaxString's data is just bytes, safe to send across threads
unsafe impl Send for ZyntaxString {}
// ZyntaxString provides only immutable access to its data via &self methods
unsafe impl Sync for ZyntaxString {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_read() {
        let s = ZyntaxString::from_str("Hello, Zyntax!");
        assert_eq!(s.len(), 14);
        assert_eq!(s.as_str().unwrap(), "Hello, Zyntax!");
    }

    #[test]
    fn test_empty_string() {
        let s = ZyntaxString::empty();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.as_str().unwrap(), "");
    }

    #[test]
    fn test_clone() {
        let s1 = ZyntaxString::from_str("test");
        let s2 = s1.clone();
        assert_eq!(s1, s2);
        assert_ne!(s1.as_ptr(), s2.as_ptr()); // Different allocations
    }

    #[test]
    fn test_to_string() {
        let s = ZyntaxString::from_str("convert me");
        let rust_string = s.to_string().unwrap();
        assert_eq!(rust_string, "convert me");
    }

    #[test]
    fn test_from_string() {
        let s: ZyntaxString = "from str".into();
        assert_eq!(s.as_str().unwrap(), "from str");

        let s: ZyntaxString = String::from("from String").into();
        assert_eq!(s.as_str().unwrap(), "from String");
    }

    #[test]
    fn test_equality() {
        let s1 = ZyntaxString::from_str("equal");
        let s2 = ZyntaxString::from_str("equal");
        let s3 = ZyntaxString::from_str("different");

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert!(s1 == "equal");
        assert!(s1 == String::from("equal"));
    }

    #[test]
    fn test_into_raw() {
        let s = ZyntaxString::from_str("raw pointer test");
        let ptr = s.into_raw();

        unsafe {
            // Verify the data is still valid
            let len = *ptr;
            assert_eq!(len, 16);

            // Manually free
            let total_size = ZyntaxString::HEADER_SIZE + len as usize;
            let layout = std::alloc::Layout::from_size_align_unchecked(total_size, 4);
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
    }
}
