//! ZyntaxString - Direct interop with Zyntax's string format
//!
//! A Zyntax string is the ZRTL SDK's: a sixteen-byte header, then the
//! bytes (see [`zrtl::string`]).
//!
//! This module provides `ZyntaxString` for direct manipulation of these strings
//! without intermediate conversion to Rust `String`.

use crate::error::{ConversionError, ConversionResult};
use std::ptr::NonNull;

/// A string in Zyntax's native format.
///
/// # Memory Ownership
///
/// `ZyntaxString` can either own its memory (will free on drop) or borrow
/// from Zyntax runtime (must not be freed by Rust).
pub struct ZyntaxString {
    /// Pointer to the header
    ptr: NonNull<i32>,
    /// Whether we own this memory (should free on drop)
    owned: bool,
}

impl ZyntaxString {
    /// Header size in bytes
    pub const HEADER_SIZE: usize = zrtl::STRING_HEADER_SIZE;

    /// Create a new ZyntaxString from a Rust string
    pub fn from_str(s: &str) -> Self {
        Self::owning(zrtl::string_new(s))
    }

    fn owning(ptr: *mut i32) -> Self {
        Self {
            ptr: NonNull::new(ptr).expect("Failed to allocate ZyntaxString"),
            owned: true,
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
    /// - The string must be one `zrtl::string_free` releases
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
        unsafe { zrtl::string_length(self.ptr.as_ptr()) as usize }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the string data as a byte slice
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { zrtl::string_as_bytes(self.ptr.as_ptr()) }
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
        unsafe { zrtl::string_size(self.ptr.as_ptr()) }
    }
}

impl Drop for ZyntaxString {
    fn drop(&mut self) {
        if self.owned {
            unsafe { zrtl::string_free(self.ptr.as_ptr()) }
        }
    }
}

impl Clone for ZyntaxString {
    fn clone(&self) -> Self {
        // Always an owned string of the same bytes.
        Self::owning(zrtl::string_from_bytes(self.as_bytes()))
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
    fn test_reads_through_the_header() {
        let s = ZyntaxString::from_str("héllo wörld");
        assert_eq!(s.len(), 13);
        assert_eq!(s.as_str().unwrap(), "héllo wörld");
        assert_eq!(s.allocation_size(), zrtl::STRING_HEADER_SIZE + 13);
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

            zrtl::string_free(ptr);
        }
    }
}
