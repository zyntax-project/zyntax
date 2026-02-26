//! Iterator traits for ZRTL-compatible iteration
//!
//! This module provides traits and types for implementing iterators that are
//! compatible with Zyntax's iteration protocol. These traits enable Rust types
//! to be used seamlessly in Zyntax's for-in loops and iterator chains.
//!
//! ## Protocol
//!
//! The ZRTL iterator protocol uses three operations:
//! 1. `has_next()` - Check if more elements exist
//! 2. `next()` - Get the next element
//! 3. `reset()` - Reset to the beginning (optional)
//!
//! ## Example
//!
//! ```rust,ignore
//! use zyntax_embed::iterator::{ZrtlIterable, ZrtlIterator};
//!
//! // Any type implementing ZrtlIterable can be iterated
//! fn sum_all<I: ZrtlIterable<Item = i64>>(iterable: &I) -> i64 {
//!     let mut iter = iterable.zrtl_iter();
//!     let mut sum = 0;
//!     while iter.has_next() {
//!         if let Some(value) = iter.next() {
//!             sum += value;
//!         }
//!     }
//!     sum
//! }
//! ```

use crate::array::ZyntaxArray;
use crate::string::ZyntaxString;
use crate::value::ZyntaxValue;

/// Trait for types that can be iterated in a ZRTL-compatible way.
///
/// This is similar to `IntoIterator` but provides a protocol that matches
/// Zyntax's iteration semantics, which uses explicit `has_next()` checks
/// rather than returning `None` from `next()`.
pub trait ZrtlIterable {
    /// The type of elements yielded by the iterator
    type Item;
    /// The iterator type
    type Iterator: ZrtlIterator<Item = Self::Item>;

    /// Create a new iterator over this collection
    fn zrtl_iter(&self) -> Self::Iterator;
}

/// Trait for ZRTL-compatible iterators.
///
/// This iterator protocol uses explicit `has_next()` checks, which maps
/// directly to Zyntax's iteration semantics and enables efficient
/// cross-language iteration.
pub trait ZrtlIterator {
    /// The type of elements yielded by this iterator
    type Item;

    /// Check if there are more elements
    fn has_next(&mut self) -> bool;

    /// Get the next element, or None if exhausted
    fn next(&mut self) -> Option<Self::Item>;

    /// Reset the iterator to the beginning (if supported)
    fn reset(&mut self) -> bool {
        false // Not all iterators support reset
    }

    /// Get the remaining count (if known)
    fn remaining(&self) -> Option<usize> {
        None
    }

    /// Advance by n elements, returning the number actually skipped
    fn skip(&mut self, mut n: usize) -> usize {
        let mut skipped = 0;
        while n > 0 && self.has_next() {
            self.next();
            skipped += 1;
            n -= 1;
        }
        skipped
    }

    /// Collect remaining elements into a Vec
    fn collect_vec(mut self) -> Vec<Self::Item>
    where
        Self: Sized,
    {
        let capacity = self.remaining().unwrap_or(0);
        let mut result = Vec::with_capacity(capacity);
        while self.has_next() {
            if let Some(item) = self.next() {
                result.push(item);
            }
        }
        result
    }

    /// Fold over remaining elements
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        while self.has_next() {
            if let Some(item) = self.next() {
                acc = f(acc, item);
            }
        }
        acc
    }

    /// Find the first element satisfying a predicate
    fn find<P>(mut self, mut predicate: P) -> Option<Self::Item>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        while self.has_next() {
            if let Some(item) = self.next() {
                if predicate(&item) {
                    return Some(item);
                }
            }
        }
        None
    }

    /// Check if any element satisfies a predicate
    fn any<P>(mut self, mut predicate: P) -> bool
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        while self.has_next() {
            if let Some(item) = self.next() {
                if predicate(&item) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if all elements satisfy a predicate
    fn all<P>(mut self, mut predicate: P) -> bool
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        while self.has_next() {
            if let Some(item) = self.next() {
                if !predicate(&item) {
                    return false;
                }
            }
        }
        true
    }

    /// Count the remaining elements
    fn count(mut self) -> usize
    where
        Self: Sized,
    {
        let mut n = 0;
        while self.has_next() {
            self.next();
            n += 1;
        }
        n
    }
}

/// Iterator over ZyntaxArray elements
pub struct ZyntaxArrayIterator<'a, T: Copy> {
    array: &'a ZyntaxArray<T>,
    index: usize,
}

impl<'a, T: Copy> ZyntaxArrayIterator<'a, T> {
    /// Create a new iterator over a ZyntaxArray
    pub fn new(array: &'a ZyntaxArray<T>) -> Self {
        Self { array, index: 0 }
    }
}

impl<'a, T: Copy> ZrtlIterator for ZyntaxArrayIterator<'a, T> {
    type Item = T;

    fn has_next(&mut self) -> bool {
        self.index < self.array.len()
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.array.len() {
            let value = self.array[self.index];
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }

    fn reset(&mut self) -> bool {
        self.index = 0;
        true
    }

    fn remaining(&self) -> Option<usize> {
        Some(self.array.len() - self.index)
    }
}

impl<'a, T: Copy> ZrtlIterable for &'a ZyntaxArray<T> {
    type Item = T;
    type Iterator = ZyntaxArrayIterator<'a, T>;

    fn zrtl_iter(&self) -> Self::Iterator {
        ZyntaxArrayIterator::new(self)
    }
}

// Note: Direct ZrtlIterable impl for ZyntaxArray<T> removed because
// it requires unsafe transmute to create 'static lifetime.
// Use (&array).zrtl_iter() instead for safe iteration.

/// Iterator over UTF-8 codepoints in a ZyntaxString
pub struct ZyntaxStringCharsIterator<'a> {
    bytes: &'a [u8],
    index: usize,
}

impl<'a> ZyntaxStringCharsIterator<'a> {
    /// Create a new iterator over characters in a ZyntaxString
    pub fn new(string: &'a ZyntaxString) -> Self {
        Self {
            bytes: string.as_bytes(),
            index: 0,
        }
    }
}

impl<'a> ZrtlIterator for ZyntaxStringCharsIterator<'a> {
    type Item = char;

    fn has_next(&mut self) -> bool {
        self.index < self.bytes.len()
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.bytes.len() {
            return None;
        }

        // Decode UTF-8 codepoint
        let first = self.bytes[self.index];
        let (codepoint, len) = if first < 0x80 {
            // ASCII
            (first as u32, 1)
        } else if first < 0xE0 {
            // 2-byte
            if self.index + 1 >= self.bytes.len() {
                return None;
            }
            let second = self.bytes[self.index + 1];
            (((first as u32 & 0x1F) << 6) | (second as u32 & 0x3F), 2)
        } else if first < 0xF0 {
            // 3-byte
            if self.index + 2 >= self.bytes.len() {
                return None;
            }
            let second = self.bytes[self.index + 1];
            let third = self.bytes[self.index + 2];
            (
                ((first as u32 & 0x0F) << 12)
                    | ((second as u32 & 0x3F) << 6)
                    | (third as u32 & 0x3F),
                3,
            )
        } else {
            // 4-byte
            if self.index + 3 >= self.bytes.len() {
                return None;
            }
            let second = self.bytes[self.index + 1];
            let third = self.bytes[self.index + 2];
            let fourth = self.bytes[self.index + 3];
            (
                ((first as u32 & 0x07) << 18)
                    | ((second as u32 & 0x3F) << 12)
                    | ((third as u32 & 0x3F) << 6)
                    | (fourth as u32 & 0x3F),
                4,
            )
        };

        self.index += len;
        char::from_u32(codepoint)
    }

    fn reset(&mut self) -> bool {
        self.index = 0;
        true
    }
}

/// Iterator over bytes in a ZyntaxString
pub struct ZyntaxStringBytesIterator<'a> {
    bytes: &'a [u8],
    index: usize,
}

impl<'a> ZyntaxStringBytesIterator<'a> {
    /// Create a new iterator over bytes in a ZyntaxString
    pub fn new(string: &'a ZyntaxString) -> Self {
        Self {
            bytes: string.as_bytes(),
            index: 0,
        }
    }
}

impl<'a> ZrtlIterator for ZyntaxStringBytesIterator<'a> {
    type Item = u8;

    fn has_next(&mut self) -> bool {
        self.index < self.bytes.len()
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.bytes.len() {
            let byte = self.bytes[self.index];
            self.index += 1;
            Some(byte)
        } else {
            None
        }
    }

    fn reset(&mut self) -> bool {
        self.index = 0;
        true
    }

    fn remaining(&self) -> Option<usize> {
        Some(self.bytes.len() - self.index)
    }
}

impl ZyntaxString {
    /// Create an iterator over characters (UTF-8 codepoints)
    pub fn chars_zrtl(&self) -> ZyntaxStringCharsIterator<'_> {
        ZyntaxStringCharsIterator::new(self)
    }

    /// Create an iterator over bytes
    pub fn bytes_zrtl(&self) -> ZyntaxStringBytesIterator<'_> {
        ZyntaxStringBytesIterator::new(self)
    }
}

/// Iterator over ZyntaxValue elements (for arrays stored as ZyntaxValue::Array)
pub struct ZyntaxValueIterator {
    values: Vec<ZyntaxValue>,
    index: usize,
}

impl ZyntaxValueIterator {
    /// Create a new iterator from a Vec of ZyntaxValues
    pub fn new(values: Vec<ZyntaxValue>) -> Self {
        Self { values, index: 0 }
    }
}

impl ZrtlIterator for ZyntaxValueIterator {
    type Item = ZyntaxValue;

    fn has_next(&mut self) -> bool {
        self.index < self.values.len()
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.values.len() {
            let value = self.values[self.index].clone();
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }

    fn reset(&mut self) -> bool {
        self.index = 0;
        true
    }

    fn remaining(&self) -> Option<usize> {
        Some(self.values.len() - self.index)
    }
}

/// Range iterator for integer ranges
pub struct ZrtlRangeIterator {
    current: i64,
    end: i64,
    step: i64,
}

impl ZrtlRangeIterator {
    /// Create a range from start to end (exclusive) with step 1
    pub fn new(start: i64, end: i64) -> Self {
        Self {
            current: start,
            end,
            step: 1,
        }
    }

    /// Create a range with custom step
    pub fn with_step(start: i64, end: i64, step: i64) -> Self {
        Self {
            current: start,
            end,
            step,
        }
    }
}

impl ZrtlIterator for ZrtlRangeIterator {
    type Item = i64;

    fn has_next(&mut self) -> bool {
        if self.step > 0 {
            self.current < self.end
        } else {
            self.current > self.end
        }
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_next() {
            let value = self.current;
            self.current += self.step;
            Some(value)
        } else {
            None
        }
    }

    fn remaining(&self) -> Option<usize> {
        if self.step > 0 && self.current < self.end {
            Some(
                ((self.end - self.current) as usize + (self.step as usize - 1))
                    / self.step as usize,
            )
        } else if self.step < 0 && self.current > self.end {
            Some(
                ((self.current - self.end) as usize + (-self.step as usize - 1))
                    / (-self.step) as usize,
            )
        } else {
            Some(0)
        }
    }
}

/// Adapter to convert std::iter::Iterator to ZrtlIterator
pub struct StdIteratorAdapter<I: Iterator> {
    inner: std::iter::Peekable<I>,
}

impl<I: Iterator> StdIteratorAdapter<I> {
    /// Wrap a standard iterator
    pub fn new(iter: I) -> Self {
        Self {
            inner: iter.peekable(),
        }
    }
}

impl<I: Iterator> ZrtlIterator for StdIteratorAdapter<I> {
    type Item = I::Item;

    fn has_next(&mut self) -> bool {
        self.inner.peek().is_some()
    }

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Adapter to convert ZrtlIterator to std::iter::Iterator
pub struct ZrtlIteratorAdapter<I: ZrtlIterator> {
    inner: I,
}

impl<I: ZrtlIterator> ZrtlIteratorAdapter<I> {
    /// Wrap a ZRTL iterator
    pub fn new(iter: I) -> Self {
        Self { inner: iter }
    }
}

impl<I: ZrtlIterator> Iterator for ZrtlIteratorAdapter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.inner.remaining() {
            Some(n) => (n, Some(n)),
            None => (0, None),
        }
    }
}

/// Extension trait for converting iterators
pub trait IntoZrtlIterator: Iterator {
    /// Convert a standard iterator to a ZRTL iterator
    fn into_zrtl(self) -> StdIteratorAdapter<Self>
    where
        Self: Sized,
    {
        StdIteratorAdapter::new(self)
    }
}

impl<I: Iterator> IntoZrtlIterator for I {}

/// Extension trait for ZRTL iterators
pub trait ZrtlIteratorExt: ZrtlIterator {
    /// Convert a ZRTL iterator to a standard iterator
    fn into_std(self) -> ZrtlIteratorAdapter<Self>
    where
        Self: Sized,
    {
        ZrtlIteratorAdapter::new(self)
    }
}

impl<I: ZrtlIterator> ZrtlIteratorExt for I {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_iterator() {
        let arr: ZyntaxArray<i32> = [1, 2, 3, 4, 5].into();
        let mut iter = (&arr).zrtl_iter();

        assert!(iter.has_next());
        assert_eq!(iter.remaining(), Some(5));

        let mut sum = 0;
        while iter.has_next() {
            if let Some(v) = iter.next() {
                sum += v;
            }
        }
        assert_eq!(sum, 15);

        // Test reset
        assert!(iter.reset());
        assert!(iter.has_next());
        assert_eq!(iter.next(), Some(1));
    }

    #[test]
    fn test_string_chars_iterator() {
        let s = ZyntaxString::from_str("Hello");
        let mut iter = s.chars_zrtl();

        let chars: Vec<char> = iter.collect_vec();
        assert_eq!(chars, vec!['H', 'e', 'l', 'l', 'o']);
    }

    #[test]
    fn test_string_unicode_iterator() {
        let s = ZyntaxString::from_str("Hé🎉");
        let mut iter = s.chars_zrtl();

        assert_eq!(iter.next(), Some('H'));
        assert_eq!(iter.next(), Some('é'));
        assert_eq!(iter.next(), Some('🎉'));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_string_bytes_iterator() {
        let s = ZyntaxString::from_str("ABC");
        let mut iter = s.bytes_zrtl();

        assert_eq!(iter.next(), Some(b'A'));
        assert_eq!(iter.next(), Some(b'B'));
        assert_eq!(iter.next(), Some(b'C'));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_range_iterator() {
        let mut iter = ZrtlRangeIterator::new(0, 5);

        assert!(iter.has_next());
        assert_eq!(iter.remaining(), Some(5));

        let values: Vec<i64> = iter.collect_vec();
        assert_eq!(values, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_range_iterator_with_step() {
        let mut iter = ZrtlRangeIterator::with_step(0, 10, 2);
        let values: Vec<i64> = iter.collect_vec();
        assert_eq!(values, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_range_iterator_negative_step() {
        let mut iter = ZrtlRangeIterator::with_step(5, 0, -1);
        let values: Vec<i64> = iter.collect_vec();
        assert_eq!(values, vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_iterator_fold() {
        let arr: ZyntaxArray<i32> = [1, 2, 3, 4, 5].into();
        let iter = (&arr).zrtl_iter();
        let sum = iter.fold(0i32, |acc, x| acc + x);
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_iterator_find() {
        let arr: ZyntaxArray<i32> = [1, 2, 3, 4, 5].into();
        let iter = (&arr).zrtl_iter();
        let found = iter.find(|&x| x > 3);
        assert_eq!(found, Some(4));
    }

    #[test]
    fn test_iterator_any() {
        let arr: ZyntaxArray<i32> = [1, 2, 3, 4, 5].into();
        let iter = (&arr).zrtl_iter();
        assert!(iter.any(|&x| x == 3));

        let iter = (&arr).zrtl_iter();
        assert!(!iter.any(|&x| x == 10));
    }

    #[test]
    fn test_iterator_all() {
        let arr: ZyntaxArray<i32> = [2, 4, 6, 8].into();
        let iter = (&arr).zrtl_iter();
        assert!(iter.all(|&x| x % 2 == 0));

        let arr: ZyntaxArray<i32> = [2, 3, 6].into();
        let iter = (&arr).zrtl_iter();
        assert!(!iter.all(|&x| x % 2 == 0));
    }

    #[test]
    fn test_std_to_zrtl_adapter() {
        let vec = vec![1, 2, 3];
        let mut zrtl_iter = vec.into_iter().into_zrtl();

        assert!(zrtl_iter.has_next());
        assert_eq!(zrtl_iter.next(), Some(1));
        assert_eq!(zrtl_iter.next(), Some(2));
        assert_eq!(zrtl_iter.next(), Some(3));
        assert!(!zrtl_iter.has_next());
    }

    #[test]
    fn test_zrtl_to_std_adapter() {
        let arr: ZyntaxArray<i32> = [10, 20, 30].into();
        let zrtl_iter = (&arr).zrtl_iter();
        let std_iter = zrtl_iter.into_std();

        let doubled: Vec<i32> = std_iter.map(|x| x * 2).collect();
        assert_eq!(doubled, vec![20, 40, 60]);
    }
}
