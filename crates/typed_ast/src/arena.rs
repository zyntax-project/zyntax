//! # Arena-based Memory Management
//!
//! Provides efficient memory management for TypedAST nodes using string interning.
//! This approach significantly reduces memory usage and improves performance by:
//! - Deduplicating strings through interning
//! - Enabling efficient string comparison and storage
//! - Supporting serialization and deserialization

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use string_interner::{DefaultBackend, StringInterner, Symbol as SymbolTrait};

/// Symbol type used for string interning
pub type Symbol = string_interner::DefaultSymbol;

/// A symbol representing an interned string
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InternedString(Symbol);

impl Default for InternedString {
    fn default() -> Self {
        InternedString::new_global("")
    }
}

impl InternedString {
    /// Create a new interned string symbol (internal use only)
    pub(crate) fn new(symbol: Symbol) -> Self {
        Self(symbol)
    }

    /// Create from a raw symbol (for testing)
    pub fn from_symbol(symbol: Symbol) -> Self {
        Self(symbol)
    }

    /// Get the underlying symbol
    pub fn symbol(&self) -> Symbol {
        self.0
    }
}

// Custom serialization for InternedString
impl Serialize for InternedString {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as the actual string value for portability
        // This makes the JSON human-readable and works across processes
        if let Some(resolved) = self.resolve_global() {
            resolved.serialize(serializer)
        } else {
            // Fallback to empty string if resolution fails
            "".serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for InternedString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Use a visitor pattern to support all serde formats (postcard, bincode, JSON, etc.)
        // We serialize as a string, so we deserialize as a string
        struct InternedStringVisitor;

        impl<'de> serde::de::Visitor<'de> for InternedStringVisitor {
            type Value = InternedString;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(InternedString::new_global(value))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(InternedString::new_global(&value))
            }

            fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(InternedString::new_global(value))
            }
        }

        deserializer.deserialize_str(InternedStringVisitor)
    }
}

impl fmt::Display for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InternedString({:?})", self.0)
    }
}

// Global string interner for JSON deserialization
// This is used when deserializing TypedAST from JSON without an explicit arena context
use once_cell::sync::Lazy;
use std::sync::Mutex;

static GLOBAL_INTERNER: Lazy<Mutex<StringInterner<DefaultBackend>>> =
    Lazy::new(|| Mutex::new(StringInterner::new()));

impl InternedString {
    /// Create an InternedString from a string using the global interner
    /// This is primarily used during JSON deserialization
    pub fn new_global(s: &str) -> Self {
        let mut interner = GLOBAL_INTERNER.lock().unwrap();
        InternedString(interner.get_or_intern(s))
    }

    /// Resolve this InternedString to a string using the global interner
    pub fn resolve_global(&self) -> Option<String> {
        let interner = GLOBAL_INTERNER.lock().unwrap();
        interner.resolve(self.0).map(|s| s.to_string())
    }
}

/// Arena-based memory manager for strings and identifiers
pub struct AstArena {
    /// String interner for deduplicating string literals
    string_interner: StringInterner<DefaultBackend>,

    /// Statistics for memory usage tracking
    stats: ArenaStatistics,
}

impl fmt::Debug for AstArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AstArena")
            .field("interned_strings", &self.string_interner.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl AstArena {
    /// Create a new AST arena with default capacity
    pub fn new() -> Self {
        Self {
            string_interner: StringInterner::new(),
            stats: ArenaStatistics::default(),
        }
    }

    /// Create a new AST arena with specified initial capacity
    pub fn with_capacity(_nodes: usize, _strings: usize) -> Self {
        Self {
            string_interner: StringInterner::new(),
            stats: ArenaStatistics::default(),
        }
    }

    /// Intern a string and return its symbol
    pub fn intern_string(&mut self, string: impl AsRef<str>) -> InternedString {
        let string_ref = string.as_ref();

        // Add to local interner for statistics
        let _local_symbol = self.string_interner.get_or_intern(string_ref);

        // Use global interner for the actual symbol to enable cross-arena resolution
        let global_interned = InternedString::new_global(string_ref);

        // Update statistics
        if self.string_interner.len()
            > self
                .stats
                .max_interned_strings
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.stats.max_interned_strings.store(
                self.string_interner.len(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        global_interned
    }

    /// Resolve an interned string back to its value
    ///
    /// Since `intern_string` returns symbols from the global interner,
    /// we must resolve via global interner. The string is leaked to provide
    /// a stable &str reference (acceptable since interned strings are never freed).
    pub fn resolve_string(&self, interned: InternedString) -> Option<&str> {
        // Always use global interner since intern_string returns global symbols
        interned.resolve_global().map(|s| {
            // Leak the string to provide stable &str lifetime
            // This is safe because interned strings live for the program duration
            Box::leak(s.into_boxed_str()) as &str
        })
    }

    /// Get statistics about arena usage
    pub fn statistics(&self) -> &ArenaStatistics {
        &self.stats
    }

    /// Get the number of interned strings
    pub fn interned_string_count(&self) -> usize {
        self.string_interner.len()
    }

    /// Clear statistics (arena cannot be cleared once allocated)
    pub fn clear_stats(&mut self) {
        // Note: We can't actually clear the typed_arena::Arena as it doesn't support it
        // StringInterner also doesn't support clearing in this version
        self.stats.reset();
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory_usage(&self) -> MemoryUsage {
        let strings_size = self.string_interner.len() * 16; // Rough estimate

        MemoryUsage {
            nodes_bytes: 0, // No longer storing nodes in arena
            strings_bytes: strings_size,
            total_bytes: strings_size,
        }
    }
}

impl Default for AstArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about arena usage
#[derive(Debug, Default)]
pub struct ArenaStatistics {
    /// Maximum number of interned strings seen
    pub max_interned_strings: std::sync::atomic::AtomicUsize,

    /// Number of times strings were deduplicated
    pub string_dedup_count: std::sync::atomic::AtomicUsize,
}

impl ArenaStatistics {
    /// Reset all statistics
    pub fn reset(&self) {
        self.max_interned_strings
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.string_dedup_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current statistics as a snapshot
    pub fn snapshot(&self) -> ArenaStatsSnapshot {
        ArenaStatsSnapshot {
            max_interned_strings: self
                .max_interned_strings
                .load(std::sync::atomic::Ordering::Relaxed),
            string_dedup_count: self
                .string_dedup_count
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

/// Snapshot of arena statistics
#[derive(Debug, Clone, Copy)]
pub struct ArenaStatsSnapshot {
    pub max_interned_strings: usize,
    pub string_dedup_count: usize,
}

/// Memory usage information
#[derive(Debug, Clone, Copy)]
pub struct MemoryUsage {
    pub nodes_bytes: usize,
    pub strings_bytes: usize,
    pub total_bytes: usize,
}

impl fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryUsage {{ nodes: {} bytes, strings: {} bytes, total: {} bytes }}",
            self.nodes_bytes, self.strings_bytes, self.total_bytes
        )
    }
}

/// Thread-safe arena manager for multi-threaded environments
pub struct ThreadSafeArenaManager {
    arenas: std::sync::RwLock<HashMap<std::thread::ThreadId, AstArena>>,
}

impl ThreadSafeArenaManager {
    /// Create a new thread-safe arena manager
    pub fn new() -> Self {
        Self {
            arenas: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Get or create an arena for the current thread
    pub fn get_arena(
        &self,
    ) -> std::sync::RwLockWriteGuard<'_, HashMap<std::thread::ThreadId, AstArena>> {
        let thread_id = std::thread::current().id();
        let mut arenas = self.arenas.write().unwrap();

        if !arenas.contains_key(&thread_id) {
            arenas.insert(thread_id, AstArena::new());
        }

        arenas
    }

    /// Get global statistics across all threads
    pub fn global_statistics(&self) -> ArenaStatsSnapshot {
        let arenas = self.arenas.read().unwrap();
        let mut total_strings = 0;
        let mut total_dedup = 0;

        for arena in arenas.values() {
            let stats = arena.statistics().snapshot();
            total_strings += stats.max_interned_strings;
            total_dedup += stats.string_dedup_count;
        }

        ArenaStatsSnapshot {
            max_interned_strings: total_strings,
            string_dedup_count: total_dedup,
        }
    }

    /// Clean up arenas for dead threads
    pub fn cleanup_dead_threads(&self) {
        let mut arenas = self.arenas.write().unwrap();
        let current_thread_id = std::thread::current().id();

        // Note: This is a simplified cleanup - in practice, you'd need a more
        // sophisticated approach to track which threads are still alive
        arenas.retain(|thread_id, _| *thread_id == current_thread_id);
    }
}

impl Default for ThreadSafeArenaManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for working with arenas
pub mod utils {
    use super::*;

    /// Calculate memory savings from string interning
    pub fn calculate_string_savings(original_strings: &[String], interned_count: usize) -> usize {
        let original_size: usize = original_strings.iter().map(|s| s.len()).sum();
        let interned_size = interned_count * std::mem::size_of::<InternedString>();

        if original_size > interned_size {
            original_size - interned_size
        } else {
            0
        }
    }

    /// Estimate optimal arena capacity based on expected usage
    pub fn estimate_arena_capacity(
        estimated_nodes: usize,
        estimated_strings: usize,
    ) -> (usize, usize) {
        // Add 20% buffer to avoid frequent reallocations
        let node_capacity = (estimated_nodes as f64 * 1.2) as usize;
        let string_capacity = (estimated_strings as f64 * 1.2) as usize;

        (node_capacity, string_capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_interning() {
        let mut arena = AstArena::new();

        let str1 = arena.intern_string("hello");
        let str2 = arena.intern_string("hello");
        let str3 = arena.intern_string("world");

        // Same strings should have same symbol
        assert_eq!(str1, str2);
        assert_ne!(str1, str3);

        // Should be able to resolve back
        assert_eq!(arena.resolve_string(str1), Some("hello"));
        assert_eq!(arena.resolve_string(str3), Some("world"));
    }

    #[test]
    fn test_memory_usage_estimation() {
        let mut arena = AstArena::new();

        // Intern some strings
        arena.intern_string("hello");
        arena.intern_string("world");
        arena.intern_string("foo");

        let usage = arena.estimated_memory_usage();
        assert!(usage.strings_bytes > 0);
        assert_eq!(usage.total_bytes, usage.strings_bytes);
    }

    #[test]
    fn test_capacity_estimation() {
        let (nodes, strings) = utils::estimate_arena_capacity(100, 50);

        // Should add buffer
        assert!(nodes > 100);
        assert!(strings > 50);
    }

    #[test]
    fn test_cross_arena_resolution() {
        // Simulate what happens when multiple tests create separate arenas
        // All arenas share the GLOBAL_INTERNER

        // Arena 1 interns "std"
        let std_symbol = {
            let mut arena1 = AstArena::new();
            arena1.intern_string("std")
        };

        // Arena 2 interns different strings, then tries to resolve "std" symbol
        let mut arena2 = AstArena::new();
        arena2.intern_string("hash_fn");
        arena2.intern_string("malloc");

        // The symbol from arena1 should still resolve correctly via global interner
        let resolved = arena2.resolve_string(std_symbol);
        assert_eq!(
            resolved,
            Some("std"),
            "Cross-arena resolution should work via global interner"
        );
    }

    #[test]
    fn test_sequential_arena_builds() {
        // Simulates the stdlib test pattern where each test builds parts of stdlib

        // Test 1: builds hashmap functions
        {
            let mut arena = AstArena::new();
            let hash_fn = arena.intern_string("hash_fn");
            let eq_fn = arena.intern_string("eq_fn");
            assert_eq!(arena.resolve_string(hash_fn), Some("hash_fn"));
            assert_eq!(arena.resolve_string(eq_fn), Some("eq_fn"));
        }

        // Test 2: builds memory functions
        {
            let mut arena = AstArena::new();
            let malloc = arena.intern_string("malloc");
            let free = arena.intern_string("free");
            assert_eq!(arena.resolve_string(malloc), Some("malloc"));
            assert_eq!(arena.resolve_string(free), Some("free"));
        }

        // Test 3: builds full stdlib with module name "std"
        {
            let mut arena = AstArena::new();
            let std_name = arena.intern_string("std");
            // The symbol should resolve correctly even though global interner
            // already has many other strings from previous "tests"
            assert_eq!(arena.resolve_string(std_name), Some("std"));
        }
    }
}
