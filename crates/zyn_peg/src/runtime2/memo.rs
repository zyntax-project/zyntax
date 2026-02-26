//! Packrat Memoization for ZynPEG 2.0
//!
//! Implements packrat parsing with memoization to achieve O(n) parsing time.
//! Each (position, rule) pair is memoized to avoid re-parsing.

use super::state::ParsedValue;
use std::collections::HashMap;

/// Key for memoization cache
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoKey {
    /// Position in input
    pub pos: usize,
    /// Rule identifier (index in rule table)
    pub rule_id: usize,
}

/// Cached result of parsing a rule at a position
#[derive(Debug, Clone)]
pub enum MemoEntry {
    /// Rule matched, consumed up to `end_pos`, produced `value`
    Success { value: ParsedValue, end_pos: usize },
    /// Rule failed at this position
    Failure,
    /// Currently being computed (for left recursion detection)
    InProgress,
}

/// Memoization cache for packrat parsing
pub struct MemoCache {
    entries: HashMap<MemoKey, MemoEntry>,
}

impl MemoCache {
    /// Create a new empty cache
    pub fn new() -> Self {
        MemoCache {
            entries: HashMap::new(),
        }
    }

    /// Get a cached entry
    pub fn get(&self, key: MemoKey) -> Option<&MemoEntry> {
        self.entries.get(&key)
    }

    /// Insert an entry
    pub fn insert(&mut self, key: MemoKey, entry: MemoEntry) {
        self.entries.insert(key, entry);
    }

    /// Check if a key is in progress (for left recursion detection)
    pub fn is_in_progress(&self, key: MemoKey) -> bool {
        matches!(self.get(key), Some(MemoEntry::InProgress))
    }

    /// Mark a rule as in progress
    pub fn mark_in_progress(&mut self, key: MemoKey) {
        self.insert(key, MemoEntry::InProgress);
    }

    /// Clear the cache (useful for testing or when input changes)
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get statistics about cache usage
    pub fn stats(&self) -> MemoStats {
        let mut successes = 0;
        let mut failures = 0;
        let mut in_progress = 0;

        for entry in self.entries.values() {
            match entry {
                MemoEntry::Success { .. } => successes += 1,
                MemoEntry::Failure => failures += 1,
                MemoEntry::InProgress => in_progress += 1,
            }
        }

        MemoStats {
            total_entries: self.entries.len(),
            successes,
            failures,
            in_progress,
        }
    }
}

impl Default for MemoCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about memoization cache
#[derive(Debug, Clone)]
pub struct MemoStats {
    pub total_entries: usize,
    pub successes: usize,
    pub failures: usize,
    pub in_progress: usize,
}

/// Rule identifier generator
/// Assigns unique IDs to rules for memoization
pub struct RuleIdGenerator {
    next_id: usize,
    rule_names: HashMap<String, usize>,
}

impl RuleIdGenerator {
    pub fn new() -> Self {
        RuleIdGenerator {
            next_id: 0,
            rule_names: HashMap::new(),
        }
    }

    /// Get or create an ID for a rule name
    pub fn get_id(&mut self, rule_name: &str) -> usize {
        if let Some(&id) = self.rule_names.get(rule_name) {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.rule_names.insert(rule_name.to_string(), id);
            id
        }
    }

    /// Get ID for a rule name without creating
    pub fn lookup(&self, rule_name: &str) -> Option<usize> {
        self.rule_names.get(rule_name).copied()
    }
}

impl Default for RuleIdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memo_cache() {
        let mut cache = MemoCache::new();

        let key = MemoKey { pos: 0, rule_id: 1 };

        // Initially not present
        assert!(cache.get(key).is_none());

        // Mark in progress
        cache.mark_in_progress(key);
        assert!(cache.is_in_progress(key));

        // Store success
        cache.insert(
            key,
            MemoEntry::Success {
                value: ParsedValue::Text("test".to_string()),
                end_pos: 4,
            },
        );

        // Retrieve
        match cache.get(key) {
            Some(MemoEntry::Success { value, end_pos }) => {
                assert_eq!(*end_pos, 4);
                assert!(matches!(value, ParsedValue::Text(s) if s == "test"));
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_rule_id_generator() {
        let mut gen = RuleIdGenerator::new();

        let id1 = gen.get_id("expr");
        let id2 = gen.get_id("stmt");
        let id3 = gen.get_id("expr"); // Same as id1

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);

        assert_eq!(gen.lookup("expr"), Some(id1));
        assert_eq!(gen.lookup("unknown"), None);
    }
}
