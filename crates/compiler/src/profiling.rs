//! # Runtime Profiling Infrastructure
//!
//! Provides execution counters and profiling data for tiered compilation.
//! Tracks function and block execution frequencies to identify hot code paths.

use crate::hir::HirId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Runtime profiling data collector
#[derive(Clone)]
pub struct ProfileData {
    /// Per-function execution counters
    function_counts: Arc<RwLock<HashMap<HirId, Arc<AtomicU64>>>>,

    /// Per-basic-block execution counters
    block_counts: Arc<RwLock<HashMap<(HirId, HirId), Arc<AtomicU64>>>>,

    /// Configuration for hotness detection
    config: ProfileConfig,
}

/// Configuration for profiling and hotness detection
#[derive(Debug, Clone, Copy)]
pub struct ProfileConfig {
    /// Number of executions before considering function "warm"
    pub warm_threshold: u64,

    /// Number of executions before considering function "hot"
    pub hot_threshold: u64,

    /// Enable block-level profiling (more overhead, better granularity)
    pub enable_block_profiling: bool,

    /// Sample rate (1 = profile every call, 2 = every other call, etc.)
    pub sample_rate: u64,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            warm_threshold: 100, // Warm after 100 executions
            hot_threshold: 1000, // Hot after 1000 executions
            enable_block_profiling: false,
            sample_rate: 1, // Profile every call
        }
    }
}

impl ProfileConfig {
    /// Create a configuration optimized for development (lower thresholds)
    pub fn development() -> Self {
        Self {
            warm_threshold: 10,
            hot_threshold: 100,
            enable_block_profiling: true,
            sample_rate: 1,
        }
    }

    /// Create a configuration optimized for production (higher thresholds)
    pub fn production() -> Self {
        Self {
            warm_threshold: 1000,
            hot_threshold: 10000,
            enable_block_profiling: false,
            sample_rate: 10, // Sample 1/10 calls for lower overhead
        }
    }
}

impl ProfileData {
    /// Create a new profiling data collector
    pub fn new(config: ProfileConfig) -> Self {
        Self {
            function_counts: Arc::new(RwLock::new(HashMap::new())),
            block_counts: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Record a function execution
    pub fn record_function_call(&self, func_id: HirId) {
        let mut counts = self.function_counts.write().unwrap();
        let counter = counts
            .entry(func_id)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)));
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a basic block execution
    pub fn record_block_execution(&self, func_id: HirId, block_id: HirId) {
        if !self.config.enable_block_profiling {
            return;
        }

        let mut counts = self.block_counts.write().unwrap();
        let counter = counts
            .entry((func_id, block_id))
            .or_insert_with(|| Arc::new(AtomicU64::new(0)));
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Get execution count for a function
    pub fn get_function_count(&self, func_id: HirId) -> u64 {
        let counts = self.function_counts.read().unwrap();
        counts
            .get(&func_id)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Get execution count for a basic block
    pub fn get_block_count(&self, func_id: HirId, block_id: HirId) -> u64 {
        let counts = self.block_counts.read().unwrap();
        counts
            .get(&(func_id, block_id))
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Check if a function is warm (executed moderately)
    pub fn is_warm(&self, func_id: HirId) -> bool {
        let count = self.get_function_count(func_id);
        count >= self.config.warm_threshold && count < self.config.hot_threshold
    }

    /// Check if a function is hot (executed frequently)
    pub fn is_hot(&self, func_id: HirId) -> bool {
        self.get_function_count(func_id) >= self.config.hot_threshold
    }

    /// Get the hotness level of a function
    pub fn get_hotness(&self, func_id: HirId) -> HotnessLevel {
        let count = self.get_function_count(func_id);

        if count >= self.config.hot_threshold {
            HotnessLevel::Hot
        } else if count >= self.config.warm_threshold {
            HotnessLevel::Warm
        } else {
            HotnessLevel::Cold
        }
    }

    /// Get all hot functions (sorted by execution count, descending)
    pub fn get_hot_functions(&self) -> Vec<(HirId, u64)> {
        let counts = self.function_counts.read().unwrap();
        let mut hot_funcs: Vec<_> = counts
            .iter()
            .map(|(id, counter)| (*id, counter.load(Ordering::Relaxed)))
            .filter(|(_, count)| *count >= self.config.hot_threshold)
            .collect();

        hot_funcs.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by count
        hot_funcs
    }

    /// Get all warm functions
    pub fn get_warm_functions(&self) -> Vec<(HirId, u64)> {
        let counts = self.function_counts.read().unwrap();
        let mut warm_funcs: Vec<_> = counts
            .iter()
            .map(|(id, counter)| (*id, counter.load(Ordering::Relaxed)))
            .filter(|(_, count)| {
                *count >= self.config.warm_threshold && *count < self.config.hot_threshold
            })
            .collect();

        warm_funcs.sort_by(|a, b| b.1.cmp(&a.1));
        warm_funcs
    }

    /// Reset all profiling counters
    pub fn reset(&self) {
        let mut func_counts = self.function_counts.write().unwrap();
        let mut block_counts = self.block_counts.write().unwrap();

        func_counts.clear();
        block_counts.clear();
    }

    /// Get a function's counter reference for direct instrumentation
    pub fn get_or_create_function_counter(&self, func_id: HirId) -> Arc<AtomicU64> {
        let mut counts = self.function_counts.write().unwrap();
        counts
            .entry(func_id)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone()
    }

    /// Get profiling statistics
    pub fn get_statistics(&self) -> ProfileStatistics {
        let func_counts = self.function_counts.read().unwrap();

        let total_functions = func_counts.len();
        let hot_count = func_counts
            .values()
            .filter(|c| c.load(Ordering::Relaxed) >= self.config.hot_threshold)
            .count();
        let warm_count = func_counts
            .values()
            .filter(|c| {
                let count = c.load(Ordering::Relaxed);
                count >= self.config.warm_threshold && count < self.config.hot_threshold
            })
            .count();
        let cold_count = total_functions - hot_count - warm_count;

        let total_executions: u64 = func_counts
            .values()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();

        ProfileStatistics {
            total_functions,
            hot_functions: hot_count,
            warm_functions: warm_count,
            cold_functions: cold_count,
            total_executions,
        }
    }
}

/// Hotness level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HotnessLevel {
    Cold, // Below warm threshold
    Warm, // Between warm and hot thresholds
    Hot,  // Above hot threshold
}

/// Profiling statistics summary
#[derive(Debug, Clone)]
pub struct ProfileStatistics {
    pub total_functions: usize,
    pub hot_functions: usize,
    pub warm_functions: usize,
    pub cold_functions: usize,
    pub total_executions: u64,
}

impl ProfileStatistics {
    /// Format as a human-readable string
    pub fn format(&self) -> String {
        format!(
            "Profile Stats: {} total functions ({} hot, {} warm, {} cold), {} total executions",
            self.total_functions,
            self.hot_functions,
            self.warm_functions,
            self.cold_functions,
            self.total_executions
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_data_basic() {
        let profile = ProfileData::new(ProfileConfig {
            warm_threshold: 5,
            hot_threshold: 10,
            enable_block_profiling: false,
            sample_rate: 1,
        });

        let func_id = HirId::new();

        // Initially cold
        assert_eq!(profile.get_hotness(func_id), HotnessLevel::Cold);
        assert_eq!(profile.get_function_count(func_id), 0);

        // Execute 7 times -> warm
        for _ in 0..7 {
            profile.record_function_call(func_id);
        }
        assert_eq!(profile.get_hotness(func_id), HotnessLevel::Warm);
        assert!(profile.is_warm(func_id));
        assert!(!profile.is_hot(func_id));

        // Execute 5 more times -> hot
        for _ in 0..5 {
            profile.record_function_call(func_id);
        }
        assert_eq!(profile.get_hotness(func_id), HotnessLevel::Hot);
        assert!(!profile.is_warm(func_id));
        assert!(profile.is_hot(func_id));
        assert_eq!(profile.get_function_count(func_id), 12);
    }

    #[test]
    fn test_get_hot_functions() {
        let profile = ProfileData::new(ProfileConfig {
            warm_threshold: 5,
            hot_threshold: 10,
            enable_block_profiling: false,
            sample_rate: 1,
        });

        let func1 = HirId::new();
        let func2 = HirId::new();
        let func3 = HirId::new();

        // func1: cold (3 executions)
        for _ in 0..3 {
            profile.record_function_call(func1);
        }

        // func2: warm (7 executions)
        for _ in 0..7 {
            profile.record_function_call(func2);
        }

        // func3: hot (15 executions)
        for _ in 0..15 {
            profile.record_function_call(func3);
        }

        let hot_funcs = profile.get_hot_functions();
        assert_eq!(hot_funcs.len(), 1);
        assert_eq!(hot_funcs[0].0, func3);
        assert_eq!(hot_funcs[0].1, 15);

        let warm_funcs = profile.get_warm_functions();
        assert_eq!(warm_funcs.len(), 1);
        assert_eq!(warm_funcs[0].0, func2);
        assert_eq!(warm_funcs[0].1, 7);
    }

    #[test]
    fn test_statistics() {
        let profile = ProfileData::new(ProfileConfig::default());

        let func1 = HirId::new();
        let func2 = HirId::new();

        for _ in 0..50 {
            profile.record_function_call(func1);
        }

        for _ in 0..500 {
            profile.record_function_call(func2);
        }

        let stats = profile.get_statistics();
        assert_eq!(stats.total_functions, 2);
        assert_eq!(stats.cold_functions, 1); // func1 is cold (< 100)
        assert_eq!(stats.warm_functions, 1); // func2 is warm (>= 100, < 1000)
        assert_eq!(stats.hot_functions, 0);
        assert_eq!(stats.total_executions, 550);
    }
}
