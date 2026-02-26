//! # Tiered Compilation Backend
//!
//! Implements multi-tier JIT compilation with automatic optimization based on runtime profiling.
//! Combines Cranelift (fast baseline) with optional LLVM JIT (maximum optimization).
//!
//! ## Optimization Tiers
//! - **Tier 0 (Baseline)**: Cranelift with minimal optimization (for cold code)
//! - **Tier 1 (Standard)**: Cranelift with moderate optimization (for warm code)
//! - **Tier 2 (Optimized)**: Cranelift or LLVM with aggressive optimization (for hot code)
//!
//! ## How It Works
//! 1. All functions start at Tier 0 (baseline JIT with Cranelift)
//! 2. Execution counters track how often functions are called
//! 3. When a function crosses the "warm" threshold, it's recompiled at Tier 1 (Cranelift)
//! 4. When it crosses the "hot" threshold, it's recompiled at Tier 2 (Cranelift or LLVM)
//! 5. Function pointers are atomically swapped after recompilation
//!
//! ## Backend Selection
//! - **Cranelift**: Fast compilation, good optimization (default for all tiers)
//! - **LLVM MCJIT**: Slower compilation, maximum optimization (optional for Tier 2)
//!
//! Use `TieredConfig::production_llvm()` to enable LLVM for hot-path optimization.
//!
//! ## Future Extensions
//! - On-Stack Replacement (OSR) for upgrading running functions
//! - Deoptimization support for debugging
//! - Profile-guided inlining decisions

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;

use crate::cranelift_backend::CraneliftBackend;
use crate::hir::{HirFunction, HirId, HirModule};
use crate::profiling::{ProfileConfig, ProfileData};
use crate::{CompilerError, CompilerResult};

/// Runtime symbol entry for FFI
#[derive(Clone)]
struct RuntimeSymbol {
    name: String,
    ptr: usize, // Store as usize for thread safety
}

#[cfg(feature = "llvm-backend")]
use crate::llvm_jit_backend::LLVMJitBackend;
#[cfg(feature = "llvm-backend")]
use inkwell::context::Context;

/// Tiered compilation backend
pub struct TieredBackend {
    /// Primary Cranelift backend (Tier 0 & 1)
    cranelift: CraneliftBackend,

    /// Optional LLVM JIT backend for Tier 2 hot-path optimization
    #[cfg(feature = "llvm-backend")]
    llvm_jit: Option<LLVMJitBackend<'static>>,

    /// LLVM context (must outlive the JIT backend)
    #[cfg(feature = "llvm-backend")]
    llvm_context: Option<Box<Context>>,

    /// Runtime profiling data
    profile_data: ProfileData,

    /// Current optimization tier for each function
    function_tiers: Arc<RwLock<HashMap<HirId, OptimizationTier>>>,

    /// Function pointers (usize for thread safety, cast to *const u8 when needed)
    function_pointers: Arc<RwLock<HashMap<HirId, usize>>>,

    /// Queue of functions waiting for recompilation at higher tier
    optimization_queue: Arc<Mutex<VecDeque<(HirId, OptimizationTier)>>>,

    /// Functions currently being optimized
    optimizing: Arc<Mutex<HashSet<HirId>>>,

    /// The HIR module (needed for recompilation)
    module: Arc<RwLock<Option<HirModule>>>,

    /// Runtime symbols for FFI (from ZRTL plugins)
    runtime_symbols: Arc<RwLock<Vec<RuntimeSymbol>>>,

    /// Configuration
    config: TieredConfig,

    /// Background optimization worker handle
    worker_handle: Option<thread::JoinHandle<()>>,

    /// Shutdown signal
    shutdown: Arc<Mutex<bool>>,
}

/// Optimization tier level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptimizationTier {
    Baseline,  // Tier 0: Fast compilation, minimal optimization
    Standard,  // Tier 1: Moderate optimization
    Optimized, // Tier 2: Aggressive optimization
}

impl OptimizationTier {
    /// Get Cranelift optimization level for this tier
    pub fn cranelift_opt_level(&self) -> &'static str {
        match self {
            OptimizationTier::Baseline => "none",            // -O0
            OptimizationTier::Standard => "speed",           // -O2
            OptimizationTier::Optimized => "speed_and_size", // -O3
        }
    }

    /// Get the next higher tier
    pub fn next_tier(&self) -> Option<OptimizationTier> {
        match self {
            OptimizationTier::Baseline => Some(OptimizationTier::Standard),
            OptimizationTier::Standard => Some(OptimizationTier::Optimized),
            OptimizationTier::Optimized => None, // Already at max
        }
    }
}

/// Configuration for tiered compilation
#[derive(Debug, Clone)]
pub struct TieredConfig {
    /// Profiling configuration
    pub profile_config: ProfileConfig,

    /// Enable background optimization (async optimization in separate thread)
    pub enable_background_optimization: bool,

    /// How often to check for hot functions (in milliseconds)
    pub optimization_check_interval_ms: u64,

    /// Maximum number of functions to optimize in parallel
    pub max_parallel_optimizations: usize,

    /// Backend to use for Tier 2 (hot code)
    pub tier2_backend: Tier2Backend,

    /// Verbosity level (0 = silent, 1 = basic, 2 = detailed)
    pub verbosity: u8,
}

/// Backend choice for Tier 2 (hot code optimization)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier2Backend {
    /// Use Cranelift with maximum optimization
    Cranelift,
    /// Use LLVM MCJIT with aggressive optimization (requires llvm-backend feature)
    #[cfg(feature = "llvm-backend")]
    LLVM,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            profile_config: ProfileConfig::default(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 100,
            max_parallel_optimizations: 4,
            tier2_backend: Tier2Backend::Cranelift,
            verbosity: 0,
        }
    }
}

impl TieredConfig {
    /// Development configuration (aggressive optimization, verbose)
    pub fn development() -> Self {
        Self {
            profile_config: ProfileConfig::development(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 50,
            max_parallel_optimizations: 2,
            tier2_backend: Tier2Backend::Cranelift,
            verbosity: 2,
        }
    }

    /// Production configuration (conservative, low overhead)
    pub fn production() -> Self {
        Self {
            profile_config: ProfileConfig::production(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 1000,
            max_parallel_optimizations: 8,
            tier2_backend: Tier2Backend::Cranelift,
            verbosity: 0,
        }
    }

    /// Production configuration with LLVM for maximum optimization
    #[cfg(feature = "llvm-backend")]
    pub fn production_llvm() -> Self {
        Self {
            profile_config: ProfileConfig::production(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 1000,
            max_parallel_optimizations: 8,
            tier2_backend: Tier2Backend::LLVM,
            verbosity: 0,
        }
    }
}

impl TieredBackend {
    /// Create a new tiered backend
    pub fn new(config: TieredConfig) -> CompilerResult<Self> {
        let cranelift = CraneliftBackend::new()?;
        let profile_data = ProfileData::new(config.profile_config);

        // Initialize LLVM JIT backend if configured for Tier 2
        #[cfg(feature = "llvm-backend")]
        let (llvm_context, llvm_jit) = if matches!(config.tier2_backend, Tier2Backend::LLVM) {
            // Create context (must outlive the backend)
            let context = Box::new(Context::create());
            // SAFETY: We ensure the context outlives the backend by storing both
            let context_ref = unsafe { &*(context.as_ref() as *const Context) };
            let jit = LLVMJitBackend::new(context_ref)?;
            (Some(context), Some(jit))
        } else {
            (None, None)
        };

        Ok(Self {
            cranelift,
            #[cfg(feature = "llvm-backend")]
            llvm_jit,
            #[cfg(feature = "llvm-backend")]
            llvm_context,
            profile_data,
            function_tiers: Arc::new(RwLock::new(HashMap::new())),
            function_pointers: Arc::new(RwLock::new(HashMap::new())),
            optimization_queue: Arc::new(Mutex::new(VecDeque::new())),
            optimizing: Arc::new(Mutex::new(HashSet::new())),
            module: Arc::new(RwLock::new(None)),
            runtime_symbols: Arc::new(RwLock::new(Vec::new())),
            config,
            worker_handle: None,
            shutdown: Arc::new(Mutex::new(false)),
        })
    }

    /// Compile a HIR module (initially at Tier 0 - Baseline)
    pub fn compile_module(&mut self, module: HirModule) -> CompilerResult<()> {
        if self.config.verbosity >= 1 {
            eprintln!(
                "[TieredBackend] Compiling {} functions at Tier 0 (Baseline)",
                module.functions.len()
            );
        }

        // Compile everything at baseline (Tier 0)
        self.cranelift.compile_module(&module)?;

        // Store function pointers and mark all as Baseline tier
        for func_id in module.functions.keys() {
            if let Some(ptr) = self.cranelift.get_function_ptr(*func_id) {
                self.function_pointers
                    .write()
                    .unwrap()
                    .insert(*func_id, ptr as usize);
                self.function_tiers
                    .write()
                    .unwrap()
                    .insert(*func_id, OptimizationTier::Baseline);
            }
        }

        // Store module for later recompilation
        *self.module.write().unwrap() = Some(module);

        // Start background optimization if enabled
        if self.config.enable_background_optimization {
            self.start_background_optimization();
        }

        Ok(())
    }

    /// Get a function pointer
    pub fn get_function_pointer(&self, func_id: HirId) -> Option<*const u8> {
        self.function_pointers
            .read()
            .unwrap()
            .get(&func_id)
            .map(|addr| *addr as *const u8)
    }

    /// Record a function call (for profiling and tier promotion)
    pub fn record_call(&self, func_id: HirId) {
        // Sample based on config
        let count = self.profile_data.get_function_count(func_id);
        if count % self.config.profile_config.sample_rate != 0 {
            return;
        }

        self.profile_data.record_function_call(func_id);

        // Check if function should be promoted to next tier
        let should_promote = {
            let tiers = self.function_tiers.read().unwrap();
            let current_tier = tiers
                .get(&func_id)
                .copied()
                .unwrap_or(OptimizationTier::Baseline);

            match current_tier {
                OptimizationTier::Baseline if self.profile_data.is_warm(func_id) => {
                    Some(OptimizationTier::Standard)
                }
                OptimizationTier::Standard if self.profile_data.is_hot(func_id) => {
                    Some(OptimizationTier::Optimized)
                }
                _ => None,
            }
        };

        if let Some(target_tier) = should_promote {
            self.enqueue_for_optimization(func_id, target_tier);
        }
    }

    /// Enqueue a function for optimization at a specific tier
    fn enqueue_for_optimization(&self, func_id: HirId, target_tier: OptimizationTier) {
        let mut queue = self.optimization_queue.lock().unwrap();
        let optimizing = self.optimizing.lock().unwrap();

        // Don't enqueue if already optimizing or already in queue at this tier
        if !optimizing.contains(&func_id)
            && !queue
                .iter()
                .any(|(id, tier)| *id == func_id && *tier == target_tier)
        {
            if self.config.verbosity >= 2 {
                let count = self.profile_data.get_function_count(func_id);
                eprintln!(
                    "[TieredBackend] Enqueuing {:?} for {:?} (count: {})",
                    func_id, target_tier, count
                );
            }
            queue.push_back((func_id, target_tier));
        }
    }

    /// Manually trigger recompilation of a function at a specific tier
    pub fn optimize_function(
        &mut self,
        func_id: HirId,
        target_tier: OptimizationTier,
    ) -> CompilerResult<()> {
        // Clone the function to avoid holding the read lock
        let function = {
            let module_lock = self.module.read().unwrap();
            let module = module_lock
                .as_ref()
                .ok_or_else(|| CompilerError::Backend("No module loaded".into()))?;

            module
                .functions
                .get(&func_id)
                .ok_or_else(|| CompilerError::Backend(format!("Function {:?} not found", func_id)))?
                .clone()
        };

        self.optimize_function_internal(func_id, &function, target_tier)
    }

    /// Internal: Recompile a single function at a specific tier
    fn optimize_function_internal(
        &mut self,
        func_id: HirId,
        function: &HirFunction,
        target_tier: OptimizationTier,
    ) -> CompilerResult<()> {
        if self.config.verbosity >= 1 {
            let count = self.profile_data.get_function_count(func_id);
            eprintln!(
                "[TieredBackend] Recompiling {:?} at {:?} (count: {})",
                func_id, target_tier, count
            );
        }

        // Choose backend based on tier and configuration
        let new_ptr = if target_tier == OptimizationTier::Optimized {
            // Tier 2: Use configured backend (Cranelift or LLVM)
            #[cfg(feature = "llvm-backend")]
            if matches!(self.config.tier2_backend, Tier2Backend::LLVM) {
                // Use LLVM JIT for maximum optimization
                if let Some(ref mut llvm) = self.llvm_jit {
                    llvm.compile_function(func_id, function)?;
                    llvm.get_function_pointer(func_id)
                } else {
                    return Err(CompilerError::Backend(
                        "LLVM JIT backend not initialized".to_string(),
                    ));
                }
            } else {
                // Use Cranelift with aggressive optimization
                self.cranelift.compile_function(func_id, function)?;
                self.cranelift.get_function_ptr(func_id)
            }

            #[cfg(not(feature = "llvm-backend"))]
            {
                // LLVM not available, use Cranelift
                self.cranelift.compile_function(func_id, function)?;
                self.cranelift.get_function_ptr(func_id)
            }
        } else {
            // Tier 0 & 1: Always use Cranelift
            self.cranelift.compile_function(func_id, function)?;
            self.cranelift.get_function_ptr(func_id)
        };

        // Atomically swap the function pointer if compilation succeeded
        if let Some(ptr) = new_ptr {
            self.function_pointers
                .write()
                .unwrap()
                .insert(func_id, ptr as usize);
            self.function_tiers
                .write()
                .unwrap()
                .insert(func_id, target_tier);

            if self.config.verbosity >= 1 {
                let backend_name = if target_tier == OptimizationTier::Optimized {
                    #[cfg(feature = "llvm-backend")]
                    if matches!(self.config.tier2_backend, Tier2Backend::LLVM) {
                        "LLVM"
                    } else {
                        "Cranelift"
                    }
                    #[cfg(not(feature = "llvm-backend"))]
                    "Cranelift"
                } else {
                    "Cranelift"
                };
                eprintln!(
                    "[TieredBackend] Successfully promoted {:?} to {:?} using {}",
                    func_id, target_tier, backend_name
                );
            }
        }

        Ok(())
    }

    /// Start background optimization worker thread
    fn start_background_optimization(&mut self) {
        if self.worker_handle.is_some() {
            return; // Already started
        }

        let queue = Arc::clone(&self.optimization_queue);
        let optimizing = Arc::clone(&self.optimizing);
        let module = Arc::clone(&self.module);
        let function_pointers = Arc::clone(&self.function_pointers);
        let function_tiers = Arc::clone(&self.function_tiers);
        let shutdown = Arc::clone(&self.shutdown);
        let profile_data = self.profile_data.clone();
        let config = self.config.clone();

        let handle = thread::spawn(move || {
            if config.verbosity >= 1 {
                eprintln!("[TieredBackend] Background optimization worker started");
            }

            loop {
                // Check for shutdown
                if *shutdown.lock().unwrap() {
                    if config.verbosity >= 1 {
                        eprintln!("[TieredBackend] Background worker shutting down");
                    }
                    break;
                }

                // Process optimization queue
                Self::background_worker_iteration(
                    &queue,
                    &optimizing,
                    &module,
                    &function_pointers,
                    &function_tiers,
                    &profile_data,
                    &config,
                );

                // Sleep before next iteration
                thread::sleep(Duration::from_millis(config.optimization_check_interval_ms));
            }
        });

        self.worker_handle = Some(handle);
    }

    /// Background worker iteration
    fn background_worker_iteration(
        queue: &Arc<Mutex<VecDeque<(HirId, OptimizationTier)>>>,
        optimizing: &Arc<Mutex<HashSet<HirId>>>,
        module: &Arc<RwLock<Option<HirModule>>>,
        function_pointers: &Arc<RwLock<HashMap<HirId, usize>>>,
        function_tiers: &Arc<RwLock<HashMap<HirId, OptimizationTier>>>,
        profile_data: &ProfileData,
        config: &TieredConfig,
    ) {
        let mut queue_lock = queue.lock().unwrap();
        let mut optimizing_lock = optimizing.lock().unwrap();

        // Don't start new optimizations if at capacity
        if optimizing_lock.len() >= config.max_parallel_optimizations {
            return;
        }

        // Dequeue a function to optimize
        if let Some((func_id, target_tier)) = queue_lock.pop_front() {
            optimizing_lock.insert(func_id);
            drop(queue_lock);
            drop(optimizing_lock);

            // Perform optimization
            let result = Self::worker_optimize_function(
                func_id,
                target_tier,
                module,
                function_pointers,
                function_tiers,
                profile_data,
                config,
            );

            // Mark as done
            optimizing.lock().unwrap().remove(&func_id);

            if let Err(e) = result {
                if config.verbosity >= 1 {
                    eprintln!("[TieredBackend] Failed to optimize {:?}: {}", func_id, e);
                }
            }
        }
    }

    /// Worker function to optimize a single function
    fn worker_optimize_function(
        func_id: HirId,
        target_tier: OptimizationTier,
        module: &Arc<RwLock<Option<HirModule>>>,
        function_pointers: &Arc<RwLock<HashMap<HirId, usize>>>,
        function_tiers: &Arc<RwLock<HashMap<HirId, OptimizationTier>>>,
        profile_data: &ProfileData,
        config: &TieredConfig,
    ) -> CompilerResult<()> {
        if config.verbosity >= 1 {
            let count = profile_data.get_function_count(func_id);
            eprintln!(
                "[TieredBackend] Worker optimizing {:?} at {:?} (count: {})",
                func_id, target_tier, count
            );
        }

        // Get module and function
        let module_lock = module.read().unwrap();
        let hir_module = module_lock
            .as_ref()
            .ok_or_else(|| CompilerError::Backend("No module loaded".into()))?;

        let function = hir_module
            .functions
            .get(&func_id)
            .ok_or_else(|| CompilerError::Backend(format!("Function {:?} not found", func_id)))?;

        // Create a new Cranelift backend configured for this tier
        // TODO: Configure Cranelift settings based on tier
        let mut backend = CraneliftBackend::new()?;
        backend.compile_function(func_id, function)?;

        // Get the optimized function pointer
        if let Some(new_ptr) = backend.get_function_ptr(func_id) {
            // Atomically swap
            function_pointers
                .write()
                .unwrap()
                .insert(func_id, new_ptr as usize);
            function_tiers.write().unwrap().insert(func_id, target_tier);

            if config.verbosity >= 1 {
                eprintln!(
                    "[TieredBackend] Worker successfully promoted {:?} to {:?}",
                    func_id, target_tier
                );
            }
        }

        Ok(())
    }

    /// Get profiling and tiering statistics
    pub fn get_statistics(&self) -> TieredStatistics {
        let profile_stats = self.profile_data.get_statistics();
        let tiers = self.function_tiers.read().unwrap();

        let baseline_count = tiers
            .values()
            .filter(|&&t| t == OptimizationTier::Baseline)
            .count();
        let standard_count = tiers
            .values()
            .filter(|&&t| t == OptimizationTier::Standard)
            .count();
        let optimized_count = tiers
            .values()
            .filter(|&&t| t == OptimizationTier::Optimized)
            .count();

        TieredStatistics {
            profile_stats,
            baseline_functions: baseline_count,
            standard_functions: standard_count,
            optimized_functions: optimized_count,
            queued_for_optimization: self.optimization_queue.lock().unwrap().len(),
            currently_optimizing: self.optimizing.lock().unwrap().len(),
        }
    }

    /// Shutdown the tiered backend (stops background worker)
    pub fn shutdown(&mut self) {
        *self.shutdown.lock().unwrap() = true;

        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }

    /// Register a runtime symbol for FFI/plugin linking
    ///
    /// This allows external functions from ZRTL plugins to be called from JIT code.
    /// Note: Symbols registered after module compilation will be available for
    /// recompilation in background workers.
    pub fn register_runtime_symbol(&mut self, name: &str, ptr: *const u8) {
        self.runtime_symbols.write().unwrap().push(RuntimeSymbol {
            name: name.to_string(),
            ptr: ptr as usize,
        });
    }

    /// Get all registered runtime symbols as a vector of (name, ptr) tuples
    ///
    /// Used when creating new Cranelift backends for background optimization.
    fn get_runtime_symbols(&self) -> Vec<(String, *const u8)> {
        self.runtime_symbols
            .read()
            .unwrap()
            .iter()
            .map(|s| (s.name.clone(), s.ptr as *const u8))
            .collect()
    }
}

impl Drop for TieredBackend {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Statistics about the tiered backend
#[derive(Debug, Clone)]
pub struct TieredStatistics {
    pub profile_stats: crate::profiling::ProfileStatistics,
    pub baseline_functions: usize,
    pub standard_functions: usize,
    pub optimized_functions: usize,
    pub queued_for_optimization: usize,
    pub currently_optimizing: usize,
}

impl TieredStatistics {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "Tiered Compilation: {} Baseline (T0), {} Standard (T1), {} Optimized (T2)\n\
             Queue: {} waiting, {} optimizing\n\
             {}",
            self.baseline_functions,
            self.standard_functions,
            self.optimized_functions,
            self.queued_for_optimization,
            self.currently_optimizing,
            self.profile_stats.format()
        )
    }
}
