//! # Beadie Adapter
//!
//! Wraps the existing `CraneliftBackend` and (optionally) `LLVMJitBackend`
//! with the [`beadie::JitBackend`] trait so the tiered compilation pipeline
//! can drive them through `beadie::TieredAdapter`.
//!
//! Why the unsafe Send/Sync impls: `JITModule` (Cranelift) and the LLVM
//! execution-engine handles aren't `Send`/`Sync` on their own. We serialize
//! all access through an internal `Mutex`, so concurrent access from
//! beadie's per-tier broker threads is safe.
//!
//! The wrappers don't create the underlying backends — pass an existing
//! instance in. That lets the eager `compile_module` step in
//! [`crate::tiered_backend::TieredBackend`] share the same Cranelift module
//! that promotion jobs will later recompile into.
//!
//! No backend state is exposed publicly — anything the parent
//! `TieredBackend` needs goes through `with_lock`.

use std::sync::Mutex;

use beadie::{Bead, JitBackend};

use crate::cranelift_backend::CraneliftBackend;
use crate::hir::{HirFunction, HirId};

/// IR container handed to the JIT backend per-compile.
///
/// `tier` drives OSR codegen: tier 0 emits back-edge probes; tier ≥ 1
/// emits OSR helpers and skips probes. `bead_id` is the OSR registry key
/// embedded as a constant into tier-0 probe call sites.
pub struct ZyntaxFunctionDef {
    pub id: HirId,
    pub function: HirFunction,
    pub tier: usize,
    pub bead_id: u64,
}

/// Convert a beadie [`beadie::CompileError`]-bearing closure error into our
/// own type. We use `beadie::CompileError` directly as `JitBackend::Error` to
/// keep the trait bound simple (`std::error::Error + Send + Sync + 'static`).
type CompileError = beadie::CompileError;

// ─────────────────────────────────────────────────────────────────────────────
// Cranelift wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// `JitBackend` wrapper around [`CraneliftBackend`].
pub struct ZyntaxCraneliftBackend {
    inner: Mutex<CraneliftBackend>,
}

// SAFETY: All access to the inner `CraneliftBackend` is serialized through
// the `Mutex`. The `JITModule` it owns contains `RefCell` and raw pointers
// that aren't auto-`Send`/`Sync`, but a single-threaded critical section per
// access satisfies their actual safety requirements.
unsafe impl Send for ZyntaxCraneliftBackend {}
unsafe impl Sync for ZyntaxCraneliftBackend {}

impl ZyntaxCraneliftBackend {
    pub fn new(backend: CraneliftBackend) -> Self {
        Self {
            inner: Mutex::new(backend),
        }
    }

    /// Run `f` with exclusive access to the wrapped backend.
    pub fn with_lock<R>(&self, f: impl FnOnce(&mut CraneliftBackend) -> R) -> R {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }
}

impl JitBackend for ZyntaxCraneliftBackend {
    type FunctionDef = ZyntaxFunctionDef;
    type Error = CompileError;

    fn compile(
        &self,
        bead: &std::sync::Arc<Bead>,
        def: Self::FunctionDef,
    ) -> Result<*mut (), Self::Error> {
        let tier = def.tier;
        let bead_id = def.bead_id;
        self.with_lock(|backend| {
            backend.set_compile_tier(tier);
            backend.set_compile_bead_id(bead_id);
            backend
                .compile_function(def.id, &def.function)
                .map_err(|e| {
                    CompileError::new(format!("cranelift compile_function failed: {e}"))
                })?;
            let entry = backend
                .get_function_ptr(def.id)
                .map(|p| p as *mut ())
                .ok_or_else(|| {
                    CompileError::new(format!("cranelift produced no fn ptr for {:?}", def.id))
                })?;

            // Tier ≥ 1 may have produced OSR helpers — install them
            // alongside the new entry pointer atomically. Returning the
            // null sentinel suppresses the broker's own swap_compiled
            // call so the OSR-aware swap is the only one that runs.
            let osr_pairs = backend.take_pending_osr_helpers();
            if !osr_pairs.is_empty() {
                let osr_entries: Vec<beadie::OsrEntry> = osr_pairs
                    .into_iter()
                    .map(|(site, code)| beadie::OsrEntry { site, code })
                    .collect();
                bead.swap_compiled_with_osr(entry, osr_entries);
                Ok(std::ptr::null_mut())
            } else {
                Ok(entry)
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LLVM wrapper (optional)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "llvm-backend")]
pub use llvm_impl::{build_llvm_backend, LlvmContextKeepAlive, ZyntaxLlvmBackend};

#[cfg(feature = "llvm-backend")]
mod llvm_impl {
    use super::{Bead, CompileError, JitBackend, Mutex, ZyntaxFunctionDef};
    use crate::llvm_jit_backend::LLVMJitBackend;

    /// `JitBackend` wrapper around [`LLVMJitBackend`].
    ///
    /// Drop order matters: the inkwell `ExecutionEngine` inside `inner`
    /// borrows from the `Context` boxed in `_keepalive`. Rust drops
    /// fields in declaration order, so `inner` (the backend +
    /// ExecutionEngine) drops first, then `_keepalive` (the Context).
    /// Reversing these fields would crash on drop.
    pub struct ZyntaxLlvmBackend {
        inner: Mutex<LLVMJitBackend<'static>>,
        _keepalive: Option<std::sync::Arc<LlvmContextKeepAlive>>,
    }

    // SAFETY: same justification as `ZyntaxCraneliftBackend` — the `Mutex`
    // serializes all access to the inkwell `ExecutionEngine` handles.
    unsafe impl Send for ZyntaxLlvmBackend {}
    unsafe impl Sync for ZyntaxLlvmBackend {}

    impl ZyntaxLlvmBackend {
        pub fn new(backend: LLVMJitBackend<'static>) -> Self {
            Self {
                inner: Mutex::new(backend),
                _keepalive: None,
            }
        }

        pub fn with_lock<R>(&self, f: impl FnOnce(&mut LLVMJitBackend<'static>) -> R) -> R {
            let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            f(&mut guard)
        }
    }

    impl JitBackend for ZyntaxLlvmBackend {
        type FunctionDef = ZyntaxFunctionDef;
        type Error = CompileError;

        fn compile(
            &self,
            _bead: &std::sync::Arc<Bead>,
            def: Self::FunctionDef,
        ) -> Result<*mut (), Self::Error> {
            // LLVM tier-2 ignores `tier` / `bead_id` for now — OSR is wired
            // through the Cranelift path (tiers 0/1) only. Once LLVM gains
            // an OSR-aware codegen path these would propagate similarly to
            // the Cranelift wrapper above.
            self.with_lock(|backend| {
                backend
                    .compile_function(def.id, &def.function)
                    .map_err(|e| CompileError::new(format!("llvm compile_function failed: {e}")))?;
                backend
                    .get_function_pointer(def.id)
                    .map(|p| p as *mut ())
                    .ok_or_else(|| {
                        CompileError::new(format!("llvm produced no fn ptr for {:?}", def.id))
                    })
            })
        }
    }

    /// Opaque keep-alive handle for the inkwell `Context` that
    /// `ZyntaxLlvmBackend` borrows from. Stored alongside the backend
    /// in the runtime; dropped only after every consumer is dropped.
    ///
    /// `Send + Sync` are unsafely implemented because the inner
    /// `Context` is only touched through the `ZyntaxLlvmBackend`'s
    /// own `Mutex`-serialised access — the keep-alive itself never
    /// hands out direct references.
    pub struct LlvmContextKeepAlive {
        // Boxed so the heap address is stable; the LLVMJitBackend
        // borrows from this address for its lifetime.
        _context: Box<inkwell::context::Context>,
    }
    // SAFETY: the wrapped `Context` is only ever accessed through the
    // sibling `ZyntaxLlvmBackend`'s `Mutex`. This handle exists only
    // to keep the address alive; it never exposes the inner reference.
    unsafe impl Send for LlvmContextKeepAlive {}
    unsafe impl Sync for LlvmContextKeepAlive {}

    /// Build an `Arc<ZyntaxLlvmBackend>` along with the
    /// `inkwell::Context` keep-alive that owns its lifetime.
    ///
    /// `LLVMJitBackend` borrows from a `Context` that must outlive
    /// every JIT'd module. This helper self-pins the `Context` in a
    /// boxed keep-alive and hands a `'static` reference to the
    /// backend; callers hold the returned `Arc<LlvmContextKeepAlive>`
    /// to keep the storage alive for the runtime's lifetime.
    ///
    /// Encapsulated here so callers (`zyntax_embed::ZyntaxRuntime`)
    /// never need to depend on `inkwell` directly.
    ///
    /// # Safety
    /// The reference inside `ZyntaxLlvmBackend` is logically bound to
    /// the lifetime of the returned `Arc<LlvmContextKeepAlive>`. Drop
    /// the keep-alive only after dropping every consumer of the
    /// backend.
    pub fn build_llvm_backend() -> Result<
        (
            std::sync::Arc<ZyntaxLlvmBackend>,
            std::sync::Arc<LlvmContextKeepAlive>,
        ),
        crate::CompilerError,
    > {
        use inkwell::context::Context;

        let context: Box<Context> = Box::new(Context::create());
        // SAFETY: `context` is moved into the `LlvmContextKeepAlive`
        // which the caller stores in an `Arc` for the runtime's
        // lifetime. The `'static` reference we hand to
        // `LLVMJitBackend` therefore points to storage that outlives
        // every consumer.
        let context_ref: &'static Context = unsafe { &*(context.as_ref() as *const Context) };
        let inner = LLVMJitBackend::new(context_ref)
            .map_err(|e| crate::CompilerError::Backend(format!("llvm init failed: {e}")))?;
        let keepalive = std::sync::Arc::new(LlvmContextKeepAlive { _context: context });
        let backend = ZyntaxLlvmBackend {
            inner: Mutex::new(inner),
            _keepalive: Some(std::sync::Arc::clone(&keepalive)),
        };
        Ok((std::sync::Arc::new(backend), keepalive))
    }
}
