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
use crate::hir::{HirFunction, HirId, HirModule};

/// IR container handed to the JIT backend per-compile.
///
/// `tier` drives OSR codegen: tier 0 emits back-edge probes; tier ≥ 1
/// emits OSR helpers and skips probes. `bead_id` is the OSR registry key
/// embedded as a constant into tier-0 probe call sites.
#[derive(Clone)]
pub struct ZyntaxFunctionDef {
    pub id: HirId,
    /// Shared with whatever handed it over: a tier compiles the body it
    /// is given, not a copy made on the way.
    pub function: std::sync::Arc<HirFunction>,
    /// Module-level effect, handler, global, and callee context required when
    /// a single hot function is recompiled outside the initial bulk pass.
    pub module: std::sync::Arc<HirModule>,
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

    /// The baseline's resume points for `def`'s function: one helper per
    /// loop header the layout admits, for an interpreted frame to leave
    /// through, or for a loop the optimizing tier made none for. The
    /// function is compiled again here for its helpers alone; its call
    /// cell is left to whatever code is installed.
    pub fn resume_points(&self, def: &ZyntaxFunctionDef) -> Vec<(u64, *mut ())> {
        let (translated, isa) = self
            .with_lock(|backend| {
                backend.set_compile_tier(1);
                backend.set_compile_bead_id(def.bead_id);
                backend
                    .translate_function_in_shared_module(def.id, &def.function, &def.module)
                    .ok()
                    .flatten()
                    .map(|t| (t, backend.isa()))
            })
            .map_or((None, None), |(t, isa)| (Some(t), Some(isa)));
        let (Some(mut translated), Some(isa)) = (translated, isa) else {
            return Vec::new();
        };
        if translated.compile(&*isa).is_err() {
            return Vec::new();
        }
        self.with_lock(|backend| {
            backend.set_compile_tier(1);
            backend.set_compile_bead_id(def.bead_id);
            if backend
                .install_function_in_shared_module(translated, &def.function)
                .is_err()
            {
                return Vec::new();
            }
            backend.set_defer_cell_publish(true);
            let finalized = backend.finalize_definitions();
            backend.set_defer_cell_publish(false);
            // The cells stay with the higher tier.
            backend.take_deferred_cells();
            if finalized.is_err() {
                return Vec::new();
            }
            backend.take_pending_osr_helpers()
        })
    }
}

impl ZyntaxCraneliftBackend {
    /// The resume point of `def` at `header` alone, compiled and
    /// finalised, as `(site, code)`; the body is compiled already. The
    /// cells stay with whatever tier holds them.
    pub fn resume_point_at(&self, def: &ZyntaxFunctionDef, header: HirId) -> Vec<(u64, *mut ())> {
        // As in `compile`: translation and installation under the lock,
        // Cranelift's compile between them without it, so the frame's
        // own thread is not held behind a helper it is waiting for when
        // it compiles a callee.
        let (translated, isa) = self.with_lock(|backend| {
            backend.set_compile_tier(1);
            backend.set_compile_bead_id(def.bead_id);
            (
                backend.translate_resume_point(def.id, &def.function, &def.module, header),
                backend.isa(),
            )
        });
        let Ok(Some((mut translated, site))) = translated else {
            return Vec::new();
        };
        if translated.compile(&*isa).is_err() {
            return Vec::new();
        }
        self.with_lock(|backend| {
            backend.set_compile_tier(1);
            backend.set_compile_bead_id(def.bead_id);
            if backend.install_resume_point(translated, site).is_err() {
                return Vec::new();
            }
            backend.set_defer_cell_publish(true);
            let finalized = backend.finalize_definitions();
            backend.set_defer_cell_publish(false);
            backend.take_deferred_cells();
            if finalized.is_err() {
                return Vec::new();
            }
            backend.take_pending_osr_helpers()
        })
    }
}

/// A resume point outlined and compiled: the sites to publish, and the
/// region as optimised, for the tier above to compile its own resume
/// points from.
pub struct OutlinedResumePoint {
    pub sites: Vec<(u64, *mut ())>,
    pub region_id: HirId,
    pub region: crate::hir::HirFunction,
}

impl ZyntaxCraneliftBackend {
    /// The resume point at `header`, outlined into a function of its own
    /// (see [`crate::osr::outline`]), `optimize` run over it, compiled
    /// and installed with the adapter the site is published with.
    /// `None` when the header admits no layout or the region cannot
    /// stand alone; the caller falls back to [`Self::resume_point_at`].
    pub fn outlined_resume_point_at(
        &self,
        def: &ZyntaxFunctionDef,
        header: HirId,
        optimize: &dyn Fn(crate::hir::HirFunction) -> crate::hir::HirFunction,
    ) -> Option<OutlinedResumePoint> {
        let Ok(layout) = crate::osr::osr_layout(&def.function, header) else {
            return None;
        };
        let region_id = HirId::new();
        let name = zyntax_typed_ast::InternedString::new_global(&format!(
            "{}$resume{}",
            def.function.name.resolve_global().unwrap_or_default(),
            layout.loop_ordinal
        ));
        let Some(mut outlined) = crate::osr::outline(&def.function, &layout, region_id, name)
        else {
            return None;
        };
        // `ZYNTAX_DUMP_HIR_DIR` gets the region before and after its
        // optimisation, and the adapter.
        let dump = |f: &crate::hir::HirFunction, stage: &str| {
            crate::hir_dump::dump_function_to_dir(
                f,
                &def.module,
                &format!("{}-{stage}", f.name.resolve_global().unwrap_or_default()),
            );
        };
        dump(&outlined.function, "outlined");
        dump(&outlined.adapter, "adapter");
        outlined.function = optimize(outlined.function);
        dump(&outlined.function, "outlined-opt");
        // Translation and installation under the lock, the compiles
        // between them without it, as in `compile`. The region is a
        // function of the baseline tier: it probes, and makes no resume
        // points of its own.
        let (region, adapter, isa) = self.with_lock(|backend| {
            backend.set_compile_bead_id(def.bead_id);
            backend.set_compile_tier(0);
            let region = backend.translate_function_in_shared_module(
                region_id,
                &outlined.function,
                &def.module,
            );
            backend.set_compile_tier(1);
            let adapter = backend.translate_resume_adapter(
                def.id,
                &outlined.adapter,
                &outlined.adapter_layout,
            );
            (region, adapter, backend.isa())
        });
        let trace = crate::osr::osr_trace_enabled();
        let (mut region, (mut adapter, site)) = match (region, adapter) {
            (Ok(Some(region)), Ok(adapter)) => (region, adapter),
            (region, adapter) => {
                if trace {
                    eprintln!(
                        "[osr] outline {}: translation failed: {:?} / {:?}",
                        def.function.name.resolve_global().unwrap_or_default(),
                        region.err(),
                        adapter.err()
                    );
                }
                return None;
            }
        };
        if let Err(e) = region.compile(&*isa) {
            if trace {
                eprintln!("[osr] outline: region compile failed: {e}");
            }
            return None;
        }
        if let Err(e) = adapter.compile(&*isa) {
            if trace {
                eprintln!("[osr] outline: adapter compile failed: {e}");
            }
            return None;
        }
        let sites = self.with_lock(|backend| {
            backend.set_compile_bead_id(def.bead_id);
            backend.set_compile_tier(0);
            let installed = backend
                .install_function_in_shared_module(region, &outlined.function)
                .and_then(|_| {
                    backend.set_compile_tier(1);
                    backend.install_resume_point(adapter, site)
                })
                // Cells publish: the adapter reaches the region through
                // its cell, as every compiled call reaches its callee.
                .and_then(|_| backend.finalize_definitions());
            if let Err(e) = installed {
                if trace {
                    eprintln!("[osr] outline: install failed: {e}");
                }
                return Vec::new();
            }
            backend.take_pending_osr_helpers()
        });
        if sites.is_empty() {
            return None;
        }
        Some(OutlinedResumePoint {
            sites,
            region_id,
            region: outlined.function,
        })
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
        // Translation and installation need the module and run under the
        // lock; Cranelift's own compile, the long part, does not, so
        // another thread's compile is not held behind it.
        let (translated, isa) = self.with_lock(|backend| {
            backend.set_compile_tier(tier);
            backend.set_compile_bead_id(bead_id);
            let translated = backend
                .translate_function_in_shared_module(def.id, &def.function, &def.module)
                .map_err(|e| {
                    CompileError::new(format!("cranelift compile_function failed: {e}"))
                })?;
            Ok::<_, CompileError>((translated, backend.isa()))
        })?;
        let translated = match translated {
            Some(mut translated) => {
                translated.compile(&*isa).map_err(|e| {
                    CompileError::new(format!("cranelift compile_function failed: {e}"))
                })?;
                Some(translated)
            }
            None => None,
        };
        self.with_lock(|backend| {
            backend.set_compile_tier(tier);
            backend.set_compile_bead_id(bead_id);
            if let Some(translated) = translated {
                backend
                    .install_function_in_shared_module(translated, &def.function)
                    .map_err(|e| {
                        CompileError::new(format!("cranelift compile_function failed: {e}"))
                    })?;
            }
            // Resolve the new definition before reading any pointer from
            // it: `get_function_ptr` would otherwise hand back the
            // *previous* generation's code, and resolving an OSR helper
            // FuncId panics outright on an unfinalized module.
            backend.finalize_definitions().map_err(|e| {
                CompileError::new(format!("cranelift finalize_definitions failed: {e}"))
            })?;
            let entry = backend
                .get_function_ptr(def.id)
                .map(|p| p as *mut ())
                .ok_or_else(|| {
                    CompileError::new(format!("cranelift produced no fn ptr for {:?}", def.id))
                })?;

            // A resume-point compile (tier ≥ 1) has produced OSR
            // helpers: install them alongside the new entry pointer
            // atomically. Returning the null sentinel suppresses the
            // broker's own swap_compiled call so the OSR-aware swap is
            // the only one that runs.
            //
            // Under hot reload the backend keeps them unpublished: the
            // reload decides which resume points still fit the running
            // code and publishes those itself.
            let osr_pairs = backend.take_pending_osr_helpers();
            if !osr_pairs.is_empty() && backend.publish_osr_helpers() {
                let osr_entries: Vec<beadie::OsrEntry> = osr_pairs
                    .into_iter()
                    .map(|(site, code)| beadie::OsrEntry { site, code })
                    .collect();
                // Publish each helper into the slot its back-edge loads,
                // after the bead swap so a transfer never lands on a
                // half-installed entry.
                for e in &osr_entries {
                    crate::osr::publish_helper(bead_id, e.site, e.code);
                }
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
pub use llvm_impl::{LlvmContextKeepAlive, ZyntaxLlvmBackend, build_llvm_backend};

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

        /// This tier's resume points for `def`'s function at `sites`
        /// alone, as `(site, code)`, published under the bead. The
        /// function is compiled again for them; its entry is left to
        /// whatever code is installed.
        pub fn resume_points(
            &self,
            def: &ZyntaxFunctionDef,
            sites: std::collections::HashSet<u64>,
        ) -> Vec<(u64, *mut ())> {
            if sites.is_empty() {
                return Vec::new();
            }
            crate::opt_audit::note_llvm_body(def.id, &def.function);
            self.with_lock(|backend| {
                backend.set_compile_tier(def.tier);
                backend.set_module_context(std::sync::Arc::clone(&def.module));
                backend.set_osr_helper_sites(Some(sites));
                if backend.compile_function(def.id, &def.function).is_err() {
                    return Vec::new();
                }
                Self::publish_helpers(backend, def)
            })
        }

        /// Publish the helpers of the last install that belong to `def`'s
        /// function under its bead, so back-edges still running tier-0
        /// code can finish here instead of waiting for the next call.
        fn publish_helpers(
            backend: &mut LLVMJitBackend<'static>,
            def: &ZyntaxFunctionDef,
        ) -> Vec<(u64, *mut ())> {
            backend
                .take_pending_osr_helpers()
                .into_iter()
                .filter(|(id, _, _)| *id == def.id)
                .map(|(_, site, code)| {
                    crate::osr::note_llvm_helper(code as usize);
                    crate::osr::publish_helper(def.bead_id, site, code);
                    (site, code)
                })
                .collect()
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
            let tier = def.tier;
            // Resume points where a frame can take one: the sites frames
            // asked at, and an outlined region's own header.
            let sites = crate::osr::wanted_resume_points(def.bead_id, &def.function);
            crate::opt_audit::note_llvm_body(def.id, &def.function);
            self.with_lock(|backend| {
                backend.set_compile_tier(tier);
                backend.set_module_context(std::sync::Arc::clone(&def.module));
                backend.set_osr_helper_sites(Some(sites));
                backend
                    .compile_function(def.id, &def.function)
                    .map_err(|e| CompileError::new(format!("llvm compile_function failed: {e}")))?;
                Self::publish_helpers(backend, &def);

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
