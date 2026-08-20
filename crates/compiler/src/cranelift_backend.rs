//! # Cranelift Backend Integration
//!
//! Provides JIT compilation capabilities using Cranelift for fast development
//! cycles and hot-reloading support.

use cranelift::prelude::*;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::entities::Value;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{
    AbiParam, BlockArg, MemFlagsData as MemFlags, Signature, UserFuncName,
};

/// Wraps `&[Value]` as `Vec<BlockArg>` for jump/brif calls.
/// Cranelift 0.133+ requires `BlockArg` rather than raw `Value`.
#[inline]
fn ba(args: &[Value]) -> Vec<BlockArg> {
    args.iter().map(|&v| BlockArg::Value(v)).collect()
}
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use cranelift_codegen::verify_function;
use cranelift_frontend::Switch as ClifSwitch;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};
use log::{debug, error, info, warn};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use crate::effect_codegen::{
    analyze_handle_effect, analyze_perform_effect, get_handler_ops_info, mangle_handler_op_name,
    runtime as effect_runtime, EffectCodegenContext, HandlerStackEntry, PerformStrategy,
};
use crate::hir::{
    BinaryOp, HirCallable, HirConstant, HirFunction, HirGlobal, HirId, HirInstruction, HirModule,
    HirPatternKind, HirPhi, HirStructType, HirTerminator, HirType, HirVTable, HirValueKind,
    Intrinsic, UnaryOp, VectorMinMaxKind, VectorUnaryKind,
};
use crate::zrtl::{
    default_dynamic_box_opaque_tag_and_size, dynamic_box_tag_and_size_for_hir_type,
    dynamic_box_uses_direct_pointer,
};
use crate::{CompilerError, CompilerResult};

static CRANELIFT_SKIPPED_FUNCTIONS: AtomicUsize = AtomicUsize::new(0);

/// Maximum aggregate size (in bytes) for which inline scalar copy is
/// emitted in place of a libc `memcpy` thunk. Above this threshold the
/// per-byte amortised cost of the call/return is negligible and the
/// inlined unrolled sequence would bloat code size for no gain.
///
/// 64 B covers the hot nbody `Body` struct (56 B = pos×3 + vel×3 + mass)
/// without spilling code-cache.
const INLINE_COPY_MAX_BYTES: u32 = 64;

/// Emit an inline byte-by-byte aggregate copy using straight-line
/// scalar loads/stores. Used in place of `call_memcpy` for small
/// aggregates (`size <= INLINE_COPY_MAX_BYTES`) where the libc thunk's
/// call/return overhead dominates the actual byte movement.
///
/// Strategy: greedy power-of-two chunking — 8 bytes (i64) while at
/// least 8 bytes remain, then a single trailing i32 / i16 / i8 for
/// the residual. Total emit cost is `(size+7)/8` load/store pairs
/// (≤ 9 pairs for size=64), all `notrap` so the optimiser is free
/// to fold or reorder.
///
/// Free function (not a `&self` method) so the call site can hold a
/// `&mut FunctionBuilder` that already mutably borrows
/// `self.codegen_context.func` without a re-borrow conflict.
fn emit_inline_aggregate_copy(
    builder: &mut cranelift_frontend::FunctionBuilder,
    dst: Value,
    src: Value,
    size: u32,
) {
    let flags = cranelift_codegen::ir::MemFlagsData::new().with_notrap();
    let mut offset: i32 = 0;
    let size_i32 = size as i32;

    // 8-byte chunks
    while offset + 8 <= size_i32 {
        let v = builder.ins().load(types::I64, flags, src, offset);
        builder.ins().store(flags, v, dst, offset);
        offset += 8;
    }
    // 4-byte tail
    if offset + 4 <= size_i32 {
        let v = builder.ins().load(types::I32, flags, src, offset);
        builder.ins().store(flags, v, dst, offset);
        offset += 4;
    }
    // 2-byte tail
    if offset + 2 <= size_i32 {
        let v = builder.ins().load(types::I16, flags, src, offset);
        builder.ins().store(flags, v, dst, offset);
        offset += 2;
    }
    // 1-byte tail
    if offset + 1 <= size_i32 {
        let v = builder.ins().load(types::I8, flags, src, offset);
        builder.ins().store(flags, v, dst, offset);
    }
}

/// Cranelift's width-changing casts (`fpromote`, `fdemote`, `ireduce`) carry the
/// `Wider` constraint, which resolves over scalars only. A vector on either side
/// has no encoding here, so the cast must decline rather than be built.
fn cast_needs_scalar_lanes(
    op: &crate::hir::CastOp,
    operand_ty: types::Type,
    target_ty: types::Type,
) -> bool {
    matches!(
        op,
        crate::hir::CastOp::FpExt | crate::hir::CastOp::FpTrunc | crate::hir::CastOp::Trunc
    ) && (operand_ty.is_vector() || target_ty.is_vector())
}

/// The field a single-field struct is carried as, when it is carried as
/// its field rather than by address.
///
/// A struct wrapping one scalar has the same shape as that scalar, so it
/// travels in a register. Anything else needs memory, and the rest of
/// this backend represents it as a pointer to that memory.
fn struct_carried_as_its_field(struct_ty: &crate::hir::HirStructType) -> Option<&HirType> {
    if struct_ty.fields.len() != 1 {
        return None;
    }
    let field = struct_ty.fields.first()?;
    matches!(
        field,
        HirType::I8
            | HirType::I16
            | HirType::I32
            | HirType::I64
            | HirType::I128
            | HirType::U8
            | HirType::U16
            | HirType::U32
            | HirType::U64
            | HirType::U128
            | HirType::F32
            | HirType::F64
            | HirType::Bool
    )
    .then_some(field)
}

/// The struct a function hands back through a destination its caller
/// provides, if it hands one back that way.
///
/// This backend represents a struct SSA value as the address of the
/// bytes, and the only memory a function can put those bytes in on its
/// own is its frame, which its caller outlives. So the caller supplies
/// the memory instead and the function writes through it. Anything with
/// a shape that fits in a register is returned in one and is not covered
/// here.
///
/// Only structs. `Array` and `Union` translate to a pointer too, but an
/// array-typed value is not reliably the frame memory a struct's is:
/// copying one that already points at a buffer would hand back a copy
/// where the buffer itself was meant. A list literal builds a struct
/// holding a separate data pointer, so the shape that actually reaches
/// here is the one covered.
fn destination_return_type(function: &HirFunction) -> Option<&HirType> {
    if function.is_external {
        return None;
    }
    if function.signature.returns.len() != 1 {
        return None;
    }
    match function.signature.returns.first()? {
        ret @ HirType::Struct(s) if struct_carried_as_its_field(s).is_none() => Some(ret),
        _ => None,
    }
}

/// Function bodies this backend had no encoding for, across the process.
///
/// Separate from the skip count: a decline is a routing fact, and a skip
/// is a defect that degraded to interpreter speed.
static CRANELIFT_DECLINED_FUNCTIONS: AtomicUsize = AtomicUsize::new(0);

/// How many function bodies Cranelift declined for want of an encoding.
pub fn cranelift_declined_function_count() -> usize {
    CRANELIFT_DECLINED_FUNCTIONS.load(Ordering::Relaxed)
}

/// Reset the decline count.
pub fn reset_cranelift_declined_function_count() {
    CRANELIFT_DECLINED_FUNCTIONS.store(0, Ordering::Relaxed);
}

impl CraneliftBackend {
    /// Functions this backend had no encoding for, in this instance.
    ///
    /// A caller running a tier ladder can compile exactly these with a
    /// backend that has the encoding, rather than re-compiling the
    /// module or leaving them to run interpreted.
    pub fn declined_functions(&self) -> &std::collections::HashSet<HirId> {
        &self.declined_functions
    }
}

/// Number of function bodies skipped by Cranelift due to recoverable codegen errors.
pub fn cranelift_skipped_function_count() -> usize {
    CRANELIFT_SKIPPED_FUNCTIONS.load(Ordering::Relaxed)
}

/// Reset Cranelift skip diagnostics counter.
pub fn reset_cranelift_skipped_function_count() {
    CRANELIFT_SKIPPED_FUNCTIONS.store(0, Ordering::Relaxed);
}

/// Convert ZRTL TypeTag to Cranelift type
fn type_tag_to_cranelift_type(tag: &crate::zrtl::TypeTag) -> types::Type {
    use crate::zrtl::{PrimitiveSize, TypeCategory};

    match tag.category() {
        TypeCategory::Void => types::I8, // Void represented as i8
        TypeCategory::Bool => types::I8, // Bools as i8
        TypeCategory::Int => {
            let size = tag.type_id();
            if size == PrimitiveSize::Bits8 as u16 {
                types::I8
            } else if size == PrimitiveSize::Bits16 as u16 {
                types::I16
            } else if size == PrimitiveSize::Bits32 as u16 {
                types::I32
            } else if size == PrimitiveSize::Bits64 as u16 {
                types::I64
            } else {
                types::I32 // Default
            }
        }
        TypeCategory::UInt => {
            let size = tag.type_id();
            if size == PrimitiveSize::Bits8 as u16 {
                types::I8
            } else if size == PrimitiveSize::Bits16 as u16 {
                types::I16
            } else if size == PrimitiveSize::Bits32 as u16 {
                types::I32 // u32 uses I32 in Cranelift
            } else if size == PrimitiveSize::Bits64 as u16 {
                types::I64 // u64 uses I64 in Cranelift
            } else {
                types::I32 // Default
            }
        }
        TypeCategory::Float => {
            let size = tag.type_id();
            if size == PrimitiveSize::Bits32 as u16 {
                types::F32
            } else if size == PrimitiveSize::Bits64 as u16 {
                types::F64
            } else {
                types::F32 // Default
            }
        }
        // All other types (pointers, opaques, etc.) are i64
        _ => types::I64,
    }
}

/// Cranelift backend for JIT compilation
pub struct CraneliftBackend {
    /// Functions this backend had no encoding for. A caller with a
    /// backend that does can compile exactly these and leave the rest
    /// alone.
    declined_functions: std::collections::HashSet<HirId>,
    /// JIT module for code generation
    module: JITModule,
    /// Current function builder
    builder_context: FunctionBuilderContext,
    /// Codegen context
    codegen_context: codegen::Context,
    /// When set, `compile_function_body` captures the final CLIF + native
    /// disassembly of each function (for verification tests). Off by default —
    /// enabling it turns on Cranelift's disasm generation.
    capture_ir: bool,
    /// Last captured `(CLIF, Option<disassembly>)` when `capture_ir` is on.
    captured_ir: Option<(String, Option<String>)>,
    /// Data description for constants
    data_desc: DataDescription,
    /// Mapping from HIR functions to JIT function IDs
    function_map: HashMap<HirId, FuncId>,
    /// Mapping from HIR globals to Cranelift data IDs
    global_map: HashMap<HirId, cranelift_module::DataId>,
    /// Mapping from HIR values to Cranelift values
    value_map: HashMap<HirId, Value>,
    /// Mapping from HIR blocks to Cranelift blocks
    block_map: HashMap<HirId, Block>,
    /// Compiled function metadata
    compiled_functions: HashMap<HirId, CompiledFunction>,
    /// Hot-reload state
    hot_reload: HotReloadState,
    /// Exported symbols from compiled functions (name → pointer)
    /// Used for cross-module linking when loading multiple modules
    exported_symbols: HashMap<String, *const u8>,
    /// Runtime symbols registered for external linking
    runtime_symbols: Vec<(String, *const u8)>,
    /// Symbol signatures for auto-boxing support (symbol_name → signature)
    symbol_signatures: HashMap<String, crate::zrtl::ZrtlSymbolSig>,
    /// External function link names (HirId → symbol name) for boxing support
    external_link_names: HashMap<HirId, String>,
    /// Functions that take the address of their result as a leading
    /// argument and write the returned aggregate through it, valued by
    /// how many bytes that destination holds.
    ///
    /// Recorded when a function is declared, and read at each call site,
    /// so a caller compiled from a different view of the program than
    /// its callee still agrees with how the callee was defined. Deriving
    /// it independently at both ends would let the two disagree exactly
    /// when the module is compiled a function at a time.
    destination_returns: HashMap<HirId, u32>,
    /// Functions kept off that convention because their address is
    /// observable, so a call to them can arrive through a pointer that
    /// says nothing about it. See [`crate::dce::address_taken_functions`].
    address_taken: HashSet<HirId>,
    /// Pre-scanned call-site inferred signatures for extern functions with 0-param placeholders
    /// Maps HirId → (param_types, return_type) inferred from first call site
    inferred_extern_sigs: HashMap<HirId, (Vec<HirType>, Option<HirType>)>,
    /// Effect codegen context for algebraic effects
    effect_context: EffectCodegenContext,
    /// Tier the next `compile_function` call should target. 0 = baseline,
    /// 1+ = optimized. Used by OSR codegen to decide whether to emit
    /// back-edge probes (tier 0 only) and OSR helpers (tier ≥ 1 only).
    /// Whether OSR helpers from this backend get published. See
    /// [`Self::publish_osr_helpers`].
    publish_osr_helpers: bool,

    /// Route calls between compiled functions through per-function
    /// reload cells instead of direct references. See
    /// [`Self::set_reloadable_calls`].
    reloadable_calls: bool,

    /// This backend's identity in the reload cell registry.
    reload_key: u64,

    /// When set, finalization records cell updates into
    /// `deferred_cells` instead of publishing them, so a reload can
    /// compile a whole edit set and publish nothing unless every
    /// function compiled.
    defer_cell_publish: bool,
    /// Cell updates recorded while `defer_cell_publish` was set, in
    /// finalization order.
    deferred_cells: Vec<(HirId, usize)>,
    /// The function currently being compiled. A FuncRef or
    /// CreateClosure that names it is a frame handing out its own
    /// continuation (an async poll fn re-parking itself, a recursive
    /// closure); that address must stay inside this generation — a
    /// reload-cell read there would resume a suspended frame in code
    /// with different state numbering.
    current_compile_id: Option<HirId>,

    /// Defaults to 0; set via [`Self::set_compile_tier`] before each
    /// `compile_function` call from the tiered runtime.
    compile_tier: usize,
    /// When `false`, tier-0 codegen skips the back-edge OSR probe and
    /// dispatch emission entirely — no arm-slot load and no
    /// `__zyntax_osr_probe` call is inserted at loop headers.
    ///
    /// Defaults to `true` for backwards compatibility. Embedders that
    /// know no tier ≥ 1 backend will ever install OSR helpers (e.g. the
    /// Cranelift-only ladder without `feature = "llvm-backend"`, or any
    /// runtime that drives tier-up via full recompile + `swap_compiled`)
    /// should set this to `false` via [`Self::set_emit_osr_probes`] to
    /// avoid the per-iteration tick-call + branch in every JIT'd loop.
    /// On a ~100 M-iteration mandelbrot the unconditional probe stream
    /// costs ~100 ms of pure overhead.
    emit_osr_probes: bool,
    /// Per-function bead ids for whole-module compiles. `compile_module`
    /// emits every function in one pass, but helper slots are keyed by
    /// bead, so a single `compile_bead_id` would make every function probe
    /// the same slot — and site keys are only unique within a function.
    bead_ids: HashMap<HirId, u64>,
    /// Bead id this function is registered under in the OSR registry.
    /// Embedded as a constant into probe call sites so the runtime can
    /// look up the bead without a global function-pointer table.
    /// Set via [`Self::set_compile_bead_id`].
    compile_bead_id: u64,
    /// OSR helpers compiled in tier ≥ 1. Cleared at the start of every
    /// tier ≥ 1 `compile_function` and drained by callers after
    /// `finalize_definitions` to populate the bead's OSR table via
    /// `swap_compiled_with_osr`.
    ///
    /// Stored as `(site_key, FuncId)` pre-finalize; the FuncId is
    /// resolved to a code pointer after `finalize_definitions`.
    pending_osr_helpers: Vec<(u64, FuncId)>,
    /// When `Some`, `compile_function_body` is compiling an OSR helper —
    /// not the function itself. The body builds:
    ///   1. A synthetic Cranelift prologue block with sig
    ///      `(i64,i64,i64,i64) -> ReturnType`.
    ///   2. Bit-casts the first `phi_count` args to phi types and uses
    ///      them as block params on the jump to `layout.header`.
    ///   3. Bit-casts the remaining args to non-phi-live-in types and
    ///      stores them in `value_map` under their HirIds.
    ///   4. Iterates only blocks reachable from `layout.header`; the
    ///      original function entry and pre-loop blocks are skipped.
    /// Set by `compile_osr_helpers`, cleared after each helper compile.
    compile_osr_layout: Option<crate::osr::OsrLayout>,
    /// FuncId the helper is being compiled into. Set alongside
    /// `compile_osr_layout`; consumed by `compile_function_body` in
    /// place of the usual `function_map` lookup.
    compile_osr_func_id: Option<FuncId>,
    /// When `Some`, the compile-module loop skips function bodies whose
    /// `HirId` is not in this set. Declarations are still emitted for every
    /// function so cross-module symbol resolution stays correct; only the
    /// (Cranelift-expensive) body compilation is skipped.
    ///
    /// Default `None` preserves the historical behaviour of compiling every
    /// function in the module. Embedders that know the reachable closure of
    /// their entry point (typically computed by
    /// [`crate::dce::reachable_function_ids`]) can opt in via
    /// [`Self::set_only_compile_reachable`] to shave the ~30-40 ms spent on
    /// prelude helpers that a benchmark kernel never calls.
    only_compile_reachable: Option<HashSet<HirId>>,
    /// Code offsets of tier-0 probe sites from the most recent compile,
    /// as `(site_key, offset_from_function_start)`. Recovered from the
    /// source-location table, which is the only post-codegen mapping from
    /// an instruction back to where it landed.
    probe_sites: Vec<(u64, u32)>,
    /// Byte length of the most recently emitted function body.
    last_code_len: Option<usize>,
    /// Redefinition counter per function. Bumped each time an
    /// already-defined function is declared again so the new body gets a
    /// distinct symbol and the previous one stays resident.
    compile_generation: HashMap<HirId, u32>,
}

/// Hot-reload state management
#[derive(Clone)]
struct HotReloadState {
    /// Version counter for functions
    versions: Arc<RwLock<HashMap<HirId, u64>>>,
    /// Previous function versions for rollback
    previous_versions: Arc<RwLock<HashMap<HirId, Vec<CompiledFunction>>>>,
    /// Active function pointers
    function_pointers: Arc<RwLock<HashMap<HirId, *const u8>>>,
}

/// Compiled function metadata
struct CompiledFunction {
    function_id: FuncId,
    version: u64,
    code_ptr: *const u8,
    size: usize,
    signature: Signature,
}

/// Struct layout information
#[derive(Debug, Clone)]
struct StructLayout {
    /// Offset of each field in bytes
    #[allow(dead_code)]
    field_offsets: Vec<u32>,
    /// Total size of the struct
    total_size: u32,
    /// Alignment requirement
    alignment: u32,
}

impl CraneliftBackend {
    /// Create a new Cranelift backend with custom runtime symbols
    ///
    /// # Arguments
    /// * `additional_symbols` - Frontend-specific runtime symbols to register
    ///   Format: (symbol_name, function_pointer)
    pub fn with_runtime_symbols(additional_symbols: &[(&str, *const u8)]) -> CompilerResult<Self> {
        Self::new_internal(Some(additional_symbols))
    }

    /// Create a new Cranelift backend
    pub fn new() -> CompilerResult<Self> {
        Self::new_internal(None)
    }

    fn new_internal(additional_symbols: Option<&[(&str, *const u8)]>) -> CompilerResult<Self> {
        // Configure Cranelift for the current platform
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        // Opt level is fixed on the ISA, so it is process-wide rather than
        // per-tier. `ZYNTAX_OPT_LEVEL` exists to measure what Cranelift's own
        // optimizer contributes, which the tier ladder cannot currently vary.
        flag_builder
            .set(
                "opt_level",
                &std::env::var("ZYNTAX_OPT_LEVEL").unwrap_or_else(|_| "speed".into()),
            )
            .unwrap();
        flag_builder.set("enable_verifier", "true").unwrap();

        let isa_builder = cranelift_native::builder().unwrap();
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        // Create JIT module and register runtime functions
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // OSR runtime symbols are always available — tier-0 codegen emits
        // probe call sites unconditionally for any loop header. In paths
        // that don't use beadie (Classic runtime), the probe returns null
        // harmlessly because no beads are registered.
        for (name, ptr) in crate::osr::osr_runtime_symbols() {
            builder.symbol(name, ptr);
        }
        // String-op runtime intrinsics — referenced unconditionally
        // by `BinaryOp::Eq` / `Ne` on `Ptr(I8)` operands.
        for (name, ptr) in crate::string_intrinsics::string_runtime_symbols() {
            builder.symbol(name, ptr);
        }

        // Register all runtime symbols (both stdlib and frontend-specific)
        // All symbols are now provided via the plugin system
        if let Some(symbols) = additional_symbols {
            for (name, ptr) in symbols {
                builder.symbol(*name, *ptr);
            }
        }

        let module = JITModule::new(builder);

        // Store runtime symbols for potential backend recreation
        let runtime_symbols: Vec<(String, *const u8)> = additional_symbols
            .map(|syms| syms.iter().map(|(n, p)| (n.to_string(), *p)).collect())
            .unwrap_or_default();

        Ok(Self {
            declined_functions: std::collections::HashSet::new(),
            module,
            builder_context: FunctionBuilderContext::new(),
            codegen_context: codegen::Context::new(),
            capture_ir: false,
            captured_ir: None,
            data_desc: DataDescription::new(),
            function_map: HashMap::new(),
            global_map: HashMap::new(),
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            compiled_functions: HashMap::new(),
            hot_reload: HotReloadState {
                versions: Arc::new(RwLock::new(HashMap::new())),
                previous_versions: Arc::new(RwLock::new(HashMap::new())),
                function_pointers: Arc::new(RwLock::new(HashMap::new())),
            },
            exported_symbols: HashMap::new(),
            runtime_symbols,
            symbol_signatures: HashMap::new(),
            external_link_names: HashMap::new(),
            destination_returns: HashMap::new(),
            address_taken: HashSet::new(),
            inferred_extern_sigs: HashMap::new(),
            effect_context: EffectCodegenContext::new(),
            compile_tier: 0,
            publish_osr_helpers: true,
            reloadable_calls: false,
            reload_key: crate::reload::next_backend_key(),
            defer_cell_publish: false,
            deferred_cells: Vec::new(),
            current_compile_id: None,
            emit_osr_probes: true,
            compile_bead_id: 0,
            pending_osr_helpers: Vec::new(),
            compile_osr_layout: None,
            compile_osr_func_id: None,
            only_compile_reachable: None,
            compile_generation: HashMap::new(),
            bead_ids: HashMap::new(),
            probe_sites: Vec::new(),
            last_code_len: None,
        })
    }

    /// Drain the OSR helpers compiled during the most recent
    /// `compile_function`. Each entry is `(site_key, code_ptr)` ready for
    /// `beadie::OsrEntry`.
    ///
    /// Must be called after `finalize_definitions` — the FuncIds
    /// recorded during compilation are resolved to native code pointers
    /// here.
    pub fn take_pending_osr_helpers(&mut self) -> Vec<(u64, *mut ())> {
        let pending = std::mem::take(&mut self.pending_osr_helpers);
        pending
            .into_iter()
            .map(|(site, fid)| {
                let ptr = self.module.get_finalized_function(fid);
                (site, ptr as *mut ())
            })
            .collect()
    }

    /// Runtime address and size of a compiled global's data, once the
    /// module is finalized. Reload uses this to patch dispatch-table
    /// slots in place.
    pub fn global_data_addr(&self, id: HirId) -> Option<(*const u8, usize)> {
        let data_id = self.global_map.get(&id)?;
        Some(self.module.get_finalized_data(*data_id))
    }

    /// Drain the probe-site offsets recorded by the last compile.
    pub fn take_probe_sites(&mut self) -> Vec<(u64, u32)> {
        std::mem::take(&mut self.probe_sites)
    }

    /// Byte length of the most recently emitted function body.
    pub fn last_code_len(&self) -> Option<usize> {
        self.last_code_len
    }

    /// Set the tier for subsequent `compile_function` calls. Read by OSR
    /// codegen to decide whether to emit back-edge probes / helpers.
    pub fn set_compile_tier(&mut self, tier: usize) {
        self.compile_tier = tier;
    }

    /// Whether this backend's OSR helpers should be published as resume
    /// points. When a higher tier exists its helper is the one worth
    /// resuming into; publishing this tier's would win the race to the
    /// slot with code no better than what is already running.
    pub fn publish_osr_helpers(&self) -> bool {
        self.publish_osr_helpers
    }

    /// See [`Self::publish_osr_helpers`].
    pub fn set_publish_osr_helpers(&mut self, publish: bool) {
        self.publish_osr_helpers = publish;
    }

    /// Dispatch calls between compiled functions through per-function
    /// cells, so replacing a function's code is one store that every
    /// caller observes on its next call. Off by default: the cell load
    /// is a cost only a runtime that intends to reload should pay.
    pub fn set_reloadable_calls(&mut self, on: bool) {
        self.reloadable_calls = on;
    }

    /// This backend's key in the reload cell registry.
    pub fn reload_key(&self) -> u64 {
        self.reload_key
    }

    /// Defer reload-cell publication: while set, finalization records
    /// cell updates instead of applying them. A reload turns this on
    /// for its compile pass so nothing becomes callable unless the
    /// whole edit set compiled.
    pub fn set_defer_cell_publish(&mut self, on: bool) {
        self.defer_cell_publish = on;
    }

    /// Drain the cell updates recorded while deferral was on, in
    /// finalization order (a later entry for the same id supersedes an
    /// earlier one).
    pub fn take_deferred_cells(&mut self) -> Vec<(HirId, usize)> {
        std::mem::take(&mut self.deferred_cells)
    }

    /// Publish `ptr` as `id`'s current entry in this backend's cell.
    pub fn publish_call_target(&self, id: HirId, ptr: usize) {
        crate::reload::set_call_target(self.reload_key, id, ptr);
    }

    /// Set the bead id for subsequent `compile_function` calls. Embedded
    /// as a constant in tier-0 probe sites so a back-edge loads the right
    /// helper slot.
    pub fn set_compile_bead_id(&mut self, bead_id: u64) {
        self.compile_bead_id = bead_id;
    }

    /// Supply per-function bead ids for whole-module compiles, where one
    /// `compile_bead_id` cannot be right for every function. Falls back to
    /// [`Self::set_compile_bead_id`] for functions absent from the map.
    pub fn set_bead_ids(&mut self, bead_ids: HashMap<HirId, u64>) {
        self.bead_ids = bead_ids;
    }

    /// Enable / disable OSR back-edge probe emission for subsequent
    /// `compile_function` calls. When enabled, each loop header loads the
    /// bead's arm byte and branches to a `__zyntax_osr_probe` call only
    /// once a tier ≥ 1 compile has installed helpers for that bead.
    pub fn set_emit_osr_probes(&mut self, enabled: bool) {
        self.emit_osr_probes = enabled;
    }

    /// Restrict subsequent [`Self::compile_module`] calls to only emit
    /// bodies for functions whose `HirId` is in `allowed`. Pass `None` to
    /// restore the default behaviour of compiling everything.
    ///
    /// Declarations are always emitted for every function in the module —
    /// only function-body codegen is gated. This keeps cross-function
    /// symbol resolution intact for the BC interpreter's lazy fallback.
    pub fn set_only_compile_reachable(&mut self, allowed: Option<HashSet<HirId>>) {
        self.only_compile_reachable = allowed;
    }

    /// Whether OSR back-edge probes will be emitted at tier 0.
    pub fn emit_osr_probes(&self) -> bool {
        self.emit_osr_probes
    }

    /// Tier currently configured for compilation.
    pub fn compile_tier(&self) -> usize {
        self.compile_tier
    }

    /// Bead id currently configured for compilation.
    pub fn compile_bead_id(&self) -> u64 {
        self.compile_bead_id
    }

    /// Register symbol signatures for auto-boxing support
    pub fn register_symbol_signatures(&mut self, symbols: &[crate::zrtl::RuntimeSymbolInfo]) {
        log::info!(
            "[DynamicBox] Registering {} symbol signatures",
            symbols.len()
        );
        for sym in symbols {
            if let Some(sig) = &sym.sig {
                log::debug!(
                    "[DynamicBox] Registering signature for {}: params={}, dynamic_params={:?}",
                    sym.name,
                    sig.param_count,
                    (0..sig.param_count)
                        .filter(|&i| sig.param_is_dynamic(i as usize))
                        .collect::<Vec<_>>()
                );
                self.symbol_signatures
                    .insert(sym.name.to_string(), sig.clone());
            }
        }
        log::info!(
            "[DynamicBox] Registered {} signatures total",
            self.symbol_signatures.len()
        );
    }

    /// Check if a symbol parameter expects DynamicBox
    fn param_needs_boxing(&self, symbol_name: &str, param_index: usize) -> bool {
        self.symbol_signatures
            .get(symbol_name)
            .map(|sig| sig.param_is_dynamic(param_index))
            .unwrap_or(false)
    }

    /// Compile a HIR module to native code
    pub fn compile_module(&mut self, module: &HirModule) -> CompilerResult<()> {
        // Dump HIR if ZYNTAX_DUMP_HIR is set
        if std::env::var("ZYNTAX_DUMP_HIR").is_ok() {
            let dump = crate::hir_dump::dump_module(module);
            log::trace!("{}", dump);
        }

        // Two-pass function compilation:
        // Pass 1: Declare all functions (populate function_map)
        // Pre-scan: Build map of call-site-inferred param types for extern functions
        // with 0-param placeholder signatures. This must happen BEFORE declaration so
        // the first Cranelift declaration is already correct.
        self.prescan_extern_call_sites(module);

        // Before anything is declared: which functions may be reached
        // through a pointer, and so must keep the convention their
        // pointer type implies.
        self.note_address_taken(module);

        for (id, function) in &module.functions {
            self.declare_function(*id, function, module)?;
        }

        // Process globals AFTER declaring functions. Vtable / effect
        // op-table globals materialise fn-pointer arrays via
        // `write_function_addr`, which needs every referenced function
        // already in `function_map` (declaration is enough — bodies
        // compile in Pass 2). Compiling globals before declaration left
        // those relocations unresolved (null slots).
        for (id, global) in &module.globals {
            self.compile_global(*id, global)?;
        }

        // Pass 2: Compile all function bodies
        for (id, function) in &module.functions {
            if std::env::var("ZYNTAX_TRACE_CRANELIFT_SKIP").is_ok() {
                eprintln!(
                    "[pass2] {:?} {} external={}",
                    id,
                    function.name.resolve_global().unwrap_or_default(),
                    function.is_external
                );
            }
            if !function.is_external {
                // Reachability DCE: when a caller has restricted the set of
                // functions we should compile, skip bodies outside that set.
                // The BC interp keeps lazy compile available, so any
                // unexpected call still works — just at interp speed.
                if let Some(allowed) = &self.only_compile_reachable {
                    if !allowed.contains(id) {
                        continue;
                    }
                }
                // Skip functions that fail to compile (e.g., signature mismatches with ZRTL)
                let body_result = self.compile_function_body(*id, function, module);
                if std::env::var("ZYNTAX_TRACE_CRANELIFT_SKIP").is_ok() {
                    eprintln!("[pass2] {:?} result_err={}", id, body_result.is_err());
                }
                if let Err(e) = body_result {
                    let name = function.name.resolve_global().unwrap_or_default();
                    // A function this backend has no encoding for is
                    // declined, not failed: it is recorded so a caller
                    // can hand it to a backend that does, and it is kept
                    // out of the defect count so a real defect is not
                    // buried among capability differences.
                    if e.is_unsupported_by_backend() {
                        CRANELIFT_DECLINED_FUNCTIONS.fetch_add(1, Ordering::Relaxed);
                        self.declined_functions.insert(*id);
                        log::debug!("[CRANELIFT] Declining function '{}': {}", name, e);
                        if std::env::var("ZYNTAX_TRACE_CRANELIFT_SKIP").is_ok() {
                            eprintln!("[CRANELIFT-DECLINE] '{name}': {e}");
                        }
                    } else {
                        CRANELIFT_SKIPPED_FUNCTIONS.fetch_add(1, Ordering::Relaxed);
                        log::debug!("[CRANELIFT] Skipping function '{}': {:?}", function.name, e);
                        if std::env::var("ZYNTAX_TRACE_CRANELIFT_SKIP").is_ok() {
                            eprintln!("[CRANELIFT-SKIP] '{name}': {e:?}");
                        }
                    }
                    // Remove from function_map to prevent later lookup failures
                    self.function_map.remove(id);
                }
            }
        }

        // Finalize the module
        let _ = self.module.finalize_definitions();

        // Update function pointers after finalization
        for (hir_id, compiled_func) in &self.compiled_functions {
            let code_ptr = self
                .module
                .get_finalized_function(compiled_func.function_id);
            if std::env::var("ZYNTAX_TRACE_CRANELIFT_SKIP").is_ok() {
                eprintln!("[ptrs] {hir_id:?} -> {code_ptr:?}");
            }
            self.hot_reload
                .function_pointers
                .write()
                .unwrap()
                .insert(*hir_id, code_ptr);
            if self.defer_cell_publish {
                self.deferred_cells.push((*hir_id, code_ptr as usize));
            } else {
                crate::reload::set_call_target(self.reload_key, *hir_id, code_ptr as usize);
            }
        }

        // Note: Symbols are NOT automatically exported for cross-module linking.
        // Use export_function() or export_functions() to explicitly export symbols.

        Ok(())
    }

    /// Export a compiled function for cross-module linking
    ///
    /// Returns an error if the function doesn't exist or if there's a symbol conflict.
    pub fn export_function(&mut self, name: &str) -> CompilerResult<()> {
        // Check for existing export conflict
        if let Some(existing_ptr) = self.exported_symbols.get(name) {
            // Find the current function pointer for this name
            let current_ptr = self
                .hot_reload
                .function_pointers
                .read()
                .unwrap()
                .values()
                .find(|&&p| p == *existing_ptr)
                .copied();

            if current_ptr.is_some() {
                return Err(CompilerError::Backend(format!(
                    "Symbol conflict: '{}' is already exported. Use a different name or unexport the existing symbol.",
                    name
                )));
            }
        }

        // Find the function by name in the function pointers
        let ptr = {
            let ptrs = self.hot_reload.function_pointers.read().unwrap();
            // We need to find the HirId for this function name
            // The function_map maps HirId -> FuncId, so we iterate compiled_functions
            let mut found_ptr = None;
            for (hir_id, _) in &self.compiled_functions {
                if let Some(ptr) = ptrs.get(hir_id) {
                    found_ptr = Some(*ptr);
                    break;
                }
            }
            found_ptr
        };

        // Note: The above search is not ideal because we're not tracking name->HirId mapping here.
        // The runtime layer should provide the function pointer directly.

        if let Some(ptr) = ptr {
            self.exported_symbols.insert(name.to_string(), ptr);
            Ok(())
        } else {
            Err(CompilerError::Backend(format!(
                "Function '{}' not found in compiled functions",
                name
            )))
        }
    }

    /// Export a function with an explicit function pointer
    ///
    /// This is the preferred method as it avoids lookup issues.
    /// Returns an error if there's a symbol conflict and `allow_overwrite` is false.
    pub fn export_function_ptr(&mut self, name: &str, ptr: *const u8) -> CompilerResult<()> {
        self.export_function_ptr_internal(name, ptr, false)
    }

    /// Export a function, allowing overwrite if the symbol already exists
    ///
    /// Returns the old pointer if it was overwritten.
    pub fn export_function_ptr_overwrite(
        &mut self,
        name: &str,
        ptr: *const u8,
    ) -> Option<*const u8> {
        let old = self.exported_symbols.insert(name.to_string(), ptr);
        old
    }

    /// Internal export with configurable overwrite behavior
    fn export_function_ptr_internal(
        &mut self,
        name: &str,
        ptr: *const u8,
        allow_overwrite: bool,
    ) -> CompilerResult<()> {
        // Check for existing export conflict
        if !allow_overwrite && self.exported_symbols.contains_key(name) {
            return Err(CompilerError::Backend(format!(
                "Symbol conflict: '{}' is already exported. Use a different name or unexport the existing symbol.",
                name
            )));
        }

        self.exported_symbols.insert(name.to_string(), ptr);
        Ok(())
    }

    /// Check if exporting a function would cause a symbol conflict
    ///
    /// Returns Some(existing_ptr) if a conflict exists, None otherwise.
    pub fn check_export_conflict(&self, name: &str) -> Option<*const u8> {
        self.exported_symbols.get(name).copied()
    }

    /// Record which of `module`'s functions have an observable address.
    ///
    /// Accumulates rather than replaces. A function compiled one at a
    /// time is seen through whichever module view the caller had, and a
    /// later view that happens not to mention an address already taken
    /// must not un-take it: the convention was already fixed when the
    /// function was declared.
    fn note_address_taken(&mut self, module: &HirModule) {
        self.address_taken
            .extend(crate::dce::address_taken_functions(module));
    }

    /// Declare a function signature without compiling its body
    fn declare_function(
        &mut self,
        id: HirId,
        function: &HirFunction,
        module: &HirModule,
    ) -> CompilerResult<()> {
        let mut sig = self.translate_signature_for(id, function)?;
        // Record what that signature committed to, so call sites and the
        // body agree with it rather than each deriving it again.
        if let Some(bytes) = self
            .destination_returns
            .get(&id)
            .copied()
            .or_else(|| self.returns_through_destination(id, function))
        {
            self.destination_returns.insert(id, bytes);
        }

        if function.is_external {
            // External functions use Import linkage
            // Use link_name if specified (maps alias to actual symbol, e.g., "image_load" -> "$Image$load")
            // Otherwise fall back to function name
            let link_name = function
                .link_name
                .as_ref()
                .map(|s| s.clone())
                .unwrap_or_else(|| {
                    function
                        .name
                        .resolve_global()
                        .unwrap_or_else(|| format!("{:?}", function.name))
                });

            // If the HIR has a placeholder signature (0 params from import resolution),
            // override with the real signature from either:
            // 1. ZRTL plugin registration (symbol_signatures), or
            // 2. Call-site inferred types (inferred_extern_sigs)
            if function.signature.params.is_empty() {
                let func_name_str = function.name.resolve_global().unwrap_or_default();
                let real_sig = self
                    .symbol_signatures
                    .get(&link_name)
                    .or_else(|| self.symbol_signatures.get(&func_name_str));
                if let Some(zrtl_sig) = real_sig {
                    // Use ZRTL plugin signature
                    let base_call_conv = sig.call_conv;
                    sig = self.module.make_signature();
                    sig.call_conv = base_call_conv;
                    for i in 0..zrtl_sig.param_count as usize {
                        let ty = type_tag_to_cranelift_type(&zrtl_sig.params[i]);
                        sig.params.push(AbiParam::new(ty));
                    }
                    let ret_ty = type_tag_to_cranelift_type(&zrtl_sig.return_type);
                    if ret_ty != types::I8
                        || !zrtl_sig
                            .return_type
                            .is_category(crate::zrtl::TypeCategory::Void)
                    {
                        sig.returns.push(AbiParam::new(ret_ty));
                    }
                } else if let Some((param_hir_types, ret_hir_type)) =
                    self.inferred_extern_sigs.get(&id).cloned()
                {
                    // Use call-site inferred types
                    let base_call_conv = sig.call_conv;
                    sig = self.module.make_signature();
                    sig.call_conv = base_call_conv;
                    for hir_ty in &param_hir_types {
                        let ty = self.translate_type(hir_ty)?;
                        sig.params.push(AbiParam::new(ty));
                    }
                    // Use inferred return type if available, else keep original
                    if let Some(ret_ty) = &ret_hir_type {
                        if *ret_ty != HirType::Void {
                            let ty = self.translate_type(ret_ty)?;
                            sig.returns.push(AbiParam::new(ty));
                        }
                    } else {
                        // Preserve original returns
                        for ret_ty in &function.signature.returns {
                            if *ret_ty != HirType::Void {
                                let ty = self.translate_type(ret_ty)?;
                                sig.returns.push(AbiParam::new(ty));
                            }
                        }
                    }
                }
            }

            log::debug!(
                "[Cranelift] Declaring external function: {:?} -> '{}' (params: {})",
                function.name,
                link_name,
                sig.params.len()
            );

            let func_id = match self
                .module
                .declare_function(&link_name, Linkage::Import, &sig)
            {
                Ok(fid) => fid,
                Err(_) => {
                    // Declaration failed due to incompatible signature — another
                    // HirFunction already declared this symbol (e.g., both
                    // `tensor_sqrt` and `$Tensor$sqrt` map to the same ZRTL symbol).
                    // Reuse the existing FuncId.
                    if let Some((&existing_hir_id, _)) = self
                        .external_link_names
                        .iter()
                        .find(|(_, name)| **name == link_name)
                    {
                        *self.function_map.get(&existing_hir_id).unwrap()
                    } else {
                        return Err(CompilerError::Backend(format!(
                            "Failed to declare extern function '{}' and no prior declaration found",
                            link_name
                        )));
                    }
                }
            };

            self.function_map.insert(id, func_id);
            // Store link_name for external function boxing support
            self.external_link_names.insert(id, link_name.clone());

            log::debug!(
                "[Cranelift] External function registered: '{}' with HirId {:?} -> FuncId {:?}",
                link_name,
                id,
                func_id
            );
        } else {
            // Regular functions use Export linkage with unique name
            // Use resolve_global to get the actual function name
            let base_name = function
                .name
                .resolve_global()
                .unwrap_or_else(|| format!("{:?}", function.name));

            // Recompiling a function that already has a definition (tier-up,
            // hot reload) must produce a *new* symbol: Cranelift refuses a
            // second `define_function` on the same `FuncId`, and the old code
            // has to stay resident because frames may still be executing it.
            // Each redefinition therefore gets its own generation suffix.
            let generation = if self.compiled_functions.contains_key(&id) {
                let next = self.compile_generation.entry(id).or_insert(0);
                *next += 1;
                *next
            } else {
                *self.compile_generation.entry(id).or_insert(0)
            };
            let unique_name = if generation == 0 {
                format!("{}__{:?}", base_name, id)
            } else {
                format!("{}__{:?}__g{}", base_name, id, generation)
            };

            let func_id = self
                .module
                .declare_function(&unique_name, Linkage::Export, &sig)
                .map_err(|e| {
                    CompilerError::Backend(format!("Failed to declare function: {}", e))
                })?;

            self.function_map.insert(id, func_id);
        }

        Ok(())
    }

    /// Pre-scan call sites to infer param types for extern functions with 0-param placeholders.
    /// Must be called BEFORE declare_function pass so the first declaration is already correct.
    fn prescan_extern_call_sites(&mut self, module: &HirModule) {
        // Collect HirIds of extern functions with 0 params
        let mut zero_param_externs: HashSet<HirId> = HashSet::new();
        for (id, function) in &module.functions {
            if function.is_external && function.signature.params.is_empty() {
                zero_param_externs.insert(*id);
            }
        }

        if zero_param_externs.is_empty() {
            return;
        }

        // Scan all function bodies for calls to these extern functions
        for (_caller_id, caller_func) in &module.functions {
            if caller_func.is_external {
                continue;
            }
            for block in caller_func.blocks.values() {
                for inst in &block.instructions {
                    if let HirInstruction::Call {
                        callee: HirCallable::Function(func_id),
                        args,
                        result,
                        ..
                    } = inst
                    {
                        if zero_param_externs.contains(func_id)
                            && !args.is_empty()
                            && !self.inferred_extern_sigs.contains_key(func_id)
                        {
                            // Infer param types from call arguments' HIR types
                            let param_types: Vec<HirType> = args
                                .iter()
                                .map(|arg_id| {
                                    caller_func
                                        .values
                                        .get(arg_id)
                                        .map(|v| v.ty.clone())
                                        .unwrap_or(HirType::I64)
                                })
                                .collect();

                            // Infer return type from call result
                            let ret_type = result
                                .and_then(|r| caller_func.values.get(&r).map(|v| v.ty.clone()));

                            self.inferred_extern_sigs
                                .insert(*func_id, (param_types, ret_type));
                        }
                    }
                }
            }
        }
    }

    /// Compile a single function with hot-reload support (legacy, calls declare + compile_body)
    ///
    /// Note: This legacy path does not support algebraic effects. Use compile_module() for
    /// full effect support.
    pub fn compile_function(&mut self, id: HirId, function: &HirFunction) -> CompilerResult<()> {
        let empty_module =
            HirModule::new(zyntax_typed_ast::InternedString::new_global("__legacy__"));
        self.compile_function_in_module(id, function, &empty_module)
    }

    /// Declare `function` without compiling its body, so data
    /// definitions that take its address (a dispatch table) and other
    /// functions that call it can be emitted before it is compiled.
    pub fn declare_function_only(
        &mut self,
        id: HirId,
        function: &HirFunction,
        module: &HirModule,
    ) -> CompilerResult<()> {
        self.declare_function(id, function, module)
    }

    /// [`Self::compile_function`] against a module, so codegen that
    /// consults module-level declarations — an effect's operation
    /// order at a perform site, for one — resolves against the running
    /// program rather than an empty stand-in.
    pub fn compile_function_in_module(
        &mut self,
        id: HirId,
        function: &HirFunction,
        module: &HirModule,
    ) -> CompilerResult<()> {
        log::trace!("[Backend] compile_function called for {:?}", id);
        self.note_address_taken(module);
        self.declare_function(id, function, module)?;
        log::trace!(
            "[Backend] After declare_function, IR:\n{}",
            self.codegen_context.func
        );
        if !function.is_external {
            self.compile_function_body(id, function, module)?;
            log::trace!(
                "[Backend] After compile_function_body, IR:\n{}",
                self.codegen_context.func
            );

            // Tier ≥ 1: emit one OSR helper per accepted layout. Stub
            // bodies for now (increment 5a) — increment 5b will replace
            // them with full loop-body emission.
            if self.compile_tier >= 1 {
                self.compile_osr_helpers(id, function)?;
            }
        }
        Ok(())
    }

    /// Emit OSR helpers for tier-≥1 compilation of `function`. For each
    /// loop header with an acceptable [`crate::osr::OsrLayout`], declare a
    /// fresh Cranelift function with signature
    /// `(i64,i64,i64,i64) -> <return_type>` and drive
    /// [`Self::compile_function_body`] to emit the loop body, entered at
    /// the header. Stores `(site_key, FuncId)` pairs in
    /// `pending_osr_helpers` for the caller to drain after
    /// `finalize_definitions`.
    fn compile_osr_helpers(&mut self, id: HirId, function: &HirFunction) -> CompilerResult<()> {
        let headers = crate::osr::find_loop_headers(function);
        if headers.is_empty() {
            return Ok(());
        }

        let bead_id = self.compile_bead_id;
        let trace = std::env::var_os("ZYNTAX_OSR_TRACE").is_some();

        for header in headers {
            let layout = match crate::osr::osr_layout(function, header) {
                Ok(l) => l,
                Err(reason) => {
                    if trace {
                        eprintln!(
                            "[osr] reject helper for bead={} header_idx={}: {:?}",
                            bead_id,
                            crate::osr::block_index_of(function, header).unwrap_or(u64::MAX),
                            reason
                        );
                    }
                    continue;
                }
            };

            // Compile the helper as a separate Cranelift function. If the
            // body emission fails (e.g. an instruction case the lowering
            // doesn't yet handle in helper mode), log and skip this layout
            // — the main function still works, we just don't OSR for this
            // header.
            if let Err(e) = self.emit_osr_helper_via_body(id, function, &layout) {
                if trace {
                    eprintln!(
                        "[osr] helper emission failed for bead={} header_idx={}: {}",
                        bead_id, layout.loop_ordinal, e
                    );
                }
                // Reset OSR-helper state in case it was partially set.
                self.compile_osr_layout = None;
                self.compile_osr_func_id = None;
                self.codegen_context.clear();
                self.value_map.clear();
                self.block_map.clear();
                continue;
            }
        }

        Ok(())
    }

    /// Compile a single OSR helper by reusing
    /// [`Self::compile_function_body`] under helper mode. Saves and
    /// restores any auxiliary state that mode mutates so the caller's
    /// view of the backend is unchanged on success.
    fn emit_osr_helper_via_body(
        &mut self,
        id: HirId,
        function: &HirFunction,
        layout: &crate::osr::OsrLayout,
    ) -> CompilerResult<()> {
        // One pointer to the frame carrying the live-ins, matching what
        // `compile_function_body` builds the definition with and what
        // the back-edge probe calls through. Declaring a parameter per
        // possible live-in described the earlier design, where they
        // travelled in registers; against a definition taking one
        // pointer it leaves the declared and defined signatures
        // disagreeing about the whole argument area, which an ABI that
        // reserves caller space for it does not survive.
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        if !matches!(layout.return_type, HirType::Void) {
            let ret_clir = self.translate_type(&layout.return_type)?;
            sig.returns.push(AbiParam::new(ret_clir));
        }

        // Generation-suffixed like entry symbols: a recompile — promotion
        // or reload — declares fresh helpers rather than colliding with
        // the previous generation's definitions.
        let generation = self.compile_generation.get(&id).copied().unwrap_or(0);
        let helper_name = format!(
            "__zyntax_osr_{}_{}_g{}",
            self.compile_bead_id, layout.loop_ordinal, generation
        );
        let func_id = self
            .module
            .declare_function(&helper_name, Linkage::Local, &sig)
            .map_err(|e| {
                CompilerError::Backend(format!("declare OSR helper {}: {}", helper_name, e))
            })?;

        // Drive compile_function_body in helper mode.
        let prev_layout = self.compile_osr_layout.take();
        let prev_func_id = self.compile_osr_func_id.take();
        self.compile_osr_layout = Some(layout.clone());
        self.compile_osr_func_id = Some(func_id);

        // Empty hir_module — helpers don't need cross-module info; the
        // main function compile already populated trait dispatch state.
        let empty_module = HirModule::new(zyntax_typed_ast::InternedString::new_global("__osr__"));
        let result = self.compile_function_body(id, function, &empty_module);

        self.compile_osr_layout = prev_layout;
        self.compile_osr_func_id = prev_func_id;

        result?;

        self.pending_osr_helpers.push((layout.site_key(), func_id));

        if std::env::var_os("ZYNTAX_OSR_TRACE").is_some() {
            eprintln!(
                "[osr] declared helper {} site=0x{:x}",
                helper_name,
                layout.site_key()
            );
        }

        Ok(())
    }

    /// Build a single stub OSR helper. Sig: `(i64,i64,i64,i64) -> ReturnType`
    /// where ReturnType is the function's HIR return type translated to a
    /// Cranelift type. The body takes the four params, immediately returns
    /// Compile just the body of a function (assumes signature already declared)
    fn compile_function_body(
        &mut self,
        id: HirId,
        function: &HirFunction,
        hir_module: &HirModule,
    ) -> CompilerResult<()> {
        // Address-taking sites use this to spot self-references, which
        // pin to this generation instead of the reload cell.
        self.current_compile_id = Some(id);
        // Reset per-function scratch state on every entry. The success
        // path clears these at the *end* of the function (~line 4784);
        // the OSR helper error path clears them inline. But any error
        // returning via `?` between `FunctionBuilder::new`
        // (~line 1365) and `builder.finalize()` (~line 4705) leaves
        // them dirty. The next call to this function then crashes
        // inside Cranelift instead of producing a clean compiler
        // error — `FunctionBuilder::new` asserts its context is
        // empty: `assertion failed: func_ctx.is_empty()`.
        //
        // Resetting at entry is cheap and decouples error paths from
        // the next call's correctness — any future early-return is
        // automatically safe.
        self.builder_context = FunctionBuilderContext::new();
        self.codegen_context.clear();
        self.value_map.clear();
        self.block_map.clear();

        // Get the already-declared function ID. In OSR-helper mode the
        // FuncId comes from `compile_osr_func_id` (set by the helper
        // emission code) since the helper isn't in `function_map`.
        let func_id = if self.compile_osr_layout.is_some() {
            self.compile_osr_func_id.ok_or_else(|| {
                CompilerError::Backend(
                    "OSR helper compile started without compile_osr_func_id set".into(),
                )
            })?
        } else {
            *self
                .function_map
                .get(&id)
                .ok_or_else(|| CompilerError::Backend(format!("Function {:?} not declared", id)))?
        };

        // OSR-helper compile mode: signature, params, and block iteration
        // are all overridden. Captured up-front so later borrows of `self`
        // don't conflict with `self.compile_osr_layout`.
        let osr_helper = self.compile_osr_layout.clone();

        let sig = if let Some(layout) = &osr_helper {
            // One pointer to the frame carrying the live-ins.
            let mut s = self.module.make_signature();
            s.params.push(AbiParam::new(types::I64));
            if !matches!(layout.return_type, HirType::Void) {
                let ret_ty = self.translate_type(&layout.return_type)?;
                s.returns.push(AbiParam::new(ret_ty));
            }
            s
        } else {
            self.translate_signature_for(id, function)?
        };

        // Whether this body writes its result through a destination its
        // caller passed. An OSR helper takes only the frame pointer and
        // is never on that convention, which is also why a function that
        // is gets no probes (see `osr_layouts` below). Read from what was
        // recorded at declaration rather than derived again, so the body
        // cannot disagree with the signature callers were emitted against.
        let destination_return = if osr_helper.is_some() {
            None
        } else {
            self.destination_returns.get(&id).copied()
        };

        // Pre-calculate parameter types before creating builder
        let param_types: Vec<_> = if osr_helper.is_some() {
            vec![types::I64]
        } else {
            let pt: Result<Vec<_>, _> = function
                .signature
                .params
                .iter()
                .map(|param| self.translate_type(&param.ty))
                .collect();
            let mut pt = pt?;
            // Matches the leading destination `translate_signature_for`
            // put on the signature.
            if destination_return.is_some() {
                pt.insert(0, self.module.target_config().pointer_type());
            }
            pt
        };

        // Pre-translate ALL types used in the function to avoid borrow checker issues
        // Build a cache of HirType -> cranelift::Type mappings
        // and HirType -> size (in bytes) for GEP calculations
        // and struct layouts for GEP field access
        // and constant values for GEP struct field indices
        let mut type_cache: HashMap<HirType, cranelift_codegen::ir::Type> = HashMap::new();
        let mut size_cache: HashMap<HirType, usize> = HashMap::new();
        let mut struct_layout_cache: HashMap<crate::hir::HirStructType, StructLayout> =
            HashMap::new();

        // Build a cache of constant integer values (for struct field indices)
        let mut const_value_cache: HashMap<HirId, i64> = HashMap::new();
        for (value_id, value) in &function.values {
            if let HirValueKind::Constant(constant) = &value.kind {
                let int_val = match constant {
                    HirConstant::I8(v) => Some(*v as i64),
                    HirConstant::I16(v) => Some(*v as i64),
                    HirConstant::I32(v) => Some(*v as i64),
                    HirConstant::I64(v) => Some(*v),
                    HirConstant::U8(v) => Some(*v as i64),
                    HirConstant::U16(v) => Some(*v as i64),
                    HirConstant::U32(v) => Some(*v as i64),
                    HirConstant::U64(v) => Some(*v as i64),
                    _ => None,
                };
                if let Some(val) = int_val {
                    const_value_cache.insert(*value_id, val);
                }
            }
        }

        // Build a cache of value types (for ExtractValue/InsertValue to know aggregate types)
        let mut value_type_cache: HashMap<HirId, HirType> = HashMap::new();
        for (value_id, value) in &function.values {
            value_type_cache.insert(*value_id, value.ty.clone());
        }

        for (_block_id, block) in &function.blocks {
            // Pre-translate phi types
            for phi in &block.phis {
                if let Ok(cranelift_ty) = self.translate_type(&phi.ty) {
                    type_cache.insert(phi.ty.clone(), cranelift_ty);
                }
            }

            // Pre-translate instruction types
            for inst in &block.instructions {
                match inst {
                    HirInstruction::ExtractValue { ty, aggregate, .. } => {
                        if let Ok(cranelift_ty) = self.translate_type(ty) {
                            type_cache.insert(ty.clone(), cranelift_ty);
                        }
                        // Cache struct layouts from the aggregate type
                        if let Some(agg_value) = function.values.get(aggregate) {
                            self.cache_struct_layouts_recursive(
                                &agg_value.ty,
                                &mut struct_layout_cache,
                                &mut size_cache,
                            );
                        }
                    }
                    HirInstruction::InsertValue { ty, aggregate, .. } => {
                        if let Ok(cranelift_ty) = self.translate_type(ty) {
                            type_cache.insert(ty.clone(), cranelift_ty);
                        }
                        // Cache struct layouts from the aggregate type
                        if let Some(agg_value) = function.values.get(aggregate) {
                            self.cache_struct_layouts_recursive(
                                &agg_value.ty,
                                &mut struct_layout_cache,
                                &mut size_cache,
                            );
                        }
                    }
                    HirInstruction::Binary { ty, .. } => {
                        if let Ok(cranelift_ty) = self.translate_type(ty) {
                            type_cache.insert(ty.clone(), cranelift_ty);
                        }
                    }
                    HirInstruction::Cast { ty, .. } => {
                        if let Ok(cranelift_ty) = self.translate_type(ty) {
                            type_cache.insert(ty.clone(), cranelift_ty);
                        }
                    }
                    HirInstruction::ExtractUnionValue { ty, .. } => {
                        if let Ok(cranelift_ty) = self.translate_type(ty) {
                            type_cache.insert(ty.clone(), cranelift_ty);
                        }
                    }
                    HirInstruction::Alloca { ty, .. } => {
                        if let Ok(cranelift_ty) = self.translate_type(ty) {
                            type_cache.insert(ty.clone(), cranelift_ty);
                        }
                        if let Ok(size) = self.type_size(ty) {
                            size_cache.insert(ty.clone(), size);
                        }
                    }
                    HirInstruction::Load { ty, .. } => {
                        if let Ok(cranelift_ty) = self.translate_type(ty) {
                            type_cache.insert(ty.clone(), cranelift_ty);
                        }
                        // Aggregate-typed Loads memcpy into a fresh
                        // stack slot — seed `size_cache` so the
                        // codegen path has the byte count without
                        // borrowing &self while FunctionBuilder is live.
                        if matches!(
                            ty,
                            HirType::Struct(_) | HirType::Array(_, _) | HirType::Union(_)
                        ) {
                            if let Ok(size) = self.type_size(ty) {
                                size_cache.insert(ty.clone(), size);
                            }
                        }
                    }
                    HirInstruction::Store { value, .. } => {
                        // Aggregate-valued Stores get rewritten to a
                        // memcpy below — seed `size_cache` so that
                        // code path can look up the struct's size
                        // without borrowing &self while the
                        // FunctionBuilder is live.
                        if let Some(v) = function.values.get(value) {
                            if matches!(
                                v.ty,
                                HirType::Struct(_) | HirType::Array(_, _) | HirType::Union(_)
                            ) {
                                if let Ok(size) = self.type_size(&v.ty) {
                                    size_cache.insert(v.ty.clone(), size);
                                }
                            }
                        }
                    }
                    HirInstruction::GetElementPtr { ty, .. } => {
                        // Pre-compute sizes for all types in GEP chain
                        // We need to walk through the type structure
                        if let Ok(size) = self.type_size(ty) {
                            size_cache.insert(ty.clone(), size);
                        }
                        // Also cache sizes for nested types (Ptr, Array, Struct fields)
                        match ty {
                            HirType::Ptr(inner) => {
                                if let Ok(size) = self.type_size(inner) {
                                    size_cache.insert((**inner).clone(), size);
                                }
                            }
                            HirType::Array(elem_ty, _) => {
                                if let Ok(size) = self.type_size(elem_ty) {
                                    size_cache.insert((**elem_ty).clone(), size);
                                }
                            }
                            HirType::Struct(struct_ty) => {
                                for field_ty in &struct_ty.fields {
                                    if let Ok(size) = self.type_size(field_ty) {
                                        size_cache.insert(field_ty.clone(), size);
                                    }
                                }
                                // Pre-compute struct layout
                                if let Ok(layout) = self.calculate_struct_layout(struct_ty) {
                                    struct_layout_cache.insert(struct_ty.clone(), layout);
                                }
                            }
                            _ => {}
                        }
                    }
                    HirInstruction::VectorSplat { ty, .. }
                    | HirInstruction::VectorExtractLane { ty, .. }
                    | HirInstruction::VectorInsertLane { ty, .. }
                    | HirInstruction::VectorHorizontalReduce { ty, .. }
                    | HirInstruction::VectorLoad { ty, .. } => {
                        if let Ok(cranelift_ty) = self.translate_type(ty) {
                            type_cache.insert(ty.clone(), cranelift_ty);
                        }
                    }
                    _ => {} // Add more as needed
                }
            }
        }

        // Build function body
        log::debug!("[Cranelift] Building function body with signature:");
        log::debug!("[Cranelift]   sig.returns = {:?}", sig.returns);
        self.codegen_context.func = cranelift_codegen::ir::Function::with_name_signature(
            UserFuncName::user(0, func_id.as_u32()),
            sig.clone(),
        );

        // `(site_key, srcloc_tag)` for every probe emitted in this function;
        // resolved to code offsets after `define_function` below.
        let mut probe_site_tags: Vec<(u64, u32)> = Vec::new();

        {
            // ================================================================
            // MULTI-BLOCK CONTROL FLOW IMPLEMENTATION
            // ================================================================

            // Phase 1: Analyze function structure
            let block_order = self.compute_block_order(function);
            // In OSR-helper mode, restrict to blocks reachable from the
            // header — the function entry and pre-loop blocks are dead
            // code in the helper and would reference HirIds (function
            // params, etc.) that we don't put in `value_map`.
            let block_order: Vec<HirId> = if let Some(layout) = &osr_helper {
                let reachable: std::collections::HashSet<HirId> = {
                    let mut visited = std::collections::HashSet::new();
                    let mut stack = vec![layout.header];
                    while let Some(id) = stack.pop() {
                        if !visited.insert(id) {
                            continue;
                        }
                        if let Some(b) = function.blocks.get(&id) {
                            stack.extend(get_successors(&b.terminator));
                        }
                    }
                    visited
                };
                block_order
                    .into_iter()
                    .filter(|b| reachable.contains(b))
                    .collect()
            } else {
                block_order
            };

            // OSR pre-pass: identify loop headers in tier 0 only. Tier ≥ 1
            // emits OSR helpers (separate functions) and skips probes.
            // Additionally suppress when `emit_osr_probes` is false — the
            // embedder has declared no tier ≥ 1 backend will ever install
            // OSR helpers, so the probe stream is pure overhead.
            let osr_loop_headers: std::collections::HashSet<HirId> =
                if self.compile_tier == 0 && self.emit_osr_probes {
                    crate::osr::find_loop_headers(function)
                        .into_iter()
                        .collect()
                } else {
                    std::collections::HashSet::new()
                };
            // Stable per-function block index (matches `osr::block_index_of`).
            let osr_block_index: HashMap<HirId, u64> = function
                .blocks
                .keys()
                .enumerate()
                .map(|(i, id)| (*id, i as u64))
                .collect();
            let osr_bead_id = self
                .bead_ids
                .get(&id)
                .copied()
                .unwrap_or(self.compile_bead_id);
            // Pre-resolve per-header layouts and the function's return
            // Cranelift type. Doing this before the FunctionBuilder is
            // created avoids re-borrowing `self` during emission.
            //
            // A function that returns through a caller-provided
            // destination is left without probes. A probe finishes the
            // frame inside the helper and returns whatever the helper
            // returned, and the helper has no destination to write
            // through — it would hand back its own frame, which is the
            // thing the destination exists to avoid. Losing the transfer
            // costs speed on one shape; returning a dead frame is wrong.
            let osr_layouts: HashMap<HirId, crate::osr::OsrLayout> =
                if self.compile_tier == 0 && self.emit_osr_probes && destination_return.is_none() {
                    osr_loop_headers
                        .iter()
                        .filter_map(|h| crate::osr::osr_layout(function, *h).ok().map(|l| (*h, l)))
                        .collect()
                } else {
                    HashMap::new()
                };
            let osr_return_clir: Option<cranelift_codegen::ir::Type> =
                match function.signature.returns.as_slice() {
                    [] => None,
                    [ty] if !matches!(ty, HirType::Void) => self.translate_type(ty).ok(),
                    _ => None,
                };
            log::debug!("[Cranelift] Compiling function: {:?}", function.name);
            log::debug!("[Cranelift] Block order: {:?} blocks", block_order.len());
            log::debug!("[Cranelift] Entry block: {:?}", function.entry_block);
            log::debug!(
                "[Cranelift] Total blocks in function: {:?}",
                function.blocks.len()
            );
            log::debug!("[Cranelift] Values count: {:?}", function.values.len());
            for (i, block_id) in block_order.iter().enumerate() {
                if let Some(block) = function.blocks.get(block_id) {
                    log::debug!(
                        "[Cranelift]   [{}] {:?} - {} instructions, terminator: {:?}",
                        i,
                        block_id,
                        block.instructions.len(),
                        block.terminator
                    );
                    for (j, inst) in block.instructions.iter().enumerate() {
                        log::debug!("[Cranelift]     inst[{}]: {:?}", j, inst);
                    }
                } else {
                    log::debug!("[Cranelift]   [{}] {:?} - MISSING!", i, block_id);
                }
            }
            let predecessor_map = self.build_predecessor_map(function);

            // Pre-compute type mappings for global references to avoid borrow checker issues
            let mut global_types = HashMap::new();
            for value in function.values.values() {
                if let HirValueKind::Global(_) = value.kind {
                    let ptr_ty = self.translate_type(&value.ty)?;
                    global_types.insert(value.id, ptr_ty);
                }
            }

            // Pre-compute symbol parameter boxing requirements to avoid borrow checker issues
            let mut symbol_boxing: HashMap<(String, usize), bool> = HashMap::new();
            for block in function.blocks.values() {
                for inst in &block.instructions {
                    match inst {
                        HirInstruction::Call {
                            callee: HirCallable::Symbol(symbol_name),
                            args,
                            ..
                        } => {
                            for param_index in 0..args.len() {
                                let key = (symbol_name.clone(), param_index);
                                if !symbol_boxing.contains_key(&key) {
                                    symbol_boxing.insert(
                                        key,
                                        self.param_needs_boxing(symbol_name, param_index),
                                    );
                                }
                            }
                        }
                        HirInstruction::Call {
                            callee: HirCallable::Function(func_id),
                            args,
                            ..
                        } => {
                            // Also check external functions
                            if let Some(link_name) = self.external_link_names.get(func_id) {
                                for param_index in 0..args.len() {
                                    let key = (link_name.clone(), param_index);
                                    if !symbol_boxing.contains_key(&key) {
                                        symbol_boxing.insert(
                                            key,
                                            self.param_needs_boxing(link_name, param_index),
                                        );
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Pre-compute sizes for aggregate undef values (to avoid borrow checker issues later)
            let mut undef_aggregate_sizes: HashMap<HirId, u32> = HashMap::new();
            for value in function.values.values() {
                if let HirValueKind::Undef = value.kind {
                    match &value.ty {
                        HirType::Struct(_) | HirType::Array(_, _) | HirType::Union(_) => {
                            let alloc_size = self.type_size(&value.ty).unwrap_or(8) as u32;
                            let alloc_size = std::cmp::max(alloc_size, 8); // At least 8 bytes
                            undef_aggregate_sizes.insert(value.id, alloc_size);
                        }
                        _ => {}
                    }
                }
            }

            // Pre-compute pointer type for use inside builder scope
            let pointer_type = self.module.target_config().pointer_type();

            // Phase 2: Create builder and all Cranelift blocks
            let mut builder =
                FunctionBuilder::new(&mut self.codegen_context.func, &mut self.builder_context);

            let mut block_map = HashMap::new();

            // Create all Cranelift blocks and add phi node parameters
            for hir_block_id in &block_order {
                let cranelift_block = builder.create_block();
                block_map.insert(*hir_block_id, cranelift_block);

                // Add block parameters for phi nodes
                if let Some(hir_block) = function.blocks.get(hir_block_id) {
                    for phi in &hir_block.phis {
                        let phi_type = type_cache.get(&phi.ty).copied().unwrap_or(types::I64);
                        builder.append_block_param(cranelift_block, phi_type);
                    }
                }
            }

            // Phase 3: Setup entry block. Helper mode uses a synthetic
            // prologue that bit-casts the helper's i64 args to live-in
            // types and jumps to `layout.header`. Main mode appends the
            // function's params to the original entry block.
            let entry_block = if let Some(layout) = &osr_helper {
                // Synthetic prologue — fresh Cranelift block, NOT in block_map.
                let prologue = builder.create_block();
                for cranelift_type in &param_types {
                    builder.append_block_param(prologue, *cranelift_type);
                }
                prologue
            } else {
                block_map[&function.entry_block]
            };

            let entry_param_start = if osr_helper.is_some() {
                0
            } else {
                builder.block_params(entry_block).len()
            };
            if osr_helper.is_none() {
                // Add function parameters to entry block (in addition to any phi params)
                for cranelift_type in &param_types {
                    builder.append_block_param(entry_block, *cranelift_type);
                }
            }

            // Switch to entry block for parameter and constant setup
            // We need an active block to create constant instructions
            builder.switch_to_block(entry_block);
            if osr_helper.is_some() {
                // Synthetic prologue has only one predecessor (none) — seal
                // immediately so we can emit instructions without warnings.
                builder.seal_block(entry_block);
            }

            // A tier-0 function with a resumable loop says so on entry, so a
            // frame that never returns can still reach the tier worth
            // transferring into. Promotion is otherwise driven by
            // invocation count, which advances one tier per call. One call
            // per invocation, not per iteration; the runtime dedups.
            if osr_helper.is_none() && !osr_layouts.is_empty() {
                let mut sig = self.module.make_signature();
                sig.params.push(AbiParam::new(types::I64));
                if let Ok(fid) = self.module.declare_function(
                    crate::osr::OSR_REQUEST_SYMBOL,
                    Linkage::Import,
                    &sig,
                ) {
                    let f = self.module.declare_func_in_func(fid, builder.func);
                    let bead_v = builder.ins().iconst(types::I64, osr_bead_id as i64);
                    builder.ins().call(f, &[bead_v]);
                }
            }

            // Store block map for use in helper methods
            self.block_map = block_map.clone();

            // Map HIR function parameters / helper live-ins to Cranelift values.
            // In helper mode, capture the phi-typed casts here so we can
            // pass them as block-params on the prologue → header jump
            // emitted at the end of Phase 3.
            let mut osr_phi_jump_args: Vec<cranelift_codegen::ir::Value> = Vec::new();
            // Where a returned aggregate is written, when the caller
            // provided the memory for it.
            let mut destination_ptr: Option<cranelift_codegen::ir::Value> = None;
            if let Some(layout) = &osr_helper {
                // The sole argument is the frame's address. Recover each
                // live-in from its offset: a value held by reference is
                // already in the frame, so its address is the value; a
                // scalar is loaded out.
                let frame_ptr = builder.block_params(entry_block)[0];
                for (i, hir_id) in layout.live_ins.iter().enumerate() {
                    let Some(&offset) = layout.frame.offsets.get(i) else {
                        break;
                    };
                    let hir_ty = &layout.live_in_types[i];
                    let recovered = if crate::osr::is_held_by_reference(hir_ty) {
                        builder.ins().iadd_imm(frame_ptr, offset as i64)
                    } else {
                        let target = type_cache.get(hir_ty).copied().unwrap_or(types::I64);
                        builder
                            .ins()
                            .load(target, MemFlags::new(), frame_ptr, offset as i32)
                    };
                    if i < layout.phi_count {
                        osr_phi_jump_args.push(recovered);
                    } else {
                        // Non-phi live-in — value_map for body consumption.
                        self.value_map.insert(*hir_id, recovered);
                    }
                }
            } else {
                let entry_params = builder.block_params(entry_block);
                let function_params = &entry_params[entry_param_start..];

                // The destination leads the declared parameters, so it
                // is taken off before the rest are matched up by index.
                let function_params = if destination_return.is_some() {
                    destination_ptr = function_params.first().copied();
                    function_params.get(1..).unwrap_or(&[])
                } else {
                    function_params
                };

                // Get HIR parameter value IDs sorted by parameter index
                let mut param_value_ids = Vec::new();
                for value in function.values.values() {
                    if let HirValueKind::Parameter(param_index) = value.kind {
                        param_value_ids.push((param_index, value.id));
                    }
                }
                param_value_ids.sort_by_key(|(index, _)| *index);

                // Map HIR params to Cranelift values
                for (i, (_, hir_value_id)) in param_value_ids.iter().enumerate() {
                    if let Some(&cranelift_val) = function_params.get(i) {
                        self.value_map.insert(*hir_value_id, cranelift_val);
                    }
                }
            }

            // Map all constant values (needs active block for instruction creation)
            log::debug!("[Cranelift] Processing {} values", function.values.len());
            for value in function.values.values() {
                log::debug!("[Cranelift]   Value {:?} kind={:?}", value.id, value.kind);
                if let HirValueKind::Constant(constant) = &value.kind {
                    let cranelift_val = match constant {
                        // For narrow integer types, Cranelift expects zero-extended values, not sign-extended
                        // This avoids "immediate out of bounds" verifier errors for negative constants
                        // See: https://github.com/bytecodealliance/wasmtime/issues/9041
                        HirConstant::I8(v) => {
                            let extended = (*v as u8) as i64;
                            builder.ins().iconst(types::I8, extended)
                        }
                        HirConstant::I16(v) => {
                            let extended = (*v as u16) as i64;
                            builder.ins().iconst(types::I16, extended)
                        }
                        HirConstant::I32(v) => {
                            // Zero-extend i32 to i64 instead of sign-extending
                            let extended = (*v as u32) as i64;
                            builder.ins().iconst(types::I32, extended)
                        }
                        HirConstant::U32(v) => builder.ins().iconst(types::I32, *v as i64),
                        HirConstant::I64(v) => builder.ins().iconst(types::I64, *v),
                        HirConstant::U64(v) => builder.ins().iconst(types::I64, *v as i64),
                        HirConstant::Bool(v) => {
                            builder.ins().iconst(types::I8, if *v { 1 } else { 0 })
                        }
                        HirConstant::F32(v) => builder.ins().f32const(*v),
                        HirConstant::F64(v) => builder.ins().f64const(*v),
                        _ => continue, // Complex constants not yet supported
                    };
                    self.value_map.insert(value.id, cranelift_val);
                }
            }

            // Map global references (like string constants)
            for value in function.values.values() {
                if let HirValueKind::Global(global_id) = value.kind {
                    // Get the Cranelift data ID for this global
                    if let Some(&data_id) = self.global_map.get(&global_id) {
                        // Declare the data in the function context
                        let local_data =
                            self.module.declare_data_in_func(data_id, &mut builder.func);

                        // Get the address of the global data
                        let ptr_ty = global_types[&value.id];
                        let addr = builder.ins().symbol_value(ptr_ty, local_data);
                        self.value_map.insert(value.id, addr);
                    }
                }
            }

            // Map undef values to zero constants (for IDF-based SSA)
            // For aggregate types (structs, arrays), allocate stack space instead of using a zero constant
            for value in function.values.values() {
                if let HirValueKind::Undef = value.kind {
                    // Check if this is an aggregate type with pre-computed size
                    if let Some(&alloc_size) = undef_aggregate_sizes.get(&value.id) {
                        let slot = builder.create_sized_stack_slot(
                            cranelift_codegen::ir::StackSlotData::new(
                                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                alloc_size,
                                0,
                            ),
                        );
                        let ptr = builder.ins().stack_addr(pointer_type, slot, 0);
                        self.value_map.insert(value.id, ptr);
                        log::trace!(
                            "[CRANELIFT UNDEF] Allocated {} bytes stack for aggregate undef {:?}",
                            alloc_size,
                            value.id
                        );
                    } else {
                        // For scalar types, use zero constant
                        let ty = type_cache.get(&value.ty).copied().unwrap_or(types::I64);
                        let cranelift_val = if ty.is_vector() {
                            // A vector-typed undef materialises as an
                            // all-zero vector. `iconst` is scalar-only —
                            // giving it a vector type is not a legal
                            // constant — so build the lane zero and
                            // broadcast it.
                            let lane = ty.lane_type();
                            let zero = if lane == types::F32 {
                                builder.ins().f32const(0.0)
                            } else if lane == types::F64 {
                                builder.ins().f64const(0.0)
                            } else {
                                builder.ins().iconst(lane, 0)
                            };
                            builder.ins().splat(ty, zero)
                        } else if ty.is_int() {
                            builder.ins().iconst(ty, 0)
                        } else if ty.is_float() {
                            if ty == types::F32 {
                                builder.ins().f32const(0.0)
                            } else {
                                builder.ins().f64const(0.0)
                            }
                        } else {
                            // For other types, use null constant
                            builder.ins().iconst(ty, 0)
                        };
                        self.value_map.insert(value.id, cranelift_val);
                    }
                }
            }

            // Helper mode — close out the synthetic prologue with a jump to
            // the loop header carrying the phi-typed live-ins as block
            // params. The header's existing block params (set up in Phase 2
            // for its phis) consume these directly.
            if let Some(layout) = &osr_helper {
                let header_block = block_map[&layout.header];
                builder.ins().jump(header_block, &ba(&osr_phi_jump_args));
            }

            // Track which blocks can be sealed and which are already sealed
            let mut seal_tracker: HashMap<HirId, usize> = HashMap::new();
            let mut sealed_blocks: std::collections::HashSet<HirId> =
                std::collections::HashSet::new();
            for hir_block_id in &block_order {
                let pred_count = predecessor_map
                    .get(hir_block_id)
                    .map(|v| v.len())
                    .unwrap_or(0);
                seal_tracker.insert(*hir_block_id, pred_count);
            }

            // Phase 4: Process each block in order
            for hir_block_id in &block_order {
                log::debug!("[Cranelift] Processing block {:?}", hir_block_id);
                let cranelift_block = block_map[hir_block_id];
                let hir_block = match function.blocks.get(hir_block_id) {
                    Some(b) => b,
                    None => {
                        log::debug!("[Cranelift]   Block not found in function.blocks!");
                        continue;
                    }
                };

                // Switch to this block. In helper mode, entry_block is the
                // synthetic prologue (NOT in block_order), so every block
                // here needs an explicit switch. In main mode, the function
                // entry block is already active from Phase 3.
                if osr_helper.is_some() || *hir_block_id != function.entry_block {
                    builder.switch_to_block(cranelift_block);
                }

                // Seal blocks with no predecessors immediately (like entry block)
                if let Some(&pred_count) = seal_tracker.get(hir_block_id) {
                    if pred_count == 0 && !sealed_blocks.contains(hir_block_id) {
                        builder.seal_block(cranelift_block);
                        sealed_blocks.insert(*hir_block_id);
                    }
                }

                // Map phi node results to block parameters
                let block_params = builder.block_params(cranelift_block).to_vec();
                log::debug!(
                    "[Cranelift] Block {:?} has {} phis and {} block_params",
                    hir_block_id,
                    hir_block.phis.len(),
                    block_params.len()
                );
                for (i, phi) in hir_block.phis.iter().enumerate() {
                    log::debug!(
                        "[Cranelift]   phi[{}]: result={:?}, block_params[{}]={:?}",
                        i,
                        phi.result,
                        i,
                        block_params.get(i)
                    );
                    if let Some(&param_val) = block_params.get(i) {
                        self.value_map.insert(phi.result, param_val);
                    }
                }

                // Tier-0 OSR back-edge probe: at every loop header, emit a
                // sample-and-call sequence that asks the runtime whether
                // a tier-1 OSR helper is available. When a layout is
                // representable, also emit the dispatch path: marshal
                // phi results into i64 args, call_indirect the helper,
                // return its result.
                // A header with no representable layout can never have a
                // helper, so a probe there would load a slot that stays null
                // for the life of the program. Skip it rather than pay a
                // load and a branch per iteration for a transfer that cannot
                // happen.
                if osr_loop_headers.contains(hir_block_id) && osr_layouts.contains_key(hir_block_id)
                {
                    let block_index = osr_block_index.get(hir_block_id).copied().unwrap_or(0);

                    let empty_frame = crate::osr::OsrFrame::for_types(&[]);
                    let (site_key, live_in_clir, live_in_types, frame, return_clir) =
                        if let Some(layout) = osr_layouts.get(hir_block_id) {
                            // Collect the Cranelift value backing each live-in.
                            let mut clir_vals: Vec<cranelift_codegen::ir::Value> = Vec::new();
                            for hir_id in &layout.live_ins {
                                if let Some(&v) = self.value_map.get(hir_id) {
                                    clir_vals.push(v);
                                }
                            }
                            if clir_vals.len() == layout.live_ins.len() {
                                (
                                    layout.site_key(),
                                    clir_vals,
                                    layout.live_in_types.clone(),
                                    layout.frame.clone(),
                                    osr_return_clir,
                                )
                            } else {
                                (
                                    crate::osr::encode_osr_site(block_index, 0),
                                    Vec::new(),
                                    Vec::new(),
                                    empty_frame.clone(),
                                    None,
                                )
                            }
                        } else {
                            // Layout rejected → probe only, no dispatch.
                            (
                                crate::osr::encode_osr_site(block_index, 0),
                                Vec::new(),
                                Vec::new(),
                                empty_frame.clone(),
                                None,
                            )
                        };

                    // Tag the probe's first instruction with a srcloc that
                    // encodes which site it is. `get_srclocs_sorted` is the
                    // only post-codegen mapping from an instruction back to
                    // its code offset, and a patchable site needs that
                    // offset. Bit 31 marks the tag as ours so it can't be
                    // confused with a real source position.
                    let tag = crate::osr::probe_srcloc_tag(site_key);
                    builder.set_srcloc(cranelift_codegen::ir::SourceLoc::new(tag));
                    emit_osr_back_edge_probe(
                        &mut builder,
                        &mut self.module,
                        osr_bead_id,
                        site_key,
                        &frame,
                        &live_in_clir,
                        &live_in_types,
                        return_clir,
                    );
                    builder.set_srcloc(cranelift_codegen::ir::SourceLoc::default());
                    probe_site_tags.push((site_key, tag));
                }

                // Process all instructions in this block
                for inst in &hir_block.instructions {
                    // Inline instruction translation to avoid borrow checker issues
                    match inst {
                        HirInstruction::Binary {
                            op,
                            result,
                            ty,
                            left,
                            right,
                        } => {
                            let lhs = self.value_map.get(left).copied().unwrap_or_else(|| {
                                panic!("Binary op left operand {:?} not in value_map", left)
                            });
                            let rhs = self.value_map.get(right).copied().unwrap_or_else(|| {
                                panic!("Binary op right operand {:?} not in value_map", right)
                            });

                            let (lhs, rhs) = Self::reconcile_binary_operands(
                                &mut builder,
                                ty.is_float(),
                                lhs,
                                rhs,
                            );

                            let value = match op {
                                BinaryOp::Add => {
                                    if ty.is_float() {
                                        builder.ins().fadd(lhs, rhs)
                                    } else {
                                        builder.ins().iadd(lhs, rhs)
                                    }
                                }
                                BinaryOp::Sub => {
                                    if ty.is_float() {
                                        builder.ins().fsub(lhs, rhs)
                                    } else {
                                        builder.ins().isub(lhs, rhs)
                                    }
                                }
                                BinaryOp::Mul => {
                                    if ty.is_float() {
                                        builder.ins().fmul(lhs, rhs)
                                    } else {
                                        builder.ins().imul(lhs, rhs)
                                    }
                                }
                                BinaryOp::Div => {
                                    if ty.is_float() {
                                        builder.ins().fdiv(lhs, rhs)
                                    } else if ty.is_signed() {
                                        builder.ins().sdiv(lhs, rhs)
                                    } else {
                                        builder.ins().udiv(lhs, rhs)
                                    }
                                }
                                BinaryOp::Rem => {
                                    if ty.is_signed() {
                                        builder.ins().srem(lhs, rhs)
                                    } else {
                                        builder.ins().urem(lhs, rhs)
                                    }
                                }
                                BinaryOp::And => builder.ins().band(lhs, rhs),
                                BinaryOp::Or => builder.ins().bor(lhs, rhs),
                                BinaryOp::Xor => builder.ins().bxor(lhs, rhs),
                                BinaryOp::Shl => builder.ins().ishl(lhs, rhs),
                                BinaryOp::Shr => {
                                    if ty.is_signed() {
                                        builder.ins().sshr(lhs, rhs)
                                    } else {
                                        builder.ins().ushr(lhs, rhs)
                                    }
                                }
                                // Comparisons - use operand type (not result type) to determine signed/unsigned
                                // Comparisons always return bool (i8) - never uextend
                                BinaryOp::Eq
                                | BinaryOp::Ne
                                | BinaryOp::Lt
                                | BinaryOp::Le
                                | BinaryOp::Gt
                                | BinaryOp::Ge => {
                                    // Get operand type from left value, not result type
                                    let operand_ty =
                                        function.values.get(left).map(|v| &v.ty).unwrap_or(ty);

                                    // String equality / inequality routes
                                    // through the `zrtl_string_equals`
                                    // runtime helper — raw pointer
                                    // comparison would only succeed when
                                    // both operands point at the SAME
                                    // interned string allocation, not
                                    // whenever the byte contents match.
                                    // Strings lower to `Ptr(I8)`;
                                    // Lt/Le/Gt/Ge on strings would need
                                    // a separate strcmp-style helper and
                                    // isn't covered here yet.
                                    let is_string_op = matches!(op, BinaryOp::Eq | BinaryOp::Ne)
                                        && match operand_ty {
                                            HirType::Ptr(inner) => {
                                                matches!(**inner, HirType::I8)
                                            }
                                            _ => false,
                                        };
                                    if is_string_op {
                                        let mut sig = self.module.make_signature();
                                        let ptr_ty = self.module.target_config().pointer_type();
                                        sig.params.push(AbiParam::new(ptr_ty));
                                        sig.params.push(AbiParam::new(ptr_ty));
                                        sig.returns.push(AbiParam::new(types::I32));
                                        let func_id = self
                                            .module
                                            .declare_function(
                                                "zrtl_string_equals",
                                                Linkage::Import,
                                                &sig,
                                            )
                                            .map_err(|e| {
                                                CompilerError::Backend(format!(
                                                    "Failed to declare zrtl_string_equals: {}",
                                                    e
                                                ))
                                            })?;
                                        let func_ref =
                                            self.module.declare_func_in_func(func_id, builder.func);
                                        let call = builder.ins().call(func_ref, &[lhs, rhs]);
                                        let eq_i32 = builder.inst_results(call)[0];
                                        let eq_i8 = builder.ins().ireduce(types::I8, eq_i32);
                                        let result_val = if matches!(op, BinaryOp::Eq) {
                                            eq_i8
                                        } else {
                                            let one = builder.ins().iconst(types::I8, 1);
                                            builder.ins().bxor(eq_i8, one)
                                        };
                                        self.value_map.insert(*result, result_val);
                                        continue;
                                    }

                                    let cc = match op {
                                        BinaryOp::Eq => IntCC::Equal,
                                        BinaryOp::Ne => IntCC::NotEqual,
                                        BinaryOp::Lt => {
                                            if operand_ty.is_signed() {
                                                IntCC::SignedLessThan
                                            } else {
                                                IntCC::UnsignedLessThan
                                            }
                                        }
                                        BinaryOp::Le => {
                                            if operand_ty.is_signed() {
                                                IntCC::SignedLessThanOrEqual
                                            } else {
                                                IntCC::UnsignedLessThanOrEqual
                                            }
                                        }
                                        BinaryOp::Gt => {
                                            if operand_ty.is_signed() {
                                                IntCC::SignedGreaterThan
                                            } else {
                                                IntCC::UnsignedGreaterThan
                                            }
                                        }
                                        BinaryOp::Ge => {
                                            if operand_ty.is_signed() {
                                                IntCC::SignedGreaterThanOrEqual
                                            } else {
                                                IntCC::UnsignedGreaterThanOrEqual
                                            }
                                        }
                                        _ => unreachable!(),
                                    };
                                    // Widen operands to match types if needed (e.g., i32 vs i64).
                                    // Use zero-extension for unsigned operand types so an
                                    // `UnsignedLessThan` between U32 and U64 doesn't sign-
                                    // extend the U32 value into a negative i64.
                                    let lhs_ty = builder.func.dfg.value_type(lhs);
                                    let rhs_ty = builder.func.dfg.value_type(rhs);
                                    let (lhs, rhs) =
                                        if lhs_ty != rhs_ty && lhs_ty.is_int() && rhs_ty.is_int() {
                                            let extend = |b: &mut FunctionBuilder<'_>,
                                                          target: cranelift_codegen::ir::Type,
                                                          v: cranelift_codegen::ir::Value|
                                             -> cranelift_codegen::ir::Value {
                                                if operand_ty.is_signed() {
                                                    b.ins().sextend(target, v)
                                                } else {
                                                    b.ins().uextend(target, v)
                                                }
                                            };
                                            if lhs_ty.bits() < rhs_ty.bits() {
                                                (extend(&mut builder, rhs_ty, lhs), rhs)
                                            } else {
                                                (lhs, extend(&mut builder, lhs_ty, rhs))
                                            }
                                        } else {
                                            (lhs, rhs)
                                        };
                                    builder.ins().icmp(cc, lhs, rhs)
                                }
                                BinaryOp::FAdd => builder.ins().fadd(lhs, rhs),
                                BinaryOp::FSub => builder.ins().fsub(lhs, rhs),
                                BinaryOp::FMul => builder.ins().fmul(lhs, rhs),
                                BinaryOp::FDiv => builder.ins().fdiv(lhs, rhs),
                                BinaryOp::FRem => {
                                    Self::call_libm_fmod(&mut self.module, &mut builder, lhs, rhs)?
                                }
                                // Float comparisons also always return bool (i8)
                                BinaryOp::FEq
                                | BinaryOp::FNe
                                | BinaryOp::FLt
                                | BinaryOp::FLe
                                | BinaryOp::FGt
                                | BinaryOp::FGe => {
                                    let cc = match op {
                                        BinaryOp::FEq => FloatCC::Equal,
                                        BinaryOp::FNe => FloatCC::NotEqual,
                                        BinaryOp::FLt => FloatCC::LessThan,
                                        BinaryOp::FLe => FloatCC::LessThanOrEqual,
                                        BinaryOp::FGt => FloatCC::GreaterThan,
                                        BinaryOp::FGe => FloatCC::GreaterThanOrEqual,
                                        _ => unreachable!(),
                                    };
                                    // fcmp always returns i8 (bool) - no extension needed
                                    builder.ins().fcmp(cc, lhs, rhs)
                                }
                            };

                            // The op's width comes from its operands, so a
                            // float expression with one f64 operand produces
                            // an f64 even where the HIR result is f32. A
                            // Store takes its width from the stored value,
                            // so leaving it wide writes eight bytes into a
                            // four-byte element. Bring the result back to
                            // the type the HIR declared for it.
                            let value = match type_cache.get(ty).copied() {
                                Some(want) if want.is_float() && !want.is_vector() => {
                                    let got = builder.func.dfg.value_type(value);
                                    if got != want && got.is_float() && !got.is_vector() {
                                        Self::coerce_value(&mut builder, value, got, want)
                                    } else {
                                        value
                                    }
                                }
                                _ => value,
                            };

                            self.value_map.insert(*result, value);
                        }

                        HirInstruction::Unary {
                            op,
                            result,
                            ty,
                            operand,
                        } => {
                            let val = self.value_map[operand];

                            let value = match op {
                                UnaryOp::Neg => {
                                    if ty.is_float() {
                                        builder.ins().fneg(val)
                                    } else {
                                        builder.ins().ineg(val)
                                    }
                                }
                                UnaryOp::FNeg => builder.ins().fneg(val),
                                UnaryOp::Not => builder.ins().bnot(val),
                            };

                            self.value_map.insert(*result, value);
                        }

                        HirInstruction::Select {
                            result,
                            ty,
                            condition,
                            true_val,
                            false_val,
                        } => {
                            let cond = self.value_map[condition];
                            let true_v = self.value_map[true_val];
                            let false_v = self.value_map[false_val];

                            let value = builder.ins().select(cond, true_v, false_v);
                            self.value_map.insert(*result, value);
                        }

                        HirInstruction::Call {
                            result,
                            callee,
                            args,
                            is_tail: _,
                            ..
                        } => {
                            let arg_values: Vec<Value> = args.iter()
                                .map(|arg_id| {
                                    match self.value_map.get(arg_id) {
                                        Some(v) => *v,
                                        None => panic!("Call instruction: arg {:?} not in value_map. Available: {:?}",
                                            arg_id, self.value_map.keys().collect::<Vec<_>>()),
                                    }
                                })
                                .collect();

                            match callee {
                                HirCallable::Intrinsic(intrinsic) => {
                                    // Handle intrinsic calls
                                    let value = match intrinsic {
                                        Intrinsic::Sqrt => {
                                            if let Some(&arg) = arg_values.first() {
                                                builder.ins().sqrt(arg)
                                            } else {
                                                continue; // Skip if no argument
                                            }
                                        }
                                        Intrinsic::Rsqrt => {
                                            // Reciprocal square root, lowered as
                                            // `1.0 / sqrt(x)`. Cranelift has no
                                            // direct FRSQRTE op; this keeps the
                                            // math correct on all targets.
                                            // Hardware estimate + Newton
                                            // refinement is a future optimisation.
                                            if let Some(&arg) = arg_values.first() {
                                                let sq = builder.ins().sqrt(arg);
                                                let arg_ty = builder.func.dfg.value_type(arg);
                                                let one = if arg_ty == types::F32 {
                                                    builder.ins().f32const(1.0_f32)
                                                } else {
                                                    builder.ins().f64const(1.0_f64)
                                                };
                                                builder.ins().fdiv(one, sq)
                                            } else {
                                                continue; // Skip if no argument
                                            }
                                        }
                                        Intrinsic::Fabs => {
                                            if let Some(&arg) = arg_values.first() {
                                                builder.ins().fabs(arg)
                                            } else {
                                                continue; // Skip if no argument
                                            }
                                        }
                                        Intrinsic::Fma => {
                                            if arg_values.len() == 3 {
                                                builder.ins().fma(
                                                    arg_values[0],
                                                    arg_values[1],
                                                    arg_values[2],
                                                )
                                            } else {
                                                continue; // Skip if wrong arity
                                            }
                                        }
                                        Intrinsic::Ctpop => {
                                            if let Some(&arg) = arg_values.first() {
                                                builder.ins().popcnt(arg)
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Malloc => {
                                            // Call external malloc from libc
                                            if let Some(&size_arg) = arg_values.first() {
                                                // Get or declare malloc function
                                                let malloc_sig = {
                                                    let mut sig = self.module.make_signature();
                                                    sig.params.push(
                                                        cranelift_codegen::ir::AbiParam::new(
                                                            types::I64,
                                                        ),
                                                    );
                                                    sig.returns.push(
                                                        cranelift_codegen::ir::AbiParam::new(
                                                            types::I64,
                                                        ),
                                                    ); // pointer as i64
                                                    sig
                                                };

                                                let malloc_func = self
                                                    .module
                                                    .declare_function(
                                                        "malloc",
                                                        Linkage::Import,
                                                        &malloc_sig,
                                                    )
                                                    .expect("Failed to declare malloc");

                                                let local_malloc =
                                                    self.module.declare_func_in_func(
                                                        malloc_func,
                                                        builder.func,
                                                    );

                                                let call =
                                                    builder.ins().call(local_malloc, &[size_arg]);
                                                builder.inst_results(call)[0]
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Free => {
                                            // Call external free from libc
                                            if let Some(&ptr_arg) = arg_values.first() {
                                                let free_sig = {
                                                    let mut sig = self.module.make_signature();
                                                    sig.params.push(
                                                        cranelift_codegen::ir::AbiParam::new(
                                                            types::I64,
                                                        ),
                                                    );
                                                    // free returns void
                                                    sig
                                                };

                                                let free_func = self
                                                    .module
                                                    .declare_function(
                                                        "free",
                                                        Linkage::Import,
                                                        &free_sig,
                                                    )
                                                    .expect("Failed to declare free");

                                                let local_free = self
                                                    .module
                                                    .declare_func_in_func(free_func, builder.func);

                                                builder.ins().call(local_free, &[ptr_arg]);
                                                // Free returns void, so create a dummy value
                                                builder.ins().iconst(types::I64, 0)
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Sin => {
                                            if let Some(&arg) = arg_values.first() {
                                                // Inline libm sin call to avoid borrow issues
                                                let mut sig = self.module.make_signature();
                                                sig.params.push(AbiParam::new(types::F64));
                                                sig.returns.push(AbiParam::new(types::F64));
                                                let sin_id = self
                                                    .module
                                                    .declare_function("sin", Linkage::Import, &sig)
                                                    .map_err(|e| {
                                                        CompilerError::Backend(format!(
                                                            "Failed to declare sin: {}",
                                                            e
                                                        ))
                                                    })?;
                                                let sin_func = self
                                                    .module
                                                    .declare_func_in_func(sin_id, builder.func);
                                                let call = builder.ins().call(sin_func, &[arg]);
                                                builder.inst_results(call)[0]
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Cos => {
                                            if let Some(&arg) = arg_values.first() {
                                                let mut sig = self.module.make_signature();
                                                sig.params.push(AbiParam::new(types::F64));
                                                sig.returns.push(AbiParam::new(types::F64));
                                                let cos_id = self
                                                    .module
                                                    .declare_function("cos", Linkage::Import, &sig)
                                                    .map_err(|e| {
                                                        CompilerError::Backend(format!(
                                                            "Failed to declare cos: {}",
                                                            e
                                                        ))
                                                    })?;
                                                let cos_func = self
                                                    .module
                                                    .declare_func_in_func(cos_id, builder.func);
                                                let call = builder.ins().call(cos_func, &[arg]);
                                                builder.inst_results(call)[0]
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Log => {
                                            if let Some(&arg) = arg_values.first() {
                                                let mut sig = self.module.make_signature();
                                                sig.params.push(AbiParam::new(types::F64));
                                                sig.returns.push(AbiParam::new(types::F64));
                                                let log_id = self
                                                    .module
                                                    .declare_function("log", Linkage::Import, &sig)
                                                    .map_err(|e| {
                                                        CompilerError::Backend(format!(
                                                            "Failed to declare log: {}",
                                                            e
                                                        ))
                                                    })?;
                                                let log_func = self
                                                    .module
                                                    .declare_func_in_func(log_id, builder.func);
                                                let call = builder.ins().call(log_func, &[arg]);
                                                builder.inst_results(call)[0]
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Exp => {
                                            if let Some(&arg) = arg_values.first() {
                                                let mut sig = self.module.make_signature();
                                                sig.params.push(AbiParam::new(types::F64));
                                                sig.returns.push(AbiParam::new(types::F64));
                                                let exp_id = self
                                                    .module
                                                    .declare_function("exp", Linkage::Import, &sig)
                                                    .map_err(|e| {
                                                        CompilerError::Backend(format!(
                                                            "Failed to declare exp: {}",
                                                            e
                                                        ))
                                                    })?;
                                                let exp_func = self
                                                    .module
                                                    .declare_func_in_func(exp_id, builder.func);
                                                let call = builder.ins().call(exp_func, &[arg]);
                                                builder.inst_results(call)[0]
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Pow => {
                                            if arg_values.len() >= 2 {
                                                let base = arg_values[0];
                                                let exp = arg_values[1];
                                                let mut sig = self.module.make_signature();
                                                sig.params.push(AbiParam::new(types::F64));
                                                sig.params.push(AbiParam::new(types::F64));
                                                sig.returns.push(AbiParam::new(types::F64));
                                                let pow_id = self
                                                    .module
                                                    .declare_function("pow", Linkage::Import, &sig)
                                                    .map_err(|e| {
                                                        CompilerError::Backend(format!(
                                                            "Failed to declare pow: {}",
                                                            e
                                                        ))
                                                    })?;
                                                let pow_func = self
                                                    .module
                                                    .declare_func_in_func(pow_id, builder.func);
                                                let call =
                                                    builder.ins().call(pow_func, &[base, exp]);
                                                builder.inst_results(call)[0]
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Ctlz => {
                                            if let Some(&arg) = arg_values.first() {
                                                builder.ins().clz(arg)
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Cttz => {
                                            if let Some(&arg) = arg_values.first() {
                                                builder.ins().ctz(arg)
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Bswap => {
                                            if let Some(&arg) = arg_values.first() {
                                                builder.ins().bswap(arg)
                                            } else {
                                                continue;
                                            }
                                        }
                                        Intrinsic::Await => {
                                            // An `Intrinsic::Await` that survives to codegen
                                            // means a lowering invariant was violated: the
                                            // async state-machine transform should have
                                            // rewritten every await into save/suspend/resume
                                            // before this point. Silently skipping it (the old
                                            // catch-all behaviour) left the result HirId out of
                                            // `value_map`, so the NEXT instruction that read it
                                            // panicked with a cryptic "not in value_map". Fail
                                            // here with a descriptive error instead.
                                            return Err(crate::CompilerError::Backend(
                                                "unlowered `await` reached codegen — the async \
                                                 state-machine transform did not run for this \
                                                 function (await outside an `async def`?)"
                                                    .to_string(),
                                            ));
                                        }
                                        _ => {
                                            // Other intrinsics not implemented yet
                                            continue;
                                        }
                                    };

                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, value);
                                    }
                                }
                                HirCallable::Function(func_id) => {
                                    // Check if this is an external function with boxing requirements
                                    if let Some(link_name) =
                                        self.external_link_names.get(func_id).cloned()
                                    {
                                        // External function - check if boxing is needed
                                        let mut boxed_args = Vec::new();
                                        for (param_index, &arg_val) in arg_values.iter().enumerate()
                                        {
                                            let needs_boxing = symbol_boxing
                                                .get(&(link_name.clone(), param_index))
                                                .copied()
                                                .unwrap_or(false);

                                            if needs_boxing {
                                                // Apply DynamicBox wrapping - same logic as HirCallable::Symbol
                                                let arg_hir_id = args[param_index];
                                                let is_pointer_type = function
                                                    .values
                                                    .get(&arg_hir_id)
                                                    .map(|hir_value| {
                                                        dynamic_box_uses_direct_pointer(
                                                            &hir_value.ty,
                                                        )
                                                    })
                                                    .unwrap_or(false);

                                                let (tag_value, size_value) = function
                                                    .values
                                                    .get(&arg_hir_id)
                                                    .map(|hir_value| {
                                                        dynamic_box_tag_and_size_for_hir_type(
                                                            &hir_value.ty,
                                                        )
                                                    })
                                                    .unwrap_or_else(
                                                        default_dynamic_box_opaque_tag_and_size,
                                                    );

                                                let data_ptr_value = if is_pointer_type {
                                                    // Pointer types (opaque, string): value IS the pointer
                                                    match builder.func.dfg.value_type(arg_val) {
                                                        types::I64 => arg_val,
                                                        _ => builder
                                                            .ins()
                                                            .uextend(types::I64, arg_val),
                                                    }
                                                } else {
                                                    // Primitives: allocate stack slot and store pointer to it
                                                    let value_slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                                                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                                        8, 0,));
                                                    let value_addr = builder.ins().stack_addr(
                                                        types::I64,
                                                        value_slot,
                                                        0,
                                                    );
                                                    let data_value = match builder
                                                        .func
                                                        .dfg
                                                        .value_type(arg_val)
                                                    {
                                                        types::I8 | types::I16 | types::I32 => {
                                                            builder
                                                                .ins()
                                                                .sextend(types::I64, arg_val)
                                                        }
                                                        types::F32 => {
                                                            let as_i32 = builder.ins().bitcast(types::I32, cranelift_codegen::ir::MemFlagsData::new(), arg_val);
                                                            builder
                                                                .ins()
                                                                .uextend(types::I64, as_i32)
                                                        }
                                                        _ => arg_val,
                                                    };
                                                    builder.ins().store(
                                                        cranelift_codegen::ir::MemFlagsData::new(),
                                                        data_value,
                                                        value_addr,
                                                        0,
                                                    );
                                                    value_addr
                                                };

                                                // Create DynamicBox on stack
                                                let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                                                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                                    32, 0,));
                                                let box_addr =
                                                    builder.ins().stack_addr(types::I64, slot, 0);
                                                let tag_val = builder
                                                    .ins()
                                                    .iconst(types::I32, tag_value as i64);
                                                builder.ins().store(
                                                    cranelift_codegen::ir::MemFlagsData::new(),
                                                    tag_val,
                                                    box_addr,
                                                    0,
                                                );
                                                let size_val = builder
                                                    .ins()
                                                    .iconst(types::I32, size_value as i64);
                                                builder.ins().store(
                                                    cranelift_codegen::ir::MemFlagsData::new(),
                                                    size_val,
                                                    box_addr,
                                                    4,
                                                );
                                                builder.ins().store(
                                                    cranelift_codegen::ir::MemFlagsData::new(),
                                                    data_ptr_value,
                                                    box_addr,
                                                    8,
                                                );
                                                let null_dropper =
                                                    builder.ins().iconst(types::I64, 0);
                                                builder.ins().store(
                                                    cranelift_codegen::ir::MemFlagsData::new(),
                                                    null_dropper,
                                                    box_addr,
                                                    16,
                                                );

                                                // Check for Display trait impl
                                                let display_fn_value = if let Some(hir_value) =
                                                    function.values.get(&arg_hir_id)
                                                {
                                                    let opaque_name = match &hir_value.ty {
                                                        HirType::Opaque(type_name) => {
                                                            Some(type_name)
                                                        }
                                                        HirType::Ptr(inner) => {
                                                            if let HirType::Opaque(type_name) =
                                                                inner.as_ref()
                                                            {
                                                                Some(type_name)
                                                            } else {
                                                                None
                                                            }
                                                        }
                                                        _ => None,
                                                    };
                                                    if let Some(type_name) = opaque_name {
                                                        let type_name_str = type_name
                                                            .resolve_global()
                                                            .unwrap_or_default();
                                                        // Strip leading $ if present (opaque types may have $ prefix)
                                                        let clean_type_name =
                                                            if type_name_str.starts_with('$') {
                                                                &type_name_str[1..]
                                                            } else {
                                                                &type_name_str
                                                            };
                                                        if !clean_type_name.is_empty() {
                                                            let display_symbol = format!(
                                                                "${}$to_string",
                                                                clean_type_name
                                                            );
                                                            if let Some((_, func_ptr)) = self
                                                                .runtime_symbols
                                                                .iter()
                                                                .find(|(n, _)| n == &display_symbol)
                                                            {
                                                                builder.ins().iconst(
                                                                    types::I64,
                                                                    *func_ptr as i64,
                                                                )
                                                            } else {
                                                                builder.ins().iconst(types::I64, 0)
                                                            }
                                                        } else {
                                                            builder.ins().iconst(types::I64, 0)
                                                        }
                                                    } else {
                                                        builder.ins().iconst(types::I64, 0)
                                                    }
                                                } else {
                                                    builder.ins().iconst(types::I64, 0)
                                                };
                                                builder.ins().store(
                                                    cranelift_codegen::ir::MemFlagsData::new(),
                                                    display_fn_value,
                                                    box_addr,
                                                    24,
                                                );
                                                boxed_args.push(box_addr);
                                            } else {
                                                boxed_args.push(arg_val);
                                            }
                                        }

                                        // Call the external function with boxed args
                                        if let Some(&cranelift_func_id) =
                                            self.function_map.get(func_id)
                                        {
                                            let local_callee = self.module.declare_func_in_func(
                                                cranelift_func_id,
                                                builder.func,
                                            );

                                            // Coerce argument types to match declared signature
                                            let sig_ref =
                                                builder.func.dfg.ext_funcs[local_callee].signature;
                                            let expected_types: Vec<_> =
                                                builder.func.dfg.signatures[sig_ref]
                                                    .params
                                                    .iter()
                                                    .map(|p| p.value_type)
                                                    .collect();
                                            let mut coerced = boxed_args.clone();
                                            for (i, &expected) in expected_types.iter().enumerate()
                                            {
                                                if i < coerced.len() {
                                                    let actual =
                                                        builder.func.dfg.value_type(coerced[i]);
                                                    if actual != expected {
                                                        coerced[i] = Self::coerce_value(
                                                            &mut builder,
                                                            coerced[i],
                                                            actual,
                                                            expected,
                                                        );
                                                    }
                                                }
                                            }

                                            let call = builder.ins().call(local_callee, &coerced);
                                            if let Some(result_id) = result {
                                                if let Some(&ret_val) =
                                                    builder.inst_results(call).first()
                                                {
                                                    self.value_map.insert(*result_id, ret_val);
                                                }
                                            }
                                        }
                                    } else {
                                        // Regular (non-external) function call - no boxing needed
                                        if let Some(&cranelift_func_id) =
                                            self.function_map.get(func_id)
                                        {
                                            let declared_sig = self
                                                .module
                                                .declarations()
                                                .get_function_decl(cranelift_func_id)
                                                .signature
                                                .clone();

                                            // Coerce argument types to match declared signature
                                            let expected_types: Vec<_> = declared_sig
                                                .params
                                                .iter()
                                                .map(|p| p.value_type)
                                                .collect();
                                            let mut coerced = arg_values.clone();

                                            // A callee that returns an aggregate needs
                                            // somewhere to put it that outlives its own
                                            // frame, so the memory comes from here. One
                                            // slot per call site, which is what keeps two
                                            // calls from landing on top of each other.
                                            if let Some(&bytes) =
                                                self.destination_returns.get(func_id)
                                            {
                                                let slot = builder.create_sized_stack_slot(
                                                    cranelift_codegen::ir::StackSlotData::new(
                                                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                                        bytes,
                                                        0,
                                                    ),
                                                );
                                                let dest =
                                                    builder.ins().stack_addr(pointer_type, slot, 0);
                                                coerced.insert(0, dest);
                                            }
                                            for (i, &expected) in expected_types.iter().enumerate()
                                            {
                                                if i < coerced.len() {
                                                    let actual =
                                                        builder.func.dfg.value_type(coerced[i]);
                                                    if actual != expected {
                                                        coerced[i] = Self::coerce_value(
                                                            &mut builder,
                                                            coerced[i],
                                                            actual,
                                                            expected,
                                                        );
                                                    }
                                                }
                                            }

                                            let call = if self.reloadable_calls {
                                                let sig_ref =
                                                    builder.import_signature(declared_sig);
                                                let ptr_ty =
                                                    self.module.target_config().pointer_type();
                                                let cell = crate::reload::call_cell_addr(
                                                    self.reload_key,
                                                    *func_id,
                                                )
                                                    as i64;
                                                let cell_v = builder.ins().iconst(ptr_ty, cell);
                                                let entry = builder.ins().load(
                                                    ptr_ty,
                                                    MemFlags::trusted(),
                                                    cell_v,
                                                    0,
                                                );
                                                builder
                                                    .ins()
                                                    .call_indirect(sig_ref, entry, &coerced)
                                            } else {
                                                let local_callee =
                                                    self.module.declare_func_in_func(
                                                        cranelift_func_id,
                                                        builder.func,
                                                    );
                                                builder.ins().call(local_callee, &coerced)
                                            };

                                            if let Some(result_id) = result {
                                                if let Some(&ret_val) =
                                                    builder.inst_results(call).first()
                                                {
                                                    self.value_map.insert(*result_id, ret_val);
                                                }
                                            }
                                        } else {
                                            warn!(" Function {:?} not in function_map", func_id);
                                            if let Some(result_id) = result {
                                                self.value_map.insert(
                                                    *result_id,
                                                    builder.ins().iconst(types::I64, 0),
                                                );
                                            }
                                        }
                                    }
                                }
                                HirCallable::FuncRef(func_id) => {
                                    // Get function address as a pointer value
                                    if let Some(&cranelift_func_id) = self.function_map.get(func_id)
                                    {
                                        let ptr_ty = types::I64;
                                        let addr = if self.reloadable_calls
                                            && !self.external_link_names.contains_key(func_id)
                                            && self.current_compile_id != Some(*func_id)
                                        {
                                            let cell = crate::reload::call_cell_addr(
                                                self.reload_key,
                                                *func_id,
                                            )
                                                as i64;
                                            let cell_v = builder.ins().iconst(ptr_ty, cell);
                                            builder.ins().load(
                                                ptr_ty,
                                                MemFlags::trusted(),
                                                cell_v,
                                                0,
                                            )
                                        } else {
                                            let func_ref = self.module.declare_func_in_func(
                                                cranelift_func_id,
                                                builder.func,
                                            );
                                            builder.ins().func_addr(ptr_ty, func_ref)
                                        };
                                        if let Some(result_id) = result {
                                            self.value_map.insert(*result_id, addr);
                                        }
                                    } else {
                                        warn!(
                                            "FuncRef: function {:?} not in function_map",
                                            func_id
                                        );
                                        if let Some(result_id) = result {
                                            let zero = builder.ins().iconst(types::I64, 0);
                                            self.value_map.insert(*result_id, zero);
                                        }
                                    }
                                }
                                HirCallable::Indirect(func_ptr_id) => {
                                    // Indirect call through closure/function pointer.
                                    // The closure value IS the function pointer (for
                                    // non-capturing lambdas).
                                    //
                                    // If the func-ptr value isn't in `value_map`, the
                                    // SSA referenced a value that was never defined
                                    // (or never reached Cranelift via an emitted
                                    // instruction). Historically we warned + fell
                                    // back to `iconst(0)`, then `call_indirect`'d
                                    // into address 0 — silent SIGSEGV at runtime
                                    // with no path back to the actual cause. Same
                                    // anti-pattern as the `lower_function` silent
                                    // swallow (an embedder whose
                                    // — silent-drop mechanism identified").
                                    //
                                    // Return a proper compile-time error instead.
                                    // callers get a `CompilerError::Backend`
                                    // pointing at the offending HirId; we can then
                                    // chase the missing definition.
                                    let func_ptr_val =
                                        match self.value_map.get(func_ptr_id).copied() {
                                            Some(v) => v,
                                            None => {
                                                return Err(crate::CompilerError::Backend(
                                                    format!(
                                                "indirect call: function-pointer value {:?} \
                                                 is referenced but never defined in this \
                                                 function's value_map. SSA lowering produced \
                                                 a call to a value that wasn't emitted by an \
                                                 earlier instruction (CreateClosure / FuncRef \
                                                 / Load / Parameter).",
                                                func_ptr_id
                                            ),
                                                ));
                                            }
                                        };

                                    // Create signature for the indirect call
                                    let mut sig = self.module.make_signature();
                                    for arg_val in &arg_values {
                                        let arg_ty = builder.func.dfg.value_type(*arg_val);
                                        sig.params
                                            .push(cranelift_codegen::ir::AbiParam::new(arg_ty));
                                    }
                                    // Use i64 for return type (pointer-sized for opaque types)
                                    sig.returns
                                        .push(cranelift_codegen::ir::AbiParam::new(types::I64));

                                    let sig_ref = builder.import_signature(sig);
                                    let call = builder.ins().call_indirect(
                                        sig_ref,
                                        func_ptr_val,
                                        &arg_values,
                                    );

                                    if let Some(result_id) = result {
                                        if let Some(&ret_val) = builder.inst_results(call).first() {
                                            self.value_map.insert(*result_id, ret_val);
                                        }
                                    }
                                }
                                HirCallable::Symbol(symbol_name) => {
                                    // Call external runtime symbol by name (e.g., "$Image$load")
                                    // Check if any parameters need DynamicBox wrapping
                                    let mut boxed_args = Vec::new();
                                    for (param_index, &arg_val) in arg_values.iter().enumerate() {
                                        // Look up boxing requirement from pre-computed map
                                        let needs_boxing = symbol_boxing
                                            .get(&(symbol_name.clone(), param_index))
                                            .copied()
                                            .unwrap_or(false);
                                        log::debug!(
                                            "[DynamicBox] Symbol: {}, param {}: needs_boxing = {}",
                                            symbol_name,
                                            param_index,
                                            needs_boxing
                                        );
                                        if let Some(sig) = self.symbol_signatures.get(symbol_name) {
                                        } else {
                                        }
                                        if needs_boxing {
                                            // This parameter expects DynamicBox - wrap it
                                            // For opaque types (i64 pointer), we need to create a DynamicBox struct
                                            // DynamicBox layout: { tag: u32, size: u32, data: i64, dropper: i64, display_fn: i64 }

                                            // Allocate stack space for DynamicBox (32 bytes on 64-bit)
                                            // Determine TypeTag and size based on HIR type
                                            let arg_hir_id = args[param_index];
                                            let (tag_value, size_value) = function
                                                .values
                                                .get(&arg_hir_id)
                                                .map(|hir_value| {
                                                    dynamic_box_tag_and_size_for_hir_type(
                                                        &hir_value.ty,
                                                    )
                                                })
                                                .unwrap_or_else(|| {
                                                    log::warn!(
                                                        "[Boxing] No HirValue found for arg_hir_id {:?}",
                                                        arg_hir_id
                                                    );
                                                    default_dynamic_box_opaque_tag_and_size()
                                                });

                                            // Check if this is a pointer type (opaque types or strings that should be passed directly)
                                            let is_pointer_type = function
                                                .values
                                                .get(&args[param_index])
                                                .map(|hir_value| {
                                                    dynamic_box_uses_direct_pointer(&hir_value.ty)
                                                })
                                                .unwrap_or(false);

                                            // For pointer types, the value IS already a pointer - store directly
                                            // For primitives, allocate stack space and store a pointer to the value
                                            let data_ptr_value = if is_pointer_type {
                                                // Pointer types: the i64 value is the pointer itself
                                                // Cast to i64 if needed
                                                match builder.func.dfg.value_type(arg_val) {
                                                    types::I64 => arg_val,
                                                    _ => builder.ins().uextend(types::I64, arg_val),
                                                }
                                            } else {
                                                // Primitives: allocate stack space and store pointer to it
                                                let value_slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                                                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                                    8, 0,));
                                                let value_addr = builder.ins().stack_addr(
                                                    types::I64,
                                                    value_slot,
                                                    0,
                                                );

                                                // Store the actual value in the value slot
                                                // Extend smaller integers to i64 for storage
                                                let data_value = match builder
                                                    .func
                                                    .dfg
                                                    .value_type(arg_val)
                                                {
                                                    types::I8 | types::I16 | types::I32 => {
                                                        // Sign-extend to i64
                                                        builder.ins().sextend(types::I64, arg_val)
                                                    }
                                                    types::F32 => {
                                                        // Bitcast f32 to i32, then zero-extend to i64
                                                        let as_i32 = builder.ins().bitcast(
                                                            types::I32,
                                                            cranelift_codegen::ir::MemFlagsData::new(),
                                                            arg_val,
                                                        );
                                                        builder.ins().uextend(types::I64, as_i32)
                                                    }
                                                    _ => arg_val, // Already i64 or pointer
                                                };
                                                builder.ins().store(
                                                    cranelift_codegen::ir::MemFlagsData::new(),
                                                    data_value,
                                                    value_addr,
                                                    0,
                                                );
                                                value_addr
                                            };

                                            // Allocate stack space for DynamicBox (32 bytes on 64-bit)
                                            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                                                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                                32, 0,));
                                            let box_addr =
                                                builder.ins().stack_addr(types::I64, slot, 0);

                                            // Set tag
                                            let tag_val =
                                                builder.ins().iconst(types::I32, tag_value as i64);
                                            builder.ins().store(
                                                cranelift_codegen::ir::MemFlagsData::new(),
                                                tag_val,
                                                box_addr,
                                                0,
                                            );

                                            // Set size
                                            let size_val =
                                                builder.ins().iconst(types::I32, size_value as i64);
                                            builder.ins().store(
                                                cranelift_codegen::ir::MemFlagsData::new(),
                                                size_val,
                                                box_addr,
                                                4,
                                            );

                                            // Set data - for opaque types this IS the pointer, for primitives it's pointer to value
                                            builder.ins().store(
                                                cranelift_codegen::ir::MemFlagsData::new(),
                                                data_ptr_value,
                                                box_addr,
                                                8,
                                            );

                                            // Set dropper to null (0)
                                            let null_dropper = builder.ins().iconst(types::I64, 0);
                                            builder.ins().store(
                                                cranelift_codegen::ir::MemFlagsData::new(),
                                                null_dropper,
                                                box_addr,
                                                16,
                                            );

                                            // Set display_fn - check if this opaque type implements Display trait
                                            // Convention: Display::to_string is at symbol ${type_name}$to_string
                                            let display_fn_value = {
                                                // Get the HIR value to check if it's an opaque type
                                                let arg_hir_id = args[param_index];
                                                if let Some(hir_value) =
                                                    function.values.get(&arg_hir_id)
                                                {
                                                    // Extract opaque type name, handling both Opaque and Ptr(Opaque(...))
                                                    let opaque_name = match &hir_value.ty {
                                                        HirType::Opaque(type_name) => {
                                                            Some(type_name)
                                                        }
                                                        HirType::Ptr(inner) => {
                                                            if let HirType::Opaque(type_name) =
                                                                inner.as_ref()
                                                            {
                                                                Some(type_name)
                                                            } else {
                                                                None
                                                            }
                                                        }
                                                        _ => None,
                                                    };

                                                    if let Some(type_name) = opaque_name {
                                                        // Extract the type name string
                                                        let type_name_str = type_name
                                                            .resolve_global()
                                                            .unwrap_or_else(|| String::new());
                                                        // Strip leading $ if present (opaque types may have $ prefix)
                                                        let clean_type_name =
                                                            if type_name_str.starts_with('$') {
                                                                &type_name_str[1..]
                                                            } else {
                                                                type_name_str.as_str()
                                                            };

                                                        if !clean_type_name.is_empty() {
                                                            // Construct Display method symbol: ${type_name}$to_string
                                                            let display_symbol = format!(
                                                                "${}$to_string",
                                                                clean_type_name
                                                            );

                                                            // Look up in runtime symbols
                                                            if let Some((_, func_ptr)) = self
                                                                .runtime_symbols
                                                                .iter()
                                                                .find(|(name, _)| {
                                                                    name == &display_symbol
                                                                })
                                                            {
                                                                // Found Display implementation!
                                                                builder.ins().iconst(
                                                                    types::I64,
                                                                    *func_ptr as i64,
                                                                )
                                                            } else {
                                                                // No Display implementation
                                                                builder.ins().iconst(types::I64, 0)
                                                            }
                                                        } else {
                                                            // Empty type name
                                                            builder.ins().iconst(types::I64, 0)
                                                        }
                                                    } else {
                                                        // Not an opaque type
                                                        builder.ins().iconst(types::I64, 0)
                                                    }
                                                } else {
                                                    // HIR value not found
                                                    builder.ins().iconst(types::I64, 0)
                                                }
                                            };
                                            builder.ins().store(
                                                cranelift_codegen::ir::MemFlagsData::new(),
                                                display_fn_value,
                                                box_addr,
                                                24,
                                            );

                                            // Pass the box by value (load struct fields and pass as args)
                                            // Actually, DynamicBox is passed by value, so we need to load the struct
                                            // For now, pass the pointer to the struct (this might need ABI adjustment)
                                            boxed_args.push(box_addr);
                                        } else {
                                            // No boxing needed - use value as-is
                                            boxed_args.push(arg_val);
                                        }
                                    }

                                    // Create signature - prefer ZRTL signature if available
                                    let mut sig = self.module.make_signature();

                                    // Check if we have a ZRTL signature for this symbol
                                    let return_cranelift_ty = if let Some(sym_sig) =
                                        self.symbol_signatures.get(symbol_name)
                                    {
                                        // Use ZRTL signature for parameters AND return type
                                        for i in 0..sym_sig.param_count {
                                            // Convert ZRTL TypeTag to Cranelift type
                                            let type_tag = &sym_sig.params[i as usize];
                                            let cranelift_ty = type_tag_to_cranelift_type(type_tag);
                                            sig.params.push(cranelift_codegen::ir::AbiParam::new(
                                                cranelift_ty,
                                            ));
                                        }

                                        // Use signature return type
                                        if sym_sig.return_type == crate::zrtl::TypeTag::VOID {
                                            None
                                        } else {
                                            // Convert TypeTag to Cranelift type
                                            Some(type_tag_to_cranelift_type(&sym_sig.return_type))
                                        }
                                    } else {
                                        // No ZRTL signature - infer from boxed args
                                        for arg_val in &boxed_args {
                                            let arg_ty = builder.func.dfg.value_type(*arg_val);
                                            sig.params
                                                .push(cranelift_codegen::ir::AbiParam::new(arg_ty));
                                        }

                                        // Determine return type from HIR
                                        if let Some(result_id) = result {
                                            if let Some(value) = function.values.get(result_id) {
                                                // Map HIR type to Cranelift type inline
                                                match &value.ty {
                                                    HirType::I32 => Some(types::I32),
                                                    HirType::I64 => Some(types::I64),
                                                    HirType::F32 => Some(types::F32),
                                                    HirType::F64 => Some(types::F64),
                                                    HirType::Bool => Some(types::I8),
                                                    HirType::Ptr(_) => Some(types::I64),
                                                    HirType::Void => None,
                                                    _ => Some(types::I64), // Default to i64 for handles/complex types
                                                }
                                            } else {
                                                // Default to i64 for unknown external call results
                                                Some(types::I64)
                                            }
                                        } else {
                                            None
                                        }
                                    };

                                    if let Some(ret_ty) = return_cranelift_ty {
                                        sig.returns
                                            .push(cranelift_codegen::ir::AbiParam::new(ret_ty));
                                    }

                                    // Coerce argument types to match the declared signature
                                    // e.g., narrow i64 to i32 when ZRTL expects u32/i32
                                    let mut coerced_args = boxed_args.clone();
                                    for (i, param) in sig.params.iter().enumerate() {
                                        if i < coerced_args.len() {
                                            let actual_ty =
                                                builder.func.dfg.value_type(coerced_args[i]);
                                            let expected_ty = param.value_type;
                                            if actual_ty != expected_ty {
                                                coerced_args[i] = Self::coerce_value(
                                                    &mut builder,
                                                    coerced_args[i],
                                                    actual_ty,
                                                    expected_ty,
                                                );
                                            }
                                        }
                                    }

                                    let func = self
                                        .module
                                        .declare_function(symbol_name, Linkage::Import, &sig)
                                        .expect(&format!(
                                            "Failed to declare symbol {}",
                                            symbol_name
                                        ));

                                    let local_func =
                                        self.module.declare_func_in_func(func, builder.func);

                                    let call = builder.ins().call(local_func, &coerced_args);

                                    if let Some(result_id) = result {
                                        if let Some(&ret_val) = builder.inst_results(call).first() {
                                            self.value_map.insert(*result_id, ret_val);
                                        } else {
                                            // No return value - store dummy with correct type
                                            let dummy_ty =
                                                return_cranelift_ty.unwrap_or(types::I32);
                                            self.value_map.insert(
                                                *result_id,
                                                builder.ins().iconst(dummy_ty, 0),
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        HirInstruction::IndirectCall {
                            result,
                            func_ptr,
                            args,
                            return_ty,
                        } => {
                            // Indirect call through function pointer (for
                            // trait dispatch). Same silent-SIGSEGV anti-pattern
                            // as the `HirCallable::Indirect` arm above —
                            // hitting iconst(0) means we'd call_indirect into
                            // address 0 at runtime. Surface the missing-value
                            // as a real `CompilerError::Backend` so callers
                            // get a usable diagnostic
                            // instead of a runtime crash.
                            let func_ptr_val = match self.value_map.get(func_ptr).copied() {
                                Some(v) => v,
                                None => {
                                    return Err(crate::CompilerError::Backend(format!(
                                        "IndirectCall: function-pointer value {:?} \
                                         is referenced but never defined in this \
                                         function's value_map. Trait-dispatch / fn-ptr \
                                         lowering produced a call to a value that \
                                         wasn't emitted by an earlier instruction.",
                                        func_ptr
                                    )));
                                }
                            };

                            let arg_values: Vec<Value> = args
                                .iter()
                                .filter_map(|arg_id| self.value_map.get(arg_id).copied())
                                .collect();

                            // Create signature for the indirect call.
                            // Use the module's default calling convention —
                            // the platform ABI that maps from HIR's
                            // `CallingConvention::C` in `translate_signature`
                            // (SystemV on Unix x86_64, WindowsFastcall on
                            // x86_64-msvc, AppleAarch64 on ARM macOS, etc.).
                            //
                            // This previously hard-coded `CallConv::Fast` to
                            // match what `translate_signature` emitted when
                            // HIR `CallingConvention::Fast` was the lowered-
                            // function default. After commit 2581f86 made `C`
                            // (= platform default) the default for every
                            // user TypedFunction, that pinning became a
                            // mismatch: the callee was declared with
                            // `WindowsFastcall` on Windows but the indirect
                            // call site forced Fast, so args landed in
                            // SysV-flavoured registers (e.g. rdi) while the
                            // callee read from MS x64 registers (e.g. rcx).
                            //
                            // The mismatch manifested as `sync_entry`'s poll
                            // loop indirectly calling `poll_fn(sm_ptr)` and
                            // `poll_fn` reading garbage for its state-machine
                            // pointer — dispatcher loads garbage state, Switch
                            // falls through to `default = resume_entries[0]`
                            // (the perform site) over and over, the outer loop
                            // spins on rc=0 (Pending) because poll_fn never
                            // reaches a Ready return on the wrong sm pointer.
                            // The hang manifested on `phase_j1_multi_shot`,
                            // `phase_j2_abort`, `phase_j3_async`, and
                            // `tier3_abort_pattern` on Windows CI — none of
                            // which hit `__zyntax_effect_resume`'s re-entry
                            // guard because the runaway was outside it.
                            //
                            // i5 happened to pass on Windows because its single-
                            // shot resume path goes through __zyntax_effect_resume
                            // which directly invokes poll_fn via Rust's extern
                            // "C" transmute (= MS x64), bypassing this code path
                            // entirely on the inner re-entry.
                            let mut sig = self.module.make_signature();

                            // Add parameters
                            for _ in &arg_values {
                                sig.params
                                    .push(cranelift_codegen::ir::AbiParam::new(types::I64));
                            }

                            // Add return type if not void
                            let cranelift_return_ty = match return_ty {
                                HirType::Void => None,
                                HirType::I32 => Some(types::I32),
                                HirType::I64 => Some(types::I64),
                                HirType::F32 => Some(types::F32),
                                HirType::F64 => Some(types::F64),
                                HirType::Bool => Some(types::I8),
                                _ => Some(types::I64), // Default to i64 for complex types
                            };

                            if let Some(ret_ty) = cranelift_return_ty {
                                sig.returns
                                    .push(cranelift_codegen::ir::AbiParam::new(ret_ty));
                            }

                            let sig_ref = builder.import_signature(sig);
                            let call_inst =
                                builder
                                    .ins()
                                    .call_indirect(sig_ref, func_ptr_val, &arg_values);

                            // Map result if present
                            if let Some(result_id) = result {
                                let results = builder.inst_results(call_inst);
                                if !results.is_empty() {
                                    self.value_map.insert(*result_id, results[0]);
                                }
                            }
                        }

                        HirInstruction::Alloca {
                            result,
                            ty,
                            count,
                            align,
                        } => {
                            // Stack allocation
                            let cranelift_ty = type_cache.get(ty).copied().unwrap_or(types::I64);

                            let slot = if let Some(count_val_id) = count {
                                // Dynamic allocation (array)
                                let _count_val = self.value_map[count_val_id];
                                // Cranelift stack_store/load work with fixed slots
                                // For dynamic count, we need to calculate size and use stack_alloc
                                // For now, create a fixed-size slot and warn
                                warn!(" Dynamic alloca not fully supported, using fixed size");
                                builder.create_sized_stack_slot(
                                    cranelift_codegen::ir::StackSlotData::new(
                                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                        64, // Fixed size for now
                                        0,  // align_shift
                                    ),
                                )
                            } else {
                                // Static allocation (single value)
                                // IMPORTANT: Check size_cache FIRST because arrays/structs translate
                                // to I64 (pointer) in Cranelift but need actual allocated size
                                let size = if let Some(&cached_size) = size_cache.get(ty) {
                                    cached_size as u32
                                } else {
                                    // Fallback to Cranelift type size for primitives
                                    match cranelift_ty.bytes() {
                                        1 => 1,
                                        2 => 2,
                                        4 => 4,
                                        8 => 8,
                                        16 => 16,
                                        _ => 8, // Default fallback
                                    }
                                };

                                builder.create_sized_stack_slot(
                                    cranelift_codegen::ir::StackSlotData::new(
                                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                        size,
                                        0,
                                    ),
                                )
                            };

                            // Get pointer to stack slot
                            let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                            self.value_map.insert(*result, ptr);
                        }

                        HirInstruction::Load {
                            result,
                            ty,
                            ptr,
                            align,
                            volatile,
                        } => {
                            // Load value from memory
                            let ptr_val = match self.value_map.get(ptr) {
                                Some(v) => *v,
                                None => panic!(
                                    "Load instruction: ptr {:?} not in value_map. Available: {:?}",
                                    ptr,
                                    self.value_map.keys().collect::<Vec<_>>()
                                ),
                            };

                            // Aggregate-typed Load: the loaded "value"
                            // in Cranelift IR is a pointer to a fresh
                            // copy of the struct's bytes (matching the
                            // by-value SSA semantics where the load
                            // result is consumed by ExtractValue /
                            // InsertValue / Store as a struct, all of
                            // which treat the aggregate as a pointer).
                            // We allocate a stack slot, memcpy the
                            // struct's bytes into it, and route the
                            // result HirId at that slot's address.
                            // Otherwise the scalar Load below would
                            // only read the first 8 bytes — the bug
                            // that surfaces as nbody's bodies[i] reads
                            // returning a half-initialised Body.
                            let is_aggregate = matches!(
                                ty,
                                HirType::Struct(_) | HirType::Array(_, _) | HirType::Union(_)
                            );
                            if is_aggregate {
                                let size = size_cache.get(ty).copied().unwrap_or(0);
                                if size > 0 {
                                    let slot = builder.create_sized_stack_slot(
                                        cranelift_codegen::ir::StackSlotData::new(
                                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                            size as u32,
                                            0,
                                        ),
                                    );
                                    let dst = builder.ins().stack_addr(pointer_type, slot, 0);
                                    // Inline scalar copy for small aggregates
                                    // (≤ INLINE_COPY_MAX_BYTES) — avoids
                                    // libc memcpy thunk overhead on hot
                                    // struct-load paths (nbody's bodies[i]).
                                    if (size as u32) <= INLINE_COPY_MAX_BYTES {
                                        emit_inline_aggregate_copy(
                                            &mut builder,
                                            dst,
                                            ptr_val,
                                            size as u32,
                                        );
                                    } else {
                                        let size_val =
                                            builder.ins().iconst(types::I64, size as i64);
                                        builder.call_memcpy(
                                            self.module.target_config(),
                                            dst,
                                            ptr_val,
                                            size_val,
                                        );
                                    }
                                    self.value_map.insert(*result, dst);
                                } else {
                                    // Zero-sized aggregate: just route
                                    // the source pointer (nothing to copy).
                                    self.value_map.insert(*result, ptr_val);
                                }
                            } else {
                                let cranelift_ty =
                                    type_cache.get(ty).copied().unwrap_or(types::I64);

                                // TODO: Properly handle volatile flag
                                let flags = cranelift_codegen::ir::MemFlagsData::new();

                                let loaded = builder.ins().load(cranelift_ty, flags, ptr_val, 0);
                                self.value_map.insert(*result, loaded);
                            }
                        }

                        HirInstruction::Store {
                            value,
                            ptr,
                            align,
                            volatile,
                        } => {
                            // Store value to memory
                            let val = self.value_map[value];
                            let ptr_val = self.value_map[ptr];

                            // TODO: Properly handle volatile flag
                            let flags = cranelift_codegen::ir::MemFlagsData::new();

                            // Aggregate-typed Stores need a memcpy, not a
                            // scalar store. The InsertValue chain at line
                            // 3795 represents the aggregate's "value" as
                            // a pointer to a stack slot (`value_map[v]`
                            // = stack_addr), so `val` here is a pointer
                            // to source bytes. A plain `builder.store`
                            // would copy just the pointer (8 bytes)
                            // instead of the struct's contents — that's
                            // the bug that surfaces as nbody's
                            // `bodies[i]` reading null and segfaulting.
                            let value_ty = value_type_cache.get(value);
                            let is_aggregate = matches!(
                                value_ty,
                                Some(HirType::Struct(_))
                                    | Some(HirType::Array(_, _))
                                    | Some(HirType::Union(_))
                            );
                            if is_aggregate {
                                let size = size_cache.get(value_ty.unwrap()).copied().unwrap_or(0);
                                if size > 0 {
                                    // Inline scalar copy for small aggregates
                                    // (≤ INLINE_COPY_MAX_BYTES) — avoids
                                    // libc memcpy thunk overhead on hot
                                    // struct-store paths (nbody's bodies[i] =).
                                    if (size as u32) <= INLINE_COPY_MAX_BYTES {
                                        emit_inline_aggregate_copy(
                                            &mut builder,
                                            ptr_val,
                                            val,
                                            size as u32,
                                        );
                                    } else {
                                        let size_val =
                                            builder.ins().iconst(types::I64, size as i64);
                                        builder.call_memcpy(
                                            self.module.target_config(),
                                            ptr_val,
                                            val,
                                            size_val,
                                        );
                                    }
                                }
                            } else {
                                builder.ins().store(flags, val, ptr_val, 0);
                            }
                            // Store has no result value
                        }

                        HirInstruction::GetElementPtr {
                            result,
                            ty,
                            ptr,
                            indices,
                        } => {
                            // Calculate pointer offset for struct field or array element access
                            let mut current_ptr = self.value_map[ptr];
                            let mut current_type = ty.clone();

                            for index_id in indices {
                                let raw_index = self.value_map[index_id];
                                // GEP index narrower than I64 (typically
                                // I32 from integer literals or i32
                                // arithmetic) needs sign-extending so
                                // the subsequent `imul` against the
                                // I64-typed elem_size constant doesn't
                                // trip the Cranelift verifier with
                                // "arg N has type i32, expected i64".
                                let raw_ty = builder.func.dfg.value_type(raw_index);
                                let index = if raw_ty.is_int() && raw_ty.bits() < 64 {
                                    builder.ins().sextend(types::I64, raw_index)
                                } else {
                                    raw_index
                                };

                                match &current_type {
                                    HirType::Ptr(inner) => {
                                        // Pointer dereference - calculate offset
                                        let elem_size =
                                            size_cache.get(&**inner).copied().unwrap_or(1) as i64;

                                        // offset = index * elem_size
                                        let size_val = builder.ins().iconst(types::I64, elem_size);
                                        let offset = builder.ins().imul(index, size_val);
                                        current_ptr = builder.ins().iadd(current_ptr, offset);
                                        current_type = (**inner).clone();
                                    }
                                    HirType::Array(elem_ty, _) => {
                                        // Array indexing
                                        let elem_size =
                                            size_cache.get(&**elem_ty).copied().unwrap_or(1) as i64;

                                        let size_val = builder.ins().iconst(types::I64, elem_size);
                                        let offset = builder.ins().imul(index, size_val);
                                        current_ptr = builder.ins().iadd(current_ptr, offset);
                                        current_type = (**elem_ty).clone();
                                    }
                                    HirType::Struct(struct_ty) => {
                                        // Struct field access - index must be a constant
                                        // Extract the constant field index
                                        let field_index = if let Some(&const_val) =
                                            const_value_cache.get(index_id)
                                        {
                                            if const_val >= 0
                                                && (const_val as usize) < struct_ty.fields.len()
                                            {
                                                const_val as usize
                                            } else {
                                                warn!(" GEP struct field index {} out of bounds (struct has {} fields)",
                                                    const_val, struct_ty.fields.len());
                                                0 // Fallback to first field
                                            }
                                        } else {
                                            warn!(" GEP for struct requires constant index, got non-constant value");
                                            0 // Fallback to first field
                                        };

                                        if let Some(layout) = struct_layout_cache.get(struct_ty) {
                                            if let Some(&field_offset) =
                                                layout.field_offsets.get(field_index)
                                            {
                                                let offset_val = builder
                                                    .ins()
                                                    .iconst(types::I64, field_offset as i64);
                                                current_ptr =
                                                    builder.ins().iadd(current_ptr, offset_val);
                                                if let Some(field_ty) =
                                                    struct_ty.fields.get(field_index)
                                                {
                                                    current_type = field_ty.clone();
                                                }
                                            }
                                        } else {
                                            warn!(" No struct layout found for GEP");
                                        }
                                    }
                                    // For scalar types (U8, I8, etc.), treat as byte-level pointer arithmetic
                                    // This is used when the SSA layer wants to do direct byte offset calculation
                                    HirType::U8 | HirType::I8 => {
                                        // Index is already a byte offset, just add it
                                        current_ptr = builder.ins().iadd(current_ptr, index);
                                        // Type stays the same for subsequent indices
                                    }
                                    _ => {
                                        warn!(" GEP on unsupported type: {:?}", current_type);
                                        break;
                                    }
                                }
                            }

                            self.value_map.insert(*result, current_ptr);
                        }

                        HirInstruction::CreateUnion {
                            result,
                            union_ty,
                            variant_index,
                            value,
                        } => {
                            // Compute union layout without calling &self methods
                            // (avoids E0502 while builder holds &mut into self).
                            // Tagged layout: [discriminant:i32][pad][max_variant_data], 8-byte aligned.
                            let union_size: u32 = if let HirType::Union(u) = union_ty {
                                let discrim_sz: u32 = match u.discriminant_type.as_ref() {
                                    HirType::I64 | HirType::U64 | HirType::F64 => 8,
                                    HirType::I16 | HirType::U16 => 2,
                                    HirType::I8 | HirType::U8 | HirType::Bool => 1,
                                    _ => 4, // i32/u32/f32 and default
                                };
                                let max_var: u32 = u
                                    .variants
                                    .iter()
                                    .map(|v| match &v.ty {
                                        HirType::Void => 0,
                                        HirType::Bool | HirType::I8 | HirType::U8 => 1,
                                        HirType::I16 | HirType::U16 => 2,
                                        HirType::I32 | HirType::U32 | HirType::F32 => 4,
                                        HirType::I128 | HirType::U128 => 16,
                                        _ => 8, // i64/u64/f64/ptr/complex → 8 bytes
                                    })
                                    .max()
                                    .unwrap_or(0);
                                let data_offset = (discrim_sz + 7) & !7;
                                ((data_offset + max_var) + 7) & !7
                            } else {
                                16 // non-union fallback
                            };
                            let ptr_ty = self.module.target_config().pointer_type();

                            // Allocate space for the union on the stack
                            let union_slot = builder.create_sized_stack_slot(StackSlotData::new(
                                StackSlotKind::ExplicitSlot,
                                union_size,
                                0,
                            ));
                            let union_ptr = builder.ins().stack_addr(ptr_ty, union_slot, 0);

                            // Store the discriminant in the first field
                            let discriminant =
                                builder.ins().iconst(types::I32, *variant_index as i64);
                            builder
                                .ins()
                                .store(MemFlags::new(), discriminant, union_ptr, 0);

                            // Store the value in the data field (after discriminant)
                            let value_val = self.value_map[value];
                            let data_offset = 4; // Assuming 4-byte discriminant
                            builder
                                .ins()
                                .store(MemFlags::new(), value_val, union_ptr, data_offset);

                            self.value_map.insert(*result, union_ptr);
                        }

                        HirInstruction::GetUnionDiscriminant { result, union_val } => {
                            // Extract discriminant from union
                            let union_ptr = self.value_map[union_val];
                            let discriminant = builder.ins().load(
                                types::I32,
                                MemFlags::new(),
                                union_ptr,
                                0, // Discriminant is at offset 0
                            );
                            self.value_map.insert(*result, discriminant);
                        }

                        HirInstruction::ExtractUnionValue {
                            result,
                            ty,
                            union_val,
                            variant_index: _,
                        } => {
                            // Extract value from union variant (unsafe - assumes correct variant)
                            let union_ptr = self.value_map[union_val];
                            // Use type_cache if available, otherwise default to i64
                            let cranelift_ty = type_cache.get(ty).copied().unwrap_or(types::I64);
                            let data_offset = 4; // Skip discriminant

                            let value = builder.ins().load(
                                cranelift_ty,
                                MemFlags::new(),
                                union_ptr,
                                data_offset,
                            );
                            self.value_map.insert(*result, value);
                        }

                        HirInstruction::CreateTraitObject {
                            result,
                            trait_id,
                            data_ptr,
                            vtable_id,
                        } => {
                            // Create trait object as fat pointer: { *data, *vtable }
                            // A trait object is represented as a struct with two pointer fields

                            // Get the data pointer value
                            let data_ptr_val = self
                                .value_map
                                .get(data_ptr)
                                .copied()
                                .unwrap_or_else(|| builder.ins().iconst(types::I64, 0));

                            // Get the vtable pointer
                            // Note: Vtable globals may not be in value_map yet during compilation
                            // They will be resolved during linking/finalization
                            let vtable_ptr_val =
                                self.value_map.get(vtable_id).copied().unwrap_or_else(|| {
                                    // Use placeholder - actual vtable address resolved at link time
                                    builder.ins().iconst(types::I64, 0)
                                });

                            // Allocate space for fat pointer on stack (2 pointers = 16 bytes on 64-bit)
                            let ptr_type = self.module.target_config().pointer_type();
                            let fat_ptr_size = ptr_type.bytes() * 2; // 2 pointers
                            let stack_slot = builder.create_sized_stack_slot(StackSlotData::new(
                                StackSlotKind::ExplicitSlot,
                                fat_ptr_size,
                                0,
                            ));

                            // Get pointer to stack slot
                            let fat_ptr = builder.ins().stack_addr(ptr_type, stack_slot, 0);

                            // Store data pointer at offset 0
                            builder
                                .ins()
                                .store(MemFlags::new(), data_ptr_val, fat_ptr, 0);

                            // Store vtable pointer at offset pointer_size
                            let vtable_offset = ptr_type.bytes() as i32;
                            builder.ins().store(
                                MemFlags::new(),
                                vtable_ptr_val,
                                fat_ptr,
                                vtable_offset,
                            );

                            // Return the fat pointer address
                            self.value_map.insert(*result, fat_ptr);
                        }

                        HirInstruction::UpcastTraitObject {
                            result,
                            sub_trait_object,
                            sub_trait_id,
                            super_trait_id,
                            super_vtable_id,
                        } => {
                            // Upcast trait object: extract data pointer from sub-trait, combine with super-trait vtable
                            // Fat pointer layout: { *data (offset 0), *vtable (offset ptr_size) }

                            let ptr_type = self.module.target_config().pointer_type();
                            let ptr_size = ptr_type.bytes() as i32;

                            // Step 1: Get sub-trait fat pointer
                            let sub_trait_fat_ptr = self
                                .value_map
                                .get(sub_trait_object)
                                .copied()
                                .ok_or_else(|| {
                                CompilerError::CodeGen(format!(
                                    "Sub-trait object {:?} not found",
                                    sub_trait_object
                                ))
                            })?;

                            // Step 2: Extract data pointer from sub-trait object (offset 0)
                            let data_ptr =
                                builder
                                    .ins()
                                    .load(ptr_type, MemFlags::new(), sub_trait_fat_ptr, 0);

                            // Step 3: Get super-trait vtable pointer
                            // Note: Vtable globals may not be in value_map yet during compilation
                            // They will be resolved during linking/finalization
                            let super_vtable_ptr = self
                                .value_map
                                .get(super_vtable_id)
                                .copied()
                                .unwrap_or_else(|| {
                                    // Use placeholder - actual vtable address resolved at link time
                                    builder.ins().iconst(ptr_type, 0)
                                });

                            // Step 4: Allocate space for new fat pointer on stack
                            let fat_ptr_size = ptr_type.bytes() * 2;
                            let stack_slot = builder.create_sized_stack_slot(StackSlotData::new(
                                StackSlotKind::ExplicitSlot,
                                fat_ptr_size,
                                0,
                            ));

                            // Get pointer to stack slot
                            let super_trait_fat_ptr =
                                builder.ins().stack_addr(ptr_type, stack_slot, 0);

                            // Step 5: Store data pointer at offset 0 (same as sub-trait)
                            builder
                                .ins()
                                .store(MemFlags::new(), data_ptr, super_trait_fat_ptr, 0);

                            // Step 6: Store super-trait vtable pointer at offset ptr_size
                            builder.ins().store(
                                MemFlags::new(),
                                super_vtable_ptr,
                                super_trait_fat_ptr,
                                ptr_size,
                            );

                            // Return the new fat pointer
                            self.value_map.insert(*result, super_trait_fat_ptr);
                        }

                        HirInstruction::TraitMethodCall {
                            result,
                            trait_object,
                            method_index,
                            method_sig,
                            args,
                            return_ty,
                        } => {
                            // Dynamic dispatch: call method on trait object
                            // Fat pointer layout: { *data (offset 0), *vtable (offset ptr_size) }

                            let ptr_type = self.module.target_config().pointer_type();
                            let ptr_size = ptr_type.bytes() as i32;

                            // Get fat pointer (trait object)
                            let fat_ptr =
                                self.value_map.get(trait_object).copied().ok_or_else(|| {
                                    CompilerError::CodeGen(format!(
                                        "Trait object {:?} not found",
                                        trait_object
                                    ))
                                })?;

                            // TODO: Type-safe signatures
                            // Current limitation: Using ptr_type for all parameters
                            // Proper implementation requires method_sig.params translation
                            // Issue: Builder borrows prevent calling self.translate_type() here
                            // Will be fixed in next refactoring pass
                            let _ = method_sig; // Silence unused warning

                            // Step 1: Load data pointer from fat_ptr[0]
                            let data_ptr =
                                builder.ins().load(ptr_type, MemFlags::new(), fat_ptr, 0);

                            // Step 2: Load vtable pointer from fat_ptr[ptr_size]
                            let vtable_ptr =
                                builder
                                    .ins()
                                    .load(ptr_type, MemFlags::new(), fat_ptr, ptr_size);

                            // Step 3: Load function pointer from vtable[method_index]
                            let method_offset = (*method_index as i32) * ptr_size;
                            let func_ptr = builder.ins().load(
                                ptr_type,
                                MemFlags::new(),
                                vtable_ptr,
                                method_offset,
                            );

                            // Step 4: Build arguments: prepend self (data_ptr) to args
                            let mut call_args = vec![data_ptr];
                            for arg_id in args {
                                let arg_val =
                                    self.value_map.get(arg_id).copied().ok_or_else(|| {
                                        CompilerError::CodeGen(format!(
                                            "Argument {:?} not found",
                                            arg_id
                                        ))
                                    })?;
                                call_args.push(arg_val);
                            }

                            // Step 5: Create function signature for indirect call.
                            // Start from the module's default calling convention
                            // (the platform ABI: SystemV on Unix x86_64,
                            // WindowsFastcall on x86_64-msvc, AppleAarch64 on ARM
                            // macOS) so the indirect call matches how the callee
                            // was compiled. Hard-coding `SystemV` here made the
                            // async poll fn — reached via `CreateClosure` — read a
                            // garbage state-machine pointer on Windows (args landed
                            // in SysV registers like rdi while the WindowsFastcall
                            // callee read rcx), returning garbage for every async
                            // execution. Same root cause and fix as the indirect
                            // call site above.
                            let mut sig = self.module.make_signature();
                            sig.params.push(AbiParam::new(ptr_type)); // self (data ptr)
                            for _ in args {
                                // TODO: Use actual types from method_sig.params
                                sig.params.push(AbiParam::new(ptr_type));
                            }
                            if !matches!(return_ty, HirType::Void) {
                                // TODO: Translate return_ty from method_sig
                                sig.returns.push(AbiParam::new(ptr_type));
                            }
                            let sig = builder.func.import_signature(sig);

                            // Step 6: Perform indirect call
                            let call_inst = builder.ins().call_indirect(sig, func_ptr, &call_args);

                            // Step 7: Get return value if non-void
                            if let Some(result_id) = result {
                                let return_vals = builder.inst_results(call_inst);
                                if !return_vals.is_empty() {
                                    self.value_map.insert(*result_id, return_vals[0]);
                                }
                            }
                        }

                        HirInstruction::CreateClosure {
                            result,
                            closure_ty,
                            function,
                            captures,
                        } => {
                            // Create a closure with function pointer
                            let ptr_ty = self.module.target_config().pointer_type();

                            // For simple single-block functions, just return the function pointer
                            if let Some(&cranelift_func_id) = self.function_map.get(function) {
                                // The address taken is what a fiber or callback
                                // enters through later — under reload it has to
                                // be the function's *current* entry, read at
                                // creation time, not the one this caller was
                                // compiled against. Except when a function hands
                                // out its own address: that is a frame's
                                // continuation, and it must stay in this
                                // generation.
                                let func_ptr = if self.reloadable_calls
                                    && !self.external_link_names.contains_key(function)
                                    && self.current_compile_id != Some(*function)
                                {
                                    let cell =
                                        crate::reload::call_cell_addr(self.reload_key, *function)
                                            as i64;
                                    let cell_v = builder.ins().iconst(ptr_ty, cell);
                                    builder.ins().load(ptr_ty, MemFlags::trusted(), cell_v, 0)
                                } else {
                                    let local_func_ref = self
                                        .module
                                        .declare_func_in_func(cranelift_func_id, builder.func);
                                    builder.ins().func_addr(ptr_ty, local_func_ref)
                                };
                                self.value_map.insert(*result, func_ptr);
                            } else {
                                warn!(" CreateClosure: Lambda function {:?} not found", function);
                                let null_ptr = builder.ins().iconst(ptr_ty, 0);
                                self.value_map.insert(*result, null_ptr);
                            }
                        }

                        HirInstruction::ExtractValue {
                            result,
                            ty,
                            aggregate,
                            indices,
                        } => {
                            // Extract value from aggregate (struct/array)
                            // Strategy: Use GEP-like logic to calculate pointer, then Load the value

                            // A struct carried as its single field is that
                            // field: there is no memory to address and the
                            // value is already in hand. Reading it as a
                            // pointer would dereference whatever the scalar
                            // happens to be.
                            let flattened = value_type_cache
                                .get(aggregate)
                                .and_then(|t| match t {
                                    HirType::Struct(st) => struct_carried_as_its_field(st),
                                    _ => None,
                                })
                                .is_some();
                            if flattened {
                                if let Some(&agg) = self.value_map.get(aggregate) {
                                    self.value_map.insert(*result, agg);
                                }
                                continue;
                            }

                            // aggregate is a POINTER to the struct/array
                            let mut current_ptr = self.value_map[aggregate];

                            // Get the type of the aggregate from our cache
                            let mut current_type = value_type_cache
                                .get(aggregate)
                                .cloned()
                                .unwrap_or(HirType::Void);

                            // If the aggregate type is a pointer, dereference it first
                            if let HirType::Ptr(inner) = &current_type {
                                current_type = (**inner).clone();
                            }

                            if indices.is_empty() {
                                // No indices - this shouldn't happen, but handle gracefully
                                // Just load the value at the pointer
                                let cranelift_ty =
                                    type_cache.get(ty).copied().unwrap_or(types::I64);
                                let flags = cranelift_codegen::ir::MemFlagsData::new();
                                let loaded =
                                    builder.ins().load(cranelift_ty, flags, current_ptr, 0);
                                self.value_map.insert(*result, loaded);
                            } else {
                                // Navigate through indices using GEP-like logic
                                for (i, &index_u32) in indices.iter().enumerate() {
                                    let is_last = i == indices.len() - 1;

                                    match &current_type {
                                        HirType::Array(elem_ty, _) => {
                                            // Array indexing - index can be constant or we treat it as constant
                                            let elem_size =
                                                size_cache.get(&**elem_ty).copied().unwrap_or(1)
                                                    as i64;
                                            let offset = (index_u32 as i64) * elem_size;

                                            if !is_last {
                                                // Intermediate: just calculate new pointer
                                                let offset_val =
                                                    builder.ins().iconst(types::I64, offset);
                                                current_ptr =
                                                    builder.ins().iadd(current_ptr, offset_val);
                                                current_type = (**elem_ty).clone();
                                            } else {
                                                // Last index: calculate pointer and load
                                                let offset_val =
                                                    builder.ins().iconst(types::I64, offset);
                                                let elem_ptr =
                                                    builder.ins().iadd(current_ptr, offset_val);

                                                let cranelift_ty = type_cache
                                                    .get(ty)
                                                    .copied()
                                                    .unwrap_or(types::I64);
                                                let flags =
                                                    cranelift_codegen::ir::MemFlagsData::new();
                                                let loaded = builder.ins().load(
                                                    cranelift_ty,
                                                    flags,
                                                    elem_ptr,
                                                    0,
                                                );
                                                self.value_map.insert(*result, loaded);
                                            }
                                        }
                                        HirType::Struct(struct_ty) => {
                                            // Struct field access - index is always constant
                                            let field_index = index_u32 as usize;

                                            if let Some(layout) = struct_layout_cache.get(struct_ty)
                                            {
                                                if let Some(&field_offset) =
                                                    layout.field_offsets.get(field_index)
                                                {
                                                    if !is_last {
                                                        // Intermediate: calculate new pointer
                                                        let offset_val = builder.ins().iconst(
                                                            types::I64,
                                                            field_offset as i64,
                                                        );
                                                        current_ptr = builder
                                                            .ins()
                                                            .iadd(current_ptr, offset_val);
                                                        if let Some(field_ty) =
                                                            struct_ty.fields.get(field_index)
                                                        {
                                                            current_type = field_ty.clone();
                                                        }
                                                    } else {
                                                        // Last index: calculate pointer and load
                                                        let offset_val = builder.ins().iconst(
                                                            types::I64,
                                                            field_offset as i64,
                                                        );
                                                        let field_ptr = builder
                                                            .ins()
                                                            .iadd(current_ptr, offset_val);

                                                        let cranelift_ty = type_cache
                                                            .get(ty)
                                                            .copied()
                                                            .unwrap_or(types::I64);
                                                        let flags =
                                                            cranelift_codegen::ir::MemFlagsData::new();
                                                        let loaded = builder.ins().load(
                                                            cranelift_ty,
                                                            flags,
                                                            field_ptr,
                                                            0,
                                                        );
                                                        self.value_map.insert(*result, loaded);
                                                    }
                                                }
                                            }
                                        }
                                        HirType::Union(union_def) => {
                                            // Union field access: index 0 = discriminant, index 1 = data
                                            let field_index = index_u32 as usize;

                                            // Calculate union layout inline
                                            // Tagged union layout: [discriminant][padding][data]
                                            // Discriminant is always 4 bytes (u32), data starts at offset 4
                                            let discriminant_offset = 0u32;
                                            let data_offset = 4u32; // u32 discriminant size

                                            let field_offset = if field_index == 0 {
                                                discriminant_offset
                                            } else {
                                                data_offset
                                            };

                                            if !is_last {
                                                // Intermediate: calculate new pointer
                                                let offset_val = builder
                                                    .ins()
                                                    .iconst(types::I64, field_offset as i64);
                                                current_ptr =
                                                    builder.ins().iadd(current_ptr, offset_val);
                                                // Update current_type based on what we're extracting
                                                if field_index == 0 {
                                                    // Discriminant field
                                                    current_type = union_def
                                                        .discriminant_type
                                                        .as_ref()
                                                        .clone();
                                                } else {
                                                    // Data field - use the first variant's type as representative
                                                    // (all variants share the same memory location)
                                                    if let Some(variant) =
                                                        union_def.variants.first()
                                                    {
                                                        current_type = variant.ty.clone();
                                                    }
                                                }
                                            } else {
                                                // Last index: calculate pointer and load
                                                let offset_val = builder
                                                    .ins()
                                                    .iconst(types::I64, field_offset as i64);
                                                let field_ptr =
                                                    builder.ins().iadd(current_ptr, offset_val);

                                                let cranelift_ty = type_cache
                                                    .get(ty)
                                                    .copied()
                                                    .unwrap_or(types::I64);
                                                let flags =
                                                    cranelift_codegen::ir::MemFlagsData::new();
                                                let loaded = builder.ins().load(
                                                    cranelift_ty,
                                                    flags,
                                                    field_ptr,
                                                    0,
                                                );
                                                self.value_map.insert(*result, loaded);
                                            }
                                        }
                                        _ => {
                                            // Unsupported type - just return a dummy value
                                            warn!(
                                                " ExtractValue on unsupported type: {:?}",
                                                current_type
                                            );
                                            let cranelift_ty =
                                                type_cache.get(ty).copied().unwrap_or(types::I64);
                                            let dummy = builder.ins().iconst(cranelift_ty, 0);
                                            self.value_map.insert(*result, dummy);
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        HirInstruction::InsertValue {
                            result,
                            ty: _,
                            aggregate,
                            value,
                            indices,
                        } => {
                            // Insert value into aggregate (struct/array)

                            // The same struct as above: setting its only
                            // field produces the field, with nothing to
                            // store into.
                            let flattened = value_type_cache
                                .get(aggregate)
                                .and_then(|t| match t {
                                    HirType::Struct(st) => struct_carried_as_its_field(st),
                                    _ => None,
                                })
                                .is_some();
                            if flattened {
                                if let Some(&val) = self.value_map.get(value) {
                                    self.value_map.insert(*result, val);
                                }
                                continue;
                            }

                            // Get the type of the aggregate from our cache
                            let mut current_type = value_type_cache
                                .get(aggregate)
                                .cloned()
                                .unwrap_or(HirType::Void);

                            // If the aggregate type is a pointer, dereference it first
                            if let HirType::Ptr(inner) = &current_type {
                                current_type = (**inner).clone();
                            }

                            // Special case: single-field struct being returned by value
                            // In this case, the struct is flattened and we just return the value
                            if let HirType::Struct(struct_ty) = &current_type {
                                if struct_ty.fields.len() == 1
                                    && indices.len() == 1
                                    && indices[0] == 0
                                {
                                    // Check if the aggregate is Undef (building a new struct for return)
                                    if let Some(agg_value) = function.values.get(aggregate) {
                                        if matches!(agg_value.kind, crate::hir::HirValueKind::Undef)
                                        {
                                            // Flattened single-field struct: just return the value directly
                                            let val = self.value_map[value];
                                            self.value_map.insert(*result, val);
                                            continue;
                                        }
                                    }
                                }
                            }

                            // Standard case: aggregate is a POINTER to the struct/array
                            let base_ptr = self.value_map[aggregate];
                            let val = self.value_map[value];
                            let mut current_ptr = base_ptr;

                            if indices.is_empty() {
                                // No indices - just store at the pointer
                                let flags = cranelift_codegen::ir::MemFlagsData::new();
                                builder.ins().store(flags, val, current_ptr, 0);
                                self.value_map.insert(*result, base_ptr);
                            } else {
                                // Navigate through indices using GEP-like logic
                                for (i, &index_u32) in indices.iter().enumerate() {
                                    let is_last = i == indices.len() - 1;

                                    match &current_type {
                                        HirType::Array(elem_ty, _) => {
                                            // Array indexing
                                            let elem_size =
                                                size_cache.get(&**elem_ty).copied().unwrap_or(1)
                                                    as i64;
                                            let offset = (index_u32 as i64) * elem_size;

                                            if !is_last {
                                                // Intermediate: calculate new pointer
                                                let offset_val =
                                                    builder.ins().iconst(types::I64, offset);
                                                current_ptr =
                                                    builder.ins().iadd(current_ptr, offset_val);
                                                current_type = (**elem_ty).clone();
                                            } else {
                                                // Last index: calculate pointer and store
                                                let offset_val =
                                                    builder.ins().iconst(types::I64, offset);
                                                let elem_ptr =
                                                    builder.ins().iadd(current_ptr, offset_val);

                                                let flags =
                                                    cranelift_codegen::ir::MemFlagsData::new();
                                                builder.ins().store(flags, val, elem_ptr, 0);
                                                self.value_map.insert(*result, base_ptr);
                                            }
                                        }
                                        HirType::Struct(struct_ty) => {
                                            // Struct field access
                                            let field_index = index_u32 as usize;

                                            if let Some(layout) = struct_layout_cache.get(struct_ty)
                                            {
                                                if let Some(&field_offset) =
                                                    layout.field_offsets.get(field_index)
                                                {
                                                    if !is_last {
                                                        // Intermediate: calculate new pointer
                                                        let offset_val = builder.ins().iconst(
                                                            types::I64,
                                                            field_offset as i64,
                                                        );
                                                        current_ptr = builder
                                                            .ins()
                                                            .iadd(current_ptr, offset_val);
                                                        if let Some(field_ty) =
                                                            struct_ty.fields.get(field_index)
                                                        {
                                                            current_type = field_ty.clone();
                                                        }
                                                    } else {
                                                        // Last index: calculate pointer and store
                                                        let offset_val = builder.ins().iconst(
                                                            types::I64,
                                                            field_offset as i64,
                                                        );
                                                        let field_ptr = builder
                                                            .ins()
                                                            .iadd(current_ptr, offset_val);

                                                        let flags =
                                                            cranelift_codegen::ir::MemFlagsData::new();
                                                        builder
                                                            .ins()
                                                            .store(flags, val, field_ptr, 0);
                                                        self.value_map.insert(*result, base_ptr);
                                                    }
                                                }
                                            }
                                        }
                                        _ => {
                                            // Unsupported type
                                            warn!(" InsertValue on unsupported type");
                                            self.value_map.insert(*result, base_ptr);
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        HirInstruction::Cast {
                            result,
                            ty,
                            op,
                            operand,
                        } => {
                            // Cast instruction: convert between types
                            let val = self.value_map.get(operand).copied().unwrap_or_else(|| {
                                panic!("Cast operand {:?} not in value_map", operand)
                            });
                            // The cache holds every type this
                            // backend could translate, so a miss means
                            // the type has no Cranelift encoding.
                            // Substituting I64 builds the cast at the
                            // wrong width and everything downstream
                            // reads garbage; decline instead and let a
                            // backend that has the encoding take it.
                            let target_ty = match type_cache.get(ty).copied() {
                                Some(t) => t,
                                None => {
                                    return Err(CompilerError::UnsupportedByBackend {
                                        backend: "cranelift",
                                        construct: format!("a cast to {:?}", ty),
                                        detail: "this backend has no encoding for that type"
                                            .to_string(),
                                    });
                                }
                            };
                            let operand_clif_ty = builder.func.dfg.value_type(val);
                            if cast_needs_scalar_lanes(op, operand_clif_ty, target_ty) {
                                return Err(CompilerError::UnsupportedByBackend {
                                    backend: "cranelift",
                                    construct: format!("a vector {:?} cast", op),
                                    detail: format!(
                                        "no lane-width conversion from {:?} to {:?}",
                                        operand_clif_ty, target_ty
                                    ),
                                });
                            }

                            let cast_val = match op {
                                crate::hir::CastOp::Bitcast => val,
                                crate::hir::CastOp::ZExt => builder.ins().uextend(target_ty, val),
                                crate::hir::CastOp::SExt => builder.ins().sextend(target_ty, val),
                                crate::hir::CastOp::Trunc => builder.ins().ireduce(target_ty, val),
                                crate::hir::CastOp::FpToSi => {
                                    builder.ins().fcvt_to_sint(target_ty, val)
                                }
                                crate::hir::CastOp::FpToUi => {
                                    builder.ins().fcvt_to_uint(target_ty, val)
                                }
                                crate::hir::CastOp::SiToFp => {
                                    builder.ins().fcvt_from_sint(target_ty, val)
                                }
                                crate::hir::CastOp::UiToFp => {
                                    builder.ins().fcvt_from_uint(target_ty, val)
                                }
                                crate::hir::CastOp::FpExt => builder.ins().fpromote(target_ty, val),
                                crate::hir::CastOp::FpTrunc => {
                                    builder.ins().fdemote(target_ty, val)
                                }
                                _ => val, // PtrToInt, IntToPtr handled as bitcasts for now
                            };

                            self.value_map.insert(*result, cast_val);
                        }

                        // ================================================================
                        // Algebraic Effects Instructions (Tier 1: Simple Effects)
                        // ================================================================
                        HirInstruction::PerformEffect {
                            result,
                            effect_id,
                            op_name,
                            args,
                            return_ty,
                        } => {
                            // Tier 1 implementation: Direct call to handler function
                            //
                            // For simple (non-resumable) effects, we can compile PerformEffect
                            // as a direct function call to the handler implementation.
                            //
                            // TODO: Use analyze_perform_effect() to determine optimal strategy
                            // For now, we use a simple implementation that calls a mangled function.

                            // Look up the handler for this effect in the module.
                            // Capture `is_resumable` per the matched impl so we
                            // can pad the call args with a Resume<T> sentinel
                            // (Phase H Tier 3: resumable handlers receive an
                            // extra Resume<T> = i64 placeholder as the last
                            // arg; the SSA builder already rewrote `k(v)` in
                            // the handler body to `__zyntax_effect_resume(k, v)`,
                            // so the sentinel value itself is irrelevant —
                            // only the arity matters today).
                            // The calling convention belongs to the
                            // EFFECT, not to whichever handler happens to
                            // be declared first: this site passes its
                            // arguments to whatever handler is in scope at
                            // run time. Reading state-ness or resumability
                            // off one handler makes every scope's ABI
                            // depend on declaration order, and a scope
                            // whose handler disagrees reads its arguments
                            // one slot out.
                            let effect_has_state = hir_module
                                .handlers
                                .values()
                                .any(|h| h.effect_id == *effect_id && !h.state_fields.is_empty());
                            let op_is_resumable = hir_module
                                .handlers
                                .values()
                                .filter(|h| h.effect_id == *effect_id)
                                .flat_map(|h| h.implementations.iter())
                                .any(|i| i.op_name == *op_name && i.is_resumable);
                            let (handler_func_name, is_resumable, has_state) =
                                if let Some(handler) = hir_module
                                    .handlers
                                    .values()
                                    .find(|h| h.effect_id == *effect_id)
                                {
                                    if let Some(impl_) = handler
                                        .implementations
                                        .iter()
                                        .find(|i| i.op_name == *op_name)
                                    {
                                        (
                                            mangle_handler_op_name(handler.name, impl_.op_name),
                                            op_is_resumable,
                                            // A resumable op takes its
                                            // continuation, not an implicit
                                            // `self` — the synthesis that
                                            // adds `self` skips those.
                                            effect_has_state && !op_is_resumable,
                                        )
                                    } else {
                                        warn!(
                                        "[Effect] No implementation for operation {:?} in handler",
                                        op_name
                                    );
                                        // Fall through to trap
                                        if let Some(result_id) = result {
                                            self.value_map.insert(
                                                *result_id,
                                                builder.ins().iconst(types::I64, 0),
                                            );
                                        }
                                        continue;
                                    }
                                } else {
                                    warn!("[Effect] No handler found for effect {:?}", effect_id);
                                    // Unhandled effect - trap at runtime
                                    builder
                                        .ins()
                                        .trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
                                    if let Some(result_id) = result {
                                        self.value_map.insert(
                                            *result_id,
                                            builder.ins().iconst(types::I64, 0),
                                        );
                                    }
                                    continue;
                                };

                            // Try to find the handler function in the module
                            // Look up by name in hir_module.functions, then get FuncId from function_map
                            let handler_hir_id = hir_module
                                .functions
                                .iter()
                                .find(|(_, f)| {
                                    f.name
                                        .resolve_global()
                                        .map(|n| n == handler_func_name)
                                        .unwrap_or(false)
                                })
                                .map(|(id, _)| *id);

                            if let Some(hir_id) = handler_hir_id {
                                if let Some(&func_id) = self.function_map.get(&hir_id) {
                                    // Regional dispatch. The compile-time
                                    // `func_id` is the STATIC handler (the
                                    // one in module scope). A `with H { }`
                                    // block may override it at runtime by
                                    // pushing H's op-table onto the handler
                                    // stack. Resolve the op function pointer
                                    // dynamically:
                                    //   dyn = __zyntax_effect_lookup_op(effect, op_index)
                                    //   fn_ptr = select(dyn != 0, dyn, &static)
                                    //   call_indirect fn_ptr(args)
                                    // No branch — one `select` + one
                                    // indirect call. When no `with` is in
                                    // scope `dyn` is null and the static
                                    // handler is used (unchanged behaviour).
                                    let ptr_ty = self.module.target_config().pointer_type();

                                    let static_ref =
                                        self.module.declare_func_in_func(func_id, builder.func);
                                    let static_addr = builder.ins().func_addr(ptr_ty, static_ref);

                                    let mut arg_values: Vec<Value> = args
                                        .iter()
                                        .filter_map(|a| self.value_map.get(a).copied())
                                        .collect();
                                    // Stateful handler op takes an implicit
                                    // `self: H$state` first arg — the state
                                    // region the innermost `with H { }`
                                    // allocated. Fetch it from the handler
                                    // stack (null outside any `with`, which is
                                    // why a stateful handler must be entered
                                    // via `with`). Non-resumable only; the
                                    // continuation-lifting backend threads its
                                    // own state.
                                    if has_state && !is_resumable {
                                        let mut ls_sig = self.module.make_signature();
                                        ls_sig.params.push(AbiParam::new(types::I64));
                                        ls_sig.returns.push(AbiParam::new(ptr_ty));
                                        let state_ptr = match self.module.declare_function(
                                            "__zyntax_effect_lookup_state",
                                            Linkage::Import,
                                            &ls_sig,
                                        ) {
                                            Ok(fid) => {
                                                let lref = self
                                                    .module
                                                    .declare_func_in_func(fid, builder.func);
                                                let effc = builder
                                                    .ins()
                                                    .iconst(types::I64, effect_id.as_u32() as i64);
                                                let c = builder.ins().call(lref, &[effc]);
                                                builder.inst_results(c)[0]
                                            }
                                            Err(_) => builder.ins().iconst(ptr_ty, 0),
                                        };
                                        arg_values.insert(0, state_ptr);
                                    }
                                    // Resumable handler takes an extra
                                    // Resume<T> = i64 sentinel (arity match).
                                    if is_resumable {
                                        arg_values.push(builder.ins().iconst(types::I64, 0));
                                    }

                                    // op_index = position of this op in the
                                    // effect's operation list (stable order,
                                    // shared with op-table construction).
                                    let op_index = hir_module
                                        .effects
                                        .get(effect_id)
                                        .and_then(|e| {
                                            e.operations.iter().position(|o| o.name == *op_name)
                                        })
                                        .unwrap_or(0)
                                        as i64;
                                    let effect_u64 =
                                        builder.ins().iconst(types::I64, effect_id.as_u32() as i64);
                                    let op_index_val = builder.ins().iconst(types::I64, op_index);

                                    // dyn_fn = __zyntax_effect_lookup_op(effect, op_index)
                                    let mut lookup_sig = self.module.make_signature();
                                    lookup_sig.params.push(AbiParam::new(types::I64));
                                    lookup_sig.params.push(AbiParam::new(types::I64));
                                    lookup_sig.returns.push(AbiParam::new(ptr_ty));
                                    let dyn_fn = match self.module.declare_function(
                                        "__zyntax_effect_lookup_op",
                                        Linkage::Import,
                                        &lookup_sig,
                                    ) {
                                        Ok(fid) => {
                                            let lref =
                                                self.module.declare_func_in_func(fid, builder.func);
                                            let c = builder
                                                .ins()
                                                .call(lref, &[effect_u64, op_index_val]);
                                            builder.inst_results(c)[0]
                                        }
                                        // Lookup symbol unavailable — fall
                                        // back to static only.
                                        Err(_) => static_addr,
                                    };

                                    let zero = builder.ins().iconst(ptr_ty, 0);
                                    let is_dyn = builder.ins().icmp(IntCC::NotEqual, dyn_fn, zero);
                                    let fn_ptr = builder.ins().select(is_dyn, dyn_fn, static_addr);

                                    // Build the indirect-call signature from
                                    // the actual argument value types (matches
                                    // the handler op's compiled ABI for scalar
                                    // params) + return type.
                                    let mut call_sig = self.module.make_signature();
                                    for &av in &arg_values {
                                        let vt = builder.func.dfg.value_type(av);
                                        call_sig.params.push(AbiParam::new(vt));
                                    }
                                    let ret_ct = match return_ty {
                                        HirType::I8 | HirType::U8 | HirType::Bool => {
                                            Some(types::I8)
                                        }
                                        HirType::I16 | HirType::U16 => Some(types::I16),
                                        HirType::I32 | HirType::U32 => Some(types::I32),
                                        HirType::I64 | HirType::U64 | HirType::Ptr(_) => {
                                            Some(types::I64)
                                        }
                                        HirType::F32 => Some(types::F32),
                                        HirType::F64 => Some(types::F64),
                                        HirType::Void => None,
                                        _ => Some(types::I64),
                                    };
                                    if let Some(rc) = ret_ct {
                                        call_sig.returns.push(AbiParam::new(rc));
                                    }
                                    let sig_ref = builder.import_signature(call_sig);
                                    let call =
                                        builder.ins().call_indirect(sig_ref, fn_ptr, &arg_values);
                                    if let Some(result_id) = result {
                                        if let Some(&ret_val) = builder.inst_results(call).first() {
                                            self.value_map.insert(*result_id, ret_val);
                                        } else if matches!(return_ty, HirType::Void) {
                                            self.value_map.insert(
                                                *result_id,
                                                builder.ins().iconst(types::I64, 0),
                                            );
                                        }
                                    }
                                    continue;
                                }
                            }
                            // Handler not yet compiled or not in module - declare as external and call
                            {
                                // Handler not yet compiled - declare as external and call
                                let return_cranelift_ty = match return_ty {
                                    HirType::I8 | HirType::U8 | HirType::Bool => types::I8,
                                    HirType::I16 | HirType::U16 => types::I16,
                                    HirType::I32 | HirType::U32 => types::I32,
                                    HirType::I64 | HirType::U64 | HirType::Ptr(_) => types::I64,
                                    HirType::F32 => types::F32,
                                    HirType::F64 => types::F64,
                                    _ => types::I64, // Default for complex types
                                };
                                let mut sig = self.module.make_signature();
                                for _arg_id in args {
                                    // Get arg type from value_map (simplified - assume i64)
                                    sig.params.push(AbiParam::new(types::I64));
                                }
                                // Phase H Tier 3: resumable handler takes an
                                // extra Resume<T> = i64 sentinel param.
                                if is_resumable {
                                    sig.params.push(AbiParam::new(types::I64));
                                }
                                if !matches!(return_ty, HirType::Void) {
                                    sig.returns.push(AbiParam::new(return_cranelift_ty));
                                }

                                match self.module.declare_function(
                                    &handler_func_name,
                                    Linkage::Import,
                                    &sig,
                                ) {
                                    Ok(extern_func_id) => {
                                        let local_callee = self
                                            .module
                                            .declare_func_in_func(extern_func_id, builder.func);
                                        let mut arg_values: Vec<Value> = args
                                            .iter()
                                            .filter_map(|a| self.value_map.get(a).copied())
                                            .collect();
                                        if is_resumable {
                                            arg_values.push(builder.ins().iconst(types::I64, 0));
                                        }
                                        let call = builder.ins().call(local_callee, &arg_values);
                                        if let Some(result_id) = result {
                                            if let Some(&ret_val) =
                                                builder.inst_results(call).first()
                                            {
                                                self.value_map.insert(*result_id, ret_val);
                                            } else if matches!(return_ty, HirType::Void) {
                                                // Void return - create dummy value
                                                self.value_map.insert(
                                                    *result_id,
                                                    builder.ins().iconst(types::I64, 0),
                                                );
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        warn!(
                                            "[Effect] Failed to declare handler function {}: {}",
                                            handler_func_name, e
                                        );
                                        if let Some(result_id) = result {
                                            self.value_map.insert(
                                                *result_id,
                                                builder.ins().iconst(types::I64, 0),
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        HirInstruction::HandleEffect {
                            result,
                            handler_id,
                            handler_state,
                            body_block,
                            continuation_block,
                            return_ty,
                        } => {
                            // Tier 1 implementation: Inline handler scope
                            //
                            // For simple effects, HandleEffect sets up a handler context and
                            // the body block is compiled inline. Effect operations in the body
                            // will call the handler directly.
                            //
                            // TODO: Push handler onto effect_context stack
                            // TODO: Handle stateful handlers (allocate state)

                            if let Some(handler) = hir_module.handlers.get(handler_id) {
                                log::debug!(
                                    "[Effect] HandleEffect: installing handler {} for effect {:?}",
                                    handler.name.resolve_global().unwrap_or_default(),
                                    handler.effect_id
                                );

                                // For Tier 1, we simply note the handler is installed.
                                // The body block will be compiled as part of normal block processing.
                                // PerformEffect instructions will look up the handler.

                                // Store handler state if needed
                                if !handler_state.is_empty() {
                                    log::debug!(
                                        "[Effect] Handler has {} state values",
                                        handler_state.len()
                                    );
                                    // TODO: Allocate stack slot for handler state
                                }

                                // The body_block and continuation_block are handled by normal
                                // control flow - no special codegen needed for Tier 1.

                                // If result is expected, it comes from the body block's return
                                // This will be set when we process the body block
                            } else {
                                warn!("[Effect] HandleEffect: handler {:?} not found", handler_id);
                            }

                            // Create dummy result value - actual value comes from body
                            if let Some(result_id) = result {
                                let return_cranelift_ty = match return_ty {
                                    HirType::I8 | HirType::U8 | HirType::Bool => types::I8,
                                    HirType::I16 | HirType::U16 => types::I16,
                                    HirType::I32 | HirType::U32 => types::I32,
                                    HirType::I64 | HirType::U64 | HirType::Ptr(_) => types::I64,
                                    HirType::F32 => types::F32,
                                    HirType::F64 => types::F64,
                                    _ => types::I64,
                                };
                                if matches!(return_ty, HirType::Void) {
                                    self.value_map
                                        .insert(*result_id, builder.ins().iconst(types::I64, 0));
                                } else {
                                    self.value_map.insert(
                                        *result_id,
                                        builder.ins().iconst(return_cranelift_ty, 0),
                                    );
                                }
                            }
                        }

                        HirInstruction::Resume {
                            value,
                            continuation,
                        } => {
                            // Tier 1 implementation: No-op for non-resumable handlers
                            //
                            // For simple (non-resumable) handlers, Resume is essentially
                            // returning the value from the handler implementation.
                            // Since we compile handlers as direct calls, this is handled
                            // by the normal return mechanism.
                            //
                            // Tier 3 will need actual continuation invocation.

                            log::debug!("[Effect] Resume instruction (Tier 1: no-op, value flows via return)");

                            // The value should be returned from the handler
                            // In Tier 1, this is handled by the handler function's return
                            let _ = (value, continuation); // Suppress unused warnings
                        }

                        HirInstruction::AbortEffect {
                            value,
                            handler_scope,
                        } => {
                            // Tier 1 implementation: Jump to handler exit
                            //
                            // AbortEffect terminates the handled computation early,
                            // returning the value as the result of HandleEffect.
                            //
                            // For Tier 1, this can be implemented as a branch to the
                            // continuation block of the handler.

                            log::debug!(
                                "[Effect] AbortEffect: aborting handler scope {:?} with value {:?}",
                                handler_scope,
                                value
                            );

                            // TODO: Branch to the handler's continuation block
                            // For now, we treat this as returning the value
                            let _ = (value, handler_scope); // Suppress unused warnings
                        }

                        HirInstruction::CaptureContinuation { result, resume_ty } => {
                            // Tier 3: Not yet implemented
                            //
                            // CaptureContinuation creates a reified continuation that can
                            // be stored and resumed later. This requires:
                            // - CPS transformation
                            // - Stack frame capture
                            // - Trampoline execution model
                            //
                            // For now, we create a null/dummy continuation.

                            warn!(
                                "[Effect] CaptureContinuation: Tier 3 feature not yet implemented"
                            );

                            // Create a dummy continuation value (null pointer)
                            self.value_map
                                .insert(*result, builder.ins().iconst(types::I64, 0));
                        }

                        // Lifetime instructions (no codegen needed - used for analysis only)
                        HirInstruction::BeginLifetime { .. }
                        | HirInstruction::EndLifetime { .. }
                        | HirInstruction::LifetimeConstraint { .. } => {
                            // These are analysis-only instructions, no runtime code needed
                        }

                        // Copy instruction (simple value copy)
                        HirInstruction::Copy {
                            result,
                            source,
                            ty: _,
                        } => {
                            if let Some(&val) = self.value_map.get(source) {
                                self.value_map.insert(*result, val);
                            } else {
                                warn!("[Cranelift] Copy: source {:?} not in value_map", source);
                            }
                        }

                        // SIMD instructions — inlined (cannot delegate to translate_instruction
                        // while builder holds &mut references into self).
                        HirInstruction::VectorSplat { result, ty, scalar } => {
                            if let Some(&scalar_val) = self.value_map.get(scalar) {
                                // Resolve vector type without borrowing self (avoid E0502).
                                let clif_ty_opt = if let HirType::Vector(elem_ty, count) = ty {
                                    match (&**elem_ty, *count) {
                                        (HirType::F32, 4) => Some(types::F32X4),
                                        (HirType::F64, 2) => Some(types::F64X2),
                                        (HirType::I8, 16) | (HirType::U8, 16) => Some(types::I8X16),
                                        (HirType::I16, 8) | (HirType::U16, 8) => Some(types::I16X8),
                                        (HirType::I32, 4) | (HirType::U32, 4) => Some(types::I32X4),
                                        (HirType::I64, 2) | (HirType::U64, 2) => Some(types::I64X2),
                                        _ => None,
                                    }
                                } else {
                                    None
                                };
                                if let Some(vec_clif_ty) = clif_ty_opt {
                                    let value = builder.ins().splat(vec_clif_ty, scalar_val);
                                    self.value_map.insert(*result, value);
                                } else {
                                    // Declining hands the function to a
                                    // backend that has the encoding.
                                    // Skipping silently would leave the
                                    // result undefined and whatever
                                    // reads it wrong.
                                    return Err(CompilerError::UnsupportedByBackend {
                                        backend: "cranelift",
                                        construct: format!("a {:?} broadcast", ty),
                                        detail: "this backend holds 128-bit vectors only"
                                            .to_string(),
                                    });
                                }
                            }
                        }
                        HirInstruction::VectorExtractLane {
                            result,
                            vector,
                            lane,
                            ..
                        } => {
                            if let Some(&vec_val) = self.value_map.get(vector) {
                                let value = builder.ins().extractlane(vec_val, *lane);
                                self.value_map.insert(*result, value);
                            }
                        }
                        HirInstruction::VectorInsertLane {
                            result,
                            vector,
                            scalar,
                            lane,
                            ..
                        } => {
                            if let (Some(&vec_val), Some(&scalar_val)) =
                                (self.value_map.get(vector), self.value_map.get(scalar))
                            {
                                let value = builder.ins().insertlane(vec_val, scalar_val, *lane);
                                self.value_map.insert(*result, value);
                            }
                        }
                        HirInstruction::VectorHorizontalReduce {
                            result,
                            ty: _,
                            vector,
                            op,
                        } => {
                            if let Some(&vec_val) = self.value_map.get(vector) {
                                // Derive lane count from the input vector's Cranelift type
                                // (avoids borrowing self via translate_type while builder is live).
                                let vec_clif_ty = builder.func.dfg.value_type(vec_val);
                                let lane_count: u8 = match vec_clif_ty {
                                    types::I8X16 => 16,
                                    types::I16X8 => 8,
                                    types::F32X4 | types::I32X4 => 4,
                                    types::F64X2 | types::I64X2 => 2,
                                    _ => {
                                        warn!("[Cranelift] VectorHorizontalReduce: unsupported type {:?}", vec_clif_ty);
                                        0
                                    }
                                };
                                // Integer horizontal-add via a pairwise-reduction
                                // tree instead of a serial extractlane chain: for
                                // i32x4/i64x2 the AArch64 backend folds
                                // `extractlane(iadd_pairwise(iadd_pairwise(x,x),
                                // iadd_pairwise(x,x)), 0)` into a single `addv`
                                // (a pattern the serial chain never matches). GVN
                                // shares the repeated operand so the tree stays the
                                // exact matched shape. Restricted to widths that
                                // can't overflow the pairwise accumulation.
                                let pairwise = matches!(op, BinaryOp::Add)
                                    && matches!(vec_clif_ty, types::I32X4 | types::I64X2);
                                if pairwise {
                                    let mut v = vec_val;
                                    let mut n = lane_count;
                                    while n > 1 {
                                        v = builder.ins().iadd_pairwise(v, v);
                                        n /= 2;
                                    }
                                    let acc = builder.ins().extractlane(v, 0u8);
                                    self.value_map.insert(*result, acc);
                                } else if lane_count > 0 {
                                    // Serial fallback: float-add, or ops/widths with
                                    // no horizontal hardware idiom.
                                    let mut acc = builder.ins().extractlane(vec_val, 0u8);
                                    for lane_idx in 1..lane_count {
                                        let lane_val = builder.ins().extractlane(vec_val, lane_idx);
                                        acc = match op {
                                            BinaryOp::Add => builder.ins().iadd(acc, lane_val),
                                            BinaryOp::FAdd => builder.ins().fadd(acc, lane_val),
                                            BinaryOp::Sub => builder.ins().isub(acc, lane_val),
                                            BinaryOp::FSub => builder.ins().fsub(acc, lane_val),
                                            BinaryOp::Mul => builder.ins().imul(acc, lane_val),
                                            BinaryOp::FMul => builder.ins().fmul(acc, lane_val),
                                            _ => {
                                                warn!("[Cranelift] VectorHorizontalReduce: unsupported op {:?}", op);
                                                acc
                                            }
                                        };
                                    }
                                    self.value_map.insert(*result, acc);
                                }
                            }
                        }

                        // SIMD memory operations: load/store an entire vector register
                        // from/to a contiguous block of elements in memory.
                        HirInstruction::VectorLoad {
                            result,
                            ty,
                            ptr,
                            align: _,
                        } => {
                            if let Some(&ptr_val) = self.value_map.get(ptr) {
                                let clif_ty_opt = if let HirType::Vector(elem_ty, count) = ty {
                                    match (&**elem_ty, *count) {
                                        (HirType::F32, 4) => Some(types::F32X4),
                                        (HirType::F64, 2) => Some(types::F64X2),
                                        (HirType::I8, 16) | (HirType::U8, 16) => Some(types::I8X16),
                                        (HirType::I16, 8) | (HirType::U16, 8) => Some(types::I16X8),
                                        (HirType::I32, 4) | (HirType::U32, 4) => Some(types::I32X4),
                                        (HirType::I64, 2) | (HirType::U64, 2) => Some(types::I64X2),
                                        _ => None,
                                    }
                                } else {
                                    None
                                };
                                if let Some(vec_clif_ty) = clif_ty_opt {
                                    let flags = MemFlags::new().with_notrap();
                                    let value = builder.ins().load(vec_clif_ty, flags, ptr_val, 0);
                                    self.value_map.insert(*result, value);
                                } else {
                                    return Err(CompilerError::UnsupportedByBackend {
                                        backend: "cranelift",
                                        construct: format!("a {:?} load", ty),
                                        detail: "this backend holds 128-bit vectors only"
                                            .to_string(),
                                    });
                                }
                            }
                        }
                        HirInstruction::VectorStore {
                            value,
                            ptr,
                            align: _,
                        } => {
                            if let (Some(&vec_val), Some(&ptr_val)) =
                                (self.value_map.get(value), self.value_map.get(ptr))
                            {
                                let flags = MemFlags::new().with_notrap();
                                builder.ins().store(flags, vec_val, ptr_val, 0);
                            }
                        }
                        HirInstruction::VectorUnaryOp {
                            result,
                            ty: _,
                            op,
                            operand,
                        } => {
                            if let Some(&v) = self.value_map.get(operand) {
                                let r = match op {
                                    VectorUnaryKind::Sqrt => builder.ins().sqrt(v),
                                    VectorUnaryKind::Abs => builder.ins().fabs(v),
                                    VectorUnaryKind::Neg => builder.ins().fneg(v),
                                    VectorUnaryKind::Ceil => builder.ins().ceil(v),
                                    VectorUnaryKind::Floor => builder.ins().floor(v),
                                    VectorUnaryKind::Trunc => builder.ins().trunc(v),
                                    VectorUnaryKind::Round => builder.ins().nearest(v),
                                };
                                self.value_map.insert(*result, r);
                            }
                        }
                        HirInstruction::VectorMinMax {
                            result,
                            ty: _,
                            op,
                            left,
                            right,
                        } => {
                            if let (Some(&l), Some(&r)) =
                                (self.value_map.get(left), self.value_map.get(right))
                            {
                                let out = match op {
                                    VectorMinMaxKind::Min => builder.ins().fmin(l, r),
                                    VectorMinMaxKind::Max => builder.ins().fmax(l, r),
                                };
                                self.value_map.insert(*result, out);
                            }
                        }
                        HirInstruction::VectorDot {
                            result,
                            acc,
                            a,
                            b,
                            rhs_i7: _,
                            rhs_unsigned,
                        } => {
                            if let (Some(&a_v), Some(&b_v), Some(&acc_v)) = (
                                self.value_map.get(a),
                                self.value_map.get(b),
                                self.value_map.get(acc),
                            ) {
                                // The exact widen→imul→(widen-each-product)→
                                // iadd_pairwise×3→iadd(acc) tree the AArch64
                                // backend folds into ONE `sdot` on FEAT_DotProd
                                // hosts. Both operands are sign-widened and each
                                // i16 product is widened to i32 SEPARATELY, then
                                // reduced entirely in i32 — early narrowing or
                                // reassociation silently drops the fold to
                                // smull/addp. `rhs_unsigned` uses uwiden (correct,
                                // but no SDOT rule → generic pairwise fallback).
                                let a_lo = builder.ins().swiden_low(a_v);
                                let a_hi = builder.ins().swiden_high(a_v);
                                let (b_lo, b_hi) = if *rhs_unsigned {
                                    (
                                        builder.ins().uwiden_low(b_v),
                                        builder.ins().uwiden_high(b_v),
                                    )
                                } else {
                                    (
                                        builder.ins().swiden_low(b_v),
                                        builder.ins().swiden_high(b_v),
                                    )
                                };
                                let lo = builder.ins().imul(a_lo, b_lo);
                                let hi = builder.ins().imul(a_hi, b_hi);
                                let lo_w_lo = builder.ins().swiden_low(lo);
                                let lo_w_hi = builder.ins().swiden_high(lo);
                                let hi_w_lo = builder.ins().swiden_low(hi);
                                let hi_w_hi = builder.ins().swiden_high(hi);
                                let inner_lo = builder.ins().iadd_pairwise(lo_w_lo, lo_w_hi);
                                let inner_hi = builder.ins().iadd_pairwise(hi_w_lo, hi_w_hi);
                                let pair = builder.ins().iadd_pairwise(inner_lo, inner_hi);
                                let out = builder.ins().iadd(pair, acc_v);
                                self.value_map.insert(*result, out);
                            }
                        }

                        // Phase E3: async state-machine slot ops.
                        // Frame is a future-struct pointer; each slot is an
                        // 8-byte cell at offset (slot * 8). Saves bit-cast
                        // the value through i64; Loads bit-cast back. The
                        // slot 0 reserved for the state-id is just an i64
                        // store/load with no bit-cast needed.
                        HirInstruction::AsyncSaveSlot { frame, slot, value } => {
                            if let (Some(&frame_val), Some(&store_val)) =
                                (self.value_map.get(frame), self.value_map.get(value))
                            {
                                let actual_ty = builder.func.dfg.value_type(store_val);
                                // Bit-cast through i64 so every slot is
                                // uniformly 8 bytes. For ints smaller than
                                // i64, zero-extend; for floats, bitcast;
                                // for pointers/i64 use as-is.
                                let i64_val = if actual_ty == types::I64 {
                                    store_val
                                } else if actual_ty.is_int() {
                                    builder.ins().uextend(types::I64, store_val)
                                } else if actual_ty == types::F32 {
                                    let as_i32 = builder.ins().bitcast(
                                        types::I32,
                                        MemFlags::new(),
                                        store_val,
                                    );
                                    builder.ins().uextend(types::I64, as_i32)
                                } else if actual_ty == types::F64 {
                                    builder
                                        .ins()
                                        .bitcast(types::I64, MemFlags::new(), store_val)
                                } else {
                                    builder
                                        .ins()
                                        .bitcast(types::I64, MemFlags::new(), store_val)
                                };
                                let flags = MemFlags::new().with_notrap();
                                builder
                                    .ins()
                                    .store(flags, i64_val, frame_val, (*slot as i32) * 8);
                            }
                        }
                        HirInstruction::AsyncLoadSlot {
                            result,
                            ty,
                            frame,
                            slot,
                        } => {
                            if let Some(&frame_val) = self.value_map.get(frame) {
                                let target_ty = type_cache.get(ty).copied().unwrap_or(types::I64);
                                let flags = MemFlags::new().with_notrap();
                                let raw = builder.ins().load(
                                    types::I64,
                                    flags,
                                    frame_val,
                                    (*slot as i32) * 8,
                                );
                                // Inverse of save's bit-cast.
                                let typed = if target_ty == types::I64 {
                                    raw
                                } else if target_ty.is_int() {
                                    builder.ins().ireduce(target_ty, raw)
                                } else if target_ty == types::F32 {
                                    let low = builder.ins().ireduce(types::I32, raw);
                                    builder.ins().bitcast(types::F32, MemFlags::new(), low)
                                } else if target_ty == types::F64 {
                                    builder.ins().bitcast(types::F64, MemFlags::new(), raw)
                                } else {
                                    builder.ins().bitcast(target_ty, MemFlags::new(), raw)
                                };
                                self.value_map.insert(*result, typed);
                            }
                        }

                        _ => {
                            // Other instructions not yet implemented
                            // This will cause values to be unmapped, leading to verifier errors
                            warn!(" Unimplemented instruction type: {:?}", inst);
                        }
                    }
                }

                // Process terminator (inline to avoid borrow checker)
                match &hir_block.terminator {
                    HirTerminator::Return { values } => {
                        log::debug!("[Cranelift] Return terminator with values: {:?}", values);

                        // An aggregate built here lives in this frame,
                        // which the caller outlives. Copy it into the
                        // destination the caller passed and hand that
                        // back, so the value survives the return and two
                        // calls do not share one buffer.
                        if let (Some(dest), Some(bytes)) = (destination_ptr, destination_return) {
                            if let Some(&src) = values.first().and_then(|v| self.value_map.get(v)) {
                                if bytes <= INLINE_COPY_MAX_BYTES {
                                    emit_inline_aggregate_copy(&mut builder, dest, src, bytes);
                                } else {
                                    let size_val = builder.ins().iconst(types::I64, bytes as i64);
                                    builder.call_memcpy(
                                        self.module.target_config(),
                                        dest,
                                        src,
                                        size_val,
                                    );
                                }
                            }
                            builder.ins().return_(&[dest]);
                        } else {
                            let expected_returns = builder.func.signature.returns.clone();
                            let mut cranelift_vals = Vec::new();
                            for (i, v) in values.iter().enumerate() {
                                if let Some(&val) = self.value_map.get(v) {
                                    let coerced =
                                        if let Some(expected_abi) = expected_returns.get(i) {
                                            let actual_ty = builder.func.dfg.value_type(val);
                                            Self::coerce_value(
                                                &mut builder,
                                                val,
                                                actual_ty,
                                                expected_abi.value_type,
                                            )
                                        } else {
                                            val
                                        };
                                    cranelift_vals.push(coerced);
                                } else {
                                    log::trace!(
                                        "[Cranelift ERROR] Return value {:?} not in value_map",
                                        v
                                    );
                                }
                            }
                            // Bare return: pad with zero values if signature expects more
                            while cranelift_vals.len() < expected_returns.len() {
                                let ty = expected_returns[cranelift_vals.len()].value_type;
                                let zero = Self::emit_zero_value(&mut builder, ty);
                                cranelift_vals.push(zero);
                            }
                            log::debug!("[Cranelift] Returning {} values", cranelift_vals.len());
                            builder.ins().return_(&cranelift_vals);
                        }
                    }

                    HirTerminator::Branch { target } => {
                        let target_block = self.block_map[target];

                        // Inline phi args extraction
                        // FIXED: phi.incoming format is (value, block), not (block, value)
                        let args: Vec<Value> =
                            if let Some(target_hir_block) = function.blocks.get(target) {
                                let mut args_vec = Vec::new();
                                for phi in &target_hir_block.phis {
                                    if let Some((value, _)) = phi
                                        .incoming
                                        .iter()
                                        .find(|(_, pred_block)| *pred_block == *hir_block_id)
                                    {
                                        if let Some(&cranelift_val) = self.value_map.get(value) {
                                            // A block parameter has the
                                            // phi's declared type, and
                                            // an incoming value is not
                                            // obliged to already carry
                                            // it: an `i64` counter
                                            // starting at a literal `0`
                                            // arrives as an i32. The
                                            // return path coerces for
                                            // the same reason; a branch
                                            // argument that disagrees
                                            // with its parameter is
                                            // rejected by the verifier.
                                            let want =
                                                type_cache.get(&phi.ty).copied().unwrap_or_else(
                                                    || builder.func.dfg.value_type(cranelift_val),
                                                );
                                            let actual = builder.func.dfg.value_type(cranelift_val);
                                            args_vec.push(Self::coerce_value(
                                                &mut builder,
                                                cranelift_val,
                                                actual,
                                                want,
                                            ));
                                        }
                                        // Note: Values should now all be in value_map after pure IDF fix
                                    }
                                }
                                args_vec
                            } else {
                                vec![]
                            };

                        builder.ins().jump(target_block, &ba(&args));

                        if let Some(count) = seal_tracker.get_mut(target) {
                            *count = count.saturating_sub(1);
                            if *count == 0 && !sealed_blocks.contains(target) {
                                builder.seal_block(target_block);
                                sealed_blocks.insert(*target);
                            }
                        }
                    }

                    HirTerminator::CondBranch {
                        condition,
                        true_target,
                        false_target,
                    } => {
                        let cond = self.value_map[condition];
                        let true_block = self.block_map[true_target];
                        let false_block = self.block_map[false_target];

                        // Inline phi args extraction for true branch
                        // FIXED: phi.incoming format is (value, block), not (block, value)
                        let true_args: Vec<Value> = if let Some(target_hir_block) =
                            function.blocks.get(true_target)
                        {
                            target_hir_block
                                .phis
                                .iter()
                                .filter_map(|phi| {
                                    phi.incoming
                                        .iter()
                                        .find(|(_, pred_block)| *pred_block == *hir_block_id)
                                        .and_then(|(value, _)| self.value_map.get(value).copied())
                                })
                                .collect()
                        } else {
                            vec![]
                        };

                        // Inline phi args extraction for false branch
                        // FIXED: phi.incoming format is (value, block), not (block, value)
                        let false_args: Vec<Value> = if let Some(target_hir_block) =
                            function.blocks.get(false_target)
                        {
                            target_hir_block
                                .phis
                                .iter()
                                .filter_map(|phi| {
                                    phi.incoming
                                        .iter()
                                        .find(|(_, pred_block)| *pred_block == *hir_block_id)
                                        .and_then(|(value, _)| self.value_map.get(value).copied())
                                })
                                .collect()
                        } else {
                            vec![]
                        };

                        builder.ins().brif(
                            cond,
                            true_block,
                            &ba(&true_args),
                            false_block,
                            &ba(&false_args),
                        );

                        for target in [true_target, false_target] {
                            let target_block = self.block_map[target];
                            if let Some(count) = seal_tracker.get_mut(target) {
                                *count = count.saturating_sub(1);
                                if *count == 0 && !sealed_blocks.contains(target) {
                                    builder.seal_block(target_block);
                                    sealed_blocks.insert(*target);
                                }
                            }
                        }
                    }

                    HirTerminator::Switch {
                        value,
                        default,
                        cases,
                    } => {
                        let switch_val = self.value_map[value];
                        let default_block = self.block_map[default];

                        // Use cranelift_frontend::Switch which automatically emits
                        // br_table for dense integer ranges and brif chains for sparse
                        // ranges, choosing the more efficient representation.
                        let mut clif_switch = ClifSwitch::new();

                        for (constant, target) in cases.iter() {
                            let target_block = self.block_map[target];
                            // Extract the integer discriminant value
                            let disc: Option<u128> = match constant {
                                HirConstant::I8(v) => Some(*v as u8 as u128),
                                HirConstant::I16(v) => Some(*v as u16 as u128),
                                HirConstant::I32(v) => Some(*v as u32 as u128),
                                HirConstant::I64(v) => Some(*v as u64 as u128),
                                HirConstant::U8(v) => Some(*v as u128),
                                HirConstant::U16(v) => Some(*v as u128),
                                HirConstant::U32(v) => Some(*v as u128),
                                HirConstant::U64(v) => Some(*v as u128),
                                _ => None,
                            };
                            if let Some(idx) = disc {
                                clif_switch.set_entry(idx, target_block);
                            }
                        }

                        clif_switch.emit(&mut builder, switch_val, default_block);

                        // Seal all target blocks (Switch already finalised the terminator)
                        for (_, target) in cases.iter() {
                            if let Some(count) = seal_tracker.get_mut(target) {
                                *count = count.saturating_sub(1);
                                if *count == 0 && !sealed_blocks.contains(target) {
                                    builder.seal_block(self.block_map[target]);
                                    sealed_blocks.insert(*target);
                                }
                            }
                        }
                        if let Some(count) = seal_tracker.get_mut(default) {
                            *count = count.saturating_sub(1);
                            if *count == 0 && !sealed_blocks.contains(default) {
                                builder.seal_block(default_block);
                                sealed_blocks.insert(*default);
                            }
                        }
                    }

                    HirTerminator::PatternMatch {
                        value,
                        patterns,
                        default,
                    } => {
                        // For now, pattern matching on constants is lowered to Switch
                        // Extract constant patterns
                        let cases: Vec<(HirConstant, HirId)> = patterns
                            .iter()
                            .filter_map(|pattern| {
                                if let HirPatternKind::Constant(ref c) = pattern.kind {
                                    Some((c.clone(), pattern.target))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        let default_target = default.unwrap_or_else(|| {
                            // If no default, this is unreachable - create unreachable
                            warn!(" PatternMatch without default target, using unreachable");
                            *hir_block_id // Fallback to current block (will cause error)
                        });

                        // Reuse Switch implementation
                        let switch_val = self.value_map[value];
                        let default_block = self.block_map[&default_target];

                        let mut current_block_filled = false;

                        for (i, (constant, target)) in cases.iter().enumerate() {
                            let target_block = self.block_map[target];

                            let const_val = match constant {
                                HirConstant::I8(v) => {
                                    let extended = (*v as u8) as i64;
                                    builder.ins().iconst(types::I8, extended)
                                }
                                HirConstant::I16(v) => {
                                    let extended = (*v as u16) as i64;
                                    builder.ins().iconst(types::I16, extended)
                                }
                                HirConstant::I32(v) => {
                                    let extended = (*v as u32) as i64;
                                    builder.ins().iconst(types::I32, extended)
                                }
                                HirConstant::I64(v) => builder.ins().iconst(types::I64, *v),
                                _ => continue,
                            };

                            let cmp = builder.ins().icmp(IntCC::Equal, switch_val, const_val);

                            if i == cases.len() - 1 {
                                builder
                                    .ins()
                                    .brif(cmp, target_block, &[], default_block, &[]);
                                current_block_filled = true;

                                for target_id in [target, &default_target] {
                                    if let Some(count) = seal_tracker.get_mut(target_id) {
                                        *count = count.saturating_sub(1);
                                        if *count == 0 && !sealed_blocks.contains(target_id) {
                                            builder.seal_block(self.block_map[target_id]);
                                            sealed_blocks.insert(*target_id);
                                        }
                                    }
                                }
                            } else {
                                let next_block = builder.create_block();
                                builder.ins().brif(cmp, target_block, &[], next_block, &[]);

                                if let Some(count) = seal_tracker.get_mut(target) {
                                    *count = count.saturating_sub(1);
                                    if *count == 0 && !sealed_blocks.contains(target) {
                                        builder.seal_block(target_block);
                                        sealed_blocks.insert(*target);
                                    }
                                }

                                builder.switch_to_block(next_block);
                                builder.seal_block(next_block);
                            }
                        }

                        if cases.is_empty() && !current_block_filled {
                            builder.ins().jump(default_block, &[]);
                            if let Some(count) = seal_tracker.get_mut(&default_target) {
                                *count = count.saturating_sub(1);
                                if *count == 0 && !sealed_blocks.contains(&default_target) {
                                    builder.seal_block(default_block);
                                    sealed_blocks.insert(default_target);
                                }
                            }
                        }
                    }

                    HirTerminator::Unreachable => {
                        // For void-returning functions, emit a return instead of trap
                        // This handles Haxe/other languages where main() returns Void and has no explicit return
                        if function.signature.returns.is_empty()
                            || function
                                .signature
                                .returns
                                .iter()
                                .all(|r| matches!(r, HirType::Void))
                        {
                            builder.ins().return_(&[]);
                        } else {
                            builder
                                .ins()
                                .trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
                        }
                    }

                    _ => {
                        // Other terminators not implemented
                        warn!(" Unimplemented terminator in block {:?}", hir_block_id);
                    }
                }
            }

            // CRITICAL: Seal all blocks before finalization
            // Some blocks may have been created but not sealed (e.g., break targets)
            for &hir_block_id in &block_order {
                let cranelift_block = block_map[&hir_block_id];
                if !sealed_blocks.contains(&hir_block_id) {
                    builder.seal_block(cranelift_block);
                    sealed_blocks.insert(hir_block_id);
                }
            }

            // Finalize builder. Cranelift's `FunctionBuilder::finalize` now
            // takes the target frontend config (used to validate/complete the
            // function against the target's ABI); source it from the JIT module.
            let frontend_config = self.module.target_config();
            builder.finalize(frontend_config);
        }

        // Debug: Print IR after finalize
        log::debug!(
            "[Cranelift] IR after finalize (inside compile_function_body):\n{}",
            self.codegen_context.func
        );

        // Verify the generated IR (catches errors before they become cryptic panics)
        if let Err(errors) = verify_function(&self.codegen_context.func, self.module.isa()) {
            debug!(
                "[Cranelift] IR verification failed for function '{}': {}",
                function.name, errors
            );
            if std::env::var("ZYNTAX_TRACE_CRANELIFT_SKIP").is_ok() {
                eprintln!(
                    "[CRANELIFT-SKIP] failing CLIF:\n{}",
                    self.codegen_context.func.display()
                );
            }
            return Err(CompilerError::Backend(format!(
                "Cranelift IR verification failed for function '{}': {}",
                function.name, errors
            )));
        }
        log::debug!("[Cranelift] IR verification passed for '{}'", function.name);

        // Debug: Uncomment to dump IR for all functions
        // self.dump_cranelift_ir(&function.name.to_string());

        // CLIF dump gate: ZYNTAX_DUMP_CRANELIFT=<substring> dumps any
        // function whose name contains <substring>. Useful for hot-loop
        // audits — see WORKFLOW: re-audit nbody dominant cycle cost.
        if let Ok(filter) = std::env::var("ZYNTAX_DUMP_CRANELIFT") {
            let fname = function
                .name
                .resolve_global()
                .unwrap_or_else(|| function.name.to_string());
            if filter.is_empty() || fname.contains(&filter) {
                eprintln!(
                    "===== ZYNTAX_DUMP_CRANELIFT: {} =====\n{}\n===== end {} =====",
                    fname,
                    self.codegen_context.func.display(),
                    fname
                );
            }
        }

        // Verification capture: enable disasm generation + snapshot the CLIF
        // before the module consumes the context.
        let clif_snapshot = if self.capture_ir {
            self.codegen_context.set_disasm(true);
            Some(self.codegen_context.func.display().to_string())
        } else {
            None
        };

        // Compile the function
        log::debug!(
            "[Cranelift] About to call define_function for {:?}",
            function.name
        );
        let code = self
            .module
            .define_function(func_id, &mut self.codegen_context)
            .map_err(|e| {
                error!("Function compilation failed for: {}", function.name);
                error!("Error: {}", e);
                debug!("Function dump:\n{}", self.codegen_context.func.display());
                CompilerError::Backend(format!("Failed to compile function: {}", e))
            })?;

        // Store compiled function info - will get actual pointer after finalization.
        // OSR helpers don't share id-space with HIR functions; we track their
        // FuncIds in pending_osr_helpers instead, so skip this insert.
        if std::env::var("ZYNTAX_TRACE_CRANELIFT_SKIP").is_ok() {
            eprintln!(
                "[reg] {id:?} helper_mode={} registered={}",
                osr_helper.is_some(),
                osr_helper.is_none()
            );
        }
        if osr_helper.is_none() {
            let compiled_func = CompiledFunction {
                function_id: func_id,
                version: 1,
                code_ptr: std::ptr::null(),
                size: 0, // Will be updated after finalization
                signature: sig,
            };
            self.compiled_functions.insert(id, compiled_func);
        }

        // Resolve the tagged probe sites to code offsets. `MachSrcLoc.start`
        // is relative to the start of the function, which is what a patcher
        // needs once the JIT hands back the function's address.
        self.probe_sites.clear();
        self.last_code_len = None;
        if let Some(cc) = self.codegen_context.compiled_code() {
            self.last_code_len = Some(cc.buffer.data().len());
            if !probe_site_tags.is_empty() {
                for loc in cc.buffer.get_srclocs_sorted() {
                    let bits = loc.loc.bits();
                    if let Some((site_key, _)) =
                        probe_site_tags.iter().find(|(_, tag)| *tag == bits)
                    {
                        self.probe_sites.push((*site_key, loc.start));
                    }
                }
            }
        }

        // Verification capture: grab the native disassembly the just-completed
        // compile produced, pair it with the CLIF snapshot.
        if let Some(clif) = clif_snapshot {
            let disasm = self
                .codegen_context
                .compiled_code()
                .and_then(|c| c.vcode.clone());
            self.captured_ir = Some((clif, disasm));
        }

        // Clear context for next function
        self.codegen_context.clear();
        self.value_map.clear();
        self.block_map.clear();

        Ok(())
    }

    /// Enable/disable CLIF + disassembly capture (verification tests only).
    pub fn set_capture_ir(&mut self, on: bool) {
        self.capture_ir = on;
    }

    /// Take the last captured `(CLIF, Option<native disassembly>)`.
    pub fn take_captured_ir(&mut self) -> Option<(String, Option<String>)> {
        self.captured_ir.take()
    }

    /// Compile a global variable (including vtables)
    ///
    /// This emits global data declarations for constants, static variables,
    /// and vtables (arrays of function pointers for trait dispatch).
    pub fn compile_global(&mut self, id: HirId, global: &HirGlobal) -> CompilerResult<()> {
        self.declare_global(id, global)?;
        self.define_global(id, global)
    }

    /// Reserve `id`'s data slot without emitting its contents, so code
    /// that references the global can compile before the definition —
    /// which a dispatch table needs, since its contents are the
    /// addresses of functions compiled afterwards.
    pub fn declare_global(&mut self, id: HirId, global: &HirGlobal) -> CompilerResult<()> {
        // A vtable is a dispatch table — hot reload patches its slots
        // in place, so it must stay writable; everything else is
        // immutable.
        let unique_name = format!("global__{:?}", id);
        let writable = matches!(&global.initializer, Some(HirConstant::VTable(_)));
        let data_id = self
            .module
            .declare_data(
                &unique_name,
                cranelift_module::Linkage::Export,
                writable,
                false,
            )
            .map_err(|e| CompilerError::CodeGen(format!("Failed to declare global: {}", e)))?;
        self.global_map.insert(id, data_id);
        Ok(())
    }

    /// Reserve a dispatch table and define it as empty slots, for a
    /// caller that fills them with function addresses afterwards.
    ///
    /// Emitting the table with relocations instead would bind it to
    /// symbols that are not defined yet, and every later finalize —
    /// including the ones that follow each function compile — would
    /// fail on them. Empty slots are a complete definition, so nothing
    /// dangles, and the addresses land through the same atomic store
    /// that reload uses to patch a table already in use.
    pub fn define_empty_vtable(&mut self, id: HirId, slots: usize) -> CompilerResult<()> {
        let ptr_size = self.module.target_config().pointer_bytes() as usize;
        let data_id = *self.global_map.get(&id).ok_or_else(|| {
            CompilerError::CodeGen(format!("global {id:?} defined before it was declared"))
        })?;
        self.data_desc.clear();
        self.data_desc.define_zeroinit(slots * ptr_size);
        self.module
            .define_data(data_id, &self.data_desc)
            .map_err(|e| CompilerError::CodeGen(format!("Failed to define global: {}", e)))?;
        Ok(())
    }

    /// Emit the contents of a global declared by [`Self::declare_global`].
    pub fn define_global(&mut self, id: HirId, global: &HirGlobal) -> CompilerResult<()> {
        let data_id = *self.global_map.get(&id).ok_or_else(|| {
            CompilerError::CodeGen(format!("global {id:?} defined before it was declared"))
        })?;
        self.data_desc.clear();

        // If this is a vtable, emit function pointer array
        if let Some(HirConstant::VTable(vtable)) = &global.initializer {
            // Emit vtable as array of function pointers
            let ptr_size = self.module.target_config().pointer_bytes() as usize;
            let vtable_size = vtable.methods.len() * ptr_size;

            // Define the vtable data with correct size
            self.data_desc.define_zeroinit(vtable_size);

            // Emit function pointers using declare_func_in_data + write_function_addr
            // This creates relocations that the linker will resolve
            for (index, method_entry) in vtable.methods.iter().enumerate() {
                // Get the FuncId from function_map
                if let Some(func_id) = self.function_map.get(&method_entry.function_id) {
                    // Declare function in data context - this gives us a FuncRef
                    let func_ref = self
                        .module
                        .declare_func_in_data(*func_id, &mut self.data_desc);

                    // Write function address at appropriate offset
                    let offset = (index * ptr_size) as u32;
                    self.data_desc.write_function_addr(offset, func_ref);
                } else {
                    warn!(
                        " Vtable method function {:?} not found in function_map",
                        method_entry.function_id
                    );
                }
            }

            info!(
                "Vtable with {} methods emitted with function pointer relocations",
                vtable.methods.len()
            );
        } else if let Some(HirConstant::String(s)) = &global.initializer {
            // String constants - emit as ZRTL String format: [length: i32][utf8_bytes...]
            let string_val = s.resolve_global().ok_or_else(|| {
                CompilerError::CodeGen(format!("Failed to resolve string constant: {:?}", s))
            })?;

            // Get UTF-8 bytes
            let bytes = string_val.as_bytes();
            let length = bytes.len() as i32;

            // Create ZRTL String structure: length header (i32) + UTF-8 bytes
            let mut data = Vec::with_capacity(4 + bytes.len());
            data.extend_from_slice(&length.to_le_bytes()); // Length as little-endian i32
            data.extend_from_slice(bytes);

            // Set 4-byte alignment for the i32 length header
            self.data_desc.set_align(4);
            self.data_desc.define(data.into_boxed_slice());
        } else if global.initializer.is_some() {
            // Other constants - emit as zeroinit placeholder for now
            // TODO: Implement other constant types
            let size = self.type_size(&global.ty).unwrap_or(8) as usize;
            self.data_desc.define_zeroinit(size);
        } else {
            // No initializer, define as zeroinit based on type
            let size = self.type_size(&global.ty).unwrap_or(8) as usize;
            self.data_desc.define_zeroinit(size);
        }

        // Define the data
        self.module
            .define_data(data_id, &self.data_desc)
            .map_err(|e| CompilerError::CodeGen(format!("Failed to define global: {}", e)))?;

        Ok(())
    }

    /// Translate function signature
    pub fn translate_signature(&self, function: &HirFunction) -> CompilerResult<Signature> {
        self.translate_signature_for(function.id, function)
    }

    /// [`Self::translate_signature`] for a function known by `id`.
    ///
    /// A body can be registered under an id other than its own — hot
    /// reload swaps one in under the id its callers already name — and
    /// the convention belongs to the id those callers use.
    fn translate_signature_for(
        &self,
        id: HirId,
        function: &HirFunction,
    ) -> CompilerResult<Signature> {
        log::debug!(
            "[Cranelift] translate_signature for function {:?}",
            function.name
        );
        log::debug!(
            "[Cranelift]   params: {:?}",
            function.signature.params.len()
        );
        log::debug!("[Cranelift]   returns: {:?}", function.signature.returns);

        let mut cranelift_sig = self.module.make_signature();
        // What was decided when this function was first declared, if it
        // was. The convention is fixed at that point: callers have been
        // emitted against it, and a second declaration under the same
        // name with a signature that moved is one Cranelift rejects.
        let destination_return = self
            .destination_returns
            .get(&id)
            .copied()
            .or_else(|| self.returns_through_destination(id, function));

        // Set calling convention. `module.make_signature()` starts with the
        // ISA default, which is the native ABI for host externs: SystemV on
        // Unix x86_64, WindowsFastcall on x86_64-msvc, AppleAarch64 on ARM
        // macOS, etc. Keep both C and System on that default so calls into
        // Rust `extern "C"`/ZRTL symbols use the platform ABI.
        cranelift_sig.call_conv = match function.calling_convention {
            crate::hir::CallingConvention::C => cranelift_sig.call_conv,
            crate::hir::CallingConvention::Fast => CallConv::Fast,
            crate::hir::CallingConvention::System => cranelift_sig.call_conv,
            crate::hir::CallingConvention::WebKit => CallConv::Fast,
        };

        // The destination for a returned aggregate leads the declared
        // parameters, so a caller can pass it without knowing anything
        // about the rest of them.
        if destination_return.is_some() {
            cranelift_sig
                .params
                .push(AbiParam::new(self.module.target_config().pointer_type()));
        }

        // Add parameters
        for param in &function.signature.params {
            let ty = self.translate_type(&param.ty)?;
            cranelift_sig.params.push(AbiParam::new(ty));
        }

        // Add return types
        for ret_ty in &function.signature.returns {
            if *ret_ty != HirType::Void {
                let ty = self.translate_type(ret_ty)?;
                cranelift_sig.returns.push(AbiParam::new(ty));
            }
        }

        Ok(cranelift_sig)
    }

    /// How many bytes `function` writes through a caller-provided
    /// destination, when that is how it returns.
    ///
    /// See [`destination_return_type`] for which shapes qualify. A
    /// function whose address is observable is excluded: its callers may
    /// hold only a pointer, which carries no room for the extra
    /// argument.
    fn returns_through_destination(&self, id: HirId, function: &HirFunction) -> Option<u32> {
        if self.address_taken.contains(&id) || self.address_taken.contains(&function.id) {
            return None;
        }
        let ty = destination_return_type(function)?;
        let size = self.type_size(ty).unwrap_or(0);
        (size > 0).then_some(size as u32)
    }

    /// Convert `val` to `expected`, preserving its numeric value.
    ///
    /// Distinct from [`Self::coerce_value`], which reinterprets the
    /// bits when the widths happen to match. That is right for an ABI
    /// slot and wrong for arithmetic: `2i64` bitcast to `f64` is not
    /// `2.0`.
    fn convert_numeric(
        builder: &mut FunctionBuilder,
        val: Value,
        actual: types::Type,
        expected: types::Type,
    ) -> Value {
        if actual == expected {
            return val;
        }
        match (actual.is_int(), expected.is_int()) {
            (true, true) => {
                if actual.bits() > expected.bits() {
                    builder.ins().ireduce(expected, val)
                } else {
                    builder.ins().sextend(expected, val)
                }
            }
            (false, false) => {
                if actual.bits() > expected.bits() {
                    builder.ins().fdemote(expected, val)
                } else {
                    builder.ins().fpromote(expected, val)
                }
            }
            // `fcvt_from_sint` takes i32 or i64, so narrow types get
            // widened first.
            (true, false) => {
                let widened = if actual.bits() < 32 {
                    builder.ins().sextend(types::I32, val)
                } else {
                    val
                };
                builder.ins().fcvt_from_sint(expected, widened)
            }
            (false, true) => {
                let converted = builder.ins().fcvt_to_sint_sat(types::I64, val);
                if expected.bits() < 64 {
                    builder.ins().ireduce(expected, converted)
                } else {
                    converted
                }
            }
        }
    }

    /// Bring both operands of a binary op to the type the op works in.
    ///
    /// The HIR's result type picks the instruction, but the operands
    /// can arrive in a different one: an integer literal in a float
    /// expression (`pct + 2.0 + 2`) is materialised as an `iconst`, and
    /// Cranelift's verifier rejects an `fadd` with an integer argument.
    /// Mixed-width integers are widened to the larger of the two, which
    /// is what the arithmetic below assumes.
    fn reconcile_binary_operands(
        builder: &mut FunctionBuilder,
        result_is_float: bool,
        lhs: Value,
        rhs: Value,
    ) -> (Value, Value) {
        let lhs_ty = builder.func.dfg.value_type(lhs);
        let rhs_ty = builder.func.dfg.value_type(rhs);
        if lhs_ty == rhs_ty {
            return (lhs, rhs);
        }
        if lhs_ty.is_vector() || rhs_ty.is_vector() {
            return (lhs, rhs);
        }
        // A comparison's result type is a bool, so it can't say whether
        // the operands are floats -- take that from the operands.
        let want_float = result_is_float || lhs_ty.is_float() || rhs_ty.is_float();
        let target = if want_float {
            if lhs_ty == types::F64 || rhs_ty == types::F64 {
                types::F64
            } else {
                types::F32
            }
        } else if lhs_ty.bits() >= rhs_ty.bits() {
            lhs_ty
        } else {
            rhs_ty
        };
        (
            Self::convert_numeric(builder, lhs, lhs_ty, target),
            Self::convert_numeric(builder, rhs, rhs_ty, target),
        )
    }

    /// Coerce a Cranelift value from one type to another (e.g., i64 -> i32, f64 -> f32)
    fn coerce_value(
        builder: &mut FunctionBuilder,
        val: Value,
        actual: types::Type,
        expected: types::Type,
    ) -> Value {
        if actual == expected {
            return val;
        }
        if actual.is_int() && expected.is_int() {
            if actual.bits() > expected.bits() {
                builder.ins().ireduce(expected, val)
            } else {
                builder.ins().uextend(expected, val)
            }
        } else if actual.is_float() && expected.is_float() {
            if actual.bits() > expected.bits() {
                builder.ins().fdemote(expected, val)
            } else {
                builder.ins().fpromote(expected, val)
            }
        } else if actual.is_int() && expected.is_float() {
            if actual.bits() == expected.bits() {
                // Same size: bitcast (e.g., i32 -> f32, i64 -> f64)
                builder
                    .ins()
                    .bitcast(expected, cranelift_codegen::ir::MemFlagsData::new(), val)
            } else {
                // Different sizes: signed int to float conversion (e.g., i64 -> f32)
                // First narrow/widen the int to match the float's bit width if needed,
                // then use fcvt_from_sint
                let int_for_cvt = if actual.bits() > 32 && expected == types::F32 {
                    // i64 -> f32: ireduce to i32 first (lossy but matches semantics)
                    builder.ins().ireduce(types::I32, val)
                } else if actual.bits() < 64 && expected == types::F64 {
                    // i32 -> f64: extend to i64 first
                    builder.ins().sextend(types::I64, val)
                } else {
                    val
                };
                builder.ins().fcvt_from_sint(expected, int_for_cvt)
            }
        } else if actual.is_float() && expected.is_int() {
            if actual.bits() == expected.bits() {
                // Same size: bitcast (e.g., f32 -> i32, f64 -> i64)
                builder
                    .ins()
                    .bitcast(expected, cranelift_codegen::ir::MemFlagsData::new(), val)
            } else {
                // Different sizes: float to signed int conversion (e.g., f32 -> i64)
                let int_from_cvt = builder.ins().fcvt_to_sint_sat(types::I32, val);
                if expected.bits() > 32 {
                    builder.ins().sextend(expected, int_from_cvt)
                } else if expected.bits() < 32 {
                    builder.ins().ireduce(expected, int_from_cvt)
                } else {
                    int_from_cvt
                }
            }
        } else {
            // Can't coerce (e.g., vector types), return as-is
            log::warn!("[Cranelift] Cannot coerce {:?} to {:?}", actual, expected);
            val
        }
    }

    /// Emit a zero/default value for the given Cranelift type
    fn emit_zero_value(builder: &mut FunctionBuilder, ty: types::Type) -> Value {
        if ty.is_float() {
            if ty == types::F32 {
                builder.ins().f32const(0.0)
            } else {
                builder.ins().f64const(0.0)
            }
        } else {
            builder.ins().iconst(ty, 0)
        }
    }

    /// Translate HIR type to Cranelift type
    pub fn translate_type(&self, ty: &HirType) -> CompilerResult<types::Type> {
        match ty {
            HirType::Void => Ok(types::I8), // Void represented as i8
            HirType::Bool => Ok(types::I8), // Cranelift uses i8 for bools
            HirType::I8 => Ok(types::I8),
            HirType::I16 => Ok(types::I16),
            HirType::I32 => Ok(types::I32),
            HirType::I64 => Ok(types::I64),
            HirType::I128 => Ok(types::I128),
            HirType::U8 => Ok(types::I8),
            HirType::U16 => Ok(types::I16),
            HirType::U32 => Ok(types::I32),
            HirType::U64 => Ok(types::I64),
            HirType::U128 => Ok(types::I128),
            HirType::F32 => Ok(types::F32),
            HirType::F64 => Ok(types::F64),
            HirType::Ptr(_) => Ok(self.module.target_config().pointer_type()),
            HirType::Ref { .. } => {
                // References are treated as pointers in Cranelift
                Ok(self.module.target_config().pointer_type())
            }
            HirType::Array(_elem_ty, _) => {
                // Arrays decay to pointers in Cranelift
                Ok(self.module.target_config().pointer_type())
            }
            HirType::Struct(struct_ty) => {
                // A struct wrapping one scalar travels as that scalar.
                if let Some(field_ty) = struct_carried_as_its_field(struct_ty) {
                    return self.translate_type(field_ty);
                }
                // Multi-field or complex structs are passed by reference
                Ok(self.module.target_config().pointer_type())
            }
            HirType::Function(_) => {
                // Function pointers
                Ok(self.module.target_config().pointer_type())
            }
            HirType::Vector(elem_ty, count) => match (&**elem_ty, *count) {
                (HirType::F32, 4) => Ok(types::F32X4),
                (HirType::F64, 2) => Ok(types::F64X2),
                (HirType::I8, 16) | (HirType::U8, 16) => Ok(types::I8X16),
                (HirType::I16, 8) | (HirType::U16, 8) => Ok(types::I16X8),
                (HirType::I32, 4) | (HirType::U32, 4) => Ok(types::I32X4),
                (HirType::I64, 2) | (HirType::U64, 2) => Ok(types::I64X2),
                _ => Err(CompilerError::CodeGen(format!(
                    "unsupported SIMD vector lane shape in Cranelift backend: Vector({:?}, {})",
                    elem_ty, count
                ))),
            },
            HirType::Union(_) => {
                // Unions are treated as pointers to stack-allocated memory
                Ok(self.module.target_config().pointer_type())
            }
            HirType::Closure(_) => {
                // Closures are treated as pointers to their environment
                Ok(self.module.target_config().pointer_type())
            }
            HirType::Opaque(_) => {
                // Opaque types are treated as pointers
                Ok(self.module.target_config().pointer_type())
            }
            HirType::ConstGeneric(_) => {
                // Const generic parameters should be resolved before codegen
                Err(CompilerError::Backend(
                    "Unresolved const generic parameter".into(),
                ))
            }
            HirType::Generic { .. } => {
                // Generic types should be monomorphized before codegen
                Err(CompilerError::Backend(
                    "Generic types must be monomorphized before codegen".into(),
                ))
            }
            HirType::TraitObject { .. } => {
                // Trait objects are represented as fat pointers: { *data, *vtable }
                // For now, treat as pointer (will need struct with two pointers later)
                Ok(self.module.target_config().pointer_type())
            }
            HirType::Interface { .. } => {
                // Interface types are also trait objects
                Ok(self.module.target_config().pointer_type())
            }
            HirType::AssociatedType {
                trait_id,
                self_ty,
                name,
            } => {
                // Associated types must be resolved to concrete types before codegen
                Err(CompilerError::Backend(format!(
                    "Unresolved associated type: <{:?} as {:?}>::{}",
                    self_ty, trait_id, name
                )))
            }
            HirType::Promise(_) => {
                // Promise is a struct containing {state_machine_ptr, poll_fn_ptr}
                // Treated as a pointer to the Promise struct
                Ok(self.module.target_config().pointer_type())
            }
            HirType::Fiber(_) => {
                // Fiber is an opaque `*mut FiberRepr` handle. Same
                // pointer-sized storage as any other handle; SSA
                // intercepts the methods that actually need fiber-
                // specific semantics, so codegen treats it as a
                // bare pointer.
                Ok(self.module.target_config().pointer_type())
            }
            HirType::Continuation { .. } => {
                // Continuations are represented as pointers to continuation frames
                Ok(self.module.target_config().pointer_type())
            }
            HirType::EffectRow { .. } => {
                // Effect rows are compile-time only (used for effect polymorphism)
                // Should never appear in codegen
                Err(CompilerError::Backend(
                    "Effect rows should be resolved before codegen".into(),
                ))
            }
        }
    }

    /// Call libm fmod function for float remainder operation.
    /// Supports scalar f32 (fmodf) and f64 (fmod); vector frem is rejected.
    fn call_libm_fmod<M: Module>(
        module: &mut M,
        builder: &mut FunctionBuilder,
        lhs: Value,
        rhs: Value,
    ) -> CompilerResult<Value> {
        let lhs_ty = builder.func.dfg.value_type(lhs);
        let rhs_ty = builder.func.dfg.value_type(rhs);
        if lhs_ty != rhs_ty {
            return Err(CompilerError::CodeGen(format!(
                "frem operand type mismatch: lhs={:?}, rhs={:?}",
                lhs_ty, rhs_ty
            )));
        }

        let (symbol_name, param_ty) = match lhs_ty {
            types::F64 => ("fmod", types::F64),
            types::F32 => ("fmodf", types::F32),
            other => {
                return Err(CompilerError::CodeGen(format!(
                    "frem currently supports scalar f32/f64 only, got {:?}",
                    other
                )));
            }
        };

        // Declare fmod signature for the selected scalar float width.
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(param_ty));
        sig.params.push(AbiParam::new(param_ty));
        sig.returns.push(AbiParam::new(param_ty));

        // Declare fmod/fmodf as an external function.
        let fmod_id = module
            .declare_function(symbol_name, Linkage::Import, &sig)
            .map_err(|e| {
                CompilerError::Backend(format!("Failed to declare {}: {}", symbol_name, e))
            })?;

        // Import the function into the current function
        let fmod_func = module.declare_func_in_func(fmod_id, builder.func);

        // Call fmod
        let call = builder.ins().call(fmod_func, &[lhs, rhs]);
        let result = builder.inst_results(call)[0];

        Ok(result)
    }

    /// Call malloc for heap allocation
    fn call_malloc(&mut self, builder: &mut FunctionBuilder, size: Value) -> CompilerResult<Value> {
        let ptr_ty = self.module.target_config().pointer_type();

        // Declare malloc signature: void* malloc(size_t size)
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty)); // size_t is pointer-sized
        sig.returns.push(AbiParam::new(ptr_ty));

        // Declare malloc as an external function
        let malloc_id = self
            .module
            .declare_function("malloc", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare malloc: {}", e)))?;

        // Import the function into the current function
        let malloc_func = self
            .module
            .declare_func_in_func(malloc_id, &mut builder.func);

        // Call malloc
        let call = builder.ins().call(malloc_func, &[size]);
        let result = builder.inst_results(call)[0];

        Ok(result)
    }

    /// Call free for heap deallocation
    fn call_free(&mut self, builder: &mut FunctionBuilder, ptr: Value) -> CompilerResult<()> {
        let ptr_ty = self.module.target_config().pointer_type();

        // Declare free signature: void free(void* ptr)
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty));

        // Declare free as an external function
        let free_id = self
            .module
            .declare_function("free", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare free: {}", e)))?;

        // Import the function into the current function
        let free_func = self.module.declare_func_in_func(free_id, builder.func);

        // Call free
        builder.ins().call(free_func, &[ptr]);

        Ok(())
    }

    /// Call realloc for heap reallocation
    fn call_realloc(
        &mut self,
        builder: &mut FunctionBuilder,
        ptr: Value,
        new_size: Value,
    ) -> CompilerResult<Value> {
        let ptr_ty = self.module.target_config().pointer_type();

        // Declare realloc signature: void* realloc(void* ptr, size_t size)
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty));
        sig.params.push(AbiParam::new(ptr_ty));
        sig.returns.push(AbiParam::new(ptr_ty));

        // Declare realloc as an external function
        let realloc_id = self
            .module
            .declare_function("realloc", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare realloc: {}", e)))?;

        // Import the function into the current function
        let realloc_func = self.module.declare_func_in_func(realloc_id, builder.func);

        // Call realloc
        let call = builder.ins().call(realloc_func, &[ptr, new_size]);
        let result = builder.inst_results(call)[0];

        Ok(result)
    }

    /// Call libm sin function
    fn call_libm_sin(
        &mut self,
        builder: &mut FunctionBuilder,
        val: Value,
    ) -> CompilerResult<Value> {
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));

        let sin_id = self
            .module
            .declare_function("sin", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare sin: {}", e)))?;
        let sin_func = self.module.declare_func_in_func(sin_id, builder.func);

        let call = builder.ins().call(sin_func, &[val]);
        Ok(builder.inst_results(call)[0])
    }

    /// Call libm cos function
    fn call_libm_cos(
        &mut self,
        builder: &mut FunctionBuilder,
        val: Value,
    ) -> CompilerResult<Value> {
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));

        let cos_id = self
            .module
            .declare_function("cos", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare cos: {}", e)))?;
        let cos_func = self.module.declare_func_in_func(cos_id, builder.func);

        let call = builder.ins().call(cos_func, &[val]);
        Ok(builder.inst_results(call)[0])
    }

    /// Call libm pow function
    fn call_libm_pow(
        &mut self,
        builder: &mut FunctionBuilder,
        base: Value,
        exp: Value,
    ) -> CompilerResult<Value> {
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));

        let pow_id = self
            .module
            .declare_function("pow", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare pow: {}", e)))?;
        let pow_func = self.module.declare_func_in_func(pow_id, builder.func);

        let call = builder.ins().call(pow_func, &[base, exp]);
        Ok(builder.inst_results(call)[0])
    }

    /// Call libm log function
    fn call_libm_log(
        &mut self,
        builder: &mut FunctionBuilder,
        val: Value,
    ) -> CompilerResult<Value> {
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));

        let log_id = self
            .module
            .declare_function("log", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare log: {}", e)))?;
        let log_func = self.module.declare_func_in_func(log_id, builder.func);

        let call = builder.ins().call(log_func, &[val]);
        Ok(builder.inst_results(call)[0])
    }

    /// Call libm exp function
    fn call_libm_exp(
        &mut self,
        builder: &mut FunctionBuilder,
        val: Value,
    ) -> CompilerResult<Value> {
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));

        let exp_id = self
            .module
            .declare_function("exp", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare exp: {}", e)))?;
        let exp_func = self.module.declare_func_in_func(exp_id, builder.func);

        let call = builder.ins().call(exp_func, &[val]);
        Ok(builder.inst_results(call)[0])
    }

    /// Call libc memcpy function
    fn call_memcpy(
        &mut self,
        builder: &mut FunctionBuilder,
        dest: Value,
        src: Value,
        len: Value,
    ) -> CompilerResult<Value> {
        let ptr_ty = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty)); // dest
        sig.params.push(AbiParam::new(ptr_ty)); // src
        sig.params.push(AbiParam::new(ptr_ty)); // len
        sig.returns.push(AbiParam::new(ptr_ty)); // returns dest

        let memcpy_id = self
            .module
            .declare_function("memcpy", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare memcpy: {}", e)))?;
        let memcpy_func = self.module.declare_func_in_func(memcpy_id, builder.func);

        let call = builder.ins().call(memcpy_func, &[dest, src, len]);
        Ok(builder.inst_results(call)[0])
    }

    /// Call libc memset function
    fn call_memset(
        &mut self,
        builder: &mut FunctionBuilder,
        dest: Value,
        val: Value,
        len: Value,
    ) -> CompilerResult<Value> {
        let ptr_ty = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty)); // dest
        sig.params.push(AbiParam::new(types::I32)); // val (int)
        sig.params.push(AbiParam::new(ptr_ty)); // len
        sig.returns.push(AbiParam::new(ptr_ty)); // returns dest

        let memset_id = self
            .module
            .declare_function("memset", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare memset: {}", e)))?;
        let memset_func = self.module.declare_func_in_func(memset_id, builder.func);

        let call = builder.ins().call(memset_func, &[dest, val, len]);
        Ok(builder.inst_results(call)[0])
    }

    /// Call libc memmove function
    fn call_memmove(
        &mut self,
        builder: &mut FunctionBuilder,
        dest: Value,
        src: Value,
        len: Value,
    ) -> CompilerResult<Value> {
        let ptr_ty = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty)); // dest
        sig.params.push(AbiParam::new(ptr_ty)); // src
        sig.params.push(AbiParam::new(ptr_ty)); // len
        sig.returns.push(AbiParam::new(ptr_ty)); // returns dest

        let memmove_id = self
            .module
            .declare_function("memmove", Linkage::Import, &sig)
            .map_err(|e| CompilerError::Backend(format!("Failed to declare memmove: {}", e)))?;
        let memmove_func = self.module.declare_func_in_func(memmove_id, builder.func);

        let call = builder.ins().call(memmove_func, &[dest, src, len]);
        Ok(builder.inst_results(call)[0])
    }

    /// Translate a HIR instruction to Cranelift
    #[allow(dead_code)]
    fn translate_hir_instruction(
        &mut self,
        builder: &mut FunctionBuilder,
        inst: &HirInstruction,
    ) -> CompilerResult<()> {
        match inst {
            HirInstruction::Binary {
                op,
                result,
                left,
                right,
                ..
            } => {
                let lhs = self.value_map[left];
                let rhs = self.value_map[right];

                let value = match op {
                    // Integer arithmetic
                    BinaryOp::Add => builder.ins().iadd(lhs, rhs),
                    BinaryOp::Sub => builder.ins().isub(lhs, rhs),
                    BinaryOp::Mul => builder.ins().imul(lhs, rhs),
                    BinaryOp::Div => builder.ins().sdiv(lhs, rhs),
                    BinaryOp::Rem => builder.ins().srem(lhs, rhs),

                    // Float arithmetic
                    BinaryOp::FAdd => builder.ins().fadd(lhs, rhs),
                    BinaryOp::FSub => builder.ins().fsub(lhs, rhs),
                    BinaryOp::FMul => builder.ins().fmul(lhs, rhs),
                    BinaryOp::FDiv => builder.ins().fdiv(lhs, rhs),
                    // Float remainder requires libm fmod function call
                    BinaryOp::FRem => Self::call_libm_fmod(&mut self.module, builder, lhs, rhs)?,

                    // Bitwise
                    BinaryOp::And => builder.ins().band(lhs, rhs),
                    BinaryOp::Or => builder.ins().bor(lhs, rhs),
                    BinaryOp::Xor => builder.ins().bxor(lhs, rhs),
                    BinaryOp::Shl => builder.ins().ishl(lhs, rhs),
                    BinaryOp::Shr => builder.ins().sshr(lhs, rhs),

                    // Integer comparisons
                    BinaryOp::Eq => builder.ins().icmp(IntCC::Equal, lhs, rhs),
                    BinaryOp::Ne => builder.ins().icmp(IntCC::NotEqual, lhs, rhs),
                    BinaryOp::Lt => builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs),
                    BinaryOp::Le => builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs),
                    BinaryOp::Gt => builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs),
                    BinaryOp::Ge => builder
                        .ins()
                        .icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs),

                    // Float comparisons
                    BinaryOp::FEq => builder.ins().fcmp(FloatCC::Equal, lhs, rhs),
                    BinaryOp::FNe => builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs),
                    BinaryOp::FLt => builder.ins().fcmp(FloatCC::LessThan, lhs, rhs),
                    BinaryOp::FLe => builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs),
                    BinaryOp::FGt => builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs),
                    BinaryOp::FGe => builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs),
                };

                self.value_map.insert(*result, value);
                Ok(())
            }

            _ => {
                // TODO: Implement other instruction types
                Ok(())
            }
        }
    }

    /// Translate a HIR terminator to Cranelift
    #[allow(dead_code)]
    fn translate_hir_terminator(
        &mut self,
        builder: &mut FunctionBuilder,
        terminator: &HirTerminator,
    ) -> CompilerResult<()> {
        match terminator {
            HirTerminator::Return { values } => {
                let expected_returns = builder.func.signature.returns.clone();
                let mut ret_vals = Vec::new();
                for (i, v) in values.iter().enumerate() {
                    let val = self.value_map[v];
                    let coerced = if let Some(expected_abi) = expected_returns.get(i) {
                        let actual_ty = builder.func.dfg.value_type(val);
                        Self::coerce_value(builder, val, actual_ty, expected_abi.value_type)
                    } else {
                        val
                    };
                    ret_vals.push(coerced);
                }
                // Bare return: pad with zero values if signature expects more
                while ret_vals.len() < expected_returns.len() {
                    let ty = expected_returns[ret_vals.len()].value_type;
                    let zero = Self::emit_zero_value(builder, ty);
                    ret_vals.push(zero);
                }
                builder.ins().return_(&ret_vals);
                Ok(())
            }

            HirTerminator::Unreachable => {
                builder.ins().return_(&[]);
                Ok(())
            }

            _ => {
                // NOTE: Some terminator types not yet implemented.
                // Main terminators (Return, Branch, CondBranch, Switch) already handled elsewhere.
                // Missing: Resume (for exception handling), IndirectBr (for computed gotos)
                //
                // WORKAROUND: Emits return instruction (safe fallback)
                // FUTURE (v2.0): Add missing terminator types as needed
                // Estimated effort: 2-3 hours per terminator type
                builder.ins().return_(&[]);
                Ok(())
            }
        }
    }

    /// Translate phi node
    #[allow(dead_code)]
    fn translate_phi(&mut self, builder: &mut FunctionBuilder, phi: &HirPhi) -> CompilerResult<()> {
        // Phi nodes in SSA form are handled by block parameters in Cranelift
        let block = builder.current_block().unwrap();
        let ty = self.translate_type(&phi.ty)?;
        let param = builder.append_block_param(block, ty);
        self.value_map.insert(phi.result, param);

        // The incoming values will be added when we process the predecessor blocks
        Ok(())
    }

    /// Translate instruction
    #[allow(dead_code)]
    fn translate_instruction(
        &mut self,
        builder: &mut FunctionBuilder,
        inst: &HirInstruction,
    ) -> CompilerResult<()> {
        match inst {
            HirInstruction::Binary {
                op,
                result,
                ty,
                left,
                right,
            } => {
                let lhs = self.value_map[left];
                let rhs = self.value_map[right];

                let value = match op {
                    BinaryOp::Add => builder.ins().iadd(lhs, rhs),
                    BinaryOp::Sub => builder.ins().isub(lhs, rhs),
                    BinaryOp::Mul => builder.ins().imul(lhs, rhs),
                    BinaryOp::Div => {
                        if ty.is_float() {
                            builder.ins().fdiv(lhs, rhs)
                        } else if ty.is_signed() {
                            builder.ins().sdiv(lhs, rhs)
                        } else {
                            builder.ins().udiv(lhs, rhs)
                        }
                    }
                    BinaryOp::Rem => {
                        if ty.is_signed() {
                            builder.ins().srem(lhs, rhs)
                        } else {
                            builder.ins().urem(lhs, rhs)
                        }
                    }
                    BinaryOp::And => builder.ins().band(lhs, rhs),
                    BinaryOp::Or => builder.ins().bor(lhs, rhs),
                    BinaryOp::Xor => builder.ins().bxor(lhs, rhs),
                    BinaryOp::Shl => builder.ins().ishl(lhs, rhs),
                    BinaryOp::Shr => {
                        if ty.is_signed() {
                            builder.ins().sshr(lhs, rhs)
                        } else {
                            builder.ins().ushr(lhs, rhs)
                        }
                    }
                    // Integer comparisons - always return i8 (bool), never uextend
                    BinaryOp::Eq => builder.ins().icmp(IntCC::Equal, lhs, rhs),
                    BinaryOp::Ne => builder.ins().icmp(IntCC::NotEqual, lhs, rhs),
                    // For comparisons, default to signed since ZynML uses signed integers
                    BinaryOp::Lt => builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs),
                    BinaryOp::Le => builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs),
                    BinaryOp::Gt => builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs),
                    BinaryOp::Ge => builder
                        .ins()
                        .icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs),
                    // Floating point operations
                    BinaryOp::FAdd => builder.ins().fadd(lhs, rhs),
                    BinaryOp::FSub => builder.ins().fsub(lhs, rhs),
                    BinaryOp::FMul => builder.ins().fmul(lhs, rhs),
                    BinaryOp::FDiv => builder.ins().fdiv(lhs, rhs),
                    BinaryOp::FRem => {
                        // Cranelift doesn't have frem, use libm fmod
                        Self::call_libm_fmod(&mut self.module, builder, lhs, rhs)?
                    }
                    // Floating point comparisons - always return i8 (bool), never uextend
                    BinaryOp::FEq => builder.ins().fcmp(FloatCC::Equal, lhs, rhs),
                    BinaryOp::FNe => builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs),
                    BinaryOp::FLt => builder.ins().fcmp(FloatCC::LessThan, lhs, rhs),
                    BinaryOp::FLe => builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs),
                    BinaryOp::FGt => builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs),
                    BinaryOp::FGe => builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs),
                };

                self.value_map.insert(*result, value);
            }

            HirInstruction::Unary {
                op,
                result,
                ty,
                operand,
            } => {
                let val = self.value_map[operand];

                let value = match op {
                    UnaryOp::Neg => {
                        if ty.is_float() {
                            builder.ins().fneg(val)
                        } else {
                            builder.ins().ineg(val)
                        }
                    }
                    UnaryOp::Not => builder.ins().bnot(val),
                    UnaryOp::FNeg => builder.ins().fneg(val),
                };

                self.value_map.insert(*result, value);
            }

            HirInstruction::Load {
                result,
                ty,
                ptr,
                align: _,
                volatile,
            } => {
                let ptr_val = self.value_map[ptr];
                let cranelift_ty = self.translate_type(ty)?;
                let flags = if *volatile {
                    // Cranelift 0.106 has no explicit volatile flag; omitting with_aligned()
                    // prevents alignment-based coalescing for volatile-marked accesses.
                    MemFlags::new().with_notrap()
                } else {
                    MemFlags::new().with_aligned().with_notrap()
                };
                let value = builder.ins().load(cranelift_ty, flags, ptr_val, 0);
                self.value_map.insert(*result, value);
            }

            HirInstruction::Store {
                value,
                ptr,
                align: _,
                volatile,
            } => {
                let val = self.value_map[value];
                let ptr_val = self.value_map[ptr];
                let flags = if *volatile {
                    // Cranelift 0.106 has no explicit volatile flag; omitting with_aligned()
                    // prevents alignment-based coalescing for volatile-marked accesses.
                    MemFlags::new().with_notrap()
                } else {
                    MemFlags::new().with_aligned().with_notrap()
                };
                builder.ins().store(flags, val, ptr_val, 0);
            }

            HirInstruction::Alloca {
                result,
                ty,
                count,
                align: _,
            } => {
                // Cranelift doesn't have alloca - use stack slots
                let size = self.type_size(ty)?;
                // For alloca with count, we need to allocate count * size
                let alloc_size = if let Some(_count_val) = count {
                    // Multiply size by count - simplified for now
                    size
                } else {
                    size
                };
                let slot_data =
                    StackSlotData::new(StackSlotKind::ExplicitSlot, alloc_size as u32, 0);
                let slot = builder.create_sized_stack_slot(slot_data);
                let addr =
                    builder
                        .ins()
                        .stack_addr(self.module.target_config().pointer_type(), slot, 0);
                self.value_map.insert(*result, addr);
            }

            HirInstruction::Call {
                result,
                callee,
                args,
                type_args,
                is_tail: _,
                ..
            } => {
                let arg_vals: Vec<_> = args.iter().map(|arg| self.value_map[arg]).collect();

                match callee {
                    HirCallable::Function(func_id) => {
                        let cranelift_func = self.function_map[func_id];

                        // `return_call` is a Cranelift terminator. Our
                        // block-level lowering still emits the HIR
                        // block's Return terminator after the Call, so
                        // a `return_call` here would leave the block
                        // double-terminated and trip Cranelift's
                        // verifier. The HIR tco marker still flips
                        // `is_tail = true` for downstream backends
                        // (LLVM gets the hint), but Cranelift falls
                        // back to the standard call until the
                        // terminator-skip plumbing lands.
                        let call = if self.reloadable_calls
                            && !self.external_link_names.contains_key(func_id)
                        {
                            // Reloadable dispatch: the callee's current
                            // entry lives in its cell, and replacing it
                            // is one store every caller observes.
                            let sig = self
                                .module
                                .declarations()
                                .get_function_decl(cranelift_func)
                                .signature
                                .clone();
                            let sig_ref = builder.import_signature(sig);
                            let ptr_ty = self.module.target_config().pointer_type();
                            let cell =
                                crate::reload::call_cell_addr(self.reload_key, *func_id) as i64;
                            let cell_v = builder.ins().iconst(ptr_ty, cell);
                            let entry = builder.ins().load(ptr_ty, MemFlags::trusted(), cell_v, 0);
                            builder.ins().call_indirect(sig_ref, entry, &arg_vals)
                        } else {
                            let func_ref = self
                                .module
                                .declare_func_in_func(cranelift_func, builder.func);
                            builder.ins().call(func_ref, &arg_vals)
                        };

                        if let Some(result_id) = result {
                            let results = builder.inst_results(call);
                            if !results.is_empty() {
                                self.value_map.insert(*result_id, results[0]);
                            }
                        }
                    }
                    HirCallable::FuncRef(func_id) => {
                        // Get function address (same as main path)
                        let ptr_ty = self.module.target_config().pointer_type();
                        let addr = if self.reloadable_calls
                            && self.function_map.contains_key(func_id)
                            && !self.external_link_names.contains_key(func_id)
                            && self.current_compile_id != Some(*func_id)
                        {
                            let cell =
                                crate::reload::call_cell_addr(self.reload_key, *func_id) as i64;
                            let cell_v = builder.ins().iconst(ptr_ty, cell);
                            builder.ins().load(ptr_ty, MemFlags::trusted(), cell_v, 0)
                        } else {
                            let cranelift_func = self.function_map[func_id];
                            let func_ref = self
                                .module
                                .declare_func_in_func(cranelift_func, builder.func);
                            builder.ins().func_addr(ptr_ty, func_ref)
                        };
                        if let Some(result_id) = result {
                            self.value_map.insert(*result_id, addr);
                        }
                    }
                    HirCallable::Indirect(func_ptr) => {
                        // Same missing-value handling as the other Indirect
                        // arms above — surface a proper Backend error instead
                        // of `self.value_map[func_ptr]` panicking with an
                        // unhelpful index-out-of-bounds.
                        let ptr_val = match self.value_map.get(func_ptr).copied() {
                            Some(v) => v,
                            None => {
                                return Err(crate::CompilerError::Backend(format!(
                                    "HirCallable::Indirect: function-pointer value {:?} \
                                     is referenced but never defined in this function's \
                                     value_map.",
                                    func_ptr
                                )));
                            }
                        };
                        // For indirect calls, we need a signature reference
                        // This is a simplified version - in reality we'd need to track function signatures
                        let sig_ref = builder.import_signature(self.module.make_signature());

                        // Same terminator-already-emitted concern as the
                        // direct-call path above. Fall back to a
                        // standard `call_indirect` until the lowering
                        // can skip the trailing Return terminator.
                        let call = builder.ins().call_indirect(sig_ref, ptr_val, &arg_vals);

                        if let Some(result_id) = result {
                            let results = builder.inst_results(call);
                            if !results.is_empty() {
                                self.value_map.insert(*result_id, results[0]);
                            }
                        }
                    }
                    HirCallable::Intrinsic(intrinsic) => {
                        // Handle intrinsics
                        match intrinsic {
                            crate::hir::Intrinsic::Memcpy => {
                                // memcpy(dst, src, size)
                                if args.len() == 3 {
                                    let dst = arg_vals[0];
                                    let src = arg_vals[1];
                                    let size = arg_vals[2];
                                    builder.call_memcpy(
                                        self.module.target_config(),
                                        dst,
                                        src,
                                        size,
                                    );
                                    // memcpy returns void, but HIR might expect a value
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, dst); // Return destination
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "memcpy requires 3 arguments".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Memset => {
                                // memset(dst, value, size)
                                if args.len() == 3 {
                                    let dst = arg_vals[0];
                                    let val = arg_vals[1];
                                    let size = arg_vals[2];
                                    builder.call_memset(
                                        self.module.target_config(),
                                        dst,
                                        val,
                                        size,
                                    );
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, dst); // Return destination
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "memset requires 3 arguments".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Memmove => {
                                // memmove(dst, src, size)
                                if args.len() == 3 {
                                    let dst = arg_vals[0];
                                    let src = arg_vals[1];
                                    let size = arg_vals[2];
                                    builder.call_memmove(
                                        self.module.target_config(),
                                        dst,
                                        src,
                                        size,
                                    );
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, dst); // Return destination
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "memmove requires 3 arguments".into(),
                                    ));
                                }
                            }
                            // Math intrinsics
                            crate::hir::Intrinsic::Sqrt => {
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let sqrt_val = builder.ins().sqrt(val);
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, sqrt_val);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "sqrt requires 1 argument".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Rsqrt => {
                                // Reciprocal square root, lowered as
                                // `1.0 / sqrt(x)`. Cranelift has no direct
                                // FRSQRTE op; hardware estimate + Newton
                                // refinement is a future optimisation.
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let sq = builder.ins().sqrt(val);
                                    let arg_ty = builder.func.dfg.value_type(val);
                                    let one = if arg_ty == types::F32 {
                                        builder.ins().f32const(1.0_f32)
                                    } else {
                                        builder.ins().f64const(1.0_f64)
                                    };
                                    let rsqrt_val = builder.ins().fdiv(one, sq);
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, rsqrt_val);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "rsqrt requires 1 argument".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Fabs => {
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let abs_val = builder.ins().fabs(val);
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, abs_val);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "fabs requires 1 argument".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Fma => {
                                if args.len() == 3 {
                                    let a = arg_vals[0];
                                    let b = arg_vals[1];
                                    let c = arg_vals[2];
                                    let fma_val = builder.ins().fma(a, b, c);
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, fma_val);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "fma requires 3 arguments".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Sin => {
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let sin_val = self.call_libm_sin(builder, val)?;
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, sin_val);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "sin requires 1 argument".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Cos => {
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let cos_val = self.call_libm_cos(builder, val)?;
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, cos_val);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "cos requires 1 argument".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Log => {
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let log_val = self.call_libm_log(builder, val)?;
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, log_val);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "log requires 1 argument".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Exp => {
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let exp_val = self.call_libm_exp(builder, val)?;
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, exp_val);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "exp requires 1 argument".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Pow => {
                                if args.len() == 2 {
                                    let base = arg_vals[0];
                                    let exp = arg_vals[1];
                                    let pow_val = self.call_libm_pow(builder, base, exp)?;
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, pow_val);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "pow requires 2 arguments".into(),
                                    ));
                                }
                            }
                            // Bit manipulation intrinsics
                            crate::hir::Intrinsic::Ctpop => {
                                // Count population (number of 1 bits)
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let popcnt = builder.ins().popcnt(val);
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, popcnt);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "ctpop requires 1 argument".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Ctlz => {
                                // Count leading zeros
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let clz = builder.ins().clz(val);
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, clz);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "ctlz requires 1 argument".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Cttz => {
                                // Count trailing zeros
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let ctz = builder.ins().ctz(val);
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, ctz);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "cttz requires 1 argument".into(),
                                    ));
                                }
                            }
                            crate::hir::Intrinsic::Bswap => {
                                // Byte swap (endianness conversion)
                                if args.len() == 1 {
                                    let val = arg_vals[0];
                                    let swapped = builder.ins().bswap(val);
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, swapped);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "bswap requires 1 argument".into(),
                                    ));
                                }
                            }
                            // Type query intrinsics
                            crate::hir::Intrinsic::SizeOf => {
                                // Get the size of the type from type_args
                                if type_args.is_empty() {
                                    return Err(CompilerError::Backend(
                                        "sizeof requires a type argument".into(),
                                    ));
                                }
                                let ty = &type_args[0];
                                let size = self.type_size(ty)? as u64;
                                let size_val = builder.ins().iconst(types::I64, size as i64);
                                if let Some(result_id) = result {
                                    self.value_map.insert(*result_id, size_val);
                                }
                            }
                            crate::hir::Intrinsic::AlignOf => {
                                // Get the alignment of the type from type_args
                                if type_args.is_empty() {
                                    return Err(CompilerError::Backend(
                                        "alignof requires a type argument".into(),
                                    ));
                                }
                                let ty = &type_args[0];
                                let align = self.type_alignment(ty)? as u64;
                                let align_val = builder.ins().iconst(types::I64, align as i64);
                                if let Some(result_id) = result {
                                    self.value_map.insert(*result_id, align_val);
                                }
                            }
                            // Overflow checking intrinsics
                            crate::hir::Intrinsic::AddWithOverflow
                            | crate::hir::Intrinsic::SubWithOverflow
                            | crate::hir::Intrinsic::MulWithOverflow => {
                                if args.len() == 2 {
                                    let lhs = arg_vals[0];
                                    let rhs = arg_vals[1];

                                    // Perform the operation
                                    let result_val = match intrinsic {
                                        crate::hir::Intrinsic::AddWithOverflow => {
                                            // For overflow detection, we'd need to use specific overflow instructions
                                            // Cranelift has iadd_overflow_trap but not a direct overflow flag
                                            // For now, just perform the operation
                                            builder.ins().iadd(lhs, rhs)
                                        }
                                        crate::hir::Intrinsic::SubWithOverflow => {
                                            // Similar to add, just perform the operation for now
                                            builder.ins().isub(lhs, rhs)
                                        }
                                        crate::hir::Intrinsic::MulWithOverflow => {
                                            // Similar to add/sub, just perform the operation for now
                                            builder.ins().imul(lhs, rhs)
                                        }
                                        _ => unreachable!(),
                                    };

                                    // NOTE: Overflow intrinsics should return (result, bool overflow_flag).
                                    // Cranelift has overflow checking via: iadd_cout, isub_bout, etc.
                                    // Need: (1) Use overflow-checking instructions, (2) Create tuple/struct return,
                                    // (3) Map both values to result.
                                    //
                                    // WORKAROUND: Returns only result value (overflow unchecked)
                                    // FUTURE (v2.0): Use Cranelift overflow instructions + tuple returns
                                    // Estimated effort: 4-6 hours
                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, result_val);
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "overflow intrinsics require 2 arguments".into(),
                                    ));
                                }
                            }

                            // Memory management intrinsics
                            crate::hir::Intrinsic::Malloc => {
                                if !args.is_empty() {
                                    if let Some(&size) =
                                        args.get(0).and_then(|id| self.value_map.get(id))
                                    {
                                        let ptr = self.call_malloc(builder, size)?;
                                        if let Some(result_id) = result {
                                            self.value_map.insert(*result_id, ptr);
                                        }
                                    } else {
                                        return Err(CompilerError::Backend(
                                            "malloc: size argument not found in value map".into(),
                                        ));
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "malloc requires size argument".into(),
                                    ));
                                }
                            }

                            crate::hir::Intrinsic::Free => {
                                if !args.is_empty() {
                                    if let Some(&ptr) =
                                        args.get(0).and_then(|id| self.value_map.get(id))
                                    {
                                        self.call_free(builder, ptr)?;
                                    } else {
                                        return Err(CompilerError::Backend(
                                            "free: pointer argument not found in value map".into(),
                                        ));
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "free requires pointer argument".into(),
                                    ));
                                }
                            }

                            crate::hir::Intrinsic::Realloc => {
                                if args.len() >= 2 {
                                    if let (Some(&ptr), Some(&new_size)) = (
                                        args.get(0).and_then(|id| self.value_map.get(id)),
                                        args.get(1).and_then(|id| self.value_map.get(id)),
                                    ) {
                                        let new_ptr = self.call_realloc(builder, ptr, new_size)?;
                                        if let Some(result_id) = result {
                                            self.value_map.insert(*result_id, new_ptr);
                                        }
                                    } else {
                                        return Err(CompilerError::Backend(
                                            "realloc: arguments not found in value map".into(),
                                        ));
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "realloc requires pointer and size arguments".into(),
                                    ));
                                }
                            }

                            crate::hir::Intrinsic::Drop => {
                                // Call destructor for type
                                if !args.is_empty() {
                                    if let Some(&_ptr) =
                                        args.get(0).and_then(|id| self.value_map.get(id))
                                    {
                                        // NOTE: Destructor dispatch requires type information + vtable.
                                        // Need: (1) Type ID from pointer, (2) Destructor lookup table,
                                        // (3) Indirect call to destructor function.
                                        //
                                        // WORKAROUND: No-op (works for POD types without destructors)
                                        // FUTURE (v2.0): Implement destructor dispatch system
                                        // Estimated effort: 10-15 hours (depends on trait system)
                                    } else {
                                        return Err(CompilerError::Backend(
                                            "drop: pointer argument not found in value map".into(),
                                        ));
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "drop requires pointer argument".into(),
                                    ));
                                }
                            }

                            crate::hir::Intrinsic::IncRef => {
                                // Increment reference count
                                if !args.is_empty() {
                                    if let Some(&ptr) =
                                        args.get(0).and_then(|id| self.value_map.get(id))
                                    {
                                        // Assume refcount is first field of struct
                                        let ref_count =
                                            builder.ins().load(types::I32, MemFlags::new(), ptr, 0);
                                        let one = builder.ins().iconst(types::I32, 1);
                                        let new_count = builder.ins().iadd(ref_count, one);
                                        builder.ins().store(MemFlags::new(), new_count, ptr, 0);
                                    } else {
                                        return Err(CompilerError::Backend(
                                            "incref: pointer argument not found in value map"
                                                .into(),
                                        ));
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "incref requires pointer argument".into(),
                                    ));
                                }
                            }

                            crate::hir::Intrinsic::DecRef => {
                                // Decrement reference count
                                if !args.is_empty() {
                                    if let Some(&ptr) =
                                        args.get(0).and_then(|id| self.value_map.get(id))
                                    {
                                        // Assume refcount is first field of struct
                                        let ref_count =
                                            builder.ins().load(types::I32, MemFlags::new(), ptr, 0);
                                        let one = builder.ins().iconst(types::I32, 1);
                                        let new_count = builder.ins().isub(ref_count, one);
                                        builder.ins().store(MemFlags::new(), new_count, ptr, 0);

                                        // Check if count is zero and free if so
                                        let zero = builder.ins().iconst(types::I32, 0);
                                        let is_zero =
                                            builder.ins().icmp(IntCC::Equal, new_count, zero);

                                        // Create blocks for conditional free
                                        let free_block = builder.create_block();
                                        let continue_block = builder.create_block();

                                        // Branch based on refcount
                                        builder.ins().brif(
                                            is_zero,
                                            free_block,
                                            &[],
                                            continue_block,
                                            &[],
                                        );

                                        // Free block: call free and jump to continue
                                        builder.seal_block(free_block);
                                        builder.switch_to_block(free_block);
                                        self.call_free(builder, ptr)?;
                                        builder.ins().jump(continue_block, &[]);

                                        // Continue block
                                        builder.seal_block(continue_block);
                                        builder.switch_to_block(continue_block);
                                    } else {
                                        return Err(CompilerError::Backend(
                                            "decref: pointer argument not found in value map"
                                                .into(),
                                        ));
                                    }
                                } else {
                                    return Err(CompilerError::Backend(
                                        "decref requires pointer argument".into(),
                                    ));
                                }
                            }

                            crate::hir::Intrinsic::Alloca => {
                                // Stack allocation - should be handled as instruction, not intrinsic call
                                return Err(CompilerError::Backend(
                                    "Alloca should be an instruction, not intrinsic call".into(),
                                ));
                            }

                            crate::hir::Intrinsic::GCSafepoint => {
                                // GC safepoint - no-op for now in Cranelift
                                // In a real implementation, this would:
                                // 1. Mark a safepoint for the GC
                                // 2. Potentially spill registers
                                // 3. Update GC metadata
                                // For now, we just ignore it
                            }

                            crate::hir::Intrinsic::Await => {
                                // Await intrinsic - complex implementation
                                // This would suspend the current function and register a continuation
                                // For now, just call a runtime function
                                return Err(CompilerError::Backend(
                                    "Await intrinsic not yet implemented in Cranelift backend"
                                        .into(),
                                ));
                            }

                            crate::hir::Intrinsic::Yield => {
                                // Yield intrinsic for generators
                                // This would suspend execution and yield a value
                                return Err(CompilerError::Backend(
                                    "Yield intrinsic not yet implemented in Cranelift backend"
                                        .into(),
                                ));
                            }

                            crate::hir::Intrinsic::Panic => {
                                // Gap 8 Phase 3: Panic with message
                                // For now, we call abort() which terminates immediately
                                // Future: Add message printing, stack unwinding

                                // Call abort() from libc
                                let abort_sig = {
                                    let mut sig = self.module.make_signature();
                                    // abort() takes no arguments and doesn't return
                                    sig
                                };

                                let abort_name = "abort";
                                let abort_func = self
                                    .module
                                    .declare_function(
                                        abort_name,
                                        cranelift_module::Linkage::Import,
                                        &abort_sig,
                                    )
                                    .map_err(|e| {
                                        CompilerError::Backend(format!(
                                            "Failed to declare abort: {}",
                                            e
                                        ))
                                    })?;
                                let abort_func_ref = self
                                    .module
                                    .declare_func_in_func(abort_func, &mut builder.func);

                                // Call abort() - doesn't return
                                builder.ins().call(abort_func_ref, &[]);

                                // Add unreachable to satisfy control flow
                                builder
                                    .ins()
                                    .trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
                            }

                            crate::hir::Intrinsic::Abort => {
                                // Gap 8 Phase 3: Immediate abort
                                // Calls abort() from libc

                                let abort_sig = {
                                    let mut sig = self.module.make_signature();
                                    sig
                                };

                                let abort_name = "abort";
                                let abort_func = self
                                    .module
                                    .declare_function(
                                        abort_name,
                                        cranelift_module::Linkage::Import,
                                        &abort_sig,
                                    )
                                    .map_err(|e| {
                                        CompilerError::Backend(format!(
                                            "Failed to declare abort: {}",
                                            e
                                        ))
                                    })?;
                                let abort_func_ref = self
                                    .module
                                    .declare_func_in_func(abort_func, &mut builder.func);

                                builder.ins().call(abort_func_ref, &[]);
                                builder
                                    .ins()
                                    .trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
                            }

                            // ZRTL Value Conversion Intrinsics
                            crate::hir::Intrinsic::ClosureToZrtl => {
                                // Convert closure to ZrtlClosure
                                // Args: fn_ptr (pointer), env_ptr (pointer), env_size (i64)
                                // Returns: pointer to ZrtlClosure
                                if arg_vals.len() >= 3 {
                                    let fn_ptr = arg_vals[0];
                                    let env_ptr = arg_vals[1];
                                    let env_size = arg_vals[2];

                                    // Declare zyntax_closure_to_zrtl
                                    let mut sig = self.module.make_signature();
                                    sig.params
                                        .push(cranelift_codegen::ir::AbiParam::new(types::I64)); // fn_ptr
                                    sig.params
                                        .push(cranelift_codegen::ir::AbiParam::new(types::I64)); // env_ptr
                                    sig.params
                                        .push(cranelift_codegen::ir::AbiParam::new(types::I64)); // env_size
                                    sig.returns
                                        .push(cranelift_codegen::ir::AbiParam::new(types::I64)); // result ptr

                                    let func = self
                                        .module
                                        .declare_function(
                                            "zyntax_closure_to_zrtl",
                                            cranelift_module::Linkage::Import,
                                            &sig,
                                        )
                                        .map_err(|e| {
                                            CompilerError::Backend(format!(
                                                "Failed to declare zyntax_closure_to_zrtl: {}",
                                                e
                                            ))
                                        })?;
                                    let func_ref =
                                        self.module.declare_func_in_func(func, builder.func);

                                    let call_inst =
                                        builder.ins().call(func_ref, &[fn_ptr, env_ptr, env_size]);
                                    let result_val = builder.inst_results(call_inst)[0];

                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, result_val);
                                    }
                                }
                            }

                            crate::hir::Intrinsic::BoxToZrtl => {
                                // Convert value to DynamicBox
                                // Args: value_ptr (pointer), type_tag (i32), size (i32)
                                // Returns: pointer to DynamicBox
                                if arg_vals.len() >= 3 {
                                    let value_ptr = arg_vals[0];
                                    let type_tag = arg_vals[1];
                                    let size = arg_vals[2];

                                    // Declare zyntax_primitive_to_box
                                    let mut sig = self.module.make_signature();
                                    sig.params
                                        .push(cranelift_codegen::ir::AbiParam::new(types::I64)); // value_ptr
                                    sig.params
                                        .push(cranelift_codegen::ir::AbiParam::new(types::I32)); // type_tag
                                    sig.params
                                        .push(cranelift_codegen::ir::AbiParam::new(types::I32)); // size
                                    sig.returns
                                        .push(cranelift_codegen::ir::AbiParam::new(types::I64)); // result ptr

                                    let func = self
                                        .module
                                        .declare_function(
                                            "zyntax_primitive_to_box",
                                            cranelift_module::Linkage::Import,
                                            &sig,
                                        )
                                        .map_err(|e| {
                                            CompilerError::Backend(format!(
                                                "Failed to declare zyntax_primitive_to_box: {}",
                                                e
                                            ))
                                        })?;
                                    let func_ref =
                                        self.module.declare_func_in_func(func, builder.func);

                                    let call_inst =
                                        builder.ins().call(func_ref, &[value_ptr, type_tag, size]);
                                    let result_val = builder.inst_results(call_inst)[0];

                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, result_val);
                                    }
                                }
                            }

                            crate::hir::Intrinsic::PrimitiveToBox => {
                                // Convert primitive value to DynamicBox based on type
                                // Args: value (any), type_tag (i32)
                                // Returns: pointer to DynamicBox
                                if arg_vals.len() >= 2 {
                                    let value = arg_vals[0];
                                    let _type_tag = arg_vals[1];
                                    let value_ty = builder.func.dfg.value_type(value);

                                    // Determine which boxing function to call based on type
                                    let func_name = if value_ty == types::I32 {
                                        "zyntax_box_i32"
                                    } else if value_ty == types::I64 {
                                        "zyntax_box_i64"
                                    } else if value_ty == types::F32 {
                                        "zyntax_box_f32"
                                    } else if value_ty == types::F64 {
                                        "zyntax_box_f64"
                                    } else {
                                        "zyntax_box_i64" // fallback
                                    };

                                    let mut sig = self.module.make_signature();
                                    sig.params
                                        .push(cranelift_codegen::ir::AbiParam::new(value_ty));
                                    sig.returns
                                        .push(cranelift_codegen::ir::AbiParam::new(types::I64));

                                    let func = self
                                        .module
                                        .declare_function(
                                            func_name,
                                            cranelift_module::Linkage::Import,
                                            &sig,
                                        )
                                        .map_err(|e| {
                                            CompilerError::Backend(format!(
                                                "Failed to declare {}: {}",
                                                func_name, e
                                            ))
                                        })?;
                                    let func_ref =
                                        self.module.declare_func_in_func(func, builder.func);

                                    let call_inst = builder.ins().call(func_ref, &[value]);
                                    let result_val = builder.inst_results(call_inst)[0];

                                    if let Some(result_id) = result {
                                        self.value_map.insert(*result_id, result_val);
                                    }
                                }
                            }

                            crate::hir::Intrinsic::TypeTagOf => {
                                // Get TypeTag constant for a type
                                // This is resolved at compile time based on the type argument
                                // For now, return 0 (void) as placeholder
                                let result_val = builder.ins().iconst(types::I32, 0);
                                if let Some(result_id) = result {
                                    self.value_map.insert(*result_id, result_val);
                                }
                            }
                        }
                    }
                    HirCallable::Symbol(symbol_name) => {
                        // External symbol call - import and call by name
                        // Build signature from argument types
                        let mut sig = self.module.make_signature();
                        for arg_val in &arg_vals {
                            let arg_ty = builder.func.dfg.value_type(*arg_val);
                            sig.params
                                .push(cranelift_codegen::ir::AbiParam::new(arg_ty));
                        }
                        // For now, assume i64 return unless we have no result
                        if result.is_some() {
                            sig.returns
                                .push(cranelift_codegen::ir::AbiParam::new(types::I64));
                        }

                        // Use @___ prefix internally for runtime symbols
                        let internal_name = if symbol_name.starts_with('$') {
                            format!("@{}", symbol_name)
                        } else {
                            symbol_name.clone()
                        };

                        let func = self
                            .module
                            .declare_function(
                                &internal_name,
                                cranelift_module::Linkage::Import,
                                &sig,
                            )
                            .map_err(|e| {
                                CompilerError::Backend(format!(
                                    "Failed to declare symbol {}: {}",
                                    symbol_name, e
                                ))
                            })?;
                        let func_ref = self.module.declare_func_in_func(func, builder.func);

                        // Symbol calls share the same terminator-skip
                        // concern as the direct-call arms above; fall
                        // back to a standard `call` until that lands.
                        let call = builder.ins().call(func_ref, &arg_vals);

                        if let Some(result_id) = result {
                            let results = builder.inst_results(call);
                            if !results.is_empty() {
                                self.value_map.insert(*result_id, results[0]);
                            }
                        }
                    }
                }
            }

            HirInstruction::GetElementPtr {
                result,
                ty: _,
                ptr,
                indices,
            } => {
                // Simplified GEP - just add offsets
                let mut addr = self.value_map[ptr];
                let ptr_ty = self.module.target_config().pointer_type();

                for (i, idx) in indices.iter().enumerate() {
                    let idx_val = self.value_map[idx];
                    // Calculate offset based on type
                    let offset = if i == 0 {
                        // First index is array/struct offset
                        idx_val
                    } else {
                        // Subsequent indices need size calculation
                        let elem_size = 8; // Simplified - would need proper type size
                        let size_val = builder.ins().iconst(ptr_ty, elem_size);
                        builder.ins().imul(idx_val, size_val)
                    };
                    addr = builder.ins().iadd(addr, offset);
                }

                self.value_map.insert(*result, addr);
            }

            HirInstruction::Cast {
                result,
                ty,
                op,
                operand,
            } => {
                log::trace!(
                    "[Cranelift Cast] operand={:?}, op={:?}, target_ty={:?}",
                    operand,
                    op,
                    ty
                );
                let val = self
                    .value_map
                    .get(operand)
                    .copied()
                    .unwrap_or_else(|| panic!("Cast operand {:?} not in value_map", operand));
                let target_ty = self.translate_type(ty)?;
                log::trace!("[Cranelift Cast] val={:?}, target_ty={:?}", val, target_ty);
                let operand_clif_ty = builder.func.dfg.value_type(val);
                if cast_needs_scalar_lanes(op, operand_clif_ty, target_ty) {
                    return Err(CompilerError::UnsupportedByBackend {
                        backend: "cranelift",
                        construct: format!("a vector {:?} cast", op),
                        detail: format!(
                            "no lane-width conversion from {:?} to {:?}",
                            operand_clif_ty, target_ty
                        ),
                    });
                }

                let cast_val = match op {
                    crate::hir::CastOp::Bitcast => {
                        // Bitcast - just return the value for now
                        val
                    }
                    crate::hir::CastOp::ZExt => builder.ins().uextend(target_ty, val),
                    crate::hir::CastOp::SExt => builder.ins().sextend(target_ty, val),
                    crate::hir::CastOp::Trunc => builder.ins().ireduce(target_ty, val),
                    crate::hir::CastOp::FpToUi => builder.ins().fcvt_to_uint(target_ty, val),
                    crate::hir::CastOp::FpToSi => builder.ins().fcvt_to_sint(target_ty, val),
                    crate::hir::CastOp::UiToFp => builder.ins().fcvt_from_uint(target_ty, val),
                    crate::hir::CastOp::SiToFp => builder.ins().fcvt_from_sint(target_ty, val),
                    crate::hir::CastOp::FpExt => builder.ins().fpromote(target_ty, val),
                    crate::hir::CastOp::FpTrunc => builder.ins().fdemote(target_ty, val),
                    _ => {
                        // Other cast operations like PtrToInt, IntToPtr
                        val // Placeholder
                    }
                };

                self.value_map.insert(*result, cast_val);
            }

            HirInstruction::ExtractValue {
                result,
                ty,
                aggregate: _,
                indices: _,
            } => {
                // NOTE: Dead code - not used. See inline implementation in compile_function
                let cranelift_ty = self.translate_type(ty)?;
                let dummy = builder.ins().iconst(cranelift_ty, 0);
                self.value_map.insert(*result, dummy);
            }

            HirInstruction::InsertValue {
                result,
                ty: _,
                aggregate,
                value: _,
                indices: _,
            } => {
                // NOTE: Dead code - not used. See inline implementation in compile_function
                let struct_ptr = self.value_map[aggregate];
                self.value_map.insert(*result, struct_ptr);
            }

            HirInstruction::Atomic {
                op,
                result,
                ty,
                ptr,
                value,
                ordering,
            } => {
                // Atomic operations
                let ptr_val = self.value_map[ptr];
                let target_ty = self.translate_type(ty)?;

                // Convert HIR atomic ordering to Cranelift memory ordering
                let mem_flags = match ordering {
                    crate::hir::AtomicOrdering::Relaxed => MemFlags::new(),
                    crate::hir::AtomicOrdering::Acquire => MemFlags::new(),
                    crate::hir::AtomicOrdering::Release => MemFlags::new(),
                    crate::hir::AtomicOrdering::AcqRel => MemFlags::new(),
                    crate::hir::AtomicOrdering::SeqCst => MemFlags::new(),
                };

                let atomic_result = match op {
                    crate::hir::AtomicOp::Load => {
                        // Atomic load
                        builder
                            .ins()
                            .load(target_ty, mem_flags.with_notrap(), ptr_val, 0)
                    }
                    crate::hir::AtomicOp::Store => {
                        // Atomic store
                        if let Some(val_id) = value {
                            let val = self.value_map[val_id];
                            builder
                                .ins()
                                .store(mem_flags.with_notrap(), val, ptr_val, 0);
                            // Store doesn't return a value, but HIR expects one for SSA
                            val // Return the stored value
                        } else {
                            return Err(CompilerError::Backend(
                                "AtomicStore requires a value".into(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::Exchange => {
                        // Atomic exchange (swap)
                        if let Some(val_id) = value {
                            let val = self.value_map[val_id];
                            // For now, use regular load/store (Cranelift has limited atomic support)
                            let old_val =
                                builder
                                    .ins()
                                    .load(target_ty, mem_flags.with_notrap(), ptr_val, 0);
                            builder
                                .ins()
                                .store(mem_flags.with_notrap(), val, ptr_val, 0);
                            old_val
                        } else {
                            return Err(CompilerError::Backend(
                                "AtomicExchange requires a value".into(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::Add
                    | crate::hir::AtomicOp::Sub
                    | crate::hir::AtomicOp::And
                    | crate::hir::AtomicOp::Or
                    | crate::hir::AtomicOp::Xor => {
                        // Atomic read-modify-write operations
                        if let Some(val_id) = value {
                            let val = self.value_map[val_id];
                            // For now, use load-op-store pattern (not truly atomic)
                            let old_val =
                                builder
                                    .ins()
                                    .load(target_ty, mem_flags.with_notrap(), ptr_val, 0);
                            let new_val = match op {
                                crate::hir::AtomicOp::Add => builder.ins().iadd(old_val, val),
                                crate::hir::AtomicOp::Sub => builder.ins().isub(old_val, val),
                                crate::hir::AtomicOp::And => builder.ins().band(old_val, val),
                                crate::hir::AtomicOp::Or => builder.ins().bor(old_val, val),
                                crate::hir::AtomicOp::Xor => builder.ins().bxor(old_val, val),
                                _ => unreachable!(),
                            };
                            builder
                                .ins()
                                .store(mem_flags.with_notrap(), new_val, ptr_val, 0);
                            old_val // Return the old value
                        } else {
                            return Err(CompilerError::Backend(
                                "Atomic RMW operations require a value".into(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::CompareExchange => {
                        // NOTE: CompareExchange requires two values (expected, desired).
                        // Current HIR AtomicRMW has single `value` field - architecture limitation.
                        // Needs: (1) Extend HIR instruction, (2) Use Cranelift atomic_cas instruction.
                        //
                        // WORKAROUND: Returns error (unimplemented)
                        // FUTURE (v2.0): Extend AtomicRMW HIR instruction for compare-exchange
                        // Estimated effort: 4-5 hours (HIR change + Cranelift mapping)
                        return Err(CompilerError::Backend(
                            "CompareExchange not yet fully implemented".into(),
                        ));
                    }
                };

                self.value_map.insert(*result, atomic_result);
            }

            HirInstruction::Fence { ordering: _ } => {
                // Memory fence
                // Cranelift doesn't have explicit fence instructions in the same way
                // This is typically handled at the LLVM level or through target-specific intrinsics
                // For now, we'll emit a no-op that preserves ordering semantics
                // In a real implementation, this might emit platform-specific barrier instructions
            }

            HirInstruction::CreateUnion {
                result,
                union_ty,
                variant_index,
                value,
            } => {
                // Create a tagged union value
                let union_layout = self.calculate_union_layout(union_ty)?;
                let ptr_ty = self.module.target_config().pointer_type();

                // Allocate space for the union on the stack
                let union_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    union_layout.total_size,
                    0,
                ));
                let union_ptr = builder.ins().stack_addr(ptr_ty, union_slot, 0);

                // Store the discriminant in the first field
                let discriminant = builder.ins().iconst(types::I32, *variant_index as i64);
                builder
                    .ins()
                    .store(MemFlags::new(), discriminant, union_ptr, 0);

                // Store the value in the data field (after discriminant)
                let value_val = self.value_map[value];
                let data_offset = 4; // Assuming 4-byte discriminant
                builder
                    .ins()
                    .store(MemFlags::new(), value_val, union_ptr, data_offset);

                self.value_map.insert(*result, union_ptr);
            }

            HirInstruction::GetUnionDiscriminant { result, union_val } => {
                // Extract discriminant from union
                let union_ptr = self.value_map[union_val];
                let discriminant = builder.ins().load(
                    types::I32,
                    MemFlags::new(),
                    union_ptr,
                    0, // Discriminant is at offset 0
                );
                self.value_map.insert(*result, discriminant);
            }

            HirInstruction::ExtractUnionValue {
                result,
                ty,
                union_val,
                variant_index: _,
            } => {
                // Extract value from union variant (unsafe - assumes correct variant)
                let union_ptr = self.value_map[union_val];
                let cranelift_ty = self.translate_type(ty)?;
                let data_offset = 4; // Skip discriminant

                let value =
                    builder
                        .ins()
                        .load(cranelift_ty, MemFlags::new(), union_ptr, data_offset);
                self.value_map.insert(*result, value);
            }

            HirInstruction::CreateClosure {
                result,
                closure_ty,
                function,
                captures,
            } => {
                // Create a closure with captured environment
                let ptr_ty = self.module.target_config().pointer_type();

                // Calculate closure layout: function pointer + captured values
                let closure_layout = self.calculate_closure_layout(closure_ty)?;

                // Allocate closure on stack (simplified - should use proper allocator for escaping closures)
                let closure_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    closure_layout.total_size,
                    0,
                ));
                let closure_ptr = builder.ins().stack_addr(ptr_ty, closure_slot, 0);

                // Store function pointer - get the actual function address
                if let Some(&cranelift_func_id) = self.function_map.get(function) {
                    // Declare the function reference in the current function
                    let local_func_ref = self
                        .module
                        .declare_func_in_func(cranelift_func_id, builder.func);
                    // Get the function address as a pointer value
                    let func_ptr = builder.ins().func_addr(ptr_ty, local_func_ref);
                    builder
                        .ins()
                        .store(MemFlags::new(), func_ptr, closure_ptr, 0);
                } else {
                    // Lambda function not found - store null pointer as fallback
                    warn!(" Lambda function {:?} not found in function_map", function);
                    let null_ptr = builder.ins().iconst(ptr_ty, 0);
                    builder
                        .ins()
                        .store(MemFlags::new(), null_ptr, closure_ptr, 0);
                }

                // Store captured values
                let mut offset = 8; // After function pointer
                for capture_id in captures {
                    if let Some(&capture_val) = self.value_map.get(capture_id) {
                        builder
                            .ins()
                            .store(MemFlags::new(), capture_val, closure_ptr, offset);
                        offset += 8; // Simplified - should calculate proper size
                    }
                }

                self.value_map.insert(*result, closure_ptr);
            }

            HirInstruction::CallClosure {
                result,
                closure,
                args,
            } => {
                // Call a closure (simplified implementation)
                let closure_ptr = self.value_map[closure];
                let ptr_ty = self.module.target_config().pointer_type();

                // Load function pointer from closure
                let func_ptr = builder.ins().load(ptr_ty, MemFlags::new(), closure_ptr, 0);

                // Collect arguments (including captured environment)
                let mut call_args = vec![closure_ptr]; // Pass closure as first arg
                for arg_id in args {
                    if let Some(&arg_val) = self.value_map.get(arg_id) {
                        call_args.push(arg_val);
                    }
                }

                // In a real implementation, we'd call through the function pointer
                // For now, this is a placeholder that demonstrates the structure
                if let Some(result_id) = result {
                    // Create a dummy result for now (use i64 for pointer-sized values)
                    let dummy_result = builder.ins().iconst(types::I64, 0);
                    self.value_map.insert(*result_id, dummy_result);
                }
            }

            // ----------------------------------------------------------------
            // SIMD / Vector instructions
            // ----------------------------------------------------------------
            HirInstruction::VectorSplat { result, ty, scalar } => {
                let scalar_val = self.value_map[scalar];
                let vec_clif_ty = self.translate_type(ty)?;
                let value = builder.ins().splat(vec_clif_ty, scalar_val);
                self.value_map.insert(*result, value);
            }

            HirInstruction::VectorExtractLane {
                result,
                ty: _,
                vector,
                lane,
            } => {
                let vec_val = self.value_map[vector];
                let value = builder.ins().extractlane(vec_val, *lane);
                self.value_map.insert(*result, value);
            }

            HirInstruction::VectorInsertLane {
                result,
                ty: _,
                vector,
                scalar,
                lane,
            } => {
                let vec_val = self.value_map[vector];
                let scalar_val = self.value_map[scalar];
                let value = builder.ins().insertlane(vec_val, scalar_val, *lane);
                self.value_map.insert(*result, value);
            }

            HirInstruction::VectorHorizontalReduce {
                result,
                ty,
                vector,
                op,
            } => {
                // Cranelift has no native horizontal-reduce CLIF op.
                // Extract all lanes and fold with the requested scalar operation.
                let vec_val = self.value_map[vector];
                let clif_ty = self.translate_type(ty)?;
                let lane_count =
                    match clif_ty {
                        cranelift_codegen::ir::types::F32X4
                        | cranelift_codegen::ir::types::I32X4 => 4u8,
                        cranelift_codegen::ir::types::F64X2
                        | cranelift_codegen::ir::types::I64X2 => 2u8,
                        _ => {
                            return Err(CompilerError::Backend(format!(
                                "VectorHorizontalReduce: unsupported vector type {:?}",
                                clif_ty
                            )))
                        }
                    };

                // Extract lane 0 as the accumulator seed
                let mut acc = builder.ins().extractlane(vec_val, 0u8);
                for lane_idx in 1..lane_count {
                    let lane_val = builder.ins().extractlane(vec_val, lane_idx);
                    acc = match op {
                        BinaryOp::Add => builder.ins().iadd(acc, lane_val),
                        BinaryOp::FAdd => builder.ins().fadd(acc, lane_val),
                        BinaryOp::Sub => builder.ins().isub(acc, lane_val),
                        BinaryOp::FSub => builder.ins().fsub(acc, lane_val),
                        BinaryOp::Mul => builder.ins().imul(acc, lane_val),
                        BinaryOp::FMul => builder.ins().fmul(acc, lane_val),
                        _ => {
                            return Err(CompilerError::Backend(format!(
                                "VectorHorizontalReduce: unsupported op {:?}",
                                op
                            )))
                        }
                    };
                }
                self.value_map.insert(*result, acc);
            }

            _ => {
                // TODO: Implement remaining instructions
                return Err(CompilerError::Backend("Unimplemented instruction".into()));
            }
        }

        Ok(())
    }

    /// Translate terminator
    #[allow(dead_code)]

    /// Get function pointer for JIT execution
    pub fn get_function_ptr(&self, id: HirId) -> Option<*const u8> {
        self.hot_reload
            .function_pointers
            .read()
            .unwrap()
            .get(&id)
            .copied()
    }

    /// Finalize compiled functions and update function pointers
    /// This must be called after compile_function() before get_function_ptr() will work
    pub fn finalize_definitions(&mut self) -> CompilerResult<()> {
        use cranelift_module::Module;

        // Finalize the module
        self.module.finalize_definitions().map_err(|e| {
            CompilerError::Backend(format!("Failed to finalize definitions: {}", e))
        })?;

        // Update function pointers after finalization
        for (hir_id, compiled_func) in &self.compiled_functions {
            let code_ptr = self
                .module
                .get_finalized_function(compiled_func.function_id);
            if std::env::var("ZYNTAX_TRACE_CRANELIFT_SKIP").is_ok() {
                eprintln!("[ptrs] {hir_id:?} -> {code_ptr:?}");
            }
            self.hot_reload
                .function_pointers
                .write()
                .unwrap()
                .insert(*hir_id, code_ptr);
            if self.defer_cell_publish {
                self.deferred_cells.push((*hir_id, code_ptr as usize));
            } else {
                crate::reload::set_call_target(self.reload_key, *hir_id, code_ptr as usize);
            }
        }

        Ok(())
    }

    /// Hot-reload a function
    pub fn hot_reload_function(&mut self, id: HirId, function: &HirFunction) -> CompilerResult<()> {
        // Recompile the function
        self.compile_function(id, function)?;

        // The function pointer is automatically updated in compile_function

        Ok(())
    }

    /// Rollback to previous function version
    pub fn rollback_function(&mut self, id: HirId) -> CompilerResult<()> {
        let mut prev_versions = self.hot_reload.previous_versions.write().unwrap();

        if let Some(versions) = prev_versions.get_mut(&id) {
            if let Some(prev) = versions.pop() {
                // Restore previous function pointer
                let mut ptrs = self.hot_reload.function_pointers.write().unwrap();
                ptrs.insert(id, prev.code_ptr);

                // Update version counter
                let mut versions = self.hot_reload.versions.write().unwrap();
                if let Some(version) = versions.get_mut(&id) {
                    *version = prev.version;
                }

                Ok(())
            } else {
                Err(CompilerError::Backend(
                    "No previous version to rollback to".into(),
                ))
            }
        } else {
            Err(CompilerError::Backend(
                "Function has no previous versions".into(),
            ))
        }
    }

    // =========================================================================
    // Multi-Block Instruction Translation
    // =========================================================================

    /// Translate a single instruction in multi-block context
    fn translate_instruction_multiblock(
        &mut self,
        builder: &mut FunctionBuilder,
        inst: &HirInstruction,
        function: &HirFunction,
        type_cache: &HashMap<HirType, types::Type>,
    ) -> CompilerResult<()> {
        match inst {
            HirInstruction::Binary {
                op,
                result,
                ty,
                left,
                right,
            } => {
                let lhs = self.value_map[left];
                let rhs = self.value_map[right];

                let value = match op {
                    BinaryOp::Add => builder.ins().iadd(lhs, rhs),
                    BinaryOp::Sub => builder.ins().isub(lhs, rhs),
                    BinaryOp::Mul => builder.ins().imul(lhs, rhs),
                    BinaryOp::Div => {
                        if ty.is_float() {
                            builder.ins().fdiv(lhs, rhs)
                        } else if ty.is_signed() {
                            builder.ins().sdiv(lhs, rhs)
                        } else {
                            builder.ins().udiv(lhs, rhs)
                        }
                    }
                    BinaryOp::Rem => {
                        if ty.is_signed() {
                            builder.ins().srem(lhs, rhs)
                        } else {
                            builder.ins().urem(lhs, rhs)
                        }
                    }
                    BinaryOp::And => builder.ins().band(lhs, rhs),
                    BinaryOp::Or => builder.ins().bor(lhs, rhs),
                    BinaryOp::Xor => builder.ins().bxor(lhs, rhs),
                    BinaryOp::Shl => builder.ins().ishl(lhs, rhs),
                    BinaryOp::Shr => {
                        if ty.is_signed() {
                            builder.ins().sshr(lhs, rhs)
                        } else {
                            builder.ins().ushr(lhs, rhs)
                        }
                    }
                    // Comparison operations - use operand type to determine signed/unsigned
                    // Comparisons always return bool (i8) - never uextend
                    BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge => {
                        // Get operand type from left value, not result type
                        let operand_ty = function.values.get(left).map(|v| &v.ty).unwrap_or(ty);

                        // String equality / inequality routes through
                        // the `zrtl_string_equals` runtime helper —
                        // raw pointer comparison would only succeed
                        // when both operands point at the SAME interned
                        // string allocation, not whenever the byte
                        // contents match. Strings lower to
                        // `Ptr(I8)`; Lt/Le/Gt/Ge on strings would need
                        // a separate strcmp-style helper and isn't
                        // covered here yet.
                        let is_string_op = matches!(op, BinaryOp::Eq | BinaryOp::Ne)
                            && match operand_ty {
                                HirType::Ptr(inner) => matches!(**inner, HirType::I8),
                                _ => false,
                            };
                        if is_string_op {
                            let mut sig = self.module.make_signature();
                            let ptr_ty = self.module.target_config().pointer_type();
                            sig.params.push(AbiParam::new(ptr_ty));
                            sig.params.push(AbiParam::new(ptr_ty));
                            sig.returns.push(AbiParam::new(types::I32));
                            let func_id = self
                                .module
                                .declare_function("zrtl_string_equals", Linkage::Import, &sig)
                                .map_err(|e| {
                                    CompilerError::Backend(format!(
                                        "Failed to declare zrtl_string_equals: {}",
                                        e
                                    ))
                                })?;
                            let func_ref = self.module.declare_func_in_func(func_id, builder.func);
                            let call = builder.ins().call(func_ref, &[lhs, rhs]);
                            let eq_i32 = builder.inst_results(call)[0];
                            // Truncate i32 → i8 to match the rest of
                            // the comparison path (which returns the
                            // raw icmp result, an i8 bool).
                            let eq_i8 = builder.ins().ireduce(types::I8, eq_i32);
                            self.value_map.insert(
                                *result,
                                if matches!(op, BinaryOp::Eq) {
                                    eq_i8
                                } else {
                                    // For Ne: bxor with 1 to invert
                                    // the 0/1 boolean payload.
                                    let one = builder.ins().iconst(types::I8, 1);
                                    builder.ins().bxor(eq_i8, one)
                                },
                            );
                            return Ok(());
                        }

                        let cc = match op {
                            BinaryOp::Eq => IntCC::Equal,
                            BinaryOp::Ne => IntCC::NotEqual,
                            BinaryOp::Lt => {
                                if operand_ty.is_signed() {
                                    IntCC::SignedLessThan
                                } else {
                                    IntCC::UnsignedLessThan
                                }
                            }
                            BinaryOp::Le => {
                                if operand_ty.is_signed() {
                                    IntCC::SignedLessThanOrEqual
                                } else {
                                    IntCC::UnsignedLessThanOrEqual
                                }
                            }
                            BinaryOp::Gt => {
                                if operand_ty.is_signed() {
                                    IntCC::SignedGreaterThan
                                } else {
                                    IntCC::UnsignedGreaterThan
                                }
                            }
                            BinaryOp::Ge => {
                                if operand_ty.is_signed() {
                                    IntCC::SignedGreaterThanOrEqual
                                } else {
                                    IntCC::UnsignedGreaterThanOrEqual
                                }
                            }
                            _ => unreachable!(),
                        };
                        // icmp always returns i8 (bool) - no extension needed
                        builder.ins().icmp(cc, lhs, rhs)
                    }
                    // Float operations
                    BinaryOp::FAdd => builder.ins().fadd(lhs, rhs),
                    BinaryOp::FSub => builder.ins().fsub(lhs, rhs),
                    BinaryOp::FMul => builder.ins().fmul(lhs, rhs),
                    BinaryOp::FDiv => builder.ins().fdiv(lhs, rhs),
                    BinaryOp::FRem => Self::call_libm_fmod(&mut self.module, builder, lhs, rhs)?,
                    // Float comparisons - also always return bool (i8)
                    BinaryOp::FEq
                    | BinaryOp::FNe
                    | BinaryOp::FLt
                    | BinaryOp::FLe
                    | BinaryOp::FGt
                    | BinaryOp::FGe => {
                        let cc = match op {
                            BinaryOp::FEq => FloatCC::Equal,
                            BinaryOp::FNe => FloatCC::NotEqual,
                            BinaryOp::FLt => FloatCC::LessThan,
                            BinaryOp::FLe => FloatCC::LessThanOrEqual,
                            BinaryOp::FGt => FloatCC::GreaterThan,
                            BinaryOp::FGe => FloatCC::GreaterThanOrEqual,
                            _ => unreachable!(),
                        };
                        // fcmp always returns i8 (bool) - no extension needed
                        builder.ins().fcmp(cc, lhs, rhs)
                    }
                };

                self.value_map.insert(*result, value);
                Ok(())
            }

            HirInstruction::Unary {
                op,
                result,
                ty,
                operand,
            } => {
                let val = self.value_map[operand];

                let value = match op {
                    UnaryOp::Neg => {
                        if ty.is_float() {
                            builder.ins().fneg(val)
                        } else {
                            builder.ins().ineg(val)
                        }
                    }
                    UnaryOp::FNeg => builder.ins().fneg(val),
                    UnaryOp::Not => builder.ins().bnot(val),
                };

                self.value_map.insert(*result, value);
                Ok(())
            }

            _ => {
                // For now, ignore other instruction types
                // They can be added incrementally as needed
                Ok(())
            }
        }
    }

    /// Translate a terminator instruction
    fn translate_terminator(
        &mut self,
        terminator: &HirTerminator,
        current_block: HirId,
        function: &HirFunction,
        builder: &mut FunctionBuilder,
        seal_tracker: &mut HashMap<HirId, usize>,
    ) -> CompilerResult<()> {
        match terminator {
            HirTerminator::Return { values } => {
                let expected_returns = builder.func.signature.returns.clone();
                let mut cranelift_vals = Vec::new();
                for (i, v) in values.iter().enumerate() {
                    if let Some(&val) = self.value_map.get(v) {
                        let coerced = if let Some(expected_abi) = expected_returns.get(i) {
                            let actual_ty = builder.func.dfg.value_type(val);
                            Self::coerce_value(builder, val, actual_ty, expected_abi.value_type)
                        } else {
                            val
                        };
                        cranelift_vals.push(coerced);
                    }
                }
                // Bare return: pad with zero values if signature expects more
                while cranelift_vals.len() < expected_returns.len() {
                    let ty = expected_returns[cranelift_vals.len()].value_type;
                    let zero = Self::emit_zero_value(builder, ty);
                    cranelift_vals.push(zero);
                }
                builder.ins().return_(&cranelift_vals);
                Ok(())
            }

            HirTerminator::Branch { target } => {
                let target_block = self.block_map[target];
                let args = self.get_phi_args(*target, current_block, function);
                builder.ins().jump(target_block, &ba(&args));

                // Update seal tracker
                if let Some(count) = seal_tracker.get_mut(target) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        builder.seal_block(target_block);
                    }
                }

                Ok(())
            }

            HirTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                let cond = self.value_map[condition];
                let true_block = self.block_map[true_target];
                let false_block = self.block_map[false_target];

                let true_args = self.get_phi_args(*true_target, current_block, function);
                let false_args = self.get_phi_args(*false_target, current_block, function);

                // Use brif instruction
                builder.ins().brif(
                    cond,
                    true_block,
                    &ba(&true_args),
                    false_block,
                    &ba(&false_args),
                );

                // Update seal tracker for both targets
                for target in [true_target, false_target] {
                    let target_block = self.block_map[target];
                    if let Some(count) = seal_tracker.get_mut(target) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            builder.seal_block(target_block);
                        }
                    }
                }

                Ok(())
            }

            _ => {
                // Other terminators not yet implemented
                Err(CompilerError::Backend(format!(
                    "Terminator {:?} not yet implemented",
                    terminator
                )))
            }
        }
    }

    /// Debug helper: Dump Cranelift IR for a function
    #[allow(dead_code)]
    fn dump_cranelift_ir(&self, func_name: &str) {
        debug!("=== Cranelift IR for {} ===", func_name);
        debug!("{}", self.codegen_context.func.display());
        debug!("=== End IR ===\n");
    }

    // =========================================================================
    // Control Flow Helper Methods
    // =========================================================================

    /// Compute block ordering for processing (reverse post-order)
    /// This ensures dominators are processed before dominated blocks
    fn compute_block_order(&self, function: &HirFunction) -> Vec<HirId> {
        use std::collections::HashSet;

        let mut visited = HashSet::new();
        let mut post_order = Vec::new();

        fn visit(
            block_id: HirId,
            function: &HirFunction,
            visited: &mut HashSet<HirId>,
            post_order: &mut Vec<HirId>,
        ) {
            if visited.contains(&block_id) {
                return;
            }
            visited.insert(block_id);

            // Visit successors first (post-order)
            if let Some(block) = function.blocks.get(&block_id) {
                for successor in get_successors(&block.terminator) {
                    visit(successor, function, visited, post_order);
                }
            }

            post_order.push(block_id);
        }

        visit(
            function.entry_block,
            function,
            &mut visited,
            &mut post_order,
        );

        // Reverse to get reverse post-order
        post_order.reverse();
        post_order
    }

    /// Build predecessor map for all blocks
    fn build_predecessor_map(&self, function: &HirFunction) -> HashMap<HirId, Vec<HirId>> {
        let mut predecessors: HashMap<HirId, Vec<HirId>> = HashMap::new();

        for (block_id, block) in &function.blocks {
            for successor in get_successors(&block.terminator) {
                predecessors.entry(successor).or_default().push(*block_id);
            }
        }

        predecessors
    }

    /// Get phi arguments for jumping from one block to another
    fn get_phi_args(
        &self,
        target_block: HirId,
        from_block: HirId,
        function: &HirFunction,
    ) -> Vec<Value> {
        let target = match function.blocks.get(&target_block) {
            Some(b) => b,
            None => return vec![],
        };

        target
            .phis
            .iter()
            .map(|phi| {
                // Find the incoming value from from_block
                // FIXED: phi.incoming format is (value, block), not (block, value)
                phi.incoming
                    .iter()
                    .find(|(_, pred_block)| *pred_block == from_block)
                    .map(|(value, _)| self.value_map[value])
                    .expect(&format!(
                        "Phi node in block {:?} must have incoming value from predecessor {:?}",
                        target_block, from_block
                    ))
            })
            .collect()
    }

    /// Get the Cranelift IR as a string (for debugging)
    pub fn get_ir_string(&self) -> String {
        format!("{}", self.codegen_context.func)
    }

    // =========================================================================
    // Cross-Module Symbol Management
    // =========================================================================

    /// Register an exported symbol for cross-module linking
    ///
    /// Call this after compiling a module to make its public functions available
    /// as extern symbols for subsequent modules.
    ///
    /// # Arguments
    /// * `name` - The function name (without mangling/HirId suffix)
    /// * `ptr` - The function pointer
    pub fn register_exported_symbol(&mut self, name: &str, ptr: *const u8) {
        self.exported_symbols.insert(name.to_string(), ptr);
    }

    /// Get an exported symbol by name
    pub fn get_exported_symbol(&self, name: &str) -> Option<*const u8> {
        self.exported_symbols.get(name).copied()
    }

    /// Get all exported symbols
    ///
    /// Returns a list of (name, pointer) pairs for all exported functions.
    pub fn exported_symbols(&self) -> Vec<(&str, *const u8)> {
        self.exported_symbols
            .iter()
            .map(|(n, p)| (n.as_str(), *p))
            .collect()
    }

    /// Register a runtime symbol for external linking
    ///
    /// These symbols will be included when the JIT module needs to be recreated.
    pub fn register_runtime_symbol(&mut self, name: &str, ptr: *const u8) {
        // Check if symbol already exists
        if !self.runtime_symbols.iter().any(|(n, _)| n == name) {
            self.runtime_symbols.push((name.to_string(), ptr));
        }
    }

    /// Get all runtime symbols (for backend recreation)
    pub fn runtime_symbols(&self) -> &[(String, *const u8)] {
        &self.runtime_symbols
    }

    /// Check if a module has unresolved external symbols that require existing exports
    ///
    /// Returns the list of extern function names that need to be resolved.
    pub fn collect_extern_dependencies(module: &HirModule) -> Vec<String> {
        module
            .functions
            .values()
            .filter(|f| f.is_external)
            .map(|f| {
                f.name
                    .resolve_global()
                    .unwrap_or_else(|| f.name.to_string())
            })
            .collect()
    }

    /// Check if we need to rebuild the JIT module to include new symbols
    ///
    /// Returns true if the module has extern dependencies that aren't in the current JIT.
    pub fn needs_rebuild_for_module(&self, module: &HirModule) -> bool {
        let externs = Self::collect_extern_dependencies(module);
        externs.iter().any(|name| {
            // Check if the extern is in our exported symbols but not in runtime symbols
            self.exported_symbols.contains_key(name)
                && !self.runtime_symbols.iter().any(|(n, _)| n == name)
        })
    }

    /// Rebuild the JIT module with all accumulated symbols
    ///
    /// This creates a new JITModule with all runtime symbols and exported symbols.
    /// NOTE: This invalidates previously compiled function pointers!
    /// Use with caution - primarily for cross-module linking setup.
    pub fn rebuild_with_accumulated_symbols(&mut self) -> CompilerResult<()> {
        // Collect all symbols (runtime + exported)
        let mut all_symbols: Vec<(&str, *const u8)> = self
            .runtime_symbols
            .iter()
            .map(|(n, p)| (n.as_str(), *p))
            .collect();

        for (name, ptr) in &self.exported_symbols {
            if !all_symbols.iter().any(|(n, _)| *n == name) {
                all_symbols.push((name.as_str(), *ptr));
            }
        }

        // Configure Cranelift for the current platform
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        flag_builder.set("opt_level", "speed").unwrap();
        flag_builder.set("enable_verifier", "false").unwrap();

        let isa_builder = cranelift_native::builder().unwrap();
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        // Create new JIT module with all symbols
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        // OSR runtime symbols must be re-registered on every module rebuild —
        // tier-0 codegen emits probe call sites unconditionally for any loop.
        for (name, ptr) in crate::osr::osr_runtime_symbols() {
            builder.symbol(name, ptr);
        }
        // String-op runtime intrinsics — referenced unconditionally
        // by `BinaryOp::Eq` / `Ne` on `Ptr(I8)` operands.
        for (name, ptr) in crate::string_intrinsics::string_runtime_symbols() {
            builder.symbol(name, ptr);
        }
        for (name, ptr) in &all_symbols {
            builder.symbol(*name, *ptr);
        }

        // Update runtime_symbols to include everything
        for (name, ptr) in &self.exported_symbols {
            if !self.runtime_symbols.iter().any(|(n, _)| n == name) {
                self.runtime_symbols.push((name.clone(), *ptr));
            }
        }

        // Replace the JIT module
        self.module = JITModule::new(builder);

        // Clear state that depends on the old module
        self.function_map.clear();
        self.global_map.clear();
        self.compiled_functions.clear();
        // Note: hot_reload.function_pointers still has the old pointers
        // They remain valid but won't be part of the new module

        Ok(())
    }
}

/// Helper function to get successors from a terminator
fn get_successors(terminator: &HirTerminator) -> Vec<HirId> {
    match terminator {
        HirTerminator::Return { .. } => vec![],
        HirTerminator::Branch { target } => vec![*target],
        HirTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => {
            vec![*true_target, *false_target]
        }
        HirTerminator::Switch { default, cases, .. } => {
            let mut succs = vec![*default];
            succs.extend(cases.iter().map(|(_, target)| *target));
            succs
        }
        HirTerminator::PatternMatch {
            patterns, default, ..
        } => {
            let mut succs: Vec<HirId> = patterns.iter().map(|p| p.target).collect();
            if let Some(def) = default {
                succs.push(*def);
            }
            succs
        }
        HirTerminator::Unreachable => vec![],
        _ => vec![], // Handle other terminators as needed
    }
}

/// Emit a tier-0 OSR back-edge probe at the start of a loop header block.
///
/// Layout — control flow through the header becomes:
///
/// ```text
///   <existing header content above>
///   v_armed = load.i8 [arm_slot]                  ;; per-bead byte
///   brif v_armed, b_call_probe, b_post_probe      ;; zero ⇒ skip the probe
///
/// b_call_probe:
///   v_helper = call __zyntax_osr_probe(bead_id, site_key)
///   brif v_helper, b_dispatch, b_post_probe       ;; helper non-null ⇒ dispatch
///
/// b_dispatch:                                     (only when live_ins is Some)
///   ;; marshal live-ins → 4 i64 args (pad with zero), call_indirect helper,
///   ;; return its result
///
/// b_post_probe:
///   <continues with the original header instructions, terminator, etc>
/// ```
///
/// When `live_ins` is empty (layout rejected, or zero phi results), the
/// dispatch block is skipped — the probe call is still emitted (cheap)
/// but the result is unconditionally discarded. This keeps the probe
/// instruction shape stable across loops with different layouts and
/// makes the symbol-resolution path uniform.
///
/// On exit the builder is positioned in `b_post_probe` so subsequent
/// instruction/terminator emission lands there.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn emit_osr_back_edge_probe(
    builder: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    bead_id: u64,
    site_key: u64,
    frame: &crate::osr::OsrFrame,
    live_ins: &[cranelift_codegen::ir::Value],
    live_in_types: &[HirType],
    return_clir: Option<cranelift_codegen::ir::Type>,
) {
    let target_config = module.target_config();

    // Helper signature: one pointer to the frame carrying the live-ins.
    // Passing them as arguments would force each to fit a register, which
    // rules out the aggregates real loops carry.
    let dispatch_enabled = !live_ins.is_empty() || return_clir.is_some();
    let helper_sig_ref = if dispatch_enabled {
        let mut hs = module.make_signature();
        hs.params.push(AbiParam::new(types::I64));
        if let Some(rt) = return_clir {
            hs.returns.push(AbiParam::new(rt));
        }
        Some(builder.import_signature(hs))
    } else {
        None
    };

    // Arm check: load the bead's arm byte and branch on it. The byte flips
    // only when a tier ≥ 1 compile installs helpers for this bead. The load
    // must observe a store from a broker thread, so it cannot be hoisted
    // out of the loop, and it splits the header into extra blocks.
    // Load the helper pointer for this site. Null means no tier >= 1 code
    // exists yet, which is the steady state, so an unarmed loop pays one
    // load and a not-taken branch.
    //
    // Storing the helper rather than a flag is what keeps this cheap: a
    // flag would still need a runtime lookup, and a call that returns into
    // the loop forces the register allocator to treat caller-saved
    // registers as clobbered across the whole body. The dispatch below
    // returns instead of continuing, so nothing in the loop is constrained
    // by it.
    let slot = crate::osr::helper_slot_addr(bead_id, site_key);
    let slot_v = builder.ins().iconst(types::I64, slot as i64);
    let helper_ptr = builder.ins().load(types::I64, MemFlags::new(), slot_v, 0);

    let dispatch_block = builder.create_block();
    let post_probe_block = builder.create_block();
    builder
        .ins()
        .brif(helper_ptr, dispatch_block, &[], post_probe_block, &[]);

    builder.switch_to_block(dispatch_block);
    builder.seal_block(dispatch_block);
    match helper_sig_ref {
        Some(sig_ref) => {
            // Write each live-in into the frame at its offset, then hand
            // the frame's address over; the helper's result is this
            // function's result.
            // Trace hook: the dispatch path runs exactly once per transfer,
            // which is the only place one can be observed outside a test.
            if crate::osr::osr_trace_enabled() {
                let mut sig = module.make_signature();
                sig.params.push(AbiParam::new(types::I64));
                sig.params.push(AbiParam::new(types::I64));
                if let Ok(id) =
                    module.declare_function(crate::osr::OSR_TRANSFER_SYMBOL, Linkage::Import, &sig)
                {
                    let f = module.declare_func_in_func(id, builder.func);
                    let site_v = builder.ins().iconst(types::I64, site_key as i64);
                    builder.ins().call(f, &[site_v, helper_ptr]);
                }
            }
            let frame_addr =
                emit_osr_frame_store(builder, target_config, frame, live_ins, live_in_types);
            let call = builder
                .ins()
                .call_indirect(sig_ref, helper_ptr, &[frame_addr]);
            let results: smallvec::SmallVec<[cranelift_codegen::ir::Value; 1]> =
                builder.inst_results(call).iter().copied().collect();
            builder.ins().return_(&results);
        }
        // No representable layout for this header — nothing to transfer to.
        None => {
            builder.ins().jump(post_probe_block, &[]);
        }
    }

    builder.switch_to_block(post_probe_block);
    builder.seal_block(post_probe_block);
}

/// Spill the live-ins into a stack frame and return its address.
///
/// A value held by reference has its bytes copied out of whatever it points
/// at; a scalar is stored directly. Either way the frame ends up holding
/// the value itself, which is what lets a helper compiled by a different
/// backend read it without agreeing on how it is held in registers.
fn emit_osr_frame_store(
    builder: &mut FunctionBuilder<'_>,
    target_config: cranelift_codegen::isa::TargetFrontendConfig,
    frame: &crate::osr::OsrFrame,
    live_ins: &[cranelift_codegen::ir::Value],
    live_in_types: &[HirType],
) -> cranelift_codegen::ir::Value {
    use cranelift_codegen::ir::{StackSlotData, StackSlotKind};

    let slot = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        frame.size.max(1),
        frame.align.max(1).trailing_zeros() as u8,
    ));
    let base = builder.ins().stack_addr(types::I64, slot, 0);

    for (i, &v) in live_ins.iter().enumerate() {
        let (Some(&offset), Some(ty)) = (frame.offsets.get(i), live_in_types.get(i)) else {
            continue;
        };
        let dst = builder.ins().iadd_imm(base, offset as i64);
        if crate::osr::is_held_by_reference(ty) {
            let size = crate::osr::frame_size_of(ty) as u64;
            let align = crate::osr::frame_align_of(ty) as u8;
            builder.emit_small_memory_copy(
                target_config,
                dst,
                v,
                size,
                align,
                align,
                true,
                MemFlags::new(),
            );
        } else {
            builder.ins().store(MemFlags::new(), v, dst, 0);
        }
    }
    base
}

/// Reverse of [`marshal_to_i64`]: take an i64 helper arg and bit-cast /
/// reduce it to `target`. Used in the OSR helper's synthetic prologue to
/// recover the original-typed live-in values.
fn bitcast_from_i64(
    builder: &mut FunctionBuilder<'_>,
    v: cranelift_codegen::ir::Value,
    target: cranelift_codegen::ir::Type,
) -> cranelift_codegen::ir::Value {
    if target == types::I64 {
        v
    } else if target.is_int() {
        // Smaller int — truncate from i64.
        builder.ins().ireduce(target, v)
    } else if target == types::F32 {
        // i64 → i32 (low 32 bits) → bitcast f32.
        let low = builder.ins().ireduce(types::I32, v);
        builder
            .ins()
            .bitcast(types::F32, cranelift_codegen::ir::MemFlagsData::new(), low)
    } else if target == types::F64 {
        builder
            .ins()
            .bitcast(types::F64, cranelift_codegen::ir::MemFlagsData::new(), v)
    } else {
        // Pointers / other 64-bit values — reinterpret directly.
        builder
            .ins()
            .bitcast(target, cranelift_codegen::ir::MemFlagsData::new(), v)
    }
}

/// Bit-cast / zero-extend a Cranelift value to i64 for the OSR helper ABI.
fn marshal_to_i64(
    builder: &mut FunctionBuilder<'_>,
    v: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let ty = builder.func.dfg.value_type(v);
    if ty == types::I64 {
        v
    } else if ty.is_int() {
        // Smaller int → zero-extend.
        builder.ins().uextend(types::I64, v)
    } else if ty == types::F32 {
        // F32 → reinterpret as i32 → zero-extend.
        let as_i32 =
            builder
                .ins()
                .bitcast(types::I32, cranelift_codegen::ir::MemFlagsData::new(), v);
        builder.ins().uextend(types::I64, as_i32)
    } else if ty == types::F64 {
        builder
            .ins()
            .bitcast(types::I64, cranelift_codegen::ir::MemFlagsData::new(), v)
    } else {
        // Pointers / other 64-bit values: reinterpret directly.
        builder
            .ins()
            .bitcast(types::I64, cranelift_codegen::ir::MemFlagsData::new(), v)
    }
}

impl CraneliftBackend {
    /// Calculate struct layout with proper alignment
    /// Byte offset and size of each field of a struct type, in
    /// declaration order — what a caller needs to move a value from one
    /// generation's layout into another's.
    pub fn struct_field_extents(&self, ty: &HirType) -> Option<Vec<(usize, usize)>> {
        let struct_ty = match ty {
            HirType::Struct(s) => s,
            HirType::Ptr(inner) | HirType::Ref { pointee: inner, .. } => match inner.as_ref() {
                HirType::Struct(s) => s,
                _ => return None,
            },
            _ => return None,
        };
        let layout = self.calculate_struct_layout(struct_ty).ok()?;
        let mut out = Vec::with_capacity(struct_ty.fields.len());
        for (i, field_ty) in struct_ty.fields.iter().enumerate() {
            let offset = *layout.field_offsets.get(i)? as usize;
            out.push((offset, self.type_size(field_ty).ok()?));
        }
        Some(out)
    }

    fn calculate_struct_layout(&self, struct_ty: &HirStructType) -> CompilerResult<StructLayout> {
        let mut offset = 0u32;
        let mut field_offsets = Vec::new();
        let mut max_align = 1u32;

        for field_ty in &struct_ty.fields {
            let field_size = self.type_size(field_ty)? as u32;
            let field_align = self.type_alignment(field_ty)? as u32;

            // Update maximum alignment
            max_align = max_align.max(field_align);

            // Align offset to field alignment
            if !struct_ty.packed {
                offset = (offset + field_align - 1) & !(field_align - 1);
            }

            field_offsets.push(offset);
            offset += field_size;
        }

        // Align total size to struct alignment
        if !struct_ty.packed {
            offset = (offset + max_align - 1) & !(max_align - 1);
        }

        Ok(StructLayout {
            field_offsets,
            total_size: offset,
            alignment: max_align,
        })
    }

    /// Recursively cache struct layouts for a type (including nested structs)
    fn cache_struct_layouts_recursive(
        &self,
        ty: &HirType,
        struct_layout_cache: &mut HashMap<crate::hir::HirStructType, StructLayout>,
        size_cache: &mut HashMap<HirType, usize>,
    ) {
        match ty {
            HirType::Ptr(inner) => {
                self.cache_struct_layouts_recursive(inner, struct_layout_cache, size_cache);
            }
            HirType::Array(elem_ty, _) => {
                // Cache the ARRAY's size (critical for nested arrays like [[i32; 3]; 2])
                if let Ok(array_size) = self.type_size(ty) {
                    size_cache.insert(ty.clone(), array_size);
                }
                // Cache element type size
                if let Ok(elem_size) = self.type_size(elem_ty) {
                    size_cache.insert((**elem_ty).clone(), elem_size);
                }
                // Recursively cache nested arrays/structs
                self.cache_struct_layouts_recursive(elem_ty, struct_layout_cache, size_cache);
            }
            HirType::Struct(struct_ty) => {
                // Cache this struct's layout
                if let Ok(layout) = self.calculate_struct_layout(struct_ty) {
                    struct_layout_cache.insert(struct_ty.clone(), layout);
                }
                // Recursively cache field types
                for field_ty in &struct_ty.fields {
                    if let Ok(size) = self.type_size(field_ty) {
                        size_cache.insert(field_ty.clone(), size);
                    }
                    self.cache_struct_layouts_recursive(field_ty, struct_layout_cache, size_cache);
                }
            }
            _ => {}
        }
    }

    /// Get the size of a type in bytes
    fn type_size(&self, ty: &HirType) -> CompilerResult<usize> {
        match ty {
            HirType::Void => Ok(0),
            HirType::Bool | HirType::I8 | HirType::U8 => Ok(1),
            HirType::I16 | HirType::U16 => Ok(2),
            HirType::I32 | HirType::U32 | HirType::F32 => Ok(4),
            HirType::I64 | HirType::U64 | HirType::F64 => Ok(8),
            HirType::I128 | HirType::U128 => Ok(16),
            HirType::Ptr(_) => Ok(self.module.target_config().pointer_bytes() as usize),
            HirType::Array(elem_ty, count) => {
                let elem_size = self.type_size(elem_ty)?;
                Ok(elem_size * (*count as usize))
            }
            HirType::Struct(struct_ty) => {
                let layout = self.calculate_struct_layout(struct_ty)?;
                Ok(layout.total_size as usize)
            }
            HirType::Union(_) => {
                let layout = self.calculate_union_layout(ty)?;
                Ok(layout.total_size as usize)
            }
            HirType::Closure(_) => {
                let layout = self.calculate_closure_layout(ty)?;
                Ok(layout.total_size as usize)
            }
            _ => Ok(self.module.target_config().pointer_bytes() as usize), // Default to pointer size
        }
    }

    /// Get the alignment requirement of a type
    fn type_alignment(&self, ty: &HirType) -> CompilerResult<usize> {
        match ty {
            HirType::Void => Ok(1),
            HirType::Bool | HirType::I8 | HirType::U8 => Ok(1),
            HirType::I16 | HirType::U16 => Ok(2),
            HirType::I32 | HirType::U32 | HirType::F32 => Ok(4),
            HirType::I64 | HirType::U64 | HirType::F64 => Ok(8),
            HirType::I128 | HirType::U128 => Ok(16),
            HirType::Ptr(_) => Ok(self.module.target_config().pointer_bytes() as usize),
            HirType::Array(elem_ty, _) => self.type_alignment(elem_ty),
            HirType::Struct(struct_ty) => {
                if struct_ty.packed {
                    Ok(1)
                } else {
                    // Struct alignment is the maximum of field alignments
                    let mut max_align = 1;
                    for field_ty in &struct_ty.fields {
                        max_align = max_align.max(self.type_alignment(field_ty)?);
                    }
                    Ok(max_align)
                }
            }
            HirType::Union(_) => {
                let layout = self.calculate_union_layout(ty)?;
                Ok(layout.alignment as usize)
            }
            HirType::Closure(_) => {
                let layout = self.calculate_closure_layout(ty)?;
                Ok(layout.alignment as usize)
            }
            _ => Ok(self.module.target_config().pointer_bytes() as usize),
        }
    }

    /// Calculate union layout (discriminant + largest variant)
    fn calculate_union_layout(&self, union_ty: &HirType) -> CompilerResult<StructLayout> {
        if let HirType::Union(union_def) = union_ty {
            let discriminant_size = self.type_size(union_def.discriminant_type.as_ref())? as u32;
            let discriminant_align =
                self.type_alignment(union_def.discriminant_type.as_ref())? as u32;

            if union_def.is_c_union {
                // C-style union: no discriminant, just largest variant
                let mut max_size = 0u32;
                let mut max_align = 1u32;

                for variant in &union_def.variants {
                    let variant_size = self.type_size(&variant.ty)? as u32;
                    let variant_align = self.type_alignment(&variant.ty)? as u32;
                    max_size = max_size.max(variant_size);
                    max_align = max_align.max(variant_align);
                }

                Ok(StructLayout {
                    field_offsets: vec![0], // Single field at offset 0
                    total_size: max_size,
                    alignment: max_align,
                })
            } else {
                // Tagged union: discriminant + data
                let mut max_variant_size = 0u32;
                let mut max_variant_align = 1u32;

                for variant in &union_def.variants {
                    let variant_size = self.type_size(&variant.ty)? as u32;
                    let variant_align = self.type_alignment(&variant.ty)? as u32;
                    max_variant_size = max_variant_size.max(variant_size);
                    max_variant_align = max_variant_align.max(variant_align);
                }

                // Layout: [discriminant][padding][data]
                let data_align = max_variant_align.max(discriminant_align);
                let data_offset = (discriminant_size + data_align - 1) & !(data_align - 1);
                let total_size =
                    (data_offset + max_variant_size + data_align - 1) & !(data_align - 1);

                Ok(StructLayout {
                    field_offsets: vec![0, data_offset], // [discriminant_offset, data_offset]
                    total_size,
                    alignment: data_align,
                })
            }
        } else {
            Err(CompilerError::Backend("Expected union type".into()))
        }
    }

    /// Calculate closure layout (function pointer + captured values)
    fn calculate_closure_layout(&self, closure_ty: &HirType) -> CompilerResult<StructLayout> {
        if let HirType::Closure(closure_def) = closure_ty {
            let ptr_size = self.module.target_config().pointer_bytes() as u32;
            let ptr_align = ptr_size; // Pointer alignment equals pointer size

            let mut offset = ptr_size; // Start after function pointer
            let mut field_offsets = vec![0]; // Function pointer at offset 0
            let mut max_align = ptr_align;

            for capture in &closure_def.captures {
                let capture_size = self.type_size(&capture.ty)? as u32;
                let capture_align = self.type_alignment(&capture.ty)? as u32;

                max_align = max_align.max(capture_align);

                // Align offset to capture alignment
                offset = (offset + capture_align - 1) & !(capture_align - 1);
                field_offsets.push(offset);
                offset += capture_size;
            }

            // Align total size to closure alignment
            let total_size = (offset + max_align - 1) & !(max_align - 1);

            Ok(StructLayout {
                field_offsets,
                total_size,
                alignment: max_align,
            })
        } else {
            Err(CompilerError::Backend("Expected closure type".into()))
        }
    }
}

impl HirType {
    #[allow(dead_code)]
    fn is_float(&self) -> bool {
        match self {
            HirType::F32 | HirType::F64 => true,
            HirType::Vector(elem_ty, _) => elem_ty.is_float(),
            _ => false,
        }
    }

    #[allow(dead_code)]
    fn is_signed(&self) -> bool {
        match self {
            HirType::I8 | HirType::I16 | HirType::I32 | HirType::I64 | HirType::I128 => true,
            HirType::Vector(elem_ty, _) => elem_ty.is_signed(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cranelift_backend_creation() {
        let backend = CraneliftBackend::new();
        assert!(backend.is_ok());
    }

    /// 5b end-to-end: build a small counted-loop function in HIR, compile
    /// it at tier 0 then again at tier 1, and verify a real OSR helper
    /// gets emitted (`pending_osr_helpers` non-empty) and finalizes to a
    /// non-null code pointer.
    #[test]
    fn test_osr_helper_emission_for_counted_loop() {
        use crate::hir::{BinaryOp, HirValue, HirValueKind};
        use crate::hir::{HirBlock, HirInstruction, HirPhi, HirTerminator};
        use crate::hir::{HirConstant, HirFunction, HirFunctionSignature, HirParam};
        use indexmap::IndexMap;
        use zyntax_typed_ast::InternedString;

        // Hand-build:
        //   fn count_to(n: i32) -> i32 {
        //       let mut i = 0;
        //       let mut sum = 0;
        //       while i < n { sum += i; i += 1 }
        //       sum
        //   }
        //
        // Live-ins at the loop header: i, sum (phi results) + n (function
        // param used in the comparison). 3 total — within the 4-arg cap.
        let i32_ty = HirType::I32;

        // HirIds for everything.
        let func_id = HirId::new();
        let n_id = HirId::new();
        let zero_i_id = HirId::new();
        let zero_sum_id = HirId::new();
        let one_id = HirId::new();
        let entry_id = HirId::new();
        let header_id = HirId::new();
        let body_id = HirId::new();
        let exit_id = HirId::new();
        let phi_i = HirId::new();
        let phi_sum = HirId::new();
        let cmp_id = HirId::new();
        let next_sum_id = HirId::new();
        let next_i_id = HirId::new();

        let mut values: IndexMap<HirId, HirValue> = IndexMap::new();
        values.insert(
            n_id,
            HirValue {
                id: n_id,
                ty: i32_ty.clone(),
                kind: HirValueKind::Parameter(0),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );
        for (id, v) in [(zero_i_id, 0i32), (zero_sum_id, 0), (one_id, 1)] {
            values.insert(
                id,
                HirValue {
                    id,
                    ty: i32_ty.clone(),
                    kind: HirValueKind::Constant(HirConstant::I32(v)),
                    uses: std::collections::HashSet::new(),
                    span: None,
                },
            );
        }
        for id in [phi_i, phi_sum, next_sum_id, next_i_id] {
            values.insert(
                id,
                HirValue {
                    id,
                    ty: i32_ty.clone(),
                    kind: HirValueKind::Instruction,
                    uses: std::collections::HashSet::new(),
                    span: None,
                },
            );
        }
        values.insert(
            cmp_id,
            HirValue {
                id: cmp_id,
                ty: HirType::Bool,
                kind: HirValueKind::Instruction,
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );

        // Entry block: branch to header.
        let mut entry_block = HirBlock {
            id: entry_id,
            label: Some(InternedString::new_global("entry")),
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Branch { target: header_id },
            dominance_frontier: std::collections::HashSet::new(),
            predecessors: vec![],
            successors: vec![header_id],
        };
        // Constants don't need to be emitted as instructions in HIR — they
        // live in `values` and Cranelift's Phase-3 constant emission picks
        // them up.
        // Add a dummy iconst use to anchor the constants in the entry path.
        let _ = &mut entry_block;

        // Loop header: phi(i), phi(sum), cmp i < n, cond_br body/exit.
        let header_block = HirBlock {
            id: header_id,
            label: Some(InternedString::new_global("header")),
            phis: vec![
                HirPhi {
                    result: phi_i,
                    ty: i32_ty.clone(),
                    incoming: vec![(zero_i_id, entry_id), (next_i_id, body_id)],
                },
                HirPhi {
                    result: phi_sum,
                    ty: i32_ty.clone(),
                    incoming: vec![(zero_sum_id, entry_id), (next_sum_id, body_id)],
                },
            ],
            instructions: vec![HirInstruction::Binary {
                op: BinaryOp::Lt,
                result: cmp_id,
                ty: HirType::Bool,
                left: phi_i,
                right: n_id,
            }],
            terminator: HirTerminator::CondBranch {
                condition: cmp_id,
                true_target: body_id,
                false_target: exit_id,
            },
            dominance_frontier: std::collections::HashSet::new(),
            predecessors: vec![entry_id, body_id],
            successors: vec![body_id, exit_id],
        };

        // Body: next_sum = sum + i; next_i = i + 1; br header.
        let body_block = HirBlock {
            id: body_id,
            label: Some(InternedString::new_global("body")),
            phis: vec![],
            instructions: vec![
                HirInstruction::Binary {
                    op: BinaryOp::Add,
                    result: next_sum_id,
                    ty: i32_ty.clone(),
                    left: phi_sum,
                    right: phi_i,
                },
                HirInstruction::Binary {
                    op: BinaryOp::Add,
                    result: next_i_id,
                    ty: i32_ty.clone(),
                    left: phi_i,
                    right: one_id,
                },
            ],
            terminator: HirTerminator::Branch { target: header_id },
            dominance_frontier: std::collections::HashSet::new(),
            predecessors: vec![header_id],
            successors: vec![header_id],
        };

        // Exit: return sum.
        let exit_block = HirBlock {
            id: exit_id,
            label: Some(InternedString::new_global("exit")),
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Return {
                values: vec![phi_sum],
            },
            dominance_frontier: std::collections::HashSet::new(),
            predecessors: vec![header_id],
            successors: vec![],
        };

        let mut blocks: IndexMap<HirId, HirBlock> = IndexMap::new();
        blocks.insert(entry_id, entry_block);
        blocks.insert(header_id, header_block);
        blocks.insert(body_id, body_block);
        blocks.insert(exit_id, exit_block);

        let signature = HirFunctionSignature {
            params: vec![HirParam {
                id: n_id,
                name: InternedString::new_global("n"),
                ty: i32_ty.clone(),
                attributes: crate::hir::ParamAttributes::default(),
                ownership: Default::default(),
            }],
            returns: vec![i32_ty.clone()],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: true,
        };

        let mut function = HirFunction::new(InternedString::new_global("count_to"), signature);
        function.values = values;
        function.blocks = blocks;
        function.entry_block = entry_id;
        function.is_external = false;

        // ── osr_layout sanity ────────────────────────────────────────────
        let layout = crate::osr::osr_layout(&function, header_id)
            .expect("loop header should have a representable OSR layout");
        assert_eq!(layout.phi_count, 2, "expected 2 phi live-ins (i, sum)");
        assert!(
            layout.live_ins.contains(&n_id),
            "n should be a non-phi live-in"
        );
        assert!(
            layout.live_ins.len() <= crate::osr::OSR_MAX_LIVE_INS,
            "live-ins ({}) exceed cap",
            layout.live_ins.len()
        );

        // ── helper emission ──────────────────────────────────────────────
        // We only exercise the tier-1 path here. The legacy
        // compile_function declares + compiles the function fresh; its
        // tier-1 hook calls compile_osr_helpers afterwards. We don't
        // need a prior tier-0 compile_module — that path also defines
        // the function in the JIT module and would collide on the
        // re-define here.
        let mut backend = CraneliftBackend::new().unwrap();
        backend.set_compile_tier(1);
        backend.set_compile_bead_id(0xBEAD);
        backend
            .compile_function(func_id, &function)
            .expect("tier-1 compile_function");
        let _ = backend.module.finalize_definitions();

        let helpers = backend.take_pending_osr_helpers();
        assert!(
            !helpers.is_empty(),
            "tier-1 compile of a counted loop must emit at least one OSR helper"
        );
        let (site, code) = helpers[0];
        assert_eq!(
            site,
            layout.site_key(),
            "helper site_key should match the layout"
        );
        assert!(
            !code.is_null(),
            "finalized helper code pointer must be non-null"
        );
    }
}
