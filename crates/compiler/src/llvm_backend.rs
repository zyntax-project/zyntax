// LLVM Backend Implementation for Zyntax
//
// This backend compiles HIR (High-level Intermediate Representation) to LLVM IR,
// enabling production-quality code generation with world-class optimizations.
//
// Architecture:
// - LLVMBackend: Main compiler struct managing LLVM context, module, and builder
// - Type Translation: Maps HIR types to LLVM types
// - Instruction Compilation: Converts HIR instructions to LLVM IR
// - Function Compilation: Handles function signatures, bodies, and calling conventions
//
// Use cases:
// 1. AOT (Ahead-of-Time): Full program optimization for production binaries
// 2. Tiered JIT: Optimize hot functions while keeping cold paths in Cranelift/VM
// 3. Profile-guided optimization: Recompile based on runtime profiling

use crate::hir::{
    BinaryOp, CastOp, HirBlock, HirCallable, HirConstant, HirFunction, HirGlobal, HirId,
    HirInstruction, HirModule, HirPhi, HirTerminator, HirType, HirVTable, HirValueKind, UnaryOp,
    VectorMinMaxKind, VectorUnaryKind,
};
use crate::{CompilerError, CompilerResult};
use indexmap::IndexMap;
use inkwell::{
    basic_block::BasicBlock,
    builder::Builder,
    context::Context,
    module::Module,
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType, IntType},
    values::{
        BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, PhiValue, PointerValue,
        ValueKind,
    },
    AddressSpace, AtomicOrdering as LLVMAtomicOrdering, AtomicRMWBinOp, FloatPredicate,
    IntPredicate,
};

// Helper macro to convert inkwell errors to CompilerError
macro_rules! llvm_try {
    ($expr:expr) => {
        $expr.map_err(|e| CompilerError::CodeGen(format!("LLVM error: {}", e)))?
    };
}

/// Main LLVM backend compiler
///
/// Manages the LLVM context, module, and compilation state.
/// Lifetime 'ctx ties all LLVM objects to the context they were created in.
pub struct LLVMBackend<'ctx> {
    /// LLVM context - all types and values are tied to this
    context: &'ctx Context,

    /// LLVM module - container for all compiled functions and globals
    module: Module<'ctx>,

    /// LLVM IR builder - used to construct instructions
    builder: Builder<'ctx>,

    /// Maps HIR value IDs to compiled LLVM values
    value_map: IndexMap<HirId, BasicValueEnum<'ctx>>,

    /// Maps HIR value IDs to their original HIR types (for indirect calls and other type lookups)
    type_map: IndexMap<HirId, HirType>,

    /// Maps HIR function IDs to compiled LLVM functions
    functions: IndexMap<HirId, FunctionValue<'ctx>>,

    /// Maps HIR basic block IDs to LLVM basic blocks
    block_map: IndexMap<HirId, BasicBlock<'ctx>>,

    /// Maps HIR phi result IDs to LLVM phi nodes (for adding incoming edges later)
    phi_map: IndexMap<HirId, PhiValue<'ctx>>,

    /// Current function being compiled (for accessing locals, blocks, etc.)
    current_function: Option<FunctionValue<'ctx>>,

    /// Maps HIR global IDs to compiled LLVM global values (persists across functions)
    globals_map: IndexMap<HirId, BasicValueEnum<'ctx>>,

    /// Symbol signatures for auto-boxing (symbol name → signature)
    symbol_signatures: std::collections::HashMap<String, crate::zrtl::ZrtlSymbolSig>,

    /// Phase H: effect-handler lookup index, populated at the top of
    /// `compile_module` from `hir_module.handlers`. Keyed by
    /// `(effect_id, op_name)` so a `PerformEffect` lowering can map
    /// directly to `(handler_fn_hir_id, is_resumable)` without
    /// re-walking `hir_module.handlers` on every emit. Mirrors the
    /// runtime-side handler-stack lookup at
    /// `effect_runtime::__zyntax_effect_lookup_handler` but at
    /// compile time — Tier 1 path is single-handler-per-effect, so a
    /// static map suffices.
    ///
    /// The `HirId` value is the *function* id of the standalone
    /// handler-op function (mangled `{Handler}${op}`) the
    /// algebraic_effects pass emitted — not the handler/effect id.
    /// `self.functions[hir_id]` produces the LLVM `FunctionValue`.
    effect_handler_index:
        std::collections::HashMap<(HirId, zyntax_typed_ast::InternedString), (HirId, bool)>,

    /// Per-function calling-convention selection, populated during
    /// the declare-all walk in `compile_module`. Used at every direct
    /// call site so the call-site cc matches the declared cc — LLVM's
    /// verifier rejects mismatches.
    ///
    /// Keys are the callee's HirId; values are LLVM cc numbers
    /// (0 = ccc, 8 = fastcc). Indirect / intrinsic / runtime-symbol
    /// calls intentionally aren't represented — they stay on C cc.
    func_cc: std::collections::HashMap<HirId, u32>,

    /// Set of internal functions that are exported via dlsym for the
    /// JIT host (entry point + named exports). Members stay on C cc
    /// even though `is_external == false`, because the harness calls
    /// them via `transmute<extern "C" fn(...)>` and would crash on
    /// fastcc. Populated once at the top of `compile_module`.
    dlsym_set: std::collections::HashSet<HirId>,

    /// Set of functions that directly recurse (their body contains a
    /// `HirCallable::Function` callee equal to their own id). Used to
    /// gate `inlinehint`: a hint on a recursive function is at best
    /// noise and at worst feeds the inliner into a loop.
    self_recursive_set: std::collections::HashSet<HirId>,

    /// Optional reachable-function filter, mirroring Cranelift's
    /// `set_only_compile_reachable`. When `Some`, `compile_module`
    /// skips declaration + body emission for every function whose
    /// HirId is not in the set. The runtime computes this from
    /// `reachable_function_ids(&module, &["main"])` once per
    /// install — the BC interp's lazy-compile fallback still covers
    /// any unexpected reach at execution time. Without this filter
    /// LLVM compiles every prelude helper (`Option<T>`, `Result<T, E>`,
    /// List/Iterator/Tensor stdlib — ~100 functions on the bench
    /// kernels) and pays 300-900 ms of codegen + verifier time per
    /// install plus surfaces every type-translation gap the prelude
    /// has.
    only_compile_reachable: Option<std::collections::HashSet<HirId>>,

    /// The functions a host enters through. These keep the name they
    /// were written with; every other local function is mangled to its
    /// id so two of them cannot share a symbol. Configured, because
    /// this layer has no entry point of its own.
    entry_names: std::collections::HashSet<String>,

    /// Whether the compilation **target** (not the host) has x86 VNNI
    /// (`VPDPBUSD`). Set by the caller from the target-machine features:
    /// for a JIT the target is the host, so it comes from host feature
    /// detection; for a portable AOT object it must come from the AOT
    /// target's declared features, NOT the build host — otherwise a
    /// host-only `vpdpbusd` would be baked into a non-portable binary.
    /// The `VectorDot` lowering reads this to choose the fused intrinsic
    /// vs. the portable widening fallback. Defaults to `false`
    /// (conservative — no VNNI unless the caller opts in for its target).
    #[allow(dead_code)] // read only under #[cfg(target_arch = "x86_64")]
    x86_target_vnni: bool,
}

/// Approximate byte size of a `HirType` for laying out union payloads
/// in the LLVM backend.
///
/// We only need this for the `Union` arm of `translate_type` — picking
/// the widest non-Void variant's storage size so the payload slot can
/// hold any of them. Goes through the fixed-width primitive sizes,
/// recurses into composites by summing fields, and falls back to a
/// conservative 8 bytes (one pointer's worth) for everything else
/// (Opaque, named structs we haven't walked yet, function pointers,
/// vectors with non-standard lane shapes). Conservative wins here:
/// over-allocating the union payload is a layout no-op; under-
/// allocating corrupts whichever variant is widest.
fn hir_type_size_bytes(ty: &HirType) -> u32 {
    match ty {
        HirType::Void => 0,
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 => 8,
        HirType::I128 | HirType::U128 => 16,
        HirType::Ptr(_) | HirType::Ref { .. } => 8,
        HirType::Array(elem, n) => hir_type_size_bytes(elem) * (*n as u32),
        HirType::Struct(s) => s.fields.iter().map(hir_type_size_bytes).sum(),
        HirType::Vector(elem, n) => hir_type_size_bytes(elem) * (*n),
        HirType::Union(u) => {
            let disc = hir_type_size_bytes(&u.discriminant_type);
            let payload = u
                .variants
                .iter()
                .map(|v| hir_type_size_bytes(&v.ty))
                .max()
                .unwrap_or(0);
            disc + payload
        }
        // Conservative pointer-width fallback for variants whose
        // layout we can't easily compute here (Function, Opaque,
        // closures, named/forward types). Wider-than-needed payload
        // is harmless; under-sized would corrupt.
        _ => 8,
    }
}

impl<'ctx> LLVMBackend<'ctx> {
    /// Create a new LLVM backend
    ///
    /// # Arguments
    /// * `context` - LLVM context (must outlive the backend)
    /// * `module_name` - Name for the LLVM module
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        Self {
            context,
            module,
            builder,
            value_map: IndexMap::new(),
            type_map: IndexMap::new(),
            functions: IndexMap::new(),
            block_map: IndexMap::new(),
            phi_map: IndexMap::new(),
            current_function: None,
            globals_map: IndexMap::new(),
            symbol_signatures: std::collections::HashMap::new(),
            effect_handler_index: std::collections::HashMap::new(),
            func_cc: std::collections::HashMap::new(),
            dlsym_set: std::collections::HashSet::new(),
            self_recursive_set: std::collections::HashSet::new(),
            only_compile_reachable: None,
            entry_names: Default::default(),
            x86_target_vnni: false,
        }
    }

    /// Register symbol signatures for auto-boxing support
    pub fn register_symbol_signatures(&mut self, symbols: &[crate::zrtl::RuntimeSymbolInfo]) {
        for sym in symbols {
            if let Some(sig) = &sym.sig {
                self.symbol_signatures
                    .insert(sym.name.to_string(), sig.clone());
            }
        }
    }

    /// Check if a symbol parameter expects DynamicBox
    fn param_needs_boxing(&self, symbol_name: &str, param_index: usize) -> bool {
        self.symbol_signatures
            .get(symbol_name)
            .map(|sig| sig.param_is_dynamic(param_index))
            .unwrap_or(false)
    }

    /// Compile an entire HIR module to LLVM IR
    ///
    /// This is the main entry point for compilation. It:
    /// 1. Processes global variables (including vtables)
    /// 2. Declares all functions (for forward references)
    /// 3. Compiles function bodies
    /// 4. Returns the compiled LLVM module
    pub fn compile_module(&mut self, hir_module: &HirModule) -> CompilerResult<String> {
        // Phase H: build the effect-handler lookup index up front so
        // each PerformEffect emission is an O(1) map probe.  Keyed by
        // (effect_id, op_name) → (handler-fn HirId, is_resumable).
        // See the `effect_handler_index` field docstring for rationale.
        self.effect_handler_index.clear();
        for handler in hir_module.handlers.values() {
            for impl_ in &handler.implementations {
                let mangled =
                    crate::effect_codegen::mangle_handler_op_name(handler.name, impl_.op_name);
                // Resolve the standalone fn the algebraic_effects pass
                // emitted (named `{Handler}${op}`) to its HirId so
                // `self.functions[hir_id]` finds the LLVM FunctionValue.
                let fn_hir_id = hir_module
                    .functions
                    .iter()
                    .find(|(_, f)| f.name.resolve_global().as_deref() == Some(mangled.as_str()))
                    .map(|(id, _)| *id);
                if let Some(fn_hir_id) = fn_hir_id {
                    self.effect_handler_index.insert(
                        (handler.effect_id, impl_.op_name),
                        (fn_hir_id, impl_.is_resumable),
                    );
                } else {
                    log::warn!(
                        "[LLVM] effect handler op '{}' has no matching function in module — \
                         PerformEffect calls to it will produce a dummy zero",
                        mangled
                    );
                }
            }
        }

        // Pre-pass: compute cc / attribute policy inputs.
        // dlsym_set tells us which internals the JIT host will
        // reach via `transmute<extern "C" fn(...)>`; those must
        // stay on C cc. self_recursive_set gates `inlinehint`.
        self.dlsym_set = Self::build_dlsym_set(hir_module);
        self.self_recursive_set = Self::build_self_recursive_set(hir_module);
        self.func_cc.clear();

        // Phase 1: Process globals first (including vtables) in deterministic sorted order
        let mut global_ids: Vec<_> = hir_module.globals.keys().cloned().collect();
        global_ids.sort_by_key(|id| format!("{:?}", id));

        for id in &global_ids {
            if let Some(global) = hir_module.globals.get(id) {
                self.compile_global(*id, global)?;
            }
        }

        // Phase 2: Declare all functions (allows forward references) in deterministic sorted order
        let mut declare_ids: Vec<_> = hir_module.functions.keys().cloned().collect();
        declare_ids.sort_by_key(|id| format!("{:?}", id));

        for id in &declare_ids {
            if let Some(allowed) = &self.only_compile_reachable {
                if !allowed.contains(id) {
                    continue;
                }
            }
            if let Some(func) = hir_module.functions.get(id) {
                self.declare_function(*id, func)?;
            }
        }

        // Phase 3: Compile function bodies in deterministic sorted order
        let mut function_ids: Vec<_> = hir_module.functions.keys().cloned().collect();
        function_ids.sort_by_key(|id| format!("{:?}", id));

        for id in &function_ids {
            if let Some(allowed) = &self.only_compile_reachable {
                if !allowed.contains(id) {
                    continue;
                }
            }
            if let Some(func) = hir_module.functions.get(id) {
                self.compile_function(*id, func)?;
            }
        }

        // Return LLVM IR as string for inspection/debugging
        let ir = self.module.print_to_string().to_string();
        log::debug!("[LLVM] Generated LLVM IR:\n{}", ir);
        Ok(ir)
    }

    /// Limit `compile_module` to a specific reachable subset.
    ///
    /// Mirrors `CraneliftBackend::set_only_compile_reachable`. When
    /// `Some`, the declare/body loops skip every HirId not in the set.
    /// Call before `compile_module`. Globals + effect-handler index
    /// are still walked in full because they're constant-time and
    /// keeping them intact preserves cross-function symbol resolution
    /// against any reachable site.
    /// Name the functions a host enters the program through.
    pub fn set_entry_names(&mut self, names: std::collections::HashSet<String>) {
        self.entry_names = names;
    }

    pub fn set_only_compile_reachable(
        &mut self,
        allowed: Option<std::collections::HashSet<HirId>>,
    ) {
        self.only_compile_reachable = allowed;
    }

    /// Declare whether the compilation **target** has x86 VNNI, gating
    /// the fused `VPDPBUSD` lowering of `VectorDot`. Call before
    /// `compile_module`. A JIT passes its host's capability (target ==
    /// host); a portable AOT object passes its AOT target's declared
    /// features so no host-only instruction is baked in.
    pub fn set_x86_target_vnni(&mut self, has_vnni: bool) {
        self.x86_target_vnni = has_vnni;
    }

    /// Get a reference to the compiled LLVM module
    ///
    /// This is useful for creating an execution engine or writing to a file.
    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Consume the backend and return the LLVM module
    ///
    /// This transfers ownership of the module, which is required for MCJIT
    /// since the execution engine takes ownership of the module.
    pub fn into_module(self) -> Module<'ctx> {
        self.module
    }

    /// Build the set of functions that will be dlsym'd by the JIT
    /// host. Mirrors the export logic in
    /// `LLVMJitBackend::get_function_symbols` (jit_backend.rs:366):
    /// every emitted-with-body function gets exported. We use it to
    /// keep those functions on the platform C calling convention so
    /// the host's `transmute<extern "C" fn(...)>` is sound.
    fn build_dlsym_set(hir_module: &HirModule) -> std::collections::HashSet<HirId> {
        let mut set = std::collections::HashSet::new();
        for (id, func) in &hir_module.functions {
            if !func.is_external {
                set.insert(*id);
            }
        }
        set
    }

    /// Cheap direct-recursion pass: a function is "self-recursive"
    /// when its body contains a `HirCallable::Function` callee equal
    /// to its own id. Mutual recursion is intentionally not modelled;
    /// the inliner has its own cycle safeguards.
    fn build_self_recursive_set(hir_module: &HirModule) -> std::collections::HashSet<HirId> {
        let mut set = std::collections::HashSet::new();
        for (id, func) in &hir_module.functions {
            if func.is_external {
                continue;
            }
            'outer: for block in func.blocks.values() {
                for instr in &block.instructions {
                    if let HirInstruction::Call {
                        callee: HirCallable::Function(callee_id),
                        ..
                    } = instr
                    {
                        if callee_id == id {
                            set.insert(*id);
                            break 'outer;
                        }
                    }
                }
            }
        }
        set
    }

    /// Count the total HIR instructions in a function's body. Used
    /// as a coarse "size" measure for `inlinehint` gating — we don't
    /// want the inliner to pull `fib`-sized recursive bodies inline.
    fn function_body_size(func: &HirFunction) -> usize {
        func.blocks.values().map(|b| b.instructions.len()).sum()
    }

    /// Decide what LLVM cc to declare a function with.
    ///
    /// - External and dlsym'd internals stay on C cc (0) — the host
    ///   reaches them via `transmute<extern "C" fn(...)>`.
    /// - Everything else with `CallingConvention::Fast` becomes
    ///   fastcc (8).
    /// - Other HIR conventions (System, WebKit) keep the LLVM default.
    fn function_calling_convention(
        func: &HirFunction,
        id: HirId,
        dlsym_set: &std::collections::HashSet<HirId>,
    ) -> u32 {
        use crate::hir::CallingConvention;
        if func.is_external {
            return 0;
        }
        if dlsym_set.contains(&id) {
            return 0;
        }
        match func.calling_convention {
            CallingConvention::Fast => 8,
            _ => 0,
        }
    }

    /// Stamp the performance-relevant function attributes onto an
    /// internal function. Externals are not handled here — the
    /// caller must already have decided this is an internal that
    /// matches the policy. `dlsym_exported` true means the function
    /// is reachable from the host's symbol table, which restricts us
    /// to attributes that don't surprise an external observer.
    fn apply_internal_attributes(
        &self,
        fv: FunctionValue<'ctx>,
        is_dlsym_exported: bool,
        body_size: usize,
        is_recursive: bool,
    ) {
        use inkwell::attributes::{Attribute, AttributeLoc};

        let mut add = |name: &str| {
            let kind = Attribute::get_named_enum_kind_id(name);
            if kind == 0 {
                // Unknown attribute — ignore rather than panic. Inkwell
                // returns 0 for names LLVM doesn't recognise.
                return;
            }
            let attr = self.context.create_enum_attribute(kind, 0);
            fv.add_attribute(AttributeLoc::Function, attr);
        };

        // Always-safe attributes for both fastcc internals and the
        // dlsym'd exports: we never emit `invoke` (effect-Resume is
        // a plain call), and we want the inliner / loop-deleter to
        // assume forward progress.
        add("nounwind");
        add("mustprogress");

        if !is_dlsym_exported {
            // nofree: we never emit free/realloc from internal HIR
            // lowering — those are runtime calls, and the runtime
            // funcs themselves stay on C cc with no attributes.
            add("nofree");

            // inlinehint: small + non-recursive only. The inliner
            // has its own cycle safeguards but we don't want to pile
            // on with recursive bodies, and large bodies just bloat
            // the IR.
            const INLINE_HINT_MAX_BODY: usize = 50;
            if !is_recursive && body_size <= INLINE_HINT_MAX_BODY {
                add("inlinehint");
            }
        }
    }

    /// Declare a function signature without compiling its body
    ///
    /// This allows other functions to call this one before it's fully compiled.
    fn declare_function(
        &mut self,
        id: HirId,
        func: &HirFunction,
    ) -> CompilerResult<FunctionValue<'ctx>> {
        // Translate parameter types
        let param_types: Vec<BasicMetadataTypeEnum> = func
            .signature
            .params
            .iter()
            .map(|param| self.translate_type(&param.ty).map(|t| t.into()))
            .collect::<CompilerResult<Vec<_>>>()?;

        // Translate return type
        let fn_type = if func.signature.returns.is_empty() {
            // Void function
            self.context.void_type().fn_type(&param_types, false)
        } else if func.signature.returns.len() == 1 {
            // Function returning a single value
            let return_type = self.translate_type(&func.signature.returns[0])?;
            return_type.fn_type(&param_types, false)
        } else {
            // Multiple return values - represent as struct (tuple)
            let return_types: Vec<BasicTypeEnum> = func
                .signature
                .returns
                .iter()
                .map(|ty| self.translate_type(ty))
                .collect::<CompilerResult<Vec<_>>>()?;

            let tuple_type = self.context.struct_type(&return_types, false);
            tuple_type.fn_type(&param_types, false)
        };

        // Add function to module
        // Use actual name for:
        // - External functions (for linking with C libraries)
        // - Main function (for linker entry point in AOT compilation)
        // Otherwise use mangled name with HirId for internal functions
        let actual_name = func
            .name
            .resolve_global()
            .unwrap_or_else(|| format!("{:?}", func.name));
        let fn_name = if func.is_external {
            // An extern's declared name is an alias; the symbol the host
            // actually provides is the link name, when one is set.
            func.link_name.clone().unwrap_or(actual_name)
        } else if self.entry_names.contains(&actual_name) {
            // An entry keeps the name a host asks for it by.
            actual_name
        } else {
            // Regular functions use mangled name with HirId
            format!("func_{:?}", id)
        };
        let fn_value = self.module.add_function(&fn_name, fn_type, None);

        // Set parameter names (helps with debugging IR)
        for (i, param) in func.signature.params.iter().enumerate() {
            let param_name = format!("param_{}", i);
            fn_value
                .get_nth_param(i as u32)
                .unwrap()
                .set_name(&param_name);
        }

        // Apply calling-convention + function attributes per the
        // fastcc/attribute design. Externals stay on platform C cc
        // with no attributes; dlsym'd internals get C cc but a
        // conservative attribute set; other internals get fastcc
        // and the full attribute set (gated on size/recursion).
        let cc = Self::function_calling_convention(func, id, &self.dlsym_set);
        if cc != 0 {
            fn_value.set_call_conventions(cc);
        }
        self.func_cc.insert(id, cc);
        if !func.is_external {
            let is_dlsym_exported = self.dlsym_set.contains(&id);
            let body_size = Self::function_body_size(func);
            let is_recursive = self.self_recursive_set.contains(&id);
            self.apply_internal_attributes(fn_value, is_dlsym_exported, body_size, is_recursive);
        }

        // Store for later reference
        self.functions.insert(id, fn_value);

        Ok(fn_value)
    }

    /// Compile a function body
    fn compile_function(&mut self, id: HirId, func: &HirFunction) -> CompilerResult<()> {
        let fn_value = self.functions[&id];
        self.current_function = Some(fn_value);

        // Skip external functions (declarations only)
        if func.is_external {
            self.current_function = None;
            return Ok(());
        }

        // Clear block, value, and phi maps for this function
        self.block_map.clear();
        self.phi_map.clear();
        self.value_map.clear(); // Clear value_map between functions
        self.type_map.clear(); // Clear type_map between functions

        // Map function parameters to HIR value IDs and store their types
        for (i, param) in func.signature.params.iter().enumerate() {
            let param_value = fn_value.get_nth_param(i as u32).unwrap();
            self.value_map.insert(param.id, param_value);
            self.type_map.insert(param.id, param.ty.clone());
        }

        // Map constant values, parameters, and special instruction values to LLVM values.
        //
        // Populate `type_map` for *every* value so backend peepholes that
        // need to recover the HIR-level pointee type (the field-access
        // struct-GEP rewrite, the ExtractValue/InsertValue ptr arms) can
        // do so for instruction-produced values too, not just parameters.
        for (value_id, value) in &func.values {
            self.type_map.insert(*value_id, value.ty.clone());
            match &value.kind {
                HirValueKind::Constant(constant) => {
                    let llvm_constant = self.compile_constant(constant)?;
                    self.value_map.insert(*value_id, llvm_constant);
                }
                HirValueKind::Parameter(param_index) => {
                    // SSA creates new value IDs for parameters with HirValueKind::Parameter
                    // Map these to the actual LLVM function parameters
                    if let Some(param_value) = fn_value.get_nth_param(*param_index) {
                        self.value_map.insert(*value_id, param_value);
                        self.type_map.insert(*value_id, value.ty.clone());
                    }
                }
                HirValueKind::Instruction => {
                    // For instruction values that appear in func.values (like undef structs),
                    // create an undef value of the appropriate type
                    let llvm_type = self.translate_type(&value.ty)?;
                    let undef_value = llvm_type.const_zero(); // or use undef if available
                    self.value_map.insert(*value_id, undef_value);
                }
                HirValueKind::Undef => {
                    // Map undef values to zero constants (for IDF-based SSA)
                    // This handles void-returning function calls where SSA creates undef placeholders
                    let llvm_type = self.translate_type(&value.ty)?;
                    let undef_value = llvm_type.const_zero();
                    self.value_map.insert(*value_id, undef_value);
                }
                HirValueKind::Global(global_id) => {
                    // Map global references to their LLVM global values
                    // The global should have been compiled in phase 1
                    if let Some(&global_value) = self.globals_map.get(global_id) {
                        self.value_map.insert(*value_id, global_value);
                    }
                }
                _ => {}
            }
        }

        // Phase 1: Create LLVM basic blocks for all HIR blocks
        // IMPORTANT: Create entry block FIRST, as the first block added becomes the entry in LLVM IR
        let entry_block_name = format!("bb_{:?}", func.entry_block);
        let entry_llvm_block = self.context.append_basic_block(fn_value, &entry_block_name);
        self.block_map.insert(func.entry_block, entry_llvm_block);

        // Create remaining blocks in insertion order (IndexMap preserves insertion order)
        // This ensures deterministic LLVM IR generation and correct phi node handling
        for (block_id, _) in func.blocks.iter() {
            if *block_id != func.entry_block {
                let block_name = format!("bb_{:?}", block_id);
                let llvm_block = self.context.append_basic_block(fn_value, &block_name);
                self.block_map.insert(*block_id, llvm_block);
            }
        }

        // Phase 2: Compile blocks in reverse post-order, so every
        // definition is built before anything that reads it.
        //
        // Insertion order was used here and held only by accident: a
        // block's definitions happened to be written before its readers
        // because of the order passes created blocks. It is not a
        // property of the IR. A value defined in a preheader that was
        // inserted after the loop body, which is what hoisting a
        // loop-invariant broadcast out of a vectorized loop produces,
        // was read before it existed and silently became a zero vector.
        // Reverse post-order is the order that actually guarantees what
        // the old comment claimed.
        let rpo = crate::analysis::DominatorTree::new(func);
        let mut emitted: std::collections::HashSet<HirId> = std::collections::HashSet::new();
        for block_id in rpo.rpo() {
            if let (Some(llvm_block), Some(hir_block)) =
                (self.block_map.get(block_id), func.blocks.get(block_id))
            {
                self.builder.position_at_end(*llvm_block);
                self.compile_block_with_terminator(block_id, hir_block, func)?;
                emitted.insert(*block_id);
            }
        }
        // A block the entry cannot reach is absent from the traversal.
        // It still needs a body, since it was declared above and LLVM
        // rejects a block without a terminator.
        for (block_id, hir_block) in func.blocks.iter() {
            if emitted.contains(block_id) {
                continue;
            }
            if let Some(llvm_block) = self.block_map.get(block_id) {
                self.builder.position_at_end(*llvm_block);
                self.compile_block_with_terminator(block_id, hir_block, func)?;
            }
        }

        // Phase 3: Add incoming edges to phi nodes
        // Now that all blocks are compiled and all values are in value_map,
        // we can add the incoming edges to phi nodes
        // Iterate in insertion order (IndexMap preserves this)
        log::debug!(
            "[LLVM] Phase 3: Adding phi incoming edges. value_map has {} entries",
            self.value_map.len()
        );

        for (block_id, hir_block) in func.blocks.iter() {
            for phi in &hir_block.phis {
                log::debug!(
                    "[LLVM] Processing phi {:?} in block {:?}",
                    phi.result,
                    block_id
                );
                if let Some(phi_value) = self.phi_map.get(&phi.result) {
                    // Iterate phi incoming in original order (preserved by data structure)
                    for (value_id, pred_block_id) in &phi.incoming {
                        log::debug!(
                            "[LLVM]   incoming: value={:?} from block={:?}",
                            value_id,
                            pred_block_id
                        );
                        let incoming_value = self.get_value(*value_id)?;
                        log::debug!("[LLVM]     resolved to: {:?}", incoming_value);
                        let incoming_block =
                            self.block_map.get(pred_block_id).ok_or_else(|| {
                                CompilerError::CodeGen(format!(
                                    "Phi node references unknown block: {:?}",
                                    pred_block_id
                                ))
                            })?;
                        let incoming_value = self.coerce_incoming_for_phi(
                            incoming_value,
                            phi_value.as_basic_value().get_type(),
                            *incoming_block,
                        )?;
                        phi_value.add_incoming(&[(&incoming_value, *incoming_block)]);
                    }
                }
            }
        }

        // Phase 4: Verify the function
        // (LLVM will check that all blocks have terminators and phi nodes are valid)

        self.current_function = None;
        Ok(())
    }

    /// Emit an OSR helper for `header`: a standalone function that resumes
    /// `func` at that loop header instead of at its entry.
    ///
    /// The signature is `(ptr) -> <func's return type>`: the pointer is the
    /// frame a tier-0 back-edge fills with the live-ins. A synthetic
    /// prologue reads them back and jumps to the header, so the header's
    /// phis take their loop-entry values from the prologue rather than from
    /// the original preheader.
    ///
    /// Returns the helper's name; the caller resolves it to an address once
    /// the module is installed.
    pub fn compile_osr_helper(
        &mut self,
        func: &HirFunction,
        layout: &crate::osr::OsrLayout,
    ) -> CompilerResult<String> {
        // Reject shapes the helper cannot express before creating anything.
        // An LLVM function or block cannot be safely removed once other
        // values reference it — deleting one leaves dangling uses that crash
        // the pass pipeline rather than failing cleanly.
        let in_loop = crate::osr::blocks_reachable_from(func, layout.header);
        for block_id in &in_loop {
            if *block_id == layout.header {
                continue;
            }
            if let Some(block) = func.blocks.get(block_id) {
                if block.predecessors.iter().any(|p| !in_loop.contains(p)) {
                    return Err(CompilerError::CodeGen(format!(
                        "OSR helper for {:?}: block {block_id:?} is also reached from outside the loop",
                        layout.header
                    )));
                }
            }
        }

        // One pointer to the frame carrying the live-ins.
        let params: Vec<BasicMetadataTypeEnum> = vec![self
            .context
            .ptr_type(inkwell::AddressSpace::default())
            .into()];
        let fn_ty = match &layout.return_type {
            HirType::Void => self.context.void_type().fn_type(&params, false),
            ty => self.translate_type(ty)?.fn_type(&params, false),
        };

        let name = format!(
            "osr${}${:x}",
            func.name
                .resolve_global()
                .unwrap_or_else(|| format!("{:?}", func.name)),
            layout.site_key()
        );
        let helper = self.module.add_function(&name, fn_ty, None);

        self.current_function = Some(helper);
        self.block_map.clear();
        self.phi_map.clear();
        self.value_map.clear();
        self.type_map.clear();

        // Seed the constants and globals the body refers to, exactly as a
        // normal compile would.
        for (value_id, value) in &func.values {
            self.type_map.insert(*value_id, value.ty.clone());
            match &value.kind {
                HirValueKind::Constant(constant) => {
                    let c = self.compile_constant(constant)?;
                    self.value_map.insert(*value_id, c);
                }
                HirValueKind::Undef | HirValueKind::Instruction => {
                    let llvm_type = self.translate_type(&value.ty)?;
                    self.value_map.insert(*value_id, llvm_type.const_zero());
                }
                HirValueKind::Global(global_id) => {
                    if let Some(&g) = self.globals_map.get(global_id) {
                        self.value_map.insert(*value_id, g);
                    }
                }
                _ => {}
            }
        }

        // Everything reachable from the header is materialised — the loop
        // and whatever follows it — so the helper runs to the function's
        // own return. Blocks before the header are not: their values arrive
        // in the frame instead, and recompiling them would redefine those.
        //
        // The prologue is created first so it becomes the entry block.
        let prologue = self.context.append_basic_block(helper, "osr_prologue");
        for (block_id, _) in func.blocks.iter() {
            if !in_loop.contains(block_id) {
                continue;
            }
            let llvm_block = self
                .context
                .append_basic_block(helper, &format!("bb_{block_id:?}"));
            self.block_map.insert(*block_id, llvm_block);
        }

        // Recover each live-in from its argument. The leading `phi_count`
        // become the header's phi inputs; the rest are ordinary values the
        // body reads, so they go straight into the value map.
        self.builder.position_at_end(prologue);
        let frame_ptr = helper
            .get_nth_param(0)
            .ok_or_else(|| CompilerError::CodeGen("OSR helper missing its frame pointer".into()))?
            .into_pointer_value();
        let i8_ty = self.context.i8_type();
        let mut phi_seeds: Vec<(HirId, BasicValueEnum<'ctx>)> = Vec::new();
        let mut phi_seed_slots: Vec<(HirId, inkwell::values::PointerValue<'ctx>, HirType)> =
            Vec::new();
        for (i, hir_id) in layout.live_ins.iter().enumerate() {
            let Some(&offset) = layout.frame.offsets.get(i) else {
                break;
            };
            let hir_ty = &layout.live_in_types[i];
            let target = self.translate_type(hir_ty)?;
            // Byte-offset into the frame.
            let slot = unsafe {
                self.builder
                    .build_in_bounds_gep(
                        i8_ty,
                        frame_ptr,
                        &[self.context.i32_type().const_int(offset as u64, false)],
                        "osr_slot",
                    )
                    .map_err(|e| CompilerError::CodeGen(format!("OSR frame gep: {e}")))?
            };
            if i < layout.phi_count {
                // Deferred: the phi this seeds does not exist yet, and its
                // type is what the load has to match.
                phi_seed_slots.push((*hir_id, slot, hir_ty.clone()));
            } else {
                // A value the backends hold by reference was written as the
                // pointee's bytes, so the slot already is the pointer the
                // body expects.
                let recovered = if crate::osr::is_held_by_reference(hir_ty) {
                    slot.into()
                } else {
                    self.builder
                        .build_load(target, slot, "osr_live_in")
                        .map_err(|e| CompilerError::CodeGen(format!("OSR frame load: {e}")))?
                };
                self.value_map.insert(*hir_id, recovered);
                self.type_map.insert(*hir_id, hir_ty.clone());
            }
        }
        let header_block = *self.block_map.get(&layout.header).ok_or_else(|| {
            CompilerError::CodeGen(format!(
                "OSR header {:?} is not in the function",
                layout.header
            ))
        })?;
        self.builder
            .build_unconditional_branch(header_block)
            .map_err(|e| CompilerError::CodeGen(format!("OSR prologue branch: {e}")))?;

        for (block_id, hir_block) in func.blocks.iter() {
            if !in_loop.contains(block_id) {
                continue;
            }
            if let Some(llvm_block) = self.block_map.get(block_id) {
                self.builder.position_at_end(*llvm_block);
                self.compile_block_with_terminator(block_id, hir_block, func)?;
            }
        }

        // Now that the header's phis exist, seed them from the frame. Their
        // LLVM types are the ones the loads must match, and `translate_type`
        // cannot stand in: the backends disagree over whether a multi-field
        // struct travels as a value or as a pointer, and a single loop can
        // carry both shapes.
        if let Some(terminator) = prologue.get_terminator() {
            self.builder.position_before(&terminator);
        }
        for (hir_id, slot, hir_ty) in &phi_seed_slots {
            let Some(phi_value) = self.phi_map.get(hir_id).copied() else {
                continue;
            };
            let want = phi_value.as_basic_value().get_type();
            // Held by reference means the writer copied the pointee's bytes,
            // so the slot is itself the pointer; load only where the phi
            // wants the value.
            let seed: BasicValueEnum<'ctx> =
                if crate::osr::is_held_by_reference(hir_ty) && want.is_pointer_type() {
                    (*slot).into()
                } else {
                    self.builder
                        .build_load(want, *slot, "osr_live_in")
                        .map_err(|e| CompilerError::CodeGen(format!("OSR frame load: {e}")))?
                };
            phi_seeds.push((*hir_id, seed));
        }

        // Wire the phis. Back-edges carry over unchanged, the header takes
        // the prologue edge in place of its loop-entry edges, and blocks
        // outside the loop stop contributing entirely.
        for (block_id, hir_block) in func.blocks.iter() {
            if !in_loop.contains(block_id) {
                continue;
            }
            for phi in &hir_block.phis {
                let Some(phi_value) = self.phi_map.get(&phi.result).copied() else {
                    continue;
                };
                let is_header = *block_id == layout.header;
                for (value_id, pred_block_id) in &phi.incoming {
                    if !in_loop.contains(pred_block_id) {
                        continue;
                    }
                    let incoming_value = self.get_value(*value_id)?;
                    let incoming_block = self.block_map.get(pred_block_id).ok_or_else(|| {
                        CompilerError::CodeGen(format!(
                            "Phi node references unknown block: {pred_block_id:?}"
                        ))
                    })?;
                    let incoming_value = self.coerce_incoming_for_phi(
                        incoming_value,
                        phi_value.as_basic_value().get_type(),
                        *incoming_block,
                    )?;
                    phi_value.add_incoming(&[(&incoming_value, *incoming_block)]);
                }
                if is_header {
                    if let Some((_, seed)) = phi_seeds.iter().find(|(id, _)| *id == phi.result) {
                        phi_value.add_incoming(&[(seed, prologue)]);
                    }
                }
            }
        }

        self.current_function = None;

        // The precondition above should have caught anything malformed, so
        // a failure here means the emitted body is wrong rather than the
        // loop being unrepresentable. Report it and leave the function in
        // place: removing it would dangle whatever already refers to it.
        if helper.verify(crate::osr::osr_trace_enabled()) {
            Ok(name)
        } else {
            Err(CompilerError::CodeGen(format!(
                "OSR helper for {:?} did not verify",
                layout.header
            )))
        }
    }

    /// Recover a live-in from the i64 slot a back-edge marshalled it into.
    fn reinterpret_from_i64(
        &self,
        raw: BasicValueEnum<'ctx>,
        target: BasicTypeEnum<'ctx>,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        let raw = raw.into_int_value();
        Ok(match target {
            BasicTypeEnum::IntType(t) => {
                if t.get_bit_width() == 64 {
                    raw.into()
                } else {
                    self.builder
                        .build_int_truncate(raw, t, "osr_live_in")
                        .map_err(|e| CompilerError::CodeGen(format!("OSR live-in trunc: {e}")))?
                        .into()
                }
            }
            BasicTypeEnum::FloatType(t) => {
                let bits = if t == self.context.f32_type() {
                    self.builder
                        .build_int_truncate(raw, self.context.i32_type(), "osr_live_in_bits")
                        .map_err(|e| CompilerError::CodeGen(format!("OSR live-in trunc: {e}")))?
                } else {
                    raw
                };
                self.builder
                    .build_bit_cast(bits, t, "osr_live_in")
                    .map_err(|e| CompilerError::CodeGen(format!("OSR live-in bitcast: {e}")))?
            }
            BasicTypeEnum::PointerType(t) => self
                .builder
                .build_int_to_ptr(raw, t, "osr_live_in")
                .map_err(|e| CompilerError::CodeGen(format!("OSR live-in int-to-ptr: {e}")))?
                .into(),
            other => {
                return Err(CompilerError::CodeGen(format!(
                    "OSR live-in type {other:?} does not fit an i64 slot"
                )))
            }
        })
    }

    /// Compile a global variable (including vtables)
    ///
    /// This creates LLVM global variables with appropriate linkage and initializers.
    /// For vtables, the initializer contains an array of function pointers.
    fn compile_global(&mut self, id: HirId, global: &HirGlobal) -> CompilerResult<()> {
        // Create unique name for the global
        let global_name = format!("global__{:?}", id);

        // Handle string constants specially - emit in Haxe String format: [length: i32][utf8_bytes...]
        // This matches the Cranelift backend format so runtime functions work correctly
        if let Some(HirConstant::String(s)) = &global.initializer {
            let actual_string = s.resolve_global().unwrap_or_else(|| {
                log::warn!("Could not resolve InternedString for global, using empty string");
                std::string::String::new()
            });

            // Get UTF-8 bytes
            let bytes = actual_string.as_bytes();
            let length = bytes.len() as i32;

            // Create Haxe String structure: [length: i32][utf8_bytes...]
            // The struct is { i32, [N x i8] }
            let i32_type = self.context.i32_type();
            let byte_array_type = self.context.i8_type().array_type(bytes.len() as u32);
            let haxe_string_type = self
                .context
                .struct_type(&[i32_type.into(), byte_array_type.into()], false);

            // Create the length constant
            let length_const = i32_type.const_int(length as u64, false);

            // Create the byte array constant (no null terminator needed)
            let byte_const = self.context.const_string(bytes, false);

            // Create the struct constant
            let haxe_string_const =
                haxe_string_type.const_named_struct(&[length_const.into(), byte_const.into()]);

            let global_value = self.module.add_global(
                haxe_string_type,
                Some(AddressSpace::default()),
                &global_name,
            );
            global_value.set_linkage(inkwell::module::Linkage::External);
            global_value.set_initializer(&haxe_string_const);

            // Store the pointer to the global (address of the Haxe string struct)
            self.globals_map
                .insert(id, global_value.as_pointer_value().into());
            return Ok(());
        }

        // Translate the global's type to LLVM type
        let llvm_ty = self.translate_type(&global.ty)?;

        // Add global variable to module
        let global_value =
            self.module
                .add_global(llvm_ty, Some(AddressSpace::default()), &global_name);

        // Set linkage (export for now - could be internal for private globals)
        global_value.set_linkage(inkwell::module::Linkage::External);

        // Set initializer based on whether this is a vtable or regular global
        if let Some(HirConstant::VTable(vtable)) = &global.initializer {
            // Emit vtable as array of function pointers
            let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());

            // Create array of function pointers
            let mut func_ptrs = Vec::new();
            for method_entry in &vtable.methods {
                if let Some(func_value) = self.functions.get(&method_entry.function_id) {
                    // Cast function to i8*
                    let func_ptr = func_value.as_global_value().as_pointer_value();
                    func_ptrs.push(func_ptr);
                } else {
                    eprintln!(
                        "WARNING: Vtable method function {:?} not found",
                        method_entry.function_id
                    );
                    // Use null pointer as fallback
                    func_ptrs.push(ptr_type.const_null());
                }
            }

            // Create constant array
            let vtable_array = ptr_type.const_array(&func_ptrs);
            // If the global was declared with a non-array (e.g. `ptr`)
            // type that doesn't match the vtable-array's type, fall
            // back to a zero of the declared type rather than feed
            // the verifier a mismatched initializer.
            if vtable_array.as_basic_value_enum().get_type() == llvm_ty {
                global_value.set_initializer(&vtable_array);
            } else {
                global_value.set_initializer(&llvm_ty.const_zero());
            }
        } else if let Some(initializer) = &global.initializer {
            // Other constants - compile them using compile_constant
            match self.compile_constant(initializer) {
                Ok(const_value) => {
                    // Defensive: if the compiled constant's LLVM type
                    // diverged from the declared global type
                    // (translate_type may re-uniquify named structs,
                    // or the initializer may be a placeholder zero of
                    // the wrong type), fall back to a zero of the
                    // declared type so the verifier accepts the
                    // module.
                    if const_value.get_type() == llvm_ty {
                        global_value.set_initializer(&const_value);
                    } else {
                        global_value.set_initializer(&llvm_ty.const_zero());
                    }
                }
                Err(e) => {
                    eprintln!(
                        "WARNING: Failed to compile global initializer for {:?}: {}",
                        id, e
                    );
                    // Fall back to zero initializer
                    global_value.set_initializer(&llvm_ty.const_zero());
                }
            }
        } else {
            // Regular global without explicit initializer - use zero
            global_value.set_initializer(&llvm_ty.const_zero());
        }

        // Store the global in globals_map so it can be referenced across functions
        self.globals_map
            .insert(id, global_value.as_pointer_value().into());

        Ok(())
    }

    /// Compile a basic block (instructions only, no terminator)
    fn compile_block(&mut self, block: &HirBlock) -> CompilerResult<()> {
        log::debug!(
            "[LLVM] compile_block: {} instructions",
            block.instructions.len()
        );
        for instruction in &block.instructions {
            log::debug!("[LLVM]   inst: {:?}", std::mem::discriminant(instruction));
            self.compile_instruction(instruction)?;
        }
        Ok(())
    }

    /// Compile a basic block with its terminator
    fn compile_block_with_terminator(
        &mut self,
        block_id: &HirId,
        block: &HirBlock,
        function: &HirFunction,
    ) -> CompilerResult<()> {
        log::debug!(
            "[LLVM] compile_block_with_terminator: block={:?}, {} phis, {} instructions",
            block_id,
            block.phis.len(),
            block.instructions.len()
        );

        // Compile phi nodes first
        for phi in &block.phis {
            self.compile_phi(phi, block_id, function)?;
        }

        // Compile instructions
        self.compile_block(block)?;

        // Compile terminator
        self.compile_terminator(&block.terminator)?;

        Ok(())
    }

    /// Compile a phi node
    fn compile_phi(
        &mut self,
        phi: &HirPhi,
        _current_block: &HirId,
        function: &HirFunction,
    ) -> CompilerResult<()> {
        // The phi-declared HIR type can disagree with the actual SSA-producer
        // types when an array-literal binding was annotated `Array<T,N>` but
        // the lowering produces a `Ptr<T>` (the Alloca / malloc result that
        // backs heap-allocated array literals). Trust the producer if all
        // non-self-reference incoming values agree on a single HIR type —
        // that's the type the call sites and stores were emitted against.
        let producer_ty = if phi.incoming.len() > 1 {
            let mut consensus: Option<&HirType> = None;
            let mut agree = true;
            for (value_id, _) in &phi.incoming {
                if *value_id == phi.result {
                    continue;
                }
                if let Some(v) = function.values.get(value_id) {
                    match consensus {
                        None => consensus = Some(&v.ty),
                        Some(prev) if prev == &v.ty => {}
                        _ => {
                            agree = false;
                            break;
                        }
                    }
                }
            }
            if agree {
                consensus.cloned()
            } else {
                None
            }
        } else {
            None
        };

        let chosen = match (&phi.ty, &producer_ty) {
            (HirType::Array(_, _), Some(p @ HirType::Ptr(_))) => p.clone(),
            _ => phi.ty.clone(),
        };
        let llvm_ty = self.translate_type(&chosen)?;

        // Create the phi node
        let phi_value = self.builder.build_phi(llvm_ty, "phi")?;

        // Store the phi value in both maps:
        // - value_map: so other instructions can use it
        // - phi_map: so we can add incoming edges later (Phase 3)
        // - type_map: so downstream ExtractValue / pointer-based loads
        //   can recover the *aggregate* HIR type (the field type they
        //   carry as `ty` is the result type, not the pointee). Without
        //   this, ExtractValue on a ptr-typed phi falls back to using
        //   the field type as the GEP element type and emits an invalid
        //   two-index GEP off a scalar.
        self.value_map
            .insert(phi.result, phi_value.as_basic_value());
        self.phi_map.insert(phi.result, phi_value);
        self.type_map.insert(phi.result, chosen);

        // Note: Incoming edges will be added in Phase 3 of compile_function
        // after all blocks are compiled and all values are available

        Ok(())
    }

    /// Compile a terminator instruction
    fn compile_terminator(&mut self, terminator: &HirTerminator) -> CompilerResult<()> {
        match terminator {
            HirTerminator::Return { values } => {
                if values.is_empty() {
                    self.builder.build_return(None)?;
                } else if values.len() == 1 {
                    let val = self.get_value(values[0])?;
                    // Defensive: if the operand's LLVM type does not match
                    // the function's declared return type, synthesize a
                    // zero/undef of the declared type. This guards against
                    // stub-body terminators that emit a placeholder
                    // constant (typically i32 0) for an unreachable block
                    // while the function's signature was derived from the
                    // trait method's declared return type. The verifier
                    // would otherwise reject `define ptr ... ret i32 0`.
                    let fn_value = self.current_function.expect("No current function");
                    let expected = fn_value.get_type().get_return_type();
                    let needs_fixup = match expected {
                        Some(et) => et != val.get_type(),
                        None => true, // void function but we have a value
                    };
                    if needs_fixup {
                        if let Some(et) = expected {
                            let synth = self.zero_of_basic_type(et);
                            self.builder.build_return(Some(&synth))?;
                        } else {
                            // Void return expected but we have a value — drop it.
                            self.builder.build_return(None)?;
                        }
                    } else {
                        self.builder.build_return(Some(&val))?;
                    }
                } else {
                    // Multiple return values - pack into a struct (tuple)
                    let return_values: Vec<BasicValueEnum> = values
                        .iter()
                        .map(|id| self.get_value(*id))
                        .collect::<CompilerResult<Vec<_>>>()?;

                    // Get the function's return type (should be a struct)
                    let fn_value = self.current_function.expect("No current function");
                    let fn_type = fn_value.get_type();
                    let return_type = fn_type
                        .get_return_type()
                        .expect("Function should have a return type");

                    // Build the struct value
                    let mut tuple_value = return_type.into_struct_type().get_undef();
                    for (i, val) in return_values.iter().enumerate() {
                        tuple_value = self
                            .builder
                            .build_insert_value(
                                tuple_value,
                                *val,
                                i as u32,
                                &format!("tuple_field_{}", i),
                            )?
                            .into_struct_value();
                    }

                    self.builder
                        .build_return(Some(&tuple_value.as_basic_value_enum()))?;
                }
            }

            HirTerminator::Branch { target } => {
                let target_block = self.block_map.get(target).ok_or_else(|| {
                    CompilerError::CodeGen(format!("Branch target block not found: {:?}", target))
                })?;
                self.builder.build_unconditional_branch(*target_block)?;
            }

            HirTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                let cond = self.get_value(*condition)?;
                let true_block = self.block_map.get(true_target).ok_or_else(|| {
                    CompilerError::CodeGen(format!(
                        "True branch target block not found: {:?}",
                        true_target
                    ))
                })?;
                let false_block = self.block_map.get(false_target).ok_or_else(|| {
                    CompilerError::CodeGen(format!(
                        "False branch target block not found: {:?}",
                        false_target
                    ))
                })?;
                self.builder.build_conditional_branch(
                    cond.into_int_value(),
                    *true_block,
                    *false_block,
                )?;
            }

            HirTerminator::Switch {
                value,
                default,
                cases,
            } => {
                let switch_val = self.get_value(*value)?;
                let default_block = self.block_map.get(default).ok_or_else(|| {
                    CompilerError::CodeGen(format!("Switch default block not found: {:?}", default))
                })?;

                // Build switch instruction with all cases at once
                let case_values: Vec<_> = cases
                    .iter()
                    .map(|(const_val, target)| {
                        let target_block = self.block_map.get(target).ok_or_else(|| {
                            CompilerError::CodeGen(format!(
                                "Switch case target block not found: {:?}",
                                target
                            ))
                        })?;

                        // Convert HIR constant to LLVM constant
                        let llvm_const = match const_val {
                            HirConstant::I8(v) => self.context.i8_type().const_int(*v as u64, true),
                            HirConstant::I16(v) => {
                                self.context.i16_type().const_int(*v as u64, true)
                            }
                            HirConstant::I32(v) => {
                                self.context.i32_type().const_int(*v as u64, true)
                            }
                            HirConstant::I64(v) => {
                                self.context.i64_type().const_int(*v as u64, true)
                            }
                            HirConstant::U8(v) => {
                                self.context.i8_type().const_int(*v as u64, false)
                            }
                            HirConstant::U16(v) => {
                                self.context.i16_type().const_int(*v as u64, false)
                            }
                            HirConstant::U32(v) => {
                                self.context.i32_type().const_int(*v as u64, false)
                            }
                            HirConstant::U64(v) => self.context.i64_type().const_int(*v, false),
                            _ => {
                                return Err(CompilerError::CodeGen(
                                    "Switch cases must be integer constants".to_string(),
                                ));
                            }
                        };

                        Ok((llvm_const, *target_block))
                    })
                    .collect::<CompilerResult<Vec<_>>>()?;

                self.builder.build_switch(
                    switch_val.into_int_value(),
                    *default_block,
                    &case_values,
                )?;
            }

            HirTerminator::Unreachable => {
                // For void-returning functions, emit a return instead of unreachable/trap
                // This handles Haxe/other languages where main() returns Void and has no explicit return
                if let Some(func) = self.current_function {
                    let return_type = func.get_type().get_return_type();
                    if return_type.is_none() {
                        // Void function - emit return
                        self.builder.build_return(None)?;
                    } else if let Some(ty) = return_type {
                        // Check if it's an empty struct (our representation of Unit/Void type)
                        if let inkwell::types::BasicTypeEnum::StructType(st) = ty {
                            if st.count_fields() == 0 {
                                // Empty struct (Unit) - emit return with undef value
                                let ret_val = st.get_undef();
                                self.builder.build_return(Some(&ret_val))?;
                            } else {
                                self.builder.build_unreachable()?;
                            }
                        } else {
                            self.builder.build_unreachable()?;
                        }
                    } else {
                        self.builder.build_unreachable()?;
                    }
                } else {
                    self.builder.build_unreachable()?;
                }
            }

            HirTerminator::PatternMatch {
                value,
                patterns,
                default,
            } => {
                // Pattern matching is lowered to a switch for now
                // Extract constant patterns
                let switch_val = self.get_value(*value)?;

                let default_target = default.ok_or_else(|| {
                    CompilerError::CodeGen("PatternMatch requires a default target".to_string())
                })?;

                let default_block = self.block_map.get(&default_target).ok_or_else(|| {
                    CompilerError::CodeGen(format!(
                        "Pattern match default block not found: {:?}",
                        default_target
                    ))
                })?;

                // Build switch with pattern cases
                let pattern_cases: Vec<_> = patterns
                    .iter()
                    .filter_map(|pattern| {
                        if let crate::hir::HirPatternKind::Constant(ref const_val) = pattern.kind {
                            let target_block = self.block_map.get(&pattern.target)?;

                            let llvm_const = match const_val {
                                HirConstant::I8(v) => {
                                    self.context.i8_type().const_int(*v as u64, true)
                                }
                                HirConstant::I16(v) => {
                                    self.context.i16_type().const_int(*v as u64, true)
                                }
                                HirConstant::I32(v) => {
                                    self.context.i32_type().const_int(*v as u64, true)
                                }
                                HirConstant::I64(v) => {
                                    self.context.i64_type().const_int(*v as u64, true)
                                }
                                _ => return None, // Skip non-integer patterns for now
                            };

                            Some((llvm_const, *target_block))
                        } else {
                            None
                        }
                    })
                    .collect();

                self.builder.build_switch(
                    switch_val.into_int_value(),
                    *default_block,
                    &pattern_cases,
                )?;
            }

            _ => {
                return Err(CompilerError::CodeGen(format!(
                    "Terminator not yet implemented: {:?}",
                    terminator
                )));
            }
        }

        Ok(())
    }

    /// Compile a single HIR instruction to LLVM IR
    fn compile_instruction(&mut self, instruction: &HirInstruction) -> CompilerResult<()> {
        match instruction {
            // ========== Arithmetic & Logic ==========
            HirInstruction::Binary {
                result,
                op,
                ty,
                left,
                right,
            } => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                // Element-wise vector arithmetic: LLVM's build_int_*/build_float_*
                // accept vector operands directly (VectorValue is Int/FloatMathValue),
                // but the scalar path's `.into_int_value()` would panic on them.
                let result_val = if let HirType::Vector(elem, _) = ty {
                    self.compile_vector_binary(*op, left_val, right_val, elem)?
                } else {
                    self.compile_binary_op(*op, left_val, right_val)?
                };
                self.value_map.insert(*result, result_val);
            }

            HirInstruction::Unary {
                result,
                op,
                ty: _,
                operand,
            } => {
                let operand_val = self.get_value(*operand)?;
                let result_val = self.compile_unary_op(*op, operand_val)?;
                self.value_map.insert(*result, result_val);
            }

            HirInstruction::Select {
                result,
                ty,
                condition,
                true_val,
                false_val,
            } => {
                let cond = self.get_value(*condition)?;
                let true_v = self.get_value(*true_val)?;
                let false_v = self.get_value(*false_val)?;
                let selected =
                    self.builder
                        .build_select(cond.into_int_value(), true_v, false_v, "select")?;
                self.value_map.insert(*result, selected);
            }

            // ========== Type Conversions ==========
            HirInstruction::Cast {
                result,
                ty,
                op,
                operand,
            } => {
                let operand_val = self.get_value(*operand)?;
                let target_ty = self.translate_type(ty)?;
                let casted = self.compile_cast(*op, operand_val, target_ty)?;
                self.value_map.insert(*result, casted);
            }

            // ========== Function Calls ==========
            HirInstruction::Call {
                result,
                callee,
                args,
                type_args: _,
                const_args: _,
                is_tail,
            } => {
                let result_val = self.compile_call(callee, args, *is_tail, result.is_some())?;
                if let Some(res_id) = result {
                    self.value_map.insert(*res_id, result_val);
                }
            }

            // ========== Memory Operations ==========
            HirInstruction::Load {
                result,
                ty,
                ptr,
                align: _,
                volatile: _,
            } => {
                let addr = self.get_value(*ptr)?;
                let ptr_val = addr.into_pointer_value();
                let llvm_ty = self.translate_type(ty)?;
                let loaded = self.builder.build_load(llvm_ty, ptr_val, "load")?;
                self.value_map.insert(*result, loaded);
            }

            HirInstruction::Store {
                value,
                ptr,
                align: _,
                volatile: _,
            } => {
                let addr = self.get_value(*ptr)?;
                let val = self.get_value(*value)?;
                let ptr_val = addr.into_pointer_value();
                self.builder.build_store(ptr_val, val)?;
            }

            HirInstruction::Alloca {
                result,
                ty,
                count,
                align: _,
            } => {
                let llvm_ty = self.translate_type(ty)?;
                let alloca = if let Some(count_id) = count {
                    // Array allocation
                    let count_val = self.get_value(*count_id)?;
                    self.builder.build_array_alloca(
                        llvm_ty,
                        count_val.into_int_value(),
                        "array_alloca",
                    )?
                } else {
                    // Single value allocation
                    self.builder.build_alloca(llvm_ty, "alloca")?
                };
                self.value_map.insert(*result, alloca.into());
                // Record the type as a pointer to the allocated type
                self.type_map
                    .insert(*result, HirType::Ptr(Box::new(ty.clone())));
            }

            HirInstruction::GetElementPtr {
                result,
                ty,
                ptr,
                indices,
            } => {
                let ptr_val = self.get_value(*ptr)?;
                // `ty` is the HIR result type (typically `Ptr(elem)`). LLVM
                // GEP's first operand is the *element* type being indexed —
                // the type that sets the stride. Unwrap one Ptr layer so
                // `Ptr(Body)` → `Body` (stride = sizeof(Body)) rather than
                // `ptr` (stride = sizeof(ptr) = 8). For HIR shapes where
                // `ty` is not `Ptr(_)` (rare; raw byte-offset GEPs from
                // array-literal lowering use `HirType::U8`), fall through.
                let elem_hir = match ty {
                    HirType::Ptr(inner) => inner.as_ref(),
                    other => other,
                };
                let llvm_ty = self.translate_type(elem_hir)?;

                // Convert all index HIR values to LLVM values
                let index_values: Vec<_> = indices
                    .iter()
                    .map(|idx| self.get_value(*idx).map(|v| v.into_int_value()))
                    .collect::<CompilerResult<Vec<_>>>()?;

                // Use GEP to compute the address
                let gep_result = unsafe {
                    self.builder.build_gep(
                        llvm_ty,
                        ptr_val.into_pointer_value(),
                        &index_values,
                        "gep",
                    )?
                };
                self.value_map.insert(*result, gep_result.into());
            }

            // ========== Aggregate Operations ==========
            HirInstruction::ExtractValue {
                result,
                ty,
                aggregate,
                indices,
            } => {
                let agg_value = self.get_value(*aggregate)?;

                // Extract value from struct or array using chained extraction
                // LLVM's build_extract_value only takes a single index, so we need to
                // apply it iteratively for nested access
                if indices.is_empty() {
                    return Err(CompilerError::CodeGen(
                        "ExtractValue requires at least one index".to_string(),
                    ));
                }

                // Check if aggregate is a pointer (e.g., from Alloca)
                // In this case, we use GEP + Load instead of ExtractValue
                if agg_value.is_pointer_value() {
                    // ty is the result type (the field type), not the aggregate type
                    // We need to get the aggregate type from the HIR value info
                    let result_ty = self.translate_type(ty)?;
                    let ptr = agg_value.into_pointer_value();

                    // For pointer-based struct access, we need the aggregate type.
                    // We can get this from the instruction that produced the pointer.
                    // For now, get it from the type_map
                    let agg_hir_type = self.type_map.get(aggregate).cloned();

                    let pointee_type = if let Some(HirType::Ptr(inner)) = agg_hir_type {
                        self.translate_type(&inner)?
                    } else if let Some(hir_ty) = agg_hir_type {
                        // If it's not a pointer type in HIR but is a pointer value,
                        // the type might already be the struct type
                        self.translate_type(&hir_ty)?
                    } else {
                        // Fallback: try to infer from the result type
                        // This won't work correctly, but provides a fallback
                        result_ty
                    };

                    // Build GEP indices: first index is always 0 to dereference the pointer
                    // then follow with the struct field indices
                    let mut gep_indices: Vec<inkwell::values::IntValue> =
                        vec![self.context.i32_type().const_int(0, false)];
                    for &idx in indices {
                        gep_indices.push(self.context.i32_type().const_int(idx as u64, false));
                    }

                    // GEP to get address of the field
                    let field_ptr = unsafe {
                        self.builder
                            .build_gep(pointee_type, ptr, &gep_indices, "field_ptr")?
                    };

                    // Load the field value using the result type
                    let loaded = self
                        .builder
                        .build_load(result_ty, field_ptr, "field_load")?;
                    self.value_map.insert(*result, loaded);
                } else {
                    // Original behavior: work on value types
                    let mut current_value = agg_value;

                    // Apply each index in sequence for nested extraction
                    for (i, &index) in indices.iter().enumerate() {
                        let is_last = i == indices.len() - 1;
                        let name = if is_last {
                            "extract"
                        } else {
                            &format!("extract_nested_{}", i)
                        };

                        // Try to extract from struct
                        if let Ok(struct_val) =
                            TryInto::<inkwell::values::StructValue>::try_into(current_value)
                        {
                            let extracted =
                                self.builder.build_extract_value(struct_val, index, name)?;
                            current_value = extracted.as_basic_value_enum();
                        } else if let Ok(array_val) =
                            TryInto::<inkwell::values::ArrayValue>::try_into(current_value)
                        {
                            // For arrays, we can also use extract_value
                            let extracted =
                                self.builder.build_extract_value(array_val, index, name)?;
                            current_value = extracted.as_basic_value_enum();
                        } else {
                            return Err(CompilerError::CodeGen(format!(
                                "ExtractValue can only be used on struct or array types, got: {:?}",
                                current_value.get_type()
                            )));
                        }
                    }

                    self.value_map.insert(*result, current_value);
                }
            }

            HirInstruction::InsertValue {
                result,
                ty,
                aggregate,
                value,
                indices,
            } => {
                let current_agg = self.get_value(*aggregate)?;
                let val = self.get_value(*value)?;

                if indices.is_empty() {
                    return Err(CompilerError::CodeGen(
                        "InsertValue requires at least one index".to_string(),
                    ));
                }

                // Check if aggregate is a pointer (e.g., from Alloca)
                // In this case, we use GEP + Store instead of InsertValue.
                //
                // `ty` is the HIR result type — typically the aggregate
                // type itself (`Struct(Body)`) for value-type insertvalue
                // emitted by SSA, or `Ptr(Struct(Body))` when the
                // aggregate's annotated type is a pointer (reference-class
                // lowering, or a phi we demoted from `Array<T,N>` to
                // `Ptr(T)`). For the GEP element type we want the *pointee*
                // — `Struct(Body)` — so a `[0, field_idx]` index pair lands
                // on the right field. Unwrap one Ptr layer to match.
                if current_agg.is_pointer_value() {
                    let elem_hir = match ty {
                        HirType::Ptr(inner) => inner.as_ref(),
                        other => other,
                    };
                    let llvm_ty = self.translate_type(elem_hir)?;
                    let ptr = current_agg.into_pointer_value();

                    // Build GEP indices: first index is always 0 to dereference the pointer
                    // then follow with the struct field indices
                    let mut gep_indices: Vec<inkwell::values::IntValue> =
                        vec![self.context.i32_type().const_int(0, false)];
                    for &idx in indices {
                        gep_indices.push(self.context.i32_type().const_int(idx as u64, false));
                    }

                    // GEP to get address of the field
                    let field_ptr = unsafe {
                        self.builder
                            .build_gep(llvm_ty, ptr, &gep_indices, "field_ptr")?
                    };

                    // Store the value
                    self.builder.build_store(field_ptr, val)?;

                    // The result is the original pointer (for chaining)
                    self.value_map.insert(*result, current_agg);
                } else if indices.len() == 1 {
                    // Simple case: single-level insertion on a value
                    let inserted = if let Ok(struct_val) =
                        TryInto::<inkwell::values::StructValue>::try_into(current_agg)
                    {
                        // Defensive coerce: when the operand is a
                        // placeholder integer-zero constant and the
                        // declared field type differs, synthesize a
                        // zero of the field type. This catches stub
                        // bodies that emit `insertvalue { ptr, ptr }
                        // %x, i32 0, 1` (Bug B / Bug C twin). Only
                        // trips on integer-zero operands so real
                        // SSA values (f64, ptr, etc.) feeding
                        // legitimate struct construction are
                        // left alone.
                        let expected = struct_val.get_type().get_field_type_at_index(indices[0]);
                        let is_placeholder_zero = matches!(
                            val,
                            BasicValueEnum::IntValue(iv)
                                if iv.is_const() && iv.get_zero_extended_constant() == Some(0)
                        );
                        let coerced = match expected {
                            Some(et) if et != val.get_type() && is_placeholder_zero => {
                                self.zero_of_basic_type(et)
                            }
                            _ => val,
                        };
                        self.builder
                            .build_insert_value(struct_val, coerced, indices[0], "insert")?
                    } else if let Ok(array_val) =
                        TryInto::<inkwell::values::ArrayValue>::try_into(current_agg)
                    {
                        self.builder
                            .build_insert_value(array_val, val, indices[0], "insert")?
                    } else {
                        return Err(CompilerError::CodeGen(format!(
                            "InsertValue can only be used on struct or array types, got: {:?}",
                            current_agg.get_type()
                        )));
                    };
                    self.value_map
                        .insert(*result, inserted.as_basic_value_enum());
                } else {
                    // Nested insertion: extract nested aggregate, insert value, then insert back
                    let mut extracted_path = Vec::new();

                    // Extract the nested aggregate at all indices except the last
                    let mut nested_agg = current_agg;
                    for &index in &indices[..indices.len() - 1] {
                        if let Ok(struct_val) =
                            TryInto::<inkwell::values::StructValue>::try_into(nested_agg)
                        {
                            let extracted = self.builder.build_extract_value(
                                struct_val,
                                index,
                                "nested_extract_for_insert",
                            )?;
                            extracted_path.push(index);
                            nested_agg = extracted.as_basic_value_enum();
                        } else if let Ok(array_val) =
                            TryInto::<inkwell::values::ArrayValue>::try_into(nested_agg)
                        {
                            let extracted = self.builder.build_extract_value(
                                array_val,
                                index,
                                "nested_extract_for_insert",
                            )?;
                            extracted_path.push(index);
                            nested_agg = extracted.as_basic_value_enum();
                        } else {
                            return Err(CompilerError::CodeGen(format!(
                                "Nested InsertValue path contains non-aggregate type: {:?}",
                                nested_agg.get_type()
                            )));
                        }
                    }

                    // Insert the value at the final index in the nested aggregate
                    let final_index = indices[indices.len() - 1];
                    let modified_nested = if let Ok(struct_val) =
                        TryInto::<inkwell::values::StructValue>::try_into(nested_agg)
                    {
                        self.builder.build_insert_value(
                            struct_val,
                            val,
                            final_index,
                            "nested_insert",
                        )?
                    } else if let Ok(array_val) =
                        TryInto::<inkwell::values::ArrayValue>::try_into(nested_agg)
                    {
                        self.builder.build_insert_value(
                            array_val,
                            val,
                            final_index,
                            "nested_insert",
                        )?
                    } else {
                        return Err(CompilerError::CodeGen(format!(
                            "Cannot insert into non-aggregate type: {:?}",
                            nested_agg.get_type()
                        )));
                    };

                    // Now insert the modified nested aggregate back into the original
                    // We need to work backwards through the path, inserting at each level
                    let mut current_value = modified_nested.as_basic_value_enum();

                    // Start with the original aggregate
                    let mut result_agg = current_agg;

                    // Re-extract and rebuild the path, inserting the modified value
                    // This is complex, so for now we'll use a simpler approach:
                    // Build the insertion from the innermost to outermost
                    for (depth, &index) in extracted_path.iter().enumerate().rev() {
                        // Extract up to this depth
                        let mut temp_agg = current_agg;
                        for &idx in &extracted_path[..depth] {
                            if let Ok(struct_val) =
                                TryInto::<inkwell::values::StructValue>::try_into(temp_agg)
                            {
                                let extracted = self.builder.build_extract_value(
                                    struct_val,
                                    idx,
                                    "rebuild_extract",
                                )?;
                                temp_agg = extracted.as_basic_value_enum();
                            } else if let Ok(array_val) =
                                TryInto::<inkwell::values::ArrayValue>::try_into(temp_agg)
                            {
                                let extracted = self.builder.build_extract_value(
                                    array_val,
                                    idx,
                                    "rebuild_extract",
                                )?;
                                temp_agg = extracted.as_basic_value_enum();
                            }
                        }

                        // Insert the current value at this index
                        current_value = if let Ok(struct_val) =
                            TryInto::<inkwell::values::StructValue>::try_into(temp_agg)
                        {
                            self.builder
                                .build_insert_value(
                                    struct_val,
                                    current_value,
                                    index,
                                    &format!("rebuild_insert_{}", depth),
                                )?
                                .as_basic_value_enum()
                        } else if let Ok(array_val) =
                            TryInto::<inkwell::values::ArrayValue>::try_into(temp_agg)
                        {
                            self.builder
                                .build_insert_value(
                                    array_val,
                                    current_value,
                                    index,
                                    &format!("rebuild_insert_{}", depth),
                                )?
                                .as_basic_value_enum()
                        } else {
                            return Err(CompilerError::CodeGen(
                                "InsertValue rebuild failed: non-aggregate type".to_string(),
                            ));
                        };
                    }

                    self.value_map.insert(*result, current_value);
                }
            }

            // ========== Atomic Operations ==========
            HirInstruction::Atomic {
                op,
                result,
                ty,
                ptr,
                value,
                ordering,
            } => {
                // Atomic operations with proper LLVM support
                let ptr_val = self.get_value(*ptr)?.into_pointer_value();
                let llvm_ty = self.translate_type(ty)?;

                // Convert HIR atomic ordering to LLVM atomic ordering
                let llvm_ordering = match ordering {
                    crate::hir::AtomicOrdering::Relaxed => LLVMAtomicOrdering::Monotonic,
                    crate::hir::AtomicOrdering::Acquire => LLVMAtomicOrdering::Acquire,
                    crate::hir::AtomicOrdering::Release => LLVMAtomicOrdering::Release,
                    crate::hir::AtomicOrdering::AcqRel => LLVMAtomicOrdering::AcquireRelease,
                    crate::hir::AtomicOrdering::SeqCst => {
                        LLVMAtomicOrdering::SequentiallyConsistent
                    }
                };

                let atomic_result = match op {
                    crate::hir::AtomicOp::Load => {
                        // Atomic load - LLVM uses volatile load with ordering
                        // For proper atomics, we need the int type for load
                        if let BasicTypeEnum::IntType(int_ty) = llvm_ty {
                            self.builder.build_load(int_ty, ptr_val, "atomic_load")?
                        } else {
                            return Err(CompilerError::CodeGen(
                                "Atomic load requires integer type".to_string(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::Store => {
                        // Atomic store
                        if let Some(val_id) = value {
                            let val = self.get_value(*val_id)?;
                            self.builder.build_store(ptr_val, val)?;
                            // Store doesn't return a meaningful value, return the stored value
                            val
                        } else {
                            return Err(CompilerError::CodeGen(
                                "AtomicStore requires a value".to_string(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::Exchange => {
                        // Atomic exchange using LLVM's atomicrmw xchg
                        if let Some(val_id) = value {
                            let val = self.get_value(*val_id)?.into_int_value();
                            self.builder
                                .build_atomicrmw(AtomicRMWBinOp::Xchg, ptr_val, val, llvm_ordering)?
                                .into()
                        } else {
                            return Err(CompilerError::CodeGen(
                                "AtomicExchange requires a value".to_string(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::Add => {
                        // Atomic add
                        if let Some(val_id) = value {
                            let val = self.get_value(*val_id)?.into_int_value();
                            self.builder
                                .build_atomicrmw(AtomicRMWBinOp::Add, ptr_val, val, llvm_ordering)?
                                .into()
                        } else {
                            return Err(CompilerError::CodeGen(
                                "AtomicAdd requires a value".to_string(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::Sub => {
                        // Atomic sub
                        if let Some(val_id) = value {
                            let val = self.get_value(*val_id)?.into_int_value();
                            self.builder
                                .build_atomicrmw(AtomicRMWBinOp::Sub, ptr_val, val, llvm_ordering)?
                                .into()
                        } else {
                            return Err(CompilerError::CodeGen(
                                "AtomicSub requires a value".to_string(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::And => {
                        // Atomic and
                        if let Some(val_id) = value {
                            let val = self.get_value(*val_id)?.into_int_value();
                            self.builder
                                .build_atomicrmw(AtomicRMWBinOp::And, ptr_val, val, llvm_ordering)?
                                .into()
                        } else {
                            return Err(CompilerError::CodeGen(
                                "AtomicAnd requires a value".to_string(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::Or => {
                        // Atomic or
                        if let Some(val_id) = value {
                            let val = self.get_value(*val_id)?.into_int_value();
                            self.builder
                                .build_atomicrmw(AtomicRMWBinOp::Or, ptr_val, val, llvm_ordering)?
                                .into()
                        } else {
                            return Err(CompilerError::CodeGen(
                                "AtomicOr requires a value".to_string(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::Xor => {
                        // Atomic xor
                        if let Some(val_id) = value {
                            let val = self.get_value(*val_id)?.into_int_value();
                            self.builder
                                .build_atomicrmw(AtomicRMWBinOp::Xor, ptr_val, val, llvm_ordering)?
                                .into()
                        } else {
                            return Err(CompilerError::CodeGen(
                                "AtomicXor requires a value".to_string(),
                            ));
                        }
                    }
                    crate::hir::AtomicOp::CompareExchange => {
                        // NOTE: CompareExchange requires two values (expected, desired)
                        // Current HIR has single value field - architecture limitation
                        // FUTURE: Extend HIR instruction for compare-exchange
                        return Err(CompilerError::CodeGen(
                            "CompareExchange not yet implemented - requires HIR extension"
                                .to_string(),
                        ));
                    }
                };

                self.value_map.insert(*result, atomic_result);
            }

            HirInstruction::Fence { ordering } => {
                // Memory fence instruction
                let llvm_ordering = match ordering {
                    crate::hir::AtomicOrdering::Relaxed => LLVMAtomicOrdering::Monotonic,
                    crate::hir::AtomicOrdering::Acquire => LLVMAtomicOrdering::Acquire,
                    crate::hir::AtomicOrdering::Release => LLVMAtomicOrdering::Release,
                    crate::hir::AtomicOrdering::AcqRel => LLVMAtomicOrdering::AcquireRelease,
                    crate::hir::AtomicOrdering::SeqCst => {
                        LLVMAtomicOrdering::SequentiallyConsistent
                    }
                };

                // Build fence instruction
                self.builder.build_fence(llvm_ordering, 0, "fence")?;
            }

            // ========== Union Type Operations ==========
            HirInstruction::CreateUnion {
                result,
                union_ty: _,
                variant_index,
                value,
            } => {
                // Create a tagged union value
                // Union layout: 16 bytes (4 bytes discriminant + 12 bytes data)
                // This matches Cranelift's implementation for backend parity

                // Create union type: struct { i32 discriminant, [12 x i8] data }
                let i32_type = self.context.i32_type();
                let data_array_type = self.context.i8_type().array_type(12);
                let union_type = self
                    .context
                    .struct_type(&[i32_type.into(), data_array_type.into()], false);

                // Allocate space for the union on the stack
                let union_alloca = self.builder.build_alloca(union_type, "union")?;

                // Store the discriminant at offset 0
                let discriminant = self
                    .context
                    .i32_type()
                    .const_int(*variant_index as u64, false);
                let discriminant_ptr = self.builder.build_struct_gep(
                    union_type,
                    union_alloca,
                    0,
                    "union_discriminant_ptr",
                )?;
                self.builder.build_store(discriminant_ptr, discriminant)?;

                // Store the value at offset 4 (in the data field)
                // We need to bitcast the data field pointer to the value's type
                let value_val = self.get_value(*value)?;
                let data_ptr =
                    self.builder
                        .build_struct_gep(union_type, union_alloca, 1, "union_data_ptr")?;

                // Cast data pointer to the value's type pointer and store
                let value_type = value_val.get_type();
                let typed_data_ptr = self.builder.build_pointer_cast(
                    data_ptr,
                    value_type.ptr_type(AddressSpace::default()),
                    "typed_data_ptr",
                )?;
                self.builder.build_store(typed_data_ptr, value_val)?;

                // Store the union pointer as the result
                self.value_map.insert(*result, union_alloca.into());
            }

            HirInstruction::GetUnionDiscriminant { result, union_val } => {
                // Extract discriminant from union (at offset 0)
                let union_ptr = self.get_value(*union_val)?.into_pointer_value();

                // Union type for GEP
                let i32_type = self.context.i32_type();
                let data_array_type = self.context.i8_type().array_type(12);
                let union_type = self
                    .context
                    .struct_type(&[i32_type.into(), data_array_type.into()], false);

                // Get pointer to discriminant field
                let discriminant_ptr = self.builder.build_struct_gep(
                    union_type,
                    union_ptr,
                    0,
                    "union_discriminant_ptr",
                )?;

                // Load the discriminant value
                let discriminant =
                    self.builder
                        .build_load(i32_type, discriminant_ptr, "union_discriminant")?;

                self.value_map.insert(*result, discriminant);
            }

            HirInstruction::ExtractUnionValue {
                result,
                ty,
                union_val,
                variant_index: _,
            } => {
                // Extract value from union variant (unsafe - assumes correct variant)
                let union_ptr = self.get_value(*union_val)?.into_pointer_value();

                // Union type for GEP
                let i32_type = self.context.i32_type();
                let data_array_type = self.context.i8_type().array_type(12);
                let union_type = self
                    .context
                    .struct_type(&[i32_type.into(), data_array_type.into()], false);

                // Get pointer to data field (offset 4, after discriminant)
                let data_ptr =
                    self.builder
                        .build_struct_gep(union_type, union_ptr, 1, "union_data_ptr")?;

                // Translate the target type
                let llvm_ty = self.translate_type(ty)?;

                // Cast data pointer to the expected type and load
                let typed_data_ptr = self.builder.build_pointer_cast(
                    data_ptr,
                    llvm_ty.ptr_type(AddressSpace::default()),
                    "typed_data_ptr",
                )?;

                let value = self
                    .builder
                    .build_load(llvm_ty, typed_data_ptr, "union_value")?;

                self.value_map.insert(*result, value);
            }

            // ========== Closure Operations ==========
            HirInstruction::CreateClosure {
                result,
                closure_ty: _,
                function,
                captures,
            } => {
                // Create a closure with function pointer and captured environment
                // Closure layout: { fn_ptr: *(), captures... }
                // Simplified: 8 bytes for fn_ptr + 8 bytes per capture

                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let i64_type = self.context.i64_type();

                // Create closure type as array of i64s for simplicity
                let num_slots = 1 + captures.len() as u32; // fn_ptr + captures
                let closure_type = i64_type.array_type(num_slots);

                // Allocate closure on stack
                let closure_alloca = self.builder.build_alloca(closure_type, "closure")?;

                // Store function pointer at offset 0
                // Try to get function from functions map first
                if let Some(&llvm_func) = self.functions.get(function) {
                    // Get function pointer
                    let func_ptr = llvm_func.as_global_value().as_pointer_value();
                    let func_ptr_as_i64 =
                        self.builder
                            .build_ptr_to_int(func_ptr, i64_type, "fn_ptr_int")?;

                    // Store at slot 0 using GEP
                    let fn_slot_ptr = unsafe {
                        self.builder.build_in_bounds_gep(
                            closure_type,
                            closure_alloca,
                            &[
                                self.context.i32_type().const_zero(),
                                self.context.i32_type().const_zero(),
                            ],
                            "fn_slot",
                        )?
                    };
                    self.builder.build_store(fn_slot_ptr, func_ptr_as_i64)?;
                } else {
                    // Function not found - store null
                    let null_i64 = i64_type.const_zero();
                    let fn_slot_ptr = unsafe {
                        self.builder.build_in_bounds_gep(
                            closure_type,
                            closure_alloca,
                            &[
                                self.context.i32_type().const_zero(),
                                self.context.i32_type().const_zero(),
                            ],
                            "fn_slot",
                        )?
                    };
                    self.builder.build_store(fn_slot_ptr, null_i64)?;
                }

                // Store captured values
                for (i, capture_id) in captures.iter().enumerate() {
                    if let Ok(capture_val) = self.get_value(*capture_id) {
                        // Convert to i64 for storage (simplified - assumes 64-bit values)
                        let capture_as_i64 = if capture_val.is_int_value() {
                            let int_val = capture_val.into_int_value();
                            if int_val.get_type().get_bit_width() < 64 {
                                self.builder
                                    .build_int_z_extend(int_val, i64_type, "capture_ext")?
                            } else {
                                int_val
                            }
                        } else if capture_val.is_pointer_value() {
                            self.builder.build_ptr_to_int(
                                capture_val.into_pointer_value(),
                                i64_type,
                                "capture_ptr_int",
                            )?
                        } else {
                            // For other types, store as bitcast (simplified)
                            i64_type.const_zero()
                        };

                        // Store at slot i+1 (after function pointer)
                        let capture_slot_ptr = unsafe {
                            self.builder.build_in_bounds_gep(
                                closure_type,
                                closure_alloca,
                                &[
                                    self.context.i32_type().const_zero(),
                                    self.context.i32_type().const_int((i + 1) as u64, false),
                                ],
                                &format!("capture_slot_{}", i),
                            )?
                        };
                        self.builder.build_store(capture_slot_ptr, capture_as_i64)?;
                    }
                }

                // Return closure pointer
                self.value_map.insert(*result, closure_alloca.into());
            }

            HirInstruction::CallClosure {
                result,
                closure,
                args,
            } => {
                // Call a closure through its function pointer
                // Closure layout: { fn_ptr: *(), captures... }

                let i64_type = self.context.i64_type();
                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());

                // Get closure pointer
                let closure_ptr = self.get_value(*closure)?.into_pointer_value();

                // Load function pointer from closure (at offset 0)
                // Treat closure as array of i64
                let closure_type = i64_type.array_type(1); // Just need first element
                let fn_slot_ptr = unsafe {
                    self.builder.build_in_bounds_gep(
                        closure_type,
                        closure_ptr,
                        &[
                            self.context.i32_type().const_zero(),
                            self.context.i32_type().const_zero(),
                        ],
                        "fn_slot",
                    )?
                };
                let fn_ptr_int = self
                    .builder
                    .build_load(i64_type, fn_slot_ptr, "fn_ptr_int")?;
                let fn_ptr = self.builder.build_int_to_ptr(
                    fn_ptr_int.into_int_value(),
                    ptr_type,
                    "fn_ptr",
                )?;

                // Build argument list: closure pointer (for environment) + actual args
                let mut call_args: Vec<BasicMetadataValueEnum> = vec![closure_ptr.into()];
                for arg_id in args {
                    let arg_val = self.get_value(*arg_id)?;
                    call_args.push(arg_val.into());
                }

                // Create function type for indirect call
                // Signature: (closure_ptr, args...) -> result
                let mut param_types: Vec<BasicMetadataTypeEnum> = vec![ptr_type.into()];
                for _ in args {
                    param_types.push(ptr_type.into()); // Simplified: treat all args as pointers
                }

                // For now, assume i64 return type (simplified)
                let fn_type = i64_type.fn_type(&param_types, false);

                // Perform indirect call
                let call_result = self.builder.build_indirect_call(
                    fn_type,
                    fn_ptr,
                    &call_args,
                    "closure_call",
                )?;

                // Store result if needed
                if let Some(result_id) = result {
                    let return_val = match call_result.try_as_basic_value() {
                        ValueKind::Basic(val) => val,
                        ValueKind::Instruction(_) => i64_type.const_zero().into(),
                    };
                    self.value_map.insert(*result_id, return_val);
                }
            }

            // ========== Trait Objects ==========
            HirInstruction::CreateTraitObject {
                result,
                trait_id,
                data_ptr,
                vtable_id,
            } => {
                // Create trait object as fat pointer: { *data, *vtable }
                // This matches Cranelift's implementation for backend parity

                let data_ptr_val = self.get_value(*data_ptr)?;
                let vtable_ptr_val = self.get_value(*vtable_id)?;

                // Create a struct type for the fat pointer (two pointers)
                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let fat_ptr_type = self
                    .context
                    .struct_type(&[ptr_type.into(), ptr_type.into()], false);

                // Allocate space for fat pointer on stack
                let fat_ptr_alloca = self.builder.build_alloca(fat_ptr_type, "trait_obj")?;

                // Store data pointer at field 0
                let data_field_ptr =
                    self.builder
                        .build_struct_gep(fat_ptr_type, fat_ptr_alloca, 0, "data_field")?;
                self.builder.build_store(data_field_ptr, data_ptr_val)?;

                // Store vtable pointer at field 1
                let vtable_field_ptr = self.builder.build_struct_gep(
                    fat_ptr_type,
                    fat_ptr_alloca,
                    1,
                    "vtable_field",
                )?;
                self.builder.build_store(vtable_field_ptr, vtable_ptr_val)?;

                // Return the fat pointer (as pointer to struct)
                self.value_map.insert(*result, fat_ptr_alloca.into());
            }

            HirInstruction::UpcastTraitObject {
                result,
                sub_trait_object,
                sub_trait_id,
                super_trait_id,
                super_vtable_id,
            } => {
                // Upcast trait object: extract data pointer from sub-trait, combine with super-trait vtable
                // Fat pointer layout: struct { *data, *vtable }

                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let fat_ptr_type = self
                    .context
                    .struct_type(&[ptr_type.into(), ptr_type.into()], false);

                // Step 1: Get sub-trait fat pointer
                let sub_trait_fat_ptr = self.get_value(*sub_trait_object)?.into_pointer_value();

                // Step 2: Extract data pointer from sub-trait object (field 0)
                let data_field_ptr = self.builder.build_struct_gep(
                    fat_ptr_type,
                    sub_trait_fat_ptr,
                    0,
                    "data_field",
                )?;
                let data_ptr = self
                    .builder
                    .build_load(ptr_type, data_field_ptr, "data_ptr")?;

                // Step 3: Get super-trait vtable pointer
                let super_vtable_ptr = self.get_value(*super_vtable_id)?;

                // Step 4: Allocate space for new super-trait fat pointer on stack
                let super_trait_fat_ptr_alloca =
                    self.builder.build_alloca(fat_ptr_type, "super_trait_obj")?;

                // Step 5: Store data pointer at field 0 (same as sub-trait)
                let super_data_field_ptr = self.builder.build_struct_gep(
                    fat_ptr_type,
                    super_trait_fat_ptr_alloca,
                    0,
                    "super_data_field",
                )?;
                self.builder.build_store(super_data_field_ptr, data_ptr)?;

                // Step 6: Store super-trait vtable pointer at field 1
                let super_vtable_field_ptr = self.builder.build_struct_gep(
                    fat_ptr_type,
                    super_trait_fat_ptr_alloca,
                    1,
                    "super_vtable_field",
                )?;
                self.builder
                    .build_store(super_vtable_field_ptr, super_vtable_ptr)?;

                // Return the new fat pointer
                self.value_map
                    .insert(*result, super_trait_fat_ptr_alloca.into());
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
                // Fat pointer layout: struct { *data, *vtable }

                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let fat_ptr_type = self
                    .context
                    .struct_type(&[ptr_type.into(), ptr_type.into()], false);

                // Get fat pointer (trait object)
                let fat_ptr = self.get_value(*trait_object)?.into_pointer_value();

                // Step 1: Load data pointer from fat_ptr.field[0]
                let data_field_ptr =
                    self.builder
                        .build_struct_gep(fat_ptr_type, fat_ptr, 0, "data_field")?;
                let data_ptr = self
                    .builder
                    .build_load(ptr_type, data_field_ptr, "data_ptr")?
                    .into_pointer_value();

                // Step 2: Load vtable pointer from fat_ptr.field[1]
                let vtable_field_ptr =
                    self.builder
                        .build_struct_gep(fat_ptr_type, fat_ptr, 1, "vtable_field")?;
                let vtable_ptr = self
                    .builder
                    .build_load(ptr_type, vtable_field_ptr, "vtable_ptr")?
                    .into_pointer_value();

                // Step 3: Load function pointer from vtable[method_index]
                // Vtable is an array of function pointers
                let method_index_val = self
                    .context
                    .i32_type()
                    .const_int(*method_index as u64, false);
                let func_ptr_ptr = unsafe {
                    self.builder.build_gep(
                        ptr_type,
                        vtable_ptr,
                        &[method_index_val],
                        "func_ptr_ptr",
                    )?
                };
                let func_ptr = self
                    .builder
                    .build_load(ptr_type, func_ptr_ptr, "func_ptr")?
                    .into_pointer_value();

                // Step 4: Build arguments: prepend self (data_ptr) to args
                let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> =
                    vec![data_ptr.into()];
                for arg_id in args {
                    let arg_val = self.get_value(*arg_id)?;
                    call_args.push(arg_val.into());
                }

                // Step 5: Create function type for indirect call
                // First parameter is always self (data pointer)
                let mut param_types: Vec<inkwell::types::BasicMetadataTypeEnum> =
                    vec![ptr_type.into()]; // self

                // Translate actual parameter types from method signature
                for param_ty in &method_sig.params {
                    let llvm_ty = self.translate_type(param_ty)?;
                    param_types.push(llvm_ty.into());
                }

                // Translate return type from method signature
                let return_type = if matches!(method_sig.return_type, HirType::Void) {
                    None
                } else {
                    let ret_ty = self.translate_type(&method_sig.return_type)?;
                    Some(ret_ty)
                };

                let func_type = if let Some(ret_ty) = return_type {
                    ret_ty.fn_type(&param_types, false)
                } else {
                    self.context.void_type().fn_type(&param_types, false)
                };

                // Step 6: Cast function pointer to correct type
                let typed_func_ptr = self.builder.build_pointer_cast(
                    func_ptr,
                    func_type.ptr_type(AddressSpace::default()),
                    "typed_func_ptr",
                )?;

                // Step 7: Perform indirect call
                let call_site = self.builder.build_indirect_call(
                    func_type,
                    typed_func_ptr,
                    &call_args,
                    "trait_method_call",
                )?;

                // Step 8: Get return value if non-void
                if let Some(result_id) = result {
                    if let ValueKind::Basic(return_val) = call_site.try_as_basic_value() {
                        self.value_map.insert(*result_id, return_val);
                    }
                }
            }

            // ========== Algebraic Effects (Phase H) ==========
            //
            // Mirrors `cranelift_backend.rs`'s Tier 1 / Tier 3
            // placeholder implementations so the LLVM backend doesn't
            // diverge from Cranelift on effect semantics. See
            // `effect_codegen.rs` for the design notes; in short:
            //
            //   * `PerformEffect` is lowered as a direct call to the
            //     mangled handler op fn (`{Handler}${op}`), with an
            //     extra i64 Resume<T> sentinel padded onto the args
            //     when the matched impl has `is_resumable = true`.
            //   * `HandleEffect`/`Resume`/`AbortEffect`/`CaptureContinuation`
            //     are no-ops at the LLVM level for now — Tier 3 full
            //     ABI is still placeholder. They produce a dummy
            //     result (i64 0) so downstream uses don't trap.
            HirInstruction::PerformEffect {
                result,
                effect_id,
                op_name,
                args,
                return_ty,
            } => {
                let (handler_fn_id, is_resumable) = match self
                    .effect_handler_index
                    .get(&(*effect_id, *op_name))
                    .copied()
                {
                    Some(entry) => entry,
                    None => {
                        // No handler — stash a dummy result so downstream
                        // uses link. Cranelift traps in this case; we
                        // could too, but the LLVM IR verifier is happier
                        // with a value than an unreachable.
                        if let Some(res_id) = result {
                            self.value_map
                                .insert(*res_id, self.context.i64_type().const_zero().into());
                        }
                        return Ok(());
                    }
                };

                let llvm_fn = self.functions.get(&handler_fn_id).copied().ok_or_else(|| {
                    CompilerError::CodeGen(format!(
                        "PerformEffect: handler fn HirId {:?} not in self.functions",
                        handler_fn_id
                    ))
                })?;

                let mut arg_values: Vec<inkwell::values::BasicMetadataValueEnum> = args
                    .iter()
                    .map(|arg_id| self.get_value(*arg_id).map(|v| v.into()))
                    .collect::<CompilerResult<Vec<_>>>()?;
                if is_resumable {
                    arg_values.push(self.context.i64_type().const_zero().into());
                }

                let call_site = self
                    .builder
                    .build_call(llvm_fn, &arg_values, "perform_effect")?;
                // Mirror the handler-fn's declared cc at the call site
                // so the verifier accepts the IR. The handler-fn is an
                // internal HIR function, so it follows the same cc
                // policy as any other internal direct call.
                if let Some(&cc) = self.func_cc.get(&handler_fn_id) {
                    if cc != 0 {
                        call_site.set_call_convention(cc);
                    }
                }
                if let Some(res_id) = result {
                    if let ValueKind::Basic(ret_val) = call_site.try_as_basic_value() {
                        self.value_map.insert(*res_id, ret_val);
                    } else if matches!(return_ty, HirType::Void) {
                        self.value_map
                            .insert(*res_id, self.context.i64_type().const_zero().into());
                    }
                }
            }

            HirInstruction::HandleEffect {
                result,
                handler_id: _,
                handler_state: _,
                body_block: _,
                continuation_block: _,
                return_ty: _,
            } => {
                // Placeholder: HandleEffect is currently structural
                // information only (the Cranelift backend doesn't yet
                // push/pop a handler-stack frame around the body —
                // see `effect_runtime.rs` for the runtime side).
                // Materialise a dummy result so downstream uses link.
                if let Some(res_id) = result {
                    self.value_map
                        .insert(*res_id, self.context.i64_type().const_zero().into());
                }
            }

            HirInstruction::Resume {
                value,
                continuation: _,
            } => {
                // Placeholder: in the Tier 3 placeholder ABI, the SSA
                // builder has already rewritten `k(v)` inside handler
                // bodies into Call(Symbol("__zyntax_effect_resume"),
                // ...) — so by the time the LLVM backend runs this
                // shape, the rewrite has already happened. The direct
                // `HirInstruction::Resume` arm exists for handcrafted
                // HIR (tests) and the future full ABI work; for now
                // it just plumbs the value through the
                // __zyntax_effect_resume runtime symbol.
                let resume_struct = self.context.i64_type().const_zero();
                let value_arg = self.get_value(*value)?;
                let resume_fn = self
                    .module
                    .get_function("__zyntax_effect_resume")
                    .ok_or_else(|| {
                        CompilerError::CodeGen(
                            "Resume: __zyntax_effect_resume not registered (build runtime via \
                                 register_effect_runtime_symbols)"
                                .to_string(),
                        )
                    })?;
                let _ = self.builder.build_call(
                    resume_fn,
                    &[resume_struct.into(), value_arg.into()],
                    "resume",
                )?;
            }

            HirInstruction::AbortEffect {
                value,
                handler_scope: _,
            } => {
                // Placeholder: drop `value` and continue. The full
                // Tier 3 ABI will route through __zyntax_effect_abort
                // to unwind the caller's state machine.
                let _ = self.get_value(*value)?;
            }

            HirInstruction::CaptureContinuation {
                result,
                resume_ty: _,
            } => {
                // Placeholder: produces a zero i64 sentinel. Real
                // continuation capture is a future Tier 3 milestone.
                self.value_map
                    .insert(*result, self.context.i64_type().const_zero().into());
            }

            // ========== SIMD / Vector ==========
            HirInstruction::VectorLoad {
                result, ty, ptr, ..
            } => {
                let ptr_val = self.get_value(*ptr)?.into_pointer_value();
                let llvm_ty = self.translate_type(ty)?;
                let loaded = self.builder.build_load(llvm_ty, ptr_val, "vload")?;
                self.value_map.insert(*result, loaded);
            }
            HirInstruction::VectorStore { value, ptr, .. } => {
                let val = self.get_value(*value)?;
                let ptr_val = self.get_value(*ptr)?.into_pointer_value();
                self.builder.build_store(ptr_val, val)?;
            }
            HirInstruction::VectorSplat { result, ty, scalar } => {
                let scalar_val = self.get_value(*scalar)?;
                let vec_ty = self.translate_type(ty)?.into_vector_type();
                let lanes = vec_ty.get_size();
                let mut vec = vec_ty.get_undef();
                for i in 0..lanes {
                    let idx = self.context.i32_type().const_int(i as u64, false);
                    vec = self
                        .builder
                        .build_insert_element(vec, scalar_val, idx, "splat")?;
                }
                self.value_map.insert(*result, vec.into());
            }
            HirInstruction::VectorExtractLane {
                result,
                vector,
                lane,
                ..
            } => {
                let vec = self.get_value(*vector)?.into_vector_value();
                let idx = self.context.i32_type().const_int(*lane as u64, false);
                let elem = self.builder.build_extract_element(vec, idx, "vext")?;
                self.value_map.insert(*result, elem);
            }
            HirInstruction::VectorInsertLane {
                result,
                vector,
                scalar,
                lane,
                ..
            } => {
                let vec = self.get_value(*vector)?.into_vector_value();
                let s = self.get_value(*scalar)?;
                let idx = self.context.i32_type().const_int(*lane as u64, false);
                let out = self.builder.build_insert_element(vec, s, idx, "vins")?;
                self.value_map.insert(*result, out.into());
            }
            HirInstruction::VectorHorizontalReduce {
                result,
                ty,
                vector,
                op,
            } => {
                let vec = self.get_value(*vector)?.into_vector_value();
                let vec_ty = vec.get_type();
                let lanes = vec_ty.get_size();
                let out_ty = self.translate_type(ty)?;
                let is_float = matches!(ty, HirType::F32 | HirType::F64);
                if matches!(op, BinaryOp::Add) && !is_float {
                    // llvm.vector.reduce.add.<vec> → the backend lowers to addv.
                    let bits = out_ty.into_int_type().get_bit_width();
                    let name = format!("llvm.vector.reduce.add.v{lanes}i{bits}");
                    let f = self.module.get_function(&name).unwrap_or_else(|| {
                        let ft = out_ty.fn_type(&[vec_ty.into()], false);
                        self.module.add_function(&name, ft, None)
                    });
                    let r = self.call_basic(f, &[vec.into()], "vreduce")?;
                    self.value_map.insert(*result, r);
                } else {
                    // Serial extract + scalar combine.
                    let mut acc = self.builder.build_extract_element(
                        vec,
                        self.context.i32_type().const_zero(),
                        "l0",
                    )?;
                    for i in 1..lanes {
                        let idx = self.context.i32_type().const_int(i as u64, false);
                        let lane = self.builder.build_extract_element(vec, idx, "l")?;
                        acc = match op {
                            BinaryOp::Add => self
                                .builder
                                .build_int_add(acc.into_int_value(), lane.into_int_value(), "r")?
                                .into(),
                            BinaryOp::FAdd => self
                                .builder
                                .build_float_add(
                                    acc.into_float_value(),
                                    lane.into_float_value(),
                                    "r",
                                )?
                                .into(),
                            _ => acc,
                        };
                    }
                    self.value_map.insert(*result, acc);
                }
            }
            HirInstruction::VectorUnaryOp {
                result,
                ty,
                op,
                operand,
            } => {
                let v = self.get_value(*operand)?;
                let vt = self.translate_type(ty)?;
                let iname = match op {
                    VectorUnaryKind::Sqrt => "llvm.sqrt",
                    VectorUnaryKind::Abs => "llvm.fabs",
                    VectorUnaryKind::Ceil => "llvm.ceil",
                    VectorUnaryKind::Floor => "llvm.floor",
                    VectorUnaryKind::Trunc => "llvm.trunc",
                    VectorUnaryKind::Round => "llvm.roundeven",
                    VectorUnaryKind::Neg => "",
                };
                if op == &VectorUnaryKind::Neg {
                    let r = self
                        .builder
                        .build_float_neg(v.into_vector_value(), "vneg")?;
                    self.value_map.insert(*result, r.into());
                } else {
                    let vecty = vt.into_vector_type();
                    let etype = if vecty.get_element_type().into_float_type().get_bit_width() == 32
                    {
                        "f32"
                    } else {
                        "f64"
                    };
                    let name = format!("{iname}.v{}{etype}", vecty.get_size());
                    let f = self.module.get_function(&name).unwrap_or_else(|| {
                        let ft = vt.fn_type(&[vt.into()], false);
                        self.module.add_function(&name, ft, None)
                    });
                    let r = self.call_basic(f, &[v.into()], "vunary")?;
                    self.value_map.insert(*result, r);
                }
            }
            HirInstruction::VectorMinMax {
                result,
                ty,
                op,
                left,
                right,
            } => {
                let l = self.get_value(*left)?;
                let r = self.get_value(*right)?;
                let vt = self.translate_type(ty)?;
                let vecty = vt.into_vector_type();
                let etype = if vecty.get_element_type().into_float_type().get_bit_width() == 32 {
                    "f32"
                } else {
                    "f64"
                };
                let base = match op {
                    VectorMinMaxKind::Min => "llvm.minnum",
                    VectorMinMaxKind::Max => "llvm.maxnum",
                };
                let name = format!("{base}.v{}{etype}", vecty.get_size());
                let f = self.module.get_function(&name).unwrap_or_else(|| {
                    let ft = vt.fn_type(&[vt.into(), vt.into()], false);
                    self.module.add_function(&name, ft, None)
                });
                let out = self.call_basic(f, &[l.into(), r.into()], "vminmax")?;
                self.value_map.insert(*result, out);
            }
            HirInstruction::VectorDot {
                result,
                acc,
                a,
                b,
                rhs_i7,
                rhs_unsigned,
            } => {
                let acc_v = self.get_value(*acc)?;
                let a_v = self.get_value(*a)?;
                let b_v = self.get_value(*b)?;
                #[cfg(not(target_arch = "x86_64"))]
                let _ = rhs_i7;

                // AArch64 → one SDOT/UDOT. x86_64 with AVX-VNNI / AVX512-
                // VNNI → one VPDPBUSD when the op authorizes the unsigned-
                // RHS form (`rhs_i7`/`rhs_unsigned`). Otherwise the
                // portable widening fallback. On a JIT the host feature
                // probe is exact; a portable AOT object would instead gate
                // on the target-machine features.
                #[cfg(target_arch = "aarch64")]
                let dot = {
                    let i32x4 = self.context.i32_type().vec_type(4);
                    let i8x16 = self.context.i8_type().vec_type(16);
                    let name = if *rhs_unsigned {
                        "llvm.aarch64.neon.udot.v4i32.v16i8"
                    } else {
                        "llvm.aarch64.neon.sdot.v4i32.v16i8"
                    };
                    let f = self.module.get_function(name).unwrap_or_else(|| {
                        let ft = i32x4.fn_type(&[i32x4.into(), i8x16.into(), i8x16.into()], false);
                        self.module.add_function(name, ft, None)
                    });
                    self.call_basic(f, &[acc_v.into(), a_v.into(), b_v.into()], "sdot")?
                };
                #[cfg(target_arch = "x86_64")]
                let dot = if (*rhs_i7 || *rhs_unsigned) && self.x86_target_vnni {
                    self.emit_x86_vpdpbusd(acc_v, a_v, b_v)?
                } else {
                    self.emit_portable_dot(acc_v, a_v, b_v, *rhs_unsigned)?
                };
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                let dot = self.emit_portable_dot(acc_v, a_v, b_v, *rhs_unsigned)?;

                self.value_map.insert(*result, dot);
            }
            _ => {
                return Err(CompilerError::CodeGen(format!(
                    "Instruction not yet implemented: {:?}",
                    instruction
                )));
            }
        }
        Ok(())
    }

    /// Portable widening dot-accumulate: sext/zext each `i8` lane to
    /// `i32`, multiply, and sum consecutive groups of four into the four
    /// `i32x4` output lanes, added to `acc`. The fallback for targets
    /// without a fused dot instruction.
    #[cfg(not(target_arch = "aarch64"))]
    fn emit_portable_dot(
        &self,
        acc_v: BasicValueEnum<'ctx>,
        a_v: BasicValueEnum<'ctx>,
        b_v: BasicValueEnum<'ctx>,
        rhs_unsigned: bool,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        let i32ty = self.context.i32_type();
        let av = a_v.into_vector_value();
        let bv = b_v.into_vector_value();
        let mut out = acc_v.into_vector_value();
        for j in 0..4u32 {
            let mut lane = self
                .builder
                .build_extract_element(out, i32ty.const_int(j as u64, false), "acc_j")?
                .into_int_value();
            for k in 0..4u32 {
                let idx = i32ty.const_int((4 * j + k) as u64, false);
                let ai = self
                    .builder
                    .build_extract_element(av, idx, "ai")?
                    .into_int_value();
                let bi = self
                    .builder
                    .build_extract_element(bv, idx, "bi")?
                    .into_int_value();
                let aw = if rhs_unsigned {
                    self.builder.build_int_z_extend(ai, i32ty, "aw")?
                } else {
                    self.builder.build_int_s_extend(ai, i32ty, "aw")?
                };
                let bw = if rhs_unsigned {
                    self.builder.build_int_z_extend(bi, i32ty, "bw")?
                } else {
                    self.builder.build_int_s_extend(bi, i32ty, "bw")?
                };
                let prod = self.builder.build_int_mul(aw, bw, "prod")?;
                lane = self.builder.build_int_add(lane, prod, "acc")?;
            }
            out = self.builder.build_insert_element(
                out,
                lane,
                i32ty.const_int(j as u64, false),
                "outj",
            )?;
        }
        Ok(out.into())
    }

    /// x86 VNNI fused dot-accumulate via `@llvm.x86.avx512.vpdpbusd.128`:
    /// `acc += dot(u8 lanes, i8 lanes)`, grouped 16→4. The intrinsic's
    /// first operand slot is unsigned and the second signed — the op's
    /// `b` (the `rhs_i7`/`rhs_unsigned` operand) goes in the unsigned
    /// slot, `a` in the signed slot. The `i8x16` inputs are bit-cast to
    /// the intrinsic's `i32x4` operand type (a no-op reinterpret). LLVM
    /// selects the VEX (AVX-VNNI) or EVEX (AVX512-VNNI) encoding from the
    /// target-machine features.
    #[cfg(target_arch = "x86_64")]
    fn emit_x86_vpdpbusd(
        &self,
        acc_v: BasicValueEnum<'ctx>,
        a_v: BasicValueEnum<'ctx>,
        b_v: BasicValueEnum<'ctx>,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        let i32x4 = self.context.i32_type().vec_type(4);
        let name = "llvm.x86.avx512.vpdpbusd.128";
        let f = self.module.get_function(name).unwrap_or_else(|| {
            let ft = i32x4.fn_type(&[i32x4.into(), i32x4.into(), i32x4.into()], false);
            self.module.add_function(name, ft, None)
        });
        let a_i32 = self
            .builder
            .build_bit_cast(a_v.into_vector_value(), i32x4, "vnni_a")?;
        let b_i32 = self
            .builder
            .build_bit_cast(b_v.into_vector_value(), i32x4, "vnni_b")?;
        self.call_basic(f, &[acc_v.into(), b_i32.into(), a_i32.into()], "vpdpbusd")
    }

    /// Compile a binary operation
    /// Element-wise SIMD arithmetic — `VectorValue` implements both
    /// `IntMathValue` and `FloatMathValue`, so the `build_int_*` / `build_float_*`
    /// builders take the vectors directly (no per-lane loop).
    fn compile_vector_binary(
        &self,
        op: BinaryOp,
        left: BasicValueEnum<'ctx>,
        right: BasicValueEnum<'ctx>,
        elem: &HirType,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        let l = left.into_vector_value();
        let r = right.into_vector_value();
        let out = if matches!(elem, HirType::F32 | HirType::F64) {
            match op {
                BinaryOp::Add | BinaryOp::FAdd => self.builder.build_float_add(l, r, "vfadd")?,
                BinaryOp::Sub | BinaryOp::FSub => self.builder.build_float_sub(l, r, "vfsub")?,
                BinaryOp::Mul | BinaryOp::FMul => self.builder.build_float_mul(l, r, "vfmul")?,
                BinaryOp::Div | BinaryOp::FDiv => self.builder.build_float_div(l, r, "vfdiv")?,
                other => {
                    return Err(CompilerError::CodeGen(format!(
                        "vector float binop {other:?}"
                    )))
                }
            }
        } else {
            match op {
                BinaryOp::Add => self.builder.build_int_add(l, r, "vadd")?,
                BinaryOp::Sub => self.builder.build_int_sub(l, r, "vsub")?,
                BinaryOp::Mul => self.builder.build_int_mul(l, r, "vmul")?,
                BinaryOp::And => self.builder.build_and(l, r, "vand")?,
                BinaryOp::Or => self.builder.build_or(l, r, "vor")?,
                BinaryOp::Xor => self.builder.build_xor(l, r, "vxor")?,
                other => {
                    return Err(CompilerError::CodeGen(format!(
                        "vector int binop {other:?}"
                    )))
                }
            }
        };
        Ok(out.into())
    }

    /// Coerce a phi incoming to the phi's own type, emitting the cast in the
    /// predecessor so it dominates the edge.
    ///
    /// A phi whose incomings disagree in width is invalid IR. HIR carries the
    /// declared type on the phi, but an incoming reaching it unnarrowed (an
    /// f64 literal feeding an f32 phi) keeps its own. Mismatches other than
    /// scalar float/int width pass through for the verifier to report.
    ///
    /// Uses its own builder rather than moving `self.builder`: the caller may
    /// be parked mid-block, as `compile_osr_helper` is when it sits before the
    /// prologue's terminator to wire phis, and an insert point restored only
    /// to the block would land after that terminator.
    fn coerce_incoming_for_phi(
        &self,
        v: BasicValueEnum<'ctx>,
        want: BasicTypeEnum<'ctx>,
        pred: inkwell::basic_block::BasicBlock<'ctx>,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        if v.get_type() == want {
            return Ok(v);
        }
        let b = self.context.create_builder();
        match pred.get_terminator() {
            Some(t) => b.position_before(&t),
            None => b.position_at_end(pred),
        }
        let out = if want.is_float_type() {
            let target = want.into_float_type();
            if v.is_int_value() {
                b.build_signed_int_to_float(v.into_int_value(), target, "phi.sitofp")?
                    .into()
            } else if v.is_float_value() {
                if target == self.context.f64_type() {
                    b.build_float_ext(v.into_float_value(), target, "phi.fpext")?
                        .into()
                } else {
                    b.build_float_trunc(v.into_float_value(), target, "phi.fptrunc")?
                        .into()
                }
            } else {
                v
            }
        } else if want.is_int_type() && v.is_int_value() {
            let (from, to) = (v.into_int_value().get_type(), want.into_int_type());
            if from.get_bit_width() > to.get_bit_width() {
                b.build_int_truncate(v.into_int_value(), to, "phi.trunc")?
                    .into()
            } else {
                b.build_int_s_extend(v.into_int_value(), to, "phi.sext")?
                    .into()
            }
        } else {
            v
        };
        Ok(out)
    }

    /// Coerce a scalar operand to `target` float (int -> sitofp, f32<->f64 via
    /// ext/trunc, already-`target` untouched).
    fn coerce_to_float(
        &self,
        v: BasicValueEnum<'ctx>,
        target: inkwell::types::FloatType<'ctx>,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        if v.is_int_value() {
            Ok(self
                .builder
                .build_signed_int_to_float(v.into_int_value(), target, "sitofp")?
                .into())
        } else if v.into_float_value().get_type() == target {
            Ok(v)
        } else if target == self.context.f64_type() {
            Ok(self
                .builder
                .build_float_ext(v.into_float_value(), target, "fpext")?
                .into())
        } else {
            Ok(self
                .builder
                .build_float_trunc(v.into_float_value(), target, "fptrunc")?
                .into())
        }
    }

    /// A float op with a mixed (int) or narrower-float operand is invalid LLVM
    /// IR, and `into_float_value()` on an int operand panics. When either
    /// operand is a float, coerce both to the wider float type (f64 over f32).
    /// Non-float ops pass through. Mirrors the Cranelift backend's reconcile.
    fn reconcile_float_binary_operands(
        &self,
        left: BasicValueEnum<'ctx>,
        right: BasicValueEnum<'ctx>,
    ) -> CompilerResult<(BasicValueEnum<'ctx>, BasicValueEnum<'ctx>)> {
        if !(left.is_float_value() || right.is_float_value()) {
            return Ok((left, right));
        }
        let f64_ty = self.context.f64_type();
        let want_f64 = (left.is_float_value() && left.into_float_value().get_type() == f64_ty)
            || (right.is_float_value() && right.into_float_value().get_type() == f64_ty);
        let target = if want_f64 {
            f64_ty
        } else {
            self.context.f32_type()
        };
        Ok((
            self.coerce_to_float(left, target)?,
            self.coerce_to_float(right, target)?,
        ))
    }

    fn compile_binary_op(
        &mut self,
        op: BinaryOp,
        left: BasicValueEnum<'ctx>,
        right: BasicValueEnum<'ctx>,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        use BinaryOp::*;

        // Float op with a mixed or narrower operand: coerce both to one float
        // type so `into_float_value()` never sees an int and the IR is valid.
        let (left, right) = self.reconcile_float_binary_operands(left, right)?;

        let result = match op {
            // Integer arithmetic
            Add => {
                if left.is_int_value() {
                    self.builder
                        .build_int_add(left.into_int_value(), right.into_int_value(), "add")?
                        .into()
                } else {
                    self.builder
                        .build_float_add(left.into_float_value(), right.into_float_value(), "fadd")?
                        .into()
                }
            }
            Sub => {
                if left.is_int_value() {
                    self.builder
                        .build_int_sub(left.into_int_value(), right.into_int_value(), "sub")?
                        .into()
                } else {
                    self.builder
                        .build_float_sub(left.into_float_value(), right.into_float_value(), "fsub")?
                        .into()
                }
            }
            Mul => {
                if left.is_int_value() {
                    self.builder
                        .build_int_mul(left.into_int_value(), right.into_int_value(), "mul")?
                        .into()
                } else {
                    self.builder
                        .build_float_mul(left.into_float_value(), right.into_float_value(), "fmul")?
                        .into()
                }
            }
            Div => {
                if left.is_int_value() {
                    self.builder
                        .build_int_signed_div(left.into_int_value(), right.into_int_value(), "div")?
                        .into()
                } else {
                    self.builder
                        .build_float_div(left.into_float_value(), right.into_float_value(), "fdiv")?
                        .into()
                }
            }
            Rem => {
                if left.is_int_value() {
                    self.builder
                        .build_int_signed_rem(left.into_int_value(), right.into_int_value(), "rem")?
                        .into()
                } else {
                    self.builder
                        .build_float_rem(left.into_float_value(), right.into_float_value(), "frem")?
                        .into()
                }
            }

            // Bitwise operations
            And => self
                .builder
                .build_and(left.into_int_value(), right.into_int_value(), "and")?
                .into(),
            Or => self
                .builder
                .build_or(left.into_int_value(), right.into_int_value(), "or")?
                .into(),
            Xor => self
                .builder
                .build_xor(left.into_int_value(), right.into_int_value(), "xor")?
                .into(),
            Shl => self
                .builder
                .build_left_shift(left.into_int_value(), right.into_int_value(), "shl")?
                .into(),
            Shr => {
                self.builder
                    .build_right_shift(
                        left.into_int_value(),
                        right.into_int_value(),
                        true, // arithmetic shift (sign-extend)
                        "shr",
                    )?
                    .into()
            }

            // Comparison operations
            Eq => {
                if left.is_int_value() {
                    self.builder
                        .build_int_compare(
                            IntPredicate::EQ,
                            left.into_int_value(),
                            right.into_int_value(),
                            "eq",
                        )?
                        .into()
                } else {
                    self.builder
                        .build_float_compare(
                            FloatPredicate::OEQ,
                            left.into_float_value(),
                            right.into_float_value(),
                            "feq",
                        )?
                        .into()
                }
            }
            Ne => {
                if left.is_int_value() {
                    self.builder
                        .build_int_compare(
                            IntPredicate::NE,
                            left.into_int_value(),
                            right.into_int_value(),
                            "ne",
                        )?
                        .into()
                } else {
                    self.builder
                        .build_float_compare(
                            FloatPredicate::ONE,
                            left.into_float_value(),
                            right.into_float_value(),
                            "fne",
                        )?
                        .into()
                }
            }
            Lt => {
                if left.is_int_value() {
                    self.builder
                        .build_int_compare(
                            IntPredicate::SLT,
                            left.into_int_value(),
                            right.into_int_value(),
                            "lt",
                        )?
                        .into()
                } else {
                    self.builder
                        .build_float_compare(
                            FloatPredicate::OLT,
                            left.into_float_value(),
                            right.into_float_value(),
                            "flt",
                        )?
                        .into()
                }
            }
            Le => {
                if left.is_int_value() {
                    self.builder
                        .build_int_compare(
                            IntPredicate::SLE,
                            left.into_int_value(),
                            right.into_int_value(),
                            "le",
                        )?
                        .into()
                } else {
                    self.builder
                        .build_float_compare(
                            FloatPredicate::OLE,
                            left.into_float_value(),
                            right.into_float_value(),
                            "fle",
                        )?
                        .into()
                }
            }
            Gt => {
                if left.is_int_value() {
                    self.builder
                        .build_int_compare(
                            IntPredicate::SGT,
                            left.into_int_value(),
                            right.into_int_value(),
                            "gt",
                        )?
                        .into()
                } else {
                    self.builder
                        .build_float_compare(
                            FloatPredicate::OGT,
                            left.into_float_value(),
                            right.into_float_value(),
                            "fgt",
                        )?
                        .into()
                }
            }
            Ge => {
                if left.is_int_value() {
                    self.builder
                        .build_int_compare(
                            IntPredicate::SGE,
                            left.into_int_value(),
                            right.into_int_value(),
                            "ge",
                        )?
                        .into()
                } else {
                    self.builder
                        .build_float_compare(
                            FloatPredicate::OGE,
                            left.into_float_value(),
                            right.into_float_value(),
                            "fge",
                        )?
                        .into()
                }
            }

            // Explicit floating-point operations (for when type is already known)
            FAdd => self
                .builder
                .build_float_add(left.into_float_value(), right.into_float_value(), "fadd")?
                .into(),
            FSub => self
                .builder
                .build_float_sub(left.into_float_value(), right.into_float_value(), "fsub")?
                .into(),
            FMul => self
                .builder
                .build_float_mul(left.into_float_value(), right.into_float_value(), "fmul")?
                .into(),
            FDiv => self
                .builder
                .build_float_div(left.into_float_value(), right.into_float_value(), "fdiv")?
                .into(),
            FRem => self
                .builder
                .build_float_rem(left.into_float_value(), right.into_float_value(), "frem")?
                .into(),
            FEq => self
                .builder
                .build_float_compare(
                    FloatPredicate::OEQ,
                    left.into_float_value(),
                    right.into_float_value(),
                    "feq",
                )?
                .into(),
            FNe => self
                .builder
                .build_float_compare(
                    FloatPredicate::ONE,
                    left.into_float_value(),
                    right.into_float_value(),
                    "fne",
                )?
                .into(),
            FLt => self
                .builder
                .build_float_compare(
                    FloatPredicate::OLT,
                    left.into_float_value(),
                    right.into_float_value(),
                    "flt",
                )?
                .into(),
            FLe => self
                .builder
                .build_float_compare(
                    FloatPredicate::OLE,
                    left.into_float_value(),
                    right.into_float_value(),
                    "fle",
                )?
                .into(),
            FGt => self
                .builder
                .build_float_compare(
                    FloatPredicate::OGT,
                    left.into_float_value(),
                    right.into_float_value(),
                    "fgt",
                )?
                .into(),
            FGe => self
                .builder
                .build_float_compare(
                    FloatPredicate::OGE,
                    left.into_float_value(),
                    right.into_float_value(),
                    "fge",
                )?
                .into(),
        };

        Ok(result)
    }

    /// Compile a unary operation
    fn compile_unary_op(
        &mut self,
        op: UnaryOp,
        operand: BasicValueEnum<'ctx>,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        use UnaryOp::*;

        let result = match op {
            Neg => {
                if operand.is_int_value() {
                    self.builder
                        .build_int_neg(operand.into_int_value(), "neg")?
                        .into()
                } else {
                    self.builder
                        .build_float_neg(operand.into_float_value(), "fneg")?
                        .into()
                }
            }
            Not => self
                .builder
                .build_not(operand.into_int_value(), "not")?
                .into(),
            FNeg => self
                .builder
                .build_float_neg(operand.into_float_value(), "fneg")?
                .into(),
        };

        Ok(result)
    }

    /// Compile a cast operation
    fn compile_cast(
        &mut self,
        op: CastOp,
        operand: BasicValueEnum<'ctx>,
        target_ty: BasicTypeEnum<'ctx>,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        use CastOp::*;

        let result = match op {
            // Integer truncation (i64 -> i32, etc.)
            Trunc => self
                .builder
                .build_int_truncate(operand.into_int_value(), target_ty.into_int_type(), "trunc")?
                .into(),

            // Zero extension (unsigned: i32 -> i64, etc.)
            ZExt => self
                .builder
                .build_int_z_extend(operand.into_int_value(), target_ty.into_int_type(), "zext")?
                .into(),

            // Sign extension (signed: i32 -> i64, etc.)
            SExt => self
                .builder
                .build_int_s_extend(operand.into_int_value(), target_ty.into_int_type(), "sext")?
                .into(),

            // Float truncation (f64 -> f32)
            FpTrunc => self
                .builder
                .build_float_trunc(
                    operand.into_float_value(),
                    target_ty.into_float_type(),
                    "fptrunc",
                )?
                .into(),

            // Float extension (f32 -> f64)
            FpExt => self
                .builder
                .build_float_ext(
                    operand.into_float_value(),
                    target_ty.into_float_type(),
                    "fpext",
                )?
                .into(),

            // Float to unsigned int
            FpToUi => self
                .builder
                .build_float_to_unsigned_int(
                    operand.into_float_value(),
                    target_ty.into_int_type(),
                    "fptoui",
                )?
                .into(),

            // Float to signed int
            FpToSi => self
                .builder
                .build_float_to_signed_int(
                    operand.into_float_value(),
                    target_ty.into_int_type(),
                    "fptosi",
                )?
                .into(),

            // Unsigned int to float
            UiToFp => self
                .builder
                .build_unsigned_int_to_float(
                    operand.into_int_value(),
                    target_ty.into_float_type(),
                    "uitofp",
                )?
                .into(),

            // Signed int to float
            SiToFp => self
                .builder
                .build_signed_int_to_float(
                    operand.into_int_value(),
                    target_ty.into_float_type(),
                    "sitofp",
                )?
                .into(),

            // Pointer to integer
            PtrToInt => self
                .builder
                .build_ptr_to_int(
                    operand.into_pointer_value(),
                    target_ty.into_int_type(),
                    "ptrtoint",
                )?
                .into(),

            // Integer to pointer.
            //
            // After the List<T>.data layout change (i64 → typed ptr), some
            // HIR Cast{op: IntToPtr} nodes that used to consume an i64
            // value now see a ptr — typically a leftover cast emitted by
            // an opt pass that was canonicalising address arithmetic.
            // When the operand is already a pointer, the cast is a no-op:
            // just return it as the target ptr type (LLVM opaque pointers
            // are typeless so no real conversion is needed).
            IntToPtr => {
                if operand.is_pointer_value() {
                    operand
                } else {
                    self.builder
                        .build_int_to_ptr(
                            operand.into_int_value(),
                            target_ty.into_pointer_type(),
                            "inttoptr",
                        )?
                        .into()
                }
            }

            // Bitcast (reinterpret bits as different type)
            Bitcast => self
                .builder
                .build_bit_cast(operand, target_ty, "bitcast")?
                .into(),
        };

        Ok(result)
    }

    /// Compile a function call
    /// Box the arguments a registered signature marks dynamic.
    ///
    /// A runtime symbol taking dynamic values declares those parameters as
    /// integer handles, so a raw scalar or pointer has to go through the
    /// matching `zyntax_box_*` before it can be passed. Arguments the
    /// signature does not mark travel unchanged.
    fn box_dynamic_args(
        &mut self,
        sig: &crate::zrtl::ZrtlSymbolSig,
        raw: &[BasicValueEnum<'ctx>],
        hir_types: &[Option<HirType>],
    ) -> CompilerResult<Vec<BasicMetadataValueEnum<'ctx>>> {
        raw.iter()
            .enumerate()
            .map(|(i, &arg_val)| {
                if !sig.param_is_dynamic(i) {
                    return Ok(arg_val.into());
                }

                // A value whose dynamic representation the box carries by
                // pointer — a string, an opaque handle — is boxed by its
                // HIR type, not by how this backend happens to hold it: a
                // string travels as an address, and choosing the box from
                // the register type would box the address as an integer.
                if let Some(Some(hir_ty)) = hir_types.get(i) {
                    if crate::zrtl::dynamic_box_uses_direct_pointer(hir_ty) {
                        let (tag, size) =
                            crate::zrtl::dynamic_box_tag_and_size_for_hir_type(hir_ty);
                        return self
                            .build_stack_dynamic_box(arg_val, tag, size)
                            .map(Into::into);
                    }
                }
                let func_name = if arg_val.is_int_value() {
                    let int_ty = arg_val.into_int_value().get_type();
                    if int_ty == self.context.i32_type() {
                        "zyntax_box_i32"
                    } else if int_ty == self.context.i8_type() {
                        "zyntax_box_bool"
                    } else {
                        "zyntax_box_i64"
                    }
                } else if arg_val.is_float_value() {
                    if arg_val.into_float_value().get_type() == self.context.f32_type() {
                        "zyntax_box_f32"
                    } else {
                        "zyntax_box_f64"
                    }
                } else {
                    "zyntax_box_ptr"
                };

                let box_fn_type = self
                    .context
                    .i64_type()
                    .fn_type(&[arg_val.get_type().into()], false);
                let box_fn = self
                    .module
                    .get_function(func_name)
                    .unwrap_or_else(|| self.module.add_function(func_name, box_fn_type, None));

                match self.builder.build_call(box_fn, &[arg_val.into()], "box") {
                    Ok(call_site) => Ok(call_site
                        .try_as_basic_value()
                        .basic()
                        .unwrap_or(arg_val)
                        .into()),
                    Err(_) => Ok(arg_val.into()),
                }
            })
            .collect()
    }

    /// Build the 32-byte dynamic box the runtime reads — tag, payload
    /// size, data word, then a null dropper and display slot — and hand
    /// back its address as the i64 the callee takes. `data` is stored
    /// as-is: for a by-pointer payload the pointer itself is the data.
    fn build_stack_dynamic_box(
        &mut self,
        data: BasicValueEnum<'ctx>,
        tag: u32,
        size: u32,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        let i32t = self.context.i32_type();
        let i64t = self.context.i64_type();
        let box_ty = self.context.struct_type(
            &[
                i32t.into(),
                i32t.into(),
                i64t.into(),
                i64t.into(),
                i64t.into(),
            ],
            false,
        );
        let slot = self
            .builder
            .build_alloca(box_ty, "dynbox")
            .map_err(|e| CompilerError::CodeGen(format!("dynbox alloca: {e}")))?;

        let data_word: BasicValueEnum = match data {
            BasicValueEnum::PointerValue(p) => self
                .builder
                .build_ptr_to_int(p, i64t, "dynbox_data")
                .map_err(|e| CompilerError::CodeGen(format!("dynbox data: {e}")))?
                .into(),
            BasicValueEnum::IntValue(v) => {
                if v.get_type().get_bit_width() == 64 {
                    v.into()
                } else {
                    self.builder
                        .build_int_z_extend(v, i64t, "dynbox_data")
                        .map_err(|e| CompilerError::CodeGen(format!("dynbox data: {e}")))?
                        .into()
                }
            }
            other => other,
        };

        let stores: [(u32, BasicValueEnum); 5] = [
            (0, i32t.const_int(tag as u64, false).into()),
            (1, i32t.const_int(size as u64, false).into()),
            (2, data_word),
            (3, i64t.const_zero().into()),
            (4, i64t.const_zero().into()),
        ];
        for (idx, value) in stores {
            let field = self
                .builder
                .build_struct_gep(box_ty, slot, idx, "dynbox_field")
                .map_err(|e| CompilerError::CodeGen(format!("dynbox gep: {e}")))?;
            self.builder
                .build_store(field, value)
                .map_err(|e| CompilerError::CodeGen(format!("dynbox store: {e}")))?;
        }

        Ok(self
            .builder
            .build_ptr_to_int(slot, i64t, "dynbox_addr")
            .map_err(|e| CompilerError::CodeGen(format!("dynbox addr: {e}")))?
            .into())
    }

    /// `expects_value` is whether the HIR call binds a result. A void
    /// callee still has to yield something for the return type; when no
    /// result was asked for that stand-in is dropped, and when one was it
    /// means the call site and the callee disagree.
    fn compile_call(
        &mut self,
        callee: &HirCallable,
        args: &[HirId],
        is_tail: bool,
        expects_value: bool,
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        match callee {
            HirCallable::Function(func_id) => {
                // Direct function call
                let function = self.functions.get(func_id).ok_or_else(|| {
                    CompilerError::CodeGen(format!("Function not found: {:?}", func_id))
                })?;

                // An extern declared from a runtime signature describes its
                // dynamic parameters as integer handles, so the arguments
                // need the same boxing the symbol path applies. Without it
                // the call passes a raw pointer where the declaration says
                // integer and the module fails to verify.
                let function = *function;
                let sig_info = self
                    .symbol_signatures
                    .get(function.get_name().to_string_lossy().as_ref())
                    .cloned();
                let raw_args: Vec<BasicValueEnum> = args
                    .iter()
                    .map(|arg_id| self.get_value(*arg_id))
                    .collect::<CompilerResult<Vec<_>>>()?;
                let arg_hir_types: Vec<Option<HirType>> = args
                    .iter()
                    .map(|id| self.type_map.get(id).cloned())
                    .collect();
                let boxed: Vec<BasicMetadataValueEnum> = match &sig_info {
                    Some(sig) => self.box_dynamic_args(sig, &raw_args, &arg_hir_types)?,
                    None => raw_args.iter().map(|&v| v.into()).collect(),
                };
                // Reconcile what remains with the declared parameters. A
                // declaration can describe an address as an integer — the
                // two are one register class to the ground tier, so only
                // LLVM sees the difference — and the declared form is what
                // the callee reads. Boxing above already produced i64
                // handles for the parameters the signature marks dynamic,
                // so this touches only the pass-through ones.
                let param_types = function.get_type().get_param_types();
                let mut arg_values: Vec<BasicMetadataValueEnum> = Vec::with_capacity(boxed.len());
                for (i, v) in boxed.into_iter().enumerate() {
                    let coerced: BasicMetadataValueEnum = match (param_types.get(i), v) {
                        (
                            Some(BasicMetadataTypeEnum::IntType(it)),
                            BasicMetadataValueEnum::PointerValue(pv),
                        ) => self
                            .builder
                            .build_ptr_to_int(pv, *it, "call_arg_p2i")?
                            .into(),
                        (
                            Some(BasicMetadataTypeEnum::PointerType(pt)),
                            BasicMetadataValueEnum::IntValue(iv),
                        ) => self
                            .builder
                            .build_int_to_ptr(iv, *pt, "call_arg_i2p")?
                            .into(),
                        (_, other) => other,
                    };
                    arg_values.push(coerced);
                }

                // Build call
                let call_site = self.builder.build_call(function, &arg_values, "call")?;

                // Mirror the callee's declared calling convention at
                // the call site — LLVM's verifier rejects fastcc
                // declarations being invoked with the default C cc.
                if let Some(&cc) = self.func_cc.get(func_id) {
                    if cc != 0 {
                        call_site.set_call_convention(cc);
                    }
                }

                // Tail-call hint. The HIR tco marker restricts
                // `is_tail = true` to self-recursive direct calls, so
                // by the time we get here the call is structurally a
                // candidate for LLVM's sibling-call optimisation. The
                // flag is purely advisory — LLVM ignores it when it
                // can't prove TCO is safe — so it never causes a
                // miscompile, only enables one when applicable.
                if is_tail {
                    call_site.set_tail_call(true);
                }

                // Return value (or void).
                //
                // A bound result on a void callee is not a disagreement
                // worth refusing. A function written without a return
                // type lowers its definition to `-> void`, while its
                // call sites take the `HirType::I64` that `ssa.rs`
                // assumes for an unannotated callee, so the two differ
                // by construction rather than by mistake. Nothing can
                // read the value either, because the callee returns
                // none. Cranelift has always answered this with a
                // stand-in; refusing it here disabled the whole LLVM
                // tier for any program with a void function, silently,
                // and published Cranelift's numbers under LLVM's name.
                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    ValueKind::Instruction(_) => Ok(self.context.i32_type().get_undef().into()),
                }
            }
            HirCallable::Indirect(func_ptr_id) => {
                // Indirect call through function pointer
                let func_ptr_val = self.get_value(*func_ptr_id)?;

                // The function pointer should be a pointer value
                let func_ptr = func_ptr_val.into_pointer_value();

                // Get the HIR type for this function pointer to extract the signature
                let hir_type = self.type_map.get(func_ptr_id).ok_or_else(|| {
                    CompilerError::CodeGen(format!(
                        "Type not found for function pointer: {:?}",
                        func_ptr_id
                    ))
                })?;

                // Extract the function type from the HIR type
                let func_hir_type = match hir_type {
                    HirType::Function(ft) => ft.as_ref(),
                    _ => {
                        return Err(CompilerError::CodeGen(format!(
                            "Expected function type for indirect call, got: {:?}",
                            hir_type
                        )))
                    }
                };

                // Translate the HIR function type to LLVM function type
                let param_types: Result<Vec<BasicMetadataTypeEnum>, _> = func_hir_type
                    .params
                    .iter()
                    .map(|param_ty| self.translate_type(param_ty).map(|t| t.into()))
                    .collect();
                let param_types = param_types?;

                // Handle return type
                let fn_type = if func_hir_type.returns.is_empty() {
                    self.context
                        .void_type()
                        .fn_type(&param_types, func_hir_type.is_variadic)
                } else if func_hir_type.returns.len() == 1 {
                    let ret_ty = self.translate_type(&func_hir_type.returns[0])?;
                    ret_ty.fn_type(&param_types, func_hir_type.is_variadic)
                } else {
                    let ret_types: Result<Vec<BasicTypeEnum>, _> = func_hir_type
                        .returns
                        .iter()
                        .map(|ret_ty| self.translate_type(ret_ty))
                        .collect();
                    let ret_types = ret_types?;
                    let struct_ret = self.context.struct_type(&ret_types, false);
                    struct_ret.fn_type(&param_types, func_hir_type.is_variadic)
                };

                // Compile arguments
                let raw_arg_values: Vec<BasicValueEnum> = args
                    .iter()
                    .map(|arg_id| self.get_value(*arg_id))
                    .collect::<CompilerResult<Vec<_>>>()?;

                // Defensive: coerce each arg to match the callee's declared
                // param type. Generic trait-method stubs sometimes pass an
                // `i64 0` placeholder where the function pointer expects a
                // `ptr` (or vice versa). LLVM 17+ infers the callee
                // signature from `fn_type`, so a mismatch fails verifier.
                let mut arg_values: Vec<BasicMetadataValueEnum> =
                    Vec::with_capacity(raw_arg_values.len());
                for (i, raw) in raw_arg_values.iter().enumerate() {
                    let expected = func_hir_type
                        .params
                        .get(i)
                        .map(|p| self.translate_type(p))
                        .transpose()?;
                    let actual = raw.get_type();
                    let coerced: BasicValueEnum<'ctx> = match expected {
                        Some(et) if et != actual => match (et, actual) {
                            (BasicTypeEnum::PointerType(pt), BasicTypeEnum::IntType(_)) => self
                                .builder
                                .build_int_to_ptr(raw.into_int_value(), pt, "icall_arg_i2p")?
                                .into(),
                            (BasicTypeEnum::IntType(it), BasicTypeEnum::PointerType(_)) => self
                                .builder
                                .build_ptr_to_int(raw.into_pointer_value(), it, "icall_arg_p2i")?
                                .into(),
                            _ => *raw,
                        },
                        _ => *raw,
                    };
                    arg_values.push(coerced.into());
                }

                // Build indirect call
                let call_site = self.builder.build_indirect_call(
                    fn_type,
                    func_ptr,
                    &arg_values,
                    "indirect_call",
                )?;
                if is_tail {
                    call_site.set_tail_call(true);
                }

                // Return value (or void)
                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    // Same reasoning as the direct call above.
                    ValueKind::Instruction(_) => Ok(self.context.i32_type().get_undef().into()),
                }
            }
            HirCallable::Intrinsic(intrinsic) => self.compile_intrinsic(*intrinsic, args),
            HirCallable::Symbol(symbol_name) => {
                // Call external runtime symbol by name (e.g., "$haxe$trace$int")
                // Check if any parameters need auto-boxing based on symbol signature
                let sig_info = self.symbol_signatures.get(symbol_name).cloned();

                // Compile arguments first to infer their types
                let raw_arg_values: Vec<BasicValueEnum> = args
                    .iter()
                    .map(|arg_id| self.get_value(*arg_id))
                    .collect::<CompilerResult<Vec<_>>>()?;

                // Process arguments - box if needed
                let final_arg_values: Vec<BasicMetadataValueEnum> = if let Some(ref sig) = sig_info
                {
                    raw_arg_values
                        .iter()
                        .enumerate()
                        .map(|(i, &arg_val)| {
                            if sig.param_is_dynamic(i) {
                                // This argument needs to be boxed as DynamicBox
                                // Determine which boxing function to call based on type
                                let func_name = if arg_val.is_int_value() {
                                    let int_ty = arg_val.into_int_value().get_type();
                                    if int_ty == self.context.i32_type() {
                                        "zyntax_box_i32"
                                    } else if int_ty == self.context.i64_type() {
                                        "zyntax_box_i64"
                                    } else if int_ty == self.context.i8_type() {
                                        "zyntax_box_bool"
                                    } else {
                                        "zyntax_box_i64"
                                    }
                                } else if arg_val.is_float_value() {
                                    let float_ty = arg_val.into_float_value().get_type();
                                    if float_ty == self.context.f32_type() {
                                        "zyntax_box_f32"
                                    } else {
                                        "zyntax_box_f64"
                                    }
                                } else {
                                    // Pointers and other types
                                    "zyntax_box_ptr"
                                };

                                // Declare and call boxing function
                                let box_fn_type = self
                                    .context
                                    .i64_type()
                                    .fn_type(&[arg_val.get_type().into()], false);
                                let box_fn =
                                    self.module.get_function(func_name).unwrap_or_else(|| {
                                        self.module.add_function(func_name, box_fn_type, None)
                                    });

                                if let Ok(call_site) =
                                    self.builder.build_call(box_fn, &[arg_val.into()], "box")
                                {
                                    call_site
                                        .try_as_basic_value()
                                        .basic()
                                        .unwrap_or(arg_val)
                                        .into()
                                } else {
                                    arg_val.into()
                                }
                            } else {
                                arg_val.into()
                            }
                        })
                        .collect()
                } else {
                    raw_arg_values.iter().map(|&v| v.into()).collect()
                };

                // Infer parameter types from (potentially boxed) argument values
                let param_types: Vec<BasicMetadataTypeEnum> = final_arg_values
                    .iter()
                    .map(|v| match v {
                        BasicMetadataValueEnum::IntValue(i) => i.get_type().into(),
                        BasicMetadataValueEnum::FloatValue(f) => f.get_type().into(),
                        BasicMetadataValueEnum::PointerValue(p) => p.get_type().into(),
                        BasicMetadataValueEnum::ArrayValue(a) => a.get_type().into(),
                        BasicMetadataValueEnum::StructValue(s) => s.get_type().into(),
                        BasicMetadataValueEnum::VectorValue(v) => v.get_type().into(),
                        _ => self.context.i64_type().into(),
                    })
                    .collect();

                // Pick the return type from the registered signature
                // when present. Without this the function declaration
                // defaulted to `void(args)` and the caller — which had
                // already typed the call result based on the SSA value
                // type — would consume an "i32 0" dummy and panic the
                // first time it tried to use the result as an f64 /
                // f32 / bool. Mirrors `type_tag_to_cranelift_type` in
                // the Cranelift backend.
                let returns_void = sig_info
                    .as_ref()
                    .map(|s| matches!(s.return_type.category(), crate::zrtl::TypeCategory::Void))
                    .unwrap_or(true);
                let call_name = if returns_void { "" } else { symbol_name };
                let fn_type = if let Some(ref sig) = sig_info {
                    use crate::zrtl::{PrimitiveSize, TypeCategory};
                    let bits = sig.return_type.type_id();
                    match sig.return_type.category() {
                        TypeCategory::Void => self.context.void_type().fn_type(&param_types, false),
                        TypeCategory::Bool => self.context.bool_type().fn_type(&param_types, false),
                        TypeCategory::Int | TypeCategory::UInt => {
                            if bits == PrimitiveSize::Bits8 as u16 {
                                self.context.i8_type().fn_type(&param_types, false)
                            } else if bits == PrimitiveSize::Bits16 as u16 {
                                self.context.i16_type().fn_type(&param_types, false)
                            } else if bits == PrimitiveSize::Bits32 as u16 {
                                self.context.i32_type().fn_type(&param_types, false)
                            } else {
                                self.context.i64_type().fn_type(&param_types, false)
                            }
                        }
                        TypeCategory::Float => {
                            if bits == PrimitiveSize::Bits32 as u16 {
                                self.context.f32_type().fn_type(&param_types, false)
                            } else {
                                self.context.f64_type().fn_type(&param_types, false)
                            }
                        }
                        // Pointers / opaques / closures: ptr return.
                        _ => self
                            .context
                            .ptr_type(AddressSpace::default())
                            .fn_type(&param_types, false),
                    }
                } else {
                    self.context.void_type().fn_type(&param_types, false)
                };
                let func = self
                    .module
                    .get_function(symbol_name)
                    .unwrap_or_else(|| self.module.add_function(symbol_name, fn_type, None));

                // Build call
                let call_site = self
                    .builder
                    .build_call(func, &final_arg_values, call_name)?;
                if returns_void {
                    Ok(self.context.i32_type().const_zero().into())
                } else {
                    match call_site.try_as_basic_value() {
                        ValueKind::Basic(val) => Ok(val),
                        ValueKind::Instruction(_) => {
                            Ok(self.context.i32_type().const_zero().into())
                        }
                    }
                }
            }
            HirCallable::FuncRef(_) => Err(CompilerError::CodeGen(
                "HirCallable::FuncRef is not callable directly — \
                 use it only as a value (function address); for calls go \
                 through Indirect"
                    .to_string(),
            )),
        }
    }

    /// Compile an intrinsic function call
    fn compile_intrinsic(
        &mut self,
        intrinsic: crate::hir::Intrinsic,
        args: &[HirId],
    ) -> CompilerResult<BasicValueEnum<'ctx>> {
        use crate::hir::Intrinsic::*;

        match intrinsic {
            // ========== Memory Management ==========
            Malloc => {
                // malloc(size: usize) -> *mut u8
                if args.len() != 1 {
                    return Err(CompilerError::CodeGen(
                        format!("malloc expects 1 argument, got {}", args.len())
                    ));
                }

                let size = self.get_value(args[0])?;

                // Declare or get malloc function: declare ptr @malloc(i64)
                let malloc_fn = self.module.get_function("zyntax_alloc").unwrap_or_else(|| {
                    let i64_type = self.context.i64_type();
                    let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                    let fn_type = ptr_type.fn_type(&[i64_type.into()], false);
                    self.module.add_function("zyntax_alloc", fn_type, None)
                });

                let call_site = self.builder.build_call(
                    malloc_fn,
                    &[size.into()],
                    "malloc"
                )?;

                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    ValueKind::Instruction(_) => Err(CompilerError::CodeGen(
                        "malloc returned void".to_string()
                    ))
                }
            }

            Free => {
                // free(ptr: *mut u8)
                if args.len() != 1 {
                    return Err(CompilerError::CodeGen(
                        format!("free expects 1 argument, got {}", args.len())
                    ));
                }

                let ptr = self.get_value(args[0])?;

                // Declare or get free function: declare void @free(ptr)
                let free_fn = self.module.get_function("zyntax_free").unwrap_or_else(|| {
                    let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                    let fn_type = self.context.void_type().fn_type(&[ptr_type.into()], false);
                    self.module.add_function("zyntax_free", fn_type, None)
                });

                self.builder.build_call(
                    free_fn,
                    &[ptr.into()],
                    "free"
                )?;

                // free returns void, but we need to return something
                // Return a dummy i8 value (caller should ignore it)
                Ok(self.context.i8_type().const_zero().into())
            }

            Realloc => {
                // realloc(ptr: *mut u8, new_size: usize) -> *mut u8
                if args.len() != 2 {
                    return Err(CompilerError::CodeGen(
                        format!("realloc expects 2 arguments, got {}", args.len())
                    ));
                }

                let ptr = self.get_value(args[0])?;
                let new_size = self.get_value(args[1])?;

                // Declare or get realloc function: declare ptr @realloc(ptr, i64)
                let realloc_fn = self.module.get_function("realloc").unwrap_or_else(|| {
                    let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                    let i64_type = self.context.i64_type();
                    let fn_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
                    self.module.add_function("realloc", fn_type, None)
                });

                let call_site = self.builder.build_call(
                    realloc_fn,
                    &[ptr.into(), new_size.into()],
                    "realloc"
                )?;

                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    ValueKind::Instruction(_) => Err(CompilerError::CodeGen(
                        "realloc returned void".to_string()
                    ))
                }
            }

            // ========== Math Intrinsics ==========
            Sqrt => {
                if args.len() != 1 {
                    return Err(CompilerError::CodeGen(
                        format!("sqrt expects 1 argument, got {}", args.len())
                    ));
                }

                let value = self.get_value(args[0])?;

                // Use LLVM's sqrt intrinsic
                let intrinsic_name = if value.is_float_value() {
                    let float_val = value.into_float_value();
                    if float_val.get_type() == self.context.f32_type() {
                        "llvm.sqrt.f32"
                    } else {
                        "llvm.sqrt.f64"
                    }
                } else {
                    return Err(CompilerError::CodeGen(
                        "sqrt requires float argument".to_string()
                    ));
                };

                let sqrt_fn = self.get_or_declare_intrinsic(intrinsic_name, value.get_type())?;
                let call_site = self.builder.build_call(sqrt_fn, &[value.into()], "sqrt")?;

                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    ValueKind::Instruction(_) => Err(CompilerError::CodeGen("sqrt returned void".to_string()))
                }
            }

            Rsqrt => {
                // Reciprocal square root: `1.0 / sqrt(x)`. Lowered via
                // `llvm.sqrt.{f32,f64}` + `fdiv`; LLVM's instruction
                // selector can pattern-match this to `rsqrt`-style
                // hardware (e.g. AArch64 FRSQRTE) under fast-math /
                // unsafe-fp-math, while staying correct under the
                // default IEEE-754 model.
                if args.len() != 1 {
                    return Err(CompilerError::CodeGen(
                        format!("rsqrt expects 1 argument, got {}", args.len())
                    ));
                }

                let value = self.get_value(args[0])?;

                if !value.is_float_value() {
                    return Err(CompilerError::CodeGen(
                        "rsqrt requires float argument".to_string()
                    ));
                }

                let float_val = value.into_float_value();
                let is_f32 = float_val.get_type() == self.context.f32_type();
                let intrinsic_name = if is_f32 {
                    "llvm.sqrt.f32"
                } else {
                    "llvm.sqrt.f64"
                };

                let sqrt_fn = self.get_or_declare_intrinsic(intrinsic_name, value.get_type())?;
                let sqrt_call = self.builder.build_call(sqrt_fn, &[value.into()], "sqrt")?;
                let sqrt_val = match sqrt_call.try_as_basic_value() {
                    ValueKind::Basic(val) => val,
                    ValueKind::Instruction(_) => {
                        return Err(CompilerError::CodeGen(
                            "sqrt returned void".to_string()
                        ));
                    }
                };

                let one = if is_f32 {
                    self.context.f32_type().const_float(1.0)
                } else {
                    self.context.f64_type().const_float(1.0)
                };

                let result = self.builder.build_float_div(
                    one,
                    sqrt_val.into_float_value(),
                    "rsqrt",
                )?;
                Ok(result.into())
            }

            Fabs => {
                if args.len() != 1 {
                    return Err(CompilerError::CodeGen(
                        format!("fabs expects 1 argument, got {}", args.len())
                    ));
                }

                let value = self.get_value(args[0])?;
                let intrinsic_name = if value.is_float_value() {
                    let float_val = value.into_float_value();
                    if float_val.get_type() == self.context.f32_type() {
                        "llvm.fabs.f32"
                    } else {
                        "llvm.fabs.f64"
                    }
                } else {
                    return Err(CompilerError::CodeGen(
                        "fabs requires float argument".to_string()
                    ));
                };

                let fabs_fn = self.get_or_declare_intrinsic(intrinsic_name, value.get_type())?;
                let call_site = self.builder.build_call(fabs_fn, &[value.into()], "fabs")?;

                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    ValueKind::Instruction(_) => Err(CompilerError::CodeGen("fabs returned void".to_string()))
                }
            }

            Fma => {
                // Fused multiply-add: `fma(a, b, c) = a * b + c` with a
                // single IEEE-754 round. Emitted by the `fma_contract`
                // HIR pass.
                if args.len() != 3 {
                    return Err(CompilerError::CodeGen(
                        format!("fma expects 3 arguments, got {}", args.len())
                    ));
                }

                let a = self.get_value(args[0])?;
                let b = self.get_value(args[1])?;
                let c = self.get_value(args[2])?;

                // Polymorphic: scalar float lanes select `llvm.fma.f32/f64`,
                // float-lane vectors select the overloaded vector form
                // (`llvm.fma.v4f32`, `llvm.fma.v2f64`).
                let intrinsic_name = if a.is_float_value() {
                    let float_val = a.into_float_value();
                    if float_val.get_type() == self.context.f32_type() {
                        "llvm.fma.f32".to_string()
                    } else {
                        "llvm.fma.f64".to_string()
                    }
                } else if a.is_vector_value() {
                    let vecty = a.into_vector_value().get_type();
                    let etype = if vecty.get_element_type().into_float_type().get_bit_width() == 32 {
                        "f32"
                    } else {
                        "f64"
                    };
                    format!("llvm.fma.v{}{}", vecty.get_size(), etype)
                } else {
                    return Err(CompilerError::CodeGen(
                        "fma requires float or float-vector arguments".to_string()
                    ));
                };

                let fma_fn = self.get_or_declare_intrinsic_ternary(&intrinsic_name, a.get_type())?;
                let call_site = self.builder.build_call(fma_fn, &[a.into(), b.into(), c.into()], "fma")?;

                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    ValueKind::Instruction(_) => Err(CompilerError::CodeGen("fma returned void".to_string()))
                }
            }

            Sin | Cos | Log | Exp => {
                if args.len() != 1 {
                    return Err(CompilerError::CodeGen(
                        format!("{:?} expects 1 argument, got {}", intrinsic, args.len())
                    ));
                }

                let value = self.get_value(args[0])?;

                let intrinsic_name = if value.is_float_value() {
                    let float_val = value.into_float_value();
                    let suffix = if float_val.get_type() == self.context.f32_type() {
                        "f32"
                    } else {
                        "f64"
                    };

                    match intrinsic {
                        Sin => format!("llvm.sin.{}", suffix),
                        Cos => format!("llvm.cos.{}", suffix),
                        Log => format!("llvm.log.{}", suffix),
                        Exp => format!("llvm.exp.{}", suffix),
                        _ => unreachable!(),
                    }
                } else {
                    return Err(CompilerError::CodeGen(
                        format!("{:?} requires float argument", intrinsic)
                    ));
                };

                let math_fn = self.get_or_declare_intrinsic(&intrinsic_name, value.get_type())?;
                let call_site = self.builder.build_call(math_fn, &[value.into()], &format!("{:?}", intrinsic).to_lowercase())?;

                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    ValueKind::Instruction(_) => Err(CompilerError::CodeGen(format!("{:?} returned void", intrinsic)))
                }
            }

            Pow => {
                if args.len() != 2 {
                    return Err(CompilerError::CodeGen(
                        format!("pow expects 2 arguments, got {}", args.len())
                    ));
                }

                let base = self.get_value(args[0])?;
                let exponent = self.get_value(args[1])?;

                let intrinsic_name = if base.is_float_value() {
                    let float_val = base.into_float_value();
                    if float_val.get_type() == self.context.f32_type() {
                        "llvm.pow.f32"
                    } else {
                        "llvm.pow.f64"
                    }
                } else {
                    return Err(CompilerError::CodeGen(
                        "pow requires float arguments".to_string()
                    ));
                };

                let pow_fn = self.get_or_declare_intrinsic_binary(intrinsic_name, base.get_type())?;
                let call_site = self.builder.build_call(pow_fn, &[base.into(), exponent.into()], "pow")?;

                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    ValueKind::Instruction(_) => Err(CompilerError::CodeGen("pow returned void".to_string()))
                }
            }

            // ========== Memory Operations ==========
            Memcpy => {
                if args.len() != 3 {
                    return Err(CompilerError::CodeGen(
                        format!("memcpy expects 3 arguments (dst, src, len), got {}", args.len())
                    ));
                }

                let dst = self.get_value(args[0])?;
                let src = self.get_value(args[1])?;
                let len = self.get_value(args[2])?;

                // Use LLVM's memcpy intrinsic: llvm.memcpy.p0.p0.i64(ptr dst, ptr src, i64 len, i1 isvolatile)
                let memcpy_fn = self.module.get_function("llvm.memcpy.p0.p0.i64").unwrap_or_else(|| {
                    let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                    let i64_type = self.context.i64_type();
                    let i1_type = self.context.bool_type();
                    let fn_type = self.context.void_type().fn_type(
                        &[ptr_type.into(), ptr_type.into(), i64_type.into(), i1_type.into()],
                        false
                    );
                    self.module.add_function("llvm.memcpy.p0.p0.i64", fn_type, None)
                });

                let is_volatile = self.context.bool_type().const_zero(); // not volatile
                self.builder.build_call(
                    memcpy_fn,
                    &[dst.into(), src.into(), len.into(), is_volatile.into()],
                    "memcpy"
                )?;

                // memcpy returns void, return dummy value
                Ok(self.context.i8_type().const_zero().into())
            }

            Memset => {
                if args.len() != 3 {
                    return Err(CompilerError::CodeGen(
                        format!("memset expects 3 arguments (dst, val, len), got {}", args.len())
                    ));
                }

                let dst = self.get_value(args[0])?;
                let val = self.get_value(args[1])?;
                let len = self.get_value(args[2])?;

                // Use LLVM's memset intrinsic
                let memset_fn = self.module.get_function("llvm.memset.p0.i64").unwrap_or_else(|| {
                    let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                    let i8_type = self.context.i8_type();
                    let i64_type = self.context.i64_type();
                    let i1_type = self.context.bool_type();
                    let fn_type = self.context.void_type().fn_type(
                        &[ptr_type.into(), i8_type.into(), i64_type.into(), i1_type.into()],
                        false
                    );
                    self.module.add_function("llvm.memset.p0.i64", fn_type, None)
                });

                let is_volatile = self.context.bool_type().const_zero();
                self.builder.build_call(
                    memset_fn,
                    &[dst.into(), val.into(), len.into(), is_volatile.into()],
                    "memset"
                )?;

                Ok(self.context.i8_type().const_zero().into())
            }

            Memmove => {
                if args.len() != 3 {
                    return Err(CompilerError::CodeGen(
                        format!("memmove expects 3 arguments (dst, src, len), got {}", args.len())
                    ));
                }

                let dst = self.get_value(args[0])?;
                let src = self.get_value(args[1])?;
                let len = self.get_value(args[2])?;

                let memmove_fn = self.module.get_function("llvm.memmove.p0.p0.i64").unwrap_or_else(|| {
                    let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                    let i64_type = self.context.i64_type();
                    let i1_type = self.context.bool_type();
                    let fn_type = self.context.void_type().fn_type(
                        &[ptr_type.into(), ptr_type.into(), i64_type.into(), i1_type.into()],
                        false
                    );
                    self.module.add_function("llvm.memmove.p0.p0.i64", fn_type, None)
                });

                let is_volatile = self.context.bool_type().const_zero();
                self.builder.build_call(
                    memmove_fn,
                    &[dst.into(), src.into(), len.into(), is_volatile.into()],
                    "memmove"
                )?;

                Ok(self.context.i8_type().const_zero().into())
            }

            // ========== Bit Manipulation ==========
            Ctpop | Ctlz | Cttz | Bswap => {
                if args.len() != 1 {
                    return Err(CompilerError::CodeGen(
                        format!("{:?} expects 1 argument, got {}", intrinsic, args.len())
                    ));
                }

                let value = self.get_value(args[0])?;

                if !value.is_int_value() {
                    return Err(CompilerError::CodeGen(
                        format!("{:?} requires integer argument", intrinsic)
                    ));
                }

                let int_val = value.into_int_value();
                let int_type = int_val.get_type();

                let bit_width = int_type.get_bit_width();
                let intrinsic_name = match intrinsic {
                    Ctpop => format!("llvm.ctpop.i{}", bit_width),
                    Ctlz => format!("llvm.ctlz.i{}", bit_width),
                    Cttz => format!("llvm.cttz.i{}", bit_width),
                    Bswap => format!("llvm.bswap.i{}", bit_width),
                    _ => unreachable!(),
                };

                let bit_fn = if intrinsic == Ctlz || intrinsic == Cttz {
                    // ctlz/cttz need extra i1 parameter (is_zero_undef)
                    self.get_or_declare_intrinsic_with_bool(&intrinsic_name, int_type)?
                } else {
                    self.get_or_declare_intrinsic(&intrinsic_name, int_type.into())?
                };

                let call_args = if intrinsic == Ctlz || intrinsic == Cttz {
                    vec![value.into(), self.context.bool_type().const_zero().into()]
                } else {
                    vec![value.into()]
                };

                let call_site = self.builder.build_call(bit_fn, &call_args, &format!("{:?}", intrinsic).to_lowercase())?;

                match call_site.try_as_basic_value() {
                    ValueKind::Basic(val) => Ok(val),
                    ValueKind::Instruction(_) => Err(CompilerError::CodeGen(format!("{:?} returned void", intrinsic)))
                }
            }

            // ========== Type Queries ==========
            SizeOf | AlignOf => {
                // These should be resolved at compile time, not runtime
                // For now, return error
                Err(CompilerError::CodeGen(
                    format!("{:?} should be resolved at compile time", intrinsic)
                ))
            }

            // ========== Not Yet Implemented ==========
            AddWithOverflow | SubWithOverflow | MulWithOverflow |
            // ========== Error Handling (Gap 8) ==========
            Panic => {
                // Gap 8 Phase 3: Panic with message
                // Calls abort() from libc, which terminates immediately
                // Future: Add message printing, stack unwinding

                // Declare or get abort function: declare void @abort()
                let abort_fn = self.module.get_function("abort").unwrap_or_else(|| {
                    let fn_type = self.context.void_type().fn_type(&[], false);
                    self.module.add_function("abort", fn_type, None)
                });

                // Call abort() - doesn't return
                self.builder.build_call(abort_fn, &[], "panic")?;

                // Add unreachable to satisfy control flow
                self.builder.build_unreachable()?;

                // Return dummy value (unreachable anyway)
                Ok(self.context.i8_type().const_zero().into())
            }

            Abort => {
                // Gap 8 Phase 3: Immediate abort
                // Calls abort() from libc

                let abort_fn = self.module.get_function("abort").unwrap_or_else(|| {
                    let fn_type = self.context.void_type().fn_type(&[], false);
                    self.module.add_function("abort", fn_type, None)
                });

                self.builder.build_call(abort_fn, &[], "abort")?;
                self.builder.build_unreachable()?;

                Ok(self.context.i8_type().const_zero().into())
            }

            Drop | IncRef | DecRef | Alloca | GCSafepoint | Await | Yield => {
                Err(CompilerError::CodeGen(
                    format!("Intrinsic {:?} not yet implemented in LLVM backend", intrinsic)
                ))
            }

            ClosureToZrtl | BoxToZrtl | PrimitiveToBox | TypeTagOf => {
                Err(CompilerError::CodeGen(
                    format!("Intrinsic {:?} not yet implemented in LLVM backend", intrinsic)
                ))
            }
        }
    }

    /// Helper to get or declare a unary LLVM intrinsic
    /// Build a call to `f` and return its basic result value (all our
    /// intrinsics return a value, never void).
    fn call_basic(
        &self,
        f: FunctionValue<'ctx>,
        args: &[inkwell::values::BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> CompilerResult<inkwell::values::BasicValueEnum<'ctx>> {
        match self.builder.build_call(f, args, name)?.try_as_basic_value() {
            ValueKind::Basic(v) => Ok(v),
            ValueKind::Instruction(_) => Err(CompilerError::CodeGen(format!(
                "{name}: expected a value result"
            ))),
        }
    }

    fn get_or_declare_intrinsic(
        &self,
        name: &str,
        arg_type: BasicTypeEnum<'ctx>,
    ) -> CompilerResult<FunctionValue<'ctx>> {
        Ok(self.module.get_function(name).unwrap_or_else(|| {
            let fn_type = arg_type.fn_type(&[arg_type.into()], false);
            self.module.add_function(name, fn_type, None)
        }))
    }

    /// Helper to get or declare a binary LLVM intrinsic
    fn get_or_declare_intrinsic_binary(
        &self,
        name: &str,
        arg_type: BasicTypeEnum<'ctx>,
    ) -> CompilerResult<FunctionValue<'ctx>> {
        Ok(self.module.get_function(name).unwrap_or_else(|| {
            let fn_type = arg_type.fn_type(&[arg_type.into(), arg_type.into()], false);
            self.module.add_function(name, fn_type, None)
        }))
    }

    /// Helper to get or declare a ternary LLVM intrinsic with three
    /// same-typed arguments and a same-typed return — e.g.
    /// `llvm.fma.f64(f64, f64, f64) -> f64`.
    fn get_or_declare_intrinsic_ternary(
        &self,
        name: &str,
        arg_type: BasicTypeEnum<'ctx>,
    ) -> CompilerResult<FunctionValue<'ctx>> {
        Ok(self.module.get_function(name).unwrap_or_else(|| {
            let fn_type =
                arg_type.fn_type(&[arg_type.into(), arg_type.into(), arg_type.into()], false);
            self.module.add_function(name, fn_type, None)
        }))
    }

    /// Helper to get or declare an LLVM intrinsic with bool parameter
    fn get_or_declare_intrinsic_with_bool(
        &self,
        name: &str,
        arg_type: IntType<'ctx>,
    ) -> CompilerResult<FunctionValue<'ctx>> {
        Ok(self.module.get_function(name).unwrap_or_else(|| {
            let fn_type =
                arg_type.fn_type(&[arg_type.into(), self.context.bool_type().into()], false);
            self.module.add_function(name, fn_type, None)
        }))
    }

    /// Compile a constant value
    fn compile_constant(&self, value: &HirConstant) -> CompilerResult<BasicValueEnum<'ctx>> {
        use HirConstant::*;

        let result = match value {
            // Primitive integers (signed)
            I8(v) => self.context.i8_type().const_int(*v as u64, true).into(),
            I16(v) => self.context.i16_type().const_int(*v as u64, true).into(),
            I32(v) => self.context.i32_type().const_int(*v as u64, true).into(),
            I64(v) => self.context.i64_type().const_int(*v as u64, true).into(),
            I128(v) => {
                // Split i128 into high and low u64 parts
                let low = (*v as u128 & 0xFFFFFFFFFFFFFFFF) as u64;
                let high = ((*v as u128 >> 64) & 0xFFFFFFFFFFFFFFFF) as u64;
                self.context
                    .i128_type()
                    .const_int_arbitrary_precision(&[low, high])
                    .into()
            }

            // Primitive integers (unsigned)
            U8(v) => self.context.i8_type().const_int(*v as u64, false).into(),
            U16(v) => self.context.i16_type().const_int(*v as u64, false).into(),
            U32(v) => self.context.i32_type().const_int(*v as u64, false).into(),
            U64(v) => self.context.i64_type().const_int(*v as u64, false).into(),
            U128(v) => {
                // Split u128 into high and low u64 parts
                let low = (*v & 0xFFFFFFFFFFFFFFFF) as u64;
                let high = ((*v >> 64) & 0xFFFFFFFFFFFFFFFF) as u64;
                self.context
                    .i128_type()
                    .const_int_arbitrary_precision(&[low, high])
                    .into()
            }

            // Floating point
            F32(v) => self.context.f32_type().const_float(*v as f64).into(),
            F64(v) => self.context.f64_type().const_float(*v).into(),

            // Boolean
            Bool(v) => self.context.bool_type().const_int(*v as u64, false).into(),

            // Null pointer
            Null(ty) => {
                let llvm_ty = self.translate_type(ty)?;
                if let BasicTypeEnum::PointerType(ptr_ty) = llvm_ty {
                    ptr_ty.const_null().into()
                } else {
                    return Err(CompilerError::CodeGen(format!(
                        "Null constant must have pointer type, got: {:?}",
                        ty
                    )));
                }
            }

            // Array constant
            Array(elements) => {
                if elements.is_empty() {
                    // Empty array - create zero-sized array of i8
                    let arr_ty = self.context.i8_type().array_type(0);
                    arr_ty.const_zero().into()
                } else {
                    // Compile each element
                    let compiled_elements: Vec<BasicValueEnum> = elements
                        .iter()
                        .map(|elem| self.compile_constant(elem))
                        .collect::<CompilerResult<Vec<_>>>()?;

                    // Determine element type from first element
                    let elem_type = compiled_elements[0].get_type();

                    // Create constant array based on element type
                    match elem_type {
                        BasicTypeEnum::IntType(int_ty) => {
                            let int_values: Vec<_> = compiled_elements
                                .iter()
                                .map(|v| v.into_int_value())
                                .collect();
                            int_ty.const_array(&int_values).into()
                        }
                        BasicTypeEnum::FloatType(float_ty) => {
                            let float_values: Vec<_> = compiled_elements
                                .iter()
                                .map(|v| v.into_float_value())
                                .collect();
                            float_ty.const_array(&float_values).into()
                        }
                        BasicTypeEnum::PointerType(ptr_ty) => {
                            let ptr_values: Vec<_> = compiled_elements
                                .iter()
                                .map(|v| v.into_pointer_value())
                                .collect();
                            ptr_ty.const_array(&ptr_values).into()
                        }
                        BasicTypeEnum::StructType(struct_ty) => {
                            let struct_values: Vec<_> = compiled_elements
                                .iter()
                                .map(|v| v.into_struct_value())
                                .collect();
                            struct_ty.const_array(&struct_values).into()
                        }
                        BasicTypeEnum::ArrayType(arr_ty) => {
                            let arr_values: Vec<_> = compiled_elements
                                .iter()
                                .map(|v| v.into_array_value())
                                .collect();
                            arr_ty.const_array(&arr_values).into()
                        }
                        BasicTypeEnum::VectorType(_) | BasicTypeEnum::ScalableVectorType(_) => {
                            return Err(CompilerError::CodeGen(
                                "Vector type arrays not yet supported in constants".to_string(),
                            ));
                        }
                    }
                }
            }

            // Struct constant
            Struct(fields) => {
                if fields.is_empty() {
                    // Empty struct (unit type)
                    let struct_ty = self.context.struct_type(&[], false);
                    struct_ty.const_named_struct(&[]).into()
                } else {
                    // Compile each field
                    let compiled_fields: Vec<BasicValueEnum> = fields
                        .iter()
                        .map(|field| self.compile_constant(field))
                        .collect::<CompilerResult<Vec<_>>>()?;

                    // Create constant struct
                    let struct_ty = self.context.const_struct(&compiled_fields, false);
                    struct_ty.into()
                }
            }

            // String constant
            String(s) => {
                // Resolve the InternedString to get the actual string value
                let actual_string = s.resolve_global().unwrap_or_else(|| {
                    log::warn!("Could not resolve InternedString, using empty string");
                    std::string::String::new()
                });
                let string_value = self.context.const_string(actual_string.as_bytes(), true);
                string_value.into()
            }

            // VTable should not go through compile_constant - handled separately
            VTable(_) => {
                return Err(CompilerError::CodeGen(
                    "VTable constants should be compiled via compile_vtable, not compile_constant"
                        .to_string(),
                ));
            }
        };

        Ok(result)
    }

    /// Translate HIR type to LLVM type
    fn translate_type(&self, ty: &HirType) -> CompilerResult<BasicTypeEnum<'ctx>> {
        use HirType::*;

        let result = match ty {
            Void => {
                // Void/Unit is represented as an empty struct (zero-sized type)
                self.context.struct_type(&[], false).into()
            }
            I8 => self.context.i8_type().into(),
            I16 => self.context.i16_type().into(),
            I32 => self.context.i32_type().into(),
            I64 => self.context.i64_type().into(),
            I128 => self.context.i128_type().into(),
            U8 => self.context.i8_type().into(),
            U16 => self.context.i16_type().into(),
            U32 => self.context.i32_type().into(),
            U64 => self.context.i64_type().into(),
            U128 => self.context.i128_type().into(),
            F32 => self.context.f32_type().into(),
            F64 => self.context.f64_type().into(),
            Bool => self.context.bool_type().into(),
            Ptr(inner) => {
                // `Ptr(Opaque(X))` and bare `Opaque(X)` both collapse to
                // `opaque.X*` — the Opaque arm below already returns a
                // pointer-to-opaque (opaque values have unknown size so
                // they can never be materialised as first-class LLVM
                // values). Avoid the extra `ptr_type()` wrap that would
                // turn `Ptr(Opaque(X))` into `opaque.X**`.
                if matches!(inner.as_ref(), Opaque(_)) {
                    self.translate_type(inner)?
                } else {
                    let inner_ty = self.translate_type(inner)?;
                    inner_ty.ptr_type(AddressSpace::default()).into()
                }
            }
            Ref { pointee, .. } => {
                // References are compiled as pointers
                let inner_ty = self.translate_type(pointee)?;
                inner_ty.ptr_type(AddressSpace::default()).into()
            }
            Array(element_ty, size) => {
                let elem_ty = self.translate_type(element_ty)?;
                elem_ty.array_type(*size as u32).into()
            }
            Struct(struct_ty) => {
                // Create LLVM struct type
                // Use opaque struct if it has a name (for recursive types)
                if let Some(name) = struct_ty.name {
                    let name_str = format!("struct.{:?}", name);
                    // Reuse the existing named struct if we've already
                    // registered it on the LLVM context. `opaque_struct_type`
                    // unconditionally creates a fresh type with the next
                    // available `.N` suffix, which breaks type-equality
                    // for repeated `translate_type` calls on the same HIR
                    // struct (e.g. a phi node typed against an annotated
                    // variable type and the loaded/loaded-back value).
                    if let Some(existing) = self.context.get_struct_type(&name_str) {
                        return Ok(existing.into());
                    }
                    // First sighting — register the opaque shell *before*
                    // field translation so any recursive reference
                    // resolves to the same shell rather than spawning a
                    // `.N` suffix.
                    let struct_type = self.context.opaque_struct_type(&name_str);
                    let field_types: Result<Vec<BasicTypeEnum>, _> = struct_ty
                        .fields
                        .iter()
                        .map(|field_ty| self.translate_type(field_ty))
                        .collect();
                    let field_types = field_types?;
                    struct_type.set_body(&field_types, struct_ty.packed);
                    struct_type.into()
                } else {
                    // Anonymous struct — translate fields then create.
                    let field_types: Result<Vec<BasicTypeEnum>, _> = struct_ty
                        .fields
                        .iter()
                        .map(|field_ty| self.translate_type(field_ty))
                        .collect();
                    let field_types = field_types?;
                    self.context
                        .struct_type(&field_types, struct_ty.packed)
                        .into()
                }
            }
            Function(func_ty) => {
                // Translate function type to function pointer
                let param_types: Result<Vec<BasicMetadataTypeEnum>, _> = func_ty
                    .params
                    .iter()
                    .map(|param_ty| self.translate_type(param_ty).map(|t| t.into()))
                    .collect();

                let param_types = param_types?;

                // Handle return types
                let fn_type = if func_ty.returns.is_empty() {
                    // Void return
                    self.context
                        .void_type()
                        .fn_type(&param_types, func_ty.is_variadic)
                } else if func_ty.returns.len() == 1 {
                    // Single return value
                    let ret_ty = self.translate_type(&func_ty.returns[0])?;
                    ret_ty.fn_type(&param_types, func_ty.is_variadic)
                } else {
                    // Multiple returns - wrap in struct
                    let ret_types: Result<Vec<BasicTypeEnum>, _> = func_ty
                        .returns
                        .iter()
                        .map(|ret_ty| self.translate_type(ret_ty))
                        .collect();
                    let ret_types = ret_types?;
                    let struct_ret = self.context.struct_type(&ret_types, false);
                    struct_ret.fn_type(&param_types, func_ty.is_variadic)
                };

                // Return function pointer type
                fn_type.ptr_type(AddressSpace::default()).into()
            }
            Vector(elem_ty, count) => match (&**elem_ty, *count) {
                (F32, 4) => self.context.f32_type().vec_type(4).into(),
                (F64, 2) => self.context.f64_type().vec_type(2).into(),
                (I8, 16) | (U8, 16) => self.context.i8_type().vec_type(16).into(),
                (I16, 8) | (U16, 8) => self.context.i16_type().vec_type(8).into(),
                (I32, 4) | (U32, 4) => self.context.i32_type().vec_type(4).into(),
                (I64, 2) | (U64, 2) => self.context.i64_type().vec_type(2).into(),
                _ => {
                    return Err(CompilerError::CodeGen(format!(
                        "unsupported SIMD vector lane shape in LLVM backend: Vector({:?}, {})",
                        elem_ty, count
                    )));
                }
            },
            Union(union_ty) => {
                // LLVM has no native tagged-union type. Lower a HIR
                // Union as a struct of `{ discriminant_ty,
                // [N x i8] }` where N is the widest non-Void variant's
                // size in bytes. Backends store the discriminant
                // first, then bitcast the payload bytes to the variant
                // type they want. For C-style unions
                // (`is_c_union: true`) the discriminant slot is still
                // emitted but unused — codegen sites read/write only
                // the payload.
                //
                // This shape covers Option<T>, Result<T, E>, ZynML
                // enums with payloads, etc. — every kernel importing
                // the prelude trips it before this arm landed.
                let disc_ty = self.translate_type(&union_ty.discriminant_type)?;
                let mut max_payload_bytes: u32 = 0;
                for variant in &union_ty.variants {
                    if matches!(variant.ty, HirType::Void) {
                        continue;
                    }
                    let variant_bytes = hir_type_size_bytes(&variant.ty);
                    if variant_bytes > max_payload_bytes {
                        max_payload_bytes = variant_bytes;
                    }
                }
                let payload_ty = self.context.i8_type().array_type(max_payload_bytes);
                self.context
                    .struct_type(&[disc_ty, payload_ty.into()], false)
                    .into()
            }
            Opaque(name) => {
                // Opaque HIR types (forward declarations, prelude
                // extern types like Tensor, `@reference class` types
                // before the body is registered) lower to **a pointer
                // to** a named opaque LLVM struct. Opaque values have
                // unknown size so they can never appear as a first-
                // class LLVM value — every use site already expects a
                // pointer, whether the HIR source said `Opaque(X)`
                // directly or `Ptr(Opaque(X))`. Materialising the
                // opaque struct directly here panics downstream when
                // the consumer (function entry block, call site,
                // return slot) tries `into_pointer_value()`.
                //
                // Note this means `Ptr(Opaque(X))` lowers to `opaque.X*`
                // not `opaque.X**` — both HIR shapes collapse to the
                // same pointer-to-opaque, which matches how the runtime
                // already treats them.
                let name_str = format!("opaque.{:?}", name);
                let opaque_struct = self.context.opaque_struct_type(&name_str);
                opaque_struct.ptr_type(AddressSpace::default()).into()
            }
            _ => {
                return Err(CompilerError::CodeGen(format!(
                    "Type translation not yet implemented: {:?}",
                    ty
                )));
            }
        };

        Ok(result)
    }

    /// Get a value from the value map
    fn get_value(&self, id: HirId) -> CompilerResult<BasicValueEnum<'ctx>> {
        self.value_map
            .get(&id)
            .copied()
            .ok_or_else(|| CompilerError::CodeGen(format!("Value not found: {:?}", id)))
    }

    /// Synthesize a zero / null / undef of any LLVM basic type.
    /// Used as a defensive coercion when a stub-generated placeholder
    /// value's type doesn't match an aggregate-field or return slot.
    fn zero_of_basic_type(&self, ty: BasicTypeEnum<'ctx>) -> BasicValueEnum<'ctx> {
        match ty {
            BasicTypeEnum::IntType(t) => t.const_zero().into(),
            BasicTypeEnum::FloatType(t) => t.const_zero().into(),
            BasicTypeEnum::PointerType(t) => t.const_null().into(),
            BasicTypeEnum::StructType(t) => t.get_undef().into(),
            BasicTypeEnum::ArrayType(t) => t.get_undef().into(),
            BasicTypeEnum::VectorType(t) => t.get_undef().into(),
            _ => self.context.i64_type().const_zero().into(),
        }
    }

    /// Create a default value for a type (used for implicit returns)
    fn default_value(&self, ty: &HirType) -> CompilerResult<BasicValueEnum<'ctx>> {
        use HirType::*;

        let result = match ty {
            I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 => {
                let llvm_ty = self.translate_type(ty)?;
                llvm_ty.into_int_type().const_zero().into()
            }
            F32 | F64 => {
                let llvm_ty = self.translate_type(ty)?;
                llvm_ty.into_float_type().const_zero().into()
            }
            Bool => self.context.bool_type().const_zero().into(),
            _ => {
                return Err(CompilerError::CodeGen(format!(
                    "Cannot create default value for type: {:?}",
                    ty
                )));
            }
        };

        Ok(result)
    }

    /// Get the compiled LLVM module
    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Verify the module (checks for LLVM IR errors)
    pub fn verify(&self) -> Result<(), String> {
        self.module.verify().map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::*;

    #[test]
    fn test_llvm_backend_creation() {
        let context = Context::create();
        let backend = LLVMBackend::new(&context, "test_module");
        assert_eq!(backend.module.get_name().to_str().unwrap(), "test_module");
    }

    #[test]
    fn test_basic_type_translation() {
        let context = Context::create();
        let backend = LLVMBackend::new(&context, "test");

        // Test integer types
        assert!(backend.translate_type(&HirType::I32).is_ok());
        assert!(backend.translate_type(&HirType::I64).is_ok());
        assert!(backend.translate_type(&HirType::U32).is_ok());

        // Test float types
        assert!(backend.translate_type(&HirType::F32).is_ok());
        assert!(backend.translate_type(&HirType::F64).is_ok());

        // Test bool
        assert!(backend.translate_type(&HirType::Bool).is_ok());
    }

    #[test]
    fn test_constant_compilation() {
        let context = Context::create();
        let backend = LLVMBackend::new(&context, "test");

        // Integer constants
        let i32_const = backend.compile_constant(&HirConstant::I32(42));
        assert!(i32_const.is_ok());

        // Float constants
        let f64_const = backend.compile_constant(&HirConstant::F64(3.14));
        assert!(f64_const.is_ok());

        // Bool constants
        let bool_const = backend.compile_constant(&HirConstant::Bool(true));
        assert!(bool_const.is_ok());
    }
}
