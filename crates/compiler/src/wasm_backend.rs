//! HIR → WebAssembly code generation for the wasm-side tier-1 JIT.
//!
//! Same role on wasm32 that [`crate::cranelift_backend`] plays on
//! native: when a function gets hot, the BC interpreter hands its
//! `HirFunction` to this emitter, gets back a `WasmModule` (raw
//! wasm bytes), and the host's wasm crate hands the bytes to
//! `WebAssembly.compile(bytes)` (or the sync `new
//! WebAssembly.Module(bytes)`) to produce a `Module`. The host then
//! `new WebAssembly.Instance(mod, importObject)`s it and slots the
//! exported `entry` function into a shared funcref table.
//! Subsequent calls to that HIR function dispatch through
//! `call_indirect` against the table entry instead of stepping
//! through bytecode.
//!
//! Modeled on `wren_lift/src/codegen/wasm.rs` but consuming our HIR
//! directly:
//!
//! * `WasmBackend::compile_function(func)` is the only public entry —
//!   takes an `&HirFunction`, returns `Result<WasmModule>`.
//! * `WasmModule::validate()` checks the magic + version; the
//!   stronger `validate_full()` (gated on `cfg(test)`) runs the
//!   full wasmparser pipeline.
//!
//! Current coverage: primitive i64/f64 arithmetic; multi-block
//! control flow (`Return` / `Branch` / `CondBranch` / `Switch` over
//! integer constants) lowered via a universal dispatch-loop pattern;
//! phi nodes resolved out-of-SSA via predecessor-edge moves
//! (`local.set` on the matching incoming value before `br` to the
//! loop header); extern-symbol calls (`HirCallable::Symbol`)
//! lowered to wasm imports under module `"extern"` with an
//! `<name>@<arity>` suffix that lets the JS host pick the matching
//! per-arity dispatcher without parsing the wasm type section;
//! 8-byte `Load` / `Store` (i64 / u64 / f64) against the host's
//! linear memory imported as `(import "host" "memory")`. Internal
//! HIR-function calls, indirect calls, intrinsics, `Alloca` /
//! `GetElementPtr`, sub-i64 memory widths, effects, and inter-phi
//! cycles still stub out to a clear `WasmEmitError::Unsupported`.
//! The emitter is intentionally conservative — any HIR shape it
//! can't handle bails so the interpreter keeps that function in BC.
//!
//! Value representation:
//! * `HirType::I64` / `I32` / `I16` / `I8` / `Bool` / `UInt`-family
//!   → wasm `i64` (32-bit narrows widened on emit; the wasm side
//!   keeps everything in i64 to match the BC dispatch ABI).
//! * `HirType::F32` / `F64` → wasm `f64`.
//! * Other types → `WasmEmitError::Unsupported` for now.

use crate::hir::{
    BinaryOp, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirTerminator, HirType,
    HirValueKind,
};
use std::collections::HashMap;

use wasm_encoder::{
    CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection, ImportSection,
    Instruction as WasmInst, MemArg, MemoryType, Module, TypeSection, ValType,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Things the emitter can't (yet) turn into wasm.
///
/// Every variant is recoverable from the caller's perspective:
/// when the wasm-tier compile fails the interpreter keeps the HIR
/// function in bytecode mode, so we never need to make these
/// fatal.
#[derive(Debug)]
pub enum WasmEmitError {
    /// HIR shape outside the emitter's current coverage. Carries
    /// the offending construct's name for diagnostics.
    Unsupported(String),
    /// HIR references a value id with no defining instruction in
    /// the function — usually a pass-time bug, not a coverage gap.
    UnknownValue(HirId),
    /// Wasm encoder rejected the produced module structurally.
    /// Should never happen if the emitter is correct; surfaces
    /// bugs in our lowering rather than caller-visible failures.
    Validation(String),
}

impl std::fmt::Display for WasmEmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "wasm-jit can't lower: {what}"),
            Self::UnknownValue(id) => write!(f, "wasm-jit: unknown value {:?}", id),
            Self::Validation(msg) => write!(f, "wasm-jit produced invalid module: {msg}"),
        }
    }
}

impl std::error::Error for WasmEmitError {}

type Result<T> = std::result::Result<T, WasmEmitError>;

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// A compiled wasm module (raw bytes). Hand-off shape to the JS
/// host:
///
/// ```js
/// const mod  = await WebAssembly.compile(bytes);
/// const inst = await WebAssembly.instantiate(mod, importObject);
/// const fn   = inst.exports.entry;
/// jitTable.set(slot, fn);
/// ```
///
/// (Or the synchronous `new WebAssembly.Module(bytes)` / `new
/// WebAssembly.Instance(mod, ...)` pair when the host wants to
/// avoid the `await`.)
///
/// The `bytes` field is the only state — there's no associated
/// metadata to thread through. Validation and instantiation are
/// the host's responsibility.
#[derive(Clone, Debug)]
pub struct WasmModule {
    pub bytes: Vec<u8>,
}

impl WasmModule {
    /// Quick magic-number check. Catches catastrophic encoder
    /// failure; doesn't validate section structure.
    pub fn validate(&self) -> Result<()> {
        if self.bytes.len() < 8 {
            return Err(WasmEmitError::Validation(format!(
                "module too small ({} bytes)",
                self.bytes.len()
            )));
        }
        if &self.bytes[0..4] != b"\0asm" {
            return Err(WasmEmitError::Validation(
                "missing wasm magic number".to_string(),
            ));
        }
        Ok(())
    }

    /// Full structural validation. Test-only because pulling
    /// `wasmparser` into the production hot path is wasted work —
    /// the host's `WebAssembly.instantiate` validates anyway.
    #[cfg(any(test, feature = "wasm-jit"))]
    pub fn validate_full(&self) -> Result<()> {
        wasmparser::validate(&self.bytes)
            .map(|_| ())
            .map_err(|e| WasmEmitError::Validation(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// HIR → wasm emitter. One instance per `compile_function` call;
/// there's no cross-function state to amortise yet (each emitted
/// module currently holds exactly one function).
pub struct WasmBackend;

impl WasmBackend {
    pub fn new() -> Self {
        Self
    }

    /// Lower one HIR function to a standalone wasm module.
    ///
    /// The resulting module has shape:
    /// ```text
    /// (module
    ///   (type (func (param i64*N) (result i64-or-f64)))
    ///   (func (export "entry") ...)
    /// )
    /// ```
    ///
    /// The single export is always named `"entry"` so the host
    /// shim doesn't have to thread the original HIR symbol through.
    /// Type pre-imports / function imports come in Phase E.5
    /// (extern call support).
    pub fn compile_function(&self, func: &HirFunction) -> Result<WasmModule> {
        let mut emitter = FunctionEmitter::new(func);
        emitter.emit()
    }
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Per-function emitter
// ---------------------------------------------------------------------------

struct FunctionEmitter<'a> {
    func: &'a HirFunction,
    /// HirId → wasm local index. We allocate one local per SSA
    /// value referenced by the function (parameters first, then
    /// instruction results), which keeps lowering simple at the
    /// cost of being non-minimal — the encoder + downstream
    /// `WebAssembly.instantiate` optimiser cleans up.
    local_map: HashMap<HirId, u32>,
    /// Allocated local types, in the order they appear in the
    /// final `Function::new` parameter. Excludes function
    /// parameters, which are implicit-local 0..N.
    locals: Vec<ValType>,
    /// Number of explicitly declared params. The first N locals
    /// (indices 0..N) refer to these and aren't redeclared in the
    /// locals section.
    n_params: u32,
    /// Wasm imports we need to declare so `HirCallable::Symbol`
    /// calls can resolve. Insertion order maps directly to wasm
    /// function-index 0..imports.len(); the compiled function is
    /// at function-index `imports.len()`.
    ///
    /// Each entry records `(import_name_with_arity_suffix, arity)`.
    /// The arity-suffixed name (e.g. `"$IO$println@1"`) lets the
    /// JS host pick the matching dispatcher at instantiation time
    /// without parsing the wasm type section.
    imports: Vec<(String, u32)>,
    /// Lookup table from a `HirCallable::Symbol(name)` value to the
    /// wasm function-index of the matching import. Built lazily as
    /// the body emit encounters Call instructions.
    import_indices: HashMap<String, u32>,
    /// `true` once `scan_imports` has spotted at least one
    /// `Load`/`Store` instruction. Triggers a `(import "host"
    /// "memory" (memory 0))` declaration so the emitted wasm
    /// shares the host runtime's linear memory — that's how
    /// JIT'd code reads structs / arrays the BC interp built.
    needs_memory_import: bool,
}

/// Per-block state threaded through `emit_terminator` so jumps
/// can translate HIR `Branch` / `CondBranch` targets into the right
/// wasm `br` relative-depth + `$next_block` assignment for the
/// dispatch-loop lowering. `None` in single-block functions — the
/// terminator never `br`s, only `return`s.
struct DispatchCtx<'b> {
    /// HIR block id of the predecessor whose terminator is being
    /// emitted. Phi resolution needs this to pick the right
    /// incoming edge — each `HirPhi.incoming` is a list of
    /// `(value, predecessor_block_id)` pairs.
    from_block: HirId,
    /// Local index of the i32 "next block" dispatch variable.
    next_block_local: u32,
    /// Relative depth from the terminator's emission site outward
    /// to the enclosing `loop`. `br loop_depth` re-enters the
    /// dispatch table for the next block.
    loop_depth: u32,
    /// HIR block id → 0-based dispatch index.
    block_indices: &'b HashMap<HirId, u32>,
}

impl<'a> FunctionEmitter<'a> {
    fn new(func: &'a HirFunction) -> Self {
        Self {
            func,
            local_map: HashMap::new(),
            locals: Vec::new(),
            n_params: 0,
            imports: Vec::new(),
            import_indices: HashMap::new(),
            needs_memory_import: false,
        }
    }

    fn emit(&mut self) -> Result<WasmModule> {
        // 1. Map parameters to locals 0..N.
        self.allocate_params()?;

        // 2. Pre-allocate locals for every instruction result the
        //    function defines. Doing this in a pre-pass keeps the
        //    instruction emit code linear — by the time we emit a
        //    use, the def's local index is already in `local_map`.
        self.allocate_instruction_locals()?;

        // 3. Discover wasm imports. Each `HirCallable::Symbol(name)`
        //    becomes one import; arity-suffixed so the JS host can
        //    pick the right dispatcher without parsing the wasm
        //    type section.
        self.scan_imports()?;

        // 4. Emit the function body.
        let body = self.emit_body()?;

        // 5. Build the module: type → import → function → export → code.
        let mut module = Module::new();

        // Type section: one entry per unique import signature, then
        // the compiled function's own type. Each import shares the
        // same wasm-side shape — N i64 params, 1 i64 result —
        // matching the BC interpreter's i64-funneled ABI.
        let mut types = TypeSection::new();
        let mut arity_to_type_idx: HashMap<u32, u32> = HashMap::new();
        for &(_, arity) in &self.imports {
            if arity_to_type_idx.contains_key(&arity) {
                continue;
            }
            let idx = types.len() as u32;
            let params: Vec<ValType> = (0..arity).map(|_| ValType::I64).collect();
            types.function(params, [ValType::I64]);
            arity_to_type_idx.insert(arity, idx);
        }
        let func_type_idx = types.len() as u32;
        let params: Vec<ValType> = self
            .func
            .signature
            .params
            .iter()
            .map(|p| lower_type(&p.ty))
            .collect::<Result<Vec<_>>>()?;
        let results: Vec<ValType> = self
            .func
            .signature
            .returns
            .iter()
            .map(lower_type)
            .collect::<Result<Vec<_>>>()?;
        types.function(params.iter().copied(), results.iter().copied());
        module.section(&types);

        // Import section. One `(import "extern" "<name>@<arity>"
        // (func (type <idx>)))` per discovered Symbol callee, plus
        // a host-shared linear memory if any Load/Store fires
        // inside the body.
        if !self.imports.is_empty() || self.needs_memory_import {
            let mut imports = ImportSection::new();
            for (name, arity) in &self.imports {
                let type_idx = *arity_to_type_idx
                    .get(arity)
                    .expect("arity → type-index mapping populated above");
                imports.import("extern", name, EntityType::Function(type_idx));
            }
            if self.needs_memory_import {
                // `minimum: 0` — accept whatever the host has.
                // `maximum: None` — no upper bound. `memory64: false`
                // — host wasm is 32-bit memory (linear-memory addrs
                // are i32). Shared/threading flags off for now.
                imports.import(
                    "host",
                    "memory",
                    EntityType::Memory(MemoryType {
                        minimum: 0,
                        maximum: None,
                        memory64: false,
                        shared: false,
                        page_size_log2: None,
                    }),
                );
            }
            module.section(&imports);
        }

        let mut functions = FunctionSection::new();
        functions.function(func_type_idx);
        module.section(&functions);

        // The compiled function lives at wasm function-index
        // `imports.len()` — imports occupy indices 0..imports.len().
        let func_index = self.imports.len() as u32;

        let mut exports = ExportSection::new();
        exports.export("entry", ExportKind::Func, func_index);
        module.section(&exports);

        let mut code = CodeSection::new();
        // Group locals by ValType for the compact wasm encoding.
        let local_decls = compact_locals(&self.locals);
        let mut wfunc = Function::new(local_decls);
        for inst in body {
            wfunc.instruction(&inst);
        }
        wfunc.instruction(&WasmInst::End);
        code.function(&wfunc);
        module.section(&code);

        let bytes = module.finish();
        let m = WasmModule { bytes };
        m.validate()?;
        Ok(m)
    }

    fn allocate_params(&mut self) -> Result<()> {
        for (i, p) in self.func.signature.params.iter().enumerate() {
            // Parameter values in HIR are referenced via the
            // synthetic value-id stored on the param. Some HIR
            // shapes also create a separate `HirValueKind::Parameter`
            // value; we map both to the same local.
            self.local_map.insert(p.id, i as u32);
            // Cross-check that the param's wasm type lowering
            // succeeds — fails fast for unsupported shapes (Struct
            // params etc.) rather than mid-body.
            let _ = lower_type(&p.ty)?;
        }
        // Also register any `HirValueKind::Parameter(i)` value
        // entries that point back at the same parameter index.
        for (id, value) in &self.func.values {
            if let HirValueKind::Parameter(idx) = value.kind {
                self.local_map.insert(*id, idx);
            }
        }
        self.n_params = self.func.signature.params.len() as u32;
        Ok(())
    }

    fn allocate_instruction_locals(&mut self) -> Result<()> {
        // Walk the function's value table to find instruction
        // results. SSA puts one entry per defined value; constants
        // get their own value-id and we allocate a local for them
        // too so the use site can `local.get` rather than re-
        // emitting the literal.
        for (id, value) in &self.func.values {
            if self.local_map.contains_key(id) {
                continue;
            }
            match value.kind {
                HirValueKind::Instruction | HirValueKind::Constant(_) | HirValueKind::Undef => {
                    let ty = lower_type(&value.ty)?;
                    let local_idx = self.n_params + self.locals.len() as u32;
                    self.local_map.insert(*id, local_idx);
                    self.locals.push(ty);
                }
                HirValueKind::Parameter(_) => { /* already mapped */ }
                _ => {
                    // Globals / etc. unsupported in this initial
                    // slice; the use site will hit UnknownValue and
                    // bail.
                }
            }
        }

        // Phi-result locals. Each `HirBlock::phis[i].result` carries
        // an SSA value-id and a type; emit a local of that type so
        // predecessor-edge moves (Phase E.3.1 out-of-SSA lowering)
        // have a single destination they can `local.set` regardless
        // of which incoming edge fires.
        for block in self.func.blocks.values() {
            for phi in &block.phis {
                if self.local_map.contains_key(&phi.result) {
                    continue;
                }
                let ty = lower_type(&phi.ty)?;
                let local_idx = self.n_params + self.locals.len() as u32;
                self.local_map.insert(phi.result, local_idx);
                self.locals.push(ty);
            }
        }
        Ok(())
    }

    /// Walk every `HirInstruction::Call` in every block and register
    /// each `HirCallable::Symbol(name)` callee as a wasm import. The
    /// import name carries the arity suffix (`name@N`) so the JS
    /// host can dispatch without parsing the wasm type section.
    /// Internal calls (`Function(id)` / `Indirect(..)` / etc.) hit
    /// `Unsupported` here — they'd need a more elaborate
    /// cross-function dispatch story (E.5.2+).
    fn scan_imports(&mut self) -> Result<()> {
        for block in self.func.blocks.values() {
            for inst in &block.instructions {
                if matches!(
                    inst,
                    HirInstruction::Load { .. } | HirInstruction::Store { .. }
                ) {
                    self.needs_memory_import = true;
                }
                if let HirInstruction::Call { callee, args, .. } = inst {
                    match callee {
                        HirCallable::Symbol(name) => {
                            let arity = args.len() as u32;
                            let import_name = format!("{}@{}", name, arity);
                            if !self.import_indices.contains_key(&import_name) {
                                let idx = self.imports.len() as u32;
                                self.imports.push((import_name.clone(), arity));
                                self.import_indices.insert(import_name, idx);
                            }
                        }
                        HirCallable::Function(_) => {
                            return Err(WasmEmitError::Unsupported(
                                "internal HIR function call (HirCallable::Function) — \
                                 cross-function dispatch not yet wired"
                                    .into(),
                            ));
                        }
                        HirCallable::Indirect(_) => {
                            return Err(WasmEmitError::Unsupported(
                                "indirect call (HirCallable::Indirect)".into(),
                            ));
                        }
                        HirCallable::Intrinsic(_) => {
                            return Err(WasmEmitError::Unsupported(
                                "intrinsic call (HirCallable::Intrinsic)".into(),
                            ));
                        }
                        HirCallable::FuncRef(_) => {
                            return Err(WasmEmitError::Unsupported(
                                "function-as-pointer (HirCallable::FuncRef)".into(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn emit_body(&mut self) -> Result<Vec<WasmInst<'static>>> {
        let mut out = Vec::new();

        // Emit constants up front so subsequent uses can `local.get`.
        // Walk in deterministic order (the function's value map)
        // so the emitted module is reproducible.
        for (id, value) in &self.func.values {
            if let HirValueKind::Constant(c) = &value.kind {
                let dst = *self
                    .local_map
                    .get(id)
                    .ok_or(WasmEmitError::UnknownValue(*id))?;
                self.emit_constant(&mut out, c, &value.ty)?;
                out.push(WasmInst::LocalSet(dst));
            }
        }

        // Single-block fast path — straight-line code with one
        // terminator, no dispatch infrastructure needed.
        if self.func.blocks.len() == 1 {
            let (_, block) = self
                .func
                .blocks
                .iter()
                .next()
                .ok_or_else(|| WasmEmitError::Unsupported("function with no blocks".into()))?;
            if !block.phis.is_empty() {
                return Err(WasmEmitError::Unsupported(
                    "phi node in single-block function (should be impossible)".into(),
                ));
            }
            for inst in &block.instructions {
                self.emit_instruction(&mut out, inst)?;
            }
            self.emit_terminator(&mut out, &block.terminator, None)?;
            return Ok(out);
        }

        // Multi-block path — dispatch-loop lowering.
        //
        // Wasm only has structured control flow (`block`/`loop`/`if`
        // with `br` to enclosing labels). HIR is an arbitrary CFG.
        // The universal correct lowering: a `loop` wrapping a stack
        // of `block`s with a `br_table` at the start that dispatches
        // on a single i32 "next block" local. Each terminator either:
        //   * sets `$next = <succ_idx>` and `br` to the loop header,
        //   * or returns directly out of the function.
        //
        //   (loop $dispatch
        //     (block $b_N-1
        //       ...
        //       (block $b_1
        //         (block $b_0
        //           (br_table 0 1 ... N-1 (local.get $next)))
        //         ;; block 0 body + terminator
        //       )
        //       ;; block 1 body + terminator
        //     )
        //     ;; block N-1 body + terminator
        //   )
        //
        // The dispatching label-stack depth equals the number of
        // blocks. `br $dispatch` from inside (depth N) re-enters the
        // loop and the `br_table` jumps to the right block on the
        // next iteration. `Return` terminators don't `br` at all —
        // they emit operands and `return`, leaving the entire loop
        // structure intact.
        //
        // Phi nodes are NOT yet handled. A function with any
        // `block.phis` non-empty bails to `Unsupported` so the
        // interpreter keeps it in BC. Out-of-SSA via predecessor-
        // edge moves lands in a follow-up (Phase E.3.1).
        self.emit_multi_block(&mut out)?;
        Ok(out)
    }

    fn emit_multi_block(&mut self, out: &mut Vec<WasmInst<'static>>) -> Result<()> {
        // Inter-phi conflict check: if a phi's incoming value is
        // another phi's result in the SAME block, we'd need parallel-
        // copy resolution (with temporaries) to avoid stomping on the
        // value before reading it. Bail rather than emit incorrect
        // code — production HIR from `SsaBuilder` doesn't produce
        // these shapes for the loops we care about; if it ever does,
        // a follow-up can implement the parallel-copy split.
        for block in self.func.blocks.values() {
            let phi_results: std::collections::HashSet<HirId> =
                block.phis.iter().map(|p| p.result).collect();
            for phi in &block.phis {
                for (val, _pred) in &phi.incoming {
                    if phi_results.contains(val) {
                        return Err(WasmEmitError::Unsupported(format!(
                            "phi cycle in block {:?}: phi result {:?} \
                             references another phi in the same block",
                            block.id, phi.result
                        )));
                    }
                }
            }
        }

        // Assign a 0-based index to each block. Entry must be index
        // 0 so the dispatch-loop falls into it on the first
        // iteration (we initialise `$next = 0` implicitly via the
        // default-zero of the local).
        let entry = self.func.entry_block;
        let mut block_indices: HashMap<HirId, u32> = HashMap::new();
        let mut block_order: Vec<HirId> = Vec::new();
        block_indices.insert(entry, 0);
        block_order.push(entry);
        for id in self.func.blocks.keys() {
            if *id == entry {
                continue;
            }
            block_indices.insert(*id, block_order.len() as u32);
            block_order.push(*id);
        }
        let n_blocks = block_order.len() as u32;

        // Allocate the "next block" dispatch local (i32). Its
        // default-zero value picks `entry` (block 0) on first entry.
        let next_block_local = self.n_params + self.locals.len() as u32;
        self.locals.push(ValType::I32);

        // Emit the loop + nested blocks. Walk depth from N-1 down
        // to 0, opening a `block` at each step. The innermost
        // `block` (depth 0, i.e. closest to the `br_table`) holds
        // the dispatch; bodies fan out as we close blocks.
        out.push(WasmInst::Loop(wasm_encoder::BlockType::Empty));

        // Open one wasm `block` per HIR block.
        for _ in 0..n_blocks {
            out.push(WasmInst::Block(wasm_encoder::BlockType::Empty));
        }

        // br_table for dispatch. Targets are relative-depth indices
        // from the br_table location outward:
        //   depth 0 → innermost block → block 0 body
        //   depth 1 → next block → block 1 body
        //   ...
        //   depth N-1 → outermost block → block N-1 body
        //
        // br_table's "default" target uses the same relative-depth
        // scheme. We use depth 0 as the safe default — if a
        // mis-set `$next` somehow falls through, it just executes
        // the entry block again rather than corrupting control
        // flow.
        out.push(WasmInst::LocalGet(next_block_local));
        out.push(WasmInst::BrTable(
            (0..n_blocks).collect::<Vec<u32>>().into(),
            0,
        ));

        // Emit each block's body followed by its `End` (closing the
        // wasm `block` we opened above). Loop runs in `block_order`
        // (entry first) so the textual emission order matches the
        // index numbering.
        for (depth_from_inside, block_id) in block_order.iter().enumerate() {
            let block = &self.func.blocks[block_id];

            // Close the surrounding `block` for THIS HIR block.
            out.push(WasmInst::End);

            // Instructions in this block.
            for inst in &block.instructions {
                self.emit_instruction(out, inst)?;
            }

            // Terminator. `dispatch_ctx` lets the terminator emit
            // `br $dispatch` jumps via the right relative depth.
            //
            // After closing the `end` for THIS HIR block (just
            // pushed above), we're sitting inside `n_blocks -
            // depth_from_inside - 1` remaining wasm `block`s, which
            // themselves sit inside the `loop`. Relative depth 0
            // from this emission point is the nearest enclosing
            // block (or the loop if all blocks have been closed).
            //
            // The `loop` label sits at relative depth equal to the
            // count of enclosing wasm `block`s — NOT +1 — because
            // br-to-loop counts the loop itself as one of the
            // labels in scope.
            //
            //   for n_blocks = 2:
            //     after closing inner block (depth_from_inside=0):
            //       remaining outer blocks = 1, loop_depth = 1
            //     after closing outer block (depth_from_inside=1):
            //       remaining outer blocks = 0, loop_depth = 0
            //
            // CondBranch additionally opens an `if` block inside
            // this scope, so depths emitted from inside the if/else
            // body need +1 — handled at the terminator emit site.
            let outer_blocks_remaining = (n_blocks as usize) - depth_from_inside - 1;
            let loop_depth = outer_blocks_remaining as u32;
            let dispatch_ctx = DispatchCtx {
                from_block: *block_id,
                next_block_local,
                loop_depth,
                block_indices: &block_indices,
            };
            self.emit_terminator(out, &block.terminator, Some(&dispatch_ctx))?;
        }

        // Close the outer `loop`, then emit `unreachable` at the
        // function level. Two purposes:
        //
        // 1. **Trap on bug.** Every well-formed HIR terminator
        //    either `return`s out of the function or `br`s back to
        //    the loop header — control should never structurally
        //    fall off the loop. `unreachable` makes "fell off" a
        //    clean trap rather than executing into whatever
        //    follows in the code section.
        //
        // 2. **Satisfy the function's return type.** The loop has
        //    `BlockType::Empty`, so after its `end` the stack is
        //    empty. The function's signature may demand an i64 /
        //    f64; `unreachable` after the loop end makes the stack
        //    polymorphic-bottom, which satisfies any expected
        //    result type at the function end. Putting `unreachable`
        //    *inside* the loop would leave an empty stack after the
        //    loop closes and trip the wasm validator with "expected
        //    i64 but nothing on stack".
        out.push(WasmInst::End);
        out.push(WasmInst::Unreachable);
        Ok(())
    }

    fn emit_constant(
        &self,
        out: &mut Vec<WasmInst<'static>>,
        c: &HirConstant,
        ty: &HirType,
    ) -> Result<()> {
        match c {
            HirConstant::I64(n) => out.push(WasmInst::I64Const(*n)),
            HirConstant::I32(n) => out.push(WasmInst::I64Const(*n as i64)),
            HirConstant::I16(n) => out.push(WasmInst::I64Const(*n as i64)),
            HirConstant::I8(n) => out.push(WasmInst::I64Const(*n as i64)),
            HirConstant::U64(n) => out.push(WasmInst::I64Const(*n as i64)),
            HirConstant::U32(n) => out.push(WasmInst::I64Const(*n as i64)),
            HirConstant::U16(n) => out.push(WasmInst::I64Const(*n as i64)),
            HirConstant::U8(n) => out.push(WasmInst::I64Const(*n as i64)),
            HirConstant::F64(n) => out.push(WasmInst::F64Const((*n).into())),
            HirConstant::F32(n) => out.push(WasmInst::F64Const((*n as f64).into())),
            HirConstant::Bool(b) => out.push(WasmInst::I64Const(if *b { 1 } else { 0 })),
            other => {
                return Err(WasmEmitError::Unsupported(format!(
                    "constant {:?} (ty {:?})",
                    other, ty
                )))
            }
        }
        Ok(())
    }

    fn emit_instruction(
        &self,
        out: &mut Vec<WasmInst<'static>>,
        inst: &HirInstruction,
    ) -> Result<()> {
        match inst {
            HirInstruction::Binary {
                result,
                op,
                ty,
                left,
                right,
            } => {
                let lhs = self.local_get(*left)?;
                let rhs = self.local_get(*right)?;
                let dst = self.local_set(*result)?;
                out.push(lhs);
                out.push(rhs);
                emit_binary_op(out, *op, ty)?;
                out.push(dst);
                Ok(())
            }
            HirInstruction::Load {
                result, ty, ptr, ..
            } => {
                // Load addresses come in as i64 (the BC interp's
                // funneled ABI funnels pointers through i64). Wasm
                // memory operands are i32, so wrap before the load.
                //
                // Width selection: i64 / u64 / f64 / pointers (any
                // 8-byte HIR type) → 8-byte load. Smaller widths
                // / unsupported types stay `Unsupported` until we
                // grow the table; the BC interp keeps that function.
                out.push(self.local_get(*ptr)?);
                out.push(WasmInst::I32WrapI64);
                let memarg = MemArg {
                    offset: 0,
                    align: 3, // log2(8) — 8-byte aligned
                    memory_index: 0,
                };
                let load = match ty {
                    HirType::I64 | HirType::U64 => WasmInst::I64Load(memarg),
                    HirType::F64 => WasmInst::F64Load(memarg),
                    other => {
                        return Err(WasmEmitError::Unsupported(format!(
                            "Load of HirType {:?} (only I64/U64/F64 supported in this slice)",
                            other
                        )));
                    }
                };
                out.push(load);
                out.push(self.local_set(*result)?);
                Ok(())
            }
            HirInstruction::Store { value, ptr, .. } => {
                // Wasm `*.store` takes (addr, value) on the stack;
                // push them in that order. Value's wasm type
                // dictates the store opcode.
                out.push(self.local_get(*ptr)?);
                out.push(WasmInst::I32WrapI64);
                out.push(self.local_get(*value)?);

                let value_ty = self
                    .func
                    .values
                    .get(value)
                    .ok_or(WasmEmitError::UnknownValue(*value))?
                    .ty
                    .clone();
                let memarg = MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                };
                let store = match value_ty {
                    HirType::I64 | HirType::U64 => WasmInst::I64Store(memarg),
                    HirType::F64 => WasmInst::F64Store(memarg),
                    other => {
                        return Err(WasmEmitError::Unsupported(format!(
                            "Store of HirType {:?} (only I64/U64/F64 supported in this slice)",
                            other
                        )));
                    }
                };
                out.push(store);
                Ok(())
            }
            HirInstruction::Call {
                result,
                callee,
                args,
                ..
            } => {
                // Only `HirCallable::Symbol` reaches here — other
                // variants were rejected by `scan_imports`.
                let name = match callee {
                    HirCallable::Symbol(n) => n,
                    _ => {
                        return Err(WasmEmitError::Unsupported(format!(
                            "non-Symbol callee in emit_instruction: {:?}",
                            std::mem::discriminant(callee)
                        )))
                    }
                };
                let arity = args.len() as u32;
                let import_name = format!("{}@{}", name, arity);
                let import_idx = *self.import_indices.get(&import_name).ok_or_else(|| {
                    // `scan_imports` walked the same instructions, so this
                    // is a structural bug rather than a coverage gap.
                    WasmEmitError::Validation(format!(
                        "call site for `{}` has no matching import index",
                        import_name
                    ))
                })?;

                // Push each arg via `local.get`.
                for arg in args {
                    out.push(self.local_get(*arg)?);
                }
                out.push(WasmInst::Call(import_idx));

                // Imports always have i64 result type (matches the
                // BC interp's i64-funneled ABI). If the HIR call
                // produces a value, capture it; otherwise drop the
                // return value off the stack.
                if let Some(result_id) = result {
                    out.push(self.local_set(*result_id)?);
                } else {
                    out.push(WasmInst::Drop);
                }
                Ok(())
            }
            other => Err(WasmEmitError::Unsupported(format!(
                "instruction {:?}",
                std::mem::discriminant(other)
            ))),
        }
    }

    fn emit_terminator(
        &self,
        out: &mut Vec<WasmInst<'static>>,
        term: &HirTerminator,
        dispatch: Option<&DispatchCtx<'_>>,
    ) -> Result<()> {
        match term {
            HirTerminator::Return { values } => {
                for v in values {
                    out.push(self.local_get(*v)?);
                }
                out.push(WasmInst::Return);
                Ok(())
            }
            HirTerminator::Branch { target } => {
                let ctx = dispatch.ok_or_else(|| {
                    WasmEmitError::Unsupported(
                        "unconditional Branch in single-block function (should be unreachable)"
                            .into(),
                    )
                })?;
                // Out-of-SSA: write each of `target`'s phi results
                // BEFORE we jump, so the destination block sees the
                // value matching the edge we came from. Then set
                // `$next = target_idx; br loop_depth` so the next
                // dispatch iteration enters the target block.
                self.emit_phi_moves(out, *target, ctx.from_block)?;
                let idx = *ctx
                    .block_indices
                    .get(target)
                    .ok_or(WasmEmitError::UnknownValue(*target))?;
                out.push(WasmInst::I32Const(idx as i32));
                out.push(WasmInst::LocalSet(ctx.next_block_local));
                out.push(WasmInst::Br(ctx.loop_depth));
                Ok(())
            }
            HirTerminator::Switch {
                value,
                default,
                cases,
            } => {
                let ctx = dispatch.ok_or_else(|| {
                    WasmEmitError::Unsupported(
                        "Switch in single-block function (should be unreachable)".into(),
                    )
                })?;
                // Lower to a sequence of `if`/`end` blocks — one per
                // case. Each `if` tests `value == case_constant` and
                // conditionally jumps to that case's target; the
                // default fires if none of the cases matched (fall-
                // through after the last `end`).
                //
                // Wasm depth math: each `if` adds 1 to the relative
                // depth needed to reach the surrounding `loop`. Since
                // each case's `br` lives INSIDE its own `if`, the
                // depth is `loop_depth + 1`. The default `br` sits
                // OUTSIDE all the `if`s, so its depth is just
                // `loop_depth`.
                for (case_const, case_target) in cases {
                    // Emit the case-constant + equality test.
                    out.push(self.local_get(*value)?);
                    let case_i64 = case_constant_to_i64(case_const)?;
                    out.push(WasmInst::I64Const(case_i64));
                    out.push(WasmInst::I64Eq);
                    out.push(WasmInst::If(wasm_encoder::BlockType::Empty));
                    // Match: phi moves + dispatch.
                    self.emit_phi_moves(out, *case_target, ctx.from_block)?;
                    let case_idx = *ctx
                        .block_indices
                        .get(case_target)
                        .ok_or(WasmEmitError::UnknownValue(*case_target))?;
                    out.push(WasmInst::I32Const(case_idx as i32));
                    out.push(WasmInst::LocalSet(ctx.next_block_local));
                    out.push(WasmInst::Br(ctx.loop_depth + 1));
                    out.push(WasmInst::End);
                }
                // Default — reached only if no case matched.
                self.emit_phi_moves(out, *default, ctx.from_block)?;
                let default_idx = *ctx
                    .block_indices
                    .get(default)
                    .ok_or(WasmEmitError::UnknownValue(*default))?;
                out.push(WasmInst::I32Const(default_idx as i32));
                out.push(WasmInst::LocalSet(ctx.next_block_local));
                out.push(WasmInst::Br(ctx.loop_depth));
                Ok(())
            }
            HirTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                let ctx = dispatch.ok_or_else(|| {
                    WasmEmitError::Unsupported(
                        "CondBranch in single-block function (should be unreachable)".into(),
                    )
                })?;
                let true_idx = *ctx
                    .block_indices
                    .get(true_target)
                    .ok_or(WasmEmitError::UnknownValue(*true_target))?;
                let false_idx = *ctx
                    .block_indices
                    .get(false_target)
                    .ok_or(WasmEmitError::UnknownValue(*false_target))?;
                // Wasm `if`/`else` consumes an i32; HIR conditions
                // are bool widened to i64 in our representation, so
                // narrow it to i32 here. `i32.wrap_i64` truncates
                // the low 32 bits which is exactly the right
                // boolean check (any non-zero → true).
                out.push(self.local_get(*condition)?);
                out.push(WasmInst::I32WrapI64);
                out.push(WasmInst::If(wasm_encoder::BlockType::Empty));
                // True arm: phi moves for `true_target` from this
                // predecessor, then dispatch.
                self.emit_phi_moves(out, *true_target, ctx.from_block)?;
                out.push(WasmInst::I32Const(true_idx as i32));
                out.push(WasmInst::LocalSet(ctx.next_block_local));
                out.push(WasmInst::Br(ctx.loop_depth + 1));
                out.push(WasmInst::Else);
                // False arm: phi moves for `false_target` from this
                // predecessor, then dispatch.
                self.emit_phi_moves(out, *false_target, ctx.from_block)?;
                out.push(WasmInst::I32Const(false_idx as i32));
                out.push(WasmInst::LocalSet(ctx.next_block_local));
                out.push(WasmInst::Br(ctx.loop_depth + 1));
                out.push(WasmInst::End);
                Ok(())
            }
            other => Err(WasmEmitError::Unsupported(format!(
                "terminator {:?}",
                std::mem::discriminant(other)
            ))),
        }
    }

    fn local_get(&self, id: HirId) -> Result<WasmInst<'static>> {
        let idx = *self
            .local_map
            .get(&id)
            .ok_or(WasmEmitError::UnknownValue(id))?;
        Ok(WasmInst::LocalGet(idx))
    }

    fn local_set(&self, id: HirId) -> Result<WasmInst<'static>> {
        let idx = *self
            .local_map
            .get(&id)
            .ok_or(WasmEmitError::UnknownValue(id))?;
        Ok(WasmInst::LocalSet(idx))
    }

    /// Emit `local.set` moves for each phi in `target` block whose
    /// incoming edge matches `from_block`. This is the out-of-SSA
    /// step — the wasm-side equivalent of inserting predecessor-edge
    /// copies. Each phi result has a pre-allocated local; the
    /// matching incoming value's local is `local.get`'d and pushed
    /// into the result local right before the `br` to the dispatch
    /// loop.
    ///
    /// If a phi has no incoming edge from `from_block` (shouldn't
    /// happen for well-formed SSA but defensive) the phi is left
    /// unchanged on this edge.
    ///
    /// Phi cycles (one phi's result referenced by another phi's
    /// incoming in the same target) are rejected by
    /// `emit_multi_block`'s up-front check, so this loop emits the
    /// moves in declaration order without parallel-copy resolution.
    fn emit_phi_moves(
        &self,
        out: &mut Vec<WasmInst<'static>>,
        target: HirId,
        from_block: HirId,
    ) -> Result<()> {
        let block = self
            .func
            .blocks
            .get(&target)
            .ok_or(WasmEmitError::UnknownValue(target))?;
        for phi in &block.phis {
            let incoming_val = phi.incoming.iter().find_map(|(value, pred)| {
                if *pred == from_block {
                    Some(*value)
                } else {
                    None
                }
            });
            if let Some(value_id) = incoming_val {
                let dst = *self
                    .local_map
                    .get(&phi.result)
                    .ok_or(WasmEmitError::UnknownValue(phi.result))?;
                out.push(self.local_get(value_id)?);
                out.push(WasmInst::LocalSet(dst));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a `HirType` to its wasm-side wire type. Integers all widen
/// to `i64`; floats all widen to `f64`. This matches the BC
/// interpreter's "everything funnels through i64 for the FFI ABI"
/// convention so handoffs between interp and JIT don't need
/// narrowing trampolines.
fn lower_type(ty: &HirType) -> Result<ValType> {
    use HirType::*;
    Ok(match ty {
        I64 | I32 | I16 | I8 | U64 | U32 | U16 | U8 | Bool => ValType::I64,
        F64 | F32 => ValType::F64,
        other => return Err(WasmEmitError::Unsupported(format!("type {:?}", other))),
    })
}

/// Coerce a `HirConstant` case discriminator into an i64 for the
/// switch dispatcher. Mirrors the `emit_constant` ABI — every
/// integer / bool widens to i64; floats and other types aren't
/// switchable today (rejected by HIR lowering long before us, but
/// the explicit error here keeps the contract observable).
fn case_constant_to_i64(c: &HirConstant) -> Result<i64> {
    Ok(match c {
        HirConstant::I64(n) => *n,
        HirConstant::I32(n) => *n as i64,
        HirConstant::I16(n) => *n as i64,
        HirConstant::I8(n) => *n as i64,
        HirConstant::U64(n) => *n as i64,
        HirConstant::U32(n) => *n as i64,
        HirConstant::U16(n) => *n as i64,
        HirConstant::U8(n) => *n as i64,
        HirConstant::Bool(b) => {
            if *b {
                1
            } else {
                0
            }
        }
        other => {
            return Err(WasmEmitError::Unsupported(format!(
                "Switch case constant {:?} (only integer/bool case discriminators supported)",
                other
            )))
        }
    })
}

fn emit_binary_op(out: &mut Vec<WasmInst<'static>>, op: BinaryOp, ty: &HirType) -> Result<()> {
    let is_float = matches!(ty, HirType::F32 | HirType::F64);
    let inst = match (op, is_float) {
        (BinaryOp::Add, false) => WasmInst::I64Add,
        (BinaryOp::Sub, false) => WasmInst::I64Sub,
        (BinaryOp::Mul, false) => WasmInst::I64Mul,
        (BinaryOp::Div, false) => {
            // Default to signed; HIR carries signed-vs-unsigned
            // through the type, so we'll fork on `ty`'s
            // signedness once unsigned division is exercised by
            // an actual test case.
            WasmInst::I64DivS
        }
        (BinaryOp::Rem, false) => WasmInst::I64RemS,
        (BinaryOp::Add, true) => WasmInst::F64Add,
        (BinaryOp::Sub, true) => WasmInst::F64Sub,
        (BinaryOp::Mul, true) => WasmInst::F64Mul,
        (BinaryOp::Div, true) => WasmInst::F64Div,
        (op, _) => {
            return Err(WasmEmitError::Unsupported(format!(
                "binary op {:?} on {:?}",
                op, ty
            )))
        }
    };
    out.push(inst);
    Ok(())
}

/// Compress a `[ValType; N]` list into the (count, type) runs that
/// wasm's locals section expects.
fn compact_locals(types: &[ValType]) -> Vec<(u32, ValType)> {
    let mut out: Vec<(u32, ValType)> = Vec::new();
    for t in types {
        if let Some(last) = out.last_mut() {
            if last.1 == *t {
                last.0 += 1;
                continue;
            }
        }
        out.push((1, *t));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirFunctionSignature, HirParam, HirValue, ParamAttributes};
    use std::collections::HashSet;
    use zyntax_typed_ast::InternedString;

    fn add_value(func: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
        let id = HirId::new();
        func.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind,
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }

    #[test]
    fn emits_trivial_return_constant() {
        // `def trivial(): i64 { return 7 }`
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("trivial"), sig);
        let seven = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(7)),
        );
        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.terminator = HirTerminator::Return {
            values: vec![seven],
        };

        let m = WasmBackend::new()
            .compile_function(&func)
            .expect("emit trivial");
        m.validate_full().expect("module structurally valid");
    }

    #[test]
    fn emits_two_param_add() {
        // `def add(a: i64, b: i64): i64 { return a + b }`
        let p0 = HirId::new();
        let p1 = HirId::new();
        let sig = HirFunctionSignature {
            params: vec![
                HirParam {
                    id: p0,
                    name: InternedString::new_global("a"),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                },
                HirParam {
                    id: p1,
                    name: InternedString::new_global("b"),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                },
            ],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("add"), sig);
        // Mirror what SSA builder does: a separate value entry
        // per param with `HirValueKind::Parameter(i)`.
        func.values.insert(
            p0,
            HirValue {
                id: p0,
                ty: HirType::I64,
                kind: HirValueKind::Parameter(0),
                uses: HashSet::new(),
                span: None,
            },
        );
        func.values.insert(
            p1,
            HirValue {
                id: p1,
                ty: HirType::I64,
                kind: HirValueKind::Parameter(1),
                uses: HashSet::new(),
                span: None,
            },
        );
        let sum = add_value(&mut func, HirType::I64, HirValueKind::Instruction);
        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.instructions.push(HirInstruction::Binary {
            result: sum,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: p0,
            right: p1,
        });
        entry.terminator = HirTerminator::Return { values: vec![sum] };

        let m = WasmBackend::new()
            .compile_function(&func)
            .expect("emit add");
        m.validate_full().expect("module structurally valid");
    }

    /// Build an empty extra block keyed by `id` and graft it onto
    /// `func.blocks`. Helper for control-flow tests below.
    fn add_block(
        func: &mut HirFunction,
        id: HirId,
        instructions: Vec<HirInstruction>,
        terminator: HirTerminator,
    ) {
        func.blocks.insert(
            id,
            HirBlock {
                id,
                label: None,
                phis: Vec::new(),
                instructions,
                terminator,
                dominance_frontier: HashSet::new(),
                predecessors: Vec::new(),
                successors: Vec::new(),
            },
        );
    }

    /// `def chained(): i64 { goto B; B: return 13 }` — entry block
    /// unconditionally branches to a successor that returns. Tests
    /// the dispatch-loop lowering with two blocks linked by a
    /// `Branch` terminator and no phis.
    #[test]
    fn emits_two_block_branch_chain() {
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("chained"), sig);
        let thirteen = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(13)),
        );

        // Block B: returns the constant.
        let b_id = HirId::new();
        add_block(
            &mut func,
            b_id,
            Vec::new(),
            HirTerminator::Return {
                values: vec![thirteen],
            },
        );

        // Entry block: unconditional branch to B.
        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.terminator = HirTerminator::Branch { target: b_id };

        let m = WasmBackend::new()
            .compile_function(&func)
            .expect("emit two-block chain");
        m.validate_full().expect("module structurally valid");
    }

    /// `def pick(cond: bool): i64 { if cond then return 1 else return 2 }`
    /// CondBranch with both arms returning — exercises `If`/`Else`
    /// lowering of the two-target terminator and the I32WrapI64
    /// narrowing of the boolean condition.
    #[test]
    fn emits_cond_branch_with_two_returning_arms() {
        let cond_id = HirId::new();
        let sig = HirFunctionSignature {
            params: vec![HirParam {
                id: cond_id,
                name: InternedString::new_global("cond"),
                ty: HirType::Bool,
                attributes: ParamAttributes::default(),
            }],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("pick"), sig);
        // Param value-id with `Parameter(0)` kind so the use site
        // can `local.get` it.
        func.values.insert(
            cond_id,
            HirValue {
                id: cond_id,
                ty: HirType::Bool,
                kind: HirValueKind::Parameter(0),
                uses: HashSet::new(),
                span: None,
            },
        );

        let one = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(1)),
        );
        let two = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(2)),
        );

        let then_id = HirId::new();
        let else_id = HirId::new();
        add_block(
            &mut func,
            then_id,
            Vec::new(),
            HirTerminator::Return { values: vec![one] },
        );
        add_block(
            &mut func,
            else_id,
            Vec::new(),
            HirTerminator::Return { values: vec![two] },
        );

        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.terminator = HirTerminator::CondBranch {
            condition: cond_id,
            true_target: then_id,
            false_target: else_id,
        };

        let m = WasmBackend::new()
            .compile_function(&func)
            .expect("emit cond-branch");
        m.validate_full().expect("module structurally valid");
    }

    /// `def pick_phi(cond: bool): i64 {
    ///     bb_entry: if cond -> bb_then; else -> bb_else
    ///     bb_then:  Branch bb_join (phi.incoming += (one, bb_then))
    ///     bb_else:  Branch bb_join (phi.incoming += (two, bb_else))
    ///     bb_join:  result = phi(bb_then -> one, bb_else -> two); return result
    /// }`
    ///
    /// Tests out-of-SSA lowering: the join block's phi result must be
    /// written by predecessor-edge moves (`local.set` before the
    /// `br`) so the destination block reads the value matching the
    /// edge we came from.
    #[test]
    fn emits_phi_at_join_block_via_predecessor_edge_moves() {
        use crate::hir::HirPhi;
        let cond_id = HirId::new();
        let sig = HirFunctionSignature {
            params: vec![HirParam {
                id: cond_id,
                name: InternedString::new_global("cond"),
                ty: HirType::Bool,
                attributes: ParamAttributes::default(),
            }],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("pick_phi"), sig);
        func.values.insert(
            cond_id,
            HirValue {
                id: cond_id,
                ty: HirType::Bool,
                kind: HirValueKind::Parameter(0),
                uses: HashSet::new(),
                span: None,
            },
        );
        let one = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(1)),
        );
        let two = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(2)),
        );
        let phi_result = add_value(&mut func, HirType::I64, HirValueKind::Instruction);

        let then_id = HirId::new();
        let else_id = HirId::new();
        let join_id = HirId::new();
        add_block(
            &mut func,
            then_id,
            Vec::new(),
            HirTerminator::Branch { target: join_id },
        );
        add_block(
            &mut func,
            else_id,
            Vec::new(),
            HirTerminator::Branch { target: join_id },
        );
        // Join block carries the phi.
        let mut join_block = crate::hir::HirBlock {
            id: join_id,
            label: None,
            phis: vec![HirPhi {
                result: phi_result,
                ty: HirType::I64,
                incoming: vec![(one, then_id), (two, else_id)],
            }],
            instructions: Vec::new(),
            terminator: HirTerminator::Return {
                values: vec![phi_result],
            },
            dominance_frontier: HashSet::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        };
        // Insert join block.
        func.blocks.insert(join_id, join_block.clone());
        // Suppress unused-mut warning in some compilers.
        let _ = &mut join_block;

        // Entry CondBranch.
        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.terminator = HirTerminator::CondBranch {
            condition: cond_id,
            true_target: then_id,
            false_target: else_id,
        };

        let m = WasmBackend::new()
            .compile_function(&func)
            .expect("emit phi-join function");
        m.validate_full().expect("module structurally valid");
    }

    /// `def switch_demo(tag: i64): i64`
    ///
    /// ```text
    /// entry: switch tag {
    ///     0 -> bb_zero,
    ///     1 -> bb_one,
    ///     default -> bb_other,
    /// }
    /// bb_zero:  return 100
    /// bb_one:   return 200
    /// bb_other: return 999
    /// ```
    ///
    /// Validates the `Switch` lowering: three `if cmp; phi+set+br;
    /// end` blocks for the explicit cases, plus a default that fires
    /// when none match.
    #[test]
    fn emits_switch_three_way() {
        let tag_id = HirId::new();
        let sig = HirFunctionSignature {
            params: vec![HirParam {
                id: tag_id,
                name: InternedString::new_global("tag"),
                ty: HirType::I64,
                attributes: ParamAttributes::default(),
            }],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("switch_demo"), sig);
        func.values.insert(
            tag_id,
            HirValue {
                id: tag_id,
                ty: HirType::I64,
                kind: HirValueKind::Parameter(0),
                uses: HashSet::new(),
                span: None,
            },
        );

        let hundred = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(100)),
        );
        let two_hundred = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(200)),
        );
        let nine_nine_nine = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(999)),
        );

        let bb_zero = HirId::new();
        let bb_one = HirId::new();
        let bb_other = HirId::new();
        add_block(
            &mut func,
            bb_zero,
            Vec::new(),
            HirTerminator::Return {
                values: vec![hundred],
            },
        );
        add_block(
            &mut func,
            bb_one,
            Vec::new(),
            HirTerminator::Return {
                values: vec![two_hundred],
            },
        );
        add_block(
            &mut func,
            bb_other,
            Vec::new(),
            HirTerminator::Return {
                values: vec![nine_nine_nine],
            },
        );

        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.terminator = HirTerminator::Switch {
            value: tag_id,
            default: bb_other,
            cases: vec![
                (HirConstant::I64(0), bb_zero),
                (HirConstant::I64(1), bb_one),
            ],
        };

        let m = WasmBackend::new()
            .compile_function(&func)
            .expect("emit Switch");
        m.validate_full().expect("module structurally valid");
    }

    /// `def double_via_extern(x: i64): i64 { return ext_double(x) }`
    ///
    /// Verifies that:
    ///   - The wasm module declares an import named `"ext_double@1"`
    ///     under module `"extern"`, with `(i64) -> i64` signature.
    ///   - The function call site emits `local.get $x; call <imp>;
    ///     local.set $result; local.get $result; return`.
    ///   - The result type lines up with the function's return
    ///     signature so wasmparser accepts the module.
    #[test]
    fn emits_call_to_extern_symbol() {
        let x_id = HirId::new();
        let sig = HirFunctionSignature {
            params: vec![HirParam {
                id: x_id,
                name: InternedString::new_global("x"),
                ty: HirType::I64,
                attributes: ParamAttributes::default(),
            }],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("double_via_extern"), sig);
        func.values.insert(
            x_id,
            HirValue {
                id: x_id,
                ty: HirType::I64,
                kind: HirValueKind::Parameter(0),
                uses: HashSet::new(),
                span: None,
            },
        );

        // Result of the call.
        let call_result = add_value(&mut func, HirType::I64, HirValueKind::Instruction);
        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.instructions.push(HirInstruction::Call {
            result: Some(call_result),
            callee: HirCallable::Symbol("ext_double".to_string()),
            args: vec![x_id],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        entry.terminator = HirTerminator::Return {
            values: vec![call_result],
        };

        let m = WasmBackend::new()
            .compile_function(&func)
            .expect("emit extern-call function");
        m.validate_full().expect("module structurally valid");

        // The import we emit should be discoverable via
        // `wasmparser::Parser` — sanity check that the suffix-name
        // convention round-trips. The host's JS-side dispatcher
        // splits on `@` to recover (name, arity).
        let parser = wasmparser::Parser::new(0);
        let mut found_import = false;
        for payload in parser.parse_all(&m.bytes) {
            if let wasmparser::Payload::ImportSection(reader) = payload.unwrap() {
                for import in reader {
                    let import = import.unwrap();
                    if import.module == "extern" && import.name == "ext_double@1" {
                        found_import = true;
                    }
                }
            }
        }
        assert!(
            found_import,
            "expected `(import \"extern\" \"ext_double@1\")` in emitted module"
        );
    }

    /// Phase E.6.1: higher-arity extern calls. The JS dispatcher
    /// (zynml.mjs `makeExternDispatcher`) covers arities 0–8 today;
    /// the wasm bytes need to import the same `name@N` convention so
    /// the host can route the call. This test asserts arity-5 round-
    /// trips through the import-section encoding identically to arity
    /// 1 (covered above).
    #[test]
    fn emits_call_to_extern_symbol_arity_5() {
        let param_ids: Vec<HirId> = (0..5).map(|_| HirId::new()).collect();
        let sig = HirFunctionSignature {
            params: param_ids
                .iter()
                .enumerate()
                .map(|(i, id)| HirParam {
                    id: *id,
                    name: InternedString::new_global(&format!("a{}", i)),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                })
                .collect(),
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("call_5"), sig);
        for (idx, id) in param_ids.iter().enumerate() {
            func.values.insert(
                *id,
                HirValue {
                    id: *id,
                    ty: HirType::I64,
                    kind: HirValueKind::Parameter(idx as u32),
                    uses: HashSet::new(),
                    span: None,
                },
            );
        }
        let call_result = add_value(&mut func, HirType::I64, HirValueKind::Instruction);
        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.instructions.push(HirInstruction::Call {
            result: Some(call_result),
            callee: HirCallable::Symbol("ext_five_arg".to_string()),
            args: param_ids.clone(),
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        entry.terminator = HirTerminator::Return {
            values: vec![call_result],
        };

        let m = WasmBackend::new()
            .compile_function(&func)
            .expect("emit arity-5 extern-call function");
        m.validate_full().expect("module structurally valid");

        let parser = wasmparser::Parser::new(0);
        let mut found_import = false;
        for payload in parser.parse_all(&m.bytes) {
            if let wasmparser::Payload::ImportSection(reader) = payload.unwrap() {
                for import in reader {
                    let import = import.unwrap();
                    if import.module == "extern" && import.name == "ext_five_arg@5" {
                        found_import = true;
                    }
                }
            }
        }
        assert!(
            found_import,
            "expected `(import \"extern\" \"ext_five_arg@5\")` in emitted module — \
             the JS dispatcher relies on the `name@arity` convention to route \
             through `_zyntax_call_extern_5`"
        );
    }

    /// `def load_then_store(addr: i64, value: i64): i64 {
    ///     let cur = *addr;        // Load
    ///     *addr = cur + value;    // Add + Store
    ///     return cur;
    /// }`
    ///
    /// Validates that Load/Store lower correctly AND that the
    /// emitted module imports `(host, memory)` so the JIT'd code
    /// can read/write the host's linear memory.
    #[test]
    fn emits_load_and_store_with_memory_import() {
        let addr_id = HirId::new();
        let value_id = HirId::new();
        let sig = HirFunctionSignature {
            params: vec![
                HirParam {
                    id: addr_id,
                    name: InternedString::new_global("addr"),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                },
                HirParam {
                    id: value_id,
                    name: InternedString::new_global("value"),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                },
            ],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("load_then_store"), sig);
        for (idx, id) in [(0u32, addr_id), (1, value_id)] {
            func.values.insert(
                id,
                HirValue {
                    id,
                    ty: HirType::I64,
                    kind: HirValueKind::Parameter(idx),
                    uses: HashSet::new(),
                    span: None,
                },
            );
        }
        let cur = add_value(&mut func, HirType::I64, HirValueKind::Instruction);
        let sum = add_value(&mut func, HirType::I64, HirValueKind::Instruction);
        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.instructions.push(HirInstruction::Load {
            result: cur,
            ty: HirType::I64,
            ptr: addr_id,
            align: 8,
            volatile: false,
        });
        entry.instructions.push(HirInstruction::Binary {
            result: sum,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: cur,
            right: value_id,
        });
        entry.instructions.push(HirInstruction::Store {
            value: sum,
            ptr: addr_id,
            align: 8,
            volatile: false,
        });
        entry.terminator = HirTerminator::Return { values: vec![cur] };

        let m = WasmBackend::new()
            .compile_function(&func)
            .expect("emit load/store function");
        m.validate_full().expect("module structurally valid");

        // The module must declare a memory import so the host can
        // pass its memory in at instantiate time.
        let parser = wasmparser::Parser::new(0);
        let mut found_memory_import = false;
        for payload in parser.parse_all(&m.bytes) {
            if let wasmparser::Payload::ImportSection(reader) = payload.unwrap() {
                for import in reader {
                    let import = import.unwrap();
                    if import.module == "host"
                        && import.name == "memory"
                        && matches!(import.ty, wasmparser::TypeRef::Memory(_))
                    {
                        found_memory_import = true;
                    }
                }
            }
        }
        assert!(
            found_memory_import,
            "expected `(import \"host\" \"memory\" (memory ...))` declaration"
        );
    }

    /// Internal HIR-function calls (`HirCallable::Function`) still
    /// bail because cross-function JIT dispatch isn't wired yet.
    /// Asserts the documented Unsupported path.
    #[test]
    fn internal_function_call_bails_cleanly() {
        let other_func = HirId::new();
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("calls_other"), sig);
        let call_result = add_value(&mut func, HirType::I64, HirValueKind::Instruction);
        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.instructions.push(HirInstruction::Call {
            result: Some(call_result),
            callee: HirCallable::Function(other_func),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        entry.terminator = HirTerminator::Return {
            values: vec![call_result],
        };

        match WasmBackend::new().compile_function(&func) {
            Err(WasmEmitError::Unsupported(msg)) => {
                assert!(
                    msg.contains("internal HIR function"),
                    "expected internal-call bail, got: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {:?}", other),
        }
    }

    /// Inter-phi cycle (one phi's result feeds another phi's
    /// incoming in the SAME block) still bails — that case needs
    /// parallel-copy resolution with temporaries and isn't
    /// implemented. The bail keeps the interpreter-fallback contract
    /// documented in code.
    #[test]
    fn phi_cycle_bails_cleanly() {
        use crate::hir::HirPhi;
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("phi_cycle"), sig);
        let one = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(1)),
        );
        let pred = HirId::new();
        add_block(
            &mut func,
            pred,
            Vec::new(),
            HirTerminator::Return { values: vec![one] },
        );

        let phi_a = HirId::new();
        let phi_b = HirId::new();
        let join_id = HirId::new();
        let join_block = crate::hir::HirBlock {
            id: join_id,
            label: None,
            phis: vec![
                HirPhi {
                    result: phi_a,
                    ty: HirType::I64,
                    // phi_a's incoming references phi_b's result —
                    // forms a cycle.
                    incoming: vec![(phi_b, pred)],
                },
                HirPhi {
                    result: phi_b,
                    ty: HirType::I64,
                    incoming: vec![(one, pred)],
                },
            ],
            instructions: Vec::new(),
            terminator: HirTerminator::Return {
                values: vec![phi_a],
            },
            dominance_frontier: HashSet::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        };
        func.blocks.insert(join_id, join_block);

        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.terminator = HirTerminator::Branch { target: join_id };

        match WasmBackend::new().compile_function(&func) {
            Err(WasmEmitError::Unsupported(msg)) => {
                assert!(
                    msg.contains("phi cycle"),
                    "expected phi-cycle bail, got: {msg}"
                );
            }
            other => panic!("expected Unsupported(phi cycle), got {:?}", other),
        }
    }
}
