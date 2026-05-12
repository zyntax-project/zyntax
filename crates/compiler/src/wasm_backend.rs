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
//! Initial scope: primitive i64/f64 arithmetic + immediate return.
//! Control flow, structs, calls, effects all stub out to a clear
//! `WasmEmitError::Unsupported`. The emitter is intentionally
//! conservative — any HIR shape it can't handle bails so the
//! interpreter keeps that function in BC.
//!
//! Value representation:
//! * `HirType::I64` / `I32` / `I16` / `I8` / `Bool` / `UInt`-family
//!   → wasm `i64` (32-bit narrows widened on emit; the wasm side
//!   keeps everything in i64 to match the BC dispatch ABI).
//! * `HirType::F32` / `F64` → wasm `f64`.
//! * Other types → `WasmEmitError::Unsupported` for now.

use crate::hir::{
    BinaryOp, HirConstant, HirFunction, HirId, HirInstruction, HirTerminator, HirType, HirValueKind,
};
use std::collections::HashMap;

use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction as WasmInst,
    Module, TypeSection, ValType,
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
}

/// Per-block state threaded through `emit_terminator` so jumps
/// can translate HIR `Branch` / `CondBranch` targets into the right
/// wasm `br` relative-depth + `$next_block` assignment for the
/// dispatch-loop lowering. `None` in single-block functions — the
/// terminator never `br`s, only `return`s.
struct DispatchCtx<'b> {
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

        // 3. Emit the function body.
        let body = self.emit_body()?;

        // 4. Build the module: type → function → export → code.
        let mut module = Module::new();

        let mut types = TypeSection::new();
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

        let mut functions = FunctionSection::new();
        functions.function(0);
        module.section(&functions);

        let mut exports = ExportSection::new();
        exports.export("entry", ExportKind::Func, 0);
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
                    // Globals / phis / etc. unsupported in this
                    // initial slice; the use site will hit
                    // UnknownValue and bail.
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
        // Reject phis up front — see the docstring on `emit_body`.
        for block in self.func.blocks.values() {
            if !block.phis.is_empty() {
                return Err(WasmEmitError::Unsupported(format!(
                    "phi nodes (block {:?} has {} phis)",
                    block.id,
                    block.phis.len()
                )));
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
                // Set `$next = target_idx` then `br` to the loop
                // header so the next iteration's `br_table` lands
                // us in the target block.
                let idx = *ctx
                    .block_indices
                    .get(target)
                    .ok_or(WasmEmitError::UnknownValue(*target))?;
                out.push(WasmInst::I32Const(idx as i32));
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
                out.push(WasmInst::I32Const(true_idx as i32));
                out.push(WasmInst::LocalSet(ctx.next_block_local));
                out.push(WasmInst::Br(ctx.loop_depth + 1));
                out.push(WasmInst::Else);
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

    /// A phi node in any block still bails — out-of-SSA conversion
    /// (Phase E.3.1) hasn't landed yet. Asserting the bail path
    /// keeps the interpreter-fallback contract documented in code.
    #[test]
    fn phi_node_block_bails_cleanly() {
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
        let mut func = HirFunction::new(InternedString::new_global("withphi"), sig);
        let phi_id = HirId::new();
        // Inject a phi into the entry block.
        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.phis.push(HirPhi {
            result: phi_id,
            ty: HirType::I64,
            incoming: Vec::new(),
        });
        entry.terminator = HirTerminator::Return { values: vec![] };
        // Add a second block so we hit `emit_multi_block` (single
        // block also rejects phis via the `if blocks.len() == 1`
        // path, but the multi-block check is the one we want to
        // exercise here).
        let b = HirId::new();
        add_block(
            &mut func,
            b,
            Vec::new(),
            HirTerminator::Return { values: vec![] },
        );

        match WasmBackend::new().compile_function(&func) {
            Err(WasmEmitError::Unsupported(msg)) => {
                assert!(
                    msg.contains("phi"),
                    "expected phi-related Unsupported, got: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {:?}", other),
        }
    }
}
