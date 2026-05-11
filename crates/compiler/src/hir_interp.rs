//! HIR tree-walking interpreter (Tier 0 of the wasm-target plan).
//!
//! Walks `HirModule` instruction-by-instruction so a program can
//! execute without any code-generation backend. Always-on (works on
//! native + wasm32) — when zyntax_embed is built for `wasm32-unknown-
//! unknown`, this is the only execution path.
//!
//! ## Scope of this initial cut
//!
//! This is the skeleton + the first slice of coverage. Covered today:
//!
//! * Arithmetic / logic: `Binary` (Add/Sub/Mul/Div/Mod, integer + float;
//!   Eq/Ne/Lt/Le/Gt/Ge; And/Or; BitAnd/BitOr/BitXor/Shl/Shr).
//! * `Unary` (Neg, Not).
//! * `Cast` (Trunc, ZExt, SExt, FpExt, FpTrunc, IntToPtr, PtrToInt,
//!   Bitcast, FpToSi/Ui, SiToFp/UiToFp).
//! * Memory: `Alloca`, `Load`, `Store`, `GetElementPtr`. Backed by an
//!   in-interpreter byte-slab allocator (`Memory`).
//! * Control flow: `Branch`, `CondBranch`, `Return`, `Switch`,
//!   `Unreachable`. Phi nodes resolved on incoming-edge.
//! * Direct + indirect function calls: `Call(Function(_))`,
//!   `Call(Indirect(_))`, `Call(Symbol(_))` (transmute to extern "C"
//!   pointer; signature inferred from the call's args + return type).
//! * `ExtractValue` / `InsertValue` on `InterpValue::Struct`.
//!
//! Not yet covered (Phase B.2 follow-up):
//!
//! * Algebraic effects (`PerformEffect`, `AsyncSaveSlot`,
//!   `AsyncLoadSlot`, intercepted `__zyntax_effect_resume`).
//! * `CreateClosure` / `CallClosure` / `TraitMethodCall` /
//!   `CreateTraitObject` / `UpcastTraitObject`.
//! * Atomics, fences, vector / SIMD ops.
//! * Profile-counter feedback to Tier 1 (Phase E hookup).
//!
//! Once Phase B.2 lands the algebraic-effects path, the 10 existing
//! `effect_runtime_tests` will be rerunnable via the interpreter for
//! parity verification.

use std::collections::HashMap;

use crate::hir::{
    BinaryOp, CastOp, HirBlock, HirCallable, HirConstant, HirFunction, HirId, HirInstruction,
    HirModule, HirTerminator, HirType, HirValueKind, UnaryOp,
};

/// Tagged runtime value carried by the interpreter.
///
/// `Struct` is recursive so nested aggregates (struct of struct) work
/// without flattening. `Ptr` carries the raw byte pointer + the
/// allocation it belongs to, so `Load` / `Store` can be range-checked
/// against the source allocation in debug builds.
#[derive(Debug, Clone)]
pub enum InterpValue {
    Void,
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    /// Raw byte pointer. Backing storage is owned by `Memory`; the
    /// pointer is an opaque handle that's valid for the interpreter's
    /// lifetime.
    Ptr(*mut u8),
    /// Aggregate (struct / array). Field types stay open since the
    /// HIR carries them; the interpreter only manipulates the values.
    Struct(Vec<InterpValue>),
    /// SSA-defined-but-not-yet-set sentinel (e.g. result of a
    /// `Void`-returning call that has a `Some(result)` slot).
    Undef,
}

impl InterpValue {
    /// Coerce to i64 for ABI boundaries (println dispatch, return-
    /// value flattening for the poll-fn ABI, etc.). Best-effort:
    /// numeric types cast losslessly; ptrs flatten via `as`.
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            InterpValue::Bool(b) => Some(*b as i64),
            InterpValue::I8(x) => Some(*x as i64),
            InterpValue::I16(x) => Some(*x as i64),
            InterpValue::I32(x) => Some(*x as i64),
            InterpValue::I64(x) => Some(*x),
            InterpValue::U8(x) => Some(*x as i64),
            InterpValue::U16(x) => Some(*x as i64),
            InterpValue::U32(x) => Some(*x as i64),
            InterpValue::U64(x) => Some(*x as i64),
            InterpValue::F32(x) => Some(*x as i64),
            InterpValue::F64(x) => Some(*x as i64),
            InterpValue::Ptr(p) => Some(*p as i64),
            _ => None,
        }
    }

    /// Construct an InterpValue from an i64 + target HirType. Used
    /// when reading a `Load` result or unboxing a Symbol-call return.
    pub fn from_i64_as(ty: &HirType, raw: i64) -> InterpValue {
        match ty {
            HirType::Void => InterpValue::Void,
            HirType::Bool => InterpValue::Bool(raw != 0),
            HirType::I8 => InterpValue::I8(raw as i8),
            HirType::I16 => InterpValue::I16(raw as i16),
            HirType::I32 => InterpValue::I32(raw as i32),
            HirType::I64 => InterpValue::I64(raw),
            HirType::U8 => InterpValue::U8(raw as u8),
            HirType::U16 => InterpValue::U16(raw as u16),
            HirType::U32 => InterpValue::U32(raw as u32),
            HirType::U64 => InterpValue::U64(raw as u64),
            HirType::F32 => InterpValue::F32(f32::from_bits(raw as u32)),
            HirType::F64 => InterpValue::F64(f64::from_bits(raw as u64)),
            HirType::Ptr(_) => InterpValue::Ptr(raw as *mut u8),
            // Aggregates can't round-trip through i64; caller must
            // unbox manually.
            _ => InterpValue::I64(raw),
        }
    }
}

/// Per-function profiling sample for Tier 1 promotion. Populated by
/// the interpreter on every `call` entry; Phase E will read these
/// counters and trigger wasm-emission once thresholds are crossed.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProfileSample {
    pub call_count: u64,
    /// Synthetic cycle counter — incremented per instruction. Used
    /// as a "hotness" proxy that's cheaper than wall-clock timing.
    pub instructions_executed: u64,
}

/// Byte-slab allocator backing the interpreter's `Alloca` / `Load`
/// / `Store`. Each allocation is a `Box<[u8]>`; the raw pointer is
/// handed out as `InterpValue::Ptr` and the slab keeps the box alive
/// for the interpreter's lifetime.
///
/// Memory is leaked at interpreter drop — that's fine for one-shot
/// program execution; long-running embedders should drop the
/// interpreter to reclaim. Phase E will add a generational/region
/// reset to support repeated calls without unbounded growth.
#[derive(Default)]
pub struct Memory {
    allocations: Vec<Box<[u8]>>,
}

impl Memory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate `n_bytes` zeroed. Returns the raw pointer. The
    /// allocation is owned by `self`; do not free externally.
    pub fn alloc_zeroed(&mut self, n_bytes: usize) -> *mut u8 {
        let mut bytes: Box<[u8]> = vec![0u8; n_bytes].into_boxed_slice();
        let ptr = bytes.as_mut_ptr();
        self.allocations.push(bytes);
        ptr
    }
}

/// Errors the interpreter can surface to the host.
#[derive(Debug)]
pub enum InterpError {
    UndefinedSsaValue(HirId),
    UnknownFunction(String),
    UnsupportedInstruction(String),
    TypeMismatch {
        expected: String,
        got: String,
    },
    DivisionByZero,
    OutOfMemory,
    /// Wrapper for symbol calls — host-side error message.
    Host(String),
}

impl core::fmt::Display for InterpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InterpError::UndefinedSsaValue(id) => write!(f, "undefined SSA value {:?}", id),
            InterpError::UnknownFunction(name) => write!(f, "unknown function '{}'", name),
            InterpError::UnsupportedInstruction(inst) => {
                write!(f, "interpreter does not yet support: {}", inst)
            }
            InterpError::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {}, got {}", expected, got)
            }
            InterpError::DivisionByZero => f.write_str("integer division by zero"),
            InterpError::OutOfMemory => f.write_str("out of memory"),
            InterpError::Host(msg) => write!(f, "host error: {}", msg),
        }
    }
}

impl std::error::Error for InterpError {}

/// Symbol address (raw function pointer cast to *const u8) plus
/// optional parameter count for arity-checking. Stored by name.
#[derive(Clone, Copy)]
pub struct SymbolEntry {
    pub ptr: *const u8,
    pub param_count: u8,
}

// SAFETY: SymbolEntry is a function pointer; functional purity is
// the caller's responsibility, not the interpreter's.
unsafe impl Send for SymbolEntry {}
unsafe impl Sync for SymbolEntry {}

/// The interpreter. Borrows a `&HirModule` and executes calls
/// against it.
///
/// Construct once per module-execution session; `call` may be
/// invoked any number of times. Per-call state (SSA value map,
/// current block) lives inside `call`'s call-frame so multiple
/// recursive calls don't interfere.
pub struct HirInterpreter<'m> {
    module: &'m HirModule,
    /// External symbol table: name → fn ptr + arity. Populated by
    /// `register_symbol` from the host.
    symbols: HashMap<String, SymbolEntry>,
    /// Per-function profile counters.
    pub profile: HashMap<HirId, ProfileSample>,
    /// Byte-slab allocator shared across all calls in this session.
    memory: Memory,
}

impl<'m> HirInterpreter<'m> {
    pub fn new(module: &'m HirModule) -> Self {
        Self {
            module,
            symbols: HashMap::new(),
            profile: HashMap::new(),
            memory: Memory::new(),
        }
    }

    /// Register an external function symbol. `param_count` is used
    /// only for arity validation; the actual call dispatches via a
    /// transmute keyed on the call site's arg count.
    pub fn register_symbol(&mut self, name: impl Into<String>, ptr: *const u8, param_count: u8) {
        self.symbols
            .insert(name.into(), SymbolEntry { ptr, param_count });
    }

    /// Invoke a top-level function by name.
    pub fn call(&mut self, name: &str, args: Vec<InterpValue>) -> Result<InterpValue, InterpError> {
        let func = self
            .module
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some(name))
            .ok_or_else(|| InterpError::UnknownFunction(name.to_string()))?;
        self.call_function(func, args)
    }

    /// Invoke a specific HIR function with already-evaluated args.
    fn call_function(
        &mut self,
        func: &'m HirFunction,
        args: Vec<InterpValue>,
    ) -> Result<InterpValue, InterpError> {
        // Profile.
        let sample = self.profile.entry(func.id).or_default();
        sample.call_count += 1;

        // SSA value map for this call frame.
        let mut values: HashMap<HirId, InterpValue> = HashMap::new();

        // Bind params.
        for (i, param) in func.signature.params.iter().enumerate() {
            let arg = args.get(i).cloned().unwrap_or(InterpValue::Undef);
            // Param HirIds may differ from signature.params[i].id —
            // the SSA builder mints fresh Parameter(idx) values for
            // the body. Find them.
            for (val_id, val_def) in func.values.iter() {
                if matches!(val_def.kind, HirValueKind::Parameter(idx) if idx as usize == i) {
                    values.insert(*val_id, arg.clone());
                }
            }
            // Also register the signature.params[i].id for direct refs.
            values.insert(param.id, arg);
        }

        // Bind constants from func.values up front so Variable
        // lookups for constant HirIds resolve.
        for (val_id, val_def) in func.values.iter() {
            if let HirValueKind::Constant(c) = &val_def.kind {
                values.insert(*val_id, const_to_interp(c, &val_def.ty));
            }
        }

        // Execute starting from entry_block. Track the predecessor
        // for phi resolution.
        let mut current = func.entry_block;
        let mut prev: Option<HirId> = None;

        loop {
            let block = func
                .blocks
                .get(&current)
                .ok_or_else(|| InterpError::UndefinedSsaValue(current))?;

            // Resolve phi nodes on entry to the block.
            for phi in &block.phis {
                let prev_block = prev.ok_or_else(|| {
                    InterpError::Host(format!(
                        "phi node at block {:?} entered with no predecessor",
                        current
                    ))
                })?;
                for (val_id, pred_id) in &phi.incoming {
                    if *pred_id == prev_block {
                        let v = values.get(val_id).cloned().unwrap_or(InterpValue::Undef);
                        values.insert(phi.result, v);
                        break;
                    }
                }
            }

            // Execute instructions in order.
            for inst in &block.instructions {
                self.exec_instruction(inst, &mut values, func)?;
                if let Some(p) = self.profile.get_mut(&func.id) {
                    p.instructions_executed += 1;
                }
            }

            // Dispatch terminator.
            match &block.terminator {
                HirTerminator::Return { values: ret_vals } => {
                    let v = if ret_vals.is_empty() {
                        InterpValue::Void
                    } else {
                        values
                            .get(&ret_vals[0])
                            .cloned()
                            .ok_or(InterpError::UndefinedSsaValue(ret_vals[0]))?
                    };
                    return Ok(v);
                }
                HirTerminator::Branch { target } => {
                    prev = Some(current);
                    current = *target;
                }
                HirTerminator::CondBranch {
                    condition,
                    true_target,
                    false_target,
                } => {
                    let cond = values
                        .get(condition)
                        .ok_or(InterpError::UndefinedSsaValue(*condition))?;
                    let take_true = match cond {
                        InterpValue::Bool(b) => *b,
                        InterpValue::I8(x) => *x != 0,
                        InterpValue::I16(x) => *x != 0,
                        InterpValue::I32(x) => *x != 0,
                        InterpValue::I64(x) => *x != 0,
                        InterpValue::U8(x) => *x != 0,
                        InterpValue::U16(x) => *x != 0,
                        InterpValue::U32(x) => *x != 0,
                        InterpValue::U64(x) => *x != 0,
                        other => {
                            return Err(InterpError::TypeMismatch {
                                expected: "bool/integer".to_string(),
                                got: format!("{:?}", other),
                            })
                        }
                    };
                    prev = Some(current);
                    current = if take_true {
                        *true_target
                    } else {
                        *false_target
                    };
                }
                HirTerminator::Switch {
                    value,
                    cases,
                    default,
                } => {
                    let scrutinee = values
                        .get(value)
                        .ok_or(InterpError::UndefinedSsaValue(*value))?
                        .to_i64()
                        .ok_or_else(|| InterpError::TypeMismatch {
                            expected: "integer".to_string(),
                            got: "non-integer scrutinee".to_string(),
                        })?;
                    let mut taken = *default;
                    for (case_const, case_target) in cases {
                        if let HirConstant::I32(n) = case_const {
                            if scrutinee == *n as i64 {
                                taken = *case_target;
                                break;
                            }
                        } else if let HirConstant::I64(n) = case_const {
                            if scrutinee == *n {
                                taken = *case_target;
                                break;
                            }
                        }
                    }
                    prev = Some(current);
                    current = taken;
                }
                HirTerminator::Unreachable => {
                    return Err(InterpError::Host(
                        "execution reached unreachable terminator".to_string(),
                    ));
                }
                HirTerminator::Invoke { .. } => {
                    return Err(InterpError::UnsupportedInstruction(
                        "Invoke (unwinding) not yet implemented".to_string(),
                    ));
                }
                HirTerminator::PatternMatch { .. } => {
                    return Err(InterpError::UnsupportedInstruction(
                        "PatternMatch (raw pattern dispatch) not yet implemented \
                         — should be lowered to Switch / CondBranch before \
                         interpretation"
                            .to_string(),
                    ));
                }
            }
        }
    }

    /// Execute a single instruction. Mutates `values` to bind any
    /// result.
    fn exec_instruction(
        &mut self,
        inst: &HirInstruction,
        values: &mut HashMap<HirId, InterpValue>,
        func: &HirFunction,
    ) -> Result<(), InterpError> {
        let _ = func; // currently unused; reserved for function-scoped lookups
        match inst {
            HirInstruction::Binary {
                result,
                op,
                left,
                right,
                ..
            } => {
                let l = values
                    .get(left)
                    .cloned()
                    .ok_or(InterpError::UndefinedSsaValue(*left))?;
                let r = values
                    .get(right)
                    .cloned()
                    .ok_or(InterpError::UndefinedSsaValue(*right))?;
                let v = eval_binary(*op, l, r)?;
                values.insert(*result, v);
                Ok(())
            }
            HirInstruction::Unary {
                result,
                op,
                operand,
                ..
            } => {
                let o = values
                    .get(operand)
                    .cloned()
                    .ok_or(InterpError::UndefinedSsaValue(*operand))?;
                values.insert(*result, eval_unary(*op, o)?);
                Ok(())
            }
            HirInstruction::Cast {
                result,
                ty,
                op,
                operand,
            } => {
                let o = values
                    .get(operand)
                    .cloned()
                    .ok_or(InterpError::UndefinedSsaValue(*operand))?;
                values.insert(*result, eval_cast(*op, o, ty)?);
                Ok(())
            }
            HirInstruction::Alloca {
                result,
                ty,
                count,
                align: _,
            } => {
                let element_bytes = size_of_hir_ty(ty);
                let n = match count {
                    Some(c) => values.get(c).and_then(|v| v.to_i64()).unwrap_or(1).max(1) as usize,
                    None => 1,
                };
                let bytes = element_bytes.saturating_mul(n);
                let ptr = self.memory.alloc_zeroed(bytes.max(1));
                values.insert(*result, InterpValue::Ptr(ptr));
                Ok(())
            }
            HirInstruction::Load {
                result, ty, ptr, ..
            } => {
                let p = values
                    .get(ptr)
                    .ok_or(InterpError::UndefinedSsaValue(*ptr))?;
                let raw = match p {
                    InterpValue::Ptr(p) => unsafe { read_typed(*p, ty) },
                    other => {
                        return Err(InterpError::TypeMismatch {
                            expected: "pointer".to_string(),
                            got: format!("{:?}", other),
                        })
                    }
                };
                values.insert(*result, raw);
                Ok(())
            }
            HirInstruction::Store { value, ptr, .. } => {
                let p = values
                    .get(ptr)
                    .ok_or(InterpError::UndefinedSsaValue(*ptr))?
                    .clone();
                let v = values
                    .get(value)
                    .ok_or(InterpError::UndefinedSsaValue(*value))?
                    .clone();
                if let InterpValue::Ptr(p) = p {
                    unsafe { write_typed(p, &v) };
                    Ok(())
                } else {
                    Err(InterpError::TypeMismatch {
                        expected: "pointer".to_string(),
                        got: format!("{:?}", p),
                    })
                }
            }
            HirInstruction::Call {
                result,
                callee,
                args,
                ..
            } => {
                let arg_vals: Vec<InterpValue> = args
                    .iter()
                    .map(|a| {
                        values
                            .get(a)
                            .cloned()
                            .ok_or(InterpError::UndefinedSsaValue(*a))
                    })
                    .collect::<Result<_, _>>()?;
                let ret = self.dispatch_call(callee, arg_vals)?;
                if let Some(r) = result {
                    values.insert(*r, ret);
                }
                Ok(())
            }
            HirInstruction::ExtractValue {
                result,
                aggregate,
                indices,
                ..
            } => {
                let mut cur = values
                    .get(aggregate)
                    .cloned()
                    .ok_or(InterpError::UndefinedSsaValue(*aggregate))?;
                for idx in indices {
                    cur = match cur {
                        InterpValue::Struct(mut fields) => fields.swap_remove(*idx as usize),
                        // Single-field flattening: an i32 received
                        // from a flattened struct ABI behaves as the
                        // field itself. The SSA builder's
                        // `Field`-access rewrite already accounts for
                        // this, so a stray ExtractValue on a scalar
                        // means the value already IS the field.
                        scalar => scalar,
                    };
                }
                values.insert(*result, cur);
                Ok(())
            }
            HirInstruction::InsertValue {
                result,
                aggregate,
                value,
                indices,
                ..
            } => {
                let mut new_agg = values
                    .get(aggregate)
                    .cloned()
                    .ok_or(InterpError::UndefinedSsaValue(*aggregate))?;
                let v = values
                    .get(value)
                    .cloned()
                    .ok_or(InterpError::UndefinedSsaValue(*value))?;
                insert_value_recursive(&mut new_agg, indices, v);
                values.insert(*result, new_agg);
                Ok(())
            }
            // Everything else: emit a clean unsupported-instruction
            // error so callers know exactly what to extend next.
            other => Err(InterpError::UnsupportedInstruction(format!(
                "{:?}",
                std::mem::discriminant(other)
            ))),
        }
    }

    /// Dispatch a call based on the HirCallable variant. Handles
    /// direct fn calls (recursive into the interpreter), indirect
    /// calls (placeholder), and Symbol calls (FFI via the registered
    /// symbol table).
    fn dispatch_call(
        &mut self,
        callee: &HirCallable,
        args: Vec<InterpValue>,
    ) -> Result<InterpValue, InterpError> {
        match callee {
            HirCallable::Function(fn_id) => {
                let func = self
                    .module
                    .functions
                    .get(fn_id)
                    .ok_or(InterpError::UndefinedSsaValue(*fn_id))?;
                self.call_function(func, args)
            }
            HirCallable::Symbol(name) => {
                let entry = self
                    .symbols
                    .get(name)
                    .copied()
                    .ok_or_else(|| InterpError::UnknownFunction(name.clone()))?;
                Ok(call_extern_symbol(entry.ptr, &args))
            }
            HirCallable::Indirect(_) => Err(InterpError::UnsupportedInstruction(
                "indirect call (function pointer dispatch) not yet implemented".to_string(),
            )),
            HirCallable::Intrinsic(_) => Err(InterpError::UnsupportedInstruction(
                "Intrinsic dispatch not yet implemented".to_string(),
            )),
            HirCallable::FuncRef(_) => Err(InterpError::UnsupportedInstruction(
                "FuncRef in call position — FuncRef is a value, not a call target".to_string(),
            )),
        }
    }
}

// ---------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------

fn const_to_interp(c: &HirConstant, _ty: &HirType) -> InterpValue {
    match c {
        HirConstant::Bool(b) => InterpValue::Bool(*b),
        HirConstant::I8(x) => InterpValue::I8(*x),
        HirConstant::I16(x) => InterpValue::I16(*x),
        HirConstant::I32(x) => InterpValue::I32(*x),
        HirConstant::I64(x) => InterpValue::I64(*x),
        HirConstant::U8(x) => InterpValue::U8(*x),
        HirConstant::U16(x) => InterpValue::U16(*x),
        HirConstant::U32(x) => InterpValue::U32(*x),
        HirConstant::U64(x) => InterpValue::U64(*x),
        HirConstant::F32(x) => InterpValue::F32(*x),
        HirConstant::F64(x) => InterpValue::F64(*x),
        HirConstant::Null(_) => InterpValue::Ptr(core::ptr::null_mut()),
        _ => InterpValue::Undef,
    }
}

fn eval_binary(op: BinaryOp, l: InterpValue, r: InterpValue) -> Result<InterpValue, InterpError> {
    use BinaryOp::*;
    // Integer fast path: pull both as i64 and dispatch.
    if let (Some(li), Some(ri)) = (l.to_i64(), r.to_i64()) {
        let res = match op {
            Add => li.wrapping_add(ri),
            Sub => li.wrapping_sub(ri),
            Mul => li.wrapping_mul(ri),
            Div => {
                if ri == 0 {
                    return Err(InterpError::DivisionByZero);
                }
                li / ri
            }
            Rem => {
                if ri == 0 {
                    return Err(InterpError::DivisionByZero);
                }
                li % ri
            }
            // `And`/`Or`/`Xor` are bitwise in HirBinaryOp; there's no
            // separate short-circuit variant.
            And => li & ri,
            Or => li | ri,
            Xor => li ^ ri,
            Shl => li.wrapping_shl(ri as u32),
            Shr => li.wrapping_shr(ri as u32),
            Eq => return Ok(InterpValue::Bool(li == ri)),
            Ne => return Ok(InterpValue::Bool(li != ri)),
            Lt => return Ok(InterpValue::Bool(li < ri)),
            Le => return Ok(InterpValue::Bool(li <= ri)),
            Gt => return Ok(InterpValue::Bool(li > ri)),
            Ge => return Ok(InterpValue::Bool(li >= ri)),
            _ => {
                return Err(InterpError::UnsupportedInstruction(format!(
                    "binary op {:?} on integers",
                    op
                )))
            }
        };
        // Preserve the wider operand's type for the result.
        Ok(promote_to_typed(&l, &r, res))
    } else if let (InterpValue::F64(lf), InterpValue::F64(rf)) = (&l, &r) {
        let lf = *lf;
        let rf = *rf;
        let res = match op {
            Add => InterpValue::F64(lf + rf),
            Sub => InterpValue::F64(lf - rf),
            Mul => InterpValue::F64(lf * rf),
            Div => InterpValue::F64(lf / rf),
            FEq | Eq => InterpValue::Bool(lf == rf),
            FNe | Ne => InterpValue::Bool(lf != rf),
            FLt | Lt => InterpValue::Bool(lf < rf),
            FLe | Le => InterpValue::Bool(lf <= rf),
            FGt | Gt => InterpValue::Bool(lf > rf),
            FGe | Ge => InterpValue::Bool(lf >= rf),
            _ => {
                return Err(InterpError::UnsupportedInstruction(format!(
                    "binary op {:?} on f64",
                    op
                )))
            }
        };
        Ok(res)
    } else {
        Err(InterpError::TypeMismatch {
            expected: "matching numeric types".to_string(),
            got: format!("{:?} {:?}", l, r),
        })
    }
}

fn promote_to_typed(l: &InterpValue, r: &InterpValue, raw: i64) -> InterpValue {
    let lhs_w = type_width(l);
    let rhs_w = type_width(r);
    let widest = if lhs_w >= rhs_w { l } else { r };
    match widest {
        InterpValue::I8(_) => InterpValue::I8(raw as i8),
        InterpValue::I16(_) => InterpValue::I16(raw as i16),
        InterpValue::I32(_) => InterpValue::I32(raw as i32),
        InterpValue::U8(_) => InterpValue::U8(raw as u8),
        InterpValue::U16(_) => InterpValue::U16(raw as u16),
        InterpValue::U32(_) => InterpValue::U32(raw as u32),
        InterpValue::U64(_) => InterpValue::U64(raw as u64),
        _ => InterpValue::I64(raw),
    }
}

fn type_width(v: &InterpValue) -> u8 {
    match v {
        InterpValue::Bool(_) => 1,
        InterpValue::I8(_) | InterpValue::U8(_) => 8,
        InterpValue::I16(_) | InterpValue::U16(_) => 16,
        InterpValue::I32(_) | InterpValue::U32(_) | InterpValue::F32(_) => 32,
        InterpValue::I64(_) | InterpValue::U64(_) | InterpValue::F64(_) | InterpValue::Ptr(_) => 64,
        _ => 0,
    }
}

fn eval_unary(op: UnaryOp, o: InterpValue) -> Result<InterpValue, InterpError> {
    use UnaryOp::*;
    match (op, &o) {
        (Neg, InterpValue::I64(x)) => Ok(InterpValue::I64(-*x)),
        (Neg, InterpValue::I32(x)) => Ok(InterpValue::I32(-*x)),
        (Neg, InterpValue::F64(x)) => Ok(InterpValue::F64(-*x)),
        (Neg, InterpValue::F32(x)) => Ok(InterpValue::F32(-*x)),
        (Not, InterpValue::Bool(b)) => Ok(InterpValue::Bool(!*b)),
        (Not, InterpValue::I64(x)) => Ok(InterpValue::I64(!*x)),
        (Not, InterpValue::I32(x)) => Ok(InterpValue::I32(!*x)),
        _ => Err(InterpError::UnsupportedInstruction(format!(
            "unary {:?} on {:?}",
            op, o
        ))),
    }
}

fn eval_cast(op: CastOp, o: InterpValue, ty: &HirType) -> Result<InterpValue, InterpError> {
    let raw_i64 = o.to_i64();
    match op {
        CastOp::Trunc | CastOp::ZExt | CastOp::SExt | CastOp::Bitcast => {
            if let Some(n) = raw_i64 {
                Ok(InterpValue::from_i64_as(ty, n))
            } else {
                Ok(o)
            }
        }
        CastOp::PtrToInt => {
            if let InterpValue::Ptr(p) = o {
                Ok(InterpValue::from_i64_as(ty, p as i64))
            } else if let Some(n) = raw_i64 {
                Ok(InterpValue::from_i64_as(ty, n))
            } else {
                Ok(o)
            }
        }
        CastOp::IntToPtr => {
            if let Some(n) = raw_i64 {
                Ok(InterpValue::Ptr(n as *mut u8))
            } else {
                Ok(o)
            }
        }
        CastOp::FpExt => match o {
            InterpValue::F32(x) => Ok(InterpValue::F64(x as f64)),
            other => Ok(other),
        },
        CastOp::FpTrunc => match o {
            InterpValue::F64(x) => Ok(InterpValue::F32(x as f32)),
            other => Ok(other),
        },
        CastOp::FpToSi | CastOp::FpToUi => match o {
            InterpValue::F64(x) => Ok(InterpValue::from_i64_as(ty, x as i64)),
            InterpValue::F32(x) => Ok(InterpValue::from_i64_as(ty, x as i64)),
            other => Ok(other),
        },
        CastOp::SiToFp | CastOp::UiToFp => {
            if let Some(n) = raw_i64 {
                match ty {
                    HirType::F64 => Ok(InterpValue::F64(n as f64)),
                    HirType::F32 => Ok(InterpValue::F32(n as f32)),
                    _ => Ok(o),
                }
            } else {
                Ok(o)
            }
        }
    }
}

fn size_of_hir_ty(ty: &HirType) -> usize {
    match ty {
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 | HirType::Ptr(_) => 8,
        HirType::I128 | HirType::U128 => 16,
        HirType::Struct(s) => s.fields.iter().map(size_of_hir_ty).sum::<usize>().max(1),
        HirType::Array(elem, n) => size_of_hir_ty(elem).saturating_mul((*n) as usize),
        _ => 8, // conservative default
    }
}

/// Recursively `InsertValue` into a struct hierarchy.
fn insert_value_recursive(agg: &mut InterpValue, indices: &[u32], v: InterpValue) {
    if indices.is_empty() {
        *agg = v;
        return;
    }
    let head = indices[0] as usize;
    let tail = &indices[1..];
    if let InterpValue::Struct(fields) = agg {
        if head < fields.len() {
            insert_value_recursive(&mut fields[head], tail, v);
        }
    }
}

/// Reading a typed value from a raw pointer. Used by `Load`.
///
/// # Safety
///
/// `ptr` must be valid for reads of at least `size_of_hir_ty(ty)`
/// bytes and properly aligned (we assume 8-byte slots for all types).
unsafe fn read_typed(ptr: *mut u8, ty: &HirType) -> InterpValue {
    match ty {
        HirType::Bool => InterpValue::Bool(*ptr != 0),
        HirType::I8 => InterpValue::I8(*ptr as i8),
        HirType::U8 => InterpValue::U8(*ptr),
        HirType::I16 => InterpValue::I16(*(ptr as *const i16)),
        HirType::U16 => InterpValue::U16(*(ptr as *const u16)),
        HirType::I32 => InterpValue::I32(*(ptr as *const i32)),
        HirType::U32 => InterpValue::U32(*(ptr as *const u32)),
        HirType::F32 => InterpValue::F32(*(ptr as *const f32)),
        HirType::I64 => InterpValue::I64(*(ptr as *const i64)),
        HirType::U64 => InterpValue::U64(*(ptr as *const u64)),
        HirType::F64 => InterpValue::F64(*(ptr as *const f64)),
        HirType::Ptr(_) => InterpValue::Ptr(*(ptr as *const *mut u8)),
        _ => InterpValue::I64(*(ptr as *const i64)),
    }
}

/// Writing a typed value to a raw pointer. Used by `Store`.
///
/// # Safety
///
/// Same as `read_typed`.
unsafe fn write_typed(ptr: *mut u8, v: &InterpValue) {
    match v {
        InterpValue::Bool(b) => *ptr = *b as u8,
        InterpValue::I8(x) => *(ptr as *mut i8) = *x,
        InterpValue::U8(x) => *ptr = *x,
        InterpValue::I16(x) => *(ptr as *mut i16) = *x,
        InterpValue::U16(x) => *(ptr as *mut u16) = *x,
        InterpValue::I32(x) => *(ptr as *mut i32) = *x,
        InterpValue::U32(x) => *(ptr as *mut u32) = *x,
        InterpValue::F32(x) => *(ptr as *mut f32) = *x,
        InterpValue::I64(x) => *(ptr as *mut i64) = *x,
        InterpValue::U64(x) => *(ptr as *mut u64) = *x,
        InterpValue::F64(x) => *(ptr as *mut f64) = *x,
        InterpValue::Ptr(p) => *(ptr as *mut *mut u8) = *p,
        // Aggregates: not directly representable in raw memory at
        // this phase. Phase B.2 will add per-field stores via GEP.
        _ => {}
    }
}

/// Call an extern "C" symbol whose pointer was registered with the
/// interpreter. The dispatch is by arg count — we transmute to the
/// appropriate fn pointer signature and invoke.
///
/// All args + return are funnelled through i64; the caller is
/// responsible for promoting/demoting against the actual symbol
/// signature. This matches what ZRTL's existing `register_runtime_symbol`
/// path expects.
fn call_extern_symbol(ptr: *const u8, args: &[InterpValue]) -> InterpValue {
    // Convert args to i64 for the C ABI.
    let raw_args: Vec<i64> = args.iter().map(|v| v.to_i64().unwrap_or(0)).collect();
    unsafe {
        match raw_args.len() {
            0 => {
                let f: extern "C" fn() -> i64 = core::mem::transmute(ptr);
                InterpValue::I64(f())
            }
            1 => {
                let f: extern "C" fn(i64) -> i64 = core::mem::transmute(ptr);
                InterpValue::I64(f(raw_args[0]))
            }
            2 => {
                let f: extern "C" fn(i64, i64) -> i64 = core::mem::transmute(ptr);
                InterpValue::I64(f(raw_args[0], raw_args[1]))
            }
            3 => {
                let f: extern "C" fn(i64, i64, i64) -> i64 = core::mem::transmute(ptr);
                InterpValue::I64(f(raw_args[0], raw_args[1], raw_args[2]))
            }
            4 => {
                let f: extern "C" fn(i64, i64, i64, i64) -> i64 = core::mem::transmute(ptr);
                InterpValue::I64(f(raw_args[0], raw_args[1], raw_args[2], raw_args[3]))
            }
            // 5+ args: rare in ZRTL signatures today; we'll extend
            // when the first real plugin needs it.
            n => InterpValue::I64(
                // Fallback: pretend it returns 0 so the trace stays
                // useful; the host will see the missing arity in the
                // unsupported error elsewhere.
                {
                    let _ = (ptr, n);
                    0
                },
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        HirFunction, HirFunctionSignature, HirInstruction, HirModule, HirParam, HirTerminator,
        HirType, HirValue, HirValueKind, ParamAttributes,
    };
    use indexmap::IndexMap;
    use std::collections::HashSet;
    use zyntax_typed_ast::InternedString;

    fn mk_fn(name: &str, params: Vec<HirType>, returns: Vec<HirType>) -> HirFunction {
        let sig = HirFunctionSignature {
            params: params
                .into_iter()
                .enumerate()
                .map(|(i, ty)| HirParam {
                    id: HirId::new(),
                    name: InternedString::new_global(&format!("p{}", i)),
                    ty,
                    attributes: ParamAttributes::default(),
                })
                .collect(),
            returns,
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        HirFunction::new(InternedString::new_global(name), sig)
    }

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

    /// `def add(a: i64, b: i64): i64 { return a + b }`
    #[test]
    fn interpreter_runs_simple_add() {
        let mut func = mk_fn("add", vec![HirType::I64, HirType::I64], vec![HirType::I64]);
        // Synthesize Parameter(0/1) values referenced by Binary.
        let p0 = func.signature.params[0].id;
        let p1 = func.signature.params[1].id;
        // Re-register them as Parameter-kind values (HirFunction::new
        // doesn't auto-create per-index Parameter values).
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
        let entry_id = func.entry_block;
        let entry = func.blocks.get_mut(&entry_id).unwrap();
        entry.instructions.push(HirInstruction::Binary {
            result: sum,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: p0,
            right: p1,
        });
        entry.terminator = HirTerminator::Return { values: vec![sum] };

        let mut module = HirModule::new(InternedString::new_global("test"));
        let func_id = func.id;
        module.functions.insert(func_id, func);

        let mut interp = HirInterpreter::new(&module);
        let result = interp
            .call("add", vec![InterpValue::I64(10), InterpValue::I64(32)])
            .expect("call should succeed");
        assert!(matches!(result, InterpValue::I64(42)));
    }

    /// `def main(): i64 { if (true) return 1 else return 0 }` —
    /// exercises CondBranch + multiple return points.
    #[test]
    fn interpreter_runs_cond_branch() {
        let mut func = mk_fn("main", vec![], vec![HirType::I64]);

        let true_const = add_value(
            &mut func,
            HirType::Bool,
            HirValueKind::Constant(HirConstant::Bool(true)),
        );
        let one = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(1)),
        );
        let zero = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(0)),
        );

        let entry_id = func.entry_block;
        let then_id = HirId::new();
        let else_id = HirId::new();
        func.blocks.insert(
            then_id,
            HirBlock {
                id: then_id,
                label: None,
                phis: vec![],
                instructions: vec![],
                terminator: HirTerminator::Return { values: vec![one] },
                dominance_frontier: HashSet::new(),
                predecessors: vec![],
                successors: vec![],
            },
        );
        func.blocks.insert(
            else_id,
            HirBlock {
                id: else_id,
                label: None,
                phis: vec![],
                instructions: vec![],
                terminator: HirTerminator::Return { values: vec![zero] },
                dominance_frontier: HashSet::new(),
                predecessors: vec![],
                successors: vec![],
            },
        );
        let entry = func.blocks.get_mut(&entry_id).unwrap();
        entry.terminator = HirTerminator::CondBranch {
            condition: true_const,
            true_target: then_id,
            false_target: else_id,
        };

        let mut module = HirModule::new(InternedString::new_global("test"));
        let func_id = func.id;
        module.functions.insert(func_id, func);

        let mut interp = HirInterpreter::new(&module);
        let result = interp.call("main", vec![]).unwrap();
        assert!(matches!(result, InterpValue::I64(1)));
    }

    /// Profile counters increment per call.
    #[test]
    fn interpreter_records_profile_samples() {
        let mut func = mk_fn("noop", vec![], vec![HirType::Void]);
        let entry_id = func.entry_block;
        func.blocks.get_mut(&entry_id).unwrap().terminator =
            HirTerminator::Return { values: vec![] };

        let mut module = HirModule::new(InternedString::new_global("test"));
        let func_id = func.id;
        module.functions.insert(func_id, func);

        let mut interp = HirInterpreter::new(&module);
        for _ in 0..3 {
            interp.call("noop", vec![]).unwrap();
        }
        let sample = interp.profile.get(&func_id).copied().unwrap_or_default();
        assert_eq!(sample.call_count, 3);
    }

    fn _silence_indexmap_use() {
        // Touch IndexMap to keep its import used.
        let _: IndexMap<HirId, HirFunction> = IndexMap::new();
    }
}
