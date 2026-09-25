//! HIR bytecode interpreter — universal Tier 0 cold-start execution.
//!
//! Compiles each `HirFunction` once into a compact, register-based
//! bytecode and runs it in a tight dispatch loop. Used as the cold-
//! start tier across every Zyntax target. On native, hot functions
//! tier up to the Cranelift baseline JIT (Tier 1); on wasm targets,
//! hot functions tier up to a wasm-emitting backend (also Tier 1 —
//! same rung in the ladder, different code generator).
//!
//! ## Design
//!
//! * **Register VM**, not stack VM. HIR is already in SSA form — each
//!   `HirId` maps to one register slot, so lowering is a 1:1 walk
//!   with no stack-shuffling overhead. Modern dynamic-language VMs
//!   (Lua 5+, V8 Ignition, Hermes) all use register VMs for the same
//!   reason.
//! * **Out-of-SSA at lowering time**: phi nodes are resolved during
//!   compilation by emitting `Move` ops at each branch site (the
//!   classic "exit SSA" rewrite). The interpreter itself never sees
//!   phi nodes.
//! * **Constant / type / args / switch-table pools** are owned by the
//!   `CompiledFunction` and indexed by `u32`. Keeping these out of the
//!   `Op` enum keeps each opcode small (≤16 bytes) so the bytecode
//!   stream stays cache-friendly.
//! * **Tagged values on the bus**. `InterpValue` carries width and
//!   signedness so we don't lose precision crossing call boundaries,
//!   load/store layouts, or extern symbol invocations. Integer ops
//!   funnel through `i64`; the result is re-tagged from operand width.
//!
//! ## Coverage in this initial slice
//!
//! Covered: `Binary` / `Unary` / `Cast`; `Alloca` / `Load` / `Store`;
//! `Branch` / `CondBranch` / `Switch` / `Return` / `Unreachable`; phis;
//! direct calls (`HirCallable::Function`); FFI calls (`HirCallable::
//! Symbol`); `ExtractValue` / `InsertValue`.
//!
//! Phase B.2 will add: algebraic effects (`PerformEffect`,
//! `AsyncSaveSlot` / `AsyncLoadSlot`, intercepted `__zyntax_effect_
//! resume`), closures, trait-object dispatch, atomics, SIMD.

use std::collections::HashMap;

/// A map keyed by `HirId`, hashed cheaply: these are probed on every
/// call the interpreter makes.
type IdMap<V> = HashMap<HirId, V, std::hash::BuildHasherDefault<fnv::FnvHasher>>;

/// What a function uses that the bytecode interpreter cannot execute.
///
/// The interpreter rejects these when it compiles a function to
/// bytecode, which is lazy, so a program that only reaches them on some
/// path installs cleanly and fails much later. Callers for which the
/// interpreter is the ONLY engine should ask up front. Where it is a
/// tier with a JIT underneath, these are not errors: the JIT runs them.
pub fn unsupported_constructs(module: &HirModule) -> Vec<(String, &'static str)> {
    let mut out = Vec::new();
    for func in module.functions.values() {
        if func.is_external {
            continue;
        }
        let mut found: Option<&'static str> = None;
        'blocks: for block in func.blocks.values() {
            for inst in &block.instructions {
                found = match inst {
                    HirInstruction::PerformEffect { .. } => Some("algebraic effects"),
                    HirInstruction::FiberNew { .. }
                    | HirInstruction::FiberResume { .. }
                    | HirInstruction::FiberResumeWith { .. }
                    | HirInstruction::FiberYield { .. }
                    | HirInstruction::FiberTransfer { .. }
                    | HirInstruction::FiberDrop { .. } => Some("fibers"),
                    _ => None,
                };
                if found.is_some() {
                    break 'blocks;
                }
            }
        }
        if let Some(what) = found {
            out.push((
                func.name
                    .resolve_global()
                    .unwrap_or_else(|| "?".to_string()),
                what,
            ));
        }
    }
    out
}

use crate::hir::{
    BinaryOp, CastOp, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule,
    HirTerminator, HirType, HirValueKind, UnaryOp, VectorMinMaxKind, VectorUnaryKind,
};
use crate::value::ZyntaxValue;

// ─────────────────────────────────────────────────────────────────────────────
// Value model
// ─────────────────────────────────────────────────────────────────────────────
//
// The interpreter uses [`crate::value::ZyntaxValue`] directly. No
// separate `InterpValue` type. Width info comes from the `HirType`
// stored per-register in [`CompiledFunction::reg_types`] (and on
// each HIR instruction); the interpreter masks integer arithmetic
// results to fit that width on output.

/// `ZYNTAX_TRACE_INTERP=1` prints every call the interpreter makes and
/// every extern it reaches; safe to run with.
pub fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ZYNTAX_TRACE_INTERP").is_some())
}

/// The address of `name` in the running process, if the dynamic linker
/// knows it.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn process_symbol(name: &str) -> Option<*const u8> {
    let c = std::ffi::CString::new(name).ok()?;
    // SAFETY: `dlsym` with the default handle reads the process's own
    // symbol tables; a null result is a miss.
    let p = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c.as_ptr()) };
    (!p.is_null()).then_some(p as *const u8)
}

/// Windows has no default handle: every loaded module is asked in turn.
#[cfg(windows)]
fn process_symbol(name: &str) -> Option<*const u8> {
    use std::ffi::{c_char, c_void};
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32EnumProcessModules(
            process: *mut c_void,
            modules: *mut *mut c_void,
            size: u32,
            needed: *mut u32,
        ) -> i32;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *const c_void;
    }
    let c = std::ffi::CString::new(name).ok()?;
    let mut modules = [std::ptr::null_mut::<c_void>(); 512];
    let mut needed = 0u32;
    // SAFETY: the buffer is passed with its own byte size, and only the
    // handles written into it are read; `c` outlives the lookups.
    unsafe {
        let size = size_of_val(&modules) as u32;
        if K32EnumProcessModules(GetCurrentProcess(), modules.as_mut_ptr(), size, &mut needed) == 0
        {
            return None;
        }
        let n = (needed as usize / size_of::<*mut c_void>()).min(modules.len());
        modules[..n]
            .iter()
            .map(|&m| GetProcAddress(m, c.as_ptr()))
            .find(|p| !p.is_null())
            .map(|p| p as *const u8)
    }
}

#[cfg(not(any(all(unix, not(target_arch = "wasm32")), windows)))]
fn process_symbol(_name: &str) -> Option<*const u8> {
    None
}

/// The machine word a value travels as into native code: an integer or
/// pointer as itself, a float as its bits, an aggregate as its address.
fn word_of(v: &ZyntaxValue) -> u64 {
    match v {
        ZyntaxValue::Float(x) => x.to_bits(),
        ZyntaxValue::F32(x) => x.to_bits() as u64,
        other => value_to_i64(other).unwrap_or(0) as u64,
    }
}

/// The value of type `ty` a native call handed back as the word `w`.
fn value_from_word(ty: &HirType, w: u64) -> ZyntaxValue {
    match ty {
        HirType::Void => ZyntaxValue::Void,
        HirType::F32 => ZyntaxValue::F32(f32::from_bits(w as u32)),
        HirType::F64 => ZyntaxValue::Float(f64::from_bits(w)),
        HirType::Struct(st) => match crate::abi::struct_carried_as_its_field(st) {
            Some(field) => value_from_word(field, w),
            None => ZyntaxValue::Pointer(w as usize as *mut u8),
        },
        HirType::Array(_, _) | HirType::Union(_) | HirType::Ptr(_) | HirType::Opaque(_) => {
            ZyntaxValue::Pointer(w as usize as *mut u8)
        }
        other => value_from_i64_as(other, w as i64),
    }
}

/// Coerce a `ZyntaxValue` to `i64` for the interpreter's i64-funneled
/// integer bus. Accepts every integer-shaped variant (generic +
/// width-precise siblings).
pub fn value_to_i64(v: &ZyntaxValue) -> Option<i64> {
    match v {
        ZyntaxValue::Bool(b) => Some(*b as i64),
        ZyntaxValue::Int(x) => Some(*x),
        ZyntaxValue::UInt(x) => Some(*x as i64),
        ZyntaxValue::I8(x) => Some(*x as i64),
        ZyntaxValue::I16(x) => Some(*x as i64),
        ZyntaxValue::I32(x) => Some(*x as i64),
        ZyntaxValue::U8(x) => Some(*x as i64),
        ZyntaxValue::U16(x) => Some(*x as i64),
        ZyntaxValue::U32(x) => Some(*x as i64),
        ZyntaxValue::Float(x) => Some(*x as i64),
        ZyntaxValue::F32(x) => Some(*x as i64),
        ZyntaxValue::Pointer(p) => Some(*p as i64),
        _ => None,
    }
}

/// Coerce a `ZyntaxValue` to `f64` for the interpreter's f64-funneled
/// float bus.
pub fn value_to_f64(v: &ZyntaxValue) -> Option<f64> {
    match v {
        ZyntaxValue::Float(x) => Some(*x),
        ZyntaxValue::F32(x) => Some(*x as f64),
        _ => value_to_i64(v).map(|n| n as f64),
    }
}

/// Construct a `ZyntaxValue` from a raw `i64`, tagging with the
/// width-precise variant that matches `ty`. Narrow widths produce
/// `I8`/`I16`/`I32`/`U8`/`U16`/`U32`; i64/u64/f64 reuse the generic
/// `Int`/`UInt`/`Float` variants (no separate I64/U64/F64 siblings).
pub fn value_from_i64_as(ty: &HirType, raw: i64) -> ZyntaxValue {
    match ty {
        HirType::Void => ZyntaxValue::Void,
        HirType::Bool => ZyntaxValue::Bool(raw != 0),
        HirType::I8 => ZyntaxValue::I8(raw as i8),
        HirType::I16 => ZyntaxValue::I16(raw as i16),
        HirType::I32 => ZyntaxValue::I32(raw as i32),
        HirType::I64 => ZyntaxValue::Int(raw),
        HirType::U8 => ZyntaxValue::U8(raw as u8),
        HirType::U16 => ZyntaxValue::U16(raw as u16),
        HirType::U32 => ZyntaxValue::U32(raw as u32),
        HirType::U64 => ZyntaxValue::UInt(raw as u64),
        HirType::F32 => ZyntaxValue::F32(f32::from_bits(raw as u32)),
        HirType::F64 => ZyntaxValue::Float(f64::from_bits(raw as u64)),
        HirType::Ptr(_) => ZyntaxValue::Pointer(raw as *mut u8),
        _ => ZyntaxValue::Int(raw),
    }
}

/// The value of an integer constant, whatever width it was written at.
fn constant_int(c: &HirConstant) -> Option<i64> {
    match c {
        HirConstant::Bool(b) => Some(*b as i64),
        HirConstant::I8(x) => Some(*x as i64),
        HirConstant::I16(x) => Some(*x as i64),
        HirConstant::I32(x) => Some(*x as i64),
        HirConstant::I64(x) => Some(*x),
        HirConstant::U8(x) => Some(*x as i64),
        HirConstant::U16(x) => Some(*x as i64),
        HirConstant::U32(x) => Some(*x as i64),
        HirConstant::U64(x) => Some(*x as i64),
        HirConstant::USize(x) => Some(*x as i64),
        HirConstant::ISize(x) => Some(*x),
        _ => None,
    }
}

/// Write constant `c` of type `ty` into `out` at the native layout.
fn write_constant(c: &HirConstant, ty: &HirType, out: &mut [u8]) {
    match (c, ty) {
        (HirConstant::Struct(fields), HirType::Struct(s)) => {
            let layout = struct_layout(s);
            for ((field, field_ty), offset) in fields.iter().zip(&s.fields).zip(&layout.offsets) {
                let end = (offset + size_of_hir_ty(field_ty)).min(out.len());
                if *offset < end {
                    write_constant(field, field_ty, &mut out[*offset..end]);
                }
            }
        }
        (HirConstant::Array(items), HirType::Array(elem, _)) => {
            let stride = size_of_hir_ty(elem);
            for (i, item) in items.iter().enumerate() {
                let start = i * stride;
                let end = (start + stride).min(out.len());
                if start < end {
                    write_constant(item, elem, &mut out[start..end]);
                }
            }
        }
        (c, ty) => {
            let v = const_to_zyntax(c);
            if size_of_hir_ty(ty) <= out.len() {
                // SAFETY: `out` holds at least the type's size.
                unsafe { write_typed(out.as_mut_ptr(), &v, ty) };
            }
        }
    }
}

fn const_to_zyntax(c: &HirConstant) -> ZyntaxValue {
    match c {
        HirConstant::Bool(b) => ZyntaxValue::Bool(*b),
        HirConstant::I8(x) => ZyntaxValue::I8(*x),
        HirConstant::I16(x) => ZyntaxValue::I16(*x),
        HirConstant::I32(x) => ZyntaxValue::I32(*x),
        HirConstant::I64(x) => ZyntaxValue::Int(*x),
        HirConstant::USize(x) => ZyntaxValue::UInt(*x),
        HirConstant::ISize(x) => ZyntaxValue::Int(*x),
        HirConstant::U8(x) => ZyntaxValue::U8(*x),
        HirConstant::U16(x) => ZyntaxValue::U16(*x),
        HirConstant::U32(x) => ZyntaxValue::U32(*x),
        HirConstant::U64(x) => ZyntaxValue::UInt(*x),
        HirConstant::F32(x) => ZyntaxValue::F32(*x),
        HirConstant::F64(x) => ZyntaxValue::Float(*x),
        HirConstant::Null(_) => ZyntaxValue::Pointer(core::ptr::null_mut()),
        _ => ZyntaxValue::Undef,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bytecode IR
// ─────────────────────────────────────────────────────────────────────────────

/// Register slot index. One per HirId in a function's value table; all
/// SSA values fit in `u16::MAX` slots in practice (largest modules
/// today have < 5k SSA values per fn).
pub type Reg = u16;

/// Program-counter index into a `CompiledFunction::code` vector.
pub type Pc = u32;

/// Compact, register-based opcode set. Every variant is ≤ 16 bytes on
/// 64-bit; the most common variants (3-reg arithmetic) are 8 bytes,
/// keeping the bytecode stream cache-friendly.
/// The one-argument libm intrinsics the interpreter computes itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FUnaryOp {
    Sin,
    Cos,
    Log,
    Exp,
}

#[derive(Debug, Clone)]
pub enum Op {
    /// `dst = const_pool[c]`
    LoadConst {
        dst: Reg,
        c: u32,
    },
    /// `dst = src`. Emitted at branch sites for out-of-SSA phi copies.
    Move {
        dst: Reg,
        src: Reg,
    },
    /// Choose one SSA value without branching.
    Select {
        dst: Reg,
        cond: Reg,
        yes: Reg,
        no: Reg,
    },

    // ── integer arithmetic (operands flow through i64 on the bus) ──
    IAdd {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    ISub {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    IMul {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    IDiv {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    IRem {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    IAnd {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    IOr {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    IXor {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    IShl {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    IShr {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    INeg {
        dst: Reg,
        src: Reg,
    },
    BNot {
        dst: Reg,
        src: Reg,
    },

    // ── float arithmetic ──
    FAdd {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    FSub {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    FMul {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    FDiv {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    FNeg {
        dst: Reg,
        src: Reg,
    },
    /// `dst = sqrt(src)` — single-arg math intrinsic.
    /// Emitted when the source-side `sqrt` name is routed through
    /// `HirCallable::Intrinsic(Intrinsic::Sqrt)`. Mirrors the
    /// Cranelift backend's `builder.ins().sqrt(...)` lowering.
    FSqrt {
        dst: Reg,
        src: Reg,
    },
    /// `dst = 1.0 / sqrt(src)` — single-arg math intrinsic.
    /// Emitted when the source-side `rsqrt` name is routed through
    /// `HirCallable::Intrinsic(Intrinsic::Rsqrt)`. Mirrors the
    /// Cranelift backend's `fdiv(fconst 1.0, sqrt(x))` lowering and
    /// LLVM's `fdiv(1.0, llvm.sqrt)` form.
    FRsqrt {
        dst: Reg,
        src: Reg,
    },
    /// `dst = fabs(src)` — single-arg math intrinsic.
    /// Emitted when the source-side `abs` name is routed through
    /// `HirCallable::Intrinsic(Intrinsic::Fabs)`. Mirrors the
    /// Cranelift backend's `builder.ins().fabs(...)` lowering.
    FAbs {
        dst: Reg,
        src: Reg,
    },
    /// `dst = floor(src)`, for `HirCallable::Intrinsic(Intrinsic::Floor)`.
    FFloor {
        dst: Reg,
        src: Reg,
    },
    /// `dst = f(src)` for the one-argument libm intrinsics (sin, cos,
    /// log, exp), as the native tiers call libm for them.
    FUnary {
        dst: Reg,
        src: Reg,
        op: FUnaryOp,
    },
    /// `dst = pow(a, b)`, for `HirCallable::Intrinsic(Intrinsic::Pow)`.
    FPow {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    /// `dst = a * b + c` — fused multiply-add, single round.
    /// Emitted by the `fma_contract` HIR pass when it rewrites
    /// `fadd(fmul a b, c)` to `Intrinsic::Fma`. Mirrors the
    /// Cranelift backend's `builder.ins().fma(...)` lowering; the
    /// interpreter computes via `f64::mul_add` so the rounding
    /// semantics match a hardware FMA.
    FMulAdd {
        dst: Reg,
        a: Reg,
        b: Reg,
        c: Reg,
    },

    // ── comparisons (result is Bool) ──
    ICmpEq {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    ICmpNe {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    ICmpLt {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    ICmpLe {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    ICmpGt {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    ICmpGe {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// The unsigned forms: an operation whose result depends on the
    /// sign, on an unsigned type.
    UDiv {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    URem {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    UShr {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    UCmpLt {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    UCmpLe {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    UCmpGt {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    UCmpGe {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    FCmpEq {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    FCmpNe {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    FCmpLt {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    FCmpLe {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    FCmpGt {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    FCmpGe {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },

    /// Cast `dst = (ty_pool[ty]) src` with kind `op`.
    Cast {
        dst: Reg,
        src: Reg,
        op: CastOp,
        ty: u32,
    },

    // ── memory ──
    /// `dst = alloca(size_bytes)` — bytes are interpreter-allocated;
    /// `size_bytes` already accounts for any count multiplier.
    Alloca {
        dst: Reg,
        size_bytes: u32,
    },
    /// `dst = *(ptr_pool[ty]*) regs[ptr]`
    Load {
        dst: Reg,
        ptr: Reg,
        ty: u32,
    },
    /// `*(ptr_pool[ty]*) regs[ptr] = regs[val]`
    Store {
        ptr: Reg,
        val: Reg,
        ty: u32,
    },

    // ── aggregates (held as the address of their storage) ──
    /// `dst` = the field of the aggregate at `src` named by the index
    /// path `indices_pool[idx]`; `ty` is the aggregate's type.
    ExtractValue {
        dst: Reg,
        src: Reg,
        idx: u32,
        ty: u32,
    },
    /// Write `val` into the aggregate at `agg` in place; `dst` is that
    /// same address.
    InsertValue {
        dst: Reg,
        agg: Reg,
        val: Reg,
        idx: u32,
        ty: u32,
    },

    // ── control flow ──
    /// The entry of a loop header that promoted code can resume at:
    /// `osr_sites[site]` says what the frame hands over.
    LoopHeader {
        site: u32,
    },
    Jump {
        target: Pc,
    },
    JumpIf {
        cond: Reg,
        t: Pc,
        f: Pc,
    },
    /// Linear scan of `switch_pool[table]` against `regs[scrut]`;
    /// fall through to `default` on miss.
    Switch {
        scrut: Reg,
        table: u32,
        default: Pc,
    },
    /// Return `regs[src]` to the caller.
    Ret {
        src: Reg,
    },
    RetVoid,
    Unreachable,

    /// Direct call into another HIR function.
    /// `has_dst` controls whether the return value is bound to `dst`.
    CallFn {
        dst: Reg,
        has_dst: bool,
        fn_id: HirId,
        args: u32,
    },
    /// Call the symbol `symbol_pool[sym]` with the shape `sig_pool[sig]`.
    CallSym {
        dst: Reg,
        has_dst: bool,
        sym: u32,
        args: u32,
        sig: u32,
    },
    /// `Intrinsic::Malloc` lowered to a runtime-sized allocation via
    /// the interpreter's `Memory` arena. `size_reg` carries the
    /// byte count at runtime; result is a `ZyntaxValue::Pointer`.
    /// Allocations leak until the InterpRuntime is dropped — fine
    /// for Phase J's short-lived cooperative-async tasks.
    Malloc {
        dst: Reg,
        has_dst: bool,
        size_reg: Reg,
    },
    Realloc {
        dst: Reg,
        old_reg: Reg,
        size_reg: Reg,
    },
    /// `Intrinsic::Free`: returns the block to the allocator the
    /// program's other tiers use; nothing to do while the interpreter
    /// keeps its own memory.
    Free {
        ptr: Reg,
    },
    /// `memcpy`/`memmove`/`memset` over `len` bytes: `src` is the source
    /// pointer, or the byte value for a fill.
    MemOp {
        kind: MemOpKind,
        dst_ptr: Reg,
        src: Reg,
        len: Reg,
    },
    /// `dst` = the current entry of function `fn_id`, read from its
    /// call cell, as compiled code takes a function's address.
    FuncAddr {
        dst: Reg,
        fn_id: HirId,
    },
    /// Call the code at `fn_ptr_reg` with the shape `sig_pool[sig]`. A
    /// host dispatcher, when one is installed, resolves the handle
    /// itself; otherwise the handle is native code.
    CallIndirect {
        dst: Reg,
        has_dst: bool,
        fn_ptr_reg: Reg,
        args: u32,
        sig: u32,
    },
    /// `HirInstruction::AsyncSaveSlot { frame, slot, value }` — store
    /// `value` (as i64) at `frame + slot * 8`. Krio's SM layout uses
    /// uniform 8-byte slots; `frame` is the SM-ptr param after
    /// `reshape_to_poll_abi`.
    AsyncSaveSlot {
        frame_reg: Reg,
        slot: u32,
        val_reg: Reg,
    },
    /// `HirInstruction::AsyncLoadSlot { result, ty, frame, slot }` —
    /// load typed value at `frame + slot * 8` into `dst`.
    AsyncLoadSlot {
        dst: Reg,
        frame_reg: Reg,
        slot: u32,
        ty: u32,
    },

    /// `HirInstruction::GetElementPtr { result, ty, ptr, indices }` —
    /// compute `regs[ptr] + sum_i(regs[indices[i]] * stride[i])`. For
    /// the common single-index case (array indexing emitted by
    /// `TypedExpression::Index` in [`crate::ssa`]) `stride[0]` is just
    /// `size_of_hir_ty(elem_ty)`. The `args` field re-uses the
    /// `args_pool` to hold the index register list; `stride_idx`
    /// indexes into `gep_stride_pool` for the per-index byte stride.
    Gep {
        dst: Reg,
        ptr: Reg,
        args: u32,
        stride: u32,
    },

    // ── SIMD / vector (scalarized) ──
    // The interpreter has no vector registers; a vector value is held as
    // `ZyntaxValue::Array` of `lanes` scalar lanes, each in the element's
    // precise width. Every op below is a native lane loop — inline, never
    // an FFI/plugin call — so `@kernel`/auto-vectorized code executes here
    // with the SAME numeric result the Cranelift/LLVM/wasm backends produce.
    /// `dst = splat(regs[scalar])` → `lanes` copies.
    VSplat {
        dst: Reg,
        scalar: Reg,
        lanes: u8,
    },
    /// `dst = [read_typed(regs[ptr] + i*sizeof(elem)) for i in 0..lanes]`.
    VLoad {
        dst: Reg,
        ptr: Reg,
        lanes: u8,
        elem_ty: u32,
    },
    /// `write_typed(regs[ptr] + i*sizeof(elem)) = lane_i` for each lane.
    VStore {
        val: Reg,
        ptr: Reg,
        elem_ty: u32,
    },
    /// `dst = regs[vector][lane]`.
    VExtract {
        dst: Reg,
        vector: Reg,
        lane: u8,
    },
    /// `dst = regs[vector] with lane := regs[scalar]`.
    VInsert {
        dst: Reg,
        vector: Reg,
        scalar: Reg,
        lane: u8,
    },
    /// `dst = fold(regs[vector], op)` — horizontal reduction to a scalar.
    VReduce {
        dst: Reg,
        vector: Reg,
        op: BinaryOp,
    },
    /// `dst[i] = regs[lhs][i] op regs[rhs][i]` — element-wise arithmetic.
    VBinOp {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
        op: BinaryOp,
    },
    /// `dst[i] = kind(regs[operand][i])` — element-wise unary (float lanes).
    VUnary {
        dst: Reg,
        operand: Reg,
        kind: VectorUnaryKind,
    },
    /// `dst[i] = min|max(regs[lhs][i], regs[rhs][i])` — element-wise.
    VMinMax {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
        kind: VectorMinMaxKind,
    },
    /// `dst = fused widening dot-accumulate` — 16×i8 → 4×i32:
    /// `dst[j] = acc[j] + Σ_{k<4} a[4j+k] * b[4j+k]` (widened to i32).
    VDot {
        dst: Reg,
        acc: Reg,
        a: Reg,
        b: Reg,
        unsigned: bool,
    },
}

/// One compiled function: bytecode stream + side pools.
#[derive(Debug, Clone, Default)]
pub struct CompiledFunction {
    pub code: Vec<Op>,
    pub const_pool: Vec<ZyntaxValue>,
    /// Constant-pool slots whose pointer names a module global.
    pub global_consts: Vec<(HirId, u32)>,
    pub type_pool: Vec<HirType>,
    pub args_pool: Vec<Vec<Reg>>,
    /// Each entry is a switch table: list of `(case_i64, target_pc)`.
    pub switch_pool: Vec<Vec<(i64, Pc)>>,
    /// Each entry is the index path for `ExtractValue` / `InsertValue`.
    pub indices_pool: Vec<Vec<u32>>,
    /// Symbol-name pool for FFI calls; we hold names (not raw fn ptrs)
    /// at compile time because symbols are registered at the
    /// interpreter level, not the compiler level.
    pub symbol_pool: Vec<String>,
    /// Per-`Op::Gep` steps, one per index: `(stride, offset)`. At
    /// runtime `Gep` computes `ptr + sum(stride[i] * regs[index[i]] +
    /// offset[i])`; a struct field is a fixed offset with stride 0.
    pub gep_stride_pool: Vec<Vec<(i64, i64)>>,
    /// The shape of each native call the function makes.
    pub sig_pool: Vec<NativeSig>,
    /// The thunk made for each entry of `sig_pool`, 0 until one is.
    pub sig_thunks: Vec<std::cell::Cell<usize>>,
    /// Aggregate values the function needs storage for at entry: undef
    /// values and struct constants, with the bytes to start from.
    pub entry_storage: Vec<(Reg, Vec<u8>)>,
    /// Whether the function writes its returned struct into storage the
    /// caller provides.
    pub returns_through_destination: bool,
    /// Size in bytes of what the function returns through a destination.
    pub destination_size: usize,
    /// Loop headers a promoted helper can resume at.
    pub osr_sites: Vec<OsrSite>,
    /// Per-register hint of the SSA value's HirType. Used to size
    /// extern-call returns and width-correct integer ops.
    pub reg_types: Vec<HirType>,
    /// Total number of registers (== `reg_types.len()`).
    pub n_regs: u32,
    /// Number of parameters in the original signature; arg-binding
    /// fills regs[0..n_params].
    pub n_params: u16,
    /// Which function each `FuncRef` result names, so a call that
    /// hands one straight to a runtime entry can name the callee.
    pub func_refs: HashMap<HirId, HirId>,
}

/// Bytes of stack [`clear_stack_below`] zeroes.
const CLEARED_STACK_BYTES: usize = 8 << 10;

/// Zero the stack below the caller's frame, where a compiled callee's
/// frames are about to lie. A slot such a frame reads only after
/// writing it still holds, until then, what an earlier call at the
/// same depth left there, and the collector reads every word of the
/// stack: a stale pointer there keeps that call's garbage alive.
#[inline(never)]
fn clear_stack_below() {
    let mut below = [0u8; CLEARED_STACK_BYTES];
    std::hint::black_box(&mut below);
}

/// A frame's register file registered with the collector as a root
/// range for as long as the frame runs interpreted. Once the frame
/// transfers to compiled code its live-ins are the compiled frame's,
/// and what the registers still name is no longer held by this frame.
struct FrameRoots(*const u8);

/// Storage a frame took for itself: the blocks it owns until it
/// finishes, as a stack frame owns its slots. A block may be named by
/// nothing else once the register that held it is reused, so the list
/// is a root range of its own, registered again whenever it moves.
struct Scratch {
    blocks: Vec<*mut u8>,
    registered: *const u8,
}

impl Scratch {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            registered: std::ptr::null(),
        }
    }

    fn push(&mut self, block: *mut u8) {
        self.blocks.push(block);
        if !crate::collector::is_enabled() {
            return;
        }
        let now = self.blocks.as_ptr() as *const u8;
        if now != self.registered && !self.registered.is_null() {
            crate::collector::remove_root_range(self.registered);
        }
        crate::collector::add_root_range(now, std::mem::size_of_val(self.blocks.as_slice()));
        self.registered = now;
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if !self.registered.is_null() {
            crate::collector::remove_root_range(self.registered);
        }
    }
}

impl FrameRoots {
    fn new(regs: &[ZyntaxValue]) -> Self {
        let ptr = regs.as_ptr() as *const u8;
        if crate::collector::is_enabled() {
            crate::collector::add_root_range(ptr, std::mem::size_of_val(regs));
        }
        Self(ptr)
    }

    /// Stop rooting the registers: the frame has handed its live
    /// values elsewhere.
    fn release(&mut self) {
        if !self.0.is_null() {
            crate::collector::remove_root_range(self.0);
            self.0 = std::ptr::null();
        }
    }
}

impl Drop for FrameRoots {
    fn drop(&mut self) {
        self.release();
    }
}

/// A loop header the interpreter can leave for promoted code: the
/// registers holding the live-ins, laid out in the frame the helper
/// reads.
#[derive(Debug, Clone)]
pub struct OsrSite {
    pub site_key: u64,
    pub live_ins: Vec<Reg>,
    pub types: Vec<HirType>,
    pub frame: crate::osr::OsrFrame,
    pub ret: HirType,
    /// Frame offset of the destination pointer, for a function that
    /// returns through one.
    pub destination: Option<u32>,
}

/// Header visits before an interpreted frame asks for promoted code.
const OSR_REQUEST_VISITS: u32 = 64;
/// Header visits at which a loop counts as warm: the frame asks for
/// its promoted code then, when there is a worker to take the request,
/// so the code is there by the time the loop is hot.
const OSR_WARM_VISITS: u32 = 16;

/// Which memory intrinsic an [`Op::MemOp`] performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemOpKind {
    Copy,
    Move,
    Set,
}

/// The shape of a call into native code: what travels in, what comes
/// back, and whether the callee writes its result through a destination
/// the caller passes first. One Cranelift thunk is compiled per shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeSig {
    pub params: Vec<HirType>,
    pub ret: HirType,
    pub destination: bool,
}

impl NativeSig {
    /// The shape of calling `function` directly. `address_taken` says
    /// whether callers can be made to pass a destination.
    pub fn of_function(function: &HirFunction, address_taken: bool) -> Self {
        let abi = crate::abi::function_abi(function, address_taken);
        Self {
            params: function
                .signature
                .params
                .iter()
                .map(|p| p.ty.clone())
                .collect(),
            ret: function
                .signature
                .returns
                .first()
                .cloned()
                .unwrap_or(HirType::Void),
            destination: abi.destination.is_some(),
        }
    }

    /// The shape of a call site that only knows its operand types: a
    /// symbol or a function pointer. Neither takes a destination.
    pub fn of_site(params: Vec<HirType>, ret: HirType) -> Self {
        Self {
            params,
            ret,
            destination: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lowering: HIR → bytecode
// ─────────────────────────────────────────────────────────────────────────────

/// Compile a single `HirFunction` to bytecode. Performed once per
/// function on first call; the result is cached in the interpreter.
///
/// `module` + `memory` are threaded in so the lowerer can resolve
/// `HirValueKind::Global` references: each global with a string
/// initializer gets a ZRTL-format buffer allocated in `memory` and
/// the resulting pointer pre-loaded into the consuming SSA value's
/// register via `LoadConst`. Without that, `println("hello")` would
/// pass `0` to the host bridge (and surface as an empty line).
pub fn compile_function(
    module: &HirModule,
    memory: &mut Memory,
    func: &HirFunction,
) -> Result<CompiledFunction, InterpError> {
    compile_function_with(module, memory, func, false)
}

/// [`compile_function`] for a function whose address may be taken,
/// which rules out returning a struct through a destination.
pub fn compile_function_with(
    module: &HirModule,
    memory: &mut Memory,
    func: &HirFunction,
    address_taken: bool,
) -> Result<CompiledFunction, InterpError> {
    // Every value takes a register, and a register is a `Reg`: a
    // function past that is refused, and runs natively.
    if func.values.len() + func.signature.params.len() > Reg::MAX as usize {
        return Err(InterpError::UnsupportedInstruction(
            "register overflow".to_string(),
        ));
    }
    let mut cf = CompiledFunction::default();
    let mut reg_of: HashMap<HirId, Reg> = HashMap::new();
    let abi = crate::abi::function_abi(func, address_taken);
    cf.returns_through_destination = abi.destination.is_some();
    cf.destination_size = abi.destination.as_ref().map(size_of_hir_ty).unwrap_or(0);
    // Struct field indices in a GEP are constants; the walk needs their
    // values.
    let const_ints: HashMap<HirId, i64> = func
        .values
        .iter()
        .filter_map(|(id, v)| match &v.kind {
            HirValueKind::Constant(c) => constant_int(c).map(|n| (*id, n)),
            _ => None,
        })
        .collect();

    // Assign a register to every SSA value. Parameters first so
    // `regs[0..n_params]` lines up with the call ABI.
    let mut next_reg: Reg = 0;
    let mut alloc = |id: HirId,
                     ty: &HirType,
                     reg_of: &mut HashMap<HirId, Reg>,
                     reg_types: &mut Vec<HirType>,
                     next: &mut Reg|
     -> Reg {
        if let Some(r) = reg_of.get(&id).copied() {
            return r;
        }
        let r = *next;
        *next = next.checked_add(1).expect("register overflow");
        reg_of.insert(id, r);
        reg_types.push(ty.clone());
        r
    };

    // Bind parameter regs (index 0..n_params) using both the
    // signature.params[i].id and any matching Parameter(i) HirValue.
    for (i, param) in func.signature.params.iter().enumerate() {
        let r = alloc(
            param.id,
            &param.ty,
            &mut reg_of,
            &mut cf.reg_types,
            &mut next_reg,
        );
        for (val_id, val_def) in func.values.iter() {
            if matches!(val_def.kind, HirValueKind::Parameter(idx) if idx as usize == i) {
                reg_of.insert(*val_id, r);
            }
        }
    }
    cf.n_params = func.signature.params.len() as u16;

    // Allocate regs for every remaining value (constants, instruction
    // results, phi results). For constants, also push to const_pool.
    let mut const_idx_for: HashMap<HirId, u32> = HashMap::new();
    for (val_id, val_def) in func.values.iter() {
        if reg_of.contains_key(val_id) {
            continue;
        }
        let _ = alloc(
            *val_id,
            &val_def.ty,
            &mut reg_of,
            &mut cf.reg_types,
            &mut next_reg,
        );
        if let HirValueKind::Constant(c) = &val_def.kind {
            if held(&val_def.ty) == Held::ByReference {
                // An aggregate constant gets storage of its own each
                // time the function runs, as a stack slot would.
                let mut bytes = vec![0u8; size_of_hir_ty(&val_def.ty).max(8)];
                write_constant(c, &val_def.ty, &mut bytes);
                cf.entry_storage.push((reg_of[val_id], bytes));
                continue;
            }
            let idx = cf.const_pool.len() as u32;
            cf.const_pool.push(const_to_zyntax(c));
            const_idx_for.insert(*val_id, idx);
        } else if matches!(val_def.kind, HirValueKind::Undef)
            && held(&val_def.ty) == Held::ByReference
        {
            // An undef aggregate is storage the function fills in.
            let size = size_of_hir_ty(&val_def.ty).max(8);
            cf.entry_storage.push((reg_of[val_id], vec![0u8; size]));
        } else if let HirValueKind::Global(global_id) = &val_def.kind {
            // Resolve the global to a ZRTL string image in `memory`,
            // then pre-load the pointer into the value's register via
            // `LoadConst`.
            // The BC interpreter has no `Op::AddressOfGlobal` today,
            // so we bake the address into the const pool exactly
            // like a numeric Constant — the host bridge sees a real
            // ZRTL string pointer and reads it the same way as a
            // runtime-allocated one.
            if let Some(global) = module.globals.get(global_id) {
                if let Some(&ptr) = memory.globals.get(global_id) {
                    let idx = cf.const_pool.len() as u32;
                    cf.const_pool.push(ZyntaxValue::Pointer(ptr));
                    const_idx_for.insert(*val_id, idx);
                    cf.global_consts.push((*global_id, idx));
                } else if let Some(HirConstant::String(interned)) = &global.initializer {
                    let s = interned.resolve_global().unwrap_or_default();
                    let ptr = memory.string_constant(s.as_bytes());
                    let idx = cf.const_pool.len() as u32;
                    cf.const_pool.push(ZyntaxValue::Pointer(ptr));
                    const_idx_for.insert(*val_id, idx);
                    cf.global_consts.push((*global_id, idx));
                } else {
                    // A variable: one zeroed slot for the whole module.
                    let ptr = memory.global_slot(*global_id, size_of_hir_ty(&global.ty));
                    let idx = cf.const_pool.len() as u32;
                    cf.const_pool.push(ZyntaxValue::Pointer(ptr));
                    const_idx_for.insert(*val_id, idx);
                    cf.global_consts.push((*global_id, idx));
                }
            }
        }
    }
    // Phi result regs.
    for block in func.blocks.values() {
        for phi in &block.phis {
            let _ = alloc(
                phi.result,
                &phi.ty,
                &mut reg_of,
                &mut cf.reg_types,
                &mut next_reg,
            );
        }
    }
    // Instruction result regs (in case a value isn't in `func.values`).
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let Some(r) = inst_result(inst) {
                let ty = inst_result_ty(inst).unwrap_or(HirType::I64);
                let _ = alloc(r, &ty, &mut reg_of, &mut cf.reg_types, &mut next_reg);
            }
        }
    }
    cf.n_regs = next_reg as u32;

    // Loop headers promoted code can resume at, keyed by block.
    let mut header_sites: HashMap<HirId, u32> = HashMap::new();
    if !address_taken {
        let headers = crate::osr::find_loop_headers(func);
        if !headers.is_empty() {
            let dominators = crate::osr::Dominators::compute(func);
            for header in headers {
                let layout = match crate::osr::osr_layout_with(func, header, &dominators) {
                    Ok(layout) => layout,
                    Err(why) => {
                        if trace_enabled() {
                            eprintln!(
                                "[interp] {} header {:?}: no site, {why:?}",
                                func.name.resolve_global().unwrap_or_default(),
                                header
                            );
                        }
                        continue;
                    }
                };
                let live_ins: Option<Vec<Reg>> = layout
                    .live_ins
                    .iter()
                    .map(|id| reg_of.get(id).copied())
                    .collect();
                let Some(live_ins) = live_ins else {
                    if trace_enabled() {
                        eprintln!(
                            "[interp] {} header {:?}: no site, a live-in has no register",
                            func.name.resolve_global().unwrap_or_default(),
                            header
                        );
                    }
                    continue;
                };
                header_sites.insert(header, cf.osr_sites.len() as u32);
                cf.osr_sites.push(OsrSite {
                    site_key: layout.site_key(),
                    live_ins,
                    types: layout.live_in_types.clone(),
                    frame: layout.frame.clone(),
                    ret: layout.return_type.clone(),
                    destination: layout.destination,
                });
            }
        }
    }

    // ── Code emission ──
    // First pass: emit ops; record block-id → start PC; track every
    // jump-target site so we can backpatch after pass 1.
    let mut block_pcs: HashMap<HirId, Pc> = HashMap::new();
    // Patch entry: (op_index, slot) — slot 0..N selects which Pc field
    // in the op to overwrite.
    let mut patches: Vec<(usize, u8, HirId)> = Vec::new();
    // Per-switch-case patches: (table_idx, case_idx, target_block).
    let mut switch_case_patches: Vec<(u32, usize, HirId)> = Vec::new();

    // Emit the entry block first, then the rest in iteration order.
    let mut order: Vec<HirId> = Vec::new();
    order.push(func.entry_block);
    for bid in func.blocks.keys() {
        if *bid != func.entry_block {
            order.push(*bid);
        }
    }

    for bid in &order {
        let pc = cf.code.len() as Pc;
        block_pcs.insert(*bid, pc);
        let block = func
            .blocks
            .get(bid)
            .ok_or_else(|| InterpError::Host(format!("missing block {:?}", bid)))?;

        // Constants used in this block get their LoadConst emitted at
        // the block entry. Simplest correct strategy; refinement is to
        // hoist to entry block. (Constants from `func.values` only —
        // they're function-scoped, not block-scoped, in SSA, so a
        // single load at the block prefix is fine since each block
        // dominates its uses for constants.)

        if let Some(&site) = header_sites.get(bid) {
            cf.code.push(Op::LoopHeader { site });
        }

        // Lower each instruction.
        for inst in &block.instructions {
            lower_inst(inst, &mut cf, &reg_of, &const_ints)?;
        }

        // Lower terminator (phi-copy preamble for branch targets is
        // emitted by lower_terminator).
        lower_terminator(
            &block.terminator,
            *bid,
            func,
            &mut cf,
            &reg_of,
            &mut patches,
            &mut switch_case_patches,
        )?;
    }

    // Backpatch jump targets.
    for (op_idx, slot, target_bid) in patches {
        let target_pc = *block_pcs
            .get(&target_bid)
            .ok_or_else(|| InterpError::Host(format!("unresolved block {:?}", target_bid)))?;
        match (&mut cf.code[op_idx], slot) {
            (Op::Jump { target }, 0) => *target = target_pc,
            (Op::JumpIf { t, .. }, 0) => *t = target_pc,
            (Op::JumpIf { f, .. }, 1) => *f = target_pc,
            (Op::Switch { default, .. }, 0) => *default = target_pc,
            _ => {}
        }
    }
    // Backpatch switch-case PCs.
    for (table_idx, case_idx, target_bid) in switch_case_patches {
        let target_pc = *block_pcs.get(&target_bid).ok_or_else(|| {
            InterpError::Host(format!("unresolved switch block {:?}", target_bid))
        })?;
        cf.switch_pool[table_idx as usize][case_idx].1 = target_pc;
    }

    // Hoist constant LoadConst ops to the entry-block prefix in
    // source order. This is the simplest correct approach: re-emit the
    // entry block's prefix by shifting `code[entry_pc..]` right by N
    // ops where N = number of constants used.
    // Skipped for now — instead, when the dispatcher hits a register
    // that's never been written, it falls through to consulting the
    // function's const_pool by examining `const_idx_for` via a
    // side-channel. To keep things simple, we instead emit one
    // `LoadConst` per constant at the top of the entry block.
    inject_const_loads_at_entry(func, &mut cf, &reg_of, &const_idx_for, &block_pcs)?;

    Ok(cf)
}

/// Insert `LoadConst` ops at the start of the entry block (which is
/// always at PC 0) for every constant in the function. Patches all
/// recorded backpatch sites + block_pcs to account for the shift.
fn inject_const_loads_at_entry(
    _func: &HirFunction,
    cf: &mut CompiledFunction,
    _reg_of: &HashMap<HirId, Reg>,
    const_idx_for: &HashMap<HirId, u32>,
    _block_pcs: &HashMap<HirId, Pc>,
) -> Result<(), InterpError> {
    if const_idx_for.is_empty() {
        return Ok(());
    }
    // Build prefix ops.
    let mut prefix: Vec<Op> = Vec::with_capacity(const_idx_for.len());
    let mut entries: Vec<(HirId, u32)> =
        const_idx_for.iter().map(|(id, idx)| (*id, *idx)).collect();
    entries.sort_by_key(|(_, idx)| *idx);
    for (id, idx) in entries {
        let r = _reg_of[&id];
        prefix.push(Op::LoadConst { dst: r, c: idx });
    }
    let shift = prefix.len() as Pc;

    // Shift every Pc inside existing ops by `shift`.
    for op in cf.code.iter_mut() {
        match op {
            Op::Jump { target } => *target += shift,
            Op::JumpIf { t, f, .. } => {
                *t += shift;
                *f += shift;
            }
            Op::Switch { default, .. } => *default += shift,
            _ => {}
        }
    }
    // Shift switch-pool case targets too.
    for table in cf.switch_pool.iter_mut() {
        for (_, pc) in table.iter_mut() {
            *pc += shift;
        }
    }

    // Splice prefix into the front of code.
    let mut new_code = Vec::with_capacity(prefix.len() + cf.code.len());
    new_code.extend(prefix);
    new_code.extend(cf.code.drain(..));
    cf.code = new_code;
    Ok(())
}

fn inst_result(inst: &HirInstruction) -> Option<HirId> {
    match inst {
        HirInstruction::Binary { result, .. }
        | HirInstruction::Unary { result, .. }
        | HirInstruction::Cast { result, .. }
        | HirInstruction::Select { result, .. }
        | HirInstruction::Alloca { result, .. }
        | HirInstruction::Load { result, .. }
        | HirInstruction::ExtractValue { result, .. }
        | HirInstruction::InsertValue { result, .. } => Some(*result),
        HirInstruction::Call { result, .. } => *result,
        _ => None,
    }
}

fn inst_result_ty(inst: &HirInstruction) -> Option<HirType> {
    match inst {
        HirInstruction::Binary { ty, .. }
        | HirInstruction::Unary { ty, .. }
        | HirInstruction::Cast { ty, .. }
        | HirInstruction::Select { ty, .. }
        | HirInstruction::Alloca { ty, .. }
        | HirInstruction::Load { ty, .. } => Some(ty.clone()),
        // Calls return whatever the signature says — left as I64 for
        // now and re-tagged in the dispatcher on the way back.
        HirInstruction::Call { .. } => Some(HirType::I64),
        HirInstruction::ExtractValue { ty, .. } | HirInstruction::InsertValue { ty, .. } => {
            Some(ty.clone())
        }
        _ => None,
    }
}

fn lower_inst(
    inst: &HirInstruction,
    cf: &mut CompiledFunction,
    reg_of: &HashMap<HirId, Reg>,
    const_ints: &HashMap<HirId, i64>,
) -> Result<(), InterpError> {
    let reg = |id: HirId| -> Result<Reg, InterpError> {
        reg_of
            .get(&id)
            .copied()
            .ok_or(InterpError::UndefinedSsaValue(id))
    };

    let type_idx = |cf: &mut CompiledFunction, ty: &HirType| -> u32 {
        let idx = cf.type_pool.len() as u32;
        cf.type_pool.push(ty.clone());
        idx
    };

    match inst {
        HirInstruction::Binary {
            result,
            op,
            left,
            right,
            ty,
        } => {
            let dst = reg(*result)?;
            let lhs = reg(*left)?;
            let rhs = reg(*right)?;
            // Vector-typed arithmetic scalarizes to a lane-wise op — the
            // vectorization passes and `@kernel` emit `Binary` with a
            // `HirType::Vector` operand type (no dedicated vector-arith op).
            if matches!(ty, HirType::Vector(..)) {
                cf.code.push(Op::VBinOp {
                    dst,
                    lhs,
                    rhs,
                    op: *op,
                });
                return Ok(());
            }
            // Float vs integer selected by HirType.
            let is_float = matches!(ty, HirType::F32 | HirType::F64);
            let op = if is_float {
                match op {
                    BinaryOp::Add | BinaryOp::FAdd => Op::FAdd { dst, lhs, rhs },
                    BinaryOp::Sub | BinaryOp::FSub => Op::FSub { dst, lhs, rhs },
                    BinaryOp::Mul | BinaryOp::FMul => Op::FMul { dst, lhs, rhs },
                    BinaryOp::Div | BinaryOp::FDiv => Op::FDiv { dst, lhs, rhs },
                    BinaryOp::Eq | BinaryOp::FEq => Op::FCmpEq { dst, lhs, rhs },
                    BinaryOp::Ne | BinaryOp::FNe => Op::FCmpNe { dst, lhs, rhs },
                    BinaryOp::Lt | BinaryOp::FLt => Op::FCmpLt { dst, lhs, rhs },
                    BinaryOp::Le | BinaryOp::FLe => Op::FCmpLe { dst, lhs, rhs },
                    BinaryOp::Gt | BinaryOp::FGt => Op::FCmpGt { dst, lhs, rhs },
                    BinaryOp::Ge | BinaryOp::FGe => Op::FCmpGe { dst, lhs, rhs },
                    other => {
                        return Err(InterpError::UnsupportedInstruction(format!(
                            "float binary op {:?}",
                            other
                        )));
                    }
                }
            } else {
                // On an unsigned type, what the sign would change is
                // done unsigned.
                let unsigned = matches!(
                    ty,
                    HirType::U8 | HirType::U16 | HirType::U32 | HirType::U64 | HirType::USize
                );
                match op {
                    BinaryOp::Add => Op::IAdd { dst, lhs, rhs },
                    BinaryOp::Sub => Op::ISub { dst, lhs, rhs },
                    BinaryOp::Mul => Op::IMul { dst, lhs, rhs },
                    BinaryOp::Div if unsigned => Op::UDiv { dst, lhs, rhs },
                    BinaryOp::Div => Op::IDiv { dst, lhs, rhs },
                    BinaryOp::Rem if unsigned => Op::URem { dst, lhs, rhs },
                    BinaryOp::Rem => Op::IRem { dst, lhs, rhs },
                    BinaryOp::And => Op::IAnd { dst, lhs, rhs },
                    BinaryOp::Or => Op::IOr { dst, lhs, rhs },
                    BinaryOp::Xor => Op::IXor { dst, lhs, rhs },
                    BinaryOp::Shl => Op::IShl { dst, lhs, rhs },
                    BinaryOp::Shr if unsigned => Op::UShr { dst, lhs, rhs },
                    BinaryOp::Shr => Op::IShr { dst, lhs, rhs },
                    BinaryOp::Eq => Op::ICmpEq { dst, lhs, rhs },
                    BinaryOp::Ne => Op::ICmpNe { dst, lhs, rhs },
                    BinaryOp::Lt if unsigned => Op::UCmpLt { dst, lhs, rhs },
                    BinaryOp::Le if unsigned => Op::UCmpLe { dst, lhs, rhs },
                    BinaryOp::Gt if unsigned => Op::UCmpGt { dst, lhs, rhs },
                    BinaryOp::Ge if unsigned => Op::UCmpGe { dst, lhs, rhs },
                    BinaryOp::Lt => Op::ICmpLt { dst, lhs, rhs },
                    BinaryOp::Le => Op::ICmpLe { dst, lhs, rhs },
                    BinaryOp::Gt => Op::ICmpGt { dst, lhs, rhs },
                    BinaryOp::Ge => Op::ICmpGe { dst, lhs, rhs },
                    // Float opcodes on integer type fall back to int.
                    BinaryOp::FAdd => Op::IAdd { dst, lhs, rhs },
                    BinaryOp::FSub => Op::ISub { dst, lhs, rhs },
                    BinaryOp::FMul => Op::IMul { dst, lhs, rhs },
                    BinaryOp::FDiv => Op::IDiv { dst, lhs, rhs },
                    BinaryOp::FRem => Op::IRem { dst, lhs, rhs },
                    BinaryOp::FEq => Op::ICmpEq { dst, lhs, rhs },
                    BinaryOp::FNe => Op::ICmpNe { dst, lhs, rhs },
                    BinaryOp::FLt => Op::ICmpLt { dst, lhs, rhs },
                    BinaryOp::FLe => Op::ICmpLe { dst, lhs, rhs },
                    BinaryOp::FGt => Op::ICmpGt { dst, lhs, rhs },
                    BinaryOp::FGe => Op::ICmpGe { dst, lhs, rhs },
                }
            };
            cf.code.push(op);
        }
        HirInstruction::Unary {
            result,
            op,
            operand,
            ty,
        } => {
            let dst = reg(*result)?;
            let src = reg(*operand)?;
            let is_float = matches!(ty, HirType::F32 | HirType::F64);
            let op = match (op, is_float) {
                (UnaryOp::Neg, true) | (UnaryOp::FNeg, _) => Op::FNeg { dst, src },
                (UnaryOp::Neg, false) => Op::INeg { dst, src },
                (UnaryOp::Not, _) => Op::BNot { dst, src },
            };
            cf.code.push(op);
        }
        HirInstruction::Cast {
            result,
            ty,
            op,
            operand,
        } => {
            let dst = reg(*result)?;
            let src = reg(*operand)?;
            let ty_idx = type_idx(cf, ty);
            cf.code.push(Op::Cast {
                dst,
                src,
                op: *op,
                ty: ty_idx,
            });
        }
        HirInstruction::Select {
            result,
            condition,
            true_val,
            false_val,
            ..
        } => {
            cf.code.push(Op::Select {
                dst: reg(*result)?,
                cond: reg(*condition)?,
                yes: reg(*true_val)?,
                no: reg(*false_val)?,
            });
        }
        HirInstruction::Alloca {
            result, ty, count, ..
        } => {
            let dst = reg(*result)?;
            let elem_bytes = size_of_hir_ty(ty) as u32;
            // Count is dynamic if it's an SSA value — for now, take 1
            // as a static fallback. (Most Allocas in current code have
            // count = None.)
            let n = match count {
                Some(_) => 1,
                None => 1,
            };
            cf.code.push(Op::Alloca {
                dst,
                size_bytes: elem_bytes.saturating_mul(n).max(1),
            });
        }
        HirInstruction::Load {
            result, ty, ptr, ..
        } => {
            let dst = reg(*result)?;
            let p = reg(*ptr)?;
            let ty_idx = type_idx(cf, ty);
            cf.code.push(Op::Load {
                dst,
                ptr: p,
                ty: ty_idx,
            });
        }
        HirInstruction::Store { value, ptr, .. } => {
            let v = reg(*value)?;
            let p = reg(*ptr)?;
            // Store has no inline type — pull it from the value's reg
            // type so Load/Store agree on width.
            let val_ty = cf
                .reg_types
                .get(v as usize)
                .cloned()
                .unwrap_or(HirType::I64);
            let ty_idx = type_idx(cf, &val_ty);
            cf.code.push(Op::Store {
                ptr: p,
                val: v,
                ty: ty_idx,
            });
        }
        HirInstruction::ExtractValue {
            result,
            aggregate,
            indices,
            ..
        } => {
            let dst = reg(*result)?;
            let src = reg(*aggregate)?;
            let idx = cf.indices_pool.len() as u32;
            cf.indices_pool.push(indices.clone());
            let agg_ty = cf.reg_types[src as usize].clone();
            let ty = type_idx(cf, &agg_ty);
            cf.code.push(Op::ExtractValue { dst, src, idx, ty });
        }
        HirInstruction::GetElementPtr {
            result,
            ty,
            ptr,
            indices,
        } => {
            // The same walk Cranelift makes: `ty` names what `ptr`
            // points at; a pointer or array index scales by the element
            // size, a struct index is a constant naming a field, and a
            // byte type adds the index as it is.
            let mut steps: Vec<(i64, i64)> = Vec::with_capacity(indices.len());
            let mut current = ty.clone();
            for index in indices {
                match current.clone() {
                    HirType::Ptr(inner) => {
                        steps.push((size_of_hir_ty(&inner).max(1) as i64, 0));
                        current = *inner;
                    }
                    HirType::Array(elem, _) => {
                        steps.push((size_of_hir_ty(&elem).max(1) as i64, 0));
                        current = *elem;
                    }
                    HirType::Struct(st) => {
                        let field = const_ints.get(index).copied().unwrap_or(0).max(0) as usize;
                        let field = field.min(st.fields.len().saturating_sub(1));
                        let layout = struct_layout(&st);
                        steps.push((0, layout.offsets.get(field).copied().unwrap_or(0) as i64));
                        current = st.fields.get(field).cloned().unwrap_or(HirType::U8);
                    }
                    HirType::U8 | HirType::I8 => steps.push((1, 0)),
                    _ => {
                        steps.push((size_of_hir_ty(&current).max(1) as i64, 0));
                    }
                }
            }
            let dst = reg(*result)?;
            let p = reg(*ptr)?;
            let idx_regs: Result<Vec<Reg>, InterpError> = indices.iter().map(|i| reg(*i)).collect();
            let args = cf.args_pool.len() as u32;
            cf.args_pool.push(idx_regs?);
            let stride = cf.gep_stride_pool.len() as u32;
            cf.gep_stride_pool.push(steps);
            cf.code.push(Op::Gep {
                dst,
                ptr: p,
                args,
                stride,
            });
        }
        HirInstruction::InsertValue {
            result,
            aggregate,
            value,
            indices,
            ..
        } => {
            let dst = reg(*result)?;
            let agg = reg(*aggregate)?;
            let val = reg(*value)?;
            let idx = cf.indices_pool.len() as u32;
            cf.indices_pool.push(indices.clone());
            let agg_ty = cf.reg_types[agg as usize].clone();
            let ty = type_idx(cf, &agg_ty);
            cf.code.push(Op::InsertValue {
                dst,
                agg,
                val,
                idx,
                ty,
            });
        }
        HirInstruction::Call {
            result,
            callee,
            args,
            ..
        } => {
            let arg_regs: Result<Vec<Reg>, InterpError> = args.iter().map(|a| reg(*a)).collect();
            let arg_regs = arg_regs?;
            let args_idx = cf.args_pool.len() as u32;
            cf.args_pool.push(arg_regs);
            let (dst, has_dst) = match result {
                Some(r) => (reg(*r)?, true),
                None => (0, false),
            };
            match callee {
                HirCallable::Function(fn_id) => cf.code.push(Op::CallFn {
                    dst,
                    has_dst,
                    fn_id: *fn_id,
                    args: args_idx,
                }),
                // Handing a range to a thread pool is one way of
                // running it, and running it here is another. The
                // interpreter has no address to give the pool, so it
                // calls the band over the whole range instead, which
                // computes the same thing in the same order the loop
                // would have.
                HirCallable::Symbol(name) if name == "zyntax_parallel_for" && args.len() == 5 => {
                    let band = cf.func_refs.get(&args[3]).copied().ok_or_else(|| {
                        InterpError::UnsupportedInstruction(
                            "a spread loop whose band is not a known function".to_string(),
                        )
                    })?;
                    let whole = vec![reg(args[0])?, reg(args[1])?, reg(args[4])?];
                    let band_args = cf.args_pool.len() as u32;
                    cf.args_pool.push(whole);
                    cf.code.push(Op::CallFn {
                        dst: 0,
                        has_dst: false,
                        fn_id: band,
                        args: band_args,
                    });
                }
                HirCallable::Symbol(name) => {
                    let sym_idx = cf.symbol_pool.len() as u32;
                    cf.symbol_pool.push(name.clone());
                    // The call's shape is its operand types: a symbol
                    // has no declaration here.
                    let ret = result
                        .and_then(|r| reg_of.get(&r).copied())
                        .and_then(|r| cf.reg_types.get(r as usize).cloned())
                        .unwrap_or(HirType::Void);
                    let params: Vec<HirType> = cf.args_pool[args_idx as usize]
                        .iter()
                        .map(|r| cf.reg_types[*r as usize].clone())
                        .collect();
                    let sig = cf.sig_pool.len() as u32;
                    cf.sig_pool.push(NativeSig::of_site(params, ret));
                    cf.sig_thunks.push(Default::default());
                    cf.code.push(Op::CallSym {
                        dst,
                        has_dst,
                        sym: sym_idx,
                        args: args_idx,
                        sig,
                    });
                }
                HirCallable::Indirect(target) => {
                    // A call through a pointer: its shape is the operand
                    // types and the result's, as a call by symbol's is.
                    let fn_ptr_reg = reg(*target)?;
                    let ret = result
                        .and_then(|r| reg_of.get(&r).copied())
                        .and_then(|r| cf.reg_types.get(r as usize).cloned())
                        .unwrap_or(HirType::Void);
                    let params: Vec<HirType> = cf.args_pool[args_idx as usize]
                        .iter()
                        .map(|r| cf.reg_types[*r as usize].clone())
                        .collect();
                    let sig = cf.sig_pool.len() as u32;
                    cf.sig_pool.push(NativeSig::of_site(params, ret));
                    cf.sig_thunks.push(Default::default());
                    cf.code.push(Op::CallIndirect {
                        dst,
                        has_dst,
                        fn_ptr_reg,
                        args: args_idx,
                        sig,
                    });
                }
                HirCallable::Intrinsic(crate::hir::Intrinsic::Malloc) => {
                    // First arg carries the size in bytes.
                    let size_reg = cf
                        .args_pool
                        .get(args_idx as usize)
                        .and_then(|args| args.first().copied())
                        .unwrap_or(0);
                    cf.code.push(Op::Malloc {
                        dst,
                        has_dst,
                        size_reg,
                    });
                }
                HirCallable::Intrinsic(crate::hir::Intrinsic::Realloc) => {
                    let regs = &cf.args_pool[args_idx as usize];
                    if regs.len() != 2 || !has_dst {
                        return Err(InterpError::UnsupportedInstruction(
                            "realloc expects pointer and size".to_string(),
                        ));
                    }
                    cf.code.push(Op::Realloc {
                        dst,
                        old_reg: regs[0],
                        size_reg: regs[1],
                    });
                }
                HirCallable::Intrinsic(crate::hir::Intrinsic::Free) => {
                    let regs = &cf.args_pool[args_idx as usize];
                    if let Some(&ptr) = regs.first() {
                        cf.code.push(Op::Free { ptr });
                    }
                }
                HirCallable::Intrinsic(crate::hir::Intrinsic::IncRef)
                | HirCallable::Intrinsic(crate::hir::Intrinsic::DecRef)
                | HirCallable::Intrinsic(crate::hir::Intrinsic::Drop) => {
                    // Reference counts are not kept here; the storage
                    // outlives the run or the collector takes it.
                }
                HirCallable::Intrinsic(crate::hir::Intrinsic::Sqrt) => {
                    // Single-arg math intrinsic — mirror Cranelift's
                    // hardware FSQRT so source-side `sqrt(x)` works
                    // through the BC interp too.
                    let src_reg = cf
                        .args_pool
                        .get(args_idx as usize)
                        .and_then(|args| args.first().copied())
                        .unwrap_or(0);
                    cf.code.push(Op::FSqrt { dst, src: src_reg });
                }
                HirCallable::Intrinsic(crate::hir::Intrinsic::Rsqrt) => {
                    // Single-arg reciprocal-square-root intrinsic —
                    // mirror Cranelift's `fdiv(1.0, sqrt(x))` lowering
                    // so source-side `rsqrt(x)` works through the BC
                    // interp too.
                    let src_reg = cf
                        .args_pool
                        .get(args_idx as usize)
                        .and_then(|args| args.first().copied())
                        .unwrap_or(0);
                    cf.code.push(Op::FRsqrt { dst, src: src_reg });
                }
                HirCallable::Intrinsic(crate::hir::Intrinsic::Fabs) => {
                    // Single-arg math intrinsic — mirror Cranelift's
                    // hardware FABS so source-side `abs(x)` works
                    // through the BC interp too.
                    let src_reg = cf
                        .args_pool
                        .get(args_idx as usize)
                        .and_then(|args| args.first().copied())
                        .unwrap_or(0);
                    cf.code.push(Op::FAbs { dst, src: src_reg });
                }
                HirCallable::Intrinsic(crate::hir::Intrinsic::Floor) => {
                    let src_reg = cf
                        .args_pool
                        .get(args_idx as usize)
                        .and_then(|args| args.first().copied())
                        .unwrap_or(0);
                    cf.code.push(Op::FFloor { dst, src: src_reg });
                }
                HirCallable::Intrinsic(
                    intrinsic @ (crate::hir::Intrinsic::Sin
                    | crate::hir::Intrinsic::Cos
                    | crate::hir::Intrinsic::Log
                    | crate::hir::Intrinsic::Exp),
                ) => {
                    let src_reg = cf
                        .args_pool
                        .get(args_idx as usize)
                        .and_then(|args| args.first().copied())
                        .unwrap_or(0);
                    let op = match intrinsic {
                        crate::hir::Intrinsic::Sin => FUnaryOp::Sin,
                        crate::hir::Intrinsic::Cos => FUnaryOp::Cos,
                        crate::hir::Intrinsic::Log => FUnaryOp::Log,
                        _ => FUnaryOp::Exp,
                    };
                    cf.code.push(Op::FUnary {
                        dst,
                        src: src_reg,
                        op,
                    });
                }
                HirCallable::Intrinsic(crate::hir::Intrinsic::Pow) => {
                    let arg_regs = cf
                        .args_pool
                        .get(args_idx as usize)
                        .cloned()
                        .unwrap_or_default();
                    if arg_regs.len() != 2 {
                        return Err(InterpError::UnsupportedInstruction(
                            "pow with other than two arguments".to_string(),
                        ));
                    }
                    cf.code.push(Op::FPow {
                        dst,
                        a: arg_regs[0],
                        b: arg_regs[1],
                    });
                }
                HirCallable::Intrinsic(crate::hir::Intrinsic::Fma) => {
                    // Three-arg math intrinsic — emitted by the
                    // `fma_contract` HIR pass when it rewrites
                    // `fadd(fmul a b, c)`. Mirror Cranelift's
                    // hardware FMA via `f64::mul_add`.
                    let arg_regs = cf
                        .args_pool
                        .get(args_idx as usize)
                        .cloned()
                        .unwrap_or_default();
                    let a = arg_regs.first().copied().unwrap_or(0);
                    let b = arg_regs.get(1).copied().unwrap_or(0);
                    let c = arg_regs.get(2).copied().unwrap_or(0);
                    cf.code.push(Op::FMulAdd { dst, a, b, c });
                }
                HirCallable::Intrinsic(
                    intrinsic @ (crate::hir::Intrinsic::Memcpy
                    | crate::hir::Intrinsic::Memmove
                    | crate::hir::Intrinsic::Memset),
                ) => {
                    let regs = &cf.args_pool[args_idx as usize];
                    if regs.len() != 3 {
                        return Err(InterpError::UnsupportedInstruction(
                            "memory intrinsic expects destination, source and length".to_string(),
                        ));
                    }
                    let kind = match intrinsic {
                        crate::hir::Intrinsic::Memcpy => MemOpKind::Copy,
                        crate::hir::Intrinsic::Memmove => MemOpKind::Move,
                        _ => MemOpKind::Set,
                    };
                    cf.code.push(Op::MemOp {
                        kind,
                        dst_ptr: regs[0],
                        src: regs[1],
                        len: regs[2],
                    });
                }
                HirCallable::Intrinsic(intrinsic) => {
                    return Err(InterpError::UnsupportedInstruction(format!(
                        "intrinsic call {intrinsic:?}"
                    )));
                }
                HirCallable::FuncRef(fn_id) => {
                    if let Some(r) = result {
                        cf.func_refs.insert(*r, *fn_id);
                    }
                    cf.code.push(Op::FuncAddr { dst, fn_id: *fn_id });
                }
            }
        }
        HirInstruction::IndirectCall {
            result,
            func_ptr,
            args,
            return_ty,
        } => {
            let fn_ptr_reg = reg(*func_ptr)?;
            let arg_regs: Result<Vec<Reg>, InterpError> = args.iter().map(|a| reg(*a)).collect();
            let args_idx = cf.args_pool.len() as u32;
            cf.args_pool.push(arg_regs?);
            let (dst, has_dst) = match result {
                Some(r) => (reg(*r)?, true),
                None => (0, false),
            };
            let params: Vec<HirType> = cf.args_pool[args_idx as usize]
                .iter()
                .map(|r| cf.reg_types[*r as usize].clone())
                .collect();
            let sig = cf.sig_pool.len() as u32;
            cf.sig_pool
                .push(NativeSig::of_site(params, return_ty.clone()));
            cf.sig_thunks.push(Default::default());
            cf.code.push(Op::CallIndirect {
                dst,
                has_dst,
                fn_ptr_reg,
                args: args_idx,
                sig,
            });
        }
        HirInstruction::AsyncSaveSlot { frame, slot, value } => {
            // Slot is 8-byte (i64-sized) per krio's SM layout. Lower
            // to a one-step Op::AsyncSaveSlot that at execute time
            // computes `frame + slot * 8` and stores `value` as i64.
            let frame_reg = reg(*frame)?;
            let val_reg = reg(*value)?;
            cf.code.push(Op::AsyncSaveSlot {
                frame_reg,
                slot: *slot,
                val_reg,
            });
        }
        HirInstruction::AsyncLoadSlot {
            result,
            ty,
            frame,
            slot,
        } => {
            let dst = reg(*result)?;
            let frame_reg = reg(*frame)?;
            let ty_idx = cf.type_pool.len() as u32;
            cf.type_pool.push(ty.clone());
            cf.code.push(Op::AsyncLoadSlot {
                dst,
                frame_reg,
                slot: *slot,
                ty: ty_idx,
            });
        }
        HirInstruction::CreateClosure {
            result,
            function,
            captures,
            ..
        } => {
            // Capture-free closures only — Phase I.2's
            // cooperative-await emit produces these (the krio
            // emitter's CreateClosure with captures=[] referring to
            // the SM's own poll fn). The closure value is the
            // function's HirId hash so it can be re-dispatched
            // through `ACTIVE_CLOSURE_FNS` in the wasm shim
            // (Phase I.3). Native callers see the same shape and
            // can transmute through the symbol table if needed.
            if !captures.is_empty() {
                return Err(InterpError::UnsupportedInstruction(
                    "CreateClosure with captures".to_string(),
                ));
            }
            let dst = reg(*result)?;
            cf.code.push(Op::FuncAddr {
                dst,
                fn_id: *function,
            });
        }
        // ── SIMD / vector (scalarized) ──
        HirInstruction::VectorSplat { result, ty, scalar } => {
            let dst = reg(*result)?;
            let s = reg(*scalar)?;
            let lanes = match ty {
                HirType::Vector(_, n) => *n as u8,
                _ => {
                    return Err(InterpError::UnsupportedInstruction(
                        "VectorSplat with non-vector type".to_string(),
                    ));
                }
            };
            cf.code.push(Op::VSplat {
                dst,
                scalar: s,
                lanes,
            });
        }
        HirInstruction::VectorLoad {
            result, ty, ptr, ..
        } => {
            let dst = reg(*result)?;
            let p = reg(*ptr)?;
            let (elem, lanes) = match ty {
                HirType::Vector(e, n) => ((**e).clone(), *n as u8),
                _ => {
                    return Err(InterpError::UnsupportedInstruction(
                        "VectorLoad with non-vector type".to_string(),
                    ));
                }
            };
            let elem_ty = type_idx(cf, &elem);
            cf.code.push(Op::VLoad {
                dst,
                ptr: p,
                lanes,
                elem_ty,
            });
        }
        HirInstruction::VectorStore { value, ptr, .. } => {
            let v = reg(*value)?;
            let p = reg(*ptr)?;
            // The store carries no type; recover the element type from the
            // value register's `HirType::Vector(elem, _)`.
            let elem = match cf.reg_types.get(v as usize) {
                Some(HirType::Vector(e, _)) => (**e).clone(),
                _ => {
                    return Err(InterpError::UnsupportedInstruction(
                        "VectorStore of non-vector value".to_string(),
                    ));
                }
            };
            let elem_ty = type_idx(cf, &elem);
            cf.code.push(Op::VStore {
                val: v,
                ptr: p,
                elem_ty,
            });
        }
        HirInstruction::VectorExtractLane {
            result,
            vector,
            lane,
            ..
        } => {
            let dst = reg(*result)?;
            let vec = reg(*vector)?;
            cf.code.push(Op::VExtract {
                dst,
                vector: vec,
                lane: *lane,
            });
        }
        HirInstruction::VectorInsertLane {
            result,
            vector,
            scalar,
            lane,
            ..
        } => {
            let dst = reg(*result)?;
            let vec = reg(*vector)?;
            let s = reg(*scalar)?;
            cf.code.push(Op::VInsert {
                dst,
                vector: vec,
                scalar: s,
                lane: *lane,
            });
        }
        HirInstruction::VectorHorizontalReduce {
            result, vector, op, ..
        } => {
            let dst = reg(*result)?;
            let vec = reg(*vector)?;
            cf.code.push(Op::VReduce {
                dst,
                vector: vec,
                op: *op,
            });
        }
        HirInstruction::VectorUnaryOp {
            result,
            op,
            operand,
            ..
        } => {
            let dst = reg(*result)?;
            let o = reg(*operand)?;
            cf.code.push(Op::VUnary {
                dst,
                operand: o,
                kind: *op,
            });
        }
        HirInstruction::VectorMinMax {
            result,
            op,
            left,
            right,
            ..
        } => {
            let dst = reg(*result)?;
            let l = reg(*left)?;
            let r = reg(*right)?;
            cf.code.push(Op::VMinMax {
                dst,
                lhs: l,
                rhs: r,
                kind: *op,
            });
        }
        HirInstruction::VectorDot {
            result,
            acc,
            a,
            b,
            rhs_unsigned,
            ..
        } => {
            let dst = reg(*result)?;
            let acc_r = reg(*acc)?;
            let a_r = reg(*a)?;
            let b_r = reg(*b)?;
            cf.code.push(Op::VDot {
                dst,
                acc: acc_r,
                a: a_r,
                b: b_r,
                unsigned: *rhs_unsigned,
            });
        }
        other => {
            return Err(InterpError::UnsupportedInstruction(instruction_name(other)));
        }
    }
    Ok(())
}

/// Name an instruction the bytecode compiler will not take, in terms a
/// caller can act on. The default used to be the discriminant, which
/// says nothing about what the program did or what to do instead.
fn instruction_name(inst: &HirInstruction) -> String {
    let what = match inst {
        HirInstruction::PerformEffect { op_name, .. } => {
            return format!(
                "performing effect operation `{}`: the bytecode interpreter cannot run \
                 algebraic effects, so this function needs a JIT tier",
                op_name.resolve_global().unwrap_or_default()
            );
        }
        HirInstruction::FiberNew { .. } => "creating a fiber",
        HirInstruction::FiberResume { .. } | HirInstruction::FiberResumeWith { .. } => {
            "resuming a fiber"
        }
        HirInstruction::FiberYield { .. } => "yielding from a fiber",
        HirInstruction::FiberTransfer { .. } => "transferring between fibers",
        HirInstruction::FiberDrop { .. } => "dropping a fiber",
        other => {
            return format!(
                "an instruction the bytecode interpreter does not implement ({:?})",
                std::mem::discriminant(other)
            );
        }
    };
    format!("{what}: the bytecode interpreter cannot run fibers, so this function needs a JIT tier")
}

fn lower_terminator(
    term: &HirTerminator,
    src_block: HirId,
    func: &HirFunction,
    cf: &mut CompiledFunction,
    reg_of: &HashMap<HirId, Reg>,
    patches: &mut Vec<(usize, u8, HirId)>,
    switch_case_patches: &mut Vec<(u32, usize, HirId)>,
) -> Result<(), InterpError> {
    let reg = |id: HirId| -> Result<Reg, InterpError> {
        reg_of
            .get(&id)
            .copied()
            .ok_or(InterpError::UndefinedSsaValue(id))
    };

    // Helper: emit phi-copy preamble for any phis in `target` whose
    // incoming edge originates at `src_block`. Standard out-of-SSA. The
    // copies are parallel: when one phi's result feeds another's on the
    // same edge, every source is read into a temporary first.
    let mut emit_phi_copies =
        |cf: &mut CompiledFunction, target: HirId| -> Result<(), InterpError> {
            let mut copies: Vec<(Reg, Reg)> = Vec::new();
            if let Some(blk) = func.blocks.get(&target) {
                for phi in &blk.phis {
                    let dst = reg(phi.result)?;
                    for (val_id, pred) in &phi.incoming {
                        if *pred == src_block {
                            let src = reg(*val_id)?;
                            if dst != src {
                                copies.push((dst, src));
                            }
                            break;
                        }
                    }
                }
            }
            let conflict = copies
                .iter()
                .any(|(dst, _)| copies.iter().any(|(_, src)| src == dst));
            if !conflict {
                for (dst, src) in copies {
                    cf.code.push(Op::Move { dst, src });
                }
                return Ok(());
            }
            let mut temps = Vec::with_capacity(copies.len());
            for (_, src) in &copies {
                let tmp = Reg::try_from(cf.n_regs).map_err(|_| {
                    InterpError::UnsupportedInstruction("register overflow".to_string())
                })?;
                cf.n_regs += 1;
                let ty = cf.reg_types[*src as usize].clone();
                cf.reg_types.push(ty);
                cf.code.push(Op::Move {
                    dst: tmp,
                    src: *src,
                });
                temps.push(tmp);
            }
            for ((dst, _), tmp) in copies.iter().zip(temps) {
                cf.code.push(Op::Move {
                    dst: *dst,
                    src: tmp,
                });
            }
            Ok(())
        };

    match term {
        HirTerminator::Return { values } => {
            if values.is_empty() {
                cf.code.push(Op::RetVoid);
            } else {
                let r = reg(values[0])?;
                cf.code.push(Op::Ret { src: r });
            }
        }
        HirTerminator::Branch { target } => {
            emit_phi_copies(cf, *target)?;
            let op_idx = cf.code.len();
            cf.code.push(Op::Jump { target: 0 });
            patches.push((op_idx, 0, *target));
        }
        HirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } => {
            // Phi copies need to run on the taken edge only; emit them
            // inside small "thunk" blocks. Simplest correct lowering:
            // emit two jumps after the conditional, each with its
            // phi-copy preamble. Hot-path optimisation can fold this
            // later.
            let cond = reg(*condition)?;
            // Reserve thunk PCs.
            let cond_op_idx = cf.code.len();
            cf.code.push(Op::JumpIf { cond, t: 0, f: 0 });

            // True-edge thunk.
            let true_thunk_pc = cf.code.len() as Pc;
            emit_phi_copies(cf, *true_target)?;
            let true_jump_idx = cf.code.len();
            cf.code.push(Op::Jump { target: 0 });
            patches.push((true_jump_idx, 0, *true_target));

            // False-edge thunk.
            let false_thunk_pc = cf.code.len() as Pc;
            emit_phi_copies(cf, *false_target)?;
            let false_jump_idx = cf.code.len();
            cf.code.push(Op::Jump { target: 0 });
            patches.push((false_jump_idx, 0, *false_target));

            // Backpatch the JumpIf's PCs to point at the thunks.
            if let Op::JumpIf { t, f, .. } = &mut cf.code[cond_op_idx] {
                *t = true_thunk_pc;
                *f = false_thunk_pc;
            }
        }
        HirTerminator::Switch {
            value,
            cases,
            default,
        } => {
            let scrut = reg(*value)?;
            let table_idx = cf.switch_pool.len() as u32;
            // Build table now; we patch case targets afterwards.
            let mut table: Vec<(i64, Pc)> = Vec::with_capacity(cases.len());
            for (c, _target_bid) in cases {
                let k = match c {
                    HirConstant::I32(n) => *n as i64,
                    HirConstant::I64(n) => *n,
                    HirConstant::U32(n) => *n as i64,
                    HirConstant::U64(n) => *n as i64,
                    HirConstant::Bool(b) => *b as i64,
                    other => {
                        return Err(InterpError::UnsupportedInstruction(format!(
                            "switch case const {:?}",
                            other
                        )));
                    }
                };
                table.push((k, 0));
            }
            cf.switch_pool.push(table);
            let switch_op_idx = cf.code.len();
            cf.code.push(Op::Switch {
                scrut,
                table: table_idx,
                default: 0,
            });
            patches.push((switch_op_idx, 0, *default));
            for (i, (_, target_bid)) in cases.iter().enumerate() {
                switch_case_patches.push((table_idx, i, *target_bid));
            }
        }
        HirTerminator::Unreachable => {
            cf.code.push(Op::Unreachable);
        }
        HirTerminator::Invoke { .. } => {
            return Err(InterpError::UnsupportedInstruction(
                "Invoke (unwinding) not yet implemented".to_string(),
            ));
        }
        HirTerminator::PatternMatch { .. } => {
            return Err(InterpError::UnsupportedInstruction(
                "PatternMatch raw terminator — should be lowered before BC compilation".to_string(),
            ));
        }
    }
    Ok(())
}

/// Byte size of a value of `ty` in memory, laid out as the native
/// backends lay it out: fields at their natural alignment, the total
/// rounded up to the struct's alignment.
pub fn size_of_hir_ty(ty: &HirType) -> usize {
    match ty {
        HirType::Void => 0,
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 => 8,
        HirType::Ptr(_) | HirType::USize | HirType::ISize => 8,
        HirType::I128 | HirType::U128 => 16,
        HirType::Struct(s) => struct_layout(s).size,
        HirType::Array(elem, n) => size_of_hir_ty(elem).saturating_mul((*n) as usize),
        HirType::Vector(elem, n) => size_of_hir_ty(elem).saturating_mul((*n) as usize),
        _ => 8,
    }
}

/// Alignment of a value of `ty` in memory.
pub fn align_of_hir_ty(ty: &HirType) -> usize {
    match ty {
        HirType::Void => 1,
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 => 8,
        HirType::Ptr(_) | HirType::USize | HirType::ISize => 8,
        HirType::I128 | HirType::U128 => 16,
        HirType::Struct(s) => struct_layout(s).align,
        HirType::Array(elem, _) | HirType::Vector(elem, _) => align_of_hir_ty(elem),
        _ => 8,
    }
}

/// Where a struct's fields sit.
pub struct StructLayout {
    pub offsets: Vec<usize>,
    pub size: usize,
    pub align: usize,
}

/// The layout of `s`: each field at its natural alignment unless the
/// struct is packed, the size rounded up to the struct's alignment.
pub fn struct_layout(s: &crate::hir::HirStructType) -> StructLayout {
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(s.fields.len());
    let mut align = 1usize;
    for field in &s.fields {
        let a = if s.packed { 1 } else { align_of_hir_ty(field) };
        align = align.max(a);
        offset = offset.div_ceil(a) * a;
        offsets.push(offset);
        offset += size_of_hir_ty(field);
    }
    let size = offset.div_ceil(align) * align;
    StructLayout {
        offsets,
        size,
        align,
    }
}

/// How the interpreter holds a value of `ty` in a register.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Held {
    /// As the value itself.
    Scalar,
    /// As the address of its storage: a multi-field struct, an array,
    /// a union.
    ByReference,
    /// A struct of one scalar field, held as that field.
    Flattened,
}

/// How a value of `ty` is held, as the native backends hold it.
pub fn held(ty: &HirType) -> Held {
    match ty {
        HirType::Struct(s) if crate::abi::struct_carried_as_its_field(s).is_some() => {
            Held::Flattened
        }
        HirType::Struct(_) | HirType::Array(_, _) | HirType::Union(_) => Held::ByReference,
        _ => Held::Scalar,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Profiling + memory + symbol table
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone, Copy)]
pub struct ProfileSample {
    pub call_count: u64,
    pub instructions_executed: u64,
}

#[derive(Default)]
pub struct Memory {
    allocations: Vec<Box<[u8]>>,
    native_allocator: bool,
    /// One storage slot per module global, shared by every function
    /// that names it.
    globals: HashMap<HirId, *mut u8>,
}

impl Memory {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn use_native_allocator(&mut self) {
        self.native_allocator = true;
    }
    /// A string constant's image, sixteen-aligned, never written or
    /// released.
    pub fn string_constant(&mut self, bytes: &[u8]) -> *mut u8 {
        let image = ::zrtl::string::encode_constant(bytes);
        // A pool block is sixteen-aligned; room to align any other.
        let block = self.alloc_zeroed(image.len() + 15);
        let ptr = unsafe { block.add(block.align_offset(16)) };
        unsafe { std::ptr::copy_nonoverlapping(image.as_ptr(), ptr, image.len()) };
        ptr
    }

    pub fn alloc_zeroed(&mut self, n_bytes: usize) -> *mut u8 {
        if self.native_allocator {
            // The compiled tier uses this same allocator. In particular,
            // a native realloc may receive a pointer created here.
            let ptr = unsafe { crate::pool_alloc::zyntax_alloc(n_bytes.max(1)) };
            if !ptr.is_null() {
                unsafe { std::ptr::write_bytes(ptr, 0, n_bytes.max(1)) };
            }
            return ptr;
        }
        let mut bytes: Box<[u8]> = vec![0u8; n_bytes].into_boxed_slice();
        let ptr = bytes.as_mut_ptr();
        self.allocations.push(bytes);
        ptr
    }
    /// Return a block the program frees. Only the shared allocator can
    /// take one back; the interpreter's own memory lives until it drops.
    pub fn free(&mut self, ptr: *mut u8) {
        if self.native_allocator && !ptr.is_null() {
            // SAFETY: the program frees what it allocated through the
            // same allocator.
            unsafe { crate::pool_alloc::zyntax_free(ptr) };
        }
    }
    /// Return storage a frame took for itself.
    pub fn release_scratch(&mut self, ptr: *mut u8) {
        self.free(ptr);
    }
    pub fn realloc(&mut self, old: *mut u8, size: usize) -> *mut u8 {
        if self.native_allocator {
            return unsafe { crate::pool_alloc::zyntax_realloc(old, size.max(1)) };
        }
        if old.is_null() {
            return self.alloc_zeroed(size.max(1));
        }
        let Some(index) = self.allocations.iter().position(|a| a.as_ptr() == old) else {
            return std::ptr::null_mut();
        };
        let mut replacement = vec![0u8; size.max(1)].into_boxed_slice();
        let n = replacement.len().min(self.allocations[index].len());
        replacement[..n].copy_from_slice(&self.allocations[index][..n]);
        let ptr = replacement.as_mut_ptr();
        self.allocations[index] = replacement;
        ptr
    }
    /// The slot of global `id`, allocated zeroed on first use.
    pub fn global_slot(&mut self, id: HirId, n_bytes: usize) -> *mut u8 {
        if let Some(ptr) = self.globals.get(&id) {
            return *ptr;
        }
        let ptr = self.alloc_zeroed(n_bytes.max(8));
        self.globals.insert(id, ptr);
        ptr
    }

    /// Bind a module global to storage owned by another execution tier.
    /// The caller keeps `ptr` alive while bytecode using this global runs.
    pub fn bind_global_slot(&mut self, id: HirId, ptr: *mut u8) {
        self.globals.insert(id, ptr);
    }
}

#[derive(Clone, Copy)]
pub struct SymbolEntry {
    pub ptr: *const u8,
    pub param_count: u8,
    /// Optional ZRTL signature describing arg / return TypeTags. When
    /// `Some`, `Op::CallSym` dispatches through a typed-marshalling
    /// path that respects the platform's float ABI (float args ride
    /// xmm/v registers, not the int register file). Required for any
    /// FFI symbol whose signature includes f32 / f64 — without it,
    /// `value_to_i64` truncates floats to integers and the callee
    /// reads garbage.
    pub sig: Option<crate::zrtl::ZrtlSymbolSig>,
}

unsafe impl Send for SymbolEntry {}
unsafe impl Sync for SymbolEntry {}

#[derive(Debug)]
pub enum InterpError {
    UndefinedSsaValue(HirId),
    UnknownFunction(String),
    UnsupportedInstruction(String),
    TypeMismatch { expected: String, got: String },
    DivisionByZero,
    OutOfMemory,
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

// ─────────────────────────────────────────────────────────────────────────────
// Interpreter (dispatch loop)
// ─────────────────────────────────────────────────────────────────────────────

pub struct HirInterpreter {
    symbols: HashMap<String, SymbolEntry>,
    pub profile: IdMap<ProfileSample>,
    memory: Memory,
    /// Per-HIR-function compiled bytecode cache. First call to a fn
    /// triggers compilation; subsequent calls reuse the same
    /// `CompiledFunction`. Keyed by `HirFunction::id`.
    cache: IdMap<CompiledFunction>,
    /// Functions this interpreter has already refused, and why.
    ///
    /// A refusal is as stable as a success: a function that performs an
    /// effect will never become interpretable. Without this, every call
    /// re-compiles the whole body just to fail the same way, and the
    /// host takes its fallback path afterwards regardless.
    uncompilable: IdMap<String>,
    /// Per-function tick callbacks. Invoked once per call entry; the
    /// callback returns the function's native entry once one exists,
    /// and the call goes there instead of into the bytecode.
    #[allow(clippy::type_complexity)]
    tick_callbacks: IdMap<Box<dyn FnMut() -> Option<*const u8> + Send>>,
    /// Where a function's body comes from when the runtime keeps one
    /// apart from the module's: see [`Self::set_body_source`].
    body_source: Option<Box<dyn FnMut(HirId) -> Option<std::sync::Arc<HirFunction>> + Send>>,
    /// Compiles, or finds, the thunk that calls native code of a given
    /// shape: `fn(target, words, out)`. Installed by a runtime with a
    /// native tier; without one, calls into native code use the fixed
    /// set of shapes `call_jit_dispatch` knows.
    #[allow(clippy::type_complexity)]
    thunk_source: Option<Box<dyn FnMut(&NativeSig) -> Option<*const u8> + Send>>,
    /// The current native entry of a function, as compiled callers
    /// reach it: what a `FuncRef` evaluates to, and where a call to a
    /// function the interpreter cannot run goes.
    #[allow(clippy::type_complexity)]
    entry_source: Option<Box<dyn Fn(HirId) -> Option<*const u8> + Send + Sync>>,
    /// The bead a function is promoted under, for a loop that asks.
    #[allow(clippy::type_complexity)]
    bead_source: Option<Box<dyn Fn(HirId) -> Option<u64> + Send + Sync>>,
    /// The module function behind a code address, for a call through a
    /// pointer: the call then goes by function id, and a function that
    /// has no code yet is interpreted rather than compiled on the spot.
    #[allow(clippy::type_complexity)]
    address_source: Option<Box<dyn Fn(usize) -> Option<HirId> + Send + Sync>>,
    /// The beads of the running frames that asked for resume points,
    /// innermost last; `run` reports each frame gone as it returns.
    waiting_marks: Vec<u64>,
    /// While the collector clears dead frames: the stack below `run`'s
    /// frame as it calls `run_frame`, and the most stack a `run_frame`
    /// frame has taken below that.
    run_top: usize,
    frame_extent: usize,
    /// Thunks already made, by shape.
    thunks: HashMap<NativeSig, usize>,
    /// The call shape of each function called so far, with the thunk
    /// that makes calls of that shape once one has been made.
    shapes: IdMap<(std::sync::Arc<NativeSig>, usize)>,
    /// Functions whose address the module takes, per module (by
    /// address; modules are shared and stay put).
    address_taken: HashMap<usize, std::collections::HashSet<HirId>>,
    /// Wasm-JIT compile hook (Phase E.6 — wasm32 only path).
    ///
    /// Called the first time a function crosses
    /// `wasm_jit_threshold` invocations. The host (zyntax_wasm
    /// crate) wires this to:
    ///   1. `WasmBackend::compile_function(func)` to produce wasm bytes,
    ///   2. ship the bytes to JS via a wasm-bindgen extern,
    ///   3. JS `WebAssembly.compile + instantiate`s + stashes the exported function in a funcref table,
    ///   4. returns the table index as the `u32` handle.
    ///
    /// Returning `None` keeps the function in BC forever (clean
    /// fallback for HIR shapes the wasm emitter can't lower yet).
    #[allow(clippy::type_complexity)]
    wasm_compile_hook: Option<Box<dyn FnMut(&HirFunction) -> Option<u32> + Send>>,
    /// Wasm-JIT dispatch hook. Once a function has a cached handle,
    /// every subsequent call routes through this hook instead of
    /// the BC dispatch loop. The host implementation dispatches via
    /// a wasm-bindgen extern that takes (handle, args).
    ///
    /// The args slice carries `ZyntaxValue`s in the same i64-/f64-
    /// funneled form as `call_extern_symbol` (the FFI ABI); the
    /// hook narrows / boxes them on the JS side per the function's
    /// signature.
    #[allow(clippy::type_complexity)]
    wasm_dispatch_hook:
        Option<Box<dyn FnMut(u32, &[ZyntaxValue]) -> Result<ZyntaxValue, InterpError> + Send>>,
    /// Cached `u32` handles returned by the wasm compile hook,
    /// keyed by HIR function id. Presence triggers dispatch via
    /// `wasm_dispatch_hook`.
    wasm_jit_handles: HashMap<HirId, u32>,
    /// Hot threshold for the wasm tier-up. Defaults to 1 so the
    /// demo path JITs on first reuse; tunable via
    /// [`Self::set_wasm_jit_threshold`].
    wasm_jit_threshold: u64,
    /// IndirectCall dispatcher. The HIR `IndirectCall { func_ptr,
    /// args, return_ty }` instruction reads `func_ptr` from a
    /// register; on wasm32 the value carries a 32-bit-truncated
    /// closure handle (Phase I.3's `HirId::to_handle_hash()` →
    /// `ZyntaxValue::Pointer((hash as usize) as *mut u8)`). The
    /// host runtime (`zyntax_wasm`) installs a dispatcher here
    /// that resolves the handle through `ACTIVE_CLOSURE_FNS` and
    /// re-enters `call_function`. Native callers can install a
    /// transmute-based dispatcher instead; without one, IndirectCall
    /// returns `InterpError::UnsupportedInstruction("indirect call")`.
    #[allow(clippy::type_complexity)]
    indirect_call_dispatcher:
        Option<Box<dyn FnMut(i64, Vec<ZyntaxValue>) -> Result<ZyntaxValue, InterpError> + Send>>,

    /// Optional escape hatch for symbol-callable dispatch. On wasm32,
    /// `call_extern_symbol`'s transmute-to-`extern "C" fn(i64,...)` only
    /// works when the wasm function table holds an entry whose signature
    /// matches exactly — Rust fns with mixed `*const u8` + `i64` params
    /// (e.g. `__zyntax_register_future`) don't, so the indirect call
    /// traps with "function signature mismatch". When this dispatcher
    /// is set, `Op::CallSym` consults it FIRST: a `Some(value)` return
    /// short-circuits the transmute path. The dispatcher receives the
    /// symbol name + the resolved arg values; it returns `Ok(Some(v))`
    /// to hand back a value, `Ok(None)` to fall through to the native
    /// transmute path, or `Err` to propagate.
    #[allow(clippy::type_complexity)]
    symbol_call_dispatcher: Option<
        Box<dyn FnMut(&str, Vec<ZyntaxValue>) -> Result<Option<ZyntaxValue>, InterpError> + Send>,
    >,
}

/// When a `tick_callback` returns one of these, the interpreter
/// dispatches the JIT'd code instead of running its bytecode.
///
/// `n_params` is the user-visible parameter count for ABI-safe
/// dispatch (max 8). `float_mask` encodes which parameters are
/// `f64` so the dispatcher can pick the correct `extern "C"`
/// transmute and place arguments in the float register class —
/// AArch64 routes f64 args through d0..d7, separate from the x0..x7
/// integer registers, and an all-i64 transmute would silently leave
/// f64 args in the wrong register class.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum JitRet {
    /// Integer/pointer/bool return — read from the integer register.
    Int,
    /// `f32` return — read from the float register as a 32-bit float.
    F32,
    /// `f64` return — read from the float register as a 64-bit float.
    F64,
}

#[derive(Clone, Copy)]
pub struct JitDispatch {
    pub ptr: *const u8,
    pub n_params: u8,
    /// Bit `i` set ⇒ argument `i` is `f64` (otherwise treated as i64).
    /// Only the lowest 8 bits are honoured; functions with more than
    /// 8 user-visible parameters fall back to the all-i64 dispatcher.
    pub float_mask: u8,
    /// Return register class. Float returns land in the float register
    /// (xmm0 / d0), not the integer register, so the dispatcher must
    /// transmute to a float-returning signature to read them — an
    /// `-> i64` transmute would read the (unrelated) integer register.
    pub ret: JitRet,
}

unsafe impl Send for JitDispatch {}

impl Default for HirInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl HirInterpreter {
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            profile: IdMap::default(),
            memory: Memory::new(),
            cache: IdMap::default(),
            uncompilable: IdMap::default(),
            tick_callbacks: IdMap::default(),
            body_source: None,
            thunk_source: None,
            entry_source: None,
            bead_source: None,
            address_source: None,
            waiting_marks: Vec::new(),
            run_top: 0,
            frame_extent: 0,
            thunks: HashMap::new(),
            shapes: IdMap::default(),
            address_taken: HashMap::new(),
            wasm_compile_hook: None,
            wasm_dispatch_hook: None,
            wasm_jit_handles: HashMap::new(),
            wasm_jit_threshold: 1,
            indirect_call_dispatcher: None,
            symbol_call_dispatcher: None,
        }
    }

    /// Share allocations with Cranelift/LLVM when execution can tier up.
    pub fn use_native_allocator(&mut self) {
        self.memory.use_native_allocator();
    }

    /// Use the native tier's storage for a module global. Existing bytecode
    /// constants are rebound as well, so a later native module rebuild can
    /// replace the backing address without keeping stale pointers in code.
    pub fn bind_global_slot(&mut self, id: HirId, ptr: *mut u8) {
        self.memory.bind_global_slot(id, ptr);
        for function in self.cache.values_mut() {
            for &(global_id, idx) in &function.global_consts {
                if global_id == id {
                    function.const_pool[idx as usize] = ZyntaxValue::Pointer(ptr);
                }
            }
        }
    }

    /// Install the IndirectCall dispatcher. See the field doc on
    /// `indirect_call_dispatcher` for the contract.
    pub fn set_indirect_call_dispatcher(
        &mut self,
        dispatcher: Box<
            dyn FnMut(i64, Vec<ZyntaxValue>) -> Result<ZyntaxValue, InterpError> + Send,
        >,
    ) {
        self.indirect_call_dispatcher = Some(dispatcher);
    }

    /// Install a symbol-call escape hatch (wasm32 register_future et
    /// al). See the field doc on `symbol_call_dispatcher`.
    pub fn set_symbol_call_dispatcher(
        &mut self,
        dispatcher: Box<
            dyn FnMut(&str, Vec<ZyntaxValue>) -> Result<Option<ZyntaxValue>, InterpError> + Send,
        >,
    ) {
        self.symbol_call_dispatcher = Some(dispatcher);
    }

    pub fn register_symbol(&mut self, name: impl Into<String>, ptr: *const u8, param_count: u8) {
        self.symbols.insert(
            name.into(),
            SymbolEntry {
                ptr,
                param_count,
                sig: None,
            },
        );
    }

    /// Typed registration variant — stores a ZRTL signature alongside
    /// the function pointer so the BC interp's `Op::CallSym` can
    /// route float arguments through the platform float ABI instead
    /// of bit-truncating them.
    pub fn register_symbol_typed(
        &mut self,
        name: impl Into<String>,
        ptr: *const u8,
        sig: crate::zrtl::ZrtlSymbolSig,
    ) {
        let param_count = sig.param_count;
        self.symbols.insert(
            name.into(),
            SymbolEntry {
                ptr,
                param_count,
                sig: Some(sig),
            },
        );
    }

    /// Snapshot of every registered FFI symbol — name → `(ptr,
    /// param_count)`. Returns owned strings + raw pointers; the
    /// pointers stay valid as long as the registering plugin's
    /// statics outlive the snapshot (always true for the
    /// `register_static_plugin` path on wasm32).
    ///
    /// Used by the wasm-JIT host to mirror the table into a
    /// thread-local store that `_zyntax_call_extern_*` exports
    /// dispatch through.
    pub fn symbol_table_snapshot(&self) -> Vec<(String, *const u8, u8)> {
        self.symbols
            .iter()
            .map(|(name, entry)| (name.clone(), entry.ptr, entry.param_count))
            .collect()
    }

    /// Register a per-function tick callback. Invoked on every entry to
    /// the function; returning `Some(entry)` sends the call to that
    /// native code. The host-side beadie wrapper plugs `on_invoke` in
    /// here.
    pub fn register_tick_callback(
        &mut self,
        func_id: HirId,
        cb: Box<dyn FnMut() -> Option<*const u8> + Send>,
    ) {
        self.tick_callbacks.insert(func_id, cb);
    }

    /// The body to run for a function, asked once at its first run,
    /// before it is compiled to bytecode: the body the tiers above
    /// compile, so a frame can move to them. `None` means the module's.
    pub fn set_body_source(
        &mut self,
        source: Box<dyn FnMut(HirId) -> Option<std::sync::Arc<HirFunction>> + Send>,
    ) {
        self.body_source = Some(source);
    }

    /// Drop the bridge and every tick callback. They hold the native
    /// tiers' backends, which the runtime shuts down after them.
    pub fn clear_native_bridge(&mut self) {
        self.tick_callbacks = IdMap::default();
        self.thunk_source = None;
        self.entry_source = None;
        self.bead_source = None;
    }

    /// Install the bridge to a native tier: `thunk` compiles the caller
    /// for a call shape, `entry` reads a function's current native entry.
    #[allow(clippy::type_complexity)]
    pub fn set_native_bridge(
        &mut self,
        thunk: Box<dyn FnMut(&NativeSig) -> Option<*const u8> + Send>,
        entry: Box<dyn Fn(HirId) -> Option<*const u8> + Send + Sync>,
        bead: Box<dyn Fn(HirId) -> Option<u64> + Send + Sync>,
    ) {
        self.thunk_source = Some(thunk);
        self.entry_source = Some(entry);
        self.bead_source = Some(bead);
    }

    /// How a code address is told to be one of the module's functions;
    /// see `address_source`.
    pub fn set_address_source(
        &mut self,
        source: Box<dyn Fn(usize) -> Option<HirId> + Send + Sync>,
    ) {
        self.address_source = Some(source);
    }

    /// Leave an interpreted frame at a loop header for `helper`: the
    /// live-ins go into a frame laid out as the helper expects, and its
    /// result is the frame's result. A function returning through a
    /// destination hands `dest` over in the frame, and gets it back as
    /// the result, the way its own return would. Once the frame holds
    /// the live-ins the registers stop being a root: a register that
    /// died before the loop would otherwise hold its value for the rest
    /// of the run.
    fn transfer(
        &mut self,
        helper: *const u8,
        site: &OsrSite,
        regs: &[ZyntaxValue],
        roots: &mut FrameRoots,
        scratch: &mut Scratch,
        dest: *mut u8,
    ) -> Result<ZyntaxValue, InterpError> {
        let frame = self.frame_alloc(scratch, site.frame.size.max(8) as usize);
        if frame.is_null() {
            return Err(InterpError::OutOfMemory);
        }
        if let Some(offset) = site.destination {
            // SAFETY: the slot lies within the frame.
            unsafe {
                *(frame.add(offset as usize) as *mut *mut u8) = dest;
            }
        }
        for ((reg, ty), offset) in site
            .live_ins
            .iter()
            .zip(&site.types)
            .zip(&site.frame.offsets)
        {
            let value = &regs[*reg as usize];
            // SAFETY: the frame has `site.frame.size` bytes and the
            // offsets lie within it.
            unsafe {
                let at = frame.add(*offset as usize);
                if crate::osr::is_held_by_reference(ty) {
                    *(at as *mut i64) = value_to_i64(value).unwrap_or(0);
                } else {
                    write_typed(at, value, ty);
                }
            }
        }
        roots.release();
        let sig = NativeSig::of_site(vec![HirType::Ptr(Box::new(HirType::U8))], site.ret.clone());
        if trace_enabled() {
            eprintln!("[interp] transfer site=0x{:x} -> {helper:?}", site.site_key);
        } else if crate::osr::osr_trace_enabled() {
            eprintln!(
                "[osr] interpreted frame leaves at site=0x{:x} -> {helper:?}",
                site.site_key
            );
        }
        self.call_native(
            helper,
            &sig,
            &[ZyntaxValue::Pointer(frame)],
            core::ptr::null_mut(),
        )
    }

    /// Box the arguments an extern declares as dynamic, as a compiled call
    /// site does: a pointer-shaped value is the box's payload itself, a
    /// scalar is copied into a slot the box points at. Returns the boxes
    /// made, to free after the call; a value already boxed goes through.
    fn box_dynamic_args(
        &mut self,
        sig: &crate::zrtl::ZrtlSymbolSig,
        arg_types: &[HirType],
        args: &mut [ZyntaxValue],
    ) -> Vec<*mut u8> {
        let mut made = Vec::new();
        for (i, arg) in args.iter_mut().enumerate() {
            if !sig.param_is_dynamic(i) {
                continue;
            }
            let ty = match arg_types.get(i) {
                Some(t) => t,
                None => continue,
            };
            if crate::zrtl::is_dynamic_box_pointer(ty) {
                continue;
            }
            let (tag, size) = crate::zrtl::dynamic_box_tag_and_size_for_hir_type(ty);
            // Box, then the payload slot behind it for a scalar.
            let block = self.memory.alloc_zeroed(40);
            if block.is_null() {
                continue;
            }
            let payload = if crate::zrtl::dynamic_box_uses_direct_pointer(ty) {
                value_to_i64(arg).unwrap_or(0)
            } else {
                // SAFETY: the block has 40 bytes; the slot is at 32.
                unsafe {
                    let slot = block.add(32);
                    let bits = match &*arg {
                        ZyntaxValue::Float(f) => f.to_bits() as i64,
                        ZyntaxValue::F32(f) => f.to_bits() as i64,
                        other => value_to_i64(other).unwrap_or(0),
                    };
                    *(slot as *mut i64) = bits;
                    slot as i64
                }
            };
            let display = match ty {
                HirType::Opaque(name) => Some(name),
                HirType::Ptr(inner) => match inner.as_ref() {
                    HirType::Opaque(name) => Some(name),
                    _ => None,
                },
                _ => None,
            }
            .and_then(|name| {
                let name = name.resolve_global().unwrap_or_default();
                let name = name.strip_prefix('$').unwrap_or(&name).to_string();
                (!name.is_empty())
                    .then(|| {
                        self.symbols
                            .get(&format!("${name}$to_string"))
                            .map(|e| e.ptr as i64)
                    })
                    .flatten()
            })
            .unwrap_or(0);
            // SAFETY: the block has 40 bytes.
            unsafe {
                *(block as *mut u32) = tag;
                *(block.add(4) as *mut u32) = size;
                *(block.add(8) as *mut i64) = payload;
                *(block.add(16) as *mut i64) = 0;
                *(block.add(24) as *mut i64) = display;
            }
            *arg = ZyntaxValue::Pointer(block);
            made.push(block);
        }
        made
    }

    /// The thunk for the call site `sig` of `cf`, made on first use and
    /// kept with the site; 0 when there is no native tier.
    fn site_thunk(&mut self, cf: &CompiledFunction, sig: u32) -> Result<usize, InterpError> {
        if self.thunk_source.is_none() {
            return Ok(0);
        }
        let slot = &cf.sig_thunks[sig as usize];
        if slot.get() != 0 {
            return Ok(slot.get());
        }
        let made = self.thunk_for(&cf.sig_pool[sig as usize])? as usize;
        slot.set(made);
        Ok(made)
    }

    /// The thunk for calls of shape `sig`, made on first use.
    fn thunk_for(&mut self, sig: &NativeSig) -> Result<*const u8, InterpError> {
        if let Some(&t) = self.thunks.get(sig) {
            return Ok(t as *const u8);
        }
        let source = self.thunk_source.as_mut().ok_or_else(|| {
            InterpError::UnsupportedInstruction(
                "a call into native code with no native tier to make the call".to_string(),
            )
        })?;
        let t = source(sig).ok_or_else(|| {
            InterpError::UnsupportedInstruction(format!(
                "a call into native code of a shape the native tier cannot make: {sig:?}"
            ))
        })?;
        self.thunks.insert(sig.clone(), t as usize);
        Ok(t)
    }

    /// The native entry of `func`, if a native tier holds one.
    fn native_entry(&self, func: HirId) -> Option<*const u8> {
        self.entry_source
            .as_ref()
            .and_then(|f| f(func))
            .filter(|p| !p.is_null())
    }

    /// Whether `module` takes the address of `func`, which fixes the
    /// convention its callers use.
    fn is_address_taken(&mut self, module: &HirModule, func: HirId) -> bool {
        let key = module as *const HirModule as usize;
        let set = self
            .address_taken
            .entry(key)
            .or_insert_with(|| crate::dce::address_taken_functions(module));
        set.contains(&func)
    }

    /// The shape of a direct call to `func`, and its thunk once made.
    fn shape_of(
        &mut self,
        module: &HirModule,
        func: HirId,
    ) -> Option<(std::sync::Arc<NativeSig>, usize)> {
        if let Some(entry) = self.shapes.get(&func) {
            return Some(entry.clone());
        }
        let taken = self.is_address_taken(module, func);
        let function = module.functions.get(&func)?;
        let sig = std::sync::Arc::new(NativeSig::of_function(function, taken));
        self.shapes.insert(func, (std::sync::Arc::clone(&sig), 0));
        Some((sig, 0))
    }

    /// Call `func`'s native code at `entry`, through the thunk cached for
    /// its shape.
    fn call_function_natively(
        &mut self,
        module: &HirModule,
        func: HirId,
        entry: *const u8,
        args: &[ZyntaxValue],
        dest: *mut u8,
    ) -> Result<ZyntaxValue, InterpError> {
        let (sig, mut thunk) = self
            .shape_of(module, func)
            .ok_or(InterpError::UndefinedSsaValue(func))?;
        if thunk == 0 && self.thunk_source.is_some() {
            thunk = self.thunk_for(&sig)? as usize;
            if let Some(cached) = self.shapes.get_mut(&func) {
                cached.1 = thunk;
            }
        }
        if crate::collector::is_enabled() {
            clear_stack_below();
        }
        self.call_native_through(entry, &sig, thunk, args, dest)
    }

    /// Call native code at `entry` with the shape `sig`. Aggregates go as
    /// the addresses they are held at; `dest` is the storage a
    /// destination-returning callee fills.
    fn call_native(
        &mut self,
        entry: *const u8,
        sig: &NativeSig,
        args: &[ZyntaxValue],
        dest: *mut u8,
    ) -> Result<ZyntaxValue, InterpError> {
        let thunk = if self.thunk_source.is_some() {
            self.thunk_for(sig)? as usize
        } else {
            0
        };
        self.call_native_through(entry, sig, thunk, args, dest)
    }

    /// [`Self::call_native`] with the shape's thunk already in hand; 0
    /// when there is no native tier to make one.
    fn call_native_through(
        &mut self,
        entry: *const u8,
        sig: &NativeSig,
        thunk: usize,
        args: &[ZyntaxValue],
        dest: *mut u8,
    ) -> Result<ZyntaxValue, InterpError> {
        crate::collector::note_frame_depth();
        if thunk == 0 {
            // No native tier to make a thunk: the fixed set of shapes.
            if !jit_dispatch_supported(&sig.params, &sig.ret) {
                return Err(InterpError::UnsupportedInstruction(format!(
                    "a call into native code of a shape the interpreter cannot make: {sig:?}"
                )));
            }
            let dispatch = JitDispatch {
                ptr: entry,
                n_params: sig.params.len().min(255) as u8,
                float_mask: jit_float_mask(&sig.params),
                ret: match sig.ret {
                    HirType::F32 => JitRet::F32,
                    HirType::F64 => JitRet::F64,
                    _ => JitRet::Int,
                },
            };
            let raw = call_jit_dispatch(dispatch, args);
            return Ok(match raw {
                ZyntaxValue::Int(i) => value_from_i64_as(&sig.ret, i),
                other => other,
            });
        }
        if args.len() != sig.params.len() {
            return Err(InterpError::Host(format!(
                "a native call with {} arguments for {} parameters",
                args.len(),
                sig.params.len()
            )));
        }
        // The words stay on the stack for the shapes that fit.
        let mut small = [0u64; 16];
        let mut large: Vec<u64> = Vec::new();
        let words: &[u64] = if args.len() <= small.len() {
            for (w, a) in small.iter_mut().zip(args) {
                *w = word_of(a);
            }
            &small[..args.len()]
        } else {
            large.extend(args.iter().map(word_of));
            &large
        };
        let mut out = [0u8; 16];
        let out_ptr = if sig.destination {
            if dest.is_null() {
                return Err(InterpError::Host(
                    "a destination-returning native call with no destination".to_string(),
                ));
            }
            dest
        } else {
            out.as_mut_ptr()
        };
        // SAFETY: the thunk was compiled for exactly this shape, `words`
        // holds one word per parameter, and `out_ptr` has room for the
        // result the shape describes.
        unsafe {
            let f: extern "C" fn(*const u8, *const u64, *mut u8) =
                core::mem::transmute(thunk as *const u8);
            f(entry, words.as_ptr(), out_ptr);
        }
        if trace_enabled() {
            eprintln!("[interp] native {entry:?} {sig:?} {args:?}");
        }
        if sig.destination {
            return Ok(ZyntaxValue::Pointer(dest));
        }
        let word = u64::from_ne_bytes(out[..8].try_into().unwrap());
        Ok(value_from_word(&sig.ret, word))
    }

    /// Install the wasm-JIT compile hook. See `Self::wasm_compile_hook`
    /// docs. Called at most once per function — the returned handle is
    /// cached for all subsequent calls.
    pub fn set_wasm_compile_hook(
        &mut self,
        hook: Box<dyn FnMut(&HirFunction) -> Option<u32> + Send>,
    ) {
        self.wasm_compile_hook = Some(hook);
    }

    /// Install the wasm-JIT dispatch hook. Called every time a function
    /// with a cached handle is invoked. See `Self::wasm_dispatch_hook`
    /// docs.
    pub fn set_wasm_dispatch_hook(
        &mut self,
        hook: Box<dyn FnMut(u32, &[ZyntaxValue]) -> Result<ZyntaxValue, InterpError> + Send>,
    ) {
        self.wasm_dispatch_hook = Some(hook);
    }

    /// Tune the wasm-JIT hot threshold (default 1 — JIT on first
    /// reuse). 0 means JIT eagerly on first call; large values keep
    /// functions in BC longer.
    pub fn set_wasm_jit_threshold(&mut self, n: u64) {
        self.wasm_jit_threshold = n;
    }

    /// Whether `func_id` has a cached wasm-JIT handle. Diagnostic /
    /// test hook; production callers don't need to know.
    pub fn has_wasm_jit_handle(&self, func_id: HirId) -> bool {
        self.wasm_jit_handles.contains_key(&func_id)
    }

    /// Profile snapshot for a function.
    pub fn profile_for(&self, func_id: HirId) -> ProfileSample {
        self.profile.get(&func_id).copied().unwrap_or_default()
    }

    pub fn call(
        &mut self,
        module: &HirModule,
        name: &str,
        args: Vec<ZyntaxValue>,
    ) -> Result<ZyntaxValue, InterpError> {
        let func = module
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some(name))
            .ok_or_else(|| InterpError::UnknownFunction(name.to_string()))?;
        let func_id = func.id;
        // A struct handed back through a destination needs storage that
        // outlives the callee's frame.
        let dest = match crate::abi::destination_return_type(func) {
            Some(ty) if !self.is_address_taken(module, func_id) => {
                self.memory.alloc_zeroed(size_of_hir_ty(ty).max(8))
            }
            _ => core::ptr::null_mut(),
        };
        self.call_by_id(module, func_id, args, dest)
    }

    /// Call `func_id` from the host or from bytecode. `dest` is where a
    /// callee that returns a struct through a destination writes it;
    /// null when it returns nothing that way.
    fn call_by_id(
        &mut self,
        module: &HirModule,
        func_id: HirId,
        args: Vec<ZyntaxValue>,
        dest: *mut u8,
    ) -> Result<ZyntaxValue, InterpError> {
        if let Some(func) = module.functions.get(&func_id) {
            if func.is_external {
                let name = func
                    .link_name
                    .clone()
                    .or_else(|| func.name.resolve_global())
                    .ok_or_else(|| InterpError::UnknownFunction(format!("{:?}", func_id)))?;
                let entry = match self.symbols.get(&name).copied() {
                    Some(e) => e,
                    None => {
                        // A symbol nothing registered is the process's own,
                        // as the native tiers resolve one: libc's `exit`.
                        let ptr = process_symbol(&name)
                            .ok_or_else(|| InterpError::UnknownFunction(name.clone()))?;
                        let e = SymbolEntry {
                            ptr,
                            param_count: func.signature.params.len().min(255) as u8,
                            sig: None,
                        };
                        self.symbols.insert(name.clone(), e);
                        e
                    }
                };
                let result = if self.thunk_source.is_some() {
                    let sig = NativeSig::of_function(func, true);
                    self.call_native(entry.ptr, &sig, &args, dest)
                } else {
                    match entry.sig {
                        Some(sig) => call_extern_symbol_typed(entry.ptr, &args, &sig),
                        None => {
                            let raw = call_extern_symbol(entry.ptr, &args);
                            let ret_ty = func.signature.returns.first().unwrap_or(&HirType::Void);
                            Ok(value_from_i64_as(ret_ty, raw))
                        }
                    }
                };
                if trace_enabled() {
                    eprintln!("[interp] extern {name}({args:?}) -> {result:?}");
                }
                return result;
            }
        }
        if trace_enabled() {
            let name = module
                .functions
                .get(&func_id)
                .and_then(|f| f.name.resolve_global())
                .unwrap_or_default();
            eprintln!("[interp] call {name}({args:?})");
        }

        // Profile.
        let call_count = {
            let p = self.profile.entry(func_id).or_default();
            p.call_count += 1;
            p.call_count
        };

        // Wasm-JIT fast path. If we already have a handle, route
        // straight to the host's dispatch hook — same role the
        // native tick_callback plays for Cranelift / LLVM ptrs, but
        // through an opaque u32 handle (a JS funcref table index)
        // because wasm32 doesn't have addressable function ptrs.
        if let Some(handle) = self.wasm_jit_handles.get(&func_id).copied() {
            if let Some(cb) = self.wasm_dispatch_hook.as_mut() {
                return cb(handle, &args);
            }
            // Dispatch hook was uninstalled after compile fired —
            // stale handle, drop it and fall back to BC.
            self.wasm_jit_handles.remove(&func_id);
        }

        // A native entry, once the tiers above have made one, takes the
        // call instead of the bytecode.
        let first = !self.cache.contains_key(&func_id);
        let phases = first && std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some();
        let native = match self.tick_callbacks.get_mut(&func_id) {
            Some(cb) => cb(),
            None => None,
        };
        if let Some(entry) = native {
            return self.call_function_natively(module, func_id, entry, &args, dest);
        }

        // Wasm-JIT hot detection. Once `call_count` crosses the
        // threshold, hand the HirFunction to the host compile hook.
        // Hook returns `Some(handle)` on success, `None` on failure
        // (unsupported HIR shape) — in either case we still run the
        // current call through BC; the JIT entry kicks in on the
        // NEXT invocation.
        if call_count >= self.wasm_jit_threshold
            && !self.wasm_jit_handles.contains_key(&func_id)
            && self.wasm_compile_hook.is_some()
        {
            if let Some(func) = module.functions.get(&func_id) {
                if let Some(hook) = self.wasm_compile_hook.as_mut() {
                    if let Some(handle) = hook(func) {
                        self.wasm_jit_handles.insert(func_id, handle);
                    }
                }
            }
        }

        // Compile-on-first-use, and refuse-once. What the interpreter
        // cannot run, native code runs, when there is native code.
        if !self.cache.contains_key(&func_id) {
            if let Some(why) = self.uncompilable.get(&func_id).cloned() {
                return self.run_natively_or(module, func_id, args, dest, why);
            }
            // The body the tiers above compile, when the runtime keeps
            // one apart from the module's; else the module's. A callee
            // the module does not hold is a function, not a value:
            // reported as one.
            // The body's making is the first call's cost: reported when
            // it shows.
            let started = web_time::Instant::now();
            let shared = self.body_source.as_mut().and_then(|source| source(func_id));
            let body_ms = started.elapsed().as_secs_f64() * 1000.0;
            let func: &HirFunction = match &shared {
                Some(f) => f,
                None => module
                    .functions
                    .get(&func_id)
                    .ok_or_else(|| InterpError::UnknownFunction(format!("{func_id:?}")))?,
            };
            if phases && body_ms >= 1.0 {
                eprintln!(
                    "[INTERP] body of {} {body_ms:8.2} ms",
                    func.name.resolve_global().unwrap_or_default()
                );
            }
            let taken = self.is_address_taken(module, func_id);
            match compile_function_with(module, &mut self.memory, func, taken) {
                Ok(cf) => {
                    self.cache.insert(func_id, cf);
                }
                Err(InterpError::UnsupportedInstruction(why)) => {
                    if trace_enabled() {
                        let name = module
                            .functions
                            .get(&func_id)
                            .and_then(|f| f.name.resolve_global())
                            .unwrap_or_default();
                        eprintln!("[interp] {name} runs natively: {why}");
                    }
                    self.uncompilable.insert(func_id, why.clone());
                    return self.run_natively_or(module, func_id, args, dest, why);
                }
                Err(e) => return Err(e),
            }
        }

        // Lift the compiled function out of the cache so we can mutate
        // `self.memory` / `self.symbols` / `self.cache` during nested
        // calls. The map ownership returns at the end.
        let cf = self.cache.remove(&func_id).unwrap();
        let result = self.run(module, &cf, args, func_id, dest);
        // Put the (immutable) compiled function back.
        self.cache.insert(func_id, cf);
        result
    }

    /// Run `func_id` natively because the interpreter refused it for
    /// `why`, or report the refusal when nothing native exists.
    fn run_natively_or(
        &mut self,
        module: &HirModule,
        func_id: HirId,
        args: Vec<ZyntaxValue>,
        dest: *mut u8,
        why: String,
    ) -> Result<ZyntaxValue, InterpError> {
        match self.native_entry(func_id) {
            Some(entry) => self.call_function_natively(module, func_id, entry, &args, dest),
            None => Err(InterpError::UnsupportedInstruction(why)),
        }
    }

    /// Run one frame of `cf`. Storage the frame takes for itself (its
    /// allocas, aggregate copies, undef aggregates) is returned when it
    /// finishes, as a stack frame's would be.
    fn run(
        &mut self,
        module: &HirModule,
        cf: &CompiledFunction,
        args: Vec<ZyntaxValue>,
        func_id: HirId,
        dest: *mut u8,
    ) -> Result<ZyntaxValue, InterpError> {
        let mut scratch = Scratch::new();
        let marks = self.waiting_marks.len();
        // The frame starts on cleared stack, so a slot it never writes
        // holds no word an earlier frame left there.
        if crate::collector::clears_dead_frames() {
            crate::collector::clear_stack_below(self.frame_extent);
            self.run_top = crate::collector::stack_mark();
        }
        let result = self.run_frame(module, cf, args, func_id, dest, &mut scratch);
        if let Err(e) = &result
            && trace_enabled()
        {
            let name = module
                .functions
                .get(&func_id)
                .and_then(|f| f.name.resolve_global())
                .unwrap_or_default();
            eprintln!("[interp] {name} failed: {e}");
        }
        // The frame is gone, whether it returned or left through a
        // resume point: no resume point is owed to it any more.
        while self.waiting_marks.len() > marks {
            if let Some(bead) = self.waiting_marks.pop() {
                crate::osr::frame_left(bead);
            }
        }
        for block in scratch.blocks.drain(..) {
            self.memory.release_scratch(block);
        }
        result
    }

    /// Storage for this frame: zeroed, released with the frame.
    fn frame_alloc(&mut self, scratch: &mut Scratch, size: usize) -> *mut u8 {
        let p = self.memory.alloc_zeroed(size.max(1));
        if !p.is_null() {
            scratch.push(p);
        }
        p
    }

    fn run_frame(
        &mut self,
        module: &HirModule,
        cf: &CompiledFunction,
        args: Vec<ZyntaxValue>,
        func_id: HirId,
        dest: *mut u8,
        scratch: &mut Scratch,
    ) -> Result<ZyntaxValue, InterpError> {
        if crate::collector::clears_dead_frames() {
            crate::collector::note_frame_depth();
            // A frame on another stack (a fiber's) measures nothing.
            let taken = self.run_top.wrapping_sub(crate::collector::stack_mark());
            if taken < 1 << 20 {
                self.frame_extent = self.frame_extent.max(taken);
            }
        }
        let mut regs: Vec<ZyntaxValue> = vec![ZyntaxValue::Undef; cf.n_regs as usize];
        // The registers hold pointers the collector cannot see on any
        // stack; they are a root for as long as the frame runs here.
        let mut roots = FrameRoots::new(&regs);

        // Bind params into regs[0..n_params].
        for (i, a) in args.into_iter().enumerate() {
            if i < cf.n_params as usize {
                regs[i] = a;
            }
        }
        // Loop headers: visits so far, and the helper slot each reads.
        let bead = if cf.osr_sites.is_empty() {
            None
        } else {
            self.bead_source.as_ref().and_then(|f| f(func_id))
        };
        let mut visits: Vec<u32> = vec![0; cf.osr_sites.len()];
        let mut slots: Vec<*const std::sync::atomic::AtomicU64> =
            vec![core::ptr::null(); cf.osr_sites.len()];
        // Whether this frame has asked for resume points; it is counted
        // as waiting until `run` sees it leave.
        let mut waits = false;
        // Aggregates the function owns from the start.
        for (reg, bytes) in &cf.entry_storage {
            let p = self.frame_alloc(scratch, bytes.len());
            if p.is_null() {
                return Err(InterpError::OutOfMemory);
            }
            // SAFETY: `p` has `bytes.len()` writable bytes.
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
            regs[*reg as usize] = ZyntaxValue::Pointer(p);
        }

        // Pre-resolve `cf.symbol_pool` (Vec<String>) into a parallel
        // `Vec<Option<SymbolEntry>>` once at the top of dispatch. The
        // Op::CallSym hot path then indexes this Vec directly instead
        // of cloning the name, hashing it, and probing
        // `self.symbols: HashMap<String, _>` per call. Unresolved
        // entries (rare — registration race or wasm dispatcher path)
        // fall back to the HashMap at the dispatch site.
        let resolved_symbols: Vec<Option<SymbolEntry>> = cf
            .symbol_pool
            .iter()
            .map(|name| self.symbols.get(name).copied())
            .collect();

        let mut pc: usize = 0;
        let code = &cf.code;

        while pc < code.len() {
            if let Some(p) = self.profile.get_mut(&func_id) {
                p.instructions_executed += 1;
            }
            match &code[pc] {
                Op::LoadConst { dst, c } => {
                    regs[*dst as usize] = cf.const_pool[*c as usize].clone();
                    pc += 1;
                }
                Op::Move { dst, src } => {
                    regs[*dst as usize] = regs[*src as usize].clone();
                    pc += 1;
                }
                Op::Select { dst, cond, yes, no } => {
                    let src = if value_to_i64(&regs[*cond as usize]).unwrap_or(0) != 0 {
                        *yes
                    } else {
                        *no
                    };
                    regs[*dst as usize] = regs[src as usize].clone();
                    pc += 1;
                }
                Op::IAdd { dst, lhs, rhs } => {
                    let v = ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| {
                        a.wrapping_add(b)
                    })?;
                    regs[*dst as usize] = v;
                    pc += 1;
                }
                Op::ISub { dst, lhs, rhs } => {
                    let v = ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| {
                        a.wrapping_sub(b)
                    })?;
                    regs[*dst as usize] = v;
                    pc += 1;
                }
                Op::IMul { dst, lhs, rhs } => {
                    let v = ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| {
                        a.wrapping_mul(b)
                    })?;
                    regs[*dst as usize] = v;
                    pc += 1;
                }
                Op::IDiv { dst, lhs, rhs } => {
                    let rv = ireg_i64(&regs[*rhs as usize])?;
                    if rv == 0 {
                        return Err(InterpError::DivisionByZero);
                    }
                    let v = ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| a / b)?;
                    regs[*dst as usize] = v;
                    pc += 1;
                }
                Op::IRem { dst, lhs, rhs } => {
                    let rv = ireg_i64(&regs[*rhs as usize])?;
                    if rv == 0 {
                        return Err(InterpError::DivisionByZero);
                    }
                    let v = ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| a % b)?;
                    regs[*dst as usize] = v;
                    pc += 1;
                }
                Op::IAnd { dst, lhs, rhs } => {
                    regs[*dst as usize] =
                        ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| a & b)?;
                    pc += 1;
                }
                Op::IOr { dst, lhs, rhs } => {
                    regs[*dst as usize] =
                        ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| a | b)?;
                    pc += 1;
                }
                Op::IXor { dst, lhs, rhs } => {
                    regs[*dst as usize] =
                        ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| a ^ b)?;
                    pc += 1;
                }
                Op::IShl { dst, lhs, rhs } => {
                    regs[*dst as usize] =
                        ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| {
                            a.wrapping_shl(b as u32)
                        })?;
                    pc += 1;
                }
                Op::IShr { dst, lhs, rhs } => {
                    regs[*dst as usize] =
                        ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| {
                            a.wrapping_shr(b as u32)
                        })?;
                    pc += 1;
                }
                Op::INeg { dst, src } => {
                    let v = ireg_i64(&regs[*src as usize])?;
                    let dst_ty = &cf.reg_types[*dst as usize];
                    regs[*dst as usize] = value_from_i64_as(dst_ty, v.wrapping_neg());
                    pc += 1;
                }
                Op::BNot { dst, src } => {
                    let dst_ty = &cf.reg_types[*dst as usize];
                    regs[*dst as usize] = match &regs[*src as usize] {
                        ZyntaxValue::Bool(b) => ZyntaxValue::Bool(!*b),
                        other => {
                            let v = value_to_i64(other).unwrap_or(0);
                            value_from_i64_as(dst_ty, !v)
                        }
                    };
                    pc += 1;
                }
                Op::FAdd { dst, lhs, rhs } => {
                    let x = freg_f64(&regs[*lhs as usize])?;
                    let y = freg_f64(&regs[*rhs as usize])?;
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], x + y);
                    pc += 1;
                }
                Op::FSub { dst, lhs, rhs } => {
                    let x = freg_f64(&regs[*lhs as usize])?;
                    let y = freg_f64(&regs[*rhs as usize])?;
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], x - y);
                    pc += 1;
                }
                Op::FMul { dst, lhs, rhs } => {
                    let x = freg_f64(&regs[*lhs as usize])?;
                    let y = freg_f64(&regs[*rhs as usize])?;
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], x * y);
                    pc += 1;
                }
                Op::FDiv { dst, lhs, rhs } => {
                    let x = freg_f64(&regs[*lhs as usize])?;
                    let y = freg_f64(&regs[*rhs as usize])?;
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], x / y);
                    pc += 1;
                }
                Op::FNeg { dst, src } => {
                    let x = freg_f64(&regs[*src as usize])?;
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], -x);
                    pc += 1;
                }
                Op::FSqrt { dst, src } => {
                    let x = freg_f64(&regs[*src as usize])?;
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], x.sqrt());
                    pc += 1;
                }
                Op::FRsqrt { dst, src } => {
                    let x = freg_f64(&regs[*src as usize])?;
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], 1.0 / x.sqrt());
                    pc += 1;
                }
                Op::FAbs { dst, src } => {
                    let x = freg_f64(&regs[*src as usize])?;
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], x.abs());
                    pc += 1;
                }
                Op::FFloor { dst, src } => {
                    let x = freg_f64(&regs[*src as usize])?;
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], x.floor());
                    pc += 1;
                }
                Op::FUnary { dst, src, op } => {
                    let x = freg_f64(&regs[*src as usize])?;
                    let y = match op {
                        FUnaryOp::Sin => x.sin(),
                        FUnaryOp::Cos => x.cos(),
                        FUnaryOp::Log => x.ln(),
                        FUnaryOp::Exp => x.exp(),
                    };
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], y);
                    pc += 1;
                }
                Op::FPow { dst, a, b } => {
                    let x = freg_f64(&regs[*a as usize])?;
                    let y = freg_f64(&regs[*b as usize])?;
                    regs[*dst as usize] = fval(&cf.reg_types[*dst as usize], x.powf(y));
                    pc += 1;
                }
                Op::FMulAdd { dst, a, b, c } => {
                    // Vector operands (Array of lanes) fuse element-wise;
                    // scalars fuse once. Lane width is preserved.
                    if let ZyntaxValue::Array(_) = &regs[*a as usize] {
                        let av = as_vector(&regs[*a as usize])?;
                        let bv = as_vector(&regs[*b as usize])?;
                        let cv = as_vector(&regs[*c as usize])?;
                        let n = av.len().min(bv.len()).min(cv.len());
                        let out: Vec<ZyntaxValue> = (0..n)
                            .map(|i| lane_mul_add(&av[i], &bv[i], &cv[i]))
                            .collect::<Result<_, _>>()?;
                        regs[*dst as usize] = ZyntaxValue::Array(out);
                    } else {
                        let av = freg_f64(&regs[*a as usize])?;
                        let bv = freg_f64(&regs[*b as usize])?;
                        let cv = freg_f64(&regs[*c as usize])?;
                        regs[*dst as usize] =
                            fval(&cf.reg_types[*dst as usize], av.mul_add(bv, cv));
                    }
                    pc += 1;
                }
                Op::ICmpEq { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        ireg_i64(&regs[*lhs as usize])? == ireg_i64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::ICmpNe { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        ireg_i64(&regs[*lhs as usize])? != ireg_i64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::ICmpLt { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        ireg_i64(&regs[*lhs as usize])? < ireg_i64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::UCmpLt { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        (ireg_i64(&regs[*lhs as usize])? as u64)
                            < (ireg_i64(&regs[*rhs as usize])? as u64),
                    );
                    pc += 1;
                }
                Op::UCmpLe { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        (ireg_i64(&regs[*lhs as usize])? as u64)
                            <= (ireg_i64(&regs[*rhs as usize])? as u64),
                    );
                    pc += 1;
                }
                Op::UCmpGt { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        (ireg_i64(&regs[*lhs as usize])? as u64)
                            > (ireg_i64(&regs[*rhs as usize])? as u64),
                    );
                    pc += 1;
                }
                Op::UCmpGe { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        (ireg_i64(&regs[*lhs as usize])? as u64)
                            >= (ireg_i64(&regs[*rhs as usize])? as u64),
                    );
                    pc += 1;
                }
                Op::UDiv { dst, lhs, rhs } => {
                    let rv = ireg_i64(&regs[*rhs as usize])?;
                    if rv == 0 {
                        return Err(InterpError::DivisionByZero);
                    }
                    let v = ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| {
                        ((a as u64) / (b as u64)) as i64
                    })?;
                    regs[*dst as usize] = v;
                    pc += 1;
                }
                Op::URem { dst, lhs, rhs } => {
                    let rv = ireg_i64(&regs[*rhs as usize])?;
                    if rv == 0 {
                        return Err(InterpError::DivisionByZero);
                    }
                    let v = ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| {
                        ((a as u64) % (b as u64)) as i64
                    })?;
                    regs[*dst as usize] = v;
                    pc += 1;
                }
                Op::UShr { dst, lhs, rhs } => {
                    regs[*dst as usize] =
                        ibin(&regs[*lhs as usize], &regs[*rhs as usize], |a, b| {
                            (a as u64).wrapping_shr(b as u32) as i64
                        })?;
                    pc += 1;
                }
                Op::ICmpLe { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        ireg_i64(&regs[*lhs as usize])? <= ireg_i64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::ICmpGt { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        ireg_i64(&regs[*lhs as usize])? > ireg_i64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::ICmpGe { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        ireg_i64(&regs[*lhs as usize])? >= ireg_i64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::FCmpEq { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        freg_f64(&regs[*lhs as usize])? == freg_f64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::FCmpNe { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        freg_f64(&regs[*lhs as usize])? != freg_f64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::FCmpLt { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        freg_f64(&regs[*lhs as usize])? < freg_f64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::FCmpLe { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        freg_f64(&regs[*lhs as usize])? <= freg_f64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::FCmpGt { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        freg_f64(&regs[*lhs as usize])? > freg_f64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::FCmpGe { dst, lhs, rhs } => {
                    regs[*dst as usize] = ZyntaxValue::Bool(
                        freg_f64(&regs[*lhs as usize])? >= freg_f64(&regs[*rhs as usize])?,
                    );
                    pc += 1;
                }
                Op::Cast { dst, src, op, ty } => {
                    let target = &cf.type_pool[*ty as usize];
                    regs[*dst as usize] = eval_cast(*op, regs[*src as usize].clone(), target)?;
                    pc += 1;
                }
                Op::Alloca { dst, size_bytes } => {
                    let ptr = self.frame_alloc(scratch, (*size_bytes).max(1) as usize);
                    if ptr.is_null() {
                        return Err(InterpError::OutOfMemory);
                    }
                    regs[*dst as usize] = ZyntaxValue::Pointer(ptr);
                    pc += 1;
                }
                Op::Malloc {
                    dst,
                    has_dst,
                    size_reg,
                } => {
                    let size = match &regs[*size_reg as usize] {
                        ZyntaxValue::Int(n) => (*n).max(1) as usize,
                        ZyntaxValue::UInt(n) => (*n).max(1) as usize,
                        other => {
                            return Err(InterpError::TypeMismatch {
                                expected: "integer (Malloc size)".to_string(),
                                got: format!("{:?}", other),
                            });
                        }
                    };
                    let ptr = self.memory.alloc_zeroed(size);
                    if *has_dst {
                        regs[*dst as usize] = ZyntaxValue::Pointer(ptr);
                    }
                    pc += 1;
                }
                Op::Realloc {
                    dst,
                    old_reg,
                    size_reg,
                } => {
                    let old = match &regs[*old_reg as usize] {
                        ZyntaxValue::Pointer(p) => *p,
                        ZyntaxValue::Int(0) | ZyntaxValue::UInt(0) => std::ptr::null_mut(),
                        other => {
                            return Err(InterpError::TypeMismatch {
                                expected: "pointer (Realloc old allocation)".to_string(),
                                got: format!("{:?}", other),
                            });
                        }
                    };
                    let size = value_to_i64(&regs[*size_reg as usize])
                        .ok_or_else(|| InterpError::TypeMismatch {
                            expected: "integer (Realloc size)".to_string(),
                            got: format!("{:?}", regs[*size_reg as usize]),
                        })?
                        .max(1) as usize;
                    let ptr = self.memory.realloc(old, size);
                    if ptr.is_null() {
                        return Err(InterpError::OutOfMemory);
                    }
                    regs[*dst as usize] = ZyntaxValue::Pointer(ptr);
                    pc += 1;
                }
                Op::Free { ptr } => {
                    if let Some(p) = value_to_i64(&regs[*ptr as usize]) {
                        self.memory.free(p as usize as *mut u8);
                    }
                    pc += 1;
                }
                Op::MemOp {
                    kind,
                    dst_ptr,
                    src,
                    len,
                } => {
                    let dst =
                        value_to_i64(&regs[*dst_ptr as usize]).unwrap_or(0) as usize as *mut u8;
                    let src_v = value_to_i64(&regs[*src as usize]).unwrap_or(0);
                    let n = value_to_i64(&regs[*len as usize]).unwrap_or(0).max(0) as usize;
                    if n > 0 && !dst.is_null() {
                        // SAFETY: the program computed these ranges, as
                        // the compiled tiers trust them.
                        unsafe {
                            match kind {
                                MemOpKind::Copy => core::ptr::copy_nonoverlapping(
                                    src_v as usize as *const u8,
                                    dst,
                                    n,
                                ),
                                MemOpKind::Move => {
                                    core::ptr::copy(src_v as usize as *const u8, dst, n)
                                }
                                MemOpKind::Set => core::ptr::write_bytes(dst, src_v as u8, n),
                            }
                        }
                    }
                    pc += 1;
                }
                Op::FuncAddr { dst, fn_id } => {
                    // With a native tier the address is the function's
                    // entry; without one it is the handle a host
                    // dispatcher resolves (the wasm shim's closure table).
                    let addr = match self.native_entry(*fn_id) {
                        Some(entry) => entry as usize,
                        None if self.entry_source.is_none() => fn_id.to_handle_hash() as usize,
                        None => {
                            return Err(InterpError::UnsupportedInstruction(
                                "taking the address of a function no tier holds".to_string(),
                            ));
                        }
                    };
                    regs[*dst as usize] = ZyntaxValue::Pointer(addr as *mut u8);
                    pc += 1;
                }
                Op::Load { dst, ptr, ty } => {
                    let target = &cf.type_pool[*ty as usize];
                    // Same tolerance as Store: accept Int/UInt as a
                    // raw address since `IAdd(Pointer, Int)` produces
                    // Int through the i64-funneled arithmetic path.
                    let p = match &regs[*ptr as usize] {
                        ZyntaxValue::Pointer(p) => *p,
                        ZyntaxValue::Int(n) => *n as usize as *mut u8,
                        ZyntaxValue::UInt(n) => *n as usize as *mut u8,
                        other => {
                            return Err(InterpError::TypeMismatch {
                                expected: "pointer".to_string(),
                                got: format!("{:?}", other),
                            });
                        }
                    };
                    if p.is_null() {
                        return Err(InterpError::Host(
                            "a load through a null pointer".to_string(),
                        ));
                    }
                    // `ZYNTAX_TRACE_MISALIGNED=1` names a load of a wide
                    // value from an address not aligned to it, before the
                    // debug build's own check aborts; safe to run with.
                    if cfg!(debug_assertions)
                        && (p as usize) % 8 != 0
                        && matches!(
                            target,
                            HirType::I64 | HirType::U64 | HirType::F64 | HirType::Ptr(_)
                        )
                        && std::env::var_os("ZYNTAX_TRACE_MISALIGNED").is_some()
                    {
                        eprintln!(
                            "[interp] misaligned load of {:?} at {:p} in {} pc={pc}",
                            target,
                            p,
                            module
                                .functions
                                .get(&func_id)
                                .and_then(|f| f.name.resolve_global())
                                .unwrap_or_default()
                        );
                    }
                    regs[*dst as usize] = if held(target) == Held::ByReference {
                        // The loaded aggregate is a copy of its own.
                        let size = size_of_hir_ty(target);
                        let copy = self.frame_alloc(scratch, size.max(8));
                        if copy.is_null() {
                            return Err(InterpError::OutOfMemory);
                        }
                        // SAFETY: both point at `size` bytes.
                        unsafe { core::ptr::copy_nonoverlapping(p, copy, size) };
                        ZyntaxValue::Pointer(copy)
                    } else {
                        unsafe { read_typed(p, target) }
                    };
                    pc += 1;
                }
                Op::Store { ptr, val, ty } => {
                    let target = &cf.type_pool[*ty as usize];
                    // Accept Pointer (the natural shape) and Int/UInt
                    // (raw address — produced by `IAdd` of a Pointer
                    // and an offset, since the BC interp's arithmetic
                    // is i64-funneled and loses the Pointer tag).
                    let p = match &regs[*ptr as usize] {
                        ZyntaxValue::Pointer(p) => *p,
                        ZyntaxValue::Int(n) => *n as usize as *mut u8,
                        ZyntaxValue::UInt(n) => *n as usize as *mut u8,
                        other => {
                            return Err(InterpError::TypeMismatch {
                                expected: "pointer".to_string(),
                                got: format!("{:?}", other),
                            });
                        }
                    };
                    let v = regs[*val as usize].clone();
                    unsafe { write_typed(p, &v, target) };
                    pc += 1;
                }
                Op::ExtractValue { dst, src, idx, ty } => {
                    let indices = &cf.indices_pool[*idx as usize];
                    let agg_ty = &cf.type_pool[*ty as usize];
                    regs[*dst as usize] = match held(agg_ty) {
                        // The struct is its field.
                        Held::Flattened | Held::Scalar => regs[*src as usize].clone(),
                        Held::ByReference => {
                            let base = ptr_of(&regs[*src as usize])?;
                            let (offset, leaf) = field_path(agg_ty, indices);
                            // SAFETY: `base` holds a value of `agg_ty`.
                            unsafe { read_typed(base.add(offset), &leaf) }
                        }
                    };
                    pc += 1;
                }
                Op::InsertValue {
                    dst,
                    agg,
                    val,
                    idx,
                    ty,
                } => {
                    let indices = &cf.indices_pool[*idx as usize];
                    let agg_ty = &cf.type_pool[*ty as usize];
                    regs[*dst as usize] = match held(agg_ty) {
                        Held::Flattened | Held::Scalar => regs[*val as usize].clone(),
                        Held::ByReference => {
                            // Written in place: the aggregate's storage
                            // is the value, as it is in compiled code.
                            let base = ptr_of(&regs[*agg as usize])?;
                            let (offset, leaf) = field_path(agg_ty, indices);
                            // SAFETY: `base` holds a value of `agg_ty`.
                            unsafe { write_typed(base.add(offset), &regs[*val as usize], &leaf) };
                            ZyntaxValue::Pointer(base)
                        }
                    };
                    pc += 1;
                }
                Op::Gep {
                    dst,
                    ptr,
                    args,
                    stride,
                } => {
                    // Read base pointer — accept Pointer (natural shape)
                    // or Int/UInt (raw address — produced when an earlier
                    // GEP / arithmetic on a pointer flowed through the
                    // i64-funneled bus and lost the Pointer tag).
                    let base = match &regs[*ptr as usize] {
                        ZyntaxValue::Pointer(p) => *p as i64,
                        ZyntaxValue::Int(n) => *n,
                        ZyntaxValue::UInt(n) => *n as i64,
                        other => {
                            return Err(InterpError::TypeMismatch {
                                expected: "pointer".to_string(),
                                got: format!("{:?}", other),
                            });
                        }
                    };
                    let idx_regs = &cf.args_pool[*args as usize];
                    let strides = &cf.gep_stride_pool[*stride as usize];
                    let mut addr = base;
                    for (i, r) in idx_regs.iter().enumerate() {
                        let idx_val = value_to_i64(&regs[*r as usize]).ok_or_else(|| {
                            InterpError::TypeMismatch {
                                expected: "integer (GEP index)".to_string(),
                                got: format!("{:?}", regs[*r as usize]),
                            }
                        })?;
                        let (s, fixed) = strides.get(i).copied().unwrap_or((0, 0));
                        addr = addr
                            .wrapping_add(idx_val.wrapping_mul(s))
                            .wrapping_add(fixed);
                    }
                    regs[*dst as usize] = ZyntaxValue::Pointer(addr as usize as *mut u8);
                    pc += 1;
                }

                // ── SIMD / vector (scalarized: value is Array of lanes) ──
                Op::VSplat { dst, scalar, lanes } => {
                    let s = regs[*scalar as usize].clone();
                    regs[*dst as usize] = ZyntaxValue::Array(vec![s; *lanes as usize]);
                    pc += 1;
                }
                Op::VLoad {
                    dst,
                    ptr,
                    lanes,
                    elem_ty,
                } => {
                    let elem = cf.type_pool[*elem_ty as usize].clone();
                    let esize = size_of_hir_ty(&elem);
                    let p = ptr_of(&regs[*ptr as usize])?;
                    let mut lane_vals = Vec::with_capacity(*lanes as usize);
                    for i in 0..*lanes as usize {
                        lane_vals.push(unsafe { read_typed(p.wrapping_add(i * esize), &elem) });
                    }
                    regs[*dst as usize] = ZyntaxValue::Array(lane_vals);
                    pc += 1;
                }
                Op::VStore { val, ptr, elem_ty } => {
                    let elem = cf.type_pool[*elem_ty as usize].clone();
                    let esize = size_of_hir_ty(&elem);
                    let p = ptr_of(&regs[*ptr as usize])?;
                    let lane_vals = as_vector(&regs[*val as usize])?;
                    for (i, lane) in lane_vals.iter().enumerate() {
                        unsafe { write_typed(p.wrapping_add(i * esize), lane, &elem) };
                    }
                    pc += 1;
                }
                Op::VExtract { dst, vector, lane } => {
                    let lane_vals = as_vector(&regs[*vector as usize])?;
                    regs[*dst as usize] = lane_vals
                        .get(*lane as usize)
                        .cloned()
                        .unwrap_or(ZyntaxValue::Undef);
                    pc += 1;
                }
                Op::VInsert {
                    dst,
                    vector,
                    scalar,
                    lane,
                } => {
                    let mut lane_vals = as_vector(&regs[*vector as usize])?;
                    let s = regs[*scalar as usize].clone();
                    if let Some(slot) = lane_vals.get_mut(*lane as usize) {
                        *slot = s;
                    }
                    regs[*dst as usize] = ZyntaxValue::Array(lane_vals);
                    pc += 1;
                }
                Op::VReduce { dst, vector, op } => {
                    let lane_vals = as_vector(&regs[*vector as usize])?;
                    let mut acc = lane_vals.first().cloned().unwrap_or(ZyntaxValue::Int(0));
                    for lane in lane_vals.iter().skip(1) {
                        acc = apply_lane_binop(*op, &acc, lane)?;
                    }
                    regs[*dst as usize] = acc;
                    pc += 1;
                }
                Op::VBinOp { dst, lhs, rhs, op } => {
                    let a = as_vector(&regs[*lhs as usize])?;
                    let b = as_vector(&regs[*rhs as usize])?;
                    let n = a.len().min(b.len());
                    let mut out = Vec::with_capacity(n);
                    for i in 0..n {
                        out.push(apply_lane_binop(*op, &a[i], &b[i])?);
                    }
                    regs[*dst as usize] = ZyntaxValue::Array(out);
                    pc += 1;
                }
                Op::VUnary { dst, operand, kind } => {
                    let v = as_vector(&regs[*operand as usize])?;
                    let out: Vec<ZyntaxValue> =
                        v.iter().map(|lane| apply_lane_unary(*kind, lane)).collect();
                    regs[*dst as usize] = ZyntaxValue::Array(out);
                    pc += 1;
                }
                Op::VMinMax {
                    dst,
                    lhs,
                    rhs,
                    kind,
                } => {
                    let a = as_vector(&regs[*lhs as usize])?;
                    let b = as_vector(&regs[*rhs as usize])?;
                    let n = a.len().min(b.len());
                    let out: Vec<ZyntaxValue> = (0..n)
                        .map(|i| apply_lane_minmax(*kind, &a[i], &b[i]))
                        .collect();
                    regs[*dst as usize] = ZyntaxValue::Array(out);
                    pc += 1;
                }
                Op::VDot {
                    dst,
                    acc,
                    a,
                    b,
                    unsigned,
                } => {
                    // 16×i8 · 16×i8 → 4×i32: each output lane accumulates the 4
                    // (widened) products in its byte group onto `acc`. `a` is
                    // signed; `b` is unsigned when `unsigned` (VPDPBUSD/USDOT).
                    let acc_v = as_vector(&regs[*acc as usize])?;
                    let a_v = as_vector(&regs[*a as usize])?;
                    let b_v = as_vector(&regs[*b as usize])?;
                    let zero = ZyntaxValue::Int(0);
                    let mut out = Vec::with_capacity(4);
                    for j in 0..4 {
                        let mut sum = value_to_i64(acc_v.get(j).unwrap_or(&zero)).unwrap_or(0);
                        for k in 0..4 {
                            let idx = 4 * j + k;
                            let av = value_to_i64(a_v.get(idx).unwrap_or(&zero)).unwrap_or(0);
                            let bv_raw = value_to_i64(b_v.get(idx).unwrap_or(&zero)).unwrap_or(0);
                            let bv = if *unsigned {
                                (bv_raw as u8) as i64
                            } else {
                                (bv_raw as i8) as i64
                            };
                            sum = sum.wrapping_add(av.wrapping_mul(bv));
                        }
                        out.push(ZyntaxValue::I32(sum as i32));
                    }
                    regs[*dst as usize] = ZyntaxValue::Array(out);
                    pc += 1;
                }
                Op::Jump { target } => {
                    pc = *target as usize;
                }
                Op::JumpIf { cond, t, f } => {
                    let take = match &regs[*cond as usize] {
                        ZyntaxValue::Bool(b) => *b,
                        other => value_to_i64(other).map(|n| n != 0).ok_or_else(|| {
                            InterpError::TypeMismatch {
                                expected: "bool/int".to_string(),
                                got: format!("{:?}", other),
                            }
                        })?,
                    };
                    pc = if take { *t as usize } else { *f as usize };
                }
                Op::Switch {
                    scrut,
                    table,
                    default,
                } => {
                    let v = value_to_i64(&regs[*scrut as usize]).ok_or_else(|| {
                        InterpError::TypeMismatch {
                            expected: "integer".to_string(),
                            got: "non-integer scrutinee".to_string(),
                        }
                    })?;
                    let tbl = &cf.switch_pool[*table as usize];
                    let mut taken = *default;
                    for (k, target) in tbl {
                        if *k == v {
                            taken = *target;
                            break;
                        }
                    }
                    pc = taken as usize;
                }
                Op::Ret { src } => {
                    let value = regs[*src as usize].clone();
                    if cf.returns_through_destination && !dest.is_null() {
                        // The caller's storage receives the struct; the
                        // frame's own copy goes with the frame.
                        if let Some(from) = value_to_i64(&value) {
                            let from = from as usize as *const u8;
                            if !from.is_null() && from != dest as *const u8 {
                                // SAFETY: both hold `destination_size` bytes.
                                unsafe {
                                    core::ptr::copy(from, dest, cf.destination_size);
                                }
                            }
                        }
                        return Ok(ZyntaxValue::Pointer(dest));
                    }
                    return Ok(value);
                }
                Op::LoopHeader { site } => {
                    pc += 1;
                    let Some(bead) = bead else { continue };
                    let i = *site as usize;
                    visits[i] = visits[i].saturating_add(1);
                    let osr_site = &cf.osr_sites[i];
                    // Ask once: early while the loop is warm when a worker
                    // will take it, else when it is hot. From then on the
                    // slot is watched.
                    let asks = if slots[i].is_null() {
                        visits[i] == OSR_REQUEST_VISITS
                            || (visits[i] == OSR_WARM_VISITS
                                && crate::osr::compile_worker_present())
                    } else {
                        false
                    };
                    if asks {
                        // Waiting from here until this frame leaves `run`.
                        if !waits {
                            waits = true;
                            self.waiting_marks.push(bead);
                            crate::osr::frame_waits(bead);
                        }
                        if trace_enabled() {
                            eprintln!(
                                "[interp] asks at site=0x{:x} after {} visits",
                                osr_site.site_key, visits[i]
                            );
                        }
                        crate::osr::osr_request_promotion_interpreted(bead, osr_site.site_key);
                        slots[i] = crate::osr::helper_slot_addr(bead, osr_site.site_key)
                            as *const std::sync::atomic::AtomicU64;
                    } else if slots[i].is_null() {
                        continue;
                    }
                    // SAFETY: the slot lives for the process.
                    let helper = unsafe { &*slots[i] }.load(std::sync::atomic::Ordering::Acquire);
                    // A frame with nowhere to write its result stays here.
                    if helper != 0 && (osr_site.destination.is_none() || !dest.is_null()) {
                        // Leaving through this one: the frame runs natively
                        // from here, so no other resume point is owed to it.
                        if waits {
                            self.waiting_marks.pop();
                            crate::osr::frame_left(bead);
                        }
                        return self.transfer(
                            helper as *const u8,
                            osr_site,
                            &regs,
                            &mut roots,
                            scratch,
                            dest,
                        );
                    }
                }
                Op::RetVoid => {
                    return Ok(ZyntaxValue::Void);
                }
                Op::Unreachable => {
                    return Err(InterpError::Host(
                        "execution reached unreachable terminator".to_string(),
                    ));
                }
                Op::CallFn {
                    dst,
                    has_dst,
                    fn_id,
                    args,
                } => {
                    let arg_regs = &cf.args_pool[*args as usize];
                    let mut arg_vals: Vec<ZyntaxValue> =
                        arg_regs.iter().map(|r| regs[*r as usize].clone()).collect();
                    let callee = module.functions.get(fn_id);
                    // An extern declared over dynamic boxes receives its
                    // arguments boxed, as compiled call sites box them.
                    let mut boxes: Vec<*mut u8> = Vec::new();
                    if let Some(f) = callee.filter(|f| f.is_external) {
                        let link = f.link_name.clone().or_else(|| f.name.resolve_global());
                        if let Some(sig) =
                            link.and_then(|n| self.symbols.get(&n)).and_then(|e| e.sig)
                        {
                            let arg_types: Vec<HirType> = arg_regs
                                .iter()
                                .map(|r| cf.reg_types[*r as usize].clone())
                                .collect();
                            boxes = self.box_dynamic_args(&sig, &arg_types, &mut arg_vals);
                        }
                    }
                    // Storage for a struct the callee hands back through
                    // a destination.
                    let callee_dest = match self.shape_of(module, *fn_id) {
                        Some((sig, _)) if sig.destination => {
                            self.frame_alloc(scratch, size_of_hir_ty(&sig.ret).max(8))
                        }
                        _ => core::ptr::null_mut(),
                    };
                    let ret = self.call_by_id(module, *fn_id, arg_vals, callee_dest);
                    crate::collector::clear_dead_frames();
                    for b in boxes {
                        self.memory.free(b);
                    }
                    let ret = ret?;
                    if *has_dst {
                        regs[*dst as usize] = ret;
                    }
                    pc += 1;
                }
                Op::CallSym {
                    dst,
                    has_dst,
                    sym,
                    args,
                    sig,
                } => {
                    let arg_regs = &cf.args_pool[*args as usize];
                    let mut arg_vals: Vec<ZyntaxValue> =
                        arg_regs.iter().map(|r| regs[*r as usize].clone()).collect();
                    let sym_idx = *sym as usize;
                    let shape = &cf.sig_pool[*sig as usize];

                    // Phase J.5 wasm32 escape hatch: route through the
                    // installed symbol-call dispatcher first. A `Some(v)`
                    // return short-circuits the transmute path; `None`
                    // falls through to the native transmute below.
                    let dispatched = if let Some(disp) = self.symbol_call_dispatcher.as_mut() {
                        let name = &cf.symbol_pool[sym_idx];
                        disp(name, arg_vals.clone())?
                    } else {
                        None
                    };

                    let result_val = if let Some(v) = dispatched {
                        v
                    } else {
                        // Fast path: pre-resolved entry from `run()`
                        // entry. Slow path: re-probe `self.symbols`
                        // for symbols registered after this `run` started
                        // (rare).
                        let entry = match resolved_symbols.get(sym_idx).and_then(|e| *e) {
                            Some(e) => e,
                            None => {
                                let name = &cf.symbol_pool[sym_idx];
                                match self.symbols.get(name).copied() {
                                    Some(e) => e,
                                    None => {
                                        let ptr = process_symbol(name).ok_or_else(|| {
                                            InterpError::UnknownFunction(name.clone())
                                        })?;
                                        let e = SymbolEntry {
                                            ptr,
                                            param_count: arg_vals.len().min(255) as u8,
                                            sig: None,
                                        };
                                        self.symbols.insert(name.clone(), e);
                                        e
                                    }
                                }
                            }
                        };
                        if self.thunk_source.is_some() {
                            // A symbol declared over dynamic boxes gets
                            // its arguments boxed, as compiled call
                            // sites box them.
                            let boxes = match entry.sig {
                                Some(zsig) => {
                                    self.box_dynamic_args(&zsig, &shape.params, &mut arg_vals)
                                }
                                None => Vec::new(),
                            };
                            let thunk = self.site_thunk(cf, *sig)?;
                            let result = self.call_native_through(
                                entry.ptr,
                                shape,
                                thunk,
                                &arg_vals,
                                core::ptr::null_mut(),
                            );
                            for b in boxes {
                                self.memory.free(b);
                            }
                            result?
                        } else if let Some(zsig) = entry.sig {
                            // Typed marshalling path: the ZRTL signature
                            // supplies the register classes.
                            call_extern_symbol_typed(entry.ptr, &arg_vals, &zsig)?
                        } else {
                            let raw = call_extern_symbol(entry.ptr, &arg_vals);
                            value_from_i64_as(&shape.ret, raw)
                        }
                    };
                    crate::collector::clear_dead_frames();

                    if *has_dst {
                        let v = match result_val {
                            ZyntaxValue::Int(i) => value_from_i64_as(&shape.ret, i),
                            other => other,
                        };
                        regs[*dst as usize] = v;
                    }
                    pc += 1;
                }
                Op::AsyncSaveSlot {
                    frame_reg,
                    slot,
                    val_reg,
                } => {
                    let frame_ptr = match &regs[*frame_reg as usize] {
                        ZyntaxValue::Pointer(p) => *p,
                        ZyntaxValue::Int(n) => *n as usize as *mut u8,
                        ZyntaxValue::UInt(n) => *n as usize as *mut u8,
                        other => {
                            return Err(InterpError::TypeMismatch {
                                expected: "pointer (AsyncSaveSlot frame)".to_string(),
                                got: format!("{:?}", other),
                            });
                        }
                    };
                    let val_i64 = value_to_i64(&regs[*val_reg as usize]).ok_or_else(|| {
                        InterpError::TypeMismatch {
                            expected: "integer (AsyncSaveSlot value)".to_string(),
                            got: format!("{:?}", regs[*val_reg as usize]),
                        }
                    })?;
                    unsafe {
                        let dst_ptr = frame_ptr.add((*slot as usize) * 8) as *mut i64;
                        *dst_ptr = val_i64;
                    }
                    pc += 1;
                }
                Op::AsyncLoadSlot {
                    dst,
                    frame_reg,
                    slot,
                    ty,
                } => {
                    let frame_ptr = match &regs[*frame_reg as usize] {
                        ZyntaxValue::Pointer(p) => *p,
                        ZyntaxValue::Int(n) => *n as usize as *mut u8,
                        ZyntaxValue::UInt(n) => *n as usize as *mut u8,
                        other => {
                            return Err(InterpError::TypeMismatch {
                                expected: "pointer (AsyncLoadSlot frame)".to_string(),
                                got: format!("{:?}", other),
                            });
                        }
                    };
                    let target_ty = &cf.type_pool[*ty as usize];
                    let src = unsafe { frame_ptr.add((*slot as usize) * 8) };
                    let v = unsafe { read_typed(src, target_ty) };
                    regs[*dst as usize] = v;
                    pc += 1;
                }
                Op::CallIndirect {
                    dst,
                    has_dst,
                    fn_ptr_reg,
                    args,
                    sig,
                } => {
                    let handle = match &regs[*fn_ptr_reg as usize] {
                        ZyntaxValue::Pointer(p) => *p as usize as i64,
                        ZyntaxValue::Int(n) => *n,
                        ZyntaxValue::UInt(n) => *n as i64,
                        other => {
                            return Err(InterpError::TypeMismatch {
                                expected: "function-pointer / handle".to_string(),
                                got: format!("{:?}", other),
                            });
                        }
                    };
                    let arg_regs = &cf.args_pool[*args as usize];
                    let arg_vals: Vec<ZyntaxValue> =
                        arg_regs.iter().map(|r| regs[*r as usize].clone()).collect();
                    let shape = &cf.sig_pool[*sig as usize];
                    // A pointer to one of the module's functions is called
                    // by id, which runs it here while it has no code. A
                    // function of another module (one the program
                    // compiled while running) is native code as far as
                    // this module knows.
                    let own = if handle != 0 && self.indirect_call_dispatcher.is_none() {
                        self.address_source
                            .as_ref()
                            .and_then(|f| f(handle as usize))
                            .filter(|id| module.functions.contains_key(id))
                    } else {
                        None
                    };
                    // A host dispatcher (wasm) resolves the handle itself;
                    // otherwise the handle is native code.
                    let result = if let Some(target) = own {
                        self.call_by_id(module, target, arg_vals, core::ptr::null_mut())?
                    } else if let Some(dispatcher) = self.indirect_call_dispatcher.as_mut() {
                        dispatcher(handle, arg_vals)?
                    } else {
                        if self.thunk_source.is_none() {
                            return Err(InterpError::UnsupportedInstruction(
                                "indirect call without dispatcher".to_string(),
                            ));
                        }
                        if handle == 0 {
                            return Err(InterpError::Host(
                                "a call through a null function pointer".to_string(),
                            ));
                        }
                        let thunk = self.site_thunk(cf, *sig)?;
                        self.call_native_through(
                            handle as usize as *const u8,
                            shape,
                            thunk,
                            &arg_vals,
                            core::ptr::null_mut(),
                        )?
                    };
                    crate::collector::clear_dead_frames();
                    if *has_dst {
                        let v = match result {
                            ZyntaxValue::Int(i) => value_from_i64_as(&shape.ret, i),
                            other => other,
                        };
                        regs[*dst as usize] = v;
                    }
                    pc += 1;
                }
            }
        }
        // Fell off the end without a Ret → void.
        Ok(ZyntaxValue::Void)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (free functions)
// ─────────────────────────────────────────────────────────────────────────────

/// Integer-binop helper. Pulls i64 from each operand, runs `f`,
/// returns the raw i64 — callers (the dispatch loop) mask to the
/// target HirType from `cf.reg_types[dst]` via `value_from_i64_as`.
fn ibin(
    l: &ZyntaxValue,
    r: &ZyntaxValue,
    f: impl FnOnce(i64, i64) -> i64,
) -> Result<ZyntaxValue, InterpError> {
    let li = ireg_i64(l)?;
    let ri = ireg_i64(r)?;
    Ok(ZyntaxValue::Int(f(li, ri)))
}

fn ireg_i64(v: &ZyntaxValue) -> Result<i64, InterpError> {
    value_to_i64(v).ok_or_else(|| InterpError::TypeMismatch {
        expected: "integer".to_string(),
        got: format!("{:?}", v),
    })
}

fn freg_f64(v: &ZyntaxValue) -> Result<f64, InterpError> {
    value_to_f64(v).ok_or_else(|| InterpError::TypeMismatch {
        expected: "float".to_string(),
        got: format!("{:?}", v),
    })
}

/// Wrap an `f64` arithmetic result in the destination's precise float width.
/// The interpreter funnels float math through `f64`; narrowing the result to
/// `F32` when the register's `HirType` says so keeps `f32` values correctly
/// typed — and bit-matches the native backends, since for `+`/`-`/`*` a single
/// f64 op on f32-representable inputs rounds to the same `f32` the hardware
/// produces. Without this, every `f32` scalar op returned an `f64`-tagged
/// `Float`, diverging from Cranelift/LLVM (and mistyping the value).
fn fval(dst_ty: &HirType, r: f64) -> ZyntaxValue {
    match dst_ty {
        HirType::F32 => ZyntaxValue::F32(r as f32),
        _ => ZyntaxValue::Float(r),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD / vector helpers (scalarized lanes)
// ─────────────────────────────────────────────────────────────────────────────

/// Coerce a register value to a raw memory pointer (accepts the raw-address
/// `Int`/`UInt` shapes the i64-funneled arithmetic bus produces, same as the
/// scalar `Op::Load`/`Op::Store`).
fn ptr_of(v: &ZyntaxValue) -> Result<*mut u8, InterpError> {
    match v {
        ZyntaxValue::Pointer(p) => Ok(*p),
        ZyntaxValue::Int(n) => Ok(*n as usize as *mut u8),
        ZyntaxValue::UInt(n) => Ok(*n as usize as *mut u8),
        other => Err(InterpError::TypeMismatch {
            expected: "pointer".to_string(),
            got: format!("{:?}", other),
        }),
    }
}

/// Read the lanes of a scalarized vector value (held as `ZyntaxValue::Array`).
/// Returns an owned copy so the caller can write its result register without a
/// borrow conflict.
fn as_vector(v: &ZyntaxValue) -> Result<Vec<ZyntaxValue>, InterpError> {
    match v {
        ZyntaxValue::Array(lanes) => Ok(lanes.clone()),
        other => Err(InterpError::TypeMismatch {
            expected: "vector (Array of lanes)".to_string(),
            got: format!("{:?}", other),
        }),
    }
}

/// Map a scalar `ZyntaxValue` back to its precise `HirType`, so a lane-wise op
/// can reconstruct the result in the element's own width (e.g. keep `f32x4`
/// lanes as `F32`, not widen to `f64`).
fn hir_ty_of_value(v: &ZyntaxValue) -> HirType {
    match v {
        ZyntaxValue::Bool(_) => HirType::Bool,
        ZyntaxValue::I8(_) => HirType::I8,
        ZyntaxValue::I16(_) => HirType::I16,
        ZyntaxValue::I32(_) => HirType::I32,
        ZyntaxValue::Int(_) => HirType::I64,
        ZyntaxValue::U8(_) => HirType::U8,
        ZyntaxValue::U16(_) => HirType::U16,
        ZyntaxValue::U32(_) => HirType::U32,
        ZyntaxValue::UInt(_) => HirType::U64,
        ZyntaxValue::F32(_) => HirType::F32,
        ZyntaxValue::Float(_) => HirType::F64,
        _ => HirType::I64,
    }
}

/// Fused multiply-add on one vector lane triple: `a * b + c`, preserving
/// the lane element width (F32 lanes stay F32). Matches the width
/// convention of `apply_lane_binop` (compute in f64, round to the lane
/// type), so mixed vector arithmetic and FMA agree lane-for-lane.
fn lane_mul_add(
    a: &ZyntaxValue,
    b: &ZyntaxValue,
    c: &ZyntaxValue,
) -> Result<ZyntaxValue, InterpError> {
    let r = freg_f64(a)?.mul_add(freg_f64(b)?, freg_f64(c)?);
    Ok(match a {
        ZyntaxValue::F32(_) => ZyntaxValue::F32(r as f32),
        _ => ZyntaxValue::Float(r),
    })
}

/// Apply a binary op to one lane pair, preserving the left lane's element
/// width. Float when either lane is float; integer otherwise. This is the
/// scalar kernel of `VBinOp` (element-wise) and `VReduce` (fold).
fn apply_lane_binop(
    op: BinaryOp,
    a: &ZyntaxValue,
    b: &ZyntaxValue,
) -> Result<ZyntaxValue, InterpError> {
    let is_float = matches!(a, ZyntaxValue::Float(_) | ZyntaxValue::F32(_))
        || matches!(b, ZyntaxValue::Float(_) | ZyntaxValue::F32(_));
    if is_float {
        let x = freg_f64(a)?;
        let y = freg_f64(b)?;
        let r = match op {
            BinaryOp::Add | BinaryOp::FAdd => x + y,
            BinaryOp::Sub | BinaryOp::FSub => x - y,
            BinaryOp::Mul | BinaryOp::FMul => x * y,
            BinaryOp::Div | BinaryOp::FDiv => x / y,
            other => {
                return Err(InterpError::UnsupportedInstruction(format!(
                    "vector float lane op {:?}",
                    other
                )));
            }
        };
        Ok(match a {
            ZyntaxValue::F32(_) => ZyntaxValue::F32(r as f32),
            _ => ZyntaxValue::Float(r),
        })
    } else {
        let x = ireg_i64(a)?;
        let y = ireg_i64(b)?;
        let r = match op {
            BinaryOp::Add | BinaryOp::FAdd => x.wrapping_add(y),
            BinaryOp::Sub | BinaryOp::FSub => x.wrapping_sub(y),
            BinaryOp::Mul | BinaryOp::FMul => x.wrapping_mul(y),
            BinaryOp::Div | BinaryOp::FDiv => {
                if y == 0 {
                    0
                } else {
                    x.wrapping_div(y)
                }
            }
            BinaryOp::And => x & y,
            BinaryOp::Or => x | y,
            BinaryOp::Xor => x ^ y,
            other => {
                return Err(InterpError::UnsupportedInstruction(format!(
                    "vector int lane op {:?}",
                    other
                )));
            }
        };
        Ok(value_from_i64_as(&hir_ty_of_value(a), r))
    }
}

/// One lane of `VectorUnaryOp` — float only, width-preserving. `Round` uses
/// round-half-to-even to match hardware `nearest` (Cranelift `nearest`, wasm
/// `f*.nearest`), not Rust's half-away-from-zero `round`.
fn apply_lane_unary(kind: VectorUnaryKind, a: &ZyntaxValue) -> ZyntaxValue {
    let x = value_to_f64(a).unwrap_or(0.0);
    let r = match kind {
        VectorUnaryKind::Sqrt => x.sqrt(),
        VectorUnaryKind::Abs => x.abs(),
        VectorUnaryKind::Neg => -x,
        VectorUnaryKind::Ceil => x.ceil(),
        VectorUnaryKind::Floor => x.floor(),
        VectorUnaryKind::Trunc => x.trunc(),
        VectorUnaryKind::Round => x.round_ties_even(),
    };
    match a {
        ZyntaxValue::F32(_) => ZyntaxValue::F32(r as f32),
        _ => ZyntaxValue::Float(r),
    }
}

/// One lane of `VectorMinMax` — float only, width-preserving.
fn apply_lane_minmax(kind: VectorMinMaxKind, a: &ZyntaxValue, b: &ZyntaxValue) -> ZyntaxValue {
    let x = value_to_f64(a).unwrap_or(0.0);
    let y = value_to_f64(b).unwrap_or(0.0);
    let r = match kind {
        VectorMinMaxKind::Min => x.min(y),
        VectorMinMaxKind::Max => x.max(y),
    };
    match a {
        ZyntaxValue::F32(_) => ZyntaxValue::F32(r as f32),
        _ => ZyntaxValue::Float(r),
    }
}

fn eval_cast(op: CastOp, o: ZyntaxValue, ty: &HirType) -> Result<ZyntaxValue, InterpError> {
    let raw_i64 = value_to_i64(&o);
    let raw_f64 = value_to_f64(&o);
    match op {
        CastOp::Trunc | CastOp::ZExt | CastOp::SExt | CastOp::Bitcast => {
            if let Some(n) = raw_i64 {
                Ok(value_from_i64_as(ty, n))
            } else {
                Ok(o)
            }
        }
        CastOp::PtrToInt => {
            if let ZyntaxValue::Pointer(p) = o {
                Ok(value_from_i64_as(ty, p as i64))
            } else if let Some(n) = raw_i64 {
                Ok(value_from_i64_as(ty, n))
            } else {
                Ok(o)
            }
        }
        CastOp::IntToPtr => {
            if let Some(n) = raw_i64 {
                Ok(ZyntaxValue::Pointer(n as *mut u8))
            } else {
                Ok(o)
            }
        }
        // Float widening / narrowing. ZyntaxValue stores all floats as
        // f64; for `FpTrunc` to f32 we narrow then re-widen so the
        // resulting `Float` has f32-precision content.
        CastOp::FpExt => match raw_f64 {
            Some(x) => Ok(ZyntaxValue::Float(x)),
            None => Ok(o),
        },
        CastOp::FpTrunc => match raw_f64 {
            Some(x) => Ok(ZyntaxValue::Float(x as f32 as f64)),
            None => Ok(o),
        },
        CastOp::FpToSi | CastOp::FpToUi => match raw_f64 {
            Some(x) => Ok(value_from_i64_as(ty, x as i64)),
            None => Ok(o),
        },
        CastOp::SiToFp | CastOp::UiToFp => {
            if let Some(n) = raw_i64 {
                match ty {
                    HirType::F64 => Ok(ZyntaxValue::Float(n as f64)),
                    HirType::F32 => Ok(ZyntaxValue::Float(n as f32 as f64)),
                    _ => Ok(o),
                }
            } else {
                Ok(o)
            }
        }
    }
}

/// The byte offset and type of the field `indices` names inside a value
/// of `ty`, at the native layout.
fn field_path(ty: &HirType, indices: &[u32]) -> (usize, HirType) {
    let mut offset = 0usize;
    let mut current = ty.clone();
    for &i in indices {
        match current {
            HirType::Struct(s) => {
                let layout = struct_layout(&s);
                let i = (i as usize).min(s.fields.len().saturating_sub(1));
                offset += layout.offsets.get(i).copied().unwrap_or(0);
                current = s.fields.get(i).cloned().unwrap_or(HirType::I64);
            }
            HirType::Array(elem, _) => {
                offset += size_of_hir_ty(&elem) * i as usize;
                current = *elem;
            }
            other => {
                current = other;
                break;
            }
        }
    }
    (offset, current)
}

unsafe fn read_typed(ptr: *mut u8, ty: &HirType) -> ZyntaxValue {
    match ty {
        HirType::Bool => ZyntaxValue::Bool(*ptr != 0),
        HirType::I8 => ZyntaxValue::I8(*ptr as i8),
        HirType::U8 => ZyntaxValue::U8(*ptr),
        HirType::I16 => ZyntaxValue::I16(*(ptr as *const i16)),
        HirType::U16 => ZyntaxValue::U16(*(ptr as *const u16)),
        HirType::I32 => ZyntaxValue::I32(*(ptr as *const i32)),
        HirType::U32 => ZyntaxValue::U32(*(ptr as *const u32)),
        HirType::F32 => ZyntaxValue::F32(*(ptr as *const f32)),
        HirType::I64 => ZyntaxValue::Int(*(ptr as *const i64)),
        HirType::U64 => ZyntaxValue::UInt(*(ptr as *const u64)),
        HirType::F64 => ZyntaxValue::Float(*(ptr as *const f64)),
        HirType::Ptr(_) => ZyntaxValue::Pointer(*(ptr as *const *mut u8)),
        HirType::Struct(s) if crate::abi::struct_carried_as_its_field(s).is_some() => {
            // A struct of one scalar field is that field.
            read_typed(ptr, &s.fields[0])
        }
        // An aggregate is held as the address of its storage; a reader
        // that wants its own copy makes one (see `Op::Load`).
        HirType::Struct(_) | HirType::Array(_, _) | HirType::Union(_) => ZyntaxValue::Pointer(ptr),
        // A vector is its lanes.
        HirType::Vector(elem, n) => {
            let stride = size_of_hir_ty(elem);
            ZyntaxValue::Array(
                (0..*n as usize)
                    .map(|i| read_typed(ptr.add(i * stride), elem))
                    .collect(),
            )
        }
        _ => ZyntaxValue::Int(*(ptr as *const i64)),
    }
}

/// Width-aware store. The `HirType` tells us how many bytes to write
/// regardless of whether the value uses a generic or precise variant.
unsafe fn write_typed(ptr: *mut u8, v: &ZyntaxValue, ty: &HirType) {
    // Coerce both sides to i64/f64 so we don't need a 2D table of
    // (HirType, variant) match arms.
    match ty {
        // A truth value arrives as a bool, or as the integer a compare or
        // a bit operation produced.
        HirType::Bool => match v {
            ZyntaxValue::Bool(b) => *ptr = *b as u8,
            other => {
                if let Some(n) = value_to_i64(other) {
                    *ptr = (n != 0) as u8;
                }
            }
        },
        HirType::I8 => {
            if let Some(n) = value_to_i64(v) {
                *(ptr as *mut i8) = n as i8;
            }
        }
        HirType::I16 => {
            if let Some(n) = value_to_i64(v) {
                *(ptr as *mut i16) = n as i16;
            }
        }
        HirType::I32 => {
            if let Some(n) = value_to_i64(v) {
                *(ptr as *mut i32) = n as i32;
            }
        }
        HirType::I64 => {
            if let Some(n) = value_to_i64(v) {
                *(ptr as *mut i64) = n;
            }
        }
        HirType::U8 => {
            if let Some(n) = value_to_i64(v) {
                *ptr = n as u8;
            }
        }
        HirType::U16 => {
            if let Some(n) = value_to_i64(v) {
                *(ptr as *mut u16) = n as u16;
            }
        }
        HirType::U32 => {
            if let Some(n) = value_to_i64(v) {
                *(ptr as *mut u32) = n as u32;
            }
        }
        HirType::U64 => {
            if let Some(n) = value_to_i64(v) {
                *(ptr as *mut u64) = n as u64;
            }
        }
        HirType::F32 => {
            if let Some(f) = value_to_f64(v) {
                *(ptr as *mut f32) = f as f32;
            }
        }
        HirType::F64 => {
            if let Some(f) = value_to_f64(v) {
                *(ptr as *mut f64) = f;
            }
        }
        HirType::Ptr(_) => {
            if let ZyntaxValue::Pointer(p) = v {
                *(ptr as *mut *mut u8) = *p;
            }
        }
        HirType::Struct(s) if crate::abi::struct_carried_as_its_field(s).is_some() => {
            write_typed(ptr, v, &s.fields[0]);
        }
        HirType::Vector(elem, _) => {
            if let ZyntaxValue::Array(lanes) = v {
                let stride = size_of_hir_ty(elem);
                for (i, lane) in lanes.iter().enumerate() {
                    write_typed(ptr.add(i * stride), lane, elem);
                }
            }
        }
        // The value is the address of the bytes to copy.
        HirType::Struct(_) | HirType::Array(_, _) | HirType::Union(_) => {
            if let Some(src) = value_to_i64(v) {
                let src = src as usize as *const u8;
                if !src.is_null() && src != ptr as *const u8 {
                    core::ptr::copy(src, ptr, size_of_hir_ty(ty));
                }
            }
        }
        _ => {
            if let Some(n) = value_to_i64(v) {
                *(ptr as *mut i64) = n;
            }
        }
    }
}

/// Dispatch into JIT'd code via the `JitDispatch` produced by a
/// tick callback. Routes f64 arguments through the float register
/// class (d0..d7 on AArch64, xmm0..xmm7 on x86_64) so the JIT'd
/// signature `(…, f64, …) -> …` actually sees its f64 argument in
/// the register Cranelift emitted code to read it from. AAPCS64
/// (and the System V x86_64 ABI) fill NGRN and NSRN independently,
/// so a transmute that normalises "all i64s first, then all f64s"
/// hits the same register layout as the JIT'd function's original
/// `(i64, i64, f64, …)` declaration order — we just reorder the
/// runtime arg values to match.
fn call_jit_dispatch(d: JitDispatch, args: &[ZyntaxValue]) -> ZyntaxValue {
    let n = args.len();
    // n > 8 or the function has unsupported param shape: fall back
    // to the all-i64 dispatcher. Wrong for f64 args, but the
    // tier-up install layer filters those out (only signatures the
    // bridge can handle reach this point).
    let n_capped = n.min(d.n_params as usize);

    // Partition into i64 channel (Pointer/Int/u*/i*) and f64 channel.
    let mut i64_args: [i64; 8] = [0; 8];
    let mut f64_args: [f64; 8] = [0.0; 8];
    let mut n_i64 = 0usize;
    let mut n_f64 = 0usize;
    for (i, v) in args.iter().take(n_capped).enumerate() {
        let is_float = (d.float_mask >> i) & 1 == 1;
        if is_float {
            f64_args[n_f64] = value_to_f64(v).unwrap_or(0.0);
            n_f64 += 1;
        } else {
            i64_args[n_i64] = value_to_i64(v).unwrap_or(0);
            n_i64 += 1;
        }
    }

    // If nothing's a float argument, route to the all-integer-args
    // path. The return still needs the right register class: a float
    // return lands in xmm0/d0, so transmute to a float-returning
    // signature rather than reading the integer register.
    if n_f64 == 0 {
        return match d.ret {
            JitRet::Int => ZyntaxValue::Int(call_extern_symbol(d.ptr, args)),
            JitRet::F32 => ZyntaxValue::F32(call_extern_symbol_ret_f32(d.ptr, args)),
            JitRet::F64 => ZyntaxValue::Float(call_extern_symbol_ret_f64(d.ptr, args)),
        };
    }

    // Mixed / all-float dispatch. Covers (n_i64, n_f64) pairs up to
    // a total of 4 arguments — enough for every JIT'd ZynML function
    // we hit today (nbody's `advance` is (2,1), mandelbrot's
    // `mandel_count` is (0,2)). The tier-up install filter rejects
    // shapes outside this matrix so a missing arm can't reach here.
    // Float returns combined with float args are rejected by
    // `jit_dispatch_supported`, so the mixed dispatcher only ever
    // returns through the integer register here.
    let raw_i64 =
        unsafe { call_extern_mixed::<i64>(d.ptr, n_i64, n_f64, &i64_args, &f64_args) }.unwrap_or(0);
    ZyntaxValue::Int(raw_i64)
}

/// All-integer-argument dispatch with an `f32` return — mirrors
/// `call_extern_symbol` but transmutes to a float-returning signature
/// so the result is read from the float register (xmm0 / s0).
fn call_extern_symbol_ret_f32(ptr: *const u8, args: &[ZyntaxValue]) -> f32 {
    let a: Vec<i64> = args.iter().map(|v| value_to_i64(v).unwrap_or(0)).collect();
    unsafe {
        match a.len() {
            0 => (core::mem::transmute::<_, extern "C" fn() -> f32>(ptr))(),
            1 => (core::mem::transmute::<_, extern "C" fn(i64) -> f32>(ptr))(a[0]),
            2 => (core::mem::transmute::<_, extern "C" fn(i64, i64) -> f32>(ptr))(a[0], a[1]),
            3 => (core::mem::transmute::<_, extern "C" fn(i64, i64, i64) -> f32>(ptr))(
                a[0], a[1], a[2],
            ),
            4 => (core::mem::transmute::<_, extern "C" fn(i64, i64, i64, i64) -> f32>(ptr))(
                a[0], a[1], a[2], a[3],
            ),
            _ => f32::from_bits(call_extern_symbol(ptr, args) as u32),
        }
    }
}

/// All-integer-argument dispatch with an `f64` return. See
/// `call_extern_symbol_ret_f32`.
fn call_extern_symbol_ret_f64(ptr: *const u8, args: &[ZyntaxValue]) -> f64 {
    let a: Vec<i64> = args.iter().map(|v| value_to_i64(v).unwrap_or(0)).collect();
    unsafe {
        match a.len() {
            0 => (core::mem::transmute::<_, extern "C" fn() -> f64>(ptr))(),
            1 => (core::mem::transmute::<_, extern "C" fn(i64) -> f64>(ptr))(a[0]),
            2 => (core::mem::transmute::<_, extern "C" fn(i64, i64) -> f64>(ptr))(a[0], a[1]),
            3 => (core::mem::transmute::<_, extern "C" fn(i64, i64, i64) -> f64>(ptr))(
                a[0], a[1], a[2],
            ),
            4 => (core::mem::transmute::<_, extern "C" fn(i64, i64, i64, i64) -> f64>(ptr))(
                a[0], a[1], a[2], a[3],
            ),
            _ => f64::from_bits(call_extern_symbol(ptr, args) as u64),
        }
    }
}

/// Dispatch into a native function whose signature has `n_i64`
/// integer/pointer args followed by `n_f64` floating-point args.
///
/// Safety: caller must have validated that `ptr` points to a
/// function with exactly the expected register-class
/// layout. The integer-first-then-float normalisation is sound on
/// AAPCS64 and System V x86_64 because both ABIs fill NGRN (or its
/// equivalent) and NSRN independently — argument position in the
/// declaration matters only within each class.
#[allow(clippy::too_many_arguments)]
unsafe fn call_extern_mixed<R: Copy>(
    ptr: *const u8,
    n_i64: usize,
    n_f64: usize,
    i: &[i64; 8],
    f: &[f64; 8],
) -> Option<R> {
    // The integer and float register sequences are independent on the
    // native ABIs supported by this dispatcher.
    match (n_i64, n_f64) {
        (0, 0) => {
            let g: extern "C" fn() -> R = core::mem::transmute(ptr);
            Some(g())
        }
        // 1-arg
        (1, 0) => {
            let g: extern "C" fn(i64) -> R = core::mem::transmute(ptr);
            Some(g(i[0]))
        }
        (0, 1) => {
            let g: extern "C" fn(f64) -> R = core::mem::transmute(ptr);
            Some(g(f[0]))
        }
        // 2-arg
        (2, 0) => {
            let g: extern "C" fn(i64, i64) -> R = core::mem::transmute(ptr);
            Some(g(i[0], i[1]))
        }
        (1, 1) => {
            let g: extern "C" fn(i64, f64) -> R = core::mem::transmute(ptr);
            Some(g(i[0], f[0]))
        }
        (0, 2) => {
            let g: extern "C" fn(f64, f64) -> R = core::mem::transmute(ptr);
            Some(g(f[0], f[1]))
        }
        // 3-arg
        (3, 0) => {
            let g: extern "C" fn(i64, i64, i64) -> R = core::mem::transmute(ptr);
            Some(g(i[0], i[1], i[2]))
        }
        (2, 1) => {
            let g: extern "C" fn(i64, i64, f64) -> R = core::mem::transmute(ptr);
            Some(g(i[0], i[1], f[0]))
        }
        (1, 2) => {
            let g: extern "C" fn(i64, f64, f64) -> R = core::mem::transmute(ptr);
            Some(g(i[0], f[0], f[1]))
        }
        (0, 3) => {
            let g: extern "C" fn(f64, f64, f64) -> R = core::mem::transmute(ptr);
            Some(g(f[0], f[1], f[2]))
        }
        // 4-arg
        (4, 0) => {
            let g: extern "C" fn(i64, i64, i64, i64) -> R = core::mem::transmute(ptr);
            Some(g(i[0], i[1], i[2], i[3]))
        }
        (3, 1) => {
            let g: extern "C" fn(i64, i64, i64, f64) -> R = core::mem::transmute(ptr);
            Some(g(i[0], i[1], i[2], f[0]))
        }
        (2, 2) => {
            let g: extern "C" fn(i64, i64, f64, f64) -> R = core::mem::transmute(ptr);
            Some(g(i[0], i[1], f[0], f[1]))
        }
        (1, 3) => {
            let g: extern "C" fn(i64, f64, f64, f64) -> R = core::mem::transmute(ptr);
            Some(g(i[0], f[0], f[1], f[2]))
        }
        (0, 4) => {
            let g: extern "C" fn(f64, f64, f64, f64) -> R = core::mem::transmute(ptr);
            Some(g(f[0], f[1], f[2], f[3]))
        }
        _ => None,
    }
}

/// Whether the JIT FFI bridge can dispatch a function with this
/// signature. Used by the tier-up install layer to skip functions
/// whose ABI exceeds the dispatcher's matrix — those functions stay
/// in BC interp instead of crashing at the bridge.
pub fn jit_dispatch_supported(params: &[HirType], ret: &HirType) -> bool {
    let n = params.len();
    if n > 8 {
        return false;
    }
    // The dispatcher marshals each parameter through either the integer
    // register file (int/pointer/bool) or the float register file
    // (f64). Anything else — f32 params (needs a 32-bit float
    // transmute the matrix doesn't have), vectors, structs — would be
    // placed in the wrong register class, so leave those functions in
    // the BC interpreter where they evaluate correctly.
    let param_ok = |t: &HirType| {
        matches!(
            t,
            HirType::Bool
                | HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::F64
                | HirType::Ptr(_)
                // A reference is a pointer at the ABI level — same size,
                // same register class. Source-level `Ptr<T>` lowers to
                // this, so excluding it left kernels taking typed buffers
                // compiled but never dispatched to, silently running in
                // the interpreter.
                | HirType::Ref { .. }
        )
    };
    if !params.iter().all(param_ok) {
        return false;
    }
    // Supported return classes: the integer register (int/pointer/
    // bool), or the float register (f32/f64) — the latter read via a
    // float-returning transmute in `call_jit_dispatch`.
    if !matches!(
        ret,
        HirType::Void
            | HirType::Bool
            | HirType::I8
            | HirType::I16
            | HirType::I32
            | HirType::I64
            | HirType::U8
            | HirType::U16
            | HirType::U32
            | HirType::U64
            | HirType::F32
            | HirType::F64
            | HirType::Ptr(_)
    ) {
        return false;
    }
    let n_f64 = params.iter().filter(|t| matches!(t, HirType::F64)).count();
    let n_i64 = n - n_f64;
    // All-integer-args: supports up to 8 args, including float returns
    // (read from the float register in `call_jit_dispatch`).
    if n_f64 == 0 {
        return n_i64 <= 8;
    }
    // Mixed args go through the integer-return matrix (total ≤ 4); a
    // float return combined with float args has no arm, so keep those
    // in the interpreter.
    if matches!(ret, HirType::F32 | HirType::F64) {
        return false;
    }
    n_i64 + n_f64 <= 4
}

/// Pack the `float_mask` byte for the `JitDispatch`. Bit `i` set ⇒
/// parameter `i` is `f64`. Only the lowest 8 bits are meaningful.
pub fn jit_float_mask(params: &[HirType]) -> u8 {
    let mut mask = 0u8;
    for (i, t) in params.iter().take(8).enumerate() {
        if matches!(t, HirType::F64) {
            mask |= 1 << i;
        }
    }
    mask
}

/// Typed marshalling of an FFI symbol call.
///
/// Routes each argument through the correct register file based on
/// its declared `TypeTag` — float categories take the float ABI
/// (`f64` / `f32` register), everything else flows through the
/// integer register file via `value_to_i64`. The return is decoded
/// from the matching register according to `sig.return_type`.
///
/// Shapes the bridge cannot represent return an interpreter error before
/// entering a function with the wrong C ABI.
fn call_extern_symbol_typed(
    ptr: *const u8,
    args: &[ZyntaxValue],
    sig: &crate::zrtl::ZrtlSymbolSig,
) -> Result<ZyntaxValue, InterpError> {
    use crate::hir::HirType;
    use crate::zrtl::{MAX_PARAMS, TypeCategory, TypeTag, ZrtlSigFlags};

    crate::collector::note_frame_depth();
    let pcount = sig.param_count as usize;
    if pcount != args.len() || pcount > MAX_PARAMS || pcount > 8 {
        return Err(InterpError::UnsupportedInstruction(format!(
            "typed extern ABI with {} parameters and {} arguments",
            pcount,
            args.len()
        )));
    }
    if sig.flags.contains(ZrtlSigFlags::VARIADIC) {
        return Err(InterpError::UnsupportedInstruction(
            "variadic typed extern ABI".to_string(),
        ));
    }
    let ret_cat = sig.return_type.category();
    let ret_hir = match ret_cat {
        TypeCategory::Void => HirType::Void,
        TypeCategory::Bool => HirType::Bool,
        TypeCategory::Int | TypeCategory::UInt => HirType::I64,
        TypeCategory::Float => HirType::F64,
        TypeCategory::Pointer | TypeCategory::Opaque => HirType::I64,
        _ => HirType::I64,
    };
    let mut i = [0i64; 8];
    let mut f = [0f64; 8];
    let mut n_i64 = 0;
    let mut n_f64 = 0;
    for (idx, arg) in args.iter().enumerate() {
        let tag = sig.params[idx];
        if matches!(
            tag.category(),
            TypeCategory::Void
                | TypeCategory::Struct
                | TypeCategory::Union
                | TypeCategory::Tuple
                | TypeCategory::Optional
                | TypeCategory::Result
                | TypeCategory::TraitObject
                | TypeCategory::Custom
        ) {
            return Err(InterpError::UnsupportedInstruction(format!(
                "typed extern ABI with aggregate parameter {idx}"
            )));
        }
        if tag.category() == TypeCategory::Float {
            if tag.type_id() != TypeTag::F64.type_id() {
                return Err(InterpError::UnsupportedInstruction(
                    "typed extern ABI with f32 parameter".to_string(),
                ));
            }
            f[n_f64] = value_to_f64(arg).ok_or_else(|| InterpError::TypeMismatch {
                expected: "float extern argument".to_string(),
                got: format!("{arg:?}"),
            })?;
            n_f64 += 1;
        } else {
            i[n_i64] = value_to_i64(arg).ok_or_else(|| InterpError::TypeMismatch {
                expected: "scalar extern argument".to_string(),
                got: format!("{arg:?}"),
            })?;
            n_i64 += 1;
        }
    }

    if cfg!(target_os = "windows") && n_i64 != 0 && n_f64 != 0 {
        return Err(InterpError::UnsupportedInstruction(
            "mixed integer/float extern ABI on Windows".to_string(),
        ));
    }

    if pcount > 4 {
        if n_f64 != 0 || matches!(ret_cat, TypeCategory::Float | TypeCategory::Void) {
            return Err(InterpError::UnsupportedInstruction(
                "typed extern ABI with more than four mixed or float-returning arguments"
                    .to_string(),
            ));
        }
        return Ok(value_from_i64_as(&ret_hir, call_extern_symbol(ptr, args)));
    }

    // SAFETY: the signature supplies the C register classes and the
    // dispatcher has an arm for every shape with up to four arguments.
    unsafe {
        match ret_cat {
            TypeCategory::Void => {
                call_extern_mixed::<()>(ptr, n_i64, n_f64, &i, &f).unwrap();
                Ok(ZyntaxValue::Void)
            }
            TypeCategory::Float if sig.return_type.type_id() == TypeTag::F32.type_id() => Ok(
                ZyntaxValue::F32(call_extern_mixed::<f32>(ptr, n_i64, n_f64, &i, &f).unwrap()),
            ),
            TypeCategory::Float => Ok(ZyntaxValue::Float(
                call_extern_mixed::<f64>(ptr, n_i64, n_f64, &i, &f).unwrap(),
            )),
            _ => Ok(value_from_i64_as(
                &ret_hir,
                call_extern_mixed::<i64>(ptr, n_i64, n_f64, &i, &f).unwrap(),
            )),
        }
    }
}

fn call_extern_symbol(ptr: *const u8, args: &[ZyntaxValue]) -> i64 {
    crate::collector::note_frame_depth();
    let raw_args: Vec<i64> = args.iter().map(|v| value_to_i64(v).unwrap_or(0)).collect();
    unsafe {
        match raw_args.len() {
            0 => {
                let f: extern "C" fn() -> i64 = core::mem::transmute(ptr);
                f()
            }
            1 => {
                let f: extern "C" fn(i64) -> i64 = core::mem::transmute(ptr);
                f(raw_args[0])
            }
            2 => {
                let f: extern "C" fn(i64, i64) -> i64 = core::mem::transmute(ptr);
                f(raw_args[0], raw_args[1])
            }
            3 => {
                let f: extern "C" fn(i64, i64, i64) -> i64 = core::mem::transmute(ptr);
                f(raw_args[0], raw_args[1], raw_args[2])
            }
            4 => {
                let f: extern "C" fn(i64, i64, i64, i64) -> i64 = core::mem::transmute(ptr);
                f(raw_args[0], raw_args[1], raw_args[2], raw_args[3])
            }
            5 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64) -> i64 = core::mem::transmute(ptr);
                f(
                    raw_args[0],
                    raw_args[1],
                    raw_args[2],
                    raw_args[3],
                    raw_args[4],
                )
            }
            6 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 =
                    core::mem::transmute(ptr);
                f(
                    raw_args[0],
                    raw_args[1],
                    raw_args[2],
                    raw_args[3],
                    raw_args[4],
                    raw_args[5],
                )
            }
            7 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    core::mem::transmute(ptr);
                f(
                    raw_args[0],
                    raw_args[1],
                    raw_args[2],
                    raw_args[3],
                    raw_args[4],
                    raw_args[5],
                    raw_args[6],
                )
            }
            8 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    core::mem::transmute(ptr);
                f(
                    raw_args[0],
                    raw_args[1],
                    raw_args[2],
                    raw_args[3],
                    raw_args[4],
                    raw_args[5],
                    raw_args[6],
                    raw_args[7],
                )
            }
            _ => 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        HirBlock, HirFunctionSignature, HirParam, HirTerminator, HirType, HirValue, HirValueKind,
        ParamAttributes,
    };
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
                    ownership: Default::default(),
                })
                .collect(),
            returns,
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
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

    #[test]
    fn typed_extern_uses_float_registers_for_mixed_arguments_and_return() {
        use crate::zrtl::{MAX_PARAMS, TypeTag, ZrtlSigFlags, ZrtlSymbolSig};

        extern "C" fn mixed(a: i64, b: f64, c: f64) -> f64 {
            a as f64 + b * 2.0 + c
        }
        let mut params = [TypeTag::VOID; MAX_PARAMS];
        params[..3].copy_from_slice(&[TypeTag::I64, TypeTag::F64, TypeTag::F64]);
        let sig = ZrtlSymbolSig {
            param_count: 3,
            flags: ZrtlSigFlags::NONE,
            return_type: TypeTag::F64,
            params,
        };
        let actual = call_extern_symbol_typed(
            mixed as *const u8,
            &[
                ZyntaxValue::Int(3),
                ZyntaxValue::Float(1.25),
                ZyntaxValue::Float(0.5),
            ],
            &sig,
        )
        .unwrap();
        assert_eq!(actual, ZyntaxValue::Float(6.0));
    }

    #[test]
    fn typed_extern_rejects_unrepresentable_float_argument() {
        use crate::zrtl::{MAX_PARAMS, TypeTag, ZrtlSigFlags, ZrtlSymbolSig};

        extern "C" fn narrow(_: f32) -> i64 {
            panic!("unsupported ABI must not enter the function")
        }
        let mut params = [TypeTag::VOID; MAX_PARAMS];
        params[0] = TypeTag::F32;
        let sig = ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::NONE,
            return_type: TypeTag::I64,
            params,
        };
        assert!(matches!(
            call_extern_symbol_typed(narrow as *const u8, &[ZyntaxValue::F32(2.5)], &sig),
            Err(InterpError::UnsupportedInstruction(_))
        ));
    }

    #[test]
    fn bytecode_reads_native_global_storage() {
        let global_id = HirId::new();
        let mut func = mk_fn("read_global", vec![], vec![HirType::I64]);
        let global_ptr = add_value(
            &mut func,
            HirType::Ptr(Box::new(HirType::I64)),
            HirValueKind::Global(global_id),
        );
        let loaded = add_value(&mut func, HirType::I64, HirValueKind::Instruction);
        let entry = func.blocks.get_mut(&func.entry_block).unwrap();
        entry.instructions.push(HirInstruction::Load {
            result: loaded,
            ty: HirType::I64,
            ptr: global_ptr,
            align: 8,
            volatile: false,
        });
        entry.terminator = HirTerminator::Return {
            values: vec![loaded],
        };

        let mut module = HirModule::new(InternedString::new_global("test"));
        module.globals.insert(
            global_id,
            crate::hir::HirGlobal {
                id: global_id,
                name: InternedString::new_global("shared"),
                ty: HirType::I64,
                initializer: Some(HirConstant::I64(0)),
                is_const: false,
                is_thread_local: false,
                linkage: crate::hir::Linkage::Internal,
                visibility: crate::hir::Visibility::Default,
            },
        );
        module.functions.insert(func.id, func);

        let mut backing = 17i64;
        let mut interp = HirInterpreter::new();
        interp.bind_global_slot(global_id, (&mut backing as *mut i64).cast());
        assert_eq!(
            interp.call(&module, "read_global", vec![]).unwrap(),
            ZyntaxValue::Int(17)
        );
        backing = 29;
        assert_eq!(
            interp.call(&module, "read_global", vec![]).unwrap(),
            ZyntaxValue::Int(29)
        );
        let mut replacement = 41i64;
        interp.bind_global_slot(global_id, (&mut replacement as *mut i64).cast());
        assert_eq!(
            interp.call(&module, "read_global", vec![]).unwrap(),
            ZyntaxValue::Int(41)
        );
        assert_eq!(backing, 29);
    }

    /// `def add(a: i64, b: i64): i64 { return a + b }`
    #[test]
    fn bc_runs_simple_add() {
        let mut func = mk_fn("add", vec![HirType::I64, HirType::I64], vec![HirType::I64]);
        let p0 = func.signature.params[0].id;
        let p1 = func.signature.params[1].id;
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

        let mut interp = HirInterpreter::new();
        let result = interp
            .call(
                &module,
                "add",
                vec![ZyntaxValue::Int(10), ZyntaxValue::Int(32)],
            )
            .expect("call should succeed");
        assert!(matches!(result, ZyntaxValue::Int(42)));
    }

    /// SIMD scalarization: splat → element-wise add → insert → extract →
    /// horizontal reduce, all inline (no FFI). `c = splat(2)+splat(3)` =
    /// `[5,5,5,5]`; insert lane 1 := 10 → `[5,10,5,5]`; reduce_add = 25;
    /// extract lane 1 = 10; return 25 + 10 = 35.
    #[test]
    fn bc_runs_vector_splat_binop_reduce() {
        let mut func = mk_fn("vsum", vec![], vec![HirType::F32]);
        let vec_ty = HirType::Vector(Box::new(HirType::F32), 4);
        let c2 = add_value(
            &mut func,
            HirType::F32,
            HirValueKind::Constant(HirConstant::F32(2.0)),
        );
        let c3 = add_value(
            &mut func,
            HirType::F32,
            HirValueKind::Constant(HirConstant::F32(3.0)),
        );
        let c10 = add_value(
            &mut func,
            HirType::F32,
            HirValueKind::Constant(HirConstant::F32(10.0)),
        );
        let a = add_value(&mut func, vec_ty.clone(), HirValueKind::Instruction);
        let b = add_value(&mut func, vec_ty.clone(), HirValueKind::Instruction);
        let c = add_value(&mut func, vec_ty.clone(), HirValueKind::Instruction);
        let c_ins = add_value(&mut func, vec_ty.clone(), HirValueKind::Instruction);
        let e = add_value(&mut func, HirType::F32, HirValueKind::Instruction);
        let r = add_value(&mut func, HirType::F32, HirValueKind::Instruction);
        let out = add_value(&mut func, HirType::F32, HirValueKind::Instruction);

        let entry_id = func.entry_block;
        let entry = func.blocks.get_mut(&entry_id).unwrap();
        entry.instructions.push(HirInstruction::VectorSplat {
            result: a,
            ty: vec_ty.clone(),
            scalar: c2,
        });
        entry.instructions.push(HirInstruction::VectorSplat {
            result: b,
            ty: vec_ty.clone(),
            scalar: c3,
        });
        entry.instructions.push(HirInstruction::Binary {
            result: c,
            op: BinaryOp::FAdd,
            ty: vec_ty.clone(),
            left: a,
            right: b,
        });
        entry.instructions.push(HirInstruction::VectorInsertLane {
            result: c_ins,
            ty: vec_ty.clone(),
            vector: c,
            scalar: c10,
            lane: 1,
        });
        entry.instructions.push(HirInstruction::VectorExtractLane {
            result: e,
            ty: HirType::F32,
            vector: c_ins,
            lane: 1,
        });
        entry
            .instructions
            .push(HirInstruction::VectorHorizontalReduce {
                result: r,
                ty: HirType::F32,
                vector: c_ins,
                op: BinaryOp::FAdd,
            });
        entry.instructions.push(HirInstruction::Binary {
            result: out,
            op: BinaryOp::FAdd,
            ty: HirType::F32,
            left: r,
            right: e,
        });
        entry.terminator = HirTerminator::Return { values: vec![out] };

        let mut module = HirModule::new(InternedString::new_global("test"));
        module.functions.insert(func.id, func);
        let mut interp = HirInterpreter::new();
        let result = interp.call(&module, "vsum", vec![]).expect("call ok");
        // The trailing scalar `r + e` is an `f32` add — it must stay `F32`
        // (not widen to `Float`), matching the native backends.
        match result {
            ZyntaxValue::F32(x) => assert!((x - 35.0).abs() < 1e-6, "got {x}"),
            other => panic!("expected F32(35.0), got {other:?}"),
        }
    }

    /// Scalar `f32` arithmetic must stay `f32`-typed and `f32`-precise, not
    /// funnel through `f64`/`Float`. `def f(a,b): f32 { return a*b - a }`
    /// with a=0.1, b=0.2 → the `F32`-rounded result, which differs from the
    /// `f64` value in the low bits — asserting the exact `f32` bit pattern
    /// pins that the op computed in `f32`.
    #[test]
    fn bc_scalar_f32_stays_f32() {
        let mut func = mk_fn("f", vec![HirType::F32, HirType::F32], vec![HirType::F32]);
        let a = func.signature.params[0].id;
        let b = func.signature.params[1].id;
        for (id, idx) in [(a, 0u32), (b, 1u32)] {
            func.values.insert(
                id,
                HirValue {
                    id,
                    ty: HirType::F32,
                    kind: HirValueKind::Parameter(idx),
                    uses: HashSet::new(),
                    span: None,
                },
            );
        }
        let prod = add_value(&mut func, HirType::F32, HirValueKind::Instruction);
        let out = add_value(&mut func, HirType::F32, HirValueKind::Instruction);
        let entry_id = func.entry_block;
        let entry = func.blocks.get_mut(&entry_id).unwrap();
        entry.instructions.push(HirInstruction::Binary {
            result: prod,
            op: BinaryOp::FMul,
            ty: HirType::F32,
            left: a,
            right: b,
        });
        entry.instructions.push(HirInstruction::Binary {
            result: out,
            op: BinaryOp::FSub,
            ty: HirType::F32,
            left: prod,
            right: a,
        });
        entry.terminator = HirTerminator::Return { values: vec![out] };

        let mut module = HirModule::new(InternedString::new_global("test"));
        module.functions.insert(func.id, func);
        let mut interp = HirInterpreter::new();
        let result = interp
            .call(
                &module,
                "f",
                vec![ZyntaxValue::F32(0.1), ZyntaxValue::F32(0.2)],
            )
            .expect("call ok");
        let expect: f32 = 0.1f32 * 0.2f32 - 0.1f32;
        match result {
            ZyntaxValue::F32(x) => assert_eq!(x.to_bits(), expect.to_bits(), "got {x}"),
            other => panic!("expected F32, got {other:?}"),
        }
    }

    /// SIMD memory roundtrip: `store` a splatted vector into a stack buffer,
    /// `load` it back, and reduce. Exercises `VectorStore`/`VectorLoad`'s
    /// per-lane `write_typed`/`read_typed`. splat(7) → store → load →
    /// reduce_add = 28.
    #[test]
    fn bc_runs_vector_load_store_roundtrip() {
        let mut func = mk_fn("vld", vec![], vec![HirType::F32]);
        let vec_ty = HirType::Vector(Box::new(HirType::F32), 4);
        let buf = add_value(
            &mut func,
            HirType::Ptr(Box::new(HirType::F32)),
            HirValueKind::Instruction,
        );
        let c7 = add_value(
            &mut func,
            HirType::F32,
            HirValueKind::Constant(HirConstant::F32(7.0)),
        );
        let v = add_value(&mut func, vec_ty.clone(), HirValueKind::Instruction);
        let w = add_value(&mut func, vec_ty.clone(), HirValueKind::Instruction);
        let r = add_value(&mut func, HirType::F32, HirValueKind::Instruction);

        let entry_id = func.entry_block;
        let entry = func.blocks.get_mut(&entry_id).unwrap();
        entry.instructions.push(HirInstruction::Alloca {
            result: buf,
            ty: vec_ty.clone(),
            count: None,
            align: 16,
        });
        entry.instructions.push(HirInstruction::VectorSplat {
            result: v,
            ty: vec_ty.clone(),
            scalar: c7,
        });
        entry.instructions.push(HirInstruction::VectorStore {
            value: v,
            ptr: buf,
            align: 16,
        });
        entry.instructions.push(HirInstruction::VectorLoad {
            result: w,
            ty: vec_ty.clone(),
            ptr: buf,
            align: 16,
        });
        entry
            .instructions
            .push(HirInstruction::VectorHorizontalReduce {
                result: r,
                ty: HirType::F32,
                vector: w,
                op: BinaryOp::FAdd,
            });
        entry.terminator = HirTerminator::Return { values: vec![r] };

        let mut module = HirModule::new(InternedString::new_global("test"));
        module.functions.insert(func.id, func);
        let mut interp = HirInterpreter::new();
        let result = interp.call(&module, "vld", vec![]).expect("call ok");
        let got = value_to_f64(&result).unwrap_or(f64::NAN);
        assert!((got - 28.0).abs() < 1e-6, "got {result:?}");
    }

    /// Rayzor-parity ops #1/#2: `VectorUnaryOp` (Sqrt) + `VectorMinMax` (Max).
    /// sqrt(splat 9.0) = [3,3,3,3]; max(that, splat 5.0) = [5,5,5,5];
    /// reduce_add = 20.
    #[test]
    fn bc_runs_vector_unary_and_minmax() {
        let mut func = mk_fn("vum", vec![], vec![HirType::F32]);
        let vt = HirType::Vector(Box::new(HirType::F32), 4);
        let c9 = add_value(
            &mut func,
            HirType::F32,
            HirValueKind::Constant(HirConstant::F32(9.0)),
        );
        let c5 = add_value(
            &mut func,
            HirType::F32,
            HirValueKind::Constant(HirConstant::F32(5.0)),
        );
        let v9 = add_value(&mut func, vt.clone(), HirValueKind::Instruction);
        let sq = add_value(&mut func, vt.clone(), HirValueKind::Instruction);
        let v5 = add_value(&mut func, vt.clone(), HirValueKind::Instruction);
        let mx = add_value(&mut func, vt.clone(), HirValueKind::Instruction);
        let r = add_value(&mut func, HirType::F32, HirValueKind::Instruction);
        let e = func.entry_block;
        let blk = func.blocks.get_mut(&e).unwrap();
        blk.instructions.push(HirInstruction::VectorSplat {
            result: v9,
            ty: vt.clone(),
            scalar: c9,
        });
        blk.instructions.push(HirInstruction::VectorUnaryOp {
            result: sq,
            ty: vt.clone(),
            op: VectorUnaryKind::Sqrt,
            operand: v9,
        });
        blk.instructions.push(HirInstruction::VectorSplat {
            result: v5,
            ty: vt.clone(),
            scalar: c5,
        });
        blk.instructions.push(HirInstruction::VectorMinMax {
            result: mx,
            ty: vt.clone(),
            op: VectorMinMaxKind::Max,
            left: sq,
            right: v5,
        });
        blk.instructions
            .push(HirInstruction::VectorHorizontalReduce {
                result: r,
                ty: HirType::F32,
                vector: mx,
                op: BinaryOp::FAdd,
            });
        blk.terminator = HirTerminator::Return { values: vec![r] };
        let mut module = HirModule::new(InternedString::new_global("test"));
        module.functions.insert(func.id, func);
        let mut interp = HirInterpreter::new();
        let got = value_to_f64(&interp.call(&module, "vum", vec![]).expect("ok")).unwrap();
        assert!((got - 20.0).abs() < 1e-6, "got {got}");
    }

    /// Rayzor-parity op #3: fused widening `VectorDot` (the SDOT primitive).
    /// dot(acc=0:i32x4, a=splat 2:i8x16, b=splat 3:i8x16) → each output lane
    /// sums 4 products of 2*3=6 → [24,24,24,24]; reduce_add = 96.
    #[test]
    fn bc_runs_vector_dot() {
        let mut func = mk_fn("vdot", vec![], vec![HirType::I32]);
        let i32x4 = HirType::Vector(Box::new(HirType::I32), 4);
        let i8x16 = HirType::Vector(Box::new(HirType::I8), 16);
        let c0 = add_value(
            &mut func,
            HirType::I32,
            HirValueKind::Constant(HirConstant::I32(0)),
        );
        let c2 = add_value(
            &mut func,
            HirType::I8,
            HirValueKind::Constant(HirConstant::I8(2)),
        );
        let c3 = add_value(
            &mut func,
            HirType::I8,
            HirValueKind::Constant(HirConstant::I8(3)),
        );
        let acc = add_value(&mut func, i32x4.clone(), HirValueKind::Instruction);
        let a = add_value(&mut func, i8x16.clone(), HirValueKind::Instruction);
        let b = add_value(&mut func, i8x16.clone(), HirValueKind::Instruction);
        let d = add_value(&mut func, i32x4.clone(), HirValueKind::Instruction);
        let r = add_value(&mut func, HirType::I32, HirValueKind::Instruction);
        let e = func.entry_block;
        let blk = func.blocks.get_mut(&e).unwrap();
        blk.instructions.push(HirInstruction::VectorSplat {
            result: acc,
            ty: i32x4.clone(),
            scalar: c0,
        });
        blk.instructions.push(HirInstruction::VectorSplat {
            result: a,
            ty: i8x16.clone(),
            scalar: c2,
        });
        blk.instructions.push(HirInstruction::VectorSplat {
            result: b,
            ty: i8x16.clone(),
            scalar: c3,
        });
        blk.instructions.push(HirInstruction::VectorDot {
            result: d,
            acc,
            a,
            b,
            rhs_i7: false,
            rhs_unsigned: false,
        });
        blk.instructions
            .push(HirInstruction::VectorHorizontalReduce {
                result: r,
                ty: HirType::I32,
                vector: d,
                op: BinaryOp::Add,
            });
        blk.terminator = HirTerminator::Return { values: vec![r] };
        let mut module = HirModule::new(InternedString::new_global("test"));
        module.functions.insert(func.id, func);
        let mut interp = HirInterpreter::new();
        let got = value_to_i64(&interp.call(&module, "vdot", vec![]).expect("ok")).unwrap();
        assert_eq!(got, 96, "got {got}");
    }

    /// `def main(): i64 { if (true) return 1 else return 0 }`
    #[test]
    fn bc_runs_cond_branch() {
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
        let mut interp = HirInterpreter::new();
        let result = interp.call(&module, "main", vec![]).unwrap();
        assert!(matches!(result, ZyntaxValue::Int(1)));
    }

    /// `def loop_to_ten(): i64 { let i = 0; while i < 10 { i += 1 }; i }`
    /// — exercises phi nodes (i carries across iterations).
    #[test]
    fn bc_phi_loop() {
        let mut func = mk_fn("loop_to_ten", vec![], vec![HirType::I64]);
        let zero = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(0)),
        );
        let one = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(1)),
        );
        let ten = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(10)),
        );

        let entry_id = func.entry_block;
        let header_id = HirId::new();
        let body_id = HirId::new();
        let exit_id = HirId::new();

        // phi result for `i`
        let i_phi = add_value(&mut func, HirType::I64, HirValueKind::Instruction);
        // i + 1 result
        let i_next = add_value(&mut func, HirType::I64, HirValueKind::Instruction);
        // i < 10 result
        let cmp = add_value(&mut func, HirType::Bool, HirValueKind::Instruction);

        // Entry → header (no instructions, just jump).
        let entry = func.blocks.get_mut(&entry_id).unwrap();
        entry.terminator = HirTerminator::Branch { target: header_id };

        // Header: phi i = [zero from entry, i_next from body]; cmp = i < 10; cond-branch.
        let header = HirBlock {
            id: header_id,
            label: None,
            phis: vec![crate::hir::HirPhi {
                result: i_phi,
                ty: HirType::I64,
                incoming: vec![(zero, entry_id), (i_next, body_id)],
            }],
            instructions: vec![HirInstruction::Binary {
                result: cmp,
                op: BinaryOp::Lt,
                ty: HirType::I64,
                left: i_phi,
                right: ten,
            }],
            terminator: HirTerminator::CondBranch {
                condition: cmp,
                true_target: body_id,
                false_target: exit_id,
            },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        };
        func.blocks.insert(header_id, header);

        // Body: i_next = i + 1; branch back to header.
        let body = HirBlock {
            id: body_id,
            label: None,
            phis: vec![],
            instructions: vec![HirInstruction::Binary {
                result: i_next,
                op: BinaryOp::Add,
                ty: HirType::I64,
                left: i_phi,
                right: one,
            }],
            terminator: HirTerminator::Branch { target: header_id },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        };
        func.blocks.insert(body_id, body);

        // Exit: return i_phi.
        let exit = HirBlock {
            id: exit_id,
            label: None,
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Return {
                values: vec![i_phi],
            },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        };
        func.blocks.insert(exit_id, exit);

        let mut module = HirModule::new(InternedString::new_global("test"));
        let func_id = func.id;
        module.functions.insert(func_id, func);
        let mut interp = HirInterpreter::new();
        let result = interp.call(&module, "loop_to_ten", vec![]).unwrap();
        assert!(matches!(result, ZyntaxValue::Int(10)), "got {:?}", result);
    }

    /// Profile counters increment per call.
    #[test]
    fn bc_records_profile_samples() {
        let mut func = mk_fn("noop", vec![], vec![HirType::Void]);
        let entry_id = func.entry_block;
        func.blocks.get_mut(&entry_id).unwrap().terminator =
            HirTerminator::Return { values: vec![] };
        let mut module = HirModule::new(InternedString::new_global("test"));
        let func_id = func.id;
        module.functions.insert(func_id, func);
        let mut interp = HirInterpreter::new();
        for _ in 0..3 {
            interp.call(&module, "noop", vec![]).unwrap();
        }
        let sample = interp.profile.get(&func_id).copied().unwrap_or_default();
        assert_eq!(sample.call_count, 3);
    }
}
