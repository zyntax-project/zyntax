//! # CLIF-Inspired HIR Dump
//!
//! Human-readable text format for dumping the Zyntax HIR (MIR),
//! inspired by Cranelift's CLIF text format. Enables debugging
//! MIR-level issues before they reach Cranelift codegen.
//!
//! ## Format Example
//! ```text
//! function @main() -> void {
//! block0:                                ; entry
//!     %v0: i64 = const 42
//!     %v1: opaque(Tensor) = call @Tensor$zeros(%v0)
//!     call @println(%v1)
//!     return
//! }
//! ```

use crate::hir::*;
use std::collections::HashMap;
use std::fmt::Write;

/// Maps HirIds to human-readable sequential names.
struct IdMapper {
    /// HirId -> value name like "v0", "v1", etc.
    values: HashMap<HirId, String>,
    /// HirId -> block name like "block0", "block1", etc.
    blocks: HashMap<HirId, String>,
    /// HirId -> function name (resolved from module)
    functions: HashMap<HirId, String>,
    next_value: u32,
    next_block: u32,
}

impl IdMapper {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
            blocks: HashMap::new(),
            functions: HashMap::new(),
            next_value: 0,
            next_block: 0,
        }
    }

    fn value(&mut self, id: &HirId) -> String {
        if let Some(name) = self.values.get(id) {
            return name.clone();
        }
        let name = format!("%v{}", self.next_value);
        self.next_value += 1;
        self.values.insert(*id, name.clone());
        name
    }

    fn block(&mut self, id: &HirId) -> String {
        if let Some(name) = self.blocks.get(id) {
            return name.clone();
        }
        let name = format!("block{}", self.next_block);
        self.next_block += 1;
        self.blocks.insert(*id, name.clone());
        name
    }

    fn func_name(&self, id: &HirId) -> String {
        self.functions
            .get(id)
            .cloned()
            .unwrap_or_else(|| format!("fn?({:?})", id))
    }
}

/// Resolve an InternedString to a display string.
fn resolve(s: &zyntax_typed_ast::InternedString) -> String {
    s.resolve_global().unwrap_or_else(|| format!("{}", s))
}

/// Format an HirType to a compact text representation.
fn fmt_type(ty: &HirType) -> String {
    match ty {
        HirType::Void => "void".into(),
        HirType::Bool => "bool".into(),
        HirType::I8 => "i8".into(),
        HirType::I16 => "i16".into(),
        HirType::I32 => "i32".into(),
        HirType::I64 => "i64".into(),
        HirType::I128 => "i128".into(),
        HirType::U8 => "u8".into(),
        HirType::U16 => "u16".into(),
        HirType::U32 => "u32".into(),
        HirType::U64 => "u64".into(),
        HirType::U128 => "u128".into(),
        HirType::F32 => "f32".into(),
        HirType::F64 => "f64".into(),
        HirType::Ptr(inner) => format!("*{}", fmt_type(inner)),
        HirType::Ref {
            pointee, mutable, ..
        } => {
            if *mutable {
                format!("&mut {}", fmt_type(pointee))
            } else {
                format!("&{}", fmt_type(pointee))
            }
        }
        HirType::Array(elem, len) => format!("[{}; {}]", fmt_type(elem), len),
        HirType::Vector(elem, len) => format!("<{} x {}>", len, fmt_type(elem)),
        HirType::Struct(s) => {
            let fields: Vec<String> = s.fields.iter().map(|f| fmt_type(f)).collect();
            let name = s.name.as_ref().map(|n| resolve(n));
            match name {
                Some(n) => format!("%{}{{ {} }}", n, fields.join(", ")),
                None => format!("{{ {} }}", fields.join(", ")),
            }
        }
        HirType::Union(u) => {
            let name = u.name.as_ref().map(|n| resolve(n));
            let variants: Vec<String> = u
                .variants
                .iter()
                .map(|v| format!("{}: {}", resolve(&v.name), fmt_type(&v.ty)))
                .collect();
            match name {
                Some(n) => format!("union %{}{{ {} }}", n, variants.join(" | ")),
                None => format!("union {{ {} }}", variants.join(" | ")),
            }
        }
        HirType::Function(ft) => {
            let params: Vec<String> = ft.params.iter().map(|p| fmt_type(p)).collect();
            let rets: Vec<String> = ft.returns.iter().map(|r| fmt_type(r)).collect();
            let ret = if rets.is_empty() {
                "void".into()
            } else {
                rets.join(", ")
            };
            format!("fn({}) -> {}", params.join(", "), ret)
        }
        HirType::Closure(ct) => {
            let params: Vec<String> = ct
                .function_type
                .params
                .iter()
                .map(|p| fmt_type(p))
                .collect();
            let rets: Vec<String> = ct
                .function_type
                .returns
                .iter()
                .map(|r| fmt_type(r))
                .collect();
            let ret = if rets.is_empty() {
                "void".into()
            } else {
                rets.join(", ")
            };
            format!("closure({}) -> {}", params.join(", "), ret)
        }
        HirType::Opaque(name) => format!("opaque({})", resolve(name)),
        HirType::ConstGeneric(name) => format!("const_generic({})", resolve(name)),
        HirType::Generic {
            base,
            type_args,
            const_args,
        } => {
            let mut args: Vec<String> = type_args.iter().map(|t| fmt_type(t)).collect();
            for c in const_args {
                args.push(fmt_constant(c));
            }
            format!("{}<{}>", fmt_type(base), args.join(", "))
        }
        HirType::TraitObject { trait_id, .. } => format!("dyn trait#{}", trait_id.as_u32()),
        HirType::Interface {
            methods,
            is_structural,
        } => {
            let kind = if *is_structural {
                "structural"
            } else {
                "nominal"
            };
            format!("interface({}, {} methods)", kind, methods.len())
        }
        HirType::Promise(inner) => format!("promise<{}>", fmt_type(inner)),
        HirType::AssociatedType {
            trait_id,
            self_ty,
            name,
        } => {
            format!(
                "<{} as trait#{}>::{}",
                fmt_type(self_ty),
                trait_id.as_u32(),
                resolve(name)
            )
        }
        HirType::Continuation {
            resume_ty,
            result_ty,
        } => {
            format!(
                "continuation({} -> {})",
                fmt_type(resume_ty),
                fmt_type(result_ty)
            )
        }
        HirType::EffectRow { effects, tail } => {
            let effs: Vec<String> = effects.iter().map(|e| resolve(e)).collect();
            match tail {
                Some(t) => format!("effects<{}, ..{}>", effs.join(", "), resolve(t)),
                None => format!("effects<{}>", effs.join(", ")),
            }
        }
    }
}

/// Format an HirConstant to text.
fn fmt_constant(c: &HirConstant) -> String {
    match c {
        HirConstant::Bool(v) => format!("{}", v),
        HirConstant::I8(v) => format!("{}i8", v),
        HirConstant::I16(v) => format!("{}i16", v),
        HirConstant::I32(v) => format!("{}i32", v),
        HirConstant::I64(v) => format!("{}i64", v),
        HirConstant::I128(v) => format!("{}i128", v),
        HirConstant::U8(v) => format!("{}u8", v),
        HirConstant::U16(v) => format!("{}u16", v),
        HirConstant::U32(v) => format!("{}u32", v),
        HirConstant::U64(v) => format!("{}u64", v),
        HirConstant::U128(v) => format!("{}u128", v),
        HirConstant::F32(v) => format!("{}f32", v),
        HirConstant::F64(v) => format!("{}f64", v),
        HirConstant::Null(ty) => format!("null:{}", fmt_type(ty)),
        HirConstant::Array(elems) => {
            let vals: Vec<String> = elems.iter().map(|e| fmt_constant(e)).collect();
            format!("[{}]", vals.join(", "))
        }
        HirConstant::Struct(fields) => {
            let vals: Vec<String> = fields.iter().map(|f| fmt_constant(f)).collect();
            format!("{{ {} }}", vals.join(", "))
        }
        HirConstant::String(s) => format!("\"{}\"", resolve(s)),
        HirConstant::VTable(vt) => {
            format!(
                "vtable(trait#{}, {} methods)",
                vt.trait_id.as_u32(),
                vt.methods.len()
            )
        }
    }
}

fn fmt_binary_op(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Rem => "rem",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
        BinaryOp::Xor => "xor",
        BinaryOp::Shl => "shl",
        BinaryOp::Shr => "shr",
        BinaryOp::Eq => "eq",
        BinaryOp::Ne => "ne",
        BinaryOp::Lt => "lt",
        BinaryOp::Le => "le",
        BinaryOp::Gt => "gt",
        BinaryOp::Ge => "ge",
        BinaryOp::FAdd => "fadd",
        BinaryOp::FSub => "fsub",
        BinaryOp::FMul => "fmul",
        BinaryOp::FDiv => "fdiv",
        BinaryOp::FRem => "frem",
        BinaryOp::FEq => "feq",
        BinaryOp::FNe => "fne",
        BinaryOp::FLt => "flt",
        BinaryOp::FLe => "fle",
        BinaryOp::FGt => "fgt",
        BinaryOp::FGe => "fge",
    }
}

fn fmt_unary_op(op: &UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "neg",
        UnaryOp::Not => "not",
        UnaryOp::FNeg => "fneg",
    }
}

fn fmt_cast_op(op: &CastOp) -> &'static str {
    match op {
        CastOp::Trunc => "trunc",
        CastOp::ZExt => "zext",
        CastOp::SExt => "sext",
        CastOp::FpTrunc => "fptrunc",
        CastOp::FpExt => "fpext",
        CastOp::FpToUi => "fptoui",
        CastOp::FpToSi => "fptosi",
        CastOp::UiToFp => "uitofp",
        CastOp::SiToFp => "sitofp",
        CastOp::PtrToInt => "ptrtoint",
        CastOp::IntToPtr => "inttoptr",
        CastOp::Bitcast => "bitcast",
    }
}

fn fmt_atomic_op(op: &AtomicOp) -> &'static str {
    match op {
        AtomicOp::Load => "atomic.load",
        AtomicOp::Store => "atomic.store",
        AtomicOp::Exchange => "atomic.xchg",
        AtomicOp::Add => "atomic.add",
        AtomicOp::Sub => "atomic.sub",
        AtomicOp::And => "atomic.and",
        AtomicOp::Or => "atomic.or",
        AtomicOp::Xor => "atomic.xor",
        AtomicOp::CompareExchange => "atomic.cmpxchg",
    }
}

fn fmt_atomic_ordering(ord: &AtomicOrdering) -> &'static str {
    match ord {
        AtomicOrdering::Relaxed => "relaxed",
        AtomicOrdering::Acquire => "acquire",
        AtomicOrdering::Release => "release",
        AtomicOrdering::AcqRel => "acqrel",
        AtomicOrdering::SeqCst => "seqcst",
    }
}

fn fmt_intrinsic(i: &Intrinsic) -> &'static str {
    match i {
        Intrinsic::Memcpy => "memcpy",
        Intrinsic::Memset => "memset",
        Intrinsic::Memmove => "memmove",
        Intrinsic::Sqrt => "sqrt",
        Intrinsic::Sin => "sin",
        Intrinsic::Cos => "cos",
        Intrinsic::Pow => "pow",
        Intrinsic::Log => "log",
        Intrinsic::Exp => "exp",
        Intrinsic::Ctpop => "ctpop",
        Intrinsic::Ctlz => "ctlz",
        Intrinsic::Cttz => "cttz",
        Intrinsic::Bswap => "bswap",
        Intrinsic::SizeOf => "sizeof",
        Intrinsic::AlignOf => "alignof",
        Intrinsic::AddWithOverflow => "add_overflow",
        Intrinsic::SubWithOverflow => "sub_overflow",
        Intrinsic::MulWithOverflow => "mul_overflow",
        Intrinsic::Malloc => "malloc",
        Intrinsic::Free => "free",
        Intrinsic::Realloc => "realloc",
        Intrinsic::Drop => "drop",
        Intrinsic::IncRef => "incref",
        Intrinsic::DecRef => "decref",
        Intrinsic::Alloca => "alloca",
        Intrinsic::GCSafepoint => "gc_safepoint",
        Intrinsic::Await => "await",
        Intrinsic::Yield => "yield",
        Intrinsic::Panic => "panic",
        Intrinsic::Abort => "abort",
        Intrinsic::ClosureToZrtl => "closure_to_zrtl",
        Intrinsic::BoxToZrtl => "box_to_zrtl",
        Intrinsic::PrimitiveToBox => "primitive_to_box",
        Intrinsic::TypeTagOf => "type_tag_of",
    }
}

fn fmt_calling_convention(cc: &CallingConvention) -> &'static str {
    match cc {
        CallingConvention::Fast => "fast",
        CallingConvention::C => "ccc",
        CallingConvention::System => "system_v",
        CallingConvention::WebKit => "webkit",
    }
}

/// Format a callable target.
fn fmt_callable(c: &HirCallable, mapper: &IdMapper) -> String {
    match c {
        HirCallable::Function(id) => format!("@{}", mapper.func_name(id)),
        HirCallable::Indirect(id) => {
            let name = mapper
                .values
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("?({:?})", id));
            format!("indirect {}", name)
        }
        HirCallable::Intrinsic(i) => format!("intrinsic.{}", fmt_intrinsic(i)),
        HirCallable::Symbol(s) => format!("sym \"{}\"", s),
    }
}

/// Format an instruction.
fn fmt_instruction(inst: &HirInstruction, mapper: &mut IdMapper) -> String {
    match inst {
        HirInstruction::Binary {
            op,
            result,
            ty,
            left,
            right,
        } => {
            let r = mapper.value(result);
            let l = mapper.value(left);
            let ri = mapper.value(right);
            format!(
                "{}: {} = {} {}, {}",
                r,
                fmt_type(ty),
                fmt_binary_op(op),
                l,
                ri
            )
        }
        HirInstruction::Unary {
            op,
            result,
            ty,
            operand,
        } => {
            let r = mapper.value(result);
            let o = mapper.value(operand);
            format!("{}: {} = {} {}", r, fmt_type(ty), fmt_unary_op(op), o)
        }
        HirInstruction::Alloca {
            result,
            ty,
            count,
            align,
        } => {
            let r = mapper.value(result);
            match count {
                Some(c) => {
                    let cv = mapper.value(c);
                    format!(
                        "{}: *{} = alloca {}, count {}, align {}",
                        r,
                        fmt_type(ty),
                        fmt_type(ty),
                        cv,
                        align
                    )
                }
                None => format!(
                    "{}: *{} = alloca {}, align {}",
                    r,
                    fmt_type(ty),
                    fmt_type(ty),
                    align
                ),
            }
        }
        HirInstruction::Load {
            result,
            ty,
            ptr,
            align,
            volatile,
        } => {
            let r = mapper.value(result);
            let p = mapper.value(ptr);
            let vol = if *volatile { ", volatile" } else { "" };
            format!(
                "{}: {} = load {}, align {}{}",
                r,
                fmt_type(ty),
                p,
                align,
                vol
            )
        }
        HirInstruction::Store {
            value,
            ptr,
            align,
            volatile,
        } => {
            let v = mapper.value(value);
            let p = mapper.value(ptr);
            let vol = if *volatile { ", volatile" } else { "" };
            format!("store {}, {}, align {}{}", v, p, align, vol)
        }
        HirInstruction::GetElementPtr {
            result,
            ty,
            ptr,
            indices,
        } => {
            let r = mapper.value(result);
            let p = mapper.value(ptr);
            let idxs: Vec<String> = indices.iter().map(|i| mapper.value(i)).collect();
            format!("{}: {} = gep {}, [{}]", r, fmt_type(ty), p, idxs.join(", "))
        }
        HirInstruction::Call {
            result,
            callee,
            args,
            type_args,
            const_args,
            is_tail,
        } => {
            let arg_strs: Vec<String> = args.iter().map(|a| mapper.value(a)).collect();
            let callee_str = fmt_callable(callee, mapper);
            let tail = if *is_tail { "tail " } else { "" };
            let mut extras = Vec::new();
            if !type_args.is_empty() {
                let tas: Vec<String> = type_args.iter().map(|t| fmt_type(t)).collect();
                extras.push(format!("<{}>", tas.join(", ")));
            }
            if !const_args.is_empty() {
                let cas: Vec<String> = const_args.iter().map(|c| fmt_constant(c)).collect();
                extras.push(format!("const<{}>", cas.join(", ")));
            }
            let extra = extras.join(" ");
            let extra_sp = if extra.is_empty() {
                String::new()
            } else {
                format!(" {}", extra)
            };
            match result {
                Some(res) => {
                    let r = mapper.value(res);
                    // We don't have a return type field on Call, infer "?" or use callee info
                    format!(
                        "{} = {}call {}{}({})",
                        r,
                        tail,
                        callee_str,
                        extra_sp,
                        arg_strs.join(", ")
                    )
                }
                None => {
                    format!(
                        "{}call {}{}({})",
                        tail,
                        callee_str,
                        extra_sp,
                        arg_strs.join(", ")
                    )
                }
            }
        }
        HirInstruction::IndirectCall {
            result,
            func_ptr,
            args,
            return_ty,
        } => {
            let fp = mapper.value(func_ptr);
            let arg_strs: Vec<String> = args.iter().map(|a| mapper.value(a)).collect();
            match result {
                Some(res) => {
                    let r = mapper.value(res);
                    format!(
                        "{}: {} = call_indirect {}({})",
                        r,
                        fmt_type(return_ty),
                        fp,
                        arg_strs.join(", ")
                    )
                }
                None => {
                    format!("call_indirect {}({})", fp, arg_strs.join(", "))
                }
            }
        }
        HirInstruction::Cast {
            op,
            result,
            ty,
            operand,
        } => {
            let r = mapper.value(result);
            let o = mapper.value(operand);
            format!("{}: {} = {} {}", r, fmt_type(ty), fmt_cast_op(op), o)
        }
        HirInstruction::Select {
            result,
            ty,
            condition,
            true_val,
            false_val,
        } => {
            let r = mapper.value(result);
            let c = mapper.value(condition);
            let t = mapper.value(true_val);
            let f = mapper.value(false_val);
            format!("{}: {} = select {}, {}, {}", r, fmt_type(ty), c, t, f)
        }
        HirInstruction::ExtractValue {
            result,
            ty,
            aggregate,
            indices,
        } => {
            let r = mapper.value(result);
            let a = mapper.value(aggregate);
            let idxs: Vec<String> = indices.iter().map(|i| format!("{}", i)).collect();
            format!(
                "{}: {} = extractvalue {}, [{}]",
                r,
                fmt_type(ty),
                a,
                idxs.join(", ")
            )
        }
        HirInstruction::InsertValue {
            result,
            ty,
            aggregate,
            value,
            indices,
        } => {
            let r = mapper.value(result);
            let a = mapper.value(aggregate);
            let v = mapper.value(value);
            let idxs: Vec<String> = indices.iter().map(|i| format!("{}", i)).collect();
            format!(
                "{}: {} = insertvalue {}, {}, [{}]",
                r,
                fmt_type(ty),
                a,
                v,
                idxs.join(", ")
            )
        }
        HirInstruction::Atomic {
            op,
            result,
            ty,
            ptr,
            value,
            ordering,
        } => {
            let r = mapper.value(result);
            let p = mapper.value(ptr);
            let v_str = match value {
                Some(v) => format!(", {}", mapper.value(v)),
                None => String::new(),
            };
            format!(
                "{}: {} = {} {}{}, {}",
                r,
                fmt_type(ty),
                fmt_atomic_op(op),
                p,
                v_str,
                fmt_atomic_ordering(ordering)
            )
        }
        HirInstruction::Fence { ordering } => {
            format!("fence {}", fmt_atomic_ordering(ordering))
        }
        HirInstruction::CreateUnion {
            result,
            union_ty,
            variant_index,
            value,
        } => {
            let r = mapper.value(result);
            let v = mapper.value(value);
            format!(
                "{}: {} = create_union variant {}, {}",
                r,
                fmt_type(union_ty),
                variant_index,
                v
            )
        }
        HirInstruction::GetUnionDiscriminant { result, union_val } => {
            let r = mapper.value(result);
            let u = mapper.value(union_val);
            format!("{} = get_discriminant {}", r, u)
        }
        HirInstruction::ExtractUnionValue {
            result,
            ty,
            union_val,
            variant_index,
        } => {
            let r = mapper.value(result);
            let u = mapper.value(union_val);
            format!(
                "{}: {} = extract_union_value {}, variant {}",
                r,
                fmt_type(ty),
                u,
                variant_index
            )
        }
        HirInstruction::CreateTraitObject {
            result,
            trait_id,
            data_ptr,
            vtable_id,
        } => {
            let r = mapper.value(result);
            let d = mapper.value(data_ptr);
            let vt = mapper.value(vtable_id);
            format!(
                "{} = create_trait_object trait#{}, data {}, vtable {}",
                r,
                trait_id.as_u32(),
                d,
                vt
            )
        }
        HirInstruction::UpcastTraitObject {
            result,
            sub_trait_object,
            sub_trait_id,
            super_trait_id,
            super_vtable_id,
        } => {
            let r = mapper.value(result);
            let s = mapper.value(sub_trait_object);
            let svt = mapper.value(super_vtable_id);
            format!(
                "{} = upcast_trait_object {}, trait#{} -> trait#{}, vtable {}",
                r,
                s,
                sub_trait_id.as_u32(),
                super_trait_id.as_u32(),
                svt
            )
        }
        HirInstruction::TraitMethodCall {
            result,
            trait_object,
            method_index,
            method_sig,
            args,
            return_ty,
        } => {
            let to = mapper.value(trait_object);
            let arg_strs: Vec<String> = args.iter().map(|a| mapper.value(a)).collect();
            let mname = resolve(&method_sig.name);
            match result {
                Some(res) => {
                    let r = mapper.value(res);
                    format!(
                        "{}: {} = trait_call {}.{}[{}]({})",
                        r,
                        fmt_type(return_ty),
                        to,
                        mname,
                        method_index,
                        arg_strs.join(", ")
                    )
                }
                None => {
                    format!(
                        "trait_call {}.{}[{}]({})",
                        to,
                        mname,
                        method_index,
                        arg_strs.join(", ")
                    )
                }
            }
        }
        HirInstruction::CreateClosure {
            result,
            closure_ty,
            function,
            captures,
        } => {
            let r = mapper.value(result);
            let f = mapper.value(function);
            let caps: Vec<String> = captures.iter().map(|c| mapper.value(c)).collect();
            format!(
                "{}: {} = create_closure {}, captures [{}]",
                r,
                fmt_type(closure_ty),
                f,
                caps.join(", ")
            )
        }
        HirInstruction::CallClosure {
            result,
            closure,
            args,
        } => {
            let cl = mapper.value(closure);
            let arg_strs: Vec<String> = args.iter().map(|a| mapper.value(a)).collect();
            match result {
                Some(res) => {
                    let r = mapper.value(res);
                    format!("{} = call_closure {}({})", r, cl, arg_strs.join(", "))
                }
                None => {
                    format!("call_closure {}({})", cl, arg_strs.join(", "))
                }
            }
        }
        HirInstruction::CreateRef {
            result,
            value,
            lifetime: _,
            mutable,
        } => {
            let r = mapper.value(result);
            let v = mapper.value(value);
            let kind = if *mutable { "&mut" } else { "&" };
            format!("{} = create_ref {} {}", r, kind, v)
        }
        HirInstruction::Deref {
            result,
            ty,
            reference,
        } => {
            let r = mapper.value(result);
            let rf = mapper.value(reference);
            format!("{}: {} = deref {}", r, fmt_type(ty), rf)
        }
        HirInstruction::Move { result, ty, source } => {
            let r = mapper.value(result);
            let s = mapper.value(source);
            format!("{}: {} = move {}", r, fmt_type(ty), s)
        }
        HirInstruction::Copy { result, ty, source } => {
            let r = mapper.value(result);
            let s = mapper.value(source);
            format!("{}: {} = copy {}", r, fmt_type(ty), s)
        }
        HirInstruction::BeginLifetime { .. } => "begin_lifetime".into(),
        HirInstruction::EndLifetime { .. } => "end_lifetime".into(),
        HirInstruction::LifetimeConstraint { .. } => "lifetime_constraint".into(),
        HirInstruction::PerformEffect {
            result,
            effect_id,
            op_name,
            args,
            return_ty,
        } => {
            let eid = mapper.value(effect_id);
            let arg_strs: Vec<String> = args.iter().map(|a| mapper.value(a)).collect();
            let oname = resolve(op_name);
            match result {
                Some(res) => {
                    let r = mapper.value(res);
                    format!(
                        "{}: {} = perform {}.{}({})",
                        r,
                        fmt_type(return_ty),
                        eid,
                        oname,
                        arg_strs.join(", ")
                    )
                }
                None => {
                    format!("perform {}.{}({})", eid, oname, arg_strs.join(", "))
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
            let hid = mapper.value(handler_id);
            let state: Vec<String> = handler_state.iter().map(|s| mapper.value(s)).collect();
            let body = mapper.block(body_block);
            let cont = mapper.block(continuation_block);
            match result {
                Some(res) => {
                    let r = mapper.value(res);
                    format!(
                        "{}: {} = handle_effect {}, state [{}], body {}, cont {}",
                        r,
                        fmt_type(return_ty),
                        hid,
                        state.join(", "),
                        body,
                        cont
                    )
                }
                None => {
                    format!(
                        "handle_effect {}, state [{}], body {}, cont {}",
                        hid,
                        state.join(", "),
                        body,
                        cont
                    )
                }
            }
        }
        HirInstruction::Resume {
            value,
            continuation,
        } => {
            let v = mapper.value(value);
            let c = mapper.value(continuation);
            format!("resume {}, {}", v, c)
        }
        HirInstruction::AbortEffect {
            value,
            handler_scope,
        } => {
            let v = mapper.value(value);
            let h = mapper.value(handler_scope);
            format!("abort_effect {}, scope {}", v, h)
        }
        HirInstruction::CaptureContinuation { result, resume_ty } => {
            let r = mapper.value(result);
            format!(
                "{}: continuation = capture_continuation resume_ty {}",
                r,
                fmt_type(resume_ty)
            )
        }
        // SIMD instructions
        HirInstruction::VectorSplat { result, ty, scalar } => {
            let r = mapper.value(result);
            let s = mapper.value(scalar);
            format!("{}: {} = vector_splat {}", r, fmt_type(ty), s)
        }
        HirInstruction::VectorExtractLane { result, ty, vector, lane } => {
            let r = mapper.value(result);
            let v = mapper.value(vector);
            format!("{}: {} = extract_lane {}, lane {}", r, fmt_type(ty), v, lane)
        }
        HirInstruction::VectorInsertLane { result, ty, vector, scalar, lane } => {
            let r = mapper.value(result);
            let v = mapper.value(vector);
            let s = mapper.value(scalar);
            format!("{}: {} = insert_lane {}, lane {}, {}", r, fmt_type(ty), v, lane, s)
        }
        HirInstruction::VectorHorizontalReduce { result, ty, vector, op } => {
            let r = mapper.value(result);
            let v = mapper.value(vector);
            format!("{}: {} = hreduce.{} {}", r, fmt_type(ty), fmt_binary_op(op), v)
        }
        HirInstruction::VectorLoad { result, ty, ptr, align } => {
            let r = mapper.value(result);
            let p = mapper.value(ptr);
            format!("{}: {} = vload {}, align {}", r, fmt_type(ty), p, align)
        }
        HirInstruction::VectorStore { value, ptr, align } => {
            let v = mapper.value(value);
            let p = mapper.value(ptr);
            format!("vstore {}, {}, align {}", v, p, align)
        }
    }
}

/// Format a terminator instruction.
fn fmt_terminator(term: &HirTerminator, mapper: &mut IdMapper) -> String {
    match term {
        HirTerminator::Return { values } => {
            if values.is_empty() {
                "return".into()
            } else {
                let vals: Vec<String> = values.iter().map(|v| mapper.value(v)).collect();
                format!("return {}", vals.join(", "))
            }
        }
        HirTerminator::Branch { target } => {
            let t = mapper.block(target);
            format!("br {}", t)
        }
        HirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } => {
            let c = mapper.value(condition);
            let tt = mapper.block(true_target);
            let ft = mapper.block(false_target);
            format!("brcond {}, {}, {}", c, tt, ft)
        }
        HirTerminator::Switch {
            value,
            default,
            cases,
        } => {
            let v = mapper.value(value);
            let d = mapper.block(default);
            let case_strs: Vec<String> = cases
                .iter()
                .map(|(c, b)| format!("{} => {}", fmt_constant(c), mapper.block(b)))
                .collect();
            format!("switch {}, default {}, [{}]", v, d, case_strs.join(", "))
        }
        HirTerminator::Unreachable => "unreachable".into(),
        HirTerminator::Invoke {
            callee,
            args,
            normal,
            unwind,
        } => {
            let callee_str = fmt_callable(callee, mapper);
            let arg_strs: Vec<String> = args.iter().map(|a| mapper.value(a)).collect();
            let n = mapper.block(normal);
            let u = mapper.block(unwind);
            format!(
                "invoke {}({}) normal {}, unwind {}",
                callee_str,
                arg_strs.join(", "),
                n,
                u
            )
        }
        HirTerminator::PatternMatch {
            value,
            patterns,
            default,
        } => {
            let v = mapper.value(value);
            let pats: Vec<String> = patterns
                .iter()
                .map(|p| {
                    let target = mapper.block(&p.target);
                    format!("{:?} => {}", p.kind, target)
                })
                .collect();
            let def = match default {
                Some(d) => format!(", default {}", mapper.block(d)),
                None => String::new(),
            };
            format!("match {}, [{}]{}", v, pats.join(", "), def)
        }
    }
}

/// Format a phi node.
fn fmt_phi(phi: &HirPhi, mapper: &mut IdMapper) -> String {
    let r = mapper.value(&phi.result);
    let incoming: Vec<String> = phi
        .incoming
        .iter()
        .map(|(val, blk)| {
            let v = mapper.value(val);
            let b = mapper.block(blk);
            format!("[{}, {}]", v, b)
        })
        .collect();
    format!("{}: {} = phi {}", r, fmt_type(&phi.ty), incoming.join(", "))
}

/// Dump a single HirFunction to a CLIF-inspired text format.
pub fn dump_function(func: &HirFunction, module: &HirModule) -> String {
    let mut mapper = IdMapper::new();
    let mut out = String::new();

    // Pre-populate function name map from module
    for (fid, f) in &module.functions {
        let name = resolve(&f.name);
        mapper.functions.insert(*fid, name);
    }

    // Pre-assign parameter values — use the HirValueKind::Parameter entries from
    // function.values (created by SSA builder), not signature param IDs, since SSA
    // creates fresh HirIds for parameters that may differ from signature.params[i].id.
    {
        let mut param_vals: Vec<(u32, HirId, String)> = Vec::new();
        for (vid, value) in &func.values {
            if let HirValueKind::Parameter(idx) = value.kind {
                let name = func
                    .signature
                    .params
                    .get(idx as usize)
                    .map(|p| resolve(&p.name))
                    .unwrap_or_else(|| format!("p{}", idx));
                param_vals.push((idx, *vid, name));
            }
        }
        param_vals.sort_by_key(|(idx, _, _)| *idx);
        for (_, vid, name) in param_vals {
            mapper.values.insert(vid, format!("%{}", name));
        }
    }

    // Pre-assign block IDs (entry first)
    mapper.block(&func.entry_block);
    for (bid, _) in &func.blocks {
        mapper.block(bid);
    }

    // Function header
    let func_name = resolve(&func.name);
    let params: Vec<String> = func
        .signature
        .params
        .iter()
        .map(|p| {
            let pname = resolve(&p.name);
            format!("{} %{}", fmt_type(&p.ty), pname)
        })
        .collect();
    let returns: Vec<String> = func.signature.returns.iter().map(|r| fmt_type(r)).collect();
    let ret_str = if returns.is_empty() {
        "void".into()
    } else {
        returns.join(", ")
    };

    let ext = if func.is_external { " extern" } else { "" };
    let cc = fmt_calling_convention(&func.calling_convention);

    let _ = writeln!(
        out,
        "function @{}({}) -> {} {} {{{}",
        func_name,
        params.join(", "),
        ret_str,
        cc,
        ext
    );

    // Link name
    if let Some(ref link) = func.link_name {
        let _ = writeln!(out, "    ; link_name: \"{}\"", link);
    }

    // Value declarations: constants, undefs, globals (not parameters or instructions)
    // This makes it visible what implicit values exist in the function
    let mut has_decls = false;
    for (vid, value) in &func.values {
        match &value.kind {
            HirValueKind::Constant(c) => {
                let name = mapper.value(vid);
                let _ = writeln!(
                    out,
                    "    {}: {} = const {}",
                    name,
                    fmt_type(&value.ty),
                    fmt_constant(c)
                );
                has_decls = true;
            }
            HirValueKind::Undef => {
                let name = mapper.value(vid);
                let _ = writeln!(out, "    {}: {} = undef", name, fmt_type(&value.ty));
                has_decls = true;
            }
            HirValueKind::Global(gid) => {
                let name = mapper.value(vid);
                let _ = writeln!(
                    out,
                    "    {}: {} = global @{:?}",
                    name,
                    fmt_type(&value.ty),
                    gid
                );
                has_decls = true;
            }
            _ => {} // Parameters and Instructions shown elsewhere
        }
    }
    if has_decls {
        let _ = writeln!(out);
    }

    // Blocks
    for (bid, block) in &func.blocks {
        let bname = mapper.block(bid);
        let is_entry = *bid == func.entry_block;
        let entry_comment = if is_entry { "  ; entry" } else { "" };

        // Predecessors comment
        let preds: Vec<String> = block.predecessors.iter().map(|p| mapper.block(p)).collect();
        let preds_comment = if preds.is_empty() || is_entry {
            String::new()
        } else {
            format!("  ; preds: {}", preds.join(", "))
        };

        let _ = writeln!(out, "{}:{}{}", bname, entry_comment, preds_comment);

        // Phi nodes
        for phi in &block.phis {
            let _ = writeln!(out, "    {}", fmt_phi(phi, &mut mapper));
        }

        // Instructions
        for inst in &block.instructions {
            let _ = writeln!(out, "    {}", fmt_instruction(inst, &mut mapper));
        }

        // Terminator
        let _ = writeln!(
            out,
            "    {}",
            fmt_terminator(&block.terminator, &mut mapper)
        );
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "}}");
    out
}

/// Dump an entire HirModule.
pub fn dump_module(module: &HirModule) -> String {
    let mut out = String::new();
    let mod_name = resolve(&module.name);
    let _ = writeln!(out, "; module: {}", mod_name);
    let _ = writeln!(out, "; version: {}", module.version);
    let _ = writeln!(out);

    // Globals
    for (_gid, global) in &module.globals {
        let gname = resolve(&global.name);
        let kind = if global.is_const { "const" } else { "global" };
        let init = match &global.initializer {
            Some(c) => format!(" = {}", fmt_constant(c)),
            None => String::new(),
        };
        let _ = writeln!(out, "{} @{}: {}{}", kind, gname, fmt_type(&global.ty), init);
    }
    if !module.globals.is_empty() {
        let _ = writeln!(out);
    }

    // Functions
    for (_fid, func) in &module.functions {
        let _ = write!(out, "{}", dump_function(func, module));
        let _ = writeln!(out);
    }

    out
}
