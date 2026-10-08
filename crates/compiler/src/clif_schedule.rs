//! Reduce floating-point values live across calls after egraph elaboration.

use cranelift_codegen::{
    CodegenResult, CompiledCode, Context,
    entity::SecondaryMap,
    ir::{Function, Opcode, ProgramPoint, ValueDef, types},
    isa::TargetIsa,
};
use std::sync::OnceLock;

fn arithmetic(op: Opcode) -> bool {
    matches!(
        op,
        Opcode::Fadd | Opcode::Fsub | Opcode::Fmul | Opcode::Fdiv | Opcode::Fma
    )
}

fn schedule(func: &mut Function) -> usize {
    let mut has_call = false;
    let mut has_arithmetic = false;
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            let op = func.dfg.insts[inst].opcode();
            has_call |= op.is_call();
            has_arithmetic |= arithmetic(op);
        }
    }
    if !has_call || !has_arithmetic {
        return 0;
    }

    // Count all uses, including successor arguments and uses in other blocks.
    let mut uses = SecondaryMap::with_default(0usize);
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            for value in func.dfg.inst_values(inst) {
                uses[func.dfg.resolve_aliases(value)] += 1;
            }
        }
    }
    let mut moved = 0;
    let mut block = func.layout.entry_block();
    while let Some(b) = block {
        let mut latest_call = None;
        let mut next = func.layout.first_inst(b);
        while let Some(inst) = next {
            next = func.layout.next_inst(inst);
            let op = func.dfg.insts[inst].opcode();
            if op.is_call() {
                latest_call = Some(inst);
                continue;
            }
            if !arithmetic(op) {
                continue;
            }
            let ty = func.dfg.value_type(func.dfg.first_result(inst));
            if ty != types::F32 && ty != types::F64 {
                continue;
            }
            let Some(call) = latest_call else {
                continue;
            };
            let args: smallvec::SmallVec<[_; 3]> = func
                .dfg
                .inst_args(inst)
                .iter()
                .map(|&arg| func.dfg.resolve_aliases(arg))
                .collect();
            let mut dying = 0;
            let mut available_after = ProgramPoint::Block(b);
            let mut available = true;
            for (index, &arg) in args.iter().enumerate() {
                let definition = match func.dfg.value_def(arg) {
                    ValueDef::Param(parent, _) if parent == b => ProgramPoint::Block(b),
                    ValueDef::Result(def, _) => {
                        // Constants can be rematerialized without a spill.
                        if func.layout.inst_block(def) == Some(b)
                            && !matches!(
                                func.dfg.insts[def].opcode(),
                                Opcode::F32const | Opcode::F64const
                            )
                        {
                            ProgramPoint::Inst(def)
                        } else {
                            available = false;
                            break;
                        }
                    }
                    _ => {
                        available = false;
                        break;
                    }
                };
                if func.layout.pp_cmp(available_after, definition).is_lt() {
                    available_after = definition;
                }
                if !args[..index].contains(&arg) && uses[arg] == 1 {
                    dying += 1;
                }
            }
            // Only pure FP arithmetic moves. Its one result replaces at least
            // two dying operands; loads, traps and control flow stay in place.
            if available && dying > 1 && func.layout.pp_cmp(available_after, call).is_lt() {
                let before = match available_after {
                    ProgramPoint::Block(block) => func.layout.first_inst(block).unwrap(),
                    ProgramPoint::Inst(def) => func.layout.next_inst(def).unwrap(),
                };
                func.layout.remove_inst(inst);
                func.layout.insert_inst(inst, before);
                moved += 1;
            }
        }
        block = func.layout.next_block(b);
    }
    moved
}

pub(crate) fn compile(ctx: &mut Context, isa: &dyn TargetIsa) -> CodegenResult<CompiledCode> {
    let mut control = Default::default();
    ctx.verify_if(isa)?;
    ctx.optimize(isa, &mut control)?;
    static DISABLED: OnceLock<bool> = OnceLock::new();
    // ZYNTAX_DISABLE_CLIF_CALL_SCHEDULE retains egraph placement; safe.
    if !*DISABLED.get_or_init(|| std::env::var_os("ZYNTAX_DISABLE_CLIF_CALL_SCHEDULE").is_some()) {
        schedule(&mut ctx.func);
        ctx.verify_if(isa)?;
    }
    // Compile the optimized layout directly: Context::compile would run
    // egraph elaboration again and undo the placement.
    let code = isa.compile_function(
        &ctx.func,
        &ctx.domtree,
        &mut Default::default(),
        ctx.want_disasm,
        &mut control,
    )?;
    Ok(code.apply_params(&ctx.func.params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::{
        ir::{self, AbiParam, InstBuilder, MemFlagsData, types},
        settings::{self, Configurable},
    };
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_module::{Linkage, Module};

    fn isa() -> std::sync::Arc<dyn TargetIsa> {
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").unwrap();
        flags.set("enable_verifier", "true").unwrap();
        cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(flags))
            .unwrap()
    }

    #[derive(Clone, Copy)]
    enum Uses {
        OnlySums,
        Shared,
        Successor,
        Duplicate,
        LateLoad,
    }

    fn fixture(isa: &dyn TargetIsa, ty: ir::Type, op: Opcode, uses: Uses) -> Context {
        let mut ctx = Context::new();
        let cc = isa.default_call_conv();
        ctx.func.signature.call_conv = cc;
        ctx.func.signature.params = vec![AbiParam::new(types::I64); 3];
        ctx.func.signature.returns = vec![AbiParam::new(types::I64)];
        let signature = ctx.func.signature.clone();
        let sig = ctx.func.import_signature(signature);
        let name = ctx
            .func
            .declare_imported_user_function(ir::UserExternalName::new(0, 0));
        let callee = ctx.func.import_function(ir::ExtFuncData {
            name: ir::ExternalName::User(name),
            signature: sig,
            colocated: false,
            patchable: false,
        });
        let mut fbc = FunctionBuilderContext::new();
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbc);
        let block = fb.create_block();
        fb.append_block_params_for_function_params(block);
        fb.switch_to_block(block);
        let p = fb.block_params(block)[0];
        let q = fb.block_params(block)[1];
        let out = fb.block_params(block)[2];
        let mut pairs = Vec::new();
        for i in 0..3 {
            let a = fb
                .ins()
                .load(ty, MemFlagsData::new(), p, i * ty.bytes() as i32);
            let b = if matches!(uses, Uses::Duplicate) {
                a
            } else {
                fb.ins()
                    .load(ty, MemFlagsData::new(), q, i * ty.bytes() as i32)
            };
            pairs.push((a, b));
        }
        let call = fb.ins().call(callee, &[p, q, out]);
        let result = fb.inst_results(call)[0];
        for (i, &(a, b)) in pairs.iter().enumerate() {
            let b = if matches!(uses, Uses::LateLoad) {
                fb.ins().load(
                    ty,
                    MemFlagsData::new(),
                    q,
                    (i + 3) as i32 * ty.bytes() as i32,
                )
            } else {
                b
            };
            let sum = match op {
                Opcode::Fadd => fb.ins().fadd(a, b),
                Opcode::Fsub => fb.ins().fsub(a, b),
                Opcode::Fmul => fb.ins().fmul(a, b),
                Opcode::Fdiv => fb.ins().fdiv(a, b),
                _ => unreachable!(),
            };
            fb.ins().store(
                MemFlagsData::new(),
                sum,
                result,
                i as i32 * ty.bytes() as i32,
            );
        }
        if matches!(uses, Uses::Successor) {
            let next = fb.create_block();
            fb.ins().jump(next, &[]);
            fb.switch_to_block(next);
        }
        if matches!(uses, Uses::Shared | Uses::Successor) {
            for (i, &(a, _)) in pairs.iter().enumerate() {
                fb.ins().store(
                    MemFlagsData::new(),
                    a,
                    result,
                    (i + 3) as i32 * ty.bytes() as i32,
                );
            }
        }
        fb.ins().return_(&[result]);
        fb.seal_all_blocks();
        fb.finalize(isa.frontend_config());
        ctx
    }

    #[test]
    fn native_code_spills_results_instead_of_both_operands() {
        let isa = isa();
        for ty in [types::F32, types::F64] {
            let mut ctx = fixture(&*isa, ty, Opcode::Fadd, Uses::OnlySums);
            ctx.optimize(&*isa, &mut Default::default()).unwrap();
            let before = isa
                .compile_function(
                    &ctx.func,
                    &ctx.domtree,
                    &mut Default::default(),
                    true,
                    &mut Default::default(),
                )
                .unwrap();
            assert_eq!(schedule(&mut ctx.func), 3);
            ctx.verify(&*isa).unwrap();
            let after = isa
                .compile_function(
                    &ctx.func,
                    &ctx.domtree,
                    &mut Default::default(),
                    true,
                    &mut Default::default(),
                )
                .unwrap();
            let before = before.vcode.unwrap();
            let after = after.vcode.unwrap();
            let spills = |s: &str| -> usize {
                if cfg!(target_arch = "aarch64") {
                    s.lines()
                        .filter(|l| l.contains("str q") && l.contains("[sp"))
                        .count()
                } else if cfg!(target_arch = "x86_64") {
                    s.lines()
                        .filter(|l| {
                            l.contains("movdqu")
                                && l.contains("%xmm")
                                && l.contains("(%rsp)")
                                && l.find("%xmm") < l.find("(%rsp)")
                        })
                        .count()
                } else {
                    return 0;
                }
            };
            if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
                assert_eq!(spills(&before), 6, "{before}");
                assert_eq!(spills(&after), 3, "{after}");
            }
            assert!(after.len() < before.len());
            assert_eq!(schedule(&mut ctx.func), 0);
        }
    }

    #[test]
    fn shared_inputs_duplicates_and_late_loads_stay_after_the_call() {
        let isa = isa();
        for uses in [
            Uses::Shared,
            Uses::Successor,
            Uses::Duplicate,
            Uses::LateLoad,
        ] {
            let mut ctx = fixture(&*isa, types::F64, Opcode::Fadd, uses);
            ctx.optimize(&*isa, &mut Default::default()).unwrap();
            let before = ctx.func.display().to_string();
            assert_eq!(schedule(&mut ctx.func), 0, "{before}");
            assert_eq!(ctx.func.display().to_string(), before);
            ctx.verify(&*isa).unwrap();
        }
    }

    #[test]
    fn fused_chain_crosses_all_available_calls_in_dependency_order() {
        let isa = isa();
        let mut ctx = Context::new();
        ctx.func.signature.call_conv = isa.default_call_conv();
        ctx.func.signature.params = vec![AbiParam::new(types::F64); 4];
        ctx.func.signature.returns = vec![AbiParam::new(types::F64)];
        let sig = ctx
            .func
            .import_signature(ir::Signature::new(isa.default_call_conv()));
        let name = ctx
            .func
            .declare_imported_user_function(ir::UserExternalName::new(0, 0));
        let callee = ctx.func.import_function(ir::ExtFuncData {
            name: ir::ExternalName::User(name),
            signature: sig,
            colocated: false,
            patchable: false,
        });
        let mut fbc = FunctionBuilderContext::new();
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbc);
        let block = fb.create_block();
        fb.append_block_params_for_function_params(block);
        fb.switch_to_block(block);
        let args = fb.block_params(block).to_vec();
        let call1 = fb.ins().call(callee, &[]);
        let call2 = fb.ins().call(callee, &[]);
        let fused = fb.ins().fma(args[0], args[1], args[2]);
        let sum = fb.ins().fadd(fused, args[3]);
        fb.ins().return_(&[sum]);
        fb.seal_all_blocks();
        fb.finalize(isa.frontend_config());
        ctx.compute_cfg();
        ctx.compute_domtree();
        assert_eq!(schedule(&mut ctx.func), 2);
        ctx.verify(&*isa).unwrap();
        let insts: Vec<_> = ctx.func.layout.block_insts(block).collect();
        assert_eq!(ctx.func.dfg.insts[insts[0]].opcode(), Opcode::Fma);
        assert_eq!(ctx.func.dfg.insts[insts[1]].opcode(), Opcode::Fadd);
        assert_eq!(&insts[2..4], &[call1, call2]);
    }

    #[test]
    fn call_results_and_constants_are_not_available_spill_savings() {
        use cranelift_codegen::cursor::{Cursor, FuncCursor};
        let isa = isa();
        for constant in [false, true] {
            let mut ctx = fixture(&*isa, types::F64, Opcode::Fadd, Uses::OnlySums);
            let block = ctx.func.layout.entry_block().unwrap();
            let call = ctx
                .func
                .layout
                .block_insts(block)
                .find(|&i| ctx.func.dfg.insts[i].opcode().is_call())
                .unwrap();
            let adds: Vec<_> = ctx
                .func
                .layout
                .block_insts(block)
                .filter(|&i| ctx.func.dfg.insts[i].opcode() == Opcode::Fadd)
                .collect();
            if constant {
                for add in adds {
                    let value = FuncCursor::new(&mut ctx.func)
                        .at_inst(call)
                        .ins()
                        .f64const(0.25);
                    ctx.func.dfg.inst_args_mut(add)[0] = value;
                }
            } else {
                // The returned pointer supplies a new load, after the call.
                let ptr = ctx.func.dfg.first_result(call);
                for add in adds {
                    let value = FuncCursor::new(&mut ctx.func).at_inst(add).ins().load(
                        types::F64,
                        MemFlagsData::new(),
                        ptr,
                        0,
                    );
                    ctx.func.dfg.inst_args_mut(add)[0] = value;
                }
            }
            ctx.compute_cfg();
            ctx.compute_domtree();
            assert_eq!(schedule(&mut ctx.func), 0);
            ctx.verify(&*isa).unwrap();
        }
    }

    // The call overwrites both inputs; the generated code must use the snapshots.
    unsafe extern "C" fn overwrite64(p: *mut f64, q: *mut f64, out: *mut f64) -> *mut f64 {
        for i in 0..3 {
            unsafe {
                p.add(i).write(91.0);
                q.add(i).write(37.0);
            }
        }
        out
    }
    unsafe extern "C" fn overwrite32(p: *mut f32, q: *mut f32, out: *mut f32) -> *mut f32 {
        for i in 0..3 {
            unsafe {
                p.add(i).write(91.0);
                q.add(i).write(37.0);
            }
        }
        out
    }

    #[test]
    fn compiled_arithmetic_keeps_ieee_values_across_aliasing_calls() {
        for ty in [types::F32, types::F64] {
            for op in [Opcode::Fadd, Opcode::Fsub, Opcode::Fmul, Opcode::Fdiv] {
                let isa = isa();
                let mut builder = cranelift_jit::JITBuilder::with_isa(
                    isa.clone(),
                    cranelift_module::default_libcall_names(),
                );
                builder.symbol(
                    "overwrite",
                    if ty == types::F64 {
                        overwrite64 as *const u8
                    } else {
                        overwrite32 as *const u8
                    },
                );
                let mut module = cranelift_jit::JITModule::new(builder);
                let mut ctx = fixture(&*isa, ty, op, Uses::OnlySums);
                let callee = module
                    .declare_function("overwrite", Linkage::Import, &ctx.func.signature)
                    .unwrap();
                assert_eq!(callee.as_u32(), 0);
                let id = module
                    .declare_function("arithmetic", Linkage::Export, &ctx.func.signature)
                    .unwrap();
                let compiled = compile(&mut ctx, &*isa).unwrap();
                let relocs: Vec<_> = compiled
                    .buffer
                    .relocs()
                    .iter()
                    .map(|r| cranelift_module::ModuleReloc::from_mach_reloc(r, &ctx.func, id))
                    .collect();
                module
                    .define_function_bytes(
                        id,
                        compiled.buffer.alignment.into(),
                        compiled.code_buffer(),
                        &relocs,
                    )
                    .unwrap();
                module.finalize_definitions().unwrap();
                let ptr = module.get_finalized_function(id);
                let samples = [
                    0.0,
                    -0.0,
                    1.0,
                    -1.0,
                    f64::MIN_POSITIVE,
                    f64::from_bits(1),
                    f64::MAX,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NAN,
                ];
                for &x in &samples {
                    for &y in &samples {
                        for alias in [false, true] {
                            if ty == types::F64 {
                                let expected = match op {
                                    Opcode::Fadd => x + y,
                                    Opcode::Fsub => x - y,
                                    Opcode::Fmul => x * y,
                                    Opcode::Fdiv => x / y,
                                    _ => unreachable!(),
                                };
                                let mut p = [x; 3];
                                let mut q = [y; 3];
                                let mut out = [0.0; 3];
                                let dest = if alias {
                                    p.as_mut_ptr()
                                } else {
                                    out.as_mut_ptr()
                                };
                                // SAFETY: the fixture has this C ABI and every buffer has three elements.
                                let call: unsafe extern "C" fn(
                                    *mut f64,
                                    *mut f64,
                                    *mut f64,
                                )
                                    -> *mut f64 = unsafe { std::mem::transmute(ptr) };
                                assert_eq!(
                                    unsafe { call(p.as_mut_ptr(), q.as_mut_ptr(), dest) },
                                    dest
                                );
                                for got in if alias { p } else { out } {
                                    assert!(
                                        got.to_bits() == expected.to_bits()
                                            || got.is_nan() && expected.is_nan()
                                    );
                                }
                                assert_eq!(q, [37.0; 3]);
                            } else {
                                let x = x as f32;
                                let y = y as f32;
                                let expected = match op {
                                    Opcode::Fadd => x + y,
                                    Opcode::Fsub => x - y,
                                    Opcode::Fmul => x * y,
                                    Opcode::Fdiv => x / y,
                                    _ => unreachable!(),
                                };
                                let mut p = [x; 3];
                                let mut q = [y; 3];
                                let mut out = [0.0; 3];
                                let dest = if alias {
                                    p.as_mut_ptr()
                                } else {
                                    out.as_mut_ptr()
                                };
                                // SAFETY: the fixture has this C ABI and every buffer has three elements.
                                let call: unsafe extern "C" fn(
                                    *mut f32,
                                    *mut f32,
                                    *mut f32,
                                )
                                    -> *mut f32 = unsafe { std::mem::transmute(ptr) };
                                assert_eq!(
                                    unsafe { call(p.as_mut_ptr(), q.as_mut_ptr(), dest) },
                                    dest
                                );
                                for got in if alias { p } else { out } {
                                    assert!(
                                        got.to_bits() == expected.to_bits()
                                            || got.is_nan() && expected.is_nan()
                                    );
                                }
                                assert_eq!(q, [37.0; 3]);
                            }
                        }
                    }
                }
            }
        }
    }
}
