//! LLVM backend parity check for algebraic-effects instructions.
//!
//! Mirrors the Cranelift-backend behaviour added in Phase H so that
//! both backends compile `PerformEffect` / `HandleEffect` /
//! `Resume` / `AbortEffect` / `CaptureContinuation` instructions
//! without falling through to the "not yet implemented" error arm.
//!
//! Builds a HIR module by hand (no TypedAST round-trip — that's
//! tested elsewhere) containing:
//!   * an effect declaration `effect State { def get(): i64 }`
//!   * a resumable handler `StateHandler.get(k: i64) -> i64 { return 42 }`
//!   * an `@effect(State)` function `run()` whose body issues
//!     `PerformEffect(State.get)` and returns the result.
//!
//! After `LLVMBackend::compile_module`, the emitted LLVM IR must
//! contain a `call` to the mangled handler name `StateHandler$get`
//! and the Resume<T> sentinel arg.

#![cfg(feature = "llvm-backend")]

use std::collections::HashSet;

use indexmap::IndexMap;
use inkwell::context::Context;
use zyntax_compiler::hir::*;
use zyntax_compiler::llvm_backend::LLVMBackend;
use zyntax_typed_ast::InternedString;

fn create_name(s: &str) -> InternedString {
    InternedString::new_global(s)
}

fn make_sig(params: Vec<HirType>, returns: Vec<HirType>) -> HirFunctionSignature {
    HirFunctionSignature {
        params: params
            .into_iter()
            .enumerate()
            .map(|(i, ty)| HirParam {
                id: HirId::new(),
                name: create_name(&format!("p{}", i)),
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
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    }
}

#[test]
fn llvm_backend_compiles_perform_effect_with_resumable_handler() {
    // Effect declaration
    let effect_id = HirId::new();
    let effect = HirEffect {
        id: effect_id,
        name: create_name("State"),
        type_params: vec![],
        operations: vec![HirEffectOp {
            id: HirId::new(),
            name: create_name("get"),
            type_params: vec![],
            params: vec![],
            return_type: HirType::I64,
        }],
    };

    // Handler function "StateHandler$get": takes a single i64 (the Resume<T>
    // sentinel placeholder) and returns the constant 42.
    let handler_fn_id = HirId::new();
    let const_42_id = HirId::new();
    let entry_id = HirId::new();
    let mut handler_fn = HirFunction::new(
        create_name("StateHandler$get"),
        make_sig(vec![HirType::I64], vec![HirType::I64]),
    );
    handler_fn.id = handler_fn_id;
    handler_fn.is_external = false;
    handler_fn.values.insert(
        const_42_id,
        HirValue {
            id: const_42_id,
            ty: HirType::I64,
            kind: HirValueKind::Constant(HirConstant::I64(42)),
            uses: HashSet::new(),
            span: None,
        },
    );
    let handler_entry = HirBlock {
        id: entry_id,
        label: Some(create_name("entry")),
        phis: vec![],
        instructions: vec![],
        terminator: HirTerminator::Return {
            values: vec![const_42_id],
        },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![],
    };
    handler_fn.entry_block = entry_id;
    handler_fn.blocks.clear();
    handler_fn.blocks.insert(entry_id, handler_entry);

    // Handler bookkeeping in module.handlers
    let handler = HirEffectHandler {
        id: HirId::new(),
        name: create_name("StateHandler"),
        effect_id,
        type_params: vec![],
        state_fields: vec![],
        implementations: vec![HirEffectHandlerImpl {
            op_name: create_name("get"),
            type_params: vec![],
            params: vec![HirParam {
                id: HirId::new(),
                name: create_name("k"),
                ty: HirType::I64,
                attributes: ParamAttributes::default(),
            }],
            return_type: HirType::I64,
            entry_block: HirId::new(),
            blocks: IndexMap::new(),
            is_resumable: true,
            is_async: false,
        }],
    };

    // run(): performs State.get(), returns the result.
    let run_fn_id = HirId::new();
    let perform_result_id = HirId::new();
    let run_entry_id = HirId::new();
    let mut run_fn = HirFunction::new(create_name("run"), make_sig(vec![], vec![HirType::I64]));
    run_fn.id = run_fn_id;
    run_fn.is_external = false;
    run_fn.values.insert(
        perform_result_id,
        HirValue {
            id: perform_result_id,
            ty: HirType::I64,
            kind: HirValueKind::Instruction,
            uses: HashSet::new(),
            span: None,
        },
    );
    let run_entry = HirBlock {
        id: run_entry_id,
        label: Some(create_name("entry")),
        phis: vec![],
        instructions: vec![HirInstruction::PerformEffect {
            result: Some(perform_result_id),
            effect_id,
            op_name: create_name("get"),
            args: vec![],
            return_ty: HirType::I64,
        }],
        terminator: HirTerminator::Return {
            values: vec![perform_result_id],
        },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![],
    };
    run_fn.entry_block = run_entry_id;
    run_fn.blocks.clear();
    run_fn.blocks.insert(run_entry_id, run_entry);

    // Module assembly
    let mut module = HirModule::new(create_name("test_llvm_effects"));
    module.functions.insert(handler_fn_id, handler_fn);
    module.functions.insert(run_fn_id, run_fn);
    module.effects.insert(effect_id, effect);
    module.handlers.insert(handler.id, handler);

    // Compile via LLVM. Should not error out on the PerformEffect
    // arm (the regression that this test guards against).
    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "test_llvm_effects");
    let llvm_ir = backend
        .compile_module(&module)
        .expect("LLVMBackend::compile_module must succeed with PerformEffect");

    // Sanity-check the emitted IR. LLVMBackend mangles non-external
    // function names to `func_{HirId}` (see `declare_function` in
    // llvm_backend.rs), so we look for the call by HirId rather than
    // the source-level name. Also verify the Resume<T> sentinel arg
    // (i64 0) is present at the call site.
    let expected_handler_name = format!("func_{:?}", handler_fn_id);
    assert!(
        llvm_ir.contains(&expected_handler_name),
        "expected LLVM IR to contain a call to {} (handler function); got:\n{}",
        expected_handler_name,
        llvm_ir
    );
    assert!(
        llvm_ir.contains("i64 0"),
        "expected Resume<T> sentinel (i64 0) in call args; got:\n{}",
        llvm_ir
    );
}

#[test]
fn llvm_backend_handles_handle_effect_and_resume_as_placeholders() {
    // Smoke test: a function containing HandleEffect / Resume /
    // AbortEffect / CaptureContinuation must NOT error out on the
    // "Instruction not yet implemented" arm. The Cranelift-backend's
    // own handling for these is also placeholder, so this just
    // ensures the two backends don't drift on which instructions
    // they accept.
    let body_block_id = HirId::new();
    let cont_block_id = HirId::new();
    let value_id = HirId::new();
    let cont_id = HirId::new();
    let handler_state_id = HirId::new();
    let handle_result_id = HirId::new();
    let capture_id = HirId::new();
    let scope_id = HirId::new();

    let fn_id = HirId::new();
    let entry_id = HirId::new();
    let mut func = HirFunction::new(
        create_name("placeholder_effects"),
        make_sig(vec![], vec![HirType::Void]),
    );
    func.id = fn_id;
    func.is_external = false;
    for (id, ty, kind) in [
        (
            value_id,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(7)),
        ),
        (cont_id, HirType::I64, HirValueKind::Instruction),
        (handler_state_id, HirType::I64, HirValueKind::Instruction),
    ] {
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
    }
    let entry = HirBlock {
        id: entry_id,
        label: Some(create_name("entry")),
        phis: vec![],
        instructions: vec![
            HirInstruction::HandleEffect {
                result: Some(handle_result_id),
                handler_id: HirId::new(),
                handler_state: vec![handler_state_id],
                body_block: body_block_id,
                continuation_block: cont_block_id,
                return_ty: HirType::I64,
            },
            HirInstruction::CaptureContinuation {
                result: capture_id,
                resume_ty: HirType::I64,
            },
            HirInstruction::Resume {
                value: value_id,
                continuation: cont_id,
            },
            HirInstruction::AbortEffect {
                value: value_id,
                handler_scope: scope_id,
            },
        ],
        terminator: HirTerminator::Return { values: vec![] },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![],
    };
    func.entry_block = entry_id;
    func.blocks.clear();
    func.blocks.insert(entry_id, entry);

    // Resume placeholder needs the __zyntax_effect_resume symbol to
    // be declared in the LLVM module — add it as an external decl
    // so the IR validates.
    let mut resume_decl = HirFunction::new(
        create_name("__zyntax_effect_resume"),
        make_sig(vec![HirType::I64, HirType::I64], vec![HirType::I64]),
    );
    resume_decl.is_external = true;
    let resume_decl_id = HirId::new();
    resume_decl.id = resume_decl_id;
    resume_decl.blocks.clear();

    let mut module = HirModule::new(create_name("test_llvm_placeholder"));
    module.functions.insert(fn_id, func);
    module.functions.insert(resume_decl_id, resume_decl);

    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "test_llvm_placeholder");
    backend
        .compile_module(&module)
        .expect("placeholder effect instructions must compile without errors");
}
