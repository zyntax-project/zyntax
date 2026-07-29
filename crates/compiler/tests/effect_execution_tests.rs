//! # Effect Execution Tests
//!
//! Does an algebraic effect actually RUN?
//!
//! The three existing effect suites — `effect_compilation_tests`,
//! `effect_emission_tests`, `llvm_effect_parity_tests` — check analysis,
//! handler resolution, and emitted IR shape. None of them JITs the
//! result and calls it, so "the effect pipeline works" has so far meant
//! "the IR looked right", which is a weaker claim than it reads as.
//!
//! These compile a module, take the function pointer, call it, and
//! assert on the returned value. That is the only evidence that a
//! `perform` reaches its handler at run time.
//!
//! Regional dispatch means a perform site lowers to
//! `select(__zyntax_effect_lookup_op(...) != 0, dyn, static)`. Those
//! runtime symbols live in `zyntax_embed`, which this crate cannot
//! depend on, so the harness registers stubs. Returning null is not a
//! cop-out: null is precisely "no handler installed at run time", which
//! selects the STATIC handler — the path being tested.
//!
//! Tier 1 shape, which is what the Cranelift backend implements today:
//! `PerformEffect` lowers to a direct call to a function named
//! `<Handler>$<op>` (see `mangle_handler_op_name`). So the handler body
//! has to exist in the module as an ordinary function under that name —
//! the `HirEffectHandler` entry alone declares the dispatch, it does not
//! supply the code.

use indexmap::IndexMap;
use std::collections::HashSet;
use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::*;
use zyntax_typed_ast::InternedString;

/// A backend with the effect-runtime symbols stubbed out. Without them
/// `finalize_definitions` panics on an unresolved relocation rather than
/// failing the compile, which reads as a crash in Cranelift rather than
/// a missing host dependency.
fn backend_with_effect_runtime() -> CraneliftBackend {
    extern "C" fn lookup_op(_effect_id: u64, _op_index: u64) -> *mut u8 {
        std::ptr::null_mut()
    }
    extern "C" fn lookup_state(_effect_id: u64) -> *mut u8 {
        std::ptr::null_mut()
    }
    extern "C" fn lookup_op_is_async(_effect_id: u64, _op_index: u64) -> i64 {
        0
    }
    CraneliftBackend::with_runtime_symbols(&[
        ("__zyntax_effect_lookup_op", lookup_op as *const u8),
        ("__zyntax_effect_lookup_state", lookup_state as *const u8),
        (
            "__zyntax_effect_lookup_op_is_async",
            lookup_op_is_async as *const u8,
        ),
    ])
    .expect("backend")
}

fn name(s: &str) -> InternedString {
    InternedString::new_global(s)
}

fn empty_module() -> HirModule {
    HirModule {
        id: HirId::new(),
        name: name("effect_exec"),
        functions: IndexMap::new(),
        globals: IndexMap::new(),
        types: IndexMap::new(),
        imports: vec![],
        exports: vec![],
        version: 0,
        dependencies: HashSet::new(),
        effects: IndexMap::new(),
        handlers: IndexMap::new(),
    }
}

fn signature(returns: Vec<HirType>) -> HirFunctionSignature {
    HirFunctionSignature {
        params: vec![],
        returns,
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_async: false,
        is_fiber: false,
        is_variadic: false,
        effects: vec![],
        is_pure: false,
    }
}

/// An effect with one nullary operation returning `ret`.
fn declare_effect(effect_name: &str, op: &str, ret: HirType) -> (HirId, HirEffect) {
    let effect_id = HirId::new();
    (
        effect_id,
        HirEffect {
            id: effect_id,
            name: name(effect_name),
            type_params: vec![],
            operations: vec![HirEffectOp {
                id: HirId::new(),
                name: name(op),
                type_params: vec![],
                params: vec![],
                return_type: ret,
            }],
        },
    )
}

/// A handler declaration for `effect_id`. Non-resumable: the Tier 1
/// path calls it like a plain function, with no continuation argument.
fn declare_handler(
    handler_name: &str,
    effect_id: HirId,
    op: &str,
    ret: HirType,
) -> (HirId, HirEffectHandler) {
    let handler_id = HirId::new();
    let block_id = HirId::new();
    let mut blocks = IndexMap::new();
    blocks.insert(
        block_id,
        HirBlock {
            id: block_id,
            label: None,
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Return { values: vec![] },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        },
    );
    (
        handler_id,
        HirEffectHandler {
            id: handler_id,
            name: name(handler_name),
            effect_id,
            type_params: vec![],
            state_fields: vec![],
            implementations: vec![HirEffectHandlerImpl {
                op_name: name(op),
                type_params: vec![],
                params: vec![],
                return_type: ret,
                entry_block: block_id,
                blocks,
                is_resumable: false,
                is_async: false,
            }],
        },
    )
}

/// The handler's actual code, as an ordinary function under the mangled
/// name the backend calls. Returns `value`.
fn handler_body(mangled: &str, value: i32) -> HirFunction {
    let mut func = HirFunction::new(name(mangled), signature(vec![HirType::I32]));
    func.calling_convention = CallingConvention::C;
    let konst = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(value)));
    let entry = func.entry_block;
    let block = func.blocks.get_mut(&entry).unwrap();
    block.set_terminator(HirTerminator::Return {
        values: vec![konst],
    });
    func
}

/// `fn run() -> i32 { perform <effect>.<op>() }`
fn performing_function(fn_name: &str, effect_id: HirId, op: &str) -> HirFunction {
    let mut func = HirFunction::new(name(fn_name), signature(vec![HirType::I32]));
    func.calling_convention = CallingConvention::C;
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let entry = func.entry_block;
    let block = func.blocks.get_mut(&entry).unwrap();
    block.add_instruction(HirInstruction::PerformEffect {
        result: Some(result),
        effect_id,
        op_name: name(op),
        args: vec![],
        return_ty: HirType::I32,
    });
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });
    func
}

/// The whole question, in one assertion: a `perform` compiled to native
/// code reaches its handler and returns what the handler returned.
#[test]
fn a_performed_operation_runs_its_handler() {
    let mut module = empty_module();

    let (effect_id, effect) = declare_effect("Counter", "bump", HirType::I32);
    module.effects.insert(effect_id, effect);

    let (handler_id, handler) = declare_handler("CounterHandler", effect_id, "bump", HirType::I32);
    module.handlers.insert(handler_id, handler);

    let body = handler_body("CounterHandler$bump", 42);
    module.functions.insert(body.id, body);

    let run = performing_function("run", effect_id, "bump");
    let run_id = run.id;
    module.functions.insert(run_id, run);

    let mut backend = backend_with_effect_runtime();
    backend
        .compile_module(&module)
        .expect("a module with one effect and one handler must compile");
    backend.finalize_definitions().expect("finalize");

    let raw = backend
        .get_function_ptr(run_id)
        .expect("the performing function must be JIT-compiled");
    let f = unsafe { std::mem::transmute::<*const u8, unsafe extern "C" fn() -> i32>(raw) };
    assert_eq!(
        unsafe { f() },
        42,
        "perform must dispatch to the handler and return its value"
    );
}

/// The handler is what supplies the value, not a constant folded into
/// the performing function. Changing only the handler body changes the
/// result — otherwise the test above would pass against a backend that
/// quietly returned a fixed number.
#[test]
fn the_handler_body_is_what_supplies_the_value() {
    fn run_with(handler_returns: i32) -> i32 {
        let mut module = empty_module();
        let (effect_id, effect) = declare_effect("Counter", "bump", HirType::I32);
        module.effects.insert(effect_id, effect);
        let (handler_id, handler) =
            declare_handler("CounterHandler", effect_id, "bump", HirType::I32);
        module.handlers.insert(handler_id, handler);
        let body = handler_body("CounterHandler$bump", handler_returns);
        module.functions.insert(body.id, body);
        let run = performing_function("run", effect_id, "bump");
        let run_id = run.id;
        module.functions.insert(run_id, run);

        let mut backend = backend_with_effect_runtime();
        backend.compile_module(&module).expect("compile");
        backend.finalize_definitions().expect("finalize");
        let raw = backend.get_function_ptr(run_id).expect("fn ptr");
        let f = unsafe { std::mem::transmute::<*const u8, unsafe extern "C" fn() -> i32>(raw) };
        unsafe { f() }
    }

    assert_eq!(run_with(7), 7);
    assert_eq!(run_with(-3), -3);
}
