#![cfg(feature = "cranelift-backend")]

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
    let konst = func.create_value(
        HirType::I32,
        HirValueKind::Constant(HirConstant::I32(value)),
    );
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

// ── Cost ─────────────────────────────────────────────────────────────
//
// Step 2 of the spike: what does going through a handler cost, against
// the same work reached by a direct call?
//
// The loop runs in Rust, not in the JIT: both variants pay an identical
// Rust→native call per iteration, so the DIFFERENCE isolates the perform
// overhead without hand-writing loop blocks and phis in HIR.
//
// The reference point is a `HashMap` lookup, because that is what a
// signal read costs today. A handler per read is only interesting if the
// overhead sits somewhere near that.
//
// RUN THIS IN RELEASE. The perform-vs-call ratio holds either way: both
// sides are the same Cranelift-compiled code behind an identical
// Rust->native call, so the build profile cancels out. The HashMap
// figure does NOT — in debug it is unoptimised `std` hashing, ~20x
// slower than it should be, which makes a perform look far cheaper
// against a signal read than it really is.

/// `fn direct() -> i32 { CounterHandler$bump() }` — the same handler
/// body, reached by an ordinary call instead of a perform.
fn calling_function(fn_name: &str, callee: HirId) -> HirFunction {
    let mut func = HirFunction::new(name(fn_name), signature(vec![HirType::I32]));
    func.calling_convention = CallingConvention::C;
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let entry = func.entry_block;
    let block = func.blocks.get_mut(&entry).unwrap();
    block.add_instruction(HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Function(callee),
        args: vec![],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });
    func
}

/// Compile one module holding both variants, so they share a backend
/// and neither gets an allocation or layout advantage over the other.
fn compile_both() -> (CraneliftBackend, HirId, HirId) {
    let mut module = empty_module();

    let (effect_id, effect) = declare_effect("Counter", "bump", HirType::I32);
    module.effects.insert(effect_id, effect);
    let (handler_id, handler) = declare_handler("CounterHandler", effect_id, "bump", HirType::I32);
    module.handlers.insert(handler_id, handler);

    let body = handler_body("CounterHandler$bump", 42);
    let body_id = body.id;
    module.functions.insert(body_id, body);

    let performing = performing_function("via_perform", effect_id, "bump");
    let perform_id = performing.id;
    module.functions.insert(perform_id, performing);

    let calling = calling_function("via_call", body_id);
    let call_id = calling.id;
    module.functions.insert(call_id, calling);

    let mut backend = backend_with_effect_runtime();
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    (backend, perform_id, call_id)
}

fn time_calls(raw: *const u8, iterations: u32) -> std::time::Duration {
    let f = unsafe { std::mem::transmute::<*const u8, unsafe extern "C" fn() -> i32>(raw) };
    // Warm the branch predictor and the icache so the first iterations
    // don't dominate a short run.
    for _ in 0..1_000 {
        std::hint::black_box(unsafe { f() });
    }
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(unsafe { f() });
    }
    start.elapsed()
}

/// Reports the numbers; asserts only that a perform stays within an
/// order of magnitude of a direct call. A tighter bound would be a
/// flaky test rather than a better one — the point is to catch the
/// perform site degrading into something structurally worse (a lookup
/// per call that misses, a trap-and-recover), not to police jitter.
#[test]
fn a_perform_costs_about_what_a_call_costs() {
    const N: u32 = 2_000_000;

    let (backend, perform_id, call_id) = compile_both();
    let perform_ptr = backend.get_function_ptr(perform_id).expect("perform ptr");
    let call_ptr = backend.get_function_ptr(call_id).expect("call ptr");

    let via_call = time_calls(call_ptr, N);
    let via_perform = time_calls(perform_ptr, N);

    // What a signal read costs today, for scale.
    let mut map: std::collections::HashMap<u64, i64> = std::collections::HashMap::new();
    for k in 0..64u64 {
        map.insert(k, k as i64);
    }
    let start = std::time::Instant::now();
    for i in 0..N {
        std::hint::black_box(map.get(&(u64::from(i) % 64)));
    }
    let via_hashmap = start.elapsed();

    let ns = |d: std::time::Duration| d.as_nanos() as f64 / f64::from(N);
    println!(
        "per op over {N} iterations: direct call {:.2}ns, perform {:.2}ns, \
         HashMap lookup {:.2}ns",
        ns(via_call),
        ns(via_perform),
        ns(via_hashmap),
    );
    println!(
        "perform overhead vs direct call: {:.2}x; vs a HashMap read: {:.2}x",
        ns(via_perform) / ns(via_call).max(f64::EPSILON),
        ns(via_perform) / ns(via_hashmap).max(f64::EPSILON),
    );

    assert!(
        ns(via_perform) < ns(via_call) * 10.0 + 20.0,
        "a perform should be a call plus a dispatch check, not something \
         structurally worse: {:.2}ns vs {:.2}ns direct",
        ns(via_perform),
        ns(via_call),
    );
}

// ── Composition across the host boundary ─────────────────────────────
//
// Step 3 of the spike: can a handler live on the HOST side, so JIT'd
// code performing an operation reaches Rust state?
//
// This is the shape a reactive read would take. `set_stateful_deps_notifier`
// is already a process-global hook that compiled code reaches through
// without knowing `Stateful` exists — but it runs in the other direction
// (the write notifies), and the read side is inferred rather than
// observed. A read handler inverts that: it sees every read in its
// scope, exactly, as it happens.
//
// Worth stating plainly, because step 1 flagged continuations as the
// open risk: a read handler is TAIL-RESUMPTIVE. `perform read(id)`
// lowers to a plain call that returns the value and lets execution
// continue, which is "resume with a value" without capturing anything.
// So this needs neither `Resume<T>` nor `CaptureContinuation` — the two
// pieces that are unproven and unimplemented respectively.

use std::sync::Mutex;

/// Signal ids the JIT'd code read, in order. Stands in for what a
/// `Stateful` would collect as its dependency set.
static READS: Mutex<Vec<i64>> = Mutex::new(Vec::new());

/// The host side of the handler: record the read, return the value.
/// Values are `id * 10` so the assertion can tell which id produced
/// which contribution to the sum.
extern "C" fn host_read(id: i64) -> i64 {
    READS.lock().unwrap_or_else(|e| e.into_inner()).push(id);
    id * 10
}

fn i64_param(param_name: &str) -> HirParam {
    HirParam {
        id: HirId::new(),
        name: name(param_name),
        ty: HirType::I64,
        attributes: ParamAttributes::default(),
        ownership: Default::default(),
    }
}

/// `fn ReactiveHandler$read(id: i64) -> i64 { __test_host_read(id) }`
fn delegating_handler_body(mangled: &str) -> HirFunction {
    let mut sig = signature(vec![HirType::I64]);
    sig.params = vec![i64_param("id")];
    let mut func = HirFunction::new(name(mangled), sig);
    func.calling_convention = CallingConvention::C;

    let id = func.create_value(HirType::I64, HirValueKind::Parameter(0));
    let out = func.create_value(HirType::I64, HirValueKind::Instruction);
    let entry = func.entry_block;
    let block = func.blocks.get_mut(&entry).unwrap();
    block.add_instruction(HirInstruction::Call {
        result: Some(out),
        callee: HirCallable::Symbol("__test_host_read".to_string()),
        args: vec![id],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    block.set_terminator(HirTerminator::Return { values: vec![out] });
    func
}

/// `fn run() -> i64 { perform read(1) + perform read(2) }`
fn two_reads_function(fn_name: &str, effect_id: HirId, op: &str) -> HirFunction {
    let mut func = HirFunction::new(name(fn_name), signature(vec![HirType::I64]));
    func.calling_convention = CallingConvention::C;

    let one = func.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(1)));
    let two = func.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(2)));
    let a = func.create_value(HirType::I64, HirValueKind::Instruction);
    let b = func.create_value(HirType::I64, HirValueKind::Instruction);
    let sum = func.create_value(HirType::I64, HirValueKind::Instruction);

    let entry = func.entry_block;
    let block = func.blocks.get_mut(&entry).unwrap();
    block.add_instruction(HirInstruction::PerformEffect {
        result: Some(a),
        effect_id,
        op_name: name(op),
        args: vec![one],
        return_ty: HirType::I64,
    });
    block.add_instruction(HirInstruction::PerformEffect {
        result: Some(b),
        effect_id,
        op_name: name(op),
        args: vec![two],
        return_ty: HirType::I64,
    });
    block.add_instruction(HirInstruction::Binary {
        op: BinaryOp::Add,
        result: sum,
        ty: HirType::I64,
        left: a,
        right: b,
    });
    block.set_terminator(HirTerminator::Return { values: vec![sum] });
    func
}

/// A handler that takes one `i64` and returns one, delegating to host
/// code. Params must match the perform site or the call is malformed.
fn declare_reading_handler(
    handler_name: &str,
    effect_id: HirId,
    op: &str,
) -> (HirId, HirEffectHandler) {
    let (handler_id, mut handler) = declare_handler(handler_name, effect_id, op, HirType::I64);
    handler.implementations[0].params = vec![i64_param("id")];
    (handler_id, handler)
}

/// The composition question, answered by what the host observed: JIT'd
/// code performed two reads, the host handler saw both ids in order,
/// and the values it returned are what the computation summed.
#[test]
fn a_handler_can_delegate_to_host_state() {
    READS.lock().unwrap_or_else(|e| e.into_inner()).clear();

    let mut module = empty_module();
    let effect_id = HirId::new();
    module.effects.insert(
        effect_id,
        HirEffect {
            id: effect_id,
            name: name("Reactive"),
            type_params: vec![],
            operations: vec![HirEffectOp {
                id: HirId::new(),
                name: name("read"),
                type_params: vec![],
                params: vec![i64_param("id")],
                return_type: HirType::I64,
            }],
        },
    );

    let (handler_id, handler) = declare_reading_handler("ReactiveHandler", effect_id, "read");
    module.handlers.insert(handler_id, handler);

    let body = delegating_handler_body("ReactiveHandler$read");
    module.functions.insert(body.id, body);

    let run = two_reads_function("run_reads", effect_id, "read");
    let run_id = run.id;
    module.functions.insert(run_id, run);

    extern "C" fn lookup_op(_effect_id: u64, _op_index: u64) -> *mut u8 {
        std::ptr::null_mut()
    }
    extern "C" fn lookup_state(_effect_id: u64) -> *mut u8 {
        std::ptr::null_mut()
    }
    extern "C" fn lookup_op_is_async(_effect_id: u64, _op_index: u64) -> i64 {
        0
    }
    let mut backend = CraneliftBackend::with_runtime_symbols(&[
        ("__zyntax_effect_lookup_op", lookup_op as *const u8),
        ("__zyntax_effect_lookup_state", lookup_state as *const u8),
        (
            "__zyntax_effect_lookup_op_is_async",
            lookup_op_is_async as *const u8,
        ),
        ("__test_host_read", host_read as *const u8),
    ])
    .expect("backend");

    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let raw = backend.get_function_ptr(run_id).expect("fn ptr");
    let f = unsafe { std::mem::transmute::<*const u8, unsafe extern "C" fn() -> i64>(raw) };
    let sum = unsafe { f() };

    let observed = READS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(
        observed,
        vec![1, 2],
        "the host handler must see every read in its scope, in order"
    );
    assert_eq!(
        sum, 30,
        "and the values it returned are what the computation used \
         (1*10 + 2*10); a wrong sum means the args or the return value \
         did not survive the perform"
    );
}
