//! Host-driven fibers: a framework constructs a machine from a
//! compiled `fiber def`, holds a token, and steps it on its own
//! schedule — installing effect handlers around each step when the
//! machine observes events. The token survives reloads; the edges an
//! edit creates (function deleted, yield shape changed) surface as
//! values and handle metadata, never as traps.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{
    HostFiberStep, TieredConfig, TieredRuntime, TypeTag, ZrtlSigFlags, ZrtlSymbolSig, ZyntaxValue,
};
use zyntax_typed_ast::{PrimitiveType, Span, Type, TypedASTBuilder, TypedStatement, Visibility};

fn runtime() -> TieredRuntime {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    config.enable_hot_reload = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    TieredRuntime::new(config).expect("runtime should start")
}

fn parse(src: &str) -> zyntax_embed::TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    grammar
        .parse_with_filename(src, "<host_fiber_api>")
        .expect("source should parse")
}

/// The host steps a plain machine through its yields to completion,
/// and stepping past completion stays `Done`.
#[test]
fn a_host_drives_a_machine_through_its_yields() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(
        r#"
fiber def counter(): i64 {
    let mut n: i64 = 0
    while n < 3 {
        n = n + 1
        yield n
    }
    return n
}
"#,
    ))
    .expect("should compile");

    let token = rt.get_fiber("counter").expect("create");
    for expect in 1..=3 {
        assert_eq!(
            rt.resume_fiber(token).expect("resume"),
            HostFiberStep::Yielded(ZyntaxValue::Int(expect))
        );
    }
    assert_eq!(rt.resume_fiber(token).expect("resume"), HostFiberStep::Done);
    assert_eq!(
        rt.resume_fiber(token).expect("resume"),
        HostFiberStep::Done,
        "a completed machine stays Done"
    );
    rt.drop_fiber(token).expect("drop");
    assert!(rt.resume_fiber(token).is_err(), "the token is dead");
}

/// Only `fiber def` functions are constructible.
#[test]
fn a_plain_function_is_not_a_machine() {
    let mut rt = runtime();
    rt.compile_typed_program(parse("def plain(): i64 { return 1 }"))
        .expect("should compile");
    assert!(rt.get_fiber("plain").is_err());
}

extern "C" fn host_fiber_value() -> i64 {
    17
}

/// Typed embedders can batch-register host functions, publish them into
/// the tier-0 JIT once, and call them from a host-driven fiber.
#[test]
fn a_tiered_host_symbol_is_finalized_before_fiber_compilation() {
    let mut rt = runtime();
    rt.register_function_typed(
        "$Host$fiber_value",
        host_fiber_value as *const u8,
        ZrtlSymbolSig {
            param_count: 0,
            flags: ZrtlSigFlags::NONE,
            return_type: TypeTag::I64,
            params: [TypeTag::VOID; 16],
        },
    );
    rt.finalize_runtime_symbols().expect("publish host symbol");

    let mut builder = TypedASTBuilder::new();
    let span = Span::new(0, 0);
    let i64_type = Type::Primitive(PrimitiveType::I64);
    let external = builder.extern_function(
        "$Host$fiber_value",
        vec![],
        i64_type.clone(),
        Visibility::Private,
        span,
    );
    let callee = builder.variable("$Host$fiber_value", i64_type.clone(), span);
    let value = builder.call_positional(callee, vec![], i64_type.clone(), span);
    let yielded = zyntax_typed_ast::typed_ast::typed_node(
        TypedStatement::Yield(Box::new(value)),
        Type::Primitive(PrimitiveType::Unit),
        span,
    );
    let body = builder.block(vec![yielded], span);
    let mut machine = builder.function(
        "host_value_machine",
        vec![],
        i64_type,
        body,
        Visibility::Public,
        false,
        span,
    );
    let zyntax_typed_ast::TypedDeclaration::Function(function) = &mut machine.node else {
        unreachable!("builder returned a non-function")
    };
    function.is_fiber = true;
    let program = builder.program(vec![external, machine], span);

    rt.compile_typed_program(program).expect("compile");
    let token = rt.get_fiber("host_value_machine").expect("get");
    assert_eq!(
        rt.resume_fiber(token).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(17))
    );
    rt.drop_fiber(token).expect("drop");
}

const OBSERVER: &str = r#"
effect Event {
    def next_event(): i64
}

handler Feed for Event {
    def next_event(): i64 { return 3 }
}

@effect(Event)
fiber def machine(): i64 {
    let mut state: i64 = 0
    while state < 100 {
        let e = next_event()
        state = state + e
        yield state
    }
    return state
}
"#;

/// The host installs the event source around each step — the host
/// equivalent of `with Feed { ... f.next() ... }` — and a handler edit
/// mid-run retargets what the machine observes at its next step.
#[test]
fn a_host_installs_the_event_source_around_each_step() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(OBSERVER))
        .expect("should compile");

    let token = rt.get_fiber("machine").expect("create");
    assert_eq!(
        rt.resume_fiber_within(token, &["Feed"]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(3))
    );
    assert_eq!(
        rt.resume_fiber_within(token, &["Feed"]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(6))
    );

    let edited = OBSERVER.replace("return 3", "return 5");
    let report = rt
        .reload_typed_program(parse(&edited))
        .expect("reload should succeed");
    assert!(
        report
            .dispatch_patched
            .iter()
            .any(|n| n.contains("next_event")),
        "{report:?}"
    );

    assert_eq!(
        rt.resume_fiber_within(token, &["Feed"]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(11)),
        "the running machine observes the edited event source"
    );

    let info = rt.fiber_info(token).expect("info");
    assert!(!info.machine_gone);
    assert!(!info.shape_stale, "the machine's own shape never changed");
}

/// An edit deletes the machine's function while the host still holds
/// its token: the next resume fails as a value — the UI's cue to drop
/// and remount — and dropping the fiber still works.
#[test]
fn a_deleted_machine_fails_as_a_value() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(OBSERVER))
        .expect("should compile");

    let token = rt.get_fiber("machine").expect("create");
    assert_eq!(
        rt.resume_fiber_within(token, &["Feed"]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(3))
    );

    // The edit removes the machine entirely; effect + handler remain.
    let edited = r#"
effect Event {
    def next_event(): i64
}

handler Feed for Event {
    def next_event(): i64 { return 3 }
}
"#;
    let report = rt
        .reload_typed_program(parse(edited))
        .expect("reload should succeed");
    assert!(
        report.removed_retained.iter().any(|n| n == "machine"),
        "{report:?}"
    );

    assert_eq!(
        rt.resume_fiber_within(token, &["Feed"]).expect("step"),
        HostFiberStep::MachineGone,
        "resume fails as a value, not a trap"
    );
    let info = rt.fiber_info(token).expect("info");
    assert!(info.machine_gone);
    rt.drop_fiber(token).expect("a gone machine still drops");
}

/// An edit changes the machine's yield shape. The handle detects it —
/// generation on the handle, staleness in `fiber_info` — instead of
/// misreading payloads: the running fiber still yields the shape the
/// handle was created against, and a machine created after the edit
/// carries the new generation.
#[test]
fn a_yield_shape_edit_marks_the_handle_stale() {
    let v1 = r#"
fiber def wave(): i64 {
    yield 1
    yield 2
}
"#;
    let v2 = r#"
fiber def wave(): f64 {
    yield 1.5
    yield 2.5
}
"#;
    let mut rt = runtime();
    rt.compile_typed_program(parse(v1)).expect("should compile");

    let old_handle = rt.get_fiber("wave").expect("create");
    assert_eq!(
        rt.resume_fiber(old_handle).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(1))
    );

    rt.reload_typed_program(parse(v2))
        .expect("reload should succeed");

    let info = rt.fiber_info(old_handle).expect("info");
    assert!(
        info.shape_stale,
        "the handle detects the shape change: {info:?}"
    );

    // The suspended machine still speaks its own generation's shape.
    assert_eq!(
        rt.resume_fiber(old_handle).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(2))
    );

    // A machine created after the edit is of the new generation.
    let new_handle = rt.get_fiber("wave").expect("create");
    let new_info = rt.fiber_info(new_handle).expect("info");
    assert!(!new_info.shape_stale);
    assert!(new_info.shape_generation > info.shape_generation);

    rt.drop_fiber(old_handle).expect("drop old");
    rt.drop_fiber(new_handle).expect("drop new");
}

const STATEFUL_OBSERVER: &str = r#"
effect Counter {
    def next(): i64
}

handler Seq for Counter {
    var n: i64 = 0
    def next(): i64 {
        self.n = self.n + 1
        return self.n
    }
}

@effect(Counter)
fiber def watcher(): i64 {
    let mut steps: i64 = 0
    while steps < 3 {
        steps = steps + 1
        yield next()
    }
    return steps
}
"#;

/// Handler-state persistence is explicit in the API: a handler BOUND
/// to the machine allocates its state once and carries it across
/// steps, while `resume_fiber_within` opens a fresh scope — fresh
/// state — per step.
#[test]
fn a_bound_handler_carries_state_across_steps() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(STATEFUL_OBSERVER))
        .expect("should compile");

    // Per-step scopes: every resume constructs new handler state, so
    // the machine observes 1 each time.
    let per_step = rt.get_fiber("watcher").expect("get");
    for _ in 0..3 {
        assert_eq!(
            rt.resume_fiber_within(per_step, &["Seq"]).expect("step"),
            HostFiberStep::Yielded(ZyntaxValue::Int(1))
        );
    }
    rt.drop_fiber(per_step).expect("drop");

    // Bound: one state for the machine's lifetime.
    let bound = rt.get_fiber("watcher").expect("get");
    let seq = rt.get_handler("Seq").expect("resolve once");
    assert_eq!(rt.handler_name(seq), Some("Seq"));
    rt.bind_fiber_handler(bound, seq).expect("bind");
    for expect in 1..=3 {
        assert_eq!(
            rt.resume_fiber(bound).expect("step"),
            HostFiberStep::Yielded(ZyntaxValue::Int(expect)),
            "bound handler state persists across steps"
        );
    }
    assert_eq!(rt.resume_fiber(bound).expect("step"), HostFiberStep::Done);
    rt.drop_fiber(bound).expect("drop");
}

/// A partial handler install unwinds: when one of several named
/// handlers fails to resolve, the frames already pushed are popped
/// before the error surfaces, leaving the thread's handler stack as
/// it was.
#[test]
fn a_failed_handler_install_leaves_no_frames_behind() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(OBSERVER))
        .expect("should compile");
    let token = rt.get_fiber("machine").expect("get");

    let depth = zyntax_embed::handler_stack_depth();
    assert!(
        rt.resume_fiber_within(token, &["Feed", "NoSuchHandler"])
            .is_err(),
        "the unknown handler must fail the install"
    );
    assert_eq!(
        zyntax_embed::handler_stack_depth(),
        depth,
        "the frames pushed before the failure must be unwound"
    );

    // The machine is untouched and still drivable.
    assert_eq!(
        rt.resume_fiber_within(token, &["Feed"]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(3))
    );
    rt.drop_fiber(token).expect("drop");
}

/// A deleted machine's declaration leaves the shape registry with it:
/// no new machine of a removed function can be constructed, and a
/// rollback restores both the code AND the handles' metadata — the
/// gone mark lifts, the declaration is constructible again.
#[test]
fn a_rollback_restores_the_handles_view() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(OBSERVER))
        .expect("should compile");
    let token = rt.get_fiber("machine").expect("get");
    assert_eq!(
        rt.resume_fiber_within(token, &["Feed"]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(3))
    );

    let edited = r#"
effect Event {
    def next_event(): i64
}

handler Feed for Event {
    def next_event(): i64 { return 3 }
}
"#;
    let report = rt
        .reload_typed_program(parse(edited))
        .expect("reload should succeed");
    assert!(report.removed_retained.iter().any(|n| n == "machine"));

    assert!(
        rt.get_fiber("machine").is_err(),
        "a deleted declaration is not constructible"
    );
    assert_eq!(
        rt.resume_fiber_within(token, &["Feed"]).expect("step"),
        HostFiberStep::MachineGone
    );

    rt.rollback_last_reload().expect("rollback");
    assert!(
        !rt.fiber_info(token).expect("info").machine_gone,
        "rollback lifts the gone mark"
    );
    assert_eq!(
        rt.resume_fiber_within(token, &["Feed"]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(6)),
        "the machine resumes where it stood"
    );
    let again = rt.get_fiber("machine").expect("constructible again");
    rt.drop_fiber(again).expect("drop");
    rt.drop_fiber(token).expect("drop");
}

/// Rollback also restores the shape registry: a shape-changing edit
/// marks the handle stale, and rolling the edit back un-marks it.
#[test]
fn a_rollback_restores_shape_generations() {
    let v1 = "fiber def pulse(): i64 {\n    yield 1\n    yield 2\n}\n";
    let v2 = "fiber def pulse(): f64 {\n    yield 1.5\n    yield 2.5\n}\n";
    let mut rt = runtime();
    rt.compile_typed_program(parse(v1)).expect("should compile");
    let token = rt.get_fiber("pulse").expect("get");

    rt.reload_typed_program(parse(v2)).expect("reload");
    assert!(rt.fiber_info(token).expect("info").shape_stale);

    rt.rollback_last_reload().expect("rollback");
    assert!(
        !rt.fiber_info(token).expect("info").shape_stale,
        "rollback restores the shape generation"
    );
    rt.drop_fiber(token).expect("drop");
}

/// Dropping the runtime frees the machines it still owns — exercised
/// with live suspended fibers so the teardown path runs for real.
#[test]
fn a_runtime_shutdown_frees_its_machines() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(OBSERVER))
        .expect("should compile");
    let a = rt.get_fiber("machine").expect("get");
    let _b = rt.get_fiber("machine").expect("get");
    assert_eq!(
        rt.resume_fiber_within(a, &["Feed"]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(3))
    );
    drop(rt); // frees both fibers (one mid-run, one never started)
}

/// The tiered runtime exposes the same built-in-class seam as the
/// classic runtime: a class registered before compilation joins the
/// registry snapshot the lowering consults, without disturbing the
/// default classes. (Downstream frameworks register typed built-ins
/// like result and notification handles here.)
#[test]
fn a_registered_builtin_class_joins_the_tiered_compilation() {
    use std::sync::Arc;

    struct Probe;
    impl zyntax_compiler::builtin_class::BuiltinClass for Probe {
        fn name(&self) -> &str {
            "HostProbe"
        }
        fn matches(&self, _ty: &zyntax_typed_ast::Type) -> bool {
            false
        }
        fn dispatch(
            &self,
            _ssa: &mut zyntax_compiler::ssa::SsaBuilder,
            _block_id: zyntax_compiler::hir::HirId,
            _method: &str,
            _receiver: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedExpression>,
            _receiver_ty: &zyntax_typed_ast::Type,
            _args: &[zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedExpression>],
            _result_ty: &zyntax_typed_ast::Type,
        ) -> zyntax_compiler::CompilerResult<Option<zyntax_compiler::hir::HirId>> {
            Ok(None)
        }
    }

    let mut rt = runtime();
    rt.register_builtin_class(Arc::new(Probe));
    rt.compile_typed_program(parse(OBSERVER))
        .expect("compilation with an extra registered class succeeds");
    let token = rt.get_fiber("machine").expect("get");
    assert_eq!(
        rt.resume_fiber_within(token, &["Feed"]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(3)),
        "default dispatch (Fiber, effects) is undisturbed"
    );
    rt.drop_fiber(token).expect("drop");
}

/// A handler token pins its resolution: driving through tokens is
/// exact — no bare-name lookup at step time — and an unknown token
/// fails cleanly.
#[test]
fn a_handler_token_pins_its_resolution() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(OBSERVER))
        .expect("should compile");
    let machine = rt.get_fiber("machine").expect("get");
    let feed = rt.get_handler("Feed").expect("resolve once");

    assert_eq!(
        rt.resume_fiber_handled(machine, &[feed]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(3))
    );
    assert_eq!(
        rt.resume_fiber_handled(machine, &[feed]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(6))
    );
    assert!(rt.get_handler("NoSuchHandler").is_err());
    rt.drop_fiber(machine).expect("drop");
}
