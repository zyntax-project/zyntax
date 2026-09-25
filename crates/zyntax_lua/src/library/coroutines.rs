//! Coroutines over the runtime's fibers.
//!
//! A coroutine is a record under the thread tag: the fiber's handle,
//! its status, the function it runs, and a slot the values in flight
//! pass through, in either direction. The fiber's body is one
//! trampoline that reads the record back as its environment, calls the
//! function with the first resume's arguments and leaves the results
//! in the slot. A yield hands its values to the resumer through the
//! runtime and takes the next resume's arguments back the same way;
//! being stackful, it works at any depth of call inside the body.
//!
//! The fiber's handle is carried as a plain word: the coroutine, not
//! the compiler, owns its lifetime, and frees it once it has finished.

use super::*;
use zyntax_typed_ast::typed_ast::{TypedBlock, TypedDeclaration, TypedFunction};
use zyntax_typed_ast::{Type, Visibility};

/// The record's slots. `ERR` keeps the error a coroutine died of for
/// the close that reports it; `STARTED` says the body has run, so a
/// close has something to unwind. A dead coroutine that had a debug
/// hook keeps it in one more slot, `DEAD_HOOK`.
pub(super) const HANDLE: i64 = 0;
pub(super) const STATUS: i64 = 1;
const BODY: i64 = 2;
const SLOT: i64 = 3;
const ERR: i64 = 4;
const STARTED: i64 = 5;
pub(super) const DEAD_HOOK: i64 = 6;
/// The slot a coroutine that has run C code keeps that code's State
/// in, nil-padded to it; a coroutine that never has lacks it.
pub(super) const STATE_SLOT: i64 = DEAD_HOOK + 1;

/// Statuses, as the record stores them.
const SUSPENDED: i64 = 0;
const RUNNING: i64 = 1;
const NORMAL: i64 = 2;
pub(super) const DEAD: i64 = 3;

/// The packed step a resume returns: the tag in the low two bits.
const STEP_YIELDED: i64 = 0;
const STEP_DONE: i64 = 1;

/// The coroutine running now, or nil: what `coroutine.running` and
/// `coroutine.isyieldable` answer, and what a nested resume restores.
pub const CURRENT: &str = "zl_co_current";
/// The main thread as a coroutine value, once something asks for it.
const MAIN: &str = "zl_co_main_thread";

/// The stack a coroutine runs on. Committed by the page as it is
/// touched, so a large reservation costs little.
/// Reserved, not touched: the depth limit reaches this only through
/// frames far larger than compiled code makes.
const STACK_BYTES: i64 = 64 << 20;

pub(super) fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let co = kept("co", any());
    let f = kept("f", any());
    let x = kept("x", any());
    let rec = borrowed("rec", anys.clone());
    let prev = kept("prev", any());
    let args = kept("args", anys.clone());
    let out = borrowed("out", anys.clone());
    let step = local("step", i64());
    let depth = local("depth", i64());
    let ccalls = local("ccalls", i64());
    let status = local("status", i64());
    let handle = local("handle", i64());
    let env = borrowed("env", anys.clone());
    let mut d = vec![extern_fn(
        "zl_fiber_new",
        &[("code", usize()), ("env", any()), ("stack", i64())],
        i64(),
        Some("krio_fiber_new_with_env"),
    )];
    d.push(extern_fn(
        "zl_fiber_resume_with",
        &[("fiber", i64()), ("value", any())],
        i64(),
        Some("krio_fiber_resume_with"),
    ));
    d.push(extern_fn(
        "zl_fiber_yield",
        &[("value", any())],
        unit(),
        Some("krio_fiber_yield"),
    ));
    d.push(extern_fn(
        "zl_fiber_take_input",
        &[],
        any(),
        Some("krio_fiber_take_input"),
    ));
    d.push(extern_fn(
        "zl_fiber_free",
        &[("fiber", i64())],
        unit(),
        Some("krio_fiber_free"),
    ));
    // A word read back as the dynamic value whose address it is.
    d.push(extern_fn(
        "zl_word_as_any",
        &[("w", i64())],
        any(),
        Some("$Lua$word"),
    ));

    // The coroutine running now.
    d.push(zyntax_typed_ast::TypedNode::new(
        TypedDeclaration::Variable(zyntax_typed_ast::typed_ast::TypedVariable {
            name: intern(CURRENT),
            ty: any(),
            mutability: zyntax_typed_ast::Mutability::Mutable,
            initializer: None,
            visibility: Visibility::Public,
        }),
        Type::Unknown,
        SPAN,
    ));
    let current = || {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(CURRENT)),
            any(),
        )
    };
    let set_current = |v: Expr| {
        expr(node(
            zyntax_typed_ast::typed_ast::TypedExpression::Binary(
                zyntax_typed_ast::typed_ast::TypedBinary {
                    op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                    left: Box::new(current()),
                    right: Box::new(v),
                },
            ),
            any(),
        ))
    };
    let record_of = |co: Expr| call("zb_unbox_list_raw_any", vec![co], anys.clone());
    // The fiber handle of a coroutine, or 0 for the main thread (nil).
    let handle_of = |co: Expr| {
        if_expr(
            is_nil(co.clone()),
            int(0),
            call(
                "zb_box_get_i64",
                vec![at(record_of(co), int(HANDLE))],
                i64(),
            ),
        )
    };
    let status_of = |rec: Expr| call("zb_box_get_i64", vec![at(rec, int(STATUS))], i64());
    let set_status = |rec: Expr, s: i64| set_idx(rec, int(STATUS), box_i64(int(s)));
    let code_of = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            usize(),
        )
    };

    // The body every coroutine's fiber runs: the function in the
    // record, with the arguments the first resume left in the slot;
    // the results go back into the slot.
    d.push(zyntax_typed_ast::TypedNode::new(
        TypedDeclaration::Function(TypedFunction {
            name: intern("zl_co_body"),
            annotations: Vec::new(),
            effects: Vec::new(),
            with_handlers: Vec::new(),
            type_params: Vec::new(),
            params: Vec::new(),
            return_type: any(),
            body: Some(TypedBlock {
                statements: vec![
                    env.decl(record_of(call("zb_fiber_env", vec![], any()))),
                    args.decl(call(
                        "zl_values",
                        vec![at(env.e(), int(SLOT))],
                        anys.clone(),
                    )),
                    // The resume counted the C-stack level already.
                    set_idx(
                        env.e(),
                        int(SLOT),
                        call(
                            "zl_apply_packed",
                            vec![at(env.e(), int(BODY)), args.e()],
                            any(),
                        ),
                    ),
                    ret(nil()),
                ],
                span: SPAN,
            }),
            visibility: Visibility::Public,
            is_async: false,
            is_fiber: true,
            is_pure: false,
            is_external: false,
            calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
            link_name: None,
            module: Some(intern(zyntax_builtins::MODULE)),
        }),
        Type::Unknown,
        SPAN,
    ));

    // `coroutine.create(f)`; `what` names the function checking, for
    // `wrap` shares it.
    let what = kept("what", string());
    d.push(define(
        "zl_co_create",
        &[&f, &what],
        any(),
        vec![
            when(
                not(is_func(f.e())),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to '"),
                    what.e(),
                    text("' (function expected, got "),
                    arg_type_name(f.e()),
                    text(")"),
                ]))],
            ),
            rec.decl(list(
                vec![
                    box_i64(int(0)),
                    box_i64(int(SUSPENDED)),
                    f.e(),
                    nil(),
                    nil(),
                    box_bool(bool(false)),
                ],
                anys.clone(),
            )),
            co.decl(call(
                "zb_box_list_raw_any",
                vec![rec.e(), int32(thread_tag() as i32)],
                any(),
            )),
            set_idx(
                rec.e(),
                int(HANDLE),
                box_i64(call(
                    "zl_fiber_new",
                    vec![code_of("zl_co_body"), co.e(), int(STACK_BYTES)],
                    i64(),
                )),
            ),
            // A new thread starts with its creator's debug hook.
            expr(call(
                "zl_dbg_new_thread",
                vec![call(
                    "zb_box_get_i64",
                    vec![at(rec.e(), int(HANDLE))],
                    i64(),
                )],
                unit(),
            )),
            ret(co.e()),
        ],
    ));
    d.push(define(
        "zl_co_create_of",
        &[&f],
        any(),
        vec![ret(call(
            "zl_co_create",
            vec![f.e(), text("create")],
            any(),
        ))],
    ));
    let coroutine_expected = |what: &str| {
        lua_error(concat(vec![
            text(&format!(
                "bad argument #1 to '{what}' (thread expected, got "
            )),
            arg_type_name(co.e()),
            text(")"),
        ]))
    };
    // The main thread as a value, made when first asked for: a record
    // with no fiber, running, or normal while a coroutine runs.
    d.push(global_var(MAIN, any()));
    let main_thread = || read_global(MAIN, any());
    d.push(define(
        "zl_co_main",
        &[],
        any(),
        vec![
            when(
                is_nil(main_thread()),
                vec![set_global(
                    MAIN,
                    call(
                        "zb_box_list_raw_any",
                        vec![
                            list(
                                vec![box_i64(int(0)), box_i64(int(RUNNING)), nil(), nil()],
                                anys.clone(),
                            ),
                            int32(thread_tag() as i32),
                        ],
                        any(),
                    ),
                )],
            ),
            ret(main_thread()),
        ],
    ));
    // `coroutine.resume(co, ...)`: true and the yielded or returned
    // values, or false and a message.
    d.push(define(
        "zl_co_resume",
        &[&co, &args],
        any(),
        vec![
            when(not(is_thread(co.e())), vec![coroutine_expected("resume")]),
            rec.decl(record_of(co.e())),
            status.decl(status_of(rec.e())),
            when(
                eq(status.e(), int(DEAD)),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![
                            box_bool(bool(false)),
                            box_str(text("cannot resume dead coroutine")),
                        ],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            when(
                ne(status.e(), int(SUSPENDED)),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![
                            box_bool(bool(false)),
                            box_str(text("cannot resume non-suspended coroutine")),
                        ],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            // A resume nests on the reference's C stack.
            when(
                ge(read_global(CCALLS, i64()), int(CCALLS_LIMIT)),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![box_bool(bool(false)), box_str(text("C stack overflow"))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            set_idx(rec.e(), int(SLOT), call("zl_pack", vec![args.e()], any())),
            set_idx(rec.e(), int(STARTED), box_bool(bool(true))),
            prev.decl(current()),
            if_(
                not(is_nil(prev.e())),
                vec![set_status(record_of(prev.e()), NORMAL)],
                vec![when(
                    not(is_nil(main_thread())),
                    vec![set_status(record_of(main_thread()), NORMAL)],
                )],
            ),
            set_status(rec.e(), RUNNING),
            set_current(co.e()),
            set_global(LINE, int(0)),
            handle.decl(call(
                "zb_box_get_i64",
                vec![at(rec.e(), int(HANDLE))],
                i64(),
            )),
            // The resumer's depth again afterwards: frames the fiber
            // keeps while suspended are not on this stack.
            depth.decl(read_global(DEPTH, i64())),
            ccalls.decl(read_global(CCALLS, i64())),
            set_global(CCALLS, add(ccalls.e(), int(1))),
            expr(call("zl_dbg_switch", vec![handle.e()], unit())),
            step.decl(call(
                "zl_fiber_resume_with",
                vec![handle.e(), at(rec.e(), int(SLOT))],
                i64(),
            )),
            expr(call("zl_dbg_switch", vec![handle_of(prev.e())], unit())),
            set_global(DEPTH, depth.e()),
            set_global(CCALLS, ccalls.e()),
            set_current(prev.e()),
            if_(
                not(is_nil(prev.e())),
                vec![set_status(record_of(prev.e()), RUNNING)],
                vec![when(
                    not(is_nil(main_thread())),
                    vec![set_status(record_of(main_thread()), RUNNING)],
                )],
            ),
            out.decl(list(vec![box_bool(bool(true))], anys.clone())),
            when(
                eq(bitand(step.e(), int(3)), int(STEP_YIELDED)),
                vec![
                    set_status(rec.e(), SUSPENDED),
                    expr(call(
                        "zl_append_values",
                        vec![
                            out.e(),
                            call("zl_word_as_any", vec![shr(step.e(), int(2))], any()),
                        ],
                        unit(),
                    )),
                    ret(call("zb_box_tuple", vec![out.e()], any())),
                ],
            ),
            set_status(rec.e(), DEAD),
            expr(call("zl_fiber_free", vec![handle.e()], unit())),
            expr(call(
                "zl_dbg_drop",
                vec![handle.e(), rec.e(), not(is_nil(pending()))],
                unit(),
            )),
            // `os.exit` with the state to close leaves the coroutine
            // and goes on unwinding in the resumer.
            when(is_exiting(pending()), vec![ret(nil())]),
            // The body raised: the error comes back as the result, and
            // is kept for a close to report.
            when(
                not(is_nil(pending())),
                vec![
                    x.decl(call("zl_take_pending", vec![], any())),
                    set_idx(rec.e(), int(ERR), x.e()),
                    ret(call(
                        "zb_box_tuple",
                        vec![list(vec![box_bool(bool(false)), x.e()], anys.clone())],
                        any(),
                    )),
                ],
            ),
            when(
                eq(bitand(step.e(), int(3)), int(STEP_DONE)),
                vec![
                    expr(call(
                        "zl_append_values",
                        vec![out.e(), at(rec.e(), int(SLOT))],
                        unit(),
                    )),
                    ret(call("zb_box_tuple", vec![out.e()], any())),
                ],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(
                    vec![box_bool(bool(false)), box_str(text("error in coroutine"))],
                    anys.clone(),
                )],
                any(),
            )),
        ],
    ));
    // `coroutine.yield(...)`: the values to the resumer; the next
    // resume's arguments back.
    d.push(define(
        "zl_co_yield",
        &[&args],
        any(),
        vec![
            when(
                is_nil(current()),
                vec![lua_error(text("attempt to yield from outside a coroutine"))],
            ),
            depth.decl(read_global(DEPTH, i64())),
            expr(call(
                "zl_fiber_yield",
                vec![call("zl_pack", vec![args.e()], any())],
                unit(),
            )),
            set_global(DEPTH, depth.e()),
            x.decl(call("zl_fiber_take_input", vec![], any())),
            // Resumed to be closed: the error every block leaves on,
            // its `<close>` handlers seeing nil.
            when(
                is_closing(x.e()),
                vec![set_global(PENDING, x.e()), ret(nil())],
            ),
            ret(x.e()),
        ],
    ));
    d.push(define(
        "zl_co_status",
        &[&co],
        string(),
        vec![
            when(not(is_thread(co.e())), vec![coroutine_expected("status")]),
            status.decl(status_of(record_of(co.e()))),
            when(eq(status.e(), int(SUSPENDED)), vec![ret(text("suspended"))]),
            when(eq(status.e(), int(RUNNING)), vec![ret(text("running"))]),
            when(eq(status.e(), int(NORMAL)), vec![ret(text("normal"))]),
            ret(text("dead")),
        ],
    ));
    // `coroutine.running()`: the running coroutine and whether it is
    // the main one.
    d.push(define(
        "zl_co_running",
        &[],
        any(),
        vec![
            when(
                is_nil(current()),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![call("zl_co_main", vec![], any()), box_bool(bool(true))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![current(), box_bool(bool(false))], anys.clone())],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_co_isyieldable",
        &[&co],
        boolean(),
        vec![
            // Asked about a coroutine: any but the main thread can
            // yield; asked about nothing: whether one is running.
            when(is_nil(co.e()), vec![ret(not(is_nil(current())))]),
            when(
                not(is_thread(co.e())),
                vec![coroutine_expected("isyieldable")],
            ),
            ret(ne(
                call(
                    "zb_box_get_i64",
                    vec![at(record_of(co.e()), int(HANDLE))],
                    i64(),
                ),
                int(0),
            )),
        ],
    ));
    // `coroutine.close(co)`: a suspended coroutine is resumed to
    // unwind, its `<close>` handlers running innermost first; an
    // error one raises is the result. A dead one reports the error
    // it died of, once.
    let failed = |err: Expr| {
        ret(call(
            "zb_box_tuple",
            vec![list(vec![box_bool(bool(false)), err], anys.clone())],
            any(),
        ))
    };
    d.push(define(
        "zl_co_close",
        &[&co],
        any(),
        vec![
            when(not(is_thread(co.e())), vec![coroutine_expected("close")]),
            rec.decl(record_of(co.e())),
            status.decl(status_of(rec.e())),
            when(
                eq(status.e(), int(RUNNING)),
                vec![lua_error(text("cannot close a running coroutine"))],
            ),
            when(
                eq(status.e(), int(NORMAL)),
                vec![lua_error(text("cannot close a normal coroutine"))],
            ),
            when(
                eq(status.e(), int(DEAD)),
                vec![
                    x.decl(at(rec.e(), int(ERR))),
                    when(is_nil(x.e()), vec![ret(box_bool(bool(true)))]),
                    set_idx(rec.e(), int(ERR), nil()),
                    failed(x.e()),
                ],
            ),
            handle.decl(call(
                "zb_box_get_i64",
                vec![at(rec.e(), int(HANDLE))],
                i64(),
            )),
            // Never run: nothing to unwind.
            when(
                not(get_bool(at(rec.e(), int(STARTED)))),
                vec![
                    expr(call("zl_fiber_free", vec![handle.e()], unit())),
                    set_status(rec.e(), DEAD),
                    ret(box_bool(bool(true))),
                ],
            ),
            // Resumed with the closing error as the yield's result; it
            // runs, so a close from one of its handlers is refused.
            set_idx(rec.e(), int(SLOT), closing_marker()),
            prev.decl(current()),
            set_status(rec.e(), RUNNING),
            set_current(co.e()),
            // The handlers it runs count on the C stack; the close
            // itself does not.
            depth.decl(read_global(DEPTH, i64())),
            ccalls.decl(read_global(CCALLS, i64())),
            expr(call("zl_dbg_switch", vec![handle.e()], unit())),
            expr(call(
                "zl_fiber_resume_with",
                vec![handle.e(), at(rec.e(), int(SLOT))],
                i64(),
            )),
            expr(call("zl_dbg_switch", vec![handle_of(prev.e())], unit())),
            set_global(DEPTH, depth.e()),
            set_global(CCALLS, ccalls.e()),
            set_current(prev.e()),
            set_status(rec.e(), DEAD),
            expr(call("zl_fiber_free", vec![handle.e()], unit())),
            expr(call(
                "zl_dbg_drop",
                vec![handle.e(), rec.e(), bool(false)],
                unit(),
            )),
            when(is_exiting(pending()), vec![ret(nil())]),
            when(
                is_closing(pending()),
                vec![
                    expr(call("zl_take_pending", vec![], any())),
                    ret(box_bool(bool(true))),
                ],
            ),
            when(
                not(is_nil(pending())),
                vec![failed(call("zl_take_pending", vec![], any()))],
            ),
            ret(box_bool(bool(true))),
        ],
    ));
    // `coroutine.wrap(f)`: a function resuming the coroutine; a failed
    // resume is an error.
    let packed = kept("packed", any());
    let results = borrowed("results", anys.clone());
    let line = local("line", i64());
    d.push(define(
        "zl_co_wrap_code",
        &[&env, &packed],
        any(),
        vec![
            // An error is raised again at the caller, a string with
            // the caller's position in front.
            line.decl(read_global(LINE, i64())),
            x.decl(call(
                "zl_co_resume",
                vec![
                    at(env.e(), int(2)),
                    call("zl_values", vec![packed.e()], anys.clone()),
                ],
                any(),
            )),
            // An error still pending after the resume, an exit in
            // flight, goes on unwinding.
            when(not(is_nil(pending())), vec![ret(nil())]),
            results.decl(call("zl_values", vec![x.e()], anys.clone())),
            when(
                not(call("zl_truthy", vec![at(results.e(), int(0))], boolean())),
                vec![
                    set_global(LINE, line.e()),
                    expr(call(
                        "zl_error",
                        vec![
                            call("zl_value_at", vec![results.e(), int(2)], any()),
                            int(1),
                        ],
                        unit(),
                    )),
                    ret(nil()),
                ],
            ),
            ret(call("zl_values_from", vec![results.e(), int(2)], any())),
        ],
    ));
    d.push(define(
        "zl_co_wrap",
        &[&f],
        any(),
        vec![
            co.decl(call("zl_co_create", vec![f.e(), text("wrap")], any())),
            ret(call(
                "zb_func_new",
                vec![
                    code_of("zl_co_wrap_code"),
                    int(zyntax_builtins::functions::VARIADIC_ARITY),
                    list(vec![co.e()], anys.clone()),
                ],
                any(),
            )),
        ],
    ));
    d
}
