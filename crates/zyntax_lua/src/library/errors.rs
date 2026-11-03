//! Errors support for the Lua library.

use super::*;

/// Whether a stored line is in a chunk without debug information.
pub fn stripped_line(line: Expr) -> Expr {
    eq(bitand(line, int(STRIPPED_LINE)), int(STRIPPED_LINE))
}

pub fn global_var(name: &str, ty: Type) -> Decl {
    zyntax_typed_ast::TypedNode::new(
        TypedDeclaration::Variable(zyntax_typed_ast::typed_ast::TypedVariable {
            name: intern(name),
            ty,
            mutability: Mutability::Mutable,
            initializer: None,
            visibility: Visibility::Public,
        }),
        Type::Unknown,
        SPAN,
    )
}

pub fn read_global(name: &str, ty: Type) -> Expr {
    node(
        zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
        ty,
    )
}

pub fn set_global(name: &str, value: Expr) -> Stmt {
    let ty = value.ty.clone();
    expr(node(
        zyntax_typed_ast::typed_ast::TypedExpression::Binary(
            zyntax_typed_ast::typed_ast::TypedBinary {
                op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                left: Box::new(read_global(name, ty.clone())),
                right: Box::new(value),
            },
        ),
        ty,
    ))
}

pub fn pending() -> Expr {
    read_global(PENDING, any())
}

/// An error was just raised: a program that keeps its call stack notes
/// where, for the report of an error nothing catches.
pub(super) fn note_raise() -> Stmt {
    when(
        read_global(debug::DBG_ON, boolean()),
        vec![expr(call(
            "zl_dbg_note_raise",
            vec![read_global(LINE, i64())],
            unit(),
        ))],
    )
}

/// Raising: the first error stands until it is taken; a message gets
/// the running statement's position, an error value is kept as it is.
pub(super) fn raising(t: &Types) -> Vec<Decl> {
    let kind = local("kind", string());
    let message = local("message", string());
    let v = kept("v", any());
    let s = local("s", string());
    let level = local("level", i64());
    let line = local("line", i64());
    let mut d = vec![
        global_var(PENDING, any()),
        global_var(LINE, i64()),
        global_var(CHUNK, string()),
        global_var(CHUNKS, any()),
        global_var(DEPTH, i64()),
        global_var(CCALLS, i64()),
        global_var(VARINFO, i64()),
        global_var(FOR_SKIP, boolean()),
        global_var(OVERFLOWED, boolean()),
        global_var(HANDLING_OVERFLOW, boolean()),
        global_var(METHOD_CALL, boolean()),
        global_var(EXIT_ERROR, any()),
        global_var(ARR_EMPTY, any()),
    ];
    // Entered below the floor: the error every deeper call would raise.
    // Under a message handler for an overflow it is an error in error
    // handling, a value with no position.
    d.push(define_cold(
        "zl_stack_overflow",
        &[],
        unit(),
        vec![
            set_global(OVERFLOWED, bool(true)),
            when(
                read_global(HANDLING_OVERFLOW, boolean()),
                vec![
                    expr(call(
                        "zl_raise_value",
                        vec![box_str(text("error in error handling"))],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            lua_error(text("stack overflow")),
            ret_void(),
        ],
    ));
    // The name of the chunk a stored line belongs to.
    let found = local("found", any());
    d.push(define(
        "zl_chunk_of",
        &[&line],
        string(),
        vec![
            when(
                or(
                    eq(shr(line.e(), int(LINE_BITS)), int(0)),
                    is_nil(read_global(CHUNKS, any())),
                ),
                vec![ret(read_global(CHUNK, string()))],
            ),
            // An internal table, read raw: nothing of the program's
            // runs while an error is positioned.
            found.decl(call(
                "zl_rawgeti",
                vec![
                    unbox_table(read_global(CHUNKS, any()), t),
                    shr(line.e(), int(LINE_BITS)),
                ],
                any(),
            )),
            when(is_nil(found.e()), vec![ret(read_global(CHUNK, string()))]),
            ret(get_str(found.e())),
        ],
    ));
    // A chunk a program runs besides its main one, under the number its
    // lines carry.
    let name = kept("name", string());
    let index = local("index", i64());
    d.push(define(
        "zl_chunk_add",
        &[&name, &index],
        unit(),
        vec![
            when(
                is_nil(read_global(CHUNKS, any())),
                vec![set_global(
                    CHUNKS,
                    box_table(call("zl_table_new", vec![], t.table())),
                )],
            ),
            expr(call(
                "zl_rawseti",
                vec![
                    unbox_table(read_global(CHUNKS, any()), t),
                    index.e(),
                    box_str(name.e()),
                ],
                unit(),
            )),
            ret_void(),
        ],
    ));
    // `message` positioned at a stored line: its chunk and line.
    d.push(define(
        "zl_position_at",
        &[&line, &message],
        string(),
        vec![
            when(le(line.e(), int(0)), vec![ret(message.e())]),
            ret(concat(vec![
                call("zl_chunk_of", vec![line.e()], string()),
                text(":"),
                if_expr(
                    stripped_line(line.e()),
                    text("-1"),
                    call(
                        "zb_str_of_int",
                        vec![bitand(line.e(), int((1i64 << LINE_BITS) - 1))],
                        string(),
                    ),
                ),
                text(": "),
                message.e(),
            ])),
        ],
    ));
    // The message with the position of the statement running.
    d.push(define(
        "zl_position",
        &[&message],
        string(),
        vec![ret(call(
            "zl_position_at",
            vec![read_global(LINE, i64()), message.e()],
            string(),
        ))],
    ));
    d.push(define_cold(
        "zl_raise_value",
        &[&v],
        unit(),
        vec![
            when(not(is_nil(pending())), vec![ret_void()]),
            set_global(VARINFO, int(0)),
            if_(
                is_nil(v.e()),
                vec![set_global(PENDING, nil_error())],
                vec![set_global(PENDING, v.e())],
            ),
            note_raise(),
            ret_void(),
        ],
    ));
    // The shared library's hook: a message, positioned.
    d.push(define_cold(
        "zb_hook_raise",
        &[&kind, &message],
        unit(),
        vec![
            when(not(is_nil(pending())), vec![ret_void()]),
            set_global(VARINFO, int(0)),
            set_global(
                PENDING,
                box_str(call("zl_position", vec![message.e()], string())),
            ),
            note_raise(),
            ret_void(),
        ],
    ));
    // The shared library asks whether an error is pending after each
    // call back into the program.
    d.push(define(
        "zb_hook_pending",
        &[],
        boolean(),
        vec![ret(not(is_nil(pending())))],
    ));
    // A type error, positioned, with the operand it is about noted for
    // the site to describe.
    let operand = local("operand", i64());
    d.push(define_cold(
        "zl_type_error",
        &[&message, &operand],
        unit(),
        vec![
            when(not(is_nil(pending())), vec![ret_void()]),
            set_global(VARINFO, operand.e()),
            set_global(
                PENDING,
                box_str(call("zl_position", vec![message.e()], string())),
            ),
            note_raise(),
            ret_void(),
        ],
    ));
    // Calling nil a lookup found for a method or a field: the message
    // names it, as the reference's does. `how` is 0 for a method, 1 for
    // a field.
    let callee = local("callee", string());
    let how = local("how", i64());
    d.push(define_cold(
        "zl_raise_call_nil",
        &[&callee, &how],
        unit(),
        vec![
            lua_error(concat(vec![
                text("attempt to call a nil value ("),
                if_expr(eq(how.e(), int(0)), text("method '"), text("field '")),
                callee.e(),
                text("')"),
            ])),
            ret_void(),
        ],
    ));
    // The pending type error with the description of the operand it is
    // about, `left` or `right`, appended; the note is consumed. A
    // call's error is described at a call site only, and an
    // operator's only at an operator's. Called with an error pending,
    // which it leaves pending.
    let left = kept("left", string());
    let right = kept("right", string());
    let calling = local("calling", boolean());
    let mark = local("mark", i64());
    let described = local("described", string());
    let tail = " has no integer representation";
    d.push(define_cold(
        "zl_annotate",
        &[&left, &right, &calling],
        unit(),
        vec![
            mark.decl(read_global(VARINFO, i64())),
            when(eq(mark.e(), int(0)), vec![ret_void()]),
            set_global(VARINFO, int(0)),
            when(
                ne(ne(bitand(mark.e(), int(VARINFO_CALL)), int(0)), calling.e()),
                vec![ret_void()],
            ),
            described.decl(if_expr(
                eq(bitand(mark.e(), int(3)), int(OPERAND_LEFT)),
                left.e(),
                right.e(),
            )),
            when(
                eq(call("zb_str_len", vec![described.e()], i64()), int(0)),
                vec![ret_void()],
            ),
            v.decl(pending()),
            s.decl(get_str(v.e())),
            when(
                ne(bitand(mark.e(), int(VARINFO_INSIDE)), int(0)),
                vec![
                    set_global(
                        PENDING,
                        box_str(concat(vec![
                            call(
                                "zb_str_substring",
                                vec![
                                    s.e(),
                                    int(0),
                                    sub(
                                        call("zb_str_len", vec![s.e()], i64()),
                                        int(tail.len() as i64),
                                    ),
                                ],
                                string(),
                            ),
                            described.e(),
                            text(tail),
                        ])),
                    ),
                    ret_void(),
                ],
            ),
            set_global(PENDING, box_str(add(s.e(), described.e()))),
            ret_void(),
        ],
    ));
    // `error(v, level)`: a string message at a level above zero is
    // positioned; anything else is the error value itself.
    d.push(define_cold(
        "zl_error",
        &[&v, &level],
        unit(),
        vec![
            // A stripped chunk's line is none to `error`.
            if_(
                and(
                    and(not(is_nil(v.e())), eq(category(v.e()), int(STR))),
                    and(
                        gt(level.e(), int(0)),
                        not(stripped_line(read_global(LINE, i64()))),
                    ),
                ),
                vec![expr(call(
                    "zl_raise_value",
                    vec![box_str(call("zl_position", vec![get_str(v.e())], string()))],
                    unit(),
                ))],
                vec![expr(call("zl_raise_value", vec![v.e()], unit()))],
            ),
            ret_void(),
        ],
    ));
    // `error(v, 2)`: the position is the caller's, the line the
    // function was entered at.
    d.push(define_cold(
        "zl_error_at",
        &[&v, &line],
        unit(),
        vec![
            if_(
                and(
                    and(not(is_nil(v.e())), eq(category(v.e()), int(STR))),
                    and(gt(line.e(), int(0)), not(stripped_line(line.e()))),
                ),
                vec![expr(call(
                    "zl_raise_value",
                    vec![box_str(call(
                        "zl_position_at",
                        vec![line.e(), get_str(v.e())],
                        string(),
                    ))],
                    unit(),
                ))],
                vec![expr(call("zl_raise_value", vec![v.e()], unit()))],
            ),
            ret_void(),
        ],
    ));
    // The error taken by whoever handles it: the value, then nothing
    // pending.
    d.push(define(
        "zl_take_pending",
        &[],
        any(),
        vec![
            v.decl(pending()),
            set_global(PENDING, nil()),
            set_global(OVERFLOWED, bool(false)),
            when(is_nil_error(v.e()), vec![ret(nil())]),
            ret(v.e()),
        ],
    ));
    // An error nothing caught, reported by the host the way `lua`
    // reports it; `zl_exit_uncaught` then ends the program with status
    // 1. The host formats the message itself, so the chunk's entry
    // reaches nothing of the library's `tostring`.
    d.push(extern_fn(
        "zl_report_uncaught",
        &[("err", any())],
        unit(),
        Some("$Lua$report_pending"),
    ));
    d.push(extern_fn(
        "zl_exit_uncaught",
        &[],
        unit(),
        Some("$Lua$exit_uncaught"),
    ));
    d.push(define_cold(
        "zl_report_pending",
        &[],
        unit(),
        vec![
            v.decl(pending()),
            // The state closes after the message, running every
            // pending finalizer, as `lua` closes it.
            when(
                is_nil(v.e()),
                vec![expr(call("zl_gc_at_exit", vec![], unit())), ret_void()],
            ),
            // `os.exit(code, true)`: the main thread's variables are
            // closed; the finalizers run and the program exits.
            when(
                is_exiting(v.e()),
                vec![
                    set_global(PENDING, nil()),
                    expr(call("zl_gc_at_exit", vec![], unit())),
                    expr(call("zl_io_flush_all", vec![], unit())),
                    expr(call("zb_exit", vec![cast(exit_code(v.e()), i32())], unit())),
                    ret_void(),
                ],
            ),
            set_global(PENDING, nil()),
            set_global(OVERFLOWED, bool(false)),
            expr(call("zl_report_uncaught", vec![v.e()], unit())),
            expr(call("zl_gc_at_exit", vec![], unit())),
            expr(call("zl_exit_uncaught", vec![], unit())),
            ret_void(),
        ],
    ));
    // The text an uncaught table error reports: what its `__tostring`
    // gives when that is a string, the error it raises if it raises one,
    // else nil. Reached only from the host, which compiles it when such
    // an error is reported.
    let h = local("h", any());
    let r = local("r", any());
    d.push(define_cold(
        ERROR_TEXT,
        &[&v],
        any(),
        vec![
            h.decl(call("zl_meta_of", vec![v.e(), text("__tostring")], any())),
            when(is_nil(h.e()), vec![ret(nil())]),
            r.decl(call(
                "zl_first",
                vec![call("zl_call_1", vec![h.e(), v.e()], any())],
                any(),
            )),
            when(
                not(is_nil(pending())),
                vec![ret(call("zl_take_pending", vec![], any()))],
            ),
            when(
                and(not(is_nil(r.e())), eq(category(r.e()), int(STR))),
                vec![ret(r.e())],
            ),
            ret(nil()),
        ],
    ));
    d
}
