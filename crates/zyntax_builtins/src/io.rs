//! Printing, reading, the program's arguments, and ending the program on
//! an error nothing can catch.

use crate::build::*;
use crate::{list_of, Policy};
use zyntax_typed_ast::TypeId;

pub(crate) fn declarations(policy: &Policy, list_type: TypeId) -> Vec<Decl> {
    let s = local("s", string());
    // The exception the frontend raises keeps the message, so it is the
    // callee's from the call on.
    let kind = local("kind", string());
    let message = kept("message", string());
    // A frontend with exceptions turns the error into one and gets
    // control back; otherwise the program ends the way an uncaught
    // exception ends it: the kind and message on stderr, and status 1.
    let fatal_body = if policy.exceptions {
        vec![
            expr(call("zb_hook_raise", vec![kind.e(), message.e()], unit())),
            ret_void(),
        ]
    } else {
        vec![
            expr(call(
                "zb_eprintln",
                vec![text("Traceback (most recent call last):")],
                unit(),
            )),
            expr(call(
                "zb_eprintln",
                vec![add(add(kind.e(), text(": ")), message.e())],
                unit(),
            )),
            expr(call("zb_exit", vec![int32(1)], unit())),
            ret_void(),
        ]
    };
    let mut out = vec![
        extern_fn(
            "zb_println",
            &[("v", any())],
            unit(),
            Some("$IO$println_dynamic"),
        ),
        extern_fn(
            "zb_print",
            &[("v", any())],
            unit(),
            Some("$IO$print_dynamic"),
        ),
        extern_fn(
            "zb_eprintln",
            &[("v", any())],
            unit(),
            Some("$IO$eprintln_dynamic"),
        ),
        extern_fn(
            "zb_format_dynamic",
            &[("v", any())],
            string(),
            Some("$IO$format_dynamic"),
        ),
        extern_fn("zb_exit", &[("code", i32())], unit(), Some("exit")),
        // One line of text to stdout, and text with no line break.
        define(
            "zb_print_line",
            &[&s],
            unit(),
            vec![expr(call("zb_println", vec![s.e()], unit())), ret_void()],
        ),
        define(
            "zb_print_text",
            &[&s],
            unit(),
            vec![expr(call("zb_print", vec![s.e()], unit())), ret_void()],
        ),
        define_cold("zb_fatal", &[&kind, &message], unit(), fatal_body),
    ];
    // An exception nothing caught ends the program: its kind and text on
    // stderr, status 1. Cold, so the formatting it reaches stays out of
    // the program's own code.
    let exc = local("exc", any());
    let shown = local("shown", string());
    let type_name = call("zb_any_type", vec![exc.e()], string());
    out.push(define_cold(
        "zb_uncaught",
        &[&exc],
        unit(),
        vec![
            shown.decl(call("zb_any_str", vec![exc.e()], string())),
            expr(call(
                "zb_eprintln",
                vec![text("Traceback (most recent call last):")],
                unit(),
            )),
            if_(
                call("zb_str_truthy", vec![shown.e()], boolean()),
                vec![expr(call(
                    "zb_eprintln",
                    vec![add(add(type_name.clone(), text(": ")), shown.e())],
                    unit(),
                ))],
                vec![expr(call("zb_eprintln", vec![type_name], unit()))],
            ),
            expr(call("zb_exit", vec![int32(1)], unit())),
            ret_void(),
        ],
    ));
    out.extend(host(list_type));
    out
}

/// What the program reads from the world around it: a line of input,
/// the arguments it was started with, and the clocks. The host supplies
/// the arguments as `$Host$argc` and `$Host$argv`, the clocks as
/// `$Host$time` and `$Host$perf_counter`.
fn host(list_type: TypeId) -> Vec<Decl> {
    let strs = list_of(list_type, string());
    let prompt = local("prompt", string());
    let line = local("line", string());
    let out = local("out", strs.clone());
    let i = local("i", i64());
    let n = local("n", i64());
    let mut d = vec![
        extern_fn("zb_read_line_raw", &[], string(), Some("$IO$read_line")),
        extern_fn(
            "zb_input_raw",
            &[("prompt", string())],
            string(),
            Some("$IO$input"),
        ),
        extern_fn("zb_host_argc", &[], i64(), Some("$Host$argc")),
        extern_fn(
            "zb_host_argv",
            &[("i", i64())],
            string(),
            Some("$Host$argv"),
        ),
        // The clocks: seconds since the epoch, and seconds on a clock
        // that only goes forward, for timing.
        extern_fn("zb_time_time", &[], f64(), Some("$Host$time")),
        extern_fn(
            "zb_time_perf_counter",
            &[],
            f64(),
            Some("$Host$perf_counter"),
        ),
    ];
    // The end of input is an error to a program that asked for a line.
    d.push(define(
        "zb_input",
        &[&prompt],
        string(),
        vec![
            line.decl(call("zb_input_raw", vec![prompt.e()], string())),
            when(
                eq(line.e(), null(string())),
                vec![fatal("EOFError", text("EOF when reading a line"))],
            ),
            ret(line.e()),
        ],
    ));
    d.push(define(
        "zb_read_line",
        &[],
        string(),
        vec![
            line.decl(call("zb_read_line_raw", vec![], string())),
            when(
                eq(line.e(), null(string())),
                vec![fatal("EOFError", text("EOF when reading a line"))],
            ),
            ret(line.e()),
        ],
    ));
    // `sys.version_info`: the Python this frontend speaks, as the
    // tuple `(major, minor, micro, releaselevel, serial)`.
    let anys = list_of(list_type, any());
    let info = local("info", anys.clone());
    d.push(define(
        "zb_sys_version_info",
        &[],
        anys.clone(),
        vec![
            info.decl(list(Vec::new(), anys.clone())),
            expr(mcall(
                info.e(),
                "push",
                vec![call("zb_box_i64", vec![int(3)], any())],
                unit(),
            )),
            expr(mcall(
                info.e(),
                "push",
                vec![call("zb_box_i64", vec![int(12)], any())],
                unit(),
            )),
            expr(mcall(
                info.e(),
                "push",
                vec![call("zb_box_i64", vec![int(0)], any())],
                unit(),
            )),
            expr(mcall(
                info.e(),
                "push",
                vec![call("zb_box_str", vec![text("final")], any())],
                unit(),
            )),
            expr(mcall(
                info.e(),
                "push",
                vec![call("zb_box_i64", vec![int(0)], any())],
                unit(),
            )),
            ret(info.e()),
        ],
    ));
    d.push(define("zb_sys_argv", &[], strs.clone(), {
        let mut s = vec![
            out.decl(list(Vec::new(), strs.clone())),
            n.decl(call("zb_host_argc", vec![], i64())),
        ];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(mcall(
                out.e(),
                "push",
                vec![call("zb_host_argv", vec![i.e()], string())],
                unit(),
            ))],
        ));
        s.push(ret(out.e()));
        s
    }));
    d
}
