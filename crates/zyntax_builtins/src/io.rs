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
    out.extend(host(list_type));
    out
}

/// What the program reads from the world around it: a line of input,
/// and the arguments it was started with. The host supplies the
/// arguments as `$Host$argc` and `$Host$argv`.
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
