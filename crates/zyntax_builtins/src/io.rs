//! Printing, and ending the program on an error nothing can catch.

use crate::build::*;
use crate::Policy;

pub(crate) fn declarations(policy: &Policy) -> Vec<Decl> {
    let s = local("s", string());
    let kind = local("kind", string());
    let message = local("message", string());
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
    vec![
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
        define("zb_fatal", &[&kind, &message], unit(), fatal_body),
    ]
}
