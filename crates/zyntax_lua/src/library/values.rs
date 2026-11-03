//! Lua's operators on dynamic values: arithmetic with the integer and
//! float subtypes and string coercion, comparison, concatenation,
//! length, truth, and the conversions `tostring` and `tonumber`. Each
//! falls back to the metamethod of the event it implements.

use super::*;

/// The operator codes `zl_arith` takes, in the order of [`EVENTS`].
pub const OP_ADD: i64 = 0;
pub const OP_SUB: i64 = 1;
pub const OP_MUL: i64 = 2;
pub const OP_DIV: i64 = 3;
pub const OP_MOD: i64 = 4;
pub const OP_POW: i64 = 5;
pub const OP_IDIV: i64 = 6;
pub const OP_BAND: i64 = 7;
pub const OP_BOR: i64 = 8;
pub const OP_BXOR: i64 = 9;
pub const OP_SHL: i64 = 10;
pub const OP_SHR: i64 = 11;
pub const OP_CONCAT: i64 = 12;
pub const OP_UNM: i64 = 13;
pub const OP_BNOT: i64 = 14;

/// The metamethod each operator resolves to.
pub const EVENTS: [&str; 15] = [
    "__add", "__sub", "__mul", "__div", "__mod", "__pow", "__idiv", "__band", "__bor", "__bxor",
    "__shl", "__shr", "__concat", "__unm", "__bnot",
];

mod arithmetic;
mod comparison;
mod conversions;
mod strings;

struct Context {
    x: Local,
    a: Local,
    b: Local,
    op: Local,
    ca: Local,
    cb: Local,
    cat: Local,
    ia: Local,
    ib: Local,
    fa: Local,
    fb: Local,
    f: Local,
    i: Local,
    n: Local,
    s: Local,
    h: Local,
    r: Local,
    na: Local,
    nb: Local,
}

pub(super) fn declarations(_policy: &zyntax_builtins::Policy, t: &Types) -> Vec<Decl> {
    let x = kept("x", any());
    let a = kept("a", any());
    let b = kept("b", any());
    let op = local("op", i64());
    let ca = local("ca", i64());
    let cb = local("cb", i64());
    let cat = local("cat", i64());
    let ia = local("ia", i64());
    let ib = local("ib", i64());
    let fa = local("fa", f64());
    let fb = local("fb", f64());
    let f = local("f", f64());
    let i = local("i", i64());
    let n = local("n", i64());
    let s = kept("s", string());
    let h = kept("h", any());
    let r = kept("r", any());
    let na = kept("na", any());
    let nb = kept("nb", any());
    let mut d = Vec::new();
    let context = Context {
        x,
        a,
        b,
        op,
        ca,
        cb,
        cat,
        ia,
        ib,
        fa,
        fb,
        f,
        i,
        n,
        s,
        h,
        r,
        na,
        nb,
    };
    conversions::append(&context, t, &mut d);
    arithmetic::append(&context, t, &mut d);
    comparison::append(&context, t, &mut d);
    strings::append(&context, t, &mut d);
    d
}

fn is_int_cat_of(x: Expr) -> Expr {
    let c = category(x);
    or(eq(c.clone(), int(INT)), eq(c, int(UINT)))
}
