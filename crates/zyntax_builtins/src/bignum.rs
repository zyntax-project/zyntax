//! The big-int library's magnitude operations, as the Python library
//! calls them: lists of 30-bit limbs in, lists of limbs out, the work
//! done by the host over the lists' storage.

use crate::build::*;
use crate::list_of;
use zyntax_typed_ast::TypeId;

pub(crate) fn declarations(list_type: TypeId) -> Vec<Decl> {
    let ints = list_of(list_type, i64());
    let mut d = Vec::new();

    // The host's functions, which take each list as its data address
    // and length.
    for (name, params, ret, symbol) in [
        (
            "zb_big_cmp_raw",
            vec!["a", "an", "b", "bn"],
            i64(),
            "$Host$big_cmp",
        ),
        (
            "zb_big_add_raw",
            vec!["a", "an", "b", "bn", "out", "on"],
            unit(),
            "$Host$big_add",
        ),
        (
            "zb_big_sub_raw",
            vec!["a", "an", "b", "bn", "out", "on"],
            unit(),
            "$Host$big_sub",
        ),
        (
            "zb_big_mul_raw",
            vec!["a", "an", "b", "bn", "out", "on"],
            unit(),
            "$Host$big_mul",
        ),
        (
            "zb_big_divmod_small_raw",
            vec!["a", "an", "d", "q", "qn"],
            i64(),
            "$Host$big_divmod_small",
        ),
        (
            "zb_big_shl_raw",
            vec!["a", "an", "n", "out", "on"],
            unit(),
            "$Host$big_shl",
        ),
        (
            "zb_big_shr_raw",
            vec!["a", "an", "n", "out", "on"],
            unit(),
            "$Host$big_shr",
        ),
        (
            "zb_big_divmod_raw",
            vec!["a", "an", "b", "bn", "q", "qn", "r", "rn"],
            unit(),
            "$Host$big_divmod",
        ),
        (
            "zb_big_mul_small_add_raw",
            vec!["m", "n", "k", "add"],
            i64(),
            "$Host$big_mul_small_add",
        ),
    ] {
        let params: Vec<(&str, _)> = params.into_iter().map(|p| (p, i64())).collect();
        d.push(extern_fn(name, &params, ret, Some(symbol)));
    }

    let a = borrowed("a", ints.clone());
    let b = borrowed("b", ints.clone());
    let out = borrowed("out", ints.clone());
    let q = borrowed("q", ints.clone());
    let r = borrowed("r", ints.clone());
    let m = borrowed("m", ints.clone());
    let n = local("n", i64());
    let k = local("k", i64());
    let dv = local("d", i64());
    let add_in = local("add", i64());
    let data = |l: &Local| fld(l.e(), "data", i64());
    let size = |l: &Local| mcall(l.e(), "len", vec![], i64());

    d.push(define(
        "zb_big_cmp",
        &[&a, &b],
        i64(),
        vec![ret(call(
            "zb_big_cmp_raw",
            vec![data(&a), size(&a), data(&b), size(&b)],
            i64(),
        ))],
    ));
    for (name, raw) in [
        ("zb_big_add", "zb_big_add_raw"),
        ("zb_big_sub", "zb_big_sub_raw"),
        ("zb_big_mul", "zb_big_mul_raw"),
    ] {
        d.push(define(
            name,
            &[&a, &b, &out],
            unit(),
            vec![expr(call(
                raw,
                vec![
                    data(&a),
                    size(&a),
                    data(&b),
                    size(&b),
                    data(&out),
                    size(&out),
                ],
                unit(),
            ))],
        ));
    }
    d.push(define(
        "zb_big_divmod_small",
        &[&a, &dv, &q],
        i64(),
        vec![ret(call(
            "zb_big_divmod_small_raw",
            vec![data(&a), size(&a), dv.e(), data(&q), size(&q)],
            i64(),
        ))],
    ));
    for (name, raw) in [
        ("zb_big_shl", "zb_big_shl_raw"),
        ("zb_big_shr", "zb_big_shr_raw"),
    ] {
        d.push(define(
            name,
            &[&a, &n, &out],
            unit(),
            vec![expr(call(
                raw,
                vec![data(&a), size(&a), n.e(), data(&out), size(&out)],
                unit(),
            ))],
        ));
    }
    d.push(define(
        "zb_big_divmod",
        &[&a, &b, &q, &r],
        unit(),
        vec![expr(call(
            "zb_big_divmod_raw",
            vec![
                data(&a),
                size(&a),
                data(&b),
                size(&b),
                data(&q),
                size(&q),
                data(&r),
                size(&r),
            ],
            unit(),
        ))],
    ));
    d.push(define(
        "zb_big_mul_small_add",
        &[&m, &k, &add_in],
        i64(),
        vec![ret(call(
            "zb_big_mul_small_add_raw",
            vec![data(&m), size(&m), k.e(), add_in.e()],
            i64(),
        ))],
    ));
    d
}
