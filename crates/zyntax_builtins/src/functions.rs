//! Functions as values. A function value is a dynamic value under
//! [`FUNC_TAG`] holding a record: a list of dynamic values whose first
//! element is the code address, second the arity, and the rest the
//! variables the function shares with the scope that made it. The code
//! takes the record and every argument as dynamic values and returns
//! one, so a call through a value has one shape whatever it reaches.

use crate::build::*;
use crate::{CODE_TAG, FUNC_TAG, list_of};
use zyntax_typed_ast::type_registry::{AsyncKind, CallingConvention, NullabilityKind, ParamInfo};
use zyntax_typed_ast::{Type, TypeId};

/// The most arguments a call through a value passes.
pub const MAX_CALL_ARITY: usize = 8;

/// A record whose code takes the record and a boxed list of arguments.
pub const VARIADIC_ARITY: i64 = -1;

/// The arity word of a record whose function takes `min` to `max`
/// arguments: the code takes `max`, and a call passing fewer fills the
/// rest with the missing-argument marker for the code to replace by
/// the defaults kept in the record. A word without the high half is a
/// function taking exactly `max`.
pub fn arity_word(min: usize, max: usize) -> i64 {
    if min == max {
        max as i64
    } else {
        (max as i64) | (((min as i64) + 1) << 16)
    }
}

/// The type of a function value's code: the record, then `arity`
/// dynamic arguments, to a dynamic result.
pub fn code_type(list_type: TypeId, arity: usize) -> Type {
    let param = |ty: Type| ParamInfo {
        name: None,
        ty,
        is_optional: false,
        is_varargs: false,
        is_keyword_only: false,
        is_positional_only: false,
        is_out: false,
        is_ref: false,
        is_inout: false,
    };
    let mut params = vec![param(list_of(list_type, any()))];
    params.extend((0..arity).map(|_| param(any())));
    Type::Function {
        params,
        return_type: Box::new(any()),
        is_varargs: false,
        has_named_params: false,
        has_default_params: false,
        async_kind: AsyncKind::Sync,
        calling_convention: CallingConvention::Default,
        nullability: NullabilityKind::NonNull,
    }
}

/// A fiber yielding dynamic values.
pub fn fiber_type() -> Type {
    Type::Fiber(Box::new(any()))
}

pub(crate) fn declarations(list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let mut d = vec![
        extern_fn(
            "zb_box_fnptr_raw",
            &[("f", usize()), ("tag", i32())],
            any(),
            Some("zyntax_box_ptr"),
        ),
        // The marker a call through a value passes for an argument it
        // leaves out: one address, compared by identity.
        extern_fn("zb_missing_arg", &[], any(), Some("$Host$missing_arg")),
    ];

    // A fiber body takes nothing; what it needs travels as an
    // environment it reads back when it starts.
    d.push(extern_fn(
        "zb_fiber_new_raw",
        &[("code", usize()), ("env", any()), ("stack", i64())],
        fiber_type(),
        Some("krio_fiber_new_with_env"),
    ));
    d.push(extern_fn(
        "zb_fiber_env",
        &[],
        any(),
        Some("krio_fiber_env"),
    ));
    // A fiber nothing will resume again: its stack and record go.
    d.push(extern_fn(
        "zb_fiber_free",
        &[("f", fiber_type())],
        unit(),
        Some("krio_fiber_free"),
    ));
    let code = local("code", usize());
    let env = kept("env", any());
    d.push(define(
        "zb_fiber_start",
        &[&code, &env],
        fiber_type(),
        vec![ret(call(
            "zb_fiber_new_raw",
            vec![code.e(), env.e(), int(0)],
            fiber_type(),
        ))],
    ));

    // A record from a code address, an arity and the shared cells.
    let code = local("code", usize());
    let arity = local("arity", i64());
    let most = local("most", i64());
    let least = local("least", i64());
    let cells = local("cells", anys.clone());
    let rec = local("rec", anys.clone());
    d.push(define(
        "zb_func_new",
        &[&code, &arity, &cells],
        any(),
        vec![
            rec.decl(list(
                vec![
                    call(
                        "zb_box_fnptr_raw",
                        vec![code.e(), int32(CODE_TAG as i32)],
                        any(),
                    ),
                    call("zb_box_i64", vec![arity.e()], any()),
                ],
                anys.clone(),
            )),
            expr(call("zb_list_extend_any", vec![rec.e(), cells.e()], unit())),
            ret(call(
                "zb_box_list_raw_any",
                vec![rec.e(), int32(FUNC_TAG as i32)],
                any(),
            )),
        ],
    ));

    let f = local("f", any());
    let tag = local("tag", i64());
    d.push(define(
        "zb_func_unbox",
        &[&f],
        anys.clone(),
        vec![
            tag.decl(cast(call("zb_box_tag", vec![f.e()], i32()), i64())),
            when(
                ne(tag.e(), int(FUNC_TAG)),
                vec![fatal(
                    "TypeError",
                    add(
                        add(text("'"), call("zb_any_type", vec![f.e()], string())),
                        text("' object is not callable"),
                    ),
                )],
            ),
            ret(call("zb_unbox_list_raw_any", vec![f.e()], anys.clone())),
        ],
    ));

    // One entry point per arity: check the arity, then jump to the code.
    // The code may keep or return any argument.
    let args: Vec<Local> = (0..MAX_CALL_ARITY)
        .map(|i| {
            let name: &'static str = Box::leak(format!("a{i}").into_boxed_str());
            kept(name, any())
        })
        .collect();
    let fp_name = |n: usize| format!("zb_unbox_fnptr_raw_{n}");
    for n in 0..=MAX_CALL_ARITY {
        let code_ty = code_type(list_type, n);
        d.push(extern_fn(
            &fp_name(n),
            &[("x", any())],
            code_ty.clone(),
            Some("zyntax_box_pointer"),
        ));
        let fp = local("fp", code_ty.clone());
        let mut params: Vec<&Local> = vec![&f];
        params.extend(args[..n].iter());
        let mut call_args = vec![rec.e()];
        call_args.extend(args[..n].iter().map(|a| a.e()));
        d.push(define(
            &format!("zb_call_{n}"),
            &params,
            any(),
            vec![
                // A foreign object is called by the embedder.
                when(
                    crate::foreign::is_foreign(f.e()),
                    vec![ret(call(
                        "zb_foreign_call",
                        vec![
                            f.e(),
                            list(args[..n].iter().map(|a| a.e()).collect(), anys.clone()),
                        ],
                        any(),
                    ))],
                ),
                rec.decl(call("zb_func_unbox", vec![f.e()], anys.clone())),
                arity.decl(call(
                    "zb_box_get_i64",
                    vec![idx(rec.e(), int(1), any())],
                    i64(),
                )),
                when(eq(arity.e(), int(VARIADIC_ARITY)), {
                    let packed_fp = local("packed_fp", code_type(list_type, 1));
                    vec![
                        packed_fp.decl(call(
                            &fp_name(1),
                            vec![idx(rec.e(), int(0), any())],
                            code_type(list_type, 1),
                        )),
                        ret(call(
                            "packed_fp",
                            vec![
                                rec.e(),
                                call(
                                    "zb_list_box_any",
                                    vec![list(
                                        args[..n].iter().map(|a| a.e()).collect(),
                                        anys.clone(),
                                    )],
                                    any(),
                                ),
                            ],
                            any(),
                        )),
                    ]
                }),
                // The word is `max`, with `min + 1` above bit 16 when
                // the function has defaults.
                most.decl(bitand(arity.e(), int(0xFFFF))),
                least.decl(shr(arity.e(), int(16))),
                if_(
                    eq(least.e(), int(0)),
                    vec![least.set(most.e())],
                    vec![least.set(sub(least.e(), int(1)))],
                ),
                when(
                    or(lt(int(n as i64), least.e()), gt(int(n as i64), most.e())),
                    vec![fatal(
                        "TypeError",
                        add(
                            add(
                                text("function takes "),
                                call("zb_str_of_int", vec![most.e()], string()),
                            ),
                            text(&format!(" positional arguments but {n} were given")),
                        ),
                    )],
                ),
            ]
            .into_iter()
            .chain((n + 1..=MAX_CALL_ARITY).map(|m| {
                // A function taking more than was passed gets the marker
                // for the rest.
                let wide_ty = code_type(list_type, m);
                let wide = local("wide", wide_ty.clone());
                let mut wide_args = vec![rec.e()];
                wide_args.extend(args[..n].iter().map(|a| a.e()));
                for _ in n..m {
                    wide_args.push(call("zb_missing_arg", vec![], any()));
                }
                when(
                    eq(most.e(), int(m as i64)),
                    vec![
                        wide.decl(call(
                            &fp_name(m),
                            vec![idx(rec.e(), int(0), any())],
                            wide_ty,
                        )),
                        ret(call("wide", wide_args, any())),
                    ],
                )
            }))
            .chain([
                fp.decl(call(
                    &fp_name(n),
                    vec![idx(rec.e(), int(0), any())],
                    code_ty,
                )),
                ret(call("fp", call_args, any())),
            ])
            .collect(),
        ));
    }
    d
}
