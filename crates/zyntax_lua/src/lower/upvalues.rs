//! Upvalues for Lua lowering.

use super::*;

/// The name of a chunk's upvalue accessor.
pub(super) fn upvalue_accessor_name(tag: &str) -> String {
    format!("lua${tag}dbg$upvalues")
}

/// A chunk's upvalue accessor, as a function value's code: given a
/// function's key, the function, the upvalue's number, a value and
/// what is asked (`debug::UP_*`), it reads or writes the upvalue where
/// the function keeps it (a cell or a copy in its record, a module
/// variable, the environment), gives its identity, joins it to
/// another's cell or hands over its own.
pub(super) fn upvalue_accessor(module: &Module<'_>, span: Span) -> TypedFunction {
    use crate::library::debug::{UP_CELL, UP_GET, UP_ID, UP_JOIN, UP_SET};
    use crate::scope::Upvalue;
    let i64_t = prim(PrimitiveType::I64);
    let bool_t = prim(PrimitiveType::Bool);
    let anys = module.anys();
    let arg = |i: usize| var(intern(&format!("a{i}")), Type::Any, span);
    let int_of = |x: Node| call("zb_box_get_i64", vec![x], i64_t.clone(), span);
    let is = |x: Node, v: i64| binary(BinaryOp::Eq, x, int_lit(v, span), bool_t.clone(), span);
    let fid = || var(intern("fid"), i64_t.clone(), span);
    let n = || var(intern("n"), i64_t.clone(), span);
    let how = || var(intern("how"), i64_t.clone(), span);
    let rec = || var(intern("rec"), anys.clone(), span);
    let value = arg(3);
    let mut lowerer = Lowerer::new(module, CHUNK);
    // An upvalue's identity as a value: a light userdata, equal to
    // another exactly when the two identities are.
    let upvalue_id = |identity: Node| {
        call(
            "zb_box_instance_raw",
            vec![identity, int32_lit(library::light_tag() as i32, span)],
            Type::Any,
            span,
        )
    };
    // An identity for an upvalue with no cell: one per variable of the
    // chunk, never an address.
    let constant_id = |k: i64| upvalue_id(int_lit(((module.chunk_index + 1) << 40) | k, span));
    let mut body = vec![
        let_(
            intern("fid"),
            i64_t.clone(),
            binary(
                BinaryOp::BitAnd,
                int_of(arg(0)),
                int_lit(0xffff_ffff, span),
                i64_t.clone(),
                span,
            ),
            span,
        ),
        let_(
            intern("rec"),
            anys.clone(),
            call("zb_unbox_list_raw_any", vec![arg(1)], anys.clone(), span),
            span,
        ),
        let_(intern("n"), i64_t.clone(), int_of(arg(2)), span),
        let_(intern("how"), i64_t.clone(), int_of(arg(4)), span),
    ];
    for (k, info) in module.scopes.funcs.iter().enumerate() {
        if info.upvalues.is_empty() {
            continue;
        }
        let mut arms = Vec::new();
        for (i, up) in info.upvalues.iter().enumerate() {
            let mut answers: Vec<St> = Vec::new();
            match up {
                Upvalue::Var(v) => {
                    let vinfo = module.scopes.var(*v);
                    if vinfo.is_module_var() {
                        let ty = module.inferred.var(*v).settled();
                        let symbol = module.module_local_symbol(*v);
                        let read = lowerer.coerce(
                            Val {
                                node: var(symbol, module.ir(ty), span),
                                ty,
                            },
                            Ty::Any,
                        );
                        let stored = lowerer.coerce(
                            Val {
                                node: value.clone(),
                                ty: Ty::Any,
                            },
                            ty,
                        );
                        answers.push(if_(
                            is(how(), UP_GET),
                            vec![ret(Some(read), span)],
                            None,
                            span,
                        ));
                        answers.push(if_(
                            is(how(), UP_SET),
                            vec![assign(var(symbol, module.ir(ty), span), stored, span)],
                            None,
                            span,
                        ));
                        answers.push(if_(
                            is(how(), UP_ID),
                            vec![ret(Some(constant_id(v.0 as i64 + 1)), span)],
                            None,
                            span,
                        ));
                    } else if let Some(j) = info.captures.iter().position(|c| c == v) {
                        let slot = || {
                            index(
                                rec(),
                                int_lit((RECORD_CELLS_AT + j) as i64, span),
                                Type::Any,
                                span,
                            )
                        };
                        if vinfo.needs_cell() {
                            let cell = || {
                                index(
                                    call("zb_unbox_list_raw_any", vec![slot()], anys.clone(), span),
                                    int_lit(0, span),
                                    Type::Any,
                                    span,
                                )
                            };
                            answers.push(if_(
                                is(how(), UP_GET),
                                vec![ret(Some(cell()), span)],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                is(how(), UP_SET),
                                vec![assign(cell(), value.clone(), span)],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                is(how(), UP_ID),
                                vec![ret(
                                    Some(upvalue_id(call(
                                        "zb_unbox_instance_raw",
                                        vec![slot()],
                                        i64_t.clone(),
                                        span,
                                    ))),
                                    span,
                                )],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                binary(
                                    BinaryOp::And,
                                    is(how(), UP_JOIN),
                                    binary(
                                        BinaryOp::Ne,
                                        value.clone(),
                                        nil(span),
                                        bool_t.clone(),
                                        span,
                                    ),
                                    bool_t.clone(),
                                    span,
                                ),
                                vec![assign(slot(), value.clone(), span)],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                is(how(), UP_CELL),
                                vec![ret(Some(slot()), span)],
                                None,
                                span,
                            ));
                        } else {
                            // A chunk that names `debug.upvalueid` keeps
                            // every capture in a cell, so a copy is asked
                            // its identity only from another chunk: it
                            // answers the variable's, one for all its
                            // closure instances.
                            answers.push(if_(
                                is(how(), UP_GET),
                                vec![ret(Some(slot()), span)],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                is(how(), UP_SET),
                                vec![assign(slot(), value.clone(), span)],
                                None,
                                span,
                            ));
                            answers.push(if_(
                                is(how(), UP_ID),
                                vec![ret(Some(constant_id(v.0 as i64 + 1)), span)],
                                None,
                                span,
                            ));
                        }
                    }
                }
                Upvalue::Env => {
                    let env = match module.env_var {
                        Some(env) => var(env, Type::Any, span),
                        None => call("zl_globals_value", vec![], Type::Any, span),
                    };
                    answers.push(if_(
                        is(how(), UP_GET),
                        vec![ret(Some(env), span)],
                        None,
                        span,
                    ));
                    if let Some(env) = module.env_var {
                        answers.push(if_(
                            is(how(), UP_SET),
                            vec![assign(var(env, Type::Any, span), value.clone(), span)],
                            None,
                            span,
                        ));
                    }
                    answers.push(if_(
                        is(how(), UP_ID),
                        vec![ret(Some(constant_id(0)), span)],
                        None,
                        span,
                    ));
                }
            }
            answers.push(ret(Some(nil(span)), span));
            arms.push(if_(is(n(), i as i64 + 1), answers, None, span));
        }
        body.push(if_(is(fid(), k as i64), arms, None, span));
    }
    body.push(ret(Some(nil(span)), span));
    let mut params = vec![parameter(intern("env"), anys, span)];
    params.extend(
        (0..library::debug::UP_ARITY as usize)
            .map(|i| parameter(intern(&format!("a{i}")), Type::Any, span)),
    );
    typed_function(
        &upvalue_accessor_name(&module.tag),
        params,
        Type::Any,
        body,
        span,
    )
}
