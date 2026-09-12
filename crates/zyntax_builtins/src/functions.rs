//! Functions as values. A function value is a dynamic value under
//! [`FUNC_TAG`] holding a record: a list of dynamic values whose first
//! element is the code address, second the arity, and the rest the
//! variables the function shares with the scope that made it. The code
//! takes the record and every argument as dynamic values and returns
//! one, so a call through a value has one shape whatever it reaches.

use crate::build::*;
use crate::{list_of, CODE_TAG, FUNC_TAG};
use zyntax_typed_ast::type_registry::{AsyncKind, CallingConvention, NullabilityKind, ParamInfo};
use zyntax_typed_ast::{Type, TypeId};

/// The most arguments a call through a value passes.
pub const MAX_CALL_ARITY: usize = 8;

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

pub(crate) fn declarations(list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let mut d = vec![extern_fn(
        "zb_box_fnptr_raw",
        &[("f", i64()), ("tag", i32())],
        any(),
        Some("zyntax_box_ptr"),
    )];

    // A record from a code address, an arity and the shared cells.
    let code = local("code", i64());
    let arity = local("arity", i64());
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
            owned(name, any())
        })
        .collect();
    let fp_name = |n: usize| format!("zb_unbox_fnptr_raw_{n}");
    for n in 0..=MAX_CALL_ARITY {
        let code_ty = code_type(list_type, n);
        d.push(extern_fn(
            &fp_name(n),
            &[("x", any())],
            code_ty.clone(),
            Some("zyntax_box_get_opaque"),
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
                rec.decl(call("zb_func_unbox", vec![f.e()], anys.clone())),
                arity.decl(call(
                    "zb_box_get_i64",
                    vec![idx(rec.e(), int(1), any())],
                    i64(),
                )),
                when(
                    ne(arity.e(), int(n as i64)),
                    vec![fatal(
                        "TypeError",
                        add(
                            add(
                                text("function takes "),
                                call("zb_str_of_int", vec![arity.e()], string()),
                            ),
                            text(&format!(" positional arguments but {n} were given")),
                        ),
                    )],
                ),
                fp.decl(call(
                    &fp_name(n),
                    vec![idx(rec.e(), int(0), any())],
                    code_ty,
                )),
                ret(call("fp", call_args, any())),
            ],
        ));
    }
    d
}
