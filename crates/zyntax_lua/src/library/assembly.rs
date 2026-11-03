//! Assembly support for the Lua library.

use super::*;

/// The whole library for Lua: the shared library under Lua's
/// spellings, the table struct, and everything in this module.
pub fn library(policy: &zyntax_builtins::Policy) -> (zyntax_builtins::Library, Types) {
    let mut lib = zyntax_builtins::library(policy);
    let table_type = declare_table_type(&mut lib.type_registry);
    let t = Types {
        list_type: lib.list_type,
        table_type,
    };
    lib.declarations.push(table_class(table_type));
    lib.declarations.extend(shape_hook_decls(&t));
    // No Lua value is a dict or set of a shape.
    lib.declarations
        .extend(zyntax_builtins::lists::keyed_hook_stubs(t.list_type));
    lib.declarations.extend(instance_hooks(&t));
    lib.declarations.extend(raising(&t));
    lib.declarations.extend(tables::declarations(&t));
    lib.declarations.extend(values::declarations(policy, &t));
    lib.declarations.extend(calls::declarations(&t));
    lib.declarations.extend(coroutines::declarations(&t));
    lib.declarations.extend(patterns::declarations(&t));
    lib.declarations.extend(utf8::declarations(&t));
    lib.declarations.extend(io::declarations(&t));
    lib.declarations.extend(stdlib::declarations(policy, &t));
    lib.declarations.extend(debug::declarations(&t));
    lib.declarations.extend(foreign::declarations(&t));
    lib.declarations.extend(gc::declarations(&t));
    lib.declarations.extend(capi::declarations());
    lib.declarations.push(func_code_decl(&t));
    lib.declarations.extend(type_names());
    for d in &mut lib.declarations {
        if let TypedDeclaration::Function(f) = &mut d.node {
            f.annotations.push(strict_fp());
            // Library entries may run while an error is reported or unwound.
            f.annotations.push(TypedAnnotation {
                name: intern("queries_error_flag"),
                args: Vec::new(),
                span: SPAN,
            });
            // These return with an error pending on every path: the
            // pending one stays, else they set one.
            if f.name.resolve_global().is_some_and(|n| {
                matches!(
                    n.as_str(),
                    "zl_type_error" | "zl_raise_value" | "zb_hook_raise" | "zl_annotate"
                )
            }) {
                f.annotations.push(TypedAnnotation {
                    name: intern("sets_error_flag"),
                    args: Vec::new(),
                    span: SPAN,
                });
            }
        }
    }
    lib.fallible = fallible_functions(&lib.declarations);
    (lib, t)
}

/// Every library function that may run the program's code before it
/// returns, through any number of calls: one that calls through a
/// code pointer, a local rather than a function's name, as the
/// callers of function values and the metamethod paths do.
pub fn reentrant_functions(declarations: &[Decl]) -> std::collections::BTreeSet<String> {
    use std::collections::{BTreeMap, BTreeSet};
    let declared: BTreeSet<String> = declarations
        .iter()
        .filter_map(|d| match &d.node {
            TypedDeclaration::Function(f) => f.name.resolve_global(),
            _ => None,
        })
        .collect();
    let mut calls: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut reentrant = BTreeSet::new();
    for d in declarations {
        if let TypedDeclaration::Function(f) = &d.node
            && let (Some(name), Some(body)) = (f.name.resolve_global(), &f.body)
        {
            let mut callees = BTreeSet::new();
            for s in &body.statements {
                callee_names(s, &mut callees);
            }
            if callees.iter().any(|c| !declared.contains(c)) {
                reentrant.insert(name.clone());
            }
            calls.insert(name, callees);
        }
    }
    loop {
        let before = reentrant.len();
        for (name, callees) in &calls {
            if callees.iter().any(|c| reentrant.contains(c)) {
                reentrant.insert(name.clone());
            }
        }
        if reentrant.len() == before {
            break;
        }
    }
    reentrant
}

/// The names called anywhere in a statement, block expressions and
/// all: what a lowered function calls.
pub fn callee_names(stmt: &Stmt, out: &mut std::collections::BTreeSet<String>) {
    use zyntax_typed_ast::typed_ast::{TypedExpression as E, TypedStatement as S};
    fn expr(e: &Expr, out: &mut std::collections::BTreeSet<String>) {
        match &e.node {
            E::Call(c) => {
                if let E::Variable(n) = &c.callee.node
                    && let Some(name) = n.resolve_global()
                {
                    out.insert(name);
                } else {
                    expr(&c.callee, out);
                }
                for a in &c.positional_args {
                    expr(a, out);
                }
            }
            E::MethodCall(m) => {
                expr(&m.receiver, out);
                for a in &m.positional_args {
                    expr(a, out);
                }
            }
            E::Binary(b) => {
                expr(&b.left, out);
                expr(&b.right, out);
            }
            E::Unary(u) => expr(&u.operand, out),
            E::Index(i) => {
                expr(&i.object, out);
                expr(&i.index, out);
            }
            E::Field(f) => expr(&f.object, out),
            E::Cast(c) => expr(&c.expr, out),
            E::If(i) => {
                expr(&i.condition, out);
                expr(&i.then_branch, out);
                expr(&i.else_branch, out);
            }
            E::Array(items) | E::Tuple(items) => {
                for a in items {
                    expr(a, out);
                }
            }
            E::Struct(st) => {
                for f in &st.fields {
                    expr(&f.value, out);
                }
            }
            E::Block(b) => {
                for s in &b.statements {
                    callee_names(s, out);
                }
            }
            _ => {}
        }
    }
    match &stmt.node {
        S::Expression(e) => expr(e, out),
        S::Let(l) => {
            if let Some(init) = &l.initializer {
                expr(init, out);
            }
        }
        S::Return(Some(e)) => expr(e, out),
        S::If(i) => {
            expr(&i.condition, out);
            for s in &i.then_block.statements {
                callee_names(s, out);
            }
            if let Some(e) = &i.else_block {
                for s in &e.statements {
                    callee_names(s, out);
                }
            }
        }
        S::While(w) => {
            expr(&w.condition, out);
            for s in &w.body.statements {
                callee_names(s, out);
            }
        }
        S::Block(b) => {
            for s in &b.statements {
                callee_names(s, out);
            }
        }
        _ => {}
    }
}

/// The names the error messages give a value's type, built only when
/// an error is raised. A `__name` is read raw and used only when it is
/// a string.
pub(super) fn type_names() -> Vec<Decl> {
    let x = kept("x", any());
    let h = kept("h", any());
    let named = |h: &Local| and(not(is_nil(h.e())), eq(category(h.e()), int(STR)));
    vec![
        define_cold(
            "zl_objtypename",
            &[&x],
            string(),
            vec![
                when(
                    is_table(x.e()),
                    vec![
                        h.decl(call("zl_meta_of", vec![x.e(), text("__name")], any())),
                        when(named(&h), vec![ret(get_str(h.e()))]),
                    ],
                ),
                ret(type_name(x.e())),
            ],
        ),
        define_cold(
            "zl_argtypename",
            &[&x],
            string(),
            vec![
                h.decl(call("zl_meta_of", vec![x.e(), text("__name")], any())),
                when(named(&h), vec![ret(get_str(h.e()))]),
                when(is_light(x.e()), vec![ret(text("light userdata"))]),
                ret(type_name(x.e())),
            ],
        ),
    ]
}

/// `zl_func_id(f)`: the number of the program function a function
/// value is, or -1 for any other value, for a call site that knows
/// which functions a method may be and calls them directly. A
/// function of the program's making has its number after its code
/// and arity; the library's values carry no number there.
pub(super) fn func_code_decl(t: &Types) -> Decl {
    let f = kept("f", any());
    let rec = borrowed("rec", t.anys());
    let slot = kept("slot", any());
    define(
        "zl_func_id",
        &[&f],
        i64(),
        vec![
            when(not(is_func(f.e())), vec![ret(int(-1))]),
            rec.decl(call("zb_unbox_list_raw_any", vec![f.e()], t.anys())),
            when(le(len(rec.e()), int(2)), vec![ret(int(-1))]),
            slot.decl(at(rec.e(), int(2))),
            when(
                or(is_nil(slot.e()), ne(category(slot.e()), int(INT))),
                vec![ret(int(-1))],
            ),
            ret(get_i64(slot.e())),
        ],
    )
}

/// Lua's floats round at every operation: no multiply and add of the
/// program's, or of the library's, is fused into one rounding.
pub fn strict_fp() -> TypedAnnotation {
    TypedAnnotation {
        name: intern("strict_fp"),
        args: Vec::new(),
        span: SPAN,
    }
}

/// Every function that raises, through any number of calls: one that
/// reaches the shared library's `zb_fatal`, or Lua's `zl_raise_value`
/// or `zl_type_error`.
pub(super) fn fallible_functions(declarations: &[Decl]) -> std::collections::BTreeSet<String> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut calls: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for d in declarations {
        if let TypedDeclaration::Function(f) = &d.node
            && let (Some(name), Some(body)) = (f.name.resolve_global(), &f.body)
        {
            let mut callees = BTreeSet::new();
            for s in &body.statements {
                callee_names(s, &mut callees);
            }
            calls.insert(name, callees);
        }
    }
    let mut fallible: BTreeSet<String> = [
        "zb_fatal".to_string(),
        "zl_raise_value".to_string(),
        "zl_type_error".to_string(),
    ]
    .into_iter()
    .collect();
    loop {
        let before = fallible.len();
        for (name, callees) in &calls {
            if callees.iter().any(|c| fallible.contains(c)) {
                fallible.insert(name.clone());
            }
        }
        if fallible.len() == before {
            break;
        }
    }
    fallible
}
