//! The library's side of the C API: the plugin's symbols it calls, and
//! the bridge the plugin calls back through.
//!
//! The bridge is a program of `lua$capi$*` functions, one per entry of
//! `zyntax_lua_capi::bridge::ENTRIES`, in that order and of those
//! arities, each a thin wrapper over the library's own paths so C code
//! indexes, calls and raises as Lua code does. It is compiled into the
//! running program the first time a native library opens. Signatures
//! carry only pointers and 64-bit integers.

use super::*;
use zyntax_builtins::functions::VARIADIC_ARITY;
use zyntax_typed_ast::typed_ast::{TypedImport, TypedImportItem};
use zyntax_typed_ast::{InternedString, Span, TypeRegistry, TypedNode, TypedProgram};

/// Set by a program's entry when it keeps its globals in the globals
/// table, where C code reaches them.
pub const SHARED: &str = "zl_shared_globals";

/// The plugin's symbols, as the library calls them.
pub(super) fn declarations() -> Vec<Decl> {
    vec![
        global_var(SHARED, i64()),
        extern_fn(
            "zl_c_loadlib",
            &[("path", string()), ("sym", string())],
            i64(),
            Some("$LuaC$loadlib"),
        ),
        extern_fn(
            "zl_c_loadfunc",
            &[("file", string()), ("name", string())],
            i64(),
            Some("$LuaC$loadfunc"),
        ),
        extern_fn("zl_c_loaded", &[], any(), Some("$LuaC$loaded")),
        extern_fn("zl_c_error", &[], string(), Some("$LuaC$error")),
        extern_fn("zl_ud_meta", &[("x", any())], any(), Some("$LuaC$ud_meta")),
        extern_fn(
            "zl_ud_set_meta",
            &[("x", any()), ("mt", any())],
            unit(),
            Some("$LuaC$ud_set_meta"),
        ),
    ]
}

/// The bridge's entry functions.
pub fn bridge_declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let o = kept("o", any());
    let k = kept("k", any());
    let v = kept("v", any());
    let a = kept("a", any());
    let b = kept("b", any());
    let f = kept("f", any());
    let h = kept("h", any());
    let co = kept("co", any());
    let xs = kept("xs", anys.clone());
    let i = local("i", i64());
    let n = local("n", i64());
    let op = local("op", i64());
    let arg = local("arg", i64());
    let code = local("code", i64());
    let cfn = local("cfn", i64());
    let acc = local("acc", any());
    let y = local("y", any());
    let rec = borrowed("rec", anys.clone());
    let cells = local("cells", anys.clone());
    let reg = local("reg", t.table());
    let live = local("live", i64());
    let entry = |name: &str| format!("lua$capi${name}");
    let as_int = |c: Expr| if_expr(c, int(1), int(0));
    let tb = |x: &Local| unbox_table(x.e(), t);
    let state_slot = coroutines::STATE_SLOT;
    vec![
        define(
            &entry("index"),
            &[&o, &k],
            any(),
            vec![ret(call("zl_index", vec![o.e(), k.e()], any()))],
        ),
        define(
            &entry("setindex"),
            &[&o, &k, &v],
            i64(),
            vec![
                expr(call("zl_setindex", vec![o.e(), k.e(), v.e()], unit())),
                ret(int(0)),
            ],
        ),
        define(
            &entry("rawget"),
            &[&o, &k],
            any(),
            vec![ret(call("zl_rawget", vec![tb(&o), k.e()], any()))],
        ),
        define(
            &entry("rawset"),
            &[&o, &k, &v],
            i64(),
            vec![
                expr(call("zl_rawset", vec![tb(&o), k.e(), v.e()], unit())),
                ret(int(0)),
            ],
        ),
        define(
            &entry("rawgeti"),
            &[&o, &i],
            any(),
            vec![ret(call("zl_rawgeti", vec![tb(&o), i.e()], any()))],
        ),
        define(
            &entry("rawseti"),
            &[&o, &i, &v],
            i64(),
            vec![
                expr(call("zl_rawseti", vec![tb(&o), i.e(), v.e()], unit())),
                ret(int(0)),
            ],
        ),
        define(
            &entry("next"),
            &[&o, &k],
            anys.clone(),
            vec![ret(call("zl_next", vec![tb(&o), k.e()], anys.clone()))],
        ),
        define(
            &entry("len"),
            &[&o],
            any(),
            vec![ret(call("zl_len_any", vec![o.e()], any()))],
        ),
        define(
            &entry("rawlen"),
            &[&o],
            i64(),
            vec![ret(call("zl_len", vec![tb(&o)], i64()))],
        ),
        // Right to left, as `..` associates.
        define(
            &entry("concat"),
            &[&xs],
            any(),
            vec![
                n.decl(len(xs.e())),
                when(eq(n.e(), int(0)), vec![ret(box_str(text("")))]),
                acc.decl(at(xs.e(), sub(n.e(), int(1)))),
                i.decl(sub(n.e(), int(2))),
                while_(
                    ge(i.e(), int(0)),
                    vec![
                        acc.set(call("zl_concat", vec![at(xs.e(), i.e()), acc.e()], any())),
                        when(not(is_nil(pending())), vec![ret(nil())]),
                        i.set(sub(i.e(), int(1))),
                    ],
                ),
                ret(acc.e()),
            ],
        ),
        define(
            &entry("arith"),
            &[&op, &a, &b],
            any(),
            vec![
                when(
                    eq(op.e(), int(values::OP_UNM)),
                    vec![ret(call("zl_unm", vec![a.e()], any()))],
                ),
                when(
                    eq(op.e(), int(values::OP_BNOT)),
                    vec![ret(call("zl_bnot", vec![a.e()], any()))],
                ),
                ret(call("zl_arith", vec![op.e(), a.e(), b.e()], any())),
            ],
        ),
        define(
            &entry("compare"),
            &[&a, &b, &op],
            i64(),
            vec![
                when(
                    eq(op.e(), int(0)),
                    vec![ret(as_int(call("zl_eq", vec![a.e(), b.e()], boolean())))],
                ),
                when(
                    eq(op.e(), int(1)),
                    vec![ret(as_int(call("zl_lt", vec![a.e(), b.e()], boolean())))],
                ),
                ret(as_int(call("zl_le", vec![a.e(), b.e()], boolean()))),
            ],
        ),
        define(
            &entry("rawequal"),
            &[&a, &b],
            i64(),
            vec![ret(as_int(call(
                "zl_rawequal",
                vec![a.e(), b.e()],
                boolean(),
            )))],
        ),
        define(
            &entry("getmetatable"),
            &[&o],
            any(),
            vec![ret(call("zl_debug_getmetatable", vec![o.e()], any()))],
        ),
        define(
            &entry("setmetatable"),
            &[&o, &v],
            i64(),
            vec![
                expr(call("zl_debug_setmetatable", vec![o.e(), v.e()], any())),
                ret(int(0)),
            ],
        ),
        define(
            &entry("call"),
            &[&f, &xs],
            any(),
            vec![ret(call("zl_call_packed", vec![f.e(), xs.e()], any()))],
        ),
        // `pcall`'s or `xpcall`'s tuple, or nil while a coroutine is
        // being closed.
        define(
            &entry("pcall"),
            &[&f, &xs, &h],
            any(),
            vec![
                when(
                    is_nil(h.e()),
                    vec![ret(call("zl_pcall", vec![f.e(), xs.e()], any()))],
                ),
                ret(call("zl_xpcall", vec![f.e(), h.e(), xs.e()], any())),
            ],
        ),
        define(
            &entry("raise"),
            &[&v],
            i64(),
            vec![
                expr(call("zl_raise_value", vec![v.e()], unit())),
                ret(int(0)),
            ],
        ),
        define(
            &entry("take"),
            &[],
            any(),
            vec![ret(call("zl_take_pending", vec![], any()))],
        ),
        define(&entry("pending"), &[], any(), vec![ret(pending())]),
        define(
            &entry("globals"),
            &[],
            any(),
            vec![ret(call("zl_globals_value", vec![], any()))],
        ),
        define(
            &entry("registry"),
            &[],
            any(),
            vec![ret(call("zl_debug_getregistry", vec![], any()))],
        ),
        define(
            &entry("current"),
            &[],
            any(),
            vec![ret(read_global(coroutines::CURRENT, any()))],
        ),
        define(
            &entry("main_thread"),
            &[],
            any(),
            vec![ret(call("zl_co_main", vec![], any()))],
        ),
        // The State a coroutine keeps past its other slots, 0 for none.
        define(
            &entry("thread_state"),
            &[&co],
            i64(),
            vec![
                rec.decl(call("zb_unbox_list_raw_any", vec![co.e()], anys.clone())),
                when(le(len(rec.e()), int(state_slot)), vec![ret(int(0))]),
                y.decl(at(rec.e(), int(state_slot))),
                when(is_nil(y.e()), vec![ret(int(0))]),
                ret(call("zb_unbox_instance_raw", vec![y.e()], i64())),
            ],
        ),
        define(
            &entry("set_thread_state"),
            &[&co, &v],
            i64(),
            vec![
                rec.decl(call("zb_unbox_list_raw_any", vec![co.e()], anys.clone())),
                while_(
                    lt(len(rec.e()), int(state_slot)),
                    vec![push(rec.e(), nil())],
                ),
                if_(
                    eq(len(rec.e()), int(state_slot)),
                    vec![push(rec.e(), v.e())],
                    vec![set_idx(rec.e(), int(state_slot), v.e())],
                ),
                ret(int(0)),
            ],
        ),
        // A variadic function value whose code is `code` and whose
        // record keeps the C function after the arity word.
        define(
            &entry("func_new"),
            &[&code, &cfn, &xs],
            any(),
            vec![
                cells.decl(list(
                    vec![call(
                        "zb_box_fnptr_raw",
                        vec![
                            cast(cfn.e(), usize()),
                            int32(zyntax_builtins::CODE_TAG as i32),
                        ],
                        any(),
                    )],
                    anys.clone(),
                )),
                expr(call("zb_list_extend_any", vec![cells.e(), xs.e()], unit())),
                ret(call(
                    "zb_func_new",
                    vec![cast(code.e(), usize()), int(VARIADIC_ARITY), cells.e()],
                    any(),
                )),
            ],
        ),
        define(
            &entry("list_new"),
            &[],
            anys.clone(),
            vec![ret(list(vec![], anys.clone()))],
        ),
        define(
            &entry("list_push"),
            &[&xs, &v],
            i64(),
            vec![push(xs.e(), v.e()), ret(int(0))],
        ),
        define(
            &entry("pack"),
            &[&xs],
            any(),
            vec![ret(call("zl_pack", vec![xs.e()], any()))],
        ),
        define(
            &entry("values"),
            &[&v],
            anys.clone(),
            vec![ret(call("zl_values", vec![v.e()], anys.clone()))],
        ),
        define(
            &entry("new_table"),
            &[],
            any(),
            vec![ret(box_table(call("zl_table_new", vec![], t.table())))],
        ),
        define(
            &entry("line"),
            &[],
            i64(),
            vec![ret(read_global(LINE, i64()))],
        ),
        define(
            &entry("set_line"),
            &[&i],
            i64(),
            vec![set_global(LINE, i.e()), ret(int(0))],
        ),
        define(
            &entry("chunk_of"),
            &[&i],
            string(),
            vec![ret(call("zl_chunk_of", vec![i.e()], string()))],
        ),
        define(
            &entry("shared"),
            &[],
            i64(),
            vec![ret(read_global(SHARED, i64()))],
        ),
        define(
            &entry("number_str"),
            &[&v],
            string(),
            vec![ret(call("zl_number_str", vec![v.e()], string()))],
        ),
        define(
            &entry("tonumber"),
            &[&v],
            any(),
            vec![ret(call("zl_tonumber", vec![v.e()], any()))],
        ),
        // `lua_gc` by its option codes (stop 0, restart 1, collect 2,
        // count 3, countb 4, step 5, setpause 6, setstepmul 7,
        // isrunning 9, generational 10, incremental 11), over the
        // collector `collectgarbage` drives; -1 while a finalizer runs.
        define(&entry("gc"), &[&op, &arg], i64(), {
            let opt = |name: &str| int(gc::option(name));
            let code = local("gcop", i64());
            let mut body = vec![code.decl(int(-1))];
            for (what, name) in [
                (0, "stop"),
                (1, "restart"),
                (2, "collect"),
                (3, "count"),
                (4, "count"),
                (5, "step"),
                (6, "setpause"),
                (7, "setstepmul"),
                (9, "isrunning"),
                (10, "generational"),
                (11, "incremental"),
            ] {
                body.push(when(eq(op.e(), int(what)), vec![code.set(opt(name))]));
            }
            body.extend([
                when(lt(code.e(), int(0)), vec![ret(int(-1))]),
                live.decl(call("zl_gc", vec![code.e(), arg.e()], i64())),
                when(eq(live.e(), int(-1)), vec![ret(int(-1))]),
                when(
                    or(eq(op.e(), int(2)), eq(op.e(), int(5))),
                    vec![
                        expr(call("zl_gc_finalize", vec![], unit())),
                        ret(if_expr(eq(op.e(), int(5)), live.e(), int(0))),
                    ],
                ),
                when(eq(op.e(), int(3)), vec![ret(div(live.e(), int(1024)))]),
                when(eq(op.e(), int(4)), vec![ret(rem(live.e(), int(1024)))]),
                // The mode it was in, as lua.h numbers modes.
                when(
                    or(eq(op.e(), int(10)), eq(op.e(), int(11))),
                    vec![ret(if_expr(ne(live.e(), int(0)), int(10), int(11)))],
                ),
                ret(live.e()),
            ]);
            body
        }),
        // The registry as the reference seeds it: the main thread at 1,
        // the globals at 2, and the tables loaders keep.
        define(
            &entry("init"),
            &[],
            any(),
            vec![
                reg.decl(unbox_table(call("zl_debug_getregistry", vec![], any()), t)),
                expr(call(
                    "zl_rawseti",
                    vec![reg.e(), int(1), call("zl_co_main", vec![], any())],
                    unit(),
                )),
                expr(call(
                    "zl_rawseti",
                    vec![reg.e(), int(2), call("zl_globals_value", vec![], any())],
                    unit(),
                )),
                expr(call(
                    "zl_rawset_str",
                    vec![
                        reg.e(),
                        text("_LOADED"),
                        call("zl_package_loaded", vec![], any()),
                    ],
                    unit(),
                )),
                expr(call(
                    "zl_rawset_str",
                    vec![
                        reg.e(),
                        text("_PRELOAD"),
                        call("zl_package_preload", vec![], any()),
                    ],
                    unit(),
                )),
                when(
                    is_nil(call("zl_rawget_str", vec![reg.e(), text("_CLIBS")], any())),
                    vec![expr(call(
                        "zl_rawset_str",
                        vec![
                            reg.e(),
                            text("_CLIBS"),
                            box_table(call("zl_table_new", vec![], t.table())),
                        ],
                        unit(),
                    ))],
                ),
                ret(box_table(reg.e())),
            ],
        ),
    ]
}

/// Every name the declarations refer to: the library's functions and
/// globals they reach, which the bridge program imports.
fn referenced(declarations: &[Decl]) -> std::collections::BTreeSet<String> {
    use zyntax_typed_ast::typed_ast::{TypedExpression as E, TypedStatement as S};
    fn e(x: &Expr, out: &mut std::collections::BTreeSet<String>) {
        match &x.node {
            E::Variable(n) => {
                if let Some(name) = n.resolve_global() {
                    out.insert(name);
                }
            }
            E::Call(c) => {
                e(&c.callee, out);
                for a in &c.positional_args {
                    e(a, out);
                }
            }
            E::MethodCall(m) => {
                e(&m.receiver, out);
                for a in &m.positional_args {
                    e(a, out);
                }
            }
            E::Binary(b) => {
                e(&b.left, out);
                e(&b.right, out);
            }
            E::Unary(u) => e(&u.operand, out),
            E::Index(i) => {
                e(&i.object, out);
                e(&i.index, out);
            }
            E::Field(f) => e(&f.object, out),
            E::Cast(c) => e(&c.expr, out),
            E::If(i) => {
                e(&i.condition, out);
                e(&i.then_branch, out);
                e(&i.else_branch, out);
            }
            E::Array(items) | E::Tuple(items) => {
                for a in items {
                    e(a, out);
                }
            }
            E::Block(b) => {
                for st in &b.statements {
                    s(st, out);
                }
            }
            _ => {}
        }
    }
    fn s(st: &Stmt, out: &mut std::collections::BTreeSet<String>) {
        match &st.node {
            S::Expression(x) => e(x, out),
            S::Let(l) => {
                if let Some(init) = &l.initializer {
                    e(init, out);
                }
            }
            S::Return(Some(x)) => e(x, out),
            S::If(i) => {
                e(&i.condition, out);
                for st in &i.then_block.statements {
                    s(st, out);
                }
                if let Some(b) = &i.else_block {
                    for st in &b.statements {
                        s(st, out);
                    }
                }
            }
            S::While(w) => {
                e(&w.condition, out);
                for st in &w.body.statements {
                    s(st, out);
                }
            }
            S::Block(b) => {
                for st in &b.statements {
                    s(st, out);
                }
            }
            _ => {}
        }
    }
    let mut out = std::collections::BTreeSet::new();
    for d in declarations {
        if let TypedDeclaration::Function(f) = &d.node
            && let Some(body) = &f.body
        {
            for st in &body.statements {
                s(st, &mut out);
            }
        }
    }
    out
}

/// The bridge as a program joined to the running one: its functions,
/// and an import of exactly what of the library they name.
pub fn bridge_program(t: &Types, type_registry: TypeRegistry, module: &str) -> TypedProgram {
    let mut declarations = bridge_declarations(t);
    // The program's own functions, not the library's: they are what
    // the program is entered through.
    for d in &mut declarations {
        if let TypedDeclaration::Function(f) = &mut d.node {
            f.module = None;
        }
    }
    let names = referenced(&declarations);
    let items = names
        .into_iter()
        .filter(|n| !n.starts_with("lua$capi$"))
        .map(|n| TypedImportItem::Named {
            name: InternedString::new_global(&n),
            alias: None,
        })
        .collect();
    declarations.push(TypedNode::new(
        TypedDeclaration::Import(TypedImport {
            language: Some(intern("lua")),
            module_path: vec![intern(module)],
            items,
            span: Span::new(0, 0),
        }),
        Type::Unknown,
        Span::new(0, 0),
    ));
    TypedProgram {
        declarations,
        language: Some(intern("lua")),
        span: Span::new(0, 0),
        source_files: Vec::new(),
        type_registry,
    }
}

/// The names of the bridge's entries, in the order they are defined.
pub fn bridge_entry_names(t: &Types) -> Vec<(String, usize)> {
    bridge_declarations(t)
        .iter()
        .filter_map(|d| match &d.node {
            TypedDeclaration::Function(f) => f.name.resolve_global().map(|n| (n, f.params.len())),
            _ => None,
        })
        .collect()
}
