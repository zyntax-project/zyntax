//! The dict and set library run, not only built: each case is a
//! function written with the library's own builder, compiled with the
//! library and called.

use zyntax_builtins::build::*;
use zyntax_builtins::dicts::{
    dict_declarations, dict_entry_type, dict_shape_tag, dict_type, set_declarations,
    set_entry_type, set_shape_tag,
};
use zyntax_builtins::lists::Field;
use zyntax_builtins::{Policy, TypeNames, library, list_of};
use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};
use zyntax_typed_ast::{Span, TypeId, TypedProgram};

fn policy() -> Policy {
    Policy {
        true_text: "True",
        false_text: "False",
        none_text: "None",
        single_quotes: true,
        float_fraction: true,
        instance_hooks: false,
        exceptions: false,
        bool_is_number: true,
        type_names: TypeNames {
            none: "NoneType",
            bool: "bool",
            int: "int",
            float: "float",
            str: "str",
            bytes: "bytes",
            list: "list",
            tuple: "tuple",
            dict: "dict",
            set: "set",
            frozenset: "frozenset",
            function: "function",
            object: "object",
        },
    }
}

/// The symbols a frontend's host plugin gives the library: no case here
/// reaches one, but code that names them must link.
extern "C" fn unreached() -> i64 {
    eprintln!("a host function was called in a test that uses none");
    std::process::abort()
}
static HOST_INFO: zrtl::ZrtlInfo = zrtl::ZrtlInfo::new(c"test_host".as_ptr());
static HOST_SYMBOLS: [zrtl::ZrtlSymbol; 35] = [
    zrtl::ZrtlSymbol::new(c"$Host$argc".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$argv".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_at".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_box".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_copy_out".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_decode".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_eq".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_from_buffer".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Host$bytes_from_buffer_range".as_ptr(),
        unreached as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_from_hex".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_from_ints".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_hash".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_hex".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_of_byte".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_of_storage".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_repeat".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_repr".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_slice".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$bytes_zeros".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$file_exists".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$file_read".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$file_remove".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$file_write".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$is_float_literal".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$is_int_literal".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$json_escape".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$md5".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$missing_arg".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$path_basename".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$path_dirname".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$path_join".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$perf_counter".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$struct_float".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$struct_int".as_ptr(), unreached as *const u8),
    zrtl::ZrtlSymbol::new(c"$Host$time".as_ptr(), unreached as *const u8),
];
fn host() -> zrtl::StaticPlugin {
    zrtl::StaticPlugin {
        info: &HOST_INFO,
        symbols: &HOST_SYMBOLS,
    }
}

/// A case: the functions it declares over the library's list type, the
/// one it is entered through, which takes nothing and returns an
/// integer, and what that returns.
struct Case {
    decls: Vec<Decl>,
    entry: &'static str,
    expected: i64,
}

/// Compile the library with every case and call each case's entry. One
/// runtime serves them all: a process holds one at a time.
#[test]
fn the_tables_run() {
    let lib = library(&policy());
    let list_type = lib.list_type;
    let cases = [
        small_distinct_literal_lookups_hit(list_type),
        int_dict_deletes_and_compacts(list_type),
        int_set_keeps_its_mask(list_type),
        dynamic_keys_meet_typed_ones(list_type),
    ];
    let mut declarations = lib.declarations;
    for case in &cases {
        for decl in &case.decls {
            let mut decl = decl.clone();
            // The entry is the program's own, which the library's
            // functions are built as the program reaches them from.
            if let zyntax_typed_ast::typed_ast::TypedDeclaration::Function(f) = &mut decl.node
                && f.name.resolve_global().as_deref() == Some(case.entry)
            {
                f.module = None;
            }
            declarations.push(decl);
        }
    }
    let program = TypedProgram {
        declarations,
        language: None,
        span: Span::new(0, 0),
        source_files: Vec::new(),
        type_registry: lib.type_registry,
    };
    let config = TieredConfig::default();
    let mut rt = TieredRuntime::new(config).expect("runtime");
    rt.set_automatic_release(true);
    rt.set_pattern_rewrites(false);
    rt.declare_entry_points(cases.iter().map(|c| c.entry));
    rt.register_static_plugins([
        zrtl_io::static_plugin(),
        zrtl_string::static_plugin(),
        zrtl_math::static_plugin(),
        host(),
        zyntax_embed::foreign::static_plugin(),
    ])
    .expect("plugins");
    rt.compile_typed_program(program).expect("compiles");
    let mut wrong = Vec::new();
    for case in &cases {
        match rt.call_raw(case.entry, &[]).expect("runs") {
            ZyntaxValue::Int(n) if n == case.expected => {}
            other => wrong.push(format!(
                "{}: {other:?}, expected {}",
                case.entry, case.expected
            )),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn boxed(n: i64) -> Expr {
    call("zb_box_i64", vec![int(n)], any())
}
fn boxed_str(s: &str) -> Expr {
    call("zb_str_to_dynamic", vec![text(s)], any())
}
fn as_int(x: Expr) -> Expr {
    call("zb_any_as_i64", vec![x], i64())
}

/// A literal of a few distinct keys lays its entries out unhashed and
/// takes no index; every lookup by an equal key, boxed anew, hits, and
/// an absent key misses. The same for a literal large enough to take
/// an index.
fn small_distinct_literal_lookups_hit(list_type: TypeId) -> Case {
    let decls = {
        let dt = dict_type(list_type);
        let entry_ty = dict_entry_type(&Field::Any, &Field::Any);
        let d = local("d", dt.clone());
        let big = local("big", dt.clone());
        let r = local("r", i64());
        let get = |d: &Local, k: Expr| as_int(call("zb_dict_get", vec![d.e(), k], any()));
        // Laid out as a literal lays them out: each pair with its hash
        // word zero.
        let entry = |k: Expr, v: Expr| tuple(vec![int(0), k, v], entry_ty.clone());
        let mut pairs = Vec::new();
        for (k, v) in [(boxed(1), 10), (boxed(2), 20), (boxed_str("a"), 30)] {
            pairs.push(entry(k, boxed(v)));
        }
        let mut big_pairs = Vec::new();
        for i in 0..20 {
            big_pairs.push(entry(boxed(i * 7), boxed(i)));
        }
        vec![define(
            "t_small_literal",
            &[],
            i64(),
            vec![
                d.decl(call(
                    "zb_dict_from_distinct",
                    vec![list(pairs, dt.clone())],
                    dt.clone(),
                )),
                big.decl(call(
                    "zb_dict_from_distinct",
                    vec![list(big_pairs, dt.clone())],
                    dt.clone(),
                )),
                r.decl(add(get(&d, boxed(1)), get(&d, boxed(2)))),
                r.set(add(r.e(), get(&d, boxed_str("a")))),
                r.set(add(
                    r.e(),
                    as_int(call("zb_dict_get_str", vec![d.e(), text("a")], any())),
                )),
                when(
                    call("zb_dict_contains", vec![d.e(), boxed(3)], boolean()),
                    vec![r.set(add(r.e(), int(1000)))],
                ),
                r.set(add(
                    r.e(),
                    mul(call("zb_dict_len", vec![d.e()], i64()), int(100)),
                )),
                r.set(add(r.e(), mul(get(&big, boxed(19 * 7)), int(10_000)))),
                when(
                    call("zb_dict_contains", vec![big.e(), boxed(8)], boolean()),
                    vec![r.set(add(r.e(), int(1_000_000)))],
                ),
                ret(r.e()),
            ],
        )]
    };
    Case {
        decls,
        entry: "t_small_literal",
        expected: 10 + 20 + 30 + 30 + 300 + 190_000,
    }
}

/// A typed shape of int keys and values: stores, deletions of most
/// keys (tombstones, then compaction), lookups of the survivors and of
/// the deleted, a key stored again after its deletion, and popitem.
fn int_dict_deletes_and_compacts(list_type: TypeId) -> Case {
    let decls = {
        let key = Field::Int;
        let mut decls = dict_declarations(list_type, "ii", dict_shape_tag(0), &key, &key);
        let dt = list_of(
            list_type,
            zyntax_builtins::dicts::dict_entry_type(&key, &key),
        );
        let d = local("d", dt.clone());
        let i = local("i", i64());
        let s = local("s", i64());
        let f = |op: &str, args: Vec<Expr>, ty: zyntax_typed_ast::Type| {
            call(&format!("zb_dict_{op}_ii"), args, ty)
        };
        let mut body = vec![d.decl(f("new", vec![], dt.clone()))];
        body.extend(for_range(
            &i,
            int(0),
            int(1000),
            vec![expr(f(
                "set",
                vec![d.e(), i.e(), mul(i.e(), int(3))],
                unit(),
            ))],
        ));
        body.extend(for_range(
            &i,
            int(0),
            int(1000),
            vec![when(
                ne(rem(i.e(), int(3)), int(0)),
                vec![expr(f("del", vec![d.e(), i.e()], unit()))],
            )],
        ));
        body.push(s.decl(mul(f("len", vec![d.e()], i64()), int(1_000_000))));
        body.extend(for_range(
            &i,
            int(0),
            int(1000),
            vec![s.set(add(
                s.e(),
                f("get_default", vec![d.e(), i.e(), int(-1)], i64()),
            ))],
        ));
        // 5 comes back at the end; popitem takes it first.
        body.push(expr(f("set", vec![d.e(), int(5), int(7)], unit())));
        body.push(s.set(add(
            s.e(),
            mul(
                as_int(call(
                    "zb_list_get_any",
                    vec![
                        call(
                            "zb_unbox_tuple",
                            vec![f("popitem", vec![d.e()], any())],
                            list_of(list_type, any()),
                        ),
                        int(1),
                    ],
                    any(),
                )),
                int(1_000_000_000),
            ),
        )));
        body.push(ret(s.e()));
        decls.push(define("t_int_dict", &[], i64(), body));
        decls
    };
    // 334 survivors (the multiples of 3 below 1000); each keeps 3i and
    // each of the 666 deleted reads -1.
    let survivors: i64 = (0..1000).filter(|i| i % 3 == 0).map(|i| i * 3).sum();
    Case {
        decls,
        entry: "t_int_dict",
        expected: 334 * 1_000_000 + survivors - 666 + 7 * 1_000_000_000,
    }
}

/// A typed set of ints: the mask answers membership, intersection,
/// difference and the least value while every value is small, and the
/// table does once one is not; removals clear their bits.
fn int_set_keeps_its_mask(list_type: TypeId) -> Case {
    let key = Field::Int;
    let mut decls = set_declarations(list_type, "si", set_shape_tag(0), &key);
    let st = list_of(list_type, set_entry_type(&key));
    let a = local("a", st.clone());
    let b = local("b", st.clone());
    let i = local("i", i64());
    let r = local("r", i64());
    let f = |op: &str, args: Vec<Expr>, ty: zyntax_typed_ast::Type| {
        call(&format!("zb_set_{op}_si"), args, ty)
    };
    let has = |s: &Local, v: i64| f("contains", vec![s.e(), int(v)], boolean());
    let count = |s: Expr| f("len", vec![s], i64());
    let mut body = vec![
        a.decl(f("new", vec![], st.clone())),
        b.decl(f("new", vec![], st.clone())),
    ];
    // a: 0..40 step 2, b: 0..40 step 3; both masks stand.
    body.extend(for_range(
        &i,
        int(0),
        int(20),
        vec![expr(f("add", vec![a.e(), mul(i.e(), int(2))], unit()))],
    ));
    body.extend(for_range(
        &i,
        int(0),
        int(14),
        vec![expr(f("add", vec![b.e(), mul(i.e(), int(3))], unit()))],
    ));
    body.push(r.decl(mul(count(f("and", vec![a.e(), b.e()], st.clone())), int(1))));
    body.push(r.set(add(
        r.e(),
        mul(count(f("sub", vec![a.e(), b.e()], st.clone())), int(100)),
    )));
    body.push(r.set(add(r.e(), mul(f("mask63", vec![a.e()], i64()), int(0)))));
    body.push(expr(f("discard", vec![a.e(), int(0)], unit())));
    body.push(r.set(add(r.e(), mul(f("min", vec![a.e()], i64()), int(10_000)))));
    body.push(when(has(&a, 0), vec![r.set(add(r.e(), int(1_000_000)))]));
    // A large value ends the mask; the table answers the same.
    body.push(expr(f("add", vec![a.e(), int(1000)], unit())));
    body.push(when(
        has(&a, 1000),
        vec![r.set(add(r.e(), int(10_000_000)))],
    ));
    body.push(when(has(&a, 4), vec![r.set(add(r.e(), int(100_000_000)))]));
    body.push(r.set(add(
        r.e(),
        mul(
            count(f("and", vec![a.e(), b.e()], st.clone())),
            int(1_000_000_000),
        ),
    )));
    body.push(when(
        lt(f("mask63", vec![a.e()], i64()), int(0)),
        vec![r.set(add(r.e(), int(5)))],
    ));
    body.push(ret(r.e()));
    decls.push(define("t_int_set", &[], i64(), body));
    // Common: multiples of 6 below 40 (7); a alone: 20 - 7 = 13; min
    // after removing 0 is 2; 0 is gone; 1000 and 4 are in; common
    // after removing 0 is 6; the mask no longer stands.
    Case {
        decls,
        entry: "t_int_set",
        expected: 7 + 13 * 100 + 2 * 10_000 + 10_000_000 + 100_000_000 + 6 * 1_000_000_000 + 5,
    }
}

/// A dynamic key probes a typed shape by the rule keys of different
/// kinds meet by: a bool or an integral float finds an int key, a float
/// that is not integral or a string finds none, and an int finds a
/// float key only when the conversion is exact. A typed dict equals a
/// dynamic one of equal keys and values across kinds.
fn dynamic_keys_meet_typed_ones(list_type: TypeId) -> Case {
    let int_key = Field::Int;
    let float_key = Field::Float;
    let mut decls = dict_declarations(list_type, "ij", dict_shape_tag(1), &int_key, &int_key);
    decls.extend(dict_declarations(
        list_type,
        "fk",
        dict_shape_tag(2),
        &float_key,
        &int_key,
    ));
    let ij = list_of(list_type, dict_entry_type(&int_key, &int_key));
    let fk = list_of(list_type, dict_entry_type(&float_key, &int_key));
    let dt = dict_type(list_type);
    let a = local("a", ij.clone());
    let f = local("f", fk.clone());
    let dy = local("dy", dt.clone());
    let r = local("r", i64());
    let boxed_f = |v: f64| call("zb_box_f64", vec![float(v)], any());
    let boxed_b = |v: bool| call("zb_box_bool", vec![bool(v)], any());
    let found = |d: &Local, shape: &str, k: Expr| {
        ge(
            call(&format!("zb_dict_find_any_{shape}"), vec![d.e(), k], i64()),
            int(0),
        )
    };
    let bit = |cond: Expr, weight: i64| when(cond, vec![r.set(add(r.e(), int(weight)))]);
    let body = vec![
        a.decl(call("zb_dict_new_ij", vec![], ij.clone())),
        expr(call("zb_dict_set_ij", vec![a.e(), int(1), int(10)], unit())),
        expr(call("zb_dict_set_ij", vec![a.e(), int(2), int(20)], unit())),
        f.decl(call("zb_dict_new_fk", vec![], fk.clone())),
        expr(call(
            "zb_dict_set_fk",
            vec![f.e(), float(1.0), int(5)],
            unit(),
        )),
        expr(call(
            "zb_dict_set_fk",
            vec![f.e(), float(4_611_686_018_427_387_904.0), int(6)],
            unit(),
        )),
        r.decl(int(0)),
        bit(found(&a, "ij", boxed_f(1.0)), 1),
        bit(found(&a, "ij", boxed_b(true)), 2),
        bit(found(&a, "ij", boxed(2)), 4),
        bit(found(&a, "ij", boxed_f(1.5)), 8),
        bit(found(&a, "ij", boxed_str("1")), 16),
        bit(found(&f, "fk", boxed(1)), 32),
        bit(found(&f, "fk", boxed(4_611_686_018_427_387_904)), 64),
        bit(found(&f, "fk", boxed(4_611_686_018_427_387_905)), 128),
        // {1.0: 10.0, 2: 20} equals a; {1: 10, 2: 21} does not.
        dy.decl(call("zb_dict_new", vec![], dt.clone())),
        expr(call(
            "zb_dict_set",
            vec![dy.e(), boxed_f(1.0), boxed_f(10.0)],
            unit(),
        )),
        expr(call(
            "zb_dict_set",
            vec![dy.e(), boxed(2), boxed(20)],
            unit(),
        )),
        bit(
            call(
                "zb_dict_eq_any_ij",
                vec![a.e(), call("zb_dict_box", vec![dy.e()], any())],
                boolean(),
            ),
            256,
        ),
        expr(call(
            "zb_dict_set",
            vec![dy.e(), boxed(2), boxed(21)],
            unit(),
        )),
        bit(
            call(
                "zb_dict_eq_any_ij",
                vec![a.e(), call("zb_dict_box", vec![dy.e()], any())],
                boolean(),
            ),
            512,
        ),
        ret(r.e()),
    ];
    decls.push(define("t_dynamic_keys", &[], i64(), body));
    Case {
        decls,
        entry: "t_dynamic_keys",
        expected: 1 + 2 + 4 + 32 + 64 + 256,
    }
}
