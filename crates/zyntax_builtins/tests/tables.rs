//! The dict and set library run, not only built: each case is a
//! function written with the library's own builder, compiled with the
//! library and called.

use zyntax_builtins::build::*;
use zyntax_builtins::dicts::{dict_declarations, dict_shape_tag, dict_type};
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
    ];
    let mut declarations = lib.declarations;
    for case in &cases {
        declarations.extend(case.decls.iter().cloned());
    }
    let program = TypedProgram {
        declarations,
        language: None,
        span: Span::new(0, 0),
        source_files: Vec::new(),
        type_registry: lib.type_registry,
    };
    // A frame stays in the tier it started in: these cases test the
    // library, and moving a running frame between tiers is tested where
    // the tiers are.
    let config = TieredConfig {
        enable_osr: false,
        ..TieredConfig::default()
    };
    let mut rt = TieredRuntime::new(config).expect("runtime");
    rt.set_automatic_release(true);
    rt.set_pattern_rewrites(false);
    rt.declare_entry_points(cases.iter().map(|c| c.entry));
    rt.register_static_plugins([
        zrtl_io::static_plugin(),
        zrtl_string::static_plugin(),
        zrtl_math::static_plugin(),
        host(),
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
        let anys = list_of(list_type, any());
        let dt = dict_type(list_type);
        let d = local("d", dt.clone());
        let big = local("big", dt.clone());
        let r = local("r", i64());
        let get = |d: &Local, k: Expr| as_int(call("zb_dict_get", vec![d.e(), k], any()));
        let mut pairs = vec![null(any())];
        for (k, v) in [(boxed(1), 10), (boxed(2), 20), (boxed_str("a"), 30)] {
            pairs.push(k);
            pairs.push(boxed(v));
        }
        let mut big_pairs = vec![null(any())];
        for i in 0..20 {
            big_pairs.push(boxed(i * 7));
            big_pairs.push(boxed(i));
        }
        vec![define(
            "t_small_literal",
            &[],
            i64(),
            vec![
                d.decl(call(
                    "zb_dict_from_distinct",
                    vec![list(pairs, anys.clone())],
                    dt.clone(),
                )),
                big.decl(call(
                    "zb_dict_from_distinct",
                    vec![list(big_pairs, anys.clone())],
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
