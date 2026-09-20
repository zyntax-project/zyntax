//! Byte strings and files. A byte string is stored as a string is,
//! `[i32 length][bytes]`, and shares its release; every operation on
//! one is binary-safe, in the host, since nothing here reads it as
//! text. Boxed, it carries the [`BYTES`] category. A file is a record
//! of its path, its mode and a byte buffer: writes fill the buffer and
//! `close` hands it to the host whole, reads take the whole file.

use crate::build::*;
use crate::list_of;
use zyntax_typed_ast::TypeId;

/// The box category of a byte string; see `dynamic`.
pub(crate) const BYTES: i64 = 6;

/// Mode bits of an open file.
const READ: i64 = 1;
const WRITE: i64 = 2;
const APPEND: i64 = 4;
const CLOSED: i64 = 16;

pub(crate) fn declarations(list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let bytes_buf = list_of(list_type, u8());
    let mut d = Vec::new();

    for (name, params, ret, symbol) in [
        (
            "zb_bytes_of_byte_raw",
            vec![("v", i64())],
            string(),
            "$Host$bytes_of_byte",
        ),
        (
            "zb_box_bytes",
            vec![("a", string())],
            any(),
            "$Host$bytes_box",
        ),
        (
            "zb_bytes_repeat",
            vec![("a", string()), ("n", i64())],
            string(),
            "$Host$bytes_repeat",
        ),
        (
            "zb_bytes_at",
            vec![("a", string()), ("i", i64())],
            i64(),
            "$Host$bytes_at",
        ),
        (
            "zb_bytes_slice_raw",
            vec![("a", string()), ("start", i64()), ("end", i64())],
            string(),
            "$Host$bytes_slice",
        ),
        (
            "zb_bytes_eq",
            vec![("a", string()), ("b", string())],
            i32(),
            "$Host$bytes_eq",
        ),
        (
            "zb_bytes_hash",
            vec![("a", string())],
            i64(),
            "$Host$bytes_hash",
        ),
        (
            "zb_bytes_repr",
            vec![("a", string())],
            string(),
            "$Host$bytes_repr",
        ),
        (
            "zb_bytes_decode_raw",
            vec![("a", string())],
            string(),
            "$Host$bytes_decode",
        ),
        (
            "zb_bytes_from_ints_raw",
            vec![("xs", list_of(list_type, i64()))],
            string(),
            "$Host$bytes_from_ints",
        ),
        (
            "zb_bytes_zeros",
            vec![("n", i64())],
            string(),
            "$Host$bytes_zeros",
        ),
        // The list itself is passed, so it stays live past the read.
        (
            "zb_bytes_from_buffer",
            vec![("xs", list_of(list_type, u8()))],
            string(),
            "$Host$bytes_from_buffer",
        ),
        (
            "zb_bytes_copy_out",
            vec![("a", string()), ("data", i64())],
            i64(),
            "$Host$bytes_copy_out",
        ),
        (
            "zb_file_write_raw",
            vec![("path", string()), ("data", string()), ("append", i64())],
            i64(),
            "$Host$file_write",
        ),
        (
            "zb_file_read_raw",
            vec![("path", string())],
            string(),
            "$Host$file_read",
        ),
        (
            "zb_file_exists",
            vec![("path", string())],
            i64(),
            "$Host$file_exists",
        ),
        (
            "zb_file_remove_raw",
            vec![("path", string())],
            i64(),
            "$Host$file_remove",
        ),
    ] {
        d.push(extern_fn(name, &params, ret, Some(symbol)));
    }

    let b = local("b", string());
    let i = local("i", i64());
    let n = local("n", i64());
    let v = local("v", i64());
    let x = local("x", any());

    d.push(define(
        "zb_any_as_bytes",
        &[&x],
        string(),
        vec![
            when(
                ne(call("zb_any_category", vec![x.e()], i64()), int(BYTES)),
                vec![fatal("TypeError", text("a bytes-like object is required"))],
            ),
            ret(call("zb_box_get_str", vec![x.e()], string())),
        ],
    ));

    // b[i]: the byte, IndexError past either end.
    d.push(define(
        "zb_bytes_index",
        &[&b, &i],
        i64(),
        vec![
            v.decl(call("zb_bytes_at", vec![b.e(), i.e()], i64())),
            when(
                lt(v.e(), int(0)),
                vec![fatal("IndexError", text("index out of range"))],
            ),
            ret(v.e()),
        ],
    ));

    // `%c` of an int: the one byte, when it is one.
    d.push(define(
        "zb_bytes_byte",
        &[&v],
        string(),
        vec![
            when(
                or(lt(v.e(), int(0)), gt(v.e(), int(255))),
                vec![fatal("OverflowError", text("%c arg not in range(256)"))],
            ),
            ret(call("zb_bytes_of_byte_raw", vec![v.e()], string())),
        ],
    ));

    // b[start:stop:step]; `mask` bits 1, 2, 4 say which bounds were given.
    let start = local("start", i64());
    let stop = local("stop", i64());
    let step = local("step", i64());
    let mask = local("mask", i64());
    let lo = local("lo", i64());
    let hi = local("hi", i64());
    let st = local("st", i64());
    let out = local("out", bytes_buf.clone());
    let push_byte = |value: Expr| expr(mcall(out.e(), "push", vec![value], unit()));
    let bytes_of_buffer = || call("zb_bytes_from_buffer", vec![out.e()], string());
    d.push(define(
        "zb_bytes_slice",
        &[&b, &start, &stop, &step, &mask],
        string(),
        vec![
            n.decl(call("zb_str_len", vec![b.e()], i64())),
            st.decl(int(1)),
            when(ne(bitand(mask.e(), int(4)), int(0)), vec![st.set(step.e())]),
            when(
                eq(st.e(), int(0)),
                vec![fatal("ValueError", text("slice step cannot be zero"))],
            ),
            lo.decl(int(0)),
            hi.decl(n.e()),
            when(
                lt(st.e(), int(0)),
                vec![lo.set(sub(n.e(), int(1))), hi.set(int(-1))],
            ),
            when(
                ne(bitand(mask.e(), int(1)), int(0)),
                vec![lo.set(call(
                    "zb_slice_bound",
                    vec![start.e(), n.e(), st.e()],
                    i64(),
                ))],
            ),
            when(
                ne(bitand(mask.e(), int(2)), int(0)),
                vec![hi.set(call("zb_slice_bound", vec![stop.e(), n.e(), st.e()], i64()))],
            ),
            when(
                eq(st.e(), int(1)),
                vec![ret(call(
                    "zb_bytes_slice_raw",
                    vec![b.e(), lo.e(), hi.e()],
                    string(),
                ))],
            ),
            out.decl(list(Vec::new(), bytes_buf.clone())),
            i.decl(lo.e()),
            if_(
                gt(st.e(), int(0)),
                vec![while_(
                    lt(i.e(), hi.e()),
                    vec![
                        push_byte(cast(call("zb_bytes_at", vec![b.e(), i.e()], i64()), u8())),
                        i.add_assign(st.e()),
                    ],
                )],
                vec![while_(
                    gt(i.e(), hi.e()),
                    vec![
                        push_byte(cast(call("zb_bytes_at", vec![b.e(), i.e()], i64()), u8())),
                        i.add_assign(st.e()),
                    ],
                )],
            ),
            ret(bytes_of_buffer()),
        ],
    ));

    // The bytes as a list of ints, for iteration.
    let ints = list_of(list_type, i64());
    let items = local("items", ints.clone());
    d.push(define(
        "zb_bytes_to_list",
        &[&b],
        ints.clone(),
        vec![
            items.decl(list(Vec::new(), ints.clone())),
            n.decl(call("zb_str_len", vec![b.e()], i64())),
            block_of(for_range(
                &i,
                int(0),
                n.e(),
                vec![expr(mcall(
                    items.e(),
                    "push",
                    vec![call("zb_bytes_at", vec![b.e(), i.e()], i64())],
                    unit(),
                ))],
            )),
            ret(items.e()),
        ],
    ));

    // bytes(ints): each 0 to 255.
    let xs = borrowed("xs", ints.clone());
    d.push(define(
        "zb_bytes_from_ints",
        &[&xs],
        string(),
        vec![
            b.decl(call("zb_bytes_from_ints_raw", vec![xs.e()], string())),
            when(
                eq(b.e(), null(string())),
                vec![fatal("ValueError", text("bytes must be in range(0, 256)"))],
            ),
            ret(b.e()),
        ],
    ));
    // b.decode(): text, when the bytes are UTF-8.
    let s = local("s", string());
    d.push(define(
        "zb_bytes_decode",
        &[&b],
        string(),
        vec![
            s.decl(call("zb_bytes_decode_raw", vec![b.e()], string())),
            when(
                eq(s.e(), null(string())),
                vec![fatal(
                    "UnicodeDecodeError",
                    text("'utf-8' codec can't decode the bytes"),
                )],
            ),
            ret(s.e()),
        ],
    ));

    // b % args: %d %i %c %s %x %X %o %% with an optional `-`, `0` and a
    // width. The output is built in a byte buffer.
    let fmt = local("fmt", string());
    let args = borrowed("args", anys.clone());
    let at = local("at", i64());
    let c = local("c", i64());
    let arg = local("arg", any());
    let piece = local("piece", string());
    let left = local("left", i64());
    let zero = local("zero", i64());
    let width = local("width", i64());
    let pad = local("pad", i64());
    let j = local("j", i64());
    let next_arg = || {
        vec![
            when(
                ge(at.e(), mcall(args.e(), "len", vec![], i64())),
                vec![fatal(
                    "TypeError",
                    text("not enough arguments for format string"),
                )],
            ),
            arg.set(idx(args.e(), at.e(), any())),
            at.add_assign(int(1)),
        ]
    };
    let push_str_bytes = |s: Expr| {
        // Every byte of a blob, `piece` holding it.
        vec![
            piece.set(s),
            j.decl(int(0)),
            while_(
                lt(j.e(), call("zb_str_len", vec![piece.e()], i64())),
                vec![
                    push_byte(cast(
                        call("zb_bytes_at", vec![piece.e(), j.e()], i64()),
                        u8(),
                    )),
                    j.add_assign(int(1)),
                ],
            ),
        ]
    };
    let padded = |body: Vec<Stmt>| {
        // `piece` is set by `body`; pad it to `width` per the flags.
        let mut s = body;
        s.push(pad.set(sub(width.e(), call("zb_str_len", vec![piece.e()], i64()))));
        s.push(when(
            and(gt(pad.e(), int(0)), eq(left.e(), int(0))),
            vec![
                j.decl(int(0)),
                while_(
                    lt(j.e(), pad.e()),
                    vec![
                        push_byte(cast(if_expr(ne(zero.e(), int(0)), int(48), int(32)), u8())),
                        j.add_assign(int(1)),
                    ],
                ),
            ],
        ));
        s.extend(push_str_bytes(piece.e()));
        s.push(when(
            and(gt(pad.e(), int(0)), ne(left.e(), int(0))),
            vec![
                j.decl(int(0)),
                while_(
                    lt(j.e(), pad.e()),
                    vec![push_byte(cast(int(32), u8())), j.add_assign(int(1))],
                ),
            ],
        ));
        s
    };
    let int_of_arg = || call("zb_any_int", vec![arg.e()], i64());
    d.push(define(
        "zb_bytes_format",
        &[&fmt, &args],
        string(),
        vec![
            out.decl(list(Vec::new(), bytes_buf.clone())),
            n.decl(call("zb_str_len", vec![fmt.e()], i64())),
            at.decl(int(0)),
            arg.decl(null(any())),
            piece.decl(text("")),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    c.decl(call("zb_bytes_at", vec![fmt.e(), i.e()], i64())),
                    i.add_assign(int(1)),
                    if_(
                        ne(c.e(), int(37)),
                        vec![push_byte(cast(c.e(), u8()))],
                        vec![
                            // Flags and width.
                            left.decl(int(0)),
                            zero.decl(int(0)),
                            width.decl(int(0)),
                            c.set(call("zb_bytes_at", vec![fmt.e(), i.e()], i64())),
                            while_(
                                or(eq(c.e(), int(45)), eq(c.e(), int(48))),
                                vec![
                                    when(eq(c.e(), int(45)), vec![left.set(int(1))]),
                                    when(eq(c.e(), int(48)), vec![zero.set(int(1))]),
                                    i.add_assign(int(1)),
                                    c.set(call("zb_bytes_at", vec![fmt.e(), i.e()], i64())),
                                ],
                            ),
                            while_(
                                and(ge(c.e(), int(48)), le(c.e(), int(57))),
                                vec![
                                    width.set(add(mul(width.e(), int(10)), sub(c.e(), int(48)))),
                                    i.add_assign(int(1)),
                                    c.set(call("zb_bytes_at", vec![fmt.e(), i.e()], i64())),
                                ],
                            ),
                            i.add_assign(int(1)),
                            if_(
                                eq(c.e(), int(37)),
                                vec![push_byte(cast(int(37), u8()))],
                                vec![if_(
                                    or(eq(c.e(), int(100)), eq(c.e(), int(105))),
                                    {
                                        let mut s = next_arg();
                                        s.extend(padded(vec![piece.set(call(
                                            "zb_str_of_int",
                                            vec![int_of_arg()],
                                            string(),
                                        ))]));
                                        s
                                    },
                                    vec![if_(
                                        eq(c.e(), int(99)),
                                        {
                                            let mut s = next_arg();
                                            s.push(v.decl(int_of_arg()));
                                            s.push(when(
                                                or(lt(v.e(), int(0)), gt(v.e(), int(255))),
                                                vec![fatal(
                                                    "OverflowError",
                                                    text("%c arg not in range(256)"),
                                                )],
                                            ));
                                            s.push(push_byte(cast(v.e(), u8())));
                                            s
                                        },
                                        vec![if_(
                                            or(eq(c.e(), int(120)), eq(c.e(), int(88))),
                                            {
                                                let mut s = next_arg();
                                                s.extend(padded(vec![piece.set(call(
                                                    "zb_str_of_int_radix",
                                                    vec![int_of_arg(), int32(16)],
                                                    string(),
                                                ))]));
                                                s
                                            },
                                            vec![if_(
                                                eq(c.e(), int(111)),
                                                {
                                                    let mut s = next_arg();
                                                    s.extend(padded(vec![piece.set(call(
                                                        "zb_str_of_int_radix",
                                                        vec![int_of_arg(), int32(8)],
                                                        string(),
                                                    ))]));
                                                    s
                                                },
                                                vec![if_(
                                                    eq(c.e(), int(115)),
                                                    {
                                                        let mut s = next_arg();
                                                        s.extend(padded(vec![piece.set(call(
                                                            "zb_any_as_bytes",
                                                            vec![arg.e()],
                                                            string(),
                                                        ))]));
                                                        s
                                                    },
                                                    vec![fatal(
                                                        "ValueError",
                                                        text("unsupported format character"),
                                                    )],
                                                )],
                                            )],
                                        )],
                                    )],
                                )],
                            ),
                        ],
                    ),
                ],
            ),
            when(
                lt(at.e(), mcall(args.e(), "len", vec![], i64())),
                vec![fatal(
                    "TypeError",
                    text("not all arguments converted during bytes formatting"),
                )],
            ),
            ret(bytes_of_buffer()),
        ],
    ));

    // Files: [path, mode, buffer].
    let path = local("path", string());
    let mode = local("mode", string());
    let flags = local("flags", i64());
    let f = borrowed("f", anys.clone());
    let record = local("record", anys.clone());
    let buffer = local("buffer", bytes_buf.clone());
    let data = local("data", string());
    let mode_len = local("mode_len", i64());
    d.push(define(
        "zb_file_open",
        &[&path, &mode],
        anys.clone(),
        vec![
            flags.decl(int(0)),
            mode_len.decl(call("zb_str_len", vec![mode.e()], i64())),
            block_of(for_range(
                &i,
                int(0),
                mode_len.e(),
                vec![
                    c.decl(call("zb_bytes_at", vec![mode.e(), i.e()], i64())),
                    when(
                        eq(c.e(), int(114)),
                        vec![flags.set(bitor(flags.e(), int(READ)))],
                    ),
                    when(
                        eq(c.e(), int(119)),
                        vec![flags.set(bitor(flags.e(), int(WRITE)))],
                    ),
                    when(
                        eq(c.e(), int(97)),
                        vec![flags.set(bitor(flags.e(), int(APPEND)))],
                    ),
                    when(
                        eq(c.e(), int(43)),
                        vec![flags.set(bitor(flags.e(), int(READ | WRITE)))],
                    ),
                ],
            )),
            when(
                eq(flags.e(), int(0)),
                vec![fatal("ValueError", text("invalid mode"))],
            ),
            when(
                and(
                    eq(bitand(flags.e(), int(WRITE | APPEND)), int(0)),
                    eq(call("zb_file_exists", vec![path.e()], i64()), int(0)),
                ),
                vec![fatal(
                    "FileNotFoundError",
                    add(text("No such file or directory: "), path.e()),
                )],
            ),
            record.decl(list(Vec::new(), anys.clone())),
            expr(mcall(
                record.e(),
                "push",
                vec![call("zb_box_str", vec![path.e()], any())],
                unit(),
            )),
            expr(mcall(
                record.e(),
                "push",
                vec![call("zb_box_i64", vec![flags.e()], any())],
                unit(),
            )),
            expr(mcall(
                record.e(),
                "push",
                vec![call(
                    "zb_list_box_u8",
                    vec![list(Vec::new(), bytes_buf.clone())],
                    any(),
                )],
                unit(),
            )),
            ret(record.e()),
        ],
    ));
    let file_flags = || call("zb_box_get_i64", vec![idx(f.e(), int(1), any())], i64());
    let file_path = || call("zb_box_get_str", vec![idx(f.e(), int(0), any())], string());
    let file_buffer = || {
        call(
            "zb_unbox_list_raw_u8",
            vec![idx(f.e(), int(2), any())],
            bytes_buf.clone(),
        )
    };
    let closed_check = || {
        when(
            ne(bitand(file_flags(), int(CLOSED)), int(0)),
            vec![fatal("ValueError", text("I/O operation on closed file"))],
        )
    };
    let old_len = local("old_len", i64());
    d.push(define(
        "zb_file_write",
        &[&f, &data],
        unit(),
        vec![
            closed_check(),
            when(
                eq(bitand(file_flags(), int(WRITE | APPEND)), int(0)),
                vec![fatal("IOError", text("file not open for writing"))],
            ),
            buffer.decl(file_buffer()),
            n.decl(call("zb_str_len", vec![data.e()], i64())),
            old_len.decl(mcall(buffer.e(), "len", vec![], i64())),
            block_of(for_range(
                &i,
                int(0),
                n.e(),
                vec![expr(mcall(
                    buffer.e(),
                    "push",
                    vec![cast(int(0), u8())],
                    unit(),
                ))],
            )),
            expr(call(
                "zb_bytes_copy_out",
                vec![data.e(), add(fld(buffer.e(), "data", i64()), old_len.e())],
                i64(),
            )),
            ret_void(),
        ],
    ));
    d.push(define(
        "zb_file_read",
        &[&f],
        string(),
        vec![
            closed_check(),
            when(
                eq(bitand(file_flags(), int(READ)), int(0)),
                vec![fatal("IOError", text("file not open for reading"))],
            ),
            data.decl(call("zb_file_read_raw", vec![file_path()], string())),
            when(
                eq(data.e(), null(string())),
                vec![fatal(
                    "FileNotFoundError",
                    add(text("No such file or directory: "), file_path()),
                )],
            ),
            ret(data.e()),
        ],
    ));
    d.push(define(
        "zb_file_close",
        &[&f],
        unit(),
        vec![
            when(
                ne(bitand(file_flags(), int(CLOSED)), int(0)),
                vec![ret_void()],
            ),
            flags.decl(file_flags()),
            when(
                ne(bitand(flags.e(), int(WRITE | APPEND)), int(0)),
                vec![
                    buffer.decl(file_buffer()),
                    when(
                        lt(
                            call(
                                "zb_file_write_raw",
                                vec![
                                    file_path(),
                                    call("zb_bytes_from_buffer", vec![buffer.e()], string()),
                                    bitand(flags.e(), int(APPEND)),
                                ],
                                i64(),
                            ),
                            int(0),
                        ),
                        vec![fatal("IOError", add(text("cannot write "), file_path()))],
                    ),
                ],
            ),
            set_idx(
                f.e(),
                int(1),
                call("zb_box_i64", vec![bitor(flags.e(), int(CLOSED))], any()),
            ),
            ret_void(),
        ],
    ));
    // os.remove(path)
    let path = local("path", string());
    d.push(define(
        "zb_file_remove",
        &[&path],
        unit(),
        vec![when(
            lt(call("zb_file_remove_raw", vec![path.e()], i64()), int(0)),
            vec![fatal(
                "FileNotFoundError",
                add(text("No such file or directory: "), path.e()),
            )],
        )],
    ));
    d
}
