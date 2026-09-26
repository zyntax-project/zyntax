//! Dictionaries and sets: open-addressing hash tables over one list of
//! entries, generated per key and value kind.
//!
//! A dict of `K` to `V` is a `List<(i64, K, V)>` and a set of `K` a
//! `List<(i64, K)>`: each entry holds the key's mixed hash, the key and
//! the value inline. Entries from 1 on are in insertion order, so
//! iteration and printing show the order things were added in. Entry 0
//! is the control entry: its hash word holds the address of the hash
//! index, 0 while there is none, and its other words belong to the
//! shape (a set of small integers keeps its mask there).
//!
//! The index is a `List<i64>`: word 0 counts the dead entries, word 1 is
//! a shape's own, and the slots follow, a power of two of entry numbers,
//! -1 where empty, probed linearly from the hash and never more than
//! half full.
//!
//! A container of at most [`SMALL`] entries has no index. It is scanned
//! comparing keys, and its hash words are never read: a literal leaves
//! them zero, and they are computed when the index is first built. A
//! deletion from it removes the entry. A deletion from an indexed
//! container leaves a tombstone: hash [`TOMB`], which no mixed hash is,
//! key and value zeroed, and the slot still naming it so a probe passes
//! over it. The entries are compacted in order and the index rebuilt
//! when the table grows or the dead outnumber half the live.
//!
//! [`dict_declarations`] generates the functions of one shape,
//! `zb_dict_<op><suffix>`; the library's own dict is the shape of
//! dynamic keys and values with no suffix, and a frontend generates the
//! typed shapes its program uses.

use crate::build::*;
use crate::lists::Field;
use crate::{DICT_TAG, SET_TAG, TUPLE_TAG, list_of};
use zyntax_typed_ast::{Type, TypeId};

/// Entries a container holds before it takes an index.
pub const SMALL: i64 = 8;
/// The fewest slots an index has.
const FIRST_TABLE: i64 = 32;
/// The hash word of a deleted entry. The mixer never produces it.
pub const TOMB: i64 = i64::MIN;
/// The words of an index before its slots: the dead count, then the
/// shape's word.
const INDEX_HEAD: i64 = 2;
/// A set of dynamic values keeps its small-integer mask in the index's
/// shape word; this is the word before the mask is computed.
const MASK_UNKNOWN: i64 = -2;
/// Kinds of the table shapes a frontend registers sit this far above
/// [`crate::lists::SHAPE_KIND_BASE`], dicts first, then sets.
const DICT_SHAPES: i64 = 1 << 16;
const SET_SHAPES: i64 = 2 << 16;

/// The type of a dict entry of key `key` and value `value`.
pub fn dict_entry_type(key: &Field, value: &Field) -> Type {
    Type::Tuple(vec![i64(), key.ty(), value.ty()])
}

/// The type of the library's dict: dynamic keys and values.
pub fn dict_type(list_type: TypeId) -> Type {
    list_of(list_type, dict_entry_type(&Field::Any, &Field::Any))
}

/// The box tag of the dict shape a frontend registered as `index`.
pub fn dict_shape_tag(index: u16) -> i64 {
    ((crate::lists::SHAPE_KIND_BASE + DICT_SHAPES + index as i64) << 8) | 255
}

/// The box tag of the set shape a frontend registered as `index`.
pub fn set_shape_tag(index: u16) -> i64 {
    ((crate::lists::SHAPE_KIND_BASE + SET_SHAPES + index as i64) << 8) | 255
}

fn any_eq(a: Expr, b: Expr) -> Expr {
    call("zb_any_eq", vec![a, b], boolean())
}
/// Whether a stored key matches the one looked up: the same box, as a
/// key read back from the dict or a shared constant is, before the
/// values are compared.
fn key_matches(stored: Expr, k: Expr) -> Expr {
    or(eq(stored.clone(), k.clone()), any_eq(stored, k))
}
/// The statements that return `found` when `stored` matches `k`, whose
/// category is `kcat`: the same box, or two strings compared as
/// strings, or anything else compared as dynamic values.
fn when_key_matches(stored: &Local, k: &Local, kcat: &Local, found: Expr) -> Vec<Stmt> {
    let category = |x: Expr| call("zb_any_category", vec![x], i64());
    let text = |x: Expr| call("zb_box_get_str", vec![x], string());
    vec![
        when(eq(stored.e(), k.e()), vec![ret(found.clone())]),
        if_(
            and(
                eq(kcat.e(), int(crate::dynamic::STR)),
                eq(category(stored.e()), int(crate::dynamic::STR)),
            ),
            vec![when(
                call("zb_str_eq", vec![text(stored.e()), text(k.e())], boolean()),
                vec![ret(found.clone())],
            )],
            vec![when(any_eq(stored.e(), k.e()), vec![ret(found)])],
        ),
    ]
}
fn any_str(x: Expr) -> Expr {
    call("zb_any_str", vec![x], string())
}
fn len(xs: Expr) -> Expr {
    mcall(xs, "len", vec![], i64())
}
fn push(xs: Expr, v: Expr) -> Stmt {
    expr(mcall(xs, "push", vec![v], unit()))
}
fn at(xs: Expr, i: Expr) -> Expr {
    idx(xs, i, any())
}

pub(crate) fn declarations(list_type: TypeId) -> Vec<Decl> {
    let mut d = shared(list_type);
    d.extend(dict_declarations(
        list_type,
        "",
        DICT_TAG,
        &Field::Any,
        &Field::Any,
    ));
    d.extend(dynamic_dict(list_type));
    d.extend(dict_probe_by(list_type, &Field::Str, "str"));
    d.extend(set(list_type));
    d
}

/// What every table shape shares: the hash mixer and the index.
fn shared(list_type: TypeId) -> Vec<Decl> {
    let ints = list_of(list_type, i64());
    let anys = list_of(list_type, any());
    let hv = local("h", i64());
    let h = local("m", i64());
    let cap = local("cap", i64());
    let n = local("n", i64());
    let index = borrowed("index", ints.clone());
    let entry = local("e", i64());
    let mask = local("mask", i64());
    let slot = local("s", i64());
    let k = kept("k", any());
    let mut d = Vec::new();
    // The hash a key lands by: the value's hash with its bits spread,
    // since the table takes the low ones, and never `TOMB`.
    d.push(define(
        "zb_hash_mix",
        &[&hv],
        i64(),
        vec![
            h.decl(bitxor(hv.e(), shr(hv.e(), int(32)))),
            h.set(mul(h.e(), int(-7_046_029_254_386_353_131))),
            h.set(bitxor(h.e(), shr(h.e(), int(29)))),
            when(eq(h.e(), int(TOMB)), vec![h.set(bitxor(h.e(), int(1)))]),
            ret(h.e()),
        ],
    ));
    d.push(define(
        "zb_dict_key_hash",
        &[&k],
        i64(),
        vec![ret(call(
            "zb_hash_mix",
            vec![call("zb_any_hash", vec![k.e()], i64())],
            i64(),
        ))],
    ));
    // An index of `cap` slots, every one empty, no entry dead.
    d.push(define("zb_table_index_new", &[&cap], ints.clone(), {
        let table = local("table", ints.clone());
        vec![
            table.decl(list(Vec::new(), ints.clone())),
            expr(mcall(
                table.e(),
                "resize_filled",
                vec![add(cap.e(), int(INDEX_HEAD)), int(0xFF)],
                unit(),
            )),
            set_idx(table.e(), int(0), int(0)),
            set_idx(table.e(), int(1), int(MASK_UNKNOWN)),
            ret(table.e()),
        ]
    }));
    // Name entry `e`, whose hash is `h`, in the first empty slot from
    // the hash on; the index has room.
    d.push(define(
        "zb_table_place",
        &[&index, &entry, &hv],
        unit(),
        vec![
            mask.decl(sub(sub(len(index.e()), int(INDEX_HEAD)), int(1))),
            slot.decl(bitand(hv.e(), mask.e())),
            while_(
                ge(
                    idx(index.e(), add(slot.e(), int(INDEX_HEAD)), i64()),
                    int(0),
                ),
                vec![slot.set(bitand(add(slot.e(), int(1)), mask.e()))],
            ),
            set_idx(index.e(), add(slot.e(), int(INDEX_HEAD)), entry.e()),
            ret_void(),
        ],
    ));
    // The slots an index of `n` entries takes: at most a third full, so
    // the table doubles before it is half full.
    d.push(define(
        "zb_table_cap_for",
        &[&n],
        i64(),
        vec![
            cap.decl(int(FIRST_TABLE)),
            while_(
                lt(cap.e(), mul(n.e(), int(3))),
                vec![cap.set(mul(cap.e(), int(2)))],
            ),
            ret(cap.e()),
        ],
    ));
    // An index no entry list names any more goes back to the pool.
    d.push(extern_fn(
        "zb_table_index_free",
        &[("index", ints.clone())],
        unit(),
        Some("zyntax_list_free"),
    ));
    // The raw lists of a boxed dict and set.
    d.push(extern_fn(
        "zb_dict_raw",
        &[("x", any())],
        dict_type(list_type),
        Some("zyntax_box_pointer"),
    ));
    let _ = anys;
    d
}

/// How a set keeps the mask of its small integers.
#[derive(Clone, Copy, PartialEq)]
enum Mask {
    /// It does not.
    None,
    /// In the control entry's key word, kept up to date: bit `v` for
    /// each value `0 <= v < 63`, or -1 once any other value is added.
    Eager,
}

/// One table shape: a dict when it has a value, else a set.
struct Table<'a> {
    list_type: TypeId,
    prefix: &'static str,
    /// Empty for the library's own shape, else `_` and the shape's name.
    sfx: String,
    key: &'a Field,
    value: Option<&'a Field>,
    tag: i64,
    mask: Mask,
}

impl Table<'_> {
    fn name(&self, op: &str) -> String {
        format!("{}_{op}{}", self.prefix, self.sfx)
    }
    fn entry_ty(&self) -> Type {
        let mut fields = vec![i64(), self.key.ty()];
        if let Some(v) = self.value {
            fields.push(v.ty());
        }
        Type::Tuple(fields)
    }
    fn list_ty(&self) -> Type {
        list_of(self.list_type, self.entry_ty())
    }
    fn ints(&self) -> Type {
        list_of(self.list_type, i64())
    }
    fn anys(&self) -> Type {
        list_of(self.list_type, any())
    }
    fn value_field(&self) -> &Field {
        self.value.expect("a dict shape")
    }
    fn entry(&self, d: Expr, e: Expr) -> Expr {
        idx(d, e, self.entry_ty())
    }
    fn hash_of(&self, en: Expr) -> Expr {
        idx(en, int(0), i64())
    }
    fn key_of(&self, en: Expr) -> Expr {
        idx(en, int(1), self.key.ty())
    }
    fn value_of(&self, en: Expr) -> Expr {
        idx(en, int(2), self.value_field().ty())
    }
    fn make(&self, h: Expr, k: Expr, v: Option<Expr>) -> Expr {
        let mut fields = vec![h, k];
        fields.extend(v);
        tuple(fields, self.entry_ty())
    }
    /// An entry of hash `h` whose other words hold nothing.
    fn hollow(&self, h: Expr) -> Expr {
        self.make(h, self.key.zero(), self.value.map(|v| v.zero()))
    }
    /// The control entry's index word.
    fn ctrl(&self, d: Expr) -> Expr {
        self.hash_of(self.entry(d, int(0)))
    }
    fn index_at(&self, c: Expr) -> Expr {
        cast(c, self.ints())
    }
    /// The control entry naming the index at `addr`, its other words
    /// kept.
    fn set_ctrl(&self, d: Expr, addr: Expr) -> Stmt {
        let en = self.entry(d.clone(), int(0));
        let v = self.value.map(|_| self.value_of(en.clone()));
        set_idx(d, int(0), self.make(addr, self.key_of(en), v))
    }
    fn mixed(&self, k: Expr) -> Expr {
        call("zb_hash_mix", vec![self.key.hash(k)], i64())
    }
    /// Whether a replaced index can be released at once: a probe that
    /// compares keys runs none of the program's code, so none is still
    /// reading it. Otherwise the collector takes it.
    fn frees_index(&self) -> bool {
        self.key.is_scalar()
    }
    /// Whether comparing two containers can reach a lookup of this
    /// shape again, through equality of keys that hold containers; the
    /// comparison then looks up through copies of its own, so the
    /// lookups everything else uses stay outside that cycle.
    fn recursive_keys(&self) -> bool {
        !self.key.is_scalar()
    }
    /// A key as an error reports it.
    fn key_text(&self, k: Expr) -> Expr {
        match self.key {
            Field::Any => any_str(k),
            Field::Str => k,
            other => other.repr(k),
        }
    }
    fn call(&self, op: &str, args: Vec<Expr>, ret: Type) -> Expr {
        call(&self.name(op), args, ret)
    }
    fn go(&self, op: &str, args: Vec<Expr>) -> Stmt {
        expr(self.call(op, args, unit()))
    }
    fn fresh(&self) -> Expr {
        self.hollow(int(0))
    }
    /// The statements that return `found` when `stored` matches the
    /// probe `k` of this shape's key.
    fn when_matches(&self, stored: &Local, k: &Local, kcat: &Local, found: Expr) -> Vec<Stmt> {
        match self.key {
            Field::Any => when_key_matches(stored, k, kcat, found),
            key => vec![when(key.eq(stored.e(), k.e()), vec![ret(found)])],
        }
    }
    /// The statements updating the mask after `k` was added.
    fn note_added(&self, d: Expr, k: Expr) -> Vec<Stmt> {
        match self.mask {
            Mask::None => Vec::new(),
            Mask::Eager => {
                let m = self.key_of(self.entry(d.clone(), int(0)));
                let fits = and(ge(k.clone(), int(0)), lt(k.clone(), int(63)));
                let grown = if_expr(fits, bitor(m.clone(), shl(int(1), k)), int(-1));
                vec![when(
                    ge(m.clone(), int(0)),
                    vec![set_idx(
                        d.clone(),
                        int(0),
                        self.make(self.ctrl(d), grown, None),
                    )],
                )]
            }
        }
    }
    /// The statements updating the mask before `k` is removed.
    fn note_removed(&self, d: Expr, k: Expr) -> Vec<Stmt> {
        match self.mask {
            Mask::Eager => {
                let m = self.key_of(self.entry(d.clone(), int(0)));
                let fits = and(ge(k.clone(), int(0)), lt(k.clone(), int(63)));
                vec![when(
                    and(ge(m.clone(), int(0)), fits),
                    vec![set_idx(
                        d.clone(),
                        int(0),
                        self.make(
                            self.ctrl(d),
                            bitand(m, bitxor(shl(int(1), k), int(-1))),
                            None,
                        ),
                    )],
                )]
            }
            Mask::None => Vec::new(),
        }
    }
}

/// The functions every table shape has: creation, length, lookup,
/// insertion of an absent key, deletion, the index's upkeep, copying
/// and the entry accessors.
fn table_core(t: &Table) -> Vec<Decl> {
    let lt_ = t.list_ty();
    let ints = t.ints();
    let d = borrowed("d", lt_.clone());
    let out = local("out", lt_.clone());
    let key = borrowed("k", t.key.ty());
    let k = kept("k", t.key.ty());
    let vf = t.value;
    let v = kept("v", vf.map_or(unit(), |f| f.ty()));
    let hp = local("h", i64());
    let c = local("c", i64());
    let e = local("e", i64());
    let en = local("en", t.entry_ty());
    let i = local("i", i64());
    let n = local("n", i64());
    let w = local("w", i64());
    let cap = local("cap", i64());
    let index = local("index", ints.clone());
    let old = local("old", ints.clone());
    let mask = local("mask", i64());
    let slot = local("slot", i64());
    let dead = local("dead", i64());
    let live = local("live", i64());
    let hashed = local("hashed", boolean());
    let stored = local("stored", t.key.ty());
    let kcat = local("kcat", i64());
    let p = local("p", i64());
    let place =
        |index: Expr, e: Expr, h: Expr| expr(call("zb_table_place", vec![index, e, h], unit()));
    let cap_for = |n: Expr| call("zb_table_cap_for", vec![n], i64());
    let with_kcat = |mut body: Vec<Stmt>| {
        if matches!(t.key, Field::Any) {
            body.insert(0, kcat.decl(call("zb_any_category", vec![key.e()], i64())));
        }
        body
    };
    let vals = |x: Option<Expr>| if vf.is_some() { x } else { None };
    let mut out_decls = Vec::new();

    out_decls.push(define(&t.name("new"), &[], lt_.clone(), {
        vec![
            out.decl(list(Vec::new(), lt_.clone())),
            push(out.e(), t.fresh()),
            ret(out.e()),
        ]
    }));
    out_decls.push(define(&t.name("len"), &[&d], i64(), {
        vec![
            c.decl(t.ctrl(d.e())),
            when(eq(c.e(), int(0)), vec![ret(sub(len(d.e()), int(1)))]),
            ret(sub(
                sub(len(d.e()), int(1)),
                idx(t.index_at(c.e()), int(0), i64()),
            )),
        ]
    }));

    // The entry of `k`, or -1: through the index, given the key's mixed
    // hash, and in any container. See `Table::recursive_keys` for the
    // copies equality uses.
    let variants: &[&str] = if t.recursive_keys() {
        &["", "_eq"]
    } else {
        &[""]
    };
    for variant in variants {
        let find_hashed = format!("find_hashed{variant}");
        out_decls.push(define(
            &t.name(&find_hashed),
            &[&d, &key, &hp],
            i64(),
            with_kcat(vec![
                index.decl(t.index_at(t.ctrl(d.e()))),
                mask.decl(sub(sub(len(index.e()), int(INDEX_HEAD)), int(1))),
                slot.decl(bitand(hp.e(), mask.e())),
                i.decl(int(0)),
                while_(le(i.e(), mask.e()), {
                    let mut hit = vec![stored.decl(t.key_of(en.e()))];
                    hit.extend(t.when_matches(&stored, &key, &kcat, e.e()));
                    vec![
                        e.decl(idx(index.e(), add(slot.e(), int(INDEX_HEAD)), i64())),
                        when(lt(e.e(), int(0)), vec![ret(int(-1))]),
                        en.decl(t.entry(d.e(), e.e())),
                        when(eq(t.hash_of(en.e()), hp.e()), hit),
                        slot.set(bitand(add(slot.e(), int(1)), mask.e())),
                        i.add_assign(int(1)),
                    ]
                }),
                ret(int(-1)),
            ]),
        ));
        out_decls.push(define(
            &t.name(&format!("find{variant}")),
            &[&d, &key],
            i64(),
            vec![
                when(
                    eq(t.ctrl(d.e()), int(0)),
                    with_kcat(vec![
                        n.decl(len(d.e())),
                        i.decl(int(1)),
                        while_(lt(i.e(), n.e()), {
                            let mut body = vec![stored.decl(t.key_of(t.entry(d.e(), i.e())))];
                            body.extend(t.when_matches(&stored, &key, &kcat, i.e()));
                            body.push(i.add_assign(int(1)));
                            body
                        }),
                        ret(int(-1)),
                    ]),
                ),
                ret(t.call(&find_hashed, vec![d.e(), key.e(), t.mixed(key.e())], i64())),
            ],
        ));
    }

    // A fresh index of `cap` slots over the entries, compacted first
    // when some are dead. The hash words are read when `hashed`, else
    // computed from the keys: a container that had no index never
    // wrote them.
    out_decls.push(define(&t.name("rebuild"), &[&d, &cap, &hashed], unit(), {
        let mut compact = vec![n.decl(len(d.e())), w.decl(int(1)), i.decl(int(1))];
        compact.push(while_(
            lt(i.e(), n.e()),
            vec![
                en.decl(t.entry(d.e(), i.e())),
                when(
                    ne(t.hash_of(en.e()), int(TOMB)),
                    vec![
                        when(ne(w.e(), i.e()), vec![set_idx(d.e(), w.e(), en.e())]),
                        w.add_assign(int(1)),
                    ],
                ),
                i.add_assign(int(1)),
            ],
        ));
        compact.push(expr(mcall(d.e(), "truncate", vec![w.e()], unit())));
        let mut rehash = vec![n.decl(len(d.e()))];
        rehash.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![
                en.decl(t.entry(d.e(), i.e())),
                set_idx(
                    d.e(),
                    i.e(),
                    t.make(
                        t.mixed(t.key_of(en.e())),
                        t.key_of(en.e()),
                        vals(vf.map(|_| t.value_of(en.e()))),
                    ),
                ),
            ],
        ));
        let mut st = vec![
            c.decl(t.ctrl(d.e())),
            old.decl(t.index_at(c.e())),
            when(
                ne(c.e(), int(0)),
                vec![when(
                    gt(idx(old.e(), int(0), i64()), int(0)),
                    vec![block_of(compact)],
                )],
            ),
            when(not(hashed.e()), vec![block_of(rehash)]),
            index.decl(call("zb_table_index_new", vec![cap.e()], ints.clone())),
            n.decl(len(d.e())),
        ];
        st.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![place(index.e(), i.e(), t.hash_of(t.entry(d.e(), i.e())))],
        ));
        st.push(t.set_ctrl(d.e(), cast(index.e(), i64())));
        if t.frees_index() {
            st.push(when(
                ne(c.e(), int(0)),
                vec![expr(call("zb_table_index_free", vec![old.e()], unit()))],
            ));
        }
        st.push(ret_void());
        st
    }));
    // Back to no index, the dead dropped: what a container that has
    // shrunk to a few entries is.
    out_decls.push(define(&t.name("unindex"), &[&d], unit(), {
        let mut st = vec![
            c.decl(t.ctrl(d.e())),
            when(eq(c.e(), int(0)), vec![ret_void()]),
            old.decl(t.index_at(c.e())),
            n.decl(len(d.e())),
            w.decl(int(1)),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    en.decl(t.entry(d.e(), i.e())),
                    when(
                        ne(t.hash_of(en.e()), int(TOMB)),
                        vec![
                            when(ne(w.e(), i.e()), vec![set_idx(d.e(), w.e(), en.e())]),
                            w.add_assign(int(1)),
                        ],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            expr(mcall(d.e(), "truncate", vec![w.e()], unit())),
            t.set_ctrl(d.e(), int(0)),
        ];
        if t.frees_index() {
            st.push(expr(call("zb_table_index_free", vec![old.e()], unit())));
        }
        st.push(ret_void());
        st
    }));

    // Add `k`, known to be absent, whose mixed hash is `h`, to a
    // container with an index; the index grows first when the entry
    // would bring it to half full.
    let mut params: Vec<&Local> = vec![&d, &k];
    if vf.is_some() {
        params.push(&v);
    }
    let entry_of = |h: Expr| t.make(h, k.e(), vals(Some(v.e())));
    let mut hashed_params = params.clone();
    hashed_params.push(&hp);
    out_decls.push(define(&t.name("insert_hashed"), &hashed_params, unit(), {
        let mut st = vec![
            index.decl(t.index_at(t.ctrl(d.e()))),
            n.decl(len(d.e())),
            when(
                gt(mul(n.e(), int(2)), sub(len(index.e()), int(INDEX_HEAD))),
                vec![
                    live.decl(sub(sub(n.e(), int(1)), idx(index.e(), int(0), i64()))),
                    t.go(
                        "rebuild",
                        vec![d.e(), cap_for(add(live.e(), int(1))), bool(true)],
                    ),
                ],
            ),
            e.decl(len(d.e())),
            push(d.e(), entry_of(hp.e())),
            place(t.index_at(t.ctrl(d.e())), e.e(), hp.e()),
        ];
        st.extend(t.note_added(d.e(), k.e()));
        st.push(ret_void());
        st
    }));
    // Add `k`, known to be absent; a container past a few entries takes
    // its index.
    out_decls.push(define(&t.name("insert"), &params, unit(), {
        let mut small = vec![push(d.e(), entry_of(int(0)))];
        small.extend(t.note_added(d.e(), k.e()));
        small.push(n.decl(sub(len(d.e()), int(1))));
        small.push(when(
            gt(n.e(), int(SMALL)),
            vec![t.go("rebuild", vec![d.e(), cap_for(n.e()), bool(false)])],
        ));
        small.push(ret_void());
        let mut args: Vec<Expr> = params.iter().map(|p| p.e()).collect();
        args.push(t.mixed(k.e()));
        vec![
            when(eq(t.ctrl(d.e()), int(0)), small),
            t.go("insert_hashed", args),
            ret_void(),
        ]
    }));
    // Remove entry `e`: shifted out of a container without an index,
    // else left a tombstone, the entries compacted once the dead
    // outnumber half the live.
    out_decls.push(define(&t.name("delete_at"), &[&d, &e], unit(), {
        let mut st = t.note_removed(d.e(), t.key_of(t.entry(d.e(), e.e())));
        st.extend([
            c.decl(t.ctrl(d.e())),
            when(
                eq(c.e(), int(0)),
                vec![
                    expr(mcall(d.e(), "remove_at", vec![e.e()], t.entry_ty())),
                    ret_void(),
                ],
            ),
            set_idx(d.e(), e.e(), t.hollow(int(TOMB))),
            index.decl(t.index_at(c.e())),
            dead.decl(add(idx(index.e(), int(0), i64()), int(1))),
            set_idx(index.e(), int(0), dead.e()),
            live.decl(sub(sub(len(d.e()), int(1)), dead.e())),
            when(
                gt(mul(dead.e(), int(2)), live.e()),
                vec![if_(
                    le(live.e(), int(SMALL)),
                    vec![t.go("unindex", vec![d.e()])],
                    vec![t.go("rebuild", vec![d.e(), cap_for(live.e()), bool(true)])],
                )],
            ),
            ret_void(),
        ]);
        st
    }));
    out_decls.push(define(&t.name("clear"), &[&d], unit(), {
        let mut st = vec![
            c.decl(t.ctrl(d.e())),
            old.decl(t.index_at(c.e())),
            expr(mcall(d.e(), "truncate", vec![int(1)], unit())),
            set_idx(d.e(), int(0), t.fresh()),
        ];
        if t.frees_index() {
            st.push(when(
                ne(c.e(), int(0)),
                vec![expr(call("zb_table_index_free", vec![old.e()], unit()))],
            ));
        }
        st.push(ret_void());
        st
    }));
    // A copy holding the live entries in order, with an index of its
    // own built from their hashes.
    out_decls.push(define(&t.name("copy"), &[&d], lt_.clone(), {
        let control = t.entry(d.e(), int(0));
        let first = t.make(
            int(0),
            if t.mask == Mask::Eager {
                t.key_of(control)
            } else {
                t.key.zero()
            },
            vf.map(|f| f.zero()),
        );
        vec![
            n.decl(len(d.e())),
            out.decl(list(Vec::new(), lt_.clone())),
            expr(mcall(out.e(), "reserve", vec![n.e()], unit())),
            push(out.e(), first),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    en.decl(t.entry(d.e(), i.e())),
                    when(
                        ne(t.hash_of(en.e()), int(TOMB)),
                        vec![push(out.e(), en.e())],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            live.decl(sub(len(out.e()), int(1))),
            when(
                and(ne(t.ctrl(d.e()), int(0)), gt(live.e(), int(SMALL))),
                vec![t.go("rebuild", vec![out.e(), cap_for(live.e()), bool(true)])],
            ),
            ret(out.e()),
        ]
    }));
    // The entries by position, 0 up to `pair_count`, dead ones included:
    // what a walk that must see the storage as it is reads.
    out_decls.push(define(
        &t.name("pair_count"),
        &[&d],
        i64(),
        vec![ret(sub(len(d.e()), int(1)))],
    ));
    out_decls.push(define(
        &t.name("live_at"),
        &[&d, &p],
        boolean(),
        vec![ret(ne(
            t.hash_of(t.entry(d.e(), add(p.e(), int(1)))),
            int(TOMB),
        ))],
    ));
    out_decls.push(define(
        &t.name("key_at"),
        &[&d, &p],
        t.key.ty(),
        vec![ret(t.key_of(t.entry(d.e(), add(p.e(), int(1)))))],
    ));
    if let Some(vf) = vf {
        out_decls.push(define(
            &t.name("value_at"),
            &[&d, &p],
            vf.ty(),
            vec![ret(t.value_of(t.entry(d.e(), add(p.e(), int(1)))))],
        ));
        out_decls.push(define(&t.name("value_set_at"), &[&d, &p, &v], unit(), {
            vec![
                en.decl(t.entry(d.e(), add(p.e(), int(1)))),
                set_idx(
                    d.e(),
                    add(p.e(), int(1)),
                    t.make(t.hash_of(en.e()), t.key_of(en.e()), Some(v.e())),
                ),
                ret_void(),
            ]
        }));
    }
    // The raw list of a box this shape's tag was checked on, and a box
    // for the list.
    out_decls.push(extern_fn(
        &t.name("box_raw"),
        &[("d", lt_.clone()), ("tag", i32())],
        any(),
        Some("zyntax_box_ptr"),
    ));
    if !t.sfx.is_empty() {
        out_decls.push(extern_fn(
            &t.name("raw"),
            &[("x", any())],
            lt_.clone(),
            Some("zyntax_box_pointer"),
        ));
    }
    let carried = local("d", lt_.clone());
    out_decls.push(define(
        &t.name("box"),
        &[&carried],
        any(),
        vec![ret(call(
            &t.name("box_raw"),
            vec![carried.e(), int32(t.tag as i32)],
            any(),
        ))],
    ));
    let x = local("x", any());
    let what = if vf.is_some() { "a dict" } else { "a set" };
    out_decls.push(define(
        &t.name("unbox"),
        &[&x],
        lt_.clone(),
        vec![
            when(
                ne(
                    cast(call("zb_box_tag", vec![x.e()], i32()), i64()),
                    int(t.tag),
                ),
                vec![fatal(
                    "TypeError",
                    add(
                        text(&format!("expected {what}, got ")),
                        call("zb_any_type", vec![x.e()], string()),
                    ),
                )],
            ),
            ret(call(&t.name("raw"), vec![x.e()], lt_.clone())),
        ],
    ));
    out_decls
}

/// A list of `f`'s values as the library lists them: numbers and
/// strings unboxed, anything else as a dynamic value.
fn list_elem(f: &Field) -> (Type, Box<dyn Fn(Expr) -> Expr + '_>) {
    match f {
        Field::Int | Field::Float | Field::Str => (f.ty(), Box::new(|x| x)),
        other => (any(), Box::new(move |x| other.boxed(x))),
    }
}

/// The functions of the dict shape of `key` to `value` whose tag is
/// `tag`: `zb_dict_<op>` for the library's own shape (`suffix` empty),
/// `zb_dict_<op>_<suffix>` for one a frontend registers. Shapes a field
/// names must have been declared before.
pub fn dict_declarations(
    list_type: TypeId,
    suffix: &str,
    tag: i64,
    key: &Field,
    value: &Field,
) -> Vec<Decl> {
    let t = Table {
        list_type,
        prefix: "zb_dict",
        sfx: if suffix.is_empty() {
            String::new()
        } else {
            format!("_{suffix}")
        },
        key,
        value: Some(value),
        tag,
        mask: Mask::None,
    };
    let mut out = table_core(&t);
    out.extend(dict_ops(&t));
    if suffix.is_empty() {
        out
    } else {
        crate::lists::generated(out)
    }
}

fn dict_ops(t: &Table) -> Vec<Decl> {
    let lt_ = t.list_ty();
    let anys = t.anys();
    let vf = t.value_field();
    let d = borrowed("d", lt_.clone());
    let other = borrowed("other", lt_.clone());
    let key = borrowed("k", t.key.ty());
    let k = kept("k", t.key.ty());
    let v = kept("v", vf.ty());
    let default = kept("default", vf.ty());
    let h = local("h", i64());
    let e = local("e", i64());
    let j = local("j", i64());
    let i = local("i", i64());
    let n = local("n", i64());
    let en = local("en", t.entry_ty());
    let x = local("x", vf.ty());
    let out = local("out", any());
    let oth = local("indexed", boolean());
    let find = |d: Expr, k: Expr| t.call("find", vec![d, k], i64());
    let value_at = |d: Expr, e: Expr| t.value_of(t.entry(d, e));
    // The value at entry `e` replaced by `v`, the key kept: a store
    // under an equal key of another kind keeps the first key.
    let store_at = |d: Expr, e: Expr, v: Expr| {
        vec![
            en.decl(t.entry(d.clone(), e.clone())),
            set_idx(d, e, t.make(t.hash_of(en.e()), t.key_of(en.e()), Some(v))),
        ]
    };
    let missing = || t.call("missing", vec![key.e()], vf.ty());
    let mut d_out = Vec::new();

    // A lookup that found nothing: an error path, kept out of line.
    d_out.push(define_cold(
        &t.name("missing"),
        &[&key],
        vf.ty(),
        vec![fatal("KeyError", t.key_text(key.e())), ret(vf.zero())],
    ));
    d_out.push(define(
        &t.name("contains"),
        &[&d, &key],
        boolean(),
        vec![ret(ge(find(d.e(), key.e()), int(0)))],
    ));
    d_out.push(define(
        &t.name("get"),
        &[&d, &key],
        vf.ty(),
        vec![
            e.decl(find(d.e(), key.e())),
            when(lt(e.e(), int(0)), vec![ret(missing())]),
            ret(value_at(d.e(), e.e())),
        ],
    ));
    d_out.push(define(
        &t.name("get_default"),
        &[&d, &key, &default],
        vf.ty(),
        vec![
            e.decl(find(d.e(), key.e())),
            when(lt(e.e(), int(0)), vec![ret(default.e())]),
            ret(value_at(d.e(), e.e())),
        ],
    ));
    // Store `v` under `k`; the key is hashed once on the indexed path.
    d_out.push(define(&t.name("set"), &[&d, &k, &v], unit(), {
        let mut small = vec![e.decl(find(d.e(), k.e()))];
        small.push(when(ge(e.e(), int(0)), {
            let mut s = store_at(d.e(), e.e(), v.e());
            s.push(ret_void());
            s
        }));
        small.push(t.go("insert", vec![d.e(), k.e(), v.e()]));
        small.push(ret_void());
        let mut hit = store_at(d.e(), e.e(), v.e());
        hit.push(ret_void());
        vec![
            when(eq(t.ctrl(d.e()), int(0)), small),
            h.decl(t.mixed(k.e())),
            e.decl(t.call("find_hashed", vec![d.e(), k.e(), h.e()], i64())),
            when(ge(e.e(), int(0)), hit),
            t.go("insert_hashed", vec![d.e(), k.e(), v.e(), h.e()]),
            ret_void(),
        ]
    }));
    // The same, into a container with an index, the key's mixed hash
    // known.
    d_out.push(define(&t.name("set_hashed"), &[&d, &k, &v, &h], unit(), {
        let mut hit = store_at(d.e(), e.e(), v.e());
        hit.push(ret_void());
        vec![
            e.decl(t.call("find_hashed", vec![d.e(), k.e(), h.e()], i64())),
            when(ge(e.e(), int(0)), hit),
            t.go("insert_hashed", vec![d.e(), k.e(), v.e(), h.e()]),
            ret_void(),
        ]
    }));
    // The position of `k`'s entry, `k` stored with the value's zero
    // first when absent: a read and a store of one key through one
    // lookup.
    d_out.push(define(&t.name("entry_or_insert"), &[&d, &k], i64(), {
        vec![
            e.decl(find(d.e(), k.e())),
            when(ge(e.e(), int(0)), vec![ret(sub(e.e(), int(1)))]),
            t.go("insert", vec![d.e(), k.e(), vf.zero()]),
            ret(sub(len(d.e()), int(2))),
        ]
    }));
    d_out.push(define(&t.name("setdefault"), &[&d, &k, &v], vf.ty(), {
        vec![
            e.decl(find(d.e(), k.e())),
            when(ge(e.e(), int(0)), vec![ret(value_at(d.e(), e.e()))]),
            t.go("insert", vec![d.e(), k.e(), v.e()]),
            ret(v.e()),
        ]
    }));
    d_out.push(define(&t.name("del"), &[&d, &key], unit(), {
        vec![
            e.decl(find(d.e(), key.e())),
            when(lt(e.e(), int(0)), vec![expr(missing()), ret_void()]),
            t.go("delete_at", vec![d.e(), e.e()]),
            ret_void(),
        ]
    }));
    d_out.push(define(&t.name("pop"), &[&d, &key], vf.ty(), {
        vec![
            e.decl(find(d.e(), key.e())),
            when(lt(e.e(), int(0)), vec![ret(missing())]),
            x.decl(value_at(d.e(), e.e())),
            t.go("delete_at", vec![d.e(), e.e()]),
            ret(x.e()),
        ]
    }));
    d_out.push(define(
        &t.name("pop_default"),
        &[&d, &key, &default],
        vf.ty(),
        {
            vec![
                e.decl(find(d.e(), key.e())),
                when(lt(e.e(), int(0)), vec![ret(default.e())]),
                x.decl(value_at(d.e(), e.e())),
                t.go("delete_at", vec![d.e(), e.e()]),
                ret(x.e()),
            ]
        },
    ));
    // The last pair, removed, as a tuple.
    d_out.push(define(&t.name("popitem"), &[&d], any(), {
        vec![
            e.decl(sub(len(d.e()), int(1))),
            while_(
                and(
                    ge(e.e(), int(1)),
                    eq(t.hash_of(t.entry(d.e(), e.e())), int(TOMB)),
                ),
                vec![e.set(sub(e.e(), int(1)))],
            ),
            when(
                lt(e.e(), int(1)),
                vec![fatal("KeyError", text("popitem(): dictionary is empty"))],
            ),
            en.decl(t.entry(d.e(), e.e())),
            out.decl(call(
                "zb_box_tuple",
                vec![list(
                    vec![t.key.boxed(t.key_of(en.e())), vf.boxed(t.value_of(en.e()))],
                    anys.clone(),
                )],
                any(),
            )),
            t.go("delete_at", vec![d.e(), e.e()]),
            ret(out.e()),
        ]
    }));
    // Every key, every value, every (key, value) tuple, in order.
    let live_walk = |d: &Local, body: Vec<Stmt>| {
        vec![
            n.decl(len(d.e())),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    en.decl(t.entry(d.e(), i.e())),
                    when(ne(t.hash_of(en.e()), int(TOMB)), body),
                    i.add_assign(int(1)),
                ],
            ),
        ]
    };
    for (op, field, read) in [("keys", t.key, 1i64), ("values", vf, 2i64)] {
        let (elem, conv) = list_elem(field);
        let elems = list_of(t.list_type, elem);
        let xs = local("xs", elems.clone());
        let item = idx(en.e(), int(read), field.ty());
        let mut st = vec![
            xs.decl(list(Vec::new(), elems.clone())),
            expr(mcall(xs.e(), "reserve", vec![len(d.e())], unit())),
        ];
        st.extend(live_walk(&d, vec![push(xs.e(), conv(item))]));
        st.push(ret(xs.e()));
        d_out.push(define(&t.name(op), &[&d], elems, st));
    }
    let xs = local("xs", anys.clone());
    d_out.push(define(&t.name("items"), &[&d], anys.clone(), {
        let mut st = vec![
            xs.decl(list(Vec::new(), anys.clone())),
            expr(mcall(xs.e(), "reserve", vec![len(d.e())], unit())),
        ];
        st.extend(live_walk(
            &d,
            vec![push(
                xs.e(),
                call(
                    "zb_box_tuple",
                    vec![list(
                        vec![t.key.boxed(t.key_of(en.e())), vf.boxed(t.value_of(en.e()))],
                        anys.clone(),
                    )],
                    any(),
                ),
            )],
        ));
        st.push(ret(xs.e()));
        st
    }));
    // Every pair of `other` stored into `d`; when both have an index the
    // hashes `other` holds are used again.
    d_out.push(define(&t.name("update"), &[&d, &other], unit(), {
        let kk = t.key_of(en.e());
        let vv = t.value_of(en.e());
        let mut st = vec![oth.decl(ne(t.ctrl(other.e()), int(0)))];
        st.extend(live_walk(
            &other,
            vec![if_(
                and(oth.e(), ne(t.ctrl(d.e()), int(0))),
                vec![t.go(
                    "set_hashed",
                    vec![d.e(), kk.clone(), vv.clone(), t.hash_of(en.e())],
                )],
                vec![t.go("set", vec![d.e(), kk, vv])],
            )],
        ));
        st.push(ret_void());
        st
    }));
    // Equal when the sizes match and every pair of one is in the other.
    let find_eq = if t.recursive_keys() {
        "find_eq"
    } else {
        "find"
    };
    d_out.push(define(&t.name("eq"), &[&d, &other], boolean(), {
        let mut st = vec![when(
            ne(
                t.call("len", vec![d.e()], i64()),
                t.call("len", vec![other.e()], i64()),
            ),
            vec![ret(bool(false))],
        )];
        st.extend(live_walk(
            &d,
            vec![
                j.decl(t.call(find_eq, vec![other.e(), t.key_of(en.e())], i64())),
                when(lt(j.e(), int(0)), vec![ret(bool(false))]),
                when(
                    not(vf.eq(t.value_of(en.e()), value_at(other.e(), j.e()))),
                    vec![ret(bool(false))],
                ),
            ],
        ));
        st.push(ret(bool(true)));
        st
    }));
    // Pieces joined once, so the text costs what its length is.
    let strs = list_of(t.list_type, string());
    let pieces = local("pieces", strs.clone());
    let first = local("first", boolean());
    let piece = |p: Expr| expr(mcall(pieces.e(), "push", vec![p], unit()));
    d_out.push(define(&t.name("repr"), &[&d], string(), {
        let mut st = vec![
            pieces.decl(list(Vec::new(), strs.clone())),
            piece(text("{")),
            first.decl(bool(true)),
        ];
        st.extend(live_walk(
            &d,
            vec![
                when(not(first.e()), vec![piece(text(", "))]),
                first.set(bool(false)),
                piece(t.key.repr(t.key_of(en.e()))),
                piece(text(": ")),
                piece(vf.repr(t.value_of(en.e()))),
            ],
        ));
        st.push(piece(text("}")));
        st.push(ret(call(
            "zb_str_join",
            vec![text(""), pieces.e()],
            string(),
        )));
        st
    }));
    // A dict is not a key.
    d_out.push(define(
        &t.name("hash"),
        &[&d],
        i64(),
        vec![
            fatal("TypeError", text("unhashable type: 'dict'")),
            ret(int(0)),
        ],
    ));
    d_out
}

/// The library dict's own functions beyond a shape's: built from a
/// literal's keys and values, and read out of a box as the box itself.
fn dynamic_dict(list_type: TypeId) -> Vec<Decl> {
    let dt = dict_type(list_type);
    let anys = list_of(list_type, any());
    let t = Table {
        list_type,
        prefix: "zb_dict",
        sfx: String::new(),
        key: &Field::Any,
        value: Some(&Field::Any),
        tag: DICT_TAG,
        mask: Mask::None,
    };
    let pairs = borrowed("pairs", anys.clone());
    let out = local("out", dt.clone());
    let n = local("n", i64());
    let i = local("i", i64());
    let live = local("live", i64());
    let x = local("x", any());
    let mut d = Vec::new();
    // A dict from a literal's keys and values, later pairs winning.
    d.push(define("zb_dict_from_pairs", &[&pairs], dt.clone(), {
        vec![
            out.decl(call("zb_dict_new", vec![], dt.clone())),
            n.decl(len(pairs.e())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    expr(call(
                        "zb_dict_set",
                        vec![
                            out.e(),
                            at(pairs.e(), i.e()),
                            at(pairs.e(), add(i.e(), int(1))),
                        ],
                        unit(),
                    )),
                    i.add_assign(int(2)),
                ],
            ),
            ret(out.e()),
        ]
    }));
    // A dict from a literal whose keys are known to be distinct, laid
    // out as a slot then key, value, key, value: the pairs become the
    // entries as they are, unhashed until the dict takes an index.
    d.push(define("zb_dict_from_distinct", &[&pairs], dt.clone(), {
        vec![
            n.decl(len(pairs.e())),
            out.decl(list(Vec::new(), dt.clone())),
            expr(mcall(
                out.e(),
                "reserve",
                vec![add(div(n.e(), int(2)), int(1))],
                unit(),
            )),
            push(out.e(), t.fresh()),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    push(
                        out.e(),
                        t.make(
                            int(0),
                            at(pairs.e(), i.e()),
                            Some(at(pairs.e(), add(i.e(), int(1)))),
                        ),
                    ),
                    i.add_assign(int(2)),
                ],
            ),
            live.decl(sub(len(out.e()), int(1))),
            when(
                gt(live.e(), int(SMALL)),
                vec![expr(call(
                    "zb_dict_rebuild",
                    vec![
                        out.e(),
                        call("zb_table_cap_for", vec![live.e()], i64()),
                        bool(false),
                    ],
                    unit(),
                ))],
            ),
            ret(out.e()),
        ]
    }));
    // The box itself, once it is known to hold a dict: what a slot that
    // stores dicts as boxes takes from a dynamic value.
    d.push(define(
        "zb_dict_as_box",
        &[&x],
        any(),
        vec![
            expr(call("zb_dict_unbox", vec![x.e()], dt.clone())),
            ret(x.e()),
        ],
    ));
    d
}

/// The lookups of the library's dict by a key that is not a box, of
/// field `probe`: `zb_dict_{find_hashed,find,contains,get,get_default,
/// set}_<suffix>`. The key hashes as its box would and compares with a
/// stored box as the box would, so a key already present costs no box;
/// one stored is boxed then.
pub(crate) fn dict_probe_by(list_type: TypeId, probe: &Field, suffix: &str) -> Vec<Decl> {
    let dt = dict_type(list_type);
    let t = Table {
        list_type,
        prefix: "zb_dict",
        sfx: String::new(),
        key: &Field::Any,
        value: Some(&Field::Any),
        tag: DICT_TAG,
        mask: Mask::None,
    };
    let d = borrowed("d", dt.clone());
    let key = borrowed("key", probe.ty());
    let v = kept("v", any());
    let default = kept("default", any());
    let n = local("n", i64());
    let i = local("i", i64());
    let e = local("e", i64());
    let en = local("en", t.entry_ty());
    let index = local("index", list_of(list_type, i64()));
    let mask = local("mask", i64());
    let slot = local("slot", i64());
    let h = local("h", i64());
    let stored = local("stored", any());
    let name = |op: &str| format!("zb_dict_{op}_{suffix}");
    // Whether the box `stored` holds a value equal to the key. A string
    // box's text is read only once the box is known to hold one.
    let when_matches = |found: Expr| match probe {
        Field::Str => when(
            eq(
                call("zb_any_category", vec![stored.e()], i64()),
                int(crate::dynamic::STR),
            ),
            vec![when(
                call(
                    "zb_str_eq",
                    vec![call("zb_box_get_str", vec![stored.e()], string()), key.e()],
                    boolean(),
                ),
                vec![ret(found)],
            )],
        ),
        other => when(other.eq_boxed(stored.e(), key.e()), vec![ret(found)]),
    };
    // The box a stored key becomes. A string is copied: the caller may
    // release the one it passed once the call returns.
    let boxed_key = || match probe {
        Field::Str => call("zb_str_to_dynamic", vec![key.e()], any()),
        other => other.boxed(key.e()),
    };
    let key_text = || match probe {
        Field::Str => key.e(),
        other => other.repr(key.e()),
    };
    let mixed = || call("zb_hash_mix", vec![probe.hash(key.e())], i64());
    let mut out = Vec::new();
    out.push(define(
        &name("find_hashed"),
        &[&d, &key, &h],
        i64(),
        vec![
            index.decl(t.index_at(t.ctrl(d.e()))),
            mask.decl(sub(sub(len(index.e()), int(INDEX_HEAD)), int(1))),
            slot.decl(bitand(h.e(), mask.e())),
            i.decl(int(0)),
            while_(
                le(i.e(), mask.e()),
                vec![
                    e.decl(idx(index.e(), add(slot.e(), int(INDEX_HEAD)), i64())),
                    when(lt(e.e(), int(0)), vec![ret(int(-1))]),
                    en.decl(t.entry(d.e(), e.e())),
                    when(
                        eq(t.hash_of(en.e()), h.e()),
                        vec![stored.decl(t.key_of(en.e())), when_matches(e.e())],
                    ),
                    slot.set(bitand(add(slot.e(), int(1)), mask.e())),
                    i.add_assign(int(1)),
                ],
            ),
            ret(int(-1)),
        ],
    ));
    out.push(define(
        &name("find"),
        &[&d, &key],
        i64(),
        vec![
            when(
                eq(t.ctrl(d.e()), int(0)),
                vec![
                    n.decl(len(d.e())),
                    i.decl(int(1)),
                    while_(
                        lt(i.e(), n.e()),
                        vec![
                            stored.decl(t.key_of(t.entry(d.e(), i.e()))),
                            when_matches(i.e()),
                            i.add_assign(int(1)),
                        ],
                    ),
                    ret(int(-1)),
                ],
            ),
            ret(call(
                &name("find_hashed"),
                vec![d.e(), key.e(), mixed()],
                i64(),
            )),
        ],
    ));
    let find = || call(&name("find"), vec![d.e(), key.e()], i64());
    let value_at = |e: Expr| t.value_of(t.entry(d.e(), e));
    out.push(define(
        &name("contains"),
        &[&d, &key],
        boolean(),
        vec![ret(ge(find(), int(0)))],
    ));
    out.push(define(
        &name("get"),
        &[&d, &key],
        any(),
        vec![
            e.decl(find()),
            when(lt(e.e(), int(0)), vec![fatal("KeyError", key_text())]),
            ret(value_at(e.e())),
        ],
    ));
    out.push(define(
        &name("get_default"),
        &[&d, &key, &default],
        any(),
        vec![
            e.decl(find()),
            when(lt(e.e(), int(0)), vec![ret(default.e())]),
            ret(value_at(e.e())),
        ],
    ));
    let store = |e: Expr| {
        vec![
            en.decl(t.entry(d.e(), e.clone())),
            set_idx(
                d.e(),
                e,
                t.make(t.hash_of(en.e()), t.key_of(en.e()), Some(v.e())),
            ),
            ret_void(),
        ]
    };
    out.push(define(
        &name("set"),
        &[&d, &key, &v],
        unit(),
        vec![
            when(
                eq(t.ctrl(d.e()), int(0)),
                vec![
                    e.decl(find()),
                    when(ge(e.e(), int(0)), store(e.e())),
                    expr(call(
                        "zb_dict_insert",
                        vec![d.e(), boxed_key(), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            h.decl(mixed()),
            e.decl(call(
                &name("find_hashed"),
                vec![d.e(), key.e(), h.e()],
                i64(),
            )),
            when(ge(e.e(), int(0)), store(e.e())),
            expr(call(
                "zb_dict_insert_hashed",
                vec![d.e(), boxed_key(), v.e(), h.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));
    out
}

/// `name(s, key) -> bool`: whether a set holds a value matching `key`,
/// a value of `key_ty` that is not a box. `hash_of` is the key's hash
/// as [`zb_any_hash`] would hash its boxed form, so the probe lands on
/// the same slot; `matches(stored, key)` compares a stored box with it.
/// A set below its table size is scanned instead.
pub(crate) fn set_contains_by(
    name: &str,
    list_type: TypeId,
    key_ty: Type,
    hash_of: &dyn Fn(Expr) -> Expr,
    matches: &dyn Fn(Expr, Expr) -> Expr,
) -> Decl {
    let anys = list_of(list_type, any());
    let ints = list_of(list_type, i64());
    let s = borrowed("s", anys.clone());
    let key = borrowed("key", key_ty);
    let n = local("n", i64());
    let i = local("i", i64());
    let index = local("index", ints.clone());
    let mask = local("mask", i64());
    let slot = local("slot", i64());
    let entry = local("e", i64());
    let h = local("h", i64());
    let mut scan = vec![n.decl(len(s.e()))];
    scan.extend(for_range(
        &i,
        int(1),
        n.e(),
        vec![when(
            matches(at(s.e(), i.e()), key.e()),
            vec![ret(bool(true))],
        )],
    ));
    scan.push(ret(bool(false)));
    define(
        name,
        &[&s, &key],
        boolean(),
        vec![
            when(eq(at(s.e(), int(0)), null(any())), scan),
            index.decl(call(
                "zb_unbox_list_raw_i64",
                vec![at(s.e(), int(0))],
                ints.clone(),
            )),
            mask.decl(sub(div(sub(len(index.e()), int(1)), int(2)), int(1))),
            h.decl(call("zb_hash_mix", vec![hash_of(key.e())], i64())),
            slot.decl(bitand(h.e(), mask.e())),
            i.decl(int(0)),
            while_(
                le(i.e(), mask.e()),
                vec![
                    entry.decl(idx(index.e(), add(mul(slot.e(), int(2)), int(1)), i64())),
                    when(lt(entry.e(), int(0)), vec![ret(bool(false))]),
                    when(
                        and(
                            eq(
                                idx(index.e(), add(mul(slot.e(), int(2)), int(2)), i64()),
                                h.e(),
                            ),
                            matches(at(s.e(), add(entry.e(), int(1))), key.e()),
                        ),
                        vec![ret(bool(true))],
                    ),
                    slot.set(bitand(add(slot.e(), int(1)), mask.e())),
                    i.add_assign(int(1)),
                ],
            ),
            ret(bool(false)),
        ],
    )
}

fn set(list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let ints = list_of(list_type, i64());
    // Every function edits the set in place; only boxing keeps it.
    let s = borrowed("s", anys.clone());
    let other = borrowed("other", anys.clone());
    let v = kept("v", any());
    // Lookups and removals read the value and keep nothing.
    let value = borrowed("v", any());
    let i = local("i", i64());
    let n = local("n", i64());
    let out = local("out", anys.clone());
    let text_out = local("text", string());
    let x = local("x", any());
    let tag = local("tag", i64());
    let mask = local("mask", i64());
    let left_mask = local("left_mask", i64());
    let right_mask = local("right_mask", i64());
    let member = local("member", any());
    let bit = local("bit", i64());
    let wanted = local("wanted", i64());
    let index = borrowed("index", ints.clone());
    let slot = local("slot", i64());
    let entry = local("e", i64());
    let cap = local("cap", i64());
    let h = local("h", i64());
    let kcat = local("kcat", i64());
    let stored = local("stored", any());
    let mut d = Vec::new();

    // A set is laid out as a dict is: position 0 holds the hash index
    // (None while the set is small), the values follow in insertion
    // order, so value `e` is at position 1 + e. The table's first word
    // caches the set's small-int mask (MASK_UNKNOWN until asked for,
    // -1 when the values are not all small ints); then two words per
    // slot, the entry number (-1 when empty) and the entry's hash, so
    // growing it and copying it hash nothing again and a probe reads a
    // value only when its hash agrees.
    let count = |s: Expr| sub(len(s), int(1));
    let value_at = |s: Expr, e: Expr| at(s, add(e, int(1)));
    let index_of = |s: Expr| call("zb_unbox_list_raw_i64", vec![at(s, int(0))], ints.clone());
    let box_index = |index: Expr| call("zb_list_box_i64", vec![index], any());
    let entry_at = |index: Expr, slot: Expr| idx(index, add(mul(slot, int(2)), int(1)), i64());
    let hash_at = |index: Expr, slot: Expr| idx(index, add(mul(slot, int(2)), int(2)), i64());
    let slots_of = |index: Expr| div(sub(len(index), int(1)), int(2));
    /// The cached mask word before it is computed.
    const MASK_UNKNOWN: i64 = -2;
    let forget_mask = |index: Expr| set_idx(index, int(0), int(MASK_UNKNOWN));
    let hash = |k: Expr| call("zb_dict_key_hash", vec![k], i64());
    let next_slot = |s: Expr, mask: Expr| bitand(add(s, int(1)), mask);
    let unindexed = |s: Expr| eq(at(s, int(0)), null(any()));
    let reuse = local("reuse", boolean());
    let old = local("old", ints.clone());
    let j = local("j", i64());
    let contains = |s: Expr, v: Expr| call("zb_set_contains", vec![s, v], boolean());
    let find = |s: Expr, v: Expr| call("zb_set_find", vec![s, v], i64());
    /// Values a set holds before it takes a table.
    const SMALL: i64 = 8;
    /// The first table's size, for a set just past `SMALL`.
    const FIRST_TABLE: i64 = 32;

    d.push(define(
        "zb_set_new",
        &[],
        anys.clone(),
        vec![
            out.decl(list(Vec::new(), anys.clone())),
            push(out.e(), null(any())),
            ret(out.e()),
        ],
    ));
    d.push(define("zb_set_len", &[&s], i64(), vec![ret(count(s.e()))]));
    // An empty table of `cap` slots: every word -1, which is the empty
    // entry mark, and a hash no probe reads.
    d.push(define("zb_set_index_new", &[&cap], ints.clone(), {
        let table = local("table", ints.clone());
        vec![
            table.decl(list(Vec::new(), ints.clone())),
            expr(mcall(
                table.e(),
                "resize_filled",
                vec![add(mul(cap.e(), int(2)), int(1)), int(0xFF)],
                unit(),
            )),
            forget_mask(table.e()),
            ret(table.e()),
        ]
    }));
    // Record entry `e`, whose hash is `h`, in `index`, which has room.
    d.push(define(
        "zb_set_place_hashed",
        &[&index, &entry, &h],
        unit(),
        vec![
            mask.decl(sub(slots_of(index.e()), int(1))),
            slot.decl(bitand(h.e(), mask.e())),
            while_(
                ge(entry_at(index.e(), slot.e()), int(0)),
                vec![slot.set(next_slot(slot.e(), mask.e()))],
            ),
            set_idx(index.e(), add(mul(slot.e(), int(2)), int(1)), entry.e()),
            set_idx(index.e(), add(mul(slot.e(), int(2)), int(2)), h.e()),
            ret_void(),
        ],
    ));
    // A fresh table of `cap` slots over every value of `s`: from the
    // hashes the old table holds when the entries kept their positions
    // (`reuse`), else from the values.
    d.push(define("zb_set_reindex", &[&s, &cap, &reuse], unit(), {
        let mut from_values = vec![n.decl(count(s.e()))];
        from_values.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(call(
                "zb_set_place_hashed",
                vec![index.e(), i.e(), hash(value_at(s.e(), i.e()))],
                unit(),
            ))],
        ));
        let mut from_table = vec![old.decl(index_of(s.e())), n.decl(slots_of(old.e()))];
        from_table.extend(for_range(
            &j,
            int(0),
            n.e(),
            vec![
                entry.decl(entry_at(old.e(), j.e())),
                when(
                    ge(entry.e(), int(0)),
                    vec![expr(call(
                        "zb_set_place_hashed",
                        vec![index.e(), entry.e(), hash_at(old.e(), j.e())],
                        unit(),
                    ))],
                ),
            ],
        ));
        vec![
            index.decl(call("zb_set_index_new", vec![cap.e()], ints.clone())),
            if_(
                and(reuse.e(), not(unindexed(s.e()))),
                from_table,
                from_values,
            ),
            set_idx(s.e(), int(0), box_index(index.e())),
            ret_void(),
        ]
    }));
    // A set built by appending distinct values takes its table once it
    // is past a few of them.
    d.push(define(
        "zb_set_settle",
        &[&s],
        unit(),
        vec![
            when(
                and(unindexed(s.e()), gt(count(s.e()), int(SMALL))),
                vec![
                    cap.decl(int(FIRST_TABLE)),
                    while_(
                        lt(cap.e(), mul(count(s.e()), int(2))),
                        vec![cap.set(mul(cap.e(), int(2)))],
                    ),
                    expr(call(
                        "zb_set_reindex",
                        vec![s.e(), cap.e(), bool(false)],
                        unit(),
                    )),
                ],
            ),
            ret_void(),
        ],
    ));
    // The position of `v` in a set with a table, given its hash, or -1;
    // and its position in any set, or -1.
    d.push(define(
        "zb_set_find_hashed",
        &[&s, &value, &h],
        i64(),
        vec![
            index.decl(index_of(s.e())),
            mask.decl(sub(slots_of(index.e()), int(1))),
            slot.decl(bitand(h.e(), mask.e())),
            kcat.decl(call("zb_any_category", vec![v.e()], i64())),
            i.decl(int(0)),
            while_(le(i.e(), mask.e()), {
                let mut matched = vec![stored.decl(value_at(s.e(), entry.e()))];
                matched.extend(when_key_matches(&stored, &v, &kcat, add(entry.e(), int(1))));
                vec![
                    entry.decl(entry_at(index.e(), slot.e())),
                    when(lt(entry.e(), int(0)), vec![ret(int(-1))]),
                    when(eq(hash_at(index.e(), slot.e()), h.e()), matched),
                    slot.set(next_slot(slot.e(), mask.e())),
                    i.add_assign(int(1)),
                ]
            }),
            ret(int(-1)),
        ],
    ));
    d.push(define(
        "zb_set_find",
        &[&s, &value],
        i64(),
        vec![
            when(
                unindexed(s.e()),
                vec![
                    n.decl(len(s.e())),
                    i.decl(int(1)),
                    while_(
                        lt(i.e(), n.e()),
                        vec![
                            when(key_matches(at(s.e(), i.e()), v.e()), vec![ret(i.e())]),
                            i.add_assign(int(1)),
                        ],
                    ),
                    ret(int(-1)),
                ],
            ),
            ret(call(
                "zb_set_find_hashed",
                vec![s.e(), v.e(), hash(v.e())],
                i64(),
            )),
        ],
    ));
    d.push(define(
        "zb_set_contains",
        &[&s, &value],
        boolean(),
        vec![ret(ge(find(s.e(), v.e()), int(0)))],
    ));
    // Add a value known to be absent whose hash is `h`, growing the
    // table first when the value would bring it to half full.
    d.push(define(
        "zb_set_insert_hashed",
        &[&s, &v, &h],
        unit(),
        vec![
            entry.decl(count(s.e())),
            cap.decl(slots_of(index_of(s.e()))),
            when(
                gt(mul(add(entry.e(), int(1)), int(2)), cap.e()),
                vec![expr(call(
                    "zb_set_reindex",
                    vec![s.e(), mul(cap.e(), int(2)), bool(true)],
                    unit(),
                ))],
            ),
            push(s.e(), v.e()),
            expr(call(
                "zb_set_place_hashed",
                vec![index_of(s.e()), entry.e(), h.e()],
                unit(),
            )),
            forget_mask(index_of(s.e())),
            ret_void(),
        ],
    ));
    // The hash of each value, by entry: read off the table when the set
    // has one, else computed; a set past `SMALL` values always has one.
    let hs = local("hs", ints.clone());
    d.push(define("zb_set_entry_hashes", &[&s], ints.clone(), {
        let from_values = for_range(
            &i,
            int(0),
            n.e(),
            vec![set_idx(hs.e(), i.e(), hash(value_at(s.e(), i.e())))],
        );
        let mut from_table = vec![index.decl(index_of(s.e())), cap.decl(slots_of(index.e()))];
        from_table.extend(for_range(
            &j,
            int(0),
            cap.e(),
            vec![
                entry.decl(entry_at(index.e(), j.e())),
                when(
                    ge(entry.e(), int(0)),
                    vec![set_idx(hs.e(), entry.e(), hash_at(index.e(), j.e()))],
                ),
            ],
        ));
        vec![
            n.decl(count(s.e())),
            hs.decl(list(Vec::new(), ints.clone())),
            expr(mcall(hs.e(), "resize_filled", vec![n.e(), int(0)], unit())),
            if_(unindexed(s.e()), from_values, from_table),
            ret(hs.e()),
        ]
    }));
    // Whether `v`, whose hash is `h`, is in `s`: by the table when there
    // is one, else by the short scan.
    d.push(define(
        "zb_set_has_hashed",
        &[&s, &value, &h],
        boolean(),
        vec![
            when(unindexed(s.e()), vec![ret(ge(find(s.e(), v.e()), int(0)))]),
            ret(ge(
                call("zb_set_find_hashed", vec![s.e(), v.e(), h.e()], i64()),
                int(0),
            )),
        ],
    ));
    // A set built from distinct values whose hashes are known: the
    // table, when the count asks for one, is filled from those hashes.
    let out_hs = borrowed("out_hs", ints.clone());
    d.push(define("zb_set_settle_hashed", &[&s, &out_hs], unit(), {
        let mut place = vec![
            cap.decl(int(FIRST_TABLE)),
            while_(
                lt(cap.e(), mul(n.e(), int(2))),
                vec![cap.set(mul(cap.e(), int(2)))],
            ),
            index.decl(call("zb_set_index_new", vec![cap.e()], ints.clone())),
        ];
        place.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(call(
                "zb_set_place_hashed",
                vec![index.e(), i.e(), idx(out_hs.e(), i.e(), i64())],
                unit(),
            ))],
        ));
        place.push(set_idx(s.e(), int(0), box_index(index.e())));
        vec![
            n.decl(count(s.e())),
            when(gt(n.e(), int(SMALL)), place),
            ret_void(),
        ]
    }));
    d.push(define(
        "zb_set_add",
        &[&s, &v],
        unit(),
        vec![
            when(
                unindexed(s.e()),
                vec![
                    when(
                        lt(find(s.e(), v.e()), int(0)),
                        vec![
                            push(s.e(), v.e()),
                            when(
                                gt(count(s.e()), int(SMALL)),
                                vec![expr(call(
                                    "zb_set_reindex",
                                    vec![s.e(), int(FIRST_TABLE), bool(false)],
                                    unit(),
                                ))],
                            ),
                        ],
                    ),
                    ret_void(),
                ],
            ),
            h.decl(hash(v.e())),
            when(
                lt(
                    call("zb_set_find_hashed", vec![s.e(), v.e(), h.e()], i64()),
                    int(0),
                ),
                vec![expr(call(
                    "zb_set_insert_hashed",
                    vec![s.e(), v.e(), h.e()],
                    unit(),
                ))],
            ),
            ret_void(),
        ],
    ));
    // Removal shifts the later values down, so the table is rebuilt.
    let remove_at = |i: &Local| {
        vec![
            expr(mcall(s.e(), "remove_at", vec![i.e()], any())),
            when(
                not(unindexed(s.e())),
                vec![expr(call(
                    "zb_set_reindex",
                    vec![s.e(), slots_of(index_of(s.e())), bool(false)],
                    unit(),
                ))],
            ),
        ]
    };
    d.push(define("zb_set_discard", &[&s, &value], unit(), {
        let mut st = vec![i.decl(find(s.e(), v.e()))];
        st.push(when(ge(i.e(), int(0)), remove_at(&i)));
        st.push(ret_void());
        st
    }));
    d.push(define("zb_set_remove", &[&s, &value], unit(), {
        let mut st = vec![
            i.decl(find(s.e(), v.e())),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", any_str(v.e()))]),
        ];
        st.extend(remove_at(&i));
        st.push(ret_void());
        st
    }));
    d.push(define(
        "zb_set_clear",
        &[&s],
        unit(),
        vec![
            expr(mcall(s.e(), "clear", vec![], unit())),
            push(s.e(), null(any())),
            ret_void(),
        ],
    ));
    // The values, in order, as a list of their own.
    d.push(define("zb_set_items", &[&s], anys.clone(), {
        vec![
            out.decl(list(Vec::new(), anys.clone())),
            n.decl(len(s.e())),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![push(out.e(), at(s.e(), i.e())), i.add_assign(int(1))],
            ),
            ret(out.e()),
        ]
    }));
    // A copy with a table of its own: the values keep their positions,
    // so the table is copied rather than built again.
    d.push(define(
        "zb_set_copy",
        &[&s],
        anys.clone(),
        vec![
            out.decl(call("zb_list_copy_any", vec![s.e()], anys.clone())),
            when(
                not(unindexed(out.e())),
                vec![set_idx(
                    out.e(),
                    int(0),
                    box_index(call(
                        "zb_list_copy_i64",
                        vec![index_of(s.e())],
                        ints.clone(),
                    )),
                )],
            ),
            ret(out.e()),
        ],
    ));
    // A set from any list, keeping first occurrences.
    let xs = local("xs", anys.clone());
    d.push(define("zb_set_from", &[&xs], anys.clone(), {
        let mut st = vec![
            out.decl(call("zb_set_new", vec![], anys.clone())),
            n.decl(len(xs.e())),
        ];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(call(
                "zb_set_add",
                vec![out.e(), at(xs.e(), i.e())],
                unit(),
            ))],
        ));
        st.push(ret(out.e()));
        st
    }));
    // The hash of a set is the same whatever order it was filled in.
    d.push(define("zb_set_hash", &[&s], i64(), {
        vec![
            h.decl(int(0x2545_F491_4F6C_DD1D)),
            n.decl(len(s.e())),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    h.set(bitxor(h.e(), hash(at(s.e(), i.e())))),
                    i.add_assign(int(1)),
                ],
            ),
            ret(add(mul(h.e(), int(1_000_003)), count(s.e()))),
        ]
    }));

    // Integer board positions fit in a word. -1 means the set also
    // holds a value that needs the general equality path.
    // Computed by a scan of the values; a set with a table keeps the
    // answer in the table's first word until a value is added.
    d.push(define("zb_set_mask63_scan", &[&s], i64(), {
        let mut st = vec![mask.decl(int(0)), n.decl(len(s.e()))];
        st.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![
                member.decl(at(s.e(), i.e())),
                when(
                    ne(
                        cast(call("zb_box_tag", vec![member.e()], i32()), i64()),
                        int(crate::dynamic::I64_TAG),
                    ),
                    vec![ret(int(-1))],
                ),
                bit.decl(call("zb_box_payload_i64", vec![member.e()], i64())),
                when(
                    or(lt(bit.e(), int(0)), ge(bit.e(), int(63))),
                    vec![ret(int(-1))],
                ),
                mask.set(bitor(mask.e(), shl(int(1), bit.e()))),
            ],
        ));
        st.push(ret(mask.e()));
        st
    }));
    d.push(define(
        "zb_set_mask63",
        &[&s],
        i64(),
        vec![
            when(
                unindexed(s.e()),
                vec![ret(call("zb_set_mask63_scan", vec![s.e()], i64()))],
            ),
            index.decl(index_of(s.e())),
            mask.decl(idx(index.e(), int(0), i64())),
            when(ne(mask.e(), int(MASK_UNKNOWN)), vec![ret(mask.e())]),
            mask.set(call("zb_set_mask63_scan", vec![s.e()], i64())),
            set_idx(index.e(), int(0), mask.e()),
            ret(mask.e()),
        ],
    ));
    d.push(define("zb_set_all_kind", &[&s, &wanted], boolean(), {
        let mut st = vec![
            n.decl(len(s.e())),
            when(eq(n.e(), int(1)), vec![ret(bool(false))]),
        ];
        st.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![when(
                ne(
                    call("zb_any_kind", vec![at(s.e(), i.e())], i64()),
                    wanted.e(),
                ),
                vec![ret(bool(false))],
            )],
        ));
        st.push(ret(bool(true)));
        st
    }));
    d.push(define(
        "zb_set_disjoint_kinds",
        &[&s, &other],
        boolean(),
        vec![
            // Two sets whose first values are of one kind share it, and
            // an empty set has no kind to be all of.
            when(
                or(
                    or(eq(len(s.e()), int(1)), eq(len(other.e()), int(1))),
                    eq(
                        call("zb_any_kind", vec![at(s.e(), int(1))], i64()),
                        call("zb_any_kind", vec![at(other.e(), int(1))], i64()),
                    ),
                ),
                vec![ret(bool(false))],
            ),
            when(
                call("zb_set_all_kind", vec![s.e(), int(SET_TAG >> 8)], boolean()),
                vec![ret(call(
                    "zb_set_all_kind",
                    vec![other.e(), int(TUPLE_TAG >> 8)],
                    boolean(),
                ))],
            ),
            when(
                call(
                    "zb_set_all_kind",
                    vec![s.e(), int(TUPLE_TAG >> 8)],
                    boolean(),
                ),
                vec![ret(call(
                    "zb_set_all_kind",
                    vec![other.e(), int(SET_TAG >> 8)],
                    boolean(),
                ))],
            ),
            ret(bool(false)),
        ],
    ));

    // Intersection, union, difference, symmetric difference. A result
    // built from values known to be distinct is appended to and takes
    // its table at the end.
    let settled = |out: &Local| {
        vec![
            expr(call("zb_set_settle", vec![out.e()], unit())),
            ret(out.e()),
        ]
    };
    d.push(define("zb_set_and", &[&s, &other], anys.clone(), {
        let mut st = vec![
            out.decl(call("zb_set_new", vec![], anys.clone())),
            n.decl(len(s.e())),
            left_mask.decl(call("zb_set_mask63", vec![s.e()], i64())),
            when(ge(left_mask.e(), int(0)), {
                let mut fast = vec![right_mask.decl(call("zb_set_mask63", vec![other.e()], i64()))];
                let mut matched = for_range(
                    &i,
                    int(1),
                    n.e(),
                    vec![
                        member.decl(at(s.e(), i.e())),
                        bit.decl(call("zb_box_payload_i64", vec![member.e()], i64())),
                        when(
                            ne(bitand(right_mask.e(), shl(int(1), bit.e())), int(0)),
                            vec![push(out.e(), member.e())],
                        ),
                    ],
                );
                matched.extend(settled(&out));
                fast.push(when(ge(right_mask.e(), int(0)), matched));
                fast
            }),
            when(
                call("zb_set_disjoint_kinds", vec![s.e(), other.e()], boolean()),
                vec![ret(out.e())],
            ),
            hs.decl(call("zb_set_entry_hashes", vec![s.e()], ints.clone())),
            out_hs.decl(list(Vec::new(), ints.clone())),
            expr(mcall(out.e(), "reserve", vec![n.e()], unit())),
            expr(mcall(out_hs.e(), "reserve", vec![n.e()], unit())),
        ];
        st.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![
                h.decl(idx(hs.e(), sub(i.e(), int(1)), i64())),
                when(
                    call(
                        "zb_set_has_hashed",
                        vec![other.e(), at(s.e(), i.e()), h.e()],
                        boolean(),
                    ),
                    vec![push(out.e(), at(s.e(), i.e())), push(out_hs.e(), h.e())],
                ),
            ],
        ));
        st.push(expr(call(
            "zb_set_settle_hashed",
            vec![out.e(), out_hs.e()],
            unit(),
        )));
        st.push(ret(out.e()));
        st
    }));
    // Room for `extra` more values without the table growing as they
    // come: a table sized once for what a union adds.
    let extra = local("extra", i64());
    let need = local("need", i64());
    d.push(define(
        "zb_set_reserve",
        &[&s, &extra],
        unit(),
        vec![
            need.decl(mul(add(count(s.e()), extra.e()), int(2))),
            when(
                unindexed(s.e()),
                vec![
                    when(
                        gt(add(count(s.e()), extra.e()), int(SMALL)),
                        vec![
                            cap.decl(int(FIRST_TABLE)),
                            while_(lt(cap.e(), need.e()), vec![cap.set(mul(cap.e(), int(2)))]),
                            expr(call(
                                "zb_set_reindex",
                                vec![s.e(), cap.e(), bool(false)],
                                unit(),
                            )),
                        ],
                    ),
                    ret_void(),
                ],
            ),
            cap.decl(slots_of(index_of(s.e()))),
            when(
                lt(cap.e(), need.e()),
                vec![
                    while_(lt(cap.e(), need.e()), vec![cap.set(mul(cap.e(), int(2)))]),
                    expr(call(
                        "zb_set_reindex",
                        vec![s.e(), cap.e(), bool(true)],
                        unit(),
                    )),
                ],
            ),
            ret_void(),
        ],
    ));
    // Every value of `other` added to `s`, by the hashes `other` holds:
    // into the table when `s` has one after making room, else by the
    // short path.
    d.push(define("zb_set_update", &[&s, &other], unit(), {
        let mut st = vec![
            expr(call(
                "zb_set_reserve",
                vec![s.e(), count(other.e())],
                unit(),
            )),
            n.decl(len(other.e())),
        ];
        let mut by_hash = vec![hs.decl(call("zb_set_entry_hashes", vec![other.e()], ints.clone()))];
        by_hash.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![
                h.decl(idx(hs.e(), sub(i.e(), int(1)), i64())),
                when(
                    lt(
                        call(
                            "zb_set_find_hashed",
                            vec![s.e(), at(other.e(), i.e()), h.e()],
                            i64(),
                        ),
                        int(0),
                    ),
                    vec![expr(call(
                        "zb_set_insert_hashed",
                        vec![s.e(), at(other.e(), i.e()), h.e()],
                        unit(),
                    ))],
                ),
            ],
        ));
        let mut by_add = Vec::new();
        by_add.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![expr(call(
                "zb_set_add",
                vec![s.e(), at(other.e(), i.e())],
                unit(),
            ))],
        ));
        st.push(if_(unindexed(s.e()), by_add, by_hash));
        st.push(ret_void());
        st
    }));
    d.push(define(
        "zb_set_or",
        &[&s, &other],
        anys.clone(),
        vec![
            out.decl(call("zb_set_copy", vec![s.e()], anys.clone())),
            expr(call("zb_set_update", vec![out.e(), other.e()], unit())),
            ret(out.e()),
        ],
    ));
    d.push(define("zb_set_sub", &[&s, &other], anys.clone(), {
        let mut st = vec![
            out.decl(call("zb_set_new", vec![], anys.clone())),
            n.decl(len(s.e())),
            left_mask.decl(call("zb_set_mask63", vec![s.e()], i64())),
            when(ge(left_mask.e(), int(0)), {
                let mut fast = vec![right_mask.decl(call("zb_set_mask63", vec![other.e()], i64()))];
                let mut unmatched = for_range(
                    &i,
                    int(1),
                    n.e(),
                    vec![
                        member.decl(at(s.e(), i.e())),
                        bit.decl(call("zb_box_payload_i64", vec![member.e()], i64())),
                        when(
                            eq(bitand(right_mask.e(), shl(int(1), bit.e())), int(0)),
                            vec![push(out.e(), member.e())],
                        ),
                    ],
                );
                unmatched.extend(settled(&out));
                fast.push(when(ge(right_mask.e(), int(0)), unmatched));
                fast
            }),
            when(
                call("zb_set_disjoint_kinds", vec![s.e(), other.e()], boolean()),
                vec![ret(call("zb_set_copy", vec![s.e()], anys.clone()))],
            ),
            hs.decl(call("zb_set_entry_hashes", vec![s.e()], ints.clone())),
            out_hs.decl(list(Vec::new(), ints.clone())),
            expr(mcall(out.e(), "reserve", vec![n.e()], unit())),
            expr(mcall(out_hs.e(), "reserve", vec![n.e()], unit())),
        ];
        st.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![
                h.decl(idx(hs.e(), sub(i.e(), int(1)), i64())),
                when(
                    not(call(
                        "zb_set_has_hashed",
                        vec![other.e(), at(s.e(), i.e()), h.e()],
                        boolean(),
                    )),
                    vec![push(out.e(), at(s.e(), i.e())), push(out_hs.e(), h.e())],
                ),
            ],
        ));
        st.push(expr(call(
            "zb_set_settle_hashed",
            vec![out.e(), out_hs.e()],
            unit(),
        )));
        st.push(ret(out.e()));
        st
    }));
    d.push(define(
        "zb_set_xor",
        &[&s, &other],
        anys.clone(),
        vec![ret(call(
            "zb_set_or",
            vec![
                call("zb_set_sub", vec![s.e(), other.e()], anys.clone()),
                call("zb_set_sub", vec![other.e(), s.e()], anys.clone()),
            ],
            anys.clone(),
        ))],
    ));
    d.push(define("zb_set_issubset", &[&s, &other], boolean(), {
        let mut st = vec![
            when(gt(count(s.e()), count(other.e())), vec![ret(bool(false))]),
            left_mask.decl(call("zb_set_mask63", vec![s.e()], i64())),
            when(
                ge(left_mask.e(), int(0)),
                vec![
                    right_mask.decl(call("zb_set_mask63", vec![other.e()], i64())),
                    when(
                        ge(right_mask.e(), int(0)),
                        vec![ret(eq(
                            bitand(left_mask.e(), right_mask.e()),
                            left_mask.e(),
                        ))],
                    ),
                ],
            ),
            when(
                call("zb_set_disjoint_kinds", vec![s.e(), other.e()], boolean()),
                vec![ret(bool(false))],
            ),
            n.decl(len(s.e())),
        ];
        st.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![when(
                not(contains(other.e(), at(s.e(), i.e()))),
                vec![ret(bool(false))],
            )],
        ));
        st.push(ret(bool(true)));
        st
    }));
    d.push(define(
        "zb_set_eq",
        &[&s, &other],
        boolean(),
        vec![ret(and(
            eq(count(s.e()), count(other.e())),
            call("zb_set_issubset", vec![s.e(), other.e()], boolean()),
        ))],
    ));
    d.push(define(
        "zb_set_repr",
        &[&s],
        string(),
        vec![
            when(eq(count(s.e()), int(0)), vec![ret(text("set()"))]),
            text_out.decl(call(
                "zb_list_items_any",
                vec![
                    call("zb_set_items", vec![s.e()], anys.clone()),
                    text("{"),
                    text("}"),
                ],
                string(),
            )),
            ret(text_out.e()),
        ],
    ));
    d.push(extern_fn(
        "zb_set_raw",
        &[("x", any())],
        anys.clone(),
        Some("zyntax_box_pointer"),
    ));
    d.push(extern_fn(
        "zb_box_set_raw",
        &[("s", anys.clone()), ("tag", i32())],
        any(),
        Some("zyntax_box_ptr"),
    ));
    let boxed = local("s", anys.clone());
    d.push(define(
        "zb_set_box",
        &[&boxed],
        any(),
        vec![ret(call(
            "zb_box_set_raw",
            vec![boxed.e(), int32(SET_TAG as i32)],
            any(),
        ))],
    ));
    d.push(define(
        "zb_set_unbox",
        &[&x],
        anys.clone(),
        vec![
            tag.decl(cast(call("zb_box_tag", vec![x.e()], i32()), i64())),
            when(
                ne(tag.e(), int(SET_TAG)),
                vec![fatal(
                    "TypeError",
                    add(
                        text("expected a set, got "),
                        call("zb_any_type", vec![x.e()], string()),
                    ),
                )],
            ),
            ret(call("zb_unbox_list_raw_any", vec![x.e()], anys.clone())),
        ],
    ));
    // The box itself, once it is known to hold a set: what a slot that
    // stores sets as boxes takes from a dynamic value.
    d.push(define(
        "zb_set_as_box",
        &[&x],
        any(),
        vec![
            expr(call("zb_set_unbox", vec![x.e()], anys.clone())),
            ret(x.e()),
        ],
    ));
    d
}
