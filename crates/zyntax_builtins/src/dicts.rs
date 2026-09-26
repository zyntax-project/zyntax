//! Dictionaries and sets: open-addressing hash tables over one list of
//! entries, generated per key and value kind.
//!
//! A dict of `K` to `V` is a `List<(i64, K, V)>` and a set of `K` a
//! `List<(i64, K)>`: each entry holds the key's mixed hash, the key and
//! the value inline, in insertion order, so iteration and printing show
//! the order things were added in.
//!
//! A container of at most [`SMALL`] entries has no index. Its entries
//! are all keys, it is scanned comparing keys, and its hash words are
//! all zero (a literal lays its entries out that way). A deletion from
//! it removes the entry.
//!
//! An indexed container's first entry is its control entry: its hash
//! word holds the address of the index, never zero, and its other words
//! belong to the shape (a set of ints keeps its small-value mask there).
//! The keys follow from entry 1. The index is a `List<i64>`: word 0
//! counts the dead entries, word 1 is a shape's own, and the slots
//! follow, a power of two of entry numbers, -1 where empty, probed
//! linearly from the hash and never more than half full. A deletion
//! leaves a tombstone: hash [`TOMB`], which no mixed hash is, key and
//! value zeroed, and the slot still naming it so a probe passes over it.
//! The entries are compacted in order and the index rebuilt when the
//! table grows or the dead outnumber half the live; one left with a few
//! entries drops its index and control entry.
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
/// A frozen set's kind is its shape's set kind this far up.
const FROZEN_SETS: i64 = 1 << 15;

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

/// The box tag of the frozen sets of the set shape a frontend registered
/// as `index`: the kinds past [`FROZEN_SETS`] above its set shapes'.
pub fn frozen_set_shape_tag(index: u16) -> i64 {
    assert!(
        i64::from(index) < FROZEN_SETS,
        "a set shape index below the frozen ones"
    );
    set_shape_tag(index) + (FROZEN_SETS << 8)
}

/// The box kinds of a frontend's frozen sets: from the first, up to but
/// not including the second.
pub(crate) fn frozen_kind_range() -> (i64, i64) {
    let from = crate::lists::SHAPE_KIND_BASE + SET_SHAPES + FROZEN_SETS;
    (from, from + FROZEN_SETS)
}

/// The box kinds of a frontend's dict and set shapes: from the first,
/// up to but not including the second, dicts below sets.
pub(crate) fn keyed_kind_range() -> (i64, i64) {
    (
        crate::lists::SHAPE_KIND_BASE + DICT_SHAPES,
        crate::lists::SHAPE_KIND_BASE + SET_SHAPES + DICT_SHAPES,
    )
}

/// The box kind a frontend's set shapes start from.
pub(crate) fn set_kinds_from() -> i64 {
    crate::lists::SHAPE_KIND_BASE + SET_SHAPES
}

fn any_eq(a: Expr, b: Expr) -> Expr {
    call("zb_any_eq", vec![a, b], boolean())
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
    d.extend(set_declarations(list_type, "", SET_TAG, &Field::Any));
    d.extend(dynamic_set(list_type));
    // A copy of any dict or set as the library's, for a frontend whose
    // typed tables meet dynamic ones.
    for (value, tag) in [(Some(&Field::Any), DICT_TAG), (None, SET_TAG)] {
        let t = Table {
            list_type,
            prefix: if value.is_some() { "zb_dict" } else { "zb_set" },
            sfx: String::new(),
            key: &Field::Any,
            value,
            tag,
            mask: Mask::None,
        };
        let from = t.name("from_dyn");
        d.extend(
            probe_any_ops(&t)
                .into_iter()
                .filter(|decl| match &decl.node {
                    zyntax_typed_ast::typed_ast::TypedDeclaration::Function(f) => {
                        f.name.resolve_global().as_deref() == Some(from.as_str())
                    }
                    _ => false,
                }),
        );
    }
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
    d.push(extern_fn(
        "zb_set_raw",
        &[("x", any())],
        set_type(list_type),
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
    /// In the index's shape word, computed when asked for and forgotten
    /// when the set changes; a set without an index is scanned.
    Lazy,
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
    /// The index's address, or 0 for a container without one: an
    /// indexed container's first entry is its control entry, whose hash
    /// word is that address; a small one's hash words are all zero.
    ///
    /// Read without looking at the length: a list's storage is never
    /// null and holds zeros past its length (a new list's slots are
    /// cleared, and so are those a truncation gives up), and entry 0 of
    /// a container without an index, present or left over from a
    /// removal, has a zero hash word as every such entry does.
    fn ctrl(&self, d: Expr) -> Expr {
        self.hash_of(self.entry(d, int(0)))
    }
    /// The first entry that holds a key, given `ctrl`: 1 past the
    /// control entry of an indexed container, else 0.
    fn first(&self, c: Expr) -> Expr {
        if_expr(ne(c, int(0)), int(1), int(0))
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
            // Only an indexed set has a control entry to keep it in; a
            // small one is scanned.
            Mask::Eager => {
                let m = self.key_of(self.entry(d.clone(), int(0)));
                let fits = and(ge(k.clone(), int(0)), lt(k.clone(), int(63)));
                let grown = if_expr(fits, bitor(m.clone(), shl(int(1), k)), int(-1));
                vec![when(
                    ne(self.ctrl(d.clone()), int(0)),
                    vec![when(
                        ge(m.clone(), int(0)),
                        vec![set_idx(
                            d.clone(),
                            int(0),
                            self.make(self.ctrl(d), grown, None),
                        )],
                    )],
                )]
            }
            Mask::Lazy => vec![when(
                ne(self.ctrl(d.clone()), int(0)),
                vec![set_idx(
                    self.index_at(self.ctrl(d)),
                    int(1),
                    int(MASK_UNKNOWN),
                )],
            )],
        }
    }
    /// The statements updating the mask before `k` is removed.
    fn note_removed(&self, d: Expr, k: Expr) -> Vec<Stmt> {
        match self.mask {
            Mask::Eager => {
                let m = self.key_of(self.entry(d.clone(), int(0)));
                let fits = and(ge(k.clone(), int(0)), lt(k.clone(), int(63)));
                vec![when(
                    ne(self.ctrl(d.clone()), int(0)),
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
                    )],
                )]
            }
            // Forgotten as an addition forgets it.
            Mask::Lazy => self.note_added(d, k),
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

    out_decls.push(define(
        &t.name("new"),
        &[],
        lt_.clone(),
        vec![out.decl(list(Vec::new(), lt_.clone())), ret(out.e())],
    ));
    out_decls.push(define(&t.name("len"), &[&d], i64(), {
        vec![
            c.decl(t.ctrl(d.e())),
            when(eq(c.e(), int(0)), vec![ret(len(d.e()))]),
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
                        i.decl(int(0)),
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
    // when some are dead; a container without an index takes its
    // control entry first. The hash words are read when `hashed`, else
    // computed from the keys: a container that had no index keeps them
    // zero.
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
            if_(
                ne(c.e(), int(0)),
                vec![when(
                    gt(idx(old.e(), int(0), i64()), int(0)),
                    vec![block_of(compact)],
                )],
                vec![expr(mcall(
                    d.e(),
                    "insert_at",
                    vec![int(0), t.hollow(int(0))],
                    unit(),
                ))],
            ),
            when(not(hashed.e()), vec![block_of(rehash)]),
            t.go("index_entries", vec![d.e(), cap.e()]),
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
    // An index of `cap` slots over entries 1 on, whose hash words hold
    // their mixed hashes, named by the control entry at 0 in place of
    // whatever index it named.
    out_decls.push(define(&t.name("index_entries"), &[&d, &cap], unit(), {
        let mut st = vec![
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
        if t.mask == Mask::Eager {
            st.push(t.go("remask", vec![d.e()]));
        }
        st.push(ret_void());
        st
    }));
    // Back to no index: the live entries moved down over the control
    // entry, in order, their hash words zeroed. What a container that
    // has shrunk to a few entries is.
    out_decls.push(define(&t.name("unindex"), &[&d], unit(), {
        let mut st = vec![
            c.decl(t.ctrl(d.e())),
            when(eq(c.e(), int(0)), vec![ret_void()]),
            old.decl(t.index_at(c.e())),
            n.decl(len(d.e())),
            w.decl(int(0)),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    en.decl(t.entry(d.e(), i.e())),
                    when(
                        ne(t.hash_of(en.e()), int(TOMB)),
                        vec![
                            set_idx(
                                d.e(),
                                w.e(),
                                t.make(
                                    int(0),
                                    t.key_of(en.e()),
                                    vals(vf.map(|_| t.value_of(en.e()))),
                                ),
                            ),
                            w.add_assign(int(1)),
                        ],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            expr(mcall(d.e(), "truncate", vec![w.e()], unit())),
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
        small.push(n.decl(len(d.e())));
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
            expr(mcall(d.e(), "truncate", vec![int(0)], unit())),
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
    // own built from their hashes when it is large enough for one.
    out_decls.push(define(&t.name("copy"), &[&d], lt_.clone(), {
        vec![
            c.decl(t.ctrl(d.e())),
            live.decl(t.call("len", vec![d.e()], i64())),
            hashed.decl(and(ne(c.e(), int(0)), gt(live.e(), int(SMALL)))),
            n.decl(len(d.e())),
            out.decl(list(Vec::new(), lt_.clone())),
            expr(mcall(
                out.e(),
                "reserve",
                vec![add(live.e(), int(1))],
                unit(),
            )),
            when(hashed.e(), vec![push(out.e(), t.hollow(int(0)))]),
            i.decl(t.first(c.e())),
            while_(
                lt(i.e(), n.e()),
                vec![
                    en.decl(t.entry(d.e(), i.e())),
                    when(
                        ne(t.hash_of(en.e()), int(TOMB)),
                        vec![push(
                            out.e(),
                            t.make(
                                if_expr(hashed.e(), t.hash_of(en.e()), int(0)),
                                t.key_of(en.e()),
                                vals(vf.map(|_| t.value_of(en.e()))),
                            ),
                        )],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            when(
                hashed.e(),
                vec![t.go("index_entries", vec![out.e(), cap_for(live.e())])],
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
        vec![ret(sub(len(d.e()), t.first(t.ctrl(d.e()))))],
    ));
    out_decls.push(define(
        &t.name("live_at"),
        &[&d, &p],
        boolean(),
        vec![ret(ne(
            t.hash_of(t.entry(d.e(), add(p.e(), t.first(t.ctrl(d.e()))))),
            int(TOMB),
        ))],
    ));
    out_decls.push(define(
        &t.name("key_at"),
        &[&d, &p],
        t.key.ty(),
        vec![ret(
            t.key_of(t.entry(d.e(), add(p.e(), t.first(t.ctrl(d.e())))))
        )],
    ));
    if let Some(vf) = vf {
        out_decls.push(define(
            &t.name("value_at"),
            &[&d, &p],
            vf.ty(),
            vec![ret(t.value_of(
                t.entry(d.e(), add(p.e(), t.first(t.ctrl(d.e())))),
            ))],
        ));
        out_decls.push(define(&t.name("value_set_at"), &[&d, &p, &v], unit(), {
            vec![
                en.decl(t.entry(d.e(), add(p.e(), t.first(t.ctrl(d.e()))))),
                set_idx(
                    d.e(),
                    add(p.e(), t.first(t.ctrl(d.e()))),
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
    // The same for a box of another tag the frontend gave this shape's
    // storage (a frozen set's), and the box itself once checked.
    let want = local("tag", i64());
    let checked = |x: &Local| {
        when(
            ne(
                cast(call("zb_box_tag", vec![x.e()], i32()), i64()),
                want.e(),
            ),
            vec![fatal(
                "TypeError",
                add(
                    text(&format!("expected {what}, got ")),
                    call("zb_any_type", vec![x.e()], string()),
                ),
            )],
        )
    };
    out_decls.push(define(
        &t.name("unbox_tagged"),
        &[&x, &want],
        lt_.clone(),
        vec![
            checked(&x),
            ret(call(&t.name("raw"), vec![x.e()], lt_.clone())),
        ],
    ));
    out_decls.push(define(
        &t.name("as_box_tagged"),
        &[&x, &want],
        any(),
        vec![checked(&x), ret(x.e())],
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
        out.extend(probe_any_ops(&t));
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
    // The value as a dynamic value, or `default` when `k` is absent:
    // a read whose default the value field cannot hold.
    let default_box = kept("default", any());
    d_out.push(define(
        &t.name("get_boxed"),
        &[&d, &key, &default_box],
        any(),
        vec![
            e.decl(find(d.e(), key.e())),
            when(lt(e.e(), int(0)), vec![ret(default_box.e())]),
            ret(vf.boxed(value_at(d.e(), e.e()))),
        ],
    ));
    let boxed_out = local("x", any());
    d_out.push(define(
        &t.name("pop_boxed"),
        &[&d, &key, &default_box],
        any(),
        vec![
            e.decl(find(d.e(), key.e())),
            when(lt(e.e(), int(0)), vec![ret(default_box.e())]),
            boxed_out.decl(vf.boxed(value_at(d.e(), e.e()))),
            t.go("delete_at", vec![d.e(), e.e()]),
            ret(boxed_out.e()),
        ],
    ));
    // A dict whose entries a literal laid out itself, its keys known to
    // be distinct and its hash words zero: the storage is the dict, and
    // only a large one takes an index.
    if !t.sfx.is_empty() {
        let entries = kept("d", lt_.clone());
        d_out.push(define(
            &t.name("from_distinct"),
            &[&entries],
            lt_.clone(),
            {
                vec![
                    n.decl(len(entries.e())),
                    when(
                        gt(n.e(), int(SMALL)),
                        vec![t.go(
                            "rebuild",
                            vec![
                                entries.e(),
                                call("zb_table_cap_for", vec![n.e()], i64()),
                                bool(false),
                            ],
                        )],
                    ),
                    ret(entries.e()),
                ]
            },
        ));
    }
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
            when(
                ge(e.e(), int(0)),
                vec![ret(sub(e.e(), t.first(t.ctrl(d.e()))))],
            ),
            t.go("insert", vec![d.e(), k.e(), vf.zero()]),
            ret(sub(sub(len(d.e()), int(1)), t.first(t.ctrl(d.e())))),
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
                    ge(e.e(), t.first(t.ctrl(d.e()))),
                    eq(t.hash_of(t.entry(d.e(), e.e())), int(TOMB)),
                ),
                vec![e.set(sub(e.e(), int(1)))],
            ),
            when(
                lt(e.e(), t.first(t.ctrl(d.e()))),
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
            i.decl(t.first(t.ctrl(d.e()))),
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
    // A dict whose entries a literal laid out itself, its keys known to
    // be distinct and its hash words zero: the storage is the dict, and
    // only a large one takes an index.
    let entries = kept("d", dt.clone());
    d.push(define("zb_dict_from_distinct", &[&entries], dt.clone(), {
        vec![
            live.decl(len(entries.e())),
            when(
                gt(live.e(), int(SMALL)),
                vec![expr(call(
                    "zb_dict_rebuild",
                    vec![
                        entries.e(),
                        call("zb_table_cap_for", vec![live.e()], i64()),
                        bool(false),
                    ],
                    unit(),
                ))],
            ),
            ret(entries.e()),
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
                    i.decl(int(0)),
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

/// The type of a set entry of key `key`.
pub fn set_entry_type(key: &Field) -> Type {
    Type::Tuple(vec![i64(), key.ty()])
}

/// The type of the library's set: dynamic values.
pub fn set_type(list_type: TypeId) -> Type {
    list_of(list_type, set_entry_type(&Field::Any))
}

/// The functions of the set shape of `key` whose tag is `tag`:
/// `zb_set_<op>` for the library's own shape (`suffix` empty),
/// `zb_set_<op>_<suffix>` for one a frontend registers. A set of ints
/// keeps the mask of its small values up to date in its control entry.
pub fn set_declarations(list_type: TypeId, suffix: &str, tag: i64, key: &Field) -> Vec<Decl> {
    let t = Table {
        list_type,
        prefix: "zb_set",
        sfx: if suffix.is_empty() {
            String::new()
        } else {
            format!("_{suffix}")
        },
        key,
        value: None,
        tag,
        mask: match key {
            Field::Int => Mask::Eager,
            Field::Any if suffix.is_empty() => Mask::Lazy,
            _ => Mask::None,
        },
    };
    let mut out = table_core(&t);
    out.extend(set_ops(&t));
    if suffix.is_empty() {
        out
    } else {
        out.extend(probe_any_ops(&t));
        crate::lists::generated(out)
    }
}

fn set_ops(t: &Table) -> Vec<Decl> {
    let lt_ = t.list_ty();
    let anys = t.anys();
    let s = borrowed("s", lt_.clone());
    let other = borrowed("other", lt_.clone());
    let key = borrowed("k", t.key.ty());
    let k = kept("k", t.key.ty());
    let out = local("out", lt_.clone());
    let e = local("e", i64());
    let h = local("h", i64());
    let i = local("i", i64());
    let n = local("n", i64());
    let c = local("count", i64());
    let en = local("en", t.entry_ty());
    let x = local("x", t.key.ty());
    let m = local("m", i64());
    let mo = local("mo", i64());
    let hashed = local("hashed", boolean());
    let found = local("found", boolean());
    let extra = local("extra", i64());
    let total = local("total", i64());
    let cap = local("cap", i64());
    let find = |s: Expr, k: Expr| t.call("find", vec![s, k], i64());
    let masked = t.mask != Mask::None;
    let mask_of = |s: Expr| t.call("mask63", vec![s], i64());
    let mut d = Vec::new();

    // The live entries of `s` in order, `en` each.
    let live_walk = |s: &Local, body: Vec<Stmt>| {
        vec![
            n.decl(len(s.e())),
            i.decl(t.first(t.ctrl(s.e()))),
            while_(
                lt(i.e(), n.e()),
                vec![
                    en.decl(t.entry(s.e(), i.e())),
                    when(ne(t.hash_of(en.e()), int(TOMB)), body),
                    i.add_assign(int(1)),
                ],
            ),
        ]
    };
    // Whether the key of `en`, an entry of a set that is indexed when
    // `hashed` (so `en` holds its mixed hash), is in `other`. `find`
    // names the lookups to use.
    let member_of = |other: &Local, find: &str, find_hashed: &str| {
        if_expr(
            eq(t.ctrl(other.e()), int(0)),
            ge(
                t.call(find, vec![other.e(), t.key_of(en.e())], i64()),
                int(0),
            ),
            ge(
                t.call(
                    find_hashed,
                    vec![
                        other.e(),
                        t.key_of(en.e()),
                        if_expr(hashed.e(), t.hash_of(en.e()), t.mixed(t.key_of(en.e()))),
                    ],
                    i64(),
                ),
                int(0),
            ),
        )
    };
    // A result built by appending distinct entries takes its index, and
    // a set of ints its mask, once it is complete. The entries' hash
    // words are the mixed hashes when `hashed`.
    // A result built by appending distinct entries, each with the hash
    // word it had in a set that is indexed when `hashed`, takes its
    // index once complete: from those words, its control entry put in
    // front, or from its keys. One that stays small has the words
    // zeroed.
    let settle = |out: &Local, hashed: Expr| {
        let zero = for_range(
            &i,
            int(0),
            n.e(),
            vec![
                en.decl(t.entry(out.e(), i.e())),
                set_idx(out.e(), i.e(), t.make(int(0), t.key_of(en.e()), None)),
            ],
        );
        vec![
            n.decl(len(out.e())),
            if_(
                gt(n.e(), int(SMALL)),
                vec![if_(
                    hashed.clone(),
                    vec![
                        expr(mcall(
                            out.e(),
                            "insert_at",
                            vec![int(0), t.hollow(int(0))],
                            unit(),
                        )),
                        t.go(
                            "index_entries",
                            vec![out.e(), call("zb_table_cap_for", vec![n.e()], i64())],
                        ),
                    ],
                    vec![t.go(
                        "rebuild",
                        vec![
                            out.e(),
                            call("zb_table_cap_for", vec![n.e()], i64()),
                            bool(false),
                        ],
                    )],
                )],
                vec![when(hashed, zero)],
            ),
        ]
    };

    if masked {
        // The mask of the small ints `s` holds: bit `v` for each value
        // `0 <= v < 63`, or -1 when it holds anything else.
        let scan = {
            let fits = |v: Expr| and(ge(v.clone(), int(0)), lt(v, int(63)));
            let mut st = vec![m.decl(int(0))];
            let bit = match t.key {
                Field::Int => {
                    let v = t.key_of(en.e());
                    vec![if_(
                        fits(v.clone()),
                        vec![m.set(bitor(m.e(), shl(int(1), v)))],
                        vec![ret(int(-1))],
                    )]
                }
                _ => {
                    let stored = t.key_of(en.e());
                    let v = call("zb_box_payload_i64", vec![stored.clone()], i64());
                    vec![
                        when(
                            ne(
                                cast(call("zb_box_tag", vec![stored], i32()), i64()),
                                int(crate::dynamic::I64_TAG),
                            ),
                            vec![ret(int(-1))],
                        ),
                        if_(
                            fits(v.clone()),
                            vec![m.set(bitor(m.e(), shl(int(1), v)))],
                            vec![ret(int(-1))],
                        ),
                    ]
                }
            };
            st.extend(live_walk(&s, bit));
            st.push(ret(m.e()));
            st
        };
        d.push(define(&t.name("mask_scan"), &[&s], i64(), scan));
        match t.mask {
            Mask::Eager => {
                d.push(define(
                    &t.name("mask63"),
                    &[&s],
                    i64(),
                    vec![
                        when(
                            eq(t.ctrl(s.e()), int(0)),
                            vec![ret(t.call("mask_scan", vec![s.e()], i64()))],
                        ),
                        ret(t.key_of(t.entry(s.e(), int(0)))),
                    ],
                ));
                d.push(define(&t.name("remask"), &[&s], unit(), {
                    vec![
                        set_idx(
                            s.e(),
                            int(0),
                            t.make(t.ctrl(s.e()), t.call("mask_scan", vec![s.e()], i64()), None),
                        ),
                        ret_void(),
                    ]
                }));
            }
            _ => {
                // Kept in the index's shape word until the set changes;
                // a set without an index is scanned.
                let index = local("index", t.ints());
                d.push(define(&t.name("mask63"), &[&s], i64(), {
                    vec![
                        when(
                            eq(t.ctrl(s.e()), int(0)),
                            vec![ret(t.call("mask_scan", vec![s.e()], i64()))],
                        ),
                        index.decl(t.index_at(t.ctrl(s.e()))),
                        m.decl(idx(index.e(), int(1), i64())),
                        when(ne(m.e(), int(MASK_UNKNOWN)), vec![ret(m.e())]),
                        m.set(t.call("mask_scan", vec![s.e()], i64())),
                        set_idx(index.e(), int(1), m.e()),
                        ret(m.e()),
                    ]
                }));
            }
        }
    }

    // A value that is not there: an error path, kept out of line.
    d.push(define_cold(
        &t.name("missing"),
        &[&key],
        unit(),
        vec![fatal("KeyError", t.key_text(key.e())), ret_void()],
    ));
    d.push(define(&t.name("contains"), &[&s, &key], boolean(), {
        let mut st = Vec::new();
        if t.mask == Mask::Eager {
            // Every value of an indexed set is a small int when its mask
            // stands.
            st.push(when(
                ne(t.ctrl(s.e()), int(0)),
                vec![
                    m.decl(t.key_of(t.entry(s.e(), int(0)))),
                    when(
                        ge(m.e(), int(0)),
                        vec![ret(and(
                            and(ge(key.e(), int(0)), lt(key.e(), int(63))),
                            ne(bitand(shr(m.e(), key.e()), int(1)), int(0)),
                        ))],
                    ),
                ],
            ));
        }
        st.push(ret(ge(find(s.e(), key.e()), int(0))));
        st
    }));
    d.push(define(&t.name("add"), &[&s, &k], unit(), {
        vec![
            when(
                eq(t.ctrl(s.e()), int(0)),
                vec![
                    when(
                        lt(find(s.e(), k.e()), int(0)),
                        vec![t.go("insert", vec![s.e(), k.e()])],
                    ),
                    ret_void(),
                ],
            ),
            h.decl(t.mixed(k.e())),
            when(
                lt(
                    t.call("find_hashed", vec![s.e(), k.e(), h.e()], i64()),
                    int(0),
                ),
                vec![t.go("insert_hashed", vec![s.e(), k.e(), h.e()])],
            ),
            ret_void(),
        ]
    }));
    d.push(define(&t.name("discard"), &[&s, &key], unit(), {
        vec![
            e.decl(find(s.e(), key.e())),
            when(
                ge(e.e(), int(0)),
                vec![t.go("delete_at", vec![s.e(), e.e()])],
            ),
            ret_void(),
        ]
    }));
    d.push(define(&t.name("remove"), &[&s, &key], unit(), {
        vec![
            e.decl(find(s.e(), key.e())),
            when(
                lt(e.e(), int(0)),
                vec![t.go("missing", vec![key.e()]), ret_void()],
            ),
            t.go("delete_at", vec![s.e(), e.e()]),
            ret_void(),
        ]
    }));
    // The last value added, removed.
    d.push(define(&t.name("pop"), &[&s], t.key.ty(), {
        vec![
            e.decl(sub(len(s.e()), int(1))),
            while_(
                and(
                    ge(e.e(), t.first(t.ctrl(s.e()))),
                    eq(t.hash_of(t.entry(s.e(), e.e())), int(TOMB)),
                ),
                vec![e.set(sub(e.e(), int(1)))],
            ),
            when(
                lt(e.e(), t.first(t.ctrl(s.e()))),
                vec![fatal("KeyError", text("pop from an empty set"))],
            ),
            x.decl(t.key_of(t.entry(s.e(), e.e()))),
            t.go("delete_at", vec![s.e(), e.e()]),
            ret(x.e()),
        ]
    }));
    // The values, in order, as a list of their own.
    let (elem, conv) = list_elem(t.key);
    let elems = list_of(t.list_type, elem.clone());
    let xs = local("xs", elems.clone());
    d.push(define(&t.name("items"), &[&s], elems.clone(), {
        let mut st = vec![
            xs.decl(list(Vec::new(), elems.clone())),
            expr(mcall(xs.e(), "reserve", vec![len(s.e())], unit())),
        ];
        st.extend(live_walk(&s, vec![push(xs.e(), conv(t.key_of(en.e())))]));
        st.push(ret(xs.e()));
        st
    }));
    // A loop over the set by position: the count, the dead dropped
    // first so the positions are dense, then the value at each.
    d.push(define(&t.name("iter_len"), &[&s], i64(), {
        vec![
            when(
                ne(t.ctrl(s.e()), int(0)),
                vec![when(
                    gt(idx(t.index_at(t.ctrl(s.e())), int(0), i64()), int(0)),
                    vec![t.go(
                        "rebuild",
                        vec![
                            s.e(),
                            sub(len(t.index_at(t.ctrl(s.e()))), int(INDEX_HEAD)),
                            bool(true),
                        ],
                    )],
                )],
            ),
            ret(sub(len(s.e()), t.first(t.ctrl(s.e())))),
        ]
    }));
    d.push(define(&t.name("iter_at"), &[&s, &i], elem.clone(), {
        vec![ret(conv(
            t.key_of(t.entry(s.e(), add(i.e(), t.first(t.ctrl(s.e()))))),
        ))]
    }));

    // Room for `extra` more values, the index sized once rather than
    // grown as they come.
    d.push(define(&t.name("reserve"), &[&s, &extra], unit(), {
        vec![
            total.decl(add(t.call("len", vec![s.e()], i64()), extra.e())),
            when(
                eq(t.ctrl(s.e()), int(0)),
                vec![
                    when(
                        gt(total.e(), int(SMALL)),
                        vec![t.go(
                            "rebuild",
                            vec![
                                s.e(),
                                call("zb_table_cap_for", vec![total.e()], i64()),
                                bool(false),
                            ],
                        )],
                    ),
                    ret_void(),
                ],
            ),
            cap.decl(sub(len(t.index_at(t.ctrl(s.e()))), int(INDEX_HEAD))),
            when(
                gt(mul(total.e(), int(2)), cap.e()),
                vec![t.go(
                    "rebuild",
                    vec![
                        s.e(),
                        call("zb_table_cap_for", vec![total.e()], i64()),
                        bool(true),
                    ],
                )],
            ),
            ret_void(),
        ]
    }));
    // Every value of `other` added to `s`, by the hashes `other` holds
    // when both have an index.
    d.push(define(&t.name("update"), &[&s, &other], unit(), {
        let mut st = vec![
            t.go(
                "reserve",
                vec![s.e(), t.call("len", vec![other.e()], i64())],
            ),
            hashed.decl(ne(t.ctrl(other.e()), int(0))),
        ];
        st.extend(live_walk(
            &other,
            vec![if_(
                and(hashed.e(), ne(t.ctrl(s.e()), int(0))),
                vec![when(
                    lt(
                        t.call(
                            "find_hashed",
                            vec![s.e(), t.key_of(en.e()), t.hash_of(en.e())],
                            i64(),
                        ),
                        int(0),
                    ),
                    vec![t.go(
                        "insert_hashed",
                        vec![s.e(), t.key_of(en.e()), t.hash_of(en.e())],
                    )],
                )],
                vec![t.go("add", vec![s.e(), t.key_of(en.e())])],
            )],
        ));
        st.push(ret_void());
        st
    }));
    d.push(define(&t.name("or"), &[&s, &other], lt_.clone(), {
        vec![
            out.decl(t.call("copy", vec![s.e()], lt_.clone())),
            t.go("update", vec![out.e(), other.e()]),
            ret(out.e()),
        ]
    }));
    // The values of `s` that are in `other` (`keep` true) or not, in
    // `s`'s order. Small ints are matched by the masks when both stand;
    // two sets whose values are of kinds nothing in the other equals
    // share none.
    for (op, keep) in [("and", true), ("sub", false)] {
        d.push(define(&t.name(op), &[&s, &other], lt_.clone(), {
            let wanted = |hit: Expr| if keep { hit } else { not(hit) };
            let mut st = Vec::new();
            // Both masks, or -1 when either does not stand.
            if masked {
                st.push(m.decl(mask_of(s.e())));
                st.push(mo.decl(if_expr(ge(m.e(), int(0)), mask_of(other.e()), int(-1))));
            } else {
                st.push(mo.decl(int(-1)));
            }
            if matches!(t.key, Field::Any) {
                st.push(when(
                    lt(mo.e(), int(0)),
                    vec![when(
                        call("zb_set_disjoint_kinds", vec![s.e(), other.e()], boolean()),
                        vec![ret(if keep {
                            t.call("new", vec![], lt_.clone())
                        } else {
                            t.call("copy", vec![s.e()], lt_.clone())
                        })],
                    )],
                ));
            }
            let bit = match t.key {
                Field::Int => t.key_of(en.e()),
                _ => call("zb_box_payload_i64", vec![t.key_of(en.e())], i64()),
            };
            // By the mask when both stand, else by a lookup; a result the
            // masks give grows as it is filled.
            let hit = if masked {
                if_expr(
                    ge(mo.e(), int(0)),
                    ne(bitand(shr(mo.e(), bit), int(1)), int(0)),
                    member_of(&other, "find", "find_hashed"),
                )
            } else {
                member_of(&other, "find", "find_hashed")
            };
            st.extend([
                out.decl(t.call("new", vec![], lt_.clone())),
                hashed.decl(ne(t.ctrl(s.e()), int(0))),
                when(
                    lt(mo.e(), int(0)),
                    vec![expr(mcall(out.e(), "reserve", vec![len(s.e())], unit()))],
                ),
            ]);
            st.extend(live_walk(
                &s,
                vec![when(
                    wanted(hit),
                    // The hash word as it stands: the mixed hash of an
                    // indexed set's entry, zero in a small one's.
                    vec![push(
                        out.e(),
                        t.make(t.hash_of(en.e()), t.key_of(en.e()), None),
                    )],
                )],
            ));
            st.extend(settle(&out, hashed.e()));
            st.push(ret(out.e()));
            st
        }));
    }
    d.push(define(&t.name("xor"), &[&s, &other], lt_.clone(), {
        vec![
            out.decl(t.call("sub", vec![s.e(), other.e()], lt_.clone())),
            t.go(
                "update",
                vec![out.e(), t.call("sub", vec![other.e(), s.e()], lt_.clone())],
            ),
            ret(out.e()),
        ]
    }));
    // How many values of `s` are in `other`: the size of their
    // intersection, from which every other combination's size follows.
    d.push(define(&t.name("and_len"), &[&s, &other], i64(), {
        let mut st = vec![c.decl(int(0)), hashed.decl(ne(t.ctrl(s.e()), int(0)))];
        st.extend(live_walk(
            &s,
            vec![when(
                member_of(&other, "find", "find_hashed"),
                vec![c.add_assign(int(1))],
            )],
        ));
        st.push(ret(c.e()));
        st
    }));
    let size = |s: &Local| t.call("len", vec![s.e()], i64());
    let common = || t.call("and_len", vec![s.e(), other.e()], i64());
    for (op, value) in [
        ("sub_len", sub(size(&s), common())),
        ("or_len", sub(add(size(&s), size(&other)), common())),
        (
            "xor_len",
            sub(add(size(&s), size(&other)), mul(common(), int(2))),
        ),
    ] {
        d.push(define(&t.name(op), &[&s, &other], i64(), vec![ret(value)]));
    }
    // Whether every value of `s` is in `other`. Equality compares
    // through lookups of its own, as a dict's does.
    let variants: &[(&str, &str, &str)] = if t.recursive_keys() {
        &[
            ("issubset", "find", "find_hashed"),
            ("subset_eq", "find_eq", "find_hashed_eq"),
        ]
    } else {
        &[("issubset", "find", "find_hashed")]
    };
    for (op, find, find_hashed) in variants {
        d.push(define(&t.name(op), &[&s, &other], boolean(), {
            let mut st = vec![when(gt(size(&s), size(&other)), vec![ret(bool(false))])];
            if masked {
                st.push(m.decl(mask_of(s.e())));
                st.push(when(
                    ge(m.e(), int(0)),
                    vec![
                        mo.decl(mask_of(other.e())),
                        when(
                            ge(mo.e(), int(0)),
                            vec![ret(eq(bitand(m.e(), mo.e()), m.e()))],
                        ),
                    ],
                ));
            }
            if matches!(t.key, Field::Any) {
                st.push(when(
                    call("zb_set_disjoint_kinds", vec![s.e(), other.e()], boolean()),
                    vec![ret(bool(false))],
                ));
            }
            st.push(hashed.decl(ne(t.ctrl(s.e()), int(0))));
            st.extend(live_walk(
                &s,
                vec![when(
                    not(member_of(&other, find, find_hashed)),
                    vec![ret(bool(false))],
                )],
            ));
            st.push(ret(bool(true)));
            st
        }));
    }
    let subset_eq = if t.recursive_keys() {
        "subset_eq"
    } else {
        "issubset"
    };
    d.push(define(&t.name("eq"), &[&s, &other], boolean(), {
        vec![ret(and(
            eq(size(&s), size(&other)),
            t.call(subset_eq, vec![s.e(), other.e()], boolean()),
        ))]
    }));
    d.push(define(&t.name("isdisjoint"), &[&s, &other], boolean(), {
        vec![ret(eq(common(), int(0)))]
    }));
    // The hash of a frozen set: the same whatever order it was filled
    // in, from the hash words of an indexed set.
    d.push(define(&t.name("hash"), &[&s], i64(), {
        let mut st = vec![
            h.decl(int(0x2545_F491_4F6C_DD1D)),
            hashed.decl(ne(t.ctrl(s.e()), int(0))),
        ];
        st.extend(live_walk(
            &s,
            vec![h.set(bitxor(
                h.e(),
                if_expr(hashed.e(), t.hash_of(en.e()), t.mixed(t.key_of(en.e()))),
            ))],
        ));
        st.push(ret(add(mul(h.e(), int(1_000_003)), size(&s))));
        st
    }));
    let text_out = local("text", string());
    d.push(define(&t.name("repr"), &[&s], string(), {
        let strs = list_of(t.list_type, string());
        let pieces = local("pieces", strs.clone());
        let first = local("first", boolean());
        let piece = |p: Expr| expr(mcall(pieces.e(), "push", vec![p], unit()));
        let mut st = vec![
            when(eq(size(&s), int(0)), vec![ret(text("set()"))]),
            pieces.decl(list(Vec::new(), strs)),
            piece(text("{")),
            first.decl(bool(true)),
        ];
        st.extend(live_walk(
            &s,
            vec![
                when(not(first.e()), vec![piece(text(", "))]),
                first.set(bool(false)),
                piece(t.key.repr(t.key_of(en.e()))),
            ],
        ));
        st.push(piece(text("}")));
        st.push(text_out.decl(call("zb_str_join", vec![text(""), pieces.e()], string())));
        st.push(ret(text_out.e()));
        st
    }));
    if t.mask == Mask::Eager {
        // The least value: the lowest bit of the mask when it stands.
        d.push(define(&t.name("min"), &[&s], i64(), {
            let mut st = vec![
                when(
                    eq(size(&s), int(0)),
                    vec![fatal("ValueError", text("min() arg is an empty sequence"))],
                ),
                m.decl(mask_of(s.e())),
                when(
                    ge(m.e(), int(0)),
                    vec![
                        x.decl(int(0)),
                        while_(
                            eq(bitand(shr(m.e(), x.e()), int(1)), int(0)),
                            vec![x.add_assign(int(1))],
                        ),
                        ret(x.e()),
                    ],
                ),
                found.decl(bool(false)),
                x.decl(int(0)),
            ];
            st.extend(live_walk(
                &s,
                vec![when(
                    or(not(found.e()), lt(t.key_of(en.e()), x.e())),
                    vec![x.set(t.key_of(en.e())), found.set(bool(true))],
                )],
            ));
            st.push(ret(x.e()));
            st
        }));
    }
    // A set from a list of its kind, first occurrences kept.
    if elem == t.key.ty() {
        let xs = borrowed("xs", elems.clone());
        let from = if t.sfx.is_empty() {
            "zb_set_from".to_string()
        } else {
            t.name("from_list")
        };
        d.push(define(&from, &[&xs], lt_.clone(), {
            let mut st = vec![
                out.decl(t.call("new", vec![], lt_.clone())),
                n.decl(len(xs.e())),
                t.go("reserve", vec![out.e(), n.e()]),
            ];
            st.extend(for_range(
                &i,
                int(0),
                n.e(),
                vec![t.go("add", vec![out.e(), idx(xs.e(), i.e(), elem.clone())])],
            ));
            st.push(ret(out.e()));
            st
        }));
    }
    let _ = anys;
    d
}

/// The library set's own functions beyond a shape's: the kind test
/// that lets two sets of disjoint kinds skip their values, and the box
/// read as itself.
fn dynamic_set(list_type: TypeId) -> Vec<Decl> {
    let st_ = set_type(list_type);
    let t = Table {
        list_type,
        prefix: "zb_set",
        sfx: String::new(),
        key: &Field::Any,
        value: None,
        tag: SET_TAG,
        mask: Mask::Lazy,
    };
    let s = borrowed("s", st_.clone());
    let other = borrowed("other", st_.clone());
    let wanted = local("wanted", i64());
    let n = local("n", i64());
    let i = local("i", i64());
    let x = local("x", any());
    let kind = |x: Expr| call("zb_any_kind", vec![x], i64());
    let first = |s: &Local| t.key_of(t.entry(s.e(), t.first(t.ctrl(s.e()))));
    let mut d = Vec::new();
    // Whether every value of `s` is of box kind `wanted`; an empty set
    // is not.
    d.push(define("zb_set_all_kind", &[&s, &wanted], boolean(), {
        let mut st = vec![
            when(
                eq(call("zb_set_len", vec![s.e()], i64()), int(0)),
                vec![ret(bool(false))],
            ),
            n.decl(len(s.e())),
        ];
        st.extend(for_range(
            &i,
            t.first(t.ctrl(s.e())),
            n.e(),
            vec![when(
                and(
                    ne(t.hash_of(t.entry(s.e(), i.e())), int(TOMB)),
                    ne(kind(t.key_of(t.entry(s.e(), i.e()))), wanted.e()),
                ),
                vec![ret(bool(false))],
            )],
        ));
        st.push(ret(bool(true)));
        st
    }));
    // Two sets one of which holds only sets and the other only tuples:
    // no value of one equals a value of the other.
    // Out of line: the kinds are looked at only when the masks do not
    // answer, and a caller need not carry the walk.
    d.push(define_cold(
        "zb_set_disjoint_kinds",
        &[&s, &other],
        boolean(),
        vec![
            when(
                or(
                    eq(call("zb_set_len", vec![s.e()], i64()), int(0)),
                    eq(call("zb_set_len", vec![other.e()], i64()), int(0)),
                ),
                vec![ret(bool(false))],
            ),
            when(
                eq(kind(first(&s)), kind(first(&other))),
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
    // The box itself, once it is known to hold a set: what a slot that
    // stores sets as boxes takes from a dynamic value.
    d.push(define(
        "zb_set_as_box",
        &[&x],
        any(),
        vec![
            expr(call("zb_set_unbox", vec![x.e()], st_.clone())),
            ret(x.e()),
        ],
    ));
    d
}

/// `zb_set_contains_<suffix>`: whether the library's set holds a value
/// equal to a key of field `probe` that is not a box. The key hashes as
/// its box would, so the probe lands where the box's would.
pub(crate) fn set_probe_by(list_type: TypeId, probe: &Field, suffix: &str) -> Decl {
    let t = Table {
        list_type,
        prefix: "zb_set",
        sfx: String::new(),
        key: &Field::Any,
        value: None,
        tag: SET_TAG,
        mask: Mask::Lazy,
    };
    let s = borrowed("s", set_type(list_type));
    let key = borrowed("key", probe.ty());
    let n = local("n", i64());
    let i = local("i", i64());
    let e = local("e", i64());
    let en = local("en", t.entry_ty());
    let index = local("index", list_of(list_type, i64()));
    let mask = local("mask", i64());
    let slot = local("slot", i64());
    let h = local("h", i64());
    let matches = |stored: Expr| probe.eq_boxed(stored, key.e());
    let mut scan = vec![n.decl(len(s.e()))];
    scan.extend(for_range(
        &i,
        int(0),
        n.e(),
        vec![when(
            matches(t.key_of(t.entry(s.e(), i.e()))),
            vec![ret(bool(true))],
        )],
    ));
    scan.push(ret(bool(false)));
    define(
        &format!("zb_set_contains_{suffix}"),
        &[&s, &key],
        boolean(),
        vec![
            when(eq(t.ctrl(s.e()), int(0)), scan),
            index.decl(t.index_at(t.ctrl(s.e()))),
            mask.decl(sub(sub(len(index.e()), int(INDEX_HEAD)), int(1))),
            h.decl(call("zb_hash_mix", vec![probe.hash(key.e())], i64())),
            slot.decl(bitand(h.e(), mask.e())),
            i.decl(int(0)),
            while_(
                le(i.e(), mask.e()),
                vec![
                    e.decl(idx(index.e(), add(slot.e(), int(INDEX_HEAD)), i64())),
                    when(lt(e.e(), int(0)), vec![ret(bool(false))]),
                    en.decl(t.entry(s.e(), e.e())),
                    when(
                        eq(t.hash_of(en.e()), h.e()),
                        vec![when(matches(t.key_of(en.e())), vec![ret(bool(true))])],
                    ),
                    slot.set(bitand(add(slot.e(), int(1)), mask.e())),
                    i.add_assign(int(1)),
                ],
            ),
            ret(bool(false)),
        ],
    )
}

/// What the dynamic layer asks of a shape through a frontend's hooks: a
/// lookup by a dynamic key, equality with any boxed dict or set, and a
/// copy from one.
///
/// `find_any` applies the rule a key of one kind meets a stored key of
/// another by: a bool is the int it stands for; a float is an int only
/// when integral and within the ints, and an int a float only when the
/// conversion is exact; a string, None or an instance is never a
/// number or a tuple. Any other stored kind is probed by the dynamic
/// hash and compared as boxes, which agree across kinds already.
fn probe_any_ops(t: &Table) -> Vec<Decl> {
    use crate::dynamic::{BOOL, CUSTOM, FLOAT, INT, STR, UINT};
    let lt_ = t.list_ty();
    let anys = t.anys();
    let d = borrowed("d", lt_.clone());
    let k = borrowed("k", any());
    let other = kept("other", any());
    let cat = local("cat", i64());
    let f = local("f", f64());
    let n = local("n", i64());
    let i = local("i", i64());
    let e = local("e", i64());
    let en = local("en", t.entry_ty());
    let h = local("h", i64());
    let index = local("index", t.ints());
    let mask = local("mask", i64());
    let slot = local("slot", i64());
    let out = local("out", lt_.clone());
    let keys = local("keys", anys.clone());
    let kb = local("kb", any());
    let category = |x: Expr| call("zb_any_category", vec![x], i64());
    let number_i64 = |x: Expr, c: Expr| call("zb_number_i64", vec![x, c], i64());
    let is = |c: i64| eq(cat.e(), int(c));
    let integral = || or(is(BOOL), or(is(INT), is(UINT)));
    let find = |k: Expr| ret(t.call("find", vec![d.e(), k], i64()));
    // An integral float within the ints, as the int it is.
    let int_of_float = || {
        vec![
            f.decl(call("zb_number_f64", vec![k.e(), int(FLOAT)], f64())),
            // Converted only once known to fit: the conversion traps
            // otherwise.
            when(
                and(
                    ge(f.e(), float(-9_223_372_036_854_775_808.0)),
                    lt(f.e(), float(9_223_372_036_854_775_808.0)),
                ),
                vec![when(
                    eq(f.e(), cast(cast(f.e(), i64()), f64())),
                    vec![find(cast(f.e(), i64()))],
                )],
            ),
            ret(int(-1)),
        ]
    };
    let body = match t.key {
        Field::Any => vec![find(k.e())],
        Field::Int => vec![
            cat.decl(category(k.e())),
            when(integral(), vec![find(number_i64(k.e(), cat.e()))]),
            when(is(FLOAT), int_of_float()),
            ret(int(-1)),
        ],
        Field::Bool => vec![
            cat.decl(category(k.e())),
            when(
                integral(),
                vec![
                    n.decl(number_i64(k.e(), cat.e())),
                    when(
                        or(eq(n.e(), int(0)), eq(n.e(), int(1))),
                        vec![find(eq(n.e(), int(1)))],
                    ),
                ],
            ),
            when(
                is(FLOAT),
                vec![
                    f.decl(call("zb_number_f64", vec![k.e(), int(FLOAT)], f64())),
                    when(
                        or(eq(f.e(), float(0.0)), eq(f.e(), float(1.0))),
                        vec![find(eq(f.e(), float(1.0)))],
                    ),
                ],
            ),
            ret(int(-1)),
        ],
        Field::Float => vec![
            cat.decl(category(k.e())),
            when(
                integral(),
                vec![
                    n.decl(number_i64(k.e(), cat.e())),
                    f.decl(cast(n.e(), f64())),
                    // Exact only below 2^63, where the way back cannot
                    // overflow.
                    when(
                        lt(f.e(), float(9_223_372_036_854_775_808.0)),
                        vec![when(eq(cast(f.e(), i64()), n.e()), vec![find(f.e())])],
                    ),
                    ret(int(-1)),
                ],
            ),
            when(
                is(FLOAT),
                vec![find(call("zb_number_f64", vec![k.e(), int(FLOAT)], f64()))],
            ),
            ret(int(-1)),
        ],
        Field::Str => vec![
            when(
                eq(category(k.e()), int(STR)),
                vec![find(call("zb_box_get_str", vec![k.e()], string()))],
            ),
            ret(int(-1)),
        ],
        key => {
            let stored = |en: Expr| key.boxed(t.key_of(en));
            let matches = |en: Expr, found: Expr| {
                when(
                    call("zb_any_eq", vec![stored(en), k.e()], boolean()),
                    vec![ret(found)],
                )
            };
            let mut scan = vec![n.decl(len(d.e()))];
            scan.extend(for_range(
                &i,
                int(0),
                n.e(),
                vec![matches(t.entry(d.e(), i.e()), i.e())],
            ));
            scan.push(ret(int(-1)));
            vec![
                when(eq(t.ctrl(d.e()), int(0)), scan),
                h.decl(call(
                    "zb_hash_mix",
                    vec![call("zb_any_hash", vec![k.e()], i64())],
                    i64(),
                )),
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
                        when(eq(t.hash_of(en.e()), h.e()), vec![matches(en.e(), e.e())]),
                        slot.set(bitand(add(slot.e(), int(1)), mask.e())),
                        i.add_assign(int(1)),
                    ],
                ),
                ret(int(-1)),
            ]
        }
    };
    let mut out_decls = vec![define(&t.name("find_any"), &[&d, &k], i64(), body)];
    let found = || t.call("find_any", vec![d.e(), k.e()], i64());
    out_decls.push(define(
        &t.name("contains_any"),
        &[&d, &k],
        boolean(),
        vec![ret(ge(found(), int(0)))],
    ));
    // Whether a box holds a dict (for a dict shape) or a set of any
    // shape.
    let (first, last) = keyed_kind_range();
    let own_tag = if t.value.is_some() { DICT_TAG } else { SET_TAG };
    let (shapes_from, shapes_to) = if t.value.is_some() {
        (first, set_kinds_from())
    } else {
        (set_kinds_from(), last)
    };
    let kind = |x: Expr| call("zb_any_kind", vec![x], i64());
    let is_like = |x: Expr| {
        and(
            eq(category(x.clone()), int(CUSTOM)),
            or(
                eq(kind(x.clone()), int(own_tag >> 8)),
                and(
                    ge(kind(x.clone()), int(shapes_from)),
                    lt(kind(x), int(shapes_to)),
                ),
            ),
        )
    };
    let boxed_key = |en: Expr| t.key.boxed(t.key_of(en));
    if let Some(vf) = t.value {
        out_decls.push(define(&t.name("get_any"), &[&d, &k], vf.ty(), {
            vec![
                e.decl(found()),
                when(
                    lt(e.e(), int(0)),
                    vec![fatal("KeyError", call("zb_any_str", vec![k.e()], string()))],
                ),
                ret(t.value_of(t.entry(d.e(), e.e()))),
            ]
        }));
        // The value as a dynamic value, or `default` when absent; and the
        // same with the entry removed.
        let default = kept("default", any());
        let x = local("x", any());
        out_decls.push(define(
            &t.name("get_default_any"),
            &[&d, &k, &default],
            any(),
            vec![
                e.decl(found()),
                when(lt(e.e(), int(0)), vec![ret(default.e())]),
                ret(vf.boxed(t.value_of(t.entry(d.e(), e.e())))),
            ],
        ));
        out_decls.push(define(
            &t.name("pop_default_any"),
            &[&d, &k, &default],
            any(),
            vec![
                e.decl(found()),
                when(lt(e.e(), int(0)), vec![ret(default.e())]),
                x.decl(vf.boxed(t.value_of(t.entry(d.e(), e.e())))),
                t.go("delete_at", vec![d.e(), e.e()]),
                ret(x.e()),
            ],
        ));
        out_decls.push(define(&t.name("pop_any"), &[&d, &k], any(), {
            vec![
                e.decl(found()),
                when(
                    lt(e.e(), int(0)),
                    vec![fatal("KeyError", call("zb_any_str", vec![k.e()], string()))],
                ),
                x.decl(vf.boxed(t.value_of(t.entry(d.e(), e.e())))),
                t.go("delete_at", vec![d.e(), e.e()]),
                ret(x.e()),
            ]
        }));
        out_decls.push(define(&t.name("del_any"), &[&d, &k], unit(), {
            vec![
                e.decl(found()),
                when(
                    lt(e.e(), int(0)),
                    vec![fatal("KeyError", call("zb_any_str", vec![k.e()], string()))],
                ),
                t.go("delete_at", vec![d.e(), e.e()]),
                ret_void(),
            ]
        }));
        // Equal to any dict of the same keys, each keyed to an equal
        // value, whatever either's shape.
        out_decls.push(define(&t.name("eq_any"), &[&d, &other], boolean(), {
            let mut st = vec![
                when(not(is_like(other.e())), vec![ret(bool(false))]),
                when(
                    ne(
                        call("zb_any_len", vec![other.e()], i64()),
                        t.call("len", vec![d.e()], i64()),
                    ),
                    vec![ret(bool(false))],
                ),
                n.decl(len(d.e())),
            ];
            st.extend(for_range(
                &i,
                t.first(t.ctrl(d.e())),
                n.e(),
                vec![
                    en.decl(t.entry(d.e(), i.e())),
                    when(
                        ne(t.hash_of(en.e()), int(TOMB)),
                        vec![
                            kb.decl(boxed_key(en.e())),
                            when(
                                not(call("zb_any_contains", vec![other.e(), kb.e()], boolean())),
                                vec![ret(bool(false))],
                            ),
                            when(
                                not(call(
                                    "zb_any_eq",
                                    vec![
                                        vf.boxed(t.value_of(en.e())),
                                        call("zb_any_getitem", vec![other.e(), kb.e()], any()),
                                    ],
                                    boolean(),
                                )),
                                vec![ret(bool(false))],
                            ),
                        ],
                    ),
                ],
            ));
            st.push(ret(bool(true)));
            st
        }));
        // A copy of any dict, each key and value read as this shape's.
        out_decls.push(define(&t.name("from_dyn"), &[&other], lt_.clone(), {
            let mut st = vec![
                when(
                    not(is_like(other.e())),
                    vec![fatal(
                        "TypeError",
                        add(
                            text("expected a dict, got "),
                            call("zb_any_type", vec![other.e()], string()),
                        ),
                    )],
                ),
                out.decl(t.call("new", vec![], lt_.clone())),
                keys.decl(call("zb_any_iter", vec![other.e()], anys.clone())),
                n.decl(len(keys.e())),
            ];
            st.extend(for_range(
                &i,
                int(0),
                n.e(),
                vec![
                    kb.decl(idx(keys.e(), i.e(), any())),
                    t.go(
                        "set",
                        vec![
                            out.e(),
                            t.key.read(kb.e()),
                            vf.read(call("zb_any_getitem", vec![other.e(), kb.e()], any())),
                        ],
                    ),
                ],
            ));
            st.push(ret(out.e()));
            st
        }));
    } else {
        // A value found by the rule, removed; absent, nothing, or the
        // KeyError of `remove`.
        for (op, raises) in [("discard_any", false), ("remove_any", true)] {
            out_decls.push(define(&t.name(op), &[&d, &k], unit(), {
                let mut st = vec![e.decl(found())];
                if raises {
                    st.push(when(
                        lt(e.e(), int(0)),
                        vec![fatal("KeyError", call("zb_any_str", vec![k.e()], string()))],
                    ));
                } else {
                    st.push(when(lt(e.e(), int(0)), vec![ret_void()]));
                }
                st.push(t.go("delete_at", vec![d.e(), e.e()]));
                st.push(ret_void());
                st
            }));
        }
        // Equal to any set of equal values, whatever either's shape.
        out_decls.push(define(&t.name("eq_any"), &[&d, &other], boolean(), {
            let mut st = vec![
                when(not(is_like(other.e())), vec![ret(bool(false))]),
                when(
                    ne(
                        call("zb_any_len", vec![other.e()], i64()),
                        t.call("len", vec![d.e()], i64()),
                    ),
                    vec![ret(bool(false))],
                ),
                n.decl(len(d.e())),
            ];
            st.extend(for_range(
                &i,
                t.first(t.ctrl(d.e())),
                n.e(),
                vec![
                    en.decl(t.entry(d.e(), i.e())),
                    when(
                        ne(t.hash_of(en.e()), int(TOMB)),
                        vec![when(
                            not(call(
                                "zb_any_contains",
                                vec![other.e(), boxed_key(en.e())],
                                boolean(),
                            )),
                            vec![ret(bool(false))],
                        )],
                    ),
                ],
            ));
            st.push(ret(bool(true)));
            st
        }));
        out_decls.push(define(&t.name("from_dyn"), &[&other], lt_.clone(), {
            let mut st = vec![
                when(
                    not(is_like(other.e())),
                    vec![fatal(
                        "TypeError",
                        add(
                            text("expected a set, got "),
                            call("zb_any_type", vec![other.e()], string()),
                        ),
                    )],
                ),
                out.decl(t.call("new", vec![], lt_.clone())),
                keys.decl(call("zb_any_iter", vec![other.e()], anys.clone())),
                n.decl(len(keys.e())),
            ];
            st.extend(for_range(
                &i,
                int(0),
                n.e(),
                vec![t.go(
                    "add",
                    vec![out.e(), t.key.read(idx(keys.e(), i.e(), any()))],
                )],
            ));
            st.push(ret(out.e()));
            st
        }));
    }
    out_decls
}
