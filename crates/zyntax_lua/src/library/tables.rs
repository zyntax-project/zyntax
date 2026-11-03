//! Tables: the array part, the hash part, the border `#` reports,
//! metatables and the `__index` / `__newindex` events.
//!
//! Keys `1..=n` live in the array part; every other key in the hash
//! part, a dict of the shared library. A float key with an integral
//! value is the integer key. Storing nil under a hash key keeps the
//! key with a nil value, so a traversal that clears entries as it goes
//! sees every one; storing nil at the end of the array part pops it and
//! whatever nils it uncovers, so the array's length stays a border.

use super::*;

/// How many `__index` or `__newindex` handlers one access looks up
/// before it takes them for a loop, as the reference does.
const MAXTAGLOOP: i64 = 2000;

mod indexing;
mod metatables;
mod raw;
mod traversal;

struct Context {
    anys: Type,
    table: Type,
    tb: Local,
    o: Local,
    k: Local,
    v: Local,
    s: Local,
    i: Local,
    n: Local,
    arr: Local,
    h: Local,
    x: Local,
    f: Local,
    cat: Local,
    mt: Local,
    handler: Local,
}

pub(super) fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let table = t.table();
    let tb = kept("t", table.clone());
    let o = kept("o", any());
    let k = kept("k", any());
    let v = kept("v", any());
    let s = kept("s", string());
    let i = local("i", i64());
    let n = local("n", i64());
    let arr = borrowed("arr", anys.clone());
    let h = borrowed("h", t.dict());
    let x = kept("x", any());
    let f = local("f", f64());
    let cat = local("cat", i64());
    let mt = kept("mt", any());
    let handler = kept("handler", any());
    let mut d = Vec::new();
    let context = Context {
        anys,
        table,
        tb,
        o,
        k,
        v,
        s,
        i,
        n,
        arr,
        h,
        x,
        f,
        cat,
        mt,
        handler,
    };
    raw::append(&context, t, &mut d);
    metatables::append(&context, t, &mut d);
    indexing::append(&context, t, &mut d);
    traversal::append(&context, t, &mut d);
    d
}
