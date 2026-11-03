//! The `io` library: files as values of their own kind over the host's
//! streams, the default input and output, and the reference's file
//! methods reached through the file metatable.

use super::*;

/// The last of the standard streams' handles, as the host numbers
/// them: stdin, stdout and stderr are 1, 2 and 3.
const STDERR: i64 = 3;

mod files;
mod lines;
mod streams;

struct Context {
    anys: Type,
    x: Local,
    y: Local,
    a: Local,
    args: Local,
    out: Local,
    vals: Local,
    env: Local,
    packed: Local,
    what: Local,
    name: Local,
    mode: Local,
    s: Local,
    fmt: Local,
    h: Local,
    i: Local,
    n: Local,
    k: Local,
    r: Local,
    ok: Local,
    tb: Local,
}

pub(super) fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let table = t.table();
    let x = kept("x", any());
    let y = kept("y", any());
    let a = kept("a", any());
    let args = kept("args", anys.clone());
    let out = borrowed("out", anys.clone());
    let vals = borrowed("vals", anys.clone());
    let env = borrowed("env", anys.clone());
    let packed = kept("packed", any());
    let what = kept("what", string());
    let name = kept("name", string());
    let mode = kept("mode", string());
    let s = kept("s", string());
    let fmt = kept("fmt", string());
    let h = local("h", i64());
    let i = local("i", i64());
    let n = local("n", i64());
    let k = local("k", i64());
    let r = local("r", i64());
    let ok = local("ok", boolean());
    let tb = kept("t", table.clone());
    let mut d = Vec::new();
    let context = Context {
        anys,
        x,
        y,
        a,
        args,
        out,
        vals,
        env,
        packed,
        what,
        name,
        mode,
        s,
        fmt,
        h,
        i,
        n,
        k,
        r,
        ok,
        tb,
    };
    files::append(&context, t, &mut d);
    streams::append(&context, t, &mut d);
    lines::append(&context, t, &mut d);
    d
}
