//! Binary chunks: what `string.dump` writes and `load` reads back.
//!
//! A binary chunk starts with the reference's header, so `load` tells
//! it from text and refuses a corrupted one as `lundump.c` does. After
//! the header come flags (stripped, main chunk) and the function as
//! source: unless stripped, the
//! chunk's source name; the line its text starts on; its upvalues'
//! names in Lua's order; the folded constants it reads; its text.
//! Every field is length-prefixed, so a chunk cut anywhere is refused
//! as truncated.
//!
//! Loading one compiles the text again. A main chunk is compiled as
//! it was; a function is compiled inside a chunk that declares its
//! constants and upvalues as locals and returns it (see
//! `wrapper_text`).
//!
//! What `string.dump` needs of a function value is its key, which
//! every record carries: each chunk's text is kept here when it is
//! compiled, and the functions of a chunk are found in its text again
//! the first time one of them is dumped.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// `LUA_SIGNATURE`, `LUAC_VERSION`, `LUAC_FORMAT` and `LUAC_DATA`.
pub(crate) const SIGNATURE: &[u8] = b"\x1bLua";
const VERSION: u8 = 0x54;
const FORMAT: u8 = 0;
const DATA: &[u8] = b"\x19\x93\r\n\x1a\n";
/// The sizes of an instruction, a `lua_Integer` and a `lua_Number`.
const SIZES: [(u8, &str); 3] = [
    (4, "Instruction"),
    (8, "lua_Integer"),
    (8, "lua_Number"),
];
/// `LUAC_INT` and `LUAC_NUM`, which tell the byte order and the float
/// format.
const LUAC_INT: i64 = 0x5678;
const LUAC_NUM: f64 = 370.5;

/// The flags byte after the header.
const STRIPPED: u8 = 1;
const MAIN: u8 = 2;

/// A chunk of the program as it was compiled: its text (as parsed,
/// private-use characters standing for bytes that are not UTF-8), its
/// source name and whether it was loaded without debug information.
struct Chunk {
    text: String,
    source: String,
    stripped: bool,
    /// Its functions, by number, found the first time one is dumped.
    funcs: OnceLock<Vec<Func>>,
}

/// A function of a chunk as a dump holds it.
struct Func {
    text: String,
    line: usize,
    upvalues: Vec<String>,
    /// The declarations of the folded constants it reads, which hold
    /// no upvalue: `local N <const> = 1;` each, in source order.
    consts: String,
    main: bool,
}

fn chunks() -> &'static Mutex<HashMap<i64, Arc<Chunk>>> {
    static CHUNKS: OnceLock<Mutex<HashMap<i64, Arc<Chunk>>>> = OnceLock::new();
    CHUNKS.get_or_init(Default::default)
}

/// Keep the text chunk `index` was compiled from, for dumps of its
/// functions.
pub(crate) fn note_chunk(index: i64, text: &str, source: &str, stripped: bool) {
    let chunk = Chunk {
        text: text.to_string(),
        source: source.to_string(),
        stripped,
        funcs: OnceLock::new(),
    };
    if let Ok(mut chunks) = chunks().lock() {
        chunks.insert(index, Arc::new(chunk));
    }
}

/// The functions of a chunk's text, numbered as its compilation
/// numbered them: the resolver walks the same text the same way.
fn functions(text: &str) -> Vec<Func> {
    let Ok(ast) = full_moon::parse_fallible(text, full_moon::LuaVersion::lua54()).into_result()
    else {
        return Vec::new();
    };
    let scopes = crate::scope::resolve(&ast);
    let text_of_chunk = text;
    let consts = Consts::of(&scopes);
    scopes
        .funcs
        .iter()
        .enumerate()
        .map(|(k, info)| {
            let upvalues = info
                .upvalues
                .iter()
                .map(|u| match u {
                    crate::scope::Upvalue::Var(v) => scopes.var(*v).name.clone(),
                    crate::scope::Upvalue::Env => "_ENV".to_string(),
                })
                .collect();
            if k == 0 {
                return Func {
                    text: text.to_string(),
                    line: 1,
                    upvalues,
                    consts: String::new(),
                    main: true,
                };
            }
            // `(params) body end`, with a method's `self` spelled out.
            let body = &text[info.body.0..info.body.1];
            let text = match body.strip_prefix('(') {
                Some(rest) if info.method => {
                    let more = info.params.len() > 1 || info.is_vararg;
                    format!("function(self{}{rest}", if more { ", " } else { "" })
                }
                _ => format!("function{body}"),
            };
            Func {
                text,
                line: info.line,
                upvalues,
                consts: consts.read_by(info.body, text_of_chunk),
                main: false,
            }
        })
        .collect()
}

/// The folded constants of a chunk: Lua keeps no upvalue for one, so
/// a dumped function that reads one carries its declaration.
struct Consts<'s> {
    scopes: &'s crate::scope::Scopes,
    /// Each name that reads a folded constant, by byte offset.
    uses: Vec<(usize, crate::scope::VarId)>,
    declared_at: HashMap<crate::scope::VarId, usize>,
}

impl<'s> Consts<'s> {
    fn of(scopes: &'s crate::scope::Scopes) -> Self {
        use crate::scope::Binding;
        let uses = scopes
            .names
            .iter()
            .filter_map(|(&at, binding)| match binding {
                Binding::Local(v) | Binding::Upvalue(v) if scopes.var(*v).folded => Some((at, *v)),
                _ => None,
            })
            .collect();
        let declared_at = scopes.decls.iter().map(|(&at, &v)| (v, at)).collect();
        Consts {
            scopes,
            uses,
            declared_at,
        }
    }

    /// The declarations, in source order, of the constants declared
    /// before `body` that it reads, directly or through another
    /// constant's value: `local N <const> = 1; ` each.
    fn read_by(&self, body: (usize, usize), text: &str) -> String {
        let read_in = |(a, b): (usize, usize)| {
            self.uses
                .iter()
                .filter(move |(at, _)| (a..b).contains(at))
                .map(|&(_, v)| v)
        };
        let mut found = std::collections::BTreeMap::new();
        let mut pending: Vec<_> = read_in(body)
            .filter(|v| self.declared_at.get(v).is_some_and(|&at| at < body.0))
            .collect();
        while let Some(v) = pending.pop() {
            let at = self.declared_at.get(&v);
            let init = self.scopes.const_inits.get(&v);
            let (Some(&at), Some(&init)) = (at, init) else {
                continue;
            };
            if found.insert(at, (v, init)).is_none() {
                pending.extend(read_in(init));
            }
        }
        found
            .values()
            .map(|&(v, (a, b))| {
                format!(
                    "local {} <const> = {}; ",
                    self.scopes.var(v).name,
                    &text[a..b]
                )
            })
            .collect()
    }
}

/// `string.dump` of the function with key `key`: the binary chunk, or
/// none for a function no chunk of the program defines.
pub(crate) fn dump(key: i64, strip: bool) -> Option<Vec<u8>> {
    if key < 0 {
        return None;
    }
    let chunk = chunks().lock().ok()?.get(&(key >> 32)).cloned()?;
    let funcs = chunk.funcs.get_or_init(|| functions(&chunk.text));
    let f = funcs.get((key & 0xffff_ffff) as usize)?;
    let stripped = strip || chunk.stripped;
    let mut out = Vec::with_capacity(f.text.len() + 64);
    out.extend_from_slice(SIGNATURE);
    out.extend_from_slice(&[VERSION, FORMAT]);
    out.extend_from_slice(DATA);
    out.extend(SIZES.iter().map(|(size, _)| size));
    out.extend_from_slice(&LUAC_INT.to_le_bytes());
    out.extend_from_slice(&LUAC_NUM.to_le_bytes());
    out.push(if stripped { STRIPPED } else { 0 } | if f.main { MAIN } else { 0 });
    if !stripped {
        put_bytes(&mut out, chunk.source.as_bytes());
    }
    put_varint(&mut out, f.line as u64);
    put_varint(&mut out, f.upvalues.len() as u64);
    for name in &f.upvalues {
        put_bytes(&mut out, name.as_bytes());
    }
    put_bytes(&mut out, &crate::source_bytes(&f.consts));
    put_bytes(&mut out, &crate::source_bytes(&f.text));
    Some(out)
}

fn put_varint(out: &mut Vec<u8>, mut n: u64) {
    loop {
        let byte = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// A binary chunk read back.
pub(crate) struct Undumped {
    /// The source name, `=?` when stripped.
    pub source: String,
    pub stripped: bool,
    pub main: bool,
    pub line: usize,
    pub upvalues: Vec<String>,
    pub consts: Vec<u8>,
    pub text: Vec<u8>,
}

/// Reads a binary chunk, refusing it as `lundump.c` does: the reason
/// is what goes in `bad binary format (...)`.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

const TRUNCATED: &str = "truncated chunk";

impl<'a> Reader<'a> {
    fn block(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len());
        let end = end.ok_or(TRUNCATED)?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn byte(&mut self) -> Result<u8, String> {
        Ok(self.block(1)?[0])
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut n = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            n |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(n);
            }
        }
        Err("corrupted chunk".to_string())
    }

    fn bytes(&mut self) -> Result<&'a [u8], String> {
        let n = usize::try_from(self.varint()?).map_err(|_| TRUNCATED)?;
        self.block(n)
    }

    fn literal(&mut self, expected: &[u8], why: &str) -> Result<(), String> {
        if self.block(expected.len())? != expected {
            return Err(why.to_string());
        }
        Ok(())
    }
}

/// A binary chunk's function, or why it is refused.
pub(crate) fn undump(bytes: &[u8]) -> Result<Undumped, String> {
    let mut r = Reader { bytes, at: 0 };
    r.literal(SIGNATURE, "not a binary chunk")?;
    if r.byte()? != VERSION {
        return Err("version mismatch".to_string());
    }
    if r.byte()? != FORMAT {
        return Err("format mismatch".to_string());
    }
    r.literal(DATA, "corrupted chunk")?;
    for (size, name) in SIZES {
        if r.byte()? != size {
            return Err(format!("{name} size mismatch"));
        }
    }
    let int = r.block(8)?;
    if i64::from_le_bytes(int.try_into().expect("8 bytes")) != LUAC_INT {
        return Err("integer format mismatch".to_string());
    }
    let num = r.block(8)?;
    if f64::from_le_bytes(num.try_into().expect("8 bytes")) != LUAC_NUM {
        return Err("float format mismatch".to_string());
    }
    let flags = r.byte()?;
    if flags & !(STRIPPED | MAIN) != 0 {
        return Err("corrupted chunk".to_string());
    }
    let stripped = flags & STRIPPED != 0;
    let source = if stripped {
        "=?".to_string()
    } else {
        String::from_utf8_lossy(r.bytes()?).into_owned()
    };
    let line = usize::try_from(r.varint()?).map_err(|_| "corrupted chunk")?;
    let count = r.varint()?;
    let mut upvalues = Vec::new();
    for _ in 0..count {
        upvalues.push(String::from_utf8_lossy(r.bytes()?).into_owned());
    }
    let consts = r.bytes()?.to_vec();
    let text = r.bytes()?.to_vec();
    Ok(Undumped {
        source,
        stripped,
        main: flags & MAIN != 0,
        line: line.max(1),
        upvalues,
        consts,
        text,
    })
}

/// The name a binary chunk's errors give it, as `lundump.c` has it:
/// `@name` and `=name` as `name`, a chunk named for its own bytes as
/// `binary string`.
pub(crate) fn error_name(name: &[u8]) -> String {
    match name.first() {
        Some(b'@' | b'=') => String::from_utf8_lossy(&name[1..]).into_owned(),
        Some(0x1b) => "binary string".to_string(),
        _ => String::from_utf8_lossy(name).into_owned(),
    }
}

/// The text a dumped function is compiled from: on the line it
/// started on, inside a chunk that declares the constants it reads and
/// then its upvalues as locals, in their order, the first holding the
/// chunk's environment; the chunk returns the function, or with `call`
/// what the function returns given the chunk's arguments. A main chunk
/// is its own text.
pub(crate) fn wrapper_text(u: &Undumped, call: bool) -> String {
    let text = crate::source_text(&u.text);
    if u.main {
        return text.into_owned();
    }
    let mut out = crate::source_text(&u.consts).into_owned();
    let lines = out.matches('\n').count();
    out.push_str(&"\n".repeat((u.line - 1).saturating_sub(lines)));
    if !u.upvalues.is_empty() {
        out.push_str("local ");
        out.push_str(&u.upvalues.join(", "));
        out.push_str(" = _ENV; ");
    }
    if call {
        out.push_str("return (");
        out.push_str(&text);
        out.push_str(")(...)");
    } else {
        out.push_str("return ");
        out.push_str(&text);
    }
    out
}
