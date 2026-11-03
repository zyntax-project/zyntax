//! Info support.

use super::*;

/// Whether `what` is a valid option string: 0 when it is, 1 for an
/// unknown option, 2 for `>`, which only the C API takes.
pub(crate) extern "C" fn host_dbg_check_options(what: zrtl::StringConstPtr) -> i64 {
    let what = unsafe { bytes_of(what) };
    if what.first() == Some(&b'>') {
        return 2;
    }
    if what.iter().all(|c| b"SlnrutfL".contains(c)) {
        0
    } else {
        1
    }
}

pub(super) fn fill(s: &State, what: &[u8], key: i64, level: Option<&Level>) -> Info {
    let func = func_of(s, key);
    let mut fields: Vec<(&'static str, Value)> = Vec::new();
    for c in what {
        match c {
            b'S' => match func {
                Some((chunk, f)) => {
                    let main = is_main(key);
                    fields.push(("source", Value::Str(chunk.source.clone())));
                    fields.push(("short_src", Value::Str(chunk.short_src.clone())));
                    fields.push(("linedefined", Value::Int(if main { 0 } else { f.line })));
                    fields.push((
                        "lastlinedefined",
                        Value::Int(if main { 0 } else { f.last_line }),
                    ));
                    fields.push((
                        "what",
                        Value::Str(if main { "main" } else { "Lua" }.to_string()),
                    ));
                }
                None => {
                    fields.push(("source", Value::Str("=[C]".to_string())));
                    fields.push(("short_src", Value::Str("[C]".to_string())));
                    fields.push(("linedefined", Value::Int(-1)));
                    fields.push(("lastlinedefined", Value::Int(-1)));
                    fields.push(("what", Value::Str("C".to_string())));
                }
            },
            b'l' => {
                let line = match (func, level) {
                    (Some(_), Some(l)) if l.line > 0 => line_part(l.line),
                    _ => -1,
                };
                fields.push(("currentline", Value::Int(line)));
            }
            b'u' => match func {
                Some((_, f)) => {
                    fields.push(("nups", Value::Int(f.upvalues.len() as i64)));
                    fields.push(("nparams", Value::Int(f.nparams)));
                    fields.push(("isvararg", Value::Bool(f.vararg)));
                }
                None => {
                    fields.push(("nups", Value::Int(0)));
                    fields.push(("nparams", Value::Int(0)));
                    fields.push(("isvararg", Value::Bool(true)));
                }
            },
            b'n' => {
                let (name, namewhat, _) = match level {
                    Some(l) => name_of(s, l),
                    None => Default::default(),
                };
                if namewhat.is_empty() {
                    fields.push(("name", Value::Nil));
                } else {
                    fields.push(("name", Value::Str(name)));
                }
                fields.push(("namewhat", Value::Str(namewhat)));
            }
            b't' => {
                fields.push(("istailcall", Value::Bool(level.is_some_and(|l| l.tail))));
            }
            b'r' => {
                fields.push(("ftransfer", Value::Int(0)));
                fields.push(("ntransfer", Value::Int(0)));
            }
            b'L' => match func {
                Some((_, f)) => fields.push(("activelines", Value::Lines(f.active.clone()))),
                None => fields.push(("activelines", Value::Nil)),
            },
            b'f' => fields.push(("func", Value::Func)),
            _ => {}
        }
    }
    Info {
        fields,
        key: func.map(|_| key),
    }
}

/// `getinfo` of stack level `level` of `thread`: 1 and the record
/// ready, or 0 when there is no such level.
pub(crate) extern "C" fn host_dbg_info_level(
    level: i64,
    thread: i64,
    line: i64,
    what: zrtl::StringConstPtr,
) -> i64 {
    let what = unsafe { bytes_of(what) };
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let levels = levels(&s, thread, line, ("getinfo", "field", "debug.getinfo"));
        let Some(l) = usize::try_from(level).ok().and_then(|i| levels.get(i)) else {
            return 0;
        };
        let info = fill(&s, what, l.key, Some(l));
        s.info = info;
        1
    })
}

/// `getinfo` of a function: `key` names one of the program's, -1 one
/// of the library's.
pub(crate) extern "C" fn host_dbg_info_func(key: i64, what: zrtl::StringConstPtr) {
    let what = unsafe { bytes_of(what) };
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let info = fill(&s, what, key, None);
        s.info = info;
    });
}

/// How many fields the record has.
pub(crate) extern "C" fn host_dbg_info_count() -> i64 {
    STATE.with(|s| s.borrow().info.fields.len() as i64)
}

pub(crate) extern "C" fn host_dbg_info_name(i: i64) -> StringPtr {
    STATE.with(|s| {
        let s = s.borrow();
        string_out(s.info.fields.get(i as usize).map_or("", |f| f.0))
    })
}

/// A field's kind: 0 nil, 1 integer, 2 string, 3 boolean, 4 the
/// function, 5 the active lines.
pub(crate) extern "C" fn host_dbg_info_kind(i: i64) -> i64 {
    STATE.with(|s| {
        let s = s.borrow();
        match s.info.fields.get(i as usize).map(|f| &f.1) {
            Some(Value::Int(_)) => 1,
            Some(Value::Str(_)) => 2,
            Some(Value::Bool(_)) => 3,
            Some(Value::Func) => 4,
            Some(Value::Lines(_)) => 5,
            _ => 0,
        }
    })
}

pub(crate) extern "C" fn host_dbg_info_int(i: i64) -> i64 {
    STATE.with(|s| {
        let s = s.borrow();
        match s.info.fields.get(i as usize).map(|f| &f.1) {
            Some(Value::Int(v)) => *v,
            Some(Value::Bool(b)) => *b as i64,
            Some(Value::Lines(lines)) => lines.len() as i64,
            _ => 0,
        }
    })
}

pub(crate) extern "C" fn host_dbg_info_str(i: i64) -> StringPtr {
    STATE.with(|s| {
        let s = s.borrow();
        match s.info.fields.get(i as usize).map(|f| &f.1) {
            Some(Value::Str(v)) => string_out(v),
            _ => string_out(""),
        }
    })
}

/// The `j`th active line of field `i`.
pub(crate) extern "C" fn host_dbg_info_line(i: i64, j: i64) -> i64 {
    STATE.with(|s| {
        let s = s.borrow();
        match s.info.fields.get(i as usize).map(|f| &f.1) {
            Some(Value::Lines(lines)) => lines.get(j as usize).copied().unwrap_or(0),
            _ => 0,
        }
    })
}

/// The key of the function the record is about: -1 for the library's.
pub(crate) extern "C" fn host_dbg_info_key() -> i64 {
    STATE.with(|s| s.borrow().info.key.unwrap_or(C_KEY))
}
