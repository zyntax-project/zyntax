//! Foreign objects in Lua, against an embedder standing in for a real
//! one: modules it answers `require` with, a class the program calls,
//! instances it indexes, and a method called with `:`. Its own process,
//! since the embedder is installed once per process.

use std::sync::Mutex;

use zyntax_embed::foreign::{self, Any, Foreign, ForeignError, Value};
use zyntax_embed::{TieredConfig, TieredRuntime};

#[derive(Clone, Debug)]
enum Obj {
    Shapes,
    Log,
    /// A function of a module: takes no receiver.
    Function(&'static str),
    Point {
        x: f64,
        y: f64,
    },
    /// A method as a value: its receiver comes first.
    Method(String),
}

struct State {
    objects: Vec<Obj>,
    /// Boxes of each object the program holds.
    held: Vec<i64>,
    records: Vec<String>,
}

static STATE: Mutex<State> = Mutex::new(State {
    objects: Vec::new(),
    held: Vec::new(),
    records: Vec::new(),
});

fn make(obj: Obj) -> Any {
    let mut s = STATE.lock().unwrap();
    s.objects.push(obj);
    s.held.push(1);
    foreign::boxed(s.objects.len())
}

/// Another box of the object behind `word`.
fn again(word: usize) -> Any {
    STATE.lock().unwrap().held[word - 1] += 1;
    foreign::boxed(word)
}

fn object(word: usize) -> Obj {
    STATE.lock().unwrap().objects[word - 1].clone()
}

fn number(any: Any) -> Result<f64, ForeignError> {
    match unsafe { foreign::read(any) } {
        Value::Int(i) => Ok(i as f64),
        Value::Float(f) => Ok(f),
        other => Err(ForeignError::new(
            "TypeError",
            format!("expected a number, got {other:?}"),
        )),
    }
}

fn text(any: Any) -> String {
    match unsafe { foreign::read(any) } {
        Value::None => "nil".into(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => format!("{f:?}"),
        Value::Str(s) => s.to_string(),
        Value::Foreign(w) => Stand.text(w),
        Value::Other(tag) => format!("<tag {tag:#x}>"),
    }
}

fn type_of(obj: &Obj) -> &'static str {
    match obj {
        Obj::Shapes | Obj::Log => "module",
        Obj::Function(_) => "function",
        Obj::Point { .. } => "Point",
        Obj::Method(_) => "method",
    }
}

struct Stand;

impl Foreign for Stand {
    fn get(&self, word: usize, name: &str) -> Result<Any, ForeignError> {
        match (object(word), name) {
            (Obj::Shapes, "Point") => Ok(make(Obj::Function("Point"))),
            (Obj::Shapes, "same") => Ok(make(Obj::Function("same"))),
            (Obj::Shapes, "version") => Ok(foreign::int(3)),
            (Obj::Log, "record") => Ok(make(Obj::Function("record"))),
            (Obj::Point { x, .. }, "x") => Ok(foreign::float(x)),
            (Obj::Point { y, .. }, "y") => Ok(foreign::float(y)),
            (Obj::Point { .. }, "length" | "scaled") => Ok(make(Obj::Method(name.into()))),
            (obj, _) => Err(ForeignError::new(
                "AttributeError",
                format!("{} has no member '{name}'", type_of(&obj)),
            )),
        }
    }

    fn set(&self, word: usize, name: &str, value: Any) -> Result<(), ForeignError> {
        let v = number(value)?;
        let mut s = STATE.lock().unwrap();
        match (&mut s.objects[word - 1], name) {
            (Obj::Point { x, .. }, "x") => *x = v,
            (Obj::Point { y, .. }, "y") => *y = v,
            _ => {
                return Err(ForeignError::new(
                    "AttributeError",
                    format!("cannot write '{name}'"),
                ));
            }
        }
        Ok(())
    }

    fn call(&self, word: usize, args: &[Any]) -> Result<Any, ForeignError> {
        match object(word) {
            Obj::Function("Point") => Ok(make(Obj::Point {
                x: number(args[0])?,
                y: number(args[1])?,
            })),
            Obj::Function("same") => match unsafe { foreign::read(args[0]) } {
                Value::Foreign(w) => Ok(again(w)),
                _ => Err(ForeignError::new("TypeError", "same() takes a point")),
            },
            Obj::Function("record") => {
                let line = args.iter().map(|&a| text(a)).collect::<Vec<_>>().join("\t");
                STATE.lock().unwrap().records.push(line);
                Ok(foreign::none())
            }
            Obj::Method(name) => match unsafe { foreign::read(args[0]) } {
                Value::Foreign(receiver) => self.invoke(receiver, &name, &args[1..]),
                _ => Err(ForeignError::new(
                    "TypeError",
                    "a method needs its receiver",
                )),
            },
            obj => Err(ForeignError::new(
                "TypeError",
                format!("{} is not callable", type_of(&obj)),
            )),
        }
    }

    fn invoke(&self, word: usize, name: &str, args: &[Any]) -> Result<Any, ForeignError> {
        match (object(word), name) {
            (Obj::Point { x, y }, "length") => Ok(foreign::float((x * x + y * y).sqrt())),
            (Obj::Point { x, y }, "scaled") => {
                let k = number(args[0])?;
                Ok(make(Obj::Point { x: x * k, y: y * k }))
            }
            // A module's function: its member, called.
            (Obj::Shapes | Obj::Log, _) => {
                let member = self.get(word, name)?;
                let Some(member) = (unsafe { foreign::word(member) }) else {
                    return Err(ForeignError::new(
                        "TypeError",
                        format!("'{name}' is not callable"),
                    ));
                };
                self.call(member, args)
            }
            (obj, _) => Err(ForeignError::new(
                "AttributeError",
                format!("{} has no method '{name}'", type_of(&obj)),
            )),
        }
    }

    fn text(&self, word: usize) -> String {
        match object(word) {
            Obj::Point { x, y } => format!("Point({x}, {y})"),
            obj => type_of(&obj).to_string(),
        }
    }

    fn type_name(&self, word: usize) -> String {
        type_of(&object(word)).to_string()
    }

    fn equals(&self, a: usize, b: usize) -> bool {
        a == b
    }

    fn hash(&self, word: usize) -> i64 {
        word as i64
    }

    fn import(&self, name: &str) -> Result<Option<Any>, ForeignError> {
        Ok(match name {
            "shapes" => Some(make(Obj::Shapes)),
            "log" => Some(make(Obj::Log)),
            _ => None,
        })
    }

    fn release(&self, word: usize) {
        STATE.lock().unwrap().held[word - 1] -= 1;
    }
}

const PROGRAM: &str = r#"
local shapes = require("shapes")
local log = require("log")
local p = shapes.Point(3, 4)
log.record(p.x + p.y)
log.record(p:length())
local q = p:scaled(2)
log.record(tostring(q))
p.x = 6
log.record(p.x, p == q, p == shapes.same(p), p ~= q)
log.record(type(p), shapes.version)
local ok, err = pcall(function() return p.missing end)
log.record(ok, err)
log.record(pcall(require, "nowhere"))
local seen = {}
seen[p] = "seen"
log.record(seen[shapes.same(p)])
log.record(require("shapes") == shapes)
log.record(debug.setuservalue(p, 1), debug.getuservalue(p), pcall(debug.setuservalue, p))
"#;

#[test]
fn a_program_uses_the_embedders_objects() {
    assert_eq!(
        zyntax_builtins::FOREIGN_TAG as u32,
        foreign::FOREIGN_TAG,
        "the library and the embedding agree on the tag"
    );
    assert!(foreign::install(Box::new(Stand)));
    let program = zyntax_lua::parse_program(PROGRAM, "foreign.lua").expect("parses");
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    zyntax_lua::register_runtime(&mut rt).expect("registered");
    rt.enter_only_through_entry_points();
    zyntax_lua::set_runtime(&mut rt);
    rt.compile_typed_program(program).expect("compiles");
    rt.call_raw(zyntax_lua::ENTRY, &[]).expect("runs");
    let s = STATE.lock().unwrap();
    let records: Vec<&str> = s.records.iter().map(String::as_str).collect();
    assert_eq!(
        records[..5],
        [
            "7.0",
            "5.0",
            "Point(6, 8)",
            "6.0\tfalse\ttrue\ttrue",
            "userdata\t3",
        ]
    );
    assert!(
        records[5].starts_with("false\t") && records[5].ends_with("Point has no member 'missing'"),
        "{}",
        records[5]
    );
    assert!(records[6].starts_with("false\t"), "{}", records[6]);
    // `require` gives the module it loaded before: the same value.
    assert_eq!(records[7..9], ["seen", "true"]);
    // A foreign object is a full userdata with no user values.
    assert_eq!(
        records[9],
        "nil\tnil\tfalse\tbad argument #2 to 'debug.setuservalue' (value expected)"
    );
    assert!(
        s.held.iter().all(|&n| n >= 0),
        "an object was released more often than boxed: {:?}",
        s.held
    );
    std::mem::forget(rt);
}
