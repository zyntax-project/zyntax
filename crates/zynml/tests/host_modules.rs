//! Importing a module the host owns: its classes become types, its
//! members typed calls through the foreign-object protocol.

use std::sync::Mutex;

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::foreign::{self, Any, Foreign, ForeignError, Value};
use zyntax_embed::host::{HostClass, HostField, HostMethod, HostModule, HostType};
use zyntax_embed::{TieredConfig, TieredRuntime};

#[derive(Clone)]
enum Obj {
    Module,
    Class,
    Prompt { count: i64, level: i64 },
}

struct State {
    objects: Vec<Obj>,
    total: i64,
    said: Vec<String>,
    released: usize,
}

static STATE: Mutex<State> = Mutex::new(State {
    objects: Vec::new(),
    total: 0,
    said: Vec::new(),
    released: 0,
});

fn make(obj: Obj) -> Any {
    let mut state = STATE.lock().unwrap();
    state.objects.push(obj);
    foreign::boxed(state.objects.len())
}

fn int(any: Any) -> Result<i64, ForeignError> {
    match unsafe { foreign::read(any) } {
        Value::Int(v) => Ok(v),
        _ => Err(ForeignError::new("TypeError", "expected an int")),
    }
}

fn text(any: Any) -> Result<String, ForeignError> {
    match unsafe { foreign::read(any) } {
        Value::Str(s) => Ok(s.to_string()),
        _ => Err(ForeignError::new("TypeError", "expected a string")),
    }
}

fn float(any: Any) -> Result<f64, ForeignError> {
    match unsafe { foreign::read(any) } {
        Value::Float(v) => Ok(v),
        _ => Err(ForeignError::new("TypeError", "expected a float")),
    }
}

struct Stand;

impl Foreign for Stand {
    fn import(&self, name: &str) -> Result<Option<Any>, ForeignError> {
        Ok(matches!(name, "game" | "game.Prompt").then(|| make(Obj::Module)))
    }

    fn get(&self, object: usize, name: &str) -> Result<Any, ForeignError> {
        let obj = STATE.lock().unwrap().objects[object - 1].clone();
        match (obj, name) {
            (Obj::Module, "Prompt") => Ok(make(Obj::Class)),
            (Obj::Class, "total") => Ok(foreign::int(STATE.lock().unwrap().total)),
            (Obj::Prompt { level, .. }, "level") => Ok(foreign::int(level)),
            _ => Err(ForeignError::new("AttributeError", format!("no {name}"))),
        }
    }

    fn set(&self, object: usize, name: &str, value: Any) -> Result<(), ForeignError> {
        let v = int(value)?;
        let mut state = STATE.lock().unwrap();
        match (&mut state.objects[object - 1], name) {
            (Obj::Prompt { level, .. }, "level") => *level = v,
            (Obj::Class, "total") => state.total = v,
            _ => return Err(ForeignError::new("AttributeError", format!("no {name}"))),
        }
        Ok(())
    }

    fn call(&self, object: usize, args: &[Any]) -> Result<Any, ForeignError> {
        let obj = STATE.lock().unwrap().objects[object - 1].clone();
        match obj {
            Obj::Class => {
                let count = int(args[0])?;
                Ok(make(Obj::Prompt { count, level: 0 }))
            }
            _ => Err(ForeignError::new("TypeError", "not callable")),
        }
    }

    fn invoke(&self, object: usize, name: &str, args: &[Any]) -> Result<Any, ForeignError> {
        let obj = STATE.lock().unwrap().objects[object - 1].clone();
        match (obj, name) {
            (Obj::Module, "greet") => Ok(foreign::string(&format!("hello, {}", text(args[0])?))),
            (Obj::Class, "question") => {
                Ok(foreign::string(&format!("answer to {}", text(args[0])?)))
            }
            (Obj::Class, "say") => {
                STATE.lock().unwrap().said.push(text(args[0])?);
                Ok(foreign::none())
            }
            (Obj::Prompt { .. }, "bump") => {
                let by = int(args[0])?;
                if let Obj::Prompt { count, .. } = &mut STATE.lock().unwrap().objects[object - 1] {
                    *count += by;
                }
                Ok(foreign::none())
            }
            (Obj::Prompt { count, .. }, "count") => Ok(foreign::int(count)),
            (Obj::Prompt { count, .. }, "scaled") => {
                Ok(foreign::float(count as f64 * float(args[0])?))
            }
            (Obj::Prompt { count, .. }, "is_big") => Ok(foreign::boolean(count > 10)),
            (Obj::Prompt { count, .. }, "twin") => Ok(make(Obj::Prompt { count, level: 0 })),
            (Obj::Prompt { .. }, "fail") => {
                Err(ForeignError::new("RuntimeError", "the host refused"))
            }
            _ => Err(ForeignError::new(
                "AttributeError",
                format!("no method {name}"),
            )),
        }
    }

    fn text(&self, _object: usize) -> String {
        "<host>".into()
    }

    fn type_name(&self, _object: usize) -> String {
        "host".into()
    }

    fn equals(&self, a: usize, b: usize) -> bool {
        a == b
    }

    fn hash(&self, object: usize) -> i64 {
        object as i64
    }

    fn release(&self, _object: usize) {
        STATE.lock().unwrap().released += 1;
    }
}

fn method(name: &str, params: Vec<HostType>, ret: HostType, is_static: bool) -> HostMethod {
    HostMethod {
        name: name.into(),
        key: 0,
        params,
        ret,
        is_static,
        native: None,
    }
}

fn game() -> HostModule {
    let prompt = HostType::Object("game.Prompt".into());
    HostModule {
        name: "game".into(),
        classes: vec![HostClass {
            name: "Prompt".into(),
            type_name: "game.Prompt".into(),
            fields: vec![
                HostField {
                    name: "level".into(),
                    key: 0,
                    ty: HostType::Int,
                    is_static: false,
                    writable: true,
                    native: None,
                },
                HostField {
                    name: "total".into(),
                    key: 0,
                    ty: HostType::Int,
                    is_static: true,
                    writable: true,
                    native: None,
                },
            ],
            methods: vec![
                method("question", vec![HostType::Str], HostType::Str, true),
                method("say", vec![HostType::Str], HostType::Void, true),
                method("bump", vec![HostType::Int], HostType::Void, false),
                method("count", vec![], HostType::Int, false),
                method("scaled", vec![HostType::Float], HostType::Float, false),
                method("is_big", vec![], HostType::Bool, false),
                method("twin", vec![], prompt.clone(), false),
                method("fail", vec![], HostType::Int, false),
            ],
            constructor: Some(method("new", vec![HostType::Int], prompt, true)),
            word: false,
        }],
        functions: vec![method("greet", vec![HostType::Str], HostType::Str, false)],
    }
}

const SRC: &str = r#"
from game import Prompt

def ask(): i64 {
    let answer = Prompt::question("the meaning")
    Prompt::say(answer)
    Prompt::say(greet("world"))
    return 1
}

effect Ask {
    def question(q: String): String
}

handler HostAsk for Ask {
    def question(q: String): String {
        return Prompt::question(q)
    }
}

@effect(Ask)
def interview(): String {
    return question("why")
}

def gate(): i64 {
    with HostAsk {
        Prompt::say(interview())
    }
    return 1
}

def objects(): i64 {
    let p = Prompt::new(4)
    p.bump(3)
    let q = p.twin()
    q.bump(10)
    return p.count() * 100 + q.count()
}

def chained(): i64 {
    return Prompt::new(4).twin().count()
}

def floats(): f64 {
    let p = Prompt::new(4)
    return p.scaled(0.5)
}

def bools(): i64 {
    let small = Prompt::new(4)
    let big = Prompt::new(40)
    if big.is_big() && !small.is_big() {
        return 1
    }
    return 0
}

def fields(): i64 {
    let p = Prompt::new(1)
    p.set_level(7)
    Prompt::set_total(5)
    return p.level() * 10 + Prompt::total()
}

def failing(): i64 {
    let p = Prompt::new(1)
    return p.fail()
}
"#;

/// The host's modules: `game`, and the same class as module
/// `game.Prompt`, the shape a host whose classes sit in modules of
/// their own answers with.
fn resolve(path: &str) -> Option<HostModule> {
    match path {
        "game" => Some(game()),
        "game.Prompt" => Some(HostModule {
            name: "game.Prompt".into(),
            ..game()
        }),
        _ => None,
    }
}

fn compile(src: &str) -> Result<TieredRuntime, String> {
    foreign::install(Box::new(Stand));
    let mut rt = TieredRuntime::new(TieredConfig::development()).expect("runtime should start");
    rt.add_host_modules(resolve).expect("host modules");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(src, "<host_modules>")
        .expect("parse");
    rt.compile_typed_program(program)
        .map_err(|e| e.to_string())?;
    Ok(rt)
}

#[test]
fn a_host_module_is_imported_and_called_typed() {
    let rt = compile(SRC).expect("compile");
    let i64_of = |f: &str| rt.call::<i64>(f, &[]).map_err(|e| e.to_string());
    assert_eq!(i64_of("ask"), Ok(1));
    let said = STATE.lock().unwrap().said.clone();
    assert_eq!(said, ["answer to the meaning", "hello, world"]);
    // A ZynML handler answers the program's effect by calling the host.
    assert_eq!(i64_of("gate"), Ok(1));
    let said = STATE.lock().unwrap().said.clone();
    assert_eq!(said.last().map(String::as_str), Some("answer to why"));
    assert_eq!(i64_of("objects"), Ok(717));
    assert_eq!(i64_of("chained"), Ok(4));
    assert_eq!(
        rt.call::<f64>("floats", &[]).map_err(|e| e.to_string()),
        Ok(2.0)
    );
    assert_eq!(i64_of("bools"), Ok(1));
    assert_eq!(i64_of("fields"), Ok(75));
    assert!(foreign::take_error().is_none());
    assert_eq!(i64_of("failing"), Ok(0));
    let error = foreign::take_error().expect("the host's error is left pending");
    assert_eq!(
        (error.kind, error.message.as_str()),
        ("RuntimeError", "the host refused")
    );
}

/// Each spelling of the import names the module by its whole path,
/// longest first.
#[test]
fn every_import_form_reaches_the_module() {
    for import in [
        "import game",
        "import game.Prompt",
        "from game import Prompt",
        "from game.Prompt import Prompt",
    ] {
        let src = format!("{import}\ndef run(): i64 {{ return Prompt::new(9).count() }}\n");
        let rt = compile(&src).unwrap_or_else(|e| panic!("{import}: {e}"));
        assert_eq!(
            rt.call::<i64>("run", &[]).map_err(|e| e.to_string()),
            Ok(9),
            "{import}"
        );
    }
}

#[test]
fn a_member_the_module_lacks_is_refused_by_name() {
    let Err(error) = compile("from game import Nope\ndef run(): i64 { return 1 }\n") else {
        panic!("an import of a missing member compiled");
    };
    assert!(
        error.contains("Nope") && error.contains("Prompt"),
        "the error names the member and what the module has: {error}"
    );
}
