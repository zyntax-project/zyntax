//! What a module exports, read from its chunk's types without running
//! it: the table the chunk returns, its functions with their
//! parameters, the tables it holds, a class's methods and the fields of
//! its instances, with the metatable's own fields left out; and the
//! types its LuaLS annotations declare.

use zyntax_lua::{
    DeclaredClass, Error, Exported, ExportedFunction, Exports, LuaType, Returned, Signature,
    exports, is_metafield,
};

fn returned(ty: LuaType, name: Option<&str>) -> Returned {
    Returned {
        ty,
        name: name.map(str::to_owned),
    }
}

fn function(params: &[&str], variadic: bool) -> Exported {
    Exported::Function(ExportedFunction {
        params: params.iter().map(|p| p.to_string()).collect(),
        variadic,
        method: params.first() == Some(&"self"),
        signature: None,
    })
}

/// The names and values of the fields of the table `value` is.
fn fields<'a>(e: &'a Exports, value: &Exported) -> Vec<(&'a str, &'a Exported)> {
    e.table(value)
        .unwrap_or_else(|| panic!("{value:?} is no table in {e:?}"))
        .fields
        .iter()
        .map(|(name, v)| (name.as_str(), v))
        .collect()
}

#[test]
fn a_module_table_with_a_class() {
    let e = exports(
        r#"
local Point = {}
Point.__index = Point
function Point.new(x, y) return setmetatable({x = x, y = y}, Point) end
function Point:len() return math.sqrt(self.x * self.x + self.y * self.y) end

local M = { origin = 0 }
M.Point = Point
function M.add(a, b) return a + b end
function M.sum(...) local s = 0 for _, v in ipairs({...}) do s = s + v end return s end
M.version = "1.0"
return M
"#,
    )
    .unwrap();
    let module = fields(&e, &e.value);
    let names: Vec<&str> = module.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, ["origin", "Point", "add", "sum", "version"]);
    assert_eq!(module[0].1, &Exported::Value(None));
    assert_eq!(module[2].1, &function(&["a", "b"], false));
    assert_eq!(module[3].1, &function(&[], true));
    assert_eq!(module[4].1, &Exported::Value(None));
    // `__index` is the metatable's own, and `len` is a method.
    let point = e.table(module[1].1).unwrap();
    assert_eq!(
        fields(&e, module[1].1),
        [
            ("new", &function(&["x", "y"], false)),
            ("len", &function(&["self"], false)),
        ]
    );
    assert!(matches!(
        &point.fields[1].1,
        Exported::Function(ExportedFunction { method: true, .. })
    ));
    assert!(!point.open);
    // What `new` makes has Point as its metatable.
    let [instance] = point.instances[..] else {
        panic!("{point:?}");
    };
    let names: Vec<&str> = e.tables[instance]
        .fields
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(names, ["x", "y"]);
}

#[test]
fn a_table_constructor_naming_locals() {
    let e = exports(
        r#"
local Scale = {}
local helper = require("helper")
function Scale.run(device, queue) return helper(device, queue) end
local function version() return 1 end
return { Scale = Scale, version = version, helper = helper }
"#,
    )
    .unwrap();
    let module = fields(&e, &e.value);
    let names: Vec<&str> = module.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, ["Scale", "helper", "version"]);
    assert_eq!(
        fields(&e, module[0].1),
        [("run", &function(&["device", "queue"], false))]
    );
    // What `require` gives is not followed.
    assert_eq!(module[1].1, &Exported::Value(None));
    assert_eq!(module[2].1, &function(&[], false));
}

#[test]
fn stores_under_keys_not_known_leave_the_table_open() {
    let e = exports(
        r#"
local M = {}
for _, n in ipairs({"a", "b"}) do M[n] = function() return n end end
function M.fixed(x) return x end
return M
"#,
    )
    .unwrap();
    let table = e.table(&e.value).unwrap();
    assert!(table.open);
    let names: Vec<&str> = table.fields.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["fixed"]);
}

#[test]
fn a_chunk_returning_no_table() {
    for source in [
        "local x = 1",
        "return 42",
        "if os.getenv('X') then return {} end return { f = print }",
    ] {
        let e = exports(source).unwrap();
        assert!(e.table(&e.value).is_none(), "{source}: {e:?}");
    }
}

#[test]
fn a_syntax_error_is_an_error() {
    assert!(exports("return {").is_err());
}

#[test]
fn metafields_are_named_by_two_underscores() {
    assert!(is_metafield("__index"));
    assert!(is_metafield("__call"));
    assert!(!is_metafield("_private"));
    assert!(!is_metafield("new"));
}

#[test]
fn annotations_declare_types() {
    let e = exports(
        r#"
---@class Counter
---@field n integer
---@field label string the name
local Counter = {}
Counter.__index = Counter
---@type integer
Counter.LIMIT = 10

---@param start integer
---@return Counter
function Counter.new(start)
  return setmetatable({ n = start, label = "count" }, Counter)
end

---@param by integer how far
---@return integer
function Counter:bump(by)
  self.n = self.n + by
  return self.n
end

---@param f fun(i: integer): number
---@param ... string
function Counter.each(f, ...) end

---@return integer count
---@return string label the name
function Counter:state() return self.n, self.label end

return { Counter = Counter }
"#,
    )
    .unwrap();
    let module = fields(&e, &e.value);
    let counter = e.table(module[0].1).unwrap();
    assert_eq!(
        counter.class,
        Some(DeclaredClass {
            name: "Counter".into(),
            fields: vec![
                ("n".into(), LuaType::Integer),
                ("label".into(), LuaType::String)
            ],
        })
    );
    let names: Vec<&str> = counter.fields.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["LIMIT", "new", "bump", "each", "state"]);
    assert_eq!(counter.fields[0].1, Exported::Value(Some(LuaType::Integer)));
    let signature = |i: usize| match &counter.fields[i].1 {
        Exported::Function(f) => f.signature.clone().expect("annotated"),
        other => panic!("{other:?}"),
    };
    assert_eq!(
        signature(1),
        Signature {
            params: vec![LuaType::Integer],
            variadic: None,
            returns: vec![returned(LuaType::Named("Counter".into()), None)],
        }
    );
    // `self` has no annotation of its own.
    assert_eq!(
        signature(2),
        Signature {
            params: vec![LuaType::Any, LuaType::Integer],
            variadic: None,
            returns: vec![returned(LuaType::Integer, None)],
        }
    );
    assert_eq!(
        signature(3),
        Signature {
            params: vec![LuaType::Fun {
                params: vec![LuaType::Integer],
                returns: vec![LuaType::Number],
            }],
            variadic: Some(LuaType::String),
            returns: vec![],
        }
    );
    // Several results, each with its name.
    assert_eq!(
        signature(4).returns,
        [
            returned(LuaType::Integer, Some("count")),
            returned(LuaType::String, Some("label"))
        ]
    );
}

#[test]
fn an_annotation_that_names_no_parameter_is_an_error() {
    let source = "local M = {}\n---@param count integer\nfunction M.f(n) return n end\nreturn M\n";
    match exports(source) {
        Err(e @ Error::Annotation { .. }) => {
            assert_eq!(
                e.one_line("m.lua", source),
                "m.lua:2: @param `count` names no parameter of the function"
            );
        }
        other => panic!("{other:?}"),
    }
}
