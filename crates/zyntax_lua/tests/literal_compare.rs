use zyntax_typed_ast::{InternedString, TypedDeclaration};

#[test]
fn exactly_representable_literals_compare_directly_with_floats() {
    let p = zyntax_lua::parse_program(
        r#"
local function small(x)
  return x == 1 and 0 < x and x <= 9007199254740992 and 1 >= x
end
local function large(x)
  return x == 9007199254740993 or 9007199254740993 < x
end
print(small(1.0), large(1.0))
"#,
        "compare.lua",
    )
    .unwrap();
    let helpers: Vec<_> = ["zl_eq_if", "zl_lt_if", "zl_lt_fi", "zl_le_if", "zl_le_fi"]
        .iter()
        .map(|name| format!("{:?}", InternedString::new_global(name)))
        .collect();
    let mut small = false;
    let mut large = false;
    for d in &p.declarations {
        let TypedDeclaration::Function(f) = &d.node else {
            continue;
        };
        let name = f.name.resolve_global().unwrap_or_default();
        if name.ends_with("$fn") {
            continue;
        }
        if name.starts_with("lua$small$") {
            small = true;
            let body = format!("{:?}", f.body);
            assert!(
                !helpers.iter().any(|h| body.contains(h)),
                "{name} still calls an exact mixed comparison helper"
            );
        }
        if name.starts_with("lua$large$") {
            large = true;
            let body = format!("{:?}", f.body);
            assert!(
                helpers.iter().any(|h| body.contains(h)),
                "{name} must retain exact comparison for rounded integers"
            );
        }
    }
    assert!(small && large, "both comparison functions must be checked");
}
