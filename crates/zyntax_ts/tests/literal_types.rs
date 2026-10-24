//! String and number literal types: a matching literal is assignable, other values are not.

fn kind(source: &str) -> Result<(), zyntax_ts::ErrorKind> {
    zyntax_ts::parse_program(source, "literals.ts", Default::default())
        .map(|_| ())
        .map_err(|error| error.kind)
}

#[test]
fn literal_values_are_checked_like_typescript() {
    use zyntax_ts::ErrorKind::{Type, Unsupported};
    for (source, expected) in [
        (
            "type D = \"a\" | \"b\";\nconst x: D = \"a\";\nconst one = \"b\";\nconst y: D = one;",
            Ok(()),
        ),
        (
            "type D = 1 | 2;\nfunction f(d: D): number { return d * 2; }\nf(2);",
            Ok(()),
        ),
        ("type D = \"a\" | \"b\";\nconst x: D = \"c\";", Err(Type)),
        (
            "type D = \"a\" | \"b\";\nlet s = \"a\";\nconst x: D = s;",
            Err(Type),
        ),
        (
            "type D = 1 | 2;\nfunction f(n: number): D { return n; }",
            Err(Type),
        ),
        (
            "type D = \"a\" | \"b\";\nfunction f(d: D): boolean { return d === \"c\"; }",
            Err(Type),
        ),
        ("let x: \"a\" | number = 1;", Err(Unsupported)),
        (
            "interface S { kind: \"circle\"; r: number }",
            Err(Unsupported),
        ),
        ("let flag: true = true;", Err(Unsupported)),
    ] {
        assert_eq!(kind(source), expected, "{source}");
    }
}

#[test]
fn literal_switches_are_exhaustive() {
    let source = "type D = \"n\" | \"s\";\nfunction f(d: D): number {\n  switch (d) {\n    case \"n\":\n      return 1;\n    case \"s\":\n      return 2;\n  }\n}\nconsole.log(f(\"s\"));";
    assert_eq!(kind(source), Ok(()));
}
