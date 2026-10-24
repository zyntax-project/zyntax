//! Instance `get` and `set` accessors: calls on reads and writes, dispatched like methods.

fn kind(source: &str) -> Result<(), zyntax_ts::ErrorKind> {
    zyntax_ts::parse_program(source, "accessors.ts", Default::default())
        .map(|_| ())
        .map_err(|error| error.kind)
}

#[test]
fn accessors_follow_typescript() {
    use zyntax_ts::ErrorKind::{Type, Unsupported};
    for (source, expected) in [
        (
            "class A { v = 1; get x(): number { return this.v; } set x(n: number) { this.v = n; } }\nconst a = new A();\na.x = 3;\nconsole.log(a.x);",
            Ok(()),
        ),
        (
            "class A { get x(): number { return 1; } }\nnew A().x = 2;",
            Err(Type),
        ),
        ("class A { get x() { } }", Err(Type)),
        (
            "class A { x = 1; get x(): number { return 1; } }",
            Err(Type),
        ),
        (
            "class A { v = 1; get x(): number { return this.v; } set x(n: number) { this.v = n; } }\nconst a = new A();\na.x += 1;",
            Err(Unsupported),
        ),
        (
            "class A { static get x(): number { return 1; } }",
            Err(Unsupported),
        ),
        (
            "interface I { x: number }\nclass A { get x(): number { return 1; } }\nconst i: I = new A();",
            Err(Unsupported),
        ),
    ] {
        assert_eq!(kind(source), expected, "{source}");
    }
}
