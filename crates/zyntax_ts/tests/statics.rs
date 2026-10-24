//! Static class members: fields as module globals, static blocks, and `this` in static code.

fn kind(source: &str) -> Result<(), zyntax_ts::ErrorKind> {
    zyntax_ts::parse_program(source, "statics.ts", Default::default())
        .map(|_| ())
        .map_err(|error| error.kind)
}

#[test]
fn static_members_follow_typescript() {
    use zyntax_ts::ErrorKind::{Type, Unsupported};
    for (source, expected) in [
        (
            "class C { static a = 1; static b = this.a + 1; static f(): number { return this.b; } }\nconsole.log(C.f());",
            Ok(()),
        ),
        (
            "class C { static a = 1; }\nclass D extends C { static b = super.a + 1; static { D.b = D.b * 2; } }\nconsole.log(D.b);",
            Ok(()),
        ),
        ("console.log(C.x);\nclass C { static x = 1; }", Err(Type)),
        (
            "class B { private static p = 1; }\nclass D extends B { static q = B.p; }",
            Err(Type),
        ),
        ("class C { static readonly r = 1; }\nC.r = 2;", Err(Type)),
        (
            "class C { static { console.log(C.later); } static later = 1; }",
            Err(Type),
        ),
        ("class C { static f = 1; static f(): void {} }", Err(Type)),
        (
            "class B { static x = 1; }\nclass D extends B { static x = 'two'; }",
            Err(Type),
        ),
        (
            "class B { static x = 1; }\nclass D extends B {}\nD.x = 2;",
            Err(Unsupported),
        ),
        (
            "class B { static x = 1; static f(): number { return this.x; } }\nclass D extends B { static x = 2; }",
            Err(Unsupported),
        ),
    ] {
        assert_eq!(kind(source), expected, "{source}");
    }
}

#[test]
fn unused_bindings_need_no_annotation() {
    assert_eq!(kind("var a, b;\nlet c;\nconsole.log(1);"), Ok(()));
    assert_eq!(
        kind("let d;\nd = 1;"),
        Err(zyntax_ts::ErrorKind::Unsupported)
    );
}
