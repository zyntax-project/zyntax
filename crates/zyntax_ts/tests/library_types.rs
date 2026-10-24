//! Library types the frontend does not model are explicit gaps; undeclared names are errors.

#[test]
fn library_types_are_gaps_and_unknown_names_are_errors() {
    for (source, kind) in [
        ("let o: Object;", zyntax_ts::ErrorKind::Unsupported),
        (
            "function f(s: String): void {}",
            zyntax_ts::ErrorKind::Unsupported,
        ),
        ("let e: HTMLElement;", zyntax_ts::ErrorKind::Unsupported),
        ("let x: Missing;", zyntax_ts::ErrorKind::Type),
    ] {
        let error = zyntax_ts::parse_program(source, "library.ts", Default::default()).unwrap_err();
        assert_eq!(error.kind, kind, "{source}: {error}");
    }
    let shadowed = "type Date = { x: number };\nconst d: Date = { x: 1 };\nconsole.log(d.x);";
    zyntax_ts::parse_program(shadowed, "library.ts", Default::default()).unwrap();
}

#[test]
fn library_values_and_unmodelled_forms_are_gaps() {
    for (source, message) in [
        ("console.log(JSON.stringify(1));", "library value `JSON`"),
        ("const d = Date.UTC(2020, 1);", "library value `Date`"),
        ("const s = Symbol();", "library value `Symbol`"),
        ("const a = [1, 2] as const;", "const assertions"),
        ("const s = 'a';\nconsole.log(+s);", "numeric conversion"),
    ] {
        let error = zyntax_ts::parse_program(source, "library.ts", Default::default()).unwrap_err();
        assert_eq!(
            error.kind,
            zyntax_ts::ErrorKind::Unsupported,
            "{source}: {error}"
        );
        assert!(
            error.message.contains(message),
            "{source}: {}",
            error.message
        );
    }
    // Every class in a declaration file is ambient.
    let error = zyntax_ts::parse_program(
        "export class C {\n  start: number;\n}\n",
        "library.d.ts",
        Default::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, zyntax_ts::ErrorKind::Unsupported, "{error}");
    // A possibly undefined operand is still a type error.
    let error = zyntax_ts::parse_program(
        "let n: number | undefined = undefined;\nconsole.log(-n);",
        "library.ts",
        Default::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, zyntax_ts::ErrorKind::Type, "{error}");
}

#[test]
fn ambient_functions_are_typed_and_calls_are_gaps() {
    let typed = "declare function f(x: number): string;\nconst a = 1;\nconsole.log(a);";
    zyntax_ts::parse_program(typed, "ambient.ts", Default::default()).unwrap();
    for (source, kind) in [
        (
            "declare function f(x: number): string;\nconst s = f(1);",
            zyntax_ts::ErrorKind::Unsupported,
        ),
        (
            "declare function f(x: number): string;\nf(\"x\");",
            zyntax_ts::ErrorKind::Type,
        ),
        ("declare function f(x: number);", zyntax_ts::ErrorKind::Type),
    ] {
        let error = zyntax_ts::parse_program(source, "ambient.ts", Default::default()).unwrap_err();
        assert_eq!(error.kind, kind, "{source}: {error}");
    }
}

#[test]
fn ambient_classes_are_types_without_values() {
    let kind = |source: &str| {
        zyntax_ts::parse_program(source, "ambient.ts", Default::default())
            .map(|_| ())
            .map_err(|error| error.kind)
    };
    use zyntax_ts::ErrorKind::{Type, Unsupported};
    assert_eq!(
        kind(
            "declare class C { x: number; constructor(n: number); m(): number; }\nfunction f(c: C): number { return c.x; }"
        ),
        Ok(())
    );
    assert_eq!(
        kind("declare class C {}\nconst c = new C();"),
        Err(Unsupported)
    );
    assert_eq!(
        kind("declare class C { m(): number; }\nfunction f(c: C): number { return c.m(); }"),
        Err(Unsupported)
    );
    assert_eq!(kind("declare class C { m(); }"), Err(Type));
    assert_eq!(
        kind("declare class B {}\nclass D extends B {}"),
        Err(Unsupported)
    );
}

#[test]
fn class_objects_as_values_are_gaps() {
    for source in [
        "class C {}\nconst x = C;",
        "class C {}\nconst c = new C();\nconsole.log(c instanceof C);",
    ] {
        let error = zyntax_ts::parse_program(source, "classes.ts", Default::default()).unwrap_err();
        assert_eq!(
            error.kind,
            zyntax_ts::ErrorKind::Unsupported,
            "{source}: {error}"
        );
    }
}
