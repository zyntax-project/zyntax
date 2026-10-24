//! Single inheritance among a module's classes; forms it does not model are explicit gaps.

#[test]
fn unmodelled_inheritance_forms_are_gaps() {
    for (source, message) in [
        (
            "class B { x = 1; }\nclass D extends B { x = 2; }",
            "redeclared inherited fields",
        ),
        (
            "function F() {}\nclass D extends F {}",
            "extending values that are not class declarations",
        ),
        ("class D extends null {}", "classes extending null"),
        (
            "class B { constructor(public n: number) {} }\nclass D extends B {}",
            "implicit constructors forwarding",
        ),
        (
            "class B { m(a: number): number { return a; } }\nclass D extends B { m(): number { return 1; } }\nfunction f(b: B): number { return b.m(1); }",
            "overrides with a different signature",
        ),
        (
            "class B { x = 1; }\nclass D extends B {}\nconst b: B = { x: 2 };",
            "structural values of classes with subclasses",
        ),
        (
            "class B {}\nclass D extends B { constructor() { if (true) { super(); } } }",
            "super calls inside nested statements",
        ),
        ("class D extends Map {}", "extending library class"),
    ] {
        let error = zyntax_ts::parse_program(source, "inheritance.ts", Default::default())
            .expect_err(source);
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
}

#[test]
fn inheritance_errors_follow_typescript() {
    for source in [
        "interface I { x: number }\nclass D extends I {}",
        "class B { x = 1; }\nclass D extends B { x(): number { return 1; } }",
        "class B { m(): number { return 1; } }\nclass D extends B { private m(): number { return 2; } }",
        "class B {}\nclass D extends B { m(): number { return super.x; } }",
        "class C { m(): void { super.m(); } }",
        "class D extends Missing {}",
        "class D extends 42 {}",
        "class B { private constructor() {} }\nclass D extends B {}",
        "class B { protected constructor() {} }\nconst b = new B();",
    ] {
        let error = zyntax_ts::parse_program(source, "inheritance.ts", Default::default())
            .expect_err(source);
        assert_eq!(error.kind, zyntax_ts::ErrorKind::Type, "{source}: {error}");
    }
}

#[test]
fn protected_constructors_are_reachable_from_subclasses() {
    let source = "class B { protected constructor() {} }\nclass D extends B { static make(): number { new B(); return 1; } }\nconsole.log(D.make());";
    zyntax_ts::parse_program(source, "inheritance.ts", Default::default()).unwrap();
}
