//! Interfaces extending record interfaces, and classes implementing them.

#[test]
fn extended_records_keep_unmodelled_forms_explicit() {
    for (source, message) in [
        (
            "interface P { x: number; y: number }\ninterface Q extends P { a: boolean }\nfunction f(p: P): number { return p.x; }\nconst q: Q = { a: true, x: 1, y: 2 };\nf(q);",
            "different field offsets",
        ),
        (
            "class C { x = 1; }\ninterface I extends C { y: number }",
            "extending classes",
        ),
        (
            "interface G<T> { x: number }\ninterface I extends G<number> { y: number }",
            "generic",
        ),
        (
            "class C implements I { f(): void {} }\ninterface I { f(): void }",
            "interface method",
        ),
        (
            "declare class C { x: number }\ninterface C { y: number }\nclass D implements C { y = 1; }",
            "declaration merging",
        ),
        (
            "interface N { v: number; next: N }\nconst a = 1;",
            "recursive or unresolved",
        ),
    ] {
        let error =
            zyntax_ts::parse_program(source, "interfaces.ts", Default::default()).unwrap_err();
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
fn nocheck_files_that_need_a_type_error_are_gaps() {
    let suppressed =
        "// @ts-nocheck\ninterface A { q: number }\nclass B implements A { q: string = 'x'; }";
    let error = zyntax_ts::parse_program(suppressed, "nocheck.ts", Default::default()).unwrap_err();
    assert_eq!(error.kind, zyntax_ts::ErrorKind::Unsupported);
    zyntax_ts::parse_program(
        "// @ts-nocheck\nconst a = 1;",
        "nocheck.ts",
        Default::default(),
    )
    .unwrap();
}

#[test]
fn unresolved_declarations_report_the_declaration_that_fails() {
    let source = "type S = \"a\" | \"b\";\ntype T = S[] | S;\nconst t = 1;";
    let error = zyntax_ts::parse_program(source, "aliases.ts", Default::default()).unwrap_err();
    assert_eq!(error.kind, zyntax_ts::ErrorKind::Unsupported, "{error}");
    assert!(
        error.message.starts_with("this type representation"),
        "{}",
        error.message
    );
    let missing = "type T = S[];\nconst t = 1;";
    let error = zyntax_ts::parse_program(missing, "aliases.ts", Default::default()).unwrap_err();
    assert_eq!(error.kind, zyntax_ts::ErrorKind::Type, "{error}");
}
