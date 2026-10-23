//! Library types the frontend does not model are explicit gaps; undeclared names are errors.

#[test]
fn library_types_are_gaps_and_unknown_names_are_errors() {
    for (source, kind) in [
        ("let o: Object;", zyntax_ts::ErrorKind::Unsupported),
        ("function f(s: String): void {}", zyntax_ts::ErrorKind::Unsupported),
        ("let e: HTMLElement;", zyntax_ts::ErrorKind::Unsupported),
        ("let x: Missing;", zyntax_ts::ErrorKind::Type),
    ] {
        let error = zyntax_ts::parse_program(source, "library.ts", Default::default()).unwrap_err();
        assert_eq!(error.kind, kind, "{source}: {error}");
    }
    let shadowed = "type Date = { x: number };\nconst d: Date = { x: 1 };\nconsole.log(d.x);";
    zyntax_ts::parse_program(shadowed, "library.ts", Default::default()).unwrap();
}
