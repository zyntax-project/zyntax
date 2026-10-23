//! String fields hold static strings; other string values are explicit gaps.

#[test]
fn string_fields_need_a_static_lifetime() {
    for source in [
        "class P { name: string; constructor(name: string) { this.name = name; } }",
        "class P { constructor(public name: string) {} }",
        "function f(s: string): void { const r: { s: string } = { s: s }; }",
        "type R = { s: string };\nfunction f(r: R, x: number): void { r.s = 'n' + x; }",
        "class P { name: string = 'a'; }\nfunction f(p: P, s: string): void { p.name = s; }",
    ] {
        let error = zyntax_ts::parse_program(source, "fields.ts", Default::default()).unwrap_err();
        assert_eq!(
            error.kind,
            zyntax_ts::ErrorKind::Unsupported,
            "{source}: {error}"
        );
        assert!(
            error.message.contains("static lifetime"),
            "{source}: {}",
            error.message
        );
    }
    let accepted = "type R = { s: string };\nconst r: R = { s: `k${2}` };\nconst copy: R = { s: r.s };\nconst xs: string[] = [copy.s];\nconsole.log(xs[0]);";
    zyntax_ts::parse_program(accepted, "fields.ts", Default::default()).unwrap();
}
