//! `#name` members: fields, methods and static fields private to the declaring class.

fn kind(source: &str) -> Result<(), zyntax_ts::ErrorKind> {
    zyntax_ts::parse_program(source, "private.ts", Default::default())
        .map(|_| ())
        .map_err(|error| error.kind)
}

#[test]
fn private_names_follow_typescript() {
    use zyntax_ts::ErrorKind::{Type, Unsupported};
    for (source, expected) in [
        (
            "class A { #x = 1; x = 2; #m(): number { return this.#x + this.x; } run(): number { return this.#m(); } }\nconsole.log(new A().run());",
            Ok(()),
        ),
        (
            "class A { static #n = 0; static next(): number { A.#n = A.#n + 1; return A.#n; } }\nconsole.log(A.next());",
            Ok(()),
        ),
        (
            "class A { #x: number; constructor() { this.#x = 1; } get(): number { return this.#x; } }",
            Ok(()),
        ),
        ("class A { #x: number; }", Err(Type)),
        ("class A { #x = 1; }\nconsole.log(new A().#x);", Err(Type)),
        (
            "class A { #x = 1; }\nclass B extends A { #x = 2; }",
            Err(Unsupported),
        ),
    ] {
        assert_eq!(kind(source), expected, "{source}");
    }
}
