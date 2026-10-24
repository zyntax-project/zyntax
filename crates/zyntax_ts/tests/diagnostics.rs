//! Shared Ariadne formatting and structured byte positions agree with source.

use std::io::Write;
use std::process::Command;
use zyntax_ts::ErrorKind;

#[test]
fn type_errors_render_native_type_names_at_unicode_byte_offsets() {
    let source = "// 🦀\nconst π: boolean = 3;\n";
    let error = zyntax_ts::parse_program(source, "unicode.ts", Default::default()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Type);
    assert_eq!(error.start, source.find('π').unwrap());
    let rendered = error.render("unicode.ts", source, false);
    assert!(rendered.contains("unicode.ts:2:7"), "{rendered}");
    assert!(rendered.contains("const π: boolean = 3"), "{rendered}");
    assert!(
        rendered.contains("type `number` is not assignable to `boolean`"),
        "{rendered}"
    );
    assert!(!rendered.contains('\u{1b}'));
}

#[test]
fn syntax_and_duplicate_declarations_keep_parser_labels() {
    let source = "// preceding line\nconst value: number = ;\n";
    let error = zyntax_ts::parse_program(source, "syntax.ts", Default::default()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Syntax);
    assert_eq!(error.start, source.find(';').unwrap());
    let rendered = error.render("syntax.ts", source, false);
    assert!(rendered.contains("syntax.ts:2:"), "{rendered}");
    assert!(rendered.contains("const value: number = ;"), "{rendered}");

    let source = "let value = 1;\nlet value = 2;\n";
    let error = zyntax_ts::parse_program(source, "duplicate.ts", Default::default()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Type);
    let rendered = error.render("duplicate.ts", source, false);
    assert!(
        rendered.contains("has already been declared here"),
        "{rendered}"
    );
    assert!(
        rendered.contains("It can not be redeclared here"),
        "{rendered}"
    );
    assert!(rendered.contains("let value = 1"), "{rendered}");
    assert!(rendered.contains("let value = 2"), "{rendered}");
}

#[test]
fn eof_errors_render_without_losing_the_filename() {
    let source = "function broken(";
    let error = zyntax_ts::parse_program(source, "eof.ts", Default::default()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Syntax);
    let rendered = error.render("eof.ts", source, false);
    assert!(rendered.contains("eof.ts:1:"), "{rendered}");
    assert!(rendered.contains("function broken("), "{rendered}");
}

#[test]
fn cli_colors_and_json_follow_the_shared_diagnostic_controls() {
    let source = "// preceding line\nconst value: boolean = 3;\n";
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(source.as_bytes()).unwrap();
    let run = |json: bool, no_color: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_zynode"));
        command.arg("check").env("CLICOLOR_FORCE", "1");
        if json {
            command.arg("--diagnostics=json");
        }
        if no_color {
            command.env("NO_COLOR", "1");
        } else {
            command.env_remove("NO_COLOR");
        }
        let output = command.arg(file.path()).output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        String::from_utf8(output.stderr).unwrap()
    };
    assert!(run(false, false).contains('\u{1b}'));
    assert!(!run(false, true).contains('\u{1b}'));
    let json = run(true, false);
    assert!(!json.contains('\u{1b}'));
    let diagnostic: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(diagnostic["kind"], "Type");
    assert_eq!(diagnostic["start"], source.find("value").unwrap());
}

#[test]
fn every_file_is_checked_under_strict_mode_rules() {
    for (source, kind) in [
        ("function yield() {}\n", ErrorKind::Type),
        ("01;\n", ErrorKind::Syntax),
        ("08;\n", ErrorKind::Syntax),
    ] {
        let error =
            zyntax_ts::parse_program(source, "strict.ts", Default::default()).expect_err(source);
        assert_eq!(error.kind, kind, "{source}");
    }
    let valid =
        "const a = 0o17 + 0b1 + 1_0 + 0.5;\nconst o = { yield: 1 };\nconsole.log(o.yield, a);\n";
    zyntax_ts::parse_program(valid, "valid.ts", Default::default()).unwrap();
}

#[test]
fn declaration_files_reject_implementations_and_statements() {
    for source in [
        "function f() {}\n",
        "declare function f() {}\n",
        "declare class C { *generator(): any; }\n",
        "for (;;) {}\n",
        "return;\n",
        "module \"Foo\" {}\n",
    ] {
        let error =
            zyntax_ts::parse_program(source, "entry.d.ts", Default::default()).expect_err(source);
        assert_eq!(error.kind, ErrorKind::Type, "{source}");
    }
    // Declaration syntax is checked as ambient, not rejected as executable code.
    let error = zyntax_ts::parse_program(
        "export const a: number | null;\n",
        "entry.d.ts",
        Default::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Unsupported);
    // The same source is an ordinary file when its name is not a declaration file.
    let error = zyntax_ts::parse_program("function f() {}\n", "entry.ts", Default::default());
    assert!(error.is_ok());
}

#[test]
fn checker_grammar_errors_are_type_errors_and_parse_failures_are_syntax_errors() {
    for (source, kind) in [
        ("class C { public public x = 1; }\n", ErrorKind::Type),
        ("for (let a = 1, b = 2 in {}) {}\n", ErrorKind::Type),
        (
            "function* g() {}\nfunction h() { yield 1; }\n",
            ErrorKind::Type,
        ),
        ("interface any {}\n", ErrorKind::Type),
        ("const value: number = ;\n", ErrorKind::Syntax),
        ("let s = \"unterminated;\n", ErrorKind::Syntax),
    ] {
        let error =
            zyntax_ts::parse_program(source, "grammar.ts", Default::default()).expect_err(source);
        assert_eq!(error.kind, kind, "{source}");
    }
}

#[test]
fn forms_the_reference_checks_after_parsing_report_its_category_and_message() {
    for (source, kind, message) in [
        (
            "declare function dec(t: any): any;\n@dec\nenum E { A }\n",
            ErrorKind::Type,
            "Decorators are not valid here.",
        ),
        (
            "var ;\n",
            ErrorKind::Type,
            "Variable declaration list cannot be empty.",
        ),
        (
            "let a = [1];\nfor (var of a) {}\n",
            ErrorKind::Type,
            "Variable declaration list cannot be empty.",
        ),
        (
            "var a, b,;\n",
            ErrorKind::Type,
            "Trailing comma not allowed.",
        ),
        (
            "let a = 1;\nlet o = { a? };\n",
            ErrorKind::Type,
            "An object member cannot be declared optional.",
        ),
        (
            "class A {}\nclass C extends A implements {}\n",
            ErrorKind::Type,
            "'implements' list cannot be empty.",
        ),
        (
            "class C extends {}\n",
            ErrorKind::Type,
            "'extends' list cannot be empty.",
        ),
        (
            "interface I {\n  []: number\n}\n",
            ErrorKind::Type,
            "An index signature must have exactly one parameter.",
        ),
        (
            "enum E {\n  #x\n}\n",
            ErrorKind::Type,
            "An enum member cannot be named with a private identifier.",
        ),
        ("let x = 1;\nx++ = 4;\n", ErrorKind::Syntax, "';' expected."),
        (
            "let x = 1;\n++-x;\n",
            ErrorKind::Syntax,
            "Expression expected.",
        ),
    ] {
        let error =
            zyntax_ts::parse_program(source, "grammar.ts", Default::default()).expect_err(source);
        assert_eq!(
            (error.kind, error.message.as_str()),
            (kind, message),
            "{source}"
        );
    }
    // The same messages from forms the reference also rejects while parsing stay syntax errors.
    for source in [
        "const {a?} = {a: 1};\n",
        "let f = ({a?}) => 1;\n",
        "var 1 = 2;\n",
    ] {
        let error =
            zyntax_ts::parse_program(source, "grammar.ts", Default::default()).expect_err(source);
        assert_eq!(error.kind, ErrorKind::Syntax, "{source}: {}", error.message);
    }
    // Invalid targets the reference parses as assignments are checker errors.
    for source in [
        "let x = 1;\n1 = x;\n",
        "let v: number = 1;\nfor (v++ of []) {}\n",
    ] {
        let error =
            zyntax_ts::parse_program(source, "grammar.ts", Default::default()).expect_err(source);
        assert_eq!(error.kind, ErrorKind::Type, "{source}: {}", error.message);
    }
}

#[test]
fn delete_needs_a_property_reference() {
    let error = zyntax_ts::parse_program("delete `abc${0}`;\n", "delete.ts", Default::default())
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Type, "{error}");
    assert_eq!((error.start, error.end), (7, 16));
    let error = zyntax_ts::parse_program(
        "let o = { a: 1 };\ndelete o.a;\n",
        "delete.ts",
        Default::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Unsupported, "{error}");
}

#[test]
fn delete_checks_its_operand_first() {
    let error =
        zyntax_ts::parse_program("delete a[0];\n", "delete.ts", Default::default()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Type, "{error}");
}
