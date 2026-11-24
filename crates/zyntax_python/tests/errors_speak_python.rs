//! What a program is told when zypy cannot compile it: the Python
//! construct by its Python name, what zypy compiles in its place, and
//! what to write instead.

fn render(source: &str) -> String {
    let err = zyntax_python::parse_program(source).expect_err("is not compiled");
    err.render("prog.py", source, false)
}

#[test]
fn a_method_zypy_lacks_names_the_type_and_the_methods_it_has() {
    let shown = render("data = bytes([104, 105])\nprint(data.find(b\"i\"))\n");
    assert!(
        shown.contains("the `bytes` method `find()`"),
        "got: {shown}"
    );
    assert!(shown.contains("method call"), "got: {shown}");
    assert!(shown.contains("`decode`"), "got: {shown}");
    assert!(shown.contains("decode them to `str` first"), "got: {shown}");
    assert!(!shown.contains("this frontend"), "got: {shown}");
    assert!(!shown.contains("argument(s)"), "got: {shown}");
}

#[test]
fn a_method_with_an_unsupported_argument_count_says_so_in_words() {
    let shown = render("f = open(\"x\", \"w\")\nf.write(\"a\", \"b\")\n");
    assert!(
        shown.contains("`file.write()` with 2 arguments"),
        "got: {shown}"
    );
    let shown = render("f = open(\"x\", \"w\")\nf.read(1, 2)\n");
    assert!(shown.contains("with 2 arguments"), "got: {shown}");
}

#[test]
fn a_keyword_argument_it_cannot_place_says_how_to_call_instead() {
    let shown = render("def f(x):\n    return x\ng = f\nprint(g(1, key=2))\n");
    assert!(shown.contains("keyword arguments"), "got: {shown}");
    assert!(
        shown.contains("pass the arguments by position"),
        "got: {shown}"
    );
    assert!(shown.contains("call"), "got: {shown}");
}

#[test]
fn a_with_of_a_class_without_the_protocol_names_the_protocol() {
    let shown = render("class A:\n    pass\nwith A() as a:\n    pass\n");
    assert!(
        shown.contains("__enter__") && shown.contains("__exit__"),
        "got: {shown}"
    );
    assert!(shown.contains("`try`/`finally`"), "got: {shown}");
}

#[test]
fn a_statement_is_labelled_as_python_names_it() {
    let shown = render("def f():\n    pass\n\ntype T = int\n");
    assert!(shown.contains("type alias"), "got: {shown}");
}
