//! Python source reaches native code through the same pipeline as
//! everything else.
//!
//! The frontend hands the runtime a `TypedProgram`, the way the Lua
//! frontend does; nothing downstream knows it came from Python. So a
//! function that computes something is the whole test of the seam:
//! parsing, the rewrite to typed nodes, type inference over `Unknown`,
//! lowering, and the compiled result agreeing with arithmetic.

use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};

fn run(source: &str, entry: &str) -> ZyntaxValue {
    let program = zyntax_python::parse_program(source).expect("should lower");
    let mut rt = TieredRuntime::new(TieredConfig::default()).expect("runtime");
    zyntax_python::register_runtime(&mut rt).expect("runtime plugins");
    rt.compile_typed_program(program).expect("should compile");
    rt.call_raw(entry, &[]).expect("should run")
}

/// Recursion, comparison, arithmetic, an annotated signature.
#[test]
fn fib_of_twenty() {
    assert_eq!(
        run(
            r#"
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def main() -> int:
    return fib(20)
"#,
            "main",
        ),
        ZyntaxValue::Int(6765)
    );
}

/// Assignment is a binding the first time and an assignment after, a
/// `while` loop, and `+=`.
#[test]
fn a_loop_with_reassignment() {
    assert_eq!(
        run(
            r#"
def total(n: int) -> int:
    acc = 0
    i = 0
    while i < n:
        acc += i
        i = i + 1
    return acc

def main() -> int:
    return total(101)
"#,
            "main",
        ),
        ZyntaxValue::Int(5050)
    );
}

/// `for` over `range`, and `elif`.
#[test]
fn for_range_and_elif() {
    assert_eq!(
        run(
            r#"
def classify(n: int) -> int:
    if n < 0:
        return -1
    elif n == 0:
        return 0
    else:
        return 1

def main() -> int:
    s = 0
    for i in range(10):
        s += classify(i - 3)
    return s
"#,
            "main",
        ),
        // -1 for i in 0..3, 0 for i=3, +1 for i in 4..10: -3 + 0 + 6
        ZyntaxValue::Int(3)
    );
}

/// A chained comparison is a conjunction, and a conditional expression
/// is an expression.
#[test]
fn chained_comparison_and_ternary() {
    assert_eq!(
        run(
            r#"
def clamp(x: int) -> int:
    return x if 0 <= x <= 10 else 10

def main() -> int:
    return clamp(4) + clamp(40)
"#,
            "main",
        ),
        ZyntaxValue::Int(14)
    );
}

/// Something outside the subset is a message naming what and where,
/// not a crash and not a silent mis-compile.
#[test]
fn an_unsupported_form_is_named() {
    // Reached from the module body: a function nothing reaches may use
    // what the frontend lacks, as a Python that never runs it may.
    let err = zyntax_python::parse_program("async def f():\n    return 1\nf()\n")
        .expect_err("an async def is not in the subset yet");
    let msg = err.to_string();
    assert!(
        msg.contains("async def"),
        "should name the form, got: {msg}"
    );
    assert!(
        msg.contains("not supported"),
        "should say it is unsupported, got: {msg}"
    );
}

/// An error renders the way the compiler's own diagnostics do: against
/// the source, with the line and the offending text marked.
#[test]
fn an_error_renders_against_its_source() {
    let source = "x = 1\ny = x +\n";
    let err = zyntax_python::parse_program(source).expect_err("does not parse");
    let shown = err.render("prog.py", source, false);
    assert!(shown.contains("syntax error"), "got: {shown}");
    assert!(
        shown.contains("prog.py:2:"),
        "should locate line 2, got: {shown}"
    );
    assert!(
        shown.contains("y = x +"),
        "should show the line, got: {shown}"
    );

    let source = "def f():\n    pass\n\nwith f() as g:\n    pass\n";
    let err = zyntax_python::parse_program(source).expect_err("with is not in the subset");
    let shown = err.render("prog.py", source, false);
    assert!(shown.contains("not supported yet"), "got: {shown}");
    assert!(
        shown.contains("prog.py:4:"),
        "should locate line 4, got: {shown}"
    );
    assert!(err.module().is_none(), "the main file has no module name");
}

/// A program of two files carries both as source files, then the
/// prelude as a file of its own, and what came from the second names
/// it, so a diagnostic about it quotes the right text.
#[test]
fn a_module_span_names_its_file() {
    let helper = "def f(x):\n    return x * 2\n";
    let resolve = |name: &str| (name == "helper").then(|| helper.to_string());
    let program = zyntax_python::parse_program_with(
        "import helper\nprint(helper.f(2))\n",
        "main.py",
        &resolve,
    )
    .expect("links");
    let names: Vec<&str> = program
        .source_files
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(names, ["main.py", "helper", zyntax_python::PRELUDE]);
    let f = program
        .declarations
        .iter()
        .find(|d| match &d.node {
            zyntax_typed_ast::TypedDeclaration::Function(f) => {
                f.name.resolve_global().as_deref() == Some("helper$f")
            }
            _ => false,
        })
        .expect("the module's function is declared");
    assert_eq!(
        f.span.file, 1,
        "the module's function is in the second file"
    );
    let main = program
        .declarations
        .iter()
        .find(|d| match &d.node {
            zyntax_typed_ast::TypedDeclaration::Function(f) => {
                f.name.resolve_global().as_deref() == Some(zyntax_python::ENTRY)
            }
            _ => false,
        })
        .expect("the entry is declared");
    assert_eq!(main.span.file, 0, "the entry is in the main file");
}

/// A short-circuit condition on an `if` statement and on a `while`,
/// which reach the branch through the CFG builder rather than through
/// the expression translator.
#[test]
fn short_circuit_in_statement_conditions() {
    assert_eq!(
        run(
            r#"
def clamp(x: int) -> int:
    if 0 <= x and x <= 10:
        return x
    return 10

def count_down(n: int) -> int:
    steps = 0
    while n > 0 and steps < 100:
        n -= 3
        steps += 1
    return steps

def main() -> int:
    return clamp(4) + clamp(40) + count_down(10)
"#,
            "main",
        ),
        // 4 + 10 + 4 steps (10, 7, 4, 1, then -2 stops)
        ZyntaxValue::Int(18)
    );
}

/// A parameter reassigned inside a loop is the reassigned value on the
/// next iteration.
///
/// Every block used to be seeded with a copy of each parameter's entry
/// value, on the assumption that parameters are never reassigned. A
/// loop body writing one then read the copy rather than the header's
/// phi, so `n -= 3` computed from the original `n` every time and the
/// loop never ended. Bounded here so a regression fails rather than
/// hangs.
#[test]
fn a_parameter_reassigned_in_a_loop_advances() {
    assert_eq!(
        run(
            r#"
def count_down(n: int) -> int:
    steps = 0
    while n > 0:
        n -= 3
        steps += 1
        if steps > 50:
            return -1
    return steps

def main() -> int:
    return count_down(10)
"#,
            "main",
        ),
        ZyntaxValue::Int(4)
    );
}

/// The annotations the lowering reads as error-flag facts: raisers set
/// the flag, a function the raise facts prove quiet is nothrow and one
/// that raises is not.
#[test]
fn error_flag_facts_are_annotated() {
    use zyntax_typed_ast::typed_ast::TypedDeclaration;
    let source = r#"
def quiet(a, b):
    return a + b

def loud(n):
    assert n > 0, "must be positive"
    return n

print(quiet(1, 2), loud(3))
"#;
    let program = zyntax_python::parse_program(source).expect("should lower");
    let marks = |name: &str| -> Vec<String> {
        program
            .declarations
            .iter()
            .find_map(|d| match &d.node {
                TypedDeclaration::Function(f)
                    if f.name.resolve_global().as_deref() == Some(name) =>
                {
                    Some(
                        f.annotations
                            .iter()
                            .filter_map(|a| a.name.resolve_global())
                            .collect(),
                    )
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("no function {name}"))
    };
    let has = |name: &str, mark: &str| marks(name).iter().any(|m| m == mark);
    assert!(has("py$raise$AssertionError", "sets_error_flag"));
    assert!(has("zb_hook_raise", "sets_error_flag"));
    assert!(has("quiet", "nothrow"));
    assert!(!has("loud", "nothrow"));
    assert!(!has("py$raise$AssertionError", "nothrow"));
}

/// Of a three-level hierarchy and an unrelated class, only the leaves
/// are declared exact.
#[test]
fn only_leaf_classes_are_exact() {
    let source = r#"
class A:
    def __init__(self, x):
        self.x = x
    def f(self):
        return self.x
class B(A):
    def f(self):
        return self.x + 1
class C(B):
    def f(self):
        return self.x + 2
class D:
    def __init__(self):
        self.y = 3

print(C(1).f(), B(2).f(), A(3).f(), D().y)
"#;
    let program = zyntax_python::parse_program(source).expect("should lower");
    let key = zyntax_typed_ast::InternedString::new_global(zyntax_embed::EXACT_STRUCT_KEY);
    let mut exact: Vec<String> = program
        .type_registry
        .get_all_types()
        .filter(|def| def.metadata.custom.contains_key(&key))
        .filter_map(|def| def.name.resolve_global())
        .filter(|n| ["A", "B", "C", "D"].contains(&n.as_str()))
        .collect();
    exact.sort();
    assert_eq!(exact, ["C", "D"]);
}
