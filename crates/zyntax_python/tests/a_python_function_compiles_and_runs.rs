//! Python source reaches native code through the same pipeline as
//! everything else.
//!
//! The frontend hands the runtime a `TypedProgram`, the way the Haxe
//! path does; nothing downstream knows it came from Python. So a
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
    let err = zyntax_python::parse_program("async def f():\n    return 1\n")
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
