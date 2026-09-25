//! `zyntax compile --backend llvm` builds a ZynML program into an
//! executable that runs on its own.
//!
//! Each case compiles a program through the CLI, links it with the
//! system C compiler, runs the result and checks what it printed and
//! the status it exited with. The programs print through libc's
//! `putchar` where they can, so a case needs no runtime plugin.
#![cfg(feature = "llvm-backend")]

use std::path::{Path, PathBuf};
use std::process::Command;

/// Prints an integer and a newline through libc, for programs that
/// cannot use the prelude's `println`.
const PRINT_I64: &str = r#"
extern def putchar(c: i32): i32

def print_digits(n: i64) {
    if n >= 10 {
        print_digits(n / 10)
    }
    putchar((48 + n % 10) as i32)
}

def print_i64(n: i64) {
    if n < 0 {
        putchar(45)
        print_digits(0 - n)
    } else {
        print_digits(n)
    }
    putchar(10)
}
"#;

/// A scratch directory for one case, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("zyntax_aot_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch directory");
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn grammar() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../zynml/ml.zyn")
}

/// Build `source` ahead of time and run it: its stdout and exit status.
fn build_and_run(name: &str, source: &str) -> (String, i32) {
    let scratch = Scratch::new(name);
    let src = scratch.0.join(format!("{name}.zynml"));
    let exe = scratch.0.join(name);
    std::fs::write(&src, source).expect("write source");

    let build = Command::new(env!("CARGO_BIN_EXE_zyntax"))
        .arg("compile")
        .args(["--backend", "llvm"])
        .arg("-g")
        .arg(grammar())
        .arg("-s")
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("run zyntax compile");
    assert!(
        build.status.success() && exe.exists(),
        "zyntax compile failed for {name}:\n{}{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );

    let run = Command::new(&exe).output().expect("run the executable");
    let status = run.status.code().expect("the executable exited normally");
    (String::from_utf8_lossy(&run.stdout).into_owned(), status)
}

/// Integer arithmetic, recursion, `if`, `while` and a counted `for`,
/// printed through libc, with the exit status taken from `main`.
#[test]
fn arithmetic_and_control_flow_run_and_exit_with_mains_value() {
    let source = format!(
        "{PRINT_I64}
def fib(n: i64): i64 {{
    if n < 2 {{
        return n
    }}
    return fib(n - 1) + fib(n - 2)
}}

def main(): i64 {{
    print_i64(fib(20))
    let mut s: i64 = 0
    let mut i: i64 = 0
    while i < 10 {{
        if i % 2 == 0 {{
            s = s + i
        }} else {{
            s = s - 1
        }}
        i = i + 1
    }}
    print_i64(s)
    for k in range(0, 3) {{
        print_i64(k * -7)
    }}
    return 3
}}
"
    );
    let (stdout, status) = build_and_run("control_flow", &source);
    assert_eq!(stdout, "6765\n15\n0\n-7\n-14\n");
    assert_eq!(status, 3);
}

/// Floating-point arithmetic and conversions between `f64` and `i64`.
#[test]
fn float_arithmetic_and_casts() {
    let source = format!(
        "{PRINT_I64}
def main(): i64 {{
    let mut x: f64 = 0.0
    let mut i: i64 = 0
    while i < 100 {{
        x = x + (i as f64) * 0.5
        i = i + 1
    }}
    print_i64(x as i64)
    return 0
}}
"
    );
    let (stdout, status) = build_and_run("floats", &source);
    assert_eq!(stdout, "2475\n");
    assert_eq!(status, 0);
}

/// A `main` that returns nothing exits with status 0.
#[test]
#[ignore = "a void main leaves the exit status undefined; git-bug 5558584fa8dd059b27d5c043d5d9f49e92bb8b1f039b93c863c90cc4d8398fef"]
fn a_main_returning_nothing_exits_zero() {
    let source = format!(
        "{PRINT_I64}
def main() {{
    print_i64(42)
}}
"
    );
    let (stdout, status) = build_and_run("void_main", &source);
    assert_eq!(stdout, "42\n");
    assert_eq!(status, 0);
}

/// A method call on a struct.
#[test]
#[ignore = "zyntax compile does not register ZynML structs or impl blocks; git-bug e0f4d4af72eda1f6336af3b865b6715114db6cd1ca4ee466a8a66edc1e829fd9"]
fn struct_methods_run() {
    let source = format!(
        "{PRINT_I64}
struct Point {{
    x: i64,
    y: i64
}}

impl Point {{
    def sum(self): i64 {{
        return self.x + self.y
    }}
}}

def main(): i64 {{
    let p = Point {{ x: 3, y: 4 }}
    print_i64(p.sum())
    return 0
}}
"
    );
    let (stdout, status) = build_and_run("struct_methods", &source);
    assert_eq!(stdout, "7\n");
    assert_eq!(status, 0);
}

/// A lambda held in a local and called.
#[test]
#[ignore = "the LLVM backend has no lowering for closure types; git-bug 018d88f114e47bd5c04dca1a77a4a27899d253d6f17f8e3e8ac585ee977f1f57"]
fn a_lambda_is_called() {
    let source = format!(
        "{PRINT_I64}
def main(): i64 {{
    let double: (i64) => i64 = def(x): x * 2
    print_i64(double(21))
    return 0
}}
"
    );
    let (stdout, status) = build_and_run("lambda", &source);
    assert_eq!(stdout, "42\n");
    assert_eq!(status, 0);
}

/// The program every ZynML example starts from: the prelude and its
/// `println`, which reaches the IO plugin.
#[test]
#[ignore = "zyntax compile resolves no imports or builtins, and cannot link plugin symbols; git-bug e0f4d4af72eda1f6336af3b865b6715114db6cd1ca4ee466a8a66edc1e829fd9, 36aa93b6dee9ff5ff902aa1cd3f979b6ad02fe040d4cdfda45d2b290f9aedc1c"]
fn a_program_using_the_prelude_prints() {
    let source = "import prelude

def main(): i64 {
    println(42)
    println(\"hello\")
    return 0
}
";
    let (stdout, status) = build_and_run("prelude", source);
    assert_eq!(stdout, "42\nhello\n");
    assert_eq!(status, 0);
}
