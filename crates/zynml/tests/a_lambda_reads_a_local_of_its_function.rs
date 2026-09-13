//! A lambda that reads a local of the function it is written in.
//!
//! The variable is captured: the lambda's body reads it from the
//! closure, and the closure is made with it. The value is what the
//! local held where the lambda was made.

use std::path::Path;
use zynml::{ZynML, ZynMLConfig};

fn run(src: &str) -> i64 {
    let plugins = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../plugins/target/zrtl");
    let cfg = ZynMLConfig {
        plugins_dir: plugins.to_string_lossy().to_string(),
        ..ZynMLConfig::default()
    };
    let mut z = ZynML::with_config(cfg).expect("runtime");
    z.load_source(src).expect("should compile");
    z.call_with_result::<i64>("main").expect("should run")
}

/// Each of the ten steps adds `(i + 3) % 8`: 3+4+5+6+7+0+1+2+3+4.
#[test]
#[ignore = "the lambda body names the local's SSA value instead of a capture; git-bug 091519d"]
fn a_lambda_adds_a_captured_local() {
    assert_eq!(
        run(r#"
import prelude

def main(): i64 {
    let k: i64 = 3
    let step: (i64, i64) => i64 = def(acc, i): acc + (i + k) % 8
    let mut sum: i64 = 0
    let mut i: i64 = 0
    while i < 10 {
        sum = step(sum, i)
        i = i + 1
    }
    return sum
}
"#),
        35
    );
}

/// A parameter of the function is captured the same way.
#[test]
#[ignore = "the lambda body names the local's SSA value instead of a capture; git-bug 091519d"]
fn a_lambda_adds_a_captured_parameter() {
    assert_eq!(
        run(r#"
import prelude

def offset_by(k: i64): i64 {
    let add: (i64) => i64 = def(x): x + k
    return add(10)
}

def main(): i64 {
    return offset_by(5)
}
"#),
        15
    );
}
