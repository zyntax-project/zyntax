//! Startup placement must preserve initialization priority and normal calls.
#![cfg(target_os = "linux")]

use std::process::Command;

#[test]
fn grouped_startup_preserves_constructor_order() {
    let directory = tempfile::Builder::new()
        .prefix("native startup , ")
        .tempdir()
        .unwrap();
    let first = directory.path().join("first.c");
    let second = directory.path().join("second.c");
    std::fs::write(
        &first,
        r#"
int value;
__attribute__((constructor(101), section(".text.startup.first")))
static void first(void) { value = 1; }
int read_value(void) { return value; }
"#,
    )
    .unwrap();
    std::fs::write(
        &second,
        r#"
#include <stdio.h>
extern int value;
extern int read_value(void);
__attribute__((constructor(102), section(".text.startup.second")))
static void second(void) { value = value * 10 + 2; }
int main(void) { printf("%d\n", read_value()); return value != 12; }
"#,
    )
    .unwrap();
    let layout = zrtl_native::build::startup_link_args("linux", directory.path()).unwrap();
    for (name, args) in [("ordinary", vec![]), ("grouped", layout)] {
        let binary = directory.path().join(name);
        let output = Command::new("cc")
            .args([
                "-O2",
                "-ffunction-sections",
                "-fdata-sections",
                "-fPIE",
                "-pie",
                "-Wl,-z,relro,-z,now,--gc-sections",
            ])
            .args(args)
            .args([&first, &second])
            .arg("-o")
            .arg(&binary)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output = Command::new(binary).output().unwrap();
        assert!(output.status.success(), "{name}");
        assert_eq!(output.stdout, b"12\n", "{name}");
    }
}
