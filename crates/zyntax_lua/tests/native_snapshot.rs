use std::path::Path;
use std::process::Command;

#[test]
fn collector_drop_policy_keeps_loading_native_snapshot_code() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/method_call.lua");
    for disabled in [false, true] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_zylua"));
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("ZYNTAX_") {
                cmd.env_remove(key);
            }
        }
        cmd.arg(&script)
            .env("ZYNTAX_GC_MANAGED_DROPS", "1")
            .env("ZYNTAX_TRACE_PORTABLE", "1");
        if disabled {
            cmd.env("ZYNTAX_DISABLE_PORTABLE_CODE", "1");
        }
        let output = cmd.output().expect("zylua starts");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stderr}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("result: 5"));
        if disabled {
            assert!(
                stderr.contains("ZYNTAX_DISABLE_PORTABLE_CODE is set"),
                "{stderr}"
            );
            assert!(!stderr.contains("[portable] placed"), "{stderr}");
        } else {
            assert!(stderr.contains("[portable] placed"), "{stderr}");
            assert!(!stderr.contains("no machine code"), "{stderr}");
        }
    }
}
