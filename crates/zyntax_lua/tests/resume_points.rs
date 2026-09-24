use std::path::Path;
use std::process::Command;

/// The function returns the table its loop fills. The frame leaves
/// through a resume point that hands the table back, and the call after
/// runs the compiled body.
#[test]
fn a_frame_in_a_function_returning_a_table_leaves_through_a_resume_point() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/table_return.lua");
    let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
        .arg("run")
        .arg(script)
        .env("ZYNTAX_OSR_TRACE", "1")
        .env("ZYNTAX_TRACE_INTERP", "1")
        .output()
        .expect("zylua starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(
        stdout.trim(),
        "30\t0\t1483337328\n30\t1483337328",
        "{stderr}"
    );
    assert!(
        stderr.contains("lua$main$1 site=") && stderr.contains("[interp] transfer site="),
        "the frame never left through a resume point:\n{stderr}"
    );
}

/// A frame reaches a second long loop after its function was promoted,
/// through a region outlined only then: the region's late resume point
/// is made at the optimizing tier and the frame moves into it.
#[cfg(feature = "llvm-backend")]
#[test]
fn a_region_outlined_after_the_promotion_gets_a_late_resume_point() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/late_region.lua");
    let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
        .arg("run")
        .arg(script)
        .env("ZYLUA_LLVM", "1")
        .env("ZYNTAX_OSR_TRACE", "1")
        .output()
        .expect("zylua starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "1804062937", "{stderr}");
    let site = stderr
        .lines()
        .find(|l| l.contains("late resume point (llvm)"))
        .and_then(|l| l.split("site=").nth(1))
        .and_then(|s| s.split(':').next())
        .unwrap_or_else(|| panic!("no late resume point was made:\n{stderr}"));
    assert!(
        stderr.lines().any(|l| l.contains("FIRST TRANSFER")
            && l.contains(&format!("site {site} "))
            && l.ends_with("(llvm)")),
        "the frame never moved into the late resume point:\n{stderr}"
    );
}

/// A frame asks at a loop header while the promotion its function's
/// call count raised is compiling, or after it landed: it gets the
/// optimizing tier's resume point, never the baseline's, every time.
#[cfg(feature = "llvm-backend")]
#[test]
fn a_site_asked_during_an_entry_count_promotion_gets_the_optimizing_tier() {
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/entry_count_in_flight.lua");
    let mut transfers = 0;
    for _ in 0..20 {
        let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
            .arg("run")
            .arg(&script)
            .env("ZYLUA_LLVM", "1")
            .env("ZYNTAX_OSR_TRACE", "1")
            .output()
            .expect("zylua starts");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stderr}");
        assert_eq!(stdout.trim(), "600272934\t12", "{stderr}");
        for line in stderr.lines().filter(|l| l.contains("FIRST TRANSFER")) {
            assert!(line.ends_with("(llvm)"), "{line}\n{stderr}");
            transfers += 1;
        }
    }
    assert!(transfers > 0, "no frame moved to a resume point in 20 runs");
}

/// A loop whose number is an integer on one path and a float on the
/// other runs in an outlined region taking a Number aggregate. LLVM
/// compiles that region with the aggregate by address, and the frame
/// transfers into it.
#[cfg(feature = "llvm-backend")]
#[test]
fn a_region_taking_a_number_aggregate_runs_on_llvm() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/number_region.lua");
    let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
        .arg("run")
        .arg(script)
        .env("ZYLUA_LLVM", "1")
        .env("ZYNTAX_OSR_TRACE", "1")
        .output()
        .expect("zylua starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    // lua5.4 prints the same.
    assert_eq!(stdout.trim(), "16757736", "{stderr}");
    assert!(
        stderr.contains("(lua$main$1$resume0) at tier 1"),
        "the region was not recompiled at tier 1:\n{stderr}"
    );
    assert!(stderr.contains("[osr] llvm install tier=1"), "{stderr}");
    assert!(
        stderr.contains("(llvm)") && stderr.contains("FIRST TRANSFER"),
        "no transfer into LLVM code:\n{stderr}"
    );
    assert!(!stderr.contains("unsupported LLVM entry ABI"), "{stderr}");
    assert!(!stderr.contains("LLVM compile failed"), "{stderr}");
}
