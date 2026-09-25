use std::path::Path;
use std::process::Command;

/// An interpreted frame leaves through a resume point compiled before
/// the function's own body; the resumed code calls the function, whose
/// definition does not exist yet.
#[test]
fn a_resumed_frame_calls_its_function_before_the_body_is_compiled() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recursive_loop.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYNTAX_OSR_TRACE", "1")
        .env("ZYNTAX_TRACE_INTERP", "1")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "2399976", "{stderr}");
    assert!(
        stderr.contains("outlined resume point"),
        "the resume point was not outlined:\n{stderr}"
    );
    assert!(
        stderr.contains("[interp] transfer site="),
        "the frame never left through a resume point:\n{stderr}"
    );
}

/// The frame leaves at an inner loop's header: the outer loop's counter
/// and the bodies loop's state come in from the frame and are repaired
/// at the header, and the region runs the rest of the call.
#[test]
fn a_frame_leaves_an_inner_loop_into_the_outlined_region() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nested_loops.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYNTAX_OSR_TRACE", "1")
        .env("ZYNTAX_TRACE_INTERP", "1")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "1297500.0\n129.75", "{stderr}");
    assert!(
        stderr.contains("outlined resume point") && stderr.contains("[interp] transfer site="),
        "{stderr}"
    );
}

/// The frame resumed into the outlined region moves on to the
/// optimizing tier mid-loop: the region is compiled there too, and its
/// resume points answer the probes of the region's baseline code.
#[cfg(feature = "llvm-backend")]
#[test]
fn a_frame_in_the_outlined_region_moves_to_the_optimizing_tier() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/long_single_call.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYPY_LLVM", "1")
        .env("ZYNTAX_OSR_TRACE", "1")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "25574991424", "{stderr}");
    assert!(
        stderr.contains("outlined resume point"),
        "the resume point was not outlined:\n{stderr}"
    );
    let region = stderr
        .find("(run$resume0) at tier 1")
        .expect("the region is compiled at the optimizing tier");
    assert!(
        stderr[region..].contains("(llvm)"),
        "the frame never moved to the optimizing tier:\n{stderr}"
    );
}

/// A loop reading the result of an inlined callee with two return
/// paths moves to the optimizing tier mid-loop and finishes with
/// CPython's answer: the resume point compiles its region in an order
/// that defines the merged result before the loop reads it.
#[cfg(feature = "llvm-backend")]
#[test]
fn a_loop_reading_a_two_way_callee_moves_to_the_optimizing_tier() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/two_return_paths.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYPY_LLVM", "1")
        .env("ZYNTAX_OSR_TRACE", "1")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "140000000", "{stderr}");
    assert!(
        stderr
            .lines()
            .any(|l| l.contains("FIRST TRANSFER") && l.ends_with("(llvm)")),
        "the frame never moved into an LLVM resume point:\n{stderr}"
    );
}

/// The function returns the list its loop fills. The frame leaves
/// through a resume point that hands the list back as the address of
/// its header, and the call after runs the compiled body.
#[test]
fn a_frame_in_a_function_returning_a_list_leaves_through_a_resume_point() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/list_return.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYNTAX_OSR_TRACE", "1")
        .env("ZYNTAX_TRACE_INTERP", "1")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "30 0 1483337328\n30 1483337328", "{stderr}");
    assert!(
        !stderr.contains("main header HirId"),
        "main's loop header has no resume point:\n{stderr}"
    );
    assert!(
        stderr.contains("[osr] main site=") && stderr.contains("[interp] transfer site="),
        "the frame never left through a resume point:\n{stderr}"
    );
}

/// The same, with the optimizing tier on: the region outlined for the
/// frame is compiled there too, with the list returned as its address.
#[cfg(feature = "llvm-backend")]
#[test]
fn a_region_returning_a_list_compiles_at_the_optimizing_tier() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/list_return.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYPY_LLVM", "1")
        .env("ZYNTAX_OSR_TRACE", "1")
        .env("ZYNTAX_TRACE_INTERP", "1")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "30 0 1483337328\n30 1483337328", "{stderr}");
    assert!(
        stderr.contains("[osr] main site=") && stderr.contains("[interp] transfer site="),
        "the frame never left through a resume point:\n{stderr}"
    );
    assert!(!stderr.contains("LLVM compile failed"), "{stderr}");
}

/// A frame that leaves at a loop header stops rooting its registers:
/// a structure it finished with before the loop is reclaimed by a
/// collection during the loop.
#[test]
fn a_frame_that_left_no_longer_roots_what_died_before_the_loop() {
    if std::env::var_os("ZYNTAX_DISABLE_GC").is_some() {
        return;
    }
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dead_before_loop.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYNTAX_TRACE_GC", "1")
        .env("ZYNTAX_TRACE_INTERP", "1")
        .env("ZYNTAX_GC_FLOOR_KB", "4096")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "10700000", "{stderr}");
    // Reached KB per collection, split at the last transfer, which is
    // the outer frame's.
    let reached = |line: &str| -> Option<u64> {
        let rest = line.strip_prefix("[gc] #")?;
        let (_, rest) = rest.split_once("since last, ")?;
        let (kb, _) = rest.split_once(" KB reached")?;
        kb.trim().parse().ok()
    };
    let mut before = Vec::new();
    let mut after = Vec::new();
    for line in stderr.lines() {
        if line.starts_with("[interp] transfer site=") {
            before.append(&mut after);
        } else if let Some(kb) = reached(line) {
            after.push(kb);
        }
    }
    assert!(
        !before.is_empty() && !after.is_empty(),
        "collections on both sides of the transfer are expected:\n{stderr}"
    );
    let built = before.iter().copied().max().unwrap_or(0);
    let left = after.iter().copied().min().unwrap_or(u64::MAX);
    assert!(
        built >= 3 << 10,
        "the structure was never marked: {built} KB\n{stderr}"
    );
    assert!(
        left < 2 << 10,
        "a collection after the transfer still reached {left} KB:\n{stderr}"
    );
}

/// A frame reaches a second long loop after its function was promoted,
/// through a region outlined only then: the region's late resume point
/// is made at the optimizing tier and the frame moves into it.
#[cfg(feature = "llvm-backend")]
#[test]
fn a_region_outlined_after_the_promotion_gets_a_late_resume_point() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/late_region.py");
    let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
        .arg("run")
        .arg(script)
        .env("ZYPY_LLVM", "1")
        .env("ZYNTAX_OSR_TRACE", "1")
        .output()
        .expect("zypy starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(stdout.trim(), "1550062952", "{stderr}");
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
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/entry_count_in_flight.py");
    let mut transfers = 0;
    for _ in 0..20 {
        let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
            .arg("run")
            .arg(&script)
            .env("ZYPY_LLVM", "1")
            .env("ZYNTAX_OSR_TRACE", "1")
            .output()
            .expect("zypy starts");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stderr}");
        assert_eq!(stdout.trim(), "600296790 3013", "{stderr}");
        for line in stderr.lines().filter(|l| l.contains("FIRST TRANSFER")) {
            assert!(line.ends_with("(llvm)"), "{line}\n{stderr}");
            transfers += 1;
        }
    }
    assert!(transfers > 0, "no frame moved to a resume point in 20 runs");
}

/// A frame leaves into a region the optimizing tier compiled, whose
/// parameters include a loop value of no type: every parameter after it
/// reaches the region where the region reads it, and the list the
/// region returns is the one the frame filled. The frame asks while
/// its loop is warm, which takes the warm-up worker; whether the region
/// is promoted before the frame leaves is up to timing, so the program
/// runs until it has been, each run answering as CPython does.
#[cfg(feature = "llvm-backend")]
#[test]
fn a_region_taking_a_void_loop_value_gets_the_parameters_after_it() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/void_loop_value.py");
    let mut entered = false;
    for _ in 0..5 {
        let output = Command::new(env!("CARGO_BIN_EXE_zypy"))
            .arg("run")
            .arg(&script)
            .env("ZYPY_LLVM", "1")
            .env("ZYNTAX_OSR_TRACE", "1")
            .env_remove("ZYNTAX_DISABLE_WARM_UP")
            .output()
            .expect("zypy starts");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stderr}");
        assert_eq!(stdout.trim(), "71 900854.0", "{stderr}");
        let site = stderr
            .lines()
            .find(|l| l.starts_with("[osr] main site=") && l.ends_with("outlined resume point"))
            .and_then(|l| l.split("site=").nth(1))
            .and_then(|s| s.split(':').next());
        let promoted = stderr.find("(main$resume0) at tier 1");
        if let (Some(site), Some(promoted)) = (site, promoted)
            && stderr[promoted..].contains(&format!("interpreted frame leaves at site={site} "))
        {
            entered = true;
            break;
        }
    }
    assert!(
        entered,
        "in 5 runs the frame never entered the region after its promotion"
    );
}
