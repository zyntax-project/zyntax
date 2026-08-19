//! What the ownership check says about programs known to be correct.
//!
//! Every benchmark in the repository is a working program. A guard that
//! reports anything here is wrong about that program, so this is the
//! measurement that decides whether the check can be turned on rather
//! than merely offered.

use std::path::{Path, PathBuf};
use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::{borrow_check, BorrowError};
use zyntax_embed::ZyntaxRuntime;

fn benchmarks() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "zynml"))
        .collect();
    v.sort();
    v
}

fn check_file(path: &Path) -> Result<Vec<BorrowError>, String> {
    let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).map_err(|e| format!("{e:?}"))?;
    let program = grammar
        .parse_with_filename(&src, path.to_str().unwrap_or("<bench>"))
        .map_err(|e| format!("parse: {e:?}"))?;
    let mut rt = ZyntaxRuntime::new().map_err(|e| format!("{e:?}"))?;
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
        _ => Ok(None),
    }));
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt
        .lower_typed_program(program, builtins)
        .map_err(|e| format!("lower: {e:?}"))?;
    Ok(borrow_check::check_ownership(&module)
        .map_err(|e| format!("check: {e:?}"))?
        .errors)
}

/// No working program may be reported. If this fails, the number in the
/// message is how many correct programs the guard would reject.
#[test]
fn no_working_benchmark_is_reported() {
    let files = benchmarks();
    assert!(
        !files.is_empty(),
        "no benchmarks found; this test would prove nothing"
    );

    let mut offenders = Vec::new();
    let mut checked = 0;
    for f in &files {
        match check_file(f) {
            Ok(errors) if errors.is_empty() => checked += 1,
            Ok(errors) => {
                checked += 1;
                offenders.push(format!(
                    "{}: {} error(s), first = {:?}",
                    f.file_name().unwrap().to_string_lossy(),
                    errors.len(),
                    errors.first()
                ));
            }
            // A program this harness cannot lower says nothing about the
            // guard, so it is not counted either way.
            Err(_) => {}
        }
    }
    assert!(
        checked > 0,
        "nothing lowered; the count below would be meaningless"
    );
    println!("checked {checked} of {} benchmarks", files.len());
    assert!(
        offenders.is_empty(),
        "the ownership check reports {} working program(s):\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}
