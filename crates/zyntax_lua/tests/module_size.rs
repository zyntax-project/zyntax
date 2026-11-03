use std::path::Path;

fn oversized_modules(dir: &Path, failures: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).expect("source directory is readable") {
        let path = entry.expect("source entry is readable").path();
        if path.is_dir() {
            oversized_modules(&path, failures);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let lines = std::fs::read_to_string(&path)
                .expect("Rust source is readable")
                .lines()
                .count();
            if lines > 1_000 {
                failures.push(format!("{}: {lines} lines", path.display()));
            }
        }
    }
}

#[test]
fn source_modules_stay_within_one_thousand_lines() {
    let mut failures = Vec::new();
    oversized_modules(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut failures,
    );
    failures.sort();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
