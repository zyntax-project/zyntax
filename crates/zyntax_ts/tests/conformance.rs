//! Pinned TypeScript acceptance and Node output, without a reference runtime in CI.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;
use sha2::{Digest, Sha256};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance")
}

fn bounded(action: &str, file: &Path, soundness: &str, interpreter: bool) -> (i32, String, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_zynode"));
    command.args([action, "--diagnostics=json", "--soundness", soundness]);
    for (key, _) in std::env::vars_os() {
        if ["ZYNTAX_", "ZYPY_", "ZYLUA_"]
            .iter()
            .any(|prefix| key.to_string_lossy().starts_with(prefix))
        {
            command.env_remove(key);
        }
    }
    if interpreter {
        command.args(["--baseline-threshold", "4294967295", "--no-osr"]);
        command.env("ZYNTAX_DISABLE_WARM_UP", "1");
    } else {
        command.args(["--baseline-threshold", "2"]);
    }
    let mut child = command
        .arg(file)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let reader = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut text = String::new();
            pipe.read_to_string(&mut text).unwrap();
            text.replace("\r\n", "\n")
        })
    };
    let out = reader(Box::new(stdout));
    let err = reader(Box::new(stderr));
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(30) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("{} timed out", file.display());
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    (
        status.code().unwrap_or(-1),
        out.join().unwrap(),
        err.join().unwrap(),
    )
}

#[test]
fn reference_pins_match_native_acceptance_and_execution() {
    let package: Value = serde_json::from_str(include_str!("../conformance/package.json")).unwrap();
    let compiler = package["devDependencies"]["typescript"].as_str().unwrap();
    let cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(root().join("cases.json")).unwrap()).unwrap();
    let known: HashMap<_, _> = std::fs::read_to_string(root().join("KNOWN_FAILURES"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let parts: Vec<_> = l.split_whitespace().collect();
            assert_eq!(parts.len(), 2);
            assert_eq!(parts[1].len(), 64);
            (parts[0].to_owned(), parts[1].to_owned())
        })
        .collect();
    let mut failures = vec![];
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let file = root().join(case["file"].as_str().unwrap());
        let reference: Value = serde_json::from_str(
            &std::fs::read_to_string(file.with_extension("reference.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(reference["compiler"], compiler);
        let mut hash = Sha256::new();
        let files: Vec<_> = case["files"].as_array().map_or_else(
            || vec![file.clone()],
            |files| {
                files
                    .iter()
                    .map(|f| root().join(f.as_str().unwrap()))
                    .collect()
            },
        );
        assert_eq!(files.first(), Some(&file), "entry must be the first file");
        let mut base = file.parent().unwrap();
        while !files.iter().all(|path| path.starts_with(base)) {
            base = base.parent().unwrap();
        }
        for source in files {
            hash.update(
                source
                    .strip_prefix(base)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
                    .as_bytes(),
            );
            hash.update([0]);
            hash.update(std::fs::read(source).unwrap());
        }
        assert_eq!(
            reference["source_sha256"],
            format!("{:x}", hash.finalize()),
            "stale oracle for {id}"
        );
        for interpreter in [true, false] {
            let soundness = case["soundness"].as_str().unwrap_or("typescript");
            let (code, _, stderr) = bounded("check", &file, soundness, interpreter);
            let diagnostic: Value = stderr
                .lines()
                .rev()
                .find_map(|line| serde_json::from_str::<Value>(line).ok())
                .unwrap_or(Value::Null);
            let accepted = reference["accepted"].as_bool().unwrap();
            let ok = if case["policy_error"].as_bool() == Some(true) {
                accepted && code == 5 && diagnostic["kind"] == "Soundness"
            } else if accepted {
                if code != 0 {
                    false
                } else if case["mode"] == "run" {
                    let (status, stdout, err) = bounded("run", &file, soundness, interpreter);
                    if status != reference["exit"].as_i64().unwrap() as i32
                        || stdout != reference["stdout"].as_str().unwrap()
                    {
                        failures.push(format!(
                            "{id}: output/status differs ({status}): {stdout:?}; {err}"
                        ));
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            } else {
                code == 1 && diagnostic["kind"] == reference["kind"]
            };
            if known.contains_key(id) {
                if ok || code != 3 || diagnostic["kind"] != "Unsupported" {
                    failures.push(format!("{id}: stale or unrelated known failure: {stderr}"));
                }
            } else if !ok {
                failures.push(format!("{id}, interpreter={interpreter}: {stderr}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
