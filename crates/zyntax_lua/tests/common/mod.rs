//! What the conformance and official-suite targets share: the known
//! failures list, pinned reference answers, and a bounded run.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance")
}

/// The directory the official suite runs in.
pub fn official_dir() -> PathBuf {
    root().join("official")
}

/// The globals `all.lua` derives from `_U=true` (its lines 21 to 35),
/// set ahead of each official file run on its own. `all.lua` also sets
/// `debug = nil`; that stays out because `-e` joins the file's first
/// line, where a global write of `debug` changes how the file compiles.
pub const OFFICIAL_PRELUDE: &str = "_U=true _soft=true _port=true _nomsg=true";

/// Official files that are never run, and why.
pub const NOT_RUN: &[(&str, &str)] = &[(
    "heavy.lua",
    "a memory stress test with no pass criterion; all.lua never runs it",
)];

/// The deadline of one file's run: a debug build compiles and runs
/// several times slower.
pub const FILE_DEADLINE: Duration =
    Duration::from_secs(if cfg!(debug_assertions) { 480 } else { 120 });

pub mod c_modules;

/// Build the C test libraries the complete configuration loads, or say
/// why they cannot be built here.
pub fn official_libs() -> Result<(), String> {
    c_modules::build(&official_dir().join("libs"), c_modules::OFFICIAL_LIBS)
}

/// One KNOWN_FAILURES entry: the issue, and the platform family the
/// case's pinned output is limited to when the entry names one.
#[derive(Debug, PartialEq, Eq)]
pub struct Listed {
    pub issue: String,
    pub only_on: Option<String>,
}

/// The platform families an entry may name.
pub const FAMILIES: &[&str] = &["unix", "windows", "macos", "linux"];

/// `category/file.lua` -> its entry, from KNOWN_FAILURES.
pub fn known_failures() -> HashMap<String, Listed> {
    fs::read_to_string(root().join("KNOWN_FAILURES"))
        .map(|text| parse_known_failures(&text))
        .unwrap_or_default()
}

/// The entries of a KNOWN_FAILURES text: `<case> <issue> [family]` per
/// line, `#` starting a comment line.
pub fn parse_known_failures(text: &str) -> HashMap<String, Listed> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        if let (Some(case), Some(issue)) = (parts.next(), parts.next()) {
            let only_on = parts.next().map(str::to_string);
            if let Some(family) = &only_on {
                assert!(
                    FAMILIES.contains(&family.as_str()),
                    "KNOWN_FAILURES: `{case}` names platform `{family}`; the families are {FAMILIES:?}"
                );
            }
            out.insert(
                case.to_string(),
                Listed {
                    issue: issue.to_string(),
                    only_on,
                },
            );
        }
    }
    out
}

/// Whether this build targets the platform family `family`.
pub fn on_family(family: &str) -> bool {
    match family {
        "unix" => cfg!(unix),
        "windows" => cfg!(windows),
        "macos" => cfg!(target_os = "macos"),
        "linux" => cfg!(target_os = "linux"),
        _ => false,
    }
}

/// What one run produced.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// Its bytes, whatever they are: a Lua string need not be UTF-8.
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub status: i32,
    /// The uncaught error's line from stderr (see `message_of`), less
    /// the program's name that `lua` puts before it.
    pub message: String,
}

impl Outcome {
    fn failed(what: String, status: i32) -> Outcome {
        Outcome {
            stdout: what.into_bytes(),
            status,
            ..Outcome::default()
        }
    }
}

/// The message line of what a run wrote to stderr: the line before the
/// last `stack traceback:`, which is the uncaught error's however much
/// the program and its children wrote before it, else the first line.
pub fn message_of(stderr: &str) -> String {
    let lines: Vec<&str> = stderr.lines().collect();
    let line = match lines.iter().rposition(|l| *l == "stack traceback:") {
        Some(at) if at > 0 => lines[at - 1],
        _ => lines.first().copied().unwrap_or(""),
    };
    match line.split_once(": ") {
        Some((_, message)) => message.to_string(),
        None => line.to_string(),
    }
}

/// Run a command with a deadline. A case that hangs is a failure that
/// must be reported, not a suite that never finishes. stdin is a pipe
/// closed at once: reads see end of file and a seek fails, as on a pipe
/// or a terminal.
pub fn run_bounded(mut cmd: Command, limit: Duration) -> Outcome {
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return Outcome::failed(format!("<could not start: {e}>"), -1),
    };
    drop(child.stdin.take());
    // Both drained as they are written, so a program that prints more
    // than a pipe holds does not wait on a reader that waits on its
    // exit.
    fn drain<R: Read + Send + 'static>(s: Option<R>) -> Option<std::thread::JoinHandle<Vec<u8>>> {
        s.map(|mut s| {
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = s.read_to_end(&mut bytes);
                bytes
            })
        })
    }
    let reader = drain(child.stdout.take());
    let errors = drain(child.stderr.take());
    let collect = |reader: Option<std::thread::JoinHandle<Vec<u8>>>| {
        reader.and_then(|r| r.join().ok()).unwrap_or_default()
    };
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stderr = collect(errors);
                return Outcome {
                    stdout: collect(reader),
                    status: status.code().unwrap_or(-2),
                    message: message_of(&String::from_utf8_lossy(&stderr)),
                    stderr,
                };
            }
            Ok(None) if start.elapsed() > limit => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = collect(reader);
                let _ = collect(errors);
                return Outcome::failed(format!("<timed out after {limit:?}>"), -3);
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => return Outcome::failed(format!("<wait failed: {e}>"), -1),
        }
    }
}

/// The reference interpreter, wherever it is: `LUA` names it, else
/// `lua5.4` then `lua` on the path.
pub fn reference() -> Command {
    if let Ok(lua) = std::env::var("LUA") {
        return Command::new(lua);
    }
    for candidate in ["lua5.4", "/opt/homebrew/opt/lua@5.4/bin/lua5.4", "lua"] {
        if Command::new(candidate).arg("-v").output().is_ok() {
            return Command::new(candidate);
        }
    }
    Command::new("lua5.4")
}

/// Where a case's reference answer is pinned: `<dir>/<stem>.expected`,
/// with `.status` and `.message` beside it when the reference exits
/// non-zero.
pub struct Pin {
    pub dir: PathBuf,
    pub stem: String,
}

impl Pin {
    fn file(&self, ext: &str) -> PathBuf {
        self.dir.join(format!("{}.{ext}", self.stem))
    }

    /// The pinned answer, with its message if one is pinned.
    pub fn read(&self) -> Option<(Outcome, Option<String>)> {
        let bytes = fs::read(self.file("expected")).ok()?;
        let status = fs::read_to_string(self.file("status"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let message = fs::read_to_string(self.file("message"))
            .ok()
            .map(|m| m.trim_end_matches('\n').to_string());
        Some((
            Outcome {
                stdout: bytes,
                status,
                message: message.clone().unwrap_or_default(),
                ..Outcome::default()
            },
            message,
        ))
    }

    /// The pinned answer, or the reference's when nothing is pinned yet,
    /// which is then written so the next run needs neither Lua nor a
    /// decision. `cmd` is the reference command for the case.
    pub fn read_or_pin(&self, cmd: Command, limit: Duration) -> Option<(Outcome, Option<String>)> {
        if let Some(pinned) = self.read() {
            return Some(pinned);
        }
        let got = run_bounded(cmd, limit);
        if got.status < 0 || got.status > 100 {
            eprintln!(
                "  cannot pin {}: the reference interpreter exited {} (is lua5.4 installed?)",
                self.file("expected").display(),
                got.status
            );
            return None;
        }
        let _ = fs::create_dir_all(&self.dir);
        let _ = fs::write(self.file("expected"), &got.stdout);
        let mut message = None;
        if got.status != 0 {
            let _ = fs::write(self.file("status"), got.status.to_string());
            let _ = fs::write(self.file("message"), format!("{}\n", got.message));
            message = Some(got.message.clone());
        }
        Some((got, message))
    }
}

/// The last line that is not blank.
pub fn last_line(bytes: &[u8]) -> &[u8] {
    bytes
        .split(|&b| b == b'\n')
        .rev()
        .find(|l| !l.trim_ascii().is_empty())
        .unwrap_or(&[])
}

/// Whether `line` is one of the lines of `bytes`.
pub fn has_line(bytes: &[u8], line: &[u8]) -> bool {
    bytes.split(|&b| b == b'\n').any(|l| l == line)
}

/// The first and last `n` lines of `bytes`, indented, with a count of
/// what lies between; bytes that are not UTF-8 show as U+FFFD.
pub fn head_and_tail(bytes: &[u8], n: usize) -> String {
    let s = String::from_utf8_lossy(bytes);
    let lines: Vec<&str> = s.lines().collect();
    let shown = |ls: &[&str]| {
        ls.iter()
            .map(|l| format!("        {l}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    if lines.len() <= 2 * n {
        return shown(&lines);
    }
    format!(
        "{}\n        ... {} lines ...\n{}",
        shown(&lines[..n]),
        lines.len() - 2 * n,
        shown(&lines[lines.len() - n..])
    )
}
