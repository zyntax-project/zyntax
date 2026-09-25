//! The C modules the conformance cases load, built from the sources
//! beside them with the C compiler the environment names (`CC`, else
//! `cc`), against the Lua headers the C API vendors. Users of zylua
//! need no compiler; these fixtures do.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// One build at a time: the categories that share a directory run on
/// several test threads.
static BUILDING: Mutex<()> = Mutex::new(());

/// The vendored headers a module is compiled against.
pub fn include_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("capi")
        .join("vendor")
}

/// The compiler, or why there is none to use.
fn compiler() -> Result<String, String> {
    if cfg!(windows) {
        return Err("C modules do not load on Windows yet (c17ca09)".to_string());
    }
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    match Command::new(&cc).arg("--version").output() {
        Ok(out) if out.status.success() => Ok(cc),
        _ => Err(format!(
            "no C compiler (`{cc}`) to build the C modules with"
        )),
    }
}

fn newer(out: &Path, inputs: &[PathBuf]) -> bool {
    let Ok(built) = std::fs::metadata(out).and_then(|m| m.modified()) else {
        return false;
    };
    inputs.iter().all(|i| {
        std::fs::metadata(i)
            .and_then(|m| m.modified())
            .is_ok_and(|t| t <= built)
    })
}

/// Build each `(library, source)` in `dir` unless it is up to date.
/// `Err` says why nothing can be built here; a source that does not
/// compile fails the test.
pub fn build(dir: &Path, modules: &[(&str, &str)]) -> Result<(), String> {
    let _held = BUILDING.lock().unwrap_or_else(|p| p.into_inner());
    let cc = compiler()?;
    let include = include_dir();
    let headers: Vec<PathBuf> = ["lua.h", "luaconf.h", "lauxlib.h"]
        .iter()
        .map(|h| include.join(h))
        .collect();
    for (library, source) in modules {
        let out = dir.join(library);
        let src = dir.join(source);
        let mut inputs = headers.clone();
        inputs.push(src.clone());
        if newer(&out, &inputs) {
            continue;
        }
        let tmp = dir.join(format!("{library}.tmp"));
        let mut cmd = Command::new(&cc);
        cmd.args(["-std=gnu99", "-O2", "-fPIC", "-shared", "-I"])
            .arg(&include);
        if cfg!(target_os = "macos") {
            // The API's symbols are the executable's, bound when the
            // library is opened.
            cmd.args(["-undefined", "dynamic_lookup"]);
        }
        cmd.arg("-o").arg(&tmp).arg(&src);
        let result = cmd
            .output()
            .map_err(|e| format!("could not run {cc}: {e}"))?;
        assert!(
            result.status.success(),
            "{cc} could not build {}:\n{}",
            src.display(),
            String::from_utf8_lossy(&result.stderr)
        );
        std::fs::rename(&tmp, &out).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
    }
    Ok(())
}

/// The libraries `official/attrib.lua` loads, as `libs/makefile` builds
/// them.
#[allow(dead_code)]
pub const OFFICIAL_LIBS: &[(&str, &str)] = &[
    ("lib1.so", "lib1.c"),
    ("lib11.so", "lib11.c"),
    ("lib2.so", "lib2.c"),
    ("lib21.so", "lib21.c"),
    ("lib2-v2.so", "lib22.c"),
];

/// Every `name.c` in `dir` as `name.so`.
pub fn sources_in(dir: &Path) -> Vec<(String, String)> {
    let mut found: Vec<(String, String)> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "c"))
                .filter_map(|p| {
                    let stem = p.file_stem()?.to_str()?.to_string();
                    Some((format!("{stem}.so"), format!("{stem}.c")))
                })
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    found
}
