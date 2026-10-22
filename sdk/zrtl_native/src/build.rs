//! For build scripts: native executable exports and startup text placement.
//!
//! An executable exports nothing a library opened later can bind to
//! unless it is told to. These write the list in the form each linker
//! reads and return the arguments naming it, for a build script to
//! pass with `cargo:rustc-link-arg-bins`.

use std::io;
use std::path::Path;

/// Emit startup placement for a Cargo binary build. Setting
/// `ZYNTAX_KEEP_LINKER_LAYOUT` retains a custom linker's original layout.
pub fn emit_startup_layout(dir: &Path) -> io::Result<()> {
    println!("cargo:rerun-if-env-changed=ZYNTAX_KEEP_LINKER_LAYOUT");
    if std::env::var_os("ZYNTAX_KEEP_LINKER_LAYOUT").is_some() {
        return Ok(());
    }
    let os = std::env::var("CARGO_CFG_TARGET_OS")
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    for arg in startup_link_args(&os, dir)? {
        println!("cargo:rustc-link-arg-bins={arg}");
    }
    Ok(())
}

/// Group ELF startup text without changing constructor-array order or the
/// loader ABI. Other targets retain their native linker's layout.
pub fn startup_link_args(target_os: &str, dir: &Path) -> io::Result<Vec<String>> {
    if target_os != "linux" {
        return Ok(Vec::new());
    }
    let path = dir.join("native_startup.ld");
    std::fs::write(
        &path,
        "SECTIONS {\n  .text.startup : { *(.text.startup .text.startup.*) }\n}\nINSERT BEFORE .text;\n",
    )?;
    Ok(vec![
        "-Xlinker".into(),
        format!("--script={}", path.display()),
    ])
}

/// Write `names` as an export list under `dir` for a target of
/// `target_os` and `target_env` (`CARGO_CFG_TARGET_OS` and
/// `CARGO_CFG_TARGET_ENV`), and return the linker arguments that apply
/// it: an exported-symbols list on Apple targets, which keeps `main`
/// exported beside the names; a dynamic list on ELF targets; a module
/// definition on Windows. Empty for a target with no such form.
pub fn export_link_args(
    target_os: &str,
    target_env: &str,
    names: &[String],
    dir: &Path,
) -> io::Result<Vec<String>> {
    match target_os {
        "macos" | "ios" | "tvos" | "watchos" | "visionos" => {
            let path = dir.join("exported_symbols.txt");
            let mut text = String::from("_main\n");
            for name in names {
                text.push('_');
                text.push_str(name);
                text.push('\n');
            }
            std::fs::write(&path, text)?;
            Ok(vec![format!(
                "-Wl,-exported_symbols_list,{}",
                path.display()
            )])
        }
        "windows" => {
            let path = dir.join("exports.def");
            let mut text = String::from("EXPORTS\n");
            for name in names {
                text.push_str("    ");
                text.push_str(name);
                text.push('\n');
            }
            std::fs::write(&path, text)?;
            if target_env == "msvc" {
                Ok(vec![format!("/DEF:{}", path.display())])
            } else {
                Ok(vec![path.display().to_string()])
            }
        }
        "linux" | "android" | "freebsd" | "netbsd" | "openbsd" | "dragonfly" | "illumos"
        | "solaris" => {
            let path = dir.join("exports.dynlist");
            let mut text = String::from("{\n");
            for name in names {
                text.push_str("  ");
                text.push_str(name);
                text.push_str(";\n");
            }
            text.push_str("};\n");
            std::fs::write(&path, text)?;
            Ok(vec![format!("-Wl,--dynamic-list={}", path.display())])
        }
        _ => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("zrtl_native_exports_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn startup_layout_leaves_non_linux_targets_untouched() {
        let dir = scratch("startup_other_targets");
        for os in ["macos", "windows", "unknown", "freebsd", "android"] {
            assert!(startup_link_args(os, &dir).unwrap().is_empty());
        }
        assert!(!dir.join("native_startup.ld").exists());
    }

    #[test]
    fn apple_lists_main_and_each_name_with_an_underscore() {
        let dir = scratch("apple");
        let args =
            export_link_args("macos", "", &["f".to_string(), "g".to_string()], &dir).unwrap();
        assert_eq!(args.len(), 1);
        assert!(args[0].starts_with("-Wl,-exported_symbols_list,"));
        let text = std::fs::read_to_string(dir.join("exported_symbols.txt")).unwrap();
        assert_eq!(text, "_main\n_f\n_g\n");
    }

    #[test]
    fn elf_writes_a_dynamic_list() {
        let dir = scratch("elf");
        let args = export_link_args("linux", "gnu", &["f".to_string()], &dir).unwrap();
        assert!(args[0].starts_with("-Wl,--dynamic-list="));
        let text = std::fs::read_to_string(dir.join("exports.dynlist")).unwrap();
        assert_eq!(text, "{\n  f;\n};\n");
    }

    #[test]
    fn windows_writes_a_module_definition() {
        let dir = scratch("windows");
        let args = export_link_args("windows", "msvc", &["f".to_string()], &dir).unwrap();
        assert!(args[0].starts_with("/DEF:"));
        let text = std::fs::read_to_string(dir.join("exports.def")).unwrap();
        assert_eq!(text, "EXPORTS\n    f\n");
    }
}
