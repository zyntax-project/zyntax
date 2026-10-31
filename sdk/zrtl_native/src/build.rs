//! For build scripts: native executable exports, startup text placement
//! and deferred static initialisers.
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

/// Link the package's executables so that the loader runs none of LLVM's
/// static initialisers: the runtime runs them before its first use of
/// LLVM (`zyntax_compiler`'s LLVM tier), and a process that never gets
/// there skips them. On Linux a linker script moves the initialiser
/// pointers of LLVM's archives out of `.init_array` into a section of
/// their own. On macOS, where every static initialiser of such an
/// executable must be LLVM's, the link goes through a small linker built
/// here with the build's `rustc`, which runs `ld` and retypes the
/// output's initialiser offsets as regular data. Setting
/// `ZYNTAX_KEEP_LINKER_LAYOUT` links as usual.
pub fn emit_deferred_initializers(dir: &Path) -> io::Result<()> {
    println!("cargo:rerun-if-env-changed=ZYNTAX_KEEP_LINKER_LAYOUT");
    let os = std::env::var("CARGO_CFG_TARGET_OS")
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    if std::env::var_os("ZYNTAX_KEEP_LINKER_LAYOUT").is_some() {
        return Ok(());
    }
    if os == "linux" {
        // The runtime reads the section back from a 64-bit little-endian
        // image; any other runs LLVM's initialisers at launch as before.
        let little = std::env::var("CARGO_CFG_TARGET_ENDIAN").as_deref() == Ok("little");
        let wide = std::env::var("CARGO_CFG_TARGET_POINTER_WIDTH").as_deref() == Ok("64");
        if little && wide {
            for arg in deferred_initializer_link_args(&os, dir)? {
                println!("cargo:rustc-link-arg-bins={arg}");
            }
        }
        return Ok(());
    }
    if os != "macos" || !cfg!(target_os = "macos") {
        return Ok(());
    }
    let source = dir.join("zyntax_ld.rs");
    std::fs::write(&source, include_str!("defer_initializers_ld.rs"))?;
    let linker = dir.join("zyntax-ld");
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let built = std::process::Command::new(rustc)
        .args(["--edition", "2024", "-O", "--crate-name", "zyntax_ld", "-o"])
        .arg(&linker)
        .arg(&source)
        .status()?;
    if !built.success() {
        return Err(io::Error::other("building the deferring linker failed"));
    }
    println!("cargo:rustc-link-arg-bins=--ld-path={}", linker.display());
    Ok(())
}

/// The section a Linux link moves LLVM's initialiser pointers to, which
/// the runtime reads back.
pub const DEFERRED_INITIALIZERS_SECTION: &str = ".zyntax_llvm_init";

/// The linker arguments that move LLVM's initialiser pointers out of
/// `.init_array`, for an ELF target; empty for others. LLVM's objects
/// arrive bundled in llvm-sys's rlib, or in LLVM's own archives when
/// linked unbundled. Placed ahead of `.init_array` in the script, the
/// section claims those pointers first.
pub fn deferred_initializer_link_args(target_os: &str, dir: &Path) -> io::Result<Vec<String>> {
    if target_os != "linux" {
        return Ok(Vec::new());
    }
    let path = dir.join("deferred_initializers.ld");
    let sections = ".init_array .init_array.* .ctors .ctors.*";
    std::fs::write(
        &path,
        format!(
            "SECTIONS {{\n  {DEFERRED_INITIALIZERS_SECTION} : {{\n    KEEP(*libllvm_sys-*.rlib:*({sections}))\n    KEEP(*libLLVM*.a:*({sections}))\n  }}\n}}\nINSERT BEFORE .init_array;\n"
        ),
    )?;
    Ok(vec![
        "-Xlinker".into(),
        format!("--script={}", path.display()),
    ])
}

/// Group ELF startup text without changing constructor-array order or the
/// loader ABI. On macOS, link with chained fixups, which dyld applies to
/// a data page when it is first touched rather than to every page at
/// launch. Other targets retain their native linker's layout.
pub fn startup_link_args(target_os: &str, dir: &Path) -> io::Result<Vec<String>> {
    if target_os == "macos" {
        return Ok(vec!["-Wl,-fixup_chains".into()]);
    }
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
    fn startup_layout_leaves_other_targets_untouched() {
        let dir = scratch("startup_other_targets");
        for os in ["windows", "unknown", "freebsd", "android"] {
            assert!(startup_link_args(os, &dir).unwrap().is_empty());
        }
        assert!(!dir.join("native_startup.ld").exists());
    }

    #[test]
    fn macos_links_with_chained_fixups() {
        let dir = scratch("startup_macos");
        assert_eq!(
            startup_link_args("macos", &dir).unwrap(),
            vec!["-Wl,-fixup_chains".to_string()]
        );
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
