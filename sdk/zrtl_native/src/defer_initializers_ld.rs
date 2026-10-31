//! A linker for macOS executables whose static initialisers the runtime
//! runs itself: it runs `ld` with the arguments it was given, then
//! retypes the output's `__TEXT,__init_offsets` section as regular data,
//! so dyld runs none of them at launch, and signs the output again.
//! Built by `zrtl_native::build::emit_deferred_initializers` with the
//! build's own `rustc`, and named to the compiler driver with `--ld-path`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

const MH_MAGIC_64: u32 = 0xfeed_facf;
const LC_SEGMENT_64: u32 = 0x19;
const SECTION_TYPE: u32 = 0xff;
const S_INIT_FUNC_OFFSETS: u32 = 0x16;
const S_REGULAR: u32 = 0x0;

fn main() {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let status = match Command::new("ld").args(&args).status() {
        Ok(status) => status,
        Err(e) => {
            eprintln!("zyntax-ld: cannot run ld: {e}");
            exit(1);
        }
    };
    if !status.success() {
        exit(status.code().unwrap_or(1));
    }
    let out = args
        .windows(2)
        .find(|pair| pair[0] == "-o")
        .map(|pair| PathBuf::from(&pair[1]))
        .unwrap_or_else(|| PathBuf::from("a.out"));
    match retype(&out) {
        Ok(false) => {}
        Ok(true) => {
            let signed = Command::new("codesign")
                .args(["--sign", "-", "--force"])
                .arg(&out)
                .output();
            match signed {
                Ok(signed) if signed.status.success() => {}
                Ok(signed) => {
                    eprintln!(
                        "zyntax-ld: cannot sign {}: {}",
                        out.display(),
                        String::from_utf8_lossy(&signed.stderr)
                    );
                    exit(1);
                }
                Err(e) => {
                    eprintln!("zyntax-ld: cannot run codesign: {e}");
                    exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("zyntax-ld: {}: {e}", out.display());
            exit(1);
        }
    }
}

/// Retype the initialiser offsets of the 64-bit image at `path` as
/// regular data. False when the image has none, or is not one this
/// reads.
fn retype(path: &Path) -> std::io::Result<bool> {
    let mut bytes = std::fs::read(path)?;
    let word = |bytes: &[u8], at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
    };
    if word(&bytes, 0) != Some(MH_MAGIC_64) {
        return Ok(false);
    }
    let commands = word(&bytes, 16).unwrap_or(0);
    let mut at = 32;
    let mut flags_at = None;
    for _ in 0..commands {
        let (Some(cmd), Some(size)) = (word(&bytes, at), word(&bytes, at + 4)) else {
            return Ok(false);
        };
        if cmd == LC_SEGMENT_64
            && bytes
                .get(at + 8..at + 24)
                .is_some_and(|n| named(n, b"__TEXT"))
        {
            let sections = word(&bytes, at + 64).unwrap_or(0) as usize;
            for i in 0..sections {
                let section = at + 72 + i * 80;
                let flags = section + 64;
                if bytes
                    .get(section..section + 16)
                    .is_some_and(|n| named(n, b"__init_offsets"))
                    && word(&bytes, flags).is_some_and(|f| f & SECTION_TYPE == S_INIT_FUNC_OFFSETS)
                {
                    flags_at = Some(flags);
                }
            }
        }
        at += size as usize;
    }
    let Some(flags_at) = flags_at else {
        return Ok(false);
    };
    let flags = word(&bytes, flags_at).unwrap_or(0) & !SECTION_TYPE | S_REGULAR;
    bytes[flags_at..flags_at + 4].copy_from_slice(&flags.to_le_bytes());
    // A new file renamed over the old: the kernel keeps what it checked of
    // a signed file by the file, not by its contents.
    let staged = path.with_extension("zyntax-ld");
    std::fs::write(&staged, bytes)?;
    std::fs::set_permissions(&staged, std::fs::metadata(path)?.permissions())?;
    std::fs::rename(&staged, path)?;
    Ok(true)
}

/// Whether a NUL-padded 16-byte Mach-O name is `name`.
fn named(field: &[u8], name: &[u8]) -> bool {
    field.split(|&b| b == 0).next() == Some(name)
}
