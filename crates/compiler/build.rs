//! Fingerprints the compiler, serialized types and lowering dependencies.

mod build_fingerprint;

use std::path::Path;

fn main() -> std::io::Result<()> {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR").unwrap();
    let root = Path::new(&manifest).join("../..");
    let mut watched = String::new();
    for input in build_fingerprint::watched_inputs(&root) {
        let path = input.to_str().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "build inputs must have UTF-8 paths",
            )
        })?;
        println!("cargo:rerun-if-changed={path}");
        watched.push_str(path);
        watched.push('\n');
    }
    // Build-time cross-lowering and the target runtime share this source identity.
    // Target pointer size is checked separately by snapshots.
    let id = build_fingerprint::fingerprint(&root)?;
    let hex: String = id.iter().map(|byte| format!("{byte:02x}")).collect();
    println!("cargo:rustc-env=ZYNTAX_COMPILER_BUILD_ID={hex}");
    let output = std::env::var_os("OUT_DIR").unwrap();
    std::fs::write(Path::new(&output).join("build_id.bin"), id)?;
    std::fs::write(Path::new(&output).join("build_inputs.txt"), watched)
}
