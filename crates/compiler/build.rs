//! Records which compiler this is, so an artifact of lowered HIR can say
//! which compiler produced it and be refused by any other.

use std::path::Path;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let src = Path::new(&manifest).join("src");
    println!("cargo:rerun-if-changed={}", src.display());

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    };
    let mut files = Vec::new();
    collect(&src, &mut files);
    files.sort();
    for file in files {
        feed(file.to_string_lossy().as_bytes());
        if let Ok(bytes) = std::fs::read(&file) {
            feed(&bytes);
        }
    }
    println!("cargo:rustc-env=ZYNTAX_COMPILER_BUILD_ID={hash:016x}");
}

fn collect(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}
