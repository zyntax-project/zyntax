use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../..")
        .canonicalize()
        .unwrap();
    let mut files = vec![
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
        root.join("crates/zyntax_ts/build.rs"),
    ];
    // Checked types, snapshot encoding and the native primitive ABI affect reuse.
    for path in [
        "crates/zyntax_ts",
        "crates/typed_ast",
        "crates/zyntax_embed",
        "plugins/zrtl_io",
        "plugins/zrtl_string",
    ] {
        let dir = root.join(path);
        files.push(dir.join("Cargo.toml"));
        let src = dir.join("src");
        println!("cargo:rerun-if-changed={}", src.display());
        collect(&src, &mut files);
    }
    files.sort();
    let mut hash = Sha256::new();
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        feed(
            &mut hash,
            file.strip_prefix(&root)
                .unwrap()
                .to_str()
                .unwrap()
                .as_bytes(),
        );
        feed(&mut hash, &std::fs::read(file).unwrap());
    }
    for name in [
        "TARGET",
        "PROFILE",
        "OPT_LEVEL",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTC",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
        feed(&mut hash, name.as_bytes());
        feed(
            &mut hash,
            std::env::var(name).unwrap_or_default().as_bytes(),
        );
    }
    let mut features: Vec<_> = std::env::vars()
        .filter(|(name, _)| name.starts_with("CARGO_FEATURE_"))
        .collect();
    features.sort();
    for (name, value) in features {
        feed(&mut hash, name.as_bytes());
        feed(&mut hash, value.as_bytes());
    }
    let rustc = std::process::Command::new(std::env::var_os("RUSTC").unwrap())
        .arg("-vV")
        .output()
        .unwrap();
    assert!(rustc.status.success());
    feed(&mut hash, &rustc.stdout);
    println!(
        "cargo:rustc-env=ZYNTAX_TS_CACHE_BUILD_ID={:x}",
        hash.finalize()
    );
}

fn feed(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}
