use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub const SOURCE_ROOTS: &[&str] = &[
    "crates/compiler",
    "crates/typed_ast",
    "crates/zyntax_embed",
    "crates/pattern_engine",
    "crates/passes/normalization",
    "crates/passes/algebraic_effects",
    "crates/passes/krio_adapter",
    "sdk/zrtl",
];

pub const BUILD_INPUTS: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "crates/compiler/build.rs",
    "crates/compiler/build_fingerprint.rs",
];

pub fn watched_inputs(root: &Path) -> Vec<PathBuf> {
    SOURCE_ROOTS
        .iter()
        .flat_map(|source| {
            [
                root.join(source).join("src"),
                root.join(source).join("Cargo.toml"),
            ]
        })
        .chain(BUILD_INPUTS.iter().map(|input| root.join(input)))
        .collect()
}

pub fn fingerprint(root: &Path) -> io::Result<[u8; 32]> {
    let mut files: Vec<_> = BUILD_INPUTS.iter().map(PathBuf::from).collect();
    for source in SOURCE_ROOTS {
        let source = Path::new(source);
        files.push(source.join("Cargo.toml"));
        collect(root, &source.join("src"), &mut files)?;
    }
    files.sort();
    let mut hash = Sha256::new();
    for file in files {
        let name = file
            .to_str()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "build inputs must have UTF-8 paths",
                )
            })?
            .replace('\\', "/");
        feed(&mut hash, name.as_bytes());
        feed(&mut hash, &std::fs::read(root.join(file))?);
    }
    Ok(hash.finalize().into())
}

fn feed(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn collect(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in std::fs::read_dir(root.join(directory))? {
        let entry = entry?;
        let path = directory.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            collect(root, &path, files)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(())
}
