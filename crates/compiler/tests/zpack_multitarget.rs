//! Phase D — multi-target ZPack round trip.
//!
//! Writes a zpack with runtime sections for both a native triple and
//! `wasm32-unknown-unknown`, then verifies that
//! [`ZPack::read_runtime_bytes`] can pull either slice back out by
//! target triple. This is the universal extraction path the wasm
//! browser glue (Phase F) uses to fetch the wasm-module slice
//! without `dlopen`.
//!
//! Loader v1 (`ZPack::load`) is single-target — it only resolves the
//! runtime for the current platform via `dlopen`. The new accessor
//! here is what makes the format genuinely multi-target.

use std::io::Cursor;

use zyntax_compiler::zpack::{ZPack, ZPackManifest, ZPackWriter, ZPACK_VERSION};

const NATIVE_RUNTIME_BYTES: &[u8] = b"FAKE-NATIVE-DYLIB-PLACEHOLDER";
const WASM_RUNTIME_BYTES: &[u8] = b"\0asm\x01\x00\x00\x00FAKE-WASM-MODULE-PLACEHOLDER";

fn build_multitarget_pack() -> Vec<u8> {
    let manifest = ZPackManifest {
        version: ZPACK_VERSION,
        name: "multitarget-smoke".to_string(),
        package_version: "0.1.0".to_string(),
        source_language: "zynml".to_string(),
        ..Default::default()
    };

    let buf = Cursor::new(Vec::new());
    let mut writer = ZPackWriter::new(buf, manifest);
    writer
        .add_runtime_bytes("aarch64-apple-darwin", NATIVE_RUNTIME_BYTES)
        .expect("add native runtime");
    writer
        .add_runtime_bytes("wasm32-unknown-unknown", WASM_RUNTIME_BYTES)
        .expect("add wasm runtime");
    let cursor = writer.finish().expect("finish writer");
    cursor.into_inner()
}

#[test]
fn read_runtime_bytes_by_target() {
    let archive_bytes = build_multitarget_pack();

    // Pull native slice back out.
    let native = ZPack::read_runtime_bytes(Cursor::new(&archive_bytes), "aarch64-apple-darwin")
        .expect("read native bytes")
        .expect("native runtime present");
    assert_eq!(native.as_slice(), NATIVE_RUNTIME_BYTES);

    // Pull wasm slice back out — same archive, different target.
    let wasm = ZPack::read_runtime_bytes(Cursor::new(&archive_bytes), "wasm32-unknown-unknown")
        .expect("read wasm bytes")
        .expect("wasm runtime present");
    assert_eq!(wasm.as_slice(), WASM_RUNTIME_BYTES);

    // Missing target should be Ok(None), not an error.
    let missing = ZPack::read_runtime_bytes(Cursor::new(&archive_bytes), "x86_64-pc-windows-msvc")
        .expect("missing target should not error");
    assert!(missing.is_none(), "absent target slice should be None");
}

#[test]
fn manifest_records_each_target_added() {
    let archive_bytes = build_multitarget_pack();
    // Decode the zip directly so we can read the manifest without
    // hitting the dlopen path in `ZPack::load`.
    let mut archive = zip::ZipArchive::new(Cursor::new(&archive_bytes)).expect("open archive");
    let mut manifest_json = String::new();
    {
        use std::io::Read;
        let mut f = archive.by_name("manifest.json").expect("manifest present");
        f.read_to_string(&mut manifest_json).expect("read manifest");
    }
    let manifest: ZPackManifest = serde_json::from_str(&manifest_json).expect("parse manifest");
    assert!(manifest.targets.iter().any(|t| t == "aarch64-apple-darwin"));
    assert!(manifest
        .targets
        .iter()
        .any(|t| t == "wasm32-unknown-unknown"));
}
