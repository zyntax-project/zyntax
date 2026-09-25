//! Compile the protect frame, and the C frames its tests raise through.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/boundary.c");
    println!("cargo:rerun-if-changed=include/zrtl_native.h");
    println!("cargo:rerun-if-changed=tests/frames.c");
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("the manifest's directory"));
    let include = root.join("include");
    println!("cargo:include={}", include.display());
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        return;
    }
    cc::Build::new()
        .file(root.join("src/boundary.c"))
        .include(&include)
        .warnings(true)
        .compile("zrtl_native_boundary");
    // Linked only by the tests that name it.
    cc::Build::new()
        .file(root.join("tests/frames.c"))
        .include(&include)
        .cargo_metadata(false)
        .compile("zrtl_native_test_frames");
    let out = PathBuf::from(env::var("OUT_DIR").expect("the output directory"));
    println!("cargo:rustc-link-search=native={}", out.display());
}
