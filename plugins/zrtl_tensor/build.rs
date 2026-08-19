fn main() {
    // Apple's Accelerate reaches the AMX coprocessor, which no vector
    // instruction the compiler can emit gets to. It ships with the OS,
    // so this is a link line rather than a dependency.
    if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
