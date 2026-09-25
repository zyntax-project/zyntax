//! What creating and dropping a runtime costs, with no guest work.
//!
//! The kernels no longer leak per iteration, but a pass over the suite
//! still grows. This isolates the runtime itself: the same trivial
//! program, loaded into a fresh runtime and dropped, many times over.

mod resident;

use zynml::ZynML;

#[test]
fn creating_and_dropping_runtimes() {
    const SRC: &str = "def main(): i64 { return 1 }";
    // Pay the one-time costs first.
    for _ in 0..3 {
        let mut rt = ZynML::new().unwrap();
        rt.load_source(SRC).unwrap();
        let _: i64 = rt.call_with_result("main").unwrap();
    }
    let base = resident::mb();
    const N: usize = 40;
    for _ in 0..N {
        let mut rt = ZynML::new().unwrap();
        rt.load_source(SRC).unwrap();
        let _: i64 = rt.call_with_result("main").unwrap();
    }
    let after = resident::mb();
    let grew = after as i64 - base as i64;
    println!("\n  {N} runtimes: {base} MB -> {after} MB, grew {grew} MB");
    println!("  per runtime: {:.2} MB", grew as f64 / N as f64);
}
