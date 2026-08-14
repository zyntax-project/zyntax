//! Reports how much of the ZynML grammar the parsing machine can express.
use zynml::{Grammar2, ZYNML_GRAMMAR};

fn main() {
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let ir = g.grammar_ir();
    let t = std::time::Instant::now();
    let program = zyn_peg::runtime2::machine::compile(ir);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "compiled {} rules to {} instructions in {ms:.2} ms\ncoverage {:.1}% ({} unsupported)",
        program.names.len(),
        program.len(),
        program.coverage(),
        program.unsupported.len()
    );
    for name in program.unsupported.iter().take(10) {
        eprintln!("  unsupported: {name}");
    }
}
