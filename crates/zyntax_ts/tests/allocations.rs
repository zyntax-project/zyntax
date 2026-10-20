//! Reproduce frontend allocation counts without JIT or native-library allocations.
#![cfg(feature = "allocation-audit")]

use zyntax_compiler::allocation_audit::{self, Phase, Scope};
use zyntax_typed_ast::typed_ast::TypedDeclaration;

#[global_allocator]
static ALLOCATOR: allocation_audit::CountingAllocator = allocation_audit::CountingAllocator;

#[test]
#[ignore = "allocation profile; run alone with --ignored --nocapture"]
fn frontend_allocations() {
    let mut source = String::from("type Point = { x: number; y: number };\n");
    for i in 0..1000 {
        source.push_str(&format!(
            "function f{i}(x: number): number {{ const p: Point = {{y: x, x: x + 1}}; return p.x + p.y; }}\n"
        ));
    }
    for pass in ["cold", "warm"] {
        let before = allocation_audit::for_phase(Phase::Parse);
        let program = {
            let _phase = Scope::enter(Phase::Parse);
            zyntax_ts::parse_program(&source, "allocations.ts", Default::default()).unwrap()
        };
        let after = allocation_audit::for_phase(Phase::Parse);
        assert_eq!(
            program
                .declarations
                .iter()
                .filter(|d| matches!(
                    &d.node, TypedDeclaration::Function(f) if !f.is_external
                ))
                .count(),
            1001
        );
        println!(
            "{pass}: allocations={}, reallocations={}, requested_bytes={}",
            after.allocations - before.allocations,
            after.reallocations - before.reallocations,
            after.requested_bytes - before.requested_bytes
        );
    }
}
