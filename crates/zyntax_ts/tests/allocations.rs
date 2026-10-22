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
            1000
        );
        println!(
            "{pass}: allocations={}, reallocations={}, requested_bytes={}",
            after.allocations - before.allocations,
            after.reallocations - before.reallocations,
            after.requested_bytes - before.requested_bytes
        );
    }
}

#[test]
#[ignore = "allocation profile; run alone with --ignored --nocapture"]
fn project_cache_allocations() {
    let directory = tempfile::tempdir().unwrap();
    let entry = directory.path().join("entry.ts");
    let cache = directory.path().join("cache");
    let mut source = String::new();
    for index in 0..1000 {
        source.push_str(&format!(
            "function f{index}(x:number):number{{const a=[x,x+1];return a[0]+a[1];}}\n"
        ));
    }
    std::fs::write(&entry, source).unwrap();
    for pass in ["uncached", "store", "hit"] {
        let phase = if pass == "hit" {
            Phase::Decode
        } else {
            Phase::Parse
        };
        let before = allocation_audit::for_phase(phase);
        let start = std::time::Instant::now();
        let project = {
            let _scope = Scope::enter(phase);
            if pass == "uncached" {
                zyntax_ts::load_project(&entry, Default::default()).unwrap()
            } else {
                let (project, status) =
                    zyntax_ts::load_project_cached(&entry, Default::default(), &cache).unwrap();
                assert_eq!(
                    status,
                    if pass == "store" {
                        zyntax_ts::CacheStatus::Stored
                    } else {
                        zyntax_ts::CacheStatus::Hit
                    }
                );
                project
            }
        };
        let elapsed = start.elapsed();
        let after = allocation_audit::for_phase(phase);
        assert_eq!(project.module_count(), 1);
        println!(
            "{pass}: ms={:.3}, allocations={}, reallocations={}, requested_bytes={}",
            elapsed.as_secs_f64() * 1000.0,
            after.allocations - before.allocations,
            after.reallocations - before.reallocations,
            after.requested_bytes - before.requested_bytes,
        );
    }
}
