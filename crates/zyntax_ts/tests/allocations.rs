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

#[test]
#[ignore = "allocation profile; run alone with --ignored --nocapture"]
fn lowered_project_cache_allocations() {
    use zyntax_embed::{TieredConfig, TieredRuntime};

    fn totals() -> (u64, u64, u64) {
        [
            Phase::Other,
            Phase::Parse,
            Phase::Runtime,
            Phase::Register,
            Phase::Lower,
            Phase::Execute,
            Phase::Optimize,
            Phase::Scratch,
            Phase::HirClone,
            Phase::Decode,
            Phase::Cranelift,
            Phase::Llvm,
            Phase::Osr,
            Phase::InterpBody,
        ]
        .into_iter()
        .map(allocation_audit::for_phase)
        .fold((0, 0, 0), |sum, stats| {
            (
                sum.0 + stats.allocations,
                sum.1 + stats.reallocations,
                sum.2 + stats.requested_bytes,
            )
        })
    }

    let directory = tempfile::tempdir().unwrap();
    let entry = directory.path().join("entry.ts");
    let cache = directory.path().join("cache");
    let mut source = String::new();
    for index in 0..1000 {
        source.push_str(&format!(
            "function f{index}(x:number):number{{const a=[x,x+1];return a[0]+a[1];}}\n"
        ));
    }
    source.push_str("const answer=f0(20);\n");
    std::fs::write(&entry, source).unwrap();
    for pass in [
        "uncached",
        "checked-store",
        "checked-hit",
        "hir-store",
        "hir-hit",
    ] {
        let mut runtime = TieredRuntime::new(TieredConfig {
            baseline_threshold: u32::MAX,
            enable_osr: false,
            ..Default::default()
        })
        .unwrap();
        let before = totals();
        let start = std::time::Instant::now();
        let project = if pass == "uncached" {
            zyntax_ts::register_runtime(&mut runtime).unwrap();
            zyntax_ts::load_project(&entry, Default::default()).unwrap()
        } else if pass.starts_with("checked") {
            zyntax_ts::register_runtime(&mut runtime).unwrap();
            zyntax_ts::load_project_cached(&entry, Default::default(), &cache)
                .unwrap()
                .0
        } else {
            let (project, status) = zyntax_ts::load_project_for_runtime(
                &entry,
                Default::default(),
                &cache,
                &mut runtime,
            )
            .unwrap();
            assert_eq!(
                status,
                if pass == "hir-store" {
                    zyntax_ts::CacheStatus::Stored
                } else {
                    zyntax_ts::CacheStatus::Hit
                }
            );
            project
        };
        let loaded = start.elapsed();
        let (program, resolver) = project.into_parts();
        runtime.add_compiled_import_resolver(resolver);
        runtime.enter_only_through_entry_points();
        runtime.compile_typed_program(program).unwrap();
        let elapsed = start.elapsed();
        let after = totals();
        println!(
            "{pass}: load_ms={:.3}, total_ms={:.3}, allocations={}, reallocations={}, requested_bytes={}",
            loaded.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0,
            after.0 - before.0,
            after.1 - before.1,
            after.2 - before.2
        );
        runtime.stop();
    }
}
