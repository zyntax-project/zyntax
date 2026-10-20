//! The same typed reference/alias program runs before and after native promotion.

use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{TieredRuntime, ZyntaxValue};

#[test]
fn typed_object_aliases_agree_across_tiers_without_collection() {
    let source = r#"
        type Point = { x: number; y: number };
        function make(x: number): Point { return { y: 2, x: x }; }
        function score(x: number): number {
            const p: Point = make(x);
            const q: Point = p;
            q.x += 1;
            return p.x * 10 + q.y;
        }
    "#;
    let program = zyntax_ts::parse_program(source, "tiers.ts", Default::default()).unwrap();
    let config = TieredConfig {
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    runtime.compile_typed_program(program).unwrap();
    assert_eq!(runtime.statistics().baseline_functions, 0);
    assert_eq!(runtime.statistics().optimized_functions, 0);
    assert_eq!(
        runtime
            .call_raw("ts$fn$score", &[ZyntaxValue::Float(3.0)])
            .unwrap(),
        ZyntaxValue::Float(42.0)
    );
    runtime
        .optimize_function("ts$fn$score", OptimizationTier::Baseline)
        .unwrap();
    assert_eq!(
        runtime
            .call_raw("ts$fn$score", &[ZyntaxValue::Float(5.0)])
            .unwrap(),
        ZyntaxValue::Float(62.0)
    );
    #[cfg(feature = "llvm-backend")]
    {
        runtime
            .optimize_function("ts$fn$score", OptimizationTier::Optimized)
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while runtime.statistics().optimized_functions == 0 {
            assert!(std::time::Instant::now() < deadline, "LLVM did not install");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(
            runtime
                .call_raw("ts$fn$score", &[ZyntaxValue::Float(7.0)])
                .unwrap(),
            ZyntaxValue::Float(82.0)
        );
    }
    assert!(!zyntax_compiler::collector::is_enabled());
    runtime.stop();
}
