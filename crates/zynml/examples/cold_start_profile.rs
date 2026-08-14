//! Fresh-process cold-start probe for serverless deployment builds.
//!
//! Run the release example binary directly for meaningful numbers.
//! Set ZYNML_COLD_START_BUDGET_MS to fail when process-internal startup
//! exceeds a deployment's chosen budget.

use std::time::Instant;
use zynml::ZynML;

const SOURCE: &str = r#"
import prelude

fn main() -> i64 {
    return sqrt(1.0) as i64
}
"#;

fn main() {
    let process_start = Instant::now();

    let setup_start = Instant::now();
    let mut runtime = ZynML::new().expect("create ZynML runtime");
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

    let compile_start = Instant::now();
    runtime
        .load_source(SOURCE)
        .expect("compile cold-start probe");
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;

    let execute_start = Instant::now();
    let result: i64 = runtime
        .call_with_result("main")
        .expect("execute cold-start probe");
    let execute_ms = execute_start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(result, 1);

    let total_ms = process_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{{\"setup_ms\":{setup_ms:.3},\"compile_ms\":{compile_ms:.3},\
         \"execute_ms\":{execute_ms:.3},\"total_ms\":{total_ms:.3}}}"
    );

    if let Some(budget_ms) = std::env::var("ZYNML_COLD_START_BUDGET_MS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
    {
        assert!(
            total_ms <= budget_ms,
            "cold start {total_ms:.3} ms exceeded budget {budget_ms:.3} ms"
        );
    }
}
