//! Loop-boundary hot reload: a loop already running when the edit
//! lands finishes in the edited code.
//!
//! The loop is entered once and never called again, so the call
//! boundary can never deliver the edit — only an OSR transfer into a
//! resume point compiled from the edited body can. The final value
//! tells the story: iterations before the transfer step by 1, after it
//! by 3, so a strictly-between result proves the migration happened
//! mid-flight.

use zynml::ZYNML_GRAMMAR;
use zyntax_embed::{LanguageGrammar, TieredConfig, TieredRuntime};

fn source(step: i64) -> String {
    format!(
        "def slow(n: i64): i64 {{\n    let mut acc: i64 = 0\n    let mut i: i64 = 0\n    \
         while i < n {{\n        acc = acc + {step}\n        i = i + 1\n    }}\n    \
         return acc\n}}\n"
    )
}

#[test]
fn a_running_loop_finishes_in_the_edited_code() {
    const N: i64 = 2_000_000_000;

    let mut config = TieredConfig::development();
    config.enable_osr = true;
    config.enable_hot_reload = true;
    // Keep call-count promotion out of the picture: the loop's owner is
    // called exactly once, and the edit must arrive through its probes.
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;

    let mut rt = TieredRuntime::new(config).expect("runtime should start");
    let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("grammar should compile");
    rt.register_grammar("x", grammar);
    rt.load_module("x", &source(1)).expect("v1 should compile");

    let entry = rt
        .function_pointer("slow")
        .expect("slow should have an entry pointer");
    // SAFETY: `slow` was compiled with signature (i64) -> i64, and the
    // code stays mapped for the life of the runtime.
    let slow: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(entry) };

    let worker = std::thread::spawn(move || slow(N));

    // Let the loop get well underway, then edit it under its feet.
    std::thread::sleep(std::time::Duration::from_millis(150));
    // Asserting before the join would tear the runtime down under the
    // running loop on failure; collect first, judge after.
    let report = rt.reload_module_source("x", &source(3)).expect("reload");
    let result = worker.join().expect("loop should complete");
    assert!(report.reloaded.contains(&"slow".to_string()), "{report:?}");
    assert!(
        report.resume_published.contains(&"slow".to_string()),
        "the edited loop must get a resume point: {report:?}"
    );
    assert!(
        result > N,
        "no iteration ran the edited step; the loop never migrated (result = {result})"
    );
    assert!(
        result < 3 * N,
        "every iteration ran the edited step; the transfer was not mid-loop (result = {result})"
    );
    // The loop ran to completion: k iterations at step 1, N-k at step 3
    // for some 0 < k < N, so the result is exact, not corrupted.
    assert_eq!(
        (result - N) % 2,
        0,
        "result {result} fits no split of the two steps"
    );
}

/// An edit that changes the loop's live-in shape cannot migrate the
/// running frame; the reload must fall back to the call boundary and
/// say so, and the running loop must still complete correctly on the
/// old code.
#[test]
fn a_layout_changing_edit_falls_back_to_the_call_boundary() {
    const N: i64 = 300_000_000;

    let mut config = TieredConfig::development();
    config.enable_osr = true;
    config.enable_hot_reload = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;

    let mut rt = TieredRuntime::new(config).expect("runtime should start");
    let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("grammar should compile");
    rt.register_grammar("x", grammar);
    rt.load_module("x", &source(1)).expect("v1 should compile");

    let entry = rt
        .function_pointer("slow")
        .expect("slow should have an entry pointer");
    // SAFETY: as above.
    let slow: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(entry) };
    let worker = std::thread::spawn(move || slow(N));

    std::thread::sleep(std::time::Duration::from_millis(50));
    // The edited loop carries an extra loop-carried variable — a
    // different frame than the running probe writes.
    let edited = "def slow(n: i64): i64 {\n    let mut acc: i64 = 0\n    let mut extra: i64 = 0\n    let mut i: i64 = 0\n    \
                  while i < n {\n        acc = acc + 3\n        extra = extra + acc\n        i = i + 1\n    }\n    \
                  return acc + (extra - extra)\n}\n";
    let report = rt.reload_module_source("x", edited).expect("reload");
    let result = worker.join().expect("loop should complete");
    assert!(report.reloaded.contains(&"slow".to_string()), "{report:?}");
    assert!(
        !report.resume_published.contains(&"slow".to_string()),
        "a layout-changing edit must not publish a resume point: {report:?}"
    );
    assert!(
        report
            .resume_fell_back
            .iter()
            .any(|(name, _)| name == "slow"),
        "the fallback must be reported: {report:?}"
    );
    assert_eq!(
        result, N,
        "the running loop completes on the code it started with"
    );
}
