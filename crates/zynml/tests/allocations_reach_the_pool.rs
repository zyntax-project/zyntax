//! A compiled program's allocations arrive at the runtime's pools.
//!
//! `pool_alloc`'s own tests prove what a pool does with a block: that
//! it reuses one, poisons one it has released, keeps its classes apart.
//! Every one of them would pass against a pool that no compiled program
//! ever reaches, because they call it directly.
//!
//! What connects them to a program is the backends emitting a call to
//! `zyntax_alloc` for `Intrinsic::Malloc` instead of libc's `malloc`,
//! and that is a fact about the compiler, not about the pool. This
//! asserts it, so the two together mean something.
//!
//! **What the pool does not reach.** Only compiled code. The bytecode
//! interpreter serves `Intrinsic::Malloc` from an arena of its own, so
//! a program that never leaves tier 0 allocates without touching any of
//! this. The setup below compiles on the first call for that reason: a
//! loop inside one invocation of `main` never makes the function hot,
//! and a test left on the defaults reports that compiled allocations do
//! not arrive — which is true, of code that was never compiled.

use std::sync::Mutex;
use zynml::{ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::pool_alloc::pooled_allocation_count;
use zyntax_compiler::profiling::ProfileConfig;
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::{LanguageGrammar, ZyntaxRuntime, ZyntaxValue};

/// The counter is one number for the whole process, so two tests
/// reading it at once are each measuring the other.
static COUNTING: Mutex<()> = Mutex::new(());

/// Run `main` with the first call compiled, and hand back its answer.
fn run(src: &str) -> ZyntaxValue {
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
        _ => Ok(None),
    }));
    let g = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).expect("grammar");
    rt.register_grammar("zynml", g);
    rt.load_module("zynml", src).expect("should compile");
    let mut cfg = TieredConfig::default();
    cfg.profile_config = ProfileConfig {
        warm_threshold: 0,
        hot_threshold: u32::MAX as u64,
        ..Default::default()
    };
    rt.install_interp_jit_with(cfg).expect("install jit");
    rt.call_function_raw("main", vec![]).expect("should run")
}

/// A compiled program that allocates moves the counter.
///
/// 24 bytes on purpose: three words, the size a tree node takes, and
/// well inside what a pool serves. Above the cap the request goes to
/// libc and would prove the opposite of what is wanted.
///
/// The blocks are all held at once and released at the end, rather
/// than taken and dropped one per iteration. A pair that is allocated,
/// written, read back and freed inside one block is dead by
/// construction, and `scalar_replace_alloc` removes it: the first
/// version of this test compiled down to an integer sum with no
/// allocation in it at all, and correctly reported that nothing
/// reached the pool. Keeping them live is what makes them exist.
#[test]
fn a_compiled_program_is_served_by_the_pool() {
    let _serialised = COUNTING.lock().unwrap_or_else(|e| e.into_inner());
    let before = pooled_allocation_count();
    let got = run(r#"
import prelude
import simd

def main(): i64 {
    // Somewhere to keep the addresses, so every block is live at once
    // and none of them is dead where it is taken.
    let raw: Ptr<i8> = alloc_i8(512)
    let held: Ptr<i64> = raw as Ptr<i64>
    let mut i: i64 = 0
    while i < 64 {
        let p: Ptr<i8> = alloc_i8(24)
        let q: Ptr<i64> = p as Ptr<i64>
        q[0] = i
        held[i] = q as i64
        i = i + 1
    }
    let mut total: i64 = 0
    let mut j: i64 = 0
    while j < 64 {
        let r: Ptr<i64> = held[j] as Ptr<i64>
        total = total + r[0]
        free(held[j] as Ptr<i8>)
        j = j + 1
    }
    free(raw)
    return total
}
"#);
    // Asserted first: without it, "the pool served nothing" cannot be
    // told apart from "the program did not run".
    assert_eq!(
        got,
        ZyntaxValue::Int(2016),
        "the program did not produce its answer, so nothing can be \
         concluded about where its allocations went"
    );

    let served = pooled_allocation_count() - before;
    assert!(
        served >= 64,
        "the program made 64 allocations and the pools served {served}. \
         Either a backend is emitting libc's malloc again, or the \
         intrinsic is not reaching one — the pool's own tests cannot \
         tell the difference."
    );
}

/// And a program that allocates nothing does not move it meaningfully,
/// so the assertion above is about the program rather than about
/// whatever the harness does around it.
#[test]
fn a_program_that_allocates_nothing_leaves_it_alone() {
    let _serialised = COUNTING.lock().unwrap_or_else(|e| e.into_inner());
    // Warm whatever setting up a runtime costs, so the measured window
    // holds only the second program.
    let _ = run("import prelude\ndef main(): i64 { return 1 }");

    let before = pooled_allocation_count();
    let got = run("import prelude\ndef main(): i64 { return 2 }");
    assert_eq!(got, ZyntaxValue::Int(2));
    let served = pooled_allocation_count() - before;

    assert!(
        served < 64,
        "a program that allocates nothing was served {served} times, so \
         the counter is dominated by the harness and the test above \
         would pass whatever the program did"
    );
}
