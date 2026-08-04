//! Can a tier-0 probe site be located in the finalized machine code?
//!
//! The arm-byte check costs 13–80% because the load is an optimization
//! barrier inside the loop. A patchable site would cost nothing when
//! unarmed, but only if codegen can hand back the code offset of a
//! specific instruction. Cranelift's source-location table is the
//! candidate: tag the instruction, then look up where it landed.

mod common;

use common::counted_loop;
use zyntax_compiler::cranelift_backend::CraneliftBackend;

/// A tagged probe emits one or more source-location ranges, and each must
/// resolve to an offset inside the emitted function. Patching needs the
/// arm check specifically, which is the lowest offset because it is
/// emitted first — the later ranges are the probe-call and dispatch
/// blocks, which the block layout places elsewhere.
#[test]
fn a_tagged_probe_is_locatable_in_the_code() {
    let (function, _header) = counted_loop();
    let func_id = function.id;

    let mut backend = CraneliftBackend::new().expect("backend");
    backend.set_compile_tier(0);
    backend.set_compile_bead_id(0xBEAD);
    backend
        .compile_function(func_id, &function)
        .expect("tier-0 compile");

    let sites = backend.take_probe_sites();
    assert!(
        !sites.is_empty(),
        "a loop header should record at least one probe site"
    );

    let code_len = backend.last_code_len().expect("code length");
    assert!(code_len > 0, "the function should have emitted code");
    for (site_key, offset) in &sites {
        assert!(
            (*offset as usize) < code_len,
            "probe site 0x{site_key:x} offset {offset} should fall inside the \
             {code_len}-byte function"
        );
    }

    // Offsets arrive sorted, so the first is the arm check.
    let mut sorted = sites.clone();
    sorted.sort_by_key(|(_, off)| *off);
    assert_eq!(sites, sorted, "probe sites should come back in code order");
}
