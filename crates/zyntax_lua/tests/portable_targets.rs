//! The library compiles to machine code for every target a release
//! carries that this build's Cranelift can compile for, whichever machine
//! runs the test: a build compiles it only for the target it builds for.

const TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
];

#[test]
fn the_library_compiles_for_every_target() {
    let snapshot = zyntax_embed::Snapshot::load(zyntax_embed::include_snapshot!("lua")).unwrap();
    let symbols: Vec<_> = [
        zrtl_io::static_plugin(),
        zrtl_string::static_plugin(),
        zrtl_math::static_plugin(),
    ]
    .iter()
    .flat_map(zyntax_embed::static_plugin_symbols)
    .collect();
    let mut compiled = 0;
    for module in snapshot.modules().unwrap() {
        let Some(hir) = module.hir() else {
            continue;
        };
        for triple in TARGETS
            .iter()
            .filter(|t| zyntax_compiler::portable_code::compiles_for(t))
        {
            for probes in [false, true] {
                let code = zyntax_compiler::portable_code::compile_snapshot_module(
                    hir, triple, probes, &symbols,
                )
                .unwrap_or_else(|e| panic!("{} for {triple}: {e}", module.module_name()));
                assert!(!code.functions.is_empty(), "{triple}: no functions");
                compiled += 1;
            }
        }
    }
    assert!(compiled > 0, "the snapshot carries no lowered module");
}
