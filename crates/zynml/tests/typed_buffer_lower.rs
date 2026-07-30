//! ML roadmap Phase 0.3 — typed aligned buffer.
//!
//! `Ptr<T>` + `alloc<T>` / `free` + `vload_*` / `vstore_*` let a kernel
//! own a real contiguous typed buffer in pure ZynML. This proves the
//! source forms lower to the right HIR: `alloc<T>(n)` → sized
//! `Intrinsic::Malloc`, `free(p)` → `Intrinsic::Free`, and the vector
//! load/store to `VectorLoad` / `VectorStore` — no ZRTL `List<T>`, no FFI.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::hir::{HirCallable, HirFunction, HirInstruction, HirType, Intrinsic};
use zyntax_compiler::HirModule;
use zyntax_embed::ZyntaxRuntime;

fn lower(source: &str) -> HirModule {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<typed_buffer_lower>")
        .expect("source should parse");
    let rt = ZyntaxRuntime::new().expect("runtime");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    rt.lower_typed_program(program, builtins)
        .expect("typed program should lower to HIR")
}

fn func<'m>(module: &'m HirModule, name: &str) -> &'m HirFunction {
    module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(name))
        .unwrap_or_else(|| panic!("function `{name}` should be lowered"))
}

fn insts(f: &HirFunction) -> impl Iterator<Item = &HirInstruction> {
    f.blocks.values().flat_map(|b| b.instructions.iter())
}

fn has_intrinsic_call(f: &HirFunction, want: Intrinsic) -> bool {
    insts(f).any(|i| {
        matches!(
            i,
            HirInstruction::Call { callee: HirCallable::Intrinsic(got), .. } if *got == want
        )
    })
}

/// A buffer round-trip kernel lowers to malloc / vector store / vector
/// load / free — and the `alloc_i8` result is typed `Ptr<i8>`.
#[test]
fn typed_buffer_roundtrip_lowers_to_malloc_vecmem_free() {
    let module = lower(
        r#"
        def buf(v: i8x16): i8 {
            let p: Ptr<i8> = alloc_i8(16)
            vstore_i8x16(p, v)
            let w = vload_i8x16(p)
            free(p)
            return 0
        }
        "#,
    );
    let f = func(&module, "buf");

    assert!(
        has_intrinsic_call(f, Intrinsic::Malloc),
        "alloc<i8>(16) should lower to a Malloc call"
    );
    assert!(
        has_intrinsic_call(f, Intrinsic::Free),
        "free(p) should lower to a Free call"
    );
    assert!(
        insts(f).any(|i| matches!(i, HirInstruction::VectorStore { .. })),
        "vstore_i8x16 should lower to VectorStore"
    );
    assert!(
        insts(f).any(|i| matches!(i, HirInstruction::VectorLoad { .. })),
        "vload_i8x16 should lower to VectorLoad"
    );

    // `alloc<i8>` result value is a typed pointer `Ptr<i8>`.
    let malloc_result_is_ptr_i8 = insts(f).any(|i| match i {
        HirInstruction::Call {
            result: Some(r),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            ..
        } => matches!(
            f.values.get(r).map(|v| &v.ty),
            Some(HirType::Ptr(inner)) if matches!(**inner, HirType::I8)
        ),
        _ => false,
    });
    assert!(
        malloc_result_is_ptr_i8,
        "alloc<i8> must produce a Ptr<i8>-typed value"
    );
}
