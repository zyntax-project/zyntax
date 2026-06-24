//! Shared test fixtures for krio_adapter integration tests.
//!
//! Each `tests/stage_*.rs` file pulls fixtures from here so the
//! per-stage tests share a known-good input shape. The fixtures
//! mirror what real ZynML async-fn lowering produces (a single
//! Intrinsic::Await call with surrounding plain instructions),
//! so the assertions translate to real-pipeline behavior.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use zyntax_compiler::hir::{
    HirBlock, HirCallable, HirFunction, HirFunctionSignature, HirId, HirInstruction, HirModule,
    HirTerminator, HirType, HirValue, HirValueKind, Intrinsic,
};
use zyntax_typed_ast::InternedString;

/// What `make_async_function_with_one_await` returns: the function
/// itself plus the SSA values planted in the body so tests can refer
/// to them when asserting save/load.
pub struct AsyncFnFixture {
    pub function: HirFunction,
    /// SSA value that's defined before the await and used after — the
    /// value the captures lift must save/load.
    pub live_across: HirId,
    /// SSA value for the result of the await call (i32). Used by the
    /// post-await code (e.g. as the function's return).
    pub await_result: HirId,
}

/// Build the canonical "captures-lift" test shape:
///
/// ```pseudo
/// async fn aw(input: i32) -> i32 {   // entry block:
///     let x = input + 1              //   [Binary(Add, input, 1) → live_across]
///     let r = await foo()            //   [Call(Function), Call(Intrinsic::Await)]
///     return x + r                   //   [Binary(Add, live_across, await_result), Return]
/// }
/// ```
///
/// `live_across` (`x`) is defined before the await and used after —
/// it's the canonical value that needs spilling. `await_result` (`r`)
/// is defined by the await call. Both end up in the function's
/// values map; `live_across` is in the function's `live_out` set.
pub fn make_async_function_with_one_await() -> AsyncFnFixture {
    let mut sig = HirFunctionSignature {
        params: vec![],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: true,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };
    sig.is_async = true;

    let mut function = HirFunction::new(InternedString::new_global("aw"), sig);
    function.is_external = false;

    // ── SSA values ──
    let input = HirId::new();
    let const_one = HirId::new();
    let live_across = HirId::new(); // x = input + 1
    let foo_result = HirId::new(); // result of foo()
    let await_result = HirId::new(); // result of await
    let return_val = HirId::new(); // x + r

    let foo_id = HirId::new(); // a sync function "foo" — we don't need its body

    for (id, ty) in [
        (input, HirType::I32),
        (const_one, HirType::I32),
        (live_across, HirType::I32),
        (foo_result, HirType::I32),
        (await_result, HirType::I32),
        (return_val, HirType::I32),
    ] {
        let kind = if id == const_one {
            HirValueKind::Constant(zyntax_compiler::hir::HirConstant::I32(1))
        } else {
            HirValueKind::Instruction
        };
        function.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind,
                uses: HashSet::new(),
                span: None,
            },
        );
    }

    // ── Entry block ──
    let entry_id = HirId::new();
    let mut entry = HirBlock {
        id: entry_id,
        label: Some(InternedString::new_global("entry")),
        phis: vec![],
        instructions: vec![],
        terminator: HirTerminator::Return {
            values: vec![return_val],
        },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![],
    };
    // x = input + 1
    entry.instructions.push(HirInstruction::Binary {
        op: zyntax_compiler::hir::BinaryOp::Add,
        result: live_across,
        ty: HirType::I32,
        left: input,
        right: const_one,
    });
    // foo_result = foo()
    entry.instructions.push(HirInstruction::Call {
        result: Some(foo_result),
        callee: HirCallable::Function(foo_id),
        args: vec![],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    // await_result = await foo_result  (Intrinsic::Await is the suspension)
    entry.instructions.push(HirInstruction::Call {
        result: Some(await_result),
        callee: HirCallable::Intrinsic(Intrinsic::Await),
        args: vec![foo_result],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    // return_val = x + r
    entry.instructions.push(HirInstruction::Binary {
        op: zyntax_compiler::hir::BinaryOp::Add,
        result: return_val,
        ty: HirType::I32,
        left: live_across,
        right: await_result,
    });

    let mut blocks = IndexMap::new();
    blocks.insert(entry_id, entry);
    function.blocks = blocks;
    function.entry_block = entry_id;

    AsyncFnFixture {
        function,
        live_across,
        await_result,
    }
}

/// What `make_effectful_function_with_one_perform` returns: the
/// function plus the SSA values planted in the body so tests can
/// refer to them when asserting save/load behavior.
pub struct EffectfulFnFixture {
    pub function: HirFunction,
    /// SSA value that's defined before the perform and used after.
    pub live_across: HirId,
    /// SSA value for the result of the PerformEffect (i32). Used by
    /// the post-perform code (e.g. as the function's return).
    pub perform_result: HirId,
}

/// Build the canonical "effect captures-lift" test shape:
///
/// ```pseudo
/// @effect(State) fn run(input: i32) -> i32 {   // entry block:
///     let x = input + 1                         //   [Binary(Add, input, 1) → live_across]
///     let r = perform State.get()               //   [PerformEffect → perform_result]
///     return x + r                              //   [Binary(Add, live_across, perform_result), Return]
/// }
/// ```
///
/// `live_across` (`x`) is defined before the perform and used after —
/// the canonical capture-lift target. `perform_result` (`r`) is
/// defined by the PerformEffect. The function carries
/// `signature.effects = [State]` so `HirSuspendingFns` seeds it.
pub fn make_effectful_function_with_one_perform() -> EffectfulFnFixture {
    let sig = HirFunctionSignature {
        params: vec![],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![InternedString::new_global("State")],
        is_pure: false,
    };

    let mut function = HirFunction::new(InternedString::new_global("run"), sig);
    function.is_external = false;

    // ── SSA values ──
    let input = HirId::new();
    let const_one = HirId::new();
    let live_across = HirId::new();
    let perform_result = HirId::new();
    let return_val = HirId::new();

    // Effect ID — typically resolved from the module's effect map. We
    // use a fresh HirId to stand in (lower_perform_effect_calls only
    // round-trips it; it doesn't dereference into module state).
    let effect_id = HirId::new();

    for (id, ty) in [
        (input, HirType::I32),
        (const_one, HirType::I32),
        (live_across, HirType::I32),
        (perform_result, HirType::I32),
        (return_val, HirType::I32),
    ] {
        let kind = if id == const_one {
            HirValueKind::Constant(zyntax_compiler::hir::HirConstant::I32(1))
        } else {
            HirValueKind::Instruction
        };
        function.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind,
                uses: HashSet::new(),
                span: None,
            },
        );
    }

    // ── Entry block ──
    let entry_id = HirId::new();
    let mut entry = HirBlock {
        id: entry_id,
        label: Some(InternedString::new_global("entry")),
        phis: vec![],
        instructions: vec![],
        terminator: HirTerminator::Return {
            values: vec![return_val],
        },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![],
    };
    // x = input + 1
    entry.instructions.push(HirInstruction::Binary {
        op: zyntax_compiler::hir::BinaryOp::Add,
        result: live_across,
        ty: HirType::I32,
        left: input,
        right: const_one,
    });
    // r = perform State.get()
    entry.instructions.push(HirInstruction::PerformEffect {
        result: Some(perform_result),
        effect_id,
        op_name: InternedString::new_global("get"),
        args: vec![],
        return_ty: HirType::I32,
    });
    // return_val = x + r
    entry.instructions.push(HirInstruction::Binary {
        op: zyntax_compiler::hir::BinaryOp::Add,
        result: return_val,
        ty: HirType::I32,
        left: live_across,
        right: perform_result,
    });

    let mut blocks = IndexMap::new();
    blocks.insert(entry_id, entry);
    function.blocks = blocks;
    function.entry_block = entry_id;

    EffectfulFnFixture {
        function,
        live_across,
        perform_result,
    }
}

/// Build a one-function `HirModule` containing `function`. Useful so
/// tests can run `HirSuspendingFns::from_module` against a populated
/// module rather than constructing the suspending set by hand.
pub fn module_of(function: HirFunction) -> HirModule {
    let mut module = HirModule::new(InternedString::new_global("test_module"));
    let id = function.id;
    module.functions.insert(id, function);
    module
}

/// Build an async function whose await target is an extern Symbol
/// call matching the cooperative-async host-bridge naming convention
/// (`__zyntax_async_*`). Mirrors `make_async_function_with_one_await`
/// but swaps the inner `Call(Function(foo))` for
/// `Call(Symbol(symbol_name))` so the Phase I.2 cooperative parking
/// lowering in `lower_await_calls` fires.
///
/// ```pseudo
/// async fn aw(input: i32) -> i32 {
///     let x = input + 1
///     let r = await __zyntax_async_set_timeout(input)
///     return x + r
/// }
/// ```
pub fn make_async_function_with_host_bridge_await(symbol_name: &str) -> AsyncFnFixture {
    let mut sig = HirFunctionSignature {
        params: vec![],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: true,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };
    sig.is_async = true;

    let mut function = HirFunction::new(InternedString::new_global("aw_host"), sig);
    function.is_external = false;

    let input = HirId::new();
    let const_one = HirId::new();
    let live_across = HirId::new();
    let bridge_result = HirId::new();
    let await_result = HirId::new();
    let return_val = HirId::new();

    for (id, ty) in [
        (input, HirType::I32),
        (const_one, HirType::I32),
        (live_across, HirType::I32),
        (bridge_result, HirType::I32),
        (await_result, HirType::I32),
        (return_val, HirType::I32),
    ] {
        let kind = if id == const_one {
            HirValueKind::Constant(zyntax_compiler::hir::HirConstant::I32(1))
        } else {
            HirValueKind::Instruction
        };
        function.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind,
                uses: HashSet::new(),
                span: None,
            },
        );
    }

    let entry_id = HirId::new();
    let mut entry = HirBlock {
        id: entry_id,
        label: Some(InternedString::new_global("entry")),
        phis: vec![],
        instructions: vec![],
        terminator: HirTerminator::Return {
            values: vec![return_val],
        },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![],
    };
    // x = input + 1
    entry.instructions.push(HirInstruction::Binary {
        op: zyntax_compiler::hir::BinaryOp::Add,
        result: live_across,
        ty: HirType::I32,
        left: input,
        right: const_one,
    });
    // bridge_result = __zyntax_async_set_timeout(input)
    // (Symbol callable — the cooperative-await lowering kicks in iff
    // this is what the SSA produces, i.e. an extern call to a
    // `__zyntax_async_*` name.)
    entry.instructions.push(HirInstruction::Call {
        result: Some(bridge_result),
        callee: HirCallable::Symbol(symbol_name.to_string()),
        args: vec![input],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    entry.instructions.push(HirInstruction::Call {
        result: Some(await_result),
        callee: HirCallable::Intrinsic(Intrinsic::Await),
        args: vec![bridge_result],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    entry.instructions.push(HirInstruction::Binary {
        op: zyntax_compiler::hir::BinaryOp::Add,
        result: return_val,
        ty: HirType::I32,
        left: live_across,
        right: await_result,
    });

    let mut blocks = IndexMap::new();
    blocks.insert(entry_id, entry);
    function.blocks = blocks;
    function.entry_block = entry_id;

    AsyncFnFixture {
        function,
        live_across,
        await_result,
    }
}

/// `live_out` map containing just `live_across` for the function's
/// entry block. Mirrors what zyntax's existing per-block liveness
/// analysis would produce for the canonical fixture.
pub fn live_out_for_entry_only(
    function: &HirFunction,
    live_id: HirId,
) -> HashMap<HirId, HashSet<HirId>> {
    let mut map = HashMap::new();
    let mut set = HashSet::new();
    set.insert(live_id);
    map.insert(function.entry_block, set);
    map
}
