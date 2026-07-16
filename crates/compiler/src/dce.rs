//! # Reachability-based Dead-Code Elimination for HIR functions
//!
//! Computes the set of [`HirId`]s for functions transitively callable from a
//! set of entry-point names (typically `["main"]`). Used by the Cranelift
//! backend to skip codegen for unreachable functions (e.g. the ~100 prelude
//! helpers that a one-file benchmark kernel never invokes), shaving the
//! per-install JIT-compile cost.
//!
//! ## Conservative handling of indirect calls
//!
//! - Direct calls ([`HirCallable::Function`]) are precise.
//! - Function references that escape (via [`HirCallable::FuncRef`],
//!   [`HirInstruction::CreateClosure`], or as members of a vtable that any
//!   reachable function loads from) are treated as roots — their address is
//!   observable.
//! - If any reachable function performs an indirect call
//!   ([`HirCallable::Indirect`], [`HirInstruction::IndirectCall`],
//!   [`HirInstruction::CallClosure`], or [`HirInstruction::TraitMethodCall`])
//!   AND we cannot statically resolve the target, we fall back to
//!   "compile everything" by returning the full set of function ids.
//!
//! This preserves correctness in all current call patterns while still
//! winning on the common kernels-of-known-callees case.

use crate::hir::{HirCallable, HirConstant, HirId, HirInstruction, HirModule};
use std::collections::HashSet;

/// Compute the set of function [`HirId`]s reachable from the given entry-point
/// names. Always includes extern function declarations (those have
/// `is_external = true`) because they are registered as symbols and not
/// compiled by Cranelift anyway.
///
/// If the analysis cannot prove indirect-call targets are safe to prune, it
/// returns the full set of function ids (conservative).
pub fn reachable_function_ids(module: &HirModule, entry_names: &[&str]) -> HashSet<HirId> {
    // Walk: collect direct calls, FuncRef-style escapes, and detect any
    // indirect-call site. On indirect-call detection, fall back to "everything
    // reachable" (the full function-id set).
    let mut reachable: HashSet<HirId> = HashSet::new();
    let mut worklist: Vec<HirId> = Vec::new();
    // Track every extern symbol name observed in a `HirCallable::Symbol`
    // call site so we can resolve them back to extern HirIds after the
    // walk and add only the externs actually referenced. Without this,
    // every extern in the module ends up in the reachable set even when
    // the kernel never calls it — fine for Cranelift, but bloats the
    // LLVM IR with hundreds of dead `declare` statements that `dlopen`
    // (with RTLD_NOW) then pays to resolve, adding ~270 ms per install
    // on macOS.
    let mut called_extern_names: HashSet<String> = HashSet::new();
    // Roots: entry-point function ids matched by name.
    for (id, function) in &module.functions {
        if let Some(name) = function.name.resolve_global() {
            if entry_names.iter().any(|e| *e == name) {
                worklist.push(*id);
            }
        }
    }

    // If no entry-point function is present (e.g. embedded / test scenarios
    // where the host calls arbitrary user-defined functions via
    // `call_function_raw`), there is no safe basis for pruning — fall back
    // to compiling every function in the module.
    if worklist.is_empty() {
        return all_function_ids(module);
    }

    // Function refs taken anywhere in the module (used to seed reachability
    // for FuncRef'd functions even if no direct call reaches them).
    let mut escaped_funcs: HashSet<HirId> = HashSet::new();
    // Whether we've already seeded escapes (we do this once when we first
    // detect an indirect call so we don't double-walk).
    let mut seeded_escapes = false;

    while let Some(fid) = worklist.pop() {
        if !reachable.insert(fid) {
            continue;
        }
        let func = match module.functions.get(&fid) {
            Some(f) => f,
            None => continue,
        };
        // External declarations have no body; nothing to walk.
        if func.is_external {
            continue;
        }

        for (_, block) in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    HirInstruction::Call { callee, .. } => match callee {
                        HirCallable::Function(target) => {
                            if !reachable.contains(target) {
                                worklist.push(*target);
                            }
                        }
                        HirCallable::FuncRef(target) => {
                            // The address escapes — pessimistically assume it
                            // may be invoked indirectly later.
                            escaped_funcs.insert(*target);
                            if !reachable.contains(target) {
                                worklist.push(*target);
                            }
                        }
                        HirCallable::Indirect(_) => {
                            // Unknown target. Seed escape closure once, then
                            // fall back to full set if escapes can't account
                            // for it.
                            return all_function_ids(module);
                        }
                        // Intrinsics don't reach HIR functions in this module.
                        HirCallable::Intrinsic(_) => {}
                        // Symbol calls reference an extern function by name.
                        // Record the name so we can resolve it back to a HirId
                        // (and emit a declaration for it) after the walk.
                        HirCallable::Symbol(name) => {
                            called_extern_names.insert(name.clone());
                        }
                    },
                    HirInstruction::IndirectCall { .. }
                    | HirInstruction::CallClosure { .. }
                    | HirInstruction::TraitMethodCall { .. } => {
                        // Same conservative fallback. TraitMethodCall could in
                        // principle be resolved via vtables in globals, but
                        // we keep it simple — these are rare in benchmark
                        // kernels.
                        return all_function_ids(module);
                    }
                    HirInstruction::CreateClosure { function, .. } => {
                        // Closure body is a real function reachable through
                        // the closure value.
                        escaped_funcs.insert(*function);
                        if !reachable.contains(function) {
                            worklist.push(*function);
                        }
                    }
                    HirInstruction::PerformEffect {
                        effect_id, op_name, ..
                    } => {
                        // A `perform` reaches the op function of every
                        // handler for this effect — the concrete one is
                        // selected at runtime (statically for a single
                        // handler, or via the handler stack under a
                        // `with` block). Mark ALL candidate handler op
                        // fns reachable so their bodies get compiled;
                        // otherwise the direct/indirect call resolves to
                        // an undefined symbol at JIT finalize.
                        for handler in module.handlers.values() {
                            if handler.effect_id != *effect_id {
                                continue;
                            }
                            if !handler
                                .implementations
                                .iter()
                                .any(|i| i.op_name == *op_name)
                            {
                                continue;
                            }
                            let mangled = crate::effect_codegen::mangle_handler_op_name(
                                handler.name,
                                *op_name,
                            );
                            for (fid, f) in &module.functions {
                                if f.name.resolve_global().as_deref() == Some(mangled.as_str())
                                    && !reachable.contains(fid)
                                {
                                    worklist.push(*fid);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Seed escapes once we've processed the roots. A FuncRef taken inside
        // a not-yet-reachable function would already be discovered when that
        // function is walked; this just handles the case of vtables wired
        // into globals.
        if !seeded_escapes {
            seeded_escapes = true;
            for global in module.globals.values() {
                if let Some(init) = &global.initializer {
                    collect_vtable_funcs(init, &mut escaped_funcs);
                }
            }
            for fid in &escaped_funcs {
                if !reachable.contains(fid) {
                    worklist.push(*fid);
                }
            }
        }
    }

    // Include extern declarations only when they're actually referenced
    // from reachable code — either by HirId (`HirCallable::Function`,
    // already added during the walk) or by name (`HirCallable::Symbol`,
    // resolved here from `called_extern_names`). Excluding the rest keeps
    // the LLVM IR free of hundreds of dead `declare` statements that
    // bloat the dlopen-with-RTLD_NOW resolution step.
    if !called_extern_names.is_empty() {
        for (id, function) in &module.functions {
            if !function.is_external {
                continue;
            }
            if reachable.contains(id) {
                continue;
            }
            if let Some(name) = function.name.resolve_global() {
                if called_extern_names.contains(name.as_str()) {
                    reachable.insert(*id);
                }
            }
        }
    }

    reachable
}

/// Walk a HirConstant looking for VTable entries whose `function_id` fields
/// are addresses of HIR functions that may escape.
fn collect_vtable_funcs(c: &HirConstant, out: &mut HashSet<HirId>) {
    match c {
        HirConstant::VTable(vt) => {
            for entry in &vt.methods {
                out.insert(entry.function_id);
            }
        }
        HirConstant::Array(items) | HirConstant::Struct(items) => {
            for item in items {
                collect_vtable_funcs(item, out);
            }
        }
        _ => {}
    }
}

fn all_function_ids(module: &HirModule) -> HashSet<HirId> {
    module.functions.keys().copied().collect()
}
