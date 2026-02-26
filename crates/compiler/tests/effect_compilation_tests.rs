//! # Effect Compilation Tests
//!
//! Integration tests for the algebraic effects compilation pipeline:
//! - Effect lowering from TypedAST to HIR
//! - Effect analysis and type checking
//! - Handler resolution
//! - Code generation infrastructure

use indexmap::IndexMap;
use std::collections::HashSet;
use std::sync::Arc;
use zyntax_compiler::{
    // Effect analysis
    effect_analysis::{analyze_effects, EffectErrorKind, ModuleEffectAnalysis},
    // Codegen
    effect_codegen::{
        analyze_handle_effect, analyze_perform_effect, get_handler_ops_info,
        mangle_handler_op_name, EffectCodegenContext, PerformStrategy,
    },
    // Handler resolution
    effect_handler_resolution::{resolve_handlers, HandlerOptimization, ModuleHandlerResolution},
    // HIR types
    hir::{
        CallingConvention, FunctionAttributes, HirBlock, HirEffect, HirEffectHandler,
        HirEffectHandlerImpl, HirEffectOp, HirFunction, HirFunctionSignature, HirId,
        HirInstruction, HirLifetime, HirModule, HirParam, HirTerminator, HirType,
    },
};
use zyntax_typed_ast::{InternedString, TypeId};

// =============================================================================
// Test Helpers
// =============================================================================

fn create_test_module() -> HirModule {
    HirModule {
        id: HirId::new(),
        name: InternedString::new_global("test"),
        functions: IndexMap::new(),
        globals: IndexMap::new(),
        types: IndexMap::new(),
        imports: vec![],
        exports: vec![],
        version: 0,
        dependencies: HashSet::new(),
        effects: IndexMap::new(),
        handlers: IndexMap::new(),
    }
}

fn create_test_block(id: HirId) -> HirBlock {
    HirBlock {
        id,
        label: None,
        phis: vec![],
        instructions: vec![],
        terminator: HirTerminator::Return { values: vec![] },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![],
    }
}

fn create_test_signature() -> HirFunctionSignature {
    HirFunctionSignature {
        params: vec![],
        returns: vec![],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_async: false,
        is_variadic: false,
        effects: vec![],
        is_pure: false,
    }
}

fn create_test_function(id: HirId, name: &str) -> HirFunction {
    let block_id = HirId::new();
    let mut blocks = IndexMap::new();
    blocks.insert(block_id, create_test_block(block_id));

    HirFunction {
        id,
        name: InternedString::new_global(name),
        signature: create_test_signature(),
        entry_block: block_id,
        blocks,
        locals: IndexMap::new(),
        values: IndexMap::new(),
        previous_version: None,
        is_external: false,
        calling_convention: CallingConvention::Fast,
        attributes: FunctionAttributes::default(),
        link_name: None,
    }
}

fn create_simple_effect(name: &str, ops: Vec<&str>) -> (HirId, HirEffect) {
    let effect_id = HirId::new();
    let operations = ops
        .iter()
        .map(|op_name| HirEffectOp {
            id: HirId::new(),
            name: InternedString::new_global(op_name),
            type_params: vec![],
            params: vec![],
            return_type: HirType::Void,
        })
        .collect();

    (
        effect_id,
        HirEffect {
            id: effect_id,
            name: InternedString::new_global(name),
            type_params: vec![],
            operations,
        },
    )
}

fn create_simple_handler(
    name: &str,
    effect_id: HirId,
    ops: Vec<&str>,
    is_resumable: bool,
) -> (HirId, HirEffectHandler) {
    let handler_id = HirId::new();
    let implementations = ops
        .iter()
        .map(|op_name| {
            let block_id = HirId::new();
            let mut blocks = IndexMap::new();
            blocks.insert(
                block_id,
                HirBlock {
                    id: block_id,
                    label: None,
                    phis: vec![],
                    instructions: vec![],
                    terminator: HirTerminator::Return { values: vec![] },
                    dominance_frontier: HashSet::new(),
                    predecessors: vec![],
                    successors: vec![],
                },
            );

            HirEffectHandlerImpl {
                op_name: InternedString::new_global(op_name),
                type_params: vec![],
                params: vec![],
                return_type: HirType::Void,
                entry_block: block_id,
                blocks,
                is_resumable,
            }
        })
        .collect();

    (
        handler_id,
        HirEffectHandler {
            id: handler_id,
            name: InternedString::new_global(name),
            effect_id,
            type_params: vec![],
            state_fields: vec![],
            implementations,
        },
    )
}

// =============================================================================
// Effect Analysis Tests
// =============================================================================

#[test]
fn test_effect_analysis_empty_module() {
    let module = create_test_module();
    let result = analyze_effects(&module).unwrap();

    assert!(result.functions.is_empty());
    assert!(result.defined_effects.is_empty());
    assert!(result.defined_handlers.is_empty());
    assert!(result.errors.is_empty());
    assert!(result.warnings.is_empty());
}

#[test]
fn test_effect_analysis_pure_function() {
    let mut module = create_test_module();

    // Add a pure function
    let func_id = HirId::new();
    let mut func = create_test_function(func_id, "pure_fn");
    func.signature.is_pure = true;
    module.functions.insert(func_id, func);

    let result = analyze_effects(&module).unwrap();

    // Pure function with no effects should pass
    assert!(result.errors.is_empty());
    let func_analysis = result.functions.get(&func_id).unwrap();
    assert!(func_analysis.is_pure);
    assert!(func_analysis.total_effects.is_empty());
}

#[test]
fn test_effect_analysis_pure_violation() {
    let mut module = create_test_module();

    // Add an effect
    let (effect_id, effect) = create_simple_effect("IO", vec!["log"]);
    module.effects.insert(effect_id, effect);

    // Add a pure function that performs an effect (should error)
    let func_id = HirId::new();
    let block_id = HirId::new();

    let mut blocks = IndexMap::new();
    blocks.insert(
        block_id,
        HirBlock {
            id: block_id,
            label: None,
            phis: vec![],
            instructions: vec![HirInstruction::PerformEffect {
                result: None,
                effect_id,
                op_name: InternedString::new_global("log"),
                args: vec![],
                return_ty: HirType::Void,
            }],
            terminator: HirTerminator::Return { values: vec![] },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        },
    );

    let mut sig = create_test_signature();
    sig.is_pure = true;

    module.functions.insert(
        func_id,
        HirFunction {
            id: func_id,
            name: InternedString::new_global("impure_fn"),
            signature: sig,
            entry_block: block_id,
            blocks,
            locals: IndexMap::new(),
            values: IndexMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
            link_name: None,
        },
    );

    let result = analyze_effects(&module).unwrap();

    // Should have a pure violation error
    assert!(!result.errors.is_empty());
    assert!(result
        .errors
        .iter()
        .any(|e| matches!(e.kind, EffectErrorKind::PureViolation)));
}

#[test]
fn test_effect_analysis_declared_effects() {
    let mut module = create_test_module();

    // Add an effect
    let (effect_id, effect) = create_simple_effect("State", vec!["get", "put"]);
    module.effects.insert(effect_id, effect);

    // Add a function that declares the effect
    let func_id = HirId::new();
    let mut func = create_test_function(func_id, "stateful_fn");
    func.signature.effects = vec![InternedString::new_global("State")];
    module.functions.insert(func_id, func);

    let result = analyze_effects(&module).unwrap();

    let func_analysis = result.functions.get(&func_id).unwrap();
    assert_eq!(func_analysis.declared_effects.len(), 1);
}

// =============================================================================
// Handler Resolution Tests
// =============================================================================

#[test]
fn test_handler_resolution_empty_module() {
    let module = create_test_module();
    let result = resolve_handlers(&module).unwrap();

    assert!(result.functions.is_empty());
    assert!(result.inlinable_handlers.is_empty());
    assert_eq!(result.stats.total_perform_sites, 0);
}

#[test]
fn test_handler_resolution_simple_handler_inlinable() {
    let mut module = create_test_module();

    // Add effect and simple handler
    let (effect_id, effect) = create_simple_effect("Log", vec!["print"]);
    let (handler_id, handler) =
        create_simple_handler("LogHandler", effect_id, vec!["print"], false);

    module.effects.insert(effect_id, effect);
    module.handlers.insert(handler_id, handler);

    let result = resolve_handlers(&module).unwrap();

    // Simple handler should be inlinable
    assert!(result.inlinable_handlers.contains(&handler_id));
}

#[test]
fn test_handler_resolution_resumable_not_inlinable() {
    let mut module = create_test_module();

    // Add effect and resumable handler
    let (effect_id, effect) = create_simple_effect("Async", vec!["await"]);
    let (handler_id, handler) =
        create_simple_handler("AsyncHandler", effect_id, vec!["await"], true);

    module.effects.insert(effect_id, effect);
    module.handlers.insert(handler_id, handler);

    let result = resolve_handlers(&module).unwrap();

    // Resumable handler should NOT be inlinable
    assert!(!result.inlinable_handlers.contains(&handler_id));
}

// =============================================================================
// Codegen Infrastructure Tests
// =============================================================================

#[test]
fn test_effect_codegen_context_creation() {
    let ctx = EffectCodegenContext::new();
    assert!(ctx.handler_stack.is_empty());
    assert!(ctx.handler_states.is_empty());
    assert!(ctx.handler_resolution.is_none());
}

#[test]
fn test_effect_codegen_mangle_name() {
    let handler = InternedString::new_global("StateHandler");
    let op = InternedString::new_global("get");
    let mangled = mangle_handler_op_name(handler, op);
    assert_eq!(mangled, "StateHandler$effect$get");
}

#[test]
fn test_effect_codegen_handler_ops_info() {
    let effect_id = HirId::new();
    let (_, handler) = create_simple_handler("TestHandler", effect_id, vec!["op1", "op2"], false);

    let ops_info = get_handler_ops_info(&handler);
    assert_eq!(ops_info.len(), 2);
    assert_eq!(ops_info[0].function_name, "TestHandler$effect$op1");
    assert_eq!(ops_info[1].function_name, "TestHandler$effect$op2");
    assert!(!ops_info[0].uses_continuation);
    assert!(!ops_info[1].uses_continuation);
}

#[test]
fn test_effect_codegen_handler_ops_with_continuation() {
    let effect_id = HirId::new();
    let (_, handler) = create_simple_handler("AsyncHandler", effect_id, vec!["await"], true);

    let ops_info = get_handler_ops_info(&handler);
    assert_eq!(ops_info.len(), 1);
    assert!(ops_info[0].uses_continuation);
}

#[test]
fn test_analyze_handle_effect() {
    let mut module = create_test_module();

    let (effect_id, effect) = create_simple_effect("State", vec!["get"]);
    let (handler_id, handler) =
        create_simple_handler("StateHandler", effect_id, vec!["get"], false);

    module.effects.insert(effect_id, effect);
    module.handlers.insert(handler_id, handler);

    let body_block = HirId::new();
    let continuation_block = HirId::new();

    let codegen_info =
        analyze_handle_effect(handler_id, &[], body_block, continuation_block, &module);

    assert!(codegen_info.is_some());
    let info = codegen_info.unwrap();
    assert_eq!(info.handler_id, handler_id);
    assert_eq!(info.effect_id, effect_id);
    assert!(!info.needs_state); // No state fields
    assert_eq!(info.state_size, 0);
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_full_effect_pipeline() {
    let mut module = create_test_module();

    // Step 1: Define effect
    let (effect_id, effect) = create_simple_effect("Logging", vec!["log", "error"]);
    module.effects.insert(effect_id, effect);

    // Step 2: Define handler
    let (handler_id, handler) = create_simple_handler(
        "ConsoleLogger",
        effect_id,
        vec!["log", "error"],
        false, // Not resumable
    );
    module.handlers.insert(handler_id, handler);

    // Step 3: Define function that uses the effect
    let func_id = HirId::new();
    let block_id = HirId::new();

    let mut blocks = IndexMap::new();
    blocks.insert(
        block_id,
        HirBlock {
            id: block_id,
            label: None,
            phis: vec![],
            instructions: vec![HirInstruction::PerformEffect {
                result: None,
                effect_id,
                op_name: InternedString::new_global("log"),
                args: vec![],
                return_ty: HirType::Void,
            }],
            terminator: HirTerminator::Return { values: vec![] },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        },
    );

    let mut sig = create_test_signature();
    sig.effects = vec![InternedString::new_global("Logging")]; // Declares effect

    module.functions.insert(
        func_id,
        HirFunction {
            id: func_id,
            name: InternedString::new_global("main"),
            signature: sig,
            entry_block: block_id,
            blocks,
            locals: IndexMap::new(),
            values: IndexMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
            link_name: None,
        },
    );

    // Step 4: Run effect analysis
    let effect_analysis = analyze_effects(&module).unwrap();

    // Function declares effects and performs them - should not error
    // (handler is declared, so effect is properly handled at call site)
    assert!(effect_analysis.defined_effects.contains_key(&effect_id));
    assert!(effect_analysis.defined_handlers.contains_key(&handler_id));

    // Step 5: Run handler resolution
    let handler_resolution = resolve_handlers(&module).unwrap();

    // Handler should be inlinable (simple, non-resumable)
    assert!(handler_resolution.inlinable_handlers.contains(&handler_id));

    // Should have one perform site
    assert_eq!(handler_resolution.stats.total_perform_sites, 1);
}

#[test]
fn test_nested_effects() {
    let mut module = create_test_module();

    // Define two effects
    let (io_effect_id, io_effect) = create_simple_effect("IO", vec!["print"]);
    let (state_effect_id, state_effect) = create_simple_effect("State", vec!["get", "set"]);

    module.effects.insert(io_effect_id, io_effect);
    module.effects.insert(state_effect_id, state_effect);

    // Define handlers
    let (io_handler_id, io_handler) =
        create_simple_handler("IOHandler", io_effect_id, vec!["print"], false);
    let (state_handler_id, state_handler) =
        create_simple_handler("StateHandler", state_effect_id, vec!["get", "set"], false);

    module.handlers.insert(io_handler_id, io_handler);
    module.handlers.insert(state_handler_id, state_handler);

    // Run analysis
    let effect_analysis = analyze_effects(&module).unwrap();
    assert_eq!(effect_analysis.defined_effects.len(), 2);
    assert_eq!(effect_analysis.defined_handlers.len(), 2);

    // Run handler resolution
    let handler_resolution = resolve_handlers(&module).unwrap();
    assert_eq!(handler_resolution.inlinable_handlers.len(), 2);
}
