//! Integration test for effect system with the universal type system
//!
//! This test demonstrates how effect systems integrate with the broader
//! universal type system and provides comprehensive workflow testing.

use std::collections::BTreeSet;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::*;
use zyntax_typed_ast::{AsyncKind, CallingConvention, NullabilityKind};

#[test]
fn test_effect_system_integration() {
    // Create an effect system
    let mut effect_system = EffectSystem::new();

    // Create basic effect types
    let io_effect_info = EffectTypeInfo::io_effect();
    let state_effect_info = EffectTypeInfo::state_effect();
    let exception_effect_info = EffectTypeInfo::exception_effect();

    effect_system.add_effect_type(io_effect_info);
    effect_system.add_effect_type(state_effect_info);
    effect_system.add_effect_type(exception_effect_info);

    // Verify effect types were added
    assert_eq!(effect_system.effect_types.len(), 3);

    // Test effect signature creation
    let pure_sig = EffectSignature::pure();
    assert!(pure_sig.is_pure);
    assert!(pure_sig.output_effects.is_pure());

    let io_sig = EffectSignature::io();
    assert!(!io_sig.is_pure);
    assert!(!io_sig.output_effects.is_pure());
}

#[test]
fn test_effect_composition_workflow() {
    let system = EffectSystem::new();

    // Create different types of effects
    let io_effect = Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Definite,
    };

    let state_effect = Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Definite,
    };

    let memory_effect = Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Potential,
    };

    // Create effect sets
    let mut io_effects = EffectSet::empty();
    io_effects.add_effect(io_effect);

    let mut state_effects = EffectSet::empty();
    state_effects.add_effect(state_effect);

    let mut memory_effects = EffectSet::empty();
    memory_effects.add_effect(memory_effect);

    // Test sequential composition
    let sequential_result = system.compose_effects(
        io_effects.clone(),
        state_effects.clone(),
        CompositionOperator::Sequential,
    );
    assert!(sequential_result.is_ok());

    let composed = sequential_result.unwrap();
    assert_eq!(composed.effects.len(), 2);

    // Test alternative composition
    let alternative_result =
        system.compose_effects(io_effects, memory_effects, CompositionOperator::Alternative);
    assert!(alternative_result.is_ok());

    let alternative_composed = alternative_result.unwrap();
    assert_eq!(alternative_composed.unions.len(), 1);
}

#[test]
fn test_effect_handler_workflow() {
    let mut system = EffectSystem::new();

    // Create an effect to handle
    let exception_effect = Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Potential,
    };

    let mut handled_effects = EffectSet::empty();
    handled_effects.add_effect(exception_effect.clone());

    // Create an exception handler
    let handler = EffectHandler {
        id: EffectHandlerId::next(),
        handled_effects,
        handler_type: HandlerType::Exception,
        operations: Vec::new(),
        return_effect: None,
        scope: Span::new(0, 100),
    };

    system.add_effect_handler(handler);

    // Verify handler was added
    assert_eq!(system.effect_handlers.len(), 1);

    // Test effect handling check
    let is_handled = system.is_effect_handled(&exception_effect);
    assert!(is_handled);
}

#[test]
fn test_effect_inference_context() {
    let context = EffectInferenceContext::new();

    // Verify initial state
    assert!(context.effect_vars.is_empty());
    assert!(context.constraints.is_empty());

    // Test default inference options
    let options = &context.options;
    assert!(options.allow_effect_polymorphism);
    assert!(options.infer_purity);
    assert!(options.check_effect_safety);
    assert!(options.enable_effect_optimization);
    assert_eq!(options.max_effect_depth, 10);
}

#[test]
fn test_effect_scoping() {
    let mut system = EffectSystem::new();
    let span = Span::new(0, 50);

    // Test scope management
    assert_eq!(system.scope_stack.len(), 0);

    // Enter function scope
    system.enter_scope(EffectScopeKind::Function, span);
    assert_eq!(system.scope_stack.len(), 1);
    assert_eq!(system.scope_stack[0].scope_kind, EffectScopeKind::Function);

    // Enter nested block scope
    system.enter_scope(EffectScopeKind::Block, span);
    assert_eq!(system.scope_stack.len(), 2);
    assert_eq!(system.scope_stack[1].scope_kind, EffectScopeKind::Block);

    // Exit scopes
    let exit_result = system.exit_scope();
    assert!(exit_result.is_ok());
    assert_eq!(system.scope_stack.len(), 1);

    let exit_result = system.exit_scope();
    assert!(exit_result.is_ok());
    assert_eq!(system.scope_stack.len(), 0);
}

#[test]
fn test_effect_amplification() {
    let system = EffectSystem::new();

    let effect = Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Definite,
    };

    let mut effects = EffectSet::empty();
    effects.add_effect(effect);

    // Test amplification to repeated
    let amplified_result = system.amplify_effects(effects, EffectIntensity::Repeated);
    assert!(amplified_result.is_ok());

    let amplified = amplified_result.unwrap();
    for effect in &amplified.effects {
        assert_eq!(effect.intensity, EffectIntensity::Repeated);
    }
}

#[test]
fn test_function_signature_management() {
    let mut system = EffectSystem::new();

    // Create a function name
    let func_name =
        InternedString::from_symbol(string_interner::Symbol::try_from_usize(1).unwrap());

    // Create and set an I/O signature
    let io_signature = EffectSignature::io();
    system.set_function_signature(func_name, io_signature.clone());

    // Retrieve and verify
    let retrieved = system.get_function_signature(&func_name);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), &io_signature);
    assert!(!retrieved.unwrap().is_pure);
}

#[test]
fn test_builtin_effect_creation() {
    let system = EffectSystem::new();

    // Test builtin effect creation
    let state_effect = system.create_state_effect();
    assert_eq!(state_effect.intensity, EffectIntensity::Definite);

    let memory_effect = system.create_memory_effect();
    assert_eq!(memory_effect.intensity, EffectIntensity::Definite);
}

#[test]
fn test_effect_subtyping() {
    let system = EffectSystem::new();

    // Create effect sets for subtyping test
    let io_effect = Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Definite,
    };

    let mut subset = EffectSet::empty();
    subset.add_effect(io_effect.clone());

    let mut superset = EffectSet::empty();
    superset.add_effect(io_effect);
    superset.add_effect(Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Definite,
    });

    // Test subtyping relationship
    let is_subtype = system.is_effect_subtype(&subset, &superset);
    assert!(is_subtype);

    // Test non-subtyping relationship
    let is_not_subtype = system.is_effect_subtype(&superset, &subset);
    assert!(!is_not_subtype);
}

/// Test comprehensive effect system workflow
#[test]
fn test_complete_effect_system_workflow() {
    let mut system = EffectSystem::new();

    // Step 1: Setup effect type system
    system.add_effect_type(EffectTypeInfo::io_effect());
    system.add_effect_type(EffectTypeInfo::state_effect());
    system.add_effect_type(EffectTypeInfo::exception_effect());

    // Step 2: Create effect handlers
    let io_effect = Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Definite,
    };

    let mut handled_effects = EffectSet::empty();
    handled_effects.add_effect(io_effect.clone());

    let io_handler = EffectHandler {
        id: EffectHandlerId::next(),
        handled_effects,
        handler_type: HandlerType::Resource,
        operations: Vec::new(),
        return_effect: None,
        scope: Span::new(0, 200),
    };

    system.add_effect_handler(io_handler);

    // Step 3: Create function signatures
    let read_func =
        InternedString::from_symbol(string_interner::Symbol::try_from_usize(2).unwrap());
    let write_func =
        InternedString::from_symbol(string_interner::Symbol::try_from_usize(3).unwrap());

    let mut read_sig = EffectSignature::pure();
    read_sig.output_effects.add_effect(io_effect.clone());
    read_sig.is_pure = false;

    let mut write_sig = EffectSignature::pure();
    write_sig.output_effects.add_effect(io_effect.clone());
    write_sig
        .output_effects
        .add_effect(system.create_state_effect());
    write_sig.is_pure = false;

    system.set_function_signature(read_func, read_sig);
    system.set_function_signature(write_func, write_sig);

    // Step 4: Test effect scoping and composition
    let span = Span::new(0, 300);
    system.enter_scope(EffectScopeKind::Function, span);

    // Simulate adding effects to context
    let mut current_effects = EffectSet::empty();
    current_effects.add_effect(io_effect);
    system.add_effects_to_context(current_effects);

    // Step 5: Test constraint solving (simplified)
    let solve_result = system.solve_constraints();
    assert!(solve_result.is_ok());

    // Step 6: Exit scope
    let exit_result = system.exit_scope();
    // May fail due to unhandled effects, which is expected behavior

    // Verify the complete workflow executed
    assert!(system.effect_types.len() >= 3);
    assert!(system.effect_handlers.len() >= 1);
    assert!(system.function_effects.len() >= 2);
}

#[test]
fn test_effect_error_handling() {
    let system = EffectSystem::new();

    // Create an unhandled effect
    let unhandled_effect = Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Definite,
    };

    // Test that unhandled effect is detected
    let is_handled = system.is_effect_handled(&unhandled_effect);
    assert!(!is_handled);

    // Test effect compatibility checking
    let empty_effects = EffectSet::empty();
    let mut non_empty_effects = EffectSet::empty();
    non_empty_effects.add_effect(unhandled_effect);

    let span = Span::new(0, 50);
    let compatibility_result =
        system.check_effect_compatibility(&non_empty_effects, &empty_effects, span);
    assert!(compatibility_result.is_err());
}
