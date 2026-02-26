//! Simple test for basic effect system functionality

use zyntax_typed_ast::effect_system::*;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::{AsyncKind, CallingConvention, NullabilityKind};

#[test]
fn test_effect_system_basic_creation() {
    // Test that we can create an effect system
    let mut effect_system = EffectSystem::new();

    // Test basic effect type creation
    let io_effect = Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Definite,
    };

    let state_effect = Effect {
        id: EffectId::next(),
        region: None,
        intensity: EffectIntensity::Potential,
    };

    // Test effect set creation
    let mut effects = EffectSet::empty();
    assert!(effects.is_pure());

    effects.add_effect(io_effect);
    effects.add_effect(state_effect);
    assert!(!effects.is_pure());
    assert_eq!(effects.effects.len(), 2);
}

#[test]
fn test_effect_signatures() {
    // Test pure signature
    let pure_sig = EffectSignature::pure();
    assert!(pure_sig.is_pure);
    assert!(pure_sig.output_effects.is_pure());

    // Test I/O signature
    let io_sig = EffectSignature::io();
    assert!(!io_sig.is_pure);
    assert!(!io_sig.output_effects.is_pure());

    // Test state signature
    let state_sig = EffectSignature::state();
    assert!(!state_sig.is_pure);
    assert!(!state_sig.output_effects.is_pure());
}

#[test]
fn test_effect_type_info() {
    // Test creating effect type info
    let io_info = EffectTypeInfo::io_effect();
    assert_eq!(io_info.effect_kind, EffectKind::IO);

    let state_info = EffectTypeInfo::state_effect();
    assert_eq!(state_info.effect_kind, EffectKind::State);

    let exception_info = EffectTypeInfo::exception_effect();
    assert_eq!(exception_info.effect_kind, EffectKind::Exception);
}

#[test]
fn test_effect_handler() {
    let handled_effects = EffectSet::empty();

    let handler = EffectHandler {
        id: EffectHandlerId::next(),
        handled_effects,
        handler_type: HandlerType::Exception,
        operations: Vec::new(),
        return_effect: None,
        scope: Span::new(0, 100),
    };

    assert_eq!(handler.handler_type, HandlerType::Exception);
}

#[test]
fn test_effect_inference_context() {
    let context = EffectInferenceContext::new();

    // Verify initial state
    assert!(context.effect_vars.is_empty());
    assert!(context.constraints.is_empty());

    // Test default options
    let options = &context.options;
    assert!(options.allow_effect_polymorphism);
    assert!(options.infer_purity);
    assert!(options.check_effect_safety);
    assert!(options.enable_effect_optimization);
    assert_eq!(options.max_effect_depth, 10);
}
