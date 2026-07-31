//! Unification of parser placeholders and first-class type variants.
//!
//! The parser never consults the type registry — every user-written type
//! name arrives as `Type::Unresolved(name)`, and a binding with no
//! annotation arrives as `Any` / `Unknown`. Unification has to accept
//! those, or every program using a user-defined type fails to check.
//!
//! These also cover the first-class variants (`Fiber`, `Vector`, `Result`,
//! `Nullable`) that had no arm at all and fell through to the catch-all.

use zyntax_typed_ast::arena::InternedString;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{PrimitiveType, TypeMetadata, TypeRegistry};
use zyntax_typed_ast::{InferenceContext, Type};

fn ctx() -> InferenceContext {
    InferenceContext::new(Box::new(TypeRegistry::new()))
}

/// A context whose registry knows `name` as an atomic type, plus the
/// `Type::Named` that refers to it.
fn ctx_with_type(name: &str) -> (InferenceContext, InternedString, Type) {
    let mut registry = TypeRegistry::new();
    let interned = InternedString::new_global(name);
    let id = registry.register_atomic_type(interned, TypeMetadata::default(), Span::new(0, 0));
    let named = Type::Named {
        id,
        type_args: vec![],
        const_args: vec![],
        variance: vec![],
        nullability: Default::default(),
    };
    (InferenceContext::new(Box::new(registry)), interned, named)
}

fn i64_ty() -> Type {
    Type::Primitive(PrimitiveType::I64)
}

fn f64_ty() -> Type {
    Type::Primitive(PrimitiveType::F64)
}

#[test]
fn identical_unresolved_names_unify() {
    let mut ctx = ctx();
    let name = InternedString::new_global("Tensor");
    let unified = ctx
        .unify(Type::Unresolved(name), Type::Unresolved(name))
        .expect("a placeholder should unify with itself");
    assert_eq!(unified, Type::Unresolved(name));
}

#[test]
fn different_unresolved_names_do_not_unify() {
    let mut ctx = ctx();
    let a = InternedString::new_global("Tensor");
    let b = InternedString::new_global("Buffer");
    assert!(ctx.unify(Type::Unresolved(a), Type::Unresolved(b)).is_err());
}

/// The case that made the whole engine unusable on ZynML: the parser
/// hands over `Unresolved("Tensor")` while the registry holds the real
/// `Named` type, and the two have to meet.
#[test]
fn an_unresolved_name_unifies_with_the_registered_type() {
    let (mut ctx, name, named) = ctx_with_type("Tensor");

    let unified = ctx
        .unify(Type::Unresolved(name), named.clone())
        .expect("placeholder should resolve through the registry");
    assert_eq!(unified, named);

    // Order must not matter.
    let flipped = ctx
        .unify(named.clone(), Type::Unresolved(name))
        .expect("placeholder should resolve in either position");
    assert_eq!(flipped, named);
}

#[test]
fn an_unresolved_name_does_not_unify_with_an_unrelated_type() {
    let (mut ctx, name, _) = ctx_with_type("Tensor");
    assert!(ctx.unify(Type::Unresolved(name), i64_ty()).is_err());
}

/// A name the registry has never seen still unifies with the placeholder
/// side of gradual typing, rather than erroring.
#[test]
fn an_unknown_placeholder_still_unifies_with_any() {
    let mut ctx = ctx();
    let name = InternedString::new_global("NeverRegistered");
    assert!(ctx.unify(Type::Unresolved(name), Type::Any).is_ok());
}

#[test]
fn unknown_and_dynamic_unify_like_any() {
    let mut ctx = ctx();
    assert_eq!(ctx.unify(Type::Unknown, i64_ty()).unwrap(), i64_ty());
    assert_eq!(ctx.unify(i64_ty(), Type::Unknown).unwrap(), i64_ty());
    assert_eq!(ctx.unify(Type::Dynamic, f64_ty()).unwrap(), f64_ty());
    assert_eq!(ctx.unify(f64_ty(), Type::Dynamic).unwrap(), f64_ty());
}

#[test]
fn fibers_unify_through_their_yield_type() {
    let mut ctx = ctx();
    let unified = ctx
        .unify(
            Type::Fiber(Box::new(i64_ty())),
            Type::Fiber(Box::new(Type::Unknown)),
        )
        .expect("Fiber<i64> and Fiber<_> should unify");
    assert_eq!(unified, Type::Fiber(Box::new(i64_ty())));

    assert!(ctx
        .unify(
            Type::Fiber(Box::new(i64_ty())),
            Type::Fiber(Box::new(f64_ty()))
        )
        .is_err());
}

/// Lane count is part of the type — `f32x4` and `f32x8` are distinct even
/// though their element types agree.
#[test]
fn vectors_unify_only_at_the_same_lane_count() {
    let mut ctx = ctx();
    let f32_ty = Type::Primitive(PrimitiveType::F32);

    let unified = ctx
        .unify(
            Type::Vector(Box::new(f32_ty.clone()), 4),
            Type::Vector(Box::new(f32_ty.clone()), 4),
        )
        .expect("f32x4 should unify with itself");
    assert_eq!(unified, Type::Vector(Box::new(f32_ty.clone()), 4));

    assert!(ctx
        .unify(
            Type::Vector(Box::new(f32_ty.clone()), 4),
            Type::Vector(Box::new(f32_ty), 8),
        )
        .is_err());
}

#[test]
fn results_unify_through_both_arms() {
    let mut ctx = ctx();
    let lhs = Type::Result {
        ok_type: Box::new(i64_ty()),
        err_type: Box::new(Type::Unknown),
    };
    let rhs = Type::Result {
        ok_type: Box::new(Type::Unknown),
        err_type: Box::new(f64_ty()),
    };

    let unified = ctx.unify(lhs, rhs).expect("both arms should unify");
    assert_eq!(
        unified,
        Type::Result {
            ok_type: Box::new(i64_ty()),
            err_type: Box::new(f64_ty()),
        }
    );
}

#[test]
fn nullable_unifies_through_its_inner_type() {
    let mut ctx = ctx();
    let unified = ctx
        .unify(
            Type::Nullable(Box::new(i64_ty())),
            Type::Nullable(Box::new(Type::Unknown)),
        )
        .expect("T? should unify through T");
    assert_eq!(unified, Type::Nullable(Box::new(i64_ty())));
}

/// A type variable nested inside one of the newly-recursive variants must
/// be caught by the occurs check — otherwise `T = Fiber<T>` is accepted
/// and later recursion diverges.
#[test]
fn the_occurs_check_sees_through_a_fiber() {
    let mut ctx = ctx();
    let var = ctx.fresh_type_var();
    assert!(ctx.unify(var.clone(), Type::Fiber(Box::new(var))).is_err());
}

/// Substitutions have to reach inside the new variants too, or a solved
/// variable stays unsubstituted in a `Fiber<T>` return type.
#[test]
fn substitutions_reach_inside_a_fiber() {
    let mut ctx = ctx();
    let var = ctx.fresh_type_var();
    ctx.unify(var.clone(), i64_ty()).expect("bind the variable");

    let substituted = ctx.apply_substitutions(&Type::Fiber(Box::new(var)));
    assert_eq!(substituted, Type::Fiber(Box::new(i64_ty())));
}

/// Records today's semantics, which Phase A/B changes: an unannotated
/// return type is spelled `Unit`, so it is indistinguishable from a
/// genuinely void one and cannot unify with the value a body returns.
#[test]
fn unit_does_not_unify_with_a_value_type() {
    let mut ctx = ctx();
    assert!(ctx
        .unify(Type::Primitive(PrimitiveType::Unit), i64_ty())
        .is_err());
}
