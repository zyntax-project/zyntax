//! Cast classification for the Zyntax compiler.
//!
//! Given a source type and a target type, classify the coercion as
//! identity, upcast, downcast, or numeric/pointer conversion. The
//! classification drives downstream code emission — whether to box, to
//! unbox, to widen a pointer, to extract a variant, or to emit a plain
//! numeric `Cast` instruction.
//!
//! This module is intentionally frontend-agnostic. Every frontend that
//! lowers to Zyntax `TypedAST` should consult `classify_cast` when
//! deciding how to materialize a coercion. The same classifier is used
//! at every coercion site: field stores/loads on `Type::Any` slots,
//! explicit `as` casts, let-binding initializers with type
//! annotations, function-argument coercions, and return statements.
//!
//! Only `Identity`, `UpcastBox`, `DowncastUnbox`, and `Convert` are
//! currently materializable. The remaining variants (class hierarchy
//! widening / narrowing, union variant wrap / extract) are present as
//! placeholders so the classifier shape stays stable when those
//! language features land.
//!
//! ## Specificity lattice
//!
//! ```text
//!                     Any
//!                    /   \
//!               Union     Trait object
//!                |          |
//!           Base class   ... (future)
//!                |
//!            Subclass
//!                |
//!         concrete primitives (i64, f64, ...)
//! ```
//!
//! The direction of the cast on this lattice determines `Upcast` vs
//! `Downcast`. Moving from a more specific position to a more general
//! one (toward `Any`) is an upcast and is always safe — the runtime
//! emits the appropriate `zyntax_box_X` wrap. Moving from a more
//! general position to a more specific one is a downcast and may need
//! a runtime tag check; for `Type::Any → T` the wrap-time tag stored
//! in the `DynamicBox` header is used by `zyntax_box_get_X` to widen
//! / narrow losslessly.

use zyntax_typed_ast::type_registry::{Type, TypeRegistry};
use zyntax_typed_ast::PrimitiveType;

/// Classification of a coercion from a source type to a target type.
///
/// Used by `SsaBuilder::emit_coercion` to dispatch to the right code
/// emitter. The classifier itself is pure — it does not touch any HIR
/// state and can be called from any pass that needs to know what kind
/// of cast a (source, target) pair represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    /// Source and target are the same type. No-op at the value level.
    Identity,

    /// Source is concrete, target is `Type::Any`. Emit
    /// `zyntax_box_X` to wrap the value in a `DynamicBox` with the
    /// runtime tag matching the source HIR type.
    UpcastBox,

    /// Source is `Type::Any`, target is concrete. Emit
    /// `zyntax_box_get_X` to unwrap the `DynamicBox` and widen /
    /// narrow to the requested concrete primitive.
    DowncastUnbox,

    /// Both source and target are concrete and structurally
    /// different. Emit a plain `HirInstruction::Cast` with an
    /// op selected by `select_cast_op` (sitofp, fptosi, sext, etc.).
    Convert,

    /// Class hierarchy widening: source is a subclass of target.
    /// Both ends are heap pointers; emit at most a structural
    /// pointer rebind. Not yet wired — placeholder for when ZynML
    /// (and other frontends) gain class extension syntax.
    UpcastWiden,

    /// Class hierarchy narrowing: target is a subclass of source.
    /// Needs a runtime tag check (vtable / typeid). Not yet wired.
    DowncastChecked,

    /// Union variant wrap: source is one of the union's variants,
    /// target is the union itself. Emit a tagged-variant constructor.
    /// Not yet wired.
    UpcastVariant,

    /// Union variant extract: source is the union, target is one of
    /// its variants. Needs a runtime tag check. Not yet wired.
    DowncastVariant,

    /// Types are structurally unrelated and no conversion path exists.
    /// Frontends should reject this at type-check time; if it reaches
    /// SSA the emitter falls back to a best-effort `Convert`.
    Incompatible,
}

/// Classify the coercion `source as target`.
///
/// The returned `CastKind` is the *intent*. The actual instruction
/// emission lives in `SsaBuilder::emit_coercion` so that the classifier
/// can stay frontend-agnostic and free of HIR mutation.
///
/// `type_registry` is consulted to resolve `Type::Named` aliases — a
/// `Type::Named` whose alias target is `Type::Any` is treated as `Any`
/// for classification purposes. This keeps frontends free to expose
/// any spelling (`Any`, `Object`, `dynamic`, etc.) without needing to
/// teach the classifier about each one.
pub fn classify_cast(source: &Type, target: &Type, type_registry: &TypeRegistry) -> CastKind {
    let s_is_any = is_any_type(source, type_registry);
    let t_is_any = is_any_type(target, type_registry);

    match (s_is_any, t_is_any) {
        (true, true) => return CastKind::Identity,
        (false, true) => return CastKind::UpcastBox,
        (true, false) => return CastKind::DowncastUnbox,
        (false, false) => {}
    }

    if types_equal(source, target) {
        return CastKind::Identity;
    }

    // Future: class hierarchy traversal here. When ZynML gains a
    // `class A extends B` form the registry will carry the parent
    // chain; this is where `UpcastWiden` / `DowncastChecked` would be
    // produced.

    // Future: union variant matching here. When typed-AST surfaces
    // `Type::Union(variants)` from a frontend with sum types, this
    // is where `UpcastVariant` / `DowncastVariant` would be produced.

    if is_concrete_scalar(source) && is_concrete_scalar(target) {
        return CastKind::Convert;
    }

    // Anything else (reference rebinds, opaque pointer round-trips,
    // tuple-to-tuple) falls through as Convert. The downstream emitter
    // can refuse it if the underlying HIR types disagree.
    CastKind::Convert
}

/// `true` if `ty` denotes the universal top type — spelled
/// `Type::Any` directly, aliased to it, or registered as an atomic
/// type whose name matches one of the canonical Any spellings
/// recognized at the Zyntax layer.
///
/// The name-based fallback exists because frontends commonly parse
/// their Any keyword as a Named-atomic type before the typed AST
/// reaches the compiler. Recognizing the standard spellings here
/// keeps the classifier working without forcing each frontend to
/// special-case the keyword in its own parser.
/// Whether `name` spells the language's optional type rather than a
/// nominal one the program declares.
///
/// The parser turns these spellings into [`Type::Optional`], because
/// that is the variant carrying the payload type a `case Some(v)`
/// binding needs. Lowering has to agree: a name that reaches it
/// unresolved is this type, not a type the program forgot to declare,
/// and a declaration under this name declares nothing the registry
/// should hold.
pub fn is_optional_type_name(name: &str) -> bool {
    matches!(name, "Option" | "Null")
}

pub fn is_any_type(ty: &Type, registry: &TypeRegistry) -> bool {
    match ty {
        Type::Any => true,
        Type::Alias { target, .. } => is_any_type(target, registry),
        Type::Named { id, .. } => match registry.get_type_by_id(*id) {
            Some(def) => {
                if matches!(registry.resolve_alias(def.name), Some(&Type::Any)) {
                    return true;
                }
                matches_any_spelling(def.name)
            }
            None => false,
        },
        Type::Unresolved(name) => {
            matches!(registry.resolve_alias(*name), Some(&Type::Any)) || matches_any_spelling(*name)
        }
        _ => false,
    }
}

fn matches_any_spelling(name: zyntax_typed_ast::InternedString) -> bool {
    match name.resolve_global() {
        Some(s) => matches!(s.as_str(), "Any"),
        None => false,
    }
}

fn types_equal(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Primitive(p), Type::Primitive(q)) => p == q,
        (Type::Named { id: a, .. }, Type::Named { id: b, .. }) => a == b,
        _ => false,
    }
}

fn is_concrete_scalar(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Primitive(
            PrimitiveType::I8
                | PrimitiveType::I16
                | PrimitiveType::I32
                | PrimitiveType::I64
                | PrimitiveType::I128
                | PrimitiveType::U8
                | PrimitiveType::U16
                | PrimitiveType::U32
                | PrimitiveType::U64
                | PrimitiveType::U128
                | PrimitiveType::F32
                | PrimitiveType::F64
                | PrimitiveType::Bool
        )
    )
}
