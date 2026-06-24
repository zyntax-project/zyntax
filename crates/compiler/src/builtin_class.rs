//! HIR-level wrapper classes for compiler-known built-in types.
//!
//! `Fiber<T>`, and later `SimdVector<T, N>` and other built-in types,
//! carry an associated *wrapper class* at the HIR layer. The wrapper
//! class owns the method table — every method call against a value
//! of the built-in type dispatches through it instead of normal
//! trait method resolution.
//!
//! The wrapper class is what makes the dispatch scalable. Adding a
//! new method to `Fiber<T>` (e.g. `.cancel()`, `.transfer()`) is one
//! arm in the class's `dispatch` method — not another guard in the
//! SSA `MethodCall` handler. Adding a whole new built-in type
//! (`SimdVector`, future `Channel`, etc.) is one new
//! `impl BuiltinClass for FooClass` and one line in the registry
//! constructor.
//!
//! Frontend code sees the built-in type through its prelude
//! declaration (e.g. `extern struct Fiber<T>` plus
//! `impl<T> Iterator for Fiber<T>`). The trait impls' method
//! bodies are stubs the SSA replaces before they ever execute; the
//! compiler-known dispatch table here is the source of truth for
//! what the methods actually do.

use std::sync::Arc;

use crate::hir::HirId;
use crate::CompilerResult;
use zyntax_typed_ast::type_registry::Type;
use zyntax_typed_ast::typed_ast::TypedExpression;
use zyntax_typed_ast::TypedNode;

/// Contract every built-in wrapper class implements. The class owns
/// two pieces of behaviour: a type predicate that says "I handle
/// dispatch for values of this type", and a method dispatch routine
/// that emits the HIR for a particular method call.
///
/// `dispatch` returns `Ok(None)` when the method is not recognised
/// on this class — the SSA `MethodCall` handler then falls through
/// to normal trait method resolution, which is the right behaviour
/// for user-defined extension methods on the built-in.
///
/// ## Generics
///
/// Most built-in types are generic — `Fiber<T>`, future
/// `SimdVector<T, N>`, etc. A single `BuiltinClass` implementation
/// handles every instantiation of its parametric type, NOT one
/// class per concrete T. The dispatch routine extracts the type
/// parameters from `receiver_ty` (or `result_ty`) and specialises
/// HIR emission accordingly:
///
/// ```ignore
/// fn dispatch(&self, ..., receiver_ty: &Type, ...) -> ... {
///     // For Fiber<T>, extract T:
///     let item_ty = match receiver_ty {
///         Type::Fiber(inner) => inner.as_ref(),
///         _ => unreachable!("matches() guarantees Type::Fiber"),
///     };
///     // ...specialise HIR emission based on `item_ty`...
/// }
/// ```
///
/// The class only fans out into multiple `impl BuiltinClass`
/// blocks if the instantiations need wildly different dispatch
/// behaviour (rare). For Fiber<T>, today's MVP packs the yielded
/// value into the upper bits of an i64 step; that encoding works
/// for any pointer-sized scalar T. Wider Ts (e.g. structs, tuples)
/// will need a different payload strategy — the dispatch routine
/// is the right place to handle that, branching on `item_ty`.
pub trait BuiltinClass {
    /// Human-readable name (used in diagnostics).
    fn name(&self) -> &str;

    /// Does this class own dispatch for values of `ty`? Implementors
    /// typically match on the outer type variant (e.g.
    /// `Type::Fiber(_)`) and accept any inner T; per-T
    /// specialisation lives in `dispatch`, not in `matches`.
    fn matches(&self, ty: &Type) -> bool;

    /// Emit the HIR for `receiver.<method>(args...)` returning
    /// `result_ty`. Returns the result `HirId`, or `Ok(None)` when
    /// the method is not part of the class's surface (the SSA
    /// caller then falls through to normal dispatch).
    ///
    /// `receiver_ty` is passed explicitly so the class has reliable
    /// access to the receiver's resolved type (including any
    /// generic type parameters) without depending on
    /// `receiver.ty`, which the type inference path may not have
    /// updated for variable references in the AST.
    fn dispatch(
        &self,
        ssa: &mut crate::ssa::SsaBuilder,
        block_id: HirId,
        method: &str,
        receiver: &TypedNode<TypedExpression>,
        receiver_ty: &Type,
        args: &[TypedNode<TypedExpression>],
        result_ty: &Type,
    ) -> CompilerResult<Option<HirId>>;
}

/// Registry of all built-in wrapper classes. Owned by the runtime
/// (`zyntax_embed::ZyntaxRuntime`) as an `Arc<BuiltinRegistry>` and
/// threaded into the SSA builder at construction. The compiler
/// ships a default set (`with_defaults`); embedders register
/// additional classes via the runtime's
/// `register_builtin_class(...)` API before any compilation runs.
///
/// Lookup is linear; the registry is small (a handful of entries)
/// so a hashmap would be premature. Each class declares its own
/// type predicate via `BuiltinClass::matches`, so order matters
/// only if predicates overlap — which they shouldn't, types are
/// disjoint by design.
pub struct BuiltinRegistry {
    classes: Vec<Arc<dyn BuiltinClass + Send + Sync>>,
}

impl BuiltinRegistry {
    /// Empty registry. Useful for tests where the default classes
    /// would interfere; production callers should use
    /// `with_defaults`.
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
        }
    }

    /// Registry pre-populated with the compiler's default built-in
    /// classes (Fiber<T>; future SimdVector<T,N>; ...). Embedders
    /// add to this list via `register`.
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(Arc::new(FiberClass));
        reg
    }

    /// Add a class to the registry. Called by `zyntax_embed`'s
    /// `runtime.register_builtin_class(...)` API before any
    /// compilation runs.
    pub fn register(&mut self, class: Arc<dyn BuiltinClass + Send + Sync>) {
        self.classes.push(class);
    }

    /// Iterate the registered classes. Used by `zyntax_embed` to
    /// snapshot the runtime registry into the lowering ctx.
    pub fn classes(&self) -> impl Iterator<Item = &Arc<dyn BuiltinClass + Send + Sync>> {
        self.classes.iter()
    }

    /// Find the wrapper class that owns dispatch for `ty`, if any.
    /// Returns `None` when `ty` is not a built-in. The returned
    /// `Arc` clone has its own lifetime independent of the registry,
    /// so the caller can take `&mut SsaBuilder` for the subsequent
    /// `dispatch` call without aliasing.
    pub fn class_for(&self, ty: &Type) -> Option<Arc<dyn BuiltinClass + Send + Sync>> {
        self.classes.iter().find(|c| c.matches(ty)).cloned()
    }
}

impl Default for BuiltinRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ────────────────────────────────────────────────────────────────
// Fiber<T> wrapper class
// ────────────────────────────────────────────────────────────────

/// Wrapper class for the compiler-known `Type::Fiber(_)` variant.
/// Methods declared via `impl<T> Iterator for Fiber<T> { ... }` in
/// the prelude are intercepted here; their stub bodies never run.
pub struct FiberClass;

impl BuiltinClass for FiberClass {
    fn name(&self) -> &str {
        "Fiber"
    }

    fn matches(&self, ty: &Type) -> bool {
        matches!(ty, Type::Fiber(_))
    }

    fn dispatch(
        &self,
        ssa: &mut crate::ssa::SsaBuilder,
        block_id: HirId,
        method: &str,
        receiver: &TypedNode<TypedExpression>,
        receiver_ty: &Type,
        _args: &[TypedNode<TypedExpression>],
        result_ty: &Type,
    ) -> CompilerResult<Option<HirId>> {
        // Single class handles every `Fiber<T>` instantiation. For
        // `next`, the result type is `Option<T>` after substitution
        // — but the parser defaults `TypedExpression::MethodCall`
        // expr.ty to `Type::Primitive(Unit)` when it can't infer
        // (which is most of the time for built-in dispatch), so we
        // can't trust `result_ty` directly. Synthesize `Option<T>`
        // from the receiver's `Fiber<T>` variant; fall back to
        // `result_ty` only if the receiver wasn't a Fiber (which
        // `matches()` should already preclude).
        match method {
            "next" => {
                let item_ty = match receiver_ty {
                    Type::Fiber(inner) => inner.as_ref().clone(),
                    _ => result_ty.clone(),
                };
                let option_ty = Type::Optional(Box::new(item_ty));
                ssa.emit_fiber_next(block_id, receiver, &option_ty)
                    .map(Some)
            }
            _ => Ok(None),
        }
    }
}
