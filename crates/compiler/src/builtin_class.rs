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
/// behaviour (rare). For `Fiber<T>`, today's MVP packs the yielded
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
    /// classes (`Fiber<T>`; future `SimdVector<T, N>`; ...). Embedders
    /// add to this list via `register`.
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(Arc::new(FiberClass));
        reg.register(Arc::new(VectorClass));
        reg.register(Arc::new(NumberClass));
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
            // `cancel()` — signals the fiber to stop. Subsequent
            // `next()` calls return `None`. First piece of the
            // Wren-style abort surface: a caller-side cancel signal
            // with no error payload.
            "cancel" => ssa.emit_fiber_cancel(block_id, receiver).map(Some),
            // `error()` — returns `Option<FiberError<T>>` carrying
            // the payload the fiber's `Fiber.abort(x)` passed
            // (None if the fiber completed normally or never
            // aborted). MVP: T is hardwired to String at the
            // decoder. Arbitrary T support needs the per-fiber E
            // inference pass; the runtime FFI already threads a
            // variant tag through so the upgrade is decoder-side
            // only when that pass lands.
            "error" => ssa
                .emit_fiber_error(block_id, receiver, result_ty)
                .map(Some),
            _ => Ok(None),
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Vector<T, N> wrapper class (first-class SIMD)
// ────────────────────────────────────────────────────────────────

/// Wrapper class for the compiler-known `Type::Vector(_, _)` variant.
///
/// Every method against a SIMD vector value dispatches here and lowers
/// to an inline vector instruction — horizontal reductions, element-
/// wise unary math, min/max, and lane access. Element-wise binary
/// arithmetic (`a + b`) and lane indexing (`v[i]`) are routed directly
/// from the `Binary` / `Index` expression handlers in the SSA builder
/// (a value's lowered `HirType::Vector` shape selects the vector path),
/// so they do not appear here.
///
/// A single class covers every `Type::Vector(elem, lanes)`
/// instantiation; the concrete element type and lane count are read
/// back from the lowered receiver value inside each emitter.
pub struct VectorClass;

impl BuiltinClass for VectorClass {
    fn name(&self) -> &str {
        "Vector"
    }

    fn matches(&self, ty: &Type) -> bool {
        matches!(ty, Type::Vector(_, _))
    }

    fn dispatch(
        &self,
        ssa: &mut crate::ssa::SsaBuilder,
        block_id: HirId,
        method: &str,
        receiver: &TypedNode<TypedExpression>,
        _receiver_ty: &Type,
        args: &[TypedNode<TypedExpression>],
        _result_ty: &Type,
    ) -> CompilerResult<Option<HirId>> {
        use crate::hir::{VectorMinMaxKind, VectorUnaryKind};

        match method {
            // Horizontal reductions. `sum`/`reduce` add all lanes; the
            // emitter defaults the op to the lane type's add.
            "sum" | "reduce" | "horizontal_sum" => {
                ssa.emit_vector_reduce(block_id, receiver, None).map(Some)
            }

            // Element-wise unary math: the method name selects the lane
            // operation, all sharing one emitter.
            "sqrt" => ssa_unary(ssa, block_id, receiver, VectorUnaryKind::Sqrt).map(Some),
            "abs" => ssa_unary(ssa, block_id, receiver, VectorUnaryKind::Abs).map(Some),
            "neg" => ssa_unary(ssa, block_id, receiver, VectorUnaryKind::Neg).map(Some),
            "ceil" => ssa_unary(ssa, block_id, receiver, VectorUnaryKind::Ceil).map(Some),
            "floor" => ssa_unary(ssa, block_id, receiver, VectorUnaryKind::Floor).map(Some),
            "trunc" => ssa_unary(ssa, block_id, receiver, VectorUnaryKind::Trunc).map(Some),
            "round" => ssa_unary(ssa, block_id, receiver, VectorUnaryKind::Round).map(Some),

            // Fused multiply-add `a.fma(b, c)` = `a * b + c` in one op.
            "fma" => match (args.first(), args.get(1)) {
                (Some(b), Some(c)) => ssa.emit_vector_fma(block_id, receiver, b, c).map(Some),
                _ => Ok(None),
            },

            // Quantized dot-accumulate `acc.dot_*(x, y)` = `acc + dot(x, y)`.
            // The suffix names the lane signedness in argument order, so
            // `dot_u8i8(x, y)` reads x as unsigned and y as signed. The
            // underlying operation puts its unsigned operand second, so the
            // arguments are handed over swapped; the per-lane products are
            // the same either way, only the widening differs.
            "dot_u8i8" => match (args.first(), args.get(1)) {
                (Some(x), Some(y)) => ssa
                    .emit_vector_dot(block_id, receiver, y, x, true, false)
                    .map(Some),
                _ => Ok(None),
            },
            "dot_i8i8" => match (args.first(), args.get(1)) {
                (Some(a), Some(b)) => ssa
                    .emit_vector_dot(block_id, receiver, a, b, false, false)
                    .map(Some),
                _ => Ok(None),
            },
            "dot_u8i7" => match (args.first(), args.get(1)) {
                (Some(x), Some(y)) => ssa
                    .emit_vector_dot(block_id, receiver, y, x, true, true)
                    .map(Some),
                _ => Ok(None),
            },

            "min" => match args.first() {
                Some(other) => ssa
                    .emit_vector_minmax(block_id, receiver, other, VectorMinMaxKind::Min)
                    .map(Some),
                None => Ok(None),
            },
            "max" => match args.first() {
                Some(other) => ssa
                    .emit_vector_minmax(block_id, receiver, other, VectorMinMaxKind::Max)
                    .map(Some),
                None => Ok(None),
            },

            // Lane access by method form, mirroring the `v[i]` sugar.
            "extract" | "lane" => match args.first() {
                Some(lane_expr) => {
                    let vector_val = ssa.translate_expression(block_id, receiver)?;
                    let lane_val = ssa.translate_expression(block_id, lane_expr)?;
                    let lane = ssa.const_lane_index(lane_val).ok_or_else(|| {
                        crate::CompilerError::Lowering("lane index must be a constant".to_string())
                    })?;
                    ssa.emit_vector_extract_lane(block_id, vector_val, lane)
                        .map(Some)
                }
                None => Ok(None),
            },

            // `v.insert(i, x)` — static lane, scalar value.
            "insert" => match (args.first(), args.get(1)) {
                (Some(lane_expr), Some(scalar_expr)) => ssa
                    .emit_vector_insert_lane(block_id, receiver, lane_expr, scalar_expr)
                    .map(Some),
                _ => Ok(None),
            },

            _ => Ok(None),
        }
    }
}

/// Static extensions on the primitive number types: the math methods
/// every numeric value carries (`x.sqrt()`, `x.abs()`). Each lowers to a
/// direct intrinsic, so the call site becomes a single hardware
/// instruction rather than a runtime call.
///
/// Declared in the prelude as static extensions (`impl f64 { … }`) so the
/// signatures are in scope; the implementation is the machine
/// instruction, so those declarations are bodyless.
pub struct NumberClass;

impl BuiltinClass for NumberClass {
    fn name(&self) -> &str {
        "Number"
    }

    fn matches(&self, ty: &Type) -> bool {
        matches!(ty, Type::Primitive(p) if p.is_numeric())
    }

    fn dispatch(
        &self,
        ssa: &mut crate::ssa::SsaBuilder,
        block_id: HirId,
        method: &str,
        receiver: &TypedNode<TypedExpression>,
        _receiver_ty: &Type,
        _args: &[TypedNode<TypedExpression>],
        _result_ty: &Type,
    ) -> CompilerResult<Option<HirId>> {
        use crate::hir::Intrinsic;

        match method {
            "sqrt" => ssa
                .emit_scalar_unary_intrinsic(block_id, receiver, Intrinsic::Sqrt)
                .map(Some),
            "abs" => ssa
                .emit_scalar_unary_intrinsic(block_id, receiver, Intrinsic::Fabs)
                .map(Some),
            _ => Ok(None),
        }
    }
}

/// Shared trampoline so each unary arm above stays a one-liner.
fn ssa_unary(
    ssa: &mut crate::ssa::SsaBuilder,
    block_id: HirId,
    receiver: &TypedNode<TypedExpression>,
    kind: crate::hir::VectorUnaryKind,
) -> CompilerResult<HirId> {
    ssa.emit_vector_unary(block_id, receiver, kind)
}
