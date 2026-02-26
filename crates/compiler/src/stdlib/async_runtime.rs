//! Async Runtime Types for Zyntax
//!
//! Provides the core types needed for async/await:
//! - Future trait (implemented via function-based API)
//! - Poll<T> enum (simplified as opaque type with helper functions)
//! - Context type
//! - Waker type
//!
//! ## Future Trait Status
//!
//! **Gap 7 trait dispatch infrastructure is COMPLETE** (vtables, dynamic dispatch, etc).
//! However, defining the Future trait in stdlib requires frontend syntax support for traits,
//! which is pending.
//!
//! **Current Approach** (function-based):
//! - `poll_ready<T>(value: T) -> Poll<T>` - Create Ready state
//! - `poll_pending<T>() -> Poll<T>` - Create Pending state
//! - `poll_is_ready<T>(poll: Poll<T>) -> bool` - Check if ready
//! - `poll_unwrap<T>(poll: Poll<T>) -> T` - Extract value
//!
//! **Future Approach** (once frontend supports trait syntax in stdlib):
//! ```ignore
//! trait Future {
//!     type Output;
//!     fn poll(&mut self, cx: &mut Context) -> Poll<Self::Output>;
//! }
//!
//! impl<T> Future for AsyncStateMachine<T> {
//!     type Output = T;
//!     fn poll(&mut self, cx: &mut Context) -> Poll<T> {
//!         // Call generated {func}_poll() wrapper
//!     }
//! }
//! ```
//!
//! The async state machine infrastructure is ready for Future trait integration once
//! stdlib trait syntax is supported.

use crate::hir::*;
use crate::hir_builder::HirBuilder;

/// Build the Poll enum type (simplified as opaque for now)
///
/// In the future this will be:
/// enum Poll<T> {
///     Ready(T),
///     Pending,
/// }
///
/// For now, we create helper functions that work with Poll values
pub fn build_poll_helpers(builder: &mut HirBuilder) {
    // Pre-intern type names to avoid borrow checker issues
    let t_name = builder.intern("T");
    let poll_name = builder.intern("Poll");

    // Build poll_ready<T>(value: T) -> Poll<T>
    // This creates a Poll in the Ready state
    builder
        .begin_generic_function("poll_ready", vec!["T"])
        .param("value", HirType::Opaque(t_name))
        .returns(HirType::Opaque(poll_name))
        .build();

    // Build poll_pending<T>() -> Poll<T>
    // This creates a Poll in the Pending state
    builder
        .begin_generic_function("poll_pending", vec!["T"])
        .returns(HirType::Opaque(poll_name))
        .build();

    // Build poll_is_ready<T>(poll: Poll<T>) -> bool
    builder
        .begin_generic_function("poll_is_ready", vec!["T"])
        .param("poll", HirType::Opaque(poll_name))
        .returns(HirType::Bool)
        .build();

    // Build poll_unwrap<T>(poll: Poll<T>) -> T
    builder
        .begin_generic_function("poll_unwrap", vec!["T"])
        .param("poll", HirType::Opaque(poll_name))
        .returns(HirType::Opaque(t_name))
        .build();
}

/// Build Waker helper functions
///
/// Wakers are opaque types for now. These functions provide the interface:
/// - waker_wake(waker: *Waker)
/// - waker_clone(waker: *Waker) -> *Waker
pub fn build_waker_functions(builder: &mut HirBuilder) {
    let waker_name = builder.intern("Waker");
    let waker_ty = HirType::Opaque(waker_name);

    // Build waker_wake(waker: *Waker)
    builder
        .begin_function("waker_wake")
        .param("waker", HirType::Ptr(Box::new(waker_ty.clone())))
        .returns(HirType::Void)
        .build();

    // Build waker_clone(waker: *Waker) -> *Waker
    builder
        .begin_function("waker_clone")
        .param("waker", HirType::Ptr(Box::new(waker_ty.clone())))
        .returns(HirType::Ptr(Box::new(waker_ty)))
        .build();
}

/// Build Context helper functions
///
/// Context is an opaque type containing the waker
/// - context_waker(cx: *Context) -> *Waker
pub fn build_context_functions(builder: &mut HirBuilder) {
    let context_name = builder.intern("Context");
    let waker_name = builder.intern("Waker");
    let context_ty = HirType::Opaque(context_name);
    let waker_ty = HirType::Opaque(waker_name);

    // Build context_waker(cx: *Context) -> *Waker
    builder
        .begin_function("context_waker")
        .param("cx", HirType::Ptr(Box::new(context_ty)))
        .returns(HirType::Ptr(Box::new(waker_ty)))
        .build();
}

/// Build all async runtime types and functions
///
/// This builds the complete async runtime infrastructure:
/// 1. Poll<T> helper functions
/// 2. Waker helper functions
/// 3. Context helper functions
///
/// NOTE: This is a simplified version. Full enum/trait support will be
/// added once the type system fully supports enums and traits.
pub fn build_async_runtime(builder: &mut HirBuilder) {
    // Build in dependency order
    build_poll_helpers(builder);
    build_waker_functions(builder);
    build_context_functions(builder);
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::AstArena;

    #[test]
    fn test_poll_helpers_build() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_async", &mut arena);

        build_poll_helpers(&mut builder);

        let module = builder.finish();

        // Verify Poll helper functions were created
        // Should have: poll_ready, poll_pending, poll_is_ready, poll_unwrap
        assert!(module.functions.len() >= 4);
    }

    #[test]
    fn test_waker_functions_build() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_async", &mut arena);

        build_waker_functions(&mut builder);

        let module = builder.finish();

        // Verify Waker functions were created
        // Should have: waker_wake, waker_clone
        assert!(module.functions.len() >= 2);
    }

    #[test]
    fn test_context_functions_build() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_async", &mut arena);

        build_context_functions(&mut builder);

        let module = builder.finish();

        // Verify Context functions were created
        // Should have: context_waker
        assert!(module.functions.len() >= 1);
    }

    #[test]
    fn test_complete_async_runtime() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test_async", &mut arena);

        build_async_runtime(&mut builder);

        let module = builder.finish();

        // Verify all async functions were created
        // Should have: 4 (poll) + 2 (waker) + 1 (context) = 7 functions minimum
        assert!(module.functions.len() >= 7);
    }
}
