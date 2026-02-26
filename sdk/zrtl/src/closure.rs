//! Closure Support for ZRTL
//!
//! Provides thread-safe closures that can be used with DynamicBox for
//! passing callable objects between Zyntax code and native runtime functions.
//!
//! ## Features
//!
//! - `ZrtlClosure`: Type-erased closure wrapper (C ABI compatible)
//! - `ZrtlFn`: Typed function pointer wrapper
//! - Thread-safe closures with `Send + 'static` bounds
//! - Integration with DynamicBox for dynamic dispatch
//!
//! ## Example
//!
//! ```rust,ignore
//! use zrtl::closure::{ZrtlClosure, ClosureResult};
//!
//! // Create a closure that captures state
//! let counter = Box::new(std::sync::atomic::AtomicI64::new(0));
//! let closure = ZrtlClosure::new(move |arg: i64| -> i64 {
//!     counter.fetch_add(arg, std::sync::atomic::Ordering::SeqCst)
//! });
//!
//! // Call it
//! let result = closure.call(5);
//! assert_eq!(result, ClosureResult::ok(0)); // Previous value
//! ```

use crate::dynamic_box::DynamicBox;
use crate::type_system::{TypeCategory, TypeFlags, TypeTag};
use std::sync::Arc;

/// Result of calling a closure
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClosureResult {
    /// Result value (i64 for compatibility)
    pub value: i64,
    /// Error code (0 = success, non-zero = error)
    pub error: i32,
}

impl ClosureResult {
    /// Create a successful result
    #[inline]
    pub const fn ok(value: i64) -> Self {
        Self { value, error: 0 }
    }

    /// Create an error result
    #[inline]
    pub const fn err(error: i32) -> Self {
        Self { value: 0, error }
    }

    /// Check if this is a success
    #[inline]
    pub fn is_ok(&self) -> bool {
        self.error == 0
    }

    /// Check if this is an error
    #[inline]
    pub fn is_err(&self) -> bool {
        self.error != 0
    }

    /// Get the value if successful
    #[inline]
    pub fn value(&self) -> Option<i64> {
        if self.is_ok() {
            Some(self.value)
        } else {
            None
        }
    }
}

impl Default for ClosureResult {
    fn default() -> Self {
        Self::ok(0)
    }
}

/// Error codes for closure operations
pub mod error {
    pub const SUCCESS: i32 = 0;
    pub const NULL_CLOSURE: i32 = -1;
    pub const CALL_FAILED: i32 = -2;
    pub const TYPE_MISMATCH: i32 = -3;
    pub const ALREADY_CONSUMED: i32 = -4;
}

/// Internal trait for type-erased closures
trait ClosureInner: Send + Sync {
    fn call(&self, arg: i64) -> i64;
    fn call_void(&self, arg: i64);
}

/// Implementation for Fn(i64) -> i64
struct ClosureFnI64<F: Fn(i64) -> i64 + Send + Sync + 'static>(F);

impl<F: Fn(i64) -> i64 + Send + Sync + 'static> ClosureInner for ClosureFnI64<F> {
    fn call(&self, arg: i64) -> i64 {
        (self.0)(arg)
    }
    fn call_void(&self, arg: i64) {
        let _ = (self.0)(arg);
    }
}

/// Implementation for Fn(i64) (void return)
struct ClosureFnVoid<F: Fn(i64) + Send + Sync + 'static>(F);

impl<F: Fn(i64) + Send + Sync + 'static> ClosureInner for ClosureFnVoid<F> {
    fn call(&self, arg: i64) -> i64 {
        (self.0)(arg);
        0
    }
    fn call_void(&self, arg: i64) {
        (self.0)(arg);
    }
}

/// Implementation for Fn() -> i64 (no arg)
struct ClosureFnNoArg<F: Fn() -> i64 + Send + Sync + 'static>(F);

impl<F: Fn() -> i64 + Send + Sync + 'static> ClosureInner for ClosureFnNoArg<F> {
    fn call(&self, _arg: i64) -> i64 {
        (self.0)()
    }
    fn call_void(&self, _arg: i64) {
        let _ = (self.0)();
    }
}

/// Implementation for Fn() (no arg, void return)
struct ClosureFnNoArgVoid<F: Fn() + Send + Sync + 'static>(F);

impl<F: Fn() + Send + Sync + 'static> ClosureInner for ClosureFnNoArgVoid<F> {
    fn call(&self, _arg: i64) -> i64 {
        (self.0)();
        0
    }
    fn call_void(&self, _arg: i64) {
        (self.0)();
    }
}

/// Type-erased, thread-safe closure wrapper
///
/// This is the main closure type for ZRTL. It can hold any `Fn` closure
/// that is `Send + Sync + 'static`, making it safe to pass across threads.
///
/// The closure uses a uniform ABI: `fn(i64) -> i64` for simplicity.
/// Complex types should be passed via pointers encoded as i64.
#[repr(C)]
pub struct ZrtlClosure {
    /// Arc pointer to the inner closure (type-erased)
    inner: *const (),
    /// Call function pointer
    call_fn: extern "C" fn(*const (), i64) -> i64,
    /// Void call function pointer
    call_void_fn: extern "C" fn(*const (), i64),
    /// Drop function pointer
    drop_fn: extern "C" fn(*const ()),
    /// Clone function pointer (increments Arc refcount)
    clone_fn: extern "C" fn(*const ()) -> *const (),
}

// ============================================================================
// Raw Closure Support (for compiler-generated closures)
// ============================================================================

/// Raw closure type - a function pointer + environment pointer pair
///
/// This matches the layout of compiler-generated closures:
/// ```text
/// [fn_ptr: 8 bytes][env_ptr/captures: ...]
/// ```
///
/// The function signature is: `fn(env: *mut u8, arg: i64) -> i64`
pub type RawClosureFn = extern "C" fn(*mut u8, i64) -> i64;

/// A raw closure wrapper for compiler-generated closures
///
/// This holds ownership of the environment data and provides
/// the ZrtlClosure interface.
struct RawClosureInner {
    func: RawClosureFn,
    env: *mut u8,
    env_size: usize,
}

impl Drop for RawClosureInner {
    fn drop(&mut self) {
        if !self.env.is_null() && self.env_size > 0 {
            unsafe {
                let layout = std::alloc::Layout::from_size_align(self.env_size, 8).unwrap();
                std::alloc::dealloc(self.env, layout);
            }
        }
    }
}

// SAFETY: The raw closure function and env are designed to be thread-safe
unsafe impl Send for RawClosureInner {}
unsafe impl Sync for RawClosureInner {}

// Wrapper to implement ClosureInner for RawClosureInner
impl ClosureInner for RawClosureInner {
    fn call(&self, arg: i64) -> i64 {
        (self.func)(self.env, arg)
    }
    fn call_void(&self, arg: i64) {
        let _ = (self.func)(self.env, arg);
    }
}

// ============================================================================
// C ABI function implementations
// ============================================================================

extern "C" fn closure_call<C: ClosureInner + 'static>(ptr: *const (), arg: i64) -> i64 {
    let arc = unsafe { &*(ptr as *const Arc<C>) };
    arc.call(arg)
}

extern "C" fn closure_call_void<C: ClosureInner + 'static>(ptr: *const (), arg: i64) {
    let arc = unsafe { &*(ptr as *const Arc<C>) };
    arc.call_void(arg);
}

extern "C" fn closure_drop<C: ClosureInner + 'static>(ptr: *const ()) {
    unsafe {
        let _ = Box::from_raw(ptr as *mut Arc<C>);
    }
}

extern "C" fn closure_clone<C: ClosureInner + 'static>(ptr: *const ()) -> *const () {
    let arc = unsafe { &*(ptr as *const Arc<C>) };
    let cloned = Box::new(Arc::clone(arc));
    Box::into_raw(cloned) as *const ()
}

impl ZrtlClosure {
    /// Create a new closure from a function `Fn(i64) -> i64`
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(i64) -> i64 + Send + Sync + 'static,
    {
        let inner = ClosureFnI64(f);
        let arc = Box::new(Arc::new(inner));
        let ptr = Box::into_raw(arc) as *const ();

        Self {
            inner: ptr,
            call_fn: closure_call::<ClosureFnI64<F>>,
            call_void_fn: closure_call_void::<ClosureFnI64<F>>,
            drop_fn: closure_drop::<ClosureFnI64<F>>,
            clone_fn: closure_clone::<ClosureFnI64<F>>,
        }
    }

    /// Create a closure from a void function `Fn(i64)`
    pub fn from_void<F>(f: F) -> Self
    where
        F: Fn(i64) + Send + Sync + 'static,
    {
        let inner = ClosureFnVoid(f);
        let arc = Box::new(Arc::new(inner));
        let ptr = Box::into_raw(arc) as *const ();

        Self {
            inner: ptr,
            call_fn: closure_call::<ClosureFnVoid<F>>,
            call_void_fn: closure_call_void::<ClosureFnVoid<F>>,
            drop_fn: closure_drop::<ClosureFnVoid<F>>,
            clone_fn: closure_clone::<ClosureFnVoid<F>>,
        }
    }

    /// Create a closure from a no-arg function `Fn() -> i64`
    pub fn from_no_arg<F>(f: F) -> Self
    where
        F: Fn() -> i64 + Send + Sync + 'static,
    {
        let inner = ClosureFnNoArg(f);
        let arc = Box::new(Arc::new(inner));
        let ptr = Box::into_raw(arc) as *const ();

        Self {
            inner: ptr,
            call_fn: closure_call::<ClosureFnNoArg<F>>,
            call_void_fn: closure_call_void::<ClosureFnNoArg<F>>,
            drop_fn: closure_drop::<ClosureFnNoArg<F>>,
            clone_fn: closure_clone::<ClosureFnNoArg<F>>,
        }
    }

    /// Create a closure from a thunk `Fn()`
    pub fn from_thunk<F>(f: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        let inner = ClosureFnNoArgVoid(f);
        let arc = Box::new(Arc::new(inner));
        let ptr = Box::into_raw(arc) as *const ();

        Self {
            inner: ptr,
            call_fn: closure_call::<ClosureFnNoArgVoid<F>>,
            call_void_fn: closure_call_void::<ClosureFnNoArgVoid<F>>,
            drop_fn: closure_drop::<ClosureFnNoArgVoid<F>>,
            clone_fn: closure_clone::<ClosureFnNoArgVoid<F>>,
        }
    }

    /// Create from a raw function pointer and environment
    ///
    /// This is the primary way for the compiler to create ZrtlClosures.
    /// The environment data is copied, so the original can be freed.
    ///
    /// # Arguments
    /// * `func` - Function pointer with signature `fn(env: *mut u8, arg: i64) -> i64`
    /// * `env` - Pointer to environment/captured data (can be null if env_size is 0)
    /// * `env_size` - Size of environment data in bytes
    ///
    /// # Safety
    /// * `func` must be a valid function pointer
    /// * If `env_size > 0`, `env` must point to valid memory of at least `env_size` bytes
    pub unsafe fn from_raw(func: RawClosureFn, env: *const u8, env_size: usize) -> Self {
        // Allocate and copy environment data
        let env_copy = if env_size > 0 && !env.is_null() {
            let layout = std::alloc::Layout::from_size_align(env_size, 8).unwrap();
            let ptr = std::alloc::alloc(layout);
            std::ptr::copy_nonoverlapping(env, ptr, env_size);
            ptr
        } else {
            std::ptr::null_mut()
        };

        let inner = RawClosureInner {
            func,
            env: env_copy,
            env_size,
        };

        let arc = Box::new(Arc::new(inner));
        let ptr = Box::into_raw(arc) as *const ();

        Self {
            inner: ptr,
            call_fn: closure_call::<RawClosureInner>,
            call_void_fn: closure_call_void::<RawClosureInner>,
            drop_fn: closure_drop::<RawClosureInner>,
            clone_fn: closure_clone::<RawClosureInner>,
        }
    }

    /// Check if the closure is null
    #[inline]
    pub fn is_null(&self) -> bool {
        self.inner.is_null()
    }

    /// Call the closure with an argument
    #[inline]
    pub fn call(&self, arg: i64) -> ClosureResult {
        if self.inner.is_null() {
            return ClosureResult::err(error::NULL_CLOSURE);
        }
        let value = (self.call_fn)(self.inner, arg);
        ClosureResult::ok(value)
    }

    /// Call the closure without caring about the return value
    #[inline]
    pub fn call_void(&self, arg: i64) {
        if !self.inner.is_null() {
            (self.call_void_fn)(self.inner, arg);
        }
    }

    /// Call with no argument (arg = 0)
    #[inline]
    pub fn invoke(&self) -> ClosureResult {
        self.call(0)
    }

    /// Create a null/empty closure
    pub const fn null() -> Self {
        Self {
            inner: std::ptr::null(),
            call_fn: null_call,
            call_void_fn: null_call_void,
            drop_fn: null_drop,
            clone_fn: null_clone,
        }
    }
}

// Null closure stubs
extern "C" fn null_call(_: *const (), _: i64) -> i64 {
    0
}
extern "C" fn null_call_void(_: *const (), _: i64) {}
extern "C" fn null_drop(_: *const ()) {}
extern "C" fn null_clone(ptr: *const ()) -> *const () {
    ptr
}

impl Clone for ZrtlClosure {
    fn clone(&self) -> Self {
        if self.inner.is_null() {
            return Self::null();
        }
        let cloned_inner = (self.clone_fn)(self.inner);
        Self {
            inner: cloned_inner,
            call_fn: self.call_fn,
            call_void_fn: self.call_void_fn,
            drop_fn: self.drop_fn,
            clone_fn: self.clone_fn,
        }
    }
}

impl Drop for ZrtlClosure {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            (self.drop_fn)(self.inner);
            self.inner = std::ptr::null();
        }
    }
}

// SAFETY: ZrtlClosure only holds closures that are Send + Sync
unsafe impl Send for ZrtlClosure {}
unsafe impl Sync for ZrtlClosure {}

impl std::fmt::Debug for ZrtlClosure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZrtlClosure")
            .field("inner", &self.inner)
            .field("is_null", &self.is_null())
            .finish()
    }
}

// ============================================================================
// DynamicBox Integration
// ============================================================================

/// Type tag for closures
impl TypeTag {
    /// Closure type tag (category: Function, type_id: 1 for ZrtlClosure)
    pub const CLOSURE: Self = Self::new(TypeCategory::Function, 1, TypeFlags::NONE);
}

impl DynamicBox {
    /// Create a DynamicBox containing a closure
    pub fn from_closure(closure: ZrtlClosure) -> Self {
        Self::from_value(TypeTag::CLOSURE, closure)
    }

    /// Check if this box contains a closure
    #[inline]
    pub fn is_closure(&self) -> bool {
        self.tag == TypeTag::CLOSURE || self.is_category(TypeCategory::Function)
    }

    /// Get as closure reference (type-checked)
    pub fn as_closure(&self) -> Option<&ZrtlClosure> {
        if self.is_closure() {
            unsafe { self.as_ref::<ZrtlClosure>() }
        } else {
            None
        }
    }

    /// Get as mutable closure reference (type-checked)
    pub fn as_closure_mut(&mut self) -> Option<&mut ZrtlClosure> {
        if self.is_closure() {
            unsafe { self.as_mut::<ZrtlClosure>() }
        } else {
            None
        }
    }

    /// Call the contained closure with an argument
    pub fn call_closure(&self, arg: i64) -> ClosureResult {
        match self.as_closure() {
            Some(closure) => closure.call(arg),
            None => ClosureResult::err(error::TYPE_MISMATCH),
        }
    }

    /// Call the contained closure with no argument
    pub fn invoke_closure(&self) -> ClosureResult {
        self.call_closure(0)
    }
}

// ============================================================================
// C ABI Functions for External Use
// ============================================================================

/// Create a new closure from a function pointer (C ABI)
///
/// # Safety
/// The function pointer must be valid for the lifetime of the closure.
#[no_mangle]
pub extern "C" fn zrtl_closure_from_fn(func: extern "C" fn(i64) -> i64) -> *mut ZrtlClosure {
    let closure = ZrtlClosure::new(move |arg| func(arg));
    Box::into_raw(Box::new(closure))
}

/// Call a closure (C ABI)
///
/// # Safety
/// The pointer must be a valid ZrtlClosure pointer.
#[no_mangle]
pub unsafe extern "C" fn zrtl_closure_call(closure: *const ZrtlClosure, arg: i64) -> ClosureResult {
    if closure.is_null() {
        return ClosureResult::err(error::NULL_CLOSURE);
    }
    (*closure).call(arg)
}

/// Clone a closure (C ABI)
///
/// # Safety
/// The pointer must be a valid ZrtlClosure pointer.
#[no_mangle]
pub unsafe extern "C" fn zrtl_closure_clone(closure: *const ZrtlClosure) -> *mut ZrtlClosure {
    if closure.is_null() {
        return std::ptr::null_mut();
    }
    let cloned = (*closure).clone();
    Box::into_raw(Box::new(cloned))
}

/// Free a closure (C ABI)
///
/// # Safety
/// The pointer must be a valid ZrtlClosure pointer that was created by zrtl_closure_* functions.
#[no_mangle]
pub unsafe extern "C" fn zrtl_closure_free(closure: *mut ZrtlClosure) {
    if !closure.is_null() {
        let _ = Box::from_raw(closure);
    }
}

/// Check if a closure is null (C ABI)
///
/// # Safety
/// The pointer must be a valid ZrtlClosure pointer or null.
#[no_mangle]
pub unsafe extern "C" fn zrtl_closure_is_null(closure: *const ZrtlClosure) -> i32 {
    if closure.is_null() || (*closure).is_null() {
        1
    } else {
        0
    }
}

/// Create a closure from raw function pointer and environment (C ABI)
///
/// This is the primary entry point for the compiler to create ZrtlClosures.
/// The compiler generates:
/// 1. A function with signature `fn(env: *mut u8, arg: i64) -> i64`
/// 2. An environment struct containing captured values
/// 3. Calls this function to wrap them into a ZrtlClosure
///
/// # Arguments
/// * `func` - Function pointer with signature `fn(env: *mut u8, arg: i64) -> i64`
/// * `env` - Pointer to environment/captured data
/// * `env_size` - Size of environment data in bytes (0 if no captures)
///
/// # Returns
/// A heap-allocated ZrtlClosure pointer. Caller must free with `zrtl_closure_free`.
///
/// # Safety
/// * `func` must be a valid function pointer
/// * If `env_size > 0`, `env` must point to valid memory of at least `env_size` bytes
#[no_mangle]
pub unsafe extern "C" fn zrtl_closure_from_raw(
    func: RawClosureFn,
    env: *const u8,
    env_size: usize,
) -> *mut ZrtlClosure {
    let closure = ZrtlClosure::from_raw(func, env, env_size);
    Box::into_raw(Box::new(closure))
}

/// Create a closure from raw function pointer without environment (C ABI)
///
/// Simplified version for closures with no captured state.
///
/// # Arguments
/// * `func` - Function pointer with signature `fn(env: *mut u8, arg: i64) -> i64`
///           The env parameter will be null when called.
///
/// # Returns
/// A heap-allocated ZrtlClosure pointer. Caller must free with `zrtl_closure_free`.
///
/// # Safety
/// * `func` must be a valid function pointer
#[no_mangle]
pub unsafe extern "C" fn zrtl_closure_from_raw_noenv(func: RawClosureFn) -> *mut ZrtlClosure {
    let closure = ZrtlClosure::from_raw(func, std::ptr::null(), 0);
    Box::into_raw(Box::new(closure))
}

// ============================================================================
// OnceCell-based Closures (FnOnce support)
// ============================================================================

use std::sync::Mutex;

/// A closure that can only be called once (FnOnce semantics)
///
/// This is useful for thread spawn functions where the closure
/// consumes captured values.
pub struct ZrtlOnceClosure {
    inner: Mutex<Option<Box<dyn FnOnce(i64) -> i64 + Send + 'static>>>,
}

impl ZrtlOnceClosure {
    /// Create a new once-callable closure
    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce(i64) -> i64 + Send + 'static,
    {
        Self {
            inner: Mutex::new(Some(Box::new(f))),
        }
    }

    /// Create from a void closure
    pub fn from_void<F>(f: F) -> Self
    where
        F: FnOnce(i64) + Send + 'static,
    {
        Self {
            inner: Mutex::new(Some(Box::new(move |arg| {
                f(arg);
                0
            }))),
        }
    }

    /// Create from a thunk (no args, no return)
    pub fn from_thunk<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            inner: Mutex::new(Some(Box::new(move |_| {
                f();
                0
            }))),
        }
    }

    /// Call the closure, consuming it
    ///
    /// Returns error if already consumed or lock poisoned
    pub fn call(&self, arg: i64) -> ClosureResult {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return ClosureResult::err(error::CALL_FAILED),
        };

        match guard.take() {
            Some(f) => ClosureResult::ok(f(arg)),
            None => ClosureResult::err(error::ALREADY_CONSUMED),
        }
    }

    /// Check if the closure has been consumed
    pub fn is_consumed(&self) -> bool {
        match self.inner.lock() {
            Ok(guard) => guard.is_none(),
            Err(_) => true,
        }
    }
}

// SAFETY: The inner Mutex provides synchronization
unsafe impl Sync for ZrtlOnceClosure {}

impl std::fmt::Debug for ZrtlOnceClosure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZrtlOnceClosure")
            .field("consumed", &self.is_consumed())
            .finish()
    }
}

// ============================================================================
// Thread-compatible closure handle for spawn
// ============================================================================

/// A thread entry point that can be passed to thread::spawn
///
/// This wraps a ZrtlClosure and makes it callable as a thread entry function.
pub struct ThreadEntry {
    closure: ZrtlClosure,
    arg: i64,
}

impl ThreadEntry {
    /// Create a new thread entry
    pub fn new(closure: ZrtlClosure, arg: i64) -> Self {
        Self { closure, arg }
    }

    /// Run the thread entry, returning the result
    pub fn run(self) -> i64 {
        match self.closure.call(self.arg) {
            r if r.is_ok() => r.value,
            _ => i64::MIN,
        }
    }
}

// SAFETY: ZrtlClosure is Send + Sync
unsafe impl Send for ThreadEntry {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};

    #[test]
    fn test_basic_closure() {
        let closure = ZrtlClosure::new(|x| x * 2);

        let result = closure.call(21);
        assert!(result.is_ok());
        assert_eq!(result.value(), Some(42));
    }

    #[test]
    fn test_capturing_closure() {
        let counter = Arc::new(AtomicI64::new(0));
        let counter_clone = counter.clone();

        let closure = ZrtlClosure::new(move |x| counter_clone.fetch_add(x, Ordering::SeqCst));

        assert_eq!(closure.call(10).value(), Some(0));
        assert_eq!(closure.call(5).value(), Some(10));
        assert_eq!(counter.load(Ordering::SeqCst), 15);
    }

    #[test]
    fn test_clone_closure() {
        let counter = Arc::new(AtomicI64::new(0));
        let counter_clone = counter.clone();

        let closure = ZrtlClosure::new(move |x| counter_clone.fetch_add(x, Ordering::SeqCst));

        let cloned = closure.clone();

        // Both closures share the same state
        closure.call(5);
        cloned.call(10);

        assert_eq!(counter.load(Ordering::SeqCst), 15);
    }

    #[test]
    fn test_void_closure() {
        let called = Arc::new(AtomicI64::new(0));
        let called_clone = called.clone();

        let closure = ZrtlClosure::from_void(move |x| {
            called_clone.store(x, Ordering::SeqCst);
        });

        closure.call_void(42);
        assert_eq!(called.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn test_thunk_closure() {
        let called = Arc::new(AtomicI64::new(0));
        let called_clone = called.clone();

        let closure = ZrtlClosure::from_thunk(move || {
            called_clone.store(1, Ordering::SeqCst);
        });

        closure.invoke();
        assert_eq!(called.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_null_closure() {
        let closure = ZrtlClosure::null();
        assert!(closure.is_null());

        let result = closure.call(0);
        assert!(result.is_err());
        assert_eq!(result.error, error::NULL_CLOSURE);
    }

    #[test]
    fn test_dynamic_box_closure() {
        let closure = ZrtlClosure::new(|x| x + 100);
        let boxed = DynamicBox::from_closure(closure);

        assert!(boxed.is_closure());
        assert_eq!(boxed.tag, TypeTag::CLOSURE);

        let result = boxed.call_closure(42);
        assert!(result.is_ok());
        assert_eq!(result.value(), Some(142));
    }

    #[test]
    fn test_once_closure() {
        let once = ZrtlOnceClosure::new(|x| x * 3);

        // First call succeeds
        let result = once.call(10);
        assert!(result.is_ok());
        assert_eq!(result.value(), Some(30));

        // Second call fails
        let result = once.call(10);
        assert!(result.is_err());
        assert_eq!(result.error, error::ALREADY_CONSUMED);
    }

    #[test]
    fn test_thread_entry() {
        let closure = ZrtlClosure::new(|x| x * 2);
        let entry = ThreadEntry::new(closure, 21);

        let handle = std::thread::spawn(move || entry.run());
        let result = handle.join().unwrap();

        assert_eq!(result, 42);
    }

    #[test]
    fn test_closure_across_threads() {
        let counter = Arc::new(AtomicI64::new(0));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let counter = counter.clone();
                let closure = ZrtlClosure::new(move |x| counter.fetch_add(x, Ordering::SeqCst));

                std::thread::spawn(move || closure.call(i + 1).value().unwrap())
            })
            .collect();

        for handle in handles {
            let _ = handle.join().unwrap();
        }

        // 1 + 2 + 3 + 4 = 10
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_raw_closure_no_env() {
        // Simulate a compiler-generated closure with no captures
        extern "C" fn raw_double(_env: *mut u8, arg: i64) -> i64 {
            arg * 2
        }

        unsafe {
            let closure = ZrtlClosure::from_raw(raw_double, std::ptr::null(), 0);

            let result = closure.call(21);
            assert!(result.is_ok());
            assert_eq!(result.value(), Some(42));
        }
    }

    #[test]
    fn test_raw_closure_with_env() {
        // Simulate a compiler-generated closure with captured state
        // Environment layout: [multiplier: i64]
        #[repr(C)]
        struct Env {
            multiplier: i64,
        }

        extern "C" fn raw_multiply(env: *mut u8, arg: i64) -> i64 {
            let env = unsafe { &*(env as *const Env) };
            arg * env.multiplier
        }

        let env = Env { multiplier: 3 };

        unsafe {
            let closure = ZrtlClosure::from_raw(
                raw_multiply,
                &env as *const Env as *const u8,
                std::mem::size_of::<Env>(),
            );

            let result = closure.call(14);
            assert!(result.is_ok());
            assert_eq!(result.value(), Some(42)); // 14 * 3 = 42
        }
    }

    #[test]
    fn test_raw_closure_c_abi() {
        // Test the C ABI function
        extern "C" fn raw_add(_env: *mut u8, arg: i64) -> i64 {
            arg + 100
        }

        unsafe {
            let closure_ptr = zrtl_closure_from_raw_noenv(raw_add);
            assert!(!closure_ptr.is_null());

            let result = zrtl_closure_call(closure_ptr, 42);
            assert!(result.is_ok());
            assert_eq!(result.value(), Some(142));

            zrtl_closure_free(closure_ptr);
        }
    }

    #[test]
    fn test_raw_closure_thread() {
        // Test using raw closure with thread spawn
        extern "C" fn raw_square(_env: *mut u8, arg: i64) -> i64 {
            arg * arg
        }

        unsafe {
            let closure = ZrtlClosure::from_raw(raw_square, std::ptr::null(), 0);
            let entry = ThreadEntry::new(closure, 7);

            let handle = std::thread::spawn(move || entry.run());
            let result = handle.join().unwrap();

            assert_eq!(result, 49); // 7 * 7 = 49
        }
    }
}
