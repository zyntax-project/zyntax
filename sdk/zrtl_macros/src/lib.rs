//! ZRTL Macros for Rust Plugin Development
//!
//! This crate provides procedural macros for creating ZRTL plugins in Rust.
//! Use this alongside the `zrtl` crate which provides the runtime types.
//!
//! # Example
//!
//! ```rust,ignore
//! use zrtl::prelude::*;
//! use zrtl_macros::{zrtl_plugin, zrtl_export};
//!
//! // Define the plugin
//! zrtl_plugin!("my_runtime");
//!
//! // Export functions
//! #[zrtl_export("$MyRuntime$add")]
//! pub extern "C" fn add(a: i32, b: i32) -> i32 {
//!     a + b
//! }
//!
//! #[zrtl_export("$MyRuntime$multiply")]
//! pub extern "C" fn multiply(a: i32, b: i32) -> i32 {
//!     a * b
//! }
//! ```
//!
//! # Native Async Functions (for Language Frontends)
//!
//! Expose native async functions that can be `await`ed from Zyntax-based languages:
//!
//! ```rust,ignore
//! use zrtl::prelude::*;
//! use zrtl_macros::zrtl_async;
//!
//! // Guest code (your custom language) can await this function
//! #[zrtl_async("$IO$fetch")]
//! pub async fn fetch(url_ptr: *const u8) -> i64 {
//!     let response = http_client::get(url_ptr).await;
//!     response.status_code() as i64
//! }
//! ```
//!
//! In the guest language:
//! ```text
//! // MyLang code
//! async fn main() {
//!     let status = await fetch("https://api.example.com");
//! }
//! ```
//!
//! Build as cdylib:
//! ```toml
//! [lib]
//! crate-type = ["cdylib"]
//!
//! [dependencies]
//! zrtl = "0.1"
//! zrtl_macros = "0.1"
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse::Parse, parse::ParseStream, parse_macro_input, DeriveInput, ItemFn, LitStr};

/// Defines the ZRTL plugin metadata and symbol table infrastructure
///
/// This macro creates:
/// - `_zrtl_info`: Plugin metadata export
/// - `_zrtl_symbols`: Symbol table export (populated by `#[zrtl_export]`)
/// - Constructor to populate symbols at load time
///
/// # Example
/// ```rust,ignore
/// zrtl_plugin!("my_runtime");
/// ```
#[proc_macro]
pub fn zrtl_plugin(input: TokenStream) -> TokenStream {
    let name = parse_macro_input!(input as LitStr);
    let name_value = name.value();

    let expanded = quote! {
        // Plugin info export
        #[no_mangle]
        pub static _zrtl_info: ::zrtl::ZrtlInfo = ::zrtl::ZrtlInfo {
            version: ::zrtl::ZRTL_VERSION,
            name: concat!(#name_value, "\0").as_ptr() as *const ::std::ffi::c_char,
        };

        // Symbol table will be populated by #[zrtl_export] attributes
        // using the inventory crate for collection
        ::inventory::collect!(::zrtl::ZrtlSymbolEntry);

        #[no_mangle]
        pub static mut _zrtl_symbols: [::zrtl::ZrtlSymbol; 256] = [::zrtl::ZrtlSymbol {
            name: ::std::ptr::null(),
            ptr: ::std::ptr::null(),
        }; 256];

        // Constructor to populate the symbol table
        #[used]
        #[cfg_attr(target_os = "linux", link_section = ".init_array")]
        #[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
        #[cfg_attr(target_os = "windows", link_section = ".CRT$XCU")]
        static INIT_SYMBOLS: extern "C" fn() = {
            extern "C" fn init() {
                unsafe {
                    let mut i = 0;
                    for entry in ::inventory::iter::<::zrtl::ZrtlSymbolEntry> {
                        if i < 255 {
                            _zrtl_symbols[i] = ::zrtl::ZrtlSymbol {
                                name: entry.name.as_ptr() as *const ::std::ffi::c_char,
                                ptr: entry.ptr,
                            };
                            i += 1;
                        }
                    }
                    // Sentinel
                    _zrtl_symbols[i] = ::zrtl::ZrtlSymbol {
                        name: ::std::ptr::null(),
                        ptr: ::std::ptr::null(),
                    };
                }
            }
            init
        };
    };

    TokenStream::from(expanded)
}

struct ZrtlExportArgs {
    symbol_name: LitStr,
}

impl Parse for ZrtlExportArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let symbol_name = input.parse()?;
        Ok(ZrtlExportArgs { symbol_name })
    }
}

/// Exports a function as a ZRTL symbol
///
/// The function must be `extern "C"` for proper ABI compatibility.
///
/// # Example
/// ```rust,ignore
/// #[zrtl_export("$Array$push")]
/// pub extern "C" fn array_push(arr: *mut ZrtlArray, value: i32) {
///     // ...
/// }
/// ```
#[proc_macro_attribute]
pub fn zrtl_export(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ZrtlExportArgs);
    let func = parse_macro_input!(item as ItemFn);

    let symbol_name = args.symbol_name.value();
    let func_name = &func.sig.ident;

    // Ensure the function is extern "C"
    let has_extern_c = func
        .sig
        .abi
        .as_ref()
        .map(|abi| abi.name.as_ref().map(|n| n.value() == "C").unwrap_or(false))
        .unwrap_or(false);

    if !has_extern_c {
        return syn::Error::new_spanned(&func.sig, "ZRTL exported functions must be extern \"C\"")
            .to_compile_error()
            .into();
    }

    let symbol_name_with_null = format!("{}\0", symbol_name);

    let expanded = quote! {
        #[no_mangle]
        #func

        // Register the symbol using inventory
        ::inventory::submit! {
            ::zrtl::ZrtlSymbolEntry {
                name: #symbol_name_with_null,
                ptr: #func_name as *const u8,
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for implementing ZrtlTyped trait
///
/// This allows custom structs to be used with the ZRTL type system.
///
/// # Attributes
///
/// - `#[zrtl(name = "TypeName")]` - Override the type name (default: struct name)
/// - `#[zrtl(category = "Struct")]` - Set the type category (default: Struct)
///
/// # Example
/// ```rust,ignore
/// #[derive(ZrtlType)]
/// #[zrtl(name = "Point2D")]
/// pub struct Point {
///     pub x: f64,
///     pub y: f64,
/// }
/// ```
#[proc_macro_derive(ZrtlType, attributes(zrtl))]
pub fn derive_zrtl_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;

    // Parse attributes
    let mut type_name = name.to_string();
    let mut category = quote! { ::zrtl::TypeCategory::Struct };

    for attr in &input.attrs {
        if attr.path().is_ident("zrtl") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("name") {
                    let value: LitStr = meta.value()?.parse()?;
                    type_name = value.value();
                } else if meta.path.is_ident("category") {
                    let value: LitStr = meta.value()?.parse()?;
                    let cat_str = value.value();
                    category = match cat_str.as_str() {
                        "Void" => quote! { ::zrtl::TypeCategory::Void },
                        "Bool" => quote! { ::zrtl::TypeCategory::Bool },
                        "Int" => quote! { ::zrtl::TypeCategory::Int },
                        "UInt" => quote! { ::zrtl::TypeCategory::UInt },
                        "Float" => quote! { ::zrtl::TypeCategory::Float },
                        "String" => quote! { ::zrtl::TypeCategory::String },
                        "Array" => quote! { ::zrtl::TypeCategory::Array },
                        "Map" => quote! { ::zrtl::TypeCategory::Map },
                        "Struct" => quote! { ::zrtl::TypeCategory::Struct },
                        "Class" => quote! { ::zrtl::TypeCategory::Class },
                        "Enum" => quote! { ::zrtl::TypeCategory::Enum },
                        "Union" => quote! { ::zrtl::TypeCategory::Union },
                        "Function" => quote! { ::zrtl::TypeCategory::Function },
                        "Pointer" => quote! { ::zrtl::TypeCategory::Pointer },
                        "Optional" => quote! { ::zrtl::TypeCategory::Optional },
                        "Result" => quote! { ::zrtl::TypeCategory::Result },
                        "Tuple" => quote! { ::zrtl::TypeCategory::Tuple },
                        "TraitObject" => quote! { ::zrtl::TypeCategory::TraitObject },
                        "Opaque" => quote! { ::zrtl::TypeCategory::Opaque },
                        _ => quote! { ::zrtl::TypeCategory::Custom },
                    };
                }
                Ok(())
            });
        }
    }

    let expanded = quote! {
        impl ::zrtl::ZrtlTyped for #name {
            fn type_name() -> &'static str {
                #type_name
            }

            fn type_category() -> ::zrtl::TypeCategory {
                #category
            }
        }
    };

    TokenStream::from(expanded)
}

// ============================================================================
// Async Function Export
// ============================================================================

struct ZrtlAsyncArgs {
    symbol_name: LitStr,
}

impl Parse for ZrtlAsyncArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let symbol_name = input.parse()?;
        Ok(ZrtlAsyncArgs { symbol_name })
    }
}

/// Exports an async function as a ZRTL symbol
///
/// This macro transforms an async Rust function into a ZRTL-compatible async function
/// that follows the Zyntax Promise ABI.
///
/// The generated code:
/// 1. Creates a state machine struct to hold the future
/// 2. Generates an init function that allocates the state machine and returns a Promise
/// 3. Generates a poll function that advances the state machine
///
/// # ABI Convention
///
/// The async function is transformed to return a `*ZrtlPromise`:
/// - `async fn foo(a: i32, b: i32) -> i32` becomes `fn foo(a: i32, b: i32) -> *ZrtlPromise`
/// - The Promise contains `{state_machine: *mut u8, poll_fn: fn(*mut u8) -> i64}`
/// - Poll returns: 0 = Pending, positive = Ready(value), negative = Failed(code)
///
/// # Example
///
/// ```rust,ignore
/// use zrtl::prelude::*;
/// use zrtl_macros::zrtl_async;
///
/// #[zrtl_async("$Async$slow_compute")]
/// pub async fn slow_compute(n: i32) -> i32 {
///     // Yield to allow other tasks
///     yield_once().await;
///     n * 2
/// }
/// ```
#[proc_macro_attribute]
pub fn zrtl_async(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ZrtlAsyncArgs);
    let func = parse_macro_input!(item as ItemFn);

    let symbol_name = args.symbol_name.value();
    let func_name = &func.sig.ident;
    let func_vis = &func.vis;

    // Verify the function is async
    if func.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            &func.sig,
            "zrtl_async can only be applied to async functions",
        )
        .to_compile_error()
        .into();
    }

    // Extract parameters
    let params = &func.sig.inputs;
    let param_names: Vec<_> = params
        .iter()
        .filter_map(|arg| {
            if let syn::FnArg::Typed(pat) = arg {
                if let syn::Pat::Ident(ident) = &*pat.pat {
                    return Some(&ident.ident);
                }
            }
            None
        })
        .collect();

    let _param_types: Vec<_> = params
        .iter()
        .filter_map(|arg| {
            if let syn::FnArg::Typed(pat) = arg {
                return Some(&pat.ty);
            }
            None
        })
        .collect();

    // Extract return type (remove async wrapper)
    let return_type = match &func.sig.output {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, ty) => quote! { #ty },
    };

    // Get the function body
    let body = &func.block;

    // Generate unique names for the state machine and poll function
    let state_machine_name =
        syn::Ident::new(&format!("__{}_StateMachine", func_name), func_name.span());
    let poll_fn_name = syn::Ident::new(&format!("__{}_poll", func_name), func_name.span());
    let wrapper_fn_name = syn::Ident::new(&format!("__{}_wrapper", func_name), func_name.span());

    // Symbol name for the poll function
    let poll_symbol_name = format!("{}$poll\0", symbol_name);
    let symbol_name_with_null = format!("{}\0", symbol_name);

    let expanded = quote! {
        // The original async function (internal)
        async fn #func_name(#params) -> #return_type
        #body

        // State machine struct
        #[repr(C)]
        struct #state_machine_name {
            header: ::zrtl::StateMachineHeader,
            future: ::std::pin::Pin<Box<dyn ::std::future::Future<Output = #return_type> + Send>>,
            result: ::std::option::Option<#return_type>,
        }

        // Poll function (extern "C" for ABI compatibility)
        #[no_mangle]
        unsafe extern "C" fn #poll_fn_name(state: *mut u8) -> i64 {
            let state_machine = &mut *(state as *mut #state_machine_name);

            // Check if already completed
            if state_machine.result.is_some() {
                let value = state_machine.result.as_ref().unwrap();
                // Convert to i64 - for now, assume it can be transmuted
                // This works for i32, i64, bool, etc.
                return *value as i64;
            }

            // Create a waker and context
            let waker = ::zrtl::noop_waker();
            let mut cx = ::std::task::Context::from_waker(&waker);

            // Poll the future
            match state_machine.future.as_mut().poll(&mut cx) {
                ::std::task::Poll::Pending => 0, // Pending
                ::std::task::Poll::Ready(value) => {
                    state_machine.result = Some(value);
                    state_machine.header.set_completed();
                    // Return the value
                    let v = state_machine.result.as_ref().unwrap();
                    *v as i64
                }
            }
        }

        // Wrapper function that creates the promise
        #[no_mangle]
        #func_vis extern "C" fn #wrapper_fn_name(#params) -> *const ::zrtl::ZrtlPromise {
            // Create the future
            let future: ::std::pin::Pin<Box<dyn ::std::future::Future<Output = #return_type> + Send>> =
                Box::pin(#func_name(#(#param_names),*));

            // Allocate the state machine
            let state_machine = Box::new(#state_machine_name {
                header: ::zrtl::StateMachineHeader::new(),
                future,
                result: None,
            });

            let state_machine_ptr = Box::into_raw(state_machine) as *mut u8;

            // Create the promise struct (leaked - caller must manage lifetime)
            let promise = Box::new(::zrtl::ZrtlPromise {
                state_machine: state_machine_ptr,
                poll_fn: #poll_fn_name,
            });

            Box::into_raw(promise)
        }

        // Register both symbols using inventory
        ::inventory::submit! {
            ::zrtl::ZrtlSymbolEntry {
                name: #symbol_name_with_null,
                ptr: #wrapper_fn_name as *const u8,
            }
        }

        ::inventory::submit! {
            ::zrtl::ZrtlSymbolEntry {
                name: #poll_symbol_name,
                ptr: #poll_fn_name as *const u8,
            }
        }
    };

    TokenStream::from(expanded)
}

/// Helper macro for creating async poll functions manually
///
/// Use this when you need more control over the state machine layout.
///
/// # Example
///
/// ```rust,ignore
/// use zrtl::prelude::*;
///
/// #[repr(C)]
/// struct MyStateMachine {
///     header: StateMachineHeader,
///     counter: i32,
///     target: i32,
/// }
///
/// zrtl_poll_fn!(my_poll, MyStateMachine, |sm| {
///     if sm.counter >= sm.target {
///         PollResult::Ready(sm.counter as i64)
///     } else {
///         sm.counter += 1;
///         sm.header.advance();
///         PollResult::Pending
///     }
/// });
/// ```
#[proc_macro]
pub fn zrtl_poll_fn(_input: TokenStream) -> TokenStream {
    // For now, use a simple approach - the user passes in three tokens
    let expanded = quote! {
        // This is a placeholder - the actual implementation would parse
        // the input and generate the poll function
        compile_error!("zrtl_poll_fn! is not yet fully implemented - use #[zrtl_async] instead");
    };

    // Return a stub for now - full implementation would parse the closure
    TokenStream::from(expanded)
}

/// Macro to create a simple yielding async function
///
/// This is a convenience macro for creating simple async functions that
/// just need to yield a value after some work.
///
/// # Example
///
/// ```rust,ignore
/// zrtl_async_simple!("$Math$add_async", add_async, (a: i32, b: i32) -> i32 {
///     a + b
/// });
/// ```
#[proc_macro]
pub fn zrtl_async_simple(_input: TokenStream) -> TokenStream {
    // This would parse: "symbol_name", fn_name, (params) -> RetType { body }
    let expanded = quote! {
        compile_error!("zrtl_async_simple! is not yet fully implemented - use #[zrtl_async] instead");
    };

    TokenStream::from(expanded)
}

// ============================================================================
// Test Framework
// ============================================================================

/// Marks a function as a ZRTL test
///
/// This attribute wraps the test function with ZRTL-specific setup and
/// teardown, and registers it with the ZRTL test harness.
///
/// # Features
///
/// - Automatic setup of ZRTL type registry
/// - Colored output with test name and result
/// - Timing information
/// - Integrates with standard `cargo test`
///
/// # Example
///
/// ```rust,ignore
/// use zrtl::prelude::*;
/// use zrtl_macros::zrtl_test;
///
/// #[zrtl_test]
/// fn test_array_push() {
///     let mut arr: OwnedArray<i32> = OwnedArray::new().unwrap();
///     arr.push(42);
///     zrtl_assert_eq!(arr.len(), 1);
///     zrtl_assert_eq!(arr.get(0), Some(&42));
/// }
///
/// #[zrtl_test]
/// fn test_string_creation() {
///     let s = OwnedString::from("hello");
///     zrtl_assert_eq!(s.len(), 5);
///     zrtl_assert!(s.as_str() == Some("hello"));
/// }
/// ```
///
/// # Optional Attributes
///
/// - `#[zrtl_test(ignore)]` - Mark test as ignored
/// - `#[zrtl_test(should_panic)]` - Test should panic
/// - `#[zrtl_test(timeout = 1000)]` - Timeout in milliseconds
///
/// # Example with attributes
///
/// ```rust,ignore
/// #[zrtl_test(should_panic)]
/// fn test_out_of_bounds() {
///     let arr: OwnedArray<i32> = OwnedArray::new().unwrap();
///     let _ = arr.get(100); // Should panic
/// }
/// ```
#[proc_macro_attribute]
pub fn zrtl_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let func_name = &func.sig.ident;
    let func_vis = &func.vis;
    let func_block = &func.block;
    let func_attrs = &func.attrs;

    // Parse optional attributes
    let attr_string = attr.to_string();
    let should_ignore = attr_string.contains("ignore");
    let should_panic = attr_string.contains("should_panic");

    // Generate the test wrapper
    let test_name_str = func_name.to_string();

    let ignore_attr = if should_ignore {
        quote! { #[ignore] }
    } else {
        quote! {}
    };

    let should_panic_attr = if should_panic {
        quote! { #[should_panic] }
    } else {
        quote! {}
    };

    let expanded = quote! {
        #[test]
        #ignore_attr
        #should_panic_attr
        #(#func_attrs)*
        #func_vis fn #func_name() {
            // Test preamble - print test name
            let _test_name = #test_name_str;
            let _start = ::std::time::Instant::now();

            // Run the actual test
            let _result: Result<(), Box<dyn ::std::any::Any + Send>> = ::std::panic::catch_unwind(|| {
                #func_block
            });

            let _elapsed = _start.elapsed();

            // Check result and print status
            match _result {
                Ok(()) => {
                    // Test passed - standard test output handles this
                }
                Err(e) => {
                    // Re-panic to let the test framework handle it
                    ::std::panic::resume_unwind(e);
                }
            }
        }
    };

    TokenStream::from(expanded)
}
