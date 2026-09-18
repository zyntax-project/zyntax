//! Runtime function metadata for dynamic method mapping
//!
//! This module provides a way for runtime functions to declare metadata that
//! allows the compiler to automatically map Haxe methods/properties to runtime calls.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, LitStr, Token, parse::{Parse, ParseStream}};

/// Metadata for runtime function export with automatic method mapping
///
/// # Example
///
/// ```rust
/// #[runtime_method(
///     symbol = "$Array$push",
///     haxe_type = "Array<T>",
///     haxe_method = "push",
///     mutates = true,
///     returns_self = true
/// )]
/// pub extern "C" fn Array_push(arr: *mut i32, elem: i32) -> *mut i32 { ... }
/// ```
pub struct MethodMetadata {
    /// Runtime symbol name (e.g., "$Array$push")
    pub symbol: String,

    /// Haxe type this method belongs to (e.g., "Array<T>", "String")
    pub haxe_type: String,

    /// Haxe method/property name (e.g., "push", "length")
    pub haxe_name: String,

    /// Whether this is a property getter (vs a method)
    pub is_property: bool,

    /// Whether calling this mutates the receiver (important for realloc)
    pub mutates: bool,

    /// Whether this returns the receiver (for chaining after realloc)
    pub returns_self: bool,

    /// Parameter types (for validation)
    pub params: Vec<String>,
}

/// Parser for method metadata attributes
struct MethodMetadataArgs {
    symbol: String,
    haxe_type: String,
    haxe_name: String,
    is_property: bool,
    mutates: bool,
    returns_self: bool,
}

impl Parse for MethodMetadataArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut symbol = String::new();
        let mut haxe_type = String::new();
        let mut haxe_name = String::new();
        let mut is_property = false;
        let mut mutates = false;
        let mut returns_self = false;

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            let _ = input.parse::<Token![=]>()?;

            match ident.to_string().as_str() {
                "symbol" => {
                    let lit: syn::LitStr = input.parse()?;
                    symbol = lit.value();
                }
                "haxe_type" => {
                    let lit: syn::LitStr = input.parse()?;
                    haxe_type = lit.value();
                }
                "haxe_method" | "haxe_property" => {
                    let lit: syn::LitStr = input.parse()?;
                    haxe_name = lit.value();
                    is_property = ident == "haxe_property";
                }
                "mutates" => {
                    let lit: syn::LitBool = input.parse()?;
                    mutates = lit.value;
                }
                "returns_self" => {
                    let lit: syn::LitBool = input.parse()?;
                    returns_self = lit.value;
                }
                _ => {
                    // Skip unknown fields
                    let _: syn::Expr = input.parse()?;
                }
            }

            // Optional trailing comma
            let _ = input.parse::<Token![,]>();
        }

        Ok(MethodMetadataArgs {
            symbol,
            haxe_type,
            haxe_name,
            is_property,
            mutates,
            returns_self,
        })
    }
}

/// Attribute macro for runtime functions with method mapping metadata
///
/// This both exports the function AND registers metadata for compiler use.
///
/// # Example
///
/// ```rust
/// #[runtime_method(
///     symbol = "$Array$length",
///     haxe_type = "Array<T>",
///     haxe_property = "length"
/// )]
/// pub extern "C" fn Array_length(arr: *const i32) -> i32 { ... }
/// ```
#[proc_macro_attribute]
pub fn runtime_method(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as MethodMetadataArgs);
    let func = parse_macro_input!(item as ItemFn);

    let func_name = &func.sig.ident;
    let vis = &func.vis;
    let attrs = &func.attrs;
    let sig = &func.sig;
    let block = &func.block;

    let symbol_name = &args.symbol;
    let haxe_type = &args.haxe_type;
    let haxe_name = &args.haxe_name;
    let is_property = args.is_property;
    let mutates = args.mutates;
    let returns_self = args.returns_self;

    let expanded = quote! {
        #(#attrs)*
        #[unsafe(no_mangle)]
        #vis #sig #block

        // Register the runtime symbol
        inventory::submit! {
            crate::RuntimeSymbol {
                name: #symbol_name,
                ptr: crate::FunctionPtr::new(#func_name as *const u8),
            }
        }

        // Register the method mapping metadata
        inventory::submit! {
            crate::MethodMapping {
                symbol: #symbol_name,
                haxe_type: #haxe_type,
                haxe_name: #haxe_name,
                is_property: #is_property,
                mutates: #mutates,
                returns_self: #returns_self,
            }
        }
    };

    TokenStream::from(expanded)
}
