//! Procedural macros for Zyntax plugin system
//!
//! This crate provides macros for declarative plugin registration using the inventory pattern.

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, ItemFn, LitBool, Token,
};

/// Attribute macro to export a runtime function with automatic plugin registration
///
/// # Example
///
/// ```rust
/// #[runtime_export("$Array$create")]
/// pub extern "C" fn Array_create(elem0: i32, elem1: i32) -> *mut i32 {
///     // implementation
/// }
/// ```
#[proc_macro_attribute]
pub fn runtime_export(attr: TokenStream, item: TokenStream) -> TokenStream {
    let symbol_name = parse_macro_input!(attr as syn::LitStr);
    let func = parse_macro_input!(item as ItemFn);

    let func_name = &func.sig.ident;
    let vis = &func.vis;
    let attrs = &func.attrs;
    let sig = &func.sig;
    let block = &func.block;

    // Use export_name to set the exact symbol name for both JIT and AOT linking
    // This ensures the function is exported as exactly "$haxe$trace$int" (or whatever name)
    let expanded = quote! {
        #(#attrs)*
        #[export_name = #symbol_name]
        #vis #sig #block

        // Register this symbol in the inventory for JIT runtime lookup
        inventory::submit! {
            crate::RuntimeSymbol {
                name: #symbol_name,
                ptr: crate::FunctionPtr::new(#func_name as *const u8),
            }
        }
    };

    TokenStream::from(expanded)
}

/// Parser for method metadata attributes
struct MethodArgs {
    symbol: String,
    haxe_type: String,
    haxe_name: String,
    is_property: bool,
    mutates: bool,
    returns_self: bool,
}

impl Parse for MethodArgs {
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
                "haxe_method" => {
                    let lit: syn::LitStr = input.parse()?;
                    haxe_name = lit.value();
                    is_property = false;
                }
                "haxe_property" => {
                    let lit: syn::LitStr = input.parse()?;
                    haxe_name = lit.value();
                    is_property = true;
                }
                "mutates" => {
                    let lit: LitBool = input.parse()?;
                    mutates = lit.value;
                }
                "returns_self" => {
                    let lit: LitBool = input.parse()?;
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

        Ok(MethodArgs {
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
/// This exports the function AND registers metadata for automatic method mapping.
///
/// # Example
///
/// ```rust
/// #[runtime_method(
///     symbol = "$Array$length",
///     haxe_type = "Array",
///     haxe_property = "length"
/// )]
/// pub extern "C" fn Array_length(arr: *const i32) -> i32 { ... }
/// ```
#[proc_macro_attribute]
pub fn runtime_method(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as MethodArgs);
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

    // Use export_name to set the exact symbol name for both JIT and AOT linking
    let expanded = quote! {
        #(#attrs)*
        #[export_name = #symbol_name]
        #vis #sig #block

        // Register the runtime symbol for JIT lookup
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

struct PluginArgs {
    name: String,
}

impl Parse for PluginArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = String::new();

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            let _ = input.parse::<Token![:]>()?;

            if ident == "name" {
                let lit: syn::LitStr = input.parse()?;
                name = lit.value();
            } else {
                // Skip unknown fields
                let _: syn::Expr = input.parse()?;
            }

            // Optional trailing comma
            let _ = input.parse::<Token![,]>();
        }

        Ok(PluginArgs { name })
    }
}

/// Macro to declare a runtime plugin module
///
/// This generates:
/// - RuntimeSymbol struct for inventory collection
/// - Plugin struct implementing RuntimePlugin trait
/// - get_plugin() function to retrieve the plugin instance
///
/// # Example
///
/// ```rust
/// runtime_plugin! {
///     name: "haxe",
/// }
/// ```
#[proc_macro]
pub fn runtime_plugin(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as PluginArgs);
    let plugin_struct_name = syn::Ident::new(
        &format!("{}Plugin", capitalize(&args.name)),
        proc_macro2::Span::call_site(),
    );

    let plugin_name = &args.name;

    let expanded = quote! {
        /// Runtime symbol for inventory-based registration
        ///
        /// FunctionPtr is a wrapper around *const u8 that is Send + Sync
        /// since function pointers are inherently thread-safe (they're immutable).
        pub struct FunctionPtr(*const u8);

        unsafe impl Send for FunctionPtr {}
        unsafe impl Sync for FunctionPtr {}

        impl FunctionPtr {
            pub const fn new(ptr: *const u8) -> Self {
                FunctionPtr(ptr)
            }

            pub fn as_ptr(&self) -> *const u8 {
                self.0
            }
        }

        pub struct RuntimeSymbol {
            pub name: &'static str,
            pub ptr: FunctionPtr,
        }

        inventory::collect!(RuntimeSymbol);

        /// Method mapping metadata for automatic Haxe → runtime mapping
        pub struct MethodMapping {
            pub symbol: &'static str,
            pub haxe_type: &'static str,
            pub haxe_name: &'static str,
            pub is_property: bool,
            pub mutates: bool,
            pub returns_self: bool,
        }

        inventory::collect!(MethodMapping);

        /// Plugin implementation
        pub struct #plugin_struct_name;

        impl zyntax_compiler::plugin::RuntimePlugin for #plugin_struct_name {
            fn name(&self) -> &str {
                #plugin_name
            }

            fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)> {
                inventory::iter::<RuntimeSymbol>
                    .into_iter()
                    .map(|sym| (sym.name, sym.ptr.as_ptr()))
                    .collect()
            }
        }

        /// Get the plugin instance for registration
        pub fn get_plugin() -> Box<dyn zyntax_compiler::plugin::RuntimePlugin> {
            Box::new(#plugin_struct_name)
        }

        /// Get all method mappings for compiler integration
        pub fn get_method_mappings() -> Vec<MethodMapping> {
            inventory::iter::<MethodMapping>
                .into_iter()
                .map(|m| MethodMapping {
                    symbol: m.symbol,
                    haxe_type: m.haxe_type,
                    haxe_name: m.haxe_name,
                    is_property: m.is_property,
                    mutates: m.mutates,
                    returns_self: m.returns_self,
                })
                .collect()
        }
    };

    TokenStream::from(expanded)
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().chain(chars).collect(),
    }
}
