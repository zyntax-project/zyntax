//! Procedural macros for Zyntax plugin system
//!
//! This crate provides macros for declarative plugin registration using the inventory pattern.

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    ItemFn, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

/// Attribute macro to export a runtime function with automatic plugin registration
///
/// # Example
///
/// Not compiled as a doctest: the expansion refers to `RuntimeSymbol` and
/// `FunctionPtr`, which `runtime_plugin!` generates in the plugin crate.
///
/// ```ignore
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

    // export_name keeps the symbol string verbatim (e.g. "$IO$println") for JIT and AOT linking
    let expanded = quote! {
        #(#attrs)*
        #[unsafe(export_name = #symbol_name)]
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
/// Not compiled as a doctest: the macro defines the plugin's public items,
/// so it can only be invoked once, at the plugin crate's root.
///
/// ```ignore
/// runtime_plugin! {
///     name: "stdlib",
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
