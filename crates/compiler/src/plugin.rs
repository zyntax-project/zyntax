//! Plugin system for frontend runtime registration
//!
//! This module provides a trait-based plugin architecture that allows frontend
//! implementations to register their runtime symbols without modifying core
//! compiler infrastructure.
//!
//! # Example
//!
//! ```rust
//! use zyntax_compiler::plugin::RuntimePlugin;
//! # extern "C" fn my_array_create() {}
//! # extern "C" fn my_array_push() {}
//!
//! pub struct HaxeRuntimePlugin;
//!
//! impl RuntimePlugin for HaxeRuntimePlugin {
//!     fn name(&self) -> &str {
//!         "haxe"
//!     }
//!
//!     fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)> {
//!         vec![
//!             ("$Array$create", my_array_create as *const u8),
//!             ("$Array$push", my_array_push as *const u8),
//!         ]
//!     }
//! }
//! ```

/// Trait for frontend runtime plugins
///
/// Frontend implementations create a type implementing this trait to provide
/// their runtime symbols to the JIT compiler.
pub trait RuntimePlugin: Send + Sync {
    /// Returns the name of this frontend (e.g., "haxe", "python", "javascript")
    fn name(&self) -> &str;

    /// Returns the runtime symbols this frontend provides
    ///
    /// Each symbol is a tuple of (symbol_name, function_pointer).
    /// Symbol names should follow the convention: `$Type$method`
    fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)>;

    /// Optional: Called when the plugin is loaded
    ///
    /// Use this for initialization tasks like setting up global state.
    fn on_load(&self) -> Result<(), String> {
        Ok(())
    }

    /// Optional: Called when the plugin is unloaded
    ///
    /// Use this for cleanup tasks.
    fn on_unload(&self) -> Result<(), String> {
        Ok(())
    }
}

/// Registry for runtime plugins
///
/// This is used by the CLI to manage multiple frontend plugins.
pub struct PluginRegistry {
    plugins: Vec<Box<dyn RuntimePlugin>>,
}

impl PluginRegistry {
    /// Create a new empty plugin registry
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Register a plugin
    pub fn register(&mut self, plugin: Box<dyn RuntimePlugin>) -> Result<(), String> {
        // Check for duplicate plugin names
        let name = plugin.name();
        if self.plugins.iter().any(|p| p.name() == name) {
            return Err(format!("Plugin '{}' is already registered", name));
        }

        // Call the plugin's on_load hook
        plugin.on_load()?;

        self.plugins.push(plugin);
        Ok(())
    }

    /// Get all runtime symbols from all registered plugins
    pub fn collect_symbols(&self) -> Vec<(&'static str, *const u8)> {
        let mut symbols = Vec::new();
        for plugin in &self.plugins {
            symbols.extend(plugin.runtime_symbols());
        }
        symbols
    }

    /// Get a plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<&dyn RuntimePlugin> {
        self.plugins
            .iter()
            .find(|p| p.name() == name)
            .map(|p| p.as_ref())
    }

    /// List all registered plugin names
    pub fn list_plugins(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }

    /// Unload all plugins
    pub fn unload_all(&mut self) -> Result<(), String> {
        for plugin in &self.plugins {
            plugin.on_unload()?;
        }
        self.plugins.clear();
        Ok(())
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PluginRegistry {
    fn drop(&mut self) {
        // Best effort cleanup - ignore errors during drop
        let _ = self.unload_all();
    }
}
