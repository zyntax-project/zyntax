//! Symbols an embedder supplies while a program is being lowered, after
//! the code generators were set up: a plugin loaded because the program
//! imported it. Every tier resolves a name its own table lacks here
//! before asking the process, so code compiled after an entry calls it
//! directly, with no rebuild.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

fn table() -> &'static RwLock<HashMap<String, usize>> {
    static TABLE: OnceLock<RwLock<HashMap<String, usize>>> = OnceLock::new();
    TABLE.get_or_init(Default::default)
}

/// Make `name` resolve to `address` for code compiled from now on. A
/// name already entered keeps its first address.
pub fn register(name: &str, address: *const u8) {
    table()
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .entry(name.to_string())
        .or_insert(address as usize);
}

/// The address entered for `name`, if one was.
pub fn lookup(name: &str) -> Option<*const u8> {
    table()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(name)
        .map(|a| *a as *const u8)
}
