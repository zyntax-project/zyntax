//! The addresses `HirConstant::SymbolAddress` names. Code holds the
//! name, not the address, so the same code runs in any process: each
//! tier binds the name where it binds calls to runtime symbols, asking
//! here after its own table.

/// The address of the runtime data symbol `name`: one the compiler's
/// runtime defines, else one an embedder entered late.
pub fn resolve(name: &str) -> Option<usize> {
    match name {
        crate::interned::SHARED_BOXES_SYMBOL => Some(crate::interned::shared_boxes()),
        _ => crate::late_symbols::lookup(name).map(|p| p as usize),
    }
}

/// The address a `SymbolAddress { name, offset }` constant stands for.
pub fn resolve_with_offset(name: &str, offset: i64) -> Option<usize> {
    resolve(name).map(|base| base.wrapping_add(offset as usize))
}
