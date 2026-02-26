//! I/O functions for standard input/output
//!
//! Provides basic I/O functionality including printf for formatted output.

use crate::hir::CallingConvention;
use crate::hir_builder::HirBuilder;

/// Build I/O functions (printf, println, etc.)
pub fn build_io_functions(builder: &mut HirBuilder) {
    build_printf(builder);
    build_println(builder);
}

/// Build printf function for formatted output
///
/// Signature: extern "C" fn printf(format: *u8) -> i32
///
/// Note: This is a simple version that matches C's printf.
/// We can use it with string literals.
fn build_printf(builder: &mut HirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let i32_ty = builder.i32_type();

    // extern "C" fn printf(format: *u8) -> i32
    let _printf = builder
        .begin_extern_function("printf", CallingConvention::C)
        .param("format", ptr_u8_ty)
        .returns(i32_ty)
        .build();
}

/// Build println function (print with newline)
///
/// Signature: extern "C" fn puts(str: *u8) -> i32
fn build_println(builder: &mut HirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let i32_ty = builder.i32_type();

    // extern "C" fn puts(str: *u8) -> i32
    // puts automatically adds a newline
    let _puts = builder
        .begin_extern_function("puts", CallingConvention::C)
        .param("str", ptr_u8_ty)
        .returns(i32_ty)
        .build();
}

/// Build print_i32 function for printing integers
///
/// This will be a wrapper that formats the number and calls printf
pub fn build_print_i32(builder: &mut HirBuilder) {
    let i32_ty = builder.i32_type();
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let void_ty = builder.void_type();

    // First, create a helper: extern "C" fn snprintf(buf: *u8, size: u64, format: *u8, ...) -> i32
    // For now, let's just declare print_i32 as external and implement it in the runtime
    let _print_i32 = builder
        .begin_extern_function("print_i32", CallingConvention::C)
        .param("value", i32_ty)
        .returns(void_ty)
        .build();
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::AstArena;

    #[test]
    fn test_build_io_functions() {
        let mut arena = AstArena::new();
        let mut builder = HirBuilder::new("test", &mut arena);

        build_io_functions(&mut builder);

        let module = builder.finish();

        // Should have external I/O functions
        assert!(module.functions.len() >= 2);
    }
}
