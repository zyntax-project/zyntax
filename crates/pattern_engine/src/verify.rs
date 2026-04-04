use zyntax_typed_ast::type_checker::{TypeCheckOptions, TypeChecker};
use zyntax_typed_ast::typed_ast::TypedProgram;
use zyntax_typed_ast::TypeRegistry;

/// Run the type checker on the program to verify soundness after a rewrite iteration.
/// Returns any type errors found.
pub fn verify_type_soundness(program: &mut TypedProgram, registry: &TypeRegistry) -> Vec<String> {
    let registry_box = Box::new(registry.clone());
    let mut checker = TypeChecker::with_options(
        registry_box,
        TypeCheckOptions {
            strict_nulls: false,
            strict_functions: false,
            no_implicit_any: false,
            check_unreachable: false,
        },
    );

    checker.check_program(program);

    let diagnostics = checker.diagnostics();
    if diagnostics.has_errors() {
        diagnostics
            .diagnostics()
            .iter()
            .map(|d| d.message.clone())
            .collect()
    } else {
        Vec::new()
    }
}
