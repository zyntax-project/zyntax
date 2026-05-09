//! Shared import-resolver chain logic used by both `ZyntaxRuntime` and
//! `TieredRuntime`. Originally lived as private methods on `ZyntaxRuntime`;
//! extracted so the tiered runtime can run the same trait/extern/impl
//! resolution pass when it lowers a TypedProgram.
//!
//! Each function takes only the state it actually needs — `&grammars`,
//! `&plugin_signatures`, `&import_resolvers` — so callers don't need to
//! own a particular runtime type.

use std::collections::HashMap;
use std::sync::Arc;

use crate::grammar::LanguageGrammar;
use crate::runtime::{ImportResolverCallback, RuntimeError, RuntimeResult};

/// Resolve `module_path` against an ordered list of resolver callbacks.
/// Returns `Ok(Some(source))` from the first resolver that knows about
/// the module; `Ok(None)` if nobody recognised it.
pub(crate) fn resolve_import_with(
    import_resolvers: &[ImportResolverCallback],
    module_path: &str,
) -> Result<Option<String>, String> {
    for resolver in import_resolvers {
        match resolver(module_path) {
            Ok(Some(source)) => return Ok(Some(source)),
            Ok(None) => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(None)
}

pub(crate) fn process_imports_for_traits(
    grammars: &std::collections::HashMap<String, std::sync::Arc<crate::grammar::LanguageGrammar>>,
    plugin_signatures: &std::collections::HashMap<String, zyntax_compiler::zrtl::ZrtlSymbolSig>,
    import_resolvers: &[crate::runtime::ImportResolverCallback],
    program: &mut zyntax_typed_ast::TypedProgram,
    type_registry: &mut zyntax_typed_ast::TypeRegistry,
) -> RuntimeResult<()> {
    // Use thread-local storage to track processed imports for circular dependency detection
    thread_local! {
        static PROCESSED_IMPORTS: std::cell::RefCell<std::collections::HashSet<String>> =
            std::cell::RefCell::new(std::collections::HashSet::new());
    }

    use zyntax_typed_ast::typed_ast::TypedDeclaration;

    // Collect imports to process (can't mutate while iterating)
    let mut imports_to_process = Vec::new();
    for decl in &program.declarations {
        if let TypedDeclaration::Import(import) = &decl.node {
            let module_name = import
                .module_path
                .first()
                .and_then(|s| s.resolve_global())
                .unwrap_or_else(|| "unknown".to_string());
            imports_to_process.push(module_name);
        }
    }

    // Process each import
    for module_name in imports_to_process {
        // Skip if already processed (circular import detection)
        let already_processed =
            PROCESSED_IMPORTS.with(|set| set.borrow().contains(&module_name));
        if already_processed {
            log::debug!("Skipping already processed import: {}", module_name);
            continue;
        }

        // Mark as processed BEFORE recursive processing
        PROCESSED_IMPORTS.with(|set| set.borrow_mut().insert(module_name.clone()));

        // Try to resolve the import using our import resolvers
        if let Ok(Some(source)) = resolve_import_with(import_resolvers, &module_name) {
            // Find a grammar to parse the imported module
            // Try each registered grammar until one succeeds
            // Pass plugin signatures to ensure proper extern function declarations
            let mut parsed_program = None;
            for (_lang_name, grammar) in grammars.iter() {
                match grammar.parse_with_signatures(
                    &source,
                    &module_name,
                    &plugin_signatures,
                ) {
                    Ok(imported_program) => {
                        parsed_program = Some(imported_program);
                        break;
                    }
                    Err(_) => continue,
                }
            }

            if let Some(mut imported_program) = parsed_program {
                // IMPORTANT: Recursively process imports from the imported module FIRST
                // This ensures transitive dependencies (e.g., tensor -> prelude) are loaded
                // before we process the imported module's declarations
                process_imports_for_traits(grammars, plugin_signatures, import_resolvers, &mut imported_program, type_registry)?;

                // First, merge the TypeRegistry from the imported module
                // This includes struct definitions, trait definitions, etc.
                type_registry.merge_from(&imported_program.type_registry);

                // Register struct types from Class declarations in the imported module
                // This ensures structs are available before impl block processing
                if let Err(e) =
                    register_struct_declarations(&imported_program, type_registry)
                {
                    log::warn!(
                        "Failed to register struct declarations from '{}': {}",
                        module_name,
                        e
                    );
                }

                // Process extern declarations from the imported module
                // to register opaque types in the type registry
                if let Err(e) =
                    process_extern_declarations_mut(&imported_program, type_registry)
                {
                    log::warn!(
                        "Failed to process extern declarations from '{}': {}",
                        module_name,
                        e
                    );
                }

                // Merge declarations from imported module into main program
                // Filter out the import declarations themselves to avoid re-processing
                for imported_decl in imported_program.declarations.drain(..) {
                    if !matches!(imported_decl.node, TypedDeclaration::Import(_)) {
                        program.declarations.push(imported_decl);
                    }
                }

                log::debug!("Merged declarations from '{}'", module_name);
            } else {
                log::warn!(
                    "Failed to parse imported module '{}' with any registered grammar",
                    module_name
                );
            }
        } else {
            log::warn!("Could not resolve import: {}", module_name);
        }
    }

    Ok(())
}

/// Resolve all Type::Unresolved in the TypedProgram
///
/// This walks the TypedAST and replaces Type::Unresolved(name) with the actual
/// type from TypeRegistry (e.g., Type::Extern for opaque types from imports).
/// This must be called AFTER process_imports_for_traits and process_extern_declarations_mut.
pub(crate) fn resolve_unresolved_types(
    program: &mut zyntax_typed_ast::TypedProgram,
    type_registry: &zyntax_typed_ast::TypeRegistry,
) {
    use zyntax_typed_ast::type_registry::Type;
    use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedNode};
    use zyntax_typed_ast::InternedString;

    log::debug!(
        "Resolving unresolved types in program with {} declarations",
        program.declarations.len()
    );

    // Build a map of function names to return types from declarations
    let mut function_returns: std::collections::HashMap<InternedString, Type> =
        std::collections::HashMap::new();
    for decl in program.declarations.iter() {
        match &decl.node {
            TypedDeclaration::Function(func) => {
                function_returns.insert(func.name, func.return_type.clone());
            }
            TypedDeclaration::Impl(impl_block) => {
                // Add methods from impl blocks with mangled names (TypeName$method)
                // Extract type name from for_type
                let type_name_opt = match &impl_block.for_type {
                    Type::Named { id, .. } => type_registry
                        .get_type_by_id(*id)
                        .and_then(|td| td.name.resolve_global()),
                    Type::Unresolved(name) => name.resolve_global(),
                    Type::Extern { name, .. } => {
                        // Strip $ prefix if present
                        name.resolve_global()
                            .map(|s| s.strip_prefix('$').unwrap_or(&s).to_string())
                    }
                    _ => None,
                };

                if let Some(type_name) = type_name_opt {
                    for method in &impl_block.methods {
                        let method_name = method.name.resolve_global().unwrap_or_default();
                        let mangled_name = format!("{}${}", type_name, method_name);
                        let mangled_interned = InternedString::new_global(&mangled_name);
                        function_returns.insert(mangled_interned, method.return_type.clone());
                        log::debug!(
                            "[RESOLVE_TYPES] Added impl method '{}' with return type {:?}",
                            mangled_name,
                            method.return_type
                        );
                    }
                }
            }
            _ => {}
        }
    }
    log::debug!(
        "Built function_returns map with {} entries",
        function_returns.len()
    );

    // Walk all declarations and resolve types
    for decl in &mut program.declarations {
        resolve_in_declaration(&mut decl.node, type_registry, &function_returns);
        resolve_in_type(&mut decl.ty, type_registry);
    }
}

pub(crate) fn resolve_in_declaration(
    decl: &mut zyntax_typed_ast::typed_ast::TypedDeclaration,
    type_registry: &zyntax_typed_ast::TypeRegistry,
    function_returns: &std::collections::HashMap<
        zyntax_typed_ast::InternedString,
        zyntax_typed_ast::type_registry::Type,
    >,
) {
    use zyntax_typed_ast::typed_ast::TypedDeclaration;

    match decl {
        TypedDeclaration::Function(func) => {
            // Resolve parameter types
            for param in &mut func.params {
                resolve_in_type(&mut param.ty, type_registry);
            }
            // Resolve return type
            resolve_in_type(&mut func.return_type, type_registry);
            // Resolve body expression types
            if let Some(body) = &mut func.body {
                resolve_in_block(body, type_registry, function_returns);
            }
        }
        TypedDeclaration::Impl(impl_block) => {
            // Resolve trait type arguments
            for trait_ty in &mut impl_block.trait_type_args {
                resolve_in_type(trait_ty, type_registry);
            }
            // Resolve target type
            resolve_in_type(&mut impl_block.for_type, type_registry);
            // Resolve method types
            for method in &mut impl_block.methods {
                for param in &mut method.params {
                    resolve_in_type(&mut param.ty, type_registry);
                }
                resolve_in_type(&mut method.return_type, type_registry);
                if let Some(body) = &mut method.body {
                    resolve_in_block(body, type_registry, function_returns);
                }
            }
        }
        _ => {
            // Other declarations don't have types to resolve
        }
    }
}

pub(crate) fn resolve_in_block(
    block: &mut zyntax_typed_ast::typed_ast::TypedBlock,
    type_registry: &zyntax_typed_ast::TypeRegistry,
    function_returns: &std::collections::HashMap<
        zyntax_typed_ast::InternedString,
        zyntax_typed_ast::type_registry::Type,
    >,
) {
    use zyntax_typed_ast::typed_ast::TypedStatement;

    for stmt in &mut block.statements {
        resolve_in_stmt(stmt, type_registry, function_returns);
    }
}

pub(crate) fn resolve_in_stmt(
    stmt: &mut zyntax_typed_ast::typed_ast::TypedNode<
        zyntax_typed_ast::typed_ast::TypedStatement,
    >,
    type_registry: &zyntax_typed_ast::TypeRegistry,
    function_returns: &std::collections::HashMap<
        zyntax_typed_ast::InternedString,
        zyntax_typed_ast::type_registry::Type,
    >,
) {
    use zyntax_typed_ast::typed_ast::TypedStatement;

    // Resolve the statement's type annotation
    resolve_in_type(&mut stmt.ty, type_registry);

    // Resolve types in the statement node
    match &mut stmt.node {
        TypedStatement::Expression(expr) => {
            resolve_in_expr(expr, type_registry, function_returns);
        }
        TypedStatement::Let(let_stmt) => {
            resolve_in_type(&mut let_stmt.ty, type_registry);
            if let Some(init) = &mut let_stmt.initializer {
                resolve_in_expr(init, type_registry, function_returns);
                // If let_stmt.ty is Any/Unknown, use the initializer's type
                if matches!(
                    let_stmt.ty,
                    zyntax_typed_ast::type_registry::Type::Any
                        | zyntax_typed_ast::type_registry::Type::Unknown
                ) {
                    if !matches!(
                        init.ty,
                        zyntax_typed_ast::type_registry::Type::Any
                            | zyntax_typed_ast::type_registry::Type::Unknown
                    ) {
                        let_stmt.ty = init.ty.clone();
                        log::debug!(
                            "[RESOLVE_TYPES] Propagated initializer type to let binding: {:?}",
                            let_stmt.ty
                        );
                    }
                }
            }
        }
        TypedStatement::Match(match_stmt) => {
            resolve_in_expr(&mut match_stmt.scrutinee, type_registry, function_returns);
            for arm in &mut match_stmt.arms {
                // Convert enum variant patterns to integer literal patterns
                resolve_pattern(&mut arm.pattern, type_registry);
                if let Some(guard) = &mut arm.guard {
                    resolve_in_expr(guard, type_registry, function_returns);
                }
                resolve_in_expr(&mut arm.body, type_registry, function_returns);
            }
        }
        _ => {
            // Other statement types
        }
    }
}

/// Walk a pattern and resolve enum variant patterns to literal integer patterns.
/// Both `case Color::Red` (TypedPattern::Enum) and `case Red()` (TypedPattern::Constructor)
/// are converted to literal integer patterns matching the discriminant.
pub(crate) fn resolve_pattern(
    pattern: &mut zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedPattern>,
    type_registry: &zyntax_typed_ast::TypeRegistry,
) {
    use zyntax_typed_ast::type_registry::Type;
    use zyntax_typed_ast::typed_ast::{TypedLiteralPattern, TypedPattern};

    // Try TypedPattern::Enum (full path: Color::Red)
    if let TypedPattern::Enum {
        name,
        variant,
        fields,
    } = &pattern.node
    {
        if fields.is_empty() {
            let enum_name = *name;
            let variant_name = *variant;
            if let Some(disc) =
                lookup_variant_discriminant(type_registry, enum_name, variant_name)
            {
                pattern.node = TypedPattern::Literal(TypedLiteralPattern::Integer(disc));
                return;
            }
        }
        return;
    }

    // Try TypedPattern::Constructor (bare name: Red() or Red)
    // Search ALL registered enums for one with this variant name
    if let TypedPattern::Constructor {
        constructor,
        pattern: inner,
    } = &pattern.node
    {
        // Only handle unit constructors (wildcard inner pattern)
        if !matches!(inner.node, TypedPattern::Wildcard) {
            return;
        }
        let variant_name = if let Type::Unresolved(name) = constructor {
            *name
        } else {
            return;
        };

        // Search all enum types for a variant with this name
        for type_def in type_registry.get_all_types() {
            if let zyntax_typed_ast::type_registry::TypeKind::Enum { variants } = &type_def.kind
            {
                for v in variants {
                    if v.name == variant_name {
                        let disc = v.discriminant.unwrap_or(0);
                        pattern.node =
                            TypedPattern::Literal(TypedLiteralPattern::Integer(disc as i128));
                        log::debug!(
                            "[RESOLVE_TYPES] Resolved Constructor pattern {} to discriminant {}",
                            variant_name.resolve_global().unwrap_or_default(),
                            disc
                        );
                        return;
                    }
                }
            }
        }
    }
}

/// Look up an enum variant's discriminant value by enum name and variant name.
pub(crate) fn lookup_variant_discriminant(
    type_registry: &zyntax_typed_ast::TypeRegistry,
    enum_name: zyntax_typed_ast::InternedString,
    variant_name: zyntax_typed_ast::InternedString,
) -> Option<i128> {
    let type_def = type_registry.get_type_by_name(enum_name)?;
    if let zyntax_typed_ast::type_registry::TypeKind::Enum { variants } = &type_def.kind {
        for v in variants {
            if v.name == variant_name {
                return Some(v.discriminant.unwrap_or(0) as i128);
            }
        }
    }
    None
}

pub(crate) fn resolve_in_expr(
    expr: &mut zyntax_typed_ast::typed_ast::TypedNode<
        zyntax_typed_ast::typed_ast::TypedExpression,
    >,
    type_registry: &zyntax_typed_ast::TypeRegistry,
    function_returns: &std::collections::HashMap<
        zyntax_typed_ast::InternedString,
        zyntax_typed_ast::type_registry::Type,
    >,
) {
    use zyntax_typed_ast::type_registry::Type;
    use zyntax_typed_ast::typed_ast::TypedExpression;

    // Resolve the expression's type annotation
    resolve_in_type(&mut expr.ty, type_registry);

    // Recursively resolve types in sub-expressions
    match &mut expr.node {
        TypedExpression::Binary(bin) => {
            resolve_in_expr(&mut bin.left, type_registry, function_returns);
            resolve_in_expr(&mut bin.right, type_registry, function_returns);
        }
        TypedExpression::Unary(un) => {
            resolve_in_expr(&mut un.operand, type_registry, function_returns);
        }
        TypedExpression::Call(call) => {
            resolve_in_expr(&mut call.callee, type_registry, function_returns);
            for arg in &mut call.positional_args {
                resolve_in_expr(arg, type_registry, function_returns);
            }
            for named_arg in &mut call.named_args {
                resolve_in_expr(&mut named_arg.value, type_registry, function_returns);
            }

            // If the callee is a Path expression (e.g., Tensor::arange), convert to Variable with mangled name
            if let TypedExpression::Path(path) = &call.callee.node {
                // Convert path segments to mangled function name
                let segments: Vec<String> = path
                    .segments
                    .iter()
                    .filter_map(|s| s.resolve_global())
                    .collect();
                let mangled_name = segments.join("$"); // "Tensor$arange"
                let mangled_interned =
                    zyntax_typed_ast::InternedString::new_global(&mangled_name);

                log::debug!(
                    "[RESOLVE_TYPES] Converting Path {:?} to Variable '{}'",
                    segments,
                    mangled_name
                );

                // Look up return type from function_returns using mangled name
                if let Some(return_type) = function_returns.get(&mangled_interned) {
                    let mut resolved_return_type = return_type.clone();
                    resolve_in_type(&mut resolved_return_type, type_registry);
                    if !matches!(resolved_return_type, Type::Any | Type::Unknown) {
                        expr.ty = resolved_return_type;
                        log::debug!(
                            "[RESOLVE_TYPES] Path call '{}' resolved to return type {:?}",
                            mangled_name,
                            expr.ty
                        );
                    }
                }

                // Convert Path to Variable so SSA can handle it
                call.callee.node = TypedExpression::Variable(mangled_interned);
            }
            // If the callee is a variable (function name), look up its return type
            else if let TypedExpression::Variable(func_name) = &call.callee.node {
                if let Some(return_type) = function_returns.get(func_name) {
                    let mut resolved_return_type = return_type.clone();
                    resolve_in_type(&mut resolved_return_type, type_registry);
                    if !matches!(resolved_return_type, Type::Any | Type::Unknown) {
                        expr.ty = resolved_return_type;
                    }
                }
            }
        }
        TypedExpression::Block(block) => {
            resolve_in_block(block, type_registry, function_returns);
        }
        TypedExpression::MethodCall(method_call) => {
            // Resolve receiver type
            resolve_in_expr(&mut method_call.receiver, type_registry, function_returns);

            // Resolve positional arguments
            for arg in &mut method_call.positional_args {
                resolve_in_expr(arg, type_registry, function_returns);
            }

            // Resolve named arguments
            for named_arg in &mut method_call.named_args {
                resolve_in_expr(&mut named_arg.value, type_registry, function_returns);
            }

            // Resolve type arguments
            for type_arg in &mut method_call.type_args {
                resolve_in_type(type_arg, type_registry);
            }

            // Try to resolve the return type of the method call
            // For static calls (receiver is a type name), look up the type and method
            if let TypedExpression::Variable(type_name) = &method_call.receiver.node {
                // Check if this is a type name (static method call like Tensor::arange)
                if let Some(type_def) = type_registry.get_type_by_name(*type_name) {
                    // Look for the method in the type's methods
                    let method_name = method_call.method;
                    for method in &type_def.methods {
                        if method.name == method_name {
                            // Found the method! Update the expression's type to the return type
                            let mut return_type = method.return_type.clone();
                            // If return type needs resolution, do it
                            resolve_in_type(&mut return_type, type_registry);
                            expr.ty = return_type;
                            log::debug!("[RESOLVE_TYPES] Resolved static method {}::{} return type to {:?}",
                                type_name.resolve_global().unwrap_or_default(),
                                method_name.resolve_global().unwrap_or_default(),
                                expr.ty);
                            break;
                        }
                    }
                }
            }
            // For instance method calls, we can try to resolve based on receiver type
            else if !matches!(method_call.receiver.ty, Type::Unknown) {
                // Get the receiver's type and look up the method
                let receiver_type = &method_call.receiver.ty;
                if let Type::Named { id, .. } = receiver_type {
                    if let Some(type_def) = type_registry.get_type_by_id(*id) {
                        let method_name = method_call.method;
                        for method in &type_def.methods {
                            if method.name == method_name {
                                let mut return_type = method.return_type.clone();
                                resolve_in_type(&mut return_type, type_registry);
                                expr.ty = return_type;
                                break;
                            }
                        }
                    }
                } else if let Type::Extern { name, .. } = receiver_type {
                    // For extern types, check if there's a type definition registered
                    if let Some(type_def) = type_registry.get_type_by_name(*name) {
                        let method_name = method_call.method;
                        for method in &type_def.methods {
                            if method.name == method_name {
                                let mut return_type = method.return_type.clone();
                                resolve_in_type(&mut return_type, type_registry);
                                expr.ty = return_type;
                                break;
                            }
                        }
                    }
                }
            }
        }
        TypedExpression::If(if_expr) => {
            resolve_in_expr(&mut if_expr.condition, type_registry, function_returns);
            resolve_in_expr(&mut if_expr.then_branch, type_registry, function_returns);
            resolve_in_expr(&mut if_expr.else_branch, type_registry, function_returns);
        }
        TypedExpression::Index(index) => {
            resolve_in_expr(&mut index.object, type_registry, function_returns);
            resolve_in_expr(&mut index.index, type_registry, function_returns);
        }
        TypedExpression::Field(field) => {
            resolve_in_expr(&mut field.object, type_registry, function_returns);
        }
        TypedExpression::Lambda(lambda) => {
            // Resolve parameter types if present
            for param in &mut lambda.params {
                if let Some(ty) = &mut param.ty {
                    resolve_in_type(ty, type_registry);
                }
            }
            // Resolve body
            match &mut lambda.body {
                zyntax_typed_ast::typed_ast::TypedLambdaBody::Expression(expr) => {
                    resolve_in_expr(expr, type_registry, function_returns);
                }
                zyntax_typed_ast::typed_ast::TypedLambdaBody::Block(block) => {
                    resolve_in_block(block, type_registry, function_returns);
                }
            }
        }
        TypedExpression::Match(match_expr) => {
            resolve_in_expr(&mut match_expr.scrutinee, type_registry, function_returns);
            for arm in &mut match_expr.arms {
                if let Some(guard) = &mut arm.guard {
                    resolve_in_expr(guard, type_registry, function_returns);
                }
                resolve_in_expr(&mut arm.body, type_registry, function_returns);
            }
        }
        TypedExpression::Array(elements) => {
            for elem in elements {
                resolve_in_expr(elem, type_registry, function_returns);
            }
        }
        TypedExpression::Tuple(elements) => {
            for elem in elements {
                resolve_in_expr(elem, type_registry, function_returns);
            }
        }
        TypedExpression::Struct(struct_lit) => {
            for field in &mut struct_lit.fields {
                resolve_in_expr(&mut field.value, type_registry, function_returns);
            }
        }
        TypedExpression::Path(path) => {
            // Resolve enum unit variant paths like `Color::Green` to integer literals.
            // Path { segments: [enum_name, variant_name] } → IntLiteral(discriminant)
            if path.segments.len() == 2 {
                let enum_name = path.segments[0];
                let variant_name = path.segments[1];
                if let Some(type_def) = type_registry.get_type_by_name(enum_name) {
                    if let zyntax_typed_ast::type_registry::TypeKind::Enum { variants } =
                        &type_def.kind
                    {
                        for v in variants {
                            if v.name == variant_name {
                                let disc = v.discriminant.unwrap_or(0);
                                expr.node = TypedExpression::Literal(
                                    zyntax_typed_ast::typed_ast::TypedLiteral::Integer(
                                        disc as i128,
                                    ),
                                );
                                expr.ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::I64);
                                log::debug!(
                                    "[RESOLVE_TYPES] Resolved enum variant {}::{} to discriminant {}",
                                    enum_name.resolve_global().unwrap_or_default(),
                                    variant_name.resolve_global().unwrap_or_default(),
                                    disc
                                );
                                break;
                            }
                        }
                    }
                }
            }
        }
        _ => {
            // Other expression types (Variable, Literal, etc.) don't need nested resolution
        }
    }
}

pub(crate) fn resolve_in_type(
    ty: &mut zyntax_typed_ast::type_registry::Type,
    type_registry: &zyntax_typed_ast::TypeRegistry,
) {
    use zyntax_typed_ast::type_registry::Type;

    match ty {
        Type::Unresolved(name) => {
            // First try type aliases
            if let Some(resolved) = type_registry.resolve_alias(*name) {
                log::debug!(
                    "Resolved type '{}' from alias to {:?}",
                    name.resolve_global().unwrap_or_default(),
                    resolved
                );
                *ty = resolved.clone();
            }
            // Then try looking up by name in the type definitions (for structs, enums, etc.)
            else if let Some(type_def) = type_registry.get_type_by_name(*name) {
                let resolved = Type::Named {
                    id: type_def.id,
                    type_args: vec![],
                    const_args: vec![],
                    variance: vec![],
                    nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
                };
                log::debug!(
                    "Resolved type '{}' from Unresolved to Named({:?})",
                    name.resolve_global().unwrap_or_default(),
                    type_def.id
                );
                *ty = resolved;
            } else {
                log::warn!(
                    "Could not resolve type '{}'",
                    name.resolve_global().unwrap_or_default()
                );
            }
        }
        Type::Named { id, type_args, .. } => {
            for type_arg in type_args {
                resolve_in_type(type_arg, type_registry);
            }

            // Canonicalize Named IDs by name in case this ID came from a placeholder
            // registry entry created before imports/externs were merged.
            if let Some(type_def) = type_registry.get_type_by_id(*id) {
                if let Some(canonical) = type_registry.get_type_by_name(type_def.name) {
                    if canonical.id != *id {
                        *id = canonical.id;
                    }
                }
            }
        }
        // Recursively resolve nested types
        Type::Reference { ty: inner, .. } => {
            resolve_in_type(inner, type_registry);
        }
        Type::Array { element_type, .. } => {
            resolve_in_type(element_type, type_registry);
        }
        Type::Function {
            params,
            return_type,
            ..
        } => {
            for param in params {
                resolve_in_type(&mut param.ty, type_registry);
            }
            resolve_in_type(return_type, type_registry);
        }
        Type::Tuple(types) => {
            for t in types {
                resolve_in_type(t, type_registry);
            }
        }
        _ => {
            // Other types don't need resolution
        }
    }
}

/// Process extern declarations to register opaque types in the TypeRegistry
///
/// This scans the TypedProgram for Extern::Struct declarations (created by @opaque)
/// and registers them in the TypeRegistry so they can be resolved during type checking.
pub(crate) fn process_extern_declarations_mut(
    program: &zyntax_typed_ast::TypedProgram,
    type_registry: &mut zyntax_typed_ast::TypeRegistry,
) -> RuntimeResult<()> {
    use zyntax_typed_ast::type_registry::{Type, TypeDefinition, TypeKind, TypeMetadata};
    use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedExtern};
    use zyntax_typed_ast::TypeId;

    // Collect all extern struct declarations
    for decl in &program.declarations {
        if let TypedDeclaration::Extern(extern_decl) = &decl.node {
            if let TypedExtern::Struct(extern_struct) = extern_decl {
                // Create TypeDefinition for the extern/opaque type
                let type_id = TypeId::next();
                let type_def = TypeDefinition {
                    id: type_id,
                    name: extern_struct.name,
                    kind: TypeKind::Atomic, // Extern types are atomic/opaque
                    type_params: vec![],
                    constraints: vec![],
                    fields: vec![],
                    methods: vec![],
                    constructors: vec![],
                    metadata: TypeMetadata::default(),
                    span: decl.span,
                };

                // Register the type definition
                type_registry.register_type(type_def);

                // ALSO register an alias mapping the name to Type::Extern
                // This allows the resolution pass to find the runtime name
                let extern_type = Type::Extern {
                    name: extern_struct.runtime_prefix,
                    layout: None,
                };
                type_registry.register_alias(extern_struct.name, extern_type);
            }
        }
    }

    Ok(())
}

/// Register struct types from TypedDeclaration::Class declarations
///
/// This processes struct definitions parsed by the grammar and registers them
/// in the TypeRegistry so they can be found during impl block processing.
pub(crate) fn register_struct_declarations(
    program: &zyntax_typed_ast::TypedProgram,
    type_registry: &mut zyntax_typed_ast::TypeRegistry,
) -> RuntimeResult<()> {
    use zyntax_typed_ast::type_registry::{
        FieldDef, TypeDefinition, TypeKind, TypeMetadata, TypeParam, Variance, Visibility,
    };
    use zyntax_typed_ast::typed_ast::TypedDeclaration;
    use zyntax_typed_ast::TypeId;

    // Process all Class declarations (structs are represented as Class)
    for decl in &program.declarations {
        if let TypedDeclaration::Class(class_decl) = &decl.node {
            // Check if type is already registered
            // If it exists but has 0 fields and the new one has fields, update it
            // (generic types like List<T> may be pre-registered as placeholders without fields)
            let existing_id =
                if let Some(existing) = type_registry.get_type_by_name(class_decl.name) {
                    if existing.fields.is_empty() && !class_decl.fields.is_empty() {
                        Some(existing.id)
                    } else {
                        continue;
                    }
                } else {
                    None
                };

            // Convert TypedTypeParam to TypeParam
            let type_params: Vec<TypeParam> = class_decl
                .type_params
                .iter()
                .map(|tp| {
                    TypeParam {
                        name: tp.name,
                        bounds: vec![], // Simplified - could convert bounds if needed
                        variance: Variance::Invariant,
                        default: tp.default.clone(),
                        span: tp.span,
                    }
                })
                .collect();

            // Convert TypedField to FieldDef
            let fields: Vec<FieldDef> = class_decl
                .fields
                .iter()
                .map(|f| {
                    FieldDef {
                        name: f.name,
                        ty: f.ty.clone(),
                        visibility: Visibility::Public, // Simplified
                        mutability: f.mutability.clone(),
                        is_static: false,
                        span: f.span,
                        getter: None,
                        setter: None,
                        is_synthetic: false,
                    }
                })
                .collect();

            // Create TypeDefinition for the struct
            // Use existing ID if updating a placeholder, otherwise generate new
            let type_id = existing_id.unwrap_or_else(TypeId::next);
            let type_def = TypeDefinition {
                id: type_id,
                name: class_decl.name,
                kind: TypeKind::Struct {
                    fields: fields.clone(),
                    is_tuple: false,
                },
                type_params,
                constraints: vec![],
                fields,
                methods: vec![],
                constructors: vec![],
                metadata: TypeMetadata::default(),
                span: decl.span,
            };

            // Register the type
            type_registry.register_type(type_def);
        }
    }

    Ok(())
}
