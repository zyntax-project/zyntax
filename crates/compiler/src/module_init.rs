//! Top-level blocks use the ordinary function pipeline without a source entry name.

use crate::{
    CompilerError, CompilerResult,
    hir::{HirId, HirModule},
};
use rustc_hash::{FxHashMap, FxHashSet};
use zyntax_typed_ast::typed_ast::{TypedAnnotation, TypedDeclaration, TypedFunction};
use zyntax_typed_ast::{InternedString, PrimitiveType, Type, TypedProgram};

static INITIALIZER: std::sync::LazyLock<InternedString> =
    std::sync::LazyLock::new(|| InternedString::new_global("zyntax_module_initializer"));

/// Validate the host-call ABI and retain names before installation consumes the module.
pub fn entries(module: &HirModule) -> CompilerResult<Vec<(HirId, String)>> {
    module
        .initializers
        .iter()
        .map(|id| {
            let function = module
                .functions
                .get(id)
                .ok_or_else(|| CompilerError::Analysis("module initializer is missing".into()))?;
            let signature = &function.signature;
            if function.is_external
                || !signature.params.is_empty()
                || !signature.returns.is_empty()
                || !signature.type_params.is_empty()
                || !signature.const_params.is_empty()
                || signature.is_async
                || signature.is_fiber
                || signature.is_variadic
            {
                return Err(CompilerError::Analysis(
                    "module initializer must be a defined () -> void function".into(),
                ));
            }
            let name = function.name.resolve_global().ok_or_else(|| {
                CompilerError::Analysis("module initializer name is missing".into())
            })?;
            Ok((*id, name))
        })
        .collect()
}

pub fn is_initializer(function: &TypedFunction) -> bool {
    function.annotations.iter().any(|a| a.name == *INITIALIZER)
}

/// A declaration for an initializer whose body arrives as prelowered HIR.
pub fn reference(name: InternedString) -> TypedFunction {
    TypedFunction {
        name,
        return_type: Type::Primitive(PrimitiveType::Unit),
        annotations: vec![TypedAnnotation {
            name: *INITIALIZER,
            args: vec![],
            span: Default::default(),
        }],
        ..Default::default()
    }
}

/// Move initializer blocks into private internal functions before type checking
/// and pattern rewriting. Repeated preparation leaves existing functions alone.
pub fn prepare(program: &mut TypedProgram) {
    if !program
        .declarations
        .iter()
        .any(|d| matches!(d.node, TypedDeclaration::Initializer(_)))
    {
        return;
    }
    let mut occupied: FxHashSet<_> = program
        .declarations
        .iter()
        .filter_map(|d| match &d.node {
            TypedDeclaration::Function(f) => Some(f.name),
            _ => None,
        })
        .collect();
    let language = program
        .language
        .and_then(|n| n.resolve_global())
        .unwrap_or_default();
    let own_module = program
        .type_registry
        .current_module()
        .or_else(|| {
            program
                .source_files
                .first()
                .map(|s| InternedString::new_global(&s.name))
        })
        .unwrap_or_else(|| InternedString::new_global("module"));
    let mut ordinals = FxHashMap::<InternedString, usize>::default();
    for declaration in &mut program.declarations {
        let TypedDeclaration::Initializer(initializer) = &declaration.node else {
            continue;
        };
        let module = initializer.module.unwrap_or(own_module);
        let ordinal = ordinals.entry(module).or_default();
        let module_name = module.resolve_global().unwrap_or_default();
        let name = loop {
            let candidate = InternedString::new_global(&format!(
                "__zyntax_init${language}${module_name}${ordinal}"
            ));
            *ordinal += 1;
            if occupied.insert(candidate) {
                break candidate;
            }
        };
        let TypedDeclaration::Initializer(mut initializer) = std::mem::replace(
            &mut declaration.node,
            TypedDeclaration::Function(TypedFunction::default()),
        ) else {
            unreachable!()
        };
        initializer.annotations.push(TypedAnnotation {
            name: *INITIALIZER,
            args: vec![],
            span: declaration.span,
        });
        declaration.node = TypedDeclaration::Function(TypedFunction {
            name,
            body: Some(initializer.body),
            annotations: initializer.annotations,
            return_type: Type::Primitive(PrimitiveType::Unit),
            module: initializer.module,
            ..Default::default()
        });
    }
}
