//! Build-time artifacts for imports that must not be parsed on a cold path.

use serde::{Deserialize, Serialize};
use zyntax_typed_ast::{InternedString, TypedProgram};

const MAGIC: &[u8; 4] = b"ZAST";
const SCHEMA_VERSION: u32 = 7;
const VERSION_END: usize = MAGIC.len() + std::mem::size_of::<u32>();
const HEADER_LEN: usize = VERSION_END + zyntax_compiler::BUILD_ID_BYTES.len();

/// A parsed import packaged by the application at build time.
///
/// The payload stores `TypedProgram` and its `TypeRegistry` separately because
/// ordinary TypedProgram serialization intentionally omits the registry.
#[derive(Debug, Clone)]
pub struct CompiledImport {
    language: String,
    module_name: String,
    program: TypedProgram,
    /// The module already lowered, when the artifact that carried it
    /// held HIR this compiler can use. The program's function bodies
    /// are then absent: the declarations type a caller, the HIR is what
    /// runs.
    hir: Option<std::sync::Arc<zyntax_compiler::bytecode::LazyModule>>,
}

#[derive(Serialize, Deserialize)]
struct CompiledImportPayload {
    language: String,
    module_name: String,
    program: TypedProgram,
    type_registry: zyntax_typed_ast::TypeRegistry,
}

#[derive(Serialize)]
struct CompiledImportPayloadRef<'a> {
    language: &'a str,
    module_name: &'a str,
    program: &'a TypedProgram,
    type_registry: &'a zyntax_typed_ast::TypeRegistry,
}

#[derive(Debug, thiserror::Error)]
pub enum CompiledArtifactError {
    #[error("compiled import has an invalid header")]
    InvalidHeader,

    #[error("unsupported compiled import schema {found}; expected {expected}")]
    UnsupportedSchema { found: u32, expected: u32 },

    #[error(
        "compiled import was produced by compiler {actual}; expected {expected}; rebuild the artifact from its original source"
    )]
    IncompatibleCompiler {
        expected: &'static str,
        actual: String,
    },

    #[error("failed to encode compiled import: {0}")]
    Encode(String),

    #[error("failed to decode compiled import: {0}")]
    Decode(String),
}

impl CompiledImport {
    pub fn new(
        language: impl Into<String>,
        module_name: impl Into<String>,
        mut program: TypedProgram,
    ) -> Self {
        let module_name = module_name.into();
        let language = language.into();
        program
            .type_registry
            .set_current_module(Some(InternedString::new_global(&module_name)));
        // The module carries the language it was written in, so an
        // unqualified import inside it resolves against that language
        // rather than against whoever registered first.
        program.language = Some(InternedString::new_global(&language));
        Self {
            language,
            module_name,
            program,
            hir: None,
        }
    }

    /// Attach the module's lowered form.
    pub fn with_hir(mut self, hir: std::sync::Arc<zyntax_compiler::bytecode::LazyModule>) -> Self {
        self.hir = Some(hir);
        self
    }

    /// The module's lowered form, when it arrived with one.
    pub fn hir(&self) -> Option<&std::sync::Arc<zyntax_compiler::bytecode::LazyModule>> {
        self.hir.as_ref()
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn module_name(&self) -> &str {
        &self.module_name
    }

    pub fn program(&self) -> &TypedProgram {
        &self.program
    }

    /// Move out the checked program without cloning its declarations.
    pub fn into_program(self) -> TypedProgram {
        self.program
    }

    /// A copy whose program keeps, of its functions, those `named`
    /// names; everything else it declares comes whole.
    pub(crate) fn selecting(&self, named: &std::collections::HashSet<InternedString>) -> Self {
        use zyntax_typed_ast::TypedDeclaration;
        let program = TypedProgram {
            declarations: self
                .program
                .declarations
                .iter()
                .filter(|decl| match &decl.node {
                    TypedDeclaration::Function(function) => named.contains(&function.name),
                    _ => true,
                })
                .cloned()
                .collect(),
            language: self.program.language,
            span: self.program.span,
            source_files: self.program.source_files.clone(),
            checker_interface: self.program.checker_interface.clone(),
            type_registry: self.program.type_registry.clone(),
        };
        Self {
            language: self.language.clone(),
            module_name: self.module_name.clone(),
            program,
            hir: self.hir.clone(),
        }
    }

    /// Encode an artifact for `include_bytes!` or another deployment bundle.
    pub fn encode(&self) -> Result<Vec<u8>, CompiledArtifactError> {
        Self::encode_program(&self.language, &self.module_name, &self.program)
    }

    pub(crate) fn encode_program(
        language: &str,
        module_name: &str,
        program: &TypedProgram,
    ) -> Result<Vec<u8>, CompiledArtifactError> {
        let payload = CompiledImportPayloadRef {
            language,
            module_name,
            program,
            type_registry: &program.type_registry,
        };
        // Postcard: a fixed field order and no field names, decoded in
        // a fraction of the time a self-describing format takes.
        let encoded = postcard::to_allocvec(&payload)
            .map_err(|e| CompiledArtifactError::Encode(e.to_string()))?;
        let mut bytes = Vec::with_capacity(HEADER_LEN + encoded.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&zyntax_compiler::BUILD_ID_BYTES);
        bytes.extend_from_slice(&encoded);
        Ok(bytes)
    }

    /// Decode a build-time artifact without invoking a source parser.
    pub fn decode(bytes: &[u8]) -> Result<Self, CompiledArtifactError> {
        if bytes.len() < VERSION_END || &bytes[..MAGIC.len()] != MAGIC {
            return Err(CompiledArtifactError::InvalidHeader);
        }
        let found = u32::from_le_bytes(
            bytes[MAGIC.len()..VERSION_END]
                .try_into()
                .expect("compiled artifact header length is fixed"),
        );
        if found != SCHEMA_VERSION {
            return Err(CompiledArtifactError::UnsupportedSchema {
                found,
                expected: SCHEMA_VERSION,
            });
        }
        let build_id = bytes
            .get(VERSION_END..HEADER_LEN)
            .ok_or(CompiledArtifactError::InvalidHeader)?;
        if build_id != zyntax_compiler::BUILD_ID_BYTES {
            return Err(CompiledArtifactError::IncompatibleCompiler {
                expected: zyntax_compiler::BUILD_ID,
                actual: build_id.iter().map(|byte| format!("{byte:02x}")).collect(),
            });
        }
        let payload: CompiledImportPayload = postcard::from_bytes(&bytes[HEADER_LEN..])
            .map_err(|e| CompiledArtifactError::Decode(e.to_string()))?;
        let mut program = payload.program;
        program.type_registry = payload.type_registry;
        program.type_registry.reserve_type_ids();
        Ok(Self::new(payload.language, payload.module_name, program))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_restores_module_scope() {
        let mut program = TypedProgram::default();
        let marker = InternedString::new_global("Marker");
        program.type_registry.register_atomic_type(
            marker,
            zyntax_typed_ast::TypeMetadata::default(),
            zyntax_typed_ast::Span::default(),
        );
        let artifact = CompiledImport::new("test", "prelude", program);
        let decoded = CompiledImport::decode(&artifact.encode().unwrap()).unwrap();
        assert_eq!(decoded.language(), "test");
        assert_eq!(decoded.module_name(), "prelude");
        assert_eq!(
            decoded
                .program()
                .type_registry
                .current_module()
                .and_then(|name| name.resolve_global())
                .as_deref(),
            Some("prelude")
        );
        assert!(
            decoded
                .program()
                .type_registry
                .get_type_by_name(marker)
                .is_some()
        );
    }

    #[test]
    fn rejects_unknown_schema() {
        let artifact = CompiledImport::new("test", "prelude", TypedProgram::default());
        let mut bytes = artifact.encode().unwrap();
        bytes[MAGIC.len()..VERSION_END].copy_from_slice(&(SCHEMA_VERSION + 1).to_le_bytes());
        assert!(matches!(
            CompiledImport::decode(&bytes),
            Err(CompiledArtifactError::UnsupportedSchema { .. })
        ));
    }

    #[test]
    fn borrowed_encoding_keeps_the_owned_payload_format() {
        let artifact = CompiledImport::new("demo", "library", TypedProgram::default());
        let payload = CompiledImportPayload {
            language: artifact.language.clone(),
            module_name: artifact.module_name.clone(),
            program: artifact.program.clone(),
            type_registry: artifact.program.type_registry.clone(),
        };
        assert_eq!(
            &artifact.encode().unwrap()[HEADER_LEN..],
            postcard::to_allocvec(&payload).unwrap(),
        );
    }

    #[test]
    fn compiler_identity_is_checked_before_typed_payloads() {
        let artifact = CompiledImport::new("test", "library", TypedProgram::default());
        let mut bytes = artifact.encode().unwrap();
        bytes[VERSION_END] ^= 1;
        bytes.truncate(HEADER_LEN);
        assert!(matches!(
            CompiledImport::decode(&bytes),
            Err(CompiledArtifactError::IncompatibleCompiler { .. })
        ));
        bytes.truncate(VERSION_END);
        bytes[MAGIC.len()..VERSION_END].copy_from_slice(&6u32.to_le_bytes());
        assert!(matches!(
            CompiledImport::decode(&bytes),
            Err(CompiledArtifactError::UnsupportedSchema { found: 6, .. })
        ));
    }
}
