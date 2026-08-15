//! Build-time artifacts for imports that must not be parsed on a cold path.

use serde::{Deserialize, Serialize};
use zyntax_typed_ast::{InternedString, TypedProgram};

const MAGIC: &[u8; 4] = b"ZAST";
const SCHEMA_VERSION: u32 = 2;
const HEADER_LEN: usize = MAGIC.len() + std::mem::size_of::<u32>();

/// A parsed import packaged by the application at build time.
///
/// The payload stores `TypedProgram` and its `TypeRegistry` separately because
/// ordinary TypedProgram serialization intentionally omits the registry.
#[derive(Debug, Clone)]
pub struct CompiledImport {
    language: String,
    module_name: String,
    program: TypedProgram,
}

#[derive(Serialize, Deserialize)]
struct CompiledImportPayload {
    language: String,
    module_name: String,
    program: TypedProgram,
    type_registry: zyntax_typed_ast::TypeRegistry,
}

#[derive(Debug, thiserror::Error)]
pub enum CompiledArtifactError {
    #[error("compiled import has an invalid header")]
    InvalidHeader,

    #[error("unsupported compiled import schema {found}; expected {expected}")]
    UnsupportedSchema { found: u32, expected: u32 },

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
        }
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

    pub(crate) fn into_program(self) -> TypedProgram {
        self.program
    }

    /// Encode an artifact for `include_bytes!` or another deployment bundle.
    pub fn encode(&self) -> Result<Vec<u8>, CompiledArtifactError> {
        let payload = CompiledImportPayload {
            language: self.language.clone(),
            module_name: self.module_name.clone(),
            program: self.program.clone(),
            type_registry: self.program.type_registry.clone(),
        };
        let mut encoded = Vec::new();
        ciborium::into_writer(&payload, &mut encoded)
            .map_err(|e| CompiledArtifactError::Encode(e.to_string()))?;
        let mut bytes = Vec::with_capacity(HEADER_LEN + encoded.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&encoded);
        Ok(bytes)
    }

    /// Decode a build-time artifact without invoking a source parser.
    pub fn decode(bytes: &[u8]) -> Result<Self, CompiledArtifactError> {
        if bytes.len() < HEADER_LEN || &bytes[..MAGIC.len()] != MAGIC {
            return Err(CompiledArtifactError::InvalidHeader);
        }
        let found = u32::from_le_bytes(
            bytes[MAGIC.len()..HEADER_LEN]
                .try_into()
                .expect("compiled artifact header length is fixed"),
        );
        if found != SCHEMA_VERSION {
            return Err(CompiledArtifactError::UnsupportedSchema {
                found,
                expected: SCHEMA_VERSION,
            });
        }
        let payload: CompiledImportPayload = ciborium::from_reader(&bytes[HEADER_LEN..])
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
        assert!(decoded
            .program()
            .type_registry
            .get_type_by_name(marker)
            .is_some());
    }

    #[test]
    fn rejects_unknown_schema() {
        let artifact = CompiledImport::new("test", "prelude", TypedProgram::default());
        let mut bytes = artifact.encode().unwrap();
        bytes[MAGIC.len()..HEADER_LEN].copy_from_slice(&(SCHEMA_VERSION + 1).to_le_bytes());
        assert!(matches!(
            CompiledImport::decode(&bytes),
            Err(CompiledArtifactError::UnsupportedSchema { .. })
        ));
    }
}
