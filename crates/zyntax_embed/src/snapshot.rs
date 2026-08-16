//! Everything a language brings to a runtime, in one artifact.
//!
//! A language ships a grammar and a standard library, and both have to
//! reach a runtime before it can compile a line of source. Wiring that
//! by hand means a build script per module, an include per artifact, a
//! resolver that answers for each one, and a rule about the order type
//! ids are reserved in that is easy to state and easy to forget.
//!
//! A snapshot carries all of it. A build script writes a snapshot, and a host
//! installs it, and the order things happen in belongs to the
//! runtime rather than to every language that targets it.

use crate::compiled_artifact::{CompiledArtifactError, CompiledImport};
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 5] = b"ZSNAP";
const SCHEMA_VERSION: u32 = 2;
const HEADER_LEN: usize = MAGIC.len() + std::mem::size_of::<u32>();

/// The extension a snapshot is written under.
///
/// The format owns this, not the caller. A build that writes one name
/// and a load that reads another is a failure nobody should be able to
/// spell.
pub const SNAPSHOT_EXTENSION: &str = "zsnap";

/// The file name a language's snapshot takes.
pub fn snapshot_file_name(language: &str) -> String {
    format!("{language}.{SNAPSHOT_EXTENSION}")
}

#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("snapshot has an invalid header")]
    InvalidHeader,

    #[error("unsupported snapshot schema {found}; expected {expected}")]
    UnsupportedSchema { found: u32, expected: u32 },

    #[error("failed to encode snapshot: {0}")]
    Encode(String),

    #[error("failed to decode snapshot: {0}")]
    Decode(String),

    #[error("snapshot for '{language}' has no grammar")]
    MissingGrammar { language: String },

    #[error(transparent)]
    Module(#[from] CompiledArtifactError),
}

#[derive(Serialize, Deserialize)]
struct SnapshotPayload {
    language: String,
    /// The grammar, in the form [`crate::LanguageGrammar`] loads.
    grammar: Vec<u8>,
    /// Each module as its own artifact, in the order they were added.
    /// Order is kept because installing reserves type ids by walking
    /// it, and a module's ids have to land where they did at build
    /// time.
    modules: Vec<SnapshotModule>,
}

#[derive(Serialize, Deserialize)]
struct SnapshotModule {
    name: String,
    /// The largest type id this module mentions.
    ///
    /// Reserving ids is a high-water mark, so a host can reserve from
    /// this number without decoding the module it came from. That is
    /// what lets a module stay encoded until something imports it.
    #[serde(default)]
    max_type_id: u32,
    /// The encoded [`CompiledImport`]. Held encoded so installing can
    /// decode what it needs and leave the rest, and so a module's own
    /// schema check still runs.
    artifact: Vec<u8>,
    /// The module's source, when the language chose to carry it.
    /// Development hosts fall back to parsing this.
    source: Option<String>,
}

/// A language's grammar and standard library, ready to install.
///
/// Decoding a module reserves the type ids it was built against, so it
/// happens once and is kept. A host holds one of these for the process
/// and installs it into whatever runtimes it builds.
pub struct Snapshot {
    language: String,
    grammar: Vec<u8>,
    modules: Vec<SnapshotModule>,
    /// Each module, decoded when something first asks for it. A module
    /// nobody imports is never decoded, and decoding is the cost of
    /// installing a standard library.
    decoded: Vec<std::sync::OnceLock<Result<CompiledImport, String>>>,
}

impl Snapshot {
    /// Read a snapshot, checking it is one and that this build
    /// understands it.
    pub fn load(bytes: &[u8]) -> Result<Self, SnapshotError> {
        if bytes.len() < HEADER_LEN || &bytes[..MAGIC.len()] != MAGIC {
            return Err(SnapshotError::InvalidHeader);
        }
        let found = u32::from_le_bytes(
            bytes[MAGIC.len()..HEADER_LEN]
                .try_into()
                .expect("snapshot header length is fixed"),
        );
        if found != SCHEMA_VERSION {
            return Err(SnapshotError::UnsupportedSchema {
                found,
                expected: SCHEMA_VERSION,
            });
        }
        let payload: SnapshotPayload = ciborium::from_reader(&bytes[HEADER_LEN..])
            .map_err(|e| SnapshotError::Decode(e.to_string()))?;
        let decoded = payload
            .modules
            .iter()
            .map(|_| std::sync::OnceLock::new())
            .collect();
        Ok(Self {
            language: payload.language,
            grammar: payload.grammar,
            modules: payload.modules,
            decoded,
        })
    }

    /// The language this snapshot installs.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// The grammar, as [`crate::LanguageGrammar::from_compiled_bytes`]
    /// reads it.
    pub fn grammar_bytes(&self) -> &[u8] {
        &self.grammar
    }

    /// The modules it carries, in the order they were built.
    pub fn module_names(&self) -> impl Iterator<Item = &str> {
        self.modules.iter().map(|m| m.name.as_str())
    }

    /// One module, decoded on first use and kept after.
    pub fn module(&self, name: &str) -> Result<Option<CompiledImport>, SnapshotError> {
        let Some(index) = self.modules.iter().position(|m| m.name == name) else {
            return Ok(None);
        };
        self.decoded[index]
            .get_or_init(|| {
                CompiledImport::decode(&self.modules[index].artifact).map_err(|e| e.to_string())
            })
            .as_ref()
            .map(|module| Some(module.clone()))
            .map_err(|e| SnapshotError::Decode(e.clone()))
    }

    /// Reserve the type ids every module was built against.
    ///
    /// Decoding a module reserves its ids as a side effect, which meant
    /// installing a standard library decoded all of it before anything
    /// could parse. Reserving is a high-water mark and the build wrote
    /// each module's down, so this reserves from those numbers and
    /// leaves the modules encoded until one is imported.
    pub fn reserve_type_ids(&self) {
        for module in &self.modules {
            zyntax_typed_ast::TypeId::reserve_at_least(module.max_type_id.saturating_add(1));
        }
    }

    /// A module's source, when the snapshot carries it.
    pub fn module_source(&self, name: &str) -> Option<&str> {
        self.modules
            .iter()
            .find(|m| m.name == name)
            .and_then(|m| m.source.as_deref())
    }

    /// Every module, decoded. Decodes whatever has not been asked for
    /// yet, so a host wanting all of them can still say so.
    pub fn modules(&self) -> Result<Vec<CompiledImport>, SnapshotError> {
        self.modules
            .iter()
            .map(|m| {
                self.module(&m.name)?
                    .ok_or_else(|| SnapshotError::Decode(format!("module '{}' vanished", m.name)))
            })
            .collect()
    }
}

/// Builds a snapshot, for a language's build script.
///
/// ```ignore
/// SnapshotBuilder::new("zynml")
///     .grammar(grammar.to_compiled_bytes()?)
///     .module("prelude", prelude_program)
///     .build_in(&out)?;
/// ```
pub struct SnapshotBuilder {
    language: String,
    grammar: Option<Vec<u8>>,
    modules: Vec<SnapshotModule>,
}

impl SnapshotBuilder {
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            grammar: None,
            modules: Vec::new(),
        }
    }

    /// The compiled grammar, from
    /// [`crate::LanguageGrammar::to_compiled_bytes`].
    pub fn grammar(mut self, bytes: Vec<u8>) -> Self {
        self.grammar = Some(bytes);
        self
    }

    /// Add a parsed module.
    ///
    /// Modules install in the order they are added, which is the order
    /// their type ids were reserved in.
    pub fn module(
        mut self,
        name: impl Into<String>,
        program: zyntax_typed_ast::TypedProgram,
    ) -> Result<Self, SnapshotError> {
        let name = name.into();
        let max_type_id = program.type_registry.max_type_id();
        let artifact = CompiledImport::new(self.language.clone(), name.clone(), program);
        self.modules.push(SnapshotModule {
            name,
            max_type_id,
            artifact: artifact.encode()?,
            source: None,
        });
        Ok(self)
    }

    /// Add a parsed module, keeping its source for hosts that want to
    /// reparse rather than trust the artifact.
    pub fn module_with_source(
        mut self,
        name: impl Into<String>,
        program: zyntax_typed_ast::TypedProgram,
        source: impl Into<String>,
    ) -> Result<Self, SnapshotError> {
        let name = name.into();
        let max_type_id = program.type_registry.max_type_id();
        let artifact = CompiledImport::new(self.language.clone(), name.clone(), program);
        self.modules.push(SnapshotModule {
            name,
            max_type_id,
            artifact: artifact.encode()?,
            source: Some(source.into()),
        });
        Ok(self)
    }

    /// Encode the snapshot.
    pub fn encode(self) -> Result<Vec<u8>, SnapshotError> {
        let grammar = self.grammar.ok_or_else(|| SnapshotError::MissingGrammar {
            language: self.language.clone(),
        })?;
        let payload = SnapshotPayload {
            language: self.language,
            grammar,
            modules: self.modules,
        };
        let mut encoded = Vec::new();
        ciborium::into_writer(&payload, &mut encoded)
            .map_err(|e| SnapshotError::Encode(e.to_string()))?;
        let mut bytes = Vec::with_capacity(HEADER_LEN + encoded.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&encoded);
        Ok(bytes)
    }

    /// Write the snapshot into a directory, usually `OUT_DIR`, under
    /// the name the format chooses. Returns where it went.
    pub fn build_in(self, dir: &std::path::Path) -> Result<std::path::PathBuf, SnapshotError> {
        let path = dir.join(snapshot_file_name(&self.language));
        let bytes = self.encode()?;
        std::fs::write(&path, bytes).map_err(|e| SnapshotError::Encode(e.to_string()))?;
        Ok(path)
    }
}

/// Include the snapshot a build script wrote for this language.
///
/// Resolves the same name [`SnapshotBuilder::build_in`] wrote, so the
/// two cannot disagree.
#[macro_export]
macro_rules! include_snapshot {
    ($language:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $language, ".", "zsnap")) as &[u8]
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::TypedProgram;

    fn empty_program() -> TypedProgram {
        TypedProgram::default()
    }

    #[test]
    fn a_snapshot_round_trips() {
        let bytes = SnapshotBuilder::new("demo")
            .grammar(vec![1, 2, 3])
            .module("prelude", empty_program())
            .expect("module")
            .encode()
            .expect("encode");

        let snapshot = Snapshot::load(&bytes).expect("load");
        assert_eq!(snapshot.language(), "demo");
        assert_eq!(snapshot.grammar_bytes(), &[1, 2, 3]);
        assert_eq!(snapshot.module_names().collect::<Vec<_>>(), vec!["prelude"]);
    }

    #[test]
    fn modules_keep_the_order_they_were_added() {
        // Installing reserves type ids by walking this, so the order
        // is part of what the artifact means.
        let bytes = SnapshotBuilder::new("demo")
            .grammar(vec![0])
            .module("prelude", empty_program())
            .expect("module")
            .module("tensor", empty_program())
            .expect("module")
            .module("simd", empty_program())
            .expect("module")
            .encode()
            .expect("encode");

        let snapshot = Snapshot::load(&bytes).expect("load");
        assert_eq!(
            snapshot.module_names().collect::<Vec<_>>(),
            vec!["prelude", "tensor", "simd"]
        );
    }

    #[test]
    fn the_format_names_the_file() {
        assert_eq!(snapshot_file_name("zynml"), "zynml.zsnap");
    }

    #[test]
    fn a_snapshot_without_a_grammar_is_refused() {
        let built = SnapshotBuilder::new("demo").encode();
        assert!(
            matches!(built, Err(SnapshotError::MissingGrammar { .. })),
            "a language with no grammar cannot parse anything"
        );
    }

    #[test]
    fn something_that_is_not_a_snapshot_is_refused() {
        assert!(matches!(
            Snapshot::load(b"not a snapshot"),
            Err(SnapshotError::InvalidHeader)
        ));
    }

    #[test]
    fn a_snapshot_from_another_schema_is_refused() {
        let mut bytes = SnapshotBuilder::new("demo")
            .grammar(vec![0])
            .encode()
            .expect("encode");
        bytes[MAGIC.len()..HEADER_LEN].copy_from_slice(&(SCHEMA_VERSION + 1).to_le_bytes());
        assert!(matches!(
            Snapshot::load(&bytes),
            Err(SnapshotError::UnsupportedSchema { .. })
        ));
    }

    #[test]
    fn a_module_carries_its_source_when_asked() {
        let bytes = SnapshotBuilder::new("demo")
            .grammar(vec![0])
            .module_with_source("prelude", empty_program(), "fn main() {}")
            .expect("module")
            .encode()
            .expect("encode");
        let snapshot = Snapshot::load(&bytes).expect("load");
        assert_eq!(snapshot.module_source("prelude"), Some("fn main() {}"));
        assert_eq!(snapshot.module_source("missing"), None);
    }
}
