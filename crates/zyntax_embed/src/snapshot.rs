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
const SCHEMA_VERSION: u32 = 3;
/// magic, schema, and the length of the directory that follows.
const HEADER_LEN: usize = MAGIC.len() + 2 * std::mem::size_of::<u32>();

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

    #[error("snapshot is truncated: {what} runs past the end")]
    Truncated { what: String },

    #[error(transparent)]
    Module(#[from] CompiledArtifactError),
}

/// Where something sits in the snapshot's bytes.
#[derive(Serialize, Deserialize, Clone, Copy)]
struct Extent {
    at: u32,
    len: u32,
}

impl Extent {
    fn of(bytes: &[u8], at: usize) -> Self {
        Extent {
            at: at as u32,
            len: bytes.len() as u32,
        }
    }

    fn slice<'a>(&self, blobs: &'a [u8], what: &str) -> Result<&'a [u8], SnapshotError> {
        let at = self.at as usize;
        let end = at + self.len as usize;
        blobs.get(at..end).ok_or_else(|| SnapshotError::Truncated {
            what: what.to_string(),
        })
    }
}

/// The directory: what the snapshot holds and where each part sits.
///
/// Small enough that reading it costs nothing, which is the point.
/// Everything large stays in the bytes behind it and is read in place.
#[derive(Serialize, Deserialize)]
struct Directory {
    language: String,
    grammar: Extent,
    modules: Vec<DirectoryEntry>,
}

#[derive(Serialize, Deserialize)]
struct DirectoryEntry {
    name: String,
    /// The largest type id this module mentions.
    ///
    /// Reserving ids is a high-water mark, so a host can reserve from
    /// this number without decoding the module it came from.
    max_type_id: u32,
    /// The encoded [`CompiledImport`].
    artifact: Extent,
    /// The module's source, when the language chose to carry it.
    source: Option<Extent>,
}

/// A language's grammar and standard library, ready to install.
///
/// Holds the artifact's bytes and reads what it needs out of them. The
/// directory is parsed on load; a grammar or a module is a slice of
/// what is already here, and a module is decoded only when something
/// imports it.
pub struct Snapshot {
    /// Everything after the directory, addressed by the extents in it.
    blobs: Vec<u8>,
    language: String,
    grammar: Extent,
    modules: Vec<DirectoryEntry>,
    /// Each module, decoded when something first asks for it. A module
    /// nobody imports is never decoded.
    decoded: Vec<std::sync::OnceLock<Result<CompiledImport, String>>>,
}

impl Snapshot {
    /// Read a snapshot, checking it is one and that this build
    /// understands it.
    ///
    /// Reads the directory and keeps the rest as it arrived. Parsing a
    /// container that owned every part copied the whole artifact before
    /// a line of it was wanted, which was most of what installing a
    /// language cost.
    pub fn load(bytes: &[u8]) -> Result<Self, SnapshotError> {
        if bytes.len() < HEADER_LEN || &bytes[..MAGIC.len()] != MAGIC {
            return Err(SnapshotError::InvalidHeader);
        }
        let word = |at: usize| {
            u32::from_le_bytes(
                bytes[at..at + 4]
                    .try_into()
                    .expect("snapshot header length is fixed"),
            )
        };
        let found = word(MAGIC.len());
        if found != SCHEMA_VERSION {
            return Err(SnapshotError::UnsupportedSchema {
                found,
                expected: SCHEMA_VERSION,
            });
        }
        let directory_len = word(MAGIC.len() + 4) as usize;
        let directory_end = HEADER_LEN + directory_len;
        let directory_bytes =
            bytes
                .get(HEADER_LEN..directory_end)
                .ok_or_else(|| SnapshotError::Truncated {
                    what: "the directory".to_string(),
                })?;
        let directory: Directory = ciborium::from_reader(directory_bytes)
            .map_err(|e| SnapshotError::Decode(e.to_string()))?;

        let blobs = bytes
            .get(directory_end..)
            .ok_or_else(|| SnapshotError::Truncated {
                what: "the body".to_string(),
            })?
            .to_vec();

        let decoded = directory
            .modules
            .iter()
            .map(|_| std::sync::OnceLock::new())
            .collect();
        Ok(Self {
            blobs,
            language: directory.language,
            grammar: directory.grammar,
            modules: directory.modules,
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
        self.grammar
            .slice(&self.blobs, "the grammar")
            .unwrap_or(&[])
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
                let entry = &self.modules[index];
                let bytes = entry
                    .artifact
                    .slice(&self.blobs, &entry.name)
                    .map_err(|e| e.to_string())?;
                CompiledImport::decode(bytes).map_err(|e| e.to_string())
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
        let entry = self.modules.iter().find(|m| m.name == name)?;
        let extent = entry.source.as_ref()?;
        let bytes = extent.slice(&self.blobs, &entry.name).ok()?;
        std::str::from_utf8(bytes).ok()
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
    modules: Vec<PendingModule>,
}

/// A module waiting to be written into a snapshot.
struct PendingModule {
    name: String,
    max_type_id: u32,
    artifact: Vec<u8>,
    source: Option<String>,
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
        self,
        name: impl Into<String>,
        program: zyntax_typed_ast::TypedProgram,
    ) -> Result<Self, SnapshotError> {
        self.push(name, program, None)
    }

    /// Add a parsed module, keeping its source for hosts that want to
    /// reparse rather than trust the artifact.
    pub fn module_with_source(
        self,
        name: impl Into<String>,
        program: zyntax_typed_ast::TypedProgram,
        source: impl Into<String>,
    ) -> Result<Self, SnapshotError> {
        self.push(name, program, Some(source.into()))
    }

    fn push(
        mut self,
        name: impl Into<String>,
        program: zyntax_typed_ast::TypedProgram,
        source: Option<String>,
    ) -> Result<Self, SnapshotError> {
        let name = name.into();
        let max_type_id = program.type_registry.max_type_id();
        let artifact = CompiledImport::new(self.language.clone(), name.clone(), program);
        self.modules.push(PendingModule {
            name,
            max_type_id,
            artifact: artifact.encode()?,
            source,
        });
        Ok(self)
    }

    /// Encode the snapshot.
    ///
    /// A directory of names and extents, then everything those extents
    /// point at. Reading the directory is the whole cost of loading;
    /// the parts are read where they lie.
    pub fn encode(self) -> Result<Vec<u8>, SnapshotError> {
        let grammar = self.grammar.ok_or_else(|| SnapshotError::MissingGrammar {
            language: self.language.clone(),
        })?;

        let mut blobs: Vec<u8> = Vec::new();
        let mut put = |bytes: &[u8], blobs: &mut Vec<u8>| {
            let extent = Extent::of(bytes, blobs.len());
            blobs.extend_from_slice(bytes);
            extent
        };

        let grammar_extent = put(&grammar, &mut blobs);
        let mut entries = Vec::with_capacity(self.modules.len());
        for module in &self.modules {
            let artifact = put(&module.artifact, &mut blobs);
            let source = module
                .source
                .as_ref()
                .map(|text| put(text.as_bytes(), &mut blobs));
            entries.push(DirectoryEntry {
                name: module.name.clone(),
                max_type_id: module.max_type_id,
                artifact,
                source,
            });
        }

        let directory = Directory {
            language: self.language,
            grammar: grammar_extent,
            modules: entries,
        };
        let mut encoded_directory = Vec::new();
        ciborium::into_writer(&directory, &mut encoded_directory)
            .map_err(|e| SnapshotError::Encode(e.to_string()))?;

        let mut bytes = Vec::with_capacity(HEADER_LEN + encoded_directory.len() + blobs.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(encoded_directory.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&encoded_directory);
        bytes.extend_from_slice(&blobs);
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
        // The schema sits between the magic and the directory length.
        let schema = MAGIC.len()..MAGIC.len() + 4;
        bytes[schema].copy_from_slice(&(SCHEMA_VERSION + 1).to_le_bytes());
        assert!(matches!(
            Snapshot::load(&bytes),
            Err(SnapshotError::UnsupportedSchema { .. })
        ));
    }

    #[test]
    fn a_snapshot_cut_short_is_refused() {
        // Extents address bytes that have to be there. Losing the tail
        // must say so rather than read whatever follows.
        let bytes = SnapshotBuilder::new("demo")
            .grammar(vec![7; 64])
            .module("prelude", empty_program())
            .expect("module")
            .encode()
            .expect("encode");
        let cut = &bytes[..bytes.len() - 32];
        let snapshot = Snapshot::load(cut).expect("the directory still reads");
        assert!(
            snapshot.module("prelude").is_err() || snapshot.grammar_bytes().is_empty(),
            "a part that runs past the end is refused rather than guessed at"
        );
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
